// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Checks src/vision against the Python reference (tests/ref/vision_ref.py).
//
//   python tests/ref/vision_ref.py DIR      # PIL 12.3 + jevfire.metal.vision
//   vision_test DIR                         # compare everything
//   vision_test --bench FILE.jpg [ITERS]    # decode + resize + preprocess timings
//
// Must be exact: resize (byte-identical to PIL), preprocess (float32 rows,
// grid, digest), smart_resize, mrope/expand/rows, data URLs, and decoding of
// every file the manifest marks exact (PNG, GIF, JPEG, BMP, TIFF, lossless
// WebP...). Other decodes (alpha WebP, AVIF, HEIC) are reported, not failed.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../src/util/json.h"
#include "../src/vision/vision.h"

using namespace espetero;
using namespace espetero::vision;

namespace {

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "cannot read %s\n", path.c_str());
    exit(2);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

struct Blob {
  std::string data;
  std::string_view get(json::Value& ref) const {
    return std::string_view(data).substr(size_t(ref[0].as_int()), size_t(ref[1].as_int()));
  }
};

int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    failures++;
    fprintf(stderr, "FAIL: %s\n", what);
  }
}

RGBImage make_rgb(int w, int h, std::string_view bytes) {
  RGBImage im;
  im.width = w;
  im.height = h;
  im.rgb.assign(bytes.begin(), bytes.end());
  return im;
}

std::vector<int32_t> ints(json::Value& v) {
  std::vector<int32_t> out;
  for (auto& x : v.as_array()) out.push_back(int32_t(x.as_int()));
  return out;
}

std::vector<std::array<int, 3>> grids_of(json::Value& v) {
  std::vector<std::array<int, 3>> out;
  for (auto& g : v.as_array()) out.push_back({int(g[0].as_int()), int(g[1].as_int()), int(g[2].as_int())});
  return out;
}

void test_resize(json::Value& cases, const Blob& blob) {
  int ok = 0, n = 0;
  for (auto& c : cases.as_array()) {
    const int w = int(c["in"][0].as_int()), h = int(c["in"][1].as_int());
    const int ow = int(c["out"][0].as_int()), oh = int(c["out"][1].as_int());
    const RGBImage in = make_rgb(w, h, blob.get(c["src"]));
    const RGBImage out = resize_bicubic(in, ow, oh);
    const std::string_view exp = blob.get(c["exp"]);
    n++;
    if (out.width == ow && out.height == oh && out.rgb.size() == exp.size() &&
        std::memcmp(out.rgb.data(), exp.data(), exp.size()) == 0) {
      ok++;
      continue;
    }
    size_t i = 0, nd = 0;
    int maxd = 0;
    for (size_t k = 0; k < std::min(exp.size(), out.rgb.size()); k++) {
      const int d = std::abs(int(out.rgb[k]) - int(uint8_t(exp[k])));
      if (d && !nd++) i = k;
      maxd = std::max(maxd, d);
    }
    fprintf(stderr, "  resize %dx%d -> %dx%d: %zu bytes differ (max %d), first at pixel %zu\n", w, h, ow, oh, nd,
            maxd, i / 3);
  }
  printf("resize:       %d/%d byte-identical to PIL\n", ok, n);
  check(ok == n, "resize");
}

Spec spec_of(json::Value& s) {
  Spec sp;
  sp.patch = int(s["patch"].as_int());
  sp.temporal = int(s["temporal"].as_int());
  sp.merge = int(s["merge"].as_int());
  for (int i = 0; i < 3; i++) {
    sp.mean[i] = float(s["mean"][i].as_double());
    sp.std[i] = float(s["std"][i].as_double());
  }
  sp.min_pixels = s["min_pixels"].as_int();
  sp.max_pixels = s["max_pixels"].as_int();
  return sp;
}

void test_preprocess(json::Value& cases, const Blob& blob) {
  int ok = 0, n = 0;
  for (auto& c : cases.as_array()) {
    const RGBImage in = make_rgb(int(c["size"][0].as_int()), int(c["size"][1].as_int()), blob.get(c["rgb"]));
    const Spec sp = spec_of(c["spec"]);
    const Prepared p = preprocess(in, sp);
    const std::string_view exp = blob.get(c["pixels"]);
    const bool grid_ok = p.grid_t == c["grid"][0].as_int() && p.grid_h == c["grid"][1].as_int() &&
                         p.grid_w == c["grid"][2].as_int();
    const bool px_ok = p.pixels.size() * 4 == exp.size() && std::memcmp(p.pixels.data(), exp.data(), exp.size()) == 0;
    const bool dg_ok = p.digest_hex() == c["digest"].as_string();
    n++;
    if (grid_ok && px_ok && dg_ok) {
      ok++;
    } else {
      fprintf(stderr, "  preprocess %dx%d: grid %s pixels %s digest %s\n", in.width, in.height, grid_ok ? "ok" : "BAD",
              px_ok ? "ok" : "BAD", dg_ok ? "ok" : "BAD");
    }
  }
  printf("preprocess:   %d/%d identical (float32 rows, grid, digest)\n", ok, n);
  check(ok == n, "preprocess");
}

void test_digest(json::Value& cases, const Blob& blob) {
  int ok = 0, n = 0;
  for (auto& c : cases.as_array()) {
    const std::string_view px = blob.get(c["pixels"]);
    std::vector<float> f(px.size() / 4);
    if (!f.empty()) std::memcpy(f.data(), px.data(), px.size());
    auto& g = c["grid"];
    const auto d = digest(f.data(), f.size(), {int(g[0].as_int()), int(g[1].as_int()), int(g[2].as_int())});
    Prepared tmp;
    tmp.digest = d;
    n++;
    ok += tmp.digest_hex() == c["digest"].as_string();
  }
  printf("digest:       %d/%d match hashlib.blake2b tree\n", ok, n);
  check(ok == n, "digest");
}

struct DiffStats {
  int files = 0, exact = 0, failed = 0;
  int maxd = 0;
  double sum = 0;
  size_t count = 0, nonzero = 0;
};

void test_decode(json::Value& cases, const Blob& blob, bool verbose) {
  std::map<std::string, DiffStats> by_format;
  int must_ok = 0, must_n = 0;
  for (auto& c : cases.as_array()) {
    const std::string tag = c["tag"].as_string(), fmt = c["format"].as_string();
    const bool must = c["exact"].as_bool();
    RGBImage img;
    std::string err;
    const bool ok = decode_image(blob.get(c["file"]), img, err);
    DiffStats& st = by_format[fmt];
    st.files++;
    if (c["error"].is_string()) {  // PIL cannot read it
      printf("  %-34s PIL: %s | ours: %s\n", tag.c_str(), c["error"].as_string().c_str(),
             ok ? ("decoded " + std::to_string(img.width) + "x" + std::to_string(img.height)).c_str() : err.c_str());
      continue;
    }
    must_n += must;
    const int w = int(c["size"][0].as_int()), h = int(c["size"][1].as_int());
    if (!ok || img.width != w || img.height != h) {
      st.failed++;
      printf("  %-34s FAILED: %s (got %dx%d, want %dx%d)\n", tag.c_str(), ok ? "size mismatch" : err.c_str(),
             img.width, img.height, w, h);
      continue;
    }
    const std::string_view exp = blob.get(c["exp"]);
    int maxd = 0;
    size_t nz = 0;
    double sum = 0;
    for (size_t k = 0; k < exp.size(); k++) {
      const int d = std::abs(int(img.rgb[k]) - int(uint8_t(exp[k])));
      maxd = std::max(maxd, d);
      nz += d != 0;
      sum += d;
    }
    st.maxd = std::max(st.maxd, maxd);
    st.sum += sum;
    st.count += exp.size();
    st.nonzero += nz;
    if (maxd == 0) {
      st.exact++;
      must_ok += must;
    }
    if (maxd != 0 || verbose)
      printf("  %-34s %-5s %4dx%-4d %s max=%d mean=%.4f (%.2f%% of samples differ)%s\n", tag.c_str(),
             c["mode"].as_string().c_str(), w, h, maxd ? "DIFF " : "exact", maxd, sum / double(exp.size()),
             100.0 * double(nz) / double(exp.size()), must && maxd ? "  <-- must be exact" : "");
  }
  for (const auto& [fmt, st] : by_format)
    printf("decode %-6s %2d/%-2d identical to PIL%s, max abs diff %d, mean %.4f\n", fmt.c_str(), st.exact,
           st.files, st.failed ? " (some failed)" : "", st.maxd, st.count ? st.sum / double(st.count) : 0.0);
  printf("decode:       %d/%d files that must be exact are byte-identical\n", must_ok, must_n);
  check(must_ok == must_n, "decode");
}

void test_smart_resize(json::Value& cases) {
  int ok = 0, n = 0;
  for (auto& c : cases.as_array()) {
    const int h = int(c[0].as_int()), w = int(c[1].as_int()), f = int(c[2].as_int());
    const int64_t mn = c[3].as_int(), mx = c[4].as_int();
    bool good;
    try {
      const auto [rh, rw] = smart_resize(h, w, f, mn, mx);
      good = c[5].is_array() && rh == c[5][0].as_int() && rw == c[5][1].as_int();
    } catch (const std::invalid_argument&) {
      good = c[5].is_null();
    }
    n++;
    ok += good;
    if (!good && n - ok < 5) fprintf(stderr, "  smart_resize(%d, %d, %d, %lld, %lld) mismatch\n", h, w, f, (long long)mn, (long long)mx);
  }
  printf("smart_resize: %d/%d\n", ok, n);
  check(ok == n, "smart_resize");
}

void test_tokens(json::Value& m) {
  Spec sp;
  int ok = 0, n = 0;
  for (auto& c : m["mrope"].as_array()) {
    n++;
    try {
      int next = -1;
      const auto pos = mrope_positions(ints(c["ids"]), grids_of(c["grids"]), sp, int(c["start"].as_int()), &next);
      if (!c["error"].is_null()) continue;
      const auto exp = ints(c["pos"]);
      bool good = next == c["next"].as_int() && exp.size() == pos.size() * 3;
      for (size_t i = 0; good && i < pos.size(); i++)
        good = pos[i][0] == exp[3 * i] && pos[i][1] == exp[3 * i + 1] && pos[i][2] == exp[3 * i + 2];
      ok += good;
    } catch (const std::invalid_argument& e) {
      ok += c["error"].is_string() && c["error"].as_string() == e.what();
    }
  }
  printf("mrope:        %d/%d (positions, next position, errors)\n", ok, n);
  check(ok == n, "mrope");

  ok = n = 0;
  for (auto& c : m["expand"].as_array()) {
    n++;
    try {
      const auto ids = expand_image_tokens(ints(c["prefix"]), grids_of(c["grids"]), sp);
      ok += c["ids"].is_array() && ids == ints(c["ids"]);
    } catch (const std::invalid_argument&) {
      ok += c["ids"].is_null();
    }
  }
  printf("expand:       %d/%d\n", ok, n);
  check(ok == n, "expand");

  ok = n = 0;
  for (auto& c : m["rows"].as_array()) {
    n++;
    try {
      const auto rows = image_rows(ints(c["ids"]), grids_of(c["grids"]), sp);
      if (c["rows"].is_array()) {
        std::vector<int> exp;
        for (int32_t v : ints(c["rows"])) exp.push_back(v);
        ok += rows == exp;
      }
    } catch (const std::invalid_argument&) {
      ok += c["rows"].is_null();
    }
  }
  printf("image_rows:   %d/%d\n", ok, n);
  check(ok == n, "image_rows");
}

void test_dataurl(json::Value& cases, const Blob& blob) {
  int ok = 0, n = 0;
  for (auto& c : cases.as_array()) {
    std::string bytes, err;
    const bool good = decode_data_url(c["url"].as_string(), bytes, err);
    n++;
    if (c["bytes"].is_null()) {
      ok += !good;
      if (good) fprintf(stderr, "  data URL should fail (%s): %.60s\n", c["error"].as_string().c_str(), c["url"].as_string().c_str());
    } else {
      ok += good && bytes == blob.get(c["bytes"]);
      if (!(good && bytes == blob.get(c["bytes"]))) fprintf(stderr, "  data URL mismatch (%s): %.60s\n", err.c_str(), c["url"].as_string().c_str());
    }
  }
  printf("data URLs:    %d/%d\n", ok, n);
  check(ok == n, "data URLs");
}

void test_pipeline(json::Value& cases, const Blob& blob) {
  int ok = 0, n = 0;
  for (auto& c : cases.as_array()) {
    RGBImage img;
    std::string err;
    n++;
    if (!load_rgb(blob.get(c["src"]), img, err)) {
      fprintf(stderr, "  pipeline %s: load failed: %s\n", c["tag"].as_string().c_str(), err.c_str());
      continue;
    }
    const Prepared p = preprocess(img, spec_of(c["spec"]));
    const std::string_view exp = blob.get(c["pixels"]);
    const bool good = p.grid_h == c["grid"][1].as_int() && p.grid_w == c["grid"][2].as_int() &&
                      p.pixels.size() * 4 == exp.size() && std::memcmp(p.pixels.data(), exp.data(), exp.size()) == 0 &&
                      p.digest_hex() == c["digest"].as_string();
    ok += good;
    if (!good) fprintf(stderr, "  pipeline %s: MISMATCH\n", c["tag"].as_string().c_str());
  }
  printf("end to end:   %d/%d bytes/data URL -> float rows identical to V.preprocess\n", ok, n);
  check(ok == n, "end to end");
}

void test_spec_dirs(json::Value& cases) {
  int ok = 0, n = 0;
  for (auto& c : cases.as_array()) {
    n++;
    const Spec s = spec_from_model_dir(c["dir"].as_string(), c["max_pixels"].as_int());
    bool good = s.patch == c["patch"].as_int() && s.temporal == c["temporal"].as_int() && s.merge == c["merge"].as_int() &&
                s.min_pixels == c["min_pixels"].as_int() && s.max_pixels == c["max_pixels"].as_int() &&
                s.image_token_id == c["image_token_id"].as_int() && s.vision_start_id == c["vision_start_id"].as_int() &&
                s.vision_end_id == c["vision_end_id"].as_int();
    for (int i = 0; i < 3; i++)
      good &= s.mean[i] == float(c["mean"][i].as_double()) && s.std[i] == float(c["std"][i].as_double());
    ok += good;
    if (!good) fprintf(stderr, "  spec mismatch for %s\n", c["dir"].as_string().c_str());
  }
  printf("model specs:  %d/%d checkpoints match VisionSpec.from_model\n", ok, n);
  check(ok == n, "spec_from_model_dir");
}

void test_spec_config() {
  const json::Value cfg = json::parse(
      R"({"image_token_id": 151655, "vision_start_token_id": 151652, "vision_end_token_id": 151653,
          "vision_config": {"patch_size": 14, "temporal_patch_size": 2, "spatial_merge_size": 2}})");
  const json::Value pre = json::parse(
      R"({"image_processor": {"image_mean": [0.48145466, 0.4578275, 0.40821073],
                              "image_std": [0.26862954, 0.26130258, 0.27577711]}})");
  const Spec s = spec_from_config(cfg, &pre, 50000);
  const bool good = s.patch == 14 && s.temporal == 2 && s.merge == 2 && s.image_token_id == 151655 &&
                    s.vision_start_id == 151652 && s.vision_end_id == 151653 && s.mean[0] == 0.48145466f &&
                    s.std[2] == 0.27577711f && s.max_pixels == 50000 && s.min_pixels == 50000;
  printf("spec config:  %s\n", good ? "ok" : "BAD");
  check(good, "spec_from_config");
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int bench(const char* path, int iters) {
  const std::string bytes = read_file(path);
  for (int64_t budget : {131072, 262144}) {
    Spec sp;
    sp.set_budget(budget);
    double best_dec = 1e9, best_rs = 1e9, best_pp = 1e9, best_all = 1e9;
    Prepared p;
    for (int it = 0; it < iters; it++) {
      RGBImage img;
      std::string err;
      auto t0 = std::chrono::steady_clock::now();
      if (!decode_image(bytes, img, err)) {
        fprintf(stderr, "decode failed: %s\n", err.c_str());
        return 1;
      }
      const double dec = ms_since(t0);
      auto t1 = std::chrono::steady_clock::now();
      const auto [h, w] = smart_resize(img.height, img.width, sp.factor(), sp.min_pixels, sp.max_pixels);
      const RGBImage r = resize_bicubic(img, w, h);
      const double rs = ms_since(t1);
      auto t2 = std::chrono::steady_clock::now();
      p = patchify(r, sp);
      const double pp = ms_since(t2);
      best_dec = std::min(best_dec, dec);
      best_rs = std::min(best_rs, rs);
      best_pp = std::min(best_pp, pp);
      best_all = std::min(best_all, dec + rs + pp);
    }
    printf("budget %6lld: grid %dx%d  decode %.2f ms  resize %.2f ms  patchify+digest %.2f ms  total %.2f ms (best of %d)\n",
           (long long)budget, p.grid_h, p.grid_w, best_dec, best_rs, best_pp, best_all, iters);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 3 && std::string(argv[1]) == "--bench") return bench(argv[2], argc > 3 ? atoi(argv[3]) : 20);
  if (argc < 2) {
    fprintf(stderr, "usage: %s REF_DIR [-v] | --bench FILE [ITERS]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  const bool verbose = argc > 2 && std::string(argv[2]) == "-v";
  json::Value m = json::parse(read_file(dir + "/manifest.json"));
  Blob blob{read_file(dir + "/data.bin")};
  test_resize(m["resize"], blob);
  test_preprocess(m["preprocess"], blob);
  test_digest(m["digest"], blob);
  test_smart_resize(m["smart_resize"]);
  test_tokens(m);
  test_dataurl(m["dataurl"], blob);
  test_spec_config();
  test_spec_dirs(m["spec"]);
  test_pipeline(m["pipeline"], blob);
  test_decode(m["decode"], blob, verbose);
  printf(failures ? "FAILED (%d)\n" : "all exact checks passed\n", failures);
  return failures ? 1 : 0;
}
