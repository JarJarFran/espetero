// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Vision fidelity dumps: for each image file, Espetero's processor output
// (the patch rows) and the vision tower's merged token embeddings, as raw
// little-endian float32 files beside a JSON index, for
// tools/vision_fidelity.py (which runs mlx-vlm on the same files).
//
//   build/vision_dump <MLX model dir | .gguf> <out dir> <max_pixels> <image> [<image> ...]
//
// (A .gguf model takes its mmproj next to it; its processor geometry comes
// from the loaded model: Gemma 4 only.)
//
// Writes <out>/<i>.pixels.f32 ([rows, 3*temporal*patch*patch]),
// <out>/<i>.emb.f32 ([tokens, hidden]) and <out>/index.json. An LFM2-VL
// directory goes through its own processor (max_pixels unused): every view
// (tiles, then the thumbnail) back to back, as tests/ref/lfm2_ref.py vision
// writes them.
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../src/model/api.h"
#include "../src/util/json.h"
#include "../src/vision/vision.h"

using namespace espetero;

static bool write_f32(const std::string& path, const std::vector<float>& v) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const size_t n = std::fwrite(v.data(), sizeof(float), v.size(), f);
  std::fclose(f);
  return n == v.size();
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <MLX model dir> <out dir> <max_pixels> <image> [<image> ...]\n", argv[0]);
    return 2;
  }
  const std::string model = argv[1], out = argv[2];
  if (std::getenv("ESPETERO_NO_EXIF")) vision::set_exif_orientation(false);  // the reference server's decode
  const long long max_pixels = std::atoll(argv[3]);
  ::mkdir(out.c_str(), 0755);
  const bool gguf = model.size() > 5 && model.compare(model.size() - 5, 5, ".gguf") == 0;
  bool lfm2 = false;
  {
    std::FILE* cf = std::fopen((model + "/config.json").c_str(), "rb");
    std::string txt;
    if (cf) {
      char buf[4096];
      size_t n;
      while ((n = std::fread(buf, 1, sizeof(buf), cf)) > 0) txt.append(buf, n);
      std::fclose(cf);
    }
    lfm2 = !txt.empty() && json::parse(txt).get_string("model_type", "") == "lfm2_vl";
  }
  vision::Spec spec;
  vision::Lfm2Spec lspec;
  try {
    if (lfm2) {
      lspec = vision::lfm2_spec_from_model_dir(model);
      lspec.image_id = 1, lspec.start_id = 2, lspec.end_id = 3, lspec.thumb_id = 4;  // the run's ids are not dumped
      for (size_t k = 0; k < lspec.grid_ids.size(); ++k) lspec.grid_ids[k] = int32_t(100 + k);
    } else if (!gguf) {
      spec = vision::spec_from_model_dir(model, max_pixels);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "spec: %s\n", e.what());
    return 1;
  }
  LoadOptions o;
  o.slots = 2;
  o.max_ctx = 4096;
  o.kv_gb = 1;
  auto m = NativeModel::load(model, o);
  const ModelInfo& I = m->info();
  if (!I.vision) {
    std::fprintf(stderr, "the model has no vision tower\n");
    return 1;
  }
  if (gguf) {
    if (I.family != "gemma4") {
      std::fprintf(stderr, "GGUF: Gemma 4 only (the processor geometry of other mmproj files is not read here)\n");
      return 1;
    }
    spec.set_gemma4(max_pixels, I.vision_patch, I.vision_merge);
  }
  json::Value index = json::Value::array();
  std::vector<vision::Prepared> all;   // every image, for the batched check below
  std::vector<std::vector<float>> embs;
  for (int i = 4; i < argc; ++i) {
    const std::string path = argv[i];
    vision::RGBImage img;
    std::string err;
    if (!vision::load_rgb_file(path, img, err)) {
      std::fprintf(stderr, "%s: %s\n", path.c_str(), err.c_str());
      return 1;
    }
    if (lfm2) {
      const vision::Lfm2Prepared p = vision::lfm2_preprocess(img, lspec);
      std::vector<ImageRef> refs;
      int tokens = 0;
      for (const auto& v : p.views) {
        ImageRef r;
        r.pixels = p.pixels.data() + v.offset;
        r.grid_h = v.gh;
        r.grid_w = v.gw;
        refs.push_back(r);
        tokens += v.gh * v.gw / 4;
      }
      std::vector<float> emb(size_t(tokens) * size_t(I.hidden));
      PassStats st;
      m->encode_images(refs, emb.data(), &st);
      const std::string stem = out + "/" + std::to_string(i - 4);
      if (!write_f32(stem + ".pixels.f32", p.pixels) || !write_f32(stem + ".emb.f32", emb)) return 1;
      json::Value e = json::Value::object();
      e.set("file", path);
      e.set("views", int64_t(p.views.size()));
      e.set("tokens", int64_t(tokens));
      e.set("run", int64_t(p.run.size()));
      e.set("gpu_ms", st.gpu_ms);
      index.push_back(e);
      std::printf("%s: %zu views, %d tokens (run %zu), tower %.1f ms\n", path.c_str(), p.views.size(), tokens, p.run.size(),
                  st.gpu_ms);
      continue;
    }
    const vision::Prepared p = vision::preprocess(img, spec);
    const int tokens = p.tokens(spec.merge);
    std::vector<float> emb(size_t(tokens) * size_t(I.hidden));
    ImageRef r;
    r.pixels = p.pixels.data();
    r.grid_t = p.grid_t;
    r.grid_h = p.grid_h;
    r.grid_w = p.grid_w;
    PassStats st;
    m->encode_images({r}, emb.data(), &st);
    const std::string stem = out + "/" + std::to_string(i - 4);
    if (!write_f32(stem + ".pixels.f32", p.pixels) || !write_f32(stem + ".emb.f32", emb)) {
      std::fprintf(stderr, "write failed under %s\n", out.c_str());
      return 1;
    }
    json::Value e = json::Value::object();
    e.set("file", path);
    e.set("width", int64_t(img.width));
    e.set("height", int64_t(img.height));
    json::Value g = json::Value::array();
    g.push_back(int64_t(p.grid_t));
    g.push_back(int64_t(p.grid_h));
    g.push_back(int64_t(p.grid_w));
    e.set("grid", g);
    e.set("rows", int64_t(p.rows()));
    e.set("row_dim", int64_t(spec.row_dim()));
    e.set("tokens", int64_t(tokens));
    e.set("hidden", int64_t(I.hidden));
    e.set("gpu_ms", st.gpu_ms);  // the last command buffer only when the tower commits per block (>= 2048 patches)
    e.set("wall_ms", st.wall_ms);
    e.set("stem", std::to_string(i - 4));
    index.push_back(e);
    std::printf("%s: grid %dx%dx%d, %d tokens, tower %.1f ms (wall %.1f ms)\n", path.c_str(), p.grid_t, p.grid_h, p.grid_w,
                tokens, st.gpu_ms, st.wall_ms);
    all.push_back(p);
    embs.push_back(std::move(emb));
  }
  // The same images in one call (one batched tower pass, as a prompt with
  // several images runs): each image's embeddings against its own run.
  if (all.size() > 1) {
    std::vector<ImageRef> refs;
    size_t total = 0;
    for (const vision::Prepared& p : all) {
      ImageRef r;
      r.pixels = p.pixels.data();
      r.grid_t = p.grid_t;
      r.grid_h = p.grid_h;
      r.grid_w = p.grid_w;
      refs.push_back(r);
      total += size_t(p.tokens(spec.merge)) * size_t(I.hidden);
    }
    std::vector<float> batch(total);
    m->encode_images(refs, batch.data());
    size_t off = 0;
    for (size_t i = 0; i < embs.size(); ++i) {
      const auto& e = embs[i];
      double worst = 0, scale = 0;
      for (size_t k = 0; k < e.size(); ++k) {
        worst = std::max(worst, std::fabs(double(batch[off + k]) - double(e[k])));
        scale = std::max(scale, std::fabs(double(e[k])));
      }
      off += e.size();
      std::printf("image %zu batched with the others vs alone: max |diff| %.3g (max |value| %.3g)\n", i, worst, scale);
    }
  }
  json::Value root = json::Value::object();
  root.set("model", model);
  root.set("max_pixels", int64_t(max_pixels));
  root.set("images", index);
  FILE* f = std::fopen((out + "/index.json").c_str(), "w");
  if (!f) return 1;
  const std::string s = json::dump(root);
  std::fwrite(s.data(), 1, s.size(), f);
  std::fclose(f);
  return 0;
}
