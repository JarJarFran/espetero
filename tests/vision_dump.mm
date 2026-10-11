// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Vision fidelity dumps: for each image file, Espetero's processor output
// (the patch rows) and the vision tower's merged token embeddings, as raw
// little-endian float32 files beside a JSON index, for
// tools/vision_fidelity.py (which runs mlx-vlm on the same files).
//
//   build/vision_dump <MLX model dir> <out dir> <max_pixels> <image> [<image> ...]
//
// Writes <out>/<i>.pixels.f32 ([rows, 3*temporal*patch*patch]),
// <out>/<i>.emb.f32 ([tokens, hidden]) and <out>/index.json.
#include <sys/stat.h>

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
  vision::Spec spec;
  try {
    spec = vision::spec_from_model_dir(model, max_pixels);
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
  json::Value index = json::Value::array();
  for (int i = 4; i < argc; ++i) {
    const std::string path = argv[i];
    vision::RGBImage img;
    std::string err;
    if (!vision::load_rgb_file(path, img, err)) {
      std::fprintf(stderr, "%s: %s\n", path.c_str(), err.c_str());
      return 1;
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
    e.set("gpu_ms", st.gpu_ms);
    e.set("stem", std::to_string(i - 4));
    index.push_back(e);
    std::printf("%s: grid %dx%dx%d, %d tokens, tower %.1f ms\n", path.c_str(), p.grid_t, p.grid_h, p.grid_w, tokens,
                st.gpu_ms);
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
