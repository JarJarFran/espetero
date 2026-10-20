// SPDX-License-Identifier: MIT AND Apache-2.0
// Copyright (c) 2026 Francisco José García Barbero
//
// The geometry (smart_resize, _is_image_too_large, the tile grid of
// crop_image_to_patches, the token counts and runs) is translated from Hugging
// Face transformers (models/lfm2_vl/image_processing_lfm2_vl.py and
// processing_lfm2_vl.py, Copyright 2025 the HuggingFace Inc. team); Apache
// License 2.0, see THIRD_PARTY_NOTICES.md.

// LFM2-VL image processor: views (tiles and thumbnail, or one view), patch
// rows and the placeholder's token run. See vision.h.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

#include "../util/json.h"
#include "vision.h"

// numpy evaluates x / 255, - mean, / std as separate float32 operations.
#pragma clang fp contract(off)

namespace espetero::vision {

namespace {

bool read_text(const std::string& path, std::string& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::stringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

// Python's round(x / f) * f (round half to even).
int round_by_factor(int x, int f) { return int(std::nearbyint(double(x) / double(f))) * f; }

// smart_resize: (width, height).
std::pair<int, int> lfm2_smart_resize(int h, int w, const Lfm2Spec& s) {
  const int tf = s.patch * s.factor;
  const double mn = double(s.min_tokens) * s.patch * s.patch * s.factor * s.factor;
  const double mx = double(s.max_tokens) * s.patch * s.patch * s.factor * s.factor;
  int hb = std::max(tf, round_by_factor(h, tf));
  int wb = std::max(tf, round_by_factor(w, tf));
  if (double(hb) * wb > mx) {
    const double beta = std::sqrt(double(h) * double(w) / mx);
    hb = std::max(tf, int(std::floor(double(h) / beta / tf)) * tf);
    wb = std::max(tf, int(std::floor(double(w) / beta / tf)) * tf);
  } else if (double(hb) * wb < mn) {
    const double beta = std::sqrt(mn / (double(h) * double(w)));
    hb = int(std::ceil(double(h) * beta / tf)) * tf;
    wb = int(std::ceil(double(w) * beta / tf)) * tf;
  }
  return {wb, hb};
}

bool too_large(int h, int w, const Lfm2Spec& s) {
  const int tf = s.patch * s.factor;
  const int hb = std::max(s.patch, round_by_factor(h, tf));
  const int wb = std::max(s.patch, round_by_factor(w, tf));
  return double(hb) * wb > double(s.max_tokens) * s.patch * s.patch * s.factor * s.factor * s.tolerance;
}

// find_closest_aspect_ratio over _target_ratios(min_tiles, max_tiles): (columns, rows).
std::pair<int, int> grid_of(int h, int w, const Lfm2Spec& s) {
  std::set<std::pair<int, int>> seen;
  std::vector<std::pair<int, int>> ratios;
  for (int n = s.min_tiles; n <= s.max_tiles; ++n)
    for (int a = 1; a <= n; ++a)
      for (int b = 1; b <= n; ++b)
        if (a * b >= s.min_tiles && a * b <= s.max_tiles && seen.insert({a, b}).second) ratios.push_back({a, b});
  // sorted(set(...), key=w * h): a stable sort of the set's iteration order. CPython iterates a
  // set of small int tuples in hash order, so equal products may come in another order than
  // here; ties on the product with equal aspect differences cannot happen (a * b equal and
  // a / b equal means the same pair), so the order inside a product does not change the result.
  std::stable_sort(ratios.begin(), ratios.end(), [](const auto& x, const auto& y) { return x.first * x.second < y.first * y.second; });
  const double aspect = double(w) / double(h);
  const double area = double(w) * double(h);
  double best_diff = std::numeric_limits<double>::infinity();
  std::pair<int, int> best{1, 1};
  for (const auto& r : ratios) {
    const double d = std::fabs(aspect - double(r.first) / double(r.second));
    if (d < best_diff) {
      best_diff = d;
      best = r;
    } else if (d == best_diff && area > 0.5 * double(s.tile) * s.tile * r.first * r.second) {
      best = r;
    }
  }
  return best;
}

RGBImage crop(const RGBImage& in, int x0, int y0, int w, int h) {
  RGBImage out;
  out.width = w;
  out.height = h;
  out.rgb.resize(size_t(w) * size_t(h) * 3);
  for (int y = 0; y < h; ++y)
    std::memcpy(&out.rgb[size_t(y) * size_t(w) * 3], &in.rgb[(size_t(y0 + y) * size_t(in.width) + size_t(x0)) * 3], size_t(w) * 3);
  return out;
}

// One view's patch rows appended to `out`: row-major patches, (py, px, c).
void append_patches(const RGBImage& v, const Lfm2Spec& s, std::vector<float>& out) {
  const int p = s.patch, nh = v.height / p, nw = v.width / p;
  const size_t base = out.size();
  out.resize(base + size_t(nh) * size_t(nw) * size_t(s.row_dim()));
  float* dst = out.data() + base;
  for (int i = 0; i < nh; ++i)
    for (int j = 0; j < nw; ++j)
      for (int py = 0; py < p; ++py)
        for (int px = 0; px < p; ++px) {
          const uint8_t* q = &v.rgb[(size_t(i * p + py) * size_t(v.width) + size_t(j * p + px)) * 3];
          for (int c = 0; c < 3; ++c) {
            const float x = float(q[c]) / 255.0f;
            *dst++ = (x - s.mean[c]) / s.std[c];
          }
        }
}

}  // namespace

Lfm2Spec lfm2_spec_from_model_dir(const std::string& dir) {
  Lfm2Spec s;
  std::string text;
  if (read_text(dir + "/config.json", text)) {
    const json::Value cfg = json::parse(text);
    s.patch = int(cfg.get_int("encoder_patch_size", s.patch));
    s.factor = int(cfg.get_int("downsample_factor", s.factor));
    s.min_tokens = int(cfg.get_int("min_image_tokens", s.min_tokens));
    s.max_tokens = int(cfg.get_int("max_image_tokens", s.max_tokens));
    s.min_tiles = int(cfg.get_int("min_tiles", s.min_tiles));
    s.max_tiles = int(cfg.get_int("max_tiles", s.max_tiles));
    s.tile = int(cfg.get_int("tile_size", s.tile));
    s.tolerance = cfg.get_double("max_pixels_tolerance", s.tolerance);
    s.splitting = cfg.get_bool("do_image_splitting", s.splitting);
    s.thumbnail = cfg.get_bool("use_thumbnail", s.thumbnail);
  }
  for (const char* name : {"processor_config.json", "preprocessor_config.json"}) {
    if (!read_text(dir + "/" + name, text)) continue;
    json::Value pc = json::parse(text);
    if (const json::Value* ip = pc.get("image_processor"); ip && ip->is_object()) pc = *ip;
    s.patch = int(pc.get_int("encoder_patch_size", s.patch));
    s.factor = int(pc.get_int("downsample_factor", s.factor));
    s.min_tokens = int(pc.get_int("min_image_tokens", s.min_tokens));
    s.max_tokens = int(pc.get_int("max_image_tokens", s.max_tokens));
    s.min_tiles = int(pc.get_int("min_tiles", s.min_tiles));
    s.max_tiles = int(pc.get_int("max_tiles", s.max_tiles));
    s.tile = int(pc.get_int("tile_size", s.tile));
    s.tolerance = pc.get_double("max_pixels_tolerance", s.tolerance);
    s.splitting = pc.get_bool("do_image_splitting", s.splitting);
    s.thumbnail = pc.get_bool("use_thumbnail", s.thumbnail);
    if (const json::Value* m = pc.get("image_mean"); m && m->is_array() && m->size() == 3)
      for (int i = 0; i < 3; ++i) s.mean[i] = float((*m)[size_t(i)].as_double());
    if (const json::Value* sd = pc.get("image_std"); sd && sd->is_array() && sd->size() == 3)
      for (int i = 0; i < 3; ++i) s.std[i] = float((*sd)[size_t(i)].as_double());
    const int64_t rs = pc.get_int("resample", 3);
    if (rs != 2 && rs != 3) throw std::runtime_error("LFM2-VL processor: only bilinear (2) and bicubic (3) resampling are supported");
    s.bilinear = rs == 2;
    break;
  }
  if (s.max_tiles > 10 || s.min_tiles < 1 || s.min_tiles > s.max_tiles || s.patch <= 0 || s.factor != 2 || s.tile % (s.patch * s.factor))
    throw std::runtime_error("unsupported LFM2-VL processor configuration");
  return s;
}

Lfm2Layout lfm2_layout(int height, int width, const Lfm2Spec& s) {
  if (height <= 0 || width <= 0) throw std::invalid_argument("empty image");
  Lfm2Layout L;
  const auto [rw, rh] = lfm2_smart_resize(height, width, s);
  L.view_h = rh;
  L.view_w = rw;
  const bool split = s.splitting && !(s.min_tiles == 1 && s.max_tiles == 1);
  if (split && too_large(height, width, s)) {
    const auto [gw, gh] = grid_of(height, width, s);
    L.tiled = true;
    L.rows = gh;
    L.cols = gw;
    L.thumbnail = s.thumbnail && gw * gh != 1;
  }
  return L;
}

std::vector<int32_t> lfm2_run(const Lfm2Layout& L, const Lfm2Spec& s) {
  const int per_tile = (s.tile / s.patch + s.factor - 1) / s.factor;
  const int th = (L.view_h / s.patch + s.factor - 1) / s.factor, tw = (L.view_w / s.patch + s.factor - 1) / s.factor;
  std::vector<int32_t> run{s.start_id};
  if (L.tiled && (L.rows > 1 || L.cols > 1)) {
    for (int r = 0; r < L.rows; ++r)
      for (int c = 0; c < L.cols; ++c) {
        run.push_back(s.grid_ids[size_t(r * 10 + c)]);
        run.insert(run.end(), size_t(per_tile) * size_t(per_tile), s.image_id);
      }
    if (s.thumbnail) {
      run.push_back(s.thumb_id);
      run.insert(run.end(), size_t(th) * size_t(tw), s.image_id);
    }
  } else {
    run.insert(run.end(), size_t(th) * size_t(tw), s.image_id);
  }
  run.push_back(s.end_id);
  return run;
}

Lfm2Prepared lfm2_preprocess(const RGBImage& image, const Lfm2Spec& s) {
  const Lfm2Layout L = lfm2_layout(image.height, image.width, s);
  auto resize = [&](const RGBImage& im, int w, int h) { return s.bilinear ? resize_bilinear(im, w, h) : resize_bicubic(im, w, h); };
  Lfm2Prepared out;
  out.run = lfm2_run(L, s);
  std::vector<RGBImage> views;
  if (L.tiled) {
    const RGBImage big = resize(image, s.tile * L.cols, s.tile * L.rows);
    for (int r = 0; r < L.rows; ++r)
      for (int c = 0; c < L.cols; ++c) views.push_back(crop(big, c * s.tile, r * s.tile, s.tile, s.tile));
    if (L.thumbnail) views.push_back(resize(image, L.view_w, L.view_h));
  } else {
    views.push_back(resize(image, L.view_w, L.view_h));
  }
  // The views' image tokens in the run, in order.
  std::vector<int> firsts;
  for (size_t i = 0; i < out.run.size(); ++i)
    if (out.run[i] == s.image_id && (i == 0 || out.run[i - 1] != s.image_id)) firsts.push_back(int(i));
  if (firsts.size() != views.size()) throw std::runtime_error("LFM2-VL: views and token runs differ");
  for (size_t k = 0; k < views.size(); ++k) {
    Lfm2Prepared::View v;
    v.gh = views[k].height / s.patch;
    v.gw = views[k].width / s.patch;
    v.offset = out.pixels.size();
    v.token = firsts[k];
    append_patches(views[k], s, out.pixels);
    out.views.push_back(v);
  }
  // The layout and the pixels: equal digests mean equal views and pixels.
  out.digest = digest(out.pixels.data(), out.pixels.size(),
                      {int(views.size()), L.rows * 64 + L.cols, (L.view_h / s.patch) * 4096 + L.view_w / s.patch});
  return out;
}

}  // namespace espetero::vision
