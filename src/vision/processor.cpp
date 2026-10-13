// SPDX-License-Identifier: MIT AND Apache-2.0
// Copyright (c) 2026 Francisco José García Barbero
//
// smart_resize is translated from Hugging Face transformers
// (models/qwen2_vl/image_processing_qwen2_vl.py, Copyright 2024 The Qwen team,
// Alibaba Group and the HuggingFace Inc. team); Apache License 2.0, see THIRD_PARTY_NOTICES.md.

// Qwen2/3-VL image processor: smart_resize, rescale + normalise, temporal
// duplication and patch rows in merge-window order; image token expansion
// and M-RoPE positions. Mirrors jevfire/metal/vision.py.
#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <tuple>
#include <sstream>

#include "../util/json.h"
#include "vision.h"

// numpy evaluates x * (1/255), - mean, / std as separate float32 operations;
// a fused multiply-subtract would round differently.
#pragma clang fp contract(off)

namespace espetero::vision {

// ---- spec -------------------------------------------------------------------

Spec spec_from_config(const json::Value& cfg, const json::Value* pre, int64_t max_pixels) {
  Spec s;
  s.set_budget(max_pixels);
  if (const json::Value* vc = cfg.get("vision_config"); vc && vc->is_object()) {
    s.patch = int(vc->get_int("patch_size", s.patch));
    s.temporal = int(vc->get_int("temporal_patch_size", s.temporal));
    s.merge = int(vc->get_int("spatial_merge_size", s.merge));
  }
  s.image_token_id = int32_t(cfg.get_int("image_token_id", s.image_token_id));
  s.vision_start_id = int32_t(cfg.get_int("vision_start_token_id", s.vision_start_id));
  s.vision_end_id = int32_t(cfg.get_int("vision_end_token_id", s.vision_end_id));
  if (pre && pre->is_object()) {
    const json::Value* pc = pre->get("image_processor");
    if (!pc || !pc->is_object()) pc = pre;
    auto triple = [](const json::Value* v, float out[3]) {
      if (!v || !v->is_array() || v->size() != 3) return;
      // Python floats rounded to float32, as np.asarray(..., dtype=float32).
      for (int i = 0; i < 3; i++) out[i] = float((*v)[i].as_double());
    };
    triple(pc->get("image_mean"), s.mean);
    triple(pc->get("image_std"), s.std);
  }
  return s;
}

static bool read_text(const std::string& path, std::string& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::stringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

Spec spec_from_model_dir(const std::string& dir, int64_t max_pixels) {
  std::string text;
  if (!read_text(dir + "/config.json", text))
    throw std::runtime_error("vision: cannot read " + dir + "/config.json");
  const json::Value cfg = json::parse(text);
  for (const char* name : {"preprocessor_config.json", "processor_config.json"}) {
    if (!read_text(dir + "/" + name, text)) continue;
    const json::Value pre = json::parse(text);
    return spec_from_config(cfg, &pre, max_pixels);
  }
  return spec_from_config(cfg, nullptr, max_pixels);
}

// ---- smart_resize -----------------------------------------------------------------

std::pair<int, int> smart_resize(int height, int width, int factor, int64_t min_pixels,
                                 int64_t max_pixels) {
  if (height <= 0 || width <= 0 || factor <= 0)
    throw std::invalid_argument("smart_resize: sizes must be positive");
  if (double(std::max(height, width)) / double(std::min(height, width)) > 200)
    throw std::invalid_argument("absolute aspect ratio must be smaller than 200");
  // Python's round() is round-half-to-even, like nearbyint in the default mode.
  int64_t h_bar = int64_t(std::nearbyint(double(height) / factor)) * factor;
  int64_t w_bar = int64_t(std::nearbyint(double(width) / factor)) * factor;
  const double area = double(int64_t(height) * width);  // exact below 2^53
  if (h_bar * w_bar > max_pixels) {
    const double beta = std::sqrt(area / double(max_pixels));
    h_bar = std::max<int64_t>(factor, int64_t(std::floor(height / beta / factor)) * factor);
    w_bar = std::max<int64_t>(factor, int64_t(std::floor(width / beta / factor)) * factor);
  } else if (h_bar * w_bar < min_pixels) {
    const double beta = std::sqrt(double(min_pixels) / area);
    h_bar = int64_t(std::ceil(height * beta / factor)) * factor;
    w_bar = int64_t(std::ceil(width * beta / factor)) * factor;
  }
  if (h_bar <= 0 || w_bar <= 0 || h_bar > (1 << 20) || w_bar > (1 << 20))
    throw std::invalid_argument("smart_resize: resulting size out of range");
  return {int(h_bar), int(w_bar)};
}

// ---- normalise + patchify ---------------------------------------------------------

Prepared patchify(const RGBImage& img, const Spec& spec) {
  const int ps = spec.patch, tps = spec.temporal, ms = spec.merge;
  if (ps <= 0 || tps <= 0 || ms <= 0) throw std::invalid_argument("preprocess: bad spec");
  const int h = img.height, w = img.width;
  if (h % (ps * ms) || w % (ps * ms) || h <= 0 || w <= 0)
    throw std::invalid_argument("preprocess: image size is not a multiple of patch * merge");
  const int gh = h / ps, gw = w / ps;
  const size_t dim = size_t(3) * tps * ps * ps;

  // Every value is lut[c][byte]: (byte * float32(1/255) - mean) / std, each
  // step rounded to float32 exactly as numpy does it.
  float lut[3][256];
  const float inv255 = float(1 / 255.0);
  for (int c = 0; c < 3; c++)
    for (int v = 0; v < 256; v++) {
      const float x = float(v) * inv255;
      const float y = x - spec.mean[c];
      lut[c][v] = y / spec.std[c];
    }

  Prepared out;
  out.grid_t = 1;
  out.grid_h = gh;
  out.grid_w = gw;
  out.pixels.resize(size_t(gh) * gw * dim);

  // Row order (gh/ms, gw/ms, ms, ms); columns (C, T, ps, ps), T copies equal.
  const int bw_n = gw / ms;
  const int n_rows = gh * gw;
  struct Ctx {
    const RGBImage* img;
    const float (*lut)[256];
    float* dst;
    int ps, tps, ms, bw_n, w;
    size_t dim;
  } ctx{&img, lut, out.pixels.data(), ps, tps, ms, bw_n, w, dim};
  auto rows = [](const Ctx& c, int r0, int r1) {
    const uint8_t* src = c.img->rgb.data();
    const size_t block = size_t(c.ps) * c.ps;
    for (int r = r0; r < r1; r++) {
      const int mw = r % c.ms, mh = (r / c.ms) % c.ms, win = r / (c.ms * c.ms);
      const int bw = win % c.bw_n, bh = win / c.bw_n;
      const int ph = bh * c.ms + mh, pw = bw * c.ms + mw;
      float* row = c.dst + size_t(r) * c.dim;
      for (int ch = 0; ch < 3; ch++) {
        float* o = row + size_t(ch) * c.tps * block;
        const float* lut = c.lut[ch];
        for (int py = 0; py < c.ps; py++) {
          const uint8_t* s = src + (size_t(ph * c.ps + py) * c.w + size_t(pw) * c.ps) * 3 + ch;
          float* d = o + size_t(py) * c.ps;
          for (int px = 0; px < c.ps; px++) d[px] = lut[s[3 * px]];
        }
        for (int t = 1; t < c.tps; t++) std::memcpy(o + t * block, o, block * sizeof(float));
      }
    }
  };
  const int chunks = n_rows >= 256 ? std::min(16, n_rows / 64) : 1;
  if (chunks <= 1) {
    rows(ctx, 0, n_rows);
  } else {
    struct Job {
      Ctx* ctx;
      decltype(rows)* fn;
      int n, chunks;
    } job{&ctx, &rows, n_rows, chunks};
    dispatch_apply_f(size_t(chunks), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &job,
                     [](void* p, size_t i) {
                       auto* j = static_cast<Job*>(p);
                       (*j->fn)(*j->ctx, int(int64_t(j->n) * i / j->chunks),
                                int(int64_t(j->n) * (i + 1) / j->chunks));
                     });
  }
  out.digest = digest(out.pixels.data(), out.pixels.size(), out.grid());
  return out;
}

Prepared preprocess(const RGBImage& image, const Spec& spec) {
  const auto [h, w] = smart_resize(image.height, image.width, spec.factor(), spec.min_pixels,
                                   spec.max_pixels);
  if (w == image.width && h == image.height) return patchify(image, spec);
  return patchify(resize_bicubic(image, w, h), spec);
}

// ---- DeepSeek images -----------------------------------------------------------------

namespace {

// Python's round(): half to even.
long py_round(double x) { return long(std::nearbyint(x)); }

int64_t ds_tokens(int64_t lh, int64_t lw) { return lh * (lw + 1) + 2; }

std::pair<int64_t, int64_t> ds_llm_grid(int64_t best_h, int64_t best_w, int patch, int down) {
  return {int64_t(std::ceil(double(best_h / patch) / down)), int64_t(std::ceil(double(best_w / patch) / down))};
}

}  // namespace

std::array<int, 4> deepseek_plan(int width, int height, int patch, int down, int max_tokens, int64_t min_pixels) {
  if (width <= 0 || height <= 0 || patch <= 0 || down <= 0) throw std::invalid_argument("image: bad size");
  int64_t w = width, h = height;
  if (w * h < min_pixels) {  // plan_image_grid: scale small images up
    const double ratio = std::sqrt(double(min_pixels) / double(w * h));
    w = int64_t(double(w) * ratio), h = int64_t(double(h) * ratio);
  }
  int64_t best_w = int64_t(std::ceil(double(w) / patch)) * patch, best_h = int64_t(std::ceil(double(h) / patch)) * patch;
  auto [lh, lw] = ds_llm_grid(best_h, best_w, patch, down);
  if (ds_tokens(lh, lw) > max_tokens) {  // solve_resize_ratio
    const double r = double(h) / double(w);
    const double max_w = std::sqrt((max_tokens - 2) / r + 0.25) - 0.5, max_h = max_w * r;
    const int64_t cell = int64_t(patch) * down;
    if (max_w < 1.0) {
      best_h = (max_tokens - 2) / 2 * cell, best_w = cell;
    } else if (max_h < 1.0) {
      best_h = cell, best_w = (max_tokens - 3) * cell;
    } else {
      const double beta = std::min(std::floor(max_w) * double(cell) / double(w), std::floor(max_h) * double(cell) / double(h));
      best_h = int64_t(std::floor(double(h) * beta / patch)) * patch;
      best_w = int64_t(std::floor(double(w) * beta / patch)) * patch;
    }
    std::tie(lh, lw) = ds_llm_grid(best_h, best_w, patch, down);
    if (ds_tokens(lh, lw) > max_tokens || best_h <= 0 || best_w <= 0) throw std::invalid_argument("image: no plan fits the token budget");
  }
  return {int(best_h), int(best_w), int(lh), int(lw)};
}

Prepared preprocess_deepseek(const RGBImage& image, int patch, int down, int max_tokens, int64_t min_pixels) {
  const auto plan = deepseek_plan(image.width, image.height, patch, down, max_tokens, min_pixels);
  const int H = plan[0], W = plan[1];
  // ImageOps.pad(image, (W, H), BICUBIC, color=(127, 127, 127), centering=(0.5, 0.5))
  RGBImage canvas;
  canvas.width = W, canvas.height = H;
  canvas.rgb.assign(size_t(W) * size_t(H) * 3, 127);
  const double im_ratio = double(image.width) / double(image.height), dest = double(W) / double(H);
  if (im_ratio == dest) {
    canvas = (image.width == W && image.height == H) ? image : resize_bicubic(image, W, H);
  } else {
    int nw = W, nh = H, x = 0, y = 0;
    if (im_ratio > dest) {
      nh = int(py_round(double(image.height) / double(image.width) * W));
      y = int(py_round((H - nh) * 0.5));
    } else {
      nw = int(py_round(double(image.width) / double(image.height) * H));
      x = int(py_round((W - nw) * 0.5));
    }
    nw = std::max(nw, 1), nh = std::max(nh, 1);
    const RGBImage r = (nw == image.width && nh == image.height) ? image : resize_bicubic(image, nw, nh);
    for (int yy = 0; yy < nh; ++yy) {
      const int ty = y + yy;
      if (ty < 0 || ty >= H) continue;
      for (int xx = 0; xx < nw; ++xx) {
        const int tx = x + xx;
        if (tx < 0 || tx >= W) continue;
        std::memcpy(&canvas.rgb[(size_t(ty) * size_t(W) + size_t(tx)) * 3], &r.rgb[(size_t(yy) * size_t(nw) + size_t(xx)) * 3], 3);
      }
    }
  }
  Prepared out;
  const int gh = H / patch, gw = W / patch, K = 3 * patch * patch;
  out.grid_t = 1, out.grid_h = gh, out.grid_w = gw;
  out.pixels.resize(size_t(gh) * size_t(gw) * size_t(K));
  auto bf16 = [](float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    std::memcpy(&f, &u, 4);
    return f;
  };
  for (int py = 0; py < gh; ++py)
    for (int px = 0; px < gw; ++px) {
      float* o = out.pixels.data() + (size_t(py) * size_t(gw) + size_t(px)) * size_t(K);
      for (int c = 0; c < 3; ++c)
        for (int i = 0; i < patch; ++i)
          for (int j = 0; j < patch; ++j) {
            const uint8_t v = canvas.rgb[((size_t(py) * patch + size_t(i)) * size_t(W) + size_t(px) * patch + size_t(j)) * 3 + size_t(c)];
            const float x = float(v) / 255.0f;
            o[(size_t(c) * patch + size_t(i)) * patch + size_t(j)] = bf16((x - 0.5f) / 0.5f);
          }
    }
  out.digest = digest(out.pixels.data(), out.pixels.size(), out.grid());
  return out;
}

// ---- tokens and positions ---------------------------------------------------------

int image_tokens(std::array<int, 3> grid, int merge) {
  return grid[0] * grid[1] * grid[2] / (merge * merge);
}

std::array<int, 3> grid_for(int height, int width, const Spec& spec) {
  if (spec.patch <= 0) throw std::invalid_argument("preprocess: bad spec");
  const auto [h, w] = smart_resize(height, width, spec.factor(), spec.min_pixels, spec.max_pixels);
  return {1, h / spec.patch, w / spec.patch};  // patchify's grid_t, grid_h, grid_w
}

std::vector<int32_t> expand_image_tokens(const std::vector<int32_t>& ids,
                                         const std::vector<std::array<int, 3>>& grids,
                                         const Spec& spec) {
  std::vector<int32_t> out;
  out.reserve(ids.size());
  size_t g = 0;
  for (int32_t t : ids) {
    if (t != spec.image_token_id) {
      out.push_back(t);
      continue;
    }
    if (g >= grids.size()) throw std::invalid_argument("Image placeholders and images differ in number");
    out.insert(out.end(), size_t(std::max(0, image_tokens(grids[g], spec.merge))), t);
    g++;
  }
  if (g != grids.size()) throw std::invalid_argument("Image placeholders and images differ in number");
  return out;
}

std::vector<std::array<int32_t, 3>> mrope_positions(const std::vector<int32_t>& ids,
                                                    const std::vector<std::array<int, 3>>& grids,
                                                    const Spec& spec, int start, int* next_pos) {
  std::vector<std::array<int32_t, 3>> pos(ids.size());
  int64_t nxt = start;
  size_t i = 0, g = 0;
  const size_t n = ids.size();
  while (i < n) {
    if (ids[i] != spec.image_token_id) {
      const int32_t p = int32_t(nxt);
      pos[i] = {p, p, p};
      nxt++;
      i++;
      continue;
    }
    if (g >= grids.size()) throw std::invalid_argument("more image blocks than grids");
    const auto [t, h, w] = grids[g];
    const int gh = h / spec.merge, gw = w / spec.merge;
    const int64_t count = int64_t(t) * gh * gw;
    if (t <= 0 || gh <= 0 || gw <= 0)  // numpy: max() of an empty block raises
      throw std::invalid_argument("image grid is empty");
    if (i + size_t(count) > n) throw std::invalid_argument("image token run does not match its grid");
    for (size_t k = 0; k < size_t(count); k++)
      if (ids[i + k] != spec.image_token_id)
        throw std::invalid_argument("image token run does not match its grid");
    size_t k = i;
    for (int tt = 0; tt < t; tt++)
      for (int hh = 0; hh < gh; hh++)
        for (int ww = 0; ww < gw; ww++)
          pos[k++] = {int32_t(nxt + tt), int32_t(nxt + hh), int32_t(nxt + ww)};
    nxt += std::max({t - 1, gh - 1, gw - 1}) + 1;
    i += size_t(count);
    g++;
  }
  if (g != grids.size()) throw std::invalid_argument("fewer image blocks than grids");
  if (next_pos) *next_pos = int(nxt);
  return pos;
}

std::vector<int> image_rows(const std::vector<int32_t>& ids,
                            const std::vector<std::array<int, 3>>& grids, const Spec& spec) {
  std::vector<int> rows;
  size_t i = 0;
  while (i < ids.size()) {
    if (ids[i] != spec.image_token_id) {
      i++;
      continue;
    }
    if (rows.size() >= grids.size())
      throw std::invalid_argument("image placeholders and images differ in number");
    rows.push_back(int(i));
    // Like the reference, a run always advances by the grid's token count
    // (at least one token so a degenerate grid cannot stall the scan).
    i += size_t(std::max(1, image_tokens(grids[rows.size() - 1], spec.merge)));
  }
  if (rows.size() != grids.size())
    throw std::invalid_argument("image placeholders and images differ in number");
  return rows;
}

std::string Prepared::digest_hex() const {
  static const char* hex = "0123456789abcdef";
  std::string s;
  for (uint8_t b : digest) {
    s += hex[b >> 4];
    s += hex[b & 15];
  }
  return s;
}

}  // namespace espetero::vision
