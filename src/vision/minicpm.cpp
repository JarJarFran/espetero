// SPDX-License-Identifier: MIT AND Apache-2.0
// Copyright (c) 2026 Francisco José García Barbero
//
// The geometry (find_best_resize, get_refine_size, get_sliced_grid, the
// slices of divide_to_patches, the token counts and the placeholder text) is
// translated from Hugging Face transformers
// (models/minicpm_ocr/image_processing_minicpm_ocr.py and
// processing_minicpm_ocr.py, Copyright 2026 OpenBMB and the HuggingFace Inc.
// team); Apache License 2.0, see THIRD_PARTY_NOTICES.md.

// MiniCPM-OCR image processor: the overview and the slices, their patch rows
// and the placeholder's token run. See vision.h.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include "../util/json.h"
#include "vision.h"

// The reference evaluates x / 255 and the normalisation as float32 operations.
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

// ensure_divide: max(round(length / divisor) * divisor, divisor), Python's round (half to even).
int ensure_divide(double length, int divisor) {
  return std::max(int(std::nearbyint(length / double(divisor))) * divisor, divisor);
}

// find_best_resize: (height, width), sides multiples of patch * 2 ("4x": one 2x2 merge).
std::pair<int, int> best_resize(double height, double width, const MinicpmSpec& s, bool allow_upscale) {
  if (height * width > double(s.scale) * double(s.scale) || allow_upscale) {
    const double aspect = width / height;
    height = double(int64_t(double(s.scale) / std::sqrt(aspect)));
    width = double(int64_t(height * aspect));
  }
  const int f = s.patch * 2;
  return {ensure_divide(height, f), ensure_divide(width, f)};
}

// get_sliced_grid: (rows, columns), {0, 0} when the image is not sliced.
std::pair<int, int> sliced_grid(int height, int width, const MinicpmSpec& s) {
  if (!s.slice_mode) return {0, 0};
  const double log_ratio = std::log(double(width) / double(height));
  const double ratio = double(width) * double(height) / (double(s.scale) * double(s.scale));
  const int multiple = std::min(int(std::ceil(ratio)), s.max_slices);
  if (multiple <= 1) return {0, 0};
  std::pair<int, int> best{1, 1};
  double min_error = std::numeric_limits<double>::infinity();
  for (int n : {multiple - 1, multiple, multiple + 1}) {
    if (n == 1 || n > s.max_slices) continue;
    for (int rows = 1; rows <= n; ++rows) {
      if (n % rows) continue;
      const int cols = n / rows;
      const double err = std::fabs(log_ratio - std::log(double(cols) / double(rows)));
      if (err < min_error) {
        best = {rows, cols};
        min_error = err;
      } else if (err == min_error && rows > best.first) {
        best = {rows, cols};
      }
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

// One view's patch rows appended to `out`: row-major patches, (py, px, c) inside a patch.
void append_patches(const RGBImage& v, const MinicpmSpec& s, std::vector<float>& out) {
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

MinicpmSpec minicpm_spec_from_model_dir(const std::string& dir) {
  MinicpmSpec s;
  std::string text;
  for (const char* name : {"preprocessor_config.json", "processor_config.json"}) {
    if (!read_text(dir + "/" + name, text)) continue;
    json::Value pc = json::parse(text);
    if (const json::Value* ip = pc.get("image_processor"); ip && ip->is_object()) pc = *ip;
    if (!pc.get("scale_resolution") && !pc.get("image_mean")) continue;
    s.patch = int(pc.get_int("patch_size", s.patch));
    s.scale = int(pc.get_int("scale_resolution", s.scale));
    s.max_slices = int(pc.get_int("max_slice_nums", s.max_slices));
    s.slice_mode = pc.get_bool("slice_mode", s.slice_mode);
    s.use_image_id = pc.get_bool("use_image_id", s.use_image_id);
    if (pc.get_string("downsample_mode", "4x") != "4x")
      throw std::runtime_error("MiniCPM-OCR processor: only downsample_mode \"4x\" is supported");
    if (pc.get_string("small_image_upsize", "always") != "always")
      throw std::runtime_error("MiniCPM-OCR processor: only small_image_upsize \"always\" is supported");
    if (const json::Value* m = pc.get("image_mean"); m && m->is_array() && m->size() == 3)
      for (int i = 0; i < 3; ++i) s.mean[i] = float((*m)[size_t(i)].as_double());
    if (const json::Value* sd = pc.get("image_std"); sd && sd->is_array() && sd->size() == 3)
      for (int i = 0; i < 3; ++i) s.std[i] = float((*sd)[size_t(i)].as_double());
    break;
  }
  if (s.patch <= 0 || s.scale < 2 * s.patch || s.max_slices < 1) throw std::runtime_error("unsupported MiniCPM-OCR processor configuration");
  return s;
}

MinicpmLayout minicpm_layout(int height, int width, const MinicpmSpec& s) {
  if (height <= 0 || width <= 0) throw std::invalid_argument("empty image");
  MinicpmLayout L;
  const auto [rows, cols] = sliced_grid(height, width, s);
  L.rows = rows;
  L.cols = cols;
  // Unsliced images are always enlarged to the slice budget (small_image_upsize "always").
  const auto [sh, sw] = best_resize(double(height), double(width), s, rows == 0);
  L.source_h = sh;
  L.source_w = sw;
  if (rows > 0) {
    // get_refine_size: the image resized so that its grid's cells are best_resize'd slices.
    const int rw = ensure_divide(double(width), cols), rh = ensure_divide(double(height), rows);
    const auto [bh, bw] = best_resize(double(rh) / double(rows), double(rw) / double(cols), s, true);
    L.slice_h = bh;
    L.slice_w = bw;
  }
  return L;
}

std::vector<int32_t> minicpm_run(const MinicpmLayout& L, const MinicpmSpec& s) {
  const int p = s.patch;
  const size_t n0 = size_t(L.source_h / p) * size_t(L.source_w / p) / 4;
  std::vector<int32_t> run{s.im_start_id};
  run.insert(run.end(), n0, s.image_id);
  run.push_back(s.im_end_id);
  if (L.sliced()) {
    const size_t n = size_t(L.slice_h / p) * size_t(L.slice_w / p) / 4;
    for (int r = 0; r < L.rows; ++r) {
      if (r > 0) run.push_back(s.newline_id);
      for (int c = 0; c < L.cols; ++c) {
        run.push_back(s.sl_start_id);
        run.insert(run.end(), n, s.image_id);
        run.push_back(s.sl_end_id);
      }
    }
  }
  return run;
}

MinicpmPrepared minicpm_preprocess(const RGBImage& image, const MinicpmSpec& s) {
  const MinicpmLayout L = minicpm_layout(image.height, image.width, s);
  MinicpmPrepared out;
  out.run = minicpm_run(L, s);
  std::vector<RGBImage> views;
  views.push_back(resize_bicubic(image, L.source_w, L.source_h));
  if (L.sliced()) {
    const RGBImage big = resize_bicubic(image, L.slice_w * L.cols, L.slice_h * L.rows);
    for (int r = 0; r < L.rows; ++r)
      for (int c = 0; c < L.cols; ++c) views.push_back(crop(big, c * L.slice_w, r * L.slice_h, L.slice_w, L.slice_h));
  }
  std::vector<int> firsts;  // each view's first image token in the run
  for (size_t i = 0; i < out.run.size(); ++i)
    if (out.run[i] == s.image_id && (i == 0 || out.run[i - 1] != s.image_id)) firsts.push_back(int(i));
  if (firsts.size() != views.size()) throw std::runtime_error("MiniCPM-OCR: views and token runs differ");
  for (size_t k = 0; k < views.size(); ++k) {
    MinicpmPrepared::View v;
    v.gh = views[k].height / s.patch;
    v.gw = views[k].width / s.patch;
    v.offset = out.pixels.size();
    v.token = firsts[k];
    append_patches(views[k], s, out.pixels);
    out.views.push_back(v);
  }
  out.digest = digest(out.pixels.data(), out.pixels.size(),
                      {int(views.size()), L.rows * 64 + L.cols, (L.source_h / s.patch) * 4096 + L.source_w / s.patch});
  return out;
}

}  // namespace espetero::vision
