// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "hidden_head.h"

#include <sys/stat.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "../util/json.h"

namespace espetero {

namespace {

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// The F32 tensors of a safetensors file by name, checked against `shape`.
std::vector<float> f32_tensor(const std::string& file, const std::string& bytes, const json::Value& header,
                              const std::string& name, const std::vector<int64_t>& shape) {
  const json::Value* t = header.get(name);
  if (!t || !t->is_object()) throw std::runtime_error(file + ": no tensor " + name);
  if (t->get_string("dtype", "") != "F32") throw std::runtime_error(file + ": " + name + " is not F32");
  const json::Value* sh = t->get("shape");
  const json::Value* off = t->get("data_offsets");
  if (!sh || !sh->is_array() || !off || !off->is_array() || off->size() != 2)
    throw std::runtime_error(file + ": " + name + ": malformed header entry");
  std::vector<int64_t> got;
  for (const json::Value& d : sh->as_array()) got.push_back(d.as_int());
  if (got != shape) throw std::runtime_error(file + ": " + name + " has an unexpected shape");
  size_t n = 1;
  for (int64_t d : shape) n *= size_t(d);
  const uint64_t base = 8 + uint64_t(bytes.size() >= 8 ? *reinterpret_cast<const uint64_t*>(bytes.data()) : 0);
  const uint64_t a = uint64_t((*off)[0].as_int()), b = uint64_t((*off)[1].as_int());
  if (b - a != n * 4 || base + b > bytes.size()) throw std::runtime_error(file + ": " + name + ": data out of range");
  std::vector<float> v(n);
  std::memcpy(v.data(), bytes.data() + base + a, n * 4);
  return v;
}

}  // namespace

bool HiddenHead::present(const std::string& dir) {
  struct stat st {};
  return ::stat((dir + "/hidden_head.json").c_str(), &st) == 0;
}

std::unique_ptr<HiddenHead> HiddenHead::load(const std::string& dir) {
  const json::Value cfg = json::parse(slurp(dir + "/hidden_head.json"));
  if (cfg.get_string("type", "") != "linear" || cfg.get_string("input", "") != "final_norm_last_row")
    throw std::runtime_error(dir + "/hidden_head.json: only linear heads over the last row's final-normed state");
  auto h = std::make_unique<HiddenHead>();
  h->slots_ = int(cfg.get_int("slots", 0));
  h->hidden_ = int(cfg.get_int("hidden_size", 0));
  if (h->slots_ < 1 || h->slots_ > 65536 || h->hidden_ < 1) throw std::runtime_error(dir + "/hidden_head.json: bad sizes");
  const std::string mode = cfg.get_string("attention_mode", "causal");
  if (mode != "causal" && mode != "noncausal_full_attention")
    throw std::runtime_error(dir + "/hidden_head.json: attention_mode is causal or noncausal_full_attention");
  h->noncausal_ = mode == "noncausal_full_attention";
  h->temperature_ = cfg.get_double("temperature", 1.0);
  if (!(h->temperature_ > 0) || !std::isfinite(h->temperature_))
    throw std::runtime_error(dir + "/hidden_head.json: temperature must be positive and finite");
  h->lm_head_ = cfg.get_bool("lm_head", true);
  if (const json::Value* codes = cfg.get("codes"); codes && !codes->is_null()) {
    if (!codes->is_array() || codes->size() > size_t(h->slots_))
      throw std::runtime_error(dir + "/hidden_head.json: codes is a list of at most `slots` strings");
    for (const json::Value& c : codes->as_array()) {
      if (!c.is_string() || c.as_string().empty()) throw std::runtime_error(dir + "/hidden_head.json: codes are strings");
      h->codes_.push_back(c.as_string());
    }
  }
  const std::string file = dir + "/hidden_head.safetensors";
  const std::string bytes = slurp(file);
  if (bytes.size() < 8) throw std::runtime_error(file + ": truncated");
  const uint64_t hl = *reinterpret_cast<const uint64_t*>(bytes.data());
  if (hl > bytes.size() - 8) throw std::runtime_error(file + ": truncated header");
  const json::Value header = json::parse(bytes.substr(8, size_t(hl)));
  const int64_t S = h->slots_, D = h->hidden_;
  h->w_ = f32_tensor(file, bytes, header, "weight", {S, D});
  h->b_ = f32_tensor(file, bytes, header, "bias", {S});
  if (cfg.get_bool("standardize", false)) {
    h->mu_ = f32_tensor(file, bytes, header, "mu", {D});
    std::vector<float> sd = f32_tensor(file, bytes, header, "sd", {D});
    for (float s : sd)
      if (!(s > 0)) throw std::runtime_error(file + ": sd must be positive");
    h->sd_ = std::move(sd);
  }
  return h;
}

std::vector<float> HiddenHead::logits(const float* h, int n) const {
  std::vector<double> x(static_cast<size_t>(hidden_));
  for (int j = 0; j < hidden_; ++j)
    x[size_t(j)] = mu_.empty() ? double(h[j]) : double(float(h[j] - mu_[size_t(j)]) / sd_[size_t(j)]);
  std::vector<float> out(static_cast<size_t>(n));
  for (int s = 0; s < n; ++s) {
    const float* w = w_.data() + size_t(s) * size_t(hidden_);
    double z = b_[size_t(s)];
    for (int j = 0; j < hidden_; ++j) z += double(w[j]) * x[size_t(j)];
    out[size_t(s)] = float(z);
  }
  return out;
}

}  // namespace espetero
