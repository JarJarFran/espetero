// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// A Hugging Face Qwen3.5 / Qwen3.6 or Gemma 4 checkpoint (BF16 safetensors, the names
// transformers writes) seen through the MLX names the loaders and the
// quantizer use, with the conversions mlx-lm's sanitize makes:
//   model.language_model.X -> language_model.model.X, lm_head.weight ->
//   language_model.lm_head.weight; the vision tower and the MTP head
//   (mtp.*, which mlx-lm drops too) are left out (raw() reads them);
//   conv1d [C, 1, K] -> [C, K, 1] (the same bytes);
//   decoder RMSNorm weights (input/post-attention layernorm, final norm,
//   q/k norm) shifted by +1 and rounded to bf16, as mlx-lm does when the
//   checkpoint has an MTP head or unsanitized conv weights;
//   fused experts (mlp.experts.gate_up_proj [E, 2 I, D], down_proj [E, D, I])
//   or per-expert tensors (mlp.experts.N.gate_proj.weight ...) as stacked
//   switch_mlp.{gate,up,down}_proj.weight [E, out, in];
//   Gemma 4 (gemma4_text.sanitize): no norm shift, the audio and vision
//   towers left out, fused experts (layers.N.experts.gate_up_proj /
//   down_proj) as experts.switch_glu.{gate,up,down}_proj.weight.
// Tensors that need new bytes (shifted norms, split or stacked experts) are
// materialized on first use; large ones are kept only while they are among
// the last few asked for (a loader converts each matrix before it asks for
// the next one).
#pragma once

#include <deque>
#include <regex>

#include "tensors.h"

namespace espetero::native {

class HFSource : public WeightSource {
 public:
  explicit HFSource(const std::string& dir) : raw_(dir) {
    for (const std::string& n : raw_.names()) {
      if (n.rfind("model.visual.", 0) == 0 || n.rfind("vision_tower", 0) == 0) continue;
      if (n.find("conv1d.weight") != std::string::npos && raw_.locate(n).shape.back() != 1) unsanitized_conv_ = true;
      if (n.rfind("mtp.", 0) == 0) has_mtp_ = true;
    }
    const json::Value& cfg = raw_.config();
    const json::Value* tc = cfg.get("text_config");
    const json::Value& t = tc && tc->is_object() ? *tc : cfg;
    tied_ = cfg.get_bool("tie_word_embeddings", t.get_bool("tie_word_embeddings", false));
    static const std::regex fused(R"(^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$)");
    static const std::regex single(R"(^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$)");
    static const std::regex g4fused(R"(^model\.language_model\.layers\.(\d+)\.experts\.(gate_up_proj|down_proj)$)");
    for (const std::string& n : raw_.names()) {
      std::smatch mm;
      if (std::regex_match(n, mm, g4fused)) {  // Gemma 4: experts beside the dense MLP
        const std::string p = "language_model.model.layers." + mm[1].str() + ".experts.switch_glu.";
        if (mm[2] == "gate_up_proj") {
          map_[p + "gate_proj.weight"] = {n, Kind::GateHalf};
          map_[p + "up_proj.weight"] = {n, Kind::UpHalf};
        } else {
          map_[p + "down_proj.weight"] = {n, Kind::Plain};
        }
        continue;
      }
      if (n.find("self_attn.rotary_emb") != std::string::npos) continue;  // gemma4_text.sanitize drops these
      if (std::regex_match(n, mm, fused)) {
        const std::string p = "language_model.model.layers." + mm[1].str() + ".mlp.switch_mlp.";
        if (mm[2] == "gate_up_proj") {
          map_[p + "gate_proj.weight"] = {n, Kind::GateHalf};
          map_[p + "up_proj.weight"] = {n, Kind::UpHalf};
        } else {
          map_[p + "down_proj.weight"] = {n, Kind::Plain};
        }
        continue;
      }
      if (std::regex_match(n, mm, single)) {
        const std::string key = "language_model.model.layers." + mm[1].str() + ".mlp.switch_mlp." + mm[3].str() + ".weight";
        auto& v = stacks_[key];
        const size_t e = size_t(std::stoul(mm[2].str()));
        if (v.size() <= e) v.resize(e + 1);
        v[e] = n;
        continue;
      }
      std::string mlx = mlx_name(n);
      if (mlx.empty()) continue;
      Kind k = Kind::Plain;
      if (!ends_with(n, "conv1d.weight") && shift_norms() && is_shifted_norm(mlx) && raw_.locate(n).shape.size() == 1)
        k = Kind::NormShift;
      map_[mlx] = {n, k};
    }
    for (auto& [key, v] : stacks_) {
      for (const std::string& s : v)
        if (s.empty()) fail(dir + ": " + key + " misses an expert");
      map_[key] = {v[0], Kind::Stack};
    }
    if (tied_) map_.erase("language_model.lm_head.weight");
  }

  // The HF layout this source reads: Qwen3.5-family tensors under model.language_model.
  static bool is_hf_dir(const SafetensorsSource& s) {
    return !s.has("language_model.model.embed_tokens.weight") && s.has("model.language_model.embed_tokens.weight");
  }

  const json::Value& config() const { return raw_.config(); }
  const std::string& dir() const { return raw_.dir(); }
  bool tied() const { return tied_; }
  std::string describe() const override { return "hf:" + raw_.dir(); }
  bool has(const std::string& name) const override { return map_.count(name) != 0; }

  // Every MLX name this source serves, sorted.
  std::vector<std::string> names() const {
    std::vector<std::string> v;
    v.reserve(map_.size());
    for (const auto& [k, _] : map_) v.push_back(k);
    std::sort(v.begin(), v.end());
    return v;
  }
  // The source tensor's dtype and MLX shape without materializing it.
  std::pair<std::string, std::vector<int64_t>> info(const std::string& name) const {
    auto it = map_.find(name);
    if (it == map_.end()) fail("missing tensor " + name);
    const auto l = raw_.locate(it->second.src);
    std::vector<int64_t> sh = l.shape;
    switch (it->second.kind) {
      case Kind::GateHalf:
      case Kind::UpHalf: sh[sh.size() - 2] /= 2; break;
      case Kind::Stack: sh.insert(sh.begin(), int64_t(stacks_.at(name).size())); break;
      default: break;
    }
    if (ends_with(name, "conv1d.weight") && sh.size() == 3 && sh[1] == 1) std::swap(sh[1], sh[2]);
    return {l.dtype, sh};
  }

  TensorView tensor(const std::string& name) const override {
    auto it = map_.find(name);
    if (it == map_.end()) fail("missing tensor " + name);
    const Ref& r = it->second;
    const auto l = raw_.locate(r.src);
    TensorView t;
    t.name = r.src;
    t.kind = TensorView::Dense;
    t.dtype = l.dtype;
    if (l.dtype != "BF16" && l.dtype != "F16" && l.dtype != "F32") fail(r.src + ": unsupported dtype " + l.dtype);
    const size_t es = l.dtype == "F32" ? 4 : 2;
    t.shape = info(name).second;
    if (r.kind == Kind::Plain) {
      t.data = l.data;
      t.nbytes = l.nbytes;
      return t;
    }
    std::lock_guard<std::mutex> lk(mu_);
    auto c = cache_.find(name);
    if (c == cache_.end()) {
      std::vector<uint8_t> buf;
      if (r.kind == Kind::NormShift) {
        // mlx-lm: v + 1.0 in the tensor's dtype (bf16: the exact sum rounded)
        const size_t n = l.nbytes / es;
        buf.resize(l.nbytes);
        for (size_t i = 0; i < n; ++i) {
          const float v = dense_at(l.data, l.dtype, i) + 1.0f;
          if (l.dtype == "BF16") reinterpret_cast<uint16_t*>(buf.data())[i] = f32_to_bf16_bits(v);
          else if (l.dtype == "F16") reinterpret_cast<uint16_t*>(buf.data())[i] = f32_to_f16_bits(v);
          else reinterpret_cast<float*>(buf.data())[i] = v;
        }
      } else if (r.kind == Kind::GateHalf || r.kind == Kind::UpHalf) {
        const int64_t E = l.shape[0], R2 = l.shape[1], C = l.shape[2];
        const size_t half = size_t(R2 / 2) * size_t(C) * es;
        buf.resize(size_t(E) * half);
        const size_t skip = r.kind == Kind::UpHalf ? half : 0;
        parallel_for(size_t(E), [&](size_t e) { std::memcpy(buf.data() + e * half, l.data + e * 2 * half + skip, half); });
      } else {  // Stack
        const auto& v = stacks_.at(name);
        const size_t one = l.nbytes;
        buf.resize(one * v.size());
        parallel_for(v.size(), [&](size_t e) {
          const auto le = raw_.locate(v[e]);
          if (le.nbytes != one || le.dtype != l.dtype) fail(v[e] + ": experts differ in shape or dtype");
          std::memcpy(buf.data() + e * one, le.data, one);
        });
      }
      if (buf.size() >= (64u << 20)) {
        recent_.push_back(name);
        while (recent_.size() > 4) {
          cache_.erase(recent_.front());
          recent_.pop_front();
        }
      }
      c = cache_.emplace(name, std::move(buf)).first;
    }
    t.data = c->second.data();
    t.nbytes = c->second.size();
    return t;
  }

  // Drop every materialized tensor (the quantizer, after each matrix).
  void release() const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const std::string& n : recent_) cache_.erase(n);
    recent_.clear();
  }

  // The raw HF tensors (the MTP head is converted from them).
  const SafetensorsSource& raw() const { return raw_; }

 private:
  enum class Kind { Plain, NormShift, GateHalf, UpHalf, Stack };
  struct Ref {
    std::string src;
    Kind kind = Kind::Plain;
  };

  bool shift_norms() const { return has_mtp_ || unsanitized_conv_; }
  static bool is_shifted_norm(const std::string& n) {
    for (const char* s : {".input_layernorm.weight", ".post_attention_layernorm.weight", "model.norm.weight",
                          ".q_norm.weight", ".k_norm.weight"})
      if (ends_with(n, s)) return true;
    return false;
  }
  static std::string mlx_name(const std::string& n) {
    if (n.rfind("model.language_model.", 0) == 0) return "language_model.model." + n.substr(21);
    if (n == "lm_head.weight") return "language_model.lm_head.weight";
    if (n.rfind("language_model.", 0) == 0) return n;
    return "";
  }

  SafetensorsSource raw_;
  bool tied_ = false, has_mtp_ = false, unsanitized_conv_ = false;
  std::map<std::string, Ref> map_;
  std::map<std::string, std::vector<std::string>> stacks_;
  mutable std::mutex mu_;
  mutable std::map<std::string, std::vector<uint8_t>> cache_;
  mutable std::deque<std::string> recent_;
};

}  // namespace espetero::native
