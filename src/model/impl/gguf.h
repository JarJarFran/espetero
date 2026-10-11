// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero
//
// Portions are derived from ggml, Copyright (c) 2023-2026 The ggml authors; MIT License, see THIRD_PARTY_NOTICES.md.

// GGUF checkpoints (llama.cpp): header, metadata, tensor directory, the
// qwen35 text layout and the qwen3vl_merger vision projector (mmproj), and
// exact conversion of ggml block formats into the engine's affine layout:
// every supported block type is an integer code times a per-group scale plus
// a per-group bias (codebook types map their table into 8-bit codes), so the
// same register-direct GEMM runs them without re-quantizing anything.
#pragma once

#include "../../util/model_family.h"
#include "tensors.h"

namespace espetero::native {

enum GGMLType : int {
  GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_1 = 7,
  GGML_Q8_0 = 8, GGML_Q2_K = 10, GGML_Q3_K = 11, GGML_Q4_K = 12, GGML_Q5_K = 13, GGML_Q6_K = 14,
  GGML_IQ4_NL = 20, GGML_IQ4_XS = 23, GGML_BF16 = 30,
};

struct GGMLTypeInfo {
  const char* name;
  int block;        // elements per block
  int bytes;        // bytes per block
  int code_bits;    // engine GPU code width (4, 5, 6 or 8); 0 = dense
  int group;        // engine group size
};

const GGMLTypeInfo* ggml_type_info(int t) {
  static const std::map<int, GGMLTypeInfo> table = {
      {GGML_F32, {"F32", 1, 4, 0, 0}},        {GGML_F16, {"F16", 1, 2, 0, 0}},
      {GGML_BF16, {"BF16", 1, 2, 0, 0}},      {GGML_Q4_0, {"Q4_0", 32, 18, 4, 32}},
      {GGML_Q4_1, {"Q4_1", 32, 20, 4, 32}},   {GGML_Q5_0, {"Q5_0", 32, 22, 5, 32}},
      {GGML_Q5_1, {"Q5_1", 32, 24, 5, 32}},   {GGML_Q8_0, {"Q8_0", 32, 34, 8, 32}},
      {GGML_Q2_K, {"Q2_K", 256, 84, 4, 16}},  {GGML_Q3_K, {"Q3_K", 256, 110, 4, 16}},
      {GGML_Q4_K, {"Q4_K", 256, 144, 4, 32}}, {GGML_Q5_K, {"Q5_K", 256, 176, 5, 32}},
      {GGML_Q6_K, {"Q6_K", 256, 210, 6, 16}}, {GGML_IQ4_NL, {"IQ4_NL", 32, 18, 8, 32}},
      {GGML_IQ4_XS, {"IQ4_XS", 256, 136, 8, 32}},
  };
  auto it = table.find(t);
  return it == table.end() ? nullptr : &it->second;
}

// ---------------------------------------------------------------- file
class GGUFFile {
 public:
  struct Tensor {
    std::string name;
    std::vector<int64_t> dims;  // ggml order: dims[0] is contiguous
    int type = 0;
    size_t offset = 0;          // absolute
    size_t nbytes = 0;
  };

  // Every count, size and offset of the header is checked against the file
  // with overflow-safe arithmetic (as ggml's gguf.cpp does): a crafted file
  // fails to load instead of pointing reads outside the mapping.
  explicit GGUFFile(const std::string& path) : file_(path) {
    const uint8_t* p = file_.at(0);
    const uint8_t* end = p + file_.size;
    auto need = [&](uint64_t n) {
      if (n > uint64_t(end - p)) fail(path + ": truncated GGUF");
    };
    auto u32 = [&]() { need(4); uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; };
    auto u64 = [&]() { need(8); uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; };
    auto str = [&]() {
      const uint64_t n = u64();
      need(n);
      std::string s((const char*)p, n);
      p += n;
      return s;
    };
    if (file_.size < 4 || std::memcmp(p, "GGUF", 4) != 0) fail(path + " is not a GGUF file");
    p += 4;
    version_ = int(u32());
    if (version_ < 2) fail(path + ": GGUF version " + std::to_string(version_) + " is not supported");
    const uint64_t n_tensors = u64();
    const uint64_t n_kv = u64();
    std::function<void(Value&, uint32_t, int)> read_value = [&](Value& v, uint32_t t, int depth) {
      switch (t) {
        case 0: need(1); v.i = *(const uint8_t*)p; p += 1; v.kind = Value::Int; break;
        case 1: need(1); v.i = *(const int8_t*)p; p += 1; v.kind = Value::Int; break;
        case 2: { need(2); uint16_t x; std::memcpy(&x, p, 2); p += 2; v.i = x; v.kind = Value::Int; break; }
        case 3: { need(2); int16_t x; std::memcpy(&x, p, 2); p += 2; v.i = x; v.kind = Value::Int; break; }
        case 4: v.i = u32(); v.kind = Value::Int; break;
        case 5: { need(4); int32_t x; std::memcpy(&x, p, 4); p += 4; v.i = x; v.kind = Value::Int; break; }
        case 6: { need(4); float x; std::memcpy(&x, p, 4); p += 4; v.f = x; v.kind = Value::Float; break; }
        case 7: need(1); v.i = *p != 0; p += 1; v.kind = Value::Int; break;
        case 8: v.s = str(); v.kind = Value::String; break;
        case 10: v.i = int64_t(u64()); v.kind = Value::Int; break;
        case 11: v.i = int64_t(u64()); v.kind = Value::Int; break;
        case 12: { need(8); double x; std::memcpy(&x, p, 8); p += 8; v.f = x; v.kind = Value::Float; break; }
        case 9: {
          if (depth >= 4) fail(path + ": GGUF arrays nested too deeply");
          const uint32_t at = u32();
          const uint64_t n = u64();
          v.kind = Value::Array;
          v.array_len = n;
          // Large arrays (the tokenizer's vocabulary and merges) are skipped.
          const bool keep = n <= 4096;
          for (uint64_t k = 0; k < n; ++k) {
            Value e;
            read_value(e, at, depth + 1);
            if (keep) v.items.push_back(std::move(e));
          }
          break;
        }
        default: fail(path + ": unknown GGUF metadata type " + std::to_string(t));
      }
    };
    for (uint64_t k = 0; k < n_kv; ++k) {
      const std::string key = str();
      const uint32_t t = u32();
      Value v;
      read_value(v, t, 0);
      kv_[key] = std::move(v);
    }
    std::vector<Tensor> infos;
    for (uint64_t k = 0; k < n_tensors; ++k) {
      Tensor t;
      t.name = str();
      const uint32_t nd = u32();
      if (nd > 4) fail(path + ": " + t.name + " has " + std::to_string(nd) + " dimensions (GGUF allows 4)");
      for (uint32_t d = 0; d < nd; ++d) {
        const uint64_t n = u64();
        if (n > uint64_t(INT64_MAX)) fail(path + ": " + t.name + " has a bad dimension");
        t.dims.push_back(int64_t(n));
      }
      t.type = int(u32());
      t.offset = size_t(u64());
      infos.push_back(std::move(t));
    }
    const int64_t align = get_int("general.alignment", 32);
    if (align <= 0 || align > (int64_t(1) << 30) || (align & (align - 1)))
      fail(path + ": GGUF alignment " + std::to_string(align) + " is not a power of two");
    const size_t data0 = align_up(size_t(p - file_.at(0)), size_t(align));
    if (data0 > file_.size && !infos.empty()) fail(path + ": truncated GGUF");
    for (Tensor& t : infos) {
      int64_t n = 1;
      for (auto d : t.dims) {
        if (d > 0 && n > INT64_MAX / d) fail(path + ": " + t.name + " is too large");
        n *= d;
      }
      const GGMLTypeInfo* ti = ggml_type_info(t.type);
      if (ti) {
        if (n % ti->block) fail(path + ": " + t.name + " is not a whole number of blocks");
        if (uint64_t(n / ti->block) > uint64_t(SIZE_MAX) / size_t(ti->bytes)) fail(path + ": " + t.name + " is too large");
        t.nbytes = size_t(n / ti->block) * size_t(ti->bytes);
      }
      if (t.offset > file_.size - data0 || t.nbytes > file_.size - data0 - t.offset)
        fail(path + ": " + t.name + " lies past the end of the file");
      t.offset += data0;
      tensors_[t.name] = std::move(t);
    }
  }

  struct Value {
    enum Kind { None, Int, Float, String, Array } kind = None;
    int64_t i = 0;
    double f = 0;
    std::string s;
    std::vector<Value> items;
    uint64_t array_len = 0;
  };

  bool has_key(const std::string& k) const { return kv_.count(k) != 0; }
  int64_t get_int(const std::string& k, int64_t fallback) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return fallback;
    if (it->second.kind == Value::Int) return it->second.i;
    if (it->second.kind == Value::Float) return to_int(it->second.f);
    return fallback;
  }
  double get_float(const std::string& k, double fallback) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return fallback;
    if (it->second.kind == Value::Float) return it->second.f;
    if (it->second.kind == Value::Int) return double(it->second.i);
    return fallback;
  }
  std::string get_string(const std::string& k, const std::string& fallback) const {
    auto it = kv_.find(k);
    return (it == kv_.end() || it->second.kind != Value::String) ? fallback : it->second.s;
  }
  std::vector<int64_t> get_ints(const std::string& k) const {
    std::vector<int64_t> out;
    auto it = kv_.find(k);
    if (it == kv_.end() || it->second.kind != Value::Array) return out;
    for (const Value& v : it->second.items) out.push_back(v.kind == Value::Float ? to_int(v.f) : v.i);
    return out;
  }
  std::vector<double> get_floats(const std::string& k) const {
    std::vector<double> out;
    auto it = kv_.find(k);
    if (it == kv_.end() || it->second.kind != Value::Array) return out;
    for (const Value& v : it->second.items) out.push_back(v.kind == Value::Float ? v.f : double(v.i));
    return out;
  }

  const Tensor* find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
  }
  const uint8_t* data(const Tensor& t) const { return file_.at(t.offset); }
  const std::string& path() const { return file_.path; }
  const std::unordered_map<std::string, Tensor>& tensors() const { return tensors_; }

 private:
  // A float metadata value as an integer: truncated, clamped to int64 (NaN: 0).
  static int64_t to_int(double f) {
    if (f >= -9223372036854775808.0 && f < 9223372036854775808.0) return int64_t(f);
    return f > 0 ? INT64_MAX : f < 0 ? INT64_MIN : 0;
  }
  MappedFile file_;
  int version_ = 0;
  std::unordered_map<std::string, Value> kv_;
  std::unordered_map<std::string, Tensor> tensors_;
};

// A GGUF tensor as a view: ggml dims [ne0, ne1, ...] become row-major
// [..., ne1, ne0] (matrices [out, in]).
TensorView gguf_view(const GGUFFile& f, const GGUFFile::Tensor& t, const std::string& canonical) {
  TensorView v;
  v.name = t.name + " (" + canonical + ")";
  // GGUF dims run innermost first. Matrices are [rows = out, cols = in];
  // stacked matrices (the experts' [E, out, in]) become one [E * out, in]
  // matrix, expert-major like the MLX tensors. 4-D tensors (the mmproj patch
  // kernels) keep their full reversed shape.
  if (t.dims.size() == 3) {
    v.shape = {t.dims[1] * t.dims[2], t.dims[0]};
  } else {
    for (auto it = t.dims.rbegin(); it != t.dims.rend(); ++it) v.shape.push_back(*it);
  }
  v.data = f.data(t);
  v.nbytes = t.nbytes;
  const GGMLTypeInfo* ti = ggml_type_info(t.type);
  if (!ti) fail(t.name + ": unsupported GGUF tensor type " + std::to_string(t.type));
  if (ti->code_bits == 0) {
    v.kind = TensorView::Dense;
    v.dtype = ti->name;
  } else {
    v.kind = TensorView::GGUFQuant;
    v.gguf_type = t.type;
    v.bits = ti->code_bits;
    v.group = ti->group;
  }
  return v;
}

// ---------------------------------------------------------------- text model
// Canonical (MLX) names of the Qwen3.5 decoder -> GGUF qwen35 names.
class GGUFSource : public WeightSource {
 public:
  explicit GGUFSource(const std::string& path) : f_(path) {
    const std::string arch = f_.get_string("general.architecture", "");
    if (!family::gguf_arch_supported(arch)) fail(family::unsupported_gguf(path, arch));
    gemma_ = arch == "gemma4";
    // The multi-token-prediction (nextn) layer follows the decoder blocks.
    const int64_t nextn = f_.get_int(arch + ".nextn_predict_layers", 0);
    if (nextn > 0) mtp_block_ = int(f_.get_int(arch + ".block_count", 0) - nextn);
  }
  const GGUFFile& file() const { return f_; }
  std::string describe() const override { return "gguf:" + f_.path(); }

  bool has(const std::string& name) const override {
    std::string g;
    return map(name, g) && f_.find(g) != nullptr;
  }

  TensorView tensor(const std::string& name) const override {
    std::string g;
    if (!map(name, g)) fail("no GGUF name for " + name);
    const GGUFFile::Tensor* t = f_.find(g);
    if (!t) fail("missing GGUF tensor " + g + " (" + name + ")");
    if (ends_with(name, ".A_log")) {
      // ssm_a holds -exp(A_log); the engine consumes A_log.
      TensorView v = gguf_view(f_, *t, name);
      if (v.kind != TensorView::Dense) fail(g + ": expected a dense tensor");
      auto& owned = owned_[name];
      owned.resize(size_t(v.numel()));
      for (size_t i = 0; i < owned.size(); ++i) owned[i] = float(std::log(-double(dense_at(v.data, v.dtype, i))));
      v.dtype = "F32";
      v.data = (const uint8_t*)owned.data();
      v.nbytes = owned.size() * 4;
      return v;
    }
    return gguf_view(f_, *t, name);
  }

 private:
  bool map(const std::string& name, std::string& out) const {
    static const std::string L = "language_model.model.layers.";
    static const std::string MTP = "language_model.mtp.";
    // Gemma 4 (llama.cpp's gemma4 names): the feed-forward norms, the
    // layer scalar, the router (projection, input scale, per-expert scales)
    // and the experts (gate and up fused as [E, 2 Ie, D], gate rows first).
    static const std::vector<std::pair<std::string, std::string>> gemma = {
        {".pre_feedforward_layernorm.weight", ".ffn_norm.weight"},
        {".post_feedforward_layernorm.weight", ".post_ffw_norm.weight"},
        {".post_feedforward_layernorm_1.weight", ".post_ffw_norm_1.weight"},
        {".post_feedforward_layernorm_2.weight", ".post_ffw_norm_2.weight"},
        {".pre_feedforward_layernorm_2.weight", ".pre_ffw_norm_2.weight"},
        {".layer_scalar", ".layer_output_scale.weight"},
        {".router.proj.weight", ".ffn_gate_inp.weight"},
        {".router.scale", ".ffn_gate_inp.scale"},
        {".router.per_expert_scale", ".ffn_down_exps.scale"},
        {".experts.switch_glu.gate_up_proj.weight", ".ffn_gate_up_exps.weight"},
        {".experts.switch_glu.gate_proj.weight", ".ffn_gate_exps.weight"},
        {".experts.switch_glu.up_proj.weight", ".ffn_up_exps.weight"},
        {".experts.switch_glu.down_proj.weight", ".ffn_down_exps.weight"},
    };
    if (gemma_ && name.compare(0, L.size(), L) == 0) {
      const size_t dot = name.find('.', L.size());
      if (dot != std::string::npos)
        for (const auto& [mlx, gg] : gemma)
          if (name.substr(dot) == mlx) return out = "blk." + name.substr(L.size(), dot - L.size()) + gg, true;
    }
    static const std::vector<std::pair<std::string, std::string>> suffix = {
        {".input_layernorm.weight", ".attn_norm.weight"},
        {".post_attention_layernorm.weight", ".post_attention_norm.weight"},
        {".mlp.gate_proj.weight", ".ffn_gate.weight"},
        {".mlp.up_proj.weight", ".ffn_up.weight"},
        {".mlp.down_proj.weight", ".ffn_down.weight"},
        {".linear_attn.in_proj_qkv.weight", ".attn_qkv.weight"},
        {".linear_attn.in_proj_z.weight", ".attn_gate.weight"},
        {".linear_attn.in_proj_a.weight", ".ssm_alpha.weight"},
        {".linear_attn.in_proj_b.weight", ".ssm_beta.weight"},
        {".linear_attn.out_proj.weight", ".ssm_out.weight"},
        {".linear_attn.conv1d.weight", ".ssm_conv1d.weight"},
        {".linear_attn.A_log", ".ssm_a"},
        {".linear_attn.dt_bias", ".ssm_dt.bias"},
        {".linear_attn.norm.weight", ".ssm_norm.weight"},
        {".self_attn.q_proj.weight", ".attn_q.weight"},
        {".self_attn.k_proj.weight", ".attn_k.weight"},
        {".self_attn.v_proj.weight", ".attn_v.weight"},
        {".self_attn.o_proj.weight", ".attn_output.weight"},
        {".self_attn.q_norm.weight", ".attn_q_norm.weight"},
        {".self_attn.k_norm.weight", ".attn_k_norm.weight"},
        // qwen35moe: router, routed experts (3-D, [E, out, in]), shared expert and its gate.
        {".mlp.gate.weight", ".ffn_gate_inp.weight"},
        {".mlp.switch_mlp.gate_proj.weight", ".ffn_gate_exps.weight"},
        {".mlp.switch_mlp.up_proj.weight", ".ffn_up_exps.weight"},
        {".mlp.switch_mlp.down_proj.weight", ".ffn_down_exps.weight"},
        {".mlp.shared_expert.gate_proj.weight", ".ffn_gate_shexp.weight"},
        {".mlp.shared_expert.up_proj.weight", ".ffn_up_shexp.weight"},
        {".mlp.shared_expert.down_proj.weight", ".ffn_down_shexp.weight"},
        {".mlp.shared_expert_gate.weight", ".ffn_gate_inp_shexp.weight"},
    };
    if (name == "language_model.model.embed_tokens.weight") return out = "token_embd.weight", true;
    if (name == "language_model.lm_head.weight") return out = "output.weight", true;
    if (name == "language_model.model.norm.weight") return out = "output_norm.weight", true;
    if (name.compare(0, MTP.size(), MTP) == 0) {
      // MTP head (HF mtp.*): fc/norms are nextn.*, its layer is block `mtp_block_`.
      if (mtp_block_ < 0) return false;
      const std::string blk = "blk." + std::to_string(mtp_block_);
      static const std::vector<std::pair<std::string, std::string>> heads = {
          {"fc.weight", ".nextn.eh_proj.weight"},
          {"pre_fc_norm_embedding.weight", ".nextn.enorm.weight"},
          {"pre_fc_norm_hidden.weight", ".nextn.hnorm.weight"},
          {"norm.weight", ".nextn.shared_head_norm.weight"},
      };
      const std::string rest = name.substr(MTP.size());
      for (const auto& [hf, gg] : heads)
        if (rest == hf) return out = blk + gg, true;
      static const std::string layer0 = "layers.0";
      if (rest.compare(0, layer0.size(), layer0) != 0) return false;
      for (const auto& [mlx, gg] : suffix)
        if (rest.substr(layer0.size()) == mlx) return out = blk + gg, true;
      return false;
    }
    if (name.compare(0, L.size(), L) != 0) return false;
    const size_t dot = name.find('.', L.size());
    if (dot == std::string::npos) return false;
    const std::string layer = name.substr(L.size(), dot - L.size());
    const std::string rest = name.substr(dot);
    for (const auto& [mlx, gg] : suffix)
      if (rest == mlx) return out = "blk." + layer + gg, true;
    return false;
  }

  GGUFFile f_;
  int mtp_block_ = -1;
  bool gemma_ = false;
  mutable std::unordered_map<std::string, std::vector<float>> owned_;
};

// ---------------------------------------------------------------- mmproj
// Canonical vision names (mlx-vlm's vision_tower.*) -> clip mmproj names.
class GGUFVisionSource : public WeightSource {
 public:
  explicit GGUFVisionSource(const std::string& path) : f_(path) {
    const std::string proj = f_.get_string("clip.projector_type", "");
    if (proj != "qwen3vl_merger")
      fail(path + ": mmproj projector '" + proj + "' is not qwen3vl_merger (Qwen3-VL / Qwen3.5 vision)");
  }
  const GGUFFile& file() const { return f_; }
  std::string describe() const override { return "mmproj:" + f_.path(); }

  bool has(const std::string& name) const override {
    if (name == "vision_tower.patch_embed.proj.weight")
      return f_.find("v.patch_embd.weight") && f_.find("v.patch_embd.weight.1");
    std::string g;
    return map(name, g) && f_.find(g) != nullptr;
  }

  TensorView tensor(const std::string& name) const override {
    if (name == "vision_tower.patch_embed.proj.weight") return patch_weight();
    std::string g;
    if (!map(name, g)) fail("no mmproj name for " + name);
    const GGUFFile::Tensor* t = f_.find(g);
    if (!t) fail("missing mmproj tensor " + g + " (" + name + ")");
    return gguf_view(f_, *t, name);
  }

 private:
  // The Conv3d kernel is stored as two 2D slices (temporal 0 and 1), each
  // [out, C, P, P]; rebuild MLX's [out, T, P, P, C] in fp32.
  TensorView patch_weight() const {
    const GGUFFile::Tensor* t0 = f_.find("v.patch_embd.weight");
    const GGUFFile::Tensor* t1 = f_.find("v.patch_embd.weight.1");
    if (!t0 || !t1) fail("mmproj: patch embedding slices missing");
    TensorView a = gguf_view(f_, *t0, "patch0"), b = gguf_view(f_, *t1, "patch1");
    if (a.kind != TensorView::Dense || a.shape.size() != 4 || a.shape != b.shape)
      fail("mmproj: unexpected patch embedding layout");
    const int64_t O = a.shape[0], C = a.shape[1], P = a.shape[2];
    auto& owned = owned_["patch"];
    owned.assign(size_t(O * 2 * P * P * C), 0.0f);
    for (int64_t o = 0; o < O; ++o)
      for (int t = 0; t < 2; ++t) {
        const TensorView& src = t ? b : a;
        for (int64_t c = 0; c < C; ++c)
          for (int64_t y = 0; y < P; ++y)
            for (int64_t x = 0; x < P; ++x)
              owned[size_t((((o * 2 + t) * P + y) * P + x) * C + c)] =
                  dense_at(src.data, src.dtype, size_t(((o * C + c) * P + y) * P + x));
      }
    TensorView v;
    v.name = "v.patch_embd.weight{,.1}";
    v.kind = TensorView::Dense;
    v.dtype = "F32";
    v.shape = {O, 2, P, P, C};
    v.data = (const uint8_t*)owned.data();
    v.nbytes = owned.size() * 4;
    return v;
  }

  static bool map(const std::string& name, std::string& out) {
    static const std::string B = "vision_tower.blocks.";
    static const std::vector<std::pair<std::string, std::string>> suffix = {
        {".attn.qkv.weight", ".attn_qkv.weight"},   {".attn.qkv.bias", ".attn_qkv.bias"},
        {".attn.proj.weight", ".attn_out.weight"},  {".attn.proj.bias", ".attn_out.bias"},
        {".mlp.linear_fc1.weight", ".ffn_up.weight"}, {".mlp.linear_fc1.bias", ".ffn_up.bias"},
        {".mlp.linear_fc2.weight", ".ffn_down.weight"}, {".mlp.linear_fc2.bias", ".ffn_down.bias"},
        {".norm1.weight", ".ln1.weight"}, {".norm1.bias", ".ln1.bias"},
        {".norm2.weight", ".ln2.weight"}, {".norm2.bias", ".ln2.bias"},
    };
    static const std::map<std::string, std::string> fixed = {
        {"vision_tower.merger.norm.weight", "v.post_ln.weight"},
        {"vision_tower.merger.norm.bias", "v.post_ln.bias"},
        {"vision_tower.merger.linear_fc1.weight", "mm.0.weight"},
        {"vision_tower.merger.linear_fc1.bias", "mm.0.bias"},
        {"vision_tower.merger.linear_fc2.weight", "mm.2.weight"},
        {"vision_tower.merger.linear_fc2.bias", "mm.2.bias"},
        {"vision_tower.pos_embed.weight", "v.position_embd.weight"},
        {"vision_tower.patch_embed.proj.bias", "v.patch_embd.bias"},
    };
    auto it = fixed.find(name);
    if (it != fixed.end()) return out = it->second, true;
    if (name.compare(0, B.size(), B) != 0) return false;
    const size_t dot = name.find('.', B.size());
    if (dot == std::string::npos) return false;
    const std::string blk = name.substr(B.size(), dot - B.size());
    const std::string rest = name.substr(dot);
    for (const auto& [mlx, gg] : suffix)
      if (rest == mlx) return out = "v.blk." + blk + gg, true;
    return false;
  }

  GGUFFile f_;
  mutable std::unordered_map<std::string, std::vector<float>> owned_;
};

// ---------------------------------------------------------------- blocks
// Conversion of one row of ggml blocks into engine codes (nibbles packed
// eight per little-endian word, or bytes) and per-group half scale/bias, with
// w = scale * code + bias. The formulas follow ggml's dequantize_row_* so the
// weights are the file's weights (scales rounded to fp16 once).

const int8_t kIQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline float h2f(const uint8_t* p) {
  uint16_t v;
  std::memcpy(&v, p, 2);
  return f16_to_f32(v);
}

inline void k4_scale_min(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

struct RowSink {
  int bits;              // 4: nibbles (as words); anything else: one byte per code
  uint8_t* codes;        // row start
  uint16_t* scales;      // row start, cols/group halves
  uint16_t* biases;
  // Native Q6_K (when set): int8 sub-scale per 16 codes and the block's fp16
  // super-scale per 256, instead of expanded half scales and biases.
  int8_t* q6k_sub = nullptr;
  uint16_t* q6k_super = nullptr;
  // Native Q4_K/Q5_K (when set): 6-bit scale and min per 32 codes as bytes
  // (scale, min) and the block's fp16 d and dmin per 256.
  uint8_t* kq_sm = nullptr;
  uint16_t* kq_dd = nullptr;
  void code(int64_t i, uint32_t c) {
    if (bits != 4) {
      codes[i] = uint8_t(c);
    } else {
      uint32_t* w = (uint32_t*)codes + (i >> 3);
      *w |= (c & 0xFu) << (4 * (i & 7));
    }
  }
  // Native Q8_0 leaves `biases` null: the bias is -128 * scale.
  void group(int64_t g, float s, float b) {
    scales[g] = f32_to_f16_bits(s);
    if (biases) biases[g] = f32_to_f16_bits(b);
  }
};

void convert_row(int type, const uint8_t* src, int64_t cols, RowSink& out) {
  switch (type) {
    case GGML_Q4_0:
    case GGML_Q4_1:
      for (int64_t b = 0; b < cols / 32; ++b) {
        const uint8_t* x = src + b * (type == GGML_Q4_0 ? 18 : 20);
        const float d = h2f(x);
        const float m = type == GGML_Q4_1 ? h2f(x + 2) : -8.0f * d;
        const uint8_t* qs = x + (type == GGML_Q4_0 ? 2 : 4);
        for (int j = 0; j < 16; ++j) {
          out.code(b * 32 + j, qs[j] & 0xF);
          out.code(b * 32 + j + 16, qs[j] >> 4);
        }
        out.group(b, d, m);
      }
      break;
    case GGML_Q5_0:
    case GGML_Q5_1:
      for (int64_t b = 0; b < cols / 32; ++b) {
        const uint8_t* x = src + b * (type == GGML_Q5_0 ? 22 : 24);
        const float d = h2f(x);
        const float m = type == GGML_Q5_1 ? h2f(x + 2) : -16.0f * d;
        uint32_t qh;
        std::memcpy(&qh, x + (type == GGML_Q5_0 ? 2 : 4), 4);
        const uint8_t* qs = x + (type == GGML_Q5_0 ? 6 : 8);
        for (int j = 0; j < 16; ++j) {
          const uint32_t h0 = ((qh >> (j + 0)) << 4) & 0x10;
          const uint32_t h1 = (qh >> (j + 12)) & 0x10;
          out.code(b * 32 + j, (qs[j] & 0xF) | h0);
          out.code(b * 32 + j + 16, (qs[j] >> 4) | h1);
        }
        out.group(b, d, m);
      }
      break;
    case GGML_Q8_0:
      for (int64_t b = 0; b < cols / 32; ++b) {
        const uint8_t* x = src + b * 34;
        const float d = h2f(x);
        for (int j = 0; j < 32; ++j) out.code(b * 32 + j, uint32_t(int32_t(int8_t(x[2 + j])) + 128));
        out.group(b, d, -128.0f * d);
      }
      break;
    case GGML_IQ4_NL:
      for (int64_t b = 0; b < cols / 32; ++b) {
        const uint8_t* x = src + b * 18;
        const float d = h2f(x);
        for (int j = 0; j < 16; ++j) {
          out.code(b * 32 + j, uint32_t(kIQ4NL[x[2 + j] & 0xF] + 128));
          out.code(b * 32 + j + 16, uint32_t(kIQ4NL[x[2 + j] >> 4] + 128));
        }
        out.group(b, d, -128.0f * d);
      }
      break;
    case GGML_IQ4_XS:
      for (int64_t b = 0; b < cols / 256; ++b) {
        const uint8_t* x = src + b * 136;
        const float d = h2f(x);
        uint16_t sh;
        std::memcpy(&sh, x + 2, 2);
        const uint8_t* sl = x + 4;
        const uint8_t* qs = x + 8;
        for (int ib = 0; ib < 8; ++ib) {
          const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
          const float dl = d * float(ls - 32);
          const int64_t e0 = b * 256 + ib * 32;
          for (int j = 0; j < 16; ++j) {
            out.code(e0 + j, uint32_t(kIQ4NL[qs[ib * 16 + j] & 0xF] + 128));
            out.code(e0 + j + 16, uint32_t(kIQ4NL[qs[ib * 16 + j] >> 4] + 128));
          }
          out.group(b * 8 + ib, dl, -128.0f * dl);
        }
      }
      break;
    case GGML_Q2_K:
      for (int64_t b = 0; b < cols / 256; ++b) {
        const uint8_t* x = src + b * 84;
        const uint8_t* sc = x;
        const uint8_t* q = x + 16;
        const float d = h2f(x + 80), dmin = h2f(x + 82);
        int64_t e = b * 256;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
          int shift = 0;
          for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
              const uint8_t s = sc[is];
              out.group(b * 16 + is, d * float(s & 0xF), -dmin * float(s >> 4));
              ++is;
              for (int l = 0; l < 16; ++l) out.code(e++, (q[l + 16 * half] >> shift) & 3);
            }
            shift += 2;
          }
          q += 32;
        }
      }
      break;
    case GGML_Q3_K:
      for (int64_t b = 0; b < cols / 256; ++b) {
        const uint8_t* x = src + b * 110;
        const uint8_t* hm = x;
        const uint8_t* q = x + 32;
        const float d_all = h2f(x + 108);
        uint32_t aux[4];
        std::memcpy(aux, x + 96, 12);
        const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = (const int8_t*)aux;
        int64_t e = b * 256;
        int is = 0;
        uint8_t m = 1;
        for (int n = 0; n < 256; n += 128) {
          int shift = 0;
          for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
              const float dl = d_all * float(scales[is] - 32);
              out.group(b * 16 + is, dl, -4.0f * dl);
              ++is;
              for (int l = 0; l < 16; ++l) {
                const int ll = l + 16 * half;
                out.code(e++, uint32_t(((q[ll] >> shift) & 3) + ((hm[ll] & m) ? 4 : 0)));
              }
            }
            shift += 2;
            m <<= 1;
          }
          q += 32;
        }
      }
      break;
    case GGML_Q4_K:
    case GGML_Q5_K:
      for (int64_t b = 0; b < cols / 256; ++b) {
        const bool five = type == GGML_Q5_K;
        const uint8_t* x = src + b * (five ? 176 : 144);
        const float d = h2f(x), dmin = h2f(x + 2);
        const uint8_t* sc = x + 4;
        const uint8_t* qh = five ? x + 16 : nullptr;
        const uint8_t* ql = x + (five ? 48 : 16);
        uint8_t u1 = 1, u2 = 2;
        if (out.kq_dd) std::memcpy(out.kq_dd + 2 * b, x, 4);  // d, dmin
        for (int j = 0; j < 4; ++j) {
          uint8_t s1, m1, s2, m2;
          k4_scale_min(2 * j, sc, s1, m1);
          k4_scale_min(2 * j + 1, sc, s2, m2);
          if (out.kq_sm) {
            uint8_t* sm = out.kq_sm + 2 * (b * 8 + 2 * j);
            sm[0] = s1, sm[1] = m1, sm[2] = s2, sm[3] = m2;
          } else {
            out.group(b * 8 + 2 * j, d * float(s1), -dmin * float(m1));
            out.group(b * 8 + 2 * j + 1, d * float(s2), -dmin * float(m2));
          }
          const int64_t e0 = b * 256 + j * 64;
          for (int l = 0; l < 32; ++l) {
            uint32_t lo = ql[l] & 0xF, hi = ql[l] >> 4;
            if (five) {
              lo += (qh[l] & u1) ? 16 : 0;
              hi += (qh[l] & u2) ? 16 : 0;
            }
            out.code(e0 + l, lo);
            out.code(e0 + 32 + l, hi);
          }
          ql += 32;
          u1 <<= 2;
          u2 <<= 2;
        }
      }
      break;
    case GGML_Q6_K:
      for (int64_t b = 0; b < cols / 256; ++b) {
        const uint8_t* x = src + b * 210;
        const uint8_t* ql = x;
        const uint8_t* qh = x + 128;
        const int8_t* sc = (const int8_t*)(x + 192);
        const float d = h2f(x + 208);
        if (out.q6k_sub) {
          for (int g = 0; g < 16; ++g) out.q6k_sub[b * 16 + g] = sc[g];
          std::memcpy(out.q6k_super + b, x + 208, 2);
        } else {
          for (int g = 0; g < 16; ++g) {
            const float s = d * float(sc[g]);
            out.group(b * 16 + g, s, -32.0f * s);
          }
        }
        for (int n = 0; n < 2; ++n) {
          const int64_t e0 = b * 256 + n * 128;
          for (int l = 0; l < 32; ++l) {
            out.code(e0 + l, (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4));
            out.code(e0 + l + 32, (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4));
            out.code(e0 + l + 64, (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4));
            out.code(e0 + l + 96, (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4));
          }
          ql += 64;
          qh += 32;
        }
      }
      break;
    default:
      fail(std::string("unsupported GGUF block type ") + std::to_string(type));
  }
}

}  // namespace espetero::native
