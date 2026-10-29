// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// MiniCPM-OCR checkpoints (openbmb/MiniCPM-OCR-1.0, model_type minicpm_ocr)
// as the Hugging Face release has them: a Qwen3.5 text decoder
// (model.language_model.*, bf16) and a SigLIP tower with its merger
// (model.vision_tower.*, model.merger.*). The decoder loads through the
// Qwen3.5 path (load_decoder) from a view that gives its tensors the MLX
// names and the values mlx-lm's qwen3_5 sanitize writes: the RMSNorm
// weights of the (1 + w) norms (input, post-attention, final, q and k
// norms; not the gated DeltaNet norm) shifted by one, conv1d taps
// [C][1][K] read as [C][K]. The tower runs on the SigLIP structures
// (vision.h, c.minicpm). Semantics of transformers' models/minicpm_ocr and
// qwen3_5 (Apache-2.0, see THIRD_PARTY_NOTICES.md).
#pragma once

namespace espetero::native {

inline bool is_minicpm_type(const std::string& t) { return t == "minicpm_ocr"; }

// The decoder's tensors under their MLX names (language_model.model.*).
class MinicpmTextSource : public WeightSource {
 public:
  explicit MinicpmTextSource(const SafetensorsSource& src) : src_(src) {}
  std::string describe() const override { return src_.describe(); }
  bool has(const std::string& name) const override {
    std::string hf;
    return map(name, hf) && src_.has(hf);
  }
  TensorView tensor(const std::string& name) const override {
    std::string hf;
    if (!map(name, hf)) fail("missing tensor " + name);
    TensorView t = src_.tensor(hf);
    if (shifted(name)) {
      if (t.kind != TensorView::Dense) fail(hf + ": expected a dense norm weight");
      auto it = shifted_.find(name);
      if (it == shifted_.end()) {
        std::vector<float> w(size_t(t.numel()));
        for (size_t i = 0; i < w.size(); ++i) w[i] = dense_at(t.data, t.dtype, i) + 1.0f;
        it = shifted_.emplace(name, std::move(w)).first;
      }
      t.dtype = "F32";
      t.data = reinterpret_cast<const uint8_t*>(it->second.data());
      t.nbytes = it->second.size() * 4;
    }
    return t;
  }

 private:
  static bool map(const std::string& name, std::string& out) {
    static const std::string mlx = "language_model.model.";
    if (name.compare(0, mlx.size(), mlx) != 0) return false;
    out = "model.language_model." + name.substr(mlx.size());
    return true;
  }
  static bool shifted(const std::string& name) {
    for (const char* s : {".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight", ".k_norm.weight"})
      if (ends_with(name, s)) return true;
    return name == "language_model.model.norm.weight";
  }
  const SafetensorsSource& src_;
  mutable std::map<std::string, std::vector<float>> shifted_;
};

// The SigLIP tower: Conv2d patch embedding [D][C][P][P] read for the
// processor's (py, px, channel) patch rows, q, k, v fused, layer_norm1 /
// layer_norm2 as norm1 / norm2, post_layernorm as the merger's per-patch
// norm, the merger MLP's pre_norm, linear_1 and linear_2; no rotary.
inline void load_minicpm_vision(Model& m, const WeightSource& src, VisionConfig c) {
  c.validate(m.cfg.D);
  Arena tmp(m.dev);
  const std::string v = "model.vision_tower.", mg = "model.merger.mlp.0.";
  {
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    const TensorView pw = src.tensor(v + "embeddings.patch_embedding.weight");
    if (pw.shape.size() != 4 || pw.shape[1] != c.in_ch || pw.shape[2] != c.patch || pw.shape[3] != c.patch)
      fail(pw.name + ": expected a Conv2d kernel [D][C][P][P]");
    std::vector<int> perm(size_t(c.patch_dim()));
    for (int y = 0; y < c.patch; ++y)
      for (int x = 0; x < c.patch; ++x)
        for (int ch = 0; ch < c.in_ch; ++ch) perm[size_t((y * c.patch + x) * c.in_ch + ch)] = (ch * c.patch + y) * c.patch + x;
    m.vis.patch = load_vdense(m, e, tmp, src, v + "embeddings.patch_embedding", c.D, c.patch_dim(), &perm);
    m.vis.m1 = load_vdense(m, e, tmp, src, mg + "linear_1", c.proj, 4 * c.D);
    m.vis.m2 = load_vdense(m, e, tmp, src, mg + "linear_2", c.out, c.proj);
    [e endEncoding];
    commit_wait(cb);
  }
  m.vis.pos = vec(m, src, v + "embeddings.position_embedding.weight", size_t(c.n_pos) * c.D);
  m.vis.mnw = vec(m, src, v + "post_layernorm.weight", c.D);
  m.vis.mnb = vec(m, src, v + "post_layernorm.bias", c.D);
  m.vis.pnw = vec(m, src, mg + "pre_norm.weight", size_t(4 * c.D));
  m.vis.pnb = vec(m, src, mg + "pre_norm.bias", size_t(4 * c.D));
  for (int i = 0; i < c.depth; ++i) {
    Arena btmp(m.dev);
    const std::string b = v + "encoder.layers." + std::to_string(i) + ".";
    VBlock B;
    B.n1w = vec(m, src, b + "layer_norm1.weight", c.D);
    B.n1b = vec(m, src, b + "layer_norm1.bias", c.D);
    B.n2w = vec(m, src, b + "layer_norm2.weight", c.D);
    B.n2b = vec(m, src, b + "layer_norm2.bias", c.D);
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    B.qkv = load_vstack(m, e, btmp, src, {b + "self_attn.q_proj", b + "self_attn.k_proj", b + "self_attn.v_proj"}, c.D, c.D);
    B.proj = load_vdense(m, e, btmp, src, b + "self_attn.out_proj", c.D, c.D);
    B.fc1 = load_vdense(m, e, btmp, src, b + "mlp.fc1", c.I, c.D);
    B.fc2 = load_vdense(m, e, btmp, src, b + "mlp.fc2", c.D, c.I);
    [e endEncoding];
    commit_wait(cb);
    m.vis.blocks.push_back(B);
  }
  m.vis.inv_freq = m.arena->alloc(sizeof(float) * size_t(c.hd / 4));
  std::memset(ptr(m.vis.inv_freq), 0, sizeof(float) * size_t(c.hd / 4));
  c.present = true;
  m.vis.c = c;
}

inline void load_minicpm(Model& m, const SafetensorsSource& src, bool want_vision) {
  if (!src.has("model.language_model.embed_tokens.weight"))
    fail("expected a MiniCPM-OCR checkpoint (model.language_model.* tensors)");
  const MinicpmTextSource text(src);
  load_decoder(m, text);
  if (want_vision && src.has("model.vision_tower.embeddings.patch_embedding.weight")) {
    const VisionConfig vc = vision_from_minicpm(src.config());
    if (vc.present) load_minicpm_vision(m, src, vc);
  }
}

}  // namespace espetero::native
