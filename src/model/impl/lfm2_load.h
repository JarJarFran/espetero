// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// LFM2 / LFM2-VL checkpoints (LiquidAI LFM2.5-VL and its finetunes, such as
// the d1 decision models): the decoder of an lfm2 or lfm2_vl directory and,
// for lfm2_vl, the SigLIP2 NaFlex tower with its projector. Tensor names
// as tools/lfm2_convert.py writes them (language_model.model.*,
// vision_tower.*, multi_modal_projector.*) or as the Hugging Face release
// has them (model.language_model.*, model.vision_tower.vision_model.*,
// model.multi_modal_projector.*); matrices bf16 (served as fp16) or MLX
// affine-quantized. The forward pass is in lfm2.h; the semantics follow
// transformers' models/lfm2, lfm2_vl and siglip2 (Apache-2.0, see
// THIRD_PARTY_NOTICES.md).
#pragma once

namespace espetero::native {

// Name prefixes of a checkpoint: the decoder's, the vision tower's, the projector's.
struct Lfm2Names {
  std::string lm, vt, mm;
};

inline Lfm2Names lfm2_names(const WeightSource& src) {
  if (src.has("language_model.model.embed_tokens.weight")) return {"language_model.model.", "vision_tower.", "multi_modal_projector."};
  if (src.has("model.language_model.embed_tokens.weight"))
    return {"model.language_model.", "model.vision_tower.vision_model.", "model.multi_modal_projector."};
  if (src.has("model.embed_tokens.weight")) return {"model.", "", ""};  // a text-only lfm2 checkpoint
  fail("expected an LFM2 checkpoint (language_model.model.* or model.language_model.* tensors)");
}

// One decoder layer: operator_norm / ffn_norm are the Qwen layout's input and
// post-attention norms; conv layers keep the taps tap-major ([3][D]).
inline Layer load_lfm2_layer(Model& m, const WeightSource& src, const std::string& p, bool conv, int index_in_kind) {
  const Config& c = m.cfg;
  Layer L;
  L.linear = conv;
  L.index_in_kind = index_in_kind;
  L.in_norm = vec(m, src, p + "operator_norm.weight", c.D);
  L.post_norm = vec(m, src, p + "ffn_norm.weight", c.D);
  Arena tmp(m.dev);
  const std::string f = p + "feed_forward.";
  const int I = int(src.tensor(f + "w1.weight").rows());
  if (I != c.I) fail(f + "w1.weight: " + std::to_string(I) + " rows, the configuration says " + std::to_string(c.I));
  L.I = I;
  HostMat gate = host_mat(m, tmp, src, f + "w1.weight", I, c.D);
  HostMat up = host_mat(m, tmp, src, f + "w3.weight", I, c.D);
  HostMat down = host_mat(m, tmp, src, f + "w2.weight", c.D, I);
  std::vector<HostMat> parts;
  HostMat out;
  if (conv) {
    const std::string a = p + "conv.";
    parts = {host_mat(m, tmp, src, a + "in_proj.weight", 3 * c.D, c.D)};
    out = host_mat(m, tmp, src, a + "out_proj.weight", c.D, c.D);
    const std::vector<float> w = src.floats(a + "conv.weight", size_t(c.D) * size_t(c.conv_k));  // [D][1][taps]
    std::vector<float> t(w.size());
    for (int ch = 0; ch < c.D; ++ch)
      for (int k = 0; k < c.conv_k; ++k) t[size_t(k) * size_t(c.D) + size_t(ch)] = w[size_t(ch) * size_t(c.conv_k) + size_t(k)];
    L.conv_w = m.arena->copy(t.data(), t.size() * 4);
  } else {
    const std::string a = p + "self_attn.";
    parts = {host_mat(m, tmp, src, a + "q_proj.weight", c.H * c.HD, c.D),
             host_mat(m, tmp, src, a + "k_proj.weight", c.kv_dim(), c.D),
             host_mat(m, tmp, src, a + "v_proj.weight", c.kv_dim(), c.D)};
    out = host_mat(m, tmp, src, a + "out_proj.weight", c.D, c.H * c.HD);
    L.q_norm = vec(m, src, a + "q_layernorm.weight", c.HD);
    L.k_norm = vec(m, src, a + "k_layernorm.weight", c.HD);
  }
  const bool same_layout = gate.dense || up.dense ||
                           (gate.q.bits == up.q.bits && gate.q.src_bits == up.q.src_bits &&
                            gate.q.scale_mode == up.q.scale_mode && gate.q.group == up.q.group &&
                            gate.q.half_scales == up.q.half_scales);
  const HostMat gu = same_layout ? interleave_gate_up(tmp, gate, up) : HostMat{};
  const std::vector<HostMat> runs = fuse(tmp, parts);
  id<MTLCommandBuffer> cb = [m.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  auto dn = [&](const char* k) { return m.dense_mats.count(k) > 0; };
  if (same_layout) {
    L.gate_up = finish(m, e, tmp, gu, dn("gate_up"));
  } else {
    L.gate = finish(m, e, tmp, gate, dn("gate_up"));
    L.up = finish(m, e, tmp, up, dn("gate_up"));
    m.split_mlp = true;
  }
  L.down = finish(m, e, tmp, down, dn("down"));
  L.out = finish(m, e, tmp, out, dn("out"));
  for (const HostMat& r : runs) L.in_proj.push_back(finish(m, e, tmp, r, dn("in_proj")));
  [e endEncoding];
  commit_wait(cb);
  return L;
}

// A dense linear layer of the tower whose rows are several tensors stacked
// (q, k, v: one [3 D, D] matrix and its biases).
inline HW load_vstack(Model& m, id<MTLComputeCommandEncoder> e, Arena& tmp, const WeightSource& src,
                      const std::vector<std::string>& bases, int out_each, int in) {
  const int n = int(bases.size());
  Buf host = tmp.alloc(size_t(n) * size_t(out_each) * size_t(in) * 2);
  std::vector<float> bias;
  for (int i = 0; i < n; ++i) {
    const Buf part = f16_from_view(tmp, src.tensor(bases[size_t(i)] + ".weight"), out_each, in);
    std::memcpy((uint8_t*)ptr(host) + size_t(i) * size_t(out_each) * size_t(in) * 2, ptr(part),
                size_t(out_each) * size_t(in) * 2);
    const std::vector<float> b = src.floats(bases[size_t(i)] + ".bias", size_t(out_each));
    bias.insert(bias.end(), b.begin(), b.end());
  }
  return repack_dense(m, e, host, n * out_each, in, &bias);
}

// The SigLIP2 tower into the Qwen-VL tower's structures (vision.h): q, k, v
// fused, layer_norm1 / layer_norm2 as norm1 / norm2, post_layernorm as the
// merger's norm, the projector's linear_1 / linear_2 as the merger's fc1 / fc2,
// rotary frequencies zero (the shared q/k prep then rotates by 0: no RoPE).
inline void load_lfm2_vision(Model& m, const WeightSource& src, const Lfm2Names& nm, VisionConfig c) {
  c.validate(m.cfg.D);
  Arena tmp(m.dev);
  const std::string v = nm.vt;
  {
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    m.vis.patch = load_vdense(m, e, tmp, src, v + "embeddings.patch_embedding", c.D, c.patch_dim());
    m.vis.m1 = load_vdense(m, e, tmp, src, nm.mm + "linear_1", c.proj, 4 * c.D);
    m.vis.m2 = load_vdense(m, e, tmp, src, nm.mm + "linear_2", c.out, c.proj);
    [e endEncoding];
    commit_wait(cb);
  }
  m.vis.pos = vec(m, src, v + "embeddings.position_embedding.weight", size_t(c.n_pos) * c.D);
  m.vis.mnw = vec(m, src, v + "post_layernorm.weight", c.D);
  m.vis.mnb = vec(m, src, v + "post_layernorm.bias", c.D);
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

inline void load_lfm2(Model& m, const WeightSource& src, const json::Value& root, bool want_vision) {
  Config& c = m.cfg;
  c.validate();
  const Lfm2Names nm = lfm2_names(src);
  const std::string pre = nm.lm;
  {
    Arena tmp(m.dev);
    load_embedding(m, tmp, src, pre + "embed_tokens.weight");
    if (!c.tie && !src.has("lm_head.weight") && !src.has("language_model.lm_head.weight"))
      fail("LFM2: untied checkpoint without an lm_head");
    const std::string head = c.tie ? pre + "embed_tokens.weight"
                                   : (src.has("language_model.lm_head.weight") ? "language_model.lm_head.weight" : "lm_head.weight");
    const HostMat h = host_mat(m, tmp, src, head, c.V, c.D);
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    m.lm_head = finish(m, e, tmp, h);
    [e endEncoding];
    commit_wait(cb);
  }
  m.final_norm = vec(m, src, pre + "embedding_norm.weight", c.D);
  int nf = 0, nl = 0;
  for (int i = 0; i < c.L; ++i)
    m.layers.push_back(load_lfm2_layer(m, src, pre + "layers." + std::to_string(i) + ".", c.linear[size_t(i)],
                                       c.linear[size_t(i)] ? nl++ : nf++));
  m.inv_freq = m.arena->alloc(sizeof(float) * size_t(c.rot / 2));
  float* f = (float*)ptr(m.inv_freq);
  for (int i = 0; i < c.rot / 2; ++i) f[i] = float(1.0 / std::pow(c.theta, double(2 * i) / double(c.rot)));
  m.rope_sel = m.arena->copy(c.rope_sel, sizeof(int32_t) * 32);
  if (want_vision && !nm.vt.empty() && src.has(nm.vt + "embeddings.patch_embedding.weight")) {
    const VisionConfig vc = vision_from_lfm2(root);
    if (vc.present) load_lfm2_vision(m, src, nm, vc);
  }
}

}  // namespace espetero::native
