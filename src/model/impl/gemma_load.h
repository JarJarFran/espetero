// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Gemma 4 checkpoints: the decoder of an MLX gemma4 / gemma4_text
// directory (mlx-lm 0.32's gemma4_text names: language_model.model.*), dense
// (31B) or with the mixture-of-experts block beside the dense MLP (26B-A4B).
// The forward pass is in gemma.h; the semantics follow mlx-lm's
// models/gemma4_text.py (MIT, see THIRD_PARTY_NOTICES.md).
#pragma once

namespace espetero::native {

// Rows [r0, r0 + n) of each of `blocks` runs of `stride` rows, as one host
// matrix: the gate or the up half of every expert in llama.cpp's fused
// ffn_gate_up_exps ([E, 2 Ie, D], gate rows first). Rows are copied whole
// (codes, scales, biases), so any format is exact.
inline HostMat select_rows(Arena& tmp, const HostMat& h, int blocks, int stride, int r0, int n) {
  HostMat o;
  o.out = blocks * n;
  o.in = h.in;
  auto copy = [&](const Buf& src, size_t row, Buf& dst) {
    if (!row) return;
    dst = tmp.alloc(row * size_t(o.out));
    const uint8_t* a = (const uint8_t*)ptr(src);
    uint8_t* b = (uint8_t*)ptr(dst);
    for (int k = 0; k < blocks; ++k)
      std::memcpy(b + size_t(k) * size_t(n) * row, a + (size_t(k) * size_t(stride) + size_t(r0)) * row, size_t(n) * row);
  };
  if (h.dense) {
    o.dense = true;
    copy(h.f16, size_t(h.in) * 2, o.f16);
    return o;
  }
  o.q = h.q;
  o.q.out = o.out;
  copy(h.q.w, h.q.row_bytes(), o.q.w);
  copy(h.q.s, h.q.s_row_bytes(), o.q.s);
  copy(h.q.z, h.q.z_row_bytes(), o.q.z);
  return o;
}

// A gate/up matrix of a GeGLU MLP: the gated epilogues apply the tanh GELU.
inline void set_gelu_gate(Mat& w) {
  if (w.dense) w.d.act = 1;
  else w.q.act = 1;
}

// One decoder layer (prefix `p`, e.g. "language_model.model.layers.3.").
inline Layer load_g4_layer(Model& m, const WeightSource& src, const std::string& p, int li, int index_in_kind) {
  const Config& c = m.cfg;
  Layer L;
  L.linear = false;
  L.swa = c.swa[size_t(li)];
  L.index_in_kind = index_in_kind;
  const int hd = L.swa ? c.HD : c.HDg, hkv = L.swa ? c.Hkv : c.Hkv_g;
  const bool has_v = L.swa || !c.k_eq_v;
  L.in_norm = vec(m, src, p + "input_layernorm.weight", c.D);
  L.post_norm = vec(m, src, p + "post_attention_layernorm.weight", c.D);
  L.pre_ffn = vec(m, src, p + "pre_feedforward_layernorm.weight", c.D);
  L.post_ffn = vec(m, src, p + "post_feedforward_layernorm.weight", c.D);
  // layer_scalar (bf16 [1]): every residual row is multiplied by it after the layer.
  L.scalar = src.has(p + "layer_scalar") ? src.floats(p + "layer_scalar", 1)[0] : 1.0f;
  Arena tmp(m.dev);
  const std::string a = p + "self_attn.";
  std::vector<HostMat> parts = {host_mat(m, tmp, src, a + "q_proj.weight", c.H * hd, c.D),
                                host_mat(m, tmp, src, a + "k_proj.weight", hkv * hd, c.D)};
  if (has_v) parts.push_back(host_mat(m, tmp, src, a + "v_proj.weight", hkv * hd, c.D));
  else if (src.has(a + "v_proj.weight")) fail(a + "v_proj.weight: a K = V layer with a value projection");
  HostMat out = host_mat(m, tmp, src, a + "o_proj.weight", c.D, c.H * hd);
  L.q_norm = vec(m, src, a + "q_norm.weight", size_t(hd));
  L.k_norm = vec(m, src, a + "k_norm.weight", size_t(hd));
  // The dense MLP (every layer; MoE layers run it beside the experts).
  L.I = c.I;
  HostMat gate = host_mat(m, tmp, src, p + "mlp.gate_proj.weight", c.I, c.D);
  HostMat up = host_mat(m, tmp, src, p + "mlp.up_proj.weight", c.I, c.D);
  HostMat down = host_mat(m, tmp, src, p + "mlp.down_proj.weight", c.D, c.I);
  const bool same_layout = gate.dense || up.dense ||
                           (gate.q.bits == up.q.bits && gate.q.src_bits == up.q.src_bits &&
                            gate.q.scale_mode == up.q.scale_mode && gate.q.group == up.q.group &&
                            gate.q.half_scales == up.q.half_scales);
  const HostMat gu = same_layout ? interleave_gate_up(tmp, gate, up) : HostMat{};
  const std::vector<HostMat> runs = fuse(tmp, parts);
  {
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    auto dn = [&](const char* k) { return m.dense_mats.count(k) > 0; };
    if (same_layout) {
      L.gate_up = finish(m, e, tmp, gu, dn("gate_up"));
      set_gelu_gate(L.gate_up);
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
  }
  if (c.E > 0) {
    if (!src.has(p + "router.proj.weight")) fail(p + "router.proj.weight missing (the config enables the MoE block)");
    const bool fused_gu = src.has(p + "experts.switch_glu.gate_up_proj.weight");  // GGUF ffn_gate_up_exps
    if (!fused_gu && !src.has(p + "experts.switch_glu.gate_proj.weight"))
      fail(p + "experts: expected the mlx-lm layout (experts.switch_glu.{gate,up,down}_proj)");
    L.moe = true;
    L.post_ffn1 = vec(m, src, p + "post_feedforward_layernorm_1.weight", c.D);
    L.pre_ffn2 = vec(m, src, p + "pre_feedforward_layernorm_2.weight", c.D);
    L.post_ffn2 = vec(m, src, p + "post_feedforward_layernorm_2.weight", c.D);
    // The router normalizes the residual with scale * D^-0.5 (fp32 here) and
    // scales the softmax of its top-k logits by per_expert_scale.
    {
      std::vector<float> w = src.floats(p + "router.scale", size_t(c.D));
      const float r = float(1.0 / std::sqrt(double(c.D)));
      for (float& x : w) x *= r;
      L.router_norm = m.arena->copy(w.data(), w.size() * 4);
    }
    L.expert_scale = vec(m, src, p + "router.per_expert_scale", size_t(c.E));
    Arena xt(m.dev);
    HostMat router = host_mat(m, xt, src, p + "router.proj.weight", c.E, c.D);
    HostMat eg, eu;
    if (fused_gu) {
      const HostMat gu = host_mat(m, xt, src, p + "experts.switch_glu.gate_up_proj.weight", c.E * 2 * c.Ie, c.D);
      eg = select_rows(xt, gu, c.E, 2 * c.Ie, 0, c.Ie);
      eu = select_rows(xt, gu, c.E, 2 * c.Ie, c.Ie, c.Ie);
    } else {
      eg = host_mat(m, xt, src, p + "experts.switch_glu.gate_proj.weight", c.E * c.Ie, c.D);
      eu = host_mat(m, xt, src, p + "experts.switch_glu.up_proj.weight", c.E * c.Ie, c.D);
    }
    HostMat ed = host_mat(m, xt, src, p + "experts.switch_glu.down_proj.weight", c.E * c.D, c.Ie);
    if (eg.dense || eu.dense || ed.dense) fail("dense expert weights are not supported");
    const HostMat egu = interleave_gate_up(xt, eg, eu);  // every expert's 2 Ie rows stay together (Ie % 64 == 0)
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    L.router = finish(m, e, xt, router);
    if (L.router.dense) ++m.n_dense_mtp;  // routers never take a fused-norm link
    L.experts_gu = finish(m, e, xt, egu).q;
    L.experts_gu.act = 1;
    L.experts_down = finish(m, e, xt, ed).q;
    [e endEncoding];
    commit_wait(cb);
  }
  return L;
}

// RoPE frequencies of the two layer kinds: sliding layers rotate every pair
// (d, d + HD/2) at theta^(-2d/HD) (nn.RoPE); global layers rotate the first
// rot_g / 2 pairs (d, d + HDg/2) at 1 / (factor * theta_g^(2d/HDg)) and leave
// the others unrotated (ProportionalRoPE: their frequency is 0).
inline void g4_rope_tables(Model& m) {
  const Config& c = m.cfg;
  std::vector<float> fs(size_t(c.HD / 2)), fg(size_t(c.HDg / 2), 0.0f);
  for (int i = 0; i < c.HD / 2; ++i) fs[size_t(i)] = float(1.0 / std::pow(c.theta, double(2 * i) / double(c.HD)));
  for (int i = 0; i < c.rot_g / 2; ++i)
    fg[size_t(i)] = float(1.0 / (c.rope_factor_g * std::pow(c.theta_g, double(2 * i) / double(c.HDg))));
  m.inv_freq = m.arena->copy(fs.data(), fs.size() * 4);
  m.inv_freq_g = m.arena->copy(fg.data(), fg.size() * 4);
}

inline void load_gemma(Model& m, const WeightSource& src) {
  Config& c = m.cfg;
  c.validate();
  const std::string pre = "language_model.model.";
  if (!src.has(pre + "embed_tokens.weight"))
    fail("expected an MLX-converted Gemma 4 checkpoint (language_model.model.* tensors)");
  {
    Arena tmp(m.dev);
    load_embedding(m, tmp, src, pre + "embed_tokens.weight");
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    const std::string head = c.tie ? pre + "embed_tokens.weight" : "language_model.lm_head.weight";
    m.lm_head = finish(m, e, tmp, host_mat(m, tmp, src, head, c.V, c.D));
    [e endEncoding];
    commit_wait(cb);
  }
  m.final_norm = vec(m, src, pre + "norm.weight", c.D);
  int nf = 0, ns = 0;
  for (int i = 0; i < c.L; ++i)
    m.layers.push_back(load_g4_layer(m, src, pre + "layers." + std::to_string(i) + ".", i, c.swa[size_t(i)] ? ns++ : nf++));
  g4_rope_tables(m);
}

}  // namespace espetero::native
