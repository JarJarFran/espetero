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

// A gate/up matrix of a GeGLU MLP: the gated epilogues apply the tanh GELU
// (act 1, Gemma 4) or the exact GELU (act 2, Spark).
inline void set_gelu_gate(Mat& w, int act = 1) {
  if (w.dense) w.d.act = act;
  else w.q.act = act;
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

// A dense fp16 host copy of a matrix in any format (quantized ones expanded).
inline Buf g4v_f16(Model& m, Arena& tmp, const WeightSource& src, const std::string& name, int out, int in) {
  return as_f16(tmp, host_mat(m, tmp, src, name, out, in));
}

// The Gemma 4 vision tower and embed_vision (gemma_vision.h runs them), by
// mlx-vlm's gemma4 names (vision_tower.*, embed_vision.*); GGUF mmproj files
// are read through GGUFGemmaVisionSource, which maps the same names. The
// tower's matrices become fp16, the projection keeps its format.
inline void load_gemma_vision(Model& m, const WeightSource& src, VisionConfig c) {
  c.validate(m.cfg.D);
  const std::string v = "vision_tower.", e = v + "encoder.layers.";
  const int D = c.D, pd = c.patch_dim();
  Arena tmp(m.dev);
  {
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    // input_proj: [D, (y, x, channel)] (MLX), or llama.cpp's conv kernel [D, C, P, P].
    const std::string pn = v + "patch_embedder.input_proj.weight";
    const TensorView pw = src.tensor(pn);
    Buf ph;
    if (pw.shape.size() == 4) {
      std::vector<int> perm(static_cast<size_t>(pd));
      const int P = c.patch, C = c.in_ch;
      for (int y = 0; y < P; ++y)
        for (int x = 0; x < P; ++x)
          for (int ch = 0; ch < C; ++ch) perm[size_t((y * P + x) * C + ch)] = (ch * P + y) * P + x;
      ph = f16_from_view(tmp, pw, D, pd, &perm);
    } else {
      ph = g4v_f16(m, tmp, src, pn, D, pd);
    }
    m.vis.patch = repack_dense(m, enc, ph, D, pd);
    HostMat proj = host_mat(m, tmp, src, "embed_vision.embedding_projection.weight", c.out, D);
    m.vis.proj = finish(m, enc, tmp, proj);
    [enc endEncoding];
    commit_wait(cb);
  }
  m.vis.pos = vec(m, src, v + "patch_embedder.position_embedding_table", size_t(2) * size_t(c.n_pos) * size_t(D));
  if (c.standardize) {
    m.vis.std_bias = vec(m, src, v + "std_bias", size_t(D));
    m.vis.std_scale = vec(m, src, v + "std_scale", size_t(D));
  }
  for (int i = 0; i < c.depth; ++i) {
    Arena btmp(m.dev);
    const std::string b = e + std::to_string(i) + ".";
    VBlock B;
    B.n1w = vec(m, src, b + "input_layernorm.weight", size_t(D));
    B.post_attn = vec(m, src, b + "post_attention_layernorm.weight", size_t(D));
    B.n2w = vec(m, src, b + "pre_feedforward_layernorm.weight", size_t(D));
    B.post_ffn = vec(m, src, b + "post_feedforward_layernorm.weight", size_t(D));
    B.qn = vec(m, src, b + "self_attn.q_norm.weight", size_t(c.hd));
    B.kn = vec(m, src, b + "self_attn.k_norm.weight", size_t(c.hd));
    // q, k, v as one [3D, D] matrix.
    Buf qkv = btmp.alloc(size_t(3) * D * D * 2);
    const char* names[3] = {"q_proj", "k_proj", "v_proj"};
    for (int k = 0; k < 3; ++k) {
      const Buf w = g4v_f16(m, btmp, src, b + "self_attn." + names[k] + ".linear.weight", D, D);
      std::memcpy((uint8_t*)ptr(qkv) + size_t(k) * D * D * 2, ptr(w), size_t(D) * D * 2);
    }
    const Buf o = g4v_f16(m, btmp, src, b + "self_attn.o_proj.linear.weight", D, D);
    const Buf g = g4v_f16(m, btmp, src, b + "mlp.gate_proj.linear.weight", c.I, D);
    const Buf u = g4v_f16(m, btmp, src, b + "mlp.up_proj.linear.weight", c.I, D);
    const Buf dn = g4v_f16(m, btmp, src, b + "mlp.down_proj.linear.weight", D, c.I);
    const Buf gu = interleave8_f16(btmp, g, u, c.I, D);
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    B.qkv = repack_dense(m, enc, qkv, 3 * D, D);
    B.proj = repack_dense(m, enc, o, D, D);
    B.fc1 = repack_dense(m, enc, gu, 2 * c.I, D);
    B.fc1.act = 1;  // GeGLU (tanh GELU gate)
    B.fc2 = repack_dense(m, enc, dn, D, c.I);
    [enc endEncoding];
    commit_wait(cb);
    m.vis.blocks.push_back(B);
  }
  // 2D RoPE: hd / 4 frequencies per axis, theta^(-2i / (hd / 2)).
  const int nf = c.hd / 4;
  std::vector<float> f(static_cast<size_t>(nf));
  for (int i = 0; i < nf; ++i) f[size_t(i)] = float(1.0 / std::pow(c.rope_theta, double(2 * i) / double(2 * nf)));
  m.vis.inv_freq = m.arena->copy(f.data(), f.size() * 4);
  c.present = true;
  m.vis.c = c;
}

// The Gemma 4 unified embedder (gemma_vision.h runs it) by mlx-vlm's
// gemma4_unified names (vision_embedder.*, embed_vision.*); a gemma4uv mmproj
// through GGUFGemmaVisionSource. patch_dense becomes fp16 with its bias, the
// projection keeps its format. The position table is [n_pos][2][D] (MLX) or
// [2][n_pos][D] (llama.cpp), stored [n_pos][2][D].
inline void load_gemma_unified_vision(Model& m, const WeightSource& src, VisionConfig c) {
  c.validate(m.cfg.D);
  const std::string v = "vision_embedder.";
  const int D = c.D, pd = c.unified_dim(), n = c.n_pos;
  m.vis.ln1w = vec(m, src, v + "patch_ln1.weight", size_t(pd));
  m.vis.ln1b = vec(m, src, v + "patch_ln1.bias", size_t(pd));
  m.vis.ln2w = vec(m, src, v + "patch_ln2.weight", size_t(D));
  m.vis.ln2b = vec(m, src, v + "patch_ln2.bias", size_t(D));
  m.vis.ln3w = vec(m, src, v + "pos_norm.weight", size_t(D));
  m.vis.ln3b = vec(m, src, v + "pos_norm.bias", size_t(D));
  {
    const TensorView pt = src.tensor(v + "pos_embedding");
    std::vector<float> p = src.floats(v + "pos_embedding", size_t(2) * size_t(n) * size_t(D));
    // llama.cpp's [2][n_pos][D] (a GGUF view flattens it to [2 n_pos, D]).
    if (pt.shape == std::vector<int64_t>{2, n, D} || pt.shape == std::vector<int64_t>{2 * int64_t(n), D}) {
      std::vector<float> q(p.size());
      for (int i = 0; i < n; ++i)
        for (int a = 0; a < 2; ++a)
          std::memcpy(&q[(size_t(i) * 2 + size_t(a)) * size_t(D)], &p[(size_t(a) * size_t(n) + size_t(i)) * size_t(D)],
                      size_t(D) * 4);
      p.swap(q);
    } else if (pt.shape != std::vector<int64_t>{n, 2, D}) {
      fail(v + "pos_embedding: expected [" + std::to_string(n) + ", 2, " + std::to_string(D) + "]");
    }
    m.vis.pos = m.arena->copy(p.data(), p.size() * 4);
  }
  Arena tmp(m.dev);
  id<MTLCommandBuffer> cb = [m.queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  const std::vector<float> bias = src.floats(v + "patch_dense.bias", size_t(D));
  m.vis.patch = repack_dense(m, enc, g4v_f16(m, tmp, src, v + "patch_dense.weight", D, pd), D, pd, &bias);
  m.vis.proj = finish(m, enc, tmp, host_mat(m, tmp, src, "embed_vision.embedding_projection.weight", c.out, D));
  [enc endEncoding];
  commit_wait(cb);
  c.present = true;
  m.vis.c = c;
  // Image soft tokens attend to their whole image in every layer, as
  // transformers' Gemma4UnifiedModel builds its masks (and llama.cpp
  // decodes gemma4uv images); ESPETERO_G4_BIDI still overrides it.
  m.g4_bidi = 2;
}

// Spark-X2.5 / X2.6 (spark2_5): one decoder layer (prefix `p`, e.g.
// "model.layers.3."): the fused q_k_v_proj, g_proj (H gate rows, kept fp16
// for sp_head_gate: a GEMM dispatch of 16 rows cost more than its dots),
// out_proj, the GeGLU MLP and two RMSNorms.
inline Layer load_sp_layer(Model& m, const WeightSource& src, const std::string& p, int li, int index_in_kind) {
  const Config& c = m.cfg;
  Layer L;
  L.linear = false;
  L.swa = c.swa[size_t(li)];
  L.index_in_kind = index_in_kind;
  L.in_norm = vec(m, src, p + "input_layernorm.weight", c.D);
  L.post_norm = vec(m, src, p + "post_attention_layernorm.weight", c.D);
  Arena tmp(m.dev);
  const std::string a = p + "self_attn.";
  const std::vector<HostMat> parts = {host_mat(m, tmp, src, a + "q_k_v_proj.weight", c.H * c.HD + 2 * c.kv_dim(), c.D)};
  {
    const TensorView g = src.tensor(a + "g_proj.weight");
    if (g.kind != TensorView::Dense) fail(a + "g_proj.weight: expected unquantized gates (mlx-lm and EQ keep them)");
    Buf h = f16_from_view(tmp, g, c.H, c.D);
    L.head_gate = m.arena->copy(ptr(h), size_t(c.H) * size_t(c.D) * 2);
  }
  HostMat out = host_mat(m, tmp, src, a + "out_proj.weight", c.D, c.H * c.HD);
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
  id<MTLCommandBuffer> cb = [m.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  auto dn = [&](const char* k) { return m.dense_mats.count(k) > 0; };
  if (same_layout) {
    L.gate_up = finish(m, e, tmp, gu, dn("gate_up"));
    set_gelu_gate(L.gate_up, 2);
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

// Spark checkpoints: Hugging Face (BF16, the names mlx-lm keeps), MLX
// conversions (mlx-lm's spark2_5) and EQ checkpoints, all named model.*.
// RoPE (nn.RoPE, traditional false): pairs (d, d + rot / 2) of the first
// rot dims at theta^(-2d / rot), per layer kind.
inline void load_spark(Model& m, const WeightSource& src) {
  Config& c = m.cfg;
  c.validate();
  if (!src.has("model.embedding.weight")) fail("expected a Spark checkpoint (model.embedding.weight)");
  {
    Arena tmp(m.dev);
    load_embedding(m, tmp, src, "model.embedding.weight");
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    m.lm_head = finish(m, e, tmp, host_mat(m, tmp, src, c.tie ? "model.embedding.weight" : "lm_head.weight", c.V, c.D));
    [e endEncoding];
    commit_wait(cb);
  }
  m.final_norm = vec(m, src, "model.norm.weight", c.D);
  int nf = 0, ns = 0;
  for (int i = 0; i < c.L; ++i)
    m.layers.push_back(load_sp_layer(m, src, "model.layers." + std::to_string(i) + ".", i, c.swa[size_t(i)] ? ns++ : nf++));
  auto table = [&](int rot, double theta) {
    std::vector<float> f(size_t(rot / 2));
    for (int i = 0; i < rot / 2; ++i) f[size_t(i)] = float(1.0 / std::pow(theta, double(2 * i) / double(rot)));
    return m.arena->copy(f.data(), f.size() * 4);
  };
  m.inv_freq = table(c.rot, c.theta);
  m.inv_freq_g = table(c.rot_g, c.theta_g);
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
