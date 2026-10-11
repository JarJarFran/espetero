// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Qwen4-Exp (`qwen4_exp`, Qwen3.8-Flash-Next) loader for MLX checkpoints
// (docs/qwen4_exp.md, Appendix A). Everything but the routed experts and the
// n-gram table is loaded: hyper-connection blocks and the final mixer, the
// DeltaNet and sparse-attention layers (with the indexer), routers, shared
// experts, the n-gram mix, embedding and LM head (~4 GB for the oQ4e file).
// The routed experts (67.95 GB) and the n-gram table (32 GB) stay in the
// safetensors files; their byte locations are recorded for pread.
//
// Norm weights of this family are zero-centred in the file (the runtime norm
// multiplies by 1 + w); the loader folds the +1 like the Qwen3.5 converters
// do, except for the DeltaNet gated norm, whose weight is plain.
#pragma once

#include <fcntl.h>

#include "load.h"

namespace espetero::native {

// rq 8: a dense (bf16) matrix becomes an 8-bit affine copy (the bf16 MTP
// drafter, as the loader does for the Qwen3.5 MTP layers); quantized ones stay.
inline HostMat rq8(Arena& tmp, const HostMat& h, int rq) { return rq == 8 && h.dense ? requant<8>(tmp, h) : h; }

inline HCBlock load_hc(Model& m, const SafetensorsSource& src, const std::string& p, bool inject, int rq = 0) {
  const Config& c = m.cfg;
  HCBlock B;
  B.norm = vec(m, src, p + "hc_norm.weight", size_t(c.W()), 1.0f);
  Arena tmp(m.dev);
  const HostMat down = rq8(tmp, host_mat(m, tmp, src, p + "input_mix_weight_down.weight", c.hc_rank, c.W()), rq);
  const HostMat up = rq8(tmp, host_mat(m, tmp, src, p + "input_mix_weight_up.weight", c.W(), c.hc_rank), rq);
  std::vector<HostMat> runs{down};
  if (inject) runs = fuse(tmp, {down, rq8(tmp, host_mat(m, tmp, src, p + "block_inject_weight.weight", c.hc, c.W()), rq)});
  id<MTLCommandBuffer> cb = [m.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  B.down = finish(m, e, tmp, runs[0]);
  if (inject) {
    B.inj_fused = runs.size() == 1;
    if (!B.inj_fused) B.inj = finish(m, e, tmp, runs[1]);
  }
  B.up = finish(m, e, tmp, up);
  [e endEncoding];
  commit_wait(cb);
  return B;
}

// The expert stacks of one layer: MLX switch_mlp tensors [E, out, in] (and
// their scales / biases), expert e's slice at offset + e * stride.
inline ExpertLayer locate_experts(Model& m, const SafetensorsSource& src, const std::string& p) {
  const Config& c = m.cfg;
  ExpertStore& X = m.xs;
  auto piece = [&](const std::string& name) {
    const SafetensorsSource::Location l = src.locate(name);
    if (l.shape.empty() || l.shape[0] != c.E || l.nbytes % uint64_t(c.E)) fail(name + ": not an [E, ...] expert stack");
    ExpertPiece q;
    auto it = std::find(X.files.begin(), X.files.end(), l.file);
    q.file = int(it - X.files.begin());
    if (it == X.files.end()) X.files.push_back(l.file);
    q.off = l.offset;
    q.stride = l.nbytes / uint64_t(c.E);
    return q;
  };
  const std::string s = p + "mlp.switch_mlp.";
  ExpertLayer E;
  E.gw = piece(s + "gate_proj.weight");
  E.gs = piece(s + "gate_proj.scales");
  E.gb = piece(s + "gate_proj.biases");
  E.uw = piece(s + "up_proj.weight");
  E.us = piece(s + "up_proj.scales");
  E.ub = piece(s + "up_proj.biases");
  E.dw = piece(s + "down_proj.weight");
  E.ds = piece(s + "down_proj.scales");
  E.db = piece(s + "down_proj.biases");
  const auto [bits, group] = src.quant_of(s + "gate_proj");
  for (const char* n : {"up_proj", "down_proj"})
    if (src.quant_of(s + n) != std::make_pair(bits, group)) fail(p + ": experts with mixed widths are not supported");
  E.bits = bits;
  E.group = group;
  return E;
}

// rq 8: dense matrices requantized to 8 bits (the bf16 MTP drafter);
// resident: the routed experts are loaded (8-bit) instead of located on the
// SSD (the MTP layer, which runs outside the expert cache).
inline Layer load_q4_layer(Model& m, const SafetensorsSource& src, const std::string& p, bool linear,
                           int index_in_kind, int rq = 0, bool resident = false) {
  const Config& c = m.cfg;
  Layer L;
  L.linear = linear;
  L.index_in_kind = index_in_kind;
  L.hca = load_hc(m, src, p + "attn_hyper_connection.", true, rq);
  L.hcm = load_hc(m, src, p + "mlp_hyper_connection.", true, rq);
  Arena tmp(m.dev);
  const std::string sp = p + "mlp.shared_expert.";
  L.I = c.I_shared;
  HostMat gate = rq8(tmp, host_mat(m, tmp, src, sp + "gate_proj.weight", L.I, c.D), rq);
  HostMat up = rq8(tmp, host_mat(m, tmp, src, sp + "up_proj.weight", L.I, c.D), rq);
  HostMat down = rq8(tmp, host_mat(m, tmp, src, sp + "down_proj.weight", c.D, L.I), rq);
  std::vector<HostMat> parts;
  HostMat out, idx;
  if (linear) {
    const std::string a = p + "linear_attn.";
    parts = {host_mat(m, tmp, src, a + "in_proj_qkv.weight", c.C(), c.D),
             host_mat(m, tmp, src, a + "in_proj_z.weight", c.Vd(), c.D),
             host_mat(m, tmp, src, a + "in_proj_a.weight", c.Hv, c.D),
             host_mat(m, tmp, src, a + "in_proj_b.weight", c.Hv, c.D)};
    out = host_mat(m, tmp, src, a + "out_proj.weight", c.D, c.Vd());
    L.conv_w = vec(m, src, a + "conv1d.weight", size_t(c.C()) * c.conv_k);
    L.a_log = vec(m, src, a + "A_log", c.Hv);
    L.dt_bias = vec(m, src, a + "dt_bias", c.Hv);
    L.gnorm = vec(m, src, a + "norm.weight", c.Dv);  // plain weight
  } else {
    const std::string a = p + "self_attn.";
    parts = {rq8(tmp, host_mat(m, tmp, src, a + "q_proj.weight", c.H * c.HD * 2, c.D), rq),
             rq8(tmp, host_mat(m, tmp, src, a + "k_proj.weight", c.kv_dim(), c.D), rq),
             rq8(tmp, host_mat(m, tmp, src, a + "v_proj.weight", c.kv_dim(), c.D), rq)};
    out = rq8(tmp, host_mat(m, tmp, src, a + "o_proj.weight", c.D, c.H * c.HD), rq);
    L.q_norm = vec(m, src, a + "q_norm.weight", c.HD, 1.0f);
    L.k_norm = vec(m, src, a + "k_norm.weight", c.HD, 1.0f);
    idx = rq8(tmp, host_mat(m, tmp, src, a + "indexer.index_qk_proj.weight", (c.idx_heads + c.idx_kv) * c.idx_hd, c.D), rq);
    L.iq_norm = vec(m, src, a + "indexer.q_layernorm.weight", c.idx_hd, 1.0f);
    L.ik_norm = vec(m, src, a + "indexer.k_layernorm.weight", c.idx_hd, 1.0f);
  }
  const bool same_layout = gate.dense || up.dense ||
                           (gate.q.bits == up.q.bits && gate.q.src_bits == up.q.src_bits &&
                            gate.q.scale_mode == up.q.scale_mode && gate.q.group == up.q.group &&
                            gate.q.half_scales == up.q.half_scales);
  const HostMat gu = same_layout ? interleave_gate_up(tmp, gate, up) : HostMat{};
  const std::vector<HostMat> runs = fuse(tmp, parts);
  id<MTLCommandBuffer> cb = [m.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  if (same_layout) {
    L.gate_up = finish(m, e, tmp, gu);
  } else {
    L.gate = finish(m, e, tmp, gate);
    L.up = finish(m, e, tmp, up);
    m.split_mlp = true;
  }
  L.down = finish(m, e, tmp, down);
  L.out = finish(m, e, tmp, out);
  for (const HostMat& r : runs) L.in_proj.push_back(finish(m, e, tmp, r));
  if (!linear) L.index_qk = finish(m, e, tmp, idx);
  [e endEncoding];
  commit_wait(cb);
  // Router (bf16 in the oQ4e file: the exact fp16 router path) and shared gate;
  // the routed experts stay on the SSD.
  {
    Arena t2(m.dev);
    HostMat router = host_mat(m, t2, src, p + "mlp.gate.weight", c.E, c.D);
    HostMat sgate = host_mat(m, t2, src, p + "mlp.shared_expert_gate.weight", 1, c.D);
    if (router.dense) L.router_f16 = m.arena->copy(ptr(router.f16), size_t(c.E) * size_t(c.D) * 2);
    id<MTLCommandBuffer> cb2 = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e2 = [cb2 computeCommandEncoder];
    L.router = finish(m, e2, t2, router);
    L.shared_gate = finish(m, e2, t2, sgate);
    [e2 endEncoding];
    commit_wait(cb2);
    m.n_dense_mtp += (L.router.dense ? 1 : 0) + (L.shared_gate.dense ? 1 : 0);
  }
  L.moe = true;
  if (resident) {
    // [E, out, in] stacks, requantized to 8 bits one at a time (bf16 source:
    // 1.7 GB each), gate and up interleaved by 8 rows per expert.
    const std::string s = p + "mlp.switch_mlp.";
    Arena keep(m.dev);
    auto stack = [&](const std::string& name, int out, int in) {
      Arena t1(m.dev);
      return rq8(keep, host_mat(m, t1, src, s + name, c.E * out, in), 8);
    };
    const HostMat eg = stack("gate_proj.weight", c.Ie, c.D), eu = stack("up_proj.weight", c.Ie, c.D);
    const HostMat ed = stack("down_proj.weight", c.D, c.Ie);
    if (eg.dense || eu.dense || ed.dense) fail(p + ": resident experts must be quantized");
    const HostMat egu = interleave_gate_up(keep, eg, eu);
    id<MTLCommandBuffer> cb3 = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e3 = [cb3 computeCommandEncoder];
    L.experts_gu = finish(m, e3, keep, egu).q;
    L.experts_down = finish(m, e3, keep, ed).q;
    [e3 endEncoding];
    commit_wait(cb3);
    return L;
  }
  m.xs.layers.push_back(locate_experts(m, src, p));
  // The experts' format without their data (read per pass or held in the
  // expert cache): moe_mlp chooses its kernels from it.
  const ExpertLayer& X = m.xs.layers.back();
  auto fmt = [&](int out, int in) {
    QW q;
    q.out = out;
    q.in = in;
    q.src_bits = X.bits;
    q.bits = gpu_bits_for(X.bits);
    q.group = X.group;
    return q;
  };
  L.experts_gu = fmt(c.E * 2 * c.Ie, c.D);
  L.experts_down = fmt(c.E * c.D, c.Ie);
  return L;
}

// The MTP head (docs/qwen4_exp.md 7) from a drafter directory such as
// Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16 (31 tensors without prefix): its
// dense matrices and 512 experts requantized to 8 bits at load (~2.7 GB
// resident; the oQ4e file's own MTP experts are 4-bit, reported to accept
// 0-4 % of drafts). Shares the trunk's embedding and LM head.
inline void load_q4_mtp(Model& m, const std::string& dir) {
  const Config& c = m.cfg;
  SafetensorsSource src(dir);
  if (!src.has("fc_embedding.weight") || !src.has("layers.0.mlp.gate.weight"))
    fail(dir + ": not a qwen4_exp MTP drafter (fc_embedding / layers.0.* tensors)");
  Mtp& T = m.mtp;
  const int dense0 = m.n_dense, exempt0 = m.n_dense_mtp;
  T.enorm = vec(m, src, "pre_fc_norm_embedding.weight", size_t(c.D), 1.0f);
  T.hnorm = vec(m, src, "pre_fc_norm_hidden.weight", size_t(c.W()), 1.0f);
  {
    Arena tmp(m.dev);
    const HostMat fe = rq8(tmp, host_mat(m, tmp, src, "fc_embedding.weight", c.D, c.D), 8);
    const HostMat fh = rq8(tmp, host_mat(m, tmp, src, "fc_hidden.weight", c.D, c.D), 8);
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    T.fc_emb = finish(m, e, tmp, fe);
    T.fc_hid = finish(m, e, tmp, fh);
    [e endEncoding];
    commit_wait(cb);
  }
  T.mixer = load_hc(m, src, "hyper_connection_mixer.", false, 8);
  T.layer = load_q4_layer(m, src, "layers.0.", false, -1, 8, true);
  m.n_dense_mtp += (m.n_dense - dense0) - (m.n_dense_mtp - exempt0);
  T.post_norm = false;
  T.source = dir.substr(dir.find_last_of('/') + 1);
  T.present = true;
  std::fprintf(stderr, "mtp: %s (8-bit, 512 resident experts)\n", T.source.c_str());
}

// The n-gram mix before layer c.ple_layer and the location of its table.
inline void load_q4_ple(Model& m, const SafetensorsSource& src, const std::string& p) {
  const Config& c = m.cfg;
  Ple& P = m.ple;
  Arena tmp(m.dev);
  const HostMat key = host_mat(m, tmp, src, p + "key_proj.weight", c.W(), c.D);
  const HostMat value = host_mat(m, tmp, src, p + "value_proj.weight", c.D, c.D);
  id<MTLCommandBuffer> cb = [m.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  P.key = finish(m, e, tmp, key);
  P.value = finish(m, e, tmp, value);
  [e endEncoding];
  commit_wait(cb);
  P.norm_key = vec(m, src, p + "norm_key.weight", size_t(c.W()), 1.0f);
  P.norm_query = vec(m, src, p + "norm_query.weight", size_t(c.W()), 1.0f);
  P.norm_conv = vec(m, src, p + "norm_conv.weight", size_t(c.W()), 1.0f);
  P.conv_w = vec(m, src, p + "conv1d.weight", size_t(c.W()) * size_t(c.ple_conv_k));
  NgramTable& T = P.table;
  const std::string pe = p + "ple_embedding.";
  auto i64 = [&](const std::string& name, size_t n) {
    const SafetensorsSource::Location l = src.locate(name);
    if (l.dtype != "I64" || l.nbytes != n * 8) fail(name + ": expected I64[" + std::to_string(n) + "]");
    std::vector<int64_t> v(n);
    std::memcpy(v.data(), l.data, n * 8);
    return v;
  };
  const int heads = 2 * c.ngram_heads;  // bigram and trigram hash heads
  const std::vector<int64_t> mult = i64(pe + "layer_multipliers", size_t(c.ngram));
  std::copy(mult.begin(), mult.end(), T.mult);
  T.vocab = i64(pe + "ngram_heads_vocab_sizes", size_t(heads));
  T.offs = i64(pe + "ngram_heads_offsets", size_t(heads));
  const std::string sh = pe + "ngram_embedding.";
  if (src.has(sh + "weight_scale")) {
    const std::vector<float> s = src.floats(sh + "weight_scale", 1);
    T.scale = s[0];
  }
  for (int k = 0;; ++k) {
    const std::string b = sh + "shards." + std::to_string(k) + ".";
    if (!src.has(b + "weight")) break;
    NgramTable::Shard S;
    const SafetensorsSource::Location w = src.locate(b + "weight"), s = src.locate(b + "scales"),
                                      z = src.locate(b + "biases");
    if (w.file != s.file || w.file != z.file) fail(b + ": codes, scales and biases in different files");
    auto it = std::find(T.files.begin(), T.files.end(), w.file);
    S.file = int(it - T.files.begin());
    if (it == T.files.end()) T.files.push_back(w.file);
    S.w = w.offset;
    S.s = s.offset;
    S.b = z.offset;
    const int64_t rows = w.shape[0];
    if (k == 0) T.shard_rows = rows;
    else if (rows != T.shard_rows) fail(b + ": shards of different sizes");
    if (w.shape[1] * 8 != T.width || s.shape[1] * T.group != T.width) fail(b + ": unexpected row layout");
    T.shards.push_back(S);
    T.rows += rows;
  }
  if (T.shards.empty()) fail(sh + ": no shards");
  int64_t total = 0;
  for (int64_t v : T.vocab) total += v;
  if (total > T.rows || T.offs.back() + T.vocab.back() > T.rows) fail(sh + ": hash heads exceed the table");
  for (const std::string& f : T.files) {
    const int fd = ::open(f.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open " + f);
    T.fds.push_back(fd);
  }
  P.present = true;
}

inline void load_q4(Model& m, const SafetensorsSource& src, bool want_vision) {
  Config& c = m.cfg;
  c.validate();
  const std::string pre = "language_model.model.";
  {
    Arena tmp(m.dev);
    load_embedding(m, tmp, src, pre + "embed_tokens.weight");
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    m.lm_head = finish(m, e, tmp, host_mat(m, tmp, src, c.tie ? pre + "embed_tokens.weight" : "language_model.lm_head.weight",
                                           c.V, c.D));
    [e endEncoding];
    commit_wait(cb);
  }
  m.mixer = load_hc(m, src, pre + "hyper_connection_mixer.", false);
  int nf = 0, nl = 0;
  for (int i = 0; i < c.L; ++i) {
    const std::string p = pre + "layers." + std::to_string(i) + ".";
    m.layers.push_back(load_q4_layer(m, src, p, c.linear[size_t(i)], c.linear[size_t(i)] ? nl++ : nf++));
    if (i == c.ple_layer) load_q4_ple(m, src, p + "ple.");
  }
  if (!m.ple.present) fail("qwen4_exp: n-gram embedding missing at layer " + std::to_string(c.ple_layer));
  // Expert reads bypass the unified buffer cache (F_NOCACHE; ESPETERO_XS_CACHE=1 for buffered reads),
  // so tens of GB of streamed experts do not fill memory beside the pools.
  static const bool nocache = !env_on("ESPETERO_XS_CACHE");  // buffered reads (ESPETERO_XS_CACHE=1) measured 2.8 vs 0.74 ms submission delay per flush
  for (const std::string& f : m.xs.files) {
    const int fd = ::open(f.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open " + f);
    if (nocache) ::fcntl(fd, F_NOCACHE, 1);
    m.xs.fds.push_back(fd);
  }
  m.xs.present = true;
  m.inv_freq = m.arena->alloc(sizeof(float) * size_t(c.rot / 2));
  float* f = (float*)ptr(m.inv_freq);
  for (int i = 0; i < c.rot / 2; ++i) f[i] = float(1.0 / std::pow(c.theta, double(2 * i) / double(c.rot)));
  m.rope_sel = m.arena->copy(c.rope_sel, sizeof(int32_t) * 32);
  if (want_vision && src.has("vision_tower.patch_embed.proj.weight")) {
    const VisionConfig vc = vision_from_mlx(src.config());
    if (vc.present) load_vision(m, src, vc);
  }
}

}  // namespace espetero::native
