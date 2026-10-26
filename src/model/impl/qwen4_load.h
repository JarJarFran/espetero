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

// GGUF checkpoints (llama.cpp qwen4exp): norms carry their +1 already (except
// the DeltaNet gated norm, plain in both), and native-scale blocks (Q8_0)
// become affine so every matrix has the MLX layout these kernels were built on.
inline bool q4_gguf(const Model& m) { return m.format == "gguf"; }
inline float q4_norm_add(const Model& m) { return q4_gguf(m) ? 0.0f : 1.0f; }
// native8: a Q8_0 matrix keeps its native scales (8.5 instead of 9 bits per
// weight; the plain GEMVs read both), for matrices outside the fused paths.
inline HostMat q4_mat(Model& m, Arena& tmp, const WeightSource& src, const std::string& name, int out, int in,
                      bool native8 = false) {
  HostMat h = host_mat(m, tmp, src, name, out, in);
  if (!h.dense && h.q.scale_mode && !(native8 && h.q.scale_mode == 2)) h.q = to_affine(tmp, h.q);
  return h;
}

// The up matrix's rows in the hcmix order (EPI_HCMIX, q4_hc_mix): row s * D + c
// (stream s, channel c) moves to (c / per) * 32 + (c % per) * S + s, per = 32 / S,
// so a 32-column GEMV block holds every stream of its channels.
inline HostMat hcmix_rows(Arena& tmp, const HostMat& h, int S, int D) {
  if (32 % S || D % (32 / S) || h.out != S * D) fail("hyper-connection up matrix: unexpected shape for the hcmix order");
  if (!h.dense && h.q.scale_mode) fail("hyper-connection up matrix: unexpected scale format");
  const int per = 32 / S;
  auto permuted = [&](const Buf& src, size_t row_bytes) {
    Buf dst = tmp.alloc(row_bytes * size_t(h.out));
    const uint8_t* a = static_cast<const uint8_t*>(ptr(src));
    uint8_t* b = static_cast<uint8_t*>(ptr(dst));
    for (int s = 0; s < S; ++s)
      for (int c = 0; c < D; ++c)
        std::memcpy(b + size_t((c / per) * 32 + (c % per) * S + s) * row_bytes, a + size_t(s * D + c) * row_bytes, row_bytes);
    return dst;
  };
  HostMat o = h;
  if (h.dense) {
    o.f16 = permuted(h.f16, size_t(h.in) * 2);
  } else {
    o.q.w = permuted(h.q.w, size_t(h.in) * size_t(h.q.src_bits) / 8);
    o.q.s = permuted(h.q.s, size_t(h.in / h.q.group) * 2);
    o.q.z = permuted(h.q.z, size_t(h.in / h.q.group) * 2);
  }
  return o;
}

// A file range read with F_NOCACHE preads (64 MB pieces in parallel, or in
// file order from a slow disk), so tens of GB of resident experts do not also
// fill the file cache.
inline void pread_uncached(const std::string& path, uint64_t off, size_t n, uint8_t* dst) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) fail("cannot open " + path);
  ::fcntl(fd, F_NOCACHE, 1);
  const bool ok = read_pieces(fd, dst, n, size_t(off), size_t(64) << 20);
  ::close(fd);
  if (!ok) fail("read failed: " + path);
}

inline HCBlock load_hc(Model& m, const WeightSource& src, const std::string& p, bool inject, int rq = 0) {
  const Config& c = m.cfg;
  HCBlock B;
  if (q4_gguf(m)) rq = 8;  // the F32 inject matrix as 8 bits (quantized in MLX files)
  B.norm = vec(m, src, p + "hc_norm.weight", size_t(c.W()), q4_norm_add(m));
  Arena tmp(m.dev);
  const HostMat down = rq8(tmp, q4_mat(m, tmp, src, p + "input_mix_weight_down.weight", c.hc_rank, c.W()), rq);
  const HostMat up =
      hcmix_rows(tmp, rq8(tmp, q4_mat(m, tmp, src, p + "input_mix_weight_up.weight", c.W(), c.hc_rank), rq), c.hc, c.D);
  std::vector<HostMat> runs{down};
  if (inject) runs = fuse(tmp, {down, rq8(tmp, q4_mat(m, tmp, src, p + "block_inject_weight.weight", c.hc, c.W()), rq)});
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
inline ExpertLayer locate_experts(Model& m, const WeightSource& wsrc, const std::string& p) {
  const auto* sp = dynamic_cast<const SafetensorsSource*>(&wsrc);
  if (!sp) fail(p + ": experts can only be streamed from MLX checkpoints");
  const SafetensorsSource& src = *sp;
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
inline Layer load_q4_layer(Model& m, const WeightSource& src, const std::string& p, bool linear,
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
  HostMat gate = rq8(tmp, q4_mat(m, tmp, src, sp + "gate_proj.weight", L.I, c.D), rq);
  HostMat up = rq8(tmp, q4_mat(m, tmp, src, sp + "up_proj.weight", L.I, c.D), rq);
  HostMat down = rq8(tmp, q4_mat(m, tmp, src, sp + "down_proj.weight", c.D, L.I), rq);
  std::vector<HostMat> parts;
  HostMat out, idx;
  if (linear) {
    const std::string a = p + "linear_attn.";
    // GGUF: alpha and beta are F32; 8 bits keep them in the quantized run (a dense
    // [96, 2560] fp16 GEMM measured 2.2 ms per decode step on the M4 Max).
    const int rab = q4_gguf(m) ? 8 : 0;
    parts = {q4_mat(m, tmp, src, a + "in_proj_qkv.weight", c.C(), c.D, true),
             q4_mat(m, tmp, src, a + "in_proj_z.weight", c.Vd(), c.D, true),
             rq8(tmp, q4_mat(m, tmp, src, a + "in_proj_a.weight", c.Hv, c.D), rab),
             rq8(tmp, q4_mat(m, tmp, src, a + "in_proj_b.weight", c.Hv, c.D), rab)};
    out = q4_mat(m, tmp, src, a + "out_proj.weight", c.D, c.Vd(), true);
    L.conv_w = vec(m, src, a + "conv1d.weight", size_t(c.C()) * c.conv_k);
    L.a_log = vec(m, src, a + "A_log", c.Hv);
    L.dt_bias = vec(m, src, a + "dt_bias", c.Hv);
    L.gnorm = vec(m, src, a + "norm.weight", c.Dv);  // plain weight
  } else {
    const std::string a = p + "self_attn.";
    parts = {rq8(tmp, q4_mat(m, tmp, src, a + "q_proj.weight", c.H * c.HD * 2, c.D, true), rq),
             rq8(tmp, q4_mat(m, tmp, src, a + "k_proj.weight", c.kv_dim(), c.D, true), rq),
             rq8(tmp, q4_mat(m, tmp, src, a + "v_proj.weight", c.kv_dim(), c.D, true), rq)};
    out = rq8(tmp, q4_mat(m, tmp, src, a + "o_proj.weight", c.D, c.H * c.HD, true), rq);
    L.q_norm = vec(m, src, a + "q_norm.weight", c.HD, q4_norm_add(m));
    L.k_norm = vec(m, src, a + "k_norm.weight", c.HD, q4_norm_add(m));
    idx = rq8(tmp, q4_mat(m, tmp, src, a + "indexer.index_qk_proj.weight", (c.idx_heads + c.idx_kv) * c.idx_hd, c.D),
              q4_gguf(m) ? 8 : rq);  // GGUF: bf16 q and k rows
    L.iq_norm = vec(m, src, a + "indexer.q_layernorm.weight", c.idx_hd, q4_norm_add(m));
    L.ik_norm = vec(m, src, a + "indexer.k_layernorm.weight", c.idx_hd, q4_norm_add(m));
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
    HostMat router = q4_mat(m, t2, src, p + "mlp.gate.weight", c.E, c.D);
    HostMat sgate = q4_mat(m, t2, src, p + "mlp.shared_expert_gate.weight", 1, c.D);
    if (q4_gguf(m) && sgate.dense) sgate = requant<8>(t2, sgate);  // F32 in GGUF, 8-bit in oQ4e files
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
      if (const auto* g = dynamic_cast<const GGUFSource*>(&src)) {  // the file's widths, converted exactly
        std::vector<uint8_t> raw;
        const TensorView t = g->tensor_uncached(s + name, raw);
        HostMat h;
        h.out = c.E * out;
        h.in = in;
        h.q = qw_from_view(keep, t, h.out, in);
        if (h.q.scale_mode) h.q = to_affine(keep, h.q);
        m.source_types[ggml_type_info(t.gguf_type)->name] += 1;
        return h;
      }
      if (const auto* st = dynamic_cast<const SafetensorsSource*>(&src); st && !st->locate(s + name).dtype.empty() &&
                                                                         st->has(s + name.substr(0, name.size() - 6) + "scales")) {
        // A quantized MLX stack (a converted GGUF at its own widths): codes, scales and biases read uncached.
        const std::string b = s + name.substr(0, name.size() - 6);  // "...gate_proj."
        TensorView t = st->tensor(s + name);
        std::vector<uint8_t> raw[3];
        const char* part[3] = {"weight", "scales", "biases"};
        const uint8_t** ptrs[3] = {&t.data, &t.scales, &t.biases};
        for (int k = 0; k < 3; ++k) {
          const SafetensorsSource::Location l = st->locate(b + part[k]);
          raw[k].resize(l.nbytes);
          pread_uncached(l.file, l.offset, l.nbytes, raw[k].data());
          *ptrs[k] = raw[k].data();
        }
        HostMat h;
        h.out = c.E * out;
        h.in = in;
        h.q = qw_from_view(keep, t, h.out, in);
        m.source_types["mlx" + std::to_string(t.bits) + "g" + std::to_string(t.group)] += 1;
        return h;
      }
      HostMat h = q4_mat(m, keep, src, s + name, c.E * out, in);  // the bf16 MTP drafter: requantized below
      return rq8(keep, h, 8);
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

// The MTP head (docs/qwen4_exp.md 7): the 31 tensors under `pre` ("" in a
// drafter directory such as Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16, "mtp." in
// the model file itself). rq 8 requantizes dense (bf16) matrices to 8 bits;
// quantized ones load as stored. The 512 experts load resident (the MTP layer
// runs outside the expert cache). Shares the trunk's embedding and LM head.
inline void load_q4_mtp_from(Model& m, const WeightSource& src, const std::string& pre, int rq, const std::string& source) {
  const Config& c = m.cfg;
  if (!src.has(pre + "fc_embedding.weight") || !src.has(pre + "layers.0.mlp.gate.weight"))
    fail(source + ": not a qwen4_exp MTP head (" + pre + "fc_embedding / " + pre + "layers.0.* tensors)");
  Mtp& T = m.mtp;
  const int dense0 = m.n_dense, exempt0 = m.n_dense_mtp;
  T.enorm = vec(m, src, pre + "pre_fc_norm_embedding.weight", size_t(c.D), 1.0f);
  T.hnorm = vec(m, src, pre + "pre_fc_norm_hidden.weight", size_t(c.W()), 1.0f);
  {
    Arena tmp(m.dev);
    const HostMat fe = rq8(tmp, q4_mat(m, tmp, src, pre + "fc_embedding.weight", c.D, c.D), rq);
    const HostMat fh = rq8(tmp, q4_mat(m, tmp, src, pre + "fc_hidden.weight", c.D, c.D), rq);
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    T.fc_emb = finish(m, e, tmp, fe);
    T.fc_hid = finish(m, e, tmp, fh);
    [e endEncoding];
    commit_wait(cb);
  }
  T.mixer = load_hc(m, src, pre + "hyper_connection_mixer.", false, rq);
  T.layer = load_q4_layer(m, src, pre + "layers.0.", false, -1, rq, true);
  m.n_dense_mtp += (m.n_dense - dense0) - (m.n_dense_mtp - exempt0);
  T.post_norm = false;
  T.source = source;
  T.present = true;
}

// A drafter directory (bf16: its dense matrices and experts become 8-bit copies, ~2.7 GB resident).
inline void load_q4_mtp(Model& m, const std::string& dir) {
  SafetensorsSource src(dir);
  load_q4_mtp_from(m, src, "", 8, dir.substr(dir.find_last_of('/') + 1));
  const Mtp& T = m.mtp;
  std::fprintf(stderr, "mtp: %s (%d-bit experts, %d resident)\n", T.source.c_str(), T.layer.experts_gu.bits, m.cfg.E);
}

// The model file's own `mtp.*` head, as stored (the oQ4e file: 4-bit g64
// experts and fc, 5-6-bit attention and hyper-connection matrices, 8-bit
// shared expert, bf16 router; ~1.5 GB resident).
inline void load_q4_mtp_own(Model& m, const SafetensorsSource& src) {
  load_q4_mtp_from(m, src, "mtp.", 0, "own");
  const Mtp& T = m.mtp;
  std::fprintf(stderr, "mtp: the file's own head (%d-bit experts, %d resident)\n", T.layer.experts_gu.bits, m.cfg.E);
}

// The n-gram mix before layer c.ple_layer and the location of its table.
inline void load_q4_ple(Model& m, const SafetensorsSource& src, const std::string& p) {
  const Config& c = m.cfg;
  Ple& P = m.ple;
  Arena tmp(m.dev);
  const HostMat key = q4_mat(m, tmp, src, p + "key_proj.weight", c.W(), c.D);
  const HostMat value = q4_mat(m, tmp, src, p + "value_proj.weight", c.D, c.D);
  id<MTLCommandBuffer> cb = [m.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  P.key = finish(m, e, tmp, key);
  P.value = finish(m, e, tmp, value);
  [e endEncoding];
  commit_wait(cb);
  P.norm_key = vec(m, src, p + "norm_key.weight", size_t(c.W()), q4_norm_add(m));
  P.norm_query = vec(m, src, p + "norm_query.weight", size_t(c.W()), q4_norm_add(m));
  P.norm_conv = vec(m, src, p + "norm_conv.weight", size_t(c.W()), q4_norm_add(m));
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
    // Rows are read from a mapping: three small preads per row cost a system
    // call each (16 rows per token, ~116K calls for a 2.4K-token prompt).
    struct stat st{};
    void* p = MAP_FAILED;
    if (::fstat(fd, &st) == 0 && st.st_size > 0) p = ::mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
    T.maps.push_back(p == MAP_FAILED ? nullptr : static_cast<const uint8_t*>(p));
  }
  P.present = true;
}

// The n-gram table into the file cache in the background (sequential reads
// in 64 MB chunks, 4 readers; the pages the mapping reads come from that
// cache): a cold prompt otherwise reads its rows from the SSD at random
// (M4 Max 128 GB, resident experts: 3K tokens 14.2 s prefill cold, 5.1 s
// after the preload). Opt-in (ESPETERO_NGRAM_PRELOAD=1): beside 85 GB of
// resident experts and weights on a 128 GB Mac, 32 GB more of file cache left
// macOS compressing tens of GB, and the next load took 16 minutes instead of 30 s.
inline void ensure_ngram_preload(const NgramTable& T) {
  if (const char* v = std::getenv("ESPETERO_NGRAM_PRELOAD"); !v || *v != '1') return;
  struct Job { int fd; uint64_t off, n; };
  std::vector<Job> jobs;
  const uint64_t chunk = uint64_t(64) << 20;
  const uint64_t ng = uint64_t(T.width / T.group);
  for (const NgramTable::Shard& S : T.shards) {
    const int fd = T.fds[size_t(S.file)];
    const uint64_t rows = uint64_t(T.shard_rows);
    for (auto [off, len] : {std::pair<uint64_t, uint64_t>{S.w, rows * uint64_t(T.width / 2)}, {S.s, rows * ng * 2}, {S.b, rows * ng * 2}})
      for (uint64_t o = 0; o < len; o += chunk) jobs.push_back({fd, off + o, std::min(chunk, len - o)});
  }
  std::thread([jobs = std::move(jobs)] {
    const auto t0 = Clock::now();
    std::atomic<size_t> next{0};
    std::atomic<uint64_t> bytes{0};
    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r)
      readers.emplace_back([&] {
        std::vector<uint8_t> buf(size_t(64) << 20);
        for (size_t i; (i = next.fetch_add(1)) < jobs.size();) {
          const Job& j = jobs[i];
          for (uint64_t done = 0; done < j.n;) {
            const ssize_t got = ::pread(j.fd, buf.data(), size_t(j.n - done), off_t(j.off + done));
            if (got <= 0) break;
            done += uint64_t(got);
          }
          bytes += j.n;
        }
      });
    for (std::thread& t : readers) t.join();
    std::fprintf(stderr, "n-gram table: %.1f GB in the file cache (%.0f s)\n", double(bytes) / 1e9, ms_since(t0) / 1000.0);
  }).detach();
}

// draft_head: an MTP head will draft; its 2-bit coarse LM head (mtp.coarse_c > 0) and 4-bit head (mtp.head4)
// are built from the LM head as load_decoder does (a draft step read the whole 8-bit head, 675 MB, before).
inline void load_q4(Model& m, const SafetensorsSource& src, bool want_vision, bool draft_head) {
  Config& c = m.cfg;
  c.validate();
  const std::string pre = "language_model.model.";
  {
    Arena tmp(m.dev);
    load_embedding(m, tmp, src, pre + "embed_tokens.weight");
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    const HostMat h = q4_mat(m, tmp, src, c.tie ? pre + "embed_tokens.weight" : "language_model.lm_head.weight", c.V, c.D);
    if (draft_head && m.mtp.head4) {
      m.mtp.head = finish(m, e, tmp, requant4(tmp, h));
      --m.n_q4;  // not a decoder matrix
    }
    if (draft_head && m.mtp.coarse_c > 0 && !h.dense) {
      m.mtp.coarse = finish(m, e, tmp, requant<2>(tmp, h));
      --m.n_q8;  // not a decoder matrix (finish counts widths without a counter as 8-bit)
    }
    m.lm_head = finish(m, e, tmp, h);
    [e endEncoding];
    commit_wait(cb);
  }
  m.mixer = load_hc(m, src, pre + "hyper_connection_mixer.", false);
  // Experts of mixed widths (a GGUF converted at its own widths: 4-bit gate/up
  // beside a 5- or 8-bit down) load resident, as GGUF files do; the SSD
  // stream, records and pools are 4-bit only. ESPETERO_Q4_RESIDENT_LOAD=1 forces it.
  bool mixed = env_on("ESPETERO_Q4_RESIDENT_LOAD");
  for (int i = 0; i < c.L && !mixed; ++i) {
    const std::string s = pre + "layers." + std::to_string(i) + ".mlp.switch_mlp.";
    const auto q = src.quant_of(s + "gate_proj");
    mixed = q != src.quant_of(s + "up_proj") || q != src.quant_of(s + "down_proj") || q.first != 4;
  }
  if (mixed) m.xres_l.resize(size_t(c.L));
  const auto t0 = Clock::now();
  int nf = 0, nl = 0;
  for (int i = 0; i < c.L; ++i) {
    const std::string p = pre + "layers." + std::to_string(i) + ".";
    m.layers.push_back(load_q4_layer(m, src, p, c.linear[size_t(i)], c.linear[size_t(i)] ? nl++ : nf++, 0, mixed));
    if (mixed) {
      Layer& L = m.layers.back();
      m.xres_l[size_t(i)] = {L.experts_gu, L.experts_down};
      if ((i + 1) % 8 == 0) std::fprintf(stderr, "loading: %d/%d layers (%.0f s)\n", i + 1, c.L, ms_since(t0) / 1000.0);
    }
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
  m.xs.present = !mixed;
  m.inv_freq = m.arena->alloc(sizeof(float) * size_t(c.rot / 2));
  float* f = (float*)ptr(m.inv_freq);
  for (int i = 0; i < c.rot / 2; ++i) f[i] = float(1.0 / std::pow(c.theta, double(2 * i) / double(c.rot)));
  m.rope_sel = m.arena->copy(c.rope_sel, sizeof(int32_t) * 32);
  if (want_vision && src.has("vision_tower.patch_embed.proj.weight")) {
    const VisionConfig vc = vision_from_mlx(src.config());
    if (vc.present) load_vision(m, src, vc);
  }
}


// ---------------------------------------------------------------- GGUF (llama.cpp qwen4exp)
// The n-gram mix of a GGUF file: the same matrices and norms, the table as
// one IQ4_NL tensor (per_layer_token_embd, rows of 160) read from a mapping.
inline void load_q4_ple_gguf(Model& m, const GGUFSource& src, const std::string& p) {
  const Config& c = m.cfg;
  Ple& P = m.ple;
  {
    Arena tmp(m.dev);
    const HostMat key = q4_mat(m, tmp, src, p + "key_proj.weight", c.W(), c.D);
    const HostMat value = q4_mat(m, tmp, src, p + "value_proj.weight", c.D, c.D);
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    P.key = finish(m, e, tmp, key);
    P.value = finish(m, e, tmp, value);
    [e endEncoding];
    commit_wait(cb);
  }
  P.norm_key = vec(m, src, p + "norm_key.weight", size_t(c.W()));
  P.norm_query = vec(m, src, p + "norm_query.weight", size_t(c.W()));
  P.norm_conv = vec(m, src, p + "norm_conv.weight", size_t(c.W()));
  P.conv_w = vec(m, src, p + "conv1d.weight", size_t(c.W()) * size_t(c.ple_conv_k));
  const GGUFFile& f = src.file();
  const std::string a = "qwen4exp.";
  NgramTable& T = P.table;
  const GGUFFile::Tensor* t = f.find("per_layer_token_embd.weight");
  if (!t || t->type != GGML_IQ4_NL || t->dims.size() != 2) fail("qwen4exp: per_layer_token_embd.weight is not an IQ4_NL table");
  T.fmt = 1;
  T.width = int(t->dims[0]);
  T.group = 32;
  T.scale = 1.0f;
  if (T.width != int(f.get_int(a + "embedding_length_per_layer_input", 0)) || T.width % 32 || T.width > 512)
    fail("qwen4exp: unexpected n-gram row width " + std::to_string(T.width));
  T.rows = T.shard_rows = t->dims[1];
  const std::vector<int64_t> mult = f.get_ints(a + "ple.layer_multipliers");
  T.vocab = f.get_ints(a + "ple.head_vocab_sizes");
  T.offs = f.get_ints(a + "ple.head_offsets");
  const int heads = 2 * c.ngram_heads;
  if (int(mult.size()) != c.ngram || int(T.vocab.size()) != heads || int(T.offs.size()) != heads)
    fail("qwen4exp: n-gram hash constants do not match the configuration");
  std::copy(mult.begin(), mult.end(), T.mult);
  if (T.offs.back() + T.vocab.back() > T.rows) fail("qwen4exp: hash heads exceed the n-gram table");
  NgramTable::Shard S;
  S.file = 0;
  S.w = t->offset;
  T.shards.push_back(S);
  T.files.push_back(f.path());
  const int fd = ::open(f.path().c_str(), O_RDONLY);
  if (fd < 0) fail("cannot open " + f.path());
  T.fds.push_back(fd);
  struct stat st{};
  void* mp = MAP_FAILED;
  if (::fstat(fd, &st) == 0 && st.st_size > 0) mp = ::mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
  T.maps.push_back(mp == MAP_FAILED ? nullptr : static_cast<const uint8_t*>(mp));
  const char* mv = std::getenv("ESPETERO_NGRAM_MODE");
  const std::string mode = mv && *mv ? mv : "ram";
  if (mode == "disk") {
    T.mode = 1;
    T.maps[0] = nullptr;  // every row a pread that bypasses the file cache
    ::fcntl(fd, F_NOCACHE, 1);
  } else if (mode == "gpu") {
    T.mode = 2;
    const auto t0 = Clock::now();
    const size_t rb = size_t(T.width / 32) * 18;
    const size_t max_len = std::min<size_t>(size_t([m.dev maxBufferLength]), size_t(8) << 30);
    T.gpu_rows_per = int64_t(max_len / rb);
    const int64_t nbuf = (T.rows + T.gpu_rows_per - 1) / T.gpu_rows_per;
    if (nbuf > 8) fail("n-gram table: more than 8 GPU buffers needed");
    const int rfd = ::open(f.path().c_str(), O_RDONLY);
    ::fcntl(rfd, F_NOCACHE, 1);
    for (int64_t b = 0; b < nbuf; ++b) {
      const int64_t r0 = b * T.gpu_rows_per, nr = std::min(T.gpu_rows_per, T.rows - r0);
      const size_t len = size_t(nr) * rb;
      id<MTLBuffer> buf = [m.dev newBufferWithLength:len options:MTLResourceStorageModeShared];
      if (!buf) fail("n-gram table: Metal allocation of " + std::to_string(len >> 20) + " MB failed");
      Residency::get().add(buf);
      uint8_t* dst = static_cast<uint8_t*>([buf contents]);
      if (!read_pieces(rfd, dst, len, size_t(S.w + uint64_t(r0) * rb), size_t(64) << 20))
        fail("n-gram table: read failed (" + f.path() + ")");
      T.gpu.push_back(buf);
    }
    ::close(rfd);
    std::fprintf(stderr, "n-gram table: %.1f GB in %lld GPU buffers (%.0f s)\n", double(T.rows) * double(rb) / 1e9,
                 (long long)nbuf, ms_since(t0) / 1000.0);
  }
  P.present = true;
}

// A Qwen4-Exp GGUF: the trunk as load_q4 builds it, with every routed expert
// resident in the file's own widths (Q4_K gate/up, Q5_1 or Q8_0 down in
// unsloth's UD-Q4_K_XL: ~80 GB, converted exactly) instead of streamed.
inline void load_q4_gguf(Model& m, const GGUFSource& src) {
  Config& c = m.cfg;
  c.validate();
  const std::string pre = "language_model.model.";
  {
    Arena tmp(m.dev);
    load_embedding(m, tmp, src, pre + "embed_tokens.weight");
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    m.lm_head = finish(m, e, tmp, q4_mat(m, tmp, src, c.tie ? pre + "embed_tokens.weight" : "language_model.lm_head.weight",
                                         c.V, c.D));
    [e endEncoding];
    commit_wait(cb);
  }
  m.mixer = load_hc(m, src, pre + "hyper_connection_mixer.", false);
  const auto t0 = Clock::now();
  int nf = 0, nl = 0;
  m.xres_l.resize(size_t(c.L));
  for (int i = 0; i < c.L; ++i) {
    const std::string p = pre + "layers." + std::to_string(i) + ".";
    m.layers.push_back(load_q4_layer(m, src, p, c.linear[size_t(i)], c.linear[size_t(i)] ? nl++ : nf++, 0, true));
    Layer& L = m.layers.back();
    m.xres_l[size_t(i)] = {L.experts_gu, L.experts_down};
    if (i == c.ple_layer) load_q4_ple_gguf(m, src, p + "ple.");
    if ((i + 1) % 8 == 0 || env_on("ESPETERO_LOAD_LOG"))
      std::fprintf(stderr, "loading: %d/%d layers (%.0f s; experts %d/%d bits)\n", i + 1, c.L, ms_since(t0) / 1000.0,
                   L.experts_gu.bits, L.experts_down.bits);
  }
  if (!m.ple.present) fail("qwen4exp: n-gram embedding missing at layer " + std::to_string(c.ple_layer));
  m.xs.present = false;
  m.inv_freq = m.arena->alloc(sizeof(float) * size_t(c.rot / 2));
  float* fr = (float*)ptr(m.inv_freq);
  for (int i = 0; i < c.rot / 2; ++i) fr[i] = float(1.0 / std::pow(c.theta, double(2 * i) / double(c.rot)));
  m.rope_sel = m.arena->copy(c.rope_sel, sizeof(int32_t) * 32);
}

}  // namespace espetero::native
