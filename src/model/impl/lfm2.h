// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// LFM2 forward pass: the decoder layers of a forest pass, encoded by
// run_forest (forest.h) after the embedding (and the vision tower), with the
// same rows, trees, readouts, KV block lists and slot states as the Qwen3.5
// path. Per layer:
//
//   x = rmsnorm(h) * operator_norm
//   conv layers:      [B | C | u] = x Win;  y = C * conv3(B * u);  h += y Wout
//                     (the taps before a tree's first row come from the parent
//                     slot's conv state: the last B * u rows, as the DeltaNet
//                     conv windows of the Qwen3.5 family)
//   attention layers: q, k, v = x Wq, x Wk, x Wv;  q, k = rope(rmsnorm(q) * q_norm), rope(rmsnorm(k) * k_norm)
//                     h += attention(q / 8, k, v) Wo   (GQA, paged KV pool)
//   h += down(silu(x2 W1) * x2 W3),  x2 = rmsnorm(h) * ffn_norm
//
// Semantics of transformers' models/lfm2 (Apache-2.0, THIRD_PARTY_NOTICES.md).
#pragma once

namespace espetero::native {

struct LfmConvParamsHost {
  int32_t D, ldx;
  uint64_t conv_base, conv_stride;
};
struct LfmPrepParamsHost {
  int32_t H, Hkv, ld, q_ld, kv_ld;
  float eps, scale;
};

// Everything the layer loop reads from run_forest.
struct Lfm2PassIn {
  int M = 0, N = 0;
  bool prune = false, chained = false;
  const AttnPlan* plan = nullptr;  // run_forest's plan (tiles of 8 rows; pad0 = 0 is the first batch key)
  Buf trees, tiles, comb, parts;
  size_t parts_bytes = 0;
  Buf saves;  // KVSaveRec entries into the paged pool
  size_t n_saves = 0;
  Buf conv_src, row_slot, conv_jobs;
  size_t n_conv_jobs = 0;
  Buf blocks, desc, pos, readout_rows;
};

inline std::string lfm2_attention_kernel(const Config& c) { return "lfm_attention64_g" + std::to_string(c.H / c.Hkv); }

inline void lfm2_encode_layers(Model& m, Enc& E, const Lfm2PassIn& P) {
  const Config& c = m.cfg;
  Scratch& s = m.sc;
  const StatePool& S = m.st;
  const KVPool& K = m.kv;
  const int M = P.M, N = P.N;
  const int G = c.H / c.Hkv, qd = c.H * c.HD, kvd = c.kv_dim(), ldc = c.lin_ld(), lda = c.full_ld();
  auto rmsnorm = [&](id<MTLBuffer> X, const Buf& w, id<MTLBuffer> Y, int rows) {
    E.pipe("rmsnorm");
    E.set(0, X);
    E.set(1, w);
    E.set(2, Y);
    E.bytes(3, c.D);
    E.bytes(4, c.eps);
    const int g = 0;
    E.set(5, X);
    E.bytes(6, g);
    E.rd(X);
    E.wr(Y);
    E.groups(MTLSizeMake(size_t(rows), 1, 1), MTLSizeMake(256, 1, 1));
  };
  auto gather = [&](id<MTLBuffer> src, id<MTLBuffer> dst, int width, bool f32) {
    E.pipe(f32 ? "gather_rows_f32" : "gather_rows_f16");
    E.set(0, src);
    E.set(1, dst);
    E.set(2, P.readout_rows);
    E.bytes(3, width);
    E.rd(src);
    E.wr(dst);
    E.threads(MTLSizeMake(size_t(width / 4), size_t(N), 1), MTLSizeMake(64, 1, 1));
  };
  // Long prompt passes at long contexts go in groups of tiles, each committed
  // on its own (as run_forest's attention does: macOS aborts command buffers
  // of seconds while the display is in use); same tiles, same order.
  static const size_t split_rk = [] {
    const char* e = std::getenv("ESPETERO_ATTN_SPLIT");
    return size_t(e && *e ? std::atoi(e) : 16) << 20;
  }();
  auto attention = [&](int layer_k) {
    const AttnPlan& A = *P.plan;
    if (A.tiles.empty()) return;
    const G4AttnParamsHost ap{c.H, c.Hkv, qd, kvd, K.bt, 0, 0, 0, uint64_t(K.layer_base(layer_k))};
    std::vector<size_t> cut{0};
    if (split_rk > 0 && !P.chained && !m.chain_active && M >= 64) {
      size_t acc = 0;
      for (size_t i = 0; i < A.tiles.size(); ++i) {
        const AttnTileRec& tr = A.tiles[i];
        const size_t w = size_t(8) * (size_t(tr.k1 - tr.k0) + size_t(A.trees[size_t(tr.tree)].M));
        if (acc > 0 && acc + w > split_rk) cut.push_back(i), acc = 0;
        acc += w;
      }
    }
    cut.push_back(A.tiles.size());
    for (size_t gi = 0; gi + 1 < cut.size(); ++gi) {
      if (gi > 0) E.flush();
      E.pipe(lfm2_attention_kernel(c));
      E.set(0, s.Q);
      E.set(1, s.Kb);
      E.set(2, s.Vb);
      E.set(3, K.buf);
      E.set(4, P.blocks);
      E.set(5, P.desc);
      E.set(7, s.O);
      E.bytes(8, ap);
      E.set(9, P.tiles, cut[gi] * sizeof(AttnTileRec));
      E.set(10, P.trees);
      E.set(11, P.parts);
      E.rd(s.Q);
      E.rd(s.Kb);
      E.rd(s.Vb);
      E.rd(K.buf, size_t(ap.kv_base) * 2, (size_t(ap.kv_base) + K.region_halves) * 2);
      E.wr(s.O);
      if (A.parts) E.wr(P.parts.b, P.parts.off, P.parts.off + P.parts_bytes);
      E.groups(MTLSizeMake(size_t(c.Hkv), cut[gi + 1] - cut[gi], 1), MTLSizeMake(size_t(32 * G), 1, 1));
    }
    if (!A.combine.empty()) {
      E.pipe("lfm_attention_combine64");
      E.set(0, P.parts);
      E.set(1, s.O);
      E.bytes(2, ap);
      E.set(3, P.comb);
      E.rd(P.parts.b, P.parts.off, P.parts.off + P.parts_bytes);
      E.wr(s.O);
      E.threads(MTLSizeMake(size_t(c.HD / 4), size_t(c.H), A.combine.size() * 8), MTLSizeMake(size_t(c.HD / 4), 4, 1));
    }
  };
  auto kv_save = [&](int layer_k) {
    if (!P.n_saves) return;
    const AttnParams ap{c.H, c.Hkv, qd, kvd, 0, K.bt, uint64_t(K.layer_base(layer_k))};
    E.pipe("kv_save");
    E.set(0, s.Kb);
    E.set(1, s.Vb);
    E.set(2, K.buf);
    E.set(3, P.saves);
    E.bytes(4, ap);
    E.rd(s.Kb);
    E.rd(s.Vb);
    E.wr(K.buf, size_t(ap.kv_base) * 2, (size_t(ap.kv_base) + K.region_halves) * 2);
    E.threads(MTLSizeMake(size_t(kvd / 8), P.n_saves, 1), MTLSizeMake(64, 1, 1));
  };

  for (int li = 0; li < c.L; ++li) {
    const Layer& L = m.layers[size_t(li)];
    const int k = L.index_in_kind;
    // The last layer of a pruned pass continues with the rows read out only
    // (after its mixer saved every row's K/V or conv state).
    const bool last = li == c.L - 1 && P.prune;
    rmsnorm(s.h, L.in_norm, s.xn, M);
    id<MTLBuffer> mix;  // fp16 [M][D] (conv) or [M][H * HD] (attention): the out projection's input
    int mix_w;
    if (L.linear) {
      gemm_runs(E, m.tune, L.in_proj, s.xn, 0, c.D, s.proj, 0, ldc, M, 0);
      const uint64_t conv_base = S.conv_base[size_t(k)];
      const LfmConvParamsHost cp{c.D, ldc, conv_base, S.conv_floats};
      const size_t conv_lo = conv_base * 4, conv_hi = (conv_base + S.conv_floats * size_t(S.n_slots)) * 4;
      E.pipe("lfm_conv");
      E.set(0, s.proj);
      E.set(1, S.buf);
      E.set(2, P.conv_src);
      E.set(3, L.conv_w);
      E.set(4, s.qo);
      E.bytes(5, cp);
      E.set(6, P.row_slot);
      E.rd(s.proj);
      E.rd(S.buf, conv_lo, conv_hi);
      E.wr(s.qo);
      E.threads(MTLSizeMake(size_t(c.D / 4), size_t(M), 1), MTLSizeMake(64, 1, 1));
      if (P.n_conv_jobs) {
        E.pipe("lfm_conv_save");
        E.set(0, s.proj);
        E.set(1, S.buf);
        E.set(2, P.conv_jobs);
        E.bytes(3, cp);
        E.rd(s.proj);
        E.rd(S.buf, conv_lo, conv_hi);
        E.wr(S.buf, conv_lo, conv_hi);
        E.threads(MTLSizeMake(size_t(c.D), P.n_conv_jobs, 1), MTLSizeMake(256, 1, 1));
      }
      mix = s.qo;
      mix_w = c.D;
    } else {
      gemm_runs(E, m.tune, L.in_proj, s.xn, 0, c.D, s.proj, 0, lda, M, 0);
      const LfmPrepParamsHost pp{c.H, c.Hkv, lda, qd, kvd, c.eps, float(1.0 / std::sqrt(double(c.HD)))};
      E.pipe("lfm_attn_prep64");
      E.set(0, s.proj);
      E.set(1, L.q_norm);
      E.set(2, L.k_norm);
      E.set(3, P.pos);
      E.set(4, m.inv_freq);
      E.set(5, s.Q);
      E.set(6, s.Kb);
      E.set(7, s.Vb);
      E.bytes(8, pp);
      E.rd(s.proj);
      E.wr(s.Q);
      E.wr(s.Kb);
      E.wr(s.Vb);
      E.groups(MTLSizeMake(size_t(c.H + 2 * c.Hkv + 3) / 4, size_t(M), 1), MTLSizeMake(128, 1, 1));
      attention(k);
      kv_save(k);
      mix = s.O;
      mix_w = qd;
    }
    if (last) {
      if (N == 0) break;
      gather(mix, s.mixc, mix_w, false);
      gather(s.h, s.hc, c.D, true);
      gemm(E, m.tune, L.out, s.mixc, 0, mix_w, s.hc, 0, c.D, N, 1);
      rmsnorm(s.hc, L.post_norm, s.xc, N);
      shared_mlp_in(m, E, L, s.xc, s.actc, N);
      gemm(E, m.tune, L.down, s.actc, 0, L.I, s.hc, 0, c.D, N, 1);
      break;
    }
    gemm(E, m.tune, L.out, mix, 0, mix_w, s.h, 0, c.D, M, 1);
    rmsnorm(s.h, L.post_norm, s.xn, M);
    shared_mlp_in(m, E, L, s.xn, s.act, M);
    gemm(E, m.tune, L.down, s.act, 0, L.I, s.h, 0, c.D, M, 1);
  }
}

// Kernels the LFM2 path uses (prewarm).
inline void lfm2_pipelines(const Model& m, std::vector<std::pair<std::string, int>>& out) {
  for (const char* k : {"lfm_conv", "lfm_conv_save", "lfm_attn_prep64", "lfm_attention_combine64", "kv_save", "kv_copy_blocks"})
    out.push_back({k, 0});
  out.push_back({lfm2_attention_kernel(m.cfg), 0});
}

}  // namespace espetero::native
