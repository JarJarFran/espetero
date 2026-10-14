// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Qwen3-Omni thinker forward pass (`qwen3_omni_moe`, the backbone of
// Cloudflare's Clef-Omni): the decoder layers of a forest pass, encoded by
// run_forest (forest.h) after the embedding and the vision tower, with the
// same rows, trees, readouts and paged KV block lists as the Qwen3.5 path.
// Every layer is full attention + mixture of experts:
//
//   x  = rmsnorm(h) * input_layernorm                     (the previous layer's last kernel)
//   q, k, v = x Wq, x Wk, x Wv
//   q, k = mrope(rmsnorm(q) * q_norm), mrope(rmsnorm(k) * k_norm)   (interleaved sections, whole head)
//   h += attention(q / sqrt(HD), k, v) Wo
//   x2 = rmsnorm(h) * post_attention_layernorm
//   h += sum over the top k of softmax(x2 Wr) (renormalized) of down(silu(x2 Wg) * x2 Wu)
//   layers 0 .. n - 1 of an image pass: h[image rows] += DeepStack feature n of the tower
//
// Semantics of transformers' models/qwen3_omni_moe (Apache 2.0,
// THIRD_PARTY_NOTICES.md). Attention reuses Gemma 4's forest kernels for
// 128-dim heads (gemma.h's planner, window 0), routing g4_route with unit
// expert scales, the experts the grouped MoE GEMVs (15_moe.metal, SwiGLU).
// There is no MTP head, no verify-commit and no shared expert.
#pragma once

namespace espetero::native {

struct Q3PrepParamsHost {
  int32_t H, Hkv, ld, k_off, v_off, q_ld, kv_ld;
  float eps, qscale;
};
struct Q3MoeOutParamsHost {
  int32_t D, k;
  float eps;
  int32_t nout;
};

// Kernel geometry: all G query heads of a KV head per threadgroup.
inline std::string q3m_kernel(const Config& c, bool dec) {
  return std::string(dec ? "g4_attention_dec128_g" : "g4_attention128_g") + std::to_string(c.H / c.Hkv);
}

inline size_t q3m_parts_floats(const Config& c, const G4Plan& A) {
  return size_t(A.parts) * size_t(c.Hkv) * 8 * size_t(c.H / c.Hkv) * size_t(c.HD + 2);
}

// Everything the layer loop reads from run_forest.
struct Q3PassIn {
  int M = 0, N = 0;
  bool prune = false, chained = false;
  const G4Plan* plan = nullptr;
  Buf trees, tiles, comb, parts;
  size_t parts_bytes = 0;
  Buf saves;  // KVSaveRec entries (paged pool)
  size_t n_saves = 0;
  Buf blocks, desc, pos, readout_rows;
  int vis_rows = 0;  // merged image rows of the pass (DeepStack targets: m.vs.dst_rows)
};

inline void q3m_encode_layers(Model& m, Enc& E, const Q3PassIn& P) {
  const Config& c = m.cfg;
  Scratch& s = m.sc;
  const int M = P.M, N = P.N;
  const int qd = c.H * c.HD, kvd = c.kv_dim(), ld = qd + 2 * kvd, G = c.H / c.Hkv;
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
  // Forest attention of one layer, writing s.O. Long prompt passes at long
  // contexts go in groups of tiles, each committed on its own (as gemma.h
  // does: macOS aborts command buffers of seconds while the display is in
  // use); same tiles, same order.
  static const size_t split_rk = [] {
    const char* e = std::getenv("ESPETERO_ATTN_SPLIT");
    return size_t(e && *e ? std::atoi(e) : 16) << 20;
  }();
  auto attention = [&](int layer_k) {
    const G4Plan& A = *P.plan;
    if (A.tiles.empty()) return;
    const KVPool& K = m.kv;
    const bool dec = g4_dec(m, A.trees.size(), [&](size_t i) { return A.trees[i].M; });
    const G4AttnParamsHost ap{c.H, c.Hkv, qd, kvd, K.bt, 0, 0, 0, uint64_t(K.layer_base(layer_k))};
    std::vector<size_t> cut{0};
    if (split_rk > 0 && !dec && !P.chained && !m.chain_active && M >= 64) {
      size_t acc = 0;
      for (size_t i = 0; i < A.tiles.size(); ++i) {
        const G4TileRec& tr = A.tiles[i];
        const size_t w = size_t(8) * (size_t(tr.k1 - tr.k0) + size_t(A.trees[size_t(tr.tree)].M));
        if (acc > 0 && acc + w > split_rk) cut.push_back(i), acc = 0;
        acc += w;
      }
    }
    cut.push_back(A.tiles.size());
    for (size_t gi = 0; gi + 1 < cut.size(); ++gi) {
      if (gi > 0) E.flush();
      E.pipe(q3m_kernel(c, dec));
      E.set(0, s.Q);
      E.set(1, s.Kb);
      E.set(2, s.Vb);
      E.set(3, K.buf);
      E.set(4, P.blocks);
      E.set(5, P.desc);
      E.set(6, P.pos);
      E.set(7, s.O);
      E.bytes(8, ap);
      E.set(9, P.tiles, cut[gi] * sizeof(G4TileRec));
      E.set(10, P.trees);
      E.set(11, P.parts);
      E.rd(s.Q);
      E.rd(s.Kb);
      E.rd(s.Vb);
      E.rd(K.buf, size_t(ap.kv_base) * 2, (size_t(ap.kv_base) + K.region_halves) * 2);
      E.wr(s.O);
      if (A.parts) E.wr(P.parts.b, P.parts.off, P.parts.off + P.parts_bytes);
      E.groups(MTLSizeMake(size_t(c.Hkv), cut[gi + 1] - cut[gi], 1), MTLSizeMake(size_t(dec ? 64 : 32 * G), 1, 1));
    }
    if (!A.combine.empty()) {
      E.pipe("g4_attention_combine128");
      E.set(0, P.parts);
      E.set(1, s.O);
      E.bytes(2, ap);
      E.set(3, P.comb);
      E.rd(P.parts.b, P.parts.off, P.parts.off + P.parts_bytes);
      E.wr(s.O);
      E.threads(MTLSizeMake(size_t(c.HD / 4), size_t(c.H), A.combine.size() * 8), MTLSizeMake(32, 1, 1));
    }
  };
  auto kv_save = [&](int layer_k) {
    if (!P.n_saves) return;
    const KVPool& K = m.kv;
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
  // The experts of a layer (gemma.h's path without the dense MLP): router
  // logits -> top-k softmax (g4_route, unit expert scales) -> tile table ->
  // grouped SwiGLU gate/up and down GEMVs -> z.ys.
  auto experts = [&](const Layer& L, id<MTLBuffer> X, int rows) {
    ensure_moe_scratch(m, rows);
    MoeScratch& z = m.mz;
    gemm(E, m.tune, L.router, X, 0, c.D, z.logits, 0, c.E, rows, 0);
    const bool direct = rows == 1 && m.moe_direct;
    const G4RouteParamsHost rp{c.E, c.k, direct ? 1 : 0, 0};
    E.pipe("g4_route");
    E.set(0, z.logits);
    E.set(1, L.expert_scale);
    E.set(2, z.ids);
    E.set(3, z.w);
    E.set(4, z.tiles);
    E.set(5, z.rowmap);
    E.set(6, z.slotpos);
    E.bytes(7, rp);
    E.rd(z.logits);
    E.wr(z.ids);
    E.wr(z.w);
    if (direct) {
      E.wr(z.tiles);
      E.wr(z.rowmap);
      E.wr(z.slotpos);
    }
    E.groups(MTLSizeMake(size_t(rows), 1, 1), MTLSizeMake(size_t(std::max(32, (c.E + 31) / 32 * 32)), 1, 1));
    const int gu_blocks = (2 * c.Ie + 31) / 32, down_blocks = (c.D + 31) / 32;
    if (!direct) {
      const MoeSortParams sp{c.E, c.k, rows, gu_blocks, down_blocks, 0, 0};
      E.pipe("moe_sort");
      E.set(0, z.ids);
      E.set(1, z.offsets);
      E.set(2, z.tiles);
      E.set(3, z.ntiles);
      E.set(4, z.rowmap);
      E.set(5, z.slotpos);
      E.bytes(6, sp);
      E.set(7, z.ids);
      E.rd(z.ids);
      E.wr(z.offsets);
      E.wr(z.tiles);
      E.wr(z.ntiles);
      E.wr(z.rowmap);
      E.wr(z.slotpos);
      E.groups(MTLSizeMake(1, 1, 1), MTLSizeMake(1024, 1, 1));
    }
    auto grouped = [&](const QW& w, id<MTLBuffer> A, int lda, int gather_rows, id<MTLBuffer> Y, int ldy, int Ne, int epi) {
      const bool fma_path = m.moe_fma && rows <= m.moe_fma_rows;
      const int ks_opt = epi == 2 ? m.moe_ks : m.moe_ks_dn;
      const int ks = fma_path ? (ks_opt == 4 || (ks_opt == 2 && rows <= 2) ? ks_opt : 8) : 8;
      const std::string name = moe_gemv_name(w, nullptr, fma_path, rows, epi, false, ks);
      E.pipe(name, w.fc());
      gemm_tag(E, name, w.out, w.in, rows);
      const int nbpt = (fma_path && direct) ? std::max(1, std::min(epi == 2 ? m.moe_nbpt : m.moe_nbpt_dn, Ne / 32)) : 1;
      const MoeGemmParams gp{Ne, w.out, w.in, lda, ldy, gather_rows, -1, 0, direct ? 1 : 0, nbpt, 0, 0, 0, 0, 0, 0};
      E.set(0, A);
      E.set(1, w.wr);
      E.set(2, w.sb);
      E.set(3, w.wh ? w.wh : w.wr);
      E.set(4, Y);
      E.bytes(5, gp);
      E.set(6, z.tiles);
      E.set(7, z.ntiles);
      E.set(8, z.rowmap);
      E.set(9, w.wr);  // no shared expert (shared_e = -1): unread
      E.set(10, w.sb);
      E.set(11, w.wh ? w.wh : w.wr);
      if (fma_path) {  // the folded route's buffers: unread
        E.set(12, z.rparts);
        E.set(13, z.ids);
        E.set(14, z.w);
        E.set(15, z.sgate);
        E.set(16, z.slotpos);
      } else {  // staged records and loading flags: unread (st_count 0, split 0)
        E.set(12, w.wr);
        E.set(13, w.sb);
        E.set(14, w.wh ? w.wh : w.wr);
        E.set(15, z.tiles);
      }
      E.rd(A);
      E.rd(z.tiles);
      E.rd(z.rowmap);
      if (!direct) E.rd(z.ntiles);
      E.wr(Y);
      if (direct) {
        const size_t nb = size_t((Ne / 32 + nbpt - 1) / nbpt);
        E.groups(MTLSizeMake(size_t(c.k), nb, 1), MTLSizeMake(size_t(32 * ks), 1, 1));
      } else {
        E.groups_indirect(z.ntiles, epi == 2 ? 0 : 12, MTLSizeMake(size_t(32 * ks), 1, 1));
      }
    };
    grouped(L.experts_gu, X, c.D, 1, z.act, c.Ie, 2 * c.Ie, 2);
    grouped(L.experts_down, z.act, c.Ie, 0, z.ys, c.D, c.D, 0);
  };
  // DeepStack: feature j of the tower ([vis_rows][D] in m.vs.ds) into the image rows.
  const int n_ds = P.vis_rows > 0 ? int(m.vis.ds.size()) : 0;
  auto deepstack = [&](int j, id<MTLBuffer> Hb) {
    const VScratch& v = m.vs;
    const size_t off = size_t(j) * size_t(P.vis_rows) * size_t(c.D) * 4;
    E.pipe("scatter_add_rows_f32");
    E.set(0, v.ds, off);
    E.set(1, Hb);
    E.set(2, v.dst_rows);
    E.bytes(3, c.D);
    E.rd(v.ds, off, off + size_t(P.vis_rows) * size_t(c.D) * 4);
    E.rd(v.dst_rows);
    E.rd(Hb);
    E.wr(Hb);
    E.threads(MTLSizeMake(size_t(c.D / 4), size_t(P.vis_rows), 1), MTLSizeMake(64, 1, 1));
  };

  // Development: ESPETERO_Q3M_LAYERS=n runs only the first n layers (the final norm and
  // readouts follow as usual), for layer-by-layer parity against a truncated reference.
  static const int env_layers = [] {
    const char* e = std::getenv("ESPETERO_Q3M_LAYERS");
    return e && *e ? std::max(1, std::atoi(e)) : 0;
  }();
  const int n_layers = env_layers > 0 ? std::min(env_layers, c.L) : c.L;
  rmsnorm(s.h, m.layers[0].in_norm, s.xn, M);
  for (int li = 0; li < n_layers; ++li) {
    const Layer& L = m.layers[size_t(li)];
    gemm_runs(E, m.tune, L.in_proj, s.xn, 0, c.D, s.proj, 0, ld, M, 0);
    {
      const Q3PrepParamsHost pp{c.H, c.Hkv, ld, qd, qd + kvd, qd, kvd, c.eps, float(1.0 / std::sqrt(double(c.HD)))};
      E.pipe("q3m_attn_prep128");
      E.set(0, s.proj);
      E.set(1, L.q_norm);
      E.set(2, L.k_norm);
      E.set(3, P.pos);
      E.set(4, m.inv_freq);
      E.set(5, m.rope_sel);
      E.set(6, s.Q);
      E.set(7, s.Kb);
      E.set(8, s.Vb);
      E.bytes(9, pp);
      E.rd(s.proj);
      E.wr(s.Q);
      E.wr(s.Kb);
      E.wr(s.Vb);
      E.groups(MTLSizeMake(size_t(c.H + 2 * c.Hkv + 3) / 4, size_t(M), 1), MTLSizeMake(128, 1, 1));
    }
    attention(L.index_in_kind);
    kv_save(L.index_in_kind);
    // The last layer of a pruned pass continues with the rows read out only.
    const bool last = li == n_layers - 1 && P.prune;
    int rows = M;
    id<MTLBuffer> Ob = s.O, Hb = s.h, Xb = s.xn;
    if (last) {
      if (N == 0) break;
      gather(s.O, s.mixc, qd, false);
      gather(s.h, s.hc, c.D, true);
      rows = N, Ob = s.mixc, Hb = s.hc, Xb = s.xc;
    }
    gemm(E, m.tune, L.out, Ob, 0, qd, Hb, 0, c.D, rows, 1);
    rmsnorm(Hb, L.post_norm, Xb, rows);
    experts(L, Xb, rows);
    const bool next = li + 1 < n_layers && !last;
    const bool ds = li < n_ds && !last;
    {
      MoeScratch& z = m.mz;
      const Q3MoeOutParamsHost op{c.D, c.k, c.eps, next && !ds ? 1 : 0};
      E.pipe("q3m_moe_out");
      E.set(0, z.ys);
      E.set(1, z.slotpos);
      E.set(2, z.w);
      E.set(3, Hb);
      E.set(4, next ? m.layers[size_t(li + 1)].in_norm : L.post_norm);
      E.set(5, s.xn);
      E.bytes(6, op);
      E.rd(z.ys);
      E.rd(z.slotpos);
      E.rd(z.w);
      E.rd(Hb);
      E.wr(Hb);
      if (op.nout) E.wr(s.xn);
      E.groups(MTLSizeMake(size_t(rows), 1, 1), MTLSizeMake(256, 1, 1));
    }
    if (ds) {
      deepstack(li, Hb);
      if (next) rmsnorm(Hb, m.layers[size_t(li + 1)].in_norm, s.xn, rows);
    }
    if (last) break;
  }
}

// Kernels the Qwen3-Omni path uses (prewarm).
inline void q3m_pipelines(const Model& m, std::vector<std::pair<std::string, int>>& out) {
  const Config& c = m.cfg;
  for (const char* k : {"q3m_attn_prep128", "q3m_moe_out", "scatter_add_rows_f32", "g4_attention_combine128", "g4_route",
                        "moe_sort", "kv_save", "rmsnorm", "gather_rows_f32", "gather_rows_f16"})
    out.push_back({k, 0});
  out.push_back({q3m_kernel(c, false), 0});
  out.push_back({q3m_kernel(c, true), 0});
  for (const Layer& L : m.layers) {
    if (!L.moe) continue;
    const QW& gu = L.experts_gu;
    const QW& dn = L.experts_down;
    out.push_back({moe_gemv_name(gu, nullptr, false, 8, 2, false, 8), gu.fc()});
    out.push_back({moe_gemv_name(dn, nullptr, false, 8, 0, false, 8), dn.fc()});
    for (int mm : {1, 2})
      for (int ks : {8, 4, 2}) {
        out.push_back({moe_gemv_name(gu, nullptr, true, mm, 2, false, ks), gu.fc()});
        out.push_back({moe_gemv_name(dn, nullptr, true, mm, 0, false, ks), dn.fc()});
      }
  }
}

}  // namespace espetero::native
