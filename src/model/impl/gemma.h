// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Gemma 4 forward pass: the decoder layers of a forest pass, encoded
// by run_forest (forest.h) after the embedding, with the same rows, trees,
// readouts and KV block lists as the Qwen3.5 path. Per layer:
//
//   x  = rmsnorm(h) * input_layernorm                  (the previous layer's last kernel)
//   q, k, v = x Wq, x Wk, x Wv                         (global K = V layers: v from the k rows)
//   q, k = rope(rmsnorm(q) * q_norm), rope(rmsnorm(k) * k_norm);  v = rmsnorm(v)
//   o  = attention(q, k, v), scale 1                   (sliding layers: the last `window` positions)
//   h += rmsnorm(o Wo) * post_attention_layernorm
//   dense: h = (h + rmsnorm(down(gelu(x2 Wg) * x2 Wu)) * post_feedforward_layernorm) * layer_scalar,
//          x2 = rmsnorm(h) * pre_feedforward_layernorm
//   MoE:   the dense MLP's output t1 and the experts' t2 (input rmsnorm(h) * pre_feedforward_layernorm_2,
//          routed on rmsnorm(h) * router.scale / sqrt(D)) are each normed (_1, _2), added and normed
//          again (post_feedforward_layernorm) before the residual add and the scalar.
//
// Semantics of mlx-lm's models/gemma4_text.py (MIT, THIRD_PARTY_NOTICES.md).
// Sliding layers keep K/V in per-slot rings (Model::kvs); global layers use
// the paged pool. The logits are soft-capped by run_forest after the LM head.
#pragma once

namespace espetero::native {

struct G4TileRec {
  int32_t tree, r0, k0, k1, part, last, b0, pad;
};
struct G4AttnParamsHost {
  int32_t H, Hkv, q_ld, kv_ld, bt, window, ring, pad;
  uint64_t kv_base;
};
struct G4PrepParamsHost {
  int32_t H, Hkv, ld, k_off, v_off, q_ld, kv_ld;
  float eps;
};
struct G4NormParamsHost {
  int32_t D;
  float eps, scalar;
  int32_t nout;
};
struct G4MoeOutParamsHost {
  int32_t D, k;
  float eps, scalar;
  int32_t nout;
};
struct G4RouteParamsHost {
  int32_t E, k, build, pad;
};

// The attention work of one layer kind over the pass's trees.
struct G4Plan {
  std::vector<AttnTreeRec> trees;  // table: the parent's block list offset (global) or first ring block (sliding)
  std::vector<G4TileRec> tiles;
  std::vector<AttnCombineRec> combine;
  int parts = 0;
};

// Tiles of 8 query rows per tree. Sliding layers (window > 0) read only the
// cached keys and batch rows the tile's earliest row can still see (cached
// key j and root row j sit at positions j and P + j; leaf rows are never
// earlier than their index says, so skipping by index is safe). Few-tile
// passes split the cached range (flash decoding), as plan_tiles does.
inline void g4_plan_tiles(G4Plan& A, int groups, int split, int window, const int32_t* pos) {
  A.tiles.clear();
  A.combine.clear();
  A.parts = 0;
  int n_tiles = 0;
  for (const auto& t : A.trees) n_tiles += (t.M + 7) / 8;
  const int budget = std::max(1, 256 / std::max(1, groups * n_tiles));
  for (int ti = 0; ti < int(A.trees.size()); ++ti) {
    const AttnTreeRec& t = A.trees[size_t(ti)];
    for (int r = 0; r < t.M; r += 8) {
      int k0 = 0, b0 = 0;
      if (window > 0) {
        int minpos = INT32_MAX;
        for (int i = r; i < std::min(t.M, r + 8); ++i) minpos = std::min(minpos, pos[4 * (t.base + i)]);
        k0 = std::clamp(minpos - window + 1, 0, t.P);
        b0 = std::max(0, minpos - window + 1 - t.P);
      }
      const int span = t.P - k0;
      int ns = 1;
      if (split > 0 && budget > 1 && span >= 64) {
        const int chunk = std::max(32, std::min(split, ((span + budget - 1) / budget + 31) / 32 * 32));
        ns = std::min({(span + chunk - 1) / chunk, budget, 64});
      }
      if (ns == 1) {
        A.tiles.push_back({ti, t.base + r, k0, t.P, -1, 1, b0, 0});
        continue;
      }
      const int per = (span + ns - 1) / ns;
      const int step = (per + 31) / 32 * 32;
      const int part0 = A.parts;
      int used = 0;
      for (int k = k0; k < t.P; k += step, ++used) {
        const int k1 = std::min(t.P, k + step);
        A.tiles.push_back({ti, t.base + r, k, k1, A.parts++, k1 >= t.P ? 1 : 0, b0, 0});
      }
      A.combine.push_back({t.base + r, std::min(8, t.M - r), part0, used});
    }
  }
}

// Kernel geometry of a layer kind: query heads per threadgroup and
// threadgroups per KV head (the global layers' 512-dim heads run 2 per
// threadgroup: 8 simdgroups).
struct G4Geom {
  int hd, hkv, G, Gk, split;
  std::string kernel, combine;
};
inline G4Geom g4_geom(const Config& c, bool sliding) {
  G4Geom g;
  g.hd = sliding ? c.HD : c.HDg;
  g.hkv = sliding ? c.Hkv : c.Hkv_g;
  g.G = c.H / g.hkv;
  if (sliding) {
    g.Gk = g.G;
    g.split = 1;
    g.kernel = "g4_attention256_g" + std::to_string(g.G);
    g.combine = "g4_attention_combine256";
  } else {
    g.Gk = 2;
    g.split = g.G / 2;
    g.kernel = "g4_attention512_s" + std::to_string(g.split);
    g.combine = "g4_attention_combine512";
  }
  return g;
}

// Passes whose trees are all single rows (decode steps) run g4_attention_dec
// for the layer kinds with at most 8 query heads per KV head.
template <class RowsOf>
inline bool g4_dec(const Model& m, size_t n_trees, RowsOf rows_of) {
  bool single = m.attn_gqa && n_trees > 0;
  for (size_t i = 0; i < n_trees; ++i) single = single && rows_of(i) == 1;
  return single;
}

// Floats of the split-attention partial records of a plan.
inline size_t g4_parts_floats(const Config& c, const G4Plan& A, bool sliding) {
  const G4Geom g = g4_geom(c, sliding);
  return size_t(A.parts) * size_t(g.hkv * g.split) * 8 * size_t(g.Gk) * size_t(g.hd + 2);
}

// Everything the layer loop reads from run_forest.
struct G4PassIn {
  int M = 0, N = 0;
  bool prune = false, chained = false;
  const G4Plan* plan_s = nullptr;  // sliding layers
  const G4Plan* plan_g = nullptr;  // global layers
  Buf trees_s, tiles_s, comb_s, trees_g, tiles_g, comb_g, parts;
  size_t parts_bytes = 0;
  Buf saves_s, saves_g;  // KVSaveRec entries: rings (sliding), paged pool (global)
  size_t n_saves_s = 0, n_saves_g = 0;
  Buf blocks, desc, pos, readout_rows;
};

inline void g4_encode_layers(Model& m, Enc& E, const G4PassIn& P) {
  const Config& c = m.cfg;
  Scratch& s = m.sc;
  const int M = P.M, N = P.N;
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
  // Forest attention of one layer over its kind's plan, writing s.O.
  // Long prompt passes at long contexts go in groups of tiles, each committed
  // on its own (as run_forest's attention does: macOS aborts command
  // buffers of seconds while the display is in use); same tiles, same order.
  static const size_t split_rk = [] {
    const char* e = std::getenv("ESPETERO_ATTN_SPLIT");
    return size_t(e && *e ? std::atoi(e) : 16) << 20;
  }();
  auto attention = [&](bool sliding, int layer_k) {
    const G4Plan& A = sliding ? *P.plan_s : *P.plan_g;
    if (A.tiles.empty()) return;
    const G4Geom g = g4_geom(c, sliding);
    const KVPool& K = sliding ? m.kvs : m.kv;
    // Single-row trees: every query head of a KV head in one threadgroup (the
    // combine then reads records of G heads per KV head).
    const bool dec = g.G <= 8 && g4_dec(m, A.trees.size(), [&](size_t i) { return A.trees[i].M; });
    const int groups = dec ? g.hkv : g.hkv * g.split;
    const G4AttnParamsHost ap{c.H, groups, c.H * g.hd, g.hkv * g.hd, K.bt, sliding ? c.window : 0,
                              sliding ? m.ring_blocks : 0, 0, uint64_t(K.layer_base(layer_k))};
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
      E.pipe(dec ? "g4_attention_dec" + std::to_string(g.hd) + "_g" + std::to_string(g.G) : g.kernel);
      E.set(0, s.Q);
      E.set(1, s.Kb);
      E.set(2, s.Vb);
      E.set(3, K.buf);
      E.set(4, P.blocks);
      E.set(5, P.desc);
      E.set(6, P.pos);
      E.set(7, s.O);
      E.bytes(8, ap);
      E.set(9, sliding ? P.tiles_s : P.tiles_g, cut[gi] * sizeof(G4TileRec));
      E.set(10, sliding ? P.trees_s : P.trees_g);
      E.set(11, P.parts);
      E.rd(s.Q);
      E.rd(s.Kb);
      E.rd(s.Vb);
      E.rd(K.buf, size_t(ap.kv_base) * 2, (size_t(ap.kv_base) + K.region_halves) * 2);
      E.wr(s.O);
      if (A.parts) E.wr(P.parts.b, P.parts.off, P.parts.off + P.parts_bytes);
      E.groups(MTLSizeMake(size_t(groups), cut[gi + 1] - cut[gi], 1),
               MTLSizeMake(size_t(dec ? 32 * (g.hd / 128) * 2 : 32 * (g.hd / 128) * g.Gk), 1, 1));
    }
    if (!A.combine.empty()) {
      E.pipe(g.combine);
      E.set(0, P.parts);
      E.set(1, s.O);
      E.bytes(2, ap);
      E.set(3, sliding ? P.comb_s : P.comb_g);
      E.rd(P.parts.b, P.parts.off, P.parts.off + P.parts_bytes);
      E.wr(s.O);
      E.threads(MTLSizeMake(size_t(g.hd / 4), size_t(c.H), A.combine.size() * 8), MTLSizeMake(64, 1, 1));
    }
  };
  auto kv_save = [&](bool sliding, int layer_k) {
    const size_t n = sliding ? P.n_saves_s : P.n_saves_g;
    if (!n) return;
    const G4Geom g = g4_geom(c, sliding);
    const KVPool& K = sliding ? m.kvs : m.kv;
    const int kvd = g.hkv * g.hd;
    const AttnParams ap{c.H, g.hkv, c.H * g.hd, kvd, 0, K.bt, uint64_t(K.layer_base(layer_k))};
    E.pipe("kv_save");
    E.set(0, s.Kb);
    E.set(1, s.Vb);
    E.set(2, K.buf);
    E.set(3, sliding ? P.saves_s : P.saves_g);
    E.bytes(4, ap);
    E.rd(s.Kb);
    E.rd(s.Vb);
    E.wr(K.buf, size_t(ap.kv_base) * 2, (size_t(ap.kv_base) + K.region_halves) * 2);
    E.threads(MTLSizeMake(size_t(kvd / 8), n, 1), MTLSizeMake(64, 1, 1));
  };
  // GeGLU MLP of `rows` rows: X (fp16) -> act (fp16 [rows, I]) -> T (fp32 [rows, D], stored).
  auto mlp = [&](const Layer& L, id<MTLBuffer> X, id<MTLBuffer> act, id<MTLBuffer> T, int rows) {
    if (L.gate_up.out() > 0) {
      gemm(E, m.tune, L.gate_up, X, 0, c.D, act, 0, L.I, rows, 2);
    } else {
      gemm(E, m.tune, L.gate, X, 0, c.D, s.g32, 0, L.I, rows, 0);
      gemm(E, m.tune, L.up, X, 0, c.D, s.u32, 0, L.I, rows, 0);
      E.pipe("geglu_f32");
      E.set(0, s.g32);
      E.set(1, s.u32);
      E.set(2, act);
      E.bytes(3, L.I);
      E.rd(s.g32);
      E.rd(s.u32);
      E.wr(act);
      E.threads(MTLSizeMake(size_t(L.I / 4), size_t(rows), 1), MTLSizeMake(64, 1, 1));
    }
    gemm(E, m.tune, L.down, act, 0, L.I, T, 0, c.D, rows, 0);
  };
  // The experts of an MoE layer: router logits -> top-k -> tile table ->
  // grouped GeGLU gate/up and down GEMVs (moe_gemv / moe_gemv8, the Qwen3.5
  // MoE kernels without a shared expert) -> z.ys, combined by g4_moe_out.
  auto experts = [&](const Layer& L, id<MTLBuffer> Xe, id<MTLBuffer> Xr, int rows) {
    ensure_moe_scratch(m, rows);
    MoeScratch& z = m.mz;
    gemm(E, m.tune, L.router, Xr, 0, c.D, z.logits, 0, c.E, rows, 0);
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
      if (fma_path) {  // the folded route's buffers: unread (DIAG 0)
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
    grouped(L.experts_gu, Xe, c.D, 1, z.act, c.Ie, 2 * c.Ie, 2);
    grouped(L.experts_down, z.act, c.Ie, 0, z.ys, c.D, c.D, 0);
  };

  rmsnorm(s.h, m.layers[0].in_norm, s.xn, M);
  for (int li = 0; li < c.L; ++li) {
    const Layer& L = m.layers[size_t(li)];
    const bool sw = L.swa;
    const G4Geom g = g4_geom(c, sw);
    const int qd = c.H * g.hd, kvd = g.hkv * g.hd, ld = c.g4_ld(sw);
    const bool has_v = sw || !c.k_eq_v;
    const int k = L.index_in_kind;
    gemm_runs(E, m.tune, L.in_proj, s.xn, 0, c.D, s.proj, 0, ld, M, 0);
    {
      const G4PrepParamsHost pp{c.H, g.hkv, ld, qd, has_v ? qd + kvd : qd, qd, kvd, c.eps};
      E.pipe(sw ? "g4_attn_prep256" : "g4_attn_prep512");
      E.set(0, s.proj);
      E.set(1, L.q_norm);
      E.set(2, L.k_norm);
      E.set(3, P.pos);
      E.set(4, sw ? m.inv_freq : m.inv_freq_g);
      E.set(5, s.Q);
      E.set(6, s.Kb);
      E.set(7, s.Vb);
      E.bytes(8, pp);
      E.rd(s.proj);
      E.wr(s.Q);
      E.wr(s.Kb);
      E.wr(s.Vb);
      E.groups(MTLSizeMake(size_t(c.H + 2 * g.hkv + 3) / 4, size_t(M), 1), MTLSizeMake(128, 1, 1));
    }
    attention(sw, k);
    kv_save(sw, k);
    // The last layer of a pruned pass continues with the rows read out only.
    const bool last = li == c.L - 1 && P.prune;
    int rows = M;
    id<MTLBuffer> Ob = s.O, Hb = s.h, Xb = s.xn, Ab = s.act;
    if (last) {
      if (N == 0) break;
      gather(s.O, s.mixc, qd, false);
      gather(s.h, s.hc, c.D, true);
      rows = N, Ob = s.mixc, Hb = s.hc, Xb = s.xc, Ab = s.actc;
    }
    gemm(E, m.tune, L.out, Ob, 0, qd, s.proj, 0, c.D, rows, 0);
    {
      const G4NormParamsHost np{c.D, c.eps, 1.0f, L.moe ? 3 : 1};
      E.pipe("g4_post_attn");
      E.set(0, s.proj);
      E.set(1, Hb);
      E.set(2, L.post_norm);
      E.set(3, L.pre_ffn);
      E.set(4, L.moe ? L.pre_ffn2 : L.pre_ffn);
      E.set(5, L.moe ? L.router_norm : L.pre_ffn);
      E.set(6, Xb);
      E.set(7, L.moe ? s.g_xn2 : Xb);
      E.set(8, L.moe ? s.g_xr : Xb);
      E.bytes(9, np);
      E.rd(s.proj);
      E.rd(Hb);
      E.wr(Hb);
      E.wr(Xb);
      if (L.moe) {
        E.wr(s.g_xn2);
        E.wr(s.g_xr);
      }
      E.groups(MTLSizeMake(size_t(rows), 1, 1), MTLSizeMake(256, 1, 1));
    }
    mlp(L, Xb, Ab, s.proj, rows);
    const bool next = li + 1 < c.L && !last;
    const Buf& w_next = next ? m.layers[size_t(li + 1)].in_norm : L.post_ffn;
    if (L.moe) {
      experts(L, s.g_xn2, s.g_xr, rows);
      MoeScratch& z = m.mz;
      const G4MoeOutParamsHost op{c.D, c.k, c.eps, L.scalar, next ? 1 : 0};
      E.pipe("g4_moe_out");
      E.set(0, s.proj);
      E.set(1, z.ys);
      E.set(2, z.slotpos);
      E.set(3, z.w);
      E.set(4, z.shared);
      E.set(5, Hb);
      E.set(6, L.post_ffn1);
      E.set(7, L.post_ffn2);
      E.set(8, L.post_ffn);
      E.set(9, w_next);
      E.set(10, s.xn);
      E.bytes(11, op);
      E.rd(s.proj);
      E.rd(z.ys);
      E.rd(z.slotpos);
      E.rd(z.w);
      E.wr(z.shared);
      E.rd(Hb);
      E.wr(Hb);
      if (next) E.wr(s.xn);
      E.groups(MTLSizeMake(size_t(rows), 1, 1), MTLSizeMake(256, 1, 1));
    } else {
      const G4NormParamsHost np{c.D, c.eps, L.scalar, next ? 1 : 0};
      E.pipe("g4_post_ffn");
      E.set(0, s.proj);
      E.set(1, Hb);
      E.set(2, L.post_ffn);
      E.set(3, w_next);
      E.set(4, s.xn);
      E.bytes(5, np);
      E.rd(s.proj);
      E.rd(Hb);
      E.wr(Hb);
      if (next) E.wr(s.xn);
      E.groups(MTLSizeMake(size_t(rows), 1, 1), MTLSizeMake(256, 1, 1));
    }
    if (last) break;
  }
}

// Kernels the Gemma 4 path uses (prewarm).
inline void g4_pipelines(const Model& m, std::vector<std::pair<std::string, int>>& out) {
  const Config& c = m.cfg;
  for (const char* k : {"g4_scale_rows", "softcap_rows", "g4_attn_prep256", "g4_attn_prep512", "g4_post_attn", "g4_post_ffn",
                        "g4_attention_combine256", "g4_attention_combine512", "kv_save", "kv_copy_blocks", "geglu_f32"})
    out.push_back({k, 0});
  out.push_back({g4_geom(c, true).kernel, 0});
  out.push_back({g4_geom(c, false).kernel, 0});
  for (bool sliding : {true, false}) {
    const G4Geom g = g4_geom(c, sliding);
    if (g.G <= 8) out.push_back({"g4_attention_dec" + std::to_string(g.hd) + "_g" + std::to_string(g.G), 0});
  }
  if (c.E) {
    for (const char* k : {"g4_route", "g4_moe_out", "moe_sort"}) out.push_back({k, 0});
    // Every layer: formats may differ by layer (unsloth's 26B-A4B UD-Q4_K_M:
    // Q5_1 down projections, Q8_0 in one layer); the prewarm drops repeats.
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
}

}  // namespace espetero::native
