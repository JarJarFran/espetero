// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ---------------------------------------------------------------------------
// Mixture-of-experts MLP (Qwen3.5-MoE: Qwen3.6-35B-A3B): softmax router,
// top-k experts per row renormalized, SwiGLU experts, plus a sigmoid-gated
// shared expert.
//
// The E experts of one projection form one packed matrix: expert e owns the
// output columns [e * Ne, (e + 1) * Ne), so the quantized layouts, scale
// formats and repack of 10_gemm.metal serve them unchanged. Rows are
// counting-sorted by expert on the GPU; a tile table lists (expert, first
// sorted row, rows <= 8) and the grouped GEMV gathers each tile's rows and
// computes its expert's columns. A row's result does not depend on where the
// sort puts it, and the combine sums a row's experts in slot order:
// deterministic. Ported from the JEVfire engine (checked there against
// mlx-vlm on the 8-bit 35B-A3B).
// ---------------------------------------------------------------------------

struct MoeRouteParams {
  int E;      // experts
  int k;      // experts per row
  int M;      // rows
};

// One threadgroup per row, one thread per expert (E <= 1024): softmax over
// the router logits, the k largest (ties to the lower index), renormalized.
kernel void moe_topk(
    device const float* logits [[buffer(0)]],
    device int* ids [[buffer(1)]],
    device float* w [[buffer(2)]],
    constant MoeRouteParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint ntg [[threads_per_threadgroup]]) {
  threadgroup float redf[32];
  threadgroup int redi[32];
  threadgroup float picked_p[16];
  threadgroup int picked_i[16];
  const int e = int(tid);
  float x = e < p.E ? logits[ulong(row) * p.E + e] : -INFINITY;
  float m = simd_max(x);
  if (lane == 0) redf[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float t = lane < (ntg + 31) / 32 ? redf[lane] : -INFINITY;
    t = simd_max(t);
    if (lane == 0) redf[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = redf[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ex = e < p.E ? exp(x - m) : 0.0f;
  float s = simd_sum(ex);
  if (lane == 0) redf[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float t = lane < (ntg + 31) / 32 ? redf[lane] : 0.0f;
    t = simd_sum(t);
    if (lane == 0) redf[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total = redf[0];
  float prob = ex / total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // k rounds of arg-max (value desc, index asc), removing the winner.
  float mine = e < p.E ? prob : -1.0f;
  for (int r = 0; r < p.k; ++r) {
    float bv = mine;
    int bi = e;
    for (ushort off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_down(bv, off);
      const int oi = simd_shuffle_down(bi, off);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if (lane == 0) redf[sg] = bv, redi[sg] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      const bool live = lane < (ntg + 31) / 32;
      float v = live ? redf[lane] : -2.0f;
      int i = live ? redi[lane] : 0x7fffffff;
      for (ushort off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(v, off);
        const int oi = simd_shuffle_down(i, off);
        if (ov > v || (ov == v && oi < i)) v = ov, i = oi;
      }
      if (lane == 0) picked_p[r] = v, picked_i[r] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == picked_i[r]) mine = -1.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    float sum = 0.0f;
    for (int r = 0; r < p.k; ++r) sum += picked_p[r];
    for (int r = 0; r < p.k; ++r) {
      ids[ulong(row) * p.k + r] = picked_i[r];
      w[ulong(row) * p.k + r] = picked_p[r] / sum;
    }
  }
}

// Counting sort of the M * k (row, slot) pairs by expert, and the tile table:
// one threadgroup, counting in threadgroup memory (no device-scope barriers).
// Also writes the indirect dispatch arguments of the two grouped GEMVs
// (tiles x column blocks), so their grids are exactly the tiles that exist.
struct MoeSortParams {
  int E;
  int k;
  int M;
  int gu_blocks;    // column blocks (32 wide) per expert of the gate/up GEMV
  int down_blocks;  // ... of the down GEMV
  int shared;       // 1: append tiles of expert E (the shared expert) over every row
  int remap;        // 1: ids are expert ids and map[id] is the pool slot the GEMVs read (Qwen4-Exp expert cache)
};

kernel void moe_sort(
    device const int* ids [[buffer(0)]],
    device int* offsets [[buffer(1)]],
    device int4* tiles [[buffer(2)]],
    device uint* indirect [[buffer(3)]],   // [2][3] threadgroups per grid: gate/up, down
    device int* rowmap [[buffer(4)]],
    device int* slotpos [[buffer(5)]],
    constant MoeSortParams& p [[buffer(6)]],
    device const int* map [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_int tcnt[1024];
  threadgroup atomic_int tfill[1024];
  threadgroup int toff[1025];
  threadgroup int ttile[1025];
  threadgroup int scan_c[32], scan_t[32];
  for (int e = tid; e < p.E; e += ntg) {
    atomic_store_explicit(&tcnt[e], 0, memory_order_relaxed);
    atomic_store_explicit(&tfill[e], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int n = p.M * p.k;
  for (int i = tid; i < n; i += ntg) atomic_fetch_add_explicit(&tcnt[p.remap ? map[ids[i]] : ids[i]], 1, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(ntg) >= p.E) {
    // Exclusive prefix sums of the pair counts and tile counts over the
    // experts, one expert per thread (simdgroup scans, then across simdgroups):
    // the same integers the serial scan below produces.
    const int e = int(tid);
    const int c = e < p.E ? atomic_load_explicit(&tcnt[e], memory_order_relaxed) : 0;
    const int t = (c + 7) / 8;
    const int cx = simd_prefix_exclusive_sum(c), tx = simd_prefix_exclusive_sum(t);
    if (lane == 31) {
      scan_c[sg] = cx + c;
      scan_t[sg] = tx + t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      const int nsg = (int(ntg) + 31) / 32;
      const int vc = int(lane) < nsg ? scan_c[lane] : 0, vt = int(lane) < nsg ? scan_t[lane] : 0;
      const int xc = simd_prefix_exclusive_sum(vc), xt = simd_prefix_exclusive_sum(vt);
      scan_c[lane] = xc;
      scan_t[lane] = xt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e < p.E) {
      toff[e] = scan_c[sg] + cx;
      ttile[e] = scan_t[sg] + tx;
    }
    if (e == p.E - 1) {
      toff[p.E] = scan_c[sg] + cx + c;
      ttile[p.E] = scan_t[sg] + tx + t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else if (tid == 0) {
    int o = 0, t = 0;
    for (int e = 0; e < p.E; ++e) {
      const int c = atomic_load_explicit(&tcnt[e], memory_order_relaxed);
      toff[e] = o;
      ttile[e] = t;
      o += c;
      t += (c + 7) / 8;
    }
    toff[p.E] = o;
    ttile[p.E] = t;
  }
  if (tid == 0) {
    // The shared expert: tiles over the M token rows, placed after the pairs.
    const int t = ttile[p.E];
    const int ts = p.shared ? (p.M + 7) / 8 : 0;
    indirect[0] = uint(t + ts);
    indirect[1] = uint(p.gu_blocks);
    indirect[2] = 1u;
    indirect[3] = uint(t + ts);
    indirect[4] = uint(p.down_blocks);
    indirect[5] = 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = tid; e < p.E; e += ntg) {
    offsets[e] = toff[e];
    const int c = atomic_load_explicit(&tcnt[e], memory_order_relaxed);
    for (int j = 0; j * 8 < c; ++j)
      tiles[ttile[e] + j] = int4(e, toff[e] + j * 8, min(8, c - j * 8), 0);
  }
  if (p.shared) {
    const int t0 = ttile[p.E];
    for (int j = tid; j * 8 < p.M; j += ntg) tiles[t0 + j] = int4(p.E, n + j * 8, min(8, p.M - j * 8), 0);
    for (int i = tid; i < p.M; i += ntg) rowmap[n + i] = i;
  }
  for (int i = tid; i < n; i += ntg) {
    const int e = p.remap ? map[ids[i]] : ids[i];
    const int pos = toff[e] + atomic_fetch_add_explicit(&tfill[e], 1, memory_order_relaxed);
    rowmap[pos] = i / p.k;
    slotpos[i] = pos;
  }
}

// A 32-column block of a quantized GEMV over the K steps [g_begin, g_end) for
// one fp16 row, in the rgemv layout (rows n0..n0+31 of Wr/SB/Wh; the lane's
// 8 accumulators are its 4 column pairs, summed over its fm lanes afterwards).
template <int BITS>
static inline void gemv_block_partial(device const half* arow, device const uint* Wr, device const half4* SB,
                                      device const uint* Wh, int nb, int N, int KGall, int g_begin, int g_end,
                                      ushort lane, thread float* acc) {
  const int Np2 = (N + 31) / 32 * 16;
  constexpr int MAXG = 4;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  constexpr int WPL = BITS == 8 ? 4 : 2;
  constexpr int HPL = BITS == 6 ? 4 : (BITS == 5 ? 2 : 0);
  device const uint4* wp = (device const uint4*)(Wr) + ulong(nb) * KGall * (32 * WPL) + lane * WPL;
  device const uint* hp = Wh + (ulong(nb) * KGall * 32 + lane) * (HPL ? HPL : 1);
  for (int g = g_begin; g < g_end; ++g) {
    uint words[4 * WPL];
    UNROLL for (int i = 0; i < WPL; ++i) {
      const uint4 w = wp[ulong(g) * (32 * WPL) + i];
      words[4 * i] = w.x;
      words[4 * i + 1] = w.y;
      words[4 * i + 2] = w.z;
      words[4 * i + 3] = w.w;
    }
    uint hw[4] = {0u, 0u, 0u, 0u};
    if (BITS == 6) {
      const uint4 h = *((device const uint4*)(hp + ulong(g) * 32 * 4));
      hw[0] = h.x, hw[1] = h.y, hw[2] = h.z, hw[3] = h.w;
    } else if (BITS == 5) {
      const uint2 h = *((device const uint2*)(hp + ulong(g) * 32 * 2));
      hw[0] = h.x, hw[1] = h.y;
    }
    half2 s2[MAXG * 4], b2[MAXG * 4];
    kstep_scales(SB, ulong(g), KGall, Np2, (nb * 32 + fn) / 2, s2, b2);
    float xv[8];
    UNROLL for (int jj = 0; jj < 8; ++jj) xv[jj] = float(arow[g * 64 + jj * 8 + fm]);
    UNROLL for (int jj = 0; jj < 8; ++jj) {
      UNROLL for (int j = 0; j < 4; ++j) {
        half2 q;
        if (BITS == 4) {
          const uint x = words[j * 2 + jj / 4] >> (4 * (jj % 4));
          q = as_type<half2>((x & 0x000F000Fu) | 0x64006400u);
        } else if (BITS == 6 || BITS == 5) {
          const uint x = (words[j * 2 + jj / 4] >> (4 * (jj % 4))) & 0x000F000Fu;
          const uint h = BITS == 6 ? (hw[j] >> (2 * jj)) & 0x00030003u
                                   : (hw[j / 2] >> (jj + 8 * (j % 2))) & 0x00010001u;
          q = as_type<half2>(x | (h << 4) | 0x64006400u);
        } else {
          const uint x = words[j * 4 + jj / 2] >> (8 * (jj % 2));
          q = as_type<half2>((x & 0x00FF00FFu) | 0x64006400u);
        }
        const int gi = (jj * 8) / kGroup;
        const float2 wv = float2(dequant_pair(q, s2[gi * 4 + j], b2[gi * 4 + j]));
        acc[2 * j] = fma(xv[jj], wv.x, acc[2 * j]);
        acc[2 * j + 1] = fma(xv[jj], wv.y, acc[2 * j + 1]);
      }
    }
  }
}

// Fused routing: one 1024-thread threadgroup per row computes the router
// logits (32 simdgroups = column blocks x K slices, reduced in threadgroup
// memory), the shared-expert gate (same quantized layout as the router),
// then softmax, top-k (ties to the lower index) and renormalization. For a
// single-row pass it also writes the tile table, rowmap/slotpos and the
// indirect dispatch arguments (one tile per chosen expert), so no sort
// dispatch runs. Replaces router GEMV + shared-gate GEMV + moe_topk (+ sort).
struct MoeRouteFusedParams {
  int E;
  int k;
  int K;            // hidden width
  int lda;          // row stride of X (halves)
  int gu_blocks;
  int down_blocks;
  int build_tiles;  // 1: single row, write tiles/rowmap/slotpos/indirect
  int shared;       // 1: also a tile of expert E (the shared expert) for the row
};

template <int BITS>
kernel void moe_route(
    device const half* X [[buffer(0)]],
    device const uint* Wr [[buffer(1)]],
    device const half4* SB [[buffer(2)]],
    device const uint* Wh [[buffer(3)]],
    device const uint* Gr [[buffer(4)]],
    device const half4* GSB [[buffer(5)]],
    device const uint* GWh [[buffer(6)]],
    device int* ids [[buffer(7)]],
    device float* w [[buffer(8)]],
    device float* sgate [[buffer(9)]],
    device int4* tiles [[buffer(10)]],
    device uint* indirect [[buffer(11)]],
    device int* rowmap [[buffer(12)]],
    device int* slotpos [[buffer(13)]],
    constant MoeRouteFusedParams& p [[buffer(14)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint ntg [[threads_per_threadgroup]]) {
  threadgroup float part[1024];     // [slice][column] router partials (slices * E <= 1024)
  threadgroup float gpart[32];      // shared-gate partials per slice
  threadgroup float redf[32];
  threadgroup int redi[32];
  threadgroup float picked_p[16];
  threadgroup int picked_i[16];
  const int nblocks = (p.E + 31) / 32;
  const int nsimd = int(ntg) / 32;        // threadgroup size comes from the pipeline's register budget
  const int nsl = max(1, nsimd / nblocks);  // K slices (nsl * E <= 1024 floats of partials)
  const int KGall = p.K / 64;
  const int per = (KGall + nsl - 1) / nsl;
  device const half* arow = X + ulong(row) * p.lda;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  const int block = int(sg) % nblocks, slice = int(sg) / nblocks;
  if (slice < nsl) {
    const int g0 = slice * per, g1 = min(KGall, g0 + per);
    float acc[8];
    UNROLL for (int c = 0; c < 8; ++c) acc[c] = 0.0f;
    gemv_block_partial<BITS>(arow, Wr, SB, Wh, block, p.E, KGall, g0, g1, ushort(lane), acc);
    UNROLL for (int c = 0; c < 8; ++c) {
      float v = acc[c];
      v += simd_shuffle_xor(v, ushort(2));
      v += simd_shuffle_xor(v, ushort(4));
      v += simd_shuffle_xor(v, ushort(16));
      acc[c] = v;
    }
    if (fm == 0) {
      UNROLL for (int j = 0; j < 4; ++j) {
        const int col = block * 32 + j * 8 + fn;
        if (col < p.E) part[slice * p.E + col] = acc[2 * j];
        if (col + 1 < p.E) part[slice * p.E + col + 1] = acc[2 * j + 1];
      }
    }
    if (block == 0) {
      // Shared-expert gate: row 0 of a 1 x K matrix in the same layout.
      float g[8];
      UNROLL for (int c = 0; c < 8; ++c) g[c] = 0.0f;
      gemv_block_partial<BITS>(arow, Gr, GSB, GWh, 0, 1, KGall, g0, g1, ushort(lane), g);
      float v = g[0];
      v += simd_shuffle_xor(v, ushort(2));
      v += simd_shuffle_xor(v, ushort(4));
      v += simd_shuffle_xor(v, ushort(16));
      if (lane == 0) gpart[slice] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int e = int(tid);
  float x = -INFINITY;
  if (e < p.E) {
    float t = 0.0f;
    for (int sl = 0; sl < nsl; ++sl) t += part[sl * p.E + e];
    x = t;
  }
  if (tid == 0) {
    float t = 0.0f;
    for (int sl = 0; sl < nsl; ++sl) t += gpart[sl];
    sgate[row] = t;
  }
  // softmax over the E logits
  float m = simd_max(x);
  if (lane == 0) redf[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float t = lane < (ntg + 31) / 32 ? redf[lane] : -INFINITY;
    t = simd_max(t);
    if (lane == 0) redf[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = redf[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ex = e < p.E ? exp(x - m) : 0.0f;
  float ssum = simd_sum(ex);
  if (lane == 0) redf[sg] = ssum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float t = lane < (ntg + 31) / 32 ? redf[lane] : 0.0f;
    t = simd_sum(t);
    if (lane == 0) redf[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total = redf[0];
  const float prob = ex / total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mine = e < p.E ? prob : -1.0f;
  for (int r = 0; r < p.k; ++r) {
    float bv = mine;
    int bi = e;
    for (ushort off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_down(bv, off);
      const int oi = simd_shuffle_down(bi, off);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if (lane == 0) redf[sg] = bv, redi[sg] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      const bool live = lane < (ntg + 31) / 32;
      float v = live ? redf[lane] : -2.0f;
      int i = live ? redi[lane] : 0x7fffffff;
      for (ushort off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(v, off);
        const int oi = simd_shuffle_down(i, off);
        if (ov > v || (ov == v && oi < i)) v = ov, i = oi;
      }
      if (lane == 0) picked_p[r] = v, picked_i[r] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == picked_i[r]) mine = -1.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    float sum = 0.0f;
    for (int r = 0; r < p.k; ++r) sum += picked_p[r];
    for (int r = 0; r < p.k; ++r) {
      ids[ulong(row) * p.k + r] = picked_i[r];
      w[ulong(row) * p.k + r] = picked_p[r] / sum;
    }
    if (p.build_tiles) {
      for (int r = 0; r < p.k; ++r) {
        tiles[r] = int4(picked_i[r], r, 1, 0);
        rowmap[r] = 0;
        slotpos[r] = r;
      }
      if (p.shared) {
        tiles[p.k] = int4(p.E, p.k, 1, 0);
        rowmap[p.k] = 0;
      }
      const uint nt = uint(p.k + p.shared);
      indirect[0] = nt;
      indirect[1] = uint(p.gu_blocks);
      indirect[2] = 1u;
      indirect[3] = nt;
      indirect[4] = uint(p.down_blocks);
      indirect[5] = 1u;
    }
  }
}

#define INSTANTIATE_MOE_ROUTE(PRE, BITS)                                                       \
  template [[host_name(PRE)]] kernel void moe_route<BITS>(                                     \
      device const half*, device const uint*, device const half4*, device const uint*,        \
      device const uint*, device const half4*, device const uint*, device int*, device float*, \
      device float*, device int4*, device uint*, device int*, device int*,                    \
      constant MoeRouteFusedParams&, uint, uint, uint, uint, uint);
INSTANTIATE_MOE_ROUTE("moe_route4", 4)
INSTANTIATE_MOE_ROUTE("moe_route5", 5)
INSTANTIATE_MOE_ROUTE("moe_route6", 6)
INSTANTIATE_MOE_ROUTE("moe_route8", 8)

// Parallel fused routing (tier 2): grid (nblocks * nsl, rows) threadgroups of
// 256 threads. Threadgroup (b, s) computes the 32-column block b of the
// router logits over the k-steps of slice s, one k-step per simdgroup,
// reduces the simdgroups in threadgroup memory and stores the block's slice
// partial with atomic stores; block-0 threadgroups also accumulate the
// shared-expert gate. The row's last-arriving threadgroup (atomic counter,
// self-resetting) sums the slices, then softmax, top-k by rank (each thread
// counts the entries above its expert; ties to the lower index),
// renormalization, and for a single-row pass the tile table and indirect
// arguments. Same outputs as moe_route, but the router matrix streams from
// all GPU cores instead of one and the top-k costs one barrier.
struct MoeRouteParParams {
  int E;
  int k;
  int K;            // hidden width
  int lda;          // row stride of X (halves)
  int gu_blocks;
  int down_blocks;
  int build_tiles;  // 1: single row, write tiles/rowmap/slotpos/indirect
  int shared;       // 1: also a tile of expert E (the shared expert) for the row
  int nsl;          // K slices (threadgroups per column block)
  int sync;         // 0: last arriver finalizes (device fences); 1: same without fences (diagnostic);
                    // 2: partials only, moe_route_fin finalizes in a second dispatch
};

// gemv_block_partial for a dense fp16 matrix in the repacked fragment layout
// (repack_f16: per 32-column block and 64-wide K step, 8 uint4 per lane, word
// j of uint4 jj = half2 of columns (j * 8 + fn, + 1) at k = g * 64 + jj * 8 + fm):
// the same accumulator layout, so the callers' reductions are unchanged.
static inline void gemv_block_partial_f16(device const half* arow, device const uint4* Hr, int nb, int KGall,
                                          int g_begin, int g_end, ushort lane, thread float* acc) {
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  device const uint4* hp = Hr + (ulong(nb) * KGall * 32 + lane) * 8;
  for (int g = g_begin; g < g_end; ++g) {
    float xv[8];
    UNROLL for (int jj = 0; jj < 8; ++jj) xv[jj] = float(arow[g * 64 + jj * 8 + fm]);
    UNROLL for (int jj = 0; jj < 8; ++jj) {
      const uint4 w = hp[ulong(g) * 32 * 8 + jj];
      const uint ww[4] = {w.x, w.y, w.z, w.w};
      UNROLL for (int j = 0; j < 4; ++j) {
        const float2 wv = float2(as_type<half2>(ww[j]));
        acc[2 * j] = fma(xv[jj], wv.x, acc[2 * j]);
        acc[2 * j + 1] = fma(xv[jj], wv.y, acc[2 * j + 1]);
      }
    }
  }
}

// RF16: the router is a dense fp16 matrix (Wr = its repacked fragments; SB, Wh
// unused); BITS is then the shared-expert gate's width (exact bf16 routers).
// GF16: the shared-expert gate is dense fp16 too (Gr = its fragments; GGUF MoE
// files store it in F32), so BITS is unused.
template <int BITS, int RF16, int GF16 = 0>
kernel void moe_route_par(
    device const half* X [[buffer(0)]],
    device const uint* Wr [[buffer(1)]],
    device const half4* SB [[buffer(2)]],
    device const uint* Wh [[buffer(3)]],
    device const uint* Gr [[buffer(4)]],
    device const half4* GSB [[buffer(5)]],
    device const uint* GWh [[buffer(6)]],
    device int* ids [[buffer(7)]],
    device float* w [[buffer(8)]],
    device float* sgate [[buffer(9)]],
    device int4* tiles [[buffer(10)]],
    device uint* indirect [[buffer(11)]],
    device int* rowmap [[buffer(12)]],
    device int* slotpos [[buffer(13)]],
    constant MoeRouteParParams& p [[buffer(14)]],
    device float* parts [[buffer(15)]],      // [rows][nsl][E + 1] slice partials (the gate at E)
    device atomic_int* ctr [[buffer(16)]],   // [rows] arrival counters, self-resetting
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8][32];
  threadgroup float gred[8];
  threadgroup float prob[256];
  threadgroup float redf[8];
  threadgroup int picked[16];
  threadgroup int last_tg;
  const int nblocks = (p.E + 31) / 32;
  const int row = int(tgid.y);
  const int block = int(tgid.x) % nblocks, slice = int(tgid.x) / nblocks;
  const int KGall = p.K / 64;
  const int per_slice = (KGall + p.nsl - 1) / p.nsl;
  const int per_sg = (per_slice + 7) / 8;
  const int g0 = min(KGall, slice * per_slice + int(sg) * per_sg);
  const int g1 = min(min(KGall, (slice + 1) * per_slice), g0 + per_sg);
  device const half* arow = X + ulong(row) * p.lda;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  const int stride = p.E + 1;
  device float* slot = parts + (ulong(row) * ulong(p.nsl) + ulong(slice)) * ulong(stride);
  {
    float acc[8];
    UNROLL for (int c = 0; c < 8; ++c) acc[c] = 0.0f;
    if (g0 < g1) {
      if (RF16) gemv_block_partial_f16(arow, (device const uint4*)Wr, block, KGall, g0, g1, ushort(lane), acc);
      else gemv_block_partial<BITS>(arow, Wr, SB, Wh, block, p.E, KGall, g0, g1, ushort(lane), acc);
    }
    UNROLL for (int c = 0; c < 8; ++c) {
      float v = acc[c];
      v += simd_shuffle_xor(v, ushort(2));
      v += simd_shuffle_xor(v, ushort(4));
      v += simd_shuffle_xor(v, ushort(16));
      acc[c] = v;
    }
    if (fm == 0) {
      UNROLL for (int j = 0; j < 4; ++j) {
        red[sg][j * 8 + fn] = acc[2 * j];
        red[sg][j * 8 + fn + 1] = acc[2 * j + 1];
      }
    }
    if (block == 0) {
      float g[8];
      UNROLL for (int c = 0; c < 8; ++c) g[c] = 0.0f;
      if (g0 < g1) {
        if (GF16) gemv_block_partial_f16(arow, (device const uint4*)Gr, 0, KGall, g0, g1, ushort(lane), g);
        else gemv_block_partial<BITS>(arow, Gr, GSB, GWh, 0, 1, KGall, g0, g1, ushort(lane), g);
      }
      float v = g[0];
      v += simd_shuffle_xor(v, ushort(2));
      v += simd_shuffle_xor(v, ushort(4));
      v += simd_shuffle_xor(v, ushort(16));
      if (lane == 0) gred[sg] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 32) {
    const int col = block * 32 + int(tid);
    float t = 0.0f;
    UNROLL for (int s = 0; s < 8; ++s) t += red[s][tid];
    if ((p.sync & 15) == 2) {
      // plain stores: read by moe_route_fin after the dispatch boundary
      if (col < p.E) slot[col] = t;
      if (block == 0 && tid == 0) {
        float gt = 0.0f;
        UNROLL for (int s = 0; s < 8; ++s) gt += gred[s];
        slot[p.E] = gt;
      }
    } else {
      if (col < p.E) atomic_store_explicit((device atomic_uint*)(slot + col), as_type<uint>(t), memory_order_relaxed);
      if (block == 0 && tid == 0) {
        float gt = 0.0f;
        UNROLL for (int s = 0; s < 8; ++s) gt += gred[s];
        atomic_store_explicit((device atomic_uint*)(slot + p.E), as_type<uint>(gt), memory_order_relaxed);
      }
    }
  }
  if ((p.sync & 15) == 2) return;
  if ((p.sync & 15) == 0) atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) last_tg = atomic_fetch_add_explicit(ctr + row, 1, memory_order_relaxed) == nblocks * p.nsl - 1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!last_tg) return;
  if ((p.sync & 15) == 0) atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
  // ---- last arriver: logits, softmax, top-k, outputs
  const int e = int(tid);
  device const float* rp = parts + ulong(row) * ulong(p.nsl) * ulong(stride);
  float x = -INFINITY;
  if (e < p.E) {
    float t = 0.0f;
    for (int s = 0; s < p.nsl; ++s)
      t += as_type<float>(atomic_load_explicit((device atomic_uint*)(rp + ulong(s) * ulong(stride) + e), memory_order_relaxed));
    x = t;
  }
  if (tid == 0) {
    float t = 0.0f;
    for (int s = 0; s < p.nsl; ++s)
      t += as_type<float>(atomic_load_explicit((device atomic_uint*)(rp + ulong(s) * ulong(stride) + p.E), memory_order_relaxed));
    sgate[row] = t;
    atomic_store_explicit(ctr + row, 0, memory_order_relaxed);
  }
  float m = simd_max(x);
  if (lane == 0) redf[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = max(max(max(redf[0], redf[1]), max(redf[2], redf[3])), max(max(redf[4], redf[5]), max(redf[6], redf[7])));
  const float ex = e < p.E ? exp(x - m) : 0.0f;
  const float ssum = simd_sum(ex);
  threadgroup_barrier(mem_flags::mem_threadgroup);  // redf reused
  if (lane == 0) redf[sg] = ssum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total = redf[0] + redf[1] + redf[2] + redf[3] + redf[4] + redf[5] + redf[6] + redf[7];
  const float pe = ex / total;
  if (e < p.E) prob[e] = pe;
  // top-k: k rounds of max (simd_max, then the lowest index among equals,
  // then across the 8 simdgroups). Measured 2.5x faster than ranking every
  // entry against all E in a threadgroup-memory loop.
  threadgroup float redv[8];
  threadgroup int redi[8];
  float mine = e < p.E ? pe : -1.0f;
  for (int r = 0; r < p.k; ++r) {
    const float bv = simd_max(mine);
    const int bi = simd_min(mine == bv ? e : 0x7fffffff);
    if (lane == 0) redv[sg] = bv, redi[sg] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float v = redv[0];
      int i = redi[0];
      for (int s2 = 1; s2 < 8; ++s2)
        if (redv[s2] > v || (redv[s2] == v && redi[s2] < i)) v = redv[s2], i = redi[s2];
      picked[r] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == picked[r]) mine = -1.0f;
  }
  if (tid == 0) {
    float sum = 0.0f;
    for (int r = 0; r < p.k; ++r) sum += prob[picked[r]];
    for (int r = 0; r < p.k; ++r) {
      ids[ulong(row) * p.k + r] = picked[r];
      w[ulong(row) * p.k + r] = prob[picked[r]] / sum;
    }
    if (p.build_tiles) {
      for (int r = 0; r < p.k; ++r) {
        tiles[r] = int4((p.sync & 16) ? r : picked[r], r, 1, 0);  // sync bit 4: experts 0..k-1 (timing diagnostic)
        rowmap[r] = 0;
        slotpos[r] = r;
      }
      if (p.shared) {
        tiles[p.k] = int4(p.E, p.k, 1, 0);
        rowmap[p.k] = 0;
      }
      if (p.build_tiles == 1) {  // 2: the grouped GEMVs use direct grids, no indirect arguments
        const uint nt = uint(p.k + p.shared);
        indirect[0] = nt;
        indirect[1] = uint(p.gu_blocks);
        indirect[2] = 1u;
        indirect[3] = nt;
        indirect[4] = uint(p.down_blocks);
        indirect[5] = 1u;
      }
    }
  }
}

// Second dispatch of the two-pass route (sync 2): one 256-thread threadgroup
// per row sums the slice partials written by moe_route_par, then softmax,
// rank top-k, renormalization, the shared gate and (single row) the tile
// table. Replaces the in-dispatch last arriver: a dispatch boundary costs
// less than device-scope fences on this GPU.
kernel void moe_route_fin(
    device const float* parts [[buffer(0)]],     // [rows][nsl][E + 1]
    device int* ids [[buffer(1)]],
    device float* w [[buffer(2)]],
    device float* sgate [[buffer(3)]],
    device int4* tiles [[buffer(4)]],
    device uint* indirect [[buffer(5)]],
    device int* rowmap [[buffer(6)]],
    device int* slotpos [[buffer(7)]],
    constant MoeRouteParParams& p [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float prob[256];
  threadgroup float redf[8];
  threadgroup int picked[16];
  const int stride = p.E + 1;
  const int e = int(tid);
  device const float* rp = parts + ulong(row) * ulong(p.nsl) * ulong(stride);
  float x = -INFINITY;
  {
    // All slice partials requested at once (a runtime loop serializes the load latencies).
    float t = 0.0f, gt = 0.0f;
    UNROLL for (int s = 0; s < 8; ++s)
      if (s < p.nsl) {
        if (e < p.E) t += rp[ulong(s) * ulong(stride) + e];
        if (tid == 0) gt += rp[ulong(s) * ulong(stride) + p.E];
      }
    if (e < p.E) x = t;
    if (tid == 0) sgate[row] = gt;
  }
  float m = simd_max(x);
  if (lane == 0) redf[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = max(max(max(redf[0], redf[1]), max(redf[2], redf[3])), max(max(redf[4], redf[5]), max(redf[6], redf[7])));
  const float ex = e < p.E ? exp(x - m) : 0.0f;
  const float ssum = simd_sum(ex);
  threadgroup_barrier(mem_flags::mem_threadgroup);  // redf reused
  if (lane == 0) redf[sg] = ssum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total = redf[0] + redf[1] + redf[2] + redf[3] + redf[4] + redf[5] + redf[6] + redf[7];
  const float pe = ex / total;
  if (e < p.E) prob[e] = pe;
  // top-k: k rounds of max (simd_max, then the lowest index among equals,
  // then across the 8 simdgroups). Measured 2.5x faster than ranking every
  // entry against all E in a threadgroup-memory loop.
  threadgroup float redv[8];
  threadgroup int redi[8];
  float mine = e < p.E ? pe : -1.0f;
  for (int r = 0; r < p.k; ++r) {
    const float bv = simd_max(mine);
    const int bi = simd_min(mine == bv ? e : 0x7fffffff);
    if (lane == 0) redv[sg] = bv, redi[sg] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float v = redv[0];
      int i = redi[0];
      for (int s2 = 1; s2 < 8; ++s2)
        if (redv[s2] > v || (redv[s2] == v && redi[s2] < i)) v = redv[s2], i = redi[s2];
      picked[r] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == picked[r]) mine = -1.0f;
  }
  if (tid == 0) {
    float sum = 0.0f;
    for (int r = 0; r < p.k; ++r) sum += prob[picked[r]];
    for (int r = 0; r < p.k; ++r) {
      ids[ulong(row) * p.k + r] = picked[r];
      w[ulong(row) * p.k + r] = prob[picked[r]] / sum;
    }
    if (p.build_tiles) {
      for (int r = 0; r < p.k; ++r) {
        tiles[r] = int4((p.sync & 16) ? r : picked[r], r, 1, 0);  // sync bit 4: experts 0..k-1 (timing diagnostic)
        rowmap[r] = 0;
        slotpos[r] = r;
      }
      if (p.shared) {
        tiles[p.k] = int4(p.E, p.k, 1, 0);
        rowmap[p.k] = 0;
      }
      if (p.build_tiles == 1) {  // 2: the grouped GEMVs use direct grids, no indirect arguments
        const uint nt = uint(p.k + p.shared);
        indirect[0] = nt;
        indirect[1] = uint(p.gu_blocks);
        indirect[2] = 1u;
        indirect[3] = nt;
        indirect[4] = uint(p.down_blocks);
        indirect[5] = 1u;
      }
    }
  }
}

// Softmax + top-k of one row on one simdgroup, from the parallel route's
// slice partials rp = parts + row * nsl * (E + 1): each lane holds E / 32
// experts (E <= 256: 8 per lane); k rounds of (largest probability, lowest
// expert index among equals) on simd reductions only, no barriers. pick_i /
// pick_v (k <= 16) and sum are uniform across the simdgroup; gate is the
// shared-expert gate logit (partials at column E).
static inline void route_topk_simd(device const float* rp, int E, int k, int nsl, uint lane, thread int* pick_i,
                                   thread float* pick_v, thread float& sum, thread float& gate, int noload = 0) {
  constexpr int PL = 8;
  constexpr int NSL = 8;  // slices at most (route_nsl <= 8)
  const int stride = E + 1;
  // All slice partials of this lane's experts are requested at once (a
  // runtime loop per expert would serialize 4-8 load latencies per expert).
  float x[PL];
  UNROLL for (int i = 0; i < PL; ++i) x[i] = 0.0f;
  float gt = 0.0f;
  if (noload) {  // timing diagnostic: fake logits, no partial loads
    UNROLL for (int i = 0; i < PL; ++i) x[i] = float((int(lane) + 32 * i) % 17) * 0.1f;
  } else if (nsl <= 4) {
    // Branchless: every load unconditional (clamped slice and expert index),
    // so all 4 x 8 loads are in flight together; masked accumulation.
    float v[4][PL], gv[4];
    UNROLL for (int s = 0; s < 4; ++s) {
      device const float* rs = rp + ulong(min(s, nsl - 1)) * ulong(stride);
      UNROLL for (int i = 0; i < PL; ++i) v[s][i] = rs[min(int(lane) + 32 * i, E - 1)];
      gv[s] = rs[E];
    }
    UNROLL for (int s = 0; s < 4; ++s) {
      const float msk = s < nsl ? 1.0f : 0.0f;
      UNROLL for (int i = 0; i < PL; ++i) x[i] = fma(msk, v[s][i], x[i]);
      gt = fma(msk, gv[s], gt);
    }
  } else {
    UNROLL for (int s = 0; s < NSL; ++s) {
      if (s < nsl) {
        device const float* rs = rp + ulong(s) * ulong(stride);
        UNROLL for (int i = 0; i < PL; ++i) {
          const int e = int(lane) + 32 * i;
          if (e < E) x[i] += rs[e];
        }
        if (lane == 0) gt += rs[E];
      }
    }
  }
  gate = gt;
  float mx = -INFINITY;
  UNROLL for (int i = 0; i < PL; ++i) {
    if (int(lane) + 32 * i >= E) x[i] = -INFINITY;
    mx = max(mx, x[i]);
  }
  mx = simd_max(mx);
  float lsum = 0.0f;
  UNROLL for (int i = 0; i < PL; ++i) {
    x[i] = int(lane) + 32 * i < E ? exp(x[i] - mx) : -1.0f;
    lsum += max(x[i], 0.0f);
  }
  const float inv = 1.0f / simd_sum(lsum);
  UNROLL for (int i = 0; i < PL; ++i) x[i] = x[i] >= 0.0f ? x[i] * inv : -1.0f;
  sum = 0.0f;
  UNROLL for (int r = 0; r < 16; ++r) {
    if (r >= k) break;
    float bv = -1.0f;
    int bi = 0x7fffffff;
    UNROLL for (int i = 0; i < PL; ++i)
      if (x[i] > bv) bv = x[i], bi = int(lane) + 32 * i;  // ascending index within the lane: first wins ties
    const float gv = simd_max(bv);
    const int gi = simd_min(bv == gv ? bi : 0x7fffffff);
    UNROLL for (int i = 0; i < PL; ++i)
      if (int(lane) + 32 * i == gi) x[i] = -1.0f;
    pick_i[r] = gi;
    pick_v[r] = gv;
    sum += gv;
  }
}

// Barrier-free finalize of the parallel route: one simdgroup per row
// (route_topk_simd); lane r < k writes pair r. Replaces moe_route_fin (256
// threads, 2 barriers per top-k round, one thread writing the outputs).
kernel void moe_route_fin1(
    device const float* parts [[buffer(0)]],     // [rows][nsl][E + 1]
    device int* ids [[buffer(1)]],
    device float* w [[buffer(2)]],
    device float* sgate [[buffer(3)]],
    device int4* tiles [[buffer(4)]],
    device uint* indirect [[buffer(5)]],
    device int* rowmap [[buffer(6)]],
    device int* slotpos [[buffer(7)]],
    constant MoeRouteParParams& p [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  int pick_i[16];
  float pick_v[16];
  float sum, gate;
  route_topk_simd(parts + ulong(row) * ulong(p.nsl) * ulong(p.E + 1), p.E, p.k, p.nsl, lane, pick_i, pick_v, sum, gate, p.sync & 32);
  if (lane == 0) sgate[row] = gate;
  UNROLL for (int r = 0; r < 16; ++r) {
    if (r >= p.k) break;
    if (int(lane) == r) {
      ids[ulong(row) * p.k + r] = pick_i[r];
      w[ulong(row) * p.k + r] = pick_v[r] / sum;
      if (p.build_tiles) {
        tiles[r] = int4((p.sync & 16) ? r : pick_i[r], r, 1, 0);
        rowmap[r] = 0;
        slotpos[r] = r;
      }
    }
  }
  if (p.build_tiles && lane == 0) {
    if (p.shared) {
      tiles[p.k] = int4(p.E, p.k, 1, 0);
      rowmap[p.k] = 0;
    }
    if (p.build_tiles == 1) {
      const uint nt = uint(p.k + p.shared);
      indirect[0] = nt;
      indirect[1] = uint(p.gu_blocks);
      indirect[2] = 1u;
      indirect[3] = nt;
      indirect[4] = uint(p.down_blocks);
      indirect[5] = 1u;
    }
  }
}

#define INSTANTIATE_MOE_ROUTE_PAR(PRE, BITS) INSTANTIATE_MOE_ROUTE_PAR_R(PRE, BITS, 0) INSTANTIATE_MOE_ROUTE_PAR_R(PRE "h", BITS, 1)
#define INSTANTIATE_MOE_ROUTE_PAR_R(PRE, BITS, RF16)                                           \
  template [[host_name(PRE)]] kernel void moe_route_par<BITS, RF16>(                           \
      device const half*, device const uint*, device const half4*, device const uint*,        \
      device const uint*, device const half4*, device const uint*, device int*, device float*, \
      device float*, device int4*, device uint*, device int*, device int*,                    \
      constant MoeRouteParParams&, device float*, device atomic_int*, uint2, uint, uint, uint);
INSTANTIATE_MOE_ROUTE_PAR("moe_route_par4", 4)
INSTANTIATE_MOE_ROUTE_PAR("moe_route_par5", 5)
INSTANTIATE_MOE_ROUTE_PAR("moe_route_par6", 6)
INSTANTIATE_MOE_ROUTE_PAR("moe_route_par8", 8)
// Dense fp16 router and shared gate (GGUF MoE files).
template [[host_name("moe_route_parhh")]] kernel void moe_route_par<8, 1, 1>(
    device const half*, device const uint*, device const half4*, device const uint*,
    device const uint*, device const half4*, device const uint*, device int*, device float*,
    device float*, device int4*, device uint*, device int*, device int*,
    constant MoeRouteParParams&, device float*, device atomic_int*, uint2, uint, uint, uint);

struct MoeGemmParams {
  int Ne;      // output columns per expert
  int N;       // all experts' columns (E * Ne): the packed matrix's width
  int K;
  int lda;
  int ldy;
  int gather;  // 1: tile row r reads A row rowmap[r] (tokens); 0: A row r
  int shared_e;  // tiles of this expert index use the second matrix (the shared expert), -1: none
  int N2;        // the second matrix's width (its columns start at 0)
  int direct;  // 1: exact direct grid, the tile count is not read
  int nbpt;    // moe_gemv: 32-column blocks per threadgroup (in sequence; amortizes the tile prologue)
  int swap;    // moe_gemv: 1: grid = (column blocks, tiles) so consecutive threadgroups stream one expert's contiguous columns
  int route_nsl;  // moe_gemv<..., DIAG = 2>: the route finalize folded into the gate/up prologue (0: off); fields below
  int route_E;
  int route_k;
  int route_sync;
  int route_shared;
  // Staged experts (Qwen4-Exp long prompts): tiles of experts
  // [st_first, st_first + st_count) read expert e - st_first from the third
  // matrix, one record per expert: st_words 32-bit words (st_half4 half4s) apart,
  // each laid out as a single-expert matrix. st_count 0: none.
  int st_first;
  int st_count;
  int st_words;
  int st_half4;
  // Split pass (moe_gemv8 only): 1 runs only the tiles of experts that
  // are ready, 2 only those of experts still being read (flag != 0 in the
  // per-expert flags at buffer 15); 0: every tile.
  int split;
};

// Grouped expert GEMM: one tile (expert, <= 8 sorted rows) x 32 of the
// expert's columns per threadgroup, in rgemv8's layout (K split over 8
// simdgroups, 8x8 MMA accumulators, no barriers in the loop). Output rows are
// the sorted positions; columns are local to the expert.
template <int BITS, int EPI>
kernel void moe_gemv8(
    device const half* A [[buffer(0)]],
    device const uint* Wr [[buffer(1)]],
    device const half4* SB [[buffer(2)]],
    device const uint* Wh [[buffer(3)]],
    device void* Yv [[buffer(4)]],
    constant MoeGemmParams& p [[buffer(5)]],
    device const int4* tiles [[buffer(6)]],
    device const uint* ntiles [[buffer(7)]],  // indirect arguments: [0] is the tile count
    device const int* rowmap [[buffer(8)]],
    device const uint* Wr2 [[buffer(9)]],     // the shared expert's matrix (same layout, bits and group)
    device const half4* SB2 [[buffer(10)]],
    device const uint* Wh2 [[buffer(11)]],
    device const uint* Wr3 [[buffer(12)]],    // staged experts: one single-expert record each (p.st_*)
    device const half4* SB3 [[buffer(13)]],
    device const uint* Wh3 [[buffer(14)]],
    device const uchar* loading [[buffer(15)]],  // per expert index: being read (p.split)
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int KS = 8;
  if (!p.direct && tgid.x >= ntiles[0]) return;
  const int4 ti = tiles[tgid.x];
  const int e = ti.x, r0 = ti.y, rows = ti.z;
  if (p.split && (loading[e] != 0) != (p.split == 2)) return;
  const int nloc = int(tgid.y) * 32;
  if (nloc >= p.Ne) return;
  const bool sh = e == p.shared_e;  // uniform per threadgroup
  const bool st = !sh && p.st_count > 0 && e >= p.st_first && e < p.st_first + p.st_count;
  if (sh) {
    Wr = Wr2;
    SB = SB2;
    Wh = Wh2;
  } else if (st) {
    const ulong s = ulong(e - p.st_first);
    Wr = Wr3 + s * ulong(p.st_words);
    SB = SB3 + s * ulong(p.st_half4);
    Wh = Wh3 + s * ulong(p.st_words);
  }
  const int n0 = (sh || st) ? nloc : e * p.Ne + nloc;
  const int nb = n0 / 32;
  const int KGall = p.K / 64;
  const int per = (KGall + KS - 1) / KS;
  const int g_begin = int(sgid) * per;
  const int g_end = min(KGall, g_begin + per);
  const int Np2 = ((sh ? p.N2 : (st ? p.Ne : p.N)) + 31) / 32 * 16;
  constexpr int MAXG = 4;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  constexpr int WPL = BITS == 8 ? 4 : 2;
  constexpr int HPL = BITS == 6 ? 4 : (BITS == 5 ? 2 : 0);
  device const uint4* wp = (device const uint4*)(Wr) + ulong(nb) * KGall * (32 * WPL) + lane * WPL;
  device const uint* hp = Wh + (ulong(nb) * KGall * 32 + lane) * (HPL ? HPL : 1);
  const bool row_ok = fm < rows;
  const int arow_i = row_ok ? (p.gather ? rowmap[r0 + fm] : r0 + fm) : 0;
  device const half* arow = A + ulong(arow_i) * p.lda;
  simdgroup_matrix<float, 8, 8> acc[4];
  UNROLL for (int j = 0; j < 4; ++j) acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  uint words[4 * WPL], hw[4] = {0u, 0u, 0u, 0u};
  if (ESPETERO_PF && g_begin < g_end) load_kstep<BITS, WPL>(wp, hp, g_begin, words, hw);
  for (int g = g_begin; g < g_end; ++g) {
    // Software pipeline: the next K step's weight words are requested before
    // this step's dequantization and FMAs, so two steps' loads are in flight
    // per simdgroup (the loop was latency-bound: ~2.7 us per 64-wide step).
    uint wnext[4 * WPL], hnext[4] = {0u, 0u, 0u, 0u};
    if (!ESPETERO_PF) load_kstep<BITS, WPL>(wp, hp, g, words, hw);
    if (ESPETERO_PF && g + 1 < g_end) load_kstep<BITS, WPL>(wp, hp, g + 1, wnext, hnext);
    half2 s2[MAXG * 4], b2[MAXG * 4];
    kstep_scales(SB, ulong(g), KGall, Np2, (n0 + fn) / 2, s2, b2);
    half2 av[8];
    UNROLL for (int jj = 0; jj < 8; ++jj)
      av[jj] = row_ok ? *((device const half2*)(arow + g * 64 + jj * 8 + fn)) : half2(0.0h);
    UNROLL for (int jj = 0; jj < 8; ++jj) {
      simdgroup_matrix<half, 8, 8> a;
      a.thread_elements()[0] = av[jj].x;
      a.thread_elements()[1] = av[jj].y;
      UNROLL for (int j = 0; j < 4; ++j) {
        half2 q;
        if (BITS == 4) {
          const uint x = words[j * 2 + jj / 4] >> (4 * (jj % 4));
          q = as_type<half2>((x & 0x000F000Fu) | 0x64006400u);
        } else if (BITS == 6 || BITS == 5) {
          const uint x = (words[j * 2 + jj / 4] >> (4 * (jj % 4))) & 0x000F000Fu;
          const uint h = BITS == 6 ? (hw[j] >> (2 * jj)) & 0x00030003u
                                   : (hw[j / 2] >> (jj + 8 * (j % 2))) & 0x00010001u;
          q = as_type<half2>(x | (h << 4) | 0x64006400u);
        } else {
          const uint x = words[j * 4 + jj / 2] >> (8 * (jj % 2));
          q = as_type<half2>((x & 0x00FF00FFu) | 0x64006400u);
        }
        const int gi = (jj * 8) / kGroup;
        const half2 wv = dequant_pair(q, s2[gi * 4 + j], b2[gi * 4 + j]);
        simdgroup_matrix<half, 8, 8> b;
        b.thread_elements()[0] = wv.x;
        b.thread_elements()[1] = wv.y;
        simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
      }
    }
    if (ESPETERO_PF) {
      UNROLL for (int i = 0; i < 4 * WPL; ++i) words[i] = wnext[i];
      UNROLL for (int i = 0; i < 4; ++i) hw[i] = hnext[i];
    }
  }
  threadgroup float part[KS][8][32];
  UNROLL for (int j = 0; j < 4; ++j) {
    part[sgid][fm][j * 8 + fn] = acc[j].thread_elements()[0];
    part[sgid][fm][j * 8 + fn + 1] = acc[j].thread_elements()[1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int row = int(sgid);
  const int c = int(lane);
  if (row >= rows) return;
  float v = 0.0f;
  UNROLL for (int k = 0; k < KS; ++k) v += part[k][row][c];
  const ulong orow = ulong(r0 + row) * p.ldy;
  if (EPI == EPI_SWIGLU) {
    if ((c / 8) % 2 == 0 && nloc + c < p.Ne) {
      float u = 0.0f;
      UNROLL for (int k = 0; k < KS; ++k) u += part[k][row][c + 8];
      const int out = ((nloc + (c / 8) * 8) / 16) * 8 + (c % 8);
      ((device half*)Yv)[orow + out] = half(glu_gate(v) * u);
    }
  } else if (nloc + c < p.Ne) {
    device float* y = (device float*)Yv + orow + nloc + c;
    *y = (EPI == EPI_ADD) ? *y + v : v;
  }
}

#define INSTANTIATE_MOE(PRE, BITS)                                                        \
  template [[host_name(PRE "_store")]] kernel void moe_gemv8<BITS, 0>(                    \
      device const half*, device const uint*, device const half4*, device const uint*,   \
      device void*, constant MoeGemmParams&, device const int4*, device const uint*,     \
      device const int*, device const uint*, device const half4*, device const uint*,    \
      device const uint*, device const half4*, device const uint*, device const uchar*,  \
      uint3, uint, uint);                                                                 \
  template [[host_name(PRE "_swiglu")]] kernel void moe_gemv8<BITS, 2>(                   \
      device const half*, device const uint*, device const half4*, device const uint*,   \
      device void*, constant MoeGemmParams&, device const int4*, device const uint*,     \
      device const int*, device const uint*, device const half4*, device const uint*,    \
      device const uint*, device const half4*, device const uint*, device const uchar*,  \
      uint3, uint, uint);
INSTANTIATE_MOE("moe_gemv8r4", 4)
INSTANTIATE_MOE("moe_gemv8r5", 5)
INSTANTIATE_MOE("moe_gemv8r6", 6)
INSTANTIATE_MOE("moe_gemv8r8", 8)

// h[row] += sum over slots (fixed order) of w * y_sorted[slotpos] +
// sigmoid(gate[row]) * shared[row]. Grid (rows, column slices): each
// threadgroup sums its slice; with `norm`, the row's RMSNorm for the next
// layer is finished by the last-arriving threadgroup (partial sums of
// squares and the residual are stored atomically, see rgemv's finalizer),
// which writes the fp16 row `xn` with the next norm's weight `wn`.
// Grouped expert GEMV for 1-2-row tiles (tier 2, decode): the FMA path of
// rgemv with the tile table and rowmap of moe_gemv8. At one or two rows the
// 8x8 MMA spends most of its rows on padding and its compute (151 M MACs per
// gate/up layer at batch 1, ~28 us) is not hidden under the weight stream;
// here each weight costs one FMA per row. One threadgroup per (tile,
// 32-column block), KS = 8 simdgroups slicing K, two K steps in flight.
// The K loop of moe_gemv for one 32-column block of a BITSX-bit matrix:
// KS simdgroups slice K, each lane accumulates 8 columns x MM rows over its
// fm k rows; the caller reduces across lanes and simdgroups. Templated on
// the bit width so a threadgroup of the shared expert (tile k) can stream an
// 8-bit matrix beside 4/5/6-bit routed experts (oQ4e-style files).
template <int BITSX, int MM, int KS>
static inline void moe_kloop(device const uint* Wr, device const half4* SB, device const uint* Wh, int nb, int KGall,
                             int g_begin, int g_end, int Np2, int n0, int rows, device const half* arow0,
                             device const half* arow1, ushort lane, short fm, short fn, thread float* acc) {
  constexpr int MAXG = 4;
  constexpr int WPL = BITSX == 8 ? 4 : 2;
  constexpr int HPL = BITSX == 6 ? 4 : (BITSX == 5 ? 2 : 0);
  device const uint4* wp = (device const uint4*)(Wr) + ulong(nb) * KGall * (32 * WPL) + lane * WPL;
  device const uint* hp = Wh + (ulong(nb) * KGall * 32 + lane) * (HPL ? HPL : 1);
  device const half* arow[2] = {arow0, arow1};
  uint words[4 * WPL], hw[4] = {0u, 0u, 0u, 0u};
  if (ESPETERO_PF && g_begin < g_end) load_kstep<BITSX, WPL>(wp, hp, g_begin, words, hw);
  for (int g = g_begin; g < g_end; ++g) {
    uint wnext[4 * WPL], hnext[4] = {0u, 0u, 0u, 0u};
    if (!ESPETERO_PF) load_kstep<BITSX, WPL>(wp, hp, g, words, hw);
    if (ESPETERO_PF && g + 1 < g_end) load_kstep<BITSX, WPL>(wp, hp, g + 1, wnext, hnext);
    half2 s2[MAXG * 4], b2[MAXG * 4];
    kstep_scales(SB, ulong(g), KGall, Np2, (n0 + fn) / 2, s2, b2);
    float xv[MM][8];
    UNROLL for (int m = 0; m < MM; ++m)
      UNROLL for (int jj = 0; jj < 8; ++jj)
        xv[m][jj] = m < rows ? float(arow[m][g * 64 + jj * 8 + fm]) : 0.0f;
    UNROLL for (int jj = 0; jj < 8; ++jj) {
      UNROLL for (int j = 0; j < 4; ++j) {
        half2 q;
        if (BITSX == 4) {
          const uint x = words[j * 2 + jj / 4] >> (4 * (jj % 4));
          q = as_type<half2>((x & 0x000F000Fu) | 0x64006400u);
        } else if (BITSX == 6 || BITSX == 5) {
          const uint x = (words[j * 2 + jj / 4] >> (4 * (jj % 4))) & 0x000F000Fu;
          const uint h = BITSX == 6 ? (hw[j] >> (2 * jj)) & 0x00030003u
                                    : (hw[j / 2] >> (jj + 8 * (j % 2))) & 0x00010001u;
          q = as_type<half2>(x | (h << 4) | 0x64006400u);
        } else {
          const uint x = words[j * 4 + jj / 2] >> (8 * (jj % 2));
          q = as_type<half2>((x & 0x00FF00FFu) | 0x64006400u);
        }
        const int gi = (jj * 8) / kGroup;
        const float2 wv = float2(dequant_pair(q, s2[gi * 4 + j], b2[gi * 4 + j]));
        UNROLL for (int m = 0; m < MM; ++m) {
          acc[m * 8 + 2 * j] = fma(xv[m][jj], wv.x, acc[m * 8 + 2 * j]);
          acc[m * 8 + 2 * j + 1] = fma(xv[m][jj], wv.y, acc[m * 8 + 2 * j + 1]);
        }
      }
    }
    if (ESPETERO_PF) {
      UNROLL for (int i = 0; i < 4 * WPL; ++i) words[i] = wnext[i];
      UNROLL for (int i = 0; i < 4; ++i) hw[i] = hnext[i];
    }
  }
}

template <int BITS, int MM, int EPI, int KS, int DIAG, int BITS2>
kernel void moe_gemv(
    device const half* A [[buffer(0)]],
    device const uint* Wr [[buffer(1)]],
    device const half4* SB [[buffer(2)]],
    device const uint* Wh [[buffer(3)]],
    device void* Yv [[buffer(4)]],
    constant MoeGemmParams& p [[buffer(5)]],
    device int4* tiles [[buffer(6)]],         // written by DIAG 2 (route folded in)
    device const uint* ntiles [[buffer(7)]],  // indirect arguments: [0] is the tile count
    device int* rowmap [[buffer(8)]],
    device const uint* Wr2 [[buffer(9)]],     // the shared expert's matrix (same layout and group; BITS2 bits)
    device const half4* SB2 [[buffer(10)]],
    device const uint* Wh2 [[buffer(11)]],
    device const float* rparts [[buffer(12)]],  // DIAG 2 (route folded in): the parallel route's slice partials
    device int* ids [[buffer(13)]],
    device float* w [[buffer(14)]],
    device float* sgate [[buffer(15)]],
    device int* slotpos [[buffer(16)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint tix = p.swap ? tgid.y : tgid.x, bix = p.swap ? tgid.x : tgid.y;
  int e, r0, rows;
  threadgroup int tg_e;
  if (DIAG == 1) {
    // Timing diagnostic (bench_json "moe"): the expert comes from the
    // threadgroup id, so no tile or rowmap load precedes the weight stream.
    e = int(tix) < 8 ? ((int(tix) * 31 + 5) & 255) : p.shared_e;
    r0 = int(tix), rows = 1;
  } else if (DIAG == 2) {
    // The route finalize folded in (single row, direct grid): simdgroup 0
    // recomputes the row's top-k from the partials (identical in every
    // threadgroup), threadgroup (0, 0) writes ids / weights / tiles / rowmap /
    // slotpos / shared gate for the down GEMV and the combine.
    r0 = int(tix), rows = 1;
    if (sgid == 0) {
      int pick_i[16];
      float pick_v[16];
      float sum, gate;
      route_topk_simd(rparts, p.route_E, p.route_k, p.route_nsl, lane, pick_i, pick_v, sum, gate, p.route_sync & 32);
      int ee = p.route_E;
      UNROLL for (int r = 0; r < 16; ++r) {
        if (r >= p.route_k) break;
        if (int(tix) == r) ee = (p.route_sync & 16) ? r : pick_i[r];
      }
      if (lane == 0) tg_e = ee;
      if (tix == 0 && bix == 0) {
        if (lane == 0) sgate[0] = gate;
        UNROLL for (int r = 0; r < 16; ++r) {
          if (r >= p.route_k) break;
          if (int(lane) == r) {
            ids[r] = pick_i[r];
            w[r] = pick_v[r] / sum;
            tiles[r] = int4((p.route_sync & 16) ? r : pick_i[r], r, 1, 0);
            rowmap[r] = 0;
            slotpos[r] = r;
          }
        }
        if (lane == 0 && p.route_shared) {
          tiles[p.route_k] = int4(p.route_E, p.route_k, 1, 0);
          rowmap[p.route_k] = 0;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    e = tg_e;
  } else {
    const int4 ti = tiles[tix];
    if (!p.direct && tix >= ntiles[0]) return;
    e = ti.x, r0 = ti.y, rows = ti.z;
  }
  const bool sh = e == p.shared_e;  // uniform per threadgroup
  const int KGall = p.K / 64;
  const int per = (KGall + KS - 1) / KS;
  const int g_begin = int(sgid) * per;
  const int g_end = min(KGall, g_begin + per);
  const int Np2 = ((sh ? p.N2 : p.N) + 31) / 32 * 16;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  device const half* arow[MM];
  UNROLL for (int m = 0; m < MM; ++m) {
    const int ri = m < rows ? ((p.gather && !DIAG) ? rowmap[r0 + m] : (DIAG && p.gather ? 0 : r0 + m)) : 0;  // DIAG: A row 0
    arow[m] = A + ulong(ri) * p.lda;
  }
  threadgroup float part[KS][MM][32];
  // Column blocks of this threadgroup, in sequence (p.nbpt >= 1).
  for (int blk = 0; blk < p.nbpt; ++blk) {
  const int nloc = (int(bix) * p.nbpt + blk) * 32;
  if (nloc >= p.Ne) break;
  const int n0 = sh ? nloc : e * p.Ne + nloc;
  const int nb = n0 / 32;
  if (blk) threadgroup_barrier(mem_flags::mem_threadgroup);  // `part` of the previous block was consumed
  float acc[MM][8];
  UNROLL for (int m = 0; m < MM; ++m)
    UNROLL for (int c = 0; c < 8; ++c) acc[m][c] = 0.0f;
  if (sh)
    moe_kloop<BITS2, MM, KS>(Wr2, SB2, Wh2, nb, KGall, g_begin, g_end, Np2, n0, rows, arow[0], arow[MM - 1], ushort(lane), fm, fn, &acc[0][0]);
  else
    moe_kloop<BITS, MM, KS>(Wr, SB, Wh, nb, KGall, g_begin, g_end, Np2, n0, rows, arow[0], arow[MM - 1], ushort(lane), fm, fn, &acc[0][0]);
  // Sum the 8 lanes that hold different k rows (fm: lane bits 1, 2 and 4).
  UNROLL for (int m = 0; m < MM; ++m)
    UNROLL for (int c = 0; c < 8; ++c) {
      float v = acc[m][c];
      v += simd_shuffle_xor(v, ushort(2));
      v += simd_shuffle_xor(v, ushort(4));
      v += simd_shuffle_xor(v, ushort(16));
      acc[m][c] = v;
    }
  if (fm == 0) {
    UNROLL for (int m = 0; m < MM; ++m)
      UNROLL for (int j = 0; j < 4; ++j) {
        part[sgid][m][j * 8 + fn] = acc[m][2 * j];
        part[sgid][m][j * 8 + fn + 1] = acc[m][2 * j + 1];
      }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgid == 0) {
    const int c = int(lane);
    UNROLL for (int m = 0; m < MM; ++m) {
      if (m >= rows) break;
      float v = 0.0f;
      UNROLL for (int k = 0; k < KS; ++k) v += part[k][m][c];
      const ulong orow = ulong(r0 + m) * p.ldy;
      if (EPI == EPI_SWIGLU) {
        if ((c / 8) % 2 == 0 && nloc + c < p.Ne) {
          float u = 0.0f;
          UNROLL for (int k = 0; k < KS; ++k) u += part[k][m][c + 8];
          const int out = ((nloc + (c / 8) * 8) / 16) * 8 + (c % 8);
          ((device half*)Yv)[orow + out] = half(glu_gate(v) * u);
        }
      } else if (nloc + c < p.Ne) {
        device float* y = (device float*)Yv + orow + nloc + c;
        *y = (EPI == EPI_ADD) ? *y + v : v;
      }
    }
  }
  }  // column blocks
}

#define INSTANTIATE_MOE_GEMV_M(PRE, BITS, MM, KS, DIAG, TAG) INSTANTIATE_MOE_GEMV_MB(PRE, BITS, BITS, MM, KS, DIAG, TAG)
#define INSTANTIATE_MOE_GEMV_MB(PRE, BITS, BITS2, MM, KS, DIAG, TAG)                           \
  template [[host_name(PRE #MM "_store" TAG)]] kernel void moe_gemv<BITS, MM, 0, KS, DIAG, BITS2>( \
      device const half*, device const uint*, device const half4*, device const uint*,        \
      device void*, constant MoeGemmParams&, device int4*, device const uint*,                \
      device int*, device const uint*, device const half4*, device const uint*,               \
      device const float*, device int*, device float*, device float*, device int*,           \
      uint3, uint, uint);                                                                     \
  template [[host_name(PRE #MM "_swiglu" TAG)]] kernel void moe_gemv<BITS, MM, 2, KS, DIAG, BITS2>( \
      device const half*, device const uint*, device const half4*, device const uint*,        \
      device void*, constant MoeGemmParams&, device int4*, device const uint*,                \
      device int*, device const uint*, device const half4*, device const uint*,               \
      device const float*, device int*, device float*, device float*, device int*,           \
      uint3, uint, uint);
#define INSTANTIATE_MOE_GEMV_ROUTE(PRE, BITS, KS, TAG) INSTANTIATE_MOE_GEMV_ROUTE_B(PRE, BITS, BITS, KS, TAG)
#define INSTANTIATE_MOE_GEMV_ROUTE_B(PRE, BITS, BITS2, KS, TAG)                                \
  template [[host_name(PRE "1_swiglu" TAG)]] kernel void moe_gemv<BITS, 1, 2, KS, 2, BITS2>(    \
      device const half*, device const uint*, device const half4*, device const uint*,        \
      device void*, constant MoeGemmParams&, device int4*, device const uint*,                \
      device int*, device const uint*, device const half4*, device const uint*,               \
      device const float*, device int*, device float*, device float*, device int*,           \
      uint3, uint, uint);
#define INSTANTIATE_MOE_GEMV(PRE, BITS)                                                        \
  INSTANTIATE_MOE_GEMV_M(PRE, BITS, 1, 8, 0, "") INSTANTIATE_MOE_GEMV_M(PRE, BITS, 2, 8, 0, "")      \
  INSTANTIATE_MOE_GEMV_M(PRE, BITS, 1, 4, 0, "_k4") INSTANTIATE_MOE_GEMV_M(PRE, BITS, 2, 4, 0, "_k4") \
  INSTANTIATE_MOE_GEMV_M(PRE, BITS, 1, 2, 0, "_k2") INSTANTIATE_MOE_GEMV_M(PRE, BITS, 2, 2, 0, "_k2") \
  INSTANTIATE_MOE_GEMV_M(PRE, BITS, 1, 8, 1, "_diag")                                          \
  INSTANTIATE_MOE_GEMV_ROUTE(PRE, BITS, 8, "_route") INSTANTIATE_MOE_GEMV_ROUTE(PRE, BITS, 4, "_route_k4") \
  INSTANTIATE_MOE_GEMV_ROUTE(PRE, BITS, 2, "_route_k2")
INSTANTIATE_MOE_GEMV("moe_gemv4_", 4)
INSTANTIATE_MOE_GEMV("moe_gemv5_", 5)
INSTANTIATE_MOE_GEMV("moe_gemv6_", 6)
INSTANTIATE_MOE_GEMV("moe_gemv8_", 8)
// Mixed: BITS-bit routed experts with an 8-bit shared expert as tile k
// (name suffix _s8), 1-2 rows, the K-slice variants and the folded route.
#define INSTANTIATE_MOE_GEMV_S8(PRE, BITS)                                                     \
  INSTANTIATE_MOE_GEMV_MB(PRE, BITS, 8, 1, 8, 0, "_s8") INSTANTIATE_MOE_GEMV_MB(PRE, BITS, 8, 2, 8, 0, "_s8")       \
  INSTANTIATE_MOE_GEMV_MB(PRE, BITS, 8, 1, 4, 0, "_k4_s8") INSTANTIATE_MOE_GEMV_MB(PRE, BITS, 8, 2, 4, 0, "_k4_s8") \
  INSTANTIATE_MOE_GEMV_MB(PRE, BITS, 8, 1, 2, 0, "_k2_s8") INSTANTIATE_MOE_GEMV_MB(PRE, BITS, 8, 2, 2, 0, "_k2_s8") \
  INSTANTIATE_MOE_GEMV_ROUTE_B(PRE, BITS, 8, 8, "_route_s8") INSTANTIATE_MOE_GEMV_ROUTE_B(PRE, BITS, 8, 4, "_route_k4_s8") \
  INSTANTIATE_MOE_GEMV_ROUTE_B(PRE, BITS, 8, 2, "_route_k2_s8")
INSTANTIATE_MOE_GEMV_S8("moe_gemv4_", 4)
INSTANTIATE_MOE_GEMV_S8("moe_gemv5_", 5)
INSTANTIATE_MOE_GEMV_S8("moe_gemv6_", 6)

struct MoeCombineParams {
  int D;
  int k;
  int norm;   // 1: also write xn = rmsnorm(h) * wn
  float eps;
  int slice;  // columns per threadgroup
};

kernel void moe_combine(
    device float* h [[buffer(0)]],
    device const float* ys [[buffer(1)]],
    device const int* slotpos [[buffer(2)]],
    device const float* w [[buffer(3)]],
    device const float* shared [[buffer(4)]],
    device const float* sgate [[buffer(5)]],
    constant MoeCombineParams& p [[buffer(6)]],
    device const float* wn [[buffer(7)]],
    device half* xn [[buffer(8)]],
    device float* parts [[buffer(9)]],       // [rows][slices] partial sums of squares
    device atomic_int* ctr [[buffer(10)]],   // [rows] arrival counters, self-resetting
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgpg [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 ntg2 [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  threadgroup int last_tg;
  const uint ntg = ntg2.x;
  const int row = int(tgid.x), slice = int(tgid.y), nsl = int(tgpg.y);
  const float g = 1.0f / (1.0f + exp(-sgate[row]));
  const int c0 = slice * p.slice, c1 = min(p.D, c0 + p.slice);
  float ss = 0.0f;
  for (int d4 = c0 / 4 + int(tid); d4 < c1 / 4; d4 += int(ntg)) {
    float4 acc = 0.0f;
    for (int s = 0; s < p.k; ++s) {
      const int pos = slotpos[row * p.k + s];
      acc += w[row * p.k + s] * ((device const float4*)(ys + ulong(pos) * p.D))[d4];
    }
    acc += g * ((device const float4*)(shared + ulong(row) * p.D))[d4];
    device float* hp = h + ulong(row) * p.D + d4 * 4;
    const float4 v = *((device float4*)hp) + acc;
    if (p.norm && nsl > 1) {
      // read back by the finalizing threadgroup inside this dispatch
      atomic_store_explicit((device atomic_uint*)hp, as_type<uint>(v.x), memory_order_relaxed);
      atomic_store_explicit((device atomic_uint*)hp + 1, as_type<uint>(v.y), memory_order_relaxed);
      atomic_store_explicit((device atomic_uint*)hp + 2, as_type<uint>(v.z), memory_order_relaxed);
      atomic_store_explicit((device atomic_uint*)hp + 3, as_type<uint>(v.w), memory_order_relaxed);
    } else {
      *((device float4*)hp) = v;
    }
    ss += dot(v, v);
  }
  if (!p.norm) return;
  ss = simd_sum(ss);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float t = (lane < (ntg + 31) / 32) ? red[lane] : 0.0f;
    t = simd_sum(t);
    if (lane == 0) red[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = red[0];
  if (nsl > 1) {
    if (tid == 0) atomic_store_explicit((device atomic_uint*)(parts + row * nsl + slice), as_type<uint>(total), memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) last_tg = atomic_fetch_add_explicit(ctr + row, 1, memory_order_relaxed) == nsl - 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last_tg) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
    total = 0.0f;
    for (int s2 = 0; s2 < nsl; ++s2)
      total += as_type<float>(atomic_load_explicit((device atomic_uint*)(parts + row * nsl + s2), memory_order_relaxed));
    if (tid == 0) atomic_store_explicit(ctr + row, 0, memory_order_relaxed);
  }
  const float inv = rsqrt(total / float(p.D) + p.eps);
  for (int d4 = int(tid); d4 < p.D / 4; d4 += int(ntg)) {
    float4 v;
    if (nsl > 1) {
      device atomic_uint* hp = (device atomic_uint*)(h + ulong(row) * p.D + d4 * 4);
      v = float4(as_type<float>(atomic_load_explicit(hp, memory_order_relaxed)),
                 as_type<float>(atomic_load_explicit(hp + 1, memory_order_relaxed)),
                 as_type<float>(atomic_load_explicit(hp + 2, memory_order_relaxed)),
                 as_type<float>(atomic_load_explicit(hp + 3, memory_order_relaxed)));
    } else {
      v = ((device const float4*)(h + ulong(row) * p.D))[d4];
    }
    const float4 wv = ((device const float4*)wn)[d4];
    ((device half4*)(xn + ulong(row) * p.D))[d4] = half4(v * inv * wv);
  }
}

// Single-row expert down projection with the combine folded in (the
// structure of TensorFold's expert_down, MIT): threadgroup = one 32-column
// block of the output; KSX simdgroups per slot (slots 0..k-1: the row's routed
// experts from the route's tile table, slot k: the shared expert with its own
// matrices), each over its slice of K = Ie with moe_kloop; simdgroup 0 sums
// the slots in fixed order with the routing weights and sigmoid(shared gate)
// into the residual, and with `norm` the next layer's RMSNorm is finished by
// the last-arriving threadgroup (moe_combine's partial-sum and counter
// scheme). Replaces the down GEMV's ys round trip and the combine dispatch.
struct MoeDownCombParams {
  int D;    // output width
  int N;    // routed down matrix rows (E * D)
  int N2;   // shared down matrix rows (D)
  int Ie;   // K
  int k;    // routed slots (<= 16)
  int norm;
  float eps;
};

template <int BITS, int BITS2, int KSX>
kernel void moe_down_comb(
    device const half* A [[buffer(0)]],        // act rows [k + 1][Ie], slot order
    device const uint* Wr [[buffer(1)]],
    device const half4* SB [[buffer(2)]],
    device const uint* Wh [[buffer(3)]],
    device float* h [[buffer(4)]],             // the row's residual [D]
    constant MoeDownCombParams& p [[buffer(5)]],
    device const int4* tiles [[buffer(6)]],    // tiles[s].x: slot s's expert
    device const float* w [[buffer(7)]],       // routing weights [k]
    device const float* sgate [[buffer(8)]],   // shared gate logit
    device const uint* Wr2 [[buffer(9)]],
    device const half4* SB2 [[buffer(10)]],
    device const uint* Wh2 [[buffer(11)]],
    device const float* wn [[buffer(12)]],     // next norm's weight
    device half* xn [[buffer(13)]],
    device float* parts [[buffer(14)]],        // [blocks] partial sums of squares
    device atomic_int* ctr [[buffer(15)]],     // arrival counter (self-resetting)
    device const int* slotpos [[buffer(16)]],  // top-k slot -> its tile (= act row): slot order after the fused
                                               // route, expert order after moe_sort (the unfused route)
    uint tgid [[threadgroup_position_in_grid]],
    uint ntgs [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[17 * KSX][32];
  threadgroup int last_tg;
  const int slot = int(sgid) / KSX, ksl = int(sgid) % KSX;
  const bool sh = slot == p.k;
  const int nloc = int(tgid) * 32;
  const int e = sh ? 0 : tiles[slot].x;
  const int n0 = sh ? nloc : e * p.D + nloc;
  const int nb = n0 / 32;
  const int KGall = p.Ie / 64;
  const int per = (KGall + KSX - 1) / KSX;
  const int g_begin = ksl * per, g_end = min(KGall, g_begin + per);
  const int Np2 = ((sh ? p.N2 : p.N) + 31) / 32 * 16;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  device const half* arow = A + ulong(slot) * ulong(p.Ie);
  float acc[8];
  UNROLL for (int c = 0; c < 8; ++c) acc[c] = 0.0f;
  if (sh)
    moe_kloop<BITS2, 1, KSX>(Wr2, SB2, Wh2, nb, KGall, g_begin, g_end, Np2, n0, 1, arow, arow, ushort(lane), fm, fn, acc);
  else
    moe_kloop<BITS, 1, KSX>(Wr, SB, Wh, nb, KGall, g_begin, g_end, Np2, n0, 1, arow, arow, ushort(lane), fm, fn, acc);
  UNROLL for (int c = 0; c < 8; ++c) {
    float v = acc[c];
    v += simd_shuffle_xor(v, ushort(2));
    v += simd_shuffle_xor(v, ushort(4));
    v += simd_shuffle_xor(v, ushort(16));
    acc[c] = v;
  }
  if (fm == 0) {
    UNROLL for (int j = 0; j < 4; ++j) {
      part[sgid][j * 8 + fn] = acc[2 * j];
      part[sgid][j * 8 + fn + 1] = acc[2 * j + 1];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ss = 0.0f;
  if (sgid == 0) {
    const int c = int(lane);
    float routed = 0.0f;
    for (int s = 0; s < p.k; ++s) {
      const int t = slotpos[s];  // simdgroups are laid out by tile
      float y = 0.0f;
      UNROLL for (int q = 0; q < KSX; ++q) y += part[t * KSX + q][c];
      routed += w[s] * y;
    }
    float ysh = 0.0f;
    UNROLL for (int q = 0; q < KSX; ++q) ysh += part[p.k * KSX + q][c];
    const float g = 1.0f / (1.0f + exp(-sgate[0]));
    device float* hp = h + nloc + c;
    const float v = *hp + (routed + g * ysh);
    if (p.norm && ntgs > 1) atomic_store_explicit((device atomic_uint*)hp, as_type<uint>(v), memory_order_relaxed);
    else *hp = v;
    ss = simd_sum(v * v);
  }
  if (!p.norm) return;
  float total = ss;
  if (ntgs > 1) {
    if (tid == 0) atomic_store_explicit((device atomic_uint*)(parts + tgid), as_type<uint>(ss), memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) last_tg = atomic_fetch_add_explicit(ctr, 1, memory_order_relaxed) == int(ntgs) - 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last_tg) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
    total = 0.0f;
    for (uint b = 0; b < ntgs; ++b)
      total += as_type<float>(atomic_load_explicit((device atomic_uint*)(parts + b), memory_order_relaxed));
    if (tid == 0) atomic_store_explicit(ctr, 0, memory_order_relaxed);
  }
  const float inv = rsqrt(total / float(p.D) + p.eps);
  for (int d = int(tid); d < p.D; d += int(ntg)) {
    const float v = ntgs > 1 ? as_type<float>(atomic_load_explicit((device atomic_uint*)(h + d), memory_order_relaxed)) : h[d];
    xn[d] = half(v * inv * wn[d]);
  }
}

#define INSTANTIATE_DOWN_COMB(PRE, BITS, BITS2, KSX)                                               \
  template [[host_name(PRE)]] kernel void moe_down_comb<BITS, BITS2, KSX>(                        \
      device const half*, device const uint*, device const half4*, device const uint*,           \
      device float*, constant MoeDownCombParams&, device const int4*, device const float*,       \
      device const float*, device const uint*, device const half4*, device const uint*,          \
      device const float*, device half*, device float*, device atomic_int*, device const int*,   \
      uint, uint, uint, uint, uint, uint);
#define INSTANTIATE_DOWN_COMB_B(BITS)                                                              \
  INSTANTIATE_DOWN_COMB("moe_down_comb" #BITS "_k1", BITS, BITS, 1)                                \
  INSTANTIATE_DOWN_COMB("moe_down_comb" #BITS "_k2", BITS, BITS, 2)
#define INSTANTIATE_DOWN_COMB_S8(BITS)                                                             \
  INSTANTIATE_DOWN_COMB("moe_down_comb" #BITS "_k1_s8", BITS, 8, 1)                                \
  INSTANTIATE_DOWN_COMB("moe_down_comb" #BITS "_k2_s8", BITS, 8, 2)
INSTANTIATE_DOWN_COMB_B(4)
INSTANTIATE_DOWN_COMB_B(5)
INSTANTIATE_DOWN_COMB_B(6)
INSTANTIATE_DOWN_COMB_B(8)
INSTANTIATE_DOWN_COMB_S8(4)
INSTANTIATE_DOWN_COMB_S8(5)
INSTANTIATE_DOWN_COMB_S8(6)
