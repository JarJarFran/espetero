// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Row gather (fp32 or fp16 rows of `width` elements).
kernel void gather_rows_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const int* rows [[buffer(2)]],
    constant int& width [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) * 4 >= width) return;
  *((device float4*)(dst + ulong(gid.y) * width) + gid.x) =
      *((device const float4*)(src + ulong(rows[gid.y]) * width) + gid.x);
}

kernel void gather_rows_f16(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    device const int* rows [[buffer(2)]],
    constant int& width [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) * 4 >= width) return;
  *((device half4*)(dst + ulong(gid.y) * width) + gid.x) =
      *((device const half4*)(src + ulong(rows[gid.y]) * width) + gid.x);
}

// ---------------------------------------------------------------------------
// LM head scoring: logits [N, V] fp32 from rgemm, then per-row
// log-sum-exp and candidate gather (same logits, so logprobs are consistent).
// ---------------------------------------------------------------------------
kernel void lse_rows(
    device const float* logits [[buffer(0)]],
    device float* lse [[buffer(1)]],
    constant int& V [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  device const float4* x = (device const float4*)(logits + ulong(row) * V);
  const int V4 = V / 4;
  float m = -INFINITY;
  for (int i = tid; i < V4; i += ntg) {
    const float4 v = x[i];
    m = max(m, max(max(v.x, v.y), max(v.z, v.w)));
  }
  m = simd_max(m);
  if (lane == 0) red[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float t = (lane < (ntg + 31) / 32) ? red[lane] : -INFINITY;
    t = simd_max(t);
    if (lane == 0) red[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float gm = red[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = 0.0f;
  for (int i = tid; i < V4; i += ntg) {
    const float4 v = exp(x[i] - gm);
    s += v.x + v.y + v.z + v.w;
  }
  s = simd_sum(s);
  if (lane == 0) red[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float t = (lane < (ntg + 31) / 32) ? red[lane] : 0.0f;
    t = simd_sum(t);
    if (lane == 0) lse[row] = gm + log(t);
  }
}

// First index of the row maximum (matches argmax tie-breaking).
kernel void argmax_rows(
    device const float* logits [[buffer(0)]],
    device int* out [[buffer(1)]],
    constant int& V [[buffer(2)]],
    constant int& ld [[buffer(3)]],  // row stride (>= V: only a vocabulary prefix is searched)
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float redf[32];
  threadgroup int redi[32];
  device const float* x = logits + ulong(row) * ld;
  float m = -INFINITY;
  for (int i = tid; i < V; i += ntg) m = max(m, x[i]);
  m = simd_max(m);
  if (lane == 0) redf[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float t = (lane < (ntg + 31) / 32) ? redf[lane] : -INFINITY;
    t = simd_max(t);
    if (lane == 0) redf[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float gm = redf[0];
  int best = INT_MAX;
  for (int i = tid; i < V; i += ntg)
    if (x[i] == gm) {
      best = i;
      break;
    }
  best = simd_min(best);
  if (lane == 0) redi[sg] = best;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    int t = (lane < (ntg + 31) / 32) ? redi[lane] : INT_MAX;
    t = simd_min(t);
    if (lane == 0) out[row] = t;
  }
}

// Best and runner-up ids of each row (lowest id on ties, like argmax_rows):
// the MTP draft pass's candidates for sibling leaves.
kernel void argmax2_rows(
    device const float* logits [[buffer(0)]],
    device int* out [[buffer(1)]],
    constant int& V [[buffer(2)]],
    constant int& ld [[buffer(3)]],
    device int* out2 [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float redf[32];
  threadgroup int redi[32];
  device const float* x = logits + ulong(row) * ld;
  const int nsg = int(ntg + 31) / 32;
  // max over ids other than `skip`, then the lowest id holding it
  auto top = [&](int skip) -> int {
    float m = -INFINITY;
    for (int i = tid; i < V; i += ntg)
      if (i != skip) m = max(m, x[i]);
    m = simd_max(m);
    if (lane == 0) redf[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float t = (int(lane) < nsg) ? redf[lane] : -INFINITY;
      t = simd_max(t);
      if (lane == 0) redf[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gm = redf[0];
    int best = INT_MAX;
    for (int i = tid; i < V; i += ntg)
      if (i != skip && x[i] == gm) {
        best = i;
        break;
      }
    best = simd_min(best);
    if (lane == 0) redi[sg] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      int t = (int(lane) < nsg) ? redi[lane] : INT_MAX;
      t = simd_min(t);
      if (lane == 0) redi[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int r = redi[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);  // redf / redi are reused by the next call
    return r;
  };
  const int b = top(-1);
  const int b2 = top(b);
  if (tid == 0) {
    out[row] = b;
    out2[row] = b2 == INT_MAX ? b : b2;
  }
}

// ---------------------------------------------------------------------------
// Coarse MTP draft head: the 2-bit head's logits pick C column blocks
// of 32 vocabulary rows per draft row, the file's own head recomputes those
// blocks, and the draft is the best (and runner-up) exact logit among them.
// ---------------------------------------------------------------------------

// Maximum of each 32-column block of each row over columns [0, V).
kernel void block_max_rows(
    device const float* logits [[buffer(0)]],
    device float* bmax [[buffer(1)]],
    constant int& V [[buffer(2)]],
    constant int& ld [[buffer(3)]],
    constant int& NB [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int b = int(gid.x), row = int(gid.y);
  if (b >= NB) return;
  device const float* x = logits + ulong(row) * ld + b * 32;
  float m = -INFINITY;
  if (b * 32 + 32 <= V) {
    UNROLL for (int i = 0; i < 8; ++i) {
      const float4 v = ((device const float4*)x)[i];
      m = max(m, max(max(v.x, v.y), max(v.z, v.w)));
    }
  } else {
    for (int i = 0; b * 32 + i < V; ++i) m = max(m, x[i]);
  }
  bmax[ulong(row) * NB + b] = m;
}

// The C blocks with the largest maxima of each row, in block order, ties to
// the lower block; slots past min(C, NB) get -1. One threadgroup of 1024 per
// row, NB <= 8192: a radix select of the C-th largest key (8 bits a pass)
// followed by two prefix scans, so the choice is deterministic.
kernel void topc_blocks(
    device const float* bmax [[buffer(0)]],
    device int* list [[buffer(1)]],
    constant int& NB [[buffer(2)]],
    constant int& C [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int PER = 8;
  threadgroup atomic_uint hist[256];
  threadgroup uint sel[2];
  threadgroup uint sgsum[32];
  device const float* x = bmax + ulong(row) * NB;
  // Order-preserving keys (larger float, larger key); absent blocks are 0.
  uint key[PER];
  UNROLL for (int i = 0; i < PER; ++i) {
    const int b = int(tid) * PER + i;
    const uint u = b < NB ? as_type<uint>(x[b]) : 0u;
    key[i] = b < NB ? ((u & 0x80000000u) ? ~u : (u | 0x80000000u)) : 0u;
  }
  const int take = min(C, NB);
  device int* dst = list + ulong(row) * C;
  if (take <= 0) {
    for (int i = int(tid); i < C; i += 1024) dst[i] = -1;
    return;
  }
  uint prefix = 0, pmask = 0;
  uint need = uint(take);  // rank of the wanted key among keys matching `prefix`
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tid < 256) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    UNROLL for (int i = 0; i < PER; ++i)
      if ((key[i] & pmask) == prefix) atomic_fetch_add_explicit(&hist[(key[i] >> shift) & 255u], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      // lane l holds bins 255 - 8l down to 248 - 8l
      uint cnt[8], tot = 0;
      UNROLL for (int j = 0; j < 8; ++j) {
        cnt[j] = atomic_load_explicit(&hist[255 - 8 * int(lane) - j], memory_order_relaxed);
        tot += cnt[j];
      }
      const uint above = simd_prefix_exclusive_sum(tot);
      if (above < need && above + tot >= need) {
        uint a = above;
        for (int j = 0; j < 8; ++j) {
          if (a + cnt[j] >= need) {
            sel[0] = uint(255 - 8 * int(lane) - j);
            sel[1] = a;
            break;
          }
          a += cnt[j];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= sel[0] << shift;
    pmask |= 255u << shift;
    need -= sel[1];
  }
  // prefix: the take-th largest key T; need: how many keys equal to T to keep.
  const uint T = prefix;
  auto scan = [&](uint v) -> uint {  // exclusive prefix sum over the threadgroup
    const uint e = simd_prefix_exclusive_sum(v);
    if (lane == 31) sgsum[sg] = e + v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) sgsum[lane] = simd_prefix_exclusive_sum(sgsum[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint r = sgsum[sg] + e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
  };
  uint gt = 0, eq = 0;
  UNROLL for (int i = 0; i < PER; ++i) {
    gt += key[i] > T ? 1u : 0u;
    eq += key[i] == T ? 1u : 0u;
  }
  const uint e0 = scan(eq);
  const uint tk = e0 < need ? min(eq, need - e0) : 0u;  // ties this thread keeps (its lowest blocks)
  uint o = scan(gt + tk), t = 0;
  UNROLL for (int i = 0; i < PER; ++i) {
    bool f = key[i] > T;
    if (key[i] == T && t < tk) {
      f = true;
      ++t;
    }
    if (f) dst[o++] = int(tid) * PER + i;
  }
  for (int i = take + int(tid); i < C; i += 1024) dst[i] = -1;
}

// Best and runner-up id over the columns of each row's listed blocks (C per
// row, -1 skipped; ids >= V skipped), lowest id on ties like argmax2_rows.
kernel void argmax2_list(
    device const float* logits [[buffer(0)]],
    device int* out [[buffer(1)]],
    constant int& V [[buffer(2)]],
    constant int& ld [[buffer(3)]],
    device int* out2 [[buffer(4)]],
    device const int* list [[buffer(5)]],
    constant int& C [[buffer(6)]],
    constant int& want2 [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float redf[32];
  threadgroup int redi[32];
  device const float* x = logits + ulong(row) * ld;
  device const int* bl = list + ulong(row) * C;
  const int nsg = int(ntg + 31) / 32;
  const int n = C * 32;
  auto id_at = [&](int t) -> int {
    const int b = bl[t / 32];
    const int i = b * 32 + t % 32;
    return (b < 0 || i >= V) ? -1 : i;
  };
  auto top = [&](int skip) -> int {
    float m = -INFINITY;
    for (int t = int(tid); t < n; t += int(ntg)) {
      const int i = id_at(t);
      if (i >= 0 && i != skip) m = max(m, x[i]);
    }
    m = simd_max(m);
    if (lane == 0) redf[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float v = (int(lane) < nsg) ? redf[lane] : -INFINITY;
      v = simd_max(v);
      if (lane == 0) redf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gm = redf[0];
    int best = INT_MAX;
    for (int t = int(tid); t < n; t += int(ntg)) {
      const int i = id_at(t);
      if (i >= 0 && i != skip && x[i] == gm) best = min(best, i);
    }
    best = simd_min(best);
    if (lane == 0) redi[sg] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      int v = (int(lane) < nsg) ? redi[lane] : INT_MAX;
      v = simd_min(v);
      if (lane == 0) redi[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int r = redi[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
  };
  const int b = top(-1);
  const int b2 = want2 ? top(b) : b;
  if (tid == 0) {
    out[row] = b == INT_MAX ? 0 : b;
    if (want2) out2[row] = b2 == INT_MAX ? out[row] : b2;
  }
}

kernel void gather_cands(
    device const float* logits [[buffer(0)]],
    device const int* cand_ids [[buffer(1)]],
    device const int* cand_owner [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& V [[buffer(4)]],
    constant int& total [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
  if (int(i) >= total) return;
  out[i] = logits[ulong(cand_owner[i]) * V + cand_ids[i]];
}


// dst[rows[i]] = src[i] (fp32 rows of `width` elements).
kernel void scatter_rows_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const int* rows [[buffer(2)]],
    constant int& width [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) * 4 >= width) return;
  *((device float4*)(dst + ulong(rows[gid.y]) * width) + gid.x) =
      *((device const float4*)(src + ulong(gid.y) * width) + gid.x);
}


// Rows between the scratch and slot-resident fp32 rows (the MTP draft state):
// dst[offs[i] + x] = src[rows[i] * width + x] and the reverse.
kernel void rows_to_pool(
    device const float* src [[buffer(0)]],
    device float* pool [[buffer(1)]],
    device const int* rows [[buffer(2)]],
    device const ulong* offs [[buffer(3)]],
    constant int& width [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) * 4 >= width) return;
  *((device float4*)(pool + offs[gid.y]) + gid.x) = *((device const float4*)(src + ulong(rows[gid.y]) * width) + gid.x);
}

kernel void pool_to_rows(
    device const float* pool [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const ulong* offs [[buffer(2)]],
    constant int& width [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) * 4 >= width) return;
  *((device float4*)(dst + ulong(gid.y) * width) + gid.x) = *((device const float4*)(pool + offs[gid.y]) + gid.x);
}

// ---------------------------------------------------------------------------
// Top-K candidates of sampled readouts. For a readout row r with
// scaled logits x = logit * inv_t[r]: the maximum M, the full-vocabulary sum
// of exp(x - M) and the K largest x with their token ids (ties: the lower id),
// sorted by (x desc, id asc) — exactly the head the CPU sampler builds with
// nth_element + sort for 1 <= top_k <= K. Two dispatches: topk_part takes the
// top K of every chunk of <= 4096 entries (their union contains the global
// top K), topk_merge selects and sorts the global top K.
// ---------------------------------------------------------------------------
constant constexpr int kTopKMax = 32;
struct TopKParams {
  int V;      // vocabulary (row length of the logits)
  int K;      // candidates (<= kTopKMax)
  int chunk;  // entries per part (<= 4096)
  int parts;
  int chain;  // 1: sampled tokens also go to the decode chain's token buffer
  int chain_off;  // ... at chain_tok[chain_off + row]
};
// Per-readout draw: the sampled token is argmax_i log p_i + G(seed,
// pos, id_i) over the top_k head cut by min_p and top_p, G Gumbel noise from
// the splitmix64 hash of engine/sampling.cpp (24-bit uniforms here).
struct TopKSample {
  uint seed_lo, seed_hi;
  int pos_lo, pos_hi;
  float top_p, min_p;
  int top_k;
  int draw;  // 0: no draw for this readout
};
static inline ulong tk_mix(ulong x) {
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ul;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBul;
  return x ^ (x >> 31);
}
static inline float tk_uniform(ulong seed, ulong pos, ulong id) {
  ulong x = tk_mix(seed + 0x9E3779B97F4A7C15ul);
  x = tk_mix(x ^ (pos * 0xD1B54A32D192ED03ul));
  x = tk_mix(x ^ id);
  return float(x >> 40) * 0x1.0p-24f + 0x1.0p-25f;  // (0, 1)
}

// Order-preserving map of a float to an unsigned key (larger float, larger key).
static inline uint topk_key(float x) {
  const uint u = as_type<uint>(x);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Largest key t with #{keys >= t} >= need, over NPER keys per thread of a
// 256-thread threadgroup (bitwise construction, 32 counting rounds).
template <int NPER>
static inline uint topk_threshold(thread const uint* k, int need, threadgroup int* red, uint sg, uint lane) {
  uint t = 0;
  for (int bit = 31; bit >= 0; --bit) {
    const uint cand = t | (1u << bit);
    int c = 0;
    UNROLL for (int i = 0; i < NPER; ++i) c += k[i] >= cand ? 1 : 0;
    c = simd_sum(c);
    if (lane == 0) red[sg] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int tot = 0;
    UNROLL for (int s = 0; s < 8; ++s) tot += red[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tot >= need) t = cand;
  }
  return t;
}

kernel void topk_part(
    device const float* logits [[buffer(0)]],
    device const float* inv_t [[buffer(1)]],
    device float* cand_x [[buffer(2)]],   // [rows][parts][K]
    device int* cand_i [[buffer(3)]],
    device float2* stats [[buffer(4)]],   // [rows][parts]: (chunk max, sum of exp(x - chunk max))
    constant TopKParams& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int PER = 16;  // 256 threads x 16 = 4096 entries per part at most
  threadgroup float redf[8];
  threadgroup int red[8];
  threadgroup int scan[256];
  threadgroup atomic_int cnt;
  const int part = int(tg.x), row = int(tg.y);
  const float it = inv_t[row];
  const int c0 = part * p.chunk, c1 = min(p.V, c0 + p.chunk);
  const int b0 = c0 + int(tid) * PER;  // contiguous per thread: thread order = id order
  device const float* L = logits + ulong(row) * ulong(p.V);
  float v[PER];
  uint k[PER];
  float lm = -INFINITY;
  UNROLL for (int i = 0; i < PER; ++i) {
    const int idx = b0 + i;
    const bool ok = idx < c1;
    v[i] = ok ? L[idx] * it : -INFINITY;
    k[i] = ok ? topk_key(v[i]) : 0u;
    lm = max(lm, v[i]);
  }
  lm = simd_max(lm);
  if (lane == 0) redf[sg] = lm;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  lm = redf[0];
  UNROLL for (int s = 1; s < 8; ++s) lm = max(lm, redf[s]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ls = 0.0f;
  UNROLL for (int i = 0; i < PER; ++i)
    if (b0 + i < c1) ls += exp(v[i] - lm);
  ls = simd_sum(ls);
  if (lane == 0) redf[sg] = ls;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float t = 0.0f;
    for (int s = 0; s < 8; ++s) t += redf[s];
    stats[ulong(row) * ulong(p.parts) + ulong(part)] = float2(lm, t);
    atomic_store_explicit(&cnt, 0, memory_order_relaxed);
  }
  const int need = min(p.K, max(0, c1 - c0));
  const uint t = topk_threshold<PER>(k, need, red, sg, lane);
  // Strictly above the threshold: all; at it: the lowest ids, up to K.
  int gt = 0, eq = 0;
  UNROLL for (int i = 0; i < PER; ++i) {
    gt += (b0 + i < c1 && k[i] > t) ? 1 : 0;
    eq += (b0 + i < c1 && k[i] == t) ? 1 : 0;
  }
  gt = simd_sum(gt);
  if (lane == 0) red[sg] = gt;
  scan[tid] = eq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int n_gt = 0;
  UNROLL for (int s = 0; s < 8; ++s) n_gt += red[s];
  if (tid == 0) {
    int a = 0;
    for (int j = 0; j < 256; ++j) {
      const int e = scan[j];
      scan[j] = a;
      a += e;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int rank = scan[tid];
  const ulong base = (ulong(row) * ulong(p.parts) + ulong(part)) * ulong(p.K);
  UNROLL for (int i = 0; i < PER; ++i) {
    if (b0 + i >= c1) continue;
    bool take = k[i] > t;
    if (k[i] == t) {
      take = rank < need - n_gt;
      ++rank;
    }
    if (take) {
      const int slot = atomic_fetch_add_explicit(&cnt, 1, memory_order_relaxed);
      cand_x[base + slot] = v[i];
      cand_i[base + slot] = b0 + i;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int filled = atomic_load_explicit(&cnt, memory_order_relaxed);
  if (int(tid) >= filled && int(tid) < p.K) {
    cand_x[base + tid] = -INFINITY;
    cand_i[base + tid] = INT_MAX;
  }
}

// out row (stride 2 * kTopKMax + 2 words): ids [K], x [K] (sorted), M, total.
kernel void topk_merge(
    device const float* cand_x [[buffer(0)]],
    device const int* cand_i [[buffer(1)]],
    device const float2* stats [[buffer(2)]],
    device uint* out [[buffer(3)]],
    constant TopKParams& p [[buffer(4)]],
    device const TopKSample* samp [[buffer(5)]],
    device int* chain_tok [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int PER = 8;  // 256 threads x 8 = 2048 candidates at most (parts * K)
  threadgroup int red[8];
  threadgroup atomic_int cnt, n_tie;
  threadgroup float fx[kTopKMax];
  threadgroup int fi[kTopKMax];
  threadgroup int tie_i[2048];
  threadgroup float tie_x[2048];
  const int n = p.parts * p.K;
  const ulong base = ulong(row) * ulong(n);
  float v[PER];
  int id[PER];
  uint k[PER];
  UNROLL for (int i = 0; i < PER; ++i) {
    const int c = int(tid) * PER + i;
    const bool ok = c < n;
    v[i] = ok ? cand_x[base + c] : -INFINITY;
    id[i] = ok ? cand_i[base + c] : INT_MAX;
    k[i] = (ok && id[i] != INT_MAX) ? topk_key(v[i]) : 0u;
  }
  if (tid == 0) {
    atomic_store_explicit(&cnt, 0, memory_order_relaxed);
    atomic_store_explicit(&n_tie, 0, memory_order_relaxed);
  }
  const uint t = topk_threshold<PER>(k, p.K, red, sg, lane);
  // Above the threshold: straight in; at it: gathered, then ranked by id.
  UNROLL for (int i = 0; i < PER; ++i) {
    if (k[i] == 0u) continue;
    if (k[i] > t) {
      const int s = atomic_fetch_add_explicit(&cnt, 1, memory_order_relaxed);
      fx[s] = v[i];
      fi[s] = id[i];
    } else if (k[i] == t) {
      const int s = atomic_fetch_add_explicit(&n_tie, 1, memory_order_relaxed);
      tie_x[s] = v[i];
      tie_i[s] = id[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int n_gt = atomic_load_explicit(&cnt, memory_order_relaxed);
  const int nt = atomic_load_explicit(&n_tie, memory_order_relaxed);
  for (int j = int(tid); j < nt; j += 256) {
    int r = 0;
    for (int q = 0; q < nt; ++q) r += tie_i[q] < tie_i[j] ? 1 : 0;
    if (n_gt + r < p.K) {
      fx[n_gt + r] = tie_x[j];
      fi[n_gt + r] = tie_i[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device uint* o = out + ulong(row) * ulong(2 * kTopKMax + 3);
  threadgroup float sx[kTopKMax];
  threadgroup int si[kTopKMax];
  if (int(tid) < p.K) {
    const float x = fx[tid];
    const int i = fi[tid];
    int r = 0;
    for (int q = 0; q < p.K; ++q) r += (fx[q] > x || (fx[q] == x && fi[q] < i)) ? 1 : 0;
    o[r] = uint(i);
    o[kTopKMax + r] = as_type<uint>(x);
    sx[r] = x;
    si[r] = i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float M = -INFINITY;
    for (int q = 0; q < p.parts; ++q) M = max(M, stats[ulong(row) * ulong(p.parts) + ulong(q)].x);
    float tot = 0.0f;
    for (int q = 0; q < p.parts; ++q) {
      const float2 s = stats[ulong(row) * ulong(p.parts) + ulong(q)];
      if (s.x > -INFINITY) tot += s.y * exp(s.x - M);
    }
    o[2 * kTopKMax] = as_type<uint>(M);
    o[2 * kTopKMax + 1] = as_type<uint>(tot);
    // The draw: the CPU sampler's head (p = exp(x - M) / tot, min_p,
    // top_p), then position-keyed Gumbel-max.
    int sampled = -1;
    const TopKSample sp = samp[row];
    if (sp.draw) {
      const int head = min(p.K, max(1, sp.top_k));
      float pr[kTopKMax];
      for (int q = 0; q < head; ++q) pr[q] = exp(sx[q] - M) / tot;
      int keep = head;
      if (sp.min_p > 0.0f) {
        const float floor_p = sp.min_p * pr[0];
        keep = 0;
        while (keep < head && pr[keep] >= floor_p) ++keep;
      }
      if (sp.top_p < 1.0f) {
        float sum = 0.0f;
        for (int q = 0; q < keep; ++q) sum += pr[q];
        const float target = sp.top_p * sum;
        float acc = 0.0f;
        int cut = keep;
        for (int q = 0; q < keep; ++q) {
          acc += pr[q];
          if (acc >= target) {
            cut = q + 1;
            break;
          }
        }
        keep = cut;
      }
      const ulong seed = (ulong(sp.seed_hi) << 32) | ulong(sp.seed_lo);
      const ulong pos = (ulong(uint(sp.pos_hi)) << 32) | ulong(uint(sp.pos_lo));
      float best = -INFINITY;
      sampled = si[0];
      for (int q = 0; q < max(keep, 1); ++q) {
        if (!(pr[q] > 0.0f)) continue;
        const float g = -log(-log(tk_uniform(seed, pos, ulong(long(si[q])))));
        const float sc = log(pr[q]) + g;
        if (sc > best) {
          best = sc;
          sampled = si[q];
        }
      }
      if (p.chain) chain_tok[p.chain_off + int(row)] = sampled;
    }
    o[2 * kTopKMax + 2] = uint(sampled);
  }
}
