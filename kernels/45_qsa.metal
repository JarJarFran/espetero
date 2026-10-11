// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ---------------------------------------------------------------------------
// Qwen sparse attention (QSA; Qwen4-Exp, docs/qwen4_exp.md 4). A query
// row at sequence index s sees n = s + 1 tokens; while its nb = n / 4
// complete blocks of 4 tokens number <= 512 it attends all of them (dense
// causal, exact). Past that it attends the 512 blocks whose pooled indexer
// keys score highest, plus the tail [4 nb, s] (<= 3 tokens). Per full
// attention layer and pass:
//   qsa_prep          indexer queries (4 heads x 128: q_layernorm, M-RoPE on
//                     dims 0..63) and raw keys, rounded to bf16 as in the
//                     reference (mlx-vlm runs the model in bf16)
//   qsa_pool          the pooled key of every block a pass row completes:
//                     mean of the 4 raw keys in fp32, bf16, k_layernorm,
//                     M-RoPE at the block's first token
//   qsa_score_cached  ReLU-summed head scores (fp32) of the blocks entirely in
//   qsa_score_batch   the cache (shared by a tree's rows; MMA) and of the
//                     blocks ending in pass rows (per row: drafts differ)
//   qsa_select        top 512 per row: radix select on the fp32 bits (scores
//                     are >= 0); ties at the threshold go to the lowest blocks
//   qsa_tiles         per 8-row tile: the cached blocks any of its rows reads
//                     (selected, its tail, or all for rows still dense),
//                     ascending, each with an 8-bit row mask
//   flash_attention_qsa   flash attention over those blocks, plus the batch
//                     keys with each row's own selection
//   qsa_save          raw keys, pooled keys and positions of the rows a pass
//                     saves, into the index pool (the KV pool's block ids)
// Index pool block (halves): raw keys [bt][128] | pooled keys [bt/4][128] |
// positions [bt] (int4).
// ---------------------------------------------------------------------------
constant constexpr int QSA_HD = 128;
constant constexpr int QSA_IH = 4;       // indexer query heads
constant constexpr int QSA_TOPK = 512;   // blocks per row (budget 2048 / 4)
constant constexpr int QSA_LMAX = 4112;  // union entries per 8-row tile (8 x 513, rounded up)
constant constexpr uint QSA_NONE = 0xFFFFFFFFu;

struct QsaParams {
  int ld;        // indexer projection row stride (floats)
  int bt;        // tokens per KV block
  int iblk;      // halves per index block
  int pool_off;  // halves: pooled keys within a block
  int pos_off;   // halves: positions within a block
  int nbmax;     // score row stride (blocks)
  int nbw;       // selection bitmap words per sparse row
  float eps;
  ulong ibase;   // halves: this layer's region of the index pool
};

// Rows that need sparse work, in groups of <= 8 consecutive rows of one tree
// (the attention tiles' rows).
struct QsaTile {
  int tree;
  int r0;    // absolute first row
  int rows;  // <= 8
  int pad;
};

// Per pass row (int4): x = sequence index s, y = nb = (s + 1) / 4, z = sparse
// row index (scores, bitmap) or -1 while dense, w = tree.

static inline float bf16r(float x) {
  uint u = as_type<uint>(x);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return as_type<float>(u & 0xFFFF0000u);
}

// The batch row at sequence index i >= P on the path of a row with
// descriptor d (root rows: d = 0; leaf rows: d.x = segment start, d.z = root
// rows): the root rows first, then the row's own segment.
static inline int qsa_path_row(int base, int P, int4 d, int i) {
  const int j = i - P;
  return base + (j < d.z ? j : d.x + (j - d.z));
}

// M-RoPE on dims 0..63 of a 128-wide head held as 4 dims per lane
// (rotate_half pairs (i, i + 32): lane l and l ^ 8), rounded to bf16.
static inline void qsa_rope(thread float* x, int4 pr, uint lane, device const float* inv_freq, device const int* rope_sel) {
  UNROLL for (int i = 0; i < 4; ++i) {
    const float partner = simd_shuffle_xor(x[i], ushort(8));
    if (lane < 16) {
      const int f = int(lane * 4 + i) & 31;
      const int sel = rope_sel[f];
      const float position = float(sel == 0 ? pr.x : (sel == 1 ? pr.y : pr.z));
      const float ang = position * inv_freq[f];
      const float c = precise::cos(ang);
      const float s = precise::sin(ang);
      x[i] = bf16r(lane < 8 ? (x[i] * c - partner * s) : (x[i] * c + partner * s));
    }
  }
}

// One simdgroup per (row, head): heads 0..3 the queries, 4 the raw key.
kernel void qsa_prep(
    device const float* X [[buffer(0)]],
    device const float* qn_w [[buffer(1)]],
    device const int4* pos [[buffer(2)]],
    device const float* inv_freq [[buffer(3)]],
    device const int* rope_sel [[buffer(4)]],
    device half* IQ [[buffer(5)]],
    device half* IK [[buffer(6)]],
    constant QsaParams& p [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const ulong row = tgpig.y;
  const int head = int(sg);
  device const float* src = X + row * ulong(p.ld) + head * QSA_HD + lane * 4;
  float x[4];
  UNROLL for (int i = 0; i < 4; ++i) x[i] = bf16r(src[i]);
  if (head == QSA_IH) {
    UNROLL for (int i = 0; i < 4; ++i) IK[row * QSA_HD + lane * 4 + i] = half(x[i]);
    return;
  }
  float ss = 0.0f;
  UNROLL for (int i = 0; i < 4; ++i) ss += x[i] * x[i];
  ss = simd_sum(ss);
  const float inv = rsqrt(ss / float(QSA_HD) + p.eps);
  UNROLL for (int i = 0; i < 4; ++i) x[i] = bf16r(x[i] * inv * qn_w[lane * 4 + i]);
  qsa_rope(x, pos[row], lane, inv_freq, rope_sel);
  device half* q = IQ + (row * QSA_IH + head) * QSA_HD + lane * 4;
  UNROLL for (int i = 0; i < 4; ++i) q[i] = half(x[i]);
}

// One simdgroup per row; rows whose own token completes a block (s % 4 == 3).
kernel void qsa_pool(
    device const half* IK [[buffer(0)]],
    device const half* ib [[buffer(1)]],
    device const int* blocks [[buffer(2)]],
    device const AttnTree* trees [[buffer(3)]],
    device const int4* desc [[buffer(4)]],
    device const int4* rinfo [[buffer(5)]],
    device const int4* pos [[buffer(6)]],
    device const float* kn_w [[buffer(7)]],
    device const float* inv_freq [[buffer(8)]],
    device const int* rope_sel [[buffer(9)]],
    device half* IP [[buffer(10)]],
    constant QsaParams& p [[buffer(11)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int row = int(tgpig.y);
  const int4 ri = rinfo[row];
  const int s = ri.x;
  if (((s + 1) & 3) != 0) return;
  const AttnTree T = trees[ri.w];
  const int4 d = desc[row];
  device const half* ibl = ib + p.ibase;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k = 0; k < 4; ++k) {
    const int i = s - 3 + k;
    device const half* src;
    if (i < T.P) {
      const int b = blocks[T.table + i / p.bt];
      src = ibl + ulong(b) * ulong(p.iblk) + ulong(i % p.bt) * QSA_HD;
    } else {
      src = IK + ulong(qsa_path_row(T.base, T.P, d, i)) * QSA_HD;
    }
    UNROLL for (int e = 0; e < 4; ++e) acc[e] += float(src[lane * 4 + e]);
  }
  float x[4];
  UNROLL for (int e = 0; e < 4; ++e) x[e] = bf16r(acc[e] * 0.25f);
  float ss = 0.0f;
  UNROLL for (int e = 0; e < 4; ++e) ss += x[e] * x[e];
  ss = simd_sum(ss);
  const float inv = rsqrt(ss / float(QSA_HD) + p.eps);
  UNROLL for (int e = 0; e < 4; ++e) x[e] = bf16r(x[e] * inv * kn_w[lane * 4 + e]);
  const int i0 = s - 3;
  int4 pr;
  if (i0 < T.P) {
    const int b = blocks[T.table + i0 / p.bt];
    pr = *((device const int4*)(ibl + ulong(b) * ulong(p.iblk) + ulong(p.pos_off) + ulong(i0 % p.bt) * 8));
  } else {
    pr = pos[qsa_path_row(T.base, T.P, d, i0)];
  }
  qsa_rope(x, pr, lane, inv_freq, rope_sel);
  UNROLL for (int e = 0; e < 4; ++e) IP[ulong(row) * QSA_HD + lane * 4 + e] = half(x[e]);
}

// Scores of the blocks entirely in the cache, b < P / 4: threadgroup = one
// tile (8 rows x 4 heads = 32 query vectors) x 32 blocks, 4 simdgroups of 8
// query vectors; fp16 operands (bf16 values: exact) with fp32 accumulation.
constant constexpr int QS_LD = QSA_HD + 8;

kernel void qsa_score_cached(
    device const half* IQ [[buffer(0)]],
    device const half* ib [[buffer(1)]],
    device const int* blocks [[buffer(2)]],
    device const AttnTree* trees [[buffer(3)]],
    device const int4* rinfo [[buffer(4)]],
    device const QsaTile* qtiles [[buffer(5)]],
    device float* scores [[buffer(6)]],
    constant QsaParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup half Qt[32 * QS_LD];
  threadgroup half Kt[32 * QS_LD];
  threadgroup float Dt[32 * 33];
  const QsaTile qt = qtiles[tgid.y];
  const AttnTree T = trees[qt.tree];
  const int ncb = min(T.P / 4, p.nbmax);
  const int b0 = int(tgid.x) * 32;
  if (b0 >= ncb) return;
  const int bpk = p.bt / 4;
  for (int c = int(tid); c < 32 * 16; c += 128) {
    const int qv = c / 16, col = (c % 16) * 8, r = qv / 4;
    uint4 v = uint4(0);
    if (r < qt.rows && rinfo[qt.r0 + r].z >= 0)
      v = *((device const uint4*)(IQ + (ulong(qt.r0 + r) * QSA_IH + ulong(qv % 4)) * QSA_HD + col));
    *((threadgroup uint4*)(Qt + qv * QS_LD + col)) = v;
  }
  for (int c = int(tid); c < 32 * 16; c += 128) {
    const int kb = c / 16, col = (c % 16) * 8, b = b0 + kb;
    uint4 v = uint4(0);
    if (b < ncb) {
      const int blk = blocks[T.table + b / bpk];
      v = *((device const uint4*)(ib + p.ibase + ulong(blk) * ulong(p.iblk) + ulong(p.pool_off) + ulong(b % bpk) * QSA_HD + col));
    }
    *((threadgroup uint4*)(Kt + kb * QS_LD + col)) = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_matrix<float, 8, 8> acc[4];
  UNROLL for (int n = 0; n < 4; ++n) acc[n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (int k = 0; k < QSA_HD / 8; ++k) {
    simdgroup_matrix<half, 8, 8> a;
    simdgroup_load(a, Qt + int(sg) * 8 * QS_LD + k * 8, QS_LD);
    UNROLL for (int n = 0; n < 4; ++n) {
      simdgroup_matrix<half, 8, 8> bm;
      simdgroup_load(bm, Kt + n * 8 * QS_LD + k * 8, QS_LD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(acc[n], a, bm, acc[n]);
    }
  }
  UNROLL for (int n = 0; n < 4; ++n) simdgroup_store(acc[n], Dt + int(sg) * 8 * 33 + n * 8, 33);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = int(tid); e < 256; e += 128) {
    const int r = e / 32, kb = e % 32, b = b0 + kb;
    if (r >= qt.rows || b >= ncb) continue;
    const int srow = rinfo[qt.r0 + r].z;
    if (srow < 0) continue;
    float sc = 0.0f;
    UNROLL for (int h = 0; h < QSA_IH; ++h) sc += max(Dt[(r * 4 + h) * 33 + kb], 0.0f);
    scores[ulong(srow) * ulong(p.nbmax) + ulong(b)] = sc;
  }
}

// Scores of the blocks ending in pass rows (P / 4 <= b < nb), one simdgroup
// per (row, block): the pooled key computed by the row at 4 b + 3 of its path.
kernel void qsa_score_batch(
    device const half* IQ [[buffer(0)]],
    device const half* IP [[buffer(1)]],
    device const AttnTree* trees [[buffer(2)]],
    device const int4* desc [[buffer(3)]],
    device const int4* rinfo [[buffer(4)]],
    device float* scores [[buffer(5)]],
    constant QsaParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int row = int(tgid.y);
  const int4 ri = rinfo[row];
  if (ri.z < 0) return;
  const AttnTree T = trees[ri.w];
  const int b = T.P / 4 + int(tgid.x);
  if (b >= ri.y) return;
  const int key_row = qsa_path_row(T.base, T.P, desc[row], 4 * b + 3);
  device const half* kp = IP + ulong(key_row) * QSA_HD + lane * 4;
  float k[4];
  UNROLL for (int e = 0; e < 4; ++e) k[e] = float(kp[e]);
  float sc = 0.0f;
  UNROLL for (int h = 0; h < QSA_IH; ++h) {
    device const half* qp = IQ + (ulong(row) * QSA_IH + h) * QSA_HD + lane * 4;
    float dt = 0.0f;
    UNROLL for (int e = 0; e < 4; ++e) dt += float(qp[e]) * k[e];
    sc += max(simd_sum(dt), 0.0f);
  }
  if (lane == 0) scores[ulong(ri.z) * ulong(p.nbmax) + ulong(b)] = sc;
}

// Top 512 of a sparse row's nb scores -> bitmap. One threadgroup (1024
// threads) per sparse row: four 8-bit radix passes find the 512th largest
// value v; every block above v is taken, then the first blocks equal to v.
kernel void qsa_select(
    device const float* scores [[buffer(0)]],
    device const int* srows [[buffer(1)]],
    device const int4* rinfo [[buffer(2)]],
    device uint* sel [[buffer(3)]],
    constant QsaParams& p [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NT = 1024;
  threadgroup atomic_uint hist[256];
  threadgroup uint sh[2];
  threadgroup uint part[32];
  const int sr = int(tgid);
  const int nb = clamp(rinfo[srows[sr]].y, 0, p.nbmax);  // defensive: a bad record cannot run away
  device const uint* u = (device const uint*)(scores + ulong(sr) * ulong(p.nbmax));
  uint prefix = 0, mask = 0, k = QSA_TOPK;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tid < 256) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int b = int(tid); b < nb; b += NT) {
      const uint v = u[b];
      if ((v & mask) == prefix) atomic_fetch_add_explicit(&hist[(v >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint cum = 0, bin = 0;
      for (int j = 255; j >= 0; --j) {
        const uint h = atomic_load_explicit(&hist[j], memory_order_relaxed);
        if (cum + h >= k) {
          bin = uint(j);
          break;
        }
        cum += h;
      }
      sh[0] = prefix | (bin << shift);
      sh[1] = k - cum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix = sh[0];
    k = sh[1];
    mask |= 255u << shift;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int nw = (nb + 31) / 32;
  uint running = 0;
  for (int w0 = 0; w0 < p.nbw; w0 += NT) {
    const int w = w0 + int(tid);
    uint gt = 0, eq = 0;
    if (w < nw) {
      for (int bit = 0; bit < 32; ++bit) {
        const int b = w * 32 + bit;
        if (b >= nb) break;
        const uint v = u[b];
        gt |= uint(v > prefix) << bit;
        eq |= uint(v == prefix) << bit;
      }
    }
    const uint c = popcount(eq);
    const uint incl = simd_prefix_inclusive_sum(c);
    if (lane == 31) part[sg] = incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      const uint t = part[lane];
      const uint ex = simd_prefix_exclusive_sum(t);
      part[lane] = ex;
      if (lane == 31) sh[0] = ex + t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint before = running + part[sg] + incl - c;
    uint take = before >= k ? 0u : min(c, k - before);
    uint word = gt;
    while (take > 0) {
      const uint lo = eq & (~eq + 1u);
      word |= lo;
      eq ^= lo;
      --take;
    }
    if (w < p.nbw) sel[ulong(sr) * ulong(p.nbw) + ulong(w)] = word;
    running += sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Per tile: the blocks holding cached tokens (b < ceil(P / 4)) that any of
// its rows reads, ascending, as (block << 8) | row mask. 256 threads, one
// bitmap word (32 blocks) each per round.
kernel void qsa_tiles(
    device const uint* sel [[buffer(0)]],
    device const AttnTree* trees [[buffer(1)]],
    device const int4* rinfo [[buffer(2)]],
    device const QsaTile* qtiles [[buffer(3)]],
    device uint* tlist [[buffer(4)]],
    device uint* tcount [[buffer(5)]],
    constant QsaParams& p [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NT = 256;
  threadgroup uint part[8];
  threadgroup uint total;
  const QsaTile qt = qtiles[tgid];
  const AttnTree T = trees[qt.tree];
  const int nbc = clamp((T.P + 3) / 4, 0, p.nbmax + 1);
  int srow[8], nbr[8];
  UNROLL for (int k = 0; k < 8; ++k) {
    srow[k] = -2;  // no row
    nbr[k] = 0;
    if (k < qt.rows) {
      const int4 ri = rinfo[qt.r0 + k];
      srow[k] = ri.z;
      nbr[k] = ri.y;
    }
  }
  const int nw = (nbc + 31) / 32;
  uint running = 0;
  device uint* out = tlist + ulong(tgid) * QSA_LMAX;
  for (int w0 = 0; w0 < nw; w0 += NT) {
    const int w = w0 + int(tid);
    uint rw[8];
    UNROLL for (int k = 0; k < 8; ++k) {
      uint x = 0;
      if (w < nw) {
        if (srow[k] == -1) {
          x = 0xFFFFFFFFu;  // a row still dense reads every cached block
        } else if (srow[k] >= 0) {
          x = w < p.nbw ? sel[ulong(srow[k]) * ulong(p.nbw) + ulong(w)] : 0u;
          if (nbr[k] / 32 == w) x |= 1u << (nbr[k] % 32);  // the tail's block
        }
      }
      rw[k] = x;
    }
    const int lim = nbc - w * 32;
    const uint vmask = w < nw ? (lim >= 32 ? 0xFFFFFFFFu : ((1u << uint(max(lim, 0))) - 1u)) : 0u;
    uint any = 0;
    UNROLL for (int k = 0; k < 8; ++k) any |= rw[k];
    any &= vmask;
    const uint c = popcount(any);
    const uint incl = simd_prefix_inclusive_sum(c);
    if (lane == 31) part[sg] = incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint a = 0;
      for (int g = 0; g < NT / 32; ++g) {
        const uint t = part[g];
        part[g] = a;
        a += t;
      }
      total = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at = running + part[sg] + incl - c;
    uint a = any;
    while (a) {
      const int bit = int(ctz(a));
      a &= a - 1u;
      uint m8 = 0;
      UNROLL for (int k = 0; k < 8; ++k) m8 |= ((rw[k] >> uint(bit)) & 1u) << uint(k);
      if (at < uint(QSA_LMAX)) out[at] = (uint(w * 32 + bit) << 8) | m8;
      ++at;
    }
    running += total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) tcount[tgid] = min(running, uint(QSA_LMAX));
}

// Flash attention with the QSA selection (flash_attention's structure). Tiles
// with pad0 = 1 read the cached keys of their union list (entries [k0, k1)
// of tile pad1's list, 4 blocks = 16 keys per step; a key counts for the rows
// in its entry's mask); their batch keys count for a sparse row when their
// block is selected or in its tail. Tiles with pad0 = 0 are dense.
template <int G, int SPLIT = 1>
kernel void flash_attention_qsa(
    device const half* Q [[buffer(0)]],
    device const half* Kb [[buffer(1)]],
    device const half* Vb [[buffer(2)]],
    device const half* kv [[buffer(3)]],
    device const int* blocks [[buffer(4)]],
    device const int4* desc [[buffer(5)]],
    device const float* X [[buffer(6)]],
    device half* O [[buffer(7)]],
    constant AttnParams& p [[buffer(8)]],
    device const AttnTile* tiles [[buffer(9)]],
    device const AttnTree* trees [[buffer(10)]],
    device float* parts [[buffer(11)]],
    device const uint* tlist [[buffer(12)]],
    device const uint* tcount [[buffer(13)]],
    device const int4* rinfo [[buffer(14)]],
    device const uint* sel [[buffer(15)]],
    constant QsaParams& qp [[buffer(16)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NT = 64 * G;
  threadgroup half Kt[FA_BK * FA_LD];
  threadgroup half Vt[FA_BK * FA_LD];
  threadgroup float xs[2 * G][8][FA_BK];
  threadgroup uint ents[4];

  const int kvh = tgid.x;
  const int kvr = kvh / SPLIT;
  const AttnTile tile = tiles[tgid.y];
  const AttnTree tree = trees[tile.tree];
  const bool sparse = tile.pad0 != 0;
  const int hg = sg / 2;
  const int dh = sg % 2;
  const int h = kvh * G + hg;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  const int end = tree.base + tree.M;
  const int row = tile.r0 + fm;
  const bool row_ok = row < end;
  const int rr = row - tree.base;
  const int4 d = desc[min(row, end - 1)];
  const int4 ri = rinfo[min(row, end - 1)];
  const bool row_sparse = sparse && ri.z >= 0;
  const int r_last = min(tile.r0 + 7, end - 1) - tree.base;

  simdgroup_matrix<half, 8, 8> qf[16];
  {
    device const half* qptr = Q + ulong(min(row, end - 1)) * p.q_ld + h * 256 + dh * 128 + fn;
    UNROLL for (int ks = 0; ks < 16; ++ks) {
      const half2 v = *((device const half2*)(qptr + ks * 8)) * half(0.0625h);
      qf[ks].thread_elements()[0] = v.x;
      qf[ks].thread_elements()[1] = v.y;
    }
  }
  simdgroup_matrix<float, 8, 8> of[16];
  UNROLL for (int i = 0; i < 16; ++i) of[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  float m_run = -INFINITY;
  float l_run = 0.0f;

  int anc_max = 0, seg_min = 1 << 30;
  for (int r = tile.r0; r <= tree.base + r_last; ++r) {
    const int4 dr = desc[r];
    anc_max = max(anc_max, dr.z);
    seg_min = min(seg_min, dr.x);
  }

  const ulong blk_halves = ulong(p.bt) * p.kv_ld;
  device const half* kvl = kv + p.kv_base;
  int e_end = 0;
  int n_slot_blocks;
  if (sparse) {
    e_end = min(min(tile.k1, int(min(tcount[tile.pad1], uint(QSA_LMAX)))), QSA_LMAX);
    n_slot_blocks = max(0, (e_end - tile.k0 + 3) / 4);
  } else {
    n_slot_blocks = (tile.k1 - tile.k0 + FA_BK - 1) / FA_BK;
  }
  device const uint* tl = tlist + ulong(tile.pad1) * QSA_LMAX;
  const int n_batch_blocks = tile.last ? (r_last + 1 + FA_BK - 1) / FA_BK : 0;
  for (int blk = 0; blk < n_slot_blocks + n_batch_blocks; ++blk) {
    const bool in_slot = blk < n_slot_blocks;
    const int j0 = in_slot ? tile.k0 + blk * FA_BK : (blk - n_slot_blocks) * FA_BK;
    const int j_end = in_slot ? tile.k1 : r_last + 1;
    if (!in_slot) {
      const bool anc_vis = j0 < anc_max;
      const bool own_vis = (j0 + FA_BK - 1 >= seg_min) && (j0 <= r_last);
      if (!anc_vis && !own_vis) continue;
    }
    const bool qsa_slot = sparse && in_slot;
    if (qsa_slot) {
      if (tid < 4) {
        const int e = tile.k0 + blk * 4 + int(tid);
        ents[tid] = e < e_end ? tl[e] : QSA_NONE;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int c = tid; c < FA_BK * 32 * 2; c += NT) {
      const int which = c / (FA_BK * 32);
      const int cc = c % (FA_BK * 32);
      const int key = cc / 32;
      const int col = (cc % 32) * 8;
      uint4 v = uint4(0);
      if (qsa_slot) {
        const uint ent = ents[key / 4];
        const int j = int(ent >> 8) * 4 + key % 4;
        if (ent != QSA_NONE && j < tree.P) {
          const ulong b = ulong(blocks[tree.table + j / p.bt]);
          v = *((device const uint4*)(kvl + (b * 2 + which) * blk_halves + ulong(j % p.bt) * p.kv_ld + kvr * 256 + col));
        }
      } else {
        const int j = j0 + key;
        if (j < j_end) {
          device const half* src;
          if (in_slot) {
            const ulong b = ulong(blocks[tree.table + j / p.bt]);
            src = kvl + (b * 2 + which) * blk_halves + ulong(j % p.bt) * p.kv_ld + kvr * 256 + col;
          } else {
            src = (which ? Vb : Kb) + ulong(tree.base + j) * p.kv_ld + kvr * 256 + col;
          }
          v = *((device const uint4*)src);
        }
      }
      *((threadgroup uint4*)((which ? Vt : Kt) + key * FA_LD + col)) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> sf[2];
    sf[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    sf[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    UNROLL for (int ks = 0; ks < 16; ++ks) {
      UNROLL for (int kf = 0; kf < 2; ++kf) {
        simdgroup_matrix<half, 8, 8> kb;
        const threadgroup half* src = Kt + (kf * 8 + fn) * FA_LD + dh * 128 + ks * 8 + fm;
        kb.thread_elements()[0] = src[0];
        kb.thread_elements()[1] = src[FA_LD];
        simdgroup_multiply_accumulate(sf[kf], qf[ks], kb, sf[kf]);
      }
    }
    UNROLL for (int kf = 0; kf < 2; ++kf) {
      xs[sg][fm][kf * 8 + fn] = sf[kf].thread_elements()[0];
      xs[sg][fm][kf * 8 + fn + 1] = sf[kf].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sv[4];
    UNROLL for (int kf = 0; kf < 2; ++kf) {
      UNROLL for (int e = 0; e < 2; ++e) {
        const int kc = kf * 8 + fn + e;
        float v = sf[kf].thread_elements()[e] + xs[sg ^ 1][fm][kc];
        bool vis;
        if (qsa_slot) {
          const uint ent = ents[kc / 4];
          const int j = int(ent >> 8) * 4 + kc % 4;
          vis = row_ok && ent != QSA_NONE && j < tree.P && ((ent >> uint(fm)) & 1u);
        } else {
          const int j = j0 + kc;
          vis = row_ok && j < j_end;
          if (!in_slot) {
            vis = vis && ((j < d.z) || (j >= d.x && j <= rr));
            if (vis && row_sparse) {
              const int sj = tree.P + (j < d.z ? j : d.z + (j - d.x));
              const int bj = sj >> 2;
              vis = sj >= 4 * ri.y || ((sel[ulong(ri.z) * ulong(qp.nbw) + ulong(bj >> 5)] >> uint(bj & 31)) & 1u);
            }
          }
        }
        sv[kf * 2 + e] = vis ? v : -INFINITY;
      }
    }
    float mx = max(max(sv[0], sv[1]), max(sv[2], sv[3]));
    mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
    const float m_new = max(m_run, mx);
    const float corr = (m_new == -INFINITY) ? 1.0f : exp(m_run - m_new);
    float ps = 0.0f;
    simdgroup_matrix<half, 8, 8> pf[2];
    UNROLL for (int kf = 0; kf < 2; ++kf) {
      UNROLL for (int e = 0; e < 2; ++e) {
        const float pv = (m_new == -INFINITY) ? 0.0f : exp(sv[kf * 2 + e] - m_new);
        ps += pv;
        pf[kf].thread_elements()[e] = half(pv);
      }
    }
    ps += simd_shuffle_xor(ps, ushort(1));
    ps += simd_shuffle_xor(ps, ushort(8));
    l_run = l_run * corr + ps;
    m_run = m_new;
    UNROLL for (int df = 0; df < 16; ++df) {
      of[df].thread_elements()[0] *= corr;
      of[df].thread_elements()[1] *= corr;
    }
    UNROLL for (int kf = 0; kf < 2; ++kf) {
      UNROLL for (int df = 0; df < 16; ++df) {
        simdgroup_matrix<half, 8, 8> vb;
        const half2 v2 = *((const threadgroup half2*)(Vt + (kf * 8 + fm) * FA_LD + dh * 128 + df * 8 + fn));
        vb.thread_elements()[0] = v2.x;
        vb.thread_elements()[1] = v2.y;
        simdgroup_multiply_accumulate(of[df], pf[kf], vb, of[df]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!row_ok) return;
  if (tile.part >= 0) {
    device float* rec = parts + ((ulong(tile.part) * p.Hkv + kvh) * 8 * G + ulong(fm) * G + hg) * 258;
    UNROLL for (int df = 0; df < 16; ++df) {
      rec[dh * 128 + df * 8 + fn] = of[df].thread_elements()[0];
      rec[dh * 128 + df * 8 + fn + 1] = of[df].thread_elements()[1];
    }
    if (dh == 0 && (lane & 9) == 0) {
      rec[256] = m_run;
      rec[257] = l_run;
    }
    return;
  }
  const float inv_l = l_run > 0.0f ? 1.0f / l_run : 0.0f;
  device const float* gp = X + ulong(row) * p.g_ld + h * 512 + 256 + dh * 128 + fn;
  device half* op = O + ulong(row) * p.q_ld + h * 256 + dh * 128 + fn;
  UNROLL for (int df = 0; df < 16; ++df) {
    const float2 g = *((device const float2*)(gp + df * 8));
    const float o0 = of[df].thread_elements()[0] * inv_l / (1.0f + exp(-g.x));
    const float o1 = of[df].thread_elements()[1] * inv_l / (1.0f + exp(-g.y));
    *((device half2*)(op + df * 8)) = half2(half(o0), half(o1));
  }
}

template [[host_name("flash_attention_qsa_g6s2")]] kernel void flash_attention_qsa<6, 2>(
    device const half*, device const half*, device const half*, device const half*, device const int*,
    device const int4*, device const float*, device half*, constant AttnParams&, device const AttnTile*,
    device const AttnTree*, device float*, device const uint*, device const uint*, device const int4*,
    device const uint*, constant QsaParams&, uint2, uint, uint, uint);

// Raw keys, pooled keys (rows completing a block) and positions of the saved
// rows into the index pool. Grid (16, entries): 8 halves per thread.
kernel void qsa_save(
    device const half* IK [[buffer(0)]],
    device const half* IP [[buffer(1)]],
    device const int4* pos [[buffer(2)]],
    device half* ib [[buffer(3)]],
    device const KVSave* entries [[buffer(4)]],
    constant QsaParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= 16) return;
  const KVSave e = entries[gid.y];
  device half* blk = ib + p.ibase + ulong(e.block) * ulong(p.iblk);
  const int c = int(gid.x) * 8;
  *((device uint4*)(blk + ulong(e.pos) * QSA_HD + c)) = *((device const uint4*)(IK + ulong(e.src_row) * QSA_HD + c));
  if ((e.pos & 3) == 3)
    *((device uint4*)(blk + ulong(p.pool_off) + ulong(e.pos / 4) * QSA_HD + c)) =
        *((device const uint4*)(IP + ulong(e.src_row) * QSA_HD + c));
  if (gid.x == 0) *((device int4*)(blk + ulong(p.pos_off) + ulong(e.pos) * 8)) = pos[e.src_row];
}
