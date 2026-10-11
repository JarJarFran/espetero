// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ---------------------------------------------------------------------------
// Gemma 4 (src/model/impl/gemma.h): attention-only decoder layers with
// sliding-window (head dim 256, the last `window` positions) and global
// (head dim 512, proportional RoPE, K = V) attention, per-head q / k / v
// RMSNorms and attention scale 1, four RMSNorms per layer, a layer scalar,
// GeGLU MLPs (the GEMM epilogues' GELU gate), a mixture-of-experts block
// beside the dense MLP (26B-A4B) and tanh soft-capped logits. Semantics as in
// mlx-lm's models/gemma4_text.py (MIT).
// ---------------------------------------------------------------------------

// X[i] *= s over n floats (n % 4 == 0): the embedding scale sqrt(D).
kernel void g4_scale_rows(
    device float4* X [[buffer(0)]],
    constant float& s [[buffer(1)]],
    constant uint& n4 [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < n4) X[gid] *= s;
}

// Logit soft-capping in place: x = cap * tanh(x / cap) over n4 float4s.
kernel void softcap_rows(
    device float4* X [[buffer(0)]],
    constant float& cap [[buffer(1)]],
    constant uint& n4 [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < n4) X[gid] = cap * precise::tanh(X[gid] / cap);
}

// Sum of `v` over the threadgroup, returned to every thread (fixed order:
// per-simdgroup sums, then the simdgroups in index order). `part` holds 32
// floats; the trailing barrier lets the next call reuse it.
static inline float g4_tg_sum(float v, threadgroup float* part, uint sg, uint lane, uint ntg) {
  v = simd_sum(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = lane < (ntg + 31) / 32 ? part[lane] : 0.0f;
  t = simd_sum(t);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

// ---------------------------------------------------------------------------
// Attention inputs: per-head RMSNorm of q (weight q_norm) and k (k_norm) with
// RoPE over (d, d + HD/2) pairs at inv_freq[d] (0: not rotated), and the
// weightless RMSNorm of v (from the v projection, or the keys' raw rows when
// K = V), as fp16 Q / K / V batch rows. One simdgroup per (row, head): heads
// [0, H) queries, [H, H + Hkv) keys, [H + Hkv, H + 2 Hkv) values.
// ---------------------------------------------------------------------------
struct G4PrepParams {
  int H;
  int Hkv;
  int ld;     // projection row stride (floats)
  int k_off;  // column of the keys
  int v_off;  // column of the values' source rows
  int q_ld;   // Q row stride (H * HD)
  int kv_ld;  // K / V row stride (Hkv * HD)
  float eps;
};

template <int HD>
kernel void g4_attn_prep(
    device const float* X [[buffer(0)]],
    device const float* qn_w [[buffer(1)]],
    device const float* kn_w [[buffer(2)]],
    device const int4* pos [[buffer(3)]],
    device const float* inv_freq [[buffer(4)]],
    device half* Q [[buffer(5)]],
    device half* Kb [[buffer(6)]],
    device half* Vb [[buffer(7)]],
    constant G4PrepParams& p [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NPT = HD / 32;  // consecutive elements per lane: lanes 0-15 hold the first half
  const int head = int(tgpig.x) * 4 + int(sg);
  const ulong row = tgpig.y;
  if (head >= p.H + 2 * p.Hkv) return;
  const int kind = head < p.H ? 0 : (head < p.H + p.Hkv ? 1 : 2);  // q, k, v
  const int h = kind == 0 ? head : (kind == 1 ? head - p.H : head - p.H - p.Hkv);
  device const float* src = X + row * p.ld + (kind == 0 ? 0 : (kind == 1 ? p.k_off : p.v_off)) + h * HD + lane * NPT;
  float x[NPT];
  float ss = 0.0f;
  UNROLL for (int i = 0; i < NPT; ++i) {
    x[i] = src[i];
    ss += x[i] * x[i];
  }
  ss = simd_sum(ss);
  const float inv = rsqrt(ss / float(HD) + p.eps);
  if (kind == 2) {
    device half* vo = Vb + row * p.kv_ld + h * HD + lane * NPT;
    UNROLL for (int i = 0; i < NPT; ++i) vo[i] = half(x[i] * inv);
    return;
  }
  device const float* nw = (kind == 0 ? qn_w : kn_w) + lane * NPT;
  UNROLL for (int i = 0; i < NPT; ++i) x[i] = x[i] * inv * nw[i];
  const float position = float(pos[row].x);
  UNROLL for (int i = 0; i < NPT; ++i) {
    const float partner = simd_shuffle_xor(x[i], ushort(16));
    const float ang = position * inv_freq[(lane & 15) * NPT + i];
    const float c = precise::cos(ang), s = precise::sin(ang);
    x[i] = lane < 16 ? x[i] * c - partner * s : x[i] * c + partner * s;
  }
  device half* o = (kind == 0 ? Q + row * p.q_ld : Kb + row * p.kv_ld) + h * HD + lane * NPT;
  UNROLL for (int i = 0; i < NPT; ++i) o[i] = half(x[i]);
}

template [[host_name("g4_attn_prep256")]] kernel void g4_attn_prep<256>(
    device const float*, device const float*, device const float*, device const int4*, device const float*,
    device half*, device half*, device half*, constant G4PrepParams&, uint2, uint, uint);
template [[host_name("g4_attn_prep512")]] kernel void g4_attn_prep<512>(
    device const float*, device const float*, device const float*, device const int4*, device const float*,
    device half*, device half*, device half*, constant G4PrepParams&, uint2, uint, uint);

// ---------------------------------------------------------------------------
// Forest attention of a Gemma 4 layer (flash_attention's structure, 40_attention):
// a threadgroup is one group of G query heads of a KV head (SPLIT groups per
// KV head) x one tile of 8 consecutive query rows of one tree. HD / 128
// simdgroups per head each own 128 dims; score partials are summed through
// threadgroup memory in a fixed order. Scale 1 (q and k are normalized), no
// output gate. Sliding layers (window > 0) mask keys more than window - 1
// positions behind the query, and read cached keys from the parent slot's
// ring (p.ring blocks from tree.table: key j in block (j / bt) % ring);
// global layers read the paged block list. Cached keys are at positions
// 0..P-1 (Gemma has no M-RoPE grids); batch keys at their rows' positions.
// ---------------------------------------------------------------------------
struct G4AttnParams {
  int H;      // query heads
  int Hkv;    // KV head groups of the grid (KV heads x SPLIT)
  int q_ld;   // Q / O row stride (H * HD)
  int kv_ld;  // K / V row width (KV heads * HD), batch and pool
  int bt;     // tokens per KV block
  int window; // sliding window (0: none)
  int ring;   // blocks per slot ring (sliding layers), 0: block lists
  int pad;
  ulong kv_base;  // half offset of this layer's region
};

struct G4Tile {
  int tree;
  int r0;    // absolute first query row
  int k0;    // cached-key range [k0, k1)
  int k1;
  int part;  // -1: write O; else the partial record index
  int last;  // also scan the batch keys
  int b0;    // first batch key (relative row) any row of the tile may see
  int pad;
};

template <int HD, int G, int SPLIT>
kernel void g4_attention(
    device const half* Q [[buffer(0)]],
    device const half* Kb [[buffer(1)]],
    device const half* Vb [[buffer(2)]],
    device const half* kv [[buffer(3)]],
    device const int* blocks [[buffer(4)]],
    device const int4* desc [[buffer(5)]],
    device const int4* pos [[buffer(6)]],
    device half* O [[buffer(7)]],
    constant G4AttnParams& p [[buffer(8)]],
    device const G4Tile* tiles [[buffer(9)]],
    device const AttnTree* trees [[buffer(10)]],
    device float* parts [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NSG = HD / 128;           // simdgroups per query head
  constexpr int NT = 32 * NSG * G;
  constexpr int BK = HD == 256 ? 16 : 8;  // keys per block: 2 x BK x (HD + 8) halves of threadgroup memory
  constexpr int KF = BK / 8;
  constexpr int LD = HD + 8;
  constexpr int C8 = HD / 8;              // uint4 columns of a key row
  threadgroup half Kt[BK * LD];
  threadgroup half Vt[BK * LD];
  threadgroup float xs[NSG * G][8][BK];

  const int kvg = int(tgid.x);
  const int kvr = kvg / SPLIT;  // cached KV head
  const G4Tile tile = tiles[tgid.y];
  const AttnTree tree = trees[tile.tree];
  const int hg = int(sg) / NSG;
  const int dh = int(sg) % NSG;
  const int h = kvg * G + hg;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  const int end = tree.base + tree.M;
  const int row = tile.r0 + fm;
  const bool row_ok = row < end;
  const int rr = row - tree.base;
  const int4 d = desc[min(row, end - 1)];
  const int qpos = pos[min(row, end - 1)].x;
  const int r_last = min(tile.r0 + 7, end - 1) - tree.base;

  simdgroup_matrix<half, 8, 8> qf[16];
  {
    device const half* qp = Q + ulong(min(row, end - 1)) * p.q_ld + h * HD + dh * 128 + fn;
    UNROLL for (int ks = 0; ks < 16; ++ks) {
      const half2 v = *((device const half2*)(qp + ks * 8));
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
  const int n_slot_blocks = (tile.k1 - tile.k0 + BK - 1) / BK;
  const int bb0 = tile.last ? tile.b0 / BK : 0;
  const int n_batch_blocks = tile.last ? max(0, (r_last + BK) / BK - bb0) : 0;
  for (int blk = 0; blk < n_slot_blocks + n_batch_blocks; ++blk) {
    const bool in_slot = blk < n_slot_blocks;
    const int j0 = in_slot ? tile.k0 + blk * BK : (bb0 + blk - n_slot_blocks) * BK;
    const int j_end = in_slot ? tile.k1 : r_last + 1;
    if (!in_slot) {
      const bool anc_vis = j0 < anc_max;
      const bool own_vis = (j0 + BK - 1 >= seg_min) && (j0 <= r_last);
      if (!anc_vis && !own_vis) continue;
    }
    for (int c = int(tid); c < BK * C8 * 2; c += NT) {
      const int which = c / (BK * C8);
      const int cc = c % (BK * C8);
      const int key = cc / C8;
      const int col = (cc % C8) * 8;
      const int j = j0 + key;
      uint4 v = uint4(0);
      if (j < j_end) {
        device const half* src;
        if (in_slot) {
          const ulong b = p.ring ? ulong(tree.table + (j / p.bt) % p.ring) : ulong(blocks[tree.table + j / p.bt]);
          src = kvl + (b * 2 + ulong(which)) * blk_halves + ulong(j % p.bt) * p.kv_ld + kvr * HD + col;
        } else {
          src = (which ? Vb : Kb) + ulong(tree.base + j) * p.kv_ld + kvr * HD + col;
        }
        v = *((device const uint4*)src);
      }
      *((threadgroup uint4*)((which ? Vt : Kt) + key * LD + col)) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> sf[KF];
    UNROLL for (int kf = 0; kf < KF; ++kf) sf[kf] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    UNROLL for (int ks = 0; ks < 16; ++ks) {
      UNROLL for (int kf = 0; kf < KF; ++kf) {
        simdgroup_matrix<half, 8, 8> kb;
        const threadgroup half* src = Kt + (kf * 8 + fn) * LD + dh * 128 + ks * 8 + fm;
        kb.thread_elements()[0] = src[0];
        kb.thread_elements()[1] = src[LD];
        simdgroup_multiply_accumulate(sf[kf], qf[ks], kb, sf[kf]);
      }
    }
    UNROLL for (int kf = 0; kf < KF; ++kf) {
      xs[sg][fm][kf * 8 + fn] = sf[kf].thread_elements()[0];
      xs[sg][fm][kf * 8 + fn + 1] = sf[kf].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sv[2 * KF];
    UNROLL for (int kf = 0; kf < KF; ++kf) {
      UNROLL for (int e = 0; e < 2; ++e) {
        const int kc = kf * 8 + fn + e;
        float v = 0.0f;
        UNROLL for (int o = 0; o < NSG; ++o) v += xs[hg * NSG + o][fm][kc];  // same order in every simdgroup
        const int j = j0 + kc;
        bool vis = row_ok && j < j_end;
        if (!in_slot) vis = vis && ((j < d.z) || (j >= d.x && j <= rr));
        if (p.window > 0 && vis) vis = qpos - (in_slot ? j : pos[tree.base + j].x) < p.window;
        sv[kf * 2 + e] = vis ? v : -INFINITY;
      }
    }
    float mx = sv[0];
    UNROLL for (int i = 1; i < 2 * KF; ++i) mx = max(mx, sv[i]);
    mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
    const float m_new = max(m_run, mx);
    const float corr = (m_new == -INFINITY) ? 1.0f : exp(m_run - m_new);
    float ps = 0.0f;
    simdgroup_matrix<half, 8, 8> pf[KF];
    UNROLL for (int kf = 0; kf < KF; ++kf) {
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
    UNROLL for (int kf = 0; kf < KF; ++kf) {
      UNROLL for (int df = 0; df < 16; ++df) {
        simdgroup_matrix<half, 8, 8> vb;
        const half2 v2 = *((const threadgroup half2*)(Vt + (kf * 8 + fm) * LD + dh * 128 + df * 8 + fn));
        vb.thread_elements()[0] = v2.x;
        vb.thread_elements()[1] = v2.y;
        simdgroup_multiply_accumulate(of[df], pf[kf], vb, of[df]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!row_ok) return;
  if (tile.part >= 0) {
    // Partial record [8 rows][G heads]: HD O floats (unnormalized), then m and l.
    device float* rec = parts + ((ulong(tile.part) * p.Hkv + kvg) * 8 * G + ulong(fm) * G + hg) * (HD + 2);
    UNROLL for (int df = 0; df < 16; ++df) {
      rec[dh * 128 + df * 8 + fn] = of[df].thread_elements()[0];
      rec[dh * 128 + df * 8 + fn + 1] = of[df].thread_elements()[1];
    }
    if (dh == 0 && (lane & 9) == 0) {
      rec[HD] = m_run;
      rec[HD + 1] = l_run;
    }
    return;
  }
  const float inv_l = l_run > 0.0f ? 1.0f / l_run : 0.0f;
  device half* op = O + ulong(row) * p.q_ld + h * HD + dh * 128 + fn;
  UNROLL for (int df = 0; df < 16; ++df)
    *((device half2*)(op + df * 8)) =
        half2(half(of[df].thread_elements()[0] * inv_l), half(of[df].thread_elements()[1] * inv_l));
}

#define INSTANTIATE_G4A(NAME, HD, G, SPLIT)                                                            \
  template [[host_name(NAME)]] kernel void g4_attention<HD, G, SPLIT>(                                 \
      device const half*, device const half*, device const half*, device const half*, device const int*, \
      device const int4*, device const int4*, device half*, constant G4AttnParams&, device const G4Tile*, \
      device const AttnTree*, device float*, uint2, uint, uint, uint);
// Sliding layers: all G query heads of a KV head in one threadgroup (32 x 2 x G threads).
INSTANTIATE_G4A("g4_attention256_g1", 256, 1, 1)
INSTANTIATE_G4A("g4_attention256_g2", 256, 2, 1)
INSTANTIATE_G4A("g4_attention256_g4", 256, 4, 1)
INSTANTIATE_G4A("g4_attention256_g8", 256, 8, 1)
// Global layers: 2 query heads per threadgroup (256 threads), SPLIT threadgroups per KV head.
INSTANTIATE_G4A("g4_attention512_s1", 512, 2, 1)
INSTANTIATE_G4A("g4_attention512_s2", 512, 2, 2)
INSTANTIATE_G4A("g4_attention512_s4", 512, 2, 4)
INSTANTIATE_G4A("g4_attention512_s8", 512, 2, 8)

// Decode attention of a Gemma 4 layer for passes whose trees are single rows:
// attn_decode_gqa (40_attention) with HD / 128 simdgroups per key slice
// (their partial scores summed in a fixed order), scale 1, no output gate,
// and the sliding layers' ring of blocks. The G <= 8 query heads of a KV head
// are the rows of the MMA tiles (g4_attention's 8 query rows are 7 of 8
// padding at one row, and its global layers run 2 heads per threadgroup).
// Single-row tiles need no window mask: the planner starts a sliding tile at
// its row's first visible key. Writes g4_attention's partial record (one
// threadgroup per KV head: p.Hkv = KV heads) or the output.
template <int HD, int G>
kernel void g4_attention_dec(
    device const half* Q [[buffer(0)]],
    device const half* Kb [[buffer(1)]],
    device const half* Vb [[buffer(2)]],
    device const half* kv [[buffer(3)]],
    device const int* blocks [[buffer(4)]],
    device const int4* desc [[buffer(5)]],
    device const int4* pos [[buffer(6)]],
    device half* O [[buffer(7)]],
    constant G4AttnParams& p [[buffer(8)]],
    device const G4Tile* tiles [[buffer(9)]],
    device const AttnTree* trees [[buffer(10)]],
    device float* parts [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NQ = HD / 128;  // simdgroups per key slice (128 dims each)
  constexpr int KP = 2;         // key slices
  constexpr int R = HD + 2;     // record floats
  threadgroup float mrg[KP - 1][G][R];
  threadgroup float2 xs[2][KP][NQ][32];  // [buffer][slice][dim quarter][lane]: partial scores
  const int kvh = int(tgid.x);
  const G4Tile tile = tiles[tgid.y];
  const AttnTree tree = trees[tile.tree];
  const int row = tile.r0;  // the tree's only row
  const int dq = int(sg) % NQ, pr = int(sg) / NQ;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);    // tile row: query head
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;  // tile columns fn, fn + 1
  const int hq = min(int(fm), G - 1);

  // Q fragments of this simdgroup's dims: S blocks 8t + w, t in {2 dq, 2 dq + 1} (dims 64t + 8c + w).
  simdgroup_matrix<half, 8, 8> qf[16];
  {
    device const half* qp = Q + ulong(row) * p.q_ld + (kvh * G + hq) * HD + 8 * fn;
    UNROLL for (int tt = 0; tt < 2; ++tt) {
      const int t = 2 * dq + tt;
      const uint4 a = *((device const uint4*)(qp + 64 * t));
      const uint4 b = *((device const uint4*)(qp + 64 * t + 8));
      UNROLL for (int w = 0; w < 8; ++w) {
        qf[8 * tt + w].thread_elements()[0] = as_type<half2>(a[w / 2])[w % 2];
        qf[8 * tt + w].thread_elements()[1] = as_type<half2>(b[w / 2])[w % 2];
      }
    }
  }
  simdgroup_matrix<float, 8, 8> of[16];  // O blocks 4u + v (dims 32u + 8(c / 2) + 2v + c % 2), u = 4 dq + uu
  UNROLL for (int i = 0; i < 16; ++i) of[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  float m_run = -INFINITY, l_run = 0.0f;

  const ulong blk_halves = ulong(p.bt) * p.kv_ld;
  device const half* kvl = kv + p.kv_base + kvh * HD;
  const int n_slot = tile.k1 - tile.k0;
  const int n_keys = n_slot + (tile.last ? 1 : 0);  // + the row's own key
  // Key (which 0) or value (1) row of tile key j < n_keys.
  auto key_row = [&](int j, int which) -> device const half* {
    if (j < n_slot) {
      const int jj = tile.k0 + j;
      const ulong b = p.ring ? ulong(tree.table + (jj / p.bt) % p.ring) : ulong(blocks[tree.table + jj / p.bt]);
      return kvl + (b * 2 + which) * blk_halves + ulong(jj % p.bt) * p.kv_ld;
    }
    return (which ? Vb : Kb) + ulong(tree.base) * p.kv_ld + kvh * HD;
  };
  const int n_iter = ((n_keys + 7) / 8 + KP - 1) / KP;  // the same in every simdgroup (barriers)
  for (int it = 0; it < n_iter; ++it) {
    const int j0 = 8 * (it * KP + pr);
    const bool live = j0 < n_keys;
    const int ja = j0 + fn, jb = j0 + fn + 1, jv = j0 + fm;
    uint4 va[4];
    if (live) {
      device const half* pa = key_row(min(ja, n_keys - 1), 0) + 8 * fm;
      device const half* pb = key_row(min(jb, n_keys - 1), 0) + 8 * fm;
      uint4 ka[2], kb[2];
      UNROLL for (int tt = 0; tt < 2; ++tt) {
        ka[tt] = *((device const uint4*)(pa + 64 * (2 * dq + tt)));
        kb[tt] = *((device const uint4*)(pb + 64 * (2 * dq + tt)));
      }
      device const half* vp = key_row(min(jv, n_keys - 1), 1) + 4 * fn;
      const bool v_ok = jv < n_keys;
      UNROLL for (int uu = 0; uu < 4; ++uu) va[uu] = v_ok ? *((device const uint4*)(vp + 32 * (4 * dq + uu))) : uint4(0);
      simdgroup_matrix<float, 8, 8> sf = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      UNROLL for (int tt = 0; tt < 2; ++tt) {
        UNROLL for (int w = 0; w < 8; ++w) {
          simdgroup_matrix<half, 8, 8> kf;
          kf.thread_elements()[0] = as_type<half2>(ka[tt][w / 2])[w % 2];
          kf.thread_elements()[1] = as_type<half2>(kb[tt][w / 2])[w % 2];
          simdgroup_multiply_accumulate(sf, qf[8 * tt + w], kf, sf);
        }
      }
      xs[it % 2][pr][dq][lane] = float2(sf.thread_elements()[0], sf.thread_elements()[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) continue;
    float2 x = xs[it % 2][pr][0][lane];
    UNROLL for (int q = 1; q < NQ; ++q) x += xs[it % 2][pr][q][lane];  // same order in every simdgroup
    const float s0 = ja < n_keys ? x.x : -INFINITY;
    const float s1 = jb < n_keys ? x.y : -INFINITY;
    float mx = max(s0, s1);
    mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
    const float m_new = max(m_run, mx);  // finite: key j0 is always visible
    const float corr = exp(m_run - m_new);
    const float p0 = exp(s0 - m_new), p1 = exp(s1 - m_new);
    float ps = p0 + p1;
    ps += simd_shuffle_xor(ps, ushort(1));
    ps += simd_shuffle_xor(ps, ushort(8));
    l_run = l_run * corr + ps;
    m_run = m_new;
    simdgroup_matrix<half, 8, 8> pf;
    pf.thread_elements()[0] = half(p0);
    pf.thread_elements()[1] = half(p1);
    UNROLL for (int uu = 0; uu < 4; ++uu) {
      UNROLL for (int v = 0; v < 4; ++v) {
        simdgroup_matrix<half, 8, 8> vf;
        vf.thread_elements()[0] = as_type<half2>(va[uu][v]).x;
        vf.thread_elements()[1] = as_type<half2>(va[uu][v]).y;
        of[4 * uu + v].thread_elements()[0] *= corr;
        of[4 * uu + v].thread_elements()[1] *= corr;
        simdgroup_multiply_accumulate(of[4 * uu + v], pf, vf, of[4 * uu + v]);
      }
    }
  }
  // Merge the key slices: slice 1 publishes (O, m, l) per head, slice 0 folds it in.
  if (pr > 0 && fm < G) {
    threadgroup float* pp = mrg[pr - 1][fm];
    UNROLL for (int uu = 0; uu < 4; ++uu) {
      const int u = 4 * dq + uu;
      *((threadgroup float4*)(pp + 32 * u + 4 * fn)) = float4(of[4 * uu].thread_elements()[0], of[4 * uu].thread_elements()[1],
                                                             of[4 * uu + 1].thread_elements()[0], of[4 * uu + 1].thread_elements()[1]);
      *((threadgroup float4*)(pp + 32 * u + 4 * fn + 4)) = float4(of[4 * uu + 2].thread_elements()[0], of[4 * uu + 2].thread_elements()[1],
                                                                 of[4 * uu + 3].thread_elements()[0], of[4 * uu + 3].thread_elements()[1]);
    }
    if (fn == 0 && dq == 0) pp[HD] = m_run, pp[HD + 1] = l_run;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (pr > 0 || fm >= G) return;
  float acc[16][2];
  UNROLL for (int i = 0; i < 16; ++i) acc[i][0] = of[i].thread_elements()[0], acc[i][1] = of[i].thread_elements()[1];
  {
    float M = m_run;
    UNROLL for (int s = 1; s < KP; ++s) M = max(M, mrg[s - 1][fm][HD]);
    const float w0 = (m_run == -INFINITY) ? 0.0f : exp(m_run - M);
    float l = l_run * w0;
    UNROLL for (int i = 0; i < 16; ++i) acc[i][0] *= w0, acc[i][1] *= w0;
    UNROLL for (int s = 1; s < KP; ++s) {
      threadgroup const float* pp = mrg[s - 1][fm];
      const float ms = pp[HD];
      const float w = (ms == -INFINITY) ? 0.0f : exp(ms - M);
      l += w * pp[HD + 1];
      UNROLL for (int uu = 0; uu < 4; ++uu) {
        const int u = 4 * dq + uu;
        const float4 x = *((threadgroup const float4*)(pp + 32 * u + 4 * fn));
        const float4 y = *((threadgroup const float4*)(pp + 32 * u + 4 * fn + 4));
        acc[4 * uu][0] = fma(w, x.x, acc[4 * uu][0]), acc[4 * uu][1] = fma(w, x.y, acc[4 * uu][1]);
        acc[4 * uu + 1][0] = fma(w, x.z, acc[4 * uu + 1][0]), acc[4 * uu + 1][1] = fma(w, x.w, acc[4 * uu + 1][1]);
        acc[4 * uu + 2][0] = fma(w, y.x, acc[4 * uu + 2][0]), acc[4 * uu + 2][1] = fma(w, y.y, acc[4 * uu + 2][1]);
        acc[4 * uu + 3][0] = fma(w, y.z, acc[4 * uu + 3][0]), acc[4 * uu + 3][1] = fma(w, y.w, acc[4 * uu + 3][1]);
      }
    }
    m_run = M, l_run = l;
  }
  if (tile.part >= 0) {
    // Partial record [8 rows][G heads] as g4_attention writes it (row 0).
    device float* rec = parts + ((ulong(tile.part) * p.Hkv + kvh) * 8 * G + ulong(fm)) * R;
    UNROLL for (int uu = 0; uu < 4; ++uu) {
      const int u = 4 * dq + uu;
      *((device float4*)(rec + 32 * u + 4 * fn)) = float4(acc[4 * uu][0], acc[4 * uu][1], acc[4 * uu + 1][0], acc[4 * uu + 1][1]);
      *((device float4*)(rec + 32 * u + 4 * fn + 4)) = float4(acc[4 * uu + 2][0], acc[4 * uu + 2][1], acc[4 * uu + 3][0], acc[4 * uu + 3][1]);
    }
    if (fn == 0 && dq == 0) rec[HD] = m_run, rec[HD + 1] = l_run;
    return;
  }
  const float inv_l = l_run > 0.0f ? 1.0f / l_run : 0.0f;
  device half* op = O + ulong(row) * p.q_ld + (kvh * G + fm) * HD + 4 * fn;
  UNROLL for (int uu = 0; uu < 4; ++uu) {
    const int u = 4 * dq + uu;
    *((device half4*)(op + 32 * u)) = half4(float4(acc[4 * uu][0], acc[4 * uu][1], acc[4 * uu + 1][0], acc[4 * uu + 1][1]) * inv_l);
    *((device half4*)(op + 32 * u + 4)) = half4(float4(acc[4 * uu + 2][0], acc[4 * uu + 2][1], acc[4 * uu + 3][0], acc[4 * uu + 3][1]) * inv_l);
  }
}

#define INSTANTIATE_G4D(HD, G)                                                                         \
  template [[host_name("g4_attention_dec" #HD "_g" #G)]] kernel void g4_attention_dec<HD, G>(          \
      device const half*, device const half*, device const half*, device const half*, device const int*, \
      device const int4*, device const int4*, device half*, constant G4AttnParams&, device const G4Tile*, \
      device const AttnTree*, device float*, uint2, uint, uint);
INSTANTIATE_G4D(256, 1)
INSTANTIATE_G4D(256, 2)
INSTANTIATE_G4D(256, 4)
INSTANTIATE_G4D(256, 8)
INSTANTIATE_G4D(512, 1)
INSTANTIATE_G4D(512, 2)
INSTANTIATE_G4D(512, 4)
INSTANTIATE_G4D(512, 8)

// Merge the split partial records of a tile (attention_combine without the
// output gate). Grid (HD / 4, H, jobs * 8); p.Hkv = KV head groups.
template <int HD>
kernel void g4_attention_combine(
    device const float* parts [[buffer(0)]],
    device half* O [[buffer(1)]],
    constant G4AttnParams& p [[buffer(2)]],
    device const AttnCombine* jobs [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int d4 = int(gid.x);
  const int h = int(gid.y);
  const int r = int(gid.z % 8);
  const AttnCombine job = jobs[gid.z / 8];
  if (r >= job.rows || h >= p.H) return;
  const int G = p.H / p.Hkv;
  const int kvg = h / G, hg = h % G;
  float M = -INFINITY;
  for (int s = 0; s < job.nparts; ++s) {
    device const float* rec = parts + ((ulong(job.part0 + s) * p.Hkv + kvg) * 8 * G + ulong(r) * G + hg) * (HD + 2);
    M = max(M, rec[HD]);
  }
  float L = 0.0f;
  float4 acc = float4(0.0f);
  for (int s = 0; s < job.nparts; ++s) {
    device const float* rec = parts + ((ulong(job.part0 + s) * p.Hkv + kvg) * 8 * G + ulong(r) * G + hg) * (HD + 2);
    const float w = (rec[HD] == -INFINITY) ? 0.0f : exp(rec[HD] - M);
    L += w * rec[HD + 1];
    acc += w * *((device const float4*)(rec + d4 * 4));
  }
  const float inv = L > 0.0f ? 1.0f / L : 0.0f;
  *((device half4*)(O + ulong(job.r0 + r) * p.q_ld + h * HD + d4 * 4)) = half4(acc * inv);
}
template [[host_name("g4_attention_combine256")]] kernel void g4_attention_combine<256>(
    device const float*, device half*, constant G4AttnParams&, device const AttnCombine*, uint3);
template [[host_name("g4_attention_combine512")]] kernel void g4_attention_combine<512>(
    device const float*, device half*, constant G4AttnParams&, device const AttnCombine*, uint3);

// ---------------------------------------------------------------------------
// Residual updates. One threadgroup per row; T: the projection's fp32 rows.
// ---------------------------------------------------------------------------
struct G4NormParams {
  int D;
  float eps;
  float scalar;  // the layer scalar (post-FFN updates)
  int nout;      // normalized fp16 outputs written (0-3)
};

// After attention: h += rmsnorm(T) * w_post, then y_j = rmsnorm(h) * w_j for
// j < nout (the dense MLP's input; MoE layers also the experts' and the router's).
kernel void g4_post_attn(
    device const float* T [[buffer(0)]],
    device float* H [[buffer(1)]],
    device const float* w_post [[buffer(2)]],
    device const float* w0 [[buffer(3)]],
    device const float* w1 [[buffer(4)]],
    device const float* w2 [[buffer(5)]],
    device half* y0 [[buffer(6)]],
    device half* y1 [[buffer(7)]],
    device half* y2 [[buffer(8)]],
    constant G4NormParams& p [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  const int D4 = p.D / 4;
  device const float4* t = (device const float4*)(T + ulong(row) * p.D);
  device float4* h = (device float4*)(H + ulong(row) * p.D);
  device const float4* wp = (device const float4*)w_post;
  float ss = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) ss += dot(t[i], t[i]);
  const float it = rsqrt(g4_tg_sum(ss, part, sg, lane, ntg) / float(p.D) + p.eps);
  float sh = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) {
    const float4 x = h[i] + t[i] * it * wp[i];
    h[i] = x;
    sh += dot(x, x);
  }
  const float ih = rsqrt(g4_tg_sum(sh, part, sg, lane, ntg) / float(p.D) + p.eps);
  device const float4* v0 = (device const float4*)w0;
  device const float4* v1 = (device const float4*)w1;
  device const float4* v2 = (device const float4*)w2;
  device half4* o0 = (device half4*)(y0 + ulong(row) * p.D);
  device half4* o1 = (device half4*)(y1 + ulong(row) * p.D);
  device half4* o2 = (device half4*)(y2 + ulong(row) * p.D);
  for (int i = int(tid); i < D4; i += int(ntg)) {
    const float4 x = h[i] * ih;
    o0[i] = half4(x * v0[i]);
    if (p.nout > 1) o1[i] = half4(x * v1[i]);
    if (p.nout > 2) o2[i] = half4(x * v2[i]);
  }
}

// After a dense MLP: h = (h + rmsnorm(T) * w_post) * scalar, then (nout = 1)
// y = rmsnorm(h) * w_next, the next layer's input.
kernel void g4_post_ffn(
    device const float* T [[buffer(0)]],
    device float* H [[buffer(1)]],
    device const float* w_post [[buffer(2)]],
    device const float* w_next [[buffer(3)]],
    device half* y [[buffer(4)]],
    constant G4NormParams& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  const int D4 = p.D / 4;
  device const float4* t = (device const float4*)(T + ulong(row) * p.D);
  device float4* h = (device float4*)(H + ulong(row) * p.D);
  device const float4* wp = (device const float4*)w_post;
  float ss = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) ss += dot(t[i], t[i]);
  const float it = rsqrt(g4_tg_sum(ss, part, sg, lane, ntg) / float(p.D) + p.eps);
  float sh = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) {
    const float4 x = (h[i] + t[i] * it * wp[i]) * p.scalar;
    h[i] = x;
    sh += dot(x, x);
  }
  if (p.nout == 0) return;
  const float ih = rsqrt(g4_tg_sum(sh, part, sg, lane, ntg) / float(p.D) + p.eps);
  device const float4* wn = (device const float4*)w_next;
  device half4* yo = (device half4*)(y + ulong(row) * p.D);
  for (int i = int(tid); i < D4; i += int(ntg)) yo[i] = half4(h[i] * ih * wn[i]);
}

// After an MoE layer: the experts' outputs ys (sorted pairs, slotpos) summed
// with the routing weights in slot order into T2, then
// h = (h + rmsnorm(rmsnorm(T1) * w1 + rmsnorm(T2) * w2) * w_post) * scalar and
// (nout = 1) y = rmsnorm(h) * w_next. T1: the dense MLP's rows.
struct G4MoeOutParams {
  int D;
  int k;
  float eps;
  float scalar;
  int nout;
};

kernel void g4_moe_out(
    device const float* T1 [[buffer(0)]],
    device const float* ys [[buffer(1)]],
    device const int* slotpos [[buffer(2)]],
    device const float* wts [[buffer(3)]],
    device float* T2 [[buffer(4)]],
    device float* H [[buffer(5)]],
    device const float* w1 [[buffer(6)]],
    device const float* w2 [[buffer(7)]],
    device const float* w_post [[buffer(8)]],
    device const float* w_next [[buffer(9)]],
    device half* y [[buffer(10)]],
    constant G4MoeOutParams& p [[buffer(11)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  threadgroup int sp[16];
  threadgroup float sw[16];
  if (int(tid) < p.k) {
    sp[tid] = slotpos[ulong(row) * p.k + tid];
    sw[tid] = wts[ulong(row) * p.k + tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int D4 = p.D / 4;
  device const float4* t1 = (device const float4*)(T1 + ulong(row) * p.D);
  device float4* t2 = (device float4*)(T2 + ulong(row) * p.D);
  device float4* h = (device float4*)(H + ulong(row) * p.D);
  float s1 = 0.0f, s2 = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) {
    float4 acc = float4(0.0f);
    for (int j = 0; j < p.k; ++j) acc += sw[j] * *((device const float4*)(ys + ulong(sp[j]) * p.D) + i);
    t2[i] = acc;
    s1 += dot(t1[i], t1[i]);
    s2 += dot(acc, acc);
  }
  const float i1 = rsqrt(g4_tg_sum(s1, part, sg, lane, ntg) / float(p.D) + p.eps);
  const float i2 = rsqrt(g4_tg_sum(s2, part, sg, lane, ntg) / float(p.D) + p.eps);
  device const float4* v1 = (device const float4*)w1;
  device const float4* v2 = (device const float4*)w2;
  float s3 = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) {
    const float4 x = t1[i] * i1 * v1[i] + t2[i] * i2 * v2[i];
    t2[i] = x;
    s3 += dot(x, x);
  }
  const float i3 = rsqrt(g4_tg_sum(s3, part, sg, lane, ntg) / float(p.D) + p.eps);
  device const float4* wp = (device const float4*)w_post;
  float s4 = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) {
    const float4 x = (h[i] + t2[i] * i3 * wp[i]) * p.scalar;
    h[i] = x;
    s4 += dot(x, x);
  }
  if (p.nout == 0) return;
  const float i4 = rsqrt(g4_tg_sum(s4, part, sg, lane, ntg) / float(p.D) + p.eps);
  device const float4* wn = (device const float4*)w_next;
  device half4* yo = (device half4*)(y + ulong(row) * p.D);
  for (int i = int(tid); i < D4; i += int(ntg)) yo[i] = half4(h[i] * i4 * wn[i]);
}

// ---------------------------------------------------------------------------
// Routing of a Gemma 4 MoE layer: the top k of the router logits (value
// desc, index asc), the softmax over those k, times per_expert_scale. One
// threadgroup per row with at least E threads. build: a single-row pass's
// tile table, rowmap and slot positions (pair j: expert ids[j], one row) for
// the grouped GEMVs' direct grid (no sort).
// ---------------------------------------------------------------------------
struct G4RouteParams {
  int E;
  int k;
  int build;
  int pad;
};

kernel void g4_route(
    device const float* logits [[buffer(0)]],
    device const float* escale [[buffer(1)]],
    device int* ids [[buffer(2)]],
    device float* w [[buffer(3)]],
    device int4* tiles [[buffer(4)]],
    device int* rowmap [[buffer(5)]],
    device int* slotpos [[buffer(6)]],
    constant G4RouteParams& p [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float redf[32];
  threadgroup int redi[32];
  threadgroup float pv[16];
  threadgroup int pi[16];
  const int e = int(tid);
  float mine = e < p.E ? logits[ulong(row) * p.E + e] : -INFINITY;
  for (int r = 0; r < p.k; ++r) {
    float bv = mine;
    int bi = e < p.E ? e : 0x7fffffff;
    for (ushort off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_down(bv, off);
      const int oi = simd_shuffle_down(bi, off);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if (lane == 0) redf[sg] = bv, redi[sg] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      const bool live = lane < (ntg + 31) / 32;
      float v = live ? redf[lane] : -INFINITY;
      int i = live ? redi[lane] : 0x7fffffff;
      for (ushort off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(v, off);
        const int oi = simd_shuffle_down(i, off);
        if (ov > v || (ov == v && oi < i)) v = ov, i = oi;
      }
      if (lane == 0) pv[r] = v, pi[r] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == pi[r]) mine = -INFINITY;
  }
  if (tid == 0) {
    const float m = pv[0];
    float sum = 0.0f;
    for (int r = 0; r < p.k; ++r) sum += exp(pv[r] - m);
    for (int r = 0; r < p.k; ++r) {
      const int x = pi[r];
      ids[ulong(row) * p.k + r] = x;
      w[ulong(row) * p.k + r] = exp(pv[r] - m) / sum * escale[x];
      if (p.build) {
        tiles[r] = int4(x, r, 1, 0);
        rowmap[r] = 0;
        slotpos[r] = r;
      }
    }
  }
}
