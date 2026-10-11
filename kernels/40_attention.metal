// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ---------------------------------------------------------------------------
// Full attention: per-head RMSNorm, partial M-RoPE (half split over the first
// 64 dims), K/V written as fp16 into the batch scratch. One simdgroup per
// (row, head); heads [0, H) query, [H, H + Hkv) key/value.
// ---------------------------------------------------------------------------
struct AttnPrepParams {
  int H;
  int Hkv;
  int ld;      // fused projection row stride
  int k_off;   // column of k (v follows at + Hkv*256)
  int q_ld;    // Q output stride (H*256)
  int kv_ld;   // K/V output stride (Hkv*256)
  float eps;
};

kernel void attn_prep(
    device const float* X [[buffer(0)]],
    device const float* qn_w [[buffer(1)]],
    device const float* kn_w [[buffer(2)]],
    device const int4* pos [[buffer(3)]],
    device const float* inv_freq [[buffer(4)]],
    device half* Q [[buffer(5)]],
    device half* Kb [[buffer(6)]],
    device half* Vb [[buffer(7)]],
    constant AttnPrepParams& p [[buffer(8)]],
    device const int* rope_sel [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int HD = 256;
  constexpr int NPT = HD / 32;
  const int head = tgpig.x * 4 + sg;
  const ulong row = tgpig.y;
  if (head >= p.H + p.Hkv) return;
  const bool is_q = head < p.H;
  const int h = is_q ? head : head - p.H;
  device const float* src = X + row * p.ld + (is_q ? h * (2 * HD) : p.k_off + h * HD);
  device const float* nw = is_q ? qn_w : kn_w;
  float x[NPT];
  float ss = 0.0f;
  UNROLL for (int i = 0; i < NPT; ++i) {
    x[i] = src[lane * NPT + i];
    ss += x[i] * x[i];
  }
  ss = simd_sum(ss);
  const float inv = rsqrt(ss / float(HD) + p.eps);
  UNROLL for (int i = 0; i < NPT; ++i) x[i] = x[i] * inv * nw[lane * NPT + i];
  // M-RoPE: frequency f rotates by the position component rope_sel[f]
  // (0 temporal, 1 height, 2 width); text rows carry equal components.
  const int4 pr = pos[row];
  UNROLL for (int i = 0; i < NPT; ++i) {
    const float partner = simd_shuffle_xor(x[i], ushort(4));
    if (lane < 8) {
      const int f = (lane * NPT + i) & 31;
      const int sel = rope_sel[f];
      const float position = float(sel == 0 ? pr.x : (sel == 1 ? pr.y : pr.z));
      const float ang = position * inv_freq[f];
      const float c = precise::cos(ang);
      const float s = precise::sin(ang);
      x[i] = (lane < 4) ? (x[i] * c - partner * s) : (x[i] * c + partner * s);
    }
  }
  if (is_q) {
    device half* q = Q + row * p.q_ld + h * HD + lane * NPT;
    UNROLL for (int i = 0; i < NPT; ++i) q[i] = half(x[i]);
  } else {
    device half* k = Kb + row * p.kv_ld + h * HD + lane * NPT;
    device const float* v = X + row * p.ld + p.k_off + p.Hkv * HD + h * HD + lane * NPT;
    device half* vo = Vb + row * p.kv_ld + h * HD + lane * NPT;
    UNROLL for (int i = 0; i < NPT; ++i) {
      k[i] = half(x[i]);
      vo[i] = half(v[i]);
    }
  }
}

// Forest attention over a paged KV cache.
//
// Keys of a query row: the tree's cached keys [0, P) (paged: 64-token blocks
// listed in `blocks` from index tree.table), then the batch rows of its own
// tree with the tree mask: desc[row] = (seg_start, 0, anc_len, 0) relative to
// the tree's first row; visible batch keys are j < anc_len (the root) and
// seg_start <= j <= row (its own segment). Output (fp16) = softmax(qK^T/16) V
// * sigmoid(gate).
//
// Threadgroup = one KV head x one tile of 8 consecutive query rows of one
// tree; 2 simdgroups per query head (each owns 128 of the 256 dims; score
// partials are exchanged through threadgroup memory). A tile scans the cached
// key range [k0, k1); long contexts are split over several tiles (flash
// decoding) whose partial (max, sum, O) records are merged by
// attention_combine. Only the tile with `last` set also scans the batch keys.
constant constexpr int FA_BK = 16;
constant constexpr int FA_LD = 256 + 8;

struct AttnParams {
  int H;
  int Hkv;
  int q_ld;     // Q/O row stride (H*256)
  int kv_ld;    // K/V row width (Hkv*256), batch and pool
  int g_ld;     // fused projection stride (gate lives next to q)
  int bt;       // tokens per KV block
  ulong kv_base;  // half offset of this layer's region in the KV pool
};

struct AttnTree {
  int base;   // absolute first row
  int P;      // cached keys
  int table;  // first entry of its block list
  int M;      // rows in the pass
};

struct AttnTile {
  int tree;
  int r0;     // absolute first query row
  int k0;     // cached-key range [k0, k1)
  int k1;
  int part;   // -1: write O; else partial record index (per KV head)
  int last;   // also scan the batch keys
  int pad0;
  int pad1;
};

// SPLIT > 1: SPLIT groups of G query heads share one cached KV head (the grid's
// x is KV heads x SPLIT, p.Hkv counts those groups): GQA 12 runs as 2 x 6, since
// 64 x 12 threads exceed the pipeline's 512.
template <int G, int SPLIT = 1>
kernel void flash_attention(
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
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NT = 64 * G;
  threadgroup half Kt[FA_BK * FA_LD];
  threadgroup half Vt[FA_BK * FA_LD];
  threadgroup float xs[2 * G][8][FA_BK];

  const int kvh = tgid.x;
  const int kvr = kvh / SPLIT;  // cached KV head
  const AttnTile tile = tiles[tgid.y];
  const AttnTree tree = trees[tile.tree];
  const int hg = sg / 2;
  const int dh = sg % 2;
  const int h = kvh * G + hg;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  const int end = tree.base + tree.M;  // one past the tree's last row
  const int row = tile.r0 + fm;
  const bool row_ok = row < end;
  const int rr = row - tree.base;  // relative row
  const int4 d = desc[min(row, end - 1)];
  const int r_last = min(tile.r0 + 7, end - 1) - tree.base;  // relative

  // Query fragments (scaled by 1/sqrt(256) = 1/16, exact in fp16).
  simdgroup_matrix<half, 8, 8> qf[16];
  {
    device const half* qp = Q + ulong(min(row, end - 1)) * p.q_ld + h * 256 + dh * 128 + fn;
    UNROLL for (int ks = 0; ks < 16; ++ks) {
      const half2 v = *((device const half2*)(qp + ks * 8)) * half(0.0625h);
      qf[ks].thread_elements()[0] = v.x;
      qf[ks].thread_elements()[1] = v.y;
    }
  }
  simdgroup_matrix<float, 8, 8> of[16];
  UNROLL for (int i = 0; i < 16; ++i) of[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  float m_run = -INFINITY;
  float l_run = 0.0f;

  // Block visibility for the batch range (union over the tile's rows).
  int anc_max = 0, seg_min = 1 << 30;
  for (int r = tile.r0; r <= tree.base + r_last; ++r) {
    const int4 dr = desc[r];
    anc_max = max(anc_max, dr.z);
    seg_min = min(seg_min, dr.x);
  }

  const ulong blk_halves = ulong(p.bt) * p.kv_ld;
  device const half* kvl = kv + p.kv_base;
  const int n_slot_blocks = (tile.k1 - tile.k0 + FA_BK - 1) / FA_BK;
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
    // Cooperative load of K and V blocks: 16 keys x 256 dims each.
    for (int c = tid; c < FA_BK * 32 * 2; c += NT) {
      const int which = c / (FA_BK * 32);
      const int cc = c % (FA_BK * 32);
      const int key = cc / 32;
      const int col = (cc % 32) * 8;
      const int j = j0 + key;
      uint4 v = uint4(0);
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
      *((threadgroup uint4*)((which ? Vt : Kt) + key * FA_LD + col)) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Partial scores over this simdgroup's 128 dims: 2 key fragments.
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
    // Full scores, tree mask, online softmax.
    float sv[4];
    UNROLL for (int kf = 0; kf < 2; ++kf) {
      UNROLL for (int e = 0; e < 2; ++e) {
        const int kc = kf * 8 + fn + e;
        float v = sf[kf].thread_elements()[e] + xs[sg ^ 1][fm][kc];
        const int j = j0 + kc;
        bool vis = row_ok && j < j_end;
        if (!in_slot) vis = vis && ((j < d.z) || (j >= d.x && j <= rr));
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
    // O += P V over this simdgroup's 128 dims.
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
    // Partial record [8 rows][G heads]: 256 O floats (unnormalized), then m and l.
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

#define INSTANTIATE_FA(G)                                                          \
  template [[host_name("flash_attention_g" #G)]] kernel void flash_attention<G>( \
      device const half*, device const half*, device const half*,                \
      device const half*, device const int*, device const int4*,                 \
      device const float*, device half*, constant AttnParams&,                   \
      device const AttnTile*, device const AttnTree*, device float*, uint2, uint, \
      uint, uint);

INSTANTIATE_FA(1)
INSTANTIATE_FA(2)
INSTANTIATE_FA(3)
INSTANTIATE_FA(4)
INSTANTIATE_FA(6)
INSTANTIATE_FA(8)
template [[host_name("flash_attention_g6s2")]] kernel void flash_attention<6, 2>(  // Qwen4-Exp: GQA 12 as 2 x 6
    device const half*, device const half*, device const half*, device const half*, device const int*,
    device const int4*, device const float*, device half*, constant AttnParams&, device const AttnTile*,
    device const AttnTree*, device float*, uint2, uint, uint, uint);

// ---------------------------------------------------------------------------
// Decode attention for single-row trees (attn-decode-vec): GEMV-style,
// no MMA. Threadgroup = one KV head x one tile (key range [k0, k1) of one tree
// with M == 1, plus the row's own batch key when `last`); NHH head halves x
// KS key slices of simdgroups: a simdgroup owns HH <= 4 of the G query heads
// and every KS-th batch of 4 keys. A lane owns 8 of the 256 dims: q (HH heads)
// stays in registers, K/V rows are streamed as coalesced 16-byte loads, the
// scores of 4 keys x HH heads are reduced across the lanes with a butterfly
// transpose (16 shuffles per 4 keys), the online softmax runs replicated in
// every lane, O accumulates in registers; the key slices merge their (m, l, O)
// through threadgroup memory. Writes the same partial record as
// flash_attention (attention_combine merges tiles) or the gated output. The
// 8-row MMA tiles of flash_attention spend 7/8 of their rows on padding at
// one query row; the first version of this kernel (2 simdgroups per
// threadgroup, no key slices) measured 2.8 % slower than flash at a short
// context: one 32-key chain per simdgroup on an otherwise idle core.
// ---------------------------------------------------------------------------
template <int G>
kernel void attn_decode_vec(
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
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int HH = G <= 4 ? G : G / 2;  // heads per simdgroup
  constexpr int NHH = G / HH;             // head halves
  constexpr int KS = 4;                   // key slices (simdgroups per head half)
  constexpr int KB = 4;                   // keys per reduction batch
  threadgroup float part[(KS - 1) * NHH * HH][258];
  const int kvh = tgid.x;
  const AttnTile tile = tiles[tgid.y];
  const AttnTree tree = trees[tile.tree];
  const int row = tile.r0;  // the tree's only row
  const int hf = int(sg) % NHH, ks = int(sg) / NHH;
  const int hg0 = hf * HH;
  const int d0 = int(lane) * 8;
  float q[HH][8];
  UNROLL for (int hh = 0; hh < HH; ++hh) {
    device const half4* qp = (device const half4*)(Q + ulong(row) * p.q_ld + (kvh * G + hg0 + hh) * 256 + d0);
    const half4 a = qp[0], b = qp[1];
    q[hh][0] = float(a.x) * 0.0625f, q[hh][1] = float(a.y) * 0.0625f, q[hh][2] = float(a.z) * 0.0625f, q[hh][3] = float(a.w) * 0.0625f;
    q[hh][4] = float(b.x) * 0.0625f, q[hh][5] = float(b.y) * 0.0625f, q[hh][6] = float(b.z) * 0.0625f, q[hh][7] = float(b.w) * 0.0625f;
  }
  float acc[HH][8];
  float m_run[HH], l_run[HH];
  UNROLL for (int hh = 0; hh < HH; ++hh) {
    m_run[hh] = -INFINITY, l_run[hh] = 0.0f;
    UNROLL for (int d = 0; d < 8; ++d) acc[hh][d] = 0.0f;
  }
  const ulong blk_halves = ulong(p.bt) * p.kv_ld;
  device const half* kvl = kv + p.kv_base;
  const int n_slot = tile.k1 - tile.k0;
  const int n_keys = n_slot + (tile.last ? 1 : 0);  // + the row's own key (always visible to itself)
  auto key_src = [&](int j, int which) -> device const half* {
    if (j < n_slot) {
      const int jj = tile.k0 + j;
      const ulong b = ulong(blocks[tree.table + jj / p.bt]);
      return kvl + (b * 2 + which) * blk_halves + ulong(jj % p.bt) * p.kv_ld + kvh * 256 + d0;
    }
    return (which ? Vb : Kb) + ulong(tree.base) * p.kv_ld + kvh * 256 + d0;
  };
  const bool hi16 = lane & 16, hi8 = lane & 8, hi4 = lane & 4, hi2 = lane & 2;
  for (int j0 = ks * KB; j0 < n_keys; j0 += KS * KB) {
    // Partial dot products of this lane's 8 dims: v[kk * 4 + hh] (heads padded to 4).
    float v[16];
    UNROLL for (int i = 0; i < 16; ++i) v[i] = 0.0f;
    UNROLL for (int kk = 0; kk < KB; ++kk) {
      const int j = j0 + kk;
      if (j < n_keys) {
        device const half4* kp = (device const half4*)key_src(j, 0);
        const half4 a = kp[0], b = kp[1];
        const float k8[8] = {float(a.x), float(a.y), float(a.z), float(a.w), float(b.x), float(b.y), float(b.z), float(b.w)};
        UNROLL for (int hh = 0; hh < HH; ++hh) {
          float t = 0.0f;
          UNROLL for (int d = 0; d < 8; ++d) t = fma(q[hh][d], k8[d], t);
          v[kk * 4 + hh] = t;
        }
      }
    }
    // Butterfly transpose-reduce over the 32 lanes: 16 -> 8 -> 4 -> 2 -> 1
    // values per lane, each summed over 2, 4, 8, 16 lanes; then the last bit.
    float t8[8], t4[4], t2[2], s;
    UNROLL for (int i = 0; i < 8; ++i) {
      const float send = hi16 ? v[i] : v[i + 8];
      t8[i] = (hi16 ? v[i + 8] : v[i]) + simd_shuffle_xor(send, ushort(16));
    }
    UNROLL for (int i = 0; i < 4; ++i) {
      const float send = hi8 ? t8[i] : t8[i + 4];
      t4[i] = (hi8 ? t8[i + 4] : t8[i]) + simd_shuffle_xor(send, ushort(8));
    }
    UNROLL for (int i = 0; i < 2; ++i) {
      const float send = hi4 ? t4[i] : t4[i + 2];
      t2[i] = (hi4 ? t4[i + 2] : t4[i]) + simd_shuffle_xor(send, ushort(4));
    }
    {
      const float send = hi2 ? t2[0] : t2[1];
      s = (hi2 ? t2[1] : t2[0]) + simd_shuffle_xor(send, ushort(2));
    }
    s += simd_shuffle_xor(s, ushort(1));
    // Lane L now holds the score of index (L >> 1) & 15 = kk * 4 + hh; broadcast all 16.
    float sc[KB][4];
    UNROLL for (int kk = 0; kk < KB; ++kk)
      UNROLL for (int hh = 0; hh < 4; ++hh) sc[kk][hh] = simd_shuffle(s, ushort((kk * 4 + hh) * 2));
    // Online softmax per head (replicated in every lane), then O += p V.
    float pr[KB][HH];
    UNROLL for (int hh = 0; hh < HH; ++hh) {
      float mx = -INFINITY;
      UNROLL for (int kk = 0; kk < KB; ++kk) {
        sc[kk][hh] = (j0 + kk < n_keys) ? sc[kk][hh] : -INFINITY;
        mx = max(mx, sc[kk][hh]);
      }
      const float m_new = max(m_run[hh], mx);
      const float corr = (m_new == -INFINITY) ? 1.0f : exp(m_run[hh] - m_new);
      float ps = 0.0f;
      UNROLL for (int kk = 0; kk < KB; ++kk) {
        pr[kk][hh] = (m_new == -INFINITY) ? 0.0f : exp(sc[kk][hh] - m_new);
        ps += pr[kk][hh];
      }
      l_run[hh] = l_run[hh] * corr + ps;
      m_run[hh] = m_new;
      UNROLL for (int d = 0; d < 8; ++d) acc[hh][d] *= corr;
    }
    UNROLL for (int kk = 0; kk < KB; ++kk) {
      const int j = j0 + kk;
      if (j < n_keys) {
        device const half4* vp = (device const half4*)key_src(j, 1);
        const half4 a = vp[0], b = vp[1];
        const float v8[8] = {float(a.x), float(a.y), float(a.z), float(a.w), float(b.x), float(b.y), float(b.z), float(b.w)};
        UNROLL for (int hh = 0; hh < HH; ++hh)
          UNROLL for (int d = 0; d < 8; ++d) acc[hh][d] = fma(pr[kk][hh], v8[d], acc[hh][d]);
      }
    }
  }
  // Merge the key slices: slices 1.. publish (O, m, l), slice 0 folds them in.
  if (ks > 0) {
    UNROLL for (int hh = 0; hh < HH; ++hh) {
      threadgroup float* pp = part[((ks - 1) * NHH + hf) * HH + hh];
      UNROLL for (int d = 0; d < 8; ++d) pp[d0 + d] = acc[hh][d];
      if (lane == 0) pp[256] = m_run[hh], pp[257] = l_run[hh];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ks > 0) return;
  UNROLL for (int hh = 0; hh < HH; ++hh) {
    float M = m_run[hh];
    UNROLL for (int s2 = 1; s2 < KS; ++s2) M = max(M, part[((s2 - 1) * NHH + hf) * HH + hh][256]);
    const float w0 = (M == -INFINITY) ? 1.0f : exp(m_run[hh] - M);
    float l = l_run[hh] * w0;
    UNROLL for (int d = 0; d < 8; ++d) acc[hh][d] *= w0;
    UNROLL for (int s2 = 1; s2 < KS; ++s2) {
      threadgroup const float* pp = part[((s2 - 1) * NHH + hf) * HH + hh];
      const float ms = pp[256];
      const float w = (ms == -INFINITY) ? 0.0f : exp(ms - M);
      l += w * pp[257];
      UNROLL for (int d = 0; d < 8; ++d) acc[hh][d] = fma(w, pp[d0 + d], acc[hh][d]);
    }
    m_run[hh] = M, l_run[hh] = l;
  }
  if (tile.part >= 0) {
    // Partial record [8 rows][G heads] as flash_attention writes it (row 0).
    UNROLL for (int hh = 0; hh < HH; ++hh) {
      device float* rec = parts + ((ulong(tile.part) * p.Hkv + kvh) * 8 * G + ulong(hg0 + hh)) * 258;
      *((device float4*)(rec + d0)) = float4(acc[hh][0], acc[hh][1], acc[hh][2], acc[hh][3]);
      *((device float4*)(rec + d0 + 4)) = float4(acc[hh][4], acc[hh][5], acc[hh][6], acc[hh][7]);
      if (lane == 0) rec[256] = m_run[hh], rec[257] = l_run[hh];
    }
    return;
  }
  UNROLL for (int hh = 0; hh < HH; ++hh) {
    const int h = kvh * G + hg0 + hh;
    const float inv_l = l_run[hh] > 0.0f ? 1.0f / l_run[hh] : 0.0f;
    device const float* gp = X + ulong(row) * p.g_ld + h * 512 + 256 + d0;
    device half* op = O + ulong(row) * p.q_ld + h * 256 + d0;
    const float4 g0 = *((device const float4*)gp), g1 = *((device const float4*)(gp + 4));
    const float4 o0 = float4(acc[hh][0], acc[hh][1], acc[hh][2], acc[hh][3]) * inv_l / (1.0f + exp(-g0));
    const float4 o1 = float4(acc[hh][4], acc[hh][5], acc[hh][6], acc[hh][7]) * inv_l / (1.0f + exp(-g1));
    *((device half4*)op) = half4(o0);
    *((device half4*)(op + 4)) = half4(o1);
  }
}

#define INSTANTIATE_AV(G)                                                          \
  template [[host_name("attn_decode_vec_g" #G)]] kernel void attn_decode_vec<G>( \
      device const half*, device const half*, device const half*,                \
      device const half*, device const int*, device const int4*,                 \
      device const float*, device half*, constant AttnParams&,                   \
      device const AttnTile*, device const AttnTree*, device float*, uint2,      \
      uint, uint);

INSTANTIATE_AV(1)
INSTANTIATE_AV(2)
INSTANTIATE_AV(3)
INSTANTIATE_AV(4)
INSTANTIATE_AV(6)
INSTANTIATE_AV(8)

// ---------------------------------------------------------------------------
// Decode attention for single-row trees with the GQA group as the MMA rows:
// the G <= 8 query heads of a KV head are the 8 rows of the simdgroup_matrix
// tiles (flash_attention spends 7 of its 8 rows on padding at one query row;
// rows past G here repeat head G - 1 and are never written). Threadgroup =
// one KV head x one tile (key range [k0, k1) of a tree with M == 1, plus the
// row's own batch key when `last`): 2 pairs of simdgroups take turns over
// 8-key blocks; in a pair each simdgroup owns 128 of the 256 dims (q, K, V
// and O of its half; with whole-dim simdgroups, ~128 registers each, the
// loads and the MMAs did not overlap: 241 GB/s). Per block: partial S
// (8 heads x 8 keys) = Q K^T over the simdgroup's 16 dim blocks, the pair's
// two partials summed through threadgroup memory (one barrier per block,
// double-buffered, summed in the same order in both), the online softmax,
// then O += P V over its 16 output blocks. K and V fragments come straight
// from the KV pool as 16-byte loads per lane: the dims are permuted inside
// the products (S's block 8t + w takes dims 64t + 8c + w at fragment index
// c; O's block 4u + v holds dims 32u + 8(c / 2) + 2v + c % 2), so a lane's
// elements of 8 (K) or 4 (V) consecutive blocks are 8 consecutive halves of
// one key row, and each load instruction covers whole 128-byte lines. P
// enters the PV tiles as fp16 and everything accumulates in fp32, as in
// flash_attention. The pairs merge their (m, l, O) through threadgroup
// memory; the result is flash_attention's partial record (attention_combine
// merges the tiles) or the gated output. Measured alone on the 27B
// (`bench_native --micro attn`: one decode row, each of the 16 layers over
// its own KV region): 383 µs per layer at 32K keys (351 GB/s, what the same
// loads reach without the MMAs) against 1613 µs for flash_attention.
// ---------------------------------------------------------------------------
template <int G>
kernel void attn_decode_gqa(
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
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int KP = 2;  // simdgroup pairs (key slices)
  threadgroup float mrg[KP - 1][G][258];
  threadgroup float2 xs[2][KP][2][32];  // [buffer][pair][dim half][lane]: partial scores
  const int kvh = tgid.x;
  const AttnTile tile = tiles[tgid.y];
  const AttnTree tree = trees[tile.tree];
  const int row = tile.r0;  // the tree's only row
  const int dh = int(sg) % 2, pr = int(sg) / 2;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);    // tile row: query head
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;  // tile columns fn, fn + 1
  const int hq = min(int(fm), G - 1);

  // Q fragments of this dim half: S blocks 8t + w, t in {2 dh, 2 dh + 1}.
  simdgroup_matrix<half, 8, 8> qf[16];
  {
    device const half* qp = Q + ulong(row) * p.q_ld + (kvh * G + hq) * 256 + 8 * fn;
    UNROLL for (int tt = 0; tt < 2; ++tt) {
      const int t = 2 * dh + tt;
      const uint4 a = *((device const uint4*)(qp + 64 * t));
      const uint4 b = *((device const uint4*)(qp + 64 * t + 8));
      UNROLL for (int w = 0; w < 8; ++w) {
        qf[8 * tt + w].thread_elements()[0] = as_type<half2>(a[w / 2])[w % 2];
        qf[8 * tt + w].thread_elements()[1] = as_type<half2>(b[w / 2])[w % 2];
      }
    }
  }
  simdgroup_matrix<float, 8, 8> of[16];  // O blocks 4u + v, u = 4 dh + uu
  UNROLL for (int i = 0; i < 16; ++i) of[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  float m_run = -INFINITY, l_run = 0.0f;

  const ulong blk_halves = ulong(p.bt) * p.kv_ld;
  device const half* kvl = kv + p.kv_base + kvh * 256;
  const int n_slot = tile.k1 - tile.k0;
  const int n_keys = n_slot + (tile.last ? 1 : 0);  // + the row's own key
  // Key (which 0) or value (1) row of tile key j < n_keys.
  auto key_row = [&](int j, int which) -> device const half* {
    if (j < n_slot) {
      const int jj = tile.k0 + j;
      const ulong b = ulong(blocks[tree.table + jj / p.bt]);
      return kvl + (b * 2 + which) * blk_halves + ulong(jj % p.bt) * p.kv_ld;
    }
    return (which ? Vb : Kb) + ulong(tree.base) * p.kv_ld + kvh * 256;
  };
  const int n_iter = ((n_keys + 7) / 8 + KP - 1) / KP;  // the same in every simdgroup (barriers)
  for (int it = 0; it < n_iter; ++it) {
    const int j0 = 8 * (it * KP + pr);
    const bool live = j0 < n_keys;
    const int ja = j0 + fn, jb = j0 + fn + 1, jv = j0 + fm;
    uint4 va[4];
    if (live) {
      // This lane's K elements: dims 64t + 8 fm + [0, 8) of keys fn and fn + 1; V
      // elements: dims 32u + 4 fn + [0, 8) of key fm (zero past the keys).
      device const half* pa = key_row(min(ja, n_keys - 1), 0) + 8 * fm;
      device const half* pb = key_row(min(jb, n_keys - 1), 0) + 8 * fm;
      uint4 ka[2], kb[2];
      UNROLL for (int tt = 0; tt < 2; ++tt) {
        ka[tt] = *((device const uint4*)(pa + 64 * (2 * dh + tt)));
        kb[tt] = *((device const uint4*)(pb + 64 * (2 * dh + tt)));
      }
      device const half* vp = key_row(min(jv, n_keys - 1), 1) + 4 * fn;
      const bool v_ok = jv < n_keys;
      UNROLL for (int uu = 0; uu < 4; ++uu) va[uu] = v_ok ? *((device const uint4*)(vp + 32 * (4 * dh + uu))) : uint4(0);
      simdgroup_matrix<float, 8, 8> sf = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      UNROLL for (int tt = 0; tt < 2; ++tt) {
        UNROLL for (int w = 0; w < 8; ++w) {
          simdgroup_matrix<half, 8, 8> kf;
          kf.thread_elements()[0] = as_type<half2>(ka[tt][w / 2])[w % 2];
          kf.thread_elements()[1] = as_type<half2>(kb[tt][w / 2])[w % 2];
          simdgroup_multiply_accumulate(sf, qf[8 * tt + w], kf, sf);
        }
      }
      xs[it % 2][pr][dh][lane] = float2(sf.thread_elements()[0], sf.thread_elements()[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) continue;
    // Online softmax per head (row fm: lanes ^1 and ^8 hold its other keys).
    const float2 x0 = xs[it % 2][pr][0][lane], x1 = xs[it % 2][pr][1][lane];
    const float s0 = ja < n_keys ? (x0.x + x1.x) * 0.0625f : -INFINITY;
    const float s1 = jb < n_keys ? (x0.y + x1.y) * 0.0625f : -INFINITY;
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
  // Merge the pairs: pair 1 publishes (O, m, l) per head, pair 0 folds it in.
  if (pr > 0 && fm < G) {
    threadgroup float* pp = mrg[pr - 1][fm];
    UNROLL for (int uu = 0; uu < 4; ++uu) {
      const int u = 4 * dh + uu;
      *((threadgroup float4*)(pp + 32 * u + 4 * fn)) = float4(of[4 * uu].thread_elements()[0], of[4 * uu].thread_elements()[1],
                                                             of[4 * uu + 1].thread_elements()[0], of[4 * uu + 1].thread_elements()[1]);
      *((threadgroup float4*)(pp + 32 * u + 4 * fn + 4)) = float4(of[4 * uu + 2].thread_elements()[0], of[4 * uu + 2].thread_elements()[1],
                                                                 of[4 * uu + 3].thread_elements()[0], of[4 * uu + 3].thread_elements()[1]);
    }
    if (fn == 0 && dh == 0) pp[256] = m_run, pp[257] = l_run;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (pr > 0 || fm >= G) return;
  float acc[16][2];
  UNROLL for (int i = 0; i < 16; ++i) acc[i][0] = of[i].thread_elements()[0], acc[i][1] = of[i].thread_elements()[1];
  {
    float M = m_run;
    UNROLL for (int s = 1; s < KP; ++s) M = max(M, mrg[s - 1][fm][256]);
    const float w0 = (m_run == -INFINITY) ? 0.0f : exp(m_run - M);
    float l = l_run * w0;
    UNROLL for (int i = 0; i < 16; ++i) acc[i][0] *= w0, acc[i][1] *= w0;
    UNROLL for (int s = 1; s < KP; ++s) {
      threadgroup const float* pp = mrg[s - 1][fm];
      const float ms = pp[256];
      const float w = (ms == -INFINITY) ? 0.0f : exp(ms - M);
      l += w * pp[257];
      UNROLL for (int uu = 0; uu < 4; ++uu) {
        const int u = 4 * dh + uu;
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
    // Partial record [8 rows][G heads] as flash_attention writes it (row 0).
    device float* rec = parts + ((ulong(tile.part) * p.Hkv + kvh) * 8 * G + ulong(fm)) * 258;
    UNROLL for (int uu = 0; uu < 4; ++uu) {
      const int u = 4 * dh + uu;
      *((device float4*)(rec + 32 * u + 4 * fn)) = float4(acc[4 * uu][0], acc[4 * uu][1], acc[4 * uu + 1][0], acc[4 * uu + 1][1]);
      *((device float4*)(rec + 32 * u + 4 * fn + 4)) = float4(acc[4 * uu + 2][0], acc[4 * uu + 2][1], acc[4 * uu + 3][0], acc[4 * uu + 3][1]);
    }
    if (fn == 0 && dh == 0) rec[256] = m_run, rec[257] = l_run;
    return;
  }
  const int h = kvh * G + fm;
  const float inv_l = l_run > 0.0f ? 1.0f / l_run : 0.0f;
  device const float* gp = X + ulong(row) * p.g_ld + h * 512 + 256 + 4 * fn;
  device half* op = O + ulong(row) * p.q_ld + h * 256 + 4 * fn;
  UNROLL for (int uu = 0; uu < 4; ++uu) {
    const int u = 4 * dh + uu;
    const float4 g0 = *((device const float4*)(gp + 32 * u)), g1 = *((device const float4*)(gp + 32 * u + 4));
    const float4 o0 = float4(acc[4 * uu][0], acc[4 * uu][1], acc[4 * uu + 1][0], acc[4 * uu + 1][1]) * inv_l / (1.0f + exp(-g0));
    const float4 o1 = float4(acc[4 * uu + 2][0], acc[4 * uu + 2][1], acc[4 * uu + 3][0], acc[4 * uu + 3][1]) * inv_l / (1.0f + exp(-g1));
    *((device half4*)(op + 32 * u)) = half4(o0);
    *((device half4*)(op + 32 * u + 4)) = half4(o1);
  }
}

#define INSTANTIATE_AG(G)                                                          \
  template [[host_name("attn_decode_gqa_g" #G)]] kernel void attn_decode_gqa<G>( \
      device const half*, device const half*, device const half*,                \
      device const half*, device const int*, device const int4*,                 \
      device const float*, device half*, constant AttnParams&,                   \
      device const AttnTile*, device const AttnTree*, device float*, uint2,      \
      uint, uint);

INSTANTIATE_AG(1)
INSTANTIATE_AG(2)
INSTANTIATE_AG(3)
INSTANTIATE_AG(4)
INSTANTIATE_AG(6)
INSTANTIATE_AG(8)

// Merge the split-K partial records of a tile: O = sum_s e^(m_s - M) O_s /
// sum_s e^(m_s - M) l_s, then the output gate. One job per split tile group:
// (r0 absolute, rows, first record, record count). Grid (256 / 4, H, jobs * 8).
// The records are read 8 at a time and summed in order (a loop that waited
// for each record's loads in turn took 39 µs per layer at 64 records, 11 µs
// this way).
struct AttnCombine {
  int r0;
  int rows;
  int part0;
  int nparts;
};

kernel void attention_combine(
    device const float* parts [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device half* O [[buffer(2)]],
    constant AttnParams& p [[buffer(3)]],
    device const AttnCombine* jobs [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int d4 = gid.x;  // 4 dims per thread
  const int h = gid.y;
  const int job_i = gid.z / 8;
  const int r = gid.z % 8;
  const AttnCombine job = jobs[job_i];
  if (r >= job.rows || h >= p.H) return;
  const int G = p.H / p.Hkv;
  const int kvh = h / G, hg = h % G;
  const ulong stride = ulong(p.Hkv) * 8 * G * 258;  // floats from one part's record to the next
  device const float* rec0 = parts + ((ulong(job.part0) * p.Hkv + kvh) * 8 * G + ulong(r) * G + hg) * 258;
  const int n = job.nparts;
  float M = -INFINITY;
  int s = 0;
  for (; s + 8 <= n; s += 8) {
    float m8[8];
    UNROLL for (int i = 0; i < 8; ++i) m8[i] = rec0[ulong(s + i) * stride + 256];
    UNROLL for (int i = 0; i < 8; ++i) M = max(M, m8[i]);
  }
  for (; s < n; ++s) M = max(M, rec0[ulong(s) * stride + 256]);
  float L = 0.0f;
  float4 acc = float4(0.0f);
  s = 0;
  for (; s + 8 <= n; s += 8) {
    float m8[8], l8[8];
    float4 o8[8];
    UNROLL for (int i = 0; i < 8; ++i) {
      device const float* rec = rec0 + ulong(s + i) * stride;
      m8[i] = rec[256], l8[i] = rec[257], o8[i] = *((device const float4*)(rec + d4 * 4));
    }
    UNROLL for (int i = 0; i < 8; ++i) {
      const float w = (m8[i] == -INFINITY) ? 0.0f : exp(m8[i] - M);
      L += w * l8[i];
      acc += w * o8[i];
    }
  }
  for (; s < n; ++s) {
    device const float* rec = rec0 + ulong(s) * stride;
    const float w = (rec[256] == -INFINITY) ? 0.0f : exp(rec[256] - M);
    L += w * rec[257];
    acc += w * *((device const float4*)(rec + d4 * 4));
  }
  const int row = job.r0 + r;
  const float inv = L > 0.0f ? 1.0f / L : 0.0f;
  const float4 g = *((device const float4*)(X + ulong(row) * p.g_ld + h * 512 + 256 + d4 * 4));
  const float4 o = acc * inv / (1.0f + exp(-g));
  *((device half4*)(O + ulong(row) * p.q_ld + h * 256 + d4 * 4)) = half4(o);
}

// Write batch rows' K/V into the paged pool: entry = (source row, block,
// position within the block). Grid (kv_ld / 8, entries).
struct KVSave {
  int src_row;
  int block;
  int pos;
  int pad;
};

kernel void kv_save(
    device const half* Kb [[buffer(0)]],
    device const half* Vb [[buffer(1)]],
    device half* kv [[buffer(2)]],
    device const KVSave* entries [[buffer(3)]],
    constant AttnParams& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) * 8 >= p.kv_ld) return;
  const KVSave e = entries[gid.y];
  const ulong src = ulong(e.src_row) * p.kv_ld + gid.x * 8;
  const ulong blk = ulong(p.bt) * p.kv_ld;
  device half* dst = kv + p.kv_base + ulong(e.block) * 2 * blk + ulong(e.pos) * p.kv_ld + gid.x * 8;
  *((device uint4*)dst) = *((device const uint4*)(Kb + src));
  *((device uint4*)(dst + blk)) = *((device const uint4*)(Vb + src));
}

// Copy whole KV blocks (every layer region): copy-on-write of a shared
// partially filled block. pairs[i] = (src block, dst block). Grid
// (2 * bt * kv_ld / 8, pairs, layers).
kernel void kv_copy_blocks(
    device half* kv [[buffer(0)]],
    device const int2* pairs [[buffer(1)]],
    constant AttnParams& p [[buffer(2)]],
    constant ulong& region_halves [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  const ulong blk2 = 2 * ulong(p.bt) * p.kv_ld;
  if (ulong(gid.x) * 8 >= blk2) return;
  const int2 pr = pairs[gid.y];
  device half* base = kv + ulong(gid.z) * region_halves;
  *((device uint4*)(base + ulong(pr.y) * blk2 + gid.x * 8)) = *((device const uint4*)(base + ulong(pr.x) * blk2 + gid.x * 8));
}
