// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ---------------------------------------------------------------------------
// LFM2 (src/model/impl/lfm2.h): the gated short convolution of the conv
// layers (in_proj rows [B | C | x], y = C * conv(B * x), a causal depthwise
// conv of 3 taps whose window before a tree's first row comes from the parent
// slot's state, as the Qwen3.5 conv_prep does) and GQA attention with head
// dim 64: per-head q/k RMSNorms, RoPE over every dim (half split), no output
// gate. Semantics of transformers' models/lfm2 (Apache-2.0).
// ---------------------------------------------------------------------------

struct LfmConvParams {
  int D;            // channels
  int ldx;          // row stride of the in_proj rows (3 * D)
  ulong conv_base;  // float offset of this layer's conv-state region
  ulong conv_stride;  // floats per slot (3 * D: rows oldest..newest, the last 2 used)
};

// B * x of 4 channels of absolute row s, or of row -1 - s of a conv state.
static inline float4 lfm_bx(device const float* X, int ldx, int D, device const float* state, int s, int c0) {
  if (s >= 0) {
    device const float* xr = X + ulong(s) * ldx;
    return *((device const float4*)(xr + c0)) * *((device const float4*)(xr + 2 * D + c0));
  }
  return *((device const float4*)(state + ulong(-1 - s) * D + c0));
}

// Y[r] = C[r] * (w2 * bx[r] + w1 * bx[t - 1] + w0 * bx[t - 2]) with bx = B * x and
// the taps w tap-major (Wc[k * D + c]).
// src[r * 3 + j] is the absolute row holding tap t - (j + 1), or -1 - s for
// row s of the parent's conv state (row_slot[r]), which holds B * x values.
// Grid (D / 4, rows).
kernel void lfm_conv(
    device const float* X [[buffer(0)]],
    device const float* pool [[buffer(1)]],
    device const int* src [[buffer(2)]],
    device const float* Wc [[buffer(3)]],
    device half* Y [[buffer(4)]],
    constant LfmConvParams& p [[buffer(5)]],
    device const int* row_slot [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int c0 = int(gid.x) * 4;
  const int r = int(gid.y);
  if (c0 >= p.D) return;
  device const float* state = pool + p.conv_base + ulong(row_slot[r]) * p.conv_stride;
  // Taps tap-major ([3][D]: t - 2, t - 1, t).
  const float4 acc = lfm_bx(X, p.ldx, p.D, state, src[r * 3 + 1], c0) * *((device const float4*)(Wc + c0)) +
                     lfm_bx(X, p.ldx, p.D, state, src[r * 3 + 0], c0) * *((device const float4*)(Wc + p.D + c0)) +
                     lfm_bx(X, p.ldx, p.D, state, r, c0) * *((device const float4*)(Wc + 2 * p.D + c0));
  const float4 cg = *((device const float4*)(X + ulong(r) * p.ldx + p.D + c0));
  *((device half4*)(Y + ulong(r) * p.D + c0)) = half4(acc * cg);
}

// Conv state (B * x of the last 3 rows, oldest..newest) after the first T rows
// of a root starting at absolute row row0, from the parent slot's state (the
// two may be one slot: each thread reads its three values before writing).
// One job per (tree, saved slot): ConvSave of 30_deltanet. Grid (D, jobs).
kernel void lfm_conv_save(
    device const float* X [[buffer(0)]],
    device float* pool [[buffer(1)]],
    device const ConvSave* jobs [[buffer(2)]],
    constant LfmConvParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int c = int(gid.x);
  if (c >= p.D) return;
  const ConvSave job = jobs[gid.y];
  device const float* in = pool + p.conv_base + ulong(job.in_slot) * p.conv_stride;
  device float* out = pool + p.conv_base + ulong(job.out_slot) * p.conv_stride;
  float nv[3];
  UNROLL for (int j = 0; j < 3; ++j) {
    const int idx = job.T - 3 + j;
    if (idx >= 0) {
      device const float* xr = X + ulong(job.row0 + idx) * p.ldx;
      nv[j] = xr[c] * xr[2 * p.D + c];
    } else {
      nv[j] = in[ulong(3 + idx) * p.D + c];
    }
  }
  UNROLL for (int j = 0; j < 3; ++j) out[ulong(j) * p.D + c] = nv[j];
}

// ---------------------------------------------------------------------------
// Attention inputs: RMSNorm (weights q_norm / k_norm) and RoPE over the
// (d, d + HD/2) pairs of q and k, q scaled by HD^-1/2, v as it is; fp16 Q / K / V
// batch rows. One simdgroup per (row, head): heads [0, H) queries,
// [H, H + Hkv) keys, [H + Hkv, H + 2 Hkv) values.
// ---------------------------------------------------------------------------
struct LfmPrepParams {
  int H;
  int Hkv;
  int ld;     // projection row stride (floats): [q | k | v]
  int q_ld;   // Q row stride (H * HD)
  int kv_ld;  // K / V row stride (Hkv * HD)
  float eps;
  float scale;  // HD^-1/2
};

template <int HD>
kernel void lfm_attn_prep(
    device const float* X [[buffer(0)]],
    device const float* qn_w [[buffer(1)]],
    device const float* kn_w [[buffer(2)]],
    device const int4* pos [[buffer(3)]],
    device const float* inv_freq [[buffer(4)]],
    device half* Q [[buffer(5)]],
    device half* Kb [[buffer(6)]],
    device half* Vb [[buffer(7)]],
    constant LfmPrepParams& p [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NPT = HD / 32;  // consecutive elements per lane: lanes 0-15 hold the first half
  const int head = int(tgpig.x) * 4 + int(sg);
  const ulong row = tgpig.y;
  if (head >= p.H + 2 * p.Hkv) return;
  const int kind = head < p.H ? 0 : (head < p.H + p.Hkv ? 1 : 2);  // q, k, v
  const int h = kind == 0 ? head : (kind == 1 ? head - p.H : head - p.H - p.Hkv);
  const int col = kind == 0 ? 0 : (kind == 1 ? p.q_ld : p.q_ld + p.kv_ld);
  device const float* src = X + row * p.ld + col + h * HD + lane * NPT;
  float x[NPT];
  UNROLL for (int i = 0; i < NPT; ++i) x[i] = src[i];
  if (kind == 2) {
    device half* vo = Vb + row * p.kv_ld + h * HD + lane * NPT;
    UNROLL for (int i = 0; i < NPT; ++i) vo[i] = half(x[i]);
    return;
  }
  float ss = 0.0f;
  UNROLL for (int i = 0; i < NPT; ++i) ss += x[i] * x[i];
  ss = simd_sum(ss);
  const float inv = rsqrt(ss / float(HD) + p.eps);
  device const float* nw = (kind == 0 ? qn_w : kn_w) + lane * NPT;
  UNROLL for (int i = 0; i < NPT; ++i) x[i] = x[i] * inv * nw[i];
  const float position = float(pos[row].x);
  UNROLL for (int i = 0; i < NPT; ++i) {
    const float partner = simd_shuffle_xor(x[i], ushort(16));
    const float ang = position * inv_freq[(lane & 15) * NPT + i];
    const float c = precise::cos(ang), s = precise::sin(ang);
    x[i] = lane < 16 ? x[i] * c - partner * s : x[i] * c + partner * s;
  }
  const float sc = kind == 0 ? p.scale : 1.0f;
  device half* o = (kind == 0 ? Q + row * p.q_ld : Kb + row * p.kv_ld) + h * HD + lane * NPT;
  UNROLL for (int i = 0; i < NPT; ++i) o[i] = half(x[i] * sc);
}

template [[host_name("lfm_attn_prep64")]] kernel void lfm_attn_prep<64>(
    device const float*, device const float*, device const float*, device const int4*, device const float*,
    device half*, device half*, device half*, constant LfmPrepParams&, uint2, uint, uint);

// ---------------------------------------------------------------------------
// Forest attention with heads of HD <= 128 dims (g4_attention's structure
// with one simdgroup per query head: its score fragments are the whole
// product, so no partials go through threadgroup memory): a threadgroup is the
// G query heads of a KV head x one tile of 8 consecutive query rows of one
// tree, over the cached keys [k0, k1) of the paged pool and, for the tile
// with `last` set, the batch keys of its tree under the tree mask. Q is
// pre-scaled. Writes O (fp16) or g4_attention's partial record, merged by
// g4_attention_combine.
// ---------------------------------------------------------------------------
template <int HD, int G>
kernel void lfm_attention(
    device const half* Q [[buffer(0)]],
    device const half* Kb [[buffer(1)]],
    device const half* Vb [[buffer(2)]],
    device const half* kv [[buffer(3)]],
    device const int* blocks [[buffer(4)]],
    device const int4* desc [[buffer(5)]],
    device half* O [[buffer(7)]],
    constant G4AttnParams& p [[buffer(8)]],
    device const G4Tile* tiles [[buffer(9)]],
    device const AttnTree* trees [[buffer(10)]],
    device float* parts [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NT = 32 * G;
  constexpr int BK = 32;  // keys per block: 2 x BK x (HD + 8) halves of threadgroup memory
  constexpr int KF = BK / 8;
  constexpr int DF = HD / 8;  // 8-dim fragments of a head
  constexpr int LD = HD + 8;
  constexpr int C8 = HD / 8;  // uint4 columns of a key row
  threadgroup half Kt[BK * LD];
  threadgroup half Vt[BK * LD];

  const int kvh = int(tgid.x);
  const G4Tile tile = tiles[tgid.y];
  const AttnTree tree = trees[tile.tree];
  const int h = kvh * G + int(sg);
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  const int end = tree.base + tree.M;
  const int row = tile.r0 + fm;
  const bool row_ok = row < end;
  const int rr = row - tree.base;
  const int4 d = desc[min(row, end - 1)];
  const int r_last = min(tile.r0 + 7, end - 1) - tree.base;

  simdgroup_matrix<half, 8, 8> qf[DF];
  {
    device const half* qp = Q + ulong(min(row, end - 1)) * p.q_ld + h * HD + fn;
    UNROLL for (int ks = 0; ks < DF; ++ks) {
      const half2 v = *((device const half2*)(qp + ks * 8));
      qf[ks].thread_elements()[0] = v.x;
      qf[ks].thread_elements()[1] = v.y;
    }
  }
  simdgroup_matrix<float, 8, 8> of[DF];
  UNROLL for (int i = 0; i < DF; ++i) of[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
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
  const int n_batch_blocks = tile.last ? (r_last + BK) / BK : 0;
  for (int blk = 0; blk < n_slot_blocks + n_batch_blocks; ++blk) {
    const bool in_slot = blk < n_slot_blocks;
    const int j0 = in_slot ? tile.k0 + blk * BK : (blk - n_slot_blocks) * BK;
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
          const ulong b = ulong(blocks[tree.table + j / p.bt]);
          src = kvl + (b * 2 + ulong(which)) * blk_halves + ulong(j % p.bt) * p.kv_ld + kvh * HD + col;
        } else {
          src = (which ? Vb : Kb) + ulong(tree.base + j) * p.kv_ld + kvh * HD + col;
        }
        v = *((device const uint4*)src);
      }
      *((threadgroup uint4*)((which ? Vt : Kt) + key * LD + col)) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> sf[KF];
    UNROLL for (int kf = 0; kf < KF; ++kf) sf[kf] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    UNROLL for (int ks = 0; ks < DF; ++ks) {
      UNROLL for (int kf = 0; kf < KF; ++kf) {
        simdgroup_matrix<half, 8, 8> kb;
        const threadgroup half* src = Kt + (kf * 8 + fn) * LD + ks * 8 + fm;
        kb.thread_elements()[0] = src[0];
        kb.thread_elements()[1] = src[LD];
        simdgroup_multiply_accumulate(sf[kf], qf[ks], kb, sf[kf]);
      }
    }
    float sv[2 * KF];
    UNROLL for (int kf = 0; kf < KF; ++kf) {
      UNROLL for (int e = 0; e < 2; ++e) {
        const int kc = kf * 8 + fn + e;
        const int j = j0 + kc;
        bool vis = row_ok && j < j_end;
        if (!in_slot) vis = vis && ((j < d.z) || (j >= d.x && j <= rr));
        sv[kf * 2 + e] = vis ? sf[kf].thread_elements()[e] : -INFINITY;
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
    UNROLL for (int df = 0; df < DF; ++df) {
      of[df].thread_elements()[0] *= corr;
      of[df].thread_elements()[1] *= corr;
    }
    UNROLL for (int kf = 0; kf < KF; ++kf) {
      UNROLL for (int df = 0; df < DF; ++df) {
        simdgroup_matrix<half, 8, 8> vb;
        const half2 v2 = *((const threadgroup half2*)(Vt + (kf * 8 + fm) * LD + df * 8 + fn));
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
    device float* rec = parts + ((ulong(tile.part) * p.Hkv + kvh) * 8 * G + ulong(fm) * G + sg) * (HD + 2);
    UNROLL for (int df = 0; df < DF; ++df) {
      rec[df * 8 + fn] = of[df].thread_elements()[0];
      rec[df * 8 + fn + 1] = of[df].thread_elements()[1];
    }
    if ((lane & 9) == 0) {
      rec[HD] = m_run;
      rec[HD + 1] = l_run;
    }
    return;
  }
  const float inv_l = l_run > 0.0f ? 1.0f / l_run : 0.0f;
  device half* op = O + ulong(row) * p.q_ld + h * HD + fn;
  UNROLL for (int df = 0; df < DF; ++df)
    *((device half2*)(op + df * 8)) =
        half2(half(of[df].thread_elements()[0] * inv_l), half(of[df].thread_elements()[1] * inv_l));
}

#define INSTANTIATE_LFMA(G)                                                                             \
  template [[host_name("lfm_attention64_g" #G)]] kernel void lfm_attention<64, G>(                      \
      device const half*, device const half*, device const half*, device const half*, device const int*, \
      device const int4*, device half*, constant G4AttnParams&, device const G4Tile*, device const AttnTree*, \
      device float*, uint2, uint, uint, uint);
INSTANTIATE_LFMA(1)
INSTANTIATE_LFMA(2)
INSTANTIATE_LFMA(4)
INSTANTIATE_LFMA(8)

template [[host_name("lfm_attention_combine64")]] kernel void g4_attention_combine<64>(
    device const float*, device half*, constant G4AttnParams&, device const AttnCombine*, uint3);
