// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ===========================================================================
// Vision tower (Qwen3-VL / Qwen3.5 ViT): dense fp16 weights, LayerNorm,
// 2D rotary attention over each image's patches, 2x2 patch merger.
// ===========================================================================

// LayerNorm (weight and bias): fp32 rows -> fp16 rows. One threadgroup/row.
kernel void layernorm(
    device const float* X [[buffer(0)]],
    device const float* Wt [[buffer(1)]],
    device const float* Bs [[buffer(2)]],
    device half* Y [[buffer(3)]],
    constant int& D [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float partial[32];
  device const float4* x = (device const float4*)(X + ulong(row) * D);
  const int D4 = D / 4;
  const int nsg = (ntg + 31) / 32;
  float s = 0.0f;
  for (int i = tid; i < D4; i += ntg) {
    const float4 v = x[i];
    s += v.x + v.y + v.z + v.w;
  }
  s = simd_sum(s);
  if (lane == 0) partial[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float t = (int(lane) < nsg) ? partial[lane] : 0.0f;
    t = simd_sum(t);
    if (lane == 0) partial[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mean = partial[0] / float(D);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float q = 0.0f;
  for (int i = tid; i < D4; i += ntg) {
    const float4 v = x[i] - mean;
    q += dot(v, v);
  }
  q = simd_sum(q);
  if (lane == 0) partial[sg] = q;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float t = (int(lane) < nsg) ? partial[lane] : 0.0f;
    t = simd_sum(t);
    if (lane == 0) partial[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(partial[0] / float(D) + eps);
  device const float4* w = (device const float4*)Wt;
  device const float4* b = (device const float4*)Bs;
  device half4* y = (device half4*)(Y + ulong(row) * D);
  for (int i = tid; i < D4; i += ntg) y[i] = half4((x[i] - mean) * inv * w[i] + b[i]);
}

// Learned position embedding, bilinearly interpolated to the patch grid:
// X[r] += sum_i wts[r][i] * P[idx[r][i]].
kernel void vit_pos_add(
    device float* X [[buffer(0)]],
    device const float* P [[buffer(1)]],
    device const int4* idx [[buffer(2)]],
    device const float4* wts [[buffer(3)]],
    constant int& D [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int d4 = gid.x;
  const int r = gid.y;
  if (d4 * 4 >= D) return;
  const int4 ix = idx[r];
  const float4 w = wts[r];
  device float4* x = (device float4*)(X + ulong(r) * D) + d4;
  float4 acc = w.x * *((device const float4*)(P + ulong(ix.x) * D) + d4);
  acc += w.y * *((device const float4*)(P + ulong(ix.y) * D) + d4);
  acc += w.z * *((device const float4*)(P + ulong(ix.z) * D) + d4);
  acc += w.w * *((device const float4*)(P + ulong(ix.w) * D) + d4);
  *x += acc;
}

// Split the fused qkv projection into head-major fp16 Q (scaled), K, V with
// the 2D rotary: dims [0, HD/4) rotate by the patch row, [HD/4, HD/2) by the
// patch column, each paired with the dim HD/2 further (rotate_half).
struct VitPrepParams {
  int N;
  int H;
  int HD;
  float scale;
};

kernel void vit_qkv_prep(
    device const float* QKV [[buffer(0)]],
    device const int2* hw [[buffer(1)]],
    device const float* inv_freq [[buffer(2)]],
    device half* Q [[buffer(3)]],
    device half* K [[buffer(4)]],
    device half* V [[buffer(5)]],
    constant VitPrepParams& p [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int d = gid.x;
  const int h = gid.y;
  const int r = gid.z;
  const int half_d = p.HD / 2;
  if (d >= half_d || r >= p.N) return;
  const int quarter = p.HD / 4;
  const int2 pr = hw[r];
  const float ang = d < quarter ? float(pr.x) * inv_freq[d] : float(pr.y) * inv_freq[d - quarter];
  const float c = precise::cos(ang);
  const float s = precise::sin(ang);
  const int D = p.H * p.HD;
  device const float* src = QKV + ulong(r) * 3 * D + h * p.HD;
  const ulong o = (ulong(h) * p.N + r) * p.HD + d;
  const float q1 = src[d], q2 = src[d + half_d];
  const float k1 = src[D + d], k2 = src[D + d + half_d];
  Q[o] = half((q1 * c - q2 * s) * p.scale);
  Q[o + half_d] = half((q2 * c + q1 * s) * p.scale);
  K[o] = half(k1 * c - k2 * s);
  K[o + half_d] = half(k2 * c + k1 * s);
  V[o] = half(src[2 * D + d]);
  V[o + half_d] = half(src[2 * D + d + half_d]);
}

// Bidirectional attention within each image (seg[r] = [start, end) of row r's
// image). Q/K/V head-major [H][N][HD] fp16 (Q pre-scaled); O [N][H*HD] fp16.
// Threadgroup = one head x 32 query rows (4 simdgroups x 8 rows); keys stream
// through threadgroup memory 32 at a time with an online softmax.
struct VitAttnParams {
  int N;
  int H;
};

template <int HD>
kernel void vit_attention(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const int2* seg [[buffer(3)]],
    device half* O [[buffer(4)]],
    constant VitAttnParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BK = 32;
  constexpr int LD = HD + 8;
  constexpr int NT = 128;
  constexpr int KS = HD / 8;
  threadgroup half Kt[BK * LD];
  threadgroup half Vt[BK * LD];
  const int h = tgid.x;
  const int first = tgid.y * 32;
  const int last = min(first + 32, p.N) - 1;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  const int row = first + sg * 8 + fm;
  const bool row_ok = row < p.N;
  const int2 sr = seg[min(row, p.N - 1)];
  const int kmin = seg[first].x;
  const int kmax = seg[last].y;
  device const half* Qh = Q + ulong(h) * p.N * HD;
  device const half* Kh = K + ulong(h) * p.N * HD;
  device const half* Vh = V + ulong(h) * p.N * HD;

  simdgroup_matrix<half, 8, 8> qf[KS];
  {
    device const half* qp = Qh + ulong(min(row, p.N - 1)) * HD + fn;
    UNROLL for (int ks = 0; ks < KS; ++ks) {
      const half2 v = *((device const half2*)(qp + ks * 8));
      qf[ks].thread_elements()[0] = v.x;
      qf[ks].thread_elements()[1] = v.y;
    }
  }
  simdgroup_matrix<float, 8, 8> of[KS];
  UNROLL for (int i = 0; i < KS; ++i) of[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  float m_run = -INFINITY;
  float l_run = 0.0f;

  for (int j0 = kmin; j0 < kmax; j0 += BK) {
    for (int c = tid; c < BK * KS * 2; c += NT) {
      const int which = c / (BK * KS);
      const int cc = c % (BK * KS);
      const int key = cc / KS;
      const int col = (cc % KS) * 8;
      const int j = j0 + key;
      uint4 v = uint4(0);
      if (j < kmax) v = *((device const uint4*)((which ? Vh : Kh) + ulong(j) * HD + col));
      *((threadgroup uint4*)((which ? Vt : Kt) + key * LD + col)) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> sf[4];
    UNROLL for (int kf = 0; kf < 4; ++kf) sf[kf] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    UNROLL for (int ks = 0; ks < KS; ++ks) {
      UNROLL for (int kf = 0; kf < 4; ++kf) {
        simdgroup_matrix<half, 8, 8> kb;
        const threadgroup half* src = Kt + (kf * 8 + fn) * LD + ks * 8 + fm;
        kb.thread_elements()[0] = src[0];
        kb.thread_elements()[1] = src[LD];
        simdgroup_multiply_accumulate(sf[kf], qf[ks], kb, sf[kf]);
      }
    }
    float sv[8];
    UNROLL for (int kf = 0; kf < 4; ++kf) {
      UNROLL for (int e = 0; e < 2; ++e) {
        const int j = j0 + kf * 8 + fn + e;
        const bool vis = row_ok && j >= sr.x && j < sr.y;
        sv[kf * 2 + e] = vis ? sf[kf].thread_elements()[e] : -INFINITY;
      }
    }
    float mx = sv[0];
    UNROLL for (int i = 1; i < 8; ++i) mx = max(mx, sv[i]);
    mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
    const float m_new = max(m_run, mx);
    const float corr = (m_new == -INFINITY) ? 1.0f : exp(m_run - m_new);
    float ps = 0.0f;
    simdgroup_matrix<half, 8, 8> pf[4];
    UNROLL for (int kf = 0; kf < 4; ++kf) {
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
    UNROLL for (int df = 0; df < KS; ++df) {
      of[df].thread_elements()[0] *= corr;
      of[df].thread_elements()[1] *= corr;
    }
    UNROLL for (int kf = 0; kf < 4; ++kf) {
      UNROLL for (int df = 0; df < KS; ++df) {
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
  const float inv_l = l_run > 0.0f ? 1.0f / l_run : 0.0f;
  device half* op = O + ulong(row) * (p.H * HD) + h * HD + fn;
  UNROLL for (int df = 0; df < KS; ++df) {
    const float o0 = of[df].thread_elements()[0] * inv_l;
    const float o1 = of[df].thread_elements()[1] * inv_l;
    *((device half2*)(op + df * 8)) = half2(half(o0), half(o1));
  }
}

// vit_attention with 64 query rows per threadgroup: 8 simdgroups share
// each 32-key K/V tile, so every tile load and barrier feeds twice the MMAs,
// and the K^T / V fragments come from threadgroup memory through
// simdgroup_load. Same math and accumulation order per row as vit_attention.
template <int HD, int BK = 32, int KT = 0>
kernel void vit_attention2(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const int2* seg [[buffer(3)]],
    device half* O [[buffer(4)]],
    constant VitAttnParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int LD = HD + 8;
  constexpr int NT = 256;
  constexpr int KF = BK / 8;  // key fragments per block
  constexpr int KS = HD / 8;
  // KT: K stored transposed ([dim][key], row stride BK + 8) so the K^T
  // fragments load without a transposing simdgroup_load.
  constexpr int LDT = BK + 8;
  threadgroup half Kt[KT ? HD * LDT : BK * LD];
  threadgroup half Vt[BK * LD];
  const int h = tgid.x;
  const int first = tgid.y * 64;
  const int last = min(first + 64, p.N) - 1;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  const int row = first + sg * 8 + fm;
  const bool row_ok = row < p.N;
  const int2 sr = seg[min(row, p.N - 1)];
  const int kmin = seg[first].x;
  const int kmax = seg[last].y;
  device const half* Qh = Q + ulong(h) * p.N * HD;
  device const half* Kh = K + ulong(h) * p.N * HD;
  device const half* Vh = V + ulong(h) * p.N * HD;

  simdgroup_matrix<half, 8, 8> qf[KS];
  {
    device const half* qp = Qh + ulong(min(row, p.N - 1)) * HD + fn;
    UNROLL for (int ks = 0; ks < KS; ++ks) {
      const half2 v = *((device const half2*)(qp + ks * 8));
      qf[ks].thread_elements()[0] = v.x;
      qf[ks].thread_elements()[1] = v.y;
    }
  }
  simdgroup_matrix<float, 8, 8> of[KS];
  UNROLL for (int i = 0; i < KS; ++i) of[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  float m_run = -INFINITY;
  float l_run = 0.0f;

  for (int j0 = kmin; j0 < kmax; j0 += BK) {
    for (int c = tid; c < BK * KS * 2; c += NT) {
      const int which = c / (BK * KS);
      const int cc = c % (BK * KS);
      const int key = cc / KS;
      const int col = (cc % KS) * 8;
      const int j = j0 + key;
      uint4 v = uint4(0);
      if (j < kmax) v = *((device const uint4*)((which ? Vh : Kh) + ulong(j) * HD + col));
      if (KT && !which) {
        const half2 h0 = as_type<half2>(v.x), h1 = as_type<half2>(v.y), h2 = as_type<half2>(v.z), h3 = as_type<half2>(v.w);
        const half hv[8] = {h0.x, h0.y, h1.x, h1.y, h2.x, h2.y, h3.x, h3.y};
        UNROLL for (int e = 0; e < 8; ++e) Kt[(col + e) * LDT + key] = hv[e];
      } else {
        *((threadgroup uint4*)((which ? Vt : Kt) + key * LD + col)) = v;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> sf[KF];
    UNROLL for (int kf = 0; kf < KF; ++kf) sf[kf] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    UNROLL for (int ks = 0; ks < KS; ++ks) {
      UNROLL for (int kf = 0; kf < KF; ++kf) {
        simdgroup_matrix<half, 8, 8> kb;  // K^T block: [dim ks*8.., key kf*8..]
        if (KT) simdgroup_load(kb, Kt + ks * 8 * LDT + kf * 8, LDT);
        else simdgroup_load(kb, Kt + kf * 8 * LD + ks * 8, LD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(sf[kf], qf[ks], kb, sf[kf]);
      }
    }
    float sv[2 * KF];
    UNROLL for (int kf = 0; kf < KF; ++kf) {
      UNROLL for (int e = 0; e < 2; ++e) {
        const int j = j0 + kf * 8 + fn + e;
        const bool vis = row_ok && j >= sr.x && j < sr.y;
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
    UNROLL for (int df = 0; df < KS; ++df) {
      of[df].thread_elements()[0] *= corr;
      of[df].thread_elements()[1] *= corr;
    }
    UNROLL for (int kf = 0; kf < KF; ++kf) {
      UNROLL for (int df = 0; df < KS; ++df) {
        simdgroup_matrix<half, 8, 8> vb;  // V block: [key kf*8.., dim df*8..]
        simdgroup_load(vb, Vt + kf * 8 * LD + df * 8, LD);
        simdgroup_multiply_accumulate(of[df], pf[kf], vb, of[df]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!row_ok) return;
  const float inv_l = l_run > 0.0f ? 1.0f / l_run : 0.0f;
  device half* op = O + ulong(row) * (p.H * HD) + h * HD + fn;
  UNROLL for (int df = 0; df < KS; ++df) {
    const float o0 = of[df].thread_elements()[0] * inv_l;
    const float o1 = of[df].thread_elements()[1] * inv_l;
    *((device half2*)(op + df * 8)) = half2(half(o0), half(o1));
  }
}

#define INSTANTIATE_VIT_ATTN2(HD)                                                        \
  template [[host_name("vit_attention2_" #HD)]] kernel void vit_attention2<HD, 32>(     \
      device const half*, device const half*, device const half*,                       \
      device const int2*, device half*, constant VitAttnParams&, uint2, uint,           \
      uint, uint);                                                                      \
  template [[host_name("vit_attention3_" #HD)]] kernel void vit_attention2<HD, 64>(     \
      device const half*, device const half*, device const half*,                       \
      device const int2*, device half*, constant VitAttnParams&, uint2, uint,           \
      uint, uint);                                                                      \
  template [[host_name("vit_attention4_" #HD)]] kernel void vit_attention2<HD, 64, 1>(  \
      device const half*, device const half*, device const half*,                       \
      device const int2*, device half*, constant VitAttnParams&, uint2, uint,           \
      uint, uint);

INSTANTIATE_VIT_ATTN2(64)
INSTANTIATE_VIT_ATTN2(72)
INSTANTIATE_VIT_ATTN2(80)

// vit_attention2 with RG 8-row query groups per simdgroup (64 * RG rows per
// threadgroup) and, with PF, the next K/V block's device loads issued into
// registers before this block's MMAs. With RG = 2 every K^T and V
// fragment read from threadgroup memory feeds two MMAs and a K/V block is
// loaded once per 128 queries. Per row: vit_attention2's math and
// accumulation order, over the same key blocks when the threadgroup's rows
// share one segment start (bit-identical output then).
template <int HD, int RG, int PF>
kernel void vit_attention5(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const int2* seg [[buffer(3)]],
    device half* O [[buffer(4)]],
    constant VitAttnParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BK = 32;
  constexpr int LD = HD + 8;
  constexpr int NT = 256;
  constexpr int KF = BK / 8;
  constexpr int KS = HD / 8;
  constexpr int ROWS = 64 * RG;
  constexpr int NV = BK * KS * 2;  // uint4 loads of one K + V block
  constexpr int NL = (NV + NT - 1) / NT;
  threadgroup half Kt[BK * LD];
  threadgroup half Vt[BK * LD];
  const int h = tgid.x;
  const int first = tgid.y * ROWS;
  const int last = min(first + ROWS, p.N) - 1;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  int row[RG];
  bool row_ok[RG];
  int2 sr[RG];
  UNROLL for (int g = 0; g < RG; ++g) {
    row[g] = first + (int(sg) * RG + g) * 8 + fm;
    row_ok[g] = row[g] < p.N;
    sr[g] = seg[min(row[g], p.N - 1)];
  }
  const int kmin = seg[first].x;
  const int kmax = seg[last].y;
  device const half* Qh = Q + ulong(h) * p.N * HD;
  device const half* Kh = K + ulong(h) * p.N * HD;
  device const half* Vh = V + ulong(h) * p.N * HD;

  simdgroup_matrix<half, 8, 8> qf[RG][KS];
  UNROLL for (int g = 0; g < RG; ++g) {
    device const half* qp = Qh + ulong(min(row[g], p.N - 1)) * HD + fn;
    UNROLL for (int ks = 0; ks < KS; ++ks) {
      const half2 v = *((device const half2*)(qp + ks * 8));
      qf[g][ks].thread_elements()[0] = v.x;
      qf[g][ks].thread_elements()[1] = v.y;
    }
  }
  simdgroup_matrix<float, 8, 8> of[RG][KS];
  float m_run[RG], l_run[RG];
  UNROLL for (int g = 0; g < RG; ++g) {
    UNROLL for (int i = 0; i < KS; ++i) of[g][i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    m_run[g] = -INFINITY;
    l_run[g] = 0.0f;
  }
  // This thread's share of a K/V block (NL uint4), read a block ahead with PF.
  uint4 pre[NL];
  auto fetch = [&](int j0) {
    UNROLL for (int i = 0; i < NL; ++i) {
      const int c = int(tid) + i * NT;
      pre[i] = uint4(0);
      if (c < NV) {
        const int which = c / (BK * KS), cc = c % (BK * KS);
        const int j = j0 + cc / KS;
        if (j < kmax) pre[i] = *((device const uint4*)((which ? Vh : Kh) + ulong(j) * HD + (cc % KS) * 8));
      }
    }
  };
  if (PF && kmin < kmax) fetch(kmin);
  for (int j0 = kmin; j0 < kmax; j0 += BK) {
    if (!PF) fetch(j0);
    UNROLL for (int i = 0; i < NL; ++i) {
      const int c = int(tid) + i * NT;
      if (c < NV) {
        const int which = c / (BK * KS), cc = c % (BK * KS);
        *((threadgroup uint4*)((which ? Vt : Kt) + (cc / KS) * LD + (cc % KS) * 8)) = pre[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (PF && j0 + BK < kmax) fetch(j0 + BK);
    simdgroup_matrix<float, 8, 8> sf[RG][KF];
    UNROLL for (int g = 0; g < RG; ++g)
      UNROLL for (int kf = 0; kf < KF; ++kf) sf[g][kf] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    UNROLL for (int ks = 0; ks < KS; ++ks) {
      UNROLL for (int kf = 0; kf < KF; ++kf) {
        simdgroup_matrix<half, 8, 8> kb;  // K^T block: [dim ks*8.., key kf*8..]
        simdgroup_load(kb, Kt + kf * 8 * LD + ks * 8, LD, ulong2(0, 0), true);
        UNROLL for (int g = 0; g < RG; ++g) simdgroup_multiply_accumulate(sf[g][kf], qf[g][ks], kb, sf[g][kf]);
      }
    }
    simdgroup_matrix<half, 8, 8> pf[RG][KF];
    UNROLL for (int g = 0; g < RG; ++g) {
      float sv[2 * KF];
      UNROLL for (int kf = 0; kf < KF; ++kf) {
        UNROLL for (int e = 0; e < 2; ++e) {
          const int j = j0 + kf * 8 + fn + e;
          const bool vis = row_ok[g] && j >= sr[g].x && j < sr[g].y;
          sv[kf * 2 + e] = vis ? sf[g][kf].thread_elements()[e] : -INFINITY;
        }
      }
      float mx = sv[0];
      UNROLL for (int i = 1; i < 2 * KF; ++i) mx = max(mx, sv[i]);
      mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
      mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
      const float m_new = max(m_run[g], mx);
      const float corr = (m_new == -INFINITY) ? 1.0f : exp(m_run[g] - m_new);
      float ps = 0.0f;
      UNROLL for (int kf = 0; kf < KF; ++kf) {
        UNROLL for (int e = 0; e < 2; ++e) {
          const float pv = (m_new == -INFINITY) ? 0.0f : exp(sv[kf * 2 + e] - m_new);
          ps += pv;
          pf[g][kf].thread_elements()[e] = half(pv);
        }
      }
      ps += simd_shuffle_xor(ps, ushort(1));
      ps += simd_shuffle_xor(ps, ushort(8));
      l_run[g] = l_run[g] * corr + ps;
      m_run[g] = m_new;
      UNROLL for (int df = 0; df < KS; ++df) {
        of[g][df].thread_elements()[0] *= corr;
        of[g][df].thread_elements()[1] *= corr;
      }
    }
    UNROLL for (int kf = 0; kf < KF; ++kf) {
      UNROLL for (int df = 0; df < KS; ++df) {
        simdgroup_matrix<half, 8, 8> vb;  // V block: [key kf*8.., dim df*8..]
        simdgroup_load(vb, Vt + kf * 8 * LD + df * 8, LD);
        UNROLL for (int g = 0; g < RG; ++g) simdgroup_multiply_accumulate(of[g][df], pf[g][kf], vb, of[g][df]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  UNROLL for (int g = 0; g < RG; ++g) {
    if (!row_ok[g]) continue;
    const float inv_l = l_run[g] > 0.0f ? 1.0f / l_run[g] : 0.0f;
    device half* op = O + ulong(row[g]) * (p.H * HD) + h * HD + fn;
    UNROLL for (int df = 0; df < KS; ++df) {
      const float o0 = of[g][df].thread_elements()[0] * inv_l;
      const float o1 = of[g][df].thread_elements()[1] * inv_l;
      *((device half2*)(op + df * 8)) = half2(half(o0), half(o1));
    }
  }
}

#define INSTANTIATE_VIT_ATTN5(HD)                                                          \
  template [[host_name("vit_attention5_" #HD)]] kernel void vit_attention5<HD, 2, 1>(     \
      device const half*, device const half*, device const half*,                         \
      device const int2*, device half*, constant VitAttnParams&, uint2, uint,             \
      uint, uint);                                                                        \
  template [[host_name("vit_attention6_" #HD)]] kernel void vit_attention5<HD, 2, 0>(     \
      device const half*, device const half*, device const half*,                         \
      device const int2*, device half*, constant VitAttnParams&, uint2, uint,             \
      uint, uint);                                                                        \
  template [[host_name("vit_attention7_" #HD)]] kernel void vit_attention5<HD, 1, 1>(     \
      device const half*, device const half*, device const half*,                         \
      device const int2*, device half*, constant VitAttnParams&, uint2, uint,             \
      uint, uint);

INSTANTIATE_VIT_ATTN5(64)
INSTANTIATE_VIT_ATTN5(72)
INSTANTIATE_VIT_ATTN5(80)

#define INSTANTIATE_VIT_ATTN(HD)                                                   \
  template [[host_name("vit_attention_" #HD)]] kernel void vit_attention<HD>(     \
      device const half*, device const half*, device const half*,                 \
      device const int2*, device half*, constant VitAttnParams&, uint2, uint,     \
      uint, uint);

INSTANTIATE_VIT_ATTN(64)
INSTANTIATE_VIT_ATTN(72)
INSTANTIATE_VIT_ATTN(80)
