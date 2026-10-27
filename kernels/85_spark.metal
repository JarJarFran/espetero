// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ---------------------------------------------------------------------------
// Spark-X2.5 / X2.6 (spark2_5; src/model/impl/gemma.h with Config::spark):
// the Gemma 4 layer loop (80_gemma's attention kernels for both layer kinds,
// head dim 256) with its own attention inputs (no q/k/v norms, NeoX RoPE over
// the first ROT dims, the 1/sqrt(HD) scale folded into q), a sigmoid gate per
// query head on the attention output, plain pre-norm residual updates and the
// exact-erf GeGLU of separate gate and up projections. Semantics of mlx-lm's
// models/spark2_5.py (MIT).
// ---------------------------------------------------------------------------

// Attention inputs: q, k and v of a row from the fused projection (fp32,
// [q; k; v; gate] per row) as fp16 batch rows; q and k rotated over dims
// [0, ROT) in (d, d + ROT / 2) pairs at inv_freq[d] (nn.RoPE, traditional
// false), q times p.scale. One simdgroup per (row, head): heads [0, H)
// queries, [H, H + Hkv) keys, [H + Hkv, H + 2 Hkv) values.
struct SpPrepParams {
  int H;
  int Hkv;
  int ld;     // projection row stride (floats)
  int k_off;  // column of the keys
  int v_off;  // column of the values
  int q_ld;   // Q row stride (H * HD)
  int kv_ld;  // K / V row stride (Hkv * HD)
  float scale;
  int rope_off;  // this layer kind's first frequency in the sp_rope_table rows
  int rope_ld;   // frequencies per row
};

// (cos, sin) of position * inv_freq for every row and frequency of both layer
// kinds (n[0] sliding ones from fs, then fg up to n[1]): computed once per
// pass instead of in every layer's sp_attn_prep. Grid (n[1], rows).
kernel void sp_rope_table(
    device const int4* pos [[buffer(0)]],
    device const float* fs [[buffer(1)]],
    device const float* fg [[buffer(2)]],
    device float2* cs [[buffer(3)]],
    constant int2& n [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int f = int(gid.x);
  if (f >= n.y) return;
  const float ang = float(pos[gid.y].x) * (f < n.x ? fs[f] : fg[f - n.x]);
  cs[ulong(gid.y) * ulong(n.y) + f] = float2(precise::cos(ang), precise::sin(ang));
}

template <int ROT>
kernel void sp_attn_prep(
    device const float* X [[buffer(0)]],
    device const float2* rope [[buffer(1)]],
    device half* Q [[buffer(3)]],
    device half* Kb [[buffer(4)]],
    device half* Vb [[buffer(5)]],
    constant SpPrepParams& p [[buffer(6)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int HD = 256;
  constexpr int NPT = HD / 32;    // consecutive elements per lane
  constexpr int RL = ROT / NPT;   // lanes holding rotated dims
  constexpr int HL = RL / 2;      // lanes of the first half (the partner is lane ^ HL)
  const int head = int(tgpig.x) * 4 + int(sg);
  const ulong row = tgpig.y;
  if (head >= p.H + 2 * p.Hkv) return;  // whole simdgroups
  const int kind = head < p.H ? 0 : (head < p.H + p.Hkv ? 1 : 2);  // q, k, v
  const int h = kind == 0 ? head : (kind == 1 ? head - p.H : head - p.H - p.Hkv);
  device const float* src = X + row * p.ld + (kind == 0 ? 0 : (kind == 1 ? p.k_off : p.v_off)) + h * HD + lane * NPT;
  float x[NPT];
  UNROLL for (int i = 0; i < NPT; ++i) x[i] = src[i];
  if (kind == 2) {
    device half* vo = Vb + row * p.kv_ld + h * HD + lane * NPT;
    UNROLL for (int i = 0; i < NPT; ++i) vo[i] = half(x[i]);
    return;
  }
  device const float2* cs = rope + row * ulong(p.rope_ld) + p.rope_off;
  UNROLL for (int i = 0; i < NPT; ++i) {
    const float partner = simd_shuffle_xor(x[i], ushort(HL));
    if (int(lane) < RL) {
      const float2 r = cs[(int(lane) & (HL - 1)) * NPT + i];
      x[i] = int(lane) < HL ? x[i] * r.x - partner * r.y : x[i] * r.x + partner * r.y;
    }
  }
  const float sc = kind == 0 ? p.scale : 1.0f;
  device half* o = (kind == 0 ? Q + row * p.q_ld : Kb + row * p.kv_ld) + h * HD + lane * NPT;
  UNROLL for (int i = 0; i < NPT; ++i) o[i] = half(x[i] * sc);
}

#define INSTANTIATE_SP_PREP(ROT)                                                                              \
  template [[host_name("sp_attn_prep" #ROT)]] kernel void sp_attn_prep<ROT>(                                \
      device const float*, device const float2*, device half*, device half*, device half*, \
      constant SpPrepParams&, uint2, uint, uint);
INSTANTIATE_SP_PREP(16)
INSTANTIATE_SP_PREP(32)
INSTANTIATE_SP_PREP(64)
INSTANTIATE_SP_PREP(128)
INSTANTIATE_SP_PREP(256)

// The head-wise output gate: O[row, h, :] *= sigmoid(x[row] . Wg[h]) with x
// the layer's normed input (fp16 [rows, D]) and Wg = g_proj (fp16 [H, D]).
// One threadgroup (256 threads) per (head, row); D % 4 == 0, hd <= 1024.
struct SpGateParams {
  int q_ld;  // O row stride (halves)
  int D;
  int hd;
  int pad;
};

kernel void sp_head_gate(
    device half* O [[buffer(0)]],
    device const half* X [[buffer(1)]],
    device const half* Wg [[buffer(2)]],
    constant SpGateParams& p [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tpg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  const uint ntg = tpg.x;
  const uint h = tg.x;
  const ulong row = tg.y;
  device const half4* x = (device const half4*)(X + row * ulong(p.D));
  device const half4* w = (device const half4*)(Wg + ulong(h) * ulong(p.D));
  float acc = 0.0f;
  for (int i = int(tid); i < p.D / 4; i += int(ntg)) acc += dot(float4(x[i]), float4(w[i]));
  const float g = 1.0f / (1.0f + exp(-g4_tg_sum(acc, part, sg, lane, ntg)));
  device half* o = O + row * ulong(p.q_ld) + ulong(h) * ulong(p.hd);
  for (int i = int(tid); i < p.hd; i += int(ntg)) o[i] = half(float(o[i]) * g);
}

// Residual update of a pre-norm layer: h += T, then (nout = 1)
// y = rmsnorm(h) * w (the next sublayer's input). One threadgroup per row.
kernel void sp_add_norm(
    device const float* T [[buffer(0)]],
    device float* H [[buffer(1)]],
    device const float* w [[buffer(2)]],
    device half* y [[buffer(3)]],
    constant G4NormParams& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  const int D4 = p.D / 4;
  device const float4* t = (device const float4*)(T + ulong(row) * p.D);
  device float4* h = (device float4*)(H + ulong(row) * p.D);
  float sh = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) {
    const float4 x = h[i] + t[i];
    h[i] = x;
    sh += dot(x, x);
  }
  if (p.nout == 0) return;
  const float ih = rsqrt(g4_tg_sum(sh, part, sg, lane, ntg) / float(p.D) + p.eps);
  device const float4* wv = (device const float4*)w;
  device half4* yo = (device half4*)(y + ulong(row) * p.D);
  for (int i = int(tid); i < D4; i += int(ntg)) yo[i] = half4(h[i] * ih * wv[i]);
}

// erf with |error| < 1.5e-7 (Abramowitz & Stegun 7.1.26).
static inline float4 sp_erf(float4 x) {
  const float4 ax = fabs(x);
  const float4 t = 1.0f / (1.0f + 0.3275911f * ax);
  const float4 y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t +
                           0.254829592f) * t * exp(-ax * ax);
  return copysign(y, x);
}

// GeGLU of separate gate and up projections (fp32 [rows, I]) -> half with the
// exact GELU (erf) of the gate.
kernel void geglu_erf_f32(
    device const float4* G [[buffer(0)]],
    device const float4* U [[buffer(1)]],
    device half4* Y [[buffer(2)]],
    constant int& I [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) * 4 >= I) return;
  const ulong i = ulong(gid.y) * ulong(I / 4) + gid.x;
  const float4 g = G[i];
  Y[i] = half4(0.5f * g * (1.0f + sp_erf(g * 0.70710678118654752f)) * U[i]);
}
