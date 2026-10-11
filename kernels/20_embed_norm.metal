// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ---------------------------------------------------------------------------
// Embedding gather. Quantized tables keep one code per element (4-bit
// nibbles, eight per little-endian word, or bytes) with per-group half scale
// and bias [V, D/G]; dense tables are fp16.
// ---------------------------------------------------------------------------
kernel void embed_q4(
    device const int* tokens [[buffer(0)]],
    device const uint* W [[buffer(1)]],
    device const half* S [[buffer(2)]],
    device const half* Z [[buffer(3)]],
    device float* H [[buffer(4)]],
    constant int& D [[buffer(5)]],
    constant int& G [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int j = gid.x;  // word: 8 elements
  const int row = gid.y;
  if (j >= D / 8) return;
  const ulong tok = ulong(tokens[row]);
  const uint w = W[tok * (D / 8) + j];
  const int g = (j * 8) / G;  // G >= 8, so one group per word
  const float s = float(S[tok * (D / G) + g]);
  const float b = float(Z[tok * (D / G) + g]);
  device float* out = H + ulong(row) * D + j * 8;
  UNROLL for (int i = 0; i < 8; ++i) out[i] = s * float((w >> (4 * i)) & 0xFu) + b;
}

kernel void embed_q8(
    device const int* tokens [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const half* S [[buffer(2)]],
    device const half* Z [[buffer(3)]],
    device float* H [[buffer(4)]],
    constant int& D [[buffer(5)]],
    constant int& G [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int j = gid.x;
  const int row = gid.y;
  if (j >= D / 8) return;
  const ulong tok = ulong(tokens[row]);
  const int g = (j * 8) / G;
  const float s = float(S[tok * (D / G) + g]);
  const float b = float(Z[tok * (D / G) + g]);
  device const uchar* w = W + tok * D + j * 8;
  device float* out = H + ulong(row) * D + j * 8;
  UNROLL for (int i = 0; i < 8; ++i) out[i] = s * float(w[i]) + b;
}

kernel void embed_f16(
    device const int* tokens [[buffer(0)]],
    device const half* W [[buffer(1)]],
    device float* H [[buffer(4)]],
    constant int& D [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int j = gid.x;
  const int row = gid.y;
  if (j >= D / 8) return;
  device const half* w = W + ulong(tokens[row]) * D + j * 8;
  device float* out = H + ulong(row) * D + j * 8;
  UNROLL for (int i = 0; i < 8; ++i) out[i] = float(w[i]);
}

// ---------------------------------------------------------------------------
// RMSNorm: fp32 rows -> fp16 rows, optional row gather. One threadgroup/row.
// ---------------------------------------------------------------------------
kernel void rmsnorm(
    device const float* X [[buffer(0)]],
    device const float* Wt [[buffer(1)]],
    device half* Y [[buffer(2)]],
    constant int& D [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    device const int* rows [[buffer(5)]],
    constant int& gather [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float partial[32];
  const int src_row = gather ? rows[row] : int(row);
  device const float4* x = (device const float4*)(X + ulong(src_row) * D);
  const int D4 = D / 4;
  float ss = 0.0f;
  for (int i = tid; i < D4; i += ntg) {
    const float4 v = x[i];
    ss += dot(v, v);
  }
  ss = simd_sum(ss);
  if (lane == 0) partial[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float t = (lane < (ntg + 31) / 32) ? partial[lane] : 0.0f;
    t = simd_sum(t);
    if (lane == 0) partial[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(partial[0] / float(D) + eps);
  device const float4* w = (device const float4*)Wt;
  device half4* y = (device half4*)(Y + ulong(row) * D);
  for (int i = tid; i < D4; i += ntg) y[i] = half4(x[i] * inv * w[i]);
}

// MTP input row: Y = [enorm(E) | hnorm(pre(H))] (half, 2D wide), where `pre`
// (optional) is the RMSNorm the hidden row still needs: the decoder's final
// norm for decoder rows, the MTP norm for chained draft rows.
kernel void mtp_prep(
    device const float* E [[buffer(0)]],
    device const float* H [[buffer(1)]],
    device const float* We [[buffer(2)]],
    device const float* Wh [[buffer(3)]],
    device const float* Wp [[buffer(4)]],
    device half* Y [[buffer(5)]],
    constant int& D [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant int& pre [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float3 partial[32];
  device const float4* e = (device const float4*)(E + ulong(row) * D);
  device const float4* h = (device const float4*)(H + ulong(row) * D);
  device const float4* wp = (device const float4*)Wp;
  const int D4 = D / 4;
  float3 ss = 0.0f;  // |e|^2, |h|^2, |h * wp|^2
  for (int i = tid; i < D4; i += ntg) {
    const float4 a = e[i], b = h[i];
    ss.x += dot(a, a);
    ss.y += dot(b, b);
    if (pre) {
      const float4 c = b * wp[i];
      ss.z += dot(c, c);
    }
  }
  ss = float3(simd_sum(ss.x), simd_sum(ss.y), simd_sum(ss.z));
  if (lane == 0) partial[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float3 t = (lane < (ntg + 31) / 32) ? partial[lane] : float3(0.0f);
    t = float3(simd_sum(t.x), simd_sum(t.y), simd_sum(t.z));
    if (lane == 0) partial[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float3 t = partial[0];
  const float ie = rsqrt(t.x / float(D) + eps);
  // pre: h1 = h * i1 * wp, then |h1|^2 = i1^2 |h * wp|^2.
  const float i1 = pre ? rsqrt(t.y / float(D) + eps) : 1.0f;
  const float ih = pre ? i1 * rsqrt(i1 * i1 * t.z / float(D) + eps) : rsqrt(t.y / float(D) + eps);
  device const float4* we = (device const float4*)We;
  device const float4* wh = (device const float4*)Wh;
  device half4* y = (device half4*)(Y + ulong(row) * 2 * D);
  for (int i = tid; i < D4; i += ntg) {
    y[i] = half4(e[i] * ie * we[i]);
    const float4 hp = pre ? h[i] * wp[i] : h[i];
    y[D4 + i] = half4(hp * ih * wh[i]);
  }
}

// SwiGLU of separate gate and up projections (fp32 [rows, I]) -> half.
kernel void swiglu_f32(
    device const float4* G [[buffer(0)]],
    device const float4* U [[buffer(1)]],
    device half4* Y [[buffer(2)]],
    constant int& I [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) * 4 >= I) return;
  const ulong i = ulong(gid.y) * ulong(I / 4) + gid.x;
  const float4 g = G[i];
  Y[i] = half4(g / (1.0f + exp(-g)) * U[i]);
}

// GeGLU of separate gate and up projections (fp32 [rows, I]) -> half: the
// tanh-approximated GELU of the gate times the up projection (Gemma 4).
kernel void geglu_f32(
    device const float4* G [[buffer(0)]],
    device const float4* U [[buffer(1)]],
    device half4* Y [[buffer(2)]],
    constant int& I [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) * 4 >= I) return;
  const ulong i = ulong(gid.y) * ulong(I / 4) + gid.x;
  const float4 g = G[i];
  Y[i] = half4(0.5f * g * (1.0f + precise::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g))) * U[i]);
}
