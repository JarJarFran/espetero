// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ---------------------------------------------------------------------------
// Gemma 4 vision tower (src/model/impl/gemma_vision.h): the learned 2D
// position tables, the q / k / v head norms with the 2D RoPE, and the
// pooling + standardization in front of embed_vision. The blocks' norms are
// 80_gemma's g4_post_attn / g4_post_ffn (the decoder's residual updates, D
// as a parameter) and the attention is vit_attention2 (60_vision). Semantics
// as in mlx-vlm's models/gemma4/vision.py (MIT).
// ---------------------------------------------------------------------------

// X[r] += T[0][col(r)] + T[1][row(r)]: T fp32 [2][n_pos][D], xy[r] = (column, row).
kernel void g4v_pos_add(
    device float* X [[buffer(0)]],
    device const float* T [[buffer(1)]],
    device const int2* xy [[buffer(2)]],
    constant int2& dp [[buffer(3)]],  // D, n_pos
    uint2 gid [[thread_position_in_grid]]) {
  const int d4 = int(gid.x);
  const int r = int(gid.y);
  const int D = dp.x;
  if (d4 * 4 >= D) return;
  const int2 p = xy[r];
  device float4* x = (device float4*)(X + ulong(r) * D) + d4;
  *x += *((device const float4*)(T + ulong(p.x) * D) + d4) + *((device const float4*)(T + (ulong(dp.y) + ulong(p.y)) * D) + d4);
}

// Split the fused q / k / v rows into head-major fp16 Q, K, V ([H][N][HD]):
// per head RMSNorm (q times q_norm, k times k_norm, v without a weight),
// then q and k rotated by the 2D RoPE: dims [0, HD/2) by the patch column,
// [HD/2, HD) by the row, each half as rotate_half pairs (i, i + HD/4) at
// inv_freq[i]. Scale 1 (vit_attention2 does not scale). One simdgroup per
// (row, head): lane l < HD/4 holds dims l, l + HD/4, HD/2 + l, 3 HD/4 + l.
struct G4vPrepParams {
  int N;
  int H;
  float eps;
  int pad;
};

template <int HD>
kernel void g4v_qkv_prep(
    device const float* QKV [[buffer(0)]],
    device const int2* xy [[buffer(1)]],
    device const float* inv_freq [[buffer(2)]],
    device const float* qn [[buffer(3)]],
    device const float* kn [[buffer(4)]],
    device half* Q [[buffer(5)]],
    device half* K [[buffer(6)]],
    device half* V [[buffer(7)]],
    constant G4vPrepParams& p [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NP = HD / 4;
  static_assert(NP <= 32, "one simdgroup per head");
  const int h = int(tgpig.x) * 4 + int(sg);
  const int r = int(tgpig.y);
  if (h >= p.H) return;
  const int D = p.H * HD;
  const bool on = int(lane) < NP;
  const int l = on ? int(lane) : 0;
  const int2 pos = xy[r];
  const float a0 = float(pos.x) * inv_freq[l], a1 = float(pos.y) * inv_freq[l];
  const float c0 = precise::cos(a0), s0 = precise::sin(a0), c1 = precise::cos(a1), s1 = precise::sin(a1);
  const ulong o = (ulong(h) * ulong(p.N) + ulong(r)) * HD;
  for (int kind = 0; kind < 3; ++kind) {
    device const float* src = QKV + ulong(r) * 3 * D + kind * D + h * HD;
    float x0 = on ? src[l] : 0.0f, x1 = on ? src[l + NP] : 0.0f;
    float x2 = on ? src[2 * NP + l] : 0.0f, x3 = on ? src[3 * NP + l] : 0.0f;
    const float ss = simd_sum(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3);
    const float inv = rsqrt(ss / float(HD) + p.eps);
    x0 *= inv, x1 *= inv, x2 *= inv, x3 *= inv;
    device half* dst = kind == 0 ? Q : (kind == 1 ? K : V);
    if (kind < 2) {
      device const float* w = kind == 0 ? qn : kn;
      x0 *= w[l], x1 *= w[l + NP], x2 *= w[2 * NP + l], x3 *= w[3 * NP + l];
      const float y0 = x0 * c0 - x1 * s0, y1 = x1 * c0 + x0 * s0;
      const float y2 = x2 * c1 - x3 * s1, y3 = x3 * c1 + x2 * s1;
      x0 = y0, x1 = y1, x2 = y2, x3 = y3;
    }
    if (on) {
      dst[o + l] = half(x0);
      dst[o + l + NP] = half(x1);
      dst[o + 2 * NP + l] = half(x2);
      dst[o + 3 * NP + l] = half(x3);
    }
  }
}

template [[host_name("g4v_qkv_prep64")]] kernel void g4v_qkv_prep<64>(
    device const float*, device const int2*, device const float*, device const float*, device const float*,
    device half*, device half*, device half*, constant G4vPrepParams&, uint2, uint, uint);
template [[host_name("g4v_qkv_prep72")]] kernel void g4v_qkv_prep<72>(
    device const float*, device const int2*, device const float*, device const float*, device const float*,
    device half*, device half*, device half*, constant G4vPrepParams&, uint2, uint, uint);
template [[host_name("g4v_qkv_prep80")]] kernel void g4v_qkv_prep<80>(
    device const float*, device const int2*, device const float*, device const float*, device const float*,
    device half*, device half*, device half*, constant G4vPrepParams&, uint2, uint, uint);

// One soft token per threadgroup: the mean of its k x k patch block of X
// (rec = first patch of the image, grid width, pooled row, pooled column),
// times `scale` (sqrt(D)), standardized ((t - bias) * scale, when set) and
// RMS-normalized without a weight (embed_vision's pre-projection norm), as
// fp16 rows for the projection. D <= 2048.
struct G4vPoolParams {
  int D;
  int k;
  float scale;
  float eps;
  int standardize;
  int pad;
};

kernel void g4v_pool(
    device const float* X [[buffer(0)]],
    device const int4* recs [[buffer(1)]],
    device const float* sbias [[buffer(2)]],
    device const float* sscale [[buffer(3)]],
    device half* Y [[buffer(4)]],
    constant G4vPoolParams& p [[buffer(5)]],
    uint t [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  threadgroup float4 row[512];
  const int4 rc = recs[t];
  const int D4 = p.D / 4;
  const float w = 1.0f / float(p.k * p.k);
  device const float4* sb = (device const float4*)sbias;
  device const float4* ss = (device const float4*)sscale;
  float sq = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) {
    float4 acc = float4(0.0f);
    for (int dy = 0; dy < p.k; ++dy)
      for (int dx = 0; dx < p.k; ++dx) {
        const ulong pr = ulong(rc.x) + ulong((rc.z * p.k + dy) * rc.y + rc.w * p.k + dx);
        acc += *((device const float4*)(X + pr * ulong(p.D)) + i);
      }
    float4 v = acc * w * p.scale;
    if (p.standardize) v = (v - sb[i]) * ss[i];
    row[i] = v;
    sq += dot(v, v);
  }
  const float inv = rsqrt(g4_tg_sum(sq, part, sg, lane, ntg) / float(p.D) + p.eps);
  device half4* y = (device half4*)(Y + ulong(t) * ulong(p.D));
  for (int i = int(tid); i < D4; i += int(ntg)) y[i] = half4(row[i] * inv);
}

// ---------------------------------------------------------------------------
// Gemma 4 unified (gemma4_unified_vision): no tower. A soft token's 48 x 48
// pixels go through 60_vision's layernorm and the patch projection (hgemm
// with its bias); this kernel does the rest before embed_vision, per row of
// X (fp32 [rows][D]):
//   y = LayerNorm(x) * w2 + b2 + T[col][0] + T[row][1]   (T fp32 [n_pos][2][D])
//   z = LayerNorm(y) * w3 + b3
//   Y = z * rsqrt(mean(z^2) + eps)                        (fp16, no weight)
// LayerNorms as torch's (biased variance, eps inside the root), each in
// fp32 with two passes over the row kept in threadgroup memory (D <= 4096).
// Semantics of transformers' Gemma4UnifiedVisionEmbedder (Apache-2.0).
// ---------------------------------------------------------------------------
struct G4uNormParams {
  int D;
  float ln_eps, eps;
  int pad;
};

static float g4u_tg_sum(float v, threadgroup float* part, uint tid, uint nsg, uint sg, uint lane) {
  v = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = int(lane) < int(nsg) ? part[lane] : 0.0f;
  return simd_sum(t);
}

kernel void g4u_embed_norm(
    device const float* X [[buffer(0)]],
    device const float* w2 [[buffer(1)]],
    device const float* b2 [[buffer(2)]],
    device const float* T [[buffer(3)]],
    device const int2* xy [[buffer(4)]],
    device const float* w3 [[buffer(5)]],
    device const float* b3 [[buffer(6)]],
    device half* Y [[buffer(7)]],
    constant G4uNormParams& p [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 buf[1024];
  threadgroup float part[32];
  const int D4 = p.D / 4;
  const uint nsg = (ntg + 31) / 32;
  const float inv_d = 1.0f / float(p.D);
  device const float4* x = (device const float4*)(X + ulong(row) * p.D);
  // LayerNorm 2 (patch_ln2), then the position embeddings.
  float s = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) {
    buf[i] = x[i];
    s += buf[i].x + buf[i].y + buf[i].z + buf[i].w;
  }
  const float mean2 = g4u_tg_sum(s, part, tid, nsg, sg, lane) * inv_d;
  float q = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) {
    const float4 d = buf[i] - mean2;
    q += dot(d, d);
  }
  const float inv2 = rsqrt(g4u_tg_sum(q, part, tid, nsg, sg, lane) * inv_d + p.ln_eps);
  const int2 pos = xy[row];
  device const float4* tx = (device const float4*)(T + (ulong(pos.x) * 2 + 0) * ulong(p.D));
  device const float4* ty = (device const float4*)(T + (ulong(pos.y) * 2 + 1) * ulong(p.D));
  device const float4* W2 = (device const float4*)w2;
  device const float4* B2 = (device const float4*)b2;
  s = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) {
    buf[i] = (buf[i] - mean2) * inv2 * W2[i] + B2[i] + tx[i] + ty[i];
    s += buf[i].x + buf[i].y + buf[i].z + buf[i].w;
  }
  // LayerNorm 3 (pos_norm), then the RMSNorm without a weight.
  const float mean3 = g4u_tg_sum(s, part, tid, nsg, sg, lane) * inv_d;
  q = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) {
    const float4 d = buf[i] - mean3;
    q += dot(d, d);
  }
  const float inv3 = rsqrt(g4u_tg_sum(q, part, tid, nsg, sg, lane) * inv_d + p.ln_eps);
  device const float4* W3 = (device const float4*)w3;
  device const float4* B3 = (device const float4*)b3;
  float r = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) {
    buf[i] = (buf[i] - mean3) * inv3 * W3[i] + B3[i];
    r += dot(buf[i], buf[i]);
  }
  const float invr = rsqrt(g4u_tg_sum(r, part, tid, nsg, sg, lane) * inv_d + p.eps);
  device half4* y = (device half4*)(Y + ulong(row) * p.D);
  for (int i = int(tid); i < D4; i += int(ntg)) y[i] = half4(buf[i] * invr);
}
