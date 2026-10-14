// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ---------------------------------------------------------------------------
// Qwen3-MoE decoder of the Qwen3-Omni thinker (src/model/impl/qwen3_omni.h):
// full attention in every layer (GQA, per-head q / k RMSNorm, interleaved
// M-RoPE over the whole head, scale 1 / sqrt(head dim)), 128 routed experts
// with a softmax top-k router renormalized over the k picked, no shared
// expert, and DeepStack: the vision tower's intermediate features added to
// the image rows after the first layers. Semantics as in transformers'
// models/qwen3_omni_moe (Apache 2.0, THIRD_PARTY_NOTICES.md).
//
// Attention runs on Gemma 4's forest kernels (80_gemma.metal), instantiated
// here for 128-dim heads: q is scaled by 1 / sqrt(HD) in the prep kernel, so
// their scale-1 scores are the model's; with window 0 they mask by tree only.
// Routing is g4_route with unit expert scales (the softmax over the top-k
// logits equals the top-k of the softmax renormalized), the experts the
// grouped MoE GEMVs (15_moe.metal, SwiGLU).
// ---------------------------------------------------------------------------

// Attention inputs: per-head RMSNorm of q (weight q_norm) and k (k_norm),
// RoPE over the (d, d + HD/2) pairs at inv_freq[d] with the position
// component sel[d] (0 t, 1 h, 2 w: the M-RoPE section of frequency d), q
// times qscale; v as it is. fp16 Q / K / V batch rows. One simdgroup per
// (row, head): heads [0, H) queries, [H, H + Hkv) keys, [H + Hkv, H + 2 Hkv)
// values.
struct Q3PrepParams {
  int H;
  int Hkv;
  int ld;     // projection row stride (floats): [q | k | v]
  int k_off;  // column of the keys
  int v_off;  // column of the values
  int q_ld;   // Q row stride (H * HD)
  int kv_ld;  // K / V row stride (Hkv * HD)
  float eps;
  float qscale;
};

template <int HD>
kernel void q3m_attn_prep(
    device const float* X [[buffer(0)]],
    device const float* qn_w [[buffer(1)]],
    device const float* kn_w [[buffer(2)]],
    device const int4* pos [[buffer(3)]],
    device const float* inv_freq [[buffer(4)]],
    device const int* sel [[buffer(5)]],
    device half* Q [[buffer(6)]],
    device half* Kb [[buffer(7)]],
    device half* Vb [[buffer(8)]],
    constant Q3PrepParams& p [[buffer(9)]],
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
  if (kind == 2) {
    device half* vo = Vb + row * p.kv_ld + h * HD + lane * NPT;
    UNROLL for (int i = 0; i < NPT; ++i) vo[i] = half(src[i]);
    return;
  }
  float ss = 0.0f;
  UNROLL for (int i = 0; i < NPT; ++i) {
    x[i] = src[i];
    ss += x[i] * x[i];
  }
  ss = simd_sum(ss);
  const float inv = rsqrt(ss / float(HD) + p.eps);
  device const float* nw = (kind == 0 ? qn_w : kn_w) + lane * NPT;
  UNROLL for (int i = 0; i < NPT; ++i) x[i] = x[i] * inv * nw[i];
  const int4 ps = pos[row];
  UNROLL for (int i = 0; i < NPT; ++i) {
    const int d = (lane & 15) * NPT + i;
    const float partner = simd_shuffle_xor(x[i], ushort(16));
    const int s = sel[d];
    const float ang = float(s == 0 ? ps.x : (s == 1 ? ps.y : ps.z)) * inv_freq[d];
    const float c = precise::cos(ang), sn = precise::sin(ang);
    x[i] = lane < 16 ? x[i] * c - partner * sn : x[i] * c + partner * sn;
  }
  const float scale = kind == 0 ? p.qscale : 1.0f;
  device half* o = (kind == 0 ? Q + row * p.q_ld : Kb + row * p.kv_ld) + h * HD + lane * NPT;
  UNROLL for (int i = 0; i < NPT; ++i) o[i] = half(x[i] * scale);
}

template [[host_name("q3m_attn_prep128")]] kernel void q3m_attn_prep<128>(
    device const float*, device const float*, device const float*, device const int4*, device const float*,
    device const int*, device half*, device half*, device half*, constant Q3PrepParams&, uint2, uint, uint);

// Gemma 4's forest attention (80_gemma.metal) for 128-dim heads: all G query
// heads of a KV head in one threadgroup (32 x G threads), the single-row
// decode kernel and the split-record combine.
INSTANTIATE_G4A("g4_attention128_g1", 128, 1, 1)
INSTANTIATE_G4A("g4_attention128_g2", 128, 2, 1)
INSTANTIATE_G4A("g4_attention128_g4", 128, 4, 1)
INSTANTIATE_G4A("g4_attention128_g8", 128, 8, 1)
INSTANTIATE_G4D(128, 1)
INSTANTIATE_G4D(128, 2)
INSTANTIATE_G4D(128, 4)
INSTANTIATE_G4D(128, 8)
template [[host_name("g4_attention_combine128")]] kernel void g4_attention_combine<128>(
    device const float*, device half*, constant G4AttnParams&, device const AttnCombine*, uint3);

// After the experts: h += the routed experts' rows ys (sorted pairs at
// slotpos) times their routing weights, summed in slot order; then (nout = 1)
// y = rmsnorm(h) * w_next, the next layer's input. One threadgroup per row.
struct Q3MoeOutParams {
  int D;
  int k;
  float eps;
  int nout;
};

kernel void q3m_moe_out(
    device const float* ys [[buffer(0)]],
    device const int* slotpos [[buffer(1)]],
    device const float* wts [[buffer(2)]],
    device float* H [[buffer(3)]],
    device const float* w_next [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant Q3MoeOutParams& p [[buffer(6)]],
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
  device float4* h = (device float4*)(H + ulong(row) * p.D);
  float ss = 0.0f;
  for (int i = int(tid); i < D4; i += int(ntg)) {
    float4 acc = float4(0.0f);
    for (int j = 0; j < p.k; ++j) acc += sw[j] * *((device const float4*)(ys + ulong(sp[j]) * p.D) + i);
    const float4 x = h[i] + acc;
    h[i] = x;
    ss += dot(x, x);
  }
  if (p.nout == 0) return;
  const float inv = rsqrt(g4_tg_sum(ss, part, sg, lane, ntg) / float(p.D) + p.eps);
  device const float4* wn = (device const float4*)w_next;
  device half4* yo = (device half4*)(y + ulong(row) * p.D);
  for (int i = int(tid); i < D4; i += int(ntg)) yo[i] = half4(h[i] * inv * wn[i]);
}

// dst[rows[i]] += src[i] (fp32 rows of `width` elements): DeepStack features
// added to the image rows of the residual.
kernel void scatter_add_rows_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const int* rows [[buffer(2)]],
    constant int& width [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) * 4 >= width) return;
  device float4* d = (device float4*)(dst + ulong(rows[gid.y]) * width) + gid.x;
  *d += *((device const float4*)(src + ulong(gid.y) * width) + gid.x);
}
