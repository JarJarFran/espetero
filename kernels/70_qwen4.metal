// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ---------------------------------------------------------------------------
// Qwen4-Exp (qwen4_exp, Qwen3.8-Flash-Next; docs/qwen4_exp.md): the residual
// is S = 4 streams of D values (fp32 H [rows][S * D], stream-major), read and
// written by gated hyper-connection blocks instead of pre-norms:
//   N      = per-stream RMSNorm(H) * (1 + w)                   (q4_hc_norm)
//   dn     = A_down N  (+ the S injection rows)                (GEMV, fp32)
//   a      = silu(dn / S);  inj_s = 2 sigmoid(dn_inj,s / S)    (q4_hc_gates)
//   m      = A_up a                                            (GEMV, fp32)
//   x[c]   = 1/S sum_s sigmoid(m[s, c]) N[s, c]                (q4_hc_mix)
//   y      = block(x);  H[s, c] += inj_s y[c]                  (q4_hc_inject)
// The final mixer is a block without injection whose x feeds the LM head.
// The n-gram embedding before layer 1 (q4_ple_*) mixes 16 hashed table rows
// per token into the residual through a gated key/value path and a dilated
// causal conv.
// ---------------------------------------------------------------------------

// H [rows][S * D] = the embedding row h [rows][D] in every stream.
kernel void q4_embed_streams(
    device const float* h [[buffer(0)]],
    device float* H [[buffer(1)]],
    constant int& D [[buffer(2)]],
    constant int& S [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int j = gid.x;  // float4 column
  const ulong row = gid.y;
  if (j >= D / 4) return;
  const float4 v = ((device const float4*)(h + row * D))[j];
  for (int s = 0; s < S; ++s) ((device float4*)(H + row * ulong(S * D) + ulong(s) * D))[j] = v;
}

// Per-stream RMSNorm with the (1 + w) weight folded into w: out = x rsqrt(mean(x^2) + eps) w.
// One threadgroup per (stream, row); `in_ld` / `out_ld` are the row strides.
// Writes fp16 (GEMV operand) and, when `out32` is set, fp32 too.
kernel void q4_stream_norm(
    device const float* X [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device half* Y [[buffer(2)]],
    constant int& D [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant int& in_ld [[buffer(5)]],
    constant int& out_ld [[buffer(6)]],
    device float* Y32 [[buffer(7)]],
    constant int& out32 [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 ntg2 [[threads_per_threadgroup]]) {
  const uint ntg = ntg2.x;
  threadgroup float red[32];
  const int s = tg.x;
  const ulong row = tg.y;
  device const float4* x = (device const float4*)(X + row * ulong(in_ld) + ulong(s) * D);
  float acc = 0.0f;
  for (int j = int(tid); j < D / 4; j += int(ntg)) acc += dot(x[j], x[j]);
  acc = simd_sum(acc);
  if (lane == 0) red[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float t = lane < (ntg + 31) / 32 ? red[lane] : 0.0f;
    t = simd_sum(t);
    if (lane == 0) red[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(red[0] / float(D) + eps);
  device const float4* ws = (device const float4*)(w + ulong(s) * D);
  device half4* y = (device half4*)(Y + row * ulong(out_ld) + ulong(s) * D);
  device float4* y32 = (device float4*)(Y32 + row * ulong(out_ld) + ulong(s) * D);
  for (int j = int(tid); j < D / 4; j += int(ntg)) {
    const float4 v = x[j] * inv * ws[j];
    y[j] = half4(v);
    if (out32) y32[j] = v;
  }
}

// a = silu(dn[0..r) / S) as fp16; inj[s] = 2 sigmoid(dn_inj[s] / S), where the
// injection logits are dn[r..r+S) (mode 1: fused rows) or a separate [rows][S]
// buffer (mode 0); mode 2: no injection (the final mixer).
kernel void q4_hc_gates(
    device const float* dn [[buffer(0)]],
    device half* a [[buffer(1)]],
    device const float* inj_in [[buffer(2)]],
    device float* inj [[buffer(3)]],
    constant int& r [[buffer(4)]],
    constant int& dn_ld [[buffer(5)]],
    constant int& S [[buffer(6)]],
    constant int& mode [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int j = gid.x;
  const ulong row = gid.y;
  const float invS = 1.0f / float(S);
  if (j < r) {
    a[row * ulong(r) + j] = half(silu(dn[row * ulong(dn_ld) + j] * invS));
  } else if (j < r + S && mode != 2) {
    const int s = j - r;
    const float v = mode == 1 ? dn[row * ulong(dn_ld) + r + s] : inj_in[row * ulong(S) + s];
    inj[row * ulong(S) + s] = 2.0f / (1.0f + exp(-v * invS));
  }
}

// x[c] = 1/S sum_s sigmoid(m[s, c]) N[s, c] as fp16 (the block input).
kernel void q4_hc_mix(
    device const float* m [[buffer(0)]],
    device const half* N [[buffer(1)]],
    device half* x [[buffer(2)]],
    constant int& D [[buffer(3)]],
    constant int& S [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int j = gid.x;  // float4 column
  const ulong row = gid.y;
  if (j >= D / 4) return;
  float4 acc = 0.0f;
  for (int s = 0; s < S; ++s) {
    const ulong o = row * ulong(S * D) + ulong(s) * D;
    const float4 g = ((device const float4*)(m + o))[j];
    acc += float4(((device const half4*)(N + o))[j]) / (1.0f + exp(-g));
  }
  ((device half4*)(x + row * ulong(D)))[j] = half4(acc / float(S));
}

// H[s, c] += inj[s] y[c]; y has its own row stride (y_ld) and row map.
kernel void q4_hc_inject(
    device float* H [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device const float* inj [[buffer(2)]],
    constant int& D [[buffer(3)]],
    constant int& S [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int j = gid.x;
  const ulong row = gid.y;
  if (j >= D / 4) return;
  const float4 v = ((device const float4*)(y + row * ulong(D)))[j];
  for (int s = 0; s < S; ++s)
    ((device float4*)(H + row * ulong(S * D) + ulong(s) * D))[j] += inj[row * ulong(S) + s] * v;
}

kernel void q4_zero(device float* y [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = 0.0f;
}

// DeltaNet gated norm with the sigmoid output gate (output_gate_type sigmoid):
// out = RMSNorm_w(o) * sigmoid(z) (the weight is plain, no +1).
kernel void gated_rmsnorm_sig(
    device const float* Yin [[buffer(0)]],
    device const float* Zg [[buffer(1)]],
    device const float* Wn [[buffer(2)]],
    device half* Out [[buffer(3)]],
    constant int& width [[buffer(4)]],
    constant int& z_ld [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int Dv = 128;
  const int head = tgpig.x * 4 + sg;
  const ulong row = tgpig.y;
  if (head * Dv >= width) return;
  const float4 v = *((device const float4*)(Yin + row * width + head * Dv + lane * 4));
  const float4 z = *((device const float4*)(Zg + row * z_ld + head * Dv + lane * 4));
  const float4 w = *((device const float4*)(Wn + lane * 4));
  const float ss = simd_sum(dot(v, v));
  const float inv = rsqrt(ss / float(Dv) + eps);
  *((device half4*)(Out + row * width + head * Dv + lane * 4)) = half4(v * inv * w / (1.0f + exp(-z)));
}

// ---------------------------------------------------------------------------
// n-gram mix (docs/qwen4_exp.md 5.4), per row r of the pass:
//   Kp = GroupedRMSNorm(Kraw; norm_key)   [S][D]   (Kraw = W_key e, fp32)
//   Q  = GroupedRMSNorm(H; norm_query)    [S][D]   (H entering the layer)
//   g_s = sign(d) sqrt(max(|d|, 1e-6)),  d = Kp_s . Q_s / sqrt(D)
//   G_s = sigmoid(g_s) Vp                 (Vp = W_val e, [D])
//   Gn  = GroupedRMSNorm(G; norm_conv)    -> written fp32 [rows][S * D]
// q4_ple_gate computes g_s (one threadgroup per (stream, row)); q4_ple_apply
// writes G and Gn; q4_ple_conv adds G + silu(dilated conv of Gn) to H.
// ---------------------------------------------------------------------------
kernel void q4_ple_gate(
    device const float* Kraw [[buffer(0)]],
    device const float* H [[buffer(1)]],
    device const float* wk [[buffer(2)]],
    device const float* wq [[buffer(3)]],
    device float* g [[buffer(4)]],
    constant int& D [[buffer(5)]],
    constant int& S [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 ntg2 [[threads_per_threadgroup]]) {
  const uint ntg = ntg2.x;
  threadgroup float red[3][32];
  const int s = tg.x;
  const ulong row = tg.y;
  const ulong o = row * ulong(S * D) + ulong(s) * D;
  device const float4* k = (device const float4*)(Kraw + o);
  device const float4* h = (device const float4*)(H + o);
  device const float4* wk4 = (device const float4*)(wk + ulong(s) * D);
  device const float4* wq4 = (device const float4*)(wq + ulong(s) * D);
  float kk = 0.0f, hh = 0.0f, kh = 0.0f;
  for (int j = int(tid); j < D / 4; j += int(ntg)) {
    const float4 a = k[j], b = h[j];
    kk += dot(a, a);
    hh += dot(b, b);
    kh += dot(a * wk4[j], b * wq4[j]);
  }
  kk = simd_sum(kk);
  hh = simd_sum(hh);
  kh = simd_sum(kh);
  if (lane == 0) {
    red[0][sg] = kk;
    red[1][sg] = hh;
    red[2][sg] = kh;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float a = 0.0f, b = 0.0f, c = 0.0f;
    for (uint i = 0; i < (ntg + 31) / 32; ++i) {
      a += red[0][i];
      b += red[1][i];
      c += red[2][i];
    }
    const float d = c * rsqrt(a / float(D) + eps) * rsqrt(b / float(D) + eps) * rsqrt(float(D));
    const float r = sqrt(max(abs(d), 1e-6f));
    g[row * ulong(S) + s] = d > 0.0f ? r : (d < 0.0f ? -r : 0.0f);
  }
}

// G_s = sigmoid(g_s) Vp and Gn = GroupedRMSNorm(G; norm_conv), one threadgroup per (stream, row).
kernel void q4_ple_apply(
    device const float* Vp [[buffer(0)]],
    device const float* g [[buffer(1)]],
    device const float* wc [[buffer(2)]],
    device float* G [[buffer(3)]],
    device float* Gn [[buffer(4)]],
    constant int& D [[buffer(5)]],
    constant int& S [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 ntg2 [[threads_per_threadgroup]]) {
  const uint ntg = ntg2.x;
  threadgroup float red[32];
  const int s = tg.x;
  const ulong row = tg.y;
  const float gate = 1.0f / (1.0f + exp(-g[row * ulong(S) + s]));
  device const float4* v = (device const float4*)(Vp + row * ulong(D));
  float acc = 0.0f;
  for (int j = int(tid); j < D / 4; j += int(ntg)) {
    const float4 x = gate * v[j];
    acc += dot(x, x);
  }
  acc = simd_sum(acc);
  if (lane == 0) red[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float t = lane < (ntg + 31) / 32 ? red[lane] : 0.0f;
    t = simd_sum(t);
    if (lane == 0) red[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(red[0] / float(D) + eps);
  const ulong o = row * ulong(S * D) + ulong(s) * D;
  device const float4* w4 = (device const float4*)(wc + ulong(s) * D);
  for (int j = int(tid); j < D / 4; j += int(ntg)) {
    const float4 x = gate * v[j];
    ((device float4*)(G + o))[j] = x;
    ((device float4*)(Gn + o))[j] = x * inv * w4[j];
  }
}

// H[r] += G[r] + silu(sum_j w[ch][j] * Gn at tap j) for the depthwise conv of
// kernel K = 4 and dilation `dil` (= ngram size 3): taps t-9, t-6, t-3, t with
// weights w[ch][0..3]. src[r * 3 + i] names the row holding tap t-(3 - i)*dil:
// >= 0 a row of this pass, -1 - k entry k of the sequence's history in `hist`
// (hist_slot[r] selects the slot; entries are the last 9 Gn rows, oldest first),
// INT_MIN for positions before the sequence start (zero).
kernel void q4_ple_conv(
    device float* H [[buffer(0)]],
    device const float* G [[buffer(1)]],
    device const float* Gn [[buffer(2)]],
    device const float* wconv [[buffer(3)]],
    device const int* src [[buffer(4)]],
    device const float* hist [[buffer(5)]],
    device const int* hist_slot [[buffer(6)]],
    constant int& W [[buffer(7)]],
    constant int& hist_rows [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int ch = gid.x;
  const ulong row = gid.y;
  if (ch >= W) return;
  const float4 w = ((device const float4*)wconv)[ch];
  float acc = w[3] * Gn[row * ulong(W) + ch];
  for (int i = 0; i < 3; ++i) {
    const int s = src[row * 3 + i];
    float v = 0.0f;
    if (s >= 0) v = Gn[ulong(s) * ulong(W) + ch];
    else if (s != INT_MIN) v = hist[(ulong(hist_slot[row]) * ulong(hist_rows) + ulong(-1 - s)) * ulong(W) + ch];
    acc += w[i] * v;
  }
  H[row * ulong(W) + ch] += G[row * ulong(W) + ch] + silu(acc);
}

// ---------------------------------------------------------------------------
// Expert cache: a pool of S expert slots per MoE layer in the engine's
// GPU layout, the same as the matrix of all experts but with S in place of E:
// codes block-major (a slot's rows are one contiguous range), scales and biases
// group-major SB[g][pair] (a slot's pairs are one range per group).
// ---------------------------------------------------------------------------

// Routed expert ids -> pool slots (map[e] >= 0 for every id of the pass).
kernel void q4_remap_ids(
    device int* ids [[buffer(0)]],
    device const int* map [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < n) ids[gid] = map[ids[gid]];
}

// Affine scales and biases of N rows (one expert) into a pool whose SB has
// pool_pairs pairs per group, at pair offset pair0 (repack_scales at a stride).
kernel void repack_scales_at(
    device const ushort* S [[buffer(0)]],
    device const ushort* Z [[buffer(1)]],
    device half4* SB [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& G [[buffer(5)]],
    constant int& half_src [[buffer(6)]],
    constant int& pair0 [[buffer(7)]],
    constant int& pool_pairs [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int pair = gid.x;
  const int g = gid.y;
  const int KG = K / G;
  if (pair * 2 >= N || g >= KG) return;
  const int n = pair * 2;
  half4 v;
  v.x = half(half_src ? float(as_type<half>(S[ulong(n) * KG + g])) : bf2f(S[ulong(n) * KG + g]));
  v.z = half(half_src ? float(as_type<half>(Z[ulong(n) * KG + g])) : bf2f(Z[ulong(n) * KG + g]));
  v.y = half(half_src ? float(as_type<half>(S[ulong(n + 1) * KG + g])) : bf2f(S[ulong(n + 1) * KG + g]));
  v.w = half(half_src ? float(as_type<half>(Z[ulong(n + 1) * KG + g])) : bf2f(Z[ulong(n + 1) * KG + g]));
  SB[ulong(g) * pool_pairs + pair0 + pair] = v;
}

// Copy expert `src_e` of a whole-layer matrix (E experts) into pool slot
// `dst_s`: `rows` rows per expert, `row_bytes` code bytes per row (codes are
// block-major, so a contiguous range), scales at each matrix's own pair stride.
// Grid x: max(code words, scale pairs * groups).
kernel void q4_pool_admit(
    device const uint* src_w [[buffer(0)]],
    device const half4* src_sb [[buffer(1)]],
    device uint* dst_w [[buffer(2)]],
    device half4* dst_sb [[buffer(3)]],
    constant int& rows [[buffer(4)]],
    constant int& row_bytes [[buffer(5)]],
    constant int& groups [[buffer(6)]],
    constant int& src_pairs [[buffer(7)]],
    constant int& dst_pairs [[buffer(8)]],
    constant int& src_e [[buffer(9)]],
    constant int& dst_s [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
  const ulong words = ulong(rows) * ulong(row_bytes) / 4;
  if (gid < words) dst_w[ulong(dst_s) * words + gid] = src_w[ulong(src_e) * words + gid];
  const int pairs = rows / 2;
  if (gid < ulong(pairs) * ulong(groups)) {
    const int g = int(gid) / pairs, p = int(gid) % pairs;
    dst_sb[ulong(g) * dst_pairs + ulong(dst_s) * pairs + p] = src_sb[ulong(g) * src_pairs + ulong(src_e) * pairs + p];
  }
}

// Dense fp16 router [E][D] (row-major) times the MoE input X [rows][D] (fp16)
// -> fp32 logits [rows][E]: one simdgroup per (row, expert), fp32 sums. The
// dense matrix path (MMA tiles) wastes most of its work on a 1-row pass.
kernel void q4_router_gemv(
    device const half* W [[buffer(0)]],
    device const half* X [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant int& E [[buffer(3)]],
    constant int& D [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int e = int(tg.x) * 8 + int(sg);
  const ulong row = tg.y;
  if (e >= E) return;
  device const half4* w = (device const half4*)(W + ulong(e) * ulong(D));
  device const half4* x = (device const half4*)(X + row * ulong(D));
  float acc = 0.0f;
  for (int j = int(lane); j < D / 4; j += 32) acc += dot(float4(w[j]), float4(x[j]));
  acc = simd_sum(acc);
  if (lane == 0) Y[row * ulong(E) + e] = acc;
}
