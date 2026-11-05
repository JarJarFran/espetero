// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero
// ---------------------------------------------------------------------------
// GLM-5.3-Flash (glm5_next) kernels, docs/glm5.md. Tier 0 (reference path):
// one kernel per op, fp32 activations; the mHC, latent attention, pool scores,
// top-k, SwiGLU and fp8 / bf16 GEMVs are DeepSeek-V4.1's (80_dsv4.metal).
//
// KDA (docs/glm5.md 3): the q | k | v projection rows of a pass, X [rows][3C]
// (C = H * 128), go through a causal depthwise conv of 4 taps + SiLU and the
// q / k l2 norms (glm5_kda_conv); per-channel decays g in (-5, 0) and the
// beta gates (glm5_kda_gate); the delta rule with a per-key-channel decay,
// token by token (glm5_kda_recur); the gated RMSNorm (glm5_kda_out).

// ---- conv + SiLU + l2 norm. One simdgroup per (row, 128-channel head of q | k | v), 4 channels a lane.
// Taps before the pass come from the slot's conv state [3][3C] (oldest first). Wc: [3C][4], tap i
// multiplies the input 3 - i tokens back (the checkpoint's order: index 3 is the current token).
struct Glm5ConvP {
  int rows, C3, heads3;  // rows of the pass (one sequence), 3C, 3H
  int hk;                // heads per projection (H): q heads [0, H), k [H, 2H), v [2H, 3H)
};
kernel void glm5_kda_conv(device const float* X [[buffer(0)]], device const float* state [[buffer(1)]],
                          device const float* Wc [[buffer(2)]], device float* Out [[buffer(3)]],
                          constant Glm5ConvP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int head = int(tg.x) * 8 + int(sg), r = int(tg.y);
  if (head >= p.heads3 || r >= p.rows) return;
  const int c0 = head * 128 + int(lane) * 4;
  float4 acc = 0.0f;
  for (int i = 0; i < 4; ++i) {  // tap i: token r - 3 + i
    const int t = r - 3 + i;
    const float4 x = t >= 0 ? *((device const float4*)(X + ulong(t) * ulong(p.C3) + ulong(c0)))
                            : *((device const float4*)(state + ulong(t + 3) * ulong(p.C3) + ulong(c0)));
    acc += x * float4(Wc[(c0 + 0) * 4 + i], Wc[(c0 + 1) * 4 + i], Wc[(c0 + 2) * 4 + i], Wc[(c0 + 3) * 4 + i]);
  }
  acc = acc / (1.0f + exp(-acc));
  if (head < 2 * p.hk) {  // q, k: x / sqrt(sum x^2 + 1e-6) (FLA l2norm), q also / sqrt(128)
    const float ss = simd_sum(dot(acc, acc));
    acc *= rsqrt(ss + 1e-6f) * (head < p.hk ? rsqrt(128.0f) : 1.0f);
  }
  *((device float4*)(Out + ulong(r) * ulong(p.C3) + ulong(c0))) = acc;
}

// Conv state after the pass: the last 3 rows of X (the earlier state rows when the pass is shorter). In place
// is safe: each thread reads its channel's three values before writing them.
kernel void glm5_conv_save(device const float* X [[buffer(0)]], device float* state [[buffer(1)]],
                           constant int2& p [[buffer(2)]],  // (rows, 3C)
                           uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.y) return;
  float v[3];
  for (int j = 0; j < 3; ++j) {
    const int t = p.x - 3 + j;
    v[j] = t >= 0 ? X[ulong(t) * ulong(p.y) + c] : state[ulong(t + 3) * ulong(p.y) + c];
  }
  for (int j = 0; j < 3; ++j) state[ulong(j) * ulong(p.y) + c] = v[j];
}

// g = lower_bound * sigmoid(A[h] * (f + dt_bias)) per channel (A = exp(A_log), computed at load), beta = sigmoid(b)
// per head.
struct Glm5GateP {
  int rows, H, fs, bs;  // fs, bs: row strides of F and B (floats)
  float lb;
};
kernel void glm5_kda_gate(device const float* F [[buffer(0)]], device const float* B [[buffer(1)]],
                          device const float* A [[buffer(2)]], device const float* dt [[buffer(3)]],
                          device float* G [[buffer(4)]], device float* Beta [[buffer(5)]],
                          constant Glm5GateP& p [[buffer(6)]], uint2 g [[thread_position_in_grid]]) {
  const int c = int(g.x), r = int(g.y);
  const int C = p.H * 128;
  if (c >= C || r >= p.rows) return;
  const int h = c / 128;
  const float x = A[h] * (F[ulong(r) * ulong(p.fs) + ulong(c)] + dt[c]);
  G[ulong(r) * ulong(C) + ulong(c)] = p.lb / (1.0f + exp(-x));
  if (c % 128 == 0) Beta[ulong(r) * ulong(p.H) + ulong(h)] = 1.0f / (1.0f + exp(-B[ulong(r) * ulong(p.bs) + ulong(h)]));
}

// The delta rule, token by token (exact recurrence, the reference for chunked forms):
//   S_h <- diag(exp(g)) S_h;  u = S_h^T k;  S_h <- S_h + k (beta (v - u))^T;  o = S_h^T q
// S_h [128 k][128 v] is kept as [v][k] (k contiguous). A simdgroup owns one v row of one head (its 32 lanes x 4
// k values in registers), so the 128 x 128 state of a head is 16 threadgroups of 8 simdgroups.
// Grid (16, H), 256 threads.
struct Glm5RecurP {
  int rows, H, C3;  // rows of the pass, heads, row stride of the conv output (q | k | v)
  int write_state;
};
kernel void glm5_kda_recur(device const float* QKV [[buffer(0)]], device const float* G [[buffer(1)]],
                           device const float* Beta [[buffer(2)]], device float* S [[buffer(3)]],
                           device float* O [[buffer(4)]], constant Glm5RecurP& p [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const int h = int(tg.y), vr = int(tg.x) * 8 + int(sg);
  const int C = p.H * 128;
  device float* st_p = S + (ulong(h) * 128 + ulong(vr)) * 128 + ulong(lane) * 4;
  float4 st = *((device const float4*)st_p);
  const int k0 = int(lane) * 4;
  for (int t = 0; t < p.rows; ++t) {
    device const float* row = QKV + ulong(t) * ulong(p.C3);
    const float4 q = *((device const float4*)(row + h * 128 + k0));
    const float4 k = *((device const float4*)(row + C + h * 128 + k0));
    const float v = row[2 * C + h * 128 + vr];
    const float4 g = *((device const float4*)(G + ulong(t) * ulong(C) + ulong(h * 128 + k0)));
    st *= exp(g);
    const float u = simd_sum(dot(st, k));
    const float delta = (v - u) * Beta[ulong(t) * ulong(p.H) + ulong(h)];
    st += k * delta;
    const float o = simd_sum(dot(st, q));
    if (lane == 0) O[ulong(t) * ulong(C) + ulong(h * 128 + vr)] = o;
  }
  if (p.write_state) *((device float4*)st_p) = st;
}

// out = RMSNorm(o_h; w, eps) * sigmoid(z_h) per (row, head of 128), in place over O. One simdgroup per head.
struct Glm5OutP {
  int rows, H, zs;  // zs: row stride of Z
  float eps;
};
kernel void glm5_kda_out(device float* O [[buffer(0)]], device const float* Z [[buffer(1)]],
                         device const float* w [[buffer(2)]], constant Glm5OutP& p [[buffer(3)]],
                         uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const int v = int(gid) * 8 + int(sg);
  if (v >= p.rows * p.H) return;
  const int r = v / p.H, h = v % p.H;
  device float* o = O + ulong(r) * ulong(p.H * 128) + ulong(h * 128) + ulong(lane) * 4;
  device const float* z = Z + ulong(r) * ulong(p.zs) + ulong(h * 128) + ulong(lane) * 4;
  float4 x = *((device float4*)o);
  const float rs = rsqrt(simd_sum(dot(x, x)) / 128.0f + p.eps);
  const float4 zz = *((device const float4*)z);
  x = x * rs * *((device const float4*)(w + lane * 4)) / (1.0f + exp(-zz));
  *((device float4*)o) = x;
}

// ---- DSA (docs/glm5.md 4.1): the per-head halves of kv_b_proj [H * (nope + v)][512] (bf16) around the latent
// attention. absorb: qa[h][c] = sum_i W[h * 512 + i][c] q[h][i] (i < 256), the query against latents;
// unabsorb: o[h][d] = sum_c W[h * 512 + 256 + d][c] oa[h][c], the head's output from its latent mix.
// One simdgroup per output value, lanes along the sum.
struct Glm5HeadP {
  int rows, H;
  int xs, ys;  // row strides of X and Y (floats)
};
kernel void glm5_absorb(device const ushort* W [[buffer(0)]], device const float* X [[buffer(1)]],
                        device float* Y [[buffer(2)]], constant Glm5HeadP& p [[buffer(3)]],
                        uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  // grid (512 / 8, H, rows): each simdgroup one c; lanes split i
  const int c = int(tg.x) * 8 + int(sg), h = int(tg.y), r = int(tg.z);
  device const float* q = X + ulong(r) * ulong(p.xs) + ulong(h * 256);
  float s = 0.0f;
  for (int i = int(lane); i < 256; i += 32) s += bf2f(W[(ulong(h) * 512 + ulong(i)) * 512 + ulong(c)]) * q[i];
  s = simd_sum(s);
  if (lane == 0) Y[ulong(r) * ulong(p.ys) + ulong(h * 512 + c)] = s;
}
kernel void glm5_unabsorb(device const ushort* W [[buffer(0)]], device const float* X [[buffer(1)]],
                          device float* Y [[buffer(2)]], constant Glm5HeadP& p [[buffer(3)]],
                          uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  // grid (256 / 8, H, rows): each simdgroup one d; lanes split c (16 values each)
  const int d = int(tg.x) * 8 + int(sg), h = int(tg.y), r = int(tg.z);
  device const float* oa = X + ulong(r) * ulong(p.xs) + ulong(h * 512);
  device const ushort* wr = W + (ulong(h) * 512 + 256 + ulong(d)) * 512;
  float s = 0.0f;
  for (int c = int(lane); c < 512; c += 32) s += bf2f(wr[c]) * oa[c];
  s = simd_sum(s);
  if (lane == 0) Y[ulong(r) * ulong(p.ys) + ulong(h * 256 + d)] = s;
}

// LayerNorm over n floats per row, in place (the indexer's k_norm: weight and bias, eps 1e-6).
struct Glm5LnP {
  int rows, n, stride;
  float eps;
};
kernel void glm5_layernorm(device float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                           device const float* b [[buffer(2)]], constant Glm5LnP& p [[buffer(3)]],
                           uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const int r = int(gid) * 8 + int(sg);
  if (r >= p.rows) return;
  device float* a = x + ulong(r) * ulong(p.stride);
  float s = 0.0f;
  for (int i = int(lane); i < p.n; i += 32) s += a[i];
  const float mu = simd_sum(s) / float(p.n);
  float v = 0.0f;
  for (int i = int(lane); i < p.n; i += 32) v += (a[i] - mu) * (a[i] - mu);
  const float rs = rsqrt(simd_sum(v) / float(p.n) + p.eps);
  for (int i = int(lane); i < p.n; i += 32) a[i] = (a[i] - mu) * rs * w[i] + b[i];
}

// k-pool compression (docs/glm5.md 4.2): every pool of 4 tokens a pass completes, K_b[c] = sum_i softmax_i(gate_i[c]
// + ape[i][c]) key_i[c]; members before the pass come from the slot's pending rows (token t at pending row t % 4),
// key | gate interleaved per row [256]. Out: pooled keys as halves at pool index b of the slot's pool cache.
struct Glm5PoolP {
  int p0, rows, ks;  // first position of the pass, its rows, row stride of KG (floats; key at 0, gate at 128)
  int b0, npool;     // first pool completed in the pass and their count
};
kernel void glm5_kpool(device const float* KG [[buffer(0)]], device const float* pend [[buffer(1)]],
                       device const float* ape [[buffer(2)]], device half* pooled [[buffer(3)]],
                       constant Glm5PoolP& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  const int c = int(g.x), j = int(g.y);
  if (c >= 128 || j >= p.npool) return;
  const int b = p.b0 + j;
  float key[4], lg[4], mx = -INFINITY;
  for (int i = 0; i < 4; ++i) {
    const int t = 4 * b + i;
    device const float* row = t >= p.p0 ? KG + ulong(t - p.p0) * ulong(p.ks) : pend + ulong(t % 4) * 256;
    key[i] = row[c];
    lg[i] = row[128 + c] + ape[i * 128 + c];
    mx = max(mx, lg[i]);
  }
  float s = 0.0f, v = 0.0f;
  for (int i = 0; i < 4; ++i) {
    const float e = exp(lg[i] - mx);
    s += e;
    v += e * key[i];
  }
  pooled[ulong(b) * 128 + ulong(c)] = half(v / s);
}
// The pending rows after the pass: the members of the last, incomplete pool (key | gate rows of the pass).
kernel void glm5_kpool_pend(device const float* KG [[buffer(0)]], device float* pend [[buffer(1)]],
                            constant int3& p [[buffer(2)]],  // (p0, rows, ks)
                            uint2 g [[thread_position_in_grid]]) {
  const int c = int(g.x), t = p.x + p.y - 1 - int(g.y);  // rows from the end
  if (c >= 256 || t < p.x || t < 4 * ((p.x + p.y) / 4)) return;
  pend[ulong(t % 4) * 256 + ulong(c)] = KG[ulong(t - p.x) * ulong(p.z) + ulong(c)];
}

// Selected token ids per row: the chosen pools' tokens (ds4_topk's pool ids, position order, -1 padded) and the
// incomplete tail (4 * floor((pos + 1) / 4) .. pos), -1 padded to `width`.
struct Glm5SelP {
  int rows, k, width;
};
kernel void glm5_expand_sel(device const int* pools [[buffer(0)]], device const int* pos [[buffer(1)]],
                            device int* sel [[buffer(2)]], constant Glm5SelP& p [[buffer(3)]],
                            uint r [[thread_position_in_grid]]) {
  if (int(r) >= p.rows) return;
  device int* o = sel + ulong(r) * ulong(p.width);
  int n = 0;
  for (int i = 0; i < p.k; ++i) {
    const int b = pools[ulong(r) * ulong(p.k) + ulong(i)];
    if (b < 0) break;
    for (int j = 0; j < 4; ++j) o[n++] = 4 * b + j;
  }
  const int ps = pos[r];
  for (int t = 4 * ((ps + 1) / 4); t <= ps; ++t) o[n++] = t;
  for (; n < p.width; ++n) o[n] = -1;
}

// MoE routing (docs/glm5.md 5): s = sigmoid(logit), experts = top-k(s + bias), ties to the lowest expert,
// weights = s / (sum + 1e-20) * route_scale. One simdgroup per row, up to 512 experts.
struct Glm5RouteP {
  int M, E, k;
  float route_scale;
};
kernel void glm5_route(device const float* L [[buffer(0)]], device const float* bias [[buffer(1)]],
                       device int* sel [[buffer(2)]], device float* wout [[buffer(3)]],
                       constant Glm5RouteP& p [[buffer(4)]], uint m [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
  constexpr int PER = 16;
  float sc[PER], b[PER];
  for (int i = 0; i < PER; ++i) {
    const int e = int(lane) + 32 * i;
    if (e < p.E) {
      sc[i] = 1.0f / (1.0f + exp(-L[ulong(m) * ulong(p.E) + ulong(e)]));
      b[i] = sc[i] + bias[e];
    } else {
      sc[i] = 0.0f;
      b[i] = -INFINITY;
    }
  }
  float tot = 0.0f;
  int chosen[8];
  float cw[8];
  for (int r = 0; r < p.k; ++r) {
    float best = -INFINITY;
    int bi = INT_MAX;
    for (int i = 0; i < PER; ++i) {
      const int e = int(lane) + 32 * i;
      if (b[i] > best || (b[i] == best && e < bi)) best = b[i], bi = e;
    }
    for (int off = 16; off > 0; off >>= 1) {
      const float ob = simd_shuffle_xor(best, ushort(off));
      const int oi = simd_shuffle_xor(bi, ushort(off));
      if (ob > best || (ob == best && oi < bi)) best = ob, bi = oi;
    }
    chosen[r] = bi;
    const int owner = bi % 32, slot = bi / 32;
    float s = 0.0f;
    for (int i = 0; i < PER; ++i)
      if (int(lane) == owner && i == slot) {
        s = sc[i];
        b[i] = -INFINITY;
      }
    s = simd_sum(s);
    cw[r] = s;
    tot += s;
  }
  if (lane == 0)
    for (int r = 0; r < p.k; ++r) {
      sel[ulong(m) * ulong(p.k) + ulong(r)] = chosen[r];
      wout[ulong(m) * ulong(p.k) + ulong(r)] = cw[r] / (tot + 1e-20f) * p.route_scale;
    }
}

// Y[m] = mean of the 4 streams of H[m] (the head's input before the final norm).
kernel void glm5_hc_mean(device const float* H [[buffer(0)]], device float* Y [[buffer(1)]],
                         constant int& D [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const int d = int(g.x), m = int(g.y);
  if (d >= D) return;
  device const float* h = H + ulong(m) * ulong(4 * D) + ulong(d);
  Y[ulong(m) * ulong(D) + ulong(d)] = 0.25f * (h[0] + h[D] + h[2 * D] + h[3 * D]);
}

// ---------------------------------------------------------------------------
// GEMV over GGUF blocks as stored (the UD-IQ3_XXS / UD-Q4_K_XL files, docs/glm5.md 8.2): Y[yrow[m]][n] (=|+=)
// scale_m * sum_k W[n][k] X[xrow[m]][k] for up to 8 rows of X per threadgroup row, the same contract as
// ds4_gemv (Ds4GemvP). Each lane decodes 32-value pieces of a weight row exactly as ggml's dequantize_row_*
// orders them (a piece is one Q8_0 block, or 1/8 of a 256-value super-block) and multiplies them by X.
// Tier 0: a simdgroup per output row, 8 per threadgroup. Grid (ceil(N / 8), ceil(M / 8)), 256 threads.
static inline float gguf_h(device const uchar* p) { return float(as_type<half>(ushort(ushort(p[0]) | ushort(ushort(p[1]) << 8)))); }
static inline void gguf_k4_scale_min(int j, device const uchar* q, thread int& d, thread int& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}
// Piece c (32 values) of a row: weights into w[32].
template <int T>
static inline void gguf_piece(device const uchar* row, int c, thread float* w) {
  if (T == 8) {  // Q8_0: 34-byte blocks of 32
    device const uchar* x = row + ulong(c) * 34;
    const float d = gguf_h(x);
    for (int j = 0; j < 32; ++j) w[j] = d * float(as_type<char>(x[2 + j]));
    return;
  }
  const int b = c / 8, s = c % 8;  // super-block, piece within it
  if (T == 10) {  // Q2_K: scales[16] qs[64] d dmin
    device const uchar* x = row + ulong(b) * 84;
    const float d = gguf_h(x + 80), dmin = gguf_h(x + 82);
    const int n = s / 4, j = s % 4;
    for (int half_ = 0; half_ < 2; ++half_) {
      const uchar sc = x[n * 8 + j * 2 + half_];
      const float dl = d * float(sc & 0xF), ml = dmin * float(sc >> 4);
      for (int l = 0; l < 16; ++l) w[16 * half_ + l] = dl * float((x[16 + n * 32 + l + 16 * half_] >> (2 * j)) & 3) - ml;
    }
  } else if (T == 11) {  // Q3_K: hmask[32] qs[64] scales[12] d
    device const uchar* x = row + ulong(b) * 110;
    const float d = gguf_h(x + 108);
    uint aux[4];
    for (int i = 0; i < 3; ++i) aux[i] = uint(x[96 + 4 * i]) | (uint(x[97 + 4 * i]) << 8) | (uint(x[98 + 4 * i]) << 16) | (uint(x[99 + 4 * i]) << 24);
    const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int n = s / 4, j = s % 4;
    const uchar m = uchar(1u << (n * 4 + j));
    for (int half_ = 0; half_ < 2; ++half_) {
      const int is = n * 8 + j * 2 + half_;
      const int sc = int((aux[is / 4] >> (8 * (is % 4))) & 0xFF) - 32;
      const float dl = d * float(sc);
      for (int l = 0; l < 16; ++l) {
        const int ll = l + 16 * half_;
        w[ll] = dl * float(int((x[32 + n * 32 + ll] >> (2 * j)) & 3) - ((x[ll] & m) ? 0 : 4));
      }
    }
  } else if (T == 12 || T == 13) {  // Q4_K: d dmin scales[12] qs[128]; Q5_K: d dmin scales[12] qh[32] qs[128]
    const bool five = T == 13;
    device const uchar* x = row + ulong(b) * (five ? 176 : 144);
    const float d = gguf_h(x), dmin = gguf_h(x + 2);
    int sc, mn;
    gguf_k4_scale_min(s, x + 4, sc, mn);
    const float dl = d * float(sc), ml = dmin * float(mn);
    const int j = s / 2, hi = s % 2;
    device const uchar* ql = x + (five ? 48 : 16) + 32 * j;
    for (int l = 0; l < 32; ++l) {
      int q = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
      if (five && (x[16 + l] & (uchar(1u << (2 * j + hi))))) q += 16;
      w[l] = dl * float(q) - ml;
    }
  } else if (T == 14) {  // Q6_K: ql[128] qh[64] scales[16] d
    device const uchar* x = row + ulong(b) * 210;
    const float d = gguf_h(x + 208);
    const int n = s / 4, part = s % 4;
    device const uchar* ql = x + n * 64 + (part & 1) * 32;
    device const uchar* qh = x + 128 + n * 32;
    for (int l = 0; l < 32; ++l) {
      const int lo = (part & 2) ? (ql[l] >> 4) : (ql[l] & 0xF);
      const int q = lo | (((qh[l] >> (2 * part)) & 3) << 4);
      w[l] = d * float(as_type<char>(x[192 + n * 8 + l / 16 + 2 * part])) * float(q - 32);
    }
  } else if (T == 23) {  // IQ4_XS: d scales_h(u16) scales_l[4] qs[128]
    device const uchar* x = row + ulong(b) * 136;
    const float d = gguf_h(x);
    const uint sh = uint(x[2]) | (uint(x[3]) << 8);
    const int ls = ((x[4 + s / 2] >> (4 * (s % 2))) & 0xF) | (((sh >> (2 * s)) & 3) << 4);
    const float dl = d * float(ls - 32);
    device const uchar* qs = x + 8 + 16 * s;
    for (int j = 0; j < 16; ++j) {
      w[j] = dl * float(gguf_iq4nl[qs[j] & 0xF]);
      w[j + 16] = dl * float(gguf_iq4nl[qs[j] >> 4]);
    }
  } else if (T == 22) {  // IQ2_S: d qs[32] signs[32] qh[8] scales[8]
    device const uchar* x = row + ulong(b) * 82;
    const float d = gguf_h(x);
    const float db0 = d * (0.5f + float(x[74 + s] & 0xF)) * 0.25f, db1 = d * (0.5f + float(x[74 + s] >> 4)) * 0.25f;
    for (int l = 0; l < 4; ++l) {
      const ulong g = gguf_iq2s_grid[x[2 + 4 * s + l] | ((uint(x[66 + s]) << (8 - 2 * l)) & 0x300u)];
      const uchar sg = x[34 + 4 * s + l];
      const float dl = l < 2 ? db0 : db1;
      for (int j = 0; j < 8; ++j) w[8 * l + j] = dl * float((g >> (8 * j)) & 0xFF) * ((sg >> j) & 1 ? -1.0f : 1.0f);
    }
  } else if (T == 17) {  // IQ2_XS: d qs[32] (u16: 9-bit grid index | 7-bit sign code) scales[8]
    device const uchar* x = row + ulong(b) * 74;
    const float d = gguf_h(x);
    const float db0 = d * (0.5f + float(x[66 + s] & 0xF)) * 0.25f, db1 = d * (0.5f + float(x[66 + s] >> 4)) * 0.25f;
    for (int l = 0; l < 4; ++l) {
      const uint q = uint(x[2 + 2 * (4 * s + l)]) | (uint(x[3 + 2 * (4 * s + l)]) << 8);
      const ulong g = gguf_iq2xs_grid[q & 511];
      const uchar sg = gguf_ksigns_iq2xs[q >> 9];
      const float dl = l < 2 ? db0 : db1;
      for (int j = 0; j < 8; ++j) w[8 * l + j] = dl * float((g >> (8 * j)) & 0xFF) * ((sg >> j) & 1 ? -1.0f : 1.0f);
    }
  } else if (T == 18) {  // IQ3_XXS: d qs[64] (grid indices) scales_and_signs[32] (u32: 4 x 7-bit sign codes | 4-bit scale)
    device const uchar* x = row + ulong(b) * 98;
    const float d = gguf_h(x);
    device const uchar* ss = x + 66 + 4 * s;
    const uint aux = uint(ss[0]) | (uint(ss[1]) << 8) | (uint(ss[2]) << 16) | (uint(ss[3]) << 24);
    const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
    device const uchar* qs = x + 2 + 8 * s;
    for (int l = 0; l < 4; ++l) {
      const uchar sg = gguf_ksigns_iq2xs[(aux >> (7 * l)) & 127];
      const uint g1 = gguf_iq3xxs_grid[qs[2 * l]], g2 = gguf_iq3xxs_grid[qs[2 * l + 1]];
      for (int j = 0; j < 8; ++j)
        w[8 * l + j] = db * float(((j < 4 ? g1 : g2) >> (8 * (j & 3))) & 0xFF) * ((sg >> j) & 1 ? -1.0f : 1.0f);
    }
  } else if (T == 21) {  // IQ3_S: d qs[64] qh[8] signs[32] scales[4]
    device const uchar* x = row + ulong(b) * 110;
    const float d = gguf_h(x);
    const float db = d * float(1 + 2 * ((x[106 + s / 2] >> (4 * (s % 2))) & 0xF));
    const uint h = x[66 + s];
    for (int l = 0; l < 4; ++l) {
      const uint g1 = gguf_iq3s_grid[x[2 + 8 * s + 2 * l] | ((h << (8 - 2 * l)) & 256u)];
      const uint g2 = gguf_iq3s_grid[x[2 + 8 * s + 2 * l + 1] | ((h << (7 - 2 * l)) & 256u)];
      const uchar sg = x[74 + 4 * s + l];
      for (int j = 0; j < 8; ++j)
        w[8 * l + j] = db * float(((j < 4 ? g1 : g2) >> (8 * (j & 3))) & 0xFF) * ((sg >> j) & 1 ? -1.0f : 1.0f);
    }
  }
}
template <int T>
static inline ulong gguf_row_bytes(int K) {
  return T == 8 ? ulong(K / 32) * 34
         : ulong(K / 256) * (T == 10 ? 84 : T == 11 ? 110 : T == 12 ? 144 : T == 13 ? 176 : T == 14 ? 210 : T == 23 ? 136 : T == 22 ? 82
                             : T == 17 ? 74 : T == 18 ? 98 : 110);
}
template <int T>
kernel void gguf_gemv(device const uchar* W [[buffer(0)]], device const float* X [[buffer(2)]], device float* Y [[buffer(3)]],
                      constant Ds4GemvP& p [[buffer(4)]], device const int* xrow [[buffer(5)]], device const int* yrow [[buffer(6)]],
                      device const float* ysc [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int m0 = int(tg.y) * 8, mr = min(8, p.M - m0);
  const int n = int(tg.x) * 8 + int(sg);
  if (mr <= 0 || n >= p.N) return;
  device const uchar* row = W + ulong(n) * gguf_row_bytes<T>(p.K);
  const int xoff = p.grp ? (n / p.grp) * p.gstr : 0;  // per-head matrices stacked (attn_k_b / attn_v_b)
  float acc[8];
  for (int m = 0; m < 8; ++m) acc[m] = 0.0f;
  float w[32];
  for (int c = int(lane); c < p.K / 32; c += 32) {
    gguf_piece<T>(row, c, w);
    for (int m = 0; m < 8; ++m) {
      if (m >= mr) break;
      device const float4* xv = (device const float4*)(X + ulong(p.idx ? xrow[m0 + m] : m0 + m) * ulong(p.xs) + ulong(xoff + c * 32));
      float s = 0.0f;
      for (int j = 0; j < 8; ++j) {
        const float4 x4 = xv[j];
        s += w[4 * j] * x4.x + w[4 * j + 1] * x4.y + w[4 * j + 2] * x4.z + w[4 * j + 3] * x4.w;
      }
      acc[m] += s;
    }
  }
  for (int m = 0; m < 8; ++m) {
    if (m >= mr) break;
    const float s = simd_sum(acc[m]);
    if (lane == 0) {
      const int yi = p.idx ? yrow[m0 + m] : m0 + m;
      const float v = p.idx ? s * ysc[m0 + m] : s;
      device float* y = Y + ulong(yi) * ulong(p.ys) + ulong(n);
      *y = p.accum ? *y + v : v;
    }
  }
}
#define GGUF_GEMV(NAME, T)                                                                                          \
  template [[host_name(NAME)]] kernel void gguf_gemv<T>(device const uchar*, device const float*, device float*,  \
                                                        constant Ds4GemvP&, device const int*, device const int*,   \
                                                        device const float*, uint2, uint, uint);
GGUF_GEMV("gguf_gemv_q8_0", 8)
GGUF_GEMV("gguf_gemv_q2_k", 10)
GGUF_GEMV("gguf_gemv_q3_k", 11)
GGUF_GEMV("gguf_gemv_q4_k", 12)
GGUF_GEMV("gguf_gemv_q5_k", 13)
GGUF_GEMV("gguf_gemv_q6_k", 14)
GGUF_GEMV("gguf_gemv_iq3_s", 21)
GGUF_GEMV("gguf_gemv_iq2_s", 22)
GGUF_GEMV("gguf_gemv_iq4_xs", 23)
GGUF_GEMV("gguf_gemv_iq2_xs", 17)
GGUF_GEMV("gguf_gemv_iq3_xxs", 18)

// Token embeddings from a GGUF matrix (token_embd, quantized rows of D): H[m][s][:] = E[ids[m]] for the 4 streams.
// Grid (D / 32 pieces, rows).
template <int T>
kernel void gguf_embed(device const uchar* E [[buffer(0)]], device const int* ids [[buffer(1)]], device float* H [[buffer(2)]],
                       constant int& D [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
  const int c = int(g.x), m = int(g.y);
  if (c >= D / 32) return;
  float w[32];
  gguf_piece<T>(E + ulong(ids[m]) * gguf_row_bytes<T>(D), c, w);
  for (int s = 0; s < 4; ++s) {
    device float* h = H + (ulong(m) * 4 + ulong(s)) * ulong(D) + ulong(c * 32);
    for (int j = 0; j < 32; ++j) h[j] = w[j];
  }
}
#define GGUF_EMBED(NAME, T) \
  template [[host_name(NAME)]] kernel void gguf_embed<T>(device const uchar*, device const int*, device float*, constant int&, uint2);
GGUF_EMBED("gguf_embed_q8_0", 8)
GGUF_EMBED("gguf_embed_q4_k", 12)
GGUF_EMBED("gguf_embed_q5_k", 13)
GGUF_EMBED("gguf_embed_q6_k", 14)

// Rows of n floats from src (row stride ss) to dst (row stride ds), and dst += src (the MTP layer's plain residuals).
kernel void glm5_copy_rows(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                           constant int3& p [[buffer(2)]],  // (n, ss, ds)
                           uint2 g [[thread_position_in_grid]]) {
  if (int(g.x) >= p.x) return;
  dst[ulong(g.y) * ulong(p.z) + g.x] = src[ulong(g.y) * ulong(p.y) + g.x];
}
kernel void glm5_add_rows(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                          constant int3& p [[buffer(2)]],  // (n, ss, ds)
                          uint2 g [[thread_position_in_grid]]) {
  if (int(g.x) >= p.x) return;
  dst[ulong(g.y) * ulong(p.z) + g.x] += src[ulong(g.y) * ulong(p.y) + g.x];
}

// KDA, chunked (tier 1 of glm5_kda_recur): 8 tokens per step with 8 x 8 MMAs, the per-channel decay folded into
// the keys and queries. Within a chunk, with A_t[c] = prod_{i <= t} exp(g_i[c]) (|log A| <= 40: fp32 is safe):
//   k~ = A_t k_t,  k^ = k_t / A_t,  q~ = A_t q_t
//   L = strictly-lower(beta_t k~_t . k^_i)        D = (I + L)^-1 diag(beta) (V - K~ S)      (the deltas)
//   O = Q~ S + lower-incl(q~_t . k^_i) D           S' = diag(A_C) (S + K^T D)
// (I + L)^-1 = (I - L)(I + L^2)(I + L^4) (L nilpotent). One simdgroup per 8-row slab of a head's state
// (16 fragments of S, k x v, in registers); the 4 simdgroups of a threadgroup share the staged chunk.
// Grid (32, 128 / 8, H) threads, threadgroup (32, 4, 1).
kernel void glm5_kda_chunk(device const float* QKV [[buffer(0)]], device const float* G [[buffer(1)]],
                           device const float* Beta [[buffer(2)]], device float* S [[buffer(3)]],
                           device float* O [[buffer(4)]], constant Glm5RecurP& p [[buffer(5)]],
                           uint3 tpig [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int Dk = 128, LDS = Dk + 4;
  threadgroup float Kt[8 * LDS];  // k~
  threadgroup float Kh[8 * LDS];  // k^
  threadgroup float Qt[8 * LDS];  // q~
  threadgroup float Ac[Dk];       // A_C
  threadgroup float Bs[8];
  const int slab = int(tpig.y), h = int(tpig.z);
  const int C = p.H * 128;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  device float* st_p = S + (ulong(h) * 128 + ulong(slab) * 8) * Dk;  // the slab's 8 v rows of [v][k]
  simdgroup_float8x8 ST[16];
  for (int kk = 0; kk < 16; ++kk) simdgroup_load(ST[kk], st_p + kk * 8, Dk, ulong2(0, 0), true);
  simdgroup_float8x8 I8;
  I8.thread_elements()[0] = fm == fn ? 1.0f : 0.0f;
  I8.thread_elements()[1] = fm == fn + 1 ? 1.0f : 0.0f;
  for (int t0 = 0; t0 < p.rows; t0 += 8) {
    const int nt = min(8, p.rows - t0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < Dk) {  // one channel per thread: the cumulative decay down the chunk
      const int c = int(tid);
      float acc = 0.0f;
      for (int t = 0; t < 8; ++t) {
        float kv = 0.0f, qv = 0.0f, a = 1.0f;
        if (t < nt) {
          device const float* row = QKV + ulong(t0 + t) * ulong(p.C3);
          acc += G[ulong(t0 + t) * ulong(C) + ulong(h * 128 + c)];
          a = exp(acc);
          kv = row[C + h * 128 + c];
          qv = row[h * 128 + c];
        }
        Kt[t * LDS + c] = kv * a;
        Kh[t * LDS + c] = t < nt ? kv / a : 0.0f;
        Qt[t * LDS + c] = qv * a;
      }
      Ac[c] = exp(acc);
    }
    if (tid < 8) Bs[tid] = int(tid) < nt ? Beta[ulong(t0 + int(tid)) * ulong(p.H) + ulong(h)] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 KK = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 QK = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 WS = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 QS = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int kk = 0; kk < 16; ++kk) {
      simdgroup_float8x8 Kf, KhT, Qf;
      simdgroup_load(Kf, Kt + kk * 8, LDS);
      simdgroup_load(KhT, Kh + kk * 8, LDS, ulong2(0, 0), true);
      simdgroup_load(Qf, Qt + kk * 8, LDS);
      simdgroup_multiply_accumulate(KK, Kf, KhT, KK);
      simdgroup_multiply_accumulate(QK, Qf, KhT, QK);
      simdgroup_multiply_accumulate(WS, Kf, ST[kk], WS);
      simdgroup_multiply_accumulate(QS, Qf, ST[kk], QS);
    }
    const float b_r = Bs[fm];
    const bool r_ok = fm < nt;
    simdgroup_float8x8 L, T, P, tmp;
    L.thread_elements()[0] = fn < fm ? b_r * KK.thread_elements()[0] : 0.0f;
    L.thread_elements()[1] = fn + 1 < fm ? b_r * KK.thread_elements()[1] : 0.0f;
    T.thread_elements()[0] = I8.thread_elements()[0] - L.thread_elements()[0];
    T.thread_elements()[1] = I8.thread_elements()[1] - L.thread_elements()[1];
    P = L;
    for (int step = 0; step < 2; ++step) {
      simdgroup_multiply(tmp, P, P);
      P = tmp;
      simdgroup_multiply_accumulate(tmp, T, P, T);
      T = tmp;
    }
    simdgroup_float8x8 X, Dm;
    {
      const float2 vv = r_ok ? float2(QKV[ulong(t0 + fm) * ulong(p.C3) + ulong(2 * C + h * 128 + slab * 8 + fn)],
                                      QKV[ulong(t0 + fm) * ulong(p.C3) + ulong(2 * C + h * 128 + slab * 8 + fn + 1)])
                             : float2(0.0f);
      X.thread_elements()[0] = b_r * (vv.x - WS.thread_elements()[0]);
      X.thread_elements()[1] = b_r * (vv.y - WS.thread_elements()[1]);
    }
    simdgroup_multiply(Dm, T, X);
    simdgroup_float8x8 QKm, Om;
    QKm.thread_elements()[0] = fn <= fm ? QK.thread_elements()[0] : 0.0f;
    QKm.thread_elements()[1] = fn + 1 <= fm ? QK.thread_elements()[1] : 0.0f;
    simdgroup_multiply_accumulate(Om, QKm, Dm, QS);
    if (r_ok) {
      device float* y = O + ulong(t0 + fm) * ulong(C) + ulong(h * 128 + slab * 8 + fn);
      y[0] = Om.thread_elements()[0];
      y[1] = Om.thread_elements()[1];
    }
    for (int kk = 0; kk < 16; ++kk) {  // S' = diag(A_C) (S + K^T D): rows of a fragment are k = kk * 8 + fm
      simdgroup_float8x8 KhT;
      simdgroup_load(KhT, Kh + kk * 8, LDS, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(ST[kk], KhT, Dm, ST[kk]);
      const float a = Ac[kk * 8 + fm];
      ST[kk].thread_elements()[0] *= a;
      ST[kk].thread_elements()[1] *= a;
    }
  }
  if (p.write_state)
    for (int kk = 0; kk < 16; ++kk) simdgroup_store(ST[kk], st_p + kk * 8, Dk, ulong2(0, 0), true);
}
