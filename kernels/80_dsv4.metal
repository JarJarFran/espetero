// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ---------------------------------------------------------------------------
// DeepSeek-V4.1 (deepseek_v41) kernels, docs/deepseek_v41.md.
//
// Weights stay in the checkpoint's formats: FP8 e4m3 with one e8m0 scale per
// 32x32 block (attention, shared experts, Engram projection), FP4 e2m1 with
// one e8m0 scale per 32 inputs of a row (routed experts), bf16 (routers,
// compressors, indexer keys, head). Activations are fp32; before every
// quantized matmul they are rounded to the grid the reference quantizes them
// to (fp8 per 32 with a power-of-two scale), so the products are the
// reference's up to the summation order.
//
// Decoding a code: e4m3 bits shifted into a half give value * 2^-8 (bias 7
// against 15, subnormals included); e2m1 bits give value * 2^-14. The powers
// of two fold into the block scale.

static inline float ds4_e8m0(uchar b) { return as_type<float>(uint(b) << 23); }

// Round to the e4m3fn grid, nearest-even, saturating at 448.
static inline float ds4_round_e4m3(float v) {
  const float a = min(fabs(v), 448.0f);
  float q;
  if (a < 0.015625f) {
    q = rint(a * 512.0f) * (1.0f / 512.0f);
  } else {
    const uint e = (as_type<uint>(a) >> 23) & 0xFFu;
    const float step = as_type<float>((e - 3u) << 23);
    q = rint(a / step) * step;
  }
  return copysign(q, v);
}

// Round to the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, nearest-even, saturating.
static inline float ds4_round_e2m1(float v) {
  const float a = min(fabs(v), 6.0f);
  float q;
  if (a < 1.0f) {
    q = rint(a * 2.0f) * 0.5f;
  } else {
    const uint e = (as_type<uint>(a) >> 23) & 0xFFu;
    const float step = as_type<float>((e - 1u) << 23);
    q = rint(a / step) * step;
  }
  return copysign(q, v);
}

// 2^ceil(log2(x)) for x > 0 (the reference's fast_round_scale).
static inline float ds4_pow2_ceil(float x) {
  const uint u = as_type<uint>(x);
  int e = int((u >> 23) & 0xFFu) - 127 + ((u & 0x7FFFFFu) ? 1 : 0);
  e = clamp(e, -126, 127);
  return as_type<float>(uint(e + 127) << 23);
}

// Two e4m3 bytes (bits 0-15 of w) as a half2 of value * 2^-8.
static inline half2 ds4_e4m3x2(uint w) {
  return as_type<half2>(((w & 0x7Fu) << 7) | ((w & 0x80u) << 8) | ((w & 0x7F00u) << 15) | ((w & 0x8000u) << 16));
}
// Two e2m1 nibbles (bits 0-7 of w, low nibble first) as a half2 of value * 2^-14.
static inline half2 ds4_e2m1x2(uint w) {
  return as_type<half2>(((w & 0x7u) << 9) | ((w & 0x8u) << 12) | ((w & 0x70u) << 21) | ((w & 0x80u) << 24));
}

// ---------------------------------------------------------------------------
// Activation rounding, in place, per block of 32 along rows of K:
// mode 0: fp8 e4m3, scale 2^ceil(log2(amax / 448)), amax >= 1e-4 (act_quant).
// mode 1: fp4 e2m1, scale 2^ceil(log2(amax / 6)), amax >= 6 * 2^-126 (indexer).
// mode 2: fp4 e2m1 per 16, scale e4m3(amax / 6), amax >= 6 * 2^-9 (compressed KV).
struct Ds4QuantP {
  int rows, K, stride, mode;
};
kernel void ds4_quant(device float* x [[buffer(0)]], constant Ds4QuantP& p [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
  const int B = p.mode == 2 ? 16 : 32;
  const int nb = p.K / B;
  if (int(gid) >= p.rows * nb) return;
  const int r = int(gid) / nb, b = int(gid) % nb;
  device float* v = x + ulong(r) * ulong(p.stride) + ulong(b * B);
  float amax = 0.0f;
  for (int i = 0; i < B; ++i) amax = max(amax, fabs(v[i]));
  if (p.mode == 0) {
    const float s = ds4_pow2_ceil(max(amax, 1e-4f) * (1.0f / 448.0f));
    for (int i = 0; i < B; ++i) v[i] = ds4_round_e4m3(v[i] / s) * s;
  } else if (p.mode == 1) {
    const float s = ds4_pow2_ceil(max(amax, 6.0f * 0x1.0p-126f) * (1.0f / 6.0f));
    for (int i = 0; i < B; ++i) v[i] = ds4_round_e2m1(v[i] / s) * s;
  } else {
    const float s = ds4_round_e4m3(max(amax, 6.0f * 0x1.0p-9f) / 6.0f);
    for (int i = 0; i < B; ++i) v[i] = ds4_round_e2m1(v[i] / s) * s;
  }
}

// ---------------------------------------------------------------------------
// GEMV over up to 8 activation rows: Y[yrow[m]][n] (=|+=) scale_m * sum_k W[n][k] X[xrow[m]][k].
// FMT 0: fp8 e4m3, scales [N/32][K/32]; 1: fp4 e2m1 rows of K/2 bytes, scales
// [N][K/32]; 2: bf16; 3: fp32. A simdgroup computes one output row at a time,
// each lane reading 16-byte chunks along K. Grid: (ceil(N / 8), ceil(M / 8)),
// 256 threads (8 simdgroups, one row each per step).
struct Ds4GemvP {
  int N, K, M;
  int xs, ys;      // row strides of X and Y (floats)
  int grp, gstr;   // output rows [g*grp, (g+1)*grp) read X at column offset g*gstr (grp 0: none)
  int accum;       // 1: Y += result
  int idx;         // 1: xrow / yrow / ysc are given (expert rows)
  int nrow_tg;     // output rows per threadgroup (multiple of 8)
};
constant constexpr int kDs4GemvM = 8;

template <int FMT>
static inline void ds4_gemv_impl(device const uchar* W, device const uchar* S, device const float* X, device float* Y,
                                 constant Ds4GemvP& p, device const int* xrow, device const int* yrow,
                                 device const float* ysc, uint2 tg, uint sg, uint lane) {
  constexpr int E = FMT == 0 ? 16 : FMT == 1 ? 32 : FMT == 2 ? 8 : 4;  // elements per 16-byte chunk
  const int m0 = int(tg.y) * kDs4GemvM;
  const int mr = min(kDs4GemvM, p.M - m0);
  if (mr <= 0) return;
  device const float* xr[kDs4GemvM];
  for (int m = 0; m < kDs4GemvM; ++m) {
    const int mm = min(m, mr - 1);
    const int xi = p.idx ? xrow[m0 + mm] : m0 + mm;
    xr[m] = X + ulong(xi) * ulong(p.xs);
  }
  const int kb = p.K / 32;
  const int chunks = p.K / E;
  const ulong row_bytes = FMT == 0 ? ulong(p.K) : FMT == 1 ? ulong(p.K / 2) : FMT == 2 ? ulong(p.K) * 2 : ulong(p.K) * 4;
  const int n_end = min(p.N, int(tg.x + 1) * p.nrow_tg);
  for (int n = int(tg.x) * p.nrow_tg + int(sg); n < n_end; n += 8) {
    device const uint4* wr = (device const uint4*)(W + ulong(n) * row_bytes);
    const int xoff = p.grp ? (n / p.grp) * p.gstr : 0;
    float acc[kDs4GemvM];
    for (int m = 0; m < kDs4GemvM; ++m) acc[m] = 0.0f;
    for (int c = int(lane); c < chunks; c += 32) {
      const uint4 w = wr[c];
      float wv[E];
      float sc = 1.0f;
      if (FMT == 0) {
        sc = ds4_e8m0(S[ulong(n / 32) * ulong(kb) + ulong(c / 2)]) * 256.0f;
        const uint ws[4] = {w.x, w.y, w.z, w.w};
        for (int j = 0; j < 4; ++j) {
          const half2 a = ds4_e4m3x2(ws[j]), b = ds4_e4m3x2(ws[j] >> 16);
          wv[4 * j + 0] = float(a.x);
          wv[4 * j + 1] = float(a.y);
          wv[4 * j + 2] = float(b.x);
          wv[4 * j + 3] = float(b.y);
        }
      } else if (FMT == 1) {
        sc = ds4_e8m0(S[ulong(n) * ulong(kb) + ulong(c)]) * 16384.0f;
        const uint ws[4] = {w.x, w.y, w.z, w.w};
        for (int j = 0; j < 4; ++j)
          for (int q = 0; q < 4; ++q) {
            const half2 a = ds4_e2m1x2(ws[j] >> (8 * q));
            wv[8 * j + 2 * q] = float(a.x);
            wv[8 * j + 2 * q + 1] = float(a.y);
          }
      } else if (FMT == 2) {
        const uint ws[4] = {w.x, w.y, w.z, w.w};
        for (int j = 0; j < 4; ++j) {
          wv[2 * j] = as_type<float>(ws[j] << 16);
          wv[2 * j + 1] = as_type<float>(ws[j] & 0xFFFF0000u);
        }
      } else {
        wv[0] = as_type<float>(w.x);
        wv[1] = as_type<float>(w.y);
        wv[2] = as_type<float>(w.z);
        wv[3] = as_type<float>(w.w);
      }
      const int k0 = xoff + c * E;
      for (int m = 0; m < kDs4GemvM; ++m) {
        if (m >= mr) break;
        device const float4* xv = (device const float4*)(xr[m] + k0);
        float s = 0.0f;
        for (int j = 0; j < E / 4; ++j) {
          const float4 x4 = xv[j];
          s += wv[4 * j] * x4.x + wv[4 * j + 1] * x4.y + wv[4 * j + 2] * x4.z + wv[4 * j + 3] * x4.w;
        }
        acc[m] += s * sc;
      }
    }
    for (int m = 0; m < kDs4GemvM; ++m) {
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
}

#define DS4_GEMV(NAME, FMT)                                                                                       \
  kernel void NAME(device const uchar* W [[buffer(0)]], device const uchar* S [[buffer(1)]],                      \
                   device const float* X [[buffer(2)]], device float* Y [[buffer(3)]],                           \
                   constant Ds4GemvP& p [[buffer(4)]], device const int* xrow [[buffer(5)]],                     \
                   device const int* yrow [[buffer(6)]], device const float* ysc [[buffer(7)]],                  \
                   uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],        \
                   uint lane [[thread_index_in_simdgroup]]) {                                                    \
    ds4_gemv_impl<FMT>(W, S, X, Y, p, xrow, yrow, ysc, tg, sg, lane);                                            \
  }
DS4_GEMV(ds4_gemv_f8, 0)
DS4_GEMV(ds4_gemv_f4, 1)
DS4_GEMV(ds4_gemv_bf16, 2)
DS4_GEMV(ds4_gemv_f32, 3)

// ---------------------------------------------------------------------------
struct Ds4HcP {
  int D, iters;
  float eps, hc_eps;
};

// Routed experts of one layer in one dispatch: threadgroup (row tile, group);
// a group is one expert (its slot: pool part + byte offset) and its
// assignment rows [a0, a0 + n). Slot layout: w1 | w2 | w3 codes, then
// w1 | w2 | w3 scales. mode 0: gate and up, out rows [0, 2N) of Y[a]
// (w1 rows, then w3 rows) from X[xrow[a]]; mode 1: down, Y[a] = w2 . X[a].
struct Ds4XGroup {
  uint part, pad;
  ulong off;
  int a0, n;
};
struct Ds4XP {
  int N, K, xs, ys;
  ulong wbytes, sbytes;
  int mode, nrow_tg;
};
kernel void ds4_xgemv(device const uchar* P0 [[buffer(0)]], device const uchar* P1 [[buffer(1)]],
                      device const uchar* P2 [[buffer(2)]], device const Ds4XGroup* G [[buffer(3)]],
                      device const float* X [[buffer(4)]], device float* Y [[buffer(5)]],
                      device const int* xrow [[buffer(6)]], constant Ds4XP& p [[buffer(7)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  const Ds4XGroup g = G[tg.y];
  device const uchar* base = (g.part == 0 ? P0 : g.part == 1 ? P1 : P2) + g.off;
  const int rows_out = p.mode == 0 ? 2 * p.N : p.N;
  const int kb = p.K / 32, chunks = p.K / 32;  // fp4: 32 elements per 16-byte chunk = one scale block
  const ulong row_bytes = ulong(p.K / 2);
  const int n_end = min(rows_out, int(tg.x + 1) * p.nrow_tg);
  for (int r = int(tg.x) * p.nrow_tg + int(sg); r < n_end; r += 8) {
    int mat, n;
    if (p.mode == 0) mat = r < p.N ? 0 : 2, n = r < p.N ? r : r - p.N;
    else mat = 1, n = r;
    device const uint4* wr = (device const uint4*)(base + ulong(mat) * p.wbytes + ulong(n) * row_bytes);
    device const uchar* S = base + 3 * p.wbytes + ulong(mat) * p.sbytes + ulong(n) * ulong(kb);
    for (int m0 = 0; m0 < g.n; m0 += kDs4GemvM) {
      const int mr = min(kDs4GemvM, g.n - m0);
      device const float* xr[kDs4GemvM];
      for (int m = 0; m < kDs4GemvM; ++m) {
        const int a = g.a0 + m0 + min(m, mr - 1);
        xr[m] = X + ulong(p.mode == 0 ? xrow[a] : a) * ulong(p.xs);
      }
      float acc[kDs4GemvM];
      for (int m = 0; m < kDs4GemvM; ++m) acc[m] = 0.0f;
      for (int c = int(lane); c < chunks; c += 32) {
        const uint4 w = wr[c];
        const float sc = ds4_e8m0(S[c]) * 16384.0f;
        float wv[32];
        const uint ws[4] = {w.x, w.y, w.z, w.w};
        for (int j = 0; j < 4; ++j)
          for (int q = 0; q < 4; ++q) {
            const half2 h = ds4_e2m1x2(ws[j] >> (8 * q));
            wv[8 * j + 2 * q] = float(h.x);
            wv[8 * j + 2 * q + 1] = float(h.y);
          }
        for (int m = 0; m < kDs4GemvM; ++m) {
          if (m >= mr) break;
          device const float4* xv = (device const float4*)(xr[m] + c * 32);
          float s = 0.0f;
          for (int j = 0; j < 8; ++j) {
            const float4 x4 = xv[j];
            s += wv[4 * j] * x4.x + wv[4 * j + 1] * x4.y + wv[4 * j + 2] * x4.z + wv[4 * j + 3] * x4.w;
          }
          acc[m] += s * sc;
        }
      }
      for (int m = 0; m < kDs4GemvM; ++m) {
        if (m >= mr) break;
        const float s = simd_sum(acc[m]);
        if (lane == 0) Y[ulong(g.a0 + m0 + m) * ulong(p.ys) + ulong(r)] = s;
      }
    }
  }
}

// Hyper-connection mixes, split along the 4D-long rows: part[m][slice] = (24
// partial dots, partial sum of squares) over a 1024-float slice; ds4_hc_fin
// adds the slices and applies the split + Sinkhorn of ds4_hc_mix.
kernel void ds4_hc_part(device const float* H [[buffer(0)]], device const float* fn [[buffer(1)]],
                        device float* part [[buffer(2)]], constant Ds4HcP& p [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NM = 24, SL = 1024;
  threadgroup float red[8][NM + 1];
  const int W = 4 * p.D, ns = W / SL;
  const int m = int(tg.y), sl = int(tg.x);
  const int k4 = (sl * SL) / 4 + int(tid);  // 256 threads x float4 = one slice
  const float4 x = ((device const float4*)(H + ulong(m) * ulong(W)))[k4];
  float acc[NM];
  for (int j = 0; j < NM; ++j) acc[j] = dot(x, ((device const float4*)(fn + ulong(j) * ulong(W)))[k4]);
  float ss = dot(x, x);
  for (int j = 0; j < NM; ++j) acc[j] = simd_sum(acc[j]);
  ss = simd_sum(ss);
  if (lane == 0) {
    for (int j = 0; j < NM; ++j) red[sg][j] = acc[j];
    red[sg][NM] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < NM + 1) {
    float t = 0.0f;
    for (int s = 0; s < 8; ++s) t += red[s][tid];
    part[(ulong(m) * ulong(ns) + ulong(sl)) * (NM + 1) + tid] = t;
  }
}
kernel void ds4_hc_fin(device const float* part [[buffer(0)]], device const float* scale [[buffer(1)]],
                       device const float* base [[buffer(2)]], device float* out [[buffer(3)]],
                       constant Ds4HcP& p [[buffer(4)]], uint m [[thread_position_in_grid]]) {
  constexpr int NM = 24;
  const int W = 4 * p.D, ns = W / 1024;
  float mix[NM + 1];
  for (int j = 0; j <= NM; ++j) mix[j] = 0.0f;
  for (int s = 0; s < ns; ++s)
    for (int j = 0; j <= NM; ++j) mix[j] += part[(ulong(m) * ulong(ns) + ulong(s)) * (NM + 1) + j];
  const float r = rsqrt(mix[NM] / float(W) + p.eps);
  for (int j = 0; j < NM; ++j) mix[j] *= r;
  device float* o = out + ulong(m) * NM;
  for (int j = 0; j < 4; ++j) o[j] = 1.0f / (1.0f + exp(-(mix[j] * scale[0] + base[j]))) + p.hc_eps;
  for (int j = 0; j < 4; ++j) o[4 + j] = 2.0f / (1.0f + exp(-(mix[4 + j] * scale[1] + base[4 + j])));
  float c[4][4];
  for (int j = 0; j < 4; ++j) {
    float mx = -INFINITY;
    for (int k = 0; k < 4; ++k) {
      c[j][k] = mix[8 + 4 * j + k] * scale[2] + base[8 + 4 * j + k];
      mx = max(mx, c[j][k]);
    }
    float s = 0.0f;
    for (int k = 0; k < 4; ++k) {
      c[j][k] = exp(c[j][k] - mx);
      s += c[j][k];
    }
    for (int k = 0; k < 4; ++k) c[j][k] = c[j][k] / s + p.hc_eps;
  }
  for (int k = 0; k < 4; ++k) {
    const float cs = c[0][k] + c[1][k] + c[2][k] + c[3][k] + p.hc_eps;
    for (int j = 0; j < 4; ++j) c[j][k] /= cs;
  }
  for (int it = 1; it < p.iters; ++it) {
    for (int j = 0; j < 4; ++j) {
      const float rs = c[j][0] + c[j][1] + c[j][2] + c[j][3] + p.hc_eps;
      for (int k = 0; k < 4; ++k) c[j][k] /= rs;
    }
    for (int k = 0; k < 4; ++k) {
      const float cs = c[0][k] + c[1][k] + c[2][k] + c[3][k] + p.hc_eps;
      for (int j = 0; j < 4; ++j) c[j][k] /= cs;
    }
  }
  for (int j = 0; j < 4; ++j)
    for (int k = 0; k < 4; ++k) o[8 + 4 * j + k] = c[j][k];
}

// ---------------------------------------------------------------------------
// Token embeddings into the hc_mult copies of the residual: H[m][s][:] = E[id_m].
kernel void ds4_embed(device const int* ids [[buffer(0)]], device const ushort* E [[buffer(1)]],
                      device float* H [[buffer(2)]], constant int2& p [[buffer(3)]],  // (D, hc)
                      uint2 g [[thread_position_in_grid]]) {
  const int d = int(g.x), m = int(g.y);
  if (d >= p.x) return;
  const float v = bf2f(E[ulong(ids[m]) * ulong(p.x) + ulong(d)]);
  for (int s = 0; s < p.y; ++s) H[(ulong(m) * ulong(p.y) + ulong(s)) * ulong(p.x) + ulong(d)] = v;
}

// ---------------------------------------------------------------------------
// Hyper-connection coefficients of one block (hc_mixes + hc_split_sinkhorn):
// mixes = fn . flat(H_m) * rsqrt(mean(H_m^2) + eps) (24 = (2 + 4) * 4 outputs),
// pre = sigmoid(mix * s0 + b) + hc_eps, post = 2 sigmoid(mix * s1 + b),
// comb = Sinkhorn(softmax rows of mix * s2 + b). Out row: [pre 4 | post 4 | comb 16].
// One threadgroup (256) per row.
kernel void ds4_hc_mix(device const float* H [[buffer(0)]], device const float* fn [[buffer(1)]],
                       device const float* scale [[buffer(2)]], device const float* base [[buffer(3)]],
                       device float* out [[buffer(4)]], constant Ds4HcP& p [[buffer(5)]],
                       uint m [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NM = 24;
  threadgroup float red[8][NM + 1];
  threadgroup float mix[NM];
  const int W = 4 * p.D;
  device const float4* h = (device const float4*)(H + ulong(m) * ulong(W));
  float acc[NM];
  for (int j = 0; j < NM; ++j) acc[j] = 0.0f;
  float ss = 0.0f;
  for (int k = int(tid); k < W / 4; k += 256) {
    const float4 x = h[k];
    ss += dot(x, x);
    for (int j = 0; j < NM; ++j) acc[j] += dot(x, ((device const float4*)(fn + ulong(j) * ulong(W)))[k]);
  }
  for (int j = 0; j < NM; ++j) acc[j] = simd_sum(acc[j]);
  ss = simd_sum(ss);
  if (lane == 0) {
    for (int j = 0; j < NM; ++j) red[sg][j] = acc[j];
    red[sg][NM] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < NM + 1) {
    float t = 0.0f;
    for (int s = 0; s < 8; ++s) t += red[s][tid];
    red[0][tid] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    const float r = rsqrt(red[0][NM] / float(W) + p.eps);
    for (int j = 0; j < NM; ++j) mix[j] = red[0][j] * r;
    device float* o = out + ulong(m) * NM;
    for (int j = 0; j < 4; ++j) o[j] = 1.0f / (1.0f + exp(-(mix[j] * scale[0] + base[j]))) + p.hc_eps;
    for (int j = 0; j < 4; ++j) o[4 + j] = 2.0f / (1.0f + exp(-(mix[4 + j] * scale[1] + base[4 + j])));
    float c[4][4];
    for (int j = 0; j < 4; ++j) {
      float mx = -INFINITY;
      for (int k = 0; k < 4; ++k) {
        c[j][k] = mix[8 + 4 * j + k] * scale[2] + base[8 + 4 * j + k];
        mx = max(mx, c[j][k]);
      }
      float s = 0.0f;
      for (int k = 0; k < 4; ++k) {
        c[j][k] = exp(c[j][k] - mx);
        s += c[j][k];
      }
      for (int k = 0; k < 4; ++k) c[j][k] = c[j][k] / s + p.hc_eps;
    }
    for (int k = 0; k < 4; ++k) {
      const float cs = c[0][k] + c[1][k] + c[2][k] + c[3][k] + p.hc_eps;
      for (int j = 0; j < 4; ++j) c[j][k] /= cs;
    }
    for (int it = 1; it < p.iters; ++it) {
      for (int j = 0; j < 4; ++j) {
        const float rs = c[j][0] + c[j][1] + c[j][2] + c[j][3] + p.hc_eps;
        for (int k = 0; k < 4; ++k) c[j][k] /= rs;
      }
      for (int k = 0; k < 4; ++k) {
        const float cs = c[0][k] + c[1][k] + c[2][k] + c[3][k] + p.hc_eps;
        for (int j = 0; j < 4; ++j) c[j][k] /= cs;
      }
    }
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 4; ++k) o[8 + 4 * j + k] = c[j][k];
  }
}

// X[m] = RMSNorm_w(sum_j pre[m][j] H[m][j]) (hc_pre then attn_norm / ffn_norm);
// pre is the first 4 floats of a 24-float mix row. w null: no norm (head input uses norm too).
kernel void ds4_hc_pre(device const float* H [[buffer(0)]], device const float* mix [[buffer(1)]],
                       device const float* w [[buffer(2)]], device float* X [[buffer(3)]],
                       constant Ds4HcP& p [[buffer(4)]], uint m [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8];
  device const float* h = H + ulong(m) * ulong(4 * p.D);
  device const float* pr = mix + ulong(m) * 24;
  const float p0 = pr[0], p1 = pr[1], p2 = pr[2], p3 = pr[3];
  device float* x = X + ulong(m) * ulong(p.D);
  float ss = 0.0f;
  for (int d = int(tid); d < p.D; d += 256) {
    const float v = p0 * h[d] + p1 * h[p.D + d] + p2 * h[2 * p.D + d] + p3 * h[3 * p.D + d];
    x[d] = v;
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0.0f;
  for (int s = 0; s < 8; ++s) t += red[s];
  const float r = rsqrt(t / float(p.D) + p.eps);
  for (int d = int(tid); d < p.D; d += 256) x[d] = x[d] * r * w[d];
}

// H[m][k] = post[k] * Y[m] + sum_j comb[j][k] H[m][j], in place.
kernel void ds4_hc_post(device float* H [[buffer(0)]], device const float* Y [[buffer(1)]],
                        device const float* mix [[buffer(2)]], constant Ds4HcP& p [[buffer(3)]],
                        uint2 g [[thread_position_in_grid]]) {
  const int d = int(g.x), m = int(g.y);
  if (d >= p.D) return;
  device float* h = H + ulong(m) * ulong(4 * p.D) + ulong(d);
  device const float* c = mix + ulong(m) * 24;
  const float y = Y[ulong(m) * ulong(p.D) + ulong(d)];
  const float r0 = h[0], r1 = h[p.D], r2 = h[2 * p.D], r3 = h[3 * p.D];
  for (int k = 0; k < 4; ++k)
    h[k * p.D] = c[4 + k] * y + c[8 + k] * r0 + c[12 + k] * r1 + c[16 + k] * r2 + c[20 + k] * r3;
}

// ---------------------------------------------------------------------------
// Row RMSNorm in place over `n` floats per row: x = x * rsqrt(mean(x^2) + eps) * w,
// for `heads` vectors of n per row (stride n). One simdgroup per vector.
struct Ds4NormP {
  int rows, heads, n, stride;
  float eps;
};
kernel void ds4_rmsnorm(device float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                        constant Ds4NormP& p [[buffer(2)]], uint gid [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int v = int(gid) * 8 + int(sg);
  if (v >= p.rows * p.heads) return;
  const int r = v / p.heads, h = v % p.heads;
  device float* a = x + ulong(r) * ulong(p.stride) + ulong(h) * ulong(p.n);
  float ss = 0.0f;
  for (int i = int(lane); i < p.n; i += 32) ss += a[i] * a[i];
  ss = simd_sum(ss);
  const float rs = rsqrt(ss / float(p.n) + p.eps);
  for (int i = int(lane); i < p.n; i += 32) a[i] = a[i] * rs * w[i];
}

// RoPE on the last 2 * npair dims of `heads` vectors of `hd` per row, adjacent
// pairs as complex numbers rotated by pos * inv_freq[i] (inverse: conjugate).
struct Ds4RopeP {
  int rows, heads, hd, stride, npair, inverse;
};
kernel void ds4_rope(device float* x [[buffer(0)]], device const int* pos [[buffer(1)]],
                     device const float* inv_freq [[buffer(2)]], constant Ds4RopeP& p [[buffer(3)]],
                     uint3 g [[thread_position_in_grid]]) {
  const int i = int(g.x), h = int(g.y), r = int(g.z);
  if (i >= p.npair || h >= p.heads || r >= p.rows) return;
  device float* a = x + ulong(r) * ulong(p.stride) + ulong(h) * ulong(p.hd) + ulong(p.hd - 2 * p.npair + 2 * i);
  const float ang = float(pos[r]) * inv_freq[i];
  const float c = precise::cos(ang), s = precise::sin(ang) * (p.inverse ? -1.0f : 1.0f);
  const float x0 = a[0], x1 = a[1];
  a[0] = x0 * c - x1 * s;
  a[1] = x0 * s + x1 * c;
}

// ---------------------------------------------------------------------------
// Sparse attention of one query row over its window and compressed keys
// (MQA: one 512-wide latent per key serves as K and V). Simdgroup = head,
// lane = 16 dims. Keys: window positions max(0, pos - win + 1) .. pos, read
// from the pass rows (>= p0) or the slot's ring; then compressed entries
// (dense [0, nvis) when idx is null for that row, else idx[m][0 .. 512)).
struct Ds4AttnP {
  int M, H, win, p0;
  int ratio;       // 0: window only
  int topk;        // index entries per row (0: dense compressed)
  int n_comp;      // compressed entries cached (for dense)
  float scale;
};
// One key of the online softmax: running max mx, denominator l, output o.
static inline void ds4_attn_step(thread float& mx, thread float& l, thread float* o, float s, thread const float* k) {
  const float mn = max(mx, s);
  const float cr = exp(mx - mn), e = exp(s - mn);
  l = l * cr + e;
  for (int i = 0; i < 16; ++i) o[i] = o[i] * cr + e * k[i];
  mx = mn;
}
kernel void ds4_attn(device const float* Q [[buffer(0)]], device const float* KVp [[buffer(1)]],
                     device const half* ring [[buffer(2)]], device const half* comp [[buffer(3)]],
                     device const int* idx [[buffer(4)]], device const float* sink [[buffer(5)]],
                     device float* O [[buffer(6)]], device const int* pos [[buffer(7)]],
                     constant Ds4AttnP& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int HD = 512, PL = 16;
  const int m = int(tg.y), h = int(tg.x) * 8 + int(sg);
  if (m >= p.M || h >= p.H) return;
  const int ps = pos[m];
  device const float* q = Q + (ulong(m) * ulong(p.H) + ulong(h)) * HD + ulong(lane) * PL;
  float qv[PL], o[PL];
  for (int i = 0; i < PL; ++i) {
    qv[i] = q[i] * p.scale;
    o[i] = 0.0f;
  }
  float mx = -1e30f, l = 0.0f;
  const int w0 = max(0, ps - p.win + 1);
  float kv[PL];
  for (int t = w0; t <= ps; ++t) {
    if (t >= p.p0) {
      device const float4* k4 = (device const float4*)(KVp + ulong(t - p.p0) * HD + ulong(lane) * PL);
      for (int j = 0; j < PL / 4; ++j) {
        const float4 v = k4[j];
        kv[4 * j] = v.x, kv[4 * j + 1] = v.y, kv[4 * j + 2] = v.z, kv[4 * j + 3] = v.w;
      }
    } else {
      device const half4* k4 = (device const half4*)(ring + ulong(t % p.win) * HD + ulong(lane) * PL);
      for (int j = 0; j < PL / 4; ++j) {
        const float4 v = float4(k4[j]);
        kv[4 * j] = v.x, kv[4 * j + 1] = v.y, kv[4 * j + 2] = v.z, kv[4 * j + 3] = v.w;
      }
    }
    float s = 0.0f;
    for (int i = 0; i < PL; ++i) s += qv[i] * kv[i];
    ds4_attn_step(mx, l, o, simd_sum(s), kv);
  }
  if (p.ratio > 0) {
    const int nvis = (ps + 1) / p.ratio;
    device const int* ix = idx + ulong(m) * ulong(max(p.topk, 1));
    const int n = p.topk ? p.topk : nvis;
    for (int i = 0; i < n; ++i) {
      const int j = p.topk ? ix[i] : i;
      if (j < 0 || j >= nvis) continue;
      device const half4* k4 = (device const half4*)(comp + ulong(j) * HD + ulong(lane) * PL);
      for (int u = 0; u < PL / 4; ++u) {
        const float4 v = float4(k4[u]);
        kv[4 * u] = v.x, kv[4 * u + 1] = v.y, kv[4 * u + 2] = v.z, kv[4 * u + 3] = v.w;
      }
      float s = 0.0f;
      for (int u = 0; u < PL; ++u) s += qv[u] * kv[u];
      ds4_attn_step(mx, l, o, simd_sum(s), kv);
    }
  }
  l += exp(sink[h] - mx);
  device float* out = O + (ulong(m) * ulong(p.H) + ulong(h)) * HD + ulong(lane) * PL;
  for (int i = 0; i < PL; ++i) out[i] = o[i] / l;
}

// Copy rows of fp32 into a half ring / cache: dst[(base + r) % mod (or base + r)] = src[r].
struct Ds4StoreP {
  int rows, n, src_stride, base, mod, src_row0;
};
kernel void ds4_store_half(device const float* src [[buffer(0)]], device half* dst [[buffer(1)]],
                           constant Ds4StoreP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const int i = int(g.x), r = int(g.y);
  if (i >= p.n || r >= p.rows) return;
  const int t = p.base + r;
  const int d = p.mod ? t % p.mod : t;
  dst[ulong(d) * ulong(p.n) + ulong(i)] = half(src[ulong(p.src_row0 + r) * ulong(p.src_stride) + ulong(i)]);
}

// ---------------------------------------------------------------------------
// Compressor pooling (ratio 2): for every pass row whose position closes a
// group ((pos + 1) % ratio == 0), latent[g] = sum over the group's tokens of
// kv * softmax(score) per channel; members before the pass come from the
// slot's pending rows. Out: lat [groups][n] (group order = pass order).
struct Ds4CompP {
  int M, n, ratio, p0, g0;  // g0: first group index of the pass
};
kernel void ds4_compress(device const float* KV [[buffer(0)]], device const float* SC [[buffer(1)]],
                         device const float* pend_kv [[buffer(2)]], device const float* pend_sc [[buffer(3)]],
                         device float* lat [[buffer(4)]], constant Ds4CompP& p [[buffer(5)]],
                         uint2 g [[thread_position_in_grid]]) {
  const int c = int(g.x), gi = int(g.y);
  if (c >= p.n) return;
  const int G = gi + p.g0;               // absolute group
  const int t0 = G * p.ratio;
  if (t0 + p.ratio - 1 >= p.p0 + p.M) return;
  float kv[8], sc[8];
  float mx = -INFINITY;
  for (int i = 0; i < p.ratio; ++i) {
    const int t = t0 + i;
    if (t >= p.p0) {
      kv[i] = KV[ulong(t - p.p0) * ulong(p.n) + ulong(c)];
      sc[i] = SC[ulong(t - p.p0) * ulong(p.n) + ulong(c)];
    } else {
      kv[i] = pend_kv[ulong(t % p.ratio) * ulong(p.n) + ulong(c)];
      sc[i] = pend_sc[ulong(t % p.ratio) * ulong(p.n) + ulong(c)];
    }
    mx = max(mx, sc[i]);
  }
  float s = 0.0f, v = 0.0f;
  for (int i = 0; i < p.ratio; ++i) {
    const float e = exp(sc[i] - mx);
    s += e;
    v += kv[i] * e;
  }
  lat[ulong(gi) * ulong(p.n) + ulong(c)] = v / s;
}

// ---------------------------------------------------------------------------
// Indexer scores: S[m][j] = sum_h w[m][h] * relu(q[m][h] . k[j]) for the
// compressed entries visible to row m (j < (pos + 1) / ratio), -inf beyond.
// Threadgroup: one row, 64 entries; q staged in threadgroup memory.
struct Ds4IdxP {
  int M, nh, hd, ratio, ncols;  // ncols: row stride of S
};
kernel void ds4_index_score(device const float* Qi [[buffer(0)]], device const float* Wi [[buffer(1)]],
                            device const half* Kc [[buffer(2)]], device float* S [[buffer(3)]],
                            device const int* pos [[buffer(4)]], constant Ds4IdxP& p [[buffer(5)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float q[32 * 128];
  threadgroup float w[32];
  const int m = int(tg.y);
  const int nq = p.nh * p.hd;
  for (int i = int(tid); i < nq; i += 64) q[i] = Qi[ulong(m) * ulong(nq) + ulong(i)];
  if (int(tid) < p.nh) w[tid] = Wi[ulong(m) * ulong(p.nh) + ulong(tid)];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int j = int(tg.x) * 64 + int(tid);
  if (j >= p.ncols) return;
  const int nvis = (pos[m] + 1) / p.ratio;
  device float* out = S + ulong(m) * ulong(p.ncols) + ulong(j);
  if (j >= nvis) {
    *out = -INFINITY;
    return;
  }
  float d[32];
  for (int h = 0; h < 32; ++h) d[h] = 0.0f;
  device const half4* k4 = (device const half4*)(Kc + ulong(j) * ulong(p.hd));
  for (int i = 0; i < p.hd / 4; ++i) {
    const float4 v = float4(k4[i]);
    for (int h = 0; h < p.nh; ++h) {
      threadgroup const float4* qh = (threadgroup const float4*)(q + h * p.hd);
      d[h] += dot(qh[i], v);
    }
  }
  float tot = 0.0f;
  for (int h = 0; h < p.nh; ++h) tot += max(d[h], 0.0f) * w[h];
  *out = tot;
}

// Top-k of each score row (k entries, -1 padded when fewer are visible),
// by a bitwise threshold over order-preserving keys; ties at the threshold go
// to the lowest positions. One threadgroup of 1024 per row.
struct Ds4TopkP {
  int ncols, k, stride, ratio;
};
static inline uint ds4_key(float x) {
  const uint u = as_type<uint>(x);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
kernel void ds4_topk(device const float* S [[buffer(0)]], device int* out [[buffer(1)]],
                     device const int* pos [[buffer(2)]], constant Ds4TopkP& p [[buffer(3)]],
                     uint m [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int red[32];
  threadgroup atomic_int cnt;
  device const float* s = S + ulong(m) * ulong(p.stride);
  device int* o = out + ulong(m) * ulong(p.k);
  const int nvis = min(p.ncols, (pos[m] + 1) / p.ratio);
  if (nvis <= p.k) {  // everything visible
    for (int i = int(tid); i < p.k; i += 1024) o[i] = i < nvis ? i : -1;
    return;
  }
  uint t = 0;
  for (int bit = 31; bit >= 0; --bit) {
    const uint cand = t | (1u << bit);
    int c = 0;
    for (int j = int(tid); j < nvis; j += 1024) c += ds4_key(s[j]) >= cand ? 1 : 0;
    c = simd_sum(c);
    if (lane == 0) red[sg] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int tot = 0;
    for (int q = 0; q < 32; ++q) tot += red[q];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tot >= p.k) t = cand;
  }
  if (tid == 0) atomic_store_explicit(&cnt, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = int(tid); j < nvis; j += 1024)
    if (ds4_key(s[j]) > t) o[atomic_fetch_add_explicit(&cnt, 1, memory_order_relaxed)] = j;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {  // ties, lowest positions first
    int n = atomic_load_explicit(&cnt, memory_order_relaxed);
    for (int j = 0; j < nvis && n < p.k; ++j)
      if (ds4_key(s[j]) == t) o[n++] = j;
    for (; n < p.k; ++n) o[n] = -1;
  }
}

// Candidate blocks (the indexer's first level, candidate_source_layer): per
// row, blocks of `bs` visible entries scored by their best entry, the block of
// the newest entry pinned in, the best `kb` kept (ties: lowest blocks).
// keep[m][b] = 1 for kept blocks. One threadgroup of 1024 per row.
struct Ds4CandP {
  int ncols, stride, bs, kb, ratio, cstride;
};
kernel void ds4_cand(device const float* S [[buffer(0)]], device float* BS [[buffer(1)]], device uchar* keep [[buffer(2)]],
                     device const int* pos [[buffer(3)]], constant Ds4CandP& p [[buffer(4)]],
                     uint m [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int red[32];
  threadgroup atomic_int cnt;
  device const float* s = S + ulong(m) * ulong(p.stride);
  device float* bs = BS + ulong(m) * ulong(p.cstride);
  device uchar* k = keep + ulong(m) * ulong(p.cstride);
  const int nvis = min(p.ncols, (pos[m] + 1) / p.ratio);
  const int nb = (nvis + p.bs - 1) / p.bs, last = (nvis - 1) / p.bs;
  for (int b = int(tid); b < p.cstride; b += 1024) {
    float mx = -INFINITY;
    if (b < nb)
      for (int j = b * p.bs; j < min(nvis, b * p.bs + p.bs); ++j) mx = max(mx, s[j]);
    bs[b] = b == last ? INFINITY : mx;
    k[b] = (b < nb && nb <= p.kb) ? 1 : 0;
  }
  if (nb <= p.kb) return;
  threadgroup_barrier(mem_flags::mem_device);
  uint t = 0;
  for (int bit = 31; bit >= 0; --bit) {
    const uint cand = t | (1u << bit);
    int c = 0;
    for (int b = int(tid); b < nb; b += 1024) c += ds4_key(bs[b]) >= cand ? 1 : 0;
    c = simd_sum(c);
    if (lane == 0) red[sg] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int tot = 0;
    for (int q = 0; q < 32; ++q) tot += red[q];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tot >= p.kb) t = cand;
  }
  if (tid == 0) atomic_store_explicit(&cnt, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int b = int(tid); b < nb; b += 1024)
    if (ds4_key(bs[b]) > t) {
      k[b] = 1;
      atomic_fetch_add_explicit(&cnt, 1, memory_order_relaxed);
    }
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) {
    int n = atomic_load_explicit(&cnt, memory_order_relaxed);
    for (int b = 0; b < nb && n < p.kb; ++b)
      if (ds4_key(bs[b]) == t) k[b] = 1, ++n;
  }
}
// The second level: entries outside the row's kept blocks score -inf.
kernel void ds4_cand_mask(device float* S [[buffer(0)]], device const uchar* keep [[buffer(1)]],
                          constant Ds4CandP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const int j = int(g.x), m = int(g.y);
  if (j >= p.ncols) return;
  if (!keep[ulong(m) * ulong(p.cstride) + ulong(j / p.bs)]) S[ulong(m) * ulong(p.stride) + ulong(j)] = -INFINITY;
}

// ---------------------------------------------------------------------------
// MoE routing: scores = sqrt(softplus(logits)), experts = top-k(scores + bias),
// weights = scores / sum * route_scale. One simdgroup per row.
struct Ds4RouteP {
  int M, E, k;
  float route_scale;
};
kernel void ds4_route(device const float* L [[buffer(0)]], device const float* bias [[buffer(1)]],
                      device int* sel [[buffer(2)]], device float* wout [[buffer(3)]],
                      constant Ds4RouteP& p [[buffer(4)]], uint m [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
  constexpr int PER = 16;  // up to 512 experts
  float sc[PER], b[PER];
  for (int i = 0; i < PER; ++i) {
    const int e = int(lane) + 32 * i;
    if (e < p.E) {
      const float x = L[ulong(m) * ulong(p.E) + ulong(e)];
      const float sp = x > 20.0f ? x : (x < -15.0f ? exp(x) : log(1.0f + exp(x)));
      sc[i] = sqrt(sp);
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
      wout[ulong(m) * ulong(p.k) + ulong(r)] = (p.k > 1 ? cw[r] / (tot + 1e-20f) : cw[r]) * p.route_scale;
    }
}

// SwiGLU with the training clamps: out = silu(min(g, lim)) * clamp(u, -lim, lim)
// (lim 0: no clamp); g and u side by side or in two buffers of the same layout.
struct Ds4SwiP {
  int rows, n, gs, us, os;
  float lim;
};
kernel void ds4_swiglu(device const float* G [[buffer(0)]], device const float* U [[buffer(1)]],
                       device float* O [[buffer(2)]], constant Ds4SwiP& p [[buffer(3)]],
                       uint2 g [[thread_position_in_grid]]) {
  const int i = int(g.x), r = int(g.y);
  if (i >= p.n || r >= p.rows) return;
  float a = G[ulong(r) * ulong(p.gs) + ulong(i)], u = U[ulong(r) * ulong(p.us) + ulong(i)];
  if (p.lim > 0.0f) {
    u = clamp(u, -p.lim, p.lim);
    a = min(a, p.lim);
  }
  O[ulong(r) * ulong(p.os) + ulong(i)] = a / (1.0f + exp(-a)) * u;
}

// Row-scaled SwiGLU for expert rows: like ds4_swiglu, times w[r].
kernel void ds4_swiglu_w(device const float* G [[buffer(0)]], device const float* U [[buffer(1)]],
                         device float* O [[buffer(2)]], constant Ds4SwiP& p [[buffer(3)]],
                         device const float* w [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  const int i = int(g.x), r = int(g.y);
  if (i >= p.n || r >= p.rows) return;
  float a = G[ulong(r) * ulong(p.gs) + ulong(i)], u = U[ulong(r) * ulong(p.us) + ulong(i)];
  if (p.lim > 0.0f) {
    u = clamp(u, -p.lim, p.lim);
    a = min(a, p.lim);
  }
  O[ulong(r) * ulong(p.os) + ulong(i)] = a / (1.0f + exp(-a)) * u * w[r];
}

// Y[m] += sum_i P[amap[m][i]] over the row's k expert rows; then the
// staged expert copies: dst[i] = src[i] (16-byte words).
kernel void ds4_moe_sum(device const float* P [[buffer(0)]], device float* Y [[buffer(1)]],
                        device const int* amap [[buffer(2)]], constant int2& p [[buffer(3)]],  // (D, k)
                        uint2 g [[thread_position_in_grid]]) {
  const int d = int(g.x), m = int(g.y);
  if (d >= p.x) return;
  float s = Y[ulong(m) * ulong(p.x) + ulong(d)];
  for (int i = 0; i < p.y; ++i) s += P[ulong(amap[m * p.y + i]) * ulong(p.x) + ulong(d)];
  Y[ulong(m) * ulong(p.x) + ulong(d)] = s;
}
kernel void ds4_copy(device const uint4* src [[buffer(0)]], device uint4* dst [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Engram gate: KV[m] = [key 4 x D | value D]; per copy j:
// dot = sum_d H_j wqk_j key_j * rsqrt(mean H_j^2 + eps) rsqrt(mean key_j^2 + eps) / sqrt(D),
// gate = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))), H_j += gate * value.
// One threadgroup of 4 simdgroups x 32 per row... 256 threads, 2 simdgroups per copy.
kernel void ds4_engram(device float* H [[buffer(0)]], device const float* KV [[buffer(1)]],
                       device const float* wqk [[buffer(2)]], constant Ds4HcP& p [[buffer(3)]],
                       uint m [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8][3];
  threadgroup float gate[4];
  const int D = p.D;
  const int j = int(sg) / 2, half_ = int(sg) % 2;
  device float* h = H + ulong(m) * ulong(4 * D) + ulong(j) * ulong(D);
  device const float* kv = KV + ulong(m) * ulong(5 * D);
  device const float* key = kv + ulong(j) * ulong(D);
  device const float* w = wqk + ulong(j) * ulong(D);
  float hh = 0.0f, kk = 0.0f, hk = 0.0f;
  for (int d = half_ * 32 + int(lane); d < D; d += 64) {
    const float a = h[d], b = key[d];
    hh += a * a;
    kk += b * b;
    hk += a * w[d] * b;
  }
  hh = simd_sum(hh);
  kk = simd_sum(kk);
  hk = simd_sum(hk);
  if (lane == 0) red[sg][0] = hh, red[sg][1] = kk, red[sg][2] = hk;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 4) {
    const float a = red[2 * tid][0] + red[2 * tid + 1][0], b = red[2 * tid][1] + red[2 * tid + 1][1],
                c = red[2 * tid][2] + red[2 * tid + 1][2];
    const float dot = c * rsqrt(a / float(D) + p.eps) * rsqrt(b / float(D) + p.eps) * rsqrt(float(D));
    const float sq = copysign(sqrt(max(fabs(dot), 1e-6f)), dot);
    gate[tid] = 1.0f / (1.0f + exp(-sq));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device const float* val = kv + ulong(4) * ulong(D);
  for (int i = int(tid); i < 4 * D; i += 256) {
    const int jj = i / D, d = i % D;
    H[ulong(m) * ulong(4 * D) + ulong(i)] += gate[jj] * val[d];
  }
}

// Engram rows as stored ([rows][ncol][hd codes | hd / 32 scales]) to fp32 [rows][ncol * hd].
struct Ds4ERowsP {
  int rows, ncol, hd;
};
kernel void ds4_engram_rows(device const uchar* R [[buffer(0)]], device float* E [[buffer(1)]],
                            constant Ds4ERowsP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const int j = int(g.x), m = int(g.y);
  const int hd = p.hd, rb = hd + hd / 32;
  if (j >= p.ncol * hd || m >= p.rows) return;
  const int col = j / hd, k = j % hd;
  device const uchar* row = R + (ulong(m) * ulong(p.ncol) + ulong(col)) * ulong(rb);
  const uint b = row[k];
  const half h = as_type<half>(ushort(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8)));
  E[ulong(m) * ulong(p.ncol * hd) + ulong(j)] = float(h) * 256.0f * ds4_e8m0(row[hd + k / 32]);
}

// Scale a buffer of floats: x[i] *= s.
kernel void ds4_scale(device float* x [[buffer(0)]], constant float2& p [[buffer(1)]],  // (n, s)
                      uint i [[thread_position_in_grid]]) {
  if (int(i) < int(p.x)) x[i] *= p.y;
}

// ---------------------------------------------------------------------------
// GEMM for many rows (prefill): Y = X W^T on bfloat simdgroup matrices with
// fp32 accumulators. bfloat holds every fp8-grid activation and every fp8 /
// fp4 weight times its power-of-two scale exactly, so the products are the
// reference's. Tile 64 x 64 x 32, 4 simdgroups of 32 x 32.
// Dense (FMT 0 fp8, 32 x 32 block scales): grid (N / 64, ceil(M / 64));
// grp/gstr as in ds4_gemv. Experts (fp4): grid (N_out / 64, tiles), tile t =
// (group, first row) from tiles[t].
constant constexpr int kGT = 64, kGK = 32, kGKP = 40;  // padded K stride (bank conflicts)

struct Ds4GemmP {
  int M, N, K, xs, ys, grp, gstr, accum;
};

static inline void ds4_gemm_core(threadgroup bfloat* As, threadgroup bfloat* Bs, thread simdgroup_float8x8* acc,
                                 uint sg) {
  const int sm = int(sg / 2) * 32, sn = int(sg % 2) * 32;
  for (int kk = 0; kk < kGK; kk += 8) {
    simdgroup_bfloat8x8 a[4], b[4];
    for (int i = 0; i < 4; ++i) simdgroup_load(a[i], As + (sm + 8 * i) * kGKP + kk, kGKP);
    for (int j = 0; j < 4; ++j) simdgroup_load(b[j], Bs + (sn + 8 * j) * kGKP + kk, kGKP, ulong2(0, 0), true);
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i * 4 + j], a[i], b[j], acc[i * 4 + j]);
  }
}

// Accumulators to Y rows base + r (r < rows), via threadgroup memory.
static inline void ds4_gemm_store(threadgroup float* Cs, thread simdgroup_float8x8* acc, uint sg, uint tid,
                                  device float* Y, int ys, int base, int rows, int n0, int accum) {
  const int sm = int(sg / 2) * 32, sn = int(sg % 2) * 32;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) simdgroup_store(acc[i * 4 + j], Cs + (sm + 8 * i) * kGT + sn + 8 * j, kGT);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = int(tid); e < kGT * kGT; e += 128) {
    const int r = e / kGT, c = e % kGT;
    if (r >= rows) continue;
    device float* y = Y + ulong(base + r) * ulong(ys) + ulong(n0 + c);
    *y = accum ? *y + Cs[e] : Cs[e];
  }
}

kernel void ds4_gemm_f8(device const uchar* W [[buffer(0)]], device const uchar* S [[buffer(1)]],
                        device const float* X [[buffer(2)]], device float* Y [[buffer(3)]],
                        constant Ds4GemmP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup bfloat As[kGT * kGKP], Bs[kGT * kGKP];
  threadgroup float Cs[kGT * kGT];
  const int n0 = int(tg.x) * kGT, m0 = int(tg.y) * kGT;
  const int xoff = p.grp ? (n0 / p.grp) * p.gstr : 0;
  const int kb = p.K / 32;
  simdgroup_float8x8 acc[16];
  for (int i = 0; i < 16; ++i) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  // loaders: A 64 x 32 floats (16 per thread), B 64 rows x 32 bytes (one uint4 per thread)
  const int ar = int(tid) / 2, ac = (int(tid) % 2) * 16;
  const int br = int(tid) / 2, bc = (int(tid) % 2) * 16;
  const bool arow = m0 + ar < p.M;
  device const float* xa = X + ulong(min(m0 + ar, p.M - 1)) * ulong(p.xs) + ulong(xoff + ac);
  device const uchar* wb = W + ulong(n0 + br) * ulong(p.K) + ulong(bc);
  device const uchar* sb = S + ulong((n0 + br) / 32) * ulong(kb);
  for (int k0 = 0; k0 < p.K; k0 += kGK) {
    for (int j = 0; j < 16; j += 4) {
      const float4 v = arow ? *(device const float4*)(xa + k0 + j) : float4(0.0f);
      As[ar * kGKP + ac + j] = bfloat(v.x);
      As[ar * kGKP + ac + j + 1] = bfloat(v.y);
      As[ar * kGKP + ac + j + 2] = bfloat(v.z);
      As[ar * kGKP + ac + j + 3] = bfloat(v.w);
    }
    {
      const uint4 w = *(device const uint4*)(wb + k0);
      const float sc = ds4_e8m0(sb[k0 / 32]) * 256.0f;
      const uint ws[4] = {w.x, w.y, w.z, w.w};
      for (int j = 0; j < 4; ++j) {
        const half2 a = ds4_e4m3x2(ws[j]), b = ds4_e4m3x2(ws[j] >> 16);
        Bs[br * kGKP + bc + 4 * j] = bfloat(float(a.x) * sc);
        Bs[br * kGKP + bc + 4 * j + 1] = bfloat(float(a.y) * sc);
        Bs[br * kGKP + bc + 4 * j + 2] = bfloat(float(b.x) * sc);
        Bs[br * kGKP + bc + 4 * j + 3] = bfloat(float(b.y) * sc);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ds4_gemm_core(As, Bs, acc, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  ds4_gemm_store(Cs, acc, sg, tid, Y, p.ys, m0, min(kGT, p.M - m0), n0, p.accum);
}

// Routed experts, many rows: tiles[t] = (group, first row of the group);
// groups as ds4_xgemv. mode 0: gate | up (columns [0, 2N)), A rows X[xrow[a]];
// mode 1: down, A rows X[a]. BM rows per tile (64, 32 or 16: groups are
// small in short prompts), 64 columns, 4 simdgroups.
struct Ds4XTile {
  int g, r0;
};
template <int BM>
static inline void ds4_xgemm_impl(device const uchar* P0, device const uchar* P1, device const uchar* P2,
                                  device const Ds4XGroup* G, device const float* X, device float* Y,
                                  device const int* xrow, constant Ds4XP& p, device const Ds4XTile* tiles,
                                  threadgroup bfloat* As, threadgroup bfloat* Bs, threadgroup float* Cs, uint2 tg,
                                  uint tid, uint sg) {
  constexpr int SGM = BM == 64 ? 2 : 1, SGN = 4 / SGM;
  constexpr int RM = BM / SGM, CN = kGT / SGN, FM = RM / 8, FN = CN / 8;
  constexpr int APT = BM * kGK / 128;  // A floats per thread
  const Ds4XTile t = tiles[tg.y];
  const Ds4XGroup g = G[t.g];
  device const uchar* base = (g.part == 0 ? P0 : g.part == 1 ? P1 : P2) + g.off;
  const int n0 = int(tg.x) * kGT;
  int mat = 1, nb = n0;
  if (p.mode == 0) mat = n0 < p.N ? 0 : 2, nb = n0 < p.N ? n0 : n0 - p.N;
  const int kb = p.K / 32;
  simdgroup_float8x8 acc[FM * FN];
  for (int i = 0; i < FM * FN; ++i) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const int ar = int(tid) / (kGK / APT), ac = (int(tid) % (kGK / APT)) * APT;
  const int rows = min(BM, g.n - t.r0);
  const bool arow = ar < rows;
  const int a = g.a0 + t.r0 + min(ar, rows - 1);
  device const float* xa = X + ulong(p.mode == 0 ? xrow[a] : a) * ulong(p.xs) + ulong(ac);
  // B: 64 rows x 16 bytes (32 fp4) per K step, two threads per row (16 values each)
  const int brow = int(tid) / 2, bh = int(tid) % 2;
  device const uchar* wrow = base + ulong(mat) * p.wbytes + ulong(nb + brow) * ulong(p.K / 2) + ulong(bh * 8);
  device const uchar* srow = base + 3 * p.wbytes + ulong(mat) * p.sbytes + ulong(nb + brow) * ulong(kb);
  const int sm = int(sg / SGN) * RM, sn = int(sg % SGN) * CN;
  for (int k0 = 0; k0 < p.K; k0 += kGK) {
    for (int j = 0; j < APT; j += 4) {
      const float4 v = arow ? *(device const float4*)(xa + k0 + j) : float4(0.0f);
      As[ar * kGKP + ac + j] = bfloat(v.x);
      As[ar * kGKP + ac + j + 1] = bfloat(v.y);
      As[ar * kGKP + ac + j + 2] = bfloat(v.z);
      As[ar * kGKP + ac + j + 3] = bfloat(v.w);
    }
    {
      const uint2 w = *(device const uint2*)(wrow + k0 / 2);
      const float sc = ds4_e8m0(srow[k0 / 32]) * 16384.0f;
      const uint ws[2] = {w.x, w.y};
      for (int j = 0; j < 2; ++j)
        for (int q = 0; q < 4; ++q) {
          const half2 h = ds4_e2m1x2(ws[j] >> (8 * q));
          Bs[brow * kGKP + bh * 16 + 8 * j + 2 * q] = bfloat(float(h.x) * sc);
          Bs[brow * kGKP + bh * 16 + 8 * j + 2 * q + 1] = bfloat(float(h.y) * sc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < kGK; kk += 8) {
      simdgroup_bfloat8x8 fa[FM], fb[FN];
      for (int i = 0; i < FM; ++i) simdgroup_load(fa[i], As + (sm + 8 * i) * kGKP + kk, kGKP);
      for (int j = 0; j < FN; ++j) simdgroup_load(fb[j], Bs + (sn + 8 * j) * kGKP + kk, kGKP, ulong2(0, 0), true);
      for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j) simdgroup_multiply_accumulate(acc[i * FN + j], fa[i], fb[j], acc[i * FN + j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) simdgroup_store(acc[i * FN + j], Cs + (sm + 8 * i) * kGT + sn + 8 * j, kGT);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = int(tid); e < BM * kGT; e += 128) {
    const int r = e / kGT, c = e % kGT;
    if (r < rows) Y[ulong(g.a0 + t.r0 + r) * ulong(p.ys) + ulong(n0 + c)] = Cs[e];
  }
}
#define DS4_XGEMM(NAME, BM)                                                                                      \
  kernel void NAME(device const uchar* P0 [[buffer(0)]], device const uchar* P1 [[buffer(1)]],                    \
                   device const uchar* P2 [[buffer(2)]], device const Ds4XGroup* G [[buffer(3)]],               \
                   device const float* X [[buffer(4)]], device float* Y [[buffer(5)]],                         \
                   device const int* xrow [[buffer(6)]], constant Ds4XP& p [[buffer(7)]],                      \
                   device const Ds4XTile* tiles [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],      \
                   uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {     \
    threadgroup bfloat As[BM * kGKP], Bs[kGT * kGKP];                                                          \
    threadgroup float Cs[BM * kGT];                                                                            \
    ds4_xgemm_impl<BM>(P0, P1, P2, G, X, Y, xrow, p, tiles, As, Bs, Cs, tg, tid, sg);                          \
  }
DS4_XGEMM(ds4_xgemm, 64)
DS4_XGEMM(ds4_xgemm32, 32)
DS4_XGEMM(ds4_xgemm16, 16)
