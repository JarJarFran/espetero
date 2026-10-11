// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Quantization group (codes sharing one scale and bias along K): 64 for MLX,
// 32 or 16 for GGUF block formats. A function constant, so each pipeline is
// specialized and the per-K-step group loop unrolls away.
constant int kGroup [[function_constant(0)]];
// 0: affine (half scale and bias per group). 1: GGUF Q6_K kept native (int8
// sub-scale per 16 codes times the block's fp16 super-scale; codes offset -32).
constant int kScaleMode [[function_constant(1)]];
constant bool kQ6K = kScaleMode == 1;
// Gate of the gated epilogues (EPI_SWIGLU): constant 2 = 1 is the
// tanh-approximated GELU of Gemma 4's GeGLU; undefined (every SwiGLU matrix:
// the host sets it only for GELU gates) is SiLU, so those pipelines compile as
// before.
constant int kAct [[function_constant(2)]];
constant int kActV = is_function_constant_defined(kAct) ? kAct : 0;
static inline float glu_gate(float x) {
  if (kActV == 1) return 0.5f * x * (1.0f + precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
  return silu(x);
}

// Scales of K step g for this lane's four fragment columns (pair0: the lane's
// first column pair): s[gi * 4 + j], b[gi * 4 + j] per group gi of the step.
// Native Q6_K rebuilds half(d * sub) exactly as the affine expansion does and
// leaves the bias to the code offset.
static inline void kstep_scales(device const half4* SB, ulong g, int KGall, int Np2, int pair0,
                                thread half2* s2, thread half2* b2) {
  // Scale products of a half and a small integer are exact in float, so the
  // half multiplies below round exactly like half(float(d) * float(sc)).
  if (kQ6K) {
    device const uint2* SC = (device const uint2*)SB;
    device const half2* DD = (device const half2*)(SC + ulong(KGall) * Np2);
    for (int j = 0; j < 4; ++j) {
      const uint2 v = SC[g * Np2 + pair0 + j * 4];
      const half2 d = DD[(g / 4) * Np2 + pair0 + j * 4];
      const char4 lo = as_type<char4>(v.x), hi = as_type<char4>(v.y);  // byte = 2 * group + column
      s2[0 * 4 + j] = d * half2(lo.xy);
      s2[1 * 4 + j] = d * half2(lo.zw);
      s2[2 * 4 + j] = d * half2(hi.xy);
      s2[3 * 4 + j] = d * half2(hi.zw);
      for (int gi = 0; gi < 4; ++gi) b2[gi * 4 + j] = half2(0.0h);
    }
  } else if (kScaleMode == 2) {
    // Q8_0: half2 scale per 32 codes and column pair; bias = -128 * scale.
    device const half2* S2 = (device const half2*)SB;
    for (int gi = 0; gi < 2; ++gi)
      for (int j = 0; j < 4; ++j) {
        const half2 sc = S2[(g * 2 + gi) * Np2 + pair0 + j * 4];
        s2[gi * 4 + j] = sc;
        b2[gi * 4 + j] = sc * half2(-128.0h);
      }
  } else if (kScaleMode == 3) {
    // Q4_K/Q5_K: (scale, min) bytes per 32 codes, then (d, dmin) per 256.
    device const uint2* SC = (device const uint2*)SB;
    device const half4* DD = (device const half4*)(SC + ulong(KGall) * Np2);
    for (int j = 0; j < 4; ++j) {
      const uint2 v = SC[g * Np2 + pair0 + j * 4];
      const half4 dd = DD[(g / 4) * Np2 + pair0 + j * 4];
      const uchar4 sc = as_type<uchar4>(v.x), mn = as_type<uchar4>(v.y);  // byte = 2 * group + column
      s2[0 * 4 + j] = dd.xy * half2(sc.xy);
      s2[1 * 4 + j] = dd.xy * half2(sc.zw);
      b2[0 * 4 + j] = -(dd.zw * half2(mn.xy));
      b2[1 * 4 + j] = -(dd.zw * half2(mn.zw));
    }
  } else {
    const int NG = 64 / kGroup;
    for (int gi = 0; gi < 4; ++gi)
      if (gi < NG)
        for (int j = 0; j < 4; ++j) {
          const half4 sb = SB[(g * NG + gi) * Np2 + pair0 + j * 4];
          s2[gi * 4 + j] = sb.xy;
          b2[gi * 4 + j] = sb.zw;
        }
  }
}

// A dequantized pair of weights from its codes (as exponent-biased halves,
// value 1024 + code) and scales.
static inline half2 dequant_pair(half2 biased, half2 s, half2 b) {
  return kQ6K ? (biased - half2(1056.0h)) * s : fma(biased - half2(1024.0h), s, b);
}

// ---------------------------------------------------------------------------
// Quantized GEMM parameters. Epilogues: 0 store fp32, 1 accumulate into fp32,
// 2 SwiGLU over row-interleaved gate/up blocks of 8 -> fp16 [M, N/2].
// ---------------------------------------------------------------------------
struct GemmParams {
  int M;
  int N;
  int K;
  int lda;
  int ldy;
  int g_split;    // K groups (of 64) per split; 0 = no split
  int z_stride;   // workspace elements between splits
};

constant constexpr int EPI_STORE = 0;
constant constexpr int EPI_ADD = 1;
constant constexpr int EPI_SWIGLU = 2;

// ---------------------------------------------------------------------------
// Register-direct quantized GEMM (the engine's main path).
//
// Weights are repacked once at load into simdgroup-fragment order:
//   Wr[N/32][K/64][lane][4 fragment columns][2 words]   (uint32)
// so a lane fetches every B fragment it needs for a 64-wide K step with two
// coalesced 16-byte loads. Nibbles i and i+4 of a word are rows n and n+1 at
// the same k, which the exponent-bias trick turns into one half2 FMA with
//   SB[K/64][N/2] = half4(s_n, s_n+1, b_n, b_n+1).
// Only A is staged in threadgroup memory. Each simdgroup owns BM rows x 32
// columns; WN simdgroups side by side cover BN = 32 * WN columns.
// ---------------------------------------------------------------------------
kernel void repack_weights(
    device const uint* W [[buffer(0)]],
    device uint* Wr [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& K [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  // one thread per output uint4 pair (lane) of a (block, group)
  const int KG = K / 64;
  const int lane = gid % 32;
  const int bg = gid / 32;  // block * KG + g
  const int nb = bg / KG;
  const int g = bg % KG;
  if (nb >= (N + 31) / 32) return;
  const int qid = lane / 4;
  const int fm = (qid & 4) + ((lane / 2) % 4);
  const int fn = (qid & 2) * 2 + (lane % 2) * 2;
  const int KW = K / 8;
  uint out[8];
  UNROLL for (int j = 0; j < 4; ++j) {
    const int n0 = nb * 32 + j * 8 + fn;
    uint rw[2][8];
    UNROLL for (int r = 0; r < 2; ++r) {
      const int n = n0 + r;
      UNROLL for (int jj = 0; jj < 8; ++jj)
        rw[r][jj] = n < N ? W[ulong(n) * KW + g * 8 + jj] : 0u;
    }
    UNROLL for (int w = 0; w < 2; ++w) {
      uint word = 0;
      UNROLL for (int i = 0; i < 8; ++i) {
        const int jj = 4 * w + (i % 4);
        const uint nib = (rw[i >= 4 ? 1 : 0][jj] >> (4 * fm)) & 0xFu;
        word |= nib << (4 * i);
      }
      out[j * 2 + w] = word;
    }
  }
  device uint4* dst = (device uint4*)(Wr) + (ulong(bg) * 32 + lane) * 2;
  dst[0] = uint4(out[0], out[1], out[2], out[3]);
  dst[1] = uint4(out[4], out[5], out[6], out[7]);
}

// 2-bit codes (MLX packing: 16 per uint32, code k at bits 2 * (k % 16)) ->
// Wr2[N/32][K/64][lane][4 words]. Word j holds fragment column pair j:
// bits 2 * jj of the low half are row n at k-step jj, of the high half row
// n + 1, so masking 0x00030003 after a shift of 2 * jj yields the half2
// exponent-bias pair.
kernel void repack_weights2(
    device const uint* W [[buffer(0)]],
    device uint* Wr [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& K [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const int KG = K / 64;
  const int lane = gid % 32;
  const int bg = gid / 32;
  const int nb = bg / KG;
  const int g = bg % KG;
  if (nb >= (N + 31) / 32) return;
  const int qid = lane / 4;
  const int fm = (qid & 4) + ((lane / 2) % 4);
  const int fn = (qid & 2) * 2 + (lane % 2) * 2;
  const int KW = K / 16;
  uint out[4];
  UNROLL for (int j = 0; j < 4; ++j) {
    uint word = 0;
    UNROLL for (int r = 0; r < 2; ++r) {
      const int n = nb * 32 + j * 8 + fn + r;
      UNROLL for (int jj = 0; jj < 8; ++jj) {
        const int k = g * 64 + jj * 8 + fm;
        const uint code = n < N ? (W[ulong(n) * KW + k / 16] >> (2 * (k % 16))) & 3u : 0u;
        word |= code << (16 * r + 2 * jj);
      }
    }
    out[j] = word;
  }
  ((device uint4*)Wr)[ulong(bg) * 32 + lane] = uint4(out[0], out[1], out[2], out[3]);
}

// Repacked 6-bit codes (nibble plane Wr + 2-bit plane Wh, repack_planes) ->
// the 8-bit fragment layout of repack_weights8, same codes: prefill
// GEMMs unpack bytes with fewer ALU ops. One thread per (block, K step, lane).
kernel void widen6_8(
    device const uint4* Wr [[buffer(0)]],
    device const uint4* Wh [[buffer(1)]],
    device uint4* W8 [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const int KG = K / 64;
  const ulong bg = gid / 32;
  if (int(bg / ulong(KG)) >= (N + 31) / 32) return;
  const uint4 l0 = Wr[ulong(gid) * 2], l1 = Wr[ulong(gid) * 2 + 1], h4 = Wh[gid];
  const uint lo[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
  const uint hi[4] = {h4.x, h4.y, h4.z, h4.w};
  uint out[16];
  UNROLL for (int j = 0; j < 4; ++j) {
    UNROLL for (int q = 0; q < 4; ++q) {
      uint word = 0;
      UNROLL for (int t = 0; t < 2; ++t) {
        const int jj = 2 * q + t;
        const uint nib = lo[j * 2 + jj / 4] >> (4 * (jj % 4));
        const uint h = hi[j] >> (2 * jj);
        const uint a = (nib & 0xFu) | ((h & 3u) << 4);              // column n0
        const uint b = ((nib >> 16) & 0xFu) | (((h >> 16) & 3u) << 4);  // column n0 + 1
        word |= (a << (8 * t)) | (b << (16 + 8 * t));
      }
      out[j * 4 + q] = word;
    }
  }
  UNROLL for (int i = 0; i < 4; ++i) W8[ulong(gid) * 4 + i] = uint4(out[4 * i], out[4 * i + 1], out[4 * i + 2], out[4 * i + 3]);
}

// MLX packed rows (little-endian bit stream of `bits`-wide codes, bits <= 8)
// -> one byte per code, [N, K].
kernel void unpack_bits(
    device const uchar* W [[buffer(0)]],
    device uchar* out [[buffer(1)]],
    constant int& K [[buffer(2)]],
    constant int& bits [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int k = gid.x;
  const int n = gid.y;
  if (k >= K) return;
  const ulong row_bytes = ulong(K) * bits / 8;
  const uint bit = uint(k) * bits;
  device const uchar* src = W + ulong(n) * row_bytes + (bit >> 3);
  uint v = uint(src[0]);
  if ((bit & 7) + bits > 8) v |= uint(src[1]) << 8;
  out[ulong(n) * K + k] = uchar((v >> (bit & 7)) & ((1u << bits) - 1));
}

// 8-bit codes [N, K] (bytes) -> Wr8[N/32][K/64][lane][16 words]. Word
// (j * 4 + p) packs rows n, n+1 of fragment column j at k-steps 2p and 2p+1:
// bytes (n, 2p) (n, 2p+1) (n+1, 2p) (n+1, 2p+1), so masking 0x00FF00FF after a
// shift of 0 or 8 yields the half2 exponent-bias pair for one k-step.
kernel void repack_weights8(
    device const uchar* W [[buffer(0)]],
    device uint* Wr [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& K [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const int KG = K / 64;
  const int lane = gid % 32;
  const int bg = gid / 32;
  const int nb = bg / KG;
  const int g = bg % KG;
  if (nb >= (N + 31) / 32) return;
  const int qid = lane / 4;
  const int fm = (qid & 4) + ((lane / 2) % 4);
  const int fn = (qid & 2) * 2 + (lane % 2) * 2;
  uint out[16];
  UNROLL for (int j = 0; j < 4; ++j) {
    const int n0 = nb * 32 + j * 8 + fn;
    UNROLL for (int q = 0; q < 4; ++q) {
      const int k0 = g * 64 + (2 * q) * 8 + fm;
      const int k1 = k0 + 8;
      const uint a0 = n0 < N ? uint(W[ulong(n0) * K + k0]) : 0u;
      const uint a1 = n0 < N ? uint(W[ulong(n0) * K + k1]) : 0u;
      const uint b0 = n0 + 1 < N ? uint(W[ulong(n0 + 1) * K + k0]) : 0u;
      const uint b1 = n0 + 1 < N ? uint(W[ulong(n0 + 1) * K + k1]) : 0u;
      out[j * 4 + q] = a0 | (a1 << 8) | (b0 << 16) | (b1 << 24);
    }
  }
  device uint4* dst = (device uint4*)(Wr) + (ulong(bg) * 32 + lane) * 4;
  UNROLL for (int i = 0; i < 4; ++i) dst[i] = uint4(out[4 * i], out[4 * i + 1], out[4 * i + 2], out[4 * i + 3]);
}

// Per-group scales/biases [N, K/G] (bf16 from MLX, or half) -> SB[K/G][Np/2]
// half4 (s_n, s_n+1, b_n, b_n+1).
static inline float scale_at(device const ushort* S, ulong i, int half_src) {
  return half_src ? float(as_type<half>(S[i])) : bf2f(S[i]);
}

// Byte codes [N, K] of 5- or 6-bit values -> a nibble plane in the 4-bit
// fragment layout (Wr, 8 words per lane and K group) plus a high-bit plane
// (Wh): 6-bit, word j = 2 bits per k-step (rows n at 2jj, n+1 at 16+2jj);
// 5-bit, word j/2 = 1 bit per k-step (rows n at jj + 8(j%2), n+1 at +16).
static inline uint byte_code(device const uchar* W, int N, int K, int n, int k) {
  return n < N ? uint(W[ulong(n) * K + k]) : 0u;
}

kernel void repack_planes(
    device const uchar* W [[buffer(0)]],
    device uint* Wr [[buffer(1)]],
    device uint* Wh [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& bits [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const int KG = K / 64;
  const int lane = gid % 32;
  const int bg = gid / 32;
  const int nb = bg / KG;
  const int g = bg % KG;
  if (nb >= (N + 31) / 32) return;
  const int qid = lane / 4;
  const int fm = (qid & 4) + ((lane / 2) % 4);
  const int fn = (qid & 2) * 2 + (lane % 2) * 2;
  uint lo[8];
  uint hi[4] = {0u, 0u, 0u, 0u};
  UNROLL for (int j = 0; j < 4; ++j) {
    const int n0 = nb * 32 + j * 8 + fn;
    UNROLL for (int w = 0; w < 2; ++w) {
      uint word = 0;
      UNROLL for (int i = 0; i < 8; ++i) {
        const int jj = 4 * w + (i % 4);
        word |= (byte_code(W, N, K, n0 + (i >= 4 ? 1 : 0), g * 64 + jj * 8 + fm) & 0xFu) << (4 * i);
      }
      lo[j * 2 + w] = word;
    }
    UNROLL for (int jj = 0; jj < 8; ++jj) {
      const int k = g * 64 + jj * 8 + fm;
      const uint a = byte_code(W, N, K, n0, k) >> 4, b = byte_code(W, N, K, n0 + 1, k) >> 4;
      if (bits == 6) {
        hi[j] |= ((a & 3u) << (2 * jj)) | ((b & 3u) << (16 + 2 * jj));
      } else {
        const int sh = jj + 8 * (j % 2);
        hi[j / 2] |= ((a & 1u) << sh) | ((b & 1u) << (16 + sh));
      }
    }
  }
  device uint4* dst = (device uint4*)(Wr) + (ulong(bg) * 32 + lane) * 2;
  dst[0] = uint4(lo[0], lo[1], lo[2], lo[3]);
  dst[1] = uint4(lo[4], lo[5], lo[6], lo[7]);
  if (bits == 6) ((device uint4*)Wh)[ulong(bg) * 32 + lane] = uint4(hi[0], hi[1], hi[2], hi[3]);
  else ((device uint2*)Wh)[ulong(bg) * 32 + lane] = uint2(hi[0], hi[1]);
}

kernel void repack_scales(
    device const ushort* S [[buffer(0)]],
    device const ushort* Z [[buffer(1)]],
    device half4* SB [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& G [[buffer(5)]],
    constant int& half_src [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int pair = gid.x;  // n / 2
  const int g = gid.y;
  const int KG = K / G;
  const int Np = (N + 31) / 32 * 32;
  if (pair * 2 >= Np || g >= KG) return;
  const int n = pair * 2;
  half4 v = half4(0);
  if (n < N) {
    v.x = half(scale_at(S, ulong(n) * KG + g, half_src));
    v.z = half(scale_at(Z, ulong(n) * KG + g, half_src));
  }
  if (n + 1 < N) {
    v.y = half(scale_at(S, ulong(n + 1) * KG + g, half_src));
    v.w = half(scale_at(Z, ulong(n + 1) * KG + g, half_src));
  }
  SB[ulong(g) * (Np / 2) + pair] = v;
}

// Native GGUF Q8_0 scales: fp16 S [N][K/32] -> half2 per 32 codes and pair.
kernel void repack_q8_scales(
    device const half* S [[buffer(0)]],
    device half2* SB [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& K [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int pair = gid.x;
  const int g = gid.y;
  const int Np = (N + 31) / 32 * 32;
  if (pair * 2 >= Np || g >= K / 32) return;
  const int n = pair * 2;
  const half a = n < N ? S[ulong(n) * (K / 32) + g] : half(0);
  const half b = n + 1 < N ? S[ulong(n + 1) * (K / 32) + g] : half(0);
  SB[ulong(g) * (Np / 2) + pair] = half2(a, b);
}

// Native GGUF Q4_K/Q5_K scales: (scale, min) bytes S [N][K/32][2] and fp16
// (d, dmin) D [N][K/256][2] -> per K step and column pair a uint2 (x: scales,
// y: mins; byte = 2 * group + column), then half4 (d0, d1, dmin0, dmin1) per 256.
kernel void repack_kq_scales(
    device const uchar* S [[buffer(0)]],
    device const half* D [[buffer(1)]],
    device uint2* SC [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int pair = gid.x;
  const int g = gid.y;
  const int KG = K / 64;
  const int Np = (N + 31) / 32 * 32;
  if (pair * 2 >= Np || g >= KG) return;
  const int n = pair * 2;
  uint sc = 0u, mn = 0u;
  for (int gi = 0; gi < 2; ++gi)
    for (int c = 0; c < 2; ++c) {
      const int col = n + c;
      if (col >= N) continue;
      device const uchar* sm = S + (ulong(col) * (K / 32) + g * 2 + gi) * 2;
      sc |= uint(sm[0]) << (16 * gi + 8 * c);
      mn |= uint(sm[1]) << (16 * gi + 8 * c);
    }
  SC[ulong(g) * (Np / 2) + pair] = uint2(sc, mn);
  if (g % 4 == 0) {
    device half4* DD = (device half4*)(SC + ulong(KG) * (Np / 2));
    half4 v = half4(0);
    if (n < N) v.x = D[(ulong(n) * (K / 256) + g / 4) * 2], v.z = D[(ulong(n) * (K / 256) + g / 4) * 2 + 1];
    if (n + 1 < N)
      v.y = D[(ulong(n + 1) * (K / 256) + g / 4) * 2], v.w = D[(ulong(n + 1) * (K / 256) + g / 4) * 2 + 1];
    DD[ulong(g / 4) * (Np / 2) + pair] = v;
  }
}

// Native GGUF Q6_K scales: int8 sub-scales S [N][K/16] and fp16 super-scales
// D [N][K/256] -> per K step and column pair a uint2 of 4 x 2 sub-scales
// (group-major, byte = 2 * group + column), then half2 super-scales per 256.
kernel void repack_q6k_scales(
    device const char* S [[buffer(0)]],
    device const half* D [[buffer(1)]],
    device uint2* SC [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int pair = gid.x;
  const int g = gid.y;
  const int KG = K / 64;
  const int Np = (N + 31) / 32 * 32;
  if (pair * 2 >= Np || g >= KG) return;
  const int n = pair * 2;
  uint lo = 0u, hi = 0u;
  for (int gi = 0; gi < 4; ++gi)
    for (int c = 0; c < 2; ++c) {
      const int col = n + c;
      const uint b = col < N ? uint(uchar(S[ulong(col) * (K / 16) + g * 4 + gi])) : 0u;
      const int sh = 16 * (gi % 2) + 8 * c;
      if (gi < 2) lo |= b << sh;
      else hi |= b << sh;
    }
  SC[ulong(g) * (Np / 2) + pair] = uint2(lo, hi);
  if (g % 4 == 0) {
    device half2* DD = (device half2*)(SC + ulong(KG) * (Np / 2));
    const half d0 = n < N ? D[ulong(n) * (K / 256) + g / 4] : half(0);
    const half d1 = n + 1 < N ? D[ulong(n + 1) * (K / 256) + g / 4] : half(0);
    DD[ulong(g / 4) * (Np / 2) + pair] = half2(d0, d1);
  }
}

template <int BM, int WN, int EPI, int BITS>
kernel void rgemm(
    device const half* A [[buffer(0)]],
    device const uint* Wr [[buffer(1)]],
    device const half4* SB [[buffer(2)]],
    device const uint* Wh [[buffer(3)]],
    device void* Yv [[buffer(4)]],
    constant GemmParams& p [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BK = 64;
  constexpr int BN = WN * 32;
  constexpr int NT = WN * 32;
  constexpr int FM = BM / 8;
  constexpr int LDA = BK + 8;
  constexpr int AC = (BM * 8 + NT - 1) / NT;
  threadgroup half As[BM * LDA];

  // Row blocks vary fastest in the grid, so the threadgroups that share one
  // column block (its weights) run back to back and read them from cache.
  const int m0 = tgid.x * BM;
  const int n0 = tgid.y * BN + sgid * 32;
  const int KGall = p.K / 64;
  const int g_begin = p.g_split ? int(tgid.z) * p.g_split : 0;
  const int g_end = p.g_split ? min(KGall, g_begin + p.g_split) : KGall;
  const int Np2 = (p.N + 31) / 32 * 16;  // SB row length in pairs
  constexpr int MAXG = 4;                // groups per 64-wide K step (kGroup = 16)
  const int NG = 64 / kGroup;
  const bool n_ok = n0 < p.N;

  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;

  simdgroup_matrix<float, 8, 8> acc[FM][4];
  UNROLL for (int i = 0; i < FM; ++i)
    UNROLL for (int j = 0; j < 4; ++j)
      acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  int a_row[AC], a_col[AC];
  bool a_ok[AC], a_live[AC];
  UNROLL for (int c = 0; c < AC; ++c) {
    const int idx = tid + c * NT;
    a_live[c] = idx < BM * 8;
    a_row[c] = min(idx / 8, BM - 1);
    a_col[c] = (idx % 8) * 8;
    a_ok[c] = a_live[c] && (m0 + a_row[c]) < p.M;
  }
  uint4 ar[AC];
  UNROLL for (int c = 0; c < AC; ++c)
    ar[c] = a_ok[c] ? *((device const uint4*)(A + ulong(m0 + a_row[c]) * p.lda + g_begin * 64 + a_col[c])) : uint4(0);

  // uint4 words per lane and K group: 2 for the nibble plane (4-, 5- and
  // 6-bit), 4 for 8-bit codes; 5/6-bit add a high-bit plane (2 or 4 words).
  constexpr int WPL = BITS == 8 ? 4 : 2;
  constexpr int HPL = BITS == 6 ? 4 : (BITS == 5 ? 2 : 0);
  device const uint4* wp = (device const uint4*)(Wr) + ulong(n_ok ? n0 / 32 : 0) * KGall * (32 * WPL) + lane * WPL;
  device const uint* hp = Wh + (ulong(n_ok ? n0 / 32 : 0) * KGall * 32 + lane) * (HPL ? HPL : 1);

  for (int g = g_begin; g < g_end; ++g) {
    UNROLL for (int c = 0; c < AC; ++c)
      if (a_live[c]) *((threadgroup uint4*)(As + a_row[c] * LDA + a_col[c])) = ar[c];
    uint words[4 * WPL];
    UNROLL for (int i = 0; i < WPL; ++i) {
      const uint4 w = wp[ulong(g) * (32 * WPL) + i];
      words[4 * i] = w.x;
      words[4 * i + 1] = w.y;
      words[4 * i + 2] = w.z;
      words[4 * i + 3] = w.w;
    }
    uint hw[4] = {0u, 0u, 0u, 0u};
    if (BITS == 6) {
      const uint4 h = *((device const uint4*)(hp + ulong(g) * 32 * 4));
      hw[0] = h.x, hw[1] = h.y, hw[2] = h.z, hw[3] = h.w;
    } else if (BITS == 5) {
      const uint2 h = *((device const uint2*)(hp + ulong(g) * 32 * 2));
      hw[0] = h.x, hw[1] = h.y;
    }
    half2 s2[MAXG * 4], b2[MAXG * 4];
    kstep_scales(SB, ulong(g), KGall, Np2, n_ok ? (n0 + fn) / 2 : 0, s2, b2);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (g + 1 < g_end) {
      UNROLL for (int c = 0; c < AC; ++c)
        ar[c] = a_ok[c] ? *((device const uint4*)(A + ulong(m0 + a_row[c]) * p.lda + (g + 1) * 64 + a_col[c])) : uint4(0);
    }
    UNROLL for (int jj = 0; jj < 8; ++jj) {
      simdgroup_matrix<half, 8, 8> a[FM];
      simdgroup_matrix<half, 8, 8> b[4];
      UNROLL for (int i = 0; i < FM; ++i) {
        const half2 v = *((const threadgroup half2*)(As + (i * 8 + fm) * LDA + jj * 8 + fn));
        a[i].thread_elements()[0] = v.x;
        a[i].thread_elements()[1] = v.y;
      }
      UNROLL for (int j = 0; j < 4; ++j) {
        half2 q;
        if (BITS == 2) {
          const uint x = words[j] >> (2 * jj);
          q = as_type<half2>((x & 0x00030003u) | 0x64006400u);
        } else if (BITS == 4) {
          const uint x = words[j * 2 + jj / 4] >> (4 * (jj % 4));
          q = as_type<half2>((x & 0x000F000Fu) | 0x64006400u);
        } else if (BITS == 6 || BITS == 5) {
          const uint x = (words[j * 2 + jj / 4] >> (4 * (jj % 4))) & 0x000F000Fu;
          const uint h = BITS == 6 ? (hw[j] >> (2 * jj)) & 0x00030003u
                                   : (hw[j / 2] >> (jj + 8 * (j % 2))) & 0x00010001u;
          q = as_type<half2>(x | (h << 4) | 0x64006400u);
        } else {
          const uint x = words[j * 4 + jj / 2] >> (8 * (jj % 2));
          q = as_type<half2>((x & 0x00FF00FFu) | 0x64006400u);
        }
        const int gi = (jj * 8) / kGroup;
        const half2 wv = dequant_pair(q, s2[gi * 4 + j], b2[gi * 4 + j]);
        b[j].thread_elements()[0] = wv.x;
        b[j].thread_elements()[1] = wv.y;
      }
      UNROLL for (int i = 0; i < FM; ++i)
        UNROLL for (int j = 0; j < 4; ++j)
          simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  UNROLL for (int i = 0; i < FM; ++i) {
    const int row = m0 + i * 8 + fm;
    if (row >= p.M) continue;
    if (EPI == EPI_SWIGLU) {
      // fragment columns (0,1) and (2,3) are gate/up blocks of 8 outputs
      device half* y = (device half*)Yv + ulong(row) * p.ldy;
      UNROLL for (int j = 0; j < 4; j += 2) {
        const int col = n0 + j * 8;
        if (col >= p.N) continue;
        const int out = (col / 16) * 8 + fn;
        const float g0 = acc[i][j].thread_elements()[0];
        const float g1 = acc[i][j].thread_elements()[1];
        const float u0 = acc[i][j + 1].thread_elements()[0];
        const float u1 = acc[i][j + 1].thread_elements()[1];
        *((device half2*)(y + out)) = half2(half(glu_gate(g0) * u0), half(glu_gate(g1) * u1));
      }
    } else {
      device float* y = (device float*)Yv + ulong(tgid.z) * p.z_stride + ulong(row) * p.ldy;
      UNROLL for (int j = 0; j < 4; ++j) {
        const int col = n0 + j * 8 + fn;
        float2 v = float2(acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
        if (col + 1 < p.N) {
          if (EPI == EPI_ADD) v += *((device float2*)(y + col));
          *((device float2*)(y + col)) = v;
        } else if (col < p.N) {
          y[col] = (EPI == EPI_ADD) ? y[col] + v.x : v.x;
        }
      }
    }
  }
}

// Prefill GEMM with shared dequantization (gemm-dequant-share): the
// threadgroup holds 2 * WN simdgroups; simdgroups (2c, 2c + 1) share column
// block c of the tile (32 columns) and split its BM rows in halves, so both
// need the same dequantized B fragments per K step: each dequantizes 2 of the
// 4 column fragments (its own weight words: no redundant loads) and the pair
// exchanges them through threadgroup memory under the barrier the A tile
// already needs. Halves the unpack + scale work per MMA against rgemm at the
// same per-simdgroup tile (BM/2 rows x 32 columns); the 64-row variant has
// the register footprint of rgemm's 32x4 (the plain 64x4 tile spilled).
template <int BM, int WN, int EPI, int BITS>
kernel void rgemm_s(
    device const half* A [[buffer(0)]],
    device const uint* Wr [[buffer(1)]],
    device const half4* SB [[buffer(2)]],
    device const uint* Wh [[buffer(3)]],
    device void* Yv [[buffer(4)]],
    constant GemmParams& p [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BK = 64;
  constexpr int BN = WN * 32;
  constexpr int NT = 2 * WN * 32;
  constexpr int HM = BM / 2;  // rows per simdgroup
  constexpr int FM = HM / 8;
  constexpr int LDA = BK + 8;
  constexpr int AC = (BM * 8 + NT - 1) / NT;
  threadgroup half As[BM * LDA];
  threadgroup half2 Bx[WN][2][8][2][32];  // [pair][half][jj][local j][lane]

  const int cb = int(sgid) >> 1, hf = int(sgid) & 1;
  const int m0 = tgid.x * BM;
  const int mh = m0 + hf * HM;  // this simdgroup's first row
  const int n0 = tgid.y * BN + cb * 32;
  const int KGall = p.K / 64;
  const int g_begin = p.g_split ? int(tgid.z) * p.g_split : 0;
  const int g_end = p.g_split ? min(KGall, g_begin + p.g_split) : KGall;
  const int Np2 = (p.N + 31) / 32 * 16;
  constexpr int MAXG = 4;
  const bool n_ok = n0 < p.N;

  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;

  simdgroup_matrix<float, 8, 8> acc[FM][4];
  UNROLL for (int i = 0; i < FM; ++i)
    UNROLL for (int j = 0; j < 4; ++j)
      acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  int a_row[AC], a_col[AC];
  bool a_ok[AC], a_live[AC];
  UNROLL for (int c = 0; c < AC; ++c) {
    const int idx = tid + c * NT;
    a_live[c] = idx < BM * 8;
    a_row[c] = min(idx / 8, BM - 1);
    a_col[c] = (idx % 8) * 8;
    a_ok[c] = a_live[c] && (m0 + a_row[c]) < p.M;
  }
  uint4 ar[AC];
  UNROLL for (int c = 0; c < AC; ++c)
    ar[c] = a_ok[c] ? *((device const uint4*)(A + ulong(m0 + a_row[c]) * p.lda + g_begin * 64 + a_col[c])) : uint4(0);

  // This half's column fragments j in {2 hf, 2 hf + 1}: 4-bit / 5-bit / 6-bit
  // words[j * 2 + jj / 4] live in uint4 index hf; 8-bit words[j * 4 + jj / 2]
  // in uint4 indices 2 hf, 2 hf + 1.
  constexpr int WPL = BITS == 8 ? 4 : 2;
  constexpr int WPH = WPL / 2;  // uint4 words this half loads
  constexpr int HPL = BITS == 6 ? 4 : (BITS == 5 ? 2 : 0);
  device const uint4* wp = (device const uint4*)(Wr) + ulong(n_ok ? n0 / 32 : 0) * KGall * (32 * WPL) + lane * WPL + hf * WPH;
  device const uint* hp = Wh + (ulong(n_ok ? n0 / 32 : 0) * KGall * 32 + lane) * (HPL ? HPL : 1);

  for (int g = g_begin; g < g_end; ++g) {
    UNROLL for (int c = 0; c < AC; ++c)
      if (a_live[c]) *((threadgroup uint4*)(As + a_row[c] * LDA + a_col[c])) = ar[c];
    uint words[4 * WPH];
    UNROLL for (int i = 0; i < WPH; ++i) {
      const uint4 w = wp[ulong(g) * (32 * WPL) + i];
      words[4 * i] = w.x;
      words[4 * i + 1] = w.y;
      words[4 * i + 2] = w.z;
      words[4 * i + 3] = w.w;
    }
    uint hw[4] = {0u, 0u, 0u, 0u};
    if (BITS == 6) {
      const uint4 h = *((device const uint4*)(hp + ulong(g) * 32 * 4));
      hw[0] = h.x, hw[1] = h.y, hw[2] = h.z, hw[3] = h.w;
    } else if (BITS == 5) {
      const uint2 h = *((device const uint2*)(hp + ulong(g) * 32 * 2));
      hw[0] = h.x, hw[1] = h.y;
    }
    half2 s2[MAXG * 4], b2[MAXG * 4];
    kstep_scales(SB, ulong(g), KGall, Np2, n_ok ? (n0 + fn) / 2 : 0, s2, b2);
    // Dequantize this half's 2 column fragments for all 8 jj; publish them.
    half2 mine[8][2];
    UNROLL for (int jj = 0; jj < 8; ++jj) {
      UNROLL for (int jl = 0; jl < 2; ++jl) {
        const int j = 2 * hf + jl;  // global fragment column
        half2 q;
        if (BITS == 4) {
          const uint x = words[jl * 2 + jj / 4] >> (4 * (jj % 4));  // words[j*2 + jj/4] - hf*4
          q = as_type<half2>((x & 0x000F000Fu) | 0x64006400u);
        } else if (BITS == 6 || BITS == 5) {
          const uint x = (words[jl * 2 + jj / 4] >> (4 * (jj % 4))) & 0x000F000Fu;
          const uint h = BITS == 6 ? (hw[j] >> (2 * jj)) & 0x00030003u
                                   : (hw[j / 2] >> (jj + 8 * (j % 2))) & 0x00010001u;
          q = as_type<half2>(x | (h << 4) | 0x64006400u);
        } else {
          const uint x = words[jl * 4 + jj / 2] >> (8 * (jj % 2));  // words[j*4 + jj/2] - hf*8
          q = as_type<half2>((x & 0x00FF00FFu) | 0x64006400u);
        }
        const int gi = (jj * 8) / kGroup;
        mine[jj][jl] = dequant_pair(q, s2[gi * 4 + j], b2[gi * 4 + j]);
        Bx[cb][hf][jj][jl][lane] = mine[jj][jl];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (g + 1 < g_end) {
      UNROLL for (int c = 0; c < AC; ++c)
        ar[c] = a_ok[c] ? *((device const uint4*)(A + ulong(m0 + a_row[c]) * p.lda + (g + 1) * 64 + a_col[c])) : uint4(0);
    }
    UNROLL for (int jj = 0; jj < 8; ++jj) {
      simdgroup_matrix<half, 8, 8> a[FM];
      simdgroup_matrix<half, 8, 8> b[4];
      UNROLL for (int i = 0; i < FM; ++i) {
        const half2 v = *((const threadgroup half2*)(As + (hf * HM + i * 8 + fm) * LDA + jj * 8 + fn));
        a[i].thread_elements()[0] = v.x;
        a[i].thread_elements()[1] = v.y;
      }
      UNROLL for (int j = 0; j < 4; ++j) {
        const half2 wv = (j / 2 == hf) ? mine[jj][j % 2] : Bx[cb][hf ^ 1][jj][j % 2][lane];
        b[j].thread_elements()[0] = wv.x;
        b[j].thread_elements()[1] = wv.y;
      }
      UNROLL for (int i = 0; i < FM; ++i)
        UNROLL for (int j = 0; j < 4; ++j)
          simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  UNROLL for (int i = 0; i < FM; ++i) {
    const int row = mh + i * 8 + fm;
    if (row >= p.M) continue;
    if (EPI == EPI_SWIGLU) {
      device half* y = (device half*)Yv + ulong(row) * p.ldy;
      UNROLL for (int j = 0; j < 4; j += 2) {
        const int col = n0 + j * 8;
        if (col >= p.N) continue;
        const int out = (col / 16) * 8 + fn;
        const float g0 = acc[i][j].thread_elements()[0];
        const float g1 = acc[i][j].thread_elements()[1];
        const float u0 = acc[i][j + 1].thread_elements()[0];
        const float u1 = acc[i][j + 1].thread_elements()[1];
        *((device half2*)(y + out)) = half2(half(glu_gate(g0) * u0), half(glu_gate(g1) * u1));
      }
    } else {
      device float* y = (device float*)Yv + ulong(tgid.z) * p.z_stride + ulong(row) * p.ldy;
      UNROLL for (int j = 0; j < 4; ++j) {
        const int col = n0 + j * 8 + fn;
        float2 v = float2(acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
        if (col + 1 < p.N) {
          if (EPI == EPI_ADD) v += *((device float2*)(y + col));
          *((device float2*)(y + col)) = v;
        } else if (col < p.N) {
          y[col] = (EPI == EPI_ADD) ? y[col] + v.x : v.x;
        }
      }
    }
  }
}

#define INSTANTIATE_RGEMM_S(PRE, TAG, BM, WN, BITS)                                  \
  template [[host_name(PRE TAG "_store")]] kernel void rgemm_s<BM, WN, 0, BITS>(    \
      device const half*, device const uint*, device const half4*, device const uint*, \
      device void*, constant GemmParams&, uint3, uint, uint, uint);                 \
  template [[host_name(PRE TAG "_add")]] kernel void rgemm_s<BM, WN, 1, BITS>(      \
      device const half*, device const uint*, device const half4*, device const uint*, \
      device void*, constant GemmParams&, uint3, uint, uint, uint);                 \
  template [[host_name(PRE TAG "_swiglu")]] kernel void rgemm_s<BM, WN, 2, BITS>(   \
      device const half*, device const uint*, device const half4*, device const uint*, \
      device void*, constant GemmParams&, uint3, uint, uint, uint);
#define INSTANTIATE_RGEMM_S_BITS(PRE, BITS)                                          \
  INSTANTIATE_RGEMM_S(PRE, "32x4s", 32, 4, BITS) INSTANTIATE_RGEMM_S(PRE, "64x4s", 64, 4, BITS)
INSTANTIATE_RGEMM_S_BITS("rgemm_", 4)
INSTANTIATE_RGEMM_S_BITS("rgemm5_", 5)
INSTANTIATE_RGEMM_S_BITS("rgemm6_", 6)
INSTANTIATE_RGEMM_S_BITS("rgemm8_", 8)

// Decode-sized quantized GEMV (M <= MM rows, MM = 1 or 2): no MMA, the same weight layout
// as rgemm. A threadgroup owns one 32-column block; its KS simdgroups take
// K slices, reduce their fm lanes with shuffles and then each other through
// threadgroup memory. Each weight is read and dequantized exactly once.
//
// Fused RMSNorm (decode passes), producer side: EPI_ADDP adds the residual,
// writes one partial sum of squares per row and 32-column block, and the
// last-arriving threadgroup (device-scope fence + atomic counter) sums the
// partials in a fixed order and writes the normalized fp16 row
// xn = h * rsqrt(ss / D + eps) * w for the *next* norm, so the consuming GEMV
// is the plain kernel with no extra work per weight (consumer-side
// normalization, NRM 1, measured 12% slower per step: every threadgroup paid
// a load and two multiplies per activation). Deterministic: the sum order
// does not depend on which threadgroup arrives last.
struct NormParams {
  int P;        // input partials per row (NRM 1)
  int part_ld;  // floats between input rows of `pin` (partials, or z rows for NRM 2)
  int D;        // normalized width
  float eps;
  int out_ld;   // floats between rows of `pout` (EPI_ADDP)
};

constant constexpr int EPI_ADDP = 3;

// Inverse RMS of MM rows from their P partial sums of squares: the KS
// simdgroups of the threadgroup each sum a slice (instead of every simdgroup
// reading all P partials) and combine through threadgroup memory in a fixed
// order, so the result is deterministic and costs P / (32 KS) loads per lane.
template <int MM, int KS>
static inline void row_inv(device const float* pin, constant NormParams& np, int M, ushort sgid, ushort lane,
                           threadgroup float (*red)[MM], thread float* inv) {
  UNROLL for (int m = 0; m < MM; ++m) {
    float s = 0.0f;
    if (m < M)
      for (int i = sgid * 32 + lane; i < np.P; i += KS * 32) s += pin[m * np.part_ld + i];
    s = simd_sum(s);
    if (lane == 0) red[sgid][m] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  UNROLL for (int m = 0; m < MM; ++m) {
    float t = 0.0f;
    UNROLL for (int k = 0; k < KS; ++k) t += red[k][m];
    inv[m] = rsqrt(t / float(np.D) + np.eps);
  }
}

// One K step's weight words for a lane: the nibble/byte plane (WPL uint4) and
// the high-bit plane of 5/6-bit codes.
#ifndef ESPETERO_PF
#define ESPETERO_PF 0
#endif

template <int BITS, int WPL>
static inline void load_kstep(device const uint4* wp, device const uint* hp, int g, thread uint* words, thread uint* hw) {
  UNROLL for (int i = 0; i < WPL; ++i) {
    const uint4 w = wp[ulong(g) * (32 * WPL) + i];
    words[4 * i] = w.x;
    words[4 * i + 1] = w.y;
    words[4 * i + 2] = w.z;
    words[4 * i + 3] = w.w;
  }
  if (BITS == 6) {
    const uint4 h = *((device const uint4*)(hp + ulong(g) * 32 * 4));
    hw[0] = h.x, hw[1] = h.y, hw[2] = h.z, hw[3] = h.w;
  } else if (BITS == 5) {
    const uint2 h = *((device const uint2*)(hp + ulong(g) * 32 * 2));
    hw[0] = h.x, hw[1] = h.y;
  }
}

// JIT (diagnostic instantiations only): each 8-wide k slice loads its MM
// activations right before its FMAs instead of all MM x 8 per K step up
// front, so a lane holds MM x 8 accumulators but only MM activations.
// LIST (store epilogue, no K split): threadgroup i computes column block
// blist[i] (-1: nothing), so a GEMV can cover a shortlist of blocks.
template <int BITS, int MM, int EPI, int KS, int NRM, int JIT = 0, int LIST = 0>
kernel void rgemv(
    device const void* Av [[buffer(0)]],
    device const uint* Wr [[buffer(1)]],
    device const half4* SB [[buffer(2)]],
    device const uint* Wh [[buffer(3)]],
    device void* Yv [[buffer(4)]],
    constant GemmParams& p [[buffer(5)]],
    constant NormParams& np [[buffer(6)]],
    device const float* wn [[buffer(7)]],
    device const float* pin [[buffer(8)]],
    device float* pout [[buffer(9)]],
    device half* y16 [[buffer(10)]],
    device atomic_int* ctr [[buffer(11)]],
    device atomic_int* kctr [[buffer(12)]],   // split-K: arrival counter per column block
    device float* ws [[buffer(13)]],          // split-K: partials [split][row][N]
    device const int* blist [[buffer(14)]],   // LIST: column block per threadgroup
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tgpg [[threadgroups_per_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int nb = LIST ? blist[tgid.x] : int(tgid.x);
  if (LIST && nb < 0) return;
  const int n0 = nb * 32;
  const int KGall = p.K / 64;
  // K split over threadgroups (grid y): matrices with few column blocks
  // would otherwise leave most cores idle. Each split reduces its partial
  // through the workspace; the last-arriving threadgroup of a column block
  // sums the splits in a fixed order and runs the epilogue.
  const int nsplit = p.g_split > 0 ? int(tgpg.y) : 1;
  const int gk0 = p.g_split > 0 ? int(tgid.y) * p.g_split : 0;
  const int gk1 = p.g_split > 0 ? min(KGall, gk0 + p.g_split) : KGall;
  const int per = (gk1 - gk0 + KS - 1) / KS;
  const int g_begin = gk0 + int(sgid) * per;
  const int g_end = min(gk1, g_begin + per);
  const int Np2 = (p.N + 31) / 32 * 16;
  constexpr int MAXG = 4;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  constexpr int WPL = BITS == 8 ? 4 : (BITS == 2 ? 1 : 2);
  constexpr int HPL = BITS == 6 ? 4 : (BITS == 5 ? 2 : 0);
  device const uint4* wp = (device const uint4*)(Wr) + ulong(nb) * KGall * (32 * WPL) + lane * WPL;
  device const uint* hp = Wh + (ulong(nb) * KGall * 32 + lane) * (HPL ? HPL : 1);
  device const half* A = (device const half*)Av;
  device const float* Af = (device const float*)Av;
  float inv[MM];
  threadgroup float nred[KS][MM];
  if (NRM == 1) row_inv<MM, KS>(pin, np, p.M, ushort(sgid), ushort(lane), nred, inv);
  float acc[MM][8];
  UNROLL for (int m = 0; m < MM; ++m)
    UNROLL for (int c = 0; c < 8; ++c) acc[m][c] = 0.0f;
  uint words[4 * WPL], hw[4] = {0u, 0u, 0u, 0u};
  if (ESPETERO_PF && g_begin < g_end) load_kstep<BITS, WPL>(wp, hp, g_begin, words, hw);
  for (int g = g_begin; g < g_end; ++g) {
    // Software pipeline: the next K step's weight words are requested before
    // this step's dequantization and FMAs, so two steps' loads are in flight
    // per simdgroup (the loop was latency-bound: ~2.7 us per 64-wide step).
    uint wnext[4 * WPL], hnext[4] = {0u, 0u, 0u, 0u};
    if (!ESPETERO_PF) load_kstep<BITS, WPL>(wp, hp, g, words, hw);
    if (ESPETERO_PF && g + 1 < g_end) load_kstep<BITS, WPL>(wp, hp, g + 1, wnext, hnext);
    half2 s2[MAXG * 4], b2[MAXG * 4];
    kstep_scales(SB, ulong(g), KGall, Np2, (n0 + fn) / 2, s2, b2);
    if (NRM == 2 && ((g - g_begin) & 1) == 0) {
      // Gated per-head norm (128-wide heads): inverse RMS of head g / 2 per row.
      UNROLL for (int m = 0; m < MM; ++m) {
        float4 v = m < p.M ? *((device const float4*)(Af + ulong(m) * p.lda + (g / 2) * 128 + lane * 4)) : float4(0.0f);
        inv[m] = rsqrt(simd_sum(dot(v, v)) / 128.0f + np.eps);
      }
    }
    // The activation of row m at column k (normalized on load for NRM 1/2).
    auto act = [&](int m, int k) -> float {
      if (NRM == 1) {
        // fp16 copy of the residual row (written by the producing GEMV's
        // EPI_ADDP epilogue): the same bytes as the unfused normalized row.
        return m < p.M ? float(A[ulong(m) * p.lda + k]) * inv[m] * wn[k] : 0.0f;
      } else if (NRM == 2) {
        const float z = m < p.M ? pin[ulong(m) * np.part_ld + k] : 0.0f;
        return m < p.M ? Af[ulong(m) * p.lda + k] * inv[m] * wn[k & 127] * (z / (1.0f + exp(-z))) : 0.0f;
      } else {
        return m < p.M ? float(A[ulong(m) * p.lda + k]) : 0.0f;
      }
    };
    float xv[JIT ? 1 : MM][8];
    if (!JIT) {
      UNROLL for (int m = 0; m < MM; ++m)
        UNROLL for (int jj = 0; jj < 8; ++jj) xv[JIT ? 0 : m][jj] = act(m, g * 64 + jj * 8 + fm);
    }
    UNROLL for (int jj = 0; jj < 8; ++jj) {
      float xj[JIT ? MM : 1];
      if (JIT) {
        UNROLL for (int m = 0; m < MM; ++m) xj[JIT ? m : 0] = act(m, g * 64 + jj * 8 + fm);
      }
      UNROLL for (int j = 0; j < 4; ++j) {
        half2 q;
        if (BITS == 2) {
          const uint x = words[j] >> (2 * jj);
          q = as_type<half2>((x & 0x00030003u) | 0x64006400u);
        } else if (BITS == 4) {
          const uint x = words[j * 2 + jj / 4] >> (4 * (jj % 4));
          q = as_type<half2>((x & 0x000F000Fu) | 0x64006400u);
        } else if (BITS == 6 || BITS == 5) {
          const uint x = (words[j * 2 + jj / 4] >> (4 * (jj % 4))) & 0x000F000Fu;
          const uint h = BITS == 6 ? (hw[j] >> (2 * jj)) & 0x00030003u
                                   : (hw[j / 2] >> (jj + 8 * (j % 2))) & 0x00010001u;
          q = as_type<half2>(x | (h << 4) | 0x64006400u);
        } else {
          const uint x = words[j * 4 + jj / 2] >> (8 * (jj % 2));
          q = as_type<half2>((x & 0x00FF00FFu) | 0x64006400u);
        }
        const int gi = (jj * 8) / kGroup;
        const float2 wv = float2(dequant_pair(q, s2[gi * 4 + j], b2[gi * 4 + j]));
        UNROLL for (int m = 0; m < MM; ++m) {
          const float x = JIT ? xj[JIT ? m : 0] : xv[JIT ? 0 : m][jj];
          acc[m][2 * j] = fma(x, wv.x, acc[m][2 * j]);
          acc[m][2 * j + 1] = fma(x, wv.y, acc[m][2 * j + 1]);
        }
      }
    }
    if (ESPETERO_PF) {
      UNROLL for (int i = 0; i < 4 * WPL; ++i) words[i] = wnext[i];
      UNROLL for (int i = 0; i < 4; ++i) hw[i] = hnext[i];
    }
  }
  // Sum the 8 lanes that hold different k rows (fm: lane bits 1, 2 and 4).
  UNROLL for (int m = 0; m < MM; ++m)
    UNROLL for (int c = 0; c < 8; ++c) {
      float v = acc[m][c];
      v += simd_shuffle_xor(v, ushort(2));
      v += simd_shuffle_xor(v, ushort(4));
      v += simd_shuffle_xor(v, ushort(16));
      acc[m][c] = v;
    }
  threadgroup float part[KS][MM][32];
  if (fm == 0) {
    UNROLL for (int m = 0; m < MM; ++m)
      UNROLL for (int j = 0; j < 4; ++j) {
        part[sgid][m][j * 8 + fn] = acc[m][2 * j];
        part[sgid][m][j * 8 + fn + 1] = acc[m][2 * j + 1];
      }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float red[MM][32];  // the block's 32 column sums per row
  const int c = lane;
  if (sgid == 0) UNROLL for (int m = 0; m < MM; ++m) {
    float v = 0.0f;
    UNROLL for (int k = 0; k < KS; ++k) v += part[k][m][c];
    red[m][c] = v;
  }
  if (nsplit > 1) {
    threadgroup int last_k;
    if (sgid == 0) UNROLL for (int m = 0; m < MM; ++m) {
      if (m < p.M && n0 + c < p.N)
        atomic_store_explicit((device atomic_uint*)(ws + ulong(tgid.y) * ulong(p.z_stride) + ulong(m) * p.N + n0 + c),
                              as_type<uint>(red[m][c]), memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0 && lane == 0) last_k = atomic_fetch_add_explicit(kctr + nb, 1, memory_order_relaxed) == nsplit - 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last_k) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
    if (sgid == 0) UNROLL for (int m = 0; m < MM; ++m) {
      float v = 0.0f;
      if (m < p.M && n0 + c < p.N)
        for (int sp = 0; sp < nsplit; ++sp)
          v += as_type<float>(atomic_load_explicit((device atomic_uint*)(ws + ulong(sp) * ulong(p.z_stride) + ulong(m) * p.N + n0 + c),
                                                   memory_order_relaxed));
      red[m][c] = v;
    }
    if (sgid == 0 && lane == 0) atomic_store_explicit(kctr + nb, 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (EPI != EPI_ADDP && sgid != 0) return;
  if (sgid == 0) UNROLL for (int m = 0; m < MM; ++m) {
    if (m >= p.M) break;
    const float v = red[m][c];
    if (EPI == EPI_SWIGLU) {
      // columns 0-7 / 16-23 are gate blocks, 8-15 / 24-31 their up blocks
      if ((c / 8) % 2 == 0 && n0 + c < p.N) {
        const float u = red[m][c + 8];
        const int out = ((n0 + (c / 8) * 8) / 16) * 8 + (c % 8);
        ((device half*)Yv)[ulong(m) * p.ldy + out] = half(glu_gate(v) * u);
      }
    } else if (EPI == EPI_ADDP) {
      // Values another threadgroup (the finalizer) reads inside this dispatch
      // are stored atomically: plain stores are not visible across
      // threadgroups before the dispatch ends on Apple GPUs, fences or not.
      float y = 0.0f;
      if (n0 + c < p.N) {
        device float* yp = (device float*)Yv + ulong(m) * p.ldy + n0 + c;
        y = *yp + v;
        atomic_store_explicit((device atomic_uint*)yp, as_type<uint>(y), memory_order_relaxed);
      }
      const float sq = simd_sum(y * y);
      if (lane == 0) atomic_store_explicit((device atomic_uint*)(pout + m * np.out_ld + nb), as_type<uint>(sq), memory_order_relaxed);
    } else if (n0 + c < p.N) {
      device float* y = (device float*)Yv + ulong(m) * p.ldy + n0 + c;
      *y = (EPI == EPI_ADD) ? *y + v : v;
    }
  }
  if (EPI == EPI_ADDP) {
    threadgroup int last_tg;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0 && lane == 0)
      last_tg = atomic_fetch_add_explicit(ctr, 1, memory_order_relaxed) == int(tgpg.x) - 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last_tg) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
    const int nblk = int(tgpg.x);
    float invr[MM];
    UNROLL for (int m = 0; m < MM; ++m) {
      float ss = 0.0f;
      if (m < p.M)
        for (int i = lane; i < nblk; i += 32)
          ss += as_type<float>(atomic_load_explicit((device atomic_uint*)(pout + m * np.out_ld + i), memory_order_relaxed));
      ss = simd_sum(ss);
      invr[m] = rsqrt(ss / float(np.D) + np.eps);
    }
    const int tid = int(sgid) * 32 + int(lane);
    device atomic_uint* Y = (device atomic_uint*)Yv;
    UNROLL for (int m = 0; m < MM; ++m) {
      if (m >= p.M) break;
      for (int k = tid * 4; k + 3 < p.N; k += KS * 32 * 4) {
        device atomic_uint* yr = Y + ulong(m) * p.ldy + k;
        const float4 v = float4(as_type<float>(atomic_load_explicit(yr, memory_order_relaxed)),
                                as_type<float>(atomic_load_explicit(yr + 1, memory_order_relaxed)),
                                as_type<float>(atomic_load_explicit(yr + 2, memory_order_relaxed)),
                                as_type<float>(atomic_load_explicit(yr + 3, memory_order_relaxed)));
        const float4 w = *((device const float4*)(wn + k));
        *((device half4*)(y16 + ulong(m) * p.ldy + k)) = half4(v * invr[m] * w);
      }
    }
    if (tid == 0) atomic_store_explicit(ctr, 0, memory_order_relaxed);
  }
}

#define RGEMV_ARGS                                                                   \
  device const void*, device const uint*, device const half4*, device const uint*,   \
      device void*, constant GemmParams&, constant NormParams&, device const float*, \
      device const float*, device float*, device half*, device atomic_int*,          \
      device atomic_int*, device float*, device const int*, uint3, uint3, uint, uint
#define INSTANTIATE_RGEMV_E(PRE, BITS, MM)                                                          \
  template [[host_name(PRE #MM "_store")]] kernel void rgemv<BITS, MM, 0, 8, 0>(RGEMV_ARGS);       \
  template [[host_name(PRE #MM "_add")]] kernel void rgemv<BITS, MM, 1, 8, 0>(RGEMV_ARGS);         \
  template [[host_name(PRE #MM "_swiglu")]] kernel void rgemv<BITS, MM, 2, 8, 0>(RGEMV_ARGS);      \
  template [[host_name(PRE #MM "_addp")]] kernel void rgemv<BITS, MM, 3, 8, 0>(RGEMV_ARGS);        \
  template [[host_name(PRE #MM "_store_n")]] kernel void rgemv<BITS, MM, 0, 8, 1>(RGEMV_ARGS);     \
  template [[host_name(PRE #MM "_swiglu_n")]] kernel void rgemv<BITS, MM, 2, 8, 1>(RGEMV_ARGS);     \
  template [[host_name(PRE #MM "_add_g")]] kernel void rgemv<BITS, MM, 1, 8, 2>(RGEMV_ARGS);        \
  template [[host_name(PRE #MM "_addp_g")]] kernel void rgemv<BITS, MM, 3, 8, 2>(RGEMV_ARGS);
#define INSTANTIATE_RGEMV(PRE, BITS) \
  INSTANTIATE_RGEMV_E(PRE, BITS, 1)  \
  INSTANTIATE_RGEMV_E(PRE, BITS, 2)
INSTANTIATE_RGEMV("rgemv4_", 4)
INSTANTIATE_RGEMV("rgemv5_", 5)
INSTANTIATE_RGEMV("rgemv6_", 6)
INSTANTIATE_RGEMV("rgemv8_", 8)
// One row, 16 K-slices per threadgroup (512 threads): GEMVs with few column
// blocks (the MoE's 2048-row projections: 64 threadgroups) keep twice the
// simdgroups in flight and still reduce inside the threadgroup.
#define INSTANTIATE_RGEMV_K16(PRE, BITS)                                                          \
  template [[host_name(PRE "1k16_store")]] kernel void rgemv<BITS, 1, 0, 16, 0>(RGEMV_ARGS);     \
  template [[host_name(PRE "1k16_add")]] kernel void rgemv<BITS, 1, 1, 16, 0>(RGEMV_ARGS);       \
  template [[host_name(PRE "1k16_swiglu")]] kernel void rgemv<BITS, 1, 2, 16, 0>(RGEMV_ARGS);    \
  template [[host_name(PRE "1k16_addp")]] kernel void rgemv<BITS, 1, 3, 16, 0>(RGEMV_ARGS);      \
  template [[host_name(PRE "1k16_store_n")]] kernel void rgemv<BITS, 1, 0, 16, 1>(RGEMV_ARGS);   \
  template [[host_name(PRE "1k16_swiglu_n")]] kernel void rgemv<BITS, 1, 2, 16, 1>(RGEMV_ARGS);   \
  template [[host_name(PRE "1k16_add_g")]] kernel void rgemv<BITS, 1, 1, 16, 2>(RGEMV_ARGS);      \
  template [[host_name(PRE "1k16_addp_g")]] kernel void rgemv<BITS, 1, 3, 16, 2>(RGEMV_ARGS);
INSTANTIATE_RGEMV_K16("rgemv4_", 4)
INSTANTIATE_RGEMV_K16("rgemv5_", 5)
INSTANTIATE_RGEMV_K16("rgemv6_", 6)
INSTANTIATE_RGEMV_K16("rgemv8_", 8)

// GEMV-shaped MMA kernel for 3..8 rows (draft verification passes): the
// rgemv layout (32 columns per threadgroup, K split over its 8 simdgroups,
// weights streamed straight into registers, no barriers in the loop) with
// 8x8 simdgroup MMAs over all rows instead of per-row FMAs. NRM / EPI_ADDP as
// in rgemv. DG (diagnostic instantiations only, wrong results): 1 skips
// the MMAs (one FMA per fragment keeps the operands live), 2 skips the weight
// loads (codes synthesized in registers), 3 skips the dequantization (raw
// codes into the MMA). LIST as in rgemv.
template <int BITS, int EPI, int NRM, int DG = 0, int LIST = 0>
kernel void rgemv8(
    device const void* Av [[buffer(0)]],
    device const uint* Wr [[buffer(1)]],
    device const half4* SB [[buffer(2)]],
    device const uint* Wh [[buffer(3)]],
    device void* Yv [[buffer(4)]],
    constant GemmParams& p [[buffer(5)]],
    constant NormParams& np [[buffer(6)]],
    device const float* wn [[buffer(7)]],
    device const float* pin [[buffer(8)]],
    device float* pout [[buffer(9)]],
    device half* y16 [[buffer(10)]],
    device atomic_int* ctr [[buffer(11)]],
    device const int* blist [[buffer(14)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tgpg [[threadgroups_per_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int KS = 8;
  const int nb = LIST ? blist[tgid.x] : int(tgid.x);
  if (LIST && nb < 0) return;
  const int n0 = nb * 32;
  const int KGall = p.K / 64;
  const int per = (KGall + KS - 1) / KS;
  const int g_begin = int(sgid) * per;
  const int g_end = min(KGall, g_begin + per);
  const int Np2 = (p.N + 31) / 32 * 16;
  constexpr int MAXG = 4;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  constexpr int WPL = BITS == 8 ? 4 : (BITS == 2 ? 1 : 2);
  constexpr int HPL = BITS == 6 ? 4 : (BITS == 5 ? 2 : 0);
  device const uint4* wp = (device const uint4*)(Wr) + ulong(nb) * KGall * (32 * WPL) + lane * WPL;
  device const uint* hp = Wh + (ulong(nb) * KGall * 32 + lane) * (HPL ? HPL : 1);
  const bool row_ok = fm < p.M;
  device const half* arow = (device const half*)Av + ulong(row_ok ? fm : 0) * p.lda;
  device const float* frow = (device const float*)Av + ulong(row_ok ? fm : 0) * p.lda;
  float inv_row = 1.0f;
  threadgroup float nred[KS][8];
  if (NRM == 1) {
    float inv8[8];
    row_inv<8, KS>(pin, np, p.M, ushort(sgid), ushort(lane), nred, inv8);
    UNROLL for (int m = 0; m < 8; ++m)
      if (m == fm) inv_row = inv8[m];
  }
  simdgroup_matrix<float, 8, 8> acc[4];
  UNROLL for (int j = 0; j < 4; ++j) acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (int g = g_begin; g < g_end; ++g) {
    uint words[4 * WPL];
    if (DG == 2) {
      UNROLL for (int i = 0; i < 4 * WPL; ++i) words[i] = (uint(g) * 0x9E3779B9u) ^ (lane * 0x85EBCA6Bu) ^ uint(i * 0x01010101);
    } else {
      UNROLL for (int i = 0; i < WPL; ++i) {
        const uint4 w = wp[ulong(g) * (32 * WPL) + i];
        words[4 * i] = w.x;
        words[4 * i + 1] = w.y;
        words[4 * i + 2] = w.z;
        words[4 * i + 3] = w.w;
      }
    }
    uint hw[4] = {0u, 0u, 0u, 0u};
    if (DG == 2) {
      UNROLL for (int i = 0; i < 4; ++i) hw[i] = words[i] * 0x2545F491u;
    } else if (BITS == 6) {
      const uint4 h = *((device const uint4*)(hp + ulong(g) * 32 * 4));
      hw[0] = h.x, hw[1] = h.y, hw[2] = h.z, hw[3] = h.w;
    } else if (BITS == 5) {
      const uint2 h = *((device const uint2*)(hp + ulong(g) * 32 * 2));
      hw[0] = h.x, hw[1] = h.y;
    }
    half2 s2[MAXG * 4], b2[MAXG * 4];
    kstep_scales(SB, ulong(g), KGall, Np2, (n0 + fn) / 2, s2, b2);
    if (NRM == 2 && ((g - g_begin) & 1) == 0) {
      // Gated per-head norm: the 4 lanes of a row (lane bits 0 and 3) each sum
      // 32 of the head's 128 values of that row.
      const int sub = (lane & 1) | ((lane >> 2) & 2);
      float ssq = 0.0f;
      if (row_ok) {
        device const float4* hp4 = (device const float4*)(frow + (g / 2) * 128 + sub * 32);
        UNROLL for (int i = 0; i < 8; ++i) {
          const float4 v = hp4[i];
          ssq += dot(v, v);
        }
      }
      ssq += simd_shuffle_xor(ssq, ushort(1));
      ssq += simd_shuffle_xor(ssq, ushort(8));
      inv_row = rsqrt(ssq / 128.0f + np.eps);
    }
    half2 av[8];
    UNROLL for (int jj = 0; jj < 8; ++jj) {
      const int k = g * 64 + jj * 8 + fn;
      if (NRM == 1) {
        av[jj] = row_ok ? half2(float2(*((device const half2*)(arow + k))) * inv_row * float2(wn[k], wn[k + 1]))
                        : half2(0.0h);
      } else if (NRM == 2) {
        if (row_ok) {
          const float2 z = *((device const float2*)(pin + ulong(fm) * np.part_ld + k));
          const float2 y = *((device const float2*)(frow + k));
          av[jj] = half2(y * inv_row * float2(wn[k & 127], wn[(k + 1) & 127]) * (z / (1.0f + exp(-z))));
        } else {
          av[jj] = half2(0.0h);
        }
      } else {
        av[jj] = row_ok ? *((device const half2*)(arow + k)) : half2(0.0h);
      }
    }
    UNROLL for (int jj = 0; jj < 8; ++jj) {
      simdgroup_matrix<half, 8, 8> a;
      a.thread_elements()[0] = av[jj].x;
      a.thread_elements()[1] = av[jj].y;
      UNROLL for (int j = 0; j < 4; ++j) {
        half2 q;
        if (BITS == 2) {
          const uint x = words[j] >> (2 * jj);
          q = as_type<half2>((x & 0x00030003u) | 0x64006400u);
        } else if (BITS == 4) {
          const uint x = words[j * 2 + jj / 4] >> (4 * (jj % 4));
          q = as_type<half2>((x & 0x000F000Fu) | 0x64006400u);
        } else if (BITS == 6 || BITS == 5) {
          const uint x = (words[j * 2 + jj / 4] >> (4 * (jj % 4))) & 0x000F000Fu;
          const uint h = BITS == 6 ? (hw[j] >> (2 * jj)) & 0x00030003u
                                   : (hw[j / 2] >> (jj + 8 * (j % 2))) & 0x00010001u;
          q = as_type<half2>(x | (h << 4) | 0x64006400u);
        } else {
          const uint x = words[j * 4 + jj / 2] >> (8 * (jj % 2));
          q = as_type<half2>((x & 0x00FF00FFu) | 0x64006400u);
        }
        const int gi = (jj * 8) / kGroup;
        const half2 wv = DG == 3 ? q : dequant_pair(q, s2[gi * 4 + j], b2[gi * 4 + j]);
        simdgroup_matrix<half, 8, 8> b;
        b.thread_elements()[0] = wv.x;
        b.thread_elements()[1] = wv.y;
        if (DG == 1)
          acc[j].thread_elements()[0] = fma(float(a.thread_elements()[0]), float(b.thread_elements()[1]),
                                            acc[j].thread_elements()[0] + float(b.thread_elements()[0]));
        else
          simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
      }
    }
  }
  threadgroup float part[KS][8][32];
  UNROLL for (int j = 0; j < 4; ++j) {
    part[sgid][fm][j * 8 + fn] = acc[j].thread_elements()[0];
    part[sgid][fm][j * 8 + fn + 1] = acc[j].thread_elements()[1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int row = int(sgid);  // 8 simdgroups: one output row each
  const int c = int(lane);
  if (EPI != EPI_ADDP && row >= p.M) return;
  if (row < p.M) {
    float v = 0.0f;
    UNROLL for (int k = 0; k < KS; ++k) v += part[k][row][c];
    if (EPI == EPI_SWIGLU) {
      if ((c / 8) % 2 == 0 && n0 + c < p.N) {
        float u = 0.0f;
        UNROLL for (int k = 0; k < KS; ++k) u += part[k][row][c + 8];
        const int out = ((n0 + (c / 8) * 8) / 16) * 8 + (c % 8);
        ((device half*)Yv)[ulong(row) * p.ldy + out] = half(glu_gate(v) * u);
      }
    } else if (EPI == EPI_ADDP) {
      float y = 0.0f;
      if (n0 + c < p.N) {
        device float* yp = (device float*)Yv + ulong(row) * p.ldy + n0 + c;
        y = *yp + v;
        atomic_store_explicit((device atomic_uint*)yp, as_type<uint>(y), memory_order_relaxed);
      }
      const float sq = simd_sum(y * y);
      if (lane == 0) atomic_store_explicit((device atomic_uint*)(pout + row * np.out_ld + nb), as_type<uint>(sq), memory_order_relaxed);
    } else if (n0 + c < p.N) {
      device float* y = (device float*)Yv + ulong(row) * p.ldy + n0 + c;
      *y = (EPI == EPI_ADD) ? *y + v : v;
    }
  }
  if (EPI == EPI_ADDP) {
    // Last-arriving threadgroup: normalized fp16 rows for the next norm (see rgemv).
    threadgroup int last_tg;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0 && lane == 0)
      last_tg = atomic_fetch_add_explicit(ctr, 1, memory_order_relaxed) == int(tgpg.x) - 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last_tg) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
    const int nblk = int(tgpg.x);
    float invr[8];
    UNROLL for (int m = 0; m < 8; ++m) {
      float ss = 0.0f;
      if (m < p.M)
        for (int i = lane; i < nblk; i += 32)
          ss += as_type<float>(atomic_load_explicit((device atomic_uint*)(pout + m * np.out_ld + i), memory_order_relaxed));
      ss = simd_sum(ss);
      invr[m] = rsqrt(ss / float(np.D) + np.eps);
    }
    const int tid = int(sgid) * 32 + int(lane);
    device atomic_uint* Y = (device atomic_uint*)Yv;
    for (int m = 0; m < p.M; ++m)
      for (int k = tid * 4; k + 3 < p.N; k += KS * 32 * 4) {
        device atomic_uint* yr = Y + ulong(m) * p.ldy + k;
        const float4 v = float4(as_type<float>(atomic_load_explicit(yr, memory_order_relaxed)),
                                as_type<float>(atomic_load_explicit(yr + 1, memory_order_relaxed)),
                                as_type<float>(atomic_load_explicit(yr + 2, memory_order_relaxed)),
                                as_type<float>(atomic_load_explicit(yr + 3, memory_order_relaxed)));
        const float4 w = *((device const float4*)(wn + k));
        *((device half4*)(y16 + ulong(m) * p.ldy + k)) = half4(v * invr[m] * w);
      }
    if (tid == 0) atomic_store_explicit(ctr, 0, memory_order_relaxed);
  }
}

#define RGEMV8_ARGS                                                                  \
  device const void*, device const uint*, device const half4*, device const uint*,   \
      device void*, constant GemmParams&, constant NormParams&, device const float*, \
      device const float*, device float*, device half*, device atomic_int*, device const int*, uint3, uint3, uint, uint
#define INSTANTIATE_RGEMV8(PRE, BITS)                                                       \
  template [[host_name(PRE "_store")]] kernel void rgemv8<BITS, 0, 0>(RGEMV8_ARGS);         \
  template [[host_name(PRE "_add")]] kernel void rgemv8<BITS, 1, 0>(RGEMV8_ARGS);           \
  template [[host_name(PRE "_swiglu")]] kernel void rgemv8<BITS, 2, 0>(RGEMV8_ARGS);        \
  template [[host_name(PRE "_addp")]] kernel void rgemv8<BITS, 3, 0>(RGEMV8_ARGS);          \
  template [[host_name(PRE "_store_n")]] kernel void rgemv8<BITS, 0, 1>(RGEMV8_ARGS);       \
  template [[host_name(PRE "_swiglu_n")]] kernel void rgemv8<BITS, 2, 1>(RGEMV8_ARGS);       \
  template [[host_name(PRE "_add_g")]] kernel void rgemv8<BITS, 1, 2>(RGEMV8_ARGS);          \
  template [[host_name(PRE "_addp_g")]] kernel void rgemv8<BITS, 3, 2>(RGEMV8_ARGS);
INSTANTIATE_RGEMV8("rgemv8r4", 4)
INSTANTIATE_RGEMV8("rgemv8r5", 5)
INSTANTIATE_RGEMV8("rgemv8r6", 6)
INSTANTIATE_RGEMV8("rgemv8r8", 8)

// Coarse MTP draft head: the LM head at 2 bits scanned in full, then
// the shortlisted column blocks of the file's own head (LIST variants).
template [[host_name("rgemv2_1_store")]] kernel void rgemv<2, 1, 0, 8, 0>(RGEMV_ARGS);
template [[host_name("rgemv2_2_store")]] kernel void rgemv<2, 2, 0, 8, 0>(RGEMV_ARGS);
template [[host_name("rgemv8r2_store")]] kernel void rgemv8<2, 0, 0>(RGEMV8_ARGS);
#define INSTANTIATE_LIST(BITS)                                                                                   \
  template [[host_name("rgemv" #BITS "_1_store_l")]] kernel void rgemv<BITS, 1, 0, 8, 0, 0, 1>(RGEMV_ARGS);     \
  template [[host_name("rgemv" #BITS "_2_store_l")]] kernel void rgemv<BITS, 2, 0, 8, 0, 0, 1>(RGEMV_ARGS);     \
  template [[host_name("rgemv8r" #BITS "_store_l")]] kernel void rgemv8<BITS, 0, 0, 0, 1>(RGEMV8_ARGS);
INSTANTIATE_LIST(4)
INSTANTIATE_LIST(5)
INSTANTIATE_LIST(6)
INSTANTIATE_LIST(8)

// Verify-pass anatomy (`bench_native --micro gemv`): diagnostic MMA
// variants and 3-4-row FMA GEMVs, store epilogue only. Compiled only when the
// host sets ESPETERO_DIAG=1, so serving builds do not pay for them.
#if ESPETERO_FMA34
#define INSTANTIATE_DIAG_JIT(BITS)  // the production JIT instantiations below have these names
#else
#define INSTANTIATE_DIAG_JIT(BITS)                                                                        \
  template [[host_name("rgemv" #BITS "_3j_store")]] kernel void rgemv<BITS, 3, 0, 8, 0, 1>(RGEMV_ARGS); \
  template [[host_name("rgemv" #BITS "_4j_store")]] kernel void rgemv<BITS, 4, 0, 8, 0, 1>(RGEMV_ARGS);
#endif
#if ESPETERO_DIAG
#define INSTANTIATE_DIAG(BITS)                                                                                    \
  template [[host_name("rgemv8r" #BITS "_d1_store")]] kernel void rgemv8<BITS, 0, 0, 1>(RGEMV8_ARGS);           \
  template [[host_name("rgemv8r" #BITS "_d2_store")]] kernel void rgemv8<BITS, 0, 0, 2>(RGEMV8_ARGS);           \
  template [[host_name("rgemv8r" #BITS "_d3_store")]] kernel void rgemv8<BITS, 0, 0, 3>(RGEMV8_ARGS);           \
  template [[host_name("rgemv" #BITS "_3_store")]] kernel void rgemv<BITS, 3, 0, 8, 0>(RGEMV_ARGS);             \
  template [[host_name("rgemv" #BITS "_4_store")]] kernel void rgemv<BITS, 4, 0, 8, 0>(RGEMV_ARGS);             \
  INSTANTIATE_DIAG_JIT(BITS)
INSTANTIATE_DIAG(4)
INSTANTIATE_DIAG(5)
INSTANTIATE_DIAG(6)
INSTANTIATE_DIAG(8)
#endif

// 3-4-row FMA GEMVs with just-in-time activations for verify passes (a
// follow-up to the verify-pass anatomy): compiled only when the host sets
// ESPETERO_FMA34=1, then served for 3-4 rows when gemv_max_m allows it
// (otherwise the MMA GEMV).
#if ESPETERO_FMA34
#define INSTANTIATE_RGEMV_J(PRE, BITS, MM)                                                               \
  template [[host_name(PRE #MM "j_store")]] kernel void rgemv<BITS, MM, 0, 8, 0, 1>(RGEMV_ARGS);        \
  template [[host_name(PRE #MM "j_add")]] kernel void rgemv<BITS, MM, 1, 8, 0, 1>(RGEMV_ARGS);          \
  template [[host_name(PRE #MM "j_swiglu")]] kernel void rgemv<BITS, MM, 2, 8, 0, 1>(RGEMV_ARGS);       \
  template [[host_name(PRE #MM "j_addp")]] kernel void rgemv<BITS, MM, 3, 8, 0, 1>(RGEMV_ARGS);         \
  template [[host_name(PRE #MM "j_store_n")]] kernel void rgemv<BITS, MM, 0, 8, 1, 1>(RGEMV_ARGS);      \
  template [[host_name(PRE #MM "j_swiglu_n")]] kernel void rgemv<BITS, MM, 2, 8, 1, 1>(RGEMV_ARGS);      \
  template [[host_name(PRE #MM "j_add_g")]] kernel void rgemv<BITS, MM, 1, 8, 2, 1>(RGEMV_ARGS);         \
  template [[host_name(PRE #MM "j_addp_g")]] kernel void rgemv<BITS, MM, 3, 8, 2, 1>(RGEMV_ARGS);
#define INSTANTIATE_RGEMV_J34(PRE, BITS) INSTANTIATE_RGEMV_J(PRE, BITS, 3) INSTANTIATE_RGEMV_J(PRE, BITS, 4)
INSTANTIATE_RGEMV_J34("rgemv4_", 4)
INSTANTIATE_RGEMV_J34("rgemv5_", 5)
INSTANTIATE_RGEMV_J34("rgemv6_", 6)
INSTANTIATE_RGEMV_J34("rgemv8_", 8)
#endif

// Deterministic split-K reduction: Y (+)= sum over splits in fixed order.
template <int EPI>
kernel void splitk_reduce(
    device const float* ws [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant GemmParams& p [[buffer(2)]],
    constant int& splits [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int col = gid.x * 4;
  const int row = gid.y;
  if (col >= p.N || row >= p.M) return;
  device const float* src = ws + ulong(row) * p.N + col;
  float4 acc = float4(0.0f);
  if (col + 3 < p.N) {
    for (int z = 0; z < splits; ++z) acc += *((device const float4*)(src + ulong(z) * p.z_stride));
    device float4* y = (device float4*)(Y + ulong(row) * p.ldy + col);
    *y = (EPI == EPI_ADD) ? *y + acc : acc;
  } else {
    for (int c = 0; c < 4 && col + c < p.N; ++c) {
      float v = 0.0f;
      for (int z = 0; z < splits; ++z) v += src[ulong(z) * p.z_stride + c];
      device float* y = Y + ulong(row) * p.ldy + col + c;
      *y = (EPI == EPI_ADD) ? *y + v : v;
    }
  }
}

template [[host_name("splitk_reduce_store")]] kernel void splitk_reduce<0>(
    device const float*, device float*, constant GemmParams&, constant int&, uint2);
template [[host_name("splitk_reduce_add")]] kernel void splitk_reduce<1>(
    device const float*, device float*, constant GemmParams&, constant int&, uint2);

#define INSTANTIATE_RGEMM_B(PRE, TAG, BM, WN, BITS)                                  \
  template [[host_name(PRE TAG "_store")]] kernel void rgemm<BM, WN, 0, BITS>(      \
      device const half*, device const uint*, device const half4*,                 \
      device const uint*, device void*, constant GemmParams&, uint3, uint, uint, uint); \
  template [[host_name(PRE TAG "_add")]] kernel void rgemm<BM, WN, 1, BITS>(        \
      device const half*, device const uint*, device const half4*,                 \
      device const uint*, device void*, constant GemmParams&, uint3, uint, uint, uint); \
  template [[host_name(PRE TAG "_swiglu")]] kernel void rgemm<BM, WN, 2, BITS>(     \
      device const half*, device const uint*, device const half4*,                 \
      device const uint*, device void*, constant GemmParams&, uint3, uint, uint, uint);
#define INSTANTIATE_RGEMM(TAG, BM, WN)            \
  INSTANTIATE_RGEMM_B("rgemm_", TAG, BM, WN, 4)   \
  INSTANTIATE_RGEMM_B("rgemm5_", TAG, BM, WN, 5)  \
  INSTANTIATE_RGEMM_B("rgemm6_", TAG, BM, WN, 6)  \
  INSTANTIATE_RGEMM_B("rgemm8_", TAG, BM, WN, 8)

// Tile configs: rows per threadgroup x simdgroups (each 32 columns).
INSTANTIATE_RGEMM("8x4", 8, 4)
INSTANTIATE_RGEMM("16x4", 16, 4)
INSTANTIATE_RGEMM("32x4", 32, 4)
INSTANTIATE_RGEMM("16x8", 16, 8)
INSTANTIATE_RGEMM("32x8", 32, 8)
INSTANTIATE_RGEMM("8x8", 8, 8)
INSTANTIATE_RGEMM("64x4", 64, 4)


// Quantized GPU layout (Wr, SB, Wh) -> dense fp16 hgemm layout (Hr), with
// the very half-precision FMA rgemm applies to each B fragment, so a matrix
// served densely gives bit-identical GEMM operands. One thread per (column
// block, K step, lane).
template <int BITS>
kernel void dequant_hr(
    device const uint* Wr [[buffer(0)]],
    device const half4* SB [[buffer(1)]],
    device const uint* Wh [[buffer(2)]],
    device uint4* Hr [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const int KGall = K / 64;
  const int lane = gid % 32;
  const int bg = gid / 32;
  const int nb = bg / KGall, g = bg % KGall;
  if (nb >= (N + 31) / 32) return;
  const int Np2 = (N + 31) / 32 * 16;
  const int NG = 64 / kGroup;
  const short qid = lane / 4;
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  constexpr int WPL = BITS == 8 ? 4 : 2;
  constexpr int HPL = BITS == 6 ? 4 : (BITS == 5 ? 2 : 0);
  const ulong slot = (ulong(nb) * KGall + g) * 32 + lane;
  device const uint* wp = Wr + slot * (4 * WPL);
  uint words[4 * WPL];
  UNROLL for (int i = 0; i < 4 * WPL; ++i) words[i] = wp[i];
  uint hw[4] = {0u, 0u, 0u, 0u};
  if (HPL) {
    UNROLL for (int i = 0; i < HPL; ++i) hw[i] = Wh[slot * HPL + i];
  }
  half2 s2[16], b2[16];
  kstep_scales(SB, ulong(g), KGall, Np2, (nb * 32 + fn) / 2, s2, b2);
  UNROLL for (int jj = 0; jj < 8; ++jj) {
    uint out[4];
    UNROLL for (int j = 0; j < 4; ++j) {
      half2 q;
      if (BITS == 4) {
        const uint x = words[j * 2 + jj / 4] >> (4 * (jj % 4));
        q = as_type<half2>((x & 0x000F000Fu) | 0x64006400u);
      } else if (BITS == 6 || BITS == 5) {
        const uint x = (words[j * 2 + jj / 4] >> (4 * (jj % 4))) & 0x000F000Fu;
        const uint h = BITS == 6 ? (hw[j] >> (2 * jj)) & 0x00030003u
                                 : (hw[j / 2] >> (jj + 8 * (j % 2))) & 0x00010001u;
        q = as_type<half2>(x | (h << 4) | 0x64006400u);
      } else {
        const uint x = words[j * 4 + jj / 2] >> (8 * (jj % 2));
        q = as_type<half2>((x & 0x00FF00FFu) | 0x64006400u);
      }
      const int gi = (jj * 8) / kGroup;
      out[j] = as_type<uint>(dequant_pair(q, s2[gi * 4 + j], b2[gi * 4 + j]));
    }
    Hr[slot * 8 + jj] = uint4(out[0], out[1], out[2], out[3]);
  }
}
template [[host_name("dequant_hr4")]] kernel void dequant_hr<4>(device const uint*, device const half4*, device const uint*, device uint4*, constant int&, constant int&, uint);
template [[host_name("dequant_hr5")]] kernel void dequant_hr<5>(device const uint*, device const half4*, device const uint*, device uint4*, constant int&, constant int&, uint);
template [[host_name("dequant_hr6")]] kernel void dequant_hr<6>(device const uint*, device const half4*, device const uint*, device uint4*, constant int&, constant int&, uint);
template [[host_name("dequant_hr8")]] kernel void dequant_hr<8>(device const uint*, device const half4*, device const uint*, device uint4*, constant int&, constant int&, uint);

// Dense fp16 weights [N, K] -> Hr[N/32][K/64][lane][jj][j] half2 pairs
// (rows n, n+1 of fragment column j at k-step jj), zero-padded to 32 x 64.
kernel void repack_f16(
    device const half* W [[buffer(0)]],
    device uint4* Hr [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& Kp [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const int KG = Kp / 64;
  const int lane = gid % 32;
  const int bg = gid / 32;
  const int nb = bg / KG;
  const int g = bg % KG;
  if (nb >= (N + 31) / 32) return;
  const int qid = lane / 4;
  const int fm = (qid & 4) + ((lane / 2) % 4);
  const int fn = (qid & 2) * 2 + (lane % 2) * 2;
  device uint4* dst = Hr + (ulong(bg) * 32 + lane) * 8;
  UNROLL for (int jj = 0; jj < 8; ++jj) {
    const int k = g * 64 + jj * 8 + fm;
    uint w[4];
    UNROLL for (int j = 0; j < 4; ++j) {
      const int n = nb * 32 + j * 8 + fn;
      const half a = (n < N && k < K) ? W[ulong(n) * K + k] : half(0);
      const half b = (n + 1 < N && k < K) ? W[ulong(n + 1) * K + k] : half(0);
      w[j] = as_type<uint>(half2(a, b));
    }
    dst[jj] = uint4(w[0], w[1], w[2], w[3]);
  }
}

struct HGemmParams {
  int M;
  int N;
  int K;    // padded to a multiple of 64 (A columns past the real K are zero)
  int lda;
  int ldy;
  int has_bias;
};

constant constexpr int HEPI_STORE = 0;      // fp32 = acc + bias
constant constexpr int HEPI_ADD = 1;        // fp32 += acc + bias
constant constexpr int HEPI_GELU_TANH = 2;  // fp16 = gelu_tanh(acc + bias)
constant constexpr int HEPI_GELU_ERF = 3;   // fp16 = gelu(acc + bias)
constant constexpr int HEPI_SWIGLU = 4;     // fp16 [M, N/2], gate/up interleaved in blocks of 8
constant constexpr int HEPI_GEGLU = 5;      // as HEPI_SWIGLU with a tanh-GELU gate (Gemma 4)

// erf with |error| < 1.5e-7 (Abramowitz & Stegun 7.1.26).
static inline float erf_as(float x) {
  const float ax = fabs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * ax);
  const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t +
                          0.254829592f) * t * exp(-ax * ax);
  return copysign(y, x);
}
static inline float gelu_erf(float x) { return 0.5f * x * (1.0f + erf_as(x * 0.70710678118654752f)); }
static inline float gelu_tanh(float x) {
  return 0.5f * x * (1.0f + precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

template <int BM, int WN, int EPI, int WL>
kernel void hgemm(
    device const half* A [[buffer(0)]],
    device const uint4* Wr [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device void* Yv [[buffer(4)]],
    constant HGemmParams& p [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BK = 64;
  constexpr int BN = WN * 32;
  constexpr int NT = WN * 32;
  constexpr int FM = BM / 8;
  constexpr int LDA = BK + 8;
  constexpr int AC = (BM * 8 + NT - 1) / NT;
  threadgroup half As[BM * LDA];

  const int m0 = tgid.x * BM;  // row blocks fastest (see rgemm)
  const int n0 = tgid.y * BN + sgid * 32;
  const int KG = p.K / 64;
  const bool n_ok = n0 < p.N;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;

  simdgroup_matrix<float, 8, 8> acc[FM][4];
  UNROLL for (int i = 0; i < FM; ++i)
    UNROLL for (int j = 0; j < 4; ++j)
      acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  int a_row[AC], a_col[AC];
  bool a_ok[AC], a_live[AC];
  UNROLL for (int c = 0; c < AC; ++c) {
    const int idx = tid + c * NT;
    a_live[c] = idx < BM * 8;
    a_row[c] = min(idx / 8, BM - 1);
    a_col[c] = (idx % 8) * 8;
    a_ok[c] = a_live[c] && (m0 + a_row[c]) < p.M;
  }
  uint4 ar[AC];
  UNROLL for (int c = 0; c < AC; ++c)
    ar[c] = a_ok[c] ? *((device const uint4*)(A + ulong(m0 + a_row[c]) * p.lda + a_col[c])) : uint4(0);
  device const uint4* wp = Wr + ulong(n_ok ? n0 / 32 : 0) * KG * 256 + lane * 8;

  for (int g = 0; g < KG; ++g) {
    UNROLL for (int c = 0; c < AC; ++c)
      if (a_live[c]) *((threadgroup uint4*)(As + a_row[c] * LDA + a_col[c])) = ar[c];
    // Weight fragments are loaded WL sub-steps at a time: all 8 for tall
    // tiles, 2 for short ones (fewer live registers, higher occupancy).
    uint4 w[WL];
    UNROLL for (int i = 0; i < WL; ++i) w[i] = wp[ulong(g) * 256 + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (g + 1 < KG) {
      UNROLL for (int c = 0; c < AC; ++c)
        ar[c] = a_ok[c] ? *((device const uint4*)(A + ulong(m0 + a_row[c]) * p.lda + (g + 1) * 64 + a_col[c])) : uint4(0);
    }
    UNROLL for (int jj = 0; jj < 8; ++jj) {
      if (jj % WL == 0 && jj > 0) {
        UNROLL for (int i = 0; i < WL; ++i) w[i] = wp[ulong(g) * 256 + jj + i];
      }
      simdgroup_matrix<half, 8, 8> a[FM];
      simdgroup_matrix<half, 8, 8> b[4];
      UNROLL for (int i = 0; i < FM; ++i) {
        const half2 v = *((const threadgroup half2*)(As + (i * 8 + fm) * LDA + jj * 8 + fn));
        a[i].thread_elements()[0] = v.x;
        a[i].thread_elements()[1] = v.y;
      }
      const uint4 wj = w[jj % WL];
      const uint wq[4] = {wj.x, wj.y, wj.z, wj.w};
      UNROLL for (int j = 0; j < 4; ++j) {
        const half2 wv = as_type<half2>(wq[j]);
        b[j].thread_elements()[0] = wv.x;
        b[j].thread_elements()[1] = wv.y;
      }
      UNROLL for (int i = 0; i < FM; ++i)
        UNROLL for (int j = 0; j < 4; ++j)
          simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  UNROLL for (int i = 0; i < FM; ++i) {
    const int row = m0 + i * 8 + fm;
    if (row >= p.M) continue;
    if (EPI == HEPI_SWIGLU || EPI == HEPI_GEGLU) {
      device half* y = (device half*)Yv + ulong(row) * p.ldy;
      UNROLL for (int j = 0; j < 4; j += 2) {
        const int col = n0 + j * 8;
        if (col >= p.N) continue;
        const int out = (col / 16) * 8 + fn;
        const float g0 = acc[i][j].thread_elements()[0];
        const float g1 = acc[i][j].thread_elements()[1];
        const float u0 = acc[i][j + 1].thread_elements()[0];
        const float u1 = acc[i][j + 1].thread_elements()[1];
        if (EPI == HEPI_GEGLU) *((device half2*)(y + out)) = half2(half(gelu_tanh(g0) * u0), half(gelu_tanh(g1) * u1));
        else *((device half2*)(y + out)) = half2(half(silu(g0) * u0), half(silu(g1) * u1));
      }
      continue;
    }
    UNROLL for (int j = 0; j < 4; ++j) {
      const int col = n0 + j * 8 + fn;
      if (col >= p.N) continue;
      const bool two = col + 1 < p.N;
      const float bias0 = p.has_bias ? bias[col] : 0.0f;
      const float bias1 = (p.has_bias && two) ? bias[col + 1] : 0.0f;
      const float v0 = acc[i][j].thread_elements()[0] + bias0;
      const float v1 = two ? acc[i][j].thread_elements()[1] + bias1 : 0.0f;
      if (EPI == HEPI_STORE || EPI == HEPI_ADD) {
        device float* y = (device float*)Yv + ulong(row) * p.ldy + col;
        if (two) {
          float2 v = float2(v0, v1);
          if (EPI == HEPI_ADD) v += *((device float2*)y);
          *((device float2*)y) = v;
        } else {
          y[0] = (EPI == HEPI_ADD) ? y[0] + v0 : v0;
        }
      } else {
        device half* y = (device half*)Yv + ulong(row) * p.ldy + col;
        const float g0 = EPI == HEPI_GELU_TANH ? gelu_tanh(v0) : gelu_erf(v0);
        const float g1 = EPI == HEPI_GELU_TANH ? gelu_tanh(v1) : gelu_erf(v1);
        if (two) *((device half2*)y) = half2(half(g0), half(g1));
        else y[0] = half(g0);
      }
    }
  }
}

#define INSTANTIATE_HGEMM(TAG, BM, WN, WL)                                           \
  template [[host_name("hgemm_" TAG "_store")]] kernel void hgemm<BM, WN, 0, WL>(    \
      device const half*, device const uint4*, device const float*, device void*,    \
      constant HGemmParams&, uint3, uint, uint, uint);                               \
  template [[host_name("hgemm_" TAG "_add")]] kernel void hgemm<BM, WN, 1, WL>(      \
      device const half*, device const uint4*, device const float*, device void*,    \
      constant HGemmParams&, uint3, uint, uint, uint);                               \
  template [[host_name("hgemm_" TAG "_gelu_tanh")]] kernel void hgemm<BM, WN, 2, WL>( \
      device const half*, device const uint4*, device const float*, device void*,    \
      constant HGemmParams&, uint3, uint, uint, uint);                               \
  template [[host_name("hgemm_" TAG "_gelu_erf")]] kernel void hgemm<BM, WN, 3, WL>( \
      device const half*, device const uint4*, device const float*, device void*,    \
      constant HGemmParams&, uint3, uint, uint, uint);                               \
  template [[host_name("hgemm_" TAG "_swiglu")]] kernel void hgemm<BM, WN, 4, WL>(   \
      device const half*, device const uint4*, device const float*, device void*,    \
      constant HGemmParams&, uint3, uint, uint, uint);                               \
  template [[host_name("hgemm_" TAG "_geglu")]] kernel void hgemm<BM, WN, 5, WL>(    \
      device const half*, device const uint4*, device const float*, device void*,    \
      constant HGemmParams&, uint3, uint, uint, uint);

INSTANTIATE_HGEMM("32x4", 32, 4, 8)
INSTANTIATE_HGEMM("16x4", 16, 4, 2)
INSTANTIATE_HGEMM("8x8", 8, 8, 8)
