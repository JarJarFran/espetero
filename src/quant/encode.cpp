// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Port notes. The reference runs in float32 on the GPU (MLX); this runs in
// float32 on the CPU with the same expressions in the same order where the
// order is defined (candidate loops, roundings to fp16 / bf16, ties: mx.round
// is round-half-to-even, MLX's quantize kernel rounds half away from zero).
// Reductions (sums over a sub-block, means) may add in another order, so a
// near-tie between two candidate scales can resolve the other way: codes
// match the reference's almost everywhere, GPTQ's error feedback then
// spreads such differences (same quality, different codes).

#define ACCELERATE_NEW_LAPACK
#include "encode.h"

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "../model/impl/common.h"

// No fused multiply-adds: the reference rounds each product before it adds
// it (MLX multiplies, then reduces), and with contraction near-tied
// candidates resolve differently (codebook fits drift from the reference).
#pragma clang fp contract(off)

namespace espetero::quant {

using native::fail;
using native::parallel_for;

namespace {

constexpr int kSuper = 256;
// Super-blocks of a row of C inputs (C % 64 == 0): 256 inputs each, the last
// one partial when C % 256 != 0 (640 = 256 + 256 + 128); width of block sb.
inline int n_supers(int C) { return (C + kSuper - 1) / kSuper; }
inline int super_width(int C, int sb) { return std::min(kSuper, C - sb * kSuper); }

inline float rint_even(float x) { return __builtin_rintf(x); }  // mx.round (default rounding mode: ties to even)
inline float round_away(float x) { return std::round(x); }       // Metal round(): MLX's quantize kernel

// Rows [0, R) in blocks of `rows_per` for parallel_for.
template <class F>
void for_row_blocks(int64_t R, int64_t rows_per, F&& f) {
  const size_t nb = size_t((R + rows_per - 1) / rows_per);
  parallel_for(nb, [&](size_t b) {
    const int64_t r0 = int64_t(b) * rows_per, r1 = std::min<int64_t>(R, r0 + rows_per);
    f(r0, r1);
  });
}

double damp_factor() {
  const char* v = std::getenv("QL_DAMP");
  return v && *v ? std::atof(v) : 0.01;
}

// ---------------------------------------------------------------- EQ (symmetric two-level)

EncOpts g_opts;

// Signed scale of one 16-wide sub-block: the element
// of largest magnitude maps near -2^(B-1) (end_hi: near 2^(B-1) - 1); 19
// candidates, the least-squares scale for each candidate's codes, the largest
// error reduction wins (returned in *gain when asked).
float sym_search(const float* x, const float* w, int bits, bool rtn, bool end_hi = false, float* gain = nullptr) {
  const int nmax = 1 << (bits - 1);
  int imax = 0;
  float amax = std::fabs(x[0]);
  for (int i = 1; i < 16; ++i)
    if (std::fabs(x[i]) > amax) {
      amax = std::fabs(x[i]);
      imax = i;
    }
  float mval = x[imax];
  if (!(std::fabs(mval) > 0)) mval = 1e-12f;
  if (rtn) return -mval / float(nmax);
  const float lo = float(-nmax), hi = float(nmax - 1);
  float best_s = 0, best_q = 0;
  for (int i = -9; i <= 9; ++i) {
    const float iscale = end_hi ? float(double(nmax - 1) + 0.1 * i) / mval : float(-(double(nmax) + 0.1 * i)) / mval;
    float sxl = 0, sl2 = 0;
    for (int k = 0; k < 16; ++k) {
      const float l = std::clamp(rint_even(iscale * x[k]), lo, hi);
      const float wl = w ? w[k] * l : l;
      sxl += wl * x[k];
      sl2 += wl * l;
    }
    const float sc = sl2 > 0 ? sxl / sl2 : -mval / float(nmax);
    const float q = sl2 > 0 ? sxl * sxl / sl2 : 0.0f;
    if (i == -9 || q > best_q) {
      best_s = sc;
      best_q = q;
    }
  }
  if (gain) *gain = best_q;
  return best_s;
}

// The sub-block scale search of the run: production's orientation, or both
// orientations (EncOpts::sign; the larger error reduction wins, ties keep
// production's).
float sym_search_opt(const float* x, const float* w, int bits, bool rtn, bool both) {
  if (rtn || !both) return sym_search(x, w, bits, rtn);
  float g0 = 0, g1 = 0;
  const float s0 = sym_search(x, w, bits, false, false, &g0);
  const float s1 = sym_search(x, w, bits, false, true, &g1);
  return g1 > g0 ? s1 : s0;
}

// Two-level form of nt (16; fewer in a partial last super-block) sub-block
// scales: fp16 super-scale amax / 127, int8
// sub-scales, and the products half(sc * d) the codes are chosen for.
void two_level(const float* s, uint16_t& d_out, int8_t* sc_out, float* s2, int nt = 16) {
  float amax = 0;
  for (int t = 0; t < nt; ++t) amax = std::max(amax, std::fabs(s[t]));
  const float d = f16_round(amax / 127.0f);
  const float dv = d > 0 ? d : 1.0f;
  for (int t = 0; t < nt; ++t) {
    const float sc = std::clamp(rint_even(s[t] / dv), -127.0f, 127.0f);
    sc_out[t] = int8_t(sc);
    s2[t] = f16_round(sc * d);
  }
  d_out = f16_bits(d);
}

inline float eq_code(float x, float s, int bits) {
  const int nmax = 1 << (bits - 1);
  const float ss = s != 0 ? s : 1.0f;
  return std::clamp(rint_even(x / ss), float(-nmax), float(nmax - 1));
}

// One super-block (n = 256 permuted inputs of one row, fewer in a partial
// last block) without error feedback.
void eq_superblock(const float* x, const float* w, const Fmt& f, uint16_t& d, int8_t* sc, float* codes, int n = kSuper) {
  float s[16], s2[16];
  const bool rtn = f.enc == Fmt::RTN;
  const int nt = n / 16;
  for (int t = 0; t < nt; ++t) s[t] = sym_search_opt(x + 16 * t, w ? w + 16 * t : nullptr, f.bits, rtn, g_opts.sign);
  two_level(s, d, sc, s2, nt);
  for (int k = 0; k < n; ++k) codes[k] = eq_code(x[k], s2[k / 16], f.bits);
}

// ---------------------------------------------------------------- affine

// The affine search (MLX affine, one group): nstep 0 is
// min/max round to nearest, 20 the weighted least-squares search.
void affine_search(const float* v, const float* w, int g, int bits, int nstep, const std::string& sd, float& s_out,
                   float& b_out) {
  float lo = INFINITY, hi = -INFINITY;
  for (int i = 0; i < g; ++i) {
    lo = std::min(lo, v[i]);
    hi = std::max(hi, v[i]);
  }
  const float nmax = float((1 << bits) - 1);
  const float rng = std::max(hi - lo, 1e-12f);
  float best_s = rng / nmax, best_b = lo, best_e = INFINITY;
  for (int st = 0; st <= nstep; ++st) {
    const float iscale = nstep == 0 ? nmax / rng : (-1.0f + 0.1f * float(st) + nmax) / rng;
    float sw = 0, sl = 0, sl2 = 0, sxl = 0, sx = 0;
    for (int i = 0; i < g; ++i) {
      const float l = std::clamp(rint_even(iscale * (v[i] - lo)), 0.0f, nmax);
      const float wi = w ? w[i] : 1.0f;
      sw += wi;
      sl += wi * l;
      sl2 += wi * l * l;
      sxl += wi * l * v[i];
      sx += wi * v[i];
    }
    float sc = rng / nmax, bi = lo;
    if (nstep > 0) {
      const float D = sw * sl2 - sl * sl;
      if (D > 0) {
        sc = (sw * sxl - sx * sl) / D;
        bi = (sl2 * sx - sl * sxl) / D;
      }
      if (!(sc > 0)) {
        sc = rng / nmax;
        bi = lo;
      }
    }
    float e = 0;
    for (int i = 0; i < g; ++i) {
      const float l = std::clamp(rint_even(iscale * (v[i] - lo)), 0.0f, nmax);
      const float dd = sc * l + bi - v[i];
      e += (w ? w[i] : 1.0f) * dd * dd;
    }
    if (e < best_e) {
      best_e = e;
      best_s = sc;
      best_b = bi;
    }
  }
  if (sd == "bf16") {
    best_s = bf16_round_f(best_s);
    best_b = bf16_round_f(best_b);
  } else if (sd == "f16") {
    best_s = f16_round(best_s);
    best_b = f16_round(best_b);
  }
  if (!(best_s > 0)) best_s = 1e-12f;
  s_out = best_s;
  b_out = best_b;
}

inline float affine_code(float x, float s, float b, int bits) {
  return std::clamp(rint_even((x - b) / s), 0.0f, float((1 << bits) - 1));
}

// MLX's mx.quantize (affine, its Metal kernel): codes from the unrounded
// scale and bias, ties away from zero; scale and bias stored in sd.
void mlx_quantize_group(const float* v, int g, int bits, const std::string& sd, uint8_t* q, float& s_out, float& b_out) {
  const float n_bins = float((1 << bits) - 1);
  float w_max = v[0], w_min = v[0];
  for (int i = 1; i < g; ++i) {
    w_max = std::max(w_max, v[i]);
    w_min = std::min(w_min, v[i]);
  }
  float scale = std::max((w_max - w_min) / n_bins, 1e-7f);
  const bool side = std::fabs(w_min) > std::fabs(w_max);
  scale = side ? scale : -scale;
  const float edge = side ? w_min : w_max;
  const float q0 = round_away(edge / scale);
  const bool at_zero = q0 == 0.0f;
  scale = at_zero ? scale : edge / q0;
  const float bias = at_zero ? 0.0f : edge;
  for (int i = 0; i < g; ++i) q[i] = uint8_t(std::clamp(round_away((v[i] - bias) / scale), 0.0f, n_bins));
  if (sd == "bf16") {
    s_out = bf16_round_f(scale);
    b_out = bf16_round_f(bias);
  } else if (sd == "f16") {
    s_out = f16_round(scale);
    b_out = f16_round(bias);
  } else {
    s_out = scale;
    b_out = bias;
  }
}

// ---------------------------------------------------------------- GPTQ

// The upper Cholesky factor U of H^-1 (H^-1 = U^T U) after the reference encoder's
// damping: dead inputs (diag <= 0) get 1 on the diagonal, every input
// damp * mean(live diagonal). Row-major [C][C]; dead flags out.
// Eigen-decomposition (double) of a symmetric [C][C] matrix: eigenvalues ascending in w, eigenvectors
// as the columns of the column-major V (= rows of the row-major array). False when LAPACK fails.
bool sym_eig(std::vector<double>& V, int C, std::vector<double>& w) {
  __LAPACK_int n = C, info = 0, lwork = -1, liwork = -1, iq = 0;
  double wq = 0;
  w.assign(size_t(C), 0.0);
  dsyevd_("V", "L", &n, V.data(), &n, w.data(), &wq, &lwork, &iq, &liwork, &info);
  if (info != 0) return false;
  lwork = __LAPACK_int(wq);
  liwork = iq;
  std::vector<double> work(static_cast<size_t>(lwork));
  std::vector<__LAPACK_int> iwork(static_cast<size_t>(liwork));
  dsyevd_("V", "L", &n, V.data(), &n, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
  return info == 0;
}

// H (float, [C][C] symmetric) projected to PSD when it is indefinite: λ_min < -1e-9 λ_max (a capture
// defect; Σ x xᵀ is PSD). Returns false (Hp untouched) when it is not indefinite.
bool psd_project(const float* H, int C, std::vector<double>& Hp) {
  std::vector<double> V(static_cast<size_t>(C) * size_t(C));
  for (size_t k = 0; k < V.size(); ++k) V[k] = double(H[k]);
  std::vector<double> w;
  if (!sym_eig(V, C, w)) return false;
  const double lmax = std::max(std::fabs(w.back()), 1e-300);
  if (!(w.front() < -1e-9 * lmax)) return false;
  const double lmin = w.front();
  int neg = 0;
  for (double& l : w)
    if (l < 0) {
      l = 0;
      ++neg;
    }
  std::fprintf(stderr, "\n[quantize] GPTQ Hessian (%d inputs) indefinite: lambda_min %.4g (lambda_max %.4g, %d negative); projected to PSD\n",
               C, lmin, lmax, neg);
  // Hp = V diag(w+) V^T; columns of V scaled by sqrt(w+).
  for (int j = 0; j < C; ++j) {
    const double s = std::sqrt(w[size_t(j)]);
    double* col = V.data() + size_t(j) * C;
    for (int i = 0; i < C; ++i) col[i] *= s;
  }
  Hp.assign(V.size(), 0.0);
  cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, C, C, 1.0, V.data(), C, 0.0, Hp.data(), C);
  for (int j = 0; j < C; ++j)
    for (int i = j + 1; i < C; ++i) Hp[size_t(j) * C + i] = Hp[size_t(i) * C + j] = Hp[size_t(j) * C + i];
  return true;
}

std::vector<float> hinv_upper(const float* H, int C, std::vector<uint8_t>& dead, std::vector<float>* Hproj = nullptr) {
  const double damp = damp_factor();
  std::vector<float> A(static_cast<size_t>(C) * size_t(C));
  std::memcpy(A.data(), H, A.size() * sizeof(float));
  dead.assign(size_t(C), 0);
  double sum = 0;
  for (int i = 0; i < C; ++i) {
    const float d = H[size_t(i) * C + i];
    dead[size_t(i)] = d <= 0;
    sum += d <= 0 ? 0.0 : double(d);
  }
  for (int i = 0; i < C; ++i)
    if (dead[size_t(i)]) A[size_t(i) * C + i] += 1.0f;
  const float lam = float(damp) * float(sum / C);
  for (int i = 0; i < C; ++i) A[size_t(i) * C + i] += lam;
  // Symmetric: the row-major array is also the column-major one.
  __LAPACK_int n = C, info = 0;
  spotrf_("L", &n, A.data(), &n, &info);
  if (info == 0) spotri_("L", &n, A.data(), &n, &info);
  if (info == 0) {
    for (int j = 0; j < C; ++j)  // column-major lower -> full symmetric
      for (int i = j + 1; i < C; ++i) A[size_t(i) * C + j] = A[size_t(j) * C + i] = A[size_t(i) + size_t(j) * C];
    spotrf_("L", &n, A.data(), &n, &info);
  }
  if (info != 0) {
    // Hessians with a few massive-activation inputs (Gemma 4: diagonal from
    // 1e-7 to 5e3) can lose definiteness in float32: the same steps in
    // double precision, with 10x to 10000x the damping if needed (Qwen3.8-27B's
    // layer-0 linear_attn.out_proj, captured on prose, is indefinite: 1000x).
    // An indefinite one (a capture defect: Σ x xᵀ is PSD) is projected to PSD first (EncOpts::psd):
    // damping it until it factorizes costs the matrix up to 2.4x its error (27B, EQ2 tensor lab E7d).
    std::vector<double> Hpsd;
    const bool projected = g_opts.psd && psd_project(H, C, Hpsd);
    if (projected && Hproj) {
      Hproj->resize(Hpsd.size());
      for (size_t k = 0; k < Hpsd.size(); ++k) (*Hproj)[k] = float(Hpsd[k]);
    }
    std::vector<double> D(A.size());
    for (double mult : {1.0, 10.0, 100.0, 1000.0, 10000.0}) {
      for (int i = 0; i < C; ++i)
        for (int j = 0; j < C; ++j) D[size_t(i) * C + j] = projected ? Hpsd[size_t(i) * C + j] : double(H[size_t(i) * C + j]);
      for (int i = 0; i < C; ++i) D[size_t(i) * C + i] += (dead[size_t(i)] ? 1.0 : 0.0) + mult * damp * (sum / C);
      info = 0;
      dpotrf_("L", &n, D.data(), &n, &info);
      if (info == 0) dpotri_("L", &n, D.data(), &n, &info);
      if (info == 0) {
        for (int j = 0; j < C; ++j)
          for (int i = j + 1; i < C; ++i) D[size_t(i) * C + j] = D[size_t(j) * C + i] = D[size_t(i) + size_t(j) * C];
        dpotrf_("L", &n, D.data(), &n, &info);
      }
      if (info == 0) {
        std::fprintf(stderr, "\n[quantize] GPTQ Hessian (%d inputs): float32 Cholesky failed, double precision with %.0fx damping\n", C, mult);
        for (size_t k = 0; k < D.size(); ++k) A[k] = float(D[k]);
        break;
      }
    }
    if (info != 0) fail("GPTQ: the damped Hessian is not positive definite (potrf " + std::to_string(info) + ")");
  }
  // Column-major L (lower) read row-major is L^T = U; clear the rest.
  for (int j = 0; j < C; ++j)
    for (int i = 0; i < j; ++i) A[size_t(i) + size_t(j) * C] = 0.0f;
  return A;
}

// GPTQ over rows Wr [R][C] (fp32, in place, columns in the encoder's
// order): groups of g columns; at each group's start `fit(r, w, scales)`
// fits the row's scales on the updated weights, then `code(r, k, w)` gives
// column k's quantized value; the column's error feeds the group's later
// columns (rank-1 with U's row), the group's errors the remaining columns
// (one GEMM).
template <class Fit, class Code>
void gptq(float* Wr, int64_t R, int C, const float* U, int g, Fit&& fit, Code&& code) {
  std::vector<float> Eg(static_cast<size_t>(R) * size_t(g));
  for (int i = 0; i < C; i += g) {
    const int gw = std::min(g, C - i);  // (EQ: a partial last super-block)
    const int j = i + gw;
    for_row_blocks(R, 16, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        float* w = Wr + size_t(r) * C + i;
        fit(r, i, w);
        float* e = Eg.data() + size_t(r) * g;
        for (int k = 0; k < gw; ++k) {
          const float q = code(r, i, k, w[k]);
          const float* u = U + size_t(i + k) * C + i;
          const float ek = (w[k] - q) / u[k];
          for (int c = k; c < gw; ++c) w[c] -= ek * u[c];
          e[k] = ek;
        }
      }
    });
    if (j < C)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, int(R), C - j, gw, -1.0f, Eg.data(), g,
                  U + size_t(i) * C + j, C, 1.0f, Wr + j, C);
  }
}

// GPTQ for EQ / EQC with the sub-scales chosen inside the recurrence (EncOpts::schur, the EQ2 tensor
// lab's E9 "m2"): at each super-block the sub-block searches on the updated weights give the fp16
// super-scale d (amax / 127, as production); then, sub-block by sub-block, the scale is searched again
// on the weights as updated by the earlier sub-blocks, and every int8 sub-scale within +-rad of it (and
// of the best scale of either grid orientation, `alt`) is replayed through the recurrence over the
// sub-block's 16 columns: the one with the least sum of e_k^2 (e_k = (w_k - q_k) / U_kk, the sub-block's
// exact increase of the damped objective with the later columns compensating) is kept.
// search(x16, col) / alt(x16, col): a sub-block's signed scale; val(w, s): the quantized value of w at
// scale s; emit(r, col, w, s): the same, recording the code. sc_out [R][C / 16], d_out [R][nsup].
template <class Search, class Alt, class Val, class Emit>
void gptq_e9(float* Wr, int64_t R, int C, const float* U, int rad, Search&& search, Alt&& alt, Val&& val, Emit&& emit,
             int8_t* sc_out, uint16_t* d_out) {
  const int g = kSuper, nsup = n_supers(C);
  std::vector<float> Eg(static_cast<size_t>(R) * size_t(g));
  for (int i = 0; i < C; i += g) {
    const int gw = std::min(g, C - i);
    const int j = i + gw, nt = gw / 16;
    for_row_blocks(R, 16, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        float* w = Wr + size_t(r) * C + i;
        float* e = Eg.data() + size_t(r) * g;
        float s[16];
        float amax = 0;
        for (int t = 0; t < nt; ++t) {
          s[t] = search(w + 16 * t, i + 16 * t);
          amax = std::max(amax, std::fabs(s[t]));
        }
        const float d = f16_round(amax / 127.0f);
        const float dv = d > 0 ? d : 1.0f;
        d_out[size_t(r) * size_t(nsup) + i / kSuper] = f16_bits(d);
        for (int t = 0; t < nt; ++t) {
          const int k0 = 16 * t;
          const float st = t == 0 ? s[0] : search(w + k0, i + k0);
          const float sc0 = std::clamp(rint_even(st / dv), -127.0f, 127.0f);
          float best = sc0;
          if (rad > 0) {
            const float sc1 = std::clamp(rint_even(alt(w + k0, i + k0) / dv), -127.0f, 127.0f);
            double best_l = INFINITY;
            for (int base = 0; base < 2; ++base) {
              if (base == 1 && sc1 == sc0) continue;
              const float b0 = base == 0 ? sc0 : sc1;
              for (int dd = -rad; dd <= rad; ++dd) {
                const float c = std::clamp(b0 + float(dd), -127.0f, 127.0f);
                if (c == 0) continue;
                const float sv = f16_round(c * d);
                if (sv == 0) continue;
                float tmp[16];
                std::memcpy(tmp, w + k0, sizeof tmp);
                double l = 0;
                for (int k = 0; k < 16; ++k) {
                  const float* u = U + size_t(i + k0 + k) * C + i + k0;
                  const float ek = (tmp[k] - val(tmp[k], sv)) / u[k];
                  l += double(ek) * double(ek);
                  for (int c2 = k + 1; c2 < 16; ++c2) tmp[c2] -= ek * u[c2];
                }
                if (l < best_l) {
                  best_l = l;
                  best = c;
                }
              }
            }
          }
          float sv = f16_round(best * d);
          if (sv == 0) sv = 1e-8f;
          sc_out[size_t(r) * (C / 16) + (i + k0) / 16] = int8_t(best);
          for (int k = k0; k < k0 + 16; ++k) {
            const float q = emit(r, i + k, w[k], sv);
            const float* u = U + size_t(i + k) * C + i;
            const float ek = (w[k] - q) / u[k];
            for (int c = k; c < gw; ++c) w[c] -= ek * u[c];
            e[k] = ek;
          }
        }
      }
    });
    if (j < C)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, int(R), C - j, gw, -1.0f, Eg.data(), g,
                  U + size_t(i) * C + j, C, 1.0f, Wr + j, C);
  }
}

// Least-squares refit of each row's fp16 super-scales with codes and int8 sub-scales fixed (EncOpts::refit):
// d = argmin (x - A d)^T H (x - A d), A's column j = sub-scale * code value on super-block j; kept per row
// when finite, positive and lower in that objective than the old d. X(r, xr) fills row r's weights in
// lane order; V(r, vr) its code values (code - 2^(B-1), or the codebook entry); H [C][C] lane order.
template <class FX, class FV>
void refit_super(int64_t R, int C, const float* H, const int8_t* sc, uint16_t* d, FX&& X, FV&& V) {
  const int nsup = n_supers(C);
  const int64_t chunk = std::max<int64_t>(64, std::min<int64_t>(4096, (int64_t(1) << 27) / (int64_t(C) * 4)));
  std::vector<float> A, Xc, HX, Z;
  int64_t kept = 0;
  for (int64_t r0 = 0; r0 < R; r0 += chunk) {
    const int64_t n = std::min(chunk, R - r0);
    A.assign(size_t(n) * C, 0.0f);
    Xc.assign(size_t(n) * C, 0.0f);
    for_row_blocks(n, 16, [&](int64_t a0, int64_t a1) {
      std::vector<float> v(static_cast<size_t>(C));
      for (int64_t a = a0; a < a1; ++a) {
        X(r0 + a, Xc.data() + size_t(a) * C);
        V(r0 + a, v.data());
        const int8_t* s = sc + size_t(r0 + a) * (C / 16);
        for (int k = 0; k < C; ++k) A[size_t(a) * C + k] = float(s[k / 16]) * v[size_t(k)];
      }
    });
    HX.assign(size_t(n) * C, 0.0f);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, int(n), C, C, 1.0f, Xc.data(), C, H, C, 0.0f, HX.data(), C);
    std::vector<double> G(size_t(n) * nsup * nsup), b(size_t(n) * nsup);
    for (int l = 0; l < nsup; ++l) {
      const int c0 = l * kSuper, cw = super_width(C, l);
      Z.assign(size_t(n) * C, 0.0f);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, int(n), C, cw, 1.0f, A.data() + c0, C, H + size_t(c0) * C, C, 0.0f,
                  Z.data(), C);
      for_row_blocks(n, 64, [&](int64_t a0, int64_t a1) {
        for (int64_t a = a0; a < a1; ++a) {
          const float* ar = A.data() + size_t(a) * C;
          const float* zr = Z.data() + size_t(a) * C;
          for (int jb = 0; jb < nsup; ++jb) {
            double s = 0;
            for (int k = jb * kSuper; k < jb * kSuper + super_width(C, jb); ++k) s += double(ar[k]) * double(zr[k]);
            G[(size_t(a) * nsup + jb) * nsup + l] = s;
          }
          if (l == 0)
            for (int jb = 0; jb < nsup; ++jb) {
              double s = 0;
              const float* hx = HX.data() + size_t(a) * C;
              for (int k = jb * kSuper; k < jb * kSuper + super_width(C, jb); ++k) s += double(ar[k]) * double(hx[k]);
              b[size_t(a) * nsup + jb] = s;
            }
        }
      });
    }
    std::vector<uint8_t> took(size_t(n), 0);
    for_row_blocks(n, 64, [&](int64_t a0, int64_t a1) {
      std::vector<double> M(static_cast<size_t>(nsup) * nsup), y(static_cast<size_t>(nsup));
      for (int64_t a = a0; a < a1; ++a) {
        const double* Ga = G.data() + size_t(a) * nsup * nsup;
        const double* ba = b.data() + size_t(a) * nsup;
        // symmetrize, tiny ridge, Cholesky solve
        double tr = 0;
        for (int p = 0; p < nsup; ++p) tr += Ga[p * nsup + p];
        for (int p = 0; p < nsup; ++p)
          for (int q = 0; q < nsup; ++q) M[size_t(p) * nsup + q] = 0.5 * (Ga[p * nsup + q] + Ga[q * nsup + p]);
        for (int p = 0; p < nsup; ++p) M[size_t(p) * nsup + p] += 1e-12 * (tr / nsup) + 1e-300;
        std::memcpy(y.data(), ba, sizeof(double) * size_t(nsup));
        __LAPACK_int nn = nsup, one = 1, info = 0;
        dposv_("L", &nn, &one, M.data(), &nn, y.data(), &nn, &info);
        if (info != 0) continue;
        uint16_t* dr = d + size_t(r0 + a) * nsup;
        std::vector<double> dnv(static_cast<size_t>(nsup)), doldv(static_cast<size_t>(nsup));
        double* dn = dnv.data();
        double* dold = doldv.data();
        bool ok = true;
        for (int p = 0; ok && p < nsup; ++p) {
          dn[p] = double(f16_round(float(y[size_t(p)])));
          __fp16 h;
          std::memcpy(&h, dr + p, 2);
          dold[p] = double(float(h));
          if (!std::isfinite(dn[p]) || !(dn[p] > 0)) ok = false;
        }
        if (!ok) continue;
        auto obj = [&](const double* x) {
          double o = 0;
          for (int p = 0; p < nsup; ++p) {
            double gx = 0;
            for (int q = 0; q < nsup; ++q) gx += 0.5 * (Ga[p * nsup + q] + Ga[q * nsup + p]) * x[q];
            o += x[p] * gx - 2.0 * ba[p] * x[p];
          }
          return o;
        };
        if (obj(dn) < obj(dold)) {
          for (int p = 0; p < nsup; ++p) dr[p] = f16_bits(float(dn[p]));
          took[size_t(a)] = 1;
        }
      }
    });
    for (uint8_t t : took) kept += t;
  }
  static const bool dbg = std::getenv("EQ_DEBUG") != nullptr;
  if (dbg) std::fprintf(stderr, "[refit] %lld of %lld rows refit\n", (long long)kept, (long long)R);
}

// ---------------------------------------------------------------- codebooks (EQC)

const float kCb4Init[16] = {-8.98f, -7.41f, -5.84f, -4.63f, -3.59f, -2.68f, -1.85f, -1.05f,
                            -0.20f, 0.67f,  1.58f,  2.55f,  3.59f,  4.71f,  6.04f,  7.80f};
// EQ3's start: the uniform grid code - 4.
const float kCb3Init[8] = {-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};

// Codebooks have L = 2^bits entries (EQ4 16, EQ3 8) and L - 1 midpoints.
inline int cb_nearest(float u, const float* mids, int L) {
  if (L > 16) {  // EQ5 / EQ6 codebooks (sorted): the count of midpoints below u, by bisection
    int lo = 0, hi = L - 1;
    while (lo < hi) {
      const int m = (lo + hi) / 2;
      if (u > mids[m]) lo = m + 1;
      else hi = m;
    }
    return lo;
  }
  int q = 0;
  for (int j = 0; j < L - 1; ++j) q += u > mids[j] ? 1 : 0;
  return q;
}

// Codebook scales for one sub-block: the largest element mapped near
// either end of the codebook, 19 stretches each, least-squares scale.
float cb_scale(const float* x, const float* w, const float* cb, const float* mids, int L) {
  int imax = 0;
  float amax = std::fabs(x[0]);
  for (int i = 1; i < 16; ++i)
    if (std::fabs(x[i]) > amax) {
      amax = std::fabs(x[i]);
      imax = i;
    }
  float mval = x[imax];
  if (!(std::fabs(mval) > 0)) mval = 1e-12f;
  float best_s = 0, best_q = 0;
  bool first = true;
  for (const float end : {cb[0], cb[L - 1]}) {
    for (int i = -9; i <= 9; ++i) {
      const float s = mval / (end * float(1.0 + 0.0125 * i));
      float sxv = 0, sv2 = 0;
      for (int k = 0; k < 16; ++k) {
        const float v = cb[cb_nearest(x[k] / s, mids, L)];
        const float wv = w[k] * v;
        sxv += wv * x[k];
        sv2 += wv * v;
      }
      const float sc = sv2 > 0 ? sxv / sv2 : s;
      const float gq = sv2 > 0 ? sxv * sxv / sv2 : 0.0f;
      if (first || gq > best_q) {
        best_s = sc;
        best_q = gq;
        first = false;
      }
    }
  }
  return best_s;
}

// cb_scale for four sub-blocks at once, one per vector lane (NEON): each
// lane does exactly the scalar version's operations in its order (IEEE
// division, products rounded before they are added), so the result is
// bit-identical; the codebook lookup is a byte table lookup (over a copy
// padded to 16 entries: an 8-entry codebook's codes never index the padding).
void cb_scale4(const float* x, const float* w, const float* cb, const float* mids, int L, float* out) {
  float X[16][4], Wt[16][4], mv[4];
  for (int l = 0; l < 4; ++l) {
    const float* xl = x + 16 * l;
    int imax = 0;
    float amax = std::fabs(xl[0]);
    for (int i = 1; i < 16; ++i)
      if (std::fabs(xl[i]) > amax) {
        amax = std::fabs(xl[i]);
        imax = i;
      }
    mv[l] = std::fabs(xl[imax]) > 0 ? xl[imax] : 1e-12f;
    for (int k = 0; k < 16; ++k) {
      X[k][l] = xl[k];
      Wt[k][l] = w[16 * l + k];
    }
  }
  float cb16[16] = {};
  std::memcpy(cb16, cb, sizeof(float) * size_t(L));
  uint8x16x4_t tbl;
  tbl.val[0] = vreinterpretq_u8_f32(vld1q_f32(cb16));
  tbl.val[1] = vreinterpretq_u8_f32(vld1q_f32(cb16 + 4));
  tbl.val[2] = vreinterpretq_u8_f32(vld1q_f32(cb16 + 8));
  tbl.val[3] = vreinterpretq_u8_f32(vld1q_f32(cb16 + 12));
  float32x4_t md[15];
  for (int j = 0; j < L - 1; ++j) md[j] = vdupq_n_f32(mids[j]);
  float32x4_t xv[16], wvv[16];
  for (int k = 0; k < 16; ++k) {
    xv[k] = vld1q_f32(X[k]);
    wvv[k] = vld1q_f32(Wt[k]);
  }
  const float32x4_t mval = vld1q_f32(mv), zero = vdupq_n_f32(0.0f), one = vdupq_n_f32(1.0f);
  const uint32x4_t byte_off = vdupq_n_u32(0x03020100u), x4 = vdupq_n_u32(0x04040404u);
  float32x4_t best_s = zero, best_q = zero;
  bool first = true;
  for (const float end : {cb[0], cb[L - 1]}) {
    for (int i = -9; i <= 9; ++i) {
      const float32x4_t s = vdivq_f32(mval, vdupq_n_f32(end * float(1.0 + 0.0125 * i)));
      float32x4_t sxv = zero, sv2 = zero;
      for (int k = 0; k < 16; ++k) {
        const float32x4_t u = vdivq_f32(xv[k], s);
        uint32x4_t q = vdupq_n_u32(0);
        for (int j = 0; j < L - 1; ++j) q = vsubq_u32(q, vcgtq_f32(u, md[j]));
        const uint8x16_t idx = vreinterpretq_u8_u32(vaddq_u32(vmulq_u32(q, x4), byte_off));
        const float32x4_t v = vreinterpretq_f32_u8(vqtbl4q_u8(tbl, idx));
        const float32x4_t wv = vmulq_f32(wvv[k], v);
        sxv = vaddq_f32(sxv, vmulq_f32(wv, xv[k]));
        sv2 = vaddq_f32(sv2, vmulq_f32(wv, v));
      }
      const uint32x4_t pos = vcgtq_f32(sv2, zero);
      const float32x4_t den = vbslq_f32(pos, sv2, one);
      const float32x4_t sc = vbslq_f32(pos, vdivq_f32(sxv, den), s);
      const float32x4_t g = vbslq_f32(pos, vdivq_f32(vmulq_f32(sxv, sxv), den), zero);
      if (first) {
        best_s = sc;
        best_q = g;
        first = false;
      } else {
        const uint32x4_t better = vcgtq_f32(g, best_q);
        best_s = vbslq_f32(better, sc, best_s);
        best_q = vbslq_f32(better, g, best_q);
      }
    }
  }
  vst1q_f32(out, best_s);
}

// numpy's legacy RandomState(seed).choice(n, k, replace=False) (MT19937,
// permutation by Fisher-Yates with masked rejection): the rows the reference
// fits a dense matrix's codebook on.
std::vector<int64_t> np_choice(uint32_t seed, int64_t n, int64_t k) {
  uint32_t mt[624];
  int mti = 624;
  mt[0] = seed;
  for (int i = 1; i < 624; ++i) mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + uint32_t(i);
  auto next = [&]() -> uint32_t {
    if (mti >= 624) {
      for (int i = 0; i < 624; ++i) {
        const uint32_t y = (mt[i] & 0x80000000u) | (mt[(i + 1) % 624] & 0x7fffffffu);
        mt[i] = mt[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
      }
      mti = 0;
    }
    uint32_t y = mt[mti++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
  };
  std::vector<int64_t> a(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) a[size_t(i)] = i;
  for (int64_t i = n - 1; i >= 1; --i) {
    uint64_t mask = uint64_t(i);
    for (int sh = 1; sh <= 32; sh <<= 1) mask |= mask >> sh;
    uint64_t v;
    if (uint64_t(i) <= 0xffffffffull) {
      do v = next() & mask;
      while (v > uint64_t(i));
    } else {
      do v = ((uint64_t(next()) << 32) | next()) & mask;
      while (v > uint64_t(i));
    }
    std::swap(a[size_t(i)], a[size_t(v)]);
  }
  a.resize(size_t(k));
  std::sort(a.begin(), a.end());
  return a;
}

// Codebook fit (Lloyd): rows Xp [R][C] (lane order), importance per row (or
// shared), codebook group per row; `iters` Lloyd rounds then the encoding.
// out: codes in lane order [R][C] (when q_out), d, sc; cb [ng][L] updated in place.
void eqc_fit(const float* Xp, int64_t R, int C, const std::function<const float*(int64_t)>& imp_row,
             const std::function<int(int64_t)>& grp, int ng, int iters, std::vector<float>& cb, uint8_t* q_out,
             uint16_t* d_out, int8_t* sc_out, int L) {
  const int nsb = n_supers(C);
  const size_t Lz = size_t(L), Mz = size_t(L - 1);
  for (int it = 0; it <= iters; ++it) {
    const bool last = it == iters;
    std::vector<float> mids(static_cast<size_t>(ng) * Mz);
    for (int gi = 0; gi < ng; ++gi)
      for (int j = 0; j < L - 1; ++j) mids[size_t(gi) * Mz + j] = (cb[size_t(gi) * Lz + j + 1] + cb[size_t(gi) * Lz + j]) / 2;
    // Lloyd terms as the reference adds them (float32): per row over its
    // inputs, per chunk of (1 << 21) / C rows over its rows, then over chunks.
    static const bool dbg = std::getenv("EQ_DEBUG") != nullptr;
    std::vector<float> err(dbg ? size_t(R) : 0);
    std::vector<float> rn(last ? 0 : size_t(R) * Lz), rd(rn.size());
    const int64_t per = 32;
    const size_t nblk = size_t((R + per - 1) / per);
    parallel_for(nblk, [&](size_t bi) {
      const int64_t r0 = int64_t(bi) * per, r1 = std::min<int64_t>(R, r0 + per);
      for (int64_t r = r0; r < r1; ++r) {
        const float* x = Xp + size_t(r) * C;
        const float* w = imp_row(r);
        const int gi = grp(r);
        const float* c = cb.data() + size_t(gi) * Lz;
        const float* m = mids.data() + size_t(gi) * Mz;
        float* nr = last ? nullptr : rn.data() + size_t(r) * Lz;
        float* dr = last ? nullptr : rd.data() + size_t(r) * Lz;
        for (int sb = 0; sb < nsb; ++sb) {
          const float* xs = x + sb * kSuper;
          const float* ws = w + sb * kSuper;
          const int sw = super_width(C, sb), nt = sw / 16;
          float s[16];
          static const bool scalar = std::getenv("EQ_SCALAR") != nullptr;
          if (scalar || L > 16)
            for (int t = 0; t < nt; ++t) s[t] = cb_scale(xs + 16 * t, ws + 16 * t, c, m, L);
          else
            for (int t = 0; t < nt; t += 4) cb_scale4(xs + 16 * t, ws + 16 * t, c, m, L, s + t);
          float amax = 0;
          for (int t = 0; t < nt; ++t) amax = std::max(amax, std::fabs(s[t]));
          const float d = f16_round(amax / 127.0f);
          const float dv = d > 0 ? d : 1.0f;
          float s2[16];
          for (int t = 0; t < nt; ++t) {
            const float sc = std::clamp(rint_even(s[t] / dv), -127.0f, 127.0f);
            s2[t] = f16_round(sc * d);
            if (s2[t] == 0) s2[t] = 1e-8f;
            if (last) sc_out[size_t(r) * (C / 16) + sb * 16 + t] = int8_t(sc);
          }
          if (last) d_out[size_t(r) * nsb + sb] = f16_bits(d);
          for (int k = 0; k < sw; ++k) {
            const float st = s2[k / 16];
            const int q = cb_nearest(xs[k] / st, m, L);
            if (dbg) {
              const float e = xs[k] - st * c[q];
              err[size_t(r)] += ws[k] * e * e;
            }
            if (last) {
              q_out[size_t(r) * C + sb * kSuper + k] = uint8_t(q);
            } else {
              const float wk = ws[k] * st;
              nr[q] += wk * xs[k];
              dr[q] += wk * st;
            }
          }
        }
      }
    });
    if (dbg) {
      double e = 0;
      for (float v : err) e += v;
      std::fprintf(stderr, "[eqc] R %lld it %d err %.7g cb", (long long)R, it, e);
      for (int q = 0; q < L; ++q) std::fprintf(stderr, " %.8g", cb[size_t(q)]);
      std::fprintf(stderr, "\n");
    }
    if (last) break;
    std::vector<float> num(size_t(ng) * Lz, 0.0f), den(num.size(), 0.0f), cn(num.size()), cd(num.size());
    const int64_t step = std::max<int64_t>(1, (int64_t(1) << 21) / C);
    for (int64_t c0 = 0; c0 < R; c0 += step) {
      std::fill(cn.begin(), cn.end(), 0.0f);
      std::fill(cd.begin(), cd.end(), 0.0f);
      for (int64_t r = c0; r < std::min(R, c0 + step); ++r) {
        const size_t gL = size_t(grp(r)) * Lz;
        for (int q = 0; q < L; ++q) {
          cn[gL + q] += rn[size_t(r) * Lz + q];
          cd[gL + q] += rd[size_t(r) * Lz + q];
        }
      }
      for (size_t i = 0; i < num.size(); ++i) {
        num[i] += cn[i];
        den[i] += cd[i];
      }
    }
    for (int gi = 0; gi < ng; ++gi) {
      float nc[64];
      for (int q = 0; q < L; ++q) {
        const size_t i = size_t(gi) * Lz + q;
        nc[q] = f16_round(den[i] > 0 ? num[i] / std::max(den[i], 1e-30f) : cb[i]);
      }
      std::sort(nc, nc + L);
      std::memcpy(cb.data() + size_t(gi) * Lz, nc, sizeof(float) * Lz);
    }
  }
}

// The importance a matrix's encoder weighs inputs by (imat):
// diag(H) or the stored diagonal; per expert for stacked experts (experts
// never routed: uniform).
struct Imp {
  std::vector<float> v;  // [rows_][C]
  int rows = 0;          // 0: none (unit weights), 1: shared, E: per expert
};

const float* stats_diag(const Stats* st, int C, std::vector<float>& tmp) {
  if (!st) return nullptr;
  if (st->diag) return st->diag;
  if (st->H) {
    tmp.resize(size_t(C));
    for (int i = 0; i < C; ++i) tmp[size_t(i)] = st->H[size_t(i) * C + i];
    return tmp.data();
  }
  return nullptr;
}

}  // namespace

// ---------------------------------------------------------------- public helpers

float f16_round(float x) { return float(__fp16(x)); }
uint16_t f16_bits(float x) {
  const __fp16 h = __fp16(x);
  uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}
float bf16_round_f(float x) { return native::bf16_round(x); }
uint16_t bf16_bits(float x) { return native::f32_to_bf16_bits(x); }

std::vector<int> lane_perm(int C) {
  std::vector<int> p;
  p.reserve(size_t(C));
  for (int st = 0; st < C / 64; ++st)
    for (int t = 0; t < 4; ++t)
      for (int jj = 0; jj < 8; ++jj)
        for (int f = 0; f < 2; ++f) p.push_back(st * 64 + jj * 8 + 2 * t + f);
  return p;
}

Fmt parse_fmt(const std::string& spec) {
  Fmt f;
  f.spec = spec;
  const size_t plus = spec.find('+');
  const std::string base = spec.substr(0, plus);
  const std::string enc = plus == std::string::npos ? (spec.rfind("eqv:", 0) == 0 ? "gptq" : "rtn") : spec.substr(plus + 1);
  if (enc == "rtn") f.enc = Fmt::RTN;
  else if (enc == "mse") f.enc = Fmt::MSE;
  else if (enc == "imat") f.enc = Fmt::IMAT;
  else if (enc == "gptq") f.enc = Fmt::GPTQ;
  else fail("format " + spec + ": unknown encoder '" + enc + "' (rtn, mse, imat, gptq)");
  std::vector<std::string> p;
  for (size_t a = 0;;) {
    const size_t b = base.find(':', a);
    p.push_back(base.substr(a, b - a));
    if (b == std::string::npos) break;
    a = b + 1;
  }
  auto num = [&](size_t i) {
    if (i >= p.size()) fail("format " + spec + ": missing field");
    return std::stoi(p[i]);
  };
  if (p[0] == "bf16") {
    f.kind = Fmt::BF16;
    return f;
  }
  if (p[0] == "affine") {
    f.kind = Fmt::Affine;
    f.bits = num(1);
    f.group = num(2);
    if (p.size() > 3) f.sd = p[3];
    if (f.sd != "bf16" && f.sd != "f16" && f.sd != "f32") fail("format " + spec + ": scale dtype bf16, f16 or f32");
    if (f.bits < 2 || f.bits > 8) fail("format " + spec + ": 2..8 bits");
    return f;
  }
  if (p[0] == "eqv") {
    f.kind = Fmt::VQ;
    f.bits = num(1);
    f.group = 256;
    if (f.bits < 9 || f.bits > 16) fail("format " + spec + ": eqv:K with 9 <= K <= 16");
    if (f.enc != Fmt::GPTQ && f.enc != Fmt::IMAT) fail("format " + spec + ": eqv encoders are +gptq (LDLQ) or +imat");
    return f;
  }
  if (p[0] == "eql") {
    f.bits = num(1);
    if (num(2) != 16) fail("eql: lane sub-blocks are 16 wide");
    f.group = kSuper;
    if (p.size() > 3 && p[3] == "cb") {
      if (f.bits < 3 || f.bits > 6) fail("eql codebooks: EQ3 to EQ6 (eql:4:16:cb[:ROWS], eql:3:16:cb[:ROWS]; EQ5 / EQ6: emulation only)");
      f.kind = Fmt::EQC;
      f.cbrows = p.size() > 4 ? std::stoi(p[4]) : 0;
      return f;
    }
    if (p.size() > 3 && p[3] != "8") fail("eql: 8-bit sub-scales only");
    f.kind = Fmt::EQ;
    return f;
  }
  fail("unknown format " + spec + " (eql:B:16, eql:4:16:cb, eql:3:16:cb, eqv:K, affine:B:G[:SD], bf16)");
}

EncOpts parse_enc_opts(const std::string& spec) {
  EncOpts o;
  size_t a = 0;
  while (a <= spec.size()) {
    size_t b = spec.find(',', a);
    if (b == std::string::npos) b = spec.size();
    const std::string t = spec.substr(a, b - a);
    if (t.empty() || t == "default") {
    } else if (t == "psd") o.psd = true;
    else if (t == "nopsd") o.psd = false;
    else if (t == "refit") o.refit = true;
    else if (t == "sign") o.sign = true;
    else if (t.rfind("schur", 0) == 0) o.schur = t.size() > 5 ? std::stoi(t.substr(5)) : 6;
    else fail("--enc-opts: unknown option '" + t + "' (psd, nopsd, schur[R], refit, sign)");
    a = b + 1;
  }
  if (o.schur < 0 || o.schur > 64) fail("--enc-opts: schurR with 0 <= R <= 64");
  return o;
}
void set_enc_opts(const EncOpts& o) { g_opts = o; }
const EncOpts& enc_opts() { return g_opts; }
std::string enc_opts_str(const EncOpts& o) {
  std::string s = o.psd ? "psd" : "nopsd";
  if (o.schur) s += ",schur" + std::to_string(o.schur);
  if (o.refit) s += ",refit";
  if (o.sign) s += ",sign";
  return s;
}

double hessian_min_eig_if_indefinite(const float* upper, int C, double scale, double* lmax) {
  std::vector<double> A(static_cast<size_t>(C) * size_t(C));
  for (int i = 0; i < C; ++i)
    for (int j = i; j < C; ++j) A[size_t(i) * C + j] = A[size_t(j) * C + i] = double(upper[size_t(i) * C + j]) * scale;
  std::vector<double> A2 = A;
  __LAPACK_int n = C, info = 0;
  dpotrf_("L", &n, A2.data(), &n, &info);
  if (info == 0) return NAN;
  std::vector<double>().swap(A2);
  std::vector<double> w;
  if (!sym_eig(A, C, w)) return NAN;
  if (lmax) *lmax = w.back();
  return w.front();
}

std::vector<int64_t> np_choice_sorted_vq(uint32_t seed, int64_t n, int64_t k) { return np_choice(seed, n, k); }

std::vector<uint32_t> pack_codes(const uint8_t* q, int64_t R, int C, int bits) {
  const size_t wpr = size_t(C) * size_t(bits) / 32;
  if (size_t(C) * size_t(bits) % 32) fail("pack: row width not a whole number of words");
  std::vector<uint32_t> out(static_cast<size_t>(R) * wpr, 0);
  for_row_blocks(R, 256, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      uint32_t* w = out.data() + size_t(r) * wpr;
      const uint8_t* c = q + size_t(r) * C;
      for (int k = 0; k < C; ++k) {
        const size_t bit = size_t(k) * size_t(bits);
        const uint64_t v = uint64_t(c[k]) << (bit % 32);
        w[bit / 32] |= uint32_t(v);
        if ((bit % 32) + size_t(bits) > 32) w[bit / 32 + 1] |= uint32_t(v >> 32);
      }
    }
  });
  return out;
}

// ---------------------------------------------------------------- EQ / EQC

EqOut encode_eq(const float* W, int64_t R, int C, int nexp, const Fmt& f, const Stats* st) {
  if (C % 64) fail("EQ: input width " + std::to_string(C) + " is not a multiple of 64");
  const std::vector<int> perm = lane_perm(C);
  const int nsup = n_supers(C);
  EqOut out;
  out.q.resize(size_t(R) * C);
  out.sc.resize(size_t(R) * (C / 16));
  out.d.resize(size_t(R) * size_t(nsup));
  const int64_t O = nexp ? R / nexp : R;
  std::vector<float> dtmp;

  if (f.kind == Fmt::EQC) {
    // Codebook encode: importance (imat / gptq) with +1e-3, codebook groups.
    std::vector<float> imp;  // [nimp][C] permuted
    int nimp = 0;
    const bool weighted = st && (f.enc == Fmt::IMAT || f.enc == Fmt::GPTQ);
    if (weighted && st->diag && st->diag_rows == nexp && nexp > 0) {
      nimp = nexp;
      imp.resize(size_t(nexp) * C);
      for (int e = 0; e < nexp; ++e) {
        const float* d = st->diag + size_t(e) * C;
        double md = 0;
        for (int i = 0; i < C; ++i) md += d[i];
        const float m = float(md / C);
        for (int i = 0; i < C; ++i)
          imp[size_t(e) * C + i] = (m > 0 ? d[perm[size_t(i)]] / std::max(m, 1e-30f) : 1.0f) + 1e-3f;
      }
    } else if (const float* d = weighted ? stats_diag(st, C, dtmp) : nullptr) {
      nimp = 1;
      imp.resize(size_t(C));
      double md = 0;
      for (int i = 0; i < C; ++i) md += d[i];
      const float m = float(md / C);
      for (int i = 0; i < C; ++i) imp[size_t(i)] = d[perm[size_t(i)]] / std::max(m, 1e-30f) + 1e-3f;
    } else {
      nimp = 1;
      imp.assign(size_t(C), 1.0f);
    }
    const int64_t rows = f.cbrows ? f.cbrows : (nexp ? O : R);
    if (R % rows || rows % 8) fail("codebook rows " + std::to_string(rows) + " must divide " + std::to_string(R) + " and be a multiple of 8");
    const int ng = int(R / rows);
    std::vector<float> Xp(static_cast<size_t>(R) * C);
    for_row_blocks(R, 64, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r)
        for (int i = 0; i < C; ++i) Xp[size_t(r) * C + i] = W[size_t(r) * C + perm[size_t(i)]];
    });
    auto imp_row = [&](int64_t r) -> const float* { return imp.data() + (nimp == 1 ? 0 : size_t(r / O) * C); };
    const int L = 1 << f.bits;
    std::vector<float> uni(static_cast<size_t>(L));  // EQ5 / EQ6 codebooks start from the uniform grid code - 2^(B-1)
    for (int q = 0; q < L; ++q) uni[size_t(q)] = float(q - L / 2);
    const float* init = f.bits == 4 ? kCb4Init : f.bits == 3 ? kCb3Init : uni.data();
    std::vector<float> cb;
    for (int gi = 0; gi < ng; ++gi) cb.insert(cb.end(), init, init + L);
    std::vector<uint8_t> qp(static_cast<size_t>(R) * C);
    if (ng == 1) {  // fit on a sample of whole units (experts / 8-row slices), then encode every row
      const int64_t unit = nexp ? O : 8;
      const int64_t nu = R / unit;
      const int64_t take = std::max<int64_t>(1, std::min<int64_t>(nu, 16384 / unit));
      const std::vector<int64_t> sel = np_choice(0, nu, take);
      std::vector<float> Xs(static_cast<size_t>(take * unit) * C);
      std::vector<int64_t> src(static_cast<size_t>(take * unit));
      for (int64_t a = 0; a < take; ++a)
        for (int64_t u = 0; u < unit; ++u) src[size_t(a * unit + u)] = sel[size_t(a)] * unit + u;
      for (size_t i = 0; i < src.size(); ++i) std::memcpy(&Xs[i * C], &Xp[size_t(src[i]) * C], sizeof(float) * C);
      std::vector<uint8_t> qs(Xs.size());
      std::vector<uint16_t> ds(src.size() * size_t(nsup));
      std::vector<int8_t> ss(src.size() * (C / 16));
      eqc_fit(Xs.data(), int64_t(src.size()), C, [&](int64_t r) { return imp_row(src[size_t(r)]); },
              [](int64_t) { return 0; }, 1, 8, cb, qs.data(), ds.data(), ss.data(), L);
      eqc_fit(Xp.data(), R, C, imp_row, [](int64_t) { return 0; }, 1, 0, cb, qp.data(), out.d.data(), out.sc.data(), L);
    } else {
      eqc_fit(Xp.data(), R, C, imp_row, [&](int64_t r) { return int(r / rows); }, ng, 8, cb, qp.data(), out.d.data(),
              out.sc.data(), L);
    }
    if (f.enc == Fmt::GPTQ && nexp == 0 && ng == 1 && st && st->H && !std::getenv("EQC_NO_GPTQ")) {
      // GPTQ with the fitted codebook (dense matrices): at each super-block the 16 sub-block scales are
      // fitted on the error-updated weights (cb_scale, the imatrix weights above, the two-level form of
      // eqc_fit), each column takes its nearest codebook entry, and its error feeds the later columns.
      std::vector<float> Hp(static_cast<size_t>(C) * C);
      for (int i = 0; i < C; ++i)
        for (int j = 0; j < C; ++j) Hp[size_t(i) * C + j] = st->H[size_t(perm[size_t(i)]) * C + perm[size_t(j)]];
      std::vector<uint8_t> dead;
      std::vector<float> Hproj;
      const std::vector<float> U = hinv_upper(Hp.data(), C, dead, &Hproj);
      if (!Hproj.empty()) Hp.swap(Hproj);
      if (!g_opts.refit) std::vector<float>().swap(Hp);
      std::vector<float> Wr(static_cast<size_t>(R) * C);
      for_row_blocks(R, 64, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r)
          for (int i = 0; i < C; ++i) Wr[size_t(r) * C + i] = dead[size_t(i)] ? 0.0f : Xp[size_t(r) * C + i];
      });
      std::vector<float> mids(static_cast<size_t>(L - 1));
      for (int j = 0; j < L - 1; ++j) mids[size_t(j)] = (cb[size_t(j) + 1] + cb[size_t(j)]) / 2;
      const float* wimp = imp_row(0);
      if (g_opts.schur > 0) {
        auto search = [&](const float* x, int col) { return cb_scale(x, wimp + col, cb.data(), mids.data(), L); };
        gptq_e9(
            Wr.data(), R, C, U.data(), g_opts.schur, search, search,
            [&](float w, float sv) { return sv * cb[size_t(cb_nearest(w / sv, mids.data(), L))]; },
            [&](int64_t r, int col, float w, float sv) {
              const int q = cb_nearest(w / sv, mids.data(), L);
              qp[size_t(r) * C + col] = uint8_t(q);
              return sv * cb[size_t(q)];
            },
            out.sc.data(), out.d.data());
      } else {
        std::vector<float> s2(static_cast<size_t>(R) * 16);
        gptq(
            Wr.data(), R, C, U.data(), kSuper,
            [&](int64_t r, int i, const float* w) {
              const int nt = super_width(C, i / kSuper) / 16;
              float s[16];
              if (L > 16)
                for (int t = 0; t < nt; ++t) s[t] = cb_scale(w + 16 * t, wimp + i + 16 * t, cb.data(), mids.data(), L);
              else
                for (int t = 0; t < nt; t += 4) cb_scale4(w + 16 * t, wimp + i + 16 * t, cb.data(), mids.data(), L, s + t);
              float amax = 0;
              for (int t = 0; t < nt; ++t) amax = std::max(amax, std::fabs(s[t]));
              const float d = f16_round(amax / 127.0f);
              const float dv = d > 0 ? d : 1.0f;
              for (int t = 0; t < nt; ++t) {
                const float sc = std::clamp(rint_even(s[t] / dv), -127.0f, 127.0f);
                float v = f16_round(sc * d);
                if (v == 0) v = 1e-8f;
                s2[size_t(r) * 16 + t] = v;
                out.sc[size_t(r) * (C / 16) + i / 16 + t] = int8_t(sc);
              }
              out.d[size_t(r) * size_t(nsup) + i / kSuper] = f16_bits(d);
            },
            [&](int64_t r, int i, int k, float w) {
              const float sk = s2[size_t(r) * 16 + k / 16];
              const int q = cb_nearest(w / sk, mids.data(), L);
              qp[size_t(r) * C + i + k] = uint8_t(q);
              return sk * cb[size_t(q)];
            });
      }
      if (g_opts.refit) {
        std::vector<float>().swap(Wr);
        refit_super(
            R, C, Hp.data(), out.sc.data(), out.d.data(),
            [&](int64_t r, float* x) {
              for (int i = 0; i < C; ++i) x[i] = dead[size_t(i)] ? 0.0f : Xp[size_t(r) * C + i];
            },
            [&](int64_t r, float* v) {
              for (int i = 0; i < C; ++i) v[i] = cb[size_t(qp[size_t(r) * C + i])];
            });
      }
    }
    for_row_blocks(R, 256, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r)
        for (int i = 0; i < C; ++i) out.q[size_t(r) * C + perm[size_t(i)]] = qp[size_t(r) * C + i];
    });
    out.cb.resize(cb.size());
    for (size_t i = 0; i < cb.size(); ++i) out.cb[i] = f16_bits(cb[i]);
    return out;
  }

  const int nmax = 1 << (f.bits - 1);
  // GPTQ: dense matrices with a full Hessian; stacked experts are refused
  // (the reference's EQ path cannot store them either); without a Hessian the
  // imatrix search, without statistics the plain one.
  Fmt fe = f;
  if (fe.enc == Fmt::GPTQ && nexp) fail("EQ + gptq on stacked experts is not supported (use +imat)");
  if (fe.enc == Fmt::GPTQ && !(st && st->H)) fe.enc = Fmt::IMAT;
  if (fe.enc == Fmt::IMAT && !(st && (st->H || st->diag))) fe.enc = Fmt::MSE;

  if (fe.enc == Fmt::GPTQ) {
    std::vector<float> Hp(static_cast<size_t>(C) * C);
    for (int i = 0; i < C; ++i)
      for (int j = 0; j < C; ++j) Hp[size_t(i) * C + j] = st->H[size_t(perm[size_t(i)]) * C + perm[size_t(j)]];
    std::vector<uint8_t> dead;
    std::vector<float> Hproj;
    const std::vector<float> U = hinv_upper(Hp.data(), C, dead, &Hproj);
    std::vector<float> imp(static_cast<size_t>(C));
    double md = 0;
    for (int i = 0; i < C; ++i) md += Hp[size_t(i) * C + i];
    const float m = float(md / C);
    for (int i = 0; i < C; ++i) imp[size_t(i)] = Hp[size_t(i) * C + i] / std::max(m, 1e-30f);
    if (!Hproj.empty()) Hp.swap(Hproj);  // the refit's objective: the Hessian GPTQ used
    std::vector<float> Wr(static_cast<size_t>(R) * C);
    for_row_blocks(R, 64, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r)
        for (int i = 0; i < C; ++i)
          Wr[size_t(r) * C + i] = dead[size_t(i)] ? 0.0f : W[size_t(r) * C + perm[size_t(i)]];
    });
    const bool both = g_opts.sign;
    if (g_opts.schur > 0) {
      gptq_e9(
          Wr.data(), R, C, U.data(), g_opts.schur,
          [&](const float* x, int col) { return sym_search_opt(x, imp.data() + col, f.bits, false, both); },
          [&](const float* x, int col) { return sym_search_opt(x, imp.data() + col, f.bits, false, true); },
          [&](float w, float sv) { return eq_code(w, sv, f.bits) * sv; },
          [&](int64_t r, int col, float w, float sv) {
            const float q = eq_code(w, sv, f.bits);
            out.q[size_t(r) * C + perm[size_t(col)]] = uint8_t(int(q) + nmax);
            return q * sv;
          },
          out.sc.data(), out.d.data());
    } else {
      std::vector<float> s2(static_cast<size_t>(R) * 16);  // the current super-block's scales per row
      gptq(
          Wr.data(), R, C, U.data(), kSuper,
          [&](int64_t r, int i, const float* w) {
            const int nt = super_width(C, i / kSuper) / 16;
            float s[16];
            for (int t = 0; t < nt; ++t) s[t] = sym_search_opt(w + 16 * t, imp.data() + i + 16 * t, f.bits, false, both);
            two_level(s, out.d[size_t(r) * size_t(nsup) + i / kSuper], &out.sc[size_t(r) * (C / 16) + i / 16],
                      &s2[size_t(r) * 16], nt);
          },
          [&](int64_t r, int i, int k, float w) {
            const float s = s2[size_t(r) * 16 + k / 16];
            const float q = eq_code(w, s, f.bits);
            out.q[size_t(r) * C + perm[size_t(i + k)]] = uint8_t(int(q) + nmax);
            return q * s;
          });
    }
    if (g_opts.refit) {
      std::vector<float>().swap(Wr);
      refit_super(
          R, C, Hp.data(), out.sc.data(), out.d.data(),
          [&](int64_t r, float* x) {
            for (int i = 0; i < C; ++i) x[i] = dead[size_t(i)] ? 0.0f : W[size_t(r) * C + perm[size_t(i)]];
          },
          [&](int64_t r, float* v) {
            for (int i = 0; i < C; ++i) v[i] = float(int(out.q[size_t(r) * C + perm[size_t(i)]]) - nmax);
          });
    }
    return out;
  }

  // imat / mse / rtn: the search per super-block.
  std::vector<float> imp;  // [nimp][C] permuted, normalized per row as qdq_rows does
  int nimp = 0;
  if (fe.enc == Fmt::IMAT) {
    if (nexp && st->diag && st->diag_rows == nexp) {
      nimp = nexp;
      imp.resize(size_t(nexp) * C);
      for (int e = 0; e < nexp; ++e) {
        const float* d = st->diag + size_t(e) * C;
        double md = 0;
        for (int i = 0; i < C; ++i) md += d[i];
        const float m = float(md / C);
        float* o = imp.data() + size_t(e) * C;
        for (int i = 0; i < C; ++i) o[i] = (m > 0 ? d[perm[size_t(i)]] / std::max(m, 1e-30f) : 1.0f) + 1e-3f;
        double m2d = 0;
        for (int i = 0; i < C; ++i) m2d += o[i];
        const float m2 = float(m2d / C);
        for (int i = 0; i < C; ++i) o[i] = o[i] / std::max(m2, 1e-30f);
      }
    } else {
      const float* d = stats_diag(st, C, dtmp);
      if (!d) fail("imat: no statistics");
      nimp = 1;
      imp.resize(size_t(C));
      double md = 0;
      for (int i = 0; i < C; ++i) md += d[i];
      const float m = float(md / C);
      for (int i = 0; i < C; ++i) imp[size_t(i)] = d[perm[size_t(i)]] / std::max(m, 1e-30f);
    }
  }
  for_row_blocks(R, 32, [&](int64_t r0, int64_t r1) {
    std::vector<float> xp(static_cast<size_t>(C)), codes(kSuper);
    for (int64_t r = r0; r < r1; ++r) {
      for (int i = 0; i < C; ++i) xp[size_t(i)] = W[size_t(r) * C + perm[size_t(i)]];
      const float* w = nimp == 0 ? nullptr : imp.data() + (nimp == 1 ? 0 : size_t(r / O) * C);
      for (int sb = 0; sb < nsup; ++sb) {
        const int sw = super_width(C, sb);
        eq_superblock(xp.data() + sb * kSuper, w ? w + sb * kSuper : nullptr, fe, out.d[size_t(r) * size_t(nsup) + sb],
                      &out.sc[size_t(r) * (C / 16) + sb * 16], codes.data(), sw);
        for (int k = 0; k < sw; ++k)
          out.q[size_t(r) * C + perm[size_t(sb * kSuper + k)]] = uint8_t(int(codes[size_t(k)]) + nmax);
      }
    }
  });
  return out;
}

// ---------------------------------------------------------------- affine

AffOut encode_affine(const float* W, int64_t R, int C, int nexp, const Fmt& f, const Stats* st) {
  const int g = f.group;
  if (C % g) fail("affine: input width " + std::to_string(C) + " is not a multiple of " + std::to_string(g));
  AffOut out;
  out.q.resize(size_t(R) * C);
  out.s.resize(size_t(R) * (C / g));
  out.b.resize(size_t(R) * (C / g));
  const int64_t O = nexp ? R / nexp : R;
  if (f.enc == Fmt::RTN) {  // MLX's own quantize, on the weights cast to the scale dtype
    for_row_blocks(R, 64, [&](int64_t r0, int64_t r1) {
      std::vector<float> v(static_cast<size_t>(g));
      for (int64_t r = r0; r < r1; ++r)
        for (int gi = 0; gi < C / g; ++gi) {
          for (int i = 0; i < g; ++i) {
            const float x = W[size_t(r) * C + gi * g + i];
            v[size_t(i)] = f.sd == "f16" ? f16_round(x) : f.sd == "bf16" ? bf16_round_f(x) : x;
          }
          mlx_quantize_group(v.data(), g, f.bits, f.sd, &out.q[size_t(r) * C + gi * g], out.s[size_t(r) * (C / g) + gi],
                             out.b[size_t(r) * (C / g) + gi]);
        }
    });
    return out;
  }
  Fmt fe = f;
  if (fe.enc == Fmt::GPTQ && nexp) fail("affine + gptq on stacked experts is not supported (use +imat)");
  if (fe.enc == Fmt::GPTQ && !(st && st->H)) fe.enc = Fmt::IMAT;
  if (fe.enc == Fmt::IMAT && !(st && (st->H || st->diag))) fe.enc = Fmt::MSE;
  std::vector<float> dtmp;
  if (fe.enc == Fmt::GPTQ) {
    std::vector<uint8_t> dead;
    const std::vector<float> U = hinv_upper(st->H, C, dead);
    std::vector<float> imp(static_cast<size_t>(C));
    double md = 0;
    for (int i = 0; i < C; ++i) md += st->H[size_t(i) * C + i];
    const float m = float(md / C);
    for (int i = 0; i < C; ++i) imp[size_t(i)] = st->H[size_t(i) * C + i] / std::max(m, 1e-30f);
    std::vector<float> Wr(static_cast<size_t>(R) * C);
    for_row_blocks(R, 64, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r)
        for (int i = 0; i < C; ++i) Wr[size_t(r) * C + i] = dead[size_t(i)] ? 0.0f : W[size_t(r) * C + i];
    });
    gptq(
        Wr.data(), R, C, U.data(), g,
        [&](int64_t r, int i, const float* w) {
          affine_search(w, imp.data() + i, g, f.bits, 20, f.sd, out.s[size_t(r) * (C / g) + i / g],
                        out.b[size_t(r) * (C / g) + i / g]);
        },
        [&](int64_t r, int i, int k, float w) {
          const float s = out.s[size_t(r) * (C / g) + i / g], b = out.b[size_t(r) * (C / g) + i / g];
          const float q = affine_code(w, s, b, f.bits);
          out.q[size_t(r) * C + i + k] = uint8_t(q);
          return q * s + b;
        });
    return out;
  }
  // imat / mse (the affine search, 20 steps)
  std::vector<float> imp;
  int nimp = 0;
  if (fe.enc == Fmt::IMAT) {
    if (nexp && st->diag && st->diag_rows == nexp) {
      nimp = nexp;
      imp.resize(size_t(nexp) * C);
      for (int e = 0; e < nexp; ++e) {
        const float* d = st->diag + size_t(e) * C;
        double md = 0;
        for (int i = 0; i < C; ++i) md += d[i];
        const float m = float(md / C);
        float* o = imp.data() + size_t(e) * C;
        for (int i = 0; i < C; ++i) o[i] = (m > 0 ? d[i] / std::max(m, 1e-30f) : 1.0f) + 1e-3f;
        double m2d = 0;
        for (int i = 0; i < C; ++i) m2d += o[i];
        const float m2 = float(m2d / C);
        for (int i = 0; i < C; ++i) o[i] = o[i] / std::max(m2, 1e-30f);
      }
    } else {
      const float* d = stats_diag(st, C, dtmp);
      nimp = 1;
      imp.resize(size_t(C));
      double md = 0;
      for (int i = 0; i < C; ++i) md += d[i];
      const float m = float(md / C);
      for (int i = 0; i < C; ++i) imp[size_t(i)] = d[i] / std::max(m, 1e-30f);
    }
  }
  for_row_blocks(R, 64, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float* x = W + size_t(r) * C;
      const float* w = nimp == 0 ? nullptr : imp.data() + (nimp == 1 ? 0 : size_t(r / O) * C);
      for (int gi = 0; gi < C / g; ++gi) {
        float& s = out.s[size_t(r) * (C / g) + gi];
        float& b = out.b[size_t(r) * (C / g) + gi];
        affine_search(x + gi * g, w ? w + gi * g : nullptr, g, f.bits, 20, f.sd, s, b);
        for (int i = 0; i < g; ++i) out.q[size_t(r) * C + gi * g + i] = uint8_t(affine_code(x[gi * g + i], s, b, f.bits));
      }
    }
  });
  return out;
}

}  // namespace espetero::quant
