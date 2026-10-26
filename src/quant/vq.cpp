// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// EQV encoder (vq.h): the C++ port of fnx-vq/vqlab.py's eqv_stack.
#define ACCELERATE_NEW_LAPACK
#include "vq.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include "../model/impl/common.h"

namespace espetero::quant {

using native::fail;
using native::parallel_for;

namespace {

constexpr int kG = 256;  // inputs per scale

// 256-wide scale groups of a row of C inputs, the last one partial.
std::vector<std::pair<int, int>> spans(int C) {
  std::vector<std::pair<int, int>> s;
  for (int g0 = 0; g0 < C; g0 += kG) s.push_back({g0, std::min(g0 + kG, C)});
  return s;
}

// numpy RandomState(seed).choice(n, k, replace=False), sorted (encode.cpp's np_choice).
std::vector<int64_t> choice_sorted(uint32_t seed, int64_t n, int64_t k);

// Nearest codewords for N vectors (X, importances T: [N][8]), codebook cb [K][8].
void assign(const float* X, const float* T, int64_t N, const float* cb, int K, int32_t* out) {
  if (N <= 0) return;
  if (vq_assign_gpu(X, T, N, cb, K, out)) return;
  parallel_for(size_t((N + 1023) / 1024), [&](size_t b) {
    const int64_t i0 = int64_t(b) * 1024, i1 = std::min<int64_t>(N, i0 + 1024);
    for (int64_t i = i0; i < i1; ++i) {
      const float* x = X + i * 8;
      const float* w = T + i * 8;
      float m[8];
      for (int j = 0; j < 8; ++j) m[j] = -2.0f * w[j] * x[j];
      float best = std::numeric_limits<float>::infinity();
      int bi = 0;
      for (int k = 0; k < K; ++k) {
        const float* c = cb + size_t(k) * 8;
        float d = 0.0f;
        for (int j = 0; j < 8; ++j) d += (c[j] * w[j] + m[j]) * c[j];
        if (d < best) best = d, bi = k;
      }
      out[i] = bi;
    }
  });
}

// The expert's normalized imatrix (vqlab.expert_imp): experts never routed take the routed experts'
// mean; each row divided by its mean, + 1e-3. d: [E][C] (or null: ones).
std::vector<float> expert_imp(const float* d, int E, int C) {
  std::vector<float> imp(size_t(E) * C, 1.0f);
  if (!d) return imp;
  std::vector<double> lay(size_t(C), 0.0);
  int routed = 0;
  std::vector<uint8_t> live(size_t(E), 0);
  for (int e = 0; e < E; ++e) {
    double m = 0;
    for (int j = 0; j < C; ++j) m += d[size_t(e) * C + j];
    if (m / C > 0) {
      live[size_t(e)] = 1;
      ++routed;
    }
    for (int j = 0; j < C; ++j) lay[size_t(j)] += d[size_t(e) * C + j];
  }
  for (int j = 0; j < C; ++j) lay[size_t(j)] /= std::max(routed, 1);
  for (int e = 0; e < E; ++e) {
    float* r = imp.data() + size_t(e) * C;
    for (int j = 0; j < C; ++j) r[j] = live[size_t(e)] ? d[size_t(e) * C + j] : float(lay[size_t(j)]);
    double m = 0;
    for (int j = 0; j < C; ++j) m += r[j];
    m /= C;
    for (int j = 0; j < C; ++j) r[j] = float(r[j] / std::max(m, 1e-30)) + 1e-3f;
  }
  return imp;
}

// Weighted rms of each row of a group: x [R][ldx] (G columns), w per row (row r's importances at w + r * ldw).
float wrms(const float* x, const float* w, int G) {
  double a = 0, b = 0;
  for (int j = 0; j < G; ++j) a += double(w[j]) * x[j] * x[j], b += w[j];
  return float(std::sqrt(a / std::max(b, 1e-30))) + 1e-12f;
}

// Weighted k-means of [N][8] vectors with importances, K centers, `iters` rounds (vqlab.kmeans).
std::vector<float> kmeans(const std::vector<float>& X, const std::vector<float>& T, int K, int iters) {
  const int64_t N = int64_t(X.size() / 8);
  std::vector<float> cb(size_t(K) * 8);
  {
    const std::vector<int64_t> pick = choice_sorted(0, N, std::min<int64_t>(N, K));
    for (int k = 0; k < K; ++k) std::memcpy(&cb[size_t(k) * 8], &X[size_t(pick[size_t(k) % pick.size()]) * 8], 32);
  }
  std::vector<int32_t> a(static_cast<size_t>(N));
  uint64_t lcg = 0x9e3779b97f4a7c15ull;
  for (int it = 0; it < iters; ++it) {
    assign(X.data(), T.data(), N, cb.data(), K, a.data());
    std::vector<double> num(size_t(K) * 8, 0.0), den(size_t(K) * 8, 0.0);
    for (int64_t i = 0; i < N; ++i) {
      const size_t k = size_t(a[size_t(i)]);
      for (int j = 0; j < 8; ++j) {
        num[k * 8 + j] += double(T[size_t(i) * 8 + j]) * X[size_t(i) * 8 + j];
        den[k * 8 + j] += T[size_t(i) * 8 + j];
      }
    }
    for (int k = 0; k < K; ++k) {
      double s = 0;
      for (int j = 0; j < 8; ++j) s += den[size_t(k) * 8 + j];
      if (s == 0) {  // empty: a random vector
        lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
        std::memcpy(&cb[size_t(k) * 8], &X[size_t((lcg >> 17) % uint64_t(N)) * 8], 32);
        continue;
      }
      for (int j = 0; j < 8; ++j)
        cb[size_t(k) * 8 + j] = float(num[size_t(k) * 8 + j] / std::max(den[size_t(k) * 8 + j], 1e-20));
    }
  }
  return cb;
}

// Upper factor U of (H + damping)^-1 (H^-1 = U^T U), quant_lab's damping (encode.cpp hinv_upper, double).
std::vector<float> hinv_upper_d(const double* H, int C) {
  std::vector<double> A(size_t(C) * C);
  std::memcpy(A.data(), H, A.size() * 8);
  double sum = 0;
  for (int i = 0; i < C; ++i) sum += A[size_t(i) * C + i] > 0 ? A[size_t(i) * C + i] : 0.0;
  for (int i = 0; i < C; ++i) {
    if (A[size_t(i) * C + i] <= 0) A[size_t(i) * C + i] += 1.0;
    A[size_t(i) * C + i] += 0.01 * sum / C;
  }
  __LAPACK_int n = C, info = 0;
  dpotrf_("L", &n, A.data(), &n, &info);
  if (info == 0) dpotri_("L", &n, A.data(), &n, &info);
  if (info == 0) {
    for (int j = 0; j < C; ++j)
      for (int i = j + 1; i < C; ++i) A[size_t(i) * C + j] = A[size_t(j) * C + i] = A[size_t(i) + size_t(j) * C];
    dpotrf_("L", &n, A.data(), &n, &info);
  }
  if (info != 0) fail("EQV: the damped Hessian is not positive definite (potrf " + std::to_string(info) + ")");
  std::vector<float> U(size_t(C) * C, 0.0f);
  for (int j = 0; j < C; ++j)  // column-major lower L read row-major = L^T = U
    for (int i = j; i < C; ++i) U[size_t(j) * C + i] = float(A[size_t(i) + size_t(j) * C]);
  return U;
}

// Inverse of the 8x8 diagonal block of U at c0 (upper triangular), row-major.
void inv_block(const float* U, int C, int c0, float* out) {
  double a[64], inv[64];
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) a[i * 8 + j] = j >= i ? U[size_t(c0 + i) * C + c0 + j] : 0.0;
  for (int j = 0; j < 8; ++j)  // back substitution, column j of the inverse
    for (int i = 7; i >= 0; --i) {
      double s = i == j ? 1.0 : 0.0;
      for (int k = i + 1; k < 8; ++k) s -= a[i * 8 + k] * inv[k * 8 + j];
      inv[i * 8 + j] = s / a[i * 8 + i];
    }
  for (int i = 0; i < 64; ++i) out[i] = float(inv[i]);
}

// Rows [r0, r0 + n) of the stack being encoded: their weights (updated in place), importances and the
// per-row column scaling cs of the shared factor (null: none) or each row's own factor.
struct Block {
  float* W;              // [n][C]
  int64_t n;
  int C;
  const float* imp;      // row r's importances at imp + (r / O) * C
  int O;
  const float* cs;       // [E][C] (row r: cs + (r / O) * C) or null
  const float* U;        // shared factor [C][C], or null with Ue
  const float* Ue;       // per-expert factors [E][C][C] (row r: Ue + (r / O) C^2), or null
};

// _eqv_group_scale: scales of every row of group [g0, g1).
void group_scales(const Block& b, int g0, int g1, const float* full, int K, float* sq) {
  const int G = g1 - g0, nvg = G / 8;
  std::vector<float> s(size_t(b.n));
  parallel_for(size_t(b.n), [&](size_t r) {
    s[r] = wrms(b.W + r * b.C + g0, b.imp + (r / b.O) * b.C + g0, G);
  });
  std::vector<float> X(size_t(b.n) * G), T(size_t(b.n) * G);
  std::vector<int32_t> a(size_t(b.n) * nvg);
  for (int it = 0; it < 2; ++it) {
    parallel_for(size_t(b.n), [&](size_t r) {
      const float* w = b.W + r * b.C + g0;
      const float* im = b.imp + (r / b.O) * b.C + g0;
      for (int j = 0; j < G; ++j) X[r * G + j] = w[j] / s[r], T[r * G + j] = im[j] * s[r] * s[r];
    });
    assign(X.data(), T.data(), b.n * nvg, full, K, a.data());
    parallel_for(size_t(b.n), [&](size_t r) {
      const float* w = b.W + r * b.C + g0;
      const float* im = b.imp + (r / b.O) * b.C + g0;
      double num = 0, den = 0;
      for (int v = 0; v < nvg; ++v) {
        const float* c = full + size_t(a[r * nvg + v]) * 8;
        for (int j = 0; j < 8; ++j) num += double(im[v * 8 + j]) * w[v * 8 + j] * c[j], den += double(im[v * 8 + j]) * c[j] * c[j];
      }
      const float s2 = float(num / std::max(den, 1e-30));
      if (s2 > 1e-12f) s[r] = s2;
    });
  }
  for (int64_t r = 0; r < b.n; ++r) {
    const float h = float(__fp16(s[size_t(r)]));
    sq[r] = h > 0 ? h : 6e-5f;
  }
}

// Encode the rows of `b` (vqlab.eqv_encode / eqv_encode_batched): codes [n][C/8], scales [n][ns].
void encode_rows(const Block& b, const float* full, int K, bool feedback, int32_t* codes, float* scales) {
  const int C = b.C, nv = C / 8;
  const auto sp = spans(C);
  const int ns = int(sp.size());
  std::vector<float> sq(size_t(b.n)), X, T, E, tmp;
  std::vector<int32_t> a(size_t(b.n));
  for (int si = 0; si < ns; ++si) {
    const int g0 = sp[size_t(si)].first, g1 = sp[size_t(si)].second, G = g1 - g0;
    group_scales(b, g0, g1, full, K, sq.data());
    for (int64_t r = 0; r < b.n; ++r) scales[size_t(r) * ns + si] = sq[size_t(r)];
    if (!feedback) {
      X.resize(size_t(b.n) * G), T.resize(size_t(b.n) * G);
      std::vector<int32_t> ag(size_t(b.n) * (G / 8));
      parallel_for(size_t(b.n), [&](size_t r) {
        const float* w = b.W + r * C + g0;
        const float* im = b.imp + (r / b.O) * C + g0;
        for (int j = 0; j < G; ++j) X[r * G + j] = w[j] / sq[r], T[r * G + j] = im[j] * sq[r] * sq[r];
      });
      assign(X.data(), T.data(), b.n * (G / 8), full, K, ag.data());
      for (int64_t r = 0; r < b.n; ++r)
        std::memcpy(codes + size_t(r) * nv + g0 / 8, &ag[size_t(r) * (G / 8)], size_t(G / 8) * 4);
      continue;
    }
    E.assign(size_t(b.n) * G, 0.0f);
    X.resize(size_t(b.n) * 8), T.resize(size_t(b.n) * 8);
    for (int j = 0; j < G; j += 8) {
      const int c0 = g0 + j;
      parallel_for(size_t(b.n), [&](size_t r) {
        const float* w = b.W + r * C + c0;
        const float* im = b.imp + (r / b.O) * C + c0;
        for (int t = 0; t < 8; ++t) X[r * 8 + t] = w[t] / sq[r], T[r * 8 + t] = im[t] * sq[r] * sq[r];
      });
      assign(X.data(), T.data(), b.n, full, K, a.data());
      // shared factor: one 8x8 inverse; per-expert factors: one per expert (rows of an expert are contiguous)
      float inv_s[64];
      if (b.U) inv_block(b.U, C, c0, inv_s);
      const int64_t nexp = b.Ue ? (b.n + b.O - 1) / b.O : 0;
      std::vector<float> inv_e(size_t(nexp) * 64);
      if (b.Ue)
        parallel_for(size_t(nexp), [&](size_t e) { inv_block(b.Ue + e * size_t(C) * C, C, c0, &inv_e[e * 64]); });
      parallel_for(size_t(b.n), [&](size_t r) {
        float* w = b.W + r * C;
        const float* c = full + size_t(a[r]) * 8;
        codes[r * nv + c0 / 8] = a[r];
        const float* cs = b.cs ? b.cs + (r / b.O) * C : nullptr;
        const float* U = b.U ? b.U : b.Ue + (r / b.O) * size_t(C) * C;
        const float* inv = b.U ? inv_s : &inv_e[(r / b.O) * 64];
        float dq[8], e[8];
        for (int t = 0; t < 8; ++t) dq[t] = (w[c0 + t] - c[t] * sq[r]) * (cs ? cs[c0 + t] : 1.0f);
        for (int t = 0; t < 8; ++t) {
          float s = 0.0f;
          for (int k = 0; k < 8; ++k) s += dq[k] * inv[k * 8 + t];
          e[t] = s;
          E[r * G + j + t] = s;
        }
        // the rest of the group: w[c] -= (e . U[c0 .. c0 + 8][c]) / cs[c]
        for (int col = c0 + 8; col < g1; ++col) {
          float s = 0.0f;
          for (int t = 0; t < 8; ++t) s += e[t] * U[size_t(c0 + t) * C + col];
          w[col] -= cs ? s / cs[col] : s;
        }
      });
    }
    if (g1 < C) {  // the group's errors into the later columns: W[:, g1:] -= (E U[g0:g1, g1:]) / cs
      const int rest = C - g1;
      if (b.U) {
        tmp.assign(size_t(b.n) * rest, 0.0f);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, int(b.n), rest, G, 1.0f, E.data(), G,
                    b.U + size_t(g0) * C + g1, C, 0.0f, tmp.data(), rest);
        parallel_for(size_t(b.n), [&](size_t r) {
          float* w = b.W + r * C + g1;
          const float* cs = b.cs ? b.cs + (r / b.O) * C + g1 : nullptr;
          for (int c = 0; c < rest; ++c) w[c] -= cs ? tmp[r * rest + c] / cs[c] : tmp[r * rest + c];
        });
      } else {
        const int64_t nexp = (b.n + b.O - 1) / b.O;
        parallel_for(size_t(nexp), [&](size_t e) {
          const int64_t r0 = int64_t(e) * b.O, rn = std::min<int64_t>(b.O, b.n - r0);
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, int(rn), rest, G, -1.0f, E.data() + size_t(r0) * G, G,
                      b.Ue + e * size_t(C) * C + size_t(g0) * C + g1, C, 1.0f, b.W + size_t(r0) * C + g1, C);
        });
      }
    }
  }
}

}  // namespace

std::vector<int64_t> np_choice_sorted_vq(uint32_t seed, int64_t n, int64_t k);  // encode.cpp

namespace {
std::vector<int64_t> choice_sorted(uint32_t seed, int64_t n, int64_t k) { return np_choice_sorted_vq(seed, n, k); }
}  // namespace

VqOut encode_vq(const float* W, int64_t R, int C, int nexp, int kbits, bool ldlq, const Stats* st) {
  if (C % 8) fail("EQV: inputs per row must be a multiple of 8");
  if (kbits < 9 || kbits > 16) fail("EQV: 9..16 code bits");
  const int E = std::max(nexp, 1);
  const int O = int(R / E);
  const int K = 1 << kbits;
  const auto sp = spans(C);
  const int ns = int(sp.size()), nv = C / 8;
  // importances: the per-expert imatrix (or the matrix's diagonal for every expert)
  std::vector<float> dtmp;
  const float* d = nullptr;
  if (st && st->diag && st->diag_rows == E) d = st->diag;
  else if (st && (st->diag || st->H)) {
    dtmp.resize(size_t(E) * C);
    for (int e = 0; e < E; ++e)
      for (int j = 0; j < C; ++j) dtmp[size_t(e) * C + j] = st->diag ? st->diag[j] : st->H[size_t(j) * C + j];
    d = dtmp.data();
  }
  const std::vector<float> imp = expert_imp(d, E, C);
  // codebook from a sample of rows
  const int64_t nsamp = std::min<int64_t>(R, (int64_t(1) << 20) * 8 / C + 1);  // vqlab: sample * 8 // C + 1
  const std::vector<int64_t> rows = choice_sorted(0, R, nsamp);
  std::vector<float> Xs(size_t(nsamp) * C), Ts(size_t(nsamp) * C);
  {
    // vectors in span order: all sampled rows' vectors of span 0, then span 1, ... (vqlab concatenates per span)
    size_t off = 0;
    for (const auto& [g0, g1] : sp) {
      const int G = g1 - g0;
      parallel_for(size_t(nsamp), [&](size_t i) {
        const int64_t r = rows[i];
        const float* w = W + size_t(r) * C + g0;
        const float* im = imp.data() + size_t(r / O) * C + g0;
        const float s = wrms(w, im, G);
        for (int j = 0; j < G; ++j) Xs[off + i * G + j] = w[j] / s, Ts[off + i * G + j] = im[j] * s * s;
      });
      off += size_t(nsamp) * G;
    }
  }
  std::vector<float> full = kmeans(Xs, Ts, K, 16);
  Xs.clear(), Xs.shrink_to_fit(), Ts.clear(), Ts.shrink_to_fit();
  VqOut o;
  o.kbits = kbits, o.nv = nv, o.ns = ns, o.hw = (nv * (kbits - 8) + 31) / 32;
  o.cb.resize(full.size());
  for (size_t i = 0; i < full.size(); ++i) {
    full[i] = float(__fp16(full[i]));
    o.cb[i] = f16_bits(full[i]);
  }
  std::vector<int32_t> codes(size_t(R) * nv);
  std::vector<float> scales(size_t(R) * ns);
  const bool shared = ldlq && st && st->Hshared && !(st->Hexp && st->nexp == E);
  const bool own = ldlq && st && st->Hexp && st->nexp == E && st->C == C;
  std::vector<float> U, cs, Ue;
  if (shared) {
    std::vector<double> Hd(size_t(C) * C);
    for (size_t i = 0; i < Hd.size(); ++i) Hd[i] = st->Hshared[i];
    U = hinv_upper_d(Hd.data(), C);
    cs.resize(size_t(E) * C);
    for (int e = 0; e < E; ++e) {
      double sum = 0;
      float* s = cs.data() + size_t(e) * C;
      for (int j = 0; j < C; ++j) {
        const double hd = std::max(double(st->Hshared[size_t(j) * C + j]), 1e-30);
        s[j] = float(std::sqrt(std::max(d ? double(d[size_t(e) * C + j]) : 1.0, 0.0) / hd));
        sum += s[j];
      }
      if (sum <= 0) std::fill(s, s + C, 1.0f), sum = C;
      const float lo = float(1e-3 * sum / C);
      for (int j = 0; j < C; ++j) s[j] = std::max(s[j], lo);
    }
  }
  const int chunk = 128;  // experts per pass (vqlab: 128)
  std::vector<float> Wc;
  for (int e0 = 0; e0 < E; e0 += chunk) {
    const int ec = std::min(chunk, E - e0);
    const int64_t n = int64_t(ec) * O;
    Wc.assign(W + size_t(e0) * O * C, W + size_t(e0 + ec) * O * C);
    if (own) {
      Ue.assign(size_t(ec) * C * C, 0.0f);
      parallel_for(size_t(ec), [&](size_t i) {
        const int e = e0 + int(i);
        std::vector<double> Hd(size_t(C) * C);
        bool live = false;
        for (size_t k = 0; k < Hd.size(); ++k) Hd[k] = st->Hexp[size_t(e) * C * C + k];
        for (int j = 0; j < C; ++j) live |= Hd[size_t(j) * C + j] > 0;
        if (!live) {
          std::fill(Hd.begin(), Hd.end(), 0.0);
          for (int j = 0; j < C; ++j) Hd[size_t(j) * C + j] = imp[size_t(e) * C + j];
        }
        const std::vector<float> u = hinv_upper_d(Hd.data(), C);
        std::memcpy(&Ue[i * C * C], u.data(), u.size() * 4);
      });
    }
    const Block b{Wc.data(), n, C, imp.data() + size_t(e0) * C, O, shared ? cs.data() + size_t(e0) * C : nullptr,
                  shared ? U.data() : nullptr, own ? Ue.data() : nullptr};
    encode_rows(b, full.data(), K, shared || own, codes.data() + size_t(e0) * O * nv, scales.data() + size_t(e0) * O * ns);
  }
  // planes
  o.lo.resize(size_t(R) * nv);
  o.hi.assign(size_t(R) * o.hw, 0u);
  o.sc.resize(size_t(R) * ns);
  const int kb = kbits - 8;
  parallel_for(size_t(R), [&](size_t r) {
    for (int v = 0; v < nv; ++v) {
      const uint32_t c = uint32_t(codes[r * nv + v]);
      o.lo[r * nv + v] = uint8_t(c & 255);
      const uint32_t h = c >> 8;
      const uint32_t bit = uint32_t(v) * uint32_t(kb);
      uint32_t* hr = o.hi.data() + r * o.hw;
      hr[bit / 32] |= h << (bit % 32);
      if (bit % 32 + uint32_t(kb) > 32) hr[bit / 32 + 1] |= h >> (32 - bit % 32);
    }
    for (int s = 0; s < ns; ++s) o.sc[r * ns + s] = f16_bits(scales[r * ns + s]);
  });
  return o;
}

}  // namespace espetero::quant
