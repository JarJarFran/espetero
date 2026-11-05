// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// EQ 1.x encoder options on a synthetic matrix (CPU only, seconds): Hessian-weighted error
// e = tr(dW H dW^T) / tr(W H W^T) per format and option set, the PSD projection of an indefinite Hessian, and
// (with --dump FILE) the default encodings' bytes for an identity check against another build of encode.cpp.
//   clang++ -std=c++20 -O2 -I src tests/eq1x_encode_test.cpp src/quant/encode.cpp -framework Accelerate \
//     -framework CoreFoundation -framework IOKit -o build/eq1x_encode_test
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <exception>
#include <vector>

#include "quant/encode.h"

using namespace espetero::quant;

namespace {

const int C = 768, R = 384, N = 6144;

std::vector<float> dequant(const EqOut& o, const Fmt& f) {
  const std::vector<int> perm = lane_perm(C);
  std::vector<int> pos(C);
  for (int i = 0; i < C; ++i) pos[perm[size_t(i)]] = i;
  const int nsup = (C + 255) / 256, nmax = 1 << (f.bits - 1);
  std::vector<float> Y(size_t(R) * C);
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c) {
      const int i = pos[size_t(c)];
      __fp16 dh;
      std::memcpy(&dh, &o.d[size_t(r) * nsup + i / 256], 2);
      const float s = float(__fp16(float(dh) * float(o.sc[size_t(r) * (C / 16) + i / 16])));
      const int q = o.q[size_t(r) * C + c];
      float v;
      if (f.kind == Fmt::EQC) {
        __fp16 cb;
        std::memcpy(&cb, &o.cb[size_t(q)], 2);
        v = float(cb);
      } else {
        v = float(q - nmax);
      }
      Y[size_t(r) * C + c] = s * v;
    }
  return Y;
}

double werr(const std::vector<float>& W, const std::vector<float>& Y, const std::vector<float>& H) {
  std::vector<float> D(W.size()), DH(W.size()), WH(W.size());
  for (size_t k = 0; k < W.size(); ++k) D[k] = Y[k] - W[k];
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, R, C, C, 1.0f, D.data(), C, H.data(), C, 0.0f, DH.data(), C);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, R, C, C, 1.0f, W.data(), C, H.data(), C, 0.0f, WH.data(), C);
  double a = 0, b = 0;
  for (size_t k = 0; k < W.size(); ++k) {
    a += double(DH[k]) * D[k];
    b += double(WH[k]) * W[k];
  }
  return a / b;
}

}  // namespace

int run(int argc, char** argv) {
  std::mt19937 g(1);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::student_t_distribution<float> td(4.0f);
  // Inputs: log-uniform channel scales, one massive channel, mixed by a random near-identity matrix.
  std::vector<float> sc(C), A(size_t(C) * C), Z(size_t(N) * C), X(size_t(N) * C);
  for (int j = 0; j < C; ++j) sc[size_t(j)] = std::exp(std::uniform_real_distribution<float>(-4.0f, 0.0f)(g));
  sc[37] = 30.0f;
  for (int i = 0; i < C; ++i)
    for (int j = 0; j < C; ++j) A[size_t(i) * C + j] = (i == j ? 1.0f : 0.0f) + 0.3f * nd(g) / std::sqrt(float(C));
  for (int n = 0; n < N; ++n)
    for (int j = 0; j < C; ++j) Z[size_t(n) * C + j] = nd(g) * sc[size_t(j)];
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, C, C, 1.0f, Z.data(), C, A.data(), C, 0.0f, X.data(), C);
  std::vector<float> H(size_t(C) * C);
  cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, C, N, 1.0f / N, X.data(), C, 0.0f, H.data(), C);
  for (int i = 0; i < C; ++i)
    for (int j = 0; j < i; ++j) H[size_t(i) * C + j] = H[size_t(j) * C + i];
  std::vector<float> W(size_t(R) * C);
  for (float& w : W) w = 0.02f * td(g);
  Stats st;
  st.H = H.data();
  st.C = C;
  std::vector<float> diag(C);
  for (int j = 0; j < C; ++j) diag[size_t(j)] = H[size_t(j) * C + j];
  Stats sd;
  sd.diag = diag.data();
  sd.diag_rows = 1;
  sd.C = C;

  FILE* dump = nullptr;
  for (int i = 1; i + 1 < argc; ++i)
    if (std::string(argv[i]) == "--dump") dump = std::fopen(argv[i + 1], "wb");

  const char* fmts[] = {"eql:4:16:cb+gptq", "eql:5:16+gptq", "eql:6:16+gptq", "eql:8:16+gptq", "eql:5:16+imat", "eql:3:16:cb+gptq",
#ifndef OLD
                        "eql:5:16:cb+gptq"
#endif
  };
#ifndef OLD
  const char* opts[] = {"", "schur6", "refit", "schur6,refit", "sign", "schur6,refit,sign"};
#else
  const char* opts[] = {""};
#endif
  for (const char* fs : fmts) {
    const Fmt f = parse_fmt(fs);
    const Stats* s = f.enc == Fmt::IMAT ? &sd : &st;
    double e0 = 0;
    std::printf("%-20s", fs);
    for (const char* os : opts) {
#ifndef OLD
      set_enc_opts(parse_enc_opts(os));
#endif
      const EqOut o = encode_eq(W.data(), R, C, 0, f, s);
      const double e = werr(W, dequant(o, f), H);
      if (!*os) {
        e0 = e;
        std::printf(" base %.4e", e);
        if (dump) {
          std::fwrite(o.q.data(), 1, o.q.size(), dump);
          std::fwrite(o.sc.data(), 1, o.sc.size(), dump);
          std::fwrite(o.d.data(), 2, o.d.size(), dump);
          std::fwrite(o.cb.data(), 2, o.cb.size(), dump);
        }
      } else {
        std::printf(" | %s %.4f", os, e / e0);
      }
    }
    std::printf("\n");
  }
#ifndef OLD
  // An indefinite encode Hessian (H minus a rank-1 term that makes v^T H v negative): error judged under the true H.
  std::vector<float> v(C), Hb = H;
  double nv = 0;
  for (float& x : v) {
    x = nd(g);
    nv += double(x) * x;
  }
  for (float& x : v) x = float(x / std::sqrt(nv));
  double vhv = 0;
  for (int i = 0; i < C; ++i)
    for (int j = 0; j < C; ++j) vhv += double(v[size_t(i)]) * H[size_t(i) * C + j] * v[size_t(j)];
  for (int i = 0; i < C; ++i)
    for (int j = 0; j < C; ++j) Hb[size_t(i) * C + j] -= float(2.0 * vhv) * v[size_t(i)] * v[size_t(j)];
  Stats sb;
  sb.H = Hb.data();
  sb.C = C;
  double lmax = 0;
  const double lmin = hessian_min_eig_if_indefinite(Hb.data(), C, 1.0, &lmax);
  std::printf("indefinite H: lambda_min %.4g lambda_max %.4g (check on the PSD H: %g)\n", lmin, lmax,
              hessian_min_eig_if_indefinite(H.data(), C, 1.0, &lmax));
  for (const char* fs : {"eql:4:16:cb+gptq", "eql:6:16+gptq"}) {
    const Fmt f = parse_fmt(fs);
    set_enc_opts(parse_enc_opts("nopsd"));
    const double en = werr(W, dequant(encode_eq(W.data(), R, C, 0, f, &sb), f), H);
    set_enc_opts(parse_enc_opts(""));
    const double ep = werr(W, dequant(encode_eq(W.data(), R, C, 0, f, &sb), f), H);
    const double et = werr(W, dequant(encode_eq(W.data(), R, C, 0, f, &st), f), H);
    std::printf("%-20s indefinite encode H: nopsd %.4e psd %.4e (%.3f x), true-H encode %.4e\n", fs, en, ep, ep / en, et);
  }
#endif
  if (dump) std::fclose(dump);
  return 0;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::printf("\nerror: %s\n", e.what());
    return 1;
  }
}
