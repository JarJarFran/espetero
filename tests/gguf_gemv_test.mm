// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The native GGUF-block GEMV kernels (86_glm5.metal gguf_gemv_*: Q8_0, Q2_K-Q6_K,
// IQ4_XS, IQ2_S, IQ3_S) on the GPU against gguf-py's dequantization of the same
// blocks (tests/ref/gguf_iq_ref.py): y = W x for 1, 3, 8 and 13 rows of x, plain
// and with indexed rows (xrow / yrow / ysc, accumulating), compared with the
// reference weights times x in double. Allowed: fp32 summation order.
//
//   python3 tests/ref/gguf_iq_ref.py /tmp/gg.bin --types Q8_0,Q2_K,Q3_K,Q4_K,Q5_K,Q6_K,IQ4_XS,IQ2_S,IQ3_S [--real ...]
//   build/gguf_gemv_test /tmp/gg.bin        (a GPU job: through gpuq)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../src/model/impl/runtime.h"
#include "kernels_src.h"

using namespace espetero::native;

namespace {
const char* kernel_for(int type) {
  switch (type) {
    case 8: return "gguf_gemv_q8_0";
    case 10: return "gguf_gemv_q2_k";
    case 11: return "gguf_gemv_q3_k";
    case 12: return "gguf_gemv_q4_k";
    case 13: return "gguf_gemv_q5_k";
    case 14: return "gguf_gemv_q6_k";
    case 21: return "gguf_gemv_iq3_s";
    case 22: return "gguf_gemv_iq2_s";
    case 23: return "gguf_gemv_iq4_xs";
  }
  return nullptr;
}
struct GemvP {
  int N, K, M, xs, ys, grp, gstr, accum, idx, nrow_tg;
};
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s REF.bin\n", argv[0]);
    return 2;
  }
  std::ifstream f(argv[1], std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string d = ss.str();
  size_t p = 0;
  auto take = [&](size_t n) {
    if (p + n > d.size()) {
      std::fprintf(stderr, "reference file truncated\n");
      std::exit(2);
    }
    const char* r = d.data() + p;
    p += n;
    return r;
  };
  if (d.compare(0, 8, "GIQREF01") != 0) return 2;
  take(8);
  uint32_t ncases;
  std::memcpy(&ncases, take(4), 4);
  Runtime rt;
  rt.init(std::string(espetero_kernels_metal, espetero_kernels_metal_len));
  std::mt19937 rng(7);
  std::normal_distribution<float> nd;
  int failures = 0;
  for (uint32_t c = 0; c < ncases; ++c) {
    uint32_t h[4];
    std::memcpy(h, take(16), 16);
    const int type = int(h[0]), N = int(h[1]), K = int(h[2]);
    id<MTLBuffer> W = rt.buffer(h[3]);
    std::memcpy([W contents], take(h[3]), h[3]);
    std::vector<float> ref(size_t(N) * K);
    std::memcpy(ref.data(), take(ref.size() * 4), ref.size() * 4);
    const char* kn = kernel_for(type);
    if (!kn) {
      std::printf("type %d: no kernel\n", type);
      ++failures;
      continue;
    }
    double worst = 0;
    for (int M : {1, 3, 8, 13}) {
      for (int idx = 0; idx < 2; ++idx) {
        // X: M rows (indexed: 2M rows, read in reverse order); Y: N per row (indexed: rows shuffled, scaled, += onto 1.0)
        const int XR = idx ? 2 * M : M;
        id<MTLBuffer> X = rt.buffer(size_t(XR) * K * 4), Y = rt.buffer(size_t(XR) * N * 4);
        id<MTLBuffer> xrow = rt.buffer(size_t(M) * 4), yrow = rt.buffer(size_t(M) * 4), ysc = rt.buffer(size_t(M) * 4);
        float* x = static_cast<float*>([X contents]);
        for (size_t i = 0; i < size_t(XR) * K; ++i) x[i] = nd(rng);
        float* y = static_cast<float*>([Y contents]);
        for (size_t i = 0; i < size_t(XR) * N; ++i) y[i] = 1.0f;
        for (int m = 0; m < M; ++m) {
          static_cast<int32_t*>([xrow contents])[m] = XR - 1 - m;
          static_cast<int32_t*>([yrow contents])[m] = (m * 5 + 1) % XR;
          static_cast<float*>([ysc contents])[m] = 0.5f + 0.25f * float(m);
        }
        id<MTLCommandBuffer> cb = [rt.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:rt.kernel(kn)];
        GemvP gp{N, K, M, K, N, 0, 0, idx, idx, 8};
        [e setBuffer:W offset:0 atIndex:0];
        [e setBuffer:X offset:0 atIndex:2];
        [e setBuffer:Y offset:0 atIndex:3];
        [e setBytes:&gp length:sizeof gp atIndex:4];
        [e setBuffer:xrow offset:0 atIndex:5];
        [e setBuffer:yrow offset:0 atIndex:6];
        [e setBuffer:ysc offset:0 atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake(size_t((N + 7) / 8), size_t((M + 7) / 8), 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        for (int m = 0; m < M; ++m) {
          const int xi = idx ? XR - 1 - m : m, yi = idx ? (m * 5 + 1) % XR : m;
          const double sc = idx ? 0.5 + 0.25 * m : 1.0;
          for (int n = 0; n < N; ++n) {
            double s = 0, mag = 0;
            for (int k = 0; k < K; ++k) {
              s += double(ref[size_t(n) * K + k]) * double(x[size_t(xi) * K + k]);
              mag += std::fabs(double(ref[size_t(n) * K + k]) * double(x[size_t(xi) * K + k]));
            }
            const double want = (idx ? 1.0 : 0.0) + sc * s;
            const double err = std::fabs(double(y[size_t(yi) * N + n]) - want) / (sc * mag + 1e-30);
            worst = std::max(worst, err);
          }
        }
      }
    }
    const bool ok = worst < 2e-6;
    std::printf("%-18s %4d x %-5d  worst |err| / sum|w x| %.2e%s\n", kn, N, K, worst, ok ? "" : "  FAIL");
    failures += !ok;
  }
  std::printf("%s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}
