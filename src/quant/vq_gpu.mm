// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The EQV encoder's nearest-codeword search on the GPU (vq.cpp): for N
// 8-weight vectors x with per-weight importances w, argmin_k sum_j w_j (x_j -
// cb_kj)^2 over a 2^K-entry codebook, one thread per vector, the codebook
// staged through threadgroup memory 512 entries at a time.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <mutex>

#include "vq.h"

namespace espetero::quant {
namespace {

const char* kAssignSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void vq_assign8(device const float4* X [[buffer(0)]], device const float4* W [[buffer(1)]],
                       device const float4* C [[buffer(2)]], device int* out [[buffer(3)]],
                       constant uint2& nk [[buffer(4)]], uint i [[thread_position_in_grid]],
                       uint t [[thread_position_in_threadgroup]]) {
  const uint K = nk.x, N = nk.y;
  threadgroup float4 cbt[1024];
  float4 xa = 0.0f, xb = 0.0f, wa = 0.0f, wb = 0.0f;
  if (i < N) {
    xa = X[2 * i];
    xb = X[2 * i + 1];
    wa = W[2 * i];
    wb = W[2 * i + 1];
  }
  const float4 ma = -2.0f * wa * xa, mb = -2.0f * wb * xb;
  float best = INFINITY;
  uint bi = 0;
  for (uint k0 = 0; k0 < K; k0 += 512) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = t; j < 1024; j += 256) cbt[j] = k0 + j / 2 < K ? C[2 * k0 + j] : float4(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint kn = min(512u, K - k0);
    for (uint k = 0; k < kn; ++k) {
      const float4 ca = cbt[2 * k], cb = cbt[2 * k + 1];
      const float d = dot(ca * wa + ma, ca) + dot(cb * wb + mb, cb);
      if (d < best) {
        best = d;
        bi = k0 + k;
      }
    }
  }
  if (i < N) out[i] = int(bi);
}
)MSL";

struct Gpu {
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> q = nil;
  id<MTLComputePipelineState> p = nil;
  id<MTLBuffer> x = nil, w = nil, c = nil, o = nil;
  size_t cap = 0, ccap = 0;
  std::mutex mu;
  bool ok = false;
  Gpu() {
    @autoreleasepool {
      dev = MTLCreateSystemDefaultDevice();
      if (!dev) return;
      NSError* err = nil;
      id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kAssignSrc] options:nil error:&err];
      if (!lib) return;
      id<MTLFunction> fn = [lib newFunctionWithName:@"vq_assign8"];
      p = fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
      q = [dev newCommandQueue];
      ok = p != nil && q != nil;
    }
  }
};

Gpu& gpu() {
  static Gpu g;
  return g;
}

}  // namespace

bool vq_assign_gpu(const float* X, const float* W, int64_t N, const float* cb, int K, int32_t* out) {
  Gpu& g = gpu();
  if (!g.ok || std::getenv("ESPETERO_VQ_CPU")) return false;
  std::lock_guard<std::mutex> lk(g.mu);
  @autoreleasepool {
    const int64_t step = int64_t(1) << 21;
    const size_t need = size_t(std::min<int64_t>(N, step));
    if (g.cap < need) {
      g.cap = need;
      g.x = [g.dev newBufferWithLength:need * 32 options:MTLResourceStorageModeShared];
      g.w = [g.dev newBufferWithLength:need * 32 options:MTLResourceStorageModeShared];
      g.o = [g.dev newBufferWithLength:need * 4 options:MTLResourceStorageModeShared];
    }
    if (g.ccap < size_t(K)) {
      g.ccap = size_t(K);
      g.c = [g.dev newBufferWithLength:size_t(K) * 32 options:MTLResourceStorageModeShared];
    }
    std::memcpy(g.c.contents, cb, size_t(K) * 32);
    for (int64_t i0 = 0; i0 < N; i0 += step) {
      const int64_t n = std::min(step, N - i0);
      std::memcpy(g.x.contents, X + i0 * 8, size_t(n) * 32);
      std::memcpy(g.w.contents, W + i0 * 8, size_t(n) * 32);
      const uint32_t nk[2] = {uint32_t(K), uint32_t(n)};
      id<MTLCommandBuffer> cbuf = [g.q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cbuf computeCommandEncoder];
      [e setComputePipelineState:g.p];
      [e setBuffer:g.x offset:0 atIndex:0];
      [e setBuffer:g.w offset:0 atIndex:1];
      [e setBuffer:g.c offset:0 atIndex:2];
      [e setBuffer:g.o offset:0 atIndex:3];
      [e setBytes:nk length:8 atIndex:4];
      [e dispatchThreadgroups:MTLSizeMake(size_t((n + 255) / 256), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      [cbuf commit];
      [cbuf waitUntilCompleted];
      if (cbuf.status != MTLCommandBufferStatusCompleted) return false;
      std::memcpy(out + i0, g.o.contents, size_t(n) * 4);
    }
  }
  return true;
}

}  // namespace espetero::quant
