// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Check of the EQV expert kernels (kernels/16_moe_vq.metal) against a CPU
// reference: random codes / scales / codebooks for a stack of E experts, tiles
// of 1..8 rows, every moe_vq_gemv{1,2,4,8}_{store,swiglu} and vq_dequant;
// prints the largest relative error per kernel. Needs a GPU (run it through gputurn).
//   clang++ -std=c++17 -O2 -fobjc-arc tools/vq_check.mm -framework Metal -framework Foundation -o build/vq_check
//   build/vq_check [kernels-dir]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dirent.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct MoeVqParams {
  int32_t Ne, K, lda, ldy, gather, direct, kbits, nv, hw, ns, cbsel;
};

static std::string read_kernels(const std::string& dir) {
  std::vector<std::string> files;
  if (DIR* d = ::opendir(dir.c_str())) {
    while (dirent* e = ::readdir(d)) {
      const std::string n = e->d_name;
      if (n.size() > 6 && n.compare(n.size() - 6, 6, ".metal") == 0) files.push_back(dir + "/" + n);
    }
    ::closedir(d);
  }
  std::sort(files.begin(), files.end());
  std::string src;
  for (const auto& f : files) {
    std::ifstream in(f);
    std::stringstream ss;
    ss << in.rdbuf();
    src += ss.str();
  }
  return src;
}

static float h2f(__fp16 h) { return float(h); }

int main(int argc, char** argv) {
  @autoreleasepool {
    const std::string dir = argc > 1 ? argv[1] : "kernels";
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_2;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:read_kernels(dir).c_str()]
                                           options:opts
                                             error:&err];
    if (!lib) {
      std::fprintf(stderr, "compile: %s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    auto pipe = [&](const std::string& name) {
      MTLFunctionConstantValues* v = [MTLFunctionConstantValues new];
      const int g = 8, mode = 9;
      [v setConstantValue:&g type:MTLDataTypeInt atIndex:0];
      [v setConstantValue:&mode type:MTLDataTypeInt atIndex:1];
      NSError* e2 = nil;
      id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()] constantValues:v error:&e2];
      if (!fn) {
        std::fprintf(stderr, "missing %s\n", name.c_str());
        std::exit(1);
      }
      id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&e2];
      if (!p) {
        std::fprintf(stderr, "pipeline %s: %s\n", name.c_str(), [[e2 localizedDescription] UTF8String]);
        std::exit(1);
      }
      return p;
    };
    auto buf = [&](size_t n) { return [dev newBufferWithLength:std::max<size_t>(n, 16) options:MTLResourceStorageModeShared]; };
    std::mt19937 rng(1);
    std::normal_distribution<float> nd;
    int fails = 0;
    for (int down = 0; down < 2; ++down)
      for (int kbits : {13, 12, 9}) {
        const int E = 6, Ie = 64;
        const int K = down ? Ie * 10 : 2560;      // down: 640 inputs (a partial last scale group)
        const int Ne = down ? 96 : 2 * Ie;        // rows per expert (gate/up interleaved by 8)
        const int nv = K / 8, kb = kbits - 8, hw = (nv * kb + 31) / 32, ns = (K + 255) / 256, ncb = down ? 1 : 2;
        const size_t R = size_t(E) * Ne;
        std::vector<uint32_t> codes(R * nv);
        std::uniform_int_distribution<uint32_t> ud(0, (1u << kbits) - 1);
        for (auto& c : codes) c = ud(rng);
        id<MTLBuffer> lo = buf(R * nv), hi = buf(R * hw * 4), sc = buf(R * ns * 2), cb = buf(size_t(ncb) * 16 << kbits);
        uint8_t* plo = (uint8_t*)lo.contents;
        uint32_t* phi = (uint32_t*)hi.contents;
        std::fill(phi, phi + R * hw, 0u);
        for (size_t r = 0; r < R; ++r)
          for (int v = 0; v < nv; ++v) {
            const uint32_t c = codes[r * nv + v];
            plo[r * nv + v] = uint8_t(c & 255);
            for (int b = 0; b < kb; ++b)
              if ((c >> (8 + b)) & 1) {
                const size_t bit = size_t(v) * kb + b;
                phi[r * hw + bit / 32] |= 1u << (bit % 32);
              }
          }
        __fp16* psc = (__fp16*)sc.contents;
        for (size_t i = 0; i < R * ns; ++i) psc[i] = __fp16(0.5f + 0.01f * std::fabs(nd(rng)));
        __fp16* pcb = (__fp16*)cb.contents;
        for (size_t i = 0; i < (size_t(ncb) * 8 << kbits); ++i) pcb[i] = __fp16(nd(rng));
        auto weight = [&](size_t r, int k) {
          const int v = k / 8, cbi = ncb == 2 ? int((r % Ne) / 8) & 1 : 0;
          const uint32_t c = codes[r * nv + v];
          return h2f(pcb[((size_t(cbi) << kbits) + c) * 8 + k % 8]) * h2f(psc[r * ns + v / 32]);
        };
        // dequant check
        {
          id<MTLBuffer> out = buf(R * K * 2);
          MoeVqParams p{Ne, K, K, 0, 0, 1, kbits, nv, hw, ns, ncb == 2 ? 1 : 0};
          const int r0 = 0;
          id<MTLCommandBuffer> c = [q commandBuffer];
          id<MTLComputeCommandEncoder> e = [c computeCommandEncoder];
          [e setComputePipelineState:pipe("vq_dequant")];
          [e setBuffer:lo offset:0 atIndex:0];
          [e setBuffer:hi offset:0 atIndex:1];
          [e setBuffer:sc offset:0 atIndex:2];
          [e setBuffer:cb offset:0 atIndex:3];
          [e setBuffer:out offset:0 atIndex:4];
          [e setBytes:&p length:sizeof(p) atIndex:5];
          [e setBytes:&r0 length:4 atIndex:6];
          [e dispatchThreads:MTLSizeMake(size_t(nv), R, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
          [e endEncoding];
          [c commit];
          [c waitUntilCompleted];
          const __fp16* o = (const __fp16*)out.contents;
          double mx = 0;
          for (size_t r = 0; r < R; ++r)
            for (int k = 0; k < K; ++k) mx = std::max(mx, double(std::fabs(h2f(o[r * K + k]) - weight(r, k))));
          std::printf("vq_dequant           k%2d %s: max abs err %.3g\n", kbits, down ? "down" : "gu  ", mx);
          if (mx > 1e-2) ++fails;
        }
        for (int MM : {1, 2, 4, 8, 9}) {  // 9: moe_vq_gemm8 (8-row tiles on the matrix units)
          // tiles: expert e, rows r0..r0+rows (gathered A rows via rowmap)
          const int ntok = 13;
          std::vector<int32_t> tiles, rowmap;
          int slot = 0;
          for (int e = 0; e < E; ++e) {
            const int rows = 1 + (e * 3) % std::min(MM, 8);
            tiles.insert(tiles.end(), {e, slot, rows, 0});
            for (int m = 0; m < rows; ++m) rowmap.push_back((slot + m * 5 + e) % ntok);
            slot += rows;
          }
          const int ntiles = E;
          id<MTLBuffer> A = buf(size_t(ntok) * K * 2), tb = buf(tiles.size() * 4), rb = buf(rowmap.size() * 4), nt = buf(16);
          __fp16* pa = (__fp16*)A.contents;
          for (size_t i = 0; i < size_t(ntok) * K; ++i) pa[i] = __fp16(nd(rng));
          std::memcpy(tb.contents, tiles.data(), tiles.size() * 4);
          std::memcpy(rb.contents, rowmap.data(), rowmap.size() * 4);
          ((uint32_t*)nt.contents)[0] = uint32_t(ntiles);
          for (int epi : {0, 2}) {
            if (epi == 2 && down) continue;
            const int ldy = epi == 2 ? Ne / 2 : Ne;
            id<MTLBuffer> Y = buf(size_t(slot) * ldy * 4);
            std::memset(Y.contents, 0, size_t(slot) * ldy * 4);
            MoeVqParams p{Ne, K, K, ldy, 1, 0, kbits, nv, hw, ns, ncb == 2 ? 1 : 0};
            const std::string name = (MM == 9 ? std::string("moe_vq_gemm8") : "moe_vq_gemv" + std::to_string(MM)) +
                                     (epi == 2 ? "_swiglu" : "_store");
            id<MTLCommandBuffer> c = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [c computeCommandEncoder];
            [e setComputePipelineState:pipe(name)];
            [e setBuffer:A offset:0 atIndex:0];
            [e setBuffer:lo offset:0 atIndex:1];
            [e setBuffer:hi offset:0 atIndex:2];
            [e setBuffer:sc offset:0 atIndex:3];
            [e setBuffer:Y offset:0 atIndex:4];
            [e setBytes:&p length:sizeof(p) atIndex:5];
            [e setBuffer:tb offset:0 atIndex:6];
            [e setBuffer:nt offset:0 atIndex:7];
            [e setBuffer:rb offset:0 atIndex:8];
            [e setBuffer:cb offset:0 atIndex:9];
            [e dispatchThreadgroups:MTLSizeMake(size_t(ntiles), size_t(Ne / 32), 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            [c commit];
            [c waitUntilCompleted];
            double mx = 0, ref_mx = 0;
            for (int t = 0; t < ntiles; ++t) {
              const int ex = tiles[4 * t], r0 = tiles[4 * t + 1], rows = tiles[4 * t + 2];
              for (int m = 0; m < rows; ++m) {
                const __fp16* x = pa + size_t(rowmap[r0 + m]) * K;
                std::vector<double> y(Ne);
                for (int n = 0; n < Ne; ++n) {
                  double s = 0;
                  for (int k = 0; k < K; ++k) s += double(weight(size_t(ex) * Ne + n, k)) * h2f(x[k]);
                  y[n] = s;
                }
                for (int n = 0; n < (epi == 2 ? Ne / 2 : Ne); ++n) {
                  double ref, got;
                  if (epi == 2) {
                    const int blk = n / 8, j = n % 8;
                    const double g = y[blk * 16 + j], u = y[blk * 16 + 8 + j];
                    ref = g / (1.0 + std::exp(-g)) * u;
                    got = h2f(((const __fp16*)Y.contents)[size_t(r0 + m) * ldy + n]);
                  } else {
                    ref = y[n];
                    got = ((const float*)Y.contents)[size_t(r0 + m) * ldy + n];
                  }
                  mx = std::max(mx, std::fabs(got - ref));
                  ref_mx = std::max(ref_mx, std::fabs(ref));
                }
              }
            }
            const double rel = mx / std::max(ref_mx, 1e-9);
            std::printf("%-20s k%2d %s: max rel err %.3g\n", name.c_str(), kbits, down ? "down" : "gu  ", rel);
            if (rel > (MM == 9 ? 2e-2 : 5e-3)) ++fails;  // (gemm8: fp16 dequantized weights and fp16 x products)
          }
        }
      }
    std::printf(fails ? "FAILED (%d)\n" : "ok\n", fails);
    return fails ? 1 : 0;
  }
}
