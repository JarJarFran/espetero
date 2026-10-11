// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Compile the engine's Metal kernels (kernels/*.metal, or the directory in
// ESPETERO_KERNELS) with the driver's compiler and print errors: a CPU-only
// syntax and type check that needs no model and no GPU work.
//   clang++ -std=c++17 -fobjc-arc tools/metal_check.mm -framework Metal -framework Foundation -o build/metal_check && build/metal_check
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dirent.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  @autoreleasepool {
    const char* env = std::getenv("ESPETERO_KERNELS");
    const std::string dir = argc > 1 ? argv[1] : (env ? env : "kernels");
    std::vector<std::string> files;
    if (DIR* d = ::opendir(dir.c_str())) {
      while (dirent* e = ::readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > 6 && n.compare(n.size() - 6, 6, ".metal") == 0) files.push_back(dir + "/" + n);
      }
      ::closedir(d);
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
      std::fprintf(stderr, "no .metal files in %s\n", dir.c_str());
      return 2;
    }
    std::string src;
    for (const auto& f : files) {
      std::ifstream in(f, std::ios::binary);
      std::stringstream ss;
      ss << in.rdbuf();
      src += ss.str() + "\n";
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    MTLCompileOptions* opts = [MTLCompileOptions new];
    // ESPETERO_MSL=3.1|3.2|4.0 selects the language version (the engine uses 3.1).
    const char* v = std::getenv("ESPETERO_MSL");
    const std::string msl = v ? v : "3.1";
    opts.languageVersion = msl == "4.0" ? MTLLanguageVersion4_0 : msl == "3.2" ? MTLLanguageVersion3_2 : MTLLanguageVersion3_1;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:opts error:&err];
    if (!lib) {
      std::fprintf(stderr, "%s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }
    std::printf("ok: %zu files, %lu functions\n", files.size(), (unsigned long)[[lib functionNames] count]);
    // Extra arguments: kernels whose pipeline limits to print (threads per
    // threadgroup, static threadgroup memory).
    for (int i = 2; i < argc; ++i) {
      id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:argv[i]]];
      if (!f) {
        std::printf("%s: no such function\n", argv[i]);
        continue;
      }
      id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:f error:&err];
      if (!ps) {
        std::printf("%s: %s\n", argv[i], [[err localizedDescription] UTF8String]);
        continue;
      }
      std::printf("%s: max threads %lu, threadgroup memory %lu B, simd width %lu\n", argv[i],
                  (unsigned long)ps.maxTotalThreadsPerThreadgroup, (unsigned long)ps.staticThreadgroupMemoryLength,
                  (unsigned long)ps.threadExecutionWidth);
    }
    return 0;
  }
}
