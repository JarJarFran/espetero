// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// GGUF codebook blocks (IQ2_S, IQ3_S, IQ2_XS, IQ3_XXS) converted into the engine's affine
// layout (gguf.h convert_row: w = scale * code + bias, half scale and bias per
// group) against gguf-py's dequantization of the same blocks
// (tests/ref/gguf_iq_ref.py). The only allowed difference is the fp16
// rounding of the group scale and bias; a wrong index, sign or nibble shows
// up as an error of whole grid steps.
//
//   python3 tests/ref/gguf_iq_ref.py /tmp/giq.bin [--real IQ2_S:64:4096:blocks.bin ...]
//   build/gguf_iq_test /tmp/giq.bin
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/model/impl/gguf.h"

using namespace espetero::native;

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
  if (d.compare(0, 8, "GIQREF01") != 0) {
    std::fprintf(stderr, "not a GIQREF01 file\n");
    return 2;
  }
  take(8);
  uint32_t ncases;
  std::memcpy(&ncases, take(4), 4);
  int failures = 0;
  for (uint32_t c = 0; c < ncases; ++c) {
    uint32_t h[4];
    std::memcpy(h, take(16), 16);
    const int type = int(h[0]);
    const int64_t rows = h[1], cols = h[2];
    const uint8_t* blocks = reinterpret_cast<const uint8_t*>(take(h[3]));
    std::vector<float> ref(size_t(rows * cols));
    std::memcpy(ref.data(), take(ref.size() * 4), ref.size() * 4);
    const GGMLTypeInfo* ti = ggml_type_info(type);
    if (!ti) {
      std::fprintf(stderr, "type %d: not supported by gguf.h\n", type);
      ++failures;
      continue;
    }
    const int64_t row_bytes = cols / ti->block * ti->bytes, G = cols / ti->group;
    std::vector<uint8_t> codes(size_t(ti->code_bits == 4 ? cols / 2 : cols));
    std::vector<uint16_t> sc(static_cast<size_t>(G)), bi(static_cast<size_t>(G));
    double max_err = 0, max_rel = 0;
    int64_t bad = 0, exact = 0;
    for (int64_t r = 0; r < rows; ++r) {
      std::fill(codes.begin(), codes.end(), 0);
      RowSink s{ti->code_bits, codes.data(), sc.data(), bi.data()};
      convert_row(type, blocks + r * row_bytes, cols, s);
      for (int64_t i = 0; i < cols; ++i) {
        const uint32_t code = ti->code_bits == 4 ? (reinterpret_cast<const uint32_t*>(codes.data())[i >> 3] >> (4 * (i & 7))) & 0xF
                                                 : codes[size_t(i)];
        const float s16 = f16_to_f32(sc[size_t(i / ti->group)]), b16 = f16_to_f32(bi[size_t(i / ti->group)]);
        const float w = s16 * float(code) + b16;
        const float want = ref[size_t(r * cols + i)];
        const double err = std::fabs(double(w) - double(want));
        // fp16 rounding of scale and bias: half an ulp each, relative 2^-11
        // (+ half an fp16 subnormal step, 2^-25, each when the group scale is below fp16's normal range)
        const double bound = (std::fabs(s16) * code + std::fabs(b16)) * std::ldexp(1.0, -10) + (code + 1) * std::ldexp(1.0, -24);
        if (err > bound) {
          if (bad < 5)
            std::fprintf(stderr, "  %s row %lld col %lld: engine %g, reference %g (code %u, scale %g, bias %g)\n", ti->name,
                         (long long)r, (long long)i, w, want, code, s16, b16);
          ++bad;
        }
        exact += w == want;
        max_err = std::max(max_err, err);
        if (want != 0) max_rel = std::max(max_rel, err / std::fabs(want));
      }
    }
    // the exact float decoder: bit for bit
    int64_t diff = 0;
    std::vector<float> y(static_cast<size_t>(cols));
    for (int64_t r = 0; r < rows; ++r) {
      dequant_iq_row(type, blocks + r * row_bytes, cols, y.data());
      for (int64_t i = 0; i < cols; ++i) diff += std::memcmp(&y[size_t(i)], &ref[size_t(r * cols + i)], 4) != 0;
    }
    std::printf("%-6s %4lld x %-5lld  affine: max |err| %.3g, max rel %.3g, %lld beyond the fp16 bound | exact decoder: %lld of %lld differ%s\n",
                ti->name, (long long)rows, (long long)cols, max_err, max_rel, (long long)bad, (long long)diff,
                (long long)(rows * cols), bad || diff ? "  FAIL" : "");
    (void)exact;
    failures += bad != 0 || diff != 0;
  }
  std::printf("%s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}
