// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The encoders of espetero quantize (CPU): a C++ port of the reference
// Python converter (kld/quant_lab.py + eq_convert.py), same math and the same
// float roundings, run on Accelerate (BLAS, LAPACK) and GCD.
//
// Formats (specs as in eq_convert.py's rules):
//   eql:B:16[+ENC]        EQ, B in 3, 4, 5, 6, 8: signed codes, an int8 sub-scale
//                         per lane sub-block of 16 inputs, an fp16 super-scale
//                         per 256 (the last super-block of a row partial when
//                         the width is not a multiple of 256: 640 = 256 + 256 + 128);
//                         weight = half(super * sub) * code
//   eql:4:16:cb[:R][+ENC] EQ4 with a 16-entry fp16 codebook per matrix (or per
//                         R rows; per expert for stacked experts):
//                         weight = half(super * sub) * cb[code]
//   eql:3:16:cb[:R][+ENC] EQ3 with an 8-entry codebook, the same way (Espetero
//                         runs EQ3 for routed expert stacks only)
//   affine:B:G[:SD][+ENC] MLX affine, scale and bias per G in SD (bf16, f16, f32)
//   bf16                  kept as it is
// Encoders: rtn (round to nearest; for affine, MLX's own mx.quantize), mse
// (weighted least-squares scale search, unit weights), imat (the same search
// weighted by E[x_j^2] of the layer's input), gptq (GPTQ with the input's
// E[x x^T]; group scales from the imat search on the updated weights).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace espetero::quant {

struct Fmt {
  enum Kind { BF16, Affine, EQ, EQC } kind = BF16;
  enum Enc { RTN, MSE, IMAT, GPTQ } enc = RTN;
  int bits = 0;
  int group = 0;            // affine: G; EQ / EQC: 256 (one super-block)
  std::string sd = "bf16";  // affine scale dtype
  int cbrows = 0;           // EQC: rows per codebook (0: per matrix / expert)
  std::string spec;
};
Fmt parse_fmt(const std::string& spec);

// Calibration statistics of one matrix's input (pointers into the mapped
// statistics file; any may be null).
struct Stats {
  const float* H = nullptr;        // [C][C] E[x x^T]
  const float* diag = nullptr;     // [C] E[x_j^2], or [E][C] per expert
  int diag_rows = 0;               // 1 or E
  const float* Hexp = nullptr;     // [E][C][C] per-expert E[x x^T]
  int nexp = 0;
  const float* Hshared = nullptr;  // [C][C] the router input's (stacked experts' gate/up)
  const float* Hpacked = nullptr;  // [C (C + 1) / 2] E[x x^T]'s upper triangle by rows, when H is not stored whole
  int C = 0;
};

// EQ / EQC encoding of R rows of C inputs (C % 64 == 0).
struct EqOut {
  std::vector<uint8_t> q;    // [R][C] stored codes, original column order: code + 2^(B-1), or codebook index
  std::vector<int8_t> sc;    // [R][C / 16] sub-scales, lane sub-block order (4 * step + t)
  std::vector<uint16_t> d;   // [R][ceil(C / 256)] fp16 super-scales
  std::vector<uint16_t> cb;  // EQC: [ncb][2^bits] fp16 codebooks (EQ4 16, EQ3 8)
};
// MLX affine encoding.
struct AffOut {
  std::vector<uint8_t> q;    // [R][C] codes
  std::vector<float> s, b;   // [R][C / G] scale and bias, rounded to the scale dtype
};

// W: [R][C] fp32 rows of a matrix (R = E * out for stacked experts, nexp = E;
// 0 for a plain matrix). Throws on combinations the reference does not
// support.
EqOut encode_eq(const float* W, int64_t R, int C, int nexp, const Fmt& f, const Stats* st);
AffOut encode_affine(const float* W, int64_t R, int C, int nexp, const Fmt& f, const Stats* st);

// Packed little-endian bit stream of B-bit codes per row (MLX's packing):
// [R][C * B / 32] words.
std::vector<uint32_t> pack_codes(const uint8_t* q, int64_t R, int C, int bits);

// The column order whose contiguous 16-wide groups are EQ's lane sub-blocks.
std::vector<int> lane_perm(int C);

float f16_round(float x);
float bf16_round_f(float x);
uint16_t f16_bits(float x);
uint16_t bf16_bits(float x);

}  // namespace espetero::quant
