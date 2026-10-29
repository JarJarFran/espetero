// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// EQV: vector-quantized routed experts (Qwen3.8-Flash-Next route B, ~1.7
// bits per weight; kernels/16_moe_vq.metal). A row of C inputs is C / 8
// vectors of 8 weights; vector v is scale(row, v / 32) * cb[code] with a
// K-bit code, one fp16 codebook [2^K][8] per stack and an fp16 scale per 256
// inputs (the last one partial: 640 = 256 + 256 + 128).
//
// Encoder:
//   codebook  weighted k-means (16 rounds) of rms-normalized vectors from a
//             sample of the stack's rows, importances = the expert's imatrix
//             times the group scale squared; rounded to fp16
//   scales    per row and 256-group: weighted rms, then two rounds of
//             nearest codewords + least squares, rounded to fp16
//   codes     LDLQ (GPTQ with 8-column blocks): gate/up with the router
//             input's Hessian Hs scaled per expert, H_e = S_e Hs S_e (S_e =
//             sqrt(imatrix_e / diag Hs): one factorization per layer); down
//             with each expert's own Hessian (calibration "X:" tensors,
//             --expert-hessian-max 640) or, without one, the imatrix-weighted
//             nearest codeword
// The nearest-codeword search runs on the GPU (vq_gpu.mm) when there is one.
#pragma once

#include <cstdint>
#include <vector>

#include "encode.h"

namespace espetero::quant {

struct VqOut {
  int kbits = 0, nv = 0, hw = 0, ns = 0;
  std::vector<uint8_t> lo;    // [R][nv] code bits 0-7
  std::vector<uint32_t> hi;   // [R][hw] the upper kbits - 8 bits of every vector, vector v at bit (kbits - 8) v
  std::vector<uint16_t> sc;   // [R][ns] fp16 scales
  std::vector<uint16_t> cb;   // [2^kbits][8] fp16 codebook
};

// W: [R][C] fp32 rows (R = E * out for stacked experts, nexp = E). ldlq:
// error feedback (needs Hshared or per-expert Hessians; otherwise ignored).
VqOut encode_vq(const float* W, int64_t R, int C, int nexp, int kbits, bool ldlq, const Stats* st);

// GPU nearest-codeword search (false: no GPU; the caller searches on the CPU).
bool vq_assign_gpu(const float* X, const float* W, int64_t N, const float* cb, int K, int32_t* out);

}  // namespace espetero::quant
