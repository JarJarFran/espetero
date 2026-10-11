// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// JEVfire Metal engine kernels.
//
// Qwen3.5-family hybrid decoder (Gated DeltaNet + gated full attention) for
// "one context, many decisions": a shared prefix and N field suffixes are
// processed as one tree batch. Rows are packed: the optional root segment
// first, then every leaf segment. Leaves branch from the root's final state
// (or from a cached checkpoint when there is no root).
//
// Weights: an integer code times a per-group scale plus a bias (MLX affine
// and every supported GGUF block type map to it exactly). 4-bit codes are
// nibbles, 5/6-bit a nibble plane plus a high-bit plane, other widths bytes.
// The vision tower keeps its weights as fp16 (so can selected decoder
// matrices). GEMM operands are fp16 with fp32 accumulation; the residual
// stream, recurrent state and projections are fp32.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define UNROLL _Pragma("clang loop unroll(full)")

static inline float bf2f(ushort x) { return as_type<float>(uint(x) << 16); }
static inline float silu(float x) { return x / (1.0f + exp(-x)); }
