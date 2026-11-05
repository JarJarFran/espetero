// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Calibration statistics for espetero quantize: E[x x^T] of every matrix
// input (its diagonal past --hessian-max inputs), per-expert diagonals for
// stacked experts and per-expert E[x x^T] for narrow expert inputs, captured
// from Espetero's own forward of the BF16 checkpoint (the reference harness's
// statistics, fp16 activations instead of fp32; experts as exact fp16
// copies, or 8-bit with --calib-experts q8).
//
// File: one safetensors file, tensors "H:<module>" [C, C], "U:<module>"
// [C (C + 1) / 2] (E[x x^T]'s upper triangle by rows, for inputs wider than
// --hessian-pack: half the bytes), "D:<module>" [C], "E:<module>" [E, C],
// "X:<module>" [E, C, C] (float32); metadata "ALIAS" = JSON {module: module
// whose input it shares}. tools/dev/calib_npz.py converts the Python harness's
// .npz to this form.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "encode.h"
#include "st.h"

namespace espetero::quant {

class CalibFile {
 public:
  // One statistics file, or several comma-separated ones with disjoint
  // matrices (the parts of a capture split with --calib-layers).
  explicit CalibFile(const std::string& paths);
  // Statistics of `module` (MLX path without .weight), or null.
  const Stats* get(const std::string& module) const;
  size_t size() const { return stats_.size(); }

 private:
  std::vector<std::unique_ptr<StFile>> f_;
  std::map<std::string, Stats> stats_;
};

// `st` with a whole H: a packed upper triangle is unpacked into `buf` (and
// `tmp` returned), anything else is returned as it is.
const Stats* full_hessian(const Stats* st, Stats& tmp, std::vector<float>& buf);

struct CaptureOptions {
  std::string model;    // Hugging Face checkpoint directory
  std::string data;     // file:PATH or glob:PATTERN, '+'-joined
  std::string out;      // statistics file
  int ctx = 2048;
  int64_t max_tokens = 262144;
  int hessian_max = 16384;   // full E[x x^T] up to this many inputs, else its diagonal
  int hessian_pack = 16384;  // full E[x x^T] wider than this saved as its upper triangle
  int expert_hessian_max = 1024;
  double kv_gb = 2;
  bool experts_f16 = true;  // the forward's experts exact (fp16), else 8-bit copies
  // Statistics for layers [layer_lo, layer_hi) only (the forward still runs every layer); the LM
  // head's input is captured only when the range reaches the last layer. Splitting a capture into
  // ranges caps its memory; the parts merge into one file (tools/calib_merge.py) or are read
  // together (--calib A,B,...).
  int layer_lo = 0;
  int layer_hi = 1 << 30;
  // Routed experts' statistics: each routed (token, expert) pair weighted by its routing weight
  // (the normalized top-k probability the combine applies; MoEQuant's gate-weighted Hessians) and
  // normalized by the weight sum, instead of every pair counting once (--calib-expert-weight router).
  bool expert_route_weight = false;
};
void capture(const CaptureOptions& o);

}  // namespace espetero::quant
