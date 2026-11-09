// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// NaN / Inf / outlier detector for debugging (ESPETERO_CHECK_FINITE).
//
// After chosen dispatches (the check sites: the residual stream after each
// layer, the attention output, the MoE output, the logits...) the encoder adds
// a finite_check dispatch (kernels/05_finite_check.metal) over the
// destination region. Each check owns one record of a shared buffer: the
// offending elements' count, the first offender's index and its value. When
// the pass's GPU work is done the host reads the records in encoding order
// and reports the first offender: model, layer, op, row, column, value and
// pass id; then fails the pass (the request ends with that error) or only
// logs it. Off (the default), no site encodes anything: no dispatch, no
// buffer, the same bits.
//
// This file is the host side without Metal (options, registry, report;
// tests/finite_check_test.cpp); the records and dispatches are in runtime.h.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace espetero::native {

enum class FDtype : uint32_t { F32 = 0, F16 = 1, BF16 = 2 };

// One check's record on the GPU (finite_check's `rec`).
struct FiniteRec {
  uint32_t hits = 0;           // offending elements
  uint32_t idx = 0xFFFFFFFFu;  // the first one (row * cols + col)
  uint32_t done = 0;           // threadgroups finished (the last one writes val)
  uint32_t val = 0;            // its value's fp32 bits
};

// finite_check / finite_poison parameters (FiniteCheckP).
struct FiniteParams {
  uint32_t rows, cols, ld, dtype, thr_bits, elem, pad0 = 0, pad1 = 0;
};

// A check site of the registry: what one record covers.
struct FiniteSite {
  const char* op = "";  // attn_out, attn_res, moe_out, layer_out, logits... (a string literal)
  int layer = -1;       // -1: not a layer's (logits, embeddings)
  int rows = 0, cols = 0;
  FDtype dt = FDtype::F32;
  uint64_t pass = 0;
};

struct FiniteOptions {
  bool on = false;
  bool abort = true;             // fail the pass (default) or log only
  float thr = INFINITY;          // |x| >= thr offends; NaN and Inf always
  std::string poison_op;         // ESPETERO_CHECK_FINITE_POISON=op[:layer[:row:col]]
  int poison_layer = -1, poison_row = 0, poison_col = 0;
  std::string error;             // a value that does not parse
};

// ESPETERO_CHECK_FINITE: unset, "" or "0": off. Otherwise a comma list of
// "1" / "on", a threshold (a number > 1, e.g. 1e11: |x| >= it offends as NaN
// and Inf do), "log" (report and go on) or "abort" (the default: the pass
// fails with the report as its error).
inline FiniteOptions parse_finite_options(const char* v, const char* poison) {
  FiniteOptions o;
  if (!v || !*v || std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0) return o;
  o.on = true;
  std::string s(v);
  for (size_t a = 0; a <= s.size();) {
    size_t b = s.find(',', a);
    if (b == std::string::npos) b = s.size();
    const std::string t = s.substr(a, b - a);
    a = b + 1;
    if (t.empty() || t == "1" || t == "on") continue;
    if (t == "log") {
      o.abort = false;
    } else if (t == "abort") {
      o.abort = true;
    } else {
      char* end = nullptr;
      const double x = std::strtod(t.c_str(), &end);
      if (end == t.c_str() || *end || !(x > 1.0)) {
        o.error = "ESPETERO_CHECK_FINITE: '" + t + "' is not 1, log, abort or a threshold > 1";
        continue;
      }
      o.thr = x >= 3.4e38 ? INFINITY : float(x);
    }
  }
  if (poison && *poison) {
    std::string p(poison);
    std::vector<std::string> f;
    for (size_t a = 0; a <= p.size();) {
      size_t b = p.find(':', a);
      if (b == std::string::npos) b = p.size();
      f.push_back(p.substr(a, b - a));
      a = b + 1;
    }
    auto num = [&](size_t i, int& out, long lo) {
      if (i >= f.size()) return true;
      char* end = nullptr;
      const long x = std::strtol(f[i].c_str(), &end, 10);
      if (end == f[i].c_str() || *end || x < lo) return false;
      out = int(x);
      return true;
    };
    // op[:layer[:row:col]]; layer -1 (or none): a site outside the layers (logits)
    if (f.size() > 4 || f.size() == 3 || f[0].empty() || !num(1, o.poison_layer, -1) || !num(2, o.poison_row, 0) ||
        !num(3, o.poison_col, 0))
      o.error = "ESPETERO_CHECK_FINITE_POISON: '" + p + "' is not op[:layer[:row:col]]";
    else
      o.poison_op = f[0];
  }
  return o;
}

// fp32 bits of the threshold for the kernel (|x| bits >= them offend).
inline uint32_t finite_thr_bits(float thr) {
  if (!(thr < INFINITY)) return 0x7F800000u;
  uint32_t b;
  std::memcpy(&b, &thr, 4);
  return b & 0x7FFFFFFFu;
}

inline const char* finite_dtype_name(FDtype d) { return d == FDtype::F32 ? "f32" : d == FDtype::F16 ? "f16" : "bf16"; }

inline std::string finite_value_str(uint32_t bits) {
  float x;
  std::memcpy(&x, &bits, 4);
  if (std::isnan(x)) return (bits >> 31) ? "-nan" : "nan";
  if (std::isinf(x)) return x < 0 ? "-inf" : "inf";
  char b[32];
  std::snprintf(b, sizeof(b), "%.6g", double(x));
  return b;
}

// The report of the first offender (and how many checks of the pass flagged).
inline std::string finite_message(const std::string& model, const FiniteSite& s, const FiniteRec& r, int flagged,
                                  const FiniteSite* last) {
  const uint32_t cols = uint32_t(s.cols > 0 ? s.cols : 1);
  const uint32_t row = r.idx / cols, col = r.idx % cols;
  const std::string v = finite_value_str(r.val);
  const char* kind = v.find("nan") != std::string::npos ? "NaN" : v.find("inf") != std::string::npos ? "Inf" : "outlier";
  const std::string where = (s.layer >= 0 ? "layer " + std::to_string(s.layer) + " " : std::string()) + s.op;
  char b[512];
  std::snprintf(b, sizeof(b), "check-finite: %s in %s, %s, row %u, col %u (of %dx%d %s), value %s, %u offending, pass %llu", kind,
                model.empty() ? "model" : model.c_str(), where.c_str(), row, col, s.rows, s.cols, finite_dtype_name(s.dt),
                v.c_str(), r.hits, (unsigned long long)s.pass);
  std::string m = b;
  if (flagged > 1 && last) {
    m += "; " + std::to_string(flagged - 1) + " later check" + (flagged > 2 ? "s" : "") + " also flagged (last: " +
         (last->layer >= 0 ? "layer " + std::to_string(last->layer) + " " : std::string()) + last->op + ")";
  }
  return m;
}

// The checks of the encoded passes, in encoding order, and their records.
// scan(): the first flagged record's position (-1: none) and the number flagged.
struct FiniteScan {
  int first = -1, last = -1, flagged = 0;
};
inline FiniteScan finite_scan(const FiniteRec* recs, size_t n) {
  FiniteScan s;
  for (size_t i = 0; i < n; ++i) {
    if (!recs[i].hits) continue;
    if (s.first < 0) s.first = int(i);
    s.last = int(i);
    ++s.flagged;
  }
  return s;
}

// Whether the poison option names this site; *elem: the element to overwrite.
inline bool finite_poison_at(const FiniteOptions& o, const FiniteSite& s, int ld, uint32_t* elem) {
  if (o.poison_op.empty() || o.poison_op != s.op || o.poison_layer != s.layer) return false;
  if (o.poison_row >= s.rows || o.poison_col >= s.cols) return false;
  *elem = uint32_t(o.poison_row) * uint32_t(ld) + uint32_t(o.poison_col);
  return true;
}

}  // namespace espetero::native
