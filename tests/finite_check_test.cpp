// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ESPETERO_CHECK_FINITE's host side without a GPU (src/model/impl/finite_check.h):
// option parsing, the threshold bits, poison sites, the first-offender scan
// and the report. The GPU side: tests/finite_test.mm.
#include <cstdio>
#include <string>

#include "../src/model/impl/finite_check.h"

using namespace espetero::native;

static int fails = 0;
#define CHECK(c)                                                    \
  do {                                                              \
    if (!(c)) {                                                     \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);      \
      ++fails;                                                      \
    }                                                               \
  } while (0)

static uint32_t bits(float x) {
  uint32_t b;
  std::memcpy(&b, &x, 4);
  return b;
}

int main() {
  // ---- options
  CHECK(!parse_finite_options(nullptr, nullptr).on);
  CHECK(!parse_finite_options("", nullptr).on);
  CHECK(!parse_finite_options("0", nullptr).on);
  {
    const FiniteOptions o = parse_finite_options("1", nullptr);
    CHECK(o.on && o.abort && !(o.thr < INFINITY) && o.error.empty() && o.poison_op.empty());
  }
  {
    const FiniteOptions o = parse_finite_options("1e11", nullptr);
    CHECK(o.on && o.abort && o.thr == 1e11f && o.error.empty());
  }
  {
    const FiniteOptions o = parse_finite_options("log,1e11", nullptr);
    CHECK(o.on && !o.abort && o.thr == 1e11f);
  }
  CHECK(parse_finite_options("1,abort", nullptr).abort);
  CHECK(!parse_finite_options("bogus", nullptr).error.empty());
  CHECK(!parse_finite_options("0.5", nullptr).error.empty());  // thresholds are > 1
  {
    const FiniteOptions o = parse_finite_options("1", "attn_out:3:2:5");
    CHECK(o.error.empty() && o.poison_op == "attn_out" && o.poison_layer == 3 && o.poison_row == 2 && o.poison_col == 5);
  }
  {
    const FiniteOptions o = parse_finite_options("1", "logits:-1:0:7");
    CHECK(o.error.empty() && o.poison_op == "logits" && o.poison_layer == -1 && o.poison_col == 7);
  }
  {
    const FiniteOptions o = parse_finite_options("1", "layer_out:4");
    CHECK(o.error.empty() && o.poison_layer == 4 && o.poison_row == 0 && o.poison_col == 0);
  }
  CHECK(!parse_finite_options("1", "attn_out:3:2").error.empty());   // row without col
  CHECK(!parse_finite_options("1", "attn_out:x").error.empty());
  CHECK(!parse_finite_options("1", "attn_out:1:-2:0").error.empty());

  // ---- threshold bits: |x| bits >= them offend
  CHECK(finite_thr_bits(INFINITY) == 0x7F800000u);
  CHECK(finite_thr_bits(1e11f) == bits(1e11f));
  CHECK((bits(-2e11f) & 0x7FFFFFFFu) >= finite_thr_bits(1e11f));
  CHECK((bits(9e10f) & 0x7FFFFFFFu) < finite_thr_bits(1e11f));
  CHECK((bits(NAN) & 0x7FFFFFFFu) >= finite_thr_bits(1e11f));  // NaN always offends
  CHECK((bits(65504.0f) & 0x7FFFFFFFu) < finite_thr_bits(INFINITY));

  // ---- poison sites
  {
    const FiniteOptions o = parse_finite_options("1", "attn_out:3:2:5");
    FiniteSite s;
    s.op = "attn_out", s.layer = 3, s.rows = 8, s.cols = 64;
    uint32_t e = 0;
    CHECK(finite_poison_at(o, s, 128, &e) && e == 2 * 128 + 5);
    s.layer = 4;
    CHECK(!finite_poison_at(o, s, 128, &e));
    s.layer = 3, s.rows = 2;  // the row is not in this region (a decode step)
    CHECK(!finite_poison_at(o, s, 128, &e));
    s.rows = 8, s.op = "attn_res";
    CHECK(!finite_poison_at(o, s, 128, &e));
  }

  // ---- scan and report: the first record that flagged, in encoding order
  {
    FiniteRec r[5];
    CHECK(finite_scan(r, 5).first == -1 && finite_scan(r, 5).flagged == 0);
    r[2].hits = 3, r[2].idx = 2 * 64 + 5, r[2].val = bits(NAN);
    r[4].hits = 100, r[4].idx = 0, r[4].val = bits(INFINITY);
    const FiniteScan sc = finite_scan(r, 5);
    CHECK(sc.first == 2 && sc.last == 4 && sc.flagged == 2);
    FiniteSite s2, s4;
    s2.op = "attn_out", s2.layer = 3, s2.rows = 8, s2.cols = 64, s2.dt = FDtype::F16, s2.pass = 7;
    s4.op = "logits", s4.layer = -1, s4.rows = 1, s4.cols = 1000;
    const std::string m = finite_message("qwen3_5 (Qwen3.5-0.8B-4bit)", s2, r[2], sc.flagged, &s4);
    std::printf("%s\n", m.c_str());
    CHECK(m.find("NaN in qwen3_5 (Qwen3.5-0.8B-4bit), layer 3 attn_out, row 2, col 5 (of 8x64 f16), value nan") != std::string::npos);
    CHECK(m.find("pass 7") != std::string::npos);
    CHECK(m.find("1 later check also flagged (last: logits)") != std::string::npos);
    const std::string m4 = finite_message("m", s4, r[4], 1, nullptr);
    CHECK(m4.find("Inf in m, logits, row 0, col 0") != std::string::npos);
    FiniteRec big;
    big.hits = 1, big.idx = 1, big.val = bits(-3e11f);
    CHECK(finite_message("m", s4, big, 1, nullptr).find("outlier in m, logits, row 0, col 1 (of 1x1000 f32), value -3e+11") !=
          std::string::npos);
  }
  if (fails) {
    std::printf("finite_check_test: %d failures\n", fails);
    return 1;
  }
  std::printf("finite_check_test: ok\n");
  return 0;
}
