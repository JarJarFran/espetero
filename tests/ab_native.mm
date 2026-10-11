// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// A/B timing of engine options on the same passes, alternating
// configurations every iteration so clock and thermal drift hit both.
//   build/ab_native <model> <option> <a> <b> [rows] [iters]
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../src/model/api.h"

using namespace espetero;

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <model> <option> <a> <b> [rows=2048] [iters=10] [decode_steps=0]\n", argv[0]);
    return 2;
  }
  const std::string opt = argv[2];
  const int a = std::atoi(argv[3]), b = std::atoi(argv[4]);
  const int rows = argc > 5 ? std::atoi(argv[5]) : 2048;
  const int iters = argc > 6 ? std::atoi(argv[6]) : 10;
  const int steps = argc > 7 ? std::atoi(argv[7]) : 0;
  LoadOptions o;
  o.slots = 8;
  o.max_ctx = 16384;
  o.kv_gb = 2;
  auto m = NativeModel::load(argv[1], o);
  std::mt19937 rng(3);
  std::vector<int32_t> toks(static_cast<size_t>(rows));
  for (auto& t : toks) t = int32_t(rng() % 100000u);
  const int s = m->slot_alloc();
  auto pass = [&]() {
    PassStats st;
    m->slot_reset(s);
    TreeSpec t;
    t.root = toks.data();
    t.root_len = rows;
    t.save = s;
    t.score_root = true;
    int32_t arg = 0;
    t.out_argmax = &arg;
    m->run({t}, &st);
    if (!steps) return st.gpu_ms;
    double g = 0;
    int32_t next = arg;
    for (int k = 0; k < steps; ++k) {
      TreeSpec d;
      d.parent = s;
      d.save = s;
      d.root = &next;
      d.root_len = 1;
      d.score_root = true;
      d.out_argmax = &arg;
      m->run({d}, &st);
      g += st.gpu_ms;
      next = arg;
    }
    return g / steps;
  };
  for (int w = 0; w < 3; ++w) pass();  // warm clocks
  std::vector<double> ta, tb;
  for (int i = 0; i < iters; ++i) {
    m->set_option(opt, a);
    ta.push_back(pass());
    m->set_option(opt, b);
    tb.push_back(pass());
  }
  std::sort(ta.begin(), ta.end());
  std::sort(tb.begin(), tb.end());
  std::printf("%s=%d: median %.2f ms (min %.2f)   %s=%d: median %.2f ms (min %.2f)   ratio %.3f\n", opt.c_str(), a,
              ta[ta.size() / 2], ta[0], opt.c_str(), b, tb[tb.size() / 2], tb[0], tb[tb.size() / 2] / ta[ta.size() / 2]);
  return 0;
}
