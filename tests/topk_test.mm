// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// GPU top-k candidates against the CPU sampler: for real readouts of a
// model, the distribution built from the GPU candidates must equal the one the
// CPU builds from the full logits (same ids in the same order, probabilities
// within float summation noise), for several sampling settings.
//   build/topk_test <model> [ctx=256]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../src/engine/sampling.h"
#include "../src/model/api.h"

using namespace espetero;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model> [ctx]\n", argv[0]);
    return 2;
  }
  const int ctx = argc > 2 ? std::atoi(argv[2]) : 256;
  LoadOptions o;
  o.slots = 4;
  o.max_ctx = 4096;
  o.kv_gb = 1;
  auto m = NativeModel::load(argv[1], o);
  const int V = m->info().vocab;
  std::mt19937 rng(5);
  std::vector<int32_t> root(static_cast<size_t>(ctx));
  for (auto& t : root) t = int32_t(rng() % 100000u);
  // Leaves give several readouts at different positions.
  std::vector<int32_t> lo = {0, 1, 3, 6}, lt = {11, 11, 222, 11, 222, 3333};
  struct Cfg { double t, top_p, min_p; int top_k; };
  const Cfg cfgs[] = {{1.0, 0.95, 0.0, 20}, {0.6, 0.95, 0.05, 20}, {1.5, 1.0, 0.0, 32}, {1.0, 1.0, 0.0, 1}, {0.7, 0.8, 0.0, 8}};
  int fails = 0, checked = 0;
  double worst = 0;
  for (const Cfg& c : cfgs) {
    const int s = m->slot_alloc();
    TreeSpec t;
    t.root = root.data();
    t.root_len = ctx;
    t.save = s;
    t.n_leaves = 3;
    t.leaf_offsets = lo.data();
    t.leaf_tokens = lt.data();
    t.score_root = true;
    t.want_logits = true;
    t.topk = c.top_k;
    t.temperature = float(c.t);
    int32_t arg[4];
    t.out_argmax = arg;
    m->run({t});
    SamplingParams sp;
    sp.temperature = c.t;
    sp.top_p = c.top_p;
    sp.min_p = c.min_p;
    sp.top_k = c.top_k;
    for (int r = 0; r < 4; ++r) {
      const Distribution a = distribution(m->logits(0, r), V, sp);
      const TopK g = m->topk(0, r);
      const Distribution b = distribution_topk(g.ids, g.x, g.n, g.max, g.total, sp);
      bool same = a.ids == b.ids;
      double d = 0;
      if (same)
        for (size_t i = 0; i < a.p.size(); ++i) d = std::max(d, std::fabs(a.p[i] - b.p[i]));
      worst = std::max(worst, d);
      ++checked;
      if (!same || d > 1e-5) {
        ++fails;
        std::printf("MISMATCH T=%.2f top_k=%d top_p=%.2f min_p=%.2f readout %d: %zu vs %zu candidates, max |dp| %.2e; first ids %d/%d\n",
                    c.t, c.top_k, c.top_p, c.min_p, r, a.ids.size(), b.ids.size(), d, a.ids.empty() ? -1 : a.ids[0],
                    b.ids.empty() ? -1 : b.ids[0]);
      }
    }
    m->slot_free(s);
  }
  // GPU draw against the CPU's exact draw over many positions: the two
  // use 24- and 53-bit uniforms, so only floating-point near-ties may differ.
  int agree = 0, draws = 0;
  {
    const int s2 = m->slot_alloc();
    SamplingParams sp;
    sp.temperature = 1.0;
    sp.top_p = 0.95;
    sp.top_k = 20;
    for (int pos = 0; pos < 300; ++pos) {
      TreeSpec t;
      t.root = root.data();
      t.root_len = ctx;
      t.save = s2;
      t.n_leaves = 3;
      t.leaf_offsets = lo.data();
      t.leaf_tokens = lt.data();
      t.score_root = true;
      t.want_logits = true;
      t.topk = sp.top_k;
      t.temperature = float(sp.temperature);
      t.gpu_draw = true;
      t.seed = 7;
      t.draw_pos0 = 1000 + pos * 10;
      t.top_p = float(sp.top_p);
      int32_t arg[4];
      t.out_argmax = arg;
      m->slot_reset(s2);
      m->run({t});
      for (int r = 0; r < 4; ++r) {
        const TopK g = m->topk(0, r);
        const int64_t rpos = t.draw_pos0 + (r < 3 ? int64_t(lo[size_t(r) + 1] - lo[size_t(r)]) : 0);
        const int32_t cpu = sample_at(distribution(m->logits(0, r), V, sp), 7, rpos);
        agree += g.sampled == cpu;
        ++draws;
      }
    }
    m->slot_free(s2);
  }
  std::printf("GPU draw vs CPU draw: %d/%d agree\n", agree, draws);
  if (agree < draws * 995 / 1000) ++fails;
  std::printf("%s: %d readouts checked, %d mismatches, max |dp| %.2e\n", fails ? "FAIL" : "OK", checked, fails, worst);
  return fails ? 1 : 0;
}
