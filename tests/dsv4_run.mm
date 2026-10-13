// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// DeepSeek-V4.1 development driver: prefill a token list, then greedy decode.
//   build/dsv4_run MODEL_DIR 0,1234,... [new_tokens] [chunk]
// ESPETERO_DS4_DUMP=dir writes each layer's residual streams of the prefill.
#include <chrono>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "../src/model/api.h"

using namespace espetero;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s MODEL_DIR ids [new_tokens] [chunk]\n", argv[0]);
    return 2;
  }
  std::vector<int32_t> ids;
  {
    std::stringstream ss(argv[2]);
    std::string t;
    while (std::getline(ss, t, ',')) ids.push_back(int32_t(std::stoi(t)));
  }
  const int gen = argc > 3 ? std::atoi(argv[3]) : 8;
  const int chunk = argc > 4 ? std::atoi(argv[4]) : 0;
  // the last `tf` ids are fed one decode step at a time (teacher forcing) after the prefill
  const int tf = argc > 5 ? std::atoi(argv[5]) : 0;
  std::vector<int32_t> forced(ids.end() - tf, ids.end());
  ids.resize(ids.size() - size_t(tf));
  try {
    LoadOptions o;
    o.max_ctx = 16384;
    o.slots = 2;
    if (chunk > 0) o.max_rows = chunk;
    auto t0 = std::chrono::steady_clock::now();
    auto m = NativeModel::load(argv[1], o);
    auto ms = [&](auto a) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count(); };
    std::fprintf(stderr, "loaded in %.1f s\n", ms(t0) / 1e3);
    if (const char* b = std::getenv("DS4_RUN_BATCH")) {  // two prompts: separately, then as one pass of two trees
      std::vector<int32_t> ids2;
      std::stringstream s2(b);
      std::string t;
      while (std::getline(s2, t, ',')) ids2.push_back(int32_t(std::stoi(t)));
      auto one = [&](const std::vector<int32_t>& v, std::vector<float>& lg) {
        TreeSpec t1;
        t1.root = v.data(), t1.root_len = int(v.size()), t1.score_root = true, t1.want_logits = true;
        m->run({t1});
        lg.assign(m->logits(0, 0), m->logits(0, 0) + m->info().vocab);
      };
      std::vector<float> a, bb, a2, b2;
      one(ids, a);
      one(ids2, bb);
      TreeSpec x, y;
      x.root = ids.data(), x.root_len = int(ids.size()), x.score_root = true, x.want_logits = true;
      y.root = ids2.data(), y.root_len = int(ids2.size()), y.score_root = true, y.want_logits = true;
      m->run({x, y});
      a2.assign(m->logits(0, 0), m->logits(0, 0) + m->info().vocab);
      b2.assign(m->logits(1, 0), m->logits(1, 0) + m->info().vocab);
      auto cmp = [](const std::vector<float>& u, const std::vector<float>& v) {
        double d = 0, n = 0;
        int au = 0, av = 0;
        for (size_t i = 0; i < u.size(); ++i) {
          d += (u[i] - v[i]) * (u[i] - v[i]), n += v[i] * v[i];
          if (u[i] > u[size_t(au)]) au = int(i);
          if (v[i] > v[size_t(av)]) av = int(i);
        }
        std::printf("rel %.2e argmax %d %d\n", std::sqrt(d / n), au, av);
      };
      cmp(a2, a);
      cmp(b2, bb);
      return 0;
    }
    const int s = m->slot_alloc();
    TreeSpec t;
    t.parent = -1;
    t.save = s;
    t.root = ids.data();
    t.root_len = int(ids.size());
    t.score_root = true;
    int32_t arg = -1;
    t.out_argmax = &arg;
    t.topk = 5;
    t.temperature = 1.0f;
    auto t1 = std::chrono::steady_clock::now();
    m->run({t});
    const double pf = ms(t1);
    TopK k = m->topk(0, 0);
    std::printf("prefill %zu tokens: %.1f s (%.1f tok/s); argmax %d; top5:", ids.size(), pf / 1e3, ids.size() / (pf / 1e3), arg);
    for (int i = 0; i < k.n; ++i) std::printf(" %d(%.3f)", k.ids[i], k.x[i]);
    std::printf("\n%s\n", m->profile_json().c_str());
    std::fflush(stdout);
    for (int32_t f : forced) {
      TreeSpec d;
      d.parent = s;
      d.save = s;
      d.root = &f;
      d.root_len = 1;
      d.score_root = true;
      d.out_argmax = &arg;
      auto t2 = std::chrono::steady_clock::now();
      m->run({d});
      std::printf("forced %d -> %d  %.0f ms\n", f, arg, ms(t2));
    }
    std::vector<int32_t> out;
    for (int i = 0; i < gen; ++i) {
      TreeSpec d;
      d.parent = s;
      d.save = s;
      d.root = &arg;
      d.root_len = 1;
      d.score_root = true;
      d.out_argmax = &arg;
      m->set_option("expert_stats_reset", 1);
      auto t2 = std::chrono::steady_clock::now();
      const int32_t tok = arg;
      m->run({d});
      std::printf("step %d: token %d -> %d  %.0f ms  %s\n", i, tok, arg, ms(t2), m->profile_json().c_str());
      std::fflush(stdout);
      out.push_back(arg);
    }
    std::printf("generated:");
    for (int32_t x : out) std::printf(" %d", x);
    std::printf("\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
