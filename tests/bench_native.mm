// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Native benchmarks (no tokenizer): prefill throughput, single-sequence
// decode latency, batched decode throughput, a decision-shaped tree, and a
// per-kernel profile of a decode step and a prefill pass.
//
//   build/bench_native <model> [--quick] [--profile] [--gap] [--encoders] [--sustain] [--batch 1,4,16,64] [--ctx 128]
//   build/bench_native <model> --long 1024,8192,32768 [--long-ab OPTION V1,V2,...] [--long-seqs N] [--profile]
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../src/model/api.h"

using namespace espetero;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

static std::vector<int> parse_list(const char* s) {
  std::vector<int> v;
  while (*s) {
    v.push_back(std::atoi(s));
    while (*s && *s != ',') ++s;
    if (*s == ',') ++s;
  }
  return v;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model> [--quick] [--profile] [--gap] [--encoders] [--sustain] [--batch 1,4,16,64] [--ctx N]\n", argv[0]);
    return 2;
  }
  bool quick = false, profile = false, gap = false, encoders = false, sustain = false;
  std::vector<int> batches = {1, 4, 16, 64};
  int ctx = 128;
  const char* micro = nullptr;
  int chain = 0;  // --chain K: decode chains of K steps against plain steps (tokens and paired timing)
  const char* vab = nullptr;  // --verify-ab OPTION A B: paired MTP verify passes and plain steps under set_option A / B
  int vab_a = 0, vab_b = 1;
  const char* dab = nullptr;  // --draft-ab OPTION A B: paired MTP draft passes under set_option A / B
  int dab_a = 0, dab_b = 1;
  const char* pab = nullptr;  // --prefill-ab OPTION A B: paired prefills of --ctx rows under set_option A / B, and their logits
  int pab_a = 0, pab_b = 1;
  int slots = 72;
  int q4_steps = 0;       // --q4 N: one prefill of --ctx random tokens, then N greedy decode steps, timed,
  double xcache_gb = -1;  // with the expert cache counters; --xcache GB sets the expert cache size first
  const char* q4_ids = nullptr;  // --q4-ids FILE: {"ids": [...], "prompt_len": P}: prefill P ids, then feed the rest (teacher-forced)
  std::vector<int> longs;        // --long P1,P2,...: plain decode steps after P-token contexts (--profile: per kernel)
  const char* lab = nullptr;     // --long-ab OPTION V1,V2,...: those steps interleaved under set_option V1, V2, ...
  std::vector<int> lab_v;
  int max_ctx = 8192;            // --max-ctx N: tokens per sequence (the KV pool holds 4 GB, or 2 contexts of N)
  int long_seqs = 1;             // --long-seqs N: N sequences of the --long contexts decoded together
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--quick")) quick = true;
    else if (!std::strcmp(argv[i], "--profile")) profile = true;
    else if (!std::strcmp(argv[i], "--gap")) gap = true;
    else if (!std::strcmp(argv[i], "--encoders")) encoders = true;
    else if (!std::strcmp(argv[i], "--sustain")) sustain = true;
    else if (!std::strcmp(argv[i], "--batch") && i + 1 < argc) batches = parse_list(argv[++i]);
    else if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--micro") && i + 1 < argc) micro = argv[++i];
    else if (!std::strcmp(argv[i], "--chain") && i + 1 < argc) chain = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--verify-ab") && i + 3 < argc) vab = argv[++i], vab_a = std::atoi(argv[++i]), vab_b = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--draft-ab") && i + 3 < argc) dab = argv[++i], dab_a = std::atoi(argv[++i]), dab_b = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--prefill-ab") && i + 3 < argc) pab = argv[++i], pab_a = std::atoi(argv[++i]), pab_b = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--slots") && i + 1 < argc) slots = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--q4") && i + 1 < argc) q4_steps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--xcache") && i + 1 < argc) xcache_gb = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--q4-ids") && i + 1 < argc) q4_ids = argv[++i];
    else if (!std::strcmp(argv[i], "--long") && i + 1 < argc) longs = parse_list(argv[++i]);
    else if (!std::strcmp(argv[i], "--long-ab") && i + 2 < argc) lab = argv[++i], lab_v = parse_list(argv[++i]);
    else if (!std::strcmp(argv[i], "--max-ctx") && i + 1 < argc) max_ctx = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--long-seqs") && i + 1 < argc) long_seqs = std::max(1, std::atoi(argv[++i]));
  }
  for (int P : longs) max_ctx = std::max(max_ctx, P + 1024);
  // The verify-pass anatomy needs the diagnostic GEMV instantiations.
  if (micro && !std::strcmp(micro, "gemv")) setenv("ESPETERO_DIAG", "1", 0);
  LoadOptions o;
  o.slots = slots;
  if (xcache_gb != -1) o.expert_cache_gb = xcache_gb;  // --xcache: the expert cache is sized at load
  o.max_ctx = max_ctx;
  o.kv_gb = std::max(4.0, (long_seqs + 1.0) * max_ctx * 72e3 / 1e9);
  auto m = NativeModel::load(argv[1], o);
  const ModelInfo& I = m->info();
  std::printf("# %s  %s  layers %d  vocab %d  mtp %d  load %.0f ms (tune %.0f)\n", argv[1], I.device.c_str(), I.layers,
              I.vocab, int(I.mtp), I.load_ms, I.tune_ms);
  if (q4_steps > 0) {
    // Qwen4-Exp decode with streamed experts: random tokens, so routing
    // locality is lower than on real text (served benchmarks measure that).
    std::mt19937 g(11);
    std::vector<int32_t> prompt(size_t(std::max(ctx, 1)));
    for (auto& t : prompt) t = int32_t(g() % uint32_t(I.vocab));
    std::vector<int32_t> forced;  // teacher-forced decode tokens (--q4-ids): identical inputs across A/B runs
    if (q4_ids) {
      std::FILE* f = std::fopen(q4_ids, "rb");
      if (!f) { std::fprintf(stderr, "cannot open %s\n", q4_ids); return 2; }
      std::string txt;
      char buf[65536];
      size_t n;
      while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) txt.append(buf, n);
      std::fclose(f);
      std::vector<int32_t> all;
      int p0 = 0;
      const size_t a = txt.find('['), b = txt.find(']');
      for (size_t i = a + 1; i < b;) {
        while (i < b && (txt[i] == ' ' || txt[i] == ',')) ++i;
        if (i >= b) break;
        all.push_back(int32_t(std::strtol(txt.c_str() + i, nullptr, 10)));
        while (i < b && txt[i] != ',') ++i;
      }
      const size_t pl = txt.find("\"prompt_len\"");
      if (pl != std::string::npos) p0 = std::atoi(txt.c_str() + txt.find(':', pl) + 1);
      prompt.assign(all.begin(), all.begin() + std::min<size_t>(all.size(), size_t(p0)));
      forced.assign(all.begin() + std::min<size_t>(all.size(), size_t(p0)), all.end());
      ctx = int(prompt.size());
      q4_steps = std::min(q4_steps, int(forced.size()));
    }
    const int slot = m->slot_alloc();
    TreeSpec t;
    t.save = slot;
    t.root = prompt.data();
    t.root_len = int(prompt.size());
    t.score_root = true;
    int32_t next = 0;
    t.out_argmax = &next;
    PassStats st;
    const auto t0 = Clock::now();
    m->run({t}, &st);
    std::printf("q4 prefill %d tokens: %.1f ms wall (gpu %.1f)\nxcache prefill: %s\n", ctx, ms_since(t0), st.gpu_ms,
                m->bench_json("xcache_reset").c_str());
    std::vector<double> ms;
    if (!forced.empty()) {
      next = forced[0];
      q4_steps = std::min(q4_steps, int(forced.size()) - 1);
    }
    for (int j = 0; j < q4_steps; ++j) {
      TreeSpec d;
      d.parent = slot;
      d.save = slot;
      d.root = &next;
      d.root_len = 1;
      d.score_root = true;
      int32_t out = 0;
      d.out_argmax = &out;
      const auto t1 = Clock::now();
      m->run({d});
      ms.push_back(ms_since(t1));
      next = forced.empty() ? out : forced[size_t(j) + 1];
    }
    std::vector<double> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0;
    for (double x : ms) sum += x;
    std::printf("q4 decode %d steps: median %.1f ms, mean %.1f ms (%.2f tok/s), first %.1f ms\nxcache decode: %s\n", q4_steps,
                sorted[sorted.size() / 2], sum / double(ms.size()), 1000.0 * double(ms.size()) / sum, ms[0],
                m->bench_json("xcache").c_str());
    if (profile) {  // one more decode step, every dispatch timed on its own
      m->set_profile(true);
      TreeSpec d;
      d.parent = slot;
      d.save = slot;
      d.root = &next;
      d.root_len = 1;
      d.score_root = true;
      int32_t out = 0;
      d.out_argmax = &out;
      m->run({d});
      std::printf("q4 decode profile: %s\n", m->profile_json().c_str());
      m->set_profile(false);
    }
    return 0;
  }
  if (micro && !std::strcmp(micro, "vision")) {
    // Vision tower alone: photos at the default chat budget (1 MP) and at 2 MP;
    // pixel values do not matter for timing.
    if (!I.vision) {
      std::printf("no vision tower\n");
      return 0;
    }
    std::printf("\n## micro vision\n");
    for (auto [gh, gw] : {std::pair<int, int>{54, 72}, std::pair<int, int>{76, 102}}) {
      const int n_patch = gh * gw;
      std::vector<float> pix(size_t(n_patch) * 1536, 0.1f), out(size_t(n_patch / 4) * 8192);
      // ESPETERO_VIT_DUMP=path: seeded noise pixels, the first image's embeddings written there (kernel checks).
      const char* dump = std::getenv("ESPETERO_VIT_DUMP");
      if (dump) {
        std::mt19937 g(7);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (auto& x : pix) x = u(g);
      }
      ImageRef r;
      r.pixels = pix.data();
      r.grid_h = gh;
      r.grid_w = gw;
      double best = 1e30, wall = 0;
      for (int it = 0; it < 4; ++it) {
        PassStats st;
        m->encode_images({r}, out.data(), &st);
        if (it > 0) best = std::min(best, st.gpu_ms), wall = st.wall_ms;
      }
      std::printf("vision %dx%d patches (%d tokens): %.1f ms gpu, %.1f ms wall\n", gh, gw, n_patch / 4, best, wall);
      if (dump && gh == 54) {
        FILE* f = std::fopen(dump, "wb");
        if (f) {
          std::fwrite(out.data(), 4, size_t(n_patch / 4) * size_t(I.hidden), f);
          std::fclose(f);
        }
      }
      if (profile) {
        m->set_profile(true);
        m->encode_images({r}, out.data(), nullptr);
        std::printf("vision_profile: %s\n", m->profile_json().c_str());
        m->set_profile(false);
      }
    }
    return 0;
  }
  if (micro) {
    std::printf("\n## micro %s\n%s: %s\n", micro, micro, m->bench_json(micro).c_str());
    return 0;
  }
  if (vab) {
    // Paired A/B under set_option(vab, A/B), alternating every iteration: a
    // plain decode step and an MTP verify pass (1 pending + 3 draft leaves).
    std::mt19937 rng3(13);
    std::vector<int32_t> r(static_cast<size_t>(ctx));
    for (auto& t : r) t = int32_t(rng3() % 100000u);
    const int s = m->slot_alloc();
    {
      TreeSpec t;
      t.root = r.data();
      t.root_len = ctx;
      t.save = s;
      t.mtp = I.mtp;
      m->run({t});
    }
    int32_t a = 11, ar[4];
    std::vector<int32_t> lo = {0, 1, 3, 6}, lt = {101, 101, 202, 101, 202, 303};
    auto verify = [&]() {
      TreeSpec v;
      v.parent = v.save = s;  // the root token is re-saved: the context stays the same length class
      v.root = &a;
      v.root_len = 1;
      v.n_leaves = 3;
      v.leaf_offsets = lo.data();
      v.leaf_tokens = lt.data();
      v.score_root = true;
      v.out_argmax = ar;
      v.mtp = I.mtp;
      PassStats st;
      m->run({v}, &st);
      return st.gpu_ms;
    };
    auto plain = [&]() {
      TreeSpec v;
      v.parent = v.save = s;
      v.root = &a;
      v.root_len = 1;
      v.score_root = true;
      v.out_argmax = ar;
      v.mtp = I.mtp;
      PassStats st;
      m->run({v}, &st);
      return st.gpu_ms;
    };
    for (int w = 0; w < 3; ++w) verify(), plain();
    std::vector<double> va, vb, pa, pb;
    for (int k = 0; k < 16; ++k) {
      m->set_option(vab, vab_a);
      va.push_back(verify());
      pa.push_back(plain());
      m->set_option(vab, vab_b);
      vb.push_back(verify());
      pb.push_back(plain());
    }
    for (auto* x : {&va, &vb, &pa, &pb}) std::sort(x->begin(), x->end());
    std::printf("\n## verify A/B %s\nverify: %s=%d %.2f ms  %s=%d %.2f ms  ratio %.3f\nplain:  %s=%d %.2f ms  %s=%d %.2f ms  ratio %.3f\n", vab,
                vab, vab_a, va[8], vab, vab_b, vb[8], vb[8] / va[8], vab, vab_a, pa[8], vab, vab_b, pb[8], pb[8] / pa[8]);
    return 0;
  }
  if (pab) {
    // Paired prefills of the same --ctx random tokens under set_option(pab,
    // A / B), alternating; the root's logits of the first pair are compared.
    std::mt19937 rng4(29);
    std::vector<int32_t> r(static_cast<size_t>(ctx));
    for (auto& t : r) t = int32_t(rng4() % 100000u);
    auto prefill = [&](std::vector<float>* keep, int32_t* arg) {
      const int s = m->slot_alloc();
      TreeSpec t;
      t.root = r.data();
      t.root_len = ctx;
      t.save = s;
      t.score_root = true;
      t.want_logits = keep != nullptr;
      t.out_argmax = arg;
      PassStats st;
      m->run({t}, &st);
      if (keep) {
        const float* L = m->logits(0, 0);
        keep->assign(L, L + I.vocab);
      }
      m->slot_free(s);
      return st.gpu_ms;
    };
    int32_t xa = 0, xb = 0;
    prefill(nullptr, &xa);
    std::vector<double> ta, tb;
    std::vector<float> la, lb;
    for (int k = 0; k < 6; ++k) {
      m->set_option(pab, pab_a);
      ta.push_back(prefill(k == 0 ? &la : nullptr, &xa));
      m->set_option(pab, pab_b);
      tb.push_back(prefill(k == 0 ? &lb : nullptr, &xb));
    }
    std::sort(ta.begin(), ta.end());
    std::sort(tb.begin(), tb.end());
    double md = 0;
    for (size_t i = 0; i < la.size() && i < lb.size(); ++i) md = std::max(md, double(std::fabs(la[i] - lb[i])));
    std::printf("\n## prefill A/B %s (%d rows)\nprefill: %s=%d %.1f ms  %s=%d %.1f ms  ratio %.3f\nlogits: max |diff| %.4g  argmax %d vs %d\n",
                pab, ctx, pab, pab_a, ta[3], pab, pab_b, tb[3], tb[3] / ta[3], md, xa, xb);
    return 0;
  }
  if (dab && I.mtp) {
    // Paired A/B under set_option(dab, A/B), alternating every iteration: MTP
    // draft passes of 1 and 5 tokens from one saved state, fed a different
    // token each iteration. Also the share of first drafts (and of whole
    // 5-token chains) that A and B agree on (random context: a lower bound
    // of the agreement on text).
    std::mt19937 rng3(17);
    std::vector<int32_t> r(static_cast<size_t>(ctx));
    for (auto& t : r) t = int32_t(rng3() % 100000u);
    const int s = m->slot_alloc();
    int32_t a0 = 0;
    {
      TreeSpec t;
      t.root = r.data();
      t.root_len = ctx;
      t.save = s;
      t.mtp = true;
      t.score_root = true;
      t.out_argmax = &a0;
      m->run({t});
    }
    auto draft = [&](int32_t tok, int n, int32_t* out) {
      DraftRequest q;
      q.slot = s;
      q.tokens = &tok;
      q.n_tokens = 1;
      q.n_draft = n;
      q.out = out;
      PassStats st;
      m->draft({q}, &st);
      return st.gpu_ms;
    };
    int32_t da[8], db[8];
    for (int w = 0; w < 3; ++w) draft(a0, 5, da);
    std::vector<double> a1, b1, a5, b5;
    int first = 0, chains = 0, n = 0;
    for (int k = 0; k < 48; ++k) {
      const int32_t tok = int32_t(rng3() % 100000u);
      m->set_option(dab, dab_a);
      a1.push_back(draft(tok, 1, da));
      a5.push_back(draft(tok, 5, da));
      m->set_option(dab, dab_b);
      b1.push_back(draft(tok, 1, db));
      b5.push_back(draft(tok, 5, db));
      first += da[0] == db[0];
      chains += std::equal(da, da + 5, db);
      ++n;
    }
    for (auto* x : {&a1, &b1, &a5, &b5}) std::sort(x->begin(), x->end());
    const size_t h = a1.size() / 2;
    std::printf("\n## draft A/B %s\ndraft 1: %s=%d %.3f ms  %s=%d %.3f ms  ratio %.3f\ndraft 5: %s=%d %.3f ms  %s=%d %.3f ms  ratio %.3f\n"
                "agreement: first draft %d/%d, 5-token chain %d/%d\n",
                dab, dab, dab_a, a1[h], dab, dab_b, b1[h], b1[h] / a1[h], dab, dab_a, a5[h], dab, dab_b, b5[h], b5[h] / a5[h],
                first, n, chains, n);
    m->slot_free(s);
    return 0;
  }
  if (chain > 1) {
    // (1) Tokens: from one prefilled context forked into two slots, S plain
    // greedy steps (argmax fed back on the CPU) and S chained steps must agree.
    // (2) Timing: rounds alternate S plain steps and S chained steps (chains of
    // K) on 1 and 4 sequences; medians of the per-step wall time.
    std::mt19937 rng2(11);
    auto rtoks = [&](int n) {
      std::vector<int32_t> v(static_cast<size_t>(n));
      for (auto& t : v) t = int32_t(rng2() % 100000u);
      return v;
    };
    const int S = 48;
    auto prefill = [&](int slot) {
      auto r = rtoks(ctx);
      TreeSpec t;
      t.root = r.data();
      t.root_len = ctx;
      t.save = slot;
      t.score_root = true;
      int32_t a0 = 0;
      t.out_argmax = &a0;
      m->run({t});
      return a0;
    };
    auto plain = [&](const std::vector<int>& slots, std::vector<int32_t> next, int steps, std::vector<int32_t>* toks) {
      const size_t n = slots.size();
      std::vector<int32_t> arg(n, 0);
      const auto t0 = Clock::now();
      for (int k = 0; k < steps; ++k) {
        std::vector<TreeSpec> ts;
        for (size_t i = 0; i < n; ++i) {
          TreeSpec t;
          t.parent = t.save = slots[i];
          t.root = &next[i];
          t.root_len = 1;
          t.score_root = true;
          t.out_argmax = &arg[i];
          ts.push_back(t);
        }
        m->run(ts);
        for (size_t i = 0; i < n; ++i) {
          next[i] = arg[i];
          if (toks) toks->push_back(arg[i]);
        }
      }
      return ms_since(t0) / steps;
    };
    auto chained = [&](const std::vector<int>& slots, std::vector<int32_t> next, int steps, int K, std::vector<int32_t>* toks) {
      const size_t n = slots.size();
      std::vector<int32_t> out(size_t(K) * n);
      const auto t0 = Clock::now();
      for (int k = 0; k < steps; k += K) {
        const int len = std::min(K, steps - k);
        std::vector<TreeSpec> ts;
        for (size_t i = 0; i < n; ++i) {
          TreeSpec t;
          t.parent = t.save = slots[i];
          t.root = &next[i];
          t.root_len = 1;
          t.score_root = true;
          ts.push_back(t);
        }
        m->run_chain(ts, len, out.data());
        for (int j = 0; j < len; ++j)
          for (size_t i = 0; i < n; ++i)
            if (toks) toks->push_back(out[size_t(j) * n + i]);
        for (size_t i = 0; i < n; ++i) next[i] = out[size_t(len - 1) * n + i];
      }
      return ms_since(t0) / steps;
    };
    {
      const int a = m->slot_alloc(), b = m->slot_alloc();
      const int32_t first = prefill(a);
      m->slot_fork(b, a);
      std::vector<int32_t> ta, tb;
      plain({a}, {first}, S, &ta);
      chained({b}, {first}, S, chain, &tb);
      int same = 0;
      while (same < S && ta[size_t(same)] == tb[size_t(same)]) ++same;
      std::printf("\n## decode chains (K = %d)\ntokens: %d/%d identical to plain greedy decode%s\n", chain, same, S,
                  same == S ? "" : "  MISMATCH");
      m->slot_free(a);
      m->slot_free(b);
    }
    {
      // Sampled: the GPU draws each token (T 1, top-k 20, top-p 0.95,
      // seed 7); chained steps must give the tokens of plain steps.
      const int a = m->slot_alloc(), b = m->slot_alloc();
      const int32_t first = prefill(a);
      m->slot_fork(b, a);
      auto fill = [&](TreeSpec& t, int64_t pos) {
        t.topk = 20;
        t.temperature = 1.0f;
        t.gpu_draw = true;
        t.seed = 7;
        t.top_p = 0.95f;
        t.draw_pos0 = pos;
      };
      std::vector<int32_t> ta, tb;
      int32_t next = first, arg = 0;
      for (int k = 0; k < S; ++k) {
        TreeSpec t;
        t.parent = t.save = a;
        t.root = &next;
        t.root_len = 1;
        t.score_root = true;
        t.out_argmax = &arg;
        fill(t, ctx + 1 + k);
        m->run({t});
        next = m->topk(0, 0).sampled;
        ta.push_back(next);
      }
      next = first;
      std::vector<int32_t> out(static_cast<size_t>(chain));
      for (int k = 0; k < S; k += chain) {
        const int len = std::min(chain, S - k);
        TreeSpec t;
        t.parent = t.save = b;
        t.root = &next;
        t.root_len = 1;
        t.score_root = true;
        fill(t, ctx + 1 + k);
        m->run_chain({t}, len, out.data());
        for (int j = 0; j < len; ++j) tb.push_back(out[size_t(j)]);
        next = out[size_t(len - 1)];
      }
      int same = 0, diff_argmax = 0;
      while (same < S && ta[size_t(same)] == tb[size_t(same)]) ++same;
      std::printf("sampled tokens: %d/%d identical to plain sampled decode%s\n", same, S, same == S ? "" : "  MISMATCH");
      (void)diff_argmax;
      m->slot_free(a);
      m->slot_free(b);
    }
    for (int nseq : {1, 4}) {
      std::vector<int> sa, sb;
      std::vector<int32_t> fa;
      for (int i = 0; i < nseq; ++i) {
        const int a = m->slot_alloc(), b = m->slot_alloc();
        fa.push_back(prefill(a));
        m->slot_fork(b, a);
        sa.push_back(a);
        sb.push_back(b);
      }
      plain(sa, fa, 8, nullptr);  // warm
      chained(sb, fa, 8, chain, nullptr);
      std::vector<double> wp, wc;
      for (int r = 0; r < 5; ++r) {
        wp.push_back(plain(sa, fa, S, nullptr));
        wc.push_back(chained(sb, fa, S, chain, nullptr));
      }
      std::sort(wp.begin(), wp.end());
      std::sort(wc.begin(), wc.end());
      std::printf("seqs %d: plain %.2f ms/step (min %.2f)  chain %.2f ms/step (min %.2f)  ratio %.3f\n", nseq, wp[2], wp[0],
                  wc[2], wc[0], wc[2] / wp[2]);
      for (int x : sa) m->slot_free(x);
      for (int x : sb) m->slot_free(x);
    }
    return 0;
  }
  std::mt19937 rng(7);
  auto toks = [&](int n) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    for (auto& t : v) t = int32_t(rng() % 100000u);
    return v;
  };
  // Warm the GPU clock (it ramps over the first tens of ms of work).
  {
    const int s = m->slot_alloc();
    auto r = toks(512);
    TreeSpec t;
    t.root = r.data();
    t.root_len = 512;
    t.save = s;
    for (int i = 0; i < 3; ++i) m->run({t});
    m->slot_free(s);
  }

  if (!longs.empty()) {
    // Decode at long contexts: per context P, P random tokens prefilled in
    // passes of 2048 rows, then plain decode steps (medians of the GPU and
    // wall time per step); --long-ab interleaves the steps under set_option
    // V1, V2, ...; --profile adds one step per value with every dispatch
    // timed on its own.
    std::printf("\n## long-context decode (%d sequence%s, greedy, plain)\n", long_seqs, long_seqs > 1 ? "s" : "");
    if (lab_v.empty()) lab_v.push_back(INT32_MIN);  // no option: one variant
    const size_t nv = lab_v.size();
    auto set_v = [&](size_t v) {
      if (lab && lab_v[v] != INT32_MIN) m->set_option(lab, lab_v[v]);
    };
    for (int P : longs) {
      std::vector<int> slots;
      for (int q = 0; q < long_seqs; ++q) {
        const int s = m->slot_alloc();
        slots.push_back(s);
        auto r = toks(P);
        for (int at = 0; at < P;) {
          const int n = std::min({2048, I.max_rows, P - at});
          TreeSpec t;
          t.parent = at > 0 ? s : -1;
          t.save = s;
          t.root = r.data() + at;
          t.root_len = n;
          m->run({t});
          at += n;
        }
      }
      const int s = slots[0];
      std::vector<int32_t> tok(slots.size(), 11), arg(slots.size(), 0);
      auto step = [&](double* wall) {
        std::vector<TreeSpec> ts;
        for (size_t q = 0; q < slots.size(); ++q) {
          TreeSpec d;
          d.parent = d.save = slots[q];
          d.root = &tok[q];
          d.root_len = 1;
          d.score_root = true;
          d.out_argmax = &arg[q];
          ts.push_back(d);
        }
        PassStats st;
        const auto t0 = Clock::now();
        m->run(ts, &st);
        *wall = ms_since(t0);
        for (size_t q = 0; q < slots.size(); ++q) tok[q] = arg[q] % 100000;
        return st.gpu_ms;
      };
      auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[v.size() / 2];
      };
      double w;
      for (int k = 0; k < 4; ++k) step(&w);
      std::vector<std::vector<double>> gs(nv), ws(nv);
      for (int k = 0; k < 16; ++k)
        for (size_t v = 0; v < nv; ++v) {
          set_v(v);
          gs[v].push_back(step(&w));
          ws[v].push_back(w);
        }
      std::printf("P %6d:", P);
      for (size_t v = 0; v < nv; ++v) {
        if (lab && lab_v[v] != INT32_MIN) std::printf("  %s=%d", lab, lab_v[v]);
        std::printf(" gpu %.2f wall %.2f ms (%.2f tok/s)", med(gs[v]), med(ws[v]), 1000.0 * long_seqs / med(ws[v]));
      }
      std::printf("\n");
      if (nv > 1) {
        // The same decode step (not saved) from this context under each value
        // against the first: max |delta logit|, max |delta log-prob| and argmax
        // agreement over 8 input tokens.
        std::vector<double> dl(nv, 0.0), dp(nv, 0.0);
        std::vector<int> same(nv, 0);
        for (int q = 0; q < 8; ++q) {
          int32_t in = int32_t((q * 7919 + 13) % 100000), am = 0;
          std::vector<float> ref;
          double ref_lse = 0;
          for (size_t v = 0; v < nv; ++v) {
            set_v(v);
            TreeSpec d;
            d.parent = s;
            d.root = &in;
            d.root_len = 1;
            d.score_root = true;
            d.want_logits = true;
            d.out_argmax = &am;
            m->run({d});
            const float* L = m->logits(0, 0);
            double mx = -1e30, se = 0;
            for (int i = 0; i < I.vocab; ++i) mx = std::max(mx, double(L[i]));
            for (int i = 0; i < I.vocab; ++i) se += std::exp(double(L[i]) - mx);
            const double lse = mx + std::log(se);
            if (v == 0) {
              ref.assign(L, L + I.vocab);
              ref_lse = lse;
              continue;
            }
            int a0 = 0, a1 = 0;
            for (int i = 0; i < I.vocab; ++i) {
              dl[v] = std::max(dl[v], double(std::fabs(L[i] - ref[size_t(i)])));
              dp[v] = std::max(dp[v], std::fabs((double(L[i]) - lse) - (double(ref[size_t(i)]) - ref_lse)));
              if (ref[size_t(i)] > ref[size_t(a0)]) a0 = i;
              if (L[i] > L[a1]) a1 = i;
            }
            same[v] += a0 == a1;
          }
        }
        for (size_t v = 1; v < nv; ++v)
          std::printf("  logits %s=%d vs %d: max |dlogit| %.4g, max |dlogprob| %.4g, argmax equal %d/8\n", lab, lab_v[v], lab_v[0],
                      dl[v], dp[v], same[v]);
      }
      if (profile)
        for (size_t v = 0; v < nv; ++v) {
          set_v(v);
          m->set_profile(true);
          step(&w);
          std::printf("profile P %d %s=%d: %s\n", P, lab ? lab : "-", lab_v[v], m->profile_json().c_str());
          m->set_profile(false);
        }
      set_v(0);
      for (int q : slots) m->slot_free(q);
    }
    return 0;
  }

  // ---- prefill
  std::printf("\n## prefill (one sequence, fresh)\n rows   gpu ms   wall ms   tokens/s\n");
  for (int n : quick ? std::vector<int>{512, 2048} : std::vector<int>{128, 512, 1024, 2048, 4096}) {
    if (n > I.max_rows) continue;
    const int s = m->slot_alloc();
    auto r = toks(n);
    TreeSpec t;
    t.root = r.data();
    t.root_len = n;
    t.save = s;
    t.score_root = true;
    int32_t arg = 0;
    t.out_argmax = &arg;
    PassStats st;
    double best = 1e9, wall = 1e9;
    for (int rep = 0; rep < 3; ++rep) {
      m->slot_reset(s);
      m->run({t}, &st);
      best = std::min(best, st.gpu_ms);
      wall = std::min(wall, st.wall_ms);
    }
    std::printf("%5d %8.1f %9.1f %10.0f\n", n, best, wall, n / (wall / 1000));
    m->slot_free(s);
  }

  // ---- single-sequence decode (plain)
  auto decode_run = [&](int nseq, int steps, double* gpu_per, double* wall_per, double* enc_per, int* disp, int* barr) {
    std::vector<int> slots;
    for (int i = 0; i < nseq; ++i) {
      const int s = m->slot_alloc();
      slots.push_back(s);
      auto r = toks(ctx);
      TreeSpec t;
      t.root = r.data();
      t.root_len = ctx;
      t.save = s;
      m->run({t});
    }
    std::vector<int32_t> next(size_t(nseq), 1);
    std::vector<int32_t> arg(size_t(nseq), 0);
    double g = 0, w = 0, e = 0;
    PassStats st;
    for (int k = 0; k < steps + 2; ++k) {
      std::vector<TreeSpec> ts;
      for (int i = 0; i < nseq; ++i) {
        TreeSpec t;
        t.parent = slots[size_t(i)];
        t.save = slots[size_t(i)];
        t.root = &next[size_t(i)];
        t.root_len = 1;
        t.score_root = true;
        t.out_argmax = &arg[size_t(i)];
        ts.push_back(t);
      }
      const auto t0 = Clock::now();
      m->run(ts, &st);
      const double wall = ms_since(t0);
      for (int i = 0; i < nseq; ++i) next[size_t(i)] = arg[size_t(i)] % 100000;
      if (k >= 2) {
        g += st.gpu_ms;
        w += wall;
        e += st.encode_ms;
      }
    }
    *gpu_per = g / steps;
    *wall_per = w / steps;
    *enc_per = e / steps;
    *disp = st.dispatches;
    *barr = st.barriers;
    for (int s : slots) m->slot_free(s);
  };
  std::printf("\n## decode (context %d, greedy, plain; one forest pass per step for all sequences)\n", ctx);
  std::printf(" seqs  gpu ms/step  wall ms/step  encode ms  dispatches barriers  tokens/s\n");
  for (int b : batches) {
    if (b > 70) continue;
    double g, w, e;
    int d, br;
    decode_run(b, quick ? 16 : 48, &g, &w, &e, &d, &br);
    std::printf("%5d %12.2f %13.2f %10.2f %11d %8d %9.0f\n", b, g, w, e, d, br, b / (w / 1000));
  }

  // ---- encoder modes on the same decode steps
  if (encoders) {
    std::printf("\n## encoder modes (1 sequence, context %d, %d steps)\n mode                              gpu ms/step  wall ms/step  barriers\n", ctx, quick ? 16 : 48);
    struct Mode { const char* name; int serial, ball; } modes[] = {
        {"serial", 1, 0}, {"concurrent, tracked barriers", 0, 0}, {"concurrent, barrier before all", 0, 1}};
    for (const Mode& md : modes) {
      m->set_option("serial", md.serial);
      m->set_option("barrier_all", md.ball);
      double g, w, e;
      int d, br;
      decode_run(1, quick ? 16 : 48, &g, &w, &e, &d, &br);
      std::printf(" %-32s %12.2f %13.2f %9d\n", md.name, g, w, br);
    }
    m->set_option("serial", 1);
    m->set_option("barrier_all", 0);
  }
  if (gap) std::printf("\n## dispatch gap\ngap: %s\n", m->bench_json("gap").c_str());

  // ---- sustained decode: per-step GPU time over 64 back-to-back steps (power/thermal drift)
  if (sustain) {
    const int steps = 64;
    const int s = m->slot_alloc();
    auto r = toks(ctx);
    TreeSpec t;
    t.root = r.data();
    t.root_len = ctx;
    t.save = s;
    m->run({t});
    int32_t next = 1, arg = 0;
    std::vector<double> gpu;
    PassStats st;
    for (int k = 0; k < steps; ++k) {
      TreeSpec d;
      d.parent = s;
      d.save = s;
      d.root = &next;
      d.root_len = 1;
      d.score_root = true;
      d.out_argmax = &arg;
      m->run({d}, &st);
      next = arg % 100000;
      gpu.push_back(st.gpu_ms);
    }
    m->slot_free(s);
    std::printf("\n## sustained decode (1 sequence, context %d, %d back-to-back steps)\nsteps:", ctx, steps);
    for (double g : gpu) std::printf(" %.1f", g);
    double a = 0, b = 0, mn = 1e9;
    for (int k = 0; k < 8; ++k) a += gpu[size_t(k)];
    for (int k = steps - 8; k < steps; ++k) b += gpu[size_t(k)];
    for (double g : gpu) mn = std::min(mn, g);
    std::printf("\nsustain: first8 %.2f ms  last8 %.2f ms  min %.2f ms  drift %.3f\n", a / 8, b / 8, mn, (b / 8) / (a / 8));
  }

  // ---- MTP drafting cost
  if (I.mtp) {
    const int s = m->slot_alloc();
    auto r = toks(ctx);
    TreeSpec t;
    t.root = r.data();
    t.root_len = ctx;
    t.save = s;
    t.mtp = true;
    t.score_root = true;
    int32_t a = 0;
    t.out_argmax = &a;
    m->run({t});
    int32_t dr[3];
    DraftRequest q;
    q.slot = s;
    q.tokens = &a;
    q.n_tokens = 1;
    q.n_draft = 3;
    q.out = dr;
    PassStats st;
    double g = 0, w = 0;
    for (int k = 0; k < 12; ++k) {
      m->draft({q}, &st);
      if (k >= 2) g += st.gpu_ms, w += st.wall_ms;
    }
    std::printf("\n## MTP draft (3 tokens): %.2f ms gpu, %.2f ms wall\n", g / 10, w / 10);
    if (profile) {
      // Per-kernel profile of one 5-token draft pass.
      int32_t d5[5];
      DraftRequest q5 = q;
      q5.n_draft = 5;
      q5.out = d5;
      m->draft({q5}, &st);
      m->set_profile(true);
      m->draft({q5}, &st);
      std::printf("draft5: %s\n", m->profile_json().c_str());
      m->set_profile(false);
    }
    // verification pass: 1 pending + 3 draft leaves
    std::vector<int32_t> lo = {0, 1, 3, 6}, lt = {dr[0], dr[0], dr[1], dr[0], dr[1], dr[2]};
    int32_t ar[4];
    g = w = 0;
    for (int k = 0; k < 12; ++k) {
      TreeSpec v;
      v.parent = s;
      v.save = s;
      v.root = &a;
      v.root_len = 1;
      v.n_leaves = 3;
      v.leaf_offsets = lo.data();
      v.leaf_tokens = lt.data();
      v.score_root = true;
      v.out_argmax = ar;
      v.mtp = true;
      m->run({v}, &st);
      if (k >= 2) g += st.gpu_ms, w += st.wall_ms;
    }
    std::printf("## verify pass (1 + 3 leaves): %.2f ms gpu, %.2f ms wall\n", g / 10, w / 10);
    m->slot_free(s);
  }

  // ---- decision-shaped tree: 300-token prefix, 12 leaves of 20 tokens, 4 candidates
  {
    auto root = toks(300);
    std::vector<int32_t> lo{0}, lt, co{0}, ci;
    for (int i = 0; i < 12; ++i) {
      auto l = toks(20);
      lt.insert(lt.end(), l.begin(), l.end());
      lo.push_back(int32_t(lt.size()));
      for (int c = 0; c < 4; ++c) ci.push_back(int32_t(rng() % 1000));
      co.push_back(int32_t(ci.size()));
    }
    std::vector<float> cl(ci.size()), lse(12);
    TreeSpec t;
    t.root = root.data();
    t.root_len = 300;
    t.n_leaves = 12;
    t.leaf_offsets = lo.data();
    t.leaf_tokens = lt.data();
    t.cand_offsets = co.data();
    t.cand_ids = ci.data();
    t.out_cand_logits = cl.data();
    t.out_lse = lse.data();
    PassStats st;
    double g = 1e9, w = 1e9;
    for (int k = 0; k < 5; ++k) {
      m->run({t}, &st);
      g = std::min(g, st.gpu_ms);
      w = std::min(w, st.wall_ms);
    }
    std::printf("\n## decision tree (300-token prefix, 12 fields x 20 tokens): %.1f ms gpu, %.1f ms wall, %d dispatches, %d barriers\n",
                g, w, st.dispatches, st.barriers);
    if (profile) {
      m->set_profile(true);
      m->run({t}, &st);
      std::printf("tree: %s\n", m->profile_json().c_str());
      m->set_profile(false);
    }
  }

  if (profile) {
    // Batched decode profile: 16 sequences, one step.
    {
      std::vector<int> ss;
      const int PB = std::getenv("PROFILE_B") ? std::atoi(std::getenv("PROFILE_B")) : 16;
      for (int i = 0; i < PB; ++i) {
        const int sl = m->slot_alloc();
        ss.push_back(sl);
        auto r = toks(ctx);
        TreeSpec t;
        t.root = r.data();
        t.root_len = ctx;
        t.save = sl;
        m->run({t});
      }
      std::vector<int32_t> nx(static_cast<size_t>(PB), 7), ar(static_cast<size_t>(PB), 0);
      std::vector<TreeSpec> ts;
      for (int i = 0; i < PB; ++i) {
        TreeSpec d;
        d.parent = ss[size_t(i)];
        d.save = ss[size_t(i)];
        d.root = &nx[size_t(i)];
        d.root_len = 1;
        d.score_root = true;
        d.out_argmax = &ar[size_t(i)];
        ts.push_back(d);
      }
      m->run(ts);
      m->set_profile(true);
      for (int k = 0; k < 3; ++k) m->run(ts);
      std::printf("decode16: %s\n", m->profile_json().c_str());
      m->set_profile(false);
      for (int sl : ss) m->slot_free(sl);
    }
    std::printf("\n## profile: one decode step (1 sequence), then a 2048-row prefill\n");
    const int s = m->slot_alloc();
    auto r = toks(ctx);
    TreeSpec t;
    t.root = r.data();
    t.root_len = ctx;
    t.save = s;
    m->run({t});
    int32_t one = 5, arg = 0;
    TreeSpec d;
    d.parent = s;
    d.save = s;
    d.root = &one;
    d.root_len = 1;
    d.score_root = true;
    d.out_argmax = &arg;
    m->set_profile(true);
    for (int k = 0; k < 4; ++k) m->run({d});
    std::printf("decode: %s\n", m->profile_json().c_str());
    m->set_profile(false);
    auto p = toks(2048);
    TreeSpec pf;
    pf.root = p.data();
    pf.root_len = 2048;
    pf.save = s;
    m->slot_reset(s);
    m->set_profile(true);
    m->run({pf});
    std::printf("prefill: %s\n", m->profile_json().c_str());
    m->set_profile(false);
    m->slot_free(s);
  }
  return 0;
}
