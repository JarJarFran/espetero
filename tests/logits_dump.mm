// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Teacher-forced logits and greedy generations of the native engine for given
// token ids, in every pass shape the engine runs (Gemma 4 parity against
// mlx-lm; any model works):
//
//   build/logits_dump <model> <cases.json> <out.json> [--top 64] [--kv-gb 4] [--max-ctx 8192]
//
// cases.json: {"cases": [{"ids": [...], "prompt_len": P, "gen": G}, ...]}.
// Per case, every readout row is reported as its argmax, log-sum-exp and the
// top `--top` (id, logit) pairs:
//   prefill: one pass over ids[:P], the root's last row;
//   tree:    root ids[:P] with leaves ids[P:P+j] (j = 1..), rows P-1.. (one pass, the leaves share rows);
//   decode:  in-place single-token passes over ids[P:], after the prefill;
//   verify:  after the prefill, ids[P] as the root and ids[P+1:] as a chain of leaves over the
//            cached prompt (a draft verification), rows P..;
//   split:   ids[:P] saved at P/2 (save_len) and whole (work slot); the second half continued
//            from the middle checkpoint into a third slot (row P-1), the whole one decoding ids[P] (row P);
//   gen:     G greedy tokens after ids[:P] (plain decode steps), with each step's top-2 margin,
//            and the same G tokens through a decode chain (run_chain) as gen_chain;
//   batch:   every case's prefill in one forest (their last rows).
// tests/ref/gemma4_ref.py computes the reference and compares.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/model/api.h"
#include "../src/util/json.h"

using namespace espetero;

namespace {

json::Value row_json(const float* x, int V, int top) {
  int am = 0;
  float mx = x[0];
  for (int i = 1; i < V; ++i)
    if (x[i] > mx) mx = x[i], am = i;
  double s = 0;
  for (int i = 0; i < V; ++i) s += std::exp(double(x[i]) - double(mx));
  std::vector<int> idx(static_cast<size_t>(V));
  for (int i = 0; i < V; ++i) idx[size_t(i)] = i;
  const int k = std::min(top, V);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
  json::Value o = json::Value::object();
  o.set("argmax", am);
  o.set("lse", double(mx) + std::log(s));
  json::Value t = json::Value::array();
  for (int i = 0; i < k; ++i) {
    json::Value p = json::Value::array();
    p.push_back(idx[size_t(i)]);
    p.push_back(double(x[idx[size_t(i)]]));
    t.push_back(std::move(p));
  }
  o.set("top", std::move(t));
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <model> <cases.json> <out.json> [--top N] [--kv-gb G] [--max-ctx N]\n", argv[0]);
    return 2;
  }
  int top = 64;
  LoadOptions o;
  o.kv_gb = 4;
  o.max_ctx = 8192;
  o.slots = 16;
  for (int i = 4; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--top") && i + 1 < argc) top = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--kv-gb") && i + 1 < argc) o.kv_gb = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--max-ctx") && i + 1 < argc) o.max_ctx = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--slots") && i + 1 < argc) o.slots = std::atoi(argv[++i]);
  }
  std::ifstream f(argv[2]);
  std::stringstream ss;
  ss << f.rdbuf();
  const json::Value cases = json::parse(ss.str());
  auto model = NativeModel::load(argv[1], o);
  const ModelInfo& I = model->info();
  const int V = I.vocab;
  std::fprintf(stderr, "loaded %s: %d layers, vocab %d, load %.0f ms\n", argv[1], I.layers, V, I.load_ms);
  json::Value out = json::Value::object();
  json::Value outc = json::Value::array();
  std::vector<std::vector<int32_t>> prompts;
  for (const json::Value& cs : cases.get("cases")->as_array()) {
    std::vector<int32_t> ids;
    for (const json::Value& x : cs.get("ids")->as_array()) ids.push_back(int32_t(x.as_int()));
    const int P = int(cs.get_int("prompt_len", int64_t(ids.size())));
    const int G = int(cs.get_int("gen", 0));
    const int L = int(ids.size()) - P;
    prompts.push_back(std::vector<int32_t>(ids.begin(), ids.begin() + P));
    json::Value rc = json::Value::object();
    PassStats st;
    // prefill
    {
      const int slot = model->slot_alloc();
      TreeSpec t;
      t.root = ids.data();
      t.root_len = P;
      t.save = slot;
      t.score_root = true;
      t.want_logits = true;
      model->run({t}, &st);
      rc.set("prefill", row_json(model->logits(0, 0), V, top));
      rc.set("prefill_ms", st.gpu_ms);
      // decode: in-place passes over ids[P:]
      json::Value dec = json::Value::array();
      for (int j = 0; j < L; ++j) {
        TreeSpec d;
        d.parent = d.save = slot;
        d.root = ids.data() + P + j;
        d.root_len = 1;
        d.score_root = true;
        d.want_logits = true;
        model->run({d});
        dec.push_back(row_json(model->logits(0, 0), V, top));
      }
      rc.set("decode", std::move(dec));
      model->slot_free(slot);
    }
    // tree: root ids[:P], leaves ids[P:P+j]
    if (L > 0) {
      std::vector<int32_t> offs{0}, ltok;
      const int nl = std::min(L, 200);
      for (int j = 1; j <= nl; ++j) {
        for (int q = 0; q < j; ++q) ltok.push_back(ids[size_t(P + q)]);
        offs.push_back(int32_t(ltok.size()));
      }
      TreeSpec t;
      t.root = ids.data();
      t.root_len = P;
      t.n_leaves = nl;
      t.leaf_offsets = offs.data();
      t.leaf_tokens = ltok.data();
      t.score_root = true;
      t.want_logits = true;
      model->run({t});
      json::Value tr = json::Value::array();
      tr.push_back(row_json(model->logits(0, nl), V, top));  // the root's last row (position P - 1)
      for (int j = 0; j < nl; ++j) tr.push_back(row_json(model->logits(0, j), V, top));
      rc.set("tree", std::move(tr));
    }
    // verify: after the prefill, one pass with the next token as the root and
    // the rest as a chain of leaves (a draft verification) over the cached prompt
    if (L > 1) {
      const int slot = model->slot_alloc();
      TreeSpec t;
      t.root = ids.data();
      t.root_len = P;
      t.save = slot;
      model->run({t});
      std::vector<int32_t> offs{0}, ltok;
      const int nl = std::min(L - 1, 200);
      for (int j = 1; j <= nl; ++j) {
        for (int q = 0; q < j; ++q) ltok.push_back(ids[size_t(P + 1 + q)]);
        offs.push_back(int32_t(ltok.size()));
      }
      TreeSpec v;
      v.parent = v.save = slot;
      v.root = ids.data() + P;
      v.root_len = 1;
      v.n_leaves = nl;
      v.leaf_offsets = offs.data();
      v.leaf_tokens = ltok.data();
      v.score_root = true;
      v.want_logits = true;
      model->run({v});
      json::Value vr = json::Value::array();
      vr.push_back(row_json(model->logits(0, nl), V, top));  // the root (position P)
      for (int j = 0; j < nl; ++j) vr.push_back(row_json(model->logits(0, j), V, top));
      rc.set("verify", std::move(vr));
      model->slot_free(slot);
    }
    // split: the prompt saved at its middle (save_len) and whole (work slot);
    // the second half then continues the middle checkpoint into a third slot
    // (prefix checkpoints, rings copied between slots), and the whole one
    // decodes the next token.
    if (P >= 4 && L > 0) {
      const int a = model->slot_alloc(), b = model->slot_alloc(), c = model->slot_alloc();
      const int h = P / 2;
      TreeSpec t;
      t.root = ids.data();
      t.root_len = P;
      t.save = a;
      t.save_len = h;
      t.work = b;
      model->run({t});
      TreeSpec u;
      u.parent = a;
      u.save = c;
      u.root = ids.data() + h;
      u.root_len = P - h;
      u.score_root = true;
      u.want_logits = true;
      model->run({u});
      json::Value sp = json::Value::array();
      sp.push_back(row_json(model->logits(0, 0), V, top));  // position P - 1
      TreeSpec d;
      d.parent = d.save = b;
      d.root = ids.data() + P;
      d.root_len = 1;
      d.score_root = true;
      d.want_logits = true;
      model->run({d});
      sp.push_back(row_json(model->logits(0, 0), V, top));  // position P
      rc.set("split", std::move(sp));
      for (int x : {a, b, c}) model->slot_free(x);
    }
    // gen: greedy decode steps, then the same through a decode chain
    if (G > 0) {
      const int slot = model->slot_alloc();
      TreeSpec t;
      t.root = ids.data();
      t.root_len = P;
      t.save = slot;
      t.score_root = true;
      t.want_logits = true;
      model->run({t});
      json::Value gen = json::Value::array(), margin = json::Value::array();
      int32_t tok = 0;
      auto pick = [&](const float* x) {
        int am = 0;
        for (int i = 1; i < V; ++i)
          if (x[i] > x[am]) am = i;
        float second = -INFINITY;
        for (int i = 0; i < V; ++i)
          if (i != am && x[i] > second) second = x[i];
        margin.push_back(double(x[am] - second));
        return am;
      };
      tok = pick(model->logits(0, 0));
      gen.push_back(tok);
      const int slot2 = model->slot_alloc();
      model->slot_fork(slot2, slot);
      const int32_t first = tok;
      for (int j = 1; j < G; ++j) {
        TreeSpec d;
        d.parent = d.save = slot;
        d.root = &tok;
        d.root_len = 1;
        d.score_root = true;
        d.want_logits = true;
        model->run({d});
        tok = pick(model->logits(0, 0));
        gen.push_back(tok);
      }
      rc.set("gen", std::move(gen));
      rc.set("gen_margin", std::move(margin));
      // decode chain from the fork of the prefill state: the first token fed, G - 1 steps
      if (G > 1) {
        std::vector<int32_t> chain(size_t(G - 1));
        TreeSpec d;
        d.parent = d.save = slot2;
        d.root = &first;
        d.root_len = 1;
        d.score_root = true;
        int32_t sink = 0;
        d.out_argmax = &sink;
        model->run_chain({d}, G - 1, chain.data());
        json::Value gc = json::Value::array();
        gc.push_back(first);
        for (int32_t x : chain) gc.push_back(x);
        rc.set("gen_chain", std::move(gc));
      }
      model->slot_free(slot2);
      model->slot_free(slot);
    }
    outc.push_back(std::move(rc));
    std::fprintf(stderr, "case %zu: %zu ids, prompt %d, gen %d\n", outc.size(), ids.size(), P, G);
  }
  // batch: every prefill in one forest
  {
    std::vector<TreeSpec> ts(prompts.size());
    for (size_t i = 0; i < prompts.size(); ++i) {
      ts[i].root = prompts[i].data();
      ts[i].root_len = int(prompts[i].size());
      ts[i].score_root = true;
      ts[i].want_logits = true;
    }
    model->run(ts);
    json::Value b = json::Value::array();
    for (size_t i = 0; i < prompts.size(); ++i) b.push_back(row_json(model->logits(int(i), 0), V, top));
    out.set("batch", std::move(b));
  }
  out.set("cases", std::move(outc));
  std::ofstream of(argv[3]);
  of << json::dump(out);
  std::fprintf(stderr, "wrote %s\n", argv[3]);
  return 0;
}
