// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Parity against the original JEVfire Metal engine (libjevmetal.dylib, loaded
// with dlopen): the same trees through both engines must give the same
// logits up to summation order. Covers single trees, checkpoints, in-place
// decoding, forests, long contexts (split-K attention) and MTP drafting.
//
//   build/parity_test <model dir or .gguf> [old dylib]

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <random>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include "../src/model/api.h"
#include "../src/util/json.h"
#include "jevmetal.h"  // JEVfire native engine: make JEVFIRE_NATIVE=<jevfire checkout>/native build/parity_test

static std::string read_file_text(const char* path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(std::string("cannot open ") + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

using namespace espetero;

struct Old {
  void* h = nullptr;
  decltype(&jm_load) load;
  decltype(&jm_run) run;
  decltype(&jm_run_forest) run_forest;
  decltype(&jm_info) info;
  decltype(&jm_mtp_draft) mtp_draft;
  decltype(&jm_slot_reset) slot_reset;
  jm_model* m = nullptr;
  bool open(const char* lib) {
    h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) return false;
    load = (decltype(load))dlsym(h, "jm_load");
    run = (decltype(run))dlsym(h, "jm_run");
    run_forest = (decltype(run_forest))dlsym(h, "jm_run_forest");
    info = (decltype(info))dlsym(h, "jm_info");
    mtp_draft = (decltype(mtp_draft))dlsym(h, "jm_mtp_draft");
    slot_reset = (decltype(slot_reset))dlsym(h, "jm_slot_reset");
    return load && run && run_forest;
  }
};

static int failures = 0;

static void report(const char* name, double maxdiff, double tol, const char* what = "logits") {
  const bool ok = maxdiff <= tol;
  if (!ok) ++failures;
  std::printf("%-44s max |diff| %-10.3g (%s, tol %.3g) %s\n", name, maxdiff, what, tol, ok ? "ok" : "FAIL");
}

// A tree in both APIs, holding its arrays.
struct Case {
  std::vector<int32_t> root, leaf_off{0}, leaf_tok, cand_off{0}, cand;
  int parent = -1, save = -1, save_len = 0, work = -1;
  bool score_root = false, mtp = false;
  std::vector<float> cand_new, cand_old, lse_new, lse_old;
  std::vector<int32_t> arg_new, arg_old;
  void leaf(const std::vector<int32_t>& toks, const std::vector<int32_t>& cands) {
    leaf_tok.insert(leaf_tok.end(), toks.begin(), toks.end());
    leaf_off.push_back(int32_t(leaf_tok.size()));
    cand.insert(cand.end(), cands.begin(), cands.end());
    cand_off.push_back(int32_t(cand.size()));
  }
  int n_leaves() const { return int(leaf_off.size()) - 1; }
  int readouts() const { return n_leaves() + (score_root ? 1 : 0); }
  TreeSpec spec() {
    cand_new.assign(cand.size(), 0);
    lse_new.assign(size_t(readouts()), 0);
    arg_new.assign(size_t(readouts()), -1);
    TreeSpec t;
    t.parent = parent;
    t.save = save;
    t.save_len = save_len;
    t.work = work;
    t.root = root.data();
    t.root_len = int(root.size());
    t.n_leaves = n_leaves();
    t.leaf_offsets = leaf_off.data();
    t.leaf_tokens = leaf_tok.data();
    t.cand_offsets = n_leaves() ? cand_off.data() : nullptr;
    t.cand_ids = cand.data();
    t.score_root = score_root;
    t.mtp = mtp;
    t.out_cand_logits = cand.empty() ? nullptr : cand_new.data();
    t.out_lse = lse_new.data();
    t.out_argmax = arg_new.data();
    return t;
  }
  jm_batch batch() {
    cand_old.assign(cand.size(), 0);
    lse_old.assign(size_t(readouts()), 0);
    arg_old.assign(size_t(readouts()), -1);
    jm_batch b{};
    b.parent_slot = parent;
    b.save_slot = save;
    b.root_len = int(root.size());
    b.root_tokens = root.data();
    b.save_len = save_len;
    b.n_leaves = n_leaves();
    b.leaf_offsets = leaf_off.data();
    b.leaf_tokens = leaf_tok.data();
    b.cand_offsets = n_leaves() ? cand_off.data() : nullptr;
    b.cand_ids = cand.data();
    b.out_cand_logits = cand.empty() ? nullptr : cand_old.data();
    b.out_lse = lse_old.data();
    b.out_argmax = arg_old.data();
    b.score_root = score_root ? 1 : 0;
    b.work_slot = work;
    b.mtp = mtp ? 1 : 0;
    return b;
  }
  double diff() const {
    double d = 0;
    // NaN anywhere is an infinite difference (std::max would drop it).
    auto upd = [&](float a, float b) {
      const double x = std::fabs(double(a) - double(b));
      d = std::isnan(x) ? INFINITY : std::max(d, x);
    };
    for (size_t i = 0; i < cand_new.size(); ++i) upd(cand_new[i], cand_old[i]);
    for (size_t i = 0; i < lse_new.size(); ++i) upd(lse_new[i], lse_old[i]);
    return d;
  }
  int argmax_mismatch() const {
    int n = 0;
    for (size_t i = 0; i < arg_new.size(); ++i) n += arg_new[i] != arg_old[i];
    return n;
  }
};

// Reference-logits mode: `parity_test <model> --ref <json>` compares the
// engine's last-position logits with a dump from MLX (tests/ref/moe_ref.py),
// once through a single prefill pass and once through prefill + one in-place
// decode step. Passes when every argmax matches (or the two top logits of the
// reference are within 0.05 of each other) and the mean absolute deviation
// stays under 0.2: the engine runs the same quantized weights but rounds
// differently (fp16 operands, fp32 accumulation) than MLX's bf16 path.
// MTP state must not change the decoder (a regression test). The same tokens
// run through two slots, one pass per step (1-3 root tokens, the engine's
// pending sizes), once with the MTP state kept (every pass then also extends
// the MTP layer's K/V from P - 1) and once without, across several 64-token
// block boundaries at every alignment; the readouts' logits must agree up to
// summation order (a kept MTP state turns last-layer pruning off: ~1e-3).
// The case it guards: a pass starting one row after a block boundary must
// keep the decoder's K/V at P - 1, or the MTP slot drifts from then on.
static double mtp_state_check(NativeModel& nm, int slot_on, int slot_off, std::mt19937& rng, int V) {
  auto toks = [&](int n) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    for (auto& t : v) t = int32_t(rng() % uint32_t(V));
    return v;
  };
  const int Vn = nm.info().vocab;
  std::vector<float> on(static_cast<size_t>(Vn));
  double worst = 0;
  auto step = [&](const std::vector<int32_t>& root, bool first) {
    int32_t arg[2];
    for (int k = 0; k < 2; ++k) {
      TreeSpec t;
      t.parent = first ? -1 : (k == 0 ? slot_on : slot_off);
      t.save = k == 0 ? slot_on : slot_off;
      t.root = root.data();
      t.root_len = int(root.size());
      t.score_root = true;
      t.mtp = k == 0;
      t.want_logits = true;
      t.out_argmax = &arg[k];
      nm.run({t});
      const float* lg = nm.logits(0, 0);
      if (k == 0) {
        std::copy(lg, lg + Vn, on.begin());
      } else {
        for (int v = 0; v < Vn; ++v) worst = std::max(worst, double(std::fabs(lg[v] - on[size_t(v)])));
      }
    }
  };
  step(toks(61), true);
  for (int i = 0; i < 90; ++i) step(toks(1 + (i * 7 % 3)), false);  // P runs 61 -> ~241
  return worst;
}

// The whole MTP drafting loop must leave the decoder unchanged: slot A runs
// the engine's loop (MTP draft pass, verify pass with the drafts as prefix
// leaves and MTP state, greedy acceptance, accepted tokens re-fed as the next
// root), slot B decodes the same roots without MTP state or leaves. The
// logits of every root readout must agree up to summation order.
static double mtp_loop_check(NativeModel& nm, int slot_a, int slot_b, std::mt19937& rng, int V, int* accepted_out) {
  const int Vn = nm.info().vocab;
  std::vector<int32_t> prompt(59);
  for (auto& t : prompt) t = int32_t(rng() % uint32_t(V));
  std::vector<float> la(static_cast<size_t>(Vn));
  double worst = 0;
  int accepted = 0;
  // prefill both
  int32_t next = -1;
  for (int k = 0; k < 2; ++k) {
    TreeSpec t;
    t.save = k == 0 ? slot_a : slot_b;
    t.root = prompt.data();
    t.root_len = int(prompt.size());
    t.score_root = true;
    t.mtp = k == 0;
    int32_t arg = -1;
    t.out_argmax = &arg;
    nm.run({t});
    next = arg;
  }
  std::vector<int32_t> pending{next};
  for (int step = 0; step < 60; ++step) {
    int32_t d[3], d2[3];
    DraftRequest q;
    q.slot = slot_a;
    q.tokens = pending.data();
    q.n_tokens = int(pending.size());
    q.n_draft = 3;
    q.out = d;
    q.out2 = d2;
    nm.draft({q});
    // A: root = pending, leaves = prefixes of the drafts, MTP state kept
    std::vector<int32_t> off{0}, lt;
    for (int i = 0; i < 3; ++i) {
      lt.insert(lt.end(), d, d + i + 1);
      off.push_back(int32_t(lt.size()));
    }
    int32_t arg_a[4];
    TreeSpec a;
    a.parent = slot_a;
    a.save = slot_a;
    a.root = pending.data();
    a.root_len = int(pending.size());
    a.n_leaves = 3;
    a.leaf_offsets = off.data();
    a.leaf_tokens = lt.data();
    a.score_root = true;
    a.mtp = true;
    a.want_logits = true;
    a.out_argmax = arg_a;
    nm.run({a});
    const float* lg = nm.logits(0, 3);  // the root readout comes last
    std::copy(lg, lg + Vn, la.begin());
    // B: the same root, plain
    int32_t arg_b = -1;
    TreeSpec b;
    b.parent = slot_b;
    b.save = slot_b;
    b.root = pending.data();
    b.root_len = int(pending.size());
    b.score_root = true;
    b.want_logits = true;
    b.out_argmax = &arg_b;
    nm.run({b});
    lg = nm.logits(0, 0);
    for (int v = 0; v < Vn; ++v) worst = std::max(worst, double(std::fabs(lg[v] - la[size_t(v)])));
    // greedy acceptance on A's readouts (leaf i after draft i, the root last)
    std::vector<int32_t> out;
    int32_t tok = arg_a[3];
    for (int i = 0; i < 3; ++i) {
      out.push_back(tok);
      if (tok != d[i]) break;
      ++accepted;
      tok = arg_a[i];
    }
    if (out.size() == 3 && out.back() == d[2]) out.push_back(arg_a[2]);
    pending = out;
  }
  if (accepted_out) *accepted_out = accepted;
  return worst;
}

// Verify-commit: slot A runs the engine's commit loop (k MTP drafts
// from the one pending token, a verify pass with the drafts as one commit
// chain and the MTP state, greedy acceptance, commit of the accepted drafts);
// slot B decodes the same tokens plainly (one pass of the pending token and
// the accepted drafts). After every step both must hold the same length, and
// a one-token probe of the next token from each slot (into temp slots) must
// give the same logits up to summation order: committed K/V, DeltaNet state,
// conv window and positions. Slot B keeps its MTP state too, and both slots
// draft each step: drafts that differ show a committed MTP state (K/V, index
// rows, hidden row) unlike plain decoding's (counted in *draft_diff_out;
// near-ties aside, none). A prompt over 2,051 tokens makes Qwen4-Exp's
// sparse attention read the committed index rows. Returns the worst |dlogit|.
// With slot_c >= 0, slot C decodes the same tokens one pass per token: B
// against C is the noise floor of two plain decodes that differ only in
// their passes' row counts (Qwen4-Exp's top-10 routing can turn summation
// order into ~1-logit flips), to judge A against B by.
// ESPETERO_COMMIT_TRACE=1 prints every step.
struct CommitStats {
  int accepted = 0, steps = 0, draft_diff = 0, n0 = 0, nk = 0;
  double worst_bc = 0, mean_ab = 0, mean_bc = 0;
  int am_ab = 0, am_bc = 0;  // probe argmax mismatches
  int draft_noise = 0;       // steps whose drafts differ between B and C
};
static double mtp_commit_check(NativeModel& nm, int slot_a, int slot_b, int tmp_a, int tmp_b, std::mt19937& rng, int V, int k,
                               int prompt_len, CommitStats* cs, int slot_c = -1, int tmp_c = -1) {
  const bool trace = std::getenv("ESPETERO_COMMIT_TRACE") != nullptr;
  std::vector<float> lb;
  const int Vn = nm.info().vocab;
  std::vector<int32_t> prompt(static_cast<size_t>(prompt_len));
  for (auto& t : prompt) t = int32_t(rng() % uint32_t(V));
  std::vector<float> la(static_cast<size_t>(Vn));
  double worst = 0;
  int accepted = 0, steps = 0;
  int32_t pending = -1;
  for (int s = 0; s < (slot_c >= 0 ? 3 : 2); ++s) {
    TreeSpec t;
    t.save = s == 0 ? slot_a : s == 1 ? slot_b : slot_c;
    t.root = prompt.data();
    t.root_len = int(prompt.size());
    t.score_root = true;
    t.mtp = true;
    int32_t arg = -1;
    t.out_argmax = &arg;
    nm.run({t});
    pending = arg;
  }
  for (int step = 0; step < 48; ++step, ++steps) {
    std::vector<int32_t> d(static_cast<size_t>(k)), d2(static_cast<size_t>(k)), db(static_cast<size_t>(k));
    DraftRequest q;
    q.slot = slot_a;
    q.tokens = &pending;
    q.n_tokens = 1;
    q.n_draft = k;
    q.out = d.data();
    q.out2 = d2.data();
    nm.draft({q});
    DraftRequest qb = q;
    qb.slot = slot_b;
    qb.out = db.data();
    qb.out2 = nullptr;
    nm.draft({qb});
    if (cs && db != d) ++cs->draft_diff;
    if (slot_c >= 0) {
      std::vector<int32_t> dc(static_cast<size_t>(k));
      DraftRequest qc = qb;
      qc.slot = slot_c;
      qc.out = dc.data();
      nm.draft({qc});
      if (cs && dc != db) ++cs->draft_noise;
    }
    std::vector<int32_t> off{0}, lt;
    for (int i = 0; i < k; ++i) {
      lt.insert(lt.end(), d.begin(), d.begin() + i + 1);
      off.push_back(int32_t(lt.size()));
    }
    std::vector<int32_t> arg(static_cast<size_t>(k + 1), -1);
    TreeSpec a;
    a.parent = slot_a;
    a.save = slot_a;
    a.root = &pending;
    a.root_len = 1;
    a.n_leaves = k;
    a.leaf_offsets = off.data();
    a.leaf_tokens = lt.data();
    a.score_root = true;
    a.mtp = true;
    a.commit_chain = k;
    a.out_argmax = arg.data();
    nm.run({a});
    // greedy acceptance (readout i after draft i, the root's last)
    int acc = 0;
    int32_t tok = arg[size_t(k)];
    while (acc < k && tok == d[size_t(acc)]) tok = arg[size_t(acc++)];
    accepted += acc;
    if (cs) cs->n0 += acc == 0, cs->nk += acc == k;
    nm.commit({{slot_a, acc}});
    std::vector<int32_t> broot{pending};
    broot.insert(broot.end(), d.begin(), d.begin() + acc);
    int32_t argb = -1;
    TreeSpec b;
    b.parent = slot_b;
    b.save = slot_b;
    b.root = broot.data();
    b.root_len = int(broot.size());
    b.score_root = true;
    b.mtp = true;
    b.out_argmax = &argb;
    nm.run({b});
    if (slot_c >= 0)
      for (int32_t t1 : broot) {  // slot C: one pass per token
        int32_t argc = -1;
        TreeSpec c1;
        c1.parent = slot_c;
        c1.save = slot_c;
        c1.root = &t1;
        c1.root_len = 1;
        c1.score_root = true;
        c1.mtp = true;
        c1.out_argmax = &argc;
        nm.run({c1});
      }
    if (nm.slot_len(slot_a) != nm.slot_len(slot_b) || nm.slot_next_pos(slot_a) != nm.slot_next_pos(slot_b)) {
      std::printf("   step %d: lengths %d / %d, next positions %d / %d\n", step, nm.slot_len(slot_a), nm.slot_len(slot_b),
                  nm.slot_next_pos(slot_a), nm.slot_next_pos(slot_b));
      return 1e9;
    }
    double step_ab = 0, step_bc = 0, sum_ab = 0, sum_bc = 0;
    int32_t am[3] = {-1, -1, -1};
    for (int s = 0; s < (slot_c >= 0 ? 3 : 2); ++s) {
      TreeSpec p;
      p.parent = s == 0 ? slot_a : s == 1 ? slot_b : slot_c;
      p.save = s == 0 ? tmp_a : s == 1 ? tmp_b : tmp_c;
      p.root = &tok;
      p.root_len = 1;
      p.score_root = true;
      p.want_logits = true;
      p.out_argmax = &am[s];
      nm.run({p});
      const float* lg = nm.logits(0, 0);
      if (s == 0) {
        std::copy(lg, lg + Vn, la.begin());
      } else if (s == 1) {
        lb.assign(lg, lg + Vn);
        for (int v = 0; v < Vn; ++v) {
          const double dd = std::fabs(lg[v] - la[size_t(v)]);
          step_ab = std::max(step_ab, dd), sum_ab += dd;
        }
      } else {
        for (int v = 0; v < Vn; ++v) {
          const double dd = std::fabs(lg[v] - lb[size_t(v)]);
          step_bc = std::max(step_bc, dd), sum_bc += dd;
        }
      }
    }
    worst = std::max(worst, step_ab);
    if (cs) {
      cs->worst_bc = std::max(cs->worst_bc, step_bc);
      cs->mean_ab += sum_ab / Vn / 48, cs->mean_bc += sum_bc / Vn / 48;
      cs->am_ab += am[0] != am[1], cs->am_bc += slot_c >= 0 && am[1] != am[2];
    }
    if (trace)
      std::printf("   step %2d acc %d  A-B max %.4f mean %.5f%s  B-C max %.4f mean %.5f%s\n", step, acc, step_ab, sum_ab / Vn,
                  am[0] != am[1] ? " argmax!" : "", step_bc, sum_bc / Vn, slot_c >= 0 && am[1] != am[2] ? " argmax!" : "");
    pending = tok;
  }
  if (cs) cs->accepted = accepted, cs->steps = steps;
  return worst;
}

static int ref_mode(const char* model, const char* ref_path) {
  json::Value ref;
  try {
    ref = json::parse(read_file_text(ref_path));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ref: %s\n", e.what());
    return 2;
  }
  size_t longest = 0;
  if (const json::Value* cs = ref.get("cases"); cs && cs->is_array())
    for (const json::Value& c : cs->as_array())
      if (const json::Value* ids = c.get("ids")) longest = std::max(longest, ids->size());
  LoadOptions o;
  o.slots = 4;
  o.max_ctx = std::max(8192, int(longest) + 64);
  o.kv_gb = longest > 8000 ? 2.0 : 1.0;
  std::unique_ptr<NativeModel> nm;
  try {
    nm = NativeModel::load(model, o);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "load: %s\n", e.what());
    return 2;
  }
  const ModelInfo& I = nm->info();
  if (std::getenv("ESPETERO_PARITY_PROFILE")) nm->set_profile(true);  // every dispatch alone, timed
  std::printf("model %s (%s)  ref %s (%s)\n", model, I.format.c_str(), ref_path, ref.get_string("backend", "?").c_str());
  const json::Value* cases = ref.get("cases");
  if (!cases || !cases->is_array()) {
    std::fprintf(stderr, "ref: no cases\n");
    return 2;
  }
  int fails = 0;
  for (const json::Value& cs : cases->as_array()) {
    std::vector<int32_t> ids;
    for (const json::Value& v : cs.get("ids")->as_array()) ids.push_back(int32_t(v.as_int()));
    const auto& lg = cs.get("last_logits")->as_array();
    if (int(lg.size()) != I.vocab) {
      std::printf("case: vocab mismatch (%zu vs %d)\n", lg.size(), I.vocab);
      ++fails;
      continue;
    }
    std::vector<float> want(lg.size());
    for (size_t i = 0; i < lg.size(); ++i) want[i] = float(lg[i].as_double());
    int ref_arg = 0;
    float ref_top = -1e30f, ref_second = -1e30f;
    for (size_t i = 0; i < want.size(); ++i)
      if (want[i] > ref_top) ref_second = ref_top, ref_top = want[i], ref_arg = int(i);
      else if (want[i] > ref_second) ref_second = want[i];
    auto compare = [&](const char* how, const float* got) {
      double sum = 0, mx = 0;
      int arg = 0;
      for (size_t i = 0; i < want.size(); ++i) {
        const double d = std::fabs(double(got[i]) - double(want[i]));
        sum += d;
        if (!(d <= mx)) mx = std::isnan(d) ? INFINITY : d;
        if (got[i] > got[size_t(arg)]) arg = int(i);
      }
      const double mean = sum / double(want.size());
      const bool near_tie = ref_top - ref_second < 0.05f;
      const bool ok = (arg == ref_arg || near_tie) && mean < 0.2 && !std::isinf(mx);
      if (!ok) ++fails;
      std::printf("  %-22s argmax %d/%d%s  mean|diff| %.4f  max|diff| %.3f  %s\n", how, arg, ref_arg,
                  near_tie ? " (ref near-tie)" : "", mean, mx, ok ? "ok" : "FAIL");
      std::fflush(stdout);
    };
    std::printf("case \"%.40s\" (%zu tokens)\n", cs.get_string("text", "").c_str(), ids.size());
    std::fflush(stdout);
    const int s = nm->slot_alloc();
    // Prompts longer than 4096 tokens go in 4096-row passes (the server's
    // largest; one 16K-row pass needs ~10 GB of row scratch beside the experts).
    auto prefill = [&](int n, bool score) {
      for (int at = 0; at < n; at += 4096) {
        TreeSpec t;
        if (at > 0) t.parent = s;
        t.save = s;
        t.root = ids.data() + at;
        t.root_len = std::min(4096, n - at);
        t.score_root = score && at + t.root_len == n;
        t.want_logits = t.score_root;
        nm->run({t});
      }
    };
    // 1. one prefill pass over every token
    {
      prefill(int(ids.size()), true);
      compare(ids.size() > 4096 ? "4096-row passes" : "prefill pass", nm->logits(0, 0));
      if (std::getenv("ESPETERO_PARITY_PROFILE")) std::printf("profile: %s\n", nm->profile_json().c_str());
    }
    // 2. prefill all but the last token, then decode it in place
    if (ids.size() > 1) {
      nm->slot_reset(s);
      prefill(int(ids.size()) - 1, false);
      TreeSpec d;
      d.parent = s;
      d.save = s;
      d.root = ids.data() + ids.size() - 1;
      d.root_len = 1;
      d.score_root = true;
      d.want_logits = true;
      nm->run({d});
      compare("prefill + decode step", nm->logits(0, 0));
    }
    // Long contexts (past the sparse attention's dense range): the prompt
    // in 1000-token passes (blocks of 4 straddle the passes), and 5 decode
    // steps after a prefill.
    if (ids.size() > 2100) {
      nm->slot_reset(s);
      const int n = int(ids.size());
      for (int at = 0; at < n; at += 1000) {
        TreeSpec t;
        if (at > 0) t.parent = s;
        t.save = s;
        t.root = ids.data() + at;
        t.root_len = std::min(1000, n - at);
        t.score_root = at + t.root_len == n;
        t.want_logits = t.score_root;
        nm->run({t});
      }
      compare("1000-token passes", nm->logits(0, 0));
      nm->slot_reset(s);
      prefill(n - 5, false);
      for (int k = n - 5; k < n; ++k) {
        TreeSpec d;
        d.parent = s;
        d.save = s;
        d.root = ids.data() + k;
        d.root_len = 1;
        d.score_root = k == n - 1;
        d.want_logits = d.score_root;
        nm->run({d});
      }
      compare("prefill + 5 decode steps", nm->logits(0, 0));
    }
    nm->slot_free(s);
  }
  // Expert-cache counters of the whole run (Qwen4-Exp, e.g. its split handshakes).
  if (std::getenv("ESPETERO_PARITY_XSTATS")) std::printf("xcache %s\n", nm->bench_json("xcache").c_str());
  std::printf("%s (%d failures)\n", fails ? "REF FAILED" : "REF OK", fails);
  return fails ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model> [old dylib] | %s <model> --ref <mlx logits json>\n", argv[0], argv[0]);
    return 2;
  }
  if (argc > 3 && std::string(argv[2]) == "--ref") return ref_mode(argv[1], argv[3]);
  if (argc > 2 && std::string(argv[2]) == "--mtp-state") {
    // A model with an MTP head (e.g. a GGUF with its nextn block); no reference engine needed.
    LoadOptions o;
    o.slots = 10;
    o.max_ctx = 4096;
    o.kv_gb = 2.0;
    auto nm = NativeModel::load(argv[1], o);
    if (!nm->info().mtp) {
      std::fprintf(stderr, "%s has no MTP head\n", argv[1]);
      return 2;
    }
    std::mt19937 rng(99);
    const int a = nm->slot_alloc(), b = nm->slot_alloc();
    report("MTP state leaves the decoder unchanged", mtp_state_check(*nm, a, b, rng, std::min(nm->info().vocab, 150000)), 0.02);
    const int c = nm->slot_alloc(), d = nm->slot_alloc();
    int acc = 0;
    report("MTP drafting loop leaves the decoder unchanged", mtp_loop_check(*nm, c, d, rng, std::min(nm->info().vocab, 150000), &acc), 0.02);
    std::printf("   drafts accepted: %d of 180\n", acc);
    if (nm->info().commit) {
      const int e = nm->slot_alloc(), f = nm->slot_alloc(), g = nm->slot_alloc(), h = nm->slot_alloc();
      const int i2 = nm->slot_alloc(), j2 = nm->slot_alloc();
      // (drafts, prompt tokens): the long prompt passes Qwen4-Exp's 2,051-token dense window.
      for (auto [k, plen] : {std::pair{3, 59}, std::pair{7, 59}, std::pair{7, 2300}}) {
        CommitStats cs;
        const double w = mtp_commit_check(*nm, e, f, g, h, rng, std::min(nm->info().vocab, 150000), k, plen, &cs, i2, j2);
        // Models whose plain decodes already differ with the passes' row
        // counts (Qwen4-Exp: summation order flips top-10 routing, ~1 logit)
        // are judged against that noise floor: the committed state may differ
        // from plain decoding by no more than plain decoding from itself.
        const bool noisy = cs.mean_bc > 0.005;
        const bool in_noise = noisy && cs.mean_ab <= 1.5 * cs.mean_bc + 0.002 && cs.am_ab <= cs.am_bc + 1;
        report(("verify-commit loop (" + std::to_string(k) + " drafts, " + std::to_string(plen) +
                "-token prompt) leaves the decoder unchanged" + (in_noise ? " (within plain decoding's noise floor)" : "")).c_str(),
               in_noise ? 0.0 : w, 0.05);
        std::printf("   drafts accepted: %d of %d (none: %d steps, all: %d steps); drafts differ from plain decoding's at %d of %d steps\n",
                    cs.accepted, cs.steps * k, cs.n0, cs.nk, cs.draft_diff, cs.steps);
        std::printf("   commit vs plain: worst %.4f mean %.5f, probe argmax differs at %d steps; noise floor (plain, one pass vs "
                    "a pass per token): worst %.4f mean %.5f, argmax differs at %d steps\n",
                    w, cs.mean_ab, cs.am_ab, cs.worst_bc, cs.mean_bc, cs.am_bc);
        std::printf("   drafts of two plain decodes differ at %d of %d steps\n", cs.draft_noise, cs.steps);
        if (cs.draft_diff > 4 && !(noisy && cs.draft_diff <= cs.draft_noise + 3)) {
          std::printf("FAIL verify-commit loop: the committed MTP state drafts unlike plain decoding's\n");
          ++failures;
        }
      }
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL OK", failures);
    return failures ? 1 : 0;
  }
  const char* model = argv[1];
  const char* lib = argc > 2 ? argv[2] : std::getenv("JEVMETAL_LIB");
  if (!lib) {
    std::fprintf(stderr, "pass the JEVfire libjevmetal.dylib as the second argument (or set JEVMETAL_LIB)\n");
    return 2;
  }
  Old old;
  if (!old.open(lib)) {
    std::fprintf(stderr, "cannot open %s: %s\n", lib, dlerror());
    return 2;
  }
  char err[1024] = {0};
  old.m = old.load(model, "{\"slots\":8,\"max_ctx\":8192}", err, sizeof(err));
  if (!old.m) {
    std::fprintf(stderr, "old load: %s\n", err);
    return 2;
  }
  LoadOptions o;
  o.slots = 8;
  o.max_ctx = 8192;
  o.kv_gb = 1.0;
  std::unique_ptr<NativeModel> nm;
  try {
    nm = NativeModel::load(model, o);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "new load: %s\n", e.what());
    return 2;
  }
  const ModelInfo& I = nm->info();
  std::printf("model %s  (%s, vocab %d, mtp %d, vision %d)  load %.0f ms\n", model, I.format.c_str(), I.vocab,
              int(I.mtp), int(I.vision), I.load_ms);
  std::printf("old: %s\n", old.info(old.m));
  std::mt19937 rng(1234);
  const int V = std::min(I.vocab, 150000);  // plain text ids
  auto toks = [&](int n) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    for (auto& t : v) t = int32_t(rng() % uint32_t(V));
    return v;
  };
  auto cands = [&](int n) { return toks(n); };
  // Slots: the old engine numbers them 0..7; allocate the same indices here.
  std::vector<int> slot(8);
  for (int i = 0; i < 8; ++i) slot[size_t(i)] = nm->slot_alloc();
  auto run_both = [&](std::vector<Case*> cs) {
    std::vector<TreeSpec> ts;
    std::vector<jm_batch> bs;
    for (Case* c : cs) {
      ts.push_back(c->spec());
      bs.push_back(c->batch());
    }
    PassStats st;
    nm->run(ts, &st);
    jm_stats ost{};
    if (old.run_forest(old.m, bs.data(), int(bs.size()), &ost, err, sizeof(err)) != 0) {
      std::fprintf(stderr, "old run: %s\n", err);
      std::exit(1);
    }
    return std::make_pair(st, ost);
  };
  const double tol = 2e-2;

  // 1. One tree: root + leaves with candidates (a decision request).
  {
    Case c;
    c.root = toks(300);
    for (int i = 0; i < 6; ++i) c.leaf(toks(1 + int(rng() % 12)), cands(4));
    auto [st, ost] = run_both({&c});
    report("tree: 300-token root, 6 leaves", c.diff(), tol);
    std::printf("   new %.2f ms gpu (%d dispatches, %d barriers) | old %.2f ms gpu (%d dispatches)\n", st.gpu_ms,
                st.dispatches, st.barriers, ost.gpu_ms, ost.dispatches);
  }
  // 2. Checkpoint then continue: save 200 tokens into slot 1, then a tree on it.
  {
    Case a;
    a.root = toks(200);
    a.save = 1;
    a.score_root = true;
    run_both({&a});
    report("checkpoint save (root readout)", a.diff(), tol);
    Case b;
    b.parent = 1;
    b.root = toks(37);
    for (int i = 0; i < 4; ++i) b.leaf(toks(3 + int(rng() % 6)), cands(3));
    run_both({&b});
    report("tree on checkpoint", b.diff(), tol);
    // save_len < root_len with a work slot (chat prompt layout).
    Case c;
    c.parent = 1;
    c.root = toks(80);
    c.save = 2;
    c.save_len = 50;
    c.work = 3;
    c.score_root = true;
    run_both({&c});
    report("save_len + work slot", c.diff(), tol);
    Case d;  // continue the save slot (50 tokens after the checkpoint)
    d.parent = 2;
    d.root = toks(10);
    d.score_root = true;
    run_both({&d});
    report("continue the save_len checkpoint", d.diff(), tol);
  }
  // 3. In-place decoding: parent = save = the work slot, one token per step.
  {
    double worst = 0;
    int mism = 0;
    for (int step = 0; step < 24; ++step) {
      Case c;
      c.parent = 3;
      c.save = 3;
      c.root = toks(1 + (step % 3));
      c.score_root = true;
      if (step % 4 == 0) c.leaf(toks(2), cands(2));
      run_both({&c});
      worst = std::max(worst, c.diff());
      mism += c.argmax_mismatch();
    }
    report("in-place decode, 24 steps", worst, tol);
    if (mism) std::printf("   argmax mismatches: %d\n", mism);
  }
  // 4. Forest: three trees with different parents and saves in one pass.
  {
    Case a, b, c;
    a.parent = 1;
    a.root = toks(20);
    a.save = 4;
    for (int i = 0; i < 3; ++i) a.leaf(toks(4), cands(5));
    b.root = toks(64);
    b.score_root = true;
    b.save = 5;
    c.parent = 3;
    c.root = toks(2);
    c.save = 3;
    c.score_root = true;
    c.leaf(toks(3), cands(2));
    run_both({&a, &b, &c});
    report("forest of 3 trees", std::max({a.diff(), b.diff(), c.diff()}), tol);
  }
  // 5. Long context: 4,000 cached tokens, then decode (split-K attention here).
  {
    Case a;
    a.root = toks(4000);
    a.save = 6;
    a.score_root = true;
    auto [st, ost] = run_both({&a});
    report("prefill 4,000 tokens", a.diff(), tol);
    std::printf("   new %.1f ms gpu | old %.1f ms gpu\n", st.gpu_ms, ost.gpu_ms);
    double worst = 0;
    double tn = 0, to = 0;
    for (int step = 0; step < 8; ++step) {
      Case c;
      c.parent = 6;
      c.save = 6;
      c.root = toks(1);
      c.score_root = true;
      auto [s1, s2] = run_both({&c});
      tn += s1.gpu_ms;
      to += s2.gpu_ms;
      worst = std::max(worst, c.diff());
    }
    report("decode at 4K context (split-K)", worst, tol);
    std::printf("   decode step: new %.2f ms gpu | old %.2f ms gpu\n", tn / 8, to / 8);
  }
  // 6. MTP: a prompt saved with the draft state, then drafts from it.
  if (I.mtp) {
    Case a;
    a.root = toks(120);
    a.save = 7;
    a.mtp = true;
    a.score_root = true;
    run_both({&a});
    report("MTP prompt pass", a.diff(), tol);
    int32_t next = a.arg_new.back();
    int32_t dn[3], dold[3];
    DraftRequest q;
    q.slot = 7;
    q.tokens = &next;
    q.n_tokens = 1;
    q.n_draft = 3;
    q.out = dn;
    nm->draft({q});
    jm_stats ds{};
    if (old.mtp_draft(old.m, 7, &next, 1, 3, dold, &ds, err, sizeof(err)) != 0) std::printf("old draft: %s\n", err);
    int same = 0;
    for (int k = 0; k < 3; ++k) same += dn[k] == dold[k];
    std::printf("%-44s %d/3 identical (new %d %d %d, old %d %d %d)\n", "MTP drafts", same, dn[0], dn[1], dn[2], dold[0],
                dold[1], dold[2]);
    if (same < 3) ++failures;
    // Verification pass: pending token + drafts as a leaf chain, in place.
    Case v;
    v.parent = 7;
    v.save = 7;
    v.mtp = true;
    v.root = {next};
    for (int k = 0; k < 3; ++k) v.leaf(std::vector<int32_t>(dn, dn + k + 1), {});
    v.cand_off.assign(1, 0);
    v.cand.clear();
    // leaves without candidates: argmax readouts
    v.cand_off.clear();
    v.score_root = true;
    TreeSpec ts = v.spec();
    ts.cand_offsets = nullptr;
    jm_batch bb = v.batch();
    bb.cand_offsets = nullptr;
    bb.out_cand_logits = nullptr;
    nm->run({ts});
    jm_stats vs{};
    old.run_forest(old.m, &bb, 1, &vs, err, sizeof(err));
    report("MTP verification pass", v.diff(), tol);
    std::printf("   argmax mismatches: %d\n", v.argmax_mismatch());
  }
  if (I.mtp) report("MTP state leaves the decoder unchanged", mtp_state_check(*nm, slot[5], slot[6], rng, V), tol);
  std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL OK", failures);
  return failures ? 1 : 0;
}
