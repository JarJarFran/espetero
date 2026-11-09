// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ESPETERO_CHECK_FINITE on a real checkpoint (a GPU job):
//   1. on, no poison: a prompt pass and decode steps report nothing;
//   2. a NaN poisoned into one site (ESPETERO_CHECK_FINITE_POISON=op:layer:row:col)
//      fails the pass with a report naming that op, layer, row and column (the
//      first offender, though later checks see the NaN spread);
//   3. log only: the same poison logs and the pass goes on;
//   4. the overhead: a 512-row prompt pass and 32 decode steps, off and on.
// Each case loads the model again (the options are read when it loads).
//   build/finite_test <model> [layer=1] [ops=attn_out,attn_res,layer_out,logits] [--no-timing]
//   build/finite_test <model> --dump-off FILE: only the run with the option off, its prompt logits
//   (fp32) into FILE, to compare with the same test linked against another build.
// DeepSeek-V4.1 with ESPETERO_DS4_LAYERS=n (no head): pass ops without logits.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../src/model/api.h"

using namespace espetero;

namespace {

struct Result {
  bool ok = true;
  std::string error;
  std::vector<float> logits;  // the prompt pass's last readout
  double prefill_ms = 0, decode_ms = 0;
};

void set_env(const char* k, const char* v) {
  if (v) setenv(k, v, 1);
  else unsetenv(k);
}

// A prompt pass of `n` tokens saved into a slot, then `steps` decode steps on it.
Result run_case(const std::string& path, const char* check, const char* poison, int n, int steps, int reps) {
  set_env("ESPETERO_CHECK_FINITE", check);
  set_env("ESPETERO_CHECK_FINITE_POISON", poison);
  Result r;
  LoadOptions o;
  o.slots = 2;
  o.max_ctx = 4096;
  o.kv_gb = 1;
  std::unique_ptr<NativeModel> m;
  try {
    m = NativeModel::load(path, o);
  } catch (const std::exception& e) {
    r.ok = false, r.error = std::string("load: ") + e.what();
    return r;
  }
  const int V = m->info().vocab;
  std::mt19937 rng(11);
  std::vector<int32_t> prompt(static_cast<size_t>(n));
  for (auto& t : prompt) t = int32_t(rng() % std::min(V, 100000));
  r.prefill_ms = r.decode_ms = 1e30;
  for (int rep = 0; rep < reps && r.ok; ++rep) {
    const int s = m->slot_alloc();
    try {
      TreeSpec t;
      t.root = prompt.data();
      t.root_len = n;
      t.save = s;
      t.score_root = true;
      t.want_logits = true;
      int32_t am = -1;
      t.out_argmax = &am;
      auto t0 = std::chrono::steady_clock::now();
      m->run({t});
      r.prefill_ms = std::min(r.prefill_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      if (rep == 0) r.logits.assign(m->logits(0, 0), m->logits(0, 0) + V);
      // (log only: a NaN readout's argmax may be anything; keep the token in the vocabulary)
      int32_t tok = am >= 0 && am < V ? am : 1;
      t0 = std::chrono::steady_clock::now();
      for (int k = 0; k < steps; ++k) {
        TreeSpec d;
        d.parent = s, d.save = s;
        d.root = &tok;
        d.root_len = 1;
        d.score_root = true;
        d.out_argmax = &am;
        m->run({d});
        tok = am >= 0 && am < V ? am : 1;
      }
      if (steps)
        r.decode_ms = std::min(r.decode_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / steps);
    } catch (const std::exception& e) {
      r.ok = false, r.error = e.what();
    }
    m->slot_free(s);
  }
  return r;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  for (size_t a = 0; a <= s.size();) {
    size_t b = s.find(',', a);
    if (b == std::string::npos) b = s.size();
    if (b > a) out.push_back(s.substr(a, b - a));
    a = b + 1;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> args;
  bool timing = true;
  std::string dump;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--no-timing") == 0) timing = false;
    else if (std::strcmp(argv[i], "--dump-off") == 0 && i + 1 < argc) dump = argv[++i];
    else args.push_back(argv[i]);
  }
  if (args.empty()) {
    std::fprintf(stderr, "usage: %s <model> [layer=1] [ops] [--no-timing]\n", argv[0]);
    return 2;
  }
  const std::string path = args[0];
  if (!dump.empty()) {
    const Result r = run_case(path, nullptr, nullptr, 512, 0, 1);
    if (!r.ok) {
      std::fprintf(stderr, "%s\n", r.error.c_str());
      return 1;
    }
    FILE* f = std::fopen(dump.c_str(), "wb");
    if (!f || std::fwrite(r.logits.data(), 4, r.logits.size(), f) != r.logits.size()) return 1;
    std::fclose(f);
    std::printf("wrote %zu logits to %s\n", r.logits.size(), dump.c_str());
    return 0;
  }
  const int layer = args.size() > 1 ? std::atoi(args[1].c_str()) : 1;
  const std::vector<std::string> ops = split(args.size() > 2 ? args[2] : "attn_out,attn_res,layer_out,logits");
  int fails = 0;
  auto expect = [&](bool c, const std::string& what) {
    std::printf("%s %s\n", c ? "ok  " : "FAIL", what.c_str());
    if (!c) ++fails;
  };

  // 1. on, nothing poisoned: nothing reported
  {
    const Result r = run_case(path, "1", nullptr, 64, 4, 1);
    expect(r.ok, "clean run with ESPETERO_CHECK_FINITE=1 reports nothing" + (r.ok ? std::string() : ": " + r.error));
  }
  // 2. a NaN at each site: caught at that op, layer, row and column
  for (const std::string& op : ops) {
    const bool logits = op.find("logits") != std::string::npos;
    const int L = logits ? -1 : layer, row = logits ? 0 : 3, col = 5;
    const std::string spec = op + ":" + std::to_string(L) + ":" + std::to_string(row) + ":" + std::to_string(col);
    const Result r = run_case(path, "1", spec.c_str(), 64, 0, 1);
    const std::string where = (L >= 0 ? "layer " + std::to_string(L) + " " : std::string()) + op + ", row " +
                              std::to_string(row) + ", col " + std::to_string(col) + " ";
    const bool hit = !r.ok && r.error.find("check-finite: NaN in ") == 0 && r.error.find(", " + where) != std::string::npos;
    expect(hit, "poison " + spec + " -> " + (r.ok ? std::string("(no error)") : r.error));
  }
  // 3. log only: reported on stderr, the pass goes on (the last op: a NaN that reaches a
  // router fails the pass anyway, its error naming the first offender)
  {
    const std::string& lop = ops.back();
    const std::string spec = lop + ":" + std::to_string(lop.find("logits") != std::string::npos ? -1 : layer) + ":0:1";
    const Result r = run_case(path, "log", spec.c_str(), 64, 2, 1);
    // A model whose host reads the routing (DeepSeek-V4.1) cannot go on once a NaN reaches a
    // router: then the pass fails, its error the detector's first offender.
    const bool stopped = !r.ok && r.error.find("check-finite: NaN in ") == 0 && r.error.find(" (then ") != std::string::npos;
    expect(r.ok || stopped, "log only with poison " + spec + (r.ok ? " goes on" : stopped ? " logs, then a router stops the pass: " + r.error : ": " + r.error));
  }
  // 4. overhead (min of 3), and the logits with the checks on
  if (timing) {
    const Result off = run_case(path, nullptr, nullptr, 512, 32, 3);
    const Result on = run_case(path, "1", nullptr, 512, 32, 3);
    expect(off.ok && on.ok, "timing runs");
    if (off.ok && on.ok) {
      std::printf("prompt 512 rows: off %.2f ms, on %.2f ms (%+.1f %%)\n", off.prefill_ms, on.prefill_ms,
                  100.0 * (on.prefill_ms / off.prefill_ms - 1.0));
      std::printf("decode step:     off %.2f ms, on %.2f ms (%+.1f %%)\n", off.decode_ms, on.decode_ms,
                  100.0 * (on.decode_ms / off.decode_ms - 1.0));
      double d = 0;
      for (size_t i = 0; i < off.logits.size() && i < on.logits.size(); ++i) d = std::max(d, double(std::fabs(off.logits[i] - on.logits[i])));
      std::printf("logits off vs on: max |d| %.3g%s\n", d, d == 0 ? " (bit-identical)" : "");
    }
  }
  std::printf("finite_test: %s\n", fails ? "FAILED" : "ok");
  return fails ? 1 : 0;
}
