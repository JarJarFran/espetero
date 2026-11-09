// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Preemption when the KV cache runs out (GPU, a real checkpoint):
//
//   build/preempt_test <model dir> [max_tokens (640)] [variant]
//
// Four requests that need about twice a small KV pool run at once (the
// admission reservation is off, ESPETERO_KV_RESERVE=0, so they are all
// admitted and the pool runs out mid-generation). With preemption on, every
// request must end as it does alone ("length", every token) and every parked
// one with exactly the tokens it produces alone on a server with room, in four
// variants: greedy; seeded sampling; seeded sampling with repetition /
// frequency penalties and no_repeat_ngram_size; greedy with one request under
// structured output (a JSON schema with at least 25 array items, the
// batch-priority one). A difference in a request before it was ever parked is
// reported, not failed: batching alone moves near ties (the control line shows
// the same four with room). The batch-priority request is parked; the oldest
// interactive one never is. With --no-preempt the same load stops requests
// with "KV cache full" (the old behaviour).
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../src/engine/engine.h"
#include "../src/engine/grammar.h"
#include "../src/engine/json_schema.h"
#include "../src/tok/tokenizer.h"
#include "../src/util/json.h"

using namespace espetero;

static int g_checks = 0, g_fails = 0;
#define CHECK(c)                                         \
  do {                                                   \
    ++g_checks;                                          \
    if (!(c)) {                                          \
      ++g_fails;                                         \
      std::printf("  FAIL line %d: %s\n", __LINE__, #c); \
    }                                                    \
  } while (0)

namespace {

struct Result {
  std::vector<int32_t> tokens;
  std::string reason, error;
  int completion = 0, preemptions = 0, resume_rows = 0, first_parked = -1;
  double resume_ms = 0, parked_ms = 0, first_ms = 0;
};

const char* kPreface =
    "You are a careful writer for a travel magazine. Every answer is long, concrete and well organised, with "
    "names, dates, distances and small details a visitor would want to know, and it never repeats itself. ";
const char* kTopics[4] = {
    "Write a detailed guide to a three-day walk along the coast of Galicia, day by day.",
    "Describe the history of the harbour of Málaga from Roman times to today.",
    "Explain how to plan a cycling trip across the Pyrenees in early summer, with a packing list.",
    "Write a long letter from a lighthouse keeper on a small island describing one year of weather.",
};

std::unique_ptr<Engine> load(const std::string& dir, double kv_gb, bool preempt) {
  EngineOptions o;
  o.model_path = dir;
  o.max_seqs = 4;
  o.preempt = preempt;
  o.decode_first = false;  // prompts start at once beside decoding ones: the four overlap from the start
  o.load.slots = 24;
  o.load.max_ctx = 4096;
  o.load.kv_gb = kv_gb;
  auto e = std::make_unique<Engine>(o);
  const ModelInfo& I = e->info();
  std::printf("loaded: kv pool %d tokens (%d blocks of %d), %d slots, preempt %s\n", I.kv_blocks * I.block_tokens,
              I.kv_blocks, I.block_tokens, I.slots, preempt ? "on" : "off");
  return e;
}

std::vector<int32_t> prompt_of(const Engine& e, int i) {
  return e.tokenizer().encode(std::string(kPreface) + kTopics[i] + "\n\n", false);
}

struct Variant {
  const char* name;
  double temperature = 0, top_p = 1, rep = 1, freq = 0;
  int no_repeat = 0;
  bool structured = false;  // request 0 under a JSON-object grammar
};

GenRequest request_of(const Engine& e, const Variant& v, int i, int max_tokens,
                      const std::shared_ptr<const grammar::Constraint>& json_obj) {
  GenRequest r;
  r.prompt = prompt_of(e, i);
  r.split = int(r.prompt.size());  // a prompt-end checkpoint, as a chat turn's (reply checkpoints too)
  r.cache_prompt = 1;
  r.max_tokens = max_tokens;
  r.ignore_eos = true;
  r.sampling.temperature = v.temperature;
  r.sampling.top_p = v.top_p;
  r.sampling.repetition_penalty = v.rep;
  r.sampling.frequency_penalty = v.freq;
  r.sampling.no_repeat_ngram_size = v.no_repeat;
  r.seed = 1000 + uint64_t(i);
  r.has_seed = true;
  r.priority = i == 0 ? 1 : 0;  // request 0: a batch job
  if (v.structured && i == 0) {
    r.constraint = json_obj;
    r.ignore_eos = false;  // the reply ends with the JSON value
    r.prompt = e.tokenizer().encode(std::string(kPreface) +
                                        "As a JSON object, list thirty towns of Spain with their province, population and a "
                                        "sentence about each one.\n\n",
                                    false);
    r.split = int(r.prompt.size());
  }
  return r;
}

Result drain(const std::shared_ptr<GenStream>& s) {
  Result r;
  std::vector<int32_t> chunk;
  while (s->next(chunk)) {
  }
  r.tokens = std::move(chunk);
  r.reason = s->finish_reason;
  r.error = s->error;
  r.completion = s->completion_tokens;
  if (const json::Value* p = s->timings.get("preemptions")) r.preemptions = int(p->as_int());
  if (const json::Value* p = s->timings.get("resume_rows")) r.resume_rows = int(p->as_int());
  if (const json::Value* p = s->timings.get("resume_ms")) r.resume_ms = p->as_double();
  if (const json::Value* p = s->timings.get("parked_ms")) r.parked_ms = p->as_double();
  if (const json::Value* p = s->timings.get("prefill_ms")) r.first_ms = p->as_double();
  if (const json::Value* p = s->timings.get("preempted_at"); p && p->is_array() && p->size() > 0) r.first_parked = int((*p)[0].as_int());
  return r;
}

std::vector<Result> run_alone(Engine& e, const Variant& v, int max_tokens, const std::shared_ptr<const grammar::Constraint>& c) {
  std::vector<Result> out;
  for (int i = 0; i < 4; ++i) out.push_back(drain(e.generate(request_of(e, v, i, max_tokens, c))));
  return out;
}

int64_t stat(const Engine& e, const char* k) {
  const json::Value s = e.stats();
  const json::Value* v = s.get(k);
  return v ? v->as_int() : 0;
}

std::vector<Result> run_together(Engine& e, const Variant& v, int max_tokens, const std::shared_ptr<const grammar::Constraint>& c,
                                 double* wall_ms) {
  const auto t0 = std::chrono::steady_clock::now();
  // The most requests running at once and the least free KV while they run.
  std::atomic<bool> done{false};
  int64_t most = 0, least = INT64_MAX;
  std::thread watch([&] {
    while (!done.load()) {
      most = std::max(most, stat(e, "active_sequences"));
      least = std::min(least, stat(e, "free_kv_tokens"));
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  });
  std::vector<std::shared_ptr<GenStream>> ss;
  for (int i = 0; i < 4; ++i) {
    ss.push_back(e.generate(request_of(e, v, i, max_tokens, c)));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // arrival order 0, 1, 2, 3
  }
  std::vector<Result> out;
  for (auto& s : ss) out.push_back(drain(s));
  done = true;
  watch.join();
  *wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  (at most %lld running at once, at least %lld KV tokens free)\n", (long long)most, (long long)least);
  return out;
}

// A long JSON reply: an object with at least 25 towns (the grammar enforces
// minItems, so the reply runs long enough for the pool to run out under it).
std::shared_ptr<const grammar::Constraint> towns_constraint(Engine& e) {
  const json::Value schema = json::parse(
      R"({"type": "object", "properties": {"towns": {"type": "array", "minItems": 25, "items": {"type": "object",
          "properties": {"name": {"type": "string"}, "province": {"type": "string"}, "population": {"type": "integer"},
          "note": {"type": "string"}}, "required": ["name", "province", "population", "note"]}}},
          "required": ["towns"]})");
  auto c = std::make_shared<grammar::Constraint>();
  c->compiled = std::make_shared<grammar::Compiled>(grammar::compile_schema(schema, grammar::SchemaOptions{}, nullptr),
                                                    e.grammar_vocab(), e.end_ids, std::vector<int32_t>{},
                                                    std::vector<std::string>{});
  return c;
}

int first_diff(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i)
    if (a[i] != b[i]) return int(i);
  return a.size() == b.size() ? -1 : int(n);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model dir> [max_tokens] [variant]\n", argv[0]);
    return 2;
  }
  // Admit everything (the plain free-blocks check): the pool runs out mid-generation.
  setenv("ESPETERO_KV_RESERVE", "0", 1);
  const std::string dir = argv[1];
  const int max_tokens = argc > 2 ? std::atoi(argv[2]) : 640;
  const std::string only = argc > 3 ? argv[3] : "";  // run only the variant of this name
  const Variant variants[4] = {
      {"greedy"},
      {"seeded", 0.8, 0.95},
      {"seeded+penalties", 0.8, 0.95, 1.1, 0.2, 3},
      {"greedy+structured", 0, 1, 1, 0, 0, true},
  };
  try {
    // References: each request alone on a server with room.
    std::vector<std::vector<Result>> alone;
    double tok_bytes = 0;
    int P = 0;
    double ref_wall = 0;
    {
      auto big = load(dir, 1.0, true);
      const ModelInfo& I = big->info();
      tok_bytes = double(I.kv_bytes) / double(int64_t(I.kv_blocks) * I.block_tokens);
      auto obj = towns_constraint(*big);
      for (const Variant& v : variants)
        alone.push_back(only.empty() || only == v.name ? run_alone(*big, v, max_tokens, obj) : std::vector<Result>{});
      for (int i = 0; i < 4; ++i) P = std::max(P, int(prompt_of(*big, i).size()));
      // Control: the four together with room (no preemption): what batching alone changes.
      auto ctl = run_together(*big, variants[0], max_tokens, obj, &ref_wall);
      int same = 0;
      for (int i = 0; i < 4 && !alone[0].empty(); ++i) same += first_diff(ctl[i].tokens, alone[0][i].tokens) < 0;
      std::printf("control (room for all, greedy): %d of 4 identical to alone, %.0f ms\n", same, ref_wall);
      CHECK(stat(*big, "kv_preempted_sequences") == 0);
    }
    // The plain admission check takes a request while the free KV holds its
    // prompt and up to 1,024 tokens of answer (with a block's slack each): a
    // pool of one such need plus twelve blocks admits all four as they arrive
    // (each holds up to four blocks once its prompt ran: its prompt, the
    // prompt checkpoint's copy of the partial block, its first answer block),
    // and they need about twice the pool by their last tokens.
    const int pool = P + std::min(max_tokens, 1024) + 128 + 12 * 64;
    const double kv_gb = double(pool) * tok_bytes / 1e9;
    std::printf("prompts up to %d tokens, %d max_tokens: a pool of ~%d tokens (the four need ~%d)\n", P, max_tokens, pool, 4 * (P + max_tokens));
    {
      auto small = load(dir, kv_gb, true);
      auto obj = towns_constraint(*small);
      int diverged_unparked = 0;
      for (size_t vi = 0; vi < 4; ++vi) {
        const Variant& v = variants[vi];
        if (!only.empty() && only != v.name) continue;
        // The previous variant's prompt and reply checkpoints would hold the
        // pool (the plain admission check waits for free blocks): start empty.
        small->clear_cache();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const int64_t p0 = stat(*small, "kv_preempted_sequences"), r0 = stat(*small, "kv_resume_rows");
        double wall = 0;
        auto got = run_together(*small, v, max_tokens, obj, &wall);
        const int64_t np = stat(*small, "kv_preempted_sequences") - p0, nr = stat(*small, "kv_resume_rows") - r0;
        std::printf("%s: %lld preemptions, %lld rows computed again, %.0f ms\n", v.name, (long long)np, (long long)nr, wall);
        CHECK(np > 0);
        for (int i = 0; i < 4; ++i) {
          const Result& a = alone[vi][i];
          const Result& g = got[i];
          const int d = first_diff(g.tokens, a.tokens);
          std::printf("  req %d: %s, %zu tokens (alone %zu, %s), parked %d times (first after %d tokens), %d rows again in "
                      "%.0f ms, %.0f ms parked, first token at %.0f ms%s\n",
                      i, g.reason.c_str(), g.tokens.size(), a.tokens.size(), a.reason.c_str(), g.preemptions, g.first_parked,
                      g.resume_rows, g.resume_ms, g.parked_ms, g.first_ms,
                      d < 0 ? "" : (" -- FIRST DIFFERENCE at token " + std::to_string(d)).c_str());
          if (!g.error.empty()) std::printf("    error: %s\n", g.error.c_str());
          // A difference before the request was ever parked comes from batching
          // (the control above shows how often); after it, from the resume.
          if (d >= 0 && (g.first_parked < 0 || d < g.first_parked)) {
            ++diverged_unparked;
            std::printf("    (before any preemption: batch composition, not the resume)\n");
          } else {
            CHECK(d < 0);
          }
          CHECK(g.reason == a.reason);
          CHECK(g.completion == int(g.tokens.size()));
          if (!(v.structured && i == 0)) CHECK(g.reason == "length" && int(g.tokens.size()) == max_tokens);
        }
        CHECK(got[1].preemptions == 0);  // the oldest interactive request runs through
        CHECK(got[0].preemptions > 0);  // the batch job is parked first
      }
      CHECK(stat(*small, "kv_stopped_sequences") == 0);
      std::printf("differences before any preemption (batching): %d\n", diverged_unparked);
    }
    // The old behaviour: the newest requests stop with "KV cache full".
    {
      auto old = load(dir, kv_gb, false);
      double wall = 0;
      auto got = run_together(*old, variants[0], max_tokens, nullptr, &wall);
      int stopped = 0;
      for (int i = 0; i < 4; ++i) {
        std::printf("  --no-preempt req %d: %s, %zu tokens%s%s\n", i, got[i].reason.c_str(), got[i].tokens.size(),
                    got[i].error.empty() ? "" : ": ", got[i].error.c_str());
        stopped += got[i].reason == "error" && got[i].error.find("KV cache full") != std::string::npos;
      }
      std::printf("--no-preempt: %d stopped, %.0f ms\n", stopped, wall);
      CHECK(stopped > 0);
      CHECK(stat(*old, "kv_preempted_sequences") == 0);
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL exception: %s\n", e.what());
    ++g_fails;
  }
  std::printf("preempt_test: %d checks, %d failures\n", g_checks, g_fails);
  return g_fails ? 1 : 0;
}
