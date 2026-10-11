// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Embeddings: the engine's side (a queue the scheduler serves one pass
// per step, as the Clef port reads hidden states: TreeSpec::out_hidden, no LM
// head) and the OpenAI /v1/embeddings request and response.
#include "embed.h"

#include <xlocale.h>

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../tok/tokenizer.h"
#include "requests.h"

namespace espetero {

using Clock = std::chrono::steady_clock;

namespace {
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
double round3(double x) { return std::round(x * 1000.0) / 1000.0; }
}  // namespace

// ---------------------------------------------------------------- engine
struct Engine::EmbedJob {
  std::vector<std::vector<int32_t>> inputs;
  bool mean = false;
  std::vector<std::vector<double>> acc;  // per input being pooled: the sum of its rows (mean) or its last row
  std::vector<std::vector<float>> out;   // finished vectors
  size_t next = 0;                       // the first input not finished
  int offset = 0;                        // rows of inputs[next] computed so far (an input longer than a pass)
  int slot = -1;                         // inputs[next]'s state meanwhile
  int passes = 0, rows = 0;
  double gpu_ms = 0;
  bool running = false;
  Clock::time_point queued, started;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  std::string error;
  int error_status = 502;
};

std::vector<std::vector<float>> Engine::embed(std::vector<std::vector<int32_t>> inputs, EmbedStats* stats) {
  const ModelInfo& I = model_->info();
  if (!I.hidden_states)
    throw RequestError(422, "embeddings: this model's hidden states cannot be read out yet (Qwen3.8-Flash-Next)");
  if (inputs.empty()) throw RequestError(422, "input: no input");
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].empty()) throw RequestError(422, "input[" + std::to_string(i) + "]: no tokens");
    if (int64_t(inputs[i].size()) > I.max_ctx)
      throw RequestError(422, "input[" + std::to_string(i) + "]: " + std::to_string(inputs[i].size()) +
                                  " tokens exceed the model context window (" + std::to_string(I.max_ctx) + " tokens)");
    for (int32_t t : inputs[i])
      if (t < 0 || t >= I.vocab) throw RequestError(422, "input[" + std::to_string(i) + "]: token id out of range");
  }
  auto job = std::make_shared<EmbedJob>();
  job->mean = opt_.embedding_pooling == "mean";
  job->acc.resize(inputs.size());
  job->out.resize(inputs.size());
  job->inputs = std::move(inputs);
  job->queued = Clock::now();
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (stop_) throw RequestError(503, "engine is shutting down");
    embed_jobs_.push_back(job);
  }
  cv_.notify_all();
  {
    std::unique_lock<std::mutex> lk(job->mu);
    job->cv.wait(lk, [&] { return job->done; });
  }
  if (!job->error.empty()) throw RequestError(job->error_status, job->error);
  if (stats) {
    stats->passes = job->passes;
    stats->rows = job->rows;
    stats->gpu_ms = job->gpu_ms;
    stats->queue_ms = std::chrono::duration<double, std::milli>(job->started - job->queued).count();
    stats->run_ms = ms_since(job->started);
  }
  return std::move(job->out);
}

// Scheduler thread, once per step while jobs wait: one pass of the oldest
// job. Its inputs fill the pass in order; an input that does not fit what is
// left of the pass waits for the next one, an input longer than a pass is
// computed in several, its state kept in a slot between them. While
// sequences decode, a pass holds at most --prefill-chunk rows (as prompt
// chunks then do), so they wait at most one such pass per step.
void Engine::embed_step() {
  std::shared_ptr<EmbedJob> j;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (embed_jobs_.empty()) return;
    j = embed_jobs_.front();
  }
  if (!j->running) {
    j->running = true;
    j->started = Clock::now();
  }
  const ModelInfo& I = model_->info();
  const size_t D = size_t(I.hidden);
  int budget = I.max_rows;
  if (!active_.empty()) budget = std::min(budget, std::max(64, opt_.prefill_chunk));
  struct Part {
    size_t input;
    int from, n, row;
  };
  std::vector<Part> parts;
  std::vector<TreeSpec> trees;
  std::vector<int32_t> pad;
  std::string err;
  int status = 502;
  try {
    int rows = 0;
    for (size_t i = j->next; i < j->inputs.size() && rows < budget; ++i) {
      const int len = int(j->inputs[i].size());
      const int from = i == j->next ? j->offset : 0;
      const int left = len - from;
      if (left > budget - rows && !parts.empty() && left <= budget) break;  // whole, in the next pass
      const int n = std::min(left, budget - rows);
      TreeSpec t;
      t.parent = from > 0 ? j->slot : -1;
      if (from + n < len) {  // the rest in later passes: the state stays in the slot
        if (j->slot < 0) j->slot = acquire_slot({});
        if (j->slot < 0) throw RequestError(503, "embeddings: no free state slot for an input longer than a pass (raise --slots)");
        t.save = j->slot;
      }
      t.root = j->inputs[i].data() + from;
      t.root_len = n;
      trees.push_back(t);
      parts.push_back({i, from, n, rows});
      rows += n;
      if (from + n < len) break;  // a cut input ends the pass
    }
    if (embed_hidden_.size() < size_t(rows) * D) embed_hidden_.resize(size_t(rows) * D);
    for (size_t k = 0; k < trees.size(); ++k) trees[k].out_hidden = embed_hidden_.data() + size_t(parts[k].row) * D;
    // A pass of at most 8 rows would take the few-row kernels (GEMVs, fused
    // MoE routing and norms), whose sums round differently from the many-row
    // ones: a padding tree (read out by nobody) keeps every pass on the
    // many-row kernels.
    if (rows <= 8) {
      pad.assign(size_t(9), j->inputs[j->next][0]);
      TreeSpec t;
      t.root = pad.data();
      t.root_len = int(pad.size());
      trees.push_back(t);
    }
    // KV room for a long input's rows: evict checkpoints nobody pins (LRU), as a step's pass does
    // (only when that makes the pass fit; one that cannot fails as before, the checkpoints kept).
    const int need = model_->pass_kv_tokens(trees);
    if (need > model_->free_kv_tokens() && need <= model_->free_kv_tokens() + evictable_kv_tokens({}, true))
      while (need > model_->free_kv_tokens()) {
        const int victim = evict_lru({});
        if (victim < 0) break;
        model_->slot_free(victim);
      }
    PassStats st;
    // Every embedding pass on the same GEMM tiles (the tuned tile depends on
    // the pass's row count): an input's vector is the same alone, in a batch
    // or cut into passes. Restored before any other pass runs.
    model_->set_option("gemm_fixed", 1);
    try {
      model_->run(trees, &st);
    } catch (...) {
      model_->set_option("gemm_fixed", 0);
      throw;
    }
    model_->set_option("gemm_fixed", 0);
    ++passes_;
    rows_ += uint64_t(st.rows);
    gpu_ms_ = gpu_ms_ + st.gpu_ms;
    ++j->passes;
    j->rows += rows;
    j->gpu_ms += st.gpu_ms;
    for (const Part& p : parts) {
      std::vector<double>& a = j->acc[p.input];
      const float* h = embed_hidden_.data() + size_t(p.row) * D;
      if (a.empty()) a.assign(D, 0.0);
      if (j->mean) {
        for (int r = 0; r < p.n; ++r)
          for (size_t d = 0; d < D; ++d) a[d] += double(h[size_t(r) * D + d]);
      } else {
        for (size_t d = 0; d < D; ++d) a[d] = double(h[size_t(p.n - 1) * D + d]);  // the last row so far
      }
      const int len = int(j->inputs[p.input].size());
      if (p.from + p.n < len) {
        j->next = p.input;
        j->offset = p.from + p.n;
        continue;
      }
      // Pooled (the mean of every row, or the last row), then L2-normalized.
      const double scale = j->mean ? 1.0 / double(len) : 1.0;
      double norm = 0;
      for (size_t d = 0; d < D; ++d) norm += (a[d] * scale) * (a[d] * scale);
      norm = std::sqrt(norm);
      std::vector<float>& v = j->out[p.input];
      v.resize(D);
      for (size_t d = 0; d < D; ++d) v[d] = norm > 0 ? float(a[d] * scale / norm) : 0.0f;
      std::vector<double>().swap(a);
      j->next = p.input + 1;
      j->offset = 0;
      if (p.from > 0 && j->slot >= 0) {  // a long input's last pass read its slot
        model_->slot_free(j->slot);
        j->slot = -1;
      }
    }
  } catch (const RequestError& e) {
    err = e.what();
    status = e.status;
  } catch (const std::exception& e) {
    err = std::string("embeddings: ") + e.what();
  }
  if (err.empty() && j->next < j->inputs.size()) return;
  if (j->slot >= 0) {
    model_->slot_free(j->slot);
    j->slot = -1;
  }
  bool idle = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    embed_jobs_.pop_front();  // j: only this thread removes jobs
    idle = embed_jobs_.empty();
  }
  if (idle) std::vector<float>().swap(embed_hidden_);  // up to max_rows x hidden floats
  if (!err.empty()) std::fprintf(stderr, "[espetero] embeddings failed: %s\n", err.c_str());
  {
    std::lock_guard<std::mutex> lk(j->mu);
    j->error = err;
    j->error_status = status;
    j->done = true;
  }
  j->cv.notify_all();
}

// With mu_ held (the scheduler's end).
void Engine::embed_fail_all(const std::string& why) {
  for (auto& j : embed_jobs_) {
    std::lock_guard<std::mutex> jl(j->mu);
    j->error = why;
    j->error_status = 503;
    j->done = true;
    j->cv.notify_all();
  }
  embed_jobs_.clear();
}

// ---------------------------------------------------------------- API
namespace {

// A float as the shortest decimal that reads back as it (at most 9
// significant digits), in a double: the JSON shows those digits only.
double short_double(float f) {
  char buf[48];
  auto r = std::to_chars(buf, buf + sizeof(buf) - 1, f);
  *r.ptr = 0;
  static const locale_t c_locale = newlocale(LC_ALL_MASK, "C", nullptr);
  return strtod_l(buf, nullptr, c_locale);
}

std::string base64(const uint8_t* p, size_t n) {
  static const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((n + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 2 < n; i += 3) {
    const uint32_t v = uint32_t(p[i]) << 16 | uint32_t(p[i + 1]) << 8 | p[i + 2];
    out += a[v >> 18], out += a[(v >> 12) & 63], out += a[(v >> 6) & 63], out += a[v & 63];
  }
  if (i < n) {
    const uint32_t v = uint32_t(p[i]) << 16 | (i + 1 < n ? uint32_t(p[i + 1]) << 8 : 0u);
    out += a[v >> 18], out += a[(v >> 12) & 63];
    out += i + 1 < n ? a[(v >> 6) & 63] : '=';
    out += '=';
  }
  return out;
}

}  // namespace

json::Value embeddings_response(Engine& engine, const json::Value& body) {
  const auto t0 = Clock::now();
  const ModelInfo& I = engine.info();
  EmbeddingRequest req = parse_embedding_request(body, I.vocab, I.hidden);
  std::vector<std::vector<int32_t>> inputs = std::move(req.tokens);
  if (inputs.empty()) {
    // Text: 16 bytes per token of the context window per input (as prompts),
    // and 8 windows' worth in all, before anything is tokenized.
    size_t bytes = 0;
    for (size_t i = 0; i < req.texts.size(); ++i) {
      engine.check_text_size(req.texts[i].size(), "input[" + std::to_string(i) + "]");
      bytes += req.texts[i].size();
    }
    const size_t limit = size_t(std::max(1, I.max_ctx)) * 16 * 8;
    if (bytes > limit)
      throw RequestError(413, "input: " + std::to_string(bytes) + " bytes of text in all exceed the limit of " +
                                  std::to_string(limit) + " (8 context windows per request)");
    for (size_t i = 0; i < req.texts.size(); ++i) {
      inputs.push_back(engine.tokenizer().with_bos(engine.tokenizer().encode(req.texts[i])));
      if (inputs.back().empty()) throw RequestError(422, "input[" + std::to_string(i) + "]: no tokens");
    }
  }
  size_t tokens = 0;
  for (const auto& in : inputs) tokens += in.size();
  const size_t cap = size_t(std::max(1, I.max_ctx)) * 8;
  if (tokens > cap)
    throw RequestError(413, "input: " + std::to_string(tokens) + " tokens in all, above " + std::to_string(cap) +
                                " (8 context windows per request)");
  Engine::EmbedStats st;
  std::vector<std::vector<float>> vecs = engine.embed(std::move(inputs), &st);
  json::Value data = json::Value::array();
  for (size_t i = 0; i < vecs.size(); ++i) {
    json::Value e = json::Value::object();
    e.set("object", "embedding");
    e.set("index", int64_t(i));
    if (req.base64) {  // little-endian float32 (OpenAI's base64 encoding)
      e.set("embedding", base64(reinterpret_cast<const uint8_t*>(vecs[i].data()), vecs[i].size() * sizeof(float)));
    } else {
      json::Value v = json::Value::array();
      v.as_array().reserve(vecs[i].size());
      for (float x : vecs[i]) v.push_back(short_double(x));
      e.set("embedding", std::move(v));
    }
    data.push_back(std::move(e));
  }
  json::Value usage = json::Value::object();
  usage.set("prompt_tokens", int64_t(tokens));
  usage.set("total_tokens", int64_t(tokens));
  json::Value timings = json::Value::object();
  timings.set("pooling", engine.options().embedding_pooling);
  timings.set("passes", st.passes);
  timings.set("rows", st.rows);
  timings.set("gpu_ms", round3(st.gpu_ms));
  timings.set("queue_ms", round3(st.queue_ms));
  timings.set("elapsed_ms", round3(ms_since(t0)));
  json::Value out = json::Value::object();
  out.set("object", "list");
  out.set("data", std::move(data));
  out.set("model", engine.model_name());
  out.set("usage", std::move(usage));
  out.set("timings", std::move(timings));
  return out;
}

}  // namespace espetero
