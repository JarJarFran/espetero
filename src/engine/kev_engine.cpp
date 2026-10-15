// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Kev decisions: the engine's side (kev.h). A request's state runs once
// into a checkpoint slot (cached, LRU), then every question's row continues
// it as a tree of its own (TreeSpec::parent = the state), several per pass;
// each tree reads out the final-normed states of its <|fim_suffix|> and
// </opt> rows (TreeSpec::out_rows), and the pointer head runs on the
// caller's thread. The /v1/systemone body and response follow the release's
// server (kev/serve.py, kev/api.py).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <mutex>

#include "../tok/tokenizer.h"
#include "engine.h"
#include "kev.h"

namespace espetero {

using Clock = std::chrono::steady_clock;

namespace {
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
double ms3(double x) { return std::round(x * 1000.0) / 1000.0; }
bool present(const json::Value* v) {
  return v && !v->is_null() && !(v->is_array() && v->size() == 0) && !(v->is_bool() && !v->as_bool());
}
}  // namespace

struct Engine::KevJob {
  kev::Record rec;
  std::vector<PreparedImage> images;
  std::string digest;                      // image digests and salt (the state cache key with the tokens)
  std::vector<int32_t> positions;          // [state][3] M-RoPE positions (images only)
  std::vector<std::vector<float>> hidden;  // per question: [rows][hidden] final-normed states of its read-out rows
  int cached = 0;                          // state rows served from the cache
  double gpu_ms = 0, run_ms = 0;
  int passes = 0;
  // Pass arrays.
  std::vector<int32_t> root_pos;
  std::vector<ImageRef> irefs;
  Clock::time_point queued, started;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  std::string error;
  int error_status = 502;
};

json::Value Engine::kev_systemone(const json::Value& body) {
  const auto started = Clock::now();
  if (!kev_) throw RequestError(422, "this model has no Kev pointer head (pointer_head.json)");
  if (!body.is_object()) throw RequestError(422, "request body must be a JSON object");
  const json::Value* model = body.get("model");
  if (model && !model->is_null() && !model->is_string()) throw RequestError(422, "model: a string");
  const json::Value* state = body.get("state");
  if (!state) throw RequestError(422, "state is required");
  const json::Value* questions = body.get("questions");
  if (!questions || questions->is_null()) throw RequestError(422, "questions is required");
  if (present(body.get("videos"))) throw RequestError(422, "videos are not supported");
  if (present(body.get("audio"))) throw RequestError(422, "audio is not supported");
  std::vector<std::string> urls;
  if (const json::Value* ims = body.get("images"); ims && !ims->is_null()) {
    if (!ims->is_array()) throw RequestError(422, "images: a list of image URLs");
    if (ims->size() > 16) throw RequestError(422, "images: at most 16 per request");
    for (const json::Value& im : ims->as_array()) {
      std::string url;
      if (im.is_string()) {
        url = im.as_string();
      } else if (im.is_object()) {
        if (const json::Value* u = im.get("url"); u && u->is_string()) url = u->as_string();
        else if (const json::Value* iu = im.get("image_url"); iu && iu->is_object()) url = iu->get_string("url", "");
      }
      if (url.empty()) throw RequestError(422, "images: each image is a data URL string or {\"url\": ...}");
      urls.push_back(std::move(url));
    }
  }
  const ModelInfo& I = model_->info();
  if (!urls.empty() && !I.vision)
    throw RequestError(422, "images: this Kev model has no vision tower (tools/kev_convert.py --vision-from grafts one)");
  auto job = std::make_shared<KevJob>();
  // Image budget: Espetero's levels, as for a chat request to the base model (standard ~1 MP).
  const std::string detail = body.get_string("image_detail", "auto");
  const int64_t px = detail_pixels(detail);
  if (px < 0) throw RequestError(422, "image_detail: low, auto, standard, high or max");
  check_detail(px, "image_detail", detail);
  if (!urls.empty()) {  // the image runs, read from the headers, must fit the state: checked before decoding
    int64_t total = 1;
    for (const std::string& url : urls) total += 2 + std::max(1, image_tokens(url, px > 0 ? px : 0));
    if (total > kev::kMaxState)
      throw RequestError(422, "the images take " + std::to_string(total) + " tokens; a state holds at most " +
                                  std::to_string(kev::kMaxState));
  }
  for (const std::string& url : urls) job->images.push_back(prepare_image(url, px > 0 ? px : 0));
  std::vector<int> itoks;
  for (const PreparedImage& im : job->images) itoks.push_back(im.tokens(I.vision_merge));
  const auto t_enc = Clock::now();
  job->rec = kev::encode(*tok_, kev_->delimiters(), *state, *questions, itoks, vision_start, image_token, vision_end);
  const double encode_ms = ms_since(t_enc);
  if (!job->images.empty()) job->positions = mrope_positions(job->rec.state, job->images);
  for (const PreparedImage& im : job->images) job->digest += im.digest;
  if (const json::Value* salt = body.get("cache_salt"); salt && salt->is_string()) {  // no sharing across salts
    job->digest.push_back('\0');
    job->digest += salt->as_string();
  }
  for (const kev::Question& q : job->rec.questions) job->hidden.emplace_back(q.rows.size() * size_t(I.hidden), 0.0f);
  job->queued = Clock::now();
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (stop_) throw RequestError(503, "engine is shutting down");
    kev_jobs_.push_back(job);
  }
  cv_.notify_all();
  {
    std::unique_lock<std::mutex> lk(job->mu);
    job->cv.wait(lk, [&] { return job->done; });
    if (!job->error.empty()) throw RequestError(job->error_status, job->error);
  }

  // The head on this thread: logits / T, softmax over each question's options.
  const auto t_head = Clock::now();
  const double T = kev_->temperature();
  const bool want_logits = body.get_bool("return_logits", false);
  json::Value answers = json::Value::object(), raw = json::Value::object();
  for (size_t q = 0; q < job->rec.questions.size(); ++q) {
    const kev::Question& Q = job->rec.questions[q];
    const int K = int(Q.rows.size()) - 1;
    const std::vector<float> z = kev_->logits(job->hidden[q].data(), K);
    double mx = -INFINITY, sum = 0;
    for (float x : z) mx = std::max(mx, double(x) / T);
    std::vector<double> p(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) sum += (p[size_t(k)] = std::exp(double(z[size_t(k)]) / T - mx));
    for (double& x : p) x /= sum;
    answers.set(Q.id, kev::answer(Q, p));
    if (want_logits) {
      json::Value o = json::Value::object();
      for (int k = 0; k < K; ++k) o.set(Q.keys[size_t(k)], double(z[size_t(k)]));
      raw.set(Q.id, std::move(o));
    }
  }
  const double head_ms = ms_since(t_head);
  json::Value out = json::Value::object();
  out.set("model", model && model->is_string() ? model->as_string() : std::string("kev-latest"));
  // usage as the release counts it: every token the model reads, and the tokens of the serialized answers.
  json::Value usage = json::Value::object();
  usage.set("input_tokens", int64_t(job->rec.input_tokens));
  usage.set("output_tokens", int64_t(tok_->encode(json::dump_python_default(answers, true), true).size()));
  out.set("answers", std::move(answers));
  out.set("usage", std::move(usage));
  out.set("latency_ms", std::round(job->run_ms * 10.0) / 10.0);
  if (want_logits) out.set("logits", std::move(raw));
  json::Value metal = json::Value::object();
  metal.set("mode", "kev_pointer_head");
  metal.set("temperature", T);
  metal.set("elapsed_ms", ms3(ms_since(started)));
  metal.set("queue_ms", ms3(std::chrono::duration<double, std::milli>(job->started - job->queued).count()));
  metal.set("backbone_ms", ms3(job->run_ms));
  metal.set("gpu_ms", ms3(job->gpu_ms));
  metal.set("head_ms", ms3(head_ms));
  metal.set("encode_ms", ms3(encode_ms));
  metal.set("passes", job->passes);
  metal.set("state_tokens", int64_t(job->rec.state.size()));
  metal.set("cached_state_tokens", job->cached);
  metal.set("images", int64_t(job->images.size()));
  int64_t image_tokens = 0;
  for (int n : itoks) image_tokens += n;
  metal.set("image_tokens", image_tokens);
  out.set("metal", std::move(metal));
  decisions_done_ += 1;
  return out;
}

bool Engine::kev_evict_state(const std::vector<int>& keep) {
  auto lru = kev_states_.end();
  for (auto it = kev_states_.begin(); it != kev_states_.end(); ++it)
    if (std::find(keep.begin(), keep.end(), it->slot) == keep.end() && (lru == kev_states_.end() || it->used < lru->used))
      lru = it;
  if (lru == kev_states_.end()) return false;
  model_->slot_free(lru->slot);
  kev_states_.erase(lru);
  return true;
}

void Engine::kev_room(const std::vector<TreeSpec>& trees, const std::vector<int>& keep) {
  const int need = model_->pass_kv_tokens(trees);
  while (model_->free_kv_tokens() < need && kev_evict_state(keep)) {
  }
  if (model_->free_kv_tokens() < need)
    throw RequestError(503, "the KV cache cannot hold this Kev request (" + std::to_string(need) + " tokens; raise --kv-gb)");
}

void Engine::kev_clear() {
  for (KevState& c : kev_states_) model_->slot_free(c.slot);
  kev_states_.clear();
}

// Scheduler thread: the state (unless cached) in passes of at most max_rows
// rows (an image run never straddles two), saved in place into one slot that
// joins the cache; then the question rows on top of it, as many whole rows
// per pass as max_rows takes (a longer row runs in several passes through a
// slot of its own).
void Engine::run_kev(KevJob& j) {
  j.started = Clock::now();
  const ModelInfo& I = model_->info();
  const size_t D = size_t(I.hidden);
  const std::vector<int32_t>& S = j.rec.state;
  const int Ls = int(S.size()), budget = std::max(64, I.max_rows);
  for (const kev::Question& q : j.rec.questions)
    if (Ls + int(q.ids.size()) > I.max_ctx)
      throw RequestError(422, "a question's row (state + question) is " + std::to_string(Ls + int(q.ids.size())) +
                                  " tokens, over the context window of " + std::to_string(I.max_ctx) + " (raise --max-ctx)");
  auto account = [&](const PassStats& st) {
    j.gpu_ms += st.gpu_ms;
    ++j.passes;
    ++passes_;
    rows_ += uint64_t(st.rows);
    gpu_ms_ = gpu_ms_ + st.gpu_ms;
  };
  auto new_slot = [&](const std::vector<int>& keep) {
    int s = model_->slot_alloc();
    while (s < 0 && kev_evict_state(keep)) s = model_->slot_alloc();
    if (s < 0) throw RequestError(503, "no free model slot for a Kev request (raise --slots)");
    return s;
  };

  // ---- the state
  int state = -1;
  for (KevState& c : kev_states_)
    if (c.digest == j.digest && c.tokens == S) {
      c.used = ++clock_;
      state = c.slot;
      j.cached = Ls;
      break;
    }
  if (state < 0) {
    std::vector<std::pair<int, int>> runs;  // image rows [first, last) per image
    for (int i = 0; i < Ls && runs.size() < j.images.size();) {
      if (S[size_t(i)] == image_token) {
        const int n = j.images[runs.size()].tokens(I.vision_merge);
        runs.emplace_back(i, i + n);
        i += n;
      } else {
        ++i;
      }
    }
    if (runs.size() != j.images.size()) throw RequestError(422, "image runs do not match the images");
    const int slot = new_slot({});
    try {
      size_t g = 0;
      for (int done = 0; done < Ls;) {
        int end = std::min(Ls, done + budget);
        for (const auto& [a, b] : runs)
          if (a < end && end < b) {
            end = a > done ? a : b;  // stop before the image, or take the whole image when it starts the pass
            break;
          }
        if (end - done > I.max_rows) throw RequestError(422, "an image needs more rows than one pass holds (raise --max-rows)");
        TreeSpec t;
        t.parent = done > 0 ? slot : -1;
        t.save = slot;
        t.root = S.data() + done;
        t.root_len = end - done;
        if (!j.positions.empty()) {
          j.root_pos.assign(j.positions.begin() + long(done) * 3, j.positions.begin() + long(end) * 3);
          t.root_pos = j.root_pos.data();
        }
        j.irefs.clear();
        for (; g < runs.size() && runs[g].first < end; ++g) {
          const PreparedImage& im = j.images[g];
          ImageRef ref;
          ref.pixels = im.pixels.data();
          ref.grid_t = im.grid_t;
          ref.grid_h = im.grid_h;
          ref.grid_w = im.grid_w;
          ref.row = runs[g].first - done;
          j.irefs.push_back(ref);
        }
        if (!j.irefs.empty()) {
          t.images = j.irefs.data();
          t.n_images = int(j.irefs.size());
        }
        kev_room({t}, {slot});
        PassStats st;
        model_->run({t}, &st);
        account(st);
        done = end;
      }
    } catch (...) {
      model_->slot_free(slot);
      throw;
    }
    KevState c;
    c.tokens = S;
    c.digest = j.digest;
    c.slot = slot;
    c.used = ++clock_;
    kev_states_.push_back(std::move(c));
    while (kev_states_.size() > kev_state_cap_ && kev_evict_state({slot})) {
    }
    state = slot;
  }

  // ---- the questions: rows continuing the state
  const size_t Q = j.rec.questions.size();
  for (size_t q = 0; q < Q;) {
    std::vector<TreeSpec> trees;
    int rows = 0;
    for (; q < Q; ++q) {
      const kev::Question& X = j.rec.questions[q];
      const int n = int(X.ids.size());
      if (n > budget || (!trees.empty() && rows + n > budget)) break;
      TreeSpec t;
      t.parent = state;
      t.root = X.ids.data();
      t.root_len = n;
      t.out_rows = X.rows.data();
      t.n_out_rows = int(X.rows.size());
      t.out_rows_hidden = j.hidden[q].data();
      trees.push_back(t);
      rows += n;
    }
    if (!trees.empty()) {
      kev_room(trees, {state});
      PassStats st;
      model_->run(trees, &st);
      account(st);
      continue;
    }
    // A row longer than a pass: its parts continue a slot of their own.
    const kev::Question& X = j.rec.questions[q];
    const int n = int(X.ids.size());
    const int tmp = new_slot({state});
    try {
      std::vector<int32_t> local;
      std::vector<size_t> which;
      std::vector<float> part;
      for (int done = 0; done < n;) {
        const int end = std::min(n, done + budget);
        TreeSpec t;
        t.parent = done == 0 ? state : tmp;
        if (end < n) t.save = tmp;
        t.root = X.ids.data() + done;
        t.root_len = end - done;
        local.clear();
        which.clear();
        for (size_t i = 0; i < X.rows.size(); ++i)
          if (X.rows[i] >= done && X.rows[i] < end) {
            local.push_back(X.rows[i] - done);
            which.push_back(i);
          }
        part.assign(local.size() * D, 0.0f);
        if (!local.empty()) {
          t.out_rows = local.data();
          t.n_out_rows = int(local.size());
          t.out_rows_hidden = part.data();
        }
        kev_room({t}, {state, tmp});
        PassStats st;
        model_->run({t}, &st);
        account(st);
        for (size_t i = 0; i < which.size(); ++i)
          std::copy(part.begin() + long(i * D), part.begin() + long((i + 1) * D), j.hidden[q].begin() + long(which[i] * D));
        done = end;
      }
    } catch (...) {
      model_->slot_free(tmp);
      throw;
    }
    model_->slot_free(tmp);
    ++q;
  }
  j.run_ms = ms_since(j.started);
}

bool Engine::kev_step() {
  std::shared_ptr<KevJob> j;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (kev_jobs_.empty()) return false;
    j = kev_jobs_.front();
    kev_jobs_.pop_front();
  }
  std::string err;
  int status = 502;
  try {
    run_kev(*j);
  } catch (const RequestError& e) {
    err = e.what();
    status = e.status;
  } catch (const std::exception& e) {
    err = e.what();
  }
  if (!err.empty()) std::fprintf(stderr, "[espetero] kev request failed: %s\n", err.c_str());
  {
    std::lock_guard<std::mutex> lk(j->mu);
    j->error = err;
    j->error_status = status;
    j->done = true;
  }
  j->cv.notify_all();
  return true;
}

void Engine::kev_fail_all(const std::string& why) {
  for (auto& j : kev_jobs_) {
    std::lock_guard<std::mutex> jl(j->mu);
    j->error = why;
    j->error_status = 503;
    j->done = true;
    j->cv.notify_all();
  }
  kev_jobs_.clear();
}

}  // namespace espetero
