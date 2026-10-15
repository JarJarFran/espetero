// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Francisco José García Barbero
//
// Derived from source/src/autojev/model.py of Perplexity's pplx-decider
// release (https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b),
// licensed under the Apache License, Version 2.0; translated to C++ and
// modified (see THIRD_PARTY_NOTICES.md). This file is licensed under the
// Apache License, Version 2.0 (third_party/apache-2.0/LICENSE-2.0.txt).

// /v1/systemone on pplx-decider checkpoints (pplx.h), apart from pplx.cpp so
// the prompt and answer code links without the engine (tests/gemma_test.cpp).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>

#include "chat.h"
#include "engine.h"
#include "pplx.h"

namespace espetero::pplx {

namespace {

[[noreturn]] void invalid(const std::string& msg) { throw RequestError(422, msg); }

double ms_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

double round3(double x) { return std::round(x * 1000.0) / 1000.0; }

}  // namespace

json::Value systemone(Engine& engine, const json::Value& body) {
  const auto started = std::chrono::steady_clock::now();
  const std::vector<std::string>& codes = engine.head_codes();
  if (codes.empty()) invalid("this model's head has no answer codes (hidden_head.json \"codes\"): /v1/systemone serves pplx-decider checkpoints");
  if (!body.is_object()) invalid("request body must be a JSON object");
  const json::Value* model = body.get("model");
  if (model && !model->is_null() && !model->is_string()) invalid("model: a string");
  const json::Value* state = body.get("state");
  if (!state) invalid("model and state are required");
  const json::Value* questions = body.get("questions");
  if (!questions || !questions->is_object() || questions->size() == 0)
    invalid("questions: an object with at least one question");
  for (const char* k : {"videos", "audio"})
    if (const json::Value* v = body.get(k); v && !v->is_null() && !(v->is_array() && v->size() == 0))
      invalid(std::string(k) + " are not supported for pplx-decider models");
  json::Value images = json::Value::array();
  if (const json::Value* ims = body.get("images"); ims && !ims->is_null()) {
    if (!ims->is_array()) invalid("images: a list of image URLs");
    if (ims->size() > 16) invalid("images: at most 16 per request");
    for (const json::Value& u : ims->as_array())
      if (!u.is_string()) invalid("images: a list of image URLs");
    images = *ims;
  }
  // low: the 512 x 512 = 262,144-pixel budget the release trained and was evaluated at.
  const std::string detail = body.get_string("image_detail", "low");

  // Render every question first (422s before any pass), then run them as concurrent head requests.
  std::vector<Decision> ds;
  std::vector<ChatJob> jobs;
  const json::Object& qs = questions->as_object();
  for (size_t i = 0; i < qs.keys.size(); ++i) {
    ds.push_back(encode(codes, *state, qs.keys[i], qs.vals[i], images, detail));
    jobs.push_back(prepare_chat(engine, ds.back().chat));
  }
  std::vector<std::future<Engine::HeadResult>> runs;
  for (const ChatJob& j : jobs)
    runs.push_back(std::async(std::launch::async, [&engine, &j] { return engine.head(j.gen, j.head_slots); }));
  std::vector<Engine::HeadResult> results;
  for (auto& f : runs) results.push_back(f.get());

  const bool want_logits = body.get_bool("return_logits", false);
  const double temperature = engine.head_temperature();
  json::Value answers = json::Value::object(), raw = json::Value::object();
  int64_t tokens = 0;
  double gpu_ms = 0, queue_ms = 0;
  int passes = 0;
  for (size_t q = 0; q < ds.size(); ++q) {
    const Decision& d = ds[q];
    const Engine::HeadResult& r = results[q];
    tokens += int64_t(jobs[q].gen.prompt.size());
    gpu_ms += r.gpu_ms, passes += r.passes, queue_ms = std::max(queue_ms, r.queue_ms);
    // softmax(logits / temperature) over the question's options (the release masks the other codes).
    std::vector<double> x(r.logits.size()), p(r.logits.size());
    for (size_t k = 0; k < x.size(); ++k) x[k] = double(r.logits[k]) / temperature;
    const double mx = *std::max_element(x.begin(), x.end());
    double sum = 0;
    for (size_t k = 0; k < x.size(); ++k) sum += (p[k] = std::exp(x[k] - mx));
    for (double& v : p) v /= sum;
    answers.set(d.id, answer(d, p));
    if (want_logits) {  // calibrated: already divided by the checkpoint's temperature
      json::Value o = json::Value::object();
      for (size_t k = 0; k < x.size(); ++k) o.set(d.keys[k], x[k]);
      raw.set(d.id, std::move(o));
    }
  }
  json::Value out = json::Value::object();
  out.set("model", model && model->is_string() ? model->as_string() : engine.model_name());
  out.set("answers", std::move(answers));
  if (want_logits) out.set("logits", std::move(raw));
  json::Value usage = json::Value::object();
  usage.set("input_tokens", tokens);
  usage.set("output_tokens", 0);
  out.set("usage", std::move(usage));
  json::Value metal = json::Value::object();
  metal.set("mode", "pplx_readout");
  metal.set("temperature", temperature);
  metal.set("images", int64_t(images.size()));
  metal.set("elapsed_ms", round3(ms_since(started)));
  metal.set("queue_ms", round3(queue_ms));
  metal.set("gpu_ms", round3(gpu_ms));
  metal.set("passes", passes);
  out.set("metal", std::move(metal));
  return out;
}

}  // namespace espetero::pplx
