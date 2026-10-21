// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Engine-level checks on real checkpoints (GPU): what a decision model and a
// chat model refuse, with the exact texts the routes send, and the d1 image
// run guard. Each directory is optional; a missing argument skips its checks.
//
//   build/engine_test --chat <Qwen3.5 dir> --d1 <d1 dir> --clef <Clef dir>
//
//   --chat: a chat model (no Clef head, no d1): /v1/systemone gets the 422 of
//           main ("this model has no Clef joint head ..."), chat is served.
//   --d1:   a d1 release: chat refused with the d1 text; /v1/systemone answers;
//           a state that holds an image run without an image gets a 422.
//   --clef: a Clef release: chat refused with the Clef text, unchanged.
#include <cstdio>
#include <cstring>
#include <string>

#include "../src/engine/engine.h"
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

const char* kClefRefusal = "this is a Clef decision model: it answers /v1/systemone and /v1/decisions, not chat";
const char* kD1Refusal = "this is a d1 decision model: it answers /v1/systemone and /v1/decisions, not chat";
const char* kNoClefHead = "this model has no Clef joint head (joint_head.safetensors); /v1/systemone serves Clef releases";

std::unique_ptr<Engine> load(const std::string& dir) {
  EngineOptions o;
  o.model_path = dir;
  o.max_seqs = 2;
  o.load.slots = 4;
  o.load.max_ctx = 4096;
  o.load.kv_gb = 1;
  std::printf("loading %s\n", dir.c_str());
  return std::make_unique<Engine>(o);
}

json::Value noul_request(const std::string& state, bool with_question = true) {
  json::Value body = json::Value::object();
  body.set("state", state);
  json::Value qs = json::Value::object();
  if (with_question) {
    json::Value q = json::Value::object();
    q.set("type", std::string("noul"));
    q.set("instructions", std::string("Does the patient have a fever?"));
    qs.set("fever", std::move(q));
  }
  body.set("questions", std::move(qs));
  return body;
}

// systemone(body): the RequestError's status and text, or 0 and "" when it answers.
std::pair<int, std::string> systemone_error(Engine& e, const json::Value& body) {
  try {
    e.systemone(body);
  } catch (const RequestError& err) {
    return {err.status, err.what()};
  }
  return {0, ""};
}

void chat_model(const std::string& dir) {
  auto e = load(dir);
  CHECK(!e->decision_only());
  CHECK(!e->is_clef() && !e->is_d1());
  const auto [status, text] = systemone_error(*e, noul_request("temperature 38.9 C"));
  CHECK(status == 422);
  CHECK(text == kNoClefHead);
  if (text != kNoClefHead) std::printf("  got: %s\n", text.c_str());
}

void d1_model(const std::string& dir) {
  auto e = load(dir);
  CHECK(e->decision_only() && e->is_d1() && !e->is_clef());
  CHECK(std::string(e->chat_refusal()) == kD1Refusal);
  // A state whose tokens hold an image run (<|image_start|>) without an image: 422, not a crash.
  const auto [s1, t1] = systemone_error(*e, noul_request("Note: <|image_start|> temperature 38.9 C"));
  CHECK(s1 == 422);
  CHECK(t1.find("image runs do not match the images") != std::string::npos);
  if (s1 != 422) std::printf("  got status %d: %s\n", s1, t1.c_str());
  // The same request without the run answers.
  json::Value ans = e->systemone(noul_request("temperature 38.9 C"));
  const json::Value* a = ans.get("answers");
  const json::Value* f = a ? a->get("fever") : nullptr;
  CHECK(f && f->get("noul") && f->get("noul")->is_number());
  if (f && f->get("noul")) std::printf("  d1 answers.fever.noul = %.4f\n", f->get("noul")->as_double());
}

void clef_model(const std::string& dir) {
  auto e = load(dir);
  CHECK(e->decision_only() && e->is_clef() && !e->is_d1());
  CHECK(std::string(e->chat_refusal()) == kClefRefusal);
}

}  // namespace

int main(int argc, char** argv) {
  std::string chat, d1, clef;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--chat")) chat = argv[i + 1];
    else if (!std::strcmp(argv[i], "--d1")) d1 = argv[i + 1];
    else if (!std::strcmp(argv[i], "--clef")) clef = argv[i + 1];
    else {
      std::fprintf(stderr, "usage: %s [--chat DIR] [--d1 DIR] [--clef DIR]\n", argv[0]);
      return 2;
    }
  }
  if (chat.empty() && d1.empty() && clef.empty()) {
    std::fprintf(stderr, "usage: %s [--chat DIR] [--d1 DIR] [--clef DIR]\n", argv[0]);
    return 2;
  }
  try {
    if (!chat.empty()) chat_model(chat);
    if (!d1.empty()) d1_model(d1);
    if (!clef.empty()) clef_model(clef);
  } catch (const std::exception& e) {
    std::printf("  FAIL exception: %s\n", e.what());
    ++g_fails;
  }
  std::printf("engine_test: %d checks, %d failures\n", g_checks, g_fails);
  return g_fails ? 1 : 0;
}
