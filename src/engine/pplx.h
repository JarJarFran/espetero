// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Francisco José García Barbero
//
// Derived from source/src/autojev/model.py of Perplexity's pplx-decider
// release (https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b),
// licensed under the Apache License, Version 2.0; translated to C++ and
// modified (see THIRD_PARTY_NOTICES.md). This file is licensed under the
// Apache License, Version 2.0 (third_party/apache-2.0/LICENSE-2.0.txt).

// /v1/systemone for pplx-decider checkpoints (a hidden-state head with answer
// codes, hidden_head.json "codes"; tools/pplx_decider_convert.py). Each
// question is one decision, rendered like the release's decision_messages():
// the system prompt, then one user turn with the request's images and
// "State: ... Question: ... Options: A: ... Return only the letter code of the
// best option.", through the chat template with thinking off. It runs as a
// head request of n slots (Engine::head: noncausal, uncached, one pass), and
// softmax(logits / temperature) over its n options becomes the release's
// typed answer: noul P(true), a choice and its confidence, or a score.
#pragma once

#include <string>
#include <vector>

#include "../util/json.h"

namespace espetero {

class Engine;

namespace pplx {

struct Decision {
  std::string id;                         // question key of the request
  json::Value question;                   // as requested
  std::vector<std::string> keys;          // option keys (choice: criteria keys; score: "0".."n-1"; noul: false, true)
  std::vector<json::Value> descriptions;  // as the release's options() gives them
  json::Value chat;                       // the /v1/chat/completions body of its head request
};

// Validates one question (throws RequestError 422) and renders it. images:
// the request's image URLs, each one image part before the text at `detail`.
Decision encode(const std::vector<std::string>& codes, const json::Value& state, const std::string& id,
                const json::Value& question, const json::Value& images, const std::string& detail);

// The release's answer(): the typed answer of one question from its probabilities.
json::Value answer(const Decision& d, const std::vector<double>& probs);

// POST /v1/systemone (and /v1/decisions) on a model with head answer codes:
// {"state", "questions": {id: {type, instructions, criteria}}, "images",
// "image_detail" (low: the release's 512 x 512 budget), "return_logits"}
// -> {"answers", "logits"?, "usage", "metal"}. Blocking, thread-safe.
json::Value systemone(Engine& engine, const json::Value& body);

}  // namespace pplx
}  // namespace espetero
