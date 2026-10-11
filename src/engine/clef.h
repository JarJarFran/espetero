// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Francisco José García Barbero
//
// Derived from joint_schema_model.py of Cloudflare's Clef release
// (https://huggingface.co/Cloudflare/clef-flash), licensed under the Apache
// License, Version 2.0; translated to C++ and modified (see THIRD_PARTY_NOTICES.md).
// This file is licensed under the Apache License, Version 2.0
// (third_party/apache-2.0/LICENSE-2.0.txt).

// Clef joint schema heads (Cloudflare Clef / Clef-Flash): typed
// decisions read from the backbone's final hidden states instead of the LM
// head. A request (the Jev / SystemOne `/v1/systemone` body: state, questions,
// optional images) is encoded token for token like the release's
// `encode_record` (joint_schema_model.py), the backbone runs the whole
// sequence once (no LM head, no decoding), and a small transformer head
// scores every allowed option of every question jointly. The head runs on
// the CPU in fp32 (Accelerate); the release runs it in bf16.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../util/json.h"

namespace espetero {

class Tokenizer;

namespace clef {

// True when `dir` holds a Clef release (joint_head_config.json next to the backbone).
bool is_release(const std::string& dir);

struct Question {
  std::string id;
  std::string type;                         // noul | choice | score
  int type_id = 0;                          // 0 noul, 1 choice, 2 score (the head's type embedding)
  int q0 = 0, q1 = 0;                       // instruction span, rows of the full sequence
  std::vector<std::pair<int, int>> spans;   // option spans (the option's JSON semantics)
  std::vector<std::string> option_ids;      // in the release's order (noul: true, false; choice: sorted ids)
  json::Value criteria;                     // as requested (choice: object, score: array, noul: object or null)
};

struct Record {
  std::vector<int32_t> ids;                 // the whole input, image runs expanded
  std::vector<Question> questions;
  int prefix_tokens = 0;                    // system prompt + "STATE:\n" + media block: the same for every
                                            // request about the same images (cached by the engine)
  int state_tokens = 0, truncated_state_tokens = 0;
};

// Validates a SystemOne request's questions (throws RequestError 422) and
// encodes the record. image_tokens: token count of each image, in order
// (the media block "<|vision_start|><|image_pad|><|vision_end|>" per image +
// "\n" follows "STATE:\n", as the release's processor call writes it).
// max_state_tokens (when has_max_state) slices the state like the release: state_ids[:n].
Record encode(const Tokenizer& tok, const json::Value& state, const json::Value& questions,
              const std::vector<int>& image_tokens, int32_t image_token, int max_length, bool has_max_state,
              int max_state_tokens);

// The release's systemone_answer for one question: probabilities keyed by option id.
json::Value answer(const Question& q, const std::vector<double>& probs);

class Head {
 public:
  // joint_head.safetensors + joint_head_config.json of `dir`, and the LM head
  // rows (language_model.lm_head.weight or lm_head.weight, dense or MLX
  // affine 4/8-bit) for the options' lexical vectors.
  static std::unique_ptr<Head> load(const std::string& dir);
  ~Head();
  int hidden() const;
  // One logit per option of every question. hidden: final-normed backbone
  // states of every row of `r.ids` (the LM head's input), float32 [L][hidden].
  std::vector<std::vector<float>> forward(const float* hidden, const Record& r) const;

  struct Impl;

 private:
  explicit Head(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace clef
}  // namespace espetero
