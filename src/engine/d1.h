// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// System One decision models built on LFM2-VL (LiquidAI's d1 releases):
// a state (text, any JSON value, or nothing beside images) and named typed
// questions (noul, choice, score) in, a distribution over each question's
// options out, read from the LM head at the question's answer slot. The
// prompt and the readout follow the release's own builder (prompt.py: the
// user turn holds the images' placeholders, the state as written or as
// json.dumps(indent=2), then per question its instructions, options with
// single-token codes and the reply instruction, then the assistant header);
// each option scores the best of its tokens' logits, and the scores are
// soft-maxed. Nothing is generated. The rules are those of the release's
// prompt.py and api.py (LFM Open License v1.0, see THIRD_PARTY_NOTICES.md).
//
// Token ids are those of the release's SystemOne.run: a request with one
// question is tokenized whole; with several, its prefix (system turn, images,
// state) is one sequence, the trunk of a tree, and each question's suffix its
// own, a leaf. Pictures are first bounded at 1024 x 1024 pixels (Pillow
// bicubic), then go through the LFM2-VL processor.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../util/json.h"

namespace espetero {

class Tokenizer;

namespace d1 {

// True when `dir` holds a d1 release: config.json's auto_map names
// modeling_d1.D1Model (an lfm2_vl or lfm2 checkpoint).
bool is_release(const std::string& dir);

struct Question {
  std::string name;
  std::string type;                          // noul | choice | score
  json::Value criteria;                      // as requested
  std::string suffix;                        // its text after the prefix, up to the answer slot
  std::vector<std::vector<int32_t>> groups;  // option -> token ids (max-pooled)
  std::vector<std::string> options;          // choice: the labels; score: "0".."n-1"; noul: yes, no
};

// The prompt and readout rules of a release, bound to its tokenizer.
class Release {
 public:
  Release(const Tokenizer& tok, const std::string& bos);
  // A questions object ({name: {type, instructions, criteria}}); 422 on errors.
  std::vector<Question> questions(const json::Value& questions) const;
  // Everything before the questions: `state` null means none (the question
  // follows the images directly); `images` placeholders at the user turn's head.
  std::string prefix(const json::Value& state, int images, const std::string& placeholder) const;
  // Probabilities of each option from the logits of its question's tokens
  // (logits[k] belongs to cand_ids(q)[k]).
  static std::vector<double> probabilities(const Question& q, const std::vector<int32_t>& ids, const float* logits);
  // The tokens a question's readout needs (its groups' ids, each once).
  static std::vector<int32_t> cand_ids(const Question& q);
  // The release's answer object of a question.
  static json::Value answer(const Question& q, const std::vector<double>& probs);

 private:
  int32_t single(const std::string& text) const;  // the token of a one-token text, -1 otherwise
  std::vector<int32_t> ids_of(const std::vector<std::string>& texts) const;
  std::vector<std::pair<std::string, int32_t>> aliases(const std::vector<std::string>& labels) const;
  const Tokenizer& tok_;
  std::string bos_;
};

}  // namespace d1
}  // namespace espetero
