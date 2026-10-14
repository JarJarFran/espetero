// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Hidden-state decision heads: a linear head over the final-normed hidden
// state of a prompt's last token (the LM head's input), next to the model's
// weights as hidden_head.safetensors (fp32 weight [slots, hidden], bias
// [slots], mu and sd [hidden]) and hidden_head.json, as
// tools/gemma4_unified_convert.py --head writes them:
//
//   logits = weight ((h - mu) / sd) + bias
//
// A chat request asks for them with "head" (chat.cpp); the engine runs the
// prompt without generating and reads h in fp32 (TreeSpec::out_last_hidden).
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace espetero {

class HiddenHead {
 public:
  // True when `dir` holds hidden_head.json.
  static bool present(const std::string& dir);
  // Throws std::runtime_error on a malformed head.
  static std::unique_ptr<HiddenHead> load(const std::string& dir);
  int slots() const { return slots_; }
  int hidden() const { return hidden_; }
  // The first `n` slots' logits for hidden state h [hidden] (fp32, accumulated in double).
  std::vector<float> logits(const float* h, int n) const;

 private:
  int slots_ = 0, hidden_ = 0;
  std::vector<float> w_, b_, mu_, sd_;
};

}  // namespace espetero
