// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Kev decision models (jaredpalmer/kev-27b): a causal Qwen3.5 backbone run
// prefill-only and a pointer head over each question's options, as the
// release's code builds them (kev/api.py, kev/model.py of
// https://github.com/jaredpalmer/kev). A /v1/systemone request (the TypeSafe
// System One body: state, questions) is encoded token for token like
// api.to_record and model.encode:
//
//   <|fim_prefix|> state
//   <|fim_middle|> instructions <|box_start|> option <|box_end|> ... <|fim_suffix|>
//
// the state once, then one row per question that continues it (its
// positions too), so questions never see each other and the state is
// computed once and cached. Caller text is tokenized so that it never
// yields a delimiter (`<|name|>` is read as `<¦name¦>`). The head scores
// each option's <|box_end|> state against the question's <|fim_suffix|>
// state (the final-normed hidden states, TreeSpec::out_rows):
//
//   z_k = (W_k h_k + b_k) . (W_q h_decide + b_q) / sqrt(head_dim),  p = softmax(z / T)
//
// Images (an Espetero extension: the release reads text only, and the
// vision tower is the base model's, grafted by tools/kev_convert.py) open
// the state, each as the base model's chat template writes an image part
// before the text: <|vision_start|><|image_pad|>...<|vision_end|>.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "../util/json.h"

namespace espetero {

class Tokenizer;

namespace kev {

// True when `dir` holds a Kev pointer head (pointer_head.json next to the weights).
bool is_release(const std::string& dir);

// The release's serving limits (kev/model.py SERVE_MAX_STATE, SERVE_MAX_BRANCH):
// a state of at most 65,536 tokens (the <|fim_prefix|> token included) and
// a question row (state + its branch) of at most 8,192 tokens more.
constexpr int kMaxState = 65536;
constexpr int kMaxRow = kMaxState + 8192;

// The five delimiter ids (kev/model.py SPECIAL).
struct Delimiters {
  int32_t state = -1, question = -1, option = -1, option_end = -1, decide = -1;
};

struct Question {
  std::string id;
  std::string type;                  // noul | choice | score
  std::vector<std::string> keys;     // probability keys in option order (api.question_keys)
  std::vector<std::string> legend;   // score: each level's rendered text
  std::vector<int32_t> ids;          // the branch: <q> instructions (<opt> option </opt>)... <decide>
  std::vector<int32_t> rows;         // read-out rows of the branch: <decide>, then each option's </opt>
};

struct Record {
  std::vector<int32_t> state;        // <state>, the image blocks, the state text
  std::vector<Question> questions;
  int input_tokens = 0;              // state + every branch (the release's usage.input_tokens)
};

// api.render: the text the model reads for a JSON value (field names kept as labels).
std::string render(const json::Value& v, int indent = 0);
// model.user_tokens: caller text tokenized so that it never yields a delimiter or control token.
std::vector<int32_t> user_tokens(const Tokenizer& tok, std::string_view text);

// Validates a SystemOne request's questions as the release's request models
// do (422 otherwise) and encodes the record. image_tokens: each image's
// soft-token count, in order (vision_start, image_pad and vision_end are the
// image block's ids). 422 for a state over kMaxState or a row over kMaxRow.
Record encode(const Tokenizer& tok, const Delimiters& d, const json::Value& state, const json::Value& questions,
              const std::vector<int>& image_tokens = {}, int32_t vision_start = -1, int32_t image_pad = -1,
              int32_t vision_end = -1);

// api.to_answers for one question: p are its option probabilities.
json::Value answer(const Question& q, const std::vector<double>& p);

// api.round_prob: Python's round(x, 4).
double round_prob(double x);

class PointerHead {
 public:
  // pointer_head.safetensors + pointer_head.json of `dir` (tools/kev_convert.py),
  // or a VideoJev checkpoint (`model_type == "videojev"`).
  // Throws std::runtime_error on a malformed head.
  static std::unique_ptr<PointerHead> load(const std::string& dir);
  int hidden() const { return hidden_; }
  int dim() const { return dim_; }
  double temperature() const { return temperature_; }
  bool grafted_vision() const { return grafted_; }
  bool is_videojev() const { return videojev_; }
  // The delimiters' ids in `tok` (pointer_head.json names them); throws when one is not a token.
  void bind(const Tokenizer& tok);
  const Delimiters& delimiters() const { return ids_; }
  // Raw logits (before the temperature) of k options; h: float32 [k + 1][hidden],
  // the <decide> state first, then the options' </opt> states. Accumulated in double.
  std::vector<float> logits(const float* h, int k) const;

 private:
  int hidden_ = 0, dim_ = 0;
  double temperature_ = 1.0;
  double logit_scale_ = 1.0, logit_bias_ = 0.0;
  bool grafted_ = false, videojev_ = false;
  std::vector<std::string> names_;  // state, question, option, option_end, decide
  Delimiters ids_;
  std::vector<double> wq_, bq_, wk_, bk_;  // q / k weight [head_dim][hidden] and bias [head_dim]
};

}  // namespace kev
}  // namespace espetero
