// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// JEVfire decisions: the restricted classification contract (flat boolean /
// enum fields over one shared context), prompts token-identical to the
// reference stack's `batch` strategy, and the response format.
//
// The chat template is rendered once with markers, so a request tokenizes its
// shared prefix once and each field suffix separately; the split is at a
// pre-token boundary ("}\n" | "Selected") and verified against a full render.
#pragma once

#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../util/json.h"

namespace espetero {

class Tokenizer;
namespace tmpl {
class Template;
}

extern const char* const kDecisionSystemPrompt;

struct DecisionField {
  std::string name;
  bool boolean = false;
  std::string description;
  std::vector<std::string> choices;  // enum
  size_t n_values() const { return boolean ? 2 : choices.size(); }
  json::Value value(size_t i) const { return boolean ? json::Value(i == 0) : json::Value(choices[i]); }
};

struct DecisionRequest {
  std::string context;
  std::vector<DecisionField> fields;
  std::string strategy = "auto";
  double score_temperature = 1.0;
  bool has_min_probability = false;
  double min_probability = 0;
  std::string cache_salt;
  bool has_salt = false;
  std::string image_detail = "auto";  // low | auto | high | max
  std::vector<std::string> images;  // data URLs
};

// Validates like the reference pydantic models (extra keys forbidden, strict
// types, bounds); throws RequestError(422) with a message.
DecisionRequest parse_decision_request(const json::Value& body);

struct Label {
  std::string text;
  int32_t token = -1;
};
// Distinct, round-tripping single-token answer labels: A..Z, AA..ZZ, a..z.
std::vector<Label> make_labels(const Tokenizer& tok, size_t count = 255);

struct DecisionPlan {
  std::vector<int32_t> prefix;                 // with image placeholders expanded by the caller
  std::vector<std::vector<int32_t>> suffixes;  // per field
  std::vector<int32_t> stem;                   // common start of every suffix
  std::vector<std::vector<int32_t>> leaves;    // suffixes after the stem
  double tokenize_ms = 0;
  int images = 0, image_tokens = 0;
  size_t max_prompt_tokens() const {
    size_t m = 0;
    for (const auto& s : suffixes) m = std::max(m, prefix.size() + s.size());
    return m;
  }
  size_t prompt_tokens() const {
    size_t t = 0;
    for (const auto& s : suffixes) t += prefix.size() + s.size();
    return t;
  }
};

class DecisionPromptBuilder {
 public:
  DecisionPromptBuilder(const Tokenizer& tok, const tmpl::Template& tpl, const std::vector<Label>& labels,
                        const std::string& image_block);
  DecisionPlan plan(const DecisionRequest& r, int n_images);
  // Tokens every request without a salt and without images shares (the
  // instructions up to the opening of the context), checkpointed at startup.
  const std::vector<int32_t>& header() const { return header_; }

 private:
  std::string prefix_text(const std::string& context, int n_images) const;
  std::string suffix_text(const DecisionField& f) const;

  const Tokenizer& tok_;
  const std::vector<Label>& labels_;
  std::string head_, field_intro_, tail_, image_block_;
  std::vector<int32_t> header_;
  bool verified_ = false;
  std::mutex mu_;
  // LRU cache of suffix tokens (field suffixes do not depend on the context).
  std::list<std::string> lru_;
  std::unordered_map<std::string, std::pair<std::vector<int32_t>, std::list<std::string>::iterator>> suffix_cache_;
};

// Scores of one field -> the reference response entry (value, probability,
// candidates, ...). logits: candidate logits; lse: vocabulary log-sum-exp.
json::Value decode_field(const DecisionField& f, const std::vector<Label>& labels, const float* logits, float lse,
                         const DecisionRequest& r, bool* abstained);

}  // namespace espetero
