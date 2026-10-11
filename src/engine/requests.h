// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Request fields parsed the same way by every API that takes them, kept apart
// from the engine so that tests can check them without a model.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../util/json.h"

namespace espetero {

// A chat request's tool_choice.
struct ToolChoice {
  enum Mode { Auto, None, Required, Named };
  Mode mode = Auto;
  std::string name;                  // Named: the function the reply calls
  std::vector<std::string> allowed;  // allowed_tools: the functions the prompt offers (empty: every one)
};

// OpenAI chat's tool_choice over `tools` (the request's list, OpenAI chat
// shape): "auto" (or absent), "none", "required", {"type": "function",
// "function": {"name": N}} or {"type": "allowed_tools", "allowed_tools":
// {"mode": "auto" | "required", "tools": [{"type": "function", "function":
// {"name": N}}, ...]}}. Forcing needs a declared tool; "required" among a
// single tool is that tool (Named). Throws RequestError(422).
ToolChoice parse_tool_choice(const json::Value* tool_choice, const json::Value* tools);

// The names of the functions in an OpenAI chat tools list.
std::vector<std::string> tool_names(const json::Value* tools);

// An /v1/embeddings request: its inputs, as text or as token lists,
// and the reply's encoding.
struct EmbeddingRequest {
  std::vector<std::string> texts;            // text inputs (when `tokens` is empty)
  std::vector<std::vector<int32_t>> tokens;  // token inputs
  bool base64 = false;                       // encoding_format "base64": little-endian float32, base64
  size_t size() const { return tokens.empty() ? texts.size() : tokens.size(); }
};
constexpr size_t kMaxEmbeddingInputs = 2048;  // OpenAI's limit
// `input`: a string, a list of strings, a list of token ids or a list of such
// lists; encoding_format "float" (default) or "base64"; dimensions, when
// given, the model's width (`width`: the vectors are not truncated). Throws
// RequestError(422) for anything else, an empty input or token list, more
// than kMaxEmbeddingInputs inputs or a token id outside [0, vocab).
EmbeddingRequest parse_embedding_request(const json::Value& body, int64_t vocab, int64_t width);

}  // namespace espetero
