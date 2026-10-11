// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "pretok.h"

namespace espetero::tok {

namespace {

// The Split regexes exactly as tokenizer.json stores them (after JSON unescaping).
constexpr std::string_view kQwen35Regex =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";
constexpr std::string_view kQwen2Regex =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";
constexpr std::string_view kLlama3Regex =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";

}  // namespace

std::optional<SplitPattern> split_pattern_from_regex(std::string_view regex) {
  if (regex == kQwen35Regex) return SplitPattern::kQwen35;
  if (regex == kQwen2Regex) return SplitPattern::kQwen2;
  if (regex == kLlama3Regex) return SplitPattern::kLlama3;
  return std::nullopt;
}

std::optional<SplitPattern> split_pattern_from_gguf_pre(std::string_view pre) {
  // Names from llama.cpp's llama-vocab.cpp that map to these exact regexes.
  if (pre == "qwen35") return SplitPattern::kQwen35;
  if (pre == "qwen2" || pre == "deepseek-r1-qwen") return SplitPattern::kQwen2;
  if (pre == "llama3" || pre == "llama-v3" || pre == "llama-bpe") return SplitPattern::kLlama3;
  return std::nullopt;
}

std::string_view split_pattern_name(SplitPattern p) {
  switch (p) {
    case SplitPattern::kQwen2: return "qwen2";
    case SplitPattern::kQwen35: return "qwen35";
    case SplitPattern::kLlama3: return "llama3";
  }
  return "?";
}

}  // namespace espetero::tok
