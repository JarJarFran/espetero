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
// LFM2 (LiquidAI): Llama-3's pieces written differently. '(?i:[sdmt]|ll|ve|re)
// is the same contraction set; \s*[\r\n] and \s*[\r\n]+ both end at the run's
// last CR/LF; the final \s is reached only by a single whitespace before a
// non-space, where \s+ takes one character too.
constexpr std::string_view kLfm2Regex =
    R"re('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s)re";

}  // namespace

// DeepSeek-V3's third Split regex (the first two are fixed; see deepseek_split).
constexpr std::string_view kDeepSeekV3Regex =
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| "
    "?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

std::optional<SplitPattern> split_pattern_from_regex(std::string_view regex) {
  if (regex == kQwen35Regex) return SplitPattern::kQwen35;
  if (regex == kQwen2Regex) return SplitPattern::kQwen2;
  if (regex == kLlama3Regex) return SplitPattern::kLlama3;
  if (regex == kDeepSeekV3Regex) return SplitPattern::kDeepSeekV3;
  if (regex == kLfm2Regex) return SplitPattern::kLfm2;
  return std::nullopt;
}

std::optional<SplitPattern> split_pattern_from_gguf_pre(std::string_view pre) {
  // Names from llama.cpp's llama-vocab.cpp that map to these exact regexes.
  if (pre == "qwen35") return SplitPattern::kQwen35;
  if (pre == "qwen2" || pre == "deepseek-r1-qwen") return SplitPattern::kQwen2;
  if (pre == "llama3" || pre == "llama-v3" || pre == "llama-bpe") return SplitPattern::kLlama3;
  // GLM-4.5 / GLM-5 (llama.cpp's CHATGLM4 type): Llama-3's pieces, contractions spelled case by case (the same
  // pieces but for U+017F, which (?i) folds to 's'); llama.cpp also sets ignore_merges for these, as the
  // Llama-3 pattern does here.
  if (pre == "glm4" || pre == "glm5") return SplitPattern::kLlama3;
  if (pre == "deepseek-v3") return SplitPattern::kDeepSeekV3;
  if (pre == "lfm2") return SplitPattern::kLfm2;
  return std::nullopt;
}

std::string_view split_pattern_name(SplitPattern p) {
  switch (p) {
    case SplitPattern::kQwen2: return "qwen2";
    case SplitPattern::kQwen35: return "qwen35";
    case SplitPattern::kLlama3: return "llama3";
    case SplitPattern::kDeepSeekV3: return "deepseek-v3";
    case SplitPattern::kLfm2: return "lfm2";
    case SplitPattern::kSpark: return "spark";
  }
  return "?";
}

}  // namespace espetero::tok
