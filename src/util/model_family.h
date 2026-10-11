// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The model families Espetero runs, and the message for any other: checked
// before loading (engine, Hugging Face downloads) and again by the loaders.
//
//   MLX config.json model_type (or text_config.model_type): the Qwen3.5
//   family (qwen3_5, qwen3_5_text, qwen3_5_moe, qwen3_5_moe_text, qwen3_5_mtp:
//   Qwen3.5, Qwen3.6, Qwen3.8, their finetunes and Clef backbones),
//   Qwen3.8-Flash-Next (qwen4_exp, qwen4_exp_text; qwen4_exp_mtp drafters) and
//   Gemma 4 (gemma4, gemma4_text: 31B and 26B-A4B).
//   GGUF general.architecture: qwen35, qwen35moe, gemma4.
#pragma once

#include <string>
#include <string_view>

#include "../server/version.h"
#include "json.h"

namespace espetero::family {

inline bool mlx_type_supported(std::string_view t) {
  return t.substr(0, 7) == "qwen3_5" || t.substr(0, 9) == "qwen4_exp" || t == "gemma4" || t == "gemma4_text";
}

inline bool gguf_arch_supported(std::string_view a) { return a == "qwen35" || a == "qwen35moe" || a == "gemma4"; }

inline std::string unsupported_mlx(const std::string& path, const std::string& model_type) {
  return path + ": model_type '" + model_type + "' is not supported by Espetero " + kVersion +
         "; it runs the Qwen3.5 family (qwen3_5, qwen3_5_moe: Qwen3.5, Qwen3.6, Qwen3.8, and finetunes such as "
         "Ornith 1.5), Qwen3.8-Flash-Next (qwen4_exp), Gemma 4 (gemma4: 31B, 26B-A4B; text only) and Cloudflare "
         "Clef classifiers";
}

inline std::string unsupported_gguf(const std::string& path, const std::string& arch) {
  return path + ": GGUF architecture '" + arch + "' is not supported by Espetero " + kVersion +
         "; it runs the Qwen3.5 family (qwen35, qwen35moe: Qwen3.5, Qwen3.6, Qwen3.8, and finetunes such as "
         "Ornith 1.5) and Gemma 4 (gemma4: 31B, 26B-A4B; text only); Qwen3.8-Flash-Next (qwen4_exp) and Cloudflare "
         "Clef classifiers run from MLX directories";
}

// The message for an MLX config.json of another family, "" when supported.
// A config with no model_type at all is left to the loader; one whose
// top-level or text_config model_type is of a supported family passes (the
// loader reads text_config's first).
inline std::string mlx_check(const std::string& path, const json::Value& config) {
  const json::Value* tc = config.get("text_config");
  const std::string top = config.get_string("model_type", "");
  const std::string text = tc && tc->is_object() ? tc->get_string("model_type", "") : std::string();
  if (top.empty() && text.empty()) return "";
  if (mlx_type_supported(top) || mlx_type_supported(text)) return "";
  return unsupported_mlx(path, top.empty() ? text : top);
}

}  // namespace espetero::family
