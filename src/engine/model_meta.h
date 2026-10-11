// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// What the engine reads from a model's files before loading its weights:
// whether it is a model of a supported family (MLX config.json model_type,
// GGUF general.architecture), so other models fail with one clear message
// instead of a tokenizer or geometry error, and the sampling defaults the
// model ships (MLX generation_config.json, GGUF general.sampling.*).
#pragma once

#include <optional>
#include <string>

#include "../util/json.h"

namespace espetero {

// Throws std::runtime_error with a message for the user when `path` is not
// a model Espetero can load: missing, neither a directory nor a .gguf file,
// a directory without config.json, or a model of another family.
void check_model_supported(const std::string& path);

// Sampling defaults of a model, for requests that omit them (vLLM's and
// llama.cpp's rule). Unset fields keep the OpenAI defaults.
struct SamplingDefaults {
  std::optional<double> temperature, top_p, min_p;
  std::optional<int> top_k;  // values <= 0 (off) are stored as 0
  std::string source;        // "generation_config.json", "GGUF general.sampling", or "" (none)
  bool empty() const { return !temperature && !top_p && !min_p && !top_k; }
  // The values a request that omits them gets, and where they come from.
  json::Value to_json(bool enabled) const;
};

// generation_config.json of an MLX directory (temperature, top_p, top_k,
// min_p; out-of-range values are skipped, do_sample is ignored as in vLLM),
// or a GGUF file's general.sampling.temp / top_p / top_k / min_p. Never
// throws: unreadable files give no defaults.
SamplingDefaults read_sampling_defaults(const std::string& path);

}  // namespace espetero
