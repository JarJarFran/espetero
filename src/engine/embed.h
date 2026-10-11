// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// OpenAI /v1/embeddings over the loaded chat model: each input's
// final-normed hidden states (the LM head's input) pooled and L2-normalized
// (Engine::embed). These are the chat model's own representations, not those
// of a dedicated embedding model (Qwen3-Embedding is another architecture,
// not supported yet).
#pragma once

#include "engine.h"

namespace espetero {

// The response to an /v1/embeddings body (input: a string, a list of strings,
// a token list or a list of token lists; encoding_format float or base64;
// dimensions only equal to the model's width). Throws RequestError.
json::Value embeddings_response(Engine& engine, const json::Value& body);

}  // namespace espetero
