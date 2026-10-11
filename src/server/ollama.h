// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Ollama's API on the server's own port: /api/version, /api/tags,
// /api/ps, /api/show, /api/chat, /api/generate, /api/embed and the legacy
// /api/embeddings, so Ollama clients (the ollama Python and JavaScript
// libraries, Open WebUI, editors) work against Espetero unchanged.
//
// Each request becomes the body the OpenAI route would take and runs through
// the same chat job (prepare_chat / run_chat), so the prompt and the tokens
// are those of /v1/chat/completions; only the shapes differ: NDJSON lines
// when streamed (Ollama's default), durations in nanoseconds, tool call
// arguments as objects, images as bare base64 strings.
//
// The translations below need no model (tests/ollama_test.cpp); the routes
// (ollama_api.cpp) add the engine.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>

#include "../http/http.h"
#include "../util/json.h"

namespace espetero {

class Engine;

namespace ollama {

// What a translation needs to know about the loaded model.
struct ModelTraits {
  int max_ctx = 0;         // tokens per sequence: options.num_predict -1 asks for all the room left
  bool thinking = false;   // its chat template has a thinking mode (think: true / a level)
};

// Per-request settings the chat body cannot carry, applied once the prompt
// is known (num_ctx: prompt + reply within that many tokens).
struct Extras {
  int64_t num_ctx = 0;     // options.num_ctx, 0 when absent
  bool stream = true;      // Ollama streams unless "stream": false
  bool load_only = false;  // nothing to generate: an empty /api/chat or /api/generate loads the model
};

// An /api/chat body as the chat route's body. Throws RequestError (400 for
// malformed fields, 422 for what is not supported: never ignored).
json::Value chat_body(const json::Value& body, const ModelTraits& model, Extras& extras);
// An /api/generate body: the chat body of one user turn (`raw` false), or
// with "raw": true the /v1/completions body (the prompt as is).
json::Value generate_body(const json::Value& body, const ModelTraits& model, Extras& extras, bool& raw);
// /api/embed (`legacy` false) or /api/embeddings bodies as the /v1/embeddings
// body; `empty` when there is nothing to embed (Ollama answers an empty list).
json::Value embed_body(const json::Value& body, bool legacy, bool& empty);

// options.* onto a chat or completions body (exposed for the tests).
void apply_options(const json::Value* options, const ModelTraits& model, json::Value& out, Extras& extras);
// A base64 image (or a data URL) as the data URL the chat route takes.
std::string image_data_url(const std::string& image);
// An OpenAI tool call ({"id", "type", "function": {"name", "arguments": "<json>"}})
// as Ollama's ({"function": {"index", "name", "arguments": {...}}}).
json::Value tool_call(const json::Value& openai_call, int index);
// Ollama's finish reason for the chat route's: stop, length (tool calls end with stop).
std::string done_reason(const std::string& finish_reason);
// Ollama's created_at: RFC 3339 in UTC with nanoseconds.
std::string timestamp(std::chrono::system_clock::time_point t);
// "name" -> "name:latest": Ollama's names carry a tag.
std::string tagged(const std::string& name);

}  // namespace ollama

// The routes, on `server` (install_routes calls this).
void install_ollama_routes(http::Server& server, Engine& engine);

}  // namespace espetero
