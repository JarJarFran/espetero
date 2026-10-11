// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#pragma once

#include "../engine/batch.h"
#include "../engine/engine.h"
#include "../http/http.h"

namespace espetero {

// What the server answers while the model loads (its port opens first):
// the web chat page (unless `webui` is false), which waits for the
// model, and for every other request a 503 {"status": "loading"}
// (llama-server's /health answer, with an OpenAI-style error beside it;
// Ollama's routes get Ollama's {"error": "..."}). Call before run();
// install_routes then adds the model's routes and Server::open() serves them.
void install_loading_routes(http::Server& server, bool webui);
// Registers every route on `server`. `batches` may be null (no Batch API).
void install_routes(http::Server& server, Engine& engine, BatchService* batches, const json::Value* config = nullptr);
// Anthropic Messages, OpenAI Responses and Prometheus /metrics over the chat job (compat.cpp).
void install_compat_routes(http::Server& server, Engine& engine);

}  // namespace espetero
