// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Structured output requests: OpenAI's response_format (chat and
// completions), the Responses API's text.format and Ollama's format, as the
// constraint a generation request carries (grammar.h, json_schema.h).
//
// json_object: the reply is one JSON object. json_schema: the reply is JSON
// valid under the schema (the subset json_schema.h enforces; "strict": true
// refuses schemas with keywords outside it, otherwise the reply follows the
// rest and the response lists what was left out). When the prompt opens a
// reasoning block (thinking on), the model thinks freely and the grammar
// applies after its </think>; the end of the turn is masked until then, and
// until the JSON value is complete. With tools offered (tool_choice auto),
// the reply may instead start with a tool call (<tool_call>), after which it
// is not constrained.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../util/json.h"
#include "grammar.h"

namespace espetero {

class Engine;
struct ChatJob;

struct ResponseFormat {
  enum Type { Text, JsonObject, JsonSchema } type = Text;
  json::Value schema;  // JsonSchema
  bool strict = false;
  std::string name;
};

// OpenAI's response_format: absent or null is text; {"type": "text"},
// {"type": "json_object"} or {"type": "json_schema", "json_schema": {"name",
// "schema", "strict", "description"}}. Throws RequestError(422).
ResponseFormat parse_response_format(const json::Value* response_format);

// The Responses API's text.format ({"type": "text" | "json_object" |
// "json_schema", "name", "schema", "strict"}) as a chat body's
// response_format (null for text). Throws RequestError(422).
json::Value responses_response_format(const json::Value* format);

// Ollama's format ("json", a JSON schema object; absent, null or "": none)
// as a chat body's response_format (null for none): "json" is json_object,
// a schema is a non-strict json_schema. Throws RequestError(422).
json::Value ollama_response_format(const json::Value* format);

// The constraint of a format for the engine's model. Compiled grammars and
// their masks are cached by schema. think_end >= 0: the reply opens in a
// reasoning block this token closes; `tools`: the reply may start with a
// tool call instead. `unenforced` gets a non-strict schema's keywords left
// out. Throws RequestError(422).
std::shared_ptr<const grammar::Constraint> make_constraint(const Engine& engine, const ResponseFormat& f, int32_t think_end,
                                                           bool tools, std::vector<std::string>* unenforced);

// Chat and completions: the body's response_format on a prepared job, before
// its prompt is tokenized (`prompt`: the rendered text, which gets "</think>"
// when it opens a reasoning block the model has no single token to close).
// `chat`: the prompt comes from the chat template (a reasoning block it
// opens is detected). No-op for text. Throws RequestError(422).
void apply_response_format(const Engine& engine, const json::Value& body, ChatJob& job, std::string& prompt, bool chat);

// The response's report of a non-strict schema's keywords left out
// ({"unenforced": [...]}), null when there is none.
json::Value structured_report(const ChatJob& job);

}  // namespace espetero
