// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "requests.h"

#include <algorithm>

#include "engine.h"

namespace espetero {

namespace {
[[noreturn]] void invalid(const std::string& msg) { throw RequestError(422, msg); }
}  // namespace

std::vector<std::string> tool_names(const json::Value* tools) {
  std::vector<std::string> out;
  if (!tools || !tools->is_array()) return out;
  for (const json::Value& t : tools->as_array()) {
    const json::Value* f = t.get("function");
    if (f && f->is_object()) out.push_back(f->get_string("name", ""));
  }
  return out;
}

ToolChoice parse_tool_choice(const json::Value* tc, const json::Value* tools) {
  ToolChoice out;
  if (!tc || tc->is_null()) return out;
  const std::vector<std::string> names = tool_names(tools);
  auto declared = [&](const std::string& n) { return std::find(names.begin(), names.end(), n) != names.end(); };
  auto function_name = [](const json::Value& f) {  // {"type": "function", "function": {"name": N}}
    const json::Value* fn = f.get("function");
    return fn && fn->is_object() ? fn->get_string("name", "") : std::string();
  };
  if (tc->is_string()) {
    const std::string& s = tc->as_string();
    if (s == "none") out.mode = ToolChoice::None;
    else if (s == "required") out.mode = ToolChoice::Required;
    else if (s != "auto") invalid("tool_choice: \"auto\", \"none\", \"required\", a function or allowed_tools");
  } else if (tc->is_object()) {
    const std::string type = tc->get_string("type", "");
    if (type == "function") {
      const std::string n = function_name(*tc);
      if (n.empty()) invalid("tool_choice: {\"type\": \"function\", \"function\": {\"name\": ...}}");
      if (!declared(n)) invalid("tool_choice: no function named '" + n + "' in tools");
      out.mode = ToolChoice::Named;
      out.name = n;
    } else if (type == "allowed_tools") {
      const json::Value* at = tc->get("allowed_tools");
      if (!at || !at->is_object()) invalid("tool_choice: allowed_tools needs {\"mode\": ..., \"tools\": [...]}");
      const std::string mode = at->get_string("mode", "auto");
      if (mode == "required") out.mode = ToolChoice::Required;
      else if (mode != "auto") invalid("tool_choice: allowed_tools mode \"auto\" or \"required\"");
      const json::Value* list = at->get("tools");
      if (!list || !list->is_array() || list->size() == 0) invalid("tool_choice: allowed_tools needs a nonempty tools list");
      for (const json::Value& t : list->as_array()) {
        if (t.get_string("type", "") != "function") invalid("tool_choice: allowed_tools lists functions only");
        const std::string n = function_name(t);
        if (!declared(n)) invalid("tool_choice: allowed_tools names '" + n + "', which is not in tools");
        if (std::find(out.allowed.begin(), out.allowed.end(), n) == out.allowed.end()) out.allowed.push_back(n);
      }
    } else {
      invalid("tool_choice: type \"function\" or \"allowed_tools\"");
    }
  } else {
    invalid("tool_choice: \"auto\", \"none\", \"required\", a function or allowed_tools");
  }
  if (out.mode == ToolChoice::Required) {
    const std::vector<std::string>& offered = out.allowed.empty() ? names : out.allowed;
    if (offered.empty()) invalid("tool_choice: \"required\" needs tools");
    if (offered.size() == 1) {  // a single tool to call: its name is known
      out.mode = ToolChoice::Named;
      out.name = offered[0];
    }
  }
  return out;
}

EmbeddingRequest parse_embedding_request(const json::Value& body, int64_t vocab, int64_t width) {
  if (!body.is_object()) invalid("request body must be a JSON object");
  EmbeddingRequest r;
  if (const json::Value* f = body.get("encoding_format"); f && !f->is_null()) {
    const std::string e = f->is_string() ? f->as_string() : std::string();
    if (e == "base64") r.base64 = true;
    else if (e != "float") invalid("encoding_format: \"float\" or \"base64\"");
  }
  if (const json::Value* d = body.get("dimensions"); d && !d->is_null()) {
    if (!d->is_int() || d->as_int() != width)
      invalid("dimensions: this model's embeddings have " + std::to_string(width) + " dimensions (they are not truncated)");
  }
  const json::Value* in = body.get("input");
  if (!in || in->is_null()) invalid("input: required");
  auto tokens_of = [&](const json::Value& list, const std::string& what) {
    std::vector<int32_t> ids;
    for (const json::Value& x : list.as_array()) {
      if (!x.is_int()) invalid(what + ": token ids are integers");
      if (x.as_int() < 0 || x.as_int() >= vocab) invalid(what + ": token id " + std::to_string(x.as_int()) + " out of range");
      ids.push_back(int32_t(x.as_int()));
    }
    if (ids.empty()) invalid(what + ": an empty token list");
    return ids;
  };
  if (in->is_string()) {
    if (in->as_string().empty()) invalid("input: an empty string");
    r.texts.push_back(in->as_string());
    return r;
  }
  if (!in->is_array() || in->size() == 0)
    invalid("input: a string, a list of strings, a list of token ids or a list of token lists, not empty");
  const json::Value& first = (*in)[0];
  if (first.is_int()) {  // one input as token ids
    r.tokens.push_back(tokens_of(*in, "input"));
    return r;
  }
  if (in->size() > kMaxEmbeddingInputs) invalid("input: at most " + std::to_string(kMaxEmbeddingInputs) + " inputs");
  for (size_t i = 0; i < in->size(); ++i) {
    const json::Value& x = (*in)[i];
    const std::string what = "input[" + std::to_string(i) + "]";
    if (first.is_string()) {
      if (!x.is_string()) invalid("input: strings only, or token lists only");
      if (x.as_string().empty()) invalid(what + ": an empty string");
      r.texts.push_back(x.as_string());
    } else if (first.is_array()) {
      if (!x.is_array()) invalid("input: strings only, or token lists only");
      r.tokens.push_back(tokens_of(x, what));
    } else {
      invalid("input: a string, a list of strings, a list of token ids or a list of token lists");
    }
  }
  return r;
}

}  // namespace espetero
