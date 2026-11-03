// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// GLM-4.5 / GLM-5 tool calls, apart from chat.cpp so the parser is tested
// without the engine (tests/glm_test.cpp). The template (docs/glm5.md 7)
// writes a call as
//   <tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>
// right after the reasoning block or the content, several calls back to back,
// no separators. A string argument is written as is, any other type as JSON
// (tojson): the declared parameter types decide which is which, as the Qwen
// parser does; without a declaration the value is JSON when it parses.
#include <cctype>
#include <random>
#include <string>

#include "../util/json.h"
#include "chat.h"

namespace espetero {

namespace {
std::string call_id() {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  static const char* h = "0123456789abcdef";
  std::string s = "call_";
  for (int i = 0; i < 24; ++i) s.push_back(h[rng() & 15]);
  return s;
}

// The declared type of parameter `p` of function `fn` ("" when undeclared).
std::string param_type(const json::Value& tools, const std::string& fn, const std::string& p) {
  if (!tools.is_array()) return "";
  for (const auto& t : tools.as_array()) {
    const json::Value* f = t.get("function");
    if (!f || f->get_string("name", "") != fn) continue;
    const json::Value* params = f->get("parameters");
    const json::Value* props = params ? params->get("properties") : nullptr;
    const json::Value* pp = props ? props->get(p) : nullptr;
    if (pp) return pp->get_string("type", "");
  }
  return "";
}

bool name_char(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.' || c == ':';
}
}  // namespace

bool glm_tool_call_body(const std::string& body) {
  // NAME, then <arg_key> or the end: not Qwen's "\n<function=".
  size_t n = 0;
  while (n < body.size() && name_char(body[n])) ++n;
  if (n == 0) return false;
  return n == body.size() || body.compare(n, 9, "<arg_key>") == 0 || body[n] == '\n';
}

std::string parse_glm_tool_calls(const std::string& content, const json::Value& tools, json::Value& calls) {
  calls = json::Value();
  static const std::string open = "<tool_call>", close = "</tool_call>";
  static const std::string kKey = "<arg_key>", kKeyEnd = "</arg_key>", kVal = "<arg_value>", kValEnd = "</arg_value>";
  size_t at = content.find(open);
  if (at == std::string::npos) return content;
  std::string before = content.substr(0, at);
  while (!before.empty() && (before.back() == '\n' || before.back() == ' ')) before.pop_back();
  json::Value out = json::Value::array();
  while (at != std::string::npos) {
    const size_t body0 = at + open.size();
    const size_t end = content.find(close, body0);
    const std::string body = content.substr(body0, end == std::string::npos ? std::string::npos : end - body0);
    size_t n = 0;
    while (n < body.size() && name_char(body[n])) ++n;
    const std::string name = body.substr(0, n);
    if (!name.empty() && glm_tool_call_body(body)) {
      json::Value args = json::Value::object();
      size_t p = n;
      while (true) {
        const size_t k0 = body.find(kKey, p);
        if (k0 == std::string::npos) break;
        const size_t k1 = body.find(kKeyEnd, k0 + kKey.size());
        if (k1 == std::string::npos) break;
        const std::string key = body.substr(k0 + kKey.size(), k1 - k0 - kKey.size());
        size_t v0 = body.find(kVal, k1 + kKeyEnd.size());
        if (v0 == std::string::npos) break;
        v0 += kVal.size();
        size_t v1 = body.find(kValEnd, v0);
        if (v1 == std::string::npos) v1 = body.find(kKey, v0);  // an unclosed value (the reply was cut): up to the next key
        if (v1 == std::string::npos) v1 = body.size();
        const std::string value = body.substr(v0, v1 - v0);
        json::Value v(value);
        if (param_type(tools, name, key) != "string") {
          try {
            v = json::parse(value);
          } catch (...) {
            v = json::Value(value);
          }
        }
        args.set(key, std::move(v));
        p = v1;
      }
      json::Value fn = json::Value::object();
      fn.set("name", name);
      fn.set("arguments", json::dump(args));
      json::Value call = json::Value::object();
      call.set("id", call_id());
      call.set("type", "function");
      call.set("function", std::move(fn));
      out.push_back(std::move(call));
    }
    if (end == std::string::npos) break;
    at = content.find(open, end + close.size());
  }
  if (out.size()) calls = std::move(out);
  return calls.is_null() ? content : before;
}

}  // namespace espetero
