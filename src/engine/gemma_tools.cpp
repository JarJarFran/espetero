// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Gemma 4 tool calls, apart from chat.cpp so the parser is tested
// without the engine (tests/gemma_test.cpp).
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
}  // namespace

// Gemma 4 calls: the template writes arguments with format_argument
// (escape_keys = false): strings as <|"|>text<|"|>, booleans true/false, other
// scalars as they print, mappings {key:value,...} with bare keys, sequences
// [a,b]. Parsed back into JSON; a value that is none of these stays a string.
namespace {
struct GemmaArgs {
  const std::string& s;
  size_t i = 0;
  bool ok = true;
  static constexpr const char* kQ = "<|\"|>";
  bool at_quote() const { return s.compare(i, 5, kQ) == 0; }
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
  }
  std::string quoted() {
    i += 5;
    const size_t e = s.find(kQ, i);
    std::string v = s.substr(i, e == std::string::npos ? std::string::npos : e - i);
    i = e == std::string::npos ? s.size() : e + 5;
    if (e == std::string::npos) ok = false;
    return v;
  }
  json::Value value() {
    ws();
    if (i >= s.size()) {
      ok = false;
      return json::Value();
    }
    if (at_quote()) return json::Value(quoted());
    if (s[i] == '{') return object();
    if (s[i] == '[') {
      ++i;
      json::Value a = json::Value::array();
      ws();
      if (i < s.size() && s[i] == ']') {
        ++i;
        return a;
      }
      while (ok && i < s.size()) {
        a.push_back(value());
        ws();
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        if (i < s.size() && s[i] == ']') ++i;
        else ok = false;
        break;
      }
      return a;
    }
    const size_t b = i;
    while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;
    std::string tok = s.substr(b, i - b);
    while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\n')) tok.pop_back();
    if (tok == "true") return json::Value(true);
    if (tok == "false") return json::Value(false);
    if (tok == "null" || tok == "None") return json::Value();
    try {
      json::Value n = json::parse(tok);
      if (n.is_number()) return n;
    } catch (...) {
    }
    return json::Value(tok);
  }
  json::Value object() {
    json::Value o = json::Value::object();
    ++i;  // {
    ws();
    if (i < s.size() && s[i] == '}') {
      ++i;
      return o;
    }
    while (ok && i < s.size()) {
      ws();
      std::string key;
      if (at_quote()) {
        key = quoted();
      } else {
        const size_t b = i;
        while (i < s.size() && s[i] != ':' && s[i] != '}' && s[i] != ',') ++i;
        key = s.substr(b, i - b);
        while (!key.empty() && key.back() == ' ') key.pop_back();
      }
      ws();
      if (i >= s.size() || s[i] != ':') {
        ok = false;
        break;
      }
      ++i;
      o.set(key, value());
      ws();
      if (i < s.size() && s[i] == ',') {
        ++i;
        continue;
      }
      if (i < s.size() && s[i] == '}') ++i;
      else ok = false;
      break;
    }
    return o;
  }
};
}  // namespace

std::string parse_gemma_tool_calls(const std::string& content, json::Value& calls) {
  calls = json::Value();
  const std::string open = "<|tool_call>", close = "<tool_call|>";
  size_t at = content.find(open);
  if (at == std::string::npos) return content;
  std::string before = content.substr(0, at);
  while (!before.empty() && (before.back() == '\n' || before.back() == ' ')) before.pop_back();
  json::Value out = json::Value::array();
  while (at != std::string::npos) {
    const size_t body0 = at + open.size();
    const size_t end = content.find(close, body0);
    const std::string body = content.substr(body0, end == std::string::npos ? std::string::npos : end - body0);
    // call:NAME{...}: the name is a run of word characters (the tokenizer config's x-regex)
    if (body.compare(0, 5, "call:") == 0) {
      size_t n = 5;
      while (n < body.size() && (std::isalnum(static_cast<unsigned char>(body[n])) || body[n] == '_' || body[n] == '-' ||
                                 body[n] == '.'))
        ++n;
      const std::string name = body.substr(5, n - 5);
      if (!name.empty() && n < body.size() && body[n] == '{') {
        GemmaArgs p{body, n};
        json::Value args = p.object();
        if (args.is_object()) {
          json::Value fn = json::Value::object();
          fn.set("name", name);
          fn.set("arguments", json::dump(args));
          json::Value call = json::Value::object();
          call.set("id", call_id());
          call.set("type", "function");
          call.set("function", std::move(fn));
          out.push_back(std::move(call));
        }
      }
    }
    if (end == std::string::npos) break;
    at = content.find(open, end + close.size());
  }
  if (out.size()) calls = std::move(out);
  return calls.is_null() ? content : before;
}

}  // namespace espetero
