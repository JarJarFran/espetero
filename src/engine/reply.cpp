// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The text of a reply: invalid UTF-8, the reasoning block, the chat markers
// the model generated as tokens (MarkerToken, ReplyText) and Qwen / DeepSeek
// tool calls; apart from chat.cpp so they are tested without the engine
// (tests/reply_test.cpp).
#include <random>
#include <string>
#include <vector>

#include "../tok/tokenizer.h"
#include "../util/json.h"
#include "chat.h"

namespace espetero {

std::string sanitize_utf8(std::string_view s) {
  if (json::utf8_valid(s)) return std::string(s);
  std::string out;
  size_t i = 0;
  while (i < s.size()) json::utf8_append(out, json::utf8_next(s, i));
  return out;
}

namespace {

std::string call_id() {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  static const char* h = "0123456789abcdef";
  std::string s = "call_";
  for (int i = 0; i < 24; ++i) s.push_back(h[rng() & 15]);
  return s;
}

// (reasoning, content, closed) of generated text; `opened`: the prompt
// already opened the reasoning block. `close_at` (sorted): the offsets in
// `raw` where the model generated the closing marker's token (MarkerTokens);
// null: anywhere the marker's text appears.
struct Split {
  std::string reasoning, content;
  bool closed = true;
  size_t held = 0;        // trailing bytes held back as a partial tag (their tokens' logprobs wait too)
  size_t content_at = 0;  // where `content` starts in `raw`
};
Split split_reasoning(const std::string& raw, bool opened, const ChatMarkers& mk,
                      const std::vector<size_t>* close_at = nullptr) {
  const std::string& open = mk.think_open;
  const std::string& close = mk.think_close;
  Split out;
  size_t shift = 0;
  if (!opened) {
    if (raw.compare(0, open.size(), open) != 0) {
      // A partial "<think>" at the very start is held back.
      if (!raw.empty() && open.compare(0, raw.size(), raw) == 0) {
        out.closed = false;
        out.held = raw.size();
        return out;
      }
      out.content = raw;
      return out;
    }
    shift = open.size();
  }
  std::string text = raw.substr(shift);
  size_t end = std::string::npos;
  if (!close_at) {
    end = text.find(close);
  } else {
    for (size_t p : *close_at)
      if (p >= shift && raw.compare(p, close.size(), close) == 0) {
        end = p - shift;
        break;
      }
  }
  auto lead_nl = [](const std::string& s, size_t from) {
    size_t n = 0;
    while (from + n < s.size() && s[from + n] == '\n') ++n;
    return n;
  };
  auto strip_nl = [](std::string s, bool left, bool right) {
    if (left)
      while (!s.empty() && s.front() == '\n') s.erase(s.begin());
    if (right)
      while (!s.empty() && s.back() == '\n') s.pop_back();
    return s;
  };
  if (end == std::string::npos) {
    // Hold back a partial closing tag.
    for (size_t k = close.size() - 1; k > 0; --k)
      if (text.size() >= k && text.compare(text.size() - k, k, close, 0, k) == 0) {
        text.resize(text.size() - k);
        out.held = k;
        break;
      }
    out.reasoning = strip_nl(text, true, false);
    out.closed = false;
    out.content_at = raw.size();
    return out;
  }
  out.reasoning = strip_nl(text.substr(0, end), true, true);
  const size_t c0 = end + close.size();
  out.content_at = shift + c0 + lead_nl(text, c0);
  out.content = text.substr(c0 + lead_nl(text, c0));
  return out;
}

}  // namespace

// The first added token of the marker's own tokenization is its anchor.
MarkerToken marker_token(const Tokenizer& tok, const std::string& text) {
  MarkerToken m;
  m.text = text;
  size_t off = 0;
  for (int32_t id : tok.encode(text)) {
    if (tok.is_added(id)) {
      m.anchor = id;
      m.anchor_off = off;
      break;
    }
    off += tok.piece(id).size();
  }
  return m;
}


ReplyText::ReplyText(const Tokenizer& tok, const ChatMarkers& mk, std::string tool_open, bool reasoning_open,
                     bool split)
    : tok_(tok),
      mk_(mk),
      tool_open_(std::move(tool_open)),
      reasoning_open_(reasoning_open),
      split_(split),
      close_mark_(marker_token(tok, mk.think_close)),
      tool_mark_(marker_token(tok, tool_open_)) {}

void ReplyText::note(int32_t id, size_t at) {
  if (id == close_mark_.anchor && at >= close_mark_.anchor_off) close_raw_.push_back(at - close_mark_.anchor_off);
  if (id == tool_mark_.anchor && at >= tool_mark_.anchor_off) tool_raw_.push_back(at - tool_mark_.anchor_off);
}

void ReplyText::add(int32_t id) {
  note(id, raw_.size());
  raw_ += tok_.piece(id);
}

void ReplyText::add_text(const std::string& t) {
  size_t at = raw_.size();
  for (int32_t id : tok_.encode(t)) {
    note(id, at);
    at += tok_.piece(id).size();
  }
  raw_ += t;
}

ReplyText::View ReplyText::view(bool final_view) const {
  View v;
  const std::string_view shown = std::string_view(raw_).substr(0, final_view ? raw_.size() : json::utf8_complete_prefix(raw_));
  const std::string text = sanitize_utf8(shown);
  v.held = raw_.size() - shown.size();
  // Offsets of raw_ in `text` (they differ only after invalid UTF-8).
  const bool same = text.size() == shown.size();
  auto map = [&](const std::vector<size_t>& from) {
    std::vector<size_t> to;
    for (size_t p : from)
      if (p < shown.size()) to.push_back(same ? p : sanitize_utf8(shown.substr(0, p)).size());
    return to;
  };
  size_t content_at = 0;
  if (split_) {
    const std::vector<size_t> close_at = map(close_raw_);
    Split sp = split_reasoning(text, reasoning_open_, mk_, close_mark_.anchor >= 0 ? &close_at : nullptr);
    v.reasoning = std::move(sp.reasoning);
    v.content = std::move(sp.content);
    content_at = sp.content_at;
    v.held += sp.held;
  } else {
    v.content = text;
  }
  if (tool_mark_.anchor >= 0) {
    v.anchored = true;
    for (size_t p : map(tool_raw_))
      if (p >= content_at) v.tool_at.push_back(p - content_at);
  }
  return v;
}

size_t ReplyText::calls_start(const View& v, const std::string& content) const {
  return find_marker(content, tool_open_, v.calls_at());
}

std::string ReplyText::parse_calls(const View& v, const std::string& content, const json::Value& tools,
                                   json::Value& calls) const {
  return mk_.gemma ? parse_gemma_tool_calls(content, calls, v.calls_at())
                   : parse_tool_calls(content, tools, calls, v.calls_at());
}

// ---------------------------------------------------------------- tools
// DeepSeek-V3.2 / V4 DSML calls:
//   <｜DSML｜ calls>
//   <｜DSML｜ invoke name="NAME">
//   <｜DSML｜ parameter name="P" string="true|false">VALUE</｜DSML｜ parameter>
//   </｜DSML｜ invoke>
//   </｜DSML｜ calls>
// string="true": VALUE as is; otherwise JSON (kept as a string when it does not parse).
static std::string parse_dsml_calls(const std::string& content, json::Value& calls, const std::vector<size_t>* at_ok) {
  static const std::string kCalls = "<｜DSML｜ calls>";
  static const std::string kInvoke = "<｜DSML｜ invoke name=\"";
  static const std::string kInvokeEnd = "</｜DSML｜ invoke>";
  static const std::string kParam = "<｜DSML｜ parameter name=\"";
  static const std::string kParamEnd = "</｜DSML｜ parameter>";
  const size_t at = find_marker(content, kCalls, at_ok);
  if (at == std::string::npos) return content;
  std::string before = content.substr(0, at);
  while (!before.empty() && (before.back() == '\n' || before.back() == ' ')) before.pop_back();
  json::Value out = json::Value::array();
  size_t p = at + kCalls.size();
  while (true) {
    const size_t i0 = content.find(kInvoke, p);
    if (i0 == std::string::npos) break;
    const size_t n1 = content.find('"', i0 + kInvoke.size());
    if (n1 == std::string::npos) break;
    const std::string name = content.substr(i0 + kInvoke.size(), n1 - i0 - kInvoke.size());
    size_t iend = content.find(kInvokeEnd, n1);
    if (iend == std::string::npos) iend = content.size();
    json::Value args = json::Value::object();
    size_t q = n1;
    while (true) {
      const size_t a0 = content.find(kParam, q);
      if (a0 == std::string::npos || a0 >= iend) break;
      const size_t a1 = content.find('"', a0 + kParam.size());
      if (a1 == std::string::npos) break;
      const std::string pname = content.substr(a0 + kParam.size(), a1 - a0 - kParam.size());
      const size_t gt = content.find('>', a1);
      if (gt == std::string::npos) break;
      const bool is_string = content.substr(a1, gt - a1).find("string=\"true\"") != std::string::npos;
      size_t v1 = content.find(kParamEnd, gt + 1);
      if (v1 == std::string::npos || v1 > iend) v1 = iend;
      const std::string value = content.substr(gt + 1, v1 - gt - 1);
      json::Value v(value);
      if (!is_string) {
        try {
          v = json::parse(value);
        } catch (...) {
          v = json::Value(value);
        }
      }
      args.set(pname, std::move(v));
      q = v1 == iend ? iend : v1 + kParamEnd.size();
    }
    json::Value fn = json::Value::object();
    fn.set("name", name);
    fn.set("arguments", json::dump(args));
    json::Value call = json::Value::object();
    call.set("id", call_id());
    call.set("type", "function");
    call.set("function", std::move(fn));
    out.push_back(std::move(call));
    if (iend >= content.size()) break;
    p = iend + kInvokeEnd.size();
  }
  if (!out.size()) return content;
  calls = std::move(out);
  return before;
}

std::string parse_tool_calls(const std::string& content, const json::Value& tools, json::Value& calls,
                             const std::vector<size_t>* at_ok) {
  calls = json::Value();
  if (find_marker(content, "<｜DSML｜ calls>", at_ok) != std::string::npos) return parse_dsml_calls(content, calls, at_ok);
  const std::string open = "<tool_call>", close = "</tool_call>";
  size_t at = find_marker(content, open, at_ok);
  if (at == std::string::npos) return content;
  {  // GLM: <tool_call>NAME<arg_key>... (no <function=)
    const size_t b0 = at + open.size(), e = content.find(close, b0);
    if (glm_tool_call_body(content.substr(b0, e == std::string::npos ? std::string::npos : e - b0)))
      return parse_glm_tool_calls(content, tools, calls, at_ok);
  }
  std::string before = content.substr(0, at);
  while (!before.empty() && (before.back() == '\n' || before.back() == ' ')) before.pop_back();
  json::Value out = json::Value::array();
  // Declared parameter types, to keep string arguments as strings.
  auto param_type = [&](const std::string& fn, const std::string& p) -> std::string {
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
  };
  while (at != std::string::npos) {
    const size_t body0 = at + open.size();
    size_t end = content.find(close, body0);
    const std::string body = content.substr(body0, end == std::string::npos ? std::string::npos : end - body0);
    const size_t f0 = body.find("<function=");
    if (f0 != std::string::npos) {
      const size_t f1 = body.find('>', f0);
      std::string name = body.substr(f0 + 10, f1 == std::string::npos ? std::string::npos : f1 - f0 - 10);
      json::Value args = json::Value::object();
      size_t p = f1 == std::string::npos ? body.size() : f1;
      while (true) {
        const size_t p0 = body.find("<parameter=", p);
        if (p0 == std::string::npos) break;
        const size_t p1 = body.find('>', p0);
        if (p1 == std::string::npos) break;
        const std::string pname = body.substr(p0 + 11, p1 - p0 - 11);
        size_t v0 = p1 + 1;
        if (v0 < body.size() && body[v0] == '\n') ++v0;
        size_t v1 = body.find("</parameter>", v0);
        if (v1 == std::string::npos) v1 = body.find("<parameter=", v0);
        if (v1 == std::string::npos) v1 = body.find("</function>", v0);
        if (v1 == std::string::npos) v1 = body.size();
        std::string value = body.substr(v0, v1 - v0);
        if (!value.empty() && value.back() == '\n') value.pop_back();
        const std::string type = param_type(name, pname);
        json::Value v(value);
        if (type != "string") {
          try {
            v = json::parse(value);
          } catch (...) {
            v = json::Value(value);
          }
        }
        args.set(pname, std::move(v));
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
    at = find_marker(content, open, at_ok, end + close.size());
  }
  if (out.size()) calls = std::move(out);
  return calls.is_null() ? content : before;
}

}  // namespace espetero
