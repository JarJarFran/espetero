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
      tool_mark_(marker_token(tok, tool_open_)) {
  reset_cache();
}

void ReplyText::reset_cache() const {
  finalized_ = false;
  raw_scanned_ = 0;
  clean_.clear();
  clean_same_ = true;
  close_mapped_ = 0;
  tool_mapped_ = 0;
  close_clean_.clear();
  tool_clean_.clear();
  phase_ = !split_ ? Phase::Content : (reasoning_open_ ? Phase::Reasoning : Phase::Init);
  shift_ = 0;
  r_start_ = 0;
  r_end_ = 0;
  c_start_ = 0;
  r_skip_nl_ = split_ && reasoning_open_;
  c_skip_nl_ = false;
  close_scan_ = 0;
  close_idx_ = 0;
  tag_held_ = 0;
  tool_idx_ = 0;
  tool_at_.clear();
}

void ReplyText::note(int32_t id, size_t at) {
  if (id == close_mark_.anchor && at >= close_mark_.anchor_off) close_raw_.push_back(at - close_mark_.anchor_off);
  if (id == tool_mark_.anchor && at >= tool_mark_.anchor_off) tool_raw_.push_back(at - tool_mark_.anchor_off);
}

void ReplyText::add(int32_t id) {
  if (finalized_) reset_cache();
  note(id, raw_.size());
  raw_ += tok_.piece(id);
}

void ReplyText::add_text(const std::string& t) {
  if (finalized_) reset_cache();
  size_t at = raw_.size();
  for (int32_t id : tok_.encode(t)) {
    note(id, at);
    at += tok_.piece(id).size();
  }
  raw_ += t;
}

void ReplyText::advance(bool final_view) const {
  const size_t limit = final_view ? raw_.size() : json::utf8_complete_prefix(raw_);
  if (final_view && limit > raw_scanned_) finalized_ = true;
  if (limit > raw_scanned_) {
    const std::string_view delta(raw_.data() + raw_scanned_, limit - raw_scanned_);
    const size_t clean_before = clean_.size();
    if (json::utf8_valid(delta)) {
      clean_.append(delta);
      const ptrdiff_t diff = ptrdiff_t(clean_before) - ptrdiff_t(raw_scanned_);
      while (close_mapped_ < close_raw_.size() && close_raw_[close_mapped_] < limit)
        close_clean_.push_back(size_t(ptrdiff_t(close_raw_[close_mapped_++]) + diff));
      while (tool_mapped_ < tool_raw_.size() && tool_raw_[tool_mapped_] < limit)
        tool_clean_.push_back(size_t(ptrdiff_t(tool_raw_[tool_mapped_++]) + diff));
    } else {
      size_t i = 0;
      while (i < delta.size()) {
        const size_t rpos = raw_scanned_ + i;
        const size_t cpos = clean_.size();
        while (close_mapped_ < close_raw_.size() && close_raw_[close_mapped_] <= rpos) {
          close_clean_.push_back(cpos);
          ++close_mapped_;
        }
        while (tool_mapped_ < tool_raw_.size() && tool_raw_[tool_mapped_] <= rpos) {
          tool_clean_.push_back(cpos);
          ++tool_mapped_;
        }
        json::utf8_append(clean_, json::utf8_next(delta, i));
      }
      while (close_mapped_ < close_raw_.size() && close_raw_[close_mapped_] < limit) {
        close_clean_.push_back(clean_.size());
        ++close_mapped_;
      }
      while (tool_mapped_ < tool_raw_.size() && tool_raw_[tool_mapped_] < limit) {
        tool_clean_.push_back(clean_.size());
        ++tool_mapped_;
      }
    }
    raw_scanned_ = limit;
    clean_same_ = (clean_.size() == raw_scanned_);
  }

  if (phase_ == Phase::Init) {
    const std::string& open = mk_.think_open;
    if (clean_.size() < open.size()) {
      if (!clean_.empty() && open.compare(0, clean_.size(), clean_) == 0) {
        tag_held_ = clean_.size();
        return;
      }
      if (!clean_.empty()) {
        phase_ = Phase::Content;
        c_start_ = 0;
        c_skip_nl_ = false;
        tag_held_ = 0;
      }
    } else if (clean_.compare(0, open.size(), open) == 0) {
      phase_ = Phase::Reasoning;
      shift_ = open.size();
      r_start_ = shift_;
      r_end_ = shift_;
      r_skip_nl_ = true;
      close_scan_ = shift_;
      tag_held_ = 0;
    } else {
      phase_ = Phase::Content;
      c_start_ = 0;
      c_skip_nl_ = false;
      tag_held_ = 0;
    }
  }

  if (phase_ == Phase::Reasoning) {
    const std::string& close = mk_.think_close;
    size_t end = std::string::npos;
    if (close_mark_.anchor < 0) {
      end = clean_.find(close, close_scan_);
      if (end == std::string::npos) {
        const size_t keep = close.size() > 1 ? close.size() - 1 : 0;
        close_scan_ = clean_.size() > shift_ + keep ? clean_.size() - keep : shift_;
      }
    } else {
      while (close_idx_ < close_clean_.size()) {
        const size_t p = close_clean_[close_idx_];
        if (p < shift_) {
          ++close_idx_;
          continue;
        }
        if (p + close.size() > clean_.size()) break;  // wait for the rest of the close marker
        if (clean_.compare(p, close.size(), close) == 0) {
          end = p;
          break;
        }
        ++close_idx_;
      }
    }
    if (end == std::string::npos) {
      size_t held = 0;
      const size_t rlen = clean_.size() - shift_;
      for (size_t k = close.size() > 0 ? close.size() - 1 : 0; k > 0; --k)
        if (rlen >= k && clean_.compare(clean_.size() - k, k, close, 0, k) == 0) {
          held = k;
          break;
        }
      tag_held_ = held;
      const size_t vis_end = clean_.size() - held;
      if (r_skip_nl_) {
        while (r_start_ < vis_end && clean_[r_start_] == '\n') ++r_start_;
        if (r_start_ < vis_end) r_skip_nl_ = false;
      }
      r_end_ = std::max(r_start_, vis_end);
    } else {
      tag_held_ = 0;
      if (r_skip_nl_) {
        while (r_start_ < end && clean_[r_start_] == '\n') ++r_start_;
        r_skip_nl_ = false;
      }
      r_end_ = end;
      while (r_end_ > r_start_ && clean_[r_end_ - 1] == '\n') --r_end_;
      c_start_ = end + close.size();
      c_skip_nl_ = true;
      phase_ = Phase::Content;
    }
  }

  if (phase_ == Phase::Content) {
    if (c_skip_nl_) {
      while (c_start_ < clean_.size() && clean_[c_start_] == '\n') ++c_start_;
      if (c_start_ < clean_.size()) c_skip_nl_ = false;
    }
    if (tool_mark_.anchor >= 0 && !c_skip_nl_) {
      while (tool_idx_ < tool_clean_.size()) {
        const size_t p = tool_clean_[tool_idx_++];
        if (p >= c_start_) tool_at_.push_back(p - c_start_);
      }
    }
  }
}

ReplyText::StreamView ReplyText::stream_view(bool final_view) const {
  advance(final_view);
  StreamView v;
  v.held = (raw_.size() - raw_scanned_) + tag_held_;
  if (phase_ == Phase::Reasoning) {
    v.reasoning = std::string_view(clean_).substr(r_start_, r_end_ - r_start_);
  } else if (phase_ == Phase::Content) {
    if (r_end_ > r_start_) v.reasoning = std::string_view(clean_).substr(r_start_, r_end_ - r_start_);
    if (c_start_ < clean_.size()) v.content = std::string_view(clean_).substr(c_start_);
  }
  if (tool_mark_.anchor >= 0) v.calls_at = &tool_at_;
  return v;
}

ReplyText::View ReplyText::view(bool final_view) const {
  const StreamView sv = stream_view(final_view);
  View v;
  v.reasoning.assign(sv.reasoning);
  v.content.assign(sv.content);
  v.held = sv.held;
  if (sv.calls_at) {
    v.anchored = true;
    v.tool_at = *sv.calls_at;
  }
  return v;
}

size_t ReplyText::calls_start(const View& v, const std::string& content) const {
  return find_marker(content, tool_open_, v.calls_at());
}

size_t ReplyText::calls_start(const StreamView& v, std::string_view content, size_t from) const {
  if (!v.calls_at) return content.find(tool_open_, from);
  for (size_t p : *v.calls_at)
    if (p >= from && p + tool_open_.size() <= content.size() && content.compare(p, tool_open_.size(), tool_open_) == 0)
      return p;
  return std::string::npos;
}

std::string ReplyText::parse_calls(const View& v, const std::string& content, const json::Value& tools,
                                   json::Value& calls) const {
  return mk_.gemma ? parse_gemma_tool_calls(content, calls, v.calls_at())
                   : parse_tool_calls(content, tools, calls, v.calls_at());
}

std::string ReplyText::parse_calls(const StreamView& v, const std::string& content, const json::Value& tools,
                                   json::Value& calls) const {
  return mk_.gemma ? parse_gemma_tool_calls(content, calls, v.calls_at)
                   : parse_tool_calls(content, tools, calls, v.calls_at);
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
