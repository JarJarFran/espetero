// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Other client APIs over the chat job: Anthropic Messages
// (/v1/messages, /v1/messages/count_tokens), OpenAI Responses (/v1/responses,
// stored responses for previous_response_id) and Prometheus /metrics.
//
// Each request is translated into an OpenAI chat body and run through
// prepare_chat / run_chat like /v1/chat/completions, so the prompt and the
// tokens are those of the chat route; only the request and response shapes
// differ. Text-only contents become plain strings (the template renders a
// list of text parts as their concatenation, so the prompt is the same).
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <deque>
#include <functional>
#include <mutex>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "../engine/chat.h"
#include "../engine/structured.h"
#include "api.h"
#include "version.h"

namespace espetero {

namespace {

[[noreturn]] void invalid(const std::string& msg) { throw RequestError(400, msg); }

json::Value obj() { return json::Value::object(); }
json::Value arr() { return json::Value::array(); }

json::Value parse_body(const http::Request& req) {
  json::Value v;
  try {
    v = json::parse(req.body);
  } catch (const std::exception& e) {
    throw RequestError(400, std::string("invalid JSON body: ") + e.what());
  }
  if (!v.is_object()) invalid("request body must be a JSON object");
  return v;
}

// The random part of the chat job's id ("chatcmpl-XXXX" -> "XXXX").
std::string id_suffix(const ChatJob& job) {
  const size_t dash = job.id.find('-');
  return dash == std::string::npos ? job.id : job.id.substr(dash + 1);
}

// Text of a string or of a list of text blocks.
std::string text_of(const json::Value* v) {
  if (!v || v->is_null()) return "";
  if (v->is_string()) return v->as_string();
  std::string out;
  if (v->is_array())
    for (const json::Value& b : v->as_array()) {
      const std::string type = b.get_string("type", "");
      if (type == "text" || type == "input_text" || type == "output_text") out += b.get_string("text", "");
    }
  return out;
}

json::Value text_part(const std::string& text) {
  json::Value t = obj();
  t.set("type", "text");
  t.set("text", text);
  return t;
}
json::Value image_part(const std::string& url, const std::string& detail) {
  json::Value iu = obj();
  iu.set("url", url);
  if (!detail.empty()) iu.set("detail", detail);
  json::Value t = obj();
  t.set("type", "image_url");
  t.set("image_url", std::move(iu));
  return t;
}
// A message content: the parts, or their text when there are no images and no
// cache marks (a text part's cache_control or checkpoint must reach the chat
// route as a part).
json::Value content_of(json::Value parts) {
  bool keep = false;
  for (const json::Value& p : parts.as_array())
    keep |= p.get_string("type", "") != "text" || p.get("cache_control") != nullptr || p.get("checkpoint") != nullptr;
  if (keep) return parts;
  return json::Value(text_of(&parts));
}

// A tool call's arguments as an object (Anthropic "input") or a JSON string
// (OpenAI "arguments").
json::Value arguments_object(const json::Value& call) {
  const json::Value* fn = call.get("function");
  const json::Value* args = fn ? fn->get("arguments") : nullptr;
  if (args && args->is_object()) return *args;
  if (args && args->is_string()) {
    try {
      json::Value v = json::parse(args->as_string());
      if (v.is_object()) return v;
    } catch (...) {
    }
  }
  return obj();
}
std::string arguments_string(const json::Value& call) {
  const json::Value* fn = call.get("function");
  const json::Value* args = fn ? fn->get("arguments") : nullptr;
  if (args && args->is_string()) return args->as_string();
  return args ? json::dump(*args) : "{}";
}
std::string call_name(const json::Value& call) {
  const json::Value* fn = call.get("function");
  return fn ? fn->get_string("name", "") : std::string();
}
bool has_calls(const ChatEvent& ev) { return ev.tool_calls.is_array() && ev.tool_calls.size(); }

int64_t usage_int(const json::Value& usage, const char* key) { return usage.is_object() ? usage.get_int(key, 0) : 0; }
int64_t cached_tokens(const json::Value& usage) {
  const json::Value* d = usage.is_object() ? usage.get("prompt_tokens_details") : nullptr;
  return d ? d->get_int("cached_tokens", 0) : 0;
}

// Sampling fields both APIs share with the chat body (plus Espetero's own).
void copy_sampling(const json::Value& b, json::Value& out) {
  for (const char* k : {"temperature", "top_p", "top_k", "min_p", "stream", "seed", "speculative", "cache_prompt",
                        "chat_template_kwargs", "priority"})
    if (const json::Value* v = b.get(k); v && !v->is_null()) out.set(k, *v);
}

// One Server-Sent Event with an event name (Anthropic and Responses streams).
bool event(http::Response& res, const std::string& name, const json::Value& data) {
  return res.write("event: " + name + "\ndata: " + json::dump(data) + "\n\n");
}

// ---------------------------------------------------------------- Anthropic Messages

// An image block's source as the chat route's URL: a data URL for "base64",
// the URL itself for "url" (fetched with --image-urls).
std::string image_url_of(const json::Value& block) {
  const json::Value* src = block.get("source");
  if (!src || !src->is_object()) invalid("image: source required");
  const std::string type = src->get_string("type", "");
  if (type == "url") {
    const std::string url = src->get_string("url", "");
    if (url.empty()) invalid("image: source.url required");
    return url;
  }
  if (type != "base64") invalid("image: source type base64 or url (file sources are not supported)");
  return "data:" + src->get_string("media_type", "image/jpeg") + ";base64," + src->get_string("data", "");
}

json::Value anthropic_to_chat(const json::Value& b) {
  json::Value msgs = arr();
  if (const json::Value* sys = b.get("system"); sys && !sys->is_null()) {
    json::Value m = obj();
    m.set("role", "system");
    m.set("content", text_of(sys));
    msgs.push_back(std::move(m));
  }
  const json::Value* ms = b.get("messages");
  if (!ms || !ms->is_array() || ms->size() == 0) invalid("messages: a nonempty list is required");
  for (const json::Value& m : ms->as_array()) {
    std::string role = m.get_string("role", "");
    // Claude Code puts system reminders between turns as {"role": "system"}
    // messages. Chat templates take one system message, first: before any turn
    // they extend the system prompt, later they join the conversation as user
    // text at their place.
    if (role == "system") {
      const bool leading = msgs.size() == 0 || (msgs.size() == 1 && msgs[0].get_string("role", "") == "system");
      if (leading) {
        const std::string text = text_of(m.get("content"));
        if (msgs.size() == 0) {
          json::Value s = obj();
          s.set("role", "system");
          s.set("content", text);
          msgs.push_back(std::move(s));
        } else {
          msgs.as_array()[0].set("content", text_of(msgs[0].get("content")) + "\n\n" + text);
        }
        continue;
      }
      role = "user";
    }
    if (role != "user" && role != "assistant") invalid("messages: role must be user, assistant or system");
    const json::Value* content = m.get("content");
    if (content && content->is_string()) {
      json::Value o = obj();
      o.set("role", role);
      o.set("content", content->as_string());
      msgs.push_back(std::move(o));
      continue;
    }
    if (!content || !content->is_array()) invalid("messages: content must be a string or a list of blocks");
    json::Value parts = arr(), calls = arr();
    std::string reasoning;
    std::vector<json::Value> results;  // tool_result blocks: role "tool" messages, before the rest
    for (const json::Value& blk : content->as_array()) {
      const std::string type = blk.get_string("type", "");
      if (type == "text") {
        json::Value t = text_part(blk.get_string("text", ""));
        // A cache breakpoint ends a prefix the engine keeps for later requests.
        if (const json::Value* cc = blk.get("cache_control"); cc && cc->is_object()) t.set("cache_control", *cc);
        parts.push_back(std::move(t));
      } else if (type == "image") {
        parts.push_back(image_part(image_url_of(blk), ""));
      } else if (type == "thinking") {
        reasoning += blk.get_string("thinking", "");
      } else if (type == "redacted_thinking") {
        // opaque: nothing to render
      } else if (type == "tool_use") {
        json::Value fn = obj();
        fn.set("name", blk.get_string("name", ""));
        const json::Value* in = blk.get("input");
        fn.set("arguments", in ? json::dump(*in) : std::string("{}"));
        json::Value c = obj();
        c.set("id", blk.get_string("id", ""));
        c.set("type", "function");
        c.set("function", std::move(fn));
        calls.push_back(std::move(c));
      } else if (type == "tool_result") {
        if (const json::Value* rc = blk.get("content"); rc && rc->is_array())
          for (const json::Value& part : rc->as_array())
            if (part.get_string("type", "") != "text") invalid("tool_result: only text content (images are not supported)");
        json::Value t = obj();
        t.set("role", "tool");
        t.set("tool_call_id", blk.get_string("tool_use_id", ""));
        std::string text = text_of(blk.get("content"));
        if (blk.get_bool("is_error", false)) text = "Error: " + text;
        t.set("content", text);
        results.push_back(std::move(t));
      } else {
        invalid("unsupported content block '" + type + "'");
      }
    }
    for (json::Value& t : results) msgs.push_back(std::move(t));
    if (parts.size() || calls.size() || !reasoning.empty() || results.empty()) {
      json::Value o = obj();
      o.set("role", role);
      o.set("content", content_of(std::move(parts)));
      if (!reasoning.empty()) o.set("reasoning_content", reasoning);
      if (calls.size()) o.set("tool_calls", std::move(calls));
      msgs.push_back(std::move(o));
    }
  }
  json::Value out = obj();
  out.set("messages", std::move(msgs));
  copy_sampling(b, out);
  if (const json::Value* v = b.get("max_tokens")) out.set("max_tokens", *v);
  if (const json::Value* v = b.get("stop_sequences"); v && v->is_array()) out.set("stop", *v);
  // Extended thinking is off unless the request enables it (the Anthropic
  // API's rule): "enabled" (with a budget) or "adaptive".
  bool think = false;
  if (const json::Value* t = b.get("thinking"); t && t->is_object()) {
    const std::string type = t->get_string("type", "");
    think = type == "enabled" || type == "adaptive";
  }
  out.set("enable_thinking", think);
  // output_config.effort reaches templates that take a reasoning effort
  // (Qwen3.8-Flash-Next: low, medium, high = xhigh), as on the Responses API.
  if (const json::Value* oc = b.get("output_config"); oc && oc->is_object()) {
    if (const json::Value* e = oc->get("effort"); e && e->is_string()) {
      const std::string& effort = e->as_string();
      if (effort == "low" || effort == "medium" || effort == "high" || effort == "xhigh" || effort == "max") {
        const json::Value* given = out.get("chat_template_kwargs");
        json::Value kw = given && given->is_object() ? *given : obj();
        if (!kw.get("reasoning_effort"))
          kw.set("reasoning_effort", effort == "low" || effort == "medium" ? effort : std::string("xhigh"));
        out.set("chat_template_kwargs", std::move(kw));
      }
    }
  }
  // Structured output: output_config.format (or the deprecated
  // output_format), {"type": "json_schema", "schema": {...}}: a guarantee on
  // Anthropic's API, so a strict json_schema here.
  const json::Value* fmt = nullptr;
  if (const json::Value* oc = b.get("output_config"); oc && oc->is_object()) fmt = oc->get("format");
  if (!fmt || fmt->is_null()) fmt = b.get("output_format");
  if (fmt && !fmt->is_null()) {
    if (!fmt->is_object() || fmt->get_string("type", "") != "json_schema" || !fmt->get("schema"))
      invalid("output_config.format: {\"type\": \"json_schema\", \"schema\": {...}}");
    json::Value js = obj();
    js.set("name", "output");
    js.set("schema", *fmt->get("schema"));
    js.set("strict", true);
    json::Value rf = obj();
    rf.set("type", "json_schema");
    rf.set("json_schema", std::move(js));
    out.set("response_format", std::move(rf));
  }
  // tool_choice: auto (the default), none (no tools in the prompt), any or
  // tool (a forced call, the chat route's "required" or named function).
  std::string choice = "auto";
  if (const json::Value* tc = b.get("tool_choice"); tc && !tc->is_null()) {
    if (!tc->is_object()) invalid("tool_choice: an object such as {\"type\": \"auto\"}");
    choice = tc->get_string("type", "");
    if (choice == "any") {
      out.set("tool_choice", "required");
    } else if (choice == "tool") {
      const std::string name = tc->get_string("name", "");
      if (name.empty()) invalid("tool_choice: type tool needs a name");
      json::Value fn = obj();
      fn.set("name", name);
      json::Value c = obj();
      c.set("type", "function");
      c.set("function", std::move(fn));
      out.set("tool_choice", std::move(c));
    } else if (choice != "auto" && choice != "none") {
      invalid("tool_choice: type auto, any, tool or none");
    }
  }
  if (const json::Value* tools = b.get("tools"); tools && tools->is_array() && tools->size()) {
    if (choice != "none") {
      json::Value ts = arr();
      for (const json::Value& t : tools->as_array()) {
        const std::string type = t.get_string("type", "custom");
        if (type != "custom") invalid("tools: only custom tools (server tool '" + type + "' is not supported)");
        json::Value fn = obj();
        fn.set("name", t.get_string("name", ""));
        if (const json::Value* d = t.get("description")) fn.set("description", *d);
        if (const json::Value* s = t.get("input_schema")) fn.set("parameters", *s);
        json::Value w = obj();
        w.set("type", "function");
        w.set("function", std::move(fn));
        ts.push_back(std::move(w));
      }
      out.set("tools", std::move(ts));
    }
  }
  return out;
}

std::string stop_reason(const ChatEvent& last) {
  if (last.finish_reason == "length") return "max_tokens";
  if (last.finish_reason == "tool_calls") return "tool_use";
  if (!last.stop_sequence.empty()) return "stop_sequence";
  return "end_turn";
}
json::Value stop_sequence(const ChatEvent& last) {
  return last.stop_sequence.empty() ? json::Value() : json::Value(last.stop_sequence);
}

json::Value anthropic_usage(const json::Value& usage) {
  json::Value u = obj();
  u.set("input_tokens", usage_int(usage, "prompt_tokens") - cached_tokens(usage));
  u.set("cache_creation_input_tokens", 0);
  u.set("cache_read_input_tokens", cached_tokens(usage));
  u.set("output_tokens", usage_int(usage, "completion_tokens"));
  return u;
}

json::Value tool_use_block(const json::Value& call, bool with_input) {
  json::Value t = obj();
  t.set("type", "tool_use");
  t.set("id", call.get_string("id", ""));
  t.set("name", call_name(call));
  t.set("input", with_input ? arguments_object(call) : obj());
  return t;
}

// thinking.display "omitted" asks for the reply without its thinking text.
bool thinking_shown(const json::Value& b) {
  const json::Value* t = b.get("thinking");
  return !(t && t->is_object() && t->get_string("display", "") == "omitted");
}

json::Value anthropic_message(const Engine& engine, const ChatJob& job, const std::string& content,
                              const std::string& reasoning, const ChatEvent& last, bool show_thinking = true) {
  json::Value blocks = arr();
  if (!reasoning.empty() && show_thinking) {
    json::Value t = obj();
    t.set("type", "thinking");
    t.set("thinking", reasoning);
    t.set("signature", "");
    blocks.push_back(std::move(t));
  }
  if (!content.empty() || ((reasoning.empty() || !show_thinking) && !has_calls(last))) {
    json::Value t = obj();
    t.set("type", "text");
    t.set("text", content);
    blocks.push_back(std::move(t));
  }
  if (has_calls(last))
    for (const json::Value& c : last.tool_calls.as_array()) blocks.push_back(tool_use_block(c, true));
  json::Value m = obj();
  m.set("id", "msg_" + id_suffix(job));
  m.set("type", "message");
  m.set("role", "assistant");
  m.set("model", engine.model_name());
  m.set("content", std::move(blocks));
  m.set("stop_reason", stop_reason(last));
  m.set("stop_sequence", stop_sequence(last));
  m.set("usage", anthropic_usage(last.usage));
  return m;
}

json::Value anthropic_error(int status, const std::string& msg) {
  json::Value err = obj();
  err.set("type", status == 404   ? "not_found_error"
                  : status == 413 ? "request_too_large"
                  : status >= 500 ? "api_error"
                                  : "invalid_request_error");
  err.set("message", msg);
  json::Value o = obj();
  o.set("type", "error");
  o.set("error", std::move(err));
  return o;
}

void anthropic_stream(Engine& engine, ChatJob& job, http::Response& res, bool show_thinking = true) {
  if (!res.start_stream(200, "text/event-stream")) return;
  {
    json::Value u = obj();
    u.set("input_tokens", int64_t(job.gen.prompt.size()));
    u.set("output_tokens", 0);
    json::Value m = obj();
    m.set("id", "msg_" + id_suffix(job));
    m.set("type", "message");
    m.set("role", "assistant");
    m.set("model", engine.model_name());
    m.set("content", arr());
    m.set("stop_reason", json::Value());
    m.set("stop_sequence", json::Value());
    m.set("usage", std::move(u));
    json::Value start = obj();
    start.set("type", "message_start");
    start.set("message", std::move(m));
    if (!event(res, "message_start", start)) return;
  }
  int index = -1;
  std::string open;  // the block being streamed: "thinking" or "text"
  auto block_start = [&](json::Value blk) {
    json::Value d = obj();
    d.set("type", "content_block_start");
    d.set("index", ++index);
    d.set("content_block", std::move(blk));
    return event(res, "content_block_start", d);
  };
  auto block_delta = [&](json::Value dl) {
    json::Value d = obj();
    d.set("type", "content_block_delta");
    d.set("index", index);
    d.set("delta", std::move(dl));
    return event(res, "content_block_delta", d);
  };
  auto block_stop = [&]() {
    json::Value d = obj();
    d.set("type", "content_block_stop");
    d.set("index", index);
    return event(res, "content_block_stop", d);
  };
  auto close = [&]() {
    if (open.empty()) return true;
    if (open == "thinking") {
      json::Value dl = obj();
      dl.set("type", "signature_delta");
      dl.set("signature", "");
      if (!block_delta(std::move(dl))) return false;
    }
    open.clear();
    return block_stop();
  };
  auto delta = [&](const std::string& kind, const std::string& text) {
    if (open != kind) {
      if (!close()) return false;
      open = kind;
      json::Value blk = obj();
      blk.set("type", kind);
      if (kind == "thinking") {
        blk.set("thinking", "");
        blk.set("signature", "");
      } else {
        blk.set("text", "");
      }
      if (!block_start(std::move(blk))) return false;
    }
    json::Value dl = obj();
    dl.set("type", kind == "thinking" ? "thinking_delta" : "text_delta");
    dl.set(kind == "thinking" ? "thinking" : "text", text);
    return block_delta(std::move(dl));
  };
  bool any = false;
  run_chat(engine, job, [&](const ChatEvent& ev) {
    if (show_thinking && !ev.reasoning.empty() && !delta("thinking", ev.reasoning)) return false;
    if (!ev.content.empty() && !delta("text", ev.content)) return false;
    any |= (show_thinking && !ev.reasoning.empty()) || !ev.content.empty();
    if (!ev.final) return true;
    if (!any && !has_calls(ev) && !delta("text", "")) return false;
    if (!close()) return false;
    if (has_calls(ev))
      for (const json::Value& c : ev.tool_calls.as_array()) {
        if (!block_start(tool_use_block(c, false))) return false;
        json::Value dl = obj();
        dl.set("type", "input_json_delta");
        dl.set("partial_json", json::dump(arguments_object(c)));
        if (!block_delta(std::move(dl)) || !block_stop()) return false;
      }
    json::Value md = obj();
    md.set("stop_reason", stop_reason(ev));
    md.set("stop_sequence", stop_sequence(ev));
    json::Value m = obj();
    m.set("type", "message_delta");
    m.set("delta", std::move(md));
    m.set("usage", anthropic_usage(ev.usage));
    if (!event(res, "message_delta", m)) return false;
    json::Value stop = obj();
    stop.set("type", "message_stop");
    return event(res, "message_stop", stop);
  });
  res.end();
}

// ---------------------------------------------------------------- OpenAI Responses

// Stored responses (store: true, the API's default): the response object and
// the conversation it ends (chat messages without the instructions, which
// are not carried over to a continuation), the newest kCount within kBytes.
class ResponseStore {
 public:
  static constexpr size_t kCount = 256;
  static constexpr size_t kBytes = size_t(256) << 20;
  void put(const std::string& id, json::Value response, json::Value history) {
    const size_t bytes = json::dump(response).size() + json::dump(history).size();
    std::lock_guard<std::mutex> lk(mu_);
    erase_locked(id);
    map_[id] = Entry{std::move(response), std::move(history), bytes};
    order_.push_back(id);
    bytes_ += bytes;
    while (!order_.empty() && (map_.size() > kCount || bytes_ > kBytes)) {
      const std::string old = order_.front();
      if (!erase_locked(old)) order_.pop_front();
    }
  }
  bool history(const std::string& id, json::Value& out) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = map_.find(id);
    if (it == map_.end()) return false;
    out = it->second.history;
    return true;
  }
  bool response(const std::string& id, json::Value& out) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = map_.find(id);
    if (it == map_.end()) return false;
    out = it->second.response;
    return true;
  }
  bool erase(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    return erase_locked(id);
  }

 private:
  struct Entry {
    json::Value response, history;
    size_t bytes = 0;
  };
  bool erase_locked(const std::string& id) {
    auto it = map_.find(id);
    if (it == map_.end()) return false;
    bytes_ -= it->second.bytes;
    map_.erase(it);
    for (auto o = order_.begin(); o != order_.end(); ++o)
      if (*o == id) {
        order_.erase(o);
        break;
      }
    return true;
  }
  std::mutex mu_;
  std::unordered_map<std::string, Entry> map_;
  std::deque<std::string> order_;
  size_t bytes_ = 0;
};

ResponseStore& response_store() {
  static ResponseStore store;
  return store;
}

// Input items as chat messages, appended to `msgs`.
void responses_input(const json::Value& input, json::Value& msgs) {
  if (input.is_string()) {
    json::Value m = obj();
    m.set("role", "user");
    m.set("content", input.as_string());
    msgs.push_back(std::move(m));
    return;
  }
  if (!input.is_array()) invalid("input: a string or a list of items");
  // Consecutive function_call items are one assistant turn's calls: those of
  // the assistant message right before them (a client echoing [message,
  // function_call] output items), else an assistant message of their own.
  // A reasoning item's text goes to the assistant turn after it.
  json::Value calls = arr();
  bool after_assistant = false, calls_after_assistant = false;
  std::string reasoning;
  auto take_reasoning = [&](json::Value& m) {
    if (!reasoning.empty()) m.set("reasoning_content", reasoning);
    reasoning.clear();
  };
  auto flush_calls = [&]() {
    if (!calls.size()) return;
    json::Value* last = msgs.size() ? &msgs.as_array().back() : nullptr;
    if (calls_after_assistant && last && last->get_string("role", "") == "assistant" && !last->get("tool_calls")) {
      last->set("tool_calls", std::move(calls));
    } else {
      json::Value m = obj();
      m.set("role", "assistant");
      m.set("content", "");
      take_reasoning(m);
      m.set("tool_calls", std::move(calls));
      msgs.push_back(std::move(m));
    }
    calls = arr();
  };
  for (const json::Value& it : input.as_array()) {
    if (!it.is_object()) invalid("input: items must be objects");
    const std::string type = it.get_string("type", it.get("role") ? "message" : "");
    const bool was_after_assistant = after_assistant;
    after_assistant = false;
    if (type == "function_call") {
      if (!calls.size()) calls_after_assistant = was_after_assistant;
      json::Value fn = obj();
      fn.set("name", it.get_string("name", ""));
      fn.set("arguments", it.get_string("arguments", "{}"));
      json::Value c = obj();
      c.set("id", it.get_string("call_id", ""));
      c.set("type", "function");
      c.set("function", std::move(fn));
      calls.push_back(std::move(c));
      continue;
    }
    if (type == "reasoning") {
      after_assistant = was_after_assistant;  // between a message and its calls
      for (const char* key : {"content", "summary"})
        if (const json::Value* parts = it.get(key); parts && parts->is_array() && reasoning.empty())
          for (const json::Value& p : parts->as_array()) reasoning += p.get_string("text", "");
      continue;
    }
    flush_calls();
    if (type == "function_call_output") {
      json::Value m = obj();
      m.set("role", "tool");
      m.set("tool_call_id", it.get_string("call_id", ""));
      m.set("content", text_of(it.get("output")));
      msgs.push_back(std::move(m));
    } else if (type == "message") {
      std::string role = it.get_string("role", "user");
      if (role == "developer") role = "system";
      if (role != "system" && role != "user" && role != "assistant") invalid("input: unsupported role '" + role + "'");
      const json::Value* content = it.get("content");
      json::Value m = obj();
      m.set("role", role);
      if (role == "assistant") {
        take_reasoning(m);
        after_assistant = true;
      } else {
        reasoning.clear();  // reasoning with no assistant turn after it
      }
      if (content && content->is_string()) {
        m.set("content", content->as_string());
      } else if (content && content->is_array()) {
        json::Value parts = arr();
        for (const json::Value& p : content->as_array()) {
          const std::string pt = p.get_string("type", "");
          if (pt == "input_text" || pt == "output_text" || pt == "text") {
            parts.push_back(text_part(p.get_string("text", "")));
          } else if (pt == "refusal") {
            parts.push_back(text_part(p.get_string("refusal", "")));
          } else if (pt == "input_image") {
            const json::Value* u = p.get("image_url");
            if (!u || !u->is_string())
              invalid("input_image: image_url (a data URL, or an http(s) URL with --image-urls) required; file ids are not supported");
            parts.push_back(image_part(u->as_string(), p.get_string("detail", "auto")));
          } else {
            invalid("input: unsupported content part '" + pt + "'");
          }
        }
        m.set("content", content_of(std::move(parts)));
      } else {
        invalid("input: a message needs content");
      }
      msgs.push_back(std::move(m));
    } else {
      invalid("input: unsupported item '" + type + "'");
    }
  }
  flush_calls();
}

// The chat body of a Responses request; `history` gets the conversation
// without the instructions (what a continuation starts from).
json::Value responses_to_chat(const json::Value& b, json::Value& history) {
  history = arr();
  if (const json::Value* prev = b.get("previous_response_id"); prev && prev->is_string()) {
    if (!response_store().history(prev->as_string(), history))
      throw RequestError(404, "previous_response_id: no stored response '" + prev->as_string() + "'");
  }
  const json::Value* input = b.get("input");
  if (!input) invalid("input: required");
  responses_input(*input, history);
  // Text.format json_object / json_schema: the chat route's response_format.
  json::Value response_format;
  if (const json::Value* t = b.get("text"); t && t->is_object()) response_format = responses_response_format(t->get("format"));
  // Logprobs are served on /v1/chat/completions; here they are refused, not ignored.
  if (const json::Value* tl = b.get("top_logprobs"); tl && tl->is_number() && tl->as_double() > 0)
    throw RequestError(422, "top_logprobs: not supported on /v1/responses yet (/v1/chat/completions returns logprobs)");
  if (const json::Value* inc = b.get("include"); inc && inc->is_array())
    for (const json::Value& x : inc->as_array())
      if (x.is_string() && x.as_string() == "message.output_text.logprobs")
        throw RequestError(422, "include: message.output_text.logprobs is not supported on /v1/responses yet "
                                "(/v1/chat/completions returns logprobs)");
  // One leading system message (templates take one, at the beginning): the
  // instructions, then the conversation's system and developer items.
  std::string system;
  if (const json::Value* ins = b.get("instructions"); ins && ins->is_string()) system = ins->as_string();
  json::Value msgs = arr();
  for (const json::Value& m : history.as_array()) {
    if (m.get_string("role", "") != "system") continue;
    const std::string t = text_of(m.get("content"));
    if (!t.empty()) system += (system.empty() ? "" : "\n\n") + t;
  }
  if (!system.empty()) {
    json::Value sm = obj();
    sm.set("role", "system");
    sm.set("content", system);
    msgs.push_back(std::move(sm));
  }
  for (const json::Value& m : history.as_array())
    if (m.get_string("role", "") != "system") msgs.push_back(m);
  json::Value out = obj();
  out.set("messages", std::move(msgs));
  copy_sampling(b, out);
  if (!response_format.is_null()) out.set("response_format", std::move(response_format));
  if (const json::Value* v = b.get("max_output_tokens"); v && !v->is_null()) out.set("max_tokens", *v);
  // reasoning.effort none / minimal turns thinking off, any other effort on;
  // no reasoning field keeps the template's default (as the chat route does).
  // Templates with a reasoning_effort argument (Qwen3.8-Flash-Next: low,
  // medium, xhigh) get the request's effort.
  if (const json::Value* r = b.get("reasoning"); r && r->is_object())
    if (const json::Value* e = r->get("effort"); e && e->is_string()) {
      const std::string& effort = e->as_string();
      out.set("enable_thinking", effort != "none" && effort != "minimal");
      if (effort == "low" || effort == "medium" || effort == "high" || effort == "xhigh") {
        json::Value kw = out.get("chat_template_kwargs") && out.get("chat_template_kwargs")->is_object()
                             ? *out.get("chat_template_kwargs") : obj();
        if (!kw.get("reasoning_effort")) kw.set("reasoning_effort", effort == "high" ? std::string("xhigh") : effort);
        out.set("chat_template_kwargs", std::move(kw));
      }
    }
  // tool_choice: "auto" (the default), "none" (no tools in the prompt),
  // "required", a function ({"type": "function", "name": ...}) or
  // allowed_tools, in the chat route's shape.
  bool offer_tools = true;
  if (const json::Value* tc = b.get("tool_choice"); tc && !tc->is_null()) {
    const std::string choice = tc->is_string() ? tc->as_string() : std::string();
    auto chat_function = [](const std::string& name) {
      json::Value fn = obj();
      fn.set("name", name);
      json::Value c = obj();
      c.set("type", "function");
      c.set("function", std::move(fn));
      return c;
    };
    if (choice == "none") {
      offer_tools = false;
    } else if (choice == "required") {
      out.set("tool_choice", "required");
    } else if (tc->is_object() && tc->get_string("type", "") == "function") {
      out.set("tool_choice", chat_function(tc->get_string("name", "")));
    } else if (tc->is_object() && tc->get_string("type", "") == "allowed_tools") {
      json::Value tools = arr();
      if (const json::Value* l = tc->get("tools"); l && l->is_array())
        for (const json::Value& t : l->as_array()) {
          if (t.get_string("type", "") != "function") throw RequestError(422, "tool_choice: allowed_tools lists functions only");
          tools.push_back(chat_function(t.get_string("name", "")));
        }
      json::Value at = obj();
      at.set("mode", tc->get_string("mode", "auto"));
      at.set("tools", std::move(tools));
      json::Value c = obj();
      c.set("type", "allowed_tools");
      c.set("allowed_tools", std::move(at));
      out.set("tool_choice", std::move(c));
    } else if (tc->is_object()) {
      throw RequestError(422, "tool_choice: a function, allowed_tools, \"auto\", \"none\" or \"required\" (no built-in tools)");
    } else if (choice != "auto") {
      invalid("tool_choice: \"auto\", \"none\", \"required\" or a tool");
    }
  }
  if (const json::Value* tools = b.get("tools"); tools && tools->is_array() && tools->size()) {
    // Tools this server cannot run are refused, named, with the fix for the
    // client that sends them by default (Codex CLI: its hosted web search and
    // its sub-agent tools, a namespace of functions).
    std::vector<std::string> other;
    for (const json::Value& t : tools->as_array()) {
      const std::string ty = t.get_string("type", "function");
      if (ty != "function" && std::find(other.begin(), other.end(), ty) == other.end()) other.push_back(ty);
    }
    if (!other.empty()) {
      std::string msg = "tools: only function tools (no built-in tools); this request also has ";
      for (size_t i = 0; i < other.size(); ++i) msg += (i ? ", " : "") + other[i];
      if (std::find(other.begin(), other.end(), "web_search") != other.end() ||
          std::find(other.begin(), other.end(), "namespace") != other.end())
        msg += " (Codex CLI: web_search = \"disabled\" and, under [features], multi_agent = false in config.toml)";
      throw RequestError(422, msg);
    }
    json::Value ts = arr();
    for (const json::Value& t : tools->as_array()) {
      json::Value fn = obj();
      fn.set("name", t.get_string("name", ""));
      if (const json::Value* d = t.get("description")) fn.set("description", *d);
      if (const json::Value* p = t.get("parameters")) fn.set("parameters", *p);
      json::Value w = obj();
      w.set("type", "function");
      w.set("function", std::move(fn));
      ts.push_back(std::move(w));
    }
    if (offer_tools) out.set("tools", std::move(ts));
  }
  return out;
}

json::Value responses_usage(const json::Value& usage) {
  const int64_t in = usage_int(usage, "prompt_tokens"), outp = usage_int(usage, "completion_tokens");
  json::Value id = obj();
  id.set("cached_tokens", cached_tokens(usage));
  json::Value od = obj();
  od.set("reasoning_tokens", 0);  // not counted apart from the output
  json::Value u = obj();
  u.set("input_tokens", in);
  u.set("input_tokens_details", std::move(id));
  u.set("output_tokens", outp);
  u.set("output_tokens_details", std::move(od));
  u.set("total_tokens", in + outp);
  return u;
}

struct ResponseIds {
  std::string resp, msg, rs;
};

json::Value output_text_part(const std::string& text) {
  json::Value p = obj();
  p.set("type", "output_text");
  p.set("text", text);
  p.set("annotations", arr());
  return p;
}

json::Value message_item(const ResponseIds& ids, const std::string& text, bool done, bool cut = false) {
  json::Value content = arr();
  if (done) content.push_back(output_text_part(text));
  json::Value m = obj();
  m.set("type", "message");
  m.set("id", ids.msg);
  m.set("status", !done ? "in_progress" : cut ? "incomplete" : "completed");
  m.set("role", "assistant");
  m.set("content", std::move(content));
  return m;
}

json::Value reasoning_item(const ResponseIds& ids, const std::string& reasoning, bool done) {
  json::Value content = arr();
  if (done) {
    json::Value p = obj();
    p.set("type", "reasoning_text");
    p.set("text", reasoning);
    content.push_back(std::move(p));
  }
  json::Value r = obj();
  r.set("type", "reasoning");
  r.set("id", ids.rs);
  r.set("summary", arr());
  r.set("content", std::move(content));
  return r;
}

json::Value call_item(const json::Value& c, bool done) {
  const std::string id = c.get_string("id", "");
  json::Value f = obj();
  f.set("type", "function_call");
  f.set("id", "fc_" + (id.compare(0, 5, "call_") == 0 ? id.substr(5) : id));
  f.set("call_id", id);
  f.set("name", call_name(c));
  f.set("arguments", done ? arguments_string(c) : std::string());
  f.set("status", done ? "completed" : "in_progress");
  return f;
}

json::Value response_object(const Engine& engine, const ChatJob& job, const json::Value& b, const ResponseIds& ids,
                            const std::string& status, json::Value output, const json::Value& usage,
                            const std::string& finish) {
  auto field = [&](const char* k) {
    const json::Value* v = b.get(k);
    return v ? *v : json::Value();
  };
  json::Value r = obj();
  r.set("id", ids.resp);
  r.set("object", "response");
  r.set("created_at", job.created);
  r.set("status", status);
  r.set("error", json::Value());
  if (finish == "length") {
    json::Value d = obj();
    d.set("reason", "max_output_tokens");
    r.set("incomplete_details", std::move(d));
  } else {
    r.set("incomplete_details", json::Value());
  }
  r.set("instructions", field("instructions"));
  r.set("max_output_tokens", field("max_output_tokens"));
  r.set("model", engine.model_name());
  r.set("output", std::move(output));
  r.set("parallel_tool_calls", true);
  r.set("previous_response_id", field("previous_response_id"));
  json::Value reasoning = obj();
  const json::Value* rq = b.get("reasoning");
  reasoning.set("effort", rq && rq->get("effort") ? *rq->get("effort") : json::Value());
  reasoning.set("summary", json::Value());
  r.set("reasoning", std::move(reasoning));
  r.set("store", b.get_bool("store", true));
  r.set("temperature", field("temperature"));
  json::Value fmt = obj();
  fmt.set("type", "text");
  if (const json::Value* t = b.get("text"); t && t->is_object())  // the format asked for
    if (const json::Value* f = t->get("format"); f && f->is_object()) fmt = *f;
  json::Value text = obj();
  text.set("format", std::move(fmt));
  r.set("text", std::move(text));
  if (json::Value sr = structured_report(job); !sr.is_null()) r.set("structured_output", std::move(sr));
  r.set("tool_choice", b.get("tool_choice") ? field("tool_choice") : json::Value("auto"));
  r.set("tools", b.get("tools") ? field("tools") : arr());
  r.set("top_p", field("top_p"));
  r.set("truncation", "disabled");
  r.set("usage", usage.is_null() ? json::Value() : responses_usage(usage));
  r.set("user", field("user"));
  r.set("metadata", b.get("metadata") ? field("metadata") : obj());
  return r;
}

// The conversation a continuation starts from: the history and the reply
// (with its reasoning, which templates render for past turns).
json::Value with_reply(json::Value history, const std::string& content, const std::string& reasoning, const ChatEvent& last) {
  json::Value a = obj();
  a.set("role", "assistant");
  a.set("content", content);
  if (!reasoning.empty()) a.set("reasoning_content", reasoning);
  if (has_calls(last)) a.set("tool_calls", last.tool_calls);
  history.push_back(std::move(a));
  return history;
}

json::Value responses_error(int status, const std::string& msg) {
  json::Value err = obj();
  err.set("message", msg);
  err.set("type", status >= 500 ? "server_error" : "invalid_request_error");
  err.set("code", status == 404 ? json::Value("not_found") : json::Value());
  json::Value o = obj();
  o.set("error", std::move(err));
  return o;
}

void responses_run(Engine& engine, const json::Value& body, http::Response& res) {
  json::Value history;
  ChatJob job = prepare_chat(engine, responses_to_chat(body, history));
  const std::string sfx = id_suffix(job);
  const ResponseIds ids{"resp_" + sfx, "msg_" + sfx, "rs_" + sfx};
  const bool keep = body.get_bool("store", true);
  // The final object: items in the order a stream opens them (reasoning,
  // message, function calls); stored for continuations.
  auto finish = [&](const std::string& content, const std::string& reasoning, const ChatEvent& last) {
    json::Value output = arr();
    if (!reasoning.empty()) output.push_back(reasoning_item(ids, reasoning, true));
    const bool cut = last.finish_reason == "length";
    if (!content.empty() || !has_calls(last)) output.push_back(message_item(ids, content, true, cut));
    if (has_calls(last))
      for (const json::Value& c : last.tool_calls.as_array()) output.push_back(call_item(c, true));
    const std::string status = cut ? "incomplete" : "completed";
    json::Value r = response_object(engine, job, body, ids, status, std::move(output), last.usage, last.finish_reason);
    // (f) a reply cut short by a client that left is not stored
    if (keep && last.final) response_store().put(ids.resp, r, with_reply(history, content, reasoning, last));
    return r;
  };
  if (!job.stream) {
    std::string content, reasoning;
    ChatEvent last;
    run_chat(engine, job, [&](const ChatEvent& ev) {
      content += ev.content;
      reasoning += ev.reasoning;
      if (ev.final) last = ev;
      return !res.client_gone();
    });
    res.send_json(200, finish(content, reasoning, last));
    return;
  }

  // Streaming: each output item is added, filled and done, with its
  // output_index, as the API's own streams do.
  if (!res.start_stream(200, "text/event-stream")) return;
  int64_t seq = 0;
  auto send = [&](const std::string& type, json::Value d) {
    d.set("type", type);
    d.set("sequence_number", seq++);
    return event(res, type, d);
  };
  auto snapshot = [&]() {
    json::Value d = obj();
    d.set("response", response_object(engine, job, body, ids, "in_progress", arr(), json::Value(), ""));
    return d;
  };
  if (!send("response.created", snapshot()) || !send("response.in_progress", snapshot())) return;
  int next_index = 0, rs_index = -1, msg_index = -1;
  bool rs_open = false;
  std::string content, reasoning;
  auto text_event = [&](const std::string& type, const std::string& item_id, int index, const char* key,
                        const std::string& text) {
    json::Value d = obj();
    d.set("item_id", item_id);
    d.set("output_index", index);
    d.set("content_index", 0);
    d.set(key, text);
    if (type.find("output_text") != std::string::npos) d.set("logprobs", arr());
    return send(type, std::move(d));
  };
  auto item_event = [&](const char* type, int index, json::Value item) {
    json::Value d = obj();
    d.set("output_index", index);
    d.set("item", std::move(item));
    return send(type, std::move(d));
  };
  auto end_reasoning = [&]() {
    if (!rs_open) return true;
    rs_open = false;
    return text_event("response.reasoning_text.done", ids.rs, rs_index, "text", reasoning) &&
           item_event("response.output_item.done", rs_index, reasoning_item(ids, reasoning, true));
  };
  auto open_message = [&]() {
    if (msg_index >= 0) return true;
    if (!end_reasoning()) return false;
    msg_index = next_index++;
    if (!item_event("response.output_item.added", msg_index, message_item(ids, "", false))) return false;
    json::Value d = obj();
    d.set("item_id", ids.msg);
    d.set("output_index", msg_index);
    d.set("content_index", 0);
    d.set("part", output_text_part(""));
    return send("response.content_part.added", std::move(d));
  };
  run_chat(engine, job, [&](const ChatEvent& ev) {
    // Reasoning precedes the answer in Qwen templates; any after it is dropped.
    if (!ev.reasoning.empty() && msg_index < 0) {
      if (rs_index < 0) {
        rs_index = next_index++;
        rs_open = true;
        if (!item_event("response.output_item.added", rs_index, reasoning_item(ids, "", false))) return false;
      }
      reasoning += ev.reasoning;
      if (!text_event("response.reasoning_text.delta", ids.rs, rs_index, "delta", ev.reasoning)) return false;
    }
    if (!ev.content.empty()) {
      if (!open_message()) return false;
      content += ev.content;
      if (!text_event("response.output_text.delta", ids.msg, msg_index, "delta", ev.content)) return false;
    }
    if (!ev.final) return true;
    if (!end_reasoning()) return false;
    if (!has_calls(ev) && !open_message()) return false;
    if (msg_index >= 0) {
      if (!text_event("response.output_text.done", ids.msg, msg_index, "text", content)) return false;
      json::Value d = obj();
      d.set("item_id", ids.msg);
      d.set("output_index", msg_index);
      d.set("content_index", 0);
      d.set("part", output_text_part(content));
      if (!send("response.content_part.done", std::move(d))) return false;
      if (!item_event("response.output_item.done", msg_index, message_item(ids, content, true, ev.finish_reason == "length")))
        return false;
    }
    if (has_calls(ev))
      for (const json::Value& c : ev.tool_calls.as_array()) {
        const int index = next_index++;
        json::Value item = call_item(c, true);
        const std::string item_id = item.get_string("id", "");
        if (!item_event("response.output_item.added", index, call_item(c, false))) return false;
        json::Value d = obj();
        d.set("item_id", item_id);
        d.set("output_index", index);
        d.set("delta", item.get_string("arguments", "{}"));
        if (!send("response.function_call_arguments.delta", std::move(d))) return false;
        json::Value d2 = obj();
        d2.set("item_id", item_id);
        d2.set("output_index", index);
        d2.set("name", item.get_string("name", ""));
        d2.set("arguments", item.get_string("arguments", "{}"));
        if (!send("response.function_call_arguments.done", std::move(d2))) return false;
        if (!item_event("response.output_item.done", index, std::move(item))) return false;
      }
    json::Value r = finish(content, reasoning, ev);
    const std::string status = r.get_string("status", "completed");
    json::Value d = obj();
    d.set("response", std::move(r));
    return send("response." + status, std::move(d));
  });
  res.end();
}

// ---------------------------------------------------------------- Prometheus

// A sample value in the text exposition format.
std::string prom_value(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "+Inf" : "-Inf";
  char b[64];
  std::snprintf(b, sizeof b, "%.17g", v);
  return b;
}
std::string prom_label(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '\n') {
      out += "\\n";
      continue;
    }
    if (c == '\\' || c == '"') out += '\\';
    out += c;
  }
  return out;
}
std::string prom_name(const std::string& key) {
  std::string k;
  for (char c : key) k += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
  return k;
}

// The engine's /v1/stats as metrics: nested objects flattened with '_', the
// totals since start as counters (suffix _total), the rest as gauges.
std::string metrics_text(const Engine& engine) {
  static const char* const kCounters[] = {"passes", "rows", "gpu_ms", "generated_tokens", "decisions"};
  std::string out;
  auto sample = [&](const std::string& name, const char* type, double v) {
    out += "# TYPE " + name + " " + type + "\n" + name + " " + prom_value(v) + "\n";
  };
  std::function<void(const std::string&, const json::Value&)> walk = [&](const std::string& name, const json::Value& v) {
    if (v.is_number()) {
      sample(name, "gauge", v.as_double());
    } else if (v.is_bool()) {
      sample(name, "gauge", v.as_bool() ? 1.0 : 0.0);
    } else if (v.is_object()) {
      const json::Object& o = v.as_object();
      for (size_t i = 0; i < o.keys.size(); ++i) walk(name + "_" + prom_name(o.keys[i]), o.vals[i]);
    }
  };
  const json::Value stats = engine.stats();
  if (stats.is_object()) {
    const json::Object& o = stats.as_object();
    for (size_t i = 0; i < o.keys.size(); ++i) {
      const std::string name = std::string(kProduct) + "_" + prom_name(o.keys[i]);
      bool counter = false;
      for (const char* k : kCounters) counter |= o.keys[i] == k;
      if (counter && o.vals[i].is_number()) sample(name + "_total", "counter", o.vals[i].as_double());
      else walk(name, o.vals[i]);
    }
  }
  out += std::string("# TYPE ") + kProduct + "_info gauge\n" + kProduct + "_info{model=\"" + prom_label(engine.model_name()) +
         "\",version=\"" + kVersion + "\"} 1\n";
  return out;
}

// Clef decision models serve no chat, here as on /v1/chat/completions.
void refuse_clef(const Engine& engine) {
  if (engine.is_clef())
    throw RequestError(422, "this is a Clef decision model: it answers /v1/systemone and /v1/decisions, not chat");
}

}  // namespace

void install_compat_routes(http::Server& server, Engine& engine) {
  // ---- Anthropic Messages
  server.route("POST", "/v1/messages", [&engine](const http::Request& req, http::Response& res) {
    try {
      refuse_clef(engine);
      const json::Value body = parse_body(req);
      if (!body.get("max_tokens")) invalid("max_tokens: required");
      ChatJob job = prepare_chat(engine, anthropic_to_chat(body));
      if (job.stream) {
        anthropic_stream(engine, job, res, thinking_shown(body));
        return;
      }
      std::string content, reasoning;
      ChatEvent last;
      run_chat(engine, job, [&](const ChatEvent& ev) {
        content += ev.content;
        reasoning += ev.reasoning;
        if (ev.final) last = ev;
        return !res.client_gone();
      });
      res.send_json(200, anthropic_message(engine, job, content, reasoning, last, thinking_shown(body)));
    } catch (const RequestError& e) {
      if (res.started()) event(res, "error", anthropic_error(e.status, e.what()));
      else res.send_json(e.status, anthropic_error(e.status, e.what()));
    } catch (const std::exception& e) {
      if (res.started()) event(res, "error", anthropic_error(500, e.what()));
      else res.send_json(500, anthropic_error(500, e.what()));
    }
  }, http::Body::Json);
  server.route("POST", "/v1/messages/count_tokens", [&engine](const http::Request& req, http::Response& res) {
    try {
      refuse_clef(engine);
      json::Value body = parse_body(req);
      body.set("max_tokens", 1);  // counting only: no output budget against the context
      body.set("stream", false);
      ChatJob job = prepare_chat(engine, anthropic_to_chat(body));
      json::Value o = obj();
      o.set("input_tokens", int64_t(job.gen.prompt.size()));
      res.send_json(200, o);
    } catch (const RequestError& e) {
      res.send_json(e.status, anthropic_error(e.status, e.what()));
    } catch (const std::exception& e) {
      res.send_json(500, anthropic_error(500, e.what()));
    }
  }, http::Body::Json);

  // ---- OpenAI Responses
  server.route("POST", "/v1/responses", [&engine](const http::Request& req, http::Response& res) {
    auto fail = [&res](int status, const std::string& msg) {
      json::Value d = responses_error(status, msg);
      if (!res.started()) {
        res.send_json(status, d);
        return;
      }
      d.set("type", "error");
      event(res, "error", d);
    };
    try {
      refuse_clef(engine);
      responses_run(engine, parse_body(req), res);
    } catch (const RequestError& e) {
      fail(e.status, e.what());
    } catch (const std::exception& e) {
      fail(500, e.what());
    }
  }, http::Body::Json);
  server.route_prefix("GET", "/v1/responses/", [](const http::Request& req, http::Response& res) {
    const std::string id = req.path.substr(std::string("/v1/responses/").size());
    json::Value r;
    if (!response_store().response(id, r)) {
      res.send_json(404, responses_error(404, "no stored response '" + id + "'"));
      return;
    }
    res.send_json(200, r);
  });
  server.route_prefix("DELETE", "/v1/responses/", [](const http::Request& req, http::Response& res) {
    const std::string id = req.path.substr(std::string("/v1/responses/").size());
    if (!response_store().erase(id)) {
      res.send_json(404, responses_error(404, "no stored response '" + id + "'"));
      return;
    }
    json::Value o = obj();
    o.set("id", id);
    o.set("object", "response");
    o.set("deleted", true);
    res.send_json(200, o);
  });

  // ---- Prometheus
  server.route("GET", "/metrics", [&engine](const http::Request&, http::Response& res) {
    try {
      res.send(200, "text/plain; version=0.0.4; charset=utf-8", metrics_text(engine));
    } catch (const std::exception& e) {
      res.send(500, "text/plain; charset=utf-8", std::string("metrics: ") + e.what() + "\n");
    }
  });
}

}  // namespace espetero
