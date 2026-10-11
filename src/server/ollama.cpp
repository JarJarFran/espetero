// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Ollama request translation: /api/chat, /api/generate and the
// embedding bodies as the OpenAI routes' bodies, without the engine. The
// routes are in ollama_api.cpp.
#include "ollama.h"

#include <cmath>
#include <ctime>

#include "../engine/engine.h"

namespace espetero::ollama {

namespace {

[[noreturn]] void bad(const std::string& msg) { throw RequestError(400, msg); }
// Not supported: refused, never ignored (as on every API).
[[noreturn]] void unsupported(const std::string& msg) { throw RequestError(422, msg); }

json::Value obj() { return json::Value::object(); }
json::Value arr() { return json::Value::array(); }

bool absent(const json::Value* v) { return !v || v->is_null(); }
bool blank(const json::Value* v) { return absent(v) || (v->is_string() && v->as_string().empty()); }

std::string string_field(const json::Value& b, const char* key) {
  const json::Value* v = b.get(key);
  if (absent(v)) return "";
  if (!v->is_string()) bad(std::string(key) + ": a string");
  return v->as_string();
}

double number(const json::Value& v, const std::string& key) {
  if (!v.is_number()) bad("options." + key + ": a number");
  const double x = v.as_double();
  if (!std::isfinite(x)) bad("options." + key + ": a finite number");
  return x;
}

int64_t integer(const json::Value& v, const std::string& key) {
  if (v.is_int()) return v.as_int();
  const double x = number(v, key);  // 40.0 from clients that send every number as a float
  if (x != std::floor(x) || std::fabs(x) > 9e15) bad("options." + key + ": an integer");
  return int64_t(x);
}

// format (structured output): "json" or a JSON schema, as the chat
// route's response_format: "json" is {"type": "json_object"}, a schema
// {"type": "json_schema", "json_schema": {"name": "format", "schema": ...,
// "strict": false}} (Ollama's semantics: keywords the grammar cannot
// enforce are reported, not refused). The same mapping as
// ollama_response_format() in engine/structured.h, kept here so this
// translation stays free of the engine.
void apply_format(const json::Value& b, json::Value& out) {
  const json::Value* f = b.get("format");
  if (blank(f)) return;
  json::Value rf = json::Value::object();
  if (f->is_string()) {
    if (f->as_string() != "json") bad("format: \"json\" or a JSON schema");
    rf.set("type", "json_object");
  } else if (f->is_object()) {
    json::Value js = json::Value::object();
    js.set("name", "format");
    js.set("schema", *f);
    js.set("strict", false);
    rf.set("type", "json_schema");
    rf.set("json_schema", std::move(js));
  } else {
    bad("format: \"json\" or a JSON schema");
  }
  out.set("response_format", std::move(rf));
}

// keep_alive: how long Ollama keeps a model loaded after a request. This
// server keeps its one model loaded while it runs, so any value is accepted
// and has no effect (documented).
void check_keep_alive(const json::Value& b) {
  const json::Value* k = b.get("keep_alive");
  if (!absent(k) && !k->is_string() && !k->is_number()) bad("keep_alive: a duration such as \"5m\", or seconds");
}

// think: true / false, or a level ("low", "medium", "high") for templates
// that take a reasoning effort (as /v1/responses and /v1/messages map theirs:
// high is the template's "xhigh"). Omitted: the template's own default.
void apply_think(const json::Value& b, const ModelTraits& model, json::Value& out) {
  const json::Value* t = b.get("think");
  if (absent(t)) return;
  bool on = false;
  std::string level;
  if (t->is_bool()) {
    on = t->as_bool();
  } else if (t->is_string()) {
    level = t->as_string();
    if (level != "low" && level != "medium" && level != "high") bad("think: true, false, \"low\", \"medium\" or \"high\"");
    on = true;
  } else {
    bad("think: true, false, \"low\", \"medium\" or \"high\"");
  }
  if (on && !model.thinking) unsupported("think: this model's chat template has no thinking mode");
  out.set("enable_thinking", on);
  if (!level.empty()) {
    json::Value kw = obj();
    kw.set("reasoning_effort", level == "high" ? std::string("xhigh") : level);
    out.set("chat_template_kwargs", std::move(kw));
  }
}

// logprobs / top_logprobs (Ollama 0.12): the chat route's own fields.
void apply_logprobs(const json::Value& b, json::Value& out) {
  const json::Value* lp = b.get("logprobs");
  const json::Value* top = b.get("top_logprobs");
  if (!absent(lp)) {
    if (!lp->is_bool()) bad("logprobs: true or false");
    out.set("logprobs", *lp);
  }
  if (!absent(top) && !(top->is_int() && top->as_int() == 0)) {
    if (absent(lp) || !lp->as_bool()) bad("top_logprobs: needs \"logprobs\": true");
    out.set("top_logprobs", *top);
  }
}

// Image parts (before the text, where Ollama puts a message's images).
json::Value images_of(const json::Value& m) {
  json::Value parts = arr();
  const json::Value* imgs = m.get("images");
  if (absent(imgs)) return parts;
  if (!imgs->is_array()) bad("images: a list of base64-encoded images");
  for (const json::Value& i : imgs->as_array()) {
    if (!i.is_string() || i.as_string().empty()) bad("images: a list of base64-encoded images");
    json::Value u = obj();
    u.set("url", image_data_url(i.as_string()));
    json::Value p = obj();
    p.set("type", "image_url");
    p.set("image_url", std::move(u));
    parts.push_back(std::move(p));
  }
  return parts;
}

// A message's content: the text, or its images then the text.
json::Value content_of(json::Value images, const std::string& text) {
  if (images.size() == 0) return json::Value(text);
  if (!text.empty()) {
    json::Value t = obj();
    t.set("type", "text");
    t.set("text", text);
    images.push_back(std::move(t));
  }
  return images;
}

json::Value openai_tool_calls(const json::Value& calls) {
  if (!calls.is_array()) bad("tool_calls: a list of {\"function\": {\"name\", \"arguments\"}}");
  json::Value out = arr();
  for (const json::Value& c : calls.as_array()) {
    const json::Value* fn = c.is_object() ? c.get("function") : nullptr;
    if (!fn || !fn->is_object() || !fn->get("name") || !fn->get("name")->is_string())
      bad("tool_calls: a list of {\"function\": {\"name\", \"arguments\"}}");
    const json::Value* args = fn->get("arguments");
    json::Value f = obj();
    f.set("name", *fn->get("name"));
    f.set("arguments", absent(args) ? std::string("{}") : args->is_string() ? args->as_string() : json::dump(*args));
    json::Value o = obj();
    if (const json::Value* id = c.get("id"); id && id->is_string() && !id->as_string().empty()) o.set("id", *id);
    o.set("type", "function");
    o.set("function", std::move(f));
    out.push_back(std::move(o));
  }
  return out;
}

json::Value chat_messages(const json::Value& ms) {
  if (!ms.is_array()) bad("messages: a list of messages");
  json::Value out = arr();
  for (const json::Value& m : ms.as_array()) {
    if (!m.is_object()) bad("messages: objects with a role and a content");
    const std::string role = m.get_string("role", "");
    if (role != "system" && role != "user" && role != "assistant" && role != "tool")
      bad("messages: role system, user, assistant or tool");
    json::Value o = obj();
    o.set("role", role);
    o.set("content", content_of(images_of(m), string_field(m, "content")));
    if (const std::string thinking = string_field(m, "thinking"); !thinking.empty()) o.set("reasoning_content", thinking);
    if (const json::Value* tc = m.get("tool_calls"); !absent(tc) && tc->size()) o.set("tool_calls", openai_tool_calls(*tc));
    if (const std::string name = string_field(m, "tool_name"); !name.empty()) o.set("name", name);
    if (const std::string id = string_field(m, "tool_call_id"); !id.empty()) o.set("tool_call_id", id);
    out.push_back(std::move(o));
  }
  return out;
}

// The parts of a request both generation routes share.
void common(const json::Value& b, json::Value& out, Extras& extras) {
  if (!b.is_object()) bad("request body must be a JSON object");
  if (const json::Value* m = b.get("model"); !absent(m) && !m->is_string()) bad("model: a string");
  const json::Value* st = b.get("stream");
  if (!absent(st) && !st->is_bool()) bad("stream: true or false");
  extras.stream = absent(st) || st->as_bool();
  out.set("stream", extras.stream);
  apply_format(b, out);
  check_keep_alive(b);
  // truncate (default true: Ollama drops the oldest messages that do not fit
  // num_ctx) and shift (default true: it shifts the window when the reply
  // fills it): nothing is ever cut here, so either value is accepted; a
  // prompt that does not fit gets an error and a reply stops at the window.
  for (const char* k : {"truncate", "shift"})
    if (const json::Value* v = b.get(k); !absent(v) && !v->is_bool()) bad(std::string(k) + ": true or false");
}

}  // namespace

std::string image_data_url(const std::string& image) {
  if (image.rfind("data:", 0) == 0) return image;  // already a data URL
  // The media type from the first bytes (the decoder reads the format from
  // the bytes themselves; this only labels the URL).
  const char* type = "image/jpeg";
  if (image.rfind("iVBORw0KGgo", 0) == 0) type = "image/png";
  else if (image.rfind("R0lGOD", 0) == 0) type = "image/gif";
  else if (image.rfind("UklGR", 0) == 0) type = "image/webp";
  else if (image.rfind("Qk", 0) == 0) type = "image/bmp";
  else if (image.rfind("SUkq", 0) == 0 || image.rfind("TU0AK", 0) == 0) type = "image/tiff";
  return std::string("data:") + type + ";base64," + image;
}

void apply_options(const json::Value* options, const ModelTraits& model, json::Value& out, Extras& extras) {
  // num_predict: Ollama's default is -1 (no limit: until the end of the turn
  // or the context window). The chat route caps max_tokens at the room the
  // prompt leaves.
  out.set("max_tokens", int64_t(std::max(1, model.max_ctx)));
  if (absent(options)) return;
  if (!options->is_object()) bad("options: an object such as {\"temperature\": 0.7}");
  double repeat = 1, presence = 0, frequency = 0;
  int64_t window = -1;
  bool spare_newline = false;
  const json::Object& o = options->as_object();
  for (size_t i = 0; i < o.size(); ++i) {
    const std::string& k = o.keys[i];
    const json::Value& v = o.vals[i];
    if (v.is_null()) continue;  // clients that send every field, unset ones as null
    // How Ollama's runner loads and runs a model (memory mapping, GPU
    // layers, threads, batch size, context shifting): this server decides
    // those itself, and they do not change outputs. Accepted, no effect.
    if (k == "numa" || k == "num_batch" || k == "num_gpu" || k == "main_gpu" || k == "low_vram" || k == "f16_kv" ||
        k == "logits_all" || k == "vocab_only" || k == "use_mmap" || k == "use_mlock" || k == "num_thread" ||
        k == "num_keep")
      continue;
    if (k == "temperature" || k == "top_p" || k == "min_p") {
      out.set(k, number(v, k));
    } else if (k == "top_k") {
      const int64_t n = integer(v, k);
      out.set("top_k", n > 0 ? n : 0);  // <= 0: off, as in llama.cpp
    } else if (k == "seed") {
      const int64_t n = integer(v, k);
      if (n >= 0) out.set("seed", n);  // -1: a random seed (Ollama's default)
    } else if (k == "num_predict") {
      const int64_t n = integer(v, k);
      if (n == -1 || n == -2) continue;  // no limit / fill the context: the default above
      if (n < 1) unsupported("options.num_predict: a positive number of tokens, or -1 for no limit");
      out.set("max_tokens", std::min<int64_t>(n, 1 << 20));
    } else if (k == "stop") {
      out.set("stop", v);  // a string or a list (checked by the chat route)
    } else if (k == "repeat_penalty") {
      repeat = number(v, k);
      if (repeat < 1e-3 || repeat > 10) unsupported("options.repeat_penalty: from 0.001 to 10 (1 = off)");
    } else if (k == "presence_penalty" || k == "frequency_penalty") {
      (k == "presence_penalty" ? presence : frequency) = number(v, k);
    } else if (k == "repeat_last_n") {
      window = integer(v, k);
    } else if (k == "penalize_newline") {
      if (!v.is_bool()) bad("options.penalize_newline: true or false");
      spare_newline = !v.as_bool();
    } else if (k == "num_ctx") {
      extras.num_ctx = integer(v, k);
      if (extras.num_ctx < 1) bad("options.num_ctx: a positive number of tokens");
    } else if (k == "typical_p" || k == "tfs_z") {
      if (number(v, k) < 1)
        unsupported("options." + k + ": " + (k == "tfs_z" ? "tail-free" : "locally typical") +
                    " sampling is not supported (1 = off)");
    } else if (k == "mirostat") {
      if (integer(v, k) != 0) unsupported("options.mirostat: Mirostat sampling is not supported (0 = off)");
    } else if (k == "mirostat_tau" || k == "mirostat_eta") {
      (void)number(v, k);  // only used with mirostat on, which is refused
    } else {
      unsupported("options." + k + ": not supported");
    }
  }
  // Penalties: the chat route's, over the whole conversation (repeat) and the
  // reply (presence, frequency). repeat_last_n -1 (the whole context) is
  // that; 0 turns them off, as in llama.cpp; another window only matters,
  // and is refused, while a penalty is on.
  if (window == 0) repeat = 1, presence = 0, frequency = 0;
  const bool penalized = repeat != 1 || presence != 0 || frequency != 0;
  if (penalized && window > 0)
    unsupported("options.repeat_last_n: the penalties here cover the whole conversation (-1) or nothing (0); a window "
                "of " + std::to_string(window) + " tokens is not supported");
  if (penalized && spare_newline)
    unsupported("options.penalize_newline: false is not supported with a penalty (every token is penalized)");
  if (repeat != 1) out.set("repetition_penalty", repeat);
  if (presence != 0) out.set("presence_penalty", presence);
  if (frequency != 0) out.set("frequency_penalty", frequency);
}

json::Value chat_body(const json::Value& b, const ModelTraits& model, Extras& extras) {
  json::Value out = obj();
  common(b, out, extras);
  const json::Value* ms = b.get("messages");
  if (absent(ms) || (ms->is_array() && ms->size() == 0)) {  // Ollama's way to load a model
    extras.load_only = true;
    return out;
  }
  out.set("messages", chat_messages(*ms));
  if (const json::Value* tools = b.get("tools"); !absent(tools)) {
    if (!tools->is_array()) bad("tools: a list of {\"type\": \"function\", \"function\": {...}}");
    for (const json::Value& t : tools->as_array())
      if (!t.is_object() || t.get_string("type", "function") != "function" || !t.get("function"))
        unsupported("tools: only {\"type\": \"function\", \"function\": {...}} tools");
    if (tools->size()) out.set("tools", *tools);
  }
  apply_think(b, model, out);
  apply_logprobs(b, out);
  apply_options(b.get("options"), model, out, extras);
  return out;
}

json::Value generate_body(const json::Value& b, const ModelTraits& model, Extras& extras, bool& raw) {
  json::Value out = obj();
  common(b, out, extras);
  const json::Value* r = b.get("raw");
  if (!absent(r) && !r->is_bool()) bad("raw: true or false");
  raw = !absent(r) && r->as_bool();
  const std::string prompt = string_field(b, "prompt"), system = string_field(b, "system");
  json::Value images = images_of(b);
  if (prompt.empty() && images.size() == 0) {  // Ollama's way to load a model
    extras.load_only = true;
    return out;
  }
  if (!string_field(b, "suffix").empty()) unsupported("suffix: fill-in-the-middle is not supported");
  if (!string_field(b, "template").empty())
    unsupported("template: Go templates are not supported; the model's own chat template renders the prompt "
                "(\"raw\": true sends it as is)");
  if (const json::Value* c = b.get("context"); !absent(c) && !(c->is_array() && c->size() == 0))
    unsupported("context: not supported (deprecated by Ollama); send the conversation to /api/chat");
  if (raw) {
    if (images.size()) unsupported("images: not with raw (the chat template places them)");
    if (!system.empty()) unsupported("system: not with raw (the prompt is sent as is)");
    if (!absent(b.get("think"))) unsupported("think: not with raw (the prompt is sent as is)");
    out.set("prompt", prompt);
  } else {
    json::Value msgs = arr();
    if (!system.empty()) {
      json::Value s = obj();
      s.set("role", "system");
      s.set("content", system);
      msgs.push_back(std::move(s));
    }
    json::Value u = obj();
    u.set("role", "user");
    u.set("content", content_of(std::move(images), prompt));
    msgs.push_back(std::move(u));
    out.set("messages", std::move(msgs));
    apply_think(b, model, out);
  }
  apply_logprobs(b, out);
  apply_options(b.get("options"), model, out, extras);
  return out;
}

json::Value embed_body(const json::Value& b, bool legacy, bool& empty) {
  if (!b.is_object()) bad("request body must be a JSON object");
  if (const json::Value* m = b.get("model"); !absent(m) && !m->is_string()) bad("model: a string");
  check_keep_alive(b);
  json::Value out = obj();
  if (legacy) {
    const std::string prompt = string_field(b, "prompt");
    empty = prompt.empty();
    out.set("input", prompt);
  } else {
    const json::Value* in = b.get("input");
    if (absent(in)) {
      empty = true;
    } else if (in->is_string()) {
      empty = in->as_string().empty();
      out.set("input", *in);
    } else if (in->is_array()) {
      for (const json::Value& s : in->as_array())
        if (!s.is_string()) bad("input: a string or a list of strings");
      empty = in->size() == 0;
      out.set("input", *in);
    } else {
      bad("input: a string or a list of strings");
    }
    // truncate (default true): Ollama cuts inputs longer than num_ctx. Inputs
    // are never cut here (the server's own limits apply, with an error past
    // them): accepted either way.
    if (const json::Value* t = b.get("truncate"); !absent(t) && !t->is_bool()) bad("truncate: true or false");
    if (const json::Value* d = b.get("dimensions"); !absent(d)) out.set("dimensions", *d);  // checked by /v1/embeddings
  }
  // Generation options have no effect on embeddings (as in Ollama) and the
  // runner's none on outputs; num_ctx would cut inputs, which is not done.
  if (const json::Value* o = b.get("options"); !absent(o)) {
    if (!o->is_object()) bad("options: an object");
    const json::Object& opts = o->as_object();
    for (size_t i = 0; i < opts.size(); ++i) {
      if (opts.vals[i].is_null()) continue;
      const std::string& k = opts.keys[i];
      if (k == "num_ctx")
        unsupported("options.num_ctx: not supported for embeddings (inputs are never truncated; the server's own "
                    "limits apply)");
      static const char* known[] = {"numa", "num_batch", "num_gpu", "main_gpu", "low_vram", "f16_kv", "logits_all",
                                    "vocab_only", "use_mmap", "use_mlock", "num_thread", "num_keep", "seed",
                                    "num_predict", "top_k", "top_p", "min_p", "typical_p", "repeat_last_n",
                                    "temperature", "repeat_penalty", "presence_penalty", "frequency_penalty",
                                    "mirostat", "mirostat_tau", "mirostat_eta", "penalize_newline", "stop", "tfs_z"};
      bool ok = false;
      for (const char* x : known) ok |= k == x;
      if (!ok) unsupported("options." + k + ": not supported");
    }
  }
  return out;
}

json::Value tool_call(const json::Value& c, int index) {
  const json::Value* fn = c.get("function");
  json::Value args = obj();
  if (const json::Value* a = fn ? fn->get("arguments") : nullptr) {
    if (a->is_object()) {
      args = *a;
    } else if (a->is_string()) {
      try {
        json::Value v = json::parse(a->as_string());
        if (v.is_object()) args = std::move(v);
      } catch (...) {  // arguments that are not JSON stay empty: the call's text is in the reply as the model wrote it
      }
    }
  }
  json::Value f = obj();
  f.set("index", int64_t(index));
  f.set("name", fn ? fn->get_string("name", "") : std::string());
  f.set("arguments", std::move(args));
  json::Value o = obj();
  if (const std::string id = c.get_string("id", ""); !id.empty()) o.set("id", id);
  o.set("function", std::move(f));
  return o;
}

std::string done_reason(const std::string& finish_reason) {
  return finish_reason == "length" ? "length" : "stop";
}

std::string timestamp(std::chrono::system_clock::time_point t) {
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch()).count();
  const std::time_t secs = std::time_t(us / 1000000);
  std::tm tm{};
  ::gmtime_r(&secs, &tm);
  char buf[48];
  std::snprintf(buf, sizeof buf, "%04d-%02d-%02dT%02d:%02d:%02d.%06lldZ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec, (long long)(us % 1000000));
  return buf;
}

std::string tagged(const std::string& name) {
  const size_t slash = name.rfind('/');
  return name.find(':', slash == std::string::npos ? 0 : slash) == std::string::npos ? name + ":latest" : name;
}

}  // namespace espetero::ollama
