// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "structured.h"

#include <algorithm>
#include <list>
#include <mutex>

#include "../tok/tokenizer.h"
#include "chat.h"
#include "engine.h"
#include "json_schema.h"

namespace espetero {

namespace {

[[noreturn]] void invalid(const std::string& msg) { throw RequestError(422, msg); }

// Compiled grammars by (vocabulary, end and escape tokens, strictness,
// schema): requests with the same schema share the grammar and its masks.
struct GrammarCache {
  std::mutex mu;
  struct Entry {
    std::shared_ptr<const grammar::Compiled> compiled;
    std::list<std::string>::iterator lru;
  };
  std::unordered_map<std::string, Entry> map;
  std::list<std::string> lru;
  static constexpr size_t kEntries = 32;
};
GrammarCache& grammar_cache() {
  static GrammarCache c;
  return c;
}

}  // namespace

ResponseFormat parse_response_format(const json::Value* rf) {
  ResponseFormat f;
  if (!rf || rf->is_null()) return f;
  if (!rf->is_object()) invalid("response_format: an object such as {\"type\": \"json_object\"}");
  const std::string type = rf->get_string("type", "");
  if (type == "text") return f;
  if (type == "json_object") {
    f.type = ResponseFormat::JsonObject;
    return f;
  }
  if (type != "json_schema") invalid("response_format: type \"text\", \"json_object\" or \"json_schema\"");
  const json::Value* js = rf->get("json_schema");
  if (!js || !js->is_object()) invalid("response_format: json_schema needs {\"json_schema\": {\"name\": ..., \"schema\": {...}}}");
  const json::Value* schema = js->get("schema");
  if (!schema || !(schema->is_object() || schema->is_bool()))
    invalid("response_format.json_schema.schema: a JSON Schema object is required");
  f.type = ResponseFormat::JsonSchema;
  f.schema = *schema;
  f.name = js->get_string("name", "");
  if (const json::Value* s = js->get("strict"); s && !s->is_null()) {
    if (!s->is_bool()) invalid("response_format.json_schema.strict: true or false");
    f.strict = s->as_bool();
  }
  return f;
}

json::Value responses_response_format(const json::Value* format) {
  if (!format || format->is_null()) return json::Value();
  if (!format->is_object()) invalid("text.format: an object such as {\"type\": \"json_schema\", ...}");
  const std::string type = format->get_string("type", "text");
  json::Value rf = json::Value::object();
  if (type == "text") return json::Value();
  if (type == "json_object") {
    rf.set("type", "json_object");
    return rf;
  }
  if (type != "json_schema") invalid("text.format: type \"text\", \"json_object\" or \"json_schema\"");
  json::Value js = json::Value::object();
  for (const char* k : {"name", "schema", "strict", "description"})
    if (const json::Value* v = format->get(k)) js.set(k, *v);
  rf.set("type", "json_schema");
  rf.set("json_schema", std::move(js));
  return rf;
}

json::Value ollama_response_format(const json::Value* format) {
  if (!format || format->is_null() || (format->is_string() && format->as_string().empty())) return json::Value();
  json::Value rf = json::Value::object();
  if (format->is_string()) {
    if (format->as_string() != "json") invalid("format: \"json\" or a JSON schema object");
    rf.set("type", "json_object");
    return rf;
  }
  if (!format->is_object()) invalid("format: \"json\" or a JSON schema object");
  json::Value js = json::Value::object();
  js.set("name", "format");
  js.set("schema", *format);
  js.set("strict", false);
  rf.set("type", "json_schema");
  rf.set("json_schema", std::move(js));
  return rf;
}

std::shared_ptr<const grammar::Constraint> make_constraint(const Engine& engine, const ResponseFormat& f, int32_t think_end,
                                                           bool tools, std::vector<std::string>* unenforced) {
  if (f.type == ResponseFormat::Text) return nullptr;
  const Tokenizer& tok = engine.tokenizer();
  std::shared_ptr<const grammar::Vocab> vocab = engine.grammar_vocab();
  std::vector<int32_t> ends = engine.end_ids, escapes;
  if (tools) {
    const std::string& open = chat_markers(engine).tool_open;  // <tool_call>, Gemma 4: <|tool_call>
    const int32_t tc = tok.token_id(open);
    if (tc < 0) invalid("response_format with tools: this model's tool calls do not start with a " + open + " token");
    escapes.push_back(tc);
  }
  std::string key = std::to_string(reinterpret_cast<uintptr_t>(vocab.get())) + "|";
  for (int32_t t : ends) key += std::to_string(t) + ",";
  key += "|";
  for (int32_t t : escapes) key += std::to_string(t) + ",";
  key += f.type == ResponseFormat::JsonObject ? std::string("|object") : "|schema|" + std::string(f.strict ? "s|" : "n|") + json::dump(f.schema);
  GrammarCache& cache = grammar_cache();
  std::shared_ptr<const grammar::Compiled> compiled;
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    if (auto it = cache.map.find(key); it != cache.map.end()) {
      cache.lru.splice(cache.lru.begin(), cache.lru, it->second.lru);
      compiled = it->second.compiled;
    }
  }
  if (!compiled) {
    std::shared_ptr<const grammar::Grammar> g;
    std::vector<std::string> notes;
    try {
      g = f.type == ResponseFormat::JsonObject ? grammar::any_object_grammar()
                                               : grammar::compile_schema(f.schema, grammar::SchemaOptions{f.strict}, &notes);
    } catch (const grammar::SchemaError& e) {
      invalid(std::string("response_format: json_schema: ") + e.what());
    }
    compiled = std::make_shared<grammar::Compiled>(std::move(g), vocab, ends, escapes, std::move(notes));
    std::lock_guard<std::mutex> lk(cache.mu);
    if (!cache.map.count(key)) {
      cache.lru.push_front(key);
      cache.map.emplace(key, GrammarCache::Entry{compiled, cache.lru.begin()});
      while (cache.map.size() > GrammarCache::kEntries) {
        cache.map.erase(cache.lru.back());
        cache.lru.pop_back();
      }
    }
  }
  if (unenforced) *unenforced = compiled->unenforced();
  auto c = std::make_shared<grammar::Constraint>();
  c->compiled = std::move(compiled);
  c->think_end = think_end;
  return c;
}

void apply_response_format(const Engine& engine, const json::Value& body, ChatJob& job, std::string& prompt, bool chat) {
  const ResponseFormat f = parse_response_format(body.get("response_format"));
  if (f.type == ResponseFormat::Text) return;
  GenRequest& g = job.gen;
  if (!g.forced.empty()) invalid("response_format: not with score (the continuation is given)");
  if (g.ignore_eos) invalid("response_format: not with ignore_eos (the reply ends when the JSON value does)");
  if (!job.prefill.empty() || !job.force_after_think.empty())
    invalid("response_format: not with a forced tool call (tool_choice required or a named function)");
  int32_t think_end = -1;
  if (chat) {
    // The template opened the reasoning block: the model thinks, then the
    // grammar applies after its </think>. Without a single token to close it,
    // no reasoning (as a forced call does).
    const ChatMarkers& mk = chat_markers(engine);
    auto ends = [&](const std::string& t) {
      return prompt.size() >= t.size() && prompt.compare(prompt.size() - t.size(), t.size(), t) == 0;
    };
    if (mk.gemma) {
      // Gemma 4: with thinking on its template ends the prompt with the
      // model's turn header and the model opens the thought channel itself;
      // here the prompt opens it, so the grammar applies once <channel|> closes it.
      if (ends(mk.turn + mk.assistant)) prompt += mk.think_prompt;
      if (ends(mk.think_prompt)) {
        think_end = engine.tokenizer().token_id(mk.think_close);
        if (think_end < 0) prompt += mk.think_close;
      }
    } else {
      static const std::string think = "<think>\n";
      if (ends(think)) {
        think_end = engine.tokenizer().token_id("</think>");
        if (think_end < 0) prompt += "</think>\n\n";
      }
    }
  }
  g.constraint = make_constraint(engine, f, think_end, chat && !job.tools.is_null(), &job.unenforced);
}

json::Value structured_report(const ChatJob& job) {
  if (job.unenforced.empty()) return json::Value();
  json::Value list = json::Value::array();
  for (const std::string& s : job.unenforced) list.push_back(s);
  json::Value o = json::Value::object();
  o.set("unenforced", std::move(list));
  return o;
}

}  // namespace espetero
