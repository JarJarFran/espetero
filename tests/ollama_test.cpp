// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Ollama request translation without a model (make test): /api/chat,
// /api/generate and the embedding bodies become the OpenAI routes' bodies;
// what cannot apply is refused (422), never ignored.
//
//   build/ollama_test
#include <cstdio>
#include <string>

#include "../src/engine/engine.h"
#include "../src/server/ollama.h"

using namespace espetero;

namespace {

int checks = 0, failures = 0;

void check(bool ok, const std::string& what) {
  ++checks;
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
  }
}

json::Value J(const char* s) { return json::parse(s); }
std::string D(const json::Value& v) { return json::dump(v); }

// The status of a RequestError the call throws (0 if none) and its message.
template <class F>
int status_of(F&& f, std::string* msg = nullptr) {
  try {
    f();
  } catch (const RequestError& e) {
    if (msg) *msg = e.what();
    return e.status;
  }
  return 0;
}

ollama::ModelTraits traits(bool thinking = true) {
  ollama::ModelTraits t;
  t.max_ctx = 32768;
  t.thinking = thinking;
  return t;
}

json::Value chat(const char* body, ollama::Extras* x = nullptr, bool thinking = true) {
  ollama::Extras local;
  return ollama::chat_body(J(body), traits(thinking), x ? *x : local);
}

void test_chat() {
  ollama::Extras x;
  json::Value b = chat(R"({"model": "anything:7b", "messages": [{"role": "user", "content": "hi"}]})", &x);
  check(x.stream && !x.load_only, "stream by default");
  check(D(b) == R"({"stream":true,"messages":[{"role":"user","content":"hi"}],"max_tokens":32768})", "plain: " + D(b));

  // Images go before the text, as data URLs whose type follows their bytes.
  b = chat(R"({"messages": [{"role": "user", "content": "what is it?", "images": ["iVBORw0KGgoAAA", "/9j/4AAQ"]}]})");
  const json::Value& parts = *b.get("messages")->as_array()[0].get("content");
  check(parts.size() == 3 && parts[0].get("image_url")->get_string("url", "") == "data:image/png;base64,iVBORw0KGgoAAA" &&
            parts[1].get("image_url")->get_string("url", "") == "data:image/jpeg;base64,/9j/4AAQ" &&
            parts[2].get_string("text", "") == "what is it?",
        "images then text: " + D(parts));
  check(ollama::image_data_url("data:image/webp;base64,UklGR") == "data:image/webp;base64,UklGR", "data URL kept");
  check(ollama::image_data_url("UklGRiQAAABXRUJQ") == "data:image/webp;base64,UklGRiQAAABXRUJQ", "webp sniffed");

  // History: thinking, tool calls with object arguments, tool results.
  b = chat(R"({"messages": [
      {"role": "user", "content": "weather?"},
      {"role": "assistant", "content": "", "thinking": "need the tool",
       "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Málaga"}}}]},
      {"role": "tool", "content": "sunny", "tool_name": "get_weather"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]})");
  const json::Value& ms = *b.get("messages");
  check(ms[1].get_string("reasoning_content", "") == "need the tool", "thinking -> reasoning_content");
  check(D(*ms[1].get("tool_calls")) ==
            R"([{"type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Málaga\"}"}}])",
        "tool call: " + D(*ms[1].get("tool_calls")));
  check(ms[2].get_string("name", "") == "get_weather" && ms[2].get_string("role", "") == "tool", "tool_name -> name");
  check(b.get("tools") && b.get("tools")->size() == 1, "tools passed");
  check(status_of([] { chat(R"({"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "web_search"}]})"); }) == 422,
        "non-function tools: 422");

  // Loading: no messages.
  x = {};
  chat(R"({"model": "m"})", &x);
  check(x.load_only, "no messages: load");
  x = {};
  chat(R"({"model": "m", "messages": [], "keep_alive": 0})", &x);
  check(x.load_only, "empty messages: load");
  x = {};
  b = chat(R"({"messages": [{"role": "user", "content": "x"}], "stream": false, "keep_alive": "10m"})", &x);
  check(!x.stream && b.get("stream") && !b.get("stream")->as_bool(), "stream false");

  // think
  b = chat(R"({"messages": [{"role": "user", "content": "x"}], "think": false})");
  check(b.get("enable_thinking") && !b.get("enable_thinking")->as_bool(), "think false");
  b = chat(R"({"messages": [{"role": "user", "content": "x"}], "think": "high"})");
  check(b.get("enable_thinking")->as_bool() && b.get("chat_template_kwargs")->get_string("reasoning_effort", "") == "xhigh",
        "think high -> xhigh: " + D(b));
  b = chat(R"({"messages": [{"role": "user", "content": "x"}], "think": "low"})");
  check(b.get("chat_template_kwargs")->get_string("reasoning_effort", "") == "low", "think low");
  check(!chat(R"({"messages": [{"role": "user", "content": "x"}]})").get("enable_thinking"), "think omitted: template default");
  check(status_of([] { chat(R"({"messages": [{"role": "user", "content": "x"}], "think": true})", nullptr, false); }) == 422,
        "think on a model without thinking: 422");
  check(status_of([] { chat(R"({"messages": [{"role": "user", "content": "x"}], "think": false})", nullptr, false); }) == 0,
        "think false on a model without thinking: fine");
  check(status_of([] { chat(R"({"messages": [{"role": "user", "content": "x"}], "think": "max"})"); }) == 400, "think max: 400");

  // format: "json" -> json_object, a schema -> a non-strict json_schema; empty values are no format.
  b = chat(R"({"messages": [{"role": "user", "content": "x"}], "format": "json"})");
  check(b.get("response_format") && b.get("response_format")->get_string("type", "") == "json_object",
        "format json -> json_object: " + D(b));
  b = chat(R"({"messages": [{"role": "user", "content": "x"}], "format": {"type": "object", "properties": {"a": {"type": "integer"}}}})");
  {
    const json::Value* rf = b.get("response_format");
    const json::Value* js = rf ? rf->get("json_schema") : nullptr;
    check(rf && rf->get_string("type", "") == "json_schema" && js && js->get_string("name", "") == "format" &&
              js->get("strict") && !js->get("strict")->as_bool() && js->get("schema") &&
              js->get("schema")->get_string("type", "") == "object",
          "format schema -> non-strict json_schema: " + D(b));
  }
  check(!chat(R"({"messages": [{"role": "user", "content": "x"}], "format": ""})").get("response_format"), "format \"\": none");
  check(!chat(R"({"messages": [{"role": "user", "content": "x"}], "format": null})").get("response_format"), "format null: none");
  check(status_of([] { chat(R"({"messages": [{"role": "user", "content": "x"}], "format": "yaml"})"); }) == 400, "format yaml: 400");
  check(status_of([] { chat(R"({"messages": [{"role": "user", "content": "x"}], "format": 3})"); }) == 400, "format 3: 400");

  // Malformed
  check(status_of([] { chat(R"({"messages": [{"role": "robot", "content": "x"}]})"); }) == 400, "bad role: 400");
  check(status_of([] { chat(R"({"messages": [{"role": "user", "content": ["x"]}]})"); }) == 400, "list content: 400");
  check(status_of([] { chat(R"({"messages": [{"role": "user", "content": "x", "images": "abc"}]})"); }) == 400, "images: 400");
  check(status_of([] { chat(R"({"messages": [{"role": "user", "content": "x"}], "stream": "yes"})"); }) == 400, "stream: 400");

  // logprobs
  b = chat(R"({"messages": [{"role": "user", "content": "x"}], "logprobs": true, "top_logprobs": 3})");
  check(b.get("logprobs")->as_bool() && b.get("top_logprobs")->as_int() == 3, "logprobs");
  check(status_of([] { chat(R"({"messages": [{"role": "user", "content": "x"}], "top_logprobs": 3})"); }) == 400,
        "top_logprobs without logprobs: 400");
}

json::Value opts(const char* options, ollama::Extras* x = nullptr) {
  json::Value o = J(options);
  json::Value out = json::Value::object();
  ollama::Extras local;
  ollama::apply_options(&o, traits(), out, x ? *x : local);
  return out;
}

void test_options() {
  json::Value o = opts(R"({"temperature": 0.2, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "seed": 7, "num_predict": 64,
                           "stop": ["\n\n"], "repeat_penalty": 1.1, "repeat_last_n": -1, "presence_penalty": 0.5,
                           "frequency_penalty": 0.25})");
  check(D(o) == R"({"max_tokens":64,"temperature":0.2,"top_p":0.9,"top_k":40,"min_p":0.05,"seed":7,"stop":["\n\n"],)"
                R"("repetition_penalty":1.1,"presence_penalty":0.5,"frequency_penalty":0.25})",
        "sampling options: " + D(o));
  o = opts(R"({"num_predict": -1, "seed": -1, "top_k": -1, "temperature": null})");
  check(D(o) == R"({"max_tokens":32768,"top_k":0})", "no limit, random seed, top_k off, null skipped: " + D(o));
  check(opts(R"({"num_predict": -2})").get_int("max_tokens", 0) == 32768, "num_predict -2: fill the context");
  check(opts(R"({"top_k": 40.0})").get_int("top_k", 0) == 40, "integers sent as floats");
  check(status_of([] { opts(R"({"num_predict": 0})"); }) == 422, "num_predict 0: 422");
  check(status_of([] { opts(R"({"top_k": 4.5})"); }) == 400, "top_k 4.5: 400");
  check(status_of([] { opts(R"({"temperature": "hot"})"); }) == 400, "temperature string: 400");

  // Penalties: the whole conversation (-1) or none (0); another window only while one is on.
  check(status_of([] { opts(R"({"repeat_penalty": 1.1, "repeat_last_n": 64})"); }) == 422, "window with a penalty: 422");
  check(status_of([] { opts(R"({"presence_penalty": 0.5, "repeat_last_n": 64})"); }) == 422, "window with presence: 422");
  check(status_of([] { opts(R"({"repeat_last_n": 64})"); }) == 0, "window without a penalty: no effect");
  check(status_of([] { opts(R"({"repeat_penalty": 1.0, "repeat_last_n": 64})"); }) == 0, "window with penalty 1: no effect");
  o = opts(R"({"repeat_penalty": 1.3, "presence_penalty": 1, "repeat_last_n": 0})");
  check(D(o) == R"({"max_tokens":32768})", "repeat_last_n 0: no penalties: " + D(o));
  check(status_of([] { opts(R"({"repeat_penalty": 1.1, "penalize_newline": false})"); }) == 422, "newline spared: 422");
  check(status_of([] { opts(R"({"penalize_newline": false})"); }) == 0, "newline spared, no penalty: no effect");
  check(status_of([] { opts(R"({"repeat_penalty": 0})"); }) == 422, "repeat_penalty 0: 422");

  // Samplers this engine does not have: their off values only.
  for (const char* off : {R"({"typical_p": 1})", R"({"tfs_z": 1.0})", R"({"mirostat": 0, "mirostat_tau": 5, "mirostat_eta": 0.1})"})
    check(status_of([&] { opts(off); }) == 0, std::string("off: ") + off);
  for (const char* on : {R"({"typical_p": 0.7})", R"({"tfs_z": 0.9})", R"({"mirostat": 2})"})
    check(status_of([&] { opts(on); }) == 422, std::string("on: ") + on);

  // Runner options: accepted, no effect; anything else: 422.
  o = opts(R"({"num_gpu": 99, "num_thread": 8, "num_batch": 512, "use_mmap": true, "use_mlock": false, "main_gpu": 0,
               "low_vram": false, "numa": false, "num_keep": 4, "f16_kv": true, "vocab_only": false, "logits_all": false})");
  check(D(o) == R"({"max_tokens":32768})", "runner options: " + D(o));
  std::string msg;
  check(status_of([] { opts(R"({"rope_frequency_base": 10000})"); }, &msg) == 422 &&
            msg == "options.rope_frequency_base: not supported",
        "unknown option: " + msg);
  check(status_of([] { opts(R"([1])"); }) == 400, "options not an object: 400");

  ollama::Extras x;
  opts(R"({"num_ctx": 4096})", &x);
  check(x.num_ctx == 4096, "num_ctx kept for after the prompt");
  check(status_of([] { opts(R"({"num_ctx": 0})"); }) == 400, "num_ctx 0: 400");
}

void test_generate() {
  ollama::Extras x;
  bool raw = true;
  json::Value b = ollama::generate_body(J(R"({"model": "m", "prompt": "Why is the sky blue?", "system": "Be brief.",
                                              "stream": false, "options": {"temperature": 0}})"),
                                        traits(), x, raw);
  check(!raw && !x.stream, "generate: chat, not streamed");
  check(D(*b.get("messages")) ==
            R"([{"role":"system","content":"Be brief."},{"role":"user","content":"Why is the sky blue?"}])",
        "generate messages: " + D(b));
  check(b.get_double("temperature", -1) == 0, "generate options");
  b = ollama::generate_body(J(R"({"prompt": "describe", "images": ["/9j/4AAQ"], "think": false})"), traits(), x, raw);
  check(b.get("messages")->as_array()[0].get("content")->size() == 2 && !b.get("enable_thinking")->as_bool(),
        "generate with an image: " + D(b));
  b = ollama::generate_body(J(R"({"prompt": "<|im_start|>user\nhi<|im_end|>\n", "raw": true})"), traits(), x, raw);
  check(raw && b.get_string("prompt", "") == "<|im_start|>user\nhi<|im_end|>\n" && !b.get("messages"), "raw: the prompt as is");
  x = {};
  ollama::generate_body(J(R"({"model": "m"})"), traits(), x, raw);
  check(x.load_only, "generate without a prompt: load");
  for (const char* r : {R"({"prompt": "x", "suffix": "y"})", R"({"prompt": "x", "template": "{{ .Prompt }}"})",
                        R"({"prompt": "x", "context": [1, 2, 3]})", R"({"prompt": "x", "raw": true, "images": ["/9j/"]})",
                        R"({"prompt": "x", "raw": true, "system": "s"})", R"({"prompt": "x", "raw": true, "think": true})"})
    check(status_of([&] {
            ollama::Extras e;
            bool rw;
            ollama::generate_body(J(r), traits(), e, rw);
          }) == 422,
          std::string("refused: ") + r);
  check(status_of([&] {
          ollama::Extras e;
          bool rw;
          ollama::generate_body(J(R"({"prompt": "x", "context": [], "suffix": ""})"), traits(), e, rw);
        }) == 0,
        "empty context and suffix: fine");
}

void test_embed() {
  bool empty = false;
  json::Value b = ollama::embed_body(J(R"({"model": "m", "input": ["a", "b"], "truncate": true, "dimensions": 1024})"), false, empty);
  check(!empty && D(b) == R"({"input":["a","b"],"dimensions":1024})", "embed: " + D(b));
  b = ollama::embed_body(J(R"({"model": "m", "prompt": "hello", "options": {"temperature": 0.5, "num_gpu": 1}})"), true, empty);
  check(!empty && D(b) == R"({"input":"hello"})", "legacy embeddings: " + D(b));
  ollama::embed_body(J(R"({"model": "m"})"), false, empty);
  check(empty, "no input: empty");
  ollama::embed_body(J(R"({"model": "m", "input": []})"), false, empty);
  check(empty, "empty list: empty");
  check(status_of([&] { ollama::embed_body(J(R"({"input": "a", "options": {"num_ctx": 512}})"), false, empty); }) == 422,
        "embed num_ctx: 422");
  check(status_of([&] { ollama::embed_body(J(R"({"input": "a", "options": {"foo": 1}})"), false, empty); }) == 422,
        "embed unknown option: 422");
  check(status_of([&] { ollama::embed_body(J(R"({"input": [1, 2]})"), false, empty); }) == 400, "embed tokens: 400");
}

void test_shapes() {
  json::Value call = J(R"({"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}})");
  check(D(ollama::tool_call(call, 2)) == R"({"id":"call_1","function":{"index":2,"name":"f","arguments":{"a":1}}})",
        "tool call out: " + D(ollama::tool_call(call, 2)));
  call = J(R"({"type": "function", "function": {"name": "f", "arguments": "not json"}})");
  check(D(ollama::tool_call(call, 0)) == R"({"function":{"index":0,"name":"f","arguments":{}}})", "unparsable arguments");
  check(ollama::done_reason("length") == "length" && ollama::done_reason("tool_calls") == "stop" &&
            ollama::done_reason("stop") == "stop",
        "done reasons");
  const auto t = std::chrono::system_clock::time_point(std::chrono::microseconds(1760000000123456LL));
  check(ollama::timestamp(t) == "2025-10-09T08:53:20.123456Z", "timestamp: " + ollama::timestamp(t));
  check(ollama::tagged("mlx-community/Qwen3.5-0.8B-4bit") == "mlx-community/Qwen3.5-0.8B-4bit:latest" &&
            ollama::tagged("unsloth/Qwen3.8-27B-GGUF:UD-Q6_K") == "unsloth/Qwen3.8-27B-GGUF:UD-Q6_K" &&
            ollama::tagged("qwen") == "qwen:latest" && ollama::tagged("host:8080/org/m") == "host:8080/org/m:latest",
        "tags");
}

}  // namespace

int main() {
  test_chat();
  test_options();
  test_generate();
  test_embed();
  test_shapes();
  std::printf("ollama_test: %d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
