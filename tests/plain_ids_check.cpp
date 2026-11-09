// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Prompt token ids of ordinary conversations with the request's text read
// as plain text (apply_chat_template's data spans) against the old reading
// (the whole prompt parsed for added tokens): they must be identical, since
// such conversations hold no added token in their own text. Run on model
// directories (tokenizer.json, tokenizer_config.json, the chat template):
//
//   make build/plain_ids_check && build/plain_ids_check DIR...
//
// Prints one line per conversation and template variant; exits 1 on any
// difference.
#include <cstdio>
#include <string>
#include <vector>

#include "../src/tmpl/tmpl.h"
#include "../src/tok/tokenizer.h"
#include "../src/util/json.h"

using namespace espetero;

namespace {

// Conversations as prepare_chat hands them to the template: images are
// {"type": "image"} parts, tool call arguments are objects.
const char* kConversations[] = {
    R"([{"role": "user", "content": "Hello! What's 2+2?"}])",
    R"([{"role": "system", "content": "You are a helpful assistant.\n\nAnswer briefly."},
        {"role": "user", "content": "  Translate «bonjour» to English, please.  "}])",
    R"([{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello.", "reasoning_content": "Greet back."},
        {"role": "user", "content": "Write a haiku about\ttabs and\r\nnewlines.\n\n\n"}])",
    R"([{"role": "user", "content": [{"type": "text", "text": "What is in this picture?"}, {"type": "image"},
                                     {"type": "text", "text": "\nAnd this one:"}, {"type": "image"}]}])",
    R"([{"role": "user", "content": "Weather in Málaga and 東京?"},
        {"role": "assistant", "content": "", "tool_calls": [
          {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Málaga", "days": 3}}},
          {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "東京", "units": ["c", "f"], "opts": {"x": true}}}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "{\"temp\": 24, \"sky\": \"clear\"}"},
        {"role": "tool", "tool_call_id": "call_2", "name": "get_weather", "content": "  rain, 12°C \n"},
        {"role": "assistant", "content": "Sunny in Málaga, rainy in Tokyo."},
        {"role": "user", "content": "Thanks!"}])",
    R"([{"role": "user", "content": "def f(x):\n    return x  # emoji 🙂 and code\n\n```py\nprint(f(1))\n```"}])",
};

const char* kTools = R"([
  {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
   "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
     "days": {"type": "integer"}, "units": {"type": "array", "items": {"type": "string", "enum": ["c", "f"]}}},
     "required": ["city"]}}},
  {"type": "function", "function": {"name": "search", "description": "Web search.",
   "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}])";

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s MODEL_DIR...\n", argv[0]);
    return 2;
  }
  int diffs = 0, total = 0;
  for (int a = 1; a < argc; ++a) {
    const std::string dir = argv[a];
    std::unique_ptr<Tokenizer> tok;
    std::shared_ptr<const tmpl::Template> t;
    try {
      tok = Tokenizer::from_hf_dir(dir);
      t = tmpl::Template::parse(tok->chat_template());
    } catch (const std::exception& e) {
      std::printf("%s: %s\n", dir.c_str(), e.what());
      ++diffs;
      continue;
    }
    // The special tokens a template sees (bos_token, eos_token), as chat.cpp's
    // template_special_tokens passes them for Gemma 4.
    json::Value special = json::Value::object();
    if (tok->bos_id() >= 0) special.set("bos_token", std::string(tok->token_text(tok->bos_id())));
    if (tok->eos_id() >= 0) special.set("eos_token", std::string(tok->token_text(tok->eos_id())));
    const json::Value tools = json::parse(kTools);
    for (size_t c = 0; c < sizeof(kConversations) / sizeof(kConversations[0]); ++c) {
      for (int variant = 0; variant < 4; ++variant) {
        const bool with_tools = variant & 1;
        json::Value kwargs = json::Value::object();
        kwargs.set("enable_thinking", (variant & 2) == 0);
        std::string text;
        Tokenizer::Ranges plain;
        try {
          text = tmpl::apply_chat_template(*t, json::parse(kConversations[c]), with_tools ? tools : json::Value(), true,
                                           kwargs, special, tmpl::RenderOptions(), &plain);
        } catch (const std::exception& e) {
          std::printf("%s conv %zu variant %d: template error: %s\n", dir.c_str(), c, variant, e.what());
          continue;
        }
        const std::vector<int32_t> legacy = tok->encode(text), now = tok->encode(text, true, plain);
        size_t covered = 0;
        for (const auto& [b, e] : plain) covered += e - b;
        ++total;
        const bool same = legacy == now;
        diffs += !same;
        std::printf("%-8s %s conv %zu tools=%d thinking=%d: %zu ids, %zu data spans (%zu of %zu bytes)\n",
                    same ? "same" : "DIFFERS", dir.substr(dir.find_last_of('/') + 1).c_str(), c, with_tools,
                    (variant & 2) == 0, now.size(), plain.size(), covered, text.size());
        if (!same) {
          size_t i = 0;
          while (i < legacy.size() && i < now.size() && legacy[i] == now[i]) ++i;
          std::printf("  first difference at id %zu: %d vs %d\n", i, i < legacy.size() ? legacy[i] : -1,
                      i < now.size() ? now[i] : -1);
        }
      }
    }
  }
  std::printf("plain_ids_check: %d prompts, %d differ\n", total, diffs);
  return diffs ? 1 : 0;
}
