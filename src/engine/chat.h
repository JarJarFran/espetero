// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// OpenAI-compatible chat completions and completions over the engine:
// request parsing, prompt rendering with the model's own chat template,
// streamed text assembly (incremental UTF-8, reasoning blocks, stop strings)
// and Qwen-style tool calls.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "engine.h"

namespace espetero {

struct ChatJob {
  GenRequest gen;
  bool reasoning_open = false;      // the prompt opened a <think> block
  bool split_reasoning = true;      // report reasoning separately (chat)
  std::vector<std::string> stops;
  bool stream = false;
  bool include_usage = false;
  json::Value tools;                // null when none
  std::string id;
  int64_t created = 0;
  bool chat = true;                 // false: /v1/completions (text_completion objects)
  bool echo = false;
  std::string prompt_text;          // completions echo
  int n = 1;                        // choices (OpenAI's n), one prefill for all of them
  // A forced tool call. `prefill`: the call's opening, at the end of the
  // prompt and the start of the reply (parsed with it). `force_after_think`:
  // the prompt opened the reasoning block, so generation stops at </think>
  // (`think_end`) and continues with this text after it.
  std::string prefill;
  std::string force_after_think;
  int32_t think_end = -1;
  // Keywords of a non-strict json_schema the reply does not enforce
  // (reported in the response as structured_output.unenforced).
  std::vector<std::string> unenforced;
};

constexpr int kMaxChoices = 16;     // n above this: 422

struct ChatEvent {
  std::string content, reasoning;
  bool final = false;
  std::string finish_reason;
  std::string stop_sequence;        // the stop string that ended the reply (final event; "" otherwise)
  json::Value tool_calls;           // array (final event) or null
  json::Value usage, timings;
  // Logprob records (GenRequest::logprobs >= 0): a streamed job's
  // events carry those of the tokens whose text they complete, the final one
  // the rest; otherwise the final event carries them all.
  std::vector<TokenLogprob> logprobs;
};

// Parse an OpenAI chat request (throws RequestError 422 on invalid input).
ChatJob prepare_chat(Engine& engine, const json::Value& body);
// Parse an OpenAI completions request.
ChatJob prepare_completion(Engine& engine, const json::Value& body);

// Run a job; on_event returns false to stop (client gone). The final event
// carries finish_reason, usage and timings.
void run_chat(Engine& engine, const ChatJob& job, const std::function<bool(const ChatEvent&)>& on_event);
// The same over a stream already submitted (choice `choice` of generate_n).
// With `stop`, the job also ends once *stop is set.
void run_chat_stream(Engine& engine, const ChatJob& job, std::shared_ptr<GenStream> stream,
                     const std::function<bool(const ChatEvent&)>& on_event, const std::atomic<bool>* stop = nullptr,
                     int choice = 0);
// The job's n choices, run at once (one prefill); on_event(choice, ev)
// is called on the caller's thread in arrival order, and returning false
// stops every choice. n = 1 is run_chat.
void run_choices(Engine& engine, const ChatJob& job, const std::function<bool(int, const ChatEvent&)>& on_event);

// One choice of a whole response.
struct ChoiceResult {
  std::string content, reasoning;
  ChatEvent last;
};
// A whole (non-streamed) chat or completions response: every choice run;
// `client_gone` (optional) stops them.
json::Value respond(Engine& engine, const ChatJob& job, const std::function<bool()>& client_gone = nullptr);

// Whole-response bodies.
json::Value chat_response(const Engine& engine, const ChatJob& job, const std::vector<ChoiceResult>& choices);
json::Value completion_response(const Engine& engine, const ChatJob& job, const std::vector<ChoiceResult>& choices);
json::Value chat_response(const Engine& engine, const ChatJob& job, const std::string& content,
                          const std::string& reasoning, const ChatEvent& last);
json::Value completion_response(const Engine& engine, const ChatJob& job, const std::string& text, const ChatEvent& last);
// The usage of a request's choices: the prompt once, every choice's completion tokens.
json::Value choices_usage(const std::vector<json::Value>& usages);

// Streaming chunk bodies (without the "data: " framing). `lps`: the logprob
// records this chunk carries (null: "logprobs": null); a completions stream
// passes `text_offset`, its running byte offset (legacy logprobs format).
// `index`: the choice.
json::Value chat_chunk(const Engine& engine, const ChatJob& job, const json::Value& delta, const json::Value& finish,
                       bool with_choices = true, const std::vector<TokenLogprob>* lps = nullptr,
                       int64_t* text_offset = nullptr, int index = 0);

// Tool calls in Qwen3.5/Qwen3-Coder format inside `content`; returns the text
// before the first call and fills `calls` (OpenAI shape) when any parse.
std::string parse_tool_calls(const std::string& content, const json::Value& tools, json::Value& calls);
// Tool calls in Gemma 4 format: <|tool_call>call:NAME{key:value,...}<tool_call|>,
// strings between <|"|> marks, bare keys, numbers, booleans, objects and arrays.
std::string parse_gemma_tool_calls(const std::string& content, json::Value& calls);

// The chat markers of the model's family: the reasoning block of
// generated text, the tool-call marks, and the turn header prompt checkpoints
// follow ("<|im_start|>" "assistant\n" for the Qwen3.5 family, "<|turn>"
// "model\n" for Gemma 4).
struct ChatMarkers {
  std::string think_open, think_close;  // around the reasoning in generated text
  std::string think_prompt;             // a prompt ending with this opened the reasoning block
  std::string tool_open, tool_close;    // around one tool call
  std::string turn, assistant, system;  // turn headers: turn + role
  bool gemma = false;
};
const ChatMarkers& chat_markers(const Engine& engine);

// The special tokens a chat template sees, as transformers passes its
// special_tokens_map (bos_token, eos_token, unk_token, pad_token). Gemma 4's
// tokenizers only (its template starts with {{ bos_token }}); null for
// the Qwen3.5 family, whose templates render exactly as before.
json::Value template_special_tokens(const Tokenizer& tok);

}  // namespace espetero
