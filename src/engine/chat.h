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
  std::string tool_open = "<tool_call>";  // where a reply's calls start: Qwen XML, or DeepSeek's "<｜DSML｜ calls>"
  std::string force_after_think;
  int32_t think_end = -1;
  // Keywords of a non-strict json_schema the reply does not enforce
  // (reported in the response as structured_output.unenforced).
  std::vector<std::string> unenforced;
  // "head": the hidden-state head's first head_slots logits instead of a
  // reply (0: a reply); see head_response.
  int head_slots = 0;
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
// run = false: the job is only rendered (token counts), never run, so a
// decision-only model (Engine::generates() false) renders it without "head".
ChatJob prepare_chat(Engine& engine, const json::Value& body, bool run = true);
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
// `client_gone` (optional) stops them. A "head" job gets head_response.
json::Value respond(Engine& engine, const ChatJob& job, const std::function<bool()>& client_gone = nullptr);

// The response to a "head" request: no reply ("content": ""), and "head":
// {"slots", "logits", "temperature", "probabilities"} with the hidden-state
// head's logits of the first `slots` slots (hidden_head.h) for the prompt's
// last token and softmax(logits / temperature), the head's calibration
// temperature (1 when hidden_head.json gives none).
json::Value head_response(Engine& engine, const ChatJob& job);

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

// Tool calls in Qwen3.5/Qwen3-Coder format inside `content`, or DeepSeek's
// DSML calls; returns the text before the first call and fills `calls`
// (OpenAI shape) when any parse. `at` (sorted, optional): the offsets in
// `content` where the model generated a call's opening token; a call opens
// only there, so an opening spelled with ordinary tokens is text. Null: at
// every occurrence of the opening.
std::string parse_tool_calls(const std::string& content, const json::Value& tools, json::Value& calls,
                             const std::vector<size_t>* at = nullptr);
// Tool calls in Gemma 4 format: <|tool_call>call:NAME{key:value,...}<tool_call|>,
// strings between <|"|> marks, bare keys, numbers, booleans, objects and arrays.
// `at` as for parse_tool_calls.
std::string parse_gemma_tool_calls(const std::string& content, json::Value& calls, const std::vector<size_t>* at = nullptr);
// Tool calls in GLM-4.5 / GLM-5 format: <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>
// (glm_tools.cpp); parse_tool_calls hands over to it when a call's body has that shape. `at` as for parse_tool_calls.
std::string parse_glm_tool_calls(const std::string& content, const json::Value& tools, json::Value& calls,
                                 const std::vector<size_t>* at = nullptr);
bool glm_tool_call_body(const std::string& body);  // "NAME" or "NAME<arg_key>...": not Qwen's "\n<function="
// The first position in `text` (from `from`) where `marker` starts: the first of `at` (sorted) that
// holds it, or any occurrence when `at` is null.
inline size_t find_marker(const std::string& text, const std::string& marker, const std::vector<size_t>* at,
                          size_t from = 0) {
  if (!at) return text.find(marker, from);
  for (size_t p : *at)
    if (p >= from && text.compare(p, marker.size(), marker) == 0) return p;
  return std::string::npos;
}

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

// `s` with invalid UTF-8 replaced by U+FFFD (the text layer only emits whole characters).
std::string sanitize_utf8(std::string_view s);

// A chat marker of generated text ("</think>", "<tool_call>", "<｜DSML｜
// calls>") and the added token in it (`anchor`, at byte `anchor_off` of the
// marker: DeepSeek's "<｜DSML｜ calls>" is "<", the token "｜DSML｜" and
// " calls>"). An occurrence counts only where the model generated that token;
// the same characters spelled with ordinary tokens (a quoted document, an
// echoed request) are text. No added token in it (anchor -1): every
// occurrence counts.
struct MarkerToken {
  std::string text;
  int32_t anchor = -1;
  size_t anchor_off = 0;
};
MarkerToken marker_token(const Tokenizer& tok, const std::string& text);

// A reply's text as it is generated, and where the model generated the end of
// its reasoning (mk.think_close) and the opening of its tool calls
// (`tool_open`), by their tokens (MarkerToken).
class ReplyText {
 public:
  // `reasoning_open`: the prompt opened the reasoning block; `split`: report
  // the reasoning apart (chat), else everything is content.
  ReplyText(const Tokenizer& tok, const ChatMarkers& mk, std::string tool_open, bool reasoning_open, bool split);
  void add(int32_t id);                 // a generated token
  void add_text(const std::string& t);  // text the server wrote (a forced call's opening): its markers count
  const std::string& raw() const { return raw_; }

  struct View {
    std::string reasoning, content;
    size_t held = 0;              // bytes at the end of raw() not shown yet (a partial tag or UTF-8 sequence)
    bool anchored = false;        // the tool-call opening has a token: calls open only at tool_at
    std::vector<size_t> tool_at;  // offsets in `content` where the model generated the opening
    const std::vector<size_t>* calls_at() const { return anchored ? &tool_at : nullptr; }
  };
  // `final_view`: all of it (else an incomplete UTF-8 tail waits).
  View view(bool final_view) const;
  // Where the calls start in `content` (the view's content, maybe cut by a
  // stop string), npos if nowhere.
  size_t calls_start(const View& v, const std::string& content) const;
  // The tool calls in `content` (parse_tool_calls or parse_gemma_tool_calls);
  // returns the text before them.
  std::string parse_calls(const View& v, const std::string& content, const json::Value& tools, json::Value& calls) const;

 private:
  void note(int32_t id, size_t at);
  const Tokenizer& tok_;
  const ChatMarkers& mk_;
  std::string tool_open_;
  bool reasoning_open_, split_;
  MarkerToken close_mark_, tool_mark_;
  std::string raw_;
  std::vector<size_t> close_raw_, tool_raw_;  // offsets in raw_ where the markers' tokens were generated
};

// The special tokens a chat template sees, as transformers passes its
// special_tokens_map (bos_token, eos_token, unk_token, pad_token). Gemma 4's
// tokenizers and LFM2's (its templates start with {{ bos_token }}); null for
// every other family (the Qwen3.5 family, DeepSeek, ...), whose templates
// render exactly as before.
json::Value template_special_tokens(const Tokenizer& tok);

}  // namespace espetero
