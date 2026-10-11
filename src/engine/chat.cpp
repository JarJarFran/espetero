// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "chat.h"

#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <random>
#include <system_error>
#include <thread>

#include "../tmpl/tmpl.h"
#include "../tok/tokenizer.h"
#include "../vision/fetch.h"
#include "requests.h"
#include "structured.h"

namespace espetero {

namespace {

[[noreturn]] void invalid(const std::string& msg) { throw RequestError(422, msg); }

constexpr size_t kMaxChatImages = 32;  // per request (/v1/decisions and Clef: 16)

std::string random_id(const char* prefix) {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  static const char* h = "0123456789abcdef";
  std::string s = prefix;
  for (int i = 0; i < 24; ++i) s.push_back(h[rng() & 15]);
  return s;
}

int64_t now_s() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// Replace invalid UTF-8 by U+FFFD (the text layer only emits whole characters).
std::string sanitize(std::string_view s) {
  if (json::utf8_valid(s)) return std::string(s);
  std::string out;
  size_t i = 0;
  while (i < s.size()) json::utf8_append(out, json::utf8_next(s, i));
  return out;
}

// OpenAI chat logprobs content: per token its text (invalid UTF-8
// replaced), its bytes, log-probability and the top alternatives.
json::Value logprobs_json(const Tokenizer& tok, const std::vector<TokenLogprob>& lps) {
  auto entry = [&](int32_t id, float lp) {
    const std::string& p = tok.piece(id);
    json::Value e = json::Value::object();
    e.set("token", sanitize(p));
    e.set("id", int64_t(id));  // beside OpenAI's fields: the token id
    e.set("logprob", std::isfinite(lp) ? double(lp) : -9999.0);
    json::Value b = json::Value::array();
    for (unsigned char ch : p) b.push_back(int64_t(ch));
    e.set("bytes", std::move(b));
    return e;
  };
  json::Value out = json::Value::array();
  for (const TokenLogprob& t : lps) {
    json::Value e = entry(t.token, t.logprob);
    json::Value top = json::Value::array();
    for (const auto& [id, lp] : t.top) top.push_back(entry(id, lp));
    e.set("top_logprobs", std::move(top));
    out.push_back(std::move(e));
  }
  return out;
}

// OpenAI's legacy completions logprobs: the tokens' texts, their
// log-probabilities, per token the top alternatives as {text: logprob} (the
// sampled token too when it is not among them, as OpenAI does) and each
// token's byte offset in the text, counted from `*offset` (advanced).
json::Value legacy_logprobs_json(const Tokenizer& tok, const std::vector<TokenLogprob>& lps, int64_t* offset) {
  json::Value tokens = json::Value::array(), values = json::Value::array(), tops = json::Value::array(),
              offs = json::Value::array();
  auto val = [](float lp) { return std::isfinite(lp) ? double(lp) : -9999.0; };
  int64_t off = offset ? *offset : 0;
  for (const TokenLogprob& t : lps) {
    const std::string& p = tok.piece(t.token);
    const std::string text = sanitize(p);
    tokens.push_back(text);
    values.push_back(val(t.logprob));
    json::Value top = json::Value::object();
    for (const auto& [id, lp] : t.top) {
      const std::string k = sanitize(tok.piece(id));
      if (!top.get(k)) top.set(k, val(lp));  // pieces that decode alike: the likelier one
    }
    if (!top.get(text)) top.set(text, val(t.logprob));
    tops.push_back(std::move(top));
    offs.push_back(off);
    off += int64_t(p.size());
  }
  if (offset) *offset = off;
  json::Value o = json::Value::object();
  o.set("tokens", std::move(tokens));
  o.set("token_logprobs", std::move(values));
  o.set("top_logprobs", std::move(tops));
  o.set("text_offset", std::move(offs));
  return o;
}

bool parse_bool(const json::Value* v, bool def) { return v && v->is_bool() ? v->as_bool() : def; }

double num(const json::Value& body, const char* key, double def, double lo, double hi) {
  const json::Value* v = body.get(key);
  if (!v || v->is_null()) return def;
  if (!v->is_number()) invalid(std::string(key) + ": a number is required");
  const double x = v->as_double();
  if (!std::isfinite(x) || x < lo || x > hi) invalid(std::string(key) + ": out of range");
  return x;
}

// Common sampling options of chat and completions requests. Omitted
// temperature / top_p / top_k / min_p take the model's defaults (`d`, its
// generation_config.json); greedy requests keep OpenAI's for the other three.
void sampling_options(const json::Value& body, ChatJob& job, int default_max, const SamplingDefaults& d) {
  GenRequest& g = job.gen;
  g.sampling.temperature = num(body, "temperature", d.temperature.value_or(1.0), 0, 2);
  const bool sampled = g.sampling.temperature > 0;
  g.sampling.top_p = num(body, "top_p", sampled ? d.top_p.value_or(1.0) : 1.0, 1e-9, 1);
  g.sampling.top_k = int(num(body, "top_k", sampled ? d.top_k.value_or(0) : 0, 0, 1e6));
  g.sampling.min_p = num(body, "min_p", sampled ? d.min_p.value_or(0) : 0, 0, 1);
  // Not implemented yet: refused, never silently ignored.
  if (const json::Value* lb = body.get("logit_bias"); lb && !lb->is_null() && !(lb->is_object() && lb->size() == 0))
    invalid("logit_bias is not supported yet");
  g.sampling.repetition_penalty = num(body, "repetition_penalty", 1.0, 1e-3, 10);
  g.sampling.presence_penalty = num(body, "presence_penalty", 0, -2, 2);
  g.sampling.frequency_penalty = num(body, "frequency_penalty", 0, -2, 2);
  int max_tokens = default_max;
  for (const char* k : {"max_completion_tokens", "max_tokens"}) {
    const json::Value* v = body.get(k);
    if (v && !v->is_null()) {
      if (!v->is_int() || v->as_int() < 1) invalid(std::string(k) + ": a positive integer");
      max_tokens = int(std::min<int64_t>(v->as_int(), 1 << 20));
      break;
    }
  }
  g.max_tokens = max_tokens;
  if (const json::Value* s = body.get("seed"); s && s->is_int()) {
    g.seed = uint64_t(s->as_int());
    g.has_seed = true;
  }
  if (const json::Value* s = body.get("stop"); s && !s->is_null()) {
    if (s->is_string()) job.stops.push_back(s->as_string());
    else if (s->is_array()) {
      for (const auto& x : s->as_array()) {
        if (!x.is_string()) invalid("stop: strings only");
        job.stops.push_back(x.as_string());
      }
    } else {
      invalid("stop: a string or a list of strings");
    }
    if (job.stops.size() > 16) invalid("stop: at most 16 strings");
    for (const auto& x : job.stops)
      if (x.empty() || x.size() > 512) invalid("stop: nonempty strings of at most 512 bytes");
  }
  if (const json::Value* s = body.get("stop_token_ids"); s && s->is_array())
    for (const auto& x : s->as_array())
      if (x.is_int()) g.stop_ids.push_back(int32_t(x.as_int()));
  g.ignore_eos = parse_bool(body.get("ignore_eos"), false);
  job.stream = parse_bool(body.get("stream"), false);
  if (const json::Value* so = body.get("stream_options"); so && so->is_object())
    job.include_usage = parse_bool(so->get("include_usage"), false);
  // The n choices of one prompt, its prefill computed once (Engine::generate_n).
  if (const json::Value* n = body.get("n"); n && !n->is_null()) {
    if (!n->is_int() || n->as_int() < 1 || n->as_int() > kMaxChoices)
      invalid("n: an integer from 1 to " + std::to_string(kMaxChoices));
    job.n = int(n->as_int());
  }
  // Completions' best_of (generate several, return the likeliest) is not implemented.
  if (const json::Value* b = body.get("best_of"); b && !b->is_null() && !(b->is_int() && b->as_int() == job.n))
    invalid("best_of: not supported (n returns every choice it generates)");
  if (const json::Value* p = body.get("priority"); p && p->is_int()) g.priority = int(p->as_int());
  if (const json::Value* sp = body.get("speculative"); sp && sp->is_bool()) g.speculate = sp->as_bool();
  if (const json::Value* cp = body.get("cache_prompt"); cp && cp->is_bool()) g.cache_prompt = cp->as_bool() ? 1 : 0;
  // Each generated token's log-probability and its top alternatives,
  // read from the GPU top-k while generating; sampled and streamed requests
  // too (the model's raw distribution, see GenRequest::logprobs).
  // Chat: logprobs true and top_logprobs 0-20; completions: logprobs N, 0-20
  // alternatives in OpenAI's legacy format.
  const json::Value* lp = body.get("logprobs");
  const json::Value* tl = body.get("top_logprobs");
  if (job.chat) {
    if (lp && !lp->is_null() && !lp->is_bool()) invalid("logprobs: true or false (top_logprobs sets the alternatives, 0 to 20)");
    if (lp && lp->is_bool() && lp->as_bool()) g.logprobs = int(num(body, "top_logprobs", 0, 0, 20));
    else if (tl && !tl->is_null()) invalid("top_logprobs: needs logprobs: true");
  } else {
    if (tl && !tl->is_null()) invalid("top_logprobs: a chat completions field (/v1/completions takes logprobs: N)");
    if (lp && lp->is_bool()) {
      if (lp->as_bool()) g.logprobs = 0;
    } else if (lp && !lp->is_null()) {
      if (!lp->is_int() || lp->as_int() < 0 || lp->as_int() > 20) invalid("logprobs: an integer from 0 to 20 (alternatives per token)");
      g.logprobs = int(lp->as_int());
    }
  }
}

// An http:// or https:// URL (the scheme in any case).
bool http_url(const std::string& url) {
  auto starts = [&](const char* p) {
    const size_t n = std::strlen(p);
    if (url.size() < n) return false;
    for (size_t i = 0; i < n; ++i)
      if (std::tolower(static_cast<unsigned char>(url[i])) != p[i]) return false;
    return true;
  };
  return starts("http://") || starts("https://");
}

// A request's http(s) image URLs, fetched (--image-urls) up to 4 at a
// time, each replaced by the data URL of its bytes: from there on the image
// takes the same path and limits as a client's data URL (header probe,
// decode cap, token budget). At most kMaxFetchedBytes for one request.
constexpr size_t kMaxFetchedBytes = size_t(128) << 20;

void fetch_remote_images(const Engine& engine, std::vector<std::string>& urls) {
  std::vector<size_t> remote;
  for (size_t i = 0; i < urls.size(); ++i)
    if (http_url(urls[i])) remote.push_back(i);
  if (remote.empty()) return;
  vision::FetchOptions fo;
  fo.allow_private = engine.options().image_urls_private;
  std::vector<std::string> data(remote.size()), errs(remote.size());
  std::vector<char> ok(remote.size(), 0);
  size_t total = 0;
  for (size_t b = 0; b < remote.size(); b += 4) {
    const size_t e = std::min(remote.size(), b + 4);
    auto fetch = [&](size_t k) { ok[k] = vision::fetch_image_data_url(urls[remote[k]], fo, data[k], errs[k]); };
    std::vector<std::thread> th;
    for (size_t k = b; k < e; ++k) {
      try {
        th.emplace_back(fetch, k);
      } catch (const std::system_error&) {
        fetch(k);  // no thread left: this one here
      }
    }
    for (auto& t : th) t.join();
    for (size_t k = b; k < e; ++k) {
      const std::string& u = urls[remote[k]];
      if (!ok[k]) invalid("image_url " + (u.size() > 160 ? u.substr(0, 160) + "..." : u) + ": " + errs[k]);
      total += data[k].size() / 4 * 3;
      if (total > kMaxFetchedBytes)
        throw RequestError(413, "images fetched by URL: more than " + std::to_string(kMaxFetchedBytes >> 20) +
                                    " MB in one request");
    }
  }
  for (size_t k = 0; k < remote.size(); ++k) urls[remote[k]] = std::move(data[k]);
}

// (reasoning, content, closed) of generated text; `opened`: the prompt
// already opened the reasoning block.
struct Split {
  std::string reasoning, content;
  bool closed = true;
  size_t held = 0;  // trailing bytes held back as a partial tag (their tokens' logprobs wait too)
};
Split split_reasoning(const std::string& raw, bool opened, const ChatMarkers& mk) {
  const std::string& open = mk.think_open;
  const std::string& close = mk.think_close;
  Split out;
  std::string text = raw;
  if (!opened) {
    if (text.compare(0, open.size(), open) != 0) {
      // A partial "<think>" at the very start is held back.
      if (!text.empty() && open.compare(0, text.size(), text) == 0) {
        out.closed = false;
        out.held = text.size();
        return out;
      }
      out.content = raw;
      return out;
    }
    text = text.substr(open.size());
  }
  const size_t end = text.find(close);
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
    return out;
  }
  out.reasoning = strip_nl(text.substr(0, end), true, true);
  out.content = strip_nl(text.substr(end + close.size()), true, false);
  return out;
}

}  // namespace

const ChatMarkers& chat_markers(const Engine& engine) {
  static const ChatMarkers qwen{"<think>", "</think>", "<think>\n", "<tool_call>", "</tool_call>",
                                "<|im_start|>", "assistant\n", "system\n", false};
  // Gemma 4: the thought channel, its tool-call marks and turns.
  static const ChatMarkers gemma{"<|channel>thought", "<channel|>", "<|channel>thought\n", "<|tool_call>", "<tool_call|>",
                                 "<|turn>", "model\n", "system\n", true};
  return engine.info().family == "gemma4" ? gemma : qwen;
}

json::Value template_special_tokens(const Tokenizer& tok) {
  if (tok.pre_tokenizer_name() != "spm") return json::Value();
  json::Value o = json::Value::object();
  auto put = [&](const char* key, int32_t id) {
    if (id >= 0 && !tok.token_text(id).empty()) o.set(key, std::string(tok.token_text(id)));
  };
  put("bos_token", tok.bos_id());
  put("eos_token", tok.eos_id());
  put("pad_token", tok.pad_id());
  put("unk_token", tok.token_id("<unk>"));
  return o;
}

// ---------------------------------------------------------------- tools
std::string parse_tool_calls(const std::string& content, const json::Value& tools, json::Value& calls) {
  calls = json::Value();
  const std::string open = "<tool_call>", close = "</tool_call>";
  size_t at = content.find(open);
  if (at == std::string::npos) return content;
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
      call.set("id", random_id("call_"));
      call.set("type", "function");
      call.set("function", std::move(fn));
      out.push_back(std::move(call));
    }
    if (end == std::string::npos) break;
    at = content.find(open, end + close.size());
  }
  if (out.size()) calls = std::move(out);
  return calls.is_null() ? content : before;
}

// ---------------------------------------------------------------- prepare
// Teacher-forced scoring: {"score": {"continuation": "...", "top_k": 5}}
// scores that text as the answer instead of generating one.
static void score_options(Engine& engine, const json::Value& body, ChatJob& job) {
  const json::Value* sc = body.get("score");
  if (!sc || sc->is_null()) return;
  if (!sc->is_object()) invalid("score: an object with a continuation");
  const json::Value* c = sc->get("continuation");
  if (!c || !c->is_string() || c->as_string().empty()) invalid("score.continuation: a nonempty string");
  engine.check_text_size(c->as_string().size(), "score.continuation");
  GenRequest& g = job.gen;
  g.forced = engine.tokenizer().encode(c->as_string(), false);
  if (g.forced.empty()) invalid("score.continuation: no tokens");
  g.score_topk = int(num(*sc, "top_k", 0, 0, 64));
  if (const json::Value* ids = sc->get("ids"); ids && !ids->is_null()) {
    if (!ids->is_array() || ids->size() > g.forced.size()) invalid("score.ids: at most one id list per scored token");
    const int V = engine.info().vocab;
    for (const json::Value& row : ids->as_array()) {
      if (!row.is_array() || row.size() > 256) invalid("score.ids: lists of at most 256 token ids");
      std::vector<int32_t> r;
      for (const json::Value& x : row.as_array()) {
        if (!x.is_int() || x.as_int() < 0 || x.as_int() >= V) invalid("score.ids: token id out of range");
        r.push_back(int32_t(x.as_int()));
      }
      g.score_ids.push_back(std::move(r));
    }
  }
  g.max_tokens = int(g.forced.size());
  g.ignore_eos = true;
  g.speculate = false;
  g.sampling = SamplingParams{};  // greedy
  g.sampling.temperature = 0;
  // The scored tokens' log-probabilities are the score's own readout (timings.score).
  if (g.logprobs >= 0) invalid("logprobs: not with score (timings.score has the scored tokens' log-probabilities)");
  if (job.n > 1) invalid("n: one choice with score (the continuation is given)");
}

ChatJob prepare_chat(Engine& engine, const json::Value& body) {
  if (!body.is_object()) invalid("request body must be a JSON object");
  if (!engine.chat_template()) invalid("this model has no chat template");
  ChatJob job;
  job.id = random_id("chatcmpl-");
  job.created = now_s();
  sampling_options(body, job, 1024, engine.sampling_defaults());
  score_options(engine, body, job);
  parse_response_format(body.get("response_format"));  // its shape, before the prompt is rendered
  // tool_choice: "auto" (the default); "none": the prompt is rendered without
  // the tools, so no call is parsed; allowed_tools: only those are rendered;
  // "required" or a named function force a call (below).
  const ToolChoice choice = parse_tool_choice(body.get("tool_choice"), body.get("tools"));
  const bool offer_tools = choice.mode != ToolChoice::None;
  const bool forced = choice.mode == ToolChoice::Required || choice.mode == ToolChoice::Named;
  if (forced) {
    // The reply starts with the opening of a call in the template's format,
    // which the parser must understand: Qwen's XML calls (parse_tool_calls).
    const std::string& src = engine.tokenizer().chat_template();
    if (src.find("<tool_call>") == std::string::npos || src.find("<function=") == std::string::npos)
      invalid("tool_choice: forcing a call needs a chat template with <tool_call><function=...> calls; this model's has "
              "another format (only \"auto\" and \"none\" here)");
    if (job.gen.ignore_eos) invalid("tool_choice: a forced call does not take ignore_eos");
  }
  const json::Value* msgs = body.get("messages");
  if (!msgs || !msgs->is_array() || msgs->size() < 1 || msgs->size() > 4096) invalid("messages: 1 to 4096 messages");
  // Messages for the template: images become {"type": "image"} parts; their
  // data URLs are prepared in order.
  json::Value conv = json::Value::array();
  std::vector<std::string> urls;
  std::vector<int64_t> detail_pixels;
  // Cache marks: a text part with "cache_control" (any object, as Anthropic's) or "checkpoint": true
  // ends a prefix later requests share. A private-use sequence marks its end in the rendered prompt, which
  // is tokenized in segments at the marks, so the boundary falls exactly between tokens.
  static const std::string kMark = "\xEE\x80\x80" "espetero-cache-mark\xEE\x80\x80";
  int n_marks = 0;
  for (const json::Value& m : msgs->as_array()) {
    if (!m.is_object()) invalid("messages: objects only");
    std::string role = m.get_string("role", "");
    if (role == "developer") role = "system";
    if (role != "system" && role != "user" && role != "assistant" && role != "tool")
      invalid("unsupported message role '" + role + "'");
    json::Value out = json::Value::object();
    out.set("role", role);
    const json::Value* content = m.get("content");
    if (!content || content->is_null() || content->is_string()) {
      out.set("content", content && content->is_string() ? *content : json::Value(""));
    } else if (content->is_array()) {
      json::Value parts = json::Value::array();
      for (const json::Value& part : content->as_array()) {
        const std::string type = part.get_string("type", "");
        if (type == "text" || type == "input_text") {
          json::Value t = json::Value::object();
          t.set("type", "text");
          std::string txt = part.get_string("text", "");
          if (txt.find(kMark) != std::string::npos) invalid("text parts may not contain the cache mark sequence");
          const json::Value* cc = part.get("cache_control");
          const json::Value* ck = part.get("checkpoint");
          if ((cc && cc->is_object()) || (ck && ck->is_bool() && ck->as_bool())) {
            txt += kMark;
            ++n_marks;
          }
          t.set("text", txt);
          parts.push_back(std::move(t));
        } else if (type == "image_url" || type == "image" || type == "input_image") {
          const json::Value* src = part.get("image_url");
          if (!src) src = part.get("image");
          std::string url;
          int64_t pixels = 0;
          if (src && src->is_string()) url = src->as_string();
          else if (src && src->is_object()) {
            url = src->get_string("url", "");
            const std::string detail = src->get_string("detail", "auto");
            // detail (OpenAI's field, plus "max"): low = 512x512, auto = the
            // server's standard budget, high = 4x standard, max = the
            // processor's own ceiling.
            pixels = engine.detail_pixels(detail);
            if (pixels < 0) invalid("image_url.detail: low, auto, high or max");
            engine.check_detail(pixels, "image_url.detail", detail);
          }
          // An http(s) URL is fetched (fetch_remote_images) when the operator allows it.
          if (http_url(url)) {
            if (!engine.options().image_urls)
              invalid("image_url: http(s) URLs are fetched only when the server runs with --image-urls (off by default); "
                      "send the image as a base64 data URL (data:image/...;base64,...)");
          } else if (url.compare(0, 11, "data:image/") != 0 || url.find(";base64,") == std::string::npos) {
            invalid(std::string("image_url: a base64 data URL (data:image/...;base64,...)") +
                    (engine.options().image_urls ? " or an http(s) URL" : "; http(s) URLs need --image-urls"));
          }
          if (urls.size() == kMaxChatImages) invalid("at most 32 images per request");
          urls.push_back(url);
          detail_pixels.push_back(pixels);
          json::Value t = json::Value::object();
          t.set("type", "image");
          parts.push_back(std::move(t));
        } else {
          invalid("unsupported content part '" + type + "'");
        }
      }
      out.set("content", std::move(parts));
    } else {
      invalid("content: a string or a list of parts");
    }
    if (const json::Value* rc = m.get("reasoning_content"); rc && rc->is_string()) out.set("reasoning_content", *rc);
    if (const json::Value* tc = m.get("tool_calls"); tc && tc->is_array()) {
      // Templates iterate arguments as a mapping; OpenAI sends a JSON string.
      json::Value calls = *tc;
      for (json::Value& c : calls.as_array()) {
        json::Value* fn = c.get("function");
        json::Value* args = fn ? fn->get("arguments") : nullptr;
        if (args && args->is_string()) {
          try {
            json::Value parsed = json::parse(args->as_string());
            if (parsed.is_object()) *args = std::move(parsed);
          } catch (...) {
          }
        }
      }
      out.set("tool_calls", std::move(calls));
    }
    if (const json::Value* id = m.get("tool_call_id"); id && id->is_string()) out.set("tool_call_id", *id);
    if (const json::Value* nm = m.get("name"); nm && nm->is_string()) out.set("name", *nm);
    conv.push_back(std::move(out));
  }
  json::Value kwargs = json::Value::object();
  if (const json::Value* k = body.get("chat_template_kwargs"); k && k->is_object()) kwargs = *k;
  if (const json::Value* t = body.get("enable_thinking"); t && t->is_bool()) kwargs.set("enable_thinking", *t);
  if (const json::Value* t = body.get("tools"); offer_tools && t && t->is_array() && t->size()) {
    job.tools = *t;
    if (!choice.allowed.empty()) {  // allowed_tools: the prompt offers those only
      json::Value kept = json::Value::array();
      for (const json::Value& x : t->as_array()) {
        const json::Value* f = x.get("function");
        const std::string n = f && f->is_object() ? f->get_string("name", "") : std::string();
        if (std::find(choice.allowed.begin(), choice.allowed.end(), n) != choice.allowed.end()) kept.push_back(x);
      }
      job.tools = std::move(kept);
    }
  }
  std::string text;
  try {
    text = tmpl::apply_chat_template(*engine.chat_template(), conv, job.tools, true, kwargs,
                                     template_special_tokens(engine.tokenizer()));
  } catch (const std::exception& e) {
    invalid(std::string("chat template: ") + e.what());
  }
  // A forced call. The reply begins with the call's opening, as the
  // template renders an assistant turn with calls: "<tool_call>\n<function="
  // and the name (a named function), else "<tool_call>\n<function", whose
  // continuation names one of the tools (cut there, the name's own tokens
  // stay the model's, as in its training). A prompt that opened the reasoning
  // block first lets the model think: generation stops at </think>, then a
  // second phase continues the same prompt, the reasoning, </think> and
  // "\n\n" + the opening (run_chat_stream).
  if (forced) {
    const std::string open = choice.mode == ToolChoice::Named ? "<tool_call>\n<function=" + choice.name + ">\n"
                                                              : std::string("<tool_call>\n<function");
    static const std::string think = "<think>\n";
    const int32_t think_end = engine.tokenizer().token_id("</think>");
    if (text.size() >= think.size() && text.compare(text.size() - think.size(), think.size(), think) == 0 && think_end >= 0) {
      job.force_after_think = "\n\n" + open;
      job.think_end = think_end;
      job.gen.stop_ids.push_back(think_end);
    } else {
      if (text.size() >= think.size() && text.compare(text.size() - think.size(), think.size(), think) == 0)
        text += "</think>\n\n";  // no single </think> token to stop at: no reasoning
      text += open;
      job.prefill = open;
    }
  }
  // Structured output (response_format json_object / json_schema): a
  // grammar constrains the reply, after the reasoning when the prompt opens it.
  apply_response_format(engine, body, job, text, true);
  // Rendering is cheap (tens of ms for 32 MB); tokenizing is not (12.8 s and
  // 1.2 GB for 32 MB). The rendered prompt is what must fit: history the
  // template drops (earlier turns' reasoning) does not count.
  engine.check_text_size(text.size(), "prompt");
  const Tokenizer& tok = engine.tokenizer();
  std::vector<int32_t> ids;
  std::vector<int> marks;  // token offsets of the cache marks, before the images are expanded
  if (n_marks) {
    for (size_t pos = 0;;) {
      const size_t m = text.find(kMark, pos);
      tok.encode_append(std::string_view(text).substr(pos, (m == std::string::npos ? text.size() : m) - pos), ids);
      if (m == std::string::npos) break;
      marks.push_back(int(ids.size()));
      pos = m + kMark.size();
    }
  } else {
    ids = tok.encode(text);
  }
  GenRequest& g = job.gen;
  // Gemma 4 runs text only so far: its vision tower is not loaded.
  if (!urls.empty() && chat_markers(engine).gemma)
    throw RequestError(422, "images are not supported for Gemma 4 yet: Espetero runs it text only");
  fetch_remote_images(engine, urls);
  if (!urls.empty()) {
    // Each image's token run from its header: a prompt that cannot fit the
    // context window is refused (as generate() would) before any decoding.
    std::vector<int> runs;
    for (size_t i = 0; i < urls.size(); ++i) runs.push_back(engine.image_tokens(urls[i], detail_pixels[i]));
    const size_t n = engine.expanded_length(ids, runs);
    if (n >= size_t(engine.info().max_ctx))
      throw RequestError(422, "Prompt exceeds the model context window (" + std::to_string(engine.info().max_ctx) +
                                  " tokens): " + std::to_string(n) + " tokens with its images");
  }
  // Images (decoded in this thread; preprocessing is CPU work).
  for (size_t i = 0; i < urls.size(); ++i) g.images.push_back(engine.prepare_image(urls[i], detail_pixels[i]));
  if (!g.images.empty()) {
    if (!marks.empty()) {  // each image's placeholder becomes its run of image tokens: shift the marks after it
      std::vector<int> shift(marks.size(), 0);
      size_t k = 0;
      for (size_t i = 0; i < ids.size() && k < g.images.size(); ++i)
        if (ids[i] == engine.image_token) {
          const int extra = g.images[k++].tokens(engine.info().vision_merge) - 1;
          for (size_t j = 0; j < marks.size(); ++j)
            if (marks[j] > int(i)) shift[j] += extra;
        }
      for (size_t j = 0; j < marks.size(); ++j) marks[j] += shift[j];
    }
    ids = engine.expand_images(ids, g.images);
    g.positions = engine.mrope_positions(ids, g.images);
    const int merge = engine.info().vision_merge;
    size_t k = 0;
    for (size_t i = 0; i < ids.size() && k < g.images.size();) {
      if (ids[i] == engine.image_token) {
        g.image_rows.push_back(int(i));
        i += size_t(g.images[k++].tokens(merge));
      } else {
        ++i;
      }
    }
  }
  // Checkpoint boundary: right after the final "<|im_start|>assistant\n"
  // (templates drop earlier reasoning when a conversation is re-sent, so that
  // is where the next turn's prompt still matches). Gemma 4: "<|turn>model\n"
  // (its empty thought channel after it is dropped from history too).
  const ChatMarkers& mk = chat_markers(engine);
  const int32_t im_start = tok.token_id(mk.turn);
  const std::vector<int32_t> assistant = tok.encode(mk.assistant, false);
  g.split = -1;
  for (int i = int(ids.size()) - 1; i >= 0 && im_start >= 0; --i)
    if (ids[size_t(i)] == im_start) {
      if (size_t(i) + 1 + assistant.size() <= ids.size() &&
          std::equal(assistant.begin(), assistant.end(), ids.begin() + i + 1))
        g.split = i + 1 + int(assistant.size());
      break;
    }
  // System boundary: the start of the first user turn, when the prompt opens
  // with a system message. A checkpoint there is shared by every
  // conversation with the same system prompt (the "prefix cache").
  g.sys_split = -1;
  // (A first turn after the BOS token, as Gemma 4's template writes it, counts too.)
  const size_t t0 = mk.gemma && !ids.empty() && ids[0] == tok.bos_id() && ids[0] != im_start ? 1 : 0;
  if (im_start >= 0 && ids.size() > t0 && ids[t0] == im_start) {
    const std::vector<int32_t> system = tok.encode(mk.system, false);
    if (t0 + 1 + system.size() <= ids.size() && std::equal(system.begin(), system.end(), ids.begin() + long(t0) + 1))
      for (size_t i = t0 + 1 + system.size(); i < ids.size(); ++i)
        if (ids[i] == im_start) {
          g.sys_split = int(i);
          break;
        }
    if (g.sys_split >= 0 && g.split >= 0 && g.sys_split >= g.split) g.sys_split = -1;
  }
  // Image boundary: right after the last image run, when the prompt
  // goes on before its own checkpoint. Questions about the same image(s)
  // under the same instructions share everything up to here.
  g.img_split = -1;
  if (!g.image_rows.empty()) {
    const int img_end = g.image_rows.back() + g.images.back().tokens(engine.info().vision_merge);
    if (img_end < int(ids.size()) && (g.split < 0 || img_end < g.split) && img_end > g.sys_split) g.img_split = img_end;
  }
  // Instruction boundary: just before the first image (and its
  // <|vision_start|>), when the user turn has at least 32 tokens of its own
  // text there. Requests with the same system prompt and instructions share
  // everything up to here: instructions placed before the photo are computed
  // once for a whole set of photos.
  g.pre_split = -1;
  if (!g.image_rows.empty()) {
    int p = g.image_rows[0];
    const int32_t vs = tok.token_id("<|vision_start|>");
    if (vs >= 0 && p > 0 && ids[size_t(p - 1)] == vs) --p;
    if (p - std::max(g.sys_split, 0) >= 32 && p > g.sys_split && (g.split < 0 || p < g.split)) g.pre_split = p;
  }
  // Cache marks: the last mark (or the end of the last image, if later) is the item boundary, shared by
  // every request about the same image or context; an earlier mark is the instruction boundary, shared by
  // every item under the same instructions (computed once, single-flight). A text context had only the
  // system checkpoint before: each question about it computed the context again.
  {
    std::vector<int> mk;
    for (int b : marks)
      if (b > g.sys_split && b < int(ids.size()) && (g.split < 0 || b < g.split)) mk.push_back(b);
    if (!mk.empty()) {
      const int item = std::max(g.img_split, mk.back());
      if (item < int(ids.size()) && (g.split < 0 || item < g.split)) g.img_split = item;
      if (mk.front() < g.img_split) g.pre_split = mk.front();
    }
  }
  // Prefix hashes (FNV-1a over the token ids): a prompt whose shared prefix
  // another request is computing waits for that checkpoint.
  auto fnv = [&](int n) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; ++i) {
      h ^= uint64_t(uint32_t(ids[size_t(i)]));
      h *= 1099511628211ull;
    }
    return h;
  };
  if (g.sys_split > 0) g.sys_hash = fnv(g.sys_split);
  if (g.pre_split > 0) g.pre_hash = fnv(g.pre_split);
  // (Gemma 4's <|channel> is a special token: parsed as one, as the prompt was.)
  const std::vector<int32_t> think = tok.encode(mk.think_prompt, mk.gemma);
  job.reasoning_open = ids.size() >= think.size() && std::equal(think.begin(), think.end(), ids.end() - long(think.size()));
  g.prompt = std::move(ids);
  return job;
}

ChatJob prepare_completion(Engine& engine, const json::Value& body) {
  if (!body.is_object()) invalid("request body must be a JSON object");
  ChatJob job;
  job.chat = false;
  job.split_reasoning = false;
  job.id = random_id("cmpl-");
  job.created = now_s();
  sampling_options(body, job, 16, engine.sampling_defaults());
  score_options(engine, body, job);
  job.echo = parse_bool(body.get("echo"), false);
  if (job.echo && job.gen.logprobs >= 0) invalid("logprobs: not with echo (the prompt tokens' log-probabilities are not computed)");
  {
    std::string none;
    apply_response_format(engine, body, job, none, false);  // from the reply's first token
  }
  const json::Value* p = body.get("prompt");
  if (!p) invalid("prompt: required");
  if (p->is_string()) {
    job.prompt_text = p->as_string();
    engine.check_text_size(job.prompt_text.size(), "prompt");
    job.gen.prompt = engine.tokenizer().with_bos(
        engine.tokenizer().encode(job.prompt_text, parse_bool(body.get("parse_special"), true)));
  } else if (p->is_array() && p->size() && (*p)[0].is_int()) {
    // A prompt of the whole window cannot leave room for a token (generate()
    // refuses it): no need to decode more than that for the echo text.
    if (p->size() >= size_t(engine.info().max_ctx))
      throw RequestError(422, "Prompt exceeds the model context window (" + std::to_string(engine.info().max_ctx) +
                                  " tokens)");
    const int64_t vocab = engine.info().vocab;
    for (const auto& t : p->as_array()) {
      if (!t.is_int()) invalid("prompt: token ids must be integers");
      if (t.as_int() < 0 || t.as_int() >= vocab) invalid("token id out of range");
      job.gen.prompt.push_back(int32_t(t.as_int()));
    }
    job.prompt_text = engine.tokenizer().decode(job.gen.prompt);
  } else if (p->is_array() && p->size() == 1 && (*p)[0].is_string()) {
    job.prompt_text = (*p)[0].as_string();
    engine.check_text_size(job.prompt_text.size(), "prompt");
    job.gen.prompt = engine.tokenizer().with_bos(engine.tokenizer().encode(job.prompt_text));
  } else {
    invalid("prompt: a string or a list of token ids (one prompt per request)");
  }
  return job;
}

// ---------------------------------------------------------------- run
void run_chat(Engine& engine, const ChatJob& job, const std::function<bool(const ChatEvent&)>& on_event) {
  run_chat_stream(engine, job, engine.generate(job.gen), on_event);
}

void run_chat_stream(Engine& engine, const ChatJob& job, std::shared_ptr<GenStream> stream,
                     const std::function<bool(const ChatEvent&)>& on_event, const std::atomic<bool>* stop, int choice) {
  const Tokenizer& tok = engine.tokenizer();
  std::string raw = job.prefill;  // generated bytes, after a forced call's opening
  size_t emitted_c = 0, emitted_r = 0;
  size_t hold = 0;
  for (const auto& s : job.stops) hold = std::max(hold, s.size() - 1);
  const bool tools = !job.tools.is_null();
  const ChatMarkers& mk = chat_markers(engine);
  if (tools) hold = std::max(hold, mk.tool_open.size() - 1);
  bool stop_hit = false;
  std::string stop_seq;  // the stop string that cut the content (the earliest match)
  std::vector<int32_t> got;
  // The generated tokens' logprob records and where each token's bytes
  // end in `raw`. A streamed event carries the records of the tokens whose
  // text it completes (text held back for stop strings, tool calls or a
  // partial tag keeps its tokens' records for a later event).
  const bool want_lp = job.gen.logprobs >= 0;
  std::vector<TokenLogprob> lps;
  std::vector<size_t> tok_end;
  size_t lp_sent = 0;
  // `held`: bytes at the end of `raw` the view does not show yet.
  auto view = [&](bool final_view, std::string& reasoning, std::string& content, size_t& safe, size_t& held) {
    const size_t whole = json::utf8_complete_prefix(raw);
    const std::string text = sanitize(std::string_view(raw).substr(0, final_view ? raw.size() : whole));
    held = raw.size() - (final_view ? raw.size() : whole);
    if (job.split_reasoning) {
      Split sp = split_reasoning(text, job.reasoning_open, mk);
      reasoning = sp.reasoning;
      content = sp.content;
      held += sp.held;
    } else {
      content = text;
    }
    size_t cut = std::string::npos;
    for (const auto& s : job.stops) {
      const size_t at = content.find(s);
      if (at != std::string::npos && at < cut) cut = at, stop_seq = s;
    }
    if (cut != std::string::npos) {
      content.resize(cut);
      stop_hit = true;
    }
    if (tools) {
      const size_t at = content.find(mk.tool_open);
      if (at != std::string::npos) {
        safe = std::min(safe, at);  // the rest is parsed at the end
        return;
      }
    }
    if (stop_hit || final_view) safe = content.size();
    else safe = content.size() > hold ? content.size() - hold : 0;
    // Never split a UTF-8 sequence.
    while (safe > 0 && safe < content.size() && (static_cast<unsigned char>(content[safe]) & 0xC0) == 0x80) --safe;
  };
  // A forced call after the reasoning. Phase 1 (the reasoning) stopped
  // at </think> (or at the end of the turn): phase 2 continues the same
  // prompt, phase 1's tokens, </think> and the call's opening as a sequence
  // of its own (its prompt checkpoint makes the shared part cheap).
  bool phase2 = false, no_room = false;
  std::vector<int32_t> gen1;  // phase 1's tokens
  int p1_prompt = 0, p1_cached = 0, p1_completion = 0, forced_n = 0;
  json::Value p1_timings;
  auto start_phase2 = [&]() -> bool {
    GenRequest g = job.gen;
    g.group_out.clear();
    g.group_seeds.clear();
    if (auto it = std::find(g.stop_ids.rbegin(), g.stop_ids.rend(), job.think_end); it != g.stop_ids.rend())
      g.stop_ids.erase(std::next(it).base());  // phase 1's own stop
    std::vector<int32_t> extra = gen1;
    extra.push_back(job.think_end);
    const std::vector<int32_t> open = tok.encode(job.force_after_think);
    extra.insert(extra.end(), open.begin(), open.end());
    if (!g.positions.empty()) {  // M-RoPE: text after the prompt's last (text) row
      const int32_t p = g.positions[g.positions.size() - 3];
      for (size_t j = 0; j < extra.size(); ++j)
        for (int d = 0; d < 3; ++d) g.positions.push_back(p + 1 + int32_t(j));
    }
    g.prompt.insert(g.prompt.end(), extra.begin(), extra.end());
    g.max_tokens = std::max(1, job.gen.max_tokens - stream->completion_tokens - int(1 + open.size()));
    if (g.has_seed) g.seed = Engine::choice_seed(job.gen.seed, choice);
    std::shared_ptr<GenStream> s2;
    try {
      s2 = engine.generate(std::move(g));
    } catch (const RequestError&) {  // no room left in the context window: the reply ends here
      return false;
    }
    p1_prompt = stream->prompt_tokens;
    p1_cached = stream->cached_tokens;
    p1_completion = stream->completion_tokens;
    p1_timings = stream->timings;
    forced_n = int(1 + open.size());
    raw += tok.piece(job.think_end) + job.force_after_think;
    stream = std::move(s2);
    phase2 = true;
    return true;
  };
  bool client_gone = false;
  while (true) {
    got.clear();
    // With a stop flag (the other choices of the request), poll it.
    const bool more = stream->next(got, stop ? 100 : -1, want_lp ? &lps : nullptr);
    if (stop && !client_gone && stop->load()) {
      client_gone = true;
      stream->cancel();
    }
    for (int32_t t : got) {
      raw += tok.piece(t);
      tok_end.push_back(raw.size());
    }
    if (!job.force_after_think.empty() && !phase2) gen1.insert(gen1.end(), got.begin(), got.end());
    if (!got.empty()) {
      std::string reasoning, content;
      size_t safe = SIZE_MAX, held = 0;
      view(false, reasoning, content, safe, held);
      ChatEvent ev;
      if (reasoning.size() > emitted_r) ev.reasoning = reasoning.substr(emitted_r);
      if (safe != SIZE_MAX && safe > emitted_c) ev.content = content.substr(emitted_c, safe - emitted_c);
      if (!ev.reasoning.empty() || !ev.content.empty()) {
        emitted_r = std::max(emitted_r, reasoning.size());
        emitted_c += ev.content.size();
        if (want_lp && job.stream) {  // the records of the tokens this event completes
          held += content.size() > emitted_c ? content.size() - emitted_c : 0;
          const size_t upto = raw.size() > held ? raw.size() - held : 0;
          size_t k = lp_sent;
          while (k < lps.size() && k < tok_end.size() && tok_end[k] <= upto) ++k;
          ev.logprobs.assign(lps.begin() + long(lp_sent), lps.begin() + long(k));
          lp_sent = k;
        }
        if (!on_event(ev)) {
          client_gone = true;
          stream->cancel();
        }
      }
      if (stop_hit) stream->cancel();
    }
    if (!more) {
      if (!job.force_after_think.empty() && !phase2 && !client_gone && stream->finish_reason == "stop") {
        if (start_phase2()) continue;
        no_room = true;
      }
      break;
    }
    if (client_gone) {
      // Drain until the scheduler retires the sequence.
      continue;
    }
  }
  // Final view: everything, tool calls parsed.
  std::string reasoning, content;
  size_t safe = SIZE_MAX, held = 0;
  view(true, reasoning, content, safe, held);
  ChatEvent last;
  last.final = true;
  if (tools) {
    json::Value calls;
    const std::string before = mk.gemma ? parse_gemma_tool_calls(content, calls) : parse_tool_calls(content, job.tools, calls);
    if (!calls.is_null()) {
      content = before;
      last.tool_calls = calls;
    }
  }
  if (reasoning.size() > emitted_r) last.reasoning = reasoning.substr(emitted_r);
  if (content.size() > emitted_c) last.content = content.substr(emitted_c);
  std::string reason = stream->finish_reason;
  if (stop_hit) reason = "stop";
  else if (reason == "cancelled" && !client_gone) reason = "stop";
  if (no_room) reason = "length";  // the forced call did not fit the context window
  if (!last.tool_calls.is_null()) reason = "tool_calls";
  if (reason == "error") throw RequestError(503, "generation failed: " + stream->error);
  last.finish_reason = reason;
  if (stop_hit && reason == "stop") last.stop_sequence = stop_seq;
  // A forced call's two phases are one reply: the client's prompt, and
  // as completion the reasoning, </think> with the call's opening, and the call.
  const int prompt_tokens = phase2 ? p1_prompt : stream->prompt_tokens;
  const int completion_tokens = phase2 ? p1_completion + forced_n + stream->completion_tokens : stream->completion_tokens;
  json::Value usage = json::Value::object();
  usage.set("prompt_tokens", prompt_tokens);
  usage.set("completion_tokens", completion_tokens);
  usage.set("total_tokens", prompt_tokens + completion_tokens);
  json::Value details = json::Value::object();
  details.set("cached_tokens", phase2 ? p1_cached : stream->cached_tokens);
  usage.set("prompt_tokens_details", std::move(details));
  last.usage = std::move(usage);
  last.timings = stream->timings;
  if (phase2) {  // the reasoning's timings, plus the call's phase
    json::Value t = p1_timings.is_object() ? p1_timings : json::Value::object();
    const json::Value& t2 = stream->timings;
    if (t2.is_object()) {
      t.set("tool_call_ms", t2.get_double("prefill_ms", 0) + t2.get_double("decode_ms", 0));
      t.set("passes", t.get_int("passes", 0) + t2.get_int("passes", 0));
      t.set("gpu_ms", t.get_double("gpu_ms", 0) + t2.get_double("gpu_ms", 0));
    }
    last.timings = std::move(t);
  }
  if (want_lp) last.logprobs.assign(lps.begin() + long(std::min(lp_sent, lps.size())), lps.end());
  if (!client_gone) on_event(last);
}

// ---------------------------------------------------------------- choices
void run_choices(Engine& engine, const ChatJob& job, const std::function<bool(int, const ChatEvent&)>& on_event) {
  const int n = std::max(1, job.n);
  if (n == 1) {
    run_chat(engine, job, [&](const ChatEvent& ev) { return on_event(0, ev); });
    return;
  }
  // One thread per choice reads its stream; their events reach the
  // caller's thread through a queue (responses are written from the
  // connection's own thread). Stopping cancels every choice's stream.
  std::vector<std::shared_ptr<GenStream>> streams = engine.generate_n(job.gen, n);
  struct Item {
    int index;
    ChatEvent ev;
  };
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Item> queue;
  std::atomic<bool> stop{false};
  std::exception_ptr error;
  int running = 0;
  auto halt = [&] {
    stop = true;
    for (auto& st : streams) st->cancel();
  };
  auto work = [&](int i) {
    try {
      run_chat_stream(engine, job, streams[size_t(i)], [&](const ChatEvent& ev) {
        {
          std::lock_guard<std::mutex> lk(mu);
          queue.push_back({i, ev});
        }
        cv.notify_one();
        return !stop.load();
      }, &stop, i);
    } catch (...) {
      std::lock_guard<std::mutex> lk(mu);
      if (!error) error = std::current_exception();
    }
    {
      std::lock_guard<std::mutex> lk(mu);
      --running;
    }
    cv.notify_one();
  };
  std::vector<std::thread> workers;
  try {
    for (int i = 0; i < n; ++i) {
      {
        std::lock_guard<std::mutex> lk(mu);
        ++running;
      }
      try {
        workers.emplace_back(work, i);
      } catch (...) {
        std::lock_guard<std::mutex> lk(mu);
        --running;
        throw;
      }
    }
  } catch (const std::exception& e) {
    halt();
    for (auto& w : workers) w.join();
    throw RequestError(503, std::string("cannot run the choices: ") + e.what());
  }
  std::unique_lock<std::mutex> lk(mu);
  for (;;) {
    cv.wait(lk, [&] { return !queue.empty() || running == 0 || (error && !stop); });
    if (error && !stop) {  // a failed choice fails the request: the others stop
      lk.unlock();
      halt();
      lk.lock();
      continue;
    }
    if (queue.empty()) break;
    Item it = std::move(queue.front());
    queue.pop_front();
    lk.unlock();
    if (!stop && !on_event(it.index, it.ev)) halt();
    lk.lock();
  }
  lk.unlock();
  for (auto& w : workers) w.join();
  if (error) std::rethrow_exception(error);
}

json::Value respond(Engine& engine, const ChatJob& job, const std::function<bool()>& client_gone) {
  std::vector<ChoiceResult> choices(size_t(std::max(1, job.n)));
  run_choices(engine, job, [&](int i, const ChatEvent& ev) {
    ChoiceResult& c = choices[size_t(i)];
    c.content += ev.content;
    c.reasoning += ev.reasoning;
    if (ev.final) c.last = ev;
    return !(client_gone && client_gone());
  });
  return job.chat ? chat_response(engine, job, choices) : completion_response(engine, job, choices);
}

json::Value choices_usage(const std::vector<json::Value>& usages) {
  if (usages.empty()) return json::Value();
  json::Value u = usages[0];
  if (usages.size() == 1 || !u.is_object()) return u;
  int64_t completion = 0;
  for (const json::Value& x : usages) completion += x.is_object() ? x.get_int("completion_tokens", 0) : 0;
  u.set("completion_tokens", completion);
  u.set("total_tokens", u.get_int("prompt_tokens", 0) + completion);
  return u;
}

// ---------------------------------------------------------------- bodies
// A choice's "logprobs": null without them, else OpenAI's chat object
// ({"content": [...]}) or the legacy completions one.
static json::Value choice_logprobs(const Engine& engine, const ChatJob& job, const std::vector<TokenLogprob>* lps,
                                   int64_t* text_offset) {
  if (job.gen.logprobs < 0 || !lps) return json::Value();
  if (!job.chat) return legacy_logprobs_json(engine.tokenizer(), *lps, text_offset);
  json::Value lp = json::Value::object();
  lp.set("content", logprobs_json(engine.tokenizer(), *lps));
  return lp;
}

// The top-level fields of a response: usage over the choices (the prompt
// once) and the first choice's timings, whose prefill served them all.
static void response_tail(json::Value& out, const std::vector<ChoiceResult>& choices) {
  std::vector<json::Value> usages;
  for (const ChoiceResult& c : choices) usages.push_back(c.last.usage);
  out.set("usage", choices_usage(usages));
  out.set("timings", choices.empty() ? json::Value() : choices[0].last.timings);
}

json::Value chat_response(const Engine& engine, const ChatJob& job, const std::vector<ChoiceResult>& results) {
  json::Value choices = json::Value::array();
  for (size_t i = 0; i < results.size(); ++i) {
    const ChoiceResult& r = results[i];
    json::Value msg = json::Value::object();
    msg.set("role", "assistant");
    msg.set("content", r.last.tool_calls.is_null() || !r.content.empty() ? json::Value(r.content) : json::Value());
    if (!r.reasoning.empty()) msg.set("reasoning_content", r.reasoning);
    if (!r.last.tool_calls.is_null()) msg.set("tool_calls", r.last.tool_calls);
    json::Value choice = json::Value::object();
    choice.set("index", int64_t(i));
    choice.set("message", std::move(msg));
    choice.set("logprobs", choice_logprobs(engine, job, &r.last.logprobs, nullptr));
    choice.set("finish_reason", r.last.finish_reason);
    choices.push_back(std::move(choice));
  }
  json::Value out = json::Value::object();
  out.set("id", job.id);
  out.set("object", "chat.completion");
  out.set("created", job.created);
  out.set("model", engine.model_name());
  out.set("choices", std::move(choices));
  response_tail(out, results);
  if (json::Value r = structured_report(job); !r.is_null()) out.set("structured_output", std::move(r));
  return out;
}

json::Value completion_response(const Engine& engine, const ChatJob& job, const std::vector<ChoiceResult>& results) {
  json::Value choices = json::Value::array();
  for (size_t i = 0; i < results.size(); ++i) {
    const ChoiceResult& r = results[i];
    json::Value choice = json::Value::object();
    choice.set("index", int64_t(i));
    choice.set("text", job.echo ? job.prompt_text + r.content : r.content);
    int64_t offset = 0;
    choice.set("logprobs", choice_logprobs(engine, job, &r.last.logprobs, &offset));
    choice.set("finish_reason", r.last.finish_reason);
    choices.push_back(std::move(choice));
  }
  json::Value out = json::Value::object();
  out.set("id", job.id);
  out.set("object", "text_completion");
  out.set("created", job.created);
  out.set("model", engine.model_name());
  out.set("choices", std::move(choices));
  response_tail(out, results);
  if (json::Value r = structured_report(job); !r.is_null()) out.set("structured_output", std::move(r));
  return out;
}

json::Value chat_response(const Engine& engine, const ChatJob& job, const std::string& content,
                          const std::string& reasoning, const ChatEvent& last) {
  return chat_response(engine, job, std::vector<ChoiceResult>{ChoiceResult{content, reasoning, last}});
}

json::Value completion_response(const Engine& engine, const ChatJob& job, const std::string& text, const ChatEvent& last) {
  return completion_response(engine, job, std::vector<ChoiceResult>{ChoiceResult{text, "", last}});
}

json::Value chat_chunk(const Engine& engine, const ChatJob& job, const json::Value& delta, const json::Value& finish,
                       bool with_choices, const std::vector<TokenLogprob>* lps, int64_t* text_offset, int index) {
  json::Value out = json::Value::object();
  out.set("id", job.id);
  out.set("object", job.chat ? "chat.completion.chunk" : "text_completion");
  out.set("created", job.created);
  out.set("model", engine.model_name());
  json::Value choices = json::Value::array();
  if (with_choices) {
    json::Value c = json::Value::object();
    c.set("index", index);
    if (job.chat) c.set("delta", delta);
    else c.set("text", delta.get_string("content", ""));
    // The records of this chunk's tokens (null on chunks without any).
    c.set("logprobs", lps && !lps->empty() ? choice_logprobs(engine, job, lps, text_offset) : json::Value());
    c.set("finish_reason", finish);
    choices.push_back(std::move(c));
  }
  out.set("choices", std::move(choices));
  if (!finish.is_null() && !job.unenforced.empty()) out.set("structured_output", structured_report(job));
  return out;
}

}  // namespace espetero
