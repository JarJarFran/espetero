// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Ollama's routes over the chat job; the request translation is in
// ollama.cpp. One model per server: any model name a request gives is
// accepted, answered by the loaded model and echoed back, as Ollama echoes
// the name it was asked for.
#include <CommonCrypto/CommonDigest.h>
#include <dirent.h>
#include <sys/stat.h>

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

#include "../engine/chat.h"
#include "../engine/embed.h"
#include "../tok/tokenizer.h"
#include "api.h"
#include "ollama.h"
#include "version.h"

namespace espetero {

namespace {

using Clock = std::chrono::steady_clock;

// The Ollama API level these routes follow, for clients that check
// GET /api/version against a minimum (the server's own version is at /health).
constexpr const char* kOllamaApiVersion = "0.12.0";

json::Value obj() { return json::Value::object(); }
json::Value arr() { return json::Value::array(); }

// Errors in Ollama's shape, {"error": "..."}; in a stream, as its last line.
http::Handler handler(std::function<void(const http::Request&, http::Response&)> f) {
  return [f = std::move(f)](const http::Request& req, http::Response& res) {
    auto fail = [&](int status, const std::string& msg) {
      json::Value o = obj();
      o.set("error", msg);
      if (!res.started()) res.send_json(status, o);
      else res.write(json::dump(o) + "\n");
    };
    try {
      f(req, res);
    } catch (const RequestError& e) {
      fail(e.status, e.what());
    } catch (const json::Error& e) {
      fail(400, std::string("invalid JSON body: ") + e.what());
    } catch (const std::exception& e) {
      fail(500, e.what());
    }
  };
}

json::Value parse_body(const http::Request& req) {
  if (req.body.empty()) throw RequestError(400, "missing request body");
  json::Value v = json::parse(req.body);  // json::Error -> 400
  if (!v.is_object()) throw RequestError(400, "request body must be a JSON object");
  return v;
}

int64_t ns(double ms) { return int64_t(std::llround(ms * 1e6)); }
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// ---------------------------------------------------------------- the model

std::string sha256_hex(const std::string& s) {
  unsigned char d[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(s.data(), CC_LONG(s.size()), d);
  static const char* h = "0123456789abcdef";
  std::string out;
  for (unsigned char c : d) out += h[c >> 4], out += h[c & 15];
  return out;
}

// What Ollama clients are told about the loaded model, worked out once.
struct Card {
  std::string name;          // Ollama-style (name:tag)
  std::string arch;          // GGUF-style architecture: qwen35, qwen35moe, ...
  std::string format, quantization, digest, modified_at, template_src;
  int64_t disk_bytes = 0;
  bool tools = false;
  ollama::ModelTraits traits;
};

int64_t disk_size(const std::string& path) {
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) return 0;
  if (!S_ISDIR(st.st_mode)) return int64_t(st.st_size);
  int64_t total = 0;
  if (DIR* d = ::opendir(path.c_str())) {
    while (const dirent* e = ::readdir(d)) {
      const std::string n = e->d_name;
      if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) {
        struct stat f;
        if (::stat((path + "/" + n).c_str(), &f) == 0) total += int64_t(f.st_size);
      }
    }
    ::closedir(d);
  }
  return total;
}

Card make_card(const Engine& engine) {
  Card c;
  const ModelInfo& I = engine.info();
  const std::string& path = engine.options().model_path;
  c.name = ollama::tagged(engine.model_name());
  c.format = I.format == "gguf" ? "gguf" : "safetensors";
  c.template_src = engine.tokenizer().chat_template();
  c.tools = c.template_src.find("tools") != std::string::npos;
  c.traits.max_ctx = I.max_ctx;
  c.traits.thinking = c.template_src.find("enable_thinking") != std::string::npos ||
                      c.template_src.find("<think>") != std::string::npos;
  c.disk_bytes = disk_size(path);
  c.digest = sha256_hex(path + "\n" + engine.options().model_revision + "\n" + engine.model_name());
  struct stat st;
  c.modified_at = ollama::timestamp(::stat(path.c_str(), &st) == 0 ? std::chrono::system_clock::from_time_t(st.st_mtime)
                                                                    : std::chrono::system_clock::now());
  c.arch = "qwen35";
  if (I.format == "gguf") {
    // The quantization from the file name: ...-Q6_K.gguf, ...-UD-Q4_K_XL.gguf, ...-BF16.gguf.
    std::string stem = path.substr(path.rfind('/') + 1);
    stem = stem.substr(0, stem.rfind('.'));
    for (size_t at = stem.size(); at != std::string::npos && at > 0;) {
      const size_t dash = stem.rfind('-', at - 1);
      std::string part = stem.substr(dash == std::string::npos ? 0 : dash + 1, at - (dash == std::string::npos ? 0 : dash + 1));
      for (char& ch : part) ch = char(std::toupper(static_cast<unsigned char>(ch)));
      if (part.rfind("Q", 0) == 0 || part.rfind("IQ", 0) == 0 || part == "F16" || part == "BF16" || part == "F32") {
        c.quantization = part;
        break;
      }
      at = dash;
    }
    if (I.extra.is_object() && I.extra.get_string("architecture", "").size()) c.arch = I.extra.get_string("architecture", "");
  } else {
    // config.json: model_type (qwen3_5, qwen3_5_moe, ...) and the quantization bits.
    std::ifstream f(path + "/config.json");
    std::stringstream ss;
    ss << f.rdbuf();
    try {
      const json::Value cfg = json::parse(ss.str());
      const json::Value* tc = cfg.get("text_config");
      std::string type = tc && tc->is_object() ? tc->get_string("model_type", "") : std::string();
      if (type.empty()) type = cfg.get_string("model_type", "");
      for (const char* suffix : {"_text", "_mtp"})
        if (type.size() > std::strlen(suffix) && type.compare(type.size() - std::strlen(suffix), std::string::npos, suffix) == 0)
          type.resize(type.size() - std::strlen(suffix));
      if (type == "qwen3_5") c.arch = "qwen35";
      else if (type == "qwen3_5_moe") c.arch = "qwen35moe";
      else if (!type.empty()) c.arch = type;
      if (const json::Value* q = cfg.get("quantization"); q && q->is_object() && q->get("bits"))
        c.quantization = std::to_string(q->get_int("bits", 0)) + "-bit";
    } catch (...) {  // no readable config: the defaults above
    }
  }
  return c;
}

json::Value details(const Card& c) {
  json::Value d = obj();
  d.set("parent_model", "");
  d.set("format", c.format);
  d.set("family", c.arch);
  json::Value fams = arr();
  fams.push_back(c.arch);
  d.set("families", std::move(fams));
  d.set("parameter_size", "");
  d.set("quantization_level", c.quantization);
  return d;
}

std::string requested(const json::Value& b, const Card& c) {
  std::string name = b.get_string("model", "");
  if (name.empty()) name = b.get_string("name", "");  // /api/show's older field
  return name.empty() ? c.name : name;
}

// ---------------------------------------------------------------- generation

// Ollama's logprobs entries from the chat job's records.
json::Value logprobs_json(const Tokenizer& tok, const std::vector<TokenLogprob>& lps) {
  auto entry = [&](int32_t id, float lp) {
    const std::string& p = tok.piece(id);
    std::string text;
    if (json::utf8_valid(p)) {
      text = p;
    } else {
      for (size_t i = 0; i < p.size();) json::utf8_append(text, json::utf8_next(p, i));
    }
    json::Value e = obj();
    e.set("token", text);
    e.set("logprob", std::isfinite(lp) ? double(lp) : -9999.0);
    json::Value b = arr();
    for (unsigned char ch : p) b.push_back(int64_t(ch));
    e.set("bytes", std::move(b));
    return e;
  };
  json::Value out = arr();
  for (const TokenLogprob& t : lps) {
    json::Value e = entry(t.token, t.logprob);
    if (!t.top.empty()) {
      json::Value top = arr();
      for (const auto& [id, lp] : t.top) top.push_back(entry(id, lp));
      e.set("top_logprobs", std::move(top));
    }
    out.push_back(std::move(e));
  }
  return out;
}

// num_ctx: the prompt and the reply within that many tokens (the server's own
// window, --max-ctx, still applies when it is smaller). Nothing is cut: a
// prompt that does not fit is an error.
void apply_num_ctx(ChatJob& job, const ollama::Extras& x) {
  if (x.num_ctx <= 0) return;
  const int64_t prompt = int64_t(job.gen.prompt.size());
  if (prompt >= x.num_ctx)
    throw RequestError(422, "the prompt (" + std::to_string(prompt) + " tokens) does not fit options.num_ctx (" +
                                std::to_string(x.num_ctx) + "): raise num_ctx or shorten it (prompts are not truncated)");
  job.gen.max_tokens = int(std::min<int64_t>(job.gen.max_tokens, x.num_ctx - prompt));
}

void metrics(json::Value& o, const ChatEvent& last, Clock::time_point t0) {
  const json::Value& t = last.timings;
  const json::Value& u = last.usage;
  o.set("total_duration", ns(ms_since(t0)));
  o.set("load_duration", 0);
  o.set("prompt_eval_count", u.is_object() ? u.get_int("prompt_tokens", 0) : 0);
  o.set("prompt_eval_duration", ns(t.is_object() ? t.get_double("prefill_ms", 0) : 0));
  o.set("eval_count", u.is_object() ? u.get_int("completion_tokens", 0) : 0);
  o.set("eval_duration", ns(t.is_object() ? t.get_double("decode_ms", 0) : 0));
}

// Runs a prepared job and answers in Ollama's shape: /api/chat objects with a
// "message" (`chat`), /api/generate objects with a "response".
void answer(Engine& engine, const ChatJob& job, const ollama::Extras& x, const std::string& name, bool chat,
            http::Response& res, Clock::time_point t0) {
  const Tokenizer& tok = engine.tokenizer();
  auto head = [&] {
    json::Value o = obj();
    o.set("model", name);
    o.set("created_at", ollama::timestamp(std::chrono::system_clock::now()));
    return o;
  };
  // One object: the text, the thinking, the tool calls (final) and logprobs.
  auto piece = [&](const std::string& content, const std::string& thinking, const json::Value& calls,
                   const std::vector<TokenLogprob>* lps) {
    json::Value o = head();
    if (chat) {
      json::Value m = obj();
      m.set("role", "assistant");
      m.set("content", content);
      if (!thinking.empty()) m.set("thinking", thinking);
      if (calls.is_array() && calls.size()) {
        json::Value tc = arr();
        for (size_t i = 0; i < calls.size(); ++i) tc.push_back(ollama::tool_call(calls[i], int(i)));
        m.set("tool_calls", std::move(tc));
      }
      o.set("message", std::move(m));
    } else {
      o.set("response", content);
      if (!thinking.empty()) o.set("thinking", thinking);
    }
    if (lps && !lps->empty()) o.set("logprobs", logprobs_json(tok, *lps));
    return o;
  };
  auto done = [&](json::Value o, const ChatEvent& last) {
    o.set("done", true);
    o.set("done_reason", ollama::done_reason(last.finish_reason));
    metrics(o, last, t0);
    return o;
  };
  if (x.stream) {
    if (!res.start_stream(200, "application/x-ndjson")) return;
    run_chat(engine, job, [&](const ChatEvent& ev) {
      const json::Value none;
      const bool calls = ev.final && ev.tool_calls.is_array() && ev.tool_calls.size();
      if (!ev.content.empty() || !ev.reasoning.empty() || !ev.logprobs.empty() || calls) {
        json::Value o = piece(ev.content, ev.reasoning, calls ? ev.tool_calls : none, &ev.logprobs);
        o.set("done", false);
        if (!res.write(json::dump(o) + "\n")) return false;
      }
      if (ev.final && !res.write(json::dump(done(piece("", "", none, nullptr), ev)) + "\n")) return false;
      return true;
    });
    return;
  }
  std::string content, thinking;
  std::vector<TokenLogprob> lps;
  ChatEvent last;
  run_chat(engine, job, [&](const ChatEvent& ev) {
    content += ev.content;
    thinking += ev.reasoning;
    lps.insert(lps.end(), ev.logprobs.begin(), ev.logprobs.end());
    if (ev.final) last = ev;
    return !res.client_gone();
  });
  if (res.client_gone()) return;
  res.send_json(200, done(piece(content, thinking, last.tool_calls, &lps), last));
}

// What an empty /api/chat or /api/generate (Ollama's way to load a model) gets.
json::Value loaded(const std::string& name, bool chat) {
  json::Value o = obj();
  o.set("model", name);
  o.set("created_at", ollama::timestamp(std::chrono::system_clock::now()));
  if (chat) {
    json::Value m = obj();
    m.set("role", "assistant");
    m.set("content", "");
    o.set("message", std::move(m));
  } else {
    o.set("response", "");
  }
  o.set("done_reason", "load");
  o.set("done", true);
  return o;
}

void check_chat_model(const Engine& engine) {
  if (engine.is_clef())
    throw RequestError(422, "this is a Clef decision model: it answers /v1/systemone and /v1/decisions, not chat");
}

}  // namespace

void install_ollama_routes(http::Server& server, Engine& engine) {
  auto card = std::make_shared<const Card>(make_card(engine));

  server.route("GET", "/api/version", handler([](const http::Request&, http::Response& res) {
    json::Value o = obj();
    o.set("version", kOllamaApiVersion);
    res.send_json(200, o);
  }));

  // The loaded model, the only one: as installed (/api/tags) and as running (/api/ps).
  auto entry = [card, &engine](bool running) {
    const Card& c = *card;
    const ModelInfo& I = engine.info();
    json::Value m = obj();
    m.set("name", c.name);
    m.set("model", c.name);
    if (!running) m.set("modified_at", c.modified_at);
    m.set("size", running ? int64_t(I.weight_bytes + I.kv_bytes + I.state_bytes) : c.disk_bytes);
    m.set("digest", c.digest);
    m.set("details", details(c));
    if (running) {
      m.set("expires_at", "2318-01-01T00:00:00.000000Z");  // never: loaded while the server runs
      m.set("size_vram", int64_t(I.weight_bytes + I.kv_bytes + I.state_bytes));
      m.set("context_length", int64_t(I.max_ctx));
    }
    json::Value list = arr();
    list.push_back(std::move(m));
    json::Value o = obj();
    o.set("models", std::move(list));
    return o;
  };
  server.route("GET", "/api/tags", handler([entry](const http::Request&, http::Response& res) { res.send_json(200, entry(false)); }));
  server.route("GET", "/api/ps", handler([entry](const http::Request&, http::Response& res) { res.send_json(200, entry(true)); }));

  server.route("POST", "/api/show", handler([card, &engine](const http::Request& req, http::Response& res) {
    const json::Value b = parse_body(req);
    const Card& c = *card;
    const ModelInfo& I = engine.info();
    // The model's sampling defaults, as Ollama's parameters (Modelfile PARAMETER lines).
    std::string params;
    const SamplingDefaults& sd = engine.sampling_defaults();
    auto num = [](double v) {
      std::ostringstream s;
      s << v;
      return s.str();
    };
    if (sd.temperature) params += "temperature " + num(*sd.temperature) + "\n";
    if (sd.top_p) params += "top_p " + num(*sd.top_p) + "\n";
    if (sd.top_k) params += "top_k " + std::to_string(*sd.top_k) + "\n";
    if (sd.min_p) params += "min_p " + num(*sd.min_p) + "\n";
    std::string modelfile = "# Served by " + std::string(kProduct) + " " + kVersion + "\nFROM " + engine.options().model_path + "\n";
    std::istringstream ps(params);
    for (std::string line; std::getline(ps, line);) modelfile += "PARAMETER " + line + "\n";
    json::Value info = obj();
    info.set("general.architecture", c.arch);
    info.set("general.name", engine.model_name());
    info.set(c.arch + ".context_length", int64_t(I.max_ctx));
    info.set(c.arch + ".embedding_length", int64_t(I.hidden));
    info.set(c.arch + ".block_count", int64_t(I.layers));
    info.set(c.arch + ".attention.head_count", int64_t(I.heads));
    info.set(c.arch + ".attention.head_count_kv", int64_t(I.kv_heads));
    info.set(c.arch + ".vocab_size", int64_t(I.vocab));
    json::Value caps = arr();
    if (!engine.is_clef()) {
      caps.push_back("completion");
      if (c.tools) caps.push_back("tools");
      if (c.traits.thinking) caps.push_back("thinking");
      if (I.vision) caps.push_back("vision");
    }
    json::Value o = obj();
    o.set("license", "");
    o.set("modelfile", modelfile);
    o.set("parameters", params);
    o.set("template", c.template_src);  // the model's Jinja chat template (Ollama's are Go templates)
    o.set("details", details(c));
    o.set("model_info", std::move(info));
    o.set("capabilities", std::move(caps));
    o.set("modified_at", c.modified_at);
    (void)requested(b, c);  // any name: the loaded model
    res.send_json(200, o);
  }));

  server.route("POST", "/api/chat", handler([card, &engine](const http::Request& req, http::Response& res) {
    const auto t0 = Clock::now();
    const json::Value b = parse_body(req);
    ollama::Extras x;
    const json::Value body = ollama::chat_body(b, card->traits, x);
    const std::string name = requested(b, *card);
    if (x.load_only) {
      res.send_json(200, loaded(name, true));
      return;
    }
    check_chat_model(engine);
    ChatJob job = prepare_chat(engine, body);
    apply_num_ctx(job, x);
    answer(engine, job, x, name, true, res, t0);
  }));

  server.route("POST", "/api/generate", handler([card, &engine](const http::Request& req, http::Response& res) {
    const auto t0 = Clock::now();
    const json::Value b = parse_body(req);
    ollama::Extras x;
    bool raw = false;
    const json::Value body = ollama::generate_body(b, card->traits, x, raw);
    const std::string name = requested(b, *card);
    if (x.load_only) {
      res.send_json(200, loaded(name, false));
      return;
    }
    check_chat_model(engine);
    ChatJob job = raw ? prepare_completion(engine, body) : prepare_chat(engine, body);
    apply_num_ctx(job, x);
    answer(engine, job, x, name, false, res, t0);
  }));

  // Embeddings: /v1/embeddings' vectors (L2-normalized, also on the legacy route).
  auto embed = [card, &engine](bool legacy) {
    return handler([card, &engine, legacy](const http::Request& req, http::Response& res) {
      const auto t0 = Clock::now();
      const json::Value b = parse_body(req);
      bool empty = false;
      const json::Value body = ollama::embed_body(b, legacy, empty);
      json::Value vecs = arr();
      int64_t tokens = 0;
      if (!empty) {
        const json::Value r = embeddings_response(engine, body);
        if (const json::Value* data = r.get("data"); data && data->is_array()) {
          std::vector<json::Value> by_index(data->size());
          for (const json::Value& d : data->as_array()) {
            const size_t i = size_t(d.get_int("index", 0));
            if (i < by_index.size()) by_index[i] = *d.get("embedding");
          }
          for (json::Value& v : by_index) vecs.push_back(std::move(v));
        }
        if (const json::Value* u = r.get("usage"); u && u->is_object()) tokens = u->get_int("prompt_tokens", 0);
      }
      json::Value o = obj();
      if (legacy) {
        o.set("embedding", vecs.size() ? vecs[0] : arr());
      } else {
        o.set("model", requested(b, *card));
        o.set("embeddings", std::move(vecs));
        o.set("total_duration", ns(ms_since(t0)));
        o.set("load_duration", 0);
        o.set("prompt_eval_count", tokens);
      }
      res.send_json(200, o);
    });
  };
  server.route("POST", "/api/embed", embed(false));
  server.route("POST", "/api/embeddings", embed(true));

  // Ollama's model management (pull, push, create, copy, delete, blobs):
  // this server runs the one model it started with.
  auto other = handler([](const http::Request& req, http::Response&) {
    throw RequestError(404, req.method + " " + req.path +
                                " is not supported: this server runs one model, the one it started with (--model); its "
                                "Ollama routes are /api/version, /api/tags, /api/ps, /api/show, /api/chat, "
                                "/api/generate, /api/embed and /api/embeddings");
  });
  for (const char* m : {"GET", "POST", "PUT", "DELETE"}) server.route_prefix(m, "/api/", other);
}

}  // namespace espetero
