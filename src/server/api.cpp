// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// HTTP routes: OpenAI-compatible chat/completions/embeddings/models/files/
// batches, the JEVfire decisions API, tokenizer utilities and health/stats.
#include "api.h"

#include <chrono>

#include "../engine/chat.h"
#include "../engine/embed.h"
#include "../tok/tokenizer.h"
#include "ollama.h"
#include "version.h"
#include "webui.h"

namespace espetero {

namespace {

json::Value error_body(const std::string& msg, const char* type) {
  json::Value e = json::Value::object();
  e.set("message", msg);
  e.set("type", type);
  json::Value o = json::Value::object();
  o.set("error", std::move(e));
  return o;
}

void send_error(http::Response& res, int status, const std::string& msg) {
  res.send_json(status, error_body(msg, status >= 500 ? "server_error" : "invalid_request_error"));
}

json::Value parse_body(const http::Request& req) {
  try {
    return json::parse(req.body);
  } catch (const std::exception& e) {
    throw RequestError(400, std::string("invalid JSON body: ") + e.what());
  }
}

// Wraps a handler: RequestError -> its status, anything else -> 500. A stream
// that already started keeps its 200: the error goes out as an event,
// {"error": {...}}, then [DONE], as OpenAI-compatible streams end
// (send_error on the started response would end the stream with neither, and
// a generation that failed mid-stream would look like a cut connection).
http::Handler guarded(std::function<void(const http::Request&, http::Response&)> f) {
  return [f = std::move(f)](const http::Request& req, http::Response& res) {
    auto fail = [&res](int status, const std::string& msg) {
      if (!res.started()) {
        send_error(res, status, msg);
        return;
      }
      if (res.sse(json::dump(error_body(msg, status >= 500 ? "server_error" : "invalid_request_error"))))
        res.sse("[DONE]");
    };
    try {
      f(req, res);
    } catch (const RequestError& e) {
      fail(e.status, e.what());
    } catch (const std::exception& e) {
      fail(500, e.what());
    }
  };
}

// Minimal multipart/form-data parser (file uploads): name -> (filename, bytes).
std::map<std::string, std::pair<std::string, std::string>> parse_multipart(const http::Request& req) {
  std::map<std::string, std::pair<std::string, std::string>> out;
  const std::string* ct = req.header("content-type");
  if (!ct) throw RequestError(400, "multipart upload without content-type");
  const size_t b = ct->find("boundary=");
  if (b == std::string::npos) throw RequestError(400, "multipart boundary missing");
  std::string boundary = ct->substr(b + 9);
  if (const size_t semi = boundary.find(';'); semi != std::string::npos) boundary.resize(semi);
  if (boundary.size() >= 2 && boundary.front() == '"') boundary = boundary.substr(1, boundary.size() - 2);
  const std::string delim = "--" + boundary;
  const std::string& body = req.body;
  size_t pos = body.find(delim);
  while (pos != std::string::npos) {
    pos += delim.size();
    if (body.compare(pos, 2, "--") == 0) break;
    if (body.compare(pos, 2, "\r\n") == 0) pos += 2;
    const size_t hend = body.find("\r\n\r\n", pos);
    if (hend == std::string::npos) break;
    const std::string headers = body.substr(pos, hend - pos);
    const size_t data0 = hend + 4;
    const size_t next = body.find("\r\n" + delim, data0);
    if (next == std::string::npos) break;
    std::string name, filename;
    auto attr = [&](const char* key) {
      const std::string k = std::string(key) + "=\"";
      const size_t a = headers.find(k);
      if (a == std::string::npos) return std::string();
      const size_t e = headers.find('"', a + k.size());
      return headers.substr(a + k.size(), e == std::string::npos ? std::string::npos : e - a - k.size());
    };
    name = attr("name");
    filename = attr("filename");
    out[name] = {filename, body.substr(data0, next - data0)};
    pos = next + 2;
  }
  return out;
}

}  // namespace

void install_loading_routes(http::Server& server, bool webui) {
  install_webui(server, webui);  // before the gate: served while the model loads
  server.gate([](const http::Request& req, http::Response& res) {
    const std::string msg = "the model is loading: retry when GET /health answers 200";
    json::Value o = json::Value::object();
    o.set("status", "loading");
    if (req.path.rfind("/api/", 0) == 0) {  // Ollama clients read a string
      o.set("error", msg);
    } else {
      json::Value e = json::Value::object();
      e.set("message", msg);
      e.set("type", "unavailable_error");
      e.set("code", 503);
      o.set("error", std::move(e));
    }
    res.set_header("Retry-After", "5");
    res.send_json(503, o);
  });
}

void install_routes(http::Server& server, Engine& engine, BatchService* batches, const json::Value* config) {
  if (config) {
    // The resolved configuration: preset, explicit settings and their
    // sources, and the effective values.
    server.route("GET", "/v1/config", guarded([config](const http::Request&, http::Response& res) { res.send_json(200, *config); }));
  }
  server.route("GET", "/health", guarded([&engine](const http::Request&, http::Response& res) {
    const ModelInfo& I = engine.info();
    json::Value o = json::Value::object();
    o.set("status", "ok");
    o.set("backend", "metal");
    o.set("engine", kProduct);
    o.set("version", kVersion);
    o.set("model", engine.model_name());
    o.set("device", I.device);
    o.set("max_context_tokens", I.max_ctx);
    o.set("checkpoint_slots", I.slots);
    o.set("images", I.vision);
    o.set("mtp", I.mtp);
    o.set("chat", engine.chat_template() != nullptr);
    o.set("stats", engine.stats());
    res.send_json(200, o);
  }));

  server.route("GET", "/v1/profile", guarded([&engine](const http::Request&, http::Response& res) {
    res.send(200, "application/json", engine.profile_json());
  }));
  server.route("GET", "/v1/models", guarded([&engine](const http::Request&, http::Response& res) {
    json::Value m = json::Value::object();
    m.set("id", engine.model_name());
    m.set("object", "model");
    m.set("created", 0);
    m.set("owned_by", kProduct);
    m.set("info", engine.info().to_json());
    json::Value data = json::Value::array();
    data.push_back(std::move(m));
    json::Value o = json::Value::object();
    o.set("object", "list");
    o.set("data", std::move(data));
    res.send_json(200, o);
  }));

  server.route("GET", "/v1/stats", guarded([&engine](const http::Request&, http::Response& res) {
    res.send_json(200, engine.stats());
  }));

  // Decisions: the JEVfire contract, or for Clef releases the Jev /
  // SystemOne body answered by the joint schema head (also at /v1/systemone).
  auto decisions = guarded([&engine](const http::Request& req, http::Response& res) {
    try {
      const json::Value body = parse_body(req);
      // Clef models take both bodies: SystemOne (state, questions) or JEVfire (context, fields).
      res.send_json(200, engine.is_clef() && !body.get("context") ? engine.systemone(body) : engine.decide(body));
    } catch (const RequestError& e) {
      if (e.status == 422) {
        json::Value o = json::Value::object();
        o.set("detail", e.what());
        res.send_json(422, o);
      } else if (e.status >= 500) {
        json::Value o = json::Value::object();
        o.set("detail", "Metal scoring failed; no decision was substituted");
        res.send_json(502, o);
      } else {
        throw;
      }
    }
  });
  server.route("POST", "/v1/decisions", decisions, http::Body::Json);
  server.route("POST", "/v1/systemone", decisions, http::Body::Json);

  auto generation = [&engine](bool chat) {
    return guarded([&engine, chat](const http::Request& req, http::Response& res) {
      if (engine.is_clef())
        throw RequestError(422, "this is a Clef decision model: it answers /v1/systemone and /v1/decisions, not chat");
      const json::Value body = parse_body(req);
      ChatJob job = chat ? prepare_chat(engine, body) : prepare_completion(engine, body);
      if (!job.stream) {
        res.send_json(200, respond(engine, job, [&res] { return res.client_gone(); }));
        return;
      }
      if (!res.start_stream(200, "text/event-stream")) return;
      // Per choice: its role chunk sent, its usage, and for completions
      // logprobs the next token's offset in its text.
      const size_t n = size_t(std::max(1, job.n));
      std::vector<char> first(n, 1);
      std::vector<json::Value> usages(n);
      std::vector<int64_t> text_offset(n, 0);
      run_choices(engine, job, [&](int i, const ChatEvent& ev) {
        const size_t k = size_t(i);
        if (first[k] && job.chat) {
          json::Value d = json::Value::object();
          d.set("role", "assistant");
          d.set("content", "");
          if (!res.sse(json::dump(chat_chunk(engine, job, d, json::Value(), true, nullptr, nullptr, i)))) return false;
        }
        first[k] = 0;
        json::Value d = json::Value::object();
        if (!ev.reasoning.empty()) d.set("reasoning_content", ev.reasoning);
        if (!ev.content.empty()) d.set("content", ev.content);
        if (!ev.tool_calls.is_null()) {
          json::Value calls = ev.tool_calls;
          for (size_t j = 0; j < calls.size(); ++j) calls[j].set("index", int64_t(j));
          d.set("tool_calls", calls);
        }
        // The event's logprob records go with its text, else with the finish chunk.
        const std::vector<TokenLogprob>* lps = ev.logprobs.empty() ? nullptr : &ev.logprobs;
        if (d.size()) {
          if (!res.sse(json::dump(chat_chunk(engine, job, d, json::Value(), true, lps, &text_offset[k], i)))) return false;
          lps = nullptr;
        }
        if (ev.final) {
          usages[k] = ev.usage;
          if (!res.sse(json::dump(chat_chunk(engine, job, json::Value::object(), ev.finish_reason, true, lps, &text_offset[k], i))))
            return false;
        }
        return true;
      });
      const json::Value usage = choices_usage(usages);  // the prompt once, every choice's completion tokens
      if (job.include_usage && !usage.is_null()) {
        json::Value c = chat_chunk(engine, job, json::Value(), json::Value(), false);
        c.set("usage", usage);
        res.sse(json::dump(c));
      }
      res.sse("[DONE]");
      res.end();
    });
  };
  server.route("POST", "/v1/chat/completions", generation(true), http::Body::Json);
  server.route("POST", "/v1/completions", generation(false), http::Body::Json);

  // Embeddings of the loaded model (pooled final-normed hidden states).
  server.route("POST", "/v1/embeddings", guarded([&engine](const http::Request& req, http::Response& res) {
    res.send_json(200, embeddings_response(engine, parse_body(req)));
  }), http::Body::Json);

  server.route("POST", "/v1/tokenize", guarded([&engine](const http::Request& req, http::Response& res) {
    const json::Value body = parse_body(req);
    std::vector<int32_t> ids;
    if (body.get("messages")) {
      json::Value b = body;
      ChatJob job = prepare_chat(engine, b);
      ids = job.gen.prompt;
    } else {
      const std::string text = body.get_string("prompt", body.get_string("text", ""));
      engine.check_text_size(text.size(), "text");
      ids = engine.tokenizer().encode(text, body.get_bool("parse_special", true));
    }
    json::Value arr = json::Value::array();
    for (int32_t t : ids) arr.push_back(t);
    json::Value o = json::Value::object();
    o.set("tokens", std::move(arr));
    o.set("count", int64_t(ids.size()));
    res.send_json(200, o);
  }), http::Body::Json);

  server.route("POST", "/v1/detokenize", guarded([&engine](const http::Request& req, http::Response& res) {
    const json::Value body = parse_body(req);
    std::vector<int32_t> ids;
    if (const json::Value* t = body.get("tokens"); t && t->is_array()) {
      if (t->size() > size_t(engine.info().max_ctx))
        throw RequestError(413, "tokens: at most " + std::to_string(engine.info().max_ctx) +
                                    " ids (the model context window)");
      for (const auto& x : t->as_array()) {
        int64_t v = 0;
        try {
          v = x.as_int();
        } catch (const json::Error&) {
          throw RequestError(422, "tokens: integers only");
        }
        ids.push_back(v < 0 || v > INT32_MAX ? -1 : int32_t(v));  // out of range: skipped by decode
      }
    }
    json::Value o = json::Value::object();
    o.set("text", engine.tokenizer().decode(ids));
    res.send_json(200, o);
  }), http::Body::Json);

  install_compat_routes(server, engine);
  install_ollama_routes(server, engine);  // /api/*

  server.route("POST", "/v1/cache/clear", guarded([&engine](const http::Request&, http::Response& res) {
    engine.clear_cache();
    json::Value o = json::Value::object();
    o.set("cleared", true);
    res.send_json(200, o);
  }), http::Body::Json);

  if (!batches) return;
  BatchService& B = *batches;
  server.route("POST", "/v1/files", guarded([&B](const http::Request& req, http::Response& res) {
    auto parts = parse_multipart(req);
    auto file = parts.find("file");
    if (file == parts.end()) throw RequestError(422, "file: required (multipart field 'file')");
    const std::string purpose = parts.count("purpose") ? parts["purpose"].second : "batch";
    res.send_json(200, B.upload(file->second.first, purpose, file->second.second));
  }), http::Body::Multipart);
  server.route("GET", "/v1/files", guarded([&B](const http::Request&, http::Response& res) { res.send_json(200, B.list_files()); }));
  server.route_prefix("GET", "/v1/files/", guarded([&B](const http::Request& req, http::Response& res) {
    std::string rest = req.path.substr(std::string("/v1/files/").size());
    const std::string suffix = "/content";
    if (rest.size() > suffix.size() && rest.compare(rest.size() - suffix.size(), suffix.size(), suffix) == 0) {
      rest.resize(rest.size() - suffix.size());
      res.send(200, "application/jsonl", B.file_content(rest));
      return;
    }
    res.send_json(200, B.file_info(rest));
  }));
  server.route_prefix("DELETE", "/v1/files/", guarded([&B](const http::Request& req, http::Response& res) {
    res.send_json(200, B.delete_file(req.path.substr(std::string("/v1/files/").size())));
  }));
  server.route("POST", "/v1/batches", guarded([&B](const http::Request& req, http::Response& res) {
    res.send_json(200, B.create(parse_body(req)));
  }), http::Body::Json);
  server.route("GET", "/v1/batches", guarded([&B](const http::Request& req, http::Response& res) {
    int limit = 20;
    const std::string l = req.query_param("limit");
    if (!l.empty()) limit = std::max(1, std::min(100, std::atoi(l.c_str())));
    res.send_json(200, B.list(limit, req.query_param("after")));
  }));
  server.route_prefix("GET", "/v1/batches/", guarded([&B](const http::Request& req, http::Response& res) {
    res.send_json(200, B.get(req.path.substr(std::string("/v1/batches/").size())));
  }));
  server.route_prefix("POST", "/v1/batches/", guarded([&B](const http::Request& req, http::Response& res) {
    std::string rest = req.path.substr(std::string("/v1/batches/").size());
    const std::string suffix = "/cancel";
    if (rest.size() <= suffix.size() || rest.compare(rest.size() - suffix.size(), suffix.size(), suffix) != 0)
      throw RequestError(404, "unknown route");
    rest.resize(rest.size() - suffix.size());
    res.send_json(200, B.cancel(rest));
  }), http::Body::Json);
}

}  // namespace espetero
