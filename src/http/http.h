// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// A small HTTP/1.1 server for the OpenAI-compatible API.
//
//  * Keep-alive (HTTP/1.1 default, HTTP/1.0 opt-in), pipelined requests
//    answered in order, Content-Length and chunked request bodies,
//    Expect: 100-continue.
//  * Complete responses (Content-Length) or streams (chunked transfer
//    encoding; Server-Sent Events), each write flushed immediately.
//  * Handlers run synchronously on the connection's own thread and may block
//    for minutes (waiting for the GPU, streaming tokens). write() and
//    client_gone() tell them when the client has left so they can cancel.
//  * Size limits and timeouts for untrusted clients, optional Bearer auth and
//    CORS, OpenAI-style JSON errors, graceful stop.
//  * Without an API key, protection against web pages in the user's browser:
//    requests must be addressed to a loopback name or the bind address (DNS
//    rebinding) and come from no other web origin unless CORS is on; routes
//    can require their bodies to be JSON (HTML forms cannot send that type
//    across origins without a preflight).
//
//   http::Server srv({.port = 8080});
//   srv.route("POST", "/v1/chat/completions", [&](const http::Request& req, http::Response& res) {
//     json::Value body = json::parse(req.body);          // json::Error -> 400
//     if (!body.get("messages")) throw http::Error(400, "'messages' is required");
//     res.start_stream(200, "text/event-stream");
//     while (...) if (!res.sse(chunk_json)) return;      // client gone: stop generating
//     res.sse("[DONE]");
//   });
//   std::thread t([&] { srv.run(); });  ...  srv.stop();  t.join();
//
// Register routes before run(), or, behind a gate (gate()/open(): a server
// that answers 503 while its model loads), until open(). See server.cpp for
// the threading design.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../util/json.h"

namespace espetero::http {

struct Request {
  std::string method;  // as sent, case-sensitive ("GET", "POST", ...)
  std::string target;  // request-target as sent: path[?query]
  std::string path;    // raw path, not percent-decoded (see url_decode)
  std::string query;   // raw query string without the '?', "" if none
  std::vector<std::pair<std::string, std::string>> headers;  // names lower-cased, values trimmed, in order
  std::string body;    // complete body (chunked bodies are decoded)
  std::string remote;  // peer address, "ip:port"
  int minor_version = 1;  // HTTP/1.<minor_version>

  // First header with this name (case-insensitive), or nullptr.
  const std::string* header(std::string_view name) const;
  // First query parameter with this name, percent-decoded with '+' as space;
  // "" when absent.
  std::string query_param(std::string_view name) const;
};

namespace detail {
struct Conn;
struct ServerState;
}  // namespace detail

// The answer to one request. Created by the server and passed to the handler;
// valid only until the handler returns. Not thread-safe, except client_gone().
class Response {
 public:
  Response(const Response&) = delete;
  Response& operator=(const Response&) = delete;

  // Adds a header to the response (before it starts; CR/LF are stripped).
  // Content-Length and Transfer-Encoding are managed by the server and
  // ignored; "Connection: close" closes the connection after this response.
  void set_header(std::string name, std::string value);

  // Complete response with Content-Length, head and body in one write.
  void send(int status, std::string_view content_type, std::string_view body);
  // Compact JSON (ensure_ascii = false) as application/json.
  void send_json(int status, const json::Value& v);
  // OpenAI-style error body (see error_body). The type defaults to
  // "invalid_request_error" for 4xx and "server_error" for 5xx.
  void send_error(int status, std::string_view message, std::string_view type = {},
                  std::string_view code = {});

  // Streaming response: chunked transfer encoding (close-delimited for
  // HTTP/1.0 clients), Cache-Control: no-cache, every write sent at once.
  // start_stream(), write() and sse() return false once the client is gone,
  // the server is stopping or a write stalled for Options::io_timeout_s: the
  // handler should then stop its work (the stream is abandoned, not ended).
  bool start_stream(int status, std::string_view content_type);
  bool write(std::string_view data);
  // One Server-Sent Event: "data: <line>\n" for each line of data, then "\n".
  bool sse(std::string_view data);
  // Ends the stream (zero-length chunk). Done automatically when the handler
  // returns, unless the stream failed.
  void end();

  // True once the peer has closed or reset the connection, a write to it
  // failed, or the server is stopping. Checked without blocking and without
  // consuming pipelined requests. May be called from any thread while the
  // handler runs (e.g. by a scheduler polling for cancelled requests). A
  // client that half-closes its side after sending the request counts as gone.
  bool client_gone() const;

  int status() const { return status_; }                // 0 until the response starts
  bool started() const { return phase_ != Phase::Idle; }  // head sent (or attempted)
  uint64_t body_bytes() const { return bytes_; }         // body bytes written so far

 private:
  friend struct detail::Conn;
  enum class Phase : uint8_t { Idle, Streaming, Done, Failed };
  explicit Response(detail::Conn& c) : c_(c) {}
  bool has_header(std::string_view lower_name) const;

  detail::Conn& c_;
  Phase phase_ = Phase::Idle;
  int status_ = 0;
  uint64_t bytes_ = 0;
  std::vector<std::pair<std::string, std::string>> headers_;
  std::string scratch_;  // reused buffer for SSE framing
};

using Handler = std::function<void(const Request&, Response&)>;

// Throw from a handler to answer with an OpenAI-style JSON error. (json::Error
// escaping a handler becomes a 400, any other exception a 500.)
struct Error : std::runtime_error {
  int status;
  std::string type;  // "" = default for the status
  std::string code;  // optional "code" member
  Error(int status, const std::string& message, std::string type = {}, std::string code = {})
      : std::runtime_error(message), status(status), type(std::move(type)), code(std::move(code)) {}
};

// Thrown by Server's constructor when the address cannot be bound. `error`:
// the errno of the last attempt (EADDRINUSE: another program listens there).
struct ListenError : std::runtime_error {
  int error;
  ListenError(const std::string& message, int error) : std::runtime_error(message), error(error) {}
};

// What a route accepts as a request body. A body of another Content-Type is
// refused with 415 from the head, before it is read; requests without a body
// are not checked. Json: application/json. Multipart: multipart/form-data.
enum class Body : uint8_t { Any, Json, Multipart };

// Timeouts <= 0 disable the timeout.
struct Options {
  std::string host = "127.0.0.1";  // "0.0.0.0", or "::" for IPv6 + IPv4 on all interfaces
  int port = 8080;                 // 0 = ephemeral, see Server::port()
  int max_connections = 512;       // open connections, idle keep-alive ones included; beyond: 503 (<= 0: no limit)
  size_t max_body = 256 << 20;     // larger request bodies: 413
  int header_timeout_s = 30;       // to receive a request line + headers: 408
  int idle_timeout_s = 120;        // keep-alive wait for the next request
  std::string api_key;             // non-empty: require "Authorization: Bearer <key>" (401)
  bool cors = false;               // Access-Control-Allow-* headers + OPTIONS preflight
  // Without api_key: a Host header must name localhost (or *.localhost),
  // 127.x.x.x, [::1] or `host` (any port), and an Origin header must be this
  // server's own origin unless cors is set; otherwise 403. Requests without
  // these headers (curl, SDKs: no Origin) are not affected.
  bool local_guard = true;
  // Beyond the basics:
  size_t max_header = 64 << 10;    // request line + header fields: 414 / 431 beyond
  int io_timeout_s = 60;           // max stall while reading a body or writing a response (408 / write() false)
  bool log_requests = false;       // one line per request on stderr: peer method path status bytes ms
  size_t thread_stack = 8 << 20;   // stack of connection threads (handlers run there)
};

class Server {
 public:
  // Binds and listens right away (so port() is valid before run()). Throws
  // ListenError if the address cannot be bound, also when another program
  // already listens on the port at another address of this machine (macOS
  // would let both listen; see server.cpp).
  explicit Server(Options opts);
  ~Server();  // run() must have returned
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  // Exact path match. For HEAD requests without a HEAD route the GET route
  // answers and the body is dropped. `body`: the request bodies it accepts.
  void route(std::string method, std::string path, Handler h, Body body = Body::Any);
  // Path prefix match (e.g. "/v1/batches/"); the longest prefix wins, exact
  // routes win over prefixes.
  void route_prefix(std::string method, std::string prefix, Handler h, Body body = Body::Any);
  // A route served without the API key (a static page that asks for the key
  // itself). The browser guard still applies.
  void route_keyless(std::string method, std::string path, Handler h);

  // Startup gate, set before run(): until open(), the routes
  // registered before gate() are served as usual and every other request
  // that passes the browser guard and the API key is answered by `h` (its
  // body is not read: a request with a body closes its connection), e.g. a
  // 503 while the model loads. Routes registered after gate() (while the
  // server runs, from one thread) are served once open() is called.
  void gate(Handler h);
  // Lifts the gate. Thread-safe; no route may be registered after it.
  void open();

  // Accepts connections until stop(). Then closes the listener, wakes every
  // connection (idle ones close, streams see write() == false and
  // client_gone() == true) and returns once all handlers have returned.
  void run();
  // Thread-safe and async-signal-safe (an atomic store and a write(2)).
  void stop();

  int port() const;         // actual bound port
  int connections() const;  // connections being served right now

 private:
  std::unique_ptr<detail::ServerState> s_;
};

// {"error":{"message":...,"type":...,"param":null,"code":...}} (code null if empty).
json::Value error_body(std::string_view message, std::string_view type, std::string_view code = {});
// Percent-decoding; invalid escapes are kept as they are.
std::string url_decode(std::string_view s, bool plus_as_space = false);
// "OK", "Not Found", ... ("" for unknown codes).
std::string_view status_reason(int status);
// localhost, *.localhost, 127.x.x.x, ::1 or [::1] (also IPv4-mapped 127.x.x.x),
// case-insensitive: names and addresses that only reach this machine.
bool is_loopback_host(std::string_view host);

}  // namespace espetero::http
