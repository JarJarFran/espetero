// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Tests for src/http: servers run in-process on ephemeral ports and are
// exercised by a raw-socket client, so that malformed, partial, pipelined and
// abandoned messages can be produced byte for byte.
//
//   clang++ -std=c++20 -O2 -Wall -Wextra -pthread tests/http_test.cpp src/http/*.cpp src/util/json.cpp -o http_test
//   ./http_test                  run the tests
//   ./http_test --serve [port] [--log] [--cors] [--key K]
//                                serve the test routes (for tests/http_load.py and curl)
//
// SIGPIPE is deliberately left at its default action (terminate): a server
// write path without SO_NOSIGPIPE/MSG_NOSIGNAL would kill this test.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../src/http/http.h"
#include "../src/http/parse.h"

using namespace espetero;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

// ------------------------------------------------------------------ checks

static std::atomic<int> g_checks{0}, g_failures{0};

static std::string shorten(std::string s) {
  if (s.size() > 160) s = s.substr(0, 160) + "...(" + std::to_string(s.size()) + " bytes)";
  return s;
}

template <class A, class B>
static bool check_eq(const A& a, const B& b, const char* ea, const char* eb, int line) {
  ++g_checks;
  if (a == b) return true;
  ++g_failures;
  std::ostringstream x, y;
  x << a;
  y << b;
  std::fprintf(stderr, "  FAIL line %d: %s == %s  (%s vs %s)\n", line, ea, eb, shorten(x.str()).c_str(),
               shorten(y.str()).c_str());
  return false;
}

#define CHECK(c)                                                          \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(c)) {                                                           \
      ++g_failures;                                                       \
      std::fprintf(stderr, "  FAIL line %d: %s\n", __LINE__, #c);         \
    }                                                                     \
  } while (0)
#define REQUIRE(c)                                                        \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(c)) {                                                           \
      ++g_failures;                                                       \
      std::fprintf(stderr, "  FAIL line %d: %s (giving up)\n", __LINE__, #c); \
      return;                                                             \
    }                                                                     \
  } while (0)
#define CHECK_EQ(a, b) check_eq((a), (b), #a, #b, __LINE__)

static double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
static int64_t now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now().time_since_epoch()).count();
}
static void wait_for(const std::function<bool()>& cond, std::chrono::milliseconds limit = 2000ms) {
  const auto end = Clock::now() + limit;
  while (!cond() && Clock::now() < end) std::this_thread::sleep_for(1ms);
}

static uint64_t fnv1a(std::string_view s) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

static std::string pattern(size_t n) {
  std::string s(n, '\0');
  for (size_t i = 0; i < n; ++i) s[i] = char('a' + (i * 7 + i / 4096) % 26);
  return s;
}

// ------------------------------------------------------------------ test routes

struct Stats {
  std::atomic<int> sse_running{0}, sse_written{0}, sse_finished{0}, sse_aborted{0};
  std::atomic<int64_t> sse_end_us{0};
  std::atomic<int> slow_now{0}, slow_max{0};
  std::atomic<int> gone_detected{0};
  std::atomic<int64_t> gone_us{0};
  std::atomic<int> flood_done{0};
  std::atomic<int64_t> flood_stall_us{0}, flood_bytes{0};
};
static Stats g;

static int qint(const http::Request& req, const char* name, int def) {
  const std::string v = req.query_param(name);
  return v.empty() ? def : std::atoi(v.c_str());
}

static void add_routes(http::Server& srv) {
  srv.route("GET", "/hello", [](const http::Request&, http::Response& res) { res.send(200, "text/plain", "hello"); });

  srv.route("POST", "/echo", [](const http::Request& req, http::Response& res) {
    json::Value v = json::Value::object();
    v.set("method", req.method);
    v.set("path", req.path);
    v.set("query", req.query);
    v.set("len", int64_t(req.body.size()));
    v.set("json", json::parse(req.body));  // json::Error -> 400
    json::Value h = json::Value::object();
    for (const auto& [k, val] : req.headers) h.set(k, val);
    v.set("headers", std::move(h));
    res.send_json(200, v);
  });

  srv.route("POST", "/raw", [](const http::Request& req, http::Response& res) {
    res.send(200, "application/octet-stream", req.body);
  });

  srv.route("POST", "/big", [](const http::Request& req, http::Response& res) {
    json::Value v = json::Value::object();
    v.set("bytes", int64_t(req.body.size()));
    v.set("fnv", std::to_string(fnv1a(req.body)));
    res.send_json(200, v);
  });

  srv.route("GET", "/big", [](const http::Request& req, http::Response& res) {
    res.send(200, "application/octet-stream", pattern(size_t(qint(req, "n", 1 << 20))));
  });

  // n events (JSON), ms apart, then [DONE]; stops as soon as a write fails.
  srv.route("GET", "/sse", [](const http::Request& req, http::Response& res) {
    const int n = qint(req, "n", 200), ms = qint(req, "ms", 5);
    if (!res.start_stream(200, "text/event-stream")) return;
    ++g.sse_running;
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) {
      if (i && ms) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
      ok = res.sse("{\"i\":" + std::to_string(i) + "}");
      if (ok) ++g.sse_written;
    }
    if (ok) ok = res.sse("[DONE]");
    g.sse_end_us = now_us();
    ++(ok ? g.sse_finished : g.sse_aborted);
    --g.sse_running;
  });

  srv.route("GET", "/sse_lines", [](const http::Request&, http::Response& res) {
    res.start_stream(200, "text/event-stream");
    res.sse("a\nb\r\nc");
    res.sse("");
    res.sse("x\n");
  });  // end() is implicit

  srv.route("GET", "/slow", [](const http::Request& req, http::Response& res) {
    const int now = ++g.slow_now;
    int prev = g.slow_max.load();
    while (now > prev && !g.slow_max.compare_exchange_weak(prev, now)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(qint(req, "ms", 100)));
    --g.slow_now;
    res.send(200, "text/plain", "slow");
  });

  // A non-streaming handler "waiting for the GPU", polling for cancellation.
  srv.route("GET", "/wait_gone", [](const http::Request& req, http::Response& res) {
    const auto end = Clock::now() + std::chrono::milliseconds(qint(req, "ms", 5000));
    while (!res.client_gone() && Clock::now() < end) std::this_thread::sleep_for(1ms);
    if (res.client_gone()) {
      g.gone_us = now_us();
      ++g.gone_detected;
      return;  // nobody to answer
    }
    res.send(200, "text/plain", "still here");
  });

  // Streams 64 KiB chunks as fast as possible until a write fails; records
  // how long the failing write stalled (a client that stopped reading).
  srv.route("GET", "/flood", [](const http::Request&, http::Response& res) {
    const std::string chunk(64 << 10, 'f');
    int64_t bytes = 0, last_ok = now_us();
    if (res.start_stream(200, "application/octet-stream")) {
      while (bytes < (int64_t(1) << 30) && res.write(chunk)) {
        bytes += int64_t(chunk.size());
        last_ok = now_us();
      }
    }
    g.flood_stall_us = now_us() - last_ok;
    g.flood_bytes = bytes;
    ++g.flood_done;
  });

  srv.route("GET", "/remote", [](const http::Request& req, http::Response& res) {
    res.send(200, "text/plain", req.remote);
  });

  srv.route("GET", "/throw", [](const http::Request&, http::Response&) { throw std::runtime_error("boom"); });
  srv.route("GET", "/http_error", [](const http::Request&, http::Response&) {
    throw http::Error(429, "slow down", "rate_limit_error", "too_fast");
  });
  srv.route("GET", "/bad_json", [](const http::Request&, http::Response&) { json::parse("{\"a\":"); });
  srv.route("GET", "/no_response", [](const http::Request&, http::Response&) {});
  srv.route("GET", "/stream_throw", [](const http::Request&, http::Response& res) {
    res.start_stream(200, "text/event-stream");
    res.write("partial");
    throw std::runtime_error("failed mid-stream");
  });

  srv.route_prefix("GET", "/v1/batches/", [](const http::Request& req, http::Response& res) {
    res.send(200, "text/plain", http::url_decode(std::string_view(req.path).substr(12)));
  });
  srv.route("GET", "/v1/batches/special", [](const http::Request&, http::Response& res) {
    res.send(200, "text/plain", "exact");
  });
  srv.route("GET", "/query", [](const http::Request& req, http::Response& res) {
    json::Value v = json::Value::object();
    for (const char* k : {"a", "b", "c", "missing"}) v.set(k, req.query_param(k));
    res.send_json(200, v);
  });
  srv.route(
      "POST", "/v1/chat/completions",
      [](const http::Request&, http::Response& res) { res.send_json(200, json::parse("{\"ok\":true}")); },
      http::Body::Json);
  // Declared body types: refused with 415 from the head when the type differs.
  srv.route(
      "POST", "/json",
      [](const http::Request& req, http::Response& res) { res.send(200, "text/plain", std::to_string(req.body.size())); },
      http::Body::Json);
  srv.route(
      "POST", "/upload",
      [](const http::Request& req, http::Response& res) { res.send(200, "text/plain", std::to_string(req.body.size())); },
      http::Body::Multipart);
}

struct TestServer {
  http::Server srv;
  std::thread th;
  explicit TestServer(http::Options o) : srv(std::move(o)) {
    add_routes(srv);
    th = std::thread([this] { srv.run(); });
  }
  ~TestServer() {
    srv.stop();
    if (th.joinable()) th.join();
  }
  int port() const { return srv.port(); }
};

static http::Options test_options() {
  http::Options o;
  o.port = 0;
  o.max_body = 64 << 20;
  return o;
}

// ------------------------------------------------------------------ raw client

struct Resp {
  int status = 0;
  std::vector<std::pair<std::string, std::string>> headers;
  std::string body;
  std::string header(std::string_view name) const {
    for (const auto& [k, v] : headers)
      if (http::detail::iequals(k, name)) return v;
    return {};
  }
  bool has(std::string_view name) const {
    for (const auto& h : headers)
      if (http::detail::iequals(h.first, name)) return true;
    return false;
  }
};

class Client {
 public:
  explicit Client(int port, bool ipv6 = false) {
    fd_ = ::socket(ipv6 ? AF_INET6 : AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    ::setsockopt(fd_, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    timeval tv{10, 0};  // a test must never hang on a send
    ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    int rc;
    if (ipv6) {
      sockaddr_in6 a{};
      a.sin6_family = AF_INET6;
      a.sin6_port = htons(uint16_t(port));
      a.sin6_addr = in6addr_loopback;
      rc = ::connect(fd_, reinterpret_cast<sockaddr*>(&a), sizeof a);
    } else {
      sockaddr_in a{};
      a.sin_family = AF_INET;
      a.sin_port = htons(uint16_t(port));
      a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      rc = ::connect(fd_, reinterpret_cast<sockaddr*>(&a), sizeof a);
    }
    if (rc != 0) close();
  }
  ~Client() { close(); }
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  bool ok() const { return fd_ >= 0; }
  std::string& buffered() { return buf_; }

  bool send(std::string_view s) {
    while (!s.empty()) {
      const ssize_t n = ::send(fd_, s.data(), s.size(), 0);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) return false;
      s.remove_prefix(size_t(n));
    }
    return true;
  }
  void close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
  }
  void reset() {  // abortive close: the server gets a RST
    linger l{1, 0};
    ::setsockopt(fd_, SOL_SOCKET, SO_LINGER, &l, sizeof l);
    close();
  }

  // Status line and headers.
  std::optional<Resp> head(int timeout_ms = 5000) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    const size_t end = find("\r\n\r\n", deadline);
    if (end == std::string::npos) return std::nullopt;
    std::string_view h(buf_.data(), end - 2);
    Resp r;
    size_t eol = h.find("\r\n");
    if (h.substr(0, 9) != "HTTP/1.1 ") return std::nullopt;
    r.status = std::atoi(std::string(h.substr(9, 3)).c_str());
    for (size_t pos = eol + 2; pos < h.size();) {
      eol = h.find("\r\n", pos);
      const std::string_view line = h.substr(pos, eol - pos);
      const size_t colon = line.find(':');
      r.headers.emplace_back(std::string(line.substr(0, colon)),
                             std::string(http::detail::trim_ows(line.substr(colon + 1))));
      pos = eol + 2;
    }
    buf_.erase(0, end);
    return r;
  }

  // A whole response: Content-Length, chunked or close-delimited body.
  std::optional<Resp> response(int timeout_ms = 5000, bool head_request = false) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    auto r = head(timeout_ms);
    if (!r) return std::nullopt;
    if (head_request || r->status / 100 == 1 || r->status == 204 || r->status == 304) return r;
    if (http::detail::iequals(r->header("transfer-encoding"), "chunked")) {
      for (;;) {
        auto c = chunk(int(std::max<int64_t>(0, (deadline - Clock::now()) / 1ms)));
        if (!c) return std::nullopt;
        if (c->empty()) break;
        r->body += *c;
      }
    } else if (r->has("content-length")) {
      const size_t n = std::stoull(r->header("content-length"));
      if (!need(n, deadline)) return std::nullopt;
      r->body = buf_.substr(0, n);
      buf_.erase(0, n);
    } else {
      while (fill(deadline)) {
      }
      if (!eof_) return std::nullopt;
      r->body = std::move(buf_);
      buf_.clear();
    }
    return r;
  }

  // Next chunk of a chunked body; "" for the last chunk, nullopt on
  // EOF/timeout/garbage.
  std::optional<std::string> chunk(int timeout_ms = 5000) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    const size_t e = find("\r\n", deadline);
    if (e == std::string::npos) return std::nullopt;
    const size_t size = std::strtoull(buf_.c_str(), nullptr, 16);
    buf_.erase(0, e);
    if (size == 0) {
      const size_t t = find("\r\n", deadline);  // no trailers from this server
      if (t != 2) return std::nullopt;
      buf_.erase(0, t);
      return std::string();
    }
    if (!need(size + 2, deadline) || buf_.compare(size, 2, "\r\n") != 0) return std::nullopt;
    std::string data = buf_.substr(0, size);
    buf_.erase(0, size + 2);
    return data;
  }

  // True if the server closes the connection within timeout_ms.
  bool closed(int timeout_ms = 2000) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    while (fill(deadline)) {
    }
    return eof_;
  }

 private:
  bool fill(Clock::time_point deadline) {
    if (eof_ || fd_ < 0) return false;
    pollfd p{fd_, POLLIN, 0};
    const auto left = std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now()).count();
    if (::poll(&p, 1, int(std::max<int64_t>(0, left))) <= 0) return false;
    char tmp[1 << 16];
    const ssize_t n = ::recv(fd_, tmp, sizeof tmp, 0);
    if (n <= 0) {
      eof_ = true;  // EOF or reset
      return false;
    }
    buf_.append(tmp, size_t(n));
    return true;
  }
  size_t find(std::string_view delim, Clock::time_point deadline) {
    for (;;) {
      const size_t p = buf_.find(delim);
      if (p != std::string::npos) return p + delim.size();
      if (!fill(deadline)) return std::string::npos;
    }
  }
  bool need(size_t n, Clock::time_point deadline) {
    while (buf_.size() < n)
      if (!fill(deadline)) return false;
    return true;
  }

  int fd_ = -1;
  std::string buf_;
  bool eof_ = false;
};

static std::string get(std::string_view path, std::string_view extra = "") {
  return "GET " + std::string(path) + " HTTP/1.1\r\nHost: 127.0.0.1\r\n" + std::string(extra) + "\r\n";
}
static std::string post(std::string_view path, std::string_view body, std::string_view extra = "") {
  return "POST " + std::string(path) + " HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: " +
         std::to_string(body.size()) + "\r\n" + std::string(extra) + "\r\n" + std::string(body);
}
// One request on a fresh connection; also reports whether the server closed it.
static std::optional<Resp> once(int port, std::string_view raw, bool* closed = nullptr) {
  Client c(port);
  if (!c.send(raw)) return std::nullopt;
  auto r = c.response();
  if (closed) *closed = c.closed(1000);
  return r;
}
static json::Value error_of(const Resp& r) {
  json::Value v = json::parse(r.body);
  const json::Value* e = v.get("error");
  return e ? *e : json::Value();
}

// ------------------------------------------------------------------ tests

static void test_parse_units() {
  using namespace http::detail;
  CHECK_EQ(http::url_decode("a%20b+c", true), "a b c");
  CHECK_EQ(http::url_decode("a%20b+c"), "a b+c");
  CHECK_EQ(http::url_decode("%zz%4"), "%zz%4");
  CHECK_EQ(http::url_decode("%E2%9c%93"), "\xE2\x9C\x93");

  uint64_t n = 0;
  CHECK(parse_chunk_size("1a", n) && n == 26);
  CHECK(parse_chunk_size("FF ; name=value", n) && n == 255);
  CHECK(!parse_chunk_size("", n));
  CHECK(!parse_chunk_size("x1", n));
  CHECK(!parse_chunk_size("1 2", n));
  CHECK(!parse_chunk_size("10000000000000000", n));  // 2^64

  size_t scan = 0;
  CHECK_EQ(find_head_end("GET / HTTP/1.1\r\nA: b\r\n\r\nrest", scan), 24u);
  scan = 0;
  CHECK_EQ(find_head_end("GET / HTTP/1.1\nA: b\n\nrest", scan), 21u);
  scan = 0;
  std::string partial = "GET / HTTP/1.1\r\nA: b\r\n\r";
  CHECK_EQ(find_head_end(partial, scan), 0u);
  partial += "\n";
  CHECK_EQ(find_head_end(partial, scan), partial.size());

  http::Request req;
  Head h = parse_head("POST /v1/x?y=1&z HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\nX-Up:  V \r\n\r\n", req, 100);
  CHECK_EQ(h.error, 0);
  CHECK_EQ(req.method, "POST");
  CHECK_EQ(req.path, "/v1/x");
  CHECK_EQ(req.query, "y=1&z");
  CHECK_EQ(h.content_length, 5u);
  CHECK(h.keep_alive);
  CHECK(req.header("X-UP") && *req.header("x-up") == "V");
  CHECK_EQ(req.headers[1].first, "content-length");
  CHECK_EQ(req.query_param("y"), "1");
  CHECK_EQ(req.query_param("z"), "");

  // Host / Origin / Content-Type helpers of the browser guard and body types.
  std::string host, hport;
  CHECK(split_authority("LocalHost.:8080", host, hport) && host == "localhost" && hport == "8080");
  CHECK(split_authority("[::1]:80", host, hport) && host == "[::1]" && hport == "80");
  CHECK(split_authority("example.com", host, hport) && host == "example.com" && hport.empty());
  CHECK(!split_authority("::1", host, hport));  // IPv6 needs brackets in a Host header
  CHECK(!split_authority("a:b", host, hport));
  CHECK(!split_authority("[::1", host, hport));
  for (const char* h : {"localhost", "LOCALHOST:1", "localhost.", "app.localhost", "127.0.0.1", "127.9.8.7:80", "[::1]",
                        "[::1]:8080", "::1", "[::ffff:127.0.0.1]"})
    check_eq(http::is_loopback_host(h), true, h, "loopback", __LINE__);
  for (const char* h : {"", "evillocalhost", "localhost.evil.example", "127.0.0.1.nip.io", "0.0.0.0", "[::]", "10.0.0.1",
                        "[::2]", "128.0.0.1"})
    check_eq(http::is_loopback_host(h), false, h, "not loopback", __LINE__);
  CHECK(host_allowed("", "127.0.0.1"));
  CHECK(host_allowed("localhost:9", "0.0.0.0"));
  CHECK(host_allowed("192.168.1.5:8080", "192.168.1.5"));
  CHECK(host_allowed("[fe80::1]:8080", "fe80::1"));
  CHECK(host_allowed("mybox.local", "MyBox.local"));
  CHECK(!host_allowed("192.168.1.5:8080", "0.0.0.0"));
  CHECK(!host_allowed("evil.example", "127.0.0.1"));
  CHECK(!host_allowed("evil.example", ""));
  CHECK(same_origin("http://127.0.0.1:8080", "127.0.0.1:8080"));
  CHECK(same_origin("http://LOCALHOST", "localhost:80"));
  CHECK(!same_origin("https://127.0.0.1:8080", "127.0.0.1:8080"));
  CHECK(!same_origin("http://127.0.0.1:8081", "127.0.0.1:8080"));
  CHECK(!same_origin("http://localhost:8080", "127.0.0.1:8080"));
  CHECK(!same_origin("null", "127.0.0.1:8080"));
  CHECK(!same_origin("http://evil.example", "127.0.0.1:8080"));
  CHECK_EQ(media_type(" Application/JSON ; charset=utf-8"), "application/json");
  CHECK_EQ(media_type("multipart/form-data; boundary=x"), "multipart/form-data");
  CHECK_EQ(media_type(""), "");

  req = {};
  parse_head("GET http://example.com:80/a/b?c=d HTTP/1.1\r\n\r\n", req, 100);
  CHECK_EQ(req.path, "/a/b");
  CHECK_EQ(req.query, "c=d");
  req = {};
  parse_head("GET https://example.com?q HTTP/1.1\r\n\r\n", req, 100);
  CHECK_EQ(req.path, "/");
  CHECK_EQ(req.query, "q");

  auto parse = [](std::string_view head) {
    http::Request r;
    return parse_head(head, r, 100);
  };
  CHECK_EQ(parse("GET / HTTP/1.1\r\nContent-Length: 3, 3\r\n\r\n").error, 0);
  CHECK_EQ(parse("GET / HTTP/1.1\r\nContent-Length: 1, 2\r\n\r\n").error, 400);
  CHECK_EQ(parse("GET / HTTP/1.1\r\nContent-Length: -1\r\n\r\n").error, 400);
  CHECK_EQ(parse("GET / HTTP/1.1\r\nContent-Length: 99999999999999999999999\r\n\r\n").content_length, UINT64_MAX);
  CHECK_EQ(parse("GET / HTTP/3.0\r\n\r\n").error, 505);
  CHECK_EQ(parse("GET / HTTP/1.1\r\nTransfer-Encoding: chunked, chunked\r\n\r\n").error, 400);
  CHECK_EQ(parse("GET / HTTP/1.1\r\nTransfer-Encoding: gzip\r\n\r\n").error, 400);
  CHECK_EQ(parse("GET / HTTP/1.0\r\nTransfer-Encoding: chunked\r\n\r\n").error, 400);
  CHECK_EQ(parse("GET /\x01 HTTP/1.1\r\n\r\n").error, 400);
  CHECK_EQ(parse("G(T / HTTP/1.1\r\n\r\n").error, 400);
  CHECK_EQ(parse("GET / HTTP/1.1\r\nA\r\n\r\n").error, 400);
  CHECK(!parse("GET / HTTP/1.0\r\n\r\n").keep_alive);
  CHECK(parse("GET / HTTP/1.0\r\nConnection: Keep-Alive\r\n\r\n").keep_alive);
  CHECK(!parse("GET / HTTP/1.1\r\nConnection: foo, close\r\n\r\n").keep_alive);
  CHECK(!parse("POST / HTTP/1.0\r\nExpect: 100-continue\r\n\r\n").expect_continue);
  CHECK(parse("POST / HTTP/1.1\r\nExpect: 100-Continue\r\n\r\n").expect_continue);
  CHECK(parse("POST / HTTP/1.1\r\nTransfer-Encoding: Chunked\r\n\r\n").chunked);
}

static void test_basic(int port) {
  Client c(port);
  REQUIRE(c.send(get("/hello")));
  auto r = c.response();
  REQUIRE(r && r->status == 200);
  CHECK_EQ(r->body, "hello");
  CHECK_EQ(r->header("content-length"), "5");
  CHECK_EQ(r->header("content-type"), "text/plain");
  CHECK(r->header("date").ends_with(" GMT"));

  const std::string body = R"({"a":1,"s":"héllo ✓"})";
  REQUIRE(c.send(post("/echo?x=1", body, "X-Custom:  Value \r\n")));
  r = c.response();
  REQUIRE(r && r->status == 200);
  CHECK_EQ(r->header("content-type"), "application/json");
  json::Value v = json::parse(r->body);
  CHECK_EQ(v.get_string("method", ""), "POST");
  CHECK_EQ(v.get_string("path", ""), "/echo");
  CHECK_EQ(v.get_string("query", ""), "x=1");
  CHECK_EQ(v.get_int("len", -1), int64_t(body.size()));
  CHECK_EQ(v["json"].get_string("s", ""), "héllo ✓");
  CHECK_EQ(v["headers"].get_string("x-custom", ""), "Value");
  CHECK(r->body.find("héllo ✓") != std::string::npos);  // UTF-8 out, not \u escapes
}

static void test_keepalive(int port) {
  Client c(port);
  for (int i = 0; i < 5; ++i) {
    REQUIRE(c.send(get("/hello")));
    auto r = c.response();
    REQUIRE(r && r->status == 200 && r->body == "hello");
    CHECK(!r->has("connection"));
    REQUIRE(c.send(post("/echo", "{\"i\":" + std::to_string(i) + "}")));
    r = c.response();
    REQUIRE(r && r->status == 200);
    CHECK_EQ(json::parse(r->body)["json"].get_int("i", -1), i);
  }
  CHECK(!c.closed(50));
  REQUIRE(c.send(get("/hello", "Connection: close\r\n")));
  auto r = c.response();
  REQUIRE(r && r->status == 200);
  CHECK_EQ(r->header("connection"), "close");
  CHECK(c.closed());

  // A request split into single bytes (many TCP segments).
  Client d(port);
  for (char ch : post("/echo", "{\"split\":true}")) REQUIRE(d.send(std::string_view(&ch, 1)));
  r = d.response();
  REQUIRE(r && r->status == 200);
  CHECK(json::parse(r->body)["json"].get_bool("split", false));
}

static void test_http10(int port) {
  {
    Client c(port);
    REQUIRE(c.send("GET /hello HTTP/1.0\r\n\r\n"));
    auto r = c.response();
    REQUIRE(r && r->status == 200);
    CHECK_EQ(r->body, "hello");
    CHECK_EQ(r->header("connection"), "close");
    CHECK(c.closed());
  }
  {
    Client c(port);
    for (int i = 0; i < 2; ++i) {
      REQUIRE(c.send("GET /hello HTTP/1.0\r\nConnection: keep-alive\r\n\r\n"));
      auto r = c.response();
      REQUIRE(r && r->status == 200);
      CHECK_EQ(r->header("connection"), "keep-alive");
    }
    CHECK(!c.closed(50));
  }
  {  // no chunked encoding for HTTP/1.0: the stream ends with the connection
    Client c(port);
    REQUIRE(c.send("GET /sse?n=3&ms=0 HTTP/1.0\r\n\r\n"));
    auto r = c.response();
    REQUIRE(r && r->status == 200);
    CHECK(!r->has("transfer-encoding"));
    CHECK_EQ(r->body, "data: {\"i\":0}\n\ndata: {\"i\":1}\n\ndata: {\"i\":2}\n\ndata: [DONE]\n\n");
  }
}

static void test_pipelining(int port) {
  Client c(port);
  // Five requests in one write, including a stream: while it streams, the
  // pipelined requests behind it sit in the socket and must neither be taken
  // for a disconnect nor be lost.
  REQUIRE(c.send(get("/hello") + post("/echo", "{\"n\":1}") + get("/query?a=x") + get("/sse?n=3&ms=2") +
                 get("/hello", "Connection: close\r\n")));
  auto r = c.response();
  REQUIRE(r && r->status == 200);
  CHECK_EQ(r->body, "hello");
  r = c.response();
  REQUIRE(r && r->status == 200);
  CHECK_EQ(json::parse(r->body)["json"].get_int("n", 0), 1);
  r = c.response();
  REQUIRE(r && r->status == 200);
  CHECK_EQ(json::parse(r->body).get_string("a", ""), "x");
  r = c.response();
  REQUIRE(r && r->status == 200);
  CHECK_EQ(r->body, "data: {\"i\":0}\n\ndata: {\"i\":1}\n\ndata: {\"i\":2}\n\ndata: [DONE]\n\n");
  r = c.response();
  REQUIRE(r && r->status == 200);
  CHECK_EQ(r->body, "hello");
  CHECK(c.closed());

  // Empty lines between pipelined requests are ignored.
  Client d(port);
  REQUIRE(d.send("\r\n" + get("/hello") + "\r\n\r\n" + get("/hello")));
  for (int i = 0; i < 2; ++i) {
    r = d.response();
    CHECK(r && r->status == 200);
  }
}

static void test_chunked_request(int port) {
  Client c(port);
  REQUIRE(c.send("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n"
                 "4;ext=1\r\n{\"a\"\r\n"
                 "6\r\n:[1,2,\r\n"
                 "3\r\n3]}\r\n"
                 "0\r\nX-Trailer: yes\r\n\r\n"));
  auto r = c.response();
  REQUIRE(r && r->status == 200);
  json::Value v = json::parse(r->body);
  CHECK_EQ(v.get_int("len", 0), 13);
  CHECK_EQ(v["json"]["a"].size(), 3u);
  CHECK(!v["headers"].get("x-trailer"));  // trailers are dropped

  // 8 MB in uneven chunks, on the same connection.
  const std::string big = pattern(8 << 20);
  std::string msg = "POST /big HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n";
  char line[32];
  for (size_t off = 0, step = 1; off < big.size(); off += step, step = step * 3 % 300007 + 1) {
    step = std::min(step, big.size() - off);
    std::snprintf(line, sizeof line, "%zX\r\n", step);
    msg.append(line).append(big, off, step).append("\r\n");
  }
  msg += "0\r\n\r\n";
  REQUIRE(c.send(msg));
  r = c.response();
  REQUIRE(r && r->status == 200);
  v = json::parse(r->body);
  CHECK_EQ(v.get_int("bytes", 0), int64_t(big.size()));
  CHECK_EQ(v.get_string("fnv", ""), std::to_string(fnv1a(big)));

  bool closed = false;
  r = once(port, "POST /raw HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n", &closed);
  CHECK(r && r->status == 400 && closed);
  r = once(port, "POST /raw HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabcX\r\n0\r\n\r\n", &closed);
  CHECK(r && r->status == 400 && closed);  // data longer than its chunk size
}

static void test_expect_continue(int port) {
  {
    Client c(port);
    const std::string body = pattern(1 << 20);
    REQUIRE(c.send("POST /big HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1048576\r\nExpect: 100-continue\r\n\r\n"));
    auto r = c.response(2000);
    REQUIRE(r && r->status == 100);  // the go-ahead comes before any body byte is sent
    REQUIRE(c.send(body));
    r = c.response();
    REQUIRE(r && r->status == 200);
    json::Value v = json::parse(r->body);
    CHECK_EQ(v.get_int("bytes", 0), int64_t(body.size()));
    CHECK_EQ(v.get_string("fnv", ""), std::to_string(fnv1a(body)));
    REQUIRE(c.send(get("/hello")));  // still in sync
    r = c.response();
    CHECK(r && r->status == 200);
  }
  {  // refused up front: final status instead of 100, and the connection closes
    Client c(port);
    REQUIRE(c.send("POST /big HTTP/1.1\r\nHost: localhost\r\nContent-Length: 999999999999\r\nExpect: 100-continue\r\n\r\n"));
    auto r = c.response();
    REQUIRE(r && r->status == 413);
    CHECK_EQ(r->header("connection"), "close");
    CHECK_EQ(error_of(*r).get_string("type", ""), "invalid_request_error");
    CHECK(c.closed());
  }
  {
    Client c(port);
    REQUIRE(c.send("POST /nope HTTP/1.1\r\nHost: localhost\r\nContent-Length: 10\r\nExpect: 100-continue\r\n\r\n"));
    auto r = c.response();
    REQUIRE(r && r->status == 404);
    CHECK(c.closed());
  }
  {  // body sent without waiting: no 100 needed
    Client c(port);
    REQUIRE(c.send(post("/echo", "{}", "Expect: 100-continue\r\n")));
    auto r = c.response();
    CHECK(r && r->status == 200);
  }
  auto r = once(port, get("/hello", "Expect: something-else\r\n"));
  CHECK(r && r->status == 417);
}

static void test_limits(int port) {
  bool closed = false;
  auto r = once(port, "POST /big HTTP/1.1\r\nHost: localhost\r\nContent-Length: 67108865\r\n\r\n", &closed);
  CHECK(r && r->status == 413 && closed);
  r = once(port, get("/hello", "X-Big: " + std::string(70000, 'a') + "\r\n"), &closed);
  CHECK(r && r->status == 431 && closed);
  std::string many;
  for (int i = 0; i < 150; ++i) many += "X-N" + std::to_string(i) + ": v\r\n";
  r = once(port, get("/hello", many), &closed);
  CHECK(r && r->status == 431 && closed);
  r = once(port, get("/" + std::string(70000, 'a')), &closed);
  CHECK(r && r->status == 414 && closed);

  // A chunked body over the limit, sent without waiting: 413, and the
  // lingering close drains the rest so the client still reads the answer.
  http::Options o = test_options();
  o.max_body = 1 << 20;
  TestServer small(o);
  Client c(small.port());
  std::string msg = "POST /big HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n";
  const std::string piece(64 << 10, 'z');
  for (int i = 0; i < 32; ++i) msg += "10000\r\n" + piece + "\r\n";
  msg += "0\r\n\r\n";
  c.send(msg);  // may fail part-way if the server gave up; the answer must still be readable
  r = c.response();
  CHECK(r && r->status == 413);
  CHECK(c.closed());
}

static void test_malformed(int port) {
  const std::pair<const char*, int> cases[] = {
      {"GARBAGE\r\n\r\n", 400},
      {"GET /hello\r\n\r\n", 400},  // HTTP/0.9
      {"GET  /hello HTTP/1.1\r\n\r\n", 400},
      {"GET hello HTTP/1.1\r\n\r\n", 400},
      {"GET /hello HTTP/1.1\r\nBad Header: x\r\n\r\n", 400},
      {"GET /hello HTTP/1.1\r\nA: b\r\n folded\r\n\r\n", 400},
      {"POST /raw HTTP/1.1\r\nContent-Length: 12abc\r\n\r\n", 400},
      {"POST /raw HTTP/1.1\r\nContent-Length: 5\r\nContent-Length: 6\r\n\r\nhello", 400},
      {"POST /raw HTTP/1.1\r\nContent-Length: 5\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n", 400},
      {"POST /raw HTTP/1.1\r\nTransfer-Encoding: gzip, chunked\r\n\r\n0\r\n\r\n", 501},
      {"POST /raw HTTP/1.1\r\nTransfer-Encoding: chunked, gzip\r\n\r\n", 400},
      {"GET /hello HTTP/2.0\r\n\r\n", 505},
      {"GET /hello HTTP/1.1\r\nA: b\rc\r\n\r\n", 400},
  };
  for (const auto& [raw, want] : cases) {
    bool closed = false;
    auto r = once(port, raw, &closed);
    if (!check_eq(r ? r->status : -1, want, raw, "status", __LINE__)) continue;
    CHECK(closed);
    CHECK_EQ(error_of(*r).get_string("type", ""), want >= 500 ? "server_error" : "invalid_request_error");
  }
  bool closed = true;
  auto r = once(port, std::string("GET /hello HTTP/1.1\r\nA: b\0c\r\n\r\n", 32), &closed);
  CHECK(r && r->status == 400 && closed);

  // Bad JSON in a well-framed request is the handler's 400; the connection stays usable.
  Client c(port);
  REQUIRE(c.send(post("/echo", "{bad")));
  r = c.response();
  REQUIRE(r && r->status == 400);
  CHECK_EQ(error_of(*r).get_string("type", ""), "invalid_request_error");
  CHECK(error_of(*r).get_string("message", "").starts_with("Invalid JSON body"));
  REQUIRE(c.send(get("/hello")));
  r = c.response();
  CHECK(r && r->status == 200);

  // Tolerated forms.
  r = once(port, "GET http://localhost/hello HTTP/1.1\r\nHost: localhost\r\n\r\n");
  CHECK(r && r->status == 200 && r->body == "hello");
  r = once(port, "GET /hello HTTP/1.1\nHost: localhost\n\n");
  CHECK(r && r->status == 200);
}

static void test_routing(int port) {
  Client c(port);
  REQUIRE(c.send(get("/nope")));
  auto r = c.response();
  REQUIRE(r && r->status == 404);
  CHECK_EQ(r->header("content-type"), "application/json");
  json::Value e = error_of(*r);
  CHECK_EQ(e.get_string("message", ""), "Invalid URL (GET /nope)");
  CHECK_EQ(e.get_string("type", ""), "invalid_request_error");

  REQUIRE(c.send("PUT /hello HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n"));
  r = c.response();
  REQUIRE(r && r->status == 405);
  CHECK_EQ(r->header("allow"), "GET, HEAD");

  REQUIRE(c.send("HEAD /hello HTTP/1.1\r\nHost: localhost\r\n\r\n"));
  r = c.response(5000, /*head_request=*/true);
  REQUIRE(r && r->status == 200);
  CHECK_EQ(r->header("content-length"), "5");
  REQUIRE(c.send(get("/hello")));  // no stray body bytes before this response
  r = c.response();
  CHECK(r && r->status == 200 && r->body == "hello");

  REQUIRE(c.send(get("/v1/batches/abc%20d?x=1")));
  r = c.response();
  CHECK(r && r->status == 200 && r->body == "abc d");
  REQUIRE(c.send(get("/v1/batches/special")));
  r = c.response();
  CHECK(r && r->status == 200 && r->body == "exact");

  REQUIRE(c.send(get("/query?a=hello+world&b=%E2%9C%93&c")));
  r = c.response();
  REQUIRE(r && r->status == 200);
  json::Value v = json::parse(r->body);
  CHECK_EQ(v.get_string("a", "?"), "hello world");
  CHECK_EQ(v.get_string("b", "?"), "\xE2\x9C\x93");
  CHECK_EQ(v.get_string("c", "?"), "");
  CHECK_EQ(v.get_string("missing", "?"), "");
  CHECK(!c.closed(50));  // none of the above needed to close
}

static void test_handler_errors(int port) {
  Client c(port);
  REQUIRE(c.send(get("/throw")));
  auto r = c.response();
  REQUIRE(r && r->status == 500);
  CHECK_EQ(error_of(*r).get_string("message", ""), "boom");
  CHECK_EQ(error_of(*r).get_string("type", ""), "server_error");

  REQUIRE(c.send(get("/http_error")));  // same connection: still in sync
  r = c.response();
  REQUIRE(r && r->status == 429);
  CHECK_EQ(error_of(*r).get_string("type", ""), "rate_limit_error");
  CHECK_EQ(error_of(*r).get_string("code", ""), "too_fast");

  REQUIRE(c.send(get("/bad_json")));
  r = c.response();
  CHECK(r && r->status == 400);
  REQUIRE(c.send(get("/no_response")));
  r = c.response();
  CHECK(r && r->status == 500);

  // A stream that fails after its head: cut off without the last chunk.
  REQUIRE(c.send(get("/stream_throw")));
  auto h = c.head();
  REQUIRE(h && h->status == 200);
  auto ch = c.chunk();
  CHECK(ch && *ch == "partial");
  CHECK(!c.chunk(1000));
  CHECK(c.closed());
}

static void test_auth_cors() {
  http::Options o = test_options();
  o.api_key = "sekret";
  o.cors = true;
  TestServer ts(o);
  const int port = ts.port();

  auto r = once(port, get("/hello"));
  REQUIRE(r && r->status == 401);
  CHECK_EQ(r->header("www-authenticate"), "Bearer");
  CHECK_EQ(r->header("access-control-allow-origin"), "*");
  CHECK_EQ(error_of(*r).get_string("code", ""), "invalid_api_key");
  r = once(port, get("/hello", "Authorization: Bearer wrong!\r\n"));
  CHECK(r && r->status == 401);
  r = once(port, get("/hello", "Authorization: Bearer sekret\r\n"));
  CHECK(r && r->status == 200 && r->header("access-control-allow-origin") == "*");
  r = once(port, get("/hello", "authorization: bearer   sekret\r\n"));
  CHECK(r && r->status == 200);

  // Preflight: no credentials, answered by the server itself.
  r = once(port,
           "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nOrigin: http://app\r\n"
           "Access-Control-Request-Method: POST\r\nAccess-Control-Request-Headers: authorization, content-type\r\n\r\n");
  REQUIRE(r && r->status == 204);
  CHECK(r->header("access-control-allow-methods").find("POST") != std::string::npos);
  CHECK_EQ(r->header("access-control-allow-headers"), "authorization, content-type");
  CHECK_EQ(r->header("access-control-allow-origin"), "*");
  CHECK(!r->has("content-length"));

  // Unauthenticated body: refused without reading it, connection closed.
  bool closed = false;
  r = once(port, post("/v1/chat/completions", std::string(200000, ' ')), &closed);
  CHECK(r && r->status == 401 && closed);
  r = once(port, post("/v1/chat/completions", "{}", "Authorization: Bearer sekret\r\n"));
  CHECK(r && r->status == 200 && r->body == "{\"ok\":true}");
  r = once(port, "DELETE /hello HTTP/1.1\r\nAuthorization: Bearer sekret\r\n\r\n");
  CHECK(r && r->status == 405 && r->header("allow") == "GET, HEAD, OPTIONS");
}

// Without an API key: DNS rebinding (a foreign name in Host) and cross-site
// requests (a foreign Origin) are refused from the head; with a key or --cors
// they are the operator's choice.
static void test_local_guard() {
  TestServer ts(test_options());
  const int port = ts.port();
  const std::string me = "127.0.0.1:" + std::to_string(port);
  auto req = [](std::string_view host, std::string_view extra = "") {
    return "GET /hello HTTP/1.1\r\nHost: " + std::string(host) + "\r\n" + std::string(extra) + "\r\n";
  };
  for (const char* host : {"localhost", "LocalHost:8080", "127.0.0.1", "127.9.8.7:1", "[::1]:8080", "app.localhost"}) {
    auto r = once(port, req(host));
    check_eq(r ? r->status : -1, 200, host, "200", __LINE__);
  }
  for (const char* host : {"evil.example", "evil.example:8080", "localhost.evil.example", "10.0.0.5", "[::2]", "0.0.0.0"}) {
    auto r = once(port, req(host));
    if (!check_eq(r ? r->status : -1, 403, host, "403", __LINE__)) continue;
    CHECK_EQ(error_of(*r).get_string("type", ""), "invalid_request_error");
    CHECK(error_of(*r).get_string("message", "").find("DNS rebinding") != std::string::npos);
  }
  auto r = once(port, "GET /hello HTTP/1.0\r\n\r\n");  // no Host at all: not a browser
  CHECK(r && r->status == 200);
  r = once(port, req(me, "Origin: http://" + me + "\r\n"));  // the server's own origin
  CHECK(r && r->status == 200);
  for (const std::string& origin : {std::string("http://evil.example"), std::string("null"), "https://" + me,
                                    "http://localhost:" + std::to_string(port), "http://" + me + "0"}) {
    r = once(port, req(me, "Origin: " + origin + "\r\n"));
    if (!check_eq(r ? r->status : -1, 403, origin.c_str(), "403", __LINE__)) continue;
    CHECK(error_of(*r).get_string("message", "").find("--cors") != std::string::npos);
  }
  // A cross-site form post: refused from the head, its body never read.
  bool closed = false;
  r = once(port,
           "POST /echo HTTP/1.1\r\nHost: " + me +
               "\r\nOrigin: http://evil.example\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\n{}",
           &closed);
  CHECK(r && r->status == 403 && closed);
  {  // with an API key, the key is the protection: any Host, any Origin
    http::Options o = test_options();
    o.api_key = "k3y";
    TestServer keyed(o);
    r = once(keyed.port(), req("evil.example", "Authorization: Bearer k3y\r\nOrigin: http://evil.example\r\n"));
    CHECK(r && r->status == 200);
    r = once(keyed.port(), req("evil.example", "Origin: http://evil.example\r\n"));
    CHECK(r && r->status == 401);
  }
  {  // --cors without a key: other origins are welcome, host names still checked
    http::Options o = test_options();
    o.cors = true;
    TestServer cors(o);
    const std::string host = "127.0.0.1:" + std::to_string(cors.port());
    r = once(cors.port(), req(host, "Origin: http://evil.example\r\n"));
    CHECK(r && r->status == 200 && r->header("access-control-allow-origin") == "*");
    r = once(cors.port(),
             "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: " + host +
                 "\r\nOrigin: http://evil.example\r\nAccess-Control-Request-Method: POST\r\n\r\n");
    CHECK(r && r->status == 204);
    r = once(cors.port(), req("evil.example", "Origin: http://evil.example\r\n"));
    CHECK(r && r->status == 403);
  }
  {  // the guard off: a server behind its own protection
    http::Options o = test_options();
    o.local_guard = false;
    TestServer open(o);
    r = once(open.port(), req("evil.example", "Origin: http://evil.example\r\n"));
    CHECK(r && r->status == 200);
  }
}

// Routes that declare their body type refuse other types with 415 before the
// body is read (an HTML form cannot send application/json across origins).
static void test_body_types(int port) {
  const std::string me = "127.0.0.1:" + std::to_string(port);
  auto post_as = [&](std::string_view path, std::string_view type, std::string_view body) {
    std::string h = "POST " + std::string(path) + " HTTP/1.1\r\nHost: " + me + "\r\n";
    if (!type.empty()) h += "Content-Type: " + std::string(type) + "\r\n";
    return h + "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + std::string(body);
  };
  for (const char* t : {"application/json", "application/json; charset=utf-8", "Application/JSON"}) {
    auto r = once(port, post_as("/json", t, "{\"a\":1}"));
    if (check_eq(r ? r->status : -1, 200, t, "200", __LINE__)) CHECK_EQ(r->body, "7");
  }
  for (const char* t : {"text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x", ""}) {
    bool closed = false;
    auto r = once(port, post_as("/json", t, "{\"a\":1}"), &closed);
    if (!check_eq(r ? r->status : -1, 415, t, "415", __LINE__)) continue;
    CHECK(closed);  // the body was never read
    CHECK(error_of(*r).get_string("message", "").find("application/json") != std::string::npos);
  }
  auto r = once(port, post_as("/json", "", ""));  // no body: nothing to check
  CHECK(r && r->status == 200 && r->body == "0");
  r = once(port, "POST /json HTTP/1.1\r\nHost: " + me +
                     "\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n");
  CHECK(r && r->status == 415);
  {  // refused instead of the 100-continue go-ahead
    Client c(port);
    REQUIRE(c.send("POST /json HTTP/1.1\r\nHost: " + me +
                   "\r\nContent-Type: text/plain\r\nContent-Length: 1000000\r\nExpect: 100-continue\r\n\r\n"));
    r = c.response();
    CHECK(r && r->status == 415);
    CHECK(c.closed());
  }
  r = once(port, post_as("/upload", "multipart/form-data; boundary=xyz", "--xyz--\r\n"));
  CHECK(r && r->status == 200);
  r = once(port, post_as("/upload", "application/json", "{}"));
  CHECK(r && r->status == 415);
  r = once(port, post_as("/raw", "application/octet-stream", "abc"));  // no declared type: anything goes
  CHECK(r && r->status == 200 && r->body == "abc");
}

static void test_sse(int port) {
  Client c(port);
  REQUIRE(c.send(get("/sse")));
  const auto t0 = Clock::now();
  auto h = c.head();
  REQUIRE(h && h->status == 200);
  CHECK_EQ(h->header("content-type"), "text/event-stream");
  CHECK_EQ(h->header("cache-control"), "no-cache");
  CHECK_EQ(h->header("transfer-encoding"), "chunked");
  double first_ms = 0;
  for (int i = 0; i < 200; ++i) {
    auto ch = c.chunk(2000);  // one chunk per event: flushed per write
    REQUIRE(ch);
    if (i == 0) first_ms = ms_since(t0);
    if (!check_eq(*ch, "data: {\"i\":" + std::to_string(i) + "}\n\n", "chunk", "event", __LINE__)) return;
  }
  auto ch = c.chunk();
  CHECK(ch && *ch == "data: [DONE]\n\n");
  ch = c.chunk();
  CHECK(ch && ch->empty());
  const double total = ms_since(t0);
  CHECK(first_ms < 50);
  CHECK(total > 900 && total < 30000);  // 199 sleeps of 5 ms: ~1 s here, ~7 s on a CI virtual machine
  std::printf("  200 events: first after %.1f ms, all after %.0f ms\n", first_ms, total);

  REQUIRE(c.send(get("/sse_lines")));  // the connection is reusable after a stream
  auto r = c.response();
  REQUIRE(r && r->status == 200);
  CHECK_EQ(r->body, "data: a\ndata: b\ndata: c\n\ndata: \n\ndata: x\ndata: \n\n");
}

static void test_disconnect(int port) {
  for (const bool rst : {false, true}) {
    wait_for([] { return g.sse_running.load() == 0; });
    Client c(port);
    REQUIRE(c.send(get("/sse?n=100000&ms=5")));
    auto h = c.head();
    REQUIRE(h && h->status == 200);
    for (int i = 0; i < 10; ++i) REQUIRE(c.chunk());
    const int aborted = g.sse_aborted.load();
    const int written = g.sse_written.load();
    const int64_t t_close = now_us();
    if (rst) {
      c.reset();
    } else {
      c.close();
    }
    wait_for([&] { return g.sse_aborted.load() > aborted; });
    CHECK_EQ(g.sse_aborted.load(), aborted + 1);
    const double ms = double(g.sse_end_us.load() - t_close) / 1000.0;
    const int extra = g.sse_written.load() - written;
    CHECK(ms < 100);
    CHECK(extra <= 2);
    std::printf("  %s: handler returned %.1f ms after the close, %d event(s) written after it\n",
                rst ? "RST" : "FIN", ms, extra);
  }

  // A non-streaming handler polling client_gone() while it "waits for the GPU".
  {
    const int detected = g.gone_detected.load();
    Client c(port);
    REQUIRE(c.send(get("/wait_gone")));
    std::this_thread::sleep_for(50ms);
    const int64_t t_close = now_us();
    c.close();
    wait_for([&] { return g.gone_detected.load() > detected; });
    CHECK_EQ(g.gone_detected.load(), detected + 1);
    const double ms = double(g.gone_us.load() - t_close) / 1000.0;
    CHECK(ms < 50);
    std::printf("  client_gone() noticed a close after %.1f ms\n", ms);
  }
  // ...and not fooled by a pipelined request waiting behind it.
  {
    Client c(port);
    REQUIRE(c.send(get("/wait_gone?ms=200") + get("/hello")));
    auto r = c.response();
    CHECK(r && r->status == 200 && r->body == "still here");
    r = c.response();
    CHECK(r && r->status == 200 && r->body == "hello");
  }
}

static void test_concurrency(int port) {
  g.slow_max = 0;
  constexpr int kClients = 200;
  const auto t0 = Clock::now();
  std::vector<std::unique_ptr<Client>> cs;
  for (int i = 0; i < kClients; ++i) {
    cs.push_back(std::make_unique<Client>(port));
    REQUIRE(cs.back()->ok());
    REQUIRE(cs.back()->send(get("/slow?ms=300")));
  }
  int ok = 0;
  for (auto& c : cs) {
    auto r = c->response(10000);
    ok += r && r->status == 200 && r->body == "slow";
  }
  const double ms = ms_since(t0);
  CHECK_EQ(ok, kClients);
  CHECK(g.slow_max.load() >= kClients * 9 / 10);
  CHECK(ms < 3000);
  std::printf("  %d concurrent connections: %d ok, peak %d handlers at once, %.0f ms (each handler sleeps 300 ms)\n",
              kClients, ok, g.slow_max.load(), ms);
}

static void test_max_connections() {
  http::Options o = test_options();
  o.max_connections = 4;
  TestServer ts(o);
  std::vector<std::unique_ptr<Client>> idle;
  for (int i = 0; i < 4; ++i) idle.push_back(std::make_unique<Client>(ts.port()));
  wait_for([&] { return ts.srv.connections() == 4; });
  REQUIRE(ts.srv.connections() == 4);
  bool closed = false;
  auto r = once(ts.port(), get("/hello"), &closed);
  REQUIRE(r && r->status == 503);
  CHECK_EQ(r->header("retry-after"), "1");
  CHECK_EQ(error_of(*r).get_string("type", ""), "server_error");
  CHECK(closed);
  idle[0]->close();
  wait_for([&] { return ts.srv.connections() == 3; });
  r = once(ts.port(), get("/hello"));
  CHECK(r && r->status == 200);
}

static void test_timeouts() {
  http::Options o = test_options();
  o.header_timeout_s = 1;
  o.idle_timeout_s = 1;
  o.io_timeout_s = 1;
  TestServer ts(o);
  const int port = ts.port();
  // Run the four slow cases side by side.
  std::vector<std::thread> ts_threads;
  ts_threads.emplace_back([port] {  // slowloris: head never completes
    Client c(port);
    c.send("GET /hello HTTP/1.1\r\nHost: localhost\r\n");
    const auto t0 = Clock::now();
    auto r = c.response(4000);
    CHECK(r && r->status == 408);
    CHECK(c.closed());
    const double ms = ms_since(t0);
    CHECK(ms > 900 && ms < 2500);
  });
  ts_threads.emplace_back([port] {  // connects and says nothing
    Client c(port);
    const auto t0 = Clock::now();
    CHECK(c.closed(4000));
    CHECK(c.buffered().empty());
    const double ms = ms_since(t0);
    CHECK(ms > 900 && ms < 2500);
  });
  ts_threads.emplace_back([port] {  // idle keep-alive connection
    Client c(port);
    c.send(get("/hello"));
    auto r = c.response();
    CHECK(r && r->status == 200);
    const auto t0 = Clock::now();
    CHECK(c.closed(4000));
    CHECK(c.buffered().empty());
    const double ms = ms_since(t0);
    CHECK(ms > 900 && ms < 2500);
  });
  ts_threads.emplace_back([port] {  // body stalls
    Client c(port);
    c.send("POST /raw HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n0123456789");
    auto r = c.response(4000);
    CHECK(r && r->status == 408);
    CHECK(c.closed());
  });
  ts_threads.emplace_back([port] {  // stops reading a stream: the blocked write gives up
    const int done = g.flood_done.load();
    Client c(port);
    c.send(get("/flood"));
    auto h = c.head();
    CHECK(h && h->status == 200);
    wait_for([&] { return g.flood_done.load() > done; }, 5000ms);
    CHECK_EQ(g.flood_done.load(), done + 1);
    const double stall_ms = double(g.flood_stall_us.load()) / 1000.0;
    CHECK(stall_ms > 900 && stall_ms < 2500);
    std::printf("  stream to a client that stopped reading: write() gave up after %.0f ms (%lld KiB queued)\n",
                stall_ms, static_cast<long long>(g.flood_bytes.load() >> 10));
  });
  for (auto& t : ts_threads) t.join();
}

static void test_binding(int port) {
  bool threw = false;
  try {  // the port is taken by the main test server
    http::Options o;
    o.port = port;
    http::Server clash(o);
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("cannot listen") != std::string::npos;
  }
  CHECK(threw);

  // "::" is dual-stack: IPv6 and IPv4 clients (the latter seen as plain IPv4).
  http::Options o = test_options();
  o.host = "::";
  TestServer ts(o);
  auto r = once(ts.port(), get("/remote"));
  REQUIRE(r && r->status == 200);
  CHECK(r->body.starts_with("127.0.0.1:"));
  Client c6(ts.port(), /*ipv6=*/true);
  REQUIRE(c6.ok());
  REQUIRE(c6.send(get("/remote")));
  r = c6.response();
  REQUIRE(r && r->status == 200);
  CHECK(r->body.starts_with("[::1]:"));
}

// A port another program listens on fails at once, at any address of
// this machine (macOS would let both sockets listen with SO_REUSEADDR), and
// an address held only by an earlier run's connections in TIME_WAIT binds.
static void test_port_in_use() {
  auto clash = [](const char* first, const char* second) {
    http::Options a = test_options();
    a.host = first;
    TestServer one(a);
    http::Options b = test_options();
    b.host = second;
    b.port = one.port();
    int err = 0;
    std::string what;
    try {
      http::Server two(b);
    } catch (const http::ListenError& e) {
      err = e.error;
      what = e.what();
    }
    check_eq(err, EADDRINUSE, (std::string(first) + " then " + second).c_str(), "EADDRINUSE", __LINE__);
    CHECK(what.find("another program listens on port " + std::to_string(one.port())) != std::string::npos);
  };
  clash("127.0.0.1", "127.0.0.1");
  clash("0.0.0.0", "127.0.0.1");
  clash("127.0.0.1", "0.0.0.0");
  clash("127.0.0.1", "::");

  int port = 0;
  {  // the server closes first: its side of the connection stays in TIME_WAIT
    TestServer ts(test_options());
    port = ts.port();
    auto r = once(port, get("/hello", "Connection: close\r\n"));
    CHECK(r && r->status == 200);
  }
  http::Options o = test_options();
  o.port = port;
  bool bound = false;
  try {
    TestServer again(o);
    bound = again.port() == port;
    auto r = once(port, get("/hello"));
    CHECK(r && r->status == 200);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "  rebind: %s\n", e.what());
  }
  CHECK(bound);
}

// Behind the startup gate, the routes registered before gate() are
// served, every other request gets the gate's answer without its body being
// read; routes registered while it runs serve once it opens.
static void test_gate() {
  http::Server srv(test_options());
  srv.route("GET", "/early", [](const http::Request&, http::Response& res) { res.send(200, "text/plain", "early"); });
  srv.gate([](const http::Request& req, http::Response& res) {
    res.set_header("Retry-After", "1");
    res.send(503, "application/json", "{\"status\":\"loading\",\"path\":\"" + req.path + "\"}");
  });
  std::thread th([&] { srv.run(); });
  const int port = srv.port();
  auto r = once(port, get("/early"));
  CHECK(r && r->status == 200 && r->body == "early");
  r = once(port, get("/late"));
  CHECK(r && r->status == 503 && r->body == "{\"status\":\"loading\",\"path\":\"/late\"}" && r->header("retry-after") == "1");
  r = once(port, get("/unknown"));
  CHECK(r && r->status == 503);
  {
    Client c(port);
    REQUIRE(c.send("HEAD /late HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"));
    auto h = c.response(5000, /*head_request=*/true);
    CHECK(h && h->status == 503 && h->body.empty());
  }
  bool closed = false;
  r = once(port, post("/late", std::string(100000, ' ')), &closed);
  CHECK(r && r->status == 503 && closed);  // its body was never read
  {
    Client c(port);  // no body: the connection is kept
    REQUIRE(c.send(get("/late") + get("/early")));
    auto a = c.response(), b = c.response();
    CHECK(a && a->status == 503 && b && b->status == 200);
  }
  srv.route("GET", "/late", [](const http::Request&, http::Response& res) { res.send(200, "text/plain", "late"); });
  srv.route("POST", "/late", [](const http::Request& req, http::Response& res) {
    res.send(200, "text/plain", std::to_string(req.body.size()));
  });
  r = once(port, get("/late"));
  CHECK(r && r->status == 503);  // registered, not open yet
  srv.open();
  r = once(port, get("/late"));
  CHECK(r && r->status == 200 && r->body == "late");
  r = once(port, post("/late", "{}"));
  CHECK(r && r->status == 200 && r->body == "2");
  r = once(port, get("/unknown"));
  CHECK(r && r->status == 404);
  r = once(port, get("/early"));
  CHECK(r && r->status == 200);
  srv.stop();
  th.join();

  // With an API key: the key comes before the gate; keyless routes need none.
  http::Options o = test_options();
  o.api_key = "k3y";
  http::Server keyed(o);
  keyed.route_keyless("GET", "/page", [](const http::Request&, http::Response& res) { res.send(200, "text/html", "<p>"); });
  keyed.route("GET", "/early", [](const http::Request&, http::Response& res) { res.send(200, "text/plain", "early"); });
  keyed.gate([](const http::Request&, http::Response& res) { res.send(503, "text/plain", "loading"); });
  std::thread th2([&] { keyed.run(); });
  const int kp = keyed.port();
  r = once(kp, get("/page"));
  CHECK(r && r->status == 200 && r->body == "<p>");
  {
    Client c(kp);
    REQUIRE(c.send("HEAD /page HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"));
    auto h = c.response(5000, /*head_request=*/true);
    CHECK(h && h->status == 200 && h->body.empty());
  }
  r = once(kp, get("/early"));
  CHECK(r && r->status == 401);
  r = once(kp, get("/late"));
  CHECK(r && r->status == 401);
  r = once(kp, get("/late", "Authorization: Bearer k3y\r\n"));
  CHECK(r && r->status == 503 && r->body == "loading");
  keyed.route("GET", "/late", [](const http::Request&, http::Response& res) { res.send(200, "text/plain", "late"); });
  keyed.open();
  r = once(kp, get("/late"));
  CHECK(r && r->status == 401);
  r = once(kp, get("/late", "Authorization: Bearer k3y\r\n"));
  CHECK(r && r->status == 200);
  r = once(kp, get("/page"));
  CHECK(r && r->status == 200);
  keyed.stop();
  th2.join();

  // Without a key, a keyless page still refuses other origins.
  http::Server plain(test_options());
  plain.route_keyless("GET", "/page", [](const http::Request&, http::Response& res) { res.send(200, "text/html", "<p>"); });
  std::thread th3([&] { plain.run(); });
  const std::string me = "127.0.0.1:" + std::to_string(plain.port());
  r = once(plain.port(), "GET /page HTTP/1.1\r\nHost: " + me + "\r\nOrigin: http://evil.example\r\n\r\n");
  CHECK(r && r->status == 403);
  r = once(plain.port(), "GET /page HTTP/1.1\r\nHost: " + me + "\r\nOrigin: http://" + me + "\r\n\r\n");
  CHECK(r && r->status == 200);
  plain.stop();
  th3.join();
}

static void test_stop() {
  auto ts = std::make_unique<TestServer>(test_options());
  const int port = ts->port();
  Client stream(port);
  REQUIRE(stream.send(get("/sse?n=100000&ms=5")));
  auto h = stream.head();
  REQUIRE(h && h->status == 200);
  for (int i = 0; i < 5; ++i) REQUIRE(stream.chunk());
  Client idle(port);
  REQUIRE(idle.send(get("/hello")));
  auto r = idle.response();
  REQUIRE(r && r->status == 200);
  wait_for([] { return g.sse_running.load() == 1; });

  const int aborted = g.sse_aborted.load();
  const auto t0 = Clock::now();
  ts->srv.stop();
  ts->th.join();  // run() returns once every handler has
  const double ms = ms_since(t0);
  CHECK(ms < 1000);
  CHECK_EQ(g.sse_aborted.load(), aborted + 1);
  bool terminated = false;  // abandoned, not ended: no zero-length chunk
  while (auto ch = stream.chunk(1000))
    if (ch->empty()) terminated = true;
  CHECK(!terminated);
  CHECK(stream.closed());
  CHECK(idle.closed());
  Client late(port);
  CHECK(!late.ok());  // listener closed
  std::printf("  stop() with a live stream and an idle connection: run() returned after %.1f ms\n", ms);
}

static void test_large_bodies(int port) {
  {
    const std::string body = pattern(48 << 20);
    Client c(port);
    const auto t0 = Clock::now();
    REQUIRE(c.send("POST /big HTTP/1.1\r\nHost: localhost\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n"));
    REQUIRE(c.send(body));
    auto r = c.response(20000);
    REQUIRE(r && r->status == 200);
    const double ms = ms_since(t0);
    json::Value v = json::parse(r->body);
    CHECK_EQ(v.get_int("bytes", 0), int64_t(body.size()));
    CHECK_EQ(v.get_string("fnv", ""), std::to_string(fnv1a(body)));
    std::printf("  48 MiB upload: %.0f ms (%.0f MiB/s)\n", ms, 48 / (ms / 1000));
  }
  {
    Client c(port);
    const auto t0 = Clock::now();
    REQUIRE(c.send(get("/big?n=33554432")));
    auto r = c.response(20000);
    REQUIRE(r && r->status == 200);
    const double ms = ms_since(t0);
    CHECK(r->body == pattern(32 << 20));
    std::printf("  32 MiB download: %.0f ms\n", ms);
  }
}

// ------------------------------------------------------------------ main

static void run(const char* name, const std::function<void()>& fn) {
  const int before = g_failures.load();
  std::printf("%s\n", name);
  std::fflush(stdout);
  const auto t0 = Clock::now();
  fn();
  std::printf("  -> %s (%.0f ms)\n", g_failures.load() == before ? "ok" : "FAILED", ms_since(t0));
  std::fflush(stdout);
}

static void raise_fd_limit() {
  rlimit rl{};
  if (::getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < 4096) {
    rl.rlim_cur = std::min<rlim_t>(4096, rl.rlim_max);
    ::setrlimit(RLIMIT_NOFILE, &rl);
  }
}

static http::Server* g_server = nullptr;
static void on_signal(int) {
  if (g_server) g_server->stop();  // async-signal-safe
}

static int serve_main(int argc, char** argv) {
  http::Options o;
  o.port = argc > 2 ? std::atoi(argv[2]) : 8080;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--log") o.log_requests = true;
    if (a == "--cors") o.cors = true;
    if (a == "--key" && i + 1 < argc) o.api_key = argv[++i];
  }
  http::Server srv(o);
  add_routes(srv);
  g_server = &srv;
  ::signal(SIGINT, on_signal);
  ::signal(SIGTERM, on_signal);
  std::printf("listening on %d\n", srv.port());
  std::fflush(stdout);
  srv.run();
  std::fprintf(stderr, "server stopped\n");
  return 0;
}

int main(int argc, char** argv) {
  raise_fd_limit();
  if (argc > 1 && std::string(argv[1]) == "--serve") return serve_main(argc, argv);

  TestServer main_server(test_options());
  const int port = main_server.port();
  const auto t0 = Clock::now();
  run("parser units", test_parse_units);
  run("basic GET/POST, JSON echo", [&] { test_basic(port); });
  run("keep-alive", [&] { test_keepalive(port); });
  run("HTTP/1.0", [&] { test_http10(port); });
  run("pipelining", [&] { test_pipelining(port); });
  run("chunked request bodies", [&] { test_chunked_request(port); });
  run("Expect: 100-continue", [&] { test_expect_continue(port); });
  run("limits: 413 / 431 / 414", [&] { test_limits(port); });
  run("malformed requests", [&] { test_malformed(port); });
  run("routing: 404 / 405 / HEAD / prefix / query", [&] { test_routing(port); });
  run("handler errors", [&] { test_handler_errors(port); });
  run("auth and CORS", test_auth_cors);
  run("no key: Host / Origin guard (DNS rebinding, cross-site)", test_local_guard);
  run("body types: 415", [&] { test_body_types(port); });
  run("SSE streaming", [&] { test_sse(port); });
  run("client disconnect detection", [&] { test_disconnect(port); });
  run("200 concurrent connections", [&] { test_concurrency(port); });
  run("max_connections -> 503", test_max_connections);
  run("timeouts", test_timeouts);
  run("binding: port in use, dual stack", [&] { test_binding(port); });
  run("port in use at another address, TIME_WAIT rebind", test_port_in_use);
  run("startup gate, keyless routes", test_gate);
  run("graceful stop", test_stop);
  run("large bodies", [&] { test_large_bodies(port); });
  std::printf("%d checks, %d failures, %.1f s\n", g_checks.load(), g_failures.load(), ms_since(t0) / 1000);
  return g_failures.load() ? 1 : 0;
}
