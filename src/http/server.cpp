// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// HTTP/1.1 server: listener, connection threads, request loop, responses.
//
// Threading model: the thread that calls run() accepts connections, and each
// connection is served by its own thread, bounded by max_connections.
// Why this and not kqueue + a worker pool:
//  * Handlers are synchronous and long-lived: they wait for the GPU scheduler
//    and then stream tokens for minutes. Under an event loop every in-flight
//    request would still pin a worker thread (or the handler API would have
//    to become callbacks/coroutines), so a pool would only add a second queue
//    and cross-thread handoffs.
//  * Concurrency is small: hundreds of connections at most, because the GPU,
//    not the socket layer, is the bottleneck. A parked thread costs the few
//    stack pages it has touched plus a kernel thread; 512 are cheap.
//  * Everything about a request (buffers, parsed request, response state)
//    lives on one thread: no locks or handoffs on the hot path, and a slow or
//    malicious client can only ever stall its own thread.
//  * Straight-line code over non-blocking sockets and poll() deadlines gives
//    per-phase timeouts (head, idle, I/O stall) without a timer wheel, and
//    every wait also polls a "stop pipe" that stop() makes readable forever,
//    which wakes all connections at once for a graceful shutdown.
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "http.h"
#include "parse.h"

namespace espetero::http {
namespace detail {

using Clock = std::chrono::steady_clock;

namespace {

constexpr size_t kMaxHeaderFields = 100;
constexpr size_t kMaxChunkLine = 4096;  // chunk-size line including extensions
constexpr int kMaxRejecting = 32;       // short-lived threads answering 503 beyond max_connections
constexpr size_t kReadChunk = 64 << 10;
constexpr size_t kKeepBodyCapacity = 1 << 20;  // bigger body buffers are freed between requests
constexpr int kGated = -1;                     // an "early" answer by the startup gate
constexpr auto kLingerTime = std::chrono::seconds(2);
constexpr size_t kLingerBytes = 64 << 20;

#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;  // SO_NOSIGPIPE, set on every socket, covers it
#endif

Clock::time_point after_seconds(int s) {
  return s > 0 ? Clock::now() + std::chrono::seconds(s) : Clock::time_point::max();
}

int poll_timeout(Clock::time_point deadline) {
  if (deadline == Clock::time_point::max()) return -1;
  const auto ms = std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now()).count();
  return ms <= 0 ? 0 : int(std::min<long long>(ms, INT_MAX));
}

void configure_socket(int fd) {
  const int one = 1;
  // Never raise SIGPIPE for a vanished client (MSG_NOSIGNAL too, belt and braces).
  ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
  // Each response head+body leaves in one sendmsg and each stream event must
  // go out at once, not wait for the ACK of the previous one: no Nagle.
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  // Notice peers that vanished without FIN/RST (sleeping laptop, dropped NAT
  // entry) while a handler waits on the GPU: client_gone() then sees the error.
  ::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
  const int idle = 60, interval = 10, count = 3;
  ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPALIVE, &idle, sizeof idle);
  ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &interval, sizeof interval);
  ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &count, sizeof count);
  ::fcntl(fd, F_SETFD, FD_CLOEXEC);
  ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL) | O_NONBLOCK);
}

std::string peer_name(const sockaddr_storage& ss) {
  char host[INET6_ADDRSTRLEN] = "?";
  if (ss.ss_family == AF_INET) {
    const auto* a = reinterpret_cast<const sockaddr_in*>(&ss);
    ::inet_ntop(AF_INET, &a->sin_addr, host, sizeof host);
    return std::string(host) + ":" + std::to_string(ntohs(a->sin_port));
  }
  if (ss.ss_family == AF_INET6) {
    const auto* a = reinterpret_cast<const sockaddr_in6*>(&ss);
    if (IN6_IS_ADDR_V4MAPPED(&a->sin6_addr)) {  // IPv4 client on a dual-stack listener
      ::inet_ntop(AF_INET, &a->sin6_addr.s6_addr[12], host, sizeof host);
      return std::string(host) + ":" + std::to_string(ntohs(a->sin6_port));
    }
    ::inet_ntop(AF_INET6, &a->sin6_addr, host, sizeof host);
    return "[" + std::string(host) + "]:" + std::to_string(ntohs(a->sin6_port));
  }
  return "?";
}

}  // namespace

// Outcome of an I/O step or of reading part of a request.
enum class Io { Ok, Timeout, Closed, Stopped, Error, TooLarge, Malformed };

struct Route {
  std::string method, path;
  bool prefix = false;
  Handler handler;
  Body body = Body::Any;
  bool keyless = false;  // served without the API key (route_keyless)
};

struct ServerState {
  Options opt;
  int listen_fd = -1, stop_rd = -1, stop_wr = -1;
  int port = 0;
  std::atomic<bool> stopping{false};
  std::vector<Route> routes;  // written before run(), read-only afterwards
  // The startup gate. Routes registered after gate() go to `late`,
  // written while the server runs but read only once `gated` is false:
  // open() stores false (release) after the last registration and readers
  // load it (acquire) before they look at `late`, so they see it complete.
  std::vector<Route> late;
  Route gate;
  bool gate_set = false;  // gate() was called (the registering thread's own flag)
  std::atomic<bool> gated{false};
  mutable std::mutex mu;
  std::condition_variable cv;
  int active = 0;     // connection threads (guarded by mu)
  int rejecting = 0;  // 503 threads (guarded by mu)

  ~ServerState() {
    for (int fd : {listen_fd, stop_rd, stop_wr})
      if (fd >= 0) ::close(fd);
  }
  // `with_late`: also the routes registered behind the gate (only once it is open).
  const Route* find(std::string_view method, std::string_view path, std::string* allow, bool with_late) const;
  void accept_ready(bool& backoff);
  bool spawn(int fd, std::string remote, bool reject);
  void thread_done(bool reject);
};

// One client connection, served by one thread.
struct Conn {
  ServerState& s;
  int fd;
  std::string remote;
  std::string in;     // bytes received but not consumed yet (from in_pos on)
  size_t in_pos = 0;
  std::atomic<bool> gone{false};  // peer closed/reset, or a write failed
  // Per request:
  bool keep_alive = true;
  bool http10 = false;
  bool head_req = false;

  Conn(ServerState& st, int f, std::string r) : s(st), fd(f), remote(std::move(r)) {}
  ~Conn() {
    if (fd >= 0) ::close(fd);
  }
  Conn(const Conn&) = delete;
  Conn& operator=(const Conn&) = delete;

  bool stopping() const { return s.stopping.load(std::memory_order_relaxed); }
  size_t pending() const { return in.size() - in_pos; }
  void compact() {
    if (in_pos) {
      in.erase(0, in_pos);
      in_pos = 0;
    }
  }

  Io wait(short events, Clock::time_point deadline);
  Io recv_some(char* p, size_t cap, Clock::time_point deadline, size_t& got);
  Io fill(Clock::time_point deadline);
  Io send_all(iovec* iov, int n);
  bool peer_gone();

  Io read_head(bool first, size_t& len, int& status);
  Io read_line(std::string_view& line, size_t max_len);
  Io read_fixed(uint64_t n, std::string& body);
  Io read_chunked(std::string& body);

  std::string build_head(const Response& r, int status, std::string_view type, int64_t length, bool chunked,
                         bool stream);
  bool authorized(const Request& req) const;
  std::string guard(const Request& req) const;
  void answer_early(const Request& req, Response& res, int status, const std::string& detail);
  void invoke(const Route& route, const Request& req, Response& res);
  void log(const Request& req, const Response& res, Clock::time_point t0) const;
  void serve();
  void reject();
  void close_socket(bool linger);
};

// ------------------------------------------------------------------ I/O

// Waits until fd is ready for `events`, the deadline passes or the server
// stops. Readiness wins over stopping so that finished work still drains.
Io Conn::wait(short events, Clock::time_point deadline) {
  for (;;) {
    pollfd p[2] = {{fd, events, 0}, {s.stop_rd, POLLIN, 0}};
    const int r = ::poll(p, 2, poll_timeout(deadline));
    if (r < 0) {
      if (errno == EINTR) continue;
      return Io::Error;
    }
    if (p[0].revents) return Io::Ok;  // ready, or an error that the next call reports
    if (p[1].revents) return Io::Stopped;
    if (Clock::now() >= deadline) return Io::Timeout;
  }
}

Io Conn::recv_some(char* p, size_t cap, Clock::time_point deadline, size_t& got) {
  got = 0;
  for (;;) {
    const ssize_t n = ::recv(fd, p, cap, 0);
    if (n > 0) {
      got = size_t(n);
      return Io::Ok;
    }
    if (n == 0) {
      gone = true;
      return Io::Closed;
    }
    if (errno == EINTR) continue;
    if (errno != EAGAIN && errno != EWOULDBLOCK) {
      gone = true;
      return Io::Error;
    }
    if (const Io w = wait(POLLIN, deadline); w != Io::Ok) return w;
  }
}

Io Conn::fill(Clock::time_point deadline) {
  char buf[kReadChunk];
  size_t got = 0;
  const Io r = recv_some(buf, sizeof buf, deadline, got);
  in.append(buf, got);
  return r;
}

// Writes all of iov. Times out only when the client stops reading for
// io_timeout_s (a slow but moving transfer is fine).
Io Conn::send_all(iovec* iov, int n) {
  auto deadline = after_seconds(s.opt.io_timeout_s);
  while (n > 0) {
    if (iov->iov_len == 0) {
      ++iov;
      --n;
      continue;
    }
    msghdr m{};
    m.msg_iov = iov;
    m.msg_iovlen = n;
    const ssize_t sent = ::sendmsg(fd, &m, kSendFlags);
    if (sent >= 0) {
      size_t left = size_t(sent);
      while (n > 0 && left >= iov->iov_len) {
        left -= iov->iov_len;
        ++iov;
        --n;
      }
      if (n > 0) {
        iov->iov_base = static_cast<char*>(iov->iov_base) + left;
        iov->iov_len -= left;
      }
      deadline = after_seconds(s.opt.io_timeout_s);
      continue;
    }
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      if (const Io w = wait(POLLOUT, deadline); w != Io::Ok) {
        gone = true;  // a partial message cannot be resumed later
        return w;
      }
      continue;
    }
    gone = true;  // EPIPE, ECONNRESET, ...
    return Io::Error;
  }
  return Io::Ok;
}

// Non-blocking disconnect check: EOF or an error on the read side means the
// client is gone; readable data is a pipelined request, which is only peeked
// at, never consumed. Thread-safe (the socket calls are, and `gone` is atomic).
bool Conn::peer_gone() {
  if (gone.load(std::memory_order_relaxed) || stopping()) return true;
  pollfd p{fd, POLLIN, 0};
  if (::poll(&p, 1, 0) <= 0) return false;  // nothing readable: still connected
  if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) {
    gone = true;
    return true;
  }
  char c;
  const ssize_t n = ::recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
  if (n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) {
    gone = true;
    return true;
  }
  return false;
}

// ------------------------------------------------------------------ reading requests

// Buffers a complete request head; len is its size (from in_pos). On failure,
// status is the answer the client deserves (408, 414, 431) or 0 for a silent
// close (idle timeout, EOF between requests, stop).
Io Conn::read_head(bool first, size_t& len, int& status) {
  status = 0;
  // A new connection must start its request within header_timeout; a kept-alive
  // one may idle longer. Once the first byte arrives, the head must complete
  // within header_timeout (slowloris).
  Clock::time_point deadline = after_seconds(first ? s.opt.header_timeout_s : s.opt.idle_timeout_s);
  bool started = false;
  size_t scan = 0;
  for (;;) {
    if (!started) in_pos += skip_empty_lines(std::string_view(in).substr(in_pos));
    const std::string_view buf(in.data() + in_pos, pending());
    if (!buf.empty()) {
      if (!started) {
        started = true;
        deadline = after_seconds(s.opt.header_timeout_s);
      }
      len = find_head_end(buf, scan);
      if ((len ? len : buf.size()) > s.opt.max_header) {
        const size_t eol = buf.find('\n');
        status = (eol == std::string_view::npos || eol > s.opt.max_header) ? 414 : 431;
        return Io::TooLarge;
      }
      if (len) return Io::Ok;
    }
    compact();
    // Between keep-alive requests the next one has rarely arrived yet: wait
    // for it first rather than pay for a recv that only says EAGAIN.
    if (!started && !first)
      if (const Io w = wait(POLLIN, deadline); w != Io::Ok) return w;
    const Io r = fill(deadline);
    if (r == Io::Timeout && started) status = 408;
    if (r != Io::Ok) return r;
  }
}

// One line (without its CRLF or LF) from the buffer, reading more as needed.
// The view is valid until the buffer changes.
Io Conn::read_line(std::string_view& line, size_t max_len) {
  size_t from = in_pos;
  for (;;) {
    const size_t nl = in.find('\n', from);
    if (nl != std::string::npos) {
      size_t end = nl;
      if (end > in_pos && in[end - 1] == '\r') --end;
      if (end - in_pos > max_len) return Io::Malformed;
      line = std::string_view(in).substr(in_pos, end - in_pos);
      in_pos = nl + 1;
      return Io::Ok;
    }
    if (pending() > max_len + 1) return Io::Malformed;
    compact();
    from = in.size();
    if (const Io r = fill(after_seconds(s.opt.io_timeout_s)); r != Io::Ok) return r;
  }
}

Io Conn::read_fixed(uint64_t n, std::string& body) {
  // Reserving the whole body costs address space only: pages are touched as
  // bytes arrive, so announcing a big body and sending nothing costs nothing.
  body.reserve(size_t(n));
  const size_t buffered = size_t(std::min<uint64_t>(n, pending()));
  body.assign(in, in_pos, buffered);
  in_pos += buffered;
  // The rest goes straight from the socket into the body, and never past it:
  // what follows belongs to the next (pipelined) request.
  size_t filled = body.size();
  auto deadline = after_seconds(s.opt.io_timeout_s);
  while (filled < n) {
    if (filled == body.size())  // grow the writable window geometrically
      body.resize(size_t(std::min<uint64_t>(n, filled + std::max<size_t>(filled, 256 << 10))));
    size_t got = 0;
    if (const Io r = recv_some(body.data() + filled, body.size() - filled, deadline, got); r != Io::Ok) {
      body.resize(filled);
      return r;
    }
    filled += got;
    deadline = after_seconds(s.opt.io_timeout_s);
  }
  return Io::Ok;
}

// chunked-body = *chunk last-chunk trailer-section CRLF (RFC 9112 7.1).
Io Conn::read_chunked(std::string& body) {
  const uint64_t max = s.opt.max_body;
  std::string_view line;
  for (;;) {
    if (const Io r = read_line(line, kMaxChunkLine); r != Io::Ok) return r;
    uint64_t size = 0;
    if (!parse_chunk_size(line, size)) return Io::Malformed;
    if (size == 0) break;
    if (size > max - body.size()) return Io::TooLarge;
    for (uint64_t left = size; left > 0;) {
      if (pending() == 0) {
        compact();
        if (const Io r = fill(after_seconds(s.opt.io_timeout_s)); r != Io::Ok) return r;
      }
      const size_t take = size_t(std::min<uint64_t>(left, pending()));
      body.append(in, in_pos, take);
      in_pos += take;
      left -= take;
    }
    if (const Io r = read_line(line, 0); r != Io::Ok) return r;  // CRLF after the data
  }
  // Trailer fields are read and dropped, never merged into the headers.
  size_t trailer_bytes = 0;
  for (;;) {
    if (const Io r = read_line(line, s.opt.max_header); r != Io::Ok) return r;
    if (line.empty()) return Io::Ok;
    if ((trailer_bytes += line.size()) > s.opt.max_header) return Io::Malformed;
  }
}

// ------------------------------------------------------------------ responses

std::string Conn::build_head(const Response& r, int status, std::string_view type, int64_t length, bool chunked,
                             bool stream) {
  if (stopping()) keep_alive = false;  // tell the client not to reuse the connection
  std::string h;
  h.reserve(192 + type.size() + r.headers_.size() * 64);
  h.append("HTTP/1.1 ").append(std::to_string(status)).append(" ").append(status_reason(status)).append("\r\n");
  h.append("Date: ").append(http_date()).append("\r\n");
  if (!type.empty()) h.append("Content-Type: ").append(type).append("\r\n");
  if (chunked) {
    h.append("Transfer-Encoding: chunked\r\n");
  } else if (length >= 0) {
    h.append("Content-Length: ").append(std::to_string(length)).append("\r\n");
  }
  if (stream) {
    if (!r.has_header("cache-control")) h.append("Cache-Control: no-cache\r\n");
    h.append("X-Accel-Buffering: no\r\n");  // reverse proxies (nginx) must not buffer events
  }
  if (!keep_alive) {
    h.append("Connection: close\r\n");
  } else if (http10) {
    h.append("Connection: keep-alive\r\n");
  }
  if (s.opt.cors && !r.has_header("access-control-allow-origin")) h.append("Access-Control-Allow-Origin: *\r\n");
  for (const auto& [k, v] : r.headers_) h.append(k).append(": ").append(v).append("\r\n");
  h.append("\r\n");
  return h;
}

bool Conn::authorized(const Request& req) const {
  const std::string& key = s.opt.api_key;
  if (key.empty()) return true;
  std::string_view token;
  if (const std::string* h = req.header("authorization")) {
    if (h->size() < 7 || !iequals(std::string_view(*h).substr(0, 7), "bearer ")) return false;
    token = trim_ows(std::string_view(*h).substr(7));
  } else if (const std::string* x = req.header("x-api-key")) {  // Anthropic clients
    token = trim_ows(std::string_view(*x));
  } else {
    return false;
  }
  if (token.size() != key.size()) return false;
  unsigned char diff = 0;  // constant time: timing must not reveal how much of the key matched
  for (size_t i = 0; i < key.size(); ++i) diff |= static_cast<unsigned char>(token[i] ^ key[i]);
  return diff == 0;
}

// Browser protection for a server without an API key (Options::local_guard).
// A web page can reach a server on 127.0.0.1 in two ways: DNS rebinding (its
// own host name re-resolved to 127.0.0.1, so its requests carry that name in
// Host) and cross-site requests (the page posts to http://127.0.0.1:port and
// the browser sends its Origin). Non-browser clients send neither a foreign
// Host nor an Origin. Returns why a request is refused (403), "" to serve it.
std::string Conn::guard(const Request& req) const {
  if (!s.opt.local_guard || !s.opt.api_key.empty()) return {};
  const std::string* host = req.header("host");
  if (host && !host_allowed(*host, s.opt.host))
    return "Host '" + *host +
           "' refused: without an API key this server only answers requests addressed to localhost, 127.0.0.1, "
           "[::1] or its bind address (protection against DNS rebinding); start it with --api-key to serve other "
           "host names";
  const std::string* origin = req.header("origin");
  if (origin && !trim_ows(*origin).empty() && !s.opt.cors && !(host && same_origin(*origin, *host)))
    return "Cross-origin request from '" + *origin +
           "' refused: this server has no API key and does not allow other origins (--cors allows them)";
  return {};
}

// Answers decided from the head alone, before (and without) reading a body.
void Conn::answer_early(const Request& req, Response& res, int status, const std::string& detail) {
  switch (status) {
    case 204: {  // CORS preflight (browsers never send credentials with it)
      res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS");
      const std::string* asked = req.header("access-control-request-headers");
      // "*" does not cover Authorization, so echo what the browser asks for.
      res.set_header("Access-Control-Allow-Headers", asked && !asked->empty() ? *asked : "Authorization, Content-Type");
      res.set_header("Access-Control-Max-Age", "86400");
      res.send(204, {}, {});
      return;
    }
    case 401:
      res.set_header("WWW-Authenticate", "Bearer");
      res.send_error(401, "Incorrect or missing API key: expected 'Authorization: Bearer <key>' or 'x-api-key: <key>'",
                     "invalid_request_error", "invalid_api_key");
      return;
    case 404:
      res.send_error(404, "Invalid URL (" + req.method + " " + req.path + ")");
      return;
    case 405:
      res.set_header("Allow", detail);
      res.send_error(405, "Method " + req.method + " is not allowed for " + req.path + " (allowed: " + detail + ")");
      return;
    default:
      res.send_error(status, detail);
      return;
  }
}

void Conn::invoke(const Route& route, const Request& req, Response& res) {
  int status = 0;
  std::string message, type, code;
  try {
    route.handler(req, res);
    return;
  } catch (const Error& e) {
    status = (e.status >= 400 && e.status <= 599) ? e.status : 500;
    message = e.what();
    type = e.type;
    code = e.code;
  } catch (const json::Error& e) {  // malformed or mistyped request JSON
    status = 400;
    message = std::string("Invalid JSON body: ") + e.what();
  } catch (const std::exception& e) {
    status = 500;
    message = e.what();
  } catch (...) {
    status = 500;
    message = "unknown exception";
  }
  if (status >= 500)  // a bug or an engine failure: always worth a line
    std::fprintf(stderr, "http: %s %s: handler failed: %s\n", req.method.c_str(), req.path.c_str(), message.c_str());
  if (!res.started()) {
    res.send_error(status, message, type, code);
  } else if (res.phase_ == Response::Phase::Streaming) {
    // Abandon the stream without its terminating chunk: the client must not
    // mistake a failed stream for a complete one.
    res.phase_ = Response::Phase::Failed;
  }
}

void Conn::log(const Request& req, const Response& res, Clock::time_point t0) const {
  if (!s.opt.log_requests) return;
  const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  std::fprintf(stderr, "http %s %s %s %d %lluB %.1fms%s\n", remote.c_str(),
               req.method.empty() ? "-" : req.method.c_str(), req.path.empty() ? "-" : req.path.c_str(),
               res.status_ ? res.status_ : 499, static_cast<unsigned long long>(res.bytes_), ms,
               res.phase_ == Response::Phase::Failed ? " aborted" : "");
}

// ------------------------------------------------------------------ connection loop

void Conn::serve() {
  Request req;
  req.remote = remote;
  bool first = true, linger = false;
  while (!stopping()) {
    req.method.clear();
    req.target.clear();
    req.path.clear();
    req.query.clear();
    req.headers.clear();
    if (req.body.capacity() > kKeepBodyCapacity) {
      std::string().swap(req.body);
    } else {
      req.body.clear();
    }
    req.minor_version = 1;
    keep_alive = true;
    http10 = head_req = false;

    size_t head_len = 0;
    int status = 0;
    const Io r = read_head(first, head_len, status);
    first = false;
    const auto t0 = Clock::now();
    if (r != Io::Ok) {
      if (status) {
        keep_alive = false;
        linger = true;
        Response res(*this);
        res.send_error(status, status == 408   ? "Timed out reading the request head"
                               : status == 414 ? "Request line too long"
                                               : "Request header fields too large");
        log(req, res, t0);
      }
      break;
    }
    const Head h = parse_head(std::string_view(in.data() + in_pos, head_len), req, kMaxHeaderFields);
    in_pos += head_len;
    http10 = req.minor_version == 0;
    head_req = req.method == "HEAD";
    keep_alive = h.keep_alive;
    if (h.error) {  // the framing is unknown: answer and close
      keep_alive = false;
      linger = true;
      Response res(*this);
      res.send_error(h.error, h.message);
      log(req, res, t0);
      break;
    }

    // Answers that need no body: browser guard, CORS preflight, auth,
    // startup gate, routing, size limit, body type.
    const bool has_body = h.chunked || h.content_length > 0;
    const bool gated = s.gated.load(std::memory_order_acquire);  // the startup gate
    const Route* route = nullptr;
    std::string detail;
    int early = 0;
    if (detail = guard(req); !detail.empty()) {
      early = 403;
    } else if (s.opt.cors && req.method == "OPTIONS" && !s.find("OPTIONS", req.path, nullptr, !gated)) {
      early = 204;
    } else if (route = s.find(req.method, req.path, &detail, !gated); !(route && route->keyless) && !authorized(req)) {
      early = 401;
    } else if (!route) {
      early = gated ? kGated : detail.empty() ? 404 : 405;  // behind the gate the route may be on its way
    } else if (h.content_length > s.opt.max_body) {
      early = 413;
      detail = "Request body of " + std::to_string(h.content_length) + " bytes exceeds the limit of " +
               std::to_string(s.opt.max_body) + " bytes";
    } else if (has_body && route->body != Body::Any) {
      const std::string* ct = req.header("content-type");
      const std::string type = ct ? media_type(*ct) : std::string();
      const char* want = route->body == Body::Json ? "application/json" : "multipart/form-data";
      if (type != want) {
        early = 415;
        detail = "Content-Type " + (type.empty() ? std::string("missing") : "'" + type + "'") + ": " + req.method +
                 " " + req.path + " takes " + want;
      }
    }
    if (early) {
      // An unread body leaves the connection out of sync: close after the
      // answer. With Expect: 100-continue the client then never sends it.
      if (has_body) {
        keep_alive = false;
        linger = true;
      }
      Response res(*this);
      if (early == kGated) {
        invoke(s.gate, req, res);  // the body was not read
        if (res.phase_ == Response::Phase::Idle) res.send_error(503, "The server is starting", "server_error");
        if (res.phase_ == Response::Phase::Streaming) res.end();
      } else {
        answer_early(req, res, early, detail);
      }
      log(req, res, t0);
      if (!keep_alive || res.phase_ != Response::Phase::Done) break;
      continue;
    }

    if (h.expect_continue && has_body && pending() == 0) {  // the client waits for our go-ahead
      static constexpr char k100[] = "HTTP/1.1 100 Continue\r\n\r\n";
      iovec iov{const_cast<char*>(k100), sizeof k100 - 1};
      if (send_all(&iov, 1) != Io::Ok) break;
    }
    if (const Io b = h.chunked ? read_chunked(req.body) : read_fixed(h.content_length, req.body); b != Io::Ok) {
      const int st = b == Io::TooLarge ? 413 : b == Io::Malformed ? 400 : b == Io::Timeout ? 408 : 0;
      if (st) {
        keep_alive = false;
        linger = true;
        Response res(*this);
        const std::string limit = std::to_string(s.opt.max_body);
        res.send_error(st, st == 413   ? "Request body exceeds the limit of " + limit + " bytes"
                           : st == 400 ? std::string("Malformed chunked request body")
                                       : std::string("Timed out reading the request body"));
        log(req, res, t0);
      }
      break;
    }

    Response res(*this);
    invoke(*route, req, res);
    if (res.phase_ == Response::Phase::Idle) {
      if (res.client_gone()) {
        keep_alive = false;  // nobody left to answer
      } else {
        res.send_error(500, "The handler returned without a response", "server_error");
      }
    }
    if (res.phase_ == Response::Phase::Streaming) res.end();
    if (res.phase_ != Response::Phase::Done) keep_alive = false;
    log(req, res, t0);
    if (!keep_alive) break;
  }
  close_socket(linger);
}

// Over max_connections: a short-lived thread answers 503 and closes.
void Conn::reject() {
  // Read the request first (briefly): closing with unread data resets the
  // connection, and the client would probably never see the 503.
  const auto deadline = Clock::now() + std::chrono::seconds(1);
  size_t scan = 0;
  while (!find_head_end(in, scan) && in.size() < s.opt.max_header && fill(deadline) == Io::Ok) {
  }
  keep_alive = false;
  Request req;
  Response res(*this);
  res.set_header("Retry-After", "1");
  res.send_error(503, "Too many connections (limit " + std::to_string(s.opt.max_connections) + "); retry later",
                 "server_error");
  log(req, res, Clock::now());
  close_socket(true);
}

void Conn::close_socket(bool linger) {
  if (fd < 0) return;
  if (linger && !gone && !stopping()) {
    // Lingering close: stop sending, then drain what the client still sends
    // (typically a body we refused) until it closes or a short deadline.
    // Closing with unread data makes the kernel send RST, which can destroy
    // our error response before the client has read it.
    ::shutdown(fd, SHUT_WR);
    const auto deadline = Clock::now() + kLingerTime;
    char buf[16 << 10];
    size_t total = 0, got = 0;
    while (total < kLingerBytes && recv_some(buf, sizeof buf, deadline, got) == Io::Ok) total += got;
  }
  ::close(fd);
  fd = -1;
}

// ------------------------------------------------------------------ routing, accepting

const Route* ServerState::find(std::string_view method, std::string_view path, std::string* allow, bool with_late) const {
  const Route* best = nullptr;
  const Route* get_for_head = nullptr;
  bool path_known = false;
  std::vector<std::string_view> methods;
  auto better = [](const Route& a, const Route* b) {
    if (!b) return true;
    if (a.prefix != b->prefix) return !a.prefix;  // exact beats prefix
    return a.path.size() > b->path.size();        // longer prefix wins
  };
  for (const std::vector<Route>* table : {&routes, with_late ? &late : nullptr}) {
    if (!table) continue;
    for (const Route& r : *table) {
      if (r.prefix ? !path.starts_with(r.path) : path != r.path) continue;
      path_known = true;
      if (r.method == method) {
        if (better(r, best)) best = &r;
      } else if (method == "HEAD" && r.method == "GET") {
        if (better(r, get_for_head)) get_for_head = &r;
      }
      if (std::find(methods.begin(), methods.end(), r.method) == methods.end()) methods.push_back(r.method);
    }
  }
  if (best) return best;
  if (get_for_head) return get_for_head;
  if (allow && path_known) {
    if (std::find(methods.begin(), methods.end(), "GET") != methods.end() &&
        std::find(methods.begin(), methods.end(), "HEAD") == methods.end())
      methods.push_back("HEAD");
    if (opt.cors && std::find(methods.begin(), methods.end(), "OPTIONS") == methods.end()) methods.push_back("OPTIONS");
    for (std::string_view m : methods) {
      if (!allow->empty()) *allow += ", ";
      *allow += m;
    }
  }
  return nullptr;
}

// Whether a program accepts TCP connections at `addr`'s port: on that
// address, or, for a wildcard address, on 127.0.0.1 or ::1 (a connect with
// a 500 ms limit each).
bool port_answers(const sockaddr* addr) {
  std::vector<sockaddr_storage> probes;
  auto add4 = [&](in_addr a, in_port_t port) {
    sockaddr_storage ss{};
    auto* p = reinterpret_cast<sockaddr_in*>(&ss);
    p->sin_len = sizeof(sockaddr_in);
    p->sin_family = AF_INET;
    p->sin_port = port;
    p->sin_addr = a;
    probes.push_back(ss);
  };
  auto add6 = [&](in6_addr a, in_port_t port) {
    sockaddr_storage ss{};
    auto* p = reinterpret_cast<sockaddr_in6*>(&ss);
    p->sin6_len = sizeof(sockaddr_in6);
    p->sin6_family = AF_INET6;
    p->sin6_port = port;
    p->sin6_addr = a;
    probes.push_back(ss);
  };
  in_addr lo4{};
  lo4.s_addr = htonl(INADDR_LOOPBACK);
  if (addr->sa_family == AF_INET) {
    const auto* a = reinterpret_cast<const sockaddr_in*>(addr);
    if (a->sin_addr.s_addr == htonl(INADDR_ANY)) {
      add4(lo4, a->sin_port);
      add6(in6addr_loopback, a->sin_port);
    } else {
      add4(a->sin_addr, a->sin_port);
    }
  } else if (addr->sa_family == AF_INET6) {
    const auto* a = reinterpret_cast<const sockaddr_in6*>(addr);
    if (IN6_IS_ADDR_UNSPECIFIED(&a->sin6_addr)) {
      add6(in6addr_loopback, a->sin6_port);
      add4(lo4, a->sin6_port);
    } else {
      add6(a->sin6_addr, a->sin6_port);
    }
  }
  for (const sockaddr_storage& ss : probes) {
    const int fd = ::socket(ss.ss_family, SOCK_STREAM, 0);
    if (fd < 0) continue;
    ::fcntl(fd, F_SETFL, O_NONBLOCK);
    bool up = ::connect(fd, reinterpret_cast<const sockaddr*>(&ss), ss.ss_len) == 0;
    if (!up && errno == EINPROGRESS) {
      pollfd p{fd, POLLOUT, 0};
      int so = 0;
      socklen_t sl = sizeof so;
      up = ::poll(&p, 1, 500) == 1 && ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so, &sl) == 0 && so == 0;
    }
    ::close(fd);
    if (up) return true;
  }
  return false;
}

namespace {

struct ConnJob {
  ServerState* s;
  int fd;
  std::string remote;
  bool reject;
};

void* conn_main(void* arg) {
  std::unique_ptr<ConnJob> job(static_cast<ConnJob*>(arg));
  ServerState& s = *job->s;
  const bool reject = job->reject;
  ::pthread_setname_np(reject ? "http-reject" : "http-conn");
  try {
    Conn c(s, job->fd, std::move(job->remote));
    if (reject) {
      c.reject();
    } else {
      c.serve();
    }
  } catch (const std::exception& e) {  // e.g. bad_alloc: lose the connection, not the process
    std::fprintf(stderr, "http: connection aborted: %s\n", e.what());
  } catch (...) {
    std::fprintf(stderr, "http: connection aborted by an unknown exception\n");
  }
  job.reset();
  s.thread_done(reject);  // the last access to the server state
  return nullptr;
}

}  // namespace

bool ServerState::spawn(int fd, std::string remote, bool reject) {
  auto job = std::make_unique<ConnJob>(ConnJob{this, fd, std::move(remote), reject});
  pthread_attr_t attr;
  if (::pthread_attr_init(&attr) != 0) return false;
  ::pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  // std::thread cannot set a stack size; handlers (JSON, templates) get a
  // main-thread-sized stack, of which only touched pages cost memory.
  const size_t page = size_t(::getpagesize());
  size_t stack = std::max<size_t>(reject ? size_t(256) << 10 : opt.thread_stack, PTHREAD_STACK_MIN);
  ::pthread_attr_setstacksize(&attr, (stack + page - 1) / page * page);
  pthread_t t;
  const int rc = ::pthread_create(&t, &attr, conn_main, job.get());
  ::pthread_attr_destroy(&attr);
  if (rc != 0) return false;
  job.release();  // owned by the thread now
  return true;
}

void ServerState::thread_done(bool reject) {
  std::lock_guard lk(mu);
  --(reject ? rejecting : active);
  cv.notify_all();  // under the lock: run() may destroy the server right after
}

void ServerState::accept_ready(bool& backoff) {
  for (int i = 0; i < 256; ++i) {  // bounded, so that stop() is seen even under a connection flood
    sockaddr_storage ss{};
    socklen_t sl = sizeof ss;
    const int fd = ::accept(listen_fd, reinterpret_cast<sockaddr*>(&ss), &sl);
    if (fd < 0) {
      if (errno == EINTR || errno == ECONNABORTED) continue;
      if (errno != EAGAIN && errno != EWOULDBLOCK) {
        // EMFILE, ENFILE, ENOBUFS...: pause instead of spinning on a
        // listener that stays readable.
        backoff = true;
        std::fprintf(stderr, "http: accept: %s\n", std::strerror(errno));
      }
      return;
    }
    configure_socket(fd);
    bool reject = false;
    {
      std::lock_guard lk(mu);
      if (opt.max_connections <= 0 || active < opt.max_connections) {
        ++active;
      } else if (rejecting < kMaxRejecting) {
        ++rejecting;
        reject = true;
      } else {
        ::close(fd);  // flooded: drop without an answer
        continue;
      }
    }
    if (!spawn(fd, peer_name(ss), reject)) {
      std::fprintf(stderr, "http: cannot start a connection thread\n");
      ::close(fd);
      thread_done(reject);
    }
  }
}

}  // namespace detail

// ------------------------------------------------------------------ Response

bool Response::has_header(std::string_view lower_name) const {
  for (const auto& h : headers_)
    if (detail::iequals(h.first, lower_name)) return true;
  return false;
}

void Response::set_header(std::string name, std::string value) {
  if (phase_ != Phase::Idle) return;  // the head is already out
  // Handlers may echo request data into headers: never let it split the head.
  std::erase_if(value, [](char c) { return c == '\r' || c == '\n' || c == '\0'; });
  if (!detail::is_token(name)) return;
  if (detail::iequals(name, "content-length") || detail::iequals(name, "transfer-encoding")) return;
  if (detail::iequals(name, "connection")) {
    if (detail::iequals(detail::trim_ows(value), "close")) c_.keep_alive = false;
    return;
  }
  headers_.emplace_back(std::move(name), std::move(value));
}

void Response::send(int status, std::string_view content_type, std::string_view body) {
  if (phase_ != Phase::Idle) throw std::logic_error("http::Response::send: the response has already started");
  if (status < 200 || status > 999) throw std::logic_error("http::Response::send: invalid status");
  status_ = status;
  const bool bodyless = status == 204 || status == 304;
  const std::string head =
      c_.build_head(*this, status, bodyless ? std::string_view() : content_type, bodyless ? -1 : int64_t(body.size()),
                    false, false);
  const size_t n = (bodyless || c_.head_req) ? 0 : body.size();
  iovec iov[2] = {{const_cast<char*>(head.data()), head.size()}, {const_cast<char*>(body.data()), n}};
  phase_ = c_.send_all(iov, 2) == detail::Io::Ok ? Phase::Done : Phase::Failed;
  if (phase_ == Phase::Done) bytes_ = n;
}

void Response::send_json(int status, const json::Value& v) { send(status, "application/json", json::dump(v)); }

void Response::send_error(int status, std::string_view message, std::string_view type, std::string_view code) {
  if (type.empty()) type = status >= 500 ? "server_error" : "invalid_request_error";
  send_json(status, error_body(message, type, code));
}

bool Response::start_stream(int status, std::string_view content_type) {
  if (phase_ != Phase::Idle) throw std::logic_error("http::Response::start_stream: the response has already started");
  if (status < 200 || status > 999 || status == 204 || status == 304)
    throw std::logic_error("http::Response::start_stream: invalid status for a stream");
  if (c_.peer_gone()) {  // left while the request was waiting: nothing to send
    phase_ = Phase::Failed;
    return false;
  }
  status_ = status;
  // HTTP/1.0 has no chunked encoding: the body ends when the connection closes.
  if (c_.http10) c_.keep_alive = false;
  const std::string head = c_.build_head(*this, status, content_type, -1, !c_.http10, true);
  iovec iov{const_cast<char*>(head.data()), head.size()};
  if (c_.send_all(&iov, 1) != detail::Io::Ok) {
    phase_ = Phase::Failed;
    return false;
  }
  if (c_.head_req) {  // HEAD: the head is the whole answer; no point generating
    phase_ = Phase::Done;
    return false;
  }
  phase_ = Phase::Streaming;
  return true;
}

bool Response::write(std::string_view data) {
  if (phase_ == Phase::Idle) throw std::logic_error("http::Response::write before start_stream");
  if (phase_ != Phase::Streaming) return false;
  // Checking before each write notices a closed client at the next event,
  // rather than one write later when the kernel reports the peer's RST.
  if (c_.peer_gone()) {
    phase_ = Phase::Failed;
    return false;
  }
  if (data.empty()) return true;  // an empty chunk would end the stream
  char size_line[24];
  const int len = std::snprintf(size_line, sizeof size_line, "%zx\r\n", data.size());
  iovec iov[3] = {{size_line, size_t(len)},
                  {const_cast<char*>(data.data()), data.size()},
                  {const_cast<char*>("\r\n"), 2}};
  if ((c_.http10 ? c_.send_all(&iov[1], 1) : c_.send_all(iov, 3)) != detail::Io::Ok) {
    phase_ = Phase::Failed;
    return false;
  }
  bytes_ += data.size();
  return true;
}

bool Response::sse(std::string_view data) {
  // Each line becomes its own "data:" field (the client joins them with
  // '\n'), so payloads with line breaks arrive intact.
  scratch_.clear();
  size_t i = 0;
  for (;;) {
    const size_t j = data.find_first_of("\r\n", i);
    scratch_.append("data: ").append(data.substr(i, j == std::string_view::npos ? j : j - i)).push_back('\n');
    if (j == std::string_view::npos) break;
    i = j + ((data[j] == '\r' && j + 1 < data.size() && data[j + 1] == '\n') ? 2 : 1);
  }
  scratch_.push_back('\n');
  return write(scratch_);
}

void Response::end() {
  if (phase_ != Phase::Streaming) return;
  phase_ = Phase::Done;
  if (c_.http10) return;  // closing the connection ends the body
  static constexpr char kLast[] = "0\r\n\r\n";
  iovec iov{const_cast<char*>(kLast), sizeof kLast - 1};
  if (c_.send_all(&iov, 1) != detail::Io::Ok) phase_ = Phase::Failed;
}

bool Response::client_gone() const { return c_.peer_gone(); }

// ------------------------------------------------------------------ Server

Server::Server(Options opts) : s_(std::make_unique<detail::ServerState>()) {
  detail::ServerState& s = *s_;
  s.opt = std::move(opts);
  // Each connection needs a descriptor, and macOS shells default to a soft
  // limit of 256, below max_connections. Raise the soft limit towards the
  // hard one (as the Go runtime does) so that the configured limit is
  // reachable instead of ending in EMFILE.
  if (rlimit rl{}; ::getrlimit(RLIMIT_NOFILE, &rl) == 0) {
    const rlim_t want = std::min<rlim_t>(rlim_t(std::max(0, s.opt.max_connections)) + detail::kMaxRejecting + 256,
                                         std::min<rlim_t>(rl.rlim_max, OPEN_MAX));
    if (rl.rlim_cur < want) {
      rl.rlim_cur = want;
      ::setrlimit(RLIMIT_NOFILE, &rl);
    }
  }
  int p[2];
  if (::pipe(p) != 0) throw std::runtime_error(std::string("http: pipe: ") + std::strerror(errno));
  s.stop_rd = p[0];
  s.stop_wr = p[1];
  for (int fd : p) {
    ::fcntl(fd, F_SETFD, FD_CLOEXEC);
    ::fcntl(fd, F_SETFL, O_NONBLOCK);
  }

  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
  const std::string port = std::to_string(s.opt.port);
  const std::string where = s.opt.host + ":" + port;
  addrinfo* res = nullptr;
  if (const int rc = ::getaddrinfo(s.opt.host.empty() ? nullptr : s.opt.host.c_str(), port.c_str(), &hints, &res))
    throw std::runtime_error("http: cannot resolve " + where + ": " + ::gai_strerror(rc));
  // Names such as "localhost" resolve to both families; clients try
  // 127.0.0.1 more reliably than ::1, so prefer IPv4.
  std::vector<addrinfo*> cands;
  for (addrinfo* ai = res; ai; ai = ai->ai_next) cands.push_back(ai);
  std::stable_sort(cands.begin(), cands.end(),
                   [](addrinfo* a, addrinfo* b) { return a->ai_family == AF_INET && b->ai_family != AF_INET; });
  // A port another program listens on must fail here, at once. With
  // SO_REUSEADDR, macOS lets a socket bind 127.0.0.1:P while another one
  // listens on 0.0.0.0:P (or the reverse), and both then listen: the more
  // specific address silently takes the connections. So bind without it
  // first; a refused bind is retried with it only when nothing answers on
  // the port (the address is held by connections of an earlier run in
  // TIME_WAIT, which SO_REUSEADDR exists for).
  auto try_bind = [&](const addrinfo* ai, bool reuse, int& fd_out) {
    const int fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) return errno;
    const int one = 1, zero = 0;
    if (reuse) ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (ai->ai_family == AF_INET6)  // "::" also accepts IPv4 clients
      ::setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof zero);
    if (::bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && ::listen(fd, std::max(128, s.opt.max_connections)) == 0) {
      fd_out = fd;
      return 0;
    }
    const int e = errno;
    ::close(fd);
    return e;
  };
  int err = EADDRNOTAVAIL;
  for (addrinfo* ai : cands) {
    int fd = -1;
    int e = try_bind(ai, false, fd);
    if (e == EADDRINUSE && !detail::port_answers(ai->ai_addr)) e = try_bind(ai, true, fd);
    if (e == 0) {
      s.listen_fd = fd;
      break;
    }
    err = e;
  }
  ::freeaddrinfo(res);
  if (s.listen_fd < 0)
    throw ListenError("http: cannot listen on " + where + ": " + std::strerror(err) +
                          (err == EADDRINUSE ? " (another program listens on port " + port + ")" : ""),
                      err);
  ::fcntl(s.listen_fd, F_SETFD, FD_CLOEXEC);
  ::fcntl(s.listen_fd, F_SETFL, O_NONBLOCK);

  sockaddr_storage ss{};
  socklen_t sl = sizeof ss;
  ::getsockname(s.listen_fd, reinterpret_cast<sockaddr*>(&ss), &sl);
  s.port = ntohs(ss.ss_family == AF_INET6 ? reinterpret_cast<sockaddr_in6*>(&ss)->sin6_port
                                          : reinterpret_cast<sockaddr_in*>(&ss)->sin_port);
}

Server::~Server() { stop(); }

void Server::route(std::string method, std::string path, Handler h, Body body) {
  (s_->gate_set ? s_->late : s_->routes).push_back({std::move(method), std::move(path), false, std::move(h), body});
}

void Server::route_prefix(std::string method, std::string prefix, Handler h, Body body) {
  (s_->gate_set ? s_->late : s_->routes).push_back({std::move(method), std::move(prefix), true, std::move(h), body});
}

void Server::route_keyless(std::string method, std::string path, Handler h) {
  (s_->gate_set ? s_->late : s_->routes).push_back({std::move(method), std::move(path), false, std::move(h), Body::Any, true});
}

void Server::gate(Handler h) {
  s_->gate.handler = std::move(h);
  s_->gate_set = true;
  s_->gated.store(true, std::memory_order_release);
}

void Server::open() { s_->gated.store(false, std::memory_order_release); }

void Server::run() {
  detail::ServerState& s = *s_;
  bool backoff = false;
  while (!s.stopping.load()) {
    if (backoff) {  // out of descriptors or memory: sleep 100 ms (or until stop)
      backoff = false;
      pollfd p{s.stop_rd, POLLIN, 0};
      ::poll(&p, 1, 100);
      continue;
    }
    pollfd p[2] = {{s.listen_fd, POLLIN, 0}, {s.stop_rd, POLLIN, 0}};
    if (::poll(p, 2, -1) < 0) {
      if (errno != EINTR) backoff = true;
      continue;
    }
    if (p[1].revents) break;
    if (p[0].revents) s.accept_ready(backoff);
  }
  // Refuse new connections, then wait for the served ones: the stop pipe has
  // already woken them (idle ones close, streams fail their next write).
  ::close(s.listen_fd);
  s.listen_fd = -1;
  std::unique_lock lk(s.mu);
  s.cv.wait(lk, [&] { return s.active == 0 && s.rejecting == 0; });
}

void Server::stop() {
  s_->stopping.store(true);
  const char b = 1;
  [[maybe_unused]] const ssize_t n = ::write(s_->stop_wr, &b, 1);  // never drained: stays readable
}

int Server::port() const { return s_->port; }

int Server::connections() const {
  std::lock_guard lk(s_->mu);
  return s_->active;
}

}  // namespace espetero::http
