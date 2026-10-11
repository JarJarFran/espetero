// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// HTTP/1.1 request parsing (RFC 9112) and small shared helpers.
//
// The parser is deliberately strict where leniency enables request smuggling
// or ambiguity (field names with whitespace, conflicting Content-Length,
// Content-Length together with Transfer-Encoding, obsolete line folding) and
// lenient where it costs nothing (bare LF line ends, empty lines before the
// request line, absolute-form targets, UTF-8 in targets).
#include "parse.h"

#include <arpa/inet.h>
#include <netinet/in.h>

#include <cstdio>
#include <ctime>

namespace espetero::http {
namespace detail {

namespace {

constexpr size_t npos = std::string_view::npos;

// RFC 9110 tchar.
constexpr bool is_tchar(unsigned char c) {
  if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
  switch (c) {
    case '!': case '#': case '$': case '%': case '&': case '\'': case '*': case '+':
    case '-': case '.': case '^': case '_': case '`': case '|': case '~':
      return true;
    default:
      return false;
  }
}

constexpr char ascii_lower(char c) { return (c >= 'A' && c <= 'Z') ? char(c + ('a' - 'A')) : c; }
constexpr bool is_digit(char c) { return c >= '0' && c <= '9'; }

int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// 1*DIGIT, saturating at UINT64_MAX (an absurd Content-Length is "too large",
// not "malformed").
bool parse_decimal(std::string_view s, uint64_t& out) {
  if (s.empty()) return false;
  uint64_t v = 0;
  for (char c : s) {
    if (!is_digit(c)) return false;
    const uint64_t d = uint64_t(c - '0');
    v = v > (UINT64_MAX - d) / 10 ? UINT64_MAX : v * 10 + d;
  }
  out = v;
  return true;
}

// Calls f(element) for each non-empty, trimmed element of a "#rule" list.
template <class F>
void for_each_element(std::string_view list, F&& f) {
  for (;;) {
    const size_t comma = list.find(',');
    const std::string_view e = trim_ows(list.substr(0, comma));
    if (!e.empty()) f(e);
    if (comma == npos) return;
    list.remove_prefix(comma + 1);
  }
}

bool istarts_with(std::string_view s, std::string_view prefix) {
  return s.size() >= prefix.size() && iequals(s.substr(0, prefix.size()), prefix);
}

}  // namespace

bool is_token(std::string_view s) {
  if (s.empty()) return false;
  for (char c : s)
    if (!is_tchar(static_cast<unsigned char>(c))) return false;
  return true;
}

bool iequals(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (ascii_lower(a[i]) != ascii_lower(b[i])) return false;
  return true;
}

std::string_view trim_ows(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
  return s;
}

size_t skip_empty_lines(std::string_view buf) {
  size_t i = 0;
  while (i < buf.size() && (buf[i] == '\r' || buf[i] == '\n')) ++i;
  return i;
}

size_t find_head_end(std::string_view b, size_t& scan) {
  // The head ends with an empty line: "\n\n" or "\n\r\n" (the first '\n'
  // ends the last field line, or the request line).
  for (size_t i = b.find('\n', scan); i != npos; i = b.find('\n', i + 1)) {
    if (i + 1 < b.size() && b[i + 1] == '\n') return i + 2;
    if (i + 2 < b.size() && b[i + 1] == '\r' && b[i + 2] == '\n') return i + 3;
  }
  // The terminator may straddle the end: resume two bytes back.
  scan = b.size() >= 2 ? b.size() - 2 : 0;
  return 0;
}

Head parse_head(std::string_view head, Request& req, size_t max_fields) {
  Head h;
  auto fail = [&h](int status, const char* message) {
    h.error = status;
    h.message = message;
    return h;
  };
  size_t pos = 0;
  auto next_line = [&](std::string_view& line) {
    if (pos >= head.size()) return false;
    const size_t nl = head.find('\n', pos);
    const size_t end = nl == npos ? head.size() : nl;
    line = head.substr(pos, end - pos);
    pos = nl == npos ? head.size() : nl + 1;
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    return true;
  };

  // request-line = method SP request-target SP HTTP-version
  std::string_view line;
  if (!next_line(line) || line.empty()) return fail(400, "empty request line");
  if (line.find('\r') != npos) return fail(400, "stray CR in the request line");
  const size_t sp1 = line.find(' '), sp2 = line.rfind(' ');
  if (sp1 == npos || sp1 == sp2) return fail(400, "malformed request line");
  const std::string_view method = line.substr(0, sp1);
  const std::string_view target = line.substr(sp1 + 1, sp2 - sp1 - 1);
  const std::string_view version = line.substr(sp2 + 1);
  if (!is_token(method)) return fail(400, "invalid method");
  if (target.empty()) return fail(400, "empty request target");
  for (char ch : target) {  // bytes >= 0x80 are tolerated (raw UTF-8 paths)
    const auto c = static_cast<unsigned char>(ch);
    if (c <= 0x20 || c == 0x7F) return fail(400, "invalid character in the request target");
  }
  if (version.size() != 8 || version.substr(0, 5) != "HTTP/" || !is_digit(version[5]) || version[6] != '.' ||
      !is_digit(version[7]))
    return fail(400, "malformed HTTP version");
  if (version[5] != '1') return fail(505, "only HTTP/1.x is supported");
  req.minor_version = version[7] - '0';  // 1.2+ is treated as 1.1
  req.method.assign(method);
  req.target.assign(target);

  // origin-form "/path?query", asterisk-form "*" (OPTIONS), absolute-form
  // "http://host/path?query" (servers must accept it; the authority is ignored).
  std::string_view pq = target;
  std::string abs_form;
  if (target[0] == '/' || target == "*") {
  } else if (istarts_with(target, "http://") || istarts_with(target, "https://")) {
    const size_t p = target.find_first_of("/?#", target.find("//") + 2);
    if (p == npos) {
      pq = "/";
    } else if (target[p] == '/') {
      pq = target.substr(p);
    } else {
      abs_form = "/";
      abs_form += target.substr(p);
      pq = abs_form;
    }
  } else {
    return fail(400, "invalid request target");
  }
  pq = pq.substr(0, pq.find('#'));  // clients must not send fragments; tolerate them
  const size_t q = pq.find('?');
  req.path.assign(pq.substr(0, q));
  req.query.assign(q == npos ? std::string_view() : pq.substr(q + 1));

  // field-line = field-name ":" OWS field-value OWS
  bool have_length = false, have_te = false, conn_close = false, conn_keep_alive = false;
  std::string te;  // Transfer-Encoding codings of all fields, comma-joined
  size_t fields = 0;
  while (next_line(line) && !line.empty()) {
    if (++fields > max_fields) return fail(431, "too many header fields");
    if (line[0] == ' ' || line[0] == '\t') return fail(400, "obsolete line folding is not supported");
    const size_t colon = line.find(':');
    if (colon == npos) return fail(400, "malformed header field");
    const std::string_view name = line.substr(0, colon);
    if (!is_token(name)) return fail(400, "invalid header field name");  // also "Name : value"
    const std::string_view value = trim_ows(line.substr(colon + 1));
    if (value.find_first_of(std::string_view("\0\r", 2)) != npos)
      return fail(400, "invalid character in a header field value");
    std::string lname(name);
    for (char& c : lname) c = ascii_lower(c);

    if (lname == "content-length") {
      // A list of identical values ("5, 5") is allowed; anything else that
      // disagrees is a smuggling attempt or a broken client.
      bool bad = false;
      int n = 0;
      for_each_element(value, [&](std::string_view e) {
        uint64_t v = 0;
        if (!parse_decimal(e, v) || (have_length && v != h.content_length)) {
          bad = true;
          return;
        }
        have_length = true;
        h.content_length = v;
        ++n;
      });
      if (bad || n == 0) return fail(400, "invalid Content-Length");
    } else if (lname == "transfer-encoding") {
      have_te = true;
      if (!te.empty()) te += ',';
      te += value;
    } else if (lname == "connection") {
      for_each_element(value, [&](std::string_view e) {
        if (iequals(e, "close")) conn_close = true;
        if (iequals(e, "keep-alive")) conn_keep_alive = true;
      });
    } else if (lname == "expect") {
      if (iequals(value, "100-continue")) {
        h.expect_continue = true;
      } else if (!value.empty()) {
        return fail(417, "only 'Expect: 100-continue' is supported");
      }
    }
    req.headers.emplace_back(std::move(lname), std::string(value));
  }

  if (have_te) {
    if (req.minor_version == 0) return fail(400, "Transfer-Encoding is not allowed in HTTP/1.0 requests");
    if (have_length) return fail(400, "both Transfer-Encoding and Content-Length");
    int n = 0;
    bool last_chunked = false, chunked_twice_or_early = false;
    for_each_element(te, [&](std::string_view e) {
      if (last_chunked) chunked_twice_or_early = true;
      last_chunked = iequals(e, "chunked");
      ++n;
    });
    if (!last_chunked || chunked_twice_or_early)
      return fail(400, "chunked must be the final transfer coding, applied once");
    if (n > 1) return fail(501, "only the chunked transfer coding is supported");
    h.chunked = true;
  }
  if (req.minor_version == 0) {
    h.keep_alive = conn_keep_alive && !conn_close;
    h.expect_continue = false;  // RFC 9110 10.1.1: ignore it from HTTP/1.0 clients
  } else {
    h.keep_alive = !conn_close;
  }
  return h;
}

bool parse_chunk_size(std::string_view line, uint64_t& size) {
  uint64_t v = 0;
  size_t i = 0;
  for (; i < line.size(); ++i) {
    const int d = hex_value(line[i]);
    if (d < 0) break;
    if (v >> 56) return false;  // stay below 2^60, far from overflow
    v = (v << 4) | uint64_t(d);
  }
  if (i == 0) return false;
  while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
  if (i < line.size() && line[i] != ';') return false;  // extensions are ignored
  size = v;
  return true;
}

std::string_view http_date() {
  static constexpr const char* kDays[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
  static constexpr const char* kMonths[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
  thread_local time_t cached = -1;
  thread_local char buf[40];
  thread_local int len = 0;
  const time_t now = ::time(nullptr);
  if (now != cached) {
    struct tm tm {};
    ::gmtime_r(&now, &tm);
    len = std::snprintf(buf, sizeof buf, "%s, %02d %s %04d %02d:%02d:%02d GMT", kDays[tm.tm_wday % 7], tm.tm_mday,
                        kMonths[tm.tm_mon % 12], tm.tm_year + 1900, tm.tm_hour, tm.tm_min, tm.tm_sec);
    cached = now;
  }
  return {buf, size_t(len)};
}

bool split_authority(std::string_view authority, std::string& host, std::string& port) {
  std::string_view a = trim_ows(authority), h, p;
  if (!a.empty() && a.front() == '[') {  // IPv6 literal
    const size_t close = a.find(']');
    if (close == npos) return false;
    h = a.substr(0, close + 1);
    const std::string_view rest = a.substr(close + 1);
    if (!rest.empty() && rest.front() != ':') return false;
    if (!rest.empty()) p = rest.substr(1);
  } else {
    const size_t colon = a.find(':');
    if (colon != npos && a.find(':', colon + 1) != npos) return false;  // IPv6 without brackets
    h = a.substr(0, colon);
    if (colon != npos) p = a.substr(colon + 1);
  }
  for (char c : p)
    if (!is_digit(c)) return false;
  host.clear();
  for (char c : h) host.push_back(ascii_lower(c));
  if (host.size() > 1 && host.back() == '.') host.pop_back();
  port = std::string(p);
  return true;
}

bool same_origin(std::string_view origin, std::string_view host_header) {
  origin = trim_ows(origin);
  const size_t sep = origin.find("://");
  if (sep == npos || !iequals(origin.substr(0, sep), "http")) return false;  // also "null"
  std::string_view auth = origin.substr(sep + 3);
  auth = auth.substr(0, auth.find('/'));
  std::string oh, op, hh, hp;
  if (!split_authority(auth, oh, op) || !split_authority(host_header, hh, hp) || oh.empty() || oh != hh) return false;
  auto port = [](const std::string& s) {
    uint64_t v = 80;
    return s.empty() || parse_decimal(s, v) ? v : UINT64_MAX;
  };
  return port(op) == port(hp) && port(op) != UINT64_MAX;
}

bool host_allowed(std::string_view host_header, std::string_view bind_host) {
  if (trim_ows(host_header).empty() || is_loopback_host(host_header)) return true;
  std::string h, port, bind, bport;
  const bool v6 = bind_host.find(':') != npos && bind_host.find('[') == npos;  // a bare IPv6 bind address
  return split_authority(host_header, h, port) &&
         split_authority(v6 ? "[" + std::string(bind_host) + "]" : bind_host, bind, bport) && !bind.empty() &&
         h == bind;
}

std::string media_type(std::string_view content_type) {
  const std::string_view t = trim_ows(content_type.substr(0, content_type.find(';')));
  std::string out;
  for (char c : t) out.push_back(ascii_lower(c));
  return out;
}

}  // namespace detail

// ------------------------------------------------------------------ Request

const std::string* Request::header(std::string_view name) const {
  for (const auto& [k, v] : headers)
    if (detail::iequals(k, name)) return &v;
  return nullptr;
}

std::string Request::query_param(std::string_view name) const {
  std::string_view q = query;
  while (!q.empty()) {
    const size_t amp = q.find('&');
    const std::string_view kv = q.substr(0, amp);
    const size_t eq = kv.find('=');
    if (url_decode(kv.substr(0, eq), true) == name)
      return eq == std::string_view::npos ? std::string() : url_decode(kv.substr(eq + 1), true);
    if (amp == std::string_view::npos) break;
    q.remove_prefix(amp + 1);
  }
  return {};
}

// ------------------------------------------------------------------ helpers

std::string url_decode(std::string_view s, bool plus_as_space) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (c == '%' && i + 2 < s.size() && detail::hex_value(s[i + 1]) >= 0 && detail::hex_value(s[i + 2]) >= 0) {
      out.push_back(char(detail::hex_value(s[i + 1]) * 16 + detail::hex_value(s[i + 2])));
      i += 2;
    } else if (c == '+' && plus_as_space) {
      out.push_back(' ');
    } else {
      out.push_back(c);
    }
  }
  return out;
}

std::string_view status_reason(int status) {
  switch (status) {
    case 100: return "Continue";
    case 101: return "Switching Protocols";
    case 200: return "OK";
    case 201: return "Created";
    case 202: return "Accepted";
    case 204: return "No Content";
    case 206: return "Partial Content";
    case 301: return "Moved Permanently";
    case 302: return "Found";
    case 303: return "See Other";
    case 304: return "Not Modified";
    case 307: return "Temporary Redirect";
    case 308: return "Permanent Redirect";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 406: return "Not Acceptable";
    case 408: return "Request Timeout";
    case 409: return "Conflict";
    case 410: return "Gone";
    case 411: return "Length Required";
    case 413: return "Content Too Large";
    case 414: return "URI Too Long";
    case 415: return "Unsupported Media Type";
    case 416: return "Range Not Satisfiable";
    case 417: return "Expectation Failed";
    case 422: return "Unprocessable Content";
    case 429: return "Too Many Requests";
    case 431: return "Request Header Fields Too Large";
    case 499: return "Client Closed Request";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    case 504: return "Gateway Timeout";
    case 505: return "HTTP Version Not Supported";
    default: return "";
  }
}

bool is_loopback_host(std::string_view host) {
  std::string h, port;
  host = detail::trim_ows(host);
  if (host.find(':') != host.rfind(':') && host.find('[') == std::string_view::npos) {
    h = std::string(host);  // a bare IPv6 address (a bind address, not a Host header)
  } else {
    if (!detail::split_authority(host, h, port) || h.empty()) return false;
    if (h == "localhost" || (h.size() > 10 && h.compare(h.size() - 10, 10, ".localhost") == 0)) return true;
    if (h.front() == '[') h = h.substr(1, h.size() - 2);
  }
  in_addr a4{};
  if (::inet_pton(AF_INET, h.c_str(), &a4) == 1) return (ntohl(a4.s_addr) >> 24) == 127;
  in6_addr a6{};
  if (::inet_pton(AF_INET6, h.c_str(), &a6) == 1)
    return IN6_IS_ADDR_LOOPBACK(&a6) || (IN6_IS_ADDR_V4MAPPED(&a6) && a6.s6_addr[12] == 127);
  return false;
}

json::Value error_body(std::string_view message, std::string_view type, std::string_view code) {
  // Messages may quote request bytes (paths, exception texts); keep the JSON
  // valid UTF-8 whatever they contain.
  auto utf8 = [](std::string_view s) {
    if (json::utf8_valid(s)) return std::string(s);
    std::string out;
    for (size_t i = 0; i < s.size();) json::utf8_append(out, json::utf8_next(s, i));
    return out;
  };
  json::Value err = json::Value::object();
  err.set("message", utf8(message));
  err.set("type", std::string(type));
  err.set("param", json::Value());
  err.set("code", code.empty() ? json::Value() : json::Value(code));
  json::Value v = json::Value::object();
  v.set("error", std::move(err));
  return v;
}

}  // namespace espetero::http
