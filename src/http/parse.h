// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Internal HTTP/1.1 message parsing (RFC 9112): pure functions over byte
// buffers, no I/O, so they can be unit tested on their own.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "http.h"

namespace espetero::http::detail {

// What the request head says about the rest of the message.
struct Head {
  int error = 0;              // status to answer with (400, 417, 431, 501, 505), 0 = ok
  std::string message;        // why, for the error body
  bool chunked = false;       // Transfer-Encoding: chunked
  uint64_t content_length = 0;
  bool expect_continue = false;  // Expect: 100-continue (HTTP/1.1 only)
  bool keep_alive = true;        // after the version default and Connection
};

// Number of leading CR/LF bytes: empty lines a server must ignore before a
// request line (clients may send an extra CRLF after a POST body).
size_t skip_empty_lines(std::string_view buf);

// Length of the head (request line + fields + blank line) at the start of
// buf, or 0 while it is incomplete. Accepts CRLF and bare LF line ends.
// `scan` carries the search position across calls so that a head arriving in
// many pieces is scanned once overall; start it at 0.
size_t find_head_end(std::string_view buf, size_t& scan);

// Parses a complete head into req (method, target, path, query, headers,
// minor_version). On error, Head::error is set and req is partially filled.
Head parse_head(std::string_view head, Request& req, size_t max_fields);

// chunk-size [ BWS ";" chunk-ext ] (the line without its ending). Rejects
// empty sizes, non-hex digits and sizes of 2^60 or more.
bool parse_chunk_size(std::string_view line, uint64_t& size);

bool is_token(std::string_view s);  // RFC 9110 token (methods, field names)
bool iequals(std::string_view a, std::string_view b);  // ASCII case-insensitive
std::string_view trim_ows(std::string_view s);         // strip SP / HTAB
// IMF-fixdate for the current second ("Sun, 06 Nov 1994 08:49:37 GMT"),
// formatted without locale and cached per thread.
std::string_view http_date();

// Host and port of a Host header or URL authority: the host lowercased, IPv6
// literals with their brackets, one trailing dot dropped ("LocalHost.:80" ->
// "localhost", "80"); port "" when absent. False when malformed.
bool split_authority(std::string_view authority, std::string& host, std::string& port);
// Whether an Origin header ("http://host[:port]") is the origin of a request
// sent with this Host header (scheme http, port 80 when absent).
bool same_origin(std::string_view origin, std::string_view host_header);
// Whether a server bound to `bind_host` without an API key answers a request
// with this Host header: an empty one, a loopback name or address
// (is_loopback_host), or the bind address itself; any port.
bool host_allowed(std::string_view host_header, std::string_view bind_host);
// The media type of a Content-Type value, without parameters, lowercased.
std::string media_type(std::string_view content_type);

}  // namespace espetero::http::detail
