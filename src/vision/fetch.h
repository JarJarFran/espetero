// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Images by URL: http(s) image URLs fetched by the server, only when
// the operator allows it (--image-urls), with limits for untrusted clients:
// a size cap, a total timeout, few redirects, an image/* content type, and no
// connection to this machine or its networks (SSRF) unless the operator also
// allows that (--image-urls-private).
//
// The host is resolved and every address checked before the request, again
// at each redirect, and the address each connection really used (the task's
// metrics) after it, so a name that resolves differently the second time (DNS
// rebinding) gets no image back. Proxies are not used (the address checked is
// the one connected to); cookies, caches and credentials neither.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

struct sockaddr;

namespace espetero::vision {

struct FetchOptions {
  size_t max_bytes = size_t(32) << 20;  // the image's bytes at most (after content decoding)
  double timeout_s = 20;                // the whole fetch: name resolution, redirects, transfer
  int max_redirects = 3;
  bool allow_private = false;           // also addresses blocked_address refuses
  // The address check (null: blocked_address); tests substitute their own.
  bool (*blocked)(const struct sockaddr*) = nullptr;
};

// Addresses refused without allow_private. IPv4: 0/8 (this network,
// unspecified), 10/8, 172.16/12, 192.168/16 (private), 100.64/10 (carrier
// NAT), 127/8 (loopback), 169.254/16 (link-local), 192.0.0/24 (IETF
// protocol assignments), 198.18/15 (benchmarking), 224/4 (multicast), 240/4
// (reserved, broadcast). IPv6: :: and ::1, fe80::/10 (link-local), fec0::/10
// (site-local), fc00::/7 (unique local), ff00::/8 (multicast), 100::/64
// (discard), and an IPv4 address embedded as IPv4-mapped (::ffff:0:0/96),
// IPv4-compatible (::/96), NAT64 (64:ff9b::/96) or 6to4 (2002::/16) is
// checked as IPv4.
bool blocked_ipv4(uint32_t host_order);
bool blocked_ipv6(const uint8_t address[16]);
bool blocked_address(const struct sockaddr* sa);
// An address in text form (as connection metrics report it); false for text
// that is not an address.
bool blocked_address_text(std::string_view text, bool (*blocked)(const struct sockaddr*) = nullptr);

// Every address `host` resolves to (a name or a literal), within timeout_s.
// False with `err` when it does not resolve in time.
bool resolve_host(const std::string& host, double timeout_s, std::vector<std::string>& addresses, std::string& err,
                  bool (*blocked)(const struct sockaddr*) = nullptr, bool* any_blocked = nullptr);
// The host may be fetched from: it resolves, and (without allow_private) to
// no blocked address. `err` says why not.
bool host_allowed(const std::string& host, const FetchOptions& opt, double timeout_s, std::string& err);

// The image at an http(s) URL: its bytes and the response's MIME type, or
// false with `err` (the reason, for the client).
bool fetch_image(const std::string& url, const FetchOptions& opt, std::string& bytes, std::string& mime, std::string& err);
// The same as a base64 data URL ("data:<mime>;base64,..."), which the image
// path takes as it takes a client's.
bool fetch_image_data_url(const std::string& url, const FetchOptions& opt, std::string& data_url, std::string& err);

}  // namespace espetero::vision
