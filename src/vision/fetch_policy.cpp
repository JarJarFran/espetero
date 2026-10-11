// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Images by URL: which addresses the server may fetch from, and name
// resolution with a deadline. Plain C++ (tests check it without a network).
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

#include "fetch.h"

namespace espetero::vision {

bool blocked_ipv4(uint32_t a) {
  auto in = [&](uint32_t net, int bits) { return bits == 0 || (a >> (32 - bits)) == (net >> (32 - bits)); };
  return in(0x00000000u, 8)       // 0.0.0.0/8: this network, unspecified
         || in(0x0A000000u, 8)    // 10/8
         || in(0x64400000u, 10)   // 100.64/10: carrier-grade NAT
         || in(0x7F000000u, 8)    // 127/8: loopback
         || in(0xA9FE0000u, 16)   // 169.254/16: link-local
         || in(0xAC100000u, 12)   // 172.16/12
         || in(0xC0000000u, 24)   // 192.0.0/24: IETF protocol assignments
         || in(0xC0A80000u, 16)   // 192.168/16
         || in(0xC6120000u, 15)   // 198.18/15: benchmarking
         || in(0xE0000000u, 4)    // 224/4: multicast
         || in(0xF0000000u, 4);   // 240/4: reserved, 255.255.255.255
}

bool blocked_ipv6(const uint8_t b[16]) {
  auto zero = [&](int from, int to) {
    for (int i = from; i < to; ++i)
      if (b[i]) return false;
    return true;
  };
  auto v4 = [&](int at) { return uint32_t(b[at]) << 24 | uint32_t(b[at + 1]) << 16 | uint32_t(b[at + 2]) << 8 | b[at + 3]; };
  if (zero(0, 16)) return true;                                    // ::
  if (zero(0, 15) && b[15] == 1) return true;                      // ::1
  if (zero(0, 10) && b[10] == 0xFF && b[11] == 0xFF) return blocked_ipv4(v4(12));  // ::ffff:a.b.c.d
  if (zero(0, 12)) return blocked_ipv4(v4(12));                    // ::a.b.c.d (IPv4-compatible)
  if (b[0] == 0x00 && b[1] == 0x64 && b[2] == 0xFF && b[3] == 0x9B && zero(4, 12)) return blocked_ipv4(v4(12));  // 64:ff9b::/96
  if (b[0] == 0x20 && b[1] == 0x02) return blocked_ipv4(v4(2));    // 2002::/16: 6to4
  if (b[0] == 0x01 && b[1] == 0x00 && zero(2, 8)) return true;     // 100::/64: discard
  if (b[0] == 0xFE && (b[1] & 0xC0) == 0x80) return true;          // fe80::/10: link-local
  if (b[0] == 0xFE && (b[1] & 0xC0) == 0xC0) return true;          // fec0::/10: site-local
  if ((b[0] & 0xFE) == 0xFC) return true;                          // fc00::/7: unique local
  if (b[0] == 0xFF) return true;                                   // ff00::/8: multicast
  return false;
}

bool blocked_address(const struct sockaddr* sa) {
  if (!sa) return true;
  if (sa->sa_family == AF_INET) return blocked_ipv4(ntohl(reinterpret_cast<const sockaddr_in*>(sa)->sin_addr.s_addr));
  if (sa->sa_family == AF_INET6)
    return blocked_ipv6(reinterpret_cast<const sockaddr_in6*>(sa)->sin6_addr.s6_addr);
  return true;  // other families are not fetched from
}

bool blocked_address_text(std::string_view text, bool (*blocked)(const struct sockaddr*)) {
  if (!blocked) blocked = blocked_address;
  std::string t(text);
  if (const size_t pct = t.find('%'); pct != std::string::npos) t.resize(pct);  // fe80::1%en0
  if (t.size() > 2 && t.front() == '[' && t.back() == ']') t = t.substr(1, t.size() - 2);
  if (t.size() > 1 && t.back() == '.') t.pop_back();  // "127.0.0.1.": the same host (the DNS root)
  sockaddr_in a4{};
  sockaddr_in6 a6{};
  if (::inet_pton(AF_INET, t.c_str(), &a4.sin_addr) == 1) {
    a4.sin_family = AF_INET;
    return blocked(reinterpret_cast<const sockaddr*>(&a4));
  }
  if (::inet_pton(AF_INET6, t.c_str(), &a6.sin6_addr) == 1) {
    a6.sin6_family = AF_INET6;
    return blocked(reinterpret_cast<const sockaddr*>(&a6));
  }
  return false;
}

namespace {

struct Lookup {
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  int rc = 0;
  std::vector<sockaddr_storage> addrs;
};

std::string address_text(const sockaddr_storage& ss) {
  char buf[INET6_ADDRSTRLEN] = {0};
  if (ss.ss_family == AF_INET) ::inet_ntop(AF_INET, &reinterpret_cast<const sockaddr_in*>(&ss)->sin_addr, buf, sizeof buf);
  else if (ss.ss_family == AF_INET6) ::inet_ntop(AF_INET6, &reinterpret_cast<const sockaddr_in6*>(&ss)->sin6_addr, buf, sizeof buf);
  return buf;
}

}  // namespace

bool resolve_host(const std::string& host, double timeout_s, std::vector<std::string>& addresses, std::string& err,
                  bool (*blocked)(const struct sockaddr*), bool* any_blocked) {
  if (!blocked) blocked = blocked_address;
  addresses.clear();
  if (any_blocked) *any_blocked = false;
  std::string h = host;
  if (h.size() > 2 && h.front() == '[' && h.back() == ']') h = h.substr(1, h.size() - 2);
  // A trailing dot names the same host (the DNS root): "127.0.0.1." is
  // 127.0.0.1 whether or not this machine's resolver accepts the dot.
  if (h.size() > 1 && h.back() == '.') h.pop_back();
  if (h.empty()) {
    err = "no host";
    return false;
  }
  // getaddrinfo has no deadline of its own: it runs on a thread of its own
  // and the caller waits at most timeout_s (a late answer is dropped).
  auto st = std::make_shared<Lookup>();
  try {
    std::thread([st, h] {
      addrinfo hints{};
      hints.ai_family = AF_UNSPEC;
      hints.ai_socktype = SOCK_STREAM;
      addrinfo* res = nullptr;
      const int rc = ::getaddrinfo(h.c_str(), nullptr, &hints, &res);
      std::vector<sockaddr_storage> a;
      for (addrinfo* p = res; p; p = p->ai_next) {
        sockaddr_storage ss{};
        std::memcpy(&ss, p->ai_addr, std::min<size_t>(p->ai_addrlen, sizeof ss));
        a.push_back(ss);
      }
      if (res) ::freeaddrinfo(res);
      std::lock_guard<std::mutex> lk(st->mu);
      st->rc = rc;
      st->addrs = std::move(a);
      st->done = true;
      st->cv.notify_all();
    }).detach();
  } catch (const std::exception& e) {
    err = std::string("cannot resolve ") + host + ": " + e.what();
    return false;
  }
  std::unique_lock<std::mutex> lk(st->mu);
  const auto wait = std::chrono::duration<double>(std::max(0.0, timeout_s));
  if (!st->cv.wait_for(lk, std::chrono::duration_cast<std::chrono::milliseconds>(wait), [&] { return st->done; })) {
    err = "resolving " + host + " timed out";
    return false;
  }
  if (st->rc != 0 || st->addrs.empty()) {
    err = "cannot resolve " + host + (st->rc ? std::string(": ") + ::gai_strerror(st->rc) : std::string());
    return false;
  }
  for (const sockaddr_storage& ss : st->addrs) {
    addresses.push_back(address_text(ss));
    if (any_blocked && blocked(reinterpret_cast<const sockaddr*>(&ss))) *any_blocked = true;
  }
  return true;
}

bool host_allowed(const std::string& host, const FetchOptions& opt, double timeout_s, std::string& err) {
  if (opt.allow_private) return true;  // the operator allows every address: the fetch resolves it
  std::vector<std::string> addrs;
  bool blocked = false;
  if (!resolve_host(host, timeout_s, addrs, err, opt.blocked, &blocked)) return false;
  if (blocked) {
    bool (*is_blocked)(const struct sockaddr*) = opt.blocked ? opt.blocked : blocked_address;
    std::string which;
    for (const std::string& a : addrs)
      if (blocked_address_text(a, is_blocked)) {
        which = a;
        break;
      }
    err = host + " resolves to " + which +
          ", a loopback, private, link-local or otherwise non-public address (the operator can allow those with "
          "--image-urls-private)";
    return false;
  }
  return true;
}

}  // namespace espetero::vision
