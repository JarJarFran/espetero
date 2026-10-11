// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The built-in web chat page; see webui.h.
#include "webui.h"

#include <CommonCrypto/CommonDigest.h>

#include "webui_html.h"  // generated from src/server/webui.html (Makefile)

namespace espetero {

namespace {

std::string base64(const unsigned char* p, size_t n) {
  static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (size_t i = 0; i < n; i += 3) {
    const uint32_t v = uint32_t(p[i]) << 16 | (i + 1 < n ? uint32_t(p[i + 1]) << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
    out += t[v >> 18 & 63];
    out += t[v >> 12 & 63];
    out += i + 1 < n ? t[v >> 6 & 63] : '=';
    out += i + 2 < n ? t[v & 63] : '=';
  }
  return out;
}

// CSP source expressions for every <tag>...</tag> block of the page:
// 'sha256-<base64 of the digest of its text>'.
std::string hashes(const std::string& page, const std::string& tag) {
  std::string out;
  const std::string open = "<" + tag + ">", close = "</" + tag + ">";
  for (size_t at = page.find(open); at != std::string::npos; at = page.find(open, at)) {
    const size_t start = at + open.size(), end = page.find(close, start);
    if (end == std::string::npos) break;
    unsigned char d[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(page.data() + start, CC_LONG(end - start), d);
    out += " 'sha256-" + base64(d, sizeof d) + "'";
    at = end + close.size();
  }
  return out;
}

}  // namespace

const std::string& webui_page() {
  static const std::string page = [] {
    std::string p(webui_html, sizeof webui_html - 1);
    if (!p.empty() && p[0] == '\n') p.erase(0, 1);  // the newline after the raw string's opening
    return p;
  }();
  return page;
}

std::string webui_csp(const std::string& page) {
  return "default-src 'none'; script-src" + hashes(page, "script") + "; style-src" + hashes(page, "style") +
         "; img-src data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";
}

void install_webui(http::Server& server, bool enabled) {
  if (!enabled) {
    server.route_keyless("GET", "/", [](const http::Request&, http::Response& res) {
      res.send(200, "text/plain; charset=utf-8", "Espetero is running (the web chat is off: --no-webui)\n");
    });
    return;
  }
  const std::string csp = webui_csp(webui_page());
  auto page = [csp](const http::Request&, http::Response& res) {
    res.set_header("Content-Security-Policy", csp);
    res.set_header("X-Content-Type-Options", "nosniff");
    res.set_header("X-Frame-Options", "DENY");
    res.set_header("Referrer-Policy", "no-referrer");
    res.set_header("Cache-Control", "no-cache");
    res.send(200, "text/html; charset=utf-8", webui_page());
  };
  server.route_keyless("GET", "/", page);
  server.route_keyless("GET", "/chat", page);
}

}  // namespace espetero
