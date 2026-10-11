// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The built-in web chat page (make test): served at / and /chat with a
// Content-Security-Policy whose hashes match the page's script and style
// (checked against openssl), self-contained (no other origin), never parsing
// text as HTML; without a key the page's own requests pass the browser guard
// and other origins' do not; with a key the page is served without it.
//
//   build/webui_test
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include "../src/http/http.h"
#include "../src/server/webui.h"

using namespace espetero;

namespace {

int checks = 0, failures = 0;

void check(bool ok, const std::string& what) {
  ++checks;
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
  }
}

struct Reply {
  int status = 0;
  std::string head, body;
  std::string header(const std::string& name) const {  // lower-case name
    std::string h = head;
    for (char& c : h) c = char(std::tolower(static_cast<unsigned char>(c)));
    const size_t at = h.find("\r\n" + name + ": ");
    if (at == std::string::npos) return "";
    const size_t start = at + name.size() + 4;
    return head.substr(start, head.find("\r\n", start) - start);
  }
};

// One request on a fresh connection (Connection: close), read to the end.
Reply ask(int port, const std::string& request) {
  Reply r;
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(uint16_t(port));
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) {
    ::close(fd);
    return r;
  }
  (void)!::write(fd, request.data(), request.size());
  std::string got;
  char buf[65536];
  for (ssize_t n; (n = ::read(fd, buf, sizeof buf)) > 0;) got.append(buf, size_t(n));
  ::close(fd);
  const size_t end = got.find("\r\n\r\n");
  if (end == std::string::npos) return r;
  r.head = got.substr(0, end + 2);
  r.body = got.substr(end + 4);
  r.status = std::atoi(got.c_str() + 9);
  return r;
}

std::string get(const std::string& path, const std::string& host, const std::string& extra = "") {
  return "GET " + path + " HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n" + extra + "\r\n";
}

// sha256 base64 of a file, by openssl (an implementation independent of the server's).
std::string openssl_hash(const std::string& text) {
  char tmpl[] = "/tmp/espetero-webui-XXXXXX";
  const int fd = ::mkstemp(tmpl);
  (void)!::write(fd, text.data(), text.size());
  ::close(fd);
  std::string out;
  if (FILE* p = ::popen(("/usr/bin/openssl dgst -sha256 -binary '" + std::string(tmpl) + "' | /usr/bin/base64").c_str(), "r")) {
    char buf[256];
    while (std::fgets(buf, sizeof buf, p)) out += buf;
    ::pclose(p);
  }
  ::unlink(tmpl);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  return out;
}

std::string between(const std::string& s, const std::string& open, const std::string& close) {
  const size_t a = s.find(open);
  if (a == std::string::npos) return "";
  const size_t b = s.find(close, a + open.size());
  return b == std::string::npos ? "" : s.substr(a + open.size(), b - a - open.size());
}

struct Running {
  http::Server srv;
  std::thread th;
  explicit Running(http::Options o, bool webui) : srv(std::move(o)) {
    install_webui(srv, webui);
    srv.route("POST", "/v1/chat/completions", [](const http::Request&, http::Response& res) {
      res.send(200, "application/json", "{\"ok\":true}");
    }, http::Body::Json);
    th = std::thread([this] { srv.run(); });
  }
  ~Running() {
    srv.stop();
    th.join();
  }
};

}  // namespace

int main() {
  const std::string& page = webui_page();
  check(page.size() < 60 * 1024, "page under 60 KB: " + std::to_string(page.size()));
  check(page.rfind("<!doctype html>", 0) == 0, "page starts with its doctype");
  // Self-contained: nothing from another origin.
  for (const char* bad : {"http://", "https://", "<link", "@import", "url(", "<iframe", "src=\""})
    check(page.find(bad) == std::string::npos, std::string("no ") + bad + " in the page");
  // Text is never parsed as HTML.
  for (const char* bad : {"innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"})
    check(page.find(bad) == std::string::npos, std::string("no ") + bad + " in the page");
  const std::string script = between(page, "<script>", "</script>"), style = between(page, "<style>", "</style>");
  check(!script.empty() && !style.empty(), "one script and one style block");
  const std::string csp = webui_csp(page);
  check(csp.find("script-src 'sha256-" + openssl_hash(script) + "'") != std::string::npos, "script hash: " + csp);
  check(csp.find("style-src 'sha256-" + openssl_hash(style) + "'") != std::string::npos, "style hash");
  check(csp.find("default-src 'none'") != std::string::npos && csp.find("connect-src 'self'") != std::string::npos &&
            csp.find("frame-ancestors 'none'") != std::string::npos && csp.find("unsafe") == std::string::npos,
        "policy");

  {  // no key: the page, its own requests in, other origins out
    http::Options o;
    o.port = 0;
    Running r(o, true);
    const int port = r.srv.port();
    const std::string me = "127.0.0.1:" + std::to_string(port);
    Reply a = ask(port, get("/", me));
    check(a.status == 200 && a.body == page && a.header("content-type") == "text/html; charset=utf-8" &&
              a.header("content-security-policy") == csp && a.header("x-frame-options") == "DENY",
          "GET / " + std::to_string(a.status));
    check(ask(port, get("/chat", "localhost:" + std::to_string(port))).body == page, "GET /chat");
    Reply h = ask(port, "HEAD / HTTP/1.1\r\nHost: " + me + "\r\nConnection: close\r\n\r\n");
    check(h.status == 200 && h.body.empty(), "HEAD / (Ollama's heartbeat)");
    check(ask(port, get("/", "evil.example")).status == 403, "a rebound host name: 403");
    const std::string body = "{\"messages\":[]}";
    auto post = [&](const std::string& origin) {
      return ask(port, "POST /v1/chat/completions HTTP/1.1\r\nHost: " + me + "\r\nConnection: close\r\nOrigin: " + origin +
                           "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body);
    };
    check(post("http://" + me).status == 200, "the page's own request");
    check(post("http://evil.example").status == 403, "another origin: 403");
    check(post("null").status == 403, "an opaque origin: 403");
  }
  {  // with a key: the page needs none, the API does
    http::Options o;
    o.port = 0;
    o.api_key = "k3y";
    Running r(o, true);
    const int port = r.srv.port();
    check(ask(port, get("/", "127.0.0.1")).status == 200, "keyed server: the page without the key");
    const std::string body = "{}";
    const std::string rq = "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Type: "
                           "application/json\r\nContent-Length: 2\r\n";
    check(ask(port, rq + "\r\n" + body).status == 401, "keyed server: the API without the key");
    check(ask(port, rq + "Authorization: Bearer k3y\r\n\r\n" + body).status == 200, "keyed server: the API with it");
  }
  {  // --no-webui
    http::Options o;
    o.port = 0;
    Running r(o, false);
    const int port = r.srv.port();
    Reply a = ask(port, get("/", "127.0.0.1"));
    check(a.status == 200 && a.body.find("Espetero is running") == 0 && a.header("content-type").rfind("text/plain", 0) == 0,
          "--no-webui: GET / is a line of text");
    check(ask(port, get("/chat", "127.0.0.1")).status == 404, "--no-webui: no /chat");
  }
  std::printf("webui_test: %d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
