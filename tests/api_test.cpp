// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// API request parsing without a model (make test): tool_choice,
// /v1/embeddings inputs; images by URL: the address checks, and
// fetches from a server on 127.0.0.1 (loopback refused unless allowed,
// content type, size cap, redirects counted and re-checked, timeout).
//
//   build/api_test
#include <arpa/inet.h>
#include <netinet/in.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "../src/engine/engine.h"
#include "../src/engine/requests.h"
#include "../src/http/http.h"
#include "../src/vision/fetch.h"

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

json::Value J(const char* s) { return json::parse(s); }

// The status of a RequestError the call throws, 0 if none.
template <class F>
int status_of(F&& f) {
  try {
    f();
  } catch (const RequestError& e) {
    return e.status;
  }
  return 0;
}

void test_tool_choice() {
  const json::Value tools = J(R"([{"type": "function", "function": {"name": "get_weather", "parameters": {}}},
                                  {"type": "function", "function": {"name": "get_time"}}])");
  const json::Value one = J(R"([{"type": "function", "function": {"name": "get_weather"}}])");
  auto parse = [&](const char* tc, const json::Value* t) {
    const json::Value v = J(tc);
    return parse_tool_choice(&v, t);
  };
  check(parse_tool_choice(nullptr, &tools).mode == ToolChoice::Auto, "absent: auto");
  check(parse("\"auto\"", &tools).mode == ToolChoice::Auto, "auto");
  check(parse("null", &tools).mode == ToolChoice::Auto, "null: auto");
  check(parse("\"none\"", &tools).mode == ToolChoice::None, "none");
  check(parse("\"none\"", nullptr).mode == ToolChoice::None, "none without tools");
  check(parse("\"required\"", &tools).mode == ToolChoice::Required, "required among two tools");
  {
    const ToolChoice c = parse("\"required\"", &one);
    check(c.mode == ToolChoice::Named && c.name == "get_weather", "required with one tool: that tool");
  }
  {
    const ToolChoice c = parse(R"({"type": "function", "function": {"name": "get_time"}})", &tools);
    check(c.mode == ToolChoice::Named && c.name == "get_time", "a named function");
  }
  {
    const ToolChoice c = parse(R"({"type": "allowed_tools", "allowed_tools": {"mode": "auto",
                                  "tools": [{"type": "function", "function": {"name": "get_time"}}]}})", &tools);
    check(c.mode == ToolChoice::Auto && c.allowed == std::vector<std::string>{"get_time"}, "allowed_tools auto");
  }
  {
    const ToolChoice c = parse(R"({"type": "allowed_tools", "allowed_tools": {"mode": "required",
                                  "tools": [{"type": "function", "function": {"name": "get_time"}}]}})", &tools);
    check(c.mode == ToolChoice::Named && c.name == "get_time", "allowed_tools required with one: named");
  }
  {
    const ToolChoice c = parse(R"({"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": [
                                  {"type": "function", "function": {"name": "get_time"}},
                                  {"type": "function", "function": {"name": "get_weather"}}]}})", &tools);
    check(c.mode == ToolChoice::Required && c.allowed.size() == 2, "allowed_tools required with two");
  }
  // Refused (422), never ignored.
  for (const char* bad : {"\"sometimes\"", "3", R"({"type": "function"})", R"({"type": "function", "function": {"name": "nope"}})",
                          R"({"type": "custom", "name": "x"})", R"({"type": "allowed_tools"})",
                          R"({"type": "allowed_tools", "allowed_tools": {"mode": "maybe", "tools": [{"type": "function", "function": {"name": "get_time"}}]}})",
                          R"({"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": []}})",
                          R"({"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": [{"type": "function", "function": {"name": "nope"}}]}})"})
    check(status_of([&] { parse(bad, &tools); }) == 422, std::string("refused: ") + bad);
  check(status_of([&] { parse("\"required\"", nullptr); }) == 422, "required without tools");
  check(status_of([&] {
          const json::Value empty = J("[]");
          parse("\"required\"", &empty);
        }) == 422,
        "required with an empty tools list");
  check(tool_names(&tools) == std::vector<std::string>{"get_weather", "get_time"}, "tool_names");
}

void test_embedding_request() {
  auto P = [](const char* body) { return parse_embedding_request(J(body), 1000, 64); };
  {
    const EmbeddingRequest r = P(R"({"input": "hello", "model": "any"})");
    check(r.texts == std::vector<std::string>{"hello"} && r.tokens.empty() && !r.base64 && r.size() == 1, "a string");
  }
  check(P(R"({"input": ["a", "b c"]})").texts == std::vector<std::string>{"a", "b c"}, "strings");
  check(P(R"({"input": [1, 2, 999]})").tokens == std::vector<std::vector<int32_t>>{{1, 2, 999}}, "token ids: one input");
  check(P(R"({"input": [[1, 2], [3]]})").tokens == std::vector<std::vector<int32_t>>{{1, 2}, {3}}, "token lists");
  check(P(R"({"input": "x", "encoding_format": "base64"})").base64, "base64");
  check(!P(R"({"input": "x", "encoding_format": "float", "dimensions": 64, "user": "u1"})").base64,
        "float, dimensions equal to the width, user");
  for (const char* bad : {R"({})", R"({"input": null})", R"({"input": ""})", R"({"input": []})", R"({"input": [""]})",
                          R"({"input": ["a", 1]})", R"({"input": [[1], "a"]})", R"({"input": [1, "a"]})", R"({"input": [1000]})",
                          R"({"input": [-1]})", R"({"input": [[]]})", R"({"input": [[1.5]]})", R"({"input": {"a": 1}})",
                          R"({"input": [true]})", R"({"input": "x", "dimensions": 32})", R"({"input": "x", "dimensions": "64"})",
                          R"({"input": "x", "encoding_format": "int8"})", R"([1, 2])"})
    check(status_of([&] { P(bad); }) == 422, std::string("refused: ") + bad);
  std::string many = "{\"input\": [";
  for (size_t i = 0; i <= kMaxEmbeddingInputs; ++i) many += i ? ",\"a\"" : "\"a\"";
  many += "]}";
  check(status_of([&] { parse_embedding_request(J(many.c_str()), 1000, 64); }) == 422, "more than 2048 inputs refused");
}

uint32_t v4(const char* s) {
  in_addr a{};
  ::inet_pton(AF_INET, s, &a);
  return ntohl(a.s_addr);
}

bool v6_blocked(const char* s) {
  in6_addr a{};
  if (::inet_pton(AF_INET6, s, &a) != 1) return false;
  return vision::blocked_ipv6(a.s6_addr);
}

void test_addresses() {
  for (const char* a : {"0.0.0.0", "0.1.2.3", "10.0.0.1", "10.255.255.255", "100.64.0.1", "100.127.255.254", "127.0.0.1",
                        "127.255.0.9", "169.254.169.254", "172.16.0.1", "172.31.255.255", "192.0.0.8", "192.168.1.1",
                        "198.18.0.1", "198.19.255.255", "224.0.0.1", "239.255.255.250", "240.0.0.1", "255.255.255.255"})
    check(vision::blocked_ipv4(v4(a)), std::string("blocked: ") + a);
  for (const char* a : {"1.1.1.1", "8.8.8.8", "9.255.255.255", "11.0.0.1", "100.63.255.255", "100.128.0.1", "126.255.255.255",
                        "128.0.0.1", "169.253.0.1", "172.15.255.255", "172.32.0.1", "192.0.1.1", "192.167.0.1", "192.169.0.1",
                        "198.17.255.255", "198.20.0.1", "223.255.255.255", "93.184.216.34"})
    check(!vision::blocked_ipv4(v4(a)), std::string("allowed: ") + a);
  for (const char* a : {"::", "::1", "fe80::1", "febf::1", "fec0::1", "fc00::1", "fdff:ffff::1", "ff02::1", "100::1",
                        "::ffff:127.0.0.1", "::ffff:10.1.2.3", "::ffff:169.254.169.254", "::127.0.0.1", "64:ff9b::a00:1",
                        "64:ff9b::7f00:1", "2002:c0a8:101::1", "2002:7f00:1::"})
    check(v6_blocked(a), std::string("blocked: ") + a);
  for (const char* a : {"2606:4700:4700::1111", "2001:4860:4860::8888", "::ffff:8.8.8.8", "64:ff9b::808:808", "2002:808:808::1",
                        "fe00::1", "fbff::1"})
    check(!v6_blocked(a), std::string("allowed: ") + a);
  check(vision::blocked_address_text("127.0.0.1") && vision::blocked_address_text("[::1]") &&
            vision::blocked_address_text("fe80::1%en0") && !vision::blocked_address_text("93.184.216.34") &&
            !vision::blocked_address_text("not an address"),
        "addresses in text form");
  std::vector<std::string> addrs;
  std::string err;
  bool blocked = false;
  check(vision::resolve_host("127.0.0.1", 5, addrs, err, nullptr, &blocked) && blocked && addrs == std::vector<std::string>{"127.0.0.1"},
        "a literal resolves to itself: " + err);
  check(vision::resolve_host("localhost", 5, addrs, err, nullptr, &blocked) && blocked, "localhost is loopback: " + err);
  vision::FetchOptions o;
  check(!vision::host_allowed("localhost", o, 5, err) && err.find("--image-urls-private") != std::string::npos,
        "localhost refused, naming --image-urls-private: " + err);
  check(!vision::host_allowed("[::1]", o, 5, err), "[::1] refused");
  check(!vision::host_allowed("2130706433", o, 5, err), "127.0.0.1 as a decimal number refused: " + err);
  o.allow_private = true;
  check(vision::host_allowed("localhost", o, 5, err), "localhost allowed with allow_private");
}

// 127.0.0.2 stands for a private address the redirect test reaches through a
// "public" 127.0.0.1 (no other address is reachable from a test).
bool only_127_0_0_2(const struct sockaddr* sa) {
  return sa->sa_family == AF_INET && ntohl(reinterpret_cast<const sockaddr_in*>(sa)->sin_addr.s_addr) == 0x7F000002u;
}

void test_fetch() {
  const std::string png = std::string("\x89PNG\r\n\x1a\n", 8) + std::string(200, 'p');  // bytes only: fetches do not decode
  http::Options ho;
  ho.port = 0;
  http::Server srv(ho);
  const int port = srv.port();
  const std::string base = "http://127.0.0.1:" + std::to_string(port);
  srv.route("GET", "/img.png", [&](const http::Request&, http::Response& res) { res.send(200, "image/png", png); });
  srv.route("GET", "/text", [&](const http::Request&, http::Response& res) { res.send(200, "text/plain", "hello"); });
  srv.route("GET", "/big", [&](const http::Request&, http::Response& res) {
    res.send(200, "image/png", std::string(3 << 20, 'b'));
  });
  srv.route("GET", "/missing", [&](const http::Request&, http::Response& res) { res.send(404, "text/plain", "no"); });
  srv.route("GET", "/slow", [&](const http::Request&, http::Response& res) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    res.send(200, "image/png", png);
  });
  // /r/N redirects N times, then the image; /to2 redirects to 127.0.0.2.
  srv.route_prefix("GET", "/r/", [&](const http::Request& req, http::Response& res) {
    const int n = std::atoi(req.path.substr(3).c_str());
    res.set_header("Location", n > 1 ? "/r/" + std::to_string(n - 1) : std::string("/img.png"));
    res.send(302, "text/plain", "");
  });
  srv.route("GET", "/to2", [&](const http::Request&, http::Response& res) {
    res.set_header("Location", "http://127.0.0.2:" + std::to_string(port) + "/img.png");
    res.send(302, "text/plain", "");
  });
  std::thread t([&] { srv.run(); });
  std::string bytes, mime, err;
  vision::FetchOptions o;
  check(!vision::fetch_image(base + "/img.png", o, bytes, mime, err) && err.find("--image-urls-private") != std::string::npos,
        "127.0.0.1 refused by default: " + err);
  // Other spellings of this machine, refused before any connection.
  for (const std::string& u : {"http://[::1]:" + std::to_string(port) + "/img.png", "http://127.0.0.1%2e:" + std::to_string(port) + "/img.png",
                               "http://0x7f.1:" + std::to_string(port) + "/img.png", "http://2130706433:" + std::to_string(port) + "/img.png",
                               "HTTP://LOCALHOST:" + std::to_string(port) + "/img.png", "http://[::ffff:127.0.0.1]:" + std::to_string(port) + "/img.png"})
    check(!vision::fetch_image(u, o, bytes, mime, err) && err.find("--image-urls-private") != std::string::npos,
          "refused by default: " + u + " (" + err + ")");
  o.allow_private = true;
  check(vision::fetch_image(base + "/img.png", o, bytes, mime, err) && bytes == png && mime == "image/png",
        "allowed with allow_private: " + err);
  check(!vision::fetch_image(base + "/text", o, bytes, mime, err) && err.find("image/") != std::string::npos,
        "text/plain refused: " + err);
  o.max_bytes = 1 << 20;
  check(!vision::fetch_image(base + "/big", o, bytes, mime, err) && err.find("limit") != std::string::npos,
        "a 3 MB body over a 1 MB cap refused: " + err);
  o.max_bytes = size_t(32) << 20;
  check(!vision::fetch_image(base + "/missing", o, bytes, mime, err) && err.find("404") != std::string::npos,
        "HTTP 404 refused: " + err);
  check(vision::fetch_image(base + "/r/3", o, bytes, mime, err) && bytes == png, "3 redirects followed: " + err);
  check(!vision::fetch_image(base + "/r/4", o, bytes, mime, err) && err.find("redirects") != std::string::npos,
        "4 redirects refused: " + err);
  for (const char* bad : {"ftp://127.0.0.1/x.png", "file:///etc/passwd", "http://user:pw@127.0.0.1/img.png", "http:///x.png", "not a url"})
    check(!vision::fetch_image(bad, o, bytes, mime, err), std::string("refused: ") + bad + " (" + err + ")");
  // Re-checked at each redirect: 127.0.0.1 is allowed here, 127.0.0.2 not.
  vision::FetchOptions r;
  r.blocked = only_127_0_0_2;
  check(vision::fetch_image(base + "/img.png", r, bytes, mime, err), "the test policy allows 127.0.0.1: " + err);
  check(!vision::fetch_image(base + "/to2", r, bytes, mime, err) && err.find("redirected") != std::string::npos &&
            err.find("127.0.0.2") != std::string::npos,
        "a redirect to a blocked address refused: " + err);
  o.timeout_s = 1;
  const auto t0 = std::chrono::steady_clock::now();
  const bool slow = vision::fetch_image(base + "/slow", o, bytes, mime, err);
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  check(!slow && err.find("timed out") != std::string::npos && s < 2.4, "a 2.5 s answer refused at 1 s: " + err + " (" + std::to_string(s) + " s)");
  std::string data_url;
  o.timeout_s = 20;
  check(vision::fetch_image_data_url(base + "/img.png", o, data_url, err) && data_url.rfind("data:image/png;base64,", 0) == 0,
        "as a data URL: " + err);
  srv.stop();
  t.join();
}

}  // namespace

int main() {
  std::printf("tool_choice\n");
  test_tool_choice();
  std::printf("embedding requests\n");
  test_embedding_request();
  std::printf("image URL addresses\n");
  test_addresses();
  std::printf("image URL fetches\n");
  test_fetch();
  std::printf("%d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
