// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// GLM-5.3-Flash without a GPU: its tool-call format (src/engine/glm_tools.cpp,
// docs/glm5.md 7): <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>,
// calls back to back after the content, string arguments as written and the
// other declared types as JSON.
//
//   build/glm_test
#include <cstdio>
#include <string>

#include "../src/engine/chat.h"
#include "../src/util/json.h"

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

std::string fn_name(const json::Value& calls, size_t i) { return calls[i].get("function")->get_string("name", ""); }
std::string fn_args(const json::Value& calls, size_t i) { return calls[i].get("function")->get_string("arguments", ""); }

void test_tool_calls() {
  const json::Value tools = json::parse(R"([{"type": "function", "function": {"name": "get_weather", "parameters": {
      "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
      "units": {"type": "string"}, "opts": {"type": "object"}, "zip": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "get_time"}}])");
  json::Value calls;
  // One call after the content: the content is returned, the integer parsed, the string kept.
  std::string before = parse_glm_tool_calls(
      "Let me check.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris, France</arg_value>"
      "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>", tools, calls);
  check(before == "Let me check." && calls.is_array() && calls.size() == 1, "one call: content before, one call");
  check(calls.size() == 1 && fn_name(calls, 0) == "get_weather" && fn_args(calls, 0) == R"({"city":"Paris, France","days":3})",
        "one call: name and typed arguments: " + (calls.size() ? fn_args(calls, 0) : "none"));
  check(calls.size() == 1 && calls[0].get_string("type", "") == "function" && calls[0].get_string("id", "").rfind("call_", 0) == 0,
        "one call: OpenAI shape");
  // Two calls back to back, nothing before: a declared string that looks like a number stays a string,
  // an undeclared value is JSON when it parses, an object parameter is an object.
  before = parse_glm_tool_calls(
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value><arg_key>zip</arg_key><arg_value>00100</arg_value>"
      "<arg_key>opts</arg_key><arg_value>{\"a\": [1, 2]}</arg_value><arg_key>extra</arg_key><arg_value>true</arg_value></tool_call>"
      "<tool_call>get_time</tool_call>", tools, calls);
  check(before.empty() && calls.is_array() && calls.size() == 2, "two calls: both parsed, nothing before");
  check(calls.size() == 2 && fn_args(calls, 0) == R"({"city":"Rome","zip":"00100","opts":{"a":[1,2]},"extra":true})",
        "two calls: types: " + (calls.size() ? fn_args(calls, 0) : "none"));
  check(calls.size() == 2 && fn_name(calls, 1) == "get_time" && fn_args(calls, 1) == "{}", "two calls: a call without arguments");
  // A value with markup, newlines and an unclosed call (the reply was cut at max_tokens).
  before = parse_glm_tool_calls("<tool_call>get_weather<arg_key>city</arg_key><arg_value>a <b> c\nd</arg_value>"
                            "<arg_key>units</arg_key><arg_value>C", tools, calls);
  check(calls.is_array() && calls.size() == 1 && fn_args(calls, 0) == "{\"city\":\"a <b> c\\nd\",\"units\":\"C\"}",
        "unclosed call: values up to the cut: " + (calls.size() ? fn_args(calls, 0) : "none"));
  // Plain text with the word stays text; a call without a name is not a call.
  before = parse_glm_tool_calls("No call here, just <tool_call> text.", tools, calls);
  check(calls.is_null() && before == "No call here, just <tool_call> text.", "no call");
  before = parse_glm_tool_calls("<tool_call><arg_key>x</arg_key><arg_value>1</arg_value></tool_call>", tools, calls);
  check(calls.is_null(), "a call without a name is not a call");
  // Without declared tools: every value is JSON when it parses.
  before = parse_glm_tool_calls("<tool_call>f<arg_key>n</arg_key><arg_value>12</arg_value><arg_key>s</arg_key><arg_value>x y</arg_value></tool_call>",
                            json::Value(), calls);
  check(calls.is_array() && calls.size() == 1 && fn_args(calls, 0) == R"({"n":12,"s":"x y"})", "undeclared tools: JSON when it parses");
  // The body test parse_tool_calls (chat.cpp) uses to hand a call over: Qwen's "\n<function=" bodies stay with it.
  check(glm_tool_call_body("get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value>") && glm_tool_call_body("get_time") &&
            !glm_tool_call_body("\n<function=get_time>\n</function>\n") && !glm_tool_call_body("<arg_key>x</arg_key>"),
        "body classification");
}

}  // namespace

int main() {
  std::printf("GLM tool calls\n");
  test_tool_calls();
  std::printf("%d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
