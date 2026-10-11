// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Jinja2-subset template engine for Hugging Face chat templates.
//
// Goal: render a model's chat_template.jinja byte for byte like
// transformers' apply_chat_template(..., tokenize=False), which compiles the
// template with jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True,
// lstrip_blocks=True, extensions=[loopcontrols]) plus a JSON-style `tojson`
// filter and the `raise_exception` / `strftime_now` globals. The data model is
// Python's: values print like str(), dicts keep insertion order, missing keys
// are jinja2 Undefined, and scoping follows jinja2's compiler exactly
// (variables assigned in a loop body do not leak, hence namespace()).
//
// A Template is parsed once into a resolved AST (all variable references are
// bound to frame slots at parse time) and can be rendered concurrently from
// several threads; each render only touches its own state.
#pragma once

#include <cstdint>
#include <ctime>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "../util/json.h"

namespace espetero::tmpl {

// The Python exception class a jinja2 render would have raised.
enum class ErrorKind : uint8_t {
  Template,        // jinja2.TemplateError: raise_exception(msg) called by the template
  Syntax,          // TemplateSyntaxError (parse time)
  Assertion,       // TemplateAssertionError (e.g. unknown filter, compile time)
  Undefined,       // UndefinedError: an undefined value was used
  Security,        // SecurityError: sandbox violation (e.g. list.append)
  Runtime,         // TemplateRuntimeError
  FilterArgument,  // FilterArgumentError
  Type,            // Python TypeError
  Value,           // ValueError
  ZeroDivision,    // ZeroDivisionError
  Overflow,        // OverflowError
  Key,             // KeyError
  Index,           // IndexError
  Attribute,       // AttributeError
  Recursion,       // RecursionError
  Unsupported,     // valid jinja2 that this implementation does not support
};

class TemplateError : public std::runtime_error {
 public:
  // what() is the message exactly as str(exception) in Python for runtime
  // errors; syntax errors append " (line N)".
  TemplateError(ErrorKind kind, const std::string& message, int line = 0);
  ErrorKind kind() const noexcept { return kind_; }
  // Python class name of the equivalent exception ("TypeError", ...).
  const char* type_name() const noexcept;
  int line() const noexcept { return line_; }  // 0 when unknown

 private:
  ErrorKind kind_;
  int line_;
};

struct RenderOptions {
  // Wall-clock time seen by strftime_now() (local time, like Python's
  // datetime.now()). Unset: the real clock. Tests pin it.
  std::optional<std::tm> now;
};

struct TemplateImpl;

class Template {
 public:
  // Throws TemplateError (Syntax/Assertion/Unsupported) with line info.
  static std::shared_ptr<const Template> parse(std::string_view source);

  // Renders with `context` (a JSON object) as the template variables. Throws
  // TemplateError on raise_exception() or runtime errors.
  std::string render(const json::Value& context) const;
  std::string render(const json::Value& context, const RenderOptions& opts) const;

  ~Template();
  Template(const Template&) = delete;
  Template& operator=(const Template&) = delete;

  // The parsed program; opaque outside src/tmpl.
  const TemplateImpl& impl() const { return *impl_; }

 private:
  Template();
  std::unique_ptr<TemplateImpl> impl_;
};

// transformers' apply_chat_template(tokenize=False): builds the context
// {messages, tools, documents: None, add_generation_prompt, **special_tokens,
// **extra_kwargs} and renders. Like transformers, `tools` and `documents` are
// always defined (None when null), and extra_kwargs override special tokens.
// special_tokens is the tokenizer's special_tokens_map (bos_token, eos_token,
// unk_token, ...); pass null/{} if the caller merges them into extra_kwargs.
std::string apply_chat_template(const Template& tmpl, const json::Value& messages, const json::Value& tools,
                                bool add_generation_prompt, const json::Value& extra_kwargs,
                                const json::Value& special_tokens = json::Value(),
                                const RenderOptions& opts = RenderOptions());

}  // namespace espetero::tmpl
