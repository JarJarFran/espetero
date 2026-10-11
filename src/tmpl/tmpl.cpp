// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "tmpl.h"

#include "interp.h"

namespace espetero::tmpl {

namespace {

std::string with_line(ErrorKind kind, const std::string& msg, int line) {
  const bool compile_time = kind == ErrorKind::Syntax || kind == ErrorKind::Assertion || kind == ErrorKind::Unsupported;
  if (compile_time && line > 0) return msg + " (line " + std::to_string(line) + ")";
  return msg;
}

}  // namespace

TemplateError::TemplateError(ErrorKind kind, const std::string& message, int line)
    : std::runtime_error(with_line(kind, message, line)), kind_(kind), line_(line) {}

const char* TemplateError::type_name() const noexcept {
  switch (kind_) {
    case ErrorKind::Template: return "TemplateError";
    case ErrorKind::Syntax: return "TemplateSyntaxError";
    case ErrorKind::Assertion: return "TemplateAssertionError";
    case ErrorKind::Undefined: return "UndefinedError";
    case ErrorKind::Security: return "SecurityError";
    case ErrorKind::Runtime: return "TemplateRuntimeError";
    case ErrorKind::FilterArgument: return "FilterArgumentError";
    case ErrorKind::Type: return "TypeError";
    case ErrorKind::Value: return "ValueError";
    case ErrorKind::ZeroDivision: return "ZeroDivisionError";
    case ErrorKind::Overflow: return "OverflowError";
    case ErrorKind::Key: return "KeyError";
    case ErrorKind::Index: return "IndexError";
    case ErrorKind::Attribute: return "AttributeError";
    case ErrorKind::Recursion: return "RecursionError";
    case ErrorKind::Unsupported: return "NotImplementedError";
  }
  return "Exception";
}

Template::Template() = default;
Template::~Template() = default;

std::shared_ptr<const Template> Template::parse(std::string_view source) {
  std::shared_ptr<Template> t(new Template());
  t->impl_ = std::make_unique<TemplateImpl>();
  parse_template(source, *t->impl_);
  resolve_scopes(*t->impl_);
  return t;
}

std::string Template::render(const json::Value& context) const { return render(context, RenderOptions()); }

std::string Template::render(const json::Value& context, const RenderOptions& opts) const {
  Interp in(*impl_, {}, {&context}, opts);
  return in.run();
}

std::string render_impl(const TemplateImpl& t, const std::vector<std::pair<std::string_view, Val>>& fixed,
                        const std::vector<const json::Value*>& layers, const RenderOptions& opts) {
  Interp in(t, fixed, layers, opts);
  return in.run();
}

std::string apply_chat_template(const Template& tmpl, const json::Value& messages, const json::Value& tools,
                                bool add_generation_prompt, const json::Value& extra_kwargs,
                                const json::Value& special_tokens, const RenderOptions& opts) {
  // transformers: template.render(messages=..., tools=..., documents=None,
  // add_generation_prompt=..., **{**special_tokens_map, **kwargs}); nothing is
  // copied: the render borrows the caller's JSON.
  const std::vector<std::pair<std::string_view, Val>> fixed = {
      {"messages", from_json(messages)},
      {"tools", from_json(tools)},
      {"documents", Val::none()},
      {"add_generation_prompt", Val::boolean(add_generation_prompt)},
  };
  return render_impl(tmpl.impl(), fixed, {&extra_kwargs, &special_tokens}, opts);
}

}  // namespace espetero::tmpl
