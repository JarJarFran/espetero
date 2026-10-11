// SPDX-License-Identifier: MIT AND BSD-3-Clause
// Copyright (c) 2026 Francisco José García Barbero
//
// Portions are derived from Jinja2, Copyright 2007 Pallets; BSD-3-Clause License, see THIRD_PARTY_NOTICES.md.

// Lexer and parser. Both follow jinja2 (lexer.py / parser.py) rule for rule,
// configured like transformers: trim_blocks=True, lstrip_blocks=True,
// keep_trailing_newline=False, default delimiters. Whitespace handling is
// where chat templates are most fragile, so the lexer reproduces jinja2's
// regex behaviour exactly, including `{%+`, `+%}` and the lstrip
// "line_starting" state.
#include <charconv>
#include <cmath>
#include <cstdlib>

#include "ast.h"
#include "unicode.h"

namespace espetero::tmpl {

namespace {

[[noreturn]] void syntax_error(const std::string& msg, int line) {
  throw TemplateError(ErrorKind::Syntax, msg, line);
}

// ================================================================ lexer
enum class Tok : uint8_t { Data, VarBegin, VarEnd, BlockBegin, BlockEnd, Name, String, Int, Float, Op, Eof };

enum class Op : uint8_t {
  Add,
  Sub,
  Div,
  FloorDiv,
  Mul,
  Mod,
  Pow,
  Tilde,
  LBracket,
  RBracket,
  LParen,
  RParen,
  LBrace,
  RBrace,
  Eq,
  Ne,
  Gt,
  Ge,
  Lt,
  Le,
  Assign,
  Dot,
  Colon,
  Pipe,
  Comma,
  Semicolon,
};

struct OpSpelling {
  const char* text;
  Op op;
};
// Longest first, like jinja2's operator regex.
constexpr OpSpelling kOps[] = {
    {"//", Op::FloorDiv}, {"**", Op::Pow},    {"==", Op::Eq},       {"!=", Op::Ne},       {">=", Op::Ge},
    {"<=", Op::Le},       {"+", Op::Add},     {"-", Op::Sub},       {"/", Op::Div},       {"*", Op::Mul},
    {"%", Op::Mod},       {"~", Op::Tilde},   {"[", Op::LBracket},  {"]", Op::RBracket},  {"(", Op::LParen},
    {")", Op::RParen},    {"{", Op::LBrace},  {"}", Op::RBrace},    {">", Op::Gt},        {"<", Op::Lt},
    {"=", Op::Assign},    {".", Op::Dot},     {":", Op::Colon},     {"|", Op::Pipe},      {",", Op::Comma},
    {";", Op::Semicolon},
};

const char* op_text(Op op) {
  for (const auto& o : kOps)
    if (o.op == op) return o.text;
  return "?";
}

struct Token {
  Tok type = Tok::Eof;
  Op op = Op::Add;
  int line = 1;
  std::string value;  // name, decoded string literal or template data
  int64_t ival = 0;
  double fval = 0;
};

bool is_digit(char c) { return c >= '0' && c <= '9'; }

class Lexer {
 public:
  explicit Lexer(std::string_view src) {
    // jinja2 splits on \r\n, \r and \n, drops one trailing newline
    // (keep_trailing_newline=False) and rejoins with \n.
    src_.reserve(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
      if (src[i] == '\r') {
        src_.push_back('\n');
        if (i + 1 < src.size() && src[i + 1] == '\n') ++i;
      } else {
        src_.push_back(src[i]);
      }
    }
    if (!src_.empty() && src_.back() == '\n') src_.pop_back();
  }

  std::vector<Token> run() {
    size_t pos = 0;
    while (pos < src_.size()) {
      // Find the next tag opener: {{ {% {#.
      size_t tag = pos;
      for (;;) {
        tag = src_.find('{', tag);
        if (tag == std::string::npos || tag + 1 >= src_.size()) {
          tag = std::string::npos;
          break;
        }
        const char c = src_[tag + 1];
        if (c == '{' || c == '%' || c == '#') break;
        ++tag;
      }
      if (tag == std::string::npos) {
        emit_data(src_.substr(pos), line_);
        break;
      }
      const char kind = src_[tag + 1];
      char sign = 0;
      size_t after = tag + 2;
      if (after < src_.size() && (src_[after] == '-' || src_[after] == '+')) sign = src_[after++];
      size_t raw_end = 0;
      const bool raw = kind == '%' && match_raw_begin(tag, &raw_end);
      std::string text = src_.substr(pos, tag - pos);
      const int text_line = line_;
      line_ += count_nl(text);
      apply_lstrip(text, sign, kind != '{');
      emit_data(std::move(text), text_line);
      if (raw) {
        line_ += count_nl(std::string_view(src_).substr(tag, raw_end - tag));
        line_starting_ = src_[raw_end - 1] == '\n';
        pos = lex_raw(raw_end);
      } else if (kind == '#') {
        pos = lex_comment(after);
      } else {
        push(kind == '{' ? Tok::VarBegin : Tok::BlockBegin, line_);
        line_starting_ = false;
        pos = lex_tag(after, kind == '{');
      }
    }
    Token eof;
    eof.type = Tok::Eof;
    eof.line = line_;
    toks_.push_back(std::move(eof));
    return std::move(toks_);
  }

 private:
  static int count_nl(std::string_view s) {
    int n = 0;
    for (char c : s) n += c == '\n';
    return n;
  }

  // Length of the Unicode whitespace run at s[i] (\s*).
  size_t space_run(size_t i) const {
    size_t j = i;
    while (j < src_.size()) {
      size_t k = j;
      const uint32_t c = uni::next(src_, k);
      if (!uni::is_space(c)) break;
      j = k;
    }
    return j - i;
  }

  // jinja2 OptionalLStrip handling of the text before a tag.
  void apply_lstrip(std::string& text, char sign, bool lstrip_allowed) {
    if (sign == '-') {
      // str.rstrip(): Unicode whitespace.
      size_t end = text.size();
      while (end > 0) {
        size_t k = end - 1;
        while (k > 0 && (static_cast<unsigned char>(text[k]) & 0xC0) == 0x80) --k;
        size_t t = k;
        if (!uni::is_space(uni::next(text, t))) break;
        end = k;
      }
      text.resize(end);
    } else if (sign != '+' && lstrip_allowed) {
      const size_t nl = text.rfind('\n');
      const size_t l_pos = nl == std::string::npos ? 0 : nl + 1;
      if ((l_pos > 0 || line_starting_) && l_pos < text.size()) {
        size_t i = l_pos;
        bool all_space = true;
        while (i < text.size()) {
          if (!uni::is_space(uni::next(text, i))) {
            all_space = false;
            break;
          }
        }
        if (all_space) text.resize(l_pos);
      }
    }
  }

  void emit_data(std::string text, int line) {
    if (text.empty()) return;
    Token t;
    t.type = Tok::Data;
    t.line = line;
    t.value = std::move(text);
    toks_.push_back(std::move(t));
  }

  void push(Tok type, int line) {
    Token t;
    t.type = type;
    t.line = line;
    toks_.push_back(std::move(t));
  }

  // `{%[-+]?\s*raw\s*(?:-%}\s*|%})` at `tag`; sets *end past the match.
  bool match_raw_begin(size_t tag, size_t* end) const {
    size_t i = tag + 2;
    if (i < src_.size() && (src_[i] == '-' || src_[i] == '+')) ++i;
    i += space_run(i);
    if (src_.compare(i, 3, "raw") != 0) return false;
    i += 3;
    i += space_run(i);
    if (src_.compare(i, 3, "-%}") == 0) {
      i += 3;
      i += space_run(i);
    } else if (src_.compare(i, 2, "%}") == 0) {
      i += 2;
    } else {
      return false;
    }
    *end = i;
    return true;
  }

  size_t lex_raw(size_t pos) {
    // (.*?)((?:{%[-+]?)\s*endraw\s*(?:+%}|-%}\s*|%}\n?))
    for (size_t p = pos; p + 1 < src_.size(); ++p) {
      if (src_[p] != '{' || src_[p + 1] != '%') continue;
      size_t i = p + 2;
      char sign = 0;
      if (i < src_.size() && (src_[i] == '-' || src_[i] == '+')) sign = src_[i++];
      i += space_run(i);
      if (src_.compare(i, 6, "endraw") != 0) continue;
      i += 6;
      i += space_run(i);
      size_t end;
      if (src_.compare(i, 3, "+%}") == 0) {
        end = i + 3;
      } else if (src_.compare(i, 3, "-%}") == 0) {
        end = i + 3;
        end += space_run(end);
      } else if (src_.compare(i, 2, "%}") == 0) {
        end = i + 2;
        if (end < src_.size() && src_[end] == '\n') ++end;
      } else {
        continue;
      }
      std::string text = src_.substr(pos, p - pos);
      const int text_line = line_;
      apply_lstrip(text, sign, true);
      emit_data(std::move(text), text_line);
      line_ += count_nl(std::string_view(src_).substr(pos, end - pos));
      line_starting_ = src_[end - 1] == '\n';
      return end;
    }
    syntax_error("Missing end of raw directive", line_);
  }

  size_t lex_comment(size_t pos) {
    // (.*?)((?:+#}|-#}\s*|#}\n?)): the earliest "#}" closes the comment; a
    // '-' or '+' right before it belongs to the closing tag.
    const size_t q = src_.find("#}", pos);
    if (q == std::string::npos) syntax_error("Missing end of comment tag", line_);
    size_t end = q + 2;
    if (q > pos && src_[q - 1] == '-') {
      end += space_run(end);
    } else if (q > pos && src_[q - 1] == '+') {
      // no trim_blocks
    } else if (end < src_.size() && src_[end] == '\n') {
      ++end;
    }
    line_ += count_nl(std::string_view(src_).substr(pos, end - pos));
    line_starting_ = src_[end - 1] == '\n';
    return end;
  }

  // Tokens of an expression tag until its end delimiter.
  size_t lex_tag(size_t pos, bool variable) {
    std::vector<char> balance;
    for (;;) {
      if (pos >= src_.size()) return pos;  // the parser reports the unexpected end
      if (balance.empty()) {
        size_t end = 0;
        if (variable) {
          if (src_.compare(pos, 3, "-}}") == 0) {
            end = pos + 3;
            end += space_run(end);
          } else if (src_.compare(pos, 2, "}}") == 0) {
            end = pos + 2;
          }
        } else {
          if (src_.compare(pos, 3, "+%}") == 0) {
            end = pos + 3;
          } else if (src_.compare(pos, 3, "-%}") == 0) {
            end = pos + 3;
            end += space_run(end);
          } else if (src_.compare(pos, 2, "%}") == 0) {
            end = pos + 2;
            if (end < src_.size() && src_[end] == '\n') ++end;
          }
        }
        if (end) {
          push(variable ? Tok::VarEnd : Tok::BlockEnd, line_);
          line_ += count_nl(std::string_view(src_).substr(pos, end - pos));
          line_starting_ = src_[end - 1] == '\n';
          return end;
        }
      }
      const unsigned char c = static_cast<unsigned char>(src_[pos]);
      // whitespace
      if (const size_t ws = space_run(pos)) {
        line_ += count_nl(std::string_view(src_).substr(pos, ws));
        pos += ws;
        continue;
      }
      // numbers
      if (is_digit(char(c))) {
        if (size_t e = match_float(pos)) {
          Token t;
          t.type = Tok::Float;
          t.line = line_;
          std::string digits;
          for (size_t i = pos; i < e; ++i)
            if (src_[i] != '_') digits.push_back(src_[i]);
          t.fval = std::strtod(digits.c_str(), nullptr);
          toks_.push_back(std::move(t));
          pos = e;
          continue;
        }
        pos = lex_int(pos);
        continue;
      }
      // names
      if (c == '_' || ((c | 0x20) >= 'a' && (c | 0x20) <= 'z') || c >= 0x80) {
        size_t i = pos;
        while (i < src_.size()) {
          const unsigned char d = static_cast<unsigned char>(src_[i]);
          if (d == '_' || is_digit(char(d)) || ((d | 0x20) >= 'a' && (d | 0x20) <= 'z')) {
            ++i;
            continue;
          }
          if (d >= 0x80) {
            size_t k = i;
            if (uni::is_word(uni::next(src_, k))) {
              i = k;
              continue;
            }
          }
          break;
        }
        if (i == pos) syntax_error(std::string("unexpected char '") + char(c) + "' at " + std::to_string(pos), line_);
        Token t;
        t.type = Tok::Name;
        t.line = line_;
        t.value = src_.substr(pos, i - pos);
        toks_.push_back(std::move(t));
        pos = i;
        continue;
      }
      // strings
      if (c == '\'' || c == '"') {
        size_t i = pos + 1;
        while (i < src_.size() && src_[i] != char(c)) i += src_[i] == '\\' ? 2 : 1;
        if (i >= src_.size()) syntax_error(std::string("unexpected char '") + char(c) + "' at " + std::to_string(pos), line_);
        Token t;
        t.type = Tok::String;
        t.line = line_;
        t.value = decode_string(std::string_view(src_).substr(pos + 1, i - pos - 1), line_);
        line_ += count_nl(std::string_view(src_).substr(pos, i - pos));
        toks_.push_back(std::move(t));
        pos = i + 1;
        continue;
      }
      // operators
      bool matched = false;
      for (const auto& o : kOps) {
        const size_t n = std::strlen(o.text);
        if (src_.compare(pos, n, o.text) != 0) continue;
        if (o.op == Op::LBrace || o.op == Op::LParen || o.op == Op::LBracket) {
          balance.push_back(o.op == Op::LBrace ? '}' : o.op == Op::LParen ? ')' : ']');
        } else if (o.op == Op::RBrace || o.op == Op::RParen || o.op == Op::RBracket) {
          if (balance.empty()) syntax_error(std::string("unexpected '") + o.text + "'", line_);
          const char want = balance.back();
          balance.pop_back();
          if (want != o.text[0])
            syntax_error(std::string("unexpected '") + o.text + "', expected '" + want + "'", line_);
        }
        Token t;
        t.type = Tok::Op;
        t.op = o.op;
        t.line = line_;
        toks_.push_back(std::move(t));
        pos += n;
        matched = true;
        break;
      }
      if (!matched) {
        std::string ch = src_.substr(pos, 1);
        syntax_error("unexpected char " + py_repr_str(ch) + " at " + std::to_string(pos), line_);
      }
    }
  }

  // (\d+_)*\d+ at i; returns the end or npos.
  size_t digit_run(size_t i) const {
    if (i >= src_.size() || !is_digit(src_[i])) return std::string::npos;
    while (i < src_.size() && is_digit(src_[i])) ++i;
    while (i + 1 < src_.size() && src_[i] == '_' && is_digit(src_[i + 1])) {
      ++i;
      while (i < src_.size() && is_digit(src_[i])) ++i;
    }
    return i;
  }

  // jinja2 float_re; returns the end or 0.
  size_t match_float(size_t pos) const {
    if (pos > 0 && src_[pos - 1] == '.') return 0;  // (?<!\.)
    const size_t p = digit_run(pos);
    if (p == std::string::npos) return 0;
    auto exponent = [&](size_t q) -> size_t {
      if (q >= src_.size() || (src_[q] != 'e' && src_[q] != 'E')) return 0;
      ++q;
      if (q < src_.size() && (src_[q] == '+' || src_[q] == '-')) ++q;
      const size_t e = digit_run(q);
      return e == std::string::npos ? 0 : e;
    };
    // (\.digits)? exponent
    if (p < src_.size() && src_[p] == '.') {
      const size_t f = digit_run(p + 1);
      if (f != std::string::npos) {
        if (size_t e = exponent(f)) return e;
      }
    }
    if (size_t e = exponent(p)) return e;
    // \.digits
    if (p < src_.size() && src_[p] == '.') {
      const size_t f = digit_run(p + 1);
      if (f != std::string::npos) return f;
    }
    return 0;
  }

  size_t lex_int(size_t pos) {
    size_t i = pos;
    int base = 10;
    auto prefixed = [&](char letter, int b, auto valid) -> bool {
      if (src_[pos] != '0' || pos + 1 >= src_.size() || (src_[pos + 1] | 0x20) != letter) return false;
      size_t j = pos + 2;
      size_t end = 0;
      for (;;) {  // (_?[digits])+
        size_t k = j;
        if (k < src_.size() && src_[k] == '_') ++k;
        if (k < src_.size() && valid(src_[k])) {
          j = k + 1;
          end = j;
        } else {
          break;
        }
      }
      if (!end) return false;
      i = end;
      base = b;
      return true;
    };
    const bool pref = prefixed('b', 2, [](char c) { return c == '0' || c == '1'; }) ||
                      prefixed('o', 8, [](char c) { return c >= '0' && c <= '7'; }) ||
                      prefixed('x', 16, [](char c) { return is_digit(c) || ((c | 0x20) >= 'a' && (c | 0x20) <= 'f'); });
    if (!pref) {
      if (src_[pos] != '0') {  // [1-9](_?\d)*
        i = pos + 1;
        for (;;) {
          size_t k = i;
          if (k < src_.size() && src_[k] == '_') ++k;
          if (k < src_.size() && is_digit(src_[k]))
            i = k + 1;
          else
            break;
        }
      } else {  // 0(_?0)*
        i = pos + 1;
        for (;;) {
          size_t k = i;
          if (k < src_.size() && src_[k] == '_') ++k;
          if (k < src_.size() && src_[k] == '0')
            i = k + 1;
          else
            break;
        }
      }
    }
    std::string digits;
    for (size_t k = pref ? pos + 2 : pos; k < i; ++k)
      if (src_[k] != '_') digits.push_back(src_[k]);
    Token t;
    t.type = Tok::Int;
    t.line = line_;
    int64_t v = 0;
    auto r = std::from_chars(digits.data(), digits.data() + digits.size(), v, base);
    if (r.ec != std::errc()) syntax_error("integer literal too large", line_);
    t.ival = v;
    toks_.push_back(std::move(t));
    return i;
  }

  // Python: raw.encode("ascii", "backslashreplace").decode("unicode-escape").
  static std::string decode_string(std::string_view raw, int line) {
    std::string out;
    out.reserve(raw.size());
    static const char* hex = "0123456789abcdef";
    size_t i = 0;
    while (i < raw.size()) {
      const unsigned char c = static_cast<unsigned char>(raw[i]);
      if (c != '\\') {
        out.push_back(char(c));
        ++i;
        continue;
      }
      if (i + 1 >= raw.size()) syntax_error("\\ at end of string", line);
      const unsigned char e = static_cast<unsigned char>(raw[i + 1]);
      if (e >= 0x80) {
        // backslashreplace turned the character into \xNN / \uNNNN, whose
        // backslash the preceding one escapes: the text survives literally.
        size_t k = i + 1;
        const uint32_t cp = uni::next(raw, k);
        out.push_back('\\');
        if (cp <= 0xFF) {
          out.push_back('x');
          out.push_back(hex[cp >> 4]);
          out.push_back(hex[cp & 15]);
        } else if (cp <= 0xFFFF) {
          out.push_back('u');
          for (int s = 12; s >= 0; s -= 4) out.push_back(hex[(cp >> s) & 15]);
        } else {
          out.push_back('U');
          for (int s = 28; s >= 0; s -= 4) out.push_back(hex[(cp >> s) & 15]);
        }
        i = k;
        continue;
      }
      i += 2;
      auto hexval = [&](size_t n, const char* what) -> uint32_t {
        if (i + n > raw.size()) syntax_error(std::string("truncated ") + what + " escape", line);
        uint32_t v = 0;
        for (size_t k = 0; k < n; ++k) {
          const char h = raw[i + k];
          int d;
          if (is_digit(h))
            d = h - '0';
          else if ((h | 0x20) >= 'a' && (h | 0x20) <= 'f')
            d = (h | 0x20) - 'a' + 10;
          else
            syntax_error(std::string("truncated ") + what + " escape", line);
          v = v * 16 + uint32_t(d);
        }
        i += n;
        return v;
      };
      switch (e) {
        case '\n': break;  // line continuation
        case '\\': out.push_back('\\'); break;
        case '\'': out.push_back('\''); break;
        case '"': out.push_back('"'); break;
        case 'a': out.push_back('\a'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'v': out.push_back('\v'); break;
        case 'x': uni::append(out, hexval(2, "\\xXX")); break;
        case 'u': {
          const uint32_t cp = hexval(4, "\\uXXXX");
          uni::append(out, cp >= 0xD800 && cp <= 0xDFFF ? 0xFFFD : cp);
          break;
        }
        case 'U': {
          const uint32_t cp = hexval(8, "\\UXXXXXXXX");
          if (cp > 0x10FFFF) syntax_error("illegal Unicode character", line);
          uni::append(out, cp >= 0xD800 && cp <= 0xDFFF ? 0xFFFD : cp);
          break;
        }
        case 'N': throw TemplateError(ErrorKind::Unsupported, "\\N{...} escapes are not supported", line);
        default:
          if (e >= '0' && e <= '7') {
            uint32_t v = uint32_t(e - '0');
            for (int k = 0; k < 2 && i < raw.size() && raw[i] >= '0' && raw[i] <= '7'; ++k, ++i)
              v = v * 8 + uint32_t(raw[i] - '0');
            uni::append(out, v);
          } else {
            out.push_back('\\');  // unknown escapes are kept
            out.push_back(char(e));
          }
      }
    }
    return out;
  }

  std::string src_;
  std::vector<Token> toks_;
  int line_ = 1;
  bool line_starting_ = true;
};

// ================================================================ parser
const char* describe(const Token& t) {
  switch (t.type) {
    case Tok::Data: return "template data / text";
    case Tok::VarBegin: return "begin of print statement";
    case Tok::VarEnd: return "end of print statement";
    case Tok::BlockBegin: return "begin of statement block";
    case Tok::BlockEnd: return "end of statement block";
    case Tok::String: return "string";
    case Tok::Int: return "integer";
    case Tok::Float: return "float";
    case Tok::Eof: return "end of template";
    case Tok::Op: return op_text(t.op);
    case Tok::Name: return t.value.c_str();
  }
  return "?";
}

// jinja2's recursive parser hits Python's recursion limit at roughly 80
// nested brackets; we stop at a similar depth instead of risking the C++
// stack of small server threads.
constexpr int kMaxNesting = 300;  // ~100 levels: each bracket level enters 3 guarded rules

class Parser {
 public:
  Parser(std::vector<Token> toks, std::unordered_set<std::string>* literals)
      : t_(std::move(toks)), literals_(literals) {}

  Body parse() { return subparse({}); }

 private:
  // ---------------------------------------------------------- token stream
  const Token& cur() const { return t_[p_]; }
  const Token& look() const { return t_[p_ + 1 < t_.size() ? p_ + 1 : p_]; }
  const Token& advance() {
    const Token& t = t_[p_];
    if (p_ + 1 < t_.size()) ++p_;
    return t;
  }
  bool is_op(Op op) const { return cur().type == Tok::Op && cur().op == op; }
  bool is_name(const char* v) const { return cur().type == Tok::Name && cur().value == v; }
  bool skip_op(Op op) {
    if (!is_op(op)) return false;
    advance();
    return true;
  }
  bool skip_name(const char* v) {
    if (!is_name(v)) return false;
    advance();
    return true;
  }
  [[noreturn]] void fail(const std::string& msg, int line = 0) { syntax_error(msg, line ? line : cur().line); }
  [[noreturn]] void fail_expected(const std::string& what) {
    if (cur().type == Tok::Eof) fail("unexpected end of template, expected '" + what + "'.");
    fail("expected token '" + what + "', got '" + describe(cur()) + "'");
  }
  const Token& expect(Tok type, const char* what) {
    if (cur().type != type) fail_expected(what);
    return advance();
  }
  void expect_op(Op op) {
    if (!is_op(op)) fail_expected(op_text(op));
    advance();
  }
  void expect_name(const char* v) {
    if (!is_name(v)) fail_expected(v);
    advance();
  }

  // ---------------------------------------------------------- statements
  Body subparse(const std::vector<const char*>& end_tokens) {
    Body body;
    std::unique_ptr<OutputStmt> data;
    auto flush = [&] {
      if (data) body.push_back(std::move(data));
      data.reset();
    };
    auto add = [&](int line) -> OutputStmt::Item& {
      if (!data) data = std::make_unique<OutputStmt>(line);
      data->items.emplace_back();
      return data->items.back();
    };
    if (!end_tokens.empty()) end_stack_.push_back(end_tokens);
    while (cur().type != Tok::Eof) {
      const Token& tok = cur();
      if (tok.type == Tok::Data) {
        add(tok.line).text = tok.value;
        advance();
      } else if (tok.type == Tok::VarBegin) {
        const int line = tok.line;
        advance();
        ExprPtr e = parse_tuple(false, true, {}, false, false);
        add(line).expr = std::move(e);
        expect(Tok::VarEnd, "end of print statement");
      } else if (tok.type == Tok::BlockBegin) {
        flush();
        advance();
        for (const char* e : end_tokens)
          if (is_name(e)) {
            end_stack_.pop_back();
            return body;
          }
        parse_statement(body);
        expect(Tok::BlockEnd, "end of statement block");
      } else {
        fail("internal parsing error");
      }
    }
    flush();
    if (!end_tokens.empty()) end_stack_.pop_back();
    return body;
  }

  [[noreturn]] void fail_eof(const std::vector<const char*>& end_tokens) {
    std::string msg = "Unexpected end of template.";
    if (!end_tokens.empty()) {
      msg += " Jinja was looking for the following tags: ";
      for (size_t i = 0; i < end_tokens.size(); ++i) {
        if (i) msg += " or ";
        msg += "'" + std::string(end_tokens[i]) + "'";
      }
      msg += ".";
    }
    if (!tag_stack_.empty()) msg += " The innermost block that needs to be closed is '" + tag_stack_.back() + "'.";
    fail(msg);
  }

  Body parse_statements(const std::vector<const char*>& end_tokens, bool drop_needle) {
    skip_op(Op::Colon);
    expect(Tok::BlockEnd, "end of statement block");
    Body body = subparse(end_tokens);
    if (cur().type == Tok::Eof) fail_eof(end_tokens);
    if (drop_needle) advance();
    return body;
  }

  void parse_statement(Body& out) {
    Nest nest(*this);
    const Token& tok = cur();
    if (tok.type != Tok::Name) fail("tag name expected", tok.line);
    const std::string name = tok.value;
    const int line = tok.line;
    tag_stack_.push_back(name);
    if (name == "for")
      out.push_back(parse_for());
    else if (name == "if")
      out.push_back(parse_if());
    else if (name == "set")
      out.push_back(parse_set());
    else if (name == "macro")
      out.push_back(parse_macro());
    else if (name == "call")
      out.push_back(parse_call_block());
    else if (name == "filter")
      out.push_back(parse_filter_block());
    else if (name == "with")
      out.push_back(parse_with());
    else if (name == "print")
      out.push_back(parse_print());
    else if (name == "break" || name == "continue") {
      advance();
      out.push_back(std::make_unique<LoopCtlStmt>(name == "break" ? SK::Break : SK::Continue, line));
    } else if (name == "generation") {
      out.push_back(parse_generation());
    } else if (name == "block" || name == "extends" || name == "include" || name == "import" || name == "from" ||
               name == "autoescape") {
      throw TemplateError(ErrorKind::Unsupported, "the '" + name + "' tag is not supported", line);
    } else {
      tag_stack_.pop_back();
      std::string msg = "Encountered unknown tag '" + name + "'.";
      bool expected = false;
      for (const auto& ends : end_stack_)
        for (const char* e : ends) expected |= name == e;
      if (!end_stack_.empty()) {
        std::string looking;
        for (size_t i = 0; i < end_stack_.back().size(); ++i) {
          if (i) looking += " or ";
          looking += "'" + std::string(end_stack_.back()[i]) + "'";
        }
        if (expected)
          msg += " You probably made a nesting mistake. Jinja is expecting this tag, but currently looking for " +
                 looking + ".";
        else
          msg += " Jinja was looking for the following tags: " + looking + ".";
      }
      if (!tag_stack_.empty()) msg += " The innermost block that needs to be closed is '" + tag_stack_.back() + "'.";
      fail(msg, line);
    }
    tag_stack_.pop_back();
  }

  StmtPtr parse_for() {
    const int line = cur().line;
    expect_name("for");
    auto node = std::make_unique<ForStmt>(line);
    node->target = parse_assign_target(true, false, {"in"}, false);
    expect_name("in");
    node->iter = parse_tuple(false, false, {"recursive"}, false, false);
    if (skip_name("if")) node->test = parse_expression(true);
    node->recursive = skip_name("recursive");
    node->body = parse_statements({"endfor", "else"}, false);
    if (advance().value != "endfor") node->else_ = parse_statements({"endfor"}, true);
    return node;
  }

  StmtPtr parse_if() {
    const int line = cur().line;
    expect_name("if");
    auto result = std::make_unique<IfStmt>(line);
    IfStmt* node = result.get();
    for (;;) {
      node->test = parse_tuple(false, false, {}, false, false);
      node->body = parse_statements({"elif", "else", "endif"}, false);
      const Token& tok = advance();
      if (tok.type == Tok::Name && tok.value == "elif") {
        result->elifs.push_back(std::make_unique<IfStmt>(cur().line));
        node = result->elifs.back().get();
        continue;
      }
      if (tok.type == Tok::Name && tok.value == "else") result->else_ = parse_statements({"endif"}, true);
      break;
    }
    return result;
  }

  StmtPtr parse_set() {
    const int line = advance().line;
    ExprPtr target = parse_assign_target(true, false, {}, true);
    if (skip_op(Op::Assign)) {
      auto node = std::make_unique<AssignStmt>(line);
      node->target = std::move(target);
      node->value = parse_tuple(false, true, {}, false, false);
      return node;
    }
    auto node = std::make_unique<AssignBlockStmt>(line);
    node->target = std::move(target);
    node->filter = parse_filter(nullptr, false);
    node->body = parse_statements({"endset"}, true);
    return node;
  }

  void parse_signature(MacroDef& def) {
    expect_op(Op::LParen);
    while (!is_op(Op::RParen)) {
      if (!def.args.empty()) expect_op(Op::Comma);
      const Token& tok = expect(Tok::Name, "name");
      auto arg = std::make_unique<NameExpr>(tok.line, tok.value, Ctx::Param);
      if (skip_op(Op::Assign))
        def.defaults.push_back(parse_expression(true));
      else if (!def.defaults.empty())
        fail("non-default argument follows default argument");
      def.args.push_back(std::move(arg));
    }
    expect_op(Op::RParen);
  }

  StmtPtr parse_macro() {
    const int line = advance().line;
    auto node = std::make_unique<MacroStmt>(line);
    node->def = std::make_unique<MacroDef>();
    node->def->line = line;
    node->def->name = expect(Tok::Name, "name").value;
    parse_signature(*node->def);
    node->def->body = parse_statements({"endmacro"}, true);
    return node;
  }

  StmtPtr parse_call_block() {
    const int line = advance().line;
    auto node = std::make_unique<CallBlockStmt>(line);
    node->def = std::make_unique<MacroDef>();
    node->def->name = "caller";
    node->def->line = line;
    if (is_op(Op::LParen)) parse_signature(*node->def);
    node->call = parse_expression(true);
    if (node->call->k != EK::Call) fail("expected call", line);
    node->def->body = parse_statements({"endcall"}, true);
    return node;
  }

  // transformers' AssistantTracker: {% generation %}body{% endgeneration %} is a
  // call block whose call returns caller() unchanged.
  StmtPtr parse_generation() {
    const int line = advance().line;
    auto node = std::make_unique<CallBlockStmt>(line);
    node->def = std::make_unique<MacroDef>();
    node->def->name = "caller";
    node->def->line = line;
    node->call = std::make_unique<CallExpr>(line, std::make_unique<ConstExpr>(line, Val::builtin(BuiltinId::Generation)));
    node->def->body = parse_statements({"endgeneration"}, true);
    return node;
  }

  StmtPtr parse_filter_block() {
    const int line = advance().line;
    auto node = std::make_unique<FilterBlockStmt>(line);
    node->filter = parse_filter(nullptr, true);
    node->body = parse_statements({"endfilter"}, true);
    return node;
  }

  StmtPtr parse_with() {
    const int line = advance().line;
    auto node = std::make_unique<WithStmt>(line);
    while (cur().type != Tok::BlockEnd) {
      if (!node->targets.empty()) expect_op(Op::Comma);
      ExprPtr target = parse_assign_target(true, false, {}, false);
      set_ctx(target.get(), Ctx::Param);
      node->targets.push_back(std::move(target));
      expect_op(Op::Assign);
      node->values.push_back(parse_expression(true));
    }
    node->body = parse_statements({"endwith"}, true);
    return node;
  }

  StmtPtr parse_print() {
    const int line = advance().line;
    auto node = std::make_unique<OutputStmt>(line);
    while (cur().type != Tok::BlockEnd) {
      if (!node->items.empty()) expect_op(Op::Comma);
      node->items.emplace_back();
      node->items.back().expr = parse_expression(true);
    }
    return node;
  }

  // ---------------------------------------------------------- assignment targets
  static void set_ctx(Expr* e, Ctx ctx) {
    if (e->k == EK::Name) {
      static_cast<NameExpr*>(e)->ctx = ctx;
    } else if (e->k == EK::Tuple) {
      for (auto& item : static_cast<SeqExpr*>(e)->items) set_ctx(item.get(), ctx);
    }
  }

  static bool can_assign(const Expr* e) {
    switch (e->k) {
      case EK::Name: {
        const auto& n = static_cast<const NameExpr*>(e)->name;
        return n != "true" && n != "false" && n != "none" && n != "True" && n != "False" && n != "None";
      }
      case EK::NSRef: return true;
      case EK::Tuple:
        for (const auto& item : static_cast<const SeqExpr*>(e)->items)
          if (!can_assign(item.get())) return false;
        return true;
      default: return false;
    }
  }

  static const char* node_type_name(const Expr* e) {
    switch (e->k) {
      case EK::Const: return "const";
      case EK::List: return "list";
      case EK::Dict: return "dict";
      case EK::Getattr: return "getattr";
      case EK::Getitem: return "getitem";
      case EK::Call: return "call";
      default: return "expr";
    }
  }

  ExprPtr parse_assign_target(bool with_tuple, bool name_only, const std::vector<const char*>& extra_end,
                              bool with_namespace) {
    ExprPtr target;
    if (name_only) {
      const Token& tok = expect(Tok::Name, "name");
      target = std::make_unique<NameExpr>(tok.line, tok.value, Ctx::Store);
    } else {
      if (with_tuple)
        target = parse_tuple(true, true, extra_end, false, with_namespace);
      else
        target = parse_primary(with_namespace);
      set_ctx(target.get(), Ctx::Store);
    }
    if (!can_assign(target.get()))
      fail(std::string("can't assign to '") + node_type_name(target.get()) + "'", target->line);
    return target;
  }

  // ---------------------------------------------------------- expressions
  bool is_tuple_end(const std::vector<const char*>& extra) const {
    const Tok t = cur().type;
    if (t == Tok::VarEnd || t == Tok::BlockEnd || (t == Tok::Op && cur().op == Op::RParen)) return true;
    for (const char* e : extra)
      if (is_name(e)) return true;
    return false;
  }

  ExprPtr parse_tuple(bool simplified, bool with_condexpr, const std::vector<const char*>& extra_end,
                      bool explicit_parens, bool with_namespace) {
    int line = cur().line;
    std::vector<ExprPtr> args;
    bool is_tuple = false;
    for (;;) {
      if (!args.empty()) expect_op(Op::Comma);
      if (is_tuple_end(extra_end)) break;
      args.push_back(simplified ? parse_primary(with_namespace) : parse_expression(with_condexpr));
      if (is_op(Op::Comma))
        is_tuple = true;
      else
        break;
      line = cur().line;
    }
    if (!is_tuple) {
      if (!args.empty()) return std::move(args[0]);
      if (!explicit_parens) fail(std::string("Expected an expression, got '") + describe(cur()) + "'");
    }
    auto t = std::make_unique<SeqExpr>(EK::Tuple, line);
    t->items = std::move(args);
    return t;
  }

  ExprPtr parse_expression(bool with_condexpr) {
    Nest nest(*this);
    return with_condexpr ? parse_condexpr() : parse_or();
  }

  ExprPtr parse_condexpr() {
    int line = cur().line;
    ExprPtr expr1 = parse_or();
    while (skip_name("if")) {
      auto c = std::make_unique<CondExpr>(line);
      c->test = parse_or();
      if (skip_name("else")) c->expr2 = parse_condexpr();
      c->expr1 = std::move(expr1);
      expr1 = std::move(c);
      line = cur().line;
    }
    return expr1;
  }

  ExprPtr parse_or() {
    int line = cur().line;
    ExprPtr left = parse_and();
    while (skip_name("or")) {
      ExprPtr right = parse_and();
      left = std::make_unique<BinExpr>(EK::Or, line, std::move(left), std::move(right));
      line = cur().line;
    }
    return left;
  }

  ExprPtr parse_and() {
    int line = cur().line;
    ExprPtr left = parse_not();
    while (skip_name("and")) {
      ExprPtr right = parse_not();
      left = std::make_unique<BinExpr>(EK::And, line, std::move(left), std::move(right));
      line = cur().line;
    }
    return left;
  }

  ExprPtr parse_not() {
    Nest nest(*this);
    if (is_name("not")) {
      const int line = advance().line;
      return std::make_unique<UnaryExpr>(EK::Not, line, parse_not());
    }
    return parse_compare();
  }

  ExprPtr parse_compare() {
    int line = cur().line;
    ExprPtr expr = parse_math1();
    std::unique_ptr<CompareExpr> cmp;
    for (;;) {
      CmpOp op;
      if (cur().type == Tok::Op && (cur().op == Op::Eq || cur().op == Op::Ne || cur().op == Op::Lt ||
                                     cur().op == Op::Le || cur().op == Op::Gt || cur().op == Op::Ge)) {
        switch (advance().op) {
          case Op::Eq: op = CmpOp::Eq; break;
          case Op::Ne: op = CmpOp::Ne; break;
          case Op::Lt: op = CmpOp::Lt; break;
          case Op::Le: op = CmpOp::Le; break;
          case Op::Gt: op = CmpOp::Gt; break;
          default: op = CmpOp::Ge; break;
        }
      } else if (skip_name("in")) {
        op = CmpOp::In;
      } else if (is_name("not") && look().type == Tok::Name && look().value == "in") {
        advance();
        advance();
        op = CmpOp::NotIn;
      } else {
        break;
      }
      if (!cmp) cmp = std::make_unique<CompareExpr>(line, std::move(expr));
      cmp->ops.emplace_back(op, parse_math1());
      line = cur().line;
    }
    if (!cmp) return expr;
    cmp->line = line;
    return cmp;
  }

  ExprPtr parse_math1() {
    int line = cur().line;
    ExprPtr left = parse_concat();
    while (is_op(Op::Add) || is_op(Op::Sub)) {
      const EK k = advance().op == Op::Add ? EK::Add : EK::Sub;
      ExprPtr right = parse_concat();
      left = std::make_unique<BinExpr>(k, line, std::move(left), std::move(right));
      line = cur().line;
    }
    return left;
  }

  ExprPtr parse_concat() {
    const int line = cur().line;
    ExprPtr first = parse_math2();
    if (!is_op(Op::Tilde)) return first;
    auto c = std::make_unique<ConcatExpr>(line);
    c->nodes.push_back(std::move(first));
    while (skip_op(Op::Tilde)) c->nodes.push_back(parse_math2());
    return c;
  }

  ExprPtr parse_math2() {
    int line = cur().line;
    ExprPtr left = parse_pow();
    while (is_op(Op::Mul) || is_op(Op::Div) || is_op(Op::FloorDiv) || is_op(Op::Mod)) {
      EK k;
      switch (advance().op) {
        case Op::Mul: k = EK::Mul; break;
        case Op::Div: k = EK::Div; break;
        case Op::FloorDiv: k = EK::FloorDiv; break;
        default: k = EK::Mod; break;
      }
      ExprPtr right = parse_pow();
      left = std::make_unique<BinExpr>(k, line, std::move(left), std::move(right));
      line = cur().line;
    }
    return left;
  }

  ExprPtr parse_pow() {
    int line = cur().line;
    ExprPtr left = parse_unary(true);
    while (skip_op(Op::Pow)) {
      ExprPtr right = parse_unary(true);
      left = std::make_unique<BinExpr>(EK::Pow, line, std::move(left), std::move(right));
      line = cur().line;
    }
    return left;
  }

  ExprPtr parse_unary(bool with_filter) {
    Nest nest(*this);
    const int line = cur().line;
    ExprPtr node;
    if (is_op(Op::Sub)) {
      advance();
      node = std::make_unique<UnaryExpr>(EK::Neg, line, parse_unary(false));
    } else if (is_op(Op::Add)) {
      advance();
      node = std::make_unique<UnaryExpr>(EK::Pos, line, parse_unary(false));
    } else {
      node = parse_primary(false);
    }
    node = parse_postfix(std::move(node));
    if (with_filter) node = parse_filter_expr(std::move(node));
    return node;
  }

  ExprPtr parse_primary(bool with_namespace) {
    const Token& tok = cur();
    const int line = tok.line;
    if (tok.type == Tok::Name) {
      const std::string v = tok.value;
      advance();
      if (v == "true" || v == "True") return std::make_unique<ConstExpr>(line, Val::boolean(true));
      if (v == "false" || v == "False") return std::make_unique<ConstExpr>(line, Val::boolean(false));
      if (v == "none" || v == "None") return std::make_unique<ConstExpr>(line, Val::none());
      if (with_namespace && is_op(Op::Dot)) {
        advance();
        const std::string attr = expect(Tok::Name, "name").value;
        return std::make_unique<NSRefExpr>(line, v, attr);
      }
      return std::make_unique<NameExpr>(line, v, Ctx::Load);
    }
    if (tok.type == Tok::String) {
      std::string s = advance().value;
      while (cur().type == Tok::String) s += advance().value;  // adjacent literals concatenate
      const std::string& lit = *literals_->insert(std::move(s)).first;
      return std::make_unique<ConstExpr>(line, Val::str_ref(lit));
    }
    if (tok.type == Tok::Int) return std::make_unique<ConstExpr>(line, Val::integer(advance().ival));
    if (tok.type == Tok::Float) return std::make_unique<ConstExpr>(line, Val::real(advance().fval));
    if (is_op(Op::LParen)) {
      advance();
      ExprPtr node = parse_tuple(false, true, {}, true, false);
      expect_op(Op::RParen);
      return node;
    }
    if (is_op(Op::LBracket)) return parse_list();
    if (is_op(Op::LBrace)) return parse_dict();
    fail(std::string("unexpected '") + describe(tok) + "'", line);
  }

  ExprPtr parse_list() {
    const int line = cur().line;
    expect_op(Op::LBracket);
    auto node = std::make_unique<SeqExpr>(EK::List, line);
    while (!is_op(Op::RBracket)) {
      if (!node->items.empty()) expect_op(Op::Comma);
      if (is_op(Op::RBracket)) break;
      node->items.push_back(parse_expression(true));
    }
    expect_op(Op::RBracket);
    return node;
  }

  ExprPtr parse_dict() {
    const int line = cur().line;
    expect_op(Op::LBrace);
    auto node = std::make_unique<DictExpr>(line);
    while (!is_op(Op::RBrace)) {
      if (!node->items.empty()) expect_op(Op::Comma);
      if (is_op(Op::RBrace)) break;
      ExprPtr key = parse_expression(true);
      expect_op(Op::Colon);
      ExprPtr value = parse_expression(true);
      node->items.emplace_back(std::move(key), std::move(value));
    }
    expect_op(Op::RBrace);
    return node;
  }

  ExprPtr parse_postfix(ExprPtr node) {
    for (;;) {
      if (is_op(Op::Dot) || is_op(Op::LBracket))
        node = parse_subscript(std::move(node));
      else if (is_op(Op::LParen))
        node = parse_call(std::move(node));
      else
        break;
    }
    return node;
  }

  ExprPtr parse_filter_expr(ExprPtr node) {
    for (;;) {
      if (is_op(Op::Pipe))
        node = parse_filter(std::move(node), false);
      else if (is_name("is"))
        node = parse_test(std::move(node));
      else if (is_op(Op::LParen))
        node = parse_call(std::move(node));
      else
        break;
    }
    return node;
  }

  ExprPtr parse_subscript(ExprPtr node) {
    const Token tok = advance();
    if (tok.type == Tok::Op && tok.op == Op::Dot) {
      const Token attr = advance();
      if (attr.type == Tok::Name) return std::make_unique<GetattrExpr>(tok.line, std::move(node), attr.value);
      if (attr.type != Tok::Int) fail("expected name or number", attr.line);
      return std::make_unique<GetitemExpr>(tok.line, std::move(node),
                                           std::make_unique<ConstExpr>(attr.line, Val::integer(attr.ival)));
    }
    if (tok.type == Tok::Op && tok.op == Op::LBracket) {
      std::vector<ExprPtr> args;
      while (!is_op(Op::RBracket)) {
        if (!args.empty()) expect_op(Op::Comma);
        args.push_back(parse_subscribed());
      }
      expect_op(Op::RBracket);
      ExprPtr arg;
      if (args.size() == 1) {
        arg = std::move(args[0]);
      } else {
        auto t = std::make_unique<SeqExpr>(EK::Tuple, tok.line);
        t->items = std::move(args);
        arg = std::move(t);
      }
      return std::make_unique<GetitemExpr>(tok.line, std::move(node), std::move(arg));
    }
    fail("expected subscript expression", tok.line);
  }

  ExprPtr parse_subscribed() {
    const int line = cur().line;
    auto slice = std::make_unique<SliceExpr>(line);
    if (is_op(Op::Colon)) {
      advance();
    } else {
      ExprPtr node = parse_expression(true);
      if (!is_op(Op::Colon)) return node;
      advance();
      slice->start = std::move(node);
    }
    if (is_op(Op::Colon)) {
      // stop omitted
    } else if (!is_op(Op::RBracket) && !is_op(Op::Comma)) {
      slice->stop = parse_expression(true);
    }
    if (is_op(Op::Colon)) {
      advance();
      if (!is_op(Op::RBracket) && !is_op(Op::Comma)) slice->step = parse_expression(true);
    }
    return slice;
  }

  void parse_call_args(CallArgs& a) {
    const int line = cur().line;
    expect_op(Op::LParen);
    bool require_comma = false;
    auto ensure = [&](bool ok) {
      if (!ok) fail("invalid syntax for function call expression", line);
    };
    while (!is_op(Op::RParen)) {
      if (require_comma) {
        expect_op(Op::Comma);
        if (is_op(Op::RParen)) break;  // trailing comma
      }
      if (is_op(Op::Mul)) {
        ensure(!a.dyn_args && !a.dyn_kwargs);
        advance();
        a.dyn_args = parse_expression(true);
      } else if (is_op(Op::Pow)) {
        ensure(!a.dyn_kwargs);
        advance();
        a.dyn_kwargs = parse_expression(true);
      } else if (cur().type == Tok::Name && look().type == Tok::Op && look().op == Op::Assign) {
        ensure(!a.dyn_kwargs);
        std::string key = advance().value;
        advance();
        a.kwargs.emplace_back(std::move(key), parse_expression(true));
      } else {
        ensure(!a.dyn_args && !a.dyn_kwargs && a.kwargs.empty());
        a.args.push_back(parse_expression(true));
      }
      require_comma = true;
    }
    expect_op(Op::RParen);
  }

  ExprPtr parse_call(ExprPtr node) {
    auto call = std::make_unique<CallExpr>(cur().line, std::move(node));
    parse_call_args(call->a);
    return call;
  }

  ExprPtr parse_filter(ExprPtr node, bool start_inline) {
    while (is_op(Op::Pipe) || start_inline) {
      if (!start_inline) advance();
      const Token& tok = expect(Tok::Name, "name");
      const int line = tok.line;
      std::string name = tok.value;
      while (is_op(Op::Dot)) {
        advance();
        name += "." + expect(Tok::Name, "name").value;
      }
      auto f = std::make_unique<FilterExpr>(EK::Filter, line, std::move(node), std::move(name));
      if (is_op(Op::LParen)) parse_call_args(f->a);
      node = std::move(f);
      start_inline = false;
    }
    return node;
  }

  ExprPtr parse_test(ExprPtr node) {
    const int line = advance().line;  // 'is'
    const bool negated = skip_name("not");
    std::string name = expect(Tok::Name, "name").value;
    while (is_op(Op::Dot)) {
      advance();
      name += "." + expect(Tok::Name, "name").value;
    }
    auto t = std::make_unique<FilterExpr>(EK::Test, line, std::move(node), std::move(name));
    if (is_op(Op::LParen)) {
      parse_call_args(t->a);
    } else if ((cur().type == Tok::Name || cur().type == Tok::String || cur().type == Tok::Int ||
                cur().type == Tok::Float || is_op(Op::LParen) || is_op(Op::LBracket) || is_op(Op::LBrace)) &&
               !is_name("else") && !is_name("or") && !is_name("and")) {
      if (is_name("is")) fail("You cannot chain multiple tests with is");
      ExprPtr arg = parse_postfix(parse_primary(false));
      t->a.args.push_back(std::move(arg));
    }
    if (negated) return std::make_unique<UnaryExpr>(EK::Not, line, std::move(t));
    return t;
  }

  struct Nest {  // RAII nesting counter for expressions and statements
    Parser& p;
    explicit Nest(Parser& parser) : p(parser) {
      if (++p.depth_ > kMaxNesting) p.fail("template nested too deeply");
    }
    ~Nest() { --p.depth_; }
  };

  std::vector<Token> t_;
  std::unordered_set<std::string>* literals_;
  int depth_ = 0;
  size_t p_ = 0;
  std::vector<std::vector<const char*>> end_stack_;
  std::vector<std::string> tag_stack_;
};

}  // namespace

void parse_template(std::string_view source, TemplateImpl& out) {
  Parser p(Lexer(source).run(), &out.literals);
  out.body = p.parse();
}

}  // namespace espetero::tmpl
