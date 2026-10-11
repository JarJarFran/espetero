// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "json.h"

#include <xlocale.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <numeric>

namespace espetero::json {

// ---------------------------------------------------------------- Object
Object::Object(const Object& o) : keys(o.keys), vals(o.vals) {}

Object& Object::operator=(const Object& o) {
  if (this != &o) {
    keys = o.keys;
    vals = o.vals;
    index.reset();
  }
  return *this;
}

namespace {
constexpr size_t kIndexAt = 12;  // linear search below this many keys
}

const Value* Object::find(std::string_view key) const {
  if (keys.size() < kIndexAt) {
    for (size_t i = 0; i < keys.size(); ++i)
      if (keys[i] == key) return &vals[i];
    return nullptr;
  }
  if (!index || index->size() != keys.size()) {
    index = std::make_unique<std::unordered_map<std::string_view, uint32_t>>();
    index->reserve(keys.size() * 2);
    for (size_t i = 0; i < keys.size(); ++i) (*index)[keys[i]] = uint32_t(i);  // last duplicate wins
  }
  auto it = index->find(key);
  return it == index->end() ? nullptr : &vals[it->second];
}

Value* Object::find(std::string_view key) { return const_cast<Value*>(static_cast<const Object*>(this)->find(key)); }

Value& Object::set(std::string key, Value v) {
  if (Value* cur = find(key)) {
    *cur = std::move(v);
    return *cur;
  }
  keys.push_back(std::move(key));
  vals.push_back(std::move(v));
  if (index) index.reset();  // key strings may have moved
  return vals.back();
}

bool Object::erase(std::string_view key) {
  for (size_t i = 0; i < keys.size(); ++i)
    if (keys[i] == key) {
      keys.erase(keys.begin() + long(i));
      vals.erase(vals.begin() + long(i));
      index.reset();
      return true;
    }
  return false;
}

// ---------------------------------------------------------------- Value
void Value::release() noexcept {
  switch (t_) {
    case Type::String: delete u_.s; break;
    case Type::Array: delete u_.a; break;
    case Type::Object: delete u_.o; break;
    default: break;
  }
  t_ = Type::Null;
  u_.i = 0;
}

void Value::copy_from(const Value& v) {
  switch (v.t_) {
    case Type::String: u_.s = new std::string(*v.u_.s); break;
    case Type::Array: u_.a = new std::vector<Value>(*v.u_.a); break;
    case Type::Object: u_.o = new Object(*v.u_.o); break;
    default: u_ = v.u_; break;
  }
  t_ = v.t_;
}

Value& Value::operator[](std::string_view key) {
  if (t_ == Type::Null) *this = object();
  Object& o = as_object();
  if (Value* v = o.find(key)) return *v;
  return o.set(std::string(key), Value());
}

bool Value::operator==(const Value& o) const {
  if (is_number() && o.is_number()) {
    if (t_ == Type::Int && o.t_ == Type::Int) return u_.i == o.u_.i;
    return as_double() == o.as_double();
  }
  if (t_ != o.t_) return false;
  switch (t_) {
    case Type::Null: return true;
    case Type::Bool: return u_.b == o.u_.b;
    case Type::String: return *u_.s == *o.u_.s;
    case Type::Array: return *u_.a == *o.u_.a;
    case Type::Object: {
      if (u_.o->size() != o.u_.o->size()) return false;
      for (size_t i = 0; i < u_.o->size(); ++i) {
        const Value* w = o.u_.o->find(u_.o->keys[i]);
        if (!w || !(*w == u_.o->vals[i])) return false;
      }
      return true;
    }
    default: return false;
  }
}

// ---------------------------------------------------------------- UTF-8
uint32_t utf8_next(std::string_view s, size_t& i) {
  const auto* p = reinterpret_cast<const unsigned char*>(s.data());
  const size_t n = s.size();
  const unsigned c = p[i];
  if (c < 0x80) {
    ++i;
    return c;
  }
  int len;
  uint32_t cp, min;
  if ((c & 0xE0) == 0xC0) len = 2, cp = c & 0x1F, min = 0x80;
  else if ((c & 0xF0) == 0xE0) len = 3, cp = c & 0x0F, min = 0x800;
  else if ((c & 0xF8) == 0xF0) len = 4, cp = c & 0x07, min = 0x10000;
  else {
    ++i;
    return 0xFFFD;
  }
  if (i + size_t(len) > n) {
    ++i;
    return 0xFFFD;
  }
  for (int k = 1; k < len; ++k) {
    const unsigned d = p[i + size_t(k)];
    if ((d & 0xC0) != 0x80) {
      ++i;
      return 0xFFFD;
    }
    cp = (cp << 6) | (d & 0x3F);
  }
  if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
    ++i;
    return 0xFFFD;
  }
  i += size_t(len);
  return cp;
}

void utf8_append(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(char(cp));
  } else if (cp < 0x800) {
    out.push_back(char(0xC0 | (cp >> 6)));
    out.push_back(char(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(char(0xE0 | (cp >> 12)));
    out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(char(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(char(0xF0 | (cp >> 18)));
    out.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(char(0x80 | (cp & 0x3F)));
  }
}

bool utf8_valid(std::string_view s) {
  size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
      ++i;
      continue;
    }
    const size_t before = i;
    const uint32_t cp = utf8_next(s, i);
    if (cp == 0xFFFD) {
      // A genuine U+FFFD is EF BF BD; anything else was invalid.
      if (!(i - before == 3 && s.compare(before, 3, "\xEF\xBF\xBD") == 0)) return false;
    }
  }
  return true;
}

size_t utf8_complete_prefix(std::string_view s) {
  const size_t n = s.size();
  // Walk back over at most 3 continuation bytes to the last lead byte.
  size_t k = 0;
  while (k < 4 && k < n && (static_cast<unsigned char>(s[n - 1 - k]) & 0xC0) == 0x80) ++k;
  if (k == n) return n;  // only continuation bytes: nothing to hold back sensibly
  const unsigned char lead = static_cast<unsigned char>(s[n - 1 - k]);
  int need = 1;
  if ((lead & 0xE0) == 0xC0) need = 2;
  else if ((lead & 0xF0) == 0xE0) need = 3;
  else if ((lead & 0xF8) == 0xF0) need = 4;
  else if (lead >= 0x80) return n;  // invalid lead: emit as is
  if (int(k) + 1 < need) return n - 1 - k;
  return n;
}

// ---------------------------------------------------------------- parse
namespace {

struct Parser {
  const char* p;
  const char* begin;
  const char* end;
  int max_depth;

  [[noreturn]] void error(const char* what) const {
    throw Error(std::string("invalid JSON at byte ") + std::to_string(p - begin) + ": " + what);
  }

  void ws() {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
  }

  bool lit(const char* s, size_t n) {
    if (size_t(end - p) >= n && std::memcmp(p, s, n) == 0) {
      p += n;
      return true;
    }
    return false;
  }

  static int hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  }

  uint32_t hex4() {
    if (end - p < 4) error("truncated \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      const int h = hex(p[k]);
      if (h < 0) error("bad \\u escape");
      v = (v << 4) | uint32_t(h);
    }
    p += 4;
    return v;
  }

  std::string string() {
    // p is past the opening quote.
    std::string out;
    const char* run = p;
    while (true) {
      if (p >= end) error("unterminated string");
      const unsigned char c = static_cast<unsigned char>(*p);
      if (c == '"') {
        out.append(run, size_t(p - run));
        ++p;
        return out;
      }
      if (c == '\\') {
        out.append(run, size_t(p - run));
        ++p;
        if (p >= end) error("unterminated escape");
        const char e = *p++;
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            uint32_t cp = hex4();
            if (cp >= 0xD800 && cp <= 0xDBFF) {
              if (end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                const char* save = p;
                p += 2;
                const uint32_t lo = hex4();
                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                  cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                } else {
                  p = save;
                  cp = 0xFFFD;
                }
              } else {
                cp = 0xFFFD;
              }
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
              cp = 0xFFFD;
            }
            utf8_append(out, cp);
            break;
          }
          default: error("bad escape");
        }
        run = p;
        continue;
      }
      if (c < 0x20) error("control character in string");
      if (c < 0x80) {
        ++p;
        continue;
      }
      // Validate a multi-byte sequence; replace invalid bytes.
      size_t i = size_t(p - begin);
      const std::string_view all(begin, size_t(end - begin));
      const size_t before = i;
      const uint32_t cp = utf8_next(all, i);
      if (cp == 0xFFFD && !(i - before == 3 && all.compare(before, 3, "\xEF\xBF\xBD") == 0)) {
        out.append(run, size_t(p - run));
        utf8_append(out, 0xFFFD);
        p = begin + i;
        run = p;
      } else {
        p = begin + i;
      }
    }
  }

  Value number() {
    const char* s = p;
    bool is_float = false;
    if (p < end && *p == '-') ++p;
    if (p >= end) error("bad number");
    if (*p == '0') {
      ++p;
    } else if (*p >= '1' && *p <= '9') {
      while (p < end && *p >= '0' && *p <= '9') ++p;
    } else {
      error("bad number");
    }
    if (p < end && *p == '.') {
      is_float = true;
      ++p;
      if (p >= end || !(*p >= '0' && *p <= '9')) error("bad fraction");
      while (p < end && *p >= '0' && *p <= '9') ++p;
    }
    if (p < end && (*p == 'e' || *p == 'E')) {
      is_float = true;
      ++p;
      if (p < end && (*p == '+' || *p == '-')) ++p;
      if (p >= end || !(*p >= '0' && *p <= '9')) error("bad exponent");
      while (p < end && *p >= '0' && *p <= '9') ++p;
    }
    if (!is_float) {
      int64_t v = 0;
      auto r = std::from_chars(s, p, v);
      if (r.ec == std::errc() && r.ptr == p) return Value(v);
      // Out of int64 range: fall through to double (Python keeps big ints; rare).
    }
    // strtod_l in the C locale (std::from_chars for double needs macOS 26). The
    // grammar above already limited [s, p) to a JSON number; overflow gives
    // +-inf and underflow 0, as Python's float() does.
    static const locale_t c_locale = newlocale(LC_ALL_MASK, "C", nullptr);
    const std::string text(s, p);
    char* stop = nullptr;
    const double d = strtod_l(text.c_str(), &stop, c_locale);
    if (stop != text.c_str() + text.size()) error("bad number");
    return Value(d);
  }

  Value value(int depth) {
    if (depth > max_depth) error("nesting too deep");
    ws();
    if (p >= end) error("unexpected end");
    switch (*p) {
      case '{': {
        ++p;
        Object o;
        // Duplicate-key detection for large objects (a tokenizer vocabulary has
        // ~250K keys): a hash map of positions, built once the object grows.
        std::unique_ptr<std::unordered_map<std::string, uint32_t>> seen;
        ws();
        if (p < end && *p == '}') {
          ++p;
          return Value(std::move(o));
        }
        while (true) {
          ws();
          if (p >= end || *p != '"') error("expected a key");
          ++p;
          std::string k = string();
          ws();
          if (p >= end || *p != ':') error("expected ':'");
          ++p;
          Value v = value(depth + 1);
          // Duplicate keys: the last one wins, at the first key's position (Python).
          long dup = -1;
          if (!seen) {
            for (size_t i = 0; i < o.keys.size(); ++i)
              if (o.keys[i] == k) {
                dup = long(i);
                break;
              }
            if (o.keys.size() >= 32) {
              seen = std::make_unique<std::unordered_map<std::string, uint32_t>>();
              for (size_t i = 0; i < o.keys.size(); ++i) seen->emplace(o.keys[i], uint32_t(i));
            }
          } else {
            auto it = seen->find(k);
            if (it != seen->end()) dup = long(it->second);
          }
          if (dup >= 0) {
            o.vals[size_t(dup)] = std::move(v);
          } else {
            if (seen) seen->emplace(k, uint32_t(o.keys.size()));
            o.keys.push_back(std::move(k));
            o.vals.push_back(std::move(v));
          }
          ws();
          if (p < end && *p == ',') {
            ++p;
            continue;
          }
          if (p < end && *p == '}') {
            ++p;
            o.index.reset();
            return Value(std::move(o));
          }
          error("expected ',' or '}'");
        }
      }
      case '[': {
        ++p;
        std::vector<Value> a;
        ws();
        if (p < end && *p == ']') {
          ++p;
          return Value(std::move(a));
        }
        while (true) {
          a.push_back(value(depth + 1));
          ws();
          if (p < end && *p == ',') {
            ++p;
            continue;
          }
          if (p < end && *p == ']') {
            ++p;
            return Value(std::move(a));
          }
          error("expected ',' or ']'");
        }
      }
      case '"':
        ++p;
        return Value(string());
      case 't':
        if (lit("true", 4)) return Value(true);
        error("bad literal");
      case 'f':
        if (lit("false", 5)) return Value(false);
        error("bad literal");
      case 'n':
        if (lit("null", 4)) return Value();
        error("bad literal");
      case 'N':
        if (lit("NaN", 3)) return Value(std::nan(""));  // Python json accepts these
        error("bad literal");
      case 'I':
        if (lit("Infinity", 8)) return Value(HUGE_VAL);
        error("bad literal");
      default:
        if (*p == '-' && size_t(end - p) >= 9 && std::memcmp(p, "-Infinity", 9) == 0) {
          p += 9;
          return Value(-HUGE_VAL);
        }
        return number();
    }
  }
};

}  // namespace

Value parse(std::string_view text, int max_depth) {
  Parser ps{text.data(), text.data(), text.data() + text.size(), max_depth};
  // Skip a UTF-8 byte order mark.
  if (text.size() >= 3 && std::memcmp(text.data(), "\xEF\xBB\xBF", 3) == 0) ps.p += 3;
  Value v = ps.value(0);
  ps.ws();
  if (ps.p != ps.end) ps.error("trailing characters");
  return v;
}

// ---------------------------------------------------------------- dump
std::string format_double(double d) {
  if (std::isnan(d)) return "NaN";
  if (std::isinf(d)) return d > 0 ? "Infinity" : "-Infinity";
  if (d == 0) return std::signbit(d) ? "-0.0" : "0.0";
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::scientific);
  std::string_view sci(buf, size_t(r.ptr - buf));
  // sci: [-]d[.ddd]e[+-]xx
  bool neg = false;
  if (!sci.empty() && sci[0] == '-') {
    neg = true;
    sci.remove_prefix(1);
  }
  const size_t epos = sci.find('e');
  std::string digits;
  for (size_t i = 0; i < epos; ++i)
    if (sci[i] != '.') digits.push_back(sci[i]);
  const int exp10 = std::stoi(std::string(sci.substr(epos + 1)));
  // Python float_repr_style 'short' with repr: fixed when -4 <= exp < 16.
  std::string out = neg ? "-" : "";
  const int n = int(digits.size());
  if (exp10 >= -4 && exp10 < 16) {
    const int decpt = exp10 + 1;
    if (decpt <= 0) {
      out += "0.";
      out.append(size_t(-decpt), '0');
      out += digits;
    } else if (decpt >= n) {
      out += digits;
      out.append(size_t(decpt - n), '0');
      out += ".0";
    } else {
      out += digits.substr(0, size_t(decpt));
      out += '.';
      out += digits.substr(size_t(decpt));
    }
  } else {
    out += digits[0];
    if (n > 1) {
      out += '.';
      out += digits.substr(1);
    }
    out += 'e';
    out += exp10 < 0 ? '-' : '+';
    const int ae = exp10 < 0 ? -exp10 : exp10;
    if (ae < 10) out += '0';
    out += std::to_string(ae);
  }
  return out;
}

void append_quoted(std::string& out, std::string_view s, bool ensure_ascii) {
  static const char* hexd = "0123456789abcdef";
  out.push_back('"');
  size_t i = 0;
  size_t run = 0;
  auto flush = [&](size_t upto) {
    if (upto > run) out.append(s.data() + run, upto - run);
  };
  auto u4 = [&](uint32_t v) {
    out += "\\u";
    out.push_back(hexd[(v >> 12) & 15]);
    out.push_back(hexd[(v >> 8) & 15]);
    out.push_back(hexd[(v >> 4) & 15]);
    out.push_back(hexd[v & 15]);
  };
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    const char* esc = nullptr;
    switch (c) {
      case '"': esc = "\\\""; break;
      case '\\': esc = "\\\\"; break;
      case '\n': esc = "\\n"; break;
      case '\r': esc = "\\r"; break;
      case '\t': esc = "\\t"; break;
      case '\b': esc = "\\b"; break;
      case '\f': esc = "\\f"; break;
      default: break;
    }
    if (esc) {
      flush(i);
      out += esc;
      ++i;
      run = i;
      continue;
    }
    if (c < 0x20 || (ensure_ascii && c == 0x7F)) {
      flush(i);
      u4(c);
      ++i;
      run = i;
      continue;
    }
    if (c < 0x80 || !ensure_ascii) {
      ++i;
      continue;
    }
    flush(i);
    uint32_t cp = utf8_next(s, i);
    if (cp >= 0x10000) {
      cp -= 0x10000;
      u4(0xD800 + (cp >> 10));
      u4(0xDC00 + (cp & 0x3FF));
    } else {
      u4(cp);
    }
    run = i;
  }
  flush(s.size());
  out.push_back('"');
}

namespace {

void dump_rec(std::string& out, const Value& v, const DumpOptions& o, int level) {
  auto newline = [&](int lvl) {
    out.push_back('\n');
    out.append(size_t(o.indent) * size_t(lvl), ' ');
  };
  // Python uses ',' (no trailing space) between items when indenting with the
  // default separators.
  const std::string_view item_sep = (o.indent >= 0 && o.item_sep == ", ") ? std::string_view(",") : o.item_sep;
  switch (v.type()) {
    case Type::Null: out += "null"; break;
    case Type::Bool: out += v.as_bool() ? "true" : "false"; break;
    case Type::Int: out += std::to_string(v.as_int()); break;
    case Type::Double: out += format_double(v.as_double()); break;
    case Type::String: append_quoted(out, v.as_string(), o.ensure_ascii); break;
    case Type::Array: {
      const auto& a = v.as_array();
      if (a.empty()) {
        out += "[]";
        break;
      }
      out.push_back('[');
      for (size_t i = 0; i < a.size(); ++i) {
        if (i) out += item_sep;
        if (o.indent >= 0) newline(level + 1);
        dump_rec(out, a[i], o, level + 1);
      }
      if (o.indent >= 0) newline(level);
      out.push_back(']');
      break;
    }
    case Type::Object: {
      const auto& ob = v.as_object();
      if (ob.size() == 0) {
        out += "{}";
        break;
      }
      std::vector<size_t> order(ob.size());
      std::iota(order.begin(), order.end(), size_t(0));
      if (o.sort_keys)
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return ob.keys[a] < ob.keys[b]; });
      out.push_back('{');
      for (size_t k = 0; k < order.size(); ++k) {
        const size_t i = order[k];
        if (k) out += item_sep;
        if (o.indent >= 0) newline(level + 1);
        append_quoted(out, ob.keys[i], o.ensure_ascii);
        out += o.key_sep;
        dump_rec(out, ob.vals[i], o, level + 1);
      }
      if (o.indent >= 0) newline(level);
      out.push_back('}');
      break;
    }
  }
}

}  // namespace

void dump_to(std::string& out, const Value& v, const DumpOptions& opt) { dump_rec(out, v, opt, 0); }

std::string dump(const Value& v, const DumpOptions& opt) {
  std::string out;
  dump_rec(out, v, opt, 0);
  return out;
}

}  // namespace espetero::json
