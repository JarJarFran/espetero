// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Python operators on template values: + - * / // % ** unary -/+, `in`,
// slicing, printf-style `%` formatting and str.format. Error messages are
// CPython's, because templates surface them to users verbatim.
#include <cmath>
#include <cstdio>

#include "interp.h"
#include "unicode.h"

namespace espetero::tmpl {

namespace {

[[noreturn]] void type_error(const std::string& m) { raise(ErrorKind::Type, m); }
[[noreturn]] void int_overflow() {
  raise(ErrorKind::Overflow, "integer result out of range (arbitrary precision integers are not supported)");
}
[[noreturn]] void unsupported_operands(const char* op, const Val& a, const Val& b) {
  type_error(std::string("unsupported operand type(s) for ") + op + ": '" + tname(a) + "' and '" + tname(b) + "'");
}

void check_defined(const Val& a, const Val& b) {
  if (a.is_undef() || a.is_missing()) raise_undefined(a);
  if (b.is_undef() || b.is_missing()) raise_undefined(b);
}

bool is_sequence_repeatable(const Val& v) { return v.is_str() || v.is_list() || v.is_tuple(); }

Val repeat(const Val& seq, int64_t n) {
  if (seq.is_str()) {
    std::string out;
    if (n > 0) {
      out.reserve(seq.sv().size() * size_t(n));
      for (int64_t k = 0; k < n; ++k) out += seq.sv();
    }
    return Val::str(std::move(out), seq.markup());
  }
  std::vector<Val> items;
  const size_t len = seq_size(seq);
  for (int64_t k = 0; k < n; ++k)
    for (size_t i = 0; i < len; ++i) items.push_back(seq_at(seq, i));
  return seq.is_list() ? make_list(std::move(items)) : make_tuple(std::move(items));
}

double py_fmod(double a, double b) {
  double mod = std::fmod(a, b);
  if (mod != 0) {
    if ((b < 0) != (mod < 0)) mod += b;
  } else {
    mod = std::copysign(0.0, b);
  }
  return mod;
}

double py_floordiv(double a, double b) {
  double mod = std::fmod(a, b);
  double div = (a - mod) / b;
  if (mod != 0) {
    if ((b < 0) != (mod < 0)) div -= 1.0;
  }
  if (div != 0) {
    double fl = std::floor(div);
    if (div - fl > 0.5) fl += 1.0;
    return fl;
  }
  return std::copysign(0.0, a / b);
}

}  // namespace

std::string html_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&#34;"; break;
      case '\'': out += "&#39;"; break;
      default: out.push_back(c);
    }
  }
  return out;
}

Val py_add(const Val& a, const Val& b) {
  check_defined(a, b);
  if (a.is_number() && b.is_number()) {
    if (!a.is_float() && !b.is_float()) {
      int64_t r;
      if (__builtin_add_overflow(a.i(), b.i(), &r)) int_overflow();
      return Val::integer(r);
    }
    return Val::real(a.f() + b.f());
  }
  if (a.is_str() && b.is_str()) {
    if (a.markup() || b.markup()) {  // Markup escapes the plain operand
      std::string s = a.markup() ? std::string(a.sv()) : html_escape(a.sv());
      s += b.markup() ? std::string(b.sv()) : html_escape(b.sv());
      return Val::str(std::move(s), true);
    }
    std::string s;
    s.reserve(a.sv().size() + b.sv().size());
    s += a.sv();
    s += b.sv();
    return Val::str(std::move(s));
  }
  if ((a.is_list() && b.is_list()) || (a.is_tuple() && b.is_tuple())) {
    std::vector<Val> items;
    const size_t na = seq_size(a), nb = seq_size(b);
    items.reserve(na + nb);
    for (size_t i = 0; i < na; ++i) items.push_back(seq_at(a, i));
    for (size_t i = 0; i < nb; ++i) items.push_back(seq_at(b, i));
    return a.is_list() ? make_list(std::move(items)) : make_tuple(std::move(items));
  }
  if (a.is_str()) type_error(std::string("can only concatenate str (not \"") + tname(b) + "\") to str");
  if (a.is_list()) type_error(std::string("can only concatenate list (not \"") + tname(b) + "\") to list");
  if (a.is_tuple()) type_error(std::string("can only concatenate tuple (not \"") + tname(b) + "\") to tuple");
  unsupported_operands("+", a, b);
}

Val percent_format(const Val& fmt, const Val& args);

Val py_arith(EK op, const Val& a, const Val& b) {
  check_defined(a, b);
  const bool nums = a.is_number() && b.is_number();
  const bool ints = nums && !a.is_float() && !b.is_float();
  switch (op) {
    case EK::Sub:
      if (ints) {
        int64_t r;
        if (__builtin_sub_overflow(a.i(), b.i(), &r)) int_overflow();
        return Val::integer(r);
      }
      if (nums) return Val::real(a.f() - b.f());
      unsupported_operands("-", a, b);
    case EK::Mul:
      if (ints) {
        int64_t r;
        if (__builtin_mul_overflow(a.i(), b.i(), &r)) int_overflow();
        return Val::integer(r);
      }
      if (nums) return Val::real(a.f() * b.f());
      if (is_sequence_repeatable(a) && b.is_intlike()) return repeat(a, b.i());
      if (a.is_intlike() && is_sequence_repeatable(b)) return repeat(b, a.i());
      if (is_sequence_repeatable(a)) type_error(std::string("can't multiply sequence by non-int of type '") + tname(b) + "'");
      if (is_sequence_repeatable(b)) type_error(std::string("can't multiply sequence by non-int of type '") + tname(a) + "'");
      unsupported_operands("*", a, b);
    case EK::Div:
      if (!nums) unsupported_operands("/", a, b);
      if (b.f() == 0) raise(ErrorKind::ZeroDivision, ints ? "division by zero" : "float division by zero");
      return Val::real(a.f() / b.f());
    case EK::FloorDiv:
      if (!nums) unsupported_operands("//", a, b);
      if (ints) {
        const int64_t x = a.i(), y = b.i();
        if (y == 0) raise(ErrorKind::ZeroDivision, "integer division or modulo by zero");
        if (x == INT64_MIN && y == -1) int_overflow();
        int64_t q = x / y;
        if ((x % y != 0) && ((x < 0) != (y < 0))) --q;
        return Val::integer(q);
      }
      if (b.f() == 0) raise(ErrorKind::ZeroDivision, "float floor division by zero");
      return Val::real(py_floordiv(a.f(), b.f()));
    case EK::Mod:
      if (a.is_str()) return percent_format(a, b);
      if (!nums) unsupported_operands("%", a, b);
      if (ints) {
        const int64_t x = a.i(), y = b.i();
        if (y == 0) raise(ErrorKind::ZeroDivision, "integer modulo by zero");
        if (y == -1) return Val::integer(0);
        int64_t r = x % y;
        if (r != 0 && ((r < 0) != (y < 0))) r += y;
        return Val::integer(r);
      }
      if (b.f() == 0) raise(ErrorKind::ZeroDivision, "float modulo by zero");
      return Val::real(py_fmod(a.f(), b.f()));
    case EK::Pow: {
      if (!nums) unsupported_operands("** or pow()", a, b);
      if (ints && b.i() >= 0) {
        int64_t base = a.i(), e = b.i(), r = 1;
        while (e > 0) {
          if (e & 1)
            if (__builtin_mul_overflow(r, base, &r)) int_overflow();
          e >>= 1;
          if (e > 0 && __builtin_mul_overflow(base, base, &base)) int_overflow();
        }
        return Val::integer(r);
      }
      const double x = a.f(), y = b.f();
      if (x == 0 && y < 0) raise(ErrorKind::ZeroDivision, "zero to a negative power");
      if (x < 0 && y != std::floor(y))
        raise(ErrorKind::Unsupported, "complex results of ** are not supported");
      return Val::real(std::pow(x, y));
    }
    default: raise(ErrorKind::Runtime, "bad operator");
  }
}

Val py_neg(const Val& a) {
  if (a.is_undef() || a.is_missing()) raise_undefined(a);
  if (a.is_intlike()) {
    if (a.i() == INT64_MIN) int_overflow();
    return Val::integer(-a.i());
  }
  if (a.is_float()) return Val::real(-a.f());
  type_error(std::string("bad operand type for unary -: '") + tname(a) + "'");
}

Val py_pos(const Val& a) {
  if (a.is_undef() || a.is_missing()) raise_undefined(a);
  if (a.is_intlike()) return Val::integer(a.i());
  if (a.is_float()) return a;
  type_error(std::string("bad operand type for unary +: '") + tname(a) + "'");
}

bool py_contains(const Val& c, const Val& item) {
  switch (c.kind()) {
    case Kind::Str:
      if (!item.is_str()) type_error(std::string("'in <string>' requires string as left operand, not ") + tname(item));
      return c.sv().find(item.sv()) != std::string_view::npos;
    case Kind::List:
    case Kind::Tuple: {
      const size_t n = seq_size(c);
      for (size_t i = 0; i < n; ++i)
        if (py_eq(seq_at(c, i), item)) return true;
      return false;
    }
    case Kind::Dict: {
      require_hashable(item);
      Val out;
      return dict_get(c, item, out);
    }
    case Kind::Range: {
      const auto* r = c.as<RangeObj>();
      if (item.is_intlike()) {
        const int64_t x = item.i();
        if (r->step > 0 ? (x < r->start || x >= r->stop) : (x > r->start || x <= r->stop)) return false;
        return (x - r->start) % r->step == 0;
      }
      for (int64_t i = 0, n = r->size(); i < n; ++i)
        if (py_eq(Val::integer(r->at(i)), item)) return true;
      return false;
    }
    case Kind::View: {
      const auto* v = c.as<ViewObj>();
      Val out;
      if (v->which == ViewKind::Keys) {
        require_hashable(item);
        return dict_get(v->dict, item, out);
      }
      if (v->which == ViewKind::Items) {
        if (!item.is_tuple() || seq_size(item) != 2) return false;
        const Val k = seq_at(item, 0);
        require_hashable(k);
        return dict_get(v->dict, k, out) && py_eq(out, seq_at(item, 1));
      }
      for (size_t i = 0, n = dict_size(v->dict); i < n; ++i)
        if (py_eq(dict_value(v->dict, i), item)) return true;
      return false;
    }
    case Kind::Iter: {
      auto* g = c.as<IterObj>();
      while (g->pos < g->items.size())
        if (py_eq(g->items[g->pos++], item)) return true;
      return false;
    }
    case Kind::Missing:
    case Kind::Undefined: return false;
    default: type_error(std::string("argument of type '") + tname(c) + "' is not iterable");
  }
}

// ---------------------------------------------------------------- slicing
namespace {

int64_t slice_index(const Val& v, int64_t dflt) {
  if (v.is_none()) return dflt;
  if (v.is_intlike()) return v.i();
  if (v.is_undef() || v.is_missing()) raise_undefined(v);
  type_error("slice indices must be integers or None or have an __index__ method");
}

// PySlice_Unpack + PySlice_AdjustIndices. Returns the number of items.
int64_t adjust_slice(int64_t len, const Val& a, const Val& b, const Val& c, int64_t* start, int64_t* step) {
  const int64_t st = slice_index(c, 1);
  if (st == 0) raise(ErrorKind::Value, "slice step cannot be zero");
  int64_t s = slice_index(a, st < 0 ? INT64_MAX : 0);
  int64_t e = slice_index(b, st < 0 ? INT64_MIN : INT64_MAX);
  if (s < 0) {
    s += len;
    if (s < 0) s = st < 0 ? -1 : 0;
  } else if (s >= len) {
    s = st < 0 ? len - 1 : len;
  }
  if (e < 0) {
    e += len;
    if (e < 0) e = st < 0 ? -1 : 0;
  } else if (e >= len) {
    e = st < 0 ? len - 1 : len;
  }
  *start = s;
  *step = st;
  if (st < 0) return e < s ? (s - e - 1) / (-st) + 1 : 0;
  return s < e ? (e - s - 1) / st + 1 : 0;
}

}  // namespace

Val get_slice(const Val& obj, const Val& a, const Val& b, const Val& c) {
  if (obj.is_undef() || obj.is_missing()) raise_undefined(obj);
  int64_t start, step;
  switch (obj.kind()) {
    case Kind::Str: {
      const std::string_view s = obj.sv();
      const bool ascii = uni::is_ascii(s);
      const int64_t len = ascii ? int64_t(s.size()) : int64_t(uni::length(s));
      const int64_t n = adjust_slice(len, a, b, c, &start, &step);
      if (n <= 0) return Val::str_ref("");
      if (step == 1) {
        if (ascii) return substr(obj, size_t(start), size_t(n));
        const std::string_view sub = uni::slice(s, size_t(start), size_t(start + n));
        return substr(obj, size_t(sub.data() - s.data()), sub.size());
      }
      // Collect code point byte ranges, then pick.
      std::vector<std::pair<size_t, size_t>> cps;
      cps.reserve(size_t(len));
      for (size_t i = 0; i < s.size();) {
        const size_t p = i;
        uni::next(s, i);
        cps.emplace_back(p, i - p);
      }
      std::string out;
      for (int64_t k = 0, i = start; k < n; ++k, i += step) out.append(s.substr(cps[size_t(i)].first, cps[size_t(i)].second));
      return Val::str(std::move(out), obj.markup());
    }
    case Kind::List:
    case Kind::Tuple: {
      const int64_t n = adjust_slice(int64_t(seq_size(obj)), a, b, c, &start, &step);
      std::vector<Val> items;
      items.reserve(size_t(std::max<int64_t>(n, 0)));
      for (int64_t k = 0, i = start; k < n; ++k, i += step) items.push_back(seq_at(obj, size_t(i)));
      return obj.is_list() ? make_list(std::move(items)) : make_tuple(std::move(items));
    }
    case Kind::Range: {
      const auto* r = obj.as<RangeObj>();
      const int64_t n = adjust_slice(r->size(), a, b, c, &start, &step);
      const int64_t new_step = r->step * step;
      const int64_t new_start = r->start + start * r->step;
      return Val::object(Kind::Range, new RangeObj(new_start, new_start + n * new_step, new_step));
    }
    case Kind::Dict: {
      std::string key = "slice(";
      key += to_repr(a) + ", " + to_repr(b) + ", " + to_repr(c) + ")";
      raise(ErrorKind::Key, key);
    }
    default: type_error(std::string("'") + tname(obj) + "' object is not subscriptable");
  }
}

// ---------------------------------------------------------------- % formatting
namespace {

std::string pad_str(std::string_view s, int64_t width, bool left) {
  const int64_t len = int64_t(uni::length(s));
  if (width <= len) return std::string(s);
  std::string fill(size_t(width - len), ' ');
  return left ? std::string(s) + fill : fill + std::string(s);
}

double to_double_arg(const Val& v, char conv) {
  if (v.is_number()) return v.f();
  type_error(std::string("must be real number, not ") + tname(v) + (conv ? "" : ""));
}

}  // namespace

// Python's printf-style formatting: str % args.
Val percent_format(const Val& fmtv, const Val& args) {
  const std::string_view fmt = fmtv.sv();
  std::vector<Val> pos;
  const Val* mapping = nullptr;
  if (args.is_tuple()) {
    for (size_t i = 0, n = seq_size(args); i < n; ++i) pos.push_back(seq_at(args, i));
  } else {
    pos.push_back(args);
    if (args.is_dict()) mapping = &args;
  }
  size_t argi = 0;
  bool used_mapping = false;
  auto next_arg = [&]() -> Val {
    if (argi >= pos.size()) type_error("not enough arguments for format string");
    return pos[argi++];
  };
  std::string out;
  size_t i = 0;
  while (i < fmt.size()) {
    const char ch = fmt[i];
    if (ch != '%') {
      out.push_back(ch);
      ++i;
      continue;
    }
    const size_t spec_start = i;
    ++i;
    if (i >= fmt.size()) raise(ErrorKind::Value, "incomplete format");
    Val arg;
    bool have_arg = false;
    if (fmt[i] == '(') {
      if (!mapping) type_error("format requires a mapping");
      size_t depth = 1, j = i + 1;
      while (j < fmt.size() && depth) {
        if (fmt[j] == '(') ++depth;
        if (fmt[j] == ')') --depth;
        ++j;
      }
      if (depth) raise(ErrorKind::Value, "incomplete format key");
      const std::string key(fmt.substr(i + 1, j - i - 2));
      Val found;
      if (!dict_get_str(*mapping, key, found)) raise(ErrorKind::Key, py_repr_str(key));
      arg = found;
      have_arg = true;
      used_mapping = true;
      i = j;
    }
    std::string flags;
    while (i < fmt.size() && std::strchr("-+ #0", fmt[i])) flags.push_back(fmt[i++]);
    int64_t width = -1, prec = -1;
    if (i < fmt.size() && fmt[i] == '*') {
      const Val w = next_arg();
      if (!w.is_intlike()) type_error("* wants int");
      width = w.i();
      if (width < 0) {
        flags.push_back('-');
        width = -width;
      }
      ++i;
    } else {
      while (i < fmt.size() && fmt[i] >= '0' && fmt[i] <= '9') width = (width < 0 ? 0 : width) * 10 + (fmt[i++] - '0');
    }
    if (i < fmt.size() && fmt[i] == '.') {
      ++i;
      prec = 0;
      if (i < fmt.size() && fmt[i] == '*') {
        const Val p = next_arg();
        if (!p.is_intlike()) type_error("* wants int");
        prec = p.i();
        ++i;
      } else {
        while (i < fmt.size() && fmt[i] >= '0' && fmt[i] <= '9') prec = prec * 10 + (fmt[i++] - '0');
      }
    }
    while (i < fmt.size() && (fmt[i] == 'h' || fmt[i] == 'l' || fmt[i] == 'L')) ++i;
    if (i >= fmt.size()) raise(ErrorKind::Value, "incomplete format");
    const char conv = fmt[i++];
    if (conv == '%') {
      out.push_back('%');
      continue;
    }
    if (!have_arg) arg = next_arg();
    const bool left = flags.find('-') != std::string::npos;
    switch (conv) {
      case 's':
      case 'r':
      case 'a': {
        std::string s = conv == 's' ? to_str(arg) : to_repr(arg);
        if (prec >= 0) s = std::string(uni::slice(s, 0, size_t(prec)));
        out += pad_str(s, width, left);
        break;
      }
      case 'd':
      case 'i':
      case 'u':
      case 'o':
      case 'x':
      case 'X': {
        int64_t v;
        if (arg.is_intlike()) {
          v = arg.i();
        } else if (arg.is_float() && (conv == 'd' || conv == 'i' || conv == 'u')) {
          if (!std::isfinite(arg.f())) raise(ErrorKind::Overflow, "cannot convert float infinity to integer");
          v = int64_t(arg.f());
        } else {
          type_error(std::string("%") + conv + " format: " +
                     (conv == 'd' || conv == 'i' || conv == 'u' ? "a real number" : "an integer") +
                     " is required, not " + tname(arg));
        }
        std::string cfmt = "%" + flags;
        if (width >= 0) cfmt += std::to_string(width);
        if (prec >= 0) cfmt += "." + std::to_string(prec);
        cfmt += "ll";
        cfmt += conv == 'u' || conv == 'i' ? 'd' : conv;
        char buf[128];
        std::snprintf(buf, sizeof buf, cfmt.c_str(), static_cast<long long>(v));
        std::string s = buf;
        if (conv == 'o' && flags.find('#') != std::string::npos) {
          // Python spells the alternate octal prefix 0o.
          const size_t z = s.find('0');
          if (z != std::string::npos) s.insert(z + 1, "o");
        }
        out += s;
        break;
      }
      case 'e':
      case 'E':
      case 'f':
      case 'F':
      case 'g':
      case 'G': {
        const double v = to_double_arg(arg, conv);
        std::string cfmt = "%" + flags;
        if (width >= 0) cfmt += std::to_string(width);
        cfmt += "." + std::to_string(prec >= 0 ? prec : 6);
        cfmt += conv;
        char buf[512];
        std::snprintf(buf, sizeof buf, cfmt.c_str(), v);
        out += buf;
        break;
      }
      case 'c': {
        std::string s;
        if (arg.is_intlike())
          uni::append(s, uint32_t(arg.i()));
        else if (arg.is_str() && uni::length(arg.sv()) == 1)
          s = std::string(arg.sv());
        else
          type_error("%c requires int or char");
        out += pad_str(s, width, left);
        break;
      }
      default: {
        char buf[96];
        std::snprintf(buf, sizeof buf, "unsupported format character '%c' (0x%x) at index %zu", conv,
                      static_cast<unsigned>(static_cast<unsigned char>(conv)), i - 1);
        raise(ErrorKind::Value, buf);
      }
    }
    (void)spec_start;
  }
  if (!used_mapping && argi < pos.size() && !(mapping && argi == 0))
    type_error("not all arguments converted during string formatting");
  return Val::str(std::move(out), fmtv.markup());
}

// ---------------------------------------------------------------- str.format
namespace {

struct Spec {
  uint32_t fill = ' ';
  char align = 0;
  char sign = 0;
  bool alt = false, zero = false;
  int64_t width = -1, prec = -1;
  char grouping = 0;
  char type = 0;
};

Spec parse_spec(std::string_view s) {
  Spec sp;
  size_t i = 0;
  auto is_align = [](char c) { return c == '<' || c == '>' || c == '^' || c == '='; };
  // [[fill]align]
  if (!s.empty()) {
    size_t k = 0;
    const uint32_t first = uni::next(s, k);
    if (k < s.size() && is_align(s[k])) {
      sp.fill = first;
      sp.align = s[k];
      i = k + 1;
    } else if (is_align(s[0])) {
      sp.align = s[0];
      i = 1;
    }
  }
  if (i < s.size() && (s[i] == '+' || s[i] == '-' || s[i] == ' ')) sp.sign = s[i++];
  if (i < s.size() && s[i] == '#') {
    sp.alt = true;
    ++i;
  }
  if (i < s.size() && s[i] == '0') {
    sp.zero = true;
    ++i;
  }
  while (i < s.size() && s[i] >= '0' && s[i] <= '9') sp.width = (sp.width < 0 ? 0 : sp.width) * 10 + (s[i++] - '0');
  if (i < s.size() && (s[i] == ',' || s[i] == '_')) sp.grouping = s[i++];
  if (i < s.size() && s[i] == '.') {
    ++i;
    sp.prec = 0;
    if (i >= s.size() || s[i] < '0' || s[i] > '9') raise(ErrorKind::Value, "Format specifier missing precision");
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') sp.prec = sp.prec * 10 + (s[i++] - '0');
  }
  if (i < s.size()) sp.type = s[i++];
  if (i < s.size()) raise(ErrorKind::Value, "Invalid format specifier '" + std::string(s) + "'");
  return sp;
}

std::string align_text(std::string body, const Spec& sp, char default_align) {
  const int64_t len = int64_t(uni::length(body));
  if (sp.width <= len) return body;
  const char al = sp.align ? sp.align : default_align;
  std::string fill;
  uni::append(fill, sp.fill);
  const int64_t pad = sp.width - len;
  auto rep = [&](int64_t n) {
    std::string s;
    for (int64_t k = 0; k < n; ++k) s += fill;
    return s;
  };
  if (al == '<') return body + rep(pad);
  if (al == '^') return rep(pad / 2) + body + rep(pad - pad / 2);
  if (al == '=') {
    size_t signlen = (!body.empty() && (body[0] == '-' || body[0] == '+' || body[0] == ' ')) ? 1 : 0;
    return body.substr(0, signlen) + rep(pad) + body.substr(signlen);
  }
  return rep(pad) + body;
}

std::string group_digits(const std::string& digits, char sep) {
  std::string out;
  const size_t n = digits.size();
  for (size_t i = 0; i < n; ++i) {
    if (i && (n - i) % 3 == 0) out.push_back(sep);
    out.push_back(digits[i]);
  }
  return out;
}

std::string format_value(const Val& v, std::string_view spec_text) {
  if (spec_text.empty()) return to_str(v);
  Spec sp = parse_spec(spec_text);
  if (sp.zero && !sp.align) {
    sp.fill = '0';
    sp.align = '=';
  }
  if (v.is_str() || (!v.is_number() && !sp.type)) {
    if (!v.is_str() && !v.is_number()) {
      if (!spec_text.empty()) type_error(std::string("unsupported format string passed to ") + tname(v) + ".__format__");
    }
    if (sp.type && sp.type != 's') raise(ErrorKind::Value, std::string("Unknown format code '") + sp.type + "' for object of type 'str'");
    if (sp.sign) raise(ErrorKind::Value, "Sign not allowed in string format specifier");
    std::string s(v.sv());
    if (sp.prec >= 0) s = std::string(uni::slice(s, 0, size_t(sp.prec)));
    return align_text(std::move(s), sp, '<');
  }
  if (!v.is_number()) type_error(std::string("unsupported format string passed to ") + tname(v) + ".__format__");
  char t = sp.type;
  const bool integral = v.is_intlike() && (t == 0 || t == 'd' || t == 'b' || t == 'o' || t == 'x' || t == 'X' ||
                                           t == 'c' || t == 'n');
  std::string body;
  bool neg = false;
  if (integral) {
    int64_t x = v.i();
    neg = x < 0;
    uint64_t ux = neg ? uint64_t(0) - uint64_t(x) : uint64_t(x);
    std::string digits;
    const int base = t == 'b' ? 2 : t == 'o' ? 8 : (t == 'x' || t == 'X') ? 16 : 10;
    if (t == 'c') {
      uni::append(body, uint32_t(x));
      return align_text(body, sp, '<');
    }
    do {
      const int d = int(ux % uint64_t(base));
      digits.push_back(char(d < 10 ? '0' + d : (t == 'X' ? 'A' : 'a') + d - 10));
      ux /= uint64_t(base);
    } while (ux);
    std::reverse(digits.begin(), digits.end());
    if (sp.grouping) digits = group_digits(digits, sp.grouping);
    if (sp.alt && base != 10) body = std::string("0") + (base == 2 ? 'b' : base == 8 ? 'o' : (t == 'X' ? 'X' : 'x'));
    body += digits;
  } else {
    double x = v.f();
    if (t == 0 && v.is_intlike()) t = 'd';
    neg = std::signbit(x) && !std::isnan(x);
    x = std::fabs(x);
    if (t == 0) {
      // like 'g' with repr-style shortest digits
      if (sp.prec < 0) {
        body = format_float(x);
      } else {
        char buf[512];
        std::snprintf(buf, sizeof buf, "%.*g", int(sp.prec ? sp.prec : 1), x);
        body = buf;
      }
    } else if (t == '%') {
      char buf[512];
      std::snprintf(buf, sizeof buf, "%.*f", int(sp.prec >= 0 ? sp.prec : 6), x * 100);
      body = std::string(buf) + "%";
    } else if (std::strchr("eEfFgGn", t)) {
      char buf[512];
      const char ct = t == 'n' ? 'g' : t;
      std::string cf = std::string("%") + (sp.alt ? "#" : "") + ".*" + ct;
      std::snprintf(buf, sizeof buf, cf.c_str(), int(sp.prec >= 0 ? sp.prec : 6), x);
      body = buf;
    } else {
      raise(ErrorKind::Value, std::string("Unknown format code '") + t + "' for object of type '" + tname(v) + "'");
    }
    if (sp.grouping) {
      const size_t dot = body.find_first_not_of("0123456789");
      body = group_digits(body.substr(0, dot), sp.grouping) + (dot == std::string::npos ? "" : body.substr(dot));
    }
  }
  std::string sign;
  if (neg)
    sign = "-";
  else if (sp.sign == '+')
    sign = "+";
  else if (sp.sign == ' ')
    sign = " ";
  return align_text(sign + body, sp, '>');
}

}  // namespace

// str.format(*args, **kwargs) with the sandboxed attribute/item lookup.
Val str_format(Interp& in, const Val& fmtv, const Args& args) {
  const std::string_view f = fmtv.sv();
  std::string out;
  size_t auto_idx = 0;
  bool used_auto = false, used_manual = false;
  size_t i = 0;
  while (i < f.size()) {
    const char c = f[i];
    if (c == '}') {
      if (i + 1 < f.size() && f[i + 1] == '}') {
        out.push_back('}');
        i += 2;
        continue;
      }
      raise(ErrorKind::Value, "Single '}' encountered in format string");
    }
    if (c != '{') {
      out.push_back(c);
      ++i;
      continue;
    }
    if (i + 1 < f.size() && f[i + 1] == '{') {
      out.push_back('{');
      i += 2;
      continue;
    }
    // Find the matching '}' (one level of nested {} in the spec).
    size_t j = i + 1, depth = 1;
    while (j < f.size() && depth) {
      if (f[j] == '{') ++depth;
      if (f[j] == '}') --depth;
      if (depth) ++j;
    }
    if (j >= f.size()) raise(ErrorKind::Value, "expected '}' before end of string");
    std::string_view field = f.substr(i + 1, j - i - 1);
    i = j + 1;
    std::string_view spec, conv;
    const size_t colon = field.find(':');
    if (colon != std::string_view::npos) {
      spec = field.substr(colon + 1);
      field = field.substr(0, colon);
    }
    const size_t bang = field.find('!');
    if (bang != std::string_view::npos) {
      conv = field.substr(bang + 1);
      field = field.substr(0, bang);
    }
    // arg_name then .attr / [key] parts
    size_t k = 0;
    while (k < field.size() && field[k] != '.' && field[k] != '[') ++k;
    const std::string_view name = field.substr(0, k);
    Val v;
    if (name.empty()) {
      if (used_manual) raise(ErrorKind::Value, "cannot switch from manual field specification to automatic field numbering");
      used_auto = true;
      if (auto_idx >= args.pos.size())
        raise(ErrorKind::Index, "Replacement index " + std::to_string(auto_idx) + " out of range for positional args tuple");
      v = args.pos[auto_idx++];
    } else if (name.find_first_not_of("0123456789") == std::string_view::npos) {
      if (used_auto) raise(ErrorKind::Value, "cannot switch from automatic field numbering to manual field specification");
      used_manual = true;
      const size_t idx = size_t(std::stoull(std::string(name)));
      if (idx >= args.pos.size())
        raise(ErrorKind::Index, "Replacement index " + std::to_string(idx) + " out of range for positional args tuple");
      v = args.pos[idx];
    } else {
      bool found = false;
      for (const auto& [key, val] : args.kw)
        if (key.sv() == name) {
          v = val;
          found = true;
          break;
        }
      if (!found) raise(ErrorKind::Key, py_repr_str(name));
    }
    while (k < field.size()) {
      if (field[k] == '.') {
        size_t e = k + 1;
        while (e < field.size() && field[e] != '.' && field[e] != '[') ++e;
        const std::string attr(field.substr(k + 1, e - k - 1));
        v = get_attr(in, v, Val::str(attr));
        k = e;
      } else {
        const size_t e = field.find(']', k);
        if (e == std::string_view::npos) raise(ErrorKind::Value, "Missing ']' in format string");
        const std::string key(field.substr(k + 1, e - k - 1));
        if (!key.empty() && key.find_first_not_of("0123456789") == std::string::npos)
          v = get_item(in, v, Val::integer(std::stoll(key)));
        else
          v = get_item(in, v, Val::str(key));
        k = e + 1;
      }
    }
    if (!conv.empty()) {
      if (conv == "r" || conv == "a")
        v = Val::str(to_repr(v));
      else if (conv == "s")
        v = Val::str(to_str(v));
      else
        raise(ErrorKind::Value, "Unknown conversion specifier " + std::string(conv));
    }
    std::string spec_s(spec);
    if (spec_s.find('{') != std::string::npos) spec_s = std::string(str_format(in, Val::str(spec_s), args).sv());
    if (v.is_undef() && !spec_s.empty()) raise_undefined(v);
    out += format_value(v, spec_s);
  }
  return Val::str(std::move(out), fmtv.markup());
}

}  // namespace espetero::tmpl
