// SPDX-License-Identifier: MIT AND BSD-3-Clause
// Copyright (c) 2026 Francisco José García Barbero
//
// Portions are derived from Jinja2, Copyright 2007 Pallets; BSD-3-Clause License, see THIRD_PARTY_NOTICES.md.

// Attribute and item access with jinja2's sandbox rules, the Python methods
// of str/dict/list/..., and the global functions, filters and tests of the
// transformers chat-template environment.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>

#include "interp.h"
#include "unicode.h"

namespace espetero::tmpl {

Val str_format(Interp& in, const Val& fmt, const Args& args);  // ops.cpp
Val percent_format(const Val& fmt, const Val& args);            // ops.cpp

namespace {

[[noreturn]] void type_error(const std::string& m) { raise(ErrorKind::Type, m); }
[[noreturn]] void value_error(const std::string& m) { raise(ErrorKind::Value, m); }

}  // namespace

// ================================================================ methods
#define STR_METHODS(X)                                                                                           \
  X(capitalize) X(casefold) X(center) X(count) X(encode) X(endswith) X(expandtabs) X(find) X(format) X(format_map) \
  X(index) X(isalnum) X(isalpha) X(isascii) X(isdecimal) X(isdigit) X(isidentifier) X(islower) X(isnumeric)       \
  X(isprintable) X(isspace) X(istitle) X(isupper) X(join) X(ljust) X(lower) X(lstrip) X(maketrans) X(partition)   \
  X(removeprefix) X(removesuffix) X(replace) X(rfind) X(rindex) X(rjust) X(rpartition) X(rsplit) X(rstrip)        \
  X(split) X(splitlines) X(startswith) X(strip) X(swapcase) X(title) X(translate) X(upper) X(zfill)
#define MARKUP_METHODS(X) X(escape) X(unescape) X(striptags)
#define DICT_METHODS(X) X(clear) X(copy) X(fromkeys) X(get) X(items) X(keys) X(pop) X(popitem) X(setdefault) X(update) X(values)
#define LIST_METHODS(X) X(append) X(clear) X(copy) X(count) X(extend) X(index) X(insert) X(pop) X(remove) X(reverse) X(sort)
#define TUPLE_METHODS(X) X(count) X(index)
#define INT_METHODS(X) X(as_integer_ratio) X(bit_count) X(bit_length) X(conjugate) X(from_bytes) X(is_integer) X(to_bytes)
#define FLOAT_METHODS(X) X(as_integer_ratio) X(conjugate) X(fromhex) X(hex) X(is_integer)
#define LOOP_METHODS(X) X(cycle) X(changed)
#define CYCLER_METHODS(X) X(next) X(reset)
#define GEN_METHODS(X) X(close) X(send) X(throw)

enum MethodId : uint16_t {
#define X(n) S_##n,
  STR_METHODS(X)
#undef X
#define X(n) MK_##n,
  MARKUP_METHODS(X)
#undef X
#define X(n) D_##n,
  DICT_METHODS(X)
#undef X
#define X(n) L_##n,
  LIST_METHODS(X)
#undef X
#define X(n) T_##n,
  TUPLE_METHODS(X)
#undef X
#define X(n) R_##n,
  TUPLE_METHODS(X)
#undef X
#define X(n) I_##n,
  INT_METHODS(X)
#undef X
#define X(n) F_##n,
  FLOAT_METHODS(X)
#undef X
#define X(n) LP_##n,
  LOOP_METHODS(X)
#undef X
#define X(n) CY_##n,
  CYCLER_METHODS(X)
#undef X
#define X(n) G_##n,
  GEN_METHODS(X)
#undef X
  kMethodCount
};

static const char* const kMethodNames[] = {
#define X(n) #n,
    STR_METHODS(X) MARKUP_METHODS(X) DICT_METHODS(X) LIST_METHODS(X) TUPLE_METHODS(X) TUPLE_METHODS(X)
        INT_METHODS(X) FLOAT_METHODS(X) LOOP_METHODS(X) CYCLER_METHODS(X) GEN_METHODS(X)
#undef X
};
static_assert(sizeof(kMethodNames) / sizeof(kMethodNames[0]) == kMethodCount);

const char* method_name(uint16_t id) { return id < kMethodCount ? kMethodNames[id] : "?"; }

namespace {

struct MethodRange {
  uint16_t first, last;  // [first, last)
};
constexpr MethodRange kStr{S_capitalize, S_zfill + 1};
constexpr MethodRange kMarkup{MK_escape, MK_striptags + 1};
constexpr MethodRange kDict{D_clear, D_values + 1};
constexpr MethodRange kList{L_append, L_sort + 1};
constexpr MethodRange kTuple{T_count, T_index + 1};
constexpr MethodRange kRangeM{R_count, R_index + 1};
constexpr MethodRange kInt{I_as_integer_ratio, I_to_bytes + 1};
constexpr MethodRange kFloat{F_as_integer_ratio, F_is_integer + 1};
constexpr MethodRange kLoop{LP_cycle, LP_changed + 1};
constexpr MethodRange kCycler{CY_next, CY_reset + 1};
constexpr MethodRange kGen{G_close, G_throw + 1};

int find_method(MethodRange r, std::string_view name) {
  for (uint16_t i = r.first; i < r.last; ++i)
    if (name == kMethodNames[i]) return i;
  return -1;
}

// ImmutableSandboxedEnvironment: methods that modify builtin containers.
bool modifies_mutable(int id) {
  switch (id) {
    case D_clear:
    case D_pop:
    case D_popitem:
    case D_setdefault:
    case D_update:
    case L_append:
    case L_clear:
    case L_pop:
    case L_reverse:
    case L_insert:
    case L_sort:
    case L_extend:
    case L_remove: return true;
    default: return false;
  }
}

Val bound(const Val& self, int id) { return Val::object(Kind::Method, new MethodObj(self, uint16_t(id))); }

bool is_dunder(std::string_view n) { return n.size() > 4 && n.substr(0, 2) == "__" && n.substr(n.size() - 2) == "__"; }

}  // namespace

// ================================================================ attribute access
// Python getattr(obj, name) under ImmutableSandboxedEnvironment rules. Returns
// false when the attribute does not exist; `out` may be an Undefined carrying
// a SecurityError for unsafe attributes.
AttrHint attr_hint(std::string_view name) {
  AttrHint h;
  h.str = int16_t(find_method(kStr, name));
  h.markup = int16_t(find_method(kMarkup, name));
  h.dict = int16_t(find_method(kDict, name));
  h.list = int16_t(find_method(kList, name));
  h.tuple = int16_t(find_method(kTuple, name));
  h.dunder = is_dunder(name);
  return h;
}

int builtin_method_id(const Val& obj, const AttrHint& h) {
  if (h.dunder) return -1;
  int id = -1;
  switch (obj.kind()) {
    case Kind::Str: id = h.str >= 0 ? h.str : (obj.markup() ? h.markup : -1); break;
    case Kind::Dict: id = h.dict; break;
    case Kind::List: id = h.list; break;
    case Kind::Tuple: id = h.tuple; break;
    default: return -1;
  }
  return id >= 0 && !modifies_mutable(id) ? id : -1;
}

static bool py_getattr(const Val& obj, const Val& name, Val& out, const AttrHint* hint = nullptr) {
  const std::string_view n = name.sv();
  const PyType t = type_of(obj);
  if ((hint ? hint->dunder : is_dunder(n)) && !obj.is(Kind::Namespace)) {
    out = Val::undef_unsafe(t, name);  // every object has dunders; all are unsafe
    return true;
  }
  int id = -1;
  switch (obj.kind()) {
    case Kind::Str:
      id = hint ? hint->str : find_method(kStr, n);
      if (id < 0 && obj.markup()) id = hint ? hint->markup : find_method(kMarkup, n);
      break;
    case Kind::Dict: id = hint ? hint->dict : find_method(kDict, n); break;
    case Kind::List: id = hint ? hint->list : find_method(kList, n); break;
    case Kind::Tuple: id = hint ? hint->tuple : find_method(kTuple, n); break;
    case Kind::Range: {
      id = find_method(kRangeM, n);
      const auto* r = obj.as<RangeObj>();
      if (n == "start") return out = Val::integer(r->start), true;
      if (n == "stop") return out = Val::integer(r->stop), true;
      if (n == "step") return out = Val::integer(r->step), true;
      break;
    }
    case Kind::Bool:
    case Kind::Int:
      id = find_method(kInt, n);
      if (n == "real" || n == "numerator") return out = Val::integer(obj.i()), true;
      if (n == "imag") return out = Val::integer(0), true;
      if (n == "denominator") return out = Val::integer(1), true;
      break;
    case Kind::Float:
      id = find_method(kFloat, n);
      if (n == "real") return out = obj, true;
      if (n == "imag") return out = Val::real(0.0), true;
      break;
    case Kind::Namespace: {
      const MapData& m = obj.as<NsObj>()->m;
      const int k = m.find_str(n);
      if (k < 0) return false;
      out = n[0] == '_' ? Val::undef_unsafe(t, name) : m.vals[size_t(k)];
      return true;
    }
    case Kind::Loop: {
      auto* l = obj.as<LoopObj>();
      id = find_method(kLoop, n);
      if (n == "index") return out = Val::integer(l->index0 + 1), true;
      if (n == "index0") return out = Val::integer(l->index0), true;
      if (n == "first") return out = Val::boolean(l->index0 == 0), true;
      if (n == "last") return out = Val::boolean(l->peek() == nullptr), true;
      if (n == "length") return out = Val::integer(l->len()), true;
      if (n == "revindex") return out = Val::integer(l->len() - l->index0), true;
      if (n == "revindex0") return out = Val::integer(l->len() - l->index0 - 1), true;
      if (n == "depth") return out = Val::integer(1), true;
      if (n == "depth0") return out = Val::integer(0), true;
      if (n == "previtem") {
        out = l->index0 == 0 ? Val::undef_hint("there is no previous item") : l->before;
        return true;
      }
      if (n == "nextitem") {
        const Val* nx = l->peek();
        out = nx ? *nx : Val::undef_hint("there is no next item");
        return true;
      }
      break;
    }
    case Kind::Cycler: {
      auto* c = obj.as<CyclerObj>();
      id = find_method(kCycler, n);
      if (n == "current") return out = c->items[c->pos], true;
      if (n == "items") return out = make_tuple(c->items), true;
      if (n == "pos") return out = Val::integer(int64_t(c->pos)), true;
      break;
    }
    case Kind::Joiner: {
      auto* j = obj.as<JoinerObj>();
      if (n == "sep") return out = j->sep, true;
      if (n == "used") return out = Val::boolean(j->used), true;
      break;
    }
    case Kind::Iter: id = find_method(kGen, n); break;
    case Kind::Macro: {
      auto* m = obj.as<MacroObj>();
      if (n == "name") return out = Val::str_ref(m->def->name), true;
      if (n == "arguments") {
        std::vector<Val> names;
        for (const auto& a : m->def->args) names.push_back(Val::str_ref(a->name));
        return out = make_tuple(std::move(names)), true;
      }
      if (n == "catch_kwargs") return out = Val::boolean(m->def->catch_kwargs), true;
      if (n == "catch_varargs") return out = Val::boolean(m->def->catch_varargs), true;
      if (n == "caller") return out = Val::boolean(m->def->caller), true;
      break;
    }
    default: break;
  }
  if (id < 0) return false;
  out = modifies_mutable(id) ? Val::undef_unsafe(t, name) : bound(obj, id);
  return true;
}

// SandboxedEnvironment.getattr: attribute first, then item.
Val get_attr(Interp&, const Val& obj, const Val& name, const AttrHint* hint) {
  if (obj.is_undef() || obj.is_missing()) raise_undefined(obj);
  Val out;
  if (py_getattr(obj, name, out, hint)) return out;
  if (obj.is_dict() && dict_get_str(obj, name.sv(), out)) return out;
  return Val::undef_attr(type_of(obj), name);
}

// SandboxedEnvironment.getitem: item first, then attribute for str keys.
Val get_item(Interp&, const Val& obj, const Val& key) {
  if (obj.is_undef() || obj.is_missing()) raise_undefined(obj);
  Val out;
  switch (obj.kind()) {
    case Kind::Dict:
      if (is_hashable(key) && dict_get(obj, key, out)) return out;
      break;
    case Kind::List:
    case Kind::Tuple:
      if (key.is_intlike()) {
        int64_t i = key.i();
        const int64_t n = int64_t(seq_size(obj));
        if (i < 0) i += n;
        if (i >= 0 && i < n) return seq_at(obj, size_t(i));
      }
      break;
    case Kind::Str:
      if (key.is_intlike()) {
        const std::string_view s = obj.sv();
        const bool ascii = uni::is_ascii(s);
        const int64_t n = ascii ? int64_t(s.size()) : int64_t(uni::length(s));
        int64_t i = key.i();
        if (i < 0) i += n;
        if (i >= 0 && i < n) {
          if (ascii) return substr(obj, size_t(i), 1);
          const std::string_view c = uni::slice(s, size_t(i), size_t(i) + 1);
          return substr(obj, size_t(c.data() - s.data()), c.size());
        }
      }
      break;
    case Kind::Range:
      if (key.is_intlike()) {
        const auto* r = obj.as<RangeObj>();
        int64_t i = key.i();
        const int64_t n = r->size();
        if (i < 0) i += n;
        if (i >= 0 && i < n) return Val::integer(r->at(i));
      }
      break;
    default: break;
  }
  if (key.is_str()) {
    if (py_getattr(obj, key, out)) return out;
    return Val::undef_attr(type_of(obj), key);
  }
  if (key.is_int()) return Val::undef_elem(type_of(obj), key.i());
  return Val::undef_hint_owned(object_type_repr(type_of(obj)) + " has no element " + to_repr(key));
}

// ================================================================ argument binding
namespace {

// Binds positional/keyword arguments to `names` like a Python signature;
// `out` holds the defaults on entry (Missing marks a required parameter).
void bind(const char* fname, Args& a, std::initializer_list<const char*> names, Val* out) {
  const size_t n = names.size();
  if (a.pos.size() > n)
    type_error(std::string(fname) + "() takes at most " + std::to_string(n) + " arguments (" +
               std::to_string(a.pos.size()) + " given)");
  size_t i = 0;
  for (const char* name : names) {
    if (i < a.pos.size()) {
      if (a.has(name)) type_error(std::string(fname) + "() got multiple values for argument '" + name + "'");
      out[i] = a.pos[i];
    } else {
      Val v;
      if (a.take(name, v)) out[i] = std::move(v);
    }
    if (out[i].is_missing())
      type_error(std::string(fname) + "() missing 1 required positional argument: '" + name + "'");
    ++i;
  }
  if (!a.kw.empty())
    type_error(std::string(fname) + "() got an unexpected keyword argument '" + std::string(a.kw[0].first.sv()) + "'");
}

std::string_view need_str(const Val& v, const char* what) {
  if (!v.is_str()) type_error(std::string(what) + " must be str, not " + tname(v));
  return v.sv();
}

int64_t need_int(const Val& v, const char* what) {
  if (v.is_intlike()) return v.i();
  if (v.is_undef() || v.is_missing()) raise_undefined(v);
  type_error(std::string("'") + tname(v) + "' object cannot be interpreted as an integer" + (what ? "" : ""));
}

// Python str() of a value, keeping Markup as Markup (markupsafe.soft_str).
Val soft_str(const Val& v) {
  if (v.is_str()) return v;
  return Val::str(to_str(v));
}

Val mk(const Val& like, std::string s) { return Val::str(std::move(s), like.markup()); }

// ---- string helpers on code points
size_t space_len_at(std::string_view s, size_t i) {  // bytes of the whitespace char at i, 0 if none
  const unsigned char c = static_cast<unsigned char>(s[i]);
  if (c < 0x80) return uni::is_space(c) ? 1 : 0;
  size_t k = i;
  return uni::is_space(uni::next(s, k)) ? k - i : 0;
}

std::string_view strip_ws(std::string_view s, bool left, bool right) {
  size_t b = 0, e = s.size();
  if (left)
    while (b < e) {
      const size_t l = space_len_at(s, b);
      if (!l) break;
      b += l;
    }
  if (right)
    while (e > b) {
      size_t k = e - 1;
      while (k > b && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80) --k;
      if (!space_len_at(s, k)) break;
      e = k;
    }
  return s.substr(b, e - b);
}

std::string_view strip_chars(std::string_view s, std::string_view chars, bool left, bool right) {
  auto in_set = [&](uint32_t c) {
    size_t i = 0;
    while (i < chars.size())
      if (uni::next(chars, i) == c) return true;
    return false;
  };
  size_t b = 0, e = s.size();
  if (left)
    while (b < e) {
      size_t k = b;
      if (!in_set(uni::next(s, k))) break;
      b = k;
    }
  if (right)
    while (e > b) {
      size_t k = e - 1;
      while (k > b && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80) --k;
      size_t t = k;
      if (!in_set(uni::next(s, t))) break;
      e = k;
    }
  return s.substr(b, e - b);
}

Val do_strip(const Val& self, const Val& chars, bool left, bool right, const char* fname) {
  std::string_view r;
  if (chars.is_none() || chars.is_missing())
    r = strip_ws(self.sv(), left, right);
  else if (chars.is_str())
    r = strip_chars(self.sv(), chars.sv(), left, right);
  else
    type_error(std::string(fname) + " arg must be None or str");
  return substr(self, size_t(r.data() - self.sv().data()), r.size());
}

std::vector<Val> do_split(const Val& self, const Val& sepv, int64_t maxsplit) {
  const std::string_view s = self.sv();
  std::vector<Val> out;
  if (maxsplit < 0) maxsplit = INT64_MAX;
  if (sepv.is_none()) {
    size_t i = 0;
    const size_t n = s.size();
    while (maxsplit-- > 0) {
      while (i < n) {
        const size_t l = space_len_at(s, i);
        if (!l) break;
        i += l;
      }
      if (i == n) break;
      const size_t j = i;
      while (i < n && !space_len_at(s, i)) {
        size_t k = i;
        uni::next(s, k);
        i = k;
      }
      out.push_back(substr(self, j, i - j));
    }
    if (i < n) {
      while (i < n) {
        const size_t l = space_len_at(s, i);
        if (!l) break;
        i += l;
      }
      if (i != n) out.push_back(substr(self, i, n - i));
    }
    return out;
  }
  const std::string_view sep = need_str(sepv, "sep");
  if (sep.empty()) value_error("empty separator");
  size_t start = 0;
  while (maxsplit-- > 0) {
    const size_t p = s.find(sep, start);
    if (p == std::string_view::npos) break;
    out.push_back(substr(self, start, p - start));
    start = p + sep.size();
  }
  out.push_back(substr(self, start, s.size() - start));
  return out;
}

std::vector<Val> do_rsplit(const Val& self, const Val& sepv, int64_t maxsplit) {
  const std::string_view s = self.sv();
  std::vector<Val> out;
  if (maxsplit < 0) maxsplit = INT64_MAX;
  auto prev = [&](size_t e) {  // start of the code point ending at e
    size_t k = e - 1;
    while (k > 0 && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80) --k;
    return k;
  };
  if (sepv.is_none()) {
    size_t e = s.size();  // exclusive end of the unscanned part
    while (maxsplit-- > 0) {
      while (e > 0 && space_len_at(s, prev(e))) e = prev(e);
      if (e == 0) break;
      const size_t j = e;
      while (e > 0 && !space_len_at(s, prev(e))) e = prev(e);
      out.push_back(substr(self, e, j - e));
    }
    if (e > 0) {
      while (e > 0 && space_len_at(s, prev(e))) e = prev(e);
      if (e > 0) out.push_back(substr(self, 0, e));
    }
    std::reverse(out.begin(), out.end());
    return out;
  }
  const std::string_view sep = need_str(sepv, "sep");
  if (sep.empty()) value_error("empty separator");
  size_t end = s.size();
  while (maxsplit-- > 0) {
    if (end < sep.size()) break;
    const size_t p = s.rfind(sep, end - sep.size());
    if (p == std::string_view::npos) break;
    out.push_back(substr(self, p + sep.size(), end - p - sep.size()));
    end = p;
  }
  out.push_back(substr(self, 0, end));
  std::reverse(out.begin(), out.end());
  return out;
}

// str.splitlines: \n \r \r\n \v \f \x1c \x1d \x1e \x85
std::vector<Val> do_splitlines(const Val& self, bool keepends) {
  const std::string_view s = self.sv();
  std::vector<Val> out;
  size_t i = 0, start = 0;
  while (i < s.size()) {
    const size_t p = i;
    const uint32_t c = uni::next(s, i);
    size_t eol_len = 0;
    if (c == '\r' && i < s.size() && s[i] == '\n') {
      ++i;
      eol_len = 2;
    } else if (c == '\n' || c == '\r' || c == 0x0B || c == 0x0C || c == 0x1C || c == 0x1D || c == 0x1E ||
               c == 0x85 || c == 0x2028 || c == 0x2029) {
      eol_len = i - p;
    }
    if (eol_len) {
      out.push_back(substr(self, start, (keepends ? i : p) - start));
      start = i;
    }
  }
  if (start < s.size()) out.push_back(substr(self, start, s.size() - start));
  return out;
}

Val do_replace(const Val& self, std::string_view old, std::string_view repl, int64_t count, bool markup) {
  const std::string_view s = self.sv();
  if (count < 0) count = INT64_MAX;
  std::string out;
  if (old.empty()) {
    size_t i = 0;
    while (i < s.size() && count > 0) {
      out += repl;
      --count;
      const size_t p = i;
      uni::next(s, i);
      out.append(s.substr(p, i - p));
    }
    out.append(s.substr(i));
    if (count > 0) out += repl;
    return Val::str(std::move(out), markup);
  }
  size_t start = 0;
  while (count-- > 0) {
    const size_t p = s.find(old, start);
    if (p == std::string_view::npos) break;
    out.append(s.substr(start, p - start));
    out += repl;
    start = p + old.size();
  }
  if (start == 0 && out.empty()) return self.markup() == markup ? self : Val::str(std::string(s), markup);
  out.append(s.substr(start));
  return Val::str(std::move(out), markup);
}

// Code point range [b, e) of the optional start/end arguments (slice rules).
void sub_range(std::string_view s, const Val& start, const Val& end, size_t* bb, size_t* eb) {
  const int64_t n = int64_t(uni::length(s));
  auto norm = [&](const Val& v, int64_t dflt) -> int64_t {
    if (v.is_missing() || v.is_none()) return dflt;
    int64_t x = need_int(v, nullptr);
    if (x < 0) {
      x += n;
      if (x < 0) x = 0;
    }
    return std::min(x, n);
  };
  const int64_t b = norm(start, 0), e = norm(end, n);
  *bb = uni::byte_offset(s, size_t(b));
  *eb = e < b ? *bb : uni::byte_offset(s, size_t(e));
}

bool has_affix(const Val& self, const Val& affix, const Val& start, const Val& end, bool prefix) {
  size_t b, e;
  sub_range(self.sv(), start, end, &b, &e);
  const std::string_view s = self.sv().substr(b, e - b);
  auto test = [&](std::string_view a) {
    if (a.size() > s.size()) return false;
    return prefix ? s.substr(0, a.size()) == a : s.substr(s.size() - a.size()) == a;
  };
  if (affix.is_str()) return test(affix.sv());
  if (affix.is_tuple()) {
    for (size_t i = 0, n = seq_size(affix); i < n; ++i) {
      const Val x = seq_at(affix, i);
      if (!x.is_str())
        type_error(std::string("tuple for ") + (prefix ? "startswith" : "endswith") + " must only contain str, not " +
                   tname(x));
      if (test(x.sv())) return true;
    }
    return false;
  }
  type_error(std::string(prefix ? "startswith" : "endswith") + " first arg must be str or a tuple of str, not " +
             tname(affix));
}

int64_t do_find(const Val& self, const Val& subv, const Val& start, const Val& end, bool reverse) {
  const std::string_view sub = need_str(subv, "must be str, not");
  size_t b, e;
  sub_range(self.sv(), start, end, &b, &e);
  const std::string_view s = self.sv();
  if (e < b || (e - b) < sub.size()) return -1;
  const std::string_view region = s.substr(b, e - b);
  const size_t p = reverse ? region.rfind(sub) : region.find(sub);
  if (p == std::string_view::npos) return -1;
  return int64_t(uni::length(s.substr(0, b + p)));
}

std::string pad(std::string_view s, int64_t width, std::string_view fill, int where) {  // where: 0 left 1 right 2 center
  const int64_t len = int64_t(uni::length(s));
  if (width <= len) return std::string(s);
  const int64_t marg = width - len;
  int64_t left = 0;
  if (where == 0)
    left = 0;
  else if (where == 1)
    left = marg;
  else
    left = marg / 2 + (marg & width & 1);
  std::string out;
  for (int64_t k = 0; k < left; ++k) out += fill;
  out += s;
  for (int64_t k = left; k < marg; ++k) out += fill;
  return out;
}

std::string_view fill_char(const Val& v) {
  if (v.is_missing()) return " ";
  if (!v.is_str() || uni::length(v.sv()) != 1) type_error("The fill character must be exactly one character long");
  return v.sv();
}

bool str_all(std::string_view s, bool (*pred)(uint32_t)) {
  if (s.empty()) return false;
  size_t i = 0;
  while (i < s.size())
    if (!pred(uni::next(s, i))) return false;
  return true;
}

// str.islower / str.isupper: every cased character has the case, and there is one.
bool str_islower(std::string_view s) {
  bool cased = false;
  size_t i = 0;
  while (i < s.size()) {
    const uint32_t c = uni::next(s, i);
    if (uni::is_upper(c) || uni::is_titlecase(c)) return false;
    if (uni::is_lower(c)) cased = true;
  }
  return cased;
}

bool str_isupper(std::string_view s) {
  bool cased = false;
  size_t i = 0;
  while (i < s.size()) {
    const uint32_t c = uni::next(s, i);
    if (uni::is_lower(c) || uni::is_titlecase(c)) return false;
    if (uni::is_upper(c)) cased = true;
  }
  return cased;
}

Val make_view(const Val& d, ViewKind k) { return Val::object(Kind::View, new ViewObj(d, k)); }

Val copy_dict(const Val& d) {
  MapData m;
  for (size_t i = 0, n = dict_size(d); i < n; ++i) m.set(dict_key(d, i), dict_value(d, i));
  return make_dict(std::move(m));
}

std::string html_unescape(std::string_view s) {
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    if (s[i] != '&') {
      out.push_back(s[i++]);
      continue;
    }
    const size_t semi = s.find(';', i);
    if (semi == std::string_view::npos || semi - i > 12) {
      out.push_back(s[i++]);
      continue;
    }
    const std::string_view ent = s.substr(i + 1, semi - i - 1);
    uint32_t cp = 0;
    if (!ent.empty() && ent[0] == '#') {
      const bool hex = ent.size() > 1 && (ent[1] == 'x' || ent[1] == 'X');
      const std::string digits(ent.substr(hex ? 2 : 1));
      if (!digits.empty()) cp = uint32_t(std::strtoul(digits.c_str(), nullptr, hex ? 16 : 10));
    } else if (ent == "amp") {
      cp = '&';
    } else if (ent == "lt") {
      cp = '<';
    } else if (ent == "gt") {
      cp = '>';
    } else if (ent == "quot") {
      cp = '"';
    } else if (ent == "apos") {
      cp = '\'';
    } else if (ent == "nbsp") {
      cp = 0xA0;
    }
    if (!cp) {
      out.push_back(s[i++]);
      continue;
    }
    uni::append(out, cp);
    i = semi + 1;
  }
  return out;
}

std::string strip_tags(std::string_view s) {
  std::string t;
  size_t i = 0;
  while (i < s.size()) {
    if (s.compare(i, 4, "<!--") == 0) {
      const size_t e = s.find("-->", i + 4);
      if (e != std::string_view::npos) {
        i = e + 3;
        continue;
      }
    }
    if (s[i] == '<') {
      const size_t e = s.find('>', i + 1);
      if (e != std::string_view::npos) {
        i = e + 1;
        continue;
      }
    }
    t.push_back(s[i++]);
  }
  std::string joined;
  Val tv = Val::str(std::move(t));
  for (const Val& w : do_split(tv, Val::none(), -1)) {
    if (!joined.empty()) joined.push_back(' ');
    joined += w.sv();
  }
  return html_unescape(joined);
}

}  // namespace

Val call_method(Interp& in, const MethodObj& m, Args& a) { return call_method(in, m.self, m.id, a); }

Val call_method(Interp& in, const Val& self, int id, Args& a) {
  const char* name = method_name(uint16_t(id));
  auto args = [&](std::initializer_list<const char*> names, std::initializer_list<Val> defaults, Val* out) {
    size_t i = 0;
    for (const Val& d : defaults) out[i++] = d;
    bind(name, a, names, out);
  };
  const Val M = Val::missing();
  Val v[4];
  switch (id) {
    // ---- str
    case S_capitalize: args({}, {}, v); return mk(self, uni::capitalize(self.sv()));
    case S_casefold: args({}, {}, v); return mk(self, uni::lower(self.sv()));
    case S_lower: args({}, {}, v); return mk(self, uni::lower(self.sv()));
    case S_upper: args({}, {}, v); return mk(self, uni::upper(self.sv()));
    case S_swapcase: args({}, {}, v); return mk(self, uni::swapcase(self.sv()));
    case S_title: args({}, {}, v); return mk(self, uni::title(self.sv()));
    case S_center:
    case S_ljust:
    case S_rjust:
      args({"width", "fillchar"}, {M, M}, v);
      return mk(self, pad(self.sv(), need_int(v[0], nullptr), fill_char(v[1]), id == S_ljust ? 0 : id == S_rjust ? 1 : 2));
    case S_zfill: {
      args({"width"}, {M}, v);
      const std::string_view s = self.sv();
      const int64_t w = need_int(v[0], nullptr), len = int64_t(uni::length(s));
      if (w <= len) return self;
      std::string out;
      size_t signlen = (!s.empty() && (s[0] == '+' || s[0] == '-')) ? 1 : 0;
      out.append(s.substr(0, signlen));
      out.append(size_t(w - len), '0');
      out.append(s.substr(signlen));
      return mk(self, std::move(out));
    }
    case S_count: {
      args({"sub", "start", "end"}, {M, Val::none(), Val::none()}, v);
      const std::string_view sub = need_str(v[0], "must be str, not");
      size_t b, e;
      sub_range(self.sv(), v[1], v[2], &b, &e);
      const std::string_view region = self.sv().substr(b, e - b);
      if (sub.empty()) return Val::integer(int64_t(uni::length(region)) + 1);
      int64_t c = 0;
      for (size_t p = region.find(sub); p != std::string_view::npos; p = region.find(sub, p + sub.size())) ++c;
      return Val::integer(c);
    }
    case S_startswith:
    case S_endswith:
      args({"prefix", "start", "end"}, {M, Val::none(), Val::none()}, v);
      return Val::boolean(has_affix(self, v[0], v[1], v[2], id == S_startswith));
    case S_find:
    case S_rfind:
    case S_index:
    case S_rindex: {
      args({"sub", "start", "end"}, {M, Val::none(), Val::none()}, v);
      const int64_t r = do_find(self, v[0], v[1], v[2], id == S_rfind || id == S_rindex);
      if (r < 0 && (id == S_index || id == S_rindex)) value_error("substring not found");
      return Val::integer(r);
    }
    case S_format: return str_format(in, self, a);
    case S_format_map: {
      args({"mapping"}, {M}, v);
      if (!v[0].is_dict()) type_error("format_map() argument must be a mapping");
      Args fa;
      for (size_t i = 0, n = dict_size(v[0]); i < n; ++i) fa.kw.emplace_back(dict_key(v[0], i), dict_value(v[0], i));
      return str_format(in, self, fa);
    }
    case S_isalnum: args({}, {}, v); return Val::boolean(str_all(self.sv(), uni::is_alnum));
    case S_isalpha: args({}, {}, v); return Val::boolean(str_all(self.sv(), uni::is_alpha));
    case S_isascii: args({}, {}, v); return Val::boolean(uni::is_ascii(self.sv()));
    case S_isdecimal: args({}, {}, v); return Val::boolean(str_all(self.sv(), uni::is_decimal));
    case S_isdigit: args({}, {}, v); return Val::boolean(str_all(self.sv(), uni::is_digit));
    case S_isnumeric: args({}, {}, v); return Val::boolean(str_all(self.sv(), uni::is_numeric));
    case S_isspace: args({}, {}, v); return Val::boolean(str_all(self.sv(), uni::is_space));
    case S_isprintable: args({}, {}, v); return Val::boolean(self.sv().empty() || str_all(self.sv(), uni::is_printable));
    case S_isidentifier: {
      args({}, {}, v);
      const std::string_view s = self.sv();
      if (s.empty()) return Val::boolean(false);
      size_t i = 0;
      const uint32_t c0 = uni::next(s, i);
      if (!(c0 == '_' || uni::is_alpha(c0))) return Val::boolean(false);
      while (i < s.size())
        if (!uni::is_word(uni::next(s, i))) return Val::boolean(false);
      return Val::boolean(true);
    }
    case S_islower: args({}, {}, v); return Val::boolean(str_islower(self.sv()));
    case S_isupper: args({}, {}, v); return Val::boolean(str_isupper(self.sv()));
    case S_istitle: {
      args({}, {}, v);
      bool cased = false, prev = false;
      const std::string_view s = self.sv();
      size_t i = 0;
      while (i < s.size()) {
        const uint32_t c = uni::next(s, i);
        if (uni::is_upper(c) || uni::is_titlecase(c)) {
          if (prev) return Val::boolean(false);
          prev = cased = true;
        } else if (uni::is_lower(c)) {
          if (!prev) return Val::boolean(false);
          prev = cased = true;
        } else {
          prev = false;
        }
      }
      return Val::boolean(cased);
    }
    case S_join: {
      args({"iterable"}, {M}, v);
      std::string out;
      PyIter it(v[0]);
      Val x;
      size_t i = 0;
      while (it.next(x)) {
        if (!x.is_str())
          type_error("sequence item " + std::to_string(i) + ": expected str instance, " + tname(x) + " found");
        if (i++) out += self.sv();
        out += (self.markup() && !x.markup()) ? html_escape(x.sv()) : std::string(x.sv());
      }
      return mk(self, std::move(out));
    }
    case S_strip: args({"chars"}, {Val::none()}, v); return do_strip(self, v[0], true, true, "strip");
    case S_lstrip: args({"chars"}, {Val::none()}, v); return do_strip(self, v[0], true, false, "lstrip");
    case S_rstrip: args({"chars"}, {Val::none()}, v); return do_strip(self, v[0], false, true, "rstrip");
    case S_split:
    case S_rsplit: {
      args({"sep", "maxsplit"}, {Val::none(), Val::integer(-1)}, v);
      const int64_t maxsplit = need_int(v[1], nullptr);
      return make_list(id == S_split ? do_split(self, v[0], maxsplit) : do_rsplit(self, v[0], maxsplit));
    }
    case S_splitlines: args({"keepends"}, {Val::boolean(false)}, v); return make_list(do_splitlines(self, truthy(v[0])));
    case S_replace: {
      args({"old", "new", "count"}, {M, M, Val::integer(-1)}, v);
      std::string old(need_str(v[0], "replace() argument 1")), repl(need_str(v[1], "replace() argument 2"));
      if (self.markup()) {  // Markup methods escape their arguments
        if (!v[0].markup()) old = html_escape(old);
        if (!v[1].markup()) repl = html_escape(repl);
      }
      return do_replace(self, old, repl, need_int(v[2], nullptr), self.markup());
    }
    case S_partition:
    case S_rpartition: {
      args({"sep"}, {M}, v);
      const std::string_view sep = need_str(v[0], "must be str, not");
      if (sep.empty()) value_error("empty separator");
      const std::string_view s = self.sv();
      const size_t p = id == S_partition ? s.find(sep) : s.rfind(sep);
      if (p == std::string_view::npos) {
        if (id == S_partition) return make_tuple({self, Val::str_ref(""), Val::str_ref("")});
        return make_tuple({Val::str_ref(""), Val::str_ref(""), self});
      }
      return make_tuple({substr(self, 0, p), v[0], substr(self, p + sep.size(), s.size() - p - sep.size())});
    }
    case S_removeprefix:
    case S_removesuffix: {
      args({"affix"}, {M}, v);
      const std::string_view x = need_str(v[0], "must be str, not"), s = self.sv();
      if (id == S_removeprefix && s.substr(0, x.size()) == x) return substr(self, x.size(), s.size() - x.size());
      if (id == S_removesuffix && !x.empty() && s.size() >= x.size() && s.substr(s.size() - x.size()) == x)
        return substr(self, 0, s.size() - x.size());
      return self;
    }
    case S_expandtabs: {
      args({"tabsize"}, {Val::integer(8)}, v);
      const int64_t ts = need_int(v[0], nullptr);
      std::string out;
      int64_t col = 0;
      const std::string_view s = self.sv();
      size_t i = 0;
      while (i < s.size()) {
        const size_t p = i;
        const uint32_t c = uni::next(s, i);
        if (c == '\t') {
          if (ts > 0) {
            const int64_t n = ts - col % ts;
            out.append(size_t(n), ' ');
            col += n;
          }
        } else {
          out.append(s.substr(p, i - p));
          col = (c == '\n' || c == '\r') ? 0 : col + 1;
        }
      }
      return mk(self, std::move(out));
    }
    case MK_escape: args({"s"}, {M}, v); return v[0].markup() ? v[0] : Val::str(html_escape(to_str(v[0])), true);
    case MK_unescape: args({}, {}, v); return Val::str(html_unescape(self.sv()));
    case MK_striptags: args({}, {}, v); return Val::str(strip_tags(self.sv()));
    // ---- dict
    case D_get: {
      args({"key", "default"}, {M, Val::none()}, v);
      require_hashable(v[0]);
      Val out;
      return dict_get(self, v[0], out) ? out : v[1];
    }
    case D_items: args({}, {}, v); return make_view(self, ViewKind::Items);
    case D_keys: args({}, {}, v); return make_view(self, ViewKind::Keys);
    case D_values: args({}, {}, v); return make_view(self, ViewKind::Values);
    case D_copy: args({}, {}, v); return copy_dict(self);
    // ---- list / tuple / range
    case L_copy: args({}, {}, v); return make_list(to_vector(self));
    case L_count:
    case T_count:
    case R_count: {
      args({"value"}, {M}, v);
      int64_t c = 0;
      PyIter it(self);
      Val x;
      while (it.next(x)) c += py_eq(x, v[0]);
      return Val::integer(c);
    }
    case L_index:
    case T_index:
    case R_index: {
      args({"value", "start", "stop"}, {M, Val::none(), Val::none()}, v);
      std::vector<Val> items = to_vector(self);
      const int64_t n = int64_t(items.size());
      auto norm = [&](const Val& x, int64_t d) {
        if (x.is_none()) return d;
        int64_t i = need_int(x, nullptr);
        if (i < 0) i = std::max<int64_t>(0, i + n);
        return std::min(i, n);
      };
      for (int64_t i = norm(v[1], 0), e = norm(v[2], n); i < e; ++i)
        if (py_eq(items[size_t(i)], v[0])) return Val::integer(i);
      if (id == T_index) value_error("tuple.index(x): x not in tuple");
      if (id == R_index) value_error(to_repr(v[0]) + " is not in range");
      value_error(to_repr(v[0]) + " is not in list");
    }
    // ---- loop
    case LP_cycle: {
      auto* l = self.as<LoopObj>();
      if (!a.kw.empty()) type_error("LoopContext.cycle() got an unexpected keyword argument");
      if (a.pos.empty()) type_error("no items for cycling given");
      return a.pos[size_t(l->index0 % int64_t(a.pos.size()))];
    }
    case LP_changed: {
      auto* l = self.as<LoopObj>();
      Val t = make_tuple(a.pos);
      if (l->last_changed.is_missing() || !py_eq(l->last_changed, t)) {
        l->last_changed = t;
        return Val::boolean(true);
      }
      return Val::boolean(false);
    }
    // ---- cycler
    case CY_next: {
      args({}, {}, v);
      auto* c = self.as<CyclerObj>();
      Val r = c->items[c->pos];
      c->pos = (c->pos + 1) % c->items.size();
      return r;
    }
    case CY_reset: {
      args({}, {}, v);
      self.as<CyclerObj>()->pos = 0;
      return Val::none();
    }
    case I_is_integer:
    case F_is_integer:
      args({}, {}, v);
      return Val::boolean(self.is_intlike() || (std::isfinite(self.f()) && self.f() == std::floor(self.f())));
    case I_bit_length: {
      args({}, {}, v);
      uint64_t x = self.i() < 0 ? uint64_t(0) - uint64_t(self.i()) : uint64_t(self.i());
      int64_t n = 0;
      while (x) {
        ++n;
        x >>= 1;
      }
      return Val::integer(n);
    }
    case I_conjugate:
    case F_conjugate: args({}, {}, v); return self.is_float() ? self : Val::integer(self.i());
    default: break;
  }
  raise(ErrorKind::Unsupported, std::string("method '") + name + "' is not supported");
}

// ================================================================ globals
Val lookup_global(std::string_view name) {
  if (name == "range") return Val::builtin(BuiltinId::Range);
  if (name == "dict") return Val::builtin(BuiltinId::Dict);
  if (name == "namespace") return Val::builtin(BuiltinId::Namespace);
  if (name == "lipsum") return Val::builtin(BuiltinId::Lipsum);
  if (name == "cycler") return Val::builtin(BuiltinId::Cycler);
  if (name == "joiner") return Val::builtin(BuiltinId::Joiner);
  if (name == "raise_exception") return Val::builtin(BuiltinId::RaiseException);
  if (name == "strftime_now") return Val::builtin(BuiltinId::StrftimeNow);
  return Val::missing();
}

namespace {

// dict(*args, **kwargs)
MapData build_dict(Args& a, const char* fname) {
  MapData m;
  if (a.pos.size() > 1)
    type_error(std::string(fname) + " expected at most 1 argument, got " + std::to_string(a.pos.size()));
  if (a.pos.size() == 1) {
    const Val& src = a.pos[0];
    if (src.is_dict()) {
      for (size_t i = 0, n = dict_size(src); i < n; ++i) m.set(dict_key(src, i), dict_value(src, i));
    } else {
      PyIter it(src);
      Val item;
      size_t idx = 0;
      while (it.next(item)) {
        if (!(item.is_seq() || item.is_str() || item.is_dict() || item.is(Kind::Iter) || item.is(Kind::Range) ||
              item.is(Kind::View)))
          type_error("cannot convert dictionary update sequence element #" + std::to_string(idx) + " to a sequence");
        std::vector<Val> kv = to_vector(item);
        if (kv.size() != 2)
          value_error("dictionary update sequence element #" + std::to_string(idx) + " has length " +
                      std::to_string(kv.size()) + "; 2 is required");
        require_hashable(kv[0]);
        m.set(std::move(kv[0]), std::move(kv[1]));
        ++idx;
      }
    }
  }
  for (auto& [k, v] : a.kw) m.set(k, v);
  return m;
}

std::string strftime_py(const std::string& fmt, const RenderOptions& opts) {
  std::tm tm{};
  if (opts.now) {
    tm = *opts.now;
    std::tm copy = tm;
    const time_t t = timegm(&copy);  // normalise and fill tm_wday / tm_yday
    gmtime_r(&t, &tm);
  } else {
    const time_t t = std::time(nullptr);
    localtime_r(&t, &tm);
  }
  // Python's datetime.strftime handles %f, %z and %Z itself (naive datetime:
  // microseconds, then empty offset / zone name).
  std::string f;
  for (size_t i = 0; i < fmt.size(); ++i) {
    if (fmt[i] == '%' && i + 1 < fmt.size()) {
      const char c = fmt[i + 1];
      if (c == 'f') {
        f += "000000";
        ++i;
        continue;
      }
      if (c == 'z' || c == 'Z') {
        ++i;
        continue;
      }
      if (c == ':' && i + 2 < fmt.size() && fmt[i + 2] == 'z') {
        i += 2;
        continue;
      }
      f.push_back('%');
      f.push_back(c);
      ++i;
      continue;
    }
    f.push_back(fmt[i]);
  }
  if (f.empty()) return "";
  std::string out(256, '\0');
  for (int attempt = 0; attempt < 8; ++attempt) {
    const size_t n = std::strftime(out.data(), out.size(), f.c_str(), &tm);
    if (n > 0) {
      out.resize(n);
      return out;
    }
    out.resize(out.size() * 4);
  }
  return "";
}

}  // namespace

Val call_builtin(Interp& in, BuiltinId id, Args& a) {
  switch (id) {
    case BuiltinId::Range: {
      if (!a.kw.empty())
        type_error("safe_range() got an unexpected keyword argument '" + std::string(a.kw[0].first.sv()) + "'");
      if (a.pos.empty()) type_error("range expected at least 1 argument, got 0");
      if (a.pos.size() > 3) type_error("range expected at most 3 arguments, got " + std::to_string(a.pos.size()));
      int64_t v[3];
      for (size_t i = 0; i < a.pos.size(); ++i) v[i] = need_int(a.pos[i], nullptr);
      int64_t start = 0, stop, step = 1;
      if (a.pos.size() == 1) {
        stop = v[0];
      } else {
        start = v[0];
        stop = v[1];
        if (a.pos.size() == 3) step = v[2];
      }
      if (step == 0) value_error("range() arg 3 must not be zero");
      auto* r = new RangeObj(start, stop, step);
      Val rv = Val::object(Kind::Range, r);
      if (r->size() > 100000)
        raise(ErrorKind::Overflow, "Range too big. The sandbox blocks ranges larger than MAX_RANGE (100000).");
      return rv;
    }
    case BuiltinId::Dict: return make_dict(build_dict(a, "dict"));
    case BuiltinId::Namespace: {
      auto* ns = new NsObj;
      Val v = Val::object(Kind::Namespace, ns);
      ns->m = build_dict(a, "dict");
      in.track(ns);
      return v;
    }
    case BuiltinId::Cycler: {
      if (!a.kw.empty()) type_error("Cycler.__init__() got an unexpected keyword argument");
      if (a.pos.empty()) raise(ErrorKind::Runtime, "at least one item has to be provided");
      auto* c = new CyclerObj;
      c->items = a.pos;
      return Val::object(Kind::Cycler, c);
    }
    case BuiltinId::Joiner: {
      Val v[1] = {Val::str_ref(", ")};
      bind("Joiner.__init__", a, {"sep"}, v);
      auto* j = new JoinerObj;
      j->sep = v[0];
      return Val::object(Kind::Joiner, j);
    }
    case BuiltinId::Lipsum: raise(ErrorKind::Unsupported, "lipsum() is not supported");
    case BuiltinId::RaiseException: {
      Val v[1] = {Val::missing()};
      if (a.pos.empty() && !a.has("message"))
        type_error("_cached_compile_jinja_template.<locals>.raise_exception() missing 1 required positional argument: "
                   "'message'");
      bind("raise_exception", a, {"message"}, v);
      throw TemplateError(ErrorKind::Template, to_str(v[0]));
    }
    case BuiltinId::StrftimeNow: {
      Val v[1] = {Val::missing()};
      bind("strftime_now", a, {"format"}, v);
      if (!v[0].is_str()) type_error(std::string("strftime() argument 1 must be str, not ") + tname(v[0]));
      return Val::str(strftime_py(std::string(v[0].sv()), in.opts));
    }
    case BuiltinId::TemplateReference: type_error("'TemplateReference' object is not callable");
    case BuiltinId::Generation: {
      Val caller;
      a.take("caller", caller);
      Args none;
      return in.call(caller, none);
    }
  }
  raise(ErrorKind::Runtime, "unknown builtin");
}

// ================================================================ filters
#define FILTERS(X)                                                                                             \
  X(abs) X(attr) X(batch) X(capitalize) X(center) X(count) X(d) X(default) X(dictsort) X(e) X(escape)             \
  X(filesizeformat) X(first) X(float) X(forceescape) X(format) X(groupby) X(indent) X(int) X(join) X(last)         \
  X(length) X(list) X(lower) X(items) X(map) X(min) X(max) X(pprint) X(random) X(reject) X(rejectattr) X(replace)  \
  X(reverse) X(round) X(safe) X(select) X(selectattr) X(slice) X(sort) X(string) X(striptags) X(sum) X(title)      \
  X(trim) X(truncate) X(unique) X(upper) X(urlencode) X(urlize) X(wordcount) X(wordwrap) X(xmlattr) X(tojson)

enum FilterIdEnum : int {
#define X(n) FI_##n,
  FILTERS(X)
#undef X
  kFilterCount
};
static const char* const kFilterNames[] = {
#define X(n) #n,
    FILTERS(X)
#undef X
};

int filter_id(std::string_view name) {
  for (int i = 0; i < kFilterCount; ++i)
    if (name == kFilterNames[i]) return i;
  return -1;
}

// ================================================================ tests
enum TestIdEnum : int {
  TI_odd,
  TI_even,
  TI_divisibleby,
  TI_defined,
  TI_undefined,
  TI_filter,
  TI_test,
  TI_none,
  TI_boolean,
  TI_false,
  TI_true,
  TI_integer,
  TI_float,
  TI_lower,
  TI_upper,
  TI_string,
  TI_mapping,
  TI_number,
  TI_sequence,
  TI_iterable,
  TI_callable,
  TI_sameas,
  TI_escaped,
  TI_in,
  TI_eq,
  TI_ne,
  TI_gt,
  TI_ge,
  TI_lt,
  TI_le,
};
static const struct {
  const char* name;
  int id;
} kTests[] = {
    {"odd", TI_odd},         {"even", TI_even},         {"divisibleby", TI_divisibleby}, {"defined", TI_defined},
    {"undefined", TI_undefined}, {"filter", TI_filter}, {"test", TI_test},             {"none", TI_none},
    {"boolean", TI_boolean}, {"false", TI_false},       {"true", TI_true},             {"integer", TI_integer},
    {"float", TI_float},     {"lower", TI_lower},       {"upper", TI_upper},           {"string", TI_string},
    {"mapping", TI_mapping}, {"number", TI_number},     {"sequence", TI_sequence},     {"iterable", TI_iterable},
    {"callable", TI_callable}, {"sameas", TI_sameas},   {"escaped", TI_escaped},       {"in", TI_in},
    {"==", TI_eq},           {"eq", TI_eq},             {"equalto", TI_eq},            {"!=", TI_ne},
    {"ne", TI_ne},           {">", TI_gt},              {"gt", TI_gt},                 {"greaterthan", TI_gt},
    {"ge", TI_ge},           {">=", TI_ge},             {"<", TI_lt},                  {"lt", TI_lt},
    {"lessthan", TI_lt},     {"<=", TI_le},             {"le", TI_le},
};

int test_id(std::string_view name) {
  for (const auto& t : kTests)
    if (name == t.name) return t.id;
  return -1;
}

bool call_test(Interp&, int id, const std::string& name, const Val& value, Args& a) {
  if (id < 0) raise(ErrorKind::Runtime, "No test named " + py_repr_str(name) + " found.");
  auto one_arg = [&](const char* fn, const char* pname) -> Val {
    Val v[1] = {Val::missing()};
    bind(fn, a, {pname}, v);
    return v[0];
  };
  auto no_args = [&](const char* fn) {
    Val v[1];
    bind(fn, a, {}, v);
  };
  switch (id) {
    case TI_odd: no_args("test_odd"); return py_eq(py_arith(EK::Mod, value, Val::integer(2)), Val::integer(1));
    case TI_even: no_args("test_even"); return py_eq(py_arith(EK::Mod, value, Val::integer(2)), Val::integer(0));
    case TI_divisibleby: {
      const Val num = one_arg("test_divisibleby", "num");
      return py_eq(py_arith(EK::Mod, value, num), Val::integer(0));
    }
    case TI_defined: no_args("test_defined"); return !(value.is_undef() || value.is_missing());
    case TI_undefined: no_args("test_undefined"); return value.is_undef() || value.is_missing();
    case TI_filter:
      no_args("test_filter");
      return value.is_str() && filter_id(value.sv()) >= 0;
    case TI_test:
      no_args("test_test");
      return value.is_str() && test_id(value.sv()) >= 0;
    case TI_none: no_args("test_none"); return value.is_none();
    case TI_boolean: no_args("test_boolean"); return value.is_bool();
    case TI_false: no_args("test_false"); return value.is_bool() && !value.b();
    case TI_true: no_args("test_true"); return value.is_bool() && value.b();
    case TI_integer: no_args("test_integer"); return value.is_int();
    case TI_float: no_args("test_float"); return value.is_float();
    case TI_lower: no_args("test_lower"); return str_islower(to_str(value));
    case TI_upper: no_args("test_upper"); return str_isupper(to_str(value));
    case TI_string: no_args("test_string"); return value.is_str();
    case TI_mapping: no_args("test_mapping"); return value.is_dict();
    case TI_number: no_args("test_number"); return value.is_number();
    case TI_sequence:
      no_args("test_sequence");
      switch (value.kind()) {
        case Kind::Missing:
        case Kind::Undefined:
        case Kind::Str:
        case Kind::List:
        case Kind::Tuple:
        case Kind::Dict:
        case Kind::Range: return true;
        default: return false;
      }
    case TI_iterable:
      no_args("test_iterable");
      switch (value.kind()) {
        case Kind::Missing:
        case Kind::Undefined:
        case Kind::Str:
        case Kind::List:
        case Kind::Tuple:
        case Kind::Dict:
        case Kind::Range:
        case Kind::View:
        case Kind::Iter:
        case Kind::Loop: return true;
        default: return false;
      }
    case TI_callable:
      no_args("callable");
      switch (value.kind()) {
        case Kind::Missing:
        case Kind::Undefined:
        case Kind::Macro:
        case Kind::Builtin:
        case Kind::Method:
        case Kind::Joiner:
        case Kind::Loop: return true;
        default: return false;
      }
    case TI_sameas: {
      const Val other = one_arg("test_sameas", "other");
      if (value.kind() != other.kind()) return false;
      switch (value.kind()) {
        case Kind::None: return true;
        case Kind::Bool: return value.b() == other.b();
        case Kind::Int: return value.i() == other.i() && value.i() >= -5 && value.i() <= 256;
        case Kind::Str: return value.sv().data() == other.sv().data() && value.sv().size() == other.sv().size();
        case Kind::Builtin: return value.builtin_id() == other.builtin_id();
        case Kind::List:
        case Kind::Dict:
          if (value.is_json() || other.is_json()) return value.is_json() && other.is_json() && value.json() == other.json();
          return value.same_object(other);
        case Kind::Undefined:
        case Kind::Missing: return false;
        default: return value.same_object(other);
      }
    }
    case TI_escaped: no_args("test_escaped"); return is_markup(value);
    case TI_in: return py_contains(one_arg("test_in", "seq"), value);
    case TI_eq: return py_eq(value, one_arg("eq", "b"));
    case TI_ne: return !py_eq(value, one_arg("ne", "b"));
    case TI_lt: return py_compare(0, value, one_arg("lt", "b"));
    case TI_le: return py_compare(1, value, one_arg("le", "b"));
    case TI_gt: return py_compare(2, value, one_arg("gt", "b"));
    case TI_ge: return py_compare(3, value, one_arg("ge", "b"));
  }
  return false;
}

// ================================================================ filter helpers
namespace {

// jinja2 make_attrgetter: dotted path; digit parts index as ints.
struct AttrGetter {
  std::vector<Val> parts;
  Val dflt;  // Missing: no default
  bool lower = false;
  AttrGetter(const Val& attribute, Val default_value = Val::missing(), bool lowercase = false)
      : dflt(std::move(default_value)), lower(lowercase) {
    if (attribute.is_none() || attribute.is_missing()) return;
    if (attribute.is_str()) {
      const std::string_view s = attribute.sv();
      size_t start = 0;
      for (;;) {
        const size_t dot = s.find('.', start);
        const std::string_view part = s.substr(start, dot == std::string_view::npos ? std::string_view::npos : dot - start);
        if (!part.empty() && part.find_first_not_of("0123456789") == std::string_view::npos)
          parts.push_back(Val::integer(std::stoll(std::string(part))));
        else
          parts.push_back(substr(attribute, size_t(part.data() - s.data()), part.size()));
        if (dot == std::string_view::npos) break;
        start = dot + 1;
      }
    } else {
      parts.push_back(attribute);
    }
  }
  Val operator()(Interp& in, Val item) const {
    for (const Val& p : parts) {
      item = get_item(in, item, p);
      if (!dflt.is_missing() && (item.is_undef() || item.is_missing())) item = dflt;
    }
    if (lower && item.is_str()) item = Val::str(uni::lower(item.sv()), item.markup());
    return item;
  }
};

// Python's sorted(): stable, `<` only, TypeError on unorderable pairs.
void py_sort(std::vector<size_t>& idx, const std::vector<Val>& keys, bool reverse) {
  std::stable_sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
    return reverse ? py_lt(keys[y], keys[x]) : py_lt(keys[x], keys[y]);
  });
}

Val lower_key(const Val& v, bool case_sensitive) {
  if (!case_sensitive && v.is_str()) return Val::str(uni::lower(v.sv()), v.markup());
  return v;
}

std::vector<Val> apply_select(Interp& in, const Val& value, Args& a, bool lookup_attr, bool reject) {
  std::vector<Val> out;
  if (!truthy(value)) return out;  // `if value:` in jinja2's select_or_reject
  size_t off = 0;
  Val attr;
  if (lookup_attr) {
    if (a.pos.empty()) raise(ErrorKind::FilterArgument, "Missing parameter for attribute name");
    attr = a.pos[0];
    off = 1;
  }
  AttrGetter getter(attr);
  const bool has_test = a.pos.size() > off;
  Val test_name;
  Args test_args;
  if (has_test) {
    test_name = a.pos[off];
    for (size_t i = off + 1; i < a.pos.size(); ++i) test_args.pos.push_back(a.pos[i]);
    test_args.kw = a.kw;
  }
  int tid = -1;
  if (has_test) {
    if (!test_name.is_str()) raise(ErrorKind::Runtime, "No test named " + to_repr(test_name) + ".");
    tid = test_id(test_name.sv());
    if (tid < 0) raise(ErrorKind::Runtime, "No test named " + py_repr_str(test_name.sv()) + ".");
  }
  PyIter it(value);
  Val item;
  while (it.next(item)) {
    const Val x = lookup_attr ? getter(in, item) : item;
    bool ok;
    if (has_test) {
      Args ta = test_args;
      ok = call_test(in, tid, std::string(test_name.sv()), x, ta);
    } else {
      ok = truthy(x);
    }
    if (ok != reject) out.push_back(item);
  }
  return out;
}

double to_float_or(const Val& v, bool* ok) {
  *ok = true;
  if (v.is_number()) return v.f();
  if (v.is_str()) {
    std::string s(strip_ws(v.sv(), true, true));
    std::string t;
    for (size_t i = 0; i < s.size(); ++i) {
      if (s[i] == '_') {  // underscores only between digits
        if (i == 0 || i + 1 >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i - 1])) ||
            !std::isdigit(static_cast<unsigned char>(s[i + 1]))) {
          *ok = false;
          return 0;
        }
        continue;
      }
      t.push_back(s[i]);
    }
    std::string low;
    for (char c : t) low.push_back(char(std::tolower(static_cast<unsigned char>(c))));
    const std::string body = (!low.empty() && (low[0] == '+' || low[0] == '-')) ? low.substr(1) : low;
    const bool neg = !low.empty() && low[0] == '-';
    if (body == "inf" || body == "infinity") return neg ? -HUGE_VAL : HUGE_VAL;
    if (body == "nan") return std::nan("");
    if (body.empty() || body.find_first_not_of("0123456789.e+-") != std::string::npos || body[0] == 'e') {
      *ok = false;
      return 0;
    }
    char* end = nullptr;
    const double d = std::strtod(t.c_str(), &end);
    if (!end || *end != '\0') {
      *ok = false;
      return 0;
    }
    return d;
  }
  *ok = false;
  return 0;
}

// int(str, base) with Python's syntax (whitespace, sign, underscores, prefixes).
bool parse_int(std::string_view s, int base, int64_t* out) {
  std::string t(strip_ws(s, true, true));
  bool neg = false;
  size_t i = 0;
  if (i < t.size() && (t[i] == '+' || t[i] == '-')) neg = t[i++] == '-';
  if (base == 0 || base == 16 || base == 8 || base == 2) {
    if (i + 1 < t.size() && t[i] == '0') {
      const char p = char(std::tolower(static_cast<unsigned char>(t[i + 1])));
      const int pb = p == 'x' ? 16 : p == 'o' ? 8 : p == 'b' ? 2 : 0;
      if (pb && (base == 0 || base == pb)) {
        base = pb;
        i += 2;
        if (i < t.size() && t[i] == '_') ++i;
      }
    }
    if (base == 0) {
      base = 10;
      if (i + 1 < t.size() && t[i] == '0') {  // "0..." only valid if all zeros
        for (size_t k = i; k < t.size(); ++k)
          if (t[k] != '0' && t[k] != '_') return false;
      }
    }
  }
  if (base < 2 || base > 36 || i >= t.size()) return false;
  __int128 v = 0;
  bool digit_seen = false, last_us = false;
  for (; i < t.size(); ++i) {
    const char c = t[i];
    if (c == '_') {
      if (!digit_seen || last_us) return false;
      last_us = true;
      continue;
    }
    int d;
    if (c >= '0' && c <= '9')
      d = c - '0';
    else if (std::isalpha(static_cast<unsigned char>(c)))
      d = std::tolower(static_cast<unsigned char>(c)) - 'a' + 10;
    else
      return false;
    if (d >= base) return false;
    v = v * base + d;
    if (v > (__int128(1) << 63)) raise(ErrorKind::Overflow, "integer literal too large");
    digit_seen = true;
    last_us = false;
  }
  if (!digit_seen || last_us) return false;
  if (neg) v = -v;
  if (v > INT64_MAX || v < INT64_MIN) raise(ErrorKind::Overflow, "integer literal too large");
  *out = int64_t(v);
  return true;
}

// Python round(x, ndigits) for floats: correctly rounded decimal.
double py_round(double x, int64_t nd) {
  if (!std::isfinite(x) || x == 0) return x;
  if (nd > 320) return x;
  if (nd < -320) return std::copysign(0.0, x);
  if (nd >= 0) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "%.*f", int(nd), x);
    return std::strtod(buf, nullptr);
  }
  const double p = std::pow(10.0, double(-nd));
  const double y = x / p;
  double z = std::nearbyint(y);  // round half to even
  if (std::fabs(y - std::trunc(y)) == 0.5) z = 2.0 * std::nearbyint(y / 2.0);
  return z * p;
}

Val to_json(const Val& value, Args& a) {
  Val v[4] = {Val::boolean(false), Val::none(), Val::none(), Val::boolean(false)};
  bind("tojson", a, {"ensure_ascii", "indent", "separators", "sort_keys"}, v);
  JsonOpts o;
  o.ensure_ascii = truthy(v[0]);
  if (!v[1].is_none()) {
    o.indent_set = true;
    if (v[1].is_intlike())
      o.indent = std::string(size_t(std::max<int64_t>(0, v[1].i())), ' ');
    else if (v[1].is_str())
      o.indent = std::string(v[1].sv());
    else
      type_error(std::string("can't multiply sequence by non-int of type '") + tname(v[1]) + "'");
    o.item_sep = ",";
  }
  if (!v[2].is_none()) {
    std::vector<Val> seps = to_vector(v[2]);
    if (seps.size() != 2)
      value_error(seps.size() > 2 ? "too many values to unpack (expected 2)"
                                  : "not enough values to unpack (expected 2, got " + std::to_string(seps.size()) + ")");
    o.item_sep = to_str(seps[0]);
    o.key_sep = to_str(seps[1]);
  }
  o.sort_keys = truthy(v[3]);
  std::string out;
  dump_json(out, value, o);
  return Val::str(std::move(out));
}

Val do_indent(const Val& s, Args& a) {
  Val v[3] = {Val::integer(4), Val::boolean(false), Val::boolean(false)};
  bind("do_indent", a, {"width", "first", "blank"}, v);
  if (s.is_undef() || s.is_missing()) raise_undefined(s);
  if (!s.is_str()) type_error(std::string("unsupported operand type(s) for +=: '") + tname(s) + "' and 'str'");
  std::string indention;
  if (v[0].is_str())
    indention = std::string(v[0].sv());
  else if (v[0].is_intlike())
    indention = std::string(size_t(std::max<int64_t>(0, v[0].i())), ' ');
  else
    type_error(std::string("can't multiply sequence by non-int of type '") + tname(v[0]) + "'");
  const Val text = Val::str(std::string(s.sv()) + "\n", s.markup());
  std::vector<Val> lines = do_splitlines(text, false);
  std::string rv;
  if (truthy(v[2])) {
    for (size_t i = 0; i < lines.size(); ++i) {
      if (i) rv += "\n" + indention;
      rv += lines[i].sv();
    }
  } else if (!lines.empty()) {
    rv = std::string(lines[0].sv());
    if (lines.size() > 1) {
      rv += "\n";
      for (size_t i = 1; i < lines.size(); ++i) {
        if (i > 1) rv += "\n";
        if (!lines[i].sv().empty()) rv += indention;
        rv += lines[i].sv();
      }
    }
  }
  if (truthy(v[1])) rv = indention + rv;
  return Val::str(std::move(rv), s.markup());
}

std::string jinja_title(std::string_view s) {
  // re.split(r"([-\s({\[<]+)") then item[0].upper() + item[1:].lower()
  auto is_sep = [](uint32_t c) { return c == '-' || c == '(' || c == '{' || c == '[' || c == '<' || uni::is_space(c); };
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    const size_t start = i;
    size_t k = i;
    const bool sep = is_sep(uni::next(s, k));
    while (i < s.size()) {
      k = i;
      if (is_sep(uni::next(s, k)) != sep) break;
      i = k;
    }
    const std::string_view item = s.substr(start, i - start);
    if (sep) {
      out += item;
    } else {
      size_t j = 0;
      uni::append_upper(out, uni::next(item, j));
      out += uni::lower(item.substr(j));
    }
  }
  return out;
}

int64_t word_count(std::string_view s) {
  int64_t n = 0;
  bool in_word = false;
  size_t i = 0;
  while (i < s.size()) {
    const bool w = uni::is_word(uni::next(s, i));
    if (w && !in_word) ++n;
    in_word = w;
  }
  return n;
}

Val first_item(const Val& seq, const char* hint) {
  PyIter it(seq);
  Val x;
  if (it.next(x)) return x;
  return Val::undef_hint(hint);
}

std::vector<Val> reversed_items(const Val& v) {
  std::vector<Val> items;
  switch (v.kind()) {
    case Kind::List:
    case Kind::Tuple:
    case Kind::Dict:
    case Kind::Range:
    case Kind::View:
    case Kind::Str:
    case Kind::Missing:
    case Kind::Undefined: items = to_vector(v); break;
    default: type_error(std::string("'") + tname(v) + "' object is not reversible");
  }
  std::reverse(items.begin(), items.end());
  return items;
}

}  // namespace

Val call_filter(Interp& in, int id, const std::string& name, const Val& value, Args& a) {
  if (id < 0) raise(ErrorKind::Runtime, "No filter named " + py_repr_str(name) + " found.");
  Val v[5];
  auto args = [&](const char* fn, std::initializer_list<const char*> names, std::initializer_list<Val> defaults) {
    size_t i = 0;
    for (const Val& d : defaults) v[i++] = d;
    bind(fn, a, names, v);
  };
  const Val M = Val::missing();
  switch (id) {
    case FI_abs:
      args("abs", {}, {});
      if (value.is_intlike()) {
        if (value.i() == INT64_MIN) raise(ErrorKind::Overflow, "integer overflow");
        return Val::integer(value.i() < 0 ? -value.i() : value.i());
      }
      if (value.is_float()) return Val::real(std::fabs(value.f()));
      type_error(std::string("bad operand type for abs(): '") + tname(value) + "'");
    case FI_attr: {
      args("do_attr", {"name"}, {M});
      if (value.is_undef() || value.is_missing()) raise_undefined(value);
      if (!v[0].is_str()) type_error("attribute name must be string");
      Val out;
      if (py_getattr(value, v[0], out)) return out;
      return Val::undef_attr(type_of(value), v[0]);
    }
    case FI_batch: {
      args("do_batch", {"linecount", "fill_with"}, {M, Val::none()});
      std::vector<Val> batches, tmp;
      PyIter it(value);
      Val x;
      while (it.next(x)) {
        if (py_eq(Val::integer(int64_t(tmp.size())), v[0])) {
          batches.push_back(make_list(std::move(tmp)));
          tmp.clear();
        }
        tmp.push_back(x);
      }
      if (!tmp.empty()) {
        if (!v[1].is_none()) {
          const int64_t lc = need_int(v[0], nullptr);
          while (int64_t(tmp.size()) < lc) tmp.push_back(v[1]);
        }
        batches.push_back(make_list(std::move(tmp)));
      }
      return make_iter(std::move(batches), "do_batch");
    }
    case FI_capitalize: {
      args("do_capitalize", {}, {});
      const Val s = soft_str(value);
      return mk(s, uni::capitalize(s.sv()));
    }
    case FI_center: {
      args("do_center", {"width"}, {Val::integer(80)});
      const Val s = soft_str(value);
      return mk(s, pad(s.sv(), need_int(v[0], nullptr), " ", 2));
    }
    case FI_count:
    case FI_length: args("len", {}, {}); return Val::integer(py_len(value));
    case FI_d:
    case FI_default:
      args("do_default", {"default_value", "boolean"}, {Val::str_ref(""), Val::boolean(false)});
      if (value.is_undef() || value.is_missing() || (truthy(v[1]) && !truthy(value))) return v[0];
      return value;
    case FI_dictsort: {
      args("do_dictsort", {"case_sensitive", "by", "reverse"}, {Val::boolean(false), Val::str_ref("key"), Val::boolean(false)});
      int pos;
      if (v[1].is_str() && v[1].sv() == "key")
        pos = 0;
      else if (v[1].is_str() && v[1].sv() == "value")
        pos = 1;
      else
        raise(ErrorKind::FilterArgument, "You can only sort by either \"key\" or \"value\"");
      if (value.is_undef() || value.is_missing()) raise_undefined(value);
      if (!value.is_dict())
        raise(ErrorKind::Attribute, std::string("'") + tname(value) + "' object has no attribute 'items'");
      const size_t n = dict_size(value);
      std::vector<Val> keys;
      std::vector<size_t> idx(n);
      for (size_t i = 0; i < n; ++i) {
        idx[i] = i;
        keys.push_back(lower_key(pos == 0 ? dict_key(value, i) : dict_value(value, i), truthy(v[0])));
      }
      py_sort(idx, keys, truthy(v[2]));
      std::vector<Val> out;
      out.reserve(n);
      for (size_t i : idx) out.push_back(make_tuple({dict_key(value, i), dict_value(value, i)}));
      return make_list(std::move(out));
    }
    case FI_e:
    case FI_escape:
      args("escape", {}, {});
      if (is_markup(value)) return value;
      return Val::str(html_escape(to_str(value)), true);
    case FI_forceescape: args("do_forceescape", {}, {}); return Val::str(html_escape(to_str(value)), true);
    case FI_filesizeformat: {
      args("do_filesizeformat", {"binary"}, {Val::boolean(false)});
      bool ok;
      const double bytes = to_float_or(value, &ok);
      if (!ok) value_error("could not convert string to float: " + to_repr(value));
      const bool binary = truthy(v[0]);
      const double base = binary ? 1024 : 1000;
      static const char* dec[] = {"kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"};
      static const char* bin[] = {"KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"};
      char buf[64];
      if (bytes == 1) return Val::str_ref("1 Byte");
      if (bytes < base) {
        std::snprintf(buf, sizeof buf, "%lld Bytes", static_cast<long long>(bytes));
        return Val::str(buf);
      }
      double unit = base;
      int i = 0;
      for (; i < 8; ++i) {
        unit = std::pow(base, i + 2);
        if (bytes < unit) break;
      }
      if (i == 8) i = 7;
      std::snprintf(buf, sizeof buf, "%.1f %s", base * bytes / unit, binary ? bin[i] : dec[i]);
      return Val::str(buf);
    }
    case FI_first:
      args("do_first", {}, {});
      if (value.is_undef() || value.is_missing()) return Val::undef_hint("No first item, sequence was empty.");
      return first_item(value, "No first item, sequence was empty.");
    case FI_last: {
      args("do_last", {}, {});
      std::vector<Val> r = reversed_items(value);
      if (r.empty()) return Val::undef_hint("No last item, sequence was empty.");
      return r[0];
    }
    case FI_float: {
      args("do_float", {"default"}, {Val::real(0.0)});
      if (value.is_undef() || value.is_missing()) raise_undefined(value);
      bool ok;
      const double d = to_float_or(value, &ok);
      return ok ? Val::real(d) : v[0];
    }
    case FI_int: {
      args("do_int", {"default", "base"}, {Val::integer(0), Val::integer(10)});
      if (value.is_undef() || value.is_missing()) raise_undefined(value);
      if (value.is_intlike()) return Val::integer(value.i());
      if (value.is_float()) {
        const double d = value.f();
        if (std::isinf(d)) raise(ErrorKind::Overflow, "cannot convert float infinity to integer");
        if (std::isnan(d)) return v[0];
        return Val::integer(int64_t(std::trunc(d)));
      }
      if (value.is_str()) {
        int64_t r;
        if (parse_int(value.sv(), int(need_int(v[1], nullptr)), &r)) return Val::integer(r);
        bool ok;
        const double d = to_float_or(value, &ok);
        if (!ok || !std::isfinite(d)) return v[0];
        return Val::integer(int64_t(std::trunc(d)));
      }
      return v[0];
    }
    case FI_format: {
      if (!a.pos.empty() && !a.kw.empty())
        raise(ErrorKind::FilterArgument, "can't handle positional and keyword arguments at the same time");
      const Val fmt = soft_str(value);
      if (!a.kw.empty()) {
        MapData m;
        for (auto& [k, x] : a.kw) m.set(k, x);
        return percent_format(fmt, make_dict(std::move(m)));
      }
      return percent_format(fmt, make_tuple(a.pos));
    }
    case FI_groupby: {
      args("do_groupby", {"attribute", "default", "case_sensitive"}, {M, Val::none(), Val::boolean(false)});
      const bool cs = truthy(v[2]);
      const AttrGetter key_of(v[0], v[1].is_none() ? Val::missing() : v[1], !cs);
      const AttrGetter real_key(v[0], v[1].is_none() ? Val::missing() : v[1], false);
      std::vector<Val> items = to_vector(value), keys;
      for (const Val& it : items) keys.push_back(key_of(in, it));
      std::vector<size_t> idx(items.size());
      for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
      py_sort(idx, keys, false);
      std::vector<Val> groups;
      for (size_t i = 0; i < idx.size();) {
        size_t j = i;
        std::vector<Val> members;
        while (j < idx.size() && py_eq(keys[idx[j]], keys[idx[i]])) members.push_back(items[idx[j++]]);
        const Val grouper = cs ? keys[idx[i]] : real_key(in, members[0]);
        groups.push_back(make_tuple({grouper, make_list(std::move(members))}));
        i = j;
      }
      return make_list(std::move(groups));
    }
    case FI_indent: return do_indent(value, a);
    case FI_join: {
      args("sync_do_join", {"d", "attribute"}, {Val::str_ref(""), Val::none()});
      const AttrGetter getter(v[1]);
      const std::string sep = to_str(v[0]);
      std::string out;
      PyIter it(value);
      Val x;
      bool first = true;
      while (it.next(x)) {
        if (!first) out += sep;
        first = false;
        if (!v[1].is_none()) x = getter(in, x);
        append_str(out, x);
      }
      return Val::str(std::move(out));
    }
    case FI_list: args("do_list", {}, {}); return make_list(to_vector(value));
    case FI_lower: {
      args("do_lower", {}, {});
      const Val s = soft_str(value);
      return mk(s, uni::lower(s.sv()));
    }
    case FI_upper: {
      args("do_upper", {}, {});
      const Val s = soft_str(value);
      return mk(s, uni::upper(s.sv()));
    }
    case FI_items: {
      args("do_items", {}, {});
      if (value.is_undef() || value.is_missing()) return make_iter({}, "do_items");
      if (!value.is_dict()) type_error("Can only get item pairs from a mapping.");
      std::vector<Val> out;
      for (size_t i = 0, n = dict_size(value); i < n; ++i) out.push_back(make_tuple({dict_key(value, i), dict_value(value, i)}));
      return make_iter(std::move(out), "do_items");
    }
    case FI_map: {
      std::vector<Val> out;
      if (!truthy(value)) return make_iter({}, "sync_do_map");
      if (a.pos.empty() && a.has("attribute")) {
        Val attribute, dflt = Val::none();
        a.take("attribute", attribute);
        a.take("default", dflt);
        if (!a.kw.empty())
          raise(ErrorKind::FilterArgument, "Unexpected keyword argument " + py_repr_str(a.kw[0].first.sv()));
        const AttrGetter getter(attribute, dflt.is_none() ? Val::missing() : dflt);
        PyIter it(value);
        Val x;
        while (it.next(x)) out.push_back(getter(in, x));
        return make_iter(std::move(out), "sync_do_map");
      }
      if (a.pos.empty()) raise(ErrorKind::FilterArgument, "map requires a filter argument");
      const Val fname = a.pos[0];
      Args rest;
      rest.pos.assign(a.pos.begin() + 1, a.pos.end());
      rest.kw = a.kw;
      const int fid = fname.is_str() ? filter_id(fname.sv()) : -1;
      if (fid < 0) raise(ErrorKind::Runtime, "No filter named " + to_repr(fname) + ".");
      PyIter it(value);
      Val x;
      while (it.next(x)) {
        Args fa = rest;
        out.push_back(call_filter(in, fid, std::string(fname.sv()), x, fa));
      }
      return make_iter(std::move(out), "sync_do_map");
    }
    case FI_min:
    case FI_max: {
      args(id == FI_min ? "do_min" : "do_max", {"case_sensitive", "attribute"}, {Val::boolean(false), Val::none()});
      std::vector<Val> items = to_vector(value);
      if (items.empty()) return Val::undef_hint("No aggregated item, sequence was empty.");
      const AttrGetter key_of(v[1], Val::missing(), !truthy(v[0]));
      size_t best = 0;
      Val best_key = key_of(in, items[0]);
      for (size_t i = 1; i < items.size(); ++i) {
        Val k = key_of(in, items[i]);
        if (id == FI_min ? py_compare(0, k, best_key) : py_compare(2, k, best_key)) {
          best = i;
          best_key = std::move(k);
        }
      }
      return items[best];
    }
    case FI_reject:
    case FI_select: return make_iter(apply_select(in, value, a, false, id == FI_reject), "select_or_reject");
    case FI_rejectattr:
    case FI_selectattr: return make_iter(apply_select(in, value, a, true, id == FI_rejectattr), "select_or_reject");
    case FI_replace: {
      args("do_replace", {"old", "new", "count"}, {M, M, Val::none()});
      const std::string old = to_str(v[0]), repl = to_str(v[1]);
      const Val s = Val::str(to_str(value));
      return do_replace(s, old, repl, v[2].is_none() ? -1 : need_int(v[2], nullptr), false);
    }
    case FI_reverse: {
      args("do_reverse", {}, {});
      if (value.is_str()) {
        const std::string_view s = value.sv();
        std::string out;
        out.reserve(s.size());
        size_t e = s.size();
        while (e > 0) {
          size_t k = e - 1;
          while (k > 0 && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80) --k;
          out.append(s.substr(k, e - k));
          e = k;
        }
        return Val::str(std::move(out), value.markup());
      }
      if (value.is(Kind::Iter)) {
        std::vector<Val> items = to_vector(value);
        std::reverse(items.begin(), items.end());
        return make_list(std::move(items));
      }
      if (value.is_undef() || value.is_missing() || value.is_seq() || value.is_dict() || value.is(Kind::Range) ||
          value.is(Kind::View))
        return make_iter(reversed_items(value), "reversed");
      raise(ErrorKind::FilterArgument, "argument must be iterable");
    }
    case FI_round: {
      args("do_round", {"precision", "method"}, {Val::integer(0), Val::str_ref("common")});
      const std::string_view method = v[1].is_str() ? v[1].sv() : "";
      if (method != "common" && method != "ceil" && method != "floor")
        raise(ErrorKind::FilterArgument, "method must be common, ceil or floor");
      if (value.is_undef() || value.is_missing()) type_error("type Undefined doesn't define __round__ method");
      if (!value.is_number()) type_error(std::string("type ") + tname(value) + " doesn't define __round__ method");
      const int64_t prec = need_int(v[0], nullptr);
      if (method == "common") {
        if (value.is_intlike()) {
          if (prec >= 0) return Val::integer(value.i());
          const double r = py_round(double(value.i()), prec);
          return Val::integer(int64_t(r));
        }
        return Val::real(py_round(value.f(), prec));
      }
      const double scale = std::pow(10.0, double(prec));
      const double r = method == "ceil" ? std::ceil(value.f() * scale) : std::floor(value.f() * scale);
      return Val::real(r / scale);
    }
    case FI_safe:
      args("do_mark_safe", {}, {});
      if (is_markup(value)) return value;
      return Val::str(to_str(value), true);
    case FI_slice: {
      args("sync_do_slice", {"slices", "fill_with"}, {M, Val::none()});
      std::vector<Val> seq = to_vector(value);
      const int64_t slices = need_int(v[0], nullptr);
      if (slices == 0) raise(ErrorKind::ZeroDivision, "integer division or modulo by zero");
      const int64_t length = int64_t(seq.size());
      const int64_t per = length / slices, extra = length % slices;
      int64_t offset = 0;
      std::vector<Val> out;
      for (int64_t sn = 0; sn < slices; ++sn) {
        const int64_t start = offset + sn * per;
        if (sn < extra) ++offset;
        const int64_t end = offset + (sn + 1) * per;
        std::vector<Val> tmp;
        for (int64_t i = std::max<int64_t>(start, 0); i < std::min(end, length); ++i) tmp.push_back(seq[size_t(i)]);
        if (!v[1].is_none() && sn >= extra) tmp.push_back(v[1]);
        out.push_back(make_list(std::move(tmp)));
      }
      return make_iter(std::move(out), "sync_do_slice");
    }
    case FI_sort: {
      args("do_sort", {"reverse", "case_sensitive", "attribute"}, {Val::boolean(false), Val::boolean(false), Val::none()});
      std::vector<Val> items = to_vector(value);
      std::vector<Val> keys;
      const bool cs = truthy(v[1]);
      if (v[2].is_str() && v[2].sv().find(',') != std::string_view::npos) {
        // "a,b": compare lists of attribute values
        std::vector<AttrGetter> getters;
        const std::string_view s = v[2].sv();
        size_t start = 0;
        for (;;) {
          const size_t c = s.find(',', start);
          getters.emplace_back(Val::str(std::string(s.substr(start, c == std::string_view::npos ? std::string_view::npos : c - start))),
                               Val::missing(), !cs);
          if (c == std::string_view::npos) break;
          start = c + 1;
        }
        for (const Val& it : items) {
          std::vector<Val> ks;
          for (const auto& g : getters) ks.push_back(g(in, it));
          keys.push_back(make_list(std::move(ks)));
        }
      } else {
        const AttrGetter g(v[2], Val::missing(), !cs);
        for (const Val& it : items) keys.push_back(g(in, it));
      }
      std::vector<size_t> idx(items.size());
      for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
      py_sort(idx, keys, truthy(v[0]));
      std::vector<Val> out;
      for (size_t i : idx) out.push_back(items[i]);
      return make_list(std::move(out));
    }
    case FI_string: args("soft_str", {}, {}); return soft_str(value);
    case FI_striptags: args("do_striptags", {}, {}); return Val::str(strip_tags(to_str(value)));
    case FI_sum: {
      args("sync_do_sum", {"attribute", "start"}, {Val::none(), Val::integer(0)});
      if (v[1].is_str()) type_error("sum() can't sum strings [use ''.join(seq) instead]");
      const AttrGetter g(v[0]);
      Val acc = v[1];
      PyIter it(value);
      Val x;
      while (it.next(x)) acc = py_add(acc, v[0].is_none() ? x : g(in, x));
      return acc;
    }
    case FI_title: {
      args("do_title", {}, {});
      const Val s = soft_str(value);
      return mk(s, jinja_title(s.sv()));
    }
    case FI_trim: {
      args("do_trim", {"chars"}, {Val::none()});
      return do_strip(soft_str(value), v[0], true, true, "strip");
    }
    case FI_truncate: {
      args("do_truncate", {"length", "killwords", "end", "leeway"}, {Val::integer(255), Val::boolean(false), Val::str_ref("..."), Val::none()});
      const int64_t length = need_int(v[0], nullptr);
      const int64_t leeway = v[3].is_none() ? 5 : need_int(v[3], nullptr);
      const std::string end = to_str(v[2]);
      const int64_t end_len = int64_t(uni::length(end));
      if (length < end_len) raise(ErrorKind::Runtime, "expected length >= " + std::to_string(end_len) + ", got " + std::to_string(length));
      const Val s = soft_str(value);
      const std::string_view sv = s.sv();
      if (int64_t(uni::length(sv)) <= length + leeway) return s;
      const std::string_view head = uni::slice(sv, 0, size_t(length - end_len));
      if (truthy(v[1])) return Val::str(std::string(head) + end);
      const size_t sp = head.rfind(' ');
      return Val::str(std::string(sp == std::string_view::npos ? head : head.substr(0, sp)) + end);
    }
    case FI_unique: {
      args("do_unique", {"case_sensitive", "attribute"}, {Val::boolean(false), Val::none()});
      const AttrGetter g(v[1], Val::missing(), !truthy(v[0]));
      std::vector<Val> seen, out;
      PyIter it(value);
      Val x;
      while (it.next(x)) {
        const Val k = g(in, x);
        require_hashable(k);
        bool dup = false;
        for (const Val& s : seen)
          if (py_eq(s, k)) {
            dup = true;
            break;
          }
        if (!dup) {
          seen.push_back(k);
          out.push_back(x);
        }
      }
      return make_iter(std::move(out), "sync_do_unique");
    }
    case FI_wordcount: args("do_wordcount", {}, {}); return Val::integer(word_count(soft_str(value).sv()));
    case FI_tojson: return to_json(value, a);
    default: break;
  }
  raise(ErrorKind::Unsupported, "the '" + name + "' filter is not supported");
}

}  // namespace espetero::tmpl
