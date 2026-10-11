// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "value.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "unicode.h"

namespace espetero::tmpl {

// Implemented in interp.cpp: the parts of the protocol that need the loop and
// macro objects.
int64_t loop_length(Obj* loop);
int64_t loop_index(Obj* loop);
std::string_view macro_name(Obj* macro);
const char* method_name(uint16_t id);

// ---------------------------------------------------------------- basics
Val Val::str(std::string s, bool markup) {
  Val v = object(Kind::Str, new StrObj(std::move(s)));
  if (markup) v.flags_ |= kMarkup;
  return v;
}

const char* py_type_name(PyType t) {
  switch (t) {
    case PyType::NoneType: return "NoneType";
    case PyType::Bool: return "bool";
    case PyType::Int: return "int";
    case PyType::Float: return "float";
    case PyType::Str: return "str";
    case PyType::Markup: return "Markup";
    case PyType::List: return "list";
    case PyType::Tuple: return "tuple";
    case PyType::Dict: return "dict";
    case PyType::Namespace: return "Namespace";
    case PyType::Macro: return "Macro";
    case PyType::LoopContext: return "LoopContext";
    case PyType::Generator: return "generator";
    case PyType::Range: return "range";
    case PyType::DictKeys: return "dict_keys";
    case PyType::DictValues: return "dict_values";
    case PyType::DictItems: return "dict_items";
    case PyType::Function: return "function";
    case PyType::BuiltinMethod: return "builtin_function_or_method";
    case PyType::Type: return "type";
    case PyType::Undefined: return "Undefined";
    case PyType::Cycler: return "Cycler";
    case PyType::Joiner: return "Joiner";
    case PyType::TemplateReference: return "TemplateReference";
  }
  return "object";
}

std::string object_type_repr(PyType t) {
  switch (t) {
    case PyType::NoneType: return "None";
    case PyType::Markup: return "markupsafe.Markup object";
    case PyType::Namespace: return "jinja2.utils.Namespace object";
    case PyType::Macro: return "jinja2.runtime.Macro object";
    case PyType::LoopContext: return "jinja2.runtime.LoopContext object";
    case PyType::Undefined: return "jinja2.runtime.Undefined object";
    case PyType::Cycler: return "jinja2.utils.Cycler object";
    case PyType::Joiner: return "jinja2.utils.Joiner object";
    case PyType::TemplateReference: return "jinja2.runtime.TemplateReference object";
    default: return std::string(py_type_name(t)) + " object";
  }
}

void raise(ErrorKind k, const std::string& msg) { throw TemplateError(k, msg); }

std::string py_repr_str(std::string_view s) {
  // CPython unicode_repr: prefer single quotes unless the text has single
  // quotes and no double quotes.
  const bool has_sq = s.find('\'') != std::string_view::npos;
  const bool has_dq = s.find('"') != std::string_view::npos;
  const char quote = (has_sq && !has_dq) ? '"' : '\'';
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back(quote);
  static const char* hex = "0123456789abcdef";
  size_t i = 0;
  while (i < s.size()) {
    const size_t start = i;
    const uint32_t c = uni::next(s, i);
    if (c == uint32_t(quote) || c == '\\') {
      out.push_back('\\');
      out.push_back(char(c));
    } else if (c == '\t') {
      out += "\\t";
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\r') {
      out += "\\r";
    } else if (c < 0x20 || c == 0x7F) {
      out += "\\x";
      out.push_back(hex[c >> 4]);
      out.push_back(hex[c & 15]);
    } else if (c < 0x7F || uni::is_printable(c)) {
      out.append(s.data() + start, i - start);
    } else if (c <= 0xFF) {
      out += "\\x";
      out.push_back(hex[c >> 4]);
      out.push_back(hex[c & 15]);
    } else if (c <= 0xFFFF) {
      out += "\\u";
      for (int k = 12; k >= 0; k -= 4) out.push_back(hex[(c >> k) & 15]);
    } else {
      out += "\\U";
      for (int k = 28; k >= 0; k -= 4) out.push_back(hex[(c >> k) & 15]);
    }
  }
  out.push_back(quote);
  return out;
}

std::string undefined_message(const Val& u) {
  switch (u.undef_kind()) {
    case UndefKind::Name: return py_repr_str(u.sv()) + " is undefined";
    case UndefKind::Attr:
      return py_repr_str(object_type_repr(u.undef_type())) + " has no attribute " + py_repr_str(u.sv());
    case UndefKind::ElemInt:
      return object_type_repr(u.undef_type()) + " has no element " + std::to_string(u.undef_key());
    case UndefKind::Hint: return std::string(u.sv());
    case UndefKind::Unsafe:
      return "access to attribute " + py_repr_str(u.sv()) + " of " + py_repr_str(py_type_name(u.undef_type())) +
             " object is unsafe.";
  }
  return "undefined";
}

void raise_undefined(const Val& u) {
  if (u.is_missing()) raise(ErrorKind::Undefined, "None is undefined");
  raise(u.undef_kind() == UndefKind::Unsafe ? ErrorKind::Security : ErrorKind::Undefined, undefined_message(u));
}

// ---------------------------------------------------------------- conversions
Val from_json(const json::Value& v) {
  switch (v.type()) {
    case json::Type::Null: return Val::none();
    case json::Type::Bool: return Val::boolean(v.as_bool());
    case json::Type::Int: return Val::integer(v.as_int());
    case json::Type::Double: return Val::real(v.as_double());
    case json::Type::String: return Val::str_ref(v.as_string());
    case json::Type::Array: return Val::json_list(&v);
    case json::Type::Object: return Val::json_dict(&v);
  }
  return Val::none();
}

PyType type_of(const Val& v) {
  switch (v.kind()) {
    case Kind::Missing:
    case Kind::Undefined: return PyType::Undefined;
    case Kind::None: return PyType::NoneType;
    case Kind::Bool: return PyType::Bool;
    case Kind::Int: return PyType::Int;
    case Kind::Float: return PyType::Float;
    case Kind::Str: return v.markup() ? PyType::Markup : PyType::Str;
    case Kind::List: return PyType::List;
    case Kind::Tuple: return PyType::Tuple;
    case Kind::Dict: return PyType::Dict;
    case Kind::Namespace: return PyType::Namespace;
    case Kind::Macro: return PyType::Macro;
    case Kind::Builtin:
      switch (v.builtin_id()) {
        case BuiltinId::Dict:
        case BuiltinId::Namespace:
        case BuiltinId::Cycler:
        case BuiltinId::Joiner: return PyType::Type;
        case BuiltinId::TemplateReference: return PyType::TemplateReference;
        default: return PyType::Function;
      }
    case Kind::Method: return PyType::BuiltinMethod;
    case Kind::Loop: return PyType::LoopContext;
    case Kind::Iter: return PyType::Generator;
    case Kind::Range: return PyType::Range;
    case Kind::View:
      switch (v.as<ViewObj>()->which) {
        case ViewKind::Keys: return PyType::DictKeys;
        case ViewKind::Values: return PyType::DictValues;
        case ViewKind::Items: return PyType::DictItems;
      }
      return PyType::DictKeys;
    case Kind::Cycler: return PyType::Cycler;
    case Kind::Joiner: return PyType::Joiner;
  }
  return PyType::NoneType;
}

Val substr(const Val& s, size_t pos, size_t n) {
  std::string_view v = s.sv().substr(pos, n);
  if (v.size() == s.sv().size()) return s;  // e.g. strip() removing nothing
  if (s.borrowed()) return Val::str_ref(v, s.markup());
  return Val::str(std::string(v), s.markup());
}

// ---------------------------------------------------------------- containers
int64_t RangeObj::size() const {
  if (step > 0) return start < stop ? (stop - start - 1) / step + 1 : 0;
  return start > stop ? (start - stop - 1) / (-step) + 1 : 0;
}

size_t seq_size(const Val& v) {
  if (v.is_json()) return v.json()->size();
  return v.as<ListObj>()->v.size();
}

Val seq_at(const Val& v, size_t i) {
  if (v.is_json()) return from_json((*v.json())[i]);
  return v.as<ListObj>()->v[i];
}

size_t dict_size(const Val& v) {
  if (v.is_json()) return v.json()->size();
  return v.as<DictObj>()->m.keys.size();
}

Val dict_key(const Val& v, size_t i) {
  if (v.is_json()) return Val::str_ref(v.json()->as_object().keys[i]);
  return v.as<DictObj>()->m.keys[i];
}

Val dict_value(const Val& v, size_t i) {
  if (v.is_json()) return from_json(v.json()->as_object().vals[i]);
  return v.as<DictObj>()->m.vals[i];
}

bool dict_get_str(const Val& d, std::string_view key, Val& out) {
  if (d.is_json()) {
    const json::Value* p = d.json()->as_object().find(key);
    if (!p) return false;
    out = from_json(*p);
    return true;
  }
  const MapData& m = d.as<DictObj>()->m;
  const int k = m.find_str(key);
  if (k < 0) return false;
  out = m.vals[size_t(k)];
  return true;
}

bool dict_get(const Val& d, const Val& key, Val& out) {
  if (d.is_json()) return key.is_str() && dict_get_str(d, key.sv(), out);
  const MapData& m = d.as<DictObj>()->m;
  const int k = m.find(key);
  if (k < 0) return false;
  out = m.vals[size_t(k)];
  return true;
}

int MapData::find_str(std::string_view key) const {
  for (size_t i = 0; i < keys.size(); ++i)
    if (keys[i].is_str() && keys[i].sv() == key) return int(i);
  return -1;
}

int MapData::find(const Val& key) const {
  if (key.is_str()) return find_str(key.sv());
  for (size_t i = 0; i < keys.size(); ++i)
    if (py_eq(keys[i], key)) return int(i);
  return -1;
}

void MapData::set(Val key, Val value) {
  const int k = find(key);
  if (k >= 0) {
    vals[size_t(k)] = std::move(value);  // Python keeps the original key object and position
    return;
  }
  keys.push_back(std::move(key));
  vals.push_back(std::move(value));
}

Val make_list(std::vector<Val> v) { return Val::object(Kind::List, new ListObj(std::move(v))); }
Val make_tuple(std::vector<Val> v) { return Val::object(Kind::Tuple, new ListObj(std::move(v))); }
Val make_dict(MapData m) {
  auto* d = new DictObj;
  d->m = std::move(m);
  return Val::object(Kind::Dict, d);
}
Val make_iter(std::vector<Val> items, const char* origin) {
  return Val::object(Kind::Iter, new IterObj(std::move(items), origin));
}

// ---------------------------------------------------------------- truthiness / len
bool truthy(const Val& v) {
  switch (v.kind()) {
    case Kind::Missing:
    case Kind::Undefined:
    case Kind::None: return false;
    case Kind::Bool: return v.b();
    case Kind::Int: return v.i() != 0;
    case Kind::Float: return v.f() != 0.0;
    case Kind::Str: return !v.sv().empty();
    case Kind::List:
    case Kind::Tuple: return seq_size(v) != 0;
    case Kind::Dict: return dict_size(v) != 0;
    case Kind::Range: return v.as<RangeObj>()->size() != 0;
    case Kind::View: return dict_size(v.as<ViewObj>()->dict) != 0;
    case Kind::Loop: return loop_length(v.obj()) != 0;
    default: return true;
  }
}

bool has_len(const Val& v) {
  switch (v.kind()) {
    case Kind::Missing:
    case Kind::Undefined:
    case Kind::Str:
    case Kind::List:
    case Kind::Tuple:
    case Kind::Dict:
    case Kind::Range:
    case Kind::View:
    case Kind::Loop: return true;
    default: return false;
  }
}

int64_t py_len(const Val& v) {
  switch (v.kind()) {
    case Kind::Missing:
    case Kind::Undefined: return 0;
    case Kind::Str: return int64_t(uni::length(v.sv()));
    case Kind::List:
    case Kind::Tuple: return int64_t(seq_size(v));
    case Kind::Dict: return int64_t(dict_size(v));
    case Kind::Range: return v.as<RangeObj>()->size();
    case Kind::View: return int64_t(dict_size(v.as<ViewObj>()->dict));
    case Kind::Loop: return loop_length(v.obj());
    default: raise(ErrorKind::Type, std::string("object of type '") + tname(v) + "' has no len()");
  }
}

// ---------------------------------------------------------------- str / repr
std::string format_float(double d) {
  if (std::isnan(d)) return "nan";
  if (std::isinf(d)) return d > 0 ? "inf" : "-inf";
  return json::format_double(d);
}

namespace {

void append_int(std::string& out, int64_t i) {
  char buf[24];
  const int n = snprintf(buf, sizeof buf, "%lld", static_cast<long long>(i));
  out.append(buf, size_t(n));
}

void append_items_repr(std::string& out, const Val& v, char open, char close, bool tuple) {
  out.push_back(open);
  const size_t n = seq_size(v);
  for (size_t i = 0; i < n; ++i) {
    if (i) out += ", ";
    append_repr(out, seq_at(v, i));
  }
  if (tuple && n == 1) out.push_back(',');
  out.push_back(close);
}

void append_map_repr(std::string& out, const Val& d) {
  out.push_back('{');
  const size_t n = dict_size(d);
  for (size_t i = 0; i < n; ++i) {
    if (i) out += ", ";
    append_repr(out, dict_key(d, i));
    out += ": ";
    append_repr(out, dict_value(d, i));
  }
  out.push_back('}');
}

void append_mapdata_repr(std::string& out, const MapData& m) {
  out.push_back('{');
  for (size_t i = 0; i < m.keys.size(); ++i) {
    if (i) out += ", ";
    append_repr(out, m.keys[i]);
    out += ": ";
    append_repr(out, m.vals[i]);
  }
  out.push_back('}');
}

const char* builtin_repr(BuiltinId id) {
  switch (id) {
    case BuiltinId::Range: return "<function safe_range at 0x0>";
    case BuiltinId::Dict: return "<class 'dict'>";
    case BuiltinId::Namespace: return "<class 'jinja2.utils.Namespace'>";
    case BuiltinId::Lipsum: return "<function generate_lorem_ipsum at 0x0>";
    case BuiltinId::Cycler: return "<class 'jinja2.utils.Cycler'>";
    case BuiltinId::Joiner: return "<class 'jinja2.utils.Joiner'>";
    case BuiltinId::RaiseException:
      return "<function _cached_compile_jinja_template.<locals>.raise_exception at 0x0>";
    case BuiltinId::StrftimeNow: return "<function _cached_compile_jinja_template.<locals>.strftime_now at 0x0>";
    case BuiltinId::Generation: return "<function _generation_support at 0x0>";
    case BuiltinId::TemplateReference: return "<TemplateReference None>";
  }
  return "<function>";
}

void append_obj_str(std::string& out, const Val& v) {
  switch (v.kind()) {
    case Kind::Namespace:
      out += "<Namespace ";
      append_mapdata_repr(out, v.as<NsObj>()->m);
      out += ">";
      return;
    case Kind::Macro:
      out += "<Macro ";
      out += py_repr_str(macro_name(v.obj()));
      out += ">";
      return;
    case Kind::Builtin: out += builtin_repr(v.builtin_id()); return;
    case Kind::Method: {
      const auto* m = v.as<MethodObj>();
      out += "<built-in method ";
      out += method_name(m->id);
      out += " of ";
      out += py_type_name(type_of(m->self));
      out += " object at 0x0>";
      return;
    }
    case Kind::Loop:
      out += "<LoopContext ";
      append_int(out, loop_index(v.obj()));
      out += "/";
      append_int(out, loop_length(v.obj()));
      out += ">";
      return;
    case Kind::Iter:
      out += "<generator object ";
      out += v.as<IterObj>()->origin;
      out += " at 0x0>";
      return;
    case Kind::Range: {
      const auto* r = v.as<RangeObj>();
      out += "range(";
      append_int(out, r->start);
      out += ", ";
      append_int(out, r->stop);
      if (r->step != 1) {
        out += ", ";
        append_int(out, r->step);
      }
      out += ")";
      return;
    }
    case Kind::View: {
      const auto* vw = v.as<ViewObj>();
      out += py_type_name(type_of(v));
      out += "([";
      const size_t n = dict_size(vw->dict);
      for (size_t i = 0; i < n; ++i) {
        if (i) out += ", ";
        if (vw->which == ViewKind::Keys) {
          append_repr(out, dict_key(vw->dict, i));
        } else if (vw->which == ViewKind::Values) {
          append_repr(out, dict_value(vw->dict, i));
        } else {
          out += "(";
          append_repr(out, dict_key(vw->dict, i));
          out += ", ";
          append_repr(out, dict_value(vw->dict, i));
          out += ")";
        }
      }
      out += "])";
      return;
    }
    case Kind::Cycler: out += "<jinja2.utils.Cycler object at 0x0>"; return;
    case Kind::Joiner: out += "<jinja2.utils.Joiner object at 0x0>"; return;
    default: return;
  }
}

}  // namespace

void append_str(std::string& out, const Val& v) {
  switch (v.kind()) {
    case Kind::Missing:
    case Kind::Undefined: return;
    case Kind::None: out += "None"; return;
    case Kind::Bool: out += v.b() ? "True" : "False"; return;
    case Kind::Int: append_int(out, v.i()); return;
    case Kind::Float: out += format_float(v.f()); return;
    case Kind::Str: out += v.sv(); return;
    case Kind::List: append_items_repr(out, v, '[', ']', false); return;
    case Kind::Tuple: append_items_repr(out, v, '(', ')', true); return;
    case Kind::Dict: append_map_repr(out, v); return;
    default: append_obj_str(out, v); return;
  }
}

std::string to_str(const Val& v) {
  if (v.is_str()) return std::string(v.sv());
  std::string s;
  append_str(s, v);
  return s;
}

void append_repr(std::string& out, const Val& v) {
  switch (v.kind()) {
    case Kind::Missing:
    case Kind::Undefined: out += "Undefined"; return;
    case Kind::Str:
      if (v.markup()) {
        out += "Markup(";
        out += py_repr_str(v.sv());
        out += ")";
      } else {
        out += py_repr_str(v.sv());
      }
      return;
    default: append_str(out, v); return;
  }
}

std::string to_repr(const Val& v) {
  std::string s;
  append_repr(s, v);
  return s;
}

// ---------------------------------------------------------------- comparisons
namespace {

// Exact int/float equality like Python (no rounding of the int).
bool int_eq_float(int64_t i, double f) {
  if (std::isnan(f) || std::isinf(f) || f != std::floor(f)) return false;
  if (f < -9.2233720368547758e18 || f >= 9.2233720368547758e18) return false;
  return int64_t(f) == i;
}

// Three-way numeric compare; returns false in *unordered for NaN.
int num_cmp(const Val& a, const Val& b, bool* unordered) {
  *unordered = false;
  if (!a.is_float() && !b.is_float()) {
    const int64_t x = a.i(), y = b.i();
    return x < y ? -1 : (x > y ? 1 : 0);
  }
  if (a.is_float() && b.is_float()) {
    const double x = a.f(), y = b.f();
    if (std::isnan(x) || std::isnan(y)) {
      *unordered = true;
      return 0;
    }
    return x < y ? -1 : (x > y ? 1 : 0);
  }
  // int vs float: compare exactly.
  const bool a_float = a.is_float();
  const double f = a_float ? a.f() : b.f();
  const int64_t i = a_float ? b.i() : a.i();
  if (std::isnan(f)) {
    *unordered = true;
    return 0;
  }
  int c;  // sign of (f - i)
  if (f >= 9.2233720368547758e18)
    c = 1;
  else if (f < -9.2233720368547758e18)
    c = -1;
  else {
    const double fl = std::floor(f);
    const int64_t fi = int64_t(fl);
    if (fi < i)
      c = -1;
    else if (fi > i)
      c = 1;
    else
      c = f > fl ? 1 : 0;
  }
  return a_float ? c : -c;
}

bool seq_eq(const Val& a, const Val& b) {
  const size_t n = seq_size(a);
  if (n != seq_size(b)) return false;
  for (size_t i = 0; i < n; ++i)
    if (!py_eq(seq_at(a, i), seq_at(b, i))) return false;
  return true;
}

bool dict_eq(const Val& a, const Val& b) {
  const size_t n = dict_size(a);
  if (n != dict_size(b)) return false;
  Val w;
  for (size_t i = 0; i < n; ++i) {
    if (!dict_get(b, dict_key(a, i), w) || !py_eq(dict_value(a, i), w)) return false;
  }
  return true;
}

const char* cmp_symbol(int op) {
  switch (op) {
    case 0: return "<";
    case 1: return "<=";
    case 2: return ">";
    default: return ">=";
  }
}

}  // namespace

bool py_eq(const Val& a, const Val& b) {
  const Kind ka = a.kind(), kb = b.kind();
  if (ka == Kind::Undefined || kb == Kind::Undefined) return ka == kb;
  if (a.is_number() && b.is_number()) {
    if (!a.is_float() && !b.is_float()) return a.i() == b.i();
    if (a.is_float() && b.is_float()) return a.f() == b.f();
    return a.is_float() ? int_eq_float(b.i(), a.f()) : int_eq_float(a.i(), b.f());
  }
  if (ka != kb) return false;
  switch (ka) {
    case Kind::None: return true;
    case Kind::Str: return a.sv() == b.sv();
    case Kind::List:
    case Kind::Tuple: return seq_eq(a, b);
    case Kind::Dict: return dict_eq(a, b);
    case Kind::Builtin: return a.builtin_id() == b.builtin_id();
    case Kind::Range: {
      const auto *x = a.as<RangeObj>(), *y = b.as<RangeObj>();
      const int64_t n = x->size();
      if (n != y->size()) return false;
      if (n == 0) return true;
      if (x->start != y->start) return false;
      return n == 1 || x->step == y->step;
    }
    case Kind::View: {
      const auto *x = a.as<ViewObj>(), *y = b.as<ViewObj>();
      if (x->which != y->which || x->which == ViewKind::Values) return a.same_object(b);
      const size_t n = dict_size(x->dict);
      if (n != dict_size(y->dict)) return false;
      Val w;
      for (size_t i = 0; i < n; ++i) {
        const Val k = dict_key(x->dict, i);
        if (!dict_get(y->dict, k, w)) return false;
        if (x->which == ViewKind::Items && !py_eq(dict_value(x->dict, i), w)) return false;
      }
      return true;
    }
    case Kind::Method: {
      const auto *x = a.as<MethodObj>(), *y = b.as<MethodObj>();
      return x->id == y->id && (x->self.same_object(y->self) || (x->self.borrowed() && y->self.borrowed() &&
                                                                  x->self.kind() == y->self.kind() &&
                                                                  py_eq(x->self, y->self)));
    }
    default: return a.same_object(b);
  }
}

// Rich comparison op: 0 '<', 1 '<=', 2 '>', 3 '>='.
bool py_compare(int op, const Val& a, const Val& b) {
  if (a.is_undef() || a.is_missing()) raise_undefined(a);
  if (b.is_undef() || b.is_missing()) raise_undefined(b);
  if (a.is_number() && b.is_number()) {
    bool unordered;
    const int c = num_cmp(a, b, &unordered);
    if (unordered) return false;
    switch (op) {
      case 0: return c < 0;
      case 1: return c <= 0;
      case 2: return c > 0;
      default: return c >= 0;
    }
  }
  if (a.is_str() && b.is_str()) {
    const int c = a.sv().compare(b.sv());  // UTF-8 byte order == code point order
    switch (op) {
      case 0: return c < 0;
      case 1: return c <= 0;
      case 2: return c > 0;
      default: return c >= 0;
    }
  }
  if ((a.is_list() && b.is_list()) || (a.is_tuple() && b.is_tuple())) {
    const size_t na = seq_size(a), nb = seq_size(b);
    size_t i = 0;
    for (; i < na && i < nb; ++i) {
      Val x = seq_at(a, i), y = seq_at(b, i);
      if (!py_eq(x, y)) return py_compare(op, x, y);
    }
    switch (op) {
      case 0: return na < nb;
      case 1: return na <= nb;
      case 2: return na > nb;
      default: return na >= nb;
    }
  }
  raise(ErrorKind::Type, std::string("'") + cmp_symbol(op) + "' not supported between instances of '" + tname(a) +
                             "' and '" + tname(b) + "'");
}

bool py_lt(const Val& a, const Val& b) { return py_compare(0, a, b); }

bool is_hashable(const Val& v) {
  switch (v.kind()) {
    case Kind::List:
    case Kind::Dict:
    case Kind::View: return false;
    case Kind::Tuple: {
      const size_t n = seq_size(v);
      for (size_t i = 0; i < n; ++i)
        if (!is_hashable(seq_at(v, i))) return false;
      return true;
    }
    default: return true;
  }
}

void require_hashable(const Val& v) {
  if (v.is_tuple()) {
    const size_t n = seq_size(v);
    for (size_t i = 0; i < n; ++i) require_hashable(seq_at(v, i));
    return;
  }
  if (!is_hashable(v)) raise(ErrorKind::Type, std::string("unhashable type: '") + tname(v) + "'");
}

// ---------------------------------------------------------------- iteration
PyIter::PyIter(const Val& v) : src_(v) {
  switch (v.kind()) {
    case Kind::Missing:
    case Kind::Undefined: mode_ = Empty; break;
    case Kind::List:
    case Kind::Tuple:
      mode_ = Seq;
      n_ = seq_size(v);
      break;
    case Kind::Dict:
      mode_ = Dict;
      n_ = dict_size(v);
      break;
    case Kind::Str:
      mode_ = Str;
      s_ = src_.sv();
      break;
    case Kind::Range: {
      const auto* r = v.as<RangeObj>();
      mode_ = Range;
      cur_ = r->start;
      step_ = r->step;
      n_ = size_t(r->size());
      break;
    }
    case Kind::Iter: mode_ = Gen; break;
    case Kind::View: {
      mode_ = View;
      view_ = v.as<ViewObj>()->which;
      n_ = dict_size(v.as<ViewObj>()->dict);
      break;
    }
    default: raise(ErrorKind::Type, std::string("'") + tname(v) + "' object is not iterable");
  }
}

bool PyIter::next(Val& out) {
  switch (mode_) {
    case Empty: return false;
    case Seq:
      if (i_ >= n_) return false;
      out = seq_at(src_, i_++);
      return true;
    case Dict:
      if (i_ >= n_) return false;
      out = dict_key(src_, i_++);
      return true;
    case Str: {
      if (i_ >= s_.size()) return false;
      const size_t start = i_;
      uni::next(s_, i_);
      out = substr(src_, start, i_ - start);
      return true;
    }
    case Range:
      if (i_ >= n_) return false;
      out = Val::integer(cur_);
      cur_ += step_;
      ++i_;
      return true;
    case Gen: {
      auto* g = src_.as<IterObj>();
      if (g->pos >= g->items.size()) return false;
      out = g->items[g->pos++];
      return true;
    }
    case View: {
      if (i_ >= n_) return false;
      const Val& d = src_.as<ViewObj>()->dict;
      if (view_ == ViewKind::Keys)
        out = dict_key(d, i_);
      else if (view_ == ViewKind::Values)
        out = dict_value(d, i_);
      else
        out = make_tuple({dict_key(d, i_), dict_value(d, i_)});
      ++i_;
      return true;
    }
  }
  return false;
}

std::vector<Val> to_vector(const Val& iterable) {
  std::vector<Val> out;
  if (iterable.is_seq()) {
    const size_t n = seq_size(iterable);
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.push_back(seq_at(iterable, i));
    return out;
  }
  PyIter it(iterable);
  Val x;
  while (it.next(x)) out.push_back(std::move(x));
  return out;
}

// ---------------------------------------------------------------- json.dumps
namespace {

void newline_indent(std::string& out, const JsonOpts& o, int level) {
  out.push_back('\n');
  for (int k = 0; k < level; ++k) out += o.indent;
}

void dump_float(std::string& out, double d) {
  if (std::isnan(d))
    out += "NaN";
  else if (std::isinf(d))
    out += d > 0 ? "Infinity" : "-Infinity";
  else
    out += json::format_double(d);
}

[[noreturn]] void not_serializable(const Val& v) {
  raise(ErrorKind::Type, std::string("Object of type ") + tname(v) + " is not JSON serializable");
}

// json.dumps converts these key types to strings; others are an error.
std::string json_key(const Val& k) {
  switch (k.kind()) {
    case Kind::Str: return std::string(k.sv());
    case Kind::Bool: return k.b() ? "true" : "false";
    case Kind::None: return "null";
    case Kind::Int: return std::to_string(k.i());
    case Kind::Float: {
      std::string s;
      dump_float(s, k.f());
      return s;
    }
    default:
      raise(ErrorKind::Type, std::string("keys must be str, int, float, bool or None, not ") + tname(k));
  }
}

void dump_rec(std::string& out, const Val& v, const JsonOpts& o, int level) {
  switch (v.kind()) {
    case Kind::None: out += "null"; return;
    case Kind::Bool: out += v.b() ? "true" : "false"; return;
    case Kind::Int: append_int(out, v.i()); return;
    case Kind::Float: dump_float(out, v.f()); return;
    case Kind::Str: json::append_quoted(out, v.sv(), o.ensure_ascii); return;
    case Kind::List:
    case Kind::Tuple: {
      const size_t n = seq_size(v);
      if (n == 0) {
        out += "[]";
        return;
      }
      out.push_back('[');
      for (size_t i = 0; i < n; ++i) {
        if (i) out += o.item_sep;
        if (o.indent_set) newline_indent(out, o, level + 1);
        dump_rec(out, seq_at(v, i), o, level + 1);
      }
      if (o.indent_set) newline_indent(out, o, level);
      out.push_back(']');
      return;
    }
    case Kind::Dict: {
      const size_t n = dict_size(v);
      if (n == 0) {
        out += "{}";
        return;
      }
      std::vector<std::pair<std::string, size_t>> keys;
      keys.reserve(n);
      for (size_t i = 0; i < n; ++i) keys.emplace_back(json_key(dict_key(v, i)), i);
      if (o.sort_keys) {
        // sorted(dct.items()) compares the original keys.
        std::vector<Val> orig;
        for (size_t i = 0; i < n; ++i) orig.push_back(dict_key(v, i));
        std::stable_sort(keys.begin(), keys.end(), [&](const auto& x, const auto& y) {
          return py_lt(orig[x.second], orig[y.second]);
        });
      }
      out.push_back('{');
      bool first = true;
      for (const auto& [key, i] : keys) {
        if (!first) out += o.item_sep;
        first = false;
        if (o.indent_set) newline_indent(out, o, level + 1);
        json::append_quoted(out, key, o.ensure_ascii);
        out += o.key_sep;
        dump_rec(out, dict_value(v, i), o, level + 1);
      }
      if (o.indent_set) newline_indent(out, o, level);
      out.push_back('}');
      return;
    }
    default: not_serializable(v);
  }
}

}  // namespace

void dump_json(std::string& out, const Val& v, const JsonOpts& o) { dump_rec(out, v, o, 0); }

}  // namespace espetero::tmpl
