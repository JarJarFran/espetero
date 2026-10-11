// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Runtime values of the template engine: a 16-byte tagged value with Python
// semantics (str()/repr(), ==, <, truthiness, len, iteration, json.dumps).
//
// Strings and containers coming from the render context are *borrowed*: a
// string points into the json::Value it came from and a list/dict wraps the
// json::Value itself, so rendering never deep-copies the conversation.
// Values created while rendering live in intrusively reference-counted heap
// objects. All of them die with the render call, which is why borrowing from
// the context (and from the template's constants) is safe, and why the
// reference counts need not be atomic.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../util/json.h"
#include "tmpl.h"

namespace espetero::tmpl {

// Python type of a value; drives type(x).__name__ in error messages.
enum class PyType : uint8_t {
  NoneType,
  Bool,
  Int,
  Float,
  Str,
  Markup,
  List,
  Tuple,
  Dict,
  Namespace,
  Macro,
  LoopContext,
  Generator,
  Range,
  DictKeys,
  DictValues,
  DictItems,
  Function,
  BuiltinMethod,
  Type,
  Undefined,
  Cycler,
  Joiner,
  TemplateReference,
};
const char* py_type_name(PyType t);      // type(x).__name__
std::string object_type_repr(PyType t);  // jinja2.utils.object_type_repr

enum class Kind : uint8_t {
  Missing,  // internal: an unassigned variable slot (loads turn it into Undefined)
  Undefined,
  None,
  Bool,
  Int,
  Float,
  Str,
  List,
  Tuple,
  Dict,
  Namespace,
  Macro,
  Builtin,  // global function or class (range, dict, namespace, ...)
  Method,   // bound method, e.g. the value of `s.startswith`
  Loop,     // the `loop` object of a for loop
  Iter,     // a one-shot iterator (the generators returned by map/select/...)
  Range,
  View,  // dict_keys / dict_values / dict_items
  Cycler,
  Joiner,
};

enum class BuiltinId : uint16_t {
  Range,
  Dict,
  Namespace,
  Lipsum,
  Cycler,
  Joiner,
  RaiseException,
  StrftimeNow,
  Generation,         // internal: body of transformers' {% generation %} block
  TemplateReference,  // the `self` variable (blocks are not supported)
};

// How an Undefined was produced; selects the UndefinedError message.
enum class UndefKind : uint8_t {
  Name,     // '{name}' is undefined
  Attr,     // '{type} object' has no attribute '{name}'
  ElemInt,  // {type} object has no element {int}
  Hint,     // explicit message
  Unsafe,   // SecurityError: access to attribute '{name}' of '{type}' object is unsafe.
};

enum class ViewKind : uint8_t { Keys, Values, Items };

struct Obj {
  uint32_t rc = 1;
  virtual ~Obj() = default;
};

class Val {
 public:
  Val() noexcept : kind_(Kind::None), flags_(0), aux_(0), len_(0) { u_.i = 0; }
  Val(const Val& o) noexcept : kind_(o.kind_), flags_(o.flags_), aux_(o.aux_), len_(o.len_), u_(o.u_) {
    if (flags_ & kOwn) ++u_.o->rc;
  }
  Val(Val&& o) noexcept : kind_(o.kind_), flags_(o.flags_), aux_(o.aux_), len_(o.len_), u_(o.u_) {
    o.kind_ = Kind::None;
    o.flags_ = 0;
  }
  Val& operator=(const Val& o) noexcept {
    if (this != &o) {
      if (o.flags_ & kOwn) ++o.u_.o->rc;
      release();
      kind_ = o.kind_, flags_ = o.flags_, aux_ = o.aux_, len_ = o.len_, u_ = o.u_;
    }
    return *this;
  }
  Val& operator=(Val&& o) noexcept {
    if (this != &o) {
      release();
      kind_ = o.kind_, flags_ = o.flags_, aux_ = o.aux_, len_ = o.len_, u_ = o.u_;
      o.kind_ = Kind::None;
      o.flags_ = 0;
    }
    return *this;
  }
  ~Val() { release(); }

  // ---- constructors
  static Val missing() { return make(Kind::Missing); }
  static Val none() { return Val(); }
  static Val boolean(bool b) {
    Val v = make(Kind::Bool);
    v.u_.b = b;
    return v;
  }
  static Val integer(int64_t i) {
    Val v = make(Kind::Int);
    v.u_.i = i;
    return v;
  }
  static Val real(double d) {
    Val v = make(Kind::Float);
    v.u_.f = d;
    return v;
  }
  // Borrowed string: the bytes must outlive the render (template constants,
  // strings of the render context, or slices of those).
  static Val str_ref(std::string_view s, bool markup = false) {
    Val v = make(Kind::Str);
    v.u_.s = s.data();
    v.len_ = uint32_t(s.size());
    if (markup) v.flags_ |= kMarkup;
    return v;
  }
  static Val str(std::string s, bool markup = false);
  static Val json_list(const json::Value* j) {
    Val v = make(Kind::List);
    v.flags_ = kJson;
    v.u_.j = j;
    return v;
  }
  static Val json_dict(const json::Value* j) {
    Val v = make(Kind::Dict);
    v.flags_ = kJson;
    v.u_.j = j;
    return v;
  }
  // Wraps a heap object, taking over the reference the caller holds.
  static Val object(Kind k, Obj* o, uint16_t aux = 0) {
    Val v = make(k);
    v.flags_ = kOwn;
    v.aux_ = aux;
    v.u_.o = o;
    return v;
  }
  static Val builtin(BuiltinId id) {
    Val v = make(Kind::Builtin);
    v.aux_ = uint16_t(id);
    return v;
  }
  static Val undef_name(std::string_view borrowed_name) { return undef_with(UndefKind::Name, PyType::NoneType, str_ref(borrowed_name)); }
  static Val undef_attr(PyType obj, const Val& name) { return undef_with(UndefKind::Attr, obj, name); }
  static Val undef_unsafe(PyType obj, const Val& name) { return undef_with(UndefKind::Unsafe, obj, name); }
  static Val undef_hint(std::string_view static_hint) { return undef_with(UndefKind::Hint, PyType::NoneType, str_ref(static_hint)); }
  static Val undef_hint_owned(std::string msg) { return undef_with(UndefKind::Hint, PyType::NoneType, str(std::move(msg))); }
  static Val undef_elem(PyType obj, int64_t key) {
    Val v = make(Kind::Undefined);
    v.aux_ = uint16_t(uint16_t(UndefKind::ElemInt) | (uint16_t(obj) << 8));
    v.u_.i = key;
    return v;
  }

  // ---- inspection
  Kind kind() const { return kind_; }
  bool is(Kind k) const { return kind_ == k; }
  bool is_missing() const { return kind_ == Kind::Missing; }
  bool is_undef() const { return kind_ == Kind::Undefined; }
  bool is_none() const { return kind_ == Kind::None; }
  bool is_str() const { return kind_ == Kind::Str; }
  bool is_int() const { return kind_ == Kind::Int; }
  bool is_float() const { return kind_ == Kind::Float; }
  bool is_bool() const { return kind_ == Kind::Bool; }
  bool is_list() const { return kind_ == Kind::List; }
  bool is_tuple() const { return kind_ == Kind::Tuple; }
  bool is_dict() const { return kind_ == Kind::Dict; }
  // Python int (bool is an int subclass).
  bool is_intlike() const { return kind_ == Kind::Int || kind_ == Kind::Bool; }
  bool is_number() const { return kind_ == Kind::Int || kind_ == Kind::Bool || kind_ == Kind::Float; }
  bool is_seq() const { return kind_ == Kind::List || kind_ == Kind::Tuple; }

  bool b() const { return u_.b; }
  int64_t i() const { return kind_ == Kind::Bool ? int64_t(u_.b) : u_.i; }
  double f() const { return kind_ == Kind::Float ? u_.f : double(i()); }
  // Str payload (also the name/hint of an Undefined).
  std::string_view sv() const;
  bool markup() const { return flags_ & kMarkup; }
  bool borrowed() const { return !(flags_ & kOwn); }
  bool is_json() const { return flags_ & kJson; }
  const json::Value* json() const { return u_.j; }
  Obj* obj() const { return u_.o; }
  template <class T>
  T* as() const {
    return static_cast<T*>(u_.o);
  }
  uint16_t aux() const { return aux_; }
  BuiltinId builtin_id() const { return BuiltinId(aux_); }
  UndefKind undef_kind() const { return UndefKind(aux_ & 0xFF); }
  PyType undef_type() const { return PyType(aux_ >> 8); }
  int64_t undef_key() const { return u_.i; }
  // Same heap object / same scalar identity (Python `is` for our purposes).
  bool same_object(const Val& o) const { return (flags_ & kOwn) && (o.flags_ & kOwn) && u_.o == o.u_.o; }
  void set_markup(bool m) {
    if (m)
      flags_ |= kMarkup;
    else
      flags_ &= uint8_t(~kMarkup);
  }

 private:
  enum : uint8_t { kOwn = 1, kMarkup = 2, kJson = 4 };
  static Val make(Kind k) {
    Val v;
    v.kind_ = k;
    return v;
  }
  static Val undef_with(UndefKind uk, PyType t, const Val& payload) {
    Val v(payload);
    v.kind_ = Kind::Undefined;
    v.aux_ = uint16_t(uint16_t(uk) | (uint16_t(t) << 8));
    v.flags_ &= uint8_t(~kMarkup);
    return v;
  }
  void release() noexcept {
    if ((flags_ & kOwn) && --u_.o->rc == 0) delete u_.o;
  }

  Kind kind_;
  uint8_t flags_;
  uint16_t aux_;
  uint32_t len_;
  union {
    bool b;
    int64_t i;
    double f;
    const char* s;
    const json::Value* j;
    Obj* o;
  } u_;
};
static_assert(sizeof(Val) == 16);

// ---------------------------------------------------------------- heap objects
struct StrObj final : Obj {
  std::string s;
  explicit StrObj(std::string x) : s(std::move(x)) {}
};

inline std::string_view Val::sv() const {
  if (flags_ & kOwn) return static_cast<StrObj*>(u_.o)->s;
  return {u_.s, len_};
}

struct ListObj final : Obj {  // list and tuple
  std::vector<Val> v;
  ListObj() = default;
  explicit ListObj(std::vector<Val> x) : v(std::move(x)) {}
};

// Insertion-ordered mapping with Python key equality (1 == 1.0 == True).
struct MapData {
  std::vector<Val> keys, vals;
  int find(const Val& key) const;
  int find_str(std::string_view key) const;
  void set(Val key, Val value);
};

struct DictObj final : Obj {
  MapData m;
};

struct NsObj final : Obj {  // jinja2 namespace(): mutable attributes
  MapData m;
};

struct RangeObj final : Obj {
  int64_t start, stop, step;
  RangeObj(int64_t a, int64_t b, int64_t s) : start(a), stop(b), step(s) {}
  int64_t size() const;
  int64_t at(int64_t i) const { return start + i * step; }
};

struct ViewObj final : Obj {  // dict.keys() / .values() / .items()
  Val dict;
  ViewKind which;
  ViewObj(Val d, ViewKind w) : dict(std::move(d)), which(w) {}
};

// A Python generator. Filters that return generators in jinja2 (map, select,
// items, ...) compute their items eagerly here, but consumption is shared and
// one-shot exactly like a generator: iterating it twice yields nothing the
// second time, it is always truthy and has no len().
struct IterObj final : Obj {
  std::vector<Val> items;
  size_t pos = 0;
  const char* origin;  // generator function name, for repr()
  IterObj(std::vector<Val> v, const char* o) : items(std::move(v)), origin(o) {}
};

struct MethodObj final : Obj {
  Val self;
  uint16_t id;
  MethodObj(Val s, uint16_t m) : self(std::move(s)), id(m) {}
};

struct CyclerObj final : Obj {
  std::vector<Val> items;
  size_t pos = 0;
};

struct JoinerObj final : Obj {
  Val sep;
  bool used = false;
};

// ---------------------------------------------------------------- errors
[[noreturn]] void raise(ErrorKind k, const std::string& msg);
// Raises the UndefinedError (or SecurityError) an operation on `u` raises.
[[noreturn]] void raise_undefined(const Val& u);
std::string undefined_message(const Val& u);
std::string py_repr_str(std::string_view s);  // repr() of a str

// ---------------------------------------------------------------- Python protocol
Val from_json(const json::Value& v);
PyType type_of(const Val& v);
inline const char* tname(const Val& v) { return py_type_name(type_of(v)); }

bool truthy(const Val& v);
void append_str(std::string& out, const Val& v);  // str(v)
std::string to_str(const Val& v);
void append_repr(std::string& out, const Val& v);  // repr(v)
std::string to_repr(const Val& v);
std::string format_float(double d);  // repr(float)

bool py_eq(const Val& a, const Val& b);
bool py_lt(const Val& a, const Val& b);  // a < b; TypeError when unordered
// Rich comparison: op 0 '<', 1 '<=', 2 '>', 3 '>='. Undefined operands raise.
bool py_compare(int op, const Val& a, const Val& b);
bool is_hashable(const Val& v);
void require_hashable(const Val& v);  // TypeError "unhashable type: 'list'"
int64_t py_len(const Val& v);         // len(v); TypeError when unsized
bool has_len(const Val& v);

// str payload helpers
inline bool is_markup(const Val& v) { return v.is_str() && v.markup(); }
Val substr(const Val& s, size_t byte_pos, size_t byte_len);  // keeps borrowing when possible

// List / tuple / dict access (JSON-backed or owned).
size_t seq_size(const Val& v);
Val seq_at(const Val& v, size_t i);
size_t dict_size(const Val& v);
Val dict_key(const Val& v, size_t i);
Val dict_value(const Val& v, size_t i);
bool dict_get(const Val& d, const Val& key, Val& out);  // key must be hashable
bool dict_get_str(const Val& d, std::string_view key, Val& out);

Val make_list(std::vector<Val> v);
Val make_tuple(std::vector<Val> v);
Val make_dict(MapData m);
Val make_iter(std::vector<Val> items, const char* origin);

// Iterates any Python iterable. Undefined iterates as empty; generators are
// consumed. Throws TypeError "'int' object is not iterable".
class PyIter {
 public:
  explicit PyIter(const Val& v);
  bool next(Val& out);

 private:
  enum Mode : uint8_t { Empty, Seq, Dict, Str, Range, Gen, View };
  Val src_;
  Mode mode_ = Empty;
  ViewKind view_ = ViewKind::Keys;
  size_t i_ = 0, n_ = 0;
  int64_t cur_ = 0, step_ = 0;
  std::string_view s_;
};
std::vector<Val> to_vector(const Val& iterable);  // list(iterable)

// json.dumps(v, ensure_ascii=..., indent=..., separators=..., sort_keys=...)
struct JsonOpts {
  bool ensure_ascii = false;
  bool indent_set = false;
  std::string indent;  // the indent unit when indent_set
  std::string item_sep = ", ";
  std::string key_sep = ": ";
  bool sort_keys = false;
};
void dump_json(std::string& out, const Val& v, const JsonOpts& o);

}  // namespace espetero::tmpl
