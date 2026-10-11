// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// JSON values, a strict UTF-8 parser and a serializer that reproduces
// Python's json.dumps byte for byte (key order, separators, ensure_ascii,
// float repr). Prompts are built with json.dumps in the reference stack, so
// token-identical prompts need identical serialization.
//
// A Value is 16 bytes: a type tag and a payload (scalars inline, strings,
// arrays and objects on the heap). Copies are deep; moves are cheap. Objects
// keep insertion order (like Python dicts) and build a hash index lazily
// once they grow past a few keys.
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace espetero::json {

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class Type : uint8_t { Null, Bool, Int, Double, String, Array, Object };

class Value;

struct Object {
  std::vector<std::string> keys;
  std::vector<Value> vals;
  // Lazily built for lookups in objects with many keys.
  mutable std::unique_ptr<std::unordered_map<std::string_view, uint32_t>> index;
  Object() = default;
  Object(const Object& o);
  Object& operator=(const Object& o);
  Object(Object&&) noexcept = default;
  Object& operator=(Object&&) noexcept = default;
  size_t size() const { return keys.size(); }
  const Value* find(std::string_view key) const;
  Value* find(std::string_view key);
  // Insert or replace (keeps the original position on replace).
  Value& set(std::string key, Value v);
  bool erase(std::string_view key);
};

class Value {
 public:
  Value() noexcept : t_(Type::Null) { u_.i = 0; }
  Value(std::nullptr_t) noexcept : Value() {}
  Value(bool b) noexcept : t_(Type::Bool) { u_.b = b; }
  Value(int v) noexcept : t_(Type::Int) { u_.i = v; }
  Value(unsigned v) noexcept : t_(Type::Int) { u_.i = v; }
  Value(long v) noexcept : t_(Type::Int) { u_.i = v; }
  Value(long long v) noexcept : t_(Type::Int) { u_.i = v; }
  Value(unsigned long v) noexcept : t_(Type::Int) { u_.i = int64_t(v); }
  Value(unsigned long long v) noexcept : t_(Type::Int) { u_.i = int64_t(v); }
  Value(double d) noexcept : t_(Type::Double) { u_.d = d; }
  Value(float d) noexcept : t_(Type::Double) { u_.d = d; }
  Value(const char* s) : t_(Type::String) { u_.s = new std::string(s); }
  Value(std::string_view s) : t_(Type::String) { u_.s = new std::string(s); }
  Value(const std::string& s) : t_(Type::String) { u_.s = new std::string(s); }
  Value(std::string&& s) : t_(Type::String) { u_.s = new std::string(std::move(s)); }
  Value(std::vector<Value> a) : t_(Type::Array) { u_.a = new std::vector<Value>(std::move(a)); }
  Value(Object o) : t_(Type::Object) { u_.o = new Object(std::move(o)); }

  Value(const Value& v) : t_(Type::Null) { copy_from(v); }
  Value(Value&& v) noexcept : t_(v.t_), u_(v.u_) {
    v.t_ = Type::Null;
    v.u_.i = 0;
  }
  Value& operator=(const Value& v) {
    if (this != &v) {
      Value tmp(v);
      swap(tmp);
    }
    return *this;
  }
  Value& operator=(Value&& v) noexcept {
    if (this != &v) {
      release();
      t_ = v.t_;
      u_ = v.u_;
      v.t_ = Type::Null;
      v.u_.i = 0;
    }
    return *this;
  }
  ~Value() { release(); }
  void swap(Value& o) noexcept {
    std::swap(t_, o.t_);
    std::swap(u_, o.u_);
  }

  static Value array() { return Value(std::vector<Value>{}); }
  static Value object() { return Value(Object{}); }

  Type type() const { return t_; }
  bool is_null() const { return t_ == Type::Null; }
  bool is_bool() const { return t_ == Type::Bool; }
  bool is_int() const { return t_ == Type::Int; }
  bool is_double() const { return t_ == Type::Double; }
  bool is_number() const { return t_ == Type::Int || t_ == Type::Double; }
  bool is_string() const { return t_ == Type::String; }
  bool is_array() const { return t_ == Type::Array; }
  bool is_object() const { return t_ == Type::Object; }

  bool as_bool() const {
    if (t_ != Type::Bool) throw Error("expected a boolean");
    return u_.b;
  }
  int64_t as_int() const {
    if (t_ == Type::Int) return u_.i;
    // A double converts only when it is integral and inside int64's range
    // (casting one outside it is undefined behaviour).
    if (t_ == Type::Double && in_int64_range(u_.d) && u_.d == double(int64_t(u_.d))) return int64_t(u_.d);
    throw Error("expected an integer");
  }
  double as_double() const {
    if (t_ == Type::Double) return u_.d;
    if (t_ == Type::Int) return double(u_.i);
    throw Error("expected a number");
  }
  const std::string& as_string() const {
    if (t_ != Type::String) throw Error("expected a string");
    return *u_.s;
  }
  std::string& as_string() {
    if (t_ != Type::String) throw Error("expected a string");
    return *u_.s;
  }
  const std::vector<Value>& as_array() const {
    if (t_ != Type::Array) throw Error("expected an array");
    return *u_.a;
  }
  std::vector<Value>& as_array() {
    if (t_ != Type::Array) throw Error("expected an array");
    return *u_.a;
  }
  const Object& as_object() const {
    if (t_ != Type::Object) throw Error("expected an object");
    return *u_.o;
  }
  Object& as_object() {
    if (t_ != Type::Object) throw Error("expected an object");
    return *u_.o;
  }

  // Array/object size (0 for scalars).
  size_t size() const {
    if (t_ == Type::Array) return u_.a->size();
    if (t_ == Type::Object) return u_.o->size();
    return 0;
  }
  const Value& operator[](size_t i) const { return as_array().at(i); }
  Value& operator[](size_t i) { return as_array().at(i); }
  // Object member or nullptr (also nullptr when this is not an object).
  const Value* get(std::string_view key) const { return t_ == Type::Object ? u_.o->find(key) : nullptr; }
  Value* get(std::string_view key) { return t_ == Type::Object ? u_.o->find(key) : nullptr; }
  // Object member; inserts null when missing (turns null into an object).
  Value& operator[](std::string_view key);
  Value& set(std::string key, Value v) {
    if (t_ == Type::Null) *this = object();
    return as_object().set(std::move(key), std::move(v));
  }
  void push_back(Value v) {
    if (t_ == Type::Null) *this = array();
    as_array().push_back(std::move(v));
  }

  // Typed reads of optional object members with defaults.
  bool get_bool(std::string_view key, bool def) const {
    const Value* v = get(key);
    return v && v->is_bool() ? v->u_.b : def;
  }
  int64_t get_int(std::string_view key, int64_t def) const {
    const Value* v = get(key);
    return v && v->is_number() ? v->as_int_lenient() : def;
  }
  double get_double(std::string_view key, double def) const {
    const Value* v = get(key);
    return v && v->is_number() ? v->as_double() : def;
  }
  std::string get_string(std::string_view key, std::string_view def) const {
    const Value* v = get(key);
    return v && v->is_string() ? *v->u_.s : std::string(def);
  }

  bool operator==(const Value& o) const;
  bool operator!=(const Value& o) const { return !(*this == o); }

 private:
  // -2^63 <= d < 2^63 (false for NaN).
  static bool in_int64_range(double d) { return d >= -9223372036854775808.0 && d < 9223372036854775808.0; }
  // Doubles truncate toward zero, clamped to int64's range (NaN: 0).
  int64_t as_int_lenient() const {
    if (t_ == Type::Int) return u_.i;
    if (in_int64_range(u_.d)) return int64_t(u_.d);
    return u_.d > 0 ? INT64_MAX : u_.d < 0 ? INT64_MIN : 0;
  }
  void release() noexcept;
  void copy_from(const Value& v);

  Type t_;
  union {
    bool b;
    int64_t i;
    double d;
    std::string* s;
    std::vector<Value>* a;
    Object* o;
  } u_;
};

// Strict RFC 8259 parsing of UTF-8 text (invalid UTF-8 and lone surrogate
// escapes become U+FFFD). Throws Error with the byte offset on failure.
Value parse(std::string_view text, int max_depth = 512);

struct DumpOptions {
  // Python defaults are (", ", ": "); compact is (",", ":").
  std::string_view item_sep = ",";
  std::string_view key_sep = ":";
  bool ensure_ascii = false;  // \uXXXX for everything outside 0x20..0x7E
  int indent = -1;            // >= 0: pretty print (Python indent=n)
  bool sort_keys = false;
};

// Python-compatible serialization: json.dumps(v, ensure_ascii=..., separators=..., ...).
std::string dump(const Value& v, const DumpOptions& opt = {});
void dump_to(std::string& out, const Value& v, const DumpOptions& opt = {});
inline std::string dump_python_default(const Value& v, bool ensure_ascii = true) {
  DumpOptions o;
  o.item_sep = ", ";
  o.key_sep = ": ";
  o.ensure_ascii = ensure_ascii;
  return dump(v, o);
}

// A JSON string literal (with quotes) as json.dumps would write it.
void append_quoted(std::string& out, std::string_view s, bool ensure_ascii = false);
// Python's repr(float) (shortest round trip, e.g. 1.0, 1e-05, 1e+16).
std::string format_double(double d);

// UTF-8 helpers shared by the tokenizer, templates and the server.
// Decodes one code point at s[i] (advancing i); invalid bytes yield U+FFFD
// and advance by one.
uint32_t utf8_next(std::string_view s, size_t& i);
void utf8_append(std::string& out, uint32_t cp);
bool utf8_valid(std::string_view s);
// Length of the longest prefix of s that does not end inside a UTF-8
// sequence (for streaming text out in whole characters).
size_t utf8_complete_prefix(std::string_view s);

}  // namespace espetero::json
