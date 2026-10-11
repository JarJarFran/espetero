// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "json_schema.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <unordered_map>

namespace espetero::grammar {

namespace {

// Type sets: "number" covers the integers.
enum : int { T_OBJ = 1, T_ARR = 2, T_STR = 4, T_NUM = 8, T_INT = 16, T_BOOL = 32, T_NULL = 64 };
constexpr int T_ALL = T_OBJ | T_ARR | T_STR | T_NUM | T_BOOL | T_NULL;

int intersect_types(int a, int b) {
  int t = a & b & ~(T_NUM | T_INT);
  const bool an = a & (T_NUM | T_INT), bn = b & (T_NUM | T_INT);
  if ((a & T_NUM) && (b & T_NUM)) t |= T_NUM;
  else if (an && bn) t |= T_INT;
  return t;
}

const std::set<std::string>& handled_keywords() {
  static const std::set<std::string> k = {"type", "enum", "const", "properties", "required", "additionalProperties",
                                          "minProperties", "maxProperties", "items", "prefixItems", "additionalItems",
                                          "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum",
                                          "exclusiveMinimum", "exclusiveMaximum", "anyOf", "oneOf", "allOf", "$ref",
                                          "nullable", "uniqueItems", "unevaluatedProperties", "unevaluatedItems"};
  return k;
}
const std::set<std::string>& unsupported_keywords() {
  static const std::set<std::string> k = {"pattern", "format", "multipleOf", "patternProperties", "propertyNames",
                                          "dependentRequired", "dependentSchemas", "dependencies", "if", "then",
                                          "else", "not", "contains", "minContains", "maxContains", "$dynamicRef",
                                          "$recursiveRef"};
  return k;
}
bool constraining(const std::string& k) { return handled_keywords().count(k) || unsupported_keywords().count(k); }

// Keywords that only constrain values of one type (a type-less schema takes the types they imply).
int implied_type(const std::string& k) {
  static const std::unordered_map<std::string, int> m = {
      {"properties", T_OBJ}, {"required", T_OBJ}, {"additionalProperties", T_OBJ}, {"minProperties", T_OBJ},
      {"maxProperties", T_OBJ}, {"patternProperties", T_OBJ}, {"propertyNames", T_OBJ}, {"unevaluatedProperties", T_OBJ},
      {"dependentRequired", T_OBJ}, {"dependentSchemas", T_OBJ}, {"dependencies", T_OBJ},
      {"items", T_ARR}, {"prefixItems", T_ARR}, {"additionalItems", T_ARR}, {"minItems", T_ARR}, {"maxItems", T_ARR},
      {"uniqueItems", T_ARR}, {"contains", T_ARR}, {"minContains", T_ARR}, {"maxContains", T_ARR}, {"unevaluatedItems", T_ARR},
      {"minLength", T_STR}, {"maxLength", T_STR}, {"pattern", T_STR},
      {"minimum", T_NUM}, {"maximum", T_NUM}, {"exclusiveMinimum", T_NUM}, {"exclusiveMaximum", T_NUM}, {"multipleOf", T_NUM}};
  auto it = m.find(k);
  return it == m.end() ? 0 : it->second;
}

// JSON pointer token escaping (RFC 6901).
std::string ptr_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '~') o += "~0";
    else if (c == '/') o += "~1";
    else o.push_back(c);
  }
  return o;
}

json::Value without(const json::Value& s, std::initializer_list<const char*> keys) {
  json::Value o = json::Value::object();
  if (!s.is_object()) return o;
  const json::Object& obj = s.as_object();
  for (size_t i = 0; i < obj.keys.size(); ++i) {
    bool skip = false;
    for (const char* k : keys) skip = skip || obj.keys[i] == k;
    if (!skip) o.set(obj.keys[i], obj.vals[i]);
  }
  return o;
}

// A JSON number as a decimal: integers exactly, doubles by their shortest repr.
bool to_decimal(const json::Value& v, Bound& b) {
  if (!v.is_number()) return false;
  std::string t;
  if (v.is_int()) t = std::to_string(v.as_int());
  else {
    const double d = v.as_double();
    if (!std::isfinite(d)) return false;
    t = json::format_double(d);
  }
  size_t i = 0;
  b.neg = false;
  if (i < t.size() && t[i] == '-') b.neg = true, ++i;
  std::string digits;
  int e = 0;
  bool point = false;
  for (; i < t.size() && t[i] != 'e' && t[i] != 'E'; ++i) {
    if (t[i] == '.') point = true;
    else {
      digits.push_back(t[i]);
      if (point) --e;
    }
  }
  if (i < t.size()) e += std::atoi(t.c_str() + i + 1);
  size_t z = 0;
  while (z + 1 < digits.size() && digits[z] == '0') ++z;
  digits.erase(0, z);
  while (digits.size() > 1 && digits.back() == '0') digits.pop_back(), ++e;
  if (digits.size() > 19) return false;
  b.m = std::stoull(digits);
  b.e = b.m == 0 ? 0 : e;
  if (b.m == 0) b.neg = false;
  return true;
}

// Signed comparison of two bounds' values (exclusivity apart).
int cmp_value(const Bound& a, const Bound& b) {
  auto digits = [](unsigned __int128 m) {
    int n = 0;
    while (m) m /= 10, ++n;
    return n;
  };
  const bool na = a.neg && a.m, nb = b.neg && b.m;
  if (na != nb) return na ? -1 : 1;
  int c;
  if (a.m == 0 || b.m == 0) {
    c = a.m == b.m ? 0 : (a.m == 0 ? -1 : 1);
  } else {
    const int ma = digits(a.m) + a.e, mb = digits(b.m) + b.e;
    if (ma != mb) {
      c = ma < mb ? -1 : 1;
    } else {
      unsigned __int128 x = a.m, y = b.m;
      int ea = a.e, eb = b.e;
      while (ea > eb) x *= 10, --ea;
      while (eb > ea) y *= 10, --eb;
      c = x < y ? -1 : x > y ? 1 : 0;
    }
  }
  return na ? -c : c;
}

// floor / ceil of a decimal as a signed 128-bit integer; false when beyond +-10^30.
bool round_dec(const Bound& b, bool ceil, __int128& out) {
  __int128 q;
  bool frac = false;
  if (b.e >= 0) {
    if (b.e > 30) return false;
    q = __int128(b.m);
    for (int i = 0; i < b.e; ++i) q *= 10;
  } else if (b.e < -19) {
    q = 0;
    frac = b.m != 0;
  } else {
    __int128 p = 1;
    for (int i = 0; i < -b.e; ++i) p *= 10;
    q = __int128(b.m) / p;
    frac = __int128(b.m) % p != 0;
  }
  if (b.neg) {
    q = -q;
    if (frac && !ceil) q -= 1;  // floor of a negative non-integer
  } else if (frac && ceil) {
    q += 1;
  }
  out = q;
  return true;
}

Bound int_bound(__int128 v) {
  Bound b;
  b.set = true;
  b.neg = v < 0;
  b.m = uint64_t(v < 0 ? -v : v);
  b.e = 0;
  return b;
}

class Compiler {
 public:
  Compiler(const json::Value& root, const SchemaOptions& opt, std::vector<std::string>* un)
      : root_(root), opt_(opt), un_(un), g_(std::make_shared<Grammar>()) {}

  std::shared_ptr<Grammar> run() {
    g_->root = compile(root_, "#");
    finish();
    return g_;
  }

  int any() {
    if (any_ >= 0) return any_;
    Node alt;
    alt.kind = Kind::Alt;
    any_ = g_->add(alt);
    Node o;
    o.kind = Kind::Object;
    o.additional = any_;
    const int obj = add_object(std::move(o));
    Node a;
    a.kind = Kind::Array;
    a.items = any_;
    const int arr = g_->add(std::move(a));
    Node s;
    s.kind = Kind::String;
    const int str = g_->add(std::move(s));
    Node n;
    n.kind = Kind::Number;
    const int num = g_->add(std::move(n));
    const int lit = literal_node({"true", "false", "null"});
    g_->nodes[size_t(any_)].alts = {obj, arr, str, num, lit};
    return any_;
  }

 private:
  // ---- helpers
  [[noreturn]] void error(const std::string& path, const std::string& msg) { throw SchemaError(msg + " (at " + path + ")"); }

  void unsupported(const std::string& what, const std::string& path) {
    if (opt_.strict)
      throw SchemaError("\"" + what + "\" at " + path +
                        " is not enforced by this server (strict mode refuses it; with \"strict\": false the reply follows "
                        "the rest of the schema and the response lists it)");
    const std::string note = what + " at " + path;
    if (un_ && std::find(un_->begin(), un_->end(), note) == un_->end()) un_->push_back(note);
  }

  int never() {
    if (never_ < 0) never_ = g_->add(Node{});
    return never_;
  }

  int literal_node(const std::vector<std::string>& texts) {
    if (texts.empty()) return never();
    Node n;
    n.kind = Kind::Literal;
    n.lit_trie = g_->new_trie();
    for (const std::string& t : texts) g_->trie_insert(n.lit_trie, t, 0);
    return g_->add(std::move(n));
  }

  // Derived object data: first_req, req_after and the key trie (texts after the opening quote).
  void finish_object(Node& n) {
    const int np = int(n.props.size());
    n.first_req.assign(size_t(np) + 1, np);
    for (int j = np - 1; j >= 0; --j) n.first_req[size_t(j)] = n.props[size_t(j)].required ? j : n.first_req[size_t(j) + 1];
    n.req_after.assign(size_t(np), 0);
    for (int j = np - 1, c = 0; j >= 0; --j) {
      n.req_after[size_t(j)] = c;
      c += n.props[size_t(j)].required;
    }
    n.key_trie = -1;
    if (np) {
      n.key_trie = g_->new_trie();
      for (int j = 0; j < np; ++j) g_->trie_insert(n.key_trie, canonical_text(json::Value(n.props[size_t(j)].name)).substr(1), j, j);
    }
  }
  int add_object(Node n) {
    finish_object(n);
    return g_->add(std::move(n));
  }

  bool has_constraints(const json::Value& s) const {
    if (!s.is_object()) return s.is_bool() && !s.as_bool();
    for (const std::string& k : s.as_object().keys)
      if (constraining(k)) return true;
    return false;
  }
  bool pure_ref(const json::Value& s) const {
    if (!s.is_object() || !s.get("$ref")) return false;
    for (const std::string& k : s.as_object().keys)
      if (k != "$ref" && constraining(k)) return false;
    return true;
  }

  // ---- references
  const json::Value& resolve(const json::Value& ref, const std::string& path) {
    if (!ref.is_string()) error(path, "$ref: a string");
    const std::string& r = ref.as_string();
    if (r.empty() || r[0] != '#')
      error(path, "$ref \"" + r + "\": only references inside the schema (#, #/$defs/..., #/definitions/...) are supported");
    const json::Value* v = &root_;
    size_t i = 1;
    if (i < r.size() && r[i] != '/') error(path, "$ref \"" + r + "\": anchors are not supported (use a JSON pointer)");
    while (i < r.size()) {
      size_t j = r.find('/', i + 1);
      if (j == std::string::npos) j = r.size();
      std::string tok;
      for (size_t k = i + 1; k < j; ++k) {  // ~0 ~1 and %xx
        if (r[k] == '~' && k + 1 < j) {
          tok.push_back(r[k + 1] == '1' ? '/' : '~');
          ++k;
        } else if (r[k] == '%' && k + 2 < j && std::isxdigit((unsigned char)r[k + 1]) && std::isxdigit((unsigned char)r[k + 2])) {
          tok.push_back(char(std::stoi(r.substr(k + 1, 2), nullptr, 16)));
          k += 2;
        } else {
          tok.push_back(r[k]);
        }
      }
      if (v->is_object()) {
        v = v->get(tok);
      } else if (v->is_array()) {
        char* end = nullptr;
        const long idx = std::strtol(tok.c_str(), &end, 10);
        v = (!tok.empty() && end && *end == 0 && idx >= 0 && size_t(idx) < v->size()) ? &(*v)[size_t(idx)] : nullptr;
      } else {
        v = nullptr;
      }
      if (!v) error(path, "$ref \"" + r + "\": no such definition in the schema");
      i = j;
    }
    return *v;
  }

  int compile_ref(const json::Value& ref, const std::string& path) {
    const json::Value& target = resolve(ref, path);
    const std::string& r = ref.as_string();
    if (auto it = refs_.find(r); it != refs_.end()) return it->second;
    Node alt;  // a placeholder: recursive references point at it
    alt.kind = Kind::Alt;
    const int node = g_->add(std::move(alt));
    refs_[r] = node;
    const int inner = compile(target, r);
    g_->nodes[size_t(node)].alts = {inner};
    return node;
  }

  // ---- composition: a schema as a list of alternatives (disjunctive normal
  // form) of simple schemas (no $ref with constraints beside it, no allOf /
  // anyOf / oneOf at the top) and pure references.
  std::vector<json::Value> dnf(const json::Value& s, const std::string& path) {
    if (++depth_ > 64) error(path, "schema composition nested too deep");
    struct Guard {
      int& d;
      ~Guard() { --d; }
    } guard{depth_};
    if (s.is_bool()) return s.as_bool() ? std::vector<json::Value>{json::Value::object()} : std::vector<json::Value>{};
    if (!s.is_object()) error(path, "a schema is an object or a boolean");
    std::vector<json::Value> out{without(s, {"$ref", "allOf", "anyOf", "oneOf"})};
    if (const json::Value* r = s.get("$ref")) {
      resolve(*r, path);
      json::Value t = json::Value::object();
      t.set("$ref", *r);
      out = product(out, {t}, path);
    }
    if (const json::Value* a = s.get("allOf")) {
      if (!a->is_array() || a->size() == 0) error(path, "allOf: a non-empty array of schemas");
      for (size_t i = 0; i < a->size(); ++i) out = product(out, dnf((*a)[i], path + "/allOf/" + std::to_string(i)), path);
    }
    for (const char* k : {"anyOf", "oneOf"}) {
      const json::Value* a = s.get(k);
      if (!a) continue;
      if (!a->is_array() || a->size() == 0) error(path, std::string(k) + ": a non-empty array of schemas");
      std::vector<json::Value> opts;
      for (size_t i = 0; i < a->size(); ++i) {
        std::vector<json::Value> d = dnf((*a)[i], path + "/" + k + "/" + std::to_string(i));
        opts.insert(opts.end(), d.begin(), d.end());
      }
      out = product(out, opts, path);
    }
    return out;
  }

  std::vector<json::Value> product(const std::vector<json::Value>& A, const std::vector<json::Value>& B,
                                   const std::string& path) {
    std::vector<json::Value> out;
    for (const json::Value& a : A)
      for (const json::Value& b : B) {
        std::vector<json::Value> c = conj(a, b, path);
        out.insert(out.end(), c.begin(), c.end());
        if (out.size() > 256) error(path, "allOf / anyOf / oneOf combine into more than 256 alternatives");
      }
    return out;
  }

  // Both a and b: terms of a DNF.
  std::vector<json::Value> conj(const json::Value& a, const json::Value& b, const std::string& path) {
    if (!has_constraints(a)) return {b};
    if (!has_constraints(b)) return {a};
    if (pure_ref(a) || pure_ref(b)) {
      const json::Value& r = pure_ref(a) ? a : b;
      const json::Value& o = pure_ref(a) ? b : a;
      const std::string ref = r.get_string("$ref", "");
      if (std::find(inlining_.begin(), inlining_.end(), ref) != inlining_.end()) {
        unsupported("$ref (recursive) merged with other keywords", path);
        return {r};
      }
      inlining_.push_back(ref);
      std::vector<json::Value> terms = dnf(resolve(*r.get("$ref"), path), ref);
      std::vector<json::Value> out;
      for (const json::Value& t : terms) {
        std::vector<json::Value> c = conj(t, o, path);
        out.insert(out.end(), c.begin(), c.end());
      }
      inlining_.pop_back();
      return out;
    }
    json::Value m = merge(a, b, path);
    if (m.is_bool()) return {};  // contradictory: no value
    return {m};
  }

  int type_set(const json::Value& s, const std::string& path) {
    const json::Value* t = s.get("type");
    int ts = 0;
    auto one = [&](const json::Value& v) {
      if (!v.is_string()) error(path, "type: a type name or a list of them");
      const std::string& n = v.as_string();
      if (n == "object") ts |= T_OBJ;
      else if (n == "array") ts |= T_ARR;
      else if (n == "string") ts |= T_STR;
      else if (n == "number") ts |= T_NUM;
      else if (n == "integer") ts |= T_INT;
      else if (n == "boolean") ts |= T_BOOL;
      else if (n == "null") ts |= T_NULL;
      else error(path, "type: unknown type \"" + n + "\"");
    };
    if (!t) ts = T_ALL;
    else if (t->is_array()) {
      if (t->size() == 0) error(path, "type: an empty list");
      for (const json::Value& v : t->as_array()) one(v);
    } else {
      one(*t);
    }
    if ((ts & T_NUM) && (ts & T_INT)) ts &= ~T_INT;
    if (t && s.get_bool("nullable", false)) ts |= T_NULL;
    return ts;
  }
  static json::Value type_value(int ts) {
    json::Value a = json::Value::array();
    if (ts & T_OBJ) a.push_back("object");
    if (ts & T_ARR) a.push_back("array");
    if (ts & T_STR) a.push_back("string");
    if (ts & T_NUM) a.push_back("number");
    if (ts & T_INT) a.push_back("integer");
    if (ts & T_BOOL) a.push_back("boolean");
    if (ts & T_NULL) a.push_back("null");
    return a;
  }

  // Draft-4 boolean exclusiveMinimum / exclusiveMaximum as their numeric forms.
  static json::Value normalize_bounds(json::Value s) {
    for (const auto& [ex, base] : {std::pair<const char*, const char*>{"exclusiveMinimum", "minimum"}, {"exclusiveMaximum", "maximum"}}) {
      const json::Value* e = s.get(ex);
      if (e && e->is_bool()) {
        const bool on = e->as_bool();
        json::Value b = s.get(base) ? *s.get(base) : json::Value();
        s.as_object().erase(ex);
        if (on && b.is_number()) {
          s.as_object().erase(base);
          s.set(ex, b);
        }
      }
    }
    return s;
  }

  // a and b, simple schemas: one schema with both's constraints, or false.
  json::Value merge(const json::Value& a0, const json::Value& b0, const std::string& path) {
    const json::Value a = normalize_bounds(a0), b = normalize_bounds(b0);
    json::Value out = json::Value::object();
    const int ts = intersect_types(type_set(a, path), type_set(b, path));
    if (!ts) return false;
    if (ts != T_ALL) out.set("type", type_value(ts));
    // enum / const: their common values
    auto values = [&](const json::Value& s, bool& set) {
      std::vector<json::Value> v;
      set = false;
      if (const json::Value* c = s.get("const")) v = {*c}, set = true;
      if (const json::Value* e = s.get("enum")) {
        if (!e->is_array()) error(path, "enum: an array");
        if (!set) v = e->as_array();
        else v.erase(std::remove_if(v.begin(), v.end(), [&](const json::Value& x) {
                       return std::find(e->as_array().begin(), e->as_array().end(), x) == e->as_array().end();
                     }), v.end());
        set = true;
      }
      return v;
    };
    bool sa, sb;
    std::vector<json::Value> va = values(a, sa), vb = values(b, sb);
    if (sa || sb) {
      std::vector<json::Value> v;
      if (sa && sb) {
        for (const json::Value& x : va)
          if (std::find(vb.begin(), vb.end(), x) != vb.end()) v.push_back(x);
      } else {
        v = sa ? va : vb;
      }
      if (v.empty()) return false;
      out.set("enum", json::Value(std::move(v)));
    }
    // objects: a key's schema is its property's, else the additionalProperties (absent: any value)
    const json::Value* pa = a.get("properties");
    const json::Value* pb = b.get("properties");
    const json::Value* aa = a.get("additionalProperties");
    const json::Value* ab = b.get("additionalProperties");
    if (pa || pb) {
      json::Value props = json::Value::object();
      std::vector<std::string> keys;
      for (const json::Value* p : {pa, pb})
        if (p && p->is_object())
          for (const std::string& k : p->as_object().keys)
            if (std::find(keys.begin(), keys.end(), k) == keys.end()) keys.push_back(k);
      for (const std::string& k : keys) {
        auto side = [&](const json::Value* p, const json::Value* add) {
          if (p && p->is_object() && p->get(k)) return *p->get(k);
          return add ? *add : json::Value(true);
        };
        const json::Value x = side(pa, aa), y = side(pb, ab);
        if (x.is_bool() && x.as_bool()) props.set(k, y);
        else if (y.is_bool() && y.as_bool()) props.set(k, x);
        else if ((x.is_bool() && !x.as_bool()) || (y.is_bool() && !y.as_bool())) props.set(k, false);
        else {
          json::Value all = json::Value::array();
          all.push_back(x);
          all.push_back(y);
          json::Value w = json::Value::object();
          w.set("allOf", std::move(all));
          props.set(k, std::move(w));
        }
      }
      out.set("properties", std::move(props));
    }
    if (aa || ab) {
      if (!aa) out.set("additionalProperties", *ab);
      else if (!ab) out.set("additionalProperties", *aa);
      else if ((aa->is_bool() && !aa->as_bool()) || (ab->is_bool() && !ab->as_bool())) out.set("additionalProperties", false);
      else if (aa->is_bool()) out.set("additionalProperties", *ab);
      else if (ab->is_bool()) out.set("additionalProperties", *aa);
      else {
        json::Value all = json::Value::array();
        all.push_back(*aa);
        all.push_back(*ab);
        json::Value w = json::Value::object();
        w.set("allOf", std::move(all));
        out.set("additionalProperties", std::move(w));
      }
    }
    // arrays: items of both (tuples in one branch only)
    const bool ta = a.get("prefixItems") || (a.get("items") && a.get("items")->is_array());
    const bool tb = b.get("prefixItems") || (b.get("items") && b.get("items")->is_array());
    if (ta && tb) {
      unsupported("prefixItems (tuples) in more than one branch of allOf", path);
      for (const char* k : {"prefixItems", "items", "additionalItems"})
        if (a.get(k)) out.set(k, *a.get(k));
    } else if (ta || tb) {
      const json::Value& t = ta ? a : b;
      const json::Value& o = ta ? b : a;
      for (const char* k : {"prefixItems", "items", "additionalItems"})
        if (t.get(k)) out.set(k, *t.get(k));
      if (const json::Value* oi = o.get("items")) {  // the other branch's items apply at every position
        auto both = [&](const json::Value& x) {
          json::Value all = json::Value::array();
          all.push_back(x);
          all.push_back(*oi);
          json::Value w = json::Value::object();
          w.set("allOf", std::move(all));
          return w;
        };
        const bool modern = t.get("prefixItems") != nullptr;
        json::Value pre = modern ? *t.get("prefixItems") : *t.get("items");
        for (json::Value& x : pre.as_array()) x = both(x);
        out.set(modern ? "prefixItems" : "items", std::move(pre));
        const char* rest = modern ? "items" : "additionalItems";
        out.set(rest, both(t.get(rest) ? *t.get(rest) : json::Value(true)));
      }
    } else if (a.get("items") && b.get("items")) {
      json::Value all = json::Value::array();
      all.push_back(*a.get("items"));
      all.push_back(*b.get("items"));
      json::Value w = json::Value::object();
      w.set("allOf", std::move(all));
      out.set("items", std::move(w));
    } else if (a.get("items") || b.get("items")) {
      out.set("items", a.get("items") ? *a.get("items") : *b.get("items"));
    }
    // required: both lists
    if (a.get("required") || b.get("required")) {
      json::Value req = json::Value::array();
      for (const json::Value* r : {a.get("required"), b.get("required")})
        if (r && r->is_array())
          for (const json::Value& x : r->as_array())
            if (std::find(req.as_array().begin(), req.as_array().end(), x) == req.as_array().end()) req.push_back(x);
      out.set("required", std::move(req));
    }
    // numeric limits: the stricter
    auto num_or = [&](const char* k, bool larger) {
      const json::Value* x = a.get(k);
      const json::Value* y = b.get(k);
      if (!x && !y) return;
      if (!x || !y) {
        out.set(k, x ? *x : *y);
        return;
      }
      Bound bx, by;
      if (!to_decimal(*x, bx) || !to_decimal(*y, by)) error(path, std::string(k) + ": a number");
      const int c = cmp_value(bx, by);
      out.set(k, (larger ? c >= 0 : c <= 0) ? *x : *y);
    };
    for (const char* k : {"minLength", "minItems", "minProperties", "minimum", "exclusiveMinimum"}) num_or(k, true);
    for (const char* k : {"maxLength", "maxItems", "maxProperties", "maximum", "exclusiveMaximum"}) num_or(k, false);
    for (const char* k : {"uniqueItems"})
      if (a.get_bool(k, false) || b.get_bool(k, false)) out.set(k, true);
    for (const char* k : {"unevaluatedProperties", "unevaluatedItems"}) {
      const json::Value* x = a.get(k);
      const json::Value* y = b.get(k);
      if (x && x->is_bool() && !x->as_bool()) out.set(k, false);
      else if (y && y->is_bool() && !y->as_bool()) out.set(k, false);
      else if (x || y) out.set(k, x ? *x : *y);
    }
    // the rest (unsupported keywords): a's, and b's where a has none; both differing: b's is not enforced
    for (const json::Value* s : {&a, &b})
      for (size_t i = 0; i < s->as_object().keys.size(); ++i) {
        const std::string& k = s->as_object().keys[i];
        if (!unsupported_keywords().count(k)) continue;
        if (!out.get(k)) out.set(k, s->as_object().vals[i]);
        else if (*out.get(k) != s->as_object().vals[i]) unsupported(k, path);
      }
    return out;
  }

  // ---- simple schemas
  int compile(const json::Value& s, const std::string& path) {
    if (++depth_ > 256) error(path, "schema nested too deep");
    struct Guard {
      int& d;
      ~Guard() { --d; }
    } guard{depth_};
    if (s.is_bool()) return s.as_bool() ? any() : never();
    if (!s.is_object()) error(path, "a schema is an object or a boolean");
    if (pure_ref(s)) return compile_ref(*s.get("$ref"), path);
    if (s.get("$ref") || s.get("allOf") || s.get("anyOf") || s.get("oneOf")) {
      std::vector<int> alts;
      for (const json::Value& t : dnf(s, path)) {
        if (pure_ref(t)) alts.push_back(compile_ref(*t.get("$ref"), path));
        else alts.push_back(compile_simple(t, path));
      }
      if (alts.empty()) return never();
      if (alts.size() == 1) return alts[0];
      Node n;
      n.kind = Kind::Alt;
      n.alts = std::move(alts);
      return g_->add(std::move(n));
    }
    return compile_simple(s, path);
  }

  int nonneg_int(const json::Value& s, const char* k, int def, const std::string& path) {
    const json::Value* v = s.get(k);
    if (!v || v->is_null()) return def;
    if (!v->is_number()) error(path, std::string(k) + ": a non-negative integer");
    double d = v->as_double();
    if (d < 0 || d != std::floor(d)) error(path, std::string(k) + ": a non-negative integer");
    return d > 1e9 ? kNoLimit - 1 : int(d);
  }

  void check_keywords(const json::Value& s, const std::string& path) {
    for (size_t i = 0; i < s.as_object().keys.size(); ++i) {
      const std::string& k = s.as_object().keys[i];
      const json::Value& v = s.as_object().vals[i];
      if (unsupported_keywords().count(k)) unsupported(k, path);
      else if (k == "uniqueItems" && v.is_bool() && v.as_bool()) unsupported("uniqueItems", path);
      else if ((k == "unevaluatedProperties" || k == "unevaluatedItems") && !v.is_bool()) unsupported(k, path);
    }
  }

  int compile_simple(const json::Value& s0, const std::string& path) {
    const json::Value s = normalize_bounds(s0);
    check_keywords(s, path);
    int ts = type_set(s, path);
    if (!s.get("type")) {
      int implied = 0;
      for (const std::string& k : s.as_object().keys) implied |= implied_type(k);
      if (implied) ts = implied | (s.get_bool("nullable", false) ? T_NULL : 0);
    }
    if (s.get("const") || s.get("enum")) {
      std::vector<json::Value> values;
      bool set = false;
      if (const json::Value* c = s.get("const")) values = {*c}, set = true;
      if (const json::Value* e = s.get("enum")) {
        if (!e->is_array()) error(path, "enum: an array of values");
        if (!set) values = e->as_array();
        else if (std::find(e->as_array().begin(), e->as_array().end(), values[0]) == e->as_array().end()) values.clear();
      }
      const int filter = typed(without(s, {"const", "enum"}), ts, path);
      return literals(values, filter);
    }
    return typed(s, ts, path);
  }

  int typed(const json::Value& s, int ts, const std::string& path) {
    if (ts == T_ALL && !has_constraints(s)) return any();
    std::vector<int> alts;
    if (ts & T_OBJ) alts.push_back(object_node(s, path));
    if (ts & T_ARR) alts.push_back(array_node(s, path));
    if (ts & T_STR) alts.push_back(string_node(s, path));
    if (ts & (T_NUM | T_INT)) alts.push_back(number_node(s, !(ts & T_NUM), path));
    if (ts & T_BOOL) alts.push_back(literal_node({"true", "false"}));
    if (ts & T_NULL) alts.push_back(literal_node({"null"}));
    if (alts.size() == 1) return alts[0];
    Node n;
    n.kind = Kind::Alt;
    n.alts = std::move(alts);
    return g_->add(std::move(n));
  }

  // A schema node or -1 for false.
  int sub_or_none(const json::Value& v, const std::string& path) {
    if (v.is_bool()) return v.as_bool() ? any() : -1;
    return compile(v, path);
  }

  int object_node(const json::Value& s, const std::string& path) {
    Node n;
    n.kind = Kind::Object;
    const json::Value* props = s.get("properties");
    if (props && !props->is_object()) error(path, "properties: an object");
    std::vector<std::string> required;
    if (const json::Value* r = s.get("required")) {
      if (!r->is_array()) error(path, "required: an array of property names");
      for (const json::Value& x : r->as_array()) {
        if (!x.is_string()) error(path, "required: property names (strings)");
        if (std::find(required.begin(), required.end(), x.as_string()) == required.end()) required.push_back(x.as_string());
      }
    }
    if (props)
      for (size_t i = 0; i < props->as_object().keys.size(); ++i) {
        const std::string& k = props->as_object().keys[i];
        Prop p;
        p.name = k;
        p.required = std::find(required.begin(), required.end(), k) != required.end();
        p.node = compile(props->as_object().vals[i], path + "/properties/" + ptr_escape(k));
        n.props.push_back(std::move(p));
      }
    const json::Value* add = s.get("additionalProperties");
    const json::Value* uneval = s.get("unevaluatedProperties");
    int add_node = -1;  // the schema of other keys (validation: absent = any)
    if (add) add_node = sub_or_none(*add, path + "/additionalProperties");
    else if (uneval && uneval->is_bool() && !uneval->as_bool()) add_node = -1;
    else add_node = any();
    // Generated: other keys only when the schema asks for them, or names no property.
    n.additional = add ? add_node : (n.props.empty() ? add_node : -1);
    for (const std::string& r : required)
      if (std::none_of(n.props.begin(), n.props.end(), [&](const Prop& p) { return p.name == r; })) {
        Prop p;
        p.name = r;
        p.required = true;
        p.node = add_node >= 0 ? add_node : never();
        n.props.push_back(std::move(p));
      }
    n.min_props = nonneg_int(s, "minProperties", 0, path);
    n.max_props = nonneg_int(s, "maxProperties", kNoLimit, path);
    if (n.props.size() > 65000) error(path, "properties: more than 65,000");
    return add_object(std::move(n));
  }

  int array_node(const json::Value& s, const std::string& path) {
    Node n;
    n.kind = Kind::Array;
    const json::Value* pre = s.get("prefixItems");
    const json::Value* items = s.get("items");
    const json::Value* uneval = s.get("unevaluatedItems");
    const bool closed = uneval && uneval->is_bool() && !uneval->as_bool();
    auto tuple = [&](const json::Value& arr, const std::string& where) {
      if (!arr.is_array()) error(path, where + ": an array of schemas");
      for (size_t i = 0; i < arr.size(); ++i) n.prefix.push_back(compile(arr[i], path + "/" + where + "/" + std::to_string(i)));
    };
    if (pre) {
      tuple(*pre, "prefixItems");
      n.items = items ? sub_or_none(*items, path + "/items") : (closed ? -1 : any());
    } else if (items && items->is_array()) {
      tuple(*items, "items");
      const json::Value* more = s.get("additionalItems");
      n.items = more ? sub_or_none(*more, path + "/additionalItems") : (closed ? -1 : any());
    } else {
      n.items = items ? sub_or_none(*items, path + "/items") : (closed ? -1 : any());
    }
    n.min_items = nonneg_int(s, "minItems", 0, path);
    n.max_items = nonneg_int(s, "maxItems", kNoLimit, path);
    return g_->add(std::move(n));
  }

  int string_node(const json::Value& s, const std::string& path) {
    Node n;
    n.kind = Kind::String;
    n.min_len = nonneg_int(s, "minLength", 0, path);
    n.max_len = nonneg_int(s, "maxLength", kNoLimit, path);
    return g_->add(std::move(n));
  }

  int number_node(const json::Value& s, bool integer, const std::string& path) {
    Node n;
    n.kind = Kind::Number;
    n.integer = integer;
    auto read = [&](const char* k, bool exclusive, Bound& b) {
      const json::Value* v = s.get(k);
      if (!v || v->is_null()) return false;
      if (!to_decimal(*v, b)) error(path, std::string(k) + ": a finite number");
      b.set = true;
      b.exclusive = exclusive;
      return true;
    };
    // The stricter of minimum / exclusiveMinimum (equal: exclusive), likewise above.
    Bound lo, hi, x;
    if (read("minimum", false, x)) lo = x;
    if (read("exclusiveMinimum", true, x) && (!lo.set || cmp_value(x, lo) >= 0)) lo = x;
    if (read("maximum", false, x)) hi = x;
    if (read("exclusiveMaximum", true, x) && (!hi.set || cmp_value(x, hi) <= 0)) hi = x;
    constexpr __int128 kLim = __int128(999999999999999999ll);  // 18 digits: what a bounded integer can be
    if (integer) {
      // Inclusive integer bounds.
      __int128 l = 0, h = 0;
      bool lset = false, hset = false;
      if (lo.set && round_dec(lo, true, l)) {
        if (lo.exclusive && cmp_value(lo, int_bound(l)) == 0) l += 1;
        lset = l > -kLim;
        if (l > kLim) return never();
      } else if (lo.set && !lo.neg) {
        return never();
      }
      if (hi.set && round_dec(hi, false, h)) {
        if (hi.exclusive && cmp_value(hi, int_bound(h)) == 0) h -= 1;
        hset = h < kLim;
        if (h < -kLim) return never();
      } else if (hi.set && hi.neg) {
        return never();
      }
      if (lset && hset && l > h) return never();
      if (lset) n.lo = int_bound(l);
      if (hset) n.hi = int_bound(h);
    } else {
      if (lo.set && hi.set) {
        const int c = cmp_value(lo, hi);
        if (c > 0 || (c == 0 && (lo.exclusive || hi.exclusive))) return never();
      }
      n.lo = lo;
      n.hi = hi;
    }
    return g_->add(std::move(n));
  }

  int const_node(const json::Value& v) {
    if (v.is_object()) {
      Node n;
      n.kind = Kind::Object;
      for (size_t i = 0; i < v.as_object().keys.size(); ++i) {
        Prop p;
        p.name = v.as_object().keys[i];
        p.required = true;
        p.node = const_node(v.as_object().vals[i]);
        n.props.push_back(std::move(p));
      }
      return add_object(std::move(n));
    }
    if (v.is_array()) {
      Node n;
      n.kind = Kind::Array;
      for (const json::Value& x : v.as_array()) n.prefix.push_back(const_node(x));
      n.min_items = n.max_items = int(v.size());
      return g_->add(std::move(n));
    }
    return literal_node({canonical_text(v)});
  }

  // enum / const values the rest of the schema (`filter`) accepts.
  int literals(const std::vector<json::Value>& values, int filter) {
    std::vector<std::string> scalars;
    std::vector<json::Value> structured;
    {
      const int saved = g_->root;
      g_->root = filter;
      const Automaton a(*g_);
      for (const json::Value& v : values) {
        const std::string t = canonical_text(v);
        if (!a.accepts_text(t)) continue;
        if (v.is_object() || v.is_array()) structured.push_back(v);
        else if (std::find(scalars.begin(), scalars.end(), t) == scalars.end()) scalars.push_back(t);
      }
      g_->root = saved;
    }
    std::vector<int> alts;
    if (!scalars.empty()) alts.push_back(literal_node(scalars));
    for (const json::Value& v : structured) alts.push_back(const_node(v));
    if (alts.empty()) return never();
    if (alts.size() == 1) return alts[0];
    Node n;
    n.kind = Kind::Alt;
    n.alts = std::move(alts);
    return g_->add(std::move(n));
  }

  // ---- completability: a node is kept only if some finite value of it
  // exists (least fixpoint); alternatives, optional properties and items
  // without one are dropped, so the automaton never reaches a dead end.
  void finish() {
    std::vector<Node>& N = g_->nodes;
    std::vector<char> ok(N.size(), 0);
    auto item_ok = [&](const Node& n, size_t i) {
      const int it = i < n.prefix.size() ? n.prefix[i] : n.items;
      return it >= 0 && ok[size_t(it)];
    };
    for (bool changed = true; changed;) {
      changed = false;
      for (size_t i = 0; i < N.size(); ++i) {
        if (ok[i]) continue;
        const Node& n = N[i];
        bool v = false;
        switch (n.kind) {
          case Kind::Never:
            break;
          case Kind::Literal:
          case Kind::Number:
            v = true;
            break;
          case Kind::String:
            v = n.min_len <= n.max_len;
            break;
          case Kind::Alt:
            v = std::any_of(n.alts.begin(), n.alts.end(), [&](int a) { return ok[size_t(a)]; });
            break;
          case Kind::Array: {
            v = n.min_items <= n.max_items;
            for (size_t k = 0; v && k < n.prefix.size() && int(k) < n.min_items; ++k) v = item_ok(n, k);
            if (v && size_t(n.min_items) > n.prefix.size()) v = n.items >= 0 && ok[size_t(n.items)];
            break;
          }
          case Kind::Object: {
            int req = 0, can = 0;
            v = true;
            for (const Prop& p : n.props) {
              if (p.required) {
                ++req;
                v = v && ok[size_t(p.node)];
              }
              can += ok[size_t(p.node)];
            }
            const bool more = n.additional >= 0 && ok[size_t(n.additional)];
            v = v && req <= n.max_props && (more || can >= n.min_props) && n.min_props <= n.max_props;
            break;
          }
        }
        if (v) {
          ok[i] = 1;
          changed = true;
        }
      }
    }
    if (!ok[size_t(g_->root)])
      throw SchemaError("the schema admits no value the server can generate (contradictory keywords, an empty enum, "
                        "a required property that is not allowed, or a recursion that never ends)");
    for (size_t i = 0; i < N.size(); ++i) {
      Node& n = N[i];
      if (n.kind == Kind::Alt) {
        n.alts.erase(std::remove_if(n.alts.begin(), n.alts.end(), [&](int a) { return !ok[size_t(a)]; }), n.alts.end());
      } else if (n.kind == Kind::Object) {
        const size_t before = n.props.size();
        n.props.erase(std::remove_if(n.props.begin(), n.props.end(), [&](const Prop& p) { return !p.required && !ok[size_t(p.node)]; }),
                      n.props.end());
        if (n.additional >= 0 && !ok[size_t(n.additional)]) n.additional = -1;
        if (n.props.size() != before) finish_object(n);
      } else if (n.kind == Kind::Array) {
        for (size_t k = 0; k < n.prefix.size(); ++k)
          if (!ok[size_t(n.prefix[k])]) {
            n.prefix.resize(k);
            n.items = -1;
            n.max_items = std::min(n.max_items, int(k));
            break;
          }
        if (n.items >= 0 && !ok[size_t(n.items)]) {
          n.items = -1;
          n.max_items = std::min(n.max_items, int(n.prefix.size()));
        }
      }
    }
    // Alternatives of alternatives become one list of value nodes: an anyOf
    // that refers back to itself before any value opens (A = anyOf[A, null])
    // adds nothing, and the automaton never expands a cycle.
    for (size_t i = 0; i < N.size(); ++i) {
      if (N[i].kind != Kind::Alt) continue;
      std::vector<int> leaves, stack(N[i].alts.rbegin(), N[i].alts.rend());
      std::vector<char> seen(N.size(), 0);
      seen[i] = 1;
      while (!stack.empty()) {
        const int a = stack.back();
        stack.pop_back();
        if (seen[size_t(a)]) continue;
        seen[size_t(a)] = 1;
        if (N[size_t(a)].kind == Kind::Alt) stack.insert(stack.end(), N[size_t(a)].alts.rbegin(), N[size_t(a)].alts.rend());
        else leaves.push_back(a);
      }
      N[i].alts = std::move(leaves);
    }
  }

  const json::Value& root_;
  SchemaOptions opt_;
  std::vector<std::string>* un_;
  std::shared_ptr<Grammar> g_;
  std::unordered_map<std::string, int> refs_;
  std::vector<std::string> inlining_;
  int any_ = -1, never_ = -1, depth_ = 0;
};

}  // namespace

std::shared_ptr<const Grammar> compile_schema(const json::Value& schema, const SchemaOptions& opt,
                                              std::vector<std::string>* unenforced) {
  Compiler c(schema, opt, unenforced);
  return c.run();
}

std::shared_ptr<const Grammar> any_object_grammar() {
  json::Value s = json::Value::object();
  s.set("type", "object");
  return compile_schema(s, SchemaOptions{});
}

}  // namespace espetero::grammar
