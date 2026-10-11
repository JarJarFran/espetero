// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Structured output: the JSON Schema compiler, the byte-level
// automaton and the token masks.
//   build/grammar_test [tokenizer_dir | model.gguf] [samples.jsonl]
// (1) Schemas: texts each must accept or reject; refusals (strict) and
// unenforced keywords (non-strict); unsatisfiable schemas. (2) Random
// documents walked byte by byte through each schema's automaton, checked by
// json::parse and by an independent validator. (3) With a real tokenizer
// (the cached mlx-community/Qwen3.5-0.8B-4bit one unless a directory is
// given; skipped with a message when there is none): every mask against the
// automaton token by token (the whole vocabulary for some states), random
// token-level documents, the reasoning phase and tool-call escapes, the
// masked argmax and distributions against distribution() over -inf-masked
// logits, and the cost of a mask. The generated documents go to
// samples.jsonl for tests/ref/grammar_check.py (Python jsonschema).
#include <dirent.h>
#include <sys/stat.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "../src/engine/grammar.h"
#include "../src/engine/json_schema.h"
#include "../src/engine/sampling.h"
#include "../src/tok/tokenizer.h"
#include "../src/util/json.h"

using namespace espetero;
using namespace espetero::grammar;

static int fails = 0;
#define CHECK(c, ...)                         \
  do {                                        \
    if (!(c)) {                               \
      ++fails;                                \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);               \
      std::printf("\n");                      \
    }                                         \
  } while (0)

// ---------------------------------------------------------------- validator
// An independent check of the subset (what a JSON Schema validator does).
struct Validator {
  const json::Value& root;
  int cps(const std::string& s) {
    int n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
  }
  const json::Value* resolve(const std::string& r) {
    const json::Value* v = &root;
    size_t i = 1;
    while (i < r.size()) {
      size_t j = r.find('/', i + 1);
      if (j == std::string::npos) j = r.size();
      std::string tok;
      for (size_t k = i + 1; k < j; ++k) {
        if (r[k] == '~' && k + 1 < j) tok.push_back(r[++k] == '1' ? '/' : '~');
        else tok.push_back(r[k]);
      }
      if (v->is_object()) v = v->get(tok);
      else if (v->is_array()) v = &(*v)[size_t(std::stoul(tok))];
      else return nullptr;
      if (!v) return nullptr;
      i = j;
    }
    return v;
  }
  bool type_ok(const std::string& t, const json::Value& v) {
    if (t == "object") return v.is_object();
    if (t == "array") return v.is_array();
    if (t == "string") return v.is_string();
    if (t == "number") return v.is_number();
    if (t == "integer") return v.is_int() || (v.is_double() && v.as_double() == std::floor(v.as_double()));
    if (t == "boolean") return v.is_bool();
    if (t == "null") return v.is_null();
    return false;
  }
  // (schema, value) pairs being checked: a reference back to one of them
  // (anyOf: [{"$ref": itself}, ...]) adds nothing.
  std::vector<std::pair<const json::Value*, const json::Value*>> open;
  bool ok(const json::Value& s, const json::Value& v) {
    if (s.is_bool()) return s.as_bool();
    const std::pair<const json::Value*, const json::Value*> key{&s, &v};
    if (std::find(open.begin(), open.end(), key) != open.end()) return false;
    open.push_back(key);
    const bool r = check(s, v);
    open.pop_back();
    return r;
  }
  bool check(const json::Value& s, const json::Value& v) {
    if (const json::Value* r = s.get("$ref")) {
      const json::Value* t = resolve(r->as_string());
      if (!t || !ok(*t, v)) return false;
    }
    if (const json::Value* t = s.get("type")) {
      bool any = false;
      if (t->is_string()) any = type_ok(t->as_string(), v);
      else
        for (const json::Value& x : t->as_array()) any = any || type_ok(x.as_string(), v);
      if (!any && !(s.get_bool("nullable", false) && v.is_null())) return false;
    }
    if (const json::Value* e = s.get("enum")) {
      bool in = false;
      for (const json::Value& x : e->as_array()) in = in || x == v;
      if (!in) return false;
    }
    if (const json::Value* c = s.get("const"); c && !(*c == v)) return false;
    if (v.is_object()) {
      const json::Value* props = s.get("properties");
      const json::Value* add = s.get("additionalProperties");
      for (size_t i = 0; i < v.as_object().keys.size(); ++i) {
        const std::string& k = v.as_object().keys[i];
        const json::Value& x = v.as_object().vals[i];
        if (props && props->get(k)) {
          if (!ok(*props->get(k), x)) return false;
        } else if (add && !ok(*add, x)) {
          return false;
        }
      }
      if (const json::Value* r = s.get("required"))
        for (const json::Value& k : r->as_array())
          if (!v.get(k.as_string())) return false;
      if (s.get("minProperties") && int64_t(v.size()) < s.get_int("minProperties", 0)) return false;
      if (s.get("maxProperties") && int64_t(v.size()) > s.get_int("maxProperties", 0)) return false;
    }
    if (v.is_array()) {
      const json::Value* pre = s.get("prefixItems");
      const json::Value* items = s.get("items");
      size_t np = 0;
      if (pre) np = pre->size();
      else if (items && items->is_array()) np = items->size(), pre = items, items = s.get("additionalItems");
      for (size_t i = 0; i < v.size(); ++i) {
        if (i < np) {
          if (!ok((*pre)[i], v[i])) return false;
        } else if (items && !ok(*items, v[i])) {
          return false;
        }
      }
      if (s.get("minItems") && int64_t(v.size()) < s.get_int("minItems", 0)) return false;
      if (s.get("maxItems") && int64_t(v.size()) > s.get_int("maxItems", 0)) return false;
    }
    if (v.is_string()) {
      const int n = cps(v.as_string());
      if (s.get("minLength") && n < s.get_int("minLength", 0)) return false;
      if (s.get("maxLength") && n > s.get_int("maxLength", 0)) return false;
    }
    if (v.is_number()) {
      const double x = v.as_double();
      const json::Value* emin = s.get("exclusiveMinimum");
      const json::Value* emax = s.get("exclusiveMaximum");
      if (const json::Value* m = s.get("minimum")) {
        const bool ex = emin && emin->is_bool() && emin->as_bool();
        if (ex ? !(x > m->as_double()) : !(x >= m->as_double())) return false;
      }
      if (const json::Value* m = s.get("maximum")) {
        const bool ex = emax && emax->is_bool() && emax->as_bool();
        if (ex ? !(x < m->as_double()) : !(x <= m->as_double())) return false;
      }
      if (emin && emin->is_number() && !(x > emin->as_double())) return false;
      if (emax && emax->is_number() && !(x < emax->as_double())) return false;
    }
    if (const json::Value* a = s.get("allOf"))
      for (const json::Value& x : a->as_array())
        if (!ok(x, v)) return false;
    if (const json::Value* a = s.get("anyOf")) {
      bool any = false;
      for (const json::Value& x : a->as_array()) any = any || ok(x, v);
      if (!any) return false;
    }
    if (const json::Value* a = s.get("oneOf")) {
      int n = 0;
      for (const json::Value& x : a->as_array()) n += ok(x, v);
      if (n != 1) return false;
    }
    return true;
  }
};

// ---------------------------------------------------------------- cases
struct Case {
  const char* name;
  const char* schema;
  std::vector<const char*> accept, reject;
};

static std::vector<Case> cases() {
  return {
      {"person", R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false})",
       {R"({"name":"Ann","age":31})", "{\n  \"name\": \"Bo\",\n  \"age\": -2\n}", "\n{\"name\":\"\",\"age\":0}", "{\n\t\"name\": \"x\",\n\t\"age\": 1\n}"},
       {R"({"age":31,"name":"Ann"})", R"({"name":"Ann"})", R"({"name":"Ann","age":31,"x":1})", R"({"name":"Ann","age":3.5})",
        R"({"name":"Ann","age":031})", R"({"name":"Ann","name":"B","age":1})", R"({"name":"A\/B","age":1})",
        R"({"name":"Ann","age":31} )", R"({"name":"Ann",  "age":31})", "{\n\n\"name\":\"A\",\"age\":1}", "{\r\n\"name\":\"A\",\"age\":1}",
        "{\t\"name\":\"A\",\"age\":1}"}},
      {"nested", R"({"type":"object","properties":{"id":{"type":"integer","minimum":1},"address":{"type":"object","properties":{"city":{"type":"string","minLength":1},"zip":{"type":["string","null"]}},"required":["city","zip"]},"tags":{"type":"array","items":{"type":"string","maxLength":8},"minItems":1,"maxItems":3}},"required":["id","address","tags"]})",
       {R"({"id":7,"address":{"city":"Lugo","zip":null},"tags":["a","bb"]})", R"({"id":1,"address":{"city":"É","zip":"27001"},"tags":["x","y","z"]})"},
       {R"({"id":0,"address":{"city":"L","zip":null},"tags":["a"]})", R"({"id":7,"address":{"city":"","zip":null},"tags":["a"]})",
        R"({"id":7,"address":{"city":"L","zip":null},"tags":[]})", R"({"id":7,"address":{"city":"L","zip":null},"tags":["a","b","c","d"]})",
        R"({"id":7,"address":{"city":"L","zip":null},"tags":["123456789"]})"}},
      {"enum", R"({"type":"object","properties":{"color":{"enum":["red","green","blue"]},"size":{"type":"string","enum":["S","M","L",7]}},"required":["color","size"]})",
       {R"({"color":"green","size":"M"})"}, {R"({"color":"pink","size":"M"})", R"({"color":"green","size":7})", R"({"color":"gree","size":"M"})"}},
      {"anyof", R"({"anyOf":[{"type":"string","maxLength":5},{"type":"integer","minimum":0,"maximum":100},{"type":"null"}]})",
       {R"("hello")", "42", "null", "0", "100"}, {R"("hello!")", "101", "-1", "4.5", "true"}},
      {"ref", R"({"$defs":{"pt":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"}},"required":["x","y"],"additionalProperties":false}},"type":"object","properties":{"a":{"$ref":"#/$defs/pt"},"b":{"$ref":"#/$defs/pt"}},"required":["a","b"]})",
       {R"({"a":{"x":1,"y":2.5},"b":{"x":-0.5,"y":1e3}})"}, {R"({"a":{"x":1},"b":{"x":1,"y":2}})", R"({"a":{"x":1,"y":2}})"}},
      {"tree", R"({"$defs":{"node":{"type":"object","properties":{"value":{"type":"integer"},"children":{"type":"array","items":{"$ref":"#/$defs/node"},"maxItems":3}},"required":["value"]}},"$ref":"#/$defs/node"})",
       {R"({"value":1})", R"({"value":1,"children":[{"value":2},{"value":3,"children":[]}]})"},
       {R"({"children":[]})", R"({"value":1,"children":[{"value":2},{"value":3},{"value":4},{"value":5}]})"}},
      {"optional", R"({"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"integer"},"c":{"type":"string"},"d":{"type":"null"}},"required":["a","d"]})",
       {R"({"a":true,"d":null})", R"({"a":false,"b":1,"d":null})", R"({"a":false,"c":"x","d":null})", R"({"a":true,"b":2,"c":"","d":null})"},
       {R"({"a":true})", R"({"a":true,"c":"x","b":1,"d":null})", R"({"d":null,"a":true})"}},
      {"nullable", R"({"type":"object","properties":{"p":{"type":"string","nullable":true},"q":{"anyOf":[{"type":"number"},{"type":"null"}]}},"required":["p","q"]})",
       {R"({"p":null,"q":null})", R"({"p":"x","q":1.5})"}, {R"({"p":1,"q":null})", R"({"p":null,"q":"1"})"}},
      {"const", R"({"type":"object","properties":{"kind":{"const":"circle"},"r":{"type":"number","exclusiveMinimum":0}},"required":["kind","r"]})",
       {R"({"kind":"circle","r":0.5})", R"({"kind":"circle","r":3})"}, {R"({"kind":"square","r":1})", R"({"kind":"circle","r":0})", R"({"kind":"circle","r":-1})"}},
      {"oneof", R"({"oneOf":[{"type":"object","properties":{"kind":{"const":"a"},"x":{"type":"integer"}},"required":["kind","x"],"additionalProperties":false},{"type":"object","properties":{"kind":{"const":"b"},"y":{"type":"string"}},"required":["kind","y"],"additionalProperties":false}]})",
       {R"({"kind":"a","x":1})", R"({"kind":"b","y":"z"})"}, {R"({"kind":"a","y":"z"})", R"({"kind":"c","x":1})"}},
      {"array_of_objects", R"({"type":"array","items":{"type":"object","properties":{"q":{"type":"string"},"n":{"type":"integer","minimum":1,"maximum":5}},"required":["q","n"]},"minItems":2,"maxItems":4})",
       {R"([{"q":"a","n":1},{"q":"b","n":5}])"}, {R"([{"q":"a","n":1}])", R"([{"q":"a","n":0},{"q":"b","n":1}])", R"([{"q":"a","n":6},{"q":"b","n":1}])"}},
      {"lengths", R"({"type":"string","minLength":2,"maxLength":4})", {R"("ab")", R"("ñáé€")", R"("\n\t")", R"("a\u001fb")"},
       {R"("a")", R"("abcde")", R"("ñáé€😀")", "\"\\u0041b\"", R"("a\u000ab")", R"("a\u001Fb")"}},
      {"allof", R"({"allOf":[{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]},{"type":"object","properties":{"b":{"type":"string"}},"required":["b"]}]})",
       {R"({"a":1,"b":"x"})"}, {R"({"a":1})", R"({"b":"x"})"}},
      {"additional_schema", R"({"type":"object","additionalProperties":{"type":"integer"},"minProperties":1,"maxProperties":3})",
       {R"({"x":1})", R"({"x":1,"y":2,"z":3})"}, {R"({})", R"({"x":"1"})", R"({"a":1,"b":2,"c":3,"d":4})"}},
      {"number_range", R"({"type":"number","minimum":-1.5,"maximum":2.25})", {"-1.5", "0", "2.25", "-0.5", "1.999", "-0", "0.0"},
       {"2.26", "-1.51", "3", "1e0", "-2", "10"}},
      {"integer_range", R"({"type":"integer","minimum":10,"exclusiveMaximum":300.5})", {"10", "300", "150", "99"},
       {"9", "301", "0", "-5", "1000", "3", "30.0"}},
      {"tuple", R"({"type":"array","prefixItems":[{"type":"string"},{"type":"integer"}],"items":false,"minItems":2})",
       {R"(["a",1])"}, {R"(["a"])", R"(["a",1,2])", R"([1,"a"])"}},
      {"json_object", R"({"type":"object"})", {R"({})", R"({"a":[1,{"b":null}],"c":"d","e":true,"f":-1.5e-3})"}, {R"([])", R"("x")", "1"}},
      {"mixed_enum", R"({"enum":[1,"a",null,true,{"k":[1,2]}]})", {"1", R"("a")", "null", "true", R"({"k":[1,2]})", R"({ "k" : [ 1 , 2 ] })"},
       {"2", "false", R"({"k":[1]})", "1.0"}},
      {"recursive_anyof", R"({"$defs":{"v":{"anyOf":[{"$ref":"#/$defs/v"},{"type":"array","items":{"$ref":"#/$defs/v"},"maxItems":2},{"type":"null"}]}},"$ref":"#/$defs/v"})",
       {"null", "[]", "[null,[null]]"}, {"[1]", "[null,null,null]"}},
      {"boolean_root", R"({"type":"boolean"})", {"true", "false"}, {"null", "1"}},
  };
}

// A random document of the automaton, byte by byte: random allowed bytes,
// more and more often the closing ones as it grows. Empty when it runs too long.
static std::string walk(const Automaton& a, std::mt19937& rng, int max_len = 600) {
  Pool pool;
  std::vector<int32_t> tops{a.start(pool)}, next;
  std::string text;
  static const std::string closing = "\"}]0nulltruefalse";
  while (int(text.size()) < max_len) {
    bool acc = false;
    for (int32_t t : tops) acc = acc || a.accepting(pool, t);
    if (acc && (rng() % 3 == 0 || text.size() > 200)) return text;
    std::vector<int> allowed;
    for (int c = 0; c < 256; ++c) {
      next.clear();
      for (int32_t t : tops) a.step(pool, t, uint8_t(c), next);
      if (!next.empty()) allowed.push_back(c);
    }
    if (allowed.empty()) return acc ? text : std::string("<dead end>");
    int c = allowed[rng() % allowed.size()];
    const double close_p = std::min(0.95, double(text.size()) / 120.0);
    if (std::uniform_real_distribution<double>(0, 1)(rng) < close_p) {
      for (char k : closing)
        if (std::find(allowed.begin(), allowed.end(), int(uint8_t(k))) != allowed.end()) {
          c = uint8_t(k);
          break;
        }
    } else if ((c == ' ' || c == '\n' || c == '\t' || c == '\r') && rng() % 4) {
      continue;  // less whitespace
    }
    next.clear();
    for (int32_t t : tops) a.step(pool, t, uint8_t(c), next);
    tops = next;
    text.push_back(char(c));
  }
  return std::string();
}

static std::string find_tokenizer() {
  std::string hub;
  if (const char* h = std::getenv("HF_HUB_CACHE"); h && *h) hub = h;
  else if (const char* h2 = std::getenv("HF_HOME"); h2 && *h2) hub = std::string(h2) + "/hub";
  else hub = std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/.cache/huggingface/hub";
  const std::string snaps = hub + "/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots";
  DIR* d = opendir(snaps.c_str());
  if (!d) return "";
  std::string found;
  while (dirent* e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    const std::string dir = snaps + "/" + e->d_name;
    struct stat st;
    if (::stat((dir + "/tokenizer.json").c_str(), &st) == 0) found = dir;
  }
  closedir(d);
  return found;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string samples_path = argc > 2 ? argv[2] : "build/grammar_samples.jsonl";
  std::ofstream samples(samples_path);
  std::mt19937 rng(12345);

  // ---- (1) texts each schema accepts or rejects; (2) random documents
  int texts = 0, docs = 0;
  for (const Case& c : cases()) {
    const json::Value schema = json::parse(c.schema);
    if (std::getenv("GRAMMAR_TEST_TRACE")) std::fprintf(stderr, "case %s\n", c.name);
    std::vector<std::string> notes;
    std::shared_ptr<const Grammar> g;
    try {
      g = compile_schema(schema, SchemaOptions{true}, &notes);
    } catch (const std::exception& e) {
      CHECK(false, "%s: does not compile: %s", c.name, e.what());
      continue;
    }
    const Automaton a(*g);
    Validator val{schema};
    for (const char* t : c.accept) {
      CHECK(a.accepts_text(t), "%s: rejects %s", c.name, t);
      CHECK(val.ok(schema, json::parse(t)), "%s: the validator rejects %s", c.name, t);
      ++texts;
    }
    for (const char* t : c.reject) {
      CHECK(!a.accepts_text(t), "%s: accepts %s", c.name, t);
      ++texts;
    }
    for (int k = 0; k < 40; ++k) {
      const std::string doc = walk(a, rng);
      if (doc.empty()) continue;
      CHECK(doc != "<dead end>", "%s: a dead end", c.name);
      if (doc == "<dead end>") continue;
      ++docs;
      json::Value v;
      try {
        v = json::parse(doc);
      } catch (const std::exception& e) {
        CHECK(false, "%s: invalid JSON %s (%s)", c.name, doc.c_str(), e.what());
        continue;
      }
      CHECK(val.ok(schema, v), "%s: not valid under the schema: %s", c.name, doc.c_str());
      json::Value line = json::Value::object();
      line.set("case", c.name);
      line.set("schema", schema);
      line.set("text", doc);
      samples << json::dump(line) << "\n";
    }
  }
  std::printf("schemas: %zu, texts %d, random documents %d\n", cases().size(), texts, docs);

  // ---- canonical spellings and caps
  {
    const std::shared_ptr<const Grammar> any_g = compile_schema(json::parse("{}"), SchemaOptions{});
    const Automaton any(*any_g);
    for (const char* t : {R"("\/")", "\"\\u0041\"", R"("\u000a")", "\"\\uD83D\\uDE00\"", "\"\x01\"", "\"\xC0\x80\"",
                          "\"\xED\xA0\x80\"", "\"\x80\"", "\"\xF5\x80\x80\x80\"", "01", "1.", "1.e5", ".5", "+1", "NaN", "Infinity",
                          "[1,]", "{\"a\":1,}", "tru", "nulll"})
      CHECK(!any.accepts_text(t), "any: accepts %s", t);
    for (const char* t : {R"("\u001f")", R"("\u000b")", "\"\xC3\xA9\"", "\"\xF0\x9F\x98\x80\"", "-0", "1e5", "1E-05", "0.5", "[]", "{}",
                          "[1, 2, [3]]", "\"\x7f\""})
      CHECK(any.accepts_text(t), "any: rejects %s", t);
    std::string ws(kMaxWhitespace, ' ');
    CHECK(any.accepts_text("[\n" + ws + "1]"), "whitespace: rejects %d spaces of indentation", kMaxWhitespace);
    CHECK(!any.accepts_text("[\n" + ws + " 1]"), "whitespace: accepts %d spaces of indentation", kMaxWhitespace + 1);
    CHECK(any.accepts_text("\n\n[]"), "whitespace: rejects two newlines before the value");
    CHECK(!any.accepts_text("\n\n\n[]") && !any.accepts_text(" []") && !any.accepts_text("\t[]"),
          "whitespace: accepts more than two newlines before the value");
    CHECK(!any.accepts_text("[] "), "whitespace: accepts trailing whitespace");
    CHECK(any.accepts_text(std::string(kMaxIntDigits, '9')), "digits: rejects %d digits", kMaxIntDigits);
    CHECK(!any.accepts_text(std::string(kMaxIntDigits + 1, '9')), "digits: accepts %d digits", kMaxIntDigits + 1);
    std::string deep;
    for (int i = 0; i < kMaxDepth - 1; ++i) deep += "[";
    std::string deep_closed = deep;
    for (int i = 0; i < kMaxDepth - 1; ++i) deep_closed += "]";
    CHECK(any.accepts_text(deep_closed), "depth: rejects %d arrays", kMaxDepth - 1);
    CHECK(!any.accepts_text(deep + "[]" + deep_closed.substr(deep.size())), "depth: accepts %d arrays", kMaxDepth);
  }

  // ---- refusals and notes
  {
    auto refused = [](const char* s, bool strict) {
      try {
        compile_schema(json::parse(s), SchemaOptions{strict});
      } catch (const SchemaError& e) {
        return std::string(e.what());
      }
      return std::string();
    };
    for (const char* s : {R"({"type":"string","pattern":"^a"})", R"({"type":"string","format":"date"})", R"({"type":"number","multipleOf":2})",
                          R"({"type":"array","uniqueItems":true})", R"({"type":"object","patternProperties":{"^x":{}}})",
                          R"({"not":{"type":"null"}})", R"({"if":{"type":"string"},"then":{"minLength":1}})"}) {
      const std::string e = refused(s, true);
      CHECK(!e.empty() && e.find("not enforced") != std::string::npos, "strict accepts %s", s);
      CHECK(refused(s, false).empty(), "non-strict refuses %s", s);
    }
    std::vector<std::string> notes;
    compile_schema(json::parse(R"({"type":"object","properties":{"zip":{"type":"string","pattern":"^[0-9]{5}$"}}})"), SchemaOptions{false}, &notes);
    CHECK(notes.size() == 1 && notes[0] == "pattern at #/properties/zip", "notes: %s", notes.empty() ? "none" : notes[0].c_str());
    for (const char* s : {R"({"type":"string","minLength":5,"maxLength":2})", R"({"enum":[]})", R"({"type":"integer","minimum":3,"maximum":2.5})",
                          R"({"type":"object","properties":{},"required":["x"],"additionalProperties":false})",
                          R"({"$defs":{"a":{"type":"object","properties":{"x":{"$ref":"#/$defs/a"}},"required":["x"]}},"$ref":"#/$defs/a"})",
                          R"({"$defs":{"a":{"anyOf":[{"$ref":"#/$defs/a"}]}},"$ref":"#/$defs/a"})", R"({"type":"color"})",
                          R"({"$ref":"other.json#/x"})", R"({"$ref":"#/$defs/missing"})", R"({"allOf":[{"type":"string"},{"type":"integer"}]})",
                          R"({"type":"number","exclusiveMinimum":1,"exclusiveMaximum":1})"})
      CHECK(!refused(s, false).empty(), "accepts the impossible or invalid %s", s);
  }

  // ---- (3) masks over a real vocabulary
  std::string dir = argc > 1 && argv[1][0] ? std::string(argv[1]) : find_tokenizer();
  if (dir.empty()) {
    std::printf("mask tests skipped: no tokenizer (pass a directory with tokenizer.json, or cache "
                "mlx-community/Qwen3.5-0.8B-4bit)\n");
    std::printf(fails ? "FAILED: %d\n" : "OK\n", fails);
    return fails ? 1 : 0;
  }
  const bool gguf = dir.size() > 5 && dir.compare(dir.size() - 5, 5, ".gguf") == 0;
  std::unique_ptr<Tokenizer> tok = gguf ? Tokenizer::from_gguf(dir) : Tokenizer::from_hf_dir(dir);
  const int n_model = (tok->vocab_size() + 255) / 256 * 256;  // padded like an LM head
  auto vocab = std::make_shared<Vocab>(*tok, n_model);
  std::printf("vocabulary: %d ids (%d tokens), %zu trie nodes, %zu exit tokens, built in %.0f ms\n", vocab->size(), tok->vocab_size(),
              vocab->trie_nodes(), vocab->exits.size(), vocab->build_ms);
  const int32_t im_end = tok->token_id("<|im_end|>"), eot = tok->token_id("<|endoftext|>");
  const int32_t think_end = tok->token_id("</think>"), tool_call = tok->token_id("<tool_call>");
  const std::vector<int32_t> ends{im_end, eot};
  int full_checks = 0, sampled_checks = 0, tdocs = 0, masks = 0;
  double mask_ms = 0, max_ms = 0;
  std::uniform_int_distribution<int> any_id(0, n_model - 1);
  for (const Case& c : cases()) {
    const json::Value schema = json::parse(c.schema);
    auto compiled = std::make_shared<Compiled>(compile_schema(schema, SchemaOptions{true}), vocab, ends, std::vector<int32_t>{},
                                               std::vector<std::string>{});
    auto cons = std::make_shared<Constraint>();
    cons->compiled = compiled;
    Validator val{schema};
    for (int k = 0; k < 6; ++k) {
      Matcher m(cons);
      std::vector<int32_t> ids;
      bool ended = false;
      for (int step = 0; step < 400 && !ended; ++step) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto mk = m.mask();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        mask_ms += ms;
        max_ms = std::max(max_ms, ms);
        ++masks;
        CHECK(mk->count > 0, "%s: an empty mask after %zu tokens", c.name, ids.size());
        if (mk->count == 0) break;
        // Mask against the automaton: the whole vocabulary at a few states, else a sample.
        auto agrees = [&](int32_t t) {
          const bool brute = m.valid_prefix(&t, 1) == 1;
          if (brute != mk->allows(t)) {
            CHECK(false, "%s: token %d (%s) mask %d automaton %d after \"%s\"", c.name, t, json::dump(json::Value(tok->piece(t))).c_str(),
                  int(mk->allows(t)), int(brute), tok->decode(ids).c_str());
            return false;
          }
          return true;
        };
        if (k == 0 && step < 3) {
          for (int32_t t = 0; t < n_model; ++t)
            if (!agrees(t)) break;
          ++full_checks;
        } else {
          for (int j = 0; j < 200; ++j) agrees(any_id(rng));
          std::vector<int32_t> in;
          mk->for_each([&](int32_t t) { in.push_back(t); });
          for (int j = 0; j < 50; ++j) agrees(in[rng() % in.size()]);
          ++sampled_checks;
        }
        // Next token: uniform over the mask, more and more often one that ends or closes something.
        std::vector<int32_t> in;
        mk->for_each([&](int32_t t) { in.push_back(t); });
        int32_t t = in[rng() % in.size()];
        const double close_p = std::min(0.9, double(ids.size()) / 60.0);
        if (std::uniform_real_distribution<double>(0, 1)(rng) < close_p) {
          std::vector<int32_t> pref;
          for (int32_t x : in) {
            const std::string& b = vocab->bytes(x);
            if (x == im_end || (!b.empty() && b.size() <= 2 && (b.find('"') != std::string::npos || b.find('}') != std::string::npos ||
                                                                 b.find(']') != std::string::npos)))
              pref.push_back(x);
          }
          if (!pref.empty()) t = pref[rng() % pref.size()];
        }
        CHECK(m.advance(t), "%s: advance refuses an allowed token", c.name);
        if (t == im_end || t == eot) ended = true;
        else ids.push_back(t);
      }
      if (!ended) continue;  // ran out of tokens: not a finished document
      CHECK(m.done(), "%s: ended but not done", c.name);
      const std::string doc = tok->decode(ids);
      ++tdocs;
      json::Value v;
      try {
        v = json::parse(doc);
      } catch (const std::exception& e) {
        CHECK(false, "%s: token-level document is not JSON: %s", c.name, doc.c_str());
        continue;
      }
      CHECK(val.ok(schema, v), "%s: token-level document not valid: %s", c.name, doc.c_str());
      json::Value line = json::Value::object();
      line.set("case", c.name);
      line.set("schema", schema);
      line.set("text", doc);
      samples << json::dump(line) << "\n";
    }
  }
  std::printf("masks: %d (%d checked over the whole vocabulary, %d sampled), %.3f ms each on average, %.1f ms at most; %d token-level documents\n",
              masks, full_checks, sampled_checks, mask_ms / std::max(1, masks), max_ms, tdocs);

  // ---- reasoning first, tool-call escapes, json_object
  {
    auto compiled = std::make_shared<Compiled>(any_object_grammar(), vocab, ends, std::vector<int32_t>{tool_call},
                                               std::vector<std::string>{});
    auto cons = std::make_shared<Constraint>();
    cons->compiled = compiled;
    cons->think_end = think_end;
    Matcher m(cons);
    const std::vector<int32_t> thought = tok->encode("Let me think: an object.", false);
    CHECK(!m.allows(im_end) && !m.allows(eot), "think: the end of the turn is allowed while thinking");
    for (int32_t t : thought) CHECK(m.allows(t) && m.advance(t), "think: a reasoning token is refused");
    CHECK(m.allows(tool_call), "think: <tool_call> is refused while thinking");
    CHECK(m.advance(think_end), "think: </think> is refused");
    CHECK(!m.allows(im_end), "after think: the end is allowed before the value");
    const std::vector<int32_t> nl = tok->encode("\n\n", false);
    for (int32_t t : nl) CHECK(m.advance(t), "after think: whitespace refused");
    CHECK(m.allows(tool_call), "escape: <tool_call> refused at the start");
    const std::vector<int32_t> obj = tok->encode("{\"a\": [1, 2]}", false);
    Matcher m2 = m;
    for (int32_t t : obj) CHECK(m.advance(t), "json_object: a token of {\"a\": [1, 2]} refused");
    CHECK(m.allows(im_end) && !m.allows(tool_call), "json_object: end refused or escape allowed after the value");
    CHECK(m2.advance(tool_call) && m2.allows(im_end) && m2.allows(think_end), "escape: not free after <tool_call>");
    const std::vector<int32_t> arr = tok->encode("[1]", false);
    Matcher m3(cons);
    m3.advance(think_end);
    CHECK(!m3.allows(arr[0]), "json_object: an array is allowed");
  }

  // ---- masked argmax and distributions against distribution() over -inf-masked logits
  {
    auto compiled = std::make_shared<Compiled>(compile_schema(json::parse(cases()[0].schema), SchemaOptions{}), vocab, ends,
                                               std::vector<int32_t>{}, std::vector<std::string>{});
    auto cons = std::make_shared<Constraint>();
    cons->compiled = compiled;
    std::normal_distribution<float> nd(0.0f, 3.0f);
    int checked = 0;
    for (const char* prefix : {"", "{\"", "{\"name\": \"Al", "{\"name\": \"Al\", \"age\": 4"}) {
      Matcher m(cons);
      for (int32_t t : tok->encode(prefix, false)) m.advance(t);
      const auto mk = m.mask();
      std::vector<float> logits(static_cast<size_t>(n_model));
      for (float& x : logits) x = nd(rng);
      // Make the global argmax a disallowed token, and a few allowed ones near the top.
      int32_t bad = 0;
      while (mk->allows(bad)) ++bad;
      logits[size_t(bad)] = 30;
      std::vector<float> masked = logits;
      for (int32_t t = 0; t < n_model; ++t)
        if (!mk->allows(t)) masked[size_t(t)] = -std::numeric_limits<float>::infinity();
      Readout r;
      r.argmax = bad;
      r.logits = logits.data();
      r.vocab = n_model;
      SamplingParams greedy;
      greedy.temperature = 0;
      const int32_t want = argmax(masked.data(), n_model);
      CHECK(masked_argmax(m, r, greedy, nullptr) == want, "masked argmax %d, want %d (prefix %s)", masked_argmax(m, r, greedy, nullptr), want, prefix);
      // With GPU-like candidates: the top 32 by (logit desc, id asc).
      std::vector<int32_t> order(static_cast<size_t>(n_model));
      for (int32_t t = 0; t < n_model; ++t) order[size_t(t)] = t;
      std::partial_sort(order.begin(), order.begin() + 32, order.end(), [&](int32_t a, int32_t b) {
        return logits[size_t(a)] > logits[size_t(b)] || (logits[size_t(a)] == logits[size_t(b)] && a < b);
      });
      std::vector<int32_t> cand(order.begin(), order.begin() + 32);
      std::vector<float> cand_x(32);
      for (int j = 0; j < 32; ++j) cand_x[size_t(j)] = logits[size_t(cand[size_t(j)])];
      r.cand = cand.data();
      r.cand_x = cand_x.data();
      r.n_cand = 32;
      CHECK(masked_argmax(m, r, greedy, nullptr) == want, "masked argmax with candidates (prefix %s)", prefix);
      for (const auto& [temp, top_k, top_p, min_p] : std::vector<std::tuple<double, int, double, double>>{
               {0.7, 20, 0.95, 0.0}, {1.0, 0, 0.9, 0.0}, {0.6, 0, 1.0, 0.05}, {1.0, 5, 1.0, 0.0}, {1.3, 0, 1.0, 0.0}}) {
        SamplingParams sp;
        sp.temperature = temp;
        sp.top_k = top_k;
        sp.top_p = top_p;
        sp.min_p = min_p;
        Readout rr = r;
        rr.scaled = false;
        const Distribution got = masked_distribution(m, rr, sp, nullptr);
        Distribution ref = distribution(masked.data(), n_model, sp);
        if (ref.ids.empty()) {  // the whole vocabulary: its nonzero entries
          Distribution e;
          for (int32_t t = 0; t < n_model; ++t)
            if (ref.p[size_t(t)] > 0) e.ids.push_back(t), e.p.push_back(ref.p[size_t(t)]);
          ref = e;
        } else {
          Distribution e;
          for (size_t j = 0; j < ref.ids.size(); ++j)
            if (ref.p[j] > 0) e.ids.push_back(ref.ids[j]), e.p.push_back(ref.p[j]);
          ref = e;
        }
        double l1 = 0;
        bool same = got.ids.size() == ref.ids.size();
        if (same) {
          std::vector<std::pair<int32_t, double>> a, b;
          for (size_t j = 0; j < got.ids.size(); ++j) a.push_back({got.ids[j], got.p[j]}), b.push_back({ref.ids[j], ref.p[j]});
          std::sort(a.begin(), a.end());
          std::sort(b.begin(), b.end());
          for (size_t j = 0; j < a.size(); ++j) {
            same = same && a[j].first == b[j].first;
            l1 += std::fabs(a[j].second - b[j].second);
          }
        }
        CHECK(same && l1 < 1e-4, "masked distribution T %.1f k %d p %.2f min_p %.2f: %zu vs %zu ids, L1 %.2g (prefix %s)", temp, top_k, top_p,
              min_p, got.ids.size(), ref.ids.size(), l1, prefix);
        for (int j = 0; j < 20; ++j) {
          const int32_t t = sample_at(got, 99, j);
          CHECK(mk->allows(t), "a sampled token is not allowed");
        }
        ++checked;
      }
    }
    std::printf("masked choice: %d distributions equal to distribution() over -inf-masked logits\n", checked);
  }

  std::printf("samples: %s\n", samples_path.c_str());
  std::printf(fails ? "FAILED: %d\n" : "OK\n", fails);
  return fails ? 1 : 0;
}
