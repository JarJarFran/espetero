// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Tokenizer::from_hf_dir: tokenizer.json (+ tokenizer_config.json, chat_template.jinja).
//
// tokenizer.json is 10-20 MB, almost all of it model.vocab (~250K entries) and
// model.merges (~250K pairs). Those two are read by a small streaming scanner
// straight into flat arenas; every other (small) section is cut out as raw JSON
// and handed to espetero::json::parse. Building a generic JSON tree for the vocab
// would be slow (and espetero::json::parse is quadratic in the number of keys of
// one object, see the report), so the scanner is what keeps loading fast.
#include <algorithm>
#include <charconv>
#include <climits>
#include <cstdio>
#include <stdexcept>

#include "../util/json.h"
#include "tok_internal.h"
#include "tokenizer.h"

namespace espetero {

namespace {

using json::Value;

bool read_file(const std::string& path, std::string& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  out.resize(size > 0 ? size_t(size) : 0);
  const size_t got = out.empty() ? 0 : std::fread(out.data(), 1, out.size(), f);
  std::fclose(f);
  return got == out.size();
}

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("tokenizer: " + msg); }

// Minimal pull scanner over JSON text (RFC 8259, UTF-8 passed through).
class Scanner {
 public:
  Scanner(std::string_view s, std::string what) : b_(s.data()), p_(s.data()), e_(s.data() + s.size()), what_(std::move(what)) {
    if (s.size() >= 3 && std::memcmp(p_, "\xEF\xBB\xBF", 3) == 0) p_ += 3;
  }

  void ws() {
    while (p_ < e_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_;
  }
  bool accept(char c) {
    ws();
    if (p_ < e_ && *p_ == c) {
      ++p_;
      return true;
    }
    return false;
  }
  void expect(char c) {
    if (!accept(c)) error(std::string("expected '") + c + "'");
  }
  char peek() {
    ws();
    return p_ < e_ ? *p_ : '\0';
  }

  // Appends the unescaped JSON string at the cursor to out.
  void string(std::string& out) {
    expect('"');
    while (true) {
      const char* q = p_;
      while (q < e_ && *q != '"' && *q != '\\' && static_cast<unsigned char>(*q) >= 0x20) ++q;
      out.append(p_, size_t(q - p_));
      p_ = q;
      if (p_ >= e_) error("unterminated string");
      if (*p_ == '"') {
        ++p_;
        return;
      }
      if (*p_ != '\\') error("control character in string");
      if (++p_ >= e_) error("unterminated escape");
      const char c = *p_++;
      switch (c) {
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
          if (cp >= 0xD800 && cp <= 0xDBFF && e_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
            const char* save = p_;
            p_ += 2;
            const uint32_t lo = hex4();
            if (lo >= 0xDC00 && lo <= 0xDFFF) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            else p_ = save;
          }
          if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFD;  // lone surrogate
          tok::append_utf8(out, cp);
          break;
        }
        default: error("bad escape");
      }
    }
  }
  std::string string() {
    std::string s;
    string(s);
    return s;
  }

  int64_t integer() {
    ws();
    int64_t v = 0;
    auto r = std::from_chars(p_, e_, v);
    if (r.ec != std::errc() || r.ptr == p_) error("expected an integer");
    p_ = r.ptr;
    if (p_ < e_ && (*p_ == '.' || *p_ == 'e' || *p_ == 'E')) error("expected an integer");
    return v;
  }

  // Skips one value and returns its raw text.
  std::string_view raw_value() {
    ws();
    const char* start = p_;
    if (p_ >= e_) error("unexpected end");
    if (*p_ == '"') {
      skip_string();
    } else if (*p_ == '{' || *p_ == '[') {
      int depth = 0;
      while (p_ < e_) {
        const char c = *p_;
        if (c == '"') {
          skip_string();
          continue;
        }
        ++p_;
        if (c == '{' || c == '[') ++depth;
        else if ((c == '}' || c == ']') && --depth == 0) break;
      }
      if (depth != 0) error("unterminated container");
    } else {
      while (p_ < e_ && *p_ != ',' && *p_ != '}' && *p_ != ']' && *p_ != ' ' && *p_ != '\n' && *p_ != '\r' &&
             *p_ != '\t')
        ++p_;
    }
    return std::string_view(start, size_t(p_ - start));
  }
  Value value() {
    const std::string_view raw = raw_value();
    try {
      return json::parse(raw);
    } catch (const std::exception& ex) {
      error(ex.what());
    }
  }

  // Calls f(key) for each member of the object at the cursor; f must consume the value.
  template <class F>
  void object(F&& f) {
    expect('{');
    if (accept('}')) return;
    std::string key;
    do {
      key.clear();
      string(key);
      expect(':');
      f(key);
    } while (accept(','));
    expect('}');
  }
  template <class F>
  void array(F&& f) {
    expect('[');
    if (accept(']')) return;
    do f();
    while (accept(','));
    expect(']');
  }

  // model.vocab ({"text": id, ...}) straight into the build arena.
  void vocab(tok::BuildInput& in) {
    expect('{');
    if (accept('}')) return;
    do {
      const uint32_t off = uint32_t(in.arena.size());
      string(in.arena);
      const uint32_t len = uint32_t(in.arena.size()) - off;
      expect(':');
      const int64_t id = integer();
      if (id < 0 || id > INT32_MAX) error("bad token id " + std::to_string(id));
      in.vocab.push_back({{off, len}, int32_t(id)});
    } while (accept(','));
    expect('}');
  }

  bool at_end() {
    ws();
    return p_ == e_;
  }
  [[noreturn]] void error(const std::string& msg) const {
    fail(what_ + ": invalid JSON at byte " + std::to_string(p_ - b_) + ": " + msg);
  }

 private:
  uint32_t hex4() {
    if (e_ - p_ < 4) error("truncated \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = p_[k];
      int h = -1;
      if (c >= '0' && c <= '9') h = c - '0';
      else if (c >= 'a' && c <= 'f') h = c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') h = c - 'A' + 10;
      if (h < 0) error("bad \\u escape");
      v = (v << 4) | uint32_t(h);
    }
    p_ += 4;
    return v;
  }
  void skip_string() {
    ++p_;  // opening quote
    while (p_ < e_) {
      if (*p_ == '\\') {
        p_ += 2;
        continue;
      }
      if (*p_++ == '"') return;
    }
    error("unterminated string");
  }

  const char* b_;
  const char* p_;
  const char* e_;
  std::string what_;
};

std::string describe(const Value& v) {
  std::string s = json::dump(v);
  if (s.size() > 300) s = s.substr(0, 300) + "...";
  return s;
}

bool is_null_or_empty_string(const Value* v) { return !v || v->is_null() || (v->is_string() && v->as_string().empty()); }

// normalizer: null, NFC, or a Sequence of NFC.
bool parse_normalizer(const Value& v) {
  if (v.is_null()) return false;
  const std::string type = v.get_string("type", "");
  if (type == "NFC") return true;
  if (type == "Sequence") {
    const Value* list = v.get("normalizers");
    if (list && list->is_array()) {
      bool nfc = false;
      for (const Value& n : list->as_array()) {
        if (n.get_string("type", "") != "NFC") fail("unsupported normalizer: " + describe(v));
        nfc = true;
      }
      return nfc;
    }
  }
  fail("unsupported normalizer: " + describe(v));
}

// pre_tokenizer: Sequence[Split(Regex, Isolated), ByteLevel(add_prefix_space=false, use_regex=false)].
tok::SplitPattern parse_pre_tokenizer(const Value& v) {
  const Value* list = v.get("pretokenizers");
  if (v.get_string("type", "") != "Sequence" || !list || !list->is_array() || list->size() != 2)
    fail("unsupported pre_tokenizer (expected Sequence[Split, ByteLevel]): " + describe(v));
  const Value& split = (*list)[0];
  const Value& bl = (*list)[1];
  const Value* pat = split.get("pattern");
  const Value* re = pat ? pat->get("Regex") : nullptr;
  if (split.get_string("type", "") != "Split" || !re || !re->is_string())
    fail("unsupported pre_tokenizer Split (expected a Regex pattern): " + describe(split));
  if (split.get_string("behavior", "") != "Isolated" || split.get_bool("invert", false))
    fail("unsupported Split behavior (expected Isolated, invert=false): " + describe(split));
  const auto sp = tok::split_pattern_from_regex(re->as_string());
  if (!sp)
    fail("unsupported pre-tokenizer regex (no hand-written matcher for it): " + re->as_string() +
         "\n  supported: the Qwen2, Qwen3.5 and Llama-3 patterns");
  if (bl.get_string("type", "") != "ByteLevel" || bl.get_bool("add_prefix_space", false) || bl.get_bool("use_regex", true))
    fail("unsupported pre_tokenizer ByteLevel (expected add_prefix_space=false, use_regex=false): " + describe(bl));
  return *sp;
}

// SentencePiece-style BPE (Gemma 4): normalizer Replace(" " -> U+2581)
// (alone or as the only member of a Sequence), no pre-tokenizer or a
// Split(" ", MergedWithPrevious) that the normalizer leaves nothing to split,
// decoder Sequence[Replace(U+2581 -> " "), ByteFallback, Fuse].
bool is_spm_pipeline(const Value& norm, const Value* pre, const Value& dec) {
  static const std::string sp = "\xE2\x96\x81";
  auto replace_is = [](const Value& v, const std::string& from, const std::string& to) {
    const Value* pat = v.get("pattern");
    const Value* str = pat ? pat->get("String") : nullptr;
    return v.get_string("type", "") == "Replace" && str && str->is_string() && str->as_string() == from &&
           v.get_string("content", "") == to;
  };
  bool norm_ok = replace_is(norm, " ", sp);
  if (!norm_ok && norm.get_string("type", "") == "Sequence") {
    const Value* list = norm.get("normalizers");
    norm_ok = list && list->is_array() && list->size() == 1 && replace_is((*list)[0], " ", sp);
  }
  bool pre_ok = !pre || pre->is_null();
  if (!pre_ok) {
    const Value* pat = pre->get("pattern");
    const Value* str = pat ? pat->get("String") : nullptr;
    pre_ok = pre->get_string("type", "") == "Split" && str && str->is_string() && str->as_string() == " " &&
             pre->get_string("behavior", "") == "MergedWithPrevious" && !pre->get_bool("invert", false);
  }
  const Value* decs = dec.get("decoders");
  const bool dec_ok = dec.get_string("type", "") == "Sequence" && decs && decs->is_array() && decs->size() == 3 &&
                      replace_is((*decs)[0], sp, " ") && (*decs)[1].get_string("type", "") == "ByteFallback" &&
                      (*decs)[2].get_string("type", "") == "Fuse";
  return norm_ok && pre_ok && dec_ok;
}

tok::AddedToken parse_added_token(const Value& o) {
  tok::AddedToken a;
  const Value* c = o.get("content");
  if (!c || !c->is_string()) fail("added token without content: " + describe(o));
  a.content = c->as_string();
  a.special = o.get_bool("special", false);
  a.lstrip = o.get_bool("lstrip", false);
  a.rstrip = o.get_bool("rstrip", false);
  a.single_word = o.get_bool("single_word", false);
  // The reference's default for a missing flag is normalized = !special.
  a.normalized = o.get_bool("normalized", !a.special);
  return a;
}

// eos_token & co: a string, {"content": ...}, or null.
std::string token_name(const Value* v) {
  if (!v) return {};
  if (v->is_string()) return v->as_string();
  if (v->is_object()) return v->get_string("content", "");
  return {};
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::from_hf_dir(const std::string& dir_in) {
  std::string dir = dir_in;
  while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
  std::string text;
  const std::string path = dir + "/tokenizer.json";
  if (!read_file(path, text)) fail("cannot read " + path);

  tok::BuildInput in;
  in.vocab.reserve(1 << 18);
  in.merges.reserve(1 << 18);
  in.arena.reserve(text.size() / 5);
  in.merge_arena.reserve(text.size() / 4);
  Value normalizer, pre_tokenizer, decoder, added_tokens, model_type, dropout, byte_fallback, prefix, suffix,
      ignore_merges, unk_token, fuse_unk;
  bool have_pre = false, have_decoder = false;

  Scanner sc(text, path);
  sc.object([&](const std::string& key) {
    if (key == "model") {
      sc.object([&](const std::string& mkey) {
        if (mkey == "vocab") {
          sc.vocab(in);
        } else if (mkey == "merges") {
          sc.array([&]() {
            tok::BuildInput::Text a{}, b{};
            if (sc.peek() == '"') {
              // "a b": exactly one space separates the two tokens.
              const uint32_t off = uint32_t(in.merge_arena.size());
              sc.string(in.merge_arena);
              const std::string_view m(in.merge_arena.data() + off, in.merge_arena.size() - off);
              const size_t sp = m.find(' ');
              if (sp == std::string_view::npos || m.find(' ', sp + 1) != std::string_view::npos)
                fail("bad merge #" + std::to_string(in.merges.size()) + ": '" + std::string(m) + "'");
              a = {off, uint32_t(sp)};
              b = {off + uint32_t(sp) + 1, uint32_t(m.size() - sp - 1)};
            } else {
              sc.expect('[');
              a.off = uint32_t(in.merge_arena.size());
              sc.string(in.merge_arena);
              a.len = uint32_t(in.merge_arena.size()) - a.off;
              sc.expect(',');
              b.off = uint32_t(in.merge_arena.size());
              sc.string(in.merge_arena);
              b.len = uint32_t(in.merge_arena.size()) - b.off;
              sc.expect(']');
            }
            in.merges.emplace_back(a, b);
          });
        } else if (mkey == "type") {
          model_type = sc.value();
        } else if (mkey == "dropout") {
          dropout = sc.value();
        } else if (mkey == "byte_fallback") {
          byte_fallback = sc.value();
        } else if (mkey == "continuing_subword_prefix") {
          prefix = sc.value();
        } else if (mkey == "end_of_word_suffix") {
          suffix = sc.value();
        } else if (mkey == "ignore_merges") {
          ignore_merges = sc.value();
        } else if (mkey == "unk_token") {
          unk_token = sc.value();  // no effect on a complete byte-level vocabulary; spm: characters without byte tokens
        } else if (mkey == "fuse_unk") {
          fuse_unk = sc.value();
        } else {
          sc.raw_value();
        }
      });
    } else if (key == "normalizer") {
      normalizer = sc.value();
    } else if (key == "pre_tokenizer") {
      pre_tokenizer = sc.value();
      have_pre = true;
    } else if (key == "decoder") {
      decoder = sc.value();
      have_decoder = true;
    } else if (key == "added_tokens") {
      added_tokens = sc.value();
    } else {
      sc.raw_value();  // version, truncation, padding, post_processor
    }
  });
  if (!sc.at_end()) sc.error("trailing characters");

  // ---- validate the pipeline
  if (!model_type.is_null() && !(model_type.is_string() && model_type.as_string() == "BPE"))
    fail("unsupported model type " + describe(model_type) + " (only BPE)");
  if (dropout.is_number() && dropout.as_double() > 0) fail("BPE dropout is not supported");
  if (!is_null_or_empty_string(&prefix) || !is_null_or_empty_string(&suffix))
    fail("continuing_subword_prefix / end_of_word_suffix are not supported");
  if (in.vocab.empty()) fail(path + ": model.vocab is missing or empty");
  if (byte_fallback.is_bool() && byte_fallback.as_bool()) {
    // SentencePiece-style BPE (Gemma 4) is the one byte_fallback pipeline supported.
    if (!have_decoder || !is_spm_pipeline(normalizer, have_pre ? &pre_tokenizer : nullptr, decoder))
      fail("BPE byte_fallback is supported only with the SentencePiece-style pipeline (Replace ' ' -> U+2581, "
           "decoder Replace + ByteFallback + Fuse): " + describe(normalizer));
    in.spm = true;
    in.nfc = false;
    in.unk_token = unk_token.is_string() ? unk_token.as_string() : std::string();
    in.fuse_unk = fuse_unk.is_bool() && fuse_unk.as_bool();
  } else {
    in.nfc = parse_normalizer(normalizer);
    if (!have_pre) fail("missing pre_tokenizer");
    in.pattern = parse_pre_tokenizer(pre_tokenizer);
    if (!have_decoder || decoder.get_string("type", "") != "ByteLevel")
      fail("unsupported decoder (expected ByteLevel): " + describe(decoder));
  }
  in.ignore_merges = ignore_merges.is_bool() && ignore_merges.as_bool();

  // ---- added tokens: tokenizer.json, then tokenizer_config.json. Ids are assigned by
  // tok::build the way the reference does (the "id" fields in the files are ignored).
  if (added_tokens.is_array())
    for (const Value& o : added_tokens.as_array()) in.added.push_back(parse_added_token(o));

  std::string cfg_text;
  Value cfg;
  if (read_file(dir + "/tokenizer_config.json", cfg_text)) {
    try {
      cfg = json::parse(cfg_text);
    } catch (const std::exception& ex) {
      fail(dir + "/tokenizer_config.json: " + ex.what());
    }
    // transformers re-adds these; content already present keeps its id and takes these flags.
    if (const Value* atd = cfg.get("added_tokens_decoder"); atd && atd->is_object()) {
      const json::Object& o = atd->as_object();
      std::vector<std::pair<int64_t, size_t>> order;  // by id, as transformers sorts them
      for (size_t i = 0; i < o.size(); ++i) {
        int64_t id = 0;
        std::from_chars(o.keys[i].data(), o.keys[i].data() + o.keys[i].size(), id);
        order.emplace_back(id, i);
      }
      std::sort(order.begin(), order.end());
      for (const auto& [id, i] : order) in.added.push_back(parse_added_token(o.vals[i]));
    }
  }
  auto impl = tok::build(std::move(in));

  // ---- metadata
  // BOS before raw text: tokenizer_config.json's add_bos_token, else llama.cpp's
  // default (SentencePiece vocabularies such as Gemma's: yes; byte-level BPE: no).
  impl->add_bos = impl->spm;
  if (cfg.is_object()) {
    if (const Value* ab = cfg.get("add_bos_token"); ab && ab->is_bool()) impl->add_bos = ab->as_bool();
    auto id_of = [&](const char* key) {
      const std::string name = token_name(cfg.get(key));
      return name.empty() ? -1 : impl->find(name);
    };
    impl->eos = id_of("eos_token");
    impl->bos = id_of("bos_token");
    impl->pad = id_of("pad_token");

    if (const Value* ct = cfg.get("chat_template")) {
      if (ct->is_string()) {
        impl->chat_template = ct->as_string();
      } else if (ct->is_array()) {  // [{"name": ..., "template": ...}]: the default one
        for (const Value& t : ct->as_array())
          if (t.get_string("name", "") == "default" || impl->chat_template.empty())
            impl->chat_template = t.get_string("template", "");
      }
    }
  }
  std::string jinja;
  if (read_file(dir + "/chat_template.jinja", jinja)) impl->chat_template = std::move(jinja);  // takes priority

  return std::unique_ptr<Tokenizer>(new Tokenizer(std::move(impl)));
}

}  // namespace espetero
