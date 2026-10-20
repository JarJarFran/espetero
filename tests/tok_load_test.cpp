// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The tokenizer.json loader's family gates, on synthetic files (no model, no
// reference data): the shapes LFM2.5-VL's tokenizer.json has (a Sequence of
// one ByteLevel decoder, a post-processor template that starts with the BOS
// token) are accepted for the LFM2 pre-tokenizer only; every other byte-level
// family loads exactly as before (plain ByteLevel decoder, no BOS before raw
// text unless tokenizer_config.json asks for it).
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/tok/tokenizer.h"
#include "../src/util/json.h"

using namespace espetero;

static int g_checks = 0, g_fails = 0;
#define CHECK(c)                                         \
  do {                                                   \
    ++g_checks;                                          \
    if (!(c)) {                                          \
      ++g_fails;                                         \
      std::printf("  FAIL line %d: %s\n", __LINE__, #c); \
    }                                                    \
  } while (0)

namespace {

// The Split regexes as tokenizer.json stores them (src/tok/pretok.cpp).
const char* kLlama3 = R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";
const char* kLfm2 = R"re('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s)re";

std::string g_root;

void write(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary);
  f << text;
}

json::Value special(const char* content, int64_t id) {
  json::Value o = json::Value::object();
  o.set("id", id);
  o.set("content", std::string(content));
  o.set("special", true);
  o.set("single_word", false);
  o.set("lstrip", false);
  o.set("rstrip", false);
  o.set("normalized", false);
  return o;
}

// A tiny byte-level BPE tokenizer.json: <bos>, <eos> and a few letters.
//   regex: the Split pattern; sequence_decoder: ByteLevel wrapped in a Sequence;
//   bos_template: a TemplateProcessing whose single sequence starts with <bos>.
std::string tokenizer_json(const char* regex, bool sequence_decoder, bool bos_template) {
  json::Value root = json::Value::object();
  root.set("version", std::string("1.0"));
  json::Value added = json::Value::array();
  added.push_back(special("<bos>", 0));
  added.push_back(special("<eos>", 1));
  root.set("added_tokens", std::move(added));
  root.set("normalizer", json::Value());
  {
    json::Value split = json::Value::object();
    split.set("type", std::string("Split"));
    json::Value pat = json::Value::object();
    pat.set("Regex", std::string(regex));
    split.set("pattern", std::move(pat));
    split.set("behavior", std::string("Isolated"));
    split.set("invert", false);
    json::Value bl = json::Value::object();
    bl.set("type", std::string("ByteLevel"));
    bl.set("add_prefix_space", false);
    bl.set("trim_offsets", true);
    bl.set("use_regex", false);
    json::Value list = json::Value::array();
    list.push_back(std::move(split));
    list.push_back(std::move(bl));
    json::Value pre = json::Value::object();
    pre.set("type", std::string("Sequence"));
    pre.set("pretokenizers", std::move(list));
    root.set("pre_tokenizer", std::move(pre));
  }
  if (bos_template) {
    json::Value tp = json::Value::object();
    tp.set("type", std::string("TemplateProcessing"));
    json::Value single = json::Value::array();
    {
      json::Value st = json::Value::object();
      st.set("id", std::string("<bos>"));
      st.set("type_id", int64_t(0));
      json::Value item = json::Value::object();
      item.set("SpecialToken", std::move(st));
      single.push_back(std::move(item));
      json::Value sq = json::Value::object();
      sq.set("id", std::string("A"));
      sq.set("type_id", int64_t(0));
      json::Value item2 = json::Value::object();
      item2.set("Sequence", std::move(sq));
      single.push_back(std::move(item2));
    }
    tp.set("single", std::move(single));
    json::Value st_map = json::Value::object();
    {
      json::Value bos = json::Value::object();
      bos.set("id", std::string("<bos>"));
      json::Value ids = json::Value::array();
      ids.push_back(int64_t(0));
      bos.set("ids", std::move(ids));
      json::Value toks = json::Value::array();
      toks.push_back(std::string("<bos>"));
      bos.set("tokens", std::move(toks));
      st_map.set("<bos>", std::move(bos));
    }
    tp.set("special_tokens", std::move(st_map));
    root.set("post_processor", std::move(tp));
  } else {
    root.set("post_processor", json::Value());
  }
  {
    json::Value bl = json::Value::object();
    bl.set("type", std::string("ByteLevel"));
    bl.set("add_prefix_space", true);
    bl.set("trim_offsets", true);
    bl.set("use_regex", true);
    if (sequence_decoder) {
      json::Value list = json::Value::array();
      list.push_back(std::move(bl));
      json::Value seq = json::Value::object();
      seq.set("type", std::string("Sequence"));
      seq.set("decoders", std::move(list));
      root.set("decoder", std::move(seq));
    } else {
      root.set("decoder", std::move(bl));
    }
  }
  {
    json::Value model = json::Value::object();
    model.set("type", std::string("BPE"));
    model.set("dropout", json::Value());
    model.set("unk_token", json::Value());
    model.set("fuse_unk", false);
    model.set("byte_fallback", false);
    model.set("ignore_merges", false);
    json::Value vocab = json::Value::object();
    vocab.set("<bos>", int64_t(0));
    vocab.set("<eos>", int64_t(1));
    const char* letters[] = {"x", "y", "z", "a", "b", "c"};
    int64_t id = 2;
    for (const char* l : letters) vocab.set(l, id++);
    // The byte-level alphabet (GPT-2's bytes_to_unicode), which the loader requires.
    int next = 256;
    for (int b = 0; b < 256; ++b) {
      const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
      const int cp = printable ? b : next++;
      std::string u;
      if (cp < 0x80) {
        u.push_back(char(cp));
      } else {
        u.push_back(char(0xC0 | (cp >> 6)));
        u.push_back(char(0x80 | (cp & 0x3F)));
      }
      if (!vocab.get(u)) vocab.set(u, id++);
    }
    model.set("vocab", std::move(vocab));
    model.set("merges", json::Value::array());
    root.set("model", std::move(model));
  }
  return json::dump(root);
}

// A model directory with that tokenizer.json and a tokenizer_config.json
// naming <bos> / <eos>; add_bos: "" (absent), "true" or "false".
std::string dir_with(const std::string& name, const std::string& tokenizer, const std::string& add_bos) {
  const std::string d = g_root + "/" + name;
  ::mkdir(d.c_str(), 0755);
  write(d + "/tokenizer.json", tokenizer);
  std::string cfg = "{\"bos_token\": \"<bos>\", \"eos_token\": \"<eos>\"";
  if (!add_bos.empty()) cfg += ", \"add_bos_token\": " + add_bos;
  cfg += "}";
  write(d + "/tokenizer_config.json", cfg);
  return d;
}

std::string load_error(const std::string& dir) {
  try {
    Tokenizer::from_hf_dir(dir);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool contains(const std::string& s, const std::string& what) { return s.find(what) != std::string::npos; }

}  // namespace

int main() {
  std::string buf = "/tmp/tok_load_test.XXXXXX";
  const char* root = ::mkdtemp(buf.data());
  if (!root) {
    std::printf("mkdtemp failed\n");
    return 1;
  }
  g_root = root;

  // ---- the plain shape every byte-level family has: loads for Llama 3 and LFM2 alike, no BOS before raw text.
  for (const char* regex : {kLlama3, kLfm2}) {
    const bool lfm2 = regex == kLfm2;
    auto T = Tokenizer::from_hf_dir(dir_with(lfm2 ? "plain_lfm2" : "plain_llama3", tokenizer_json(regex, false, false), ""));
    CHECK(T->pre_tokenizer_name() == (lfm2 ? "lfm2" : "llama3"));
    CHECK(T->bos_id() == 0 && T->eos_id() == 1);
    CHECK(!T->add_bos());
    const std::vector<int32_t> ids = T->with_bos(T->encode("x"));
    CHECK(ids.size() == 1 && ids[0] == 2);
  }

  // ---- a Sequence of one ByteLevel decoder: LFM2 only (LFM2.5-VL-1.6B's file); refused for Llama 3 as before.
  {
    const std::string e = load_error(dir_with("seq_llama3", tokenizer_json(kLlama3, true, false), ""));
    CHECK(contains(e, "unsupported decoder (expected ByteLevel)"));
    auto T = Tokenizer::from_hf_dir(dir_with("seq_lfm2", tokenizer_json(kLfm2, true, false), ""));
    CHECK(T->pre_tokenizer_name() == "lfm2");
    CHECK(T->decode(std::vector<int32_t>{2, 3}) == "xy");
  }

  // ---- a post-processor template that starts with <bos> and no add_bos_token:
  // BOS before raw text for LFM2 (as transformers adds it), not for Llama 3 (llama.cpp's default, as before).
  {
    auto L = Tokenizer::from_hf_dir(dir_with("tmpl_llama3", tokenizer_json(kLlama3, false, true), ""));
    CHECK(!L->add_bos());
    CHECK(L->with_bos(L->encode("x")) == (std::vector<int32_t>{2}));
    auto F = Tokenizer::from_hf_dir(dir_with("tmpl_lfm2", tokenizer_json(kLfm2, false, true), ""));
    CHECK(F->add_bos());
    CHECK(F->with_bos(F->encode("x")) == (std::vector<int32_t>{0, 2}));
    // tokenizer_config.json's add_bos_token still wins, both ways.
    auto Fno = Tokenizer::from_hf_dir(dir_with("tmpl_lfm2_off", tokenizer_json(kLfm2, false, true), "false"));
    CHECK(!Fno->add_bos());
    auto Lyes = Tokenizer::from_hf_dir(dir_with("tmpl_llama3_on", tokenizer_json(kLlama3, false, true), "true"));
    CHECK(Lyes->add_bos());
    CHECK(Lyes->with_bos(Lyes->encode("x")) == (std::vector<int32_t>{0, 2}));
  }

  std::printf("tok_load_test: %d checks, %d failures\n", g_checks, g_fails);
  return g_fails ? 1 : 0;
}
