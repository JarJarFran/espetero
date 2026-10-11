// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Tests for src/engine/model_meta: the supported-family check (MLX
// config.json model_type, GGUF general.architecture) and the sampling
// defaults a model ships (generation_config.json, GGUF general.sampling.*),
// on fake model directories and GGUF headers written to a temporary
// directory. No model, no GPU.
//   build/model_meta_test
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "../src/engine/model_meta.h"

using namespace espetero;

static int g_checks = 0, g_fails = 0;
#define CHECK(c)                                                    \
  do {                                                              \
    ++g_checks;                                                     \
    if (!(c)) {                                                     \
      ++g_fails;                                                    \
      std::printf("  FAIL line %d: %s\n", __LINE__, #c);            \
    }                                                               \
  } while (0)

static std::string g_root;

static void write(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary);
  f << text;
}

static std::string dir_with(const std::string& name, const std::string& config, const std::string& gen = "") {
  const std::string d = g_root + "/" + name;
  ::mkdir(d.c_str(), 0755);
  if (!config.empty()) write(d + "/config.json", config);
  if (!gen.empty()) write(d + "/generation_config.json", gen);
  return d;
}

// A GGUF v3 header with no tensors and the given metadata.
struct Gguf {
  std::string bytes;
  int n = 0;
  void raw(const void* p, size_t k) { bytes.append(static_cast<const char*>(p), k); }
  void u32(uint32_t v) { raw(&v, 4); }
  void u64(uint64_t v) { raw(&v, 8); }
  void str(const std::string& s) {
    u64(s.size());
    bytes += s;
  }
  void key(const std::string& k, uint32_t type) {
    str(k);
    u32(type);
    ++n;
  }
  Gguf& s(const std::string& k, const std::string& v) { return key(k, 8), str(v), *this; }
  Gguf& f32(const std::string& k, float v) { return key(k, 6), raw(&v, 4), *this; }
  Gguf& i32(const std::string& k, int32_t v) { return key(k, 5), raw(&v, 4), *this; }
  Gguf& strings(const std::string& k, int count) {  // a vocabulary-like array, skipped by the reader
    key(k, 9);
    u32(8);
    u64(uint64_t(count));
    for (int i = 0; i < count; ++i) str("tok" + std::to_string(i));
    return *this;
  }
  Gguf& ints(const std::string& k, int count) {
    key(k, 9);
    u32(5);
    u64(uint64_t(count));
    for (int i = 0; i < count; ++i) {
      const int32_t v = i;
      raw(&v, 4);
    }
    return *this;
  }
  std::string file(const std::string& name) const {
    std::string out = "GGUF";
    const uint32_t version = 3;
    const uint64_t tensors = 0, kv = uint64_t(n);
    out.append(reinterpret_cast<const char*>(&version), 4);
    out.append(reinterpret_cast<const char*>(&tensors), 8);
    out.append(reinterpret_cast<const char*>(&kv), 8);
    out += bytes;
    const std::string path = g_root + "/" + name;
    write(path, out);
    return path;
  }
};

static std::string error_of(const std::string& path) {
  try {
    check_model_supported(path);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

static bool contains(const std::string& s, const std::string& x) { return s.find(x) != std::string::npos; }

static void test_family() {
  std::printf("supported families\n");
  // Every model_type that loads today passes.
  for (const char* t : {"qwen3_5", "qwen3_5_moe", "qwen3_5_mtp", "qwen4_exp", "qwen4_exp_mtp", "gemma4", "gemma4_text"}) {
    const std::string d = dir_with(std::string("ok_") + t, std::string("{\"model_type\": \"") + t + "\"}");
    CHECK(error_of(d).empty());
  }
  CHECK(error_of(dir_with("ok_text", "{\"model_type\": \"qwen3_5\", \"text_config\": {\"model_type\": \"qwen3_5_text\"}}")).empty());
  CHECK(error_of(dir_with("ok_moe_text", "{\"text_config\": {\"model_type\": \"qwen3_5_moe_text\"}}")).empty());
  CHECK(error_of(dir_with("ok_q4_text", "{\"model_type\": \"qwen4_exp\", \"text_config\": {\"model_type\": \"qwen4_exp_text\"}}")).empty());
  CHECK(error_of(dir_with("ok_gemma", "{\"model_type\": \"gemma4\", \"text_config\": {\"model_type\": \"gemma4_text\"}}")).empty());
  // A finetune wrapper whose text model is of the family, and a config without any model_type: left to the loader.
  CHECK(error_of(dir_with("ok_wrapper", "{\"model_type\": \"ornith\", \"text_config\": {\"model_type\": \"qwen3_5_text\"}}")).empty());
  CHECK(error_of(dir_with("ok_untyped", "{\"hidden_size\": 1024}")).empty());

  // Other families: one message naming the model_type.
  const std::string llama = dir_with("llama4", "{\"model_type\": \"llama4\", \"text_config\": {\"model_type\": \"llama4_text\"}}");
  const std::string e = error_of(llama);
  CHECK(contains(e, llama + ": model_type 'llama4' is not supported by Espetero 0.1.0; it runs the Qwen3.5 family (qwen3_5, "
                            "qwen3_5_moe: Qwen3.5, Qwen3.6, Qwen3.8, and finetunes such as Ornith 1.5), "
                            "Qwen3.8-Flash-Next (qwen4_exp), Gemma 4 (gemma4: 31B, 26B-A4B; text only) and Cloudflare Clef classifiers"));
  // Other Gemma generations are not Gemma 4.
  CHECK(contains(error_of(dir_with("no_gemma3", "{\"model_type\": \"gemma3\"}")), "model_type 'gemma3' is not supported"));
  for (const char* t : {"qwen3", "qwen3_vl", "qwen3_moe", "llama", "qwen2_5_vl"})
    CHECK(contains(error_of(dir_with(std::string("no_") + t, std::string("{\"model_type\": \"") + t + "\"}")),
                   std::string("model_type '") + t + "' is not supported"));
  CHECK(contains(error_of(dir_with("no_text_only", "{\"text_config\": {\"model_type\": \"llama4_text\"}}")), "'llama4_text'"));

  // Paths that are not models.
  CHECK(contains(error_of(g_root + "/missing"), "no such file or directory"));
  CHECK(contains(error_of(g_root + "/missing"), "Hugging Face repo id"));
  CHECK(contains(error_of(dir_with("empty", "")), "no config.json"));
  CHECK(contains(error_of(dir_with("badjson", "{nope")), "config.json"));
  write(g_root + "/notes.txt", "x");
  CHECK(contains(error_of(g_root + "/notes.txt"), "not an MLX model directory or a .gguf file"));

  // GGUF architecture.
  CHECK(error_of(Gguf().s("general.architecture", "qwen35").strings("tokenizer.ggml.tokens", 300).file("q.gguf")).empty());
  CHECK(error_of(Gguf().strings("tokenizer.ggml.tokens", 10).s("general.architecture", "qwen35moe").file("m.gguf")).empty());
  CHECK(error_of(Gguf().strings("tokenizer.ggml.tokens", 10).s("general.architecture", "gemma4").file("g4.gguf")).empty());
  const std::string g = Gguf().s("general.architecture", "llama").ints("x.sections", 4).file("g.gguf");
  CHECK(contains(error_of(g), g + ": GGUF architecture 'llama' is not supported by Espetero 0.1.0; it runs the Qwen3.5 family "
                                 "(qwen35, qwen35moe:"));
  CHECK(contains(error_of(g), "and Gemma 4 (gemma4: 31B, 26B-A4B; text only)"));
  CHECK(contains(error_of(g), "Cloudflare Clef classifiers run from MLX directories"));
  write(g_root + "/fake.gguf", "this is not a gguf file at all");
  CHECK(contains(error_of(g_root + "/fake.gguf"), "not a GGUF file"));
  std::string cut = Gguf().s("general.architecture", "qwen35").file("cut.gguf");
  {
    std::ifstream f(cut, std::ios::binary);
    std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    write(cut, b.substr(0, b.size() - 3));
  }
  CHECK(contains(error_of(cut), "truncated"));
}

static bool near(double a, double b) { return std::fabs(a - b) < 1e-12; }

static void test_defaults() {
  std::printf("sampling defaults\n");
  // Qwen's generation_config.json.
  SamplingDefaults d = read_sampling_defaults(dir_with(
      "gen_qwen", "{\"model_type\": \"qwen3_5\"}",
      "{\"do_sample\": true, \"temperature\": 0.6, \"top_k\": 20, \"top_p\": 0.95, \"eos_token_id\": [1, 2]}"));
  CHECK(d.temperature && near(*d.temperature, 0.6));
  CHECK(d.top_p && near(*d.top_p, 0.95));
  CHECK(d.top_k && *d.top_k == 20);
  CHECK(!d.min_p);
  CHECK(d.source == "generation_config.json");
  json::Value j = d.to_json(true);
  CHECK(j.get_double("temperature", -1) == 0.6 && j.get_int("top_k", -1) == 20 && j.get_double("min_p", -1) == 0);
  CHECK(contains(j.get_string("source", ""), "generation_config.json (temperature, top_p, top_k; the others: OpenAI defaults)"));
  j = d.to_json(false);
  CHECK(j.get_double("temperature", -1) == 1.0 && j.get_double("top_p", -1) == 1.0 && j.get_int("top_k", -1) == 0);
  CHECK(contains(j.get_string("source", ""), "--no-generation-config"));

  // No sampling keys; nulls, strings, out of range and top_k -1 (off).
  d = read_sampling_defaults(dir_with("gen_none", "{}", "{\"_from_model_config\": true, \"eos_token_id\": 2}"));
  CHECK(d.empty() && d.source.empty());
  CHECK(contains(d.to_json(true).get_string("source", ""), "the model sets none"));
  d = read_sampling_defaults(dir_with("gen_odd", "{}", "{\"temperature\": null, \"top_p\": \"0.9\", \"top_k\": -1, \"min_p\": 1.5}"));
  CHECK(!d.temperature && !d.top_p && !d.min_p && d.top_k && *d.top_k == 0);
  d = read_sampling_defaults(dir_with("gen_hot", "{}", "{\"temperature\": 7, \"top_p\": 0, \"top_k\": 2.5, \"min_p\": 0.05}"));
  CHECK(!d.temperature && !d.top_p && !d.top_k && d.min_p && near(*d.min_p, 0.05));
  d = read_sampling_defaults(dir_with("gen_greedy", "{}", "{\"temperature\": 0.0}"));
  CHECK(d.temperature && *d.temperature == 0);
  CHECK(read_sampling_defaults(dir_with("gen_missing", "{}")).empty());
  CHECK(read_sampling_defaults(dir_with("gen_bad", "{}", "[1, 2")).empty());
  CHECK(read_sampling_defaults(g_root + "/missing").empty());

  // GGUF general.sampling.* (float32 values read back as their decimals).
  d = read_sampling_defaults(Gguf()
                                 .s("general.architecture", "qwen35")
                                 .i32("general.sampling.top_k", 20)
                                 .f32("general.sampling.top_p", 0.95f)
                                 .strings("tokenizer.ggml.tokens", 1000)
                                 .f32("general.sampling.temp", 0.6f)
                                 .file("s.gguf"));
  CHECK(d.temperature && *d.temperature == 0.6);
  CHECK(d.top_p && *d.top_p == 0.95);
  CHECK(d.top_k && *d.top_k == 20);
  CHECK(!d.min_p);
  CHECK(d.source == "GGUF general.sampling");
  d = read_sampling_defaults(Gguf().s("general.architecture", "qwen35").f32("general.sampling.min_p", 0.05f).file("p.gguf"));
  CHECK(d.min_p && *d.min_p == 0.05 && !d.temperature);
  CHECK(read_sampling_defaults(Gguf().s("general.architecture", "qwen35").file("n.gguf")).empty());
  CHECK(read_sampling_defaults(g_root + "/fake.gguf").empty());
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);  // section names in order with stderr
  const char* tmp = std::getenv("TMPDIR");
  std::string base = tmp && *tmp ? tmp : "/tmp";
  if (base.back() != '/') base += '/';
  base += "espetero_meta_XXXXXX";
  std::vector<char> buf(base.begin(), base.end());
  buf.push_back('\0');
  const char* root = ::mkdtemp(buf.data());
  if (!root) {
    std::printf("cannot create a temporary directory\n");
    return 1;
  }
  g_root = root;
  test_family();
  test_defaults();
  std::string rm = "rm -rf '" + g_root + "'";
  if (std::system(rm.c_str()) != 0) std::printf("(could not remove %s)\n", g_root.c_str());
  std::printf("%d checks, %d failures\n", g_checks, g_fails);
  return g_fails ? 1 : 0;
}
