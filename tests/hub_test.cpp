// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Tests for src/server/hub: Hugging Face repo ids for --model. Repo id
// parsing, the cache directory's environment variables, GGUF file selection
// on a real repository's file list, snapshots of a fake hub cache (refs/main,
// modification times, links to missing blobs). Safetensors and GGUF
// headers, model footprints, the memory a model needs against a Mac, the
// refusal's text. The recommendations. Against a fake Hub on
// 127.0.0.1 (HF_ENDPOINT): the API listing, ranges, tokens, the check before
// a download, and downloads into the cache's layout (config.json
// first, GGUF selection, resumed transfers, checksums, locks, gated files,
// offline). No network, no model, no GPU work.
//   build/hub_test
//   build/hub_test --live    huggingface.co, ~30 MB (see live())
#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../src/http/http.h"
#include "../src/server/download.h"
#include "../src/server/hub.h"
#include "../src/server/models.h"

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

// A check on a message, which is printed when the check fails.
#define CHECK_MSG(c, m)                                         \
  do {                                                          \
    ++g_checks;                                                 \
    if (!(c)) {                                                 \
      ++g_fails;                                                \
      std::printf("  FAIL line %d: %s\n    message: %s\n", __LINE__, #c, std::string(m).c_str()); \
    }                                                           \
  } while (0)

static bool contains(const std::string& s, const std::string& x) { return s.find(x) != std::string::npos; }

static void mkdirs(const std::string& path) {
  std::string cur;
  std::stringstream ss(path);
  std::string part;
  while (std::getline(ss, part, '/')) {
    cur += part + "/";
    ::mkdir(cur.c_str(), 0755);
  }
}

static void write(const std::string& path, const std::string& text) {
  mkdirs(path.substr(0, path.find_last_of('/')));
  std::ofstream f(path, std::ios::binary);
  f << text;
}

static bool is_regular(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static std::string read(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static void set_mtime(const std::string& path, long seconds) {
  timeval tv[2] = {{seconds, 0}, {seconds, 0}};
  ::utimes(path.c_str(), tv);
}

static void env(const char* k, const std::string& v) {
  if (v.empty()) ::unsetenv(k);
  else ::setenv(k, v.c_str(), 1);
}

static hub::Ref ref(const std::string& repo, const std::string& quant = "") {
  hub::Ref r;
  r.repo = repo;
  r.quant = quant;
  return r;
}

// The error message of f(), "" when it does not throw hub::Error.
template <class F>
static std::string error_of(F&& f) {
  try {
    f();
  } catch (const hub::Error& e) {
    return e.what();
  }
  return "";
}

// ---------------------------------------------------------------- parsing
static void test_parse() {
  std::printf("repo ids\n");
  hub::Ref r;
  CHECK(hub::parse_ref("mlx-community/Qwen3.8-27B-4bit", r) && r.repo == "mlx-community/Qwen3.8-27B-4bit" && r.quant.empty());
  CHECK(hub::parse_ref("unsloth/Qwen3.8-27B-GGUF:UD-Q6_K", r) && r.repo == "unsloth/Qwen3.8-27B-GGUF" && r.quant == "UD-Q6_K");
  CHECK(hub::parse_ref("Org_1/name.v2", r));
  for (const char* bad : {"", "model", "a/b/c", "/abs/path", "./x", "../x", "~/x", "org/", "/name", "org/na me", "org/x:",
                          "org/x:a/b", ".hidden/x", "org/-x", "org/x:Q4 K", "org/x?"})
    CHECK(!hub::parse_ref(bad, r));
  // An existing path is a path, even with one slash.
  write(g_root + "/rel/model/config.json", "{}");
  const std::string cwd = [] {
    char b[4096];
    return std::string(::getcwd(b, sizeof b) ? b : ".");
  }();
  if (::chdir(g_root.c_str()) == 0) {
    CHECK(!hub::parse_ref("rel/model", r));
    CHECK(hub::parse_ref("rel/other", r));
    if (::chdir(cwd.c_str()) != 0) std::printf("  (cannot return to %s)\n", cwd.c_str());
  }
}

static void test_env() {
  std::printf("cache directory and offline\n");
  for (const char* k : {"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"}) env(k, "");
  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
  CHECK(hub::cache_dir() == home + "/.cache/huggingface/hub");
  env("XDG_CACHE_HOME", "/x/cache");
  CHECK(hub::cache_dir() == "/x/cache/huggingface/hub");
  env("HF_HOME", "/h/hf/");
  CHECK(hub::cache_dir() == "/h/hf/hub");
  env("HUGGINGFACE_HUB_CACHE", "/old/hub");
  CHECK(hub::cache_dir() == "/old/hub");
  env("HF_HUB_CACHE", "~/hub");
  CHECK(hub::cache_dir() == home + "/hub");
  for (const char* k : {"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"}) env(k, "");
  env("HF_HUB_OFFLINE", "");
  CHECK(!hub::offline_env());
  for (const char* on : {"1", "true", "YES", "On"}) {
    env("HF_HUB_OFFLINE", on);
    CHECK(hub::offline_env());
  }
  env("HF_HUB_OFFLINE", "0");
  CHECK(!hub::offline_env());
  env("HF_HUB_OFFLINE", "");
}

// ---------------------------------------------------------------- GGUF selection
// The files of unsloth/Qwen3.8-27B-GGUF (2026-10), plus a README.
static const std::vector<std::string> kUnsloth = {
    "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf",
    "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "Qwen3.8-27B-Q4_0.gguf", "Qwen3.8-27B-Q4_1.gguf", "Qwen3.8-27B-Q8_0.gguf",
    "Qwen3.8-27B-UD-IQ1_M.gguf", "Qwen3.8-27B-UD-IQ2_XXS.gguf", "Qwen3.8-27B-UD-Q4_K_XL.gguf",
    "Qwen3.8-27B-UD-Q6_K.gguf", "Qwen3.8-27B-UD-Q6_K_L.gguf", "Qwen3.8-27B-UD-Q6_K_M.gguf",
    "Qwen3.8-27B-UD-Q6_K_XL.gguf", "imatrix_unsloth.gguf", "mmproj-BF16.gguf", "mmproj-F16.gguf", "README.md",
    ".gitattributes"};

static void test_pick() {
  std::printf("GGUF selection\n");
  hub::GgufPick p;
  CHECK(hub::pick_gguf(kUnsloth, ref("unsloth/Qwen3.8-27B-GGUF", "UD-Q6_K"), p));
  CHECK(p.model == "Qwen3.8-27B-UD-Q6_K.gguf");
  CHECK(p.mmproj == "mmproj-F16.gguf");
  CHECK(p.mtp == "MTP/mtp-Qwen3.8-27B-Q4_0.gguf");
  CHECK(hub::pick_gguf(kUnsloth, ref("unsloth/Qwen3.8-27B-GGUF", "ud-q6_k"), p) && p.model == "Qwen3.8-27B-UD-Q6_K.gguf");
  CHECK(hub::pick_gguf(kUnsloth, ref("unsloth/Qwen3.8-27B-GGUF", "UD-Q6_K_XL"), p) && p.model == "Qwen3.8-27B-UD-Q6_K_XL.gguf");
  CHECK(hub::pick_gguf(kUnsloth, ref("unsloth/Qwen3.8-27B-GGUF", "Q4_0"), p) && p.model == "Qwen3.8-27B-Q4_0.gguf");
  CHECK(hub::pick_gguf(kUnsloth, ref("unsloth/Qwen3.8-27B-GGUF", "Q6_K"), p) && p.model == "Qwen3.8-27B-UD-Q6_K.gguf");
  CHECK(hub::pick_gguf(kUnsloth, ref("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-Q8_0"), p) && p.model == "Qwen3.8-27B-Q8_0.gguf");
  CHECK(!hub::pick_gguf(kUnsloth, ref("unsloth/Qwen3.8-27B-GGUF", "Q7_K"), p));
  CHECK(!hub::pick_gguf(kUnsloth, ref("unsloth/Qwen3.8-27B-GGUF", "6_K"), p));  // not after a separator
  CHECK(!hub::pick_gguf(kUnsloth, ref("unsloth/Qwen3.8-27B-GGUF", "imatrix_unsloth"), p));
  // Split models are refused with the parts and how to merge them.
  std::string e = error_of([&] { hub::pick_gguf(kUnsloth, ref("unsloth/Qwen3.8-27B-GGUF", "BF16"), p); });
  CHECK(contains(e, "is split in 2 files") && contains(e, "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf") &&
        contains(e, "llama-gguf-split --merge"));
  // No selector and several models: the choices, without the imatrix file.
  e = error_of([&] { hub::pick_gguf(kUnsloth, ref("unsloth/Qwen3.8-27B-GGUF"), p); });
  CHECK(contains(e, "holds 11 GGUF models; name one after a colon:"));
  CHECK(contains(e, "\n  --model unsloth/Qwen3.8-27B-GGUF:UD-Q6_K\n") && contains(e, "\n  --model unsloth/Qwen3.8-27B-GGUF:Q4_0\n"));
  CHECK(contains(e, "--model unsloth/Qwen3.8-27B-GGUF:BF16   (split in 2 files, not supported)"));
  CHECK(!contains(e, "imatrix") && !contains(e, "mmproj"));
  // One model: no selector needed; projector next to the model, F16 before BF16 and F32.
  const std::vector<std::string> one = {"Model-Q6_K.gguf", "mmproj-model-f16.gguf", "MTP/mtp-x.gguf", "README.md"};
  CHECK(hub::pick_gguf(one, ref("o/one"), p) && p.model == "Model-Q6_K.gguf" && p.mmproj == "mmproj-model-f16.gguf" &&
        p.mtp == "MTP/mtp-x.gguf");
  CHECK(hub::pick_gguf({"q/M-Q4_K_M.gguf", "mmproj-F16.gguf", "q/mmproj-F32.gguf", "q/mmproj-BF16.gguf"}, ref("o/x", "Q4_K_M"), p) &&
        p.mmproj == "q/mmproj-BF16.gguf");
  CHECK(hub::pick_gguf({"M-Q4.gguf", "mmproj-F32.gguf", "mmproj-BF16.gguf"}, ref("o/x"), p) && p.mmproj == "mmproj-BF16.gguf" &&
        p.mtp.empty());
  // The shortest name wins a tie of the same kind; equal names in two directories are ambiguous.
  CHECK(hub::pick_gguf({"M-UD-Q6_K.gguf", "M-Q6_K.gguf"}, ref("o/x", "Q6_K"), p) && p.model == "M-Q6_K.gguf");
  e = error_of([&] { hub::pick_gguf({"a/M-Q4.gguf", "b/M-Q4.gguf"}, ref("o/x", "Q4"), p); });
  CHECK(contains(e, "matches several GGUF models") && contains(e, "a/M-Q4.gguf") && contains(e, "b/M-Q4.gguf"));
  CHECK(!hub::pick_gguf({"config.json", "model.safetensors"}, ref("o/x"), p));
}

// ---------------------------------------------------------------- cache
// A snapshot file that links to a blob, as huggingface_hub stores them
// (a missing blob leaves a dangling link).
static void blob_link(const std::string& repo, const std::string& snap, const std::string& rel, const std::string& text,
                      bool blob = true) {
  static int n = 0;
  const std::string name = "blob" + std::to_string(++n);
  if (blob) write(repo + "/blobs/" + name, text);
  const std::string link = repo + "/snapshots/" + snap + "/" + rel;
  mkdirs(link.substr(0, link.find_last_of('/')));
  std::string up = "../../blobs/" + name;
  for (char c : rel)
    if (c == '/') up = "../" + up;
  ::unlink(link.c_str());
  if (::symlink(up.c_str(), link.c_str()) != 0) std::printf("  (symlink %s failed)\n", link.c_str());
}

static void mlx_snapshot(const std::string& repo, const std::string& snap, bool complete) {
  blob_link(repo, snap, "config.json", "{\"model_type\": \"qwen3_5\"}");
  blob_link(repo, snap, "tokenizer.json", "{}");
  blob_link(repo, snap, "model.safetensors.index.json",
            "{\"weight_map\": {\"a\": \"model-00001-of-00002.safetensors\", \"b\": \"model-00002-of-00002.safetensors\"}}");
  blob_link(repo, snap, "model-00001-of-00002.safetensors", "x");
  blob_link(repo, snap, "model-00002-of-00002.safetensors", "x", complete);
}

static void test_cache() {
  std::printf("cache snapshots\n");
  const std::string cache = g_root + "/hub";
  hub::Model m;
  CHECK(!hub::find_cached(cache, ref("acme/Nothing"), m));

  // MLX: complete, then a shard whose blob is missing.
  const std::string mlx = cache + "/models--acme--Tiny-4bit";
  mlx_snapshot(mlx, "aaa", true);
  CHECK(hub::find_cached(cache, ref("acme/Tiny-4bit"), m) && m.path == mlx + "/snapshots/aaa" && m.mmproj.empty() &&
        m.revision == "aaa");
  CHECK(!hub::find_cached(cache, ref("acme/Tiny-4bit", "Q4_K_M"), m));  // a selector looks for GGUF files only
  const std::string part = cache + "/models--acme--Part";
  mlx_snapshot(part, "p1", false);
  CHECK(!hub::find_cached(cache, ref("acme/Part"), m));
  blob_link(part, "p1", "model.safetensors.index.json", "{\"weight_map\": {}}");
  CHECK(!hub::find_cached(cache, ref("acme/Part"), m));

  // Newest snapshot: refs/main first, then modification time; incomplete ones are skipped.
  const std::string two = cache + "/models--acme--Two";
  mlx_snapshot(two, "old", true);
  mlx_snapshot(two, "new", true);
  set_mtime(two + "/snapshots/old", 1000000);
  set_mtime(two + "/snapshots/new", 2000000);
  CHECK(hub::find_cached(cache, ref("acme/Two"), m) && m.path == two + "/snapshots/new");
  write(two + "/refs/main", "old\n");
  CHECK(hub::find_cached(cache, ref("acme/Two"), m) && m.path == two + "/snapshots/old");
  write(two + "/refs/main", "gone");
  mlx_snapshot(two, "newest", false);
  set_mtime(two + "/snapshots/newest", 3000000);
  CHECK(hub::find_cached(cache, ref("acme/Two"), m) && m.path == two + "/snapshots/new");

  // GGUF at the snapshot root: the engine finds the projector and MTP head itself.
  const std::string gg = cache + "/models--acme--Tiny-GGUF";
  for (const char* f : {"Tiny-Q4_K_M.gguf", "Tiny-UD-Q6_K.gguf", "mmproj-F16.gguf", "MTP/mtp-Tiny-Q4_0.gguf"}) blob_link(gg, "g1", f, "x");
  CHECK(hub::find_cached(cache, ref("acme/Tiny-GGUF", "q4_k_m"), m) && m.path == gg + "/snapshots/g1/Tiny-Q4_K_M.gguf" &&
        m.mmproj.empty() && m.mtp.empty());
  CHECK(!hub::find_cached(cache, ref("acme/Tiny-GGUF", "Q8_0"), m));
  CHECK(contains(error_of([&] { hub::find_cached(cache, ref("acme/Tiny-GGUF"), m); }), "holds 2 GGUF models"));
  // A model in a subdirectory: projector and MTP head at the root are passed explicitly.
  const std::string sub = cache + "/models--acme--Sub-GGUF";
  for (const char* f : {"Q4_K_M/Sub-Q4_K_M.gguf", "mmproj-F16.gguf", "MTP/mtp-Sub.gguf"}) blob_link(sub, "s1", f, "x");
  CHECK(hub::find_cached(cache, ref("acme/Sub-GGUF"), m) && m.path == sub + "/snapshots/s1/Q4_K_M/Sub-Q4_K_M.gguf" &&
        m.mmproj == sub + "/snapshots/s1/mmproj-F16.gguf" && m.mtp == sub + "/snapshots/s1/MTP/mtp-Sub.gguf");
  // A model whose blob is missing (a download cut short) is not there.
  const std::string cut = cache + "/models--acme--Cut-GGUF";
  blob_link(cut, "c1", "Cut-Q4_0.gguf", "x", false);
  CHECK(!hub::find_cached(cache, ref("acme/Cut-GGUF", "Q4_0"), m));
}

// ---------------------------------------------------------------- headers and footprints
static std::string hexs(const unsigned char* d, size_t n) {
  static const char* x = "0123456789abcdef";
  std::string o;
  for (size_t i = 0; i < n; ++i) o += x[d[i] >> 4], o += x[d[i] & 15];
  return o;
}
static std::string sha256_hex(const std::string& s) {
  unsigned char d[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(s.data(), CC_LONG(s.size()), d);
  return hexs(d, sizeof d);
}
// git's blob id: sha1("blob <size>\0" + content), the cache's name for small files.
static std::string git_blob_id(const std::string& s) {
  const std::string full = "blob " + std::to_string(s.size()) + std::string(1, '\0') + s;
  unsigned char d[CC_SHA1_DIGEST_LENGTH];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  CC_SHA1(full.data(), CC_LONG(full.size()), d);
#pragma clang diagnostic pop
  return hexs(d, sizeof d);
}

// A safetensors header: {name: [dtype, bytes]} laid out one after the other.
static std::string safetensors(const std::vector<std::pair<std::string, std::pair<std::string, int64_t>>>& ts) {
  std::string j = "{\"__metadata__\": {\"format\": \"mlx\"}";
  int64_t off = 0;
  for (const auto& [name, t] : ts) {
    j += ", \"" + name + "\": {\"dtype\": \"" + t.first + "\", \"shape\": [" + std::to_string(t.second) +
         "], \"data_offsets\": [" + std::to_string(off) + ", " + std::to_string(off + t.second) + "]}";
    off += t.second;
  }
  j += "}";
  const uint64_t n = j.size();
  return std::string(reinterpret_cast<const char*>(&n), 8) + j;
}

// A safetensors header padded with spaces to `size` bytes (JSON allows them),
// so a shard of a known size can hold a tensor that fills it.
static std::string safetensors_padded(const std::vector<std::pair<std::string, std::pair<std::string, int64_t>>>& ts, size_t size) {
  std::string h = safetensors(ts);
  std::string j = h.substr(8);
  j.resize(size - 8, ' ');
  const uint64_t n = j.size();
  return std::string(reinterpret_cast<const char*>(&n), 8) + j;
}

// A GGUF header: string/u32 metadata, a token array to skip, tensor infos.
struct GT {
  std::string name;
  std::vector<uint64_t> dims;
  uint32_t type;
};
static std::string gguf(const std::vector<std::pair<std::string, std::string>>& strs,
                        const std::vector<std::pair<std::string, uint32_t>>& ints, const std::vector<GT>& ts,
                        int vocab_strings = 3) {
  std::string b = "GGUF";
  auto u32 = [&](uint32_t v) { b.append(reinterpret_cast<const char*>(&v), 4); };
  auto u64 = [&](uint64_t v) { b.append(reinterpret_cast<const char*>(&v), 8); };
  auto str = [&](const std::string& v) { u64(v.size()), b += v; };
  u32(3);
  u64(ts.size());
  u64(strs.size() + ints.size() + 1);
  for (const auto& [k, v] : strs) str(k), u32(8), str(v);
  for (const auto& [k, v] : ints) str(k), u32(4), u32(v);
  str("tokenizer.ggml.tokens"), u32(9), u32(8), u64(uint64_t(vocab_strings));
  for (int i = 0; i < vocab_strings; ++i) str("tok" + std::to_string(i));
  for (const GT& t : ts) {
    str(t.name);
    u32(uint32_t(t.dims.size()));
    for (uint64_t d : t.dims) u64(d);
    u32(t.type);
    u64(0);
  }
  return b;
}

// Qwen3.5-4B's geometry (config.json and GGUF metadata).
static const char* k4BConfig = R"({"model_type": "qwen3_5", "tie_word_embeddings": true,
  "vision_config": {"depth": 24},
  "quantization": {"group_size": 64, "bits": 4, "mode": "affine",
                   "language_model.model.layers.0.mlp.gate_proj": {"group_size": 64, "bits": 8}},
  "text_config": {"model_type": "qwen3_5_text", "hidden_size": 2560, "intermediate_size": 9216, "num_hidden_layers": 32,
    "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256, "linear_num_key_heads": 16,
    "linear_num_value_heads": 32, "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
    "vocab_size": 248320, "full_attention_interval": 4, "mtp_num_hidden_layers": 1,
    "rope_parameters": {"partial_rotary_factor": 0.25}}})";

static std::vector<std::pair<std::string, uint32_t>> gguf_4b(uint32_t blocks = 32) {
  return {{"qwen35.embedding_length", 2560}, {"qwen35.feed_forward_length", 9216}, {"qwen35.block_count", blocks},
          {"qwen35.attention.head_count", 16}, {"qwen35.attention.head_count_kv", 4}, {"qwen35.attention.key_length", 256},
          {"qwen35.ssm.conv_kernel", 4}, {"qwen35.ssm.state_size", 128}, {"qwen35.ssm.group_count", 16},
          {"qwen35.ssm.time_step_rank", 32}, {"qwen35.full_attention_interval", 4}, {"qwen35.rope.dimension_count", 64}};
}

static void test_geometry() {
  std::printf("geometry, KV and state sizes\n");
  const models::Geometry g = models::geometry_from_config(json::parse(k4BConfig));
  CHECK(g.known && g.unsupported.empty() && g.n_full == 8 && g.n_linear == 24 && g.D == 2560 && g.V == 248320);
  // forest.h: (n_full + mtp) x K and V x kv_dim x fp16; recurrent + conv state per linear layer in floats.
  CHECK(g.kv_bytes_per_token(false) == 8 * 2 * 1024 * 2 && g.kv_bytes_per_token(true) == 9 * 2 * 1024 * 2);
  const int64_t C = 2 * 16 * 128 + 32 * 128;
  CHECK(g.slot_bytes(false) == (3 * C + 32 * 128 * 128) * 24 * 4);
  CHECK(g.slot_bytes(true) == g.slot_bytes(false) + 17 * 2560 * 4);
  CHECK(g.row_bytes() > 100000 && g.row_bytes() < 200000);
  // The engine's limits: GQA group 16 (Qwen3.5-122B-A10B) is refused.
  json::Value big = json::parse(k4BConfig);
  big.get("text_config")->set("num_attention_heads", 32);
  big.get("text_config")->set("num_key_value_heads", 2);
  CHECK(models::geometry_from_config(big).unsupported == "unsupported GQA group 16");
  // Flash-Next: hyper-connection streams and the sparse attention index.
  json::Value q4 = json::parse(k4BConfig);
  q4.get("text_config")->set("model_type", "qwen4_exp_text");
  q4.get("text_config")->set("num_attention_heads", 24);
  q4.get("text_config")->set("num_key_value_heads", 2);
  q4.get("text_config")->set("hc_count", 4);
  const models::Geometry g4 = models::geometry_from_config(q4);
  CHECK(g4.q4 && g4.unsupported.empty() && g4.kv_bytes_per_token(false) == 8 * 2 * 512 * 2 + 8 * 336);
}

static void test_headers() {
  std::printf("safetensors and GGUF headers\n");
  const std::string st = safetensors({{"a.weight", {"U32", 64}}, {"a.scales", {"BF16", 8}}});
  size_t need = 0;
  std::vector<models::Tensor> ts = models::parse_safetensors(st + "data", &need);
  CHECK(ts.size() == 2 && ts[0].name == "a.weight" && ts[0].bytes == 64 && ts[1].dtype == "BF16" && need == st.size());
  std::string e;
  try {
    models::parse_safetensors(st.substr(0, 20), &need);
  } catch (const std::exception& x) {
    e = x.what();
  }
  CHECK(contains(e, "cut short") && need == st.size());
  const std::string g = gguf({{"general.architecture", "qwen35"}}, gguf_4b(),
                             {{"token_embd.weight", {2560, 248320}, 14}, {"blk.0.attn_qkv.weight", {2560, 8192}, 8}}, 1000);
  models::GgufHeader h, h2;
  CHECK(models::parse_gguf(g, h) && h.tensors.size() == 2 && h.tensors[0].dtype == "Q6_K" &&
        h.tensors[1].elements() == 2560LL * 8192 && h.get_string("general.architecture", "") == "qwen35" &&
        h.get_int("qwen35.block_count", 0) == 32);
  CHECK(!models::parse_gguf(g.substr(0, g.size() - 5), h2));  // the tensor table cut short: read more
  CHECK(!models::parse_gguf(g.substr(0, 100), h2));
  e.clear();
  try {
    models::parse_gguf("GGML" + g.substr(4), h2);
  } catch (const std::exception& x) {
    e = x.what();
  }
  CHECK(e == "not a GGUF file");
  const models::Geometry gg = models::geometry_from_gguf(h);
  CHECK(gg.known && gg.unsupported.empty() && gg.n_full == 8 && gg.V == 248320 && gg.Hv == 32);
  // GPU bytes per GGUF type (gguf.h, matrix.h): Q8_0 and Q6_K keep their scales, K-quants expand.
  CHECK(models::gguf_engine_bytes(8, 1024) == 1088 && models::gguf_engine_bytes(14, 1024) == 840);
  CHECK(models::gguf_engine_bytes(12, 1024) == 640 && models::gguf_engine_bytes(13, 1024) == 768);
  CHECK(models::gguf_engine_bytes(23, 1024) == 1152 && models::gguf_engine_bytes(1, 1024) == 2048);
  CHECK(models::gguf_quant_factor("UD-Q6_K_XL") == 1.0 && models::gguf_quant_factor("Q4_K_M") == 1.1 &&
        models::gguf_quant_factor("UD-IQ2_XXS") == 0 && models::gguf_quant_factor("IQ4_XS") == 2.0);
}

static void test_footprints() {
  std::printf("footprints of local models\n");
  // MLX: weights, a tied head's copy, the vision tower, the MTP head and its 2-bit draft copy.
  const std::string mlx = g_root + "/models/qwen3.5/Small-4bit";
  write(mlx + "/config.json", k4BConfig);
  write(mlx + "/model.safetensors",
        safetensors({{"language_model.model.embed_tokens.weight", {"U32", 1000}},
                     {"language_model.model.embed_tokens.scales", {"BF16", 100}},
                     {"language_model.model.embed_tokens.biases", {"BF16", 100}},
                     {"language_model.model.layers.0.mlp.gate_proj.weight", {"U32", 2000}},
                     {"language_model.mtp.fc.weight", {"BF16", 400}},
                     {"vision_tower.patch_embed.proj.weight", {"BF16", 300}}}) +
            std::string(3900, '\0'));
  models::Footprint f = models::footprint_local(mlx);
  const int64_t coarse = 248320LL * 2560 * 5 / 16;
  CHECK(f.format == "MLX" && f.quant == "mixed 4/8-bit" && f.has_mtp && f.has_vision);
  CHECK(f.weights == 3900 + 1200 + coarse && f.vision == 300 && f.mtp == 400 + coarse && f.streamed == 0);
  CHECK(f.disk == int64_t(read(mlx + "/model.safetensors").size()));
  models::Plan p;
  models::Need n = models::need(f, p);
  CHECK(n.weights == f.weights);
  p.vision = false;
  p.mtp = false;
  CHECK(models::need(f, p).weights == 3900 + 1200 - 300 - 400);
  // 2- and 3-bit codes widen to bytes; group 128 scales are split in two.
  json::Value c3 = json::parse(k4BConfig);
  c3.set("quantization", json::parse(R"({"group_size": 128, "bits": 3})"));
  write(mlx + "/config.json", json::dump(c3));
  f = models::footprint_local(mlx);
  CHECK(f.quant == "3-bit" && f.weights == (1000 * 8 / 3 + 200 * 2) + (2000 * 8 / 3) + 400 + 300 + (1000 * 8 / 3 + 200 * 2) + coarse);
  write(mlx + "/config.json", k4BConfig);

  // Flash-Next: experts and the n-gram table stay on the SSD; the drafter beside it counts.
  json::Value q4 = json::parse(k4BConfig);
  q4.get("text_config")->set("model_type", "qwen4_exp_text");
  q4.get("text_config")->set("num_attention_heads", 24);
  q4.get("text_config")->set("num_key_value_heads", 2);
  q4.get("text_config")->set("num_experts", 512);
  q4.get("text_config")->set("num_experts_per_tok", 10);
  q4.get("text_config")->set("moe_intermediate_size", 640);
  q4.get("text_config")->set("shared_expert_intermediate_size", 640);
  const std::string fn = g_root + "/models/flash/Next-4bit";
  write(fn + "/config.json", json::dump(q4));
  write(fn + "/model.safetensors",
        safetensors({{"language_model.model.embed_tokens.weight", {"U32", 1000}},
                     {"language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight", {"U32", 50000}},
                     {"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight", {"U32", 70000}},
                     {"language_model.mtp.layers.0.mlp.switch_mlp.up_proj.weight", {"U32", 900}}}));
  write(g_root + "/models/flash/Next-MTP-Drafter/config.json", "{\"model_type\": \"qwen4_exp_mtp\"}");
  write(g_root + "/models/flash/Next-MTP-Drafter/model.safetensors", std::string(800, 'x'));
  f = models::footprint_local(fn);
  CHECK(f.geo.q4 && f.streamed == 120000 && f.weights == 1000 + 1000 + 400 && f.has_mtp && f.mtp == 400);
  CHECK(models::footprint_local(fn, "", "none").weights == 2000 && !models::footprint_local(fn, "", "none").has_mtp);
  n = models::need(f, models::Plan{});
  CHECK(n.pools >= int64_t(3e9));  // 3 GB of slots + KV by default (model.mm)

  // GGUF: the model, the projector next to it and the MTP head in MTP/.
  const std::string gd = g_root + "/models/qwen3.5/Small-GGUF";
  write(gd + "/Small-Q8_0.gguf",
        gguf({{"general.architecture", "qwen35"}}, gguf_4b(),
             {{"token_embd.weight", {2560, 248320}, 8}, {"output.weight", {2560, 248320}, 8},
              {"blk.0.ffn_up.weight", {2560, 9216}, 8}, {"output_norm.weight", {2560}, 0}}));
  write(gd + "/mmproj-Small-F16.gguf", gguf({{"general.architecture", "clip"}}, {}, {{"v.patch_embd.weight", {1024, 1152}, 1}}));
  write(gd + "/MTP/mtp-Small-Q8_0.gguf",
        gguf({{"general.architecture", "qwen35"}}, {{"qwen35.block_count", 1}, {"qwen35.nextn_predict_layers", 1}},
             {{"token_embd.weight", {2560, 248320}, 8}, {"blk.0.nextn.eh_proj.weight", {5120, 2560}, 8}}));
  write(gd + "/Small-Q4_K_M.gguf",
        gguf({{"general.architecture", "qwen35"}}, gguf_4b(), {{"token_embd.weight", {2560, 248320}, 12}}));
  f = models::footprint_local(gd + "/Small-Q8_0.gguf");
  const int64_t emb = 2560LL * 248320 + 2560LL * 248320 / 32 * 4;  // gathered rows: bytes + half scale and bias per 32
  const int64_t q8 = models::gguf_engine_bytes(8, 2560LL * 248320) + models::gguf_engine_bytes(8, 2560LL * 9216);
  const int64_t mtp = models::gguf_engine_bytes(8, 5120LL * 2560) + coarse;
  CHECK(f.format == "GGUF" && f.has_vision && f.has_mtp && f.vision == 1024LL * 1152 * 2 && f.mtp == mtp);
  CHECK(f.weights == emb + q8 + 2560 * 4 + 1024LL * 1152 * 2 + mtp);
  CHECK(f.base == emb + q8 + 2560 * 4 && contains(f.what, "projector mmproj-Small-F16.gguf") && contains(f.what, "MTP head mtp-Small-Q8_0.gguf"));
  f = models::footprint_local(gd + "/Small-Q8_0.gguf", "none", "none");
  CHECK(!f.has_vision && !f.has_mtp && f.weights == emb + q8 + 2560 * 4);
  // Other quantizations in the directory are the alternatives.
  const std::vector<models::Alternative> alts = models::local_alternatives(gd + "/Small-Q8_0.gguf");
  CHECK(alts.size() == 1 && contains(alts[0].model, "Small-Q4_K_M.gguf"));
}

static void test_need() {
  std::printf("memory needed against a Mac\n");
  const models::Mac m16 = models::mac_with_ram(16), m64 = models::mac_with_ram(64);
  CHECK(m16.ram == 16LL << 30 && m16.gpu == (16LL << 30) / 3 * 2 && m64.gpu == (64LL << 30) / 4 * 3);
  CHECK(models::mac_with_ram(36).gpu == (36LL << 30) / 4 * 3 && m16.usable() == m16.gpu);
  models::Footprint f;
  f.format = "MLX";
  f.quant = "8-bit";
  f.geo = models::geometry_from_config(json::parse(k4BConfig));
  f.weights = f.base = f.disk = int64_t(8e9);
  models::Plan p;
  const models::Need n = models::need(f, p);
  const int64_t slot = f.geo.slot_bytes(false), tok = f.geo.kv_bytes_per_token(false);
  CHECK(n.weights == int64_t(8e9) && n.pools == 8 * slot + 2 * 32768 * tok + 9 * slot);
  CHECK(n.scratch == 4096 * f.geo.row_bytes() + f.geo.readout_bytes() + int64_t(0.25e9));
  // --slots, --kv-gb and --memory-gb move the floor; --context halves the KV.
  models::Plan q = p;
  q.slots = 16;
  q.kv_gb = 1;
  CHECK(models::need(f, q).pools == 16 * slot + int64_t(1e9) + 9 * slot);
  q = p;
  q.memory_gb = 20;
  CHECK(models::need(f, q).pools == int64_t(20e9) + 9 * slot);
  q = p;
  q.max_ctx = 16384;
  CHECK(models::need(f, q).pools == n.pools - 32768 * tok);
  // 16 GB: 8 GB of weights plus ~4 GB at 32K does not fit 11.5 GB; a smaller context does.
  models::Verdict v = models::judge(f, p, m16);
  CHECK(!v.fits && v.usable == m16.gpu && v.context_that_fits >= 2048 && v.context_that_fits < 32768);
  q = p;
  q.max_ctx = v.context_that_fits;
  CHECK(models::judge(f, q, m16).fits);
  q.max_ctx = v.context_that_fits * 2;
  CHECK(!models::judge(f, q, m16).fits);
  CHECK(models::judge(f, p, m64).fits);
  // Too large for any context.
  f.weights = int64_t(14e9);
  v = models::judge(f, p, m16);
  CHECK(!v.fits && v.context_that_fits == 0);
  // The refusal: parts, the alternatives that fit (largest first), the way out.
  models::Footprint small = f, smaller = f, huge = f;
  small.weights = small.disk = int64_t(5e9);
  smaller.weights = smaller.disk = int64_t(3e9);
  huge.weights = huge.disk = int64_t(20e9);
  const std::string msg = models::refusal("acme/Big-8bit", f, p, m16, v,
                                          {{"acme/Big-3bit", smaller}, {"acme/Big-6bit", huge}, {"acme/Big-4bit", small}},
                                          "Nothing was downloaded.");
  CHECK(contains(msg, "acme/Big-8bit needs about " + models::gb(v.need.total()) + " of GPU memory, more than the " +
                          models::gb(m16.gpu) + " this Mac's GPU can use (16 GB of RAM"));
  CHECK(contains(msg, "14.0 GB  weights (MLX 8-bit)") && contains(msg, "KV cache for 2 x 32768 tokens (--context) and 17 sequence states"));
  CHECK(contains(msg, "scratch for 4096-token passes") && !contains(msg, "It fits with --context"));
  CHECK(contains(msg, "Smaller files of this model that fit:\n  --model acme/Big-4bit   5.0 GB of files"));
  CHECK(msg.find("acme/Big-4bit") < msg.find("acme/Big-3bit") && !contains(msg, "acme/Big-6bit"));
  CHECK(contains(msg, "\nNothing was downloaded.") && msg.back() == '.');
  f.weights = int64_t(8e9);
  v = models::judge(f, p, m16);
  CHECK(contains(models::refusal("x", f, p, m16, v, {}, ""), "It fits with --context " + std::to_string(v.context_that_fits)));
  // Bit-width siblings to suggest.
  using V = std::vector<std::string>;
  CHECK(models::bit_siblings("Qwen3.8-27B-8bit") == V({"Qwen3.8-27B-6bit", "Qwen3.8-27B-5bit", "Qwen3.8-27B-4bit"}));
  CHECK(models::bit_siblings("Qwen3.8-27B-MLX-6Bit") == V({"Qwen3.8-27B-MLX-8Bit", "Qwen3.8-27B-MLX-5Bit", "Qwen3.8-27B-MLX-4Bit"}));
  CHECK(models::bit_siblings("Qwen3.6-35B-A3B-4bit-DWQ").size() == 3 && models::bit_siblings("Qwen3.6-35B-A3B-4bit-DWQ")[0] == "Qwen3.6-35B-A3B-8bit-DWQ");
  CHECK(models::bit_siblings("Qwen3.8-27B-oQ8e-mtp") == V({"Qwen3.8-27B-oQ6e-mtp", "Qwen3.8-27B-oQ5e-mtp", "Qwen3.8-27B-oQ4e-mtp"}));
  CHECK(models::bit_siblings("Qwen3.5-0.8B").empty() && models::bit_siblings("orbit").empty());
}


static void test_recommend() {
  std::printf("recommendations by Mac memory\n");
  const std::vector<models::CatalogEntry>& cat = models::catalog();
  CHECK(cat.size() >= 20);
  for (const models::CatalogEntry& e : cat) {
    hub::Ref r;
    CHECK(hub::parse_ref(e.id, r) && (e.footprint.format == "GGUF") == !r.quant.empty());
    CHECK(e.bits == 8 || e.bits == 6 || e.bits == 4);
    CHECK(e.footprint.weights > 0 && e.footprint.disk > 0 && e.footprint.geo.known && e.footprint.geo.unsupported.empty());
  }
  auto none = [](const std::string&) { return false; };
  auto pick = [&](double ram, const std::string& model) -> const models::Recommendation* {
    static std::vector<models::Recommendation> keep;
    keep = models::recommend(models::mac_with_ram(ram), models::Plan{}, none);
    for (const models::Recommendation& r : keep)
      if (r.entry->model == model) return &r;
    return nullptr;
  };
  // Best model first; 8-bit where it fits, 4-bit only when memory forces it; a smaller context when needed.
  std::vector<models::Recommendation> r64 = models::recommend(models::mac_with_ram(64), models::Plan{}, none);
  CHECK(!r64.empty() && r64[0].entry->model == "Qwen3.8-Flash-Next" && r64[0].cache > int64_t(30e9));
  const models::Recommendation* p = pick(64, "Qwen3.8-27B");
  CHECK(p && p->entry->id == "unsloth/Qwen3.8-27B-GGUF:Q8_0" && p->context == 0);
  p = pick(48, "Qwen3.8-27B");
  CHECK(p && p->entry->bits == 8 && p->context == 0 && !pick(48, "Qwen3.8-Flash-Next"));
  p = pick(32, "Qwen3.8-27B");
  CHECK(p && p->entry->bits == 4 && p->context == 16384);
  CHECK(!pick(24, "Qwen3.8-27B") && pick(24, "Qwen3.5-9B") && pick(24, "Qwen3.5-9B")->entry->bits == 8);
  CHECK(!pick(8, "Qwen3.5-9B") && pick(8, "Qwen3.5-0.8B"));
  // Every pick fits; no file of a wider tier fits at the plan's context.
  for (double ram : {8.0, 16.0, 24.0, 32.0, 36.0, 48.0, 64.0, 96.0, 128.0, 192.0})
    for (const models::Recommendation& r : models::recommend(models::mac_with_ram(ram), models::Plan{}, none)) {
      models::Plan q;
      if (r.context) q.max_ctx = r.context;
      CHECK(models::judge(r.entry->footprint, q, models::mac_with_ram(ram)).fits);
      for (const models::CatalogEntry& e : cat)
        if (e.model == r.entry->model && e.bits > r.entry->bits)
          CHECK(!models::judge(e.footprint, models::Plan{}, models::mac_with_ram(ram)).fits);
    }
  // --all lists every file; the text marks the downloaded ones.
  CHECK(models::recommend(models::mac_with_ram(16), models::Plan{}, none, true).size() == cat.size());
  auto have = [](const std::string& id) { return id == "mlx-community/Qwen3.5-9B-4bit" || id == "mlx-community/Qwen3.5-0.8B-4bit"; };
  const models::Mac m16 = models::mac_with_ram(16);
  const std::string t = models::recommendations_text(m16, models::recommend(m16, models::Plan{}, have), false);
  CHECK_MSG(contains(t, "Models that fit a Mac with 16 GB of RAM (11.5 GB for the GPU), best first:"), t);
  CHECK_MSG(contains(t, "\n* mlx-community/Qwen3.5-9B-4bit ") && contains(t, "also downloaded: mlx-community/Qwen3.5-0.8B-4bit"), t);
  CHECK_MSG(contains(t, "* in the Hugging Face cache already."), t);
  CHECK(contains(models::recommendations_text(models::mac_with_ram(2), {}, false), "No model of the table fits a Mac with 2 GB"));
  // The catalog's files of the same model are alternatives; finetunes and other sizes are not.
  std::vector<models::Alternative> a = models::catalog_alternatives("mlx-community/Qwen3.8-27B-8bit");
  CHECK(a.size() == 5 && a.back().model == "mlx-community/Qwen3.8-27B-4bit");
  CHECK(models::catalog_alternatives("lmstudio-community/Qwen3.8-27B-MLX-8bit").size() == 6);
  CHECK(models::catalog_alternatives("Qwen3.5-2B-4bit").size() == 6 && models::catalog_alternatives("Qwen3.5-2B-4bit")[0].model.find("2B") != std::string::npos);
  CHECK(models::catalog_alternatives("acme/Tuned-Qwen3.8-27B-abliterated").empty());
}

// ---------------------------------------------------------------- a fake Hub
// The Hub's API and file endpoints on 127.0.0.1: the listing of
// /api/models/{repo}/revision/{rev}?blobs=true, files at
// /{repo}/resolve/{rev}/{path} (LFS files redirect to /cdn/ on "localhost",
// another host, as the Hub redirects to its CDN), byte ranges, a gated
// repository that wants a token. Files can be virtual: `content` is their
// start and zeros fill them to `size`.
struct FakeFile {
  std::string content;
  int64_t size = -1;
  bool lfs = false;
  int64_t length() const { return size >= 0 ? size : int64_t(content.size()); }
  std::string slice(int64_t a, int64_t b) const {  // [a, b)
    std::string out;
    if (a < int64_t(content.size())) out = content.substr(size_t(a), size_t(std::min<int64_t>(b, content.size()) - a));
    if (b > int64_t(content.size())) out.append(size_t(b - std::max<int64_t>(a, content.size())), '\0');
    return out;
  }
};
struct FakeRepoDef {
  std::string commit;
  std::map<std::string, FakeFile> files;
  bool gated = false;
};

class FakeHub {
 public:
  std::map<std::string, FakeRepoDef> repos;
  std::string token;  // the token a gated repository accepts
  std::map<std::string, int64_t> cut;  // file path -> its next whole-file answer ends after these bytes
  std::set<std::string> corrupt;       // file paths served with a byte changed
  std::mutex mu;
  std::vector<std::string> log;  // "GET <path> [range] [auth]"

  FakeHub() {
    http::Options o;
    o.host = "127.0.0.1";
    o.port = 0;
    srv_ = std::make_unique<http::Server>(o);
    srv_->route_prefix("GET", "/api/models/", [this](const http::Request& q, http::Response& r) { api(q, r); });
    srv_->route_prefix("GET", "/", [this](const http::Request& q, http::Response& r) { file(q, r); });
    th_ = std::thread([this] { srv_->run(); });
  }
  ~FakeHub() {
    srv_->stop();
    th_.join();
  }
  std::string url() const { return "http://127.0.0.1:" + std::to_string(srv_->port()); }
  int port() const { return srv_->port(); }
  std::vector<std::string> requests() {
    std::lock_guard<std::mutex> lk(mu);
    return log;
  }
  void clear() {
    std::lock_guard<std::mutex> lk(mu);
    log.clear();
  }
  static std::string blob_of(const FakeFile& f) { return f.lfs ? sha256_hex(f.size >= 0 ? f.content : f.content) : git_blob_id(f.content); }

 private:
  void note(const http::Request& q) {
    const std::string* range = q.header("range");
    const std::string* auth = q.header("authorization");
    std::lock_guard<std::mutex> lk(mu);
    log.push_back("GET " + q.path + (range ? " " + *range : "") + (auth ? " auth=" + *auth : ""));
  }
  void api(const http::Request& q, http::Response& r) {
    note(q);
    // /api/models/{org}/{name}/revision/{rev}
    const std::string rest = q.path.substr(12);
    const size_t a = rest.find('/'), b = a == std::string::npos ? a : rest.find('/', a + 1);
    const std::string repo = b == std::string::npos ? rest : rest.substr(0, b);
    auto it = repos.find(repo);
    if (it == repos.end()) {
      r.set_header("X-Error-Code", "RepoNotFound");
      r.send(401, "application/json", "{\"error\":\"Repository not found\"}");
      return;
    }
    json::Value v = json::Value::object();
    v.set("id", repo);
    v.set("sha", it->second.commit);
    v.set("gated", it->second.gated ? json::Value("manual") : json::Value(false));
    json::Value sib = json::Value::array();
    for (const auto& [path, f] : it->second.files) {
      json::Value e = json::Value::object();
      e.set("rfilename", path);
      e.set("size", f.length());
      e.set("blobId", git_blob_id(path));
      if (f.lfs) {
        json::Value l = json::Value::object();
        l.set("sha256", blob_of(f));
        l.set("size", f.length());
        l.set("pointerSize", 134);
        e.set("lfs", l);
      } else {
        e.set("blobId", blob_of(f));
      }
      sib.push_back(e);
    }
    v.set("siblings", sib);
    r.send_json(200, v);
  }
  void serve(const http::Request& q, http::Response& r, const FakeFile& f0, const std::string& path) {
    FakeFile f = f0;
    {
      std::lock_guard<std::mutex> lk(mu);
      if (corrupt.count(path) && !f.content.empty()) f.content[f.content.size() / 2] ^= 1;
      auto c = cut.find(path);
      if (c != cut.end() && !q.header("range")) {  // a connection lost after these bytes
        const int64_t n = c->second;
        cut.erase(c);
        r.start_stream(200, "application/octet-stream");
        r.write(f.slice(0, n));
        return;
      }
    }
    int64_t a = 0, b = f.length();
    int status = 200;
    if (const std::string* range = q.header("range"); range && range->rfind("bytes=", 0) == 0) {
      const std::string spec = range->substr(6);
      const size_t dash = spec.find('-');
      a = std::atoll(spec.substr(0, dash).c_str());
      if (dash + 1 < spec.size()) b = std::min<int64_t>(b, std::atoll(spec.c_str() + dash + 1) + 1);
      if (a >= f.length()) {
        r.set_header("Content-Range", "bytes */" + std::to_string(f.length()));
        r.send(416, "text/plain", "");
        return;
      }
      status = 206;
      r.set_header("Content-Range", "bytes " + std::to_string(a) + "-" + std::to_string(b - 1) + "/" + std::to_string(f.length()));
    }
    r.set_header("Accept-Ranges", "bytes");
    r.send(status, "application/octet-stream", f.slice(a, b));
  }
  void file(const http::Request& q, http::Response& r) {
    note(q);
    if (q.path.rfind("/cdn/", 0) == 0) {  // /cdn/{org}/{name}/{path}
      const std::string rest = q.path.substr(5);
      const size_t a = rest.find('/'), b = rest.find('/', a + 1);
      auto it = repos.find(rest.substr(0, b));
      if (it == repos.end() || !it->second.files.count(rest.substr(b + 1))) {
        r.send(404, "text/plain", "no such object");
        return;
      }
      serve(q, r, it->second.files.at(rest.substr(b + 1)), rest.substr(b + 1));
      return;
    }
    // /{org}/{name}/resolve/{rev}/{path}
    const std::string p = q.path.substr(1);
    const size_t a = p.find('/'), b = p.find('/', a + 1);
    const std::string repo = p.substr(0, b);
    const std::string tail = b == std::string::npos ? "" : p.substr(b + 1);
    auto it = repos.find(repo);
    if (it == repos.end() || tail.rfind("resolve/", 0) != 0) {
      r.set_header("X-Error-Code", "RepoNotFound");
      r.send(401, "text/plain", "Repository not found");
      return;
    }
    const size_t c = tail.find('/', 8);
    const std::string rev = tail.substr(8, c - 8), path = tail.substr(c + 1);
    const FakeRepoDef& d = it->second;
    if (rev != d.commit && rev != "main") {
      r.set_header("X-Error-Code", "RevisionNotFound");
      r.send(404, "text/plain", "Invalid rev id");
      return;
    }
    if (d.gated) {
      const std::string* auth = q.header("authorization");
      if (!auth || *auth != "Bearer " + token) {
        r.set_header("X-Error-Code", "GatedRepo");
        r.set_header("X-Error-Message", "Access to model " + repo + " is restricted.");
        r.send(401, "text/plain", "Access to model " + repo + " is restricted.");
        return;
      }
    }
    auto f = d.files.find(path);
    if (f == d.files.end()) {
      r.set_header("X-Error-Code", "EntryNotFound");
      r.send(404, "text/plain", "Entry not found");
      return;
    }
    r.set_header("X-Repo-Commit", d.commit);
    if (f->second.lfs) {
      r.set_header("X-Linked-Etag", "\"" + blob_of(f->second) + "\"");
      r.set_header("X-Linked-Size", std::to_string(f->second.length()));
      r.set_header("Location", "http://localhost:" + std::to_string(port()) + "/cdn/" + repo + "/" + path);
      r.send(302, "text/plain", "");
      return;
    }
    r.set_header("ETag", "\"" + blob_of(f->second) + "\"");
    serve(q, r, f->second, path);
  }
  std::unique_ptr<http::Server> srv_;
  std::thread th_;
};

static void test_hub_api(FakeHub& hub) {
  std::printf("the Hub's API: listing, ranges, errors\n");
  env("HF_ENDPOINT", hub.url() + "/");
  env("HF_HOME", g_root + "/hfhome");  // never the user's own token file
  CHECK(hub::endpoint() == hub.url());
  const hub::RepoInfo info = hub::repo_info("acme/Big-8bit");
  CHECK(info.commit == hub.repos["acme/Big-8bit"].commit && info.repo == "acme/Big-8bit" && info.files.size() == 4);
  const hub::RepoFile* cfg = info.find("config.json");
  CHECK(cfg && !cfg->lfs && cfg->blob == git_blob_id(hub.repos["acme/Big-8bit"].files["config.json"].content));
  const hub::RepoFile* sh = info.find("model-00001-of-00002.safetensors");
  CHECK(sh && sh->lfs && sh->size == int64_t(5e9) && sh->blob.size() == 64);
  // A range of an LFS file follows the redirect to the other host.
  hub.clear();
  const std::string head = hub::read_range(info, "model-00001-of-00002.safetensors", 0, 8);
  CHECK(head.size() == 8 && head == hub.repos["acme/Big-8bit"].files["model-00001-of-00002.safetensors"].content.substr(0, 8));
  std::vector<std::string> log = hub.requests();
  CHECK(log.size() == 2 && contains(log[0], "/acme/Big-8bit/resolve/" + info.commit + "/model-00001-of-00002.safetensors bytes=0-7") &&
        contains(log[1], "/cdn/acme/Big-8bit/"));
  // Errors: unknown repositories, gated files without and with a token (sent to the endpoint only).
  std::string e = error_of([&] { hub::repo_info("acme/Nothing"); });
  CHECK_MSG(contains(e, "acme/Nothing was not found on " + hub.url()) && contains(e, "HF_TOKEN"), e);
  env("HF_TOKEN", "");
  const hub::RepoInfo gi = hub::repo_info("acme/Gated");
  e = error_of([&] { hub::read_range(gi, "config.json", 0, 4); });
  CHECK(contains(e, "acme/Gated is gated: accept its conditions at " + hub.url() + "/acme/Gated") && contains(e, "HF_TOKEN"));
  write(g_root + "/hfhome/token", "hf_secret_from_file\n");  // huggingface_hub's token file
  CHECK(hub::token() == "hf_secret_from_file");
  hub.clear();
  CHECK(hub::read_range(gi, "config.json", 0, 2) == "{\"");
  log = hub.requests();
  CHECK(log.size() == 1 && contains(log[0], "auth=Bearer hf_secret_from_file"));
  env("HF_TOKEN", "hf_env");
  CHECK(hub::token() == "hf_env");
  hub.clear();
  hub::read_range(info, "model-00001-of-00002.safetensors", 0, 4);  // redirected to another host: no token there
  log = hub.requests();
  CHECK(log.size() == 2 && contains(log[0], "auth=Bearer hf_env") && !contains(log[1], "auth="));
  env("HF_TOKEN", "");
  ::unlink((g_root + "/hfhome/token").c_str());
  env("HF_ENDPOINT", "http://127.0.0.1:9");  // nothing listens there
  e = error_of([&] { hub::repo_info("acme/Big-8bit"); });
  CHECK(contains(e, "cannot reach http://127.0.0.1:9 for acme/Big-8bit") && contains(e, "--offline"));
  env("HF_ENDPOINT", "");
}

static void test_precheck(FakeHub& hub) {
  std::printf("the memory check before a download\n");
  const std::string cache = g_root + "/pre";
  env("HF_HUB_CACHE", cache);
  env("HF_ENDPOINT", hub.url());
  env("ESPETERO_MAC_RAM_GB", "16");
  env("HF_HOME", g_root + "/hfhome");
  hub::ResolveOptions opt;
  // MLX: 10 GB of weights and ~4 GB of state do not fit 11.5 GB; the 4-bit sibling does, the 6-bit does not.
  hub.clear();
  std::string e = error_of([&] { hub::resolve(ref("acme/Big-8bit"), opt); });
  CHECK_MSG(contains(e, "acme/Big-8bit needs about") && contains(e, "more than the 11.5 GB this Mac's GPU can use (16 GB of RAM):\n"), e);
  CHECK_MSG(contains(e, "10.0 GB  weights (MLX 8-bit, vision tower)") && contains(e, "--model acme/Big-4bit"), e);
  CHECK(!contains(e, "Big-6bit") && contains(e, "Its weights were not downloaded. `espetero recommend` lists the models that fit this Mac; --skip-memory-check downloads and runs it anyway."));
  int ranges = 0, gets = 0;
  for (const std::string& l : hub.requests()) ranges += contains(l, "bytes="), gets += contains(l, "/resolve/") && !contains(l, "bytes=");
  CHECK(ranges >= 2 && gets == 1);  // config.json whole (it decides the family), the two shards' headers by ranges
  CHECK(!is_regular(cache + "/models--acme--Big-8bit/blobs/" + FakeHub::blob_of(hub.repos["acme/Big-8bit"].files["model-00001-of-00002.safetensors"])));
  // Warned only: the download goes on, here into a disk without the room for it (refused before writing).
  opt.memory = hub::MemoryCheck::Warn;
  env("ESPETERO_HUB_DISK_FREE", "1000000000");
  e = error_of([&] { hub::resolve(ref("acme/Big-8bit"), opt); });
  CHECK_MSG(contains(e, "acme/Big-8bit needs 10.0 GB on the disk of ") && contains(e, "which has 1.0 GB free: its files were not downloaded"), e);
  // config.json came first (the family check), the weights did not.
  CHECK(!is_regular(cache + "/models--acme--Big-8bit/snapshots/" + hub.repos["acme/Big-8bit"].commit + "/model-00001-of-00002.safetensors"));
  env("ESPETERO_HUB_DISK_FREE", "");
  // GGUF: Q8_0 does not fit, its Q4_K_M sibling does; an unsupported geometry is refused before anything.
  opt.memory = hub::MemoryCheck::Refuse;
  e = error_of([&] { hub::resolve(ref("acme/Big-GGUF", "Q8_0"), opt); });
  CHECK(::access((cache + "/models--acme--Big-GGUF/refs/main").c_str(), F_OK) != 0);  // nothing of a refused GGUF model
  CHECK_MSG(contains(e, "acme/Big-GGUF:Q8_0 needs about") && contains(e, "Big-Q8_0.gguf, projector mmproj-F16.gguf"), e);
  CHECK_MSG(contains(e, "--model acme/Big-GGUF:Q4_K_M") && !contains(e, "IQ2_XXS"), e);
  env("ESPETERO_MAC_RAM_GB", "64");
  // Weights in the transformers layout (model.language_model.*): the loader reads MLX conversions only.
  FakeRepoDef raw = hub.repos["acme/Big-4bit"];
  raw.files["model.safetensors.index.json"] = {"{\"weight_map\": {\"model.language_model.embed_tokens.weight\": \"model-00001-of-00002.safetensors\"}}"};
  raw.files["model-00001-of-00002.safetensors"].content = safetensors_padded({{"model.language_model.embed_tokens.weight", {"BF16", 1000}}}, 4096);
  raw.files["model-00001-of-00002.safetensors"].size = 5096;
  raw.files.erase("model-00002-of-00002.safetensors");
  hub.repos["acme/Raw-BF16"] = raw;
  e = error_of([&] { hub::resolve(ref("acme/Raw-BF16"), opt); });
  CHECK_MSG(contains(e, "acme/Raw-BF16 is not an MLX conversion") && contains(e, "its weights were not downloaded"), e);
  e = error_of([&] { hub::resolve(ref("acme/Wide-GGUF", "Q8_0"), opt); });
  CHECK_MSG(contains(e, "acme/Wide-GGUF:Q8_0: unsupported GQA group 16") && contains(e, "its weights were not downloaded"), e);
  env("ESPETERO_MAC_RAM_GB", "");
  env("HF_ENDPOINT", "");
  env("HF_HUB_CACHE", "");
}

static std::string readlink_of(const std::string& p) {
  char b[4096];
  const ssize_t n = ::readlink(p.c_str(), b, sizeof b);
  return n > 0 ? std::string(b, size_t(n)) : std::string();
}

// Small files of a fake repository, the big ones as LFS files.
static FakeRepoDef small_repo(const std::string& commit, const std::map<std::string, std::string>& files) {
  FakeRepoDef d;
  d.commit = commit;
  for (const auto& [p, c] : files) {
    FakeFile f;
    f.content = c;
    f.lfs = c.size() > 4096 || p.find(".gguf") != std::string::npos || p.find(".safetensors") != std::string::npos;
    d.files[p] = f;
  }
  return d;
}

static void test_native_download(FakeHub& hub) {
  std::printf("native downloads into the hub cache\n");
  const std::string cache = g_root + "/dl";
  env("HF_HUB_CACHE", cache);
  env("HF_ENDPOINT", hub.url());
  env("HF_HOME", g_root + "/hfhome");
  env("HF_TOKEN", "");
  env("ESPETERO_MAC_RAM_GB", "64");
  const std::string cfg = "{\"model_type\": \"qwen3_5\", \"text_config\": {\"model_type\": \"qwen3_5_text\"}}";
  const std::string tok(5000, 't'), weights = safetensors({{"language_model.model.embed_tokens.weight", {"U32", 4096}}}) + std::string(4096, 'w');
  hub.repos["acme/Tiny-MLX"] = small_repo(std::string(40, 'a'), {{"config.json", cfg}, {"tokenizer.json", tok},
      {"model.safetensors", weights}, {"README.md", "# tiny"}, {"logo.png", "png"}, {"extra/more.safetensors", weights}});
  // MLX: config.json first, then the configuration, tokenizer and weights (not the README, images or subdirectories).
  hub.clear();
  const hub::Model m = hub::resolve(ref("acme/Tiny-MLX"), hub::ResolveOptions{});
  const std::string repo = cache + "/models--acme--Tiny-MLX", snap = repo + "/snapshots/" + std::string(40, 'a');
  CHECK(m.path == snap && m.revision == std::string(40, 'a'));
  CHECK(read(repo + "/refs/main") == std::string(40, 'a'));  // huggingface_hub writes it without a newline
  const std::string cfg_id = git_blob_id(cfg), tok_id = sha256_hex(tok), w_id = sha256_hex(weights);
  CHECK(readlink_of(snap + "/config.json") == "../../blobs/" + cfg_id && read(repo + "/blobs/" + cfg_id) == cfg);
  CHECK(readlink_of(snap + "/tokenizer.json") == "../../blobs/" + tok_id && read(snap + "/tokenizer.json") == tok);
  CHECK(readlink_of(snap + "/model.safetensors") == "../../blobs/" + w_id && read(snap + "/model.safetensors") == weights);
  CHECK(!is_regular(snap + "/README.md") && !is_regular(snap + "/logo.png") && !is_regular(snap + "/extra/more.safetensors"));
  CHECK(is_regular(cache + "/.locks/models--acme--Tiny-MLX/" + w_id + ".lock"));
  struct stat st;
  CHECK(::stat((repo + "/blobs/" + w_id).c_str(), &st) == 0 && (st.st_mode & 0777) == (0666 & ~[] { mode_t u = ::umask(0); ::umask(u); return u; }()));
  std::vector<std::string> log = hub.requests();
  CHECK(contains(log[0], "/api/models/acme/Tiny-MLX/revision/main") && contains(log[1], "/resolve/" + std::string(40, 'a') + "/config.json"));
  std::string all;
  for (const std::string& l : log) all += l + "\n";
  CHECK_MSG(log.size() == 1 + 1 + 2 * 2 + 2, all);  // listing, config.json, two LFS files (resolve + CDN), the weights' header range
  hub.clear();
  CHECK(hub::resolve(ref("acme/Tiny-MLX"), hub::ResolveOptions{}).path == snap && hub.requests().empty());  // cached
  // Another revision with the same contents: links only.
  hub::RepoInfo again = hub::repo_info("acme/Tiny-MLX");
  again.commit = std::string(40, 'b');
  hub.clear();
  hub::download(cache, again, {"config.json", "model.safetensors"}, "acme/Tiny-MLX");
  CHECK(read(repo + "/snapshots/" + std::string(40, 'b') + "/model.safetensors") == weights && hub.requests().empty());

  // A model of another family: refused after config.json, before its weights.
  hub.repos["acme/Llama-MLX"] = small_repo(std::string(40, 'c'), {{"config.json", "{\"model_type\": \"llama4\"}"},
      {"tokenizer.json", tok}, {"model.safetensors", weights}});
  std::string e = error_of([&] { hub::resolve(ref("acme/Llama-MLX"), hub::ResolveOptions{}); });
  CHECK(contains(e, "acme/Llama-MLX: model_type 'llama4' is not supported by Espetero"));
  CHECK(is_regular(cache + "/models--acme--Llama-MLX/snapshots/" + std::string(40, 'c') + "/config.json") &&
        !is_regular(cache + "/models--acme--Llama-MLX/blobs/" + w_id));

  // GGUF: the selected model, one projector (F16 first) and the MTP head in its directory.
  const std::string g4 = gguf({{"general.architecture", "qwen35"}}, gguf_4b(), {{"token_embd.weight", {2560, 248320}, 12}});
  const std::string g8 = gguf({{"general.architecture", "qwen35"}}, gguf_4b(), {{"token_embd.weight", {2560, 248320}, 8}});
  const std::string pj = gguf({{"general.architecture", "clip"}}, {}, {{"v.patch_embd.weight", {1024, 1152}, 1}});
  const std::string mtp = gguf({{"general.architecture", "qwen35"}}, {{"qwen35.block_count", 1}, {"qwen35.nextn_predict_layers", 1}},
                               {{"blk.0.nextn.eh_proj.weight", {5120, 2560}, 8}});
  hub.repos["acme/Tiny-GGUF"] = small_repo(std::string(40, 'd'), {{"Tiny-Q4_K_M.gguf", g4}, {"Tiny-Q4_K_M_XL.gguf", g4 + "x"},
      {"Tiny-Q8_0.gguf", g8}, {"mmproj-BF16.gguf", pj + "b"}, {"mmproj-F16.gguf", pj}, {"MTP/mtp-Tiny-Q4_0.gguf", mtp},
      {"imatrix.gguf", "imatrix"}, {"README.md", "# gguf"}});
  hub.clear();
  hub::Model g = hub::resolve(ref("acme/Tiny-GGUF", "Q4_K_M"), hub::ResolveOptions{});
  const std::string gs = cache + "/models--acme--Tiny-GGUF/snapshots/" + std::string(40, 'd');
  CHECK(g.path == gs + "/Tiny-Q4_K_M.gguf" && read(g.path) == g4 && g.mmproj.empty() && g.mtp.empty());
  CHECK(read(gs + "/mmproj-F16.gguf") == pj && !is_regular(gs + "/mmproj-BF16.gguf") && !is_regular(gs + "/Tiny-Q8_0.gguf"));
  CHECK(readlink_of(gs + "/MTP/mtp-Tiny-Q4_0.gguf") == "../../../blobs/" + sha256_hex(mtp) && read(gs + "/MTP/mtp-Tiny-Q4_0.gguf") == mtp);
  // No selector and several models: the choices, nothing downloaded; a selector that matches nothing; one on an MLX repository.
  hub.repos["acme/Many-GGUF"] = hub.repos["acme/Tiny-GGUF"];
  hub.clear();
  e = error_of([&] { hub::resolve(ref("acme/Many-GGUF"), hub::ResolveOptions{}); });
  CHECK_MSG(contains(e, "holds 3 GGUF models") && contains(e, "--model acme/Many-GGUF:Q8_0") && hub.requests().size() == 1, e);
  // Without a selector, the one model downloaded is the repository's model.
  CHECK(hub::resolve(ref("acme/Tiny-GGUF"), true).path == g.path);
  e = error_of([&] { hub::resolve(ref("acme/Tiny-GGUF", "Q2_K"), hub::ResolveOptions{}); });
  CHECK(contains(e, "no GGUF model of acme/Tiny-GGUF matches \"Q2_K\"") && contains(e, "--model acme/Tiny-GGUF:Q4_K_M"));
  e = error_of([&] { hub::resolve(ref("acme/Tiny-MLX", "Q4_K_M"), hub::ResolveOptions{}); });
  CHECK(contains(e, "has no GGUF model files; it is an MLX repository"));
  e = error_of([&] { hub::resolve(ref("acme/Nothing"), hub::ResolveOptions{}); });
  CHECK(contains(e, "acme/Nothing was not found on " + hub.url()));

  // An interrupted transfer resumes with a range; a partial file of an earlier run too.
  const std::string big = safetensors({{"language_model.model.embed_tokens.weight", {"U32", 300000}}}) + std::string(300000, 'z');
  const std::string big2(200000, 'y');
  hub.repos["acme/Cut-MLX"] = small_repo(std::string(40, 'e'), {{"config.json", cfg}, {"tokenizer.json", big2}, {"model.safetensors", big}});
  hub.cut["model.safetensors"] = 100000;
  const std::string cut_repo = cache + "/models--acme--Cut-MLX";
  mkdirs(cut_repo + "/blobs");
  write(cut_repo + "/blobs/" + sha256_hex(big2) + ".incomplete", big2.substr(0, 150000));
  hub.clear();
  const hub::Model cm = hub::resolve(ref("acme/Cut-MLX"), hub::ResolveOptions{});
  CHECK(read(cm.path + "/model.safetensors") == big && read(cm.path + "/tokenizer.json") == big2);
  CHECK(!is_regular(cut_repo + "/blobs/" + sha256_hex(big) + ".incomplete"));
  bool resumed = false, resumed2 = false;
  for (const std::string& l : hub.requests()) {
    resumed = resumed || (contains(l, "model.safetensors") && contains(l, "bytes=100000-"));
    resumed2 = resumed2 || (contains(l, "tokenizer.json") && contains(l, "bytes=150000-"));
  }
  CHECK(resumed && resumed2);
  // A file that does not match its checksum is an error, and its partial file goes.
  hub.repos["acme/Bad-MLX"] = small_repo(std::string(40, 'f'), {{"config.json", cfg}, {"tokenizer.json", big2}, {"model.safetensors", big + "!"}});
  hub.corrupt.insert("model.safetensors");
  e = error_of([&] { hub::resolve(ref("acme/Bad-MLX"), hub::ResolveOptions{}); });
  hub.corrupt.clear();
  CHECK_MSG(contains(e, "acme/Bad-MLX: model.safetensors does not match the Hub's checksum (sha256 " + sha256_hex(big + "!")), e);
  CHECK(!is_regular(cache + "/models--acme--Bad-MLX/blobs/" + sha256_hex(big + "!")) &&
        !is_regular(cache + "/models--acme--Bad-MLX/blobs/" + sha256_hex(big + "!") + ".incomplete"));

  // Another process holds a blob's lock: wait for it, then use the blob it wrote.
  const std::string lw = safetensors({{"language_model.model.embed_tokens.weight", {"U32", 70000}}}) + std::string(70000, 'l');
  hub.repos["acme/Locked-MLX"] = small_repo(std::string(40, '1'), {{"config.json", cfg}, {"tokenizer.json", tok}, {"model.safetensors", lw}});
  const std::string lrepo = cache + "/models--acme--Locked-MLX";
  mkdirs(cache + "/.locks/models--acme--Locked-MLX");
  mkdirs(lrepo + "/blobs");
  const int lfd = ::open((cache + "/.locks/models--acme--Locked-MLX/" + sha256_hex(lw) + ".lock").c_str(), O_RDWR | O_CREAT, 0644);
  CHECK(lfd >= 0 && ::flock(lfd, LOCK_EX) == 0);
  hub.clear();
  std::thread other([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    write(lrepo + "/blobs/" + sha256_hex(lw), lw);  // what the other process downloaded
    ::flock(lfd, LOCK_UN);
    ::close(lfd);
  });
  const hub::Model lm = hub::resolve(ref("acme/Locked-MLX"), hub::ResolveOptions{});
  other.join();
  CHECK(read(lm.path + "/model.safetensors") == lw);
  for (const std::string& l : hub.requests())  // its header was read for the memory check, never the whole file
    CHECK_MSG(!contains(l, "/cdn/acme/Locked-MLX/model.safetensors") || contains(l, "bytes=0-"), l);

  // Gated: the token from HF_TOKEN reaches the Hub, never its CDN.
  hub.repos["acme/Gated-MLX"] = small_repo(std::string(40, '2'), {{"config.json", cfg}, {"tokenizer.json", tok}, {"model.safetensors", weights}});
  hub.repos["acme/Gated-MLX"].gated = true;
  e = error_of([&] { hub::resolve(ref("acme/Gated-MLX"), hub::ResolveOptions{}); });
  CHECK(contains(e, "acme/Gated-MLX is gated: accept its conditions at " + hub.url() + "/acme/Gated-MLX") && contains(e, "HF_TOKEN"));
  env("HF_TOKEN", hub.token);
  hub.clear();
  CHECK(read(hub::resolve(ref("acme/Gated-MLX"), hub::ResolveOptions{}).path + "/model.safetensors") == weights);
  int to_hub = 0, to_cdn = 0;
  for (const std::string& l : hub.requests()) {
    if (contains(l, "/resolve/")) to_hub += contains(l, "auth=Bearer " + hub.token);
    if (contains(l, "/cdn/")) to_cdn += contains(l, "auth=");
  }
  CHECK(to_hub >= 3 && to_cdn == 0);
  env("HF_TOKEN", "");

  // Offline: the cache only, and the command that downloads.
  e = error_of([&] { hub::resolve(ref("acme/Far-GGUF", "UD-Q6_K"), true); });
  CHECK(contains(e, "acme/Far-GGUF:UD-Q6_K is not in the Hugging Face cache (" + cache + ")") && contains(e, "downloads are off") &&
        contains(e, "espetero pull acme/Far-GGUF:UD-Q6_K"));
  CHECK(hub::resolve(ref("acme/Tiny-MLX"), true).path == snap);
  env("ESPETERO_MAC_RAM_GB", "");
  env("HF_ENDPOINT", "");
  env("HF_HUB_CACHE", "");
}

static void fake_hub_repos(FakeHub& hub) {
  // acme/Big-{8,6,4}bit: Qwen3.5-4B's geometry with 10, 7.5 and 4 GB of weights in two virtual shards.
  auto mlx_repo = [&](const std::string& name, int64_t bytes) {
    FakeRepoDef d;
    d.commit = sha256_hex(name).substr(0, 40);
    d.files["config.json"] = {k4BConfig};
    const int64_t half = bytes / 2;
    std::string idx = "{\"weight_map\": {\"language_model.model.layers.0.mlp.gate_proj.weight\": \"model-00001-of-00002.safetensors\", "
                      "\"vision_tower.patch_embed.proj.weight\": \"model-00002-of-00002.safetensors\"}}";
    d.files["model.safetensors.index.json"] = {idx};
    for (int i = 1; i <= 2; ++i) {
      FakeFile f;  // a header of 4 KB, then the tensor's bytes fill the shard
      f.content = i == 1 ? safetensors_padded({{"language_model.model.embed_tokens.weight", {"U32", 1000}},
                                                {"language_model.model.layers.0.mlp.gate_proj.weight", {"U32", half - 4096 - 1000}}}, 4096)
                         : safetensors_padded({{"vision_tower.patch_embed.proj.weight", {"BF16", half - 4096}}}, 4096);
      f.size = half;
      f.lfs = true;
      d.files["model-0000" + std::to_string(i) + "-of-00002.safetensors"] = f;
    }
    hub.repos["acme/" + name] = d;
  };
  mlx_repo("Big-8bit", int64_t(10e9));
  json::Value c8 = json::parse(k4BConfig);
  c8.set("quantization", json::parse("{\"group_size\": 64, \"bits\": 8}"));
  hub.repos["acme/Big-8bit"].files["config.json"] = {json::dump(c8)};
  mlx_repo("Big-6bit", int64_t(7.5e9));
  mlx_repo("Big-4bit", int64_t(4e9));
  // acme/Big-GGUF: Q8_0 (virtual, 10 GB of Q8_0 rows), Q4_K_M (4 GB on disk), IQ2_XXS, a projector.
  auto gguf_file = [](const std::vector<std::pair<std::string, uint32_t>>& meta, uint64_t rows, uint32_t type, int64_t size) {
    FakeFile f;
    f.content = gguf({{"general.architecture", "qwen35"}}, meta,
                     {{"token_embd.weight", {2560, 248320}, type}, {"blk.0.ffn_up.weight", {2560, rows}, type}}, 200);
    f.size = size;
    f.lfs = true;
    return f;
  };
  FakeRepoDef g;
  g.commit = std::string(40, 'b');
  g.files["Big-Q8_0.gguf"] = gguf_file(gguf_4b(), 3600000, 8, int64_t(10.5e9));
  g.files["Big-Q4_K_M.gguf"] = gguf_file(gguf_4b(), 1000, 12, int64_t(4e9));
  g.files["Big-UD-IQ2_XXS.gguf"] = gguf_file(gguf_4b(), 1000, 16, int64_t(2e9));
  FakeFile pj;
  pj.content = gguf({{"general.architecture", "clip"}}, {}, {{"v.patch_embd.weight", {1024, 1152}, 1}});
  pj.lfs = true;
  g.files["mmproj-F16.gguf"] = pj;
  hub.repos["acme/Big-GGUF"] = g;
  std::vector<std::pair<std::string, uint32_t>> wide = gguf_4b();
  wide[3].second = 32;  // 32 heads over 4 KV heads... and 2 below
  wide[4].second = 2;
  FakeRepoDef w;
  w.commit = std::string(40, 'c');
  w.files["Wide-Q8_0.gguf"] = gguf_file(wide, 1000, 8, int64_t(1e9));
  hub.repos["acme/Wide-GGUF"] = w;
  FakeRepoDef gated;
  gated.commit = std::string(40, 'd');
  gated.gated = true;
  gated.files["config.json"] = {"{\"model_type\": \"qwen3_5\"}"};
  hub.repos["acme/Gated"] = gated;
  hub.token = "hf_secret_from_file";
}


// ---------------------------------------------------------------- pull, list, rm
static void test_manage(FakeHub& hub) {
  std::printf("pull, list and rm\n");
  const std::string cache = g_root + "/mc", mdir = g_root + "/mymodels";
  env("HF_HUB_CACHE", cache);
  env("HF_ENDPOINT", hub.url());
  env("HF_HOME", g_root + "/hfhome");
  env("ESPETERO_MAC_RAM_GB", "64");
  // A supported MLX model (Qwen3.5-4B's geometry), a GGUF repository with two quantizations, a projector and an MTP head.
  const std::string cfg = k4BConfig;
  const std::string w4 = safetensors({{"language_model.model.embed_tokens.weight", {"U32", 6000}}}) + std::string(6000, 'a');
  const std::string tok(5000, 't');
  hub.repos["acme/Small-4bit"] = small_repo(std::string(40, '3'), {{"config.json", cfg}, {"tokenizer.json", tok}, {"model.safetensors", w4}});
  const std::string q4 = gguf({{"general.architecture", "qwen35"}}, gguf_4b(), {{"token_embd.weight", {2560, 248320}, 12}}) + "q4";
  const std::string q8 = gguf({{"general.architecture", "qwen35"}}, gguf_4b(), {{"token_embd.weight", {2560, 248320}, 8}}) + "q8";
  const std::string pj = gguf({{"general.architecture", "clip"}}, {}, {{"v.patch_embd.weight", {1024, 1152}, 1}});
  const std::string mt = gguf({{"general.architecture", "qwen35"}}, {{"qwen35.block_count", 1}, {"qwen35.nextn_predict_layers", 1}},
                              {{"blk.0.nextn.eh_proj.weight", {5120, 2560}, 8}});
  hub.repos["acme/Small-GGUF"] = small_repo(std::string(40, '4'), {{"Small-Q4_K_M.gguf", q4}, {"Small-Q8_0.gguf", q8},
      {"mmproj-F16.gguf", pj}, {"MTP/mtp-Small.gguf", mt}, {"README.md", "# small"}});
  // pull: downloads; a second pull finds it up to date; a newer revision is fetched.
  hub::ResolveOptions opt;
  opt.memory = hub::MemoryCheck::Warn;
  hub::Model m = hub::pull(ref("acme/Small-4bit"), opt);
  CHECK(m.path == cache + "/models--acme--Small-4bit/snapshots/" + std::string(40, '3'));
  hub.clear();
  CHECK(hub::pull(ref("acme/Small-4bit"), opt).path == m.path && hub.requests().size() == 1);  // the listing only
  hub.repos["acme/Small-4bit"].commit = std::string(40, '5');
  hub.repos["acme/Small-4bit"].files["tokenizer_config.json"] = {"{}"};
  hub.clear();
  m = hub::pull(ref("acme/Small-4bit"), opt);
  CHECK(m.revision == std::string(40, '5') && read(cache + "/models--acme--Small-4bit/refs/main") == std::string(40, '5'));
  int transfers = 0;
  for (const std::string& l : hub.requests()) transfers += contains(l, "/cdn/") && !contains(l, "bytes=");
  CHECK(transfers == 0);  // the blobs were there: links only
  CHECK(hub::pull(ref("acme/Small-GGUF", "Q4_K_M"), opt).path.find("Small-Q4_K_M.gguf") != std::string::npos);
  CHECK(hub::pull(ref("acme/Small-GGUF", "Q8_0"), opt).path.find("Small-Q8_0.gguf") != std::string::npos);
  hub::ResolveOptions off;
  off.offline = true;
  CHECK(contains(error_of([&] { hub::pull(ref("acme/Small-4bit"), off); }), "needs the network"));
  // Local models: an MLX directory and a .gguf file under <dir>/<family>/<name>, and a model of another family.
  write(mdir + "/qwen3.5/Local-4bit/config.json", cfg);
  write(mdir + "/qwen3.5/Local-4bit/tokenizer.json", "{}");
  write(mdir + "/qwen3.5/Local-4bit/model.safetensors", w4);
  write(mdir + "/qwen3.5/local-gguf/Local-Q8_0.gguf", q8);
  write(mdir + "/qwen3.5/local-gguf/mmproj-F16.gguf", pj);
  write(mdir + "/llama/Other/config.json", "{\"model_type\": \"llama4\"}");
  write(mdir + "/llama/Other/tokenizer.json", "{}");
  write(mdir + "/llama/Other/model.safetensors", w4);
  int other = -1;
  const std::vector<hub::Installed> items = hub::installed(cache, mdir, &other);
  std::map<std::string, const hub::Installed*> by;
  for (const hub::Installed& it : items) by[it.id] = &it;
  CHECK(items.size() == 5 && other == 1);
  CHECK(by.count("acme/Small-4bit") && by["acme/Small-4bit"]->in_cache && by["acme/Small-4bit"]->footprint.format == "MLX" &&
        by["acme/Small-4bit"]->footprint.quant == "mixed 4/8-bit" && by["acme/Small-4bit"]->path == m.path);
  CHECK(by.count("acme/Small-GGUF:Q4_K_M") && by.count("acme/Small-GGUF:Q8_0"));
  const hub::Installed* g8 = by.count("acme/Small-GGUF:Q8_0") ? by["acme/Small-GGUF:Q8_0"] : nullptr;
  CHECK(g8 && g8->footprint.has_vision && g8->footprint.has_mtp && g8->disk == int64_t(q8.size() + pj.size() + mt.size()));
  CHECK(by.count(mdir + "/qwen3.5/Local-4bit") && !by[mdir + "/qwen3.5/Local-4bit"]->in_cache);
  CHECK(by.count(mdir + "/qwen3.5/local-gguf/Local-Q8_0.gguf") &&
        by[mdir + "/qwen3.5/local-gguf/Local-Q8_0.gguf"]->disk == int64_t(q8.size() + pj.size()));
  // rm of one quantization: its file and blob; the projector stays with the other model.
  const std::string grepo = cache + "/models--acme--Small-GGUF", gsnap = grepo + "/snapshots/" + std::string(40, '4');
  hub::Removal r = hub::plan_removal(cache, ref("acme/Small-GGUF", "Q4_K_M"));
  CHECK(!r.whole && r.links.size() == 1 && r.blobs.size() == 1 && r.bytes == int64_t(q4.size()) &&
        r.files == std::vector<std::string>{"Small-Q4_K_M.gguf"});
  hub::remove(cache, r);
  CHECK(!is_regular(gsnap + "/Small-Q4_K_M.gguf") && !is_regular(grepo + "/blobs/" + sha256_hex(q4)));
  CHECK(is_regular(gsnap + "/Small-Q8_0.gguf") && is_regular(gsnap + "/mmproj-F16.gguf") && is_regular(gsnap + "/MTP/mtp-Small.gguf"));
  // A lone GGUF model is listed by its quantization.
  bool lone = false;
  for (const hub::Installed& it : hub::installed(cache, mdir, nullptr)) lone = lone || it.id == "acme/Small-GGUF:Q8_0";
  CHECK(lone);
  // The last one: the whole repository goes (and its locks).
  r = hub::plan_removal(cache, ref("acme/Small-GGUF", "Q8_0"));
  CHECK(r.whole);
  hub::remove(cache, r);
  struct stat st;
  CHECK(::lstat(grepo.c_str(), &st) != 0 && ::lstat((cache + "/.locks/models--acme--Small-GGUF").c_str(), &st) != 0);
  // A blob two revisions share stays until neither links to it.
  const std::string mrepo = cache + "/models--acme--Small-4bit";
  r = hub::plan_removal(cache, ref("acme/Small-4bit"));
  CHECK(r.whole && r.bytes >= int64_t(w4.size() + tok.size()));
  // MLX repositories go whole; paths and repositories not in the cache are refused; nothing outside is touched.
  write(g_root + "/outside/keep.txt", "keep");
  ::symlink((g_root + "/outside").c_str(), (mrepo + "/snapshots/" + std::string(40, '5') + "/escape").c_str());
  hub::remove(cache, r);
  CHECK(::lstat(mrepo.c_str(), &st) != 0 && read(g_root + "/outside/keep.txt") == "keep");
  CHECK(contains(error_of([&] { hub::plan_removal(cache, ref("acme/Small-4bit")); }), "acme/Small-4bit is not in the Hugging Face cache"));
  CHECK(contains(error_of([&] { hub::plan_removal(cache, ref("acme/Nope", "Q4_K_M")); }), "is not in the Hugging Face cache"));
  // A cache entry that is a link out of the cache is refused.
  ::symlink((g_root + "/outside").c_str(), (cache + "/models--acme--Linked").c_str());
  CHECK(contains(error_of([&] { hub::plan_removal(cache, ref("acme/Linked")); }), "is not in the Hugging Face cache"));
  // Two snapshots: rm of a quantization in both; the shared blob goes only when nothing links to it.
  const std::string trepo = cache + "/models--acme--Two-GGUF";
  blob_link(trepo, "s1", "Two-Q4_0.gguf", "four");
  blob_link(trepo, "s1", "mmproj-F16.gguf", "proj");
  ::symlink("../../blobs/shared", (trepo + "/snapshots/s2/Two-Q4_0.gguf").c_str());
  write(trepo + "/blobs/shared", "four");
  blob_link(trepo, "s2", "Two-Q8_0.gguf", "eight");
  ::symlink("../../blobs/shared", (trepo + "/snapshots/s1/copy-of-shared.txt").c_str());
  write(trepo + "/refs/main", "s2");
  write(trepo + "/refs/old", "s1");
  r = hub::plan_removal(cache, ref("acme/Two-GGUF", "Q4_0"));
  hub::remove(cache, r);
  CHECK(::lstat((trepo + "/snapshots/s1").c_str(), &st) != 0);  // s1 had no other model: all of it went
  CHECK(!is_regular(trepo + "/snapshots/s2/Two-Q4_0.gguf") && is_regular(trepo + "/snapshots/s2/Two-Q8_0.gguf"));
  CHECK(!is_regular(trepo + "/blobs/shared"));                    // its links in s1 and s2 are gone
  CHECK(read(trepo + "/refs/main") == "s2" && ::lstat((trepo + "/refs/old").c_str(), &st) != 0);
  env("ESPETERO_MAC_RAM_GB", "");
  env("HF_ENDPOINT", "");
  env("HF_HUB_CACHE", "");
}

// ---------------------------------------------------------------- live (network)
// build/hub_test --live: downloads two tiny public repositories (~30 MB:
// tiny-random/qwen3.5 as MLX files, a 0.3 MB qwen35 GGUF) from huggingface.co
// into a temporary cache, checks them with Espetero's resolver, and with
// Python's huggingface_hub when one of the Pythons here has it: its own
// reader (scan_cache_dir, hf_hub_download with local_files_only) and its
// own download of two of the files into a second cache, which must be laid
// out the same (blob names, links, refs/main).
static const char* kLiveCheck = R"PY(
import os, sys
import huggingface_hub as h
from huggingface_hub import scan_cache_dir, hf_hub_download
ours, theirs = sys.argv[1], sys.argv[2]
info = scan_cache_dir(ours)
assert not info.warnings, info.warnings
repos = {r.repo_id: r for r in info.repos}
want = {"tiny-random/qwen3.5": ["config.json", "tokenizer.json", "model.safetensors"],
        "malaiwah/qwen3-5-tiny-random-gguf-q8_0-format": ["qwen35-tiny-Q8_0.gguf"]}
for repo, files in want.items():
    r = repos[repo]
    (rev,) = r.revisions
    assert "main" in rev.refs, rev.refs
    names = {f.file_name for f in rev.files}
    for f in files:
        assert f in names, (f, names)
        p = hf_hub_download(repo, f, cache_dir=ours, local_files_only=True)
        assert os.path.getsize(p) > 0, p
    print("  huggingface_hub %s reads %s: %d files, %d bytes, refs %s" % (h.__version__, repo, len(rev.files), r.size_on_disk, sorted(rev.refs)))
def layout(cache, repo, f):
    d = os.path.join(cache, "models--" + repo.replace("/", "--"))
    sha = open(os.path.join(d, "refs", "main")).read()
    link = os.path.join(d, "snapshots", sha, f)
    return sha, os.readlink(link), sorted(n for n in os.listdir(os.path.join(d, "blobs")) if not n.endswith(".incomplete"))
for f in ["config.json", "model.safetensors"]:
    hf_hub_download("tiny-random/qwen3.5", f, cache_dir=theirs)
    a, b = layout(ours, "tiny-random/qwen3.5", f), layout(theirs, "tiny-random/qwen3.5", f)
    assert a[0] == b[0] and a[1] == b[1] and set(b[2]) <= set(a[2]), (a, b)
    print("  same layout as huggingface_hub's own download:", f, "->", b[1])
print("OK")
)PY";

static int live() {
  std::printf("live: huggingface.co into %s/live\n", g_root.c_str());
  const std::string cache = g_root + "/live";
  env("HF_HUB_CACHE", cache);
  env("HF_ENDPOINT", "");
  env("HF_HOME", g_root + "/hfhome");  // no token: public repositories
  hub::ResolveOptions opt;
  opt.memory = hub::MemoryCheck::Skip;  // tiny random weights: their geometry is not one the engine runs
  const hub::Model a = hub::resolve(ref("tiny-random/qwen3.5"), opt);
  const hub::Model b = hub::resolve(ref("malaiwah/qwen3-5-tiny-random-gguf-q8_0-format", "Q8_0"), opt);
  hub::Model m;
  CHECK(hub::find_cached(cache, ref("tiny-random/qwen3.5"), m) && m.path == a.path);
  CHECK(hub::find_cached(cache, ref("malaiwah/qwen3-5-tiny-random-gguf-q8_0-format", "Q8_0"), m) && m.path == b.path);
  CHECK(is_regular(a.path + "/model.safetensors") && is_regular(a.path + "/tokenizer.json") && !is_regular(a.path + "/README.md"));
  // A second process: everything is there, nothing is fetched again.
  CHECK(hub::resolve(ref("tiny-random/qwen3.5"), opt).path == a.path);
  std::string python;
  const char* pe = std::getenv("HUB_TEST_PYTHON");
  for (const std::string& p : {std::string(pe ? pe : ""), std::string("python3"),
                               std::string("/Library/Frameworks/Python.framework/Versions/Current/bin/python3"),
                               std::string("/opt/homebrew/bin/python3"), std::string("/usr/local/bin/python3")})
    if (!p.empty() && std::system(("'" + p + "' -I -c 'import huggingface_hub' >/dev/null 2>&1").c_str()) == 0) {
      python = p;
      break;
    }
  if (python.empty()) {
    std::printf("  (no Python with huggingface_hub: its reader was not tried)\n");
  } else {
    write(g_root + "/scripts/live_check.py", kLiveCheck);
    const std::string cmd = "'" + python + "' -I '" + g_root + "/scripts/live_check.py' '" + cache + "' '" + g_root + "/theirs'";
    CHECK(std::system(cmd.c_str()) == 0);
  }
  std::printf("%d checks, %d failures\n", g_checks, g_fails);
  return g_fails ? 1 : 0;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);  // section names in order with the resolver's stderr
  const char* tmp = std::getenv("TMPDIR");
  std::string base = tmp && *tmp ? tmp : "/tmp";
  if (base.back() != '/') base += '/';
  base += "espetero_hub_XXXXXX";
  std::vector<char> buf(base.begin(), base.end());
  buf.push_back('\0');
  if (!::mkdtemp(buf.data())) {
    std::printf("cannot create a temporary directory\n");
    return 1;
  }
  g_root = buf.data();
  if (argc > 1 && std::string(argv[1]) == "--live") {
    const int rc = live();
    const std::string rm = "rm -rf '" + g_root + "'";
    if (std::system(rm.c_str()) != 0) std::printf("(could not remove %s)\n", g_root.c_str());
    return rc;
  }
  test_parse();
  test_env();
  test_pick();
  test_cache();
  test_geometry();
  test_headers();
  test_footprints();
  test_need();
  test_recommend();
  {
    FakeHub hub;
    fake_hub_repos(hub);
    test_hub_api(hub);
    test_precheck(hub);
    test_native_download(hub);
    test_manage(hub);
  }
  const std::string rm = "rm -rf '" + g_root + "'";
  if (std::system(rm.c_str()) != 0) std::printf("(could not remove %s)\n", g_root.c_str());
  std::printf("%d checks, %d failures\n", g_checks, g_fails);
  return g_fails ? 1 : 0;
}
