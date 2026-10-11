// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "models.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace espetero::models {

namespace {

std::string lower(std::string s) {
  for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}
bool starts_with(const std::string& s, const std::string& x) { return s.compare(0, x.size(), x) == 0; }
bool ends_with(const std::string& s, const std::string& x) {
  return s.size() >= x.size() && s.compare(s.size() - x.size(), x.size(), x) == 0;
}
bool contains(const std::string& s, const std::string& x) { return s.find(x) != std::string::npos; }

std::string read_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  if (f) ss << f.rdbuf();
  return ss.str();
}

int64_t file_size(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) ? int64_t(st.st_size) : -1;
}

std::vector<std::string> entries(const std::string& dir) {
  std::vector<std::string> out;
  if (DIR* d = ::opendir(dir.c_str())) {
    while (dirent* e = ::readdir(d)) {
      const std::string n = e->d_name;
      if (n != "." && n != "..") out.push_back(n);
    }
    ::closedir(d);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// pread of a local file, for ReadFn.
std::string pread_file(const std::string& path, int64_t offset, int64_t length) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error(path + ": cannot open (" + std::strerror(errno) + ")");
  std::string out(size_t(std::max<int64_t>(length, 0)), '\0');
  size_t got = 0;
  while (got < out.size()) {
    const ssize_t n = ::pread(fd, out.data() + got, out.size() - got, off_t(offset + int64_t(got)));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) break;
    got += size_t(n);
  }
  ::close(fd);
  out.resize(got);
  return out;
}

const char* kGgmlNames[] = {"F32",     "F16",    "Q4_0",   "Q4_1",    "Q4_2",  "Q4_3",  "Q5_0",  "Q5_1",
                            "Q8_0",    "Q8_1",   "Q2_K",   "Q3_K",    "Q4_K",  "Q5_K",  "Q6_K",  "Q8_K",
                            "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S",  "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS",
                            "I8",      "I16",    "I32",    "I64",     "F64",   "IQ1_M", "BF16"};

// llama.cpp's general.file_type names (LLAMA_FTYPE_MOSTLY_*).
std::string file_type_name(int64_t t) {
  static const std::map<int64_t, const char*> names = {
      {0, "F32"},     {1, "F16"},     {2, "Q4_0"},    {3, "Q4_1"},     {7, "Q8_0"},    {8, "Q5_0"},
      {9, "Q5_1"},    {10, "Q2_K"},   {11, "Q3_K_S"}, {12, "Q3_K_M"},  {13, "Q3_K_L"}, {14, "Q4_K_S"},
      {15, "Q4_K_M"}, {16, "Q5_K_S"}, {17, "Q5_K_M"}, {18, "Q6_K"},    {19, "IQ2_XXS"}, {20, "IQ2_XS"},
      {21, "Q2_K_S"}, {22, "IQ3_XS"}, {23, "IQ3_XXS"}, {24, "IQ1_S"},  {25, "IQ4_NL"}, {26, "IQ3_S"},
      {27, "IQ3_M"},  {28, "IQ2_S"},  {29, "IQ2_M"},  {30, "IQ4_XS"},  {31, "IQ1_M"},  {32, "BF16"},
      {36, "TQ1_0"},  {37, "TQ2_0"},  {38, "MXFP4_MOE"}};
  auto it = names.find(t);
  return it == names.end() ? std::string() : it->second;
}

// ---------------------------------------------------------------- geometry
void count_layers(Geometry& g, int interval, const json::Value* types) {
  g.n_full = g.n_linear = 0;
  if (interval <= 0) interval = 4;
  for (int i = 0; i < g.L; ++i) {
    bool lin = (i + 1) % interval != 0;
    if (types && types->is_array() && size_t(i) < types->size() && (*types)[size_t(i)].is_string())
      lin = (*types)[size_t(i)].as_string() == "linear_attention";
    lin ? ++g.n_linear : ++g.n_full;
  }
}

// The checks of Config::validate() (src/model/impl/config.h) that depend on
// the geometry, so a model the loader would refuse is not recommended.
void validate(Geometry& g, int rot) {
  g.known = g.D > 0 && g.L > 0 && g.V > 0;
  if (!g.D || (!g.I && !g.E) || !g.L || !g.H || !g.Hkv || !g.Hk || !g.Hv || !g.V) {
    g.unsupported = "incomplete Qwen3.5 configuration";
    return;
  }
  if (g.E && (g.E > 1024 || g.k < 1 || g.k > 16 || g.Ie % 64 || g.D % 64 || !g.I_shared))
    g.unsupported = "unsupported mixture-of-experts geometry";
  else if (g.HD != 256 || g.Dk != 128 || g.Dv != 128 || g.conv_k != 4 || rot != 64)
    g.unsupported = "unsupported Qwen3.5 geometry (head dim 256, rotary 64, delta heads 128, conv 4)";
  else if (g.H % g.Hkv || g.Hv % g.Hk)
    g.unsupported = "unsupported head grouping";
  else {
    const int G = g.H / g.Hkv;
    if (G != 1 && G != 2 && G != 3 && G != 4 && G != 6 && G != 8 && !(g.q4 && G == 12))
      g.unsupported = "unsupported GQA group " + std::to_string(G);
  }
}

// ---------------------------------------------------------------- GGUF parsing
class Cursor {
 public:
  Cursor(const std::string& b) : p_(reinterpret_cast<const unsigned char*>(b.data())), e_(p_ + b.size()) {}
  struct Short {};  // the bytes end early
  void need(uint64_t n) const {
    if (n > uint64_t(e_ - p_)) throw Short{};
  }
  template <class T>
  T get() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, p_, sizeof(T));
    p_ += sizeof(T);
    return v;
  }
  std::string str() {
    const uint64_t n = get<uint64_t>();
    if (n > (1ull << 30)) throw std::runtime_error("not a GGUF file (string length)");
    need(n);
    std::string s(reinterpret_cast<const char*>(p_), size_t(n));
    p_ += n;
    return s;
  }
  void skip(uint64_t n) {
    need(n);
    p_ += n;
  }

 private:
  const unsigned char* p_;
  const unsigned char* e_;
};

size_t gguf_width(uint32_t t) {
  switch (t) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return 0;
  }
}

json::Value gguf_value(Cursor& c, uint32_t t) {
  switch (t) {
    case 0: return json::Value(int64_t(c.get<uint8_t>()));
    case 1: return json::Value(int64_t(c.get<int8_t>()));
    case 2: return json::Value(int64_t(c.get<uint16_t>()));
    case 3: return json::Value(int64_t(c.get<int16_t>()));
    case 4: return json::Value(int64_t(c.get<uint32_t>()));
    case 5: return json::Value(int64_t(c.get<int32_t>()));
    case 6: return json::Value(double(c.get<float>()));
    case 7: return json::Value(c.get<uint8_t>() != 0);
    case 8: return json::Value(c.str());
    case 10: return json::Value(int64_t(c.get<uint64_t>()));
    case 11: return json::Value(c.get<int64_t>());
    case 12: return json::Value(c.get<double>());
    default: throw std::runtime_error("not a GGUF file (metadata type " + std::to_string(t) + ")");
  }
}

void gguf_skip_array(Cursor& c) {
  const uint32_t et = c.get<uint32_t>();
  const uint64_t n = c.get<uint64_t>();
  if (const size_t w = gguf_width(et)) {
    if (n > (1ull << 40) / w) throw std::runtime_error("not a GGUF file (array length)");
    c.skip(n * w);
  } else if (et == 8) {
    for (uint64_t i = 0; i < n; ++i) c.skip(c.get<uint64_t>());
  } else {
    throw std::runtime_error("not a GGUF file (array type " + std::to_string(et) + ")");
  }
}

// A GGUF file's header, read in growing pieces (the tokenizer's arrays put
// the tensor table ~11 MB in for the Qwen3.5 vocabulary).
GgufHeader read_gguf(const FileRef& file, const ReadFn& read) {
  // A projector has no tokenizer: its header is small.
  const int64_t first = lower(file.path).find("mmproj") != std::string::npos ? (1 << 20) : (16 << 20);
  int64_t want = std::min<int64_t>(file.size > 0 ? file.size : first, first);
  std::string buf = read(file.path, 0, want);
  GgufHeader h;
  while (!parse_gguf(buf, h)) {
    if ((file.size > 0 && int64_t(buf.size()) >= file.size) || buf.size() < size_t(want))
      throw std::runtime_error(file.path + ": truncated GGUF header");
    if (buf.size() >= (64u << 20)) throw std::runtime_error(file.path + ": GGUF header larger than 64 MB");
    const int64_t more = std::min<int64_t>(int64_t(buf.size()), (file.size > 0 ? file.size : INT64_MAX) - int64_t(buf.size()));
    want = int64_t(buf.size()) + more;
    buf += read(file.path, int64_t(buf.size()), more);
  }
  return h;
}

// A safetensors header, read with one request when it is under 1 MB.
std::vector<Tensor> read_safetensors(const FileRef& file, const ReadFn& read) {
  const int64_t first = std::min<int64_t>(file.size > 0 ? file.size : (256 << 10), 256 << 10);
  std::string buf = read(file.path, 0, first);
  size_t need = 0;
  try {
    return parse_safetensors(buf, &need);
  } catch (const std::exception&) {
    if (need <= buf.size() || need > (256u << 20)) throw;
  }
  buf += read(file.path, int64_t(buf.size()), int64_t(need - buf.size()));
  return parse_safetensors(buf);
}

// bits / group of an MLX module (config "quantization" overrides by module path).
struct Quant {
  int bits = 0, group = 0;  // 0: not quantized
};

struct MlxQuant {
  int bits = 0, group = 64;
  std::map<std::string, Quant> overrides;
  std::set<std::string> plain;  // modules the config keeps unquantized (false)
  Quant of(const std::string& module) const {
    if (plain.count(module)) return {};
    auto it = overrides.find(module);
    return it != overrides.end() ? it->second : Quant{bits, group};
  }
};

MlxQuant mlx_quant(const json::Value& config) {
  MlxQuant q;
  const json::Value* qz = config.get("quantization");
  if (!qz || !qz->is_object()) qz = config.get("quantization_config");
  if (!qz || !qz->is_object()) return q;
  const std::string mode = qz->get_string("mode", "affine");
  q.bits = int(qz->get_int("bits", 4));
  q.group = int(qz->get_int("group_size", 64));
  const json::Object& o = qz->as_object();
  for (size_t i = 0; i < o.size(); ++i) {
    const json::Value& v = o.vals[i];
    if (v.is_object()) q.overrides[o.keys[i]] = {int(v.get_int("bits", q.bits)), int(v.get_int("group_size", q.group))};
    else if (v.is_bool() && !v.as_bool()) q.plain.insert(o.keys[i]);
  }
  (void)mode;
  return q;
}

std::string mlx_quant_label(const json::Value& config, const MlxQuant& q) {
  const json::Value* qz = config.get("quantization");
  if (!qz || !qz->is_object()) qz = config.get("quantization_config");
  if (!q.bits) {
    const json::Value* tc = config.get("text_config");
    std::string dt = config.get_string("torch_dtype", config.get_string("dtype", ""));
    if (dt.empty() && tc && tc->is_object()) dt = tc->get_string("torch_dtype", tc->get_string("dtype", ""));
    if (dt == "bfloat16") return "bf16";
    if (dt == "float16") return "fp16";
    if (dt == "float32") return "fp32";
    return dt.empty() ? "unquantized" : dt;
  }
  const std::string mode = qz ? qz->get_string("mode", "affine") : "affine";
  if (mode != "affine") return mode;
  std::set<int> bits = {q.bits};
  for (const auto& [k, v] : q.overrides)
    if (v.bits) bits.insert(v.bits);
  if (bits.size() == 1) return std::to_string(q.bits) + "-bit";
  std::string s = "mixed ";
  for (int b : bits) s += (s.size() > 6 ? "/" : "") + std::to_string(b);
  return s + "-bit";
}

// The engine's GPU width for MLX codes of `bits` (matrix.h gpu_bits_for).
int gpu_bits(int bits) { return bits == 4 || bits == 5 || bits == 6 || bits == 8 ? bits : 8; }

std::string module_of(const std::string& tensor) {
  for (const char* s : {".weight", ".scales", ".biases"})
    if (ends_with(tensor, s)) return tensor.substr(0, tensor.size() - std::strlen(s));
  return tensor;
}

}  // namespace

// ---------------------------------------------------------------- geometry
int64_t Geometry::kv_bytes_per_token(bool mtp) const {
  if (gemma) return int64_t(n_full) * 2 * int64_t(Hkv_g) * HDg * 2;  // the global layers (forest.h)
  const int64_t layers = n_full + (mtp ? 1 : 0);
  // + the Flash-Next sparse attention index: 336 bytes per token and layer (forest.h)
  return layers * 2 * int64_t(Hkv) * HD * 2 + (q4 ? layers * 336 : 0);
}

int64_t Geometry::slot_bytes(bool mtp) const {
  if (gemma) return int64_t(n_swa) * ((window + 63) / 64 * 64) * 2 * int64_t(Hkv) * HD * 2;  // the slot's ring
  const int64_t C = 2 * int64_t(Hk) * Dk + int64_t(Hv) * Dv;
  return (3 * C + int64_t(Hv) * Dv * Dk) * n_linear * 4 + (mtp ? int64_t(17) * (q4 ? int64_t(hc) * D : D) * 4 : 0);
}

int64_t Geometry::row_bytes() const {
  if (gemma) {  // ensure_scratch for Gemma 4: tokens, h, xn, proj, Q, Kb, Vb, O, act (+ the MoE rows)
    const int64_t hd = std::max(HD, HDg), kv = std::max(int64_t(Hkv) * HD, int64_t(Hkv_g) * HDg);
    int64_t r = 4 + 6 * int64_t(D) + 4 * int64_t(ld) + 4 * int64_t(H) * hd + 4 * kv + 2 * std::max<int64_t>(I, 2 * int64_t(D));
    if (E) r += 4 * int64_t(D) + int64_t(k) * (2 * int64_t(Ie) + 4 * int64_t(D)) + 8 * int64_t(E) + 4 * int64_t(D) + 16 * int64_t(k);
    return r;
  }
  // ensure_scratch (forest.h): tokens, h, xn, proj, qo, gb, y, yn, Q, Kb, Vb, O, act
  const int64_t C = 2 * int64_t(Hk) * Dk + int64_t(Hv) * Dv, Vd = int64_t(Hv) * Dv, kv = int64_t(Hkv) * HD;
  const int64_t lin_ld = C + Vd + 2 * int64_t(Hv), full_ld = int64_t(H) * HD * 2 + 2 * kv;
  const int64_t imax = std::max(I, I_shared);
  int64_t r = 4 + 6 * int64_t(D) + 4 * std::max(lin_ld, full_ld) + 2 * C + 8 * int64_t(Hv) + 6 * Vd + 4 * int64_t(H) * HD +
              4 * kv + 2 * std::max<int64_t>(imax, 2 * int64_t(D));
  if (E) {  // ensure_moe_scratch: the k routed rows and the shared one, router logits, combine parts
    r += int64_t(k + 1) * (2 * int64_t(Ie) + 4 * int64_t(D)) + 4 * int64_t(E) + 32 * int64_t(E + 1) + 16 * int64_t(k);
  }
  if (q4) {  // ensure_q4_scratch: the residual streams (W = hc x D)
    const int64_t W = int64_t(hc) * D;
    r += 26 * W + 16 * int64_t(D) + 4 * int64_t(E) + 6 * int64_t(hc_rank) + 16 * int64_t(hc) + 144;
  }
  return r;
}

int64_t Geometry::readout_bytes() const {
  const int64_t N = 256;  // LoadOptions::max_readouts
  const int64_t parts = (int64_t(V) + 4095) / 4096;
  return N * (4 * int64_t(V) + 8 * int64_t(D) + 2 * std::max<int64_t>(int64_t(Hv) * Dv, int64_t(H) * std::max(HD, HDg)) +
              2 * std::max(I, I_shared) + parts * 264 + 300);
}

void gemma_validate(Geometry& g);

// Gemma 4: config_from_gemma's reading (src/model/impl/config.h) and
// the checks of its Config::validate() branch.
Geometry gemma_geometry(const json::Value& root, const json::Value& tc) {
  Geometry g;
  g.gemma = true;
  g.D = int(tc.get_int("hidden_size", 1536));
  g.I = int(tc.get_int("intermediate_size", 6144));
  g.L = int(tc.get_int("num_hidden_layers", 35));
  g.H = int(tc.get_int("num_attention_heads", 8));
  g.Hkv = int(tc.get_int("num_key_value_heads", 1));
  g.HD = int(tc.get_int("head_dim", 256));
  const json::Value* ghv = tc.get("global_head_dim");
  const int64_t ghd = !ghv ? 512 : (ghv->is_number() ? int64_t(ghv->as_double()) : 0);
  g.HDg = ghd > 0 ? int(ghd) : g.HD;
  const bool k_eq_v = tc.get_bool("attention_k_eq_v", false);
  const json::Value* gkv = tc.get("num_global_key_value_heads");
  g.Hkv_g = k_eq_v && gkv && gkv->is_number() ? int(gkv->as_double()) : g.Hkv;
  g.V = int(root.get_int("vocab_size", tc.get_int("vocab_size", 262144)));
  g.window = int(tc.get_int("sliding_window", 512));
  if (tc.get_bool("enable_moe_block", false)) {
    g.E = int(tc.get_int("num_experts", 0));
    g.k = int(tc.get_int("top_k_experts", 0));
    g.Ie = int(tc.get_int("moe_intermediate_size", 0));
  }
  const json::Value* types = tc.get("layer_types");
  const int pattern = int(tc.get_int("sliding_window_pattern", 5));
  for (int i = 0; i < g.L; ++i) {
    bool s = pattern <= 0 || (i % pattern) != pattern - 1;
    if (types && types->is_array() && size_t(i) < types->size() && (*types)[size_t(i)].is_string())
      s = (*types)[size_t(i)].as_string() == "sliding_attention";
    s ? ++g.n_swa : ++g.n_full;
  }
  g.ld = std::max({g.H * g.HD + 2 * g.Hkv * g.HD, g.H * g.HDg + (k_eq_v ? 1 : 2) * g.Hkv_g * g.HDg, g.D});
  if (tc.get_int("hidden_size_per_layer_input", 0) > 0 || tc.get_int("num_kv_shared_layers", 0) > 0)
    g.unsupported = "Gemma 4 with per-layer input embeddings or KV-shared layers (E2B, E4B)";
  gemma_validate(g);
  return g;
}

void gemma_validate(Geometry& g) {
  g.known = g.D > 0 && g.L > 0 && g.V > 0;
  if (!g.unsupported.empty()) return;
  if (!g.D || !g.I || !g.L || !g.H || !g.Hkv || !g.Hkv_g || !g.V || !g.window)
    g.unsupported = "incomplete Gemma 4 configuration";
  else if (g.HD != 256 || g.HDg != 512)
    g.unsupported = "unsupported Gemma 4 head sizes (sliding 256, global 512)";
  else if (g.H % g.Hkv || g.H % g.Hkv_g)
    g.unsupported = "unsupported head grouping";
  else if (const int Gs = g.H / g.Hkv, Gg = g.H / g.Hkv_g;
           (Gs != 1 && Gs != 2 && Gs != 4 && Gs != 8) || (Gg != 2 && Gg != 4 && Gg != 8 && Gg != 16))
    g.unsupported = "unsupported Gemma 4 GQA groups";
  else if (g.E && (g.E > 1024 || g.k < 1 || g.k > 16 || g.Ie % 64))
    g.unsupported = "unsupported mixture-of-experts geometry";
}

Geometry geometry_from_config(const json::Value& root) {
  Geometry g;
  if (!root.is_object()) return g;
  const json::Value* tcp = root.get("text_config");
  const json::Value& tc = tcp && tcp->is_object() ? *tcp : root;
  {
    const std::string top = root.get_string("model_type", ""), text = tc.get_string("model_type", "");
    if (top == "gemma4" || top == "gemma4_text" || text == "gemma4" || text == "gemma4_text") return gemma_geometry(root, tc);
  }
  g.D = int(tc.get_int("hidden_size", 0));
  g.I = int(tc.get_int("intermediate_size", 0));
  g.E = int(tc.get_int("num_experts", 0));
  if (g.E) {
    g.k = int(tc.get_int("num_experts_per_tok", 0));
    g.Ie = int(tc.get_int("moe_intermediate_size", 0));
    g.I_shared = int(tc.get_int("shared_expert_intermediate_size", 0));
  }
  g.L = int(tc.get_int("num_hidden_layers", 0));
  g.H = int(tc.get_int("num_attention_heads", 0));
  g.Hkv = int(tc.get_int("num_key_value_heads", 0));
  g.HD = int(tc.get_int("head_dim", 256));
  g.Hk = int(tc.get_int("linear_num_key_heads", 0));
  g.Hv = int(tc.get_int("linear_num_value_heads", 0));
  g.Dk = int(tc.get_int("linear_key_head_dim", 128));
  g.Dv = int(tc.get_int("linear_value_head_dim", 128));
  g.conv_k = int(tc.get_int("linear_conv_kernel_dim", 4));
  g.V = int(tc.get_int("vocab_size", 0));
  const json::Value* rp = tc.get("rope_parameters");
  if (!rp || !rp->is_object()) rp = tc.get("rope_scaling");
  if (rp && !rp->is_object()) rp = nullptr;
  const double prf = rp ? rp->get_double("partial_rotary_factor", tc.get_double("partial_rotary_factor", 0.25))
                        : tc.get_double("partial_rotary_factor", 0.25);
  count_layers(g, int(tc.get_int("full_attention_interval", 4)), tc.get("layer_types"));
  const std::string mt = tc.get_string("model_type", root.get_string("model_type", ""));
  if (starts_with(mt, "qwen4_exp")) {
    g.q4 = true;
    g.hc = int(tc.get_int("hc_count", 4));
    g.hc_rank = int(tc.get_int("hc_lowrank", 0));
  }
  validate(g, int(g.HD * prf));
  return g;
}

int64_t Tensor::elements() const {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return shape.empty() ? 0 : n;
}

// ---------------------------------------------------------------- safetensors
std::vector<Tensor> parse_safetensors(const std::string& bytes, size_t* need) {
  if (bytes.size() < 8) throw std::runtime_error("not a safetensors file (shorter than 8 bytes)");
  uint64_t n = 0;
  std::memcpy(&n, bytes.data(), 8);
  if (n > (256ull << 20)) throw std::runtime_error("not a safetensors file (header length)");
  if (need) *need = size_t(8 + n);
  if (bytes.size() < 8 + n) throw std::runtime_error("safetensors header cut short");
  const json::Value h = json::parse(std::string_view(bytes).substr(8, size_t(n)));
  if (!h.is_object()) throw std::runtime_error("not a safetensors file (header is not an object)");
  std::vector<Tensor> out;
  const json::Object& o = h.as_object();
  for (size_t i = 0; i < o.size(); ++i) {
    if (o.keys[i] == "__metadata__") continue;
    const json::Value& v = o.vals[i];
    Tensor t;
    t.name = o.keys[i];
    t.dtype = v.get_string("dtype", "");
    if (const json::Value* s = v.get("shape"); s && s->is_array())
      for (const json::Value& d : s->as_array()) t.shape.push_back(d.as_int());
    if (const json::Value* off = v.get("data_offsets"); off && off->is_array() && off->size() == 2)
      t.bytes = (*off)[1].as_int() - (*off)[0].as_int();
    out.push_back(std::move(t));
  }
  return out;
}

// ---------------------------------------------------------------- GGUF
const json::Value* GgufHeader::get(const std::string& key) const {
  for (const auto& [k, v] : meta)
    if (k == key) return &v;
  return nullptr;
}
int64_t GgufHeader::get_int(const std::string& key, int64_t def) const {
  const json::Value* v = get(key);
  return v && v->is_number() ? int64_t(v->as_double()) : def;
}
std::string GgufHeader::get_string(const std::string& key, const std::string& def) const {
  const json::Value* v = get(key);
  return v && v->is_string() ? v->as_string() : def;
}

bool parse_gguf(const std::string& bytes, GgufHeader& out) {
  out = GgufHeader{};
  if (bytes.size() >= 4 && std::memcmp(bytes.data(), "GGUF", 4) != 0) throw std::runtime_error("not a GGUF file");
  Cursor c(bytes);
  try {
    c.skip(4);
    if (c.get<uint32_t>() < 2) throw std::runtime_error("GGUF version 1 is not supported");
    const uint64_t n_tensors = c.get<uint64_t>(), n_kv = c.get<uint64_t>();
    if (n_tensors > (1u << 24) || n_kv > (1u << 24)) throw std::runtime_error("not a GGUF file (counts)");
    for (uint64_t i = 0; i < n_kv; ++i) {
      std::string key = c.str();
      const uint32_t t = c.get<uint32_t>();
      if (t == 9) gguf_skip_array(c);
      else out.meta.emplace_back(std::move(key), gguf_value(c, t));
    }
    for (uint64_t i = 0; i < n_tensors; ++i) {
      Tensor t;
      t.name = c.str();
      const uint32_t nd = c.get<uint32_t>();
      if (nd > 8) throw std::runtime_error("not a GGUF file (tensor rank)");
      for (uint32_t d = 0; d < nd; ++d) t.shape.push_back(int64_t(c.get<uint64_t>()));
      t.gguf_type = int(c.get<uint32_t>());
      t.dtype = ggml_type_name(t.gguf_type);
      c.get<uint64_t>();  // data offset
      out.tensors.push_back(std::move(t));
    }
  } catch (const Cursor::Short&) {
    return false;
  }
  return true;
}

std::string ggml_type_name(int type) {
  if (type >= 0 && size_t(type) < sizeof kGgmlNames / sizeof *kGgmlNames) return kGgmlNames[type];
  if (type == 39) return "MXFP4";
  return "type " + std::to_string(type);
}

int64_t gguf_engine_bytes(int type, int64_t elements) {
  // Bits per weight in GPU memory: codes (4, 5 and 6 bits in planes, others
  // as bytes) plus scales: affine half scale + bias per group (4 bytes per
  // group: 1 bit per weight at group 32, 2 at 16), Q6_K's int8 sub-scale per
  // 16 and fp16 super-scale per 256, Q8_0's fp16 scale per 32.
  double bits;
  switch (type) {
    case 0: bits = 32; break;   // F32 (vectors; matrices are converted to fp16 by the caller)
    case 1: case 30: bits = 16; break;
    case 2: case 3: bits = 4 + 1; break;     // Q4_0, Q4_1
    case 6: case 7: bits = 5 + 1; break;     // Q5_0, Q5_1
    case 8: bits = 8 + 0.5; break;           // Q8_0, native
    case 10: case 11: bits = 4 + 2; break;   // Q2_K, Q3_K: 4-bit codes, group 16
    case 12: bits = 4 + 1; break;            // Q4_K, affine group 32
    case 13: bits = 5 + 1; break;            // Q5_K
    case 14: bits = 6 + 0.5 + 0.0625; break; // Q6_K, native
    case 20: case 23: bits = 8 + 1; break;   // IQ4_NL, IQ4_XS: 8-bit codes
    default: bits = 8; break;                // not read by the engine (the loader refuses it)
  }
  return int64_t(std::ceil(double(elements) * bits / 8));
}

Geometry geometry_from_gguf(const GgufHeader& h) {
  Geometry g;
  const std::string arch = h.get_string("general.architecture", "");
  const std::string a = arch + ".";
  if (arch == "gemma4") {
    // Gemma 4: the per-layer arrays are not read here; each layer's
    // kind and KV heads come from its attention matrices (q: H x head size,
    // k: KV heads x head size; global K = V layers have no v).
    g.gemma = true;
    g.D = int(h.get_int(a + "embedding_length", 0));
    g.I = int(h.get_int(a + "feed_forward_length", 0));
    g.L = int(h.get_int(a + "block_count", 0));
    g.H = int(h.get_int(a + "attention.head_count", 0));
    g.HD = int(h.get_int(a + "attention.key_length_swa", 256));
    g.HDg = int(h.get_int(a + "attention.key_length", 512));
    g.window = int(h.get_int(a + "attention.sliding_window", 0));
    g.E = int(h.get_int(a + "expert_count", 0));
    g.k = int(h.get_int(a + "expert_used_count", 0));
    g.Ie = int(h.get_int(a + "expert_feed_forward_length", 0));
    std::map<std::string, int64_t> q_out, k_out;
    std::set<std::string> has_v;
    for (const Tensor& t : h.tensors) {
      if (t.name == "token_embd.weight" && t.shape.size() == 2) g.V = int(t.shape[1]);
      if (t.shape.size() != 2 || t.name.compare(0, 4, "blk.") != 0) continue;
      const size_t dot = t.name.find('.', 4);
      if (dot == std::string::npos) continue;
      const std::string blk = t.name.substr(4, dot - 4), rest = t.name.substr(dot);
      if (rest == ".attn_q.weight") q_out[blk] = t.shape[1];
      else if (rest == ".attn_k.weight") k_out[blk] = t.shape[1];
      else if (rest == ".attn_v.weight") has_v.insert(blk);
    }
    bool k_eq_v = false;
    for (int i = 0; i < g.L && g.H > 0; ++i) {
      const std::string b = std::to_string(i);
      const int hd = int(q_out[b] / g.H);
      const int hkv = hd > 0 ? int(k_out[b] / hd) : 0;
      const bool sliding = hd == g.HD;
      sliding ? ++g.n_swa : ++g.n_full;
      (sliding ? g.Hkv : g.Hkv_g) = hkv;
      if (!sliding) k_eq_v = !has_v.count(b);
    }
    g.ld = std::max({g.H * g.HD + 2 * g.Hkv * g.HD, g.H * g.HDg + (k_eq_v ? 1 : 2) * g.Hkv_g * g.HDg, g.D});
    if (h.get_int(a + "embedding_length_per_layer_input", 0) > 0 || h.get_int(a + "attention.shared_kv_layers", 0) > 0)
      g.unsupported = "Gemma 4 with per-layer input embeddings or KV-shared layers (E2B, E4B)";
    gemma_validate(g);
    return g;
  }
  g.D = int(h.get_int(a + "embedding_length", 0));
  g.I = int(h.get_int(a + "feed_forward_length", 0));
  if (arch == "qwen35moe") {
    g.E = int(h.get_int(a + "expert_count", 0));
    g.k = int(h.get_int(a + "expert_used_count", 0));
    g.Ie = int(h.get_int(a + "expert_feed_forward_length", 0));
    g.I_shared = int(h.get_int(a + "expert_shared_feed_forward_length", 0));
  }
  g.L = int(h.get_int(a + "block_count", 0) - h.get_int(a + "nextn_predict_layers", 0));
  g.H = int(h.get_int(a + "attention.head_count", 0));
  g.Hkv = int(h.get_int(a + "attention.head_count_kv", 0));
  g.HD = int(h.get_int(a + "attention.key_length", 256));
  g.conv_k = int(h.get_int(a + "ssm.conv_kernel", 4));
  g.Dk = g.Dv = int(h.get_int(a + "ssm.state_size", 128));
  g.Hk = int(h.get_int(a + "ssm.group_count", 0));
  g.Hv = int(h.get_int(a + "ssm.time_step_rank", 0));
  for (const Tensor& t : h.tensors)
    if (t.name == "token_embd.weight" && t.shape.size() == 2) g.V = int(t.shape[1]);
  count_layers(g, int(h.get_int(a + "full_attention_interval", 4)), nullptr);
  validate(g, int(h.get_int(a + "rope.dimension_count", 64)));
  return g;
}

// ---------------------------------------------------------------- footprints
namespace {

// The 2-bit draft copy of the LM head (2-bit codes, fp16 scale + bias per 64).
int64_t coarse_head_bytes(const Geometry& g) { return int64_t(g.V) * g.D * 5 / 16; }

// GPU bytes of a GGUF file's tensors; `skip_shared`: an MTP head's own
// embedding and output (the decoder's are used).
struct GgufBytes {
  int64_t all = 0, mtp = 0, embed = 0;
  int64_t dense_elements = 0, quant_elements = 0;  // matrices kept dense (F16/BF16/F32) or quantized
  std::string dense_type;
  bool tied = true;
  int embed_type = -1;
  int64_t embed_elements = 0;
  std::map<std::string, int64_t> by_type;  // quantized matrices' elements per type name
};

GgufBytes gguf_bytes(const GgufHeader& h, int decoder_layers, bool skip_shared) {
  GgufBytes b;
  for (const Tensor& t : h.tensors) {
    const int64_t n = t.elements();
    if (t.name == "output.weight") b.tied = false;
    if (skip_shared && (t.name == "token_embd.weight" || t.name == "output.weight" || t.name == "output_norm.weight"))
      continue;
    int64_t bytes;
    if (t.name == "token_embd.weight") {
      // Rows are gathered from an affine copy (load_embedding): 4-bit codes
      // stay nibbles, others become bytes, a half scale and bias per group.
      b.embed_type = t.gguf_type;
      b.embed_elements = n;
      const int code_bits = (t.gguf_type == 2 || t.gguf_type == 3 || t.gguf_type == 10 || t.gguf_type == 11 ||
                             t.gguf_type == 12) ? 4 : 8;
      const int group = (t.gguf_type == 10 || t.gguf_type == 11 || t.gguf_type == 14) ? 16 : 32;
      bytes = t.gguf_type == 0 || t.gguf_type == 1 || t.gguf_type == 30 ? n * 2 : n * code_bits / 8 + n / group * 4;
      b.embed = bytes;
    } else if (t.gguf_type == 0) {
      bytes = t.shape.size() >= 2 && t.shape[0] > 1 && t.shape[1] > 1 ? n * 2 : n * 4;  // matrices become fp16
    } else {
      bytes = gguf_engine_bytes(t.gguf_type, n);
      if (t.gguf_type != 1 && t.gguf_type != 30) b.by_type[ggml_type_name(t.gguf_type)] += n;
    }
    if (t.shape.size() >= 2 && t.shape[0] > 1 && t.shape[1] > 1) {
      if (t.gguf_type == 0 || t.gguf_type == 1 || t.gguf_type == 30) {
        b.dense_elements += n;
        if (b.dense_type.empty() || t.gguf_type != 0) b.dense_type = ggml_type_name(t.gguf_type);
      } else {
        b.quant_elements += n;
      }
    }
    b.all += bytes;
    // The MTP (nextn) block follows the decoder blocks.
    if (starts_with(t.name, "blk.")) {
      const int layer = std::atoi(t.name.c_str() + 4);
      if (layer >= decoder_layers) b.mtp += bytes;
    }
  }
  return b;
}

std::string dominant(const std::map<std::string, int64_t>& by_type) {
  std::string best;
  int64_t n = -1;
  for (const auto& [k, v] : by_type)
    if (v > n) best = k, n = v;
  return best;
}

}  // namespace

Footprint footprint_gguf(const FileRef& model, const FileRef* mmproj, const FileRef* mtp, const ReadFn& read) {
  Footprint f;
  f.format = "GGUF";
  const GgufHeader h = read_gguf(model, read);
  f.geo = geometry_from_gguf(h);
  const std::string arch = h.get_string("general.architecture", "");
  const int nextn = int(h.get_int(arch + ".nextn_predict_layers", 0));
  const GgufBytes b = gguf_bytes(h, f.geo.L, false);
  f.disk = model.size;
  f.weights = b.all;
  if (b.tied && b.embed_type >= 0)  // a tied LM head is a second, GEMM-layout copy of the embedding
    f.weights += b.embed_type == 0 ? b.embed_elements * 2 : gguf_engine_bytes(b.embed_type, b.embed_elements);
  f.base = f.weights;
  f.arch = arch;
  f.quant = file_type_name(h.get_int("general.file_type", -1));
  if (f.quant.empty() || b.dense_elements > b.quant_elements) f.quant = b.dense_elements > b.quant_elements ? b.dense_type : dominant(b.by_type);
  f.what = model.path.substr(model.path.find_last_of('/') + 1);
  f.mtp = b.mtp;
  f.has_mtp = nextn > 0;
  if (f.has_mtp) f.what += " with its MTP head";
  if (!f.has_mtp && mtp) {
    const GgufHeader mh = read_gguf(*mtp, read);
    const Geometry mg = geometry_from_gguf(mh);
    const GgufBytes mb = gguf_bytes(mh, mg.L, true);
    f.mtp = mb.all;
    f.weights += mb.all;
    f.disk += mtp->size;
    f.has_mtp = true;
    f.what += ", MTP head " + mtp->path.substr(mtp->path.find_last_of('/') + 1);
  }
  if (f.has_mtp && b.embed_type != 0 && b.embed_type != 1 && b.embed_type != 30) {
    const int64_t c = coarse_head_bytes(f.geo);
    f.mtp += c;
    f.weights += c;
  }
  if (mmproj) {
    const GgufHeader vh = read_gguf(*mmproj, read);
    int64_t v = 0;
    for (const Tensor& t : vh.tensors) {
      const int64_t n = t.elements();
      // Vision matrices are loaded as fp16 (load_vdense), vectors as floats.
      v += t.shape.size() >= 2 && t.shape[0] > 1 && t.shape[1] > 1 ? n * 2 : n * 4;
    }
    f.vision = v;
    f.weights += v;
    f.disk += mmproj->size;
    f.has_vision = true;
    f.what += ", projector " + mmproj->path.substr(mmproj->path.find_last_of('/') + 1);
  }
  return f;
}

Footprint footprint_mlx(const json::Value& config, const std::vector<FileRef>& files, const ReadFn& read) {
  Footprint f;
  f.format = "MLX";
  f.geo = geometry_from_config(config);
  {
    const json::Value* tc = config.get("text_config");
    f.arch = tc && tc->is_object() ? tc->get_string("model_type", "") : std::string();
    if (f.arch.empty()) f.arch = config.get_string("model_type", "");
  }
  const MlxQuant q = mlx_quant(config);
  f.quant = mlx_quant_label(config, q);
  auto size_of = [&](const std::string& p) -> const FileRef* {
    for (const FileRef& r : files)
      if (r.path == p) return &r;
    return nullptr;
  };
  // The files the loader reads (SafetensorsSource): the index's weight map, else model.safetensors.
  std::vector<std::string> shards;
  if (const FileRef* idx = size_of("model.safetensors.index.json")) {
    const json::Value v = json::parse(read(idx->path, 0, idx->size));
    if (const json::Value* wm = v.get("weight_map"); wm && wm->is_object())
      for (const json::Value& s : wm->as_object().vals)
        if (s.is_string() && std::find(shards.begin(), shards.end(), s.as_string()) == shards.end())
          shards.push_back(s.as_string());
  } else if (size_of("model.safetensors")) {
    shards.push_back("model.safetensors");
  }
  const json::Value* tcp = config.get("text_config");
  const json::Value* tie = tcp && tcp->is_object() ? tcp->get("tie_word_embeddings") : nullptr;
  if (!tie || tie->is_null()) tie = config.get("tie_word_embeddings");
  const bool tied = tie && tie->is_bool() && tie->as_bool();
  bool lm_head = false, own_mtp = false, vision_tower = false;
  int64_t embed_bytes = 0;
  bool embed_quantized = false;
  for (const std::string& s : shards) {
    const FileRef* r = size_of(s);
    if (!r) throw std::runtime_error(s + ": listed in model.safetensors.index.json but missing");
    f.disk += r->size;
    for (const Tensor& t : read_safetensors(*r, read)) {
      int64_t bytes = t.bytes;
      const std::string mod = module_of(t.name);
      const Quant mq = q.bits ? q.of(mod) : Quant{};
      const bool code = mq.bits && t.dtype == "U32" && ends_with(t.name, ".weight");
      if (code && gpu_bits(mq.bits) != mq.bits) bytes = bytes * gpu_bits(mq.bits) / mq.bits;  // 2- and 3-bit codes widen to bytes
      if (mq.bits && mq.group == 128 && (ends_with(t.name, ".scales") || ends_with(t.name, ".biases")))
        bytes *= 2;  // group 128 is split in two groups of 64 (exact)
      if (starts_with(t.name, "language_model.model.embed_tokens.")) {
        // Rows are gathered from nibbles or bytes (load_embedding: 5- and 6-bit codes widen).
        if (code && mq.bits != 4 && mq.bits != 8) bytes = t.bytes * 8 / mq.bits;
        embed_bytes += bytes;
        embed_quantized = embed_quantized || code;
      }
      if (starts_with(t.name, "language_model.lm_head.")) lm_head = true;
      if (t.name == "language_model.model.embed_tokens.weight") f.mlx_layout = true;
      if (t.name == "language_model.mtp.fc.weight") own_mtp = true;
      if (t.name == "vision_tower.patch_embed.proj.weight") vision_tower = true;
      if (starts_with(t.name, "language_model.mtp.") || starts_with(t.name, "mtp.")) {
        // Flash-Next drafts with a drafter directory; the file's own mtp.* head is not loaded.
        if (f.geo.q4) continue;
        f.mtp += bytes;
      } else if (f.geo.q4 && (contains(t.name, ".switch_mlp.") || contains(t.name, ".ngram_embedding."))) {
        f.streamed += t.bytes;  // read from the SSD (expert cache, n-gram rows)
        continue;
      }
      if (starts_with(t.name, "vision_tower.")) f.vision += bytes;
      f.weights += bytes;
    }
  }
  if (tied && !lm_head) f.weights += embed_bytes;  // the LM head: a GEMM-layout copy of the embedding
  f.base = f.weights;
  f.has_mtp = own_mtp && !f.geo.q4;
  if (f.has_mtp && embed_quantized) {
    const int64_t c = coarse_head_bytes(f.geo);
    f.mtp += c;
    f.weights += c;
  }
  const json::Value* vc = config.get("vision_config");
  f.has_vision = vision_tower && vc && vc->is_object();
  if (!f.has_vision) {
    f.weights -= f.vision;
    f.vision = 0;
  }
  f.what = std::to_string(shards.size()) + (shards.size() == 1 ? " weight file" : " weight files");
  return f;
}

Footprint footprint_local(const std::string& path0, const std::string& mmproj, const std::string& mtp) {
  std::string path = path0;
  while (path.size() > 1 && path.back() == '/') path.pop_back();
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) throw std::runtime_error(path + ": no such file or directory");
  if (S_ISDIR(st.st_mode)) {
    const std::string text = read_text(path + "/config.json");
    if (text.empty()) throw std::runtime_error(path + ": no config.json");
    const json::Value config = json::parse(text);
    std::vector<FileRef> files;
    for (const std::string& n : entries(path))
      if (const int64_t s = file_size(path + "/" + n); s >= 0) files.push_back({n, s});
    Footprint f = footprint_mlx(config, files, [&](const std::string& p, int64_t off, int64_t len) {
      return pread_file(path + "/" + p, off, len);
    });
    // Flash-Next: the drafter directory beside the model (load.h: a sibling
    // named *MTP* / *mtp* whose config says qwen4_exp_mtp), bf16 loaded as 8-bit.
    if (f.geo.q4 && mtp != "none") {
      std::string dir = mtp;
      if (dir.empty()) {
        const std::string parent = path.substr(0, path.find_last_of('/'));
        for (const std::string& n : entries(parent))
          if ((contains(n, "MTP") || contains(n, "mtp")) &&
              contains(read_text(parent + "/" + n + "/config.json"), "\"qwen4_exp_mtp\"")) {
            dir = parent + "/" + n;
            break;
          }
      }
      if (!dir.empty()) {
        const std::string dc = read_text(dir + "/config.json");
        int64_t bytes = 0;
        for (const std::string& n : entries(dir))
          if (ends_with(n, ".safetensors")) bytes += std::max<int64_t>(0, file_size(dir + "/" + n));
        if (!contains(dc, "\"quantization")) bytes /= 2;  // bf16 drafter requantized to 8 bits
        f.mtp += bytes;
        f.weights += bytes;
        f.has_mtp = bytes > 0;
        f.what += ", MTP drafter " + dir.substr(dir.find_last_of('/') + 1);
      }
    } else if (!mtp.empty() && mtp != "none") {
      // --mtp PATH: an MTP head from a GGUF file or an MLX directory.
      int64_t bytes = 0;
      if (ends_with(lower(mtp), ".gguf")) {
        const int64_t s = file_size(mtp);
        if (s > 0) {
          const FileRef r{mtp, s};
          const GgufHeader mh = read_gguf(r, [](const std::string& p, int64_t off, int64_t len) { return pread_file(p, off, len); });
          bytes = gguf_bytes(mh, geometry_from_gguf(mh).L, true).all;
        }
      } else {
        for (const std::string& n : entries(mtp))
          if (ends_with(n, ".safetensors")) bytes += std::max<int64_t>(0, file_size(mtp + "/" + n));
      }
      if (!f.has_mtp) {
        bytes += coarse_head_bytes(f.geo);
        f.has_mtp = true;
      }
      f.mtp += bytes;
      f.weights += bytes;
    }
    return f;
  }
  // A .gguf file and what the engine finds next to it (load.h find_mmproj / find_mtp).
  const std::string dir = path.find('/') == std::string::npos ? "." : path.substr(0, path.find_last_of('/'));
  auto first = [](const std::string& d, const std::string& prefix) -> std::string {
    for (const std::string& n : entries(d)) {
      const std::string l = lower(n);
      if (starts_with(l, prefix) && ends_with(l, ".gguf") && file_size(d + "/" + n) >= 0) return d + "/" + n;
    }
    return "";
  };
  const ReadFn rd = [](const std::string& p, int64_t off, int64_t len) { return pread_file(p, off, len); };
  const FileRef model{path, int64_t(st.st_size)};
  std::string pj = mmproj == "none" ? "" : (mmproj.empty() ? first(dir, "mmproj") : mmproj);
  std::string mt = mtp == "none" ? "" : (mtp.empty() ? first(dir, "mtp") : mtp);
  if (mt.empty() && mtp.empty()) mt = first(dir + "/MTP", "mtp");
  FileRef pr{pj, pj.empty() ? 0 : file_size(pj)}, mr{mt, mt.empty() ? 0 : file_size(mt)};
  return footprint_gguf(model, pj.empty() || pr.size < 0 ? nullptr : &pr, mt.empty() || mr.size < 0 ? nullptr : &mr, rd);
}

// ---------------------------------------------------------------- need
Need need(const Footprint& f, const Plan& p) {
  Need n;
  const bool mtp = p.mtp && f.has_mtp;
  const bool vision = p.vision && f.has_vision;
  n.weights = f.weights - (mtp ? 0 : f.mtp) - (vision ? 0 : f.vision);
  const Geometry& g = f.geo;
  if (g.known && g.n_full + g.n_linear > 0) {
    const int64_t slot = g.slot_bytes(mtp), tok = g.kv_bytes_per_token(mtp);
    // The pools' floor (model.mm): 8 user slots at least (or --slots), KV for
    // two full contexts (or --kv-gb); the automatic budget shrinks to it on a
    // small Mac. --memory-gb (3 GB by default for Flash-Next) when larger.
    // The 8 temporary slots and the zero slot come on top (make_pools).
    const int64_t user = p.slots > 0 ? p.slots : 8;
    const int64_t kv = p.kv_gb > 0 ? int64_t(p.kv_gb * 1e9) : int64_t(2) * std::max(p.max_ctx, 1) * tok;
    const int64_t budget = p.memory_gb > 0 ? int64_t(p.memory_gb * 1e9) : (g.q4 ? int64_t(3e9) : 0);
    n.pools = std::max(user * slot + kv, budget) + (g.gemma ? 0 : 9 * slot);  // Gemma 4: rings for user slots only
    n.scratch = int64_t(std::max(p.max_rows, 64)) * g.row_bytes() + g.readout_bytes() + int64_t(0.25e9) +
                (vision ? int64_t(0.25e9) : 0);
  } else {
    // Geometry unknown: the margins model.mm's expert-cache sizing uses.
    n.pools = std::max<int64_t>(int64_t(3e9), int64_t(p.memory_gb * 1e9));
    n.scratch = int64_t(2.5e9);
  }
  if (g.q4 && p.expert_cache_gb > 0) n.cache = int64_t(p.expert_cache_gb * 1e9);
  return n;
}

int64_t Mac::usable() const {
  if (gpu > 0 && ram > 0) return std::min(gpu, ram);
  return gpu > 0 ? gpu : ram * 2 / 3;
}

Mac mac_with_ram(double ram_gb) {
  Mac m;
  m.ram = int64_t(ram_gb * double(1ull << 30));
  // macOS's default GPU working set: 2/3 of the RAM up to 32 GB, 3/4 above
  // (8 GB: 5.7 GB; 16 GB: 11.5 GB; 36 GB: 29.0 GB; 64 GB: 51.5 GB).
  m.gpu = ram_gb <= 32 ? m.ram / 3 * 2 : m.ram / 4 * 3;
  return m;
}

// ---------------------------------------------------------------- the check
std::string gb(int64_t bytes) {
  char b[32];
  const double g = double(bytes) / 1e9;
  std::snprintf(b, sizeof b, g >= 99.95 ? "%.0f GB" : "%.1f GB", g);
  return b;
}

std::string size_text(int64_t b) {
  if (b >= 1000000000) return gb(b);
  char t[32];
  if (b >= 1000000) std::snprintf(t, sizeof t, "%.1f MB", double(b) / 1e6);
  else std::snprintf(t, sizeof t, "%.0f KB", double(b) / 1e3);
  return t;
}

namespace {
std::string ram_gb(const Mac& m) { return std::to_string(int64_t(std::llround(double(m.ram) / double(1ull << 30)))) + " GB"; }
std::string pad(const std::string& s, size_t n) { return s.size() >= n ? s : std::string(n - s.size(), ' ') + s; }
}  // namespace

Verdict judge(const Footprint& f, const Plan& p, const Mac& mac) {
  Verdict v;
  v.need = need(f, p);
  v.usable = mac.usable();
  v.fits = v.need.total() <= v.usable;
  if (!v.fits && p.kv_gb <= 0)
    for (int ctx = 1 << 30; ctx >= 2048; ctx /= 2) {
      if (ctx >= p.max_ctx) continue;
      Plan q = p;
      q.max_ctx = ctx;
      if (need(f, q).total() <= v.usable) {
        v.context_that_fits = ctx;
        break;
      }
    }
  return v;
}

std::string refusal(const std::string& label, const Footprint& f, const Plan& p, const Mac& mac, const Verdict& v,
                    const std::vector<Alternative>& alternatives, const std::string& ending) {
  const Need& n = v.need;
  std::string m = label + " needs about " + gb(n.total()) + " of GPU memory, more than the " + gb(v.usable) +
                  " this Mac's GPU can use (" + ram_gb(mac) + " of RAM" + (mac.chip.empty() ? "" : ", " + mac.chip) + "):\n";
  const bool mtp = p.mtp && f.has_mtp, vision = p.vision && f.has_vision;
  std::string parts = f.format == "GGUF" ? f.what : f.format + " " + f.quant;
  if (f.format != "GGUF") parts += std::string(mtp ? ", MTP head" : "") + (vision ? ", vision tower" : "");
  m += "  " + pad(gb(n.weights), 8) + "  weights (" + parts + ")\n";
  const int64_t user = p.slots > 0 ? p.slots : 8;
  m += "  " + pad(gb(n.pools), 8) + "  conversation state: " +
       (p.kv_gb > 0 ? "KV cache of --kv-gb " + gb(int64_t(p.kv_gb * 1e9))
                    : "KV cache for 2 x " + std::to_string(p.max_ctx) + " tokens (--context)") +
       " and " + std::to_string(user + 9) + " sequence states" + (p.memory_gb > 0 ? " (--memory-gb)" : "") + "\n";
  m += "  " + pad(gb(n.scratch), 8) + "  scratch for " + std::to_string(p.max_rows) + "-token passes\n";
  if (n.cache) m += "  " + pad(gb(n.cache), 8) + "  expert cache (--expert-cache-gb)\n";
  if (f.streamed) m += "(" + gb(f.streamed) + " of experts and n-gram table stay on the SSD.)\n";
  if (v.context_that_fits) {
    Plan q = p;
    q.max_ctx = v.context_that_fits;
    m += "It fits with --context " + std::to_string(v.context_that_fits) + " (about " + gb(need(f, q).total()) + ").\n";
  }
  std::vector<std::pair<int64_t, std::string>> fit;
  std::set<std::string> seen = {label};
  for (const Alternative& a : alternatives) {
    if (!seen.insert(a.model).second) continue;
    const Verdict av = judge(a.footprint, p, mac);
    if (av.fits)
      fit.push_back({av.need.total(), "  --model " + a.model + "   " + gb(a.footprint.disk) + " of files, needs about " +
                                          gb(av.need.total()) + "\n"});
  }
  std::sort(fit.begin(), fit.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  if (!fit.empty()) {
    m += "Smaller files of this model that fit:\n";
    for (size_t i = 0; i < fit.size() && i < 4; ++i) m += fit[i].second;
  }
  return m + ending;
}

std::vector<std::string> bit_siblings(const std::string& name) {
  std::vector<std::string> out;
  // "...-8bit" / "...-MLX-8Bit": the digits before the last "bit"
  const std::string l = lower(name);
  size_t at = l.rfind("bit");
  while (at != std::string::npos && at > 0) {
    size_t a = at;
    while (a > 0 && std::isdigit(static_cast<unsigned char>(l[a - 1]))) --a;
    if (a < at && (a == 0 || !std::isalnum(static_cast<unsigned char>(l[a - 1])))) {
      const std::string bits = name.substr(a, at - a);
      for (const char* b : {"8", "6", "5", "4"})
        if (bits != b) out.push_back(name.substr(0, a) + b + name.substr(at));
      return out;
    }
    at = at == 0 ? std::string::npos : l.rfind("bit", at - 1);
  }
  // oMLX names: "...-oQ8e-mtp", "...-oQ6-mtp"
  const size_t o = l.rfind("-oq");
  if (o != std::string::npos && o + 3 < l.size() && std::isdigit(static_cast<unsigned char>(l[o + 3]))) {
    const char cur = l[o + 3];
    for (char b : {'8', '6', '5', '4'})
      if (b != cur) {
        std::string s = name;
        s[o + 3] = b;
        out.push_back(s);
      }
  }
  return out;
}

std::string tilde(const std::string& path) {
  const char* h = std::getenv("HOME");
  const std::string home = h ? h : "";
  if (!home.empty() && home != "/" && starts_with(path, home + "/")) return "~" + path.substr(home.size());
  return path;
}

std::vector<Alternative> local_alternatives(const std::string& path0, const std::string& mmproj, const std::string& mtp) {
  std::vector<Alternative> out;
  std::string path = path0;
  while (path.size() > 1 && path.back() == '/') path.pop_back();
  const size_t slash = path.find_last_of('/');
  const std::string dir = slash == std::string::npos ? "." : path.substr(0, slash);
  const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
  auto add = [&](const std::string& p) {
    try {
      out.push_back({tilde(p), footprint_local(p, mmproj, mtp)});
    } catch (const std::exception&) {
    }
  };
  struct stat st;
  if (::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
    for (const std::string& s : bit_siblings(name))
      if (::stat((dir + "/" + s + "/config.json").c_str(), &st) == 0) add(dir + "/" + s);
    for (Alternative& a : catalog_alternatives(name)) out.push_back(std::move(a));
    return out;
  }
  for (const std::string& n : entries(dir)) {
    const std::string l = lower(n);
    if (n == name || !ends_with(l, ".gguf") || starts_with(l, "mmproj") || starts_with(l, "mtp") || contains(l, "imatrix"))
      continue;
    // Split models (-00001-of-00003) are not read.
    if (l.size() > 20 && l.compare(l.size() - 20, 5, "-0000") == 0 && contains(l.substr(l.size() - 15), "-of-")) continue;
    add(dir + "/" + n);
  }
  std::string stem = name.substr(0, name.size() > 5 ? name.size() - 5 : name.size());  // without .gguf
  for (Alternative& a : catalog_alternatives(stem)) out.push_back(std::move(a));
  return out;
}

double gguf_quant_factor(const std::string& name) {
  std::string u = name;
  for (char& c : u) c = char(std::toupper(static_cast<unsigned char>(c)));
  if (contains(u, "IQ1") || contains(u, "IQ2") || contains(u, "IQ3") || contains(u, "TQ1") || contains(u, "TQ2") ||
      contains(u, "MXFP4"))
    return 0;  // tensor types the engine does not read
  if (contains(u, "IQ4")) return 2.0;
  if (contains(u, "Q2_K")) return 2.2;
  if (contains(u, "Q3_K")) return 1.7;
  if (contains(u, "Q4_0") || contains(u, "Q4_K")) return 1.1;
  if (contains(u, "Q5_0") || contains(u, "Q5_K")) return 1.09;
  return 1.0;  // Q4_1, Q5_1, Q6_K, Q8_0, F16, BF16
}

}  // namespace espetero::models
