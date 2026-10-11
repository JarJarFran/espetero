// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "model_meta.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "../util/model_family.h"

namespace espetero {

namespace {

bool ends_with(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() > n && s.compare(s.size() - n, n, suffix) == 0;
}

bool read_text(const std::string& path, std::string& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::stringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

// ---- GGUF metadata: the scalars of a few keys, read from the mapped header
// (tensor data is never touched).
struct Scalar {
  enum Kind { Int, Float, String } kind = Int;
  int64_t i = 0;
  double f = 0;
  bool f32 = false;  // a float32 value (its decimal form is recovered below)
  std::string s;
};

class GgufMeta {
 public:
  // false: not a readable GGUF file (`why` says so).
  bool read(const std::string& path, const std::map<std::string, bool>& wanted, std::string& why) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return why = "cannot open", false;
    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size < 24) {
      ::close(fd);
      return why = "not a GGUF file", false;
    }
    const size_t size = size_t(st.st_size);
    void* base = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) return why = "cannot map", false;
    p_ = static_cast<const unsigned char*>(base);
    e_ = p_ + size;
    bool ok = false;
    try {
      ok = parse(wanted, why);
    } catch (const std::exception&) {
      why = "truncated GGUF metadata";
    }
    ::munmap(base, size);
    return ok;
  }
  const Scalar* get(const std::string& key) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? nullptr : &it->second;
  }

 private:
  bool parse(const std::map<std::string, bool>& wanted, std::string& why) {
    if (std::memcmp(p_, "GGUF", 4) != 0) return why = "not a GGUF file", false;
    p_ += 4;
    if (u32() < 2) return why = "GGUF version 1 is not supported", false;
    u64();  // tensors
    const uint64_t n_kv = u64();
    for (uint64_t k = 0; k < n_kv; ++k) {
      const std::string key = str();
      const uint32_t t = u32();
      if (wanted.count(key) && t != 9) {
        kv_[key] = scalar(t);
      } else {
        skip(t);
      }
    }
    return true;
  }
  void need(uint64_t n) const {
    if (n > uint64_t(e_ - p_)) throw std::runtime_error("truncated");
  }
  template <class T>
  T get() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, p_, sizeof(T));
    p_ += sizeof(T);
    return v;
  }
  uint32_t u32() { return get<uint32_t>(); }
  uint64_t u64() { return get<uint64_t>(); }
  std::string str() {
    const uint64_t n = u64();
    need(n);
    std::string s(reinterpret_cast<const char*>(p_), size_t(n));
    p_ += n;
    return s;
  }
  static size_t width(uint32_t t) {
    switch (t) {
      case 0: case 1: case 7: return 1;
      case 2: case 3: return 2;
      case 4: case 5: case 6: return 4;
      case 10: case 11: case 12: return 8;
      default: return 0;
    }
  }
  Scalar scalar(uint32_t t) {
    Scalar v;
    switch (t) {
      case 0: v.i = get<uint8_t>(); break;
      case 1: v.i = get<int8_t>(); break;
      case 2: v.i = get<uint16_t>(); break;
      case 3: v.i = get<int16_t>(); break;
      case 4: v.i = get<uint32_t>(); break;
      case 5: v.i = get<int32_t>(); break;
      case 7: v.i = get<uint8_t>() != 0; break;
      case 10: v.i = int64_t(get<uint64_t>()); break;
      case 11: v.i = get<int64_t>(); break;
      case 6: v.kind = Scalar::Float, v.f = get<float>(), v.f32 = true; break;
      case 12: v.kind = Scalar::Float, v.f = get<double>(); break;
      case 8: v.kind = Scalar::String, v.s = str(); break;
      default: throw std::runtime_error("unknown GGUF metadata type");
    }
    return v;
  }
  void skip(uint32_t t) {
    if (t == 8) {
      const uint64_t n = u64();
      need(n);
      p_ += n;
    } else if (t == 9) {
      const uint32_t et = u32();
      const uint64_t n = u64();
      if (const size_t w = width(et)) {
        if (n > uint64_t(e_ - p_) / w) throw std::runtime_error("truncated");
        p_ += n * w;
      } else {
        for (uint64_t i = 0; i < n; ++i) skip(et);
      }
    } else if (const size_t w = width(t)) {
      need(w);
      p_ += w;
    } else {
      throw std::runtime_error("unknown GGUF metadata type");
    }
  }

  const unsigned char* p_ = nullptr;
  const unsigned char* e_ = nullptr;
  std::map<std::string, Scalar> kv_;
};

// A float32 metadata value as the decimal it was written from (0.95f is
// 0.949999988...; 7 significant digits recover 0.95).
double decimal(const Scalar& s) {
  if (s.kind == Scalar::Int) return double(s.i);
  if (!s.f32) return s.f;
  char b[32];
  std::snprintf(b, sizeof b, "%.7g", s.f);
  return std::strtod(b, nullptr);
}

// The defaults kept from raw values (nullopt: absent or not a number).
void keep(SamplingDefaults& d, std::optional<double> temperature, std::optional<double> top_p,
          std::optional<double> top_k, std::optional<double> min_p) {
  if (temperature && std::isfinite(*temperature) && *temperature >= 0 && *temperature <= 2) d.temperature = *temperature;
  if (top_p && std::isfinite(*top_p) && *top_p > 0 && *top_p <= 1) d.top_p = *top_p;
  if (min_p && std::isfinite(*min_p) && *min_p >= 0 && *min_p <= 1) d.min_p = *min_p;
  if (top_k && std::isfinite(*top_k) && *top_k == std::floor(*top_k) && *top_k <= 1e6)
    d.top_k = *top_k <= 0 ? 0 : int(*top_k);
}

}  // namespace

void check_model_supported(const std::string& path) {
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0)
    throw std::runtime_error(path + ": no such file or directory (--model takes an MLX model directory, a .gguf file "
                                    "or a Hugging Face repo id such as mlx-community/Qwen3.8-27B-4bit)");
  if (S_ISDIR(st.st_mode)) {
    std::string text;
    if (!read_text(path + "/config.json", text))
      throw std::runtime_error(path + ": no config.json (an MLX model directory holds config.json, tokenizer.json and "
                                      "the *.safetensors weights)");
    json::Value config;
    try {
      config = json::parse(text);
    } catch (const std::exception& e) {
      throw std::runtime_error(path + "/config.json: " + e.what());
    }
    if (const std::string m = family::mlx_check(path, config); !m.empty()) throw std::runtime_error(m);
    return;
  }
  if (!ends_with(path, ".gguf")) throw std::runtime_error(path + ": not an MLX model directory or a .gguf file");
  GgufMeta meta;
  std::string why;
  if (!meta.read(path, {{"general.architecture", true}}, why)) throw std::runtime_error(path + ": " + why);
  const Scalar* arch = meta.get("general.architecture");
  const std::string a = arch && arch->kind == Scalar::String ? arch->s : std::string();
  if (!family::gguf_arch_supported(a)) throw std::runtime_error(family::unsupported_gguf(path, a));
}

SamplingDefaults read_sampling_defaults(const std::string& path) {
  SamplingDefaults d;
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0) return d;
  auto num = [](const json::Value* v) -> std::optional<double> {
    return v && v->is_number() ? std::optional<double>(v->as_double()) : std::nullopt;
  };
  if (S_ISDIR(st.st_mode)) {
    std::string text;
    if (!read_text(path + "/generation_config.json", text)) return d;
    json::Value g;
    try {
      g = json::parse(text);
    } catch (const std::exception&) {
      return d;
    }
    if (!g.is_object()) return d;
    keep(d, num(g.get("temperature")), num(g.get("top_p")), num(g.get("top_k")), num(g.get("min_p")));
    if (!d.empty()) d.source = "generation_config.json";
    return d;
  }
  if (!ends_with(path, ".gguf")) return d;
  static const char* const keys[] = {"general.sampling.temp", "general.sampling.top_p", "general.sampling.top_k",
                                     "general.sampling.min_p"};
  std::map<std::string, bool> wanted;
  for (const char* k : keys) wanted[k] = true;
  GgufMeta meta;
  std::string why;
  if (!meta.read(path, wanted, why)) return d;
  auto val = [&](const char* k) -> std::optional<double> {
    const Scalar* s = meta.get(k);
    return s && s->kind != Scalar::String ? std::optional<double>(decimal(*s)) : std::nullopt;
  };
  keep(d, val(keys[0]), val(keys[1]), val(keys[2]), val(keys[3]));
  if (!d.empty()) d.source = "GGUF general.sampling";
  return d;
}

json::Value SamplingDefaults::to_json(bool enabled) const {
  const SamplingDefaults none;
  const SamplingDefaults& d = enabled ? *this : none;
  json::Value o = json::Value::object();
  o.set("temperature", d.temperature.value_or(1.0));
  o.set("top_p", d.top_p.value_or(1.0));
  o.set("top_k", int64_t(d.top_k.value_or(0)));
  o.set("min_p", d.min_p.value_or(0.0));
  std::string from;
  int n = 0;
  for (const auto& [name, set] : {std::pair<const char*, bool>{"temperature", bool(d.temperature)},
                                  {"top_p", bool(d.top_p)}, {"top_k", bool(d.top_k)}, {"min_p", bool(d.min_p)}})
    if (set) from += std::string(n++ ? ", " : "") + name;
  if (!enabled) o.set("source", "OpenAI defaults (--no-generation-config)");
  else if (!n) o.set("source", "OpenAI defaults (the model sets none)");
  else o.set("source", source + " (" + from + (n == 4 ? ")" : "; the others: OpenAI defaults)"));
  return o;
}

}  // namespace espetero
