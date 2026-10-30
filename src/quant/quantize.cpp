// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// espetero quantize: a Hugging Face BF16 checkpoint (Qwen3.5 / Qwen3.6
// family, dense Gemma 4, Qwen4-Exp / Qwen3.8-Flash-Next) -> an EQ checkpoint
// (Espetero's own format), without Python. Qwen4-Exp can also be converted
// from an MLX checkpoint (oQ4e-mtp: its matrices dequantized), and its
// routed experts pruned to a kept list per layer (--keep-experts).
// What it does:
// calibration statistics from Espetero's own forward of the BF16 weights,
// the converter's presets and rules, its encoders, and its output: the MLX
// directory layout with NAME.weight / .scales / .super (/ .codebook) per EQ
// matrix, config.json "quantization" and "eq_format" {"version": 2}, the MTP
// head as MLX affine, routers and norms as the source holds them.

#include "quantize.h"

#include <dirent.h>
#include <sys/stat.h>

#include <cctype>
#include <cstdio>
#include <fstream>
#include <regex>
#include <set>

#include "../model/impl/tensors.h"

#include "../model/api.h"
#include "../model/impl/hf_source.h"
#include "calib.h"
#include "encode.h"
#include "vq.h"
#include "st.h"

namespace espetero::quant {

using native::ends_with;
using native::HFSource;
using native::TensorView;

namespace {

// ---------------------------------------------------------------- rules

// Python's fnmatch.fnmatch (case-sensitive on macOS): * ? [seq] [!seq].
bool fnmatch(const char* s, const char* p) {
  for (;; ++s, ++p) {
    if (*p == '*') {
      while (*p == '*') ++p;
      if (!*p) return true;
      for (; *s; ++s)
        if (fnmatch(s, p)) return true;
      return false;
    }
    if (!*s) return !*p;
    if (*p == '?') continue;
    if (*p == '[') {
      const char* q = p + 1;
      bool neg = false;
      if (*q == '!') neg = true, ++q;
      bool hit = false;
      const char* start = q;
      while (*q && (*q != ']' || q == start)) {
        if (q[1] == '-' && q[2] && q[2] != ']') {
          hit |= *s >= q[0] && *s <= q[2];
          q += 3;
        } else {
          hit |= *s == *q;
          ++q;
        }
      }
      if (!*q) {  // no closing bracket: a literal '['
        if (*s != '[') return false;
        continue;
      }
      if (hit == neg) return false;
      p = q;
      continue;
    }
    if (*s != *p) return false;
  }
}

const char* kSmall[] = {"*self_attn.v_proj", "*self_attn.k_proj", "*self_attn.o_proj", "*linear_attn.in_proj_[ab]",
                        "*linear_attn.out_proj"};
const char* kRouters[] = {"*mlp.gate", "*mlp.shared_expert_gate"};
const std::vector<std::string> kPresets = {"eq4", "eq5", "eq6", "eq4x", "eq5x", "eq4d", "eq5d", "eq6h", "eq5dh", "eq4dh", "eq4f", "eq4s"};

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> v;
  for (size_t a = 0;;) {
    const size_t b = s.find(c, a);
    v.push_back(s.substr(a, b - a));
    if (b == std::string::npos) break;
    a = b + 1;
  }
  return v;
}

std::string join(const std::vector<std::string>& v, const std::string& sep) {
  std::string o;
  for (size_t i = 0; i < v.size(); ++i) o += (i ? sep : "") + v[i];
  return o;
}

std::string replace_all(std::string s, const std::string& a, const std::string& b) {
  for (size_t p = 0; (p = s.find(a, p)) != std::string::npos; p += b.size()) s.replace(p, a.size(), b);
  return s;
}

// The presets
std::string preset(const std::string& name) {
  if (name == "eq3dhb") {  // eq3dh, gate/up one width up in layers 0-3 and 36-39 (block probe on Qwen3.6-35B-A3B)
    const std::vector<std::string> r = split(preset("eq3dh"), ';');
    std::vector<std::string> mine;
    for (int l : {0, 1, 2, 3, 36, 37, 38, 39})
      for (const char* g : {"gate_proj", "up_proj"})
        mine.push_back("*layers." + std::to_string(l) + ".mlp.switch_mlp." + g + "=eql:4:16:cb+imat");
    size_t nrout = 0;
    for (const std::string& x : r) nrout += ends_with(x, "=bf16") ? 1 : 0;
    std::vector<std::string> out(r.begin(), r.begin() + long(nrout));
    out.insert(out.end(), mine.begin(), mine.end());
    out.insert(out.end(), r.begin() + long(nrout), r.end());
    return join(out, ";");
  }
  if (name == "eq4dhb") {  // eq4dhc re-allocated per block of layers on calibration text (eq_alloc.py)
    const std::vector<std::string> r = split(preset("eq4dhc"), ';');
    std::vector<std::pair<std::pair<int, int>, std::string>> moves = {
        {{0, 8}, "*mlp.switch_mlp.gate_proj=eql:5:16+imat"}, {{0, 8}, "*mlp.switch_mlp.up_proj=eql:5:16+imat"},
        {{36, 40}, "*mlp.switch_mlp.gate_proj=eql:5:16+imat"}, {{36, 40}, "*mlp.switch_mlp.up_proj=eql:5:16+imat"},
        {{16, 32}, "*mlp.switch_mlp.down_proj=eql:4:16:cb+imat"}};
    for (const char* g : {"self_attn.*", "linear_attn.*", "mlp.shared_expert.*"})
      moves.push_back({{16, 32}, std::string("*") + g + "=eql:6:16+gptq"});
    std::vector<std::string> mine;
    for (const auto& [ls, g] : moves)
      for (int l = ls.first; l < ls.second; ++l) mine.push_back("*layers." + std::to_string(l) + "." + g.substr(1));
    size_t nrout = 0;
    for (const std::string& x : r) nrout += ends_with(x, "=bf16") ? 1 : 0;
    std::vector<std::string> out(r.begin(), r.begin() + long(nrout));
    out.insert(out.end(), mine.begin(), mine.end());
    out.insert(out.end(), r.begin() + long(nrout), r.end());
    return join(out, ";");
  }
  if (ends_with(name, "c") && std::find(kPresets.begin(), kPresets.end(), name.substr(0, name.size() - 1)) != kPresets.end()) {
    std::string r = replace_all(preset(name.substr(0, name.size() - 1)), "*switch_mlp*=eql:4:16+imat", "*switch_mlp*=eql:4:16:cb+imat");
    return replace_all(r, "*switch_mlp.down_proj=eql:4:16+imat", "*switch_mlp.down_proj=eql:4:16:cb+imat");
  }
  std::vector<std::string> r;
  auto small = [&](const std::string& f) {
    for (const char* g : kSmall) r.push_back(std::string(g) + "=" + f);
  };
  if (name == "eq4") {
    small("eql:6:16+gptq");
    r.insert(r.end(), {"*switch_mlp*=eql:4:16+imat", "*embed_tokens=affine:6:64:f16+gptq", "*lm_head=affine:6:64:f16+gptq",
                       "*=eql:4:16+gptq"});
  } else if (name == "eq5") {
    small("eql:6:16+gptq");
    r.insert(r.end(), {"*switch_mlp*=eql:5:16+imat", "*embed_tokens=affine:8:64:f16", "*lm_head=affine:8:64:f16", "*=eql:5:16+gptq"});
  } else if (name == "eq4x") {
    r = {"*switch_mlp*=eql:4:16+imat", "*embed_tokens=affine:6:64:f16+gptq", "*lm_head=affine:6:64:f16+gptq", "*=eql:6:16+gptq"};
  } else if (name == "eq5x") {
    r = {"*switch_mlp*=eql:5:16+imat", "*embed_tokens=affine:8:64:f16", "*lm_head=affine:6:64:f16+gptq", "*=eql:6:16+gptq"};
  } else if (name == "eq4d") {
    r = {"*switch_mlp.down_proj=eql:5:16+imat", "*switch_mlp*=eql:4:16+imat", "*embed_tokens=affine:6:64:f16+gptq",
         "*lm_head=affine:6:64:f16+gptq", "*=eql:6:16+gptq"};
  } else if (name == "eq5d") {
    r = {"*switch_mlp.down_proj=eql:6:16+imat", "*switch_mlp*=eql:5:16+imat", "*embed_tokens=affine:8:64:f16",
         "*lm_head=affine:6:64:f16+gptq", "*=eql:6:16+gptq"};
  } else if (name == "eq6h") {
    r = {"*switch_mlp*=eql:6:16+imat", "*embed_tokens=affine:8:64:f16", "*lm_head=affine:8:64:f16", "*=eql:8:16+gptq"};
  } else if (name == "eq4dh") {
    r = {"*switch_mlp.down_proj=eql:5:16+imat", "*switch_mlp*=eql:4:16+imat", "*embed_tokens=affine:8:64:f16",
         "*lm_head=affine:8:64:f16", "*=eql:8:16+gptq"};
  } else if (name == "eq5dh") {
    r = {"*switch_mlp.down_proj=eql:6:16+imat", "*switch_mlp*=eql:5:16+imat", "*embed_tokens=affine:8:64:f16",
         "*lm_head=affine:8:64:f16", "*=eql:8:16+gptq"};
  } else if (name == "eq4f") {
    for (const char* g : {"*self_attn.[qkvo]_proj", "*linear_attn.out_proj"}) r.push_back(std::string(g) + "=eql:6:16+gptq");
    r.insert(r.end(), {"*switch_mlp*=eql:4:16+imat", "*embed_tokens=affine:6:64:f16+gptq", "*lm_head=affine:6:64:f16+gptq",
                       "*=eql:4:16+gptq"});
  } else if (name == "eq4s") {
    r = {"*switch_mlp*=eql:4:16+imat", "*embed_tokens=affine:6:64:f16+gptq", "*=eql:4:16+gptq"};
  } else if (name == "eq3h" || name == "eq3dh" || name == "eq3dx") {  // EQ3 (codebook) experts
    const std::string down = name == "eq3h" ? "eql:3:16:cb+imat" : "eql:4:16:cb+imat";
    const bool x = name == "eq3dx";
    r = {"*switch_mlp.down_proj=" + down, "*switch_mlp*=eql:3:16:cb+imat", "*embed_tokens=affine:8:64:f16",
         std::string("*lm_head=") + (x ? "affine:6:64:f16+gptq" : "affine:8:64:f16"),
         std::string("*=") + (x ? "eql:6:16+gptq" : "eql:8:16+gptq")};
  } else if (name == "eq6") {
    r = {"*switch_mlp*=eql:6:16+imat", "*embed_tokens=affine:8:64:f16", "*lm_head=affine:8:64:f16", "*=eql:6:16+gptq"};
  } else if (name == "fnx3" || name == "fnx3m") {
    // Qwen3.8-Flash-Next (qwen4_exp) at ~30 GB with ~240 kept experts per layer (route A): routed
    // experts (trunk and MTP) EQ3 codebooks, everything else EQ6 (fnx3m: EQ5), LM head MLX 6-bit,
    // embedding MLX 8-bit (gathered rows: 5/6-bit ones are widened to bytes in memory anyway); the
    // n-gram table stays outside the rules (--ple-bits).
    const std::string rest = name == "fnx3" ? "eql:6:16+gptq" : "eql:5:16+gptq";
    r = {"*switch_mlp*=eql:3:16:cb+imat", "*embed_tokens=affine:8:64:f16", "*lm_head=affine:6:64:f16+gptq", "*=" + rest};
  } else if (name.rfind("fnxvq", 0) == 0) {
    // Qwen3.8-Flash-Next at ~30 GB with all 512 experts (route B): routed experts (trunk and MTP) EQV
    // with 2^K-entry codebooks (fnxvq: K 13, ~1.69 bits per weight; fnxvq12 / fnxvq14 ...), the rest as fnx3m.
    const std::string k = name.size() > 5 ? name.substr(5) : "13";
    r = {"*switch_mlp*=eqv:" + k, "*embed_tokens=affine:8:64:f16", "*lm_head=affine:6:64:f16+gptq", "*=eql:5:16+gptq"};
  } else if (name == "fnxmix") {
    // Qwen3.8-Flash-Next at ~52 GB (64 GB Macs) with all 512 experts in two formats per layer
    // (--expert-split, the most routed first by --keep-experts): EQ3 codebooks, the rest EQV 13; as fnx3m otherwise.
    r = {"*switch_mlp_b*=eqv:13", "*switch_mlp*=eql:3:16:cb+imat", "*embed_tokens=affine:8:64:f16",
         "*lm_head=affine:6:64:f16+gptq", "*=eql:5:16+gptq"};
  } else {
    fail("unknown preset " + name);
  }
  std::vector<std::string> out;
  for (const char* g : kRouters) out.push_back(std::string(g) + "=bf16");
  out.insert(out.end(), r.begin(), r.end());
  return join(out, ";");
}

// Dense presets for Qwen3.5 / Qwen3.6 dense models (linear-attention layers, full attention every 4th).
// Widths by tensor class and layer position, set with KL probes on calibration text (Qwen3.5-9B, from a
// base of every matrix codebook EQ4 + GPTQ): the linear attention of the first quarter of the layers is
// the most sensitive part per byte, then the embedding, the attention k/v/o, and the MLPs of the first and
// last quarters; the middle MLPs least. Each preset is sized at or under a llama.cpp / unsloth GGUF of
// its quality class: eqd4m ~ Q4_K_M, eqd4l ~ UD-Q4_K_XL, eqd5 ~ UD-Q5_K_XL, eqd6 ~ Q6_K.
// eqd6-27b: Qwen3.8-27B, set with KL probes on held-out text (prose and code), sized under UD-Q6_K.
const std::vector<std::string> kDensePresets = {"eqd4m", "eqd4l", "eqd5", "eqd6", "eqd6-27b"};

bool is_dense_preset(const std::string& p) {
  return std::find(kDensePresets.begin(), kDensePresets.end(), p) != kDensePresets.end();
}

// L = decoder layers; "first" / "last" = the first / last L/4 layers.
std::string dense_preset(const std::string& name, int L) {
  if (L < 4) fail("preset " + name + ": needs the model's layer count (got " + std::to_string(L) + ")");
  const int q = L / 4;
  std::vector<std::string> r;
  auto band = [&](int lo, int hi, const std::string& g, const std::string& f) {
    for (int l = lo; l < hi; ++l) r.push_back("*layers." + std::to_string(l) + "." + g + "=" + f);
  };
  auto first = [&](const std::string& g, const std::string& f) { band(0, q, g, f); };
  auto edges = [&](const std::string& g, const std::string& f) {
    band(0, q, g, f);
    band(L - q, L, g, f);
  };
  const std::string c4 = "eql:4:16:cb+gptq", e5 = "eql:5:16+gptq", e6 = "eql:6:16+gptq", e8 = "eql:8:16+gptq";
  const std::string a5 = "affine:5:64:f16+gptq", a6 = "affine:6:64:f16+gptq";
  if (name == "eqd6-27b") {
    // every matrix EQ6; the MLPs of the first 5/16 of the layers (L0-19 of 64) EQ5; v_proj, lm_head and
    // in_proj_a/b EQ8; down_proj (17408 inputs, wider than --hessian-max) +imat; the vision tower bf16.
    // KLD 0.00127 at 6.417 bpw vs UD-Q6_K 0.00137 at 6.431 with a prose-heavy calibration text (wikitext-103,
    // C4, PG19 train + calibration_v5, chat and Python; 452k tokens); with calibration_v5 alone 0.00149.
    const int b = L * 5 / 16;
    r = {"*visual*=bf16", "*linear_attn.in_proj_[ab]=" + e8, "*self_attn.v_proj=" + e8, "*lm_head=" + e8};
    band(0, b, "mlp.down_proj", "eql:5:16+imat");
    band(0, b, "mlp.[gu]*", e5);
    r.insert(r.end(), {"*mlp.down_proj=eql:6:16+imat", "*embed_tokens=" + a5, "*=" + e6});
    return join(r, ";");
  }
  // linear attention of the first quarter: qkv/z EQ6 and out_proj EQ8 (eqd4*), all EQ8 (eqd5/eqd6);
  // in_proj_a/b EQ8 everywhere (tiny)
  const bool hi = name == "eqd5" || name == "eqd6";
  r.push_back("*linear_attn.in_proj_[ab]=" + e8);
  first("linear_attn.out_proj", e8);
  first("linear_attn.*", hi ? e8 : e6);
  if (name == "eqd4m") {
    r.push_back("*self_attn.[kvo]_proj=" + e6);
    edges("mlp.down_proj", e5);
    r.insert(r.end(), {"*embed_tokens=" + a5, "*lm_head=" + e5, "*=" + c4});
  } else if (name == "eqd4l") {
    r.push_back("*self_attn.[kvo]_proj=" + e6);
    edges("mlp.[gu]*", e5);
    r.insert(r.end(), {"*mlp.down_proj=" + e5, "*embed_tokens=" + a5, "*lm_head=" + e5, "*=" + c4});
  } else if (name == "eqd5") {
    r.push_back("*self_attn.[kvo]_proj=" + e6);
    edges("mlp.[gu]*", e5);
    edges("mlp.down_proj", e6);
    r.insert(r.end(), {"*mlp.*=" + e5, "*linear_attn.*=" + e5, "*embed_tokens=" + a6, "*lm_head=" + e6, "*=" + c4});
  } else {  // eqd6
    r.insert(r.end(), {"*linear_attn.out_proj=" + e8, "*linear_attn.in_proj_qkv=" + e5, "*embed_tokens=" + a6, "*=" + e6});
  }
  return join(r, ";");
}

int text_layers(const std::string& model_dir) {
  const json::Value cfg = json::parse(native::read_file(model_dir + "/config.json"));
  const json::Value* tcp = cfg.get("text_config");
  const json::Value& tc = tcp && tcp->is_object() ? *tcp : cfg;
  return int(tc.get_int("num_hidden_layers", 0));
}

bool known_preset(const std::string& p) {
  if (is_dense_preset(p)) return true;
  if (p.rfind("fnxvq", 0) == 0 && (p.size() == 5 || (std::isdigit(p[5]) && std::stoi(p.substr(5)) >= 9 && std::stoi(p.substr(5)) <= 16)))
    return true;
  if (p == "eq4dhb" || p == "eq3h" || p == "eq3dh" || p == "eq3dx" || p == "eq3dhb" || p == "fnx3" || p == "fnx3m" || p == "fnxmix" ||
      std::find(kPresets.begin(), kPresets.end(), p) != kPresets.end())
    return true;
  const std::string b = p.substr(0, p.size() - 1);
  return ends_with(p, "c") && b.find('4') != std::string::npos && std::find(kPresets.begin(), kPresets.end(), b) != kPresets.end();
}

struct Rule {
  std::string glob;
  Fmt f;
};

std::vector<Rule> parse_rules(const std::string& spec) {
  std::vector<Rule> rules;
  for (std::string r : split(spec, ';')) {
    while (!r.empty() && std::isspace((unsigned char)r.back())) r.pop_back();
    while (!r.empty() && std::isspace((unsigned char)r.front())) r.erase(r.begin());
    if (r.empty()) continue;
    const size_t eq = r.find('=');
    Rule x;
    x.glob = eq == std::string::npos ? "*" : r.substr(0, eq);
    x.f = parse_fmt(eq == std::string::npos ? r : r.substr(eq + 1));
    rules.push_back(std::move(x));
  }
  return rules;
}

// ---------------------------------------------------------------- the plan

struct Item {
  std::string path;            // module (MLX path, no .weight)
  Fmt f;
  std::vector<int64_t> shape;  // the weight's MLX shape
  // An expert stack split in two (--expert-split): the source module and the range [e0, e1) of its
  // experts (in the --keep-experts order) this item holds; e1 0: the whole module `path`.
  std::string src;
  int e0 = 0, e1 = 0;
  const std::string& source() const { return src.empty() ? path : src; }
};

int64_t numel(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

std::string st_dtype(const std::string& sd) { return sd == "bf16" ? "BF16" : sd == "f16" ? "F16" : "F32"; }

// The tensors module `it` becomes.
std::vector<StTensor> outputs(const Item& it) {
  std::vector<int64_t> lead(it.shape.begin(), it.shape.end() - 1);
  const int64_t C = it.shape.back();
  auto with = [&](int64_t last) {
    std::vector<int64_t> s = lead;
    s.push_back(last);
    return s;
  };
  std::vector<StTensor> v;
  if (it.f.kind == Fmt::VQ) {  // EQV planes (vq.h), routed expert stacks
    const int64_t nv = C / 8, hw = (nv * (it.f.bits - 8) + 31) / 32;
    v.push_back({it.path + ".vq_lo", "U8", with(nv)});
    v.push_back({it.path + ".vq_hi", "U32", with(hw)});
    v.push_back({it.path + ".vq_scale", "F16", with((C + 255) / 256)});
    v.push_back({it.path + ".vq_cb", "F16", {int64_t(1) << it.f.bits, 8}});
    return v;
  }
  if (it.f.kind == Fmt::EQ || it.f.kind == Fmt::EQC) {
    v.push_back({it.path + ".weight", "U32", with(C * it.f.bits / 32)});
    v.push_back({it.path + ".scales", "I8", with(C / 16)});
    v.push_back({it.path + ".super", "F16", with((C + 255) / 256)});  // (the last super-block partial when C % 256)
    if (it.f.kind == Fmt::EQC) {
      const int64_t R = numel(it.shape) / C;
      const int64_t rows = it.f.cbrows ? it.f.cbrows : (it.shape.size() == 3 ? it.shape[1] : R);
      v.push_back({it.path + ".codebook", "F16", {R / rows, int64_t(1) << it.f.bits}});
    }
  } else {
    v.push_back({it.path + ".weight", "U32", with(C * it.f.bits / 32)});
    v.push_back({it.path + ".scales", st_dtype(it.f.sd), with(C / it.f.group)});
    v.push_back({it.path + ".biases", st_dtype(it.f.sd), with(C / it.f.group)});
  }
  return v;
}

// fp32 copy of a dense source tensor.
std::vector<float> to_f32(const TensorView& t) {
  const size_t n = size_t(t.numel());
  std::vector<float> w(n);
  native::parallel_for((n + (1 << 20) - 1) >> 20, [&](size_t b) {
    const size_t i0 = b << 20, i1 = std::min(n, i0 + (size_t(1) << 20));
    for (size_t i = i0; i < i1; ++i) w[i] = native::dense_at(t.data, t.dtype, i);
  });
  return w;
}

template <class T>
std::vector<uint16_t> half_bits(const std::vector<T>& v, const std::string& sd) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = sd == "bf16" ? bf16_bits(float(v[i])) : f16_bits(float(v[i]));
  return o;
}

// Encode one matrix and write its tensors. Returns its quantization entry.
json::Value encode_write(StWriter& w, const Item& it, const float* W, const Stats* st) {
  const int64_t C = it.shape.back();
  const int64_t R = numel(it.shape) / C;
  const int nexp = it.shape.size() == 3 ? int(it.shape[0]) : 0;
  json::Value q = json::Value::object();
  if (it.f.kind == Fmt::VQ) {
    if (!nexp) fail(it.path + ": EQV (eqv:K) is for routed expert stacks only");
    const VqOut o = encode_vq(W, R, int(C), nexp, it.f.bits, it.f.enc == Fmt::GPTQ, st);
    w.write(it.path + ".vq_lo", o.lo.data(), o.lo.size());
    w.write(it.path + ".vq_hi", o.hi.data(), o.hi.size() * 4);
    w.write(it.path + ".vq_scale", o.sc.data(), o.sc.size() * 2);
    w.write(it.path + ".vq_cb", o.cb.data(), o.cb.size() * 2);
    q.set("bits", json::Value(it.f.bits));
    q.set("group_size", json::Value(8));
    q.set("mode", json::Value("eqv"));
    return q;
  }
  if (it.f.kind == Fmt::EQ || it.f.kind == Fmt::EQC) {
    const EqOut o = encode_eq(W, R, int(C), nexp, it.f, st);
    const std::vector<uint32_t> packed = pack_codes(o.q.data(), R, int(C), it.f.bits);
    w.write(it.path + ".weight", packed.data(), packed.size() * 4);
    w.write(it.path + ".scales", o.sc.data(), o.sc.size());
    w.write(it.path + ".super", o.d.data(), o.d.size() * 2);
    if (it.f.kind == Fmt::EQC) w.write(it.path + ".codebook", o.cb.data(), o.cb.size() * 2);
    q.set("bits", json::Value(it.f.bits));
    q.set("group_size", json::Value(16));
    q.set("mode", json::Value(it.f.kind == Fmt::EQC ? "eqc" : "eq"));
  } else {
    const AffOut o = encode_affine(W, R, int(C), nexp, it.f, st);
    const std::vector<uint32_t> packed = pack_codes(o.q.data(), R, int(C), it.f.bits);
    w.write(it.path + ".weight", packed.data(), packed.size() * 4);
    if (it.f.sd == "f32") {
      w.write(it.path + ".scales", o.s.data(), o.s.size() * 4);
      w.write(it.path + ".biases", o.b.data(), o.b.size() * 4);
    } else {
      const auto s = half_bits(o.s, it.f.sd), b = half_bits(o.b, it.f.sd);
      w.write(it.path + ".scales", s.data(), s.size() * 2);
      w.write(it.path + ".biases", b.data(), b.size() * 2);
    }
    q.set("bits", json::Value(it.f.bits));
    q.set("group_size", json::Value(it.f.group));
  }
  return q;
}

// ---------------------------------------------------------------- sources

// What the converter reads: tensors by MLX name, their shapes, fp32 copies.
// HfSrc: a Hugging Face BF16 checkpoint (HFSource). MlxSrc: an MLX checkpoint
// whose affine-quantized matrices are dequantized to fp32 (w = q s + b with
// its own scales; Qwen4-Exp oQ4e-mtp files, while no BF16 copy is at hand).
struct QSource {
  virtual ~QSource() = default;
  virtual const json::Value& config() const = 0;
  virtual std::vector<std::string> names() const = 0;
  virtual std::pair<std::string, std::vector<int64_t>> info(const std::string& name) const = 0;
  virtual std::vector<float> f32(const std::string& name) const = 0;
  // A tensor's bytes as stored (dense tensors; I64 constants).
  virtual TensorView view(const std::string& name) const = 0;
  virtual bool quantized(const std::string& name) const { return false; }
  virtual void release() const {}
  virtual const native::SafetensorsSource& raw() const = 0;
};

// A tensor's file bytes into the page cache front to back, one 64 MB read at a
// time, before the conversion's parallel copies fault the mapping in: on a
// spinning disk (the vault) those page-sized faults from many threads read
// ~70 MB/s, a sequential read ~250 MB/s. Nothing is read when it is resident.
void prefetch_sequential(const uint8_t* p, size_t n) {
  int fd = -1;
  size_t base = 0;
  if (n < (size_t(64) << 20) || !native::mapped_at(p, n, fd, base)) return;
  const size_t pg = size_t(::getpagesize());
  const uintptr_t a = reinterpret_cast<uintptr_t>(p) / pg * pg;
  const size_t len = reinterpret_cast<uintptr_t>(p) + n - a;
  std::vector<char> in((len + pg - 1) / pg);
  if (::mincore(reinterpret_cast<void*>(a), len, in.data()) == 0) {
    size_t resident = 0;
    for (char c : in) resident += (c & 1);
    if (resident * 10 >= in.size() * 9) return;
  }
  std::vector<uint8_t> buf(size_t(64) << 20);
  for (size_t o = 0; o < n; o += buf.size()) native::pread_all(fd, buf.data(), std::min(buf.size(), n - o), base + o);
}

struct HfSrc : QSource {
  explicit HfSrc(const std::string& dir) : s(dir) {}
  HFSource s;
  const json::Value& config() const override { return s.config(); }
  std::vector<std::string> names() const override { return s.names(); }
  std::pair<std::string, std::vector<int64_t>> info(const std::string& n) const override { return s.info(n); }
  std::vector<float> f32(const std::string& n) const override {
    for (const auto& [p, len] : s.raw_ranges(n)) prefetch_sequential(p, len);
    return to_f32(s.tensor(n));
  }
  TensorView view(const std::string& n) const override {
    const auto [dtype, shape] = s.info(n);
    if (dtype == "BF16" || dtype == "F16" || dtype == "F32") return s.tensor(n);
    // other dtypes (the n-gram hashing constants, I64): their bytes as they are
    const auto l = s.raw().locate(n.rfind("language_model.model.", 0) == 0 ? "model.language_model." + n.substr(21) : n);
    TensorView t;
    t.name = n;
    t.dtype = l.dtype;
    t.shape = l.shape;
    t.data = l.data;
    t.nbytes = l.nbytes;
    return t;
  }
  void release() const override { s.release(); }
  const native::SafetensorsSource& raw() const override { return s.raw(); }
};

struct MlxSrc : QSource {
  explicit MlxSrc(const std::string& dir) : s(dir) {
    for (const std::string& n : s.names()) {
      if (n.rfind("vision_tower", 0) == 0) continue;
      const auto l = s.locate(n);
      if (ends_with(n, ".weight") && l.dtype == "U32" && s.has(n.substr(0, n.size() - 7) + ".scales")) quant_.insert(n);
    }
    for (const std::string& n : s.names()) {
      if (n.rfind("vision_tower", 0) == 0) continue;
      bool companion = false;
      for (const char* x : {".scales", ".biases", ".super", ".codebook"})
        if (ends_with(n, x) && quant_.count(n.substr(0, n.size() - std::strlen(x)) + ".weight")) companion = true;
      if (!companion) names_.push_back(n);
    }
    std::sort(names_.begin(), names_.end());
  }
  native::SafetensorsSource s;
  std::set<std::string> quant_;
  std::vector<std::string> names_;
  const json::Value& config() const override { return s.config(); }
  std::vector<std::string> names() const override { return names_; }
  bool quantized(const std::string& n) const override { return quant_.count(n) != 0; }
  std::pair<std::string, std::vector<int64_t>> info(const std::string& n) const override {
    const auto l = s.locate(n);
    if (!quant_.count(n)) return {l.dtype, l.shape};
    const auto [bits, group] = s.quant_of(n.substr(0, n.size() - 7));
    std::vector<int64_t> sh = l.shape;  // [.., in * bits / 32] codes
    sh.back() = sh.back() * 32 / bits;
    return {"BF16", sh};  // (as dequantized: written as bf16 when no rule re-encodes it)
  }
  std::vector<float> f32(const std::string& n) const override {
    const TensorView t = s.tensor(n);
    if (t.kind == TensorView::Dense) return to_f32(t);
    if (t.kind != TensorView::MLXQuant) fail(n + ": only MLX affine matrices are dequantized");
    const int64_t R = t.rows(), C = t.cols();
    const int bits = t.bits, g = t.group;
    const size_t rb = size_t(C) * size_t(bits) / 8;
    std::vector<float> w(size_t(R) * size_t(C));
    native::parallel_for(size_t((R + 255) / 256), [&](size_t b) {
      const int64_t r0 = int64_t(b) * 256, r1 = std::min<int64_t>(R, r0 + 256);
      for (int64_t r = r0; r < r1; ++r) {
        const uint8_t* row = t.data + size_t(r) * rb;
        for (int64_t c = 0; c < C; ++c) {
          const size_t bit = size_t(c) * size_t(bits);
          uint32_t v = 0;
          for (int k = 0; k < bits; ++k) v |= uint32_t((row[(bit + k) / 8] >> ((bit + k) % 8)) & 1u) << k;
          const size_t gi = size_t(r) * size_t(C / g) + size_t(c / g);
          w[size_t(r) * size_t(C) + size_t(c)] = float(v) * native::dense_at(t.scales, t.dtype, gi) + native::dense_at(t.biases, t.dtype, gi);
        }
      }
    });
    return w;
  }
  TensorView view(const std::string& n) const override {
    if (quant_.count(n)) fail(n + ": a quantized tensor has no dense bytes");
    const auto l = s.locate(n);
    TensorView t;
    t.name = n;
    t.dtype = l.dtype;
    t.shape = l.shape;
    t.data = l.data;
    t.nbytes = l.nbytes;
    return t;
  }
  const native::SafetensorsSource& raw() const override { return s; }
};

// ---------------------------------------------------------------- pruned experts

// --keep-experts FILE: {"<layer>": [expert ids], ..., "mtp": [...]}; list order
// = new expert id. Layers may keep different counts (multiples of 8): the
// config's num_experts becomes the largest, each layer's stacks and router
// say its own (text_config.num_experts_per_layer lists them). Without an
// "mtp" list the MTP layer keeps all its experts.
struct Keep {
  std::map<std::string, std::vector<int>> lists;  // "0".."L-1", "mtp"
  int E = 0, K = 0;  // K: the largest count
  // The list of the expert stack / router `name` belongs to (null: not pruned).
  const std::vector<int>* of(const std::string& name) const {
    if (lists.empty()) return nullptr;
    static const std::regex re(R"(^(language_model\.model\.|mtp\.)layers\.(\d+)\.mlp\.(switch_mlp\.(gate|up|down)_proj|gate)\.weight$)");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return nullptr;
    const std::string key = m[1] == "mtp." ? "mtp" : m[2].str();
    auto it = lists.find(key);
    if (it == lists.end() && key == "mtp") return nullptr;  // (no list: the MTP layer keeps its experts)
    if (it == lists.end()) fail("--keep-experts: no list for layer " + key);
    return &it->second;
  }
};

Keep read_keep(const std::string& path, int E, int L, bool mtp) {
  Keep k;
  k.E = E;
  const json::Value v = json::parse(native::read_file(path));
  const json::Object& o = v.as_object();
  for (size_t i = 0; i < o.size(); ++i) {
    std::vector<int> ids;
    std::set<int> seen;
    for (const json::Value& x : o.vals[i].as_array()) {
      const int e = int(x.as_int());
      if (e < 0 || e >= E || !seen.insert(e).second) fail(path + ": layer " + o.keys[i] + ": expert " + std::to_string(e) + " out of range or repeated");
      ids.push_back(e);
    }
    if (ids.empty() || ids.size() % 8) fail(path + ": layer " + o.keys[i] + " keeps " + std::to_string(ids.size()) + " experts (a positive multiple of 8)");
    k.K = std::max(k.K, int(ids.size()));
    k.lists[o.keys[i]] = std::move(ids);
  }
  for (int l = 0; l < L; ++l)
    if (!k.lists.count(std::to_string(l))) fail(path + ": no list for layer " + std::to_string(l));
  if (mtp && !k.lists.count("mtp")) k.K = E;  // the MTP layer unpruned: the config's count stays the largest
  return k;
}

// --expert-split H | FILE: a layer's routed experts in two formats, the first H (in the --keep-experts
// order, e.g. most routed first) in mlp.switch_mlp.* by the rules for that path, the others in
// mlp.switch_mlp_b.* by the rules for theirs (e.g. '*switch_mlp_b*=eqv:13;*switch_mlp*=eql:3:16:cb+imat').
// H: every layer; FILE: JSON {"<layer>": H, ..., "mtp": H}, a layer not listed keeps one stack. H and the
// rest multiples of 8.
struct Split {
  int all = 0;
  std::map<std::string, int> per;
  // H of the expert stack `name` (0: one stack).
  int of(const std::string& name) const {
    static const std::regex re(R"(^(language_model\.model\.|mtp\.)layers\.(\d+)\.mlp\.switch_mlp\.(gate|up|down)_proj\.weight$)");
    std::smatch m;
    if ((!all && per.empty()) || !std::regex_match(name, m, re)) return 0;
    if (all) return all;
    auto it = per.find(m[1] == "mtp." ? "mtp" : m[2].str());
    return it == per.end() ? 0 : it->second;
  }
};

Split read_split(const std::string& v) {
  Split s;
  if (!v.empty() && std::all_of(v.begin(), v.end(), [](char c) { return std::isdigit((unsigned char)c); })) {
    s.all = std::stoi(v);
    return s;
  }
  const json::Value j = json::parse(native::read_file(v));
  const json::Object& o = j.as_object();
  for (size_t i = 0; i < o.size(); ++i) s.per[o.keys[i]] = int(o.vals[i].as_int());
  return s;
}

// Rows of `keep`'s experts from W [E][rows_per][C] (or a router [E][C]).
std::vector<float> slice_experts(const std::vector<float>& W, int E, const std::vector<int>& keep) {
  const size_t per = W.size() / size_t(E);
  std::vector<float> o(per * keep.size());
  native::parallel_for(keep.size(), [&](size_t i) { std::memcpy(&o[i * per], &W[size_t(keep[i]) * per], per * 4); });
  return o;
}

// Per-expert statistics of a pruned stack: the kept experts' rows.
const Stats* slice_stats(const Stats* st, int E, const std::vector<int>* keep, Stats& tmp, std::vector<float>& d,
                         std::vector<float>& x) {
  if (!st || !keep) return st;
  tmp = *st;
  const size_t C = size_t(st->C);
  if (st->diag && st->diag_rows == E) {
    d.resize(keep->size() * C);
    for (size_t i = 0; i < keep->size(); ++i) std::memcpy(&d[i * C], st->diag + size_t((*keep)[i]) * C, C * 4);
    tmp.diag = d.data();
    tmp.diag_rows = int(keep->size());
  }
  if (st->Hexp && st->nexp == E) {
    x.resize(keep->size() * C * C);
    for (size_t i = 0; i < keep->size(); ++i) std::memcpy(&x[i * C * C], st->Hexp + size_t((*keep)[i]) * C * C, C * C * 4);
    tmp.Hexp = x.data();
    tmp.nexp = int(keep->size());
  }
  return &tmp;
}

// ---------------------------------------------------------------- MTP head

// eq_export.mtp_tensors: the HF mtp.* tensors as Espetero reads them from an
// MLX directory (language_model.mtp.*, fused experts split into switch_mlp,
// RMSNorm weights 1 + w in fp32, matrices MLX affine).
struct MtpPart {
  std::vector<std::string> src;  // source tensors (several: per-expert stack)
  enum Kind { Matrix, Norm, Raw, Gate, Up } kind = Raw;
  Item item;
  StTensor raw;
};

std::vector<MtpPart> mtp_plan(const native::SafetensorsSource& raw, const Fmt& fm, const Fmt& fe) {
  std::vector<std::string> names;
  for (const std::string& n : raw.names())
    if (n.rfind("mtp.", 0) == 0) names.push_back(n);
  std::sort(names.begin(), names.end());
  static const std::regex fused(R"((.*\.mlp)\.experts\.(gate_up_proj|down_proj)$)");
  static const std::regex single(R"((.*\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$)");
  static const std::regex norm(R"(norm[a-z_]*\.weight$)");
  std::vector<MtpPart> parts;
  std::map<std::pair<std::string, std::string>, std::map<int, std::string>> stacks;
  for (const std::string& k : names) {
    const auto l = raw.locate(k);
    const std::string n = "language_model." + k;
    std::smatch m;
    if (std::regex_match(n, m, fused)) {
      const std::string pre = m[1].str() + ".switch_mlp.";
      if (m[2] == "gate_up_proj") {
        std::vector<int64_t> sh = l.shape;
        sh[sh.size() - 2] /= 2;
        parts.push_back({{k}, MtpPart::Gate, {pre + "gate_proj", fe, sh}, {}});
        parts.push_back({{k}, MtpPart::Up, {pre + "up_proj", fe, sh}, {}});
      } else {
        parts.push_back({{k}, MtpPart::Matrix, {pre + "down_proj", fe, l.shape}, {}});
      }
    } else if (std::regex_match(n, m, single)) {
      stacks[{m[1].str(), m[3].str()}][std::stoi(m[2].str())] = k;
    } else if (l.shape.size() == 1 && std::regex_search(n, norm) && n.find(".linear_attn.") == std::string::npos) {
      parts.push_back({{k}, MtpPart::Norm, {}, {n, "F32", l.shape}});
    } else if (l.shape.size() >= 2 && l.shape.back() % fm.group == 0) {
      parts.push_back({{k}, MtpPart::Matrix, {n.substr(0, n.size() - 7), fm, l.shape}, {}});
    } else {
      parts.push_back({{k}, MtpPart::Raw, {}, {n, "BF16", l.shape}});
    }
  }
  for (const auto& [key, d] : stacks) {
    MtpPart p;
    p.kind = MtpPart::Matrix;
    for (int i = 0; i < int(d.size()); ++i) {
      auto it = d.find(i);
      if (it == d.end()) fail("MTP experts of " + key.first + " are not numbered 0..n-1");
      p.src.push_back(it->second);
    }
    std::vector<int64_t> sh = raw.locate(p.src[0]).shape;
    sh.insert(sh.begin(), int64_t(d.size()));
    p.item = {key.first + ".switch_mlp." + key.second, fe, sh};
    parts.push_back(std::move(p));
  }
  return parts;
}

std::vector<float> mtp_weights(const native::SafetensorsSource& raw, const MtpPart& p) {
  std::vector<float> w;
  for (const std::string& s : p.src) {
    const auto l = raw.locate(s);
    TensorView t;
    t.data = l.data;
    t.dtype = l.dtype;
    t.shape = l.shape;
    std::vector<float> one = to_f32(t);
    if (p.kind == MtpPart::Gate || p.kind == MtpPart::Up) {
      const int64_t E = l.shape[0], R2 = l.shape[1], C = l.shape[2];
      const size_t half = size_t(R2 / 2) * size_t(C);
      for (int64_t e = 0; e < E; ++e) {
        const float* b = one.data() + size_t(e) * 2 * half + (p.kind == MtpPart::Up ? half : 0);
        w.insert(w.end(), b, b + half);
      }
    } else {
      w.insert(w.end(), one.begin(), one.end());
    }
  }
  return w;
}

// ---------------------------------------------------------------- files

void copy_file(const std::string& a, const std::string& b) {
  std::ifstream in(a, std::ios::binary);
  std::ofstream out(b, std::ios::binary);
  out << in.rdbuf();
  if (!out) fail("cannot write " + b);
}

void write_text(const std::string& path, const std::string& s) {
  std::ofstream f(path, std::ios::binary);
  f << s;
  if (!f) fail("cannot write " + path);
}

void mkdirs(const std::string& d) {
  std::string cur;
  for (const std::string& part : split(d, '/')) {
    cur += part + "/";
    if (!part.empty()) ::mkdir(cur.c_str(), 0755);
  }
  if (!native::is_dir(d)) fail("cannot create " + d);
}

const char* kUsage =
    "usage: espetero quantize --model HF_DIR --out DIR [--preset NAME | --fmt RULES]\n"
    "         [--calib FILE | --calib none] [--calib-data file:PATH[+...]] [--calib-tokens N] [--ctx N]\n"
    "         [--calib-out FILE] [--calib-only] [--calib-experts f16|q8] [--calib-expert-weight count|router] [--calib-layers A:B]\n"
    "         [--hessian-max N] [--hessian-pack N] [--no-mtp] [--shard-gb G] [--tiles]\n"
    "         [--calib-model DIR] [--keep-experts FILE] [--expert-split H|FILE] [--ple-bits 4|8]\n"
    "       espetero quantize --preset NAME --print-rules [--model HF_DIR | --layers N]\n"
    "\n"
    "A Hugging Face BF16 checkpoint (Qwen3.5 / Qwen3.6, dense Gemma 4, Qwen4-Exp) to an EQ checkpoint;\n"
    "Qwen4-Exp (Qwen3.8-Flash-Next) also from an MLX checkpoint (its affine matrices dequantized).\n"
    "  --preset   eq4 eq5 eq6 eq4x eq5x eq4d eq5d eq4f eq4s eq6h eq5dh eq4dh, a 4-bit one + c\n"
    "             (EQ4 experts with codebooks: eq4dhc ...), eq4dhb (default eq4);\n"
    "             EQ3 codebook experts (MoE only): eq3h eq3dh eq3dx eq3dhb;\n"
    "             Qwen3.8-Flash-Next: fnx3 (experts EQ3 codebooks, the rest EQ6), fnx3m (the rest EQ5);\n"
    "             fnxvq[K] (all 512 experts EQV with 2^K codebooks, K 13 by default; the rest as fnx3m);\n"
    "             fnxmix (with --expert-split: the first experts EQ3 codebooks, the rest EQV 13; the rest as fnx3m);\n"
    "             dense Qwen3.5 / 3.6 (widths by tensor class and layer position, sized at or\n"
    "             under the GGUF named): eqd4m (Q4_K_M) eqd4l (UD-Q4_K_XL) eqd5 (UD-Q5_K_XL) eqd6 (Q6_K);\n"
    "             Qwen3.8-27B: eqd6-27b (UD-Q6_K; calibrate with prose as well as code)\n"
    "  --fmt      rules GLOB=FMT;... instead (eql:B:16[+enc], eql:4:16:cb[+enc], eql:3:16:cb[+enc],\n"
    "             affine:B:G[:SD][+enc], bf16; enc rtn | mse | imat | gptq)\n"
    "  --calib    statistics from an earlier run (or tools/dev/calib_npz.py); without it they are\n"
    "             captured first with Espetero's forward of the BF16 weights over --calib-data\n"
    "             (default file:~/.cache/mlx-lm/calibration_v5.txt), windows of --ctx (2048)\n"
    "             tokens up to --calib-tokens (262144), saved to --calib-out (default\n"
    "             OUT.calib.safetensors); --calib none: unweighted encoders; --calib-experts q8\n"
    "             runs the forward's experts as 8-bit copies (less memory; default f16, exact);\n"
    "             --calib-expert-weight router weights each routed pair of an expert's statistics by\n"
    "             its routing weight (default count: every pair once)\n"
    "             several comma-separated files read as one (the parts of --calib-layers captures)\n"
    "  --hessian-max   capture the full E[x x^T] of inputs up to N wide (16384), else only its\n"
    "             diagonal (+gptq then falls back to +imat); --hessian-pack N saves full ones wider\n"
    "             than N (16384) as their upper triangle (half the bytes)\n"
    "  --tiles    also write the checkpoint's tile image (faster loads)\n"
    "  --only GLOB[,GLOB]  write only the tensors whose names match a GLOB and no !GLOB (a partial\n"
    "             checkpoint, e.g. a few layers or everything but the n-gram table, to overlay on another\n"
    "             file of the model: tools/dev/fnx_overlay.py)\n"
    "  --expert-hessian-max N  capture per-expert E[x x^T] of expert inputs up to N wide (1024;\n"
    "             Qwen4-Exp 0: 40 GB, and no encoder reads them yet)\n"
    "  --calib-model DIR  capture the statistics with the forward of another checkpoint of the model\n"
    "             (Qwen4-Exp: the MLX 4-bit file, as the BF16 one does not fit in memory)\n"
    "  --keep-experts FILE  routed experts pruned to a kept list per layer: JSON {\"0\": [ids], ...,\n"
    "             \"mtp\": [ids] (optional)}, counts multiples of 8, possibly different per layer (list order = new\n"
    "             expert id); expert stacks, routers and statistics are sliced, num_experts set to the\n"
    "             largest count\n"
    "  --expert-split H|FILE  each layer's routed experts in two formats: the first H (in the --keep-experts\n"
    "             order) in mlp.switch_mlp.*, the rest in mlp.switch_mlp_b.*, each by the rules for its path\n"
    "             (the second EQV, e.g. preset fnxmix); FILE: JSON {\"<layer>\": H, ...}, unlisted layers whole\n"
    "  --ple-bits Qwen4-Exp n-gram table (read from disk at run time, outside the rules): MLX affine\n"
    "             4-bit (default, the oQ4e layout, 100 B per row) or 8-bit, group 32, bf16 scales\n";

struct Args {
  std::map<std::string, std::string> kv;
  std::set<std::string> flags;
  std::string get(const std::string& k, const std::string& d = "") const {
    auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
  bool has(const std::string& k) const { return flags.count(k) || kv.count(k); }
};

}  // namespace

int cmd_quantize(int argc, char** argv) {
  Args a;
  static const std::set<std::string> bare = {"no-mtp", "tiles", "calib-only", "help", "print-rules"};
  for (int i = 0; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) != 0) fail("quantize: unexpected argument " + s + "\n" + kUsage);
    s = s.substr(2);
    if (const size_t e = s.find('='); e != std::string::npos) {
      a.kv[s.substr(0, e)] = s.substr(e + 1);
    } else if (bare.count(s)) {
      a.flags.insert(s);
    } else {
      if (i + 1 >= argc) fail("quantize: --" + s + " needs a value");
      a.kv[s] = argv[++i];
    }
  }
  if (a.has("print-rules")) {  // a preset's rules (or --fmt's), one per line
    const std::string p = a.get("preset", "eq4");
    if (!a.has("fmt") && !known_preset(p)) fail("unknown preset " + p);
    if (!a.has("fmt") && is_dense_preset(p)) {  // the layer count from --model, else --layers (default 32)
      std::puts(dense_preset(p, a.has("model") ? text_layers(a.get("model")) : std::stoi(a.get("layers", "32"))).c_str());
      return 0;
    }
    std::puts((a.has("fmt") ? a.get("fmt") : preset(p)).c_str());
    return 0;
  }
  if (a.has("help") || !a.has("model") || (!a.has("out") && !a.has("calib-only"))) {
    std::fputs(kUsage, stderr);
    return a.has("help") ? 0 : 2;
  }
  const auto t0 = native::Clock::now();
  std::string model = a.get("model");
  while (model.size() > 1 && model.back() == '/') model.pop_back();
  std::string out = a.get("out");
  while (out.size() > 1 && out.back() == '/') out.pop_back();
  const std::string pname = a.get("preset", "eq4");
  if (!a.has("fmt") && !known_preset(pname)) fail("unknown preset " + pname);
  const std::string rules_spec = a.has("fmt")              ? a.get("fmt")
                                 : is_dense_preset(pname) ? dense_preset(pname, text_layers(model))
                                                          : preset(pname);
  const std::vector<Rule> rules = parse_rules(rules_spec);

  // Calibration statistics: given, none, or captured now.
  std::string calib = a.get("calib");
  if (calib.empty()) {
    calib = a.get("calib-out", (out.empty() ? model : out) + ".calib.safetensors");
    CaptureOptions co;
    co.model = a.get("calib-model", model);  // (another checkpoint of the model: its forward, its taps)
    while (co.model.size() > 1 && co.model.back() == '/') co.model.pop_back();
    co.data = a.get("calib-data", "file:~/.cache/mlx-lm/calibration_v5.txt");
    co.out = calib;
    co.ctx = std::stoi(a.get("ctx", "2048"));
    co.max_tokens = std::stoll(a.get("calib-tokens", "262144"));
    if (a.has("calib-layers")) {  // A:B, statistics for layers [A, B) only (lower memory; merge the parts)
      const std::string r = a.get("calib-layers", "");
      const size_t c = r.find(':');
      if (c == std::string::npos) fail("--calib-layers: A:B");
      co.layer_lo = std::stoi(r.substr(0, c));
      co.layer_hi = std::stoi(r.substr(c + 1));
    }
    const std::string ex = a.get("calib-experts", "f16");
    if (ex != "f16" && ex != "q8") fail("--calib-experts: f16 or q8");
    co.experts_f16 = ex == "f16";
    const std::string ew = a.get("calib-expert-weight", "count");
    if (ew != "count" && ew != "router") fail("--calib-expert-weight: count or router");
    co.expert_route_weight = ew == "router";
    co.hessian_max = std::stoi(a.get("hessian-max", std::to_string(co.hessian_max)));
    {  // per-expert E[x x^T] (unused by the encoders so far): Qwen4-Exp's 512 x 640^2 per layer (40 GB) only when asked
      const json::Value mc = json::parse(native::read_file(co.model + "/config.json"));
      const json::Value* t = mc.get("text_config");
      const bool fnx = mc.get_string("model_type", "").rfind("qwen4_exp", 0) == 0 ||
                       (t && t->is_object() && t->get_string("model_type", "").rfind("qwen4_exp", 0) == 0);
      co.expert_hessian_max = std::stoi(a.get("expert-hessian-max", fnx ? "0" : std::to_string(co.expert_hessian_max)));
    }
    co.hessian_pack = std::stoi(a.get("hessian-pack", std::to_string(co.hessian_pack)));
    capture(co);
  }
  if (a.has("calib-only")) return 0;
  std::unique_ptr<CalibFile> stats;
  if (calib != "none") stats = std::make_unique<CalibFile>(calib);

  std::unique_ptr<QSource> srcp;
  {
    bool hf = false;
    {
      native::SafetensorsSource probe(model);
      hf = HFSource::is_hf_dir(probe);
    }
    if (hf) srcp = std::make_unique<HfSrc>(model);
    else srcp = std::make_unique<MlxSrc>(model);
  }
  const QSource& src = *srcp;
  const json::Value& cfg = src.config();
  const json::Value* tcp = cfg.get("text_config");
  const json::Value& tc = tcp && tcp->is_object() ? *tcp : cfg;
  const bool moe = tc.get_int("num_experts", 0) > 0;
  const bool q4 = cfg.get_string("model_type", "").rfind("qwen4_exp", 0) == 0 || tc.get_string("model_type", "").rfind("qwen4_exp", 0) == 0;
  // A quantized MLX source (its matrices dequantized) only for Qwen4-Exp; an unquantized checkpoint in its own
  // names (Spark-X2.5 / X2.6: model.*) is a BF16 source like a Hugging Face directory.
  if (const auto* ms = dynamic_cast<const MlxSrc*>(&src); !q4 && ms && !ms->quant_.empty())
    fail(model + ": an MLX source is supported for Qwen4-Exp only (give the BF16 checkpoint)");
  const bool no_mtp = a.has("no-mtp");
  const int nexp = int(tc.get_int("num_experts", 0));
  Keep keep;
  if (a.has("keep-experts")) {
    if (!moe) fail("--keep-experts: not a mixture-of-experts model");
    keep = read_keep(a.get("keep-experts"), nexp, int(tc.get_int("num_hidden_layers", 0)), q4 && !no_mtp);
  }
  const Split xsplit = a.has("expert-split") ? read_split(a.get("expert-split")) : Split{};
  if (a.has("expert-split") && !moe) fail("--expert-split: not a mixture-of-experts model");
  const int ple_bits = std::stoi(a.get("ple-bits", "4"));
  if (ple_bits != 4 && ple_bits != 8) fail("--ple-bits: 4 or 8");
  // Qwen4-Exp's n-gram table (shards of [rows, 160]): MLX affine, outside the rules.
  static const std::regex ple_re(R"(^(.*\.ngram_embedding\.)shards?[._](\d+)\.weight$)");
  auto is_ple = [&](const std::string& n) { return q4 && n.find(".ngram_embedding.") != std::string::npos; };
  std::vector<std::string> only;
  if (a.has("only")) only = split(a.get("only"), ',');
  auto wanted = [&](const std::string& n) {  // a match of a GLOB and of no !GLOB
    bool in = false, pos = false;
    for (const std::string& g : only) {
      const bool neg = !g.empty() && g[0] == '!';
      pos = pos || !neg;
      if (fnmatch(n.c_str(), g.c_str() + (neg ? 1 : 0))) {
        if (neg) return false;
        in = true;
      }
    }
    return in || !pos;
  };
  auto skipped = [&](const std::string& n) { return (q4 && no_mtp && n.rfind("mtp.", 0) == 0) || !wanted(n); };

  // The plan: an explicit rule first, then the
  // model's predicate (MoE routers 8-bit), then the first matching rule;
  // bf16 and widths the format cannot divide stay as they are.
  std::vector<Item> plan;
  std::set<std::string> replaced;
  int64_t params = 0;
  for (const std::string& name : src.names()) {
    if (skipped(name)) continue;
    auto [dtype, shape] = src.info(name);
    if (const auto* kl = keep.of(name)) shape[0] = int64_t(kl->size());
    params += numel(shape);
    if (is_ple(name)) continue;
    if (!ends_with(name, ".weight") || shape.size() < 2 || name.find("conv1d") != std::string::npos) continue;
    const std::string p = name.substr(0, name.size() - 7);
    // The format of module `q`: an explicit rule first, then the predicate, then the first match.
    auto fmt_of = [&](const std::string& q) -> const Fmt* {
      const Fmt* f = nullptr;
      static Fmt pred = parse_fmt("affine:8:64");
      for (const Rule& r : rules)
        if (r.glob != "*" && fnmatch(q.c_str(), r.glob.c_str())) return &r.f;
      if (moe && (ends_with(q, "mlp.gate") || ends_with(q, "shared_expert_gate"))) return &pred;
      for (const Rule& r : rules)
        if (fnmatch(q.c_str(), r.glob.c_str())) {
          f = &r.f;
          break;
        }
      return f;
    };
    // EQ: any width that is a multiple of 64 (a partial last super-block below 256)
    auto encodable = [&](const Fmt* f) {
      return f && f->kind != Fmt::BF16 &&
             shape.back() % (f->kind == Fmt::EQ || f->kind == Fmt::EQC ? 64 : f->kind == Fmt::VQ ? 8 : f->group) == 0;
    };
    auto add = [&](Item it) {
      if (it.f.kind == Fmt::VQ && it.shape.size() != 3) fail(it.path + ": EQV (" + it.f.spec + ") is for routed expert stacks only");
      if (it.f.kind == Fmt::EQ && !(it.f.bits == 3 || it.f.bits == 4 || it.f.bits == 5 || it.f.bits == 6 || it.f.bits == 8))
        it.f = parse_fmt("affine:8:64:f16");
      plan.push_back(std::move(it));
    };
    if (const int H = xsplit.of(name)) {  // two stacks: experts [0, H) here, [H, E) in switch_mlp_b
      const int Ex = int(shape[0]);
      if (H % 8 || (Ex - H) % 8 || H <= 0 || H >= Ex)
        fail(name + ": --expert-split " + std::to_string(H) + " of " + std::to_string(Ex) + " experts (multiples of 8, both stacks non-empty)");
      const std::string pb = replace_all(p, ".switch_mlp.", ".switch_mlp_b.");
      const Fmt *fa = fmt_of(p), *fb = fmt_of(pb);
      if (!encodable(fa) || !encodable(fb)) fail(name + ": --expert-split needs both stacks quantized (" + p + ", " + pb + ")");
      std::vector<int64_t> sa = shape, sb = shape;
      sa[0] = H;
      sb[0] = Ex - H;
      add(Item{p, *fa, sa, p, 0, H});
      add(Item{pb, *fb, sb, p, H, Ex});
      replaced.insert(name);
      continue;
    }
    const Fmt* f = fmt_of(p);
    if (!encodable(f)) continue;
    add(Item{p, *f, shape});
    replaced.insert(name);
  }

  // MTP head: MLX affine 8-bit, experts 4-bit +mse.
  // (Qwen4-Exp: the mtp.* tensors are in the plan above, by the same rules)
  std::vector<MtpPart> mtp;
  if (!no_mtp && !q4) mtp = mtp_plan(src.raw(), parse_fmt("affine:8:64:f16"), parse_fmt("affine:4:64+mse"));

  // Qwen4-Exp n-gram table: shards as MLX affine ple_bits / 32, renamed to the
  // oQ4e layout (...ngram_embedding.shards.<k>.{weight,scales,biases}); an MLX
  // source of the same width is copied as it is (with its weight_scale).
  struct PleShard {
    std::string src, out;  // source .weight, output base
    int64_t rows = 0, width = 0;
    bool copy = false;
  };
  std::vector<PleShard> ple;
  for (const std::string& name : src.names()) {
    std::smatch m;
    if (!is_ple(name) || !std::regex_match(name, m, ple_re) || !wanted(name)) continue;
    PleShard p;
    p.src = name;
    p.out = m[1].str() + "shards." + m[2].str();
    const auto [dt, sh] = src.info(name);
    p.rows = sh[0];
    p.width = sh[1];
    if (p.width % 32) fail(name + ": n-gram rows of " + std::to_string(p.width) + " values (group 32)");
    if (src.quantized(name)) {
      const auto bg = src.raw().quant_of(name.substr(0, name.size() - 7));
      p.copy = bg.first == ple_bits && bg.second == 32 && src.raw().locate(name.substr(0, name.size() - 7) + ".scales").dtype == "BF16";
    }
    ple.push_back(std::move(p));
  }

  // Every output tensor, planned up front (shards and offsets known before encoding).
  std::vector<StTensor> tensors;
  for (const Item& it : plan)
    for (StTensor t : outputs(it)) tensors.push_back(std::move(t));
  std::vector<std::string> kept;
  for (const std::string& name : src.names()) {
    if (replaced.count(name) || skipped(name) || (is_ple(name) && std::regex_match(name, ple_re))) continue;
    auto [dtype, shape] = src.info(name);
    if (const auto* kl = keep.of(name)) shape[0] = int64_t(kl->size());
    tensors.push_back({name, dtype, shape});
    kept.push_back(name);
  }
  for (const PleShard& p : ple) {
    tensors.push_back({p.out + ".weight", "U32", {p.rows, p.width * ple_bits / 32}});
    tensors.push_back({p.out + ".scales", "BF16", {p.rows, p.width / 32}});
    tensors.push_back({p.out + ".biases", "BF16", {p.rows, p.width / 32}});
  }
  for (const MtpPart& p : mtp) {
    if (p.kind == MtpPart::Norm || p.kind == MtpPart::Raw) tensors.push_back(p.raw);
    else
      for (StTensor t : outputs(p.item)) tensors.push_back(std::move(t));
  }
  mkdirs(out);
  {  // shards of an earlier conversion into --out
    if (DIR* d = ::opendir(out.c_str())) {
      while (dirent* e = ::readdir(d)) {
        const std::string n = e->d_name;
        if (n.rfind("model-", 0) == 0 && ends_with(n, ".safetensors")) ::unlink((out + "/" + n).c_str());
      }
      ::closedir(d);
    }
  }
  const size_t shard = size_t(std::stod(a.get("shard-gb", "4")) * 1e9);
  StWriter w(out, tensors, shard, {{"format", "mlx"}, {"quantization", "eq"}}, [](int i, int n) {
    char b[64];
    std::snprintf(b, sizeof(b), "model-%05d-of-%05d.safetensors", i + 1, n);
    return std::string(b);
  });

  json::Value qcfg = json::Value::object();
  qcfg.set("group_size", json::Value(64));
  qcfg.set("bits", json::Value(8));
  qcfg.set("mode", json::Value("affine"));
  std::map<std::string, int64_t> counts;
  int gptq_fallbacks = 0;
  for (size_t i = 0; i < plan.size(); ++i) {
    const Item& it = plan[i];
    std::vector<float> W = src.f32(it.source() + ".weight");
    src.release();
    const std::vector<int>* kl = keep.of(it.source() + ".weight");
    std::vector<int> range;  // a split stack's experts, in the kept order
    if (it.e1) {
      for (int e = it.e0; e < it.e1; ++e) range.push_back(kl ? (*kl)[size_t(e)] : e);
      kl = &range;
    }
    if (kl) W = slice_experts(W, nexp, *kl);
    const Stats* st = stats ? stats->get(it.source()) : nullptr;
    Stats sliced;
    std::vector<float> sd, sx;
    st = slice_stats(st, nexp, kl, sliced, sd, sx);
    if (st && st->C && st->C != it.shape.back()) fail(it.path + ": statistics for " + std::to_string(st->C) + " inputs");
    // A dense +gptq matrix without its full E[x x^T] is encoded with +imat (or +mse): say so.
    if (stats && it.f.enc == Fmt::GPTQ && it.shape.size() == 2 && !(st && (st->H || st->Hpacked))) {
      std::fprintf(stderr, "\n[quantize] warning: %s: +gptq without a full E[x x^T] (%s): encoded with %s\n",
                   it.path.c_str(), st ? "only its diagonal, inputs wider than the capture's --hessian-max" : "no statistics",
                   st ? "+imat" : "+mse");
      ++gptq_fallbacks;
    }
    Stats full;
    std::vector<float> hbuf;  // a packed E[x x^T] unpacked for this matrix only
    st = full_hessian(st, full, hbuf);
    qcfg.set(it.path, encode_write(w, it, W.data(), st));
    const std::string tag = it.f.kind == Fmt::VQ    ? "EQV" + std::to_string(it.f.bits)
                            : it.f.kind == Fmt::EQC ? "EQ" + std::to_string(it.f.bits) + "cb"
                            : it.f.kind == Fmt::EQ  ? "EQ" + std::to_string(it.f.bits)
                                                    : "mlx" + std::to_string(it.f.bits);
    counts[tag] += numel(it.shape);
    std::fprintf(stderr, "\r[quantize] %zu/%zu %s %s  %.0fs\033[K", i + 1, plan.size(), it.path.c_str(), it.f.spec.c_str(),
                 native::ms_since(t0) / 1000.0);
  }
  std::fprintf(stderr, "\n");
  if (gptq_fallbacks)
    std::fprintf(stderr, "[quantize] warning: %d +gptq matrices encoded without GPTQ (capture again with a larger --hessian-max)\n",
                 gptq_fallbacks);
  for (const std::string& name : kept) {
    const std::vector<int>* kl = keep.of(name);
    if (src.quantized(name) || kl) {  // a dequantized MLX matrix kept as bf16; a pruned router
      std::vector<float> v = src.f32(name);
      if (kl) v = slice_experts(v, nexp, *kl);
      const std::string dt = src.info(name).first;
      if (dt == "F32") {
        w.write(name, v.data(), v.size() * 4);
      } else {
        std::vector<uint16_t> b(v.size());
        for (size_t i = 0; i < v.size(); ++i) b[i] = dt == "F16" ? f16_bits(v[i]) : bf16_bits(v[i]);
        w.write(name, b.data(), b.size() * 2);
      }
    } else {
      const TensorView t = src.view(name);
      w.write(name, t.data, t.nbytes);
    }
    src.release();
  }
  for (size_t i = 0; i < ple.size(); ++i) {
    const PleShard& p = ple[i];
    const std::string sb = p.src.substr(0, p.src.size() - 7);
    if (p.copy) {
      for (const char* x : {".weight", ".scales", ".biases"}) {
        const auto l = src.raw().locate(sb + x);
        w.write(p.out + x, l.data, l.nbytes);
      }
    } else {
      const std::vector<float> v = src.f32(p.src);
      const AffOut o = encode_affine(v.data(), p.rows, int(p.width), 0, parse_fmt("affine:" + std::to_string(ple_bits) + ":32:bf16"), nullptr);
      const std::vector<uint32_t> packed = pack_codes(o.q.data(), p.rows, int(p.width), ple_bits);
      w.write(p.out + ".weight", packed.data(), packed.size() * 4);
      const auto s16 = half_bits(o.s, "bf16"), b16 = half_bits(o.b, "bf16");
      w.write(p.out + ".scales", s16.data(), s16.size() * 2);
      w.write(p.out + ".biases", b16.data(), b16.size() * 2);
    }
    json::Value q = json::Value::object();
    q.set("bits", json::Value(ple_bits));
    q.set("group_size", json::Value(32));
    qcfg.set(p.out, std::move(q));
    std::fprintf(stderr, "\r[quantize] n-gram table %zu/%zu  %.0fs\033[K", i + 1, ple.size(), native::ms_since(t0) / 1000.0);
  }
  if (!ple.empty()) std::fprintf(stderr, "\n");
  for (const MtpPart& p : mtp) {
    const auto l = src.raw().locate(p.src[0]);
    TensorView t;
    t.data = l.data;
    t.dtype = l.dtype;
    t.shape = l.shape;
    if (p.kind == MtpPart::Norm) {
      std::vector<float> v = to_f32(t);
      for (float& x : v) x += 1.0f;
      w.write(p.raw.name, v.data(), v.size() * 4);
    } else if (p.kind == MtpPart::Raw) {
      const std::vector<float> v = to_f32(t);
      std::vector<uint16_t> b(v.size());
      for (size_t i = 0; i < v.size(); ++i) b[i] = bf16_bits(v[i]);
      w.write(p.raw.name, b.data(), b.size() * 2);
    } else {
      const std::vector<float> W = mtp_weights(src.raw(), p);
      qcfg.set(p.item.path, encode_write(w, p.item, W.data(), nullptr));
    }
  }
  w.finish();

  // index, config, tokenizer and the other small files
  json::Value wm = json::Value::object();
  for (const auto& [k, f] : w.weight_map()) wm.set(k, json::Value(f));
  json::Value idx = json::Value::object();
  json::Value md = json::Value::object();
  md.set("total_size", json::Value(w.total_bytes()));
  idx.set("metadata", std::move(md));
  idx.set("weight_map", std::move(wm));
  const json::DumpOptions pretty{",", ": ", false, 1, false};
  write_text(out + "/model.safetensors.index.json", json::dump(idx, pretty));
  json::Value config = json::parse(native::read_file(model + "/config.json"));
  if (keep.K) {  // the pruned expert count (the largest; each layer's own in num_experts_per_layer)
    json::Value* t = config.get("text_config");
    json::Value& tcw = t && t->is_object() ? *t : config;
    tcw.set("num_experts", json::Value(keep.K));
    if (t && config.get("num_experts")) config.set("num_experts", json::Value(keep.K));
    json::Value per = json::Value::array();
    for (int l = 0; l < int(tc.get_int("num_hidden_layers", 0)); ++l) per.push_back(json::Value(int(keep.lists.at(std::to_string(l)).size())));
    tcw.set("num_experts_per_layer", std::move(per));
    if (keep.lists.count("mtp") && !no_mtp) tcw.set("mtp_num_experts", json::Value(int(keep.lists.at("mtp").size())));
  }
  config.set("quantization", qcfg);
  config.set("quantization_config", qcfg);
  json::Value eqf = json::Value::object();
  eqf.set("version", json::Value(2));
  eqf.set("preset", json::Value(a.has("fmt") ? "custom" : pname));
  eqf.set("rules", json::Value(rules_spec));
  config.set("eq_format", std::move(eqf));
  write_text(out + "/config.json", json::dump(config, pretty));
  if (DIR* d = ::opendir(model.c_str())) {
    while (dirent* e = ::readdir(d)) {
      const std::string n = e->d_name;
      if ((ends_with(n, ".json") || ends_with(n, ".jinja")) && n != "config.json" && n != "model.safetensors.index.json")
        copy_file(model + "/" + n, out + "/" + n);
    }
    ::closedir(d);
  }
  int64_t n = 0;
  for (const auto& [_, c] : counts) n += c;
  std::string mix;
  for (const auto& [k, c] : counts) {
    char b[64];
    std::snprintf(b, sizeof(b), "%s%s %.1f%%", mix.empty() ? "" : ", ", k.c_str(), 100.0 * double(c) / double(std::max<int64_t>(n, 1)));
    mix += b;
  }
  std::fprintf(stderr, "[quantize] %s: %.2f GB, %.3f bits per weight; %s (%.0f s)\n", out.c_str(), double(w.total_bytes()) / 1e9,
               8.0 * double(w.total_bytes()) / double(params), mix.c_str(), native::ms_since(t0) / 1000.0);
  if (!ple.empty()) {  // Qwen4-Exp: what is resident (the n-gram table is read from disk)
    int64_t pb = 0, pp = 0;
    for (const PleShard& p : ple) pp += p.rows * p.width, pb += p.rows * p.width * (ple_bits + 1) / 8;
    std::fprintf(stderr, "[quantize] without the n-gram table (%.2f GB on disk): %.2f GB, %.3f bits per weight\n", double(pb) / 1e9,
                 double(int64_t(w.total_bytes()) - pb) / 1e9, 8.0 * double(int64_t(w.total_bytes()) - pb) / double(params - pp));
  }
  if (a.has("tiles")) {  // Espetero's own load of the new checkpoint writes its tile image
    ::setenv("ESPETERO_TILES", "write", 1);
    LoadOptions o;
    o.kv_gb = 1;
    o.slots = 1;
    auto m = NativeModel::load(out, o);
    std::fprintf(stderr, "[quantize] tile image written (load %.0f ms)\n", m->info().load_ms);
  }
  return 0;
}

}  // namespace espetero::quant
