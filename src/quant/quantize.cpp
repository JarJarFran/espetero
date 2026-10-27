// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// espetero quantize: a Hugging Face BF16 checkpoint (Qwen3.5 / Qwen3.6
// family, dense Gemma 4) -> an EQ checkpoint (Espetero's own format), without Python.
// The C++ form of kld/eq_convert.py (+ quant_lab.py, kld_harness.py calib):
// calibration statistics from Espetero's own forward of the BF16 weights,
// the converter's presets and rules, its encoders, and its output: the MLX
// directory layout with NAME.weight / .scales / .super (/ .codebook) per EQ
// matrix, config.json "quantization" and "eq_format" {"version": 2}, the MTP
// head as MLX affine, routers and norms as the source holds them.

#include "quantize.h"

#include <dirent.h>
#include <sys/stat.h>

#include <cstdio>
#include <fstream>
#include <regex>
#include <set>

#include "../model/api.h"
#include "../model/impl/hf_source.h"
#include "calib.h"
#include "encode.h"
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

// eq_convert.preset
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
const std::vector<std::string> kDensePresets = {"eqd4m", "eqd4l", "eqd5", "eqd6"};

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
  if (p == "eq4dhb" || p == "eq3h" || p == "eq3dh" || p == "eq3dx" || p == "eq3dhb" ||
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
  if (it.f.kind == Fmt::EQ || it.f.kind == Fmt::EQC) {
    v.push_back({it.path + ".weight", "U32", with(C * it.f.bits / 32)});
    v.push_back({it.path + ".scales", "I8", with(C / 16)});
    v.push_back({it.path + ".super", "F16", with(C / 256)});
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
    "         [--calib-out FILE] [--calib-only] [--calib-experts f16|q8] [--calib-layers A:B]\n"
    "         [--hessian-max N] [--hessian-pack N] [--no-mtp] [--shard-gb G] [--tiles]\n"
    "       espetero quantize --preset NAME --print-rules [--model HF_DIR | --layers N]\n"
    "\n"
    "A Hugging Face BF16 checkpoint (Qwen3.5 / Qwen3.6, dense Gemma 4) to an EQ checkpoint.\n"
    "  --preset   eq4 eq5 eq6 eq4x eq5x eq4d eq5d eq4f eq4s eq6h eq5dh eq4dh, a 4-bit one + c\n"
    "             (EQ4 experts with codebooks: eq4dhc ...), eq4dhb (default eq4);\n"
    "             EQ3 codebook experts (MoE only): eq3h eq3dh eq3dx eq3dhb;\n"
    "             dense Qwen3.5 / 3.6 (widths by tensor class and layer position, sized at or\n"
    "             under the GGUF named): eqd4m (Q4_K_M) eqd4l (UD-Q4_K_XL) eqd5 (UD-Q5_K_XL) eqd6 (Q6_K)\n"
    "  --fmt      rules GLOB=FMT;... instead (eql:B:16[+enc], eql:4:16:cb[+enc], eql:3:16:cb[+enc],\n"
    "             affine:B:G[:SD][+enc], bf16; enc rtn | mse | imat | gptq)\n"
    "  --calib    statistics from an earlier run (or tools/dev/calib_npz.py); without it they are\n"
    "             captured first with Espetero's forward of the BF16 weights over --calib-data\n"
    "             (default file:~/.cache/mlx-lm/calibration_v5.txt), windows of --ctx (2048)\n"
    "             tokens up to --calib-tokens (262144), saved to --calib-out (default\n"
    "             OUT.calib.safetensors); --calib none: unweighted encoders; --calib-experts q8\n"
    "             runs the forward's experts as 8-bit copies (less memory; default f16, exact);\n"
    "             several comma-separated files read as one (the parts of --calib-layers captures)\n"
    "  --hessian-max   capture the full E[x x^T] of inputs up to N wide (16384), else only its\n"
    "             diagonal (+gptq then falls back to +imat); --hessian-pack N saves full ones wider\n"
    "             than N (16384) as their upper triangle (half the bytes)\n"
    "  --tiles    also write the checkpoint's tile image (faster loads)\n";

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
    co.model = model;
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
    co.hessian_max = std::stoi(a.get("hessian-max", std::to_string(co.hessian_max)));
    co.hessian_pack = std::stoi(a.get("hessian-pack", std::to_string(co.hessian_pack)));
    capture(co);
  }
  if (a.has("calib-only")) return 0;
  std::unique_ptr<CalibFile> stats;
  if (calib != "none") stats = std::make_unique<CalibFile>(calib);

  HFSource src(model);
  const json::Value& cfg = src.config();
  const json::Value* tcp = cfg.get("text_config");
  const json::Value& tc = tcp && tcp->is_object() ? *tcp : cfg;
  const bool moe = tc.get_int("num_experts", 0) > 0;

  // The plan (kld_harness.plan_formats): an explicit rule first, then the
  // model's predicate (MoE routers 8-bit), then the first matching rule;
  // bf16 and widths the format cannot divide stay as they are.
  std::vector<Item> plan;
  std::set<std::string> replaced;
  int64_t params = 0;
  for (const std::string& name : src.names()) {
    const auto [dtype, shape] = src.info(name);
    params += numel(shape);
    if (!ends_with(name, ".weight") || shape.size() < 2 || name.find("conv1d") != std::string::npos) continue;
    const std::string p = name.substr(0, name.size() - 7);
    const Fmt* f = nullptr;
    Fmt pred;
    for (const Rule& r : rules)
      if (r.glob != "*" && fnmatch(p.c_str(), r.glob.c_str())) {
        f = &r.f;
        break;
      }
    if (!f && moe && (ends_with(p, "mlp.gate") || ends_with(p, "shared_expert_gate"))) {
      pred = parse_fmt("affine:8:64");
      f = &pred;
    }
    if (!f)
      for (const Rule& r : rules)
        if (fnmatch(p.c_str(), r.glob.c_str())) {
          f = &r.f;
          break;
        }
    if (!f || f->kind == Fmt::BF16) continue;
    if (shape.back() % f->group) continue;
    Item it{p, *f, shape};
    if (it.f.kind == Fmt::EQ && !(it.f.bits == 3 || it.f.bits == 4 || it.f.bits == 5 || it.f.bits == 6 || it.f.bits == 8))
      it.f = parse_fmt("affine:8:64:f16");
    plan.push_back(std::move(it));
    replaced.insert(name);
  }

  // MTP head (eq_convert: MLX affine 8-bit, experts 4-bit +mse).
  std::vector<MtpPart> mtp;
  if (!a.has("no-mtp")) mtp = mtp_plan(src.raw(), parse_fmt("affine:8:64:f16"), parse_fmt("affine:4:64+mse"));

  // Every output tensor, planned up front (shards and offsets known before encoding).
  std::vector<StTensor> tensors;
  for (const Item& it : plan)
    for (StTensor t : outputs(it)) tensors.push_back(std::move(t));
  std::vector<std::string> keep;
  for (const std::string& name : src.names()) {
    if (replaced.count(name)) continue;
    const auto [dtype, shape] = src.info(name);
    tensors.push_back({name, dtype, shape});
    keep.push_back(name);
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
    const TensorView t = src.tensor(it.path + ".weight");
    const std::vector<float> W = to_f32(t);
    src.release();
    const Stats* st = stats ? stats->get(it.path) : nullptr;
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
    const std::string tag = it.f.kind == Fmt::EQC ? "EQ" + std::to_string(it.f.bits) + "cb" : it.f.kind == Fmt::EQ ? "EQ" + std::to_string(it.f.bits)
                                                                                   : "mlx" + std::to_string(it.f.bits);
    counts[tag] += numel(it.shape);
    std::fprintf(stderr, "\r[quantize] %zu/%zu %s %s  %.0fs\033[K", i + 1, plan.size(), it.path.c_str(), it.f.spec.c_str(),
                 native::ms_since(t0) / 1000.0);
  }
  std::fprintf(stderr, "\n");
  if (gptq_fallbacks)
    std::fprintf(stderr, "[quantize] warning: %d +gptq matrices encoded without GPTQ (capture again with a larger --hessian-max)\n",
                 gptq_fallbacks);
  for (const std::string& name : keep) {
    const TensorView t = src.tensor(name);
    w.write(name, t.data, t.nbytes);
    src.release();
  }
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
