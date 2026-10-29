// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#define ACCELERATE_NEW_LAPACK
#include "calib.h"

#include <Accelerate/Accelerate.h>
#include <glob.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "../model/api.h"
#include "../tok/tokenizer.h"

namespace espetero::quant {

using native::parallel_for;

// ---------------------------------------------------------------- reading

CalibFile::CalibFile(const std::string& paths) {
  std::map<std::string, std::string> owner;  // tensor -> file
  std::string tokens;
  for (size_t a = 0;;) {
    const size_t b = paths.find(',', a);
    const std::string path = paths.substr(a, b - a);
    f_.push_back(std::make_unique<StFile>(path));
    const StFile& f = *f_.back();
    if (auto t = f.metadata().find("tokens"); t != f.metadata().end()) {
      if (!tokens.empty() && t->second != tokens) fail(path + ": " + t->second + " tokens, the other files " + tokens);
      tokens = t->second;
    }
    for (const auto& [k, e] : f.entries()) {
      if (k.size() < 3 || k[1] != ':') continue;
      if (k[0] == 'N') continue;  // per-expert routing counts (not an input statistic)
      if (e.dtype != "F32") fail(path + ": " + k + " is not float32");
      if (auto [it, fresh] = owner.emplace(k, path); !fresh) fail(path + ": " + k + " also in " + it->second);
      const std::string mod = k.substr(2);
      Stats& s = stats_[mod];
      s.C = int(e.shape.back());
      const float* p = reinterpret_cast<const float*>(e.data);
      switch (k[0]) {
        case 'H': s.H = p; break;
        case 'U': {
          const int64_t n = e.shape.back();
          int64_t C = int64_t((std::sqrt(8.0 * double(n) + 1.0) - 1.0) / 2.0 + 0.5);
          if (C * (C + 1) / 2 != n) fail(path + ": " + k + " has " + std::to_string(n) + " values, not a triangle");
          s.Hpacked = p, s.C = int(C);
          break;
        }
        case 'D': s.diag = p, s.diag_rows = 1; break;
        case 'E': s.diag = p, s.diag_rows = int(e.shape[0]); break;
        case 'X': s.Hexp = p, s.nexp = int(e.shape[0]); break;
        default: break;
      }
    }
    if (b == std::string::npos) break;
    a = b + 1;
  }
  for (const auto& f : f_)
    if (auto a = f->metadata().find("ALIAS"); a != f->metadata().end()) {
      const json::Value al = json::parse(a->second);
      const json::Object& o = al.as_object();
      for (size_t i = 0; i < o.size(); ++i)
        if (auto it = stats_.find(o.vals[i].as_string()); it != stats_.end()) stats_[o.keys[i]] = Stats(it->second);
    }
  // Stacked experts' gate/up read the router's input: its full E[x x^T] is
  // the shared part of their per-expert Hessians.
  for (auto& [k, v] : stats_)
    for (const char* proj : {"switch_mlp.gate_proj", "switch_mlp.up_proj"})
      if (native::ends_with(k, proj)) {
        const std::string pre = k.substr(0, k.size() - std::strlen(proj));
        for (const char* rn : {"gate", "router.layer"})
          if (auto r = stats_.find(pre + rn); r != stats_.end() && r->second.H) v.Hshared = r->second.H;
      }
}

const Stats* CalibFile::get(const std::string& module) const {
  auto it = stats_.find(module);
  return it == stats_.end() ? nullptr : &it->second;
}

const Stats* full_hessian(const Stats* st, Stats& tmp, std::vector<float>& buf) {
  if (!st || st->H || !st->Hpacked) return st;
  const size_t C = size_t(st->C);
  buf.resize(C * C);
  parallel_for(C, [&](size_t i) {  // row i of the triangle starts at i C - i (i - 1) / 2
    const float* r = st->Hpacked + i * C - i * (i - 1) / 2;
    for (size_t j = i; j < C; ++j) buf[i * C + j] = r[j - i];
  });
  parallel_for(C, [&](size_t i) {
    for (size_t j = 0; j < i; ++j) buf[i * C + j] = buf[j * C + i];
  });
  tmp = *st;
  tmp.H = buf.data();
  return &tmp;
}

// ---------------------------------------------------------------- capture

namespace {

std::string read_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fail("cannot read " + path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

std::string expand_home(const std::string& p) {
  if (p.rfind("~/", 0) == 0) {
    const char* h = std::getenv("HOME");
    return std::string(h ? h : "") + p.substr(1);
  }
  return p;
}

// Calibration text: file:PATH, or glob:PATTERN (files sorted, joined
// by blank lines).
std::string source_text(const std::string& src) {
  if (src.rfind("file:", 0) == 0) return read_text(expand_home(src.substr(5)));
  if (src.rfind("glob:", 0) == 0) {
    glob_t g{};
    const std::string pat = expand_home(src.substr(5));
    if (::glob(pat.c_str(), 0, nullptr, &g) != 0 || g.gl_pathc == 0) fail("no files for " + src);
    std::vector<std::string> files;
    for (size_t i = 0; i < g.gl_pathc; ++i)
      if (!native::is_dir(g.gl_pathv[i])) files.push_back(g.gl_pathv[i]);
    globfree(&g);
    std::sort(files.begin(), files.end());
    std::string out;
    for (size_t i = 0; i < files.size(); ++i) out += (i ? "\n\n" : "") + read_text(files[i]);
    return out;
  }
  fail("calibration data " + src + ": file:PATH or glob:PATTERN");
}

struct Dense {
  std::vector<float> a;  // [C][C] (upper triangle) or [C]
  int C = 0;
  bool full = true;
  double n = 0;
};

struct Experts {
  int E = 0, C = 0;
  std::vector<float> diag, cnt;  // [E][C], [E]
  std::vector<double> wsum;      // [E] routing-weight sums (expert_route_weight)
  std::vector<float> X;          // [E][C][C] (upper) or empty
};

class Sink : public ActivationSink {
 public:
  Sink(const json::Value& cfg, const CaptureOptions& o) : o_(o) {
    const json::Value* tcp = cfg.get("text_config");
    const json::Value& tc = tcp && tcp->is_object() ? *tcp : cfg;
    const int L = int(tc.get_int("num_hidden_layers", 0));
    linear_.assign(size_t(L), false);
    if (const json::Value* lt = tc.get("layer_types"); lt && lt->is_array()) {
      for (int i = 0; i < L && i < int(lt->size()); ++i) linear_[size_t(i)] = (*lt)[size_t(i)].as_string() == "linear_attention";
    } else {
      const int iv = int(tc.get_int("full_attention_interval", 4));
      for (int i = 0; i < L; ++i) linear_[size_t(i)] = (i + 1) % iv != 0;
    }
    moe_ = tc.get_int("num_experts", 0) > 0;
    nexp_ = int(tc.get_int("num_experts", 0));
    q4_ = cfg.get_string("model_type", "").rfind("qwen4_exp", 0) == 0 || tc.get_string("model_type", "").rfind("qwen4_exp", 0) == 0;
    nl_ = L;
    tied_ = cfg.get_bool("tie_word_embeddings", tc.get_bool("tie_word_embeddings", false));
    mlp_in_.resize(size_t(L));
    lm_ = o.layer_hi >= L;
    spark_ = cfg.get_string("model_type", "") == "spark2_5";
  }

  void tap(int layer, int kind, const uint16_t* rows, int n, int width, const int32_t* ids, const int32_t* pos,
           int k) override {
    if (q4_ && layer == nl_) {  // Qwen4-Exp: the final mixer's inputs (the LM head's: lm())
      if (!lm_) return;
      std::vector<float> X(static_cast<size_t>(n) * size_t(width));
      to_f32(rows, X.data(), X.size());
      const std::string M = "language_model.model.hyper_connection_mixer.";
      if (kind == kHcAttnN) add(M + "input_mix_weight_down", X, n, width, {});
      else if (kind == kHcAttnLow) add(M + "input_mix_weight_up", X, n, width, {});
      else fail("calibration: unknown mixer tap kind " + std::to_string(kind));
      return;
    }
    if (layer < o_.layer_lo || layer >= o_.layer_hi) return;
    if (spark_) {  // spark2_5: HF names (model.*), fused q_k_v_proj beside g_proj
      const std::string P = "model.layers." + std::to_string(layer) + ".";
      std::vector<float> X(static_cast<size_t>(n) * size_t(width));
      to_f32(rows, X.data(), X.size());
      switch (kind) {
        case kAttnIn: add(P + "self_attn.q_k_v_proj", X, n, width, {P + "self_attn.g_proj"}); break;
        case kMixOut: add(P + "self_attn.out_proj", X, n, width, {}); break;
        case kMlpIn: add(P + "mlp.gate_proj", X, n, width, {P + "mlp.up_proj"}); break;
        case kDownIn: add(P + "mlp.down_proj", X, n, width, {}); break;
        default: fail("calibration: unknown tap kind " + std::to_string(kind));
      }
      return;
    }
    const std::string P = "language_model.model.layers." + std::to_string(layer) + ".";
    const bool lin = linear_.at(size_t(layer));
    // kExpertIn: one row per routed pair (n * k rows)
    std::vector<float> X(static_cast<size_t>(n) * size_t(kind == kExpertIn ? k : 1) * size_t(width));
    to_f32(rows, X.data(), X.size());
    switch (kind) {
      case kAttnIn:
        if (lin) add(P + "linear_attn.in_proj_qkv", X, n, width,
                     {P + "linear_attn.in_proj_z", P + "linear_attn.in_proj_b", P + "linear_attn.in_proj_a"});
        else if (q4_)  // (the sparse-attention indexer reads the attention input too)
          add(P + "self_attn.q_proj", X, n, width, {P + "self_attn.k_proj", P + "self_attn.v_proj", P + "self_attn.indexer.index_qk_proj"});
        else add(P + "self_attn.q_proj", X, n, width, {P + "self_attn.k_proj", P + "self_attn.v_proj"});
        break;
      case kHcAttnN:
      case kHcMlpN: {
        const std::string B = P + (kind == kHcAttnN ? "attn" : "mlp") + "_hyper_connection.";
        add(B + "input_mix_weight_down", X, n, width, {B + "block_inject_weight"});
        break;
      }
      case kHcAttnLow:
      case kHcMlpLow: add(P + (kind == kHcAttnLow ? "attn" : "mlp") + "_hyper_connection.input_mix_weight_up", X, n, width, {}); break;
      case kPleIn: add(P + "ple.key_proj", X, n, width, {P + "ple.value_proj"}); break;
      case kMixOut: add(P + (lin ? "linear_attn.out_proj" : "self_attn.o_proj"), X, n, width, {}); break;
      case kMlpIn:
        if (moe_) {
          add(P + "mlp.gate", X, n, width,
              {P + "mlp.shared_expert.gate_proj", P + "mlp.shared_expert.up_proj", P + "mlp.shared_expert_gate"});
          mlp_in_[size_t(layer)] = std::move(X);
        } else {
          add(P + "mlp.gate_proj", X, n, width, {P + "mlp.up_proj"});
        }
        break;
      case kDownIn: add(P + (moe_ ? "mlp.shared_expert.down_proj" : "mlp.down_proj"), X, n, width, {}); break;
      case kExpertIn: experts(P, layer, X, n, width, ids, pos, k); break;
      default: fail("calibration: unknown tap kind " + std::to_string(kind));
    }
  }

  // The LM head's input: the final-normed rows of the pass (float32).
  void lm(const float* hid, int n, int D) {
    if (!lm_) return;
    std::vector<float> X(hid, hid + size_t(n) * size_t(D));
    if (spark_) add(tied_ ? "model.embedding" : "lm_head", X, n, D, {});
    else add(tied_ ? "language_model.model.embed_tokens" : "language_model.lm_head", X, n, D, {});
  }

  void save(const std::string& path, int64_t tokens) {
    std::vector<std::pair<StTensor, const void*>> ts;
    for (auto& [k, d] : dense_) {
      const float inv = float(1.0 / std::max(d.n, 1.0));
      if (d.full && d.C > o_.hessian_pack) {
        // Upper triangle by rows, packed in place (row i's start never passes its old position)
        size_t o = 0;
        for (int i = 0; i < d.C; ++i)
          for (int j = i; j < d.C; ++j) d.a[o++] = d.a[size_t(i) * d.C + j] * inv;
        ts.push_back({{"U:" + k, "F32", {int64_t(o)}}, d.a.data()});
      } else if (d.full) {
        for (int i = 0; i < d.C; ++i)
          for (int j = i; j < d.C; ++j) {
            const float v = d.a[size_t(i) * d.C + j] * inv;
            d.a[size_t(i) * d.C + j] = d.a[size_t(j) * d.C + i] = v;
          }
        ts.push_back({{"H:" + k, "F32", {d.C, d.C}}, d.a.data()});
      } else {
        for (float& v : d.a) v *= inv;
        ts.push_back({{"D:" + k, "F32", {d.C}}, d.a.data()});
      }
    }
    for (auto& [k, x] : experts_) {
      if (k.find("gate_proj") != std::string::npos) {  // routed rows per expert of the layer (the coverage)
        std::vector<float> c = x.cnt;
        std::sort(c.begin(), c.end());
        const int zero = int(std::count(c.begin(), c.end(), 0.0f));
        std::fprintf(stderr, "[calib] %s: routed rows per expert min %.0f p10 %.0f median %.0f max %.0f%s\n", k.c_str(), c.front(),
                     c[c.size() / 10], c[c.size() / 2], c.back(), zero ? (", " + std::to_string(zero) + " never routed").c_str() : "");
      }
      for (int e = 0; e < x.E; ++e) {
        const float inv = o_.expert_route_weight ? float(1.0 / std::max(x.wsum[size_t(e)], 1e-12))
                                                 : 1.0f / std::max(x.cnt[size_t(e)], 1.0f);
        for (int i = 0; i < x.C; ++i) x.diag[size_t(e) * x.C + i] *= inv;
        if (!x.X.empty()) {
          float* h = x.X.data() + size_t(e) * x.C * x.C;
          for (int i = 0; i < x.C; ++i)
            for (int j = i; j < x.C; ++j) {
              const float v = h[size_t(i) * x.C + j] * inv;
              h[size_t(i) * x.C + j] = h[size_t(j) * x.C + i] = v;
            }
        }
      }
      ts.push_back({{"E:" + k, "F32", {x.E, x.C}}, x.diag.data()});
      if (!x.X.empty()) ts.push_back({{"X:" + k, "F32", {x.E, x.C, x.C}}, x.X.data()});
      // Qwen4-Exp: how many routed rows each expert saw (for pruning: tools, the --keep-experts lists)
      if (q4_) ts.push_back({{"N:" + k, "F32", {x.E}}, x.cnt.data()});
    }
    json::Value al = json::Value::object();
    for (const auto& [a, b] : alias_) al.set(a, json::Value(b));
    std::map<std::string, std::string> meta{{"ALIAS", json::dump(al)},
                                            {"tokens", std::to_string(tokens)},
                                            {"source", "espetero quantize (fp16 activations)"}};
    if (o_.expert_route_weight) meta["expert_weight"] = "router";
    st_save(path, ts, meta);
    std::fprintf(stderr, "[calib] %zu matrices (+%zu sharing an input), %lld tokens -> %s\n", ts.size(), alias_.size(),
                 (long long)tokens, path.c_str());
  }

 private:
  static void to_f32(const uint16_t* h, float* f, size_t n) {
    parallel_for((n + 65535) / 65536, [&](size_t b) {
      const size_t i0 = b * 65536, i1 = std::min(n, i0 + 65536);
      for (size_t i = i0; i < i1; ++i) f[i] = native::f16_to_f32(h[i]);
    });
  }

  void add(const std::string& key, const std::vector<float>& X, int n, int C, std::initializer_list<std::string> also) {
    for (const std::string& a : also) alias_[a] = key;
    Dense& d = dense_[key];
    if (d.C == 0) {
      d.C = C;
      d.full = C <= o_.hessian_max;
      d.a.assign(d.full ? size_t(C) * C : size_t(C), 0.0f);
    }
    if (d.full) {
      cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, C, n, 1.0f, X.data(), C, 1.0f, d.a.data(), C);
    } else {
      for (int r = 0; r < n; ++r)
        for (int i = 0; i < C; ++i) d.a[size_t(i)] += X[size_t(r) * C + i] * X[size_t(r) * C + i];
    }
    d.n += n;
  }

  void tap_route_weights(int layer, const float* w, int pairs) override {
    route_w_.assign(w, w + pairs);
    route_w_layer_ = layer;
  }

  void experts(const std::string& P, int layer, const std::vector<float>& act, int n, int Ie, const int32_t* ids,
               const int32_t* pos, int k) {
    const bool rw = o_.expert_route_weight;
    if (rw && (route_w_layer_ != layer || route_w_.size() != size_t(n) * size_t(k)))
      fail("calibration: routing weights missing for layer " + std::to_string(layer));
    const std::vector<float>& xin = mlp_in_.at(size_t(layer));
    if (xin.empty()) fail("calibration: expert rows without the layer's MLP input");
    const int D = int(xin.size() / size_t(n));
    const int E = nexp_;
    const size_t pairs = size_t(n) * size_t(k);
    std::vector<std::vector<int>> by(static_cast<size_t>(E));
    for (size_t p = 0; p < pairs; ++p) {
      if (ids[p] < 0 || ids[p] >= E) fail("calibration: expert id out of range");
      by[size_t(ids[p])].push_back(int(p));
    }
    Experts& gu = experts_[P + "mlp.switch_mlp.gate_proj"];
    Experts& dn = experts_[P + "mlp.switch_mlp.down_proj"];
    alias_[P + "mlp.switch_mlp.up_proj"] = P + "mlp.switch_mlp.gate_proj";
    if (gu.E == 0) {
      gu.E = dn.E = E;
      gu.C = D;
      dn.C = Ie;
      gu.diag.assign(size_t(E) * D, 0.0f);
      gu.cnt.assign(size_t(E), 0.0f);
      dn.diag.assign(size_t(E) * Ie, 0.0f);
      dn.cnt.assign(size_t(E), 0.0f);
      gu.wsum.assign(size_t(E), 0.0);
      dn.wsum.assign(size_t(E), 0.0);
      if (Ie <= o_.expert_hessian_max) dn.X.assign(size_t(E) * Ie * Ie, 0.0f);
      if (D <= o_.expert_hessian_max) gu.X.assign(size_t(E) * D * D, 0.0f);
    }
    parallel_for(size_t(E), [&](size_t e) {
      const std::vector<int>& ps = by[e];
      if (ps.empty()) return;
      float* g = gu.diag.data() + e * D;
      float* d = dn.diag.data() + e * Ie;
      std::vector<float> rows_dn(ps.size() * size_t(Ie)), rows_gu(gu.X.empty() ? 0 : ps.size() * size_t(D));
      double ws = 0;
      for (size_t i = 0; i < ps.size(); ++i) {
        const int p = ps[i];
        const float* x = xin.data() + size_t(p / k) * D;
        const float* a = act.data() + size_t(pos[p]) * Ie;
        if (!rw) {
          for (int c = 0; c < D; ++c) g[c] += x[c] * x[c];
          if (!rows_gu.empty()) std::memcpy(&rows_gu[i * D], x, sizeof(float) * D);
          for (int c = 0; c < Ie; ++c) d[c] += a[c] * a[c];
          std::memcpy(&rows_dn[i * Ie], a, sizeof(float) * Ie);
          continue;
        }
        // Weighted: w x x^T (the rows scaled by sqrt(w) for the syrk)
        const float wt = std::max(route_w_[size_t(p)], 0.0f), s = std::sqrt(wt);
        ws += wt;
        for (int c = 0; c < D; ++c) g[c] += wt * x[c] * x[c];
        if (!rows_gu.empty())
          for (int c = 0; c < D; ++c) rows_gu[i * D + c] = s * x[c];
        for (int c = 0; c < Ie; ++c) d[c] += wt * a[c] * a[c];
        for (int c = 0; c < Ie; ++c) rows_dn[i * Ie + c] = s * a[c];
      }
      gu.cnt[e] += float(ps.size());
      dn.cnt[e] += float(ps.size());
      gu.wsum[e] += ws;
      dn.wsum[e] += ws;
      if (!dn.X.empty())
        cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, Ie, int(ps.size()), 1.0f, rows_dn.data(), Ie, 1.0f,
                    dn.X.data() + e * Ie * Ie, Ie);
      if (!gu.X.empty())
        cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, D, int(ps.size()), 1.0f, rows_gu.data(), D, 1.0f,
                    gu.X.data() + e * D * D, D);
    });
    mlp_in_[size_t(layer)].clear();
  }

  const CaptureOptions& o_;
  std::vector<bool> linear_;
  bool moe_ = false, tied_ = false, spark_ = false, q4_ = false;
  int nexp_ = 0, nl_ = 0;
  std::map<std::string, Dense> dense_;
  std::map<std::string, Experts> experts_;
  std::map<std::string, std::string> alias_;
  std::vector<std::vector<float>> mlp_in_;
  std::vector<float> route_w_;  // the routing weights of the next kExpertIn tap (tap_route_weights)
  int route_w_layer_ = -1;
  bool lm_ = true;
};

}  // namespace

void capture(const CaptureOptions& o) {
  const auto t0 = native::Clock::now();
  const json::Value cfg = json::parse(native::read_file(o.model + "/config.json"));
  auto tok = Tokenizer::from_hf_dir(o.model);
  std::vector<int32_t> ids;
  std::string data = o.data;
  for (size_t a = 0;;) {
    const size_t b = data.find('+', a);
    const std::string src = data.substr(a, b - a);
    tok->encode_append(source_text(src), ids, true);
    if (b == std::string::npos) break;
    a = b + 1;
  }
  // Gemma 4 expects <bos> first: every window starts with it (the harness's
  // reference windows do too), followed by the next ctx - 1 tokens.
  const json::Value* tcp = cfg.get("text_config");
  const json::Value& tc = tcp && tcp->is_object() ? *tcp : cfg;
  const std::string mt = cfg.get_string("model_type", "");
  const bool bos = mt == "gemma4" || mt == "gemma4_text";
  const int32_t bos_id = int32_t(tc.get_int("bos_token_id", cfg.get_int("bos_token_id", 2)));
  const int body = bos ? o.ctx - 1 : o.ctx;
  const int64_t nwin = std::min<int64_t>(int64_t(ids.size()) / body, o.max_tokens / o.ctx);
  if (nwin < 1) fail("calibration: " + std::to_string(ids.size()) + " tokens, fewer than one window of " + std::to_string(o.ctx));
  LoadOptions lo;
  lo.slots = 2;
  lo.kv_gb = o.kv_gb;
  lo.max_ctx = o.ctx + 64;
  lo.max_rows = std::max(lo.max_rows, o.ctx);
  lo.vision = false;
  lo.mtp = "none";
  lo.tune_max_m = 0;
  // Experts exact (fp16 copies of the bf16 weights, as the reference's fp32
  // forward reads them) unless --calib-experts q8 asks for the 8-bit copies
  // (half the memory: a 35B needs ~90 GB exact, ~60 GB with q8).
  if (o.experts_f16) ::setenv("ESPETERO_EXPERTS_F16", "1", 1);
  // Qwen4-Exp: no expert cache (its slots replace the expert ids the taps
  // need: every layer's experts are read whole per pass, or resident).
  lo.expert_cache_gb = -1;
  lo.expert_store = "none";
  auto model = NativeModel::load(o.model, lo);
  const ModelInfo& I = model->info();
  std::fprintf(stderr, "[calib] %s: %d layers, hidden %d, %zu tokens -> %lld windows of %d (load %.0f ms)\n",
               o.model.c_str(), I.layers, I.hidden, ids.size(), (long long)nwin, o.ctx, I.load_ms);
  Sink sink(cfg, o);
  model->set_activation_sink(&sink);
  std::vector<float> hid(static_cast<size_t>(o.ctx) * size_t(I.hidden));
  std::vector<int32_t> win(static_cast<size_t>(o.ctx));
  for (int64_t w = 0; w < nwin; ++w) {
    TreeSpec t;
    if (bos) {
      win[0] = bos_id;
      std::copy_n(ids.data() + size_t(w) * size_t(body), body, win.data() + 1);
      t.root = win.data();
    } else {
      t.root = ids.data() + size_t(w) * size_t(o.ctx);
    }
    t.root_len = o.ctx;
    t.score_root = true;
    t.out_hidden = hid.data();
    model->run({t});
    sink.lm(hid.data(), o.ctx, I.hidden);
    std::fprintf(stderr, "\r[calib] window %lld/%lld  %.0f tok/s", (long long)(w + 1), (long long)nwin,
                 double((w + 1) * o.ctx) / (native::ms_since(t0) / 1000.0));
  }
  std::fprintf(stderr, "\n");
  model->set_activation_sink(nullptr);
  model.reset();
  sink.save(o.out, nwin * o.ctx);
}

}  // namespace espetero::quant
