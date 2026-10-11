// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Francisco José García Barbero
//
// Derived from joint_schema_model.py of Cloudflare's Clef release
// (https://huggingface.co/Cloudflare/clef-flash), licensed under the Apache
// License, Version 2.0; translated to C++ and modified (see THIRD_PARTY_NOTICES.md).
// This file is licensed under the Apache License, Version 2.0
// (third_party/apache-2.0/LICENSE-2.0.txt).

#include "clef.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

#include "../tok/tokenizer.h"
#include "engine.h"

namespace espetero::clef {

namespace {

// joint_schema_model.py, verbatim.
const char* const kSystemPrompt =
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options.";

[[noreturn]] void invalid(const std::string& msg) { throw RequestError(422, msg); }

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return "";
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

bool file_exists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// The release's render(): a string as it is, anything else as compact JSON
// with sorted keys and raw UTF-8 (json.dumps(..., ensure_ascii=False,
// separators=(",", ":"), sort_keys=True)).
std::string render(const json::Value& v) {
  if (v.is_string()) return v.as_string();
  json::DumpOptions o;
  o.sort_keys = true;
  return json::dump(v, o);
}

// question_options(): (option id, description or null) in the release's order.
std::vector<std::pair<std::string, json::Value>> options_of(const std::string& type, const json::Value* criteria) {
  std::vector<std::pair<std::string, json::Value>> out;
  if (type == "noul") {
    json::Value t("The proposition is true or the answer is yes."), f("The proposition is false or the answer is no.");
    if (criteria && criteria->is_object()) {
      if (const json::Value* v = criteria->get("true")) t = *v;
      if (const json::Value* v = criteria->get("false")) f = *v;
    }
    out.emplace_back("true", t);
    out.emplace_back("false", f);
  } else if (type == "choice") {
    const json::Object& o = criteria->as_object();
    for (size_t i = 0; i < o.size(); ++i) out.emplace_back(o.keys[i], o.vals[i]);
    // sorted((str(key), value) ...): keys are unique, so by key; UTF-8 byte
    // order equals Python's code point order.
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  } else {
    const auto& a = criteria->as_array();
    for (size_t i = 0; i < a.size(); ++i) out.emplace_back(std::to_string(i), a[i]);
  }
  return out;
}

// ---------------------------------------------------------------- safetensors (host)
struct SafeFile {
  struct Entry {
    std::string dtype;
    std::vector<int64_t> shape;
    size_t off = 0, nbytes = 0;
    int64_t numel() const {
      int64_t n = 1;
      for (int64_t d : shape) n *= d;
      return n;
    }
  };
  std::string path;
  const uint8_t* base = nullptr;
  size_t size = 0;
  std::map<std::string, Entry> entries;

  explicit SafeFile(const std::string& p) : path(p) {
    const int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("clef: cannot open " + p);
    struct stat st;
    ::fstat(fd, &st);
    size = size_t(st.st_size);
    void* m = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("clef: cannot map " + p);
    base = static_cast<const uint8_t*>(m);
    uint64_t hlen = 0;
    if (size < 8) throw std::runtime_error("clef: " + p + " is not a safetensors file");
    std::memcpy(&hlen, base, 8);
    if (hlen > size - 8) throw std::runtime_error("clef: " + p + ": truncated header");
    const json::Value h = json::parse(std::string_view(reinterpret_cast<const char*>(base + 8), size_t(hlen)));
    const json::Object& o = h.as_object();
    const size_t data_bytes = size - 8 - size_t(hlen);
    for (size_t i = 0; i < o.size(); ++i) {
      if (o.keys[i] == "__metadata__") continue;
      Entry e;
      e.dtype = o.vals[i].get_string("dtype", "");
      const json::Value* shape = o.vals[i].get("shape");
      const json::Value* offs = o.vals[i].get("data_offsets");
      if (!shape || !shape->is_array() || !offs || !offs->is_array() || offs->size() != 2)
        throw std::runtime_error("clef: " + p + ": tensor " + o.keys[i] + " has no shape or data_offsets");
      for (const json::Value& d : shape->as_array()) {
        if (d.as_int() < 0 || d.as_int() > (int64_t(1) << 40))
          throw std::runtime_error("clef: " + p + ": tensor " + o.keys[i] + " has a bad shape");
        e.shape.push_back(d.as_int());
      }
      const int64_t begin = (*offs)[0].as_int(), end = (*offs)[1].as_int();
      if (begin < 0 || end < begin || uint64_t(end) > data_bytes)
        throw std::runtime_error("clef: " + p + ": tensor past the end of the file");
      e.off = 8 + size_t(hlen) + size_t(begin);
      e.nbytes = size_t(end - begin);
      entries[o.keys[i]] = e;
    }
  }
  ~SafeFile() {
    if (base) ::munmap(const_cast<uint8_t*>(base), size);
  }
  const Entry& entry(const std::string& name) const {
    auto it = entries.find(name);
    if (it == entries.end()) throw std::runtime_error("clef: " + path + ": missing tensor " + name);
    return it->second;
  }
  const uint8_t* data(const Entry& e) const { return base + e.off; }
};

inline float bf16_f32(uint16_t h) {
  const uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline float dense_at(const std::string& dtype, const uint8_t* p, size_t i) {
  if (dtype == "BF16") return bf16_f32(reinterpret_cast<const uint16_t*>(p)[i]);
  if (dtype == "F16") return float(reinterpret_cast<const __fp16*>(p)[i]);
  return reinterpret_cast<const float*>(p)[i];
}

std::vector<float> dense(const SafeFile& f, const std::string& name, std::vector<int64_t> shape) {
  const SafeFile::Entry& e = f.entry(name);
  if (e.dtype != "BF16" && e.dtype != "F16" && e.dtype != "F32")
    throw std::runtime_error("clef: " + name + ": unsupported dtype " + e.dtype);
  if (e.shape != shape) {
    std::string s;
    for (int64_t d : e.shape) s += std::to_string(d) + " ";
    throw std::runtime_error("clef: " + name + ": unexpected shape [ " + s + "]");
  }
  const size_t esize = e.dtype == "F32" ? 4 : 2;
  if (uint64_t(e.numel()) > e.nbytes / esize)
    throw std::runtime_error("clef: " + name + ": its data is shorter than its shape");
  std::vector<float> out(size_t(e.numel()));
  const uint8_t* p = f.data(e);
  for (size_t i = 0; i < out.size(); ++i) out[i] = dense_at(e.dtype, p, i);
  return out;
}

// ---------------------------------------------------------------- math (fp32)
struct LN {
  std::vector<float> w, b;
};
struct Linear {
  std::vector<float> w, b;  // w [out, in] row-major; b empty when the layer has no bias
  int out = 0, in = 0;
};
struct MHA {
  Linear in, out;  // in: [3E, E] (q, k, v stacked)
};
struct Evidence {
  LN qn, mn, fn;
  MHA att;
  Linear f0, f3;
};
struct Decoder {
  LN n1, n2, n3;
  MHA sa, ca;
  Linear l1, l2;
};

// y[rows, out] = x[rows, in] W^T + b (x with leading dimension ldx).
void linear(const float* x, int rows, const Linear& L, float* y, int ldx = -1) {
  if (rows <= 0) return;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, L.out, L.in, 1.0f, x, ldx < 0 ? L.in : ldx, L.w.data(),
              L.in, 0.0f, y, L.out);
  if (!L.b.empty())
    for (int r = 0; r < rows; ++r) vDSP_vadd(y + size_t(r) * L.out, 1, L.b.data(), 1, y + size_t(r) * L.out, 1, vDSP_Length(L.out));
}

// torch.nn.LayerNorm (eps 1e-5), in place allowed.
void layernorm(const float* x, int rows, int d, const LN& n, float* y) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + size_t(r) * d;
    float* yr = y + size_t(r) * d;
    double mean = 0, var = 0;
    for (int i = 0; i < d; ++i) mean += xr[i];
    mean /= d;
    for (int i = 0; i < d; ++i) {
      const double t = xr[i] - mean;
      var += t * t;
    }
    var /= d;
    const float inv = float(1.0 / std::sqrt(var + 1e-5));
    const float m = float(mean);
    for (int i = 0; i < d; ++i) yr[i] = (xr[i] - m) * inv * n.w[size_t(i)] + n.b[size_t(i)];
  }
}

inline float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f)); }

void gelu_inplace(float* x, size_t n) {
  for (size_t i = 0; i < n; ++i) x[i] = gelu(x[i]);
}

void softmax_rows(float* s, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float* row = s + size_t(r) * cols;
    float mx = row[0];
    for (int c = 1; c < cols; ++c) mx = std::max(mx, row[c]);
    double sum = 0;
    for (int c = 0; c < cols; ++c) {
      row[c] = std::exp(row[c] - mx);
      sum += row[c];
    }
    const float inv = float(1.0 / sum);
    for (int c = 0; c < cols; ++c) row[c] *= inv;
  }
}

// torch.nn.MultiheadAttention(batch_first, no masks): out[nq, E].
// kv_proj (optional): precomputed [nk, 2E] keys and values of `kv`.
void mha(const MHA& a, const float* q, int nq, const float* kv, int nk, int E, int heads, float* out) {
  const int dh = E / heads;
  std::vector<float> Q(size_t(nq) * E), K(size_t(nk) * E), V(size_t(nk) * E), O(size_t(nq) * E);
  auto proj = [&](const float* x, int rows, int part, float* y) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, E, E, 1.0f, x, E, a.in.w.data() + size_t(part) * E * E,
                E, 0.0f, y, E);
    const float* b = a.in.b.data() + size_t(part) * E;
    for (int r = 0; r < rows; ++r) vDSP_vadd(y + size_t(r) * E, 1, b, 1, y + size_t(r) * E, 1, vDSP_Length(E));
  };
  proj(q, nq, 0, Q.data());
  proj(kv, nk, 1, K.data());
  proj(kv, nk, 2, V.data());
  std::vector<float> S(size_t(nq) * nk);
  const float scale = 1.0f / std::sqrt(float(dh));
  for (int h = 0; h < heads; ++h) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nq, nk, dh, scale, Q.data() + h * dh, E, K.data() + h * dh, E,
                0.0f, S.data(), nk);
    softmax_rows(S.data(), nq, nk);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nq, dh, nk, 1.0f, S.data(), nk, V.data() + h * dh, E, 0.0f,
                O.data() + h * dh, E);
  }
  linear(O.data(), nq, a.out, out);
}

void add_inplace(float* y, const float* x, size_t n) { vDSP_vadd(y, 1, x, 1, y, 1, vDSP_Length(n)); }

double dot(const float* a, const float* b, int n) {
  double s = 0;
  for (int i = 0; i < n; ++i) s += double(a[i]) * double(b[i]);
  return s;
}

}  // namespace

// ---------------------------------------------------------------- encoding
bool is_release(const std::string& dir) {
  return file_exists(dir + "/joint_head_config.json") && file_exists(dir + "/joint_head.safetensors");
}

Record encode(const Tokenizer& tok, const json::Value& state, const json::Value& questions,
              const std::vector<int>& image_tokens, int32_t image_token, int max_length, bool has_max_state,
              int max_state_tokens) {
  if (!questions.is_object() || questions.as_object().size() == 0) invalid("at least one question is required");
  Record r;
  std::vector<int32_t> schema = tok.encode("\n\nSCHEMA FIELDS:\n");
  const json::Object& qo = questions.as_object();
  for (size_t qi = 0; qi < qo.size(); ++qi) {
    const std::string& qid = qo.keys[qi];
    const json::Value& q = qo.vals[qi];
    if (!q.is_object()) invalid(qid + ": a question must be an object");
    const json::Value* tv = q.get("type");
    const std::string type = tv && tv->is_string() ? tv->as_string() : "";
    if (type != "noul" && type != "choice" && type != "score")
      invalid(qid + ": type must be noul, choice, or score");
    const json::Value* crit = q.get("criteria");
    const bool empty = !crit || crit->is_null() || (crit->is_object() && crit->as_object().size() == 0) ||
                       (crit->is_array() && crit->size() == 0) || (crit->is_string() && crit->as_string().empty()) ||
                       (crit->is_bool() && !crit->as_bool());
    if (type != "noul" && empty) invalid(qid + ": criteria must not be empty");
    if (type == "choice" && !crit->is_object()) invalid(qid + ": choice criteria must map option ids to descriptions");
    if (type == "score" && !crit->is_array()) invalid(qid + ": score criteria must be a list of level descriptions");
    if (type == "noul" && !empty && !crit->is_object()) invalid(qid + ": noul criteria must map true/false to descriptions");
    Question Q;
    Q.id = qid;
    Q.type = type;
    Q.type_id = type == "noul" ? 0 : (type == "choice" ? 1 : 2);
    Q.criteria = crit ? *crit : json::Value();
    tok.encode_append("\nFIELD " + std::to_string(qi + 1) + "\nID: " + qid + "\nTYPE: " + type + "\nINSTRUCTION: ", schema);
    Q.q0 = int(schema.size());
    const json::Value* ins = q.get("instructions");
    const bool no_ins = !ins || ins->is_null() || (ins->is_string() && ins->as_string().empty());
    tok.encode_append(no_ins ? qid : render(*ins), schema);
    Q.q1 = int(schema.size());
    tok.encode_append("\nALLOWED OPTIONS:\n", schema);
    const auto opts = options_of(type, crit);
    for (size_t oi = 0; oi < opts.size(); ++oi) {
      tok.encode_append("OPTION " + std::to_string(oi + 1) + ": ", schema);
      const int s0 = int(schema.size());
      json::Value sem = json::Value::object();
      sem.set("option_id", opts[oi].first);
      if (!opts[oi].second.is_null()) sem.set("description", opts[oi].second);
      tok.encode_append(render(sem), schema);
      Q.spans.emplace_back(s0, int(schema.size()));
      Q.option_ids.push_back(opts[oi].first);
      tok.encode_append("\n", schema);
    }
    tok.encode_append("END FIELD\n", schema);
    if (Q.q1 <= Q.q0) invalid(qid + ": the instructions encode to no tokens");
    r.questions.push_back(std::move(Q));
  }
  std::vector<int32_t> prefix =
      tok.encode(std::string("<|im_start|>system\n") + kSystemPrompt + "<|im_end|>\n<|im_start|>user\nSTATE:\n");
  const std::vector<int32_t> suffix =
      tok.encode("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");
  if (!image_tokens.empty()) {
    std::string media;
    for (size_t i = 0; i < image_tokens.size(); ++i) media += "<|vision_start|><|image_pad|><|vision_end|>";
    media += "\n";
    size_t g = 0;
    for (int32_t t : tok.encode(media)) {
      if (t == image_token) {
        if (g >= image_tokens.size()) invalid("image placeholders do not match the images");
        prefix.insert(prefix.end(), size_t(image_tokens[g++]), t);
      } else {
        prefix.push_back(t);
      }
    }
    if (g != image_tokens.size()) invalid("image placeholders do not match the images");
  }
  std::vector<int32_t> st = tok.encode(render(state));
  r.state_tokens = int(st.size());
  if (has_max_state) {  // Python's state_ids[:n]: n >= 0 keeps the first n, n < 0 drops the last -n
    const long n = long(st.size()), k = max_state_tokens >= 0 ? std::min<long>(max_state_tokens, n)
                                                              : std::max<long>(0, n + max_state_tokens);
    st.resize(size_t(k));
  }
  const size_t fixed = prefix.size() + schema.size() + suffix.size();
  if (int(fixed) > max_length)
    invalid("schema requires " + std::to_string(fixed) + " tokens before state; maximum is " + std::to_string(max_length));
  if (st.size() > size_t(max_length) - fixed) st.resize(size_t(max_length) - fixed);
  r.truncated_state_tokens = r.state_tokens - int(st.size());
  const int off = int(prefix.size() + st.size());
  for (Question& Q : r.questions) {
    Q.q0 += off;
    Q.q1 += off;
    for (auto& sp : Q.spans) sp.first += off, sp.second += off;
  }
  r.prefix_tokens = int(prefix.size());
  r.ids = std::move(prefix);
  r.ids.insert(r.ids.end(), st.begin(), st.end());
  r.ids.insert(r.ids.end(), schema.begin(), schema.end());
  r.ids.insert(r.ids.end(), suffix.begin(), suffix.end());
  return r;
}

json::Value answer(const Question& q, const std::vector<double>& probs) {
  auto r4 = [](double x) { return std::round(x * 1e4) / 1e4; };
  auto p_of = [&](const std::string& id) {
    for (size_t i = 0; i < q.option_ids.size(); ++i)
      if (q.option_ids[i] == id) return probs[i];
    return 0.0;
  };
  json::Value a = json::Value::object();
  a.set("type", q.type);
  if (q.type == "noul") {
    a.set("noul", r4(p_of("true")));
    return a;
  }
  if (q.type == "choice") {
    const json::Object& o = q.criteria.as_object();
    std::string best;
    double bp = -1;
    json::Value pr = json::Value::object();
    for (const std::string& k : o.keys) {  // the request's order; the first maximum wins
      const double p = p_of(k);
      if (p > bp) bp = p, best = k;
      pr.set(k, r4(p));
    }
    a.set("choice", best);
    a.set("confidence", r4(bp));
    a.set("probabilities", std::move(pr));
    return a;
  }
  const auto& levels = q.criteria.as_array();
  double score = 0, conf = 0;
  json::Value legend = json::Value::object(), pr = json::Value::object();
  for (size_t i = 0; i < levels.size(); ++i) {
    const double p = p_of(std::to_string(i));
    score += double(i) * p;
    conf = std::max(conf, p);
    legend.set(std::to_string(i), levels[i]);
    pr.set(std::to_string(i), r4(p));
  }
  a.set("score", r4(score));
  a.set("confidence", r4(conf));
  a.set("legend", std::move(legend));
  a.set("probabilities", std::move(pr));
  return a;
}

// ---------------------------------------------------------------- head
struct Head::Impl {
  int Dh = 0, W = 0, heads = 0, ff = 0;
  LN hidden_norm, option_summary_norm, field_norm, option_norm;
  Linear memory_projection, question_projection, option_question_projection, global_projection,
      option_context_projection, option_lexical_projection;
  std::vector<float> type_embedding;  // [3, W]
  std::vector<Evidence> ev;
  std::vector<Decoder> dec;
  Linear rs0, rs3;
  float prior_logit_scale = 0, joint_logit_scale = 0, residual_gate = 0;
  // LM head rows (the options' lexical vectors).
  std::unique_ptr<SafeFile> lm, lm_s, lm_b;  // codes (or dense rows); scales and biases when in other shards
  SafeFile::Entry lw, ls, lb;
  const uint8_t* ls_data = nullptr;
  const uint8_t* lb_data = nullptr;
  int lm_bits = 0, lm_group = 0, V = 0;

  void lm_row(int32_t id, float* out) const {
    if (id < 0 || id >= V) throw RequestError(422, "token id outside the vocabulary");
    if (lm_bits == 0) {
      const uint8_t* p = lm->data(lw);
      for (int c = 0; c < Dh; ++c) out[c] = dense_at(lw.dtype, p, size_t(id) * size_t(Dh) + size_t(c));
      return;
    }
    const uint32_t* w = reinterpret_cast<const uint32_t*>(lm->data(lw)) + size_t(id) * size_t(Dh) * lm_bits / 32;
    const int gpr = Dh / lm_group;
    const uint8_t* sp = ls_data;
    const uint8_t* bp = lb_data;
    const uint32_t mask = (1u << lm_bits) - 1u;
    for (int c = 0; c < Dh; ++c) {
      const uint32_t code = (w[size_t(c) * lm_bits / 32] >> ((size_t(c) * lm_bits) % 32)) & mask;
      const size_t gi = size_t(id) * size_t(gpr) + size_t(c / lm_group);
      out[c] = float(code) * dense_at(ls.dtype, sp, gi) + dense_at(lb.dtype, bp, gi);
    }
  }
};

Head::Head(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Head::~Head() = default;
int Head::hidden() const { return impl_->Dh; }

std::unique_ptr<Head> Head::load(const std::string& dir) {
  auto I = std::make_unique<Impl>();
  const json::Value cfg = json::parse(read_file(dir + "/joint_head_config.json"));
  I->Dh = int(cfg.get_int("hidden_size", 0));
  I->W = int(cfg.get_int("width", 0));
  I->heads = int(cfg.get_int("heads", 0));
  I->ff = int(cfg.get_int("feedforward", 0));
  const int n_route = int(cfg.get_int("routing_layers", 0)), n_layers = int(cfg.get_int("layers", 0));
  const int D = I->Dh, W = I->W, F = I->ff;
  if (D <= 0 || W <= 0 || I->heads <= 0 || W % I->heads || F <= 0)
    throw std::runtime_error("clef: invalid joint_head_config.json");
  SafeFile f(dir + "/joint_head.safetensors");
  auto ln = [&](const std::string& p, int d) { return LN{dense(f, p + ".weight", {d}), dense(f, p + ".bias", {d})}; };
  auto lin = [&](const std::string& p, int out, int in, bool bias) {
    Linear L;
    L.out = out;
    L.in = in;
    L.w = dense(f, p + ".weight", {out, in});
    if (bias) L.b = dense(f, p + ".bias", {out});
    return L;
  };
  auto mh = [&](const std::string& p) {
    MHA a;
    a.in.out = 3 * W;
    a.in.in = W;
    a.in.w = dense(f, p + ".in_proj_weight", {3 * W, W});
    a.in.b = dense(f, p + ".in_proj_bias", {3 * W});
    a.out = lin(p + ".out_proj", W, W, true);
    return a;
  };
  I->hidden_norm = ln("hidden_norm", D);
  I->memory_projection = lin("memory_projection", W, D, false);
  I->question_projection = lin("question_projection", W, D, false);
  I->option_question_projection = lin("option_question_projection", W, D, false);
  I->global_projection = lin("global_projection", W, D, false);
  I->option_context_projection = lin("option_context_projection", W, D, false);
  I->option_lexical_projection = lin("option_lexical_projection", W, D, false);
  I->type_embedding = dense(f, "type_embedding.weight", {3, W});
  for (int i = 0; i < n_route; ++i) {
    const std::string p = "evidence_layers." + std::to_string(i) + ".";
    Evidence e;
    e.qn = ln(p + "query_norm", W);
    e.mn = ln(p + "memory_norm", W);
    e.fn = ln(p + "feedforward_norm", W);
    e.att = mh(p + "attention");
    e.f0 = lin(p + "feedforward.0", F, W, true);
    e.f3 = lin(p + "feedforward.3", W, F, true);
    I->ev.push_back(std::move(e));
  }
  I->option_summary_norm = ln("option_summary_norm", W);
  for (int i = 0; i < n_layers; ++i) {
    const std::string p = "layers." + std::to_string(i) + ".";
    Decoder d;
    d.n1 = ln(p + "norm1", W);
    d.n2 = ln(p + "norm2", W);
    d.n3 = ln(p + "norm3", W);
    d.sa = mh(p + "self_attn");
    d.ca = mh(p + "multihead_attn");
    d.l1 = lin(p + "linear1", F, W, true);
    d.l2 = lin(p + "linear2", W, F, true);
    I->dec.push_back(std::move(d));
  }
  I->field_norm = ln("field_norm", W);
  I->option_norm = ln("option_norm", W);
  I->rs0 = lin("residual_scorer.0", W, 4 * W, true);
  I->rs3 = lin("residual_scorer.3", 1, W, true);
  auto scalar = [&](const std::string& n) {  // a 0-d tensor (the release) or a 1-element one
    return dense(f, n, f.entry(n).shape.empty() ? std::vector<int64_t>{} : std::vector<int64_t>{1})[0];
  };
  I->prior_logit_scale = scalar("prior_logit_scale");
  I->joint_logit_scale = scalar("joint_logit_scale");
  I->residual_gate = scalar("residual_gate");

  // LM head rows: the backbone's own output embedding.
  std::string file;
  std::string name;
  json::Value wm;
  if (const std::string idx = read_file(dir + "/model.safetensors.index.json"); !idx.empty()) {
    wm = *json::parse(idx).get("weight_map");
    // Tied checkpoints have no LM head: their output embedding is the input one.
    for (const char* n : {"language_model.lm_head.weight", "lm_head.weight", "language_model.model.embed_tokens.weight"})
      if (const json::Value* v = wm.get(n)) {
        name = n;
        file = dir + "/" + v->as_string();
        break;
      }
  } else if (file_exists(dir + "/model.safetensors")) {
    file = dir + "/model.safetensors";
    name = "language_model.lm_head.weight";
  }
  if (file.empty()) throw std::runtime_error("clef: no LM head (language_model.lm_head.weight) in " + dir);
  I->lm = std::make_unique<SafeFile>(file);
  if (!I->lm->entries.count(name)) name = "lm_head.weight";
  I->lw = I->lm->entry(name);
  if (I->lw.dtype == "U32") {
    const std::string base = name.substr(0, name.size() - 7);
    // Scales and biases may sit in other shards than the codes (MLX shards by size).
    auto part = [&](const std::string& n, std::unique_ptr<SafeFile>& own) -> const SafeFile& {
      if (I->lm->entries.count(n)) return *I->lm;
      const json::Value* f = wm.is_object() ? wm.get(n) : nullptr;
      if (!f || !f->is_string()) throw std::runtime_error("clef: missing tensor " + n);
      own = std::make_unique<SafeFile>(dir + "/" + f->as_string());
      return *own;
    };
    const SafeFile& fs = part(base + ".scales", I->lm_s);
    const SafeFile& fb = part(base + ".biases", I->lm_b);
    I->ls = fs.entry(base + ".scales");
    I->lb = fb.entry(base + ".biases");
    I->ls_data = fs.data(I->ls);
    I->lb_data = fb.data(I->lb);
    const json::Value mc = json::parse(read_file(dir + "/config.json"));
    const json::Value* q = mc.get("quantization");
    if (!q || !q->is_object()) q = mc.get("quantization_config");
    I->lm_bits = q ? int(q->get_int("bits", 4)) : 4;
    I->lm_group = q ? int(q->get_int("group_size", 64)) : 64;
    if (q)
      if (const json::Value* ov = q->get(base); ov && ov->is_object()) {
        I->lm_bits = int(ov->get_int("bits", I->lm_bits));
        I->lm_group = int(ov->get_int("group_size", I->lm_group));
      }
    if (I->lm_bits != 4 && I->lm_bits != 8) throw std::runtime_error("clef: LM head rows need 4- or 8-bit (or dense) storage");
    I->V = int(I->lw.shape[0]);
    if (I->lw.shape[1] * 32 / I->lm_bits != D) throw std::runtime_error("clef: LM head width differs from the head's hidden size");
  } else {
    if (I->lw.shape.size() != 2 || I->lw.shape[1] != D) throw std::runtime_error("clef: LM head width differs from the head's hidden size");
    I->V = int(I->lw.shape[0]);
  }
  return std::unique_ptr<Head>(new Head(std::move(I)));
}

std::vector<std::vector<float>> Head::forward(const float* hidden, const Record& r) const {
  const Impl& H = *impl_;
  const int D = H.Dh, W = H.W, F = H.ff, NH = H.heads;
  const int L = int(r.ids.size());
  const int NQ = int(r.questions.size());
  int NO = 0;
  for (const Question& q : r.questions) NO += int(q.spans.size());

  // Normalized hidden states, memory, global vector.
  std::vector<float> hn(size_t(L) * D);
  layernorm(hidden, L, D, H.hidden_norm, hn.data());
  std::vector<float> mem(size_t(L) * W);
  linear(hn.data(), L, H.memory_projection, mem.data());
  const float* global = hn.data() + size_t(L - 1) * D;

  auto mean_rows = [&](const float* x, int a, int b, int d, float* out) {
    std::fill(out, out + d, 0.0f);
    for (int i = a; i < b; ++i) vDSP_vadd(out, 1, x + size_t(i) * d, 1, out, 1, vDSP_Length(d));
    const float inv = b > a ? 1.0f / float(b - a) : NAN;
    vDSP_vsmul(out, 1, &inv, out, 1, vDSP_Length(d));
  };
  std::vector<float> qv(size_t(NQ) * D), oc(size_t(NO) * D), lx(size_t(NO) * D), row(static_cast<size_t>(D));
  std::vector<int> owner(static_cast<size_t>(NO));
  {
    int j = 0;
    for (int q = 0; q < NQ; ++q) {
      const Question& Q = r.questions[size_t(q)];
      mean_rows(hn.data(), Q.q0, Q.q1, D, qv.data() + size_t(q) * D);
      for (const auto& [a, b] : Q.spans) {
        mean_rows(hn.data(), a, b, D, oc.data() + size_t(j) * D);
        float* l = lx.data() + size_t(j) * D;
        std::fill(l, l + D, 0.0f);
        for (int i = a; i < b; ++i) {
          H.lm_row(r.ids[size_t(i)], row.data());
          vDSP_vadd(l, 1, row.data(), 1, l, 1, vDSP_Length(D));
        }
        const float inv = b > a ? 1.0f / float(b - a) : NAN;
        vDSP_vsmul(l, 1, &inv, l, 1, vDSP_Length(D));
        owner[size_t(j)] = q;
        ++j;
      }
    }
  }

  // Option queries: context + lexical + their question.
  std::vector<float> opt(size_t(NO) * W), tmp(size_t(NO) * W), oq(size_t(NQ) * W);
  linear(oc.data(), NO, H.option_context_projection, opt.data());
  linear(lx.data(), NO, H.option_lexical_projection, tmp.data());
  add_inplace(opt.data(), tmp.data(), opt.size());
  linear(qv.data(), NQ, H.option_question_projection, oq.data());
  for (int j = 0; j < NO; ++j) add_inplace(opt.data() + size_t(j) * W, oq.data() + size_t(owner[size_t(j)]) * W, size_t(W));

  // Evidence routing: the options read the whole sequence.
  {
    std::vector<float> qn(size_t(NO) * W), mn(size_t(L) * W), att(size_t(NO) * W), h1(size_t(NO) * F);
    for (const Evidence& e : H.ev) {
      layernorm(opt.data(), NO, W, e.qn, qn.data());
      layernorm(mem.data(), L, W, e.mn, mn.data());
      mha(e.att, qn.data(), NO, mn.data(), L, W, NH, att.data());
      add_inplace(opt.data(), att.data(), opt.size());
      layernorm(opt.data(), NO, W, e.fn, qn.data());
      linear(qn.data(), NO, e.f0, h1.data());
      gelu_inplace(h1.data(), h1.size());
      linear(h1.data(), NO, e.f3, att.data());
      add_inplace(opt.data(), att.data(), opt.size());
    }
  }

  // Fields: question + routed option summary + global + type.
  std::vector<float> fields(size_t(NQ) * W), summ(size_t(NQ) * W, 0.0f), gproj(static_cast<size_t>(W));
  linear(qv.data(), NQ, H.question_projection, fields.data());
  {
    int j0 = 0;
    for (int q = 0; q < NQ; ++q) {
      const int n = int(r.questions[size_t(q)].spans.size());
      const float* fq = fields.data() + size_t(q) * W;
      std::vector<float> w(static_cast<size_t>(n));
      for (int k = 0; k < n; ++k) w[size_t(k)] = float(dot(opt.data() + size_t(j0 + k) * W, fq, W) / std::sqrt(double(W)));
      softmax_rows(w.data(), 1, n);
      float* s = summ.data() + size_t(q) * W;
      for (int k = 0; k < n; ++k) {
        const float wk = w[size_t(k)];
        cblas_saxpy(W, wk, opt.data() + size_t(j0 + k) * W, 1, s, 1);
      }
      j0 += n;
    }
  }
  layernorm(summ.data(), NQ, W, H.option_summary_norm, summ.data());
  linear(global, 1, H.global_projection, gproj.data());
  for (int q = 0; q < NQ; ++q) {
    float* fq = fields.data() + size_t(q) * W;
    add_inplace(fq, summ.data() + size_t(q) * W, size_t(W));
    add_inplace(fq, gproj.data(), size_t(W));
    add_inplace(fq, H.type_embedding.data() + size_t(r.questions[size_t(q)].type_id) * W, size_t(W));
  }

  // Decoder layers (norm first): self-attention among fields, cross-attention to the memory, MLP.
  {
    std::vector<float> xn(size_t(NQ) * W), att(size_t(NQ) * W), h1(size_t(NQ) * F);
    for (const Decoder& d : H.dec) {
      layernorm(fields.data(), NQ, W, d.n1, xn.data());
      mha(d.sa, xn.data(), NQ, xn.data(), NQ, W, NH, att.data());
      add_inplace(fields.data(), att.data(), fields.size());
      layernorm(fields.data(), NQ, W, d.n2, xn.data());
      mha(d.ca, xn.data(), NQ, mem.data(), L, W, NH, att.data());
      add_inplace(fields.data(), att.data(), fields.size());
      layernorm(fields.data(), NQ, W, d.n3, xn.data());
      linear(xn.data(), NQ, d.l1, h1.data());
      gelu_inplace(h1.data(), h1.size());
      linear(h1.data(), NQ, d.l2, att.data());
      add_inplace(fields.data(), att.data(), fields.size());
    }
  }
  layernorm(fields.data(), NQ, W, H.field_norm, fields.data());

  // Scores: lexical prior + gated (scaled cosine + residual scorer).
  std::vector<float> on(size_t(NO) * W), feat(size_t(NO) * 4 * W), hid(size_t(NO) * W), res(static_cast<size_t>(NO));
  layernorm(opt.data(), NO, W, H.option_norm, on.data());
  for (int j = 0; j < NO; ++j) {
    const float* f = fields.data() + size_t(owner[size_t(j)]) * W;
    const float* o = on.data() + size_t(j) * W;
    float* x = feat.data() + size_t(j) * 4 * W;
    for (int c = 0; c < W; ++c) {
      x[c] = f[c];
      x[W + c] = o[c];
      x[2 * W + c] = f[c] * o[c];
      x[3 * W + c] = std::fabs(f[c] - o[c]);
    }
  }
  linear(feat.data(), NO, H.rs0, hid.data());
  gelu_inplace(hid.data(), hid.size());
  linear(hid.data(), NO, H.rs3, res.data());
  const double cap = std::log(100.0);
  const double prior_scale = std::exp(std::min(double(H.prior_logit_scale), cap));
  const double joint_scale = std::exp(std::min(double(H.joint_logit_scale), cap));
  const double gate = 1.0 / (1.0 + std::exp(-double(H.residual_gate)));
  std::vector<std::vector<float>> out(static_cast<size_t>(NQ));
  std::vector<float> anchor(static_cast<size_t>(D));
  int j = 0;
  for (int q = 0; q < NQ; ++q) {
    const float* qq = qv.data() + size_t(q) * D;
    for (int c = 0; c < D; ++c) anchor[size_t(c)] = qq[c] + global[c];
    const double an = std::max(std::sqrt(dot(anchor.data(), anchor.data(), D)), 1e-12);
    const float* f = fields.data() + size_t(q) * W;
    const double fnorm = std::sqrt(dot(f, f, W));
    for (size_t k = 0; k < r.questions[size_t(q)].spans.size(); ++k, ++j) {
      const float* l = lx.data() + size_t(j) * D;
      const double ln = std::max(std::sqrt(dot(l, l, D)), 1e-12);
      const double prior = prior_scale * dot(l, anchor.data(), D) / (ln * an);
      const float* o = on.data() + size_t(j) * W;
      const double cosv = dot(f, o, W) / std::max(fnorm * std::sqrt(dot(o, o, W)), 1e-8);
      out[size_t(q)].push_back(float(prior + gate * (joint_scale * cosv + double(res[size_t(j)]))));
    }
  }
  return out;
}

}  // namespace espetero::clef
