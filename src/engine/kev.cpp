// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "kev.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <sys/stat.h>

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "../tok/tokenizer.h"
#include "engine.h"

namespace espetero::kev {

namespace {

[[noreturn]] void invalid(const std::string& msg) { throw RequestError(422, msg); }

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Python's str.isspace() for one code point (what str.lstrip() strips).
bool py_space(uint32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

std::string py_lstrip(const std::string& s) {
  size_t i = 0;
  while (i < s.size()) {
    size_t j = i;
    if (!py_space(json::utf8_next(s, j))) break;
    i = j;
  }
  return s.substr(i);
}

// f"{n:,}"
std::string with_commas(int64_t n) {
  std::string d = std::to_string(n < 0 ? -n : n), out;
  for (size_t i = 0; i < d.size(); ++i) {
    if (i && (d.size() - i) % 3 == 0) out.push_back(',');
    out.push_back(d[i]);
  }
  return n < 0 ? "-" + out : out;
}

// api.option_text: the name alone without a description, else "name: description".
std::string option_text(const std::string& name, const json::Value* desc) {
  if (!desc || desc->is_null() || (desc->is_string() && desc->as_string().empty())) return name;
  return name + ": " + render(*desc);
}

// The F32 tensor `name` of a safetensors file, checked against `shape`.
std::vector<float> f32_tensor(const std::string& file, const std::string& bytes, const json::Value& header,
                              const std::string& name, const std::vector<int64_t>& shape) {
  const json::Value* t = header.get(name);
  if (!t || !t->is_object()) throw std::runtime_error(file + ": no tensor " + name);
  if (t->get_string("dtype", "") != "F32") throw std::runtime_error(file + ": " + name + " is not F32");
  const json::Value* sh = t->get("shape");
  const json::Value* off = t->get("data_offsets");
  if (!sh || !sh->is_array() || !off || !off->is_array() || off->size() != 2)
    throw std::runtime_error(file + ": " + name + ": malformed header entry");
  std::vector<int64_t> got;
  for (const json::Value& d : sh->as_array()) got.push_back(d.as_int());
  if (got != shape) throw std::runtime_error(file + ": " + name + " has an unexpected shape");
  size_t n = 1;
  for (int64_t d : shape) n *= size_t(d);
  uint64_t hl = 0;
  std::memcpy(&hl, bytes.data(), 8);
  const uint64_t base = 8 + hl, a = uint64_t((*off)[0].as_int()), b = uint64_t((*off)[1].as_int());
  if (b < a || b - a != n * 4 || base + b > bytes.size()) throw std::runtime_error(file + ": " + name + ": data out of range");
  std::vector<float> v(n);
  std::memcpy(v.data(), bytes.data() + base + a, n * 4);
  return v;
}

std::vector<float> load_safetensor_f32(const std::string& file, const std::string& name,
                                       const std::vector<int64_t>& shape) {
  std::ifstream f(file, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + file);
  uint64_t hl = 0;
  if (!f.read(reinterpret_cast<char*>(&hl), 8) || hl > (256ull << 20))
    throw std::runtime_error(file + ": truncated header");
  std::string hbuf(size_t(hl), '\0');
  if (!f.read(hbuf.data(), std::streamsize(hl))) throw std::runtime_error(file + ": truncated header");
  const json::Value header = json::parse(hbuf);
  const json::Value* t = header.get(name);
  if (!t || !t->is_object()) throw std::runtime_error(file + ": no tensor " + name);
  const std::string dt = t->get_string("dtype", "");
  if (dt != "F32" && dt != "BF16" && dt != "F16")
    throw std::runtime_error(file + ": " + name + " has unsupported dtype " + dt);
  const json::Value* sh = t->get("shape");
  const json::Value* off = t->get("data_offsets");
  if (!sh || !sh->is_array() || !off || !off->is_array() || off->size() != 2)
    throw std::runtime_error(file + ": " + name + ": malformed header entry");
  std::vector<int64_t> got;
  for (const json::Value& d : sh->as_array()) got.push_back(d.as_int());
  if (got != shape) throw std::runtime_error(file + ": " + name + " has an unexpected shape");
  size_t n = 1;
  for (int64_t d : shape) n *= size_t(d);
  const size_t elem = dt == "F32" ? 4 : 2;
  const uint64_t base = 8 + hl, a = uint64_t((*off)[0].as_int()), b = uint64_t((*off)[1].as_int());
  if (b < a || b - a != n * elem) throw std::runtime_error(file + ": " + name + ": data out of range");
  std::string raw(n * elem, '\0');
  f.seekg(std::streamoff(base + a));
  if (!f.read(raw.data(), std::streamsize(raw.size())))
    throw std::runtime_error(file + ": " + name + ": data out of range");
  std::vector<float> v(n);
  if (dt == "F32") {
    std::memcpy(v.data(), raw.data(), n * 4);
  } else if (dt == "BF16") {
    for (size_t i = 0; i < n; ++i) {
      uint16_t u16 = 0;
      std::memcpy(&u16, raw.data() + i * 2, 2);
      uint32_t u32 = uint32_t(u16) << 16;
      std::memcpy(&v[i], &u32, 4);
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      __fp16 h16 = 0;
      std::memcpy(&h16, raw.data() + i * 2, 2);
      v[i] = float(h16);
    }
  }
  return v;
}

}  // namespace

bool is_release(const std::string& dir) {
  struct stat st {};
  if (::stat((dir + "/pointer_head.json").c_str(), &st) == 0 && S_ISREG(st.st_mode)) return true;
  if (::stat((dir + "/config.json").c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
    try {
      return json::parse(slurp(dir + "/config.json")).get_string("model_type", "") == "videojev";
    } catch (...) {
    }
  }
  return false;
}

std::string render(const json::Value& v, int indent) {
  const std::string pad(size_t(2 * indent), ' ');
  switch (v.type()) {
    case json::Type::Null:
      return "";
    case json::Type::Bool:
      return v.as_bool() ? "True" : "False";  // str(bool)
    case json::Type::Int:
      return std::to_string(v.as_int());
    case json::Type::Double:
      return json::format_double(v.as_double());  // str(float) == repr(float)
    case json::Type::String:
      return v.as_string();
    case json::Type::Array: {
      std::string out;
      const auto& a = v.as_array();
      for (size_t i = 0; i < a.size(); ++i) {
        if (i) out.push_back('\n');
        out += pad + "- " + py_lstrip(render(a[i], indent + 1));
      }
      return out;
    }
    case json::Type::Object: {
      std::string out;
      const json::Object& o = v.as_object();
      for (size_t i = 0; i < o.size(); ++i) {
        if (i) out.push_back('\n');
        const json::Value& x = o.vals[i];
        if (x.is_object() || x.is_array()) out += pad + o.keys[i] + ":\n" + render(x, indent + 1);
        else out += pad + o.keys[i] + ": " + render(x);
      }
      return out;
    }
  }
  return "";
}

std::vector<int32_t> user_tokens(const Tokenizer& tok, std::string_view s) {
  // re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", text), then the tokenizer's own split of added tokens.
  std::string t;
  t.reserve(s.size());
  const size_t n = s.size();
  for (size_t i = 0; i < n;) {
    if (s[i] == '<' && i + 1 < n && s[i + 1] == '|') {
      size_t j = i + 2;
      while (j < n && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_')) ++j;
      if (j > i + 2 && j + 1 < n && s[j] == '|' && s[j + 1] == '>') {
        t += "<\xC2\xA6";
        t.append(s.substr(i + 2, j - i - 2));
        t += "\xC2\xA6>";
        i = j + 2;
        continue;
      }
    }
    t.push_back(s[i++]);
  }
  return tok.encode(t, true);
}

Record encode(const Tokenizer& tok, const Delimiters& d, const json::Value& state, const json::Value& questions,
              const std::vector<int>& image_tokens, int32_t vision_start, int32_t image_pad, int32_t vision_end) {
  // The request models (api.Noul / Choice / Score, SystemOneRequest): what pydantic refuses with a 422.
  if (!questions.is_object() || questions.size() == 0) invalid("questions: an object with at least one question");
  struct Spec {
    const json::Value* instructions;
    const json::Value* criteria;
  };
  std::vector<Spec> specs;
  Record r;
  const json::Object& qs = questions.as_object();
  for (size_t i = 0; i < qs.size(); ++i) {
    const std::string& id = qs.keys[i];
    const json::Value& q = qs.vals[i];
    if (!q.is_object()) invalid("questions." + id + ": an object with type, instructions and criteria");
    const std::string type = q.get_string("type", "");
    const json::Value* c = q.get("criteria");
    if (c && c->is_null()) c = nullptr;
    if (type == "noul") {
      if (c && !c->is_object()) invalid("questions." + id + ".criteria: an object with optional true / false descriptions, or null");
    } else if (type == "choice") {
      if (!c || !c->is_object() || c->size() < 1 || c->size() > 255)
        invalid("questions." + id + ".criteria: a choice needs an object of 1 to 255 options");
    } else if (type == "score") {
      if (!c || !c->is_array() || c->size() < 1 || c->size() > 255)
        invalid("questions." + id + ".criteria: a score needs a list of 1 to 255 levels");
    } else {
      invalid("questions." + id + ".type: noul, choice or score");
    }
    Question Q;
    Q.id = id;
    Q.type = type;
    r.questions.push_back(std::move(Q));
    specs.push_back({q.get("instructions"), c});
  }

  // model.encode: [<state> state] then per question [<q> instr (<opt> option </opt>)... <decide>].
  const bool videojev = d.state < 0;
  if (videojev) {
    const std::vector<int32_t> pre = tok.encode("<|im_start|>user\n", true);
    r.state.insert(r.state.end(), pre.begin(), pre.end());
  } else {
    r.state.push_back(d.state);
  }
  const std::string s_text = render(state);
  bool interleaved = false;
  if (videojev && !image_tokens.empty()) {
    size_t count = 0, pos = 0;
    while ((pos = s_text.find("<image>", pos)) != std::string::npos) {
      ++count;
      pos += 7;
    }
    if (count == image_tokens.size()) {
      interleaved = true;
      size_t cur = 0;
      for (int n : image_tokens) {
        const size_t nxt = s_text.find("<image>", cur);
        if (nxt > cur) {
          const std::vector<int32_t> seg = user_tokens(tok, std::string_view(s_text).substr(cur, nxt - cur));
          r.state.insert(r.state.end(), seg.begin(), seg.end());
        }
        r.state.push_back(vision_start);
        r.state.insert(r.state.end(), size_t(n), image_pad);
        r.state.push_back(vision_end);
        cur = nxt + 7;
      }
      if (cur < s_text.size()) {
        const std::vector<int32_t> seg = user_tokens(tok, std::string_view(s_text).substr(cur));
        r.state.insert(r.state.end(), seg.begin(), seg.end());
      }
    }
  }
  if (!interleaved) {
    for (int n : image_tokens) {
      r.state.push_back(vision_start);
      r.state.insert(r.state.end(), size_t(n), image_pad);
      r.state.push_back(vision_end);
    }
    const std::vector<int32_t> text = user_tokens(tok, s_text);
    r.state.insert(r.state.end(), text.begin(), text.end());
  }
  const int Ls = int(r.state.size());
  if (Ls > kMaxState)
    invalid("state is " + with_commas(Ls) + " tokens, over the " + with_commas(kMaxState) + "-token limit (the <state> token" +
            (image_tokens.empty() ? "" : " and the images") + " included): shorten the document or split it across requests");
  r.input_tokens = Ls;
  for (size_t i = 0; i < r.questions.size(); ++i) {
    Question& Q = r.questions[i];
    const json::Value* c = specs[i].criteria;
    std::vector<std::string> opts;  // api.to_record's option texts
    if (Q.type == "noul") {
      Q.keys = {"false", "true"};
      opts.push_back(option_text("no", c ? c->get("false") : nullptr));
      opts.push_back(option_text("yes", c ? c->get("true") : nullptr));
    } else if (Q.type == "choice") {
      const json::Object& o = c->as_object();
      for (size_t k = 0; k < o.size(); ++k) {
        Q.keys.push_back(o.keys[k]);
        if (videojev) {
          const json::Value& v = o.vals[k];
          opts.push_back(v.is_null() || (v.is_string() && v.as_string().empty()) ? o.keys[k] : render(v));
        } else {
          opts.push_back(option_text(o.keys[k], &o.vals[k]));
        }
      }
    } else {
      const auto& a = c->as_array();
      for (size_t k = 0; k < a.size(); ++k) {
        Q.keys.push_back(std::to_string(k));
        opts.push_back(render(a[k]));
      }
      Q.legend = opts;
    }
    if (videojev) {
      const bool grounding = !opts.empty() && opts[0].size() > 10 && opts[0].front() == '<' &&
                             opts[0].rfind(" seconds>") == opts[0].size() - 9;
      const std::string task = grounding ? "temporal grounding" : "single-choice video question answering";
      const std::string qtxt = specs[i].instructions ? render(*specs[i].instructions) : Q.id;
      const std::vector<int32_t> hdr = tok.encode("\nTask: " + task + "\nQuestion: ", false);
      Q.ids.insert(Q.ids.end(), hdr.begin(), hdr.end());
      const std::vector<int32_t> instr = user_tokens(tok, qtxt);
      Q.ids.insert(Q.ids.end(), instr.begin(), instr.end());
      const std::vector<int32_t> ohdr = tok.encode("\nOptions:\n", false);
      Q.ids.insert(Q.ids.end(), ohdr.begin(), ohdr.end());
      const std::vector<int32_t> nl = tok.encode("\n", false);
      std::vector<int32_t> ends;
      for (const std::string& o : opts) {
        const std::vector<int32_t> t = user_tokens(tok, o);
        Q.ids.insert(Q.ids.end(), t.begin(), t.end());
        Q.ids.push_back(d.option_end);
        ends.push_back(int32_t(Q.ids.size()) - 1);
        Q.ids.insert(Q.ids.end(), nl.begin(), nl.end());
      }
      Q.ids.push_back(d.decide);
      if (int(Q.ids.size()) > kMaxRow - Ls)
        invalid("branch too long: " + std::to_string(Q.ids.size()) + " tokens with a " + std::to_string(Ls) +
                "-token state (row limit " + std::to_string(kMaxRow) + ")");
      Q.rows.push_back(int32_t(Q.ids.size()) - 1);
      Q.rows.insert(Q.rows.end(), ends.begin(), ends.end());
      r.input_tokens += int(Q.ids.size());
      continue;
    }
    Q.ids.push_back(d.question);
    const std::vector<int32_t> instr = user_tokens(tok, specs[i].instructions ? render(*specs[i].instructions) : "");
    Q.ids.insert(Q.ids.end(), instr.begin(), instr.end());
    std::vector<int32_t> ends;
    for (const std::string& o : opts) {
      Q.ids.push_back(d.option);
      const std::vector<int32_t> t = user_tokens(tok, o);
      Q.ids.insert(Q.ids.end(), t.begin(), t.end());
      Q.ids.push_back(d.option_end);
      ends.push_back(int32_t(Q.ids.size()) - 1);
    }
    Q.ids.push_back(d.decide);
    if (int(Q.ids.size()) > kMaxRow - Ls)
      invalid("branch too long: " + std::to_string(Q.ids.size()) + " tokens with a " + std::to_string(Ls) +
              "-token state (row limit " + std::to_string(kMaxRow) + ")");
    Q.rows.push_back(int32_t(Q.ids.size()) - 1);
    Q.rows.insert(Q.rows.end(), ends.begin(), ends.end());
    r.input_tokens += int(Q.ids.size());
  }
  return r;
}

double round_prob(double x) {
  // round(x, 4): the correctly rounded decimal (half to even on the exact binary value), read back.
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.4f", x);
  return std::strtod(buf, nullptr);
}

json::Value answer(const Question& q, const std::vector<double>& p0) {
  // The confidences of api.py (TypeSafe's system-one-adapter 0.2.1): p normalized to sum 1 (all zeros: uniform).
  std::vector<double> p = p0;
  double t = 0;
  for (double x : p) t += x;
  for (double& x : p) x = t == 0 ? 1.0 / double(p.size()) : x / t;
  size_t mode = 0;
  for (size_t i = 1; i < p0.size(); ++i)
    if (p0[i] > p0[mode]) mode = i;  // first most likely option of the request's probabilities
  json::Value o = json::Value::object();
  o.set("type", q.type);
  if (q.type == "noul") {
    o.set("noul", round_prob(p0[1]));
    return o;
  }
  const size_t K = p0.size();
  if (q.type == "choice") {
    size_t pm = 0;
    for (size_t i = 1; i < K; ++i)
      if (p[i] > p[pm]) pm = i;
    const double conf = K == 1 ? 1.0 : (p[pm] - 1.0 / double(K)) / (1.0 - 1.0 / double(K));
    o.set("choice", q.keys[mode]);
    o.set("confidence", round_prob(conf));
    json::Value dist = json::Value::object();
    for (size_t i = 0; i < K; ++i) dist.set(q.keys[i], round_prob(p0[i]));
    o.set("probabilities", std::move(dist));
    return o;
  }
  double score = 0;
  for (size_t i = 0; i < K; ++i) score += double(i) * p0[i];
  double conf = 1.0;
  if (K > 1) {
    size_t m = 0;
    for (size_t i = 1; i < K; ++i)
      if (p[i] > p[m]) m = i;
    double D = 0, e = 0;
    for (size_t i = 0; i < K; ++i) D += std::fabs(double(i) - double(K - 1) / 2.0);
    D /= double(K);
    for (size_t i = 0; i < K; ++i) e += p[i] * std::fabs(double(i) - double(m));
    conf = std::max(0.0, 1.0 - e / D);
  }
  o.set("score", round_prob(score));
  json::Value legend = json::Value::object(), dist = json::Value::object();
  for (size_t i = 0; i < K; ++i) legend.set(q.keys[i], q.legend[i]);
  for (size_t i = 0; i < K; ++i) dist.set(std::to_string(i), round_prob(p0[i]));
  o.set("legend", std::move(legend));
  o.set("probabilities", std::move(dist));
  o.set("confidence", round_prob(conf));
  return o;
}

std::unique_ptr<PointerHead> PointerHead::load(const std::string& dir) {
  struct stat st {};
  if (::stat((dir + "/pointer_head.json").c_str(), &st) != 0) {
    const json::Value cfg = json::parse(slurp(dir + "/config.json"));
    if (cfg.get_string("model_type", "") != "videojev")
      throw std::runtime_error("cannot read " + dir + "/pointer_head.json");
    auto h = std::make_unique<PointerHead>();
    h->videojev_ = true;
    const json::Value* bc = cfg.get("backbone_config");
    const json::Value* tc = bc && bc->is_object() ? bc->get("text_config") : nullptr;
    h->hidden_ = int(tc && tc->is_object() ? tc->get_int("hidden_size", 0) : cfg.get_int("hidden_size", 0));
    h->dim_ = int(cfg.get_int("head_dim", 512));
    if (h->hidden_ < 1 || h->dim_ < 1 || h->dim_ > 65536) throw std::runtime_error(dir + "/config.json: bad head sizes");
    h->logit_scale_ = cfg.get_double("head_logit_scale", 10.0);
    h->temperature_ = 1.0;
    h->names_ = {"", "", "", "<|option_end|>", "<|decide|>"};
    auto shard_for = [&](const std::string& tname) -> std::string {
      if (::stat((dir + "/model.safetensors.index.json").c_str(), &st) == 0) {
        const json::Value idx = json::parse(slurp(dir + "/model.safetensors.index.json"));
        if (const json::Value* wm = idx.get("weight_map"); wm && wm->is_object())
          if (const json::Value* f = wm->get(tname); f && f->is_string()) return dir + "/" + f->as_string();
      }
      return dir + "/model.safetensors";
    };
    const int64_t P = h->dim_, D = h->hidden_;
    for (auto [name, dst, shape] : {std::tuple{"head.q.weight", &h->wq_, std::vector<int64_t>{P, D}},
                                    std::tuple{"head.q.bias", &h->bq_, std::vector<int64_t>{P}},
                                    std::tuple{"head.k.weight", &h->wk_, std::vector<int64_t>{P, D}},
                                    std::tuple{"head.k.bias", &h->bk_, std::vector<int64_t>{P}}}) {
      const std::vector<float> v = load_safetensor_f32(shard_for(name), name, shape);
      dst->assign(v.begin(), v.end());
    }
    const std::vector<float> b = load_safetensor_f32(shard_for("head.bias"), "head.bias", {});
    h->logit_bias_ = b.empty() ? 0.0 : double(b[0]);
    return h;
  }
  const json::Value cfg = json::parse(slurp(dir + "/pointer_head.json"));
  if (cfg.get_string("type", "") != "kev_pointer") throw std::runtime_error(dir + "/pointer_head.json: type is not kev_pointer");
  auto h = std::make_unique<PointerHead>();
  h->hidden_ = int(cfg.get_int("hidden_size", 0));
  h->dim_ = int(cfg.get_int("head_dim", 0));
  if (h->hidden_ < 1 || h->dim_ < 1 || h->dim_ > 65536) throw std::runtime_error(dir + "/pointer_head.json: bad sizes");
  h->temperature_ = cfg.get_double("temperature", 1.0);
  if (!(h->temperature_ > 0) || !std::isfinite(h->temperature_))
    throw std::runtime_error(dir + "/pointer_head.json: temperature must be positive and finite");
  const json::Value* tk = cfg.get("tokens");
  for (const char* k : {"state", "question", "option", "option_end", "decide"}) {
    const json::Value* v = tk ? tk->get(k) : nullptr;
    if (!v || !v->is_string()) throw std::runtime_error(dir + "/pointer_head.json: tokens." + k + " missing");
    h->names_.push_back(v->as_string());
  }
  const json::Value* vis = cfg.get("vision");
  h->grafted_ = vis && vis->is_object();
  const std::string file = dir + "/pointer_head.safetensors";
  const std::string bytes = slurp(file);
  if (bytes.size() < 8) throw std::runtime_error(file + ": truncated");
  uint64_t hl = 0;
  std::memcpy(&hl, bytes.data(), 8);
  if (hl > bytes.size() - 8) throw std::runtime_error(file + ": truncated header");
  const json::Value header = json::parse(bytes.substr(8, size_t(hl)));
  const int64_t P = h->dim_, D = h->hidden_;
  for (auto [name, dst, shape] : {std::tuple{"q_weight", &h->wq_, std::vector<int64_t>{P, D}},
                                  std::tuple{"q_bias", &h->bq_, std::vector<int64_t>{P}},
                                  std::tuple{"k_weight", &h->wk_, std::vector<int64_t>{P, D}},
                                  std::tuple{"k_bias", &h->bk_, std::vector<int64_t>{P}}}) {
    const std::vector<float> v = f32_tensor(file, bytes, header, name, shape);
    dst->assign(v.begin(), v.end());
  }
  return h;
}

void PointerHead::bind(const Tokenizer& tok) {
  int32_t* ids[5] = {&ids_.state, &ids_.question, &ids_.option, &ids_.option_end, &ids_.decide};
  for (size_t i = 0; i < names_.size(); ++i) {
    if (names_[i].empty()) {
      *ids[i] = -1;
      continue;
    }
    *ids[i] = tok.token_id(names_[i]);
    if (*ids[i] < 0) throw std::runtime_error("pointer_head.json: " + names_[i] + " is not a token of this tokenizer");
  }
}

std::vector<float> PointerHead::logits(const float* h, int k) const {
  // PointerHead.forward: (k(h_opts) @ q(h_decide)) / sqrt(head_dim), projections in double.
  const int D = hidden_, P = dim_, R = k + 1;
  std::vector<double> x(size_t(R) * size_t(D)), proj(size_t(R) * size_t(P));
  for (size_t i = 0; i < x.size(); ++i) x[i] = double(h[i]);
  // q of the <decide> row, k of every option row.
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, P, D, 1.0, x.data(), D, wq_.data(), D, 0.0, proj.data(), P);
  if (k > 0)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, k, P, D, 1.0, x.data() + D, D, wk_.data(), D, 0.0,
                proj.data() + P, P);
  std::vector<double> q(static_cast<size_t>(P));
  for (int j = 0; j < P; ++j) q[size_t(j)] = proj[size_t(j)] + bq_[size_t(j)];
  if (videojev_) {
    double qn = 0;
    for (int j = 0; j < P; ++j) qn += q[size_t(j)] * q[size_t(j)];
    qn = std::max(std::sqrt(qn), 1e-6);
    for (int j = 0; j < P; ++j) q[size_t(j)] /= qn;
    std::vector<float> out(static_cast<size_t>(k));
    for (int o = 0; o < k; ++o) {
      const double* kr = proj.data() + size_t(o + 1) * size_t(P);
      double kn = 0, z = 0;
      for (int j = 0; j < P; ++j) {
        const double kj = kr[j] + bk_[size_t(j)];
        kn += kj * kj;
        z += kj * q[size_t(j)];
      }
      kn = std::max(std::sqrt(kn), 1e-6);
      out[size_t(o)] = float((z / kn) * logit_scale_ + logit_bias_);
    }
    return out;
  }
  const double scale = 1.0 / std::sqrt(double(P));
  std::vector<float> out(static_cast<size_t>(k));
  for (int o = 0; o < k; ++o) {
    const double* kr = proj.data() + size_t(o + 1) * size_t(P);
    double z = 0;
    for (int j = 0; j < P; ++j) z += (kr[j] + bk_[size_t(j)]) * q[size_t(j)];
    out[size_t(o)] = float(z * scale);
  }
  return out;
}

}  // namespace espetero::kev
