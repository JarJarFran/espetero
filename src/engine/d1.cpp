// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "d1.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "../tok/tokenizer.h"
#include "../tok/unicode.h"
#include "../vision/vision.h"
#include "decisions.h"
#include "engine.h"

namespace espetero::d1 {

namespace {

[[noreturn]] void invalid(const std::string& msg) { throw RequestError(422, msg); }

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return "";
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

// Python's repr() of a str (single quotes unless the text holds one and no double quote).
std::string py_repr_str(const std::string& s) {
  const bool dq = s.find('\'') != std::string::npos && s.find('"') == std::string::npos;
  const char q = dq ? '"' : '\'';
  std::string out(1, q);
  for (const char ch : s) {
    if (ch == '\\') out += "\\\\";
    else if (ch == q) out += std::string("\\") + q;
    else if (ch == '\n') out += "\\n";
    else if (ch == '\r') out += "\\r";
    else if (ch == '\t') out += "\\t";
    else out.push_back(ch);
  }
  return out + q;
}

std::string py_str(const json::Value& v);

std::string py_repr(const json::Value& v) { return v.is_string() ? py_repr_str(v.as_string()) : py_str(v); }

// Python's str() of a JSON value as json.loads gives it (f-strings in the release's prompt).
std::string py_str(const json::Value& v) {
  if (v.is_null()) return "None";
  if (v.is_bool()) return v.as_bool() ? "True" : "False";
  if (v.is_string()) return v.as_string();
  if (v.is_number()) return json::dump(v);  // json.dumps writes ints and floats as repr() does
  std::string out;
  if (v.is_array()) {
    out = "[";
    for (size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + py_repr(v[i]);
    return out + "]";
  }
  const json::Object& o = v.as_object();
  out = "{";
  for (size_t i = 0; i < o.size(); ++i) out += (i ? ", " : "") + py_repr_str(o.keys[i]) + ": " + py_repr(o.vals[i]);
  return out + "}";
}

// Python truthiness.
bool truthy(const json::Value* v) {
  if (!v || v->is_null()) return false;
  if (v->is_bool()) return v->as_bool();
  if (v->is_number()) return v->as_double() != 0.0;
  if (v->is_string()) return !v->as_string().empty();
  if (v->is_array()) return v->size() > 0;
  return v->as_object().size() > 0;
}

// str.strip(): whitespace off both ends (Unicode White_Space).
std::string py_strip(const std::string& s) {
  const auto* b = reinterpret_cast<const unsigned char*>(s.data());
  const auto* e = b + s.size();
  std::vector<std::pair<size_t, bool>> cps;  // (byte offset, is space)
  for (const unsigned char* p = b; p < e;) {
    uint32_t cp;
    const int n = tok::decode_utf8(p, e, cp);
    cps.push_back({size_t(p - b), cp != tok::kInvalidCp && tok::cp_class(cp) == tok::ucd::kSpace});
    p += n;
  }
  size_t i = 0, j = cps.size();
  while (i < j && cps[i].second) ++i;
  while (j > i && cps[j - 1].second) --j;
  if (i == j) return "";
  const size_t from = cps[i].first, to = j < cps.size() ? cps[j].first : s.size();
  return s.substr(from, to - from);
}

// len(k) == 1 and k.isalpha(): one code point of a letter category.
bool one_letter(const std::string& s) {
  const auto* b = reinterpret_cast<const unsigned char*>(s.data());
  if (s.empty()) return false;
  uint32_t cp;
  const int n = tok::decode_utf8(b, b + s.size(), cp);
  return size_t(n) == s.size() && cp != tok::kInvalidCp && tok::cp_class(cp) == tok::ucd::kLetter;
}

// option_codes: the labels themselves when they are single letters, else A..Z, else 00, 01, ...
std::vector<std::string> option_codes(const std::vector<std::string>& labels) {
  std::vector<std::string> labs;
  for (const std::string& l : labels) labs.push_back(py_strip(l));
  bool letters = !labs.empty();
  for (const std::string& l : labs) letters = letters && one_letter(l);
  if (letters) return labs;
  std::vector<std::string> out;
  char buf[16];
  for (size_t i = 0; i < labs.size(); ++i) {
    if (labs.size() <= 26) out.push_back(std::string(1, char('A' + i)));
    else {
      std::snprintf(buf, sizeof(buf), "%02zu", i);
      out.push_back(buf);
    }
  }
  return out;
}

// The release's fallback codes when a code is not one token (or taken).
const std::vector<std::string>& fallback_pool() {
  static const std::vector<std::string> pool = [] {
    std::vector<std::string> p;
    char buf[16];
    for (char c = 'A'; c <= 'Z'; ++c) p.push_back(std::string(1, c));
    for (int i = 0; i < 100; ++i) {
      std::snprintf(buf, sizeof(buf), "%02d", i);
      p.push_back(buf);
    }
    for (char c = 'a'; c <= 'z'; ++c) p.push_back(std::string(1, c));
    for (int i = 0; i < 200; ++i) p.push_back("#" + std::to_string(i));
    for (char a = 'A'; a <= 'Z'; ++a)
      for (char b = 'A'; b <= 'Z'; ++b) p.push_back(std::string{a, b});
    return p;
  }();
  return pool;
}

}  // namespace

bool is_release(const std::string& dir) {
  const std::string txt = read_file(dir + "/config.json");
  if (txt.empty()) return false;
  try {
    const json::Value cfg = json::parse(txt);
    const std::string mt = cfg.get_string("model_type", "");
    if (mt != "lfm2_vl" && mt != "lfm2") return false;
    const json::Value* am = cfg.get("auto_map");
    if (!am || !am->is_object()) return false;
    const std::string cls = am->get_string("AutoModel", "");
    return cls.find("modeling_d1.D1Model") != std::string::npos;
  } catch (...) {
    return false;
  }
}

Release::Release(const Tokenizer& tok, const std::string& bos) : tok_(tok), bos_(bos) {}

int32_t Release::single(const std::string& text) const {
  const std::vector<int32_t> e = tok_.encode(text);
  return e.size() == 1 ? e[0] : -1;
}

std::vector<int32_t> Release::ids_of(const std::vector<std::string>& texts) const {
  std::vector<int32_t> out;
  for (const std::string& t : texts) {
    const int32_t id = single(t);
    if (id >= 0 && std::find(out.begin(), out.end(), id) == out.end()) out.push_back(id);
  }
  return out;
}

std::vector<std::pair<std::string, int32_t>> Release::aliases(const std::vector<std::string>& labels) const {
  std::vector<std::pair<std::string, int32_t>> out;
  std::vector<int32_t> used;
  auto take = [&](const std::string& raw) {
    const int32_t id = single(raw);
    if (id < 0 || std::find(used.begin(), used.end(), id) != used.end()) return false;
    out.push_back({raw, id});
    used.push_back(id);
    return true;
  };
  const std::vector<std::string> codes = option_codes(labels);
  for (const std::string& code : codes) {
    if (take(code)) continue;
    bool ok = false;
    for (const std::string& raw : fallback_pool())
      if (take(raw)) {
        ok = true;
        break;
      }
    if (!ok) invalid("no single-token alias left for " + std::to_string(codes.size()) + " options");
  }
  return out;
}

std::vector<Question> Release::questions(const json::Value& qs) const {
  if (!qs.is_object() || qs.as_object().size() == 0) invalid("questions: an object of named questions, at least one");
  std::vector<Question> out;
  const json::Object& o = qs.as_object();
  for (size_t i = 0; i < o.size(); ++i) {
    const std::string& name = o.keys[i];
    const json::Value& q = o.vals[i];
    if (!q.is_object()) invalid(name + ": a question must be an object");
    Question Q;
    Q.name = name;
    const json::Value* tv = q.get("type");
    if (tv && !tv->is_null() && !tv->is_string()) invalid(name + ": type must be noul, choice or score");
    Q.type = tv && tv->is_string() ? tv->as_string() : "choice";
    if (Q.type != "noul" && Q.type != "score") Q.type = "choice";  // as_question: any other type is a choice
    const json::Value* ins = q.get("instructions");
    if (!ins) invalid(name + ": instructions are required");
    const std::string instructions = py_str(*ins);
    const json::Value* crit = q.get("criteria");
    if (crit) Q.criteria = *crit;
    std::string body;
    if (Q.type == "noul") {
      std::string extra;
      if (truthy(crit)) {
        if (!crit->is_object()) invalid(name + ": noul criteria must map true and false to descriptions");
        const json::Value* t = crit->get("true");
        const json::Value* f = crit->get("false");
        extra = "\nYes: " + (t ? py_str(*t) : std::string("None")) + "\nNo: " + (f ? py_str(*f) : std::string("None"));
      }
      body = instructions + extra + "\n\nReply with yes or no only.";
      const std::vector<int32_t> yes = ids_of({"yes", "Yes", "YES"}), no = ids_of({"no", "No", "NO"});
      if (yes.empty() || no.empty()) invalid("the tokenizer has no single-token yes/no");
      Q.groups = {yes, no};
      Q.options = {"yes", "no"};
    } else if (Q.type == "score") {
      if (!crit || !crit->is_array()) invalid(name + ": score criteria must be a list of level descriptions");
      const size_t n = crit->size();
      if (n == 0) invalid(name + ": score criteria must not be empty");
      std::string legend;
      for (size_t k = 0; k < n; ++k) legend += (k ? "\n" : "") + std::to_string(k) + " " + py_str((*crit)[k]);
      body = instructions + "\n\n" + legend + "\n\nReply with a single digit 0-" + std::to_string(n - 1) + " only.";
      for (size_t k = 0; k < n; ++k) {
        Q.groups.push_back(ids_of({std::to_string(k)}));
        if (Q.groups.back().empty())
          invalid(name + ": a score with " + std::to_string(n) + " levels needs single-token digits; the primitive is "
                         "defined for 2 to 10");
        Q.options.push_back(std::to_string(k));
      }
    } else {
      if (!crit || !crit->is_object()) invalid(name + ": choice criteria must map option names to descriptions");
      const json::Object& co = crit->as_object();
      if (co.size() == 0) invalid(name + ": choice with no options");
      const auto codes = aliases(co.keys);
      std::string lines;
      for (size_t k = 0; k < co.size(); ++k) {
        const json::Value& desc = co.vals[k];
        std::string label = co.keys[k];
        std::replace(label.begin(), label.end(), '_', ' ');
        lines += (k ? "\n" : "") + codes[k].first + " " + (truthy(&desc) ? py_str(desc) : label);
        std::vector<int32_t> g{codes[k].second};
        for (int32_t x : ids_of({" " + codes[k].first}))
          if (x != codes[k].second) g.push_back(x);
        Q.groups.push_back(std::move(g));
        Q.options.push_back(co.keys[k]);
      }
      body = instructions + "\n\nOptions:\n" + lines + "\n\nReply with the option code only.";
    }
    Q.suffix = body + "<|im_end|>\n<|im_start|>assistant\n";
    out.push_back(std::move(Q));
  }
  return out;
}

std::string Release::prefix(const json::Value& state, int images, const std::string& placeholder) const {
  std::string body;
  if (!state.is_null()) {
    std::string block;
    if (state.is_string()) {
      block = state.as_string() + "\n\n";
    } else {
      json::DumpOptions o;  // json.dumps(state, ensure_ascii=False, indent=2)
      o.item_sep = ",";
      o.key_sep = ": ";
      o.indent = 2;
      block = json::dump(state, o) + "\n\n";
    }
    body = block + "\nQUESTION:\n";
  }
  std::string media;
  for (int i = 0; i < images; ++i) media += placeholder;
  return bos_ + "<|im_start|>user\n" + media + body;
}

std::vector<int32_t> Release::cand_ids(const Question& q) {
  std::vector<int32_t> out;
  for (const auto& g : q.groups)
    for (int32_t id : g)
      if (std::find(out.begin(), out.end(), id) == out.end()) out.push_back(id);
  return out;
}

std::vector<double> Release::probabilities(const Question& q, const std::vector<int32_t>& ids, const float* logits) {
  std::vector<double> scores;
  for (const auto& g : q.groups) {
    double best = -INFINITY;
    for (int32_t id : g) {
      const size_t k = size_t(std::find(ids.begin(), ids.end(), id) - ids.begin());
      best = std::max(best, double(logits[k]));
    }
    scores.push_back(best);
  }
  const double m = *std::max_element(scores.begin(), scores.end());
  double sum = 0;
  for (double& s : scores) sum += (s = std::exp(s - m));
  for (double& s : scores) s /= sum;
  return scores;
}

json::Value Release::answer(const Question& q, const std::vector<double>& p) {
  json::Value a = json::Value::object();
  a.set("type", q.type);
  if (q.type == "noul") {
    a.set("noul", p[0]);
    return a;
  }
  size_t best = 0;
  for (size_t k = 1; k < p.size(); ++k)
    if (p[k] > p[best]) best = k;
  json::Value probs = json::Value::object();
  for (size_t k = 0; k < p.size(); ++k) probs.set(q.options[k], p[k]);
  if (q.type == "choice") {
    a.set("choice", q.options[best]);
    a.set("confidence", p[best]);
    a.set("probabilities", std::move(probs));
    return a;
  }
  double e = 0;
  for (size_t k = 0; k < p.size(); ++k) e += double(k) * p[k];
  a.set("score", e);
  a.set("confidence", p[best]);
  a.set("probabilities", std::move(probs));
  json::Value legend = json::Value::object();
  for (size_t k = 0; k < p.size(); ++k) legend.set(std::to_string(k), q.criteria[k]);
  a.set("legend", std::move(legend));
  return a;
}

}  // namespace espetero::d1

// ---------------------------------------------------------------- engine
namespace espetero {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
double round_ms(double x) { return std::round(x * 1000.0) / 1000.0; }
// Every picture of a d1 request is bounded at this many pixels before the processor (the release's VISION_MAX_PIXELS).
constexpr int64_t kD1MaxPixels = 1024 * 1024;
}  // namespace

struct Engine::D1Job {
  std::vector<d1::Question> qs;
  std::vector<int32_t> root;  // the prefix (several questions) or the whole prompt but its last token (one)
  std::vector<int32_t> leaf_off{0}, leaf_tok, cand_off{0}, cand_ids;
  std::vector<std::vector<int32_t>> cands;  // per question: its readout's tokens
  std::vector<PreparedImage> images;
  std::vector<int> image_rows;  // the image runs' first rows in `root`
  std::vector<float> cand_logits;
  std::vector<ImageRef> irefs;
  int input_tokens = 0;
  double gpu_ms = 0, run_ms = 0;
  int passes = 0;
  Clock::time_point queued, started;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  std::string error;
  int error_status = 502;
};

json::Value Engine::d1_systemone(const json::Value& body) {
  const auto started = Clock::now();
  if (!body.is_object()) throw RequestError(422, "request body must be a JSON object");
  const json::Value* model = body.get("model");
  if (model && !model->is_null() && !model->is_string()) throw RequestError(422, "model: a string");
  const json::Value* state = body.get("state");
  if (!state) throw RequestError(422, "state is required (null when the images are the whole state)");
  const json::Value* questions = body.get("questions");
  if (!questions) throw RequestError(422, "questions are required");
  if (const json::Value* v = body.get("videos");
      v && !v->is_null() && !(v->is_array() && v->size() == 0) && !(v->is_bool() && !v->as_bool()))
    throw RequestError(422, "videos are not supported");
  std::vector<std::string> urls;
  if (const json::Value* ims = body.get("images"); ims && !ims->is_null()) {
    if (!ims->is_array()) throw RequestError(422, "images: a list of image URLs");
    if (ims->size() > 16) throw RequestError(422, "images: at most 16 per request");
    for (const json::Value& im : ims->as_array()) {
      std::string url;
      if (im.is_string()) {
        url = im.as_string();
      } else if (im.is_object()) {
        if (const json::Value* u = im.get("url"); u && u->is_string()) url = u->as_string();
        else if (const json::Value* iu = im.get("image_url"); iu && iu->is_object()) url = iu->get_string("url", "");
      }
      if (url.empty()) throw RequestError(422, "images: each image is a data URL string or {\"url\": ...}");
      urls.push_back(std::move(url));
    }
  }
  if (!urls.empty() && !has_vision()) throw RequestError(422, "this model has no vision tower; images are not supported");
  if (state->is_null() && urls.empty()) throw RequestError(422, "state: null needs images (they are the whole state)");
  const ModelInfo& I = model_->info();
  auto job = std::make_shared<D1Job>();
  const auto t_enc = Clock::now();
  job->qs = d1_->questions(*questions);
  for (const std::string& url : urls) job->images.push_back(prepare_image_capped(url, 0, 0, kD1MaxPixels));
  const std::string prefix = d1_->prefix(*state, int(job->images.size()), tok_->decode(std::vector<int32_t>{image_token}));
  check_text_size(prefix.size(), "state");
  // Token ids as the release's run(): one question, the whole prompt in one
  // sequence; several, the prefix as the trunk and each suffix on its own.
  if (job->qs.size() == 1) {
    std::vector<int32_t> ids = tok_->encode(prefix + job->qs[0].suffix);
    if (!job->images.empty()) ids = expand_images(ids, job->images);
    job->input_tokens = int(ids.size());
    job->leaf_tok.assign(1, ids.back());
    ids.pop_back();
    job->root = std::move(ids);
    job->leaf_off.push_back(1);
  } else {
    job->root = tok_->encode(prefix);
    if (!job->images.empty()) job->root = expand_images(job->root, job->images);
    job->input_tokens = int(job->root.size());
    for (const d1::Question& q : job->qs) {
      tok_->encode_append(q.suffix, job->leaf_tok);
      job->leaf_off.push_back(int32_t(job->leaf_tok.size()));
    }
    job->input_tokens += int(job->leaf_tok.size());
  }
  size_t longest = 0;
  for (size_t i = 0; i + 1 < job->leaf_off.size(); ++i) longest = std::max(longest, size_t(job->leaf_off[i + 1] - job->leaf_off[i]));
  if (job->root.size() + longest > size_t(I.max_ctx))
    throw RequestError(422, "Request exceeds the model context window (" + std::to_string(I.max_ctx) + " tokens)");
  if (job->leaf_tok.size() > size_t(I.max_rows))
    throw RequestError(422, "the questions need " + std::to_string(job->leaf_tok.size()) + " rows in one pass, more than the " +
                                std::to_string(I.max_rows) + " of this server (--max-rows)");
  for (const d1::Question& q : job->qs) {
    job->cands.push_back(d1::Release::cand_ids(q));
    job->cand_ids.insert(job->cand_ids.end(), job->cands.back().begin(), job->cands.back().end());
    job->cand_off.push_back(int32_t(job->cand_ids.size()));
  }
  for (size_t i = 0; i < job->root.size(); ++i)
    if (job->root[i] == run_start) {
      job->image_rows.push_back(int(i));
      i += size_t(job->images[job->image_rows.size() - 1].tokens(0)) - 1;
    }
  if (job->image_rows.size() != job->images.size()) throw RequestError(422, "image runs do not match the images");
  const double encode_ms = ms_since(t_enc);
  job->queued = Clock::now();
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (stop_) throw RequestError(503, "engine is shutting down");
    d1_jobs_.push_back(job);
  }
  cv_.notify_all();
  {
    std::unique_lock<std::mutex> lk(job->mu);
    job->cv.wait(lk, [&] { return job->done; });
    if (!job->error.empty()) throw RequestError(job->error_status, job->error);
  }
  // Development: ESPETERO_D1_DUMP=dir writes each request's token ids (the
  // sequence of every question: the root, then its leaf) for parity checks.
  if (const char* dump = std::getenv("ESPETERO_D1_DUMP"); dump && *dump) {
    static std::atomic<int> counter{0};
    char base[64];
    std::snprintf(base, sizeof(base), "/d1_%04d.json", counter++);
    json::Value seqs = json::Value::array();
    for (size_t q = 0; q < job->qs.size(); ++q) {
      json::Value a = json::Value::array();
      for (int32_t t : job->root) a.push_back(int64_t(t));
      for (int32_t k = job->leaf_off[q]; k < job->leaf_off[q + 1]; ++k) a.push_back(int64_t(job->leaf_tok[size_t(k)]));
      seqs.push_back(std::move(a));
    }
    std::ofstream(std::string(dump) + base) << json::dump(seqs);
  }
  const bool want_logits = body.get_bool("return_logits", false);
  json::Value answers = json::Value::object(), raw = json::Value::object();
  for (size_t q = 0; q < job->qs.size(); ++q) {
    const d1::Question& Q = job->qs[q];
    const float* l = job->cand_logits.data() + job->cand_off[q];
    answers.set(Q.name, d1::Release::answer(Q, d1::Release::probabilities(Q, job->cands[q], l)));
    if (want_logits) {  // each option's score: the best of its tokens' logits
      json::Value o = json::Value::object();
      for (size_t k = 0; k < Q.groups.size(); ++k) {
        double best = -INFINITY;
        for (int32_t id : Q.groups[k])
          best = std::max(best, double(l[std::find(job->cands[q].begin(), job->cands[q].end(), id) - job->cands[q].begin()]));
        o.set(Q.options[k], best);
      }
      raw.set(Q.name, std::move(o));
    }
  }
  json::Value out = json::Value::object();
  out.set("model", model && model->is_string() ? model->as_string() : name_);
  out.set("answers", std::move(answers));
  if (want_logits) out.set("logits", std::move(raw));
  json::Value usage = json::Value::object();
  usage.set("input_tokens", int64_t(job->input_tokens));
  usage.set("output_tokens", 0);
  out.set("usage", std::move(usage));
  json::Value metal = json::Value::object();
  metal.set("mode", "system_one");
  metal.set("elapsed_ms", round_ms(ms_since(started)));
  metal.set("queue_ms", round_ms(std::chrono::duration<double, std::milli>(job->started - job->queued).count()));
  metal.set("backbone_ms", round_ms(job->run_ms));
  metal.set("gpu_ms", round_ms(job->gpu_ms));
  metal.set("encode_ms", round_ms(encode_ms));
  metal.set("passes", job->passes);
  metal.set("images", int64_t(job->images.size()));
  int64_t image_tokens = 0;
  for (const PreparedImage& im : job->images) image_tokens += im.tokens(0);
  metal.set("image_tokens", image_tokens);
  out.set("metal", std::move(metal));
  decisions_done_ += 1;
  return out;
}

// The JEVfire decision contract (context + boolean / enum fields) as System One
// questions: boolean fields are noul questions, enum fields choices over their
// values; the response keeps decide()'s format.
json::Value Engine::d1_decide(const json::Value& body) {
  const auto started = Clock::now();
  const DecisionRequest r = parse_decision_request(body);
  json::Value sb = json::Value::object(), qs = json::Value::object();
  sb.set("model", name_);
  sb.set("state", r.context);
  for (const DecisionField& f : r.fields) {
    json::Value q = json::Value::object();
    q.set("type", f.boolean ? "noul" : "choice");
    q.set("instructions", f.description.empty() ? f.name : f.description);
    if (!f.boolean) {
      json::Value c = json::Value::object();
      for (const std::string& ch : f.choices) c.set(ch, json::Value());
      q.set("criteria", std::move(c));
    }
    qs.set(f.name, std::move(q));
  }
  sb.set("questions", std::move(qs));
  if (!r.images.empty()) {
    json::Value ims = json::Value::array();
    for (const std::string& u : r.images) ims.push_back(u);
    sb.set("images", std::move(ims));
  }
  sb.set("return_logits", true);
  const json::Value res = d1_systemone(sb);
  const json::Value& logits = *res.get("logits");
  json::Value fields = json::Value::object(), parsed = json::Value::object(), abstained = json::Value::array();
  for (const DecisionField& f : r.fields) {
    const json::Value& lf = *logits.get(f.name);
    const size_t n = f.n_values();
    std::vector<double> l(n), prob(n), lp(n);
    for (size_t i = 0; i < n; ++i) l[i] = lf.get(f.boolean ? (i == 0 ? "yes" : "no") : f.choices[i])->as_double();
    const double mx = *std::max_element(l.begin(), l.end());
    double z = 0, zt = 0;
    for (size_t i = 0; i < n; ++i) z += std::exp(l[i] - mx), zt += (prob[i] = std::exp((l[i] - mx) / r.score_temperature));
    size_t win = 0;
    for (size_t i = 0; i < n; ++i) {
      prob[i] /= zt;
      lp[i] = l[i] - mx - std::log(z);
      if (prob[i] > prob[win]) win = i;
    }
    const bool abst = r.has_min_probability && prob[win] < r.min_probability;
    json::Value o = json::Value::object(), cands = json::Value::array();
    o.set("value", abst ? json::Value() : f.value(win));
    o.set("selected_value", f.value(win));
    o.set("probability", prob[win]);
    o.set("abstained", abst);
    o.set("candidate_probability_mass", 1.0);
    for (size_t i = 0; i < n; ++i) {
      json::Value c = json::Value::object();
      c.set("value", f.value(i));
      c.set("label", f.boolean ? (i == 0 ? "true" : "false") : f.choices[i]);
      c.set("probability", prob[i]);
      c.set("logprob", lp[i]);
      cands.push_back(std::move(c));
    }
    o.set("candidates", std::move(cands));
    parsed.set(f.name, *o.get("value"));
    if (abst) abstained.push_back(f.name);
    fields.set(f.name, std::move(o));
  }
  const int64_t tokens = res.get("usage")->get_int("input_tokens", 0);
  json::Value out = json::Value::object();
  out.set("model", name_);
  out.set("mode", "system_one");
  out.set("strategy", "system_one");
  out.set("requested_strategy", r.strategy);
  out.set("parsed_json", std::move(parsed));
  out.set("fields", std::move(fields));
  out.set("scores_are_calibrated", true);
  out.set("abstained_fields", std::move(abstained));
  out.set("elapsed_ms", round_ms(ms_since(started)));
  out.set("backend_requests", 1);
  out.set("scored_fields", int64_t(r.fields.size()));
  out.set("max_prompt_tokens", tokens);
  json::Value usage = json::Value::object();
  usage.set("prompt_tokens", tokens);
  usage.set("completion_tokens", 0);
  usage.set("total_tokens", tokens);
  out.set("usage", std::move(usage));
  out.set("metal", *res.get("metal"));
  return out;
}

// Scheduler thread: the root in passes of at most prefill_chunk rows (an image
// run never straddles two) while it and the questions do not fit one pass,
// each continuing the previous one's state; then the rest of the root and
// every question's leaf in one pass, read out at the leaves' last rows.
void Engine::run_d1(D1Job& j) {
  j.started = Clock::now();
  const ModelInfo& I = model_->info();
  const int L = int(j.root.size()), leaf_rows = int(j.leaf_tok.size());
  const int chunk = std::max(64, std::min(I.max_rows, opt_.prefill_chunk));
  int done = 0, parent = -1;
  size_t g = 0;
  auto attach = [&](TreeSpec& t, int end) {
    j.irefs.clear();
    for (; g < j.image_rows.size() && j.image_rows[g] < end; ++g) j.images[g].refs(j.image_rows[g] - done, j.irefs);
    if (!j.irefs.empty()) {
      t.images = j.irefs.data();
      t.n_images = int(j.irefs.size());
    }
  };
  try {
    while (L - done + leaf_rows > I.max_rows) {
      int end = std::min(L, done + chunk);
      for (size_t k = 0; k < j.image_rows.size(); ++k) {
        const int a = j.image_rows[k], b = a + j.images[k].tokens(0);
        if (a < end && end < b) {
          end = a > done ? a : b;  // stop before the image, or take the whole image when it starts the pass
          break;
        }
      }
      if (end - done > I.max_rows) throw RequestError(422, "an image needs more rows than one pass holds (raise --max-rows)");
      while (model_->free_kv_tokens() < (end - done) + 2 * I.block_tokens) {
        const int victim = evict_lru(parent >= 0 ? std::vector<int>{parent} : std::vector<int>{});
        if (victim < 0) break;
        model_->slot_free(victim);
      }
      const int slot = acquire_slot(parent >= 0 ? std::vector<int>{parent} : std::vector<int>{});
      if (slot < 0) throw RequestError(503, "no free model slot for a long decision request");
      TreeSpec t;
      t.parent = parent;
      t.save = slot;
      t.root = j.root.data() + done;
      t.root_len = end - done;
      PassStats st;
      try {
        attach(t, end);
        model_->run({t}, &st);
      } catch (...) {
        model_->slot_free(slot);
        throw;
      }
      j.gpu_ms += st.gpu_ms;
      ++j.passes;
      ++passes_;
      rows_ += uint64_t(st.rows);
      gpu_ms_ = gpu_ms_ + st.gpu_ms;
      if (parent >= 0) model_->slot_free(parent);
      parent = slot;
      done = end;
    }
    TreeSpec t;
    t.parent = parent;
    t.root = L > done ? j.root.data() + done : nullptr;
    t.root_len = L - done;
    t.n_leaves = int(j.qs.size());
    t.leaf_offsets = j.leaf_off.data();
    t.leaf_tokens = j.leaf_tok.data();
    t.cand_offsets = j.cand_off.data();
    t.cand_ids = j.cand_ids.data();
    j.cand_logits.assign(j.cand_ids.size(), 0.0f);
    t.out_cand_logits = j.cand_logits.data();
    attach(t, L);
    PassStats st;
    model_->run({t}, &st);
    j.gpu_ms += st.gpu_ms;
    ++j.passes;
    ++passes_;
    rows_ += uint64_t(st.rows);
    gpu_ms_ = gpu_ms_ + st.gpu_ms;
  } catch (...) {
    if (parent >= 0) model_->slot_free(parent);
    throw;
  }
  if (parent >= 0) model_->slot_free(parent);
  j.run_ms = ms_since(j.started);
}

// One queued d1 request, alone in its passes; false when none waits.
bool Engine::d1_step() {
  std::shared_ptr<D1Job> j;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (d1_jobs_.empty()) return false;
    j = d1_jobs_.front();
    d1_jobs_.pop_front();
  }
  std::string err;
  int status = 502;
  try {
    run_d1(*j);
  } catch (const RequestError& e) {
    err = e.what();
    status = e.status;
  } catch (const std::exception& e) {
    err = e.what();
  }
  if (!err.empty()) std::fprintf(stderr, "[espetero] decision request failed: %s\n", err.c_str());
  {
    std::lock_guard<std::mutex> lk(j->mu);
    j->error = err;
    j->error_status = status;
    j->done = true;
  }
  j->cv.notify_all();
  return true;
}

void Engine::d1_fail_all(const std::string& why) {
  for (auto& j : d1_jobs_) {
    std::lock_guard<std::mutex> jl(j->mu);
    j->error = why;
    j->error_status = 503;
    j->done = true;
    j->cv.notify_all();
  }
  d1_jobs_.clear();
}

}  // namespace espetero
