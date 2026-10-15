// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Kev decision models without a GPU: the release's encoding of SystemOne
// requests (render, the delimiter rewrite of caller text, the state and
// question rows, the read-out rows), its answers (rounding, confidences,
// output_tokens), the request checks and the pointer head file. The fixtures
// (tests/ref/kev_fixtures.json) come from tests/ref/kev_ref.py tokens, a
// re-implementation of the release's encoder on transformers' tokenizer; the
// encoding needs Kev's tokenizer (Qwen3.8's, the converted directory or a
// cached snapshot), else those checks are skipped. The forward pass and the
// head are checked against mlx-vlm by tests/ref/kev_ref.py decisions / check.
//
//   build/kev_test tests/ref/kev_fixtures.json [TOKENIZER_DIR]
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "../src/engine/engine.h"
#include "../src/engine/kev.h"
#include "../src/tok/tokenizer.h"
#include "../src/util/json.h"

using namespace espetero;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
  }
}

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool exists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}

// A cached snapshot with Kev's tokenizer (the release, or the Qwen3.8-27B base it shares it with).
std::string find_tokenizer() {
  std::string hub;
  if (const char* h = std::getenv("HF_HUB_CACHE"); h && *h) hub = h;
  else if (const char* h2 = std::getenv("HF_HOME"); h2 && *h2) hub = std::string(h2) + "/hub";
  else hub = std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/.cache/huggingface/hub";
  for (const char* repo : {"models--jaredpalmer--kev-27b", "models--mlx-community--Qwen3.8-27B-8bit",
                           "models--mlx-community--Qwen3.8-27B-4bit"}) {
    const std::string snaps = hub + "/" + repo + "/snapshots";
    DIR* d = opendir(snaps.c_str());
    if (!d) continue;
    std::string found;
    while (dirent* e = readdir(d)) {
      if (e->d_name[0] == '.') continue;
      if (exists(snaps + "/" + e->d_name + "/tokenizer.json")) found = snaps + "/" + e->d_name;
    }
    closedir(d);
    if (!found.empty()) return found;
  }
  return "";
}

std::vector<int32_t> ints(const json::Value& a) {
  std::vector<int32_t> v;
  for (const json::Value& x : a.as_array()) v.push_back(int32_t(x.as_int()));
  return v;
}

int status_of(const std::function<void()>& f) {
  try {
    f();
  } catch (const RequestError& e) {
    return e.status;
  } catch (...) {
    return -1;
  }
  return 0;
}

void render_tests(const json::Value& fx) {
  for (const json::Value& r : fx.get("render")->as_array())
    check(kev::render(*r.get("value")) == r.get_string("text", "?"), "render " + json::dump(*r.get("value")));
  // Python's round(x, 4) on the exact binary values.
  check(kev::round_prob(0.12345) == 0.1235 && kev::round_prob(0.00005) == 0.0001 && kev::round_prob(0.00015) == 0.0001 &&
            kev::round_prob(2.0 / 3.0) == 0.6667 && kev::round_prob(1.00005) == 1.0001,
        "round_prob");
}

void encode_tests(const json::Value& fx, const Tokenizer& tok) {
  kev::Delimiters d;
  d.state = tok.token_id("<|fim_prefix|>");
  d.question = tok.token_id("<|fim_middle|>");
  d.option = tok.token_id("<|box_start|>");
  d.option_end = tok.token_id("<|box_end|>");
  d.decide = tok.token_id("<|fim_suffix|>");
  const int32_t vs = tok.token_id("<|vision_start|>"), pad = tok.token_id("<|image_pad|>"), ve = tok.token_id("<|vision_end|>");
  int n = 0;
  for (const json::Value& c : fx.get("cases")->as_array()) {
    const std::string name = "case " + std::to_string(c.get_int("case", -1));
    const json::Value& body = *c.get("body");
    std::vector<int> itoks;
    for (const json::Value& x : c.get("image_tokens")->as_array()) itoks.push_back(int(x.as_int()));
    const kev::Record r = kev::encode(tok, d, *body.get("state"), *body.get("questions"), itoks, vs, pad, ve);
    check(r.state == ints(*c.get("state")), name + ": state ids");
    check(r.input_tokens == c.get_int("input_tokens", -1), name + ": input_tokens");
    const auto& qs = c.get("questions")->as_array();
    check(r.questions.size() == qs.size(), name + ": questions");
    json::Value answers = json::Value::object();
    for (size_t i = 0; i < qs.size() && i < r.questions.size(); ++i) {
      const kev::Question& Q = r.questions[i];
      check(Q.id == qs[i].get_string("id", ""), name + ": question id");
      check(Q.ids == ints(*qs[i].get("ids")), name + ": " + Q.id + " ids");
      check(Q.rows == ints(*qs[i].get("rows")), name + ": " + Q.id + " rows");
      std::vector<std::string> keys;
      for (const json::Value& k : qs[i].get("keys")->as_array()) keys.push_back(k.as_string());
      check(Q.keys == keys, name + ": " + Q.id + " keys");
      std::vector<double> p;
      for (const json::Value& x : (*c.get("probs"))[i].as_array()) p.push_back(x.as_double());
      answers.set(Q.id, kev::answer(Q, p));
    }
    const std::string aj = json::dump_python_default(answers, true);
    check(aj == c.get_string("answers_json", ""), name + ": answers " + aj);
    check(int64_t(tok.encode(aj, true).size()) == c.get_int("output_tokens", -1), name + ": output_tokens");
    ++n;
  }
  std::printf("kev_test: %d encoded cases\n", n);

  // The request models' refusals (422) and the serving limits.
  const json::Value state("s");
  auto enc = [&](const char* qs) { return status_of([&] { kev::encode(tok, d, state, json::parse(qs)); }); };
  check(enc(R"({})") == 422, "no questions");
  check(enc(R"([])") == 422, "questions as a list");
  check(enc(R"({"q": {"type": "bogus"}})") == 422, "unknown type");
  check(enc(R"({"q": {"type": "choice", "criteria": {}}})") == 422, "choice without options");
  check(enc(R"({"q": {"type": "choice"}})") == 422, "choice without criteria");
  check(enc(R"({"q": {"type": "score", "criteria": []}})") == 422, "score without levels");
  check(enc(R"({"q": {"type": "noul", "criteria": ["a"]}})") == 422, "noul criteria as a list");
  check(enc(R"({"q": {"type": "noul"}})") == 0, "noul without instructions");
  std::string many = R"({"q": {"type": "choice", "criteria": {)";
  for (int i = 0; i < 256; ++i) many += (i ? ",\"o" : "\"o") + std::to_string(i) + "\": null";
  check(enc((many + "}}}").c_str()) == 422, "256 options");
  std::string big(70000 * 2, ' ');
  for (size_t i = 0; i < big.size(); i += 2) big[i] = 'a' + char(i % 26);
  check(status_of([&] { kev::encode(tok, d, json::Value(big), json::parse(R"({"q": {"type": "noul"}})")); }) == 422,
        "state over 65,536 tokens");
}

void head_tests() {
  // A small pointer head written as tools/kev_convert.py writes one: logits by hand.
  char dir[] = "/tmp/kev_test_XXXXXX";
  if (!mkdtemp(dir)) return;
  const int P = 2, D = 3;
  const float wq[P * D] = {1, 0, 0, 0, 1, 0}, bq[P] = {0.5f, 0}, wk[P * D] = {0, 0, 1, 1, 1, 0}, bk[P] = {0, -1};
  json::Value hdr = json::Value::object();
  size_t off = 0;
  std::string data;
  for (auto [name, ptr, shape] : {std::tuple{"q_weight", wq, std::vector<int64_t>{P, D}}, std::tuple{"q_bias", bq, std::vector<int64_t>{P}},
                                  std::tuple{"k_weight", wk, std::vector<int64_t>{P, D}}, std::tuple{"k_bias", bk, std::vector<int64_t>{P}}}) {
    size_t n = 1;
    json::Value sh = json::Value::array();
    for (int64_t s : shape) n *= size_t(s), sh.push_back(s);
    json::Value t = json::Value::object(), offs = json::Value::array();
    offs.push_back(int64_t(off));
    offs.push_back(int64_t(off + 4 * n));
    t.set("dtype", "F32");
    t.set("shape", sh);
    t.set("data_offsets", offs);
    hdr.set(name, t);
    data.append(reinterpret_cast<const char*>(ptr), 4 * n);
    off += 4 * n;
  }
  const std::string h = json::dump(hdr);
  const uint64_t hl = h.size();
  std::ofstream(std::string(dir) + "/pointer_head.safetensors", std::ios::binary)
      << std::string(reinterpret_cast<const char*>(&hl), 8) << h << data;
  std::ofstream(std::string(dir) + "/pointer_head.json")
      << R"({"type": "kev_pointer", "hidden_size": 3, "head_dim": 2, "temperature": 1.5, "tokens": {"state": "a",
            "question": "b", "option": "c", "option_end": "d", "decide": "e"}, "lm_head": false})";
  check(kev::is_release(dir), "is_release");
  try {
    auto head = kev::PointerHead::load(dir);
    check(head->hidden() == 3 && head->dim() == 2 && head->temperature() == 1.5 && !head->grafted_vision(), "head sizes");
    // h_decide = (1, 2, 3): q = (1.5, 2); options (0, 0, 1) -> k = (1, -1), (1, 1, 0) -> k = (0, 1).
    const float hs[9] = {1, 2, 3, 0, 0, 1, 1, 1, 0};
    const std::vector<float> z = head->logits(hs, 2);
    const double s = 1.0 / std::sqrt(2.0);
    check(z.size() == 2 && std::fabs(z[0] - (1.5 - 2) * s) < 1e-6 && std::fabs(z[1] - 2 * s) < 1e-6, "head logits");
  } catch (const std::exception& e) {
    check(false, std::string("head load: ") + e.what());
  }
  for (const char* f : {"/pointer_head.safetensors", "/pointer_head.json"}) std::remove((std::string(dir) + f).c_str());
  rmdir(dir);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string fixtures = argc > 1 ? argv[1] : "tests/ref/kev_fixtures.json";
  const json::Value fx = json::parse(slurp(fixtures));
  render_tests(fx);
  head_tests();
  const std::string dir = argc > 2 ? argv[2] : find_tokenizer();
  if (dir.empty()) {
    std::printf("kev_test: encoding checks skipped (no Kev / Qwen3.8 tokenizer: pass a directory with tokenizer.json)\n");
  } else {
    encode_tests(fx, *Tokenizer::from_hf_dir(dir));
  }
  if (failures) {
    std::fprintf(stderr, "kev_test: %d failures\n", failures);
    return 1;
  }
  std::printf("kev_test: ok\n");
  return 0;
}
