// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Decision prompts token-identical to the reference Python stack
// (tests/ref/decisions_ref.py): header, labels, prefix, suffixes, stem, leaves.
//
//   build/decisions_test <model dir> <decisions_ref.json>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "../src/engine/decisions.h"
#include "../src/engine/engine.h"
#include "../src/tmpl/tmpl.h"
#include "../src/tok/tokenizer.h"

using namespace espetero;

static std::vector<int32_t> ints(const json::Value& v) {
  std::vector<int32_t> out;
  for (const auto& x : v.as_array()) out.push_back(int32_t(x.as_int()));
  return out;
}

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  std::ifstream f(argv[2]);
  std::stringstream ss;
  ss << f.rdbuf();
  json::Value ref = json::parse(ss.str());
  auto tok = Tokenizer::from_hf_dir(argv[1]);
  auto tpl = tmpl::Template::parse(tok->chat_template());
  std::vector<Label> labels = make_labels(*tok);
  int bad = 0;
  const auto& rl = ref.get("labels")->as_array();
  for (size_t i = 0; i < rl.size(); ++i)
    if (rl[i][0].as_string() != labels[i].text || rl[i][1].as_int() != labels[i].token) {
      if (bad++ < 3) std::printf("label %zu differs: %s/%d vs %s/%lld\n", i, labels[i].text.c_str(), labels[i].token,
                                 rl[i][0].as_string().c_str(), (long long)rl[i][1].as_int());
    }
  DecisionPromptBuilder b(*tok, *tpl, labels, "");
  if (b.header() != ints(*ref.get("header"))) {
    std::printf("header differs (%zu vs %zu tokens)\n", b.header().size(), ref.get("header")->size());
    ++bad;
  }
  int n = 0;
  for (const auto& c : ref.get("cases")->as_array()) {
    DecisionRequest r = parse_decision_request(*c.get("request"));
    DecisionPlan p = b.plan(r, 0);
    bool ok = p.prefix == ints(*c.get("prefix")) && p.stem == ints(*c.get("stem"));
    const auto& sfx = c.get("suffixes")->as_array();
    const auto& lv = c.get("leaves")->as_array();
    ok = ok && sfx.size() == p.suffixes.size() && lv.size() == p.leaves.size();
    for (size_t i = 0; ok && i < sfx.size(); ++i) ok = p.suffixes[i] == ints(sfx[i]) && p.leaves[i] == ints(lv[i]);
    if (!ok) {
      ++bad;
      std::printf("case %d differs (prefix %zu vs %zu tokens)\n", n, p.prefix.size(), c.get("prefix")->size());
    }
    ++n;
  }
  std::printf("decisions: %d cases, %d mismatches\n", n, bad);
  return bad ? 1 : 0;
}
