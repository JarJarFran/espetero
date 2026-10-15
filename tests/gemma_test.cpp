// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Gemma 4 support without a GPU: config.json parsing (dense, MoE, the
// refused per-layer-embedding models, the vision tower, the unified 12B and
// its embedder), hidden-state decision heads and the tool-call parser of
// its chat format. The forward pass is checked against mlx-lm by
// tests/ref/gemma4_ref.py, with images against mlx-vlm by
// tests/ref/gemma4_vision_ref.py and tests/ref/gemma4_unified_ref.py.
#include <cstdio>
#include <string>

#include <unistd.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

#include "../src/engine/chat.h"
#include "../src/engine/hidden_head.h"
#include "../src/engine/pplx.h"
#include "../src/model/impl/config.h"
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

// The text_config of the 31B (shortened: six layers) and of the 26B-A4B.
const char* kDense = R"({"model_type": "gemma4", "vocab_size": 262144, "eos_token_id": [1, 106, 50],
  "text_config": {"model_type": "gemma4_text", "hidden_size": 5376, "intermediate_size": 21504,
  "num_hidden_layers": 6, "num_attention_heads": 32, "num_key_value_heads": 16, "num_global_key_value_heads": 4,
  "head_dim": 256, "global_head_dim": 512, "attention_k_eq_v": true, "sliding_window": 1024,
  "final_logit_softcapping": 30.0, "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-06,
  "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0, "enable_moe_block": false, "eos_token_id": 1,
  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
                  "sliding_attention", "full_attention"],
  "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                      "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
  "tie_word_embeddings": true}})";

const char* kMoe = R"({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text", "hidden_size": 2816,
  "intermediate_size": 2112, "moe_intermediate_size": 704, "num_experts": 128, "top_k_experts": 8,
  "enable_moe_block": true, "num_hidden_layers": 30, "num_attention_heads": 16, "num_key_value_heads": 8,
  "num_global_key_value_heads": 2, "head_dim": 256, "global_head_dim": 512, "attention_k_eq_v": true,
  "sliding_window": 1024, "sliding_window_pattern": 6, "vocab_size": 262144}})";

const char* kE4b = R"({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text", "hidden_size": 2560,
  "hidden_size_per_layer_input": 256, "num_kv_shared_layers": 18, "num_hidden_layers": 42}})";

void config_tests() {
  {
    const native::Config c = native::config_from_gemma(json::parse(kDense));
    check(c.gemma && c.D == 5376 && c.I == 21504 && c.L == 6 && c.H == 32, "dense: sizes");
    check(c.Hkv == 16 && c.Hkv_g == 4 && c.HD == 256 && c.HDg == 512 && c.k_eq_v, "dense: heads");
    check(c.n_swa == 5 && c.n_full == 1 && c.swa[0] && !c.swa[5], "dense: layer kinds");
    check(c.rot_g == 128 && c.theta_g == 1e6 && c.theta == 1e4 && c.window == 1024, "dense: rope and window");
    check(c.softcap == 30.0f && std::fabs(c.embed_scale - std::sqrt(5376.0f)) < 1e-3f, "dense: softcap and embed scale");
    check(c.eos_ids == std::vector<int>({1, 50, 106}), "dense: end ids");
    check(c.E == 0 && c.V == 262144 && c.tie, "dense: no experts, tied head");
    bool ok = true;
    try {
      c.validate();
    } catch (const std::exception& e) {
      ok = false;
      std::fprintf(stderr, "  %s\n", e.what());
    }
    check(ok, "dense: validates");
    check(c.kv_dim_g() == 2048 && c.kv_dim() == 4096 && c.g4_ld(true) == 16384 && c.g4_ld(false) == 18432,
          "dense: projection widths");
  }
  {
    const native::Config c = native::config_from_gemma(json::parse(kMoe));
    check(c.E == 128 && c.k == 8 && c.Ie == 704 && c.I == 2112, "moe: experts");
    // No layer_types: the pattern (5 sliding, then one global) and mlx-lm's RoPE defaults.
    check(c.n_swa == 25 && c.n_full == 5 && !c.swa[5] && c.swa[6], "moe: pattern");
    check(c.rot_g == 128 && c.theta_g == 1e6, "moe: default rope");
    bool ok = true;
    try {
      c.validate();
    } catch (...) {
      ok = false;
    }
    check(ok, "moe: validates");
  }
  {
    bool refused = false;
    try {
      native::config_from_gemma(json::parse(kE4b));
    } catch (const std::exception& e) {
      refused = std::string(e.what()).find("per-layer input embeddings") != std::string::npos;
    }
    check(refused, "e4b: refused with a message");
  }
}

std::string args_of(const json::Value& calls, size_t i) {
  return (*calls[i].get("function")->get("arguments")).as_string();
}

void tool_tests() {
  json::Value calls;
  std::string before = parse_gemma_tool_calls(
      "Let me check.<|tool_call>call:get_weather{location:<|\"|>Paris, France<|\"|>,unit:<|\"|>celsius<|\"|>}<tool_call|>",
      calls);
  check(before == "Let me check." && calls.is_array() && calls.size() == 1, "tool: one call");
  if (calls.is_array() && calls.size() == 1) {
    check((*calls[0].get("function")->get("name")).as_string() == "get_weather", "tool: name");
    check(args_of(calls, 0) == R"({"location":"Paris, France","unit":"celsius"})", "tool: arguments " + args_of(calls, 0));
  }
  before = parse_gemma_tool_calls(
      "<|tool_call>call:plan{days:3,strict:true,ratio:0.5,tags:[<|\"|>a<|\"|>,<|\"|>b}<|\"|>],opts:{depth:2,name:<|\"|>x<|\"|>},none:None}"
      "<tool_call|><|tool_call>call:noop{}<tool_call|>",
      calls);
  check(before.empty() && calls.is_array() && calls.size() == 2, "tool: two calls");
  if (calls.is_array() && calls.size() == 2) {
    check(args_of(calls, 0) ==
              R"({"days":3,"strict":true,"ratio":0.5,"tags":["a","b}"],"opts":{"depth":2,"name":"x"},"none":null})",
          "tool: typed arguments " + args_of(calls, 0));
    check(args_of(calls, 1) == "{}", "tool: no arguments");
  }
  before = parse_gemma_tool_calls("No call here, just <tool_call> text.", calls);
  check(calls.is_null() && before == "No call here, just <tool_call> text.", "tool: none");
  // An unterminated call (the reply was cut) still parses what is complete.
  before = parse_gemma_tool_calls("<|tool_call>call:f{a:1}", calls);
  check(calls.is_array() && calls.size() == 1 && args_of(calls, 0) == R"({"a":1})", "tool: no closing mark");
}

}  // namespace

// The vision_config of the 26B-A4B and 31B checkpoints (gemma4_vision).
void vision_config_tests() {
  const char* kVision = R"({"model_type": "gemma4", "text_config": {"hidden_size": 2816},
    "vision_config": {"model_type": "gemma4_vision", "hidden_size": 1152, "intermediate_size": 4304,
    "num_hidden_layers": 27, "num_attention_heads": 16, "num_key_value_heads": 16, "head_dim": 72,
    "hidden_activation": "gelu_pytorch_tanh", "patch_size": 16, "pooling_kernel_size": 3,
    "position_embedding_size": 10240, "rms_norm_eps": 1e-06, "standardize": true, "use_clipped_linears": false,
    "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"}}})";
  native::VisionConfig v = native::vision_from_gemma(json::parse(kVision), 2816);
  check(v.present && v.gemma && v.depth == 27 && v.D == 1152 && v.I == 4304 && v.heads == 16, "vision: sizes");
  check(v.patch == 16 && v.merge == 3 && v.tps == 1 && v.patch_dim() == 768 && v.n_pos == 10240, "vision: patches");
  check(v.standardize && v.rope_theta == 100.0 && v.out == 2816 && v.Ip() == 4352, "vision: standardize, rope, width");
  bool ok = true;
  try {
    v.validate(2816);
  } catch (...) {
    ok = false;
  }
  check(ok && v.hd == 72, "vision: validates (head dim 72)");
  ok = false;
  try {
    native::vision_from_gemma(json::parse(R"({"vision_config": {"use_clipped_linears": true}})"), 2816);
  } catch (...) {
    ok = true;
  }
  check(ok, "vision: clipped linears refused");
  check(!native::vision_from_gemma(json::parse(kDense), 5376).present, "vision: absent without a vision_config");
}

// gemma4_unified (the 12B): the global layers' geometry as global_head_dim /
// num_global_key_value_heads or as transformers 5.17's per_layer_config, and
// the encoder-free vision_config.
void unified_tests() {
  std::string types;
  for (int i = 0; i < 48; ++i) types += std::string(i ? ", " : "") + ((i % 6 == 5) ? "\"full_attention\"" : "\"sliding_attention\"");
  const std::string text = R"("hidden_size": 3840, "intermediate_size": 15360, "num_hidden_layers": 48,
      "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256, "attention_k_eq_v": true,
      "sliding_window": 1024, "final_logit_softcapping": 30.0, "vocab_size": 262144, "layer_types": [)" + types + R"(],
      "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
      "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}})";
  const std::string explicit_cfg = R"({"model_type": "gemma4_unified", "text_config": {"model_type": "gemma4_unified_text",
      "global_head_dim": 512, "num_global_key_value_heads": 1, )" + text + "}}";
  std::string plc;
  for (int i = 5; i < 48; i += 6) plc += std::string(i > 5 ? ", " : "") + "\"" + (i < 10 ? "0" : "") + std::to_string(i) +
                                       "\": {\"head_dim\": 512, \"num_key_value_heads\": 1}";
  const std::string plc_cfg = R"({"model_type": "gemma4_unified", "text_config": {"model_type": "gemma4_unified_text",
      "per_layer_config": {)" + plc + "}, " + text + "}}";
  check(native::is_gemma4_type("gemma4_unified") && native::is_gemma4_type("gemma4_unified_text"), "unified: types");
  for (const std::string* cfg : {&explicit_cfg, &plc_cfg}) {
    const native::Config c = native::config_from_gemma(json::parse(*cfg));
    const std::string w = cfg == &plc_cfg ? "unified (per_layer_config): " : "unified: ";
    check(c.L == 48 && c.n_full == 8 && c.n_swa == 40 && !c.swa[5] && !c.swa[47] && c.swa[46], w + "layers");
    check(c.HD == 256 && c.Hkv == 8 && c.HDg == 512 && c.Hkv_g == 1 && c.k_eq_v && c.rot_g == 128, w + "geometry");
    check(c.window == 1024 && c.softcap == 30.0f && c.tie, w + "window, softcap, tied head");
    bool ok = true;
    try {
      c.validate();
    } catch (const std::exception& e) {
      ok = false;
      std::fprintf(stderr, "  %s\n", e.what());
    }
    check(ok, w + "validates (global GQA group 16)");
  }
  bool refused = false;
  try {
    native::config_from_gemma(json::parse(R"({"model_type": "gemma4_unified", "text_config": {"num_hidden_layers": 6,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "full_attention"], "per_layer_config": {"0": {"head_dim": 512}}}})"));
  } catch (const std::exception& e) {
    refused = std::string(e.what()).find("sliding layers") != std::string::npos;
  }
  check(refused, "unified: a per_layer_config override of a sliding layer is refused");
  const char* kVision = R"({"model_type": "gemma4_unified", "vision_config": {"model_type": "gemma4_unified_vision",
      "mm_embed_dim": 3840, "mm_posemb_size": 1120, "model_patch_size": 48, "num_soft_tokens": 280,
      "output_proj_dims": 3840, "patch_size": 16, "pooling_kernel_size": 3, "rms_norm_eps": 1e-06}})";
  native::VisionConfig v = native::vision_from_gemma(json::parse(kVision), 3840);
  check(v.present && v.gemma && v.unified && v.D == 3840 && v.n_pos == 1120 && v.out == 3840, "unified vision: sizes");
  check(v.patch == 16 && v.merge == 3 && v.patch_dim() == 768 && v.unified_dim() == 6912 && v.ln_eps == 1e-5f,
        "unified vision: patches (16-pixel rows, 48-pixel tokens)");
  bool ok = true;
  try {
    v.validate(3840);
  } catch (...) {
    ok = false;
  }
  check(ok, "unified vision: validates");
  ok = false;
  try {
    native::vision_from_gemma(json::parse(R"({"model_type": "gemma4_unified", "vision_config": {"model_patch_size": 32}})"), 3840);
  } catch (...) {
    ok = true;
  }
  check(ok, "unified vision: model_patch_size other than patch x pool refused");
}

// A hidden-state head (hidden_head.h) written as tools/gemma4_unified_convert.py
// writes it: logits = W ((h - mu) / sd) + b over the first n slots.
void hidden_head_tests() {
  char tmpl[] = "/tmp/espetero_head_XXXXXX";
  const char* dir = mkdtemp(tmpl);
  check(dir != nullptr, "head: temp dir");
  if (!dir) return;
  const int S = 3, D = 4;
  std::vector<float> w(S * D), b(S), mu(D), sd(D), h(D);
  for (int i = 0; i < S * D; ++i) w[size_t(i)] = 0.25f * float(i % 5) - 0.5f;
  for (int i = 0; i < S; ++i) b[size_t(i)] = 0.1f * float(i);
  for (int i = 0; i < D; ++i) mu[size_t(i)] = float(i) - 1.5f, sd[size_t(i)] = 0.5f + float(i), h[size_t(i)] = 2.0f - float(i);
  std::string header = "{";
  size_t off = 0;
  std::string data;
  auto add = [&](const char* name, const std::vector<float>& v, const std::string& shape) {
    if (header.size() > 1) header += ",";
    header += std::string("\"") + name + "\":{\"dtype\":\"F32\",\"shape\":" + shape + ",\"data_offsets\":[" +
              std::to_string(off) + "," + std::to_string(off + v.size() * 4) + "]}";
    data.append(reinterpret_cast<const char*>(v.data()), v.size() * 4);
    off += v.size() * 4;
  };
  add("weight", w, "[3,4]");
  add("bias", b, "[3]");
  add("mu", mu, "[4]");
  add("sd", sd, "[4]");
  header += "}";
  {
    std::ofstream f(std::string(dir) + "/hidden_head.safetensors", std::ios::binary);
    const uint64_t n = header.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f << header << data;
    std::ofstream(std::string(dir) + "/hidden_head.json")
        << R"({"type": "linear", "input": "final_norm_last_row", "standardize": true, "slots": 3, "hidden_size": 4})";
  }
  check(HiddenHead::present(dir), "head: present");
  try {
    auto head = HiddenHead::load(dir);
    check(head->slots() == S && head->hidden() == D, "head: sizes");
    const std::vector<float> z = head->logits(h.data(), 2);
    bool ok = z.size() == 2;
    for (int s = 0; s < 2 && ok; ++s) {
      double ref = b[size_t(s)];
      for (int j = 0; j < D; ++j) ref += double(w[size_t(s * D + j)]) * double((h[size_t(j)] - mu[size_t(j)]) / sd[size_t(j)]);
      ok = std::fabs(double(z[size_t(s)]) - ref) < 1e-6;
    }
    check(ok, "head: logits of the first two slots");
    check(!head->noncausal() && head->temperature() == 1.0 && head->lm_head(), "head: causal, temperature 1, LM head by default");
  } catch (const std::exception& e) {
    check(false, std::string("head: load failed: ") + e.what());
  }
  // A decision checkpoint's head (tools/pplx_decider_convert.py): noncausal attention, a calibration
  // temperature, a placeholder LM head; unknown modes and non-positive temperatures are refused.
  auto with_json = [&](const std::string& extra) {
    std::ofstream(std::string(dir) + "/hidden_head.json")
        << R"({"type": "linear", "input": "final_norm_last_row", "standardize": true, "slots": 3, "hidden_size": 4)" + extra + "}";
  };
  with_json(R"(, "attention_mode": "noncausal_full_attention", "temperature": 1.0087417621345625, "lm_head": false, "codes": ["A", "B"])");
  try {
    auto head = HiddenHead::load(dir);
    check(head->noncausal() && head->temperature() == 1.0087417621345625 && !head->lm_head(),
          "head: noncausal, its temperature, no LM head");
    check(head->codes() == std::vector<std::string>{"A", "B"}, "head: its answer codes");
  } catch (const std::exception& e) {
    check(false, std::string("head: noncausal load failed: ") + e.what());
  }
  for (const char* bad : {R"(, "attention_mode": "bidirectional")", R"(, "temperature": 0)", R"(, "temperature": -1)",
                          R"(, "codes": ["A", "B", "C", "D"])", R"(, "codes": ["A", 2])"}) {
    with_json(bad);
    bool refused = false;
    try {
      HiddenHead::load(dir);
    } catch (const std::exception&) {
      refused = true;
    }
    check(refused, std::string("head: refused ") + bad);
  }
  for (const char* f : {"/hidden_head.safetensors", "/hidden_head.json"}) std::remove((std::string(dir) + f).c_str());
  rmdir(dir);
}

// pplx-decider: the release's decision prompt (decision_messages, options,
// describe), its typed answers, and the 422s before any pass.
void pplx_tests() {
  const std::vector<std::string> codes = {"A", "B", "C", "D"};
  const json::Value state = json::parse(R"({"obra": "Calle Mayor 3", "fotos": 2})");
  const json::Value images = json::parse(R"(["data:image/png;base64,AAAA"])");
  auto text_of = [](const pplx::Decision& d) {
    const json::Value& content = *d.chat.get("messages")->as_array()[1].get("content");
    return content.as_array().back().get_string("text", "");
  };
  const pplx::Decision n = pplx::encode(codes, state, "q", json::parse(R"({"type": "noul", "instructions": "Is it done?"})"),
                                        images, "low");
  check(text_of(n) == "State:\n{\"obra\": \"Calle Mayor 3\", \"fotos\": 2}\n\nQuestion:\nIs it done?\n\nOptions:\n"
                      "A: No / false\nB: Yes / true\n\nReturn only the letter code of the best option.",
        "pplx: noul prompt as decision_messages renders it");
  const json::Value& msgs = *n.chat.get("messages");
  check(msgs.as_array()[0].get_string("content", "").rfind("Classify the supplied state", 0) == 0, "pplx: system prompt");
  const json::Value& parts = *msgs.as_array()[1].get("content");
  check(parts.size() == 2 && parts.as_array()[0].get_string("type", "") == "image_url" &&
            parts.as_array()[0].get("image_url")->get_string("detail", "") == "low",
        "pplx: the image part before the text, at detail low");
  check(n.chat.get("head")->get_int("slots", 0) == 2 && !n.chat.get("chat_template_kwargs")->get_bool("enable_thinking", true),
        "pplx: two head slots, thinking off");
  const pplx::Decision c = pplx::encode(codes, json::Value("plain"), "c",
                                        json::parse(R"({"type": "choice", "criteria": {"red": "warm", "blue": null}})"),
                                        json::Value(), "low");
  check(text_of(c).find("Question:\nChoose the best matching option.\n\nOptions:\nA: red: warm\nB: blue\n\n") != std::string::npos &&
            c.chat.get("messages")->as_array()[1].get("content")->size() == 1,
        "pplx: choice options, default instructions, no images");
  const json::Value ca = pplx::answer(c, {0.2, 0.8});
  check(ca.get_string("choice", "") == "blue" && std::fabs(ca.get_double("confidence", 0) - 0.6) < 1e-12,
        "pplx: choice answer and its confidence");
  const pplx::Decision s = pplx::encode(codes, state, "s", json::parse(R"({"type": "score", "criteria": ["bad", "ok", "good"]})"),
                                        json::Value(), "low");
  const json::Value sa = pplx::answer(s, {0.0, 0.5, 0.5});
  check(std::fabs(sa.get_double("score", 0) - 1.5) < 1e-12 && sa.get("legend")->get_string("2", "") == "good",
        "pplx: score answer is the expected level, with its legend");
  check(std::fabs(pplx::answer(n, {0.3, 0.7}).get_double("noul", 0) - 0.7) < 1e-12, "pplx: noul answer is P(true)");
  for (const char* bad : {R"({"type": "maybe"})", R"({"type": "choice"})", R"({"type": "score", "criteria": ["one"]})",
                          R"({"type": "choice", "criteria": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}})"}) {
    bool refused = false;
    try {
      pplx::encode(codes, state, "x", json::parse(bad), json::Value(), "low");
    } catch (const RequestError& e) {
      refused = e.status == 422;
    }
    check(refused, std::string("pplx: 422 for ") + bad);
  }
}

int main() {
  config_tests();
  hidden_head_tests();
  pplx_tests();
  vision_config_tests();
  unified_tests();
  tool_tests();
  if (failures) {
    std::fprintf(stderr, "gemma_test: %d failures\n", failures);
    return 1;
  }
  std::printf("gemma_test: ok\n");
  return 0;
}
