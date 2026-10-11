// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Gemma 4 support without a GPU: config.json parsing (dense, MoE, the
// refused per-layer-embedding models) and the tool-call parser of its chat
// format. The forward pass is checked against mlx-lm by tests/ref/gemma4_ref.py.
#include <cstdio>
#include <string>

#include "../src/engine/chat.h"
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

int main() {
  config_tests();
  tool_tests();
  if (failures) {
    std::fprintf(stderr, "gemma_test: %d failures\n", failures);
    return 1;
  }
  std::printf("gemma_test: ok\n");
  return 0;
}
