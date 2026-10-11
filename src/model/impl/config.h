// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Qwen3.5-family geometry (hybrid Gated DeltaNet + gated full attention,
// Qwen3-VL vision tower), read from an MLX config.json or GGUF metadata.
// The kernels are specialized to this family: head dim 256 with a 64-dim
// partial (M-)RoPE, 128-dim delta-rule heads, causal conv width 4. Anything
// else fails at load with a message instead of running incorrectly.
#pragma once

#include "gguf.h"

namespace espetero::native {

struct Config {
  int D = 0, I = 0, L = 0, H = 0, Hkv = 0, HD = 256;
  int Hk = 0, Hv = 0, Dk = 128, Dv = 128, conv_k = 4, V = 0;
  float eps = 1e-6f;
  double theta = 1e7;
  int rot = 64;
  bool tie = true;
  std::vector<bool> linear;
  int n_full = 0, n_linear = 0;
  int rope_sel[32] = {0};  // M-RoPE component (0 t, 1 h, 2 w) per frequency
  // Delta-rule value heads: grouped (HF/MLX: head v uses key head v / (Hv/Hk))
  // or tiled (llama.cpp GGUF: v % Hk, with every value-head tensor reordered).
  bool tiled_heads = false;
  int eos_token_id = -1;
  // Mixture of experts (qwen3_5_moe; E = 0: dense MLPs): E experts of width
  // Ie, k per token, plus a shared expert of width I_shared. Which layers are
  // MoE is read from the checkpoint's tensors by the loader.
  int E = 0, k = 0, Ie = 0, I_shared = 0;
  // Qwen4-Exp (`qwen4_exp`, Qwen3.8-Flash-Next; docs/qwen4_exp.md): a residual of
  // hc streams of D values mixed by gated hyper-connections instead of the
  // pre-norms, a sigmoid-gated DeltaNet output with FLA l2 q/k normalization,
  // Qwen sparse attention (an indexer selects idx_budget / idx_ratio blocks of
  // idx_ratio tokens; dense below the budget) and an n-gram embedding added
  // before layer ple_layer. Every layer is MoE.
  bool q4 = false;
  int hc = 1, hc_rank = 0;
  int idx_heads = 0, idx_hd = 0, idx_kv = 0, idx_ratio = 0, idx_budget = 0;
  int ple_layer = -1, ngram = 0, ngram_heads = 0, ple_conv_k = 0;
  bool gate_sigmoid = false;  // DeltaNet output gate sigma(z) (else silu(z))
  float qk_eps = 1e-6f;       // DeltaNet q/k norm as the kernels apply it: x * rsqrt(mean(x^2) + qk_eps)
  // Gemma 4 (`gemma4`, `gemma4_text`; gemma.h): an attention-only
  // decoder. Sliding-window layers (swa[i]: head dim HD, Hkv heads, the last
  // `window` positions, RoPE theta `theta` over all HD dims) interleave with
  // global layers (head dim HDg, Hkv_g heads, proportional RoPE over the first
  // rot_g / 2 frequency pairs with theta_g; K and V from one projection when
  // k_eq_v). Four RMSNorms and a scalar per layer, a GeGLU MLP (plus, in MoE
  // layers, k of E experts of width Ie beside it), embeddings times
  // embed_scale, logits soft-capped at softcap (tanh). linear[] is all false:
  // n_full counts the global layers, n_swa the sliding ones.
  bool gemma = false;
  int HDg = 0, Hkv_g = 0, window = 0, rot_g = 0, n_swa = 0;
  double theta_g = 1e6, rope_factor_g = 1.0;
  bool k_eq_v = false;
  float softcap = 0.0f, embed_scale = 1.0f;
  std::vector<bool> swa;
  std::vector<int> eos_ids;  // every end-of-turn id the checkpoint names (Gemma: <eos>, <turn|>, <|tool_response>)

  int W() const { return hc * D; }  // residual width per token
  int C() const { return 2 * Hk * Dk + Hv * Dv; }
  int Vd() const { return Hv * Dv; }
  int kv_dim() const { return Hkv * HD; }
  int lin_ld() const { return C() + Vd() + 2 * Hv; }
  int full_ld() const { return H * HD * 2 + 2 * kv_dim(); }
  // Gemma 4: the KV width of the global layers (the paged pool) and the
  // projection rows of a sliding / global layer ([q; k; v], no v when k_eq_v).
  int kv_dim_g() const { return Hkv_g * HDg; }
  int g4_ld(bool sliding) const {
    return sliding ? H * HD + 2 * kv_dim() : H * HDg + (k_eq_v ? 1 : 2) * kv_dim_g();
  }
  int proj_ld() const {
    if (gemma) return std::max({g4_ld(true), g4_ld(false), D});
    return std::max(lin_ld(), full_ld());
  }

  void set_mrope(const std::vector<int64_t>& sec, bool interleaved) {
    if (sec.size() < 3) return;
    const int s0 = int(sec[0]), s1 = int(sec[1]), s2 = int(sec[2]);
    for (int f = 0; f < rot / 2 && f < 32; ++f) {
      if (interleaved) rope_sel[f] = (f % 3 == 1 && f < 3 * s1) ? 1 : ((f % 3 == 2 && f < 3 * s2) ? 2 : 0);
      else rope_sel[f] = f < s0 ? 0 : (f < s0 + s1 ? 1 : 2);
    }
  }

  void set_layers(int interval, const json::Value* types) {
    linear.clear();
    n_full = n_linear = 0;
    for (int i = 0; i < L; ++i) {
      const bool lin = types && types->is_array() && size_t(i) < types->size()
                           ? (*types)[size_t(i)].as_string() == "linear_attention"
                           : ((i + 1) % interval != 0);
      linear.push_back(lin);
      lin ? ++n_linear : ++n_full;
    }
  }

  void validate() const {
    if (gemma) {
      // The attention kernels exist for these head sizes and groups (gemma.h).
      if (!D || !I || !L || !H || !Hkv || !Hkv_g || !V || !window) fail("incomplete Gemma 4 configuration");
      if (HD != 256 || HDg != 512) fail("unsupported Gemma 4 head sizes (sliding 256, global 512)");
      if (H % Hkv || H % Hkv_g) fail("unsupported Gemma 4 head grouping");
      const int Gs = H / Hkv, Gg = H / Hkv_g;
      if (Gs != 1 && Gs != 2 && Gs != 4 && Gs != 8) fail("unsupported Gemma 4 sliding GQA group " + std::to_string(Gs));
      if (Gg != 2 && Gg != 4 && Gg != 8 && Gg != 16) fail("unsupported Gemma 4 global GQA group " + std::to_string(Gg));
      if (D % 64 || I % 64 || rot_g < 2 || rot_g > HDg || rot_g % 2) fail("unsupported Gemma 4 geometry");
      if (E && (E > 1024 || k < 1 || k > 16 || Ie % 64))
        fail("unsupported Gemma 4 mixture-of-experts geometry (experts <= 1024, k <= 16, expert width a multiple of 64)");
      return;
    }
    if (!D || (!I && !E) || !L || !H || !Hkv || !Hk || !Hv || !V) fail("incomplete Qwen3.5 configuration");
    if (E && (E > 1024 || k < 1 || k > 16 || Ie % 64 || D % 64 || !I_shared))
      fail("unsupported mixture-of-experts geometry (experts <= 1024, k <= 16, expert width a multiple of 64, shared expert)");
    if (HD != 256 || Dk != 128 || Dv != 128 || conv_k != 4 || rot != 64)
      fail("unsupported Qwen3.5 geometry (head dim 256, rotary 64, delta heads 128, conv 4)");
    if (H % Hkv || Hv % Hk) fail("unsupported head grouping");
    const int G = H / Hkv;
    if (G != 1 && G != 2 && G != 3 && G != 4 && G != 6 && G != 8 && !(q4 && G == 12))
      fail("unsupported GQA group " + std::to_string(G));
    if (q4 && (hc != 4 || hc_rank % 64 || !E || ple_layer < 0 || ple_layer >= L || ngram != 3 || ple_conv_k != 4 ||
               idx_ratio != 4 || idx_hd != 128 || idx_kv != 1))
      fail("unsupported qwen4_exp geometry (4 residual streams, low rank a multiple of 64, MoE, n-gram 3 with conv 4, "
           "indexer blocks of 4 with one 128-wide key head)");
  }
};

struct VisionConfig {
  bool present = false;
  int depth = 0, D = 0, I = 0, heads = 0, hd = 0, out = 0;
  int patch = 16, tps = 2, merge = 2, in_ch = 3, n_pos = 0, side = 0;
  float eps = 1e-6f;
  int patch_dim() const { return in_ch * tps * patch * patch; }
  int Ip() const { return (I + 63) / 64 * 64; }

  void validate(int text_hidden) {
    hd = heads ? D / heads : 0;
    side = int(std::lround(std::sqrt(double(n_pos))));
    if (hd != 64 && hd != 72 && hd != 80) fail("unsupported vision head size " + std::to_string(hd));
    if (merge != 2 || side * side != n_pos || D % 64 || out != text_hidden || tps != 2)
      fail("unsupported vision geometry");
  }
};

// ---------------------------------------------------------------- MLX
inline Config config_from_mlx(const json::Value& root) {
  const json::Value* tcp = root.get("text_config");
  const json::Value& tc = tcp && tcp->is_object() ? *tcp : root;
  Config c;
  c.D = int(tc.get_int("hidden_size", 0));
  c.I = int(tc.get_int("intermediate_size", 0));
  c.E = int(tc.get_int("num_experts", 0));
  if (c.E) {
    c.k = int(tc.get_int("num_experts_per_tok", 0));
    c.Ie = int(tc.get_int("moe_intermediate_size", 0));
    c.I_shared = int(tc.get_int("shared_expert_intermediate_size", 0));
  }
  c.L = int(tc.get_int("num_hidden_layers", 0));
  c.H = int(tc.get_int("num_attention_heads", 0));
  c.Hkv = int(tc.get_int("num_key_value_heads", 0));
  c.HD = int(tc.get_int("head_dim", 256));
  c.Hk = int(tc.get_int("linear_num_key_heads", 0));
  c.Hv = int(tc.get_int("linear_num_value_heads", 0));
  c.Dk = int(tc.get_int("linear_key_head_dim", 128));
  c.Dv = int(tc.get_int("linear_value_head_dim", 128));
  c.conv_k = int(tc.get_int("linear_conv_kernel_dim", 4));
  c.V = int(tc.get_int("vocab_size", 0));
  c.eps = float(tc.get_double("rms_norm_eps", 1e-6));
  const json::Value* rp = tc.get("rope_parameters");
  if (!rp || !rp->is_object()) rp = tc.get("rope_scaling");
  if (rp && !rp->is_object()) rp = nullptr;
  c.theta = rp ? rp->get_double("rope_theta", tc.get_double("rope_theta", 1e7)) : tc.get_double("rope_theta", 1e7);
  const double prf = rp ? rp->get_double("partial_rotary_factor", tc.get_double("partial_rotary_factor", 0.25))
                        : tc.get_double("partial_rotary_factor", 0.25);
  c.rot = int(c.HD * prf);
  if (rp) {
    if (const json::Value* sec = rp->get("mrope_section"); sec && sec->is_array()) {
      std::vector<int64_t> s;
      for (const auto& n : sec->as_array()) s.push_back(n.as_int());
      c.set_mrope(s, rp->get_bool("mrope_interleaved", false));
    }
  }
  const json::Value* tie = tc.get("tie_word_embeddings");
  if (!tie || tie->is_null()) tie = root.get("tie_word_embeddings");
  c.tie = tie && tie->is_bool() ? tie->as_bool() : false;
  c.set_layers(int(tc.get_int("full_attention_interval", 4)), tc.get("layer_types"));
  if (const json::Value* e = tc.get("eos_token_id"); e && e->is_int()) c.eos_token_id = int(e->as_int());
  const std::string mt = tc.get_string("model_type", root.get_string("model_type", ""));
  if (mt.rfind("qwen4_exp", 0) == 0) {
    c.q4 = true;
    c.hc = int(tc.get_int("hc_count", 4));
    c.hc_rank = int(tc.get_int("hc_lowrank", 0));
    c.idx_heads = int(tc.get_int("indexer_n_heads", 0));
    c.idx_hd = int(tc.get_int("indexer_head_dim", 0));
    c.idx_kv = int(tc.get_int("indexer_kv_heads", 0));
    c.idx_ratio = int(tc.get_int("indexer_compress_ratio", 0));
    c.idx_budget = int(tc.get_int("indexer_budget", 0));
    if (const json::Value* p = tc.get("ple_layer_ids"); p && p->is_array() && p->size() == 1)
      c.ple_layer = int((*p)[0].as_int()) - 1;  // one-indexed in the config
    c.ngram = int(tc.get_int("ngram_size", 0));
    c.ngram_heads = int(tc.get_int("heads_per_ngram", 0));
    c.ple_conv_k = int(tc.get_int("ple_conv_kernel_size", 0));
    c.gate_sigmoid = tc.get_string("output_gate_type", "silu") == "sigmoid";
    // FLA l2norm: x / sqrt(sum(x^2) + 1e-6) = x * rsqrt(mean(x^2) + 1e-6 / Dk) / sqrt(Dk);
    // the kernels compute x * rsqrt(mean + eps) and fold the 1/sqrt(Dk) query scale.
    c.qk_eps = 1e-6f / float(c.Dk);
  }
  return c;
}

// Gemma 4: config.json of a gemma4 (text_config) or gemma4_text
// checkpoint, read as mlx-lm 0.32's gemma4_text.ModelArgs reads it (its
// defaults where a key is missing). Per-layer input embeddings and KV-shared
// layers (the E2B/E4B models) are refused here: the decoder does not run them.
inline bool is_gemma4_type(const std::string& t) { return t == "gemma4" || t == "gemma4_text"; }

inline Config config_from_gemma(const json::Value& root) {
  const json::Value* tcp = root.get("text_config");
  const json::Value& tc = tcp && tcp->is_object() ? *tcp : root;
  Config c;
  c.gemma = true;
  c.tie = true;
  c.D = int(tc.get_int("hidden_size", 1536));
  c.I = int(tc.get_int("intermediate_size", 6144));
  c.L = int(tc.get_int("num_hidden_layers", 35));
  c.H = int(tc.get_int("num_attention_heads", 8));
  c.Hkv = int(tc.get_int("num_key_value_heads", 1));
  c.HD = int(tc.get_int("head_dim", 256));
  // global_head_dim: 512 when absent (the ModelArgs default), head_dim when null or 0.
  const json::Value* ghv = tc.get("global_head_dim");
  const int64_t ghd = !ghv ? 512 : (ghv->is_number() ? int64_t(ghv->as_double()) : 0);
  c.HDg = ghd > 0 ? int(ghd) : c.HD;
  c.k_eq_v = tc.get_bool("attention_k_eq_v", false);
  const json::Value* gkv = tc.get("num_global_key_value_heads");
  c.Hkv_g = c.k_eq_v && gkv && gkv->is_int() ? int(gkv->as_int()) : c.Hkv;
  // The vocabulary: the top level's (gemma4.ModelArgs copies it into text_config).
  c.V = int(root.get_int("vocab_size", tc.get_int("vocab_size", 262144)));
  c.eps = float(tc.get_double("rms_norm_eps", 1e-6));
  c.window = int(tc.get_int("sliding_window", 512));
  c.Hk = c.Hv = 0;  // no linear-attention layers
  if (const json::Value* fc = tc.get("final_logit_softcapping"); fc && fc->is_number()) c.softcap = float(fc->as_double());
  // mlx-lm multiplies the bf16 embedding by hidden_size ** 0.5 (a Python float).
  c.embed_scale = float(std::sqrt(double(c.D)));
  if (const json::Value* t = tc.get("tie_word_embeddings"); t && t->is_bool()) c.tie = t->as_bool();
  else if (const json::Value* t2 = root.get("tie_word_embeddings"); t2 && t2->is_bool()) c.tie = t2->as_bool();
  const std::string act = tc.get_string("hidden_activation", "gelu_pytorch_tanh");
  if (act != "gelu_pytorch_tanh") fail("unsupported Gemma 4 activation " + act);
  if (tc.get_int("hidden_size_per_layer_input", 0) > 0)
    fail("Gemma 4 models with per-layer input embeddings (E2B, E4B) are not supported yet");
  if (tc.get_int("num_kv_shared_layers", 0) > 0) fail("Gemma 4 models with KV-shared layers (E2B, E4B) are not supported yet");
  if (tc.get_bool("enable_moe_block", false)) {
    c.E = int(tc.get_int("num_experts", 0));
    c.k = int(tc.get_int("top_k_experts", 0));
    c.Ie = int(tc.get_int("moe_intermediate_size", 0));
  }
  // RoPE: sliding layers rotate every dim ("default": mlx-lm ignores any partial
  // factor there); global layers use the proportional RoPE of the first
  // partial_rotary_factor * head_dim dims (rope_utils.ProportionalRoPE).
  double theta_s = 10000.0, theta_g = 1e6, prf_g = 0.25, factor_g = 1.0;
  if (const json::Value* rp = tc.get("rope_parameters"); rp && rp->is_object()) {
    auto rope_of = [&](const char* key, bool global) {
      const json::Value* r = rp->get(key);
      if (!r || !r->is_object()) return;
      const std::string type = r->get_string("rope_type", r->get_string("type", "default"));
      if (global) {
        if (type != "proportional" && type != "default") fail("unsupported Gemma 4 global RoPE type " + type);
        theta_g = r->get_double("rope_theta", 10000.0);
        prf_g = type == "proportional" ? r->get_double("partial_rotary_factor", 1.0) : 1.0;
        factor_g = type == "proportional" ? r->get_double("factor", 1.0) : 1.0;
      } else {
        if (type != "default") fail("unsupported Gemma 4 sliding RoPE type " + type);
        theta_s = r->get_double("rope_theta", 10000.0);
      }
    };
    rope_of("sliding_attention", false);
    rope_of("full_attention", true);
  }
  c.theta = theta_s;
  c.theta_g = theta_g;
  c.rope_factor_g = factor_g;
  c.rot = c.HD;
  c.rot_g = int(double(c.HDg) * prf_g) / 2 * 2;
  // Layer types: the config's list, else the pattern (sliding_window_pattern - 1 sliding, then one global).
  const json::Value* types = tc.get("layer_types");
  const int pattern = int(tc.get_int("sliding_window_pattern", 5));
  c.linear.assign(size_t(c.L), false);
  c.swa.clear();
  c.n_full = c.n_swa = 0;
  for (int i = 0; i < c.L; ++i) {
    bool s;
    if (types && types->is_array() && size_t(i) < types->size()) {
      const std::string t = (*types)[size_t(i)].as_string();
      if (t != "sliding_attention" && t != "full_attention") fail("unsupported Gemma 4 layer type " + t);
      s = t == "sliding_attention";
    } else {
      s = pattern <= 0 || (i % pattern) != pattern - 1;
    }
    c.swa.push_back(s);
    s ? ++c.n_swa : ++c.n_full;
  }
  // End tokens: config.json's eos_token_id (a list for the instruction-tuned models) and text_config's.
  auto add_eos = [&](const json::Value* e) {
    if (!e) return;
    if (e->is_int()) c.eos_ids.push_back(int(e->as_int()));
    else if (e->is_array())
      for (const auto& x : e->as_array())
        if (x.is_int()) c.eos_ids.push_back(int(x.as_int()));
  };
  add_eos(root.get("eos_token_id"));
  add_eos(tc.get("eos_token_id"));
  if (const json::Value* gc = root.get("generation_config"); gc && gc->is_object()) add_eos(gc->get("eos_token_id"));
  std::sort(c.eos_ids.begin(), c.eos_ids.end());
  c.eos_ids.erase(std::unique(c.eos_ids.begin(), c.eos_ids.end()), c.eos_ids.end());
  c.eos_token_id = int(tc.get_int("eos_token_id", 1));
  return c;
}

inline VisionConfig vision_from_mlx(const json::Value& root) {
  VisionConfig v;
  const json::Value* vcp = root.get("vision_config");
  if (!vcp || !vcp->is_object()) return v;
  const json::Value& vc = *vcp;
  if (const json::Value* ds = vc.get("deepstack_visual_indexes"); ds && ds->is_array() && ds->size())
    fail("vision towers with deepstack features are not supported");
  const std::string act = vc.get_string("hidden_act", "gelu_pytorch_tanh");
  if (act != "gelu_pytorch_tanh") fail("unsupported vision activation " + act);
  v.depth = int(vc.get_int("depth", 0));
  v.D = int(vc.get_int("hidden_size", 0));
  v.I = int(vc.get_int("intermediate_size", 0));
  v.heads = int(vc.get_int("num_heads", 0));
  v.out = int(vc.get_int("out_hidden_size", 0));
  v.patch = int(vc.get_int("patch_size", 16));
  v.tps = int(vc.get_int("temporal_patch_size", 2));
  v.merge = int(vc.get_int("spatial_merge_size", 2));
  v.in_ch = int(vc.get_int("in_channels", 3));
  v.n_pos = int(vc.get_int("num_position_embeddings", 0));
  v.present = v.depth > 0;
  return v;
}

// ---------------------------------------------------------------- GGUF
inline Config config_from_gguf(const GGUFFile& f) {
  const std::string arch = f.get_string("general.architecture", "qwen35");
  const std::string a = arch + ".";
  Config c;
  c.D = int(f.get_int(a + "embedding_length", 0));
  c.I = int(f.get_int(a + "feed_forward_length", 0));
  if (arch == "qwen35moe") {
    c.E = int(f.get_int(a + "expert_count", 0));
    c.k = int(f.get_int(a + "expert_used_count", 0));
    c.Ie = int(f.get_int(a + "expert_feed_forward_length", 0));
    c.I_shared = int(f.get_int(a + "expert_shared_feed_forward_length", 0));
  }
  // Trailing next-token-prediction (MTP) blocks are not decoder layers.
  c.L = int(f.get_int(a + "block_count", 0) - f.get_int(a + "nextn_predict_layers", 0));
  c.H = int(f.get_int(a + "attention.head_count", 0));
  c.Hkv = int(f.get_int(a + "attention.head_count_kv", 0));
  c.HD = int(f.get_int(a + "attention.key_length", 256));
  c.eps = float(f.get_float(a + "attention.layer_norm_rms_epsilon", 1e-6));
  c.theta = f.get_float(a + "rope.freq_base", 1e7);
  c.rot = int(f.get_int(a + "rope.dimension_count", 64));
  c.conv_k = int(f.get_int(a + "ssm.conv_kernel", 4));
  const int state = int(f.get_int(a + "ssm.state_size", 128));
  c.Dk = c.Dv = state;
  c.Hk = int(f.get_int(a + "ssm.group_count", 0));
  c.Hv = int(f.get_int(a + "ssm.time_step_rank", 0));
  if (state && f.get_int(a + "ssm.inner_size", 0) != int64_t(c.Hv) * state)
    fail("GGUF: ssm.inner_size does not match value heads x state size");
  const GGUFFile::Tensor* emb = f.find("token_embd.weight");
  if (!emb || emb->dims.size() != 2) fail("GGUF: token_embd.weight missing");
  c.V = int(emb->dims[1]);
  c.tie = f.find("output.weight") == nullptr;
  c.tiled_heads = true;
  // llama.cpp runs qwen35 with interleaved M-RoPE (imrope).
  c.set_mrope(f.get_ints(a + "rope.dimension_sections"), true);
  c.set_layers(int(f.get_int(a + "full_attention_interval", 4)), nullptr);
  c.eos_token_id = int(f.get_int("tokenizer.ggml.eos_token_id", -1));
  return c;
}

// Gemma 4 from llama.cpp's gemma4 GGUF metadata: per-layer KV head
// counts and sliding flags, the two head sizes and RoPE bases, and the
// global layers' rope_freqs factors (1 for rotated pairs, 1e30 for the
// others: their count gives the proportional RoPE's rotated dims).
inline Config config_from_gemma_gguf(const GGUFFile& f) {
  const std::string a = "gemma4.";
  Config c;
  c.gemma = true;
  c.tie = f.find("output.weight") == nullptr;
  c.D = int(f.get_int(a + "embedding_length", 0));
  c.I = int(f.get_int(a + "feed_forward_length", 0));
  c.L = int(f.get_int(a + "block_count", 0));
  c.H = int(f.get_int(a + "attention.head_count", 0));
  c.HD = int(f.get_int(a + "attention.key_length_swa", 256));
  c.HDg = int(f.get_int(a + "attention.key_length", 512));
  if (f.get_int(a + "attention.value_length_swa", c.HD) != c.HD || f.get_int(a + "attention.value_length", c.HDg) != c.HDg)
    fail("GGUF gemma4: value heads of another width than the keys are not supported");
  c.eps = float(f.get_float(a + "attention.layer_norm_rms_epsilon", 1e-6));
  c.window = int(f.get_int(a + "attention.sliding_window", 0));
  c.theta = f.get_float(a + "rope.freq_base_swa", 1e4);
  c.theta_g = f.get_float(a + "rope.freq_base", 1e6);
  c.softcap = float(f.get_float(a + "final_logit_softcapping", 0.0));
  c.embed_scale = float(std::sqrt(double(c.D)));
  if (f.get_int(a + "embedding_length_per_layer_input", 0) > 0 || f.get_int(a + "attention.shared_kv_layers", 0) > 0)
    fail("Gemma 4 models with per-layer input embeddings or KV-shared layers (E2B, E4B) are not supported yet");
  c.E = int(f.get_int(a + "expert_count", 0));
  if (c.E) {
    c.k = int(f.get_int(a + "expert_used_count", 0));
    c.Ie = int(f.get_int(a + "expert_feed_forward_length", 0));
  }
  const std::vector<int64_t> kv = f.get_ints(a + "attention.head_count_kv");
  const std::vector<int64_t> pattern = f.get_ints(a + "attention.sliding_window_pattern");
  if (int(pattern.size()) != c.L) fail("GGUF gemma4: attention.sliding_window_pattern does not list every layer");
  c.linear.assign(size_t(c.L), false);
  c.n_full = c.n_swa = 0;
  for (int i = 0; i < c.L; ++i) {
    const bool s = pattern[size_t(i)] != 0;
    c.swa.push_back(s);
    s ? ++c.n_swa : ++c.n_full;
    const int h = kv.size() == size_t(c.L) ? int(kv[size_t(i)]) : int(f.get_int(a + "attention.head_count_kv", 0));
    int& want = s ? c.Hkv : c.Hkv_g;
    if (want && want != h) fail("GGUF gemma4: KV head counts differ between layers of one kind");
    want = h;
  }
  // K = V: the global layers have no value projection.
  for (int i = 0; i < c.L; ++i)
    if (!c.swa[size_t(i)]) {
      c.k_eq_v = f.find("blk." + std::to_string(i) + ".attn_v.weight") == nullptr;
      break;
    }
  c.rot = c.HD;
  c.rot_g = c.HDg;
  if (const GGUFFile::Tensor* rf = f.find("rope_freqs.weight"); rf && rf->type == 0 && rf->dims.size() == 1) {
    const float* x = reinterpret_cast<const float*>(f.data(*rf));
    int n = 0;
    while (n < int(rf->dims[0]) && x[n] < 1e10f) ++n;
    for (int i = n; i < int(rf->dims[0]); ++i)
      if (x[i] < 1e10f) fail("GGUF gemma4: rope_freqs rotates pairs after an unrotated one");
    for (int i = 0; i < n; ++i)
      if (x[i] != 1.0f) fail("GGUF gemma4: rope_freqs scales a rotated pair (not the proportional RoPE)");
    c.rot_g = 2 * n;
  }
  const GGUFFile::Tensor* emb = f.find("token_embd.weight");
  if (!emb || emb->dims.size() != 2) fail("GGUF: token_embd.weight missing");
  c.V = int(emb->dims[1]);
  c.eos_token_id = int(f.get_int("tokenizer.ggml.eos_token_id", -1));
  // End of turn ids as the HF configs list them (the Gemma 4 vocabulary's
  // <eos>, <turn|> and <|tool_response>, which ends a turn that wrote a call).
  c.eos_ids = {1, 106, 50};
  if (c.eos_token_id >= 0 && std::find(c.eos_ids.begin(), c.eos_ids.end(), c.eos_token_id) == c.eos_ids.end())
    c.eos_ids.push_back(c.eos_token_id);
  std::sort(c.eos_ids.begin(), c.eos_ids.end());
  return c;
}

inline VisionConfig vision_from_gguf(const GGUFFile& f) {
  VisionConfig v;
  if (!f.get_int("clip.has_vision_encoder", 0)) return v;
  for (int64_t flag : f.get_ints("clip.vision.is_deepstack_layers"))
    if (flag) fail("mmproj with deepstack features is not supported");
  v.depth = int(f.get_int("clip.vision.block_count", 0));
  v.D = int(f.get_int("clip.vision.embedding_length", 0));
  v.I = int(f.get_int("clip.vision.feed_forward_length", 0));
  v.heads = int(f.get_int("clip.vision.attention.head_count", 0));
  v.out = int(f.get_int("clip.vision.projection_dim", 0));
  v.patch = int(f.get_int("clip.vision.patch_size", 16));
  v.merge = int(f.get_int("clip.vision.spatial_merge_size", 2));
  v.eps = float(f.get_float("clip.vision.attention.layer_norm_epsilon", 1e-6));
  v.tps = f.find("v.patch_embd.weight.1") ? 2 : 1;
  const GGUFFile::Tensor* pos = f.find("v.position_embd.weight");
  if (!pos || pos->dims.size() != 2) fail("mmproj: position embedding missing");
  v.n_pos = int(pos->dims[1]);
  if (!f.get_int("clip.use_gelu", 1)) fail("mmproj: unsupported vision activation");
  v.present = v.depth > 0;
  return v;
}

}  // namespace espetero::native
