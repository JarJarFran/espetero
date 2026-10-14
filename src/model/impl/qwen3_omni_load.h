// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Qwen3-Omni thinker checkpoints (`qwen3_omni_moe`): the text decoder and
// the vision encoder of a directory written by tools/clef_convert.py (the
// thinker's text_config / vision_config at the top of config.json, mlx-vlm's
// qwen3_omni_moe names: language_model.model.*, vision_tower.*). The audio
// encoder is not converted; images and text only. The forward pass is in
// qwen3_omni.h; the semantics follow transformers' models/qwen3_omni_moe
// (Apache 2.0, see THIRD_PARTY_NOTICES.md).
#pragma once

namespace espetero::native {

inline bool is_qwen3_omni_type(const std::string& t) { return t == "qwen3_omni_moe" || t == "qwen3_omni_moe_text"; }

// text_config of the thinker, read with transformers' Qwen3OmniMoeTextConfig
// defaults where a key is missing. Every layer is MoE (decoder_sparse_step 1,
// no mlp_only_layers) and the router renormalizes its top k (norm_topk_prob):
// anything else fails here instead of running another model.
inline Config config_from_qwen3_omni(const json::Value& root) {
  const json::Value* tcp = root.get("text_config");
  const json::Value& tc = tcp && tcp->is_object() ? *tcp : root;
  Config c;
  c.q3moe = true;
  c.D = int(tc.get_int("hidden_size", 2048));
  c.I = int(tc.get_int("intermediate_size", 0));
  c.L = int(tc.get_int("num_hidden_layers", 48));
  c.H = int(tc.get_int("num_attention_heads", 32));
  c.Hkv = int(tc.get_int("num_key_value_heads", 4));
  c.HD = int(tc.get_int("head_dim", c.H ? c.D / c.H : 0));
  c.V = int(tc.get_int("vocab_size", 0));
  c.eps = float(tc.get_double("rms_norm_eps", 1e-6));
  c.E = int(tc.get_int("num_experts", 0));
  c.k = int(tc.get_int("num_experts_per_tok", 0));
  c.Ie = int(tc.get_int("moe_intermediate_size", 0));
  c.I_shared = 0;
  c.Hk = c.Hv = 0;  // no linear-attention layers
  c.rot = c.HD;     // RoPE over every dim of the head
  const json::Value* rp = tc.get("rope_parameters");
  if (!rp || !rp->is_object()) rp = tc.get("rope_scaling");
  if (rp && !rp->is_object()) rp = nullptr;
  c.theta = rp ? rp->get_double("rope_theta", tc.get_double("rope_theta", 1e6)) : tc.get_double("rope_theta", 1e6);
  if (rp) {
    const std::string type = rp->get_string("rope_type", rp->get_string("type", "default"));
    if (type != "default") fail("qwen3_omni_moe: unsupported RoPE type " + type);
  }
  const json::Value* tie = tc.get("tie_word_embeddings");
  if (!tie || tie->is_null()) tie = root.get("tie_word_embeddings");
  c.tie = tie && tie->is_bool() ? tie->as_bool() : false;
  c.linear.assign(size_t(c.L), false);
  c.n_full = c.L;
  c.n_linear = 0;
  if (const json::Value* e = tc.get("eos_token_id"); e && e->is_int()) c.eos_token_id = int(e->as_int());
  if (!tc.get_bool("norm_topk_prob", true)) fail("qwen3_omni_moe: routers without norm_topk_prob are not supported");
  if (tc.get_int("decoder_sparse_step", 1) != 1) fail("qwen3_omni_moe: dense decoder layers (decoder_sparse_step > 1) are not supported");
  if (const json::Value* mo = tc.get("mlp_only_layers"); mo && mo->is_array() && mo->size())
    fail("qwen3_omni_moe: dense decoder layers (mlp_only_layers) are not supported");
  if (tc.get_bool("attention_bias", false)) fail("qwen3_omni_moe: attention biases are not supported");
  if (!tc.get_bool("use_qk_norm", true)) fail("qwen3_omni_moe: attention without q/k norms is not supported");
  if (tc.get_bool("use_sliding_window", false)) fail("qwen3_omni_moe: sliding-window attention is not supported");
  return c;
}

// The kernels exist for 128-dim heads with GQA groups up to 8 (the Gemma 4
// attention instantiations of 82_qwen3_omni.metal).
inline void validate_qwen3_omni(const Config& c) {
  if (!c.D || !c.L || !c.H || !c.Hkv || !c.V || !c.E || !c.k || !c.Ie) fail("incomplete qwen3_omni_moe configuration");
  if (c.HD != 128) fail("qwen3_omni_moe: unsupported head dim " + std::to_string(c.HD) + " (128)");
  if (c.H % c.Hkv) fail("qwen3_omni_moe: unsupported head grouping");
  const int G = c.H / c.Hkv;
  if (G != 1 && G != 2 && G != 4 && G != 8) fail("qwen3_omni_moe: unsupported GQA group " + std::to_string(G));
  if (c.D % 64 || c.Ie % 64 || c.E > 1024 || c.k > 16)
    fail("qwen3_omni_moe: unsupported geometry (hidden and expert widths multiples of 64, experts <= 1024, k <= 16)");
}

// M-RoPE tables: inv_freq[d] = theta^(-2d / HD) for the HD / 2 pairs (d, d + HD/2)
// and the position component of each (transformers' interleaved sections:
// frequency d takes h when d % 3 == 1 and d < 3 s_h, w when d % 3 == 2 and
// d < 3 s_w, else t).
inline void qwen3_omni_rope_tables(Model& m, const json::Value& root) {
  const Config& c = m.cfg;
  const json::Value* tcp = root.get("text_config");
  const json::Value& tc = tcp && tcp->is_object() ? *tcp : root;
  const json::Value* rp = tc.get("rope_parameters");
  if (!rp || !rp->is_object()) rp = tc.get("rope_scaling");
  std::vector<int64_t> sec = {24, 20, 20};  // Qwen3OmniMoeThinkerTextRotaryEmbedding's default
  bool interleaved = true;
  if (rp && rp->is_object()) {
    if (const json::Value* s = rp->get("mrope_section"); s && s->is_array()) {
      sec.clear();
      for (const auto& n : s->as_array()) sec.push_back(n.as_int());
    }
    interleaved = rp->get_bool("mrope_interleaved", rp->get_bool("interleaved", true));
  }
  if (sec.size() != 3 || sec[0] + sec[1] + sec[2] != c.HD / 2)
    fail("qwen3_omni_moe: mrope_section must split the head's " + std::to_string(c.HD / 2) + " frequencies in three");
  if (!interleaved) fail("qwen3_omni_moe: only interleaved M-RoPE sections are supported");
  std::vector<float> f(size_t(c.HD / 2));
  std::vector<int32_t> sel(size_t(c.HD / 2));
  for (int d = 0; d < c.HD / 2; ++d) {
    f[size_t(d)] = float(1.0 / std::pow(c.theta, double(2 * d) / double(c.HD)));
    sel[size_t(d)] = (d % 3 == 1 && d < 3 * sec[1]) ? 1 : ((d % 3 == 2 && d < 3 * sec[2]) ? 2 : 0);
  }
  m.inv_freq = m.arena->copy(f.data(), f.size() * 4);
  m.rope_sel = m.arena->copy(sel.data(), sel.size() * 4);
}

// One decoder layer (prefix `p`, e.g. "language_model.model.layers.3.").
inline Layer load_qwen3_omni_layer(Model& m, const WeightSource& src, const std::string& p, int li) {
  const Config& c = m.cfg;
  Layer L;
  L.linear = false;
  L.index_in_kind = li;
  L.in_norm = vec(m, src, p + "input_layernorm.weight", c.D);
  L.post_norm = vec(m, src, p + "post_attention_layernorm.weight", c.D);
  const std::string a = p + "self_attn.";
  L.q_norm = vec(m, src, a + "q_norm.weight", size_t(c.HD));
  L.k_norm = vec(m, src, a + "k_norm.weight", size_t(c.HD));
  {
    Arena tmp(m.dev);
    const std::vector<HostMat> parts = {host_mat(m, tmp, src, a + "q_proj.weight", c.H * c.HD, c.D),
                                        host_mat(m, tmp, src, a + "k_proj.weight", c.kv_dim(), c.D),
                                        host_mat(m, tmp, src, a + "v_proj.weight", c.kv_dim(), c.D)};
    HostMat out = host_mat(m, tmp, src, a + "o_proj.weight", c.D, c.H * c.HD);
    const std::vector<HostMat> runs = fuse(tmp, parts);
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    auto dn = [&](const char* k) { return m.dense_mats.count(k) > 0; };
    L.out = finish(m, e, tmp, out, dn("out"));
    for (const HostMat& r : runs) L.in_proj.push_back(finish(m, e, tmp, r, dn("in_proj")));
    [e endEncoding];
    commit_wait(cb);
  }
  // The experts (MLX switch_mlp tensors [E, out, in], read as [E * out, in])
  // and the router, which stays dense when the checkpoint keeps it in bf16.
  if (!src.has(p + "mlp.switch_mlp.gate_proj.weight"))
    fail(p + "mlp.switch_mlp: expected stacked experts (convert the release with tools/clef_convert.py)");
  Arena tmp(m.dev);
  HostMat router = host_mat(m, tmp, src, p + "mlp.gate.weight", c.E, c.D);
  HostMat eg = host_mat(m, tmp, src, p + "mlp.switch_mlp.gate_proj.weight", c.E * c.Ie, c.D);
  HostMat eu = host_mat(m, tmp, src, p + "mlp.switch_mlp.up_proj.weight", c.E * c.Ie, c.D);
  HostMat ed = host_mat(m, tmp, src, p + "mlp.switch_mlp.down_proj.weight", c.E * c.D, c.Ie);
  if (eg.dense || eu.dense || ed.dense) fail("dense expert weights are not supported (quantize them: tools/clef_convert.py --bits 8)");
  const HostMat egu = interleave_gate_up(tmp, eg, eu);  // every expert's 2 Ie rows stay together (Ie % 64 == 0)
  id<MTLCommandBuffer> cb = [m.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  L.router = finish(m, e, tmp, router);
  if (L.router.dense) ++m.n_dense_mtp;  // routers never take a fused-norm link
  L.experts_gu = finish(m, e, tmp, egu).q;
  L.experts_down = finish(m, e, tmp, ed).q;
  [e endEncoding];
  commit_wait(cb);
  const std::vector<float> ones(size_t(c.E), 1.0f);  // g4_route's per-expert output scales
  L.expert_scale = m.arena->copy(ones.data(), ones.size() * 4);
  L.moe = true;
  return L;
}

inline void load_qwen3_omni(Model& m, const SafetensorsSource& src, bool want_vision) {
  Config& c = m.cfg;
  const std::string pre = "language_model.model.";
  if (!src.has(pre + "embed_tokens.weight"))
    fail("expected a converted Qwen3-Omni thinker (language_model.model.* tensors): run tools/clef_convert.py on the release");
  validate_qwen3_omni(c);
  {
    Arena tmp(m.dev);
    load_embedding(m, tmp, src, pre + "embed_tokens.weight");
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    const std::string head = c.tie ? pre + "embed_tokens.weight" : "language_model.lm_head.weight";
    m.lm_head = finish(m, e, tmp, host_mat(m, tmp, src, head, c.V, c.D));
    [e endEncoding];
    commit_wait(cb);
  }
  m.final_norm = vec(m, src, pre + "norm.weight", c.D);
  for (int i = 0; i < c.L; ++i) m.layers.push_back(load_qwen3_omni_layer(m, src, pre + "layers." + std::to_string(i) + ".", i));
  qwen3_omni_rope_tables(m, src.config());
  if (want_vision && src.has("vision_tower.patch_embed.proj.weight")) {
    const VisionConfig vc = vision_from_mlx(src.config(), true);
    if (vc.present) load_vision(m, src, vc);
  }
}

}  // namespace espetero::native
