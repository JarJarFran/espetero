// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// GLM-5.3-Flash (glm5_next, zai-org): the configuration and the checkpoint
// index the loader works from (docs/glm5.md). The model itself (glm5_model.h,
// a model of its own beside DeepSeek-V4.1's) comes with its kernels; this
// header is what the loader has to know without them: the geometry from
// config.json, which tensors every layer has with their dtypes and shapes,
// and where each routed expert's bytes are in the shards (the expert cache
// reads them from the SSD as DeepSeek-V4.1's does: docs/glm5.md 8).
//
// Checkpoint facts it relies on (docs/glm5.md 0): FP8 e4m3 matrices with one
// fp32 scale per 128 x 128 block (`.weight_scale_inv` [N/128, K/128]) for the
// MLPs and the DSA projections, bf16 for the rest; the shards group tensors by
// dtype, so an expert's three code tensors are contiguous (down | gate | up)
// and so are its three scale tensors.
#pragma once

#include "dsv4.h"  // ds4::Ckpt: the safetensors index, pread; TInfo

namespace espetero::native::glm5 {

// ---------------------------------------------------------------- config
struct Cfg {
  int D = 4096, L = 45, V = 154880;
  int n_mtp = 1;  // num_nextn_predict_layers: layers L .. L + n_mtp - 1 of the file
  float eps = 1e-5f;
  int64_t max_pos = 1048576;
  // KDA (linear_attn_config)
  int kH = 64, kD = 128, conv = 4;
  float gate_lb = -5.0f;
  // DSA: nope-only MLA and the k-pool indexer
  int H = 64, QR = 1536, KVR = 512, NOPE = 256, ROPE = 0, VD = 256;
  int IH = 32, IHD = 128, ITOPK = 2048, KPOOL = 4;
  bool tail = true;
  // MLP: dense layers, MoE, shared expert
  int I_dense = 12288, IE = 2048, E = 288, K = 8, NSH = 1;
  int n_group = 1, topk_group = 1;
  float route_scale = 2.5f, swiglu = 10.0f;
  bool norm_topk = true;
  std::string score_func = "sigmoid", topk_method = "noaux_tc";
  // mHC
  int hc = 4, hc_iters = 20;
  float hc_eps = 1e-6f;
  // per layer 0 .. L - 1
  std::vector<char> is_dsa, is_moe, idx_full;
  std::vector<int> eos;
  int pad = -1, image_token = -1, video_token = -1;
  int block = 128;  // fp8 weight_block_size (square)

  int n_kda() const { return int(std::count(is_dsa.begin(), is_dsa.end(), 0)); }
  int n_dsa() const { return L - n_kda(); }
  int n_moe() const { return int(std::count(is_moe.begin(), is_moe.end(), 1)); }
  bool dsa(int l) const { return l >= L || is_dsa[size_t(l)]; }  // the MTP layer is a DSA layer
  bool moe(int l) const { return l >= L || is_moe[size_t(l)]; }
  bool mtp_layer(int l) const { return l >= L; }
  int layers_in_file() const { return L + n_mtp; }
};

inline Cfg read_cfg(const json::Value& root) {
  const json::Value* tcp = root.get("text_config");
  const json::Value& tc = tcp && tcp->is_object() ? *tcp : root;
  Cfg c;
  auto strs = [&](const char* k) {
    std::vector<std::string> v;
    if (const json::Value* a = tc.get(k); a && a->is_array())
      for (const json::Value& x : a->as_array()) v.push_back(x.is_string() ? x.as_string() : "");
    return v;
  };
  c.D = int(tc.get_int("hidden_size", c.D));
  c.L = int(tc.get_int("num_hidden_layers", c.L));
  c.V = int(tc.get_int("vocab_size", c.V));
  c.n_mtp = int(tc.get_int("num_nextn_predict_layers", c.n_mtp));
  c.eps = float(tc.get_double("rms_norm_eps", c.eps));
  c.max_pos = tc.get_int("max_position_embeddings", c.max_pos);
  if (const json::Value* la = tc.get("linear_attn_config"); la && la->is_object()) {
    c.kH = int(la->get_int("num_heads", c.kH));
    c.kD = int(la->get_int("head_dim", c.kD));
    c.conv = int(la->get_int("short_conv_kernel_size", c.conv));
    c.gate_lb = float(la->get_double("gate_lower_bound", c.gate_lb));
  }
  c.H = int(tc.get_int("num_attention_heads", c.H));
  c.QR = int(tc.get_int("q_lora_rank", c.QR));
  c.KVR = int(tc.get_int("kv_lora_rank", c.KVR));
  c.NOPE = int(tc.get_int("qk_nope_head_dim", c.NOPE));
  c.ROPE = int(tc.get_int("qk_rope_head_dim", c.ROPE));
  c.VD = int(tc.get_int("v_head_dim", c.VD));
  c.IH = int(tc.get_int("index_n_heads", c.IH));
  c.IHD = int(tc.get_int("index_head_dim", c.IHD));
  c.ITOPK = int(tc.get_int("index_topk", c.ITOPK));
  c.KPOOL = int(tc.get_int("index_kpool", c.KPOOL));
  c.tail = tc.get_bool("index_kpool_always_select_tail", c.tail);
  c.I_dense = int(tc.get_int("intermediate_size", c.I_dense));
  c.IE = int(tc.get_int("moe_intermediate_size", c.IE));
  c.E = int(tc.get_int("n_routed_experts", c.E));
  c.K = int(tc.get_int("num_experts_per_tok", c.K));
  c.NSH = int(tc.get_int("n_shared_experts", c.NSH));
  c.n_group = int(tc.get_int("n_group", c.n_group));
  c.topk_group = int(tc.get_int("topk_group", c.topk_group));
  c.route_scale = float(tc.get_double("routed_scaling_factor", c.route_scale));
  c.swiglu = float(tc.get_double("swiglu_limit", c.swiglu));
  c.norm_topk = tc.get_bool("norm_topk_prob", c.norm_topk);
  c.score_func = tc.get_string("scoring_func", c.score_func);
  c.topk_method = tc.get_string("topk_method", c.topk_method);
  c.hc = int(tc.get_int("hc_mult", c.hc));
  c.hc_iters = int(tc.get_int("hc_sinkhorn_iters", c.hc_iters));
  c.hc_eps = float(tc.get_double("hc_eps", c.hc_eps));
  // layer schedule: layer_types (linear_attention | deepseek_sparse_attention | indexed_attention | full_attention),
  // mlp_layer_types (dense | sparse) or first_k_dense_replace, indexer_types (full | shared)
  const std::vector<std::string> lt = strs("layer_types"), mt = strs("mlp_layer_types"), it = strs("indexer_types");
  const int first_dense = int(tc.get_int("first_k_dense_replace", 3));
  c.is_dsa.assign(size_t(c.L), 0);
  c.is_moe.assign(size_t(c.L), 0);
  c.idx_full.assign(size_t(c.L), 1);
  for (int l = 0; l < c.L; ++l) {
    if (int(lt.size()) == c.L) {
      if (lt[size_t(l)] != "linear_attention") c.is_dsa[size_t(l)] = 1;
    } else {
      c.is_dsa[size_t(l)] = l % 4 == 3;  // the default schedule of [TC]
    }
    c.is_moe[size_t(l)] = int(mt.size()) == c.L ? mt[size_t(l)] == "sparse" : l >= first_dense;
    if (int(it.size()) == c.L) c.idx_full[size_t(l)] = it[size_t(l)] != "shared";
  }
  if (const json::Value* e = tc.get("eos_token_id"); e) {
    if (e->is_array())
      for (const json::Value& x : e->as_array()) c.eos.push_back(int(x.as_int()));
    else if (e->is_number())
      c.eos.push_back(int(e->as_int()));
  }
  c.pad = int(tc.get_int("pad_token_id", -1));
  c.image_token = int(root.get_int("image_token_id", -1));
  c.video_token = int(root.get_int("video_token_id", -1));
  if (const json::Value* q = root.get("quantization_config"); q && q->is_object()) {
    if (q->get_string("quant_method", "") != "fp8" || q->get_string("fmt", "e4m3") != "e4m3")
      fail("glm5_next: quantization_config is not fp8 e4m3 (" + q->get_string("quant_method", "?") + ")");
    if (const json::Value* b = q->get("weight_block_size"); b && b->is_array() && b->size() == 2) {
      if ((*b)[0].as_int() != (*b)[1].as_int()) fail("glm5_next: non-square fp8 weight_block_size");
      c.block = int((*b)[0].as_int());
    }
  }
  // what the engine will assume (docs/glm5.md 10 and 12)
  if (c.hc != 4) fail("glm5_next: hc_mult " + std::to_string(c.hc) + " (4 expected)");
  if (c.ROPE != 0) fail("glm5_next: qk_rope_head_dim " + std::to_string(c.ROPE) + " (nope-only MLA expected)");
  if (c.KPOOL < 1 || c.ITOPK % c.KPOOL) fail("glm5_next: index_topk must be a multiple of index_kpool");
  if (c.n_group != 1 || c.topk_group != 1) fail("glm5_next: expert groups are not supported (n_group 1 expected)");
  if (c.score_func != "sigmoid" || c.topk_method != "noaux_tc")
    fail("glm5_next: router " + c.score_func + " / " + c.topk_method + " (sigmoid noaux_tc expected)");
  if (c.kD != 128 || c.NOPE != 256 || c.VD != 256 || c.KVR != 512 || c.IHD != 128 || c.IH > 32 || c.K > 8 || c.E > 512 ||
      c.conv != 4)
    fail("glm5_next: unsupported geometry (KDA head 128, nope/v 256, kv_lora 512, index head 128, <= 32 index heads, "
         "<= 8 of <= 512 experts, conv 4 expected)");
  if (c.D % c.block || c.QR % c.block || c.IE % c.block || c.I_dense % c.block)
    fail("glm5_next: widths are not multiples of the fp8 block");
  for (int l = 0; l < c.L; ++l)
    if (!c.idx_full[size_t(l)] && (l == 0 || !c.is_dsa[size_t(l)]))
      fail("glm5_next: a shared-indexer layer without a full one before it");
  return c;
}

// ---------------------------------------------------------------- tensor names and shapes
// HF names: model.language_model.layers.<l>.*, lm_head.weight, model.language_model.{embed_tokens,norm}.weight.
inline std::string layer_prefix(int l) { return "model.language_model.layers." + std::to_string(l) + "."; }

struct Want {
  std::string name;
  const char* dtype;
  std::vector<int64_t> shape;
};

// Every tensor of layer l with its dtype and shape (the fp8 matrices with their scale tensors).
inline std::vector<Want> layer_tensors(const Cfg& c, int l) {
  std::vector<Want> w;
  const std::string p = layer_prefix(l);
  const int64_t D = c.D, B = c.block;
  auto bf = [&](const std::string& n, std::vector<int64_t> s) { w.push_back({p + n, "BF16", std::move(s)}); };
  auto f32 = [&](const std::string& n, std::vector<int64_t> s) { w.push_back({p + n, "F32", std::move(s)}); };
  auto f8 = [&](const std::string& n, int64_t N, int64_t K) {
    w.push_back({p + n + ".weight", "F8_E4M3", {N, K}});
    w.push_back({p + n + ".weight_scale_inv", "F32", {(N + B - 1) / B, (K + B - 1) / B}});
  };
  bf("input_layernorm.weight", {D});
  bf("post_attention_layernorm.weight", {D});
  if (!c.mtp_layer(l)) {
    for (const char* s : {"attn", "ffn"}) {
      bf(std::string("hc_") + s + "_fn", {int64_t((2 + c.hc) * c.hc), int64_t(c.hc) * D});
      f32(std::string("hc_") + s + "_base", {int64_t((2 + c.hc) * c.hc)});
      f32(std::string("hc_") + s + "_scale", {3});
    }
  }
  if (!c.dsa(l)) {  // KDA
    const int64_t Q = int64_t(c.kH) * c.kD;
    for (const char* n : {"q_proj", "k_proj", "v_proj"}) bf(std::string("self_attn.") + n + ".weight", {Q, D});
    for (const char* n : {"q_conv1d", "k_conv1d", "v_conv1d"}) bf(std::string("self_attn.") + n + ".weight", {Q, 1, c.conv});
    bf("self_attn.f_a_proj.weight", {c.kD, D});
    bf("self_attn.f_b_proj.weight", {Q, c.kD});
    bf("self_attn.g_a_proj.weight", {c.kD, D});
    bf("self_attn.g_b_proj.weight", {Q, c.kD});
    bf("self_attn.b_proj.weight", {c.kH, D});
    f32("self_attn.A_log", {c.kH});
    f32("self_attn.dt_bias", {Q});
    bf("self_attn.o_norm.weight", {c.kD});
    bf("self_attn.o_proj.weight", {D, Q});
  } else {  // DSA: nope MLA + indexer
    f8("self_attn.q_a_proj", c.QR, D);
    bf("self_attn.q_a_layernorm.weight", {c.QR});
    f8("self_attn.q_b_proj", int64_t(c.H) * (c.NOPE + c.ROPE), c.QR);
    f8("self_attn.kv_a_proj_with_mqa", c.KVR + c.ROPE, D);
    bf("self_attn.kv_a_layernorm.weight", {c.KVR});
    bf("self_attn.kv_b_proj.weight", {int64_t(c.H) * (c.NOPE + c.VD), c.KVR});
    f8("self_attn.o_proj", D, int64_t(c.H) * c.VD);
    if (c.mtp_layer(l) || c.idx_full[size_t(l)]) {
      bf("self_attn.indexer.wq_b.weight", {int64_t(c.IH) * c.IHD, c.QR});
      bf("self_attn.indexer.wk.weight", {c.IHD, D});
      bf("self_attn.indexer.k_norm.weight", {c.IHD});
      bf("self_attn.indexer.k_norm.bias", {c.IHD});
      bf("self_attn.indexer.weights_proj.weight", {c.IH, D});
      bf("self_attn.indexer.index_kpool_compress_gate", {c.IHD, D});
      bf("self_attn.indexer.index_kpool_compress_ape", {c.KPOOL, c.IHD});
    }
  }
  if (!c.moe(l)) {
    f8("mlp.gate_proj", c.I_dense, D);
    f8("mlp.up_proj", c.I_dense, D);
    f8("mlp.down_proj", D, c.I_dense);
  } else {
    bf("mlp.gate.weight", {c.E, D});
    f32("mlp.gate.e_score_correction_bias", {c.E});
    const int64_t SH = int64_t(c.IE) * c.NSH;
    f8("mlp.shared_experts.gate_proj", SH, D);
    f8("mlp.shared_experts.up_proj", SH, D);
    f8("mlp.shared_experts.down_proj", D, SH);
    // the routed experts are indexed separately (Index::experts), same shapes as a 1-wide shared expert
  }
  if (c.mtp_layer(l)) {
    bf("eh_proj.weight", {D, 2 * D});
    bf("enorm.weight", {D});
    bf("hnorm.weight", {D});
    bf("shared_head.norm.weight", {D});
  }
  return w;
}

inline std::vector<Want> top_tensors(const Cfg& c) {
  return {{"model.language_model.embed_tokens.weight", "BF16", {c.V, c.D}},
          {"lm_head.weight", "BF16", {c.V, c.D}},
          {"model.language_model.norm.weight", "BF16", {c.D}}};
}

// ---------------------------------------------------------------- checkpoint index
// A routed expert's bytes as the cache slot wants them: codes down | gate | up
// (3 x IE x D fp8), then scales down | gate | up (3 x IE/B x D/B fp32), in
// pieces of the shard files (two when the shard stores them contiguously, as
// the HF files do; more for an expert split across shards).
struct Piece {
  int file = -1;
  uint64_t off = 0, nbytes = 0, dst = 0;  // dst: byte offset in the slot
};
struct ExpertLoc {
  std::vector<Piece> pieces;
};

struct Index {
  Cfg c;
  ds4::Ckpt ck;
  int layers = 0;                // layers present (L + n_mtp, or fewer in a partial load)
  uint64_t codes_bytes = 0, scales_bytes = 0;  // per expert
  std::vector<std::vector<ExpertLoc>> experts;  // [layer][e]; empty for dense layers
  int split_experts = 0;        // experts whose bytes are not two contiguous runs
  uint64_t bytes_resident = 0;  // everything of the text model but the routed experts (and the MTP layer's)
  uint64_t bytes_experts = 0, bytes_mtp = 0, bytes_vision = 0, bytes_other = 0;

  // partial: layers whose shards are missing are skipped (development, while a download runs).
  void open(const std::string& dir, bool partial = false) {
    c = read_cfg(json::parse(read_file(dir + "/config.json")));
    ck.open(dir, partial);
    const int LF = c.layers_in_file();
    codes_bytes = 3ull * uint64_t(c.IE) * uint64_t(c.D);
    scales_bytes = 3ull * uint64_t(c.IE / c.block) * uint64_t(c.D / c.block) * 4;
    experts.assign(size_t(LF), {});
    auto check = [&](const Want& w) {
      if (!ck.has(w.name)) fail("glm5_next: tensor " + w.name + " is missing from the checkpoint");
      const ds4::TInfo& t = ck.at(w.name);
      if (t.dtype != w.dtype) fail("glm5_next: " + w.name + " is " + t.dtype + ", expected " + w.dtype);
      if (t.shape != w.shape) {
        std::string got, want;
        for (int64_t x : t.shape) got += (got.empty() ? "" : ", ") + std::to_string(x);
        for (int64_t x : w.shape) want += (want.empty() ? "" : ", ") + std::to_string(x);
        fail("glm5_next: " + w.name + " has shape [" + got + "], expected [" + want + "]");
      }
      return t.nbytes;
    };
    for (const Want& w : top_tensors(c))
      if (!partial || ck.has(w.name)) bytes_resident += check(w);
    for (int l = 0; l < LF; ++l) {
      const std::vector<Want> ws = layer_tensors(c, l);
      if (partial && !ck.has(ws[0].name)) {
        if (layers == l) layers = l;  // the first missing layer ends the prefix
        continue;
      }
      uint64_t lb = 0;
      for (const Want& w : ws) lb += check(w);
      (c.mtp_layer(l) ? bytes_mtp : bytes_resident) += lb;
      if (c.moe(l)) {
        std::vector<ExpertLoc>& xs = experts[size_t(l)];
        xs.resize(size_t(c.E));
        const std::string p = layer_prefix(l) + "mlp.experts.";
        for (int e = 0; e < c.E; ++e) {
          const std::string q = p + std::to_string(e) + ".";
          // slot order: down | gate | up (the files' alphabetical order), codes then scales
          static const char* names[3] = {"down_proj", "gate_proj", "up_proj"};
          std::vector<Piece> ps;
          uint64_t dst = 0;
          for (int part = 0; part < 2; ++part) {
            for (int m = 0; m < 3; ++m) {
              Want w{q + names[m] + (part ? ".weight_scale_inv" : ".weight"), part ? "F32" : "F8_E4M3", {}};
              const int64_t N = m == 0 ? c.D : c.IE, Kd = m == 0 ? c.IE : c.D;
              w.shape = part ? std::vector<int64_t>{(N + c.block - 1) / c.block, (Kd + c.block - 1) / c.block}
                             : std::vector<int64_t>{N, Kd};
              const uint64_t nb = check(w);
              const ds4::TInfo& t = ck.at(w.name);
              if (!ps.empty() && ps.back().file == t.file && ps.back().off + ps.back().nbytes == t.off &&
                  ps.back().dst + ps.back().nbytes == dst)
                ps.back().nbytes += nb;  // contiguous with the previous piece
              else
                ps.push_back({t.file, t.off, nb, dst});
              dst += nb;
            }
          }
          if (dst != codes_bytes + scales_bytes) fail("glm5_next: " + q + "* sizes do not match the config");
          if (ps.size() != 2) ++split_experts;
          (c.mtp_layer(l) ? bytes_mtp : bytes_experts) += dst;
          xs[size_t(e)].pieces = std::move(ps);
        }
      }
      layers = l + 1;
    }
    for (const auto& [name, t] : ck.t) {
      if (name.rfind("model.visual.", 0) == 0) bytes_vision += t.nbytes;
    }
    uint64_t all = 0;
    for (const auto& [name, t] : ck.t) all += t.nbytes;
    bytes_other = all - bytes_resident - bytes_experts - bytes_mtp - bytes_vision;
  }

  std::string summary() const {
    char b[512];
    std::snprintf(b, sizeof b,
                  "glm5_next: %d layers (%d KDA, %d DSA, %d dense MLP, %d MoE of %d experts x %d) + %d MTP; "
                  "resident %.2f GB, routed experts %.2f GB (%.2f MB each, %d split across shards), MTP layer %.2f GB, "
                  "vision %.2f GB, other %.2f GB",
                  c.L, c.n_kda(), c.n_dsa(), c.L - c.n_moe(), c.n_moe(), c.E, c.K, c.n_mtp, bytes_resident / 1e9,
                  bytes_experts / 1e9, (codes_bytes + scales_bytes) / 1e6, split_experts, bytes_mtp / 1e9,
                  bytes_vision / 1e9, bytes_other / 1e9);
    return b;
  }
};

}  // namespace espetero::native::glm5
