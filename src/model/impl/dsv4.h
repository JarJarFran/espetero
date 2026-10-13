// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// DeepSeek-V4.1 (deepseek_v41, DeepSeek-V4.1-Flash and its finetunes) on
// Metal: docs/deepseek_v41.md. A model of its own beside the Qwen3.5 forest:
//
//   * the checkpoint's FP8 / FP4 weights are used as stored (kernels in
//     kernels/80_dsv4.metal), activations rounded to the reference's fp8 grid
//     before each quantized matmul;
//   * the 15,360 routed experts (289 GB) stay on the SSD: an LRU cache of
//     expert slots in GPU memory, and each layer's missing experts read with
//     pread once its routing is known (the pass waits for the GPU there);
//   * the two Engram n-gram tables (203 GB) stay on the SSD too: the 24 rows
//     a token needs per Engram layer are hashed and read on the host;
//   * sequences live in slots of their own (window rings, compressed KV and
//     index keys sized for the context), one pass per sequence.
#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_map>

#include "../../util/json.h"
#include "common.h"
#include "runtime.h"

namespace espetero::native::ds4 {

// ---------------------------------------------------------------- config
struct Cfg {
  int D = 5120, L = 40, V = 129280, H = 64, HD = 512, RD = 64, QR = 1280, OR = 1024, OG = 8, WIN = 128;
  int E = 384, K = 6, IE = 2304, NSH = 1;
  int IH = 32, IHD = 128, ITOPK = 512;
  int cand_src = -1, cand_blocks = 0, cand_bs = 0;
  int hc = 4, hc_iters = 20;
  float hc_eps = 1e-6f, eps = 1e-20f, route_scale = 1.5f, swiglu = 10.0f;
  std::vector<int> ratio;  // per layer (MTP layers after the backbone)
  std::vector<int> kv_src, idx_src, engram_layers;
  double rope_theta = 10000, crope_theta = 160000, rope_factor = 16, beta_fast = 32, beta_slow = 1;
  int orig_len = 65536;
  int eng_heads = 8, eng_hd = 256, eng_max_n = 4;
  int bos = 0, eos = 1;
  // DSpark draft head (mtp.* layers after the backbone)
  int n_mtp = 0, ds_block = 0, ds_noise = 0, ds_E = 0, ds_K = 0, ds_markov = 0;
  std::vector<int> ds_targets;
  // vision (DeepSeek ViT + 3x3 aligner)
  int v_layers = 0, v_dim = 1024, v_heads = 16, v_inter = 2816, v_patch = 14, v_down = 3, v_max_tok = 1024,
      v_min_px = 295936, image_token = -1;
  double v_theta = 10000;
  std::string score_func = "sqrtsoftplus";
};

inline Cfg read_cfg(const json::Value& root) {
  const json::Value* tcp = root.get("text_config");
  const json::Value& tc = tcp && tcp->is_object() ? *tcp : root;
  Cfg c;
  auto ints = [&](const char* k) {
    std::vector<int> v;
    if (const json::Value* a = tc.get(k); a && a->is_array())
      for (const json::Value& x : a->as_array()) v.push_back(int(x.as_int()));
    return v;
  };
  c.D = int(tc.get_int("hidden_size", c.D));
  c.L = int(tc.get_int("num_hidden_layers", c.L));
  c.V = int(tc.get_int("vocab_size", c.V));
  c.H = int(tc.get_int("num_attention_heads", c.H));
  c.HD = int(tc.get_int("head_dim", c.HD));
  c.RD = int(tc.get_int("qk_rope_head_dim", c.RD));
  c.QR = int(tc.get_int("q_lora_rank", c.QR));
  c.OR = int(tc.get_int("o_lora_rank", c.OR));
  c.OG = int(tc.get_int("o_groups", c.OG));
  c.WIN = int(tc.get_int("sliding_window", c.WIN));
  c.E = int(tc.get_int("n_routed_experts", c.E));
  c.K = int(tc.get_int("num_experts_per_tok", c.K));
  c.IE = int(tc.get_int("moe_intermediate_size", c.IE));
  c.NSH = int(tc.get_int("n_shared_experts", c.NSH));
  c.IH = int(tc.get_int("index_n_heads", c.IH));
  c.IHD = int(tc.get_int("index_head_dim", c.IHD));
  c.ITOPK = int(tc.get_int("index_topk", c.ITOPK));
  c.cand_src = int(tc.get_int("candidate_source_layer_id", -1));
  c.cand_blocks = int(tc.get_int("candidate_topk_blocks", 0));
  c.cand_bs = int(tc.get_int("candidate_block_size", 0));
  c.hc = int(tc.get_int("hc_mult", c.hc));
  c.hc_iters = int(tc.get_int("hc_sinkhorn_iters", c.hc_iters));
  c.hc_eps = float(tc.get_double("hc_eps", c.hc_eps));
  c.eps = float(tc.get_double("rms_norm_eps", c.eps));
  c.route_scale = float(tc.get_double("routed_scaling_factor", c.route_scale));
  c.swiglu = float(tc.get_double("swiglu_limit", c.swiglu));
  c.score_func = tc.get_string("scoring_func", c.score_func);
  c.ratio = ints("compress_ratios");
  c.kv_src = ints("kv_source_layer_ids");
  c.idx_src = ints("index_source_layer_ids");
  c.engram_layers = ints("engram_layer_ids");
  c.eng_heads = int(tc.get_int("engram_n_heads", c.eng_heads));
  c.eng_hd = int(tc.get_int("engram_head_dim", c.eng_hd));
  c.eng_max_n = int(tc.get_int("engram_max_ngram_size", c.eng_max_n));
  c.rope_theta = tc.get_double("rope_theta", c.rope_theta);
  c.crope_theta = tc.get_double("compress_rope_theta", c.crope_theta);
  if (const json::Value* rs = tc.get("rope_scaling"); rs && rs->is_object()) {
    c.rope_factor = rs->get_double("factor", c.rope_factor);
    c.beta_fast = rs->get_double("beta_fast", c.beta_fast);
    c.beta_slow = rs->get_double("beta_slow", c.beta_slow);
    c.orig_len = int(rs->get_int("original_max_position_embeddings", c.orig_len));
  }
  c.n_mtp = int(tc.get_int("num_nextn_predict_layers", 0));
  c.ds_block = int(tc.get_int("dspark_block_size", 0));
  c.ds_noise = int(tc.get_int("dspark_noise_token_id", 0));
  c.ds_targets = ints("dspark_target_layer_ids");
  c.ds_E = int(tc.get_int("dspark_n_routed_experts", c.E));
  c.ds_K = int(tc.get_int("dspark_num_experts_per_tok", c.K));
  c.ds_markov = int(tc.get_int("dspark_markov_rank", 256));
  if (const json::Value* vc = root.get("vision_config"); vc && vc->is_object()) {
    c.v_layers = int(vc->get_int("num_hidden_layers", 0));
    c.v_dim = int(vc->get_int("hidden_size", c.v_dim));
    c.v_heads = int(vc->get_int("num_attention_heads", c.v_heads));
    c.v_inter = int(vc->get_int("intermediate_size", c.v_inter));
    c.v_patch = int(vc->get_int("patch_size", c.v_patch));
    c.v_down = int(vc->get_int("downsample_ratio", c.v_down));
    c.v_max_tok = int(vc->get_int("max_image_tokens", c.v_max_tok));
    c.v_min_px = int(vc->get_int("min_pixels", c.v_min_px));
    c.v_theta = vc->get_double("rope_theta", c.v_theta);
  }
  c.image_token = int(root.get_int("image_token_id", -1));
  c.bos = int(root.get_int("bos_token_id", tc.get_int("bos_token_id", 0)));
  c.eos = int(root.get_int("eos_token_id", tc.get_int("eos_token_id", 1)));
  if (int(c.ratio.size()) < c.L) fail("deepseek_v41: compress_ratios has fewer entries than layers");
  if (c.HD != 512 || c.hc != 4 || c.IHD != 128 || c.IH > 32 || c.K > 8 || c.E > 512 || c.RD != 64)
    fail("deepseek_v41: unsupported geometry (head_dim 512, hc_mult 4, index_head_dim 128, <= 32 index heads, "
         "<= 8 of <= 512 experts expected)");
  if (c.score_func != "sqrtsoftplus") fail("deepseek_v41: scoring_func " + c.score_func + " is not supported");
  return c;
}

// RoPE frequencies (inference/model.py precompute_freqs_cis), float32 as the reference.
inline std::vector<float> inv_freq(int dim, int orig, double base, double factor, double bf, double bs) {
  std::vector<float> f(size_t(dim / 2));
  for (int i = 0; i < dim / 2; ++i) f[size_t(i)] = 1.0f / std::pow(float(base), float(2 * i) / float(dim));
  if (orig > 0) {
    auto corr = [&](double rot) { return dim * std::log(orig / (rot * 2 * M_PI)) / (2 * std::log(base)); };
    const double low = std::max(std::floor(corr(bf)), 0.0), high = std::min(std::ceil(corr(bs)), double(dim - 1));
    for (int i = 0; i < dim / 2; ++i) {
      float ramp = float((double(i) - low) / std::max(high - low, 1e-3));
      ramp = std::clamp(ramp, 0.0f, 1.0f);
      const float smooth = 1.0f - ramp;
      f[size_t(i)] = f[size_t(i)] / float(factor) * (1.0f - smooth) + f[size_t(i)] * smooth;
    }
  }
  return f;
}

// ---------------------------------------------------------------- checkpoint index
struct TInfo {
  int file = -1;
  uint64_t off = 0, nbytes = 0;
  std::string dtype;
  std::vector<int64_t> shape;
};

struct Ckpt {
  std::string dir;
  std::vector<std::string> files;
  std::vector<int> fds, fds_nc;  // buffered / F_NOCACHE
  std::unordered_map<std::string, TInfo> t;

  ~Ckpt() {
    for (int fd : fds) ::close(fd);
    for (int fd : fds_nc) ::close(fd);
  }
  // partial: files not downloaded yet are skipped (development: the first layers only).
  void open(const std::string& d, bool partial = false) {
    dir = d;
    const json::Value idx = json::parse(read_file(dir + "/model.safetensors.index.json"));
    const json::Value* wm = idx.get("weight_map");
    if (!wm || !wm->is_object()) fail(dir + ": model.safetensors.index.json has no weight_map");
    std::unordered_map<std::string, int> fid;
    for (const json::Value& f : wm->as_object().vals)
      if (!fid.count(f.as_string())) {
        fid[f.as_string()] = int(files.size());
        files.push_back(f.as_string());
      }
    fds.resize(files.size(), -1);
    fds_nc.resize(files.size(), -1);
    std::vector<std::string> err(files.size());
    std::vector<std::vector<std::pair<std::string, TInfo>>> parts(files.size());
    parallel_for(files.size(), [&](size_t i) {
      const std::string path = dir + "/" + files[i];
      const int fd = ::open(path.c_str(), O_RDONLY);
      if (fd < 0) {
        if (!partial) err[i] = "cannot open " + path + " (download incomplete?)";
        return;
      }
      fds[i] = fd;
      fds_nc[i] = ::open(path.c_str(), O_RDONLY);
      if (fds_nc[i] >= 0) ::fcntl(fds_nc[i], F_NOCACHE, 1);
      uint64_t n = 0;
      if (!pread_all(fd, &n, size_t(8), size_t(0)) || n > (512ull << 20)) {
        err[i] = path + ": not a safetensors file";
        return;
      }
      std::string h(size_t(n), '\0');
      if (!pread_all(fd, h.data(), size_t(n), size_t(8))) {
        err[i] = path + ": short header";
        return;
      }
      struct stat st;
      ::fstat(fd, &st);
      const json::Value hv = json::parse(h);
      const json::Object& o = hv.as_object();
      for (size_t k = 0; k < o.size(); ++k) {
        if (o.keys[k] == "__metadata__") continue;
        const json::Value& v = o.vals[k];
        TInfo ti;
        ti.file = int(i);
        ti.dtype = v.get_string("dtype", "");
        for (const json::Value& s : v.get("shape")->as_array()) ti.shape.push_back(s.as_int());
        const json::Value& off = *v.get("data_offsets");
        ti.off = 8 + n + uint64_t(off[0].as_int());
        ti.nbytes = uint64_t(off[1].as_int() - off[0].as_int());
        if (ti.off + ti.nbytes > uint64_t(st.st_size)) {
          err[i] = path + ": shorter than its header says (download incomplete?)";
          return;
        }
        parts[i].push_back({o.keys[k], std::move(ti)});
      }
    });
    for (const std::string& e : err)
      if (!e.empty()) fail(e);
    for (auto& p : parts)
      for (auto& [k, v] : p) t.emplace(k, std::move(v));
    for (size_t k = 0; k < wm->as_object().size() && !partial; ++k)
      if (!t.count(wm->as_object().keys[k])) fail(dir + ": tensor " + wm->as_object().keys[k] + " missing from its file");
  }
  bool has(const std::string& n) const { return t.count(n) != 0; }
  const TInfo& at(const std::string& n) const {
    auto it = t.find(n);
    if (it == t.end()) fail("deepseek_v41: missing tensor " + n);
    return it->second;
  }
  // The tensor's bytes into dst, in parallel 8 MB reads.
  void read(const TInfo& ti, void* dst) const {
    const size_t piece = size_t(8) << 20;
    const size_t n = size_t(ti.nbytes);
    std::atomic<bool> ok{true};
    parallel_for((n + piece - 1) / piece, [&](size_t i) {
      const size_t o = i * piece, m = std::min(piece, n - o);
      if (!pread_all(fds[size_t(ti.file)], static_cast<uint8_t*>(dst) + o, m, size_t(ti.off + o))) ok = false;
    });
    if (!ok) fail("deepseek_v41: read failed in " + files[size_t(ti.file)]);
  }
};

// ---------------------------------------------------------------- weights
struct Mat8 {  // fp8 e4m3 [N, K], e8m0 scales [N/32, K/32]
  Buf w, s;
  int N = 0, K = 0;
};
struct MatB {  // bf16 [N, K]
  Buf w;
  int N = 0, K = 0;
};
struct XLoc {  // a routed expert in the checkpoint: w1 | w2 | w3 codes, then w1 | w2 | w3 scales
  int file = -1;
  uint64_t w = 0, s = 0;
};
struct ETable {  // an Engram n-gram table: fp8 rows [rows, hd], e8m0 scales [rows, hd / 32]
  int file = -1;
  uint64_t w = 0, s = 0;
  int64_t rows = 0;
};

struct LayerW {
  int nE = 0, topk = 0;     // routed experts and experts per token
  int ratio = 0, src = -1;  // compressed KV source index (0..), -1: none
  bool kv_src = false, idx_src = false, cand_src = false, uses_cand = false;
  Buf attn_norm, ffn_norm, hca_fn, hca_base, hca_scale, hcf_fn, hcf_base, hcf_scale;
  Buf q_norm, kv_norm, sink;
  Mat8 wq_a, wq_b, wkv, wo_a, wo_b;
  MatB c_wkv, c_wgate;
  Buf c_norm;
  Mat8 i_wq_b;
  MatB i_wproj, i_wk;
  Buf i_knorm;
  MatB gate;
  Buf gate_bias, gate_bias_vl;  // the VL bias picks the experts of image-span rows
  Mat8 sh1, sh2, sh3;
  std::vector<XLoc> x;
  int eg = -1;  // Engram index, -1: none
  Mat8 e_wkv;
  Buf e_wqk;
  ETable tab;
};

// Engram constants (tools/dsv4_engram_prep.py writes them next to the model).
struct EngramConst {
  int vocab = 0, cvocab = 0, layers = 0, max_n = 0, heads = 0, pad = 0;
  std::vector<int64_t> mult;    // [layers][max_n]
  std::vector<int64_t> primes;  // [layers][(max_n - 1) * heads]
  std::vector<int64_t> offs;    // same shape: running offsets within a layer's table
  std::vector<uint32_t> map;    // [vocab]
};

inline EngramConst read_engram_const(const std::string& path, const Cfg& c) {
  std::string b;
  if (!file_exists(path))
    fail("deepseek_v41: " + path + " is missing; write it once with: python3 tools/dsv4_engram_prep.py " +
         dirname_of(path) + "  (needs numpy and tokenizers)");
  b = read_file(path);
  if (b.size() < 40 || std::memcmp(b.data(), "ESPDSV4E", 8) != 0) fail(path + ": not an Espetero Engram file");
  int32_t h[8];
  std::memcpy(h, b.data() + 8, sizeof h);
  EngramConst e;
  e.vocab = h[1], e.cvocab = h[2], e.layers = h[3], e.max_n = h[4], e.heads = h[5], e.pad = h[6];
  if (e.layers != int(c.engram_layers.size()) || e.max_n != c.eng_max_n || e.heads != c.eng_heads)
    fail(path + ": does not match config.json (re-run tools/dsv4_engram_prep.py)");
  size_t o = 40;
  const size_t nm = size_t(e.layers * e.max_n), np = size_t(e.layers * (e.max_n - 1) * e.heads);
  if (b.size() != o + 8 * (nm + np) + 4 * size_t(e.vocab)) fail(path + ": wrong size");
  e.mult.resize(nm);
  std::memcpy(e.mult.data(), b.data() + o, 8 * nm);
  o += 8 * nm;
  e.primes.resize(np);
  std::memcpy(e.primes.data(), b.data() + o, 8 * np);
  o += 8 * np;
  e.map.resize(size_t(e.vocab));
  std::memcpy(e.map.data(), b.data() + o, 4 * size_t(e.vocab));
  const size_t per = size_t((e.max_n - 1) * e.heads);
  e.offs.resize(np);
  for (int l = 0; l < e.layers; ++l) {
    int64_t acc = 0;
    for (size_t j = 0; j < per; ++j) {
      e.offs[size_t(l) * per + j] = acc;
      acc += e.primes[size_t(l) * per + j];
    }
  }
  return e;
}

// The Engram rows of one position for Engram layer g: tokens[s] is the
// compressed id `s` positions back (pad where the sequence has not started).
inline void engram_rows(const EngramConst& e, int g, const int64_t* tokens, int64_t* rows) {
  const int64_t* M = e.mult.data() + size_t(g * e.max_n);
  const size_t per = size_t((e.max_n - 1) * e.heads);
  const int64_t* P = e.primes.data() + size_t(g) * per;
  const int64_t* O = e.offs.data() + size_t(g) * per;
  int64_t rolling = tokens[0] * M[0];
  for (int i = 1; i < e.max_n; ++i) {
    rolling ^= tokens[i] * M[i];
    for (int h = 0; h < e.heads; ++h) {
      const size_t j = size_t((i - 1) * e.heads + h);
      rows[j] = rolling % P[j] + O[j];
    }
  }
}

// ---------------------------------------------------------------- expert cache
// Slots of one expert each (w1 | w2 | w3 codes, then scales, as in the file)
// in GPU-private buffers; LRU over (layer, expert).
struct ExpertCache {
  size_t slot_bytes = 0, wbytes = 0, sbytes = 0;
  int per_buf = 0, n = 0;
  std::vector<id<MTLBuffer>> bufs;
  std::vector<int> key;          // slot -> layer * E + expert, -1 free
  std::vector<uint64_t> stamp;   // slot -> last use
  std::vector<int> where;        // layer * E + expert -> slot, -1
  uint64_t clock = 0;
  // staging (shared) for reads
  id<MTLBuffer> stage = nil;
  int stage_n = 0;
  // stats
  uint64_t hits = 0, misses = 0;
  double read_ms = 0, read_gb = 0;

  Buf slot(int s) const { return Buf{bufs[size_t(s / per_buf)], size_t(s % per_buf) * slot_bytes}; }
  int part(int s) const { return s / per_buf; }
  size_t offset(int s) const { return size_t(s % per_buf) * slot_bytes; }
};

// ---------------------------------------------------------------- sequence slots
struct Seq {
  bool used = false;
  int len = 0;
  Buf ring;                    // [L][WIN][HD] half
  std::vector<Buf> comp, idxk; // per KV source: [cap][HD] half, [cap][IHD] half
  std::vector<int> cap;        // per KV source: entries
  Buf pend;                    // per ratio-2 source: kv [2][HD] | score [2][HD] float
  std::vector<int32_t> hist;   // the last eng_max_n - 1 token ids (oldest first)
  // DSpark: the draft layers' window rings of the target's projected states,
  // the target-layer means by position (ring of 128), the first position not
  // yet in the draft rings; a verify pass's compressor rows (to roll back).
  Buf dsring, mh, vpend;
  int ds_len = 0;
};

}  // namespace espetero::native::ds4
