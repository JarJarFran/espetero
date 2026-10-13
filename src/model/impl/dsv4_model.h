// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// DeepSeek-V4.1 as a NativeModel (dsv4.h has the formats, docs/deepseek_v41.md
// the math). Plain roots only (no leaves, drafts or images yet); every tree
// runs as its own forward, layer by layer, waiting for the GPU after each
// layer's routing to bring the missing experts in from the SSD.
#pragma once

#include <mach/mach.h>
#include <sys/sysctl.h>

#include "../api.h"
#include "dsv4.h"

namespace espetero::native::ds4 {

class Dsv4Model final : public NativeModel {
 public:
  Dsv4Model(const std::string& path, const LoadOptions& o, const std::string& kernels) {
    const auto t0 = Clock::now();
    verbose_ = o.verbose || env_on("ESPETERO_DS4_VERBOSE");
    const json::Value root = json::parse(read_file(path + "/config.json"));
    c_ = read_cfg(root);
    rt_.init(kernels);
    rt_.serial = true;
    // Development: ESPETERO_DS4_LAYERS=n loads the first n layers only (no head; the
    // files of later layers may still be downloading), ESPETERO_DS4_NO_ENGRAM=1 skips Engram.
    if (const char* v = std::getenv("ESPETERO_DS4_LAYERS"); v && std::atoi(v) > 0) {
      partial_ = true;
      c_.L = std::min(c_.L, std::atoi(v));
    }
    no_engram_ = env_on("ESPETERO_DS4_NO_ENGRAM");
    ck_.open(path, partial_);
    eng_ = read_engram_const(path + "/espetero_engram.bin", c_);
    max_ctx_ = std::max(256, o.max_ctx > 0 ? o.max_ctx : 32768);
    // Rows of a pass (a prompt up to T_ tokens runs as one, layer by layer) and
    // of a GPU sub-chunk (P_).
    P_ = 1024;
    if (const char* v = std::getenv("ESPETERO_DS4_SUB")) P_ = std::clamp(std::atoi(v), 8, 4096);
    T_ = std::min(max_ctx_, std::max(16384, o.max_rows));
    if (const char* v = std::getenv("ESPETERO_DS4_ROWS")) T_ = std::clamp(std::atoi(v), 1, max_ctx_);
    P_ = std::min(P_, T_);
    load_weights();
    const double wgb = double(rt_.arena->total()) / 1e9;
    alloc_scratch();
    // ---- sequence slots: window rings, compressed KV and index keys for max_ctx tokens each
    int nslots = o.slots > 0 ? o.slots : 4;
    seqs_.resize(size_t(nslots) + 1);  // the last one: scratch for passes that save nothing
    for (Seq& s : seqs_) alloc_seq(s);
    // ---- expert cache: what the GPU working set and the Mac's memory leave
    {
      const double ws = double([rt_.dev recommendedMaxWorkingSetSize]);
      uint64_t ram = 0;
      size_t len = sizeof ram;
      sysctlbyname("hw.memsize", &ram, &len, nullptr, 0);
      const double mine = double([rt_.dev currentAllocatedSize]);
      double gb = o.expert_cache_gb;
      if (const char* v = std::getenv("ESPETERO_EXPERT_CACHE_GB")) gb = std::atof(v);
      if (gb <= 0) gb = std::min(ws - mine - 4e9, double(ram) - mine - 26e9) / 1e9;
      const double all = double(c_.L) * c_.E * double(xc_.slot_bytes) / 1e9;
      gb = std::max(1.0, std::min(gb, all));
      make_cache(gb);
      std::fprintf(stderr, "deepseek_v41: %.1f GB of weights in memory, %d of %d experts (%.1f GB) cached in GPU memory, "
                           "the rest read from the SSD; Engram tables (%.0f GB) read from the SSD\n",
                   wgb, xc_.n, c_.L * c_.E, double(xc_.n) * double(xc_.slot_bytes) / 1e9, engram_gb_);
    }
    info_.device = [[rt_.dev name] UTF8String];
    info_.format = "deepseek_v41";
    info_.path = path;
    info_.hidden = c_.D;
    info_.intermediate = c_.IE;
    info_.layers = c_.L;
    info_.full_layers = c_.L;
    info_.heads = c_.H;
    info_.kv_heads = 1;
    info_.vocab = c_.V;
    info_.eos_token_id = c_.eos;
    info_.slots = nslots;
    info_.block_tokens = 64;
    info_.kv_blocks = nslots * max_ctx_ / 64;
    info_.max_ctx = max_ctx_;
    info_.max_rows = T_;
    info_.max_readouts = 1;
    info_.weight_bytes = rt_.arena->total();
    info_.kv_bytes = seq_bytes_ * size_t(nslots);
    info_.slot_state_bytes = seq_bytes_;
    info_.leaves = false;
    info_.commit = false;
    info_.hidden_states = false;
    info_.mtp = false;
    info_.vision = false;
    info_.load_ms = ms_since(t0);
    json::Value ex = json::Value::object();
    ex.set("expert_cache_slots", xc_.n);
    ex.set("expert_cache_gb", double(xc_.n) * double(xc_.slot_bytes) / 1e9);
    info_.extra = ex;
  }

  ~Dsv4Model() override = default;

  const ModelInfo& info() const override { return info_; }

  // ---- slots
  int slot_alloc() override {
    for (size_t i = 0; i + 1 < seqs_.size(); ++i)
      if (!seqs_[i].used) {
        seqs_[i].used = true;
        reset(seqs_[i]);
        return int(i);
      }
    return -1;
  }
  void slot_free(int s) override {
    Seq& q = seq(s);
    q.used = false;
    reset(q);
  }
  void slot_reset(int s) override { reset(seq(s)); }
  void slot_fork(int dst, int src) override { fork(seq(dst), seq(src)); }
  int slot_len(int s) const override { return seqs_.at(size_t(s)).len; }
  int slot_next_pos(int s) const override { return seqs_.at(size_t(s)).len; }
  bool slot_mtp(int) const override { return false; }
  int free_slots() const override {
    int n = 0;
    for (size_t i = 0; i + 1 < seqs_.size(); ++i) n += seqs_[i].used ? 0 : 1;
    return n;
  }
  int free_kv_tokens() const override {
    int n = 0;
    for (size_t i = 0; i + 1 < seqs_.size(); ++i) n += max_ctx_ - (seqs_[i].used ? seqs_[i].len : 0);
    return n;
  }
  int pass_kv_tokens(const std::vector<TreeSpec>&) const override { return 0; }
  // Every slot owns its rows (no shared blocks): freeing a slot gives back its length.
  int freed_kv_tokens(const std::vector<int>& slots) const override {
    int n = 0;
    for (int s : slots) n += seqs_.at(size_t(s)).used ? seqs_[size_t(s)].len : 0;
    return n;
  }
  bool slot_save(int, const std::string&) override { return false; }
  int slot_load(const std::string&) override { return -1; }
  std::string state_signature() const override { return "deepseek_v41-unsaved"; }

  // ---- passes
  void run(const std::vector<TreeSpec>& trees, PassStats* stats) override {
    @autoreleasepool {
      const auto t0 = Clock::now();
      out_.assign(trees.size(), Readout{});
      int rows = 0;
      for (size_t i = 0; i < trees.size(); ++i) {
        run_tree(trees[i], out_[i]);
        rows += trees[i].root_len;
      }
      if (stats) {
        stats->wall_ms = stats->gpu_ms = ms_since(t0);
        stats->rows = rows;
        stats->readouts = int(trees.size());
      }
    }
  }

  void run_chain(const std::vector<TreeSpec>& first, int steps, int32_t* out, PassStats* stats) override {
    std::vector<TreeSpec> cur = first;
    std::vector<int32_t> tok(first.size());
    PassStats total;
    for (int j = 0; j < steps; ++j) {
      for (size_t t = 0; t < cur.size(); ++t) {
        if (j > 0) {
          tok[t] = out[size_t(j - 1) * cur.size() + t];
          cur[t].root = &tok[t];
          cur[t].draw_pos0 = first[t].draw_pos0 + j;
        }
        cur[t].out_argmax = nullptr;
      }
      PassStats st;
      run(cur, &st);
      for (size_t t = 0; t < cur.size(); ++t) {
        const Readout& r = out_[t];
        out[size_t(j) * cur.size() + t] = cur[t].gpu_draw && r.sampled >= 0 ? r.sampled : r.argmax;
      }
      total.wall_ms += st.wall_ms;
      total.gpu_ms += st.gpu_ms;
      total.rows += st.rows;
      total.readouts += st.readouts;
    }
    if (stats) *stats = total;
  }

  const float* logits(int tree, int readout) const override {
    if (readout != 0) fail("deepseek_v41: one readout per tree");
    return out_.at(size_t(tree)).logits.data();
  }
  TopK topk(int tree, int readout) const override {
    TopK r;
    if (readout != 0) return r;
    const Readout& o = out_.at(size_t(tree));
    r.ids = o.tk_ids.data();
    r.x = o.tk_x.data();
    r.n = int(o.tk_ids.size());
    r.max = o.tk_max;
    r.total = o.tk_total;
    r.sampled = o.sampled;
    return r;
  }

  void draft(const std::vector<DraftRequest>&, PassStats*) override { fail("deepseek_v41: no draft head yet"); }
  void commit(const std::vector<CommitRequest>&, PassStats*) override { fail("deepseek_v41: no verify-commit"); }
  void encode_images(const std::vector<ImageRef>&, float*, PassStats*) override {
    fail("deepseek_v41: images are not supported yet");
  }
  void set_profile(bool on) override { profile_ = on; }
  std::string profile_json() override {
    json::Value o = json::Value::object();
    o.set("expert_hits", int64_t(xc_.hits));
    o.set("expert_misses", int64_t(xc_.misses));
    o.set("expert_read_ms", xc_.read_ms);
    o.set("expert_read_gb", xc_.read_gb);
    o.set("engram_read_ms", eng_read_ms_);
    o.set("gpu_ms", gpu_ms_);
    o.set("flushes", int64_t(flushes_));
    if (!prof_.empty()) {
      json::Value k = json::Value::object();
      for (const auto& [n, v] : prof_) k.set(n, v.first);
      o.set("kernels_ms", k);
    }
    return json::dump(o);
  }
  bool set_option(const std::string& name, int value) override {
    if (name == "expert_stats_reset") {
      xc_.hits = xc_.misses = 0;
      xc_.read_ms = xc_.read_gb = eng_read_ms_ = gpu_ms_ = 0;
      flushes_ = 0;
      prof_.clear();
      return true;
    }
    return false;
  }
  std::string bench_json(const std::string&) override { return profile_json(); }

  // Development: per-layer hidden dumps for parity (ESPETERO_DS4_DUMP=dir).
  std::string dump_dir_ = [] {
    const char* v = std::getenv("ESPETERO_DS4_DUMP");
    return std::string(v ? v : "");
  }();

 private:
  struct Readout {
    std::vector<float> logits;
    int32_t argmax = -1, sampled = -1;
    float lse = 0;
    std::vector<int32_t> tk_ids;
    std::vector<float> tk_x;
    float tk_max = 0, tk_total = 0;
  };

  Runtime rt_;
  Cfg c_;
  Ckpt ck_;
  EngramConst eng_;
  std::vector<LayerW> L_;
  Buf embed_, head_, norm_;
  Buf invf_plain_, invf_yarn_;
  ExpertCache xc_;
  std::vector<Seq> seqs_;
  ModelInfo info_;
  int P_ = 1024, T_ = 16384, max_ctx_ = 32768, tile_cap_ = 0;
  size_t seq_bytes_ = 0;
  double engram_gb_ = 0;
  bool verbose_ = false, profile_ = false, partial_ = false, no_engram_ = false;
  std::vector<Readout> out_;
  std::map<std::string, std::pair<double, int>> prof_;
  double eng_read_ms_ = 0, gpu_ms_ = 0;
  uint64_t flushes_ = 0;
  std::vector<int> src_ratio_;  // per KV source: its ratio

  // scratch (shared)
  id<MTLBuffer> ids_, pos_, gpos_, mix0_, H_, X_, XQ_, Y_, mixA_, mixF_[2], QA_, Q_, KVp_, OA_, CKV_, CSC_, LAT_, IK_, IQ_, IW_,
      ISC_, IDX_, BS_, CAND_, XGRP_, XTILE_, HCP_, RL_, SEL_, RW_, xrow_, yrow_, wsc_, amap_, ident_, ones_, XG_, XU_, XH_, XP_, SG_, SU_, SH_, EE_,
      ERAW_[2], EKV_, XF_, XN_, LOG_;
  size_t isc_floats_ = 0;
  int cstride_ = 1;
  int gemm_min_ = [] {  // rows from which the fp8 matrices and the experts use the GEMM kernels
    const char* v = std::getenv("ESPETERO_DS4_GEMM_MIN");
    return v ? std::atoi(v) : 16;
  }();

  // ---------------------------------------------------------------- loading
  Buf raw(const std::string& name, uint64_t expect = 0) {
    const TInfo& t = ck_.at(name);
    if (expect && t.nbytes != expect)
      fail("deepseek_v41: " + name + " has " + std::to_string(t.nbytes) + " bytes, expected " + std::to_string(expect));
    Buf b = rt_.arena->alloc(size_t(t.nbytes));
    ck_.read(t, ptr(b));
    return b;
  }
  Buf vecf(const std::string& name, size_t n) {
    const TInfo& t = ck_.at(name);
    std::vector<uint8_t> tmp(size_t(t.nbytes));
    ck_.read(t, tmp.data());
    std::vector<float> f(n);
    if (t.dtype == "BF16" && t.nbytes == 2 * n) {
      for (size_t i = 0; i < n; ++i) f[i] = bf16_to_f32(reinterpret_cast<const uint16_t*>(tmp.data())[i]);
    } else if (t.dtype == "F32" && t.nbytes == 4 * n) {
      std::memcpy(f.data(), tmp.data(), 4 * n);
    } else {
      fail("deepseek_v41: " + name + ": unexpected dtype " + t.dtype + " or size");
    }
    return rt_.arena->copy(f.data(), 4 * n);
  }
  Mat8 mat8(const std::string& p, int N, int K) {
    Mat8 m;
    m.N = N, m.K = K;
    if (ck_.at(p + ".weight").dtype != "F8_E4M3") fail("deepseek_v41: " + p + " is not FP8 e4m3");
    m.w = raw(p + ".weight", uint64_t(N) * uint64_t(K));
    m.s = raw(p + ".scale", uint64_t((N + 31) / 32) * uint64_t((K + 31) / 32));
    return m;
  }
  MatB matb(const std::string& n, int N, int K) {
    MatB m;
    m.N = N, m.K = K;
    if (ck_.at(n).dtype != "BF16") fail("deepseek_v41: " + n + " is not bf16");
    m.w = raw(n, uint64_t(N) * uint64_t(K) * 2);
    return m;
  }

  void load_weights() {
    const auto t0 = Clock::now();
    const int D = c_.D;
    embed_ = raw("embed.weight", uint64_t(c_.V) * uint64_t(D) * 2);
    if (!partial_ || ck_.has("head.weight")) {
      head_ = raw("head.weight", uint64_t(c_.V) * uint64_t(D) * 2);
      norm_ = vecf("norm.weight", size_t(D));
    }
    {
      const std::vector<float> a = inv_freq(c_.RD, 0, c_.rope_theta, 1, c_.beta_fast, c_.beta_slow);
      const std::vector<float> b = inv_freq(c_.RD, c_.orig_len, c_.crope_theta, c_.rope_factor, c_.beta_fast, c_.beta_slow);
      invf_plain_ = rt_.arena->copy(a.data(), a.size() * 4);
      invf_yarn_ = rt_.arena->copy(b.data(), b.size() * 4);
    }
    L_.resize(size_t(c_.L));
    int cur_src = -1;
    for (int l = 0; l < c_.L; ++l) {
      LayerW& w = L_[size_t(l)];
      const std::string p = "layers." + std::to_string(l) + ".";
      w.ratio = c_.ratio[size_t(l)];
      w.kv_src = std::find(c_.kv_src.begin(), c_.kv_src.end(), l) != c_.kv_src.end();
      w.idx_src = std::find(c_.idx_src.begin(), c_.idx_src.end(), l) != c_.idx_src.end();
      w.cand_src = l == c_.cand_src;
      w.uses_cand = c_.cand_src >= 0 && c_.cand_src < l;
      if (w.kv_src) {
        cur_src = int(src_ratio_.size());
        src_ratio_.push_back(w.ratio);
      }
      if (w.ratio > 0) {
        if (cur_src < 0 || src_ratio_[size_t(cur_src)] != w.ratio)
          fail("deepseek_v41: layer " + std::to_string(l) + " compresses without a KV source of its ratio");
        w.src = cur_src;
      }
      const int W4 = 4 * D;
      w.attn_norm = vecf(p + "attn_norm.weight", size_t(D));
      w.ffn_norm = vecf(p + "ffn_norm.weight", size_t(D));
      w.hca_fn = raw(p + "hc_attn_fn", uint64_t(24) * W4 * 4);
      w.hca_base = raw(p + "hc_attn_base", 24 * 4);
      w.hca_scale = raw(p + "hc_attn_scale", 3 * 4);
      w.hcf_fn = raw(p + "hc_ffn_fn", uint64_t(24) * W4 * 4);
      w.hcf_base = raw(p + "hc_ffn_base", 24 * 4);
      w.hcf_scale = raw(p + "hc_ffn_scale", 3 * 4);
      w.q_norm = vecf(p + "attn.q_norm.weight", size_t(c_.QR));
      w.kv_norm = vecf(p + "attn.kv_norm.weight", size_t(c_.HD));
      w.sink = vecf(p + "attn.attn_sink", size_t(c_.H));
      w.wq_a = mat8(p + "attn.wq_a", c_.QR, D);
      w.wq_b = mat8(p + "attn.wq_b", c_.H * c_.HD, c_.QR);
      w.wkv = mat8(p + "attn.wkv", c_.HD, D);
      w.wo_a = mat8(p + "attn.wo_a", c_.OG * c_.OR, c_.H * c_.HD / c_.OG);
      w.wo_b = mat8(p + "attn.wo_b", D, c_.OG * c_.OR);
      if (w.kv_src) {
        w.c_wkv = matb(p + "attn.compressor.wkv.weight", c_.HD, D);
        if (w.ratio > 1) w.c_wgate = matb(p + "attn.compressor.wgate.weight", c_.HD, D);
        w.c_norm = vecf(p + "attn.compressor.norm.weight", size_t(c_.HD));
        w.i_wk = matb(p + "attn.indexer.wk.weight", c_.IHD, c_.HD);
        w.i_knorm = vecf(p + "attn.indexer.k_norm.weight", size_t(c_.IHD));
      }
      if (w.idx_src) {
        w.i_wq_b = mat8(p + "attn.indexer.wq_b", c_.IH * c_.IHD, c_.QR);
        w.i_wproj = matb(p + "attn.indexer.weights_proj.weight", c_.IH, D);
      }
      w.gate = matb(p + "ffn.gate.weight", c_.E, D);
      w.gate_bias = vecf(p + "ffn.gate.bias", size_t(c_.E));
      w.sh1 = mat8(p + "ffn.shared_experts.w1", c_.IE, D);
      w.sh2 = mat8(p + "ffn.shared_experts.w2", D, c_.IE);
      w.sh3 = mat8(p + "ffn.shared_experts.w3", c_.IE, D);
      // routed experts: where each one's codes and scales are
      w.x.resize(size_t(c_.E));
      const uint64_t wb = uint64_t(c_.IE) * uint64_t(D) / 2, sb = uint64_t(c_.IE) * uint64_t(D) / 32;
      for (int e = 0; e < c_.E; ++e) {
        const std::string q = p + "ffn.experts." + std::to_string(e) + ".";
        const TInfo &a = ck_.at(q + "w1.weight"), &b = ck_.at(q + "w2.weight"), &cc = ck_.at(q + "w3.weight");
        const TInfo &sa = ck_.at(q + "w1.scale"), &sb2 = ck_.at(q + "w2.scale"), &sc = ck_.at(q + "w3.scale");
        if (a.nbytes != wb || b.nbytes != wb || cc.nbytes != wb || sa.nbytes != sb || sb2.nbytes != sb || sc.nbytes != sb)
          fail("deepseek_v41: " + q + "* sizes do not match moe_intermediate_size");
        if (a.file != b.file || a.file != cc.file || a.file != sa.file || b.off != a.off + wb || cc.off != b.off + wb ||
            sb2.off != sa.off + sb || sc.off != sb2.off + sb)
          fail("deepseek_v41: " + q + "* are not stored contiguously (w1, w2, w3); this layout is not supported yet");
        w.x[size_t(e)] = XLoc{a.file, a.off, sa.off};
      }
      // Engram
      const auto eit = std::find(c_.engram_layers.begin(), c_.engram_layers.end(), l);
      if (eit != c_.engram_layers.end() && !no_engram_) {
        w.eg = int(eit - c_.engram_layers.begin());
        const int ncols = (c_.eng_max_n - 1) * c_.eng_heads;
        w.e_wkv = mat8(p + "engram.wkv", D * (c_.hc + 1), ncols * c_.eng_hd);
        std::vector<float> q(size_t(4) * D), k(size_t(4) * D);
        {
          Buf a = vecf(p + "engram.q_weight", size_t(4) * D), b = vecf(p + "engram.k_weight", size_t(4) * D);
          std::memcpy(q.data(), ptr(a), q.size() * 4);
          std::memcpy(k.data(), ptr(b), k.size() * 4);
        }
        for (size_t i = 0; i < q.size(); ++i) q[i] *= k[i];
        w.e_wqk = rt_.arena->copy(q.data(), q.size() * 4);
        const TInfo &tw = ck_.at(p + "engram.embed.weight"), &ts = ck_.at(p + "engram.embed.scale");
        if (tw.dtype != "F8_E4M3" || tw.shape.size() != 2 || tw.shape[1] != c_.eng_hd || ts.file != tw.file)
          fail("deepseek_v41: " + p + "engram.embed has an unexpected layout");
        w.tab = ETable{tw.file, tw.off, ts.off, tw.shape[0]};
        engram_gb_ += double(tw.nbytes + ts.nbytes) / 1e9;
      }
    }
    xc_.wbytes = size_t(3) * size_t(c_.IE) * size_t(c_.D) / 2;
    xc_.sbytes = size_t(3) * size_t(c_.IE) * size_t(c_.D) / 32;
    xc_.slot_bytes = xc_.wbytes + xc_.sbytes;
    if (verbose_) std::fprintf(stderr, "deepseek_v41: dense weights loaded in %.1f s\n", ms_since(t0) / 1e3);
  }

  void make_cache(double gb) {
    ExpertCache& x = xc_;
    x.n = int(gb * 1e9 / double(x.slot_bytes));
    x.n = std::max(x.n, c_.E + 16);
    // at most 3 buffers (the grouped expert kernel binds them all)
    const size_t maxlen = std::min<size_t>(size_t([rt_.dev maxBufferLength]), size_t(64) << 30);
    x.per_buf = std::max(int((size_t(x.n) + 2) / 3), 1);
    x.per_buf = std::min(x.per_buf, int(maxlen / x.slot_bytes));
    if ((x.n + x.per_buf - 1) / x.per_buf > 3) x.n = 3 * x.per_buf;
    for (int s = 0; s < x.n; s += x.per_buf) x.bufs.push_back(rt_.private_buffer(size_t(std::min(x.per_buf, x.n - s)) * x.slot_bytes));
    x.key.assign(size_t(x.n), -1);
    x.stamp.assign(size_t(x.n), 0);
    x.where.assign(size_t(c_.L) * size_t(c_.E), -1);
    x.stage_n = 48;
    x.stage = rt_.buffer(size_t(x.stage_n) * x.slot_bytes);
  }

  // Pass state for T_ rows (residual streams, mixes, MoE inputs, routing, index
  // selections) and GPU scratch for one sub-chunk of P_ rows.
  void alloc_scratch() {
    const size_t T = size_t(T_), P = size_t(P_), D = size_t(c_.D), A = P * size_t(c_.K), K = size_t(c_.K);
    auto F = [&](size_t n) { return rt_.buffer(n * 4); };
    ids_ = F(T), pos_ = F(T), gpos_ = F(T);
    mix0_ = F(T * 24);
    {
      float* m = static_cast<float*>([mix0_ contents]);
      for (size_t r = 0; r < T; ++r) m[r * 24] = 1.0f;
    }
    H_ = F(T * 4 * D), XF_ = F(T * D);
    mixA_ = F(T * 24), mixF_[0] = F(T * 24), mixF_[1] = F(T * 24);
    SEL_ = F(T * K), RW_ = F(T * K);
    xrow_ = F(T * K), wsc_ = F(T * K), amap_ = F(T * K);
    IDX_ = F(T * size_t(c_.ITOPK));
    cstride_ = c_.cand_bs > 0 ? max_ctx_ / c_.cand_bs + 2 : 1;
    CAND_ = rt_.buffer(T * size_t(cstride_));
    const size_t nsub = (T + P - 1) / P;
    tile_cap_ = 3 * int(A / 16 + size_t(c_.E) + 1);  // three tile lists (64, 32, 16 rows)
    XGRP_ = rt_.buffer(nsub * size_t(2 * c_.E) * 24);
    XTILE_ = rt_.buffer(nsub * size_t(tile_cap_) * 8);
    const size_t ecols = size_t((c_.eng_max_n - 1) * c_.eng_heads), erow = ecols * size_t(c_.eng_hd + c_.eng_hd / 32);
    ERAW_[0] = rt_.buffer(T * erow), ERAW_[1] = rt_.buffer(T * erow);
    // sub-chunk scratch
    X_ = F(P * D), XQ_ = F(P * D), Y_ = F(P * D);
    QA_ = F(P * size_t(c_.QR)), Q_ = F(P * size_t(c_.H) * size_t(c_.HD)), KVp_ = F(P * size_t(c_.HD));
    OA_ = F(P * size_t(c_.OG) * size_t(c_.OR));
    CKV_ = F(P * size_t(c_.HD)), CSC_ = F(P * size_t(c_.HD)), LAT_ = F(P * size_t(c_.HD)), IK_ = F(P * size_t(c_.IHD));
    IQ_ = F(P * size_t(c_.IH) * size_t(c_.IHD)), IW_ = F(P * size_t(c_.IH));
    isc_floats_ = size_t(64) << 20;  // index scores: rows x visible entries, in row chunks
    ISC_ = F(isc_floats_);
    BS_ = F(P * size_t(cstride_));
    RL_ = F(P * size_t(c_.E));
    yrow_ = F(A), ident_ = F(A), ones_ = F(A);
    {
      int32_t* id = static_cast<int32_t*>([ident_ contents]);
      float* on = static_cast<float*>([ones_ contents]);
      for (size_t i = 0; i < A; ++i) id[i] = int32_t(i), on[i] = 1.0f;
    }
    XG_ = F(A * 2 * size_t(c_.IE)), XU_ = F(16), XH_ = F(A * size_t(c_.IE)), XP_ = F(A * D);
    HCP_ = F(P * size_t(4 * c_.D / 1024) * 25);
    SG_ = F(P * size_t(c_.IE)), SU_ = F(P * size_t(c_.IE)), SH_ = F(P * size_t(c_.IE));
    EE_ = F(P * ecols * size_t(c_.eng_hd));
    EKV_ = F(P * D * 5);
    XN_ = F(D), LOG_ = F(size_t(c_.V));
  }

  void alloc_seq(Seq& s) {
    const size_t HD = size_t(c_.HD);
    s.ring = Buf(rt_.buffer(size_t(c_.L) * size_t(c_.WIN) * HD * 2));
    seq_bytes_ = size_t(c_.L) * size_t(c_.WIN) * HD * 2;
    s.comp.clear(), s.idxk.clear(), s.cap.clear();
    for (int r : src_ratio_) {
      const int cap = max_ctx_ / r + 1;
      s.cap.push_back(cap);
      s.comp.push_back(Buf(rt_.buffer(size_t(cap) * HD * 2)));
      s.idxk.push_back(Buf(rt_.buffer(size_t(cap) * size_t(c_.IHD) * 2)));
      seq_bytes_ += size_t(cap) * (HD + size_t(c_.IHD)) * 2;
    }
    s.pend = Buf(rt_.buffer(src_ratio_.size() * 2 * 8 * HD * 4));
    reset(s);
  }

  Seq& seq(int s) {
    if (s < 0 || s + 1 >= int(seqs_.size())) fail("deepseek_v41: bad slot " + std::to_string(s));
    return seqs_[size_t(s)];
  }
  void reset(Seq& s) {
    s.len = 0;
    s.hist.clear();
  }
  void fork(Seq& d, const Seq& s) {
    if (&d == &s) return;
    d.len = s.len;
    d.hist = s.hist;
    // copy what the source holds: the rings, the compressed entries, the pending rows
    std::memcpy(ptr(d.ring), ptr(s.ring), size_t(c_.L) * size_t(c_.WIN) * size_t(c_.HD) * 2);
    for (size_t i = 0; i < src_ratio_.size(); ++i) {
      const size_t n = size_t(s.len / src_ratio_[i]) + 1;
      std::memcpy(ptr(d.comp[i]), ptr(s.comp[i]), n * size_t(c_.HD) * 2);
      std::memcpy(ptr(d.idxk[i]), ptr(s.idxk[i]), n * size_t(c_.IHD) * 2);
    }
    std::memcpy(ptr(d.pend), ptr(s.pend), src_ratio_.size() * 2 * 8 * size_t(c_.HD) * 4);
  }

  // ---------------------------------------------------------------- encoding
  struct Enc2 {
    Runtime& rt;
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> e = nil;
    double gpu_ms = 0;
    int flushes = 0;
    explicit Enc2(Runtime& r) : rt(r) { begin(); }
    void begin() {
      cb = [rt.queue commandBufferWithUnretainedReferences];
      e = [cb computeCommandEncoder];
    }
    void flush() {
      [e endEncoding];
      Residency::get().commit();
      [cb commit];
      [cb waitUntilCompleted];
      if ([cb status] != MTLCommandBufferStatusCompleted) {
        NSError* err = [cb error];
        fail(std::string("GPU command buffer failed: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
      }
      gpu_ms += ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
      ++flushes;
      begin();
    }
    void end() {
      [e endEncoding];
      Residency::get().commit();
      [cb commit];
      [cb waitUntilCompleted];
      if ([cb status] != MTLCommandBufferStatusCompleted) {
        NSError* err = [cb error];
        fail(std::string("GPU command buffer failed: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
      }
      gpu_ms += ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
      ++flushes;
      e = nil;
    }
    // ESPETERO_DS4_PROFILE=1: every dispatch in its own command buffer, timed by kernel.
    std::map<std::string, std::pair<double, int>>* prof = nullptr;
    const char* cur = "";
    void pipe(const char* n) {
      [e setComputePipelineState:rt.kernel(n)];
      cur = n;
    }
    void timed() {
      if (!prof) return;
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      auto& v = (*prof)[cur];
      v.first += ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
      v.second += 1;
      begin();
    }
    void set(int i, id<MTLBuffer> b, size_t off = 0) { [e setBuffer:b offset:off atIndex:i]; }
    void set(int i, const Buf& b, size_t extra = 0) { [e setBuffer:b.b offset:b.off + extra atIndex:i]; }
    template <class T>
    void bytes(int i, const T& v) {
      [e setBytes:&v length:sizeof(T) atIndex:i];
    }
    void groups(size_t x, size_t y, size_t z, size_t tx) {
      if (!x || !y || !z) return;
      [e dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(tx, 1, 1)];
      timed();
    }
    void threads(size_t x, size_t y, size_t z, size_t tx, size_t ty = 1) {
      if (!x || !y || !z) return;
      [e dispatchThreads:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
      timed();
    }
  };

  struct GemvP {
    int N, K, M, xs, ys, grp, gstr, accum, idx, nrow_tg;
  };
  // fmt: 0 fp8, 1 fp4, 2 bf16, 3 fp32
  void gemv(Enc2& E, int fmt, const Buf& W, const Buf& S, id<MTLBuffer> X, size_t xoff, int xs, id<MTLBuffer> Y,
            size_t yoff, int ys, int N, int K, int M, int grp = 0, int gstr = 0, bool accum = false,
            id<MTLBuffer> xrow = nil, size_t xro = 0, id<MTLBuffer> yrow = nil, size_t yro = 0, id<MTLBuffer> ysc = nil,
            size_t yso = 0) {
    if (M <= 0 || N <= 0) return;
    if (fmt == 0 && M > gemm_min_ && !xrow && N % 64 == 0 && K % 32 == 0) {  // many rows: simdgroup GEMM
      struct {
        int M, N, K, xs, ys, grp, gstr, accum;
      } p{M, N, K, xs, ys, grp, gstr, accum ? 1 : 0};
      E.pipe("ds4_gemm_f8");
      E.set(0, W);
      E.set(1, S);
      E.set(2, X, xoff);
      E.set(3, Y, yoff);
      E.bytes(4, p);
      E.groups(size_t(N / 64), size_t((M + 63) / 64), 1, 128);
      return;
    }
    static const char* names[4] = {"ds4_gemv_f8", "ds4_gemv_f4", "ds4_gemv_bf16", "ds4_gemv_f32"};
    E.pipe(names[fmt]);
    const int nrow = 8;
    GemvP p{N, K, M, xs, ys, grp, gstr, accum ? 1 : 0, xrow ? 1 : 0, nrow};
    E.set(0, W);
    E.set(1, S.b ? S : W);
    E.set(2, X, xoff);
    E.set(3, Y, yoff);
    E.bytes(4, p);
    E.set(5, xrow ? xrow : ident_, xro);
    E.set(6, yrow ? yrow : ident_, yro);
    E.set(7, ysc ? ysc : ones_, yso);
    E.groups(size_t((N + nrow - 1) / nrow), size_t((M + 7) / 8), 1, 256);
  }
  void gemv8(Enc2& E, const Mat8& w, id<MTLBuffer> X, int xs, id<MTLBuffer> Y, int ys, int M, size_t xoff = 0,
             size_t yoff = 0) {
    gemv(E, 0, w.w, w.s, X, xoff, xs, Y, yoff, ys, w.N, w.K, M);
  }
  void gemvb(Enc2& E, const MatB& w, id<MTLBuffer> X, int xs, id<MTLBuffer> Y, int ys, int M, size_t xoff = 0,
             size_t yoff = 0) {
    gemv(E, 2, w.w, Buf{}, X, xoff, xs, Y, yoff, ys, w.N, w.K, M);
  }
  void quant(Enc2& E, id<MTLBuffer> x, size_t off, int rows, int K, int stride, int mode) {
    static const bool off_ = env_on("ESPETERO_DS4_NOQUANT");  // development: parity without rounding flips
    if (off_) return;
    struct {
      int rows, K, stride, mode;
    } p{rows, K, stride, mode};
    E.pipe("ds4_quant");
    E.set(0, x, off);
    E.bytes(1, p);
    const int B = mode == 2 ? 16 : 32;
    E.threads(size_t(rows) * size_t(K / B), 1, 1, 256);
  }
  void norm(Enc2& E, id<MTLBuffer> x, size_t off, const Buf& w, int rows, int heads, int n, int stride) {
    struct {
      int rows, heads, n, stride;
      float eps;
    } p{rows, heads, n, stride, c_.eps};
    E.pipe("ds4_rmsnorm");
    E.set(0, x, off);
    E.set(1, w);
    E.bytes(2, p);
    E.groups(size_t((rows * heads + 7) / 8), 1, 1, 256);
  }
  void rope(Enc2& E, id<MTLBuffer> x, size_t off, id<MTLBuffer> posb, size_t poff, const Buf& invf, int rows, int heads,
            int hd, int stride, bool inverse) {
    struct {
      int rows, heads, hd, stride, npair, inverse;
    } p{rows, heads, hd, stride, c_.RD / 2, inverse ? 1 : 0};
    E.pipe("ds4_rope");
    E.set(0, x, off);
    E.set(1, posb, poff);
    E.set(2, invf);
    E.bytes(3, p);
    E.threads(size_t(c_.RD / 2), size_t(heads), size_t(rows), 32, 1);
  }
  void copy(Enc2& E, id<MTLBuffer> src, size_t so, id<MTLBuffer> dst, size_t dof, size_t bytes) {
    const uint32_t n = uint32_t(bytes / 16);
    E.pipe("ds4_copy");
    E.set(0, src, so);
    E.set(1, dst, dof);
    E.bytes(2, n);
    E.threads(n, 1, 1, 256);
  }
  void store_half(Enc2& E, id<MTLBuffer> src, int src_row0, int src_stride, const Buf& dst, size_t dst_off, int rows,
                  int n, int base, int mod) {
    struct {
      int rows, n, src_stride, base, mod, src_row0;
    } p{rows, n, src_stride, base, mod, src_row0};
    E.pipe("ds4_store_half");
    E.set(0, src);
    E.set(1, dst, dst_off);
    E.bytes(2, p);
    E.threads(size_t(n), size_t(rows), 1, 256);
  }

  struct HcP {
    int D, iters;
    float eps, hc_eps;
  };
  HcP hcp() const { return HcP{c_.D, c_.hc_iters, c_.eps, c_.hc_eps}; }

  // ---- hyper-connections over rows [r0, r0 + M) of the pass
  size_t hrow(int r0) const { return size_t(r0) * 4 * size_t(c_.D) * 4; }
  void hc_mix(Enc2& E, const Buf& fn, const Buf& base, const Buf& scale, id<MTLBuffer> out, int r0, int M) {
    E.pipe("ds4_hc_part");
    E.set(0, H_, hrow(r0));
    E.set(1, fn);
    E.set(2, HCP_);
    E.bytes(3, hcp());
    E.groups(size_t(4 * c_.D / 1024), size_t(M), 1, 256);
    E.pipe("ds4_hc_fin");
    E.set(0, HCP_);
    E.set(1, scale);
    E.set(2, base);
    E.set(3, out, size_t(r0) * 24 * 4);
    E.bytes(4, hcp());
    E.threads(size_t(M), 1, 1, 32);
  }
  // X = norm(sum_j pre_j H_j) for rows [r0, r0 + M); pre from `mix` rows r0...
  void hc_pre(Enc2& E, id<MTLBuffer> mix, const Buf& w, id<MTLBuffer> X, size_t xoff, int r0, int M) {
    E.pipe("ds4_hc_pre");
    E.set(0, H_, hrow(r0));
    E.set(1, mix, size_t(r0) * 24 * 4);
    E.set(2, w);
    E.set(3, X, xoff);
    E.bytes(4, hcp());
    E.groups(size_t(M), 1, 1, 256);
  }
  // H rows [r0, r0 + M) take Y (rows 0..M of the scratch) with post / comb from `mix`.
  void hc_post(Enc2& E, id<MTLBuffer> mix, int r0, int M) {
    E.pipe("ds4_hc_post");
    E.set(0, H_, hrow(r0));
    E.set(1, Y_);
    E.set(2, mix, size_t(r0) * 24 * 4);
    E.bytes(3, hcp());
    E.threads(size_t(c_.D), size_t(M), 1, 256);
  }

  // ---------------------------------------------------------------- engram (host)
  // The pass's Engram rows as stored (fp8 codes, then e8m0 scales), read from
  // the SSD into ERAW_[g] ([T][cols][hd + hd / 32] bytes); the GPU dequantizes them.
  void engram_gather(const Seq& s, const int32_t* toks, int T) {
    const auto t0 = Clock::now();
    const int nb = eng_.max_n - 1;
    const int ncol = nb * eng_.heads, hd = c_.eng_hd, rb = hd + hd / 32;
    std::vector<int64_t> cid;
    for (int32_t t : s.hist) cid.push_back(t >= 0 && t < eng_.vocab ? int64_t(eng_.map[size_t(t)]) : int64_t(eng_.pad));
    const int h0 = int(cid.size());
    for (int i = 0; i < T; ++i)
      cid.push_back(toks[i] >= 0 && toks[i] < eng_.vocab ? int64_t(eng_.map[size_t(toks[i])]) : int64_t(eng_.pad));
    for (int g = 0; g < int(c_.engram_layers.size()); ++g) {
      const LayerW* lw = nullptr;
      for (const LayerW& w : L_)
        if (w.eg == g) lw = &w;
      if (!lw) continue;
      std::vector<int64_t> rows(size_t(T) * size_t(ncol));
      for (int i = 0; i < T; ++i) {
        const int64_t p = int64_t(s.len) + i;
        int64_t tk[8];
        for (int sh = 0; sh < eng_.max_n; ++sh) {
          const int ci = h0 + i - sh;
          tk[sh] = (p - sh < 0 || ci < 0) ? int64_t(eng_.pad) : cid[size_t(ci)];
        }
        engram_rows(eng_, g, tk, rows.data() + size_t(i) * size_t(ncol));
      }
      std::unordered_map<int64_t, int> uniq;
      std::vector<int64_t> order;
      for (int64_t r : rows)
        if (uniq.try_emplace(r, int(order.size())).second) order.push_back(r);
      std::vector<uint8_t> vals(order.size() * size_t(rb));
      const ETable& Tb = lw->tab;
      const int fd = ck_.fds[size_t(Tb.file)];
      std::atomic<bool> ok{true};
      parallel_for(order.size(), [&](size_t k) {
        const int64_t r = order[k];
        uint8_t* o = vals.data() + k * size_t(rb);
        if (r < 0 || r >= Tb.rows || !pread_all(fd, o, size_t(hd), size_t(Tb.w + uint64_t(r) * uint64_t(hd))) ||
            !pread_all(fd, o + hd, size_t(hd / 32), size_t(Tb.s + uint64_t(r) * uint64_t(hd / 32))))
          ok = false;
      });
      if (!ok) fail("deepseek_v41: Engram table read failed (row out of range or I/O error)");
      uint8_t* dst = static_cast<uint8_t*>([ERAW_[g] contents]);
      for (size_t i = 0; i < rows.size(); ++i)
        std::memcpy(dst + i * size_t(rb), vals.data() + size_t(uniq[rows[i]]) * size_t(rb), size_t(rb));
    }
    eng_read_ms_ += ms_since(t0);
  }

  // ---------------------------------------------------------------- experts (host)
  // Slots for this layer's experts `need` (sorted, distinct); missing ones are
  // read from the SSD into staging and copied into their slots by `E`
  // (flushing between staging rounds).
  std::vector<int> expert_slots(Enc2& E, int l, const std::vector<int>& need) {
    ExpertCache& x = xc_;
    const uint64_t now = ++x.clock;
    std::vector<int> slot(need.size(), -1);
    std::vector<size_t> miss;
    for (size_t i = 0; i < need.size(); ++i) {
      const int k = l * c_.E + need[i];
      const int s = x.where[size_t(k)];
      if (s >= 0) {
        slot[i] = s;
        x.stamp[size_t(s)] = now;
        ++x.hits;
      } else {
        miss.push_back(i);
        ++x.misses;
      }
    }
    // evict the least recently used slots not used by this layer
    for (size_t i : miss) {
      int best = -1;
      uint64_t bs = UINT64_MAX;
      for (int s = 0; s < x.n; ++s) {
        if (x.key[size_t(s)] < 0) {
          best = s;
          break;
        }
        if (x.stamp[size_t(s)] < bs && x.stamp[size_t(s)] != now) bs = x.stamp[size_t(s)], best = s;
      }
      if (best < 0) fail("deepseek_v41: expert cache too small for one layer");
      if (x.key[size_t(best)] >= 0) x.where[size_t(x.key[size_t(best)])] = -1;
      const int k = l * c_.E + need[i];
      x.key[size_t(best)] = k;
      x.where[size_t(k)] = best;
      x.stamp[size_t(best)] = now;
      slot[i] = best;
    }
    for (size_t b0 = 0; b0 < miss.size(); b0 += size_t(x.stage_n)) {
      const size_t b1 = std::min(miss.size(), b0 + size_t(x.stage_n));
      if (b0 > 0) E.flush();  // the previous round's copies have read the staging buffer
      const auto t0 = Clock::now();
      struct Job {
        int fd;
        uint8_t* dst;
        size_t n;
        uint64_t off;
      };
      std::vector<Job> jobs;
      uint8_t* stage = static_cast<uint8_t*>([x.stage contents]);
      const size_t piece = size_t(1) << 20;  // 1 MB reads: ~8 GB/s from the internal SSD even for a few experts
      for (size_t j = b0; j < b1; ++j) {
        const XLoc& loc = L_[size_t(l)].x[size_t(need[miss[j]])];
        uint8_t* dst = stage + (j - b0) * x.slot_bytes;
        const int fd = ck_.fds_nc[size_t(loc.file)] >= 0 ? ck_.fds_nc[size_t(loc.file)] : ck_.fds[size_t(loc.file)];
        for (size_t o = 0; o < x.wbytes; o += piece) jobs.push_back({fd, dst + o, std::min(piece, x.wbytes - o), loc.w + o});
        jobs.push_back({fd, dst + x.wbytes, x.sbytes, loc.s});
      }
      std::atomic<bool> ok{true};
      parallel_for(jobs.size(), [&](size_t i) {
        if (!pread_all(jobs[i].fd, jobs[i].dst, jobs[i].n, size_t(jobs[i].off))) ok = false;
      });
      if (!ok) fail("deepseek_v41: expert read failed");
      x.read_ms += ms_since(t0);
      x.read_gb += double(b1 - b0) * double(x.slot_bytes) / 1e9;
      for (size_t j = b0; j < b1; ++j) {
        const Buf d = x.slot(slot[miss[j]]);
        copy(E, x.stage, (j - b0) * x.slot_bytes, d.b, d.off, x.slot_bytes);
      }
    }
    return slot;
  }

  // ---------------------------------------------------------------- forward
  // T rows of one sequence at positions s.len ... (T <= T_), layer by layer:
  // every layer runs over all rows (in sub-chunks of P_ rows on the GPU)
  // before the next one, so each layer's experts come from the SSD once per
  // pass. Logits of the last row into `logits` when given.
  void forward_rows(Seq& s, const int32_t* toks, int T, float* logits) {
    const Cfg& c = c_;
    const int D = c.D, K = c.K, p0 = s.len;
    if (p0 + T > max_ctx_) fail("deepseek_v41: context full (" + std::to_string(max_ctx_) + " tokens)");
    for (int i = 0; i < T; ++i) {
      if (toks[i] < 0 || toks[i] >= c.V) fail("deepseek_v41: token id out of range");
      static_cast<int32_t*>([ids_ contents])[i] = toks[i];
      static_cast<int32_t*>([pos_ contents])[i] = p0 + i;
    }
    if (!no_engram_) engram_gather(s, toks, T);
    const int nsub = (T + P_ - 1) / P_;
    std::vector<int> topk_mode(size_t(nsub), 0);
    Enc2 E(rt_);
    if (profile_ || env_on("ESPETERO_DS4_PROFILE")) E.prof = &prof_;
    {
      const int32_t p[2] = {D, 4};
      E.pipe("ds4_embed");
      E.set(0, ids_);
      E.set(1, embed_);
      E.set(2, H_);
      E.bytes(3, p);
      E.threads(size_t(D), size_t(T), 1, 256);
    }
    id<MTLBuffer> prev = mix0_;
    int cur = 0;
    for (int l = 0; l < c.L; ++l) {
      const LayerW& w = L_[size_t(l)];
      // ---- phase 1: Engram, attention, the MoE input and its routing, per sub-chunk
      for (int ci = 0; ci < nsub; ++ci) {
        const int r0 = ci * P_, M = std::min(P_, T - r0);
        if (w.eg >= 0) {
          const int ncol = (c.eng_max_n - 1) * c.eng_heads, Ke = w.e_wkv.K;
          struct {
            int rows, ncol, hd;
          } ep{M, ncol, c.eng_hd};
          E.pipe("ds4_engram_rows");
          E.set(0, ERAW_[w.eg], size_t(r0) * size_t(ncol) * size_t(c.eng_hd + c.eng_hd / 32));
          E.set(1, EE_);
          E.bytes(2, ep);
          E.threads(size_t(Ke), size_t(M), 1, 256);
          quant(E, EE_, 0, M, Ke, Ke, 0);
          gemv8(E, w.e_wkv, EE_, Ke, EKV_, 5 * D, M);
          E.pipe("ds4_engram");
          E.set(0, H_, hrow(r0));
          E.set(1, EKV_);
          E.set(2, w.e_wqk);
          E.bytes(3, hcp());
          E.groups(size_t(M), 1, 1, 256);
        }
        hc_mix(E, w.hca_fn, w.hca_base, w.hca_scale, mixA_, r0, M);
        hc_pre(E, prev, w.attn_norm, X_, 0, r0, M);
        attention(E, s, l, r0, M, p0 + r0, topk_mode[size_t(ci)]);
        if (!dump_dir_.empty()) dump_rows(E, "a" + std::to_string(l) + "_" + std::to_string(ci), Y_, size_t(M) * size_t(D));
        hc_post(E, mixA_, r0, M);
        hc_mix(E, w.hcf_fn, w.hcf_base, w.hcf_scale, mixF_[cur], r0, M);
        hc_pre(E, mixA_, w.ffn_norm, XF_, size_t(r0) * size_t(D) * 4, r0, M);
        moe_route(E, l, r0, M);
      }
      E.flush();  // routing on the host
      const int32_t* sel = static_cast<const int32_t*>([SEL_ contents]);
      std::vector<char> used(size_t(c.E), 0);
      for (int i = 0; i < T * K; ++i) {
        if (sel[i] < 0 || sel[i] >= c.E) fail("deepseek_v41: bad routing");
        used[size_t(sel[i])] = 1;
      }
      std::vector<int> need;
      for (int e = 0; e < c.E; ++e)
        if (used[size_t(e)]) need.push_back(e);
      if (!dump_dir_.empty())
        if (FILE* f = std::fopen((dump_dir_ + "/p" + std::to_string(dump_pass_) + "_route" + std::to_string(l) + ".i32").c_str(), "wb")) {
          std::fwrite(sel, 4, size_t(T * K), f);
          std::fclose(f);
        }
      const std::vector<int> slots = expert_slots(E, l, need);
      std::vector<int> slot_of(size_t(c.E), -1);
      for (size_t i = 0; i < need.size(); ++i) slot_of[size_t(need[i])] = slots[i];
      // ---- phase 3: shared and routed experts, per sub-chunk
      for (int ci = 0; ci < nsub; ++ci) {
        const int r0 = ci * P_, M = std::min(P_, T - r0);
        moe_experts(E, l, ci, r0, M, slot_of);
        if (!dump_dir_.empty()) dump_rows(E, "f" + std::to_string(l) + "_" + std::to_string(ci), Y_, size_t(M) * size_t(D));
        hc_post(E, mixF_[cur], r0, M);
      }
      prev = mixF_[cur];
      cur ^= 1;
      if (!dump_dir_.empty()) dump_rows(E, "h" + std::to_string(l), H_, size_t(T) * 4 * size_t(D));
    }
    if (logits && !head_.b) std::memset([LOG_ contents], 0, size_t(c.V) * 4);
    if (logits && head_.b) {
      hc_pre(E, prev, norm_, XN_, 0, T - 1, 1);
      gemv(E, 2, Buf(head_), Buf{}, XN_, 0, D, LOG_, 0, c.V, c.V, D, 1);
    }
    E.end();
    gpu_ms_ += E.gpu_ms;
    flushes_ += uint64_t(E.flushes);
    for (int i = 0; i < T; ++i) s.hist.push_back(toks[i]);
    const size_t keep = size_t(std::max(0, eng_.max_n - 1));
    if (s.hist.size() > keep) s.hist.erase(s.hist.begin(), s.hist.end() - long(keep));
    s.len = p0 + T;
    ++dump_pass_;
    if (logits) std::memcpy(logits, [LOG_ contents], size_t(c.V) * 4);
  }

  int dump_pass_ = 0;
  void dump_rows(Enc2& E, const std::string& name, id<MTLBuffer> b, size_t n) {
    E.flush();
    const std::string path = dump_dir_ + "/p" + std::to_string(dump_pass_) + "_" + name + ".f32";
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
      std::fwrite([b contents], 4, n, f);
      std::fclose(f);
    }
  }

  // Attention of rows [r0, r0 + M) (positions p0 ...): X_ in, Y_ out (scratch rows 0..M).
  void attention(Enc2& E, Seq& s, int l, int r0, int M, int p0, int& topk_mode) {
    const Cfg& c = c_;
    const LayerW& w = L_[size_t(l)];
    const int D = c.D, HD = c.HD, QR = c.QR, HQ = c.H * c.HD;
    const Buf& invf = w.ratio ? invf_yarn_ : invf_plain_;
    const size_t po = size_t(r0) * 4;  // pos_ rows of this sub-chunk
    copy(E, X_, 0, XQ_, 0, size_t(M) * size_t(D) * 4);
    quant(E, XQ_, 0, M, D, D, 0);
    gemv8(E, w.wq_a, XQ_, D, QA_, QR, M);
    norm(E, QA_, 0, w.q_norm, M, 1, QR, QR);
    quant(E, QA_, 0, M, QR, QR, 0);
    gemv8(E, w.wq_b, QA_, QR, Q_, HQ, M);
    rope(E, Q_, 0, pos_, po, invf, M, c.H, HD, HQ, false);
    gemv8(E, w.wkv, XQ_, D, KVp_, HD, M);
    norm(E, KVp_, 0, w.kv_norm, M, 1, HD, HD);
    rope(E, KVp_, 0, pos_, po, invf, M, 1, HD, HD, false);
    quant(E, KVp_, 0, M, HD, HD, 0);
    if (w.ratio > 0 && w.kv_src) {
      const int r = w.ratio, si = w.src;
      const int g0 = p0 / r, G = (p0 + M) / r - g0;
      if (G > s.cap[size_t(si)] - g0) fail("deepseek_v41: compressed KV full");
      int32_t* gp = static_cast<int32_t*>([gpos_ contents]) + r0;  // this sub-chunk's own rows
      for (int g = 0; g < G; ++g) gp[g] = (g0 + g) * r;
      if (r == 1) {
        gemvb(E, w.c_wkv, X_, D, LAT_, HD, M);
      } else {
        gemvb(E, w.c_wkv, X_, D, CKV_, HD, M);
        gemvb(E, w.c_wgate, X_, D, CSC_, HD, M);
        if (G > 0) {
          struct {
            int M, n, ratio, p0, g0;
          } p{M, HD, r, p0, g0};
          E.pipe("ds4_compress");
          E.set(0, CKV_);
          E.set(1, CSC_);
          E.set(2, s.pend, size_t(si) * 2 * 8 * size_t(HD) * 4);
          E.set(3, s.pend, size_t(si) * 2 * 8 * size_t(HD) * 4 + 8 * size_t(HD) * 4);
          E.set(4, LAT_);
          E.bytes(5, p);
          E.threads(size_t(HD), size_t(G), 1, 256);
        }
      }
      if (G > 0) {
        norm(E, LAT_, 0, w.c_norm, G, 1, HD, HD);
        gemvb(E, w.i_wk, LAT_, HD, IK_, c.IHD, G);  // index keys from the RoPE-free latent
        norm(E, IK_, 0, w.i_knorm, G, 1, c.IHD, c.IHD);
        rope(E, IK_, 0, gpos_, po, invf, G, 1, c.IHD, c.IHD, false);
        quant(E, IK_, 0, G, c.IHD, c.IHD, 1);
        store_half(E, IK_, 0, c.IHD, s.idxk[size_t(si)], 0, G, c.IHD, g0, 0);
        rope(E, LAT_, 0, gpos_, po, invf, G, 1, HD, HD, false);
        quant(E, LAT_, 0, G, HD, HD, 2);
        store_half(E, LAT_, 0, HD, s.comp[size_t(si)], 0, G, HD, g0, 0);
      }
      if (r > 1) {  // the incomplete group's rows wait in the slot
        const int end = p0 + M, t0 = end - end % r;
        for (int t = std::max(t0, p0); t < end; ++t) {
          const size_t pb = size_t(si) * 2 * 8 * size_t(HD) * 4;
          copy(E, CKV_, size_t(t - p0) * size_t(HD) * 4, s.pend.b, s.pend.off + pb + size_t(t % r) * size_t(HD) * 4, size_t(HD) * 4);
          copy(E, CSC_, size_t(t - p0) * size_t(HD) * 4, s.pend.b, s.pend.off + pb + 8 * size_t(HD) * 4 + size_t(t % r) * size_t(HD) * 4,
               size_t(HD) * 4);
        }
      }
    }
    const size_t io = size_t(r0) * size_t(c.ITOPK) * 4;  // IDX_ rows of this sub-chunk
    if (w.ratio > 0 && w.idx_src) {
      const int r = w.ratio, si = w.src;
      const int nvis_max = (p0 + M) / r;
      if (nvis_max > c.ITOPK) {
        // candidate blocks: only beyond cand_blocks x cand_bs visible entries do they restrict anything
        const bool cand = c.cand_src >= 0 && nvis_max > c.cand_blocks * c.cand_bs && (w.cand_src || w.uses_cand);
        const int IQN = c.IH * c.IHD;
        gemv8(E, w.i_wq_b, QA_, QR, IQ_, IQN, M);
        rope(E, IQ_, 0, pos_, po, invf, M, c.IH, c.IHD, IQN, false);
        quant(E, IQ_, 0, M * c.IH, c.IHD, c.IHD, 1);
        gemvb(E, w.i_wproj, X_, D, IW_, c.IH, M);
        {
          const float sc = 1.0f / std::sqrt(float(c.IHD)) / std::sqrt(float(c.IH));
          const float p[2] = {float(M * c.IH), sc};
          E.pipe("ds4_scale");
          E.set(0, IW_);
          E.bytes(1, p);
          E.threads(size_t(M * c.IH), 1, 1, 256);
        }
        const int ncols = nvis_max;
        const int rows_per = std::max(1, int(isc_floats_ / size_t(ncols)));
        for (int m0 = 0; m0 < M; m0 += rows_per) {
          const int mr = std::min(rows_per, M - m0);
          struct {
            int M, nh, hd, ratio, ncols;
          } ip{mr, c.IH, c.IHD, r, ncols};
          E.pipe("ds4_index_score");
          E.set(0, IQ_, size_t(m0) * size_t(IQN) * 4);
          E.set(1, IW_, size_t(m0) * size_t(c.IH) * 4);
          E.set(2, s.idxk[size_t(si)]);
          E.set(3, ISC_);
          E.set(4, pos_, po + size_t(m0) * 4);
          E.bytes(5, ip);
          E.groups(size_t((ncols + 63) / 64), size_t(mr), 1, 64);
          if (cand) {
            struct {
              int ncols, stride, bs, kb, ratio, cstride;
            } cp{ncols, ncols, c.cand_bs, c.cand_blocks, r, cstride_};
            const size_t co = size_t(r0 + m0) * size_t(cstride_);
            if (w.cand_src) {
              E.pipe("ds4_cand");
              E.set(0, ISC_);
              E.set(1, BS_);
              E.set(2, CAND_, co);
              E.set(3, pos_, po + size_t(m0) * 4);
              E.bytes(4, cp);
              E.groups(size_t(mr), 1, 1, 1024);
            } else {
              E.pipe("ds4_cand_mask");
              E.set(0, ISC_);
              E.set(1, CAND_, co);
              E.bytes(2, cp);
              E.threads(size_t(ncols), size_t(mr), 1, 256);
            }
          }
          struct {
            int ncols, k, stride, ratio;
          } tp{ncols, c.ITOPK, ncols, r};
          E.pipe("ds4_topk");
          E.set(0, ISC_);
          E.set(1, IDX_, io + size_t(m0) * size_t(c.ITOPK) * 4);
          E.set(2, pos_, po + size_t(m0) * 4);
          E.bytes(3, tp);
          E.groups(size_t(mr), 1, 1, 1024);
          if (m0 + mr < M) E.flush();  // ISC_ is reused by the next rows
        }
        topk_mode = c.ITOPK;
      } else {
        topk_mode = 0;
      }
    }
    // ---- attention over the window and the compressed entries
    {
      struct {
        int M, H, win, p0, ratio, topk, n_comp;
        float scale;
      } ap{M, c.H, c.WIN, p0, w.ratio, w.ratio ? topk_mode : 0, 0, 1.0f / std::sqrt(float(HD))};
      E.pipe("ds4_attn");
      E.set(0, Q_);
      E.set(1, KVp_);
      E.set(2, s.ring, size_t(l) * size_t(c.WIN) * size_t(HD) * 2);
      if (w.ratio > 0) E.set(3, s.comp[size_t(w.src)]);
      else E.set(3, s.ring);
      E.set(4, IDX_, io);
      E.set(5, w.sink);
      E.set(6, Q_);  // in place: each simdgroup reads its query before writing
      E.set(7, pos_, po);
      E.bytes(8, ap);
      E.groups(size_t((c.H + 7) / 8), size_t(M), 1, 256);
    }
    rope(E, Q_, 0, pos_, po, invf, M, c.H, HD, HQ, true);
    quant(E, Q_, 0, M, HQ, HQ, 0);
    gemv(E, 0, w.wo_a.w, w.wo_a.s, Q_, 0, HQ, OA_, 0, c.OG * c.OR, w.wo_a.N, w.wo_a.K, M, c.OR, w.wo_a.K);
    quant(E, OA_, 0, M, c.OG * c.OR, c.OG * c.OR, 0);
    gemv8(E, w.wo_b, OA_, c.OG * c.OR, Y_, D, M);
    // ---- the window ring keeps the last WIN rows
    const int k0 = std::max(0, M - c.WIN);
    store_half(E, KVp_, k0, HD, s.ring, size_t(l) * size_t(c.WIN) * size_t(HD) * 2, M - k0, HD, p0 + k0, c.WIN);
  }

  // Router of rows [r0, r0 + M): XF_ rows -> SEL_ / RW_ rows.
  void moe_route(Enc2& E, int l, int r0, int M) {
    const Cfg& c = c_;
    const LayerW& w = L_[size_t(l)];
    gemvb(E, w.gate, XF_, c.D, RL_, c.E, M, size_t(r0) * size_t(c.D) * 4);
    struct {
      int M, E, k;
      float rs;
    } p{M, c.E, c.K, c.route_scale};
    E.pipe("ds4_route");
    E.set(0, RL_);
    E.set(1, w.gate_bias);
    E.set(2, SEL_, size_t(r0) * size_t(c.K) * 4);
    E.set(3, RW_, size_t(r0) * size_t(c.K) * 4);
    E.bytes(4, p);
    E.groups(size_t(M), 1, 1, 32);
  }

  // Shared and routed experts of rows [r0, r0 + M) into Y_ (sub-chunk ci; the
  // routing on the host; slot_of: each expert's cache slot for this layer).
  void moe_experts(Enc2& E, int l, int ci, int r0, int M, const std::vector<int>& slot_of) {
    const Cfg& c = c_;
    const LayerW& w = L_[size_t(l)];
    const int D = c.D, IE = c.IE, K = c.K;
    copy(E, XF_, size_t(r0) * size_t(D) * 4, XQ_, 0, size_t(M) * size_t(D) * 4);
    quant(E, XQ_, 0, M, D, D, 0);
    gemv8(E, w.sh1, XQ_, D, SG_, IE, M);
    gemv8(E, w.sh3, XQ_, D, SU_, IE, M);
    swiglu(E, SG_, SU_, SH_, M, IE, nil);
    quant(E, SH_, 0, M, IE, IE, 0);
    gemv8(E, w.sh2, SH_, IE, Y_, D, M);
    // assignments of the sub-chunk grouped by expert (tables at this sub-chunk's offsets)
    const int32_t* sel = static_cast<const int32_t*>([SEL_ contents]) + size_t(r0) * size_t(K);
    const float* rw = static_cast<const float*>([RW_ contents]) + size_t(r0) * size_t(K);
    const int A = M * K;
    std::vector<int> order(static_cast<size_t>(A));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sel[a] < sel[b]; });
    const size_t ao = size_t(r0) * size_t(K);
    int32_t* xr = static_cast<int32_t*>([xrow_ contents]) + ao;
    float* ws = static_cast<float*>([wsc_ contents]) + ao;
    int32_t* am = static_cast<int32_t*>([amap_ contents]) + ao;
    std::vector<int> need, start;
    for (int a = 0; a < A; ++a) {
      const int src = order[size_t(a)];
      const int e = sel[src];
      if (need.empty() || need.back() != e) need.push_back(e), start.push_back(a);
      xr[a] = src / K;
      ws[a] = rw[src];
      am[src] = a;
    }
    start.push_back(A);
    struct XGroup {
      uint32_t part, pad;
      uint64_t off;
      int32_t a0, n;
    };
    const size_t go = size_t(ci) * size_t(2 * c.E) * sizeof(XGroup);
    XGroup* grp = reinterpret_cast<XGroup*>(static_cast<uint8_t*>([XGRP_ contents]) + go);
    for (size_t i = 0; i < need.size(); ++i) {
      const int sl = slot_of[size_t(need[i])];
      grp[i] = XGroup{uint32_t(xc_.part(sl)), 0, uint64_t(xc_.offset(sl)), start[i], start[i + 1] - start[i]};
    }
    // Many rows: the GEMM over tiles of 64, 32 or 16 rows of a group (tile lists per
    // size); a few (decode): the grouped GEMV.
    int nt[3] = {0, 0, 0};
    const size_t to = size_t(ci) * size_t(tile_cap_) * 8;
    int32_t* tl[3];
    for (int q = 0; q < 3; ++q) tl[q] = reinterpret_cast<int32_t*>(static_cast<uint8_t*>([XTILE_ contents]) + to) + 2 * q * (tile_cap_ / 3);
    const bool tiled = M > gemm_min_;
    if (tiled)
      for (size_t i = 0; i < need.size(); ++i) {
        const int n = start[i + 1] - start[i];
        for (int q = 0; q < n;) {
          const int rem = n - q, cls = rem > 32 ? 0 : rem > 16 ? 1 : 2, bm = 64 >> cls;
          tl[cls][2 * nt[cls]] = int32_t(i), tl[cls][2 * nt[cls] + 1] = q, ++nt[cls];
          q += bm;
        }
      }
    struct {
      int N, K, xs, ys;
      uint64_t wbytes, sbytes;
      int mode, nrow_tg;
    } xp{IE, D, D, 2 * IE, uint64_t(IE) * uint64_t(D) / 2, uint64_t(IE) * uint64_t(D) / 32, 0, 8};
    auto xgemv = [&](id<MTLBuffer> X, id<MTLBuffer> Y) {
      const int rows = xp.mode == 0 ? 2 * xp.N : xp.N;
      static const char* names[3] = {"ds4_xgemm", "ds4_xgemm32", "ds4_xgemm16"};
      for (int q = 0; q < 3 && tiled; ++q) {
        if (!nt[q]) continue;
        E.pipe(names[q]);
        for (int b = 0; b < 3; ++b) E.set(b, xc_.bufs[size_t(std::min<int>(b, int(xc_.bufs.size()) - 1))]);
        E.set(3, XGRP_, go);
        E.set(4, X);
        E.set(5, Y);
        E.set(6, xrow_, ao * 4);
        E.bytes(7, xp);
        E.set(8, XTILE_, to + size_t(2 * q * (tile_cap_ / 3)) * 4);
        E.groups(size_t(rows / 64), size_t(nt[q]), 1, 128);
      }
      if (!tiled) {
        E.pipe("ds4_xgemv");
        for (int b = 0; b < 3; ++b) E.set(b, xc_.bufs[size_t(std::min<int>(b, int(xc_.bufs.size()) - 1))]);
        E.set(3, XGRP_, go);
        E.set(4, X);
        E.set(5, Y);
        E.set(6, xrow_, ao * 4);
        E.bytes(7, xp);
        E.groups(size_t((rows + xp.nrow_tg - 1) / xp.nrow_tg), need.size(), 1, 256);
      }
    };
    xgemv(XQ_, XG_);
    {
      struct {
        int rows, n, gs, us, os;
        float lim;
      } p{A, IE, 2 * IE, 2 * IE, IE, c.swiglu};
      E.pipe("ds4_swiglu_w");
      E.set(0, XG_);
      E.set(1, XG_, size_t(IE) * 4);
      E.set(2, XH_);
      E.bytes(3, p);
      E.set(4, wsc_, ao * 4);
      E.threads(size_t(IE), size_t(A), 1, 256);
    }
    quant(E, XH_, 0, A, IE, IE, 0);
    xp = {D, IE, IE, D, xp.wbytes, xp.sbytes, 1, 8};
    xgemv(XH_, XP_);
    const int32_t p[2] = {D, K};
    E.pipe("ds4_moe_sum");
    E.set(0, XP_);
    E.set(1, Y_);
    E.set(2, amap_, ao * 4);
    E.bytes(3, p);
    E.threads(size_t(D), size_t(M), 1, 256);
  }

  void swiglu(Enc2& E, id<MTLBuffer> G, id<MTLBuffer> U, id<MTLBuffer> O, int rows, int n, id<MTLBuffer>) {
    struct {
      int rows, n, gs, us, os;
      float lim;
    } p{rows, n, n, n, n, c_.swiglu};
    E.pipe("ds4_swiglu");
    E.set(0, G);
    E.set(1, U);
    E.set(2, O);
    E.bytes(3, p);
    E.threads(size_t(n), size_t(rows), 1, 256);
  }

  // ---------------------------------------------------------------- trees
  void run_seq(Seq& s, const int32_t* toks, int n, Readout* out) {
    for (int i = 0; i < n; i += T_) {
      const int m = std::min(T_, n - i);
      const bool last = i + m == n;
      if (last && out) out->logits.resize(size_t(c_.V));
      forward_rows(s, toks + i, m, last && out ? out->logits.data() : nullptr);
    }
  }

  void run_tree(const TreeSpec& t, Readout& out) {
    if (t.n_leaves > 0) fail("deepseek_v41: leaves are not supported");
    if (t.images && t.n_images > 0) fail("deepseek_v41: images are not supported yet");
    if (t.root_len <= 0) fail("deepseek_v41: empty root");
    Seq& scratch = seqs_.back();
    const int save_len = t.save_len > 0 ? std::min(t.save_len, t.root_len) : t.root_len;
    Seq* tgt = t.save >= 0 ? &seq(t.save) : t.work >= 0 ? &seq(t.work) : &scratch;
    if (t.parent >= 0) fork(*tgt, seq(t.parent));
    else reset(*tgt);
    const bool want = t.score_root;
    const int first = (t.save >= 0) ? save_len : t.root_len;
    run_seq(*tgt, t.root, first, want && first == t.root_len ? &out : nullptr);
    if (first < t.root_len) {
      Seq* rest = (t.work >= 0 && &seq(t.work) != tgt) ? &seq(t.work) : &scratch;
      fork(*rest, *tgt);
      run_seq(*rest, t.root + first, t.root_len - first, want ? &out : nullptr);
    } else if (t.work >= 0 && &seq(t.work) != tgt) {
      fork(seq(t.work), *tgt);
    }
    if (!want || out.logits.empty()) return;
    finish_readout(t, out);
  }

  static uint64_t tk_mix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
  }
  static float tk_uniform(uint64_t seed, uint64_t pos, uint64_t id) {
    uint64_t x = tk_mix(seed + 0x9E3779B97F4A7C15ull);
    x = tk_mix(x ^ (pos * 0xD1B54A32D192ED03ull));
    x = tk_mix(x ^ id);
    return float(x >> 40) * 0x1.0p-24f + 0x1.0p-25f;
  }

  void finish_readout(const TreeSpec& t, Readout& o) {
    const int V = c_.V;
    const float* L = o.logits.data();
    int am = 0;
    float mx = L[0];
    for (int i = 1; i < V; ++i)
      if (L[i] > mx) mx = L[i], am = i;
    o.argmax = am;
    double se = 0;
    for (int i = 0; i < V; ++i) se += std::exp(double(L[i] - mx));
    o.lse = float(double(mx) + std::log(se));
    if (t.out_argmax) t.out_argmax[0] = am;
    if (t.out_lse) t.out_lse[0] = o.lse;
    const int K = std::min(32, std::max(t.topk, t.topk_min));
    if (K > 0) {
      const float it = (t.topk > 0 && t.temperature > 0) ? 1.0f / t.temperature : 1.0f;
      std::vector<int32_t> id(static_cast<size_t>(V));
      std::iota(id.begin(), id.end(), 0);
      std::partial_sort(id.begin(), id.begin() + K, id.end(), [&](int a, int b) {
        const float xa = L[a] * it, xb = L[b] * it;
        return xa > xb || (xa == xb && a < b);
      });
      o.tk_ids.assign(id.begin(), id.begin() + K);
      o.tk_x.resize(size_t(K));
      for (int i = 0; i < K; ++i) o.tk_x[size_t(i)] = L[o.tk_ids[size_t(i)]] * it;
      const float M = o.tk_x[0];
      float tot = 0.0f;
      for (int i = 0; i < V; ++i) tot += std::exp(L[i] * it - M);
      o.tk_max = M, o.tk_total = tot;
      o.sampled = -1;
      if (t.topk > 0 && t.gpu_draw) {
        const int head = std::min(K, std::max(1, t.topk));
        float pr[32];
        for (int q = 0; q < head; ++q) pr[q] = std::exp(o.tk_x[size_t(q)] - M) / tot;
        int keep = head;
        if (t.min_p > 0.0f) {
          const float fl = t.min_p * pr[0];
          keep = 0;
          while (keep < head && pr[keep] >= fl) ++keep;
        }
        if (t.top_p < 1.0f) {
          float sum = 0.0f;
          for (int q = 0; q < keep; ++q) sum += pr[q];
          const float target = t.top_p * sum;
          float acc = 0.0f;
          int cut = keep;
          for (int q = 0; q < keep; ++q) {
            acc += pr[q];
            if (acc >= target) {
              cut = q + 1;
              break;
            }
          }
          keep = cut;
        }
        float best = -INFINITY;
        int sampled = o.tk_ids[0];
        for (int q = 0; q < std::max(keep, 1); ++q) {
          if (!(pr[q] > 0.0f)) continue;
          const float g = -std::log(-std::log(tk_uniform(t.seed, uint64_t(t.draw_pos0), uint64_t(int64_t(o.tk_ids[size_t(q)])))));
          const float sc = std::log(pr[q]) + g;
          if (sc > best) best = sc, sampled = o.tk_ids[size_t(q)];
        }
        o.sampled = sampled;
      }
    }
    if (t.out_cand_logits && t.cand_offsets && t.cand_ids) {
      for (int j = t.cand_offsets[0]; j < t.cand_offsets[1]; ++j) t.out_cand_logits[j] = L[t.cand_ids[j]];
    }
  }
};

}  // namespace espetero::native::ds4
