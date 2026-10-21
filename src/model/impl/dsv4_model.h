// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// DeepSeek-V4.1 as a NativeModel (dsv4.h has the formats, docs/deepseek_v41.md
// the math). Plain roots only (no leaves, drafts or images yet); every tree
// runs as its own forward, layer by layer, waiting for the GPU after each
// layer's routing to bring the missing experts in from the SSD.
#pragma once

#include <array>
#include <condition_variable>
#include <deque>
#include <thread>
#include <mach/mach.h>
#include <sys/sysctl.h>

#include "../api.h"
#include "../../util/paths.h"
#include "dsv4.h"

namespace espetero::native::ds4 {

class Dsv4Model final : public NativeModel {
 public:
  Dsv4Model(const std::string& path, const LoadOptions& o, const std::string& kernels) {
    const auto t0 = Clock::now();
    verbose_ = o.verbose || env_on("ESPETERO_DS4_VERBOSE");
    // Memory the system can give before this model: free, inactive, speculative and purgeable pages
    // (other processes' wired GPU memory is not available).
    double sys_avail = 0;
    {
      vm_statistics64_data_t vs;
      mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
      if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vs), &cnt) == KERN_SUCCESS)
        sys_avail = (double(vs.free_count) + double(vs.inactive_count) + double(vs.speculative_count) +
                     double(vs.purgeable_count)) * double(vm_kernel_page_size);
    }
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
    want_vision_ = o.vision;
    ck_.open(path, partial_);
    eng_ = read_engram_const(path + "/espetero_engram.bin", c_);
    max_ctx_ = std::max(256, o.max_ctx > 0 ? o.max_ctx : 32768);
    // Rows of a pass (a prompt up to T_ tokens runs as one, layer by layer) and
    // of a GPU sub-chunk (P_).
    P_ = 1024;
    if (const char* v = std::getenv("ESPETERO_DS4_SUB")) P_ = std::clamp(std::atoi(v), 8, 4096);
    // A pass reads nearly every expert of its layers: a prompt in two passes reads them twice (~190 GB). 32 K
    // rows (~140 KB of scratch each) cost 2.3 GB of expert cache next to 16 K: +0.5 misses per decoded token.
    T_ = std::min(max_ctx_, std::max(32768, o.max_rows));
    if (const char* v = std::getenv("ESPETERO_DS4_ROWS")) T_ = std::clamp(std::atoi(v), 1, max_ctx_);
    P_ = std::min(P_, T_);
    load_weights();
    const double wgb = double(rt_.arena->total()) / 1e9;
    alloc_scratch();
    // ---- sequence slots: window rings, compressed KV and index keys for max_ctx tokens each
    int nslots = o.slots > 0 ? o.slots : 6;
    nuser_ = nslots;
    seqs_.resize(size_t(nslots) + size_t(kScratch));  // scratch slots after the user ones: trees that save nothing
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
      if (gb <= 0) {
        // what the GPU working set allows, keeping ~14 GB of the Mac for macOS, apps and the file cache
        // macOS compressed and swapped tens of GB when the process wired 128 of 137 GB (prefill 3x slower):
        // keep ~24 GB of the Mac out of it
        gb = std::min({ws - mine - 3e9, double(ram) - mine - 24e9, sys_avail - mine - 6e9}) / 1e9;
        if (gb < 20)
          std::fprintf(stderr, "deepseek_v41: only %.1f GB of memory available for the expert cache (%.1f GB free "
                               "before loading): every token will read most experts from the SSD\n", gb, sys_avail / 1e9);
      }
      const double all = double(c_.L) * c_.E * double(xc_.slot_bytes) / 1e9;
      gb = std::max(1.0, std::min(gb, all));
      make_cache(gb);
      if (!partial_) {
        uint64_t hsh = 1469598103934665603ull;
        for (unsigned char ch : path) hsh = (hsh ^ ch) * 1099511628211ull;
        char nm[64];
        std::snprintf(nm, sizeof nm, "/dsv41-%016llx.experts", (unsigned long long)hsh);
        const std::string dir = paths::user_cache_dir();
        ::mkdir(dir.c_str(), 0755);
        warm_path_ = dir + nm;
        warm_load();
      }
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
    info_.max_readouts = max_reads_;
    info_.weight_bytes = rt_.arena->total();
    info_.kv_bytes = seq_bytes_ * size_t(nslots);
    info_.slot_state_bytes = seq_bytes_;
    info_.leaves = true;
    info_.prompt_lookup = false;  // drafts verified as leaves read several tokens' experts from the SSD
    info_.commit = false;
    info_.hidden_states = false;
    info_.mtp = false;
    info_.vision = vision_;
    if (vision_) {
      info_.vision_layout = "deepseek";
      info_.vision_patch = c_.v_patch;
      info_.vision_merge = c_.v_down;
      info_.vision_patch_dim = 3 * c_.v_patch * c_.v_patch;
      info_.vision_temporal = 1;
      info_.image_token_id = c_.image_token;
      info_.vision_max_tokens = c_.v_max_tok;
      info_.vision_min_pixels = c_.v_min_px;
    }
    info_.load_ms = ms_since(t0);
    json::Value ex = json::Value::object();
    ex.set("expert_cache_slots", xc_.n);
    ex.set("expert_cache_gb", double(xc_.n) * double(xc_.slot_bytes) / 1e9);
    info_.extra = ex;
  }

  ~Dsv4Model() override {
    if (st_th_.joinable()) {
      {
        std::lock_guard<std::mutex> lk(st_mu_);
        st_stop_ = true;
      }
      st_cv_.notify_all();
      st_th_.join();
    }
  }

  const ModelInfo& info() const override { return info_; }

  // ---- slots
  int slot_alloc() override {
    for (size_t i = 0; i < size_t(nuser_); ++i)
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
    for (size_t i = 0; i < size_t(nuser_); ++i) n += seqs_[i].used ? 0 : 1;
    return n;
  }
  int free_kv_tokens() const override {
    int n = 0;
    for (size_t i = 0; i < size_t(nuser_); ++i) n += max_ctx_ - (seqs_[i].used ? seqs_[i].len : 0);
    return n;
  }
  int pass_kv_tokens(const std::vector<TreeSpec>&) const override { return 0; }
  // Every slot owns its rows (no shared blocks): freeing a slot gives back its length.
  int freed_kv_tokens(const std::vector<int>& slots) const override {
    int n = 0;
    for (int s : slots) n += seqs_.at(size_t(s)).used ? seqs_[size_t(s)].len : 0;
    return n;
  }
  // Checkpoint files: header, the window rings, each KV source's compressed
  // entries and index keys up to the length, the compressor's pending rows.
  struct CkptHead {
    char magic[8];
    int32_t version, len, nsrc, nhist;
    int32_t hist[8];
    uint64_t ring_bytes, pend_bytes;
  };
  bool slot_save(int slot, const std::string& path) override {
    const Seq& q = seq(slot);
    CkptHead h{};
    std::memcpy(h.magic, "ESPDS4CK", 8);
    h.version = 1, h.len = q.len, h.nsrc = int32_t(src_ratio_.size()), h.nhist = int32_t(q.hist.size());
    for (size_t i = 0; i < q.hist.size() && i < 8; ++i) h.hist[i] = q.hist[i];
    h.ring_bytes = uint64_t(c_.L) * uint64_t(c_.WIN) * uint64_t(c_.HD) * 2;
    h.pend_bytes = uint64_t(src_ratio_.size()) * 2 * 8 * uint64_t(c_.HD) * 4;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1 && std::fwrite(ptr(q.ring), 1, h.ring_bytes, f) == h.ring_bytes;
    for (size_t i = 0; ok && i < src_ratio_.size(); ++i) {
      const size_t n = size_t(q.len / src_ratio_[i]);
      ok = std::fwrite(ptr(q.comp[i]), 1, n * size_t(c_.HD) * 2, f) == n * size_t(c_.HD) * 2 &&
           std::fwrite(ptr(q.idxk[i]), 1, n * size_t(c_.IHD) * 2, f) == n * size_t(c_.IHD) * 2;
    }
    ok = ok && std::fwrite(ptr(q.pend), 1, h.pend_bytes, f) == h.pend_bytes;
    return std::fclose(f) == 0 && ok;
  }
  int slot_load(const std::string& path) override {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return -1;
    struct Closer {
      FILE* f;
      ~Closer() { std::fclose(f); }
    } closer{f};
    CkptHead h{};
    if (std::fread(&h, sizeof h, 1, f) != 1 || std::memcmp(h.magic, "ESPDS4CK", 8) != 0 || h.version != 1 ||
        h.nsrc != int32_t(src_ratio_.size()) || h.len < 0 || h.len > max_ctx_ || h.nhist < 0 || h.nhist > 8 ||
        h.ring_bytes != uint64_t(c_.L) * uint64_t(c_.WIN) * uint64_t(c_.HD) * 2 ||
        h.pend_bytes != uint64_t(src_ratio_.size()) * 2 * 8 * uint64_t(c_.HD) * 4)
      return -1;
    const int slot = slot_alloc();
    if (slot < 0) return -2;
    Seq& q = seq(slot);
    bool ok = std::fread(ptr(q.ring), 1, h.ring_bytes, f) == h.ring_bytes;
    for (size_t i = 0; ok && i < src_ratio_.size(); ++i) {
      const size_t n = size_t(h.len / src_ratio_[i]);
      ok = std::fread(ptr(q.comp[i]), 1, n * size_t(c_.HD) * 2, f) == n * size_t(c_.HD) * 2 &&
           std::fread(ptr(q.idxk[i]), 1, n * size_t(c_.IHD) * 2, f) == n * size_t(c_.IHD) * 2;
    }
    ok = ok && std::fread(ptr(q.pend), 1, h.pend_bytes, f) == h.pend_bytes;
    if (!ok) {
      slot_free(slot);
      return -1;
    }
    q.len = h.len;
    q.hist.assign(h.hist, h.hist + h.nhist);
    return slot;
  }
  std::string state_signature() const override { return "deepseek_v41-1|" + info_.path + "|" + std::to_string(max_ctx_); }

  // ---- passes
  void run(const std::vector<TreeSpec>& trees, PassStats* stats) override {
    @autoreleasepool {
      const auto t0 = Clock::now();
      run_trees(trees);
      int rows = 0;
      for (const TreeSpec& t : trees) rows += t.root_len;
      if (stats) {
        stats->wall_ms = stats->gpu_ms = ms_since(t0);
        stats->rows = rows;
        stats->readouts = int(trees.size());
      }
    }
  }

  // Decode chains. With the DSpark head, every round drafts a block per tree,
  // verifies it with one pass of the target and keeps the drafts the target
  // would have produced itself (its own tokens: argmax, or the position-keyed
  // draw), so the output is the plain chain's; the slots end exactly `steps`
  // tokens further, rejected rows rolled back. ESPETERO_DS4_SPEC=1 turns it on.
  void run_chain(const std::vector<TreeSpec>& first, int steps, int32_t* out, PassStats* stats) override {
    @autoreleasepool {
      const auto t0 = Clock::now();
      const size_t n = first.size();
      bool spec = dspark_;
      for (const TreeSpec& t : first)
        spec = spec && t.parent >= 0 && t.save == t.parent && t.root_len == 1 && t.n_leaves == 0;
      int rows = 0;
      if (!spec) {
        std::vector<TreeSpec> cur = first;
        std::vector<int32_t> tok(n);
        for (int j = 0; j < steps; ++j) {
          for (size_t t = 0; t < n; ++t) {
            if (j > 0) {
              tok[t] = out[size_t(j - 1) * n + t];
              cur[t].root = &tok[t];
              cur[t].draw_pos0 = first[t].draw_pos0 + j;
            }
            cur[t].out_argmax = nullptr;
          }
          run_trees(cur);
          for (size_t t = 0; t < n; ++t) out[size_t(j) * n + t] = draw_token(cur[t], out_[t].back().logits.data(), cur[t].draw_pos0);
          rows += int(n);
        }
      } else {
        rows = spec_chain(first, steps, out);
      }
      if (stats) {
        stats->wall_ms = stats->gpu_ms = ms_since(t0);
        stats->rows = rows;
        stats->readouts = int(n) * steps;
      }
    }
  }

  // The target's token from a readout's logits: argmax, or the GPU draw's
  // rule (top-k head, min-p, top-p, Gumbel noise keyed by seed, position, id).
  int32_t draw_token(const TreeSpec& t, const float* L, int64_t pos) {
    const int V = c_.V;
    if (!(t.topk > 0 && t.gpu_draw)) {
      int am = 0;
      for (int i = 1; i < V; ++i)
        if (L[i] > L[am]) am = i;
      return am;
    }
    TreeSpec u = t;
    u.draw_pos0 = pos;
    u.out_argmax = nullptr, u.out_lse = nullptr, u.out_cand_logits = nullptr;
    Readout r;
    r.logits.assign(L, L + V);
    finish_readout(u, r);
    return r.sampled >= 0 ? r.sampled : r.argmax;
  }

  int spec_rounds_ = 0, spec_drafts_ = 0, spec_accepted_ = 0;
  double spec_draft_ms_ = 0, spec_verify_ms_ = 0;
  // ESPETERO_DS4_SPEC_MIN: the predicted survival a draft needs to be verified (default below)
  static double spec_min() {
    static const double v = std::getenv("ESPETERO_DS4_SPEC_MIN") ? std::atof(std::getenv("ESPETERO_DS4_SPEC_MIN")) : 0.8;
    return v;
  }
  static bool spec_on() {
    const char* v = std::getenv("ESPETERO_DS4_SPEC");
    return v && std::atoi(v) != 0;
  }

  int spec_chain(const std::vector<TreeSpec>& first, int steps, int32_t* out) {
    const size_t n = first.size();
    const int B = c_.ds_block, V = c_.V;
    std::vector<int> produced(n, 0);
    std::vector<int32_t> cur(n);
    for (size_t t = 0; t < n; ++t) cur[t] = first[t].root[0];
    std::vector<std::vector<int32_t>> rowtok(n);
    std::vector<std::vector<float>> lg(n);
    int rows = 0;
    while (true) {
      std::vector<size_t> act;
      for (size_t t = 0; t < n; ++t)
        if (produced[t] < steps) act.push_back(t);
      if (act.empty()) break;
      // ---- drafts for the trees that can use some
      std::vector<int> nd(n, 0);
      std::vector<size_t> dr;
      for (size_t t : act) {
        nd[t] = std::min(B, steps - produced[t] - 1);
        if (nd[t] > 0) dr.push_back(t);
      }
      std::vector<std::array<int32_t, 8>> d(n);
      std::vector<std::array<float, 8>> cfr(n);
      if (!dr.empty()) {
        std::vector<Seq*> ss;
        std::vector<int32_t> tk;
        for (size_t t : dr) ss.push_back(&seq(first[t].parent)), tk.push_back(cur[t]);
        std::vector<std::array<float, 8>> cf;
        const auto td = Clock::now();
        const std::vector<std::array<int32_t, 8>> got = draft(ss, tk, &cf);
        spec_draft_ms_ += ms_since(td);
        for (size_t i = 0; i < dr.size(); ++i) {
          d[dr[i]] = got[i];
          cfr[dr[i]] = cf[i];
          // verify a draft only while the chance that it and every draft before it are accepted (the
          // confidence head's) beats what one more verified row costs, relative to a token
          double surv = 1.0;
          int k = 0;
          while (k < nd[dr[i]]) {
            surv *= 1.0 / (1.0 + std::exp(-double(cf[i][size_t(k)])));
            if (surv < spec_min()) break;
            ++k;
          }
          nd[dr[i]] = k;
        }
      }
      // ---- verification: [cur, drafts...] per tree, logits at every row
      std::vector<Seg> segs;
      std::vector<std::vector<uint8_t>> bak(n);
      std::vector<std::vector<int32_t>> hist0(n);
      for (size_t t : act) {
        Seq& s = seq(first[t].parent);
        rowtok[t].assign(1, cur[t]);
        for (int j = 1; j <= nd[t]; ++j) rowtok[t].push_back(d[t][size_t(j)]);
        lg[t].resize(size_t(1 + nd[t]) * size_t(V));
        Seg g;
        g.s = &s, g.toks = rowtok[t].data(), g.n = 1 + nd[t];
        g.logits = lg[t].data(), g.nlog = 1 + nd[t], g.verify = nd[t] > 0;
        if (g.verify) ring_backup(s, g.n, bak[t]);
        hist0[t] = s.hist;
        segs.push_back(g);
      }
      const auto tv = Clock::now();
      run_segments(segs);
      spec_verify_ms_ += ms_since(tv);
      for (size_t t : act) {
        Seq& s = seq(first[t].parent);
        const int R = 1 + nd[t], p0 = s.len - R;
        int take = 0;
        while (take < R) {
          const int32_t y = draw_token(first[t], lg[t].data() + size_t(take) * size_t(V), first[t].draw_pos0 + produced[t] + take);
          out[size_t(produced[t] + take) * n + t] = y;
          cur[t] = y;
          ++take;
          if (produced[t] + take >= steps) break;
          if (take >= R || rowtok[t][size_t(take)] != y) break;  // the next row was this draft: kept only if it matches
        }
        spec_drafts_ += nd[t];
        spec_accepted_ += take - 1;
        static FILE* slog = std::getenv("ESPETERO_DS4_SPEC_LOG") ? std::fopen(std::getenv("ESPETERO_DS4_SPEC_LOG"), "a") : nullptr;
        if (slog && nd[t] > 0) {  // per round: the confidence logits of the verified drafts, then how many were accepted
          for (int j = 0; j < nd[t]; ++j) std::fprintf(slog, "%.3f ", cfr[t][size_t(j)]);
          std::fprintf(slog, "| %d\n", take - 1);
          std::fflush(slog);
        }
        if (take < R) rollback(s, p0, R, take, bak[t], hist0[t], rowtok[t]);
        produced[t] += take;
        rows += R;
      }
      ++spec_rounds_;
    }
    return rows;
  }

  // Window ring rows a verification of `rows` rows will overwrite (every layer).
  void ring_backup(const Seq& s, int rows, std::vector<uint8_t>& bak) {
    const size_t rb = size_t(c_.HD) * 2, lb = size_t(c_.WIN) * rb;
    bak.resize(size_t(c_.L) * size_t(rows) * rb);
    const uint8_t* ring = static_cast<const uint8_t*>(ptr(s.ring));
    for (int l = 0; l < c_.L; ++l)
      for (int j = 0; j < rows; ++j)
        std::memcpy(bak.data() + (size_t(l) * size_t(rows) + size_t(j)) * rb, ring + size_t(l) * lb + size_t((s.len + j) % c_.WIN) * rb, rb);
  }

  // Keep the first `keep` of a verification's `rows` rows (positions p0 ...):
  // the ring rows of the others come back, the compressors' pending rows are
  // those of the new end, the length and the n-gram history follow.
  void rollback(Seq& s, int p0, int rows, int keep, const std::vector<uint8_t>& bak, const std::vector<int32_t>& hist0,
                const std::vector<int32_t>& toks) {
    const size_t rb = size_t(c_.HD) * 2, lb = size_t(c_.WIN) * rb, HD = size_t(c_.HD);
    uint8_t* ring = static_cast<uint8_t*>(ptr(s.ring));
    for (int l = 0; l < c_.L; ++l)
      for (int j = keep; j < rows; ++j)
        std::memcpy(ring + size_t(l) * lb + size_t((p0 + j) % c_.WIN) * rb, bak.data() + (size_t(l) * size_t(rows) + size_t(j)) * rb, rb);
    const int end = p0 + keep;
    for (size_t si = 0; si < src_ratio_.size(); ++si) {
      const int r = src_ratio_[si];
      if (r < 2) continue;
      float* pend = static_cast<float*>(ptr(s.pend)) + si * 2 * 8 * HD;
      const float* vp = static_cast<const float*>(ptr(s.vpend)) + si * 2 * 8 * HD;
      for (int t = end - end % r; t < end; ++t) {  // the incomplete group's rows
        if (t < p0) continue;  // from before the verification: still in place
        std::memcpy(pend + size_t(t % r) * HD, vp + size_t(t - p0) * HD, HD * 4);
        std::memcpy(pend + 8 * HD + size_t(t % r) * HD, vp + 8 * HD + size_t(t - p0) * HD, HD * 4);
      }
    }
    s.len = end;
    s.hist = hist0;
    for (int j = 0; j < keep; ++j) s.hist.push_back(toks[size_t(j)]);
    const size_t keep_h = size_t(std::max(0, eng_.max_n - 1));
    if (s.hist.size() > keep_h) s.hist.erase(s.hist.begin(), s.hist.end() - long(keep_h));
  }

  // DSpark drafts for slots whose next token is tok[i] (not yet processed):
  // the target states of the positions not yet in the draft rings enter them,
  // then the draft layers run over [tok, noise ...] and the Markov head picks
  // ds_block drafts. Returns [i][1 .. ds_block].
  // conf (optional): per block, each draft's acceptance logit given the ones before (conf[j] for draft j + 1)
  std::vector<std::array<int32_t, 8>> draft(const std::vector<Seq*>& ss, const std::vector<int32_t>& tok,
                                            std::vector<std::array<float, 8>>* conf = nullptr) {
    const Cfg& c = c_;
    const int D = c.D, HD = c.HD, nt = int(c.ds_targets.size()), B = c.ds_block;
    // ---- the target states into the rings
    {
      int rows = 0;
      struct Ins {
        Seq* s;
        int from, cnt, r0;
      };
      std::vector<Ins> ins;
      float* dsx = static_cast<float*>([DSX_ contents]);
      int32_t* dpos = static_cast<int32_t*>([DPOS_ contents]);
      for (Seq* s : ss) {
        const int i = s->len - 1;
        const int from = std::max({s->ds_len, i - c.WIN + 1, 0});
        const int cnt = i - from + 1;
        if (cnt <= 0) continue;
        if (rows + cnt > kDsRows) fail("deepseek_v41: too many draft states in one pass");
        const float* mh = static_cast<const float*>(ptr(s->mh));
        for (int q = 0; q < cnt; ++q) {
          std::memcpy(dsx + size_t(rows + q) * size_t(nt) * size_t(D), mh + size_t((from + q) % 128) * size_t(nt) * size_t(D),
                      size_t(nt) * size_t(D) * 4);
          dpos[rows + q] = from + q;
        }
        ins.push_back({s, from, cnt, rows});
        rows += cnt;
      }
      if (rows > 0) {
        Enc2 E(rt_);
        quant(E, DSX_, 0, rows, nt * D, nt * D, 0);
        gemv8(E, ds_main_proj_, DSX_, nt * D, DSM_, D, rows);
        norm(E, DSM_, 0, ds_main_norm_, rows, 1, D, D);
        for (int k = 0; k < c.n_mtp; ++k) {
          const LayerW& w = M_[size_t(k)];
          copy(E, DSM_, 0, DSQ_, 0, size_t(rows) * size_t(D) * 4);
          quant(E, DSQ_, 0, rows, D, D, 0);
          gemv8(E, w.wkv, DSQ_, D, DKV_, HD, rows);
          norm(E, DKV_, 0, w.kv_norm, rows, 1, HD, HD);
          rope(E, DKV_, 0, DPOS_, 0, invf_plain_, rows, 1, HD, HD, false);
          quant(E, DKV_, 0, rows, HD, HD, 0);
          for (const Ins& in : ins)
            store_half(E, DKV_, in.r0, HD, in.s->dsring, size_t(k) * size_t(c.WIN) * size_t(HD) * 2, in.cnt, HD, in.from, c.WIN);
        }
        E.end();
        gpu_ms_ += E.gpu_ms;
      }
      for (Seq* s : ss) s->ds_len = s->len;
    }
    // ---- the draft blocks
    std::vector<std::vector<int32_t>> blk(ss.size());
    std::vector<std::vector<float>> dl(ss.size());
    std::vector<Seg> segs;
    int32_t* dt = static_cast<int32_t*>([DTOK_ contents]);
    for (size_t i = 0; i < ss.size(); ++i) {
      blk[i].assign(size_t(B), c.ds_noise);
      blk[i][0] = tok[i];
      dt[i * 8] = tok[i];
      dl[i].resize(size_t(B) * size_t(c.V));
      Seg g;
      g.s = ss[i], g.toks = blk[i].data(), g.n = B, g.logits = dl[i].data(), g.nlog = B;
      segs.push_back(g);
    }
    if (segs.size() * size_t(B) > size_t(max_reads_)) fail("deepseek_v41: too many draft blocks in one pass");
    forward(segs, true);
    std::vector<std::array<int32_t, 8>> res(ss.size());
    for (size_t i = 0; i < ss.size(); ++i)
      for (int j = 0; j <= B; ++j) res[i][size_t(j)] = dt[i * 8 + size_t(j)];
    if (conf) {
      conf->assign(ss.size(), {});
      const float* cf = static_cast<const float*>([CONF_ contents]);
      for (size_t i = 0; i < ss.size(); ++i)
        for (int j = 0; j < B; ++j) (*conf)[i][size_t(j)] = ds_conf_.b ? cf[i * 8 + size_t(j)] : 30.0f;
    }
    return res;
  }

  const float* logits(int tree, int readout) const override {
    return out_.at(size_t(tree)).at(size_t(readout)).logits.data();
  }
  TopK topk(int tree, int readout) const override {
    TopK r;
    if (readout != 0) return r;
    if (size_t(tree) >= out_.size() || size_t(readout) >= out_[size_t(tree)].size()) return r;
    const Readout& o = out_[size_t(tree)][size_t(readout)];
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
  // The embedding rows of each image's run, back to back (float32 [run tokens][hidden]).
  void encode_images(const std::vector<ImageRef>& images, float* out, PassStats* stats) override {
    @autoreleasepool {
      if (!vision_) fail("deepseek_v41: no vision tower loaded");
      const auto t0 = Clock::now();
      size_t o = 0;
      for (const ImageRef& im : images) {
        const int n = image_run(im.grid_h, im.grid_w);
        const size_t bytes = size_t(n) * size_t(c_.D) * 4;
        if (!IMGS_ || bytes > imgs_bytes_) {
          imgs_bytes_ = std::max(imgs_bytes_, bytes * 2);
          IMGS_ = rt_.buffer(imgs_bytes_);
        }
        encode_image(im, IMGS_, 0);
        std::memcpy(out + o, [IMGS_ contents], bytes);
        o += size_t(n) * size_t(c_.D);
      }
      if (stats) stats->wall_ms = stats->gpu_ms = ms_since(t0);
    }
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
    o.set("wait_ms", wait_ms_);    // host time waiting for command buffers (GPU time + launch + wake-up)
    o.set("wake_ms", wake_ms_);    // from each command buffer's end to the host noticing
    o.set("forward_ms", fwd_ms_);  // wall time inside passes
    o.set("stream_issued", int64_t(st_issued_));
    o.set("stream_gb", double(st_bytes_.load()) / 1e9);
    o.set("stream_ms", double(st_us_.load()) / 1e3);
    o.set("prefetch_issued", int64_t(pf_issued_));
    o.set("prefetch_used", int64_t(pf_used_));
    o.set("prefetch_wasted", int64_t(pf_wasted_));
    o.set("prefetch_wait_ms", pf_wait_ms_);
    o.set("flushes", int64_t(flushes_));
    o.set("spec_rounds", spec_rounds_);
    o.set("spec_drafts", spec_drafts_);
    o.set("spec_accepted", spec_accepted_);
    o.set("spec_draft_ms", spec_draft_ms_);
    o.set("spec_verify_ms", spec_verify_ms_);
    if (!prof_.empty()) {
      json::Value k = json::Value::object();
      json::Value kn = json::Value::object();
      for (const auto& [n, v] : prof_) k.set(n, v.first), kn.set(n, int64_t(v.second));
      o.set("kernels_ms", k);
      o.set("kernels_n", kn);
    }
    return json::dump(o);
  }
  bool set_option(const std::string& name, int value) override {
    if (name == "expert_stats_reset") {
      xc_.hits = xc_.misses = 0;
      xc_.read_ms = xc_.read_gb = eng_read_ms_ = gpu_ms_ = wait_ms_ = wake_ms_ = fwd_ms_ = pf_wait_ms_ = 0;
      pf_issued_ = pf_used_ = pf_wasted_ = 0;
      flushes_ = 0;
      spec_rounds_ = spec_drafts_ = spec_accepted_ = 0;
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
  std::vector<LayerW> L_, M_;  // the backbone's layers, the DSpark draft layers
  bool dspark_ = false;
  Mat8 ds_main_proj_;
  Buf ds_main_norm_, ds_norm_;
  MatB ds_mk_embed_, ds_mk_head_;
  Buf ds_conf_;  // the confidence head (w over [state, Markov embedding]), or none
  Buf embed_, head_, norm_;
  Buf invf_plain_, invf_yarn_;
  ExpertCache xc_;
  std::vector<Seq> seqs_;
  ModelInfo info_;
  int P_ = 1024, T_ = 16384, max_ctx_ = 32768, tile_cap_ = 0, nuser_ = 0, nsub_cap_ = 1;
  static constexpr int kScratch = 8;  // trees without a slot, and leaves (in rounds)
  int max_reads_ = 64;  // logits rows per pass
  static constexpr int kDsRows = 1024;  // target states entering the draft rings per draft pass
  size_t seq_bytes_ = 0;
  double engram_gb_ = 0;
  bool verbose_ = false, profile_ = false, partial_ = false, no_engram_ = false;
  std::vector<std::vector<Readout>> out_;  // [tree][readout]: leaves first, then the root
  std::map<std::string, std::pair<double, int>> prof_;
  double eng_read_ms_ = 0, gpu_ms_ = 0, wait_ms_ = 0, wake_ms_ = 0, fwd_ms_ = 0;
  uint64_t flushes_ = 0;
  std::vector<int> src_ratio_;  // per KV source: its ratio

  // scratch (shared)
  id<MTLBuffer> ids_, pos_, gpos_, mix0_, H_, X_, XQ_, Y_, mixA_, mixF_[2], QA_, Q_, KVp_, OA_, CKV_, CSC_, LAT_, IK_, IQ_, IW_,
      ISC_, IDX_, BS_, CAND_, XGRP_, XTILE_, HCP_, RL_, SEL_, RW_, xrow_, yrow_, wsc_, amap_, ident_, ones_, XG_, XU_, XH_, XP_, SG_, SU_, SH_, EE_,
      ERAW_[2], EKV_, XF_, XN_, LOG_, DTOK_, MKB_, CONF_, DSX_, DSM_, DSQ_, DKV_, DPOS_, PSEL_, PRW_, YT_;
  static constexpr int kPredRows = 16;  // passes up to this many rows predict the next layer's experts
  static constexpr int kMaxSegs = 64;   // segments per pass whose CED rows may split a block
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

  // What every layer has: hyper-connections, attention projections, router,
  // shared expert, routed experts' locations.
  void load_layer(LayerW& w, const std::string& p, int nE, int topk) {
    const int D = c_.D, W4 = 4 * D;
    w.nE = nE, w.topk = topk;
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
    w.gate = matb(p + "ffn.gate.weight", nE, D);
    w.gate_bias = vecf(p + "ffn.gate.bias", size_t(nE));
    w.gate_bias_vl = ck_.has(p + "ffn.gate.bias_vl") ? vecf(p + "ffn.gate.bias_vl", size_t(nE)) : w.gate_bias;
    {  // the shared expert's gate and up stacked: one GEMV ([w1; w3], scales likewise)
      const TInfo &a = ck_.at(p + "ffn.shared_experts.w1.weight"), &b = ck_.at(p + "ffn.shared_experts.w3.weight");
      const TInfo &sa = ck_.at(p + "ffn.shared_experts.w1.scale"), &sb = ck_.at(p + "ffn.shared_experts.w3.scale");
      if (a.nbytes != uint64_t(c_.IE) * uint64_t(D) || b.nbytes != a.nbytes || sa.nbytes != sb.nbytes || c_.IE % 32)
        fail("deepseek_v41: " + p + "shared expert shapes");
      w.sh13.N = 2 * c_.IE, w.sh13.K = D;
      w.sh13.w = rt_.arena->alloc(size_t(a.nbytes + b.nbytes));
      ck_.read(a, ptr(w.sh13.w));
      ck_.read(b, static_cast<uint8_t*>(ptr(w.sh13.w)) + a.nbytes);
      w.sh13.s = rt_.arena->alloc(size_t(sa.nbytes + sb.nbytes));
      ck_.read(sa, ptr(w.sh13.s));
      ck_.read(sb, static_cast<uint8_t*>(ptr(w.sh13.s)) + sa.nbytes);
    }
    w.sh2 = mat8(p + "ffn.shared_experts.w2", D, c_.IE);
    w.x.resize(size_t(nE));
    const uint64_t wb = uint64_t(c_.IE) * uint64_t(D) / 2, sb = uint64_t(c_.IE) * uint64_t(D) / 32;
    for (int e = 0; e < nE; ++e) {
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
      load_layer(w, p, c_.E, c_.K);
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
    // DSpark draft head: window-only layers with their own experts, a projection of the
    // target layers' states, the final norm and the Markov head (ESPETERO_DS4_DSPARK=0: off)
    // Off by default: on a Mac whose experts come from the SSD, verifying a block
    // reads several times the experts of one token and the head's drafts are
    // accepted ~1 in 3, so decoding got slower (docs/deepseek_v41.md).
    if (c_.n_mtp > 0 && c_.ds_block > 0 && !partial_ && ck_.has("mtp.0.main_proj.weight") && spec_on()) {
      M_.resize(size_t(c_.n_mtp));
      for (int k = 0; k < c_.n_mtp; ++k) {
        const std::string p = "mtp." + std::to_string(k) + ".";
        if (c_.ratio.size() > size_t(c_.L + k) && c_.ratio[size_t(c_.L + k)] != 0)
          fail("deepseek_v41: DSpark layers with compressed attention are not supported");
        load_layer(M_[size_t(k)], p, c_.ds_E, c_.ds_K);
      }
      const int nt = int(c_.ds_targets.size());
      ds_main_proj_ = mat8("mtp.0.main_proj", D, nt * D);
      ds_main_norm_ = vecf("mtp.0.main_norm.weight", size_t(D));
      const std::string last = "mtp." + std::to_string(c_.n_mtp - 1) + ".";
      ds_norm_ = vecf(last + "norm.weight", size_t(D));
      ds_mk_embed_ = matb(last + "markov_head.embed.weight", c_.V, c_.ds_markov);
      ds_mk_head_ = matb(last + "markov_head.head.weight", c_.V, c_.ds_markov);
      if (ck_.has(last + "confidence_head.proj.weight")) ds_conf_ = vecf(last + "confidence_head.proj.weight", size_t(D + c_.ds_markov));
      dspark_ = true;
    }
    if (c_.v_layers > 0 && want_vision_ && !partial_ && ck_.has("vision.patch_embed.proj.weight")) load_vision();
    xc_.wbytes = size_t(3) * size_t(c_.IE) * size_t(c_.D) / 2;
    xc_.sbytes = size_t(3) * size_t(c_.IE) * size_t(c_.D) / 32;
    xc_.slot_bytes = xc_.wbytes + xc_.sbytes;
    if (verbose_) std::fprintf(stderr, "deepseek_v41: dense weights loaded in %.1f s\n", ms_since(t0) / 1e3);
  }

  // The expert pool: host memory locked with mlock (macOS compressed and paged a
  // GPU-private pool whenever the GPU went idle under memory pressure, then
  // decompressed tens of GB at the next pass), at most 3 buffers (the grouped
  // expert kernel binds them all).
  void make_cache(double gb) {
    ExpertCache& x = xc_;
    x.n = int(gb * 1e9 / double(x.slot_bytes));
    x.n = std::max(x.n, c_.E + 16);
    const size_t maxlen = std::min<size_t>(size_t([rt_.dev maxBufferLength]), size_t(64) << 30);
    x.per_buf = std::max(int((size_t(x.n) + 2) / 3), 1);
    x.per_buf = std::min(x.per_buf, int(maxlen / x.slot_bytes));
    if ((x.n + x.per_buf - 1) / x.per_buf > 3) x.n = 3 * x.per_buf;
    const size_t pg = size_t(::getpagesize());
    const auto t0 = Clock::now();
    bool locked = true;
    for (int s0 = 0; s0 < x.n; s0 += x.per_buf) {
      const size_t len = align_up(size_t(std::min(x.per_buf, x.n - s0)) * x.slot_bytes, pg);
      void* p = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
      if (p == MAP_FAILED) fail("deepseek_v41: cannot map the expert pool");
      if (locked) {  // in 1 GB pieces on every core (faulting 100 GB in on one thread is slow)
        const size_t piece = size_t(1) << 30;
        std::atomic<bool> ok{true};
        parallel_for((len + piece - 1) / piece, [&](size_t i) {
          if (::mlock(static_cast<uint8_t*>(p) + i * piece, std::min(piece, len - i * piece)) != 0) ok = false;
        });
        if (!ok) {
          locked = false;
          ::munlock(p, len);
          std::fprintf(stderr, "deepseek_v41: could not lock the expert pool in memory (vm.user_wire_limit?): macOS may "
                               "compress it when the GPU is idle\n");
        }
      }
      id<MTLBuffer> b = [rt_.dev newBufferWithBytesNoCopy:p length:len options:kShared deallocator:nil];
      if (!b) fail("deepseek_v41: cannot wrap the expert pool for the GPU");
      Residency::get().add(b);
      x.bufs.push_back(b);
      x.host.push_back(static_cast<uint8_t*>(p));
    }
    x.locked = locked;
    x.key.assign(size_t(x.n), -1);
    x.stamp.assign(size_t(x.n), 0);
    x.pref.assign(size_t(x.n), 0);
    x.spass.assign(size_t(x.n), 0);
    x.loading.reset(new std::atomic<int>[size_t(x.n)]);
    for (int i = 0; i < x.n; ++i) x.loading[size_t(i)].store(0);
    x.where.assign(size_t(c_.L + c_.n_mtp) * size_t(c_.E), -1);  // (layer, expert); draft layers after the backbone
    x.stage_n = 32;  // pipelined rounds: experts read per round
    if (verbose_ || ms_since(t0) > 2000)
      std::fprintf(stderr, "deepseek_v41: expert pool %s in %.1f s\n", locked ? "locked in memory" : "mapped", ms_since(t0) / 1e3);
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
    PSEL_ = F(size_t(kPredRows) * 8), PRW_ = F(size_t(kPredRows) * 8);
    IMG_ = rt_.buffer(T);
    xrow_ = F(2 * T * K), wsc_ = F(2 * T * K), amap_ = F(2 * T * K);  // two copies: pipelined rounds alternate
    IDX_ = F(T * size_t(c_.ITOPK));
    cstride_ = c_.cand_bs > 0 ? max_ctx_ / c_.cand_bs + 2 : 1;
    CAND_ = rt_.buffer(T * size_t(cstride_));
    const size_t nsub = (T + P - 1) / P + 1 + size_t(kMaxSegs);  // CED layers split blocks at segments' unused rows
    tile_cap_ = 3 * int(A / 16 + size_t(c_.E) + 1);  // three tile lists (64, 32, 16 rows)
    nsub_cap_ = int(nsub);
    XGRP_ = rt_.buffer(2 * nsub * size_t(2 * c_.E) * 24);
    XTILE_ = rt_.buffer(2 * nsub * size_t(tile_cap_) * 8);
    YT_ = F(T * D);  // a pipelined layer's MoE output, all rows
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
    SG_ = F(P * 2 * size_t(c_.IE)), SU_ = F(16), SH_ = F(P * size_t(c_.IE));
    EE_ = F(P * ecols * size_t(c_.eng_hd));
    EKV_ = F(P * D * 5);
    XN_ = F(D * size_t(max_reads_)), LOG_ = F(size_t(c_.V) * size_t(max_reads_));
    // DSpark: drafts per block, the Markov bias, the target states entering the draft rings
    DTOK_ = F(size_t(max_reads_) * 8), MKB_ = F(size_t(c_.V)), CONF_ = F(size_t(max_reads_) * 8);
    const size_t dr = size_t(kDsRows), nt = std::max<size_t>(1, c_.ds_targets.size());
    DSX_ = F(dr * nt * D), DSM_ = F(dr * D), DSQ_ = F(dr * D), DKV_ = F(dr * size_t(c_.HD)), DPOS_ = F(dr);
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
    s.vpend = Buf(rt_.buffer(src_ratio_.size() * 2 * 8 * HD * 4));
    if (dspark_) {
      s.dsring = Buf(rt_.buffer(size_t(c_.n_mtp) * size_t(c_.WIN) * HD * 2));
      s.mh = Buf(rt_.buffer(size_t(128) * c_.ds_targets.size() * size_t(c_.D) * 4));
    }
    reset(s);
  }

  Seq& seq(int s) {
    if (s < 0 || s >= nuser_) fail("deepseek_v41: bad slot " + std::to_string(s));
    return seqs_[size_t(s)];
  }
  void reset(Seq& s) {
    s.len = 0;
    s.ds_len = 0;
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
    d.ds_len = s.ds_len;
    if (dspark_) {
      std::memcpy(ptr(d.dsring), ptr(s.dsring), size_t(c_.n_mtp) * size_t(c_.WIN) * size_t(c_.HD) * 2);
      std::memcpy(ptr(d.mh), ptr(s.mh), size_t(128) * c_.ds_targets.size() * size_t(c_.D) * 4);
    }
  }

  // ---------------------------------------------------------------- encoding
  struct Enc2 {
    Runtime& rt;
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> e = nil;
    double gpu_ms = 0;
    int flushes = 0;
    explicit Enc2(Runtime& r) : rt(r) { begin(); }
    // ESPETERO_DS4_CONC=1: concurrent dispatch with a buffer barrier before every dispatch (experiment).
    bool conc = env_on("ESPETERO_DS4_CONC");
    void begin() {
      cb = [rt.queue commandBufferWithUnretainedReferences];
      e = conc ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [cb computeCommandEncoder];
    }
    void bar() {
      if (conc) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    // Commit and wait. Waits spin (a layer's command buffer is ~1 ms; a
    // blocking wait wakes ~0.1-0.3 ms late, 40 times a token);
    // ESPETERO_DS4_SPIN=0 blocks.
    double wait_ms = 0, wake_ms = 0;
    void submit_wait() {
      [e endEncoding];
      Residency::get().commit();
      const double t0 = Enc::host_seconds();
      [cb commit];
      static const bool spin = !(std::getenv("ESPETERO_DS4_SPIN") && std::atoi(std::getenv("ESPETERO_DS4_SPIN")) == 0);
      if (spin) {
        while ([cb status] < MTLCommandBufferStatusCompleted) {
          if (Enc::host_seconds() - t0 > 0.05) {  // long work: block
            [cb waitUntilCompleted];
            break;
          }
        }
      } else {
        [cb waitUntilCompleted];
      }
      const double t1 = Enc::host_seconds();
      if ([cb status] != MTLCommandBufferStatusCompleted) {
        NSError* err = [cb error];
        fail(std::string("GPU command buffer failed: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
      }
      gpu_ms += ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
      for (id<MTLCommandBuffer> c : chained) {  // completed before this one (ordered by events)
        wait_cb(c);
        gpu_ms += ([c GPUEndTime] - [c GPUStartTime]) * 1000.0;
      }
      chained.clear();
      wait_ms += (t1 - t0) * 1000.0;
      wake_ms += std::max(0.0, (t1 - [cb GPUEndTime]) * 1000.0);
      ++flushes;
    }
    void flush() {
      submit_wait();
      begin();
    }
    // Commit without waiting; the next command buffer waits on the GPU for this
    // one (an event) so their work stays ordered. Returns the committed buffer.
    id<MTLSharedEvent> ev = nil;
    uint64_t evv = 0;
    std::vector<id<MTLCommandBuffer>> chained;
    id<MTLCommandBuffer> submit_chained() {
      if (!ev) ev = [rt.dev newSharedEvent];
      [e endEncoding];
      [cb encodeSignalEvent:ev value:++evv];
      Residency::get().commit();
      [cb commit];
      id<MTLCommandBuffer> sent = cb;
      chained.push_back(sent);
      cb = [rt.queue commandBufferWithUnretainedReferences];
      [cb encodeWaitForEvent:ev value:evv];
      e = conc ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [cb computeCommandEncoder];
      return sent;
    }
    void wait_cb(id<MTLCommandBuffer> c) {
      [c waitUntilCompleted];
      if ([c status] != MTLCommandBufferStatusCompleted) {
        NSError* err = [c error];
        fail(std::string("GPU command buffer failed: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
      }
    }
    void end() {
      submit_wait();
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
    void raw(int i, const void* p, size_t n) { [e setBytes:p length:n atIndex:i]; }
    void groups(size_t x, size_t y, size_t z, size_t tx) {
      if (!x || !y || !z) return;
      bar();
      [e dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(tx, 1, 1)];
      timed();
    }
    void threads(size_t x, size_t y, size_t z, size_t tx, size_t ty = 1) {
      if (!x || !y || !z) return;
      bar();
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
    static const bool g1 = env_on("ESPETERO_DS4_GEMV1");  // one output row per simdgroup always
    const int nrow = fmt == 0 && M == 1 && !grp && !g1 ? 32 : 8;  // one row of X: 4 output rows per simdgroup
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
  static bool noquant() {
    static const bool v = env_on("ESPETERO_DS4_NOQUANT");
    return v;
  }
  // In place, one simdgroup per vector: RMSNorm (w non-null), RoPE on the last dims (npair > 0), fp8 rounding.
  void vec_post(Enc2& E, id<MTLBuffer> x, size_t off, const Buf* w, id<MTLBuffer> posb, size_t poff, const Buf* invf,
                int rows, int heads, int n, int stride, bool rope, bool inverse, bool q) {
    struct {
      int rows, heads, n, stride;
      float eps;
      int norm, npair, inverse, quant;
    } p{rows, heads, n, stride, c_.eps, w ? 1 : 0, rope ? c_.RD / 2 : 0, inverse ? 1 : 0, q && !noquant() ? 1 : 0};
    E.pipe("ds4_vec_post");
    E.set(0, x, off);
    E.set(1, w ? *w : invf_plain_);
    E.set(2, posb, poff);
    E.set(3, invf ? *invf : invf_plain_);
    E.bytes(4, p);
    E.groups(size_t((rows * heads + 7) / 8), 1, 1, 256);
  }
  void quant_copy(Enc2& E, id<MTLBuffer> src, size_t so, int sstride, id<MTLBuffer> dst, size_t dof, int dstride, int rows, int K) {
    if (noquant()) {
      if (sstride == K && dstride == K) return copy(E, src, so, dst, dof, size_t(rows) * size_t(K) * 4);
      fail("deepseek_v41: strided copy without quantization");
    }
    struct {
      int rows, K, sstride, dstride;
    } p{rows, K, sstride, dstride};
    E.pipe("ds4_quant_copy");
    E.set(0, src, so);
    E.set(1, dst, dof);
    E.bytes(2, p);
    E.threads(size_t(rows) * size_t(K / 32), 1, 1, 256);
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
  void hc_post(Enc2& E, id<MTLBuffer> mix, int r0, int M, id<MTLBuffer> Ys = nil, size_t yso = 0) {
    E.pipe("ds4_hc_post");
    E.set(0, H_, hrow(r0));
    E.set(1, Ys ? Ys : Y_, Ys ? yso : 0);
    E.set(2, mix, size_t(r0) * 24 * 4);
    E.bytes(3, hcp());
    E.threads(size_t(c_.D), size_t(M), 1, 256);
  }
  // The three in one dispatch (ds4_hc_fused, bit-identical): hc_post with `pmix` (post), then from the new H the
  // next block's coefficients `fn / base / scale` into `mout` (mix) and the next sub-layer's input `X` with
  // `prem` and norm `w` (pre). ESPETERO_DS4_HCFUSE=0 keeps the separate kernels (and D != 5120 does).
  bool hc_fuse() const {
    static const bool on = !(std::getenv("ESPETERO_DS4_HCFUSE") && std::atoi(std::getenv("ESPETERO_DS4_HCFUSE")) == 0);
    return on && c_.D == 5120;
  }
  void hc_fused(Enc2& E, int r0, int M, id<MTLBuffer> Ys, size_t yso, id<MTLBuffer> pmix, const Buf* fn, const Buf* base,
                const Buf* scale, id<MTLBuffer> mout, id<MTLBuffer> prem, const Buf* w, id<MTLBuffer> X, size_t xoff) {
    struct {
      int D, iters;
      float eps, hc_eps;
      int post, mix, pre;
    } p{c_.D, c_.hc_iters, c_.eps, c_.hc_eps, pmix ? 1 : 0, fn ? 1 : 0, w ? 1 : 0};
    if (!p.post && !p.mix && !p.pre) return;
    const size_t mo = size_t(r0) * 24 * 4;
    E.pipe("ds4_hc_fused");
    E.set(0, H_, hrow(r0));
    E.set(1, Ys ? Ys : Y_, Ys ? yso : 0);
    E.set(2, pmix ? pmix : mixA_, mo);
    E.set(3, fn ? *fn : norm_);
    E.set(4, scale ? *scale : norm_);
    E.set(5, base ? *base : norm_);
    E.set(6, mout ? mout : mixA_, mo);
    E.set(7, prem ? prem : mixA_, mo);
    E.set(8, w ? *w : norm_);
    E.set(9, X ? X : X_, xoff);
    E.bytes(10, p);
    E.groups(size_t(M), 1, 1, 256);
  }

  // The end of layer l for rows [r0, r0 + M): hc_post with `fmix` (the FFN's post / comb), and with the fused
  // kernel also the next layer's input mix (its hca block -> mixA_) and attention input (X_), unless an Engram
  // changes H first or this is the last layer. pre_done tells the next layer what was done.
  void hc_end(Enc2& E, int l, int r0, int M, id<MTLBuffer> Ys, size_t yso, id<MTLBuffer> fmix, const std::vector<LayerW>& LW,
              bool& pre_done) {
    if (!hc_fuse()) {
      hc_post(E, fmix, r0, M, Ys, yso);
      return;
    }
    const bool next = l + 1 < int(LW.size()) && LW[size_t(l + 1)].eg < 0;
    const LayerW* n = next ? &LW[size_t(l + 1)] : nullptr;
    hc_fused(E, r0, M, Ys, yso, fmix, n ? &n->hca_fn : nullptr, n ? &n->hca_base : nullptr, n ? &n->hca_scale : nullptr, mixA_,
             fmix, n ? &n->attn_norm : nullptr, X_, 0);
    pre_done = next;
  }

  // ---------------------------------------------------------------- vision
  struct VBlk {
    Buf n1, n2, qkv_w, qkv_b, wo_w, wo_b, w1, w2;
  };
  struct VisionW {
    Buf pe_w, pe_b, norm, al_w1, al_b1, al_w2, al_b2, img_start, img_end, img_nl, inv_freq;
    std::vector<VBlk> blocks;
    int kpad = 0;  // patch length padded to a multiple of 32
  } vis_;
  bool vision_ = false, want_vision_ = true;
  id<MTLBuffer> VP_ = nil, VX_, VXN_, VQKV_, VQh_, VKh_, VVh_, VOh_, VO_, VH_, VG_, VU_, VA1_, VA2_, VHW_, VSEG_, IMGS_ = nil, IMG_;
  int vcap_ = 0, lcap_ = 0;  // patch rows and aligner rows the vision scratch holds
  size_t imgs_bytes_ = 0;

  void load_vision() {
    const int VD = c_.v_dim, P = c_.v_patch, K = 3 * P * P;
    vis_.kpad = (K + 31) / 32 * 32;
    {  // patch embedding, columns zero-padded to kpad
      const TInfo& t = ck_.at("vision.patch_embed.proj.weight");
      std::vector<uint16_t> w(size_t(t.nbytes / 2)), wp(size_t(VD) * size_t(vis_.kpad), 0);
      ck_.read(t, w.data());
      for (int o = 0; o < VD; ++o) std::memcpy(wp.data() + size_t(o) * size_t(vis_.kpad), w.data() + size_t(o) * size_t(K), size_t(K) * 2);
      vis_.pe_w = rt_.arena->copy(wp.data(), wp.size() * 2);
    }
    vis_.pe_b = vecf("vision.patch_embed.proj.bias", size_t(VD));
    for (int b = 0; b < c_.v_layers; ++b) {
      const std::string p = "vision.blocks." + std::to_string(b) + ".";
      VBlk B;
      B.n1 = vecf(p + "norm1.weight", size_t(VD));
      B.n2 = vecf(p + "norm2.weight", size_t(VD));
      B.qkv_w = raw(p + "attn.wqkv.weight", uint64_t(3 * VD) * uint64_t(VD) * 2);
      B.qkv_b = vecf(p + "attn.wqkv.bias", size_t(3 * VD));
      B.wo_w = raw(p + "attn.wo.weight", uint64_t(VD) * uint64_t(VD) * 2);
      B.wo_b = vecf(p + "attn.wo.bias", size_t(VD));
      B.w1 = raw(p + "mlp.w1.weight", uint64_t(2 * c_.v_inter) * uint64_t(VD) * 2);
      B.w2 = raw(p + "mlp.w2.weight", uint64_t(VD) * uint64_t(c_.v_inter) * 2);
      vis_.blocks.push_back(B);
    }
    vis_.norm = vecf("vision.norm.weight", size_t(VD));
    const int r2 = c_.v_down * c_.v_down;
    vis_.al_w1 = raw("aligner.w1.weight", uint64_t(c_.D) * uint64_t(VD * r2) * 2);
    vis_.al_b1 = vecf("aligner.w1.bias", size_t(c_.D));
    vis_.al_w2 = raw("aligner.w2.weight", uint64_t(c_.D) * uint64_t(c_.D) * 2);
    vis_.al_b2 = vecf("aligner.w2.bias", size_t(c_.D));
    vis_.img_start = vecf("image_start", size_t(c_.D));
    vis_.img_end = vecf("image_end", size_t(c_.D));
    vis_.img_nl = vecf("image_newline", size_t(c_.D));
    {
      const int rd = VD / c_.v_heads / 2;  // rotary dims per axis pair: inv_freq over rd
      std::vector<float> f(size_t(rd / 2));
      for (int i = 0; i < rd / 2; ++i) f[size_t(i)] = 1.0f / std::pow(float(c_.v_theta), float(2 * i) / float(rd));
      vis_.inv_freq = rt_.arena->copy(f.data(), f.size() * 4);
    }
    if (VD % 64 || c_.D % 64 || (2 * c_.v_inter) % 64 || c_.v_inter % 32 || VD / c_.v_heads != 64)
      fail("deepseek_v41: unsupported vision geometry");
    vision_ = true;
  }

  // Tokens of an image's run for a patch grid of gh x gw.
  int image_run(int gh, int gw) const {
    const int lh = (gh + c_.v_down - 1) / c_.v_down, lw = (gw + c_.v_down - 1) / c_.v_down;
    return lh * (lw + 1) + 2;
  }

  void gemm_bf16(Enc2& E, const Buf& W, const Buf* B, id<MTLBuffer> X, size_t xo, int xs, id<MTLBuffer> Y, size_t yo, int ys,
                 int M, int N, int K, bool accum) {
    struct {
      int M, N, K, xs, ys, accum, bias;
    } p{M, N, K, xs, ys, accum ? 1 : 0, B ? 1 : 0};
    E.pipe("ds4_gemm_bf16");
    E.set(0, W);
    E.set(1, B ? *B : W);
    E.set(2, X, xo);
    E.set(3, Y, yo);
    E.bytes(4, p);
    E.groups(size_t(N / 64), size_t((M + 63) / 64), 1, 128);
  }
  void norm_eps(Enc2& E, id<MTLBuffer> x, const Buf& w, int rows, int n, float eps) {
    struct {
      int rows, heads, n, stride;
      float eps;
    } p{rows, 1, n, n, eps};
    E.pipe("ds4_rmsnorm");
    E.set(0, x);
    E.set(1, w);
    E.bytes(2, p);
    E.groups(size_t((rows + 7) / 8), 1, 1, 256);
  }

  // The embedding rows of an image's run (image_run(gh, gw) rows of D floats)
  // into dst at dst_off: the ViT over the patches, the 3 x 3 aligner, the
  // learned start / newline / end rows around the aligner's rows.
  void encode_image(const ImageRef& im, id<MTLBuffer> dst, size_t dst_off) {
    const int VD = c_.v_dim, H = c_.v_heads, gh = im.grid_h, gw = im.grid_w, N = gh * gw, D = c_.D;
    const int r = c_.v_down, lh = (gh + r - 1) / r, lw = (gw + r - 1) / r, L = lh * lw, I2 = 2 * c_.v_inter;
    const int K = 3 * c_.v_patch * c_.v_patch;
    if (N <= 0 || N > 16384) fail("deepseek_v41: image patch grid out of range");
    if (N > vcap_ || L > lcap_) {
      vcap_ = std::max(vcap_, N), lcap_ = std::max(lcap_, L);
      auto F = [&](size_t n) { return rt_.buffer(n * 4); };
      const size_t n = size_t(vcap_), l = size_t(lcap_);
      VP_ = F(n * size_t(vis_.kpad)), VX_ = F(n * VD), VXN_ = F(n * VD), VQKV_ = F(n * 3 * VD);
      VQh_ = rt_.buffer(n * VD * 2), VKh_ = rt_.buffer(n * VD * 2), VVh_ = rt_.buffer(n * VD * 2), VOh_ = rt_.buffer(n * VD * 2);
      VO_ = F(n * VD), VH_ = F(n * size_t(I2)), VG_ = F(n * size_t(c_.v_inter));
      VU_ = F(l * size_t(VD) * size_t(r * r)), VA1_ = F(l * size_t(D)), VA2_ = F(l * size_t(D));
      VHW_ = rt_.buffer(n * 8), VSEG_ = rt_.buffer(n * 8);
    }
    {
      float* vp = static_cast<float*>([VP_ contents]);
      for (int i = 0; i < N; ++i) {
        std::memcpy(vp + size_t(i) * size_t(vis_.kpad), im.pixels + size_t(i) * size_t(K), size_t(K) * 4);
        std::memset(vp + size_t(i) * size_t(vis_.kpad) + K, 0, size_t(vis_.kpad - K) * 4);
      }
      int32_t* hw = static_cast<int32_t*>([VHW_ contents]);
      int32_t* sg = static_cast<int32_t*>([VSEG_ contents]);
      for (int i = 0; i < N; ++i) hw[2 * i] = i / gw, hw[2 * i + 1] = i % gw, sg[2 * i] = 0, sg[2 * i + 1] = N;
    }
    Enc2 E(rt_);
    gemm_bf16(E, vis_.pe_w, &vis_.pe_b, VP_, 0, vis_.kpad, VX_, 0, VD, N, VD, vis_.kpad, false);
    int bi = 0;
    for (const VBlk& B : vis_.blocks) {
      if (bi++ % 4 == 3) E.flush();  // command buffers of a few blocks (macOS aborts long ones)
      copy(E, VX_, 0, VXN_, 0, size_t(N) * size_t(VD) * 4);
      norm_eps(E, VXN_, B.n1, N, VD, 1e-6f);
      gemm_bf16(E, B.qkv_w, &B.qkv_b, VXN_, 0, VD, VQKV_, 0, 3 * VD, N, 3 * VD, VD, false);
      {
        struct {
          int N, H, HD;
          float scale;
        } pp{N, H, VD / H, 1.0f / std::sqrt(float(VD / H))};
        E.pipe("vit_qkv_prep");
        E.set(0, VQKV_);
        E.set(1, VHW_);
        E.set(2, vis_.inv_freq);
        E.set(3, VQh_);
        E.set(4, VKh_);
        E.set(5, VVh_);
        E.bytes(6, pp);
        E.threads(size_t(VD / H / 2), size_t(H), size_t(N), 32, 1);
        struct {
          int N, H;
        } ap{N, H};
        E.pipe("vit_attention_64");
        E.set(0, VQh_);
        E.set(1, VKh_);
        E.set(2, VVh_);
        E.set(3, VSEG_);
        E.set(4, VOh_);
        E.bytes(5, ap);
        E.groups(size_t(H), size_t((N + 31) / 32), 1, 128);
        const uint32_t n = uint32_t(N * VD);
        E.pipe("ds4_h2f");
        E.set(0, VOh_);
        E.set(1, VO_);
        E.bytes(2, n);
        E.threads(n, 1, 1, 256);
      }
      gemm_bf16(E, B.wo_w, &B.wo_b, VO_, 0, VD, VX_, 0, VD, N, VD, VD, true);
      copy(E, VX_, 0, VXN_, 0, size_t(N) * size_t(VD) * 4);
      norm_eps(E, VXN_, B.n2, N, VD, 1e-6f);
      gemm_bf16(E, B.w1, nullptr, VXN_, 0, VD, VH_, 0, I2, N, I2, VD, false);
      {
        struct {
          int rows, n, gs, us, os;
          float lim;
        } p{N, c_.v_inter, I2, I2, c_.v_inter, 0.0f};
        E.pipe("ds4_swiglu");
        E.set(0, VH_);
        E.set(1, VH_, size_t(c_.v_inter) * 4);
        E.set(2, VG_);
        E.bytes(3, p);
        E.threads(size_t(c_.v_inter), size_t(N), 1, 256);
      }
      gemm_bf16(E, B.w2, nullptr, VG_, 0, c_.v_inter, VX_, 0, VD, N, VD, c_.v_inter, true);
    }
    norm_eps(E, VX_, vis_.norm, N, VD, 1e-6f);
    {
      struct {
        int nh, nw, C, r, lh, lw;
      } up{gh, gw, VD, r, lh, lw};
      E.pipe("ds4_unfold");
      E.set(0, VX_);
      E.set(1, VU_);
      E.bytes(2, up);
      E.threads(size_t(VD * r * r), size_t(L), 1, 256);
    }
    gemm_bf16(E, vis_.al_w1, &vis_.al_b1, VU_, 0, VD * r * r, VA1_, 0, D, L, D, VD * r * r, false);
    {
      const uint32_t n = uint32_t(L * D);
      E.pipe("ds4_gelu");
      E.set(0, VA1_);
      E.bytes(1, n);
      E.threads(n, 1, 1, 256);
    }
    // the run: [start] + (lw aligner rows + newline) x lh + [end]
    const size_t row = size_t(D) * 4;
    gemm_bf16(E, vis_.al_w2, &vis_.al_b2, VA1_, 0, D, VA2_, 0, D, L, D, D, false);
    copy(E, vis_.img_start.b, vis_.img_start.off, dst, dst_off, row);
    size_t o = dst_off + row;
    for (int i = 0; i < lh; ++i) {
      copy(E, VA2_, size_t(i) * size_t(lw) * row, dst, o, size_t(lw) * row);
      o += size_t(lw) * row;
      copy(E, vis_.img_nl.b, vis_.img_nl.off, dst, o, row);
      o += row;
    }
    copy(E, vis_.img_end.b, vis_.img_end.off, dst, o, row);
    E.end();
    gpu_ms_ += E.gpu_ms;
  }

  // ---------------------------------------------------------------- engram (host)
  // The pass's Engram rows as stored (fp8 codes, then e8m0 scales), read from
  // the SSD into ERAW_[g] ([T][cols][hd + hd / 32] bytes); the GPU dequantizes them.
  void engram_gather(const Seq& s, const int32_t* toks, int T, int row0, const uint8_t* dead) {
    const auto t0 = Clock::now();
    const int nb = eng_.max_n - 1;
    const int ncol = nb * eng_.heads, hd = c_.eng_hd, rb = hd + hd / 32;
    // compressed ids, -1 for tokens inside an image span (an n-gram never reaches across one)
    std::vector<int64_t> cid;
    for (int32_t t : s.hist) cid.push_back(t >= 0 && t < eng_.vocab ? int64_t(eng_.map[size_t(t)]) : -1);
    const int h0 = int(cid.size());
    for (int i = 0; i < T; ++i)
      cid.push_back(!dead[i] && toks[i] >= 0 && toks[i] < eng_.vocab ? int64_t(eng_.map[size_t(toks[i])]) : -1);
    for (int g = 0; g < int(c_.engram_layers.size()); ++g) {
      const LayerW* lw = nullptr;
      for (const LayerW& w : L_)
        if (w.eg == g) lw = &w;
      if (!lw) continue;
      std::vector<int64_t> rows(size_t(T) * size_t(ncol));
      for (int i = 0; i < T; ++i) {
        const int64_t p = int64_t(s.len) + i;
        int64_t tk[8];
        bool blocked = false;
        for (int sh = 0; sh < eng_.max_n; ++sh) {
          const int ci = h0 + i - sh;
          blocked = blocked || p - sh < 0 || ci < 0 || cid[size_t(ci)] < 0;
          tk[sh] = blocked ? int64_t(eng_.pad) : cid[size_t(ci)];
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
      uint8_t* dst = static_cast<uint8_t*>([ERAW_[g] contents]) + size_t(row0) * size_t(ncol) * size_t(rb);
      for (size_t i = 0; i < rows.size(); ++i)
        std::memcpy(dst + i * size_t(rb), vals.data() + size_t(uniq[rows[i]]) * size_t(rb), size_t(rb));
    }
    eng_read_ms_ += ms_since(t0);
  }

  // ---------------------------------------------------------------- experts (host)
  // Slots for this layer's experts `need` (sorted, distinct); missing ones are
  // read from the SSD into staging and copied into their slots by `E`
  // (flushing between staging rounds).
  // ---- the expert cache across restarts: the cached experts, most recently
  // used first, saved every minute while passes run; at load they are read
  // back (sorted by file offset) so a restarted server starts warm.
  // ESPETERO_DS4_WARM=0: start cold.
  std::string warm_path_;
  Clock::time_point warm_saved_ = Clock::now();
  uint64_t warm_clock_ = 0;

  void warm_save() {
    if (warm_path_.empty() || xc_.clock == warm_clock_ || ms_since(warm_saved_) < 60000) return;
    std::vector<std::pair<uint64_t, int32_t>> v;
    for (int sl = 0; sl < xc_.n; ++sl)
      if (xc_.key[size_t(sl)] >= 0) v.push_back({xc_.stamp[size_t(sl)], xc_.key[size_t(sl)]});
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    const std::string tmp = warm_path_ + ".tmp";
    if (FILE* f = std::fopen(tmp.c_str(), "wb")) {
      const int32_t h[4] = {0x34445345 /* "ESD4" */, 1, c_.E, int32_t(v.size())};
      bool ok = std::fwrite(h, sizeof h, 1, f) == 1;
      for (const auto& e : v) ok = ok && std::fwrite(&e.second, 4, 1, f) == 1;
      if (std::fclose(f) == 0 && ok) std::rename(tmp.c_str(), warm_path_.c_str());
    }
    warm_saved_ = Clock::now();
    warm_clock_ = xc_.clock;
  }

  void warm_load() {
    if (warm_path_.empty() || (std::getenv("ESPETERO_DS4_WARM") && std::atoi(std::getenv("ESPETERO_DS4_WARM")) == 0)) return;
    FILE* f = std::fopen(warm_path_.c_str(), "rb");
    if (!f) return;
    int32_t h[4] = {0, 0, 0, 0};
    std::vector<int32_t> keys;
    if (std::fread(h, sizeof h, 1, f) == 1 && h[0] == 0x34445345 && h[1] == 1 && h[2] == c_.E && h[3] > 0) {
      keys.resize(size_t(h[3]));
      if (std::fread(keys.data(), 4, keys.size(), f) != keys.size()) keys.clear();
    }
    std::fclose(f);
    const int nl = c_.L + int(M_.size());
    keys.erase(std::remove_if(keys.begin(), keys.end(), [&](int32_t k) { return k < 0 || k / c_.E >= nl || k % c_.E >= (k / c_.E < c_.L ? c_.E : c_.ds_E); }),
               keys.end());
    if (keys.size() > size_t(xc_.n)) keys.resize(size_t(xc_.n));  // the most recent ones
    if (keys.empty()) return;
    const auto t0 = Clock::now();
    // stamps: the file's order (most recent first gets the newest stamp); reads sorted by layer for locality
    std::vector<std::pair<int32_t, uint64_t>> ks;
    for (size_t i = 0; i < keys.size(); ++i) ks.push_back({keys[i], uint64_t(keys.size() - i)});
    std::sort(ks.begin(), ks.end());
    xc_.clock = keys.size() + 1;
    for (size_t b0 = 0; b0 < ks.size();) {
      // one layer's experts at a time through expert_slots (staging rounds, GPU copies)
      const int l = ks[b0].first / c_.E;
      size_t b1 = b0;
      std::vector<int> need;
      while (b1 < ks.size() && ks[b1].first / c_.E == l) need.push_back(ks[b1++].first % c_.E);
      const LayerW& lw = l < c_.L ? L_[size_t(l)] : M_[size_t(l - c_.L)];
      Enc2 E(rt_);
      const std::vector<int> sl = expert_slots(E, l, lw, need);
      E.end();
      for (size_t i = 0; i < sl.size(); ++i) xc_.stamp[size_t(sl[i])] = ks[b0 + i].second;
      b0 = b1;
    }
    xc_.hits = xc_.misses = 0;
    xc_.read_ms = xc_.read_gb = 0;
    std::fprintf(stderr, "deepseek_v41: expert cache warmed with %zu experts (%.1f GB) in %.1f s\n", ks.size(),
                 double(ks.size()) * double(xc_.slot_bytes) / 1e9, ms_since(t0) / 1e3);
  }

  // ---- expert reads go straight into their cache slots (the pool is host
  // memory locked with mlock: macOS cannot compress or page it, and the SSD
  // writes where the GPU reads). A slot being read is marked loading until done.
  void read_slots(const LayerW& lw, const std::vector<std::pair<int, int>>& es) {  // (expert, slot)
    if (es.empty()) return;
    ExpertCache& x = xc_;
    const auto t0 = Clock::now();
    struct Job {
      int fd;
      uint8_t* dst;
      size_t n;
      uint64_t off;
    };
    std::vector<Job> jobs;
    const size_t piece = size_t(1) << 20;
    for (const auto& [e, sl] : es) {
      const XLoc& loc = lw.x[size_t(e)];
      uint8_t* dst = x.host_ptr(sl);
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
    x.read_gb += double(es.size()) * double(x.slot_bytes) / 1e9;
  }

  // The least recently used slot not used by this layer (stamp `now`) and not being read.
  // (Offline simulations on a trace of real traffic, ESPETERO_DS4_TRACE + tools/dsv4_xsim.py: ARC, SLRU and
  // decayed-frequency policies come within a few percent of LRU, and frequency with a long memory is far worse.)
  // In a streamed pass, first the experts this pass streamed in for layers it is done with: a pass cycles
  // through more experts than the cache holds, and LRU would evict the coming layers' cached experts (and
  // what decoding uses) to make room for ones the pass needs no more.
  int evict_slot(uint64_t now) {
    ExpertCache& x = xc_;
    int best = -1, done = -1;
    double bs = 1e300, ds = 1e300;
    for (int sl = 0; sl < x.n; ++sl) {
      if (x.key[size_t(sl)] < 0) return sl;
      if (x.stamp[size_t(sl)] == now || x.loading[size_t(sl)].load() == 1) continue;
      const int kl = x.key[size_t(sl)] / c_.E;
      if (kl >= prot_lo_ && kl <= prot_hi_) continue;  // a streamed pass's window
      const double v = double(x.stamp[size_t(sl)]);
      if (v < bs) bs = v, best = sl;
      if (prot_lo_ >= 0 && kl < prot_lo_ && x.spass[size_t(sl)] == x.pass && v < ds) ds = v, done = sl;
    }
    if (done >= 0) best = done;
    if (best < 0) fail("deepseek_v41: expert cache too small for one layer");
    x.where[size_t(x.key[size_t(best)])] = -1;
    x.key[size_t(best)] = -1;
    return best;
  }

  // ---- streaming (long passes): a layer over many rows uses nearly every expert, so its experts need no
  // routing to be known. The main thread gives the coming layers' missing experts their slots (it alone
  // touches the cache's tables) and a reader thread reads them in layer order, up to kStreamAhead layers
  // ahead of the GPU; the slots of the layer at hand and of the streamed ones are not evicted
  // (prot_lo_..prot_hi_), and the MoE takes the experts as they arrive. The SSD reads while the GPU runs
  // attention and the MoE of the layers before. ESPETERO_DS4_STREAM=0: off.
  static constexpr int kStreamAhead = 2;     // layers read ahead of the one at hand
  static constexpr int kStreamRows = 256;    // rows from which a layer counts as using every expert
  static constexpr size_t kStreamBatch = 8;  // experts per batch of reads (their flags drop together)
  struct SJob {
    int fd;
    uint8_t* dst;
    uint64_t w, s;
    std::atomic<int>* flag;
  };
  std::thread st_th_;
  std::mutex st_mu_;
  std::condition_variable st_cv_;
  std::deque<SJob> st_q_;
  bool st_stop_ = false;
  std::atomic<uint64_t> st_bytes_{0}, st_us_{0};
  int prot_lo_ = -1, prot_hi_ = -2;  // expert-cache layers whose slots stay
  int st_next_ = 0;                  // the next layer of the pass to stream
  uint64_t st_issued_ = 0;

  void st_loop() {
    for (;;) {
      std::vector<SJob> b;
      {
        std::unique_lock<std::mutex> lk(st_mu_);
        st_cv_.wait(lk, [&] { return st_stop_ || !st_q_.empty(); });
        if (st_q_.empty()) return;
        while (!st_q_.empty() && b.size() < kStreamBatch) b.push_back(st_q_.front()), st_q_.pop_front();
      }
      const auto t0 = Clock::now();
      const size_t wb = xc_.wbytes, sb = xc_.sbytes, piece = size_t(1) << 20, np = (wb + piece - 1) / piece + 1;
      std::vector<std::atomic<char>> ok(b.size());
      for (auto& o : ok) o.store(1);
      parallel_for(b.size() * np, [&](size_t i) {
        const SJob& j = b[i / np];
        const size_t q = i % np;
        const bool r = q + 1 < np ? pread_all(j.fd, j.dst + q * piece, std::min(piece, wb - q * piece), size_t(j.w + q * piece))
                                  : pread_all(j.fd, j.dst + wb, sb, size_t(j.s));
        if (!r) ok[i / np].store(0);
      });
      for (size_t i = 0; i < b.size(); ++i) b[i].flag->store(ok[i].load() ? 0 : 2);
      st_bytes_ += uint64_t(b.size()) * uint64_t(xc_.slot_bytes);
      st_us_ += uint64_t(ms_since(t0) * 1000.0);
    }
  }

  // Streams the dense layers of the pass up to layer `upto` (dense[l]: the layer uses every expert).
  void stream_to(int upto, const std::vector<char>& dense, const std::vector<LayerW>& LW) {
    ExpertCache& x = xc_;
    const uint64_t now = x.clock;
    std::vector<SJob> jobs;
    for (; st_next_ <= upto && st_next_ < int(LW.size()); ++st_next_) {
      const int l = st_next_;
      prot_hi_ = l;
      if (!dense[size_t(l)]) continue;
      const LayerW& lw = LW[size_t(l)];
      for (int e = 0; e < lw.nE; ++e) {
        const int k = l * c_.E + e;
        if (x.where[size_t(k)] >= 0) continue;
        const int sl = evict_slot(now);
        if (x.pref[size_t(sl)]) ++pf_wasted_;
        x.key[size_t(sl)] = k;
        x.where[size_t(k)] = sl;
        x.stamp[size_t(sl)] = now;
        x.pref[size_t(sl)] = 1;
        x.spass[size_t(sl)] = x.pass;
        x.loading[size_t(sl)].store(1);
        ++st_issued_;
        const XLoc& loc = lw.x[size_t(e)];
        const int fd = ck_.fds_nc[size_t(loc.file)] >= 0 ? ck_.fds_nc[size_t(loc.file)] : ck_.fds[size_t(loc.file)];
        jobs.push_back({fd, x.host_ptr(sl), loc.w, loc.s, &x.loading[size_t(sl)]});
      }
    }
    if (jobs.empty()) return;
    if (!st_th_.joinable()) st_th_ = std::thread([this] { st_loop(); });
    {
      std::lock_guard<std::mutex> lk(st_mu_);
      for (const SJob& j : jobs) st_q_.push_back(j);
    }
    st_cv_.notify_one();
  }

  // Waits until a slot's read (prefetch or stream) is over; false: it failed (the slot is free to read again).
  bool wait_slot(int sl) {
    if (xc_.loading[size_t(sl)].load() == 1) {
      const auto tw = Clock::now();
      while (xc_.loading[size_t(sl)].load() == 1) std::this_thread::yield();
      pf_wait_ms_ += ms_since(tw);
    }
    if (xc_.loading[size_t(sl)].load() == 2) {
      xc_.loading[size_t(sl)].store(0);
      return false;
    }
    return true;
  }

  // ---- prefetch: experts the next layer will probably need, read in the
  // background into their slots while the GPU works (long passes: every expert
  // of the next layer; decode: the next layer's router on this layer's input,
  // off by default). At most kPfMax reads in flight.
  static constexpr int kPfMax = 64;
  std::atomic<int> pf_inflight_{0};
  uint64_t pf_issued_ = 0, pf_used_ = 0, pf_wasted_ = 0;
  double pf_wait_ms_ = 0;
  bool pf_on_ = !(std::getenv("ESPETERO_DS4_PREFETCH") && std::atoi(std::getenv("ESPETERO_DS4_PREFETCH")) == 0);

  void prefetch(int l, const LayerW& lw, const std::vector<int>& experts) {
    ExpertCache& x = xc_;
    if (x.n < 2 * c_.E + kPfMax) return;  // a small cache needs every slot for the layer at hand
    const uint64_t now = x.clock;
    for (int e : experts) {
      if (pf_inflight_.load() >= kPfMax) return;
      const int k = l * c_.E + e;
      if (x.where[size_t(k)] >= 0) continue;
      const int sl = evict_slot(now);
      x.key[size_t(sl)] = k;
      x.where[size_t(k)] = sl;
      x.stamp[size_t(sl)] = now;  // kept from eviction until the next layer runs
      x.pref[size_t(sl)] = 1;
      x.loading[size_t(sl)].store(1);
      ++pf_issued_;
      ++pf_inflight_;
      const XLoc loc = lw.x[size_t(e)];
      const int fd = ck_.fds_nc[size_t(loc.file)] >= 0 ? ck_.fds_nc[size_t(loc.file)] : ck_.fds[size_t(loc.file)];
      uint8_t* dst = x.host_ptr(sl);
      const size_t wb = x.wbytes, sb = x.sbytes;
      std::atomic<int>* flag = &x.loading[size_t(sl)];
      std::atomic<int>* inflight = &pf_inflight_;
      dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        const size_t piece = size_t(1) << 20, np = (wb + piece - 1) / piece;
        std::atomic<bool>* ok = new std::atomic<bool>(true);
        dispatch_apply(np + 1, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
          bool r;
          if (i < np) r = pread_all(fd, dst + i * piece, std::min(piece, wb - i * piece), size_t(loc.w + i * piece));
          else r = pread_all(fd, dst + wb, sb, size_t(loc.s));
          if (!r) ok->store(false);
        });
        flag->store(ok->load() ? 0 : 2);
        delete ok;
        inflight->fetch_sub(1);
      });
    }
  }

  // Slots for this layer's experts `need` (sorted, distinct): hits, then the
  // misses' slots (LRU), read here unless `pending` takes them (indices into
  // need: assigned but not read). Prefetched slots still loading are waited for.
  std::vector<int> expert_slots(Enc2& E, int l, const LayerW& lw, const std::vector<int>& need, std::vector<size_t>* pending = nullptr,
                                const std::vector<int>* rows = nullptr,  // rows: how many rows use each needed expert (for the trace)
                                std::vector<size_t>* waiting = nullptr) {  // waiting: experts still being read for us, not waited for here
    (void)E;
    ExpertCache& x = xc_;
    const uint64_t now = ++x.clock;
    static FILE* trace = std::getenv("ESPETERO_DS4_TRACE") ? std::fopen(std::getenv("ESPETERO_DS4_TRACE"), "w") : nullptr;
    if (trace) {  // pass layer expert rows, for offline policy studies
      for (size_t i = 0; i < need.size(); ++i)
        std::fprintf(trace, "%llu %d %d %d\n", (unsigned long long)x.pass, l, need[i], rows ? (*rows)[i] : 1);
      std::fflush(trace);
    }
    std::vector<int> slot(need.size(), -1);
    std::vector<size_t> miss;
    for (size_t i = 0; i < need.size(); ++i) {
      const int k = l * c_.E + need[i];
      const int s = x.where[size_t(k)];
      if (s >= 0) {
        if (waiting && x.loading[size_t(s)].load() == 1) {  // the pass's streamer still reading it: the MoE rounds wait
          if (x.pref[size_t(s)]) x.pref[size_t(s)] = 0, ++pf_used_;
          slot[i] = s;
          x.stamp[size_t(s)] = now;
          ++x.hits;
          waiting->push_back(i);
          continue;
        }
        if (x.loading[size_t(s)].load() == 1) {  // a prefetch still reading it
          const auto tw = Clock::now();
          while (x.loading[size_t(s)].load() == 1) std::this_thread::yield();
          pf_wait_ms_ += ms_since(tw);
        }
        if (x.loading[size_t(s)].load() == 2) {  // its read failed: read again below
          x.loading[size_t(s)].store(0);
          slot[i] = s;
          x.stamp[size_t(s)] = now;
          miss.push_back(i);
          ++x.misses;
          continue;
        }
        if (x.pref[size_t(s)]) x.pref[size_t(s)] = 0, ++pf_used_;
        slot[i] = s;
        x.stamp[size_t(s)] = now;
        ++x.hits;
      } else {
        miss.push_back(i);
        ++x.misses;
      }
    }
    for (size_t i : miss) {
      if (slot[i] >= 0) continue;  // a failed prefetch keeps its slot
      const int sl = evict_slot(now);
      const int k = l * c_.E + need[i];
      if (x.pref[size_t(sl)]) x.pref[size_t(sl)] = 0, ++pf_wasted_;
      x.key[size_t(sl)] = k;
      x.where[size_t(k)] = sl;
      x.stamp[size_t(sl)] = now;
      slot[i] = sl;
    }
    if (pending) {
      *pending = miss;
      return slot;
    }
    std::vector<std::pair<int, int>> es;
    for (size_t i : miss) es.push_back({need[i], slot[i]});
    read_slots(lw, es);
    return slot;
  }

  // ---------------------------------------------------------------- forward
  // A pass: rows of one or more sequences (segments), layer by layer: every
  // layer runs over all rows (GPU blocks of P_ rows, which may mix segments)
  // before the next one, so each layer's experts come from the SSD once per
  // pass and the dense projections serve every sequence at once. Attention
  // runs per segment part of a block.
  struct Seg {
    Seq* s = nullptr;
    const int32_t* toks = nullptr;
    int n = 0;
    float* logits = nullptr;  // logits of the last nlog rows ([nlog][V]), or null
    int nlog = 1;
    bool verify = false;      // a draft verification: keeps what a rollback needs
    std::vector<ImageRef> imgs;  // image runs (row: from the segment's first row)
    int r0 = 0, p0 = 0;       // first row in the pass, first position
  };
  struct Part {
    int seg = 0, ro = 0, m = 0, p0 = 0;  // block rows [ro, ro + m) of segment seg, at positions p0 ...
  };
  struct Blk {
    int r0 = 0, M = 0;
    std::vector<Part> parts;
  };

  // draft: the DSpark layers over draft blocks (no slot state changes; the
  // drafts land in DTOK_).
  void forward(std::vector<Seg>& segs, bool draft = false) {
    const auto tf0 = Clock::now();
    ++xc_.pass;
    struct Timer {
      double& acc;
      Clock::time_point t0;
      ~Timer() { acc += ms_since(t0); }
    } timer{fwd_ms_, tf0};
    const Cfg& c = c_;
    const int D = c.D;
    const std::vector<LayerW>& LW = draft ? M_ : L_;
    int T = 0;
    for (Seg& g : segs) {
      g.r0 = T, g.p0 = g.s->len;
      if (g.p0 + g.n > max_ctx_) fail("deepseek_v41: context full (" + std::to_string(max_ctx_) + " tokens)");
      for (int i = 0; i < g.n; ++i) {
        if (g.toks[i] < 0 || g.toks[i] >= c.V) fail("deepseek_v41: token id out of range");
        static_cast<int32_t*>([ids_ contents])[T + i] = g.toks[i];
        static_cast<int32_t*>([pos_ contents])[T + i] = g.p0 + i;
      }
      T += g.n;
    }
    int nlog = 0;
    for (const Seg& g : segs) nlog += g.logits ? g.nlog : 0;
    if (T > T_ || nlog > max_reads_) fail("deepseek_v41: pass too large");
    // images: their runs' embedding rows, and the flags of image-span rows (VL router bias, no n-grams)
    uint8_t* img = static_cast<uint8_t*>([IMG_ contents]);
    std::memset(img, 0, size_t(T));
    struct Run {
      int row, n;
      size_t off;
    };
    std::vector<Run> runs;
    {
      size_t off = 0;
      for (const Seg& g : segs)
        for (const ImageRef& im : g.imgs) {
          if (!vision_) fail("deepseek_v41: images need the vision tower");
          const int n = image_run(im.grid_h, im.grid_w);
          if (im.row < 0 || im.row + n > g.n) fail("deepseek_v41: image run outside its segment");
          for (int i = 0; i < n; ++i)
            if (g.toks[im.row + i] != c.image_token) fail("deepseek_v41: image run does not match its placeholder tokens");
          if (!IMGS_ || off + size_t(n) * size_t(D) * 4 > imgs_bytes_) {
            if (off > 0) fail("deepseek_v41: too many image rows in one pass");
            imgs_bytes_ = std::max(imgs_bytes_, size_t(n) * size_t(D) * 4 * 2);
            IMGS_ = rt_.buffer(imgs_bytes_);
          }
          encode_image(im, IMGS_, off);
          runs.push_back({g.r0 + im.row, n, off});
          std::memset(img + g.r0 + im.row, 1, size_t(n));
          off += size_t(n) * size_t(D) * 4;
        }
    }
    if (!no_engram_ && !draft)
      for (Seg& g : segs) engram_gather(*g.s, g.toks, g.n, g.r0, img + g.r0);
    std::vector<Blk> blks;
    for (int r0 = 0; r0 < T; r0 += P_) {
      Blk bk;
      bk.r0 = r0, bk.M = std::min(P_, T - r0);
      for (size_t i = 0; i < segs.size(); ++i) {
        const int a0 = std::max(r0, segs[i].r0), a1 = std::min(r0 + bk.M, segs[i].r0 + segs[i].n);
        if (a0 < a1) bk.parts.push_back(Part{int(i), a0 - r0, a1 - a0, segs[i].p0 + (a0 - segs[i].r0)});
      }
      blks.push_back(std::move(bk));
    }
    // ---- Causal encoder-decoder (DeepSeek-V4.1 report, 2.2): the layers from the last KV source on
    // (the decoder) take their compressed entries and index keys from that source's input, the
    // encoder's output; each keeps only a window ring of its own. So a decoder layer runs over a
    // segment's last rows only: those whose outputs the next layer reads (rows_out) and, before
    // them, the WIN - 1 rows they attend to, for their window latents only (rows_kv); the source
    // layer also compresses every row. Exact: each row whose output is used sees what a full pass
    // shows it. Prefill of N tokens: 20 N + ~24 K decoder rows instead of 40 N. ESPETERO_DS4_CED=0: off.
    static const bool ced_on = !(std::getenv("ESPETERO_DS4_CED") && std::atoi(std::getenv("ESPETERO_DS4_CED")) == 0);
    int dec0 = -1;
    if (ced_on && !draft) {
      for (int l = 0; l < int(LW.size()); ++l)
        if (LW[size_t(l)].kv_src) dec0 = l;
      for (int l = std::max(0, dec0); l < int(LW.size()); ++l)
        if (LW[size_t(l)].eg >= 0) dec0 = -1;  // Engram in the decoder would need every row
    }
    std::vector<std::vector<int>> rows_out, rows_kv;  // [layer - dec0][segment]: trailing rows
    if (dec0 >= 0) {
      rows_out.assign(LW.size() - size_t(dec0), std::vector<int>(segs.size()));
      rows_kv = rows_out;
      for (size_t i = 0; i < segs.size(); ++i) {
        int o = segs[i].logits ? std::min(segs[i].n, segs[i].nlog) : 0;
        for (int l = int(LW.size()) - 1; l >= dec0; --l) {
          const int k = std::min(segs[i].n, std::max(c.WIN, o + c.WIN - 1));  // the ring keeps the last WIN
          rows_out[size_t(l - dec0)][i] = o, rows_kv[size_t(l - dec0)][i] = k;
          o = k;
        }
      }
    }
    auto full_rows = [&](int l) {
      if (dec0 < 0 || l < dec0) return T;
      int n = 0;
      for (int o : rows_out[size_t(l - dec0)]) n += o;
      return n;
    };
    // ---- the layers whose experts stream ahead of the GPU (stream_to)
    static const bool stream_on = !(std::getenv("ESPETERO_DS4_STREAM") && std::atoi(std::getenv("ESPETERO_DS4_STREAM")) == 0);
    std::vector<char> dense(LW.size(), 0);
    bool streaming = false;
    st_next_ = 0, prot_lo_ = -1, prot_hi_ = -2;
    if (stream_on && !draft && xc_.n >= (kStreamAhead + 3) * c.E + 64)
      for (int l = 0; l < int(LW.size()); ++l)
        if (full_rows(l) >= kStreamRows) dense[size_t(l)] = 1, streaming = true;
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
    for (const Run& rn : runs) {
      const int32_t p[4] = {D, rn.n, rn.row, 0};
      E.pipe("ds4_embed_rows");
      E.set(0, IMGS_, rn.off);
      E.set(1, H_);
      E.raw(2, p, 16);
      E.threads(size_t(D), size_t(rn.n), 1, 256);
    }
    id<MTLBuffer> prev = mix0_;
    int cur = 0;
    bool pre_done = false;  // the fused kernel of the previous layer's end wrote this layer's mixA_ and X_
    for (int l = 0; l < int(LW.size()); ++l) {
      const LayerW& w = LW[size_t(l)];
      const int K = w.topk, key = draft ? c.L + l : l;  // the expert cache's layer
      if (streaming) prot_lo_ = std::max(0, l - 1), stream_to(l + kStreamAhead, dense, LW);  // l - 1: its MoE may still run
      std::vector<Blk> fsub, ksub;  // a decoder layer's blocks: rows with outputs, rows with window latents only
      if (dec0 >= 0 && l >= dec0) {
        std::vector<int> flo(segs.size()), fhi(segs.size()), klo(segs.size());
        for (size_t i = 0; i < segs.size(); ++i) {
          const int n = segs[i].n;
          fhi[i] = n, flo[i] = n - rows_out[size_t(l - dec0)][i];
          klo[i] = l == dec0 ? 0 : n - rows_kv[size_t(l - dec0)][i];
        }
        fsub = sub_blocks(blks, segs, flo, fhi);
        ksub = sub_blocks(blks, segs, klo, flo);
        if (int(fsub.size()) > nsub_cap_) fail("deepseek_v41: too many blocks in a decoder layer");
      }
      std::vector<Blk>& fb = dec0 >= 0 && l >= dec0 ? fsub : blks;
      const int Tf = full_rows(l);
      const bool more = pf_on_ && !draft && l + 1 < int(LW.size()) && !dense[size_t(std::min(l + 1, int(LW.size()) - 1))];
      // Decode-time prediction (the next layer's router on this layer's input) found ~1 in 3 of the
      // misses and its reads did not finish in time: off unless ESPETERO_DS4_PREDICT=1.
      static const bool pred_on = std::getenv("ESPETERO_DS4_PREDICT") && std::atoi(std::getenv("ESPETERO_DS4_PREDICT")) != 0;
      const bool predict = more && pred_on && T <= kPredRows;
      // the draft head's view of this layer's input: the mean of the hc copies
      if (!draft && dspark_)
        for (size_t k = 0; k < c.ds_targets.size(); ++k)
          if (c.ds_targets[k] == l)
            for (const Seg& g : segs) {
              const int k0 = std::max(0, g.n - 128);
              struct {
                int rows, D, nt, k;
              } mp{g.n - k0, D, int(c.ds_targets.size()), int(k)};
              E.pipe("ds4_hc_mean");
              E.set(0, H_, hrow(g.r0 + k0));
              E.set(1, pos_, size_t(g.r0 + k0) * 4);
              E.set(2, g.s->mh);
              E.bytes(3, mp);
              E.threads(size_t(D), size_t(g.n - k0), 1, 256);
            }
      // ---- phase 1: Engram, attention, the MoE input and its routing, per block
      for (Blk& kb : ksub) {
        hc_pre(E, prev, w.attn_norm, X_, 0, kb.r0, kb.M);
        attention_kv(E, w, l, kb, segs);
      }
      for (size_t ci = 0; ci < fb.size(); ++ci) {
        Blk& bk = fb[ci];
        const int r0 = bk.r0, M = bk.M;
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
          E.set(4, IMG_, size_t(r0));
          E.groups(size_t(M), 1, 1, 256);
        }
        // The block's input mix and norm: done by the fused kernel at the end of the previous layer (pre_done),
        // unless an Engram changed H first.
        if (!hc_fuse()) {
          hc_mix(E, w.hca_fn, w.hca_base, w.hca_scale, mixA_, r0, M);
          hc_pre(E, prev, w.attn_norm, X_, 0, r0, M);
        } else if (!pre_done || w.eg >= 0) {
          hc_fused(E, r0, M, nil, 0, nil, &w.hca_fn, &w.hca_base, &w.hca_scale, mixA_, prev, &w.attn_norm, X_, 0);
        }
        attention_block(E, w, l, bk, segs, draft);
        if (!dump_dir_.empty()) dump_rows(E, "a" + std::to_string(l) + "_" + std::to_string(ci), Y_, size_t(M) * size_t(D));
        if (!hc_fuse()) {
          hc_post(E, mixA_, r0, M);
          hc_mix(E, w.hcf_fn, w.hcf_base, w.hcf_scale, mixF_[cur], r0, M);
          hc_pre(E, mixA_, w.ffn_norm, XF_, size_t(r0) * size_t(D) * 4, r0, M);
        } else {
          hc_fused(E, r0, M, nil, 0, mixA_, &w.hcf_fn, &w.hcf_base, &w.hcf_scale, mixF_[cur], mixA_, &w.ffn_norm, XF_,
                   size_t(r0) * size_t(D) * 4);
        }
        moe_route(E, w, r0, M);
        if (predict) moe_route_to(E, LW[size_t(l + 1)], r0, M, PSEL_, PRW_);  // the next layer's router on this input
      }
      E.flush();  // routing on the host
      const int32_t* sel = static_cast<const int32_t*>([SEL_ contents]);
      static const char* dumpx = std::getenv("ESPETERO_DS4_DUMPX");  // MoE inputs and routes, for routing predictors
      if (dumpx && !draft) {
        const std::string b = std::string(dumpx) + "/l" + std::to_string(l);
        if (FILE* f = std::fopen((b + ".x16").c_str(), "wb")) {  // XF_ rows as fp16
          std::vector<uint16_t> h(size_t(T) * size_t(D));
          const float* xf = static_cast<const float*>([XF_ contents]);
          for (size_t i = 0; i < h.size(); ++i) h[i] = std::bit_cast<uint16_t>(_Float16(xf[i]));
          std::fwrite(h.data(), 2, h.size(), f);
          std::fclose(f);
        }
        if (FILE* f = std::fopen((b + ".sel").c_str(), "wb")) {
          std::fwrite(sel, 4, size_t(T) * size_t(K), f);
          std::fclose(f);
        }
      }
      std::vector<int> used(size_t(w.nE), 0);
      for (const Blk& bk : fb)
        for (int i = bk.r0 * K; i < (bk.r0 + bk.M) * K; ++i) {
          if (sel[i] < 0 || sel[i] >= w.nE) fail("deepseek_v41: bad routing");
          ++used[size_t(sel[i])];
        }
      std::vector<int> need, nrows;
      for (int e = 0; e < w.nE; ++e)
        if (used[size_t(e)]) need.push_back(e), nrows.push_back(used[size_t(e)]);
      if (!dump_dir_.empty())
        if (FILE* f = std::fopen((dump_dir_ + "/p" + std::to_string(dump_pass_) + "_route" + std::to_string(l) + ".i32").c_str(), "wb")) {
          std::fwrite(sel, 4, size_t(T * K), f);
          std::fclose(f);
        }
      // long passes overlap the misses' reads with the experts' GPU work (ESPETERO_DS4_PIPE=0: off)
      static const bool pipe_on = !(std::getenv("ESPETERO_DS4_PIPE") && std::atoi(std::getenv("ESPETERO_DS4_PIPE")) == 0);
      const bool pipe = pipe_on && !draft && Tf >= 256;
      std::vector<size_t> pend, waiting;
      const std::vector<int> slots = expert_slots(E, key, w, need, pipe ? &pend : nullptr, &nrows, pipe && streaming ? &waiting : nullptr);
      if (more) {  // the next layer's experts, read while the GPU works on this one
        const LayerW& nw = LW[size_t(l + 1)];
        std::vector<int> nx;
        if (predict) {
          std::vector<char> pu(size_t(nw.nE), 0);
          const int32_t* ps = static_cast<const int32_t*>([PSEL_ contents]);
          for (int i = 0; i < T * nw.topk; ++i)
            if (ps[i] >= 0 && ps[i] < nw.nE) pu[size_t(ps[i])] = 1;
          for (int e = 0; e < nw.nE; ++e)
            if (pu[size_t(e)]) nx.push_back(e);
        } else if (full_rows(l + 1) >= 64) {
          for (int e = 0; e < nw.nE; ++e) nx.push_back(e);
        }
        prefetch(key + 1, nw, nx);
      }
      std::vector<int> slot_of(size_t(w.nE), -1);
      for (size_t i = 0; i < need.size(); ++i) slot_of[size_t(need[i])] = slots[i];
      if (!pipe) {
        // ---- phase 3: shared and routed experts, per block
        for (size_t ci = 0; ci < fb.size(); ++ci) {
          const int r0 = fb[ci].r0, M = fb[ci].M;
          moe_experts(E, w, int(ci), r0, M, slot_of);
          if (!dump_dir_.empty()) dump_rows(E, "f" + std::to_string(l) + "_" + std::to_string(ci), Y_, size_t(M) * size_t(D));
          hc_end(E, l, r0, M, nil, 0, mixF_[cur], LW, pre_done);
        }
      } else {
        // ---- phase 3, pipelined (long passes): the GPU computes the experts in
        // memory while the host reads the missing ones, a round of staging at a
        // time (two halves, tables in two copies); the rounds' output adds up in YT_.
        const size_t yrow = size_t(D) * 4;
        std::vector<char> in(size_t(w.nE), 0);
        {
          std::vector<char> miss_e(size_t(w.nE), 0);
          for (size_t i : pend) miss_e[size_t(need[i])] = 1;
          for (size_t i : waiting) miss_e[size_t(need[i])] = 1;
          for (int e : need)
            if (!miss_e[size_t(e)]) in[size_t(e)] = 1;
        }
        for (size_t ci = 0; ci < fb.size(); ++ci)
          moe_experts(E, w, int(ci), fb[ci].r0, fb[ci].M, slot_of, &in, true, YT_, size_t(fb[ci].r0) * yrow, 0);
        id<MTLCommandBuffer> used[2] = {E.submit_chained(), nil};
        const size_t RB = size_t(std::max(1, xc_.stage_n));
        // rounds: the streamed experts as they arrive (in the reader's order), then the misses, read here
        std::vector<std::pair<size_t, bool>> todo;
        for (size_t i : waiting) todo.push_back({i, true});
        for (size_t i : pend) todo.push_back({i, false});
        int k = 1;
        for (size_t b0 = 0; b0 < todo.size(); b0 += RB, ++k) {
          const int par = k % 2;
          if (used[par]) E.wait_cb(used[par]);  // its host tables are free again
          const size_t b1 = std::min(todo.size(), b0 + RB);
          std::vector<std::pair<int, int>> es;
          std::fill(in.begin(), in.end(), 0);
          for (size_t j = b0; j < b1; ++j) {
            const size_t i = todo[j].first;
            if (!todo[j].second || !wait_slot(slots[i])) es.push_back({need[i], slots[i]});
            in[size_t(need[i])] = 1;
          }
          read_slots(w, es);  // straight into the slots, while the GPU runs the earlier rounds
          for (size_t ci = 0; ci < fb.size(); ++ci)
            moe_experts(E, w, int(ci), fb[ci].r0, fb[ci].M, slot_of, &in, false, YT_, size_t(fb[ci].r0) * yrow, par);
          used[par] = E.submit_chained();
        }
        for (size_t ci = 0; ci < fb.size(); ++ci) hc_end(E, l, fb[ci].r0, fb[ci].M, YT_, size_t(fb[ci].r0) * yrow, mixF_[cur], LW, pre_done);
      }
      prev = mixF_[cur];
      cur ^= 1;
      if (!dump_dir_.empty()) dump_rows(E, "h" + std::to_string(l), H_, size_t(T) * 4 * size_t(D));
    }
    prot_lo_ = -1, prot_hi_ = -2;
    // ---- logits of the segments' last rows
    int R = 0;
    for (const Seg& g : segs)
      if (g.logits && head_.b) {
        hc_pre(E, prev, draft ? ds_norm_ : norm_, XN_, size_t(R) * size_t(D) * 4, g.r0 + g.n - g.nlog, g.nlog);
        R += g.nlog;
      }
    if (R && head_.b) gemv(E, 2, Buf(head_), Buf{}, XN_, 0, D, LOG_, 0, c.V, c.V, D, R);
    if (draft) {  // the Markov head picks the drafts one after another, on the GPU
      int r = 0;
      for (size_t i = 0; i < segs.size(); ++i) {
        for (int j = 0; j < segs[i].nlog; ++j) {
          const int32_t mp[3] = {c.V, c.ds_markov, j};
          E.pipe("ds4_markov_bias");
          E.set(0, ds_mk_embed_.w);
          E.set(1, ds_mk_head_.w);
          E.set(2, DTOK_, i * 8 * 4);
          E.set(3, MKB_);
          E.bytes(4, mp);
          E.threads(size_t(c.V), 1, 1, 256);
          E.pipe("ds4_markov_argmax");
          E.set(0, LOG_, size_t(r) * size_t(c.V) * 4);
          E.set(1, MKB_);
          E.set(2, DTOK_, i * 8 * 4);
          E.bytes(3, mp);
          E.groups(1, 1, 1, 1024);
        }
        if (ds_conf_.b) {  // each draft's acceptance logit (CONF_ rows like DTOK_'s)
          const int32_t cp[2] = {D, c.ds_markov};
          E.pipe("ds4_conf");
          E.set(0, H_, hrow(segs[i].r0 + segs[i].n - segs[i].nlog));
          E.set(1, prev, size_t(segs[i].r0 + segs[i].n - segs[i].nlog) * 24 * 4);
          E.set(2, ds_conf_);
          E.set(3, ds_mk_embed_.w);
          E.set(4, DTOK_, i * 8 * 4);
          E.set(5, CONF_, i * 8 * 4);
          E.bytes(6, cp);
          E.groups(size_t(segs[i].nlog), 1, 1, 256);
        }
        r += segs[i].nlog;
      }
      E.end();
      gpu_ms_ += E.gpu_ms;
      flushes_ += uint64_t(E.flushes);
      return;
    }
    E.end();
    gpu_ms_ += E.gpu_ms;
    wait_ms_ += E.wait_ms, wake_ms_ += E.wake_ms;
    flushes_ += uint64_t(E.flushes);
    R = 0;
    for (Seg& g : segs) {
      for (int i = 0; i < g.n; ++i) g.s->hist.push_back(img[g.r0 + i] ? -1 : g.toks[i]);  // -1: no n-gram spans an image
      const size_t keep = size_t(std::max(0, eng_.max_n - 1));
      if (g.s->hist.size() > keep) g.s->hist.erase(g.s->hist.begin(), g.s->hist.end() - long(keep));
      g.s->len = g.p0 + g.n;
      if (g.logits) {
        const size_t n = size_t(g.nlog) * size_t(c.V);
        if (head_.b) std::memcpy(g.logits, static_cast<const float*>([LOG_ contents]) + size_t(R) * size_t(c.V), n * 4);
        else std::memset(g.logits, 0, n * 4);
        R += g.nlog;
      }
    }
    ++dump_pass_;
    warm_save();
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

  // Attention of a block: X_ in, Y_ out (scratch rows 0..M); the projections
  // over all rows, the sequence mixing per segment part.
  void attention_block(Enc2& E, const LayerW& w, int l, Blk& bk, std::vector<Seg>& segs, bool draft) {
    const Cfg& c = c_;
    const int D = c.D, HD = c.HD, QR = c.QR, HQ = c.H * c.HD, M = bk.M;
    const Buf& invf = w.ratio ? invf_yarn_ : invf_plain_;
    const size_t po = size_t(bk.r0) * 4;  // pos_ rows of this block
    quant_copy(E, X_, 0, D, XQ_, 0, D, M, D);
    gemv8(E, w.wq_a, XQ_, D, QA_, QR, M);
    vec_post(E, QA_, 0, &w.q_norm, pos_, po, nullptr, M, 1, QR, QR, false, false, true);
    gemv8(E, w.wq_b, QA_, QR, Q_, HQ, M);
    rope(E, Q_, 0, pos_, po, invf, M, c.H, HD, HQ, false);
    gemv8(E, w.wkv, XQ_, D, KVp_, HD, M);
    vec_post(E, KVp_, 0, &w.kv_norm, pos_, po, &invf, M, 1, HD, HD, true, false, true);
    for (Part& pt : bk.parts) attention_seq(E, w, l, segs[size_t(pt.seg)], bk.r0, pt, draft);
    vec_post(E, Q_, 0, nullptr, pos_, po, &invf, M, c.H, HD, HQ, true, true, true);
    gemv(E, 0, w.wo_a.w, w.wo_a.s, Q_, 0, HQ, OA_, 0, c.OG * c.OR, w.wo_a.N, w.wo_a.K, M, c.OR, w.wo_a.K);
    quant(E, OA_, 0, M, c.OG * c.OR, c.OG * c.OR, 0);
    gemv8(E, w.wo_b, OA_, c.OG * c.OR, Y_, D, M);
  }

  // Rows of a decoder layer whose outputs nothing reads (CED, see forward): only their window
  // latents, which later rows attend to, and at the KV source the compressed entries and index keys.
  void attention_kv(Enc2& E, const LayerW& w, int l, Blk& bk, std::vector<Seg>& segs) {
    const Cfg& c = c_;
    const int D = c.D, HD = c.HD, M = bk.M;
    const Buf& invf = w.ratio ? invf_yarn_ : invf_plain_;
    const size_t po = size_t(bk.r0) * 4;
    quant_copy(E, X_, 0, D, XQ_, 0, D, M, D);
    gemv8(E, w.wkv, XQ_, D, KVp_, HD, M);
    vec_post(E, KVp_, 0, &w.kv_norm, pos_, po, &invf, M, 1, HD, HD, true, false, true);
    for (Part& pt : bk.parts) {
      Seg& sg = segs[size_t(pt.seg)];
      if (w.ratio > 0 && w.kv_src) compress_seq(E, w, sg, bk.r0, pt);
      const int k0 = std::max(0, pt.m - c.WIN);
      store_half(E, KVp_, pt.ro + k0, HD, sg.s->ring, size_t(l) * size_t(c.WIN) * size_t(HD) * 2, pt.m - k0, HD, pt.p0 + k0, c.WIN);
    }
  }

  // The rows of `bks` in [lo[i], hi[i]) of each segment i (rows from the segment's first), as
  // blocks of contiguous rows on the same grid: a part ends where the full pass's part ends.
  static std::vector<Blk> sub_blocks(const std::vector<Blk>& bks, const std::vector<Seg>& segs, const std::vector<int>& lo,
                                     const std::vector<int>& hi) {
    std::vector<Blk> out;
    for (const Blk& bk : bks) {
      Blk cur;
      for (const Part& pt : bk.parts) {
        const Seg& g = segs[size_t(pt.seg)];
        const int a0 = std::max(bk.r0 + pt.ro, g.r0 + lo[size_t(pt.seg)]), a1 = std::min(bk.r0 + pt.ro + pt.m, g.r0 + hi[size_t(pt.seg)]);
        if (a0 >= a1) continue;
        if (cur.M > 0 && cur.r0 + cur.M != a0) out.push_back(std::move(cur)), cur = Blk{};
        if (cur.M == 0) cur.r0 = a0;
        cur.parts.push_back(Part{pt.seg, a0 - cur.r0, a1 - a0, g.p0 + (a0 - g.r0)});
        cur.M = a1 - cur.r0;
      }
      if (cur.M > 0) out.push_back(std::move(cur));
    }
    return out;
  }

  // The compressed entries and index keys of a KV source layer for one segment part
  // (X_ rows in, the slot's compressed and index-key caches out).
  void compress_seq(Enc2& E, const LayerW& w, Seg& sg, int br0, const Part& pt) {
    const Cfg& c = c_;
    Seq& s = *sg.s;
    const int D = c.D, HD = c.HD;
    const int ro = pt.ro, M = pt.m, p0 = pt.p0, r0 = br0 + ro;
    const Buf& invf = w.ratio ? invf_yarn_ : invf_plain_;
    const size_t po = size_t(r0) * 4;
    const size_t xo = size_t(ro) * size_t(D) * 4, ho = size_t(ro) * size_t(HD) * 4;
    {
      const int r = w.ratio, si = w.src;
      const int g0 = p0 / r, G = (p0 + M) / r - g0;
      if (G > s.cap[size_t(si)] - g0) fail("deepseek_v41: compressed KV full");
      int32_t* gp = static_cast<int32_t*>([gpos_ contents]) + r0;  // this part's own rows
      for (int g = 0; g < G; ++g) gp[g] = (g0 + g) * r;
      if (r == 1) {
        gemvb(E, w.c_wkv, X_, D, LAT_, HD, M, xo, ho);
      } else {
        gemvb(E, w.c_wkv, X_, D, CKV_, HD, M, xo, ho);
        gemvb(E, w.c_wgate, X_, D, CSC_, HD, M, xo, ho);
        if (G > 0) {
          struct {
            int M, n, ratio, p0, g0;
          } p{M, HD, r, p0, g0};
          E.pipe("ds4_compress");
          E.set(0, CKV_, ho);
          E.set(1, CSC_, ho);
          E.set(2, s.pend, size_t(si) * 2 * 8 * size_t(HD) * 4);
          E.set(3, s.pend, size_t(si) * 2 * 8 * size_t(HD) * 4 + 8 * size_t(HD) * 4);
          E.set(4, LAT_, ho);
          E.bytes(5, p);
          E.threads(size_t(HD), size_t(G), 1, 256);
        }
      }
      if (G > 0) {
        const size_t ko = size_t(ro) * size_t(c.IHD) * 4;
        norm(E, LAT_, ho, w.c_norm, G, 1, HD, HD);
        gemvb(E, w.i_wk, LAT_, HD, IK_, c.IHD, G, ho, ko);  // index keys from the RoPE-free latent
        norm(E, IK_, ko, w.i_knorm, G, 1, c.IHD, c.IHD);
        rope(E, IK_, ko, gpos_, po, invf, G, 1, c.IHD, c.IHD, false);
        quant(E, IK_, ko, G, c.IHD, c.IHD, 1);
        store_half(E, IK_, ro, c.IHD, s.idxk[size_t(si)], 0, G, c.IHD, g0, 0);
        rope(E, LAT_, ho, gpos_, po, invf, G, 1, HD, HD, false);
        quant(E, LAT_, ho, G, HD, HD, 2);
        store_half(E, LAT_, ro, HD, s.comp[size_t(si)], 0, G, HD, g0, 0);
      }
      if (r > 1 && sg.verify)  // a verification keeps every row's compressor inputs (for a rollback)
        for (int j = 0; j < M; ++j) {
          const int vr = pt.p0 - sg.p0 + j;
          if (vr >= 8) break;
          const size_t vb = (size_t(si) * 2 * 8 + size_t(vr)) * size_t(HD) * 4;
          copy(E, CKV_, ho + size_t(j) * size_t(HD) * 4, s.vpend.b, s.vpend.off + vb, size_t(HD) * 4);
          copy(E, CSC_, ho + size_t(j) * size_t(HD) * 4, s.vpend.b, s.vpend.off + vb + 8 * size_t(HD) * 4, size_t(HD) * 4);
        }
      if (r > 1) {  // the incomplete group's rows wait in the slot
        const int end = p0 + M, t0 = end - end % r;
        for (int t = std::max(t0, p0); t < end; ++t) {
          const size_t pb = size_t(si) * 2 * 8 * size_t(HD) * 4;
          copy(E, CKV_, ho + size_t(t - p0) * size_t(HD) * 4, s.pend.b, s.pend.off + pb + size_t(t % r) * size_t(HD) * 4, size_t(HD) * 4);
          copy(E, CSC_, ho + size_t(t - p0) * size_t(HD) * 4, s.pend.b,
               s.pend.off + pb + 8 * size_t(HD) * 4 + size_t(t % r) * size_t(HD) * 4, size_t(HD) * 4);
        }
      }
    }
  }

  // The sequence mixing of one segment part: block rows [ro, ro + m) of `s`
  // at positions p0 ...: compressor, indexer, attention, window ring.
  // draft: a DSpark layer (its own ring of the target's states; the block's rows see each other).
  void attention_seq(Enc2& E, const LayerW& w, int l, Seg& sg, int br0, Part& pt, bool draft) {
    const Cfg& c = c_;
    Seq& s = *sg.s;
    const int D = c.D, HD = c.HD, QR = c.QR, HQ = c.H * c.HD;
    const int ro = pt.ro, M = pt.m, p0 = pt.p0, r0 = br0 + ro;  // r0: pass row of the part's first row
    const Buf& invf = w.ratio ? invf_yarn_ : invf_plain_;
    const size_t po = size_t(r0) * 4;
    const size_t xo = size_t(ro) * size_t(D) * 4, ho = size_t(ro) * size_t(HD) * 4;
    // index entries per row (top-k), or 0: every visible entry. Decided by the part's last row, so the index
    // source and the layers reusing its selection agree (a layer's parts end on the same rows as the source's).
    const int topk_mode = w.ratio > 0 && (p0 + M) / w.ratio > c.ITOPK ? c.ITOPK : 0;
    if (w.ratio > 0 && w.kv_src) compress_seq(E, w, sg, br0, pt);
    const size_t io = size_t(r0) * size_t(c.ITOPK) * 4;  // IDX_ rows of this part
    if (w.ratio > 0 && w.idx_src) {
      const int r = w.ratio, si = w.src;
      const int nvis_max = (p0 + M) / r;
      if (nvis_max > c.ITOPK) {
        // candidate blocks: only beyond cand_blocks x cand_bs visible entries do they restrict anything
        const bool cand = c.cand_src >= 0 && nvis_max > c.cand_blocks * c.cand_bs && (w.cand_src || w.uses_cand);
        const int IQN = c.IH * c.IHD;
        const size_t qo = size_t(ro) * size_t(IQN) * 4, wo = size_t(ro) * size_t(c.IH) * 4;
        gemv8(E, w.i_wq_b, QA_, QR, IQ_, IQN, M, size_t(ro) * size_t(QR) * 4, qo);
        rope(E, IQ_, qo, pos_, po, invf, M, c.IH, c.IHD, IQN, false);
        quant(E, IQ_, qo, M * c.IH, c.IHD, c.IHD, 1);
        gemvb(E, w.i_wproj, X_, D, IW_, c.IH, M, xo, wo);
        {
          const float sc = 1.0f / std::sqrt(float(c.IHD)) / std::sqrt(float(c.IH));
          const float p[2] = {float(M * c.IH), sc};
          E.pipe("ds4_scale");
          E.set(0, IW_, wo);
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
          E.set(0, IQ_, qo + size_t(m0) * size_t(IQN) * 4);
          E.set(1, IW_, wo + size_t(m0) * size_t(c.IH) * 4);
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
      }
    }
    // ---- attention over the window and the compressed entries
    {
      struct {
        int M, H, win, p0, ratio, topk, n_comp;
        float scale;
      } ap{M, c.H, c.WIN, p0, w.ratio, w.ratio ? topk_mode : 0, draft ? M : 0, 1.0f / std::sqrt(float(HD))};
      // a few rows (decode): the keys split across a threadgroup per head (ESPETERO_DS4_ATTN1=1: one simdgroup per head)
      static const bool one = env_on("ESPETERO_DS4_ATTN1");
      const bool split = !draft && !one && M <= 4;
      E.pipe(split ? "ds4_attn_split" : "ds4_attn");
      E.set(0, Q_, size_t(ro) * size_t(HQ) * 4);
      E.set(1, KVp_, ho);
      E.set(2, draft ? s.dsring : s.ring, size_t(l) * size_t(c.WIN) * size_t(HD) * 2);
      if (w.ratio > 0) E.set(3, s.comp[size_t(w.src)]);
      else E.set(3, s.ring);
      E.set(4, IDX_, io);
      E.set(5, w.sink);
      E.set(6, Q_, size_t(ro) * size_t(HQ) * 4);  // in place: each simdgroup reads its query before writing
      E.set(7, pos_, po);
      E.bytes(8, ap);
      if (split) E.groups(size_t(c.H), size_t(M), 1, 256);
      else E.groups(size_t((c.H + 7) / 8), size_t(M), 1, 256);
    }
    // ---- the window ring keeps the last WIN rows (a draft block's rows are not kept)
    if (draft) return;
    const int k0 = std::max(0, M - c.WIN);
    store_half(E, KVp_, ro + k0, HD, s.ring, size_t(l) * size_t(c.WIN) * size_t(HD) * 2, M - k0, HD, p0 + k0, c.WIN);
  }

  // Router of rows [r0, r0 + M): XF_ rows -> SEL_ / RW_ rows.
  void moe_route(Enc2& E, const LayerW& w, int r0, int M) {
    const Cfg& c = c_;
    gemvb(E, w.gate, XF_, c.D, RL_, w.nE, M, size_t(r0) * size_t(c.D) * 4);
    struct {
      int M, E, k;
      float rs;
    } p{M, w.nE, w.topk, c.route_scale};
    E.pipe("ds4_route");
    E.set(0, RL_);
    E.set(1, w.gate_bias);
    E.set(2, SEL_, size_t(r0) * size_t(w.topk) * 4);
    E.set(3, RW_, size_t(r0) * size_t(w.topk) * 4);
    E.bytes(4, p);
    E.set(5, w.gate_bias_vl.b ? w.gate_bias_vl : w.gate_bias);
    E.set(6, IMG_, size_t(r0));
    E.groups(size_t(M), 1, 1, 32);
  }

  // The routing of `w` for rows [r0, r0 + M) of XF_ into sel / rw (row stride w.topk).
  void moe_route_to(Enc2& E, const LayerW& w, int r0, int M, id<MTLBuffer> sel, id<MTLBuffer> rw) {
    gemvb(E, w.gate, XF_, c_.D, RL_, w.nE, M, size_t(r0) * size_t(c_.D) * 4);
    struct {
      int M, E, k;
      float rs;
    } p{M, w.nE, w.topk, c_.route_scale};
    E.pipe("ds4_route");
    E.set(0, RL_);
    E.set(1, w.gate_bias);
    E.set(2, sel, size_t(r0) * size_t(w.topk) * 4);
    E.set(3, rw, size_t(r0) * size_t(w.topk) * 4);
    E.bytes(4, p);
    E.set(5, w.gate_bias_vl.b ? w.gate_bias_vl : w.gate_bias);
    E.set(6, IMG_, size_t(r0));
    E.groups(size_t(M), 1, 1, 32);
  }

  // Shared and routed experts of rows [r0, r0 + M) into Y_ (sub-chunk ci; the
  // routing on the host; slot_of: each expert's cache slot for this layer).
  // subset: only these experts' rows (a pipelined layer's round); shared: the shared
  // expert too (it writes the target, the routed rows add to it); Yb: the target
  // rows (null: Y_); par: which copy of the host tables (rounds in flight alternate).
  void moe_experts(Enc2& E, const LayerW& w, int ci, int r0, int M, const std::vector<int>& slot_of,
                   const std::vector<char>* subset = nullptr, bool shared = true, id<MTLBuffer> Yb = nil, size_t ybo = 0,
                   int par = 0) {
    const Cfg& c = c_;
    const int D = c.D, IE = c.IE, K = w.topk;
    id<MTLBuffer> Yt = Yb ? Yb : Y_;
    quant_copy(E, XF_, size_t(r0) * size_t(D) * 4, D, XQ_, 0, D, M, D);
    if (shared) {
      gemv8(E, w.sh13, XQ_, D, SG_, 2 * IE, M);  // gate | up in one matrix
      swiglu_q(E, SG_, 0, SG_, size_t(IE) * 4, 2 * IE, SH_, IE, M, IE, nil, 0);
      gemv8(E, w.sh2, SH_, IE, Yt, D, M, 0, ybo);
    }
    // assignments of the sub-chunk grouped by expert (tables at this sub-chunk's offsets)
    const int32_t* sel = static_cast<const int32_t*>([SEL_ contents]) + size_t(r0) * size_t(K);
    const float* rw = static_cast<const float*>([RW_ contents]) + size_t(r0) * size_t(K);
    std::vector<int> order;
    order.reserve(size_t(M * K));
    for (int a = 0; a < M * K; ++a)
      if (!subset || (*subset)[size_t(sel[a])]) order.push_back(a);
    const int A = int(order.size());
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sel[a] < sel[b]; });
    const size_t ao = size_t(r0) * size_t(K) + size_t(par) * size_t(T_) * size_t(c.K);
    int32_t* xr = static_cast<int32_t*>([xrow_ contents]) + ao;
    float* ws = static_cast<float*>([wsc_ contents]) + ao;
    int32_t* am = static_cast<int32_t*>([amap_ contents]) + ao;
    for (int a = 0; a < M * K; ++a) am[a] = -1;
    if (A == 0) return;
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
    const size_t go = (size_t(par) * size_t(nsub_cap_) + size_t(ci)) * size_t(2 * c.E) * sizeof(XGroup);
    XGroup* grp = reinterpret_cast<XGroup*>(static_cast<uint8_t*>([XGRP_ contents]) + go);
    for (size_t i = 0; i < need.size(); ++i) {
      const int sl = slot_of[size_t(need[i])];
      grp[i] = XGroup{uint32_t(xc_.part(sl)), 0, uint64_t(xc_.offset(sl)), start[i], start[i + 1] - start[i]};
    }
    // Many rows: the GEMM over tiles of 64, 32 or 16 rows of a group (tile lists per
    // size); a few (decode): the grouped GEMV.
    int nt[3] = {0, 0, 0};
    const size_t to = (size_t(par) * size_t(nsub_cap_) + size_t(ci)) * size_t(tile_cap_) * 8;
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
        // two rows per simdgroup (ds4_xgemv_b): 345 -> 419 GB/s gate/up, 236 -> 328 GB/s down; ESPETERO_DS4_XGEMV1=1: one
        static const bool one = env_on("ESPETERO_DS4_XGEMV1");
        auto q = xp;
        q.nrow_tg = one ? 8 : 16;
        E.pipe(one ? "ds4_xgemv" : "ds4_xgemv_b");
        for (int b = 0; b < 3; ++b) E.set(b, xc_.bufs[size_t(std::min<int>(b, int(xc_.bufs.size()) - 1))]);
        E.set(3, XGRP_, go);
        E.set(4, X);
        E.set(5, Y);
        E.set(6, xrow_, ao * 4);
        E.bytes(7, q);
        E.groups(size_t((rows + q.nrow_tg - 1) / q.nrow_tg), need.size(), 1, 256);
      }
    };
    xgemv(XQ_, XG_);
    swiglu_q(E, XG_, 0, XG_, size_t(IE) * 4, 2 * IE, XH_, IE, A, IE, wsc_, ao * 4);
    xp = {D, IE, IE, D, xp.wbytes, xp.sbytes, 1, 8};
    xgemv(XH_, XP_);
    const int32_t p[2] = {D, K};
    E.pipe("ds4_moe_sum");
    E.set(0, XP_);
    E.set(1, Yt, ybo);
    E.set(2, amap_, ao * 4);
    E.bytes(3, p);
    E.threads(size_t(D), size_t(M), 1, 256);
  }

  // O = act_quant(swiglu(G, U) [* w[r]]): G / U rows of stride gus, O of stride os.
  void swiglu_q(Enc2& E, id<MTLBuffer> G, size_t go, id<MTLBuffer> U, size_t uo, int gus, id<MTLBuffer> O, int os, int rows, int n,
                id<MTLBuffer> w, size_t wo) {
    if (noquant()) {  // development: unrounded
      struct {
        int rows, n, gs, us, os;
        float lim;
      } p{rows, n, gus, gus, os, c_.swiglu};
      E.pipe(w ? "ds4_swiglu_w" : "ds4_swiglu");
      E.set(0, G, go);
      E.set(1, U, uo);
      E.set(2, O);
      E.bytes(3, p);
      if (w) E.set(4, w, wo);
      E.threads(size_t(n), size_t(rows), 1, 256);
      return;
    }
    struct {
      int rows, n, gs, us, os;
      float lim;
      int weighted;
    } p{rows, n, gus, gus, os, c_.swiglu, w ? 1 : 0};
    E.pipe("ds4_swiglu_q");
    E.set(0, G, go);
    E.set(1, U, uo);
    E.set(2, O);
    E.bytes(3, p);
    E.set(4, w ? w : O, w ? wo : 0);
    E.threads(size_t(rows) * size_t(n / 32), 1, 1, 256);
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
  // Segments into passes of at most T_ rows and max_reads_ segments; a
  // segment longer than a pass runs in chunks, one per pass.
  void run_segments(std::vector<Seg> todo) {
    while (!todo.empty()) {
      std::vector<Seg> pass, rest;
      int rows = 0;
      for (Seg& g : todo) {
        if (pass.size() >= size_t(max_reads_) || rows >= T_) {
          rest.push_back(g);
          continue;
        }
        const int m = std::min(g.n, T_ - rows);
        if (m < g.n && !g.imgs.empty()) {
          if (rows > 0) {  // the next pass, alone
            rest.push_back(g);
            continue;
          }
          fail("deepseek_v41: a prompt with images longer than a pass (" + std::to_string(T_) + " rows)");
        }
        Seg h = g;
        h.n = m;
        if (m < g.n) {
          h.logits = nullptr;
          Seg tail = g;
          tail.toks += m;
          tail.n -= m;
          rest.push_back(tail);
        }
        pass.push_back(h);
        rows += m;
      }
      forward(pass);
      todo = std::move(rest);
    }
  }

  // Trees: their target slots prepared, then every tree's first segment in
  // shared passes, then the remainders (rows after save_len) likewise.
  void run_trees(const std::vector<TreeSpec>& trees) {
    struct Plan {
      Seq* tgt = nullptr;   // the slot holding parent + root (or the parent itself when there is no root)
      Seq* base = nullptr;  // the state after the whole root: where leaves start
      int first = 0;
    };
    out_.assign(trees.size(), {});
    std::vector<Plan> plan(trees.size());
    size_t scratch = size_t(nuser_);
    auto take_scratch = [&]() -> Seq* {
      if (scratch >= seqs_.size()) fail("deepseek_v41: too many trees without a slot in one pass");
      return &seqs_[scratch++];
    };
    std::vector<Seg> a;
    for (size_t i = 0; i < trees.size(); ++i) {
      const TreeSpec& t = trees[i];
      if (t.commit_chain > 0) fail("deepseek_v41: no verify-commit");
      if (t.root_len <= 0 && t.n_leaves <= 0) fail("deepseek_v41: empty tree");
      out_[i].resize(size_t(std::max(0, t.n_leaves)) + (t.score_root ? 1 : 0));
      Plan& p = plan[i];
      if (t.root_len <= 0) {  // leaves straight off the parent
        if (t.parent < 0) fail("deepseek_v41: leaves without a root need a parent");
        p.tgt = p.base = &seq(t.parent);
        continue;
      }
      const int save_len = t.save_len > 0 ? std::min(t.save_len, t.root_len) : t.root_len;
      p.tgt = t.save >= 0 ? &seq(t.save) : t.work >= 0 ? &seq(t.work) : take_scratch();
      if (t.parent >= 0) fork(*p.tgt, seq(t.parent));
      else reset(*p.tgt);
      p.first = t.save >= 0 ? save_len : t.root_len;
      p.base = p.tgt;
      Seg g;
      g.s = p.tgt, g.toks = t.root, g.n = p.first;
      for (int k = 0; k < t.n_images; ++k) {  // each image goes with the segment that holds its run
        const ImageRef& im = t.images[k];
        const int n = image_run(im.grid_h, im.grid_w);
        if (im.row + n <= p.first) g.imgs.push_back(im);
        else if (im.row < p.first) fail("deepseek_v41: a checkpoint boundary splits an image");
      }
      if (t.score_root && p.first == t.root_len) {
        Readout& r = out_[i].back();
        r.logits.resize(size_t(c_.V));
        g.logits = r.logits.data();
      }
      a.push_back(g);
    }
    run_segments(a);
    std::vector<Seg> b;
    for (size_t i = 0; i < trees.size(); ++i) {
      const TreeSpec& t = trees[i];
      Plan& p = plan[i];
      if (t.root_len <= 0) continue;
      if (p.first < t.root_len) {
        Seq* rest = (t.work >= 0 && &seq(t.work) != p.tgt) ? &seq(t.work) : take_scratch();
        fork(*rest, *p.tgt);
        p.base = rest;
        Seg g;
        g.s = rest, g.toks = t.root + p.first, g.n = t.root_len - p.first;
        for (int k = 0; k < t.n_images; ++k)
          if (t.images[k].row >= p.first) {
            ImageRef im = t.images[k];
            im.row -= p.first;
            g.imgs.push_back(im);
          }
        if (t.score_root) {
          Readout& r = out_[i].back();
          r.logits.resize(size_t(c_.V));
          g.logits = r.logits.data();
        }
        b.push_back(g);
      } else if (t.work >= 0 && &seq(t.work) != p.tgt) {
        fork(seq(t.work), *p.tgt);
      }
    }
    run_segments(b);
    // ---- leaves: each continues its tree's root in a scratch slot of its own, never seeing the others
    struct Leaf {
      size_t tree;
      int k;
    };
    std::vector<Leaf> leaves;
    for (size_t i = 0; i < trees.size(); ++i)
      for (int k = 0; k < trees[i].n_leaves; ++k) leaves.push_back({i, k});
    for (size_t l0 = 0; l0 < leaves.size();) {
      std::vector<Seg> c;
      size_t sc = scratch;
      int rows = 0;
      size_t l1 = l0;
      for (; l1 < leaves.size() && sc < seqs_.size(); ++l1) {
        const TreeSpec& t = trees[leaves[l1].tree];
        const int k = leaves[l1].k;
        const int n = t.leaf_offsets[k + 1] - t.leaf_offsets[k];
        if (n <= 0) fail("deepseek_v41: empty leaf");
        if (!c.empty() && (rows + n > T_ || int(c.size()) >= max_reads_)) break;
        Seq* s = &seqs_[sc++];
        fork(*s, *plan[leaves[l1].tree].base);
        Readout& r = out_[leaves[l1].tree][size_t(k)];
        r.logits.resize(size_t(c_.V));
        Seg g;
        g.s = s, g.toks = t.leaf_tokens + t.leaf_offsets[k], g.n = n, g.logits = r.logits.data();
        c.push_back(g);
        rows += n;
      }
      run_segments(c);
      l0 = l1;
    }
    for (size_t i = 0; i < trees.size(); ++i)
      for (size_t r = 0; r < out_[i].size(); ++r)
        if (!out_[i][r].logits.empty()) finish_readout(trees[i], out_[i][r], int(r));
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

  // Readout ri of tree t (leaves first, then the root).
  void finish_readout(const TreeSpec& t, Readout& o, int ri = 0) {
    const bool leaf = ri < t.n_leaves;
    const int64_t pos = t.draw_pos0 + (leaf ? int64_t(t.leaf_offsets[ri + 1] - t.leaf_offsets[ri]) : 0);
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
    if (t.out_argmax) t.out_argmax[ri] = am;
    if (t.out_lse) t.out_lse[ri] = o.lse;
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
          const float g = -std::log(-std::log(tk_uniform(t.seed, uint64_t(pos), uint64_t(int64_t(o.tk_ids[size_t(q)])))));
          const float sc = std::log(pr[q]) + g;
          if (sc > best) best = sc, sampled = o.tk_ids[size_t(q)];
        }
        o.sampled = sampled;
      }
    }
    if (leaf && t.out_cand_logits && t.cand_offsets && t.cand_ids)
      for (int j = t.cand_offsets[ri]; j < t.cand_offsets[ri + 1]; ++j) t.out_cand_logits[j] = L[t.cand_ids[j]];
  }
};

}  // namespace espetero::native::ds4
