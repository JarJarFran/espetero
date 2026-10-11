// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// NativeModel over Metal: loading, slots, passes, drafting, profiling.
// The only Objective-C++ translation unit of the model; everything above it
// (tokenizer, templates, scheduler, server) is plain C++.

#include "api.h"
#include "impl/bench.h"
#include "impl/forest.h"
#include "impl/load.h"
#include "impl/qwen4_load.h"
#include "kernels_src.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

namespace espetero {

json::Value ModelInfo::to_json() const {
  json::Value o = json::Value::object();
  o.set("device", device);
  o.set("format", format);
  o.set("path", path);
  o.set("hidden", hidden);
  o.set("intermediate", intermediate);
  o.set("layers", layers);
  o.set("full_layers", full_layers);
  o.set("linear_layers", linear_layers);
  o.set("heads", heads);
  o.set("kv_heads", kv_heads);
  o.set("vocab", vocab);
  o.set("slots", slots);
  o.set("kv_blocks", kv_blocks);
  o.set("block_tokens", block_tokens);
  o.set("kv_tokens", int64_t(kv_blocks) * block_tokens);
  o.set("max_ctx", max_ctx);
  o.set("max_rows", max_rows);
  o.set("max_readouts", max_readouts);
  o.set("weight_gb", double(weight_bytes) / 1e9);
  o.set("kv_gb", double(kv_bytes) / 1e9);
  o.set("state_gb", double(state_bytes) / 1e9);
  o.set("slot_state_mb", double(slot_state_bytes) / 1e6);
  o.set("vision", vision);
  o.set("mtp", mtp);
  o.set("load_ms", load_ms);
  o.set("tune_ms", tune_ms);
  if (!extra.is_null()) o.set("details", extra);
  return o;
}

namespace {

using namespace native;

class MetalModel final : public NativeModel {
 public:
  MetalModel(const std::string& path, const LoadOptions& o) {
    const auto t0 = Clock::now();
    // ESPETERO_TRACE_LOAD=1: the model's own load phases (the engine prints the rest).
    const bool trace = [] {
      const char* v = std::getenv("ESPETERO_TRACE_LOAD");
      return v && *v == '1';
    }();
    auto tlast = t0;
    auto mark = [&](const char* what) {
      if (!trace) return;
      const auto now = Clock::now();
      std::fprintf(stderr, "[espetero] load:   %-22s %8.0f ms\n", what,
                   std::chrono::duration<double, std::milli>(now - tlast).count());
      tlast = now;
    };
    Model& m = m_;
    m.path = path;
    m.max_ctx = o.max_ctx;
    m.max_rows = o.max_rows;
    m.max_readouts = o.max_readouts;
    m.prune_last = o.prune_last;
    m.attn_split = o.attn_split;
    if (const char* v = std::getenv("ESPETERO_DN_CHUNK")) m.dn_chunk = std::atoi(v);
    m.tune.gemv = o.gemv;
    m.tune.gemv8 = o.gemv8;
    m.mtp.head4 = o.mtp_head4;
    if (const char* v = std::getenv("ESPETERO_MTP_HEAD4")) m.mtp.head4 = std::atoi(v) != 0;
    m.mtp.coarse_c = o.mtp_coarse;
    if (const char* v = std::getenv("ESPETERO_MTP_COARSE")) m.mtp.coarse_c = std::atoi(v);
    m.mtp.q8 = o.mtp_q8;
    if (const char* v = std::getenv("ESPETERO_MTP_Q8")) m.mtp.q8 = std::atoi(v) != 0;
    std::string dense = o.dense == "all" ? "gate_up,down,in_proj,out" : o.dense;
    for (size_t a = 0; a < dense.size();) {
      size_t b = dense.find(',', a);
      if (b == std::string::npos) b = dense.size();
      const std::string k = dense.substr(a, b - a);
      if (!k.empty()) {
        if (k != "gate_up" && k != "down" && k != "in_proj" && k != "out")
          fail("unknown dense matrix '" + k + "' (gate_up, down, in_proj, out, all)");
        m.dense_mats.insert(k);
      }
      a = b + 1;
    }
    std::string src = kernel_source();
    const auto tk = Clock::now();
    m.init(src);
    // The first start of a newly installed binary compiles the kernels (~5 s on
    // an M1 Max); later starts take them from the system's shader cache (<0.1 s).
    if (const double s = std::chrono::duration<double>(Clock::now() - tk).count(); s > 1.5)
      std::fprintf(stderr, "[espetero] GPU kernels compiled in %.1f s (first start of this build; later starts reuse them)\n", s);
    mark("kernels (compile)");
    {  // checkpoint files are valid for this build only (kernels + host code)
      uint64_t h = 1469598103934665603ull;
      for (unsigned char ch : src + "|" __DATE__ " " __TIME__) h = (h ^ ch) * 1099511628211ull;
      char buf[24];
      std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
      build_id_ = buf;
    }
    // Memory the system can give before this model's allocations without
    // compressing or swapping anything: free pages, the file cache and
    // purgeable pages (the expert cache bound). Inactive pages are not
    // counted: many belong to other applications and would be compressed; a
    // large file cache (after reading model files) is counted.
    double sys_avail0 = 0;
    {
      vm_statistics64_data_t vs0;
      mach_msg_type_number_t cnt0 = HOST_VM_INFO64_COUNT;
      if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vs0), &cnt0) == KERN_SUCCESS)
        sys_avail0 = (double(vs0.free_count) - double(vs0.speculative_count) + double(vs0.external_page_count) +
                      double(vs0.purgeable_count)) * double(vm_kernel_page_size);
    }
    load_qwen35(m, path, o.vision, o.mmproj, o.mtp);
    m.weight_bytes = m.arena->total();
    mark("weights");

    const Config& c = m.cfg;
    // ---- Qwen4-Exp expert cache, sized before the slot/KV pools:
    // explicit (option or ESPETERO_EXPERT_CACHE_GB), or automatic (below).
    double xcache_bytes = 0;
    if (c.q4) {
      // Expert records: the store in --expert-store (default: the
      // per-model place under ~/Library/Caches, which Time Machine skips),
      // built there when missing only on request (--expert-store-build, or
      // ESPETERO_XSTORE=dir: one-time, ~68 GB written for Flash-Next).
      std::string dir = o.expert_store;
      bool build = o.expert_store_build;
      if (const char* d = std::getenv("ESPETERO_XSTORE"); d && *d) dir = d, build = true;
      if (dir.empty() || dir == "auto") dir = default_record_dir(path);
      if (dir != "none" && !dir.empty() && m.xs.present && !open_expert_records(m, dir)) {
        if (build) {
          build_expert_records(m, dir);
          if (!open_expert_records(m, dir)) fail("expert records: cannot open the store just built in " + dir);
        } else {
          std::fprintf(stderr, "expert records: none in %s; --expert-store-build writes them once (%.0f GB) "
                               "so a missing expert is one read instead of nine\n", dir.c_str(), record_store_gb(m));
        }
      }
      double gb = o.expert_cache_gb;
      if (const char* v = std::getenv("ESPETERO_EXPERT_CACHE_GB")) gb = std::atof(v);
      if (gb == 0) {
        // The pools live in GPU-private memory, which the GPU driver keeps
        // wired once used (pageable shared pools were compressed by macOS
        // between command buffers and thrashed near the limit). Everything else
        // the server will hold comes off both bounds below: what is allocated now
        // (weights), the slots + KV pools (the budget below plus make_pools'
        // 8 temporary slots and zero slot) and 2.5 GB of pass scratch and
        // long-prompt staging (about 2.3 GB at peak with 1 MP images).
        // - GPU: the working set (iogpu.wired_limit_mb) less 1.5 GB for other
        //   processes' GPU memory and 1.5 GB of margin: at peak (large images) the
        //   server comes within a few GB of the working set, and a larger cache
        //   runs out of GPU memory.
        // - System: what it could give before this model loaded (sys_avail0)
        //   less 2 GB for macOS, so other applications keep their memory.
        const double kv = (o.memory_gb > 0 ? o.memory_gb * 1e9 : 3e9) + 9.0 * double(slot_state_bytes(c, m.mtp.present));
        const double mine = double([m.dev currentAllocatedSize]), rest = mine + kv + 2.5e9;
        const double ws = double([m.dev recommendedMaxWorkingSetSize]);
        gb = std::min({record_store_gb(m), (ws - rest - 3e9) / 1e9, (sys_avail0 - rest - 2e9) / 1e9});
        std::fprintf(stderr, "expert cache: automatic size from the GPU working set %.1f GB and %.1f GB available before "
                             "loading, less %.1f GB allocated (weights %.1f GB in %.1f GB of buffers) and %.1f GB of "
                             "slots + KV -> %.1f GB\n", ws / 1e9, sys_avail0 / 1e9, mine / 1e9, double(m.weight_bytes) / 1e9,
                     double(m.arena->held()) / 1e9, kv / 1e9, gb);
        if (gb < 3) gb = -1;  // too little memory for a useful cache: whole layers per pass
      }
      if (gb > 0) {
        ensure_expert_pools(m, gb);
        xcache_bytes = double([m.dev currentAllocatedSize]) - double(m.weight_bytes);
      }
      m.expert_cache_gb = gb;
    }

    // ---- pools: user slots and KV blocks within the memory budget
    const size_t slot_bytes = slot_state_bytes(c, m.mtp.present);
    const size_t tok_bytes = kv_bytes_per_token(c, m.mtp.present);
    const double avail = double([m.dev recommendedMaxWorkingSetSize]) - double([m.dev currentAllocatedSize]);
    // Default budget: at most 55% of what is left (scratch for max_rows rows
    // grows with the model) and no more than max(4 GB, 2x the weights), so a
    // small model does not reserve tens of GB it will never use.
    double budget = o.memory_gb > 0 ? o.memory_gb * 1e9
                                    : std::max(1e9, std::min(avail * 0.55, std::max(4e9, 2.0 * double(m.weight_bytes))));
    // Qwen4-Exp: 3 GB for slots + KV unless given (~110K tokens of KV at 27 KB per
    // token beside the slot states); the expert cache took the rest.
    if (c.q4 && o.memory_gb <= 0) budget = 3e9;
    // The Metal working set is per process: other processes are not in
    // it. The automatic budget also stays under 60 % of the memory the
    // system has free now (free + inactive + speculative pages), so several
    // GPU processes do not overcommit it into swap.
    if (o.memory_gb <= 0 && !c.q4) {
      vm_statistics64_data_t vs;
      mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
      if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vs), &cnt) == KERN_SUCCESS) {
        const double page = double(vm_kernel_page_size);
        const double sys_free = (double(vs.free_count) + double(vs.inactive_count) + double(vs.speculative_count)) * page;
        budget = std::max(1e9, std::min(budget, 0.6 * sys_free - xcache_bytes));
      }
    }
    int slots = o.slots;
    size_t kv_tokens = o.kv_gb > 0 ? size_t(o.kv_gb * 1e9 / double(tok_bytes)) : 0;
    if (slots <= 0) {
      const double for_slots = kv_tokens ? budget - double(kv_tokens) * double(tok_bytes) : budget * 0.35;
      slots = int(std::clamp(for_slots / double(slot_bytes), 8.0, 256.0));
    }
    if (!kv_tokens) {
      const double rest = std::max(budget - double(slots) * double(slot_bytes), 0.5e9);
      kv_tokens = size_t(rest / double(tok_bytes));
      kv_tokens = std::min(kv_tokens, size_t(slots) * size_t(m.max_ctx));
      kv_tokens = std::max(kv_tokens, size_t(m.max_ctx) * 2);
    }
    mark("expert cache, sizing");
    make_pools(m, slots, kv_tokens, o.block_tokens);
    mark("slot and KV pools");
    if (const char* v = std::getenv("ESPETERO_FUSE_NORM")) m.fuse_norm = *v == '1';
    // Draft vocabulary: a prefix of the LM head for MTP drafts; verification reads all of it.
    if (o.mtp_vocab > 0) m.mtp.vocab = o.mtp_vocab;
    if (const char* v = std::getenv("ESPETERO_MTP_VOCAB")) m.mtp.vocab = std::atoi(v);
    if (const char* v = std::getenv("ESPETERO_FUSE_NORM_MAXM")) m.fuse_norm_max_m = std::atoi(v);
    if (const char* v = std::getenv("ESPETERO_FUSE_ROUTE")) m.fuse_route = *v == '1';
    if (const char* v = std::getenv("ESPETERO_ROUTE_PAR")) m.route_par = *v == '1';
    if (const char* v = std::getenv("ESPETERO_ROUTE_NSL")) m.route_nsl = std::atoi(v);
    if (const char* v = std::getenv("ESPETERO_ROUTE_SYNC")) m.route_sync = std::atoi(v);
    if (const char* v = std::getenv("ESPETERO_ROUTE_FIN1")) m.route_fin1 = *v == '1';
    if (const char* v = std::getenv("ESPETERO_ATTN_VEC")) m.attn_vec = *v == '1';
    if (const char* v = std::getenv("ESPETERO_ATTN_GQA")) m.attn_gqa = *v == '1';
    if (const char* v = std::getenv("ESPETERO_QSA")) m.qsa = *v == '1';
    if (const char* v = std::getenv("ESPETERO_GEMM_CFGS")) m.tune.n_cfgs = std::max(1, std::min(kGemmConfigs, std::atoi(v)));
    if (const char* v = std::getenv("ESPETERO_ROUTE_IN_GEMV")) m.route_in_gemv = *v == '1';
    if (const char* v = std::getenv("ESPETERO_ROUTE_NOTILES")) m.route_notiles = *v == '1';
    if (const char* v = std::getenv("ESPETERO_ROUTE_SEQEXP")) m.route_seqexp = *v == '1';
    if (const char* v = std::getenv("ESPETERO_MOE_FMA")) m.moe_fma = *v == '1';
    if (const char* v = std::getenv("ESPETERO_MOE_DOWN_COMB")) m.moe_down_comb = *v == '1';
    if (const char* v = std::getenv("ESPETERO_MOE_COMB_KS")) m.moe_comb_ks = std::atoi(v);
    if (const char* v = std::getenv("ESPETERO_MOE_FMA_ROWS")) m.moe_fma_rows = std::max(1, std::min(2, std::atoi(v)));
    if (const char* v = std::getenv("ESPETERO_GEMV_MAXM")) m.tune.gemv_max_m = std::max(1, std::min(m.fma34 ? 4 : kGemvInstM, std::atoi(v)));
    if (const char* v = std::getenv("ESPETERO_GEMV_KS")) m.tune.gemv_ks = std::atoi(v) == 16 ? 16 : 8;
    if (const char* v = std::getenv("ESPETERO_GEMV_KS16_BLOCKS")) m.tune.ks16_blocks = std::max(1, std::atoi(v));
    if (const char* v = std::getenv("ESPETERO_MOE_KS")) m.moe_ks = std::atoi(v);
    if (const char* v = std::getenv("ESPETERO_MOE_DIRECT")) m.moe_direct = *v == '1';
    if (const char* v = std::getenv("ESPETERO_MOE_STAGES")) m.moe_stages = std::atoi(v);
    if (const char* v = std::getenv("ESPETERO_MOE_NBPT")) m.moe_nbpt = std::atoi(v);
    if (const char* v = std::getenv("ESPETERO_MOE_KS_DN")) m.moe_ks_dn = std::atoi(v);
    if (const char* v = std::getenv("ESPETERO_MOE_NBPT_DN")) m.moe_nbpt_dn = std::atoi(v);
    if (const char* v = std::getenv("ESPETERO_MOE_SWAP")) m.moe_swap = *v == '1';
    if (const char* v = std::getenv("ESPETERO_EXPERT_CACHE_GB")) m.expert_cache_gb = std::atof(v);
    // Qwen4-Exp: split K for the few-column decode GEMVs (the hyper-connection
    // reads [324 x 10240]: 32 us unsplit; decode GPU -9 %). Off elsewhere.
    if (m.cfg.q4) m.tune.gemv_splitk = true;
    if (const char* v = std::getenv("ESPETERO_GEMV_SPLITK")) m.tune.gemv_splitk = *v == '1';
    if (const char* v = std::getenv("ESPETERO_FUSE_SHARED")) m.fuse_shared = *v == '1';
    if (const char* v = std::getenv("ESPETERO_FUSE_GDN")) m.fuse_gdn = *v == '1';
    if (const char* v = std::getenv("ESPETERO_GDN_CONV_FOLD")) m.gdn_conv_fold = *v == '1';
    if (const char* v = std::getenv("ESPETERO_WIDEN6_M")) m.tune.widen6_m = std::atoi(v);
    // The fused RMSNorm needs quantized decoder matrices (GEMV epilogues); the
    // MTP layer's passes use explicit norms and routers / shared gates never
    // take norm links, so their dense matrices do not count (the 27B
    // and oQ4e files keep the MTP layer or its fc, and oQ4e the routers, in bf16).
    m.all_quant = m.n_dense - m.n_dense_mtp == 0;

    // ---- GEMM tiles for this device and pipelines built up front
    if (o.tune_max_m > 0) {
      std::vector<const QW*> mats;
      auto add = [&](const Layer& L) {
        for (const Mat& r : L.in_proj)
          if (!r.dense) mats.push_back(&r.q);
        for (const Mat* x : {&L.gate_up, &L.gate, &L.up, &L.down, &L.out, &L.router})
          if (!x->dense && x->out() > 0) mats.push_back(&x->q);
      };
      for (const Layer& L : m.layers) add(L);
      if (m.mtp.present) {
        add(m.mtp.layer);
        if (!m.mtp.fc.dense) mats.push_back(&m.mtp.fc.q);
      }
      autotune(m, m.tune, mats, o.tune_max_m);
    }
    mark("GEMM tiles (autotune)");
    prewarm();
    mark("pipelines (prewarm)");

    // ---- info
    ModelInfo& I = info_;
    I.device = [[m.dev name] UTF8String];
    I.format = m.format;
    I.path = path;
    I.hidden = c.D;
    I.intermediate = c.I;
    I.layers = c.L;
    I.full_layers = c.n_full;
    I.linear_layers = c.n_linear;
    I.heads = c.H;
    I.kv_heads = c.Hkv;
    I.vocab = c.V;
    I.eos_token_id = c.eos_token_id;
    I.slots = m.n_user;
    I.kv_blocks = m.kv.n_blocks;
    I.block_tokens = m.kv.bt;
    I.max_ctx = m.max_ctx;
    I.max_rows = m.max_rows;
    I.max_readouts = m.max_readouts;
    I.weight_bytes = m.weight_bytes;
    I.kv_bytes = size_t([m.kv.buf length]);
    I.state_bytes = size_t([m.st.buf length]);
    I.slot_state_bytes = slot_bytes;
    I.vision = m.vis.c.present;
    I.mtp = m.mtp.present;
    I.leaves = true;  // Qwen4-Exp verifies draft leaves too
    I.hidden_states = !m.cfg.q4;  // run_forest refuses out_hidden for qwen4_exp
    {  // Verify-commit (ESPETERO_COMMIT=0: accepted drafts re-run as roots, as before)
      const char* v = std::getenv("ESPETERO_COMMIT");
      I.commit = (v == nullptr || std::atoi(v) != 0) && !m.cfg.gemma;  // Qwen4-Exp too; not Gemma 4
    }
    I.end_ids = m.cfg.eos_ids;  // Gemma 4: <eos>, <turn|>, <|tool_response>
    if (m.cfg.gemma) {
      I.family = "gemma4";
      I.kv_heads = c.Hkv_g;  // the paged pool's (global layers'); the sliding layers' rings are slot state
    }
    // Qwen4-Exp: 3 drafts per pass (measured faster than 5 drafts or
    // --draft-auto when serving: every verify row brings its own experts
    // through the cache).
    I.draft_default = m.cfg.q4 ? 3 : 0;
    if (m.vis.c.present) {
      I.vision_patch = m.vis.c.patch;
      I.vision_merge = m.vis.c.merge;
      I.vision_patch_dim = m.vis.c.patch_dim();
      I.vision_temporal = m.vis.c.tps;
    }
    json::Value d = json::Value::object();
    if (c.gemma) {
      d.set("sliding_layers", c.n_swa);
      d.set("global_layers", c.n_full);
      d.set("sliding_window", c.window);
      d.set("ring_gb", m.kvs.buf ? double([m.kvs.buf length]) / 1e9 : 0.0);
    }
    d.set("matrices_4bit", m.n_q4);
    d.set("matrices_5bit", m.n_q5);
    d.set("matrices_6bit", m.n_q6);
    d.set("matrices_8bit", m.n_q8);
    d.set("matrices_dense", m.n_dense);
    if (c.E) {
      d.set("experts", c.E);
      d.set("experts_per_token", c.k);
      d.set("expert_width", c.Ie);
    }
    json::Value types = json::Value::object();
    for (auto& [k, v] : m.source_types) types.set(k, v);
    d.set("source_types", types);
    if (m.mtp.present) {
      json::Value mt = json::Value::object();
      mt.set("source", m.mtp.source);
      mt.set("head", m.mtp.head.out() > 0 ? "q4" : "full");
      if (m.mtp.coarse.out() > 0) mt.set("coarse_blocks", m.mtp.coarse_c);
      d.set("mtp", mt);
    }
    I.extra = d;
    I.tune_ms = m.tune.ms;
    I.load_ms = ms_since(t0);
  }

  const ModelInfo& info() const override { return info_; }

  int slot_alloc() override {
    for (int i = 0; i < m_.n_user; ++i) {
      Slot& s = m_.slots[size_t(i)];
      if (!s.used) {
        s = Slot{};
        s.used = true;
        return i;
      }
    }
    return -1;
  }
  void slot_free(int slot) override {
    Slot& s = user_slot(m_, slot);
    m_.release_blocks(s);
    s = Slot{};
  }
  void slot_reset(int slot) override {
    Slot& s = user_slot(m_, slot);
    m_.release_blocks(s);
    s = Slot{};
    s.used = true;
  }
  void slot_fork(int dst, int src) override {
    if (dst == src) return;
    Slot& d = user_slot(m_, dst);
    const Slot& s = user_slot(m_, src);
    m_.release_blocks(d);
    d = s;
    const int nb = (s.len + m_.kv.bt - 1) / m_.kv.bt;
    d.blocks.assign(s.blocks.begin(), s.blocks.begin() + std::min<size_t>(size_t(nb), s.blocks.size()));
    for (int b : d.blocks) m_.kv_retain(b);
    d.cm_k = 0;  // the chain rows belong to the source
    copy_slot_state(m_, dst, src);
  }
  int slot_len(int slot) const override { return m_.slots.at(size_t(slot)).len; }
  int slot_next_pos(int slot) const override { return m_.slots.at(size_t(slot)).next_pos; }
  bool slot_mtp(int slot) const override { return m_.slots.at(size_t(slot)).mtp_ok; }
  int free_slots() const override {
    int n = 0;
    for (const Slot& s : m_.slots) n += s.used ? 0 : 1;
    return n;
  }
  int free_kv_tokens() const override { return m_.kv_free() * m_.kv.bt; }
  int pass_kv_tokens(const std::vector<TreeSpec>& trees) const override {
    const int bt = m_.kv.bt;
    int need = 0;
    for (const TreeSpec& b : trees) {
      const Slot* parent = b.parent >= 0 && b.parent < m_.n_user ? &m_.slots[size_t(b.parent)] : nullptr;
      const int P = parent ? parent->len : 0, Tr = b.root_len;
      const int save_len = b.save_len > 0 ? b.save_len : Tr;
      const bool mtp = b.mtp && m_.mtp.present && Tr > 0 && (b.save >= 0 || b.work >= 0);
      const bool mtp_run = mtp && (P == 0 || parent->mtp_ok);
      const int w0 = mtp_run ? std::max(P - 1, 0) : P;
      // As run_forest counts them (an in-place append keeps a partial block it owns alone).
      if (b.save >= 0 && Tr > 0)
        need += target_new_blocks(m_, parent, P, w0, P + save_len + std::max(b.commit_chain, 0), b.save == b.parent);  // + chain rows
      if (b.work >= 0 && Tr > 0) need += target_new_blocks(m_, parent, P, w0, P + Tr, false);
    }
    return need * bt;
  }
  int freed_kv_tokens(const std::vector<int>& slots) const override {
    // A block goes back to the free list once its last reference is gone: the
    // blocks whose every reference is held by these slots.
    std::vector<int> ids(slots), held;
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    for (int s : ids)
      if (s >= 0 && s < m_.n_user && m_.slots[size_t(s)].used)
        held.insert(held.end(), m_.slots[size_t(s)].blocks.begin(), m_.slots[size_t(s)].blocks.end());
    std::sort(held.begin(), held.end());
    int n = 0;
    for (size_t i = 0; i < held.size();) {
      size_t j = i;
      while (j < held.size() && held[j] == held[i]) ++j;
      if (int(j - i) >= m_.kv.ref[size_t(held[i])]) ++n;
      i = j;
    }
    return n * m_.kv.bt;
  }

  // ---- checkpoint files: header, KV blocks (layer-major), conv/state per
  // linear layer, MTP rows, Qwen4-Exp n-gram mixing state (version 2),
  // Qwen4-Exp sparse attention index blocks (layer-major; version 3),
  // Gemma 4 sliding-window rings (layer-major; version 4)
  struct CkptHeader {
    uint32_t magic, version;
    int32_t len, next_pos, mtp_ok, n_lh, mh_pos[3];
    int32_t n_blocks, n_layers, n_linear;
    uint64_t block_bytes, conv_bytes, st_bytes, mtp_bytes, ple_bytes;
    int32_t i_layers, pad;
    uint64_t iblock_bytes;
    int32_t r_layers, r_blocks;  // ring layers, blocks per ring
    uint64_t rblock_bytes;
  };
  static constexpr uint32_t kCkptMagic = 0x554b5043u;
  size_t ple_floats() const { return m_.cfg.q4 ? size_t(Model::kPleHist) * size_t(m_.cfg.W()) : 0; }
  std::string state_signature() const override {
    const KVPool& K = m_.kv;
    const StatePool& S = m_.st;
    // The build is part of it: a checkpoint holds what this build's kernels
    // computed, so another build computes its own once.
    return std::to_string(m_.weight_bytes) + ":" + std::to_string(K.bt) + ":" + std::to_string(K.n_layers) + ":" +
           std::to_string(K.kv_dim) + ":" + std::to_string(S.conv_floats) + ":" + std::to_string(S.st_floats) + ":" +
           std::to_string(S.mtp_floats) + ":" + std::to_string(ple_floats()) + ":" + build_id_ +
           (m_.kvs.n_layers ? ":r" + std::to_string(m_.kvs.n_layers) + "x" + std::to_string(m_.ring_blocks) : "");
  }
  bool slot_save(int slot, const std::string& path) override {
    const Slot& s = user_slot(m_, slot);
    const KVPool& K = m_.kv;
    const StatePool& S = m_.st;
    const int nb = (s.len + K.bt - 1) / K.bt;
    if (int(s.blocks.size()) < nb) return false;
    const size_t pf = ple_floats();
    if (pf) ensure_ple_state(m_);
    const KVPool& R = m_.kvs;
    CkptHeader h{kCkptMagic, 4, s.len, s.next_pos, s.mtp_ok ? 1 : 0, s.n_lh, {s.mh_pos[0], s.mh_pos[1], s.mh_pos[2]},
                 nb, K.n_layers, int32_t(S.conv_base.size()), uint64_t(K.block_halves() * 2), uint64_t(S.conv_floats * 4),
                 uint64_t(S.st_floats * 4), uint64_t(s.mtp_ok ? S.mtp_floats * 4 : 0), uint64_t(pf ? pf * 4 + 8 : 0),
                 K.i_layers, 0, uint64_t(K.iblock_halves() * 2), R.n_layers, R.n_layers ? m_.ring_blocks : 0,
                 uint64_t(R.n_layers ? R.block_halves() * 2 : 0)};
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = true;
    auto put = [&](const void* p, size_t n) { ok = ok && std::fwrite(p, 1, n, f) == n; };
    put(&h, sizeof(h));
    const uint8_t* kv = static_cast<const uint8_t*>([K.buf contents]);
    for (int l = 0; l < K.n_layers; ++l)
      for (int i = 0; i < nb; ++i) put(kv + (K.layer_base(l) + size_t(s.blocks[size_t(i)]) * K.block_halves()) * 2, h.block_bytes);
    const uint8_t* st = static_cast<const uint8_t*>([S.buf contents]);
    for (size_t k = 0; k < S.conv_base.size(); ++k) {
      put(st + (S.conv_base[k] + size_t(slot) * S.conv_floats) * 4, h.conv_bytes);
      put(st + (S.st_base[k] + size_t(slot) * S.st_floats) * 4, h.st_bytes);
    }
    if (h.mtp_bytes) put(st + S.mh_off(slot) * 4, h.mtp_bytes);
    if (pf) {
      put(static_cast<const float*>([m_.ple_hist contents]) + size_t(slot) * pf, pf * 4);
      put(m_.ple_ctx[size_t(slot)].data(), 8);
    }
    if (K.i_layers) {
      const uint8_t* ib = static_cast<const uint8_t*>([K.ibuf contents]);
      for (int l = 0; l < K.i_layers; ++l)
        for (int i = 0; i < nb; ++i)
          put(ib + (K.ilayer_base(l) + size_t(s.blocks[size_t(i)]) * K.iblock_halves()) * 2, h.iblock_bytes);
    }
    if (R.n_layers) {  // Gemma 4: the slot's ring of every sliding layer
      const uint8_t* rbuf = static_cast<const uint8_t*>([R.buf contents]);
      for (int l = 0; l < R.n_layers; ++l)
        put(rbuf + (R.layer_base(l) + size_t(slot) * size_t(m_.ring_blocks) * R.block_halves()) * 2,
            size_t(h.r_blocks) * h.rblock_bytes);
    }
    return std::fclose(f) == 0 && ok;
  }
  int slot_load(const std::string& path) override {
    const KVPool& K = m_.kv;
    const StatePool& S = m_.st;
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return -1;
    struct Closer {
      int fd;
      ~Closer() { ::close(fd); }
    } closer{fd};
    off_t off = 0;
    bool ok = true;
    auto get = [&](void* p, size_t n) {
      uint8_t* b = static_cast<uint8_t*>(p);
      while (ok && n > 0) {
        const ssize_t r = ::pread(fd, b, n, off);
        if (r <= 0) {
          ok = false;
          break;
        }
        b += r, n -= size_t(r), off += r;
      }
    };
    CkptHeader h;
    get(&h, sizeof(h));
    const size_t pf = ple_floats();
    const KVPool& R = m_.kvs;
    if (!ok || h.magic != kCkptMagic || h.version != 4 || h.n_layers != K.n_layers || h.block_bytes != K.block_halves() * 2 ||
        h.i_layers != K.i_layers || h.iblock_bytes != K.iblock_halves() * 2 || h.r_layers != R.n_layers ||
        h.r_blocks != (R.n_layers ? m_.ring_blocks : 0) || h.rblock_bytes != (R.n_layers ? R.block_halves() * 2 : 0) ||
        h.n_linear != int32_t(S.conv_base.size()) || h.conv_bytes != S.conv_floats * 4 || h.st_bytes != S.st_floats * 4 ||
        (h.mtp_bytes && h.mtp_bytes != S.mtp_floats * 4) || h.ple_bytes != (pf ? pf * 4 + 8 : 0) || h.n_blocks < 0 ||
        h.len < 0 || int64_t(h.len) > int64_t(h.n_blocks) * K.bt)
      return -1;
    const size_t need = sizeof(h) + size_t(h.n_blocks) * size_t(h.n_layers) * h.block_bytes +
                        size_t(h.n_linear) * (h.conv_bytes + h.st_bytes) + h.mtp_bytes + h.ple_bytes +
                        size_t(h.n_blocks) * size_t(h.i_layers) * h.iblock_bytes +
                        size_t(h.r_layers) * size_t(h.r_blocks) * h.rblock_bytes;
    struct stat sb;
    if (::fstat(fd, &sb) != 0 || uint64_t(sb.st_size) != need) return -1;
    if (m_.kv_free() < h.n_blocks) return -2;
    const int slot = slot_alloc();
    if (slot < 0) return -2;
    Slot& s = user_slot(m_, slot);
    s.len = h.len;
    s.next_pos = h.next_pos;
    s.mtp_ok = h.mtp_ok != 0 && h.mtp_bytes != 0;
    s.n_lh = h.n_lh;
    for (int i = 0; i < 3; ++i) s.mh_pos[i] = h.mh_pos[i];
    for (int i = 0; i < h.n_blocks; ++i) s.blocks.push_back(m_.kv_alloc());
    uint8_t* kv = static_cast<uint8_t*>([K.buf contents]);
    for (int l = 0; l < K.n_layers; ++l)
      for (int i = 0; i < h.n_blocks; ++i) get(kv + (K.layer_base(l) + size_t(s.blocks[size_t(i)]) * K.block_halves()) * 2, h.block_bytes);
    uint8_t* st = static_cast<uint8_t*>([S.buf contents]);
    for (size_t k = 0; k < S.conv_base.size(); ++k) {
      get(st + (S.conv_base[k] + size_t(slot) * S.conv_floats) * 4, h.conv_bytes);
      get(st + (S.st_base[k] + size_t(slot) * S.st_floats) * 4, h.st_bytes);
    }
    if (h.mtp_bytes) get(st + S.mh_off(slot) * 4, h.mtp_bytes);
    if (pf) {
      ensure_ple_state(m_);
      get(static_cast<float*>([m_.ple_hist contents]) + size_t(slot) * pf, pf * 4);
      get(m_.ple_ctx[size_t(slot)].data(), 8);
    }
    if (K.i_layers) {
      uint8_t* ib = static_cast<uint8_t*>([K.ibuf contents]);
      for (int l = 0; l < K.i_layers; ++l)
        for (int i = 0; i < h.n_blocks; ++i)
          get(ib + (K.ilayer_base(l) + size_t(s.blocks[size_t(i)]) * K.iblock_halves()) * 2, h.iblock_bytes);
    }
    if (R.n_layers) {  // Gemma 4 rings
      uint8_t* rbuf = static_cast<uint8_t*>([R.buf contents]);
      for (int l = 0; l < R.n_layers; ++l)
        get(rbuf + (R.layer_base(l) + size_t(slot) * size_t(m_.ring_blocks) * R.block_halves()) * 2,
            size_t(h.r_blocks) * h.rblock_bytes);
    }
    if (!ok) {
      slot_free(slot);
      return -1;
    }
    return slot;
  }

  void run(const std::vector<TreeSpec>& trees, PassStats* stats) override {
    @autoreleasepool {
      const FailMarks counters(m_);
      run_forest(m_, meta_, trees, stats);
      // Readout table for logits(): tree -> first readout.
      readout_base_.assign(trees.size(), 0);
      int n = 0;
      for (size_t t = 0; t < trees.size(); ++t) {
        readout_base_[t] = n;
        n += trees[t].n_leaves + (trees[t].score_root ? 1 : 0);
      }
    }
  }
  void run_chain(const std::vector<TreeSpec>& first, int steps, int32_t* out, PassStats* stats) override {
    @autoreleasepool {
      const FailMarks counters(m_);
      const int n = int(first.size());
      if (n < 1 || steps < 1) fail("run_chain: no trees or steps");
      for (const TreeSpec& t : first)
        if (t.parent < 0 || t.parent != t.save || t.root_len != 1 || t.n_leaves || !t.score_root || t.mtp || t.work >= 0 ||
            t.root_pos || t.want_logits || t.cand_offsets || t.out_lse || (t.topk && !t.gpu_draw))
          fail("run_chain: plain single-row greedy decode trees only");
      const auto t0 = Clock::now();
      std::vector<TreeSpec> specs = first;
      std::vector<int32_t> sink(size_t(n), 0), tok(size_t(n), 0);
      for (int i = 0; i < n; ++i) specs[size_t(i)].out_argmax = &sink[size_t(i)];
      // Profiling splits every dispatch into its own command buffer, and the
      // Qwen4-Exp path flushes inside a pass (expert cache): plain passes.
      if (m_.profile || m_.cfg.q4) {
        double g = 0, e = 0;
        for (int j = 0; j < steps; ++j) {
          PassStats ps;
          run_forest(m_, meta_, specs, &ps);
          g += ps.gpu_ms, e += ps.encode_ms;
          for (int i = 0; i < n; ++i) {
            out[size_t(j) * size_t(n) + size_t(i)] = tok[size_t(i)] = sink[size_t(i)];
            specs[size_t(i)].root = &tok[size_t(i)];
          }
        }
        if (stats) stats->gpu_ms = g, stats->encode_ms = e, stats->wall_ms = ms_since(t0);
        return;
      }
      const size_t need = size_t(steps) * size_t(n) * 4;
      if (m_.chain_cap < need) {
        m_.chain_tok = m_.buffer(std::max<size_t>(need, 4096));
        m_.chain_cap = std::max<size_t>(need, 4096);
      }
      if (!m_.chain_ev) m_.chain_ev = [m_.dev newEvent];
      const uint64_t base = m_.chain_counter;
      m_.chain_active = true;
      m_.chain_n = n;
      m_.chain_cbs.clear();
      double enc = 0;
      auto end_chain = [&]() {
        m_.chain_active = false;
        m_.chain_wait = m_.chain_signal = 0;
        m_.chain_counter = base + uint64_t(steps) + 1;  // event values only grow
        for (id<MTLCommandBuffer> cb : m_.chain_cbs) [cb waitUntilCompleted];
      };
      try {
        for (int j = 0; j < steps; ++j) {
          m_.chain_step = j;
          m_.chain_wait = j ? base + uint64_t(j) : 0;
          m_.chain_signal = base + uint64_t(j) + 1;
          for (int i = 0; i < n; ++i) specs[size_t(i)].draw_pos0 = first[size_t(i)].draw_pos0 + j;  // sampled trees
          PassStats ps;
          run_forest(m_, meta_, specs, &ps);
          enc += ps.encode_ms;
        }
      } catch (...) {
        end_chain();
        m_.chain_cbs.clear();
        throw;
      }
      end_chain();
      double g = 0;
      for (id<MTLCommandBuffer> cb : m_.chain_cbs) {
        if ([cb status] != MTLCommandBufferStatusCompleted) {
          NSError* err = [cb error];
          m_.chain_cbs.clear();
          fail(std::string("GPU command buffer failed in a decode chain: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
        }
        g += gpu_ms(cb);
      }
      m_.chain_cbs.clear();
      std::memcpy(out, [m_.chain_tok contents], need);
      readout_base_.assign(first.size(), 0);
      if (stats) {
        stats->gpu_ms = g;
        stats->encode_ms = enc;
        stats->wall_ms = ms_since(t0);
        stats->rows = n * steps;
        stats->readouts = n * steps;
      }
    }
  }
  TopK topk(int tree, int readout) const override {
    TopK r;
    if (!m_.sc.tk_k || !m_.sc.tk_out) return r;
    const uint32_t* o = static_cast<const uint32_t*>([m_.sc.tk_out contents]) +
                        size_t(readout_base_.at(size_t(tree)) + readout) * (2 * 32 + 3);
    r.ids = reinterpret_cast<const int32_t*>(o);
    r.x = reinterpret_cast<const float*>(o + 32);
    r.n = m_.sc.tk_k;
    std::memcpy(&r.max, o + 64, 4);
    std::memcpy(&r.total, o + 65, 4);
    r.sampled = int32_t(o[66]);
    return r;
  }
  const float* logits(int tree, int readout) const override {
    const float* base = static_cast<const float*>([m_.sc.logits contents]);
    return base + size_t(readout_base_.at(size_t(tree)) + readout) * size_t(m_.cfg.V);
  }

  void draft(const std::vector<DraftRequest>& requests, PassStats* stats) override {
    @autoreleasepool {
      const FailMarks counters(m_);
      run_draft(m_, meta_, requests, stats);
    }
  }

  void commit(const std::vector<CommitRequest>& requests, PassStats* stats) override {
    @autoreleasepool {
      const FailMarks counters(m_);
      if (m_.ctr_dirty) reset_arrival_counters(m_);
      run_commit(m_, meta_, requests, stats);
    }
  }

  void encode_images(const std::vector<ImageRef>& images, float* out, PassStats* stats) override {
    @autoreleasepool {
      const FailMarks counters(m_);
      Model& m = m_;
      if (!m.vis.c.present) fail("this model has no vision tower");
      const auto t0 = Clock::now();
      std::vector<ImageJob> jobs;
      int n_patch = 0;
      collect_images(images.data(), int(images.size()), -1, 0, n_patch, jobs);
      if (!n_patch) return;
      ensure_vscratch(m, n_patch);
      prepare_images(m, jobs);
      const int n_tok = n_patch / 4;
      id<MTLBuffer> dst = m.buffer(size_t(n_tok) * size_t(m.vis.c.out) * 4);
      m.dispatches = 0;
      id<MTLCommandBuffer> cb = [m.queue commandBufferWithUnretainedReferences];
      Enc E(m, cb);
      encode_vision(m, E, n_patch, dst);
      cb = E.finish();
      commit_wait(cb);
      std::memcpy(out, [dst contents], size_t(n_tok) * size_t(m.vis.c.out) * 4);
      if (stats) {
        stats->gpu_ms = gpu_ms(cb);
        stats->wall_ms = ms_since(t0);
        stats->rows = n_patch;
        stats->dispatches = m.dispatches;
      }
    }
  }

  bool set_option(const std::string& name, int value) override {
    if (name == "serial") m_.serial = value != 0;
    else if (name == "barrier_all") m_.barrier_all = value != 0;
    else if (name == "dn_chunk") m_.dn_chunk = value;
    else if (name == "fuse_norm") m_.fuse_norm = value != 0;
    else if (name == "fuse_norm_max_m") m_.fuse_norm_max_m = value;
    else if (name == "fuse_gated") m_.fuse_gated = value != 0;
    else if (name == "fuse_route") m_.fuse_route = value != 0;
    else if (name == "route_par") m_.route_par = value != 0;
    else if (name == "route_fin1") m_.route_fin1 = value != 0;
    else if (name == "attn_vec") m_.attn_vec = value != 0;
    else if (name == "attn_gqa") m_.attn_gqa = value != 0;
    else if (name == "attn_dec") m_.attn_vec = value == 1, m_.attn_gqa = value == 2;  // single-row attention: 0 MMA tiles, 1 vec, 2 gqa
    else if (name == "route_in_gemv") m_.route_in_gemv = value != 0;
    else if (name == "gemv_max_m") m_.tune.gemv_max_m = std::max(1, std::min(m_.fma34 ? 4 : kGemvInstM, value));
    else if (name == "gemm_fixed") m_.tune.fixed = value != 0;  // embedding passes
    else if (name == "gemv_ks") m_.tune.gemv_ks = value == 16 ? 16 : 8;
    else if (name == "moe_down_comb") m_.moe_down_comb = value != 0;
    else if (name == "expert_cache_gb") m_.expert_cache_gb = m_.xpool.empty() ? double(value) : m_.expert_cache_gb;
    else if (name == "mtp_coarse") m_.mtp.coarse_c = m_.mtp.coarse.out() > 0 ? std::max(0, std::min(256, value)) : 0;
    else if (name == "mtp_head4") m_.mtp.use_head4 = value != 0;
    else if (name == "moe_comb_ks") m_.moe_comb_ks = value;
    else if (name == "moe_fma_rows") m_.moe_fma_rows = std::max(1, std::min(2, value));
    else if (name == "moe_gemv_cfg") {  // A/B composites (ab_native): 0 (8 slices, 1 block), 1 (gate/up 4; down 4 x 2), 2 gate/up 4 only, 3 down 4 x 2 only
      m_.moe_ks = (value == 1 || value == 2) ? 4 : 8;
      m_.moe_ks_dn = (value == 1 || value == 3) ? 4 : 8;
      m_.moe_nbpt_dn = (value == 1 || value == 3) ? 2 : 1;
    } else if (name == "route_cfg") {  // 0 moe_route_fin, 1 moe_route_fin1, 2 finalize folded into gate/up
      m_.route_fin1 = value >= 1;
      m_.route_in_gemv = value == 2;
    }
    else if (name == "moe_fma") m_.moe_fma = value != 0;
    else if (name == "fuse_shared") m_.fuse_shared = value != 0;
    else if (name == "gemv_splitk") m_.tune.gemv_splitk = value != 0;
    else if (name == "fuse_gdn") m_.fuse_gdn = value != 0;
    else if (name == "gdn_conv_fold") m_.gdn_conv_fold = value != 0;
    else if (name == "widen6_m") m_.tune.widen6_m = std::max(0, value);
    else if (name == "attn_split") m_.attn_split = value;
    else if (name == "qsa") m_.qsa = value != 0;
    else if (name == "prune_last") m_.prune_last = value != 0;
    else return false;
    return true;
  }

  std::string bench_json(const std::string& what) override {
    @autoreleasepool {
      if (what == "gap") return gap_bench(m_);
      if (what == "moe") return moe_micro_bench(m_);
      if (what == "gemv") return gemv_micro_bench(m_);
      if (what == "hgemm") return hgemm_micro_bench(m_);
      if (what == "vitattn") return vitattn_micro_bench(m_);
      if (what == "attn") return attn_micro_bench(m_);
      if (what == "xcache" || what == "xcache_reset") {  // expert cache counters
        const ExpertCacheStats& x = m_.xstats;
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
                      "{\"slots\": %d, \"picks\": %llu, \"hits\": %llu, \"hit_rate\": %.4f, \"loads\": %llu, "
                      "\"gb_read\": %.3f, \"fallbacks\": %llu, \"admits\": %llu, \"handshakes\": %llu, "
                      "\"load_ms\": %.1f, \"wait_ms\": %.1f, \"gpu_ms\": %.1f, \"pf_issued\": %llu, \"pf_used\": %llu, "
                      "\"pf_late\": %llu, \"pf_demand\": %llu, \"pf_gb\": %.3f, \"pf_wait_ms\": %.1f, \"splits\": %llu, "
                      "\"split_ms\": %.1f}",
                      m_.xpool.empty() ? 0 : m_.xpool[0].S, (unsigned long long)x.picks, (unsigned long long)x.hits,
                      x.picks ? double(x.hits) / double(x.picks) : 0.0, (unsigned long long)x.loads, double(x.bytes) / 1e9,
                      (unsigned long long)x.fallbacks, (unsigned long long)x.admits, (unsigned long long)x.handshakes,
                      x.load_ms, x.wait_ms, x.gpu_ms, (unsigned long long)x.pf_issued, (unsigned long long)x.pf_used,
                      (unsigned long long)x.pf_late, (unsigned long long)x.pf_demand, double(x.pf_bytes) / 1e9, x.pf_wait_ms,
                      (unsigned long long)x.splits, x.split_ms);
        if (what == "xcache_reset") m_.xstats = ExpertCacheStats{};
        return buf;
      }
      fail("unknown benchmark " + what);
    }
  }

  void set_profile(bool on) override {
    m_.profile = on;
    m_.prof.clear();
  }
  std::string profile_json() override {
    json::Value o = json::Value::object();
    for (auto& [name, v] : m_.prof) {
      json::Value a = json::Value::array();
      a.push_back(v.first);
      a.push_back(v.second);
      o.set(name, a);
    }
    return json::dump(o);
  }

 private:
  // A submission that ends in an exception (a failed or aborted command
  // buffer, or a pass refused before it ran) marks the arrival counters
  // dirty; the next pass zeroes them (reset_arrival_counters, forest.h).
  class FailMarks {
   public:
    explicit FailMarks(Model& m) : m_(m), n_(std::uncaught_exceptions()) {}
    ~FailMarks() {
      if (std::uncaught_exceptions() > n_) m_.ctr_dirty = true;
    }
    FailMarks(const FailMarks&) = delete;
    FailMarks& operator=(const FailMarks&) = delete;

   private:
    Model& m_;
    int n_;
  };

  // Kernel source: embedded at build time; ESPETERO_KERNELS=<dir> loads the
  // .metal files from a directory instead (kernel development without a rebuild).
  static std::string kernel_source() {
    if (const char* dir = std::getenv("ESPETERO_KERNELS")) {
      std::string out;
      for (const std::string& f : list_files(dir, "", ".metal")) out += read_file(f) + "\n";
      if (out.empty()) fail(std::string("no .metal files in ") + dir);
      return out;
    }
    return std::string(espetero_kernels_metal, espetero_kernels_metal_len);
  }

  // Build every pipeline the model can use, in parallel.
  void prewarm() {
    Model& m = m_;
    std::vector<std::pair<std::string, int>> list;
    auto add_layer = [&](const Layer& L) {
      for (const Mat& r : L.in_proj) gemm_pipelines(r, list, m.fma34);
      for (const Mat* x : {&L.gate_up, &L.gate, &L.up, &L.down, &L.out, &L.router, &L.shared_gate, &L.index_qk})
        gemm_pipelines(*x, list, m.fma34);
      if (L.moe && !m.cfg.gemma) {
        moe_pipelines(m, L, list);  // the passes' own choice per layer (formats may differ by layer)
        for (const char* k : {"moe_topk", "moe_sort", "moe_combine"}) list.push_back({k, 0});
        if (!L.router.dense) list.push_back({"moe_route" + std::to_string(L.router.q.bits), L.router.q.fc()});
        if (!L.router.dense) list.push_back({"moe_route_par" + std::to_string(L.router.q.bits), L.router.q.fc()});
        if (L.router.dense && !L.shared_gate.dense)
          list.push_back({"moe_route_par" + std::to_string(L.shared_gate.q.bits) + "h", L.shared_gate.q.fc()});
        if (L.router.dense && L.shared_gate.dense) list.push_back({"moe_route_parhh", 64});  // F32 router and gate (GGUF), moe_mlp's constants
        list.push_back({"moe_route_fin", 0});
        list.push_back({"moe_route_fin1", 0});
      }
    };
    for (const Layer& L : m.layers) add_layer(L);
    if (m.mtp.present) {
      add_layer(m.mtp.layer);
      gemm_pipelines(m.mtp.fc, list, m.fma34);
      gemm_pipelines(m.mtp.head, list, m.fma34);
      if (m.mtp.coarse.out() > 0 && !m.mtp.coarse.dense) {
        for (const char* n : {"rgemv2_1_store", "rgemv2_2_store", "rgemv8r2_store"}) list.push_back({n, m.mtp.coarse.q.fc()});
        if (!m.lm_head.dense) {
          const std::string b = std::to_string(m.lm_head.q.bits);
          for (const std::string& n : {"rgemv" + b + "_1_store_l", "rgemv" + b + "_2_store_l", "rgemv8r" + b + "_store_l"})
            list.push_back({n, m.lm_head.q.fc()});
        }
        for (const char* k : {"block_max_rows", "topc_blocks", "argmax2_list"}) list.push_back({k, 0});
      }
    }
    gemm_pipelines(m.lm_head, list, m.fma34);
    if (m.cfg.gemma) g4_pipelines(m, list);
    for (const char* k : {"embed_q4", "embed_q8", "embed_f16", "rmsnorm", "conv_prep", "conv_state_save",
                          "gated_rmsnorm", "attn_prep", "attention_combine", "kv_save", "kv_copy_blocks",
                          "gather_rows_f32", "gather_rows_f16", "scatter_rows_f32", "lse_rows", "argmax_rows",
                          "gather_cands", "splitk_reduce_store", "splitk_reduce_add", "swiglu_f32", "mtp_prep",
                          "rows_to_pool", "pool_to_rows"})
      list.push_back({k, 0});
    for (int R : {2, 4, 8, 16}) list.push_back({"deltanet_tr" + std::to_string(R), 0});
    list.push_back({"deltanet_chunk", 0});
    for (int R : {8, 16}) list.push_back({"gdn_decode_fused_tr" + std::to_string(R), 0});
    const int G = m.cfg.H / m.cfg.Hkv;
    list.push_back({G > 8 ? "flash_attention_g6s2" : "flash_attention_g" + std::to_string(G), 0});
    if (m.kv.i_layers)  // Qwen sparse attention
      for (const char* k : {"qsa_prep", "qsa_pool", "qsa_score_cached", "qsa_score_batch", "qsa_select", "qsa_tiles",
                            "qsa_save", "flash_attention_qsa_g6s2"})
        list.push_back({k, 0});
    // The single-row decode kernels exist for groups up to 8: the GQA-row one (default) has 8 MMA rows,
    // the vector one (opt-in, ESPETERO_ATTN_VEC) would need 37 KB of threadgroup memory at G = 12.
    if (m.cfg.H / m.cfg.Hkv <= 8) list.push_back({"attn_decode_vec_g" + std::to_string(m.cfg.H / m.cfg.Hkv), 0});
    if (m.cfg.H / m.cfg.Hkv <= 8) list.push_back({"attn_decode_gqa_g" + std::to_string(m.cfg.H / m.cfg.Hkv), 0});
    if (m.vis.c.present) {
      for (const char* k : {"layernorm", "vit_pos_add", "vit_qkv_prep"}) list.push_back({k, 0});
      list.push_back({"vit_attention2_" + std::to_string(m.vis.c.hd), 0});  // the default variant (vision.h)
      for (const char* t : {"8x8", "16x4", "32x4"})
        for (const char* e : {"store", "add", "gelu_tanh", "gelu_erf", "swiglu"})
          list.push_back({std::string("hgemm_") + t + "_" + e, 0});
    }
    m.prewarm(list);
  }

  Model m_;
  std::string build_id_;  // kernel source + build time hash (checkpoint files)
  PassMeta meta_;
  ModelInfo info_;
  std::vector<int> readout_base_;
};

}  // namespace

std::unique_ptr<NativeModel> NativeModel::load(const std::string& path, const LoadOptions& options) {
  @autoreleasepool {
    return std::make_unique<MetalModel>(path, options);
  }
}

}  // namespace espetero
