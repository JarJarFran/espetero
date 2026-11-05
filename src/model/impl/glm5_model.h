// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// GLM-5.3-Flash (glm5_next) on Metal, docs/glm5.md. Development stage: the
// tier-0 forward pass of one sequence on the FP8 checkpoint (complete, or a
// partial one with the first layers: tests/ref/glm5_ref.py fake-ckpt
// --fill-layers), every layer's streams dumped in tools/glm5_ref.py's format
// so the two compare layer by layer (tests/ref/glm5_compare.py). The
// NativeModel around it (slots, trees, decoding, the expert cache, GGUF
// weights) comes once this matches the reference.
//
// Per layer (docs/glm5.md 1): mHC coefficients and the block input
// (ds4_hc_mix / ds4_hc_pre), then a KDA layer (glm5_kda_*) or a DSA layer (nope
// MLA through the absorbed kv_b halves around DeepSeek-V4.1's latent attention,
// the k-pool indexer above 2051 visible tokens), the residual update
// (ds4_hc_post); then the MLP (dense fp8, or the sigmoid router, the shared
// expert and the routed experts) the same way. Weights stay as stored: fp8
// with 128 x 128 fp32 block scales (ds4_gemv_f8b), bf16 (ds4_gemv_bf16).
#pragma once

#include <sys/stat.h>

#include <dirent.h>

#include "glm5.h"
#include "gguf.h"
#include "../api.h"

namespace espetero::native::glm5 {

// A serial encoder (DeepSeek-V4.1's Enc2 without the pipelining).
struct Enc {
  Runtime& rt;
  id<MTLCommandBuffer> cb = nil;
  id<MTLComputeCommandEncoder> e = nil;
  double gpu_ms = 0;
  explicit Enc(Runtime& r) : rt(r) { begin(); }
  void begin() {
    cb = [rt.queue commandBufferWithUnretainedReferences];
    e = [cb computeCommandEncoder];
  }
  void flush() {
    end();
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
    e = nil;
  }
  void pipe(const char* n) { [e setComputePipelineState:rt.kernel(n)]; }
  void set(int i, id<MTLBuffer> b, size_t off = 0) { [e setBuffer:b offset:off atIndex:i]; }
  void set(int i, const Buf& b, size_t extra = 0) { [e setBuffer:b.b offset:b.off + extra atIndex:i]; }
  template <class T>
  void bytes(int i, const T& v) {
    [e setBytes:&v length:sizeof(T) atIndex:i];
  }
  void groups(size_t x, size_t y, size_t z, size_t tx) {
    if (x && y && z) [e dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(tx, 1, 1)];
  }
  void threads(size_t x, size_t y, size_t z, size_t tx, size_t ty = 1) {
    if (x && y && z) [e dispatchThreads:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
  }
};

enum Fmt : int { kBF16 = 2, kF32 = 3, kF8B = 4, kGG = 5 };  // ds4_gemv formats; kGG: GGUF blocks (gguf_gemv_*)
struct Mat {
  int fmt = kBF16, N = 0, K = 0;
  Buf w, s;
  int gg = 0;              // kGG: the ggml type
  std::vector<Mat> parts;  // rows stacked from matrices of different types (UD-Q2_K_XL's q: Q5_K, k / v: Q6_K)
};
inline const char* gguf_suffix(int t) {
  switch (t) {
    case GGML_Q8_0: return "q8_0";
    case GGML_Q2_K: return "q2_k";
    case GGML_Q3_K: return "q3_k";
    case GGML_Q4_K: return "q4_k";
    case GGML_Q5_K: return "q5_k";
    case GGML_Q6_K: return "q6_k";
    case GGML_IQ3_S: return "iq3_s";
    case GGML_IQ2_S: return "iq2_s";
    case GGML_IQ4_XS: return "iq4_xs";
    case GGML_IQ2_XS: return "iq2_xs";
    case GGML_IQ3_XXS: return "iq3_xxs";
  }
  fail("glm5: GGUF type " + std::to_string(t) + " has no native kernel");
}

// A sequence's state (a slot): per KDA layer the recurrent state [H][128 v][128 k] and the conv state [3][3C]
// (fp32); per DSA layer the latents (half [cap][512]), the pooled indexer keys (half [cap / 4][128]) and the
// key | gate rows of the incomplete pool (fp32 [4][256]).
struct Seq {
  bool used = false;
  int len = 0;
  int mtp_len = 0;  // rows in the MTP layer's caches (its positions start at 0: docs/glm5.md 6)
  std::vector<id<MTLBuffer>> kda, conv, lat, pooled, pend;
};

struct LayerW {
  bool dsa = false, moe = false;
  Buf attn_norm, ffn_norm, hca_fn, hca_base, hca_scale, hcf_fn, hcf_base, hcf_scale;
  // KDA
  Mat qkv, fgb, f_b, g_b, o;  // fgb: [f_a 128 | g_a 128 | b 64] rows
  Buf conv_w, a_rate, dt_bias, o_norm;  // a_rate = exp(A_log) (GGUF: -ssm_a)
  // DSA
  Mat q_a, q_b, kv_a, o_dsa, ikg, iw, iq;  // ikg: [wk 128 | compress_gate 128] rows; iw: weights_proj; iq: wq_b
  Buf q_a_norm, kv_a_norm, kv_b, ik_w, ik_b, ape;
  Mat k_b, v_b;  // GGUF: kv_b_proj's halves as stored there, per head [512][256] and [256][512] (absorbed layout)
  // MLP
  Mat gate, up, down;           // dense, or the shared expert
  Mat router;                   // bf16 [E, D]
  Buf bias;                     // e_score_correction_bias
  int xl = -1;  // the routed experts' entry in the expert cache (ExpertCache::layers)
  // the NextN / MTP layer (a DSA + MoE layer without hyper-connections) also has
  Mat eh_proj;                      // [D, 2D] over [enorm(embed(next)) | hnorm(h)]
  Buf enorm, hnorm, shead_norm;
};

// The routed experts stay in the checkpoint files (docs/glm5.md 8): a pool of equal slots in shared Metal
// buffers, each holding one expert's three matrices as stored (codes, then fp8 scales for the FP8
// checkpoint); after a layer's routing the missing experts are read with parallel preads (F_NOCACHE) into the
// least recently used slots. A layer that needs more experts than the pool holds runs them in groups. When
// every expert fits (UD-Q2_K_XL on a 128 GB Mac) the cache is resident instead: one slot per expert, sized
// per layer (the layers' types differ), filled on first use or at load (ESPETERO_GLM5_PRELOAD=1), never evicted.
struct XMat {
  int fmt = 0, gg = 0, N = 0, K = 0;
  uint64_t woff = 0, soff = 0;  // in the slot
};
struct XPiece {
  int fd = -1;
  uint64_t off = 0, nbytes = 0, dst = 0;
};
struct XLayer {
  XMat m[3];                                // gate, up, down
  std::vector<std::vector<XPiece>> pieces;  // [expert]
  uint64_t bytes = 0;                       // per expert
};
struct ExpertCache {
  std::vector<XLayer> layers;
  uint64_t slot_bytes = 0, total_bytes = 0;
  int n = 0, per_buf = 0, total_experts = 0, E = 0;
  bool resident = false;           // slot = xl * E + e; sbuf / soff place it
  std::vector<int> sbuf;
  std::vector<uint64_t> soff;
  std::vector<id<MTLBuffer>> bufs;
  std::vector<int> key;            // slot -> xl * E + e, -1 free
  std::vector<uint64_t> stamp;     // slot -> last use
  std::unordered_map<int, int> where;
  uint64_t clock = 0, hits = 0, misses = 0;
  double read_ms = 0, read_gb = 0;
  int buf_of(int s) const { return resident ? sbuf[size_t(s)] : s / per_buf; }
  uint64_t off_of(int s) const { return resident ? soff[size_t(s)] : uint64_t(s % per_buf) * slot_bytes; }
  uint8_t* host(int s) const { return static_cast<uint8_t*>([bufs[size_t(buf_of(s))] contents]) + off_of(s); }
  Buf at(int s, uint64_t off) const { return Buf{bufs[size_t(buf_of(s))], size_t(off_of(s) + off)}; }
};

class DevModel {
 public:
  // layers: the first `layers` (0: all present); slots: sequence states (the first one is used by prefill());
  // max_rows: rows per GPU pass (longer segments run in chunks); cache_gb / cache_slots: the expert cache
  // (0: automatic; slots override for tests).
  DevModel(const std::string& dir, const std::string& kernels, int layers, int max_ctx, int slots = 1, int max_rows = 4096,
           double cache_gb = 0, int cache_slots = 0, int max_reads = 64, int eager = 0) {
    gguf_ = !file_exists(dir + "/config.json") || !file_exists(dir + "/model.safetensors.index.json");
    if (gguf_) {
      open_gguf(dir);
    } else {
      idx_.open(dir, true);
      c_ = idx_.c;
      // the MTP layer: in a complete checkpoint when indexed; in a slice only when its tensors were fetched
      const std::string eh = layer_prefix(c_.L) + "eh_proj.weight";
      if (c_.n_mtp > 0 && idx_.ck.has(eh)) {
        if (file_exists(dir + "/.filled.json")) {
          const json::Value f = json::parse(read_file(dir + "/.filled.json"));
          for (const json::Value& v : f.as_array())
            if (v.is_string() && v.as_string() == eh) mtp_ = true;
        } else {
          mtp_ = true;
        }
      }
    }
    L_ = std::min(layers > 0 ? layers : c_.L, gguf_ ? gg_layers_ : idx_.layers);
    if (L_ <= 0) fail("glm5: no complete layer in " + dir);
    max_ctx_ = max_ctx;
    T_ = std::max(16, std::min(max_rows, max_ctx));
    max_reads_ = std::max(1, max_reads);
    rt_.init(kernels);
    rt_.serial = true;
    const auto t0 = Clock::now();
    load();
    const double wgb = double(rt_.arena->total()) / 1e9;
    alloc_common();
    // the first `eager` sequences now (0: all), the others (scratch) on first use: at 128K each is ~1.7 GB
    seqs_.resize(size_t(std::max(1, slots)));
    for (int i = 0; i < int(seqs_.size()); ++i)
      if (eager <= 0 || i < eager) alloc_seq(seqs_[size_t(i)]);
    ensure_scratch(T_);
    make_cache(cache_gb, cache_slots);
    std::fprintf(stderr, "glm5: GPU allocations %.1f GB of a %.1f GB working set (sequences %.2f GB each, %d allocated; expert pool %s)\n",
                 double([rt_.dev currentAllocatedSize]) / 1e9, double([rt_.dev recommendedMaxWorkingSetSize]) / 1e9,
                 double(seq_bytes_) / 1e9, int(std::count_if(seqs_.begin(), seqs_.end(), [](const Seq& q) { return !q.kda.empty(); })),
                 pool_locked_ ? "locked" : "NOT locked");
    if (xc_.resident)
      std::fprintf(stderr, "glm5: %d layers, %.2f GB of weights in memory (%.1f s); experts: all %d resident (%.1f GB) read from %s\n",
                   L_, wgb, ms_since(t0) / 1e3, xc_.n, double(xc_.total_bytes) / 1e9, gguf_ ? "the GGUF files" : "the safetensors");
    else
      std::fprintf(stderr, "glm5: %d layers, %.2f GB of weights in memory (%.1f s); experts: %d slots of %.1f MB (%.1f GB) for %d "
                           "experts (%.1f GB) read from %s\n",
                   L_, wgb, ms_since(t0) / 1e3, xc_.n, double(xc_.slot_bytes) / 1e6, double(xc_.n) * double(xc_.slot_bytes) / 1e9,
                   xc_.total_experts, double(xc_.total_bytes) / 1e9, gguf_ ? "the GGUF files" : "the safetensors");
  }

  int layers() const { return L_; }
  const Cfg& cfg() const { return c_; }

  // Development: one pass over `ids` continuing sequence 0 (dumps per layer when dump_dir is set); the logits of
  // its last `nlog` rows when every layer is loaded and the head is in the checkpoint (else empty).
  std::vector<float> prefill(const std::vector<int32_t>& ids, const std::string& dump_dir, int nlog = 1) {
    std::vector<float> out(size_t(std::clamp(nlog, 1, std::max(1, int(ids.size())))) * size_t(c_.V));
    const bool head = has_head();
    forward(seqs_[0], ids.data(), int(ids.size()), head ? out.data() : nullptr, int(out.size() / size_t(c_.V)), dump_dir);
    if (!head) out.clear();
    return out;
  }

  int nseq() const { return int(seqs_.size()); }
  Seq& seq(int i) {
    Seq& q = seqs_.at(size_t(i));
    if (q.kda.empty()) alloc_seq(q);
    return q;
  }
  const Seq& seq(int i) const { return seqs_.at(size_t(i)); }
  int max_ctx() const { return max_ctx_; }
  int max_rows() const { return T_; }
  int max_reads() const { return max_reads_; }
  size_t seq_bytes() const { return seq_bytes_; }
  size_t weight_bytes() const { return rt_.arena->total(); }
  bool has_head() const { return (L_ == c_.L || head_partial_) && head_m_.w.b; }
  Runtime& rt() { return rt_; }
  std::string cache_json() const {
    char b[256];
    std::snprintf(b, sizeof b, "{\"resident\":%s,\"slots\":%d,\"slot_mb\":%.2f,\"hits\":%llu,\"misses\":%llu,\"read_gb\":%.3f,\"read_ms\":%.1f}",
                  xc_.resident ? "true" : "false", xc_.n, double(xc_.slot_bytes) / 1e6, (unsigned long long)xc_.hits, (unsigned long long)xc_.misses,
                  xc_.read_gb, xc_.read_ms);
    return b;
  }

  // An empty sequence, and a copy of one (the GPU is idle between passes: plain copies of the shared buffers).
  void reset(Seq& q) {
    for (size_t l = 0; l < q.kda.size(); ++l) {
      if (q.kda[l]) std::memset([q.kda[l] contents], 0, [q.kda[l] length]);
      if (q.conv[l]) std::memset([q.conv[l] contents], 0, [q.conv[l] length]);
      if (q.pend[l]) std::memset([q.pend[l] contents], 0, [q.pend[l] length]);
    }
    q.len = 0, q.mtp_len = 0;
  }
  void fork(Seq& d, const Seq& s) {
    if (&d == &s) return;
    for (size_t l = 0; l < s.kda.size(); ++l) {
      auto cp = [](id<MTLBuffer> dst, id<MTLBuffer> src, size_t n) {
        if (src && n) parallel_copy([dst contents], [src contents], n);
      };
      if (s.kda[l]) cp(d.kda[l], s.kda[l], [s.kda[l] length]);
      if (s.conv[l]) cp(d.conv[l], s.conv[l], [s.conv[l] length]);
      if (s.pend[l]) cp(d.pend[l], s.pend[l], [s.pend[l] length]);
      const int n = int(l) == L_ ? s.mtp_len : s.len;  // the MTP layer's caches have their own length
      if (s.lat[l]) cp(d.lat[l], s.lat[l], size_t(n) * size_t(c_.KVR) * 2);
      if (s.pooled[l]) cp(d.pooled[l], s.pooled[l], size_t(n / c_.KPOOL) * size_t(c_.IHD) * 2);
    }
    d.len = s.len, d.mtp_len = s.mtp_len;
  }
  // The state of a sequence as a list of (buffer, bytes): what a checkpoint file holds, in order.
  std::vector<std::pair<id<MTLBuffer>, size_t>> state_parts(const Seq& q, int len, int mtp_len = 0) const {
    std::vector<std::pair<id<MTLBuffer>, size_t>> v;
    for (size_t l = 0; l < q.kda.size(); ++l) {
      const int n = int(l) == L_ ? mtp_len : len;
      if (q.kda[l]) v.push_back({q.kda[l], [q.kda[l] length]}), v.push_back({q.conv[l], [q.conv[l] length]});
      if (q.lat[l]) {
        v.push_back({q.lat[l], size_t(n) * size_t(c_.KVR) * 2});
        v.push_back({q.pooled[l], size_t(n / c_.KPOOL) * size_t(c_.IHD) * 2});
        v.push_back({q.pend[l], [q.pend[l] length]});
      }
    }
    return v;
  }

  // n tokens continuing sequence q, in passes of up to max_rows rows; `logits` (when given) gets the last `nlog`
  // rows' logits [nlog][V] (they must lie in the last pass).
  void forward(Seq& q, const int32_t* toks, int n, float* logits, int nlog, const std::string& dump_dir = "") {
    for (int i = 0; i < n;) {
      const int m = std::min(T_, n - i);
      const bool last = i + m == n;
      pass(q, toks + i, m, last ? logits : nullptr, std::min(nlog, m), dump_dir);
      i += m;
    }
  }

  bool has_mtp() const { return mtp_ && head_m_.w.b; }

  // The MTP layer over T rows of sequence q (docs/glm5.md 6): row i takes the trunk's final-normed hidden HN[i]
  // (a GPU buffer of T x D floats) and the token after it, next[i]; its caches continue at q.mtp_len. Writes the
  // rows' logits [T][V] (T <= max_reads) and, when hout is given, their normed outputs (the next draft step's hidden).
  void mtp_pass(Seq& q, id<MTLBuffer> HN, size_t hoff, const int32_t* next, int T, float* logits, id<MTLBuffer> hout = nil,
                const std::string& dump_dir = "") {
    if (!mtp_) fail("glm5: the checkpoint has no MTP layer");
    if (T <= 0) return;
    if (T > T_ || T > max_reads_) fail("glm5: MTP pass too large");
    const int D = c_.D, p0 = q.mtp_len;
    if (p0 + T > max_ctx_) fail("glm5: MTP context full");
    const LayerW& w = L[size_t(L_)];
    for (int i = 0; i < T; ++i) {
      if (next[i] < 0 || next[i] >= c_.V) fail("glm5: token id out of range");
      static_cast<int32_t*>([ids_ contents])[i] = next[i];
      static_cast<int32_t*>([pos_ contents])[i] = p0 + i;
    }
    S_ = &q;
    Enc E(rt_);
    embed(E, T);  // H_ rows: the next tokens' embeddings, 4 copies each
    auto rows = [&](const char* k, id<MTLBuffer> src, size_t so, int ss, id<MTLBuffer> dst, size_t dof, int ds) {
      const int32_t p[3] = {D, ss, ds};
      E.pipe(k);
      E.set(0, src, so);
      E.set(1, dst, dof);
      E.bytes(2, p);
      E.threads(size_t(D), size_t(T), 1, 256);
    };
    // [enorm(e) | hnorm(h)] in P1_ rows of 2D, then u = eh_proj . it (Y_), the residual r (P6_ is too small: P3_)
    rows("glm5_copy_rows", H_, 0, 4 * D, P1_, 0, 2 * D);
    rows("glm5_copy_rows", HN, hoff, D, P1_, size_t(D) * 4, 2 * D);
    norm(E, P1_, 0, w.enorm, T, 1, D, 2 * D);
    norm(E, P1_, size_t(D) * 4, w.hnorm, T, 1, D, 2 * D);
    gemv(E, w.eh_proj, P1_, 0, 2 * D, RESID_, 0, D, T);  // u: the residual r from here on (RESID_: no block uses it)
    dump(E, dump_dir, "mtp_u", RESID_, size_t(T) * D * 4);
    rows("glm5_copy_rows", RESID_, 0, D, X_, 0, D);
    norm(E, X_, 0, w.attn_norm, T, 1, D, D);
    dsa(E, w, L_, T, p0, dump_dir.empty() ? "" : dump_dir + "/mtp");
    dump(E, dump_dir, "mtp_a", Y_, size_t(T) * D * 4);
    rows("glm5_add_rows", Y_, 0, D, RESID_, 0, D);  // r = u + attention
    rows("glm5_copy_rows", RESID_, 0, D, X_, 0, D);
    norm(E, X_, 0, w.ffn_norm, T, 1, D, D);
    moe(E, w, L_, T, dump_dir.empty() ? "" : dump_dir + "/mtp");
    dump(E, dump_dir, "mtp_f", Y_, size_t(T) * D * 4);
    rows("glm5_add_rows", Y_, 0, D, RESID_, 0, D);  // r += MoE
    rows("glm5_copy_rows", RESID_, 0, D, X_, 0, D);
    norm(E, X_, 0, w.shead_norm, T, 1, D, D);
    if (hout) rows("glm5_copy_rows", X_, 0, D, hout, 0, D);
    dump(E, dump_dir, "mtp_h", X_, size_t(T) * D * 4);
    gemv(E, head_m_, X_, 0, D, LOG_, 0, c_.V, T);
    E.end();
    if (logits) std::memcpy(logits, [LOG_ contents], size_t(T) * size_t(c_.V) * 4);
    if (!dump_dir.empty()) write_file(dump_dir + "/mtp_logits.f32", [LOG_ contents], size_t(T) * size_t(c_.V) * 4);
    q.mtp_len += T;
    S_ = nullptr;
    gpu_ms_ += E.gpu_ms;
  }

  // Development: the MTP layer on the last pass's final-normed hidden (rows 0..T-2 of `ids`, each with the next
  // id), as tools/glm5_ref.py --mtp does on a partial stack. Run right after prefill(ids) on sequence 0.
  void mtp_dev(const std::vector<int32_t>& ids, const std::string& dump_dir) {
    const int T = int(ids.size()) - 1, D = c_.D;
    if (T <= 0) return;
    {
      Enc E(rt_);
      E.pipe("glm5_hc_mean");
      E.set(0, H_);
      E.set(1, HN_);
      E.bytes(2, D);
      E.threads(size_t(D), size_t(T), 1, 256);
      norm(E, HN_, 0, norm_, T, 1, D, D);
      E.end();
    }
    std::vector<int32_t> next(ids.begin() + 1, ids.end());
    if (!dump_dir.empty()) ::mkdir((dump_dir + "/mtp").c_str(), 0755);
    mtp_pass(seqs_[0], HN_, 0, next.data(), T, nullptr, nil, dump_dir);
  }

  void pass(Seq& q, const int32_t* toks, int T, float* logits, int nlog, const std::string& dump_dir) {
    const int D = c_.D, p0 = q.len;
    if (T <= 0) return;
    if (p0 + T > max_ctx_) fail("glm5: context full (" + std::to_string(max_ctx_) + " tokens)");
    for (int i = 0; i < T; ++i) {
      if (toks[i] < 0 || toks[i] >= c_.V) fail("glm5: token id out of range");
      static_cast<int32_t*>([ids_ contents])[i] = toks[i];
      static_cast<int32_t*>([pos_ contents])[i] = p0 + i;
    }
    S_ = &q;
    Enc E(rt_);
    embed(E, T);
    for (int l = 0; l < L_; ++l) {
      const LayerW& w = L[size_t(l)];
      hc_mix(E, w.hca_fn, w.hca_base, w.hca_scale, T);
      hc_pre(E, w.attn_norm, T);
      if (w.dsa) dsa(E, w, l, T, p0, dump_dir);
      else kda(E, w, l, T);
      dump(E, dump_dir, "a" + std::to_string(l), Y_, size_t(T) * D * 4);
      hc_post(E, T);
      hc_mix(E, w.hcf_fn, w.hcf_base, w.hcf_scale, T);
      hc_pre(E, w.ffn_norm, T);
      if (w.moe) moe(E, w, l, T, dump_dir);
      else mlp(E, w.gate, w.up, w.down, X_, T, false);
      dump(E, dump_dir, "f" + std::to_string(l), Y_, size_t(T) * D * 4);
      hc_post(E, T);
      dump(E, dump_dir, "h" + std::to_string(l), H_, size_t(T) * 4 * D * 4);
    }
    nlog = std::clamp(nlog, 1, T);
    if (logits && (L_ == c_.L || head_partial_) && head_m_.w.b) {
      E.pipe("glm5_hc_mean");
      E.set(0, H_, size_t(T - nlog) * 4 * D * 4);
      E.set(1, X_);
      E.bytes(2, D);
      E.threads(size_t(D), size_t(nlog), 1, 256);
      norm(E, X_, 0, norm_, nlog, 1, D, D);
      gemv(E, head_m_, X_, 0, D, LOG_, 0, c_.V, nlog);
      E.end();
      std::memcpy(logits, [LOG_ contents], size_t(nlog) * size_t(c_.V) * 4);
      if (!dump_dir.empty()) write_file(dump_dir + "/logits.f32", logits, size_t(nlog) * size_t(c_.V) * 4);
    } else {
      E.end();
    }
    q.len += T;
    S_ = nullptr;
    gpu_ms_ += E.gpu_ms;
  }
  double gpu_ms() const { return gpu_ms_; }

 private:
  Index idx_;
  Cfg c_;
  int L_ = 0, max_ctx_ = 4096, T_ = 4096, T_cap_ = 0, max_reads_ = 64;
  // development (ESPETERO_GLM5_HEAD_ANY=1): the LM head after the layers loaded, even when that is not all of them
  // (a serving smoke test on a slice: the text is meaningless, the pipeline is the real one)
  bool head_partial_ = env_on("ESPETERO_GLM5_HEAD_ANY");
  std::vector<Seq> seqs_;
  size_t seq_bytes_ = 0;
  Runtime rt_;
  std::vector<LayerW> L;
  Buf embed_, norm_;
  Mat head_m_;
  int embed_gg_ = 0;  // GGUF: token_embd's type (0: bf16 rows)
  // GGUF split files (unsloth's): every shard mapped, tensors by name
  bool gguf_ = false;
  std::vector<std::unique_ptr<GGUFFile>> gf_;
  std::unordered_map<std::string, std::pair<const GGUFFile*, const GGUFFile::Tensor*>> gt_;
  int gg_layers_ = 0;
  double gpu_ms_ = 0;
  Seq* S_ = nullptr;  // the sequence of the pass being encoded
  bool mtp_ = false;  // the MTP layer is loaded: L[L_] (its state at index L_ of every sequence)
  ExpertCache xc_;
  std::vector<int> nc_fds_;  // F_NOCACHE descriptors of the GGUF files (the safetensors ones are ds4::Ckpt's)
  id<MTLBuffer> zeros_ = nil, ninf_ = nil, ident_ = nil, ones_ = nil;
  // pass scratch
  id<MTLBuffer> ids_ = nil, pos_ = nil, H_ = nil, X_ = nil, Y_ = nil, MIX_ = nil, HCP_ = nil, P1_ = nil, P2_ = nil, P3_ = nil,
                P4_ = nil, P5_ = nil, P6_ = nil, RL_ = nil, RSEL_ = nil, RW_ = nil, XR_ = nil, LOG_ = nil, SC_ = nil, HN_ = nil,
                RESID_ = nil;
  size_t sc_bytes_ = 0;

  // ---------------------------------------------------------------- loading
  const ds4::TInfo& t(const std::string& n) const { return idx_.ck.at(n); }
  Buf raw(const std::string& n) {
    const ds4::TInfo& ti = t(n);
    Buf b = rt_.arena->alloc(size_t(ti.nbytes));
    idx_.ck.read(ti, ptr(b));
    return b;
  }
  Buf rawcat(const std::vector<std::string>& ns) {  // tensors of one dtype stacked along their first dimension
    size_t n = 0;
    for (const auto& x : ns) n += size_t(t(x).nbytes);
    Buf b = rt_.arena->alloc(n);
    size_t o = 0;
    for (const auto& x : ns) {
      idx_.ck.read(t(x), static_cast<uint8_t*>(ptr(b)) + o);
      o += size_t(t(x).nbytes);
    }
    return b;
  }
  std::vector<float> floats(const std::string& n) {
    const ds4::TInfo& ti = t(n);
    std::vector<uint8_t> tmp(size_t(ti.nbytes));
    idx_.ck.read(ti, tmp.data());
    std::vector<float> f;
    if (ti.dtype == "BF16") {
      f.resize(tmp.size() / 2);
      for (size_t i = 0; i < f.size(); ++i) f[i] = bf16_to_f32(reinterpret_cast<const uint16_t*>(tmp.data())[i]);
    } else if (ti.dtype == "F32") {
      f.resize(tmp.size() / 4);
      std::memcpy(f.data(), tmp.data(), tmp.size());
    } else {
      fail("glm5: " + n + ": " + ti.dtype + " is not a float vector");
    }
    return f;
  }
  Buf vecf(const std::string& n) {
    const std::vector<float> f = floats(n);
    return rt_.arena->copy(f.data(), f.size() * 4);
  }
  Mat bf16(const std::string& n) {
    const ds4::TInfo& ti = t(n);
    if (ti.dtype != "BF16" || ti.shape.size() != 2) fail("glm5: " + n + " is not a bf16 matrix");
    return Mat{kBF16, int(ti.shape[0]), int(ti.shape[1]), raw(n), {}};
  }
  Mat bf16cat(const std::vector<std::string>& ns) {
    int N = 0;
    for (const auto& x : ns) N += int(t(x).shape[0]);
    return Mat{kBF16, N, int(t(ns[0]).shape[1]), rawcat(ns), {}};
  }
  Mat f8(const std::string& p) {
    const ds4::TInfo& ti = t(p + ".weight");
    if (ti.dtype != "F8_E4M3") fail("glm5: " + p + " is not fp8");
    Mat m{kF8B, int(ti.shape[0]), int(ti.shape[1]), raw(p + ".weight"), raw(p + ".weight_scale_inv")};
    if (m.N % 128 || m.K % 128) fail("glm5: " + p + ": fp8 matrix not a multiple of 128");
    return m;
  }

  void load() {
    if (gguf_) return load_gguf();
    const std::string top = "model.language_model.";
    embed_ = raw(top + "embed_tokens.weight");
    if (idx_.ck.has("lm_head.weight") && (L_ == c_.L || head_partial_)) head_m_ = Mat{kBF16, c_.V, c_.D, raw("lm_head.weight"), {}};
    norm_ = vecf(top + "norm.weight");
    L.resize(size_t(L_ + (mtp_ ? 1 : 0)));
    for (int li = 0; li < int(L.size()); ++li) {
      const int l = li < L_ ? li : c_.L;  // the MTP layer: layers.<L> of the checkpoint, L[L_] here
      LayerW& w = L[size_t(li)];
      const std::string p = layer_prefix(l);
      w.dsa = c_.dsa(l), w.moe = c_.moe(l);
      w.attn_norm = vecf(p + "input_layernorm.weight");
      w.ffn_norm = vecf(p + "post_attention_layernorm.weight");
      if (l < c_.L) {
        w.hca_fn = vecf(p + "hc_attn_fn"), w.hca_base = vecf(p + "hc_attn_base"), w.hca_scale = vecf(p + "hc_attn_scale");
        w.hcf_fn = vecf(p + "hc_ffn_fn"), w.hcf_base = vecf(p + "hc_ffn_base"), w.hcf_scale = vecf(p + "hc_ffn_scale");
      } else {
        w.eh_proj = bf16(p + "eh_proj.weight");
        w.enorm = vecf(p + "enorm.weight"), w.hnorm = vecf(p + "hnorm.weight"), w.shead_norm = vecf(p + "shared_head.norm.weight");
      }
      const std::string a = p + "self_attn.";
      if (!w.dsa) {
        w.qkv = bf16cat({a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"});
        w.fgb = bf16cat({a + "f_a_proj.weight", a + "g_a_proj.weight", a + "b_proj.weight"});
        w.f_b = bf16(a + "f_b_proj.weight");
        w.g_b = bf16(a + "g_b_proj.weight");
        w.o = bf16(a + "o_proj.weight");
        std::vector<float> cw;
        for (const char* n : {"q_conv1d", "k_conv1d", "v_conv1d"}) {
          const std::vector<float> f = floats(a + n + ".weight");  // [C, 1, 4]
          cw.insert(cw.end(), f.begin(), f.end());
        }
        w.conv_w = rt_.arena->copy(cw.data(), cw.size() * 4);
        std::vector<float> ar = floats(a + "A_log");
        for (float& x : ar) x = std::exp(x);
        w.a_rate = rt_.arena->copy(ar.data(), ar.size() * 4);
        w.dt_bias = vecf(a + "dt_bias"), w.o_norm = vecf(a + "o_norm.weight");
      } else {
        w.q_a = f8(a + "q_a_proj"), w.q_b = f8(a + "q_b_proj"), w.kv_a = f8(a + "kv_a_proj_with_mqa"), w.o_dsa = f8(a + "o_proj");
        w.q_a_norm = vecf(a + "q_a_layernorm.weight"), w.kv_a_norm = vecf(a + "kv_a_layernorm.weight");
        w.kv_b = raw(a + "kv_b_proj.weight");
        const std::string ix = a + "indexer.";
        w.ikg = bf16cat({ix + "wk.weight", ix + "index_kpool_compress_gate"});
        w.iw = bf16(ix + "weights_proj.weight");
        w.iq = bf16(ix + "wq_b.weight");
        w.ik_w = vecf(ix + "k_norm.weight"), w.ik_b = vecf(ix + "k_norm.bias"), w.ape = vecf(ix + "index_kpool_compress_ape");
      }
      const std::string m = p + "mlp.";
      if (!w.moe) {
        w.gate = f8(m + "gate_proj"), w.up = f8(m + "up_proj"), w.down = f8(m + "down_proj");
      } else {
        w.router = bf16(m + "gate.weight");
        w.bias = vecf(m + "gate.e_score_correction_bias");
        w.gate = f8(m + "shared_experts.gate_proj"), w.up = f8(m + "shared_experts.up_proj"), w.down = f8(m + "shared_experts.down_proj");
        // the routed experts stay in the shards: a slot is their run there, codes down | gate | up then scales
        XLayer xl;
        const uint64_t ID = uint64_t(c_.IE) * uint64_t(c_.D), SC = uint64_t(c_.IE / 128) * uint64_t(c_.D / 128) * 4;
        xl.m[0] = XMat{kF8B, 0, c_.IE, c_.D, ID, idx_.codes_bytes + SC};          // gate
        xl.m[1] = XMat{kF8B, 0, c_.IE, c_.D, 2 * ID, idx_.codes_bytes + 2 * SC};  // up
        xl.m[2] = XMat{kF8B, 0, c_.D, c_.IE, 0, idx_.codes_bytes};               // down
        xl.bytes = idx_.codes_bytes + idx_.scales_bytes;
        for (int e = 0; e < c_.E; ++e) {
          std::vector<XPiece> ps;
          for (const Piece& pc : idx_.experts[size_t(l)][size_t(e)].pieces) {
            const int fd = idx_.ck.fds_nc[size_t(pc.file)] >= 0 ? idx_.ck.fds_nc[size_t(pc.file)] : idx_.ck.fds[size_t(pc.file)];
            ps.push_back(XPiece{fd, pc.off, pc.nbytes, pc.dst});
          }
          xl.pieces.push_back(std::move(ps));
        }
        w.xl = int(xc_.layers.size());
        xc_.layers.push_back(std::move(xl));
      }
    }
  }

  // ---------------------------------------------------------------- GGUF (unsloth's split files)
  const GGUFFile::Tensor& gtensor(const std::string& n, const GGUFFile** f = nullptr) const {
    auto it = gt_.find(n);
    if (it == gt_.end()) fail("glm5: GGUF tensor " + n + " missing");
    if (f) *f = it->second.first;
    return *it->second.second;
  }
  void open_gguf(const std::string& dir) {
    std::vector<std::string> files;
    if (DIR* d = ::opendir(dir.c_str())) {
      while (dirent* e = ::readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > 5 && n.compare(n.size() - 5, 5, ".gguf") == 0 && n.rfind("mmproj", 0) != 0) files.push_back(dir + "/" + n);
      }
      ::closedir(d);
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) fail("glm5: no config.json or .gguf files in " + dir);
    for (const std::string& f : files) gf_.push_back(std::make_unique<GGUFFile>(f));
    for (const auto& g : gf_)
      for (const auto& [n, t] : g->tensors()) gt_[n] = {g.get(), &t};
    const GGUFFile& m = *gf_[0];  // metadata in the first shard
    if (m.get_string("general.architecture", "") != "glm5-next") fail("glm5: " + files[0] + " is not a glm5-next GGUF");
    const std::string a = "glm5-next.";
    Cfg c;
    c.n_mtp = int(m.get_int(a + "nextn_predict_layers", 1));
    c.L = int(m.get_int(a + "block_count", 46)) - c.n_mtp;
    c.D = int(m.get_int(a + "embedding_length", 4096));
    c.V = int(m.get_int(a + "vocab_size", 154880));
    c.eps = float(m.get_float(a + "attention.layer_norm_rms_epsilon", 1e-5));
    c.kH = int(m.get_int(a + "attention.head_count", 64));
    c.kD = int(m.get_int(a + "kda.head_dim", 128));
    c.conv = int(m.get_int(a + "ssm.conv_kernel", 4));
    c.gate_lb = float(m.get_float(a + "kda.gate_lower_bound", -5.0));
    c.H = c.kH;
    c.QR = int(m.get_int(a + "attention.q_lora_rank", 1536));
    c.KVR = int(m.get_int(a + "attention.kv_lora_rank", 512));
    c.NOPE = int(m.get_int(a + "attention.key_length_mla", 256));
    c.VD = int(m.get_int(a + "attention.value_length_mla", 256));
    c.ROPE = int(m.get_int(a + "rope.dimension_count", 0));
    c.IH = int(m.get_int(a + "attention.indexer.head_count", 32));
    c.IHD = int(m.get_int(a + "attention.indexer.key_length", 128));
    c.ITOPK = int(m.get_int(a + "attention.indexer.top_k", 2048));
    c.KPOOL = int(m.get_int(a + "attention.indexer.kpool", 4));
    c.I_dense = int(m.get_int(a + "feed_forward_length", 12288));
    c.IE = int(m.get_int(a + "expert_feed_forward_length", 2048));
    c.E = int(m.get_int(a + "expert_count", 288));
    c.K = int(m.get_int(a + "expert_used_count", 8));
    c.NSH = int(m.get_int(a + "expert_shared_count", 1));
    c.route_scale = float(m.get_float(a + "expert_weights_scale", 2.5));
    const std::vector<double> clamp = m.get_floats(a + "swiglu_clamp_exp");
    c.swiglu = clamp.empty() ? 10.0f : float(clamp[0]);
    c.hc = int(m.get_int(a + "hyper_connection.count", 4));
    c.hc_iters = int(m.get_int(a + "hyper_connection.sinkhorn_iterations", 20));
    c.hc_eps = float(m.get_float(a + "hyper_connection.epsilon", 1e-6));
    const std::vector<int64_t> kv = m.get_ints(a + "attention.head_count_kv");
    const int dense = int(m.get_int(a + "leading_dense_block_count", 3));
    c.is_dsa.assign(size_t(c.L), 0), c.is_moe.assign(size_t(c.L), 0), c.idx_full.assign(size_t(c.L), 1);
    for (int l = 0; l < c.L; ++l) {
      c.is_dsa[size_t(l)] = l < int(kv.size()) ? kv[size_t(l)] != 0 : l % 4 == 3;
      c.is_moe[size_t(l)] = l >= dense;
    }
    if (m.get_int(a + "expert_gating_func", 2) != 2) fail("glm5: GGUF expert gating is not sigmoid");
    if (c.hc != 4 || c.ROPE != 0 || c.kD != 128) fail("glm5: unsupported GGUF geometry");
    c_ = c;
    // a partial slice (tests/ref/glm5_ref.py fake-gguf --fill-blocks): only those blocks hold data
    gg_layers_ = c.L;
    if (file_exists(dir + "/.slice.json")) {
      const json::Value sl = json::parse(read_file(dir + "/.slice.json"));
      int n = 0;
      for (const json::Value& b : sl.get("blocks")->as_array()) {
        if (int(b.as_int()) == n) ++n;
        if (int(b.as_int()) == c.L) mtp_ = true;
      }
      gg_layers_ = n;
    } else {
      mtp_ = c.n_mtp > 0 && gt_.count("blk." + std::to_string(c.L) + ".nextn.eh_proj.weight") > 0;
    }
  }
  int gfd(const GGUFFile* f) {  // an F_NOCACHE descriptor of the file, opened once
    for (size_t i = 0; i < gf_.size(); ++i)
      if (gf_[i].get() == f) {
        if (nc_fds_.size() < gf_.size()) nc_fds_.assign(gf_.size(), -1);
        if (nc_fds_[i] < 0) {
          nc_fds_[i] = ::open(f->path().c_str(), O_RDONLY);
          if (nc_fds_[i] < 0) fail("glm5: cannot open " + f->path());
          ::fcntl(nc_fds_[i], F_NOCACHE, 1);
        }
        return nc_fds_[i];
      }
    fail("glm5: unknown GGUF file");
  }
  Buf gcopy(const std::string& n) {
    const GGUFFile* f = nullptr;
    const GGUFFile::Tensor& t = gtensor(n, &f);
    return rt_.arena->copy(f->data(t), t.nbytes);
  }
  std::vector<float> gfloats(const std::string& n) {
    const GGUFFile* f = nullptr;
    const GGUFFile::Tensor& t = gtensor(n, &f);
    std::vector<float> v;
    if (t.type == GGML_F32) {
      v.resize(t.nbytes / 4);
      std::memcpy(v.data(), f->data(t), t.nbytes);
    } else if (t.type == GGML_Q8_0) {  // (the hyper-connection fn matrices)
      const uint8_t* p = f->data(t);
      v.resize(t.nbytes / 34 * 32);
      for (size_t b = 0; b < v.size() / 32; ++b) {
        const float d = h2f(p + b * 34);
        for (int j = 0; j < 32; ++j) v[b * 32 + size_t(j)] = d * float(int8_t(p[b * 34 + 2 + size_t(j)]));
      }
    } else {
      fail("glm5: GGUF " + n + ": type " + std::to_string(t.type) + " where floats were expected");
    }
    return v;
  }
  Buf gvec(const std::string& n) {
    const std::vector<float> v = gfloats(n);
    return rt_.arena->copy(v.data(), v.size() * 4);
  }
  // A GGUF matrix (ggml dims [K, N]); F32 ones run through ds4_gemv_f32.
  Mat gmat(const std::string& n) {
    const GGUFFile::Tensor& t = gtensor(n);
    Mat m;
    m.K = int(t.dims[0]);
    m.N = int(t.nbytes / (t.type == GGML_F32 ? size_t(m.K) * 4 : size_t(m.K / ggml_type_info(t.type)->block * ggml_type_info(t.type)->bytes)));
    m.w = gcopy(n);
    if (t.type == GGML_F32) m.fmt = kF32;
    else m.fmt = kGG, m.gg = t.type;
    gguf_suffix(m.fmt == kGG ? m.gg : GGML_Q8_0);  // fails early for a type without a kernel
    return m;
  }
  // Matrices of one width stacked by rows (q | k | v, f_a | g_a | beta, wk | compress_gate): one matrix when
  // their types agree, else parts that gemv runs one after another.
  Mat gcat(const std::vector<std::string>& ns) {
    const GGUFFile::Tensor& t0 = gtensor(ns[0]);
    size_t bytes = 0;
    bool mixed = false;
    for (const auto& n : ns) {
      const GGUFFile::Tensor& t = gtensor(n);
      if (t.dims[0] != t0.dims[0]) fail("glm5: GGUF " + n + " cannot be stacked with " + ns[0]);
      mixed |= t.type != t0.type;
      bytes += t.nbytes;
    }
    if (mixed) {
      Mat m;
      for (const auto& n : ns) m.parts.push_back(gmat(n));
      m.fmt = m.parts[0].fmt, m.K = m.parts[0].K;
      for (const Mat& q : m.parts) m.N += q.N;
      return m;
    }
    Mat m = gmat(ns[0]);
    Buf b = rt_.arena->alloc(bytes);
    size_t o = 0;
    int N = 0;
    for (const auto& n : ns) {
      const GGUFFile* f = nullptr;
      const GGUFFile::Tensor& t = gtensor(n, &f);
      std::memcpy(static_cast<uint8_t*>(ptr(b)) + o, f->data(t), t.nbytes);
      o += t.nbytes;
      N += int(t.nbytes / (t.type == GGML_F32 ? size_t(t.dims[0]) * 4 : size_t(t.dims[0] / ggml_type_info(t.type)->block * ggml_type_info(t.type)->bytes)));
    }
    m.w = b, m.N = N;
    return m;
  }
  void load_gguf() {
    {
      const GGUFFile::Tensor& t = gtensor("token_embd.weight");
      if (t.type == GGML_F32 || t.type == GGML_F16 || t.type == GGML_BF16) fail("glm5: dense GGUF embeddings not handled yet");
      embed_ = gcopy("token_embd.weight");
      embed_gg_ = t.type;
      gguf_suffix(embed_gg_);
    }
    norm_ = gvec("output_norm.weight");
    if ((L_ == c_.L || head_partial_) && gt_.count("output.weight") && gtensor("output.weight").nbytes) head_m_ = gmat("output.weight");
    L.resize(size_t(L_ + (mtp_ ? 1 : 0)));
    for (int li = 0; li < int(L.size()); ++li) {
      const int l = li < L_ ? li : c_.L;  // the MTP layer: layers.<L> of the checkpoint, L[L_] here
      LayerW& w = L[size_t(li)];
      const std::string b = "blk." + std::to_string(l) + ".";
      w.dsa = c_.dsa(l), w.moe = c_.moe(l);
      w.attn_norm = gvec(b + "attn_norm.weight"), w.ffn_norm = gvec(b + "ffn_norm.weight");
      if (l < c_.L) {
        w.hca_fn = gvec(b + "hc_attn_fn.weight"), w.hca_base = gvec(b + "hc_attn_base.weight"), w.hca_scale = gvec(b + "hc_attn_scale.weight");
        w.hcf_fn = gvec(b + "hc_ffn_fn.weight"), w.hcf_base = gvec(b + "hc_ffn_base.weight"), w.hcf_scale = gvec(b + "hc_ffn_scale.weight");
      } else {
        w.eh_proj = gmat(b + "nextn.eh_proj.weight");
        w.enorm = gvec(b + "nextn.enorm.weight"), w.hnorm = gvec(b + "nextn.hnorm.weight");
        w.shead_norm = gvec(b + "nextn.shared_head_norm.weight");
      }
      if (!w.dsa) {
        w.qkv = gcat({b + "attn_q.weight", b + "attn_k.weight", b + "attn_v.weight"});
        w.fgb = gcat({b + "ssm_f_a.weight", b + "ssm_g_a.weight", b + "ssm_beta.weight"});
        w.f_b = gmat(b + "ssm_f_b.weight"), w.g_b = gmat(b + "ssm_g_b.weight"), w.o = gmat(b + "attn_output.weight");
        std::vector<float> cw;
        for (const char* n : {"ssm_conv1d_q", "ssm_conv1d_k", "ssm_conv1d_v"}) {
          const std::vector<float> f = gfloats(b + n + ".weight");  // ggml [4, 1, C]: 4 taps per channel
          cw.insert(cw.end(), f.begin(), f.end());
        }
        w.conv_w = rt_.arena->copy(cw.data(), cw.size() * 4);
        std::vector<float> ar = gfloats(b + "ssm_a");  // -exp(A_log)
        for (float& x : ar) x = -x;
        w.a_rate = rt_.arena->copy(ar.data(), ar.size() * 4);
        w.dt_bias = gvec(b + "ssm_dt.bias"), w.o_norm = gvec(b + "ssm_norm.weight");
      } else {
        w.q_a = gmat(b + "attn_q_a.weight"), w.q_b = gmat(b + "attn_q_b.weight"), w.kv_a = gmat(b + "attn_kv_a_mqa.weight");
        w.o_dsa = gmat(b + "attn_output.weight");
        w.q_a_norm = gvec(b + "attn_q_a_norm.weight"), w.kv_a_norm = gvec(b + "attn_kv_a_norm.weight");
        w.k_b = gmat(b + "attn_k_b.weight"), w.v_b = gmat(b + "attn_v_b.weight");
        w.ikg = gcat({b + "indexer.attn_k.weight", b + "indexer_compressor_gate.weight"});
        w.iw = gmat(b + "indexer.proj.weight"), w.iq = gmat(b + "indexer.attn_q_b.weight");
        w.ik_w = gvec(b + "indexer.k_norm.weight"), w.ik_b = gvec(b + "indexer.k_norm.bias");
        w.ape = gvec(b + "indexer_compressor_ape.weight");
      }
      if (!w.moe) {
        w.gate = gmat(b + "ffn_gate.weight"), w.up = gmat(b + "ffn_up.weight"), w.down = gmat(b + "ffn_down.weight");
      } else {
        w.router = gmat(b + "ffn_gate_inp.weight");
        w.bias = gvec(b + "exp_probs_b.bias");
        w.gate = gmat(b + "ffn_gate_shexp.weight"), w.up = gmat(b + "ffn_up_shexp.weight"), w.down = gmat(b + "ffn_down_shexp.weight");
        // the routed experts stay in the files: one 3-D tensor per matrix ([K, N, E]), expert e's rows contiguous;
        // a slot holds gate | up | down
        XLayer xl;
        xl.pieces.resize(size_t(c_.E));
        uint64_t at = 0;
        int mi = 0;
        for (const char* nm : {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"}) {
          const GGUFFile* f = nullptr;
          const GGUFFile::Tensor& t = gtensor(b + nm + ".weight", &f);
          if (t.dims.size() != 3 || int(t.dims[2]) != c_.E) fail("glm5: GGUF " + b + nm + " is not [K, N, E]");
          gguf_suffix(t.type);
          const uint64_t per = t.nbytes / uint64_t(c_.E);
          xl.m[mi++] = XMat{kGG, t.type, int(t.dims[1]), int(t.dims[0]), at, 0};
          const int fd = gfd(f);
          for (int e = 0; e < c_.E; ++e) xl.pieces[size_t(e)].push_back(XPiece{fd, t.offset + uint64_t(e) * per, per, at});
          at += (per + 255) / 256 * 256;
        }
        xl.bytes = at;
        w.xl = int(xc_.layers.size());
        xc_.layers.push_back(std::move(xl));
      }
    }
  }

  void make_cache(double gb, int slots) {
    ExpertCache& x = xc_;
    x.E = c_.E;
    for (const XLayer& xl : x.layers) {
      x.slot_bytes = std::max(x.slot_bytes, (xl.bytes + 255) / 256 * 256);
      x.total_bytes += xl.bytes * uint64_t(c_.E);
      x.total_experts += c_.E;
    }
    if (x.layers.empty()) return;
    if (slots <= 0) {
      const double ws = double([rt_.dev recommendedMaxWorkingSetSize]), mine = double([rt_.dev currentAllocatedSize]);
      // every expert, packed per layer, in what the working set leaves keeping 1.5 GB: resident
      uint64_t packed = 0;
      for (const XLayer& xl : x.layers) packed += (xl.bytes + 255) / 256 * 256 * uint64_t(c_.E);
      // ESPETERO_GLM5_RESIDENT=1 / 0 forces the choice (1: past the working set's recommendation if need be)
      const char* rs = std::getenv("ESPETERO_GLM5_RESIDENT");
      const int force = rs ? std::atoi(rs) : -1;
      if (force == 1 || (force < 0 && ((gb <= 0 && double(packed) <= ws - mine - 1.5e9) || (gb > 0 && double(packed) <= gb * 1e9)))) {
        make_resident();
        return;
      }
      std::fprintf(stderr, "glm5: the experts (%.1f GB) do not fit in the %.1f GB the working set leaves: LRU pool\n",
                   double(packed) / 1e9, (ws - mine) / 1e9);
      if (gb <= 0) gb = (ws - mine - 6e9) / 1e9;  // otherwise an LRU pool, keeping 6 GB
      slots = int(std::max(1.0, gb * 1e9 / double(x.slot_bytes)));
    }
    x.n = std::min(slots, x.total_experts);
    x.per_buf = int(std::max<uint64_t>(1, (uint64_t(2) << 30) / x.slot_bytes));
    for (int s = 0; s < x.n; s += x.per_buf) x.bufs.push_back(pool_buffer(size_t(std::min(x.per_buf, x.n - s)) * x.slot_bytes));
    x.key.assign(size_t(x.n), -1);
    x.stamp.assign(size_t(x.n), 0);
  }
  // Expert-pool memory locked with mlock, as DSv4's: unlocked, macOS compressed and paged out the experts a
  // pass did not touch (UD-Q2_K_XL resident: 0.1-0.3 tok/s with 22 GB swapped).
  bool pool_locked_ = true;
  id<MTLBuffer> pool_buffer(size_t n) {
    const size_t len = align_up(std::max<size_t>(n, 16), size_t(::getpagesize()));
    void* p = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) fail("glm5: cannot map the expert pool");
    if (pool_locked_) {  // in 256 MB pieces on every core
      const size_t piece = size_t(256) << 20;
      std::atomic<bool> ok{true};
      parallel_for((len + piece - 1) / piece, [&](size_t i) {
        if (::mlock(static_cast<uint8_t*>(p) + i * piece, std::min(piece, len - i * piece)) != 0) ok = false;
      });
      if (!ok) {
        pool_locked_ = false;
        ::munlock(p, len);
        std::fprintf(stderr, "glm5: could not lock the expert pool in memory (vm.user_wire_limit?): macOS may compress it\n");
      }
    }
    id<MTLBuffer> b = [rt_.dev newBufferWithBytesNoCopy:p length:len options:kShared deallocator:nil];
    if (!b) fail("glm5: cannot wrap the expert pool for the GPU");
    Residency::get().add(b);
    return b;
  }
  void make_resident() {
    ExpertCache& x = xc_;
    x.resident = true;
    x.n = x.total_experts;
    x.key.assign(size_t(x.n), -1);
    x.stamp.assign(size_t(x.n), 0);
    x.sbuf.resize(size_t(x.n)), x.soff.resize(size_t(x.n));
    const uint64_t cap = uint64_t(2) << 30;
    uint64_t used = cap;  // in the last buffer
    std::vector<uint64_t> sizes;
    for (int l = 0; l < int(x.layers.size()); ++l) {
      const uint64_t b = (x.layers[size_t(l)].bytes + 255) / 256 * 256;
      for (int e = 0; e < x.E; ++e) {
        if (used + b > cap) sizes.push_back(0), used = 0;
        const int s = l * x.E + e;
        x.sbuf[size_t(s)] = int(sizes.size()) - 1, x.soff[size_t(s)] = used;
        used += b, sizes.back() = used;
      }
    }
    for (uint64_t b : sizes) x.bufs.push_back(pool_buffer(size_t(b)));
    x.total_bytes = 0;
    for (uint64_t b : sizes) x.total_bytes += b;
    if (const char* e = std::getenv("ESPETERO_GLM5_PRELOAD"); e && std::atoi(e) == 1) {
      const auto t0 = Clock::now();
      std::vector<int> all(size_t(x.E));
      std::iota(all.begin(), all.end(), 0);
      for (int l = 0; l < int(x.layers.size()); ++l) ensure(l, all);
      std::fprintf(stderr, "glm5: %d experts (%.1f GB) preloaded in %.1f s\n", x.n, x.read_gb, ms_since(t0) / 1e3);
    }
  }
  // Slots for experts `es` of cache layer `xl` (at most xc_.n of them), reading the missing ones; the slots of
  // this call are not evicted by it. The GPU must not be reading the evicted slots (callers flush first).
  std::vector<int> ensure(int xl, const std::vector<int>& es) {
    ExpertCache& x = xc_;
    const uint64_t now = ++x.clock;
    std::vector<int> slot(es.size(), -1);
    std::vector<size_t> miss;
    if (x.resident) {
      for (size_t i = 0; i < es.size(); ++i) {
        const int s = xl * x.E + es[i];
        slot[i] = s;
        if (x.key[size_t(s)] < 0) x.key[size_t(s)] = s, miss.push_back(i), ++x.misses;
        else ++x.hits;
      }
    }
    for (size_t i = 0; i < es.size() && !x.resident; ++i) {
      auto it = x.where.find(xl * x.E + es[i]);
      if (it != x.where.end()) {
        slot[i] = it->second;
        x.stamp[size_t(it->second)] = now;
        ++x.hits;
      } else {
        miss.push_back(i);
      }
    }
    if (miss.empty()) return slot;
    if (!x.resident) evict(xl, es, miss, slot, now);
    read_experts(xl, es, miss, slot);
    return slot;
  }
  void evict(int xl, const std::vector<int>& es, const std::vector<size_t>& miss, std::vector<int>& slot, uint64_t now) {
    ExpertCache& x = xc_;
    // victims: free slots first, then the oldest stamps below now
    std::vector<int> order(size_t(x.n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      const bool fa = x.key[size_t(a)] < 0, fb = x.key[size_t(b)] < 0;
      if (fa != fb) return fa;
      return x.stamp[size_t(a)] < x.stamp[size_t(b)];
    });
    size_t vi = 0;
    for (size_t i : miss) {
      while (vi < order.size() && x.stamp[size_t(order[vi])] == now) ++vi;
      if (vi >= order.size()) fail("glm5: expert cache too small for one group");
      const int s = order[vi++];
      if (x.key[size_t(s)] >= 0) x.where.erase(x.key[size_t(s)]);
      x.key[size_t(s)] = xl * x.E + es[i];
      x.where[x.key[size_t(s)]] = s;
      x.stamp[size_t(s)] = now;
      slot[i] = s;
      ++x.misses;
    }
  }
  void read_experts(int xl, const std::vector<int>& es, const std::vector<size_t>& miss, const std::vector<int>& slot) {
    ExpertCache& x = xc_;
    const auto t0 = Clock::now();
    std::atomic<bool> ok{true};
    std::atomic<uint64_t> bytes{0};
    std::vector<std::tuple<int, uint64_t, uint64_t, uint8_t*>> reads;  // fd, file offset, bytes, destination
    for (size_t i : miss)
      for (const XPiece& pc : x.layers[size_t(xl)].pieces[size_t(es[i])])
        for (uint64_t o = 0; o < pc.nbytes; o += (4u << 20))
          reads.push_back({pc.fd, pc.off + o, std::min<uint64_t>(4u << 20, pc.nbytes - o), x.host(slot[i]) + pc.dst + o});
    parallel_for(reads.size(), [&](size_t j) {
      const auto& [fd, off, nb, dst] = reads[j];
      if (!pread_all(fd, dst, size_t(nb), size_t(off))) ok = false;
      bytes += nb;
    });
    if (!ok) fail("glm5: expert read failed");
    x.read_ms += ms_since(t0);
    x.read_gb += double(bytes.load()) / 1e9;
  }
  Mat xmat(int xl, int slot, int m) const {
    const XMat& a = xc_.layers[size_t(xl)].m[m];
    Mat r;
    r.fmt = a.fmt, r.gg = a.gg, r.N = a.N, r.K = a.K;
    r.w = xc_.at(slot, a.woff);
    if (a.fmt == kF8B) r.s = xc_.at(slot, a.soff);
    return r;
  }

  void alloc_common() {
    float ninf[64];
    for (float& x : ninf) x = -INFINITY;
    ninf_ = rt_.buffer(sizeof ninf);
    std::memcpy([ninf_ contents], ninf, sizeof ninf);
    zeros_ = rt_.buffer(4096);
    std::memset([zeros_ contents], 0, 4096);
  }
  void alloc_seq(Seq& q) {
    auto zero = [&](size_t n) {
      id<MTLBuffer> b = rt_.buffer(n);
      std::memset([b contents], 0, n);
      return b;
    };
    const int C = c_.kH * c_.kD;
    size_t bytes = 0;
    const int NL = int(L.size());  // the trunk's layers, then the MTP layer's state (index L_)
    q.kda.assign(size_t(NL), nil), q.conv.assign(size_t(NL), nil), q.lat.assign(size_t(NL), nil);
    q.pooled.assign(size_t(NL), nil), q.pend.assign(size_t(NL), nil);
    for (int l = 0; l < NL; ++l) {
      if (!L[size_t(l)].dsa) {
        q.kda[size_t(l)] = zero(size_t(c_.kH) * 128 * 128 * 4);
        q.conv[size_t(l)] = zero(size_t(3) * 3 * C * 4);
        bytes += size_t(c_.kH) * 128 * 128 * 4 + size_t(9) * C * 4;
      } else {
        q.lat[size_t(l)] = zero(size_t(max_ctx_) * c_.KVR * 2);
        q.pooled[size_t(l)] = zero(size_t(max_ctx_ / 4 + 1) * c_.IHD * 2);
        q.pend[size_t(l)] = zero(size_t(4) * 256 * 4);
        bytes += size_t(max_ctx_) * c_.KVR * 2 + size_t(max_ctx_ / 4 + 1) * c_.IHD * 2 + 4096;
      }
    }
    seq_bytes_ = bytes;
    q.len = 0, q.mtp_len = 0;
  }

  void ensure_scratch(int T) {
    if (T <= T_cap_) return;
    T_cap_ = T;
    const int D = c_.D, C3 = 3 * c_.kH * c_.kD;
    auto F = [&](size_t n) { return rt_.buffer(n * 4); };
    ids_ = F(T), pos_ = F(T);
    H_ = F(size_t(T) * 4 * D), X_ = F(size_t(T) * D), Y_ = F(size_t(T) * D), MIX_ = F(size_t(T) * 24);
    HCP_ = F(size_t(T) * (4 * D / 1024) * 25);
    const size_t big = std::max({size_t(C3), size_t(c_.H) * 512, size_t(c_.I_dense), size_t(c_.E)});
    P1_ = F(size_t(T) * big), P2_ = F(size_t(T) * big), P3_ = F(size_t(T) * big);
    P4_ = F(size_t(T) * std::max(size_t(c_.H) * 256, size_t(c_.I_dense))), P5_ = F(size_t(T) * 512), P6_ = F(size_t(T) * 512);
    RL_ = F(size_t(T) * c_.E), RSEL_ = F(size_t(T) * c_.K), RW_ = F(size_t(T) * c_.K), XR_ = F(size_t(T) * c_.K * 3 + 64);
    HN_ = F(size_t(T) * D), RESID_ = F(size_t(T) * D);
    LOG_ = F(size_t(std::max(1, max_reads_)) * c_.V);
    ident_ = F(size_t(T) * c_.K), ones_ = F(size_t(T) * c_.K);
    for (int i = 0; i < T * c_.K; ++i) {
      static_cast<int32_t*>([ident_ contents])[i] = i;
      static_cast<float*>([ones_ contents])[i] = 1.0f;
    }
  }

  // ---------------------------------------------------------------- encoding helpers
  struct GemvP {
    int N, K, M, xs, ys, grp, gstr, accum, idx, nrow_tg;
  };
  void gemv(Enc& E, const Mat& m, id<MTLBuffer> X, size_t xoff, int xs, id<MTLBuffer> Y, size_t yoff, int ys, int M,
            bool accum = false, id<MTLBuffer> xrow = nil, size_t xro = 0, id<MTLBuffer> yrow = nil, size_t yro = 0,
            id<MTLBuffer> ysc = nil, size_t yso = 0, int grp = 0, int gstr = 0) {
    if (M <= 0) return;
    if (!m.parts.empty()) {  // stacked rows of different types: each part writes its own columns of Y
      if (xrow || yrow || ysc || grp) fail("glm5: stacked mixed-type matrix used with indexed rows");
      int n0 = 0;
      for (const Mat& q : m.parts) {
        gemv(E, q, X, xoff, xs, Y, yoff + size_t(n0) * 4, ys, M, accum);
        n0 += q.N;
      }
      return;
    }
    if (m.fmt == kGG) E.pipe((std::string("gguf_gemv_") + gguf_suffix(m.gg)).c_str());
    else E.pipe(m.fmt == kF8B ? "ds4_gemv_f8b" : m.fmt == kBF16 ? "ds4_gemv_bf16" : "ds4_gemv_f32");
    // indexed rows when any of xrow / yrow / ysc is given (the others default to the identity and 1.0)
    GemvP p{m.N, m.K, M, xs, ys, grp, gstr, accum ? 1 : 0, xrow || yrow || ysc ? 1 : 0, 8};
    E.set(0, m.w);
    E.set(1, m.s.b ? m.s : m.w);
    E.set(2, X, xoff);
    E.set(3, Y, yoff);
    E.bytes(4, p);
    E.set(5, xrow ? xrow : ident_, xro);
    E.set(6, yrow ? yrow : ident_, yro);
    E.set(7, ysc ? ysc : ones_, yso);
    E.groups(size_t((m.N + 7) / 8), size_t((M + 7) / 8), 1, 256);
  }
  void norm(Enc& E, id<MTLBuffer> x, size_t off, const Buf& w, int rows, int heads, int n, int stride) {
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
  struct HcP {
    int D, iters;
    float eps, hc_eps;
  };
  HcP hcp() const { return HcP{c_.D, c_.hc_iters, c_.eps, c_.hc_eps}; }
  void hc_mix(Enc& E, const Buf& fn, const Buf& base, const Buf& scale, int T) {
    E.pipe("ds4_hc_part");
    E.set(0, H_);
    E.set(1, fn);
    E.set(2, HCP_);
    E.bytes(3, hcp());
    E.groups(size_t(4 * c_.D / 1024), size_t(T), 1, 256);
    E.pipe("ds4_hc_fin");
    E.set(0, HCP_);
    E.set(1, scale);
    E.set(2, base);
    E.set(3, MIX_);
    E.bytes(4, hcp());
    E.threads(size_t(T), 1, 1, 32);
  }
  void hc_pre(Enc& E, const Buf& w, int T) {
    E.pipe("ds4_hc_pre");
    E.set(0, H_);
    E.set(1, MIX_);
    E.set(2, w);
    E.set(3, X_);
    E.bytes(4, hcp());
    E.groups(size_t(T), 1, 1, 256);
  }
  // Token embeddings of ids_ rows into H_ (4 copies per row).
  void embed(Enc& E, int T) {
    const int D = c_.D;
    if (embed_gg_) {
      E.pipe((std::string("gguf_embed_") + gguf_suffix(embed_gg_)).c_str());
      E.set(0, embed_);
      E.set(1, ids_);
      E.set(2, H_);
      E.bytes(3, D);
      E.threads(size_t(D / 32), size_t(T), 1, 64);
    } else {
      const int32_t p[2] = {D, 4};
      E.pipe("ds4_embed");
      E.set(0, ids_);
      E.set(1, embed_);
      E.set(2, H_);
      E.bytes(3, p);
      E.threads(size_t(D), size_t(T), 1, 256);
    }
  }
  void hc_post(Enc& E, int T) {
    E.pipe("ds4_hc_post");
    E.set(0, H_);
    E.set(1, Y_);
    E.set(2, MIX_);
    E.bytes(3, hcp());
    E.threads(size_t(c_.D), size_t(T), 1, 256);
  }
  void swiglu(Enc& E, id<MTLBuffer> G, id<MTLBuffer> U, id<MTLBuffer> O, int rows, int n) {
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
  void dump(Enc& E, const std::string& dir, const std::string& name, id<MTLBuffer> b, size_t n) {
    if (dir.empty()) return;
    E.flush();
    write_file(dir + "/" + name + ".f32", [b contents], n);
  }
  static void write_file(const std::string& path, const void* p, size_t n) {
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
      std::fwrite(p, 1, n, f);
      std::fclose(f);
    }
  }

  // ---------------------------------------------------------------- blocks (X_ in, Y_ out)
  void kda(Enc& E, const LayerW& w, int l, int T) {
    const int D = c_.D, H = c_.kH, C = H * 128, C3 = 3 * C;
    gemv(E, w.qkv, X_, 0, D, P1_, 0, C3, T);  // q | k | v projections
    {
      const Glm5Conv p{T, C3, 3 * H, H};
      E.pipe("glm5_kda_conv");
      E.set(0, P1_);
      E.set(1, S_->conv[size_t(l)]);
      E.set(2, w.conv_w);
      E.set(3, P2_);
      E.bytes(4, p);
      E.groups(size_t((3 * H + 7) / 8), size_t(T), 1, 256);
      const int32_t q[2] = {T, C3};
      E.pipe("glm5_conv_save");
      E.set(0, P1_);
      E.set(1, S_->conv[size_t(l)]);
      E.bytes(2, q);
      E.threads(size_t(C3), 1, 1, 256);
    }
    const int FGB = w.fgb.N;  // 320
    gemv(E, w.fgb, X_, 0, D, P5_, 0, FGB, T);
    gemv(E, w.f_b, P5_, 0, FGB, P3_, 0, C, T);                                 // f
    gemv(E, w.g_b, P5_, size_t(c_.kD) * 4, FGB, P4_, 0, C, T);                // z (output gate)
    {
      struct {
        int rows, H, fs, bs;
        float lb;
      } p{T, H, C, FGB, c_.gate_lb};
      E.pipe("glm5_kda_gate");
      E.set(0, P3_);
      E.set(1, P5_, size_t(2 * c_.kD) * 4);
      E.set(2, w.a_rate);
      E.set(3, w.dt_bias);
      E.set(4, P3_);  // g in place over f (each thread reads its f before writing its g)
      E.set(5, P6_);
      E.bytes(6, p);
      E.threads(size_t(C), size_t(T), 1, 256);
    }
    {
      // the delta rule: chunked (8 tokens a step) from 16 rows, token by token below; ESPETERO_GLM5_KDA=recur|chunk forces one
      static const char* kv = std::getenv("ESPETERO_GLM5_KDA");
      const bool chunk = kv ? std::string(kv) == "chunk" : T >= 16;
      const int32_t p[4] = {T, H, C3, 1};
      E.pipe(chunk ? "glm5_kda_chunk" : "glm5_kda_recur");
      E.set(0, P2_);
      E.set(1, P3_);
      E.set(2, P6_);
      E.set(3, S_->kda[size_t(l)]);
      E.set(4, P1_);  // o (the projections are no longer needed)
      E.bytes(5, p);
      if (chunk) E.threads(32, 16, size_t(H), 32, 4);
      else E.groups(16, size_t(H), 1, 256);
    }
    {
      struct {
        int rows, H, zs;
        float eps;
      } p{T, H, C, c_.eps};
      E.pipe("glm5_kda_out");
      E.set(0, P1_);
      E.set(1, P4_);
      E.set(2, w.o_norm);
      E.bytes(3, p);
      E.groups(size_t((T * H + 7) / 8), 1, 1, 256);
    }
    gemv(E, w.o, P1_, 0, C, Y_, 0, D, T);
  }
  struct Glm5Conv {
    int rows, C3, heads3, hk;
  };

  void dsa(Enc& E, const LayerW& w, int l, int T, int p0, const std::string& dump_dir) {
    const int D = c_.D, H = c_.H, QR = c_.QR, KVR = c_.KVR, IH = c_.IH, IHD = c_.IHD;
    // queries: qr = norm(q_a x), q = q_b qr; latents c = norm(kv_a x) into the slot's cache
    gemv(E, w.q_a, X_, 0, D, P4_, 0, QR, T);  // qr (P4_), q (P1_), latents (P5_)
    norm(E, P4_, 0, w.q_a_norm, T, 1, QR, QR);
    gemv(E, w.q_b, P4_, 0, QR, P1_, 0, H * c_.NOPE, T);
    gemv(E, w.kv_a, X_, 0, D, P5_, 0, KVR, T);
    norm(E, P5_, 0, w.kv_a_norm, T, 1, KVR, KVR);
    {
      struct {
        int rows, n, src_stride, base, mod, src_row0;
      } p{T, KVR, KVR, p0, 0, 0};
      E.pipe("ds4_store_half");
      E.set(0, P5_);
      E.set(1, S_->lat[size_t(l)]);
      E.bytes(2, p);
      E.threads(size_t(KVR), size_t(T), 1, 256);
    }
    // indexer keys and pool gates, pooled keys of the pools this pass completes
    const int KG = w.ikg.N;  // 256: key | gate
    gemv(E, w.ikg, X_, 0, D, P6_, 0, KG, T);
    {
      struct {
        int rows, n, stride;
        float eps;
      } p{T, IHD, KG, 1e-6f};
      E.pipe("glm5_layernorm");
      E.set(0, P6_);
      E.set(1, w.ik_w);
      E.set(2, w.ik_b);
      E.bytes(3, p);
      E.groups(size_t((T + 7) / 8), 1, 1, 256);
    }
    const int b0 = p0 / 4, b1 = (p0 + T) / 4;  // pools completed: [b0, b1)
    if (b1 > b0) {
      struct {
        int p0, rows, ks, b0, npool;
      } p{p0, T, KG, b0, b1 - b0};
      E.pipe("glm5_kpool");
      E.set(0, P6_);
      E.set(1, S_->pend[size_t(l)]);
      E.set(2, w.ape);
      E.set(3, S_->pooled[size_t(l)]);
      E.bytes(4, p);
      E.threads(128, size_t(b1 - b0), 1, 128);
    }
    {
      const int32_t p[3] = {p0, T, KG};
      E.pipe("glm5_kpool_pend");
      E.set(0, P6_);
      E.set(1, S_->pend[size_t(l)]);
      E.bytes(2, p);
      E.threads(256, 3, 1, 256);
    }
    // the selection: every visible token while a row sees <= 2051 (exact, docs/glm5.md 4.2), else top-512 pools + tail
    const int width = c_.ITOPK + c_.KPOOL - 1;
    const bool sparse = (p0 + T) / 4 > c_.ITOPK / c_.KPOOL;
    if (sparse) {
      const int ncols = (p0 + T) / 4;
      const size_t need = size_t(T) * size_t(ncols) * 4 + size_t(T) * (512 + width) * 4;
      if (!SC_ || sc_bytes_ < need) SC_ = rt_.buffer(need), sc_bytes_ = need;
      gemv(E, w.iq, P4_, 0, QR, P2_, 0, IH * IHD, T);  // indexer queries
      gemv(E, w.iw, X_, 0, D, P3_, 0, IH, T);         // head weights
      {
        const float s = 1.0f / std::sqrt(float(IHD)) / std::sqrt(float(IH));
        const float q[2] = {float(T * IH), s};
        E.pipe("ds4_scale");
        E.set(0, P3_);
        E.bytes(1, q);
        E.threads(size_t(T * IH), 1, 1, 256);
      }
      {
        struct {
          int M, nh, hd, ratio, ncols;
        } p{T, IH, IHD, c_.KPOOL, ncols};
        E.pipe("ds4_index_score");
        E.set(0, P2_);
        E.set(1, P3_);
        E.set(2, S_->pooled[size_t(l)]);
        E.set(3, SC_);
        E.set(4, pos_);
        E.bytes(5, p);
        E.groups(size_t((ncols + 63) / 64), size_t(T), 1, 64);
      }
      const size_t pools_off = size_t(T) * size_t(ncols) * 4, sel_off = pools_off + size_t(T) * 512 * 4;
      {
        struct {
          int ncols, k, stride, ratio;
        } p{ncols, c_.ITOPK / c_.KPOOL, ncols, c_.KPOOL};
        E.pipe("ds4_topk");
        E.set(0, SC_);
        E.set(1, SC_, pools_off);
        E.set(2, pos_);
        E.bytes(3, p);
        E.groups(size_t(T), 1, 1, 1024);
      }
      {
        struct {
          int rows, k, width;
        } p{T, c_.ITOPK / c_.KPOOL, width};
        E.pipe("glm5_expand_sel");
        E.set(0, SC_, pools_off);
        E.set(1, pos_);
        E.set(2, SC_, sel_off);
        E.bytes(3, p);
        E.threads(size_t(T), 1, 1, 64);
      }
      if (!dump_dir.empty()) {
        E.flush();
        write_file(dump_dir + "/sel" + std::to_string(l) + ".i32", static_cast<const uint8_t*>([SC_ contents]) + sel_off,
                   size_t(T) * width * 4);
      }
    } else if (!dump_dir.empty()) {
      std::vector<int32_t> sel(size_t(T) * width, -1);
      for (int r = 0; r < T; ++r)
        for (int j = 0; j <= p0 + r; ++j) sel[size_t(r) * width + size_t(j)] = j;
      write_file(dump_dir + "/sel" + std::to_string(l) + ".i32", sel.data(), sel.size() * 4);
    }
    // absorbed queries [T][H][512], latent attention, the heads' outputs, o_proj
    if (gguf_) {
      gemv(E, w.k_b, P1_, 0, H * c_.NOPE, P2_, 0, H * KVR, T, false, nil, 0, nil, 0, nil, 0, KVR, c_.NOPE);
    } else {
      struct {
        int rows, H, xs, ys;
      } p{T, H, H * c_.NOPE, H * KVR};
      E.pipe("glm5_absorb");
      E.set(0, w.kv_b);
      E.set(1, P1_);
      E.set(2, P2_);
      E.bytes(3, p);
      E.groups(size_t(KVR / 8), size_t(H), size_t(T), 256);
    }
    {
      struct {
        int M, H, win, p0, ratio, topk, blockn;
        float scale;
      } p{T, H, 0, p0, 1, sparse ? width : 0, 0, 1.0f / std::sqrt(float(c_.NOPE))};
      const size_t sel_off = sparse ? size_t(T) * size_t((p0 + T) / 4) * 4 + size_t(T) * 512 * 4 : 0;
      E.pipe("ds4_attn");
      E.set(0, P2_);
      E.set(1, P5_);  // window rows (unused: win 0)
      E.set(2, S_->lat[size_t(l)]);
      E.set(3, S_->lat[size_t(l)]);
      E.set(4, sparse ? SC_ : zeros_, sel_off);
      E.set(5, ninf_);  // no sink
      E.set(6, P2_);
      E.set(7, pos_);
      E.bytes(8, p);
      E.groups(size_t((H + 7) / 8), size_t(T), 1, 256);
    }
    if (gguf_) {
      gemv(E, w.v_b, P2_, 0, H * KVR, P1_, 0, H * c_.VD, T, false, nil, 0, nil, 0, nil, 0, c_.VD, KVR);
    } else {
      struct {
        int rows, H, xs, ys;
      } p{T, H, H * KVR, H * c_.VD};
      E.pipe("glm5_unabsorb");
      E.set(0, w.kv_b);
      E.set(1, P2_);
      E.set(2, P1_);
      E.bytes(3, p);
      E.groups(size_t(c_.VD / 8), size_t(H), size_t(T), 256);
    }
    gemv(E, w.o_dsa, P1_, 0, H * c_.VD, Y_, 0, D, T);
  }

  void mlp(Enc& E, const Mat& g, const Mat& u, const Mat& d, id<MTLBuffer> X, int T, bool accum) {
    gemv(E, g, X, 0, c_.D, P1_, 0, g.N, T);
    gemv(E, u, X, 0, c_.D, P2_, 0, u.N, T);
    swiglu(E, P1_, P2_, P1_, T, g.N);
    gemv(E, d, P1_, 0, g.N, Y_, 0, c_.D, T, accum);
  }

  void moe(Enc& E, const LayerW& w, int l, int T, const std::string& dump_dir) {
    const int D = c_.D, K = c_.K;
    gemv(E, w.router, X_, 0, D, RL_, 0, c_.E, T);
    {
      struct {
        int M, E, k;
        float rs;
      } p{T, c_.E, K, c_.route_scale};
      E.pipe("glm5_route");
      E.set(0, RL_);
      E.set(1, w.bias);
      E.set(2, RSEL_);
      E.set(3, RW_);
      E.bytes(4, p);
      E.groups(size_t(T), 1, 1, 32);
    }
    mlp(E, w.gate, w.up, w.down, X_, T, false);  // the shared expert writes Y_
    E.flush();
    const int32_t* sel = static_cast<const int32_t*>([RSEL_ contents]);
    const float* rw = static_cast<const float*>([RW_ contents]);
    if (!dump_dir.empty()) write_file(dump_dir + "/route" + std::to_string(l) + ".i32", sel, size_t(T) * K * 4);
    // per expert: its rows (xrow), their weights (ysc) in XR_: [rows of e0 | rows of e1 ...] then the weights
    std::vector<std::vector<std::pair<int, float>>> rows(size_t(c_.E));
    for (int i = 0; i < T * K; ++i) {
      if (sel[i] < 0 || sel[i] >= c_.E) fail("glm5: bad routing");
      rows[size_t(sel[i])].push_back({i / K, rw[i]});
    }
    int32_t* xr = static_cast<int32_t*>([XR_ contents]);
    float* ws = reinterpret_cast<float*>(xr + T * K);
    std::vector<int> need;
    int at = 0;
    std::vector<int> first(size_t(c_.E), 0);
    for (int e = 0; e < c_.E; ++e) {
      const int n = int(rows[size_t(e)].size());
      if (!n) continue;
      need.push_back(e);
      first[size_t(e)] = at;
      for (int j = 0; j < n; ++j) xr[at + j] = rows[size_t(e)][size_t(j)].first, ws[at + j] = rows[size_t(e)][size_t(j)].second;
      at += n;
    }
    // the experts in groups the cache holds at once; each group's reads wait for the GPU to finish the previous one
    const size_t G = size_t(std::max(1, xc_.n));
    for (size_t g0 = 0; g0 < need.size(); g0 += G) {
      const std::vector<int> grp(need.begin() + long(g0), need.begin() + long(std::min(need.size(), g0 + G)));
      if (g0 > 0) E.flush();
      const std::vector<int> slots = ensure(w.xl, grp);
      for (size_t i = 0; i < grp.size(); ++i) {
        const int e = grp[i], n = int(rows[size_t(e)].size());
        const size_t ro = size_t(first[size_t(e)]) * 4, wo = size_t(T * K + first[size_t(e)]) * 4;
        gemv(E, xmat(w.xl, slots[i], 0), X_, 0, D, P1_, 0, c_.IE, n, false, XR_, ro);
        gemv(E, xmat(w.xl, slots[i], 1), X_, 0, D, P2_, 0, c_.IE, n, false, XR_, ro);
        swiglu(E, P1_, P2_, P1_, n, c_.IE);
        gemv(E, xmat(w.xl, slots[i], 2), P1_, 0, c_.IE, Y_, 0, D, n, true, nil, 0, XR_, ro, XR_, wo);
      }
    }
  }
};


// ---------------------------------------------------------------- the engine's model (NativeModel)
// Slots are sequences of the DevModel (user slots, then scratch slots for trees without one and for leaves); a
// pass runs its trees' segments one after another (one sequence per GPU pass for now). Trees, leaves, readouts
// and checkpoints follow DeepSeek-V4.1's model (dsv4_model.h).
class Glm5Model final : public NativeModel {
 public:
  Glm5Model(const std::string& dir, const LoadOptions& o, const std::string& kernels) {
    const auto t0 = Clock::now();
    max_ctx_ = std::max(256, o.max_ctx > 0 ? o.max_ctx : 32768);
    nuser_ = o.slots > 0 ? o.slots : 4;
    const char* cs = std::getenv("ESPETERO_GLM5_CACHE_SLOTS");
    const char* lay = std::getenv("ESPETERO_GLM5_LAYERS");  // development: the first n layers (no head)
    m_ = std::make_unique<DevModel>(dir, kernels, lay ? std::atoi(lay) : 0, max_ctx_, nuser_ + kScratch,
                                    std::max(64, std::min(o.max_rows > 0 ? o.max_rows : 4096, 8192)), o.expert_cache_gb,
                                    cs ? std::atoi(cs) : 0, std::max(1, std::min(o.max_readouts > 0 ? o.max_readouts : 64, 256)),
                                    nuser_);
    const Cfg& c = m_->cfg();
    if (!m_->has_head()) fail("glm5: no LM head after the layers loaded (a partial checkpoint needs ESPETERO_GLM5_HEAD_ANY=1 and the head)");
    info_.device = [[m_->rt().dev name] UTF8String];
    info_.format = "glm5_next";
    info_.path = dir;
    info_.hidden = c.D;
    info_.intermediate = c.IE;
    info_.layers = m_->layers();
    info_.full_layers = c.n_dsa();
    info_.linear_layers = c.n_kda();
    info_.heads = c.H;
    info_.kv_heads = 1;
    info_.vocab = c.V;
    info_.eos_token_id = c.eos.empty() ? 154820 : c.eos[0];
    info_.end_ids = c.eos.empty() ? std::vector<int>{154820, 154827, 154829} : c.eos;
    info_.slots = nuser_;
    info_.block_tokens = 64;
    info_.kv_blocks = nuser_ * max_ctx_ / 64;
    info_.max_ctx = max_ctx_;
    info_.max_rows = m_->max_rows();
    info_.max_readouts = m_->max_reads();
    info_.weight_bytes = m_->weight_bytes();
    info_.kv_bytes = m_->seq_bytes() * size_t(nuser_);
    info_.slot_state_bytes = m_->seq_bytes();
    info_.leaves = true;
    info_.prompt_lookup = false;  // verifying drafts as leaves forks whole states
    info_.commit = false;
    info_.hidden_states = false;
    info_.mtp = false;
    info_.vision = false;
    info_.load_ms = ms_since(t0);
    out_.clear();
  }

  const ModelInfo& info() const override { return info_; }

  // ---- slots
  int slot_alloc() override {
    for (int i = 0; i < nuser_; ++i)
      if (!m_->seq(i).used) {
        m_->seq(i).used = true;
        m_->reset(m_->seq(i));
        return i;
      }
    return -1;
  }
  void slot_free(int s) override {
    Seq& q = user(s);
    q.used = false;
    m_->reset(q);
  }
  void slot_reset(int s) override { m_->reset(user(s)); }
  void slot_fork(int dst, int src) override { m_->fork(user(dst), user(src)); }
  int slot_len(int s) const override { return m_->seq(s).len; }
  int slot_next_pos(int s) const override { return m_->seq(s).len; }
  bool slot_mtp(int) const override { return false; }
  int free_slots() const override {
    int n = 0;
    for (int i = 0; i < nuser_; ++i) n += m_->seq(i).used ? 0 : 1;
    return n;
  }
  int free_kv_tokens() const override {
    int n = 0;
    for (int i = 0; i < nuser_; ++i) n += max_ctx_ - (m_->seq(i).used ? m_->seq(i).len : 0);
    return n;
  }
  int pass_kv_tokens(const std::vector<TreeSpec>&) const override { return 0; }
  int freed_kv_tokens(const std::vector<int>& slots) const override {
    int n = 0;
    for (int s : slots) n += m_->seq(s).used ? m_->seq(s).len : 0;
    return n;
  }
  struct CkptHead {
    char magic[8];
    int32_t version, len, layers, max_ctx;
    uint64_t bytes;
  };
  bool slot_save(int slot, const std::string& path) override {
    const Seq& q = m_->seq(slot);
    const auto parts = m_->state_parts(q, q.len);
    CkptHead h{};
    std::memcpy(h.magic, "ESPGLM5C", 8);
    h.version = 1, h.len = q.len, h.layers = m_->layers(), h.max_ctx = max_ctx_;
    for (const auto& [b, n] : parts) h.bytes += n;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1;
    for (const auto& [b, n] : parts) ok = ok && std::fwrite([b contents], 1, n, f) == n;
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
    if (std::fread(&h, sizeof h, 1, f) != 1 || std::memcmp(h.magic, "ESPGLM5C", 8) != 0 || h.version != 1 ||
        h.layers != m_->layers() || h.len < 0 || h.len > max_ctx_)
      return -1;
    const int slot = slot_alloc();
    if (slot < 0) return -2;
    Seq& q = m_->seq(slot);
    const auto parts = m_->state_parts(q, h.len);
    uint64_t total = 0;
    for (const auto& [b, n] : parts) total += n;
    bool ok = total == h.bytes;
    for (const auto& [b, n] : parts) ok = ok && std::fread([b contents], 1, n, f) == n;
    if (!ok) {
      slot_free(slot);
      return -1;
    }
    q.len = h.len;
    return slot;
  }
  std::string state_signature() const override {
    return "glm5_next-1|" + info_.path + "|" + std::to_string(max_ctx_) + "|" + std::to_string(m_->layers());
  }

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
  void run_chain(const std::vector<TreeSpec>& first, int steps, int32_t* out, PassStats* stats) override {
    @autoreleasepool {
      const auto t0 = Clock::now();
      const size_t n = first.size();
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
      }
      if (stats) {
        stats->wall_ms = stats->gpu_ms = ms_since(t0);
        stats->rows = int(n) * steps;
        stats->readouts = int(n) * steps;
      }
    }
  }
  const float* logits(int tree, int readout) const override { return out_.at(size_t(tree)).at(size_t(readout)).logits.data(); }
  TopK topk(int tree, int readout) const override {
    TopK r;
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
  void draft(const std::vector<DraftRequest>&, PassStats*) override { fail("glm5: no MTP drafts yet"); }
  void commit(const std::vector<CommitRequest>&, PassStats*) override { fail("glm5: no verify-commit"); }
  void encode_images(const std::vector<ImageRef>&, float*, PassStats*) override { fail("glm5: text only (no vision tower yet)"); }
  void set_profile(bool) override {}
  std::string profile_json() override { return "{\"expert_cache\":" + m_->cache_json() + "}"; }
  bool set_option(const std::string&, int) override { return false; }
  std::string bench_json(const std::string&) override { return profile_json(); }
  void set_activation_sink(ActivationSink* sink) override {
    if (sink) fail("calibration taps: not supported for GLM-5");
  }

 private:
  static constexpr int kScratch = 4;  // trees without a slot, and leaves (in rounds)
  struct Readout {
    std::vector<float> logits;
    int32_t argmax = -1, sampled = -1;
    float lse = 0;
    std::vector<int32_t> tk_ids;
    std::vector<float> tk_x;
    float tk_max = 0, tk_total = 0;
  };
  std::unique_ptr<DevModel> m_;
  ModelInfo info_;
  int max_ctx_ = 32768, nuser_ = 4;
  std::vector<std::vector<Readout>> out_;  // [tree][readout]: leaves first, then the root

  Seq& user(int s) {
    if (s < 0 || s >= nuser_) fail("glm5: bad slot " + std::to_string(s));
    return m_->seq(s);
  }
  Seq& any(int s) { return m_->seq(s); }

  // One segment: n tokens continuing q, the last row's logits into `logits` (null: none).
  void segment(Seq& q, const int32_t* toks, int n, float* logits) { m_->forward(q, toks, n, logits, 1); }

  void run_trees(const std::vector<TreeSpec>& trees) {
    out_.assign(trees.size(), {});
    int scratch = nuser_;
    auto take_scratch = [&]() -> Seq& {
      if (scratch >= m_->nseq()) fail("glm5: too many trees without a slot in one pass");
      return m_->seq(scratch++);
    };
    std::vector<Seq*> base(trees.size(), nullptr);
    for (size_t i = 0; i < trees.size(); ++i) {
      const TreeSpec& t = trees[i];
      if (t.commit_chain > 0) fail("glm5: no verify-commit");
      if (t.n_images > 0) fail("glm5: text only (no vision tower yet)");
      if (t.out_hidden || t.out_last_hidden || t.out_rows_hidden) fail("glm5: hidden-state readouts are not supported");
      if (t.root_len <= 0 && t.n_leaves <= 0) fail("glm5: empty tree");
      out_[i].resize(size_t(std::max(0, t.n_leaves)) + (t.score_root ? 1 : 0));
      if (t.root_len <= 0) {  // leaves straight off the parent
        if (t.parent < 0) fail("glm5: leaves without a root need a parent");
        base[i] = &any(t.parent);
        continue;
      }
      const int save_len = t.save_len > 0 ? std::min(t.save_len, t.root_len) : t.root_len;
      Seq& tgt = t.save >= 0 ? user(t.save) : t.work >= 0 ? user(t.work) : take_scratch();
      if (t.parent >= 0) m_->fork(tgt, any(t.parent));
      else m_->reset(tgt);
      const int first = t.save >= 0 ? save_len : t.root_len;
      float* lg = nullptr;
      if (t.score_root && first == t.root_len) {
        out_[i].back().logits.resize(size_t(m_->cfg().V));
        lg = out_[i].back().logits.data();
      }
      segment(tgt, t.root, first, lg);
      base[i] = &tgt;
      if (first < t.root_len) {  // the checkpoint at save_len is taken: the rest continues in the work slot
        Seq& rest = (t.work >= 0 && &user(t.work) != &tgt) ? user(t.work) : take_scratch();
        m_->fork(rest, tgt);
        float* lg2 = nullptr;
        if (t.score_root) {
          out_[i].back().logits.resize(size_t(m_->cfg().V));
          lg2 = out_[i].back().logits.data();
        }
        segment(rest, t.root + first, t.root_len - first, lg2);
        base[i] = &rest;
      } else if (t.work >= 0 && &user(t.work) != &tgt) {
        m_->fork(user(t.work), tgt);
      }
    }
    // leaves: each continues its tree's root in a scratch slot, never seeing the others
    for (size_t i = 0; i < trees.size(); ++i) {
      const TreeSpec& t = trees[i];
      for (int k = 0; k < t.n_leaves; ++k) {
        const int n = t.leaf_offsets[k + 1] - t.leaf_offsets[k];
        if (n <= 0) fail("glm5: empty leaf");
        Seq& s = scratch < m_->nseq() ? m_->seq(scratch) : take_scratch();
        m_->fork(s, *base[i]);
        Readout& r = out_[i][size_t(k)];
        r.logits.resize(size_t(m_->cfg().V));
        segment(s, t.leaf_tokens + t.leaf_offsets[k], n, r.logits.data());
      }
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
  int32_t draw_token(const TreeSpec& t, const float* L, int64_t pos) {
    const int V = m_->cfg().V;
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
  // Readout ri of tree t (leaves first, then the root): DeepSeek-V4.1's (dsv4_model.h finish_readout).
  void finish_readout(const TreeSpec& t, Readout& o, int ri = 0) {
    const bool leaf = ri < t.n_leaves;
    const int64_t pos = t.draw_pos0 + (leaf ? int64_t(t.leaf_offsets[ri + 1] - t.leaf_offsets[ri]) : 0);
    const int V = m_->cfg().V;
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

}  // namespace espetero::native::glm5
