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

#include "glm5.h"

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
  void threads(size_t x, size_t y, size_t z, size_t tx) {
    if (x && y && z) [e dispatchThreads:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(tx, 1, 1)];
  }
};

enum Fmt : int { kBF16 = 2, kF32 = 3, kF8B = 4 };  // ds4_gemv formats
struct Mat {
  int fmt = kBF16, N = 0, K = 0;
  Buf w, s;
};

struct LayerW {
  bool dsa = false, moe = false;
  Buf attn_norm, ffn_norm, hca_fn, hca_base, hca_scale, hcf_fn, hcf_base, hcf_scale;
  // KDA
  Mat qkv, fgb, f_b, g_b, o;  // fgb: [f_a 128 | g_a 128 | b 64] rows
  Buf conv_w, a_log, dt_bias, o_norm;
  // DSA
  Mat q_a, q_b, kv_a, o_dsa, ikg, iw, iq;  // ikg: [wk 128 | compress_gate 128] rows; iw: weights_proj; iq: wq_b
  Buf q_a_norm, kv_a_norm, kv_b, ik_w, ik_b, ape;
  // MLP
  Mat gate, up, down;           // dense, or the shared expert
  Mat router;                   // bf16 [E, D]
  Buf bias;                     // e_score_correction_bias
  std::vector<Mat> xg, xu, xd;  // routed experts (development: all of a loaded layer's experts in memory)
};

class DevModel {
 public:
  DevModel(const std::string& dir, const std::string& kernels, int layers, int max_ctx) {
    idx_.open(dir, true);
    c_ = idx_.c;
    L_ = std::min(layers > 0 ? layers : c_.L, idx_.layers);
    if (L_ <= 0) fail("glm5: no complete layer in " + dir);
    max_ctx_ = max_ctx;
    rt_.init(kernels);
    const auto t0 = Clock::now();
    load();
    std::fprintf(stderr, "glm5: %d layers loaded (%.2f GB) in %.1f s\n", L_, double(rt_.arena->total()) / 1e9,
                 ms_since(t0) / 1e3);
    alloc_seq();
  }

  int layers() const { return L_; }
  const Cfg& cfg() const { return c_; }

  // One pass over `ids` continuing the sequence; dumps per layer when dump_dir is set; returns the logits of
  // the last `nlog` rows when every layer is loaded and the head is in the checkpoint (else empty).
  std::vector<float> prefill(const std::vector<int32_t>& ids, const std::string& dump_dir, int nlog = 1) {
    const int T = int(ids.size()), D = c_.D, p0 = len_;
    if (T <= 0) return {};
    if (p0 + T > max_ctx_) fail("glm5: context full");
    ensure_scratch(T);
    for (int i = 0; i < T; ++i) {
      static_cast<int32_t*>([ids_ contents])[i] = ids[size_t(i)];
      static_cast<int32_t*>([pos_ contents])[i] = p0 + i;
    }
    Enc E(rt_);
    {
      const int32_t p[2] = {D, 4};
      E.pipe("ds4_embed");
      E.set(0, ids_);
      E.set(1, embed_);
      E.set(2, H_);
      E.bytes(3, p);
      E.threads(size_t(D), size_t(T), 1, 256);
    }
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
    std::vector<float> out;
    nlog = std::clamp(nlog, 1, T);
    if (L_ == c_.L && head_.b) {
      E.pipe("glm5_hc_mean");
      E.set(0, H_, size_t(T - nlog) * 4 * D * 4);
      E.set(1, X_);
      E.bytes(2, D);
      E.threads(size_t(D), size_t(nlog), 1, 256);
      norm(E, X_, 0, norm_, nlog, 1, D, D);
      gemv(E, Mat{kBF16, c_.V, D, head_, {}}, X_, 0, D, LOG_, 0, c_.V, nlog);
      E.end();
      out.assign(static_cast<const float*>([LOG_ contents]), static_cast<const float*>([LOG_ contents]) + size_t(nlog) * c_.V);
      if (!dump_dir.empty()) write_file(dump_dir + "/logits.f32", out.data(), out.size() * 4);
    } else {
      E.end();
    }
    len_ += T;
    gpu_ms_ += E.gpu_ms;
    return out;
  }
  double gpu_ms() const { return gpu_ms_; }

 private:
  Index idx_;
  Cfg c_;
  int L_ = 0, max_ctx_ = 4096, len_ = 0, T_cap_ = 0;
  Runtime rt_;
  std::vector<LayerW> L;
  Buf embed_, head_, norm_;
  double gpu_ms_ = 0;
  // sequence state (one slot for now)
  std::vector<id<MTLBuffer>> kda_state_, conv_state_;           // per layer (KDA)
  std::vector<id<MTLBuffer>> lat_, pooled_, pend_;              // per layer (DSA): latents (half), pooled keys (half), pending key|gate rows
  id<MTLBuffer> zeros_ = nil, ninf_ = nil, ident_ = nil, ones_ = nil;
  // pass scratch
  id<MTLBuffer> ids_ = nil, pos_ = nil, H_ = nil, X_ = nil, Y_ = nil, MIX_ = nil, HCP_ = nil, P1_ = nil, P2_ = nil, P3_ = nil,
                P4_ = nil, P5_ = nil, P6_ = nil, RL_ = nil, RSEL_ = nil, RW_ = nil, XR_ = nil, LOG_ = nil, SC_ = nil;
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
    const std::string top = "model.language_model.";
    embed_ = raw(top + "embed_tokens.weight");
    if (idx_.ck.has("lm_head.weight") && L_ == c_.L) head_ = raw("lm_head.weight");
    norm_ = vecf(top + "norm.weight");
    L.resize(size_t(L_));
    for (int l = 0; l < L_; ++l) {
      LayerW& w = L[size_t(l)];
      const std::string p = layer_prefix(l);
      w.dsa = c_.dsa(l), w.moe = c_.moe(l);
      w.attn_norm = vecf(p + "input_layernorm.weight");
      w.ffn_norm = vecf(p + "post_attention_layernorm.weight");
      w.hca_fn = vecf(p + "hc_attn_fn"), w.hca_base = vecf(p + "hc_attn_base"), w.hca_scale = vecf(p + "hc_attn_scale");
      w.hcf_fn = vecf(p + "hc_ffn_fn"), w.hcf_base = vecf(p + "hc_ffn_base"), w.hcf_scale = vecf(p + "hc_ffn_scale");
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
        w.a_log = vecf(a + "A_log"), w.dt_bias = vecf(a + "dt_bias"), w.o_norm = vecf(a + "o_norm.weight");
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
        for (int e = 0; e < c_.E; ++e) {
          const std::string x = m + "experts." + std::to_string(e) + ".";
          w.xg.push_back(f8(x + "gate_proj"));
          w.xu.push_back(f8(x + "up_proj"));
          w.xd.push_back(f8(x + "down_proj"));
        }
      }
    }
  }

  void alloc_seq() {
    auto zero = [&](size_t n) {
      id<MTLBuffer> b = rt_.buffer(n);
      std::memset([b contents], 0, n);
      return b;
    };
    const int C = c_.kH * c_.kD;
    for (int l = 0; l < L_; ++l) {
      if (!L[size_t(l)].dsa) {
        kda_state_.push_back(zero(size_t(c_.kH) * 128 * 128 * 4));
        conv_state_.push_back(zero(size_t(3) * 3 * C * 4));
        lat_.push_back(nil), pooled_.push_back(nil), pend_.push_back(nil);
      } else {
        kda_state_.push_back(nil), conv_state_.push_back(nil);
        lat_.push_back(zero(size_t(max_ctx_) * c_.KVR * 2));
        pooled_.push_back(zero(size_t(max_ctx_ / 4 + 1) * c_.IHD * 2));
        pend_.push_back(zero(size_t(4) * 256 * 4));
      }
    }
    const float ninf[64] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
                            -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
                            -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
                            -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
                            -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
                            -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
                            -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
                            -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    ninf_ = rt_.buffer(sizeof ninf);
    std::memcpy([ninf_ contents], ninf, sizeof ninf);
    zeros_ = zero(4096);
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
    LOG_ = F(size_t(T) * c_.V);
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
            id<MTLBuffer> ysc = nil, size_t yso = 0) {
    if (M <= 0) return;
    E.pipe(m.fmt == kF8B ? "ds4_gemv_f8b" : m.fmt == kBF16 ? "ds4_gemv_bf16" : "ds4_gemv_f32");
    GemvP p{m.N, m.K, M, xs, ys, 0, 0, accum ? 1 : 0, xrow ? 1 : 0, 8};
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
      E.set(1, conv_state_[size_t(l)]);
      E.set(2, w.conv_w);
      E.set(3, P2_);
      E.bytes(4, p);
      E.groups(size_t((3 * H + 7) / 8), size_t(T), 1, 256);
      const int32_t q[2] = {T, C3};
      E.pipe("glm5_conv_save");
      E.set(0, P1_);
      E.set(1, conv_state_[size_t(l)]);
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
      E.set(2, w.a_log);
      E.set(3, w.dt_bias);
      E.set(4, P3_);  // g in place over f (each thread reads its f before writing its g)
      E.set(5, P6_);
      E.bytes(6, p);
      E.threads(size_t(C), size_t(T), 1, 256);
    }
    {
      const int32_t p[4] = {T, H, C3, 1};
      E.pipe("glm5_kda_recur");
      E.set(0, P2_);
      E.set(1, P3_);
      E.set(2, P6_);
      E.set(3, kda_state_[size_t(l)]);
      E.set(4, P1_);  // o (the projections are no longer needed)
      E.bytes(5, p);
      E.groups(16, size_t(H), 1, 256);
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
      E.set(1, lat_[size_t(l)]);
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
      E.set(1, pend_[size_t(l)]);
      E.set(2, w.ape);
      E.set(3, pooled_[size_t(l)]);
      E.bytes(4, p);
      E.threads(128, size_t(b1 - b0), 1, 128);
    }
    {
      const int32_t p[3] = {p0, T, KG};
      E.pipe("glm5_kpool_pend");
      E.set(0, P6_);
      E.set(1, pend_[size_t(l)]);
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
        E.set(2, pooled_[size_t(l)]);
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
    {
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
      E.set(2, lat_[size_t(l)]);
      E.set(3, lat_[size_t(l)]);
      E.set(4, sparse ? SC_ : zeros_, sel_off);
      E.set(5, ninf_);  // no sink
      E.set(6, P2_);
      E.set(7, pos_);
      E.bytes(8, p);
      E.groups(size_t((H + 7) / 8), size_t(T), 1, 256);
    }
    {
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
    int at = 0;
    for (int e = 0; e < c_.E; ++e) {
      const int n = int(rows[size_t(e)].size());
      if (!n) continue;
      for (int j = 0; j < n; ++j) xr[at + j] = rows[size_t(e)][size_t(j)].first, ws[at + j] = rows[size_t(e)][size_t(j)].second;
      const size_t ro = size_t(at) * 4, wo = size_t(T * K + at) * 4;
      gemv(E, w.xg[size_t(e)], X_, 0, D, P1_, 0, c_.IE, n, false, XR_, ro);
      gemv(E, w.xu[size_t(e)], X_, 0, D, P2_, 0, c_.IE, n, false, XR_, ro);
      swiglu(E, P1_, P2_, P1_, n, c_.IE);
      gemv(E, w.xd[size_t(e)], P1_, 0, c_.IE, Y_, 0, D, n, true, nil, 0, XR_, ro, XR_, wo);
      at += n;
    }
  }
};

}  // namespace espetero::native::glm5
