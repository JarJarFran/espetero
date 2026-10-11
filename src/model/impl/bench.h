// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Development benchmarks that need the model's internals: dispatch-gap
// characterization (chains of GEMV dispatches in one command buffer against
// the same dispatch alone, under the serial and the concurrent encoder).
#pragma once

#include <functional>
#include <random>

#include "forest.h"

namespace espetero::native {

// JSON: per case, the isolated dispatch time and the per-dispatch time inside
// chains of `n` dispatches that are dependent (each accumulates into the same
// output) or independent (distinct outputs), encoded serially or concurrently.
// The difference between chained and isolated time is the cost the engine
// pays per dispatch boundary; how it scales with the kernel size tells fixed
// cost from drain/refill tails, and the concurrent independent chain shows
// what overlap could recover.
inline std::string gap_bench(Model& m) {
  const Config& c = m.cfg;
  struct Case {
    std::string name;
    const Mat* w;
    int n_cols;
  };
  std::vector<Case> cases;
  const Layer* lin = nullptr;
  for (const Layer& L : m.layers)
    if (L.linear && !L.in_proj.empty() && !L.gate_up.dense && L.gate_up.out() > 0 && !lin) lin = &L;
  if (lin) {
    if (!lin->out.dense) cases.push_back({"out_proj", &lin->out, 0});
    if (!lin->in_proj[0].dense) cases.push_back({"in_proj0", &lin->in_proj[0], 0});
    cases.push_back({"gate_up", &lin->gate_up, 0});
  }
  if (!m.lm_head.dense)
    for (int nc : {512, 2048, 8192, 32768, 0}) cases.push_back({"lm_head/" + std::to_string(nc ? nc : c.V), &m.lm_head, nc});
  int K = 64, N = 64;
  for (const Case& cs : cases) {
    K = std::max(K, cs.w->in());
    N = std::max(N, cs.w->out());
  }
  constexpr int CHAIN = 32, TINY = 64;
  id<MTLBuffer> A = m.buffer(size_t(K) * 2);
  std::memset([A contents], 0, size_t(K) * 2);
  id<MTLBuffer> Y = m.buffer(size_t(N) * 4 * CHAIN);
  id<MTLBuffer> P = m.buffer(size_t(c.D) * 4), Q = m.buffer(size_t(c.D) * 4), rows = m.buffer(16);
  std::memset([rows contents], 0, 16);
  const bool serial0 = m.serial, ball0 = m.barrier_all;

  auto chain = [&](const Case& cs, int n, bool serial, bool independent) {
    m.serial = serial;
    m.barrier_all = false;
    id<MTLCommandBuffer> cb = [m.queue commandBufferWithUnretainedReferences];
    Enc E(m, cb);
    for (int i = 0; i < n; ++i) {
      const size_t yoff = independent ? size_t(i) * size_t(N) * 4 : 0;
      gemm(E, m.tune, *cs.w, A, 0, cs.w->in(), Y, yoff, cs.w->out(), 1, independent ? 0 : 1, cs.n_cols);
    }
    cb = E.finish();
    commit_wait(cb);
    return gpu_ms(cb);
  };
  auto tiny = [&](int n, bool serial) {
    m.serial = serial;
    m.barrier_all = false;
    id<MTLCommandBuffer> cb = [m.queue commandBufferWithUnretainedReferences];
    Enc E(m, cb);
    for (int i = 0; i < n; ++i) {
      id<MTLBuffer> src = (i % 2) ? Q : P, dst = (i % 2) ? P : Q;
      E.pipe("gather_rows_f32");
      E.set(0, src);
      E.set(1, dst);
      E.set(2, rows);
      E.bytes(3, c.D);
      E.rd(src);
      E.wr(dst);
      E.threads(MTLSizeMake(size_t(c.D / 4), 1, 1), MTLSizeMake(64, 1, 1));
    }
    cb = E.finish();
    commit_wait(cb);
    return gpu_ms(cb);
  };
  auto best = [&](auto&& f, int reps) {
    double t = 1e30;
    for (int r = 0; r < reps; ++r) t = std::min(t, f());
    return t;
  };
  // Clock ramp: ~1 s of dependent work before timing.
  if (!cases.empty())
    for (int r = 0; r < 4; ++r) chain(cases.back(), CHAIN, true, false);

  std::string out = "{";
  bool first = true;
  for (const Case& cs : cases) {
    const int n = cs.n_cols ? std::min(cs.n_cols, cs.w->out()) : cs.w->out();
    const int tg = (n + 31) / 32;
    const double iso = best([&] { return chain(cs, 1, true, false); }, 5);
    const double ser_dep = best([&] { return chain(cs, CHAIN, true, false); }, 3) / CHAIN;
    const double ser_ind = best([&] { return chain(cs, CHAIN, true, true); }, 3) / CHAIN;
    const double con_dep = best([&] { return chain(cs, CHAIN, false, false); }, 3) / CHAIN;
    const double con_ind = best([&] { return chain(cs, CHAIN, false, true); }, 3) / CHAIN;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "%s\"%s\":{\"threadgroups\":%d,\"iso_ms\":%.4f,\"serial_dep_ms\":%.4f,\"serial_indep_ms\":%.4f,"
                  "\"conc_dep_ms\":%.4f,\"conc_indep_ms\":%.4f}",
                  first ? "" : ",", cs.name.c_str(), tg, iso, ser_dep, ser_ind, con_dep, con_ind);
    out += buf;
    first = false;
  }
  {
    const double iso = best([&] { return tiny(1, true); }, 5);
    const double ser = best([&] { return tiny(TINY, true); }, 3) / TINY;
    const double con = best([&] { return tiny(TINY, false); }, 3) / TINY;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s\"tiny_gather\":{\"threadgroups\":%d,\"iso_ms\":%.4f,\"serial_dep_ms\":%.4f,\"conc_dep_ms\":%.4f}",
                  first ? "" : ",", c.D / 4 / 64, iso, ser, con);
    out += buf;
  }
  out += "}";
  m.serial = serial0;
  m.barrier_all = ball0;
  return out;
}

// Verify-pass anatomy: for every kind of decode matrix, the time per
// dispatch inside dependent chains that rotate over the layers holding that
// kind (so the 48 MB system cache cannot serve the weights), for the
// production GEMVs at M = 1, 2, 3, 4, 8 and, when the library was built with
// ESPETERO_DIAG=1, diagnostic kernels at 3-4 rows: rgemv8 without MMAs, without
// weight loads, without dequantization, and FMA GEMVs for 3-4 rows
// (activations resident in registers as in the FMA GEMV, or loaded just in time).
// JSON per kind: shape, weight MB per dispatch, ms per dispatch per variant.
inline std::string gemv_micro_bench(Model& m) {
  struct Kind {
    std::string name;
    std::vector<const Mat*> mats;
  };
  std::vector<Kind> kinds;
  auto add = [&](const std::string& base, const Mat* w) {
    if (!w || w->dense || w->out() <= 0) return;
    const std::string name = base + "/" + std::to_string(w->q.out) + "x" + std::to_string(w->q.in) + "/" +
                             std::to_string(w->q.bits) + "b" + (w->q.scale_mode ? "m" + std::to_string(w->q.scale_mode) : "") +
                             "g" + std::to_string(w->q.group);
    for (Kind& k : kinds)
      if (k.name == name) {
        if (k.mats.size() < 16) k.mats.push_back(w);
        return;
      }
    kinds.push_back({name, {w}});
  };
  for (const Layer& L : m.layers) {
    const std::string pre = L.linear ? "lin." : "attn.";
    for (size_t r = 0; r < L.in_proj.size(); ++r) add(pre + "in" + std::to_string(r), &L.in_proj[r]);
    add(pre + "out", &L.out);
    add("gate_up", &L.gate_up);
    add("gate", &L.gate);
    add("up", &L.up);
    add("down", &L.down);
  }
  add("lm_head", &m.lm_head);
  int K = 64, N = 64;
  for (const Kind& k : kinds)
    for (const Mat* w : k.mats) K = std::max(K, w->in()), N = std::max(N, w->out());
  constexpr int MMAX = 8, CHAIN = 16;
  id<MTLBuffer> A = m.buffer(size_t(MMAX) * size_t(K) * 2);
  {
    auto* a = (uint16_t*)[A contents];
    for (size_t i = 0; i < size_t(MMAX) * size_t(K); ++i) a[i] = uint16_t(0x2000 + (i % 97));  // small positive halves
  }
  id<MTLBuffer> Y = m.buffer(size_t(MMAX) * size_t(N) * 4);
  struct Variant {
    const char* label;
    int M;
    const char* kernel;  // "" = production path
    int ks = 8;          // production 1-row GEMV: K-slices per threadgroup
  };
  const std::vector<Variant> variants = {
      {"m1", 1, ""}, {"m1_ks16", 1, "", 16}, {"m2", 2, ""}, {"m3", 3, ""}, {"m4", 4, ""}, {"m8", 8, ""},
      {"m4_nomma", 4, "rgemv8r%_d1_store"}, {"m4_noload", 4, "rgemv8r%_d2_store"}, {"m4_nodq", 4, "rgemv8r%_d3_store"},
      {"m3_fma", 3, "rgemv%_3_store"}, {"m4_fma", 4, "rgemv%_4_store"},
      {"m3_fmajit", 3, "rgemv%_3j_store"}, {"m4_fmajit", 4, "rgemv%_4j_store"},
  };
  const bool serial0 = m.serial, ball0 = m.barrier_all;
  const std::string diag0 = m.tune.diag_kernel;
  m.serial = true;
  m.barrier_all = false;
  const int ks0 = m.tune.gemv_ks, ksb0 = m.tune.ks16_blocks;
  auto chain = [&](const Kind& k, const Variant& v) {
    m.tune.diag_kernel = v.kernel;
    m.tune.gemv_ks = v.ks;
    m.tune.ks16_blocks = 1 << 30;  // every shape, to compare
    id<MTLCommandBuffer> cb = [m.queue commandBufferWithUnretainedReferences];
    Enc E(m, cb);
    for (int i = 0; i < CHAIN; ++i) {
      const Mat& w = *k.mats[size_t(i) % k.mats.size()];
      gemm(E, m.tune, w, A, 0, w.in(), Y, 0, w.out(), v.M, 0);
    }
    cb = E.finish();
    commit_wait(cb);
    m.tune.diag_kernel = diag0;
    m.tune.gemv_ks = ks0;
    m.tune.ks16_blocks = ksb0;
    return gpu_ms(cb) / CHAIN;
  };
  auto available = [&](const Kind& k, const Variant& v) {
    if (!*v.kernel) return true;
    std::string name = v.kernel;
    name.replace(name.find('%'), 1, std::to_string(k.mats[0]->q.bits));
    try {
      (void)m.kernel(name, k.mats[0]->q.fc());
      return true;
    } catch (const std::exception&) {
      return false;
    }
  };
  // Clock ramp: ~1 s of work before timing.
  if (!kinds.empty())
    for (int r = 0; r < 8; ++r) chain(kinds[0], variants[0]);
  std::string out = "{";
  bool first = true;
  for (const Kind& k : kinds) {
    const QW& q = k.mats[0]->q;
    const double mb = (double(q.out) * double(q.in) * double(q.bits) / 8.0 +
                       double(q.out) * double(q.s_row_bytes() + q.z_row_bytes())) / 1e6;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s\"%s\":{\"layers\":%zu,\"mb\":%.2f,\"ms\":{", first ? "" : ",", k.name.c_str(),
                  k.mats.size(), mb);
    out += buf;
    first = false;
    bool vfirst = true;
    for (const Variant& v : variants) {
      if (!available(k, v)) continue;
      double best = 1e30;
      for (int r = 0; r < 5; ++r) best = std::min(best, chain(k, v));
      std::snprintf(buf, sizeof(buf), "%s\"%s\":%.4f", vfirst ? "" : ",", v.label, best);
      out += buf;
      vfirst = false;
    }
    out += "}}";
  }
  out += "}";
  m.serial = serial0;
  m.barrier_all = ball0;
  return out;
}

// Dense fp16 GEMM tiles on the vision tower's shapes: each block
// matrix of the first vision block at M = 3888 rows (a 1 MP photo), the two
// tile configurations interleaved (16x4, 32x4, 16x4, ...) so drift and other
// GPU clients hit both; the minimum over repetitions per configuration.
inline std::string hgemm_micro_bench(Model& m) {
  if (m.vis.blocks.empty()) return "{\"error\":\"no vision tower\"}";
  const VBlock& B = m.vis.blocks[0];
  struct Case {
    const char* name;
    const HW* w;
    int epi;
  };
  const std::vector<Case> cases = {{"qkv", &B.qkv, 0}, {"proj", &B.proj, 1}, {"fc1", &B.fc1, 2}, {"fc2", &B.fc2, 1}};
  constexpr int M = 3888, REPS = 12;
  int K = 64, N = 64;
  for (const Case& c : cases) K = std::max(K, c.w->kp), N = std::max(N, c.w->out);
  id<MTLBuffer> A = m.buffer(size_t(M) * size_t(K) * 2);
  {
    auto* a = (uint16_t*)[A contents];
    for (size_t i = 0; i < size_t(M) * size_t(K); ++i) a[i] = uint16_t(0x2c00 + (i % 61));
  }
  id<MTLBuffer> Y = m.buffer(size_t(M) * size_t(N) * 4);
  const std::string force0 = hgemm_force();
  const char* tiles[2] = {"16x4", "32x4"};
  std::string out = "{";
  bool first = true;
  for (const Case& c : cases) {
    double best[2] = {1e30, 1e30};
    for (int r = 0; r < REPS; ++r)
      for (int t = 0; t < 2; ++t) {
        hgemm_force() = tiles[t];
        id<MTLCommandBuffer> cb = [m.queue commandBufferWithUnretainedReferences];
        Enc E(m, cb);
        for (int i = 0; i < 4; ++i) gemm_h(E, *c.w, A, 0, c.w->kp, Y, 0, c.w->out, M, c.epi);
        cb = E.finish();
        commit_wait(cb);
        best[t] = std::min(best[t], gpu_ms(cb) / 4);
      }
    const double flop = 2.0 * M * double(c.w->out) * double(c.w->kp);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s\"%s\":{\"n\":%d,\"k\":%d,\"ms_16x4\":%.3f,\"ms_32x4\":%.3f,\"tflops_16x4\":%.2f,\"tflops_32x4\":%.2f}",
                  first ? "" : ",", c.name, c.w->out, c.w->kp, best[0], best[1], flop / best[0] / 1e9, flop / best[1] / 1e9);
    out += buf;
    first = false;
  }
  hgemm_force() = force0;
  return out + "}";
}

// Vision attention kernels alone: one layer's attention over a 1 MP
// photo (3,888 patches, the tower's heads and head size) with random q/k/v,
// the kernels interleaved (v1 32 rows, v2 64 rows, v3 64 rows x 64 keys) and
// the minimum over repetitions per kernel.
inline std::string vitattn_micro_bench(Model& m) {
  if (m.vis.blocks.empty()) return "{\"error\":\"no vision tower\"}";
  const VisionConfig& c = m.vis.c;
  const int N = 3888, H = c.heads, HD = c.hd;
  id<MTLBuffer> q = m.buffer(size_t(N) * H * HD * 2), k = m.buffer(size_t(N) * H * HD * 2), v = m.buffer(size_t(N) * H * HD * 2);
  id<MTLBuffer> o = m.buffer(size_t(N) * H * HD * 2), seg = m.buffer(size_t(N) * 8);
  {
    std::mt19937 g(3);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (id<MTLBuffer> b : {q, k, v}) {
      auto* x = (__fp16*)[b contents];
      for (size_t i = 0; i < size_t(N) * H * HD; ++i) x[i] = __fp16(u(g));
    }
    auto* sg = (int32_t*)[seg contents];
    for (int i = 0; i < N; ++i) sg[2 * i] = 0, sg[2 * i + 1] = N;
  }
  struct {
    int N, H;
  } at{N, H};
  // Variants 1-7 (vision.h's ESPETERO_VIT_ATTN); each writes its own output
  // buffer, compared byte for byte with variant 2's.
  constexpr int NVAR = 7;
  const char* names[NVAR] = {"vit_attention_", "vit_attention2_", "vit_attention3_", "vit_attention4_",
                             "vit_attention5_", "vit_attention6_", "vit_attention7_"};
  const int rows[NVAR] = {32, 64, 64, 64, 128, 128, 64};
  std::vector<id<MTLBuffer>> outs;
  for (int kk = 0; kk < NVAR; ++kk) outs.push_back(m.buffer(size_t(N) * H * HD * 2));
  (void)o;
  double best[NVAR];
  for (double& b : best) b = 1e30;
  for (int r = 0; r < 8; ++r)
    for (int kk = 0; kk < NVAR; ++kk) {
      id<MTLCommandBuffer> cb = [m.queue commandBufferWithUnretainedReferences];
      Enc E(m, cb);
      for (int i = 0; i < 3; ++i) {
        E.pipe(std::string(names[kk]) + std::to_string(HD));
        E.set(0, q);
        E.set(1, k);
        E.set(2, v);
        E.set(3, seg);
        E.set(4, outs[size_t(kk)]);
        E.bytes(5, at);
        E.rd(q);
        E.rd(k);
        E.rd(v);
        E.wr(outs[size_t(kk)]);
        E.groups(MTLSizeMake(size_t(H), size_t(N + rows[kk] - 1) / size_t(rows[kk]), 1), MTLSizeMake(kk == 0 ? 128 : 256, 1, 1));
      }
      cb = E.finish();
      commit_wait(cb);
      best[kk] = std::min(best[kk], gpu_ms(cb) / 3);
    }
  const double macs = 2.0 * double(N) * N * H * HD;
  std::string s = "{\"hd\":" + std::to_string(HD);
  char buf[160];
  for (int kk = 0; kk < NVAR; ++kk) {
    const bool same = std::memcmp([outs[size_t(kk)] contents], [outs[1] contents], size_t(N) * H * HD * 2) == 0;
    std::snprintf(buf, sizeof(buf), ",\"v%d\":{\"ms\":%.2f,\"tmacs\":%.2f,\"same_as_v2\":%s}", kk + 1, best[kk],
                  macs / best[kk] / 1e9, same ? "true" : "false");
    s += buf;
  }
  return s + "}";
}

// MoE expert GEMV micro-benchmark (single row): the grouped FMA kernel over
// k routed experts + the shared expert, against the dense rgemv streaming the
// same expert rows (same bytes and K), and a diagnostic variant whose expert
// index comes from the threadgroup id (no tile / rowmap loads before the
// first weight load). Every dispatch of a chain runs on a different MoE layer
// (24 layers x 23 MB = 550 MB, far beyond the 48 MB system cache: a chain over
// one layer's experts measured 385 GB/s, i.e. the cache, not DRAM). Per case:
// per-dispatch time inside a chain of 24 dependent dispatches (serial
// encoder), the isolated time (best over 7 layers) and the implied bandwidth.
inline std::string moe_micro_bench(Model& m) {
  const Config& c = m.cfg;
  std::vector<const Layer*> Ls;
  for (const Layer& l : m.layers)
    if (l.moe) Ls.push_back(&l);
  if (Ls.empty()) return "{\"error\":\"no MoE layer\"}";
  std::vector<const Layer*> lins;
  for (const Layer& l : m.layers)
    if (l.linear && !l.in_proj.empty() && !l.in_proj[0].dense) lins.push_back(&l);
  const int k = c.k, nt = k + 1;   // routed pairs + the shared expert as tile k
  const int Ne_gu = 2 * c.Ie, Ne_dn = c.D;
  constexpr int CHAIN = 24;
  id<MTLBuffer> X = m.buffer(size_t(c.D) * 2);
  id<MTLBuffer> act = m.buffer(size_t(nt) * size_t(c.Ie) * 2);
  id<MTLBuffer> ys = m.buffer(size_t(nt) * size_t(c.D) * 4);
  id<MTLBuffer> tiles = m.buffer(size_t(nt) * 16), ntiles = m.buffer(64), rowmap = m.buffer(size_t(nt) * 4);
  {
    auto* x = (uint16_t*)[X contents];
    for (int i = 0; i < c.D; ++i) x[i] = uint16_t(0x3c00 + (i % 7));  // ~1.0
    auto* a = (uint16_t*)[act contents];
    for (size_t i = 0; i < size_t(nt) * size_t(c.Ie); ++i) a[i] = uint16_t(0x3800 + (i % 5));
    auto* t = (int32_t*)[tiles contents];
    auto* r = (int32_t*)[rowmap contents];
    for (int i = 0; i < nt; ++i) {
      t[4 * i] = i < k ? ((i * 31 + 5) & (c.E - 1)) : c.E;  // the DIAG kernel's expert map
      t[4 * i + 1] = i, t[4 * i + 2] = 1, t[4 * i + 3] = 0;
      r[i] = 0;
    }
    auto* n = (uint32_t*)[ntiles contents];
    n[0] = uint32_t(nt), n[1] = uint32_t(Ne_gu / 32), n[2] = 1u, n[3] = uint32_t(nt), n[4] = uint32_t(Ne_dn / 32), n[5] = 1u;
  }
  const bool serial0 = m.serial, ball0 = m.barrier_all;
  m.serial = true;
  m.barrier_all = false;
  // One grouped dispatch on layer L: gate/up (epi 2 into act) or down (epi 0 into ys).
  auto grouped = [&](Enc& E, const Layer& L, bool down, int ks, bool direct, bool diag, int nbpt, int swap = 0) {
    const QW& w = down ? L.experts_down : L.experts_gu;
    const QW& w2 = down ? L.down.q : L.gate_up.q;
    const int Ne = down ? Ne_dn : Ne_gu;
    const std::string name = "moe_gemv" + std::to_string(w.bits) + "_1" + (down ? "_store" : "_swiglu") +
                             (diag ? "_diag" : ks == 8 ? "" : "_k" + std::to_string(ks)) +
                             (!diag && w2.bits != w.bits ? "_s" + std::to_string(w2.bits) : "");
    E.pipe(name, w.fc());
    const MoeGemmParams gp{Ne, w.out, w.in, down ? c.Ie : c.D, down ? c.D : c.Ie, down ? 0 : 1, c.E, w2.out, direct ? 1 : 0, direct ? nbpt : 1, direct ? swap : 0};
    E.set(0, down ? act : X);
    E.set(1, w.wr);
    E.set(2, w.sb);
    E.set(3, w.wh ? w.wh : w.wr);
    E.set(4, down ? ys : act);
    E.bytes(5, gp);
    E.set(6, tiles);
    E.set(7, ntiles);
    E.set(8, rowmap);
    E.set(9, w2.wr);
    E.set(10, w2.sb);
    E.set(11, w2.wh ? w2.wh : w2.wr);
    E.set(12, X);
    E.set(13, rowmap);
    E.set(14, ys);
    E.set(15, ys);
    E.set(16, rowmap);
    E.rd(down ? act : X);
    E.rd(tiles);
    E.rd(rowmap);
    E.wr(down ? ys : act);
    if (direct) E.groups(swap ? MTLSizeMake(size_t(Ne / 32 / nbpt), size_t(nt), 1) : MTLSizeMake(size_t(nt), size_t(Ne / 32 / nbpt), 1), MTLSizeMake(size_t(32 * ks), 1, 1));
    else E.groups_indirect(ntiles, down ? 12 : 0, MTLSizeMake(size_t(32 * ks), 1, 1));
  };
  // The dense rgemv over the first nt experts' rows of the same matrix.
  auto dense = [&](Enc& E, const Layer& L, bool down) {
    if (down) gemm_q(E, m.tune, L.experts_down, act, 0, c.Ie, ys, 0, nt * c.D, 1, 0, nt * Ne_dn);
    else gemm_q(E, m.tune, L.experts_gu, X, 0, c.D, act, 0, nt * c.Ie, 1, 2, nt * Ne_gu);
  };
  using Fn = std::function<void(Enc&, int)>;  // i: dispatch index (selects the layer)
  auto run = [&](const Fn& enc, int n, int base) {
    id<MTLCommandBuffer> cb = [m.queue commandBufferWithUnretainedReferences];
    Enc E(m, cb);
    for (int i = 0; i < n; ++i) enc(E, base + i);
    cb = E.finish();
    commit_wait(cb);
    return gpu_ms(cb);
  };
  auto layer = [&](int i) -> const Layer& { return *Ls[size_t(i) % Ls.size()]; };
  // Clock ramp.
  for (int r = 0; r < 6; ++r)
    run([&](Enc& E, int i) { grouped(E, layer(i), false, 8, true, false, 1); grouped(E, layer(i), true, 8, true, false, 1); }, CHAIN, 0);
  struct Case { std::string name; Fn enc; double mb; };
  auto mbytes = [&](const QW& w, const QW& w2, int Ne) {
    const double bpw = double(w.bits) / 8.0 + 4.0 / double(w.group);  // codes + scale/bias per group
    return (double(k) * double(Ne) * double(w.in) + double(w2.out) * double(w2.in)) * bpw / 1e6;
  };
  const double mb_gu = mbytes(Ls[0]->experts_gu, Ls[0]->gate_up.q, Ne_gu), mb_dn = mbytes(Ls[0]->experts_down, Ls[0]->down.q, Ne_dn);
  std::vector<Case> cases;
  cases.push_back({"gu_k4", [&](Enc& E, int i) { grouped(E, layer(i), false, 4, true, false, 1, 0); }, mb_gu});
  cases.push_back({"dn_k4_b2", [&](Enc& E, int i) { grouped(E, layer(i), true, 4, true, false, 2, 0); }, mb_dn});
  cases.push_back({"gu_indirect", [&](Enc& E, int i) { grouped(E, layer(i), false, 8, false, false, 1); }, mb_gu});
  cases.push_back({"dn_indirect", [&](Enc& E, int i) { grouped(E, layer(i), true, 8, false, false, 1); }, mb_dn});
  cases.push_back({"gu_diag", [&](Enc& E, int i) { grouped(E, layer(i), false, 8, true, true, 1); }, mb_gu});
  cases.push_back({"dn_diag", [&](Enc& E, int i) { grouped(E, layer(i), true, 8, true, true, 1); }, mb_dn});
  cases.push_back({"gu_dense", [&](Enc& E, int i) { dense(E, layer(i), false); }, mb_gu});
  cases.push_back({"dn_dense", [&](Enc& E, int i) { dense(E, layer(i), true); }, mb_dn});
  if (!lins.empty()) {
    const Mat& w0 = lins[0]->in_proj[0];
    const double mb = double(w0.out()) * double(w0.in()) * (double(w0.q.bits) / 8.0 + 4.0 / double(w0.q.group)) / 1e6;
    id<MTLBuffer> Y = m.buffer(size_t(w0.out()) * 4);
    cases.push_back({"in_proj_dense", [&, Y](Enc& E, int i) {
                       const Mat& w = lins[size_t(i) % lins.size()]->in_proj[0];
                       gemm(E, m.tune, w, X, 0, c.D, Y, 0, w.out(), 1, 0);
                     }, mb});
  }
  // The whole MoE layer at one row through moe_mlp, per stage and per knob
  // (route_sync 2: partial GEMV + finalize dispatch; 0: last arriver).
  id<MTLBuffer> Hres = m.buffer(size_t(c.D) * 4);
  id<MTLBuffer> act_sh = m.buffer(size_t(std::max(c.I_shared, 64)) * 2);
  const double mb_layer = mb_gu + mb_dn;
  struct Knob { const char* name; int stages, sync, fin1, rig, nsl; };
  const Knob knobs[] = {{"layer_all_fin0", 15, 2, 0, 0, 0}, {"layer_all_fin1", 15, 2, 1, 0, 0}, {"layer_all_rig", 15, 2, 1, 1, 0},
                        {"layer_route_par", 17, 2, 1, 0, 0}, {"layer_route_fin0", 1, 2, 0, 0, 0}, {"layer_route_fin1", 1, 2, 1, 0, 0},
                        {"layer_route_fin1_noload", 1, 2 | 32, 1, 0, 0},
                        {"layer_route_gu_fin1", 3, 2, 1, 0, 0}, {"layer_route_gu_rig", 3, 2, 1, 1, 0}, {"layer_route_gu_rig_noload", 3, 2 | 32, 1, 1, 0},
                        {"layer_combine", 8, 2, 1, 0, 0}};
  const int stages0 = m.moe_stages, sync0 = m.route_sync, nsl0 = m.route_nsl;
  const bool fin0 = m.route_fin1, rig0 = m.route_in_gemv;
  for (const Knob& kb : knobs)
    cases.push_back({kb.name, [&, kb](Enc& E, int i) {
                       m.moe_stages = kb.stages, m.route_sync = kb.sync, m.route_fin1 = kb.fin1 != 0, m.route_in_gemv = kb.rig != 0;
                       m.route_nsl = kb.nsl;
                       moe_mlp(m, E, layer(i), X, act_sh, Hres, 1, nullptr);
                       m.moe_stages = stages0, m.route_sync = sync0, m.route_fin1 = fin0, m.route_in_gemv = rig0, m.route_nsl = nsl0;
                     }, kb.stages == 15 ? mb_layer : 0.0});
  // Rounds interleave the cases so clock and background load hit all alike; min per case.
  std::vector<double> iso(cases.size(), 1e30), ch(cases.size(), 1e30);
  for (int r = 0; r < 5; ++r)
    for (size_t ci = 0; ci < cases.size(); ++ci) {
      iso[ci] = std::min(iso[ci], run(cases[ci].enc, 1, r * 5 + int(ci)));
      ch[ci] = std::min(ch[ci], run(cases[ci].enc, CHAIN, r * 7 + int(ci)) / CHAIN);
    }
  std::string out = "{";
  bool first = true;
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    const Case& cs = cases[ci];
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s\"%s\":{\"chain_us\":%.1f,\"iso_us\":%.1f,\"mb\":%.2f,\"gbs\":%.0f}", first ? "" : ",",
                  cs.name.c_str(), ch[ci] * 1e3, iso[ci] * 1e3, cs.mb, cs.mb / ch[ci]);
    out += buf;
    first = false;
  }
  out += "}";
  m.serial = serial0;
  m.barrier_all = ball0;
  return out;
}

// Decode attention alone: one query row (a single sequence's decode step)
// over P cached keys, for each P in ESPETERO_MICRO_P (default 1024, 8192,
// 32768). One command buffer holds the attention (+ combine) of every full
// layer, each over its own KV region as in a decode step (so the 48 MB system
// cache does not serve the keys), planned by plan_tiles with the model's
// attn_split (ESPETERO_MICRO_TGS: its threadgroup target, default 256). Kernels:
// the 8-row MMA tiles, the GEMV-style kernel and the GQA-row kernel. Per case
// the minimum over interleaved repetitions: µs per layer, the attention alone
// and with the combine, the cached K/V bandwidth, and the output's max |diff|
// against the MMA kernel's.
inline std::string attn_micro_bench(Model& m) {
  const Config& c = m.cfg;
  if (c.gemma || c.q4 || c.n_full == 0) return "{\"error\":\"Qwen3.5-style full attention only\"}";
  const KVPool& K = m.kv;
  std::vector<int> Ps = {1024, 8192, 32768};
  if (const char* e = std::getenv("ESPETERO_MICRO_P")) {
    Ps.clear();
    for (const char* s = e; *s;) {
      Ps.push_back(std::atoi(s));
      while (*s && *s != ',') ++s;
      if (*s == ',') ++s;
    }
  }
  const int bt = K.bt, kvd = c.kv_dim(), qdim = c.H * c.HD, ldf = c.full_ld(), G = c.H / c.Hkv;
  if (G > 8) return "{\"error\":\"GQA groups above 8\"}";
  int Pmax = 0;
  for (int P : Ps) Pmax = std::max(Pmax, P);
  const int nb = (Pmax + bt - 1) / bt;
  if (nb > K.n_blocks) return "{\"error\":\"KV pool too small for P " + std::to_string(Pmax) + "\"}";
  const int L = c.n_full;
  // Keys, values, queries, gates: small values from a fixed table.
  std::vector<uint16_t> tab(4096);
  {
    std::mt19937 g(5);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& t : tab) {
      const __fp16 h = __fp16(u(g));
      std::memcpy(&t, &h, 2);
    }
  }
  {
    auto* kv = static_cast<uint16_t*>([K.buf contents]);
    for (int l = 0; l < L; ++l) {
      uint16_t* r = kv + K.layer_base(l);
      const size_t n = size_t(nb) * K.block_halves();
      for (size_t i = 0; i < n; ++i) r[i] = tab[(i * 7 + (i >> 12) + size_t(l)) & 4095];
    }
  }
  id<MTLBuffer> Q = m.buffer(size_t(qdim) * 2), Kb = m.buffer(size_t(kvd) * 2), Vb = m.buffer(size_t(kvd) * 2);
  id<MTLBuffer> X = m.buffer(size_t(ldf) * 4), blocks = m.buffer(size_t(nb) * 4), desc = m.buffer(16);
  for (auto [b, n] : {std::pair{Q, qdim}, std::pair{Kb, kvd}, std::pair{Vb, kvd}}) {
    auto* p = static_cast<uint16_t*>([b contents]);
    for (int i = 0; i < n; ++i) p[i] = tab[size_t(i * 13 + 1) & 4095];
  }
  {
    auto* x = static_cast<float*>([X contents]);
    for (int i = 0; i < ldf; ++i) x[i] = float(int(i % 17) - 8) * 0.125f;
    auto* bl = static_cast<int32_t*>([blocks contents]);
    for (int i = 0; i < nb; ++i) bl[i] = i;
    std::memset([desc contents], 0, 16);
  }
  struct Kern {
    const char* label;
    std::string name;
    int threads;
  };
  const int nsg = (G <= 4 ? 1 : 2) * 4;
  const std::vector<Kern> kerns = {{"mma", "flash_attention_g" + std::to_string(G), 64 * G},
                                   {"vec", "attn_decode_vec_g" + std::to_string(G), 32 * nsg},
                                   {"gqa", "attn_decode_gqa_g" + std::to_string(G), 32 * 4}};
  std::vector<id<MTLBuffer>> outs;
  for (size_t k = 0; k < kerns.size(); ++k) outs.push_back(m.buffer(size_t(qdim) * 2));
  const bool serial0 = m.serial, ball0 = m.barrier_all;
  m.serial = true;
  m.barrier_all = false;
  std::string out = "{\"G\":" + std::to_string(G) + ",\"Hkv\":" + std::to_string(c.Hkv) + ",\"layers\":" + std::to_string(L) +
                    ",\"split\":" + std::to_string(m.attn_split);
  // ESPETERO_MICRO_TGS: the planner's threadgroup target (default 256).
  const int tgs = std::getenv("ESPETERO_MICRO_TGS") ? std::atoi(std::getenv("ESPETERO_MICRO_TGS")) : 256;
  out += ",\"tgs\":" + std::to_string(tgs);
  for (int P : Ps) {
    AttnPlan A;
    A.trees.push_back({0, P, 0, 1});
    plan_tiles(A, c.Hkv, m.attn_split, nullptr, tgs);
    id<MTLBuffer> trees = m.buffer(sizeof(AttnTreeRec)), tiles = m.buffer(std::max<size_t>(A.tiles.size(), 1) * sizeof(AttnTileRec));
    id<MTLBuffer> comb = m.buffer(std::max<size_t>(A.combine.size(), 1) * sizeof(AttnCombineRec));
    id<MTLBuffer> parts = m.buffer(std::max<size_t>(size_t(A.parts) * size_t(c.Hkv) * 8 * size_t(G) * 258 * 4, 16));
    std::memcpy([trees contents], A.trees.data(), sizeof(AttnTreeRec));
    std::memcpy([tiles contents], A.tiles.data(), A.tiles.size() * sizeof(AttnTileRec));
    if (!A.combine.empty()) std::memcpy([comb contents], A.combine.data(), A.combine.size() * sizeof(AttnCombineRec));
    // stage: 1 attention, 2 combine, 3 both
    auto run = [&](size_t k, int stage) {
      id<MTLCommandBuffer> cb = [m.queue commandBufferWithUnretainedReferences];
      Enc E(m, cb);
      for (int l = 0; l < L; ++l) {
        const AttnParams ap{c.H, c.Hkv, qdim, kvd, ldf, bt, uint64_t(K.layer_base(l))};
        if (stage & 1) {
          E.pipe(kerns[k].name);
          E.set(0, Q);
          E.set(1, Kb);
          E.set(2, Vb);
          E.set(3, K.buf);
          E.set(4, blocks);
          E.set(5, desc);
          E.set(6, X);
          E.set(7, outs[k]);
          E.bytes(8, ap);
          E.set(9, tiles);
          E.set(10, trees);
          E.set(11, parts);
          E.groups(MTLSizeMake(size_t(c.Hkv), A.tiles.size(), 1), MTLSizeMake(size_t(kerns[k].threads), 1, 1));
        }
        if ((stage & 2) && !A.combine.empty()) {
          E.pipe("attention_combine");
          E.set(0, parts);
          E.set(1, X);
          E.set(2, outs[k]);
          E.bytes(3, ap);
          E.set(4, comb);
          E.threads(MTLSizeMake(64, size_t(c.H), A.combine.size() * 8), MTLSizeMake(64, 1, 1));
        }
      }
      cb = E.finish();
      commit_wait(cb);
      return gpu_ms(cb) * 1e3 / L;  // µs per layer
    };
    for (int r = 0; r < 3; ++r) run(0, 3);  // clock ramp
    std::vector<double> both(kerns.size(), 1e30), alone(kerns.size(), 1e30);
    double comb_us = 1e30;
    for (int r = 0; r < 7; ++r) {
      for (size_t k = 0; k < kerns.size(); ++k) {
        both[k] = std::min(both[k], run(k, 3));
        alone[k] = std::min(alone[k], run(k, 1));
      }
      if (!A.combine.empty()) comb_us = std::min(comb_us, run(0, 2));
    }
    const double mb = double(P) * double(kvd) * 4 / 1e6;  // K + V halves read per layer
    char buf[256];
    std::snprintf(buf, sizeof(buf), ",\"P%d\":{\"tiles\":%zu,\"parts\":%d,\"combine_us\":%.1f", P, A.tiles.size(), A.parts,
                  A.combine.empty() ? 0.0 : comb_us);
    out += buf;
    // Each kernel's output (the last layer's) against the MMA tiles'.
    for (size_t k = 0; k < kerns.size(); ++k) {
      run(k, 3);
      double md = 0;
      const auto* a = static_cast<const __fp16*>([outs[0] contents]);
      const auto* b = static_cast<const __fp16*>([outs[k] contents]);
      for (int i = 0; i < qdim; ++i) md = std::max(md, double(std::fabs(float(a[i]) - float(b[i]))));
      std::snprintf(buf, sizeof(buf), ",\"%s\":{\"us\":%.1f,\"attn_us\":%.1f,\"gbs\":%.0f,\"max_diff\":%.3g}", kerns[k].label,
                    both[k], alone[k], mb / alone[k] * 1e3, md);
      out += buf;
    }
    out += "}";
  }
  m.serial = serial0;
  m.barrier_all = ball0;
  return out + "}";
}

}  // namespace espetero::native
