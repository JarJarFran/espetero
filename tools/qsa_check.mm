// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The Qwen sparse attention kernels (kernels/45_qsa.metal) on synthetic
// data, one command buffer per kernel with a deadline: a kernel that does not
// finish is named (and leaves only this tool's few MB on the GPU), and the
// results are checked against CPU references (top-512 selection, tile union
// lists, sparse attention of some rows). No model needed.
//   clang++ -std=c++20 -O2 -fobjc-arc tools/qsa_check.mm -framework Metal -framework Foundation -o build/qsa_check
//   build/qsa_check [kernels dir]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dirent.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct AttnParams { int32_t H, Hkv, q_ld, kv_ld, g_ld, bt; uint64_t kv_base; };
struct AttnTree { int32_t base, P, table, M; };
struct AttnTile { int32_t tree, r0, k0, k1, part, last, pad0, pad1; };
struct AttnComb { int32_t r0, rows, part0, nparts; };
struct KVSave { int32_t src_row, block, pos, pad; };
struct QsaParams { int32_t ld, bt, iblk, pool_off, pos_off, nbmax, nbw; float eps; uint64_t ibase; };
struct QsaTile { int32_t tree, r0, rows, pad; };

constexpr int HD = 128, IH = 4, TOPK = 512, LMAX = 4112, BT = 64;
constexpr int IBLK = BT * (HD + HD / 4 + 8), POOL_OFF = BT * HD, POS_OFF = BT * (HD + HD / 4);
constexpr int H = 24, HKV = 2, KVD = HKV * 256, QD = H * 256, LDF = H * 512 + 2 * KVD;

id<MTLDevice> dev;
id<MTLCommandQueue> queue;
id<MTLLibrary> lib;
int failures = 0;

id<MTLBuffer> buf(size_t n) { return [dev newBufferWithLength:std::max<size_t>(n, 16) options:MTLResourceStorageModeShared]; }
template <class T> T* P(id<MTLBuffer> b) { return static_cast<T*>([b contents]); }

uint16_t f2h(float x) { __fp16 h = __fp16(x); uint16_t u; std::memcpy(&u, &h, 2); return u; }
float h2f(uint16_t u) { __fp16 h; std::memcpy(&h, &u, 2); return float(h); }
float bf16r(float x) {
  uint32_t u; std::memcpy(&u, &x, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  u &= 0xFFFF0000u;
  float y; std::memcpy(&y, &u, 4);
  return y;
}

// One dispatch in its own command buffer; false when it misses the deadline.
bool run(const char* name, MTLSize grid, MTLSize tg, bool threads, const std::vector<std::pair<int, id<MTLBuffer>>>& bufs,
         const std::vector<std::pair<int, std::vector<uint8_t>>>& bytes, double deadline_s = 20) {
  NSError* err = nil;
  id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
  if (!f) { std::printf("no kernel %s\n", name); std::exit(2); }
  id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:f error:&err];
  if (!ps) { std::printf("%s: %s\n", name, [[err localizedDescription] UTF8String]); std::exit(2); }
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:ps];
  for (auto& [i, b] : bufs) [e setBuffer:b offset:0 atIndex:i];
  for (auto& [i, v] : bytes) [e setBytes:v.data() length:v.size() atIndex:i];
  if (threads) [e dispatchThreads:grid threadsPerThreadgroup:tg];
  else [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  [e endEncoding];
  const auto t0 = std::chrono::steady_clock::now();
  [cb commit];
  while (true) {
    const auto st = [cb status];
    if (st == MTLCommandBufferStatusCompleted) break;
    if (st == MTLCommandBufferStatusError) {
      std::printf("  %-26s GPU ERROR: %s\n", name, [[[cb error] localizedDescription] UTF8String]);
      ++failures;
      return false;
    }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > deadline_s) {
      std::printf("  %-26s HANG (no completion after %.0f s)\n", name, deadline_s);
      std::fflush(stdout);
      std::_Exit(3);
    }
    usleep(200);
  }
  std::printf("  %-26s %8.3f ms\n", name, ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3);
  return true;
}
template <class T> std::vector<uint8_t> by(const T& v) {
  std::vector<uint8_t> o(sizeof(T));
  std::memcpy(o.data(), &v, sizeof(T));
  return o;
}

// One tree: P cached tokens, then M pass rows (a root chain of Tr rows, then
// leaves of `leaf` rows each hanging off the whole root).
void scenario(const char* title, int Pc, int Tr, int n_leaves, int leaf, uint32_t seed) {
  std::printf("%s: P %d, root %d rows, %d leaves x %d\n", title, Pc, Tr, n_leaves, leaf);
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int M = Tr + n_leaves * leaf;
  // rows: sequence index, desc
  std::vector<int32_t> desc(size_t(M) * 4, 0), sidx(M);
  for (int r = 0; r < Tr; ++r) sidx[r] = Pc + r;
  for (int l = 0; l < n_leaves; ++l)
    for (int j = 0; j < leaf; ++j) {
      const int r = Tr + l * leaf + j;
      desc[r * 4 + 0] = Tr + l * leaf;
      desc[r * 4 + 2] = Tr;
      sidx[r] = Pc + Tr + j;
    }
  const int maxs = *std::max_element(sidx.begin(), sidx.end());
  const int nblocks_kv = (maxs + BT) / BT + 2;
  // KV and index pools: block b of the tree = pool block (b * 7 + 3) % nblocks (shuffled ids)
  const int pool_blocks = nblocks_kv;
  std::vector<int32_t> table(nblocks_kv);
  for (int i = 0; i < nblocks_kv; ++i) table[i] = (i * 7 + 3) % pool_blocks;
  id<MTLBuffer> blocks = buf(table.size() * 4);
  std::memcpy([blocks contents], table.data(), table.size() * 4);
  id<MTLBuffer> ib = buf(size_t(pool_blocks) * IBLK * 2);
  id<MTLBuffer> kv = buf(size_t(pool_blocks) * 2 * BT * KVD * 2);
  {
    uint16_t* p = P<uint16_t>(ib);
    for (size_t i = 0; i < size_t(pool_blocks) * IBLK; ++i) p[i] = f2h(bf16r(nd(rng)));
    uint16_t* k = P<uint16_t>(kv);
    for (size_t i = 0; i < size_t(pool_blocks) * 2 * BT * KVD; ++i) k[i] = f2h(nd(rng));
    // positions of cached tokens: their sequence index
    for (int i = 0; i < Pc; ++i) {
      int32_t* q = reinterpret_cast<int32_t*>(p + size_t(table[i / BT]) * IBLK + POS_OFF + size_t(i % BT) * 8);
      q[0] = q[1] = q[2] = i;
      q[3] = 0;
    }
  }
  // pass inputs
  id<MTLBuffer> X = buf(size_t(M) * 640 * 4), qn = buf(HD * 4), kn = buf(HD * 4), pos = buf(size_t(M) * 16);
  id<MTLBuffer> invf = buf(32 * 4), sel_rope = buf(32 * 4);
  for (int i = 0; i < M * 640; ++i) P<float>(X)[i] = nd(rng) * 3.0f;
  for (int i = 0; i < HD; ++i) P<float>(qn)[i] = 1.0f + 0.1f * nd(rng), P<float>(kn)[i] = 1.0f + 0.1f * nd(rng);
  for (int r = 0; r < M; ++r) for (int d = 0; d < 3; ++d) P<int32_t>(pos)[r * 4 + d] = sidx[r], P<int32_t>(pos)[r * 4 + 3] = 0;
  for (int i = 0; i < 32; ++i) P<float>(invf)[i] = std::pow(1e7f, -2.0f * i / 64.0f), P<int32_t>(sel_rope)[i] = 0;
  id<MTLBuffer> desc_b = buf(desc.size() * 4);
  std::memcpy([desc_b contents], desc.data(), desc.size() * 4);
  AttnTree tree{0, Pc, 0, M};
  id<MTLBuffer> trees = buf(sizeof(AttnTree));
  std::memcpy([trees contents], &tree, sizeof(tree));
  // plan (as qsa_plan)
  std::vector<int32_t> rinfo(size_t(M) * 4), srows;
  std::vector<QsaTile> qtiles;
  std::vector<int> qt_of((M + 7) / 8, -1);
  int nbmax = 0, nbb = 0, ncb = 0;
  for (int r = 0; r < M; ++r) {
    const int s = sidx[r], nb = (s + 1) / 4;
    const bool sp = nb > 512;
    rinfo[r * 4 + 0] = s, rinfo[r * 4 + 1] = nb, rinfo[r * 4 + 2] = sp ? int(srows.size()) : -1, rinfo[r * 4 + 3] = 0;
    if (!sp) continue;
    srows.push_back(r);
    nbmax = std::max(nbmax, nb), nbb = std::max(nbb, nb - Pc / 4), ncb = std::max(ncb, Pc / 4);
    if (qt_of[r / 8] < 0) qt_of[r / 8] = int(qtiles.size()), qtiles.push_back({0, r / 8 * 8, std::min(8, M - r / 8 * 8), 0});
  }
  const int ns = int(srows.size()), nbw = (nbmax + 31) / 32;
  std::printf("  rows %d, sparse %d, tiles %zu, nbmax %d, nbb %d, ncb %d\n", M, ns, qtiles.size(), nbmax, nbb, ncb);
  id<MTLBuffer> rinfo_b = buf(rinfo.size() * 4), srows_b = buf(srows.size() * 4), qtiles_b = buf(qtiles.size() * sizeof(QsaTile));
  std::memcpy([rinfo_b contents], rinfo.data(), rinfo.size() * 4);
  if (ns) std::memcpy([srows_b contents], srows.data(), srows.size() * 4);
  if (!qtiles.empty()) std::memcpy([qtiles_b contents], qtiles.data(), qtiles.size() * sizeof(QsaTile));
  id<MTLBuffer> IQ = buf(size_t(M) * IH * HD * 2), IK = buf(size_t(M) * HD * 2), IP = buf(size_t(M) * HD * 2);
  id<MTLBuffer> sc = buf(size_t(std::max(ns, 1)) * std::max(nbmax, 1) * 4), sel = buf(size_t(std::max(ns, 1)) * std::max(nbw, 1) * 4);
  id<MTLBuffer> tl = buf(std::max<size_t>(qtiles.size(), 1) * LMAX * 4), tc = buf(std::max<size_t>(qtiles.size(), 1) * 4);
  const QsaParams qp{640, BT, IBLK, POOL_OFF, POS_OFF, std::max(nbmax, 1), std::max(nbw, 1), 1e-6f, 0};
  run("qsa_prep", MTLSizeMake(1, M, 1), MTLSizeMake(160, 1, 1), false,
      {{0, X}, {1, qn}, {2, pos}, {3, invf}, {4, sel_rope}, {5, IQ}, {6, IK}}, {{7, by(qp)}});
  run("qsa_pool", MTLSizeMake(1, M, 1), MTLSizeMake(32, 1, 1), false,
      {{0, IK}, {1, ib}, {2, blocks}, {3, trees}, {4, desc_b}, {5, rinfo_b}, {6, pos}, {7, kn}, {8, invf}, {9, sel_rope}, {10, IP}},
      {{11, by(qp)}});
  if (!ns) return;
  if (ncb > 0)
    run("qsa_score_cached", MTLSizeMake((ncb + 31) / 32, qtiles.size(), 1), MTLSizeMake(128, 1, 1), false,
        {{0, IQ}, {1, ib}, {2, blocks}, {3, trees}, {4, rinfo_b}, {5, qtiles_b}, {6, sc}}, {{7, by(qp)}});
  if (nbb > 0)
    run("qsa_score_batch", MTLSizeMake(nbb, M, 1), MTLSizeMake(32, 1, 1), false,
        {{0, IQ}, {1, IP}, {2, trees}, {3, desc_b}, {4, rinfo_b}, {5, sc}}, {{6, by(qp)}});
  // CPU check of the scores (a few rows): the pooled keys come from the pool or IP
  auto pooled = [&](int r, int b) -> const uint16_t* {
    if (4 * b + 3 < Pc) return P<uint16_t>(ib) + size_t(table[b / (BT / 4)]) * IBLK + POOL_OFF + size_t(b % (BT / 4)) * HD;
    const int i = 4 * b + 3, j = i - Pc;
    const int row = j < desc[r * 4 + 2] ? j : desc[r * 4 + 0] + (j - desc[r * 4 + 2]);
    return P<uint16_t>(IP) + size_t(row) * HD;
  };
  double max_sd = 0;
  for (int t = 0; t < std::min(ns, 6); ++t) {
    const int r = srows[size_t(t * std::max(1, ns / 6)) % size_t(ns)], sr = rinfo[r * 4 + 2], nb = rinfo[r * 4 + 1];
    for (int b = 0; b < nb; b += std::max(1, nb / 50)) {
      const uint16_t* k = pooled(r, b);
      float s = 0;
      for (int h = 0; h < IH; ++h) {
        float d = 0;
        for (int e = 0; e < HD; ++e) d += h2f(P<uint16_t>(IQ)[(size_t(r) * IH + h) * HD + e]) * h2f(k[e]);
        s += std::max(d, 0.0f);
      }
      max_sd = std::max(max_sd, double(std::fabs(s - P<float>(sc)[size_t(sr) * nbmax + b]) / (1 + std::fabs(s))));
    }
  }
  std::printf("  scores vs CPU: max rel diff %.2e %s\n", max_sd, max_sd < 1e-3 ? "ok" : "FAIL");
  if (max_sd >= 1e-3) ++failures;
  run("qsa_select", MTLSizeMake(ns, 1, 1), MTLSizeMake(1024, 1, 1), false, {{0, sc}, {1, srows_b}, {2, rinfo_b}, {3, sel}}, {{4, by(qp)}});
  // CPU top-512 (ties: lowest blocks) against the bitmap
  int bad_sel = 0;
  for (int sr = 0; sr < ns; ++sr) {
    const int r = srows[sr], nb = rinfo[r * 4 + 1];
    std::vector<std::pair<float, int>> v(nb);
    for (int b = 0; b < nb; ++b) v[b] = {P<float>(sc)[size_t(sr) * nbmax + b], b};
    std::stable_sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<char> want(nb, 0);
    for (int i = 0; i < TOPK; ++i) want[v[i].second] = 1;
    int cnt = 0;
    for (int b = 0; b < nbw * 32; ++b) {
      const bool got = (P<uint32_t>(sel)[size_t(sr) * nbw + b / 32] >> (b % 32)) & 1u;
      cnt += got;
      if (b < nb && got != bool(want[b])) ++bad_sel;
      if (b >= nb && got) ++bad_sel;
    }
    if (cnt != TOPK) ++bad_sel;
  }
  std::printf("  selection vs CPU top-512: %d mismatches %s\n", bad_sel, bad_sel ? "FAIL" : "ok");
  if (bad_sel) ++failures;
  run("qsa_tiles", MTLSizeMake(qtiles.size(), 1, 1), MTLSizeMake(256, 1, 1), false,
      {{0, sel}, {1, trees}, {2, rinfo_b}, {3, qtiles_b}, {4, tl}, {5, tc}}, {{6, by(qp)}});
  int bad_tl = 0;
  const int nbc = (Pc + 3) / 4;
  for (size_t q = 0; q < qtiles.size(); ++q) {
    std::vector<uint32_t> want;
    for (int b = 0; b < nbc; ++b) {
      uint32_t m8 = 0;
      for (int k = 0; k < qtiles[q].rows; ++k) {
        const int r = qtiles[q].r0 + k, sr = rinfo[r * 4 + 2], nb = rinfo[r * 4 + 1];
        bool on = sr < 0 ? true : (((P<uint32_t>(sel)[size_t(sr) * nbw + b / 32] >> (b % 32)) & 1u) || b == nb);
        if (on) m8 |= 1u << k;
      }
      if (m8) want.push_back(uint32_t(b) << 8 | m8);
    }
    const uint32_t n = P<uint32_t>(tc)[q];
    if (n != want.size()) ++bad_tl;
    else
      for (uint32_t i = 0; i < n; ++i) bad_tl += P<uint32_t>(tl)[q * LMAX + i] != want[i];
  }
  std::printf("  tile lists vs CPU: %d mismatches %s\n", bad_tl, bad_tl ? "FAIL" : "ok");
  if (bad_tl) ++failures;
  // Attention: tiles as plan_tiles (no split: budget 1; or the decode split)
  std::vector<AttnTile> tiles;
  std::vector<AttnComb> comb;
  int parts = 0;
  const int n_tiles = (M + 7) / 8, budget = std::max(1, 256 / std::max(1, HKV * n_tiles));
  for (int r = 0; r < M; r += 8) {
    const int q = qt_of[r / 8];
    if (q < 0) { tiles.push_back({0, r, 0, Pc, -1, 1, 0, 0}); continue; }
    const int rows = std::min(8, M - r), est = 516 + 128 * (rows - 1);
    int nq = 1;
    if (budget > 1) {
      const int chunk = std::max(16, std::min(512 / 4, ((est + budget - 1) / budget + 3) / 4 * 4));
      nq = std::min({(est + chunk - 1) / chunk, budget, 64});
    }
    if (nq == 1) { tiles.push_back({0, r, 0, 1 << 30, -1, 1, 1, q}); continue; }
    const int step = ((est + nq - 1) / nq + 3) / 4 * 4, part0 = parts;
    for (int k = 0; k < nq; ++k) {
      const bool lastp = k + 1 == nq;
      tiles.push_back({0, r, k * step, lastp ? (1 << 30) : (k + 1) * step, parts++, lastp ? 1 : 0, 1, q});
    }
    comb.push_back({r, rows, part0, nq});
  }
  std::printf("  attention tiles %zu, parts %d\n", tiles.size(), parts);
  id<MTLBuffer> Q = buf(size_t(M) * QD * 2), Kb = buf(size_t(M) * KVD * 2), Vb = buf(size_t(M) * KVD * 2);
  id<MTLBuffer> G = buf(size_t(M) * LDF * 4), O = buf(size_t(M) * QD * 2);
  for (size_t i = 0; i < size_t(M) * QD; ++i) P<uint16_t>(Q)[i] = f2h(nd(rng));
  for (size_t i = 0; i < size_t(M) * KVD; ++i) P<uint16_t>(Kb)[i] = f2h(nd(rng)), P<uint16_t>(Vb)[i] = f2h(nd(rng));
  for (size_t i = 0; i < size_t(M) * LDF; ++i) P<float>(G)[i] = 4.0f;  // sigmoid(4) gates
  id<MTLBuffer> tiles_b = buf(tiles.size() * sizeof(AttnTile)), parts_b = buf(size_t(std::max(parts, 1)) * HKV * 2 * 8 * 6 * 258 * 4);
  std::memcpy([tiles_b contents], tiles.data(), tiles.size() * sizeof(AttnTile));
  const AttnParams ap{H, HKV * 2, QD, KVD, LDF, BT, 0};
  run("flash_attention_qsa_g6s2", MTLSizeMake(HKV * 2, tiles.size(), 1), MTLSizeMake(384, 1, 1), false,
      {{0, Q}, {1, Kb}, {2, Vb}, {3, kv}, {4, blocks}, {5, desc_b}, {6, G}, {7, O}, {9, tiles_b}, {10, trees}, {11, parts_b},
       {12, tl}, {13, tc}, {14, rinfo_b}, {15, sel}},
      {{8, by(ap)}, {16, by(qp)}});
  if (!comb.empty()) {
    id<MTLBuffer> comb_b = buf(comb.size() * sizeof(AttnComb));
    std::memcpy([comb_b contents], comb.data(), comb.size() * sizeof(AttnComb));
    run("attention_combine", MTLSizeMake(64, H, comb.size() * 8), MTLSizeMake(64, 1, 1), true,
        {{0, parts_b}, {1, G}, {2, O}, {4, comb_b}}, {{3, by(ap)}});
  }
  // CPU reference for a few sparse rows: softmax over the selected blocks + tail (cached and batch keys)
  auto key_ptr = [&](int r, int i, int which) -> const uint16_t* {
    if (i < Pc) return P<uint16_t>(kv) + (size_t(table[i / BT]) * 2 + which) * BT * KVD + size_t(i % BT) * KVD;
    const int j = i - Pc, row = j < desc[r * 4 + 2] ? j : desc[r * 4 + 0] + (j - desc[r * 4 + 2]);
    return P<uint16_t>(which ? Vb : Kb) + size_t(row) * KVD;
  };
  double max_od = 0;
  for (int t = 0; t < std::min(ns, 5); ++t) {
    const int r = srows[size_t(t * std::max(1, ns / 5)) % size_t(ns)], sr = rinfo[r * 4 + 2], nb = rinfo[r * 4 + 1], s = sidx[r];
    std::vector<int> keys;
    for (int b = 0; b < nb; ++b)
      if ((P<uint32_t>(sel)[size_t(sr) * nbw + b / 32] >> (b % 32)) & 1u)
        for (int o = 0; o < 4; ++o) keys.push_back(4 * b + o);
    for (int i = 4 * nb; i <= s; ++i) keys.push_back(i);
    for (int h : {0, 7, 13, 23}) {
      const int kvh = h / 12;
      std::vector<double> lg(keys.size());
      double mx = -1e30;
      for (size_t k = 0; k < keys.size(); ++k) {
        const uint16_t* kp = key_ptr(r, keys[k], 0) + kvh * 256;
        double d = 0;
        for (int e = 0; e < 256; ++e) d += double(h2f(f2h(h2f(P<uint16_t>(Q)[size_t(r) * QD + h * 256 + e]) * 0.0625f))) * h2f(kp[e]);
        lg[k] = d, mx = std::max(mx, d);
      }
      double z = 0;
      for (double& v : lg) v = std::exp(v - mx), z += v;
      for (int e = 0; e < 256; e += 37) {
        double o = 0;
        for (size_t k = 0; k < keys.size(); ++k) o += lg[k] * h2f(key_ptr(r, keys[k], 1)[kvh * 256 + e]);
        o = o / z / (1.0 + std::exp(-4.0));
        const double g = h2f(P<uint16_t>(O)[size_t(r) * QD + h * 256 + e]);
        max_od = std::max(max_od, std::fabs(g - o));
      }
    }
  }
  std::printf("  sparse attention vs CPU (5 rows x 4 heads): max |diff| %.4f %s\n", max_od, max_od < 0.02 ? "ok" : "FAIL");
  if (max_od >= 0.02) ++failures;
  // save: rows of the root at positions P..P+Tr into new blocks
  std::vector<KVSave> saves;
  for (int i = 0; i < Tr; ++i) saves.push_back({i, table[(Pc + i) / BT], (Pc + i) % BT, 0});
  id<MTLBuffer> saves_b = buf(saves.size() * sizeof(KVSave));
  std::memcpy([saves_b contents], saves.data(), saves.size() * sizeof(KVSave));
  run("qsa_save", MTLSizeMake(16, saves.size(), 1), MTLSizeMake(16, 4, 1), true,
      {{0, IK}, {1, IP}, {2, pos}, {3, ib}, {4, saves_b}}, {{5, by(qp)}});
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    const std::string dir = argc > 1 ? argv[1] : "kernels";
    std::vector<std::string> files;
    if (DIR* d = ::opendir(dir.c_str())) {
      while (dirent* e = ::readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > 6 && n.compare(n.size() - 6, 6, ".metal") == 0) files.push_back(dir + "/" + n);
      }
      ::closedir(d);
    }
    std::sort(files.begin(), files.end());
    std::string src;
    for (const auto& f : files) {
      std::ifstream in(f, std::ios::binary);
      std::stringstream ss;
      ss << in.rdbuf();
      src += ss.str() + "\n";
    }
    dev = MTLCreateSystemDefaultDevice();
    queue = [dev newCommandQueue];
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_2;
    NSError* err = nil;
    lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:opts error:&err];
    if (!lib) {
      std::fprintf(stderr, "%s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }
    scenario("one prompt pass (parity mode 1)", 0, 3098, 0, 0, 1);
    scenario("decode step (mode 2)", 3097, 1, 0, 0, 2);
    scenario("1000-token pass at P = 2000 (mode 3)", 2000, 1000, 0, 0, 3);
    scenario("verify tree: root 1 + 4 leaves of 3", 3101, 1, 4, 3, 4);
    scenario("decode at 45K", 45000, 1, 0, 0, 5);
    std::printf("%s (%d failures)\n", failures ? "QSA CHECK FAILED" : "QSA CHECK OK", failures);
    return failures ? 1 : 0;
  }
}
