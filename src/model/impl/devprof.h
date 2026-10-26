// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Per-device profile (Fase 3, W2): the cost of a dependent dispatch boundary
// is measured once per GPU at first load (a chain of tiny dispatches under the
// serial encoder, as `bench_native --gap` does) and cached beside the GEMM
// tile cache; the thresholds that depend on it (whether a dispatch fold pays,
// whether a fused-norm finalizer should run in its own dispatch) are set from
// the measurement before the environment and set_option override them.
//
// Measured so far (docs/kernel-guidelines.md, "Measured facts: device
// profile"): a boundary costs 5-23 us on an M1 Max and ~4 us at 16 rows /
// nothing measurable at 1 row on an M4 Max (Fase 2); the tiny-chain number
// this profile records is the fixed part of that cost.
#pragma once

#include "gemm.h"

namespace espetero::native {

struct DeviceProfile {
  std::string device;      // [dev name]
  int family = 0;          // 1: Apple7/8 (M1, M2); 2: Apple9 (M3, M4); 3: newer; 0: unknown
  double boundary_us = 0;  // GPU time per dependent tiny dispatch inside a 64-deep chain (serial encoder)
  double tiny_iso_us = 0;  // the same tiny dispatch alone in its command buffer
  bool measured = false;   // measured in this process (false: read from the cache, or disabled)
  bool enabled = true;     // ESPETERO_DEVPROF=0 disables the profile (every default as before)
  double ms = 0;           // time spent measuring
  // A boundary is "dear" when a dispatch fold saves more than it costs in
  // in-kernel work: the M1 Max measures 5-23 us per boundary, the M4 Max
  // under 2 us (Fase 2, 2026-10-21).
  bool dear_boundary() const { return enabled && boundary_us >= 3.0; }
};

inline std::string devprof_cache_path(Runtime& rt) {
  std::string name = [[rt.dev name] UTF8String];
  for (auto& ch : name)
    if (!isalnum(static_cast<unsigned char>(ch))) ch = '_';
  return cache_dir() + "/device-profile-r1-" + name + ".json";
}

inline int gpu_family_of(id<MTLDevice> dev) {
  if (@available(macOS 15.0, *)) {
    if ([dev supportsFamily:MTLGPUFamilyApple9]) return 2;
  }
  if ([dev supportsFamily:MTLGPUFamilyApple8] || [dev supportsFamily:MTLGPUFamilyApple7]) return 1;
  return 0;
}

// The chain: gather_rows_f32 over 2048 floats (512 threads of one row), P -> Q
// -> P ..., every dispatch dependent on the previous one, serial encoder;
// per-dispatch GPU time of a 64-deep chain (best of 3 after 2 warm-ups) and
// of one dispatch alone (best of 5).
inline DeviceProfile measure_device_profile(Model& m) {
  const auto t0 = Clock::now();
  DeviceProfile p;
  p.device = [[m.dev name] UTF8String];
  p.family = gpu_family_of(m.dev);
  constexpr int W = 2048, CHAIN = 64;
  // Scratch outside the model's arena (released with this scope), so the
  // measurement leaves the arena's layout, and every later buffer's
  // placement, exactly as without it.
  id<MTLBuffer> P = [m.dev newBufferWithLength:size_t(W) * 4 options:kShared];
  id<MTLBuffer> Q = [m.dev newBufferWithLength:size_t(W) * 4 options:kShared];
  id<MTLBuffer> rows = [m.dev newBufferWithLength:16 options:kShared];
  std::memset([P contents], 0, size_t(W) * 4);
  std::memset([rows contents], 0, 16);
  const bool serial0 = m.serial, ball0 = m.barrier_all;
  m.serial = true;
  m.barrier_all = false;
  auto chain = [&](int n) {
    id<MTLCommandBuffer> cb = [m.queue commandBufferWithUnretainedReferences];
    Enc E(m, cb);
    for (int i = 0; i < n; ++i) {
      id<MTLBuffer> src = (i % 2) ? Q : P, dst = (i % 2) ? P : Q;
      E.pipe("gather_rows_f32");
      E.set(0, src);
      E.set(1, dst);
      E.set(2, rows);
      E.bytes(3, W);
      E.rd(src);
      E.wr(dst);
      E.threads(MTLSizeMake(size_t(W / 4), 1, 1), MTLSizeMake(64, 1, 1));
    }
    cb = E.finish();
    commit_wait(cb);
    return gpu_ms(cb);
  };
  for (int r = 0; r < 2; ++r) chain(CHAIN);
  double best = 1e30, iso = 1e30;
  for (int r = 0; r < 3; ++r) best = std::min(best, chain(CHAIN));
  for (int r = 0; r < 5; ++r) iso = std::min(iso, chain(1));
  m.serial = serial0;
  m.barrier_all = ball0;
  p.boundary_us = best * 1e3 / CHAIN;
  p.tiny_iso_us = iso * 1e3;
  p.measured = true;
  p.ms = ms_since(t0);
  return p;
}

// The profile of this device: from the cache (device-profile-r1-<name>.json),
// else measured and cached. ESPETERO_DEVPROF=0 disables it; ESPETERO_DEVPROF_FRESH=1 re-measures.
inline DeviceProfile device_profile(Model& m) {
  if (const char* v = std::getenv("ESPETERO_DEVPROF"); v && *v == '0') {
    DeviceProfile p;
    p.device = [[m.dev name] UTF8String];
    p.family = gpu_family_of(m.dev);
    p.enabled = false;
    return p;
  }
  const std::string path = devprof_cache_path(m);
  if (file_exists(path) && !std::getenv("ESPETERO_DEVPROF_FRESH")) {
    try {
      const json::Value c = json::parse(read_file(path));
      if (c.is_object() && c.get("boundary_us") && c.get("family")) {
        DeviceProfile p;
        p.device = [[m.dev name] UTF8String];
        p.family = c.get("family")->as_int();
        p.boundary_us = c.get("boundary_us")->as_double();
        if (const json::Value* t = c.get("tiny_iso_us")) p.tiny_iso_us = t->as_double();
        return p;
      }
    } catch (...) {
    }
  }
  DeviceProfile p = measure_device_profile(m);
  json::Value c = json::Value::object();
  c.set("device", json::Value(p.device));
  c.set("family", json::Value(p.family));
  c.set("boundary_us", json::Value(p.boundary_us));
  c.set("tiny_iso_us", json::Value(p.tiny_iso_us));
  const std::string tmp = path + ".tmp";
  FILE* f = paths::make_private_file(tmp) ? std::fopen(tmp.c_str(), "wb") : nullptr;
  if (f) {
    const std::string s = json::dump(c);
    std::fwrite(s.data(), 1, s.size(), f);
    std::fclose(f);
    std::rename(tmp.c_str(), path.c_str());
  }
  return p;
}

// The kernel-selection thresholds the profile sets (before the environment
// and set_option override them). Every entry is a measured fact recorded in
// docs/kernel-guidelines.md ("Measured facts: device profile"); a threshold
// without a measurement on a device class keeps the engine's default there.
//
// Measured boundary costs (a tiny dependent dispatch in a 64-deep chain):
// M1 Max 4.20 us, M4 Max 2.12 us; dear_boundary() is >= 3 us.
inline void apply_device_profile(Model& m, const DeviceProfile& p) {
  if (!p.enabled) return;
  // Split-K partials of the out / down projections summed by the following
  // RMSNorm (fuse_nrm, Fase 2, bit-identical): pays where a boundary is dear.
  // M1 Max (MacBook, 35B-A3B 4-bit 16-row step, three alternating
  // --long-ab fuse_nrm 0,1 pairs, exclusive, all clean): 64.24 -> 62.60,
  // 63.15 -> 63.10, 69.58 -> 67.90 ms GPU (median -2.4 %); the 27B (fold
  // idle: so_max) and 1-row steps (fused norms) within +-0.45 %. M4 Max
  // (Fase 2): -1.2 / -0.6 / 0 / 0 % against a 0.7 % spread, 27B +0.7 %: off.
  m.fuse_nrm = p.dear_boundary();
  // The other dispatch folds of Fase 2 (fuse_kvsave, fuse_rsum, fuse_sort):
  // neutral or slower on the M4 Max; on the M1 Max measured only together
  // with fuse_nrm (fase2 0,2): not set until measured alone.
  // Not device-dependent (the same value measured right on both Macs; set
  // here so the table is explicit):
  // - moe_bm_rows 2048: 16-row expert tiles from 2048 rows up (M4 Max
  //   -1.4 % 4-bit / -8.3 % Q6_K at 2048 rows, 1024 rows neutral, 512 rows
  //   +3.1 %; M1 Max -3.1 % / -6.8 % at 2048 rows).
  // - gemv_max_m 2: the FMA GEMV on 3-4-row passes loses on both (M1 +22 %
  //   at 4 sequences; M4 verify pass +7.8 % 35B / +12.8 % 27B).
  // - fuse_norm_max_m 2: fused norms on 3-8-row passes lose on both (M1
  //   +2.4 % / +4.7 % at 4 / 8 rows; M4 -0.2 % 35B / +4.3 % 27B).
  // - attn_split 512, attn_min_chunk 32: 256 / 512 / 1024 keys per tile
  //   measure alike on the M4 Max (within 0.1 %, identical logits); larger
  //   minimum chunks are not output-preserving and lose (+0.9 / +2.5 %).
  m.moe_bm_rows = 2048;
  m.tune.gemv_max_m = std::min(m.tune.gemv_max_m, 2);
  m.fuse_norm_max_m = std::min(m.fuse_norm_max_m, 2);
}

}  // namespace espetero::native
