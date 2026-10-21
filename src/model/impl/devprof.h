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
  id<MTLBuffer> P = m.buffer(size_t(W) * 4), Q = m.buffer(size_t(W) * 4), rows = m.buffer(16);
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
// docs/kernel-guidelines.md; a threshold without a measurement on a device
// class keeps the engine's default there.
inline void apply_device_profile(Model& m, const DeviceProfile& p) {
  if (!p.enabled) return;
  // Dispatch folds (Fase 2, lossless): neutral on the M4 Max (boundary ~0 at
  // 1 row); expected to pay where a boundary is dear (M1 Max measurements
  // pending: kept off until they are in).
  (void)m;
  // Fused-norm finalizers as their own dispatch (Fase 3, lossless): measured
  // on the M4 Max first (see the Fase 3 facts); off until measured.
}

}  // namespace espetero::native
