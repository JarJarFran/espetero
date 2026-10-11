// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The forest pass: slots over the paged KV pool and the state pool, per-row
// and per-tree metadata, and one command buffer that runs every tree's rows
// through the Qwen3.5 decoder. Projections and MLPs cover all rows at once;
// sequence mixing (convolution, delta rule, attention, K/V and state saves)
// also runs as one dispatch per layer for the whole forest, indexed by
// per-row / per-segment / per-tile tables. Images go through the vision
// tower in the same pass; the LM head reads each readout row.
#pragma once

#include "vision.h"
#include "qwen4_io.h"

namespace espetero::native {

// ---------------------------------------------------------------- pools
inline void make_pools(Model& m, int user_slots, size_t kv_tokens, int bt) {
  const Config& c = m.cfg;
  StatePool& S = m.st;
  S.conv_floats = size_t(3) * size_t(c.C());
  S.st_floats = size_t(c.Hv) * size_t(c.Dv) * size_t(c.Dk);
  S.mtp_floats = m.mtp.present ? size_t(1 + kMaxDraft) * size_t(c.q4 ? c.W() : c.D) : 0;  // Qwen4-Exp: pre-mixer streams
  const int n_temp = 8;
  S.n_slots = user_slots + n_temp + 1;
  S.zero_slot = user_slots;
  S.first_temp = user_slots + 1;
  size_t off = 0;
  S.conv_base.clear();
  S.st_base.clear();
  for (int k = 0; k < c.n_linear; ++k) {
    S.conv_base.push_back(off);
    off += S.conv_floats * size_t(S.n_slots);
    off = align_up(off, 64);
    S.st_base.push_back(off);
    off += S.st_floats * size_t(S.n_slots);
    off = align_up(off, 64);
  }
  S.mtp_base = off;
  off += S.mtp_floats * size_t(S.n_slots);
  S.total_floats = std::max<size_t>(off, 16);
  S.buf = m.buffer(S.total_floats * 4);
  float* p = static_cast<float*>([S.buf contents]);
  // The zero slot is the state of an empty context.
  for (int k = 0; k < c.n_linear; ++k) {
    std::memset(p + S.conv_base[size_t(k)] + size_t(S.zero_slot) * S.conv_floats, 0, S.conv_floats * 4);
    std::memset(p + S.st_base[size_t(k)] + size_t(S.zero_slot) * S.st_floats, 0, S.st_floats * 4);
  }

  KVPool& K = m.kv;
  K.bt = bt;
  K.kv_dim = c.gemma ? c.kv_dim_g() : c.kv_dim();  // Gemma 4: the global layers' width
  K.n_layers = c.n_full + (m.mtp.present ? 1 : 0);
  K.n_blocks = int(std::max<size_t>(1, (kv_tokens + size_t(bt) - 1) / size_t(bt)));
  K.region_halves = size_t(K.n_blocks) * K.block_halves();
  K.buf = m.buffer(std::max<size_t>(K.region_halves * size_t(K.n_layers) * 2, 16));
  K.i_layers = c.q4 ? c.n_full + (m.mtp.present ? 1 : 0) : 0;
  if (K.i_layers) K.ibuf = m.buffer(K.iregion_halves() * size_t(K.i_layers) * 2);
  K.ref.assign(size_t(K.n_blocks), 0);
  K.free_list.clear();
  for (int b = K.n_blocks - 1; b >= 0; --b) K.free_list.push_back(b);

  // Gemma 4 sliding-window layers: a ring of KV blocks per user slot
  // holding its last `window` positions (gemma.h).
  if (c.gemma && c.n_swa > 0) {
    KVPool& R = m.kvs;
    R.bt = bt;
    R.kv_dim = c.kv_dim();
    R.n_layers = c.n_swa;
    m.ring_blocks = (c.window + bt - 1) / bt;
    R.n_blocks = user_slots * m.ring_blocks;
    R.region_halves = size_t(R.n_blocks) * R.block_halves();
    R.buf = m.buffer(std::max<size_t>(R.region_halves * size_t(R.n_layers) * 2, 16));
  }

  m.n_user = user_slots;
  m.slots.assign(size_t(user_slots), Slot{});
}

inline size_t slot_state_bytes(const Config& c, bool mtp) {
  // Gemma 4: a slot's state is its ring of sliding-window K/V.
  if (c.gemma) return size_t(c.n_swa) * size_t((c.window + 63) / 64 * 64) * 2 * size_t(c.kv_dim()) * 2;
  return (size_t(3) * size_t(c.C()) + size_t(c.Hv) * size_t(c.Dv) * size_t(c.Dk)) * size_t(c.n_linear) * 4 +
         (mtp ? size_t(1 + kMaxDraft) * size_t(c.q4 ? c.W() : c.D) * 4 : 0);
}

inline size_t kv_bytes_per_token(const Config& c, bool mtp) {
  if (c.gemma) return size_t(c.n_full) * 2 * size_t(c.kv_dim_g()) * 2;  // the global layers
  // + the Qwen4-Exp sparse attention index (KVPool::iblock_halves per token and layer)
  const size_t idx = c.q4 ? size_t(c.n_full + (mtp ? 1 : 0)) * size_t(KVPool::kIdxHd + KVPool::kIdxHd / 4 + 8) * 2 : 0;
  return size_t(c.n_full + (mtp ? 1 : 0)) * 2 * size_t(c.kv_dim()) * 2 + idx;
}

// Qwen4-Exp n-gram mixing state of every state slot (the last 9 normalized
// conv inputs and the last two token ids): allocated at the first qwen4 pass
// or the first checkpoint loaded from disk.
inline void ensure_ple_state(Model& m) {
  if (!m.cfg.q4 || m.ple_hist) return;
  const size_t n = size_t(m.st.n_slots) * Model::kPleHist * size_t(m.cfg.W());
  m.ple_hist = m.buffer(n * 4);
  std::memset([m.ple_hist contents], 0, n * 4);
  m.ple_ctx.assign(size_t(m.st.n_slots), {m.cfg.eos_token_id, m.cfg.eos_token_id});
}

inline Slot& user_slot(Model& m, int s) {
  if (s < 0 || s >= m.n_user || !m.slots[size_t(s)].used) fail("invalid slot " + std::to_string(s));
  return m.slots[size_t(s)];
}

// Copy a slot's recurrent/conv state and MTP rows (GPU, synchronous).
inline void copy_slot_state(Model& m, int dst, int src) {
  const StatePool& S = m.st;
  id<MTLCommandBuffer> cb = [m.queue commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  auto cp = [&](size_t base, size_t floats) {
    if (!floats) return;
    [bl copyFromBuffer:S.buf
          sourceOffset:(base + size_t(src) * floats) * 4
              toBuffer:S.buf
     destinationOffset:(base + size_t(dst) * floats) * 4
                  size:floats * 4];
  };
  for (size_t k = 0; k < S.conv_base.size(); ++k) {
    cp(S.conv_base[k], S.conv_floats);
    cp(S.st_base[k], S.st_floats);
  }
  cp(S.mtp_base, S.mtp_floats);
  // Gemma 4: the slot's ring of sliding-window K/V, per sliding layer.
  if (m.kvs.n_layers && dst < m.n_user && src < m.n_user) {
    const KVPool& R = m.kvs;
    const size_t bytes = size_t(m.ring_blocks) * R.block_halves() * 2;
    for (int l = 0; l < R.n_layers; ++l)
      [bl copyFromBuffer:R.buf
            sourceOffset:(R.layer_base(l) + size_t(src) * size_t(m.ring_blocks) * R.block_halves()) * 2
                toBuffer:R.buf
       destinationOffset:(R.layer_base(l) + size_t(dst) * size_t(m.ring_blocks) * R.block_halves()) * 2
                    size:bytes];
  }
  [bl endEncoding];
  commit_wait(cb);
  if (m.cfg.q4 && m.ple_hist) {  // Qwen4-Exp n-gram mix state (host-written)
    const size_t n = size_t(Model::kPleHist) * size_t(m.cfg.W());
    float* h = static_cast<float*>([m.ple_hist contents]);
    std::memcpy(h + size_t(dst) * n, h + size_t(src) * n, n * 4);
    m.ple_ctx[size_t(dst)] = m.ple_ctx[size_t(src)];
  }
}

// ---------------------------------------------------------------- scratch
inline void ensure_scratch(Model& m, int rows, int readouts) {
  Scratch& s = m.sc;
  const Config& c = m.cfg;
  if (rows > s.rows) {
    const int R = std::max(rows, std::max(64, s.rows * 3 / 2));
    auto f = [&](size_t n) { return m.buffer(n * 4); };
    auto hf = [&](size_t n) { return m.buffer(n * 2); };
    s.tokens = f(size_t(R));
    s.h = f(size_t(R) * size_t(c.D));
    s.xn = hf(size_t(R) * size_t(c.D));
    s.proj = f(size_t(R) * size_t(c.proj_ld()));
    s.qo = hf(size_t(R) * size_t(c.C()));
    s.gb = f(size_t(R) * 2 * size_t(c.Hv));
    s.y = f(size_t(R) * size_t(c.Vd()));
    s.yn = hf(size_t(R) * size_t(c.Vd()));
    // Gemma 4: the wider of its two attention geometries.
    const size_t hd = size_t(c.gemma ? std::max(c.HD, c.HDg) : c.HD);
    const size_t kvd = size_t(c.gemma ? std::max(c.kv_dim(), c.kv_dim_g()) : c.kv_dim());
    s.Q = hf(size_t(R) * size_t(c.H) * hd);
    s.Kb = hf(size_t(R) * kvd);
    s.Vb = hf(size_t(R) * kvd);
    s.O = hf(size_t(R) * size_t(c.H) * hd);
    if (c.gemma && c.E) {
      s.g_xn2 = hf(size_t(R) * size_t(c.D));
      s.g_xr = hf(size_t(R) * size_t(c.D));
    }
    const int Imax = std::max(c.I, c.I_shared);
    s.act = hf(size_t(R) * size_t(std::max(Imax, 2 * c.D)));
    if (m.split_mlp) {
      s.g32 = f(size_t(R) * size_t(Imax));
      s.u32 = f(size_t(R) * size_t(Imax));
    }
    s.rows = R;
  }
  if (readouts > s.leaves) {
    const int N = std::max(readouts, 64);
    s.lse = m.buffer(sizeof(float) * size_t(N));
    s.argmax = m.buffer(sizeof(int32_t) * size_t(N));
    s.hn = m.buffer(sizeof(uint16_t) * size_t(N) * size_t(c.D));
    s.logits = m.buffer(sizeof(float) * size_t(N) * size_t(c.V));
    s.hc = m.buffer(sizeof(float) * size_t(N) * size_t(c.D));
    s.xc = m.buffer(sizeof(uint16_t) * size_t(N) * size_t(c.D));
    s.mixc = m.buffer(sizeof(uint16_t) * size_t(N) * size_t(std::max(c.Vd(), c.H * (c.gemma ? std::max(c.HD, c.HDg) : c.HD))));
    s.actc = m.buffer(sizeof(uint16_t) * size_t(N) * size_t(std::max(c.I, c.I_shared)));
    const int parts = (c.V + 4095) / 4096;
    s.tk_x = m.buffer(sizeof(float) * size_t(N) * size_t(parts) * 32);
    s.tk_i = m.buffer(sizeof(int32_t) * size_t(N) * size_t(parts) * 32);
    s.tk_stats = m.buffer(sizeof(float) * 2 * size_t(N) * size_t(parts));
    s.tk_out = m.buffer(sizeof(uint32_t) * size_t(N) * (2 * 32 + 3));
    s.leaves = N;
  }
}

// ---------------------------------------------------------------- MoE
// Expert indices moe_sort can take (pool slots + staged records <= 2 E; its
// threadgroup tables hold 1024 + 1).
inline int moe_sort_cap(const Config& c) { return std::min(2 * c.E + 64, 1024); }

inline void ensure_moe_scratch(Model& m, int rows) {
  MoeScratch& z = m.mz;
  const Config& c = m.cfg;
  if (!c.E || rows <= z.rows) return;
  const int R = std::max(rows, std::max(64, z.rows * 3 / 2));
  const size_t pairs = size_t(R) * size_t(c.k);
  auto b = [&](size_t n) { return m.buffer(std::max<size_t>(n, 16)); };
  z.logits = b(sizeof(float) * size_t(R) * size_t(c.E));
  z.ids = b(sizeof(int32_t) * pairs);
  z.w = b(sizeof(float) * pairs);
  z.counts = b(sizeof(int32_t) * size_t(c.E));
  z.fill = b(sizeof(int32_t) * size_t(c.E));
  // moe_sort writes one offset per expert index it sorts: with the expert
  // cache, pool slots + staged records, up to S + E on a cold pool.
  // Sized for E it overflowed into the next small buffer (small MTLBuffers
  // share pages): with parity_test's layout the grouped GEMVs' indirect
  // arguments, so long passes with >= 12 GB caches ran a ~10^9-threadgroup
  // grid for minutes (an apparent stall).
  z.offsets = b(sizeof(int32_t) * size_t(moe_sort_cap(c)));
  z.tiles = b(sizeof(int32_t) * 4 * (pairs / 8 + size_t(c.E) + size_t(R) / 8 + 2));
  z.ntiles = b(sizeof(uint32_t) * 8);  // indirect dispatch arguments of the two grouped GEMVs
  z.rowmap = b(sizeof(int32_t) * (pairs + size_t(R)));
  z.slotpos = b(sizeof(int32_t) * pairs);
  // Rows [pairs, pairs + R) of act / ys are the shared expert's when it runs as a tile.
  z.act = b(sizeof(uint16_t) * (pairs + size_t(R)) * size_t(c.Ie));
  z.ys = b(sizeof(float) * (pairs + size_t(R)) * size_t(c.D));
  z.shared = b(sizeof(float) * size_t(R) * size_t(c.D));
  z.sgate = b(sizeof(float) * size_t(R));
  z.cparts = b(sizeof(float) * size_t(R) * 64);
  z.cctr = b(sizeof(int32_t) * size_t(R));
  std::memset([z.cctr contents], 0, sizeof(int32_t) * size_t(R));
  z.rparts = b(sizeof(float) * size_t(R) * 8 * size_t(c.E + 1));
  z.rctr = b(sizeof(int32_t) * size_t(R));
  std::memset([z.rctr contents], 0, sizeof(int32_t) * size_t(R));
  z.rows = R;
}

struct MoeRouteParams {
  int32_t E, k, M;
};
struct MoeSortParams {
  int32_t E, k, M, gu_blocks, down_blocks, shared, remap;
};
struct MoeRouteFusedParams {
  int32_t E, k, K, lda, gu_blocks, down_blocks, build_tiles, shared;
};
struct MoeRouteParParams {
  int32_t E, k, K, lda, gu_blocks, down_blocks, build_tiles, shared, nsl, sync;
};
struct TopKParams {
  int32_t V, K, chunk, parts, chain, chain_off;
};
struct TopKSampleRec {
  uint32_t seed_lo, seed_hi;
  int32_t pos_lo, pos_hi;
  float top_p, min_p;
  int32_t top_k, draw;
};
struct MoeGemmParams {
  int32_t Ne, N, K, lda, ldy, gather, shared_e, N2, direct, nbpt, swap, route_nsl, route_E, route_k, route_sync, route_shared;
  int32_t st_first = 0, st_count = 0, st_words = 0, st_half4 = 0;  // staged experts in records (moe_gemv8 only)
  int32_t split = 0;  // split pass, moe_gemv8 only: 1 the ready experts' tiles, 2 those of experts being read (m.xld)
};
struct MoeCombineParams {
  int32_t D, k, norm;
  float eps;
  int32_t slice;
};
struct MoeDownCombParams {
  int32_t D, N, N2, Ie, k, norm;
  float eps;
};

// Diagnostic (the long-pass stall): the sort's tile table of a pooled
// pass read back and checked against the routing it was built from (ids
// remapped by m.xmap), with the memory state; mode 2 stops before the grouped
// GEMVs of the first pass with staged experts, so it is inspected without
// running them.
inline void moe_check_tiles(Model& m, Enc& E, const MoeScratch& z, const ExpertPool* PV, int li, int Eeff, int rows,
                            int shared, int mode) {
  const Config& c = m.cfg;
  E.flush();
  const int n = rows * c.k;
  const int32_t* ids = static_cast<const int32_t*>([z.ids contents]);
  const int32_t* map = static_cast<const int32_t*>([m.xmap contents]);
  const uint32_t* ind = static_cast<const uint32_t*>([z.ntiles contents]);
  const int32_t* tl = static_cast<const int32_t*>([z.tiles contents]);
  std::vector<int> cnt(size_t(Eeff) + 1, 0);
  int bad_ids = 0;
  for (int i = 0; i < n; ++i) {
    const int id = ids[i];
    const int e = id >= 0 && id < c.E ? map[id] : -1;
    if (e < 0 || e >= Eeff) {
      ++bad_ids;
      continue;
    }
    ++cnt[size_t(e)];
  }
  int want = 0, used = 0;
  for (int e = 0; e < Eeff; ++e) {
    want += (cnt[size_t(e)] + 7) / 8;
    used += cnt[size_t(e)] > 0 ? 1 : 0;
  }
  if (shared) want += (rows + 7) / 8;
  const size_t cap = size_t([z.tiles length]) / 16;
  const int got = int(ind[0]);
  int bad_tiles = 0, first_bad = -1;
  std::vector<int> seen(size_t(Eeff) + 1, 0);
  for (int t = 0; t < std::min(got, int(cap)); ++t) {
    const int e = tl[4 * t], r0 = tl[4 * t + 1], nr = tl[4 * t + 2];
    const bool ok = e >= 0 && e <= Eeff && nr >= 1 && nr <= 8 && r0 >= 0 && r0 + nr <= n + rows;
    if (ok) seen[size_t(e)] += nr;
    if (!ok && bad_tiles++ == 0) first_bad = t;
  }
  int bad_counts = 0;
  for (int e = 0; e < Eeff; ++e) bad_counts += seen[size_t(e)] != cnt[size_t(e)] ? 1 : 0;
  vm_statistics64_data_t vs{};
  mach_msg_type_number_t vcnt = HOST_VM_INFO64_COUNT;
  host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vs), &vcnt);
  const double pg = double(vm_kernel_page_size) / 1e9;
  std::fprintf(stderr,
               "moe check: layer %d, rows %d, S %d, staged %d, Eeff %d, experts used %d, tiles %d (want %d, cap %zu), "
               "grids %u x %u, %u x %u, bad ids %d, bad tiles %d (first %d), bad counts %d; staging %.2f GB at +%zu; "
               "GPU allocated %.2f GB; system free %.2f, wired %.2f, compressed %.2f GB\n",
               li, rows, PV ? PV->S : 0, PV ? PV->st_count : 0, Eeff, used, got, want, cap, ind[0], ind[1], ind[3], ind[4],
               bad_ids, bad_tiles, first_bad, bad_counts, PV && PV->st.b ? double([PV->st.b length]) / 1e9 : 0.0,
               PV ? PV->st.off : size_t(0), double([m.dev currentAllocatedSize]) / 1e9, double(vs.free_count) * pg,
               double(vs.wire_count) * pg, double(vs.compressor_page_count) * pg);
  if (got != want || got > int(cap) || bad_ids || bad_tiles || bad_counts)
    fail("moe check: the tile table does not match the routing (layer " + std::to_string(li) + ")");
  if (mode == 2 && PV && PV->st_count > 0) fail("moe check: stopped before the staged grouped GEMVs (ESPETERO_MOE_CHECK=2)");
}

// The shared expert's gate/up (fused or split) into `act` (half [rows, I]).
inline void shared_mlp_in(Model& m, Enc& E, const Layer& L, id<MTLBuffer> X, id<MTLBuffer> act, int rows) {
  const Config& c = m.cfg;
  Scratch& s = m.sc;
  if (L.gate_up.out() > 0) {
    gemm(E, m.tune, L.gate_up, X, 0, c.D, act, 0, L.I, rows, 2);
    return;
  }
  gemm(E, m.tune, L.gate, X, 0, c.D, s.g32, 0, L.I, rows, 0);
  gemm(E, m.tune, L.up, X, 0, c.D, s.u32, 0, L.I, rows, 0);
  E.pipe("swiglu_f32");
  E.set(0, s.g32);
  E.set(1, s.u32);
  E.set(2, act);
  E.bytes(3, L.I);
  E.rd(s.g32);
  E.rd(s.u32);
  E.wr(act);
  E.threads(MTLSizeMake(size_t(L.I / 4), size_t(rows), 1), MTLSizeMake(64, 1, 1));
}

// The shared expert of MoE layer L as one more expert tile of the grouped
// GEMVs: 1 on every GEMV-sized pass (the experts' widths and layouts), 2 on
// the 1-2-row FMA kernels only (an 8-bit shared expert beside 4/5/6-bit
// experts, the _s8 kernels), 0 never: its own GEMMs (another layout, e.g. a
// Q8_0 shared expert beside Q6_K experts: a kernel takes one layout's
// function constants; or another width on both projections).
inline int moe_shared_tile_kind(const Config& c, const Layer& L) {
  if (L.I != c.Ie || L.gate_up.out() <= 0 || L.gate_up.dense || L.down.dense) return 0;
  if (L.gate_up.q.fc() != L.experts_gu.fc() || L.down.q.fc() != L.experts_down.fc()) return 0;
  if (L.gate_up.q.bits == L.experts_gu.bits && L.down.q.bits == L.experts_down.bits) return 1;
  const int b = L.experts_gu.bits;
  return L.gate_up.q.bits == 8 && L.down.q.bits == 8 && L.experts_down.bits == b && b >= 4 && b <= 6 ? 2 : 0;
}

// Kernel of a grouped expert GEMV (epi 2: gate/up with the gated epilogue, 0:
// down): the 1-2-row FMA kernel (ks K slices, the route folded into the
// gate/up prologue, w2: the shared expert as tile k, _s8 when its width
// differs) or the 8-row MMA kernel.
inline std::string moe_gemv_name(const QW& w, const QW* w2, bool fma_path, int rows, int epi, bool route, int ks) {
  const std::string tail = epi == 2 ? "_swiglu" : "_store";
  if (!fma_path) return "moe_gemv8r" + std::to_string(w.bits) + tail;
  return "moe_gemv" + std::to_string(w.bits) + "_" + std::to_string(rows) + tail + (route ? "_route" : "") +
         (ks == 8 ? "" : "_k" + std::to_string(ks)) + (w2 && w2->bits != w.bits ? "_s" + std::to_string(w2->bits) : "");
}

// The single-row down projection + combine: comb_ks K slices per slot.
inline std::string moe_down_comb_name(const QW& w, const QW& w2, int comb_ks) {
  return "moe_down_comb" + std::to_string(w.bits) + "_k" + std::to_string(comb_ks) +
         (w2.bits != w.bits ? "_s" + std::to_string(w2.bits) : "");
}

// The expert kernels moe_mlp can use on layer L (prewarm), from its own
// shared-tile choice and names: every K-slice, route and combine variant of
// the 1-2-row FMA path and the 8-row kernel. (A hand-kept copy of the choice
// took the _s8 suffix from the gate/up widths for the down kernel too: Q6_K
// experts with Q8_0 down projections in some layers and a Q8_0 shared expert
// asked for a moe_gemv8_..._s8 kernel that no pass uses, and the load failed.)
inline void moe_pipelines(const Model& m, const Layer& L, std::vector<std::pair<std::string, int>>& out) {
  const QW& gu = L.experts_gu;
  const QW& dn = L.experts_down;
  const bool tiles = m.fuse_shared && moe_shared_tile_kind(m.cfg, L) > 0;  // either kind on the FMA path
  const QW* sgu = tiles ? &L.gate_up.q : nullptr;
  const QW* sdn = tiles ? &L.down.q : nullptr;
  out.push_back({moe_gemv_name(gu, sgu, false, 8, 2, false, 8), gu.fc()});
  out.push_back({moe_gemv_name(dn, sdn, false, 8, 0, false, 8), dn.fc()});
  for (int rows : {1, 2})
    for (int ks : {8, 4, 2}) {
      out.push_back({moe_gemv_name(gu, sgu, true, rows, 2, false, ks), gu.fc()});
      out.push_back({moe_gemv_name(dn, sdn, true, rows, 0, false, ks), dn.fc()});
    }
  for (int ks : {8, 4, 2}) out.push_back({moe_gemv_name(gu, sgu, true, 1, 2, true, ks), gu.fc()});
  if (sdn)
    for (int kk : {1, 2}) out.push_back({moe_down_comb_name(dn, *sdn, kk), dn.fc()});
}

// The MoE MLP of `rows` rows: X (half, post-norm) in, H (fp32 residual)
// accumulated; with `nout`, the combine also writes the next norm's fp16 rows.
// act: scratch of [rows, I_shared] halves for the shared expert.
inline void moe_mlp(Model& m, Enc& E, const Layer& L, id<MTLBuffer> X, id<MTLBuffer> act, id<MTLBuffer> H, int rows,
                    const NormLink* nout, size_t h_off = 0) {  // h_off: byte offset of the residual rows in H
  const Config& c = m.cfg;
  ensure_moe_scratch(m, rows);
  MoeScratch& z = m.mz;
  const int gu_blocks = (2 * c.Ie + 31) / 32, down_blocks = (c.D + 31) / 32;
  // The shared expert runs as one more expert tile of the grouped GEMVs when
  // its matrices have the experts' width and layout; otherwise as its own
  // two GEMVs (a third of the MoE layer's dispatches at one row).
  // GEMV-sized passes only: at prefill sizes the shared expert's rgemm tiles
  // beat the 8-row grouped path (2048 rows: 1.87 -> 1.96 s with tiles).
  // Same bits on every path; or an 8-bit shared expert beside 4/5/6-bit
  // routed experts on the 1-2-row FMA kernels (oQ4e-style files).
  const int tile_kind = moe_shared_tile_kind(c, L);
  const bool shared_tiles = m.fuse_shared && rows <= 8 &&
                            (tile_kind == 1 || (tile_kind == 2 && m.moe_fma && rows <= m.moe_fma_rows));
  if (!shared_tiles) {
    shared_mlp_in(m, E, L, X, act, rows);
    gemm(E, m.tune, L.down, act, 0, L.I, z.shared, 0, c.D, rows, 0);
  }
  const int shared_flag = shared_tiles ? 1 : 0;
  // Routing: fused (router GEMV + shared gate + softmax + top-k in one
  // threadgroup per row, and the tile table for a single row) when the
  // router and the shared gate share a quantized layout and E <= 256;
  // otherwise the separate GEMVs, top-k and sort.
  // GEMV-sized passes only: a row's threadgroup streams the whole router
  // matrix, which is right for a few rows and a disaster for prefill.
  // A dense (fp16) router with a quantized shared gate runs the parallel
  // fused route with the fp16 partial GEMV (exact, no requantization).
  // Both dense (GGUF MoE files: F32 router and shared gate) run the parallel
  // route with fp16 partial GEMVs for both (moe_route_parhh).
  const bool gate_f16 = L.router.dense && L.shared_gate.dense && L.shared_gate.d.out == 1 && m.route_par && c.D % 64 == 0;
  const bool router_f16 = L.router.dense && (!L.shared_gate.dense || gate_f16) && m.route_par && c.D % 64 == 0;
  const bool fused_route = m.fuse_route && rows <= 8 && (!L.shared_gate.dense || gate_f16) && c.E <= 256 &&
                           (router_f16 || (!L.router.dense && L.router.q.bits == L.shared_gate.q.bits &&
                                           L.router.q.fc() == L.shared_gate.q.fc()));
  const bool par_route = fused_route && (m.route_par || router_f16) && c.D % 64 == 0;
  bool route_in_gemv = false;
  MoeRouteParParams route_p{};
  if ((m.moe_stages & 1) && par_route) {
    // Router GEMV over nblocks x nsl threadgroups (one k-step per simdgroup),
    // the row's last arriver does softmax + rank top-k + tile table.
    const QW& r = router_f16 && !gate_f16 ? L.shared_gate.q : L.router.q;  // (bits, fc) of the pipeline; for a dense router the gate's
    const QW& g = L.shared_gate.q;
    const int nblocks = (c.E + 31) / 32;
    const int KG = c.D / 64;
    const int nsl = std::max(1, std::min(8, m.route_nsl > 0 ? m.route_nsl : (KG + 7) / 8));
    const int sync = m.route_sync | (m.route_seqexp ? 16 : 0);
    // build_tiles 2: single-row pass with direct grids, the indirect arguments are not written
    const int build = (rows == 1 && !m.route_notiles) ? (m.moe_direct ? 2 : 1) : 0;
    const MoeRouteParParams fp{c.E, c.k, c.D, c.D, gu_blocks, down_blocks, build, shared_flag, nsl, sync};
    const std::string name = gate_f16 ? std::string("moe_route_parhh") : "moe_route_par" + std::to_string(r.bits) + (router_f16 ? "h" : "");
    const int rfc = gate_f16 ? 64 : r.fc();  // both dense: the quantized path is compiled out (any valid constants)
    if ([m.kernel(name, rfc) maxTotalThreadsPerThreadgroup] < 256) fail(name + ": needs 256 threads");
    E.pipe(name, rfc);
    gemm_tag(E, name, c.E, c.D, rows);
    E.set(0, X);
    if (router_f16) {
      E.set(1, L.router.d.wr);
      E.set(2, g.sb);  // unused by the fp16 path
      E.set(3, g.wr);
    } else {
      E.set(1, r.wr);
      E.set(2, r.sb);
      E.set(3, r.wh ? r.wh : r.wr);
    }
    if (gate_f16) {
      E.set(4, L.shared_gate.d.wr);
      E.set(5, L.shared_gate.d.wr);  // unread
      E.set(6, L.shared_gate.d.wr);  // unread
    } else {
      E.set(4, g.wr);
      E.set(5, g.sb);
      E.set(6, g.wh ? g.wh : g.wr);
    }
    E.set(7, z.ids);
    E.set(8, z.w);
    E.set(9, z.sgate);
    E.set(10, z.tiles);
    E.set(11, z.ntiles);
    E.set(12, z.rowmap);
    E.set(13, z.slotpos);
    E.bytes(14, fp);
    E.set(15, z.rparts);
    E.set(16, z.rctr);
    E.rd(X);
    E.wr(z.ids);
    E.wr(z.w);
    E.wr(z.sgate);
    E.wr(z.rparts);
    E.rd(z.rctr);
    E.wr(z.rctr);
    if (rows == 1) {
      E.wr(z.tiles);
      if (build == 1) E.wr(z.ntiles);
      E.wr(z.rowmap);
      E.wr(z.slotpos);
    }
    E.groups(MTLSizeMake(size_t(nblocks) * size_t(nsl), size_t(rows), 1), MTLSizeMake(256, 1, 1));
    // The finalize folded into the gate/up GEMV's prologue (single row, direct grids, FMA path).
    route_in_gemv = m.route_in_gemv && build == 2 && (sync & 15) == 2 && m.moe_fma && (m.moe_stages & 2);
    route_p = fp;
    if ((sync & 15) == 2 && !(m.moe_stages & 16) && !route_in_gemv) {
      // Finalize: one simdgroup per row without barriers (route_fin1), or the
      // 256-thread kernel with threadgroup reductions.
      E.pipe(m.route_fin1 ? "moe_route_fin1" : "moe_route_fin");
      E.set(0, z.rparts);
      E.set(1, z.ids);
      E.set(2, z.w);
      E.set(3, z.sgate);
      E.set(4, z.tiles);
      E.set(5, z.ntiles);
      E.set(6, z.rowmap);
      E.set(7, z.slotpos);
      E.bytes(8, fp);
      E.rd(z.rparts);
      E.wr(z.ids);
      E.wr(z.w);
      E.wr(z.sgate);
      if (rows == 1) {
        E.wr(z.tiles);
        if (build == 1) E.wr(z.ntiles);
        E.wr(z.rowmap);
        E.wr(z.slotpos);
      }
      E.groups(MTLSizeMake(size_t(rows), 1, 1), MTLSizeMake(size_t(m.route_fin1 ? 32 : 256), 1, 1));
    }
  } else if ((m.moe_stages & 1) && fused_route) {
    const QW& r = L.router.q;
    const QW& g = L.shared_gate.q;
    const MoeRouteFusedParams fp{c.E, c.k, c.D, c.D, gu_blocks, down_blocks, rows == 1 ? 1 : 0, shared_flag};
    const std::string name = "moe_route" + std::to_string(r.bits);
    // Threadgroup: as many simdgroups as the pipeline's register budget allows
    // (a multiple of the column blocks, at least E threads for the top-k).
    const int maxt = int([m.kernel(name, r.fc()) maxTotalThreadsPerThreadgroup]);
    const int nblocks = (c.E + 31) / 32;
    int threads = std::min(1024, maxt) / (32 * nblocks) * (32 * nblocks);
    if (threads < std::max(256, 32 * nblocks)) fail("moe_route: pipeline allows only " + std::to_string(maxt) + " threads");
    E.pipe(name, r.fc());
    gemm_tag(E, name, r.out, r.in, rows);
    E.set(0, X);
    E.set(1, r.wr);
    E.set(2, r.sb);
    E.set(3, r.wh ? r.wh : r.wr);
    E.set(4, g.wr);
    E.set(5, g.sb);
    E.set(6, g.wh ? g.wh : g.wr);
    E.set(7, z.ids);
    E.set(8, z.w);
    E.set(9, z.sgate);
    E.set(10, z.tiles);
    E.set(11, z.ntiles);
    E.set(12, z.rowmap);
    E.set(13, z.slotpos);
    E.bytes(14, fp);
    E.rd(X);
    E.wr(z.ids);
    E.wr(z.w);
    E.wr(z.sgate);
    if (rows == 1) {
      E.wr(z.tiles);
      E.wr(z.ntiles);
      E.wr(z.rowmap);
      E.wr(z.slotpos);
    }
    E.groups(MTLSizeMake(size_t(rows), 1, 1), MTLSizeMake(size_t(threads), 1, 1));
  } else if (m.moe_stages & 1) {
    gemm(E, m.tune, L.shared_gate, X, 0, c.D, z.sgate, 0, 1, rows, 0);
    static const bool router_gemv = !env_on("ESPETERO_Q4_NO_ROUTER_GEMV");
    if (router_gemv && L.router_f16 && rows <= 8) {  // dense router, few rows (Qwen4-Exp): the plain GEMV
      E.pipe("q4_router_gemv");
      E.set(0, L.router_f16);
      E.set(1, X);
      E.set(2, z.logits);
      E.bytes(3, c.E);
      E.bytes(4, c.D);
      E.rd(X);
      E.wr(z.logits);
      E.groups(MTLSizeMake(size_t((c.E + 7) / 8), size_t(rows), 1), MTLSizeMake(256, 1, 1));
    } else {
      gemm(E, m.tune, L.router, X, 0, c.D, z.logits, 0, c.E, rows, 0);
    }
    const MoeRouteParams rp{c.E, c.k, rows};
    E.pipe("moe_topk");
    E.set(0, z.logits);
    E.set(1, z.ids);
    E.set(2, z.w);
    E.bytes(3, rp);
    E.rd(z.logits);
    E.wr(z.ids);
    E.wr(z.w);
    E.groups(MTLSizeMake(size_t(rows), 1, 1), MTLSizeMake(size_t((c.E + 31) / 32 * 32), 1, 1));
  }
  // Expert cache: the hook brings the routed experts into the layer's
  // pool and remaps the ids to its slots; the MoE then runs on the pool (E = S).
  const QW* xgu = &L.experts_gu;
  const QW* xdn = &L.experts_down;
  int Eeff = c.E;
  bool pooled = false;
  const ExpertPool* PV = nullptr;  // the pool (and its staged experts) the MoE reads
  if (m.moe_after_route) {
    if (const ExpertPool* P = m.moe_after_route(L, rows)) {
      xgu = &P->gu;
      xdn = &P->dn;
      Eeff = P->S + P->st_count;
      pooled = true;
      PV = P;
    }
  }
  // Expert cache: the sort maps the routed expert ids to pool slots itself (m.xmap).
  const int remap = pooled ? 1 : 0;
  if (Eeff > moe_sort_cap(c)) fail("moe_sort: " + std::to_string(Eeff) + " expert indices exceed its tables");
  const MoeSortParams sp{Eeff, c.k, rows, gu_blocks, down_blocks, shared_flag, remap};
  if ((m.moe_stages & 1) && !(fused_route && rows == 1)) {
  E.pipe("moe_sort");
  E.set(7, remap ? m.xmap : z.ids);
  if (remap) E.rd(m.xmap);
  E.set(0, z.ids);
  E.set(1, z.offsets);
  E.set(2, z.tiles);
  E.set(3, z.ntiles);
  E.set(4, z.rowmap);
  E.set(5, z.slotpos);
  E.bytes(6, sp);
  E.rd(z.ids);
  E.wr(z.offsets);
  E.wr(z.tiles);
  E.wr(z.ntiles);
  E.wr(z.rowmap);
  E.wr(z.slotpos);
  E.groups(MTLSizeMake(1, 1, 1), MTLSizeMake(1024, 1, 1));
  }
  // ESPETERO_MOE_CHECK=1: check the tile table of long pooled passes; =2: and stop there (stall diagnostic).
  static const int moe_check = [] {
    const char* v = std::getenv("ESPETERO_MOE_CHECK");
    return v && *v ? std::atoi(v) : 0;
  }();
  if (moe_check && pooled && rows > 8 && (m.moe_stages & 1))
    moe_check_tiles(m, E, z, PV, int(&L - m.layers.data()), Eeff, rows, shared_flag, moe_check);
  auto grouped = [&](const QW& w, const QW* w2, id<MTLBuffer> A, int lda, int gather, id<MTLBuffer> Y, int ldy, int Ne,
                     int epi, int part, int split = 0) {  // part: 0 gate/up, 1 down (the staged records' halves); split: the split pass
    // 1-2 rows: the FMA kernel (no MMA padding); 3-8 rows: the MMA kernel.
    const bool fma_path = m.moe_fma && rows <= m.moe_fma_rows;  // 1-2 rows by default; up to 4
    const bool staged = PV && PV->st_count > 0;
    if (staged && fma_path) fail("expert cache: staged experts need the 8-row MoE kernel");
    const int ks_opt = epi == 2 ? m.moe_ks : m.moe_ks_dn;
    const int ks = fma_path ? (ks_opt == 4 || (ks_opt == 2 && rows <= 2) ? ks_opt : 8) : 8;  // no k2 variants above 2 rows
    const bool route_name = route_in_gemv && epi == 2 && fma_path && rows == 1 && m.moe_direct;
    const std::string name = moe_gemv_name(w, w2, fma_path, rows, epi, route_name, ks);
    E.pipe(name, w.fc());
    gemm_tag(E, name, w.out, w.in, rows);
    // Several column blocks per threadgroup only on direct grids (the indirect arguments hold the block count).
    const bool direct = rows == 1 && m.moe_direct;
    const int nbpt = (fma_path && direct) ? std::max(1, std::min(epi == 2 ? m.moe_nbpt : m.moe_nbpt_dn, Ne / 32)) : 1;
    const int swap = (fma_path && direct && m.moe_swap) ? 1 : 0;
    const bool route = route_in_gemv && epi == 2 && fma_path && direct;
    MoeGemmParams gp{Ne, w.out, w.in, lda, ldy, gather, w2 ? Eeff : -1, w2 ? w2->out : 0, direct ? 1 : 0, nbpt, swap,
                     route ? route_p.nsl : 0, route_p.E, route_p.k, route_p.sync, route_p.shared};
    if (staged) {
      gp.st_first = PV->S;
      gp.st_count = PV->st_count;
      gp.st_words = int(PV->st_stride / 4);
      gp.st_half4 = int(PV->st_stride / 8);
    }
    gp.split = split;
    E.set(0, A);
    E.set(1, w.wr);
    E.set(2, w.sb);
    E.set(3, w.wh ? w.wh : w.wr);
    E.set(4, Y);
    E.bytes(5, gp);
    E.set(6, z.tiles);
    E.set(7, z.ntiles);
    E.set(8, z.rowmap);
    const QW& s2 = w2 ? *w2 : w;
    E.set(9, s2.wr);
    E.set(10, s2.sb);
    E.set(11, s2.wh ? s2.wh : s2.wr);
    E.rd(A);
    if (route) {
      E.set(12, z.rparts);
      E.set(13, z.ids);
      E.set(14, z.w);
      E.set(15, z.sgate);
      E.set(16, z.slotpos);
      E.rd(z.rparts);
      E.wr(z.ids);
      E.wr(z.w);
      E.wr(z.sgate);
      E.wr(z.tiles);
      E.wr(z.rowmap);
      E.wr(z.slotpos);
    } else {
      E.rd(z.tiles);
      E.rd(z.rowmap);
      if (!fma_path) {  // moe_gemv8's third matrix: the staged records (or placeholders)
        if (staged && split != 1) {  // (a split's resident pass never reads them: the reads may still be filling them)
          const size_t ow = m.xs.rec_o[part == 0 ? 0 : 2], os = m.xs.rec_o[part == 0 ? 1 : 3];
          E.set(12, PV->st, ow);
          E.set(13, PV->st, os);
          E.set(14, PV->st, ow);
          E.rd(PV->st.b);
        } else {
          E.set(12, w.wr);
          E.set(13, w.sb);
          E.set(14, w.wh ? w.wh : w.wr);
        }
        E.set(15, m.xld ? m.xld : z.tiles);  // read only by split passes
        if (split) E.rd(m.xld);
      }
    }
    if (!(rows == 1 && m.moe_direct)) E.rd(z.ntiles);
    E.wr(Y);
    // Grid = tiles that exist x the expert's column blocks, from the sort.
    // One row routes to exactly k (+ shared) tiles: a direct grid avoids the
    // indirect-argument fetch between dependent dispatches (measured cost:
    // see moe.md); the kernel still bounds itself by ntiles[0].
    if (direct) {
      const size_t nt = size_t(c.k + (w2 ? 1 : 0)), nb = size_t((Ne / 32 + nbpt - 1) / nbpt);
      E.groups(swap ? MTLSizeMake(nb, nt, 1) : MTLSizeMake(nt, nb, 1), MTLSizeMake(size_t(32 * ks), 1, 1));
    }
    else
      E.groups_indirect(z.ntiles, epi == 2 ? 0 : 12, MTLSizeMake(size_t(32 * ks), 1, 1));
  };
  // Split handshake: the tiles of resident experts (gate/up and down)
  // are committed now, while the SSD reads the missing ones; the copies of the
  // loaded experts and their tiles follow in the next command buffer, which
  // the commit at the next flush holds until the reads are done.
  const bool split = pooled && m.xsplit && !(m.moe_fma && rows <= m.moe_fma_rows) && (m.moe_stages & 6) == 6;
  if (split) {
    grouped(*xgu, shared_tiles ? &L.gate_up.q : nullptr, X, c.D, 1, z.act, c.Ie, 2 * c.Ie, 2, 0, 1);
    grouped(*xdn, shared_tiles ? &L.down.q : nullptr, z.act, c.Ie, 0, z.ys, c.D, c.D, 0, 1, 1);
    ++m.xstats.splits;
    m.xsplit_cb = m.xsplit_last = E.submit();
    for (auto& f : m.xdefer) f(E.e);
    m.xdefer.clear();
    E.barrier();
    grouped(*xgu, shared_tiles ? &L.gate_up.q : nullptr, X, c.D, 1, z.act, c.Ie, 2 * c.Ie, 2, 0, 2);
  } else {
    if (!m.xdefer.empty()) {  // copies deferred for a split this pass does not make
      for (auto& f : m.xdefer) f(E.e);
      m.xdefer.clear();
      E.barrier();
    }
    if (m.moe_stages & 2) grouped(*xgu, shared_tiles ? &L.gate_up.q : nullptr, X, c.D, 1, z.act, c.Ie, 2 * c.Ie, 2, 0);
  }
  // Single-row passes: the down projection and the combine in one kernel.
  const int comb_ks = m.moe_comb_ks == 1 ? 1 : 2;
  // (The 64-block limit is the fused next norm's; without it, e.g. Qwen4-Exp's D = 2560 fits.)
  const bool down_comb = m.moe_down_comb && rows == 1 && m.moe_direct && m.moe_fma && shared_tiles && c.k <= 16 &&
                         c.D % 32 == 0 && (c.D / 32 <= 64 || !nout) && c.Ie % 64 == 0 && (m.moe_stages & 12) == 12 &&
                         (c.k + 1) * comb_ks * 32 <= 1024;
  if (down_comb) {
    const QW& w = *xdn;
    const QW& w2 = L.down.q;
    const std::string name = moe_down_comb_name(w, w2, comb_ks);
    E.pipe(name, w.fc());
    gemm_tag(E, name, c.D, c.Ie, rows);
    const MoeDownCombParams dp{c.D, w.out, w2.out, c.Ie, c.k, nout ? 1 : 0, nout ? nout->eps : 0.0f};
    E.set(0, z.act);
    E.set(1, w.wr);
    E.set(2, w.sb);
    E.set(3, w.wh ? w.wh : w.wr);
    E.set(4, H, h_off);
    E.bytes(5, dp);
    E.set(6, z.tiles);
    E.set(7, z.w);
    E.set(8, z.sgate);
    E.set(9, w2.wr);
    E.set(10, w2.sb);
    E.set(11, w2.wh ? w2.wh : w2.wr);
    E.set(12, nout ? nout->w : Buf(H));
    E.set(13, nout ? nout->xn : Buf(H));
    E.set(14, z.cparts);
    E.set(15, z.cctr);
    E.set(16, z.slotpos);
    E.rd(z.act);
    E.rd(z.tiles);
    E.rd(z.w);
    E.rd(z.sgate);
    E.rd(z.slotpos);
    E.rd(H, h_off, h_off + size_t(c.D) * 4);
    E.wr(H, h_off, h_off + size_t(c.D) * 4);
    E.wr(z.cparts);
    E.rd(z.cctr);
    E.wr(z.cctr);
    if (nout) E.wr(nout->xn.b, nout->xn.off, nout->xn.off + size_t(c.D) * 2);
    E.groups(MTLSizeMake(size_t(c.D / 32), 1, 1), MTLSizeMake(size_t((c.k + 1) * comb_ks * 32), 1, 1));
    return;
  }
  if (m.moe_stages & 4) grouped(*xdn, shared_tiles ? &L.down.q : nullptr, z.act, c.Ie, 0, z.ys, c.D, c.D, 0, 1, split ? 2 : 0);
  if (!(m.moe_stages & 8)) return;
  // Column slices of 512 (128 threads x float4): several threadgroups per row.
  const int slice = 512;
  const int nsl = std::min(64, (c.D + slice - 1) / slice);
  const MoeCombineParams cp{c.D, c.k, nout ? 1 : 0, nout ? nout->eps : 0.0f, (c.D + nsl - 1) / nsl / 4 * 4};
  E.pipe("moe_combine");
  E.set(0, H, h_off);
  E.set(1, z.ys);
  E.set(2, z.slotpos);
  E.set(3, z.w);
  // The shared expert's rows: after the pairs in ys when it ran as tiles.
  if (shared_tiles) E.set(4, z.ys, size_t(rows) * size_t(c.k) * size_t(c.D) * 4);
  else E.set(4, z.shared);
  E.set(5, z.sgate);
  E.bytes(6, cp);
  E.set(7, nout ? nout->w : Buf(H));
  E.set(8, nout ? nout->xn : Buf(H));
  E.rd(z.ys);
  E.rd(z.slotpos);
  E.rd(z.w);
  E.rd(z.shared);
  E.rd(z.sgate);
  E.set(9, z.cparts);
  E.set(10, z.cctr);
  E.rd(H, h_off, h_off + size_t(rows) * size_t(c.D) * 4);
  E.wr(H, h_off, h_off + size_t(rows) * size_t(c.D) * 4);
  E.wr(z.cparts);
  E.rd(z.cctr);
  E.wr(z.cctr);
  if (nout) E.wr(nout->xn.b, nout->xn.off, nout->xn.off + size_t(rows) * size_t(c.D) * 2);
  E.groups(MTLSizeMake(size_t(rows), size_t(nsl), 1), MTLSizeMake(128, 1, 1));
}

// Per-pass memory for metadata and small GPU scratch: a bump allocator over
// shared buffers, reset at the start of every pass. Every ensure() returns a
// fresh region, so a buffer already bound by an encoded dispatch is never
// overwritten or freed before the command buffer runs.
struct PassArena {
  std::vector<id<MTLBuffer>> chunks;
  size_t chunk = 0, used = 0;
  void reset() { chunk = 0, used = 0; }
  Buf alloc(Model& m, size_t bytes) {
    bytes = align_up(std::max<size_t>(bytes, 16), 256);
    while (true) {
      if (chunk < chunks.size()) {
        if (used + bytes <= size_t([chunks[chunk] length])) {
          Buf b{chunks[chunk], used};
          used += bytes;
          return b;
        }
        ++chunk;
        used = 0;
        continue;
      }
      chunks.push_back(m.buffer(std::max(bytes, size_t(32) << 20)));
    }
  }
};

struct HostBuf {
  PassArena* arena = nullptr;
  Buf buf;
  size_t bytes = 0;
  template <class T>
  T* ensure(Model& m, size_t count) {
    bytes = std::max<size_t>(count * sizeof(T), 16);
    buf = arena->alloc(m, bytes);
    return static_cast<T*>(ptr(buf));
  }
};

// Kernel-side records (must match the Metal structs).
struct DeltaSeg {
  int32_t row0, len, in_slot, out_slot, mid_slot, save_t, write_state, pad;
};
struct ConvSave {
  int32_t row0, T, in_slot, out_slot;
};
struct AttnTreeRec {
  int32_t base, P, table, M;
};
struct AttnTileRec {
  int32_t tree, r0, k0, k1, part, last, pad0, pad1;
};
struct AttnCombineRec {
  int32_t r0, rows, part0, nparts;
};
struct KVSaveRec {
  int32_t src_row, block, pos, pad;
};
struct ConvParams {
  int32_t C, ldx, a_off, Hk, Hv;
  float eps;
  uint64_t conv_base, conv_stride;
};
struct DeltaParams {
  int32_t Hk, Hv, C, tiled;
  uint64_t st_base, st_stride;
};
struct AttnPrepParams {
  int32_t H, Hkv, ld, k_off, q_ld, kv_ld;
  float eps;
};
struct DeltaFusedParams {
  int32_t Hk, Hv, C, ldx, a_off, tiled;
  float eps;
  int32_t z_off, yn_ld;
  uint64_t conv_base, conv_stride, st_base, st_stride;
  int32_t conv_save;  // fused conv state save job
};
struct AttnParams {
  int32_t H, Hkv, q_ld, kv_ld, g_ld, bt;
  uint64_t kv_base;
};
struct QsaParams {
  int32_t ld, bt, iblk, pool_off, pos_off, nbmax, nbw;
  float eps;
  uint64_t ibase;
};
struct QsaTileRec {
  int32_t tree, r0, rows, pad;
};

// Attention work of one set of trees (the decoder, or the MTP layer).
struct AttnPlan {
  std::vector<AttnTreeRec> trees;
  std::vector<AttnTileRec> tiles;
  std::vector<AttnCombineRec> combine;
  int parts = 0;
};

// Tiles of 8 query rows per tree; cached contexts split into tiles of at
// most `split` keys (flash decoding) when there are too few tiles to fill
// the GPU: a decode step at a 512-token context would otherwise run Hkv
// threadgroups per layer. The chunk shrinks down to one 64-token block so a
// few-tile pass reaches ~256 threadgroups; passes with many tiles (prefill)
// do not split.
//
// Qwen sparse attention: an 8-row group with rows past the dense range
// (qt[tree][group] >= 0) reads the cached blocks of its union list (built on
// the GPU, qsa_tiles) instead of a key range: k0/k1 are list entries (4 keys
// each; the last part is open-ended since the list length is known only on
// the GPU), pad0 = 1, pad1 = the list. Splits assume 513 entries for one row
// and 128 more per further row.
inline void plan_tiles(AttnPlan& A, int Hkv, int split, const std::vector<std::vector<int>>* qt = nullptr, int groups = 256) {
  A.tiles.clear();
  A.combine.clear();
  A.parts = 0;
  int n_tiles = 0;
  for (const auto& t : A.trees) n_tiles += (t.M + 7) / 8;
  const int budget = std::max(1, groups / std::max(1, Hkv * n_tiles));  // splits per tile to reach ~`groups` threadgroups
  for (int ti = 0; ti < int(A.trees.size()); ++ti) {
    const AttnTreeRec& t = A.trees[size_t(ti)];
    int ns = 1;
    if (split > 0 && budget > 1 && t.P >= 64) {
      const int chunk = std::max(32, std::min(split, ((t.P + budget - 1) / budget + 31) / 32 * 32));
      ns = std::min({(t.P + chunk - 1) / chunk, budget, 64});
    }
    for (int r = 0; r < t.M; r += 8) {
      const int q = qt ? (*qt)[size_t(ti)][size_t(r / 8)] : -1;
      if (q >= 0) {
        const int rows = std::min(8, t.M - r);
        const int est = 516 + 128 * (rows - 1);
        int nq = 1;
        if (split > 0 && budget > 1) {
          const int chunk = std::max(16, std::min(split / 4, ((est + budget - 1) / budget + 3) / 4 * 4));
          nq = std::min({(est + chunk - 1) / chunk, budget, 64});
        }
        if (nq == 1) {
          A.tiles.push_back({ti, t.base + r, 0, 1 << 30, -1, 1, 1, q});
          continue;
        }
        const int step = ((est + nq - 1) / nq + 3) / 4 * 4;
        const int part0 = A.parts;
        for (int k = 0; k < nq; ++k) {
          const bool lastp = k + 1 == nq;
          A.tiles.push_back({ti, t.base + r, k * step, lastp ? (1 << 30) : (k + 1) * step, A.parts++, lastp ? 1 : 0, 1, q});
        }
        A.combine.push_back({t.base + r, rows, part0, nq});
        continue;
      }
      if (ns == 1) {
        A.tiles.push_back({ti, t.base + r, 0, t.P, -1, 1, 0, 0});
        continue;
      }
      // Near-equal key ranges, boundaries on 32-key multiples (the kernel
      // walks 16-key blocks inside 64-token KV blocks).
      const int per = (t.P + ns - 1) / ns;
      const int step = (per + 31) / 32 * 32;
      const int part0 = A.parts;
      int k = 0, used = 0;
      for (; k < t.P; k += step, ++used) {
        const int k1 = std::min(t.P, k + step);
        A.tiles.push_back({ti, t.base + r, k, k1, A.parts++, k1 >= t.P ? 1 : 0, 0, 0});
      }
      A.combine.push_back({t.base + r, std::min(8, t.M - r), part0, used});
    }
  }
}

// Copy-on-write of shared KV blocks: every layer region of the KV pool and,
// for Qwen4-Exp, of the sparse attention index (as blocks of 2 * bt * 84
// halves). pairs = (src block, dst block).
inline void kv_cow_copy(Model& m, Enc& E, const Buf& pairs, size_t n) {
  const Config& c = m.cfg;
  const KVPool& K = m.kv;
  // The pool's own width (Gemma 4: the global layers', not c.kv_dim()).
  AttnParams ap{c.H, c.Hkv, c.H * c.HD, K.kv_dim, c.full_ld(), K.bt, 0};
  const uint64_t region = K.region_halves;
  E.pipe("kv_copy_blocks");
  E.set(0, K.buf);
  E.set(1, pairs);
  E.bytes(2, ap);
  E.bytes(3, region);
  E.threads(MTLSizeMake(K.block_halves() / 8, n, size_t(K.n_layers)), MTLSizeMake(64, 1, 1));
  if (K.i_layers) {
    AttnParams ai{0, 0, 0, int(K.iblock_halves() / (2 * size_t(K.bt))), 0, K.bt, 0};
    const uint64_t iregion = K.iregion_halves();
    E.pipe("kv_copy_blocks");
    E.set(0, K.ibuf);
    E.set(1, pairs);
    E.bytes(2, ai);
    E.bytes(3, iregion);
    E.threads(MTLSizeMake(K.iblock_halves() / 8, n, size_t(K.i_layers)), MTLSizeMake(64, 1, 1));
  }
}

// ---- Qwen sparse attention (docs/qwen4_exp.md 4): the plan and the
// encoders of one set of rows (a pass's decoder rows, its MTP rows, one step
// of the MTP draft loop). Per row its sequence index s (cached keys, then its
// path: the root rows, then its own segment) and nb = (s + 1) / 4 complete
// blocks; rows with nb > 512 are sparse. The index (raw and pooled keys,
// positions) is kept for every row so a context can cross 2051 tokens later.
constexpr int kQsaLmax = 4112;  // union entries per tile (QSA_LMAX)
struct QsaSet {
  int M = 0, n_sparse = 0, nbmax = 0, nbb = 0, ncb = 0;
  std::vector<std::vector<int>> qt_of;  // per tree, per 8-row group: its tile or -1
  std::vector<QsaTileRec> tiles;
  Buf rows, srows, qtiles;              // per row (s, nb, sparse row, tree); sparse rows; tiles
  Buf iqk, iq, ik, ip, sc, sel, tl, tc;
  size_t n_iqk = 0, n_iq = 0, n_ik = 0, n_sc = 0, n_sel = 0, n_tl = 0, n_tc = 0;  // bytes (hazard spans)
  int nbw() const { return (nbmax + 31) / 32; }
};

template <class T>
inline Buf arena_upload(Model& m, PassArena& arena, const std::vector<T>& v) {
  Buf b = arena.alloc(m, std::max<size_t>(v.size() * sizeof(T), 16));
  if (!v.empty()) std::memcpy(ptr(b), v.data(), v.size() * sizeof(T));
  return b;
}

// desc: per row (segment start, 0, root rows, 0) relative to its tree, or
// null for chains. sparse = false: index only (no attention over these rows).
inline void qsa_plan(Model& m, PassArena& arena, QsaSet& q, const std::vector<AttnTreeRec>& trees, const int32_t* desc,
                     bool sparse) {
  const Config& c = m.cfg;
  q = QsaSet{};
  for (const AttnTreeRec& t : trees) q.M = std::max(q.M, t.base + t.M);
  std::vector<int32_t> rows(size_t(q.M) * 4, 0), srows;
  q.qt_of.resize(trees.size());
  for (size_t ti = 0; ti < trees.size(); ++ti) {
    const AttnTreeRec& T = trees[ti];
    q.qt_of[ti].assign(size_t((T.M + 7) / 8), -1);
    for (int rr = 0; rr < T.M; ++rr) {
      const int r = T.base + rr;
      const int sidx = T.P + (desc ? desc[r * 4 + 2] + (rr - desc[r * 4 + 0]) : rr);
      const int nb = (sidx + 1) / 4;
      const bool sp = sparse && m.qsa && nb > 512;
      rows[size_t(r) * 4 + 0] = sidx;
      rows[size_t(r) * 4 + 1] = nb;
      rows[size_t(r) * 4 + 2] = sp ? q.n_sparse : -1;
      rows[size_t(r) * 4 + 3] = int(ti);
      if (!sp) continue;
      srows.push_back(r);
      ++q.n_sparse;
      q.nbmax = std::max(q.nbmax, nb);
      q.nbb = std::max(q.nbb, nb - T.P / 4);
      q.ncb = std::max(q.ncb, T.P / 4);
      int& g = q.qt_of[ti][size_t(rr / 8)];
      if (g < 0) {
        g = int(q.tiles.size());
        q.tiles.push_back({int(ti), T.base + rr / 8 * 8, std::min(8, T.M - rr / 8 * 8), 0});
      }
    }
  }
  q.rows = arena_upload(m, arena, rows);
  q.srows = arena_upload(m, arena, srows);
  q.qtiles = arena_upload(m, arena, q.tiles);
  const size_t R = size_t(std::max(q.M, 1));
  q.n_iqk = R * size_t((c.idx_heads + c.idx_kv) * c.idx_hd) * 4;
  q.n_iq = R * size_t(c.idx_heads * c.idx_hd) * 2;
  q.n_ik = R * size_t(c.idx_hd) * 2;
  q.iqk = arena.alloc(m, q.n_iqk);
  q.iq = arena.alloc(m, q.n_iq);
  q.ik = arena.alloc(m, q.n_ik);
  q.ip = arena.alloc(m, q.n_ik);
  if (q.n_sparse) {
    q.n_sc = size_t(q.n_sparse) * size_t(q.nbmax) * 4;
    q.n_sel = size_t(q.n_sparse) * size_t(q.nbw()) * 4;
    q.n_tl = q.tiles.size() * size_t(kQsaLmax) * 4;
    q.n_tc = q.tiles.size() * 4;
    q.sc = arena.alloc(m, q.n_sc);
    q.sel = arena.alloc(m, q.n_sel);
    q.tl = arena.alloc(m, q.n_tl);
    q.tc = arena.alloc(m, q.n_tc);
  }
}

inline QsaParams qsa_params(const Model& m, const QsaSet& q, int layer) {
  const Config& c = m.cfg;
  const KVPool& K = m.kv;
  return QsaParams{(c.idx_heads + c.idx_kv) * c.idx_hd, K.bt, int(K.iblock_halves()), int(K.ipool_off()),
                   int(K.ipos_off()), std::max(q.nbmax, 1), std::max(q.nbw(), 1), c.eps, uint64_t(K.ilayer_base(layer))};
}

// The index of every row: the indexer projection of x, the queries and raw
// keys, the pooled keys of the blocks the rows complete. track: earlier
// dispatches of this command buffer wrote the index pool (draft steps).
inline void qsa_index(Model& m, Enc& E, const QsaSet& q, const Layer& L, id<MTLBuffer> x, const Buf& pos,
                      const Buf& trees, const Buf& desc, const Buf& blocks, int layer, bool track) {
  const Config& c = m.cfg;
  const KVPool& K = m.kv;
  const QsaParams qp = qsa_params(m, q, layer);
  gemm(E, m.tune, L.index_qk, x, 0, c.D, q.iqk.b, q.iqk.off, qp.ld, q.M, 0);
  E.pipe("qsa_prep");
  E.set(0, q.iqk);
  E.set(1, L.iq_norm);
  E.set(2, pos);
  E.set(3, m.inv_freq);
  E.set(4, m.rope_sel);
  E.set(5, q.iq);
  E.set(6, q.ik);
  E.bytes(7, qp);
  E.rd(q.iqk, q.n_iqk);
  E.wr(q.iq, q.n_iq);
  E.wr(q.ik, q.n_ik);
  E.groups(MTLSizeMake(1, size_t(q.M), 1), MTLSizeMake(32 * size_t(c.idx_heads + c.idx_kv), 1, 1));
  E.pipe("qsa_pool");
  E.set(0, q.ik);
  E.set(1, K.ibuf);
  E.set(2, blocks);
  E.set(3, trees);
  E.set(4, desc);
  E.set(5, q.rows);
  E.set(6, pos);
  E.set(7, L.ik_norm);
  E.set(8, m.inv_freq);
  E.set(9, m.rope_sel);
  E.set(10, q.ip);
  E.bytes(11, qp);
  E.rd(q.ik, q.n_ik);
  if (track) E.rd(K.ibuf);
  E.wr(q.ip, q.n_ik);
  E.groups(MTLSizeMake(1, size_t(q.M), 1), MTLSizeMake(32, 1, 1));
}

// Scores, top-512 bitmaps and the tiles' union lists of the sparse rows.
inline void qsa_select(Model& m, Enc& E, const QsaSet& q, const Buf& trees, const Buf& desc, const Buf& blocks, int layer,
                       bool track) {
  const KVPool& K = m.kv;
  const QsaParams qp = qsa_params(m, q, layer);
  if (q.ncb > 0) {
    E.pipe("qsa_score_cached");
    E.set(0, q.iq);
    E.set(1, K.ibuf);
    E.set(2, blocks);
    E.set(3, trees);
    E.set(4, q.rows);
    E.set(5, q.qtiles);
    E.set(6, q.sc);
    E.bytes(7, qp);
    E.rd(q.iq, q.n_iq);
    if (track) E.rd(K.ibuf);
    E.wr(q.sc, q.n_sc);
    E.groups(MTLSizeMake(size_t((q.ncb + 31) / 32), q.tiles.size(), 1), MTLSizeMake(128, 1, 1));
  }
  if (q.nbb > 0) {
    E.pipe("qsa_score_batch");
    E.set(0, q.iq);
    E.set(1, q.ip);
    E.set(2, trees);
    E.set(3, desc);
    E.set(4, q.rows);
    E.set(5, q.sc);
    E.bytes(6, qp);
    E.rd(q.iq, q.n_iq);
    E.rd(q.ip, q.n_ik);
    E.wr(q.sc, q.n_sc);
    E.groups(MTLSizeMake(size_t(q.nbb), size_t(q.M), 1), MTLSizeMake(32, 1, 1));
  }
  if ([m.kernel("qsa_select") maxTotalThreadsPerThreadgroup] < 1024) fail("qsa_select: threadgroup too small");
  E.pipe("qsa_select");
  E.set(0, q.sc);
  E.set(1, q.srows);
  E.set(2, q.rows);
  E.set(3, q.sel);
  E.bytes(4, qp);
  E.rd(q.sc, q.n_sc);
  E.wr(q.sel, q.n_sel);
  E.groups(MTLSizeMake(size_t(q.n_sparse), 1, 1), MTLSizeMake(1024, 1, 1));
  E.pipe("qsa_tiles");
  E.set(0, q.sel);
  E.set(1, trees);
  E.set(2, q.rows);
  E.set(3, q.qtiles);
  E.set(4, q.tl);
  E.set(5, q.tc);
  E.bytes(6, qp);
  E.rd(q.sel, q.n_sel);
  E.wr(q.tl, q.n_tl);
  E.wr(q.tc, q.n_tc);
  E.groups(MTLSizeMake(q.tiles.size(), 1, 1), MTLSizeMake(256, 1, 1));
}

// The extra buffers of flash_attention_qsa (12-16).
inline void qsa_bind(Model& m, Enc& E, const QsaSet& q, int layer) {
  E.set(12, q.tl);
  E.set(13, q.tc);
  E.set(14, q.rows);
  E.set(15, q.sel);
  E.bytes(16, qsa_params(m, q, layer));
  E.rd(q.tl, q.n_tl);
  E.rd(q.tc, q.n_tc);
  E.rd(q.sel, q.n_sel);
}

// Raw keys, pooled keys and positions of the saved rows into the index pool.
inline void qsa_save(Model& m, Enc& E, const QsaSet& q, const Buf& pos, const Buf& saves, size_t n, int layer, bool track) {
  if (!n) return;
  const KVPool& K = m.kv;
  E.pipe("qsa_save");
  E.set(0, q.ik);
  E.set(1, q.ip);
  E.set(2, pos);
  E.set(3, K.ibuf);
  E.set(4, saves);
  E.bytes(5, qsa_params(m, q, layer));
  E.rd(q.ik, q.n_ik);
  E.rd(q.ip, q.n_ik);
  if (track) E.wr(K.ibuf);
  E.threads(MTLSizeMake(16, n, 1), MTLSizeMake(16, 4, 1));
}

struct PassMeta {
  PassArena arena;
  HostBuf tokens{&arena};
  HostBuf pos{&arena};
  HostBuf conv_src{&arena};
  HostBuf row_slot{&arena};
  HostBuf desc{&arena};
  HostBuf zero_desc{&arena};
  HostBuf segs_root{&arena};
  HostBuf segs_leaf{&arena};
  HostBuf conv_jobs{&arena};
  HostBuf conv_root{&arena};  // the conv save of each root segment (T = 0: none)
  HostBuf a_trees{&arena};
  HostBuf a_tiles{&arena};
  HostBuf a_comb{&arena};
  HostBuf kv_saves{&arena};
  HostBuf mtp_kv_saves{&arena};
  HostBuf blocks{&arena};
  HostBuf cow{&arena};
  HostBuf readout_rows{&arena};
  HostBuf topk_it{&arena};
  HostBuf topk_samp{&arena};
  HostBuf cand_ids{&arena};
  HostBuf cand_owner{&arena};
  HostBuf cand_out{&arena};
  HostBuf parts{&arena};
  HostBuf m_tok{&arena};
  HostBuf m_pos{&arena};
  HostBuf m_hin_off{&arena};
  HostBuf m_hin_rows{&arena};
  HostBuf m_trees{&arena};
  HostBuf m_tiles{&arena};
  HostBuf m_comb{&arena};
  HostBuf m_rows{&arena};
  HostBuf m_offs{&arena};
  HostBuf m_x{&arena};
  HostBuf m_emb{&arena};
  HostBuf m_hin{&arena};
  HostBuf m_hn{&arena};
  HostBuf m_logits{&arena};
  HostBuf m_arg{&arena};
  HostBuf m_arg2{&arena};  // runner-ups of the draft steps
  HostBuf m_bmax{&arena};  // coarse draft head: block maxima [nq][blocks]
  HostBuf m_blist{&arena}; // coarse draft head: shortlisted blocks [nq][coarse_c]
  HostBuf m_rows2{&arena};
  HostBuf m_offs2{&arena};
  HostBuf m_hc{&arena};    // Qwen4-Exp drafts: streams of the rows read out [nq][W]
  HostBuf cm_rows{&arena}; // pass rows of the commit chains, in Model::cm_x order
  // Gemma 4: the sliding layers' attention plan (the global layers' uses
  // a_trees / a_tiles / a_comb), ring saves and ring copies.
  HostBuf s_trees{&arena};
  HostBuf s_tiles{&arena};
  HostBuf s_comb{&arena};
  HostBuf ring_saves{&arena};
  HostBuf ring_copies{&arena};
};

// One tree of a forest (host-side plan).
struct TreePlan {
  const TreeSpec* spec = nullptr;
  Slot* parent = nullptr;
  int parent_slot = -1;   // user slot index or -1
  int pstate = 0;         // state slot read by the root (zero slot for an empty context)
  Slot* dst = nullptr;
  Slot* work = nullptr;
  int dst_slot = -1, work_slot = -1;
  int P = 0, Tr = 0, save_len = 0, NL = 0, N = 0, M = 0;
  int base = 0, readout_base = 0, cand_base = 0, n_cands = 0;
  int next_root = 0, next_save = 0;
  int root_final = -1;    // state slot holding the root's final state (leaves read it)
  bool mtp = false, mtp_run = false;
  int mtp_r0 = 0, mtp_R = 0, n_lh = 0;
  int cm_k = 0, cm_row0 = 0, mtp_C = 0;  // commit chain rows, their first Model::cm_x row, MTP rows for them
  int read_table = 0;     // offset of the parent's block list in the pass table
  std::vector<int> host, off;
  std::vector<int> dst_blocks, work_blocks;  // new block lists after the pass
};

struct Forest {
  Model& m;
  PassMeta& meta;
  explicit Forest(Model& model, PassMeta& pm) : m(model), meta(pm) {}

  // Prepare the new block list of a target that continues `parent` (whose
  // list covers [0, P)) and writes positions [w0, end): blocks fully below
  // w0 are shared, the rest are new; the first new block copies the shared
  // one it replaces when it holds positions below w0 (copy on write).
  // In-place appends (the target is the parent) keep a partial block they
  // own alone instead of copying it: the pass writes positions >= w0 there
  // while attention reads positions < P, so the two never overlap.
  std::vector<int> target_blocks(const std::vector<int>& parent_blocks, int P, int w0, int end,
                                 std::vector<std::pair<int, int>>& cow, bool in_place) {
    const int bt = m.kv.bt;
    std::vector<int> out;
    const int keep = w0 / bt;  // blocks entirely below w0
    for (int i = 0; i < keep && i < int(parent_blocks.size()); ++i) {
      out.push_back(parent_blocks[size_t(i)]);
      m.kv_retain(parent_blocks[size_t(i)]);
    }
    const int need = (end + bt - 1) / bt;
    for (int i = keep; i < need; ++i) {
      // Every parent position below P is valid in every layer region: w0 < P
      // (MTP: the pass rewrites the MTP entry at P - 1) must not drop the
      // decoder's K/V at P - 1 when it opens a block.
      const bool has_parent = i < int(parent_blocks.size()) && i * bt < P;
      if (has_parent && in_place && m.kv.ref[size_t(parent_blocks[size_t(i)])] == 1) {
        out.push_back(parent_blocks[size_t(i)]);
        m.kv_retain(parent_blocks[size_t(i)]);
        continue;
      }
      const int nb = m.kv_alloc();
      // The block also holds valid parent positions below min(P, w0).
      if (has_parent) cow.push_back({parent_blocks[size_t(i)], nb});
      out.push_back(nb);
    }
    return out;
  }
};

// The blocks Forest::target_blocks allocates for that target: every
// block from w0 / bt up to `end`, less the partial block an in-place append
// keeps because it owns it alone (only the block holding w0 can be one).
// run_forest's check and NativeModel::pass_kv_tokens count with it. Counting
// that block too charged a decode row one block at every step, though its
// row opens one only every bt steps: a request alone stopped up to bt - 1
// tokens before the cache was full.
inline int target_new_blocks(const Model& m, const Slot* parent, int P, int w0, int end, bool in_place) {
  const int bt = m.kv.bt;
  const int i = w0 / bt;
  int n = (end + bt - 1) / bt - i;
  if (n > 0 && in_place && parent && i < int(parent->blocks.size()) && i * bt < P &&
      m.kv.ref[size_t(parent->blocks[size_t(i)])] == 1)
    --n;
  return std::max(n, 0);
}

// Qwen4-Exp per-pass scratch (qwen4 path of run_forest).
inline void ensure_q4_scratch(Model& m, int rows) {
  Model::Q4Scratch& q = m.q4s;
  const Config& c = m.cfg;
  if (rows <= q.rows) return;
  const int R = std::max(rows, std::max(64, q.rows * 3 / 2));
  const size_t W = size_t(c.W()), D = size_t(c.D), S = size_t(c.hc), r = size_t(c.hc_rank);
  auto f = [&](size_t n) { return m.buffer(n * 4); };
  auto hf = [&](size_t n) { return m.buffer(n * 2); };
  q.H = f(size_t(R) * W);
  q.N16 = hf(size_t(R) * W);
  q.dn = f(size_t(R) * (r + S));
  q.injr = f(size_t(R) * S);
  q.inj = f(size_t(R) * S);
  q.a16 = hf(size_t(R) * r);
  q.mm = f(size_t(R) * W);
  q.y = f(size_t(R) * D);
  q.e16 = hf(size_t(R) * D);
  q.kraw = f(size_t(R) * W);
  q.vp = f(size_t(R) * D);
  q.g = f(size_t(R) * S);
  q.G = f(size_t(R) * W);
  q.Gn = f(size_t(R) * W);
  q.psrc = f(size_t(R) * 3);
  q.pslot = f(size_t(R));
  q.Hc = f(size_t(R) * W);
  q.pfx = hf(size_t(R) * D);
  q.pflog = f(size_t(R) * size_t(c.E));
  q.pfids = f(size_t(R) * 16);
  q.pfw = f(size_t(R) * 16);
  q.rows = R;
}

// ---- Qwen4-Exp hyper-connection helpers (run_forest and the MTP draft loop)
// The block input of the streams X [rows][W] into xout [rows][D] (fp16):
// grouped stream norm, the low-rank read (down), gates, up, the gated mix.
// With `inject`, the block's injection weights go to q4s.inj.
inline void q4_hc_read(Model& m, Enc& E, const HCBlock& B, bool inject, const Buf& X, int rows, const Buf& xout) {
  const Config& c = m.cfg;
  Model::Q4Scratch& q = m.q4s;
  const int W = c.W(), R = c.hc_rank, S4 = c.hc;
  E.pipe("q4_stream_norm");
  E.set(0, X);
  E.set(1, B.norm);
  E.set(2, q.N16);
  E.bytes(3, c.D);
  E.bytes(4, c.eps);
  E.bytes(5, W);
  E.bytes(6, W);
  E.set(7, q.N16);
  const int o32 = 0;
  E.bytes(8, o32);
  E.rd(X);
  E.wr(q.N16);
  E.groups(MTLSizeMake(size_t(S4), size_t(rows), 1), MTLSizeMake(256, 1, 1));
  const int dn_ld = B.down.out();
  gemm(E, m.tune, B.down, q.N16, 0, W, q.dn, 0, dn_ld, rows, 0);
  int mode = 2;
  if (inject) {
    mode = B.inj_fused ? 1 : 0;
    if (!B.inj_fused) gemm(E, m.tune, B.inj, q.N16, 0, W, q.injr, 0, S4, rows, 0);
  }
  E.pipe("q4_hc_gates");
  E.set(0, q.dn);
  E.set(1, q.a16);
  E.set(2, q.injr);
  E.set(3, q.inj);
  E.bytes(4, R);
  E.bytes(5, dn_ld);
  E.bytes(6, S4);
  E.bytes(7, mode);
  E.rd(q.dn);
  E.rd(q.injr);
  E.wr(q.a16);
  E.wr(q.inj);
  E.threads(MTLSizeMake(size_t(R + S4), size_t(rows), 1), MTLSizeMake(64, 1, 1));
  gemm(E, m.tune, B.up, q.a16, 0, R, q.mm, 0, W, rows, 0);
  E.pipe("q4_hc_mix");
  E.set(0, q.mm);
  E.set(1, q.N16);
  E.set(2, xout);
  E.bytes(3, c.D);
  E.bytes(4, S4);
  E.rd(q.mm);
  E.rd(q.N16);
  E.wr(xout);
  E.threads(MTLSizeMake(size_t(c.D / 4), size_t(rows), 1), MTLSizeMake(64, 1, 1));
}

// H [rows][W] += inj[row][s] * y [rows][D] for every stream s.
inline void q4_hc_inject(Model& m, Enc& E, const Buf& H, const Buf& y, const Buf& inj, int rows) {
  const Config& c = m.cfg;
  E.pipe("q4_hc_inject");
  E.set(0, H);
  E.set(1, y);
  E.set(2, inj);
  E.bytes(3, c.D);
  E.bytes(4, c.hc);
  E.rd(H);
  E.rd(y);
  E.rd(inj);
  E.wr(H);
  E.threads(MTLSizeMake(size_t(c.D / 4), size_t(rows), 1), MTLSizeMake(64, 1, 1));
}

// One RMS-normalized group of `d` values per row (weight w, eps): fp32 rows of
// stride `ld` in X to fp16 rows of stride `ld` in Y (the q4 stream norm with one group).
inline void q4_norm_rows(Model& m, Enc& E, const Buf& X, const Buf& w, const Buf& Y, int d, int rows) {
  E.pipe("q4_stream_norm");
  E.set(0, X);
  E.set(1, w);
  E.set(2, Y);
  E.bytes(3, d);
  E.bytes(4, m.cfg.eps);
  E.bytes(5, d);
  E.bytes(6, d);
  E.set(7, Y);
  const int o32 = 0;
  E.bytes(8, o32);
  E.rd(X);
  E.wr(Y);
  E.groups(MTLSizeMake(1, size_t(rows), 1), MTLSizeMake(256, 1, 1));
}

// The Qwen4-Exp MTP layer's input streams (docs/qwen4_exp.md 7) for R rows:
// U_s = fc_emb . norm_e(emb) + fc_hid . norm_h(H)_s with norm_h one RMS over all
// W values. emb [R][D] fp32, hin [R][W] fp32 -> streams [R][W] fp32.
// Scratch: e16 [R][D] fp16, h16 [R][W] fp16, eproj [R][D] fp32, ones [R][hc] (1.0).
inline void q4_mtp_streams(Model& m, Enc& E, const Buf& emb, const Buf& hin, const Buf& streams, const Buf& e16,
                           const Buf& h16, const Buf& eproj, const Buf& ones, int R) {
  const Config& c = m.cfg;
  const Mtp& MT = m.mtp;
  q4_norm_rows(m, E, emb, MT.enorm, e16, c.D, R);
  q4_norm_rows(m, E, hin, MT.hnorm, h16, c.W(), R);
  gemm(E, m.tune, MT.fc_emb, e16.b, e16.off, c.D, eproj.b, eproj.off, c.D, R, 0);
  gemm(E, m.tune, MT.fc_hid, h16.b, h16.off, c.D, streams.b, streams.off, c.D, R * c.hc, 0);  // [R * hc] rows of D
  q4_hc_inject(m, E, streams, eproj, ones, R);
}

// The arrival counters of the last-arriver reductions (the split-K
// GEMV, the fused norms, the MoE combine and parallel route, the fused
// DeltaNet conv fold) reset themselves as their dispatches complete. A
// command buffer that failed or was aborted part way (macOS aborts long ones
// while the display is in use) can leave some of them counting, and every
// later reduction on them would then fire early: silently wrong outputs
// until a restart (only norm_ctr was zeroed before each pass). A failed
// submission marks them dirty (model.mm); the next pass, draft or commit
// zeroes them all before it encodes. Never inside a decode chain: its earlier
// passes are still counting on the GPU.
inline void reset_arrival_counters(Model& m) {
  for (id<MTLBuffer> b : {m.tune.kctr, m.norm_ctr, m.gdn_ctr, m.mz.cctr, m.mz.rctr})
    if (b) std::memset([b contents], 0, [b length]);
  m.ctr_dirty = false;
}

// Expert-cache admissions of a failed pass: the slot maps were updated
// when the copies were planned, but the copies ran in the pass's command
// buffers, which may not have completed. Those slots are freed: their
// experts are read again when routed (a stale map would serve another
// expert's weights, or a half-written one).
inline void xpool_rollback(Model& m) {
  for (const auto& [li, sl] : m.xpass) {
    ExpertPool& P = m.xpool[size_t(li)];
    const int e = P.expert_of[size_t(sl)];
    if (e >= 0) P.slot_of[size_t(e)] = -1;
    P.expert_of[size_t(sl)] = -1;
    P.last_use[size_t(sl)] = 0;
  }
  m.xpass.clear();
}

}  // namespace espetero::native

#include "gemma.h"

namespace espetero::native {

inline void run_forest(Model& m, PassMeta& meta, const std::vector<TreeSpec>& specs, PassStats* stats) {
  const auto t_wall = Clock::now();
  const bool chained = m.chain_active && m.chain_step > 0;  // earlier passes of the chain may still run
  if (!chained) {
    meta.arena.reset();
    m.tune.retired.clear();
    if (m.ctr_dirty) reset_arrival_counters(m);
    // The staging an earlier pass admitted experts from (a long pass's
    // records, up to ~3 GB) is not held past it.
    m.xadmit.clear();
    m.xadmit_src = Buf{};
  }
  m.xpass.clear();
  const Config& c = m.cfg;
  const int n_trees = int(specs.size());
  if (n_trees < 1) fail("empty forest");
  const uint64_t pass_id = ++m.pass_seq;  // commit records name the pass that kept their rows
  std::vector<TreePlan> trees(static_cast<size_t>(n_trees));
  int M = 0, N = 0, total_cands = 0;
  std::vector<int> saved(static_cast<size_t>(m.n_user), -1), read(static_cast<size_t>(m.n_user), -1);
  int temp_next = m.st.first_temp;

  // ---- validation and sizes (nothing is modified until every tree passes)
  for (int t = 0; t < n_trees; ++t) {
    const TreeSpec& b = specs[size_t(t)];
    TreePlan& T = trees[size_t(t)];
    T.spec = &b;
    auto check_slot = [&](int s, const char* what) {
      if (s < -1 || s >= m.n_user || (s >= 0 && !m.slots[size_t(s)].used))
        fail(std::string("invalid ") + what + " slot " + std::to_string(s));
    };
    check_slot(b.parent, "parent");
    check_slot(b.save, "save");
    check_slot(b.work, "work");
    if (b.root_len < 0 || b.n_leaves < 0) fail("invalid lengths");
    if (b.root_len == 0 && b.n_leaves == 0) fail("empty tree");
    if (b.root_len > 0 && !b.root) fail("root tokens missing");
    T.Tr = b.root_len;
    T.save_len = b.save_len > 0 ? b.save_len : T.Tr;
    if (T.save_len > T.Tr) fail("save_len exceeds root_len");
    if (b.save >= 0 && T.Tr == 0) fail("nothing to save (use slot_fork)");
    if (b.work >= 0) {
      if (T.Tr == 0) fail("nothing to save in the work slot");
      if (b.work == b.save || b.work == b.parent) fail("work slot must differ from save and parent slots");
    }
    T.parent_slot = b.parent;
    T.parent = b.parent >= 0 ? &m.slots[size_t(b.parent)] : nullptr;
    T.P = T.parent ? T.parent->len : 0;
    T.pstate = T.P > 0 ? b.parent : m.st.zero_slot;
    T.dst_slot = b.save;
    T.work_slot = b.work;
    T.dst = b.save >= 0 ? &m.slots[size_t(b.save)] : nullptr;
    T.work = b.work >= 0 ? &m.slots[size_t(b.work)] : nullptr;
    for (int s : {b.save, b.work}) {
      if (s < 0) continue;
      if (saved[size_t(s)] >= 0) fail("two trees save into one slot");
      saved[size_t(s)] = t;
    }
    if (b.parent >= 0) read[size_t(b.parent)] = t;
    T.mtp = b.mtp && m.mtp.present && T.Tr > 0 && (T.dst || T.work);
    T.mtp_run = T.mtp && (T.P == 0 || T.parent->mtp_ok);
    T.NL = b.n_leaves;
    T.N = T.NL + (b.score_root ? 1 : 0);
    if (b.score_root && T.Tr == 0) fail("score_root needs root tokens");
    if (b.out_hidden && T.Tr == 0) fail("out_hidden needs root tokens");
    int leaf_rows = 0, max_leaf = 0;
    if (T.NL > 0) {
      const bool cands = b.cand_offsets != nullptr;
      if (!b.leaf_offsets || !b.leaf_tokens || (cands && (!b.cand_ids || !b.out_cand_logits)))
        fail("leaf arrays missing");
      if (!cands && !b.out_argmax && !b.want_logits && !b.out_lse) fail("leaves need candidates or a vocabulary readout");
      if (b.leaf_offsets[0] != 0 || (cands && b.cand_offsets[0] != 0)) fail("offsets must start at 0");
      auto len_of = [&](int i) { return b.leaf_offsets[i + 1] - b.leaf_offsets[i]; };
      for (int i = 0; i < T.NL; ++i) {
        if (len_of(i) <= 0) fail("every leaf needs a token");
        if (cands && b.cand_offsets[i + 1] <= b.cand_offsets[i]) fail("every leaf needs candidates");
        max_leaf = std::max(max_leaf, len_of(i));
      }
      // Leaves whose tokens are a prefix of another leaf are read from that
      // leaf's rows (causal layers make the rows identical).
      T.host.assign(size_t(T.NL), -1);
      T.off.assign(size_t(T.NL), 0);
      for (int i = 0; i < T.NL; ++i) {
        int h = i;
        const int* ti = b.leaf_tokens + b.leaf_offsets[i];
        for (int j = 0; j < T.NL; ++j) {
          if (j == i || len_of(j) < len_of(i)) continue;
          if (len_of(j) == len_of(i) && j > i) continue;
          if (len_of(j) < len_of(h) || (len_of(j) == len_of(h) && j >= h)) continue;
          if (std::equal(ti, ti + len_of(i), b.leaf_tokens + b.leaf_offsets[j])) h = j;
        }
        T.host[size_t(i)] = h;
      }
      for (int i = 0; i < T.NL; ++i)
        if (T.host[size_t(i)] == i) {
          T.off[size_t(i)] = leaf_rows;
          leaf_rows += len_of(i);
        }
      T.n_cands = cands ? b.cand_offsets[T.NL] : 0;
      T.n_lh = std::min(leaf_rows, kMaxDraft);
      // Verify-commit: leaves [0, kc) one chain, hosted by leaf kc - 1 from
      // the first leaf row, in an in-place tree with a whole root.
      if (b.commit_chain > 0) {
        const int kc = b.commit_chain;
        if (kc > T.NL) fail("commit_chain exceeds the leaves");
        if (b.save < 0 || b.save != b.parent || b.work >= 0 || T.save_len != T.Tr)
          fail("commit_chain: in-place trees with a whole root only");
        for (int i = 0; i < kc; ++i)
          if (len_of(i) != i + 1 || T.host[size_t(i)] != kc - 1) fail("commit_chain: the leaves are not one chain");
        if (T.off[size_t(kc - 1)] != 0) fail("commit_chain: the chain must be the first leaf rows");
        T.cm_k = kc;
      }
    }
    if (b.commit_chain > 0 && T.NL == 0) fail("commit_chain without leaves");
    if (c.gemma && b.commit_chain > 0) fail("commit_chain: not supported for Gemma 4");
    if (c.gemma && b.root_pos) fail("explicit positions: not supported for Gemma 4");
    if (T.P + T.Tr + max_leaf > m.max_ctx) fail("context exceeds max_ctx (" + std::to_string(m.max_ctx) + ")");
    if (b.n_images < 0 || (b.n_images > 0 && !b.images)) fail("invalid images");
    if (b.n_images > 0 && !m.vis.c.present) fail("this model has no vision tower (or it was not loaded)");
    T.M = T.Tr + leaf_rows;
    T.base = M;
    T.readout_base = N;
    T.cand_base = total_cands;
    M += T.M;
    N += T.N;
    total_cands += T.n_cands;
    // A root whose final state is not saved but read by leaves needs a temp slot.
    if (T.Tr > 0) {
      if (T.work) T.root_final = b.work;
      else if (T.dst && T.save_len == T.Tr) T.root_final = b.save;
      else if (T.NL > 0) {
        if (temp_next >= m.st.n_slots) fail("too many unsaved roots with leaves in one pass");
        T.root_final = temp_next++;
      }
    }
  }
  for (int t = 0; t < n_trees; ++t) {
    const int ps = specs[size_t(t)].parent;
    if (ps >= 0 && saved[size_t(ps)] >= 0 && saved[size_t(ps)] != t) fail("a tree saves into another tree's parent");
  }
  if (M > m.max_rows) fail("pass exceeds max_rows (" + std::to_string(M) + " > " + std::to_string(m.max_rows) + ")");
  if (N > m.max_readouts) fail("pass exceeds max_readouts (" + std::to_string(N) + ")");

  // KV blocks needed (checked before modifying anything).
  {
    int need = 0;
    for (const TreePlan& T : trees) {
      const int w0 = T.mtp_run ? std::max(T.P - 1, 0) : T.P;
      if (T.dst) need += target_new_blocks(m, T.parent, T.P, w0, T.P + T.save_len + T.cm_k, T.dst == T.parent);
      if (T.work) need += target_new_blocks(m, T.parent, T.P, w0, T.P + T.Tr, false);
    }
    // Blocks released by targets that are overwritten come back after the pass;
    // require the free list to cover the new ones now.
    if (need > m.kv_free()) fail("KV cache full: need " + std::to_string(need) + " blocks, " +
                                 std::to_string(m.kv_free()) + " free");
  }

  // Images: patch rows are batched across trees in tree order.
  std::vector<ImageJob> jobs;
  int n_patch = 0;
  for (const TreePlan& T : trees) collect_images(T.spec->images, T.spec->n_images, T.Tr, T.base, n_patch, jobs);
  if (n_patch) {
    ensure_vscratch(m, n_patch);
    prepare_images(m, jobs);
  }
  ensure_scratch(m, std::max(M, 1), std::max(N, 1));
  Scratch& s = m.sc;
  bool any_mtp = false;
  int mtp_rows = 0;
  for (TreePlan& T : trees) {
    if (!T.mtp_run) continue;
    any_mtp = true;
    const int i0 = T.P > 0 ? 0 : 1;
    T.mtp_r0 = mtp_rows;
    T.mtp_R = T.Tr - i0;
    T.mtp_C = T.cm_k;  // the chain rows' MTP K/V after the root's
    mtp_rows += T.mtp_R + T.mtp_C;
  }
  // Decode-sized passes fuse the RMSNorms into the GEMVs that produce their
  // input: the residual-add GEMV's last threadgroup writes the normalized
  // fp16 rows, so no norm dispatch runs between layers.
  const bool fuse = !c.q4 && !c.gemma && m.fuse_norm && m.all_quant && gemv_path(m.tune, M) && M <= m.fuse_norm_max_m && n_patch == 0;
  if (fuse) {
    // The kernels reset their arrival counters (the last threadgroup stores
    // 0); zeroing them here only re-establishes that after a failed pass. A
    // chained pass must not touch them: earlier passes of the chain are still
    // counting on the GPU while this one is encoded.
    const size_t bytes = sizeof(int32_t) * size_t(2 * c.L + 4);
    if (!m.norm_ctr) {
      m.norm_ctr = m.buffer(bytes);
      std::memset([m.norm_ctr contents], 0, bytes);
    } else if (!chained) {
      std::memset([m.norm_ctr contents], 0, bytes);  // the previous pass completed
    }
  }
  // Hidden-state readouts (Clef heads) need every row's final state too.
  bool any_hidden = false;
  for (const TreeSpec& b : specs) any_hidden = any_hidden || b.out_hidden != nullptr;
  if (any_hidden && c.q4) fail("hidden-state readouts are not supported for qwen4_exp");
  // MTP needs every root row's final hidden state: no last-layer pruning.
  const bool prune = m.prune_last && !any_mtp && !fuse && !c.q4 && !any_hidden;
  // The DeltaNet kernels of this pass (the rule of every linear layer below);
  // the verify-commit replays a commit chain with the same ones.
  const bool gdn_fused_pass = !c.q4 && m.fuse_gdn && gemv_path(m.tune, M) && n_patch == 0 && c.Dk == 128 && c.Dv == 128;
  if (c.q4) {
    // The Qwen4-Exp path: roots, draft leaves and MTP rows; no decode chains.
    if (chained || m.chain_active) fail("qwen4_exp: decode chains are not supported yet");
  }

  // ---- per-row metadata (absolute rows; desc relative to the tree base)
  int32_t* tok = meta.tokens.ensure<int32_t>(m, size_t(M));
  int32_t* pos = meta.pos.ensure<int32_t>(m, size_t(M) * 4);
  int32_t* csrc = meta.conv_src.ensure<int32_t>(m, size_t(M) * 3);
  int32_t* rslot = meta.row_slot.ensure<int32_t>(m, size_t(M));
  int32_t* desc = meta.desc.ensure<int32_t>(m, size_t(M) * 4);
  int32_t* last = meta.readout_rows.ensure<int32_t>(m, size_t(std::max(N, 1)));
  int32_t* cid = meta.cand_ids.ensure<int32_t>(m, size_t(std::max(total_cands, 1)));
  int32_t* cown = meta.cand_owner.ensure<int32_t>(m, size_t(std::max(total_cands, 1)));
  meta.cand_out.ensure<float>(m, size_t(std::max(total_cands, 1)));
  for (TreePlan& T : trees) {
    const TreeSpec& b = *T.spec;
    const int pnext = T.parent ? T.parent->next_pos : 0;
    int pmax = pnext - 1, pmax_save = pnext - 1;
    for (int j = 0; j < T.Tr; ++j) {
      const int t = b.root[j];
      if (t < 0 || t >= c.V) fail("token out of range");
      const int r = T.base + j;
      tok[r] = t;
      int pt = pnext + j, ph = pt, pw = pt;
      if (b.root_pos) {
        pt = b.root_pos[3 * j];
        ph = b.root_pos[3 * j + 1];
        pw = b.root_pos[3 * j + 2];
      }
      pos[4 * r + 0] = pt;
      pos[4 * r + 1] = ph;
      pos[4 * r + 2] = pw;
      pos[4 * r + 3] = 0;
      pmax = std::max(pmax, std::max(pt, std::max(ph, pw)));
      if (j < T.save_len) pmax_save = pmax;
      for (int d = 1; d <= 3; ++d) {
        const int k = j - d;
        csrc[r * 3 + d - 1] = k >= 0 ? T.base + k : -1 - (3 + k);
      }
      rslot[r] = T.pstate;
      desc[r * 4 + 0] = 0;
      desc[r * 4 + 1] = 0;
      desc[r * 4 + 2] = 0;
      desc[r * 4 + 3] = 0;
    }
    T.next_root = pmax + 1;
    T.next_save = pmax_save + 1;
    if (T.N > T.NL) last[T.readout_base + T.NL] = T.base + T.Tr - 1;  // root readout
    for (int i = 0; i < T.NL; ++i) {
      const int h = T.host[size_t(i)];
      const int start = T.Tr + T.off[size_t(h)];  // relative
      const int len = b.leaf_offsets[i + 1] - b.leaf_offsets[i];
      last[T.readout_base + i] = T.base + start + len - 1;
      for (int j = 0; j < (h == i ? len : 0); ++j) {
        const int rr = start + j;
        const int r = T.base + rr;
        const int t = b.leaf_tokens[b.leaf_offsets[i] + j];
        if (t < 0 || t >= c.V) fail("token out of range");
        tok[r] = t;
        for (int d = 0; d < 3; ++d) pos[4 * r + d] = T.next_root + j;
        pos[4 * r + 3] = 0;
        for (int d = 1; d <= 3; ++d) {
          const int k = j - d;
          int src;
          if (k >= 0) src = T.base + start + k;
          else if (T.Tr + k >= 0) src = T.base + T.Tr + k;
          else src = -1 - (3 + T.Tr + k);
          csrc[r * 3 + d - 1] = src;
        }
        rslot[r] = T.pstate;
        desc[r * 4 + 0] = start;
        desc[r * 4 + 1] = 0;
        desc[r * 4 + 2] = T.Tr;
        desc[r * 4 + 3] = 0;
      }
      for (int k = b.cand_offsets ? b.cand_offsets[i] : 0; b.cand_offsets && k < b.cand_offsets[i + 1]; ++k) {
        if (b.cand_ids[k] < 0 || b.cand_ids[k] >= c.V) fail("candidate out of range");
        cid[T.cand_base + k] = b.cand_ids[k];
        cown[T.cand_base + k] = T.readout_base + i;
      }
    }
  }

  // ---- KV block tables. Reads use the parents' lists as they are before the
  // pass; targets get new lists (shared prefix blocks, new blocks for the rows
  // written, copy-on-write of a shared partial block).
  Forest F(m, meta);
  std::vector<int32_t> table;
  std::vector<std::pair<int, int>> cow;
  std::vector<KVSaveRec> kv_saves, mtp_saves;
  const int bt = m.kv.bt;
  // From the first new block until the slots take them (commit below), a
  // failure anywhere in the pass gives the new blocks back and rolls back the
  // expert admissions it planned. A command buffer can fail in the middle of
  // a pass: the qwen4 path commits one per layer, and macOS aborts long ones
  // while the display is in use (kIOGPUCommandBufferCallbackErrorImpacting-
  // Interactivity). If only a failure of the last command buffer gave the
  // blocks back, each failed sequence would leak its new KV blocks until the
  // cache filled and every later request failed with "KV cache full".
  struct PassGuard {
    Model& m;
    std::vector<TreePlan>& trees;
    bool armed = true;
    ~PassGuard() {
      if (!armed) return;
      for (TreePlan& T : trees) {
        for (int b : T.dst_blocks) m.kv_release(b);
        for (int b : T.work_blocks) m.kv_release(b);
        T.dst_blocks.clear();
        T.work_blocks.clear();
      }
      xpool_rollback(m);
    }
  } guard{m, trees};
  for (TreePlan& T : trees) {
    T.read_table = int(table.size());
    if (T.parent)
      for (int i = 0; i < (T.P + bt - 1) / bt; ++i) table.push_back(T.parent->blocks[size_t(i)]);
    const std::vector<int> empty;
    const std::vector<int>& pb = T.parent ? T.parent->blocks : empty;
    const int w0 = T.mtp_run ? std::max(T.P - 1, 0) : T.P;
    auto add_saves = [&](const std::vector<int>& blocks, int n, int mtp_n) {
      for (int i = 0; i < n; ++i) {
        const int p = T.P + i;
        kv_saves.push_back({T.base + i, blocks[size_t(p / bt)], p % bt, 0});
      }
      const int Pm = std::max(T.P - 1, 0);
      for (int i = 0; i < mtp_n; ++i) {
        const int p = Pm + i;
        mtp_saves.push_back({T.mtp_r0 + i, blocks[size_t(p / bt)], p % bt, 0});
      }
    };
    const int i0 = T.P > 0 ? 0 : 1;
    if (T.dst) {
      T.dst_blocks = F.target_blocks(pb, T.P, w0, T.P + T.save_len + T.cm_k, cow, T.dst == T.parent);
      add_saves(T.dst_blocks, T.save_len, T.mtp_run ? T.save_len - i0 : 0);
      // The chain rows' K/V at the positions they take if committed, and
      // their MTP K/V after the root's (garbage past the slot's length until then).
      for (int j = 0; j < T.cm_k; ++j) {
        const int p = T.P + T.Tr + j;
        kv_saves.push_back({T.base + T.Tr + j, T.dst_blocks[size_t(p / bt)], p % bt, 0});
      }
      for (int j = 0; j < (T.mtp_run ? T.mtp_C : 0); ++j) {
        const int p = std::max(T.P - 1, 0) + T.mtp_R + j;
        mtp_saves.push_back({T.mtp_r0 + T.mtp_R + j, T.dst_blocks[size_t(p / bt)], p % bt, 0});
      }
    }
    if (T.work) {
      T.work_blocks = F.target_blocks(pb, T.P, w0, T.P + T.Tr, cow, false);
      add_saves(T.work_blocks, T.Tr, T.mtp_run ? T.mtp_R : 0);
    }
  }
  // ESPETERO_FAIL_PASS=N (recovery tests): every Nth pass of 64 rows or
  // more (prompt chunks; the long passes macOS aborts) fails as a GPU command
  // buffer would, with its blocks allocated (qwen4: at the middle layer, after
  // half of the layers planned their expert admissions).
  static const int fail_every = [] {
    const char* e = std::getenv("ESPETERO_FAIL_PASS");
    return e && *e ? std::atoi(e) : 0;
  }();
  const bool fault = fail_every > 0 && !chained && M >= 64 && ++m.pass_count % uint64_t(fail_every) == 0;
  const bool fault_mid = fault && c.q4 && !m.xpool.empty() && M * c.k > m.xpool[0].S;  // long qwen4 pass
  if (fault && !fault_mid) fail("GPU command buffer failed: injected (ESPETERO_FAIL_PASS)");
  int32_t* blocks_gpu = meta.blocks.ensure<int32_t>(m, std::max<size_t>(table.size(), 1));
  if (!table.empty()) std::memcpy(blocks_gpu, table.data(), table.size() * 4);
  auto upload = [&](HostBuf& hb, const auto& v) {
    using T = typename std::decay_t<decltype(v)>::value_type;
    T* p = hb.ensure<T>(m, std::max<size_t>(v.size(), 1));
    if (!v.empty()) std::memcpy(p, v.data(), v.size() * sizeof(T));
  };
  std::vector<std::array<int32_t, 2>> cow_pairs;
  for (auto& [a, b] : cow) cow_pairs.push_back({a, b});
  upload(meta.cow, cow_pairs);
  upload(meta.kv_saves, kv_saves);
  upload(meta.mtp_kv_saves, mtp_saves);
  // The commit chains' pass rows; each linear layer copies their in_proj
  // rows into Model::cm_x ([layer][row][ldl] floats) for NativeModel::commit.
  int cm_total = 0;
  {
    std::vector<int32_t> cm_rows;
    for (TreePlan& T : trees) {
      T.cm_row0 = cm_total;
      for (int j = 0; j < T.cm_k; ++j) cm_rows.push_back(T.base + T.Tr + j);
      cm_total += T.cm_k;
    }
    if (cm_total > 0) {
      upload(meta.cm_rows, cm_rows);
      const int ld = c.lin_ld();
      if (!m.cm_x || m.cm_cap < cm_total || m.cm_ld != ld) {
        m.cm_cap = std::max(cm_total, 16);
        m.cm_ld = ld;
        m.cm_x = m.buffer(size_t(std::max(c.n_linear, 1)) * size_t(m.cm_cap) * size_t(ld) * sizeof(float));
      }
    }
  }

  // ---- delta-rule segments and conv saves
  std::vector<DeltaSeg> roots, leaves;
  std::vector<ConvSave> conv_jobs;
  for (TreePlan& T : trees) {
    const TreeSpec& b = *T.spec;
    if (T.Tr > 0) {
      DeltaSeg sg{T.base, T.Tr, T.pstate, std::max(T.root_final, 0), 0, -1, T.root_final >= 0 ? 1 : 0, 0};
      // The save slot takes the state after save_len rows when the final state
      // goes elsewhere (the work slot, or a longer root).
      if (T.dst && T.root_final != T.dst_slot) {
        sg.mid_slot = T.dst_slot;
        sg.save_t = T.save_len;
      }
      roots.push_back(sg);
    }
    for (int i = 0; i < T.NL; ++i) {
      if (T.host[size_t(i)] != i) continue;
      const int len = b.leaf_offsets[i + 1] - b.leaf_offsets[i];
      const int in = T.Tr > 0 ? T.root_final : T.pstate;
      leaves.push_back({T.base + T.Tr + T.off[size_t(i)], len, in, 0, 0, -1, 0, 0});
    }
    if (T.dst) conv_jobs.push_back({T.base, T.save_len, T.pstate, T.dst_slot});
    if (T.work) conv_jobs.push_back({T.base, T.Tr, T.pstate, T.work_slot});
  }
  // Conv saves inside the fused root dispatch: only without leaves
  // (they read the parents' taps in a later dispatch), with at most one save
  // per tree, of its root rows, into a slot no other root reads.
  std::vector<ConvSave> conv_root;
  bool conv_fold = m.gdn_conv_fold && leaves.empty() && !conv_jobs.empty() && roots.size() <= size_t(Model::kGdnCtrSegs);
  if (conv_fold) {
    for (TreePlan& T : trees) {
      if (T.work || (T.dst && T.Tr <= 0) || (T.dst && T.save_len > T.Tr)) conv_fold = false;
      if (T.Tr > 0) conv_root.push_back(T.dst ? ConvSave{T.base, T.save_len, T.pstate, T.dst_slot} : ConvSave{0, 0, 0, 0});
    }
    for (size_t i = 0; i < conv_root.size() && conv_fold; ++i)
      for (size_t j = 0; j < roots.size(); ++j)
        if (j != i && conv_root[i].T > 0 && (roots[j].in_slot == conv_root[i].out_slot || roots[j].in_slot == conv_root[i].in_slot))
          conv_fold = false;
  }
  if (conv_fold) {
    upload(meta.conv_root, conv_root);
    if (!m.gdn_ctr) {
      m.gdn_ctr = m.buffer(sizeof(int32_t) * size_t(Model::kGdnCtrSegs) * size_t(c.Hk));
      std::memset([m.gdn_ctr contents], 0, sizeof(int32_t) * size_t(Model::kGdnCtrSegs) * size_t(c.Hk));
    }
  }
  upload(meta.segs_root, roots);
  upload(meta.segs_leaf, leaves);
  int max_root_len = 0, max_leaf_len = 0;
  for (const DeltaSeg& sg : roots) max_root_len = std::max(max_root_len, sg.len);
  for (const DeltaSeg& sg : leaves) max_leaf_len = std::max(max_leaf_len, sg.len);
  upload(meta.conv_jobs, conv_jobs);

  // ---- attention plan, and the sparse attention plan of the Qwen4-Exp rows
  AttnPlan A;
  for (TreePlan& T : trees) A.trees.push_back({T.base, T.P, T.read_table, T.M});
  const bool qsa_idx = c.q4 && m.kv.i_layers > 0;
  QsaSet qs;
  // Gemma 4: a plan per layer kind (sliding layers read the parent's
  // ring); every target's final window of new rows goes into its ring, after
  // the parent's part of that window when the target is another slot.
  G4Plan g4s, g4g;
  size_t n_ring_saves = 0, n_ring_copies = 0;
  if (c.gemma) {
    for (TreePlan& T : trees) {
      g4g.trees.push_back({T.base, T.P, T.read_table, T.M});
      g4s.trees.push_back({T.base, T.P, T.parent_slot >= 0 ? T.parent_slot * m.ring_blocks : 0, T.M});
    }
    // Single-row trees take g4_attention_dec (one threadgroup per KV head, all its query heads).
    const bool g4dec = g4_dec(m, trees.size(), [&](size_t i) { return trees[i].M; });
    g4_plan_tiles(g4g, c.Hkv_g * (g4dec && g4_geom(c, false).G <= 8 ? 1 : g4_geom(c, false).split), m.attn_split, 0, pos);
    g4_plan_tiles(g4s, c.Hkv, m.attn_split, c.window, pos);
    upload(meta.a_trees, g4g.trees);
    upload(meta.a_tiles, g4g.tiles);
    upload(meta.a_comb, g4g.combine);
    upload(meta.s_trees, g4s.trees);
    upload(meta.s_tiles, g4s.tiles);
    upload(meta.s_comb, g4s.combine);
    meta.parts.ensure<float>(m, std::max<size_t>({g4_parts_floats(c, g4g, false), g4_parts_floats(c, g4s, true), 1}));
    std::vector<KVSaveRec> rs;
    std::vector<std::array<int32_t, 2>> rc;
    if (m.kvs.n_layers) {
      const int rb = m.ring_blocks, W = c.window;
      for (TreePlan& T : trees) {
        auto target = [&](int slot, int S) {
          if (slot != T.parent_slot && T.P > 0) {
            std::vector<char> seen(size_t(rb), 0);
            for (int b = std::max(0, T.P + S - W + 1) / bt; b * bt < T.P; ++b)
              if (!seen[size_t(b % rb)]++) rc.push_back({T.parent_slot * rb + b % rb, slot * rb + b % rb});
          }
          for (int p = std::max(T.P, T.P + S - W + 1); p < T.P + S; ++p)
            rs.push_back({T.base + (p - T.P), slot * rb + (p / bt) % rb, p % bt, 0});
        };
        if (T.dst) target(T.dst_slot, T.save_len);
        if (T.work) target(T.work_slot, T.Tr);
      }
    }
    upload(meta.ring_saves, rs);
    upload(meta.ring_copies, rc);
    n_ring_saves = rs.size();
    n_ring_copies = rc.size();
  } else {
    if (qsa_idx) qsa_plan(m, meta.arena, qs, A.trees, desc, true);
    plan_tiles(A, c.Hkv, m.attn_split, qs.n_sparse ? &qs.qt_of : nullptr);
    upload(meta.a_trees, A.trees);
    upload(meta.a_tiles, A.tiles);
    upload(meta.a_comb, A.combine);
  }
  const int G = c.H / c.Hkv;
  if (!c.gemma) meta.parts.ensure<float>(m, std::max<size_t>(size_t(A.parts) * size_t(c.Hkv) * 8 * size_t(G) * 258, 1));

  // ---- encode
  const auto t_enc = Clock::now();
  m.dispatches = 0;
  m.barriers = 0;
  id<MTLCommandBuffer> cb = [m.queue commandBufferWithUnretainedReferences];
  Enc E(m, cb);
  const KVPool& K = m.kv;
  const StatePool& S = m.st;
  if (!cow_pairs.empty()) {
    kv_cow_copy(m, E, meta.cow.buf, cow_pairs.size());
    E.barrier();  // kv_save writes into the copied blocks
  }
  if (n_ring_copies) {  // Gemma 4: the parents' windows into the rings of other targets
    const KVPool& R = m.kvs;
    const AttnParams ap{0, 0, 0, R.kv_dim, 0, R.bt, 0};
    const uint64_t region = R.region_halves;
    E.pipe("kv_copy_blocks");
    E.set(0, R.buf);
    E.set(1, meta.ring_copies.buf);
    E.bytes(2, ap);
    E.bytes(3, region);
    E.rd(R.buf);
    E.wr(R.buf);
    E.threads(MTLSizeMake(R.block_halves() / 8, n_ring_copies, size_t(R.n_layers)), MTLSizeMake(64, 1, 1));
    E.barrier();
  }

  // Embedding (image rows are overwritten by the vision tower).
  {
    static const char* names[3] = {"embed_q4", "embed_q8", "embed_f16"};
    const Embed& T = m.embed;
    E.pipe(names[T.kind]);
    if (chained) {  // the previous pass's argmax, on the GPU
      E.set(0, m.chain_tok, size_t(m.chain_step - 1) * size_t(m.chain_n) * 4);
      E.rd(m.chain_tok);
    } else {
      E.set(0, meta.tokens.buf);
    }
    E.set(1, T.codes);
    if (T.kind != 2) {
      E.set(2, T.s);
      E.set(3, T.z);
    }
    E.set(4, s.h);
    E.bytes(5, c.D);
    if (T.kind != 2) E.bytes(6, T.group);
    E.wr(s.h);
    E.threads(MTLSizeMake(size_t(c.D / 8), size_t(M), 1), MTLSizeMake(size_t(std::min(c.D / 8, 256)), 1, 1));
  }
  if (n_patch) encode_vision(m, E, n_patch, s.h);
  if (c.gemma) {  // Gemma 4: embeddings times sqrt(D)
    const uint32_t n4 = uint32_t(size_t(M) * size_t(c.D) / 4);
    E.pipe("g4_scale_rows");
    E.set(0, s.h);
    E.bytes(1, c.embed_scale);
    E.bytes(2, n4);
    E.rd(s.h);
    E.wr(s.h);
    E.threads(MTLSizeMake(n4, 1, 1), MTLSizeMake(256, 1, 1));
  }

  const int C = c.C();
  const int Vd = c.Vd();
  const int kvd = c.kv_dim();
  const int ldl = c.lin_ld();
  const int ldf = c.full_ld();
  const int qdim = c.H * c.HD;
  auto rmsnorm = [&](const Buf& X, const Buf& w, const Buf& Y, int rows, const Buf& gather_rows) {
    E.pipe("rmsnorm");
    E.set(0, X);
    E.set(1, w);
    E.set(2, Y);
    E.bytes(3, c.D);
    E.bytes(4, c.eps);
    const int g = gather_rows ? 1 : 0;
    E.set(5, gather_rows ? gather_rows : X);
    E.bytes(6, g);
    E.rd(X.b);
    E.wr(Y.b);
    E.groups(MTLSizeMake(size_t(rows), 1, 1), MTLSizeMake(256, 1, 1));
  };
  auto gather = [&](id<MTLBuffer> src, id<MTLBuffer> dst, int width, bool f32) {
    E.pipe(f32 ? "gather_rows_f32" : "gather_rows_f16");
    E.set(0, src);
    E.set(1, dst);
    E.set(2, meta.readout_rows.buf);
    E.bytes(3, width);
    E.rd(src);
    E.wr(dst);
    E.threads(MTLSizeMake(size_t(width / 4), size_t(N), 1), MTLSizeMake(64, 1, 1));
  };
  auto mlp_in = [&](const Layer& L, id<MTLBuffer> X, id<MTLBuffer> act, int rows, const NormLink* nin = nullptr) {
    (void)nin;
    shared_mlp_in(m, E, L, X, act, rows);
  };
  // Forest attention over `plan` for the rows in Q/Kb/Vb, writing O.
  auto attention = [&](const AttnPlan& plan, HostBuf& trees_b, HostBuf& tiles_b, HostBuf& comb_b,
                       const Buf& desc_b, int kv_layer, const QsaSet* qsa = nullptr) {
    if (plan.tiles.empty()) return;
    // GQA 12 (Qwen4-Exp) runs as 2 groups of 6 query heads per cached KV head
    // (64 x 12 threads exceed the kernel's 512): the kernel and the combine see
    // Hkv x 2 head groups.
    const int hsplit = G > 8 ? G / 6 : 1, Gk = G / hsplit;
    AttnParams ap{c.H, c.Hkv * hsplit, qdim, kvd, ldf, bt, uint64_t(K.layer_base(kv_layer))};
    // Single-row trees (plain decode): the GQA-row MMA kernel or the GEMV-style kernel; else the 8-row MMA tiles.
    bool single = G <= 8 && !qsa;
    for (const auto& t : plan.trees) single = single && t.M == 1;
    const bool gqa = single && m.attn_gqa;
    const bool vec = single && !gqa && m.attn_vec;
    const int nsg = gqa ? 4 : (G <= 4 ? 1 : 2) * 4;  // gqa: 2 pairs; vec: head halves (heads per simdgroup <= 4) x 4 key slices
    // Long prompt passes at long contexts: a dense attention layer over
    // 4096 rows at about 40K keys (Qwen3.8-Flash-Next) is ~1.5 s
    // of GPU in one command buffer, which macOS aborts while the display is in
    // use (518 rows: 185 ms). Above 16M row-keys the tiles go in groups, each
    // committed on its own: same tiles, same order (bit-identical).
    // ESPETERO_ATTN_SPLIT=M (million row-keys per command buffer; 0 = one dispatch).
    static const size_t split_rk = [] {
      const char* e = std::getenv("ESPETERO_ATTN_SPLIT");
      return size_t(e && *e ? std::atoi(e) : 16) << 20;
    }();
    std::vector<size_t> cut{0};
    // Prompt passes only (64 rows or more): decode-sized qwen4 passes may keep
    // their routing handshakes as GPU events inside one command buffer, served
    // by the host after the commit (ESPETERO_Q4_EV; a flush there would wait for
    // itself).
    if (split_rk > 0 && !vec && !gqa && !chained && !m.chain_active && M >= 64) {
      size_t acc = 0;
      for (size_t i = 0; i < plan.tiles.size(); ++i) {
        const AttnTileRec& tr = plan.tiles[i];
        const size_t span = tr.pad0 ? size_t(4) * size_t(std::min(tr.k1 - tr.k0, kQsaLmax)) : size_t(tr.k1 - tr.k0);
        const size_t w = size_t(8) * (span + size_t(plan.trees[size_t(tr.tree)].M));
        if (acc > 0 && acc + w > split_rk) cut.push_back(i), acc = 0;
        acc += w;
      }
    }
    cut.push_back(plan.tiles.size());
    for (size_t g = 0; g + 1 < cut.size(); ++g) {
      if (g > 0) E.flush();
      const std::string kname = (gqa ? "attn_decode_gqa_g" : vec ? "attn_decode_vec_g" : (qsa ? "flash_attention_qsa_g" : "flash_attention_g")) +
                                std::to_string(Gk) + (hsplit > 1 ? "s" + std::to_string(hsplit) : "");
      if (qsa && [m.kernel(kname) maxTotalThreadsPerThreadgroup] < size_t(64 * Gk)) fail(kname + ": threadgroup too small");
      E.pipe(kname);
      if (qsa) qsa_bind(m, E, *qsa, kv_layer);
      E.set(0, s.Q);
      E.set(1, s.Kb);
      E.set(2, s.Vb);
      E.set(3, K.buf);
      E.set(4, meta.blocks.buf);
      E.set(5, desc_b);
      E.set(6, s.proj);
      E.set(7, s.O);
      E.bytes(8, ap);
      E.set(9, tiles_b.buf, cut[g] * sizeof(AttnTileRec));
      E.set(10, trees_b.buf);
      E.set(11, meta.parts.buf);
      E.rd(s.Q);
      E.rd(s.Kb);
      E.rd(s.Vb);
      E.rd(s.proj);
      E.wr(s.O);
      if (plan.parts) E.wr(meta.parts.buf, meta.parts.bytes);
      E.groups(MTLSizeMake(size_t(c.Hkv * hsplit), cut[g + 1] - cut[g], 1), MTLSizeMake(size_t(vec || gqa ? 32 * nsg : 64 * Gk), 1, 1));
    }
    if (!plan.combine.empty()) {
      E.pipe("attention_combine");
      E.set(0, meta.parts.buf);
      E.set(1, s.proj);
      E.set(2, s.O);
      E.bytes(3, ap);
      E.set(4, comb_b.buf);
      E.rd(meta.parts.buf, meta.parts.bytes);
      E.rd(s.proj);
      E.wr(s.O);
      E.threads(MTLSizeMake(64, size_t(c.H), plan.combine.size() * 8), MTLSizeMake(64, 1, 1));
    }
  };
  auto kv_save = [&](HostBuf& entries, size_t n, int kv_layer) {
    if (!n) return;
    AttnParams ap{c.H, c.Hkv, qdim, kvd, ldf, bt, uint64_t(K.layer_base(kv_layer))};
    E.pipe("kv_save");
    E.set(0, s.Kb);
    E.set(1, s.Vb);
    E.set(2, K.buf);
    E.set(3, entries.buf);
    E.bytes(4, ap);
    E.rd(s.Kb);
    E.rd(s.Vb);
    E.threads(MTLSizeMake(size_t(kvd / 8), n, 1), MTLSizeMake(64, 1, 1));
  };
  auto attn_prep = [&](const Layer& L, const Buf& pos_b, size_t pos_off, int rows) {
    E.pipe("attn_prep");
    E.set(0, s.proj);
    E.set(1, L.q_norm);
    E.set(2, L.k_norm);
    E.set(3, pos_b, pos_off);
    E.set(4, m.inv_freq);
    E.set(5, s.Q);
    E.set(6, s.Kb);
    E.set(7, s.Vb);
    AttnPrepParams ap{c.H, c.Hkv, ldf, 2 * qdim, qdim, kvd, c.eps};
    E.bytes(8, ap);
    E.set(9, m.rope_sel);
    E.rd(s.proj);
    E.wr(s.Q);
    E.wr(s.Kb);
    E.wr(s.Vb);
    E.groups(MTLSizeMake(size_t(c.H + c.Hkv + 3) / 4, size_t(rows), 1), MTLSizeMake(128, 1, 1));
  };

  // Norm partials of the fused path: point 2l = layer l's input norm, 2l + 1
  // its post-attention norm; [point][row][P] floats.
  const int np_P = (c.D + 31) / 32;
  Buf np_parts;
  if (fuse) np_parts = meta.arena.alloc(m, size_t(2 * c.L) * size_t(M) * size_t(np_P) * 4);
  auto link = [&](int point, const Buf& w) {
    NormLink l;
    l.parts = Buf(np_parts.b, np_parts.off + size_t(point) * size_t(M) * size_t(np_P) * 4);
    l.ld = np_P;
    l.P = np_P;
    l.D = c.D;
    l.eps = c.eps;
    l.w = w;
    l.xn = Buf(s.xn, 0);
    l.ctr = Buf(m.norm_ctr, sizeof(int32_t) * size_t(point));
    return l;
  };
  // ---- Qwen4-Exp (docs/qwen4_exp.md): the residual as hc streams of D values,
  // gated hyper-connection blocks around every block, the n-gram mix before
  // layer c.ple_layer, and (test path until the expert cache) each
  // layer's experts read from the SSD after flushing the pass so far.
  // Expert staging lives in m.xstage (reset at each handshake, after the flush).
  static const bool q4_flushlog = env_on("ESPETERO_Q4_FLUSHLOG");
  Enc::FlushLog q4_fl;
  if (q4_flushlog && c.q4) E.flush_log = &q4_fl;
  auto q4_read = [&](const HCBlock& B, bool inject, id<MTLBuffer> X, int rows, id<MTLBuffer> xout) {
    q4_hc_read(m, E, B, inject, X, rows, xout);
  };
  auto q4_inject = [&](int rows) { q4_hc_inject(m, E, m.q4s.H, m.q4s.y, m.q4s.inj, rows); };
  auto q4_ple = [&]() {
    Model::Q4Scratch& q = m.q4s;
    const Ple& P = m.ple;
    const int W = c.W();
    gemm(E, m.tune, P.key, q.e16, 0, c.D, q.kraw, 0, W, M, 0);
    gemm(E, m.tune, P.value, q.e16, 0, c.D, q.vp, 0, c.D, M, 0);
    E.pipe("q4_ple_gate");
    E.set(0, q.kraw);
    E.set(1, q.H);
    E.set(2, P.norm_key);
    E.set(3, P.norm_query);
    E.set(4, q.g);
    E.bytes(5, c.D);
    E.bytes(6, c.hc);
    E.bytes(7, c.eps);
    E.rd(q.kraw);
    E.rd(q.H);
    E.wr(q.g);
    E.groups(MTLSizeMake(size_t(c.hc), size_t(M), 1), MTLSizeMake(256, 1, 1));
    E.pipe("q4_ple_apply");
    E.set(0, q.vp);
    E.set(1, q.g);
    E.set(2, P.norm_conv);
    E.set(3, q.G);
    E.set(4, q.Gn);
    E.bytes(5, c.D);
    E.bytes(6, c.hc);
    E.bytes(7, c.eps);
    E.rd(q.vp);
    E.rd(q.g);
    E.wr(q.G);
    E.wr(q.Gn);
    E.groups(MTLSizeMake(size_t(c.hc), size_t(M), 1), MTLSizeMake(256, 1, 1));
    E.pipe("q4_ple_conv");
    E.set(0, q.H);
    E.set(1, q.G);
    E.set(2, q.Gn);
    E.set(3, P.conv_w);
    E.set(4, q.psrc);
    E.set(5, m.ple_hist);
    E.set(6, q.pslot);
    E.bytes(7, W);
    const int hr = Model::kPleHist;
    E.bytes(8, hr);
    E.rd(q.G);
    E.rd(q.Gn);
    E.rd(q.H);
    E.wr(q.H);
    E.threads(MTLSizeMake(size_t(W), size_t(M), 1), MTLSizeMake(256, 1, 1));
  };
  auto q4_experts = [&](int li) {
    E.flush();
    if (!m.xstage) m.xstage = std::make_unique<Arena>(m.dev, size_t(64) << 20);
    m.xstage->reset();  // the previous layer's stacks are no longer read
    QW gu, dn;
    read_layer_experts(m, li, *m.xstage, gu, dn);
    repack(m, E.e, gu, *m.xstage, m.xstage.get());
    repack(m, E.e, dn, *m.xstage, m.xstage.get());
    E.barrier();
    Layer& Lm = m.layers[size_t(li)];
    Lm.experts_gu = gu;
    Lm.experts_down = dn;
  };
  // Expert cache: called by moe_mlp after the router of a q4 layer.
  // Decode-sized passes (rows x k <= slots) use the event pipeline: the hook
  // only encodes the handshake and the remap; the host serves the layers in
  // order after the commit (q4_serve). Larger passes flush per layer.
  bool q4_evpipe = false;
  bool q4_swept = false;  // a layer staged whole experts (long prompts): the staging is freed after the pass
  struct Q4Pending {
    int li, rows;
    uint64_t v;
  };
  std::vector<Q4Pending> q4_pend;
  auto q4_admit = [&](const QW& src, const QW& dst, int ex, int slot, int rows) {
    const int row_bytes = src.in / 2, groups = src.in / src.group, src_pairs = src.out / 2, dst_pairs = dst.out / 2;
    E.pipe("q4_pool_admit");
    E.set(0, src.wr);
    E.set(1, src.sb);
    E.set(2, dst.wr);
    E.set(3, dst.sb);
    E.bytes(4, rows);
    E.bytes(5, row_bytes);
    E.bytes(6, groups);
    E.bytes(7, src_pairs);
    E.bytes(8, dst_pairs);
    E.bytes(9, ex);
    E.bytes(10, slot);
    E.rd(src.wr.b);
    E.rd(src.sb.b);
    E.wr(dst.wr.b, dst.wr.off + size_t(slot) * size_t(rows) * size_t(row_bytes),
         dst.wr.off + size_t(slot + 1) * size_t(rows) * size_t(row_bytes));
    E.wr(dst.sb.b);
    const size_t words = size_t(rows) * size_t(row_bytes) / 4, sp = size_t(rows / 2) * size_t(groups);
    E.threads(MTLSizeMake(std::max(words, sp), 1, 1), MTLSizeMake(256, 1, 1));
  };
  // Residency of one layer's routed experts (`need`, distinct ids): LRU
  // victims for the missing ones, their pieces read and repacked into their
  // slots (encoded into `enc`), the id -> slot map written. Returns whether
  // anything was loaded.
  // Prefetch (ESPETERO_Q4_PF=1/2): loads go through the prefetch entries
  // (demand reads at the front of the reader queue), then the guessed experts
  // of layer `guess_layer` that its pool lacks are queued behind them.
  const int pf_mode = q4_pf_mode();
  // Online pinned hot set: fraction of each pool (0: off).
  static const float xpin = [] {
    const char* v = std::getenv("ESPETERO_XPIN");
    return v && *v ? std::min(0.9f, std::max(0.0f, float(std::atof(v)))) : 0.0f;
  }();
  auto pin_update = [&](ExpertPool& P, const std::vector<int>& used, const std::vector<int>& counts) {
    if (xpin <= 0.0f) return;
    if (P.use_cnt.empty()) P.use_cnt.assign(size_t(c.E), 0), P.pinned.assign(size_t(c.E), 0);
    for (int e : used) P.use_cnt[size_t(e)] += uint32_t(counts[size_t(e)]);
    if (++P.pin_age % 256 != 0) return;  // re-rank every 256 handshakes of the layer
    if (P.pin_age % 4096 == 0)
      for (uint32_t& u : P.use_cnt) u >>= 1;  // slow decay
    std::vector<int> ord(size_t(c.E));
    for (int e = 0; e < c.E; ++e) ord[size_t(e)] = e;
    const int np = int(xpin * float(P.S));
    std::partial_sort(ord.begin(), ord.begin() + np, ord.end(),
                      [&](int a, int b) { return P.use_cnt[size_t(a)] > P.use_cnt[size_t(b)]; });
    std::fill(P.pinned.begin(), P.pinned.end(), 0);
    for (int i = 0; i < np; ++i) P.pinned[size_t(ord[size_t(i)])] = 1;
  };
  static const bool q4_split = [] {  // resident tiles before the reads are done (ESPETERO_Q4_SPLIT=0: off)
    const char* v = std::getenv("ESPETERO_Q4_SPLIT");
    return !(v && *v == '0');
  }();
  // Replacement policy: 1 LRU (default), 2 LRU-2 (ESPETERO_XPOLICY=lru2).
  static const int xpolicy = [] {
    const char* v = std::getenv("ESPETERO_XPOLICY");
    return v && std::strcmp(v, "lru2") == 0 ? 2 : 1;
  }();
  // defer: the copies are appended to m.xdefer (encoded after the
  // resident experts' tiles were committed) and the loaded slots flagged in m.xld.
  auto q4_load_routed = [&](int li, const std::vector<int>& need, id<MTLComputeCommandEncoder> enc, int guess_layer = -1,
                            int guess_rows = 0, bool defer = false) {
    ExpertPool& P = m.xpool[size_t(li)];
    ExpertCacheStats& st = m.xstats;
    // Victims only when something misses (about half the handshakes hit
    // everything): the slots not holding a routed expert, free ones first,
    // then least recently used; partial selection of as many as are missing.
    int nmiss = 0;
    for (int e : need) nmiss += P.slot_of[size_t(e)] < 0 ? 1 : 0;
    if (m.xmiss.size() != size_t(c.L)) m.xmiss.assign(size_t(c.L), 1.0f);
    m.xmiss[size_t(li)] = 0.9f * m.xmiss[size_t(li)] + 0.1f * float(nmiss);
    std::vector<std::pair<int, int>>& loads = m.xloads;
    loads.clear();
    if (nmiss > 0) {
      if (P.mark.empty()) P.mark.assign(size_t(c.E), 0);
      const uint32_t stamp = ++P.stamp;
      for (int e : need) P.mark[size_t(e)] = stamp;
      auto& cand = m.xcand;
      cand.clear();
      for (int sl = 0; sl < P.S; ++sl) {
        const int x = P.expert_of[size_t(sl)];
        if (x >= 0 && P.mark[size_t(x)] == stamp) continue;
        cand.push_back({P.evict_key(sl, xpolicy), sl});
      }
      std::partial_sort(cand.begin(), cand.begin() + nmiss, cand.end());
      size_t vi = 0;
      for (int e : need)
        if (P.slot_of[size_t(e)] < 0) {
          const int sl = cand[vi++].second;
          if (P.expert_of[size_t(sl)] >= 0) P.slot_of[size_t(P.expert_of[size_t(sl)])] = -1;
          P.expert_of[size_t(sl)] = e;
          P.slot_of[size_t(e)] = sl;
          loads.push_back({e, sl});
          m.xpass.push_back({li, sl});
        }
    }
    for (int e : need) P.last_use[size_t(P.slot_of[size_t(e)])] = m.xclock;
    if (xpolicy == 2) {
      if (P.e_last.empty()) P.e_last.assign(size_t(c.E), 0), P.e_prev.assign(size_t(c.E), 0);
      for (int e : need) {
        P.e_prev[size_t(e)] = P.e_last[size_t(e)];
        P.e_last[size_t(e)] = m.xclock;
      }
    }
    const size_t expert_bytes = size_t(3) * (size_t(c.Ie) * size_t(c.D) / 2 + size_t(c.Ie) * size_t(c.D / m.xs.layers[0].group) * 4);
    static const bool pf_hostoff = env_on("ESPETERO_Q4_PF_HOSTOFF");  // timing probe: the GPU guess without host prefetch
    if (pf_mode > 0 && !pf_hostoff) {
      ExpertPrefetch& X = ensure_prefetch(m);
      std::vector<std::pair<int, int>> ents, rest;  // (entry, slot); loads without a free entry
      std::unique_lock<std::mutex> lk(X.mu);
      for (auto [e, sl] : loads) {
        int idx = xpf_find(X, c.E, li, e);
        if (idx >= 0) {
          ExpertPrefetch::Entry& en = *X.entries[size_t(idx)];
          if (en.guessed) {
            ++st.pf_used;
            st.pf_late += en.ready ? 0 : 1;
          }
        } else if ((idx = xpf_start(m, X, li, e, true, false)) >= 0) {
          ++st.pf_demand;
        } else {
          rest.push_back({e, sl});
          continue;
        }
        X.entries[size_t(idx)]->busy = X.hs;
        X.entries[size_t(idx)]->stamp = X.hs;
        ents.push_back({idx, sl});
      }
      if (guess_layer >= 0) {
        const int32_t* g = static_cast<const int32_t*>([m.q4s.pfids contents]);
        const ExpertPool& Pn = m.xpool[size_t(guess_layer)];
        static const int pf_top = [] {  // guesses per row, best first (8: 72 % of them used; 16: 27 %)
          const char* v = std::getenv("ESPETERO_Q4_PF_TOP");
          return v && *v ? std::min(16, std::max(1, std::atoi(v))) : 8;
        }();
        if (guess_rows <= 8) {
          for (int i = 0; i < guess_rows * 16; ++i) {
            if (i % 16 >= pf_top) continue;
            const int e = g[i];
            if (e < 0 || e >= c.E || Pn.slot_of[size_t(e)] >= 0 || xpf_find(X, c.E, guess_layer, e) >= 0) continue;
            if (xpf_start(m, X, guess_layer, e, false, true) < 0) break;
            ++st.pf_issued;
            st.pf_bytes += expert_bytes;
          }
        } else {
          // Many rows: the missing experts guessed by the most rows
          // first, at most ESPETERO_Q4_PF_MAXN reads (entries stay for the demand loads).
          static const int pf_maxn = [] {
            const char* v = std::getenv("ESPETERO_Q4_PF_MAXN");
            return v && *v ? std::max(1, std::atoi(v)) : 32;
          }();
          std::vector<int>& gc = m.xgcnt;
          std::vector<int>& cand = m.xgcand;
          if (gc.size() != size_t(c.E)) gc.assign(size_t(c.E), 0);
          cand.clear();
          for (int r = 0; r < guess_rows; ++r)
            for (int j = 0; j < pf_top; ++j) {
              const int e = g[r * 16 + j];
              if (e < 0 || e >= c.E || Pn.slot_of[size_t(e)] >= 0) continue;
              if (gc[size_t(e)]++ == 0) cand.push_back(e);
            }
          std::sort(cand.begin(), cand.end(), [&](int a, int b) { return gc[size_t(a)] > gc[size_t(b)] || (gc[size_t(a)] == gc[size_t(b)] && a < b); });
          int started = 0;
          for (int e : cand) {
            if (started >= pf_maxn) break;
            if (xpf_find(X, c.E, guess_layer, e) >= 0) continue;
            if (xpf_start(m, X, guess_layer, e, false, true) < 0) break;
            ++started;
            ++st.pf_issued;
            st.pf_bytes += expert_bytes;
          }
          for (int e : cand) gc[size_t(e)] = 0;
        }
      }
      lk.unlock();
      X.cv_job.notify_all();
      if (!rest.empty()) {
        st.bytes += load_expert_slots(m, enc, li, rest, P, *m.xstage, defer ? &m.xdefer : nullptr);
        st.loads += rest.size();
      }
      if (!ents.empty()) {
        // The copies are encoded now (or deferred); the commit waits (spinning) for the reads.
        for (auto [idx, sl] : ents) {
          X.await.push_back(idx);
          if (defer) {
            ExpertPool* PP = &P;
            m.xdefer.push_back([&m, idx, sl, PP](id<MTLComputeCommandEncoder> e) { xpf_repack(m, e, *m.xpf, idx, *PP, sl); });
          } else {
            xpf_repack(m, enc, X, idx, P, sl);
          }
        }
        st.bytes += ents.size() * expert_bytes;
        st.loads += ents.size();
        // The pool holds these experts now; the entries stay unused until the
        // GPU work that reads them has completed (busy = this handshake).
        lk.lock();
        for (auto [idx, sl] : ents) {
          ExpertPrefetch::Entry& en = *X.entries[size_t(idx)];
          X.where.erase(int64_t(en.layer) * c.E + en.expert);
          en.layer = en.expert = -1;
        }
        lk.unlock();
      }
    } else if (!loads.empty()) {
      st.bytes += load_expert_slots(m, enc, li, loads, P, *m.xstage, defer ? &m.xdefer : nullptr);
      st.loads += loads.size();
    }
    int32_t* map = static_cast<int32_t*>([m.xmap contents]);
    for (int e = 0; e < c.E; ++e) map[e] = P.slot_of[size_t(e)];
    if (defer && !loads.empty()) {
      uint8_t* ld = static_cast<uint8_t*>([m.xld contents]);
      for (auto [e, sl] : loads) {
        ld[sl] = 1;
        m.xld_set.push_back(sl);
      }
    }
    return !loads.empty();
  };
  auto q4_route = [&](const Layer& L, int rows) -> const ExpertPool* {
    const int li = int(&L - m.layers.data());
    ExpertPool& P = m.xpool[size_t(li)];
    ExpertCacheStats& st = m.xstats;
    auto t0 = Clock::now();
    m.xsplit = false;
    if (q4_evpipe) {
      // Handshake on the GPU timeline; the host serves it after the commit.
      const uint64_t v = ++m.xev_v;
      E.event_signal_wait(m.xevA, v, m.xevB, v);
      q4_pend.push_back({li, rows, v});
      return &P;  // moe_sort maps the ids to slots (m.xmap, written by the host before it signals)
    }
    // Early guess: layer li + 1's MLP hyper-connection read and router
    // applied to the current streams (ESPETERO_Q4_PF=2, the default)
    // or its router alone on this layer's MoE input (1); top 16 per row.
    static const bool pf_noguess = env_on("ESPETERO_Q4_PF_NOGUESS");  // timing probe: entries without the GPU guess
    // Adaptive: guess only for layers that still miss (EWMA of misses per
    // handshake >= ESPETERO_Q4_PF_MIN; 0: every layer).
    static const float pf_min = [] {
      const char* v = std::getenv("ESPETERO_Q4_PF_MIN");
      return v && *v ? float(std::atof(v)) : 0.0f;
    }();
    // Multi-row passes too: up to ESPETERO_Q4_PF_ROWS rows (default 256:
    // decode and verify passes), not prompt passes.
    static const int pf_rows = [] {
      const char* v = std::getenv("ESPETERO_Q4_PF_ROWS");
      return v && *v ? std::max(1, std::atoi(v)) : 256;
    }();
    const bool guess = pf_mode > 0 && !pf_noguess && rows <= pf_rows && li + 1 < c.L && bool(m.layers[size_t(li + 1)].router_f16) &&
                       (pf_min <= 0.0f || m.xmiss.size() != size_t(c.L) || m.xmiss[size_t(li + 1)] >= pf_min);
    if (guess) {
      Model::Q4Scratch& q = m.q4s;
      const Layer& Ln = m.layers[size_t(li + 1)];
      id<MTLBuffer> X = s.xn;
      if (pf_mode == 2) {
        q4_read(Ln.hcm, false, q.H, rows, q.pfx);
        X = q.pfx;
      }
      if (rows <= 8) {
        E.pipe("q4_router_gemv");
        E.set(0, Ln.router_f16);
        E.set(1, X);
        E.set(2, q.pflog);
        E.bytes(3, c.E);
        E.bytes(4, c.D);
        E.rd(X);
        E.wr(q.pflog);
        E.groups(MTLSizeMake(size_t((c.E + 7) / 8), size_t(rows), 1), MTLSizeMake(256, 1, 1));
      } else {
        gemm(E, m.tune, Ln.router, X, 0, c.D, q.pflog, 0, c.E, rows, 0);  // the router once for all rows
      }
      const MoeRouteParams rp{c.E, 16, rows};
      E.pipe("moe_topk");
      E.set(0, q.pflog);
      E.set(1, q.pfids);
      E.set(2, q.pfw);
      E.bytes(3, rp);
      E.rd(q.pflog);
      E.wr(q.pfids);
      E.wr(q.pfw);
      E.groups(MTLSizeMake(size_t(rows), 1, 1), MTLSizeMake(size_t((c.E + 31) / 32 * 32), 1, 1));
    }
    if (cb_log_ms() > 0) std::snprintf(E.phase, sizeof(E.phase), "layer %d, %d rows", li, rows);
    st.gpu_ms += E.flush();  // the router ran; its ids are readable
    if (fault_mid && li == c.L / 2) fail("GPU command buffer failed: injected (ESPETERO_FAIL_PASS)");
    if (m.xpf) {
      std::lock_guard<std::mutex> lk(m.xpf->mu);
      ++m.xpf->hs;  // GPU work encoded before this flush has completed
    }
    ++st.handshakes;
    st.wait_ms += ms_since(t0);
    t0 = Clock::now();
    // Split handshake (ESPETERO_Q4_SPLIT=0 turns it off): the MoE's tiles of resident
    // experts are committed before the reads of the missing ones are done
    // (8-row MoE kernel; records read by the reader pool, whose copies can
    // wait). The previous handshake's flags are cleared: its work completed
    // in the flush above.
    if (!m.xld_set.empty()) {
      uint8_t* ld = static_cast<uint8_t*>([m.xld contents]);
      for (int i : m.xld_set) ld[i] = 0;
      m.xld_set.clear();
    }
    m.xsplit = false;
    m.xsplit_cb = nil;
    if (m.xsplit_last) {  // the previous split's resident tiles (completed before the flush above)
      st.gpu_ms += gpu_ms(m.xsplit_last);
      m.xsplit_last = nil;
    }
    const bool can_split = q4_split && m.xpf && !m.xs.rec_fds.empty() && !(m.moe_fma && rows <= m.moe_fma_rows) &&
                           (m.moe_stages & 6) == 6;
    if (can_split && !m.xld) {
      m.xld = m.buffer(size_t(2 * c.E + 64));
      std::memset([m.xld contents], 0, size_t(2 * c.E + 64));
    }
    const int32_t* ids = static_cast<const int32_t*>([m.mz.ids contents]);
    const int n = rows * c.k;
    std::vector<int>& need = m.xneed;
    std::vector<int>& cnt = m.xcnt;  // all zero between handshakes (reset below through `need`)
    need.clear();
    if (cnt.size() != size_t(c.E)) cnt.assign(size_t(c.E), 0);
    for (int i = 0; i < n; ++i) {
      if (cnt[size_t(ids[i])]++ == 0) need.push_back(ids[i]);
      st.hits += P.slot_of[size_t(ids[i])] >= 0 ? 1 : 0;
    }
    struct CntReset {  // every return path leaves the counts zero
      std::vector<int>& need;
      std::vector<int>& cnt;
      ~CntReset() {
        for (int e : need) cnt[size_t(e)] = 0;
      }
    } cnt_reset{need, cnt};
    pin_update(P, need, cnt);
    st.picks += uint64_t(n);
    ++m.xclock;
    if (!m.xstage) m.xstage = std::make_unique<Arena>(m.dev, size_t(64) << 20);
    m.xstage->reset();  // the previous staging was read by the flushed work
    // Free slots first, then least recently used ones whose expert is not in `keep`.
    auto victims = [&](const std::vector<char>& keep) {
      std::vector<int> v;
      for (int sl = 0; sl < P.S; ++sl)
        if (P.expert_of[size_t(sl)] < 0 || !keep[size_t(P.expert_of[size_t(sl)])]) v.push_back(sl);
      std::sort(v.begin(), v.end(), [&](int a, int b) { return P.evict_key(a, xpolicy) < P.evict_key(b, xpolicy); });
      return v;
    };
    auto place = [&](int e, int sl) {
      if (P.expert_of[size_t(sl)] >= 0) P.slot_of[size_t(P.expert_of[size_t(sl)])] = -1;
      P.expert_of[size_t(sl)] = e;
      P.slot_of[size_t(e)] = sl;
      P.last_use[size_t(sl)] = m.xclock;
      m.xpass.push_back({li, sl});
    };
    if (int(need.size()) > P.S && !m.xs.rec_fds.empty() && rows > m.moe_fma_rows && m.xpf) {
      // More experts than slots (long prompts), with the record store:
      // the MoE reads the resident experts in place and the missing ones
      // straight from their records, read into a staging area (its third
      // matrix: experts S .. S + Q - 1). The pass's most used experts are then
      // admitted: record -> slot copies encoded after the MoE (which still
      // reads the slots they overwrite), so the plan may evict any expert.
      const size_t RS = m.xs.rec_stride, RB = m.xs.rec_bytes;
      if (xpolicy == 2) {
        if (P.e_last.empty()) P.e_last.assign(size_t(c.E), 0), P.e_prev.assign(size_t(c.E), 0);
        for (int e : need) {
          P.e_prev[size_t(e)] = P.e_last[size_t(e)];
          P.e_last[size_t(e)] = m.xclock;
        }
      }
      std::vector<int> miss;
      for (int e : need)
        if (P.slot_of[size_t(e)] < 0) miss.push_back(e);
      const int Q = int(miss.size());
      const size_t per = align_up((RB + size_t(record_chunks()) - 1) / size_t(record_chunks()), 16384);
      auto record_jobs = [&](int layer, int e, uint8_t* dst, std::vector<ExpertPrefetch::Job>& js, size_t chunk) {
        for (size_t o = 0; o < RB; o += chunk)
          js.push_back({m.xs.rec_fds[size_t(layer)], dst + o, std::min(chunk, RB - o), uint64_t(e) * RS + o, -1, nullptr});
      };
      std::vector<int> jof(size_t(c.E), -1);
      Buf stg;
      int st_count = Q;
      // Pipeline (ESPETERO_XPIPE=1): staging in pipe buffer li % 2, which may
      // already hold this layer's prefetched records; others are appended.
      static const bool xpipe_on = env_on("ESPETERO_XPIPE");
      Model::XPipe& XP = m.xpipe;
      ExpertPrefetch& XF = *m.xpf;
      const int pb = li % 2;
      bool piped = false;
      if (xpipe_on) {
        if (!XP.buf[0]) {
          XP.cap = size_t(c.E - P.S + 16);
          for (int k = 0; k < 2; ++k) {
            XP.buf[k] = m.buffer(XP.cap * RS);
            XP.idx[k].assign(size_t(c.E), -1);
            XP.layer[k] = -1;
            XP.count[k] = 0;
          }
        }
        if (XP.layer[pb] != li) {  // nothing prefetched for this layer: start empty
          while (XF.st_pending[pb].load(std::memory_order_acquire) > 0) {
          }
          std::fill(XP.idx[pb].begin(), XP.idx[pb].end(), -1);
          XP.count[pb] = 0;
        }
        int extra = 0;
        for (int e : miss) extra += XP.idx[pb][size_t(e)] < 0 ? 1 : 0;
        if (size_t(XP.count[pb] + extra) <= XP.cap) {
          piped = true;
          stg = Buf(XP.buf[pb], 0);
          std::vector<ExpertPrefetch::Job> js;
          int got = 0;
          for (int e : miss) {
            int& ix = XP.idx[pb][size_t(e)];
            if (ix < 0) {
              ix = XP.count[pb]++;
              record_jobs(li, e, static_cast<uint8_t*>([XP.buf[pb] contents]) + size_t(ix) * RS, js, per);
            } else {
              ++got;
            }
            jof[size_t(e)] = ix;
          }
          xio_submit(XF, js);  // demand reads: the commit waits for them (and for the prefetched ones, st_need)
          XF.st_need = pb;
          st_count = XP.count[pb];
          st.bytes += size_t(Q - got) * RB;
          st.loads += uint64_t(Q - got);
          st.pf_used += uint64_t(got);
        }
        XP.layer[pb] = -1;
      }
      if (!piped) {
        stg = alloc_aligned(*m.xstage, std::max<size_t>(size_t(Q) * RS, 16384));
        std::vector<ExpertPrefetch::Job> js;
        for (int j = 0; j < Q; ++j) {
          record_jobs(li, miss[size_t(j)], static_cast<uint8_t*>(ptr(stg)) + size_t(j) * RS, js, per);
          jof[size_t(miss[size_t(j)])] = j;
        }
        xio_submit(XF, js);  // the commit waits for them
        st.bytes += size_t(Q) * RB;
        st.loads += uint64_t(Q);
      }
      int32_t* map = static_cast<int32_t*>([m.xmap contents]);
      for (int e = 0; e < c.E; ++e) map[e] = P.slot_of[size_t(e)];
      for (int e : miss) map[e] = P.S + jof[size_t(e)];
      // Next layer (very long passes: nearly every expert is routed): its
      // non-resident experts read into the other pipe buffer while this
      // layer's MoE and the next layer's attention run.
      static const float xpipe_min = [] {  // fraction of the experts routed in this layer that starts the next layer's reads
        const char* v = std::getenv("ESPETERO_XPIPE_MIN");
        return v && *v ? float(std::atof(v)) : 0.9f;
      }();
      static const bool xpipe_log = env_on("ESPETERO_XPIPE_LOG");
      if (xpipe_log) std::fprintf(stderr, "long pass layer %d: rows %d, routed experts %zu, missing %d%s\n", li, rows, need.size(), Q, piped ? " (piped)" : "");
      if (xpipe_on && li + 1 < c.L && float(need.size()) >= xpipe_min * float(c.E)) {
        const int nb = (li + 1) % 2;
        const ExpertPool& Pn = m.xpool[size_t(li + 1)];
        while (XF.st_pending[nb].load(std::memory_order_acquire) > 0) {
        }
        std::fill(XP.idx[nb].begin(), XP.idx[nb].end(), -1);
        XP.count[nb] = 0;
        std::vector<ExpertPrefetch::Job> js;
        for (int e = 0; e < c.E && size_t(XP.count[nb]) < XP.cap; ++e)
          if (Pn.slot_of[size_t(e)] < 0) {
            const int ix = XP.count[nb]++;
            XP.idx[nb][size_t(e)] = ix;
            record_jobs(li + 1, e, static_cast<uint8_t*>([XP.buf[nb] contents]) + size_t(ix) * RS, js, RB);
          }
        xio_submit(XF, js, &XF.st_pending[nb], false);  // behind the demand reads
        XP.layer[nb] = li + 1;
        st.pf_issued += uint64_t(XP.count[nb]);
        st.pf_bytes += size_t(XP.count[nb]) * RB;
      }
      std::vector<int> order = need;
      std::sort(order.begin(), order.end(), [&](int a, int b) { return cnt[size_t(a)] > cnt[size_t(b)]; });
      order.resize(size_t(P.S));
      std::vector<char> keep(size_t(c.E), 0);
      for (int e : order) keep[size_t(e)] = 1;
      const std::vector<int> vs = victims(keep);
      size_t vi = 0;
      m.xadmit.clear();
      for (int e : order) {
        if (P.slot_of[size_t(e)] >= 0) {
          P.last_use[size_t(P.slot_of[size_t(e)])] = m.xclock;
          continue;
        }
        const int sl = vs[vi++];
        place(e, sl);
        m.xadmit.push_back({jof[size_t(e)], sl});
        ++st.admits;
      }
      m.xadmit_src = stg;
      P.st_count = st_count;
      P.st = stg;
      P.st_stride = RS;
      if (can_split && st_count > 0) {  // the staged records are the experts being read
        uint8_t* ld = static_cast<uint8_t*>([m.xld contents]);
        for (int j = 0; j < st_count; ++j) {
          ld[P.S + j] = 1;
          m.xld_set.push_back(P.S + j);
        }
        m.xsplit = true;
      }
      ++st.fallbacks;
      q4_swept = true;
      st.load_ms += ms_since(t0);
      return &P;
    }
    if (int(need.size()) > P.S) {
      // More experts than slots (long prompts): this pass runs on a whole-layer
      // staging matrix that holds just the experts it needs: the resident ones
      // copied from the pool on the GPU, the others read from the SSD (a sweep:
      // the SSD serves only misses); then its most used experts are admitted.
      Layer& Lm = m.layers[size_t(li)];
      ExpertPool full;  // staging with E slots, slot = expert id
      full.S = c.E;
      auto stage = [&](const QW& fmt) {
        QW q = fmt;  // the loader's format of the whole-layer matrix
        q.wr = m.xstage->alloc(size_t(q.out) * size_t(q.in) / 2);
        q.sb = m.xstage->alloc(size_t(q.out / 2) * size_t(q.in / q.group) * 8);
        return q;
      };
      full.gu = stage(Lm.experts_gu);
      full.dn = stage(Lm.experts_down);
      std::vector<std::pair<int, int>> reads;
      for (int e : need) {
        const int sl = P.slot_of[size_t(e)];
        if (sl >= 0) {
          q4_admit(P.gu, full.gu, sl, e, 2 * c.Ie);
          q4_admit(P.dn, full.dn, sl, e, c.D);
        } else {
          reads.push_back({e, e});
        }
      }
      if (!reads.empty()) {
        st.bytes += load_expert_slots(m, E.e, li, reads, full, *m.xstage);
        st.loads += reads.size();
      }
      E.barrier();
      Lm.experts_gu = full.gu;
      Lm.experts_down = full.dn;
      std::vector<int> order = need;
      std::sort(order.begin(), order.end(), [&](int a, int b) { return cnt[size_t(a)] > cnt[size_t(b)]; });
      order.resize(size_t(P.S));
      std::vector<char> keep(size_t(c.E), 0);
      for (int e : order) keep[size_t(e)] = 1;
      const std::vector<int> vs = victims(keep);
      size_t vi = 0;
      for (int e : order) {
        if (P.slot_of[size_t(e)] >= 0) {
          P.last_use[size_t(P.slot_of[size_t(e)])] = m.xclock;
          continue;
        }
        const int sl = vs[vi++];
        place(e, sl);
        q4_admit(full.gu, P.gu, e, sl, 2 * c.Ie);
        q4_admit(full.dn, P.dn, e, sl, c.D);
        ++st.admits;
      }
      ++st.fallbacks;
      q4_swept = true;
      st.load_ms += ms_since(t0);
      return nullptr;
    }
    if (q4_load_routed(li, need, E.e, guess ? li + 1 : -1, rows, can_split)) {
      if (can_split) m.xsplit = true;  // moe_mlp encodes the copies after the resident tiles (and a barrier)
      else E.barrier();
    }
    st.load_ms += ms_since(t0);
    return &P;  // moe_sort maps the ids to slots (m.xmap)
  };
  if (c.q4) {
    ensure_expert_pools(m, m.expert_cache_gb);
    ensure_q4_scratch(m, M);
    m.xdefer.clear();  // left by a failed pass (its admissions were rolled back)
    m.xsplit = false;
    if (m.xsplit_last) m.xstats.gpu_ms += gpu_ms(m.xsplit_last);  // the previous pass's last split (completed)
    m.xsplit_cb = m.xsplit_last = nil;
    if (!m.xpool.empty()) {
      ensure_prefetch(m);  // the reader pool; the commit waits for the reads its copies depend on
      E.before_commit = [&m] {
        const auto t = Clock::now();
        if (m.xsplit_cb) {
          // The resident tiles run; the next command buffer is committed
          // after they completed (the pools are untracked, and a failed one
          // stops the pass here). Poll sleeping while the GPU runs (a spinning
          // host halves the speed of short GPU work), spin on the reads after.
          while ([m.xsplit_cb status] < MTLCommandBufferStatusCompleted) usleep(50);
          if ([m.xsplit_cb status] != MTLCommandBufferStatusCompleted) {
            NSError* err = [m.xsplit_cb error];
            m.xsplit_cb = nil;
            fail(std::string("GPU command buffer failed: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
          }
          m.xsplit_cb = nil;
          m.xstats.split_ms += ms_since(t);
        }
        xio_wait(*m.xpf);
        m.xstats.pf_wait_ms += ms_since(t);  // host time spent waiting for reads before commits
      };
    }
    static const bool no_ev = !env_on("ESPETERO_Q4_EV");  // opt-in: measured slower than per-layer flushes
    q4_evpipe = !no_ev && !m.profile && !m.xpool.empty() && M * c.k <= m.xpool[0].S;
    if (q4_evpipe && !m.xevA) {
      m.xevA = [m.dev newSharedEvent];
      m.xevB = [m.dev newSharedEvent];
      m.xqueue = [m.dev newCommandQueue];
      Residency::get().attach(m.dev, m.xqueue);
      m.xev_v = 0;
    }
    Model::Q4Scratch& q = m.q4s;
    ensure_ple_state(m);
    // n-gram rows (hashed on the host, read from the SSD) and conv taps per row.
    const int heads = 2 * c.ngram_heads;
    std::vector<int64_t> nrows(size_t(M) * size_t(heads));
    int32_t* psrc = (int32_t*)[q.psrc contents];
    int32_t* pslot = (int32_t*)[q.pslot contents];
    const int64_t eos = c.eos_token_id;
    for (const TreePlan& T : trees) {
      const std::array<int32_t, 2> ctx = m.ple_ctx[size_t(T.pstate)];
      for (int i = 0; i < T.Tr; ++i) {
        const int r = T.base + i;
        auto id_at = [&](int j) -> int64_t { return j >= 0 ? int64_t(tok[T.base + j]) : int64_t(j == -1 ? ctx[1] : ctx[0]); };
        const int64_t c1 = id_at(i - 1);
        const int64_t c2 = c1 == eos ? eos : id_at(i - 2);
        ngram_rows(m.ple.table, c.ngram_heads, int64_t(tok[r]), c1, c2, &nrows[size_t(r) * size_t(heads)]);
        for (int k = 0; k < 3; ++k) {
          const int p = i - (3 - k) * c.ngram;  // taps t - 9, t - 6, t - 3
          psrc[r * 3 + k] = p >= 0 ? T.base + p : -1 - (Model::kPleHist + p);
        }
        pslot[r] = T.pstate;
      }
      // Leaf rows: a host leaf's rows continue the root; sequence index
      // x >= Tr is the leaf's row x - Tr, 0 <= x < Tr a root row, x < 0 the
      // parent's context (ids) or history (taps), as for the root rows.
      for (int li = 0; li < T.NL; ++li) {
        const int h = T.host[size_t(li)];
        if (h != li) continue;  // a prefix of its host leaf: rows owned by the host
        const int off = T.off[size_t(h)];
        const int len = T.spec->leaf_offsets[li + 1] - T.spec->leaf_offsets[li];
        auto row_of = [&](int x) { return x >= T.Tr ? T.base + off + x : T.base + x; };
        auto id_seq = [&](int x) -> int64_t { return x >= 0 ? int64_t(tok[row_of(x)]) : int64_t(x == -1 ? ctx[1] : ctx[0]); };
        for (int j = 0; j < len; ++j) {
          const int x = T.Tr + j;
          const int r = row_of(x);
          const int64_t c1 = id_seq(x - 1);
          const int64_t c2 = c1 == eos ? eos : id_seq(x - 2);
          ngram_rows(m.ple.table, c.ngram_heads, int64_t(tok[r]), c1, c2, &nrows[size_t(r) * size_t(heads)]);
          for (int k = 0; k < 3; ++k) {
            const int p = x - (3 - k) * c.ngram;
            psrc[r * 3 + k] = p >= 0 ? row_of(p) : -1 - (Model::kPleHist + p);
          }
          pslot[r] = T.pstate;
        }
      }
    }
    ngram_gather(m.ple.table, heads, nrows, size_t(M), (__fp16*)[q.e16 contents]);
    // The embedding (and image rows) in every stream.
    E.pipe("q4_embed_streams");
    E.set(0, s.h);
    E.set(1, q.H);
    E.bytes(2, c.D);
    E.bytes(3, c.hc);
    E.rd(s.h);
    E.wr(q.H);
    E.threads(MTLSizeMake(size_t(c.D / 4), size_t(M), 1), MTLSizeMake(64, 1, 1));
  }
  if (c.gemma) {  // Gemma 4 decoder layers (gemma.h)
    G4PassIn gp;
    gp.M = M;
    gp.N = N;
    gp.prune = prune;
    gp.chained = chained;
    gp.plan_s = &g4s;
    gp.plan_g = &g4g;
    gp.trees_s = meta.s_trees.buf;
    gp.tiles_s = meta.s_tiles.buf;
    gp.comb_s = meta.s_comb.buf;
    gp.trees_g = meta.a_trees.buf;
    gp.tiles_g = meta.a_tiles.buf;
    gp.comb_g = meta.a_comb.buf;
    gp.parts = meta.parts.buf;
    gp.parts_bytes = meta.parts.bytes;
    gp.saves_s = meta.ring_saves.buf;
    gp.n_saves_s = n_ring_saves;
    gp.saves_g = meta.kv_saves.buf;
    gp.n_saves_g = kv_saves.size();
    gp.blocks = meta.blocks.buf;
    gp.desc = meta.desc.buf;
    gp.pos = meta.pos.buf;
    gp.readout_rows = meta.readout_rows.buf;
    g4_encode_layers(m, E, gp);
  }
  // (The Qwen3.5-family layers; Gemma 4 ran its own above.)
  for (int li = 0; !c.gemma && li < c.L; ++li) {
    const Layer& L = m.layers[size_t(li)];
    const bool last_layer = li == c.L - 1 && prune;
    NormLink l_post, l_next;
    const NormLink* nin = nullptr;  // (consumer-side norms are no longer used)
    if (fuse) {
      l_post = link(2 * li + 1, L.post_norm);
      if (li + 1 < c.L) l_next = link(2 * (li + 1), m.layers[size_t(li + 1)].in_norm);
    }
    // Layer 0's input norm runs as a dispatch (the embedding has no partials);
    // every later norm is written by the GEMV that produced its input.
    if (c.q4) {
      if (li == c.ple_layer) q4_ple();
      q4_read(L.hca, true, m.q4s.H, M, s.xn);
    } else if (!fuse || li == 0) {
      rmsnorm(s.h, L.in_norm, s.xn, M, nil);
    }
    id<MTLBuffer> x_in = s.xn;
    const NormLink* out_link = fuse ? &l_post : nullptr;
    if (L.linear) {
      const int k = L.index_in_kind;
      gemm_runs(E, m.tune, L.in_proj, x_in, 0, c.D, s.proj, 0, ldl, M, 0, nin);
      if (cm_total > 0) {  // the commit chains' DeltaNet inputs of this layer
        const size_t loff = size_t(k) * size_t(m.cm_cap) * size_t(ldl) * sizeof(float);
        E.pipe("gather_rows_f32");
        E.set(0, s.proj);
        E.set(1, m.cm_x, loff);
        E.set(2, meta.cm_rows.buf);
        E.bytes(3, ldl);
        E.rd(s.proj);
        E.wr(m.cm_x, loff, loff + size_t(cm_total) * size_t(ldl) * sizeof(float));
        E.threads(MTLSizeMake(size_t(ldl / 4), size_t(cm_total), 1), MTLSizeMake(64, 1, 1));
      }
      const uint64_t conv_base = S.conv_base[size_t(k)], st_base = S.st_base[size_t(k)];
      ConvParams cp{C, ldl, C + Vd, c.Hk, c.Hv, c.qk_eps, conv_base, S.conv_floats};
      const size_t conv_lo = conv_base * 4, conv_hi = (conv_base + S.conv_floats * size_t(S.n_slots)) * 4;
      const size_t st_lo = st_base * 4, st_hi = (st_base + S.st_floats * size_t(S.n_slots)) * 4;
      DeltaParams dp{c.Hk, c.Hv, C, c.tiled_heads ? 1 : 0, st_base, S.st_floats};
      // Fused decode path: conv + gates + recurrence + gated norm per
      // (segment, value head) threadgroup, roots then leaves; conv state
      // saves follow as their own dispatch (see the kernel comment).
      const bool gdn_fused = gdn_fused_pass;
      if (gdn_fused) {
        auto fused = [&](HostBuf& segs, size_t nseg, int R, bool writes, bool csave) {
          if (!nseg) return;
          const DeltaFusedParams fp{c.Hk, c.Hv, C, ldl, C + Vd, c.tiled_heads ? 1 : 0, c.eps, C, Vd,
                                    conv_base, S.conv_floats, st_base, S.st_floats, csave ? 1 : 0};
          const std::string name = "gdn_decode_fused_tr" + std::to_string(R);
          const size_t nt = 32 * size_t(c.Dv / R);
          if ([m.kernel(name) maxTotalThreadsPerThreadgroup] < nt) fail(name + ": threadgroup too small");
          E.pipe(name);
          E.set(0, s.proj);
          E.set(1, S.buf);
          E.set(2, meta.conv_src.buf);
          E.set(3, L.conv_w);
          E.set(4, meta.row_slot.buf);
          E.set(5, L.a_log);
          E.set(6, L.dt_bias);
          E.set(7, L.gnorm);
          E.set(8, s.yn);
          E.set(9, segs.buf);
          E.bytes(10, fp);
          if (csave) {
            E.set(11, meta.conv_root.buf);
            E.set(12, m.gdn_ctr);
            E.rd(m.gdn_ctr);
            E.wr(m.gdn_ctr);
            E.wr(S.buf, conv_lo, conv_hi);
          }
          E.rd(s.proj);
          E.rd(S.buf, conv_lo, conv_hi);
          E.rd(S.buf, st_lo, st_hi);
          if (writes) E.wr(S.buf, st_lo, st_hi);
          E.wr(s.yn);
          E.threads(MTLSizeMake(nt, 1, nseg * size_t(c.Hv)), MTLSizeMake(nt, 1, 1));
        };
        // Few sequences: more, thinner simdgroups hide the per-step latency.
        const int Rf_root = m.dn_rows > 0 ? m.dn_rows : (roots.size() <= 1 ? 8 : 16);
        const int Rf_leaf = m.dn_rows > 0 ? m.dn_rows : 16;
        fused(meta.segs_root, roots.size(), Rf_root, true, conv_fold);
        fused(meta.segs_leaf, leaves.size(), Rf_leaf, false, false);
      } else {
      E.pipe("conv_prep");
      E.set(0, s.proj);
      E.set(1, S.buf);
      E.set(2, meta.conv_src.buf);
      E.set(3, L.conv_w);
      E.set(4, s.qo);
      E.set(5, s.gb);
      E.set(6, L.a_log);
      E.set(7, L.dt_bias);
      E.bytes(8, cp);
      E.set(9, meta.row_slot.buf);
      E.rd(s.proj);
      E.rd(S.buf, conv_lo, conv_hi);
      E.wr(s.qo);
      E.wr(s.gb);
      E.groups(MTLSizeMake(size_t(2 * c.Hk + c.Hv + 3) / 4, size_t(M), 1), MTLSizeMake(128, 1, 1));
      }
      if (!conv_jobs.empty() && !(gdn_fused && conv_fold)) {
        E.pipe("conv_state_save");
        E.set(0, s.proj);
        E.set(1, S.buf);
        E.set(2, meta.conv_jobs.buf);
        E.bytes(3, cp);
        E.rd(s.proj);
        E.rd(S.buf, conv_lo, conv_hi);
        E.wr(S.buf, conv_lo, conv_hi);
        E.threads(MTLSizeMake(size_t(C), conv_jobs.size(), 1), MTLSizeMake(256, 1, 1));
      }
      auto delta = [&](HostBuf& segs, size_t nseg, int R, bool writes, int max_len) {
        if (!nseg) return;
        // Long segments (prefill): 8-token chunks with MMAs; short ones
        // (decode steps, draft leaves): the per-token recurrence.
        // Never with commit chains: a commit replays their rows with deltanet_tr.
        const bool chunked = m.dn_chunk > 0 && max_len >= m.dn_chunk && cm_total == 0;
        if (chunked) R = 8;
        E.pipe(chunked ? std::string("deltanet_chunk") : "deltanet_tr" + std::to_string(R));
        E.set(0, s.qo);
        E.set(1, s.gb);
        E.set(2, S.buf);
        E.set(4, s.y);
        E.set(5, segs.buf);
        E.bytes(6, dp);
        E.rd(s.qo);
        E.rd(s.gb);
        E.rd(S.buf, st_lo, st_hi);
        if (writes) E.wr(S.buf, st_lo, st_hi);
        E.wr(s.y);
        E.threads(MTLSizeMake(32, size_t(c.Dv / R), nseg * size_t(c.Hv)), MTLSizeMake(32, 4, 1));
      };
      // Few sequences: more, thinner simdgroups hide the per-step latency.
      const int R_root = m.dn_rows > 0 ? m.dn_rows : (roots.size() <= 1 ? 8 : 16);
      const int R_leaf = m.dn_rows > 0 ? m.dn_rows : 16;
      if (!gdn_fused) {
        delta(meta.segs_root, roots.size(), R_root, true, max_root_len);
        delta(meta.segs_leaf, leaves.size(), R_leaf, false, max_leaf_len);
      }
      // Fused passes feed y and z straight into the out projection (gated
      // per-head norm on load; needs head-aligned K slices of 8 simdgroups).
      const bool gated_fused = fuse && m.fuse_gated && ((Vd / 64 + 7) / 8) % 2 == 0;
      if (gated_fused) {
        NormLink g;
        g.kind = 2;
        g.parts = Buf(s.proj, sizeof(float) * size_t(C));
        g.ld = ldl;
        g.P = 0;
        g.D = Vd;
        g.eps = c.eps;
        g.w = L.gnorm;
        gemm(E, m.tune, L.out, s.y, 0, Vd, s.h, 0, c.D, M, 1, 0, &g, out_link);
      } else if (gdn_fused) {
        // yn was written by the fused kernel
        if (last_layer) {
          if (N == 0) break;
          gather(s.yn, s.mixc, Vd, false);
          gather(s.h, s.hc, c.D, true);
          gemm(E, m.tune, L.out, s.mixc, 0, Vd, s.hc, 0, c.D, N, 1);
        } else {
          gemm(E, m.tune, L.out, s.yn, 0, Vd, s.h, 0, c.D, M, 1, 0, nullptr, out_link);
        }
      } else {
      E.pipe(c.gate_sigmoid ? "gated_rmsnorm_sig" : "gated_rmsnorm");
      E.set(0, s.y);
      E.set(1, s.proj, sizeof(float) * size_t(C));
      E.set(2, L.gnorm);
      E.set(3, s.yn);
      E.bytes(4, Vd);
      E.bytes(5, ldl);
      E.bytes(6, c.eps);
      E.rd(s.y);
      E.rd(s.proj);
      E.wr(s.yn);
      E.groups(MTLSizeMake(size_t(c.Hv + 3) / 4, size_t(M), 1), MTLSizeMake(128, 1, 1));
      if (last_layer) {
        if (N == 0) break;
        gather(s.yn, s.mixc, Vd, false);
        gather(s.h, s.hc, c.D, true);
        gemm(E, m.tune, L.out, s.mixc, 0, Vd, s.hc, 0, c.D, N, 1);
      } else if (c.q4) {
        gemm(E, m.tune, L.out, s.yn, 0, Vd, m.q4s.y, 0, c.D, M, 0);
      } else {
        gemm(E, m.tune, L.out, s.yn, 0, Vd, s.h, 0, c.D, M, 1, 0, nullptr, out_link);
      }
      }
    } else {
      const int k = L.index_in_kind;
      gemm_runs(E, m.tune, L.in_proj, x_in, 0, c.D, s.proj, 0, ldf, M, 0, nin);
      attn_prep(L, meta.pos.buf, 0, M);
      if (qsa_idx) qsa_index(m, E, qs, L, x_in, meta.pos.buf, meta.a_trees.buf, meta.desc.buf, meta.blocks.buf, k, false);
      if (qs.n_sparse) qsa_select(m, E, qs, meta.a_trees.buf, meta.desc.buf, meta.blocks.buf, k, false);
      attention(A, meta.a_trees, meta.a_tiles, meta.a_comb, meta.desc.buf, k, qs.n_sparse ? &qs : nullptr);
      kv_save(meta.kv_saves, kv_saves.size(), k);
      if (qsa_idx) qsa_save(m, E, qs, meta.pos.buf, meta.kv_saves.buf, kv_saves.size(), k, false);
      if (last_layer) {
        if (N == 0) break;
        gather(s.O, s.mixc, qdim, false);
        gather(s.h, s.hc, c.D, true);
        gemm(E, m.tune, L.out, s.mixc, 0, qdim, s.hc, 0, c.D, N, 1);
      } else if (c.q4) {
        gemm(E, m.tune, L.out, s.O, 0, qdim, m.q4s.y, 0, c.D, M, 0);
      } else {
        gemm(E, m.tune, L.out, s.O, 0, qdim, s.h, 0, c.D, M, 1, 0, nullptr, out_link);
      }
    }
    if (c.q4) {
      q4_inject(M);
      q4_read(L.hcm, true, m.q4s.H, M, s.xn);
      if (m.xpool.empty()) q4_experts(li);
      E.pipe("q4_zero");
      const uint32_t ny = uint32_t(M) * uint32_t(c.D);
      E.set(0, m.q4s.y);
      E.bytes(1, ny);
      E.wr(m.q4s.y);
      E.threads(MTLSizeMake(ny, 1, 1), MTLSizeMake(256, 1, 1));
      if (!m.xpool.empty()) m.moe_after_route = q4_route;
      try {
        moe_mlp(m, E, L, s.xn, s.act, m.q4s.y, M, nullptr);
      } catch (...) {
        m.moe_after_route = nullptr;
        throw;
      }
      m.moe_after_route = nullptr;
      if (!m.xpool.empty() && m.xpool[size_t(li)].st_count > 0) {
        // Long pass: the admitted experts' records into their slots, after the MoE read them.
        ExpertPool& Pa = m.xpool[size_t(li)];
        E.barrier();
        for (auto [j, sl] : m.xadmit)
          admit_record(m, E.e, m.xadmit_src.b, m.xadmit_src.off + size_t(j) * Pa.st_stride, m.xs.rec_o, Pa, sl, c,
                       m.xs.layers[0].group);
        m.xadmit.clear();
        // The staging stays alive in m.xstage / m.xpipe until the pass
        // completes; this reference kept it (up to ~3 GB) until the next long pass.
        m.xadmit_src = Buf{};
        Pa.st_count = 0;
        Pa.st = Buf{};
      }
      // Whole-layer stacks stay alive in m.xstage until the next flush; keep
      // only their format here.
      Layer& Lm = m.layers[size_t(li)];
      Lm.experts_gu.wr = Lm.experts_gu.sb = Lm.experts_gu.wh = Buf{};
      Lm.experts_down.wr = Lm.experts_down.sb = Lm.experts_down.wh = Buf{};
      q4_inject(M);
      // Debug (parity): ESPETERO_Q4_DUMP=dir writes the residual streams after
      // every layer (P<pass>_H<li>.f32, [M][W] fp32) and the pass's n-gram rows (P<pass>_e16.f16).
      static const char* q4dump = std::getenv("ESPETERO_Q4_DUMP");
      if (q4dump) {
        E.flush();
        auto dumpf = [&](const std::string& name, id<MTLBuffer> b, size_t bytes) {
          const std::string path = std::string(q4dump) + "/" + name;
          if (FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite([b contents], 1, bytes, f);
            std::fclose(f);
          }
        };
        static int q4dump_pass = -1;
        if (li == 0) ++q4dump_pass;
        char nm[48];
        std::snprintf(nm, sizeof(nm), "P%03d_H%02d.f32", q4dump_pass, li);
        dumpf(nm, m.q4s.H, size_t(M) * size_t(c.W()) * 4);
        if (li == 0) {
          std::snprintf(nm, sizeof(nm), "P%03d_e16.f16", q4dump_pass);
          dumpf(nm, m.q4s.e16, size_t(M) * size_t(c.D) * 2);
        }
      }
      continue;
    }
    if (last_layer) {
      rmsnorm(s.hc, L.post_norm, s.xc, N, nil);
      if (L.moe) {
        moe_mlp(m, E, L, s.xc, s.actc, s.hc, N, nullptr);
      } else {
        mlp_in(L, s.xc, s.actc, N);
        gemm(E, m.tune, L.down, s.actc, 0, L.I, s.hc, 0, c.D, N, 1);
      }
      break;
    }
    if (!fuse) rmsnorm(s.h, L.post_norm, s.xn, M, nil);
    const NormLink* next_link = fuse && li + 1 < c.L ? &l_next : nullptr;
    if (L.moe) {
      moe_mlp(m, E, L, s.xn, s.act, s.h, M, next_link);
    } else {
      mlp_in(L, s.xn, s.act, M);
      gemm(E, m.tune, L.down, s.act, 0, L.I, s.h, 0, c.D, M, 1, 0, nullptr, next_link);
    }
  }

  bool want_lse = false, want_argmax = false;
  int topk_k = 0;
  for (const TreePlan& T : trees) {
    want_lse = want_lse || (T.N && T.spec->out_lse);
    want_argmax = want_argmax || (T.N && T.spec->out_argmax);
    if (T.N && T.spec->topk > 0) topk_k = std::max(topk_k, std::min(32, T.spec->topk));
    if (T.N && T.spec->topk_min > 0) topk_k = std::max(topk_k, std::min(32, T.spec->topk_min));
  }
  s.tk_k = topk_k;
  if (any_hidden) {  // every row's final-normed state; trees copy their root rows out below
    if (s.hid_rows < M) {
      s.hid = m.buffer(sizeof(uint16_t) * size_t(M) * size_t(c.D));
      s.hid_rows = M;
    }
    rmsnorm(s.h, m.final_norm, s.hid, M, nil);
  }
  if (N > 0) {
    if (c.q4) {  // the final mixer on the rows that are read out
      gather(m.q4s.H, m.q4s.Hc, c.W(), true);
      q4_read(m.mixer, false, m.q4s.Hc, N, s.hn);
    } else if (prune) {
      rmsnorm(s.hc, m.final_norm, s.hn, N, nil);
    } else {
      rmsnorm(s.h, m.final_norm, s.hn, N, meta.readout_rows.buf);
    }
    gemm(E, m.tune, m.lm_head, s.hn, 0, c.D, s.logits, 0, c.V, N, 0);
    if (c.softcap > 0) {  // Gemma 4: cap * tanh(logits / cap) before every readout
      const uint32_t n4 = uint32_t(size_t(N) * size_t(c.V) / 4);
      E.pipe("softcap_rows");
      E.set(0, s.logits);
      E.bytes(1, c.softcap);
      E.bytes(2, n4);
      E.rd(s.logits);
      E.wr(s.logits);
      E.threads(MTLSizeMake(n4, 1, 1), MTLSizeMake(256, 1, 1));
    }
    if (total_cands > 0) {
      E.pipe("gather_cands");
      E.set(0, s.logits);
      E.set(1, meta.cand_ids.buf);
      E.set(2, meta.cand_owner.buf);
      E.set(3, meta.cand_out.buf);
      E.bytes(4, c.V);
      E.bytes(5, total_cands);
      E.rd(s.logits);
      E.wr(meta.cand_out.buf, meta.cand_out.bytes);
      E.threads(MTLSizeMake(size_t(total_cands), 1, 1), MTLSizeMake(64, 1, 1));
    }
    if (want_lse) {
      E.pipe("lse_rows");
      E.set(0, s.logits);
      E.set(1, s.lse);
      E.bytes(2, c.V);
      E.rd(s.logits);
      E.wr(s.lse);
      E.groups(MTLSizeMake(size_t(N), 1, 1), MTLSizeMake(1024, 1, 1));
    }
    if (want_argmax) {
      E.pipe("argmax_rows");
      E.set(0, s.logits);
      if (m.chain_active) {  // the chain's token buffer (the next pass embeds from it)
        E.set(1, m.chain_tok, size_t(m.chain_step) * size_t(m.chain_n) * 4);
      } else {
        E.set(1, s.argmax);
      }
      E.bytes(2, c.V);
      E.bytes(3, c.V);
      E.rd(s.logits);
      if (m.chain_active) E.wr(m.chain_tok);
      else E.wr(s.argmax);
      E.groups(MTLSizeMake(size_t(N), 1, 1), MTLSizeMake(1024, 1, 1));
    }
    // After the argmax: in a chain, a drawn token overwrites the row's argmax in the chain's token buffer.
    if (topk_k > 0) {
      // GPU top-k candidates of every readout; x = logit / temperature per tree.
      float* it = meta.topk_it.ensure<float>(m, size_t(N));
      for (const TreePlan& T : trees)
        for (int i = 0; i < T.N; ++i)
          it[T.readout_base + i] = (T.spec->topk > 0 && T.spec->temperature > 0) ? 1.0f / T.spec->temperature : 1.0f;
      const int parts = (c.V + 4095) / 4096;
      const TopKParams tp{c.V, topk_k, (c.V + parts - 1) / parts, parts, m.chain_active ? 1 : 0,
                          m.chain_active ? m.chain_step * m.chain_n : 0};
      // Per-readout draws; readout positions: draw_pos0 for the root, + the leaf's length for a leaf.
      TopKSampleRec* sr = meta.topk_samp.ensure<TopKSampleRec>(m, size_t(N));
      for (const TreePlan& T : trees) {
        const TreeSpec& b = *T.spec;
        for (int i = 0; i < T.N; ++i) {
          TopKSampleRec& r = sr[T.readout_base + i];
          r = TopKSampleRec{};
          if (!(b.topk > 0 && b.gpu_draw)) continue;
          const int64_t pos = b.draw_pos0 + (i < T.NL ? int64_t(b.leaf_offsets[i + 1] - b.leaf_offsets[i]) : 0);
          r.seed_lo = uint32_t(b.seed), r.seed_hi = uint32_t(b.seed >> 32);
          r.pos_lo = int32_t(uint32_t(uint64_t(pos))), r.pos_hi = int32_t(uint32_t(uint64_t(pos) >> 32));
          r.top_p = b.top_p, r.min_p = b.min_p, r.top_k = b.topk, r.draw = 1;
        }
      }
      E.pipe("topk_part");
      E.set(0, s.logits);
      E.set(1, meta.topk_it.buf);
      E.set(2, s.tk_x);
      E.set(3, s.tk_i);
      E.set(4, s.tk_stats);
      E.bytes(5, tp);
      E.rd(s.logits);
      E.wr(s.tk_x);
      E.wr(s.tk_i);
      E.wr(s.tk_stats);
      E.groups(MTLSizeMake(size_t(parts), size_t(N), 1), MTLSizeMake(256, 1, 1));
      E.pipe("topk_merge");
      E.set(0, s.tk_x);
      E.set(1, s.tk_i);
      E.set(2, s.tk_stats);
      E.set(3, s.tk_out);
      E.bytes(4, tp);
      E.set(5, meta.topk_samp.buf);
      E.set(6, m.chain_active ? m.chain_tok : s.tk_out);  // unused unless chaining
      E.rd(s.tk_x);
      E.rd(s.tk_i);
      E.rd(s.tk_stats);
      E.wr(s.tk_out);
      if (m.chain_active) E.wr(m.chain_tok);
      E.groups(MTLSizeMake(size_t(N), 1, 1), MTLSizeMake(256, 1, 1));
    }
  }

  // ---- Qwen4-Exp MTP state: the MTP layer's K/V over the root rows
  // (row i pairs the pre-mixer streams before token i with token i; row 0 the
  // parent's stored row) and the rows drafting starts from (mh, lh: streams,
  // W wide). Only its K/V are needed here: the layer's attention output and
  // MoE matter to drafts alone (run_draft_q4).
  if (any_mtp && c.q4) {
    const Mtp& MT = m.mtp;
    const Layer& L = MT.layer;
    const int W = c.W();
    const int R = mtp_rows;
    int32_t* mtok = meta.m_tok.ensure<int32_t>(m, size_t(R));
    int32_t* mpos = meta.m_pos.ensure<int32_t>(m, size_t(R) * 4);
    int32_t* hrow = meta.m_hin_rows.ensure<int32_t>(m, size_t(R));
    std::vector<int32_t> fix_rows;
    std::vector<uint64_t> fix_offs;
    for (TreePlan& T : trees) {
      if (!T.mtp_run) continue;
      const TreeSpec& b = *T.spec;
      const int i0 = T.P > 0 ? 0 : 1;
      for (int r = 0; r < T.mtp_R; ++r) {
        const int i = i0 + r;
        const int mr = T.mtp_r0 + r;
        mtok[mr] = b.root[i];
        for (int d = 0; d < 3; ++d) mpos[4 * mr + d] = i == 0 ? T.parent->mh_pos[d] : pos[4 * (T.base + i - 1) + d];
        mpos[4 * mr + 3] = 0;
        hrow[mr] = i == 0 ? T.base : T.base + i - 1;  // i == 0: replaced by the parent's row below
        if (i == 0) {
          fix_rows.push_back(mr);
          fix_offs.push_back(uint64_t(S.mh_off(T.parent_slot)));
        }
      }
      // Verify-commit: chain row j pairs the streams of the row before it (the
      // root's last row, then chain rows) with its token, at that row's position.
      for (int j = 0; j < T.mtp_C; ++j) {
        const int mr = T.mtp_r0 + T.mtp_R + j;
        const int hr = T.base + T.Tr - 1 + j;
        mtok[mr] = tok[T.base + T.Tr + j];
        for (int d = 0; d < 3; ++d) mpos[4 * mr + d] = pos[4 * hr + d];
        mpos[4 * mr + 3] = 0;
        hrow[mr] = hr;
      }
    }
    meta.m_hin.ensure<float>(m, size_t(R) * size_t(W));
    meta.m_x.ensure<float>(m, size_t(R) * size_t(W));
    meta.m_emb.ensure<float>(m, size_t(R) * size_t(c.D));
    meta.m_hn.ensure<uint16_t>(m, size_t(R) * size_t(W));
    float* ones = meta.m_logits.ensure<float>(m, size_t(R) * size_t(c.hc));
    for (size_t i = 0; i < size_t(R) * size_t(c.hc); ++i) ones[i] = 1.0f;
    E.pipe("gather_rows_f32");
    E.set(0, m.q4s.H);
    E.set(1, meta.m_hin.buf);
    E.set(2, meta.m_hin_rows.buf);
    E.bytes(3, W);
    E.rd(m.q4s.H);
    E.wr(meta.m_hin.buf, meta.m_hin.bytes);
    E.threads(MTLSizeMake(size_t(W / 4), size_t(R), 1), MTLSizeMake(64, 1, 1));
    if (!fix_rows.empty()) {
      upload(meta.m_offs, fix_offs);
      upload(meta.m_rows, fix_rows);
      E.pipe("pool_to_rows");  // writes rows 0..n of dst: stage in m_x, then scatter
      E.set(0, S.buf);
      E.set(1, meta.m_x.buf);
      E.set(2, meta.m_offs.buf);
      E.bytes(3, W);
      E.rd(S.buf, S.mtp_base * 4, SIZE_MAX);
      E.wr(meta.m_x.buf, meta.m_x.bytes);
      E.threads(MTLSizeMake(size_t(W / 4), fix_rows.size(), 1), MTLSizeMake(64, 1, 1));
      E.pipe("scatter_rows_f32");
      E.set(0, meta.m_x.buf);
      E.set(1, meta.m_hin.buf);
      E.set(2, meta.m_rows.buf);
      E.bytes(3, W);
      E.rd(meta.m_x.buf, meta.m_x.bytes);
      E.wr(meta.m_hin.buf, meta.m_hin.bytes);
      E.threads(MTLSizeMake(size_t(W / 4), fix_rows.size(), 1), MTLSizeMake(64, 1, 1));
    }
    {
      static const char* names[3] = {"embed_q4", "embed_q8", "embed_f16"};
      const Embed& T = m.embed;
      E.pipe(names[T.kind]);
      E.set(0, meta.m_tok.buf);
      E.set(1, T.codes);
      if (T.kind != 2) {
        E.set(2, T.s);
        E.set(3, T.z);
      }
      E.set(4, meta.m_emb.buf);
      E.bytes(5, c.D);
      if (T.kind != 2) E.bytes(6, T.group);
      E.wr(meta.m_emb.buf, meta.m_emb.bytes);
      E.threads(MTLSizeMake(size_t(c.D / 8), size_t(R), 1), MTLSizeMake(size_t(std::min(c.D / 8, 256)), 1, 1));
    }
    q4_mtp_streams(m, E, meta.m_emb.buf, meta.m_hin.buf, meta.m_x.buf, Buf(s.xn), meta.m_hn.buf, Buf(m.q4s.y),
                   meta.m_logits.buf, R);
    q4_hc_read(m, E, L.hca, true, meta.m_x.buf, R, Buf(s.xn));
    gemm_runs(E, m.tune, L.in_proj, s.xn, 0, c.D, s.proj, 0, ldf, R, 0);
    attn_prep(L, meta.m_pos.buf, 0, R);
    // The MTP layer's sparse attention index of these rows (read by the draft loop).
    QsaSet qm;
    const bool qsa_mtp = m.kv.i_layers > c.n_full;
    if (qsa_mtp) {
      std::vector<AttnTreeRec> mt;
      for (TreePlan& T : trees)
        if (T.mtp_run) mt.push_back({T.mtp_r0, std::max(T.P - 1, 0), T.read_table, T.mtp_R + T.mtp_C});
      const Buf mtb = arena_upload(m, meta.arena, mt);
      const Buf zd = arena_upload(m, meta.arena, std::vector<int32_t>(size_t(R) * 4, 0));
      qsa_plan(m, meta.arena, qm, mt, nullptr, false);
      qsa_index(m, E, qm, L, s.xn, meta.m_pos.buf, mtb, zd, meta.blocks.buf, c.n_full, false);
    }
    kv_save(meta.mtp_kv_saves, mtp_saves.size(), c.n_full);
    if (qsa_mtp) qsa_save(m, E, qm, meta.m_pos.buf, meta.mtp_kv_saves.buf, mtp_saves.size(), c.n_full, false);
    // Rows drafting starts from: mh (last saved root row) and lh (first leaf rows).
    std::vector<int32_t> rows;
    std::vector<uint64_t> offs;
    for (TreePlan& T : trees) {
      if (!T.mtp_run) continue;
      for (auto [sl, is_work] : {std::pair<int, bool>{T.dst_slot, false}, std::pair<int, bool>{T.work_slot, true}}) {
        if (sl < 0) continue;
        rows.push_back(T.base + (is_work ? T.Tr : T.save_len) - 1);
        offs.push_back(S.mh_off(sl));
        const bool whole = is_work || T.save_len == T.Tr;  // leaves continue the whole root
        if (whole)
          for (int j = 0; j < T.n_lh; ++j) {
            rows.push_back(T.base + T.Tr + j);
            offs.push_back(S.lh_off(sl, j));
          }
      }
    }
    if (!rows.empty()) {
      upload(meta.m_rows2, rows);
      upload(meta.m_offs2, offs);
      E.pipe("rows_to_pool");
      E.set(0, m.q4s.H);
      E.set(1, S.buf);
      E.set(2, meta.m_rows2.buf);
      E.set(3, meta.m_offs2.buf);
      E.bytes(4, W);
      E.rd(m.q4s.H);
      E.wr(S.buf, S.mtp_base * 4, SIZE_MAX);
      E.threads(MTLSizeMake(size_t(W / 4), rows.size(), 1), MTLSizeMake(64, 1, 1));
    }
  } else if (any_mtp) {
  // ---- MTP draft state: extend each saved slot's MTP K/V over the root (row
  // i pairs the hidden row before token i with token i; row 0 uses the
  // parent's stored hidden row) and keep the hidden rows drafting starts from.
  // Only the layer's K/V are needed here (as the Qwen4-Exp path): its
  // attention output, projection and MLP/MoE matter to drafts alone, and
  // run_draft computes them for the rows it drafts from.
    const Mtp& MT = m.mtp;
    const Layer& L = MT.layer;
    int32_t* mtok = meta.m_tok.ensure<int32_t>(m, size_t(mtp_rows));
    int32_t* mpos = meta.m_pos.ensure<int32_t>(m, size_t(mtp_rows) * 4);
    uint64_t* hoff = meta.m_hin_off.ensure<uint64_t>(m, size_t(mtp_rows));
    int32_t* hrow = meta.m_hin_rows.ensure<int32_t>(m, size_t(mtp_rows));
    float* hin = meta.m_hin.ensure<float>(m, size_t(mtp_rows) * size_t(c.D));
    (void)hin;
    meta.m_x.ensure<float>(m, size_t(mtp_rows) * size_t(c.D));
    meta.m_emb.ensure<float>(m, size_t(mtp_rows) * size_t(c.D));
    std::vector<int32_t> from_parent;  // MTP rows whose hidden row comes from the parent slot
    int n_from_parent = 0;
    (void)n_from_parent;
    for (TreePlan& T : trees) {
      if (!T.mtp_run) continue;
      const TreeSpec& b = *T.spec;
      const int i0 = T.P > 0 ? 0 : 1;
      for (int r = 0; r < T.mtp_R; ++r) {
        const int i = i0 + r;
        const int mr = T.mtp_r0 + r;
        mtok[mr] = b.root[i];
        for (int d = 0; d < 3; ++d) mpos[4 * mr + d] = i == 0 ? T.parent->mh_pos[d] : pos[4 * (T.base + i - 1) + d];
        mpos[4 * mr + 3] = 0;
        // Hidden input: the parent's stored row for i == 0, else row i - 1 of this pass.
        hoff[mr] = i == 0 ? uint64_t(S.mh_off(T.parent_slot)) : UINT64_MAX;
        hrow[mr] = T.base + i - 1;
      }
      // Chain row j pairs the row before it (the root's last row, then
      // chain rows) with its token, at that row's position.
      for (int j = 0; j < T.mtp_C; ++j) {
        const int mr = T.mtp_r0 + T.mtp_R + j;
        const int hr = T.base + T.Tr - 1 + j;
        mtok[mr] = tok[T.base + T.Tr + j];
        for (int d = 0; d < 3; ++d) mpos[4 * mr + d] = pos[4 * hr + d];
        mpos[4 * mr + 3] = 0;
        hoff[mr] = UINT64_MAX;
        hrow[mr] = hr;
      }
    }
    // Hidden inputs: pass rows (gather), then parent rows (from the pool).
    {
      E.pipe("gather_rows_f32");
      E.set(0, s.h);
      E.set(1, meta.m_hin.buf);
      E.set(2, meta.m_hin_rows.buf);
      E.bytes(3, c.D);
      E.rd(s.h);
      E.wr(meta.m_hin.buf, meta.m_hin.bytes);
      // Rows with i == 0 read row base - 1 (garbage or another tree): overwritten below.
      std::vector<int32_t> fix;
      for (int r = 0; r < mtp_rows; ++r)
        if (hoff[r] != UINT64_MAX) fix.push_back(r);
      for (int r = 0; r < mtp_rows; ++r)
        if (hrow[r] < 0) hrow[r] = 0;
      E.threads(MTLSizeMake(size_t(c.D / 4), size_t(mtp_rows), 1), MTLSizeMake(64, 1, 1));
      if (!fix.empty()) {
        uint64_t* offs = meta.m_offs.ensure<uint64_t>(m, fix.size());
        int32_t* rows = meta.m_rows.ensure<int32_t>(m, fix.size());
        for (size_t i = 0; i < fix.size(); ++i) {
          offs[i] = hoff[fix[i]];
          rows[i] = fix[i];
        }
        // pool_to_rows writes rows 0..n of dst: stage into m_x, then scatter.
        E.pipe("pool_to_rows");
        E.set(0, S.buf);
        E.set(1, meta.m_x.buf);
        E.set(2, meta.m_offs.buf);
        E.bytes(3, c.D);
        E.rd(S.buf, S.mtp_base * 4, SIZE_MAX);
        E.wr(meta.m_x.buf, meta.m_x.bytes);
        E.threads(MTLSizeMake(size_t(c.D / 4), fix.size(), 1), MTLSizeMake(64, 1, 1));
        E.pipe("scatter_rows_f32");
        E.set(0, meta.m_x.buf);
        E.set(1, meta.m_hin.buf);
        E.set(2, meta.m_rows.buf);
        E.bytes(3, c.D);
        E.rd(meta.m_x.buf, meta.m_x.bytes);
        E.wr(meta.m_hin.buf, meta.m_hin.bytes);
        E.threads(MTLSizeMake(size_t(c.D / 4), fix.size(), 1), MTLSizeMake(64, 1, 1));
      }
    }
    // The MTP layer over every MTP row.
    {
      const int R = mtp_rows;
      static const char* names[3] = {"embed_q4", "embed_q8", "embed_f16"};
      const Embed& T = m.embed;
      E.pipe(names[T.kind]);
      E.set(0, meta.m_tok.buf);
      E.set(1, T.codes);
      if (T.kind != 2) {
        E.set(2, T.s);
        E.set(3, T.z);
      }
      E.set(4, meta.m_emb.buf);
      E.bytes(5, c.D);
      if (T.kind != 2) E.bytes(6, T.group);
      E.wr(meta.m_emb.buf, meta.m_emb.bytes);
      E.threads(MTLSizeMake(size_t(c.D / 8), size_t(R), 1), MTLSizeMake(size_t(std::min(c.D / 8, 256)), 1, 1));
      E.pipe("mtp_prep");
      E.set(0, meta.m_emb.buf);
      E.set(1, meta.m_hin.buf);
      E.set(2, MT.enorm);
      E.set(3, MT.hnorm);
      E.set(4, MT.post_norm ? m.final_norm : MT.hnorm);
      E.set(5, s.act);
      E.bytes(6, c.D);
      E.bytes(7, c.eps);
      const int has_pre = MT.post_norm ? 1 : 0;
      E.bytes(8, has_pre);
      E.rd(meta.m_emb.buf, meta.m_emb.bytes);
      E.rd(meta.m_hin.buf, meta.m_hin.bytes);
      E.wr(s.act);
      E.groups(MTLSizeMake(size_t(R), 1, 1), MTLSizeMake(256, 1, 1));
      gemm(E, m.tune, MT.fc, s.act, 0, 2 * c.D, meta.m_x.buf.b, meta.m_x.buf.off, c.D, R, 0);
      rmsnorm(meta.m_x.buf, L.in_norm, s.xn, R, nil);
      gemm_runs(E, m.tune, L.in_proj, s.xn, 0, c.D, s.proj, 0, ldf, R, 0);
      attn_prep(L, meta.m_pos.buf, 0, R);
      kv_save(meta.mtp_kv_saves, mtp_saves.size(), c.n_full);
    }
    // Hidden rows drafting starts from: mh (last saved row) and lh (first leaf rows).
    {
      std::vector<int32_t> rows;
      std::vector<uint64_t> offs;
      for (TreePlan& T : trees) {
        if (!T.mtp_run) continue;
        for (auto [sl, is_work] : {std::pair<int, bool>{T.dst_slot, false}, std::pair<int, bool>{T.work_slot, true}}) {
          if (sl < 0) continue;
          rows.push_back(T.base + (is_work ? T.Tr : T.save_len) - 1);
          offs.push_back(S.mh_off(sl));
          const bool whole = is_work || T.save_len == T.Tr;  // leaves continue the whole root
          if (whole)
            for (int j = 0; j < T.n_lh; ++j) {
              rows.push_back(T.base + T.Tr + j);
              offs.push_back(S.lh_off(sl, j));
            }
        }
      }
      if (!rows.empty()) {
        upload(meta.m_rows2, rows);
        upload(meta.m_offs2, offs);
        E.pipe("rows_to_pool");
        E.set(0, s.h);
        E.set(1, S.buf);
        E.set(2, meta.m_rows2.buf);
        E.set(3, meta.m_offs2.buf);
        E.bytes(4, c.D);
        E.rd(s.h);
        E.wr(S.buf, S.mtp_base * 4, SIZE_MAX);
        E.threads(MTLSizeMake(size_t(c.D / 4), rows.size(), 1), MTLSizeMake(64, 1, 1));
      }
    }
  }

  if (q4_flushlog && c.q4 && q4_fl.n)
    std::fprintf(stderr, "q4 flushes %d: queue %.2f ms, exec %.2f ms, wake %.2f ms (per flush)\n", q4_fl.n,
                 q4_fl.queue / q4_fl.n, q4_fl.exec / q4_fl.n, q4_fl.wake / q4_fl.n);
  cb = E.finish();
  const double enc_ms = ms_since(t_enc);
  if (m.chain_active) {
    // Chain pass: commit without waiting; the next pass is planned from the
    // slot state committed below and ordered after this one by chain_ev.
    [cb commit];
    m.chain_cbs.push_back(cb);
  } else if (!q4_pend.empty()) {
    // Event pipeline: serve every layer's routing handshake in order,
    // then wait for the pass.
    [cb commit];
    ExpertCacheStats& st = m.xstats;
    try {
      for (const Q4Pending& p : q4_pend) {
        const auto tw = Clock::now();
        // Block, never spin: any busy CPU core while these short GPU segments
        // run halves their speed on an M1 Max (a background spinning
        // thread took decode GPU time from 30 to 62 ms per token).
        while (![m.xevA waitUntilSignaledValue:p.v timeoutMS:1000]) {
          if ([cb status] >= MTLCommandBufferStatusCompleted) fail("q4 pass ended before its routing handshake");
        }
        st.wait_ms += ms_since(tw);
        ++st.handshakes;
        const auto t0 = Clock::now();
        const ExpertPool& P = m.xpool[size_t(p.li)];
        const int32_t* ids = static_cast<const int32_t*>([m.mz.ids contents]);
        const int n = p.rows * c.k;
        std::vector<int> need;
        std::vector<char> seen(size_t(c.E), 0);
        for (int i = 0; i < n; ++i) {
          if (!seen[size_t(ids[i])]) seen[size_t(ids[i])] = 1, need.push_back(ids[i]);
          st.hits += P.slot_of[size_t(ids[i])] >= 0 ? 1 : 0;
        }
        st.picks += uint64_t(n);
        ++m.xclock;
        if (!m.xstage) m.xstage = std::make_unique<Arena>(m.dev, size_t(64) << 20);
        m.xstage->reset();  // the previous layer's loads completed before its MoE ran
        id<MTLCommandBuffer> lcb = [m.xqueue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> le = [lcb computeCommandEncoder];
        const bool loaded = q4_load_routed(p.li, need, le);
        [le endEncoding];
        if (m.xpf) xio_wait(*m.xpf);  // the copies read staging filled asynchronously
        if (loaded) {
          [lcb encodeSignalEvent:m.xevB value:p.v];
          [lcb commit];
        } else {
          m.xevB.signaledValue = p.v;
        }
        st.load_ms += ms_since(t0);
      }
    } catch (...) {
      m.xevB.signaledValue = m.xev_v;  // release the GPU (the pass fails below)
      [cb waitUntilCompleted];
      throw;
    }
    [cb waitUntilCompleted];
    if ([cb status] != MTLCommandBufferStatusCompleted) {
      NSError* err = [cb error];
      fail(std::string("GPU command buffer failed: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
    }
  } else {
    commit_wait(cb, 20);  // on failure the guard leaves the slots as they were
    E.check_submitted();  // split handshakes' resident tiles
  }

  // Long-prompt staging (whole layers of experts, ~3 GB) is not kept for the
  // decode steps that follow: they stage a few MB (64 MB chunks).
  if (q4_swept && !m.chain_active) {
    m.xstage = std::make_unique<Arena>(m.dev, size_t(64) << 20);
    if (m.xpipe.buf[0] && m.xpf) {  // the pipe buffers too, once their reads are done
      for (int k = 0; k < 2; ++k)
        while (m.xpf->st_pending[k].load(std::memory_order_acquire) > 0) {
        }
      m.xpipe = Model::XPipe{};
    }
  }

  // ---- Qwen4-Exp: n-gram mix state (last 9 normalized conv inputs, last two
  // token ids) of every slot this pass saves, from its parent state.
  if (c.q4) {
    const int W = c.W(), HR = Model::kPleHist;
    float* hist = (float*)[m.ple_hist contents];
    const float* Gn = (const float*)[m.q4s.Gn contents];
    const int32_t* toks = static_cast<const int32_t*>(ptr(meta.tokens.buf));
    auto save = [&](int row0, int len, int in, int out) {
      std::vector<float> nh(size_t(HR) * size_t(W));
      for (int k = 0; k < HR; ++k) {
        const int j = len - HR + k;  // relative to row0; < 0: an entry of the parent's history
        const float* src = j >= 0 ? Gn + size_t(row0 + j) * size_t(W) : hist + (size_t(in) * size_t(HR) + size_t(HR + j)) * size_t(W);
        std::memcpy(nh.data() + size_t(k) * size_t(W), src, size_t(W) * 4);
      }
      std::memcpy(hist + size_t(out) * size_t(HR) * size_t(W), nh.data(), nh.size() * 4);
      std::array<int32_t, 2> cx = m.ple_ctx[size_t(in)];
      for (int j = 0; j < len; ++j) cx = {cx[1], toks[row0 + j]};
      m.ple_ctx[size_t(out)] = cx;
    };
    for (TreePlan& T : trees) {
      if (T.dst) save(T.base, T.save_len, T.pstate, T.dst_slot);
      if (T.work) save(T.base, T.Tr, T.pstate, T.work_slot);
    }
  }

  // ---- commit slot state
  guard.armed = false;  // the slots take the new blocks; the admissions stand
  m.xpass.clear();
  for (TreePlan& T : trees) {
    auto commit = [&](Slot* to, std::vector<int>& blocks, int len, int next) {
      m.release_blocks(*to);
      to->blocks = std::move(blocks);
      to->len = len;
      to->next_pos = next;
    };
    if (T.dst) commit(T.dst, T.dst_blocks, T.P + T.save_len, T.save_len == T.Tr ? T.next_root : T.next_save);
    if (T.work) commit(T.work, T.work_blocks, T.P + T.Tr, T.next_root);
    // The commit record (the chain's rows stay valid until the next pass).
    if (T.dst) {
      T.dst->cm_k = T.cm_k;
      T.dst->cm_row0 = T.cm_row0;
      T.dst->cm_prow0 = T.base + T.Tr;
      T.dst->cm_pos0 = T.next_root;
      T.dst->cm_pass = pass_id;
      T.dst->cm_scratch = m.scratch_seq;
      T.dst->cm_fused = gdn_fused_pass;
      T.dst->cm_R = m.dn_rows > 0 ? m.dn_rows : 16;
    }
    if (T.work) T.work->cm_k = 0;
    for (Slot* to : {T.dst, T.work}) {
      if (!to) continue;
      to->mtp_ok = T.mtp_run;
      to->n_lh = 0;
      if (!T.mtp_run) continue;
      const int row = T.base + (to == T.work ? T.Tr : T.save_len) - 1;
      for (int d = 0; d < 3; ++d) to->mh_pos[d] = pos[4 * row + d];
      if (to == T.work || T.save_len == T.Tr) to->n_lh = T.n_lh;
    }
  }
  if (m.chain_active) {  // outputs are read by run_chain once the chain completes
    if (stats) {
      stats->encode_ms = enc_ms;
      stats->rows = M;
      stats->readouts = N;
      stats->dispatches = m.dispatches;
      stats->barriers = m.barriers;
    }
    return;
  }
  // ---- outputs
  if (any_hidden)
    for (const TreePlan& T : trees) {
      if (!T.spec->out_hidden) continue;
      const __fp16* src = static_cast<const __fp16*>([s.hid contents]) + size_t(T.base) * size_t(c.D);
      float* dst = T.spec->out_hidden;
      const size_t n = size_t(T.Tr) * size_t(c.D);
      for (size_t i = 0; i < n; ++i) dst[i] = float(src[i]);
    }
  for (const TreePlan& T : trees) {
    if (!T.N) continue;
    const TreeSpec& b = *T.spec;
    if (T.n_cands && b.out_cand_logits)
      std::memcpy(b.out_cand_logits, static_cast<float*>(ptr(meta.cand_out.buf)) + T.cand_base,
                  sizeof(float) * size_t(T.n_cands));
    if (b.out_lse)
      std::memcpy(b.out_lse, static_cast<float*>([s.lse contents]) + T.readout_base, sizeof(float) * size_t(T.N));
    if (b.out_argmax)
      std::memcpy(b.out_argmax, static_cast<int32_t*>([s.argmax contents]) + T.readout_base,
                  sizeof(int32_t) * size_t(T.N));
  }
  if (stats) {
    stats->gpu_ms = gpu_ms(cb);
    stats->encode_ms = enc_ms;
    stats->wall_ms = ms_since(t_wall);
    stats->rows = M;
    stats->readouts = N;
    stats->dispatches = m.dispatches;
    stats->barriers = m.barriers;
  }
}

}  // namespace espetero::native

namespace espetero::native {

// Blocks of `s` from position `from` on become exclusive (copy on write of
// shared ones, recorded in `cow`), and the list grows to cover `until`
// positions. Used by drafting, which writes MTP entries past the slot's length.
// ---- verify-commit
// The first n rows of the chain each slot's last pass kept (TreeSpec::
// commit_chain) become part of the slot: their K/V and MTP K/V are already at
// their positions; the DeltaNet recurrence and conv window advance over their
// saved in_proj rows with the kernel the pass ran them with (the leaf started
// from the state the root left in the slot, so the state is bit for bit the
// one the leaf reached), and the slot's length, next position and MTP row
// follow. Requests must name the pass that just ran.
inline void run_commit(Model& m, PassMeta& meta, const std::vector<CommitRequest>& reqs, PassStats* stats) {
  const auto t_wall = Clock::now();
  const Config& c = m.cfg;
  struct Job {
    Slot* sl;
    int slot, n;
  };
  std::vector<Job> jobs;
  for (const CommitRequest& q : reqs) {
    if (q.slot < 0 || q.slot >= m.n_user || !m.slots[size_t(q.slot)].used) fail("commit: invalid slot " + std::to_string(q.slot));
    Slot& sl = m.slots[size_t(q.slot)];
    if (q.n < 0 || q.n > sl.cm_k || (q.n > 0 && (sl.cm_pass != m.pass_seq || sl.cm_scratch != m.scratch_seq)))
      fail("commit: slot " + std::to_string(q.slot) + " holds no such chain rows of the last pass");
    if (q.n > 0) jobs.push_back({&sl, q.slot, q.n});
  }
  if (jobs.empty()) {
    for (const CommitRequest& q : reqs) m.slots[size_t(q.slot)].cm_k = 0;
    return;
  }
  const bool fused = jobs[0].sl->cm_fused;
  const int R = jobs[0].sl->cm_R;
  for (const Job& j : jobs)
    if (j.sl->cm_fused != fused || j.sl->cm_R != R) fail("commit: chains of different passes");
  // Rows of Model::cm_x: every chain row of the requested slots gets valid taps
  // (conv_prep runs over all rows up to the last one); segments cover n rows.
  int rows = 0;
  for (const Job& j : jobs) rows = std::max(rows, j.sl->cm_row0 + j.sl->cm_k);
  int32_t* csrc = meta.conv_src.ensure<int32_t>(m, size_t(rows) * 3);
  int32_t* rslot = meta.row_slot.ensure<int32_t>(m, size_t(rows));
  for (int r = 0; r < rows; ++r) {
    rslot[r] = jobs[0].slot;
    for (int d = 0; d < 3; ++d) csrc[r * 3 + d] = -1 - 2;  // the last window entry (rows outside requests)
  }
  std::vector<DeltaSeg> segs;
  std::vector<ConvSave> convs;
  for (const Job& j : jobs) {
    const int r0 = j.sl->cm_row0;
    for (int x = 0; x < j.sl->cm_k; ++x) {
      rslot[r0 + x] = j.slot;
      for (int d = 1; d <= 3; ++d) {
        const int k = x - d;
        csrc[(r0 + x) * 3 + d - 1] = k >= 0 ? r0 + k : -1 - (3 + k);
      }
    }
    segs.push_back({r0, j.n, j.slot, j.slot, 0, -1, 1, 0});
    convs.push_back({r0, j.n, j.slot, j.slot});
  }
  auto upload = [&](HostBuf& hb, const auto& v) {
    using T = typename std::decay_t<decltype(v)>::value_type;
    T* p = hb.ensure<T>(m, std::max<size_t>(v.size(), 1));
    if (!v.empty()) std::memcpy(p, v.data(), v.size() * sizeof(T));
  };
  upload(meta.segs_root, segs);
  upload(meta.conv_jobs, convs);
  ensure_scratch(m, std::max(rows, 1), 1);
  Scratch& s = m.sc;
  const StatePool& S = m.st;
  const int C = c.C(), Vd = c.Vd(), ldl = c.lin_ld();
  m.dispatches = 0;
  m.barriers = 0;
  id<MTLCommandBuffer> cb = [m.queue commandBufferWithUnretainedReferences];
  Enc E(m, cb);
  for (int li = 0; li < c.L; ++li) {
    const Layer& L = m.layers[size_t(li)];
    if (!L.linear) continue;
    const int k = L.index_in_kind;
    const size_t loff = size_t(k) * size_t(m.cm_cap) * size_t(ldl) * sizeof(float);
    const uint64_t conv_base = S.conv_base[size_t(k)], st_base = S.st_base[size_t(k)];
    const ConvParams cp{C, ldl, C + Vd, c.Hk, c.Hv, c.qk_eps, conv_base, S.conv_floats};
    const size_t conv_lo = conv_base * 4, conv_hi = (conv_base + S.conv_floats * size_t(S.n_slots)) * 4;
    const size_t st_lo = st_base * 4, st_hi = (st_base + S.st_floats * size_t(S.n_slots)) * 4;
    if (fused) {
      const DeltaFusedParams fp{c.Hk, c.Hv, C, ldl, C + Vd, c.tiled_heads ? 1 : 0, c.eps, C, Vd,
                                conv_base, S.conv_floats, st_base, S.st_floats, 0};
      const std::string name = "gdn_decode_fused_tr" + std::to_string(R);
      const size_t nt = 32 * size_t(c.Dv / R);
      E.pipe(name);
      E.set(0, m.cm_x, loff);
      E.set(1, S.buf);
      E.set(2, meta.conv_src.buf);
      E.set(3, L.conv_w);
      E.set(4, meta.row_slot.buf);
      E.set(5, L.a_log);
      E.set(6, L.dt_bias);
      E.set(7, L.gnorm);
      E.set(8, s.yn);
      E.set(9, meta.segs_root.buf);
      E.bytes(10, fp);
      E.rd(m.cm_x);
      E.rd(S.buf, conv_lo, conv_hi);
      E.rd(S.buf, st_lo, st_hi);
      E.wr(S.buf, st_lo, st_hi);
      E.wr(s.yn);
      E.threads(MTLSizeMake(nt, 1, segs.size() * size_t(c.Hv)), MTLSizeMake(nt, 1, 1));
    } else {
      E.pipe("conv_prep");
      E.set(0, m.cm_x, loff);
      E.set(1, S.buf);
      E.set(2, meta.conv_src.buf);
      E.set(3, L.conv_w);
      E.set(4, s.qo);
      E.set(5, s.gb);
      E.set(6, L.a_log);
      E.set(7, L.dt_bias);
      E.bytes(8, cp);
      E.set(9, meta.row_slot.buf);
      E.rd(m.cm_x);
      E.rd(S.buf, conv_lo, conv_hi);
      E.wr(s.qo);
      E.wr(s.gb);
      E.groups(MTLSizeMake(size_t(2 * c.Hk + c.Hv + 3) / 4, size_t(rows), 1), MTLSizeMake(128, 1, 1));
      const DeltaParams dp{c.Hk, c.Hv, C, c.tiled_heads ? 1 : 0, st_base, S.st_floats};
      E.pipe("deltanet_tr" + std::to_string(R));
      E.set(0, s.qo);
      E.set(1, s.gb);
      E.set(2, S.buf);
      E.set(4, s.y);
      E.set(5, meta.segs_root.buf);
      E.bytes(6, dp);
      E.rd(s.qo);
      E.rd(s.gb);
      E.rd(S.buf, st_lo, st_hi);
      E.wr(S.buf, st_lo, st_hi);
      E.wr(s.y);
      E.threads(MTLSizeMake(32, size_t(c.Dv / R), segs.size() * size_t(c.Hv)), MTLSizeMake(32, 4, 1));
    }
    // The conv window after the committed rows (after the recurrence read the old one).
    E.pipe("conv_state_save");
    E.set(0, m.cm_x, loff);
    E.set(1, S.buf);
    E.set(2, meta.conv_jobs.buf);
    E.bytes(3, cp);
    E.rd(m.cm_x);
    E.rd(S.buf, conv_lo, conv_hi);
    E.wr(S.buf, conv_lo, conv_hi);
    E.threads(MTLSizeMake(size_t(C), convs.size(), 1), MTLSizeMake(256, 1, 1));
  }
  // MTP: the slot's row drafting starts from becomes the last committed row's
  // (kept by the pass as leaf row n - 1).
  std::vector<uint64_t> from, to;
  for (const Job& j : jobs)
    if (j.sl->mtp_ok && j.n <= j.sl->n_lh) {
      from.push_back(uint64_t(S.lh_off(j.slot, j.n - 1)));
      to.push_back(uint64_t(S.mh_off(j.slot)));
    }
  if (!from.empty()) {
    const int W = c.W();
    std::vector<int32_t> idx(from.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = int32_t(i);
    upload(meta.m_offs, from);
    upload(meta.m_offs2, to);
    upload(meta.m_rows2, idx);
    meta.m_x.ensure<float>(m, from.size() * size_t(W));
    E.pipe("pool_to_rows");
    E.set(0, S.buf);
    E.set(1, meta.m_x.buf);
    E.set(2, meta.m_offs.buf);
    E.bytes(3, W);
    E.rd(S.buf, S.mtp_base * 4, SIZE_MAX);
    E.wr(meta.m_x.buf, meta.m_x.bytes);
    E.threads(MTLSizeMake(size_t(W / 4), from.size(), 1), MTLSizeMake(64, 1, 1));
    E.pipe("rows_to_pool");
    E.set(0, meta.m_x.buf);
    E.set(1, S.buf);
    E.set(2, meta.m_rows2.buf);
    E.set(3, meta.m_offs2.buf);
    E.bytes(4, W);
    E.rd(meta.m_x.buf, meta.m_x.bytes);
    E.wr(S.buf, S.mtp_base * 4, SIZE_MAX);
    E.threads(MTLSizeMake(size_t(W / 4), from.size(), 1), MTLSizeMake(64, 1, 1));
  }
  cb = E.finish();
  commit_wait(cb, 5);
  // Qwen4-Exp: the n-gram mix state (last kPleHist normalized conv
  // inputs, last two token ids) after the committed rows, from the pass's
  // rows (q4s.Gn and the pass tokens still hold them: no pass ran since).
  if (c.q4 && m.ple_hist) {
    const int W = c.W(), HR = Model::kPleHist;
    float* hist = (float*)[m.ple_hist contents];
    const float* Gn = (const float*)[m.q4s.Gn contents];
    const int32_t* toks = static_cast<const int32_t*>(ptr(meta.tokens.buf));
    for (const Job& j : jobs) {
      const int row0 = j.sl->cm_prow0, len = j.n, sl = j.slot;
      std::vector<float> nh(size_t(HR) * size_t(W));
      for (int k = 0; k < HR; ++k) {
        const int x = len - HR + k;  // < 0: an entry of the slot's history
        const float* src = x >= 0 ? Gn + size_t(row0 + x) * size_t(W) : hist + (size_t(sl) * size_t(HR) + size_t(HR + x)) * size_t(W);
        std::memcpy(nh.data() + size_t(k) * size_t(W), src, size_t(W) * 4);
      }
      std::memcpy(hist + size_t(sl) * size_t(HR) * size_t(W), nh.data(), nh.size() * 4);
      std::array<int32_t, 2> cx = m.ple_ctx[size_t(sl)];
      for (int x = 0; x < len; ++x) cx = {cx[1], toks[row0 + x]};
      m.ple_ctx[size_t(sl)] = cx;
    }
  }
  for (const Job& j : jobs) {
    Slot& sl = *j.sl;
    const bool mtp = sl.mtp_ok && j.n <= sl.n_lh;
    if (sl.mtp_ok && !mtp) sl.mtp_ok = false;  // no stored row for the new last position
    sl.len += j.n;
    sl.next_pos = sl.cm_pos0 + j.n;
    for (int d = 0; d < 3; ++d) sl.mh_pos[d] = sl.cm_pos0 + j.n - 1;
    sl.n_lh = 0;
    sl.cm_k = 0;
  }
  for (const CommitRequest& q : reqs) m.slots[size_t(q.slot)].cm_k = 0;
  if (stats) {
    stats->gpu_ms = gpu_ms(cb);
    stats->wall_ms = ms_since(t_wall);
    stats->rows = rows;
    stats->dispatches = m.dispatches;
    stats->barriers = m.barriers;
  }
}

// What own_blocks_from changed in the drafted slots, given back unless
// the draft completes. A draft that failed (no KV block left, a GPU error)
// left a slot pointing at a fresh copy of its last shared block whose copy
// never ran: its valid positions there read garbage K/V from then on, while
// the engine went on decoding plainly ("drafting is an optimization").
struct OwnedBlocks {
  Model& m;
  struct Cow {
    int slot, index, old_b, new_b;
  };
  std::vector<Cow> cows;
  std::vector<std::pair<int, size_t>> grown;  // slot, its block count before
  bool armed = true;
  explicit OwnedBlocks(Model& mm) : m(mm) {}
  OwnedBlocks(const OwnedBlocks&) = delete;
  OwnedBlocks& operator=(const OwnedBlocks&) = delete;
  ~OwnedBlocks() {
    if (!armed) return;
    for (auto it = grown.rbegin(); it != grown.rend(); ++it) {
      Slot& s = m.slots[size_t(it->first)];
      while (s.blocks.size() > it->second) {
        m.kv_release(s.blocks.back());
        s.blocks.pop_back();
      }
    }
    for (auto it = cows.rbegin(); it != cows.rend(); ++it) {
      Slot& s = m.slots[size_t(it->slot)];
      m.kv_retain(it->old_b);
      m.kv_release(s.blocks[size_t(it->index)]);
      s.blocks[size_t(it->index)] = it->old_b;
    }
  }
};

// Blocks own_blocks_from(s, from, until) takes: a copy of each shared block
// from `from` on and the new ones up to `until` (an upper bound across slots
// that share a block).
inline int own_blocks_need(const Model& m, const Slot& s, int from, int until) {
  const int bt = m.kv.bt;
  int n = 0;
  for (int i = from / bt; i < int(s.blocks.size()); ++i) n += m.kv.ref[size_t(s.blocks[size_t(i)])] > 1 ? 1 : 0;
  return n + std::max(0, (until + bt - 1) / bt - int(s.blocks.size()));
}

inline void own_blocks_from(Model& m, Slot& s, int from, int until, std::vector<std::array<int32_t, 2>>& cow,
                            OwnedBlocks* log = nullptr) {
  const int bt = m.kv.bt;
  const int slot = int(&s - m.slots.data());
  for (int i = from / bt; i < int(s.blocks.size()); ++i) {
    const int b = s.blocks[size_t(i)];
    if (m.kv.ref[size_t(b)] > 1) {
      const int nb = m.kv_alloc();
      cow.push_back({b, nb});
      m.kv_release(b);
      s.blocks[size_t(i)] = nb;
      if (log) log->cows.push_back({slot, i, b, nb});
    }
  }
  if (log) log->grown.push_back({slot, s.blocks.size()});
  while (int(s.blocks.size()) * bt < until) s.blocks.push_back(m.kv_alloc());
}

// MTP drafts for several slots in one command buffer: step 0 runs the MTP
// layer over each request's rows (the slot's last decoder row and its first
// leaf rows, each fed the token that follows it); every further step feeds
// back the previous draft and the MTP output, one row per request. Greedy
// over LM head rows [0, mtp.vocab). The MTP entries written past a slot's
// length are scratch: the next pass recomputes them from decoder rows.
// Qwen4-Exp drafts (docs/qwen4_exp.md 7): the MTP layer on the streams
// U_s = fc_emb . norm(emb) + fc_hid . norm(H)_s, with its hyper-connection
// blocks, attention over its own K/V (layer c.n_full), its resident 8-bit
// MoE, then its own mixer and the LM head. Step 0 runs every pending token of
// a request (hidden rows: the slot's mh, then its leaf rows lh); step k >= 1
// feeds the layer's output streams of the last row back.
// The drafts of sampled requests at step k, drawn on the GPU as the
// target draws its tokens (topk_part + topk_merge with the request's seed and
// the draft's position), written over the step's argmax in arg[k * nq + i];
// greedy requests keep the argmax. Needs the logits of the whole vocabulary;
// under the coarse head only the shortlisted blocks hold exact logits
// and the rest their 2-bit estimates, which can only cost acceptance (the
// drafts are verified by equality with the target's own draw).
inline void draw_draft_step(Model& m, PassMeta& meta, Enc& E, const std::vector<DraftRequest>& reqs, int k, int vocab) {
  const Config& c = m.cfg;
  const int nq = int(reqs.size());
  int K = 0;
  for (const DraftRequest& q : reqs)
    if (q.draw && k < q.n_draft) K = std::max(K, std::min(32, q.top_k > 0 ? q.top_k : 32));
  if (K == 0 || vocab != c.V) return;
  Scratch& s = m.sc;
  float* it = meta.topk_it.ensure<float>(m, size_t(nq));
  TopKSampleRec* sr = meta.topk_samp.ensure<TopKSampleRec>(m, size_t(nq));
  for (int i = 0; i < nq; ++i) {
    const DraftRequest& q = reqs[size_t(i)];
    it[i] = q.draw && q.temperature > 0 ? 1.0f / q.temperature : 1.0f;
    sr[i] = TopKSampleRec{};
    if (!q.draw || k >= q.n_draft) continue;
    const uint64_t pos = uint64_t(q.pos0 + k);
    sr[i].seed_lo = uint32_t(q.seed), sr[i].seed_hi = uint32_t(q.seed >> 32);
    sr[i].pos_lo = int32_t(uint32_t(pos)), sr[i].pos_hi = int32_t(uint32_t(pos >> 32));
    sr[i].top_p = q.top_p, sr[i].min_p = q.min_p, sr[i].top_k = q.top_k > 0 ? q.top_k : 32, sr[i].draw = 1;
  }
  const int parts = (c.V + 4095) / 4096;
  const TopKParams tp{c.V, K, (c.V + parts - 1) / parts, parts, 1, k * nq};
  E.pipe("topk_part");
  E.set(0, s.logits);
  E.set(1, meta.topk_it.buf);
  E.set(2, s.tk_x);
  E.set(3, s.tk_i);
  E.set(4, s.tk_stats);
  E.bytes(5, tp);
  E.rd(s.logits);
  E.wr(s.tk_x);
  E.wr(s.tk_i);
  E.wr(s.tk_stats);
  E.groups(MTLSizeMake(size_t(parts), size_t(nq), 1), MTLSizeMake(256, 1, 1));
  E.pipe("topk_merge");
  E.set(0, s.tk_x);
  E.set(1, s.tk_i);
  E.set(2, s.tk_stats);
  E.set(3, s.tk_out);
  E.bytes(4, tp);
  E.set(5, meta.topk_samp.buf);
  E.set(6, meta.m_arg.buf);
  E.rd(s.tk_x);
  E.rd(s.tk_i);
  E.rd(s.tk_stats);
  E.wr(s.tk_out);
  E.wr(meta.m_arg.buf, meta.m_arg.bytes);
  E.groups(MTLSizeMake(size_t(nq), 1, 1), MTLSizeMake(256, 1, 1));
}

inline void run_draft_q4(Model& m, PassMeta& meta, const std::vector<DraftRequest>& reqs, PassStats* stats, int kmax,
                         int rows0, Clock::time_point t_wall, OwnedBlocks* owned) {
  const Config& c = m.cfg;
  const int nq = int(reqs.size());
  const StatePool& S = m.st;
  const KVPool& K = m.kv;
  const int bt = K.bt;
  const int G = c.H / c.Hkv;
  const int W = c.W();
  const int qdim = c.H * c.HD, kvd = c.kv_dim(), ldf = c.full_ld();
  const int Rmax = std::max(rows0, nq);
  ensure_scratch(m, Rmax, nq);
  ensure_q4_scratch(m, Rmax);
  Scratch& s = m.sc;
  std::vector<std::array<int32_t, 2>> cow;
  for (const DraftRequest& q : reqs) {
    Slot& sl = m.slots[size_t(q.slot)];
    own_blocks_from(m, sl, sl.len - 1, sl.len - 1 + q.n_tokens + kmax, cow, owned);
  }
  std::vector<int32_t> table;
  std::vector<int> table_off(static_cast<size_t>(nq));
  for (int i = 0; i < nq; ++i) {
    const Slot& sl = m.slots[size_t(reqs[size_t(i)].slot)];
    table_off[size_t(i)] = int(table.size());
    table.insert(table.end(), sl.blocks.begin(), sl.blocks.end());
  }
  auto upload = [&](HostBuf& hb, const auto& v) {
    using T = typename std::decay_t<decltype(v)>::value_type;
    T* p = hb.ensure<T>(m, std::max<size_t>(v.size(), 1));
    if (!v.empty()) std::memcpy(p, v.data(), v.size() * sizeof(T));
  };
  upload(meta.blocks, table);
  upload(meta.cow, cow);
  int32_t* tok = meta.m_tok.ensure<int32_t>(m, size_t(Rmax));
  int32_t* pos = meta.m_pos.ensure<int32_t>(m, size_t(Rmax) * 4 * size_t(kmax + 1));
  uint64_t* offs = meta.m_offs.ensure<uint64_t>(m, size_t(Rmax));
  meta.m_hin.ensure<float>(m, size_t(Rmax) * size_t(W));
  meta.m_x.ensure<float>(m, size_t(Rmax) * size_t(W));
  meta.m_emb.ensure<float>(m, size_t(Rmax) * size_t(c.D));
  meta.m_hn.ensure<uint16_t>(m, size_t(Rmax) * size_t(W));
  meta.m_hc.ensure<float>(m, size_t(nq) * size_t(W));
  float* ones = meta.m_logits.ensure<float>(m, size_t(Rmax) * size_t(c.hc));
  for (size_t i = 0; i < size_t(Rmax) * size_t(c.hc); ++i) ones[i] = 1.0f;
  meta.m_arg.ensure<int32_t>(m, size_t(nq) * size_t(kmax));
  bool want2 = false;
  for (const DraftRequest& q : reqs) want2 = want2 || q.out2;
  if (want2) meta.m_arg2.ensure<int32_t>(m, size_t(nq) * size_t(kmax));
  int32_t* zdesc = meta.zero_desc.ensure<int32_t>(m, size_t(Rmax) * 4);
  std::memset(zdesc, 0, size_t(Rmax) * 16);
  std::vector<int> r0(static_cast<size_t>(nq)), P(static_cast<size_t>(nq)), last_pos(static_cast<size_t>(nq));
  int r = 0;
  for (int i = 0; i < nq; ++i) {
    const DraftRequest& q = reqs[size_t(i)];
    const Slot& sl = m.slots[size_t(q.slot)];
    r0[size_t(i)] = r;
    P[size_t(i)] = sl.len - 1;
    for (int j = 0; j < q.n_tokens; ++j, ++r) {
      tok[r] = q.tokens[j];
      for (int d = 0; d < 3; ++d) pos[4 * r + d] = j == 0 ? sl.mh_pos[d] : sl.next_pos + j - 1;
      pos[4 * r + 3] = 0;
      offs[r] = j == 0 ? S.mh_off(q.slot) : S.lh_off(q.slot, q.lh_first + j - 1);
    }
    last_pos[size_t(i)] = pos[4 * (r - 1)];
  }
  int32_t* cpos = pos + 4 * size_t(Rmax);
  for (int k = 1; k < kmax; ++k)
    for (int i = 0; i < nq; ++i) {
      int32_t* p = cpos + 4 * (size_t(k - 1) * size_t(nq) + size_t(i));
      p[0] = p[1] = p[2] = last_pos[size_t(i)] + k;
      p[3] = 0;
    }
  const int vocab = m.mtp.vocab > 0 ? std::min(m.mtp.vocab, c.V) : c.V;

  const auto t_enc = Clock::now();
  m.dispatches = 0;
  m.barriers = 0;
  id<MTLCommandBuffer> cb = [m.queue commandBufferWithUnretainedReferences];
  Enc E(m, cb);
  if (!cow.empty()) {
    kv_cow_copy(m, E, meta.cow.buf, cow.size());
    E.barrier();
  }
  const Mtp& MT = m.mtp;
  const Layer& L = MT.layer;
  const int hsplit = G > 8 ? G / 6 : 1, Gk = G / hsplit;
  const AttnParams ap_kv{c.H, c.Hkv, qdim, kvd, ldf, bt, uint64_t(K.layer_base(c.n_full))};
  const AttnParams ap_at{c.H, c.Hkv * hsplit, qdim, kvd, ldf, bt, uint64_t(K.layer_base(c.n_full))};
  auto gather_rows = [&](const Buf& src, const Buf& dst, HostBuf& rows_b, const std::vector<int32_t>& rows, int width) {
    upload(rows_b, rows);
    E.pipe("gather_rows_f32");
    E.set(0, src);
    E.set(1, dst);
    E.set(2, rows_b.buf);
    E.bytes(3, width);
    E.rd(src.b);
    E.wr(dst.b);
    E.threads(MTLSizeMake(size_t(width / 4), rows.size(), 1), MTLSizeMake(64, 1, 1));
  };
  // The MTP layer over R rows (tokens at tok_b + tok_off, positions at pos_off,
  // hidden streams in m_hin): output streams in m_x.
  auto layer = [&](int R, const Buf& tok_b, size_t tok_off, size_t pos_off, AttnPlan& plan, const std::vector<KVSaveRec>& saves) {
    static const char* names[3] = {"embed_q4", "embed_q8", "embed_f16"};
    const Embed& T = m.embed;
    E.pipe(names[T.kind]);
    E.set(0, tok_b, tok_off);
    E.set(1, T.codes);
    if (T.kind != 2) {
      E.set(2, T.s);
      E.set(3, T.z);
    }
    E.set(4, meta.m_emb.buf);
    E.bytes(5, c.D);
    if (T.kind != 2) E.bytes(6, T.group);
    E.rd(tok_b.b);
    E.wr(meta.m_emb.buf, meta.m_emb.bytes);
    E.threads(MTLSizeMake(size_t(c.D / 8), size_t(R), 1), MTLSizeMake(size_t(std::min(c.D / 8, 256)), 1, 1));
    q4_mtp_streams(m, E, meta.m_emb.buf, meta.m_hin.buf, meta.m_x.buf, Buf(s.xn), meta.m_hn.buf, Buf(m.q4s.y),
                   meta.m_logits.buf, R);
    q4_hc_read(m, E, L.hca, true, meta.m_x.buf, R, Buf(s.xn));
    gemm_runs(E, m.tune, L.in_proj, s.xn, 0, c.D, s.proj, 0, ldf, R, 0);
    E.pipe("attn_prep");
    E.set(0, s.proj);
    E.set(1, L.q_norm);
    E.set(2, L.k_norm);
    E.set(3, meta.m_pos.buf, pos_off);
    E.set(4, m.inv_freq);
    E.set(5, s.Q);
    E.set(6, s.Kb);
    E.set(7, s.Vb);
    AttnPrepParams app{c.H, c.Hkv, ldf, 2 * qdim, qdim, kvd, c.eps};
    E.bytes(8, app);
    E.set(9, m.rope_sel);
    E.rd(s.proj);
    E.wr(s.Q);
    E.wr(s.Kb);
    E.wr(s.Vb);
    E.groups(MTLSizeMake(size_t(c.H + c.Hkv + 3) / 4, size_t(R), 1), MTLSizeMake(128, 1, 1));
    // Sparse attention past 2051 entries: the index of these rows, then
    // the selection of the sparse ones; the steps read what earlier steps wrote.
    QsaSet qd;
    const bool qsa_on = K.i_layers > c.n_full;
    const Buf posb(meta.m_pos.buf.b, meta.m_pos.buf.off + pos_off);
    if (qsa_on) {
      const Buf dtrees = arena_upload(m, meta.arena, plan.trees);
      const Buf zd = arena_upload(m, meta.arena, std::vector<int32_t>(size_t(R) * 4, 0));
      qsa_plan(m, meta.arena, qd, plan.trees, nullptr, true);
      qsa_index(m, E, qd, L, s.xn, posb, dtrees, zd, meta.blocks.buf, c.n_full, true);
      if (qd.n_sparse) qsa_select(m, E, qd, dtrees, zd, meta.blocks.buf, c.n_full, true);
    }
    plan_tiles(plan, c.Hkv, m.attn_split, qd.n_sparse ? &qd.qt_of : nullptr);
    upload(meta.m_trees, plan.trees);
    upload(meta.m_tiles, plan.tiles);
    upload(meta.m_comb, plan.combine);
    meta.parts.ensure<float>(m, std::max<size_t>(size_t(plan.parts) * size_t(c.Hkv) * 8 * size_t(G) * 258, 1));
    E.pipe((qd.n_sparse ? "flash_attention_qsa_g" : "flash_attention_g") + std::to_string(Gk) +
           (hsplit > 1 ? "s" + std::to_string(hsplit) : ""));
    if (qd.n_sparse) qsa_bind(m, E, qd, c.n_full);
    E.set(0, s.Q);
    E.set(1, s.Kb);
    E.set(2, s.Vb);
    E.set(3, K.buf);
    E.set(4, meta.blocks.buf);
    E.set(5, meta.zero_desc.buf);
    E.set(6, s.proj);
    E.set(7, s.O);
    E.bytes(8, ap_at);
    E.set(9, meta.m_tiles.buf);
    E.set(10, meta.m_trees.buf);
    E.set(11, meta.parts.buf);
    E.rd(s.Q);
    E.rd(s.Kb);
    E.rd(s.Vb);
    E.rd(s.proj);
    E.rd(K.buf);  // entries written by the previous steps
    E.wr(s.O);
    if (plan.parts) E.wr(meta.parts.buf, meta.parts.bytes);
    E.groups(MTLSizeMake(size_t(c.Hkv * hsplit), plan.tiles.size(), 1), MTLSizeMake(size_t(64 * Gk), 1, 1));
    if (!plan.combine.empty()) {
      E.pipe("attention_combine");
      E.set(0, meta.parts.buf);
      E.set(1, s.proj);
      E.set(2, s.O);
      E.bytes(3, ap_at);
      E.set(4, meta.m_comb.buf);
      E.rd(meta.parts.buf, meta.parts.bytes);
      E.rd(s.proj);
      E.wr(s.O);
      E.threads(MTLSizeMake(64, size_t(c.H), plan.combine.size() * 8), MTLSizeMake(64, 1, 1));
    }
    upload(meta.mtp_kv_saves, saves);
    E.pipe("kv_save");
    E.set(0, s.Kb);
    E.set(1, s.Vb);
    E.set(2, K.buf);
    E.set(3, meta.mtp_kv_saves.buf);
    E.bytes(4, ap_kv);
    E.rd(s.Kb);
    E.rd(s.Vb);
    E.wr(K.buf);
    E.threads(MTLSizeMake(size_t(kvd / 8), saves.size(), 1), MTLSizeMake(64, 1, 1));
    if (qsa_on) qsa_save(m, E, qd, posb, meta.mtp_kv_saves.buf, saves.size(), c.n_full, true);
    gemm(E, m.tune, L.out, s.O, 0, qdim, m.q4s.y, 0, c.D, R, 0);
    q4_hc_inject(m, E, meta.m_x.buf, Buf(m.q4s.y), Buf(m.q4s.inj), R);
    q4_hc_read(m, E, L.hcm, true, meta.m_x.buf, R, Buf(s.xn));
    E.pipe("q4_zero");
    const uint32_t ny = uint32_t(R) * uint32_t(c.D);
    E.set(0, m.q4s.y);
    E.bytes(1, ny);
    E.wr(m.q4s.y);
    E.threads(MTLSizeMake(ny, 1, 1), MTLSizeMake(256, 1, 1));
    moe_mlp(m, E, L, s.xn, s.act, m.q4s.y, R, nullptr);  // resident 8-bit experts (no expert-cache hook)
    q4_hc_inject(m, E, meta.m_x.buf, Buf(m.q4s.y), Buf(m.q4s.inj), R);
  };
  // Head: the MTP mixer of `rows` of m_x -> LM head -> argmax into arg[k * nq].
  auto head_step = [&](const std::vector<int32_t>& rows, int k) {
    gather_rows(meta.m_x.buf, meta.m_hc.buf, meta.m_rows, rows, W);
    q4_hc_read(m, E, MT.mixer, false, meta.m_hc.buf, nq, meta.m_hn.buf);
    gemm(E, m.tune, m.lm_head, meta.m_hn.buf.b, meta.m_hn.buf.off, c.D, s.logits, 0, c.V, nq, 0, vocab);
    E.pipe(want2 ? "argmax2_rows" : "argmax_rows");
    E.set(0, s.logits);
    E.set(1, meta.m_arg.buf, sizeof(int32_t) * size_t(k) * size_t(nq));
    E.bytes(2, vocab);
    E.bytes(3, c.V);
    if (want2) {
      E.set(4, meta.m_arg2.buf, sizeof(int32_t) * size_t(k) * size_t(nq));
      E.wr(meta.m_arg2.buf, meta.m_arg2.bytes);
    }
    E.rd(s.logits);
    E.wr(meta.m_arg.buf, meta.m_arg.bytes);
    E.groups(MTLSizeMake(size_t(nq), 1, 1), MTLSizeMake(1024, 1, 1));
    draw_draft_step(m, meta, E, reqs, k, vocab);
  };
  // Step 0: hidden streams from the slots.
  E.pipe("pool_to_rows");
  E.set(0, S.buf);
  E.set(1, meta.m_hin.buf);
  E.set(2, meta.m_offs.buf);
  E.bytes(3, W);
  E.rd(S.buf, S.mtp_base * 4, SIZE_MAX);
  E.wr(meta.m_hin.buf, meta.m_hin.bytes);
  E.threads(MTLSizeMake(size_t(W / 4), size_t(rows0), 1), MTLSizeMake(64, 1, 1));
  {
    AttnPlan plan;
    std::vector<KVSaveRec> saves;
    for (int i = 0; i < nq; ++i) {
      const DraftRequest& q = reqs[size_t(i)];
      plan.trees.push_back({r0[size_t(i)], P[size_t(i)], table_off[size_t(i)], q.n_tokens});
      const Slot& sl = m.slots[size_t(q.slot)];
      for (int j = 0; j < q.n_tokens; ++j) {
        const int p = P[size_t(i)] + j;
        saves.push_back({r0[size_t(i)] + j, sl.blocks[size_t(p / bt)], p % bt, 0});
      }
    }
    layer(rows0, meta.m_tok.buf, 0, 0, plan, saves);
    std::vector<int32_t> lastrows;
    for (int i = 0; i < nq; ++i) lastrows.push_back(r0[size_t(i)] + reqs[size_t(i)].n_tokens - 1);
    head_step(lastrows, 0);
    if (kmax > 1) gather_rows(meta.m_x.buf, meta.m_hin.buf, meta.m_hin_rows, lastrows, W);
  }
  for (int k = 1; k < kmax; ++k) {
    AttnPlan plan;
    std::vector<KVSaveRec> saves;
    for (int i = 0; i < nq; ++i) {
      const DraftRequest& q = reqs[size_t(i)];
      const Slot& sl = m.slots[size_t(q.slot)];
      const int Pk = P[size_t(i)] + q.n_tokens + k - 1;
      plan.trees.push_back({i, Pk, table_off[size_t(i)], 1});
      saves.push_back({i, sl.blocks[size_t(Pk / bt)], Pk % bt, 0});
    }
    layer(nq, meta.m_arg.buf, sizeof(int32_t) * size_t(k - 1) * size_t(nq),
          sizeof(int32_t) * 4 * (size_t(Rmax) + size_t(k - 1) * size_t(nq)), plan, saves);
    std::vector<int32_t> rows(static_cast<size_t>(nq));
    for (int i = 0; i < nq; ++i) rows[size_t(i)] = i;
    head_step(rows, k);
    if (k + 1 < kmax) gather_rows(meta.m_x.buf, meta.m_hin.buf, meta.m_hin_rows, rows, W);
  }
  cb = E.finish();
  const double enc_ms = ms_since(t_enc);
  commit_wait(cb, 5);
  const int32_t* out = static_cast<const int32_t*>(ptr(meta.m_arg.buf));
  const int32_t* out2 = want2 ? static_cast<const int32_t*>(ptr(meta.m_arg2.buf)) : nullptr;
  for (int i = 0; i < nq; ++i)
    for (int k = 0; k < reqs[size_t(i)].n_draft; ++k) {
      reqs[size_t(i)].out[k] = out[size_t(k) * size_t(nq) + size_t(i)];
      if (reqs[size_t(i)].out2) reqs[size_t(i)].out2[k] = out2[size_t(k) * size_t(nq) + size_t(i)];
    }
  if (stats) {
    stats->gpu_ms = gpu_ms(cb);
    stats->encode_ms = enc_ms;
    stats->wall_ms = ms_since(t_wall);
    stats->rows = rows0 + (kmax - 1) * nq;
    stats->readouts = nq * kmax;
    stats->dispatches = m.dispatches;
    stats->barriers = m.barriers;
  }
}

inline void run_draft(Model& m, PassMeta& meta, const std::vector<DraftRequest>& reqs, PassStats* stats) {
  const auto t_wall = Clock::now();
  ++m.scratch_seq;  // the pass arena and the q4 scratch of the last pass are reused from here
  meta.arena.reset();
  m.tune.retired.clear();
  const Config& c = m.cfg;
  if (!m.mtp.present) fail("this model has no MTP head");
  const int nq = int(reqs.size());
  if (nq == 0) return;
  int kmax = 0, rows0 = 0;
  for (const DraftRequest& q : reqs) {
    Slot& S = user_slot(m, q.slot);
    if (!S.mtp_ok || S.len < 1) fail("the slot has no MTP state (save it with mtp=true)");
    if (q.n_tokens < 1 || q.lh_first < 0 || q.lh_first + q.n_tokens - 1 > S.n_lh)
      fail("n_tokens exceeds the slot's leaf rows + 1");
    if (q.n_draft < 1 || q.n_draft > kMaxDraft) fail("n_draft out of range");
    if (S.len + q.n_tokens + q.n_draft > m.max_ctx) fail("context exceeds max_ctx");
    for (int i = 0; i < q.n_tokens; ++i)
      if (q.tokens[i] < 0 || q.tokens[i] >= c.V) fail("token out of range");
    kmax = std::max(kmax, q.n_draft);
    rows0 += q.n_tokens;
  }
  for (int i = 0; i < nq; ++i)
    for (int j = i + 1; j < nq; ++j)
      if (reqs[size_t(i)].slot == reqs[size_t(j)].slot) fail("two draft requests for one slot");
  if (m.ctr_dirty) reset_arrival_counters(m);
  {  // the blocks the draft takes, counted before any slot changes
    int need = 0;
    for (const DraftRequest& q : reqs) {
      const Slot& sl = m.slots[size_t(q.slot)];
      need += own_blocks_need(m, sl, sl.len - 1, sl.len - 1 + q.n_tokens + kmax);
    }
    if (need > m.kv_free())
      fail("KV cache full: drafting needs " + std::to_string(need) + " blocks, " + std::to_string(m.kv_free()) + " free");
  }
  OwnedBlocks owned(m);  // given back unless the draft completes
  if (c.q4) {
    run_draft_q4(m, meta, reqs, stats, kmax, rows0, t_wall, &owned);
    owned.armed = false;
    return;
  }
  const StatePool& S = m.st;
  const KVPool& K = m.kv;
  const int bt = K.bt;
  const int G = c.H / c.Hkv;
  const int qdim = c.H * c.HD, kvd = c.kv_dim(), ldf = c.full_ld();
  ensure_scratch(m, std::max(rows0, nq), nq);
  Scratch& s = m.sc;
  const int Rmax = std::max(rows0, nq);

  // Blocks: exclusive from len - 1 (first MTP entry written), covering every draft step.
  std::vector<std::array<int32_t, 2>> cow;
  for (const DraftRequest& q : reqs) {
    Slot& sl = m.slots[size_t(q.slot)];
    own_blocks_from(m, sl, sl.len - 1, sl.len - 1 + q.n_tokens + kmax, cow, &owned);
  }
  // Tables (after copy on write: drafting reads its own fresh entries).
  std::vector<int32_t> table;
  std::vector<int> table_off(static_cast<size_t>(nq));
  for (int i = 0; i < nq; ++i) {
    const Slot& sl = m.slots[size_t(reqs[size_t(i)].slot)];
    table_off[size_t(i)] = int(table.size());
    table.insert(table.end(), sl.blocks.begin(), sl.blocks.end());
  }
  auto upload = [&](HostBuf& hb, const auto& v) {
    using T = typename std::decay_t<decltype(v)>::value_type;
    T* p = hb.ensure<T>(m, std::max<size_t>(v.size(), 1));
    if (!v.empty()) std::memcpy(p, v.data(), v.size() * sizeof(T));
  };
  upload(meta.blocks, table);
  upload(meta.cow, cow);

  // Step-0 rows.
  int32_t* tok = meta.m_tok.ensure<int32_t>(m, size_t(Rmax));
  int32_t* pos = meta.m_pos.ensure<int32_t>(m, size_t(Rmax) * 4 * size_t(kmax + 1));
  uint64_t* offs = meta.m_offs.ensure<uint64_t>(m, size_t(Rmax));
  meta.m_hin.ensure<float>(m, size_t(Rmax) * size_t(c.D));
  meta.m_x.ensure<float>(m, size_t(Rmax) * size_t(c.D));
  meta.m_emb.ensure<float>(m, size_t(Rmax) * size_t(c.D));
  meta.m_hn.ensure<uint16_t>(m, size_t(nq) * size_t(c.D));
  int32_t* arg = meta.m_arg.ensure<int32_t>(m, size_t(nq) * size_t(kmax));
  (void)arg;
  bool want2 = false;  // runner-ups for sibling leaves
  for (const DraftRequest& q : reqs) want2 = want2 || q.out2;
  if (want2) meta.m_arg2.ensure<int32_t>(m, size_t(nq) * size_t(kmax));
  int32_t* zdesc = meta.zero_desc.ensure<int32_t>(m, size_t(Rmax) * 4);
  std::memset(zdesc, 0, size_t(Rmax) * 16);
  std::vector<int> r0(static_cast<size_t>(nq)), P(static_cast<size_t>(nq)), last_pos(static_cast<size_t>(nq));
  int r = 0;
  for (int i = 0; i < nq; ++i) {
    const DraftRequest& q = reqs[size_t(i)];
    const Slot& sl = m.slots[size_t(q.slot)];
    r0[size_t(i)] = r;
    P[size_t(i)] = sl.len - 1;
    for (int j = 0; j < q.n_tokens; ++j, ++r) {
      tok[r] = q.tokens[j];
      for (int d = 0; d < 3; ++d) pos[4 * r + d] = j == 0 ? sl.mh_pos[d] : sl.next_pos + j - 1;
      pos[4 * r + 3] = 0;
      offs[r] = j == 0 ? S.mh_off(q.slot) : S.lh_off(q.slot, q.lh_first + j - 1);
    }
    last_pos[size_t(i)] = pos[4 * (r - 1)];
  }
  // Positions of chained steps: [kmax][nq] after the step-0 rows.
  int32_t* cpos = pos + 4 * size_t(Rmax);
  for (int k = 1; k < kmax; ++k)
    for (int i = 0; i < nq; ++i) {
      int32_t* p = cpos + 4 * (size_t(k - 1) * size_t(nq) + size_t(i));
      p[0] = p[1] = p[2] = last_pos[size_t(i)] + k;
      p[3] = 0;
    }
  const int vocab = m.mtp.vocab > 0 ? std::min(m.mtp.vocab, c.V) : c.V;

  const auto t_enc = Clock::now();
  m.dispatches = 0;
  m.barriers = 0;
  id<MTLCommandBuffer> cb = [m.queue commandBufferWithUnretainedReferences];
  Enc E(m, cb);
  if (!cow.empty()) {
    kv_cow_copy(m, E, meta.cow.buf, cow.size());
    E.barrier();
  }
  const Mtp& MT = m.mtp;
  const Layer& L = MT.layer;
  const Mat& head = (MT.head.out() > 0 && MT.use_head4) ? MT.head : m.lm_head;
  // Coarse draft head: the 2-bit head over the draft vocabulary, then
  // the LM head over the shortlisted column blocks of each row.
  const int n_vblocks = (vocab + 31) / 32;
  const bool coarse = MT.coarse.out() > 0 && !MT.coarse.dense && MT.coarse_c > 0 && nq <= 8 && !m.lm_head.dense &&
                      n_vblocks <= 8192;
  const int cC = coarse ? std::min(MT.coarse_c, 256) : 0;
  if (coarse) {
    meta.m_bmax.ensure<float>(m, size_t(nq) * size_t(n_vblocks));
    meta.m_blist.ensure<int32_t>(m, size_t(nq) * size_t(cC));
  }
  auto rmsnorm = [&](const Buf& X, size_t xoff, const Buf& w, const Buf& Y, int rows) {
    E.pipe("rmsnorm");
    E.set(0, X, xoff);
    E.set(1, w);
    E.set(2, Y);
    E.bytes(3, c.D);
    E.bytes(4, c.eps);
    const int g = 0;
    E.set(5, X);
    E.bytes(6, g);
    E.rd(X.b);
    E.wr(Y.b);
    E.groups(MTLSizeMake(size_t(rows), 1, 1), MTLSizeMake(256, 1, 1));
  };
  auto gather_rows = [&](const Buf& src, const Buf& dst, HostBuf& rows_b, const std::vector<int32_t>& rows) {
    upload(rows_b, rows);
    E.pipe("gather_rows_f32");
    E.set(0, src);
    E.set(1, dst);
    E.set(2, rows_b.buf);
    E.bytes(3, c.D);
    E.rd(src.b);
    E.wr(dst.b);
    E.threads(MTLSizeMake(size_t(c.D / 4), rows.size(), 1), MTLSizeMake(64, 1, 1));
  };
  // One MTP layer over `R` rows: tokens at tok_b+tok_off, positions at
  // pos_b+pos_off, hidden inputs in m_hin (`pre`: the norm they still need).
  auto layer = [&](int R, const Buf& tok_b, size_t tok_off, size_t pos_off, const Buf& pre, AttnPlan& plan,
                   const std::vector<KVSaveRec>& saves) {
    static const char* names[3] = {"embed_q4", "embed_q8", "embed_f16"};
    const Embed& T = m.embed;
    E.pipe(names[T.kind]);
    E.set(0, tok_b, tok_off);
    E.set(1, T.codes);
    if (T.kind != 2) {
      E.set(2, T.s);
      E.set(3, T.z);
    }
    E.set(4, meta.m_emb.buf);
    E.bytes(5, c.D);
    if (T.kind != 2) E.bytes(6, T.group);
    E.rd(tok_b.b);
    E.wr(meta.m_emb.buf, meta.m_emb.bytes);
    E.threads(MTLSizeMake(size_t(c.D / 8), size_t(R), 1), MTLSizeMake(size_t(std::min(c.D / 8, 256)), 1, 1));
    E.pipe("mtp_prep");
    E.set(0, meta.m_emb.buf);
    E.set(1, meta.m_hin.buf);
    E.set(2, MT.enorm);
    E.set(3, MT.hnorm);
    E.set(4, pre ? pre : MT.hnorm);
    E.set(5, s.act);
    E.bytes(6, c.D);
    E.bytes(7, c.eps);
    const int has_pre = pre ? 1 : 0;
    E.bytes(8, has_pre);
    E.rd(meta.m_emb.buf, meta.m_emb.bytes);
    E.rd(meta.m_hin.buf, meta.m_hin.bytes);
    E.wr(s.act);
    E.groups(MTLSizeMake(size_t(R), 1, 1), MTLSizeMake(256, 1, 1));
    gemm(E, m.tune, MT.fc, s.act, 0, 2 * c.D, meta.m_x.buf.b, meta.m_x.buf.off, c.D, R, 0);
    rmsnorm(meta.m_x.buf.b, meta.m_x.buf.off, L.in_norm, s.xn, R);
    gemm_runs(E, m.tune, L.in_proj, s.xn, 0, c.D, s.proj, 0, ldf, R, 0);
    E.pipe("attn_prep");
    E.set(0, s.proj);
    E.set(1, L.q_norm);
    E.set(2, L.k_norm);
    E.set(3, meta.m_pos.buf, pos_off);
    E.set(4, m.inv_freq);
    E.set(5, s.Q);
    E.set(6, s.Kb);
    E.set(7, s.Vb);
    AttnPrepParams app{c.H, c.Hkv, ldf, 2 * qdim, qdim, kvd, c.eps};
    E.bytes(8, app);
    E.set(9, m.rope_sel);
    E.rd(s.proj);
    E.wr(s.Q);
    E.wr(s.Kb);
    E.wr(s.Vb);
    E.groups(MTLSizeMake(size_t(c.H + c.Hkv + 3) / 4, size_t(R), 1), MTLSizeMake(128, 1, 1));
    plan_tiles(plan, c.Hkv, m.attn_split);
    upload(meta.m_trees, plan.trees);
    upload(meta.m_tiles, plan.tiles);
    upload(meta.m_comb, plan.combine);
    meta.parts.ensure<float>(m, std::max<size_t>(size_t(plan.parts) * size_t(c.Hkv) * 8 * size_t(G) * 258, 1));
    AttnParams ap{c.H, c.Hkv, qdim, kvd, ldf, bt, uint64_t(K.layer_base(c.n_full))};
    // Single-row trees (every draft step after the first): the GQA-row decode kernel.
    bool gqa = m.attn_gqa && G <= 8;
    for (const auto& t : plan.trees) gqa = gqa && t.M == 1;
    E.pipe((gqa ? "attn_decode_gqa_g" : "flash_attention_g") + std::to_string(G));
    E.set(0, s.Q);
    E.set(1, s.Kb);
    E.set(2, s.Vb);
    E.set(3, K.buf);
    E.set(4, meta.blocks.buf);
    E.set(5, meta.zero_desc.buf);
    E.set(6, s.proj);
    E.set(7, s.O);
    E.bytes(8, ap);
    E.set(9, meta.m_tiles.buf);
    E.set(10, meta.m_trees.buf);
    E.set(11, meta.parts.buf);
    E.rd(s.Q);
    E.rd(s.Kb);
    E.rd(s.Vb);
    E.rd(s.proj);
    E.rd(K.buf);  // reads entries written by the previous steps
    E.wr(s.O);
    if (plan.parts) E.wr(meta.parts.buf, meta.parts.bytes);
    E.groups(MTLSizeMake(size_t(c.Hkv), plan.tiles.size(), 1), MTLSizeMake(size_t(gqa ? 128 : 64 * G), 1, 1));
    if (!plan.combine.empty()) {
      E.pipe("attention_combine");
      E.set(0, meta.parts.buf);
      E.set(1, s.proj);
      E.set(2, s.O);
      E.bytes(3, ap);
      E.set(4, meta.m_comb.buf);
      E.rd(meta.parts.buf, meta.parts.bytes);
      E.rd(s.proj);
      E.wr(s.O);
      E.threads(MTLSizeMake(64, size_t(c.H), plan.combine.size() * 8), MTLSizeMake(64, 1, 1));
    }
    upload(meta.mtp_kv_saves, saves);
    E.pipe("kv_save");
    E.set(0, s.Kb);
    E.set(1, s.Vb);
    E.set(2, K.buf);
    E.set(3, meta.mtp_kv_saves.buf);
    E.bytes(4, ap);
    E.rd(s.Kb);
    E.rd(s.Vb);
    E.wr(K.buf);
    E.threads(MTLSizeMake(size_t(kvd / 8), saves.size(), 1), MTLSizeMake(64, 1, 1));
    gemm(E, m.tune, L.out, s.O, 0, qdim, meta.m_x.buf.b, meta.m_x.buf.off, c.D, R, 1);
    rmsnorm(meta.m_x.buf.b, meta.m_x.buf.off, L.post_norm, s.xn, R);
    if (L.moe) {
      moe_mlp(m, E, L, s.xn, s.act, meta.m_x.buf.b, R, nullptr, meta.m_x.buf.off);
    } else {
      shared_mlp_in(m, E, L, s.xn, s.act, R);
      gemm(E, m.tune, L.down, s.act, 0, L.I, meta.m_x.buf.b, meta.m_x.buf.off, c.D, R, 1);
    }
  };
  // Head: MTP norm of `rows` of m_x -> LM head -> argmax into arg[k * nq].
  auto head_step = [&](const std::vector<int32_t>& rows, int k) {
    gather_rows(meta.m_x.buf, meta.m_emb.buf, meta.m_rows, rows);  // m_emb as a staging area
    rmsnorm(meta.m_emb.buf, 0, MT.norm, meta.m_hn.buf, nq);
    if (coarse) {
      gemv_blocks(E, MT.coarse.q, meta.m_hn.buf.b, meta.m_hn.buf.off, c.D, s.logits, 0, c.V, nq, vocab);
      E.pipe("block_max_rows");
      E.set(0, s.logits);
      E.set(1, meta.m_bmax.buf);
      E.bytes(2, vocab);
      E.bytes(3, c.V);
      E.bytes(4, n_vblocks);
      E.rd(s.logits);
      E.wr(meta.m_bmax.buf, meta.m_bmax.bytes);
      E.threads(MTLSizeMake(size_t(n_vblocks), size_t(nq), 1), MTLSizeMake(64, 1, 1));
      E.pipe("topc_blocks");
      E.set(0, meta.m_bmax.buf);
      E.set(1, meta.m_blist.buf);
      E.bytes(2, n_vblocks);
      E.bytes(3, cC);
      E.rd(meta.m_bmax.buf, meta.m_bmax.bytes);
      E.wr(meta.m_blist.buf, meta.m_blist.bytes);
      E.groups(MTLSizeMake(size_t(nq), 1, 1), MTLSizeMake(1024, 1, 1));
      gemv_blocks(E, m.lm_head.q, meta.m_hn.buf.b, meta.m_hn.buf.off, c.D, s.logits, 0, c.V, nq, 0, &meta.m_blist.buf,
                  nq * cC);
      E.pipe("argmax2_list");
      E.set(0, s.logits);
      E.set(1, meta.m_arg.buf, sizeof(int32_t) * size_t(k) * size_t(nq));
      E.bytes(2, vocab);
      E.bytes(3, c.V);
      if (want2) {
        E.set(4, meta.m_arg2.buf, sizeof(int32_t) * size_t(k) * size_t(nq));
        E.wr(meta.m_arg2.buf, meta.m_arg2.bytes);
      }
      E.set(5, meta.m_blist.buf);
      E.bytes(6, cC);
      E.bytes(7, want2 ? 1 : 0);
      E.rd(s.logits);
      E.rd(meta.m_blist.buf, meta.m_blist.bytes);
      E.wr(meta.m_arg.buf, meta.m_arg.bytes);
      E.groups(MTLSizeMake(size_t(nq), 1, 1), MTLSizeMake(1024, 1, 1));
      draw_draft_step(m, meta, E, reqs, k, vocab);
      return;
    }
    gemm(E, m.tune, head, meta.m_hn.buf.b, meta.m_hn.buf.off, c.D, s.logits, 0, c.V, nq, 0, vocab);
    E.pipe(want2 ? "argmax2_rows" : "argmax_rows");
    E.set(0, s.logits);
    E.set(1, meta.m_arg.buf, sizeof(int32_t) * size_t(k) * size_t(nq));
    E.bytes(2, vocab);
    E.bytes(3, c.V);
    if (want2) {
      E.set(4, meta.m_arg2.buf, sizeof(int32_t) * size_t(k) * size_t(nq));
      E.wr(meta.m_arg2.buf, meta.m_arg2.bytes);
    }
    E.rd(s.logits);
    E.wr(meta.m_arg.buf, meta.m_arg.bytes);
    E.groups(MTLSizeMake(size_t(nq), 1, 1), MTLSizeMake(1024, 1, 1));
    draw_draft_step(m, meta, E, reqs, k, vocab);
  };

  // Step 0: hidden inputs from the slots.
  E.pipe("pool_to_rows");
  E.set(0, S.buf);
  E.set(1, meta.m_hin.buf);
  E.set(2, meta.m_offs.buf);
  E.bytes(3, c.D);
  E.rd(S.buf, S.mtp_base * 4, SIZE_MAX);
  E.wr(meta.m_hin.buf, meta.m_hin.bytes);
  E.threads(MTLSizeMake(size_t(c.D / 4), size_t(rows0), 1), MTLSizeMake(64, 1, 1));
  {
    AttnPlan plan;
    std::vector<KVSaveRec> saves;
    for (int i = 0; i < nq; ++i) {
      const DraftRequest& q = reqs[size_t(i)];
      plan.trees.push_back({r0[size_t(i)], P[size_t(i)], table_off[size_t(i)], q.n_tokens});
      const Slot& sl = m.slots[size_t(q.slot)];
      for (int j = 0; j < q.n_tokens; ++j) {
        const int p = P[size_t(i)] + j;
        saves.push_back({r0[size_t(i)] + j, sl.blocks[size_t(p / bt)], p % bt, 0});
      }
    }
    const Buf none;
    layer(rows0, meta.m_tok.buf, 0, 0, MT.post_norm ? m.final_norm : none, plan, saves);
    std::vector<int32_t> lastrows;
    for (int i = 0; i < nq; ++i) lastrows.push_back(r0[size_t(i)] + reqs[size_t(i)].n_tokens - 1);
    head_step(lastrows, 0);
    // Hidden input of step 1: the MTP output of each request's last row.
    if (kmax > 1) gather_rows(meta.m_x.buf, meta.m_hin.buf, meta.m_hin_rows, lastrows);
  }
  for (int k = 1; k < kmax; ++k) {
    AttnPlan plan;
    std::vector<KVSaveRec> saves;
    for (int i = 0; i < nq; ++i) {
      const DraftRequest& q = reqs[size_t(i)];
      const Slot& sl = m.slots[size_t(q.slot)];
      const int Pk = P[size_t(i)] + q.n_tokens + k - 1;
      plan.trees.push_back({i, Pk, table_off[size_t(i)], 1});
      saves.push_back({i, sl.blocks[size_t(Pk / bt)], Pk % bt, 0});
    }
    layer(nq, meta.m_arg.buf, sizeof(int32_t) * size_t(k - 1) * size_t(nq),
          sizeof(int32_t) * 4 * (size_t(Rmax) + size_t(k - 1) * size_t(nq)), MT.norm, plan, saves);
    std::vector<int32_t> rows(static_cast<size_t>(nq));
    for (int i = 0; i < nq; ++i) rows[size_t(i)] = i;
    head_step(rows, k);
    if (k + 1 < kmax) gather_rows(meta.m_x.buf, meta.m_hin.buf, meta.m_hin_rows, rows);
  }
  cb = E.finish();
  const double enc_ms = ms_since(t_enc);
  commit_wait(cb, 5);
  owned.armed = false;  // the copies ran: the slots keep their own blocks
  const int32_t* out = static_cast<const int32_t*>(ptr(meta.m_arg.buf));
  const int32_t* out2 = want2 ? static_cast<const int32_t*>(ptr(meta.m_arg2.buf)) : nullptr;
  for (int i = 0; i < nq; ++i)
    for (int k = 0; k < reqs[size_t(i)].n_draft; ++k) {
      reqs[size_t(i)].out[k] = out[size_t(k) * size_t(nq) + size_t(i)];
      if (reqs[size_t(i)].out2) reqs[size_t(i)].out2[k] = out2[size_t(k) * size_t(nq) + size_t(i)];
    }
  if (stats) {
    stats->gpu_ms = gpu_ms(cb);
    stats->encode_ms = enc_ms;
    stats->wall_ms = ms_since(t_wall);
    stats->rows = rows0 + (kmax - 1) * nq;
    stats->readouts = nq * kmax;
    stats->dispatches = m.dispatches;
    stats->barriers = m.barriers;
  }
}

}  // namespace espetero::native
