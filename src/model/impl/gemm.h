// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// GEMM dispatch and per-device tile autotuning.
//
// rgemm / rgemmN: quantized weights (4-bit nibbles, 5/6-bit nibble + high
// planes, bytes), specialized per quantization group; tile = rows x
// simdgroups of 32 columns, optional deterministic split-K. rgemv: 1-2 rows,
// near memory bandwidth. rgemv8: 3-8 rows with MMA accumulators. hgemm: dense
// fp16 weights with bias/GELU/SwiGLU epilogues.
#pragma once

#include "../../util/paths.h"
#include "matrix.h"

namespace espetero::native {

struct GemmShape {
  int bm, bn, threads;
};

// rgemm configs: 8x4, 16x4, 32x4, 16x8, 32x8, 8x8, 64x4 (rows x simdgroups of 32 cols),
// then the shared-dequantization tiles 32x4s and 64x4s (2 simdgroups per column block)
constexpr int kGemmConfigs = 9;
// Largest M for which every matrix also tries split-K tiles.
constexpr int kSplitMaxM = 64;

inline GemmShape gemm_shape(int cfg) {
  static const GemmShape shapes[kGemmConfigs] = {
      {8, 128, 128}, {16, 128, 128}, {32, 128, 128}, {16, 256, 256}, {32, 256, 256}, {8, 256, 256}, {64, 128, 128},
      {32, 128, 256}, {64, 128, 256}};
  return shapes[cfg];
}

inline std::string rgemm_name(int bits, int cfg, int epi) {
  static const char* tiles[kGemmConfigs] = {"8x4", "16x4", "32x4", "16x8", "32x8", "8x8", "64x4", "32x4s", "64x4s"};
  static const char* epis[3] = {"store", "add", "swiglu"};
  const std::string pre = bits == 4 ? "rgemm_" : "rgemm" + std::to_string(bits) + "_";
  return pre + tiles[cfg] + "_" + epis[epi];
}

struct Tuner {
  int n_cfgs = 7;             // configs the autotuner tries (ESPETERO_GEMM_CFGS=9 adds the shared-dequant tiles, measured slower)
  // (N, K, bits, fc) -> choice per 32-row M bucket: config | (splits << 4)
  std::map<std::tuple<int, int, int, int>, std::vector<uint8_t>> table;
  double ms = 0;
  bool gemv = true;        // decode-sized rows use rgemv ("gemv": 0 disables)
  bool gemv8 = true;       // 3..8 rows use the MMA GEMV ("gemv8": 0 disables)
  int gemv_max_m = 2;      // rows served by the FMA GEMV (1..kGemvInstM; ESPETERO_GEMV_MAXM)
  // GEMMs of more than 8 rows take the untuned tile (config 2, no split
  // K) whatever their row count, so a row's result does not depend on the
  // rows beside it (embedding passes, "gemm_fixed").
  bool fixed = false;
  bool gemv_splitk = false; // decode GEMVs with few column blocks split K over threadgroups (+10 us per call, off)
  int gemv_ks = 8;          // K-slices per 1-row GEMV threadgroup when it has few column blocks: 8 or 16 (ESPETERO_GEMV_KS)
  int ks16_blocks = 128;    // ... "few": fewer column blocks than this
  // Micro benchmarks: a kernel name for decode-sized GEMMs without norm
  // links or epilogue ('%' stands for the code width, e.g. "rgemv8r%_d1_store");
  // it runs on the rgemv / rgemv8 grid (32 columns per 256-thread group).
  std::string diag_kernel;
  id<MTLBuffer> ws = nil;  // split-K partials
  id<MTLBuffer> kctr = nil;  // split-K GEMV arrival counters (one int per column block, self-resetting)
  // Prefill GEMMs on 6-bit weights: from this many rows the matrix is
  // widened to byte codes in `widen` right before its GEMM (0: never).
  int widen6_m = 0;  // measured 0.98-1.01x: off by default (opt-in)
  id<MTLBuffer> widen = nil;
  // Workspaces replaced during a pass stay alive until it completes (command
  // buffers do not retain their buffers).
  std::vector<id<MTLBuffer>> retired;

  void ensure_ws(Runtime& rt, size_t floats) {
    if (!ws || [ws length] < floats * 4) {
      if (ws) retired.push_back(ws);
      ws = rt.buffer(std::max<size_t>(floats * 4, size_t(4) << 20));
    }
  }
  void ensure_widen(Runtime& rt, size_t bytes) {
    if (!widen || [widen length] < bytes) {
      if (widen) retired.push_back(widen);
      widen = rt.buffer(bytes);
    }
  }
  static constexpr int kMaxSplitBlocks = 16384;
  void ensure_kctr(Runtime& rt) {
    if (kctr) return;
    kctr = rt.buffer(sizeof(int32_t) * size_t(kMaxSplitBlocks));
    std::memset([kctr contents], 0, sizeof(int32_t) * size_t(kMaxSplitBlocks));
  }
};

struct GemmParams {
  int M, N, K, lda, ldy, g_split, z_stride;
};

// Profiling key for a GEMM dispatch: kernel, shape and rows.
inline void gemm_tag(Enc& E, const std::string& kernel, int N, int K, int M) {
  if (E.rt.profile) E.label(kernel + " " + std::to_string(N) + "x" + std::to_string(K) + " M" + std::to_string(M));
}

// Decode-sized rows: the FMA matrix-vector kernel (rgemv, one multiply per
// weight and row) up to 2 rows, the 8-row MMA GEMV (rgemv8) above. At 3-4
// rows (MTP verify passes are 1 + 3 rows) the FMA kernel measured 26% slower
// on the 27B verify pass and 15% on the MoE's, so the FMA kernels stay
// instantiated for 1-2 rows only.
constexpr int kGemvInstM = 2;
constexpr int kGemvMaxM = kGemvInstM;
inline std::string rgemv_name(int bits, int M, int epi) {
  static const char* epis[3] = {"store", "add", "swiglu"};
  const int mm = std::max(1, std::min(M, kGemvInstM));
  return "rgemv" + std::to_string(bits) + "_" + std::to_string(mm) + "_" + epis[epi];
}

// Bytes of the output rows a GEMM touches (for hazard tracking).
inline size_t out_span(int M, int ldy, int N, int epi) {
  const size_t elt = epi == 2 ? 2 : 4;
  return (size_t(M - 1) * size_t(ldy) + size_t(epi == 2 ? N / 2 : N)) * elt;
}

// RMSNorm fused into decode-sized GEMVs. Producer link (kind 1, `nout`): the
// residual-add GEMV writes one partial sum of squares per row and 32-column
// block into `parts`, and its last-arriving threadgroup writes the normalized
// fp16 rows `xn` (with the next norm's weight `w`) that the following GEMVs
// consume unchanged; `ctr` is that dispatch's arrival counter (zeroed before
// the pass, reset by the finalizer). Consumer link (kind 2, `nin`): the gated
// per-head norm applied on load by the out projection (z rows in `parts`).
struct NormLink {
  int kind = 1;     // 1: RMSNorm producer; 2: gated per-head norm consumer
  Buf parts;        // kind 1: [rows][ld] partials; kind 2: z rows (fp32, stride ld)
  int ld = 0;       // floats per row (>= P)
  int P = 0;        // partials per row: 32-column blocks of the producer
  int D = 0;        // normalized width
  float eps = 1e-6f;
  Buf w;            // norm weight (fp32 [D]): kind 1 the next norm's, kind 2 the gated norm's
  Buf xn;           // kind 1: normalized fp16 rows written by the finalizer (stride = ldy)
  Buf ctr;          // kind 1: one int, arrival counter of the producing dispatch
};

struct NormParamsHost {
  int32_t P, part_ld, D;
  float eps;
  int32_t out_ld;
};

// True when a GEMM of M rows runs as a GEMV kernel (fusion is possible).
inline bool gemv_path(const Tuner& T, int M) { return (M <= T.gemv_max_m && T.gemv) || (M <= 8 && T.gemv && T.gemv8); }

// epi: 0 store fp32, 1 accumulate fp32, 2 SwiGLU -> fp16. n_cols > 0 (GEMV
// only): compute just the first ~n_cols outputs (a vocabulary prefix).
// nin: gated per-head norm applied on load; nout: residual add + norm partials
// + normalized fp16 rows for the next norm (GEMV paths only).
inline void gemm_q(Enc& E, Tuner& T, const QW& w, id<MTLBuffer> A, size_t aoff, int lda, id<MTLBuffer> Y,
                   size_t yoff, int ldy, int M, int epi, int n_cols = 0, const NormLink* nin = nullptr,
                   const NormLink* nout = nullptr) {
  if (M <= 0) return;
  // Inputs are fp16 (plain rows, or the fp16 residual copy of the fused RMSNorm);
  // the gated per-head norm (kind 2) reads fp32 rows.
  const size_t a_bytes = (size_t(M - 1) * size_t(lda) + size_t(w.in)) * ((nin && nin->kind == 2) ? 4 : 2);
  const size_t y_bytes = out_span(M, ldy, w.out, epi);
  const QW* wsrc = &w;  // the weights the GEMM reads (a byte-code copy of 6-bit ones)
  auto bind = [&](id<MTLBuffer> dst, size_t dst_off, int ld) {
    E.set(0, A, aoff);
    E.set(1, wsrc->wr);
    E.set(2, wsrc->sb);
    E.set(3, wsrc->wh ? wsrc->wh : wsrc->wr);
    E.set(4, dst, dst_off);
    (void)ld;
  };
  auto bind_norm = [&]() {
    if (!nin && !nout) return;
    const NormLink* d = nout ? nout : nin;
    NormParamsHost np{nin ? nin->P : 0, nin ? nin->ld : 0, d->D, d->eps, nout ? nout->ld : 0};
    E.bytes(6, np);
    if (nin) {
      E.set(7, nin->w);
      E.set(8, nin->parts);
      E.rd(nin->parts.b, nin->parts.off, nin->parts.off + size_t(M) * size_t(nin->ld) * 4);  // whole z rows (kind 2)
    }
    if (nout) {
      E.set(7, nout->w);  // the next norm's weight, applied by the finalizer
      E.set(9, nout->parts);
      E.wr(nout->parts.b, nout->parts.off, nout->parts.off + size_t(M) * size_t(nout->ld) * 4);
      E.set(10, nout->xn);
      E.wr(nout->xn.b, nout->xn.off, nout->xn.off + (size_t(M - 1) * size_t(ldy) + size_t(w.out)) * 2);
      E.set(11, nout->ctr);
      E.rd(nout->ctr.b, nout->ctr.off, nout->ctr.off + 4);
      E.wr(nout->ctr.b, nout->ctr.off, nout->ctr.off + 4);
    }
  };
  static const char* epis4[4] = {"_store", "_add", "_swiglu", "_addp"};
  const int e = nout ? 3 : epi;
  if ((nin || nout) && !gemv_path(T, M)) fail("fused norm needs a GEMV-sized pass");
  // An RMSNorm'd input is the residual stream itself: never also the output.
  if (nin && nin->kind == 1 && (e == 1 || e == 3)) fail("fused norm input with a residual epilogue");
  const char* nsuf = !nin ? "" : (nin->kind == 2 ? "_g" : "_n");
  if (!T.diag_kernel.empty() && M <= 8 && !nin && !nout && epi == 0 && n_cols == 0) {
    std::string name = T.diag_kernel;
    const size_t pc = name.find('%');
    if (pc != std::string::npos) name.replace(pc, 1, std::to_string(w.bits));
    GemmParams p{M, w.out, w.in, lda, ldy, 0, 0};
    E.pipe(name, w.fc());
    gemm_tag(E, name, w.out, w.in, M);
    bind(Y, yoff, ldy);
    E.bytes(5, p);
    E.rd(A, aoff, aoff + a_bytes);
    E.wr(Y, yoff, yoff + y_bytes);
    E.groups(MTLSizeMake(size_t(w.out + 31) / 32, 1, 1), MTLSizeMake(256, 1, 1));
    return;
  }
  if (M <= T.gemv_max_m && (T.gemv || n_cols > 0)) {
    const int n = n_cols > 0 ? std::min(n_cols, w.out) : w.out;
    const int nblocks = (n + 31) / 32;
    const int KG = w.in / 64;
    // Few column blocks: split K over threadgroups (last arriver reduces).
    int splits = 1, g_split = 0;
    if (T.gemv_splitk && nblocks < 128 && nblocks <= Tuner::kMaxSplitBlocks && KG >= 8) {
      splits = std::min({8, std::max(1, 256 / nblocks), KG / 4});
      g_split = (KG + splits - 1) / splits;
      splits = (KG + g_split - 1) / g_split;
      if (splits < 2) splits = 1, g_split = 0;
    }
    GemmParams p{M, w.out, w.in, lda, ldy, g_split, M * w.out};
    // Few column blocks, one row: 16 K-slices per threadgroup.
    const bool k16 = T.gemv_ks == 16 && M == 1 && splits == 1 && nblocks < T.ks16_blocks;
    // 3-4 rows (gemv_max_m raised with ESPETERO_FMA34=1): the JIT FMA GEMV.
    const std::string name = "rgemv" + std::to_string(w.bits) + "_" + std::to_string(std::max(1, M)) + (k16 ? "k16" : "") +
                             (M >= 3 ? "j" : "") + epis4[e] + nsuf;
    E.pipe(name, w.fc());
    gemm_tag(E, name + (splits > 1 ? "/s" + std::to_string(splits) : ""), w.out, w.in, M);
    bind(Y, yoff, ldy);
    E.bytes(5, p);
    bind_norm();
    if (splits > 1) {
      T.ensure_ws(E.rt, size_t(splits) * size_t(M) * size_t(w.out));
      T.ensure_kctr(E.rt);
      E.set(12, T.kctr);
      E.set(13, T.ws);
      E.rd(T.ws, 0, size_t(splits) * size_t(M) * size_t(w.out) * 4);
      E.wr(T.ws, 0, size_t(splits) * size_t(M) * size_t(w.out) * 4);
      E.rd(T.kctr, 0, size_t(nblocks) * 4);
      E.wr(T.kctr, 0, size_t(nblocks) * 4);
    }
    E.rd(A, aoff, aoff + a_bytes);
    E.wr(Y, yoff, yoff + y_bytes);
    E.groups(MTLSizeMake(size_t(nblocks), size_t(splits), 1), MTLSizeMake(k16 ? 512 : 256, 1, 1));
    return;
  }
  // 3..8 rows (draft verification): the GEMV-shaped MMA kernel.
  if (M <= 8 && T.gemv && T.gemv8) {
    GemmParams p{M, w.out, w.in, lda, ldy, 0, 0};
    const std::string name = "rgemv8r" + std::to_string(w.bits) + epis4[e] + nsuf;
    E.pipe(name, w.fc());
    gemm_tag(E, name, w.out, w.in, M);
    bind(Y, yoff, ldy);
    E.bytes(5, p);
    bind_norm();
    E.rd(A, aoff, aoff + a_bytes);
    E.wr(Y, yoff, yoff + y_bytes);
    E.groups(MTLSizeMake(size_t(w.out + 31) / 32, 1, 1), MTLSizeMake(256, 1, 1));
    return;
  }
  int choice = M <= 8 ? 0 : 2;
  auto it = T.table.find({w.out, w.in, w.bits, w.fc()});
  const int bucket = (M + 31) / 32 - 1;
  if (M > 8 && !T.fixed && it != T.table.end() && bucket < int(it->second.size())) choice = it->second[size_t(bucket)];
  else if (M > 8 && M <= 64 && !T.fixed) choice = 1;
  if (M <= 8) {
    // Decode-sized: one 8-row block per 128 columns; split K until ~256
    // threadgroups keep every core busy (memory-bound, one pass per weight).
    const int col_blocks = (w.out + 127) / 128;
    const int sp = std::min(8, std::max(1, 256 / col_blocks));
    choice = 0 | (sp << 4);
  }
  int cfg = choice & 15, splits = choice >> 4;
  if (splits < 1 || epi == 2) splits = 1;
  const GemmShape g = gemm_shape(cfg);
  // Prefill-sized GEMMs on 6-bit codes: at this many rows the GEMM is
  // bound by MMAs plus unpacking, and 6-bit codes (nibble + 2-bit planes)
  // cost ~9 ALU ops per weight pair against ~5 for bytes. Widen the matrix
  // to byte codes (same codes and scales) and run the 8-bit kernel with the
  // 6-bit matrix's tuned tile.
  QW w8;
  size_t widen_bytes = 0;
  if (M > 8 && w.bits == 6 && T.widen6_m > 0 && M >= T.widen6_m) {
    const int Np = (w.out + 31) / 32 * 32;
    widen_bytes = size_t(Np) * size_t(w.in);
    T.ensure_widen(E.rt, widen_bytes);
    w8 = w;
    w8.bits = 8;
    w8.wr = Buf(T.widen);
    w8.wh = Buf();
    E.pipe("widen6_8");
    E.set(0, w.wr);
    E.set(1, w.wh);
    E.set(2, T.widen);
    E.bytes(3, w.out);
    E.bytes(4, w.in);
    E.wr(T.widen, 0, widen_bytes);
    E.threads(MTLSizeMake(size_t(Np / 32) * size_t(w.in / 64) * 32, 1, 1), MTLSizeMake(256, 1, 1));
    wsrc = &w8;
  }
  const int wbits = wsrc->bits;
  const int KG = w.in / 64;
  const int per = (KG + splits - 1) / splits;
  splits = (KG + per - 1) / per;
  if (splits > 1) {
    T.ensure_ws(E.rt, size_t(splits) * size_t(M) * size_t(w.out));
    GemmParams p{M, w.out, w.in, lda, w.out, per, M * w.out};
    E.pipe(rgemm_name(wbits, cfg, 0), w.fc());
    gemm_tag(E, rgemm_name(wbits, cfg, 0) + "/s" + std::to_string(splits), w.out, w.in, M);
    bind(T.ws, 0, w.out);
    E.bytes(5, p);
    E.rd(A, aoff, aoff + a_bytes);
    if (widen_bytes) E.rd(T.widen, 0, widen_bytes);
    E.wr(T.ws, 0, size_t(splits) * size_t(M) * size_t(w.out) * 4);
    E.groups(MTLSizeMake(size_t(M + g.bm - 1) / size_t(g.bm), size_t(w.out + g.bn - 1) / size_t(g.bn), size_t(splits)),
             MTLSizeMake(size_t(g.threads), 1, 1));
    GemmParams r{M, w.out, w.in, lda, ldy, per, M * w.out};
    E.pipe(epi == 1 ? "splitk_reduce_add" : "splitk_reduce_store");
    E.set(0, T.ws);
    E.set(1, Y, yoff);
    E.bytes(2, r);
    E.bytes(3, splits);
    E.rd(T.ws, 0, size_t(splits) * size_t(M) * size_t(w.out) * 4);
    E.wr(Y, yoff, yoff + y_bytes);
    E.threads(MTLSizeMake(size_t(w.out + 3) / 4, size_t(M), 1), MTLSizeMake(64, 1, 1));
    return;
  }
  GemmParams p{M, w.out, w.in, lda, ldy, 0, 0};
  E.pipe(rgemm_name(wbits, cfg, epi), w.fc());
  gemm_tag(E, rgemm_name(wbits, cfg, epi), w.out, w.in, M);
  bind(Y, yoff, ldy);
  E.bytes(5, p);
  E.rd(A, aoff, aoff + a_bytes);
  if (widen_bytes) E.rd(T.widen, 0, widen_bytes);
  E.wr(Y, yoff, yoff + y_bytes);
  E.groups(MTLSizeMake(size_t(M + g.bm - 1) / size_t(g.bm), size_t(w.out + g.bn - 1) / size_t(g.bn), 1),
           MTLSizeMake(size_t(g.threads), 1, 1));
}

// Dense fp16 tile: 8 rows x 8 simdgroups for decode-sized M, otherwise
// 16 rows x 4 simdgroups (best from 13 to ~1K rows), 32x4 for wide outputs.
struct HTile {
  const char* name;
  int bm, bn;
};
// Micro benchmarks: "" = the rule below, else a tile name for M > 8.
inline std::string& hgemm_force() {
  static std::string v;
  return v;
}
inline HTile hgemm_tile(int M, int N) {
  if (M <= 8) return HTile{"8x8", 8, 256};
  if (hgemm_force() == "32x4") return HTile{"32x4", 32, 128};
  if (hgemm_force() == "16x4") return HTile{"16x4", 16, 128};
  // 32-row tiles from 256 rows (the vision tower's 3,888-row GEMMs: fc1 and fc2
  // 19-31 % faster than 16-row tiles); 16-row tiles for short prompts.
  return (N >= 8192 || M >= 256) ? HTile{"32x4", 32, 128} : HTile{"16x4", 16, 128};
}

// Dense fp16 GEMM: 0 store fp32, 1 add fp32, 2 gelu_tanh fp16, 3 gelu_erf
// fp16, 4 SwiGLU fp16, 5 GeGLU fp16; the bias is applied when the matrix has one.
inline void gemm_h(Enc& E, const HW& w, id<MTLBuffer> A, size_t aoff, int lda, id<MTLBuffer> Y, size_t yoff,
                   int ldy, int M, int epi) {
  if (M <= 0) return;
  static const char* suffix[6] = {"store", "add", "gelu_tanh", "gelu_erf", "swiglu", "geglu"};
  const HTile t = hgemm_tile(M, w.out);
  struct {
    int M, N, K, lda, ldy, has_bias;
  } p{M, w.out, w.kp, lda, ldy, w.has_bias ? 1 : 0};
  const std::string name = std::string("hgemm_") + t.name + "_" + suffix[epi];
  E.pipe(name);
  gemm_tag(E, name, w.out, w.kp, M);
  E.set(0, A, aoff);
  E.set(1, w.wr);
  if (w.has_bias) E.set(2, w.bias);
  else E.set(2, w.wr);  // unread
  E.set(4, Y, yoff);
  E.bytes(5, p);
  const size_t elt = (epi == 0 || epi == 1) ? 4 : 2;
  const int ncols = (epi == 4 || epi == 5) ? w.out / 2 : w.out;
  E.rd(A, aoff, aoff + (size_t(M - 1) * size_t(lda) + size_t(w.kp)) * 2);
  E.wr(Y, yoff, yoff + (size_t(M - 1) * size_t(ldy) + size_t(ncols)) * elt);
  E.groups(MTLSizeMake(size_t(M + t.bm - 1) / size_t(t.bm), size_t(w.out + t.bn - 1) / size_t(t.bn), 1),
           MTLSizeMake(size_t(t.bn), 1, 1));
}

// Decoder matrices: epi 0 store, 1 add, 2 SwiGLU.
inline void gemm(Enc& E, Tuner& T, const Mat& m, id<MTLBuffer> A, size_t aoff, int lda, id<MTLBuffer> Y,
                 size_t yoff, int ldy, int M, int epi, int n_cols = 0, const NormLink* nin = nullptr,
                 const NormLink* nout = nullptr) {
  if (m.dense) {
    if (nin || nout) fail("fused norm on a dense matrix");
    gemm_h(E, m.d, A, aoff, lda, Y, yoff, ldy, M, epi == 2 ? (m.d.act == 1 ? 5 : 4) : epi);
  } else {
    gemm_q(E, T, m.q, A, aoff, lda, Y, yoff, ldy, M, epi, n_cols, nin, nout);
  }
}

// A GEMV with the store epilogue and no K split over column blocks:
// every block of the first n_cols columns, or the n_list blocks in `list`
// (-1 entries skipped). 1-2 rows: rgemv; 3-8 rows: the MMA GEMV.
inline void gemv_blocks(Enc& E, const QW& w, id<MTLBuffer> A, size_t aoff, int lda, id<MTLBuffer> Y, size_t yoff,
                        int ldy, int M, int n_cols, const Buf* list = nullptr, int n_list = 0) {
  if (M < 1 || M > 8) fail("gemv_blocks: 1-8 rows");
  const int n = n_cols > 0 ? std::min(n_cols, w.out) : w.out;
  const int nblocks = list ? n_list : (n + 31) / 32;
  if (nblocks <= 0) return;
  const GemmParams p{M, w.out, w.in, lda, ldy, 0, 0};
  const std::string b = std::to_string(w.bits), l = list ? "_l" : "";
  const std::string name = M <= 2 ? "rgemv" + b + "_" + std::to_string(M) + "_store" + l : "rgemv8r" + b + "_store" + l;
  E.pipe(name, w.fc());
  gemm_tag(E, name, w.out, w.in, M);
  E.set(0, A, aoff);
  E.set(1, w.wr);
  E.set(2, w.sb);
  E.set(3, w.wh ? w.wh : w.wr);
  E.set(4, Y, yoff);
  E.bytes(5, p);
  if (list) {
    E.set(14, *list);
    E.rd(*list, size_t(n_list) * 4);
  }
  E.rd(A, aoff, aoff + (size_t(M - 1) * size_t(lda) + size_t(w.in)) * 2);
  E.wr(Y, yoff, yoff + out_span(M, ldy, w.out, 0));
  E.groups(MTLSizeMake(size_t(nblocks), 1, 1), MTLSizeMake(256, 1, 1));
}

// Column runs of one fused projection, written side by side into Y.
inline void gemm_runs(Enc& E, Tuner& T, const std::vector<Mat>& runs, id<MTLBuffer> A, size_t aoff, int lda,
                      id<MTLBuffer> Y, size_t yoff, int ldy, int M, int epi, const NormLink* nin = nullptr) {
  size_t col = 0;
  for (const Mat& w : runs) {
    gemm(E, T, w, A, aoff, lda, Y, yoff + col * sizeof(float), ldy, M, epi, 0, nin);
    col += size_t(w.out());
  }
}

// Kernels a matrix may use (for building pipelines at load).
inline void gemm_pipelines(const Mat& m, std::vector<std::pair<std::string, int>>& out, bool fma34 = false) {
  if (m.out() <= 0) return;
  if (m.dense) {
    for (const char* t : {"8x8", "16x4", "32x4"})
      for (const char* e : {"store", "add", m.d.act == 1 ? "geglu" : "swiglu"}) out.push_back({std::string("hgemm_") + t + "_" + e, 0});
    return;
  }
  const QW& w = m.q;
  for (int e = 0; e < 3; ++e)
    for (int cfg = 0; cfg < kGemmConfigs; ++cfg) {
      out.push_back({rgemm_name(w.bits, cfg, e), w.fc()});
      if (w.bits == 6) out.push_back({rgemm_name(8, cfg, e), w.fc()});  // widened prefill GEMMs
    }
  if (w.bits == 6) out.push_back({"widen6_8", 0});
  for (const char* v : {"_store", "_add", "_swiglu", "_addp", "_store_n", "_swiglu_n", "_add_g", "_addp_g"}) {
    for (int mm = 1; mm <= kGemvInstM; ++mm)
      out.push_back({"rgemv" + std::to_string(w.bits) + "_" + std::to_string(mm) + v, w.fc()});
    if (fma34)
      for (int mm = 3; mm <= 4; ++mm) out.push_back({"rgemv" + std::to_string(w.bits) + "_" + std::to_string(mm) + "j" + v, w.fc()});
    out.push_back({"rgemv8r" + std::to_string(w.bits) + v, w.fc()});
  }
}

// ---------------------------------------------------------------- autotune
inline std::string cache_dir() {
  const std::string dir = paths::user_cache_dir();
  paths::make_private_dirs(dir);  // shared with batches and checkpoints: 0700 when created
  return dir;
}

inline std::string tune_cache_path(Runtime& rt) {
  std::string name = [[rt.dev name] UTF8String];
  for (auto& ch : name)
    if (!isalnum(static_cast<unsigned char>(ch))) ch = '_';
  const char* t64 = std::getenv("ESPETERO_TILE64");
  return cache_dir() + "/gemm-tune-r9" + std::string(!(t64 && *t64 == '0') ? "t" : "") + "-" + name + ".json";
}

inline std::string tune_key(const QW* w) {
  std::string k = std::to_string(w->out) + "x" + std::to_string(w->in);
  if (w->bits != 4) k += "q" + std::to_string(w->bits);
  if (w->group != 64) k += "g" + std::to_string(w->group);
  if (w->scale_mode == 1) k += "q6k";
  else if (w->scale_mode) k += "m" + std::to_string(w->scale_mode);
  return k;
}

// Time the tile configs for every distinct GEMM shape and 32-row M bucket up
// to max_m; the fastest wins. Results are cached per device and shape.
inline void autotune(Runtime& rt, Tuner& T, const std::vector<const QW*>& all, int max_m) {
  const auto t0 = Clock::now();
  std::vector<const QW*> shapes;
  for (const QW* w : all) {
    bool seen = false;
    for (auto* s : shapes)
      seen = seen || (s->out == w->out && s->in == w->in && s->bits == w->bits && s->fc() == w->fc());
    if (!seen) shapes.push_back(w);
  }
  const int buckets = max_m / 32;
  // Every bucket up to 256 rows, every other one up to 512, every fourth
  // above; unmeasured buckets take the next measured bucket's choice.
  auto measured = [&](int b) {
    return b == buckets - 1 || b < 8 || (b < 16 ? (b + 1) % 2 == 0 : (b + 1) % 4 == 0);
  };
  // The shared-dequantization tiles tune into their own cache file, so
  // runs with ESPETERO_GEMM_CFGS=7 keep the earlier choices and runs with them
  // enabled measure them once.
  const std::string path = tune_cache_path(rt) + (T.n_cfgs > 7 ? ".s" : "");
  json::Value cache = json::Value::object();
  if (file_exists(path) && !std::getenv("ESPETERO_TUNE_FRESH")) {  // ESPETERO_TUNE_FRESH=1: re-tune everything
    try {
      cache = json::parse(read_file(path));
      if (!cache.is_object()) cache = json::Value::object();
    } catch (...) {
      cache = json::Value::object();
    }
  }
  std::vector<const QW*> todo;
  for (auto* w : shapes) {
    const json::Value* v = cache.get(tune_key(w));
    if (v && v->is_array() && int(v->size()) >= buckets) {
      std::vector<uint8_t> best;
      for (int i = 0; i < buckets; ++i) best.push_back(uint8_t((*v)[size_t(i)].as_int()));
      T.table[{w->out, w->in, w->bits, w->fc()}] = best;
    } else {
      todo.push_back(w);
    }
  }
  if (!todo.empty()) {
    size_t max_in = 0, max_out = 0;
    for (auto* w : todo) {
      max_in = std::max<size_t>(max_in, size_t(w->in));
      max_out = std::max<size_t>(max_out, size_t(w->out));
    }
    id<MTLBuffer> A = rt.buffer(size_t(max_m) * max_in * 2);
    std::memset([A contents], 0, [A length]);
    id<MTLBuffer> Y = rt.buffer(size_t(max_m) * max_out * 4);
    id<MTLBuffer> ws = rt.buffer(std::max(size_t(4) * size_t(max_m) * 2048, size_t(8) * kSplitMaxM * max_out) * 4);
    // Seconds per dispatch of one candidate (serialized: untracked command
    // buffers may overlap).
    auto time_cand = [&](const QW* w, int M, int cand, int reps) {
      const int cfg = cand & 15;
      int splits = std::max(1, cand >> 4);
      const int KG = w->in / 64;
      const int per = (KG + splits - 1) / splits;
      splits = (KG + per - 1) / per;
      if (splits > 1 && [ws length] < size_t(splits) * size_t(M) * size_t(w->out) * 4) return 1e30;
      id<MTLCommandBuffer> cb = [rt.queue commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      const GemmShape g = gemm_shape(cfg);
      for (int r = 0; r < reps; ++r) {
        [e setComputePipelineState:rt.kernel(rgemm_name(w->bits, cfg, 0), w->fc())];
        [e setBuffer:A offset:0 atIndex:0];
        [e setBuffer:w->wr.b offset:w->wr.off atIndex:1];
        [e setBuffer:w->sb.b offset:w->sb.off atIndex:2];
        const Buf& hb = w->wh ? w->wh : w->wr;
        [e setBuffer:hb.b offset:hb.off atIndex:3];
        [e setBuffer:(splits > 1 ? ws : Y) offset:0 atIndex:4];
        GemmParams p{M, w->out, w->in, w->in, w->out, splits > 1 ? per : 0, M * w->out};
        [e setBytes:&p length:sizeof(p) atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(size_t(M + g.bm - 1) / size_t(g.bm), size_t(w->out + g.bn - 1) / size_t(g.bn),
                                            size_t(splits))
          threadsPerThreadgroup:MTLSizeMake(size_t(g.threads), 1, 1)];
        if (splits > 1) {
          [e setComputePipelineState:rt.kernel("splitk_reduce_add")];
          [e setBuffer:ws offset:0 atIndex:0];
          [e setBuffer:Y offset:0 atIndex:1];
          [e setBytes:&p length:sizeof(p) atIndex:2];
          [e setBytes:&splits length:4 atIndex:3];
          [e dispatchThreads:MTLSizeMake(size_t(w->out + 3) / 4, size_t(M), 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
      }
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      return ([cb GPUEndTime] - [cb GPUStartTime]) / reps;
    };
    // The GPU clock ramps up over the first tens of ms of work after idle;
    // measure at full clock.
    {
      const QW* big = todo[0];
      for (auto* w : todo)
        if (size_t(w->out) * size_t(w->in) > size_t(big->out) * size_t(big->in)) big = w;
      for (double spent = 0; spent < 0.15;) spent += 8 * time_cand(big, std::min(max_m, 256), 2, 8);
    }
    for (auto* w : todo) {
      std::vector<uint8_t> best;
      json::Value arr = json::Value::array();
      for (int b = 0; b < buckets; ++b) {
        const int M = b * 32 + 16;  // bucket b serves M in (32b, 32b + 32]
        if (!measured(b)) {
          best.push_back(0);
          arr.push_back(0);
          continue;
        }
        double best_t = 1e30;
        int best_c = 2;
        std::vector<int> cands;
        // Taller tiles first: they dequantize every weight fewer times, which
        // a full pass rewards more than an isolated timing shows; a later
        // tile must be 3% faster to win.
        // The shared-dequantization tiles go first like the 64-row
        // tile: they dequantize every weight half as often, and a later
        // candidate must beat them by 3%.
        if (M <= 64) {
          if (T.n_cfgs > 7) cands.push_back(7);  // 32x4s
          for (int cfg : {1, 0, 3, 5, 2, 4}) cands.push_back(cfg);
        } else {
          if (T.n_cfgs > 7) cands.push_back(8), cands.push_back(7);  // 64x4s, 32x4s
          if (const char* t64 = std::getenv("ESPETERO_TILE64"); !(t64 && *t64 == '0')) cands.push_back(6);  // 64-row tile (ESPETERO_TILE64=0 disables)
          for (int cfg : {2, 4, 1, 3}) cands.push_back(cfg);
        }
        for (int cfg = 0; cfg < T.n_cfgs; ++cfg) {
          const bool narrow = w->out <= 2048 && ((cfg != 0 && cfg != 5) || M <= 64);
          const bool small = M <= kSplitMaxM && (cfg == 0 || cfg == 1 || cfg == 3);
          if (!narrow && !small) continue;
          for (int sp : {2, 4, 8})
            if (w->in / 64 >= 2 * sp) cands.push_back(cfg | (sp << 4));
        }
        std::vector<double> t(cands.size(), 1e30);
        for (int trial = 0; trial < 2; ++trial)
          for (size_t i = 0; i < cands.size(); ++i) t[i] = std::min(t[i], time_cand(w, M, cands[i], 2));
        for (size_t i = 0; i < cands.size(); ++i)
          if (t[i] < best_t * 0.97) {
            best_t = t[i];
            best_c = cands[i];
          }
        if (std::getenv("ESPETERO_TUNE_LOG")) {  // per-candidate isolated times (ms) for this matrix and M bucket
          std::fprintf(stderr, "tune %s M=%d:", tune_key(w).c_str(), M);
          for (size_t i = 0; i < cands.size(); ++i)
            std::fprintf(stderr, " %s%s=%.3f", rgemm_name(w->bits, cands[i] & 15, 0).c_str() + (w->bits == 4 ? 6 : 7),
                         (cands[i] >> 4) ? ("/s" + std::to_string(cands[i] >> 4)).c_str() : "", t[i]);
          std::fprintf(stderr, "  -> %s\n", rgemm_name(w->bits, best_c & 15, 0).c_str());
        }
        best.push_back(uint8_t(best_c));
        arr.push_back(best_c);
        for (int u = b - 1; u >= 0 && !measured(u); --u) {
          best[size_t(u)] = uint8_t(best_c);
          arr[size_t(u)] = best_c;
        }
      }
      T.table[{w->out, w->in, w->bits, w->fc()}] = best;
      cache.set(tune_key(w), std::move(arr));
    }
    const std::string tmp = path + ".tmp";
    FILE* f = paths::make_private_file(tmp) ? std::fopen(tmp.c_str(), "wb") : nullptr;
    if (f) {
      const std::string s = json::dump(cache);
      std::fwrite(s.data(), 1, s.size(), f);
      std::fclose(f);
      std::rename(tmp.c_str(), path.c_str());
    }
  }
  T.ms = ms_since(t0);
}

}  // namespace espetero::native
