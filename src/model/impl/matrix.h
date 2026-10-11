// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Engine matrices. A quantized matrix (QW) is integer codes times a
// per-group half scale plus bias, stored for the GPU either as 4-bit nibbles
// or as bytes (any other width is widened, exactly) and repacked into
// simdgroup-fragment order; a dense matrix (HW) is fp16 in fragment order.
// Builders take format-neutral TensorViews (MLX, GGUF, dense).
#pragma once

#include "gguf.h"
#include "runtime.h"

namespace espetero::native {

struct QW {
  int out = 0, in = 0;
  int bits = 4;       // GPU code width: 4 (nibbles), 5/6 (nibble + high plane), 8 (bytes)
  int src_bits = 4;   // width of the packed codes in `w` (MLX packs 2..8; bytes = 8)
  int group = 64;     // codes per scale/bias along `in`
  bool half_scales = false;  // s/z are half (else bf16, as MLX stores them)
  // How GGUF blocks keep their own scales (bits per weight as in the file):
  // 0: affine, half scale s and bias z per group.
  // 1: Q6_K: s = int8 sub-scale per 16, z = fp16 super-scale per 256;
  //    weight = (code - 32) * super * sub.
  // 2: Q8_0: s = fp16 scale per 32, no z; weight = (code - 128) * scale.
  // 3: Q4_K/Q5_K: s = (scale, min) bytes per 32, z = fp16 (d, dmin) per 256;
  //    weight = code * d * scale - dmin * min.
  int scale_mode = 0;
  // Gated epilogue of a gate/up matrix: 0 SiLU (SwiGLU), 1 tanh-GELU
  // (GeGLU, Gemma 4). A function constant of the kernels (fc() bit 24).
  int act = 0;
  Buf w, s, z;        // host layout (row-major), released after repack
  Buf wr, sb, wh;     // GPU layout (wh: high-bit plane of 5/6-bit codes)
  size_t row_bytes() const { return size_t(in) * size_t(src_bits) / 8; }
  size_t s_row_bytes() const {
    switch (scale_mode) {
      case 1: return size_t(in / 16);
      case 2: return size_t(in / 32) * 2;
      case 3: return size_t(in / 32) * 2;
      default: return size_t(in / group) * 2;
    }
  }
  size_t z_row_bytes() const {
    switch (scale_mode) {
      case 1: return size_t(in / 256) * 2;
      case 2: return 0;
      case 3: return size_t(in / 256) * 4;
      default: return size_t(in / group) * 2;
    }
  }
  // Function-constant key of this matrix's kernels: group | scale mode << 16 |
  // gated activation << 24 (0 for every SwiGLU matrix: their keys are as before).
  int fc() const { return group | (scale_mode << 16) | (act << 24); }
};

// GGUF blocks that keep their native scales: Q6_K and Q8_0 (fewer bytes
// and at least as fast). Q4_K/Q5_K native scales save 0.375 bits per weight
// but measured 15-35% slower GEMMs than the affine expansion (the kernels are
// ALU-bound there), so they expand unless ESPETERO_KQUANT_NATIVE=1.
// ESPETERO_GGUF_AFFINE=1 / ESPETERO_Q6K_AFFINE=1 force affine (A/B tests).
int native_scale_mode(int gguf_type) {
  if (env_on("ESPETERO_GGUF_AFFINE")) return 0;
  switch (gguf_type) {
    case GGML_Q6_K: return env_on("ESPETERO_Q6K_AFFINE") ? 0 : 1;
    case GGML_Q8_0: return 2;
    case GGML_Q4_K:
    case GGML_Q5_K: return env_on("ESPETERO_KQUANT_NATIVE") ? 3 : 0;
    default: return 0;
  }
}

// GPU width for codes of `bits`: 4, 5 and 6 are native; others use bytes.
int gpu_bits_for(int bits) { return (bits == 4 || bits == 5 || bits == 6) ? bits : 8; }

struct HW {
  int out = 0, in = 0, kp = 0;  // kp: input width padded to 64
  bool has_bias = false;
  int act = 0;                  // gate/up matrices: 1 for a tanh-GELU gate (QW::act)
  Buf wr, bias;
};

// A decoder matrix: quantized or dense.
struct Mat {
  bool dense = false;
  QW q;
  HW d;
  int out() const { return dense ? d.out : q.out; }
  int in() const { return dense ? d.in : q.in; }
};

// ---------------------------------------------------------------- host builders
// Quantized view -> QW in host layout (codes, scales, biases in `ar`).
QW qw_from_view(Arena& ar, const TensorView& t, int64_t out, int64_t in) {
  if (t.rows() != out || t.cols() != in)
    fail(t.name + ": shape [" + std::to_string(t.rows()) + ", " + std::to_string(t.cols()) +
         "], expected [" + std::to_string(out) + ", " + std::to_string(in) + "]");
  QW q;
  q.out = int(out);
  q.in = int(in);
  if (t.kind == TensorView::MLXQuant) {
    q.src_bits = t.bits;
    q.bits = gpu_bits_for(t.bits);
    q.group = t.group;
    q.half_scales = t.dtype == "F16";
    const size_t groups = size_t(out) * size_t(in / t.group);
    q.w = ar.copy(t.data, q.row_bytes() * size_t(out));
    q.s = ar.copy(t.scales, groups * 2);
    q.z = ar.copy(t.biases, groups * 2);
    if (q.group == 128) {
      // Split every group in two (exact): the GEMM supports 16/32/64.
      Buf s2 = ar.alloc(groups * 4), z2 = ar.alloc(groups * 4);
      const uint16_t* ss = (const uint16_t*)ptr(q.s);
      const uint16_t* zz = (const uint16_t*)ptr(q.z);
      uint16_t* sd = (uint16_t*)ptr(s2);
      uint16_t* zd = (uint16_t*)ptr(z2);
      for (size_t i = 0; i < groups; ++i) sd[2 * i] = sd[2 * i + 1] = ss[i], zd[2 * i] = zd[2 * i + 1] = zz[i];
      q.s = s2, q.z = z2, q.group = 64;
    }
    if (q.group != 16 && q.group != 32 && q.group != 64)
      fail(t.name + ": unsupported group size " + std::to_string(q.group));
    return q;
  }
  if (t.kind != TensorView::GGUFQuant) fail(t.name + ": not a quantized tensor");
  const GGMLTypeInfo* ti = ggml_type_info(t.gguf_type);
  q.bits = ti->code_bits;
  q.src_bits = q.bits == 4 ? 4 : 8;  // host: nibbles, or one byte per code
  q.group = ti->group;
  q.half_scales = true;
  q.scale_mode = native_scale_mode(t.gguf_type);
  if (in % ti->block) fail(t.name + ": row is not a whole number of blocks");
  const size_t groups_per_row = size_t(in / q.group);
  const size_t code_bytes = q.row_bytes();
  q.w = ar.alloc(code_bytes * size_t(out));
  q.s = ar.alloc(q.s_row_bytes() * size_t(out));
  q.z = ar.alloc(q.z_row_bytes() * size_t(out));
  uint8_t* codes = (uint8_t*)ptr(q.w);
  uint8_t* scales = (uint8_t*)ptr(q.s);
  uint8_t* biases = (uint8_t*)ptr(q.z);
  const size_t srow = q.s_row_bytes(), zrow = q.z_row_bytes();
  const int mode = q.scale_mode;
  const size_t src_row = size_t(in / ti->block) * size_t(ti->bytes);
  const uint8_t* src = t.data;
  const int type = t.gguf_type, bits = q.src_bits;
  prefetch_mapped(src, src_row * size_t(out));  // rows below fault no pages in one by one
  parallel_for(size_t(out), [&](size_t r) {
    RowSink sink{bits, codes + r * code_bytes, (uint16_t*)(scales + r * srow),
                 mode == 2 ? nullptr : (uint16_t*)(biases + r * zrow)};
    if (mode == 1) {
      sink.q6k_sub = (int8_t*)(scales + r * srow);
      sink.q6k_super = (uint16_t*)(biases + r * zrow);
    } else if (mode == 3) {
      sink.kq_sm = scales + r * srow;
      sink.kq_dd = (uint16_t*)(biases + r * zrow);
    }
    std::memset(sink.codes, 0, code_bytes);
    convert_row(type, src + r * src_row, in, sink);
  });
  (void)groups_per_row;
  return q;
}

// Native GGUF scales -> affine half scale and bias per group (exactly what
// convert_row writes otherwise): Q6_K per 16 (s = d * sub, b = -32 * s), Q8_0
// per 32 (b = -128 * s), Q4_K/Q5_K per 32 (s = d * scale, b = -dmin * min).
QW to_affine(Arena& ar, const QW& q) {
  if (q.scale_mode == 0) return q;
  QW o = q;
  o.scale_mode = 0;
  const int mode = q.scale_mode;
  const int gpr = q.in / (mode == 1 ? 16 : 32), bpr = q.in / 256;
  o.group = mode == 1 ? 16 : 32;
  o.s = ar.alloc(size_t(q.out) * gpr * 2);
  o.z = ar.alloc(size_t(q.out) * gpr * 2);
  const uint8_t* ss = (const uint8_t*)ptr(q.s);
  const uint16_t* zz = mode == 2 ? nullptr : (const uint16_t*)ptr(q.z);
  uint16_t* sd = (uint16_t*)ptr(o.s);
  uint16_t* zd = (uint16_t*)ptr(o.z);
  parallel_for(size_t(q.out), [&](size_t r) {
    for (int g = 0; g < gpr; ++g) {
      float sc, b;
      if (mode == 1) {
        sc = f16_to_f32(zz[r * bpr + g / 16]) * float(((const int8_t*)ss)[r * gpr + g]);
        b = -32.0f * sc;
      } else if (mode == 2) {
        sc = f16_to_f32(((const uint16_t*)ss)[r * gpr + g]);
        b = -128.0f * sc;
      } else {
        const uint8_t* sm = ss + 2 * (r * gpr + g);
        sc = f16_to_f32(zz[2 * (r * bpr + g / 8)]) * float(sm[0]);
        b = -f16_to_f32(zz[2 * (r * bpr + g / 8) + 1]) * float(sm[1]);
      }
      sd[r * gpr + g] = f32_to_f16_bits(sc);
      zd[r * gpr + g] = f32_to_f16_bits(b);
    }
  });
  return o;
}

// Dense view [out, in] -> fp16 host copy (optionally with a column permutation).
Buf f16_from_view(Arena& ar, const TensorView& t, int64_t out, int64_t in,
                  const std::vector<int>* perm = nullptr) {
  if (t.kind != TensorView::Dense) fail(t.name + ": expected a dense tensor");
  if (t.numel() != out * in) fail(t.name + ": unexpected size");
  Buf host = ar.alloc(size_t(out) * size_t(in) * 2);
  __fp16* dst = (__fp16*)ptr(host);
  const uint8_t* src = t.data;
  const std::string dtype = t.dtype;
  if (dtype != "F32" && dtype != "F16" && dtype != "BF16") fail(t.name + ": unsupported dtype " + dtype);
  const int cols = int(in);
  parallel_for(size_t(out), [&](size_t r) {
    for (int c = 0; c < cols; ++c) {
      const int sc = perm ? (*perm)[c] : c;
      dst[r * size_t(cols) + c] = __fp16(dense_at(src, dtype, r * size_t(cols) + sc));
    }
  });
  return host;
}

// Unpack packed codes of any width to one byte per code (exact); the GPU
// width stays `gpu_bits` (default: bytes).
QW widen8(Arena& ar, const QW& q_in, int gpu_bits = 8) {
  const QW q = to_affine(ar, q_in);
  if (q.src_bits == 8) {
    QW o = q;
    o.bits = gpu_bits;
    return o;
  }
  QW o = q;
  o.src_bits = 8;
  o.bits = gpu_bits;
  o.w = ar.alloc(size_t(q.out) * size_t(q.in));
  const size_t rb = q.row_bytes();
  const uint8_t* src = (const uint8_t*)ptr(q.w);
  uint8_t* dst = (uint8_t*)ptr(o.w);
  const int bits = q.src_bits, in = q.in;
  const uint32_t mask = (1u << bits) - 1;
  parallel_for(size_t(q.out), [&](size_t r) {
    const uint8_t* p = src + r * rb;
    uint8_t* d = dst + r * size_t(in);
    uint32_t acc = 0;
    int nb = 0;
    size_t bi = 0;
    for (int i = 0; i < in; ++i) {
      while (nb < bits) {
        acc |= uint32_t(p[bi++]) << nb;
        nb += 8;
      }
      d[i] = uint8_t(acc & mask);
      acc >>= bits;
      nb -= bits;
    }
  });
  return o;
}

// Scales and biases as half (converting MLX's bf16).
QW to_half_scales(Arena& ar, const QW& q) {
  if (q.half_scales || q.scale_mode) return q;
  QW o = q;
  const size_t n = size_t(q.out) * size_t(q.in / q.group);
  o.s = ar.alloc(n * 2);
  o.z = ar.alloc(n * 2);
  const uint16_t* ss = (const uint16_t*)ptr(q.s);
  const uint16_t* zz = (const uint16_t*)ptr(q.z);
  uint16_t* sd = (uint16_t*)ptr(o.s);
  uint16_t* zd = (uint16_t*)ptr(o.z);
  for (size_t i = 0; i < n; ++i) {
    sd[i] = f32_to_f16_bits(bf16_to_f32(ss[i]));
    zd[i] = f32_to_f16_bits(bf16_to_f32(zz[i]));
  }
  o.half_scales = true;
  return o;
}

// Split each group into groups of `g` (exact: the scale and bias repeat).
QW regroup(Arena& ar, const QW& q_in, int g) {
  if (q_in.group == g) return q_in;
  const QW q = to_affine(ar, q_in);
  if (q.group % g) fail("regroup: incompatible group sizes");
  const int k = q.group / g;
  QW o = q;
  const size_t n = size_t(q.out) * size_t(q.in / q.group);
  o.s = ar.alloc(n * k * 2);
  o.z = ar.alloc(n * k * 2);
  const uint16_t* ss = (const uint16_t*)ptr(q.s);
  const uint16_t* zz = (const uint16_t*)ptr(q.z);
  uint16_t* sd = (uint16_t*)ptr(o.s);
  uint16_t* zd = (uint16_t*)ptr(o.z);
  for (size_t i = 0; i < n; ++i)
    for (int j = 0; j < k; ++j) sd[i * k + j] = ss[i], zd[i * k + j] = zz[i];
  o.group = g;
  return o;
}

// Bring parts to one GPU layout: the widest GPU width (codes widened to
// bytes on the host when layouts differ), smallest group, and scale type.
std::vector<QW> unify(Arena& ar, std::vector<QW> parts) {
  bool same_bits = true, same_scales = true, same_native = parts[0].scale_mode != 0;
  int group = parts[0].group, target = 0;
  for (const QW& p : parts) {
    same_bits = same_bits && p.src_bits == parts[0].src_bits && p.bits == parts[0].bits;
    same_scales = same_scales && p.half_scales == parts[0].half_scales;
    same_native = same_native && p.scale_mode == parts[0].scale_mode && p.group == parts[0].group;
    group = std::min(group, p.group);
    target = std::max(target, p.bits);
  }
  if (same_native && same_bits) return parts;  // one native GGUF layout
  for (QW& p : parts) p = to_affine(ar, p);
  for (const QW& p : parts) group = std::min(group, p.group);
  for (QW& p : parts) {
    if (!same_bits) p = widen8(ar, p, target);
    if (!same_scales) p = to_half_scales(ar, p);
    p = regroup(ar, p, group);
  }
  return parts;
}

// Concatenate same-layout parts along output rows.
QW concat_rows(Arena& ar, const std::vector<QW>& parts) {
  QW out = parts[0];
  out.out = 0;
  for (const auto& p : parts) {
    if (p.in != out.in || p.src_bits != out.src_bits || p.group != out.group ||
        p.half_scales != out.half_scales || p.scale_mode != out.scale_mode)
      fail("concat_rows: mismatched parts");
    out.out += p.out;
  }
  const size_t wrow = out.row_bytes(), srow = out.s_row_bytes(), zrow = out.z_row_bytes();
  out.w = ar.alloc(wrow * out.out);
  out.s = ar.alloc(srow * out.out);
  out.z = ar.alloc(zrow * out.out);
  size_t r = 0;
  for (const auto& p : parts) {
    std::memcpy((uint8_t*)ptr(out.w) + r * wrow, ptr(p.w), wrow * p.out);
    std::memcpy((uint8_t*)ptr(out.s) + r * srow, ptr(p.s), srow * p.out);
    if (zrow) std::memcpy((uint8_t*)ptr(out.z) + r * zrow, ptr(p.z), zrow * p.out);
    r += size_t(p.out);
  }
  return out;
}

// Consecutive runs of one GPU code width, each fused into one matrix.
std::vector<QW> fuse_rows(Arena& ar, const std::vector<QW>& parts) {
  std::vector<QW> runs, cur;
  auto flush = [&]() {
    if (cur.empty()) return;
    cur = unify(ar, cur);
    runs.push_back(cur.size() == 1 ? cur[0] : concat_rows(ar, cur));
    cur.clear();
  };
  for (const auto& p : parts) {
    if (!cur.empty() && cur.back().bits != p.bits) flush();
    cur.push_back(p);
  }
  flush();
  return runs;
}

// Interleave two equally shaped matrices in row blocks of 8: a0..a7 b0..b7 ...
QW interleave8(Arena& ar, const QW& a_in, const QW& b_in) {
  if (a_in.out != b_in.out || a_in.in != b_in.in || a_in.out % 8) fail("interleave8: shape mismatch");
  std::vector<QW> ab = unify(ar, {a_in, b_in});
  const QW& a = ab[0];
  const QW& b = ab[1];
  QW out = a;
  out.out = 2 * a.out;
  const size_t wrow = out.row_bytes(), srow = out.s_row_bytes(), zrow = out.z_row_bytes();
  out.w = ar.alloc(wrow * out.out);
  out.s = ar.alloc(srow * out.out);
  out.z = ar.alloc(zrow * out.out);
  // Blocks of 8 rows, 64 blocks per task (on several threads: load time).
  const int blocks = a.out / 8;
  parallel_for(size_t((blocks + 63) / 64), [&](size_t t) {
    for (int blk = int(t) * 64; blk < std::min(blocks, int(t) * 64 + 64); ++blk) {
      for (int h = 0; h < 2; ++h) {
        const QW& src = h ? b : a;
        const size_t dst_row = size_t(blk) * 16 + h * 8, src_row = size_t(blk) * 8;
        std::memcpy((uint8_t*)ptr(out.w) + dst_row * wrow, (uint8_t*)ptr(src.w) + src_row * wrow, wrow * 8);
        std::memcpy((uint8_t*)ptr(out.s) + dst_row * srow, (uint8_t*)ptr(src.s) + src_row * srow, srow * 8);
        if (zrow) std::memcpy((uint8_t*)ptr(out.z) + dst_row * zrow, (uint8_t*)ptr(src.z) + src_row * zrow, zrow * 8);
      }
    }
  });
  return out;
}

// Row interleave of two dense fp16 host matrices (gate/up), blocks of 8.
Buf interleave8_f16(Arena& ar, const Buf& a, const Buf& b, int out, int in) {
  if (out % 8) fail("interleave8_f16: rows not a multiple of 8");
  Buf o = ar.alloc(size_t(out) * 2 * size_t(in) * 2);
  const size_t row = size_t(in) * 2;
  for (int blk = 0; blk < out / 8; ++blk)
    for (int h = 0; h < 2; ++h)
      std::memcpy((uint8_t*)ptr(o) + (size_t(blk) * 16 + h * 8) * row,
                  (uint8_t*)ptr(h ? b : a) + size_t(blk) * 8 * row, row * 8);
  return o;
}

// ---------------------------------------------------------------- GPU repack
// Encodes into `e`; the host buffers (and `tmp`) must live until it completes.
// GPU buffers come from `dst` (default: the runtime arena).
void repack(Runtime& rt, id<MTLComputeCommandEncoder> e, QW& q, Arena& tmp, Arena* dst = nullptr) {
  Arena& ga = dst ? *dst : *rt.arena;
  const int Np = (q.out + 31) / 32 * 32;
  const int KG = q.in / 64;
  if (q.in % 64) fail("quantized matrices need an input width that is a multiple of 64");
  if (q.scale_mode == 1 && q.bits != 6) fail("repack: native Q6_K scales need 6-bit codes");
  if ((q.scale_mode == 2 && q.bits != 8) || (q.scale_mode == 3 && q.bits != 4 && q.bits != 5))
    fail("repack: native GGUF scales with an unexpected code width");
  // Per column pair: Q6_K a uint2 of int8 sub-scales per K step then half2
  // super-scales per 256; Q8_0 a half2 per 32; Q4_K/Q5_K a uint2 of (scale,
  // min) bytes per K step then half4 (d, dmin) per 256; affine half4 per group.
  const size_t pairs = size_t(Np) / 2;
  switch (q.scale_mode) {
    case 1: q.sb = ga.alloc(pairs * KG * 8 + pairs * (q.in / 256) * 4); break;
    case 2: q.sb = ga.alloc(pairs * (q.in / 32) * 4); break;
    case 3: q.sb = ga.alloc(pairs * KG * 8 + pairs * (q.in / 256) * 8); break;
    default: q.sb = ga.alloc(pairs * size_t(q.in / q.group) * 8);
  }
  if (q.bits == 5 || q.bits == 6) {
    Buf codes = q.w;
    if (q.src_bits != 8) {
      codes = tmp.alloc(size_t(q.out) * q.in);
      [e setComputePipelineState:rt.kernel("unpack_bits")];
      [e setBuffer:q.w.b offset:q.w.off atIndex:0];
      [e setBuffer:codes.b offset:codes.off atIndex:1];
      [e setBytes:&q.in length:4 atIndex:2];
      [e setBytes:&q.src_bits length:4 atIndex:3];
      [e dispatchThreads:MTLSizeMake(q.in, q.out, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    q.wr = ga.alloc(size_t(Np) * q.in / 2);
    q.wh = ga.alloc(size_t(Np) * q.in * (q.bits - 4) / 8);
    [e setComputePipelineState:rt.kernel("repack_planes")];
    [e setBuffer:codes.b offset:codes.off atIndex:0];
    [e setBuffer:q.wr.b offset:q.wr.off atIndex:1];
    [e setBuffer:q.wh.b offset:q.wh.off atIndex:2];
    [e setBytes:&q.out length:4 atIndex:3];
    [e setBytes:&q.in length:4 atIndex:4];
    [e setBytes:&q.bits length:4 atIndex:5];
    [e dispatchThreads:MTLSizeMake(size_t(Np / 32) * KG * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  } else if (q.bits == 2) {
    // The coarse MTP draft head: packed 2-bit codes, one uint4 per lane and K step.
    if (q.src_bits != 2) fail("repack: 2-bit layout expects packed 2-bit codes");
    q.wr = ga.alloc(size_t(Np) * q.in / 4);
    [e setComputePipelineState:rt.kernel("repack_weights2")];
    [e setBuffer:q.w.b offset:q.w.off atIndex:0];
    [e setBuffer:q.wr.b offset:q.wr.off atIndex:1];
    [e setBytes:&q.out length:4 atIndex:2];
    [e setBytes:&q.in length:4 atIndex:3];
    [e dispatchThreads:MTLSizeMake(size_t(Np / 32) * KG * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  } else if (q.bits == 4) {
    if (q.src_bits != 4) fail("repack: 4-bit layout expects nibble codes");
    q.wr = ga.alloc(size_t(Np) * q.in / 8 * 4);
    [e setComputePipelineState:rt.kernel("repack_weights")];
    [e setBuffer:q.w.b offset:q.w.off atIndex:0];
  } else {
    Buf codes = q.w;
    if (q.src_bits != 8) {
      codes = tmp.alloc(size_t(q.out) * q.in);
      [e setComputePipelineState:rt.kernel("unpack_bits")];
      [e setBuffer:q.w.b offset:q.w.off atIndex:0];
      [e setBuffer:codes.b offset:codes.off atIndex:1];
      [e setBytes:&q.in length:4 atIndex:2];
      [e setBytes:&q.src_bits length:4 atIndex:3];
      [e dispatchThreads:MTLSizeMake(q.in, q.out, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    q.wr = ga.alloc(size_t(Np) * q.in);
    [e setComputePipelineState:rt.kernel("repack_weights8")];
    [e setBuffer:codes.b offset:codes.off atIndex:0];
  }
  if (q.bits == 4 || q.bits == 8) {
    [e setBuffer:q.wr.b offset:q.wr.off atIndex:1];
    [e setBytes:&q.out length:4 atIndex:2];
    [e setBytes:&q.in length:4 atIndex:3];
    [e dispatchThreads:MTLSizeMake(size_t(Np / 32) * KG * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  if (q.scale_mode == 1 || q.scale_mode == 3) {
    [e setComputePipelineState:rt.kernel(q.scale_mode == 1 ? "repack_q6k_scales" : "repack_kq_scales")];
    [e setBuffer:q.s.b offset:q.s.off atIndex:0];
    [e setBuffer:q.z.b offset:q.z.off atIndex:1];
    [e setBuffer:q.sb.b offset:q.sb.off atIndex:2];
    [e setBytes:&q.out length:4 atIndex:3];
    [e setBytes:&q.in length:4 atIndex:4];
    [e dispatchThreads:MTLSizeMake(Np / 2, KG, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return;
  }
  if (q.scale_mode == 2) {
    [e setComputePipelineState:rt.kernel("repack_q8_scales")];
    [e setBuffer:q.s.b offset:q.s.off atIndex:0];
    [e setBuffer:q.sb.b offset:q.sb.off atIndex:1];
    [e setBytes:&q.out length:4 atIndex:2];
    [e setBytes:&q.in length:4 atIndex:3];
    [e dispatchThreads:MTLSizeMake(Np / 2, q.in / 32, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return;
  }
  const int half_src = q.half_scales ? 1 : 0;
  [e setComputePipelineState:rt.kernel("repack_scales")];
  [e setBuffer:q.s.b offset:q.s.off atIndex:0];
  [e setBuffer:q.z.b offset:q.z.off atIndex:1];
  [e setBuffer:q.sb.b offset:q.sb.off atIndex:2];
  [e setBytes:&q.out length:4 atIndex:3];
  [e setBytes:&q.in length:4 atIndex:4];
  [e setBytes:&q.group length:4 atIndex:5];
  [e setBytes:&half_src length:4 atIndex:6];
  [e dispatchThreads:MTLSizeMake(Np / 2, q.in / q.group, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

// fp16 host matrix [out, in] -> fragment order (+ optional fp32 bias).
HW repack_dense(Runtime& rt, id<MTLComputeCommandEncoder> e, const Buf& host, int out, int in,
                const std::vector<float>* bias = nullptr) {
  HW h;
  h.out = out;
  h.in = in;
  h.kp = (in + 63) / 64 * 64;
  const int Np = (out + 31) / 32 * 32;
  h.wr = rt.arena->alloc(size_t(Np) * h.kp * 2);
  [e setComputePipelineState:rt.kernel("repack_f16")];
  [e setBuffer:host.b offset:host.off atIndex:0];
  [e setBuffer:h.wr.b offset:h.wr.off atIndex:1];
  [e setBytes:&out length:4 atIndex:2];
  [e setBytes:&in length:4 atIndex:3];
  [e setBytes:&h.kp length:4 atIndex:4];
  [e dispatchThreads:MTLSizeMake(size_t(Np / 32) * (h.kp / 64) * 32, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  if (bias) {
    if (bias->size() != size_t(out)) fail("dense bias size mismatch");
    h.bias = rt.arena->copy(bias->data(), bias->size() * 4);
    h.has_bias = true;
  }
  return h;
}

}  // namespace espetero::native
