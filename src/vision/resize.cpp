// SPDX-License-Identifier: MIT AND MIT-CMU
// Copyright (c) 2026 Francisco José García Barbero
//
// Portions are derived from Pillow (src/libImaging/Resample.c): Copyright (c)
// 1997-2011 by Secret Labs AB, Copyright (c) 1995-2011 by Fredrik Lundh and
// contributors, Copyright (c) 2010 by Jeffrey 'Alex' Clark and contributors;
// MIT-CMU License, see THIRD_PARTY_NOTICES.md.

// Pillow-exact bicubic resampling (port of Pillow 12 src/libImaging/Resample.c
// for 8-bit RGB with Image.BICUBIC and the default box).
//
// What has to match for byte-identical output:
//   * Coefficients: precompute_coeffs in double precision with Pillow's
//     rounding of the window bounds, normalised by their sum, then converted
//     to fixed point with PRECISION_BITS = 32 - 8 - 2 = 22 (round half away
//     from zero). The macOS arm64 Pillow wheel is compiled with FP
//     contraction: its bicubic_filter and the window centre use fmadd (checked
//     in the disassembly of _imaging.so), so we spell out the same std::fma
//     calls. (A sweep over 271k (in, out) size pairs up to 1200 found no
//     fixed-point tap that depends on it, but it keeps the doubles identical.)
//   * Passes: horizontal first (only over the source rows the vertical pass
//     reads), clipped to 8 bits, then vertical. Each output is
//     clip8((1 << 21) + sum(pixel * coeff)) with clip8(v) = clamp(v >> 22, 0, 255).
//     Integer sums are exact, so any summation order (NEON lanes) is fine.
//   * Image.resize (Python side, Pillow 12): a tall image (h > 100 * w) that
//     shrinks vertically is resized in two separate calls, vertical then
//     horizontal. reducing_gap=None means no reduce() pre-step.
//
// Speed: fixed-point NEON kernels (the horizontal pass widens each source row
// once to int32 RGBX so a tap is one load + one multiply-accumulate), and rows
// split across cores with GCD for large images.
#include "vision.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace espetero::vision {
namespace {

constexpr int kPrecisionBits = 32 - 8 - 2;
constexpr int32_t kHalf = 1 << (kPrecisionBits - 1);

inline uint8_t clip8(int32_t v) {
  v >>= kPrecisionBits;  // arithmetic shift, like Pillow's lookup index
  return uint8_t(v < 0 ? 0 : v > 255 ? 255 : v);
}

// Pillow's bicubic_filter (a = -0.5) with the fused multiply-adds of the
// shipped binary:  x<1: ((1.5x - 2.5) x) x + 1;  x<2: (((x - 5) x + 8) x - 4) * -0.5
double bicubic_filter(double x) {
  if (x < 0.0) x = -x;
  if (x < 1.0) return std::fma(x * std::fma(x, 1.5, -2.5), x, 1.0);
  if (x < 2.0) return std::fma(std::fma(x - 5.0, x, 8.0), x, -4.0) * -0.5;
  return 0.0;
}

// Fixed-point filter taps for one axis. Output i reads count[i] source
// samples starting at start[i]; coefficients are padded with zeros to
// `stride` (a multiple of 4) so NEON loops can run over whole vectors.
struct Kernel {
  int stride = 0;
  std::vector<int> start, count;
  std::vector<int32_t> k;
};

// precompute_coeffs + normalize_coeffs_8bpc for box [0, in_size].
Kernel make_kernel(int in_size, int out_size) {
  constexpr double kSupport = 2.0;  // bicubic
  const float in0 = 0.0f, in1 = float(in_size);
  const double scale = double(in1 - in0) / out_size;
  const double filterscale = scale < 1.0 ? 1.0 : scale;
  const double support = kSupport * filterscale;
  const int ksize = int(std::ceil(support)) * 2 + 1;
  const double ss = 1.0 / filterscale;

  Kernel kn;
  kn.stride = (ksize + 3) & ~3;
  kn.start.resize(out_size);
  kn.count.resize(out_size);
  kn.k.assign(size_t(out_size) * kn.stride, 0);
  std::vector<double> w(ksize);
  for (int xx = 0; xx < out_size; xx++) {
    const double center = std::fma(xx + 0.5, scale, double(in0));
    int xmin = int(center - support + 0.5);  // truncation, as the C cast
    if (xmin < 0) xmin = 0;
    int xmax = int(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    double ww = 0.0;
    for (int x = 0; x < xmax; x++) {
      const double v = bicubic_filter((double(x + xmin) - center + 0.5) * ss);
      w[x] = v;
      ww += v;
    }
    int32_t* kk = &kn.k[size_t(xx) * kn.stride];
    for (int x = 0; x < xmax; x++) {
      const double v = ww != 0.0 ? w[x] / ww : w[x];
      // (int)(±0.5 + v * 2^22): v * 2^22 is exact, so fusing cannot matter.
      kk[x] = v < 0 ? int32_t(-0.5 + v * (1 << kPrecisionBits))
                    : int32_t(0.5 + v * (1 << kPrecisionBits));
    }
    kn.start[xx] = xmin;
    kn.count[xx] = xmax;
  }
  return kn;
}

// Runs fn(begin, end) over [0, n) in chunks, in parallel when the work is
// large enough to pay for the dispatch (~10 us).
template <class Fn>
void parallel_rows(int n, double work_per_row, Fn&& fn) {
  const double total = work_per_row * n;
  int chunks = total < 2e5 ? 1 : std::min(n, std::max(1, int(total / 1e5)));
  chunks = std::min(chunks, 32);
  if (chunks <= 1) {
    fn(0, n);
    return;
  }
  struct Ctx {
    Fn* fn;
    int n, chunks;
  } ctx{&fn, n, chunks};
  dispatch_apply_f(size_t(chunks), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &ctx,
                   [](void* p, size_t i) {
                     auto* c = static_cast<Ctx*>(p);
                     const int b = int(int64_t(c->n) * int64_t(i) / c->chunks);
                     const int e = int(int64_t(c->n) * int64_t(i + 1) / c->chunks);
                     if (b < e) (*c->fn)(b, e);
                   });
}

// ---- horizontal pass: rows [r0, r1) of dst from rows r0+yoff.. of src -------

#if defined(__ARM_NEON)
// Source row -> int32 RGBX per pixel, with 4 zero pixels of padding so the
// padded tap loop may read past the row end (those taps have coefficient 0).
void widen_row(const uint8_t* src, int w, int32_t* dst) {
  int x = 0;
  const uint32x4_t zero = vdupq_n_u32(0);
  for (; x + 16 <= w; x += 16) {
    const uint8x16x3_t v = vld3q_u8(src + 3 * x);
    uint32x4_t ch[3][4];
    for (int c = 0; c < 3; c++) {
      const uint16x8_t lo = vmovl_u8(vget_low_u8(v.val[c]));
      const uint16x8_t hi = vmovl_high_u8(v.val[c]);
      ch[c][0] = vmovl_u16(vget_low_u16(lo));
      ch[c][1] = vmovl_high_u16(lo);
      ch[c][2] = vmovl_u16(vget_low_u16(hi));
      ch[c][3] = vmovl_high_u16(hi);
    }
    for (int q = 0; q < 4; q++) {
      const uint32x4x4_t o = {{ch[0][q], ch[1][q], ch[2][q], zero}};
      vst4q_u32(reinterpret_cast<uint32_t*>(dst + 4 * (x + 4 * q)), o);
    }
  }
  for (; x < w; x++) {
    dst[4 * x + 0] = src[3 * x + 0];
    dst[4 * x + 1] = src[3 * x + 1];
    dst[4 * x + 2] = src[3 * x + 2];
    dst[4 * x + 3] = 0;
  }
  std::memset(dst + 4 * w, 0, 16 * sizeof(int32_t));
}

inline uint32_t pack_pixel(int32x4_t acc) {
  const int32x4_t s = vshrq_n_s32(acc, kPrecisionBits);
  const uint16x4_t u16 = vqmovun_s32(s);                    // clamp below at 0
  const uint8x8_t u8 = vqmovn_u16(vcombine_u16(u16, u16));  // clamp above at 255
  return vget_lane_u32(vreinterpret_u32_u8(u8), 0);
}

inline void store_rgb(uint8_t* d, uint32_t v) {
  d[0] = uint8_t(v);
  d[1] = uint8_t(v >> 8);
  d[2] = uint8_t(v >> 16);
}

void horizontal_pass(const uint8_t* src, int in_w, int yoff, uint8_t* dst, int out_w,
                     const Kernel& kx, int r0, int r1) {
  // Two source rows at a time share the coefficient loads; two accumulators
  // per row break the multiply-accumulate dependency chain.
  std::unique_ptr<int32_t[]> wide(new int32_t[2 * (size_t(in_w) + 4) * 4]);
  int32_t* w0 = wide.get();
  int32_t* w1 = w0 + (size_t(in_w) + 4) * 4;
  const size_t in_stride = size_t(in_w) * 3, out_stride = size_t(out_w) * 3;
  const int32x4_t bias = vdupq_n_s32(kHalf);
  const int32x4_t zero = vdupq_n_s32(0);
  int y = r0;
  for (; y + 2 <= r1; y += 2) {
    widen_row(src + size_t(y + yoff) * in_stride, in_w, w0);
    widen_row(src + size_t(y + 1 + yoff) * in_stride, in_w, w1);
    uint8_t* d0 = dst + size_t(y) * out_stride;
    uint8_t* d1 = d0 + out_stride;
    for (int xx = 0; xx < out_w; xx++) {
      const int32_t* k = &kx.k[size_t(xx) * kx.stride];
      const int n = (kx.count[xx] + 3) & ~3;
      const int32_t* p0 = w0 + 4 * kx.start[xx];
      const int32_t* p1 = w1 + 4 * kx.start[xx];
      int32x4_t a0 = bias, b0 = zero, a1 = bias, b1 = zero;
      for (int t = 0; t < n; t += 4, p0 += 16, p1 += 16) {
        const int32x4_t kv = vld1q_s32(k + t);
        a0 = vmlaq_laneq_s32(a0, vld1q_s32(p0 + 0), kv, 0);
        b0 = vmlaq_laneq_s32(b0, vld1q_s32(p0 + 4), kv, 1);
        a1 = vmlaq_laneq_s32(a1, vld1q_s32(p1 + 0), kv, 0);
        b1 = vmlaq_laneq_s32(b1, vld1q_s32(p1 + 4), kv, 1);
        a0 = vmlaq_laneq_s32(a0, vld1q_s32(p0 + 8), kv, 2);
        b0 = vmlaq_laneq_s32(b0, vld1q_s32(p0 + 12), kv, 3);
        a1 = vmlaq_laneq_s32(a1, vld1q_s32(p1 + 8), kv, 2);
        b1 = vmlaq_laneq_s32(b1, vld1q_s32(p1 + 12), kv, 3);
      }
      store_rgb(d0 + 3 * xx, pack_pixel(vaddq_s32(a0, b0)));
      store_rgb(d1 + 3 * xx, pack_pixel(vaddq_s32(a1, b1)));
    }
  }
  for (; y < r1; y++) {
    widen_row(src + size_t(y + yoff) * in_stride, in_w, w0);
    uint8_t* d0 = dst + size_t(y) * out_stride;
    for (int xx = 0; xx < out_w; xx++) {
      const int32_t* k = &kx.k[size_t(xx) * kx.stride];
      const int n = (kx.count[xx] + 3) & ~3;
      const int32_t* p0 = w0 + 4 * kx.start[xx];
      int32x4_t a0 = bias, b0 = zero;
      for (int t = 0; t < n; t += 4, p0 += 16) {
        const int32x4_t kv = vld1q_s32(k + t);
        a0 = vmlaq_laneq_s32(a0, vld1q_s32(p0 + 0), kv, 0);
        b0 = vmlaq_laneq_s32(b0, vld1q_s32(p0 + 4), kv, 1);
        a0 = vmlaq_laneq_s32(a0, vld1q_s32(p0 + 8), kv, 2);
        b0 = vmlaq_laneq_s32(b0, vld1q_s32(p0 + 12), kv, 3);
      }
      store_rgb(d0 + 3 * xx, pack_pixel(vaddq_s32(a0, b0)));
    }
  }
}

// ---- vertical pass: output rows [r0, r1) -------------------------------------

inline void mla_u8x16(int32x4_t acc[4], uint8x16_t v, int32_t k) {
  const uint16x8_t lo = vmovl_u8(vget_low_u8(v));
  const uint16x8_t hi = vmovl_high_u8(v);
  acc[0] = vmlaq_n_s32(acc[0], vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo))), k);
  acc[1] = vmlaq_n_s32(acc[1], vreinterpretq_s32_u32(vmovl_high_u16(lo)), k);
  acc[2] = vmlaq_n_s32(acc[2], vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi))), k);
  acc[3] = vmlaq_n_s32(acc[3], vreinterpretq_s32_u32(vmovl_high_u16(hi)), k);
}

inline uint8x16_t pack16(const int32x4_t acc[4]) {
  const uint16x8_t lo = vcombine_u16(vqmovun_s32(vshrq_n_s32(acc[0], kPrecisionBits)),
                                     vqmovun_s32(vshrq_n_s32(acc[1], kPrecisionBits)));
  const uint16x8_t hi = vcombine_u16(vqmovun_s32(vshrq_n_s32(acc[2], kPrecisionBits)),
                                     vqmovun_s32(vshrq_n_s32(acc[3], kPrecisionBits)));
  return vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi));
}

void vertical_pass(const uint8_t* src, int width, uint8_t* dst, const Kernel& ky, int ybase,
                   int r0, int r1) {
  const size_t stride = size_t(width) * 3;
  const int32x4_t bias = vdupq_n_s32(kHalf);
  for (int yy = r0; yy < r1; yy++) {
    const int32_t* k = &ky.k[size_t(yy) * ky.stride];
    const int n = ky.count[yy];
    const uint8_t* s = src + size_t(ky.start[yy] - ybase) * stride;
    uint8_t* d = dst + size_t(yy) * stride;
    size_t x = 0;
    for (; x + 32 <= stride; x += 32) {
      int32x4_t a[4] = {bias, bias, bias, bias}, b[4] = {bias, bias, bias, bias};
      const uint8_t* p = s + x;
      for (int t = 0; t < n; t++, p += stride) {
        mla_u8x16(a, vld1q_u8(p), k[t]);
        mla_u8x16(b, vld1q_u8(p + 16), k[t]);
      }
      vst1q_u8(d + x, pack16(a));
      vst1q_u8(d + x + 16, pack16(b));
    }
    for (; x + 16 <= stride; x += 16) {
      int32x4_t a[4] = {bias, bias, bias, bias};
      const uint8_t* p = s + x;
      for (int t = 0; t < n; t++, p += stride) mla_u8x16(a, vld1q_u8(p), k[t]);
      vst1q_u8(d + x, pack16(a));
    }
    for (; x < stride; x++) {
      int32_t acc = kHalf;
      for (int t = 0; t < n; t++) acc += int32_t(s[x + size_t(t) * stride]) * k[t];
      d[x] = clip8(acc);
    }
  }
}

#else  // portable scalar version (same arithmetic)

void horizontal_pass(const uint8_t* src, int in_w, int yoff, uint8_t* dst, int out_w,
                     const Kernel& kx, int r0, int r1) {
  for (int y = r0; y < r1; y++) {
    const uint8_t* s = src + size_t(y + yoff) * in_w * 3;
    uint8_t* d = dst + size_t(y) * out_w * 3;
    for (int xx = 0; xx < out_w; xx++) {
      const int32_t* k = &kx.k[size_t(xx) * kx.stride];
      const uint8_t* p = s + 3 * kx.start[xx];
      int32_t a = kHalf, b = kHalf, c = kHalf;
      for (int t = 0; t < kx.count[xx]; t++) {
        a += p[3 * t] * k[t];
        b += p[3 * t + 1] * k[t];
        c += p[3 * t + 2] * k[t];
      }
      d[3 * xx] = clip8(a);
      d[3 * xx + 1] = clip8(b);
      d[3 * xx + 2] = clip8(c);
    }
  }
}

void vertical_pass(const uint8_t* src, int width, uint8_t* dst, const Kernel& ky, int ybase,
                   int r0, int r1) {
  const size_t stride = size_t(width) * 3;
  for (int yy = r0; yy < r1; yy++) {
    const int32_t* k = &ky.k[size_t(yy) * ky.stride];
    const uint8_t* s = src + size_t(ky.start[yy] - ybase) * stride;
    uint8_t* d = dst + size_t(yy) * stride;
    for (size_t x = 0; x < stride; x++) {
      int32_t acc = kHalf;
      for (int t = 0; t < ky.count[yy]; t++) acc += int32_t(s[x + size_t(t) * stride]) * k[t];
      d[x] = clip8(acc);
    }
  }
}

#endif

// ImagingResample with box = (0, 0, in.width, in.height).
RGBImage resample(const RGBImage& in, int out_w, int out_h) {
  const bool need_h = out_w != in.width;
  const bool need_v = out_h != in.height;
  if (!need_h && !need_v) return in;

  const Kernel ky = make_kernel(in.height, out_h);
  // Source rows the vertical pass reads; only those get the horizontal pass.
  const int ybox_first = ky.start[0];
  const int ybox_last = ky.start[out_h - 1] + ky.count[out_h - 1];

  RGBImage tmp;
  const RGBImage* vsrc = &in;
  int ybase = 0;
  if (need_h) {
    const Kernel kx = make_kernel(in.width, out_w);
    tmp.width = out_w;
    tmp.height = ybox_last - ybox_first;
    tmp.rgb.resize(size_t(tmp.width) * tmp.height * 3);
    parallel_rows(tmp.height, double(out_w) * kx.stride * 3, [&](int b, int e) {
      horizontal_pass(in.rgb.data(), in.width, ybox_first, tmp.rgb.data(), out_w, kx, b, e);
    });
    if (!need_v) return tmp;  // ybox covers every row when the height is unchanged
    vsrc = &tmp;
    ybase = ybox_first;
  }
  RGBImage out;
  out.width = vsrc->width;
  out.height = out_h;
  out.rgb.resize(size_t(out.width) * out_h * 3);
  parallel_rows(out_h, double(out.width) * 3 * ky.stride, [&](int b, int e) {
    vertical_pass(vsrc->rgb.data(), vsrc->width, out.rgb.data(), ky, ybase, b, e);
  });
  return out;
}

}  // namespace

RGBImage resize_bicubic(const RGBImage& in, int width, int height) {
  if (width < 1 || height < 1) throw std::invalid_argument("resize: height and width must be > 0");
  if (in.width < 1 || in.height < 1 || in.rgb.size() != size_t(in.width) * in.height * 3)
    throw std::invalid_argument("resize: malformed input image");
  if (width == in.width && height == in.height) return in;
  // Image.resize in Pillow 12: very tall images that shrink vertically are
  // resized in two calls, vertical first; the second call is a plain copy
  // (crop fast path) when the width does not change.
  if (int64_t(in.height) > int64_t(in.width) * 100 && height < in.height) {
    RGBImage mid = resample(in, in.width, height);
    return width == in.width ? mid : resample(mid, width, height);
  }
  return resample(in, width, height);
}

}  // namespace espetero::vision
