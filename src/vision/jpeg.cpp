// SPDX-License-Identifier: MIT AND IJG AND Zlib
// Copyright (c) 2026 Francisco José García Barbero
//
// Portions are derived from libjpeg-turbo 3.1.0 and remain under its terms:
// the IJG License (third_party/libjpeg-turbo/README.ijg) and, for the Arm Neon
// colour formulas, the zlib License. This software is based in part on the
// work of the Independent JPEG Group. The changes are described in
// THIRD_PARTY_NOTICES.md. Notices of the original files:
//   jidctint.c: Copyright (C) 1991-1998, Thomas G. Lane. Modification developed
//     2002-2018 by Guido Vollbeding. libjpeg-turbo Modifications: Copyright (C)
//     2015, 2020, 2022, D. R. Commander.
//   jdhuff.c: Copyright (C) 1991-1997, Thomas G. Lane. Lossless JPEG
//     Modifications: Copyright (C) 1999, Ken Murchison. libjpeg-turbo
//     Modifications: Copyright (C) 2009-2011, 2016, 2018-2019, 2022,
//     D. R. Commander. Copyright (C) 2018, Matthias Räncker.
//   jdphuff.c: Copyright (C) 1995-1997, Thomas G. Lane. Lossless JPEG
//     Modifications: Copyright (C) 1999, Ken Murchison. libjpeg-turbo
//     Modifications: Copyright (C) 2015-2016, 2018-2022, D. R. Commander.
//   jdsample.c: Copyright (C) 1991-1996, Thomas G. Lane. libjpeg-turbo
//     Modifications: Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio
//     AB. Copyright (C) 2010, 2015-2016, 2022, 2024, D. R. Commander. Copyright
//     (C) 2014, MIPS Technologies, Inc., California. Copyright (C) 2015, Google,
//     Inc. Copyright (C) 2019-2020, Arm Limited.
//   jdcolor.c: Copyright (C) 1991-1997, Thomas G. Lane. Modified 2011 by Guido
//     Vollbeding. libjpeg-turbo Modifications: Copyright 2009 Pierre Ossman
//     <ossman@cendio.se> for Cendio AB. Copyright (C) 2009, 2011-2012,
//     2014-2015, 2022, 2024, D. R. Commander. Copyright (C) 2013, Linaro Limited.
//   jdmarker.c: Copyright (C) 1991-1998, Thomas G. Lane. Lossless JPEG
//     Modifications: Copyright (C) 1999, Ken Murchison. libjpeg-turbo
//     Modifications: Copyright (C) 2012, 2015, 2022, 2024, D. R. Commander.
//   simd/arm/jdcolext-neon.c (zlib License): Copyright (C) 2020, Arm Limited.
//     Copyright (C) 2020, D. R. Commander.

// JPEG decoder that reproduces PIL 12.3 (libjpeg-turbo 3.1, default settings)
// byte for byte: Image.open(jpeg).convert("RGB").
//
// ImageIO's JPEG decoders differ from libjpeg-turbo by up to ~100 levels
// (different IDCT, chroma upsampling and colour conversion; the hardware
// decoder it picks for large images is further off), so baseline, extended
// and progressive Huffman JPEGs (what cameras, browsers and PIL write) are
// decoded here with libjpeg-turbo's exact arithmetic:
//   * entropy decoding is lossless, so any correct decoder yields the same
//     coefficients (sequential and progressive, restart markers);
//   * jpeg_idct_islow (CONST_BITS 13, PASS1_BITS 2); the Neon version PIL
//     runs is documented as bit-identical to the C code;
//   * jdsample.c upsampling as jinit_upsampler selects it with
//     do_fancy_upsampling = TRUE: h2v1 / h2v2 "fancy" triangle filters (box
//     when the chroma plane is at most 2 samples wide), h1v2 fancy, box
//     replication for other integral ratios; context rows at the image edges
//     repeat the first / last real row as jdmainct.c arranges it;
//   * jdcolor.c YCbCr->RGB (SCALEBITS 16 tables; the Neon constants used
//     below give identical results for all 8-bit inputs, checked
//     exhaustively), colour space guessed like default_decompress_parms;
//   * PIL's side: 1 component -> L, 3 -> RGB, 4 -> "CMYK;I" (inverted) and
//     its own CMYK->RGB conversion.
// Not handled (ImageIO fallback, see decode.cpp): arithmetic coding,
// lossless / hierarchical / 12-bit files, fractional sampling ratios,
// progressive files libjpeg would render with inter-block smoothing
// (coefficients never refined to full precision) and corrupt entropy data.
//
// Speed: 64-bit bit buffer with a combined code+magnitude AC lookup; restart
// intervals (Apple and most cameras emit one per MCU row) are entropy-decoded
// in parallel; single-scan files reconstruct (IDCT, upsampling, colour) in
// bands on other cores while the entropy decoder is still running; IDCT,
// upsampling and colour conversion use Neon.
#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "decode_internal.h"

namespace espetero::vision::detail {
namespace {

// Zigzag -> natural order, with libjpeg's 16 guard entries for corrupt runs.
constexpr uint8_t kNatural[64 + 16] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33,
    40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54,
    47, 55, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63};

struct Error {
  const char* what;
  bool unsupported;
};

[[noreturn]] void fail(const char* what) { throw Error{what, false}; }
[[noreturn]] void unsupported(const char* what) { throw Error{what, true}; }

#define JPEG_INLINE inline __attribute__((always_inline))

// ---- Huffman tables (jpeg_make_d_derived_tbl) ---------------------------------

constexpr int kLookBits = 9;
constexpr int kAcBits = 11;

inline int extend(int v, int n) { return v < (1 << (n - 1)) ? v - (1 << n) + 1 : v; }

struct Huffman {
  bool defined = false;
  // fast[peek] = (length << 8) | symbol for codes of <= kLookBits bits, 0 otherwise.
  uint16_t fast[1 << kLookBits];
  // AC tables: code + magnitude bits resolved in one lookup when they fit in
  // kAcBits: (value << 16) | (run << 8) | total bits; 0 otherwise.
  int32_t fast_ac[1 << kAcBits];
  int32_t maxcode[18];
  int32_t valoffset[18];
  uint8_t huffval[256];

  void build(const uint8_t counts[17], const uint8_t* vals, int nvals, bool is_dc) {
    int huffsize[257], huffcode[257];
    int p = 0;
    for (int l = 1; l <= 16; l++) {
      if (p + counts[l] > 256) fail("bad Huffman table");
      for (int i = 0; i < counts[l]; i++) huffsize[p++] = l;
    }
    huffsize[p] = 0;
    if (p != nvals) fail("bad Huffman table");
    int code = 0, si = huffsize[0];
    p = 0;
    while (huffsize[p]) {
      while (huffsize[p] == si) {
        huffcode[p++] = code;
        code++;
      }
      if (code >= (1 << si)) fail("bad Huffman table");  // all-ones codes are not allowed
      code <<= 1;
      si++;
    }
    p = 0;
    for (int l = 1; l <= 16; l++) {
      if (counts[l]) {
        valoffset[l] = p - huffcode[p];
        p += counts[l];
        maxcode[l] = huffcode[p - 1];
      } else {
        maxcode[l] = -1;
      }
    }
    maxcode[17] = 0x7fffffff;
    valoffset[17] = 0;
    std::memset(huffval, 0, sizeof huffval);
    std::memcpy(huffval, vals, size_t(nvals));
    if (is_dc)
      for (int i = 0; i < nvals; i++)
        if (vals[i] > 15) fail("bad DC Huffman table");
    std::memset(fast, 0, sizeof fast);
    p = 0;
    for (int l = 1; l <= kLookBits; l++) {
      for (int i = 0; i < counts[l]; i++, p++) {
        const int lookbits = huffcode[p] << (kLookBits - l);
        for (int ctr = 1 << (kLookBits - l); ctr > 0; ctr--)
          fast[lookbits + ctr - 1] = uint16_t(l << 8 | vals[p]);
      }
    }
    std::memset(fast_ac, 0, sizeof fast_ac);
    if (!is_dc) {
      p = 0;
      for (int l = 1; l <= kAcBits; l++) {
        for (int i = 0; i < counts[l]; i++, p++) {
          const int r = vals[p] >> 4, sz = vals[p] & 15;
          if (sz == 0 || l + sz > kAcBits) continue;  // EOB / ZRL / long: slow path
          const int base = huffcode[p] << (kAcBits - l);
          for (int ctr = 0; ctr < 1 << (kAcBits - l); ctr++) {
            const int mag = ctr >> (kAcBits - l - sz);
            fast_ac[base + ctr] = int32_t(uint32_t(extend(mag, sz)) << 16) | r << 8 | (l + sz);
          }
        }
      }
    }
    defined = true;
  }
};

// ---- entropy-coded segment reader (jdhuff.c fill_bit_buffer semantics) ----------

// Everything is force-inlined into the scan loops so the bit buffer lives in
// registers (the reader is a local that never escapes).
class BitReader {
 public:
  BitReader(const uint8_t* p, const uint8_t* end) : p_(p), end_(end) {}

  const uint8_t* pos() const { return p_; }
  int marker() const { return marker_; }  // >0: marker that ended the segment; -1: end of data

  // Guarantees 32 buffered bits (a code plus its magnitude bits).
  JPEG_INLINE void ensure32() {
    if (nbits_ < 32) fill();
  }
  JPEG_INLINE uint32_t look(int n) const { return uint32_t(buf_ >> (64 - n)); }
  JPEG_INLINE void drop(int n) {
    buf_ <<= n;
    nbits_ -= n;
    real_ -= n;
  }
  // Symbol from `h` (caller ensured 32 bits).
  JPEG_INLINE int decode(const Huffman& h) {
    const uint32_t lk = look(16);
    const uint16_t e = h.fast[lk >> (16 - kLookBits)];
    if (__builtin_expect(e != 0, 1)) {
      drop(e >> 8);
      return e & 0xff;
    }
    for (int l = kLookBits + 1; l <= 16; l++) {
      const int32_t code = int32_t(lk >> (16 - l));
      if (code <= h.maxcode[l]) {
        drop(l);
        return h.huffval[(code + h.valoffset[l]) & 0xff];
      }
    }
    fail("corrupt JPEG data: bad Huffman code");
  }
  JPEG_INLINE int bits(int n) {  // n <= 16, caller ensured the bits
    if (n == 0) return 0;
    const int v = int(look(n));
    drop(n);
    return v;
  }
  JPEG_INLINE int extend_bits(int n) { return n == 0 ? 0 : extend(bits(n), n); }
  JPEG_INLINE int bit() {  // one bit with its own refill check
    if (nbits_ < 1) fill();
    return bits(1);
  }
  // A symbol consumed the zero bits libjpeg stuffs after a marker.
  JPEG_INLINE bool overrun() const { return real_ < 0; }

 private:
  static JPEG_INLINE bool has_ff(uint64_t v) {
    const uint64_t x = ~v;  // 0xFF bytes become 0x00
    return ((x - 0x0101010101010101ull) & ~x & 0x8080808080808080ull) != 0;
  }
  JPEG_INLINE void fill() {
    // Fast path: the next 8 bytes hold no 0xFF, append as many as fit.
    if (__builtin_expect(marker_ == 0 && end_ - p_ >= 8, 1)) {
      uint64_t v;
      std::memcpy(&v, p_, 8);
      if (__builtin_expect(!has_ff(v), 1)) {
        const int k = (64 - nbits_) >> 3;
        const int total = nbits_ + 8 * k;
        uint64_t add = __builtin_bswap64(v) >> nbits_;
        if (total < 64) add &= ~(~0ull >> total);
        buf_ |= add;
        real_ += total - nbits_;
        nbits_ = total;
        p_ += k;
        return;
      }
    }
    while (nbits_ <= 56) {
      if (marker_ != 0) {  // past the end of the segment libjpeg feeds zero bits
        nbits_ += 8;
        continue;
      }
      if (p_ >= end_) {
        marker_ = -1;
        continue;
      }
      const uint32_t c = *p_++;
      if (c == 0xFF) {
        // FF 00 is a data byte, FF FF.. fill, FF xx a marker (left unread).
        const uint8_t* q = p_;
        while (q < end_ && *q == 0xFF) q++;
        if (q >= end_) {
          p_ = end_;
          marker_ = -1;
          continue;
        }
        if (*q != 0) {
          marker_ = *q;
          p_ = q - 1;
          continue;
        }
        p_ = q + 1;
      }
      buf_ |= uint64_t(c) << (56 - nbits_);
      nbits_ += 8;
      real_ += 8;
    }
  }

  const uint8_t* p_;
  const uint8_t* end_;
  uint64_t buf_ = 0;  // left-aligned bit buffer
  int nbits_ = 0;     // valid bits in buf_ (real + stuffed zeros)
  int real_ = 0;      // of which came from the data
  int marker_ = 0;
};

// ---- IDCT -------------------------------------------------------------------------

constexpr int kConstBits = 13, kPass1Bits = 2;
constexpr int32_t FIX_0_298631336 = 2446, FIX_0_390180644 = 3196, FIX_0_541196100 = 4433,
                  FIX_0_765366865 = 6270, FIX_0_899976223 = 7373, FIX_1_175875602 = 9633,
                  FIX_1_501321110 = 12299, FIX_1_847759065 = 15137, FIX_1_961570560 = 16069,
                  FIX_2_053119869 = 16819, FIX_2_562915447 = 20995, FIX_3_072711026 = 25172;

// jpeg_idct_islow (jidctint.c) with dequantisation. For valid data every
// intermediate fits in 32 bits, where this matches libjpeg's 64-bit C code and
// its Neon version; corrupt coefficients give garbage either way, but wrapping
// lanes / 64-bit scalars keep it defined. The zero-AC shortcuts of the C code
// are exact rewrites of the full computation, so skipping them changes nothing.
#if defined(__ARM_NEON)

struct IdctIn {
  int32x4_t v[8];
};

template <int Shift>
inline void idct_1d(const IdctIn& in, int32x4_t out[8]) {
  const int32x4_t z2e = in.v[2], z3e = in.v[6];
  const int32x4_t z1e = vmulq_n_s32(vaddq_s32(z2e, z3e), FIX_0_541196100);
  const int32x4_t tmp2e = vmlaq_n_s32(z1e, z3e, -FIX_1_847759065);
  const int32x4_t tmp3e = vmlaq_n_s32(z1e, z2e, FIX_0_765366865);
  const int32x4_t tmp0e = vshlq_n_s32(vaddq_s32(in.v[0], in.v[4]), kConstBits);
  const int32x4_t tmp1e = vshlq_n_s32(vsubq_s32(in.v[0], in.v[4]), kConstBits);
  const int32x4_t tmp10 = vaddq_s32(tmp0e, tmp3e), tmp13 = vsubq_s32(tmp0e, tmp3e);
  const int32x4_t tmp11 = vaddq_s32(tmp1e, tmp2e), tmp12 = vsubq_s32(tmp1e, tmp2e);

  int32x4_t tmp0 = in.v[7], tmp1 = in.v[5], tmp2 = in.v[3], tmp3 = in.v[1];
  int32x4_t z1 = vaddq_s32(tmp0, tmp3), z2 = vaddq_s32(tmp1, tmp2);
  int32x4_t z3 = vaddq_s32(tmp0, tmp2), z4 = vaddq_s32(tmp1, tmp3);
  const int32x4_t z5 = vmulq_n_s32(vaddq_s32(z3, z4), FIX_1_175875602);
  tmp0 = vmulq_n_s32(tmp0, FIX_0_298631336);
  tmp1 = vmulq_n_s32(tmp1, FIX_2_053119869);
  tmp2 = vmulq_n_s32(tmp2, FIX_3_072711026);
  tmp3 = vmulq_n_s32(tmp3, FIX_1_501321110);
  z1 = vmulq_n_s32(z1, -FIX_0_899976223);
  z2 = vmulq_n_s32(z2, -FIX_2_562915447);
  z3 = vmlaq_n_s32(z5, z3, -FIX_1_961570560);  // z3 * c + z5
  z4 = vmlaq_n_s32(z5, z4, -FIX_0_390180644);
  tmp0 = vaddq_s32(tmp0, vaddq_s32(z1, z3));
  tmp1 = vaddq_s32(tmp1, vaddq_s32(z2, z4));
  tmp2 = vaddq_s32(tmp2, vaddq_s32(z2, z3));
  tmp3 = vaddq_s32(tmp3, vaddq_s32(z1, z4));
  // DESCALE(x, n) = (x + 2^(n-1)) >> n: a rounding shift.
  out[0] = vrshrq_n_s32(vaddq_s32(tmp10, tmp3), Shift);
  out[7] = vrshrq_n_s32(vsubq_s32(tmp10, tmp3), Shift);
  out[1] = vrshrq_n_s32(vaddq_s32(tmp11, tmp2), Shift);
  out[6] = vrshrq_n_s32(vsubq_s32(tmp11, tmp2), Shift);
  out[2] = vrshrq_n_s32(vaddq_s32(tmp12, tmp1), Shift);
  out[5] = vrshrq_n_s32(vsubq_s32(tmp12, tmp1), Shift);
  out[3] = vrshrq_n_s32(vaddq_s32(tmp13, tmp0), Shift);
  out[4] = vrshrq_n_s32(vsubq_s32(tmp13, tmp0), Shift);
}

inline void transpose4(int32x4_t& a, int32x4_t& b, int32x4_t& c, int32x4_t& d) {
  const int32x4x2_t ab = vtrnq_s32(a, b), cd = vtrnq_s32(c, d);
  a = vcombine_s32(vget_low_s32(ab.val[0]), vget_low_s32(cd.val[0]));
  b = vcombine_s32(vget_low_s32(ab.val[1]), vget_low_s32(cd.val[1]));
  c = vcombine_s32(vget_high_s32(ab.val[0]), vget_high_s32(cd.val[0]));
  d = vcombine_s32(vget_high_s32(ab.val[1]), vget_high_s32(cd.val[1]));
}

// clamp(x + 128) to 0..255: the range-limit table for in-range values, and
// what libjpeg-turbo's Neon IDCT (saturating narrow) does beyond it.
inline uint8x8_t limit8(int32x4_t lo, int32x4_t hi) {
  const int16x8_t v = vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi));
  return vqmovun_s16(vaddq_s16(v, vdupq_n_s16(128)));
}

void idct_islow(const int16_t* in, const int16_t* q, uint8_t* out, size_t stride) {
  int16x8_t acc = vsetq_lane_s16(0, vld1q_s16(in), 0);  // row 0 without the DC
  for (int r = 1; r < 8; r++) acc = vorrq_s16(acc, vld1q_s16(in + 8 * r));
  if (vmaxvq_u16(vreinterpretq_u16_s16(acc)) == 0) {  // flat DC block
    const int64_t dc = int64_t(in[0]) * q[0] * (1 << kPass1Bits);  // no overflow on garbage
    const int64_t v = (dc + (1 << (kPass1Bits + 2))) >> (kPass1Bits + 3);
    const uint8_t px = uint8_t(std::clamp<int64_t>(v + 128, 0, 255));
    for (int r = 0; r < 8; r++) std::memset(out + size_t(r) * stride, px, 8);
    return;
  }
  // Pass 1: columns; lanes are 4 columns, vectors are rows.
  int32x4_t ws[2][8];
  for (int g = 0; g < 2; g++) {
    IdctIn col;
    for (int r = 0; r < 8; r++)
      col.v[r] = vmull_s16(vld1_s16(in + 8 * r + 4 * g), vld1_s16(q + 8 * r + 4 * g));
    idct_1d<kConstBits - kPass1Bits>(col, ws[g]);
  }
  for (int g = 0; g < 2; g++)
    for (int h = 0; h < 2; h++) transpose4(ws[g][4 * h], ws[g][4 * h + 1], ws[g][4 * h + 2], ws[g][4 * h + 3]);
  // Pass 2: rows; ws[g][4h + i] now holds column 4g + i of rows 4h..4h+3.
  for (int h = 0; h < 2; h++) {
    IdctIn row;
    for (int c = 0; c < 8; c++) row.v[c] = ws[c >> 2][4 * h + (c & 3)];
    int32x4_t o[8];
    idct_1d<kConstBits + kPass1Bits + 3>(row, o);
    transpose4(o[0], o[1], o[2], o[3]);
    transpose4(o[4], o[5], o[6], o[7]);
    for (int i = 0; i < 4; i++) vst1_u8(out + size_t(4 * h + i) * stride, limit8(o[i], o[4 + i]));
  }
}

#else

inline int64_t descale(int64_t x, int n) { return (x + (int64_t(1) << (n - 1))) >> n; }
inline uint8_t limit(int64_t x) { return uint8_t(std::clamp<int64_t>(x + 128, 0, 255)); }

void idct_islow(const int16_t* in, const int16_t* q, uint8_t* out, size_t stride) {
  int64_t ws[64];
  for (int c = 0; c < 8; c++) {
    const int16_t* ip = in + c;
    const int16_t* qp = q + c;
    int64_t z2 = int64_t(ip[16]) * qp[16], z3 = int64_t(ip[48]) * qp[48];
    int64_t z1 = (z2 + z3) * FIX_0_541196100;
    int64_t tmp2 = z1 + z3 * -FIX_1_847759065, tmp3 = z1 + z2 * FIX_0_765366865;
    z2 = int64_t(ip[0]) * qp[0];
    z3 = int64_t(ip[32]) * qp[32];
    int64_t tmp0 = (z2 + z3) * (1 << kConstBits), tmp1 = (z2 - z3) * (1 << kConstBits);
    const int64_t tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
    tmp0 = int64_t(ip[56]) * qp[56];
    tmp1 = int64_t(ip[40]) * qp[40];
    tmp2 = int64_t(ip[24]) * qp[24];
    tmp3 = int64_t(ip[8]) * qp[8];
    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    int64_t z4 = tmp1 + tmp3;
    const int64_t z5 = (z3 + z4) * FIX_1_175875602;
    tmp0 *= FIX_0_298631336;
    tmp1 *= FIX_2_053119869;
    tmp2 *= FIX_3_072711026;
    tmp3 *= FIX_1_501321110;
    z1 *= -FIX_0_899976223;
    z2 *= -FIX_2_562915447;
    z3 = z3 * -FIX_1_961570560 + z5;
    z4 = z4 * -FIX_0_390180644 + z5;
    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;
    constexpr int d = kConstBits - kPass1Bits;
    int64_t* wp = ws + c;
    wp[0] = descale(tmp10 + tmp3, d);
    wp[56] = descale(tmp10 - tmp3, d);
    wp[8] = descale(tmp11 + tmp2, d);
    wp[48] = descale(tmp11 - tmp2, d);
    wp[16] = descale(tmp12 + tmp1, d);
    wp[40] = descale(tmp12 - tmp1, d);
    wp[24] = descale(tmp13 + tmp0, d);
    wp[32] = descale(tmp13 - tmp0, d);
  }
  for (int r = 0; r < 8; r++) {
    const int64_t* wp = ws + 8 * r;
    uint8_t* op = out + size_t(r) * stride;
    int64_t z2 = wp[2], z3 = wp[6];
    int64_t z1 = (z2 + z3) * FIX_0_541196100;
    int64_t tmp2 = z1 + z3 * -FIX_1_847759065, tmp3 = z1 + z2 * FIX_0_765366865;
    int64_t tmp0 = (wp[0] + wp[4]) * (1 << kConstBits), tmp1 = (wp[0] - wp[4]) * (1 << kConstBits);
    const int64_t tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
    tmp0 = wp[7];
    tmp1 = wp[5];
    tmp2 = wp[3];
    tmp3 = wp[1];
    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    int64_t z4 = tmp1 + tmp3;
    const int64_t z5 = (z3 + z4) * FIX_1_175875602;
    tmp0 *= FIX_0_298631336;
    tmp1 *= FIX_2_053119869;
    tmp2 *= FIX_3_072711026;
    tmp3 *= FIX_1_501321110;
    z1 *= -FIX_0_899976223;
    z2 *= -FIX_2_562915447;
    z3 = z3 * -FIX_1_961570560 + z5;
    z4 = z4 * -FIX_0_390180644 + z5;
    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;
    constexpr int d = kConstBits + kPass1Bits + 3;
    op[0] = limit(descale(tmp10 + tmp3, d));
    op[7] = limit(descale(tmp10 - tmp3, d));
    op[1] = limit(descale(tmp11 + tmp2, d));
    op[6] = limit(descale(tmp11 - tmp2, d));
    op[2] = limit(descale(tmp12 + tmp1, d));
    op[5] = limit(descale(tmp12 - tmp1, d));
    op[3] = limit(descale(tmp13 + tmp0, d));
    op[4] = limit(descale(tmp13 - tmp0, d));
  }
}

#endif

// ---- upsampling (jdsample.c) ----------------------------------------------------------
//
// In every fancy filter the special first / last columns equal the general
// formula with the edge sample repeated, which the vector loops rely on.

// h2v1_fancy_upsample: out[2i] = (3 in[i] + in[i-1] + 1) >> 2, out[2i+1] = (3 in[i] + in[i+1] + 2) >> 2.
void h2v1_fancy(const uint8_t* in, int dw, uint8_t* out) {
  auto at = [&](int i) { return int(in[std::clamp(i, 0, dw - 1)]); };
  int i = 0;
  auto scalar = [&](int n) {
    for (; i < n; i++) {
      const int v = in[i] * 3;
      out[2 * i] = uint8_t((v + at(i - 1) + 1) >> 2);
      out[2 * i + 1] = uint8_t((v + at(i + 1) + 2) >> 2);
    }
  };
  scalar(std::min(1, dw));
#if defined(__ARM_NEON)
  for (; i + 9 <= dw; i += 8) {  // reads in[i-1 .. i+8]
    const uint16x8_t cur3 = vmull_u8(vld1_u8(in + i), vdup_n_u8(3));
    const uint8x8_t even = vshrn_n_u16(vaddq_u16(vaddw_u8(cur3, vld1_u8(in + i - 1)), vdupq_n_u16(1)), 2);
    const uint8x8_t odd = vshrn_n_u16(vaddq_u16(vaddw_u8(cur3, vld1_u8(in + i + 1)), vdupq_n_u16(2)), 2);
    const uint8x8x2_t pair = {{even, odd}};
    vst2_u8(out + 2 * i, pair);
  }
#endif
  scalar(dw);
}

// h2v2_fancy_upsample for one output row: `near` is the nearest input row,
// `far` the next nearest (above for even output rows, below for odd ones).
void h2v2_fancy(const uint8_t* near, const uint8_t* far, int dw, uint8_t* out) {
  auto sum = [&](int i) {
    i = std::clamp(i, 0, dw - 1);
    return near[i] * 3 + far[i];
  };
  int i = 0;
  auto scalar = [&](int n) {
    for (; i < n; i++) {
      const int t = sum(i) * 3;
      out[2 * i] = uint8_t((t + sum(i - 1) + 8) >> 4);
      out[2 * i + 1] = uint8_t((t + sum(i + 1) + 7) >> 4);
    }
  };
  scalar(std::min(1, dw));
#if defined(__ARM_NEON)
  const uint8x8_t three = vdup_n_u8(3);
  for (; i + 9 <= dw; i += 8) {
    const uint16x8_t prev = vmlal_u8(vmovl_u8(vld1_u8(far + i - 1)), vld1_u8(near + i - 1), three);
    const uint16x8_t cur = vmlal_u8(vmovl_u8(vld1_u8(far + i)), vld1_u8(near + i), three);
    const uint16x8_t next = vmlal_u8(vmovl_u8(vld1_u8(far + i + 1)), vld1_u8(near + i + 1), three);
    const uint16x8_t cur3 = vmulq_n_u16(cur, 3);
    const uint8x8_t even = vshrn_n_u16(vaddq_u16(vaddq_u16(cur3, prev), vdupq_n_u16(8)), 4);
    const uint8x8_t odd = vshrn_n_u16(vaddq_u16(vaddq_u16(cur3, next), vdupq_n_u16(7)), 4);
    const uint8x8x2_t pair = {{even, odd}};
    vst2_u8(out + 2 * i, pair);
  }
#endif
  scalar(dw);
}

// h1v2_fancy_upsample: bias 1 when the far row is above, 2 when below.
void h1v2_fancy(const uint8_t* near, const uint8_t* far, int n, int bias, uint8_t* out) {
  int x = 0;
#if defined(__ARM_NEON)
  const uint16x8_t b = vdupq_n_u16(uint16_t(bias));
  for (; x + 16 <= n; x += 16) {
    const uint8x16_t a = vld1q_u8(near + x), f = vld1q_u8(far + x);
    const uint16x8_t lo = vaddq_u16(vmlal_u8(vmovl_u8(vget_low_u8(f)), vget_low_u8(a), vdup_n_u8(3)), b);
    const uint16x8_t hi = vaddq_u16(vmlal_high_u8(vmovl_high_u8(f), a, vdupq_n_u8(3)), b);
    vst1q_u8(out + x, vcombine_u8(vshrn_n_u16(lo, 2), vshrn_n_u16(hi, 2)));
  }
#endif
  for (; x < n; x++) out[x] = uint8_t((near[x] * 3 + far[x] + bias) >> 2);
}

// ---- colour conversion (jdcolor.c) ---------------------------------------------------

// build_ycc_rgb_table (SCALEBITS 16) for the scalar paths.
struct YccTables {
  int cr_r[256], cb_b[256];
  int32_t cr_g[256], cb_g[256];
  YccTables() {
    constexpr int32_t kHalf = 1 << 15;
    auto fix = [](double x) { return int32_t(x * 65536.0 + 0.5); };
    for (int i = 0; i < 256; i++) {
      const int32_t x = i - 128;
      cr_r[i] = int((fix(1.40200) * x + kHalf) >> 16);
      cb_b[i] = int((fix(1.77200) * x + kHalf) >> 16);
      cr_g[i] = -fix(0.71414) * x;
      cb_g[i] = -fix(0.34414) * x + kHalf;
    }
  }
};
const YccTables kYcc;

inline uint8_t clamp255(int v) { return uint8_t(v < 0 ? 0 : v > 255 ? 255 : v); }

void ycc_to_rgb(const uint8_t* y, const uint8_t* cb, const uint8_t* cr, int n, uint8_t* out) {
  int x = 0;
#if defined(__ARM_NEON)
  // libjpeg-turbo's Neon formulas (jdcolext-neon.c): R-Y = qrdmulh(2 cr', 22971),
  // B-Y = qrdmulh(2 cb', 29033), G-Y = (-11277 cb' - 23401 cr' + 2^14) >> 15.
  // Identical to the C tables for every (cb, cr).
  const uint8x8_t c128 = vdup_n_u8(128);
  for (; x + 8 <= n; x += 8) {
    const uint8x8_t yy = vld1_u8(y + x);
    const int16x8_t cbs = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(cb + x), c128));
    const int16x8_t crs = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(cr + x), c128));
    const int16x8_t r_y = vqrdmulhq_n_s16(vshlq_n_s16(crs, 1), 22971);
    const int16x8_t b_y = vqrdmulhq_n_s16(vshlq_n_s16(cbs, 1), 29033);
    int32x4_t gl = vmull_n_s16(vget_low_s16(cbs), -11277);
    int32x4_t gh = vmull_n_s16(vget_high_s16(cbs), -11277);
    gl = vmlal_n_s16(gl, vget_low_s16(crs), -23401);
    gh = vmlal_n_s16(gh, vget_high_s16(crs), -23401);
    const int16x8_t g_y = vcombine_s16(vrshrn_n_s32(gl, 15), vrshrn_n_s32(gh, 15));
    const int16x8_t y16 = vreinterpretq_s16_u16(vmovl_u8(yy));
    uint8x8x3_t rgb;
    rgb.val[0] = vqmovun_s16(vaddq_s16(y16, r_y));
    rgb.val[1] = vqmovun_s16(vaddq_s16(y16, g_y));
    rgb.val[2] = vqmovun_s16(vaddq_s16(y16, b_y));
    vst3_u8(out + 3 * x, rgb);
  }
#endif
  for (; x < n; x++) {
    const int yy = y[x], b = cb[x], r = cr[x];
    out[3 * x] = clamp255(yy + kYcc.cr_r[r]);
    out[3 * x + 1] = clamp255(yy + int((kYcc.cb_g[b] + kYcc.cr_g[r]) >> 16));
    out[3 * x + 2] = clamp255(yy + kYcc.cb_b[b]);
  }
}

void planes_to_rgb(const uint8_t* a, const uint8_t* b, const uint8_t* c, int n, uint8_t* out) {
  int x = 0;
#if defined(__ARM_NEON)
  for (; x + 16 <= n; x += 16) {
    const uint8x16x3_t v = {{vld1q_u8(a + x), vld1q_u8(b + x), vld1q_u8(c + x)}};
    vst3q_u8(out + 3 * x, v);
  }
#endif
  for (; x < n; x++) {
    out[3 * x] = a[x];
    out[3 * x + 1] = b[x];
    out[3 * x + 2] = c[x];
  }
}

// ---- frame / component state -------------------------------------------------------

struct Component {
  int id = 0, h = 1, v = 1, tq = 0;
  int16_t quant[64];  // latched at the component's first scan (ISLOW_MULT_TYPE is short)
  bool latched = false;
  int width_in_blocks = 0, height_in_blocks = 0;  // real blocks
  int blocks_w = 0, blocks_h = 0;                 // storage (MCU-padded)
  int dw = 0, dh = 0;                             // downsampled size in samples
  std::vector<int16_t> coef;                      // blocks_w * blocks_h * 64, natural order
  int coef_bits[64];                              // progressive: last Al per coefficient, -1 unseen
  int dc_tbl = 0, ac_tbl = 0;
  int16_t* block(int bx, int by) { return &coef[(size_t(by) * blocks_w + bx) * 64]; }
};

enum class Upsample { Full, H2V1Fancy, H2V1Box, H1V2Fancy, H2V2Fancy, H2V2Box, Integral };

struct Plane {  // one component after the IDCT
  std::vector<uint8_t> px;
  int stride = 0;
  Upsample method = Upsample::Full;
  int hx = 1, vx = 1;  // integral expansion factors
};

// Entropy decoder state that restarts at each restart marker.
struct SegState {
  int dc_pred[4] = {0, 0, 0, 0};
  int eobrun = 0;
};

class Decoder {
 public:
  Decoder(const uint8_t* data, size_t n) : data_(data), end_(data + n) {}
  void run(RGBImage& out);

 private:
  enum ScanKind { kSequential, kDcFirst, kDcRefine, kAcFirst, kAcRefine };
  enum ColorSpace { kGray, kYCbCr, kRGB, kCMYK, kYCCK };
  class Pipeline;

  // markers
  int next_marker();
  uint32_t u16() {
    if (end_ - p_ < 2) fail("truncated JPEG");
    const uint32_t v = uint32_t(p_[0]) << 8 | p_[1];
    p_ += 2;
    return v;
  }
  void read_app(int marker, const uint8_t* seg, size_t len);
  void read_dqt(const uint8_t* seg, size_t len);
  void read_dht(const uint8_t* seg, size_t len);
  void read_sof(int marker, const uint8_t* seg, size_t len);
  void read_sos(const uint8_t* seg, size_t len);

  // entropy decoding
  void decode_scan();
  template <ScanKind K>
  JPEG_INLINE void decode_block(BitReader& br, SegState& st, int sc, Component& c, int16_t* blk);
  template <ScanKind K>
  JPEG_INLINE void decode_mcu(BitReader& br, SegState& st, int mx, int my);
  template <ScanKind K>
  void scan_sequential(Pipeline* pipe);
  template <ScanKind K>
  bool scan_parallel();
  bool restart(const uint8_t* pos, int marker);
  int scan_mcus_x() const { return scan_n_ == 1 ? comp_[scan_comp_[0]].width_in_blocks : mcus_x_; }
  int scan_mcus_y() const { return scan_n_ == 1 ? comp_[scan_comp_[0]].height_in_blocks : mcus_y_; }

  // reconstruction
  void check_smoothing();
  void setup_output(RGBImage& out);
  void idct_mcu_rows(int a, int b);
  void color_rows(int y0, int y1);
  int out_rows_of_mcu_row(int m) const { return std::min(m * 8 * max_v_, height_); }

  const uint8_t* data_;
  const uint8_t* end_;
  const uint8_t* p_ = nullptr;

  Huffman dc_[4], ac_[4];
  uint16_t qt_[4][64];
  bool qt_defined_[4] = {false, false, false, false};
  int restart_interval_ = 0;
  bool jfif_ = false, adobe_ = false;
  int adobe_transform_ = 0;

  bool frame_ = false, progressive_ = false;
  int width_ = 0, height_ = 0, ncomp_ = 0, max_h_ = 1, max_v_ = 1;
  int mcus_x_ = 0, mcus_y_ = 0;
  Component comp_[4];

  // current scan
  int scan_n_ = 0, scan_comp_[4] = {0, 0, 0, 0};
  int ss_ = 0, se_ = 63, ah_ = 0, al_ = 0;
  int next_restart_ = 0;
  int scans_ = 0;

  // reconstruction
  RGBImage* out_ = nullptr;
  ColorSpace cs_ = kYCbCr;
  Plane planes_[4];
  bool reconstructed_ = false;  // done by the pipeline during the scan
  bool single_scan_image_ = false;
};

// Returns the next marker code, skipping fill bytes and (like libjpeg, with a
// warning) extraneous data. 0 at the end of data.
int Decoder::next_marker() {
  for (;;) {
    while (p_ < end_ && *p_ != 0xFF) p_++;
    while (p_ < end_ && *p_ == 0xFF) p_++;
    if (p_ >= end_) return 0;
    const int m = *p_++;
    if (m != 0) return m;
  }
}

void Decoder::read_app(int marker, const uint8_t* s, size_t len) {
  // examine_app0 / examine_app14 (jdmarker.c): at most the first 14 bytes matter.
  if (marker == 0xE0 && len >= 14 && std::memcmp(s, "JFIF\0", 5) == 0) jfif_ = true;
  if (marker == 0xEE && len >= 12 && std::memcmp(s, "Adobe", 5) == 0) {
    adobe_ = true;
    adobe_transform_ = s[11];
  }
}

void Decoder::read_dqt(const uint8_t* s, size_t len) {
  size_t i = 0;
  while (i < len) {
    const int pq = s[i] >> 4, tq = s[i] & 15;
    i++;
    if (tq > 3 || pq > 1) fail("bad DQT");
    const size_t need = pq ? 128 : 64;
    if (i + need > len) fail("bad DQT length");
    for (int k = 0; k < 64; k++) {
      const uint16_t v = pq ? uint16_t(s[i + 2 * k] << 8 | s[i + 2 * k + 1]) : s[i + k];
      qt_[tq][kNatural[k]] = v;
    }
    qt_defined_[tq] = true;
    i += need;
  }
}

void Decoder::read_dht(const uint8_t* s, size_t len) {
  size_t i = 0;
  while (i < len) {
    if (i + 17 > len) fail("bad DHT length");
    const int tc = s[i] >> 4, th = s[i] & 15;
    if (tc > 1 || th > 3) fail("bad DHT");
    uint8_t counts[17] = {0};
    int total = 0;
    for (int l = 1; l <= 16; l++) {
      counts[l] = s[i + l];
      total += counts[l];
    }
    i += 17;
    if (total > 256 || i + size_t(total) > len) fail("bad DHT length");
    (tc ? ac_[th] : dc_[th]).build(counts, s + i, total, tc == 0);
    i += size_t(total);
  }
}

void Decoder::read_sof(int marker, const uint8_t* s, size_t len) {
  if (frame_) fail("duplicate SOF");
  if (marker != 0xC0 && marker != 0xC1 && marker != 0xC2)
    unsupported("JPEG process not supported here (arithmetic / lossless / hierarchical)");
  progressive_ = marker == 0xC2;
  if (len < 6) fail("bad SOF");
  if (s[0] != 8) unsupported("only 8-bit JPEG");
  height_ = s[1] << 8 | s[2];
  width_ = s[3] << 8 | s[4];
  ncomp_ = s[5];
  if (height_ == 0) unsupported("DNL-sized JPEG");
  if (width_ == 0) fail("bad JPEG size");
  if (width_ > 65500 || height_ > 65500 || over_pixel_cap(width_, height_)) fail("JPEG too large");
  if (ncomp_ != 1 && ncomp_ != 3 && ncomp_ != 4) unsupported("PIL only reads 1, 3 or 4 components");
  if (len < size_t(6 + 3 * ncomp_)) fail("bad SOF length");
  for (int i = 0; i < ncomp_; i++) {
    Component& c = comp_[i];
    c.id = s[6 + 3 * i];
    c.h = s[7 + 3 * i] >> 4;
    c.v = s[7 + 3 * i] & 15;
    c.tq = s[8 + 3 * i];
    if (c.h < 1 || c.h > 4 || c.v < 1 || c.v > 4 || c.tq > 3) fail("bad sampling factors");
    max_h_ = std::max(max_h_, c.h);
    max_v_ = std::max(max_v_, c.v);
  }
  // A lone component's scans are non-interleaved (one block per MCU), so its
  // declared sampling factors change nothing: use 1x1 so that "MCU rows" are
  // block rows everywhere.
  if (ncomp_ == 1) comp_[0].h = comp_[0].v = max_h_ = max_v_ = 1;
  mcus_x_ = (width_ + 8 * max_h_ - 1) / (8 * max_h_);
  mcus_y_ = (height_ + 8 * max_v_ - 1) / (8 * max_v_);
  for (int i = 0; i < ncomp_; i++) {
    Component& c = comp_[i];
    c.width_in_blocks = int((int64_t(width_) * c.h + 8 * max_h_ - 1) / (8 * max_h_));
    c.height_in_blocks = int((int64_t(height_) * c.v + 8 * max_v_ - 1) / (8 * max_v_));
    c.dw = int((int64_t(width_) * c.h + max_h_ - 1) / max_h_);
    c.dh = int((int64_t(height_) * c.v + max_v_ - 1) / max_v_);
    c.blocks_w = mcus_x_ * c.h;
    c.blocks_h = mcus_y_ * c.v;
    c.coef.assign(size_t(c.blocks_w) * c.blocks_h * 64, 0);
    std::fill(c.coef_bits, c.coef_bits + 64, -1);
  }
  frame_ = true;
}

void Decoder::read_sos(const uint8_t* s, size_t len) {
  if (!frame_) fail("SOS before SOF");
  if (len < 1) fail("bad SOS");
  scan_n_ = s[0];
  if (scan_n_ < 1 || scan_n_ > ncomp_ || len < size_t(4 + 2 * scan_n_)) fail("bad SOS");
  for (int i = 0; i < scan_n_; i++) {
    const int id = s[1 + 2 * i], tables = s[2 + 2 * i];
    int ci = -1;
    for (int k = 0; k < ncomp_; k++)
      if (comp_[k].id == id) ci = k;
    if (ci < 0) fail("SOS names an unknown component");
    for (int k = 0; k < i; k++)
      if (scan_comp_[k] == ci) fail("duplicate component in scan");
    scan_comp_[i] = ci;
    comp_[ci].dc_tbl = tables >> 4;
    comp_[ci].ac_tbl = tables & 15;
    if (comp_[ci].dc_tbl > 3 || comp_[ci].ac_tbl > 3) fail("bad table index");
  }
  const uint8_t* t = s + 1 + 2 * scan_n_;
  ss_ = t[0];
  se_ = t[1];
  ah_ = t[2] >> 4;
  al_ = t[2] & 15;
  if (progressive_) {
    if (ss_ == 0 ? se_ != 0 : (se_ < ss_ || se_ > 63 || scan_n_ != 1)) fail("bad progressive scan");
    if (ah_ > 13 || al_ > 13) fail("bad progressive scan");
  } else {
    // libjpeg only warns (JWRN_NOT_SEQUENTIAL) and decodes the full band.
    ss_ = 0;
    se_ = 63;
    ah_ = al_ = 0;
  }
  // Latch quantisation tables at each component's first scan (latch_quant_tables).
  for (int i = 0; i < scan_n_; i++) {
    Component& c = comp_[scan_comp_[i]];
    if (c.latched) continue;
    if (!qt_defined_[c.tq]) fail("missing quantisation table");
    for (int k = 0; k < 64; k++) c.quant[k] = int16_t(qt_[c.tq][k]);
    c.latched = true;
  }
  for (int i = 0; i < scan_n_; i++) {
    const Component& c = comp_[scan_comp_[i]];
    const bool need_dc = !progressive_ || (ss_ == 0 && ah_ == 0);
    const bool need_ac = !progressive_ || ss_ > 0;
    if (need_dc && !dc_[c.dc_tbl].defined) fail("missing DC Huffman table");
    if (need_ac && !ac_[c.ac_tbl].defined) fail("missing AC Huffman table");
  }
  // Progressive bookkeeping for the block-smoothing decision (coef_bits).
  if (progressive_)
    for (int i = 0; i < scan_n_; i++) {
      Component& c = comp_[scan_comp_[i]];
      for (int k = ss_; k <= se_; k++) c.coef_bits[k] = al_;
    }
}

// ---- entropy decoding ---------------------------------------------------------------

JPEG_INLINE int add_dc(int pred, int diff) {
  int v;
  if (__builtin_add_overflow(pred, diff, &v)) fail("corrupt JPEG data: DC coefficient overflow");
  return v;
}

// One block of the current scan (jdhuff.c decode_mcu, jdphuff.c decode_mcu_*);
// `sc` is the component's index within the scan (its DC predictor).
template <Decoder::ScanKind K>
JPEG_INLINE void Decoder::decode_block(BitReader& br, SegState& st, int sc, Component& c, int16_t* blk) {
  if constexpr (K == kSequential) {
    br.ensure32();
    st.dc_pred[sc] = add_dc(st.dc_pred[sc], br.extend_bits(br.decode(dc_[c.dc_tbl])));
    blk[0] = int16_t(st.dc_pred[sc]);
    const Huffman& ac = ac_[c.ac_tbl];
    for (int k = 1; k < 64; k++) {
      br.ensure32();
      const int32_t e = ac.fast_ac[br.look(kAcBits)];
      if (__builtin_expect(e != 0, 1)) {  // code, run and magnitude in one lookup
        k += (e >> 8) & 15;
        blk[kNatural[k]] = int16_t(e >> 16);
        br.drop(e & 0xff);
        continue;
      }
      const int rs = br.decode(ac);
      const int r = rs >> 4, sz = rs & 15;
      if (sz) {
        k += r;
        blk[kNatural[k]] = int16_t(br.extend_bits(sz));
      } else {
        if (r != 15) break;
        k += 15;
      }
    }
  } else if constexpr (K == kDcFirst) {
    br.ensure32();
    st.dc_pred[sc] = add_dc(st.dc_pred[sc], br.extend_bits(br.decode(dc_[c.dc_tbl])));
    blk[0] = int16_t(uint32_t(st.dc_pred[sc]) << al_);
  } else if constexpr (K == kDcRefine) {
    if (br.bit()) blk[0] = int16_t(blk[0] | (1 << al_));
  } else if constexpr (K == kAcFirst) {
    if (st.eobrun > 0) {
      st.eobrun--;
      return;
    }
    const Huffman& ac = ac_[c.ac_tbl];
    for (int k = ss_; k <= se_; k++) {
      br.ensure32();
      const int rs = br.decode(ac);
      const int r = rs >> 4, sz = rs & 15;
      if (sz) {
        k += r;
        blk[kNatural[k]] = int16_t(uint32_t(br.extend_bits(sz)) << al_);
      } else if (r == 15) {
        k += 15;
      } else {
        st.eobrun = (1 << r) + br.bits(r) - 1;  // this band counts as the first
        break;
      }
    }
  } else {  // kAcRefine
    const int p1 = 1 << al_, m1 = -1 * (1 << al_);
    const Huffman& ac = ac_[c.ac_tbl];
    int k = ss_;
    auto refine = [&](int16_t& coef) {  // correction bit of an already-nonzero coefficient
      if (br.bit() && (coef & p1) == 0) coef = int16_t(coef >= 0 ? coef + p1 : coef + m1);
    };
    if (st.eobrun == 0) {
      for (; k <= se_; k++) {
        br.ensure32();
        const int rs = br.decode(ac);
        int r = rs >> 4, sz = rs & 15;
        if (sz) {
          sz = br.bits(1) ? p1 : m1;  // magnitude is 1 in valid streams
        } else if (r != 15) {
          st.eobrun = (1 << r) + br.bits(r);
          break;
        }
        do {
          int16_t& coef = blk[kNatural[k]];
          if (coef != 0) {
            refine(coef);
          } else if (--r < 0) {
            break;
          }
          k++;
        } while (k <= se_);
        if (sz) blk[kNatural[k]] = int16_t(sz);
      }
    }
    if (st.eobrun > 0) {
      for (; k <= se_; k++) {
        int16_t& coef = blk[kNatural[k]];
        if (coef != 0) refine(coef);
      }
      st.eobrun--;
    }
  }
}

template <Decoder::ScanKind K>
JPEG_INLINE void Decoder::decode_mcu(BitReader& br, SegState& st, int mx, int my) {
  if (scan_n_ == 1) {  // non-interleaved: one block per MCU
    Component& c = comp_[scan_comp_[0]];
    decode_block<K>(br, st, 0, c, c.block(mx, my));
    return;
  }
  for (int i = 0; i < scan_n_; i++) {
    Component& c = comp_[scan_comp_[i]];
    for (int by = 0; by < c.v; by++)
      for (int bx = 0; bx < c.h; bx++) decode_block<K>(br, st, i, c, c.block(mx * c.h + bx, my * c.v + by));
  }
}

// process_restart + read_restart_marker + jpeg_resync_to_restart. `pos` and
// `marker` describe where the previous segment's reader stopped. Returns false
// when the marker is left unread, which makes the next segment empty.
bool Decoder::restart(const uint8_t* pos, int marker) {
  p_ = pos;
  int m = marker > 0 ? (p_ += 2, marker) : next_marker();
  for (;;) {
    if (m == 0xD0 + next_restart_) break;
    int action;
    if (m < 0xC0)
      action = 2;
    else if (m < 0xD0 || m > 0xD7)
      action = 3;
    else if (m == 0xD0 + ((next_restart_ + 1) & 7) || m == 0xD0 + ((next_restart_ + 2) & 7))
      action = 3;
    else if (m == 0xD0 + ((next_restart_ - 1) & 7) || m == 0xD0 + ((next_restart_ - 2) & 7))
      action = 2;
    else
      action = 1;
    if (action == 1) break;
    if (action == 3) {
      p_ -= 2;
      next_restart_ = (next_restart_ + 1) & 7;
      return false;
    }
    m = next_marker();
    if (m == 0) fail("truncated JPEG");
  }
  next_restart_ = (next_restart_ + 1) & 7;
  return true;
}

// Reconstruction in bands of MCU rows while a single-scan image is still
// being entropy-decoded. A band job IDCTs its MCU rows and converts the
// output rows whose upsampling context lies inside the band; the few rows at
// band borders are converted after the scan.
class Decoder::Pipeline {
 public:
  Pipeline(Decoder& d, int band_rows) : d_(d), band_rows_(band_rows), group_(dispatch_group_create()) {}
  ~Pipeline() {
    dispatch_group_wait(group_, DISPATCH_TIME_FOREVER);  // also on exceptions
    dispatch_release(group_);
  }
  // MCU row `my` is fully decoded.
  void row_done(int my) {
    const int last = d_.mcus_y_ - 1;
    if ((my + 1) % band_rows_ != 0 && my != last) return;
    const int a = my + 1 - ((my % band_rows_) + 1), b = my + 1;
    jobs_.push_back(std::make_unique<Job>(Job{&d_, a, b}));
    dispatch_group_async_f(group_, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), jobs_.back().get(),
                           [](void* p) {
                             Job* j = static_cast<Job*>(p);
                             j->d->idct_mcu_rows(j->a, j->b);
                             const auto [y0, y1] = inner_rows(*j->d, j->a, j->b);
                             if (y0 < y1) j->d->color_rows(y0, y1);
                           });
  }
  // Waits for the bands and converts the rows between them.
  void finish() {
    dispatch_group_wait(group_, DISPATCH_TIME_FOREVER);
    for (const auto& j : jobs_) {
      const auto [y0, y1] = inner_rows(d_, j->a, j->b);
      const int top = d_.out_rows_of_mcu_row(j->a), bottom = d_.out_rows_of_mcu_row(j->b);
      if (y0 >= y1) {
        d_.color_rows(top, bottom);
        continue;
      }
      if (top < y0) d_.color_rows(top, y0);
      if (y1 < bottom) d_.color_rows(y1, bottom);
    }
  }

 private:
  struct Job {
    Decoder* d;
    int a, b;
  };
  // Output rows of MCU rows [a, b) that only need component rows of the band:
  // upsampling reaches one component row (at most max_v output rows) away.
  static std::pair<int, int> inner_rows(const Decoder& d, int a, int b) {
    const int m = d.max_v_;
    const int y0 = d.out_rows_of_mcu_row(a) + (a > 0 ? m : 0);
    const int y1 = d.out_rows_of_mcu_row(b) - (b < d.mcus_y_ ? m : 0);
    return {y0, y1};
  }

  Decoder& d_;
  int band_rows_;
  dispatch_group_t group_;
  std::vector<std::unique_ptr<Job>> jobs_;
};

template <Decoder::ScanKind K>
void Decoder::scan_sequential(Pipeline* pipe) {
  const int mx = scan_mcus_x(), my = scan_mcus_y();
  BitReader br(p_, end_);
  SegState st;
  int restarts_to_go = restart_interval_;
  bool empty = false;  // libjpeg's insufficient_data after a failed resync
  for (int y = 0; y < my; y++) {
    for (int x = 0; x < mx; x++) {
      if (restart_interval_ && restarts_to_go == 0) {
        empty = !restart(br.pos(), br.marker());
        br = BitReader(p_, end_);
        st = SegState{};
        restarts_to_go = restart_interval_;
      }
      if (!empty) {
        decode_mcu<K>(br, st, x, y);
        // libjpeg stuffs zeros once the data runs into a marker and leaves
        // the rest of the segment alone; for us that is corrupt input.
        if (br.overrun()) fail("corrupt JPEG data: premature end of data segment");
      }
      if (restart_interval_) restarts_to_go--;
    }
    if (pipe) pipe->row_done(y);
  }
  p_ = br.marker() == -1 ? end_ : br.pos();
}

// Restart intervals are independent: find every RSTn marker, check the
// D0..D7 cycle and count, and decode the segments on all cores. Returns false
// (nothing decoded) when the markers are not regular; the sequential path
// then applies libjpeg's resynchronisation rules.
template <Decoder::ScanKind K>
bool Decoder::scan_parallel() {
  const int mx = scan_mcus_x(), my = scan_mcus_y();
  const int64_t total = int64_t(mx) * my, ri = restart_interval_;
  const int64_t nseg = (total + ri - 1) / ri;
  if (nseg < 4 || nseg > (1 << 20)) return false;
  std::vector<const uint8_t*> start(static_cast<size_t>(nseg)), stop(static_cast<size_t>(nseg));
  const uint8_t* q = p_;
  for (int64_t s = 0; s < nseg; s++) {
    start[size_t(s)] = q;
    for (;;) {  // next marker
      q = static_cast<const uint8_t*>(std::memchr(q, 0xFF, size_t(end_ - q)));
      if (!q) return false;
      while (q + 1 < end_ && q[1] == 0xFF) q++;
      if (q + 1 >= end_) return false;
      if (q[1] != 0) break;
      q += 2;
    }
    stop[size_t(s)] = q;
    const bool is_rst = q[1] >= 0xD0 && q[1] <= 0xD7;
    if (s + 1 < nseg) {
      if (!is_rst || q[1] != 0xD0 + (s & 7)) return false;
      q += 2;
    } else if (is_rst) {
      return false;  // more segments than MCUs
    }
  }
  struct Ctx {
    Decoder* d;
    const std::vector<const uint8_t*>* start;
    const std::vector<const uint8_t*>* stop;
    int64_t nseg, total, ri;
    int mx, chunks;
    std::atomic<const char*> error{nullptr};
  } ctx{this, &start, &stop, nseg, total, ri, mx, int(std::min<int64_t>(nseg, 64)), {}};
  dispatch_apply_f(size_t(ctx.chunks), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &ctx,
                   [](void* p, size_t i) {
                     Ctx& c = *static_cast<Ctx*>(p);
                     const int64_t s0 = c.nseg * int64_t(i) / c.chunks, s1 = c.nseg * int64_t(i + 1) / c.chunks;
                     try {
                       for (int64_t s = s0; s < s1 && !c.error.load(std::memory_order_relaxed); s++) {
                         BitReader br((*c.start)[size_t(s)], (*c.stop)[size_t(s)]);
                         SegState st;
                         const int64_t m1 = std::min(c.total, (s + 1) * c.ri);
                         for (int64_t m = s * c.ri; m < m1; m++)
                           c.d->template decode_mcu<K>(br, st, int(m % c.mx), int(m / c.mx));
                         if (br.overrun()) c.error = "corrupt JPEG data: premature end of data segment";
                       }
                     } catch (const Error& e) {
                       c.error = e.what;
                     }
                   });
  if (const char* e = ctx.error.load()) fail(e);
  p_ = stop[size_t(nseg - 1)];
  return true;
}

void Decoder::decode_scan() {
  // A sequential scan with every component is the whole image; libjpeg
  // rejects any further scan (JERR_EOI_EXPECTED).
  if (reconstructed_ || (!progressive_ && scans_ > 0 && single_scan_image_)) fail("JPEG: expected EOI");
  if (!progressive_ && scan_n_ == ncomp_) single_scan_image_ = true;
  next_restart_ = 0;
  const int64_t mcus = int64_t(scan_mcus_x()) * scan_mcus_y();
  const bool try_parallel = restart_interval_ > 0 && mcus >= 2048;
  // A sequential scan holding every component is the only scan: reconstruct
  // bands as they complete.
  const bool pipelined = !progressive_ && scan_n_ == ncomp_ && scans_ == 0 &&
                         int64_t(width_) * height_ >= (1 << 18);
  bool done = false;
  if (!progressive_) {
    if (try_parallel) done = scan_parallel<kSequential>();
    if (!done && pipelined) {
      setup_output(*out_);
      Pipeline pipe(*this, std::max(1, mcus_y_ / 24));
      scan_sequential<kSequential>(&pipe);
      pipe.finish();
      reconstructed_ = true;
      done = true;
    }
    if (!done) scan_sequential<kSequential>(nullptr);
  } else if (ss_ == 0 && ah_ == 0) {
    if (!(try_parallel && scan_parallel<kDcFirst>())) scan_sequential<kDcFirst>(nullptr);
  } else if (ss_ == 0) {
    if (!(try_parallel && scan_parallel<kDcRefine>())) scan_sequential<kDcRefine>(nullptr);
  } else if (ah_ == 0) {
    if (!(try_parallel && scan_parallel<kAcFirst>())) scan_sequential<kAcFirst>(nullptr);
  } else {
    if (!(try_parallel && scan_parallel<kAcRefine>())) scan_sequential<kAcRefine>(nullptr);
  }
  scans_++;
}

// ---- reconstruction ------------------------------------------------------------------

void Decoder::check_smoothing() {
  if (!progressive_) return;
  // smoothing_ok() in jdcoefct.c: libjpeg smooths blocks when every component
  // has usable quantisers and a known DC, and some component still lacks full
  // precision in one of its first 9 AC coefficients.
  bool usable = true, useful = false;
  for (int i = 0; i < ncomp_; i++) {
    const Component& c = comp_[i];
    for (int k : {0, 1, 8, 16, 9, 2, 3, 10, 17, 24}) usable &= c.quant[k] != 0;
    usable &= c.coef_bits[0] >= 0;
    for (int k = 1; k < 10; k++) useful |= c.coef_bits[k] != 0;
  }
  if (usable && useful) unsupported("progressive JPEG needs block smoothing");
}

// Colour space (default_decompress_parms), upsampling methods
// (jinit_upsampler with do_fancy = TRUE) and output buffers.
void Decoder::setup_output(RGBImage& out) {
  if (ncomp_ == 1) {
    cs_ = kGray;
  } else if (ncomp_ == 3) {
    if (jfif_) cs_ = kYCbCr;
    else if (adobe_) cs_ = adobe_transform_ == 0 ? kRGB : kYCbCr;
    else if (comp_[0].id == 1 && comp_[1].id == 2 && comp_[2].id == 3) cs_ = kYCbCr;
    else if (comp_[0].id == 82 && comp_[1].id == 71 && comp_[2].id == 66) cs_ = kRGB;
    else cs_ = kYCbCr;
  } else {
    cs_ = adobe_ ? (adobe_transform_ == 0 ? kCMYK : kYCCK) : kCMYK;
  }
  for (int i = 0; i < ncomp_; i++) {
    const Component& c = comp_[i];
    Plane& pl = planes_[i];
    if (c.h == max_h_ && c.v == max_v_) pl.method = Upsample::Full;
    else if (c.h * 2 == max_h_ && c.v == max_v_) pl.method = c.dw > 2 ? Upsample::H2V1Fancy : Upsample::H2V1Box;
    else if (c.h == max_h_ && c.v * 2 == max_v_) pl.method = Upsample::H1V2Fancy;
    else if (c.h * 2 == max_h_ && c.v * 2 == max_v_) pl.method = c.dw > 2 ? Upsample::H2V2Fancy : Upsample::H2V2Box;
    else if (max_h_ % c.h == 0 && max_v_ % c.v == 0) {
      pl.method = Upsample::Integral;
      pl.hx = max_h_ / c.h;
      pl.vx = max_v_ / c.v;
    } else {
      unsupported("fractional sampling ratio");
    }
    // Rows padded by 16 bytes for the vector loops' overreads.
    pl.stride = c.width_in_blocks * 8 + 16;
    pl.px.resize(size_t(pl.stride) * c.height_in_blocks * 8);
  }
  out.width = width_;
  out.height = height_;
  out.rgb.resize(size_t(width_) * height_ * 3);
}

void Decoder::idct_mcu_rows(int a, int b) {
  for (int i = 0; i < ncomp_; i++) {
    Component& c = comp_[i];
    Plane& pl = planes_[i];
    const int r1 = std::min(b * c.v, c.height_in_blocks);
    for (int by = a * c.v; by < r1; by++)
      for (int bx = 0; bx < c.width_in_blocks; bx++)
        idct_islow(c.block(bx, by), c.quant, pl.px.data() + size_t(by) * 8 * pl.stride + size_t(bx) * 8,
                   size_t(pl.stride));
  }
}

// Output rows [y0, y1): upsample each component (jdsample.c) and convert.
void Decoder::color_rows(int y0, int y1) {
  const int W = width_;
  std::vector<uint8_t> buf(size_t(W + 32) * 4);
  for (int y = y0; y < y1; y++) {
    const uint8_t* row[4];
    for (int i = 0; i < ncomp_; i++) {
      const Component& c = comp_[i];
      const Plane& pl = planes_[i];
      auto src = [&](int r) { return pl.px.data() + size_t(std::clamp(r, 0, c.dh - 1)) * pl.stride; };
      uint8_t* tmp = buf.data() + size_t(W + 32) * i;
      switch (pl.method) {
        case Upsample::Full:
          row[i] = src(y);
          break;
        case Upsample::Integral: {
          const uint8_t* in = src(y / pl.vx);
          for (int x = 0; x < W; x++) tmp[x] = in[x / pl.hx];
          row[i] = tmp;
          break;
        }
        case Upsample::H2V1Box:
        case Upsample::H2V2Box: {
          const uint8_t* in = src(pl.method == Upsample::H2V2Box ? y >> 1 : y);
          for (int x = 0; x < W; x++) tmp[x] = in[x >> 1];
          row[i] = tmp;
          break;
        }
        case Upsample::H2V1Fancy:
          h2v1_fancy(src(y), c.dw, tmp);
          row[i] = tmp;
          break;
        case Upsample::H1V2Fancy:
        case Upsample::H2V2Fancy: {
          // Even output rows pair the nearest row with the one above, odd rows
          // with the one below (clamped: the edge rows repeat).
          const int r = y >> 1;
          const uint8_t* near = src(r);
          const uint8_t* far = src(y & 1 ? r + 1 : r - 1);
          if (pl.method == Upsample::H1V2Fancy) h1v2_fancy(near, far, W, y & 1 ? 2 : 1, tmp);
          else h2v2_fancy(near, far, c.dw, tmp);
          row[i] = tmp;
          break;
        }
      }
    }
    uint8_t* d = out_->rgb.data() + size_t(y) * W * 3;
    switch (cs_) {
      case kGray:
        planes_to_rgb(row[0], row[0], row[0], W, d);
        break;
      case kRGB:
        planes_to_rgb(row[0], row[1], row[2], W, d);
        break;
      case kYCbCr:
        ycc_to_rgb(row[0], row[1], row[2], W, d);
        break;
      case kCMYK:
      case kYCCK:
        for (int x = 0; x < W; x++) {
          int v[4];
          if (cs_ == kCMYK) {
            for (int k = 0; k < 4; k++) v[k] = row[k][x];
          } else {  // ycck_cmyk_convert
            const int yy = row[0][x], cb = row[1][x], cr = row[2][x];
            v[0] = clamp255(255 - (yy + kYcc.cr_r[cr]));
            v[1] = clamp255(255 - (yy + int((kYcc.cb_g[cb] + kYcc.cr_g[cr]) >> 16)));
            v[2] = clamp255(255 - (yy + kYcc.cb_b[cb]));
            v[3] = row[3][x];
          }
          // PIL reads "CMYK;I" (inverted), then converts with its own formula.
          const int k = 255 - v[3];
          d[3 * x] = pil_cmyk_channel(255 - v[0], k);
          d[3 * x + 1] = pil_cmyk_channel(255 - v[1], k);
          d[3 * x + 2] = pil_cmyk_channel(255 - v[2], k);
        }
        break;
    }
  }
}

template <class Fn>
void parallel_for(int n, int chunks, Fn&& fn) {
  chunks = std::max(1, std::min(n, chunks));
  if (chunks == 1) {
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
                     (*c->fn)(int(int64_t(c->n) * i / c->chunks), int(int64_t(c->n) * (i + 1) / c->chunks));
                   });
}

void Decoder::run(RGBImage& out) {
  out_ = &out;
  p_ = data_;
  if (end_ - p_ < 4 || p_[0] != 0xFF || p_[1] != 0xD8) fail("not a JPEG file");
  p_ += 2;
  for (;;) {
    const int m = next_marker();
    if (m == 0 || m == 0xD9) break;  // end of data / EOI
    if (m == 0xD8 || (m >= 0xD0 && m <= 0xD7) || m == 0x01) continue;  // no payload
    const uint32_t len = u16();
    if (len < 2 || size_t(end_ - p_) < len - 2) fail("truncated JPEG segment");
    const uint8_t* seg = p_;
    const size_t n = len - 2;
    p_ += n;
    switch (m) {
      case 0xC4: read_dht(seg, n); break;
      case 0xDB: read_dqt(seg, n); break;
      case 0xDD:
        if (n < 2) fail("bad DRI");
        restart_interval_ = seg[0] << 8 | seg[1];
        break;
      case 0xDA:
        read_sos(seg, n);
        decode_scan();
        break;
      case 0xDC: unsupported("DNL marker");
      default:
        if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) read_sof(m, seg, n);
        else if (m == 0xCC) unsupported("arithmetic coding");
        else if (m >= 0xE0 && m <= 0xEF) read_app(m, seg, n);
        break;  // COM and unknown markers are skipped
    }
  }
  if (!frame_) fail("no JPEG frame");
  if (scans_ == 0) fail("JPEG has no scans");
  for (int i = 0; i < ncomp_; i++)
    if (!comp_[i].latched) fail("component never scanned");
  if (reconstructed_) return;
  check_smoothing();
  setup_output(out);
  const int n = mcus_y_;
  const int chunks = int(std::min<int64_t>(n, std::max<int64_t>(1, int64_t(width_) * height_ / 65536)));
  parallel_for(n, std::min(chunks, 32), [&](int a, int b) { idct_mcu_rows(a, b); });
  parallel_for(height_, std::min(chunks, 32), [&](int y0, int y1) { color_rows(y0, y1); });
}

}  // namespace

bool jpeg_decode(std::string_view bytes, RGBImage& out, std::string& err, bool& unsup) {
  unsup = false;
  try {
    Decoder d(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
    d.run(out);
    return true;
  } catch (const Error& e) {
    err = std::string("jpeg: ") + e.what;
    unsup = e.unsupported;
  } catch (const std::bad_alloc&) {
    err = "jpeg: out of memory";
  }
  out = RGBImage{};
  return false;
}

}  // namespace espetero::vision::detail
