// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ImageIO / CoreGraphics decoding without colour management.
//
// PIL's convert("RGB") never colour-matches (ICC profiles and gAMA are
// ignored), never rotates (EXIF orientation ignored) and drops alpha instead
// of compositing. So instead of drawing into a bitmap context (which would
// colour-match and composite), we take the decoded samples as they are via
// CGDataProviderCopyData and interpret the bitmap layout ourselves.
//
// Measured against PIL 12.3 (tests/vision_test.cpp): PNG of every colour
// type / bit depth, BMP, TIFF and WebP (lossless and lossy) decode
// byte-identical; ImageIO returns non-premultiplied alpha for them. If an
// image does come back premultiplied, colours are un-premultiplied, which
// cannot restore the exact original for alpha < 255 (and loses colour
// entirely at alpha 0): documented precision loss, affects no tested format.
// GIF transparency and JPEG are handled by gif.cpp / jpeg.cpp instead.
//
// Plain C APIs only; compiled as Objective-C++ so the build line can glob *.mm.
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "decode_internal.h"

namespace espetero::vision::detail {
namespace {

// Owns one CoreFoundation reference.
template <class T>
struct CF {
  T ref = nullptr;
  CF() = default;
  explicit CF(T r) : ref(r) {}
  CF(const CF&) = delete;
  CF& operator=(const CF&) = delete;
  ~CF() {
    if (ref) CFRelease(ref);
  }
  explicit operator bool() const { return ref != nullptr; }
};

inline uint8_t unpremultiply(int c, int a) {
  if (a == 255) return uint8_t(c);
  if (a == 0) return 0;
  const int v = (c * 255 + a / 2) / a;
  return uint8_t(v > 255 ? 255 : v);
}

inline uint8_t float_to_u8(float v) {
  if (!(v > 0.0f)) return 0;  // also NaN
  if (v >= 1.0f) return 255;
  return uint8_t(std::lround(v * 255.0f));
}

// Where each component of a pixel lives and how to read it.
struct Layout {
  int width = 0, height = 0;
  size_t bytes_per_row = 0;
  int bits_per_component = 0, bits_per_pixel = 0;
  int ncolor = 0;        // colour components (1 grey/indexed, 3 RGB, 4 CMYK)
  int alpha_pos = -1;    // component index of alpha or padding, -1 when none
  bool premultiplied = false;
  bool is_float = false;
  bool swap16 = false, swap32 = false;  // samples stored opposite to host order
  bool swap_pixel32 = false;            // 8-bit components in a little-endian 32-bit word
};

// Reads component `c` of pixel `x` in `row` as 8 bits (16-bit: PIL keeps the
// high byte; grey I;16 clipping is applied by the caller).
struct Reader {
  const Layout& L;
  int ncomp;  // components per pixel including alpha/padding

  uint32_t raw(const uint8_t* row, int x, int c) const {
    switch (L.bits_per_component) {
      case 8: {
        if (L.swap_pixel32) {  // word order reversed: component c sits at 3 - c
          return row[size_t(x) * 4 + (3 - c)];
        }
        return row[size_t(x) * ncomp + c];
      }
      case 16: {
        uint16_t v;
        std::memcpy(&v, row + (size_t(x) * ncomp + c) * 2, 2);
        return L.swap16 ? __builtin_bswap16(v) : v;
      }
      case 32: {
        uint32_t v;
        std::memcpy(&v, row + (size_t(x) * ncomp + c) * 4, 4);
        return L.swap32 ? __builtin_bswap32(v) : v;
      }
    }
    return 0;
  }
  float as_float(const uint8_t* row, int x, int c) const {
    const uint32_t r = raw(row, x, c);
    if (L.bits_per_component == 32) {
      float f;
      std::memcpy(&f, &r, 4);
      return f;
    }
    return float(__builtin_bit_cast(_Float16, uint16_t(r)));
  }
};

bool describe(CGImageRef img, Layout& L, std::string& err) {
  L.width = int(CGImageGetWidth(img));
  L.height = int(CGImageGetHeight(img));
  L.bytes_per_row = CGImageGetBytesPerRow(img);
  L.bits_per_component = int(CGImageGetBitsPerComponent(img));
  L.bits_per_pixel = int(CGImageGetBitsPerPixel(img));
  const CGBitmapInfo info = CGImageGetBitmapInfo(img);
  const CGImageAlphaInfo alpha = CGImageGetAlphaInfo(img);
  CGColorSpaceRef cs = CGImageGetColorSpace(img);
  if (!cs) {
    err = "image has no colour space (mask image)";
    return false;
  }
  L.is_float = (info & kCGBitmapFloatComponents) != 0;
  switch (CGColorSpaceGetModel(cs)) {
    case kCGColorSpaceModelMonochrome:
    case kCGColorSpaceModelIndexed:
      L.ncolor = 1;
      break;
    case kCGColorSpaceModelRGB:
      L.ncolor = 3;
      break;
    case kCGColorSpaceModelCMYK:
      L.ncolor = 4;
      break;
    default:
      L.ncolor = 0;  // Lab, DeviceN, ...: drawn through CoreGraphics instead
      return true;
  }
  const bool has_extra = alpha != kCGImageAlphaNone && alpha != kCGImageAlphaOnly;
  const bool first = alpha == kCGImageAlphaPremultipliedFirst || alpha == kCGImageAlphaFirst ||
                     alpha == kCGImageAlphaNoneSkipFirst;
  L.premultiplied =
      alpha == kCGImageAlphaPremultipliedFirst || alpha == kCGImageAlphaPremultipliedLast;
  const int ncomp = L.ncolor + (has_extra ? 1 : 0);
  if (has_extra) L.alpha_pos = first ? 0 : L.ncolor;
  if (L.bits_per_pixel != ncomp * L.bits_per_component ||
      (L.bits_per_component != 8 && L.bits_per_component != 16 && L.bits_per_component != 32) ||
      (L.is_float && L.bits_per_component == 8) ||
      (!L.is_float && L.bits_per_component == 32) || (info & kCGImagePixelFormatMask) != 0) {
    L.ncolor = 0;  // packed / unusual layout: draw instead
    return true;
  }
  if (L.bytes_per_row < size_t(L.width) * size_t(L.bits_per_pixel) / 8) {
    err = "malformed bitmap layout";
    return false;
  }
  const uint32_t order = info & kCGBitmapByteOrderMask;
  // Host is little-endian. "Default" order means big-endian samples.
  if (L.bits_per_component == 16)
    L.swap16 = order != kCGBitmapByteOrder16Little;
  if (L.bits_per_component == 32)
    L.swap32 = order != kCGBitmapByteOrder32Little;
  if (L.bits_per_component == 8 && order == kCGBitmapByteOrder32Little) {
    if (ncomp != 4) {
      L.ncolor = 0;
      return true;
    }
    L.swap_pixel32 = true;
  }
  return true;
}

// Fallback for layouts we do not interpret: draw into 8-bit RGBX. When the
// image's own colour space is RGB, the context uses it too, so no matching
// happens; other models get converted to sRGB (no PIL equivalent anyway).
bool draw_rgb(CGImageRef img, RGBImage& out, std::string& err) {
  const size_t w = CGImageGetWidth(img), h = CGImageGetHeight(img);
  CGColorSpaceRef own = CGImageGetColorSpace(img);
  CF<CGColorSpaceRef> srgb;
  CGColorSpaceRef cs = own;
  if (!own || CGColorSpaceGetModel(own) != kCGColorSpaceModelRGB) {
    srgb.ref = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    cs = srgb.ref;
  }
  std::vector<uint8_t> buf(w * h * 4, 0);
  CF<CGContextRef> ctx(CGBitmapContextCreate(buf.data(), w, h, 8, w * 4, cs,
                                             uint32_t(kCGImageAlphaNoneSkipLast) |
                                                 uint32_t(kCGBitmapByteOrderDefault)));
  if (!ctx) {
    err = "cannot create a bitmap context";
    return false;
  }
  CGContextSetBlendMode(ctx.ref, kCGBlendModeCopy);
  CGContextSetInterpolationQuality(ctx.ref, kCGInterpolationNone);
  CGContextDrawImage(ctx.ref, CGRectMake(0, 0, CGFloat(w), CGFloat(h)), img);
  out.width = int(w);
  out.height = int(h);
  out.rgb.resize(w * h * 3);
  for (size_t i = 0; i < w * h; i++) std::memcpy(&out.rgb[i * 3], &buf[i * 4], 3);
  return true;
}

// Fast path: 8-bit RGB with a trailing alpha/padding byte (the layout ImageIO
// uses for most RGB images). Straight alpha is simply dropped.
void rgbx8_to_rgb(const uint8_t* src, size_t bpr, int w, int h, uint8_t* dst) {
  for (int y = 0; y < h; y++) {
    const uint8_t* s = src + size_t(y) * bpr;
    uint8_t* d = dst + size_t(y) * w * 3;
    int x = 0;
#if defined(__ARM_NEON)
    for (; x + 16 <= w; x += 16) {
      const uint8x16x4_t v = vld4q_u8(s + 4 * x);
      const uint8x16x3_t o = {{v.val[0], v.val[1], v.val[2]}};
      vst3q_u8(d + 3 * x, o);
    }
#endif
    for (; x < w; x++) {
      d[3 * x] = s[4 * x];
      d[3 * x + 1] = s[4 * x + 1];
      d[3 * x + 2] = s[4 * x + 2];
    }
  }
}

bool convert(CGImageRef img, const Layout& L, const ImageIORules& rules, RGBImage& out,
             std::string& err) {
  CF<CFDataRef> data(CGDataProviderCopyData(CGImageGetDataProvider(img)));
  if (!data) {
    err = "cannot read decoded pixels";
    return false;
  }
  const uint8_t* base = CFDataGetBytePtr(data.ref);
  const int w = L.width, h = L.height;
  if (size_t(CFDataGetLength(data.ref)) < L.bytes_per_row * size_t(h - 1) + size_t(w) * L.bits_per_pixel / 8) {
    err = "decoded pixel buffer is too small";
    return false;
  }
  out.width = w;
  out.height = h;
  out.rgb.resize(size_t(w) * h * 3);
  uint8_t* dst = out.rgb.data();
  CGColorSpaceRef cs = CGImageGetColorSpace(img);
  const CGColorSpaceModel model = CGColorSpaceGetModel(cs);
  const int ncomp = L.ncolor + (L.alpha_pos >= 0 ? 1 : 0);
  const int c0 = L.alpha_pos == 0 ? 1 : 0;  // first colour component
  Reader rd{L, ncomp};

  if (model == kCGColorSpaceModelRGB && L.bits_per_component == 8 && !L.premultiplied &&
      !L.swap_pixel32 && ncomp == 4 && c0 == 0) {
    rgbx8_to_rgb(base, L.bytes_per_row, w, h, dst);
    return true;
  }

  // Indexed: colour table lookup (the base space of PNG/GIF palettes is RGB).
  std::vector<uint8_t> table;
  if (model == kCGColorSpaceModelIndexed) {
    CGColorSpaceRef bcs = CGColorSpaceGetBaseColorSpace(cs);
    const size_t n = CGColorSpaceGetColorTableCount(cs);
    const size_t bn = bcs ? CGColorSpaceGetNumberOfComponents(bcs) : 0;
    if (L.bits_per_component != 8 || n == 0 || bn == 0 || bn > 4) {
      return draw_rgb(img, out, err);
    }
    std::vector<uint8_t> raw(n * bn);
    CGColorSpaceGetColorTable(cs, raw.data());
    // Entries beyond the palette are black, like a short PIL palette.
    table.assign(256 * 3, 0);
    for (size_t i = 0; i < std::min<size_t>(n, 256); i++) {
      for (int c = 0; c < 3; c++) table[i * 3 + c] = raw[i * bn + (bn == 1 ? 0 : std::min<int>(c, int(bn) - 1))];
    }
  }

  for (int y = 0; y < h; y++) {
    const uint8_t* row = base + size_t(y) * L.bytes_per_row;
    uint8_t* d = dst + size_t(y) * w * 3;
    for (int x = 0; x < w; x++, d += 3) {
      int a = 255;
      if (L.premultiplied) {
        const uint32_t ar = rd.raw(row, x, L.alpha_pos);
        a = L.is_float ? int(float_to_u8(rd.as_float(row, x, L.alpha_pos)))
                       : int(L.bits_per_component == 16 ? ar >> 8 : ar);
      }
      uint8_t v[4];
      for (int c = 0; c < L.ncolor; c++) {
        if (L.is_float) {
          v[c] = float_to_u8(rd.as_float(row, x, c0 + c));
        } else {
          const uint32_t r = rd.raw(row, x, c0 + c);
          if (L.bits_per_component == 16)
            v[c] = uint8_t(model == kCGColorSpaceModelMonochrome && rules.gray16_clip
                               ? std::min<uint32_t>(r, 255)
                               : r >> 8);
          else
            v[c] = uint8_t(r);
        }
        if (L.premultiplied && model != kCGColorSpaceModelIndexed) v[c] = unpremultiply(v[c], a);
      }
      switch (model) {
        case kCGColorSpaceModelIndexed:
          std::memcpy(d, &table[size_t(v[0]) * 3], 3);
          break;
        case kCGColorSpaceModelMonochrome:
          d[0] = d[1] = d[2] = v[0];
          break;
        case kCGColorSpaceModelCMYK: {
          const auto ink = [&](int c) { return rules.cmyk_inverted ? 255 - v[c] : int(v[c]); };
          const int k = ink(3);
          d[0] = pil_cmyk_channel(ink(0), k);
          d[1] = pil_cmyk_channel(ink(1), k);
          d[2] = pil_cmyk_channel(ink(2), k);
          break;
        }
        default:
          d[0] = v[0];
          d[1] = v[1];
          d[2] = v[2];
      }
    }
  }
  return true;
}

// Pixel width and height of the first image from its properties: ImageIO
// reads them from the header, without decoding (or caching) the image.
bool source_size(CGImageSourceRef src, int64_t& w, int64_t& h) {
  CF<CFDictionaryRef> props(CGImageSourceCopyPropertiesAtIndex(src, 0, nullptr));
  if (!props) return false;
  auto num = [&](CFStringRef key, int64_t& v) {
    const void* n = CFDictionaryGetValue(props.ref, key);
    return n && CFGetTypeID(n) == CFNumberGetTypeID() &&
           CFNumberGetValue(static_cast<CFNumberRef>(n), kCFNumberSInt64Type, &v);
  };
  return num(kCGImagePropertyPixelWidth, w) && num(kCGImagePropertyPixelHeight, h);
}

// The encoded bytes, copied. ImageIO keeps decoder state keyed to the data it
// was given: over a borrowed buffer (CFDataCreateWithBytesNoCopy), a new image
// whose bytes land at the address of an earlier, freed one could meet that
// state, and HEIC decodes then failed at random ("cannot read decoded pixels",
// a few in a hundred in tests/vision_limits_test). A copy of the compressed
// file costs nothing next to the decode.
CFDataRef source_data(std::string_view bytes) {
  return CFDataCreate(nullptr, reinterpret_cast<const UInt8*>(bytes.data()), CFIndex(bytes.size()));
}

}  // namespace

bool imageio_size(std::string_view bytes, int64_t& width, int64_t& height, std::string& err) {
  CF<CFDataRef> data(source_data(bytes));
  if (!data) {
    err = "out of memory";
    return false;
  }
  CF<CFMutableDictionaryRef> opts(CFDictionaryCreateMutable(
      nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
  CFDictionarySetValue(opts.ref, kCGImageSourceShouldCache, kCFBooleanFalse);
  CF<CGImageSourceRef> src(CGImageSourceCreateWithData(data.ref, opts.ref));
  if (!src || CGImageSourceGetCount(src.ref) < 1) {
    err = "unrecognised or corrupt image data";
    return false;
  }
  if (!source_size(src.ref, width, height)) {
    err = "cannot read the image size";
    return false;
  }
  return true;
}

bool imageio_decode(std::string_view bytes, const ImageIORules& rules, RGBImage& out,
                    std::string& err) {
  CF<CFDataRef> data(source_data(bytes));
  if (!data) {
    err = "out of memory";
    return false;
  }
  CF<CFMutableDictionaryRef> opts(CFDictionaryCreateMutable(
      nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
  // Samples as stored: never let ImageIO hand back float/HDR renditions; the
  // pixels are copied out at once, so nothing goes into ImageIO's cache.
  CFDictionarySetValue(opts.ref, kCGImageSourceShouldAllowFloat, kCFBooleanFalse);
  CFDictionarySetValue(opts.ref, kCGImageSourceShouldCache, kCFBooleanFalse);
  if (rules.software_only)  // exported but undocumented key; unknown keys are ignored
    CFDictionarySetValue(opts.ref, CFSTR("kCGImageSourceUseHardwareAcceleration"), kCFBooleanFalse);
  CF<CGImageSourceRef> src(CGImageSourceCreateWithData(data.ref, opts.ref));
  if (!src || CGImageSourceGetCount(src.ref) < 1) {
    err = "unrecognised or corrupt image data";
    return false;
  }
  // The decode cap (max_image_pixels(), below PIL's DecompressionBombError
  // limit), from the header before the image is even created, then from the
  // image itself before any pixel is decoded (ImageIO decodes lazily).
  int64_t pw = 0, ph = 0;
  if (source_size(src.ref, pw, ph) && pw > 0 && ph > 0 && over_pixel_cap(pw, ph)) {
    err = "image dimensions out of range";
    return false;
  }
  CF<CGImageRef> img(CGImageSourceCreateImageAtIndex(src.ref, 0, opts.ref));
  if (!img) {
    err = "cannot decode image";
    return false;
  }
  const size_t w = CGImageGetWidth(img.ref), h = CGImageGetHeight(img.ref);
  if (w < 1 || h < 1 || w > (1u << 30) || h > (1u << 30) || over_pixel_cap(int64_t(w), int64_t(h))) {
    err = "image dimensions out of range";
    return false;
  }
  Layout L;
  if (!describe(img.ref, L, err)) return false;
  if (L.ncolor == 0) return draw_rgb(img.ref, out, err);
  return convert(img.ref, L, rules, out, err);
}

}  // namespace espetero::vision::detail
