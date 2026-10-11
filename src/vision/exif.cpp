// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// EXIF orientation, applied after decoding like PIL's ImageOps.exif_transpose
// (what mlx-vlm's load_image and transformers do before their processors):
// phone photos that store the sensor's pixels plus an orientation tag reach
// the vision tower upright. The tag is read from the first EXIF block of a
// JPEG (APP1 "Exif\0\0"), PNG (eXIf), WebP (EXIF chunk) or TIFF file (IFD0),
// orientation 0x0112; values 2..8 map to PIL's transpose methods.
#include <atomic>
#include <cstring>

#include "decode_internal.h"

namespace espetero::vision {

namespace {

std::atomic<bool> g_apply{true};

uint32_t rd(const uint8_t* p, int n, bool be) {
  uint32_t v = 0;
  for (int i = 0; i < n; ++i) v |= uint32_t(p[be ? n - 1 - i : i]) << (8 * i);
  return v;
}

// Orientation tag of a TIFF structure (IFD0), 1 when absent or malformed.
int tiff_orientation(const uint8_t* t, size_t n) {
  if (n < 8) return 1;
  bool be;
  if (t[0] == 'I' && t[1] == 'I') be = false;
  else if (t[0] == 'M' && t[1] == 'M') be = true;
  else return 1;
  if (rd(t + 2, 2, be) != 42) return 1;
  const uint32_t ifd = rd(t + 4, 4, be);
  if (ifd < 8 || size_t(ifd) + 2 > n) return 1;
  const uint32_t count = rd(t + ifd, 2, be);
  for (uint32_t i = 0; i < count; ++i) {
    const size_t e = size_t(ifd) + 2 + size_t(i) * 12;
    if (e + 12 > n) break;
    if (rd(t + e, 2, be) != 0x0112) continue;
    const uint32_t type = rd(t + e + 2, 2, be);
    if (type == 3) return int(rd(t + e + 8, 2, be));   // SHORT
    if (type == 4) return int(rd(t + e + 8, 4, be));   // LONG (lenient)
    return 1;
  }
  return 1;
}

int jpeg_orientation(const uint8_t* b, size_t n) {
  size_t i = 2;
  while (i + 4 <= n) {
    if (b[i] != 0xFF) return 1;
    const uint8_t m = b[i + 1];
    if (m == 0xFF) {  // fill byte
      ++i;
      continue;
    }
    if (m == 0xD8 || (m >= 0xD0 && m <= 0xD7) || m == 0x01) {
      i += 2;
      continue;
    }
    if (m == 0xDA || m == 0xD9) return 1;  // scan data / end: no EXIF before it
    const size_t len = size_t(b[i + 2]) << 8 | b[i + 3];
    if (len < 2 || i + 2 + len > n) return 1;
    const uint8_t* seg = b + i + 4;
    const size_t sl = len - 2;
    if (m == 0xE1 && sl >= 6 && std::memcmp(seg, "Exif\0\0", 6) == 0) return tiff_orientation(seg + 6, sl - 6);
    i += 2 + len;
  }
  return 1;
}

int png_orientation(const uint8_t* b, size_t n) {
  size_t i = 8;
  while (i + 12 <= n) {
    const size_t len = rd(b + i, 4, true);
    if (i + 12 + len > n) return 1;
    const uint8_t* type = b + i + 4;
    if (std::memcmp(type, "eXIf", 4) == 0) {
      const uint8_t* d = b + i + 8;
      if (len >= 6 && std::memcmp(d, "Exif\0\0", 6) == 0) return tiff_orientation(d + 6, len - 6);
      return tiff_orientation(d, len);
    }
    if (std::memcmp(type, "IDAT", 4) == 0 || std::memcmp(type, "IEND", 4) == 0) return 1;
    i += 12 + len;
  }
  return 1;
}

int webp_orientation(const uint8_t* b, size_t n) {
  size_t i = 12;
  while (i + 8 <= n) {
    const size_t len = rd(b + i + 4, 4, false);
    if (i + 8 + len > n) return 1;
    if (std::memcmp(b + i, "EXIF", 4) == 0) {
      const uint8_t* d = b + i + 8;
      if (len >= 6 && std::memcmp(d, "Exif\0\0", 6) == 0) return tiff_orientation(d + 6, len - 6);
      return tiff_orientation(d, len);
    }
    i += 8 + len + (len & 1);
  }
  return 1;
}

}  // namespace

void set_exif_orientation(bool on) { g_apply.store(on, std::memory_order_relaxed); }
bool exif_orientation_enabled() { return g_apply.load(std::memory_order_relaxed); }

int exif_orientation(std::string_view bytes) {
  const auto* b = reinterpret_cast<const uint8_t*>(bytes.data());
  const size_t n = bytes.size();
  if (n >= 4 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF) return jpeg_orientation(b, n);
  if (n >= 8 && std::memcmp(b, "\x89PNG\r\n\x1a\n", 8) == 0) return png_orientation(b, n);
  if (n >= 12 && std::memcmp(b, "RIFF", 4) == 0 && std::memcmp(b + 8, "WEBP", 4) == 0) return webp_orientation(b, n);
  if (n >= 8 && (std::memcmp(b, "II*\0", 4) == 0 || std::memcmp(b, "MM\0*", 4) == 0)) return tiff_orientation(b, n);
  return 1;
}

// PIL's transpose for orientation o: 2 FLIP_LEFT_RIGHT, 3 ROTATE_180,
// 4 FLIP_TOP_BOTTOM, 5 TRANSPOSE, 6 ROTATE_270, 7 TRANSVERSE, 8 ROTATE_90.
RGBImage apply_orientation(const RGBImage& in, int o) {
  if (o < 2 || o > 8) return in;
  const int w = in.width, h = in.height;
  const bool swap = o >= 5;
  RGBImage out;
  out.width = swap ? h : w;
  out.height = swap ? w : h;
  out.rgb.resize(size_t(out.width) * size_t(out.height) * 3);
  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < out.width; ++x) {
      int sx = x, sy = y;
      switch (o) {
        case 2: sx = w - 1 - x; sy = y; break;
        case 3: sx = w - 1 - x; sy = h - 1 - y; break;
        case 4: sx = x; sy = h - 1 - y; break;
        case 5: sx = y; sy = x; break;
        case 6: sx = y; sy = h - 1 - x; break;
        case 7: sx = w - 1 - y; sy = h - 1 - x; break;
        default: sx = w - 1 - y; sy = x; break;  // 8
      }
      const uint8_t* s = &in.rgb[(size_t(sy) * size_t(w) + size_t(sx)) * 3];
      uint8_t* d = &out.rgb[(size_t(y) * size_t(out.width) + size_t(x)) * 3];
      d[0] = s[0];
      d[1] = s[1];
      d[2] = s[2];
    }
  }
  return out;
}

}  // namespace espetero::vision
