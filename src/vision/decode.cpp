// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Image bytes -> RGB with PIL's Image.open(...).convert("RGB") semantics.
// Routes GIF and JPEG to exact decoders (gif.cpp, jpeg.cpp) and everything
// else to ImageIO with the PIL conversion rules for that format.
#include <algorithm>
#include <atomic>
#include <fstream>
#include <sstream>

#include "decode_internal.h"

namespace espetero::vision {

namespace {

// Python's binascii.a2b_base64 in its default (non-strict) mode: characters
// outside the alphabet are skipped, a pad sequence ends the data once a quad
// has at least two characters, and leftovers are an error.
bool base64_decode(std::string_view in, std::string& out, std::string& err) {
  static const auto table = [] {
    std::array<int8_t, 256> t{};
    t.fill(-1);
    const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; i++) t[uint8_t(a[i])] = int8_t(i);
    return t;
  }();
  out.clear();
  out.reserve(in.size() / 4 * 3);
  int quad = 0, pads = 0;
  uint32_t left = 0;
  for (char ch : in) {
    if (uint8_t(ch) >= 0x80) {  // b64decode(str) first encodes to ASCII
      err = "string argument should contain only ASCII characters";
      return false;
    }
    if (ch == '=') {
      if (quad >= 2 && quad + ++pads >= 4) return true;  // complete pad sequence
      continue;
    }
    const int v = table[uint8_t(ch)];
    if (v < 0) continue;
    pads = 0;  // CPython only counts consecutive pads within the current quad
    switch (quad) {
      case 0:
        left = uint32_t(v);
        quad = 1;
        break;
      case 1:
        out.push_back(char((left << 2) | (uint32_t(v) >> 4)));
        left = uint32_t(v) & 0x0f;
        quad = 2;
        break;
      case 2:
        out.push_back(char((left << 4) | (uint32_t(v) >> 2)));
        left = uint32_t(v) & 0x03;
        quad = 3;
        break;
      default:
        out.push_back(char((left << 6) | uint32_t(v)));
        quad = 0;
    }
  }
  if (quad != 0) {
    err = quad == 1 ? "invalid base64: number of data characters is 1 more than a multiple of 4"
                    : "invalid base64: incorrect padding";
    return false;
  }
  return true;
}

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }

std::atomic<int64_t> g_max_pixels{kDefaultMaxImagePixels};

// gif.cpp's canvas: the logical screen enlarged by the first frame, found by
// the same walk over the blocks before it (gif_decode). False when the walk
// fails, as gif_decode would (ImageIO then decodes the file).
bool gif_size(std::string_view b, int64_t& w, int64_t& h) {
  size_t pos = 13;
  auto byte = [&]() { return pos < b.size() ? int(uint8_t(b[pos++])) : -1; };
  auto take = [&](size_t n) {
    const size_t k = std::min(n, b.size() - std::min(pos, b.size()));
    const std::string_view v = b.substr(std::min(pos, b.size()), k);
    pos += k;
    return v;
  };
  auto block = [&]() {
    const int n = byte();
    return n <= 0 ? std::string_view{} : take(size_t(n));
  };
  auto u16 = [](std::string_view s, size_t i) { return int64_t(uint8_t(s[i]) | uint8_t(s[i + 1]) << 8); };
  if (b.size() < 13) return false;
  w = u16(b, 6);
  h = u16(b, 8);
  if (const int flags = uint8_t(b[10]); flags & 128) take(size_t(3) << ((flags & 7) + 1));
  for (;;) {
    const int s = byte();
    if (s < 0 || s == ';') return false;
    if (s == '!') {
      const int label = byte();
      std::string_view blk = block();
      if (label == 0xF9 && !blk.empty() && blk.size() < 4) return false;
      if (label == 0xFE) {
        while (!blk.empty()) blk = block();
        continue;
      }
      if (label == 0xFF && blk.substr(0, 11) == "NETSCAPE2.0") block();
      while (!block().empty()) {
      }
      continue;
    }
    if (s != ',') continue;
    const std::string_view d = take(9);
    if (d.size() < 9) return false;
    w = std::max(w, u16(d, 0) + u16(d, 4));
    h = std::max(h, u16(d, 2) + u16(d, 6));
    return w >= 1 && h >= 1;
  }
}

// The first SOFn of a JPEG, reached by jpeg.cpp's marker walk (every segment
// skipped by its length). False without one, or for a DNL-sized frame (height
// 0): ImageIO decodes those.
bool jpeg_size(std::string_view b, int64_t& w, int64_t& h) {
  const auto* p = reinterpret_cast<const uint8_t*>(b.data()) + 2;
  const auto* end = reinterpret_cast<const uint8_t*>(b.data()) + b.size();
  for (;;) {
    while (p < end && *p != 0xFF) p++;
    while (p < end && *p == 0xFF) p++;
    if (p >= end) return false;
    const int m = *p++;
    if (m == 0) continue;
    if (m == 0xD9 || m == 0xDA) return false;                             // EOI, or a scan before any frame
    if (m == 0xD8 || (m >= 0xD0 && m <= 0xD7) || m == 0x01) continue;  // no payload
    if (end - p < 2) return false;
    const size_t len = size_t(p[0]) << 8 | p[1];
    if (len < 2 || size_t(end - p) < len) return false;
    if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) {
      if (len < 2 + 6) return false;
      h = int64_t(p[3]) << 8 | p[4];
      w = int64_t(p[5]) << 8 | p[6];
      return w >= 1 && h >= 1;
    }
    p += len;
  }
}

}  // namespace

void set_max_image_pixels(int64_t pixels) { g_max_pixels.store(std::clamp<int64_t>(pixels, 1, 178956970)); }

int64_t max_image_pixels() { return g_max_pixels.load(std::memory_order_relaxed); }

bool probe_image(std::string_view bytes, int64_t& width, int64_t& height, std::string& err) {
  width = height = 0;
  if (bytes.size() < 8) {
    err = "image data too short";
    return false;
  }
  bool known = false;
  if (starts_with(bytes, "GIF87a") || starts_with(bytes, "GIF89a")) {
    known = gif_size(bytes, width, height);
  } else if (starts_with(bytes, "\xFF\xD8\xFF")) {
    known = jpeg_size(bytes, width, height);
  } else if (starts_with(bytes, "\x89PNG\r\n\x1a\n") && bytes.size() >= 24 && bytes.substr(12, 4) == "IHDR") {
    auto be32 = [&](size_t i) {
      return int64_t(uint8_t(bytes[i])) << 24 | int64_t(uint8_t(bytes[i + 1])) << 16 |
             int64_t(uint8_t(bytes[i + 2])) << 8 | int64_t(uint8_t(bytes[i + 3]));
    };
    width = be32(16);
    height = be32(20);
    known = width >= 1 && height >= 1;
  }
  if (!known && !detail::imageio_size(bytes, width, height, err)) return false;
  if (width < 1 || height < 1) {
    err = "bad image size";
    return false;
  }
  if (exif_orientation_enabled()) {
    const int o = exif_orientation(bytes);
    if (o >= 5 && o <= 8) std::swap(width, height);  // transposed by apply_orientation
  }
  return true;
}

bool decode_data_url(std::string_view url, std::string& bytes, std::string& err) {
  if (!starts_with(url, "data:")) {
    err = "not a data URL";
    return false;
  }
  const size_t comma = url.find(',');
  const std::string_view head = url.substr(0, comma);  // whole string when no comma
  const std::string_view payload = comma == std::string_view::npos ? std::string_view{} : url.substr(comma + 1);
  if (head.find(";base64") == std::string_view::npos) {
    err = "only base64 data URLs are supported";
    return false;
  }
  return base64_decode(payload, bytes, err);
}

// Decoding only (no EXIF orientation).
static bool decode_image_raw(std::string_view bytes, RGBImage& out, std::string& err) {
  out = RGBImage{};
  if (bytes.size() < 8) {
    err = "image data too short";
    return false;
  }
  detail::ImageIORules rules;
  if (starts_with(bytes, "GIF87a") || starts_with(bytes, "GIF89a")) {
    if (detail::gif_decode(bytes, out, err)) return true;
    // Malformed GIFs (which PIL rejects) still get ImageIO's best effort.
  } else if (starts_with(bytes, "\xFF\xD8\xFF")) {
    bool unsupported = false;
    if (detail::jpeg_decode(bytes, out, err, unsupported)) return true;
    rules.cmyk_inverted = true;  // PIL reads CMYK JPEGs as Adobe-inverted
    rules.software_only = true;
  } else if (starts_with(bytes, "\x89PNG\r\n\x1a\n") && bytes.size() >= 26 &&
             bytes.substr(12, 4) == "IHDR") {
    // PIL opens 16-bit grey (colour type 0) as I;16, whose RGB conversion
    // clips at 255; 16-bit grey+alpha (type 4) keeps the high byte.
    rules.gray16_clip = uint8_t(bytes[25]) == 0;
  }
  out = RGBImage{};
  std::string e2;
  if (!detail::imageio_decode(bytes, rules, out, e2)) {
    err = err.empty() ? e2 : err + "; " + e2;
    return false;
  }
  if (detail::over_pixel_cap(out.width, out.height)) {  // the decode cap (PIL's DecompressionBombError at most)
    err = "image too large";
    out = RGBImage{};
    return false;
  }
  err.clear();
  return true;
}

bool decode_image(std::string_view bytes, RGBImage& out, std::string& err) {
  if (!decode_image_raw(bytes, out, err)) return false;
  if (exif_orientation_enabled()) {
    const int o = exif_orientation(bytes);
    if (o >= 2 && o <= 8) out = apply_orientation(out, o);
  }
  return true;
}

bool load_rgb(std::string_view src, RGBImage& out, std::string& err) {
  if (starts_with(src, "data:")) {
    std::string bytes;
    if (!decode_data_url(src, bytes, err)) return false;
    return decode_image(bytes, out, err);
  }
  return decode_image(src, out, err);
}

bool load_rgb_file(const std::string& path, RGBImage& out, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open " + path;
    return false;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return decode_image(ss.str(), out, err);
}

}  // namespace espetero::vision
