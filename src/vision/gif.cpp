// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// GIF first frame, decoded the way PIL 12 does it (GifImagePlugin + GifDecode.c
// + convert("RGB")). ImageIO cannot be used here: it composites the frame
// onto a transparent canvas and zeroes the colour of transparent pixels,
// while PIL keeps the palette colour of the transparency index.
//
// PIL's frame 0:
//   * canvas = logical screen, enlarged if the frame extends past it;
//   * canvas filled with the frame's transparency index if its Graphic
//     Control Extension sets one, else with index 0;
//   * the frame's LZW indices are written into its rectangle (transparent
//     indices included, interlaced rows in GIF pass order);
//   * palette = local table if present, else the global one; a table that is
//     exactly the identity grey ramp is dropped (mode "L", same result);
//     without any table the index is the grey level;
//   * convert("RGB") looks indices up in the palette; entries past the end of
//     a short table are black.
#include <cstring>
#include <vector>

#include "decode_internal.h"

namespace espetero::vision::detail {
namespace {

class Reader {
 public:
  explicit Reader(std::string_view s) : s_(s) {}
  bool eof() const { return pos_ >= s_.size(); }
  size_t pos() const { return pos_; }
  // Missing bytes read as end of data (PIL's fp.read returns short).
  int byte() { return pos_ < s_.size() ? uint8_t(s_[pos_++]) : -1; }
  std::string_view take(size_t n) {
    const size_t k = std::min(n, s_.size() - std::min(pos_, s_.size()));
    std::string_view v = s_.substr(pos_, k);
    pos_ += k;
    return v;
  }
  // PIL's GifImageFile.data(): one sub-block, empty at a terminator or EOF.
  std::string_view block() {
    const int n = byte();
    if (n <= 0) return {};
    return take(size_t(n));
  }

 private:
  std::string_view s_;
  size_t pos_ = 0;
};

bool identity_gray(std::string_view p) {
  for (size_t i = 0; i + 2 < p.size(); i += 3)
    if (uint8_t(p[i]) != i / 3 || uint8_t(p[i + 1]) != i / 3 || uint8_t(p[i + 2]) != i / 3) return false;
  return true;
}

// LZW-decodes into the frame rectangle of `canvas` (canvas_w wide).
// Returns false on a corrupt code stream.
bool lzw_decode(Reader& rd, int min_bits, bool interlace, uint8_t* canvas, int canvas_w, int x0,
                int y0, int fw, int fh, std::string& err) {
  if (min_bits < 1 || min_bits > 11) {
    err = "GIF: bad LZW code size";
    return false;
  }
  const int clear = 1 << min_bits, end = clear + 1;
  constexpr int kMax = 4096;
  std::vector<uint16_t> prefix(kMax);
  std::vector<uint8_t> suffix(kMax), first(kMax);
  std::vector<uint8_t> stack(kMax + 1);
  for (int i = 0; i < clear; i++) {
    suffix[i] = uint8_t(i);
    first[i] = uint8_t(i);
  }
  int code_size = min_bits + 1, next = clear + 2, prev = -1;
  uint32_t bitbuf = 0;
  int nbits = 0;
  std::string_view blk;
  size_t blkpos = 0;

  // Output cursor over the frame rectangle, rows in interlace order.
  int x = 0, y = 0, pass = interlace ? 1 : 0, step = interlace ? 8 : 1;
  bool done = fh == 0 || fw == 0;
  auto newline = [&]() {
    x = 0;
    y += step;
    while (y >= fh && !done) {
      switch (pass) {
        case 1: y = 4; pass = 2; break;
        case 2: step = 4; y = 2; pass = 3; break;
        case 3: step = 2; y = 1; pass = 0; break;
        default: done = true;
      }
    }
  };
  auto emit = [&](uint8_t v) {
    if (done) return;
    canvas[size_t(y0 + y) * canvas_w + size_t(x0 + x)] = v;
    if (++x >= fw) newline();
  };

  while (!done) {
    // Next code, least significant bit first, across sub-blocks.
    while (nbits < code_size) {
      if (blkpos >= blk.size()) {
        blk = rd.block();
        blkpos = 0;
        if (blk.empty()) return true;  // data ended: rest keeps the fill value
      }
      bitbuf |= uint32_t(uint8_t(blk[blkpos++])) << nbits;
      nbits += 8;
    }
    const int c = int(bitbuf & ((1u << code_size) - 1));
    bitbuf >>= code_size;
    nbits -= code_size;

    if (c == clear) {
      code_size = min_bits + 1;
      next = clear + 2;
      prev = -1;
      continue;
    }
    if (c == end) return true;
    if (prev < 0) {
      if (c >= clear) {
        err = "GIF: corrupt LZW data";
        return false;
      }
      emit(uint8_t(c));
      prev = c;
      continue;
    }
    int cur = c, sp = 0;
    uint8_t lead;
    if (c < next) {
      lead = first[c];
    } else if (c == next && next < kMax) {  // KwKwK: previous string + its first byte
      lead = first[prev];
      stack[sp++] = lead;
      cur = prev;
    } else {
      err = "GIF: corrupt LZW data";
      return false;
    }
    while (cur >= clear) {
      stack[sp++] = suffix[cur];
      cur = prefix[cur];
    }
    stack[sp++] = uint8_t(cur);
    while (sp > 0) emit(stack[--sp]);
    if (next < kMax) {
      prefix[next] = uint16_t(prev);
      suffix[next] = lead;
      first[next] = first[prev];
      next++;
      if (next == (1 << code_size) && code_size < 12) code_size++;
    }
    prev = c;
  }
  return true;
}

}  // namespace

bool gif_decode(std::string_view bytes, RGBImage& out, std::string& err) {
  Reader rd(bytes);
  const std::string_view head = rd.take(13);
  if (head.size() < 13 || (head.substr(0, 6) != "GIF87a" && head.substr(0, 6) != "GIF89a")) {
    err = "not a GIF file";
    return false;
  }
  auto u16 = [](std::string_view s, size_t i) { return int(uint8_t(s[i]) | uint8_t(s[i + 1]) << 8); };
  int width = u16(head, 6), height = u16(head, 8);
  const int flags = uint8_t(head[10]);
  std::string_view global;
  if (flags & 128) {
    global = rd.take(size_t(3) << ((flags & 7) + 1));
    if (identity_gray(global)) global = {};
  }

  // Extensions up to the first image descriptor.
  int transparency = -1;
  for (;;) {
    const int s = rd.byte();
    if (s < 0 || s == ';') {
      err = "image not found in GIF frame";
      return false;
    }
    if (s == '!') {
      const int label = rd.byte();
      std::string_view blk = rd.block();
      if (label == 0xF9 && !blk.empty()) {
        if (blk.size() < 4) {
          err = "GIF: truncated graphic control extension";
          return false;
        }
        if (uint8_t(blk[0]) & 1) transparency = uint8_t(blk[3]);
      } else if (label == 0xFE) {
        while (!blk.empty()) blk = rd.block();  // comment: already consumed its terminator
        continue;
      } else if (label == 0xFF && blk.substr(0, 11) == "NETSCAPE2.0") {
        rd.block();
      }
      while (!rd.block().empty()) {
      }
      continue;
    }
    if (s != ',') continue;  // PIL skips unknown bytes between blocks

    const std::string_view d = rd.take(9);
    if (d.size() < 9) {
      err = "GIF: truncated image descriptor";
      return false;
    }
    const int x0 = u16(d, 0), y0 = u16(d, 2), fw = u16(d, 4), fh = u16(d, 6);
    width = std::max(width, x0 + fw);
    height = std::max(height, y0 + fh);
    const int dflags = uint8_t(d[8]);
    std::string_view palette = global;
    bool has_palette = !global.empty();
    if (dflags & 128) {
      const std::string_view local = rd.take(size_t(3) << ((dflags & 7) + 1));
      has_palette = !identity_gray(local);
      palette = has_palette ? local : std::string_view{};
    }
    const int min_bits = rd.byte();
    if (width < 1 || height < 1) {
      err = "GIF: bad image size";
      return false;
    }
    if (over_pixel_cap(width, height)) {  // before the canvas exists: 13 bytes can declare 65535 x 65535
      err = "GIF: image too large";
      return false;
    }
    std::vector<uint8_t> canvas(size_t(width) * height, uint8_t(transparency >= 0 ? transparency : 0));
    if (!lzw_decode(rd, min_bits, (dflags & 64) != 0, canvas.data(), width, x0, y0, fw, fh, err))
      return false;

    uint8_t lut[256][3];
    for (int i = 0; i < 256; i++) {
      if (!has_palette) {
        lut[i][0] = lut[i][1] = lut[i][2] = uint8_t(i);
      } else if (size_t(i) * 3 + 3 <= palette.size()) {
        std::memcpy(lut[i], palette.data() + i * 3, 3);
      } else {
        lut[i][0] = lut[i][1] = lut[i][2] = 0;
      }
    }
    out.width = width;
    out.height = height;
    out.rgb.resize(canvas.size() * 3);
    for (size_t i = 0; i < canvas.size(); i++) std::memcpy(&out.rgb[i * 3], lut[canvas[i]], 3);
    return true;
  }
}

}  // namespace espetero::vision::detail
