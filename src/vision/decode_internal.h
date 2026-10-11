// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Internal decoder entry points shared by decode.cpp and the format backends.
#pragma once

#include <string>
#include <string_view>

#include "vision.h"

namespace espetero::vision::detail {

// How to map what ImageIO returns onto PIL's convert("RGB") for the source
// format (ImageIO does not say how the file stored its samples).
struct ImageIORules {
  // JPEG: PIL reads CMYK JPEGs as Adobe-inverted ("CMYK;I"); ImageIO hands
  // back the stored bytes, so ink = 255 - sample. Other formats: ink = sample.
  bool cmyk_inverted = false;
  // 16-bit grey that PIL opens as mode I;16 (PNG colour type 0, TIFF): its RGB
  // conversion clips at 255. Grey+alpha 16-bit PNGs (colour type 4) are read
  // through "LA;16B" instead, which keeps the high byte.
  bool gray16_clip = true;
  // JPEG: ImageIO's software decoder is much closer to libjpeg-turbo than the
  // hardware one it picks for large images (see jpeg.cpp for the numbers).
  bool software_only = false;
};

bool imageio_decode(std::string_view bytes, const ImageIORules& rules, RGBImage& out,
                    std::string& err);

// Width and height of the first image from ImageIO's properties (the header
// only: no pixel is decoded).
bool imageio_size(std::string_view bytes, int64_t& width, int64_t& height, std::string& err);

// True when a w x h image is over the decode cap, without overflow (empty or
// negative sizes are the callers' own errors).
inline bool over_pixel_cap(int64_t w, int64_t h) { return h > 0 && w > max_image_pixels() / h; }

// GIF first frame with PIL's semantics (gif.cpp).
bool gif_decode(std::string_view bytes, RGBImage& out, std::string& err);

// libjpeg-turbo compatible baseline / progressive decoder (jpeg.cpp).
// Returns false with `unsupported` set when the file needs a feature it does
// not implement (arithmetic coding, 12-bit, lossless...), so the caller can
// fall back to ImageIO.
bool jpeg_decode(std::string_view bytes, RGBImage& out, std::string& err, bool& unsupported);

// PIL's CMYK -> RGB (Convert.c cmyk2rgb): nk = 255 - k; r = nk - round(c * nk / 255).
inline uint8_t pil_cmyk_channel(int c, int k) {
  const int nk = 255 - k;
  const int t = c * nk + 128;
  const int v = nk - (((t >> 8) + t) >> 8);
  return uint8_t(v < 0 ? 0 : v > 255 ? 255 : v);
}

}  // namespace espetero::vision::detail
