// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Unicode support for the tokenizer: UTF-8 decoding that keeps invalid bytes
// visible, code point classes for the pre-tokenizer regexes, and NFC.
//
// Classes and NFC follow the reference implementation (Hugging Face
// `tokenizers`), not the newest Unicode: see tools/gen_unicode.py.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "unicode_data.h"

namespace espetero::tok {

// Returned by decode_utf8 for a byte that does not start a valid sequence.
inline constexpr uint32_t kInvalidCp = 0xFFFFFFFFu;

// Decodes the code point at p (p < end). Returns its length in bytes (1-4).
// Invalid, overlong, surrogate or truncated sequences yield kInvalidCp with
// length 1, so every byte of malformed input is consumed exactly once.
inline int decode_utf8(const unsigned char* p, const unsigned char* end, uint32_t& cp) {
  const unsigned c = p[0];
  if (c < 0x80) {
    cp = c;
    return 1;
  }
  const ptrdiff_t avail = end - p;
  if (c >= 0xC2 && c <= 0xDF) {
    if (avail >= 2 && (p[1] & 0xC0) == 0x80) {
      cp = ((c & 0x1Fu) << 6) | (p[1] & 0x3Fu);
      return 2;
    }
  } else if (c >= 0xE0 && c <= 0xEF) {
    if (avail >= 3 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80) {
      const uint32_t v = ((c & 0x0Fu) << 12) | ((p[1] & 0x3Fu) << 6) | (p[2] & 0x3Fu);
      if (v >= 0x800 && (v < 0xD800 || v > 0xDFFF)) {
        cp = v;
        return 3;
      }
    }
  } else if (c >= 0xF0 && c <= 0xF4) {
    if (avail >= 4 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80 && (p[3] & 0xC0) == 0x80) {
      const uint32_t v = ((c & 0x07u) << 18) | ((p[1] & 0x3Fu) << 12) | ((p[2] & 0x3Fu) << 6) | (p[3] & 0x3Fu);
      if (v >= 0x10000 && v <= 0x10FFFF) {
        cp = v;
        return 4;
      }
    }
  }
  cp = kInvalidCp;
  return 1;
}

inline void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(char(cp));
  } else if (cp < 0x800) {
    out.push_back(char(0xC0 | (cp >> 6)));
    out.push_back(char(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(char(0xE0 | (cp >> 12)));
    out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(char(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(char(0xF0 | (cp >> 18)));
    out.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(char(0x80 | (cp & 0x3F)));
  }
}

// Flags (ucd::kLetter ... | ucd::kNfcUnsafe | ucd::kNfcBarrier) of a code point.
// Invalid bytes (kInvalidCp) are "other" and act as NFC barriers.
inline uint8_t cp_flags(uint32_t cp) {
  if (cp < 0x80) return ucd::kAscii[cp];
  if (cp > 0x10FFFF) return ucd::kOther | ucd::kNfcBarrier;
  // Last run whose first code point is <= cp; kRanges[0] starts at U+0000.
  const uint32_t key = (cp << 8) | 0xFFu;
  uint32_t lo = 1, hi = ucd::kNumRanges;
  while (lo < hi) {
    const uint32_t mid = (lo + hi) >> 1;
    if (ucd::kRanges[mid] <= key) lo = mid + 1;
    else hi = mid;
  }
  return uint8_t(ucd::kRanges[lo - 1] & 0xFFu);
}

inline uint8_t cp_class(uint32_t cp) { return cp_flags(cp) & ucd::kClassMask; }

// NFC as the reference computes it. Returns false when `in` is already
// normalized (ASCII and almost all real text: no allocation, `out` untouched);
// otherwise writes the normalized text to `out` and returns true. Invalid UTF-8
// bytes are passed through unchanged. Thread-safe.
bool nfc_normalize(std::string_view in, std::string& out);

}  // namespace espetero::tok
