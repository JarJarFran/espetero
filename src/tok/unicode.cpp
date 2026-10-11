// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "unicode.h"

#include <cstring>
#include <vector>

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#endif

namespace espetero::tok {

namespace {

// Spans that need normalization are merged into one CoreFoundation call when
// they are closer than kMergeGap bytes (each call has ~1-2 us of fixed cost),
// but a merged region stops growing at kMaxRegion bytes: CFStringNormalize edits
// the string in place and becomes quadratic on long inputs (~30 ns per UTF-16
// unit up to ~1K units, ~340 ns per unit at 64K).
constexpr ptrdiff_t kMergeGap = 256;
constexpr ptrdiff_t kMaxRegion = 1024;

inline void push_utf16(std::vector<uint16_t>& v, uint32_t cp) {
  if (cp < 0x10000) {
    v.push_back(uint16_t(cp));
  } else {
    cp -= 0x10000;
    v.push_back(uint16_t(0xD800 + (cp >> 10)));
    v.push_back(uint16_t(0xDC00 + (cp & 0x3FF)));
  }
}

// Appends NFC([p, e)) to out. The span holds only valid UTF-8 and no barrier
// code points (the caller guarantees it), so CoreFoundation and the reference
// agree on every code point in it.
void cf_nfc_append(const unsigned char* p, const unsigned char* e, std::string& out) {
#if defined(__APPLE__)
  // Spans are normally < ~1 KB; only one long run of marks makes a big buffer,
  // which is then not kept per thread.
  thread_local std::vector<uint16_t> tl_u16;
  std::vector<uint16_t> big_u16;
  std::vector<uint16_t>& u16 = e - p <= 4096 ? tl_u16 : big_u16;
  u16.clear();
  while (p < e) {
    uint32_t cp;
    p += decode_utf8(p, e, cp);
    push_utf16(u16, cp);
  }
  // UTF-16 in and out: CFStringCreateWithBytes(UTF-8) would drop a leading
  // U+FEFF as a byte order mark. FormD before FormC: CF's FormC alone skips some
  // compositions whose first element arrives precomposed (U+AC00 U+11A8,
  // U+0CCA U+0CD5) and mis-orders the decomposition of U+0F73/U+0F75/U+0F81;
  // on fully decomposed input it matches the reference (tools/gen_unicode.py
  // checks this on ~4.5M probe strings).
  CFMutableStringRef m = CFStringCreateMutable(kCFAllocatorDefault, 0);
  CFStringAppendCharacters(m, reinterpret_cast<const UniChar*>(u16.data()), CFIndex(u16.size()));
  CFStringNormalize(m, kCFStringNormalizationFormD);
  CFStringNormalize(m, kCFStringNormalizationFormC);
  const CFIndex n = CFStringGetLength(m);
  u16.resize(size_t(n));
  CFStringGetCharacters(m, CFRangeMake(0, n), reinterpret_cast<UniChar*>(u16.data()));
  CFRelease(m);
  for (size_t i = 0; i < u16.size(); ++i) {
    uint32_t cp = u16[i];
    if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < u16.size()) {
      cp = 0x10000 + ((cp - 0xD800) << 10) + (uint32_t(u16[i + 1]) - 0xDC00);
      ++i;
    }
    append_utf8(out, cp);
  }
#else
  // No normalizer available off Apple platforms: pass text through unchanged.
  out.append(reinterpret_cast<const char*>(p), size_t(e - p));
#endif
}

}  // namespace

bool nfc_normalize(std::string_view in, std::string& out) {
  const auto* const b = reinterpret_cast<const unsigned char*>(in.data());
  const auto* const e = b + in.size();

  // Quick check: text is NFC unless it contains an "unsafe" code point (nonzero
  // combining class or NFC_Quick_Check != Yes). ASCII is skipped 8 bytes at a time.
  const unsigned char* p = b;
  while (p < e) {
    if (e - p >= 8) {
      uint64_t w;
      std::memcpy(&w, p, 8);
      if ((w & 0x8080808080808080ull) == 0) {
        p += 8;
        continue;
      }
    }
    if (*p < 0x80) {
      ++p;
      continue;
    }
    uint32_t cp;
    const int len = decode_utf8(p, e, cp);
    // Everything below U+0300 (the first combining mark) is safe.
    if (cp >= 0x300 && cp != kInvalidCp && (cp_flags(cp) & ucd::kNfcUnsafe)) break;
    p += len;
  }
  if (p == e) return false;

  // Slow path. NFC never reaches across a "boundary" code point (a safe starter),
  // so only spans from the safe starter before an unsafe code point up to the next
  // safe one need normalizing. Barriers (code points the reference normalizer does
  // not know) and invalid bytes are boundaries too and are never handed to CF.
  out.clear();
  out.reserve(in.size() + 16);
  const unsigned char* copied = b;       // input up to here has been emitted
  const unsigned char* last_safe = nullptr;  // last safe starter in the current barrier-free run
  const unsigned char* span_begin = nullptr;  // pending span to normalize, if any
  const unsigned char* span_end = nullptr;
  auto flush = [&]() {
    if (!span_begin) return;
    out.append(reinterpret_cast<const char*>(copied), size_t(span_begin - copied));
    cf_nfc_append(span_begin, span_end, out);
    copied = span_end;
    span_begin = span_end = nullptr;
  };
  // Rescan from the start (the quick check skipped only safe code points) to
  // find the safe starter before each unsafe code point.
  p = b;
  while (p < e) {
    if (*p < 0x80) {
      last_safe = p++;
      continue;
    }
    uint32_t cp;
    const int len = decode_utf8(p, e, cp);
    const uint8_t f = cp < 0x300 ? 0 : cp_flags(cp);
    if (cp == kInvalidCp || (f & ucd::kNfcBarrier)) {
      flush();
      last_safe = nullptr;
    } else if (f & ucd::kNfcUnsafe) {
      const unsigned char* start = last_safe ? last_safe : p;
      // A new span may only be cut off at a boundary: when a safe starter separates
      // it from the pending region (start >= span_end).
      if (span_begin && start >= span_end && (start - span_end > kMergeGap || start - span_begin > kMaxRegion))
        flush();
      if (!span_begin) span_begin = start;
      span_end = p + len;
    } else {
      last_safe = p;
    }
    p += len;
  }
  flush();
  out.append(reinterpret_cast<const char*>(copied), size_t(e - copied));
  return true;
}

}  // namespace espetero::tok
