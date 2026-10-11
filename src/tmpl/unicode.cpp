// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "unicode.h"

#include <algorithm>

#include "../util/json.h"

namespace espetero::tmpl::uni {

namespace {

struct CpRange {
  uint32_t lo, hi;
};
// Code points lo, lo+stride, ..., hi map to cp + delta.
struct CaseRun {
  uint32_t lo, hi;
  int32_t delta;
  uint32_t stride;
};
// Mappings to several code points (e.g. 'ß'.upper() == 'SS').
struct CaseFull {
  uint32_t cp;
  uint32_t m[3];
};

#include "unicode_data.inc"

template <size_t N>
bool in_ranges(const CpRange (&t)[N], uint32_t c) {
  // Last range whose lo <= c.
  const CpRange* it = std::upper_bound(t, t + N, c, [](uint32_t v, const CpRange& r) { return v < r.lo; });
  return it != t && c <= (it - 1)->hi;
}

template <size_t R, size_t F>
void append_mapped(std::string& out, uint32_t c, const CaseRun (&runs)[R], const CaseFull (&full)[F]) {
  if constexpr (F > 0) {
    const CaseFull* f = std::lower_bound(full, full + F, c, [](const CaseFull& e, uint32_t v) { return e.cp < v; });
    if (f != full + F && f->cp == c) {
      for (uint32_t m : f->m)
        if (m) append(out, m);
      return;
    }
  }
  const CaseRun* it = std::upper_bound(runs, runs + R, c, [](uint32_t v, const CaseRun& r) { return v < r.lo; });
  if (it != runs) {
    const CaseRun& r = *(it - 1);
    if (c <= r.hi && (c - r.lo) % r.stride == 0) {
      append(out, uint32_t(int64_t(c) + r.delta));
      return;
    }
  }
  append(out, c);
}

// Greek capital sigma lowercases to final sigma at the end of a word
// (CPython handle_capital_sigma): \p{cased}\p{case-ignorable}* Σ !(\p{case-ignorable}* \p{cased}).
uint32_t lower_sigma(std::string_view s, size_t pos, size_t after) {
  // Scan backwards over case-ignorable code points.
  size_t j = pos;
  uint32_t c = 0;
  bool found = false;
  while (j > 0) {
    size_t k = j - 1;
    while (k > 0 && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80) --k;
    size_t t = k;
    c = next(s, t);
    j = k;
    if (!is_case_ignorable(c)) {
      found = true;
      break;
    }
  }
  bool final_sigma = found && is_cased(c);
  if (final_sigma && after < s.size()) {
    size_t t = after;
    bool stop = false;
    while (t < s.size()) {
      c = next(s, t);
      if (!is_case_ignorable(c)) {
        stop = true;
        break;
      }
    }
    final_sigma = !stop || !is_cased(c);
  }
  return final_sigma ? 0x3C2 : 0x3C3;
}

void append_lower_at(std::string& out, std::string_view s, size_t pos, size_t after, uint32_t c) {
  if (c == 0x3A3)
    append(out, lower_sigma(s, pos, after));
  else
    append_lower(out, c);
}

}  // namespace

uint32_t next(std::string_view s, size_t& i) {
  const unsigned char c = static_cast<unsigned char>(s[i]);
  if (c < 0x80) {
    ++i;
    return c;
  }
  return json::utf8_next(s, i);
}

void append(std::string& out, uint32_t cp) {
  if (cp < 0x80)
    out.push_back(char(cp));
  else
    json::utf8_append(out, cp);
}

bool is_ascii(std::string_view s) {
  for (char ch : s)
    if (static_cast<unsigned char>(ch) >= 0x80) return false;
  return true;
}

size_t length(std::string_view s) {
  size_t n = 0;
  for (char ch : s) n += (static_cast<unsigned char>(ch) & 0xC0) != 0x80;
  return n;
}

size_t byte_offset(std::string_view s, size_t cp) {
  size_t i = 0;
  while (cp > 0 && i < s.size()) {
    next(s, i);
    --cp;
  }
  return i;
}

std::string_view slice(std::string_view s, size_t b, size_t e) {
  if (e <= b) return s.substr(0, 0);
  const size_t bo = byte_offset(s, b);
  size_t eo = bo;
  size_t n = e - b;
  while (n > 0 && eo < s.size()) {
    next(s, eo);
    --n;
  }
  return s.substr(bo, eo - bo);
}

bool is_space(uint32_t c) {
  if (c < 0x80) return c == ' ' || (c >= 0x9 && c <= 0xD) || (c >= 0x1C && c <= 0x1F);
  return in_ranges(kSpace, c);
}
bool is_printable(uint32_t c) {
  if (c < 0x80) return c >= 0x20 && c < 0x7F;
  return !in_ranges(kNonPrintable, c);
}
bool is_alpha(uint32_t c) {
  if (c < 0x80) return (c | 0x20) >= 'a' && (c | 0x20) <= 'z';
  return in_ranges(kAlpha, c);
}
bool is_decimal(uint32_t c) { return c < 0x80 ? (c >= '0' && c <= '9') : in_ranges(kDecimal, c); }
bool is_digit(uint32_t c) { return c < 0x80 ? (c >= '0' && c <= '9') : in_ranges(kDigit, c); }
bool is_numeric(uint32_t c) { return c < 0x80 ? (c >= '0' && c <= '9') : in_ranges(kNumeric, c); }
bool is_alnum(uint32_t c) { return is_alpha(c) || is_decimal(c) || is_digit(c) || is_numeric(c); }
bool is_word(uint32_t c) { return c == '_' || is_alnum(c); }
bool is_lower(uint32_t c) { return c < 0x80 ? (c >= 'a' && c <= 'z') : in_ranges(kLower, c); }
bool is_upper(uint32_t c) { return c < 0x80 ? (c >= 'A' && c <= 'Z') : in_ranges(kUpper, c); }
bool is_titlecase(uint32_t c) { return c >= 0x80 && in_ranges(kTitle, c); }
bool is_cased(uint32_t c) { return c < 0x80 ? ((c | 0x20) >= 'a' && (c | 0x20) <= 'z') : in_ranges(kCased, c); }
bool is_case_ignorable(uint32_t c) { return in_ranges(kCaseIgnorable, c); }

void append_upper(std::string& out, uint32_t c) {
  if (c < 0x80) {
    out.push_back(char(c >= 'a' && c <= 'z' ? c - 32 : c));
    return;
  }
  append_mapped(out, c, kUpperRuns, kUpperFull);
}
void append_lower(std::string& out, uint32_t c) {
  if (c < 0x80) {
    out.push_back(char(c >= 'A' && c <= 'Z' ? c + 32 : c));
    return;
  }
  append_mapped(out, c, kLowerRuns, kLowerFull);
}
void append_title(std::string& out, uint32_t c) {
  if (c < 0x80) {
    out.push_back(char(c >= 'a' && c <= 'z' ? c - 32 : c));
    return;
  }
  append_mapped(out, c, kTitleRuns, kTitleFull);
}

std::string upper(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) append_upper(out, next(s, i));
  return out;
}

std::string lower(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    const size_t pos = i;
    const uint32_t c = next(s, i);
    append_lower_at(out, s, pos, i, c);
  }
  return out;
}

std::string capitalize(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  if (i < s.size()) append_title(out, next(s, i));
  while (i < s.size()) {
    const size_t pos = i;
    const uint32_t c = next(s, i);
    append_lower_at(out, s, pos, i, c);
  }
  return out;
}

std::string title(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  bool prev_cased = false;
  size_t i = 0;
  while (i < s.size()) {
    const size_t pos = i;
    const uint32_t c = next(s, i);
    if (prev_cased)
      append_lower_at(out, s, pos, i, c);
    else
      append_title(out, c);
    prev_cased = is_cased(c);
  }
  return out;
}

std::string swapcase(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    const size_t pos = i;
    const uint32_t c = next(s, i);
    if (is_upper(c))
      append_lower_at(out, s, pos, i, c);
    else if (is_lower(c))
      append_upper(out, c);
    else
      append(out, c);
  }
  return out;
}

}  // namespace espetero::tmpl::uni
