// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Hand-written matchers for the byte-level BPE pre-tokenizer regexes (the Split
// step with behavior "Isolated"). No regex engine: each supported pattern is
// recognized by its exact source string and implemented as a small state
// machine with the same semantics as Oniguruma in Hugging Face `tokenizers`:
// at each position the first alternative that matches wins (ordered
// alternation), quantifiers are greedy and backtrack where it matters.
//
// All supported patterns share this shape (Qwen3.5 variant shown):
//   1  (?i:'s|'t|'re|'ve|'m|'ll|'d)           contractions; (?i) also folds U+017F to 's'
//   2  [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+          one optional prefix char + a letter run
//   3  \p{N}                                    one digit ({1,3} for Llama-3)
//   4   ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*          optional space + symbol run + newlines
//   5  \s*[\r\n]+                               whitespace ending at the last CR/LF of the run
//   6  \s+(?!\S)                                whitespace run minus its last char (unless at end)
//   7  \s+                                      whitespace run
// Qwen2 / Llama-3 use \p{L}+ in 2 and [^\s\p{L}\p{N}] in 4 (marks are symbols).
// Every code point can start one of 2, 3, 4 or 7, so the pieces tile the input.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include "unicode.h"

namespace espetero::tok {

enum class SplitPattern : uint8_t { kQwen2, kQwen35, kLlama3 };

// Exact pattern strings as they appear in tokenizer.json.
std::optional<SplitPattern> split_pattern_from_regex(std::string_view regex);
// llama.cpp `tokenizer.ggml.pre` names.
std::optional<SplitPattern> split_pattern_from_gguf_pre(std::string_view pre);
std::string_view split_pattern_name(SplitPattern p);

class PieceSplitter {
 public:
  PieceSplitter(SplitPattern pat, std::string_view s)
      : p_(reinterpret_cast<const unsigned char*>(s.data())),
        b_(p_),
        e_(p_ + s.size()),
        marks_are_letters_(pat == SplitPattern::kQwen35),
        max_digits_(pat == SplitPattern::kLlama3 ? 3 : 1) {}

  // Yields the next piece as byte offsets [begin, end) into s; false at the end.
  bool next(size_t& begin, size_t& end) {
    if (p_ >= e_) return false;
    const unsigned char* q = match(p_);
    begin = size_t(p_ - b_);
    end = size_t(q - b_);
    p_ = q;
    return true;
  }

 private:
  struct Ch {
    uint32_t cp;
    uint8_t cls;  // ucd::kOther ... ucd::kSpace
    uint8_t len;
  };

  Ch at(const unsigned char* p) const {
    if (*p < 0x80) return {*p, uint8_t(ucd::kAscii[*p] & ucd::kClassMask), 1};
    uint32_t cp;
    const int len = decode_utf8(p, e_, cp);
    // Invalid bytes behave like symbols ("other"): they are never letters,
    // digits or spaces, and they keep the byte-level round trip intact.
    return {cp, uint8_t(cp == kInvalidCp ? ucd::kOther : cp_class(cp)), uint8_t(len)};
  }
  bool letter(uint8_t cls) const { return cls == ucd::kLetter || (marks_are_letters_ && cls == ucd::kMark); }
  bool symbol(uint8_t cls) const { return cls == ucd::kOther || (!marks_are_letters_ && cls == ucd::kMark); }
  static bool crlf(uint32_t cp) { return cp == '\r' || cp == '\n'; }

  const unsigned char* letter_run(const unsigned char* p) const {
    while (p < e_) {
      const Ch c = at(p);
      if (!letter(c.cls)) break;
      p += c.len;
    }
    return p;
  }

  static int fold_ascii(unsigned c) { return (c >= 'A' && c <= 'Z') ? int(c + 32) : int(c); }

  // Length of a contraction at p (p[0] == '\''), 0 if none.
  int contraction(const unsigned char* p) const {
    const ptrdiff_t n = e_ - p;
    if (n < 2) return 0;
    // (?i:'s): Oniguruma's case folding also matches U+017F LATIN SMALL LETTER LONG S.
    if (n >= 3 && p[1] == 0xC5 && p[2] == 0xBF) return 3;
    switch (fold_ascii(p[1])) {
      case 's':
      case 't':
      case 'm':
      case 'd':
        return 2;
      case 'r':
      case 'v':
        return (n >= 3 && fold_ascii(p[2]) == 'e') ? 3 : 0;
      case 'l':
        return (n >= 3 && fold_ascii(p[2]) == 'l') ? 3 : 0;
      default:
        return 0;
    }
  }

  const unsigned char* match(const unsigned char* p) const {
    const Ch c = at(p);
    const unsigned char* after = p + c.len;
    // 1. contractions
    if (c.cp == '\'') {
      if (const int k = contraction(p)) return p + k;
    }
    // 2. [^\r\n\p{L}\p{N}]? letter-run: try with the optional prefix first.
    if (c.cls != ucd::kLetter && c.cls != ucd::kNumber && !crlf(c.cp) && after < e_ && letter(at(after).cls))
      return letter_run(after);
    if (letter(c.cls)) return letter_run(after);
    // 3. digits
    if (c.cls == ucd::kNumber) {
      const unsigned char* q = after;
      for (int k = 1; k < max_digits_ && q < e_; ++k) {
        const Ch d = at(q);
        if (d.cls != ucd::kNumber) break;
        q += d.len;
      }
      return q;
    }
    // 4. " ?" symbol-run then [\r\n]*
    const unsigned char* q = nullptr;
    if (symbol(c.cls)) q = after;
    else if (c.cp == ' ' && after < e_ && symbol(at(after).cls)) q = after + at(after).len;
    if (q) {
      while (q < e_) {
        const Ch d = at(q);
        if (!symbol(d.cls)) break;
        q += d.len;
      }
      while (q < e_ && (*q == '\r' || *q == '\n')) ++q;
      return q;
    }
    // 5-7. whitespace (c.cls == kSpace here: every other class matched above).
    const unsigned char* last_nl_end = crlf(c.cp) ? after : nullptr;
    const unsigned char* last_start = p;
    q = after;
    while (q < e_) {
      const Ch d = at(q);
      if (d.cls != ucd::kSpace) break;
      last_start = q;
      q += d.len;
      if (crlf(d.cp)) last_nl_end = q;
    }
    if (last_nl_end) return last_nl_end;           // 5. \s*[\r\n]+
    if (q == e_ || last_start == p) return q;       // 6. at end of text, or 7. single space
    return last_start;                              // 6. leave the last space for the next piece
  }

  const unsigned char* p_;
  const unsigned char* b_;
  const unsigned char* e_;
  bool marks_are_letters_;
  int max_digits_;
};

}  // namespace espetero::tok
