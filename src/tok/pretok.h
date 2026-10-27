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
// Qwen2 / Llama-3 use \p{L}+ in 2 and [^\s\p{L}\p{N}] in 4 (marks are symbols);
// LFM2's pattern is Llama-3's (pretok.cpp).
// Every code point can start one of 2, 3, 4 or 7, so the pieces tile the input.
//
// Spark-X2.5 / X2.6 (DeepSeek-V3 style) is a sequence: Split(\p{N}{1,3}),
// Split([一-龥぀-ゟ゠-ヿ]+), Split(the pattern below), Digits(individual), all
// Isolated. Every \p{N} character ends up a piece of its own and CJK runs
// (U+4E00..9FA5, U+3040..30FF) are pieces; the text between them (a segment:
// the regex sees its end as the end of the input) is split by
//   a  [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+    ASCII punctuation + ASCII letters
//   b  [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+               one optional prefix char + a letter run
//   c   ?[\p{P}\p{S}]+                                 optional space + punctuation / symbol run
//   d  [\r\n]                                         one CR or LF
//   e  \s+(?!\S);  f  \s+
// and characters no alternative matches (controls, format characters) are
// pieces of their own runs (the gaps between matches).
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "unicode.h"
#include "unicode_ps.h"

namespace espetero::tok {

enum class SplitPattern : uint8_t { kQwen2, kQwen35, kLlama3, kDeepSeekV3, kLfm2, kSpark };  // kLfm2 splits as kLlama3

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
        spark_(pat == SplitPattern::kSpark),
        max_digits_(pat == SplitPattern::kLlama3 || pat == SplitPattern::kLfm2 ? 3 : 1) {}

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

  static bool cjk(uint32_t cp) { return (cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x30FF); }
  static bool ps(const Ch& c) { return c.cls == ucd::kOther && c.cp != kInvalidCp && ucd::is_punct_symbol(c.cp); }
  static bool ascii_punct(uint32_t cp) {
    return (cp >= '!' && cp <= '/') || (cp >= ':' && cp <= '@') || (cp >= '[' && cp <= '`') || (cp >= '{' && cp <= '~');
  }
  static bool ascii_alpha(uint32_t cp) { return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z'); }
  static bool letter_mark(const Ch& c) { return c.cls == ucd::kLetter || c.cls == ucd::kMark; }

  // Spark: the alternatives a-f at p within a segment ending at se (nullptr: none matches).
  const unsigned char* spark_alt(const unsigned char* p, const unsigned char* se) const {
    const Ch c = at(p);
    const unsigned char* after = p + c.len;
    auto run = [&](const unsigned char* q, auto pred) {
      while (q < se) {
        const Ch d = at(q);
        if (!pred(d)) break;
        q += d.len;
      }
      return q;
    };
    if (ascii_punct(c.cp) && after < se && ascii_alpha(*after))  // a
      return run(after, [](const Ch& d) { return d.cp < 0x80 && ascii_alpha(d.cp); });
    const bool prefix = !crlf(c.cp) && c.cls != ucd::kLetter && !ps(c);
    if (prefix && after < se && letter_mark(at(after))) return run(after, letter_mark);  // b with its prefix
    if (letter_mark(c)) return run(after, letter_mark);                                // b
    if (ps(c)) return run(after, ps);                                                   // c
    if (c.cp == ' ' && after < se && ps(at(after))) return run(after, ps);              // c with the space
    if (crlf(c.cp)) return after;                                                       // d
    if (c.cls == ucd::kSpace) {                                                         // e, f
      const unsigned char* last_start = p;
      const unsigned char* q = after;
      while (q < se) {
        const Ch d = at(q);
        if (d.cls != ucd::kSpace) break;
        last_start = q;
        q += d.len;
      }
      if (q == se || last_start == p) return q;
      return last_start;
    }
    return nullptr;
  }

  const unsigned char* spark_match(const unsigned char* p) const {
    const Ch c = at(p);
    if (c.cls == ucd::kNumber) return p + c.len;  // Digits(individual)
    auto boundary = [&](const Ch& d) { return d.cls == ucd::kNumber || cjk(d.cp); };
    if (cjk(c.cp)) {
      const unsigned char* q = p + c.len;
      while (q < e_) {
        const Ch d = at(q);
        if (!cjk(d.cp)) break;
        q += d.len;
      }
      return q;
    }
    const unsigned char* se = p + c.len;  // the segment's end: the next digit or CJK character
    while (se < e_) {
      const Ch d = at(se);
      if (boundary(d)) break;
      se += d.len;
    }
    if (const unsigned char* q = spark_alt(p, se)) return q;
    const unsigned char* q = p + c.len;  // a gap: up to the next position where an alternative matches
    while (q < se && !spark_alt(q, se)) q += at(q).len;
    return q;
  }

  const unsigned char* match(const unsigned char* p) const {
    if (spark_) return spark_match(p);
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
  bool spark_;
  int max_digits_;
};

// DeepSeek-V3 / V4 pre-tokenizer: three Split(Isolated) steps, each applied to
// the pieces of the previous one (matches and the gaps between them both
// become pieces):
//   1  \p{N}{1,3}
//   2  [一-龥぀-ゟ゠-ヿ]+                         (U+4E00-9FA5, U+3040-309F, U+30A0-30FF)
//   3  [!-/:-@[-`{-~][A-Za-z]+                  ASCII punctuation + ASCII letters
//      | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+   optional prefix + letters/marks
//      |  ?[\p{P}\p{S}]+[\r\n]*                 optional space + punctuation/symbols + newlines
//      | \s*[\r\n]+ | \s+(?!\S) | \s+
// Characters no alternative of step 3 matches (digits, controls, ...) stay
// together as gap pieces. f(begin, end) gets the byte ranges in order.
// \p{P}|\p{S}: the "other" class without controls, format characters and
// private use (unassigned code points count as symbols here).
inline bool ds3_ps(uint32_t cp, uint8_t cls) {
  if (cls != ucd::kOther || cp == kInvalidCp) return false;
  if (cp < 0x20 || (cp >= 0x7F && cp <= 0x9F)) return false;                   // Cc
  if (cp == 0xAD || (cp >= 0x600 && cp <= 0x605) || cp == 0x61C || cp == 0x6DD || cp == 0x70F || cp == 0x890 ||
      cp == 0x891 || cp == 0x8E2 || cp == 0x180E || (cp >= 0x200B && cp <= 0x200F) || (cp >= 0x202A && cp <= 0x202E) ||
      (cp >= 0x2060 && cp <= 0x2064) || (cp >= 0x2066 && cp <= 0x206F) || cp == 0xFEFF || (cp >= 0xFFF9 && cp <= 0xFFFB) ||
      cp == 0x110BD || cp == 0x110CD || (cp >= 0x13430 && cp <= 0x1343F) || (cp >= 0x1BCA0 && cp <= 0x1BCA3) ||
      (cp >= 0x1D173 && cp <= 0x1D17A) || cp == 0xE0001 || (cp >= 0xE0020 && cp <= 0xE007F))
    return false;                                                               // Cf
  if ((cp >= 0xE000 && cp <= 0xF8FF) || cp >= 0xF0000) return false;            // Co
  if (cp == 0x2028 || cp == 0x2029) return false;                               // Zl, Zp
  return true;
}

template <class F>
void deepseek_split(std::string_view s, F&& f) {
  struct C {
    uint32_t cp;
    uint8_t cls;
    uint32_t off;
  };
  const unsigned char* b = reinterpret_cast<const unsigned char*>(s.data());
  const unsigned char* e = b + s.size();
  std::vector<C> c;
  c.reserve(s.size());
  for (const unsigned char* p = b; p < e;) {
    uint32_t cp;
    const int len = decode_utf8(p, e, cp);
    const uint8_t cls = p[0] < 0x80 ? uint8_t(ucd::kAscii[p[0]] & ucd::kClassMask)
                                    : uint8_t(cp == kInvalidCp ? ucd::kOther : cp_class(cp));
    c.push_back({cp, cls, uint32_t(p - b)});
    p += len;
  }
  const size_t n = c.size();
  auto off = [&](size_t i) { return i < n ? size_t(c[i].off) : s.size(); };
  auto cjk = [&](size_t i) {
    const uint32_t x = c[i].cp;
    return (x >= 0x4E00 && x <= 0x9FA5) || (x >= 0x3040 && x <= 0x30FF);
  };
  auto lm = [&](size_t i) { return c[i].cls == ucd::kLetter || c[i].cls == ucd::kMark; };
  auto ps = [&](size_t i) { return ds3_ps(c[i].cp, c[i].cls); };
  auto sp = [&](size_t i) { return c[i].cls == ucd::kSpace; };
  auto nl = [&](size_t i) { return c[i].cp == '\r' || c[i].cp == '\n'; };
  auto apunct = [&](size_t i) {
    const uint32_t x = c[i].cp;
    return (x >= '!' && x <= '/') || (x >= ':' && x <= '@') || (x >= '[' && x <= '`') || (x >= '{' && x <= '~');
  };
  auto aalpha = [&](size_t i) {
    const uint32_t x = c[i].cp;
    return (x >= 'A' && x <= 'Z') || (x >= 'a' && x <= 'z');
  };
  // step 3 on [i0, i1): the end of the match at i, or i when nothing matches
  auto match3 = [&](size_t i, size_t i1) -> size_t {
    if (apunct(i) && i + 1 < i1 && aalpha(i + 1)) {
      size_t j = i + 1;
      while (j < i1 && aalpha(j)) ++j;
      return j;
    }
    {
      const bool pre = !nl(i) && c[i].cls != ucd::kLetter && !ps(i);
      size_t j = i;
      if (pre && i + 1 < i1 && lm(i + 1)) j = i + 1;
      else if (lm(i)) j = i;
      else j = SIZE_MAX;
      if (j != SIZE_MAX) {
        while (j < i1 && lm(j)) ++j;
        return j;
      }
    }
    {
      size_t j = SIZE_MAX;
      if (c[i].cp == ' ' && i + 1 < i1 && ps(i + 1)) j = i + 1;
      else if (ps(i)) j = i;
      if (j != SIZE_MAX) {
        while (j < i1 && ps(j)) ++j;
        while (j < i1 && nl(j)) ++j;
        return j;
      }
    }
    if (sp(i)) {
      size_t j = i, last_nl = SIZE_MAX, last_start = i;
      while (j < i1 && sp(j)) {
        last_start = j;
        if (nl(j)) last_nl = j + 1;
        ++j;
      }
      if (last_nl != SIZE_MAX) return last_nl;     // \s*[\r\n]+
      if (j == i1 || last_start == i) return j;    // \s+(?!\S) at the end, or \s+ of one
      return last_start;                           // \s+(?!\S): the last space starts the next piece
    }
    return i;
  };
  auto step3 = [&](size_t i0, size_t i1) {
    size_t gap = i0, i = i0;
    while (i < i1) {
      const size_t j = match3(i, i1);
      if (j == i) {
        ++i;
        continue;
      }
      if (gap < i) f(off(gap), off(i));
      f(off(i), off(j));
      i = gap = j;
    }
    if (gap < i1) f(off(gap), off(i1));
  };
  auto step2 = [&](size_t i0, size_t i1) {
    size_t i = i0, gap = i0;
    while (i < i1) {
      if (!cjk(i)) {
        ++i;
        continue;
      }
      size_t j = i;
      while (j < i1 && cjk(j)) ++j;
      if (gap < i) step3(gap, i);
      step3(i, j);
      i = gap = j;
    }
    if (gap < i1) step3(gap, i1);
  };
  size_t i = 0, gap = 0;
  while (i < n) {
    if (c[i].cls != ucd::kNumber) {
      ++i;
      continue;
    }
    if (gap < i) step2(gap, i);
    size_t j = i;
    while (j < n && j < i + 3 && c[j].cls == ucd::kNumber) ++j;
    step2(i, j);
    i = gap = j;
  }
  if (gap < n) step2(gap, n);
}

}  // namespace espetero::tok
