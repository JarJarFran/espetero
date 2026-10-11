// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "grammar.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <numeric>

#include "../tok/tokenizer.h"

namespace espetero::grammar {

namespace {

using Clock = std::chrono::steady_clock;

// Frame phases, per kind of frame.
enum : uint8_t { R_BEFORE = 0, R_DONE = 1 };                                       // root
enum : uint8_t { O_FIRST = 0, O_KEY, O_COLON, O_VALUE, O_AFTER, O_NEXT };          // object
enum : uint8_t { A_FIRST = 0, A_AFTER, A_NEXT };                                    // array
enum : uint8_t { N_START = 0, N_MINUS, N_ZERO, N_INT, N_DOT, N_FRAC, N_EXP, N_EXP_SIGN, N_EXP_DIG };  // number
constexpr uint32_t kOff = 0xffffffffu;         // key: off every defined name
constexpr uint32_t kAdditional = 0xfffffffeu;  // key read: an other (additional) key
constexpr uint8_t kNeg = 1;                    // number aux: a minus sign was read

inline bool is_ws(uint8_t c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }
inline bool is_digit(uint8_t c) { return c >= '0' && c <= '9'; }

int32_t push(Pool& pool, const Frame& f, int32_t parent) {
  const int32_t depth = parent >= 0 ? pool[size_t(parent)].depth + 1 : 1;
  pool.push_back(PNode{f, parent, depth});
  return int32_t(pool.size()) - 1;
}

// One byte of string content in sub-state `sub` (0 clean, 1 after '\',
// 2..6 inside \u00xx, 16 + 8 * r + cls: r UTF-8 continuation bytes to come,
// the next one in range class cls). `starts`: the byte starts a code point.
enum { kReject = -1, kContent = 0, kClose = 1 };
constexpr uint8_t utf8_state(int r, int cls) { return uint8_t(16 + (r << 3) + cls); }
int string_byte(uint8_t& sub, uint8_t c, bool& starts) {
  starts = false;
  switch (sub) {
    case 0:
      if (c == '"') return kClose;
      if (c == '\\') {
        sub = 1;
        starts = true;
        return kContent;
      }
      if (c < 0x20) return kReject;  // control characters are escaped
      starts = true;
      if (c < 0x80) return kContent;
      if (c < 0xC2) return kReject;  // a continuation byte, or an overlong lead
      if (c < 0xE0) sub = utf8_state(1, 0);
      else if (c == 0xE0) sub = utf8_state(2, 1);
      else if (c == 0xED) sub = utf8_state(2, 2);  // no surrogates
      else if (c < 0xF0) sub = utf8_state(2, 0);
      else if (c == 0xF0) sub = utf8_state(3, 3);
      else if (c < 0xF4) sub = utf8_state(3, 0);
      else if (c == 0xF4) sub = utf8_state(3, 4);  // up to U+10FFFF
      else return kReject;
      return kContent;
    case 1:  // json.dumps' escapes only: no \/, \u only below (below)
      if (c == '"' || c == '\\' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't') {
        sub = 0;
        return kContent;
      }
      if (c == 'u') {
        sub = 2;
        return kContent;
      }
      return kReject;
    case 2:
    case 3:
      if (c != '0') return kReject;
      ++sub;
      return kContent;
    case 4:
      if (c != '0' && c != '1') return kReject;
      sub = c == '0' ? 5 : 6;
      return kContent;
    case 5:  // \u000x: control characters without a short escape (\b \t \n \f \r are 8, 9, a, c, d)
      if ((c >= '0' && c <= '7') || c == 'b' || c == 'e' || c == 'f') {
        sub = 0;
        return kContent;
      }
      return kReject;
    case 6:  // \u001x, lowercase as json.dumps writes it
      if (is_digit(c) || (c >= 'a' && c <= 'f')) {
        sub = 0;
        return kContent;
      }
      return kReject;
    default: {
      static const uint8_t lo[5] = {0x80, 0xA0, 0x80, 0x90, 0x80}, hi[5] = {0xBF, 0xBF, 0x9F, 0xBF, 0x8F};
      const int r = (sub - 16) >> 3, cls = (sub - 16) & 7;
      if (c < lo[cls] || c > hi[cls]) return kReject;
      sub = r == 1 ? 0 : utf8_state(r - 1, 0);
      return kContent;
    }
  }
}

// ---- decimal comparisons for number bounds
// A number's digits so far as m * 10^e; values are compared exactly.
struct Dec {
  unsigned __int128 m = 0;
  int e = 0;
};
int digits_of(unsigned __int128 m) {
  int n = 0;
  while (m) {
    m /= 10;
    ++n;
  }
  return n;
}
// Compare non-negative decimals: -1, 0, 1.
int cmp_dec(Dec a, Dec b) {
  if (a.m == 0 || b.m == 0) return a.m == b.m ? 0 : (a.m == 0 ? -1 : 1);
  const int ma = digits_of(a.m) + a.e, mb = digits_of(b.m) + b.e;  // magnitude: position of the leading digit
  if (ma != mb) return ma < mb ? -1 : 1;
  // Same magnitude: align to the smaller exponent (at most ~40 digits apart: both fit after alignment
  // because their leading digits are at the same position and each has at most 38 digits).
  while (a.e > b.e) {
    a.m *= 10;
    --a.e;
  }
  while (b.e > a.e) {
    b.m *= 10;
    --b.e;
  }
  return a.m < b.m ? -1 : a.m > b.m ? 1 : 0;
}
// Signed comparison of (neg, d) values.
int cmp_signed(bool na, Dec a, bool nb, Dec b) {
  if (a.m == 0) na = false;
  if (b.m == 0) nb = false;
  if (na != nb) return na ? -1 : 1;
  const int c = cmp_dec(a, b);
  return na ? -c : c;
}
// Value v against a bound: v within it?
bool above_lo(bool neg, Dec v, const Bound& lo) {
  if (!lo.set) return true;
  const int c = cmp_signed(neg, v, lo.neg, Dec{lo.m, lo.e});
  return lo.exclusive ? c > 0 : c >= 0;
}
bool below_hi(bool neg, Dec v, const Bound& hi) {
  if (!hi.set) return true;
  const int c = cmp_signed(neg, v, hi.neg, Dec{hi.m, hi.e});
  return hi.exclusive ? c < 0 : c <= 0;
}
// Does the value interval [L, U] (L included when l_in, U when u_in) meet
// the bounds? Two intervals meet when each one's lower end is below the
// other's upper end (equal: both ends included).
bool meets(bool nl, Dec L, bool l_in, bool nu, Dec U, bool u_in, const Bound& lo, const Bound& hi) {
  if (hi.set) {
    const int c = cmp_signed(nl, L, hi.neg, Dec{hi.m, hi.e});
    if (c > 0 || (c == 0 && (!l_in || hi.exclusive))) return false;
  }
  if (lo.set) {
    const int c = cmp_signed(lo.neg, Dec{lo.m, lo.e}, nu, U);
    if (c > 0 || (c == 0 && (lo.exclusive || !u_in))) return false;
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------- grammar
void Grammar::trie_insert(int root, const std::string& text, int32_t value, int32_t below) {
  int32_t t = root;
  if (below >= 0) trie[size_t(t)].below.push_back(below);
  for (unsigned char c : text) {
    int32_t n = child(t, c);
    if (n < 0) {
      n = int32_t(trie.size());
      trie.emplace_back();
      auto& nx = trie[size_t(t)].next;
      nx.insert(std::upper_bound(nx.begin(), nx.end(), std::make_pair(uint8_t(c), int32_t(-1)),
                                 [](const auto& a, const auto& b) { return a.first < b.first; }),
                {uint8_t(c), n});
    }
    t = n;
    if (below >= 0) trie[size_t(t)].below.push_back(below);
  }
  trie[size_t(t)].value = value;
}

std::string canonical_text(const json::Value& v) { return json::dump(v); }

// ---------------------------------------------------------------- automaton
int32_t Automaton::start(Pool& pool) const { return push(pool, Frame{kRootNode, R_BEFORE, 0, 0, 0, 0, 0}, -1); }

bool Automaton::key_allowed(const Node& n, uint32_t next, uint32_t count, int j) const {
  const int np = int(n.props.size());
  if (j < int(next) || j >= np || j > n.first_req[next]) return false;  // a required property would be skipped
  if (n.max_props != kNoLimit && int64_t(count) + 1 + n.req_after[size_t(j)] > n.max_props) return false;
  if (n.min_props > 0 && n.additional < 0 && int64_t(count) + 1 + (np - 1 - j) < n.min_props) return false;
  return true;
}

bool Automaton::other_key_allowed(const Node& n, uint32_t next, uint32_t count) const {
  return n.additional >= 0 && n.first_req[next] == int(n.props.size()) &&
         (n.max_props == kNoLimit || int64_t(count) + 1 <= n.max_props);
}

bool Automaton::any_key_allowed(const Node& n, uint32_t next, uint32_t count) const {
  if (other_key_allowed(n, next, count)) return true;
  const int np = int(n.props.size());
  for (int j = int(next); j < np && j <= n.first_req[next]; ++j)
    if (key_allowed(n, next, count, j)) return true;
  return false;
}

bool Automaton::subtree_allowed(const Node& n, int32_t t, uint32_t next, uint32_t count) const {
  const auto& b = g_.trie[size_t(t)].below;
  for (auto it = std::lower_bound(b.begin(), b.end(), int32_t(next)); it != b.end() && *it <= n.first_req[next]; ++it)
    if (key_allowed(n, next, count, *it)) return true;
  return false;
}

// Bounds: can the number read so far still end within them (final:
// does it, ending now)? M is every digit read so far as one integer, `frac`
// of them after the point (aux >> 1), the value M * 10^-frac with the sign
// (aux & kNeg). Completions by phase: after "-", any value <= 0; after a
// leading 0, 0 and (numbers) [0, 1); in the integer part, M itself and
// [M, M + 1) (numbers), and for k more integer digits [M * 10^k, (M + 1) *
// 10^k) (integers: up to (M + 1) * 10^k - 1); after the point,
// [M * 10^-frac, (M + 1) * 10^-frac). Negative values mirror the
// magnitudes. Integer nodes have inclusive integer bounds (json_schema.cpp).
bool Automaton::number_feasible(const Node& n, const Frame& f, bool final) const {
  if (!n.lo.set && !n.hi.set) return true;
  const bool neg = f.aux & kNeg;
  const int frac = f.aux >> 1;
  const unsigned __int128 M = (unsigned __int128)(uint64_t(f.a) | (uint64_t(f.b) << 32));
  auto in = [&](bool vneg, Dec d) { return above_lo(vneg, d, n.lo) && below_hi(vneg, d, n.hi); };
  if (final) return in(neg, Dec{M, -frac});
  auto mag = [&](Dec a, Dec b, bool b_in) {  // magnitudes [a, b) (b too when b_in), signed
    return neg ? meets(true, b, b_in, true, a, true, n.lo, n.hi) : meets(false, a, true, false, b, b_in, n.lo, n.hi);
  };
  switch (f.phase) {
    case N_MINUS:  // some value <= 0 (integers: 0 written -0, or <= -1)
      return !n.lo.set || n.lo.neg || (n.lo.m == 0 && !n.lo.exclusive);
    case N_ZERO:
      return n.integer ? in(neg, Dec{}) : mag(Dec{}, Dec{1, 0}, false);
    case N_INT: {
      const int cap = n.integer ? 18 : kMaxRangeDigits;
      Dec a{M, 0}, b{M + 1, 0};
      for (int k = 0; int(f.sub) + k <= cap; ++k) {
        if (n.integer ? mag(a, Dec{b.m - 1, 0}, true) : mag(a, b, false)) return true;
        a.m *= 10;
        b.m *= 10;
      }
      return false;
    }
    case N_DOT:
    case N_FRAC:
      return mag(Dec{M, -frac}, Dec{M + 1, -frac}, false);
    default:
      return true;
  }
}

void Automaton::begin(Pool& pool, int node, int32_t parent, uint8_t c, std::vector<int32_t>& out) const {
  if (node < 0) return;
  const Node& n = g_.nodes[size_t(node)];
  switch (n.kind) {
    case Kind::Never:
      return;
    case Kind::Alt: {
      // Compiled grammars have no alternative of an alternative (flattened);
      // the guard covers compile-time checks of literals, before that.
      thread_local int nest = 0;
      if (nest > 64) return;
      ++nest;
      for (int a : n.alts) begin(pool, a, parent, c, out);
      --nest;
      return;
    }
    case Kind::Object:
    case Kind::Array:
      if (c != (n.kind == Kind::Object ? '{' : '[')) return;
      if (pool[size_t(parent)].depth >= kMaxDepth) return;
      out.push_back(push(pool, Frame{uint32_t(node), 0, 0, 0, 0, 0, 0}, parent));
      return;
    case Kind::String:
      if (c == '"') out.push_back(push(pool, Frame{uint32_t(node), 0, 0, 0, 0, 0, 0}, parent));
      return;
    case Kind::Number: {
      if (c != '-' && !is_digit(c)) return;
      const int32_t t = push(pool, Frame{uint32_t(node), N_START, 0, 0, 0, 0, 0}, parent);
      step(pool, t, c, out);
      return;
    }
    case Kind::Literal: {
      if (g_.child(n.lit_trie, c) < 0) return;
      const int32_t t = push(pool, Frame{uint32_t(node), 0, 0, 0, 0, uint32_t(n.lit_trie), 0}, parent);
      step(pool, t, c, out);
      return;
    }
  }
}

void Automaton::step(Pool& pool, int32_t top, uint8_t c, std::vector<int32_t>& out) const {
  const PNode pn = pool[size_t(top)];  // a copy: pushes may move the pool
  const Frame& f = pn.f;
  auto stay = [&](const Frame& g) { out.push_back(push(pool, g, pn.parent)); };
  // Whitespace in a gap between two JSON tokens: one space, or one newline
  // and up to kMaxWhitespace spaces or tabs of indentation, the shapes
  // json.dumps and pretty printers write (sampled replies padded longer
  // runs). ws: 0 none, 1 a space, 2 + n a newline and n indentation bytes.
  auto gap = [&]() {
    uint8_t w = 0xff;
    if (f.ws == 0) w = c == ' ' ? 1 : c == '\n' ? 2 : 0xff;
    else if (f.ws >= 2 && (c == ' ' || c == '\t') && f.ws - 2 < kMaxWhitespace) w = uint8_t(f.ws + 1);
    if (w == 0xff) return;
    Frame g = f;
    g.ws = w;
    stay(g);
  };
  if (f.node == kRootNode) {
    // Before the value, at most two newlines (a reasoning block ends in
    // "\n\n"; a model that wanted to write prose or a code fence there padded
    // with whatever whitespace was allowed); after it, nothing but the end.
    if (f.phase != R_BEFORE) return;
    if (is_ws(c)) {
      if (c != '\n' || f.ws >= 2) return;
      Frame g = f;
      ++g.ws;
      stay(g);
      return;
    }
    Frame g = f;
    g.phase = R_DONE;
    g.ws = 0;
    const int32_t p = push(pool, g, pn.parent);
    begin(pool, g_.root, p, c, out);
    return;
  }
  const Node& n = g_.nodes[f.node];
  switch (n.kind) {
    case Kind::Object: {
      const uint32_t next = f.a & 0xffff, count = f.a >> 16;
      if (f.phase != O_KEY && is_ws(c)) return gap();
      switch (f.phase) {
        case O_FIRST:
          if (c == '}') {
            if (n.first_req[0] == int(n.props.size()) && n.min_props == 0) out.push_back(pn.parent);
            return;
          }
          [[fallthrough]];
        case O_NEXT: {
          if (c != '"' || !any_key_allowed(n, next, count)) return;
          Frame g = f;
          g.phase = O_KEY;
          g.ws = 0;
          g.sub = 0;
          g.b = n.key_trie >= 0 ? uint32_t(n.key_trie) : kOff;
          if (g.b != kOff && !subtree_allowed(n, int32_t(g.b), next, count)) g.b = kOff;
          stay(g);
          return;
        }
        case O_KEY: {
          uint8_t sub = f.sub;
          bool starts = false;
          const int r = string_byte(sub, c, starts);
          if (r == kReject) return;
          int32_t t = f.b == kOff ? -1 : int32_t(f.b);
          if (r == kClose) {
            int32_t j = -1;
            if (t >= 0)
              if (const int32_t q = g_.child(t, '"'); q >= 0) j = g_.trie[size_t(q)].value;
            const uint32_t np = uint32_t(n.props.size());
            uint32_t cnt = count + 1;
            if (n.max_props == kNoLimit && cnt > uint32_t(n.min_props)) cnt = uint32_t(n.min_props);  // saturated
            Frame g = f;
            g.phase = O_COLON;
            g.ws = 0;
            g.sub = 0;
            if (j >= 0) {  // a defined property (never an other key, even out of order)
              if (!key_allowed(n, next, count, j)) return;
              g.b = uint32_t(j);
              g.a = uint32_t(j + 1) | (cnt << 16);
            } else {
              if (!other_key_allowed(n, next, count)) return;
              g.b = kAdditional;
              g.a = np | (cnt << 16);
            }
            stay(g);
            return;
          }
          if (t >= 0) {
            t = g_.child(t, c);
            if (t >= 0 && !subtree_allowed(n, t, next, count)) t = -1;
          }
          if (t < 0 && !other_key_allowed(n, next, count)) return;
          Frame g = f;
          g.sub = sub;
          g.b = t < 0 ? kOff : uint32_t(t);
          stay(g);
          return;
        }
        case O_COLON: {
          if (c != ':') return;
          Frame g = f;
          g.phase = O_VALUE;
          g.ws = 0;
          stay(g);
          return;
        }
        case O_VALUE: {
          const int vnode = f.b == kAdditional ? n.additional : n.props[f.b].node;
          Frame g = f;
          g.phase = O_AFTER;
          g.ws = 0;
          const int32_t p = push(pool, g, pn.parent);
          begin(pool, vnode, p, c, out);
          return;
        }
        case O_AFTER:
          if (c == ',') {
            if (!any_key_allowed(n, next, count)) return;
            Frame g = f;
            g.phase = O_NEXT;
            g.ws = 0;
            stay(g);
          } else if (c == '}') {
            if (n.first_req[next] == int(n.props.size()) && int(count) >= n.min_props) out.push_back(pn.parent);
          }
          return;
      }
      return;
    }
    case Kind::Array: {
      const uint32_t count = f.a;
      if (is_ws(c)) return gap();
      auto item_at = [&](uint32_t i) { return i < n.prefix.size() ? n.prefix[i] : n.items; };
      switch (f.phase) {
        case A_FIRST:
          if (c == ']') {
            if (n.min_items == 0) out.push_back(pn.parent);
            return;
          }
          [[fallthrough]];
        case A_NEXT: {
          const int item = item_at(count);
          if (item < 0 || int64_t(count) >= n.max_items) return;
          uint32_t cnt = count + 1;
          if (n.max_items == kNoLimit) cnt = std::min(cnt, uint32_t(std::max<size_t>(size_t(n.min_items), n.prefix.size())));
          Frame g = f;
          g.phase = A_AFTER;
          g.ws = 0;
          g.a = cnt;
          const int32_t p = push(pool, g, pn.parent);
          begin(pool, item, p, c, out);
          return;
        }
        case A_AFTER:
          if (c == ',') {
            if (int64_t(count) >= n.max_items || item_at(count) < 0) return;
            Frame g = f;
            g.phase = A_NEXT;
            g.ws = 0;
            stay(g);
          } else if (c == ']') {
            if (int(count) >= n.min_items) out.push_back(pn.parent);
          }
          return;
      }
      return;
    }
    case Kind::String: {
      uint8_t sub = f.sub;
      bool starts = false;
      const int r = string_byte(sub, c, starts);
      if (r == kReject) return;
      if (r == kClose) {
        if (int64_t(f.a) >= n.min_len) out.push_back(pn.parent);
        return;
      }
      Frame g = f;
      g.sub = sub;
      if (starts) {
        if (int64_t(f.a) >= n.max_len) return;
        g.a = f.a + 1;
        if (n.max_len == kNoLimit && g.a > uint32_t(n.min_len)) g.a = uint32_t(n.min_len);  // saturated
      }
      stay(g);
      return;
    }
    case Kind::Number: {
      const bool d = is_digit(c);
      const bool ranged = n.lo.set || n.hi.set;
      auto go = [&](uint8_t phase, uint8_t sub, bool digit) {
        Frame g = f;
        g.phase = phase;
        g.sub = sub;
        if (ranged) {
          if (phase == N_MINUS) g.aux |= kNeg;
          if (digit && phase != N_EXP_DIG) {  // the magnitude's digits (no exponent with bounds)
            const uint64_t m = (uint64_t(f.a) | (uint64_t(f.b) << 32)) * 10 + uint64_t(c - '0');
            g.a = uint32_t(m);
            g.b = uint32_t(m >> 32);
            if (phase == N_FRAC) g.aux = uint8_t((g.aux & kNeg) | ((((f.aux >> 1) + 1) & 0x7f) << 1));
          }
          if (!number_feasible(n, g, false)) return;
        }
        stay(g);
      };
      auto end = [&]() {  // the number ends before c
        if (ranged && !number_feasible(n, f, true)) return;
        step(pool, pn.parent, c, out);
      };
      // With bounds, at most kMaxRangeDigits significant digits: decimals that
      // short order as the floats a validator compares them as.
      const uint64_t mag = uint64_t(f.a) | (uint64_t(f.b) << 32);
      auto sig_room = [&]() {
        if (!ranged) return true;
        int nd = 0;
        for (uint64_t m = mag; m; m /= 10) ++nd;
        return nd < kMaxRangeDigits;
      };
      switch (f.phase) {
        case N_START:
          if (c == '-') go(N_MINUS, 0, false);
          else if (c == '0') go(N_ZERO, 1, false);
          else if (d) go(N_INT, 1, true);
          return;
        case N_MINUS:
          if (c == '0') go(N_ZERO, 1, false);
          else if (d) go(N_INT, 1, true);
          return;
        case N_ZERO:
          if (!n.integer && c == '.') go(N_DOT, 0, false);
          else if (!n.integer && !ranged && (c == 'e' || c == 'E')) go(N_EXP, 0, false);
          else end();
          return;
        case N_INT:
          if (d) {
            if (f.sub < (ranged ? (n.integer ? 18 : kMaxRangeDigits) : kMaxIntDigits)) go(N_INT, uint8_t(f.sub + 1), true);
            else end();
          } else if (!n.integer && c == '.') {
            go(N_DOT, 0, false);
          } else if (!n.integer && !ranged && (c == 'e' || c == 'E')) {
            go(N_EXP, 0, false);
          } else {
            end();
          }
          return;
        case N_DOT:
          if (d && sig_room()) go(N_FRAC, 1, true);
          return;
        case N_FRAC:
          if (d) {
            if (f.sub < kMaxFracDigits && sig_room()) go(N_FRAC, uint8_t(f.sub + 1), true);
            else end();
          } else if (!ranged && (c == 'e' || c == 'E')) {
            go(N_EXP, 0, false);
          } else {
            end();
          }
          return;
        case N_EXP:
          if (c == '+' || c == '-') go(N_EXP_SIGN, 0, false);
          else if (d) go(N_EXP_DIG, 1, false);
          return;
        case N_EXP_SIGN:
          if (d) go(N_EXP_DIG, 1, false);
          return;
        case N_EXP_DIG:
          if (d && f.sub < kMaxExpDigits) go(N_EXP_DIG, uint8_t(f.sub + 1), false);
          else end();
          return;
      }
      return;
    }
    case Kind::Literal: {
      const int32_t t = g_.child(int32_t(f.a), c);
      if (t >= 0) {
        const TrieNode& tn = g_.trie[size_t(t)];
        if (tn.value >= 0 && tn.next.empty()) {
          out.push_back(pn.parent);  // the literal is complete
          return;
        }
        Frame g = f;
        g.a = uint32_t(t);
        stay(g);
        return;
      }
      if (g_.trie[f.a].value >= 0) step(pool, pn.parent, c, out);  // it ended before c
      return;
    }
    case Kind::Never:
    case Kind::Alt:
      return;
  }
}

bool Automaton::accepting(const Pool& pool, int32_t top) const {
  while (top >= 0) {
    const Frame& f = pool[size_t(top)].f;
    if (f.node == kRootNode) return f.phase == R_DONE;
    const Node& n = g_.nodes[f.node];
    if (n.kind == Kind::Number) {
      if (f.phase != N_ZERO && f.phase != N_INT && f.phase != N_FRAC && f.phase != N_EXP_DIG) return false;
      if (!number_feasible(n, f, true)) return false;
    } else if (n.kind == Kind::Literal) {
      if (g_.trie[f.a].value < 0) return false;
    } else {
      return false;
    }
    top = pool[size_t(top)].parent;
  }
  return false;
}

bool Automaton::initial(const Pool& pool, int32_t top) const {
  const Frame& f = pool[size_t(top)].f;
  return f.node == kRootNode && f.phase == R_BEFORE;
}

int Automaton::free_string_budget(const Pool& pool, int32_t top) const {
  const Frame& f = pool[size_t(top)].f;
  if (f.node == kRootNode || f.sub != 0) return -1;
  const Node& n = g_.nodes[f.node];
  if (n.kind == Kind::String) return n.max_len == kNoLimit ? kNoLimit : std::max(0, n.max_len - int(f.a));
  if (n.kind == Kind::Object && f.phase == O_KEY && other_key_allowed(n, f.a & 0xffff, f.a >> 16)) return kNoLimit;
  return -1;
}

bool Automaton::accepts_text(const std::string& text, bool complete) const {
  Pool pool;
  std::vector<int32_t> tops{start(pool)}, next;
  for (unsigned char c : text) {
    next.clear();
    for (int32_t t : tops) step(pool, t, c, next);
    if (next.empty()) return false;
    tops.swap(next);
  }
  if (!complete) return true;
  for (int32_t t : tops)
    if (accepting(pool, t)) return true;
  return false;
}

// ---------------------------------------------------------------- vocabulary
Vocab::Vocab(const Tokenizer& tok, int n_model) : n_(n_model) {
  const auto t0 = Clock::now();
  bytes_.resize(size_t(n_));
  const int nt = std::min(n_model, int(tok.vocab_size()));
  std::vector<int32_t> order;
  for (int32_t id = 0; id < nt; ++id) {
    if (tok.is_added(id)) continue;
    bytes_[size_t(id)] = tok.piece(id);
    if (!bytes_[size_t(id)].empty()) order.push_back(id);
  }
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    const int c = bytes_[size_t(a)].compare(bytes_[size_t(b)]);
    return c < 0 || (c == 0 && a < b);
  });
  tok_next.assign(size_t(n_), -1);
  nodes.reserve(order.size() * 3);
  nodes.emplace_back();
  // The trie in one pass over the sorted strings: a node's tokens (the
  // strings that end at its depth) come first in its range, then its
  // children's ranges by byte. Iterative, with an explicit stack.
  struct Item {
    uint32_t node;
    size_t lo, hi, depth;
  };
  std::vector<Item> stack{{0, 0, order.size(), 0}};
  while (!stack.empty()) {
    Item it = stack.back();
    stack.pop_back();
    int32_t last = -1;
    while (it.lo < it.hi && bytes_[size_t(order[it.lo])].size() == it.depth) {
      const int32_t id = order[it.lo++];
      if (last < 0) nodes[it.node].token = id;
      else tok_next[size_t(last)] = id;
      last = id;
    }
    std::vector<std::pair<uint8_t, std::pair<size_t, size_t>>> groups;
    for (size_t i = it.lo; i < it.hi;) {
      const uint8_t b = uint8_t(bytes_[size_t(order[i])][it.depth]);
      size_t j = i;
      while (j < it.hi && uint8_t(bytes_[size_t(order[j])][it.depth]) == b) ++j;
      groups.push_back({b, {i, j}});
      i = j;
    }
    nodes[it.node].edge0 = uint32_t(ebyte.size());
    nodes[it.node].n_edges = uint32_t(groups.size());
    for (const auto& [b, r] : groups) {
      const uint32_t child = uint32_t(nodes.size());
      nodes.emplace_back();
      ebyte.push_back(b);
      echild.push_back(child);
      stack.push_back({child, r.first, r.second, it.depth + 1});
    }
  }
  // Free-string classes.
  plain_all.assign(size_t(words()), 0);
  plain_le.assign(kLevels, std::vector<uint64_t>(size_t(words()), 0));
  for (int32_t id : order) {
    uint8_t sub = 0;
    int cps = 0, cls = 1;
    for (unsigned char c : bytes_[size_t(id)]) {
      if (sub == 0 && (c == '"' || c == '\\')) {
        cls = 2;
        break;
      }
      bool starts = false;
      if (string_byte(sub, c, starts) != kContent) {
        cls = 0;
        break;
      }
      cps += starts;
    }
    if (cls == 2) {
      exits.push_back(id);
    } else if (cls == 1) {
      const uint64_t bit = 1ull << (id & 63);
      plain_all[size_t(id) >> 6] |= bit;
      if (cps < kLevels)
        for (int l = cps; l < kLevels; ++l) plain_le[size_t(l)][size_t(id) >> 6] |= bit;
      else
        long_plain.push_back({cps, id});
    }
  }
  std::sort(long_plain.begin(), long_plain.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  build_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------- masks
bool TokenMask::allows(int32_t t) const {
  if (t < 0) return false;
  if (bits.empty()) return std::binary_search(ids.begin(), ids.end(), t);
  const size_t w = size_t(t) >> 6;
  return w < bits.size() && (bits[w] >> (t & 63)) & 1;
}

std::shared_ptr<const TokenMask> TokenMask::from_bits(std::vector<uint64_t> bits, int n) {
  auto m = std::make_shared<TokenMask>();
  int count = 0;
  for (uint64_t w : bits) count += __builtin_popcountll(w);
  m->count = count;
  if (count <= 4096) {
    m->ids.reserve(size_t(count));
    for (size_t w = 0; w < bits.size(); ++w)
      for (uint64_t x = bits[w]; x; x &= x - 1) {
        const int32_t t = int32_t(w * 64 + size_t(__builtin_ctzll(x)));
        if (t < n) m->ids.push_back(t);
      }
    m->count = int(m->ids.size());
  } else {
    m->bits = std::move(bits);
  }
  return m;
}

Compiled::Compiled(std::shared_ptr<const Grammar> g, std::shared_ptr<const Vocab> v, std::vector<int32_t> end_ids,
                   std::vector<int32_t> escape_ids, std::vector<std::string> unenforced)
    : g_(std::move(g)), v_(std::move(v)), a_(*g_), end_(std::move(end_ids)), esc_(std::move(escape_ids)),
      unenforced_(std::move(unenforced)) {}

bool Compiled::is_end(int32_t t) const { return std::find(end_.begin(), end_.end(), t) != end_.end(); }
bool Compiled::is_escape(int32_t t) const { return std::find(esc_.begin(), esc_.end(), t) != esc_.end(); }

std::shared_ptr<const TokenMask> Compiled::think_mask() const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!think_) {
    std::vector<uint64_t> bits(size_t(v_->words()), ~0ull);
    const int n = v_->size();
    if (n % 64) bits.back() = (1ull << (n % 64)) - 1;
    for (int32_t t : end_)
      if (t >= 0 && t < n) bits[size_t(t) >> 6] &= ~(1ull << (t & 63));
    think_ = TokenMask::from_bits(std::move(bits), n);
  }
  return think_;
}

std::shared_ptr<const TokenMask> Compiled::all_mask() const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!all_) {
    std::vector<uint64_t> bits(size_t(v_->words()), ~0ull);
    const int n = v_->size();
    if (n % 64) bits.back() = (1ull << (n % 64)) - 1;
    all_ = TokenMask::from_bits(std::move(bits), n);
  }
  return all_;
}

std::shared_ptr<const TokenMask> Compiled::find(const std::string& key) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = cache_.find(key);
  if (it == cache_.end()) return nullptr;
  lru_.splice(lru_.begin(), lru_, it->second.lru);
  return it->second.mask;
}

void Compiled::store(const std::string& key, std::shared_ptr<const TokenMask> m) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (cache_.count(key)) return;
  lru_.push_front(key);
  bytes_ += m->bytes() + key.size();
  cache_.emplace(key, Entry{std::move(m), lru_.begin()});
  while (bytes_ > kCacheBytes && lru_.size() > 1) {
    auto it = cache_.find(lru_.back());
    bytes_ -= it->second.mask->bytes() + it->first.size();
    cache_.erase(it);
    lru_.pop_back();
  }
}

json::Value Compiled::cache_stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  json::Value o = json::Value::object();
  o.set("masks", int64_t(cache_.size()));
  o.set("bytes", int64_t(bytes_));
  return o;
}

// ---------------------------------------------------------------- matcher
Matcher::Matcher(std::shared_ptr<const Constraint> c)
    : c_(std::move(c)), cc_(*c_->compiled), a_(cc_.automaton()) {
  st_.phase = c_->think_end >= 0 ? Think : Json;
  st_.tops = {a_.start(pool_)};
}

bool Matcher::step_bytes(std::vector<int32_t>& tops, const std::string& bytes) {
  std::vector<int32_t> next;
  next.reserve(4);
  for (unsigned char c : bytes) {
    next.clear();
    for (int32_t t : tops) a_.step(pool_, t, c, next);
    if (next.empty()) return false;
    if (next.size() > 1) {  // the same frame on the same parent: one stack
      size_t k = 0;
      for (size_t i = 0; i < next.size(); ++i) {
        bool dup = false;
        for (size_t j = 0; j < k && !dup; ++j)
          dup = next[j] == next[i] ||
                (pool_[size_t(next[j])].parent == pool_[size_t(next[i])].parent &&
                 std::memcmp(&pool_[size_t(next[j])].f, &pool_[size_t(next[i])].f, sizeof(Frame)) == 0);
        if (!dup) next[k++] = next[i];
      }
      next.resize(k);
    }
    tops.swap(next);
  }
  return true;
}

bool Matcher::step_token(State& s, int32_t t) {
  switch (s.phase) {
    case Done:
      return false;
    case Free:
      return true;
    case Think:
      if (cc_.is_end(t)) return false;
      if (t == c_->think_end) s.phase = Json;
      return true;
    case Json:
      break;
  }
  const Vocab& V = cc_.vocab();
  if (t < 0 || t >= V.size()) return false;
  const std::string& b = V.bytes(t);
  if (b.empty()) {  // an added token: an end once the value is complete, an escape before it
    if (cc_.is_end(t)) {
      for (int32_t top : s.tops)
        if (a_.accepting(pool_, top)) {
          s.phase = Done;
          return true;
        }
      return false;
    }
    if (cc_.is_escape(t)) {
      for (int32_t top : s.tops)
        if (a_.initial(pool_, top)) {
          s.phase = Free;
          return true;
        }
    }
    return false;
  }
  return step_bytes(s.tops, b);
}

bool Matcher::allows(int32_t t) {
  switch (st_.phase) {
    case Think:
      return !cc_.is_end(t);
    case Free:
      return true;
    case Done:
      return false;
    case Json:
      break;
  }
  if (cur_) return cur_->allows(t);
  ++quick_;
  const size_t mark = pool_.size();
  State s{st_.phase, st_.tops};
  const bool ok = step_token(s, t);
  pool_.resize(mark);
  return ok;
}

bool Matcher::advance(int32_t t) {
  const size_t mark = pool_.size();
  State s{st_.phase, st_.tops};
  if (!step_token(s, t)) {
    pool_.resize(mark);
    return false;
  }
  st_ = std::move(s);
  cur_.reset();
  compact();
  return true;
}

size_t Matcher::valid_prefix(const int32_t* t, size_t n) {
  const size_t mark = pool_.size();
  State s{st_.phase, st_.tops};
  size_t k = 0;
  while (k < n && step_token(s, t[k])) ++k;
  pool_.resize(mark);
  return k;
}

void Matcher::compact() {
  Pool np;
  np.reserve(16);
  std::vector<int32_t> remap(pool_.size(), -1);
  std::vector<int32_t> chain;
  for (int32_t& top : st_.tops) {
    chain.clear();
    for (int32_t t = top; t >= 0 && remap[size_t(t)] < 0; t = pool_[size_t(t)].parent) chain.push_back(t);
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
      PNode p = pool_[size_t(*it)];
      if (p.parent >= 0) p.parent = remap[size_t(p.parent)];
      np.push_back(p);
      remap[size_t(*it)] = int32_t(np.size()) - 1;
    }
    top = remap[size_t(top)];
  }
  pool_.swap(np);
}

std::string Matcher::key(const std::vector<int32_t>& tops) const {
  std::vector<std::string> parts;
  parts.reserve(tops.size());
  std::vector<int32_t> chain;
  for (int32_t top : tops) {
    chain.clear();
    for (int32_t t = top; t >= 0; t = pool_[size_t(t)].parent) chain.push_back(t);
    std::string s;
    s.reserve(chain.size() * sizeof(Frame));
    for (auto it = chain.rbegin(); it != chain.rend(); ++it)
      s.append(reinterpret_cast<const char*>(&pool_[size_t(*it)].f), sizeof(Frame));
    parts.push_back(std::move(s));
  }
  if (parts.size() > 1) {
    std::sort(parts.begin(), parts.end());
    parts.erase(std::unique(parts.begin(), parts.end()), parts.end());
  }
  std::string k;
  for (const std::string& p : parts) {
    const uint32_t n = uint32_t(p.size());
    k.append(reinterpret_cast<const char*>(&n), 4);
    k += p;
  }
  return k;
}

namespace {
// The vocabulary trie walked with a set of stacks: every token whose bytes the
// automaton accepts gets its bit.
struct Walk {
  const Automaton& a;
  const Vocab& v;
  Pool& pool;
  std::vector<uint64_t>& bits;
  std::vector<int32_t> buf;
  void run(uint32_t node, size_t off, size_t n) {
    const Vocab::TNode tn = v.nodes[node];
    for (uint32_t e = tn.edge0; e < tn.edge0 + tn.n_edges; ++e) {
      const uint8_t c = v.ebyte[e];
      const size_t pmark = pool.size(), bmark = buf.size();
      for (size_t i = 0; i < n; ++i) a.step(pool, buf[off + i], c, buf);
      const size_t n2 = buf.size() - bmark;
      if (n2) {
        const uint32_t ch = v.echild[e];
        for (int32_t t = v.nodes[ch].token; t >= 0; t = v.tok_next[size_t(t)]) bits[size_t(t) >> 6] |= 1ull << (t & 63);
        if (v.nodes[ch].n_edges) run(ch, bmark, n2);
      }
      buf.resize(bmark);
      pool.resize(pmark);
    }
  }
};
}  // namespace

std::shared_ptr<const TokenMask> Matcher::compute(const std::vector<int32_t>& tops) {
  const Vocab& V = cc_.vocab();
  std::vector<uint64_t> bits(size_t(V.words()), 0);
  bool fast = !tops.empty();
  int budget = -1;
  for (int32_t t : tops) {
    const int b = a_.free_string_budget(pool_, t);
    if (b < 0) {
      fast = false;
      break;
    }
    budget = std::max(budget, b);
  }
  if (fast) {
    // Inside free strings: the plain tokens within the budget, then each
    // token that closes the string or starts an escape, through the automaton.
    if (budget < Vocab::kLevels) {
      bits = V.plain_le[size_t(budget)];
    } else {
      bits = V.plain_all;
      for (const auto& [cps, id] : V.long_plain) {
        if (cps <= budget) break;
        bits[size_t(id) >> 6] &= ~(1ull << (id & 63));
      }
    }
    std::vector<int32_t> s;
    for (int32_t id : V.exits) {
      const size_t mark = pool_.size();
      s = tops;
      if (step_bytes(s, V.bytes(id))) bits[size_t(id) >> 6] |= 1ull << (id & 63);
      pool_.resize(mark);
    }
  } else {
    Walk w{a_, V, pool_, bits, {}};
    w.buf = tops;
    w.run(0, 0, tops.size());
  }
  bool end = false, esc = false;
  for (int32_t t : tops) {
    end = end || a_.accepting(pool_, t);
    esc = esc || a_.initial(pool_, t);
  }
  auto set = [&](int32_t t) {
    if (t >= 0 && t < V.size()) bits[size_t(t) >> 6] |= 1ull << (t & 63);
  };
  if (end)
    for (int32_t t : cc_.end_ids()) set(t);
  if (esc)
    for (int32_t t : cc_.escape_ids()) set(t);
  return TokenMask::from_bits(std::move(bits), V.size());
}

std::shared_ptr<const TokenMask> Matcher::mask() {
  switch (st_.phase) {
    case Think:
      return cc_.think_mask();
    case Free:
      return cc_.all_mask();
    case Done:
      return std::make_shared<TokenMask>();
    case Json:
      break;
  }
  if (cur_) return cur_;
  const std::string k = key(st_.tops);
  if (auto m = cc_.find(k)) {
    ++cached_;
    return cur_ = m;
  }
  const auto t0 = Clock::now();
  cur_ = compute(st_.tops);
  mask_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  ++computed_;
  cc_.store(k, cur_);
  return cur_;
}

json::Value Matcher::stats() const {
  json::Value o = json::Value::object();
  o.set("masks_computed", computed_);
  o.set("masks_cached", cached_);
  o.set("tokens_checked", quick_);
  o.set("mask_ms", std::round(mask_ms_ * 1000) / 1000);
  if (checks_) {  // ESPETERO_GRAMMAR_CHECK
    o.set("argmax_checks", checks_);
    o.set("argmax_mismatches", check_bad_);
  }
  return o;
}

// ---------------------------------------------------------------- choice
// Development (ESPETERO_GRAMMAR_CHECK=1): every masked argmax is compared with a
// scan of every allowed token's logit; disagreements (beyond exact ties) are
// counted in the sequence's timings.structured.
static bool grammar_check() {
  static const bool on = [] {
    const char* v = std::getenv("ESPETERO_GRAMMAR_CHECK");
    return v && *v == '1';
  }();
  return on;
}

static int32_t masked_argmax_fast(Matcher& m, const Readout& r, const SamplingParams& sp, const PenaltyState* pen);

int32_t masked_argmax(Matcher& m, const Readout& r, const SamplingParams& sp, const PenaltyState* pen) {
  const int32_t t = masked_argmax_fast(m, r, sp, pen);
  if (grammar_check() && r.logits && t >= 0) {
    const float* lg = r.logits;
    std::vector<float> buf;
    if (sp.penalized() && pen && !pen->empty()) {
      buf.assign(lg, lg + r.vocab);
      pen->apply(buf.data(), sp);
      lg = buf.data();
    }
    int32_t best = -1;
    m.mask()->for_each([&](int32_t x) {
      if (x < r.vocab && (best < 0 || lg[x] > lg[best])) best = x;
    });
    m.checked(best == t || (best >= 0 && lg[best] == lg[t]));
  }
  return t;
}

static int32_t masked_argmax_fast(Matcher& m, const Readout& r, const SamplingParams& sp, const PenaltyState* pen) {
  const bool penalized = sp.penalized() && pen && !pen->empty();
  const float* lg = r.logits;
  std::vector<float> buf;
  if (!penalized) {
    if (r.argmax >= 0 && m.allows(r.argmax)) return r.argmax;
    const auto mk = m.mask();
    for (int i = 0; i < r.n_cand; ++i)
      if (mk->allows(r.cand[i])) return r.cand[i];
  } else if (lg) {
    buf.assign(lg, lg + r.vocab);
    pen->apply(buf.data(), sp);
    lg = buf.data();
  }
  if (!lg) return -1;
  const auto mk = m.mask();
  int32_t best = -1;
  float bx = 0;
  mk->for_each([&](int32_t t) {
    if (t >= r.vocab) return;
    const float x = lg[t];
    if (best < 0 || x > bx) best = t, bx = x;
  });
  return best;
}

Distribution subset_distribution(const std::vector<int32_t>& ids, const std::vector<float>& lg, int vocab,
                                 const SamplingParams& sp) {
  Distribution d;
  const size_t n = ids.size();
  if (n == 0) return d;
  if (sp.temperature <= 0) {
    size_t b = 0;
    for (size_t i = 1; i < n; ++i)
      if (lg[i] > lg[b]) b = i;
    d.ids = {ids[b]};
    d.p = {1.0};
    return d;
  }
  // As distribution(): x = logits / T, p = exp(x - max) / total; NaN as -inf, +inf as the largest float.
  std::vector<float> x(n);
  const float inv_t = float(1.0 / sp.temperature);
  float m = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < n; ++i) {
    float v = lg[i] * inv_t;
    if (std::isnan(v)) v = -std::numeric_limits<float>::infinity();
    else if (v == std::numeric_limits<float>::infinity()) v = std::numeric_limits<float>::max();
    x[i] = v;
    m = std::max(m, v);
  }
  if (!(m > -std::numeric_limits<float>::infinity())) {  // nothing finite: the first allowed token
    d.ids = {ids[0]};
    d.p = {1.0};
    return d;
  }
  std::vector<double> e(n);
  double total = 0;
  for (size_t i = 0; i < n; ++i) total += e[i] = double(std::exp(x[i] - m));
  const int k = sp.top_k > 0 ? std::min(sp.top_k, vocab) : vocab;
  std::vector<int32_t> out_ids;
  std::vector<double> p;
  if (k < vocab || sp.top_p < 1.0 || sp.min_p > 0) {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), size_t(0));
    auto by = [&](size_t a, size_t b) { return x[a] > x[b] || (x[a] == x[b] && ids[a] < ids[b]); };
    const size_t head = std::min<size_t>(size_t(std::min(k, 2048)), n);
    std::nth_element(idx.begin(), idx.begin() + long(head) - 1, idx.end(), by);
    std::sort(idx.begin(), idx.begin() + long(head), by);
    double mass = 0;
    for (size_t i = 0; i < head; ++i) mass += e[idx[i]] / total;
    size_t keep = head;
    if (k == vocab && sp.top_p < 1.0 && mass < sp.top_p && head < n) {  // a flat distribution: all of it
      std::sort(idx.begin(), idx.end(), by);
      keep = n;
    }
    for (size_t i = 0; i < keep; ++i) {
      out_ids.push_back(ids[idx[i]]);
      p.push_back(e[idx[i]] / total);
    }
    if (sp.min_p > 0) {
      const double floor = sp.min_p * p[0];
      size_t kk = 0;
      while (kk < p.size() && p[kk] >= floor) ++kk;
      out_ids.resize(kk);
      p.resize(kk);
    }
    if (sp.top_p < 1.0) {
      double sum = 0;
      for (double v : p) sum += v;
      const double target = sp.top_p * sum;
      double acc = 0;
      size_t cut = p.size();
      for (size_t i = 0; i < p.size(); ++i) {
        acc += p[i];
        if (acc >= target) {
          cut = i + 1;
          break;
        }
      }
      out_ids.resize(cut);
      p.resize(cut);
    }
  } else {
    out_ids = ids;
    p.resize(n);
    for (size_t i = 0; i < n; ++i) p[i] = e[i] / total;
  }
  double s = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    if (!(p[i] > 0)) continue;
    d.ids.push_back(out_ids[i]);
    d.p.push_back(p[i]);
    s += p[i];
  }
  if (d.ids.empty()) {
    d.ids = {out_ids.empty() ? ids[0] : out_ids[0]};
    d.p = {1.0};
    return d;
  }
  for (double& v : d.p) v /= s;
  return d;
}

Distribution masked_distribution(Matcher& m, const Readout& r, const SamplingParams& sp, const PenaltyState* pen) {
  if (sp.temperature <= 0) {
    Distribution d;
    const int32_t t = masked_argmax(m, r, sp, pen);
    if (t >= 0) {
      d.ids = {t};
      d.p = {1.0};
    }
    return d;
  }
  const bool penalized = sp.penalized() && pen && !pen->empty();
  const auto mk = m.mask();
  // The GPU candidates hold the masked head when, among them, at least
  // top_k (or every allowed token) are allowed: they are the true top-n.
  if (!penalized && sp.top_k >= 1 && r.n_cand >= sp.top_k && r.cand) {
    std::vector<int32_t> ids;
    std::vector<float> x;
    const float s = r.scaled ? 1.0f : float(1.0 / sp.temperature);
    for (int i = 0; i < r.n_cand; ++i)
      if (mk->allows(r.cand[i])) {
        ids.push_back(r.cand[i]);
        x.push_back(r.cand_x[i] * s);
      }
    if (!ids.empty() && int(ids.size()) >= std::min(sp.top_k, mk->count))
      return distribution_topk(ids.data(), x.data(), int(ids.size()), x[0], 1.0f, sp);
  }
  if (!r.logits) return Distribution{};
  const float* lg = r.logits;
  std::vector<float> buf;
  if (penalized) {
    buf.assign(lg, lg + r.vocab);
    pen->apply(buf.data(), sp);
    lg = buf.data();
  }
  std::vector<int32_t> ids;
  std::vector<float> x;
  ids.reserve(size_t(mk->count));
  x.reserve(size_t(mk->count));
  mk->for_each([&](int32_t t) {
    if (t >= r.vocab) return;
    ids.push_back(t);
    x.push_back(lg[t]);
  });
  return subset_distribution(ids, x, r.vocab, sp);
}

}  // namespace espetero::grammar
