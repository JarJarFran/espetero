// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Structured output: a byte-level pushdown automaton over a compiled
// JSON grammar, token masks over the model's vocabulary, and the constrained
// choice of a token from a readout's logits.
//
// The grammar (json_schema.h compiles a JSON Schema into one; json_object is
// "any object") is a graph of nodes: objects with properties in schema
// order, arrays, strings, numbers, literal sets (enum, const, true, false,
// null) and alternatives (anyOf, oneOf, type lists). A parse state is a set
// of stacks of frames, one frame per open value; a set because alternatives
// stay open side by side until a byte rules them out. Frames are immutable
// nodes of a pool with parent links: a byte replaces only the top of a stack,
// and stacks share their bottoms.
//
// What the automaton emits is a canonical subset of JSON, so that a key or a
// literal has one spelling: strings escape only `"`, `\` and control
// characters, as Python's json.dumps(ensure_ascii=False) writes them (\b \f
// \n \r \t, else \u00xx lowercase; no \/, no other \u escapes, raw UTF-8
// checked byte by byte); numbers are JSON numbers of at most 20 integer and
// 20 fraction digits (3 exponent digits); between two tokens one space, or
// one newline and up to kMaxWhitespace spaces or tabs of indentation (before
// the value at most two newlines, after it nothing: the end of the reply);
// objects list their properties in schema order (optional ones may be
// skipped) and other keys, where allowed, after them.
//
// Token masks: a token is allowed when the automaton accepts its bytes from
// the state. A mask is computed by walking a byte trie of the vocabulary with
// the state (a subtree is skipped as soon as its prefix is rejected), except
// inside free strings, where almost every token is allowed: there it is a
// precomputed set of plain-content tokens (by code point count, for
// maxLength) plus the few thousand tokens that close the string or start an
// escape, each checked with the automaton. Masks are cached per state (the
// exact key: every frame of every stack) and shared by every request with the
// same schema.
#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../util/json.h"
#include "sampling.h"

namespace espetero {

class Tokenizer;

namespace grammar {

constexpr int kNoLimit = 0x7fffffff;
constexpr int kMaxWhitespace = 40;  // indentation after a newline between two JSON tokens
constexpr int kMaxDepth = 48;       // frames on a stack: the root, then one per open value
constexpr int kMaxIntDigits = 20, kMaxFracDigits = 20, kMaxExpDigits = 3;
constexpr int kMaxRangeDigits = 15;  // significant digits of a number with bounds (decimal = float order)

enum class Kind : uint8_t { Never, Object, Array, String, Number, Literal, Alt };

// A number's bound: the decimal m * 10^e (sign apart), inclusive or not.
struct Bound {
  bool set = false, exclusive = false, neg = false;
  uint64_t m = 0;  // at most 18 digits
  int e = 0;
};

struct Prop {
  std::string name;  // the key, unescaped
  int node = -1;
  bool required = false;
};

struct Node {
  Kind kind = Kind::Never;
  // Object: properties in emission order; additional: the node of other
  // keys' values (-1: no other keys). first_req[j]: the first required
  // property >= j (n if none); req_after[j]: required properties after j.
  std::vector<Prop> props;
  std::vector<int> first_req, req_after;
  int additional = -1;
  int key_trie = -1;  // the properties' JSON texts after the opening quote, closing quote included
  int min_props = 0, max_props = kNoLimit;
  // Array: prefixItems, then `items` (-1: none).
  std::vector<int> prefix;
  int items = -1;
  int min_items = 0, max_items = kNoLimit;
  // String: length in code points.
  int min_len = 0, max_len = kNoLimit;
  // Number.
  bool integer = false;
  Bound lo, hi;
  // Literal: a trie of canonical JSON texts.
  int lit_trie = -1;
  // Alt.
  std::vector<int> alts;
};

struct TrieNode {
  std::vector<std::pair<uint8_t, int32_t>> next;  // sorted by byte
  int32_t value = -1;           // literal tries: 0 where a literal ends; key tries: the property index
  std::vector<int32_t> below;   // key tries: property indices whose text passes here (sorted)
};

struct Grammar {
  std::vector<Node> nodes;
  std::vector<TrieNode> trie;  // every trie of the grammar
  int root = -1;
  int add(Node n) {
    nodes.push_back(std::move(n));
    return int(nodes.size()) - 1;
  }
  int new_trie() {
    trie.emplace_back();
    return int(trie.size()) - 1;
  }
  void trie_insert(int root, const std::string& text, int32_t value, int32_t below = -1);
  int32_t child(int32_t t, uint8_t c) const {
    for (const auto& [b, n] : trie[size_t(t)].next)
      if (b == c) return n;
    return -1;
  }
};

// The canonical JSON text of a value (see above): compact separators,
// strings as json.dumps(ensure_ascii=False) writes them.
std::string canonical_text(const json::Value& v);

// ---------------------------------------------------------------- automaton
struct Frame {
  uint32_t node;  // grammar node, kRootNode for the bottom frame
  uint8_t phase;
  uint8_t ws;     // whitespace bytes in the current gap
  uint8_t sub;    // strings and keys: escape / UTF-8 state; numbers: digits of the current part
  uint8_t aux;    // numbers: flags
  uint32_t a;     // object: next property | key count << 16; array: items; string: code points; literal: trie node; number: magnitude (low)
  uint32_t b;     // object: key trie node or the property being read; number: magnitude (high)
};
static_assert(sizeof(Frame) == 16, "frames are 16 bytes");
constexpr uint32_t kRootNode = 0xffffffffu;

struct PNode {
  Frame f;
  int32_t parent;
  int32_t depth;
};
using Pool = std::vector<PNode>;

// Steps of the automaton (stateless; stacks live in the caller's pool).
class Automaton {
 public:
  explicit Automaton(const Grammar& g) : g_(g) {}
  const Grammar& grammar() const { return g_; }
  // The start state: the root frame before the value.
  int32_t start(Pool& pool) const;
  // The stacks `top` becomes after byte c (none: rejected), appended to out.
  void step(Pool& pool, int32_t top, uint8_t c, std::vector<int32_t>& out) const;
  // The value is complete: the reply may end here.
  bool accepting(const Pool& pool, int32_t top) const;
  // Nothing but whitespace yet.
  bool initial(const Pool& pool, int32_t top) const;
  // A free-string interior in a clean state (no escape or UTF-8 sequence
  // open): its budget of code points (kNoLimit: none), else -1.
  int free_string_budget(const Pool& pool, int32_t top) const;
  // Whole text from the start state (tests, compile-time literal checks).
  bool accepts_text(const std::string& text, bool complete = true) const;

 private:
  void begin(Pool& pool, int node, int32_t parent, uint8_t c, std::vector<int32_t>& out) const;
  bool key_allowed(const Node& n, uint32_t next, uint32_t count, int j) const;
  bool other_key_allowed(const Node& n, uint32_t next, uint32_t count) const;
  bool any_key_allowed(const Node& n, uint32_t next, uint32_t count) const;
  bool subtree_allowed(const Node& n, int32_t t, uint32_t next, uint32_t count) const;
  bool number_feasible(const Node& n, const Frame& f, bool final) const;
  const Grammar& g_;
};

// ---------------------------------------------------------------- vocabulary
// The model's tokens as bytes, in a byte trie, plus the classes of the free
// string fast path. Added tokens (special or not) are not JSON bytes: they
// are allowed only as end tokens or escapes.
class Vocab {
 public:
  Vocab(const Tokenizer& tok, int n_model);
  int size() const { return n_; }
  int words() const { return (n_ + 63) / 64; }
  const std::string& bytes(int32_t id) const { return bytes_[size_t(id)]; }

  struct TNode {
    uint32_t edge0 = 0;
    uint32_t n_edges = 0;
    int32_t token = -1;  // first token ending here (tok_next chains the others)
  };
  std::vector<TNode> nodes;
  std::vector<uint8_t> ebyte;
  std::vector<uint32_t> echild;
  std::vector<int32_t> tok_next;

  // Free strings: plain tokens (every byte string content from a clean
  // state) with at most L code points, for L < kLevels; all plain tokens; the
  // plain tokens of kLevels or more code points by count, descending; the
  // tokens that close a string or start an escape after plain content.
  static constexpr int kLevels = 16;
  std::vector<std::vector<uint64_t>> plain_le;
  std::vector<uint64_t> plain_all;
  std::vector<std::pair<int, int32_t>> long_plain;  // (code points, id)
  std::vector<int32_t> exits;
  double build_ms = 0;
  size_t trie_nodes() const { return nodes.size(); }

 private:
  int n_ = 0;
  std::vector<std::string> bytes_;
};

// ---------------------------------------------------------------- masks
struct TokenMask {
  int count = 0;
  std::vector<int32_t> ids;     // sorted; the whole set when `bits` is empty
  std::vector<uint64_t> bits;   // large sets
  bool allows(int32_t t) const;
  template <class F>
  void for_each(F&& f) const {
    if (bits.empty()) {
      for (int32_t t : ids) f(t);
      return;
    }
    for (size_t w = 0; w < bits.size(); ++w)
      for (uint64_t x = bits[w]; x; x &= x - 1) f(int32_t(w * 64 + size_t(__builtin_ctzll(x))));
  }
  size_t bytes() const { return ids.size() * 4 + bits.size() * 8 + 64; }
  static std::shared_ptr<const TokenMask> from_bits(std::vector<uint64_t> bits, int n);
};

// A grammar for one model's vocabulary, the tokens that end a reply (once
// the value is complete) and that escape the grammar (instead of the
// value's first byte: a tool call), and the masks computed so far. Shared by
// every request with the same schema; thread-safe.
class Compiled {
 public:
  Compiled(std::shared_ptr<const Grammar> g, std::shared_ptr<const Vocab> v, std::vector<int32_t> end_ids,
           std::vector<int32_t> escape_ids, std::vector<std::string> unenforced);
  const Grammar& grammar() const { return *g_; }
  const Automaton& automaton() const { return a_; }
  const Vocab& vocab() const { return *v_; }
  const std::vector<int32_t>& end_ids() const { return end_; }
  const std::vector<int32_t>& escape_ids() const { return esc_; }
  bool is_end(int32_t t) const;
  bool is_escape(int32_t t) const;
  // Schema keywords a non-strict format does not enforce ("pattern at #/properties/zip").
  const std::vector<std::string>& unenforced() const { return unenforced_; }
  // Every token but the end tokens (the reasoning before the value), every token.
  std::shared_ptr<const TokenMask> think_mask() const;
  std::shared_ptr<const TokenMask> all_mask() const;
  std::shared_ptr<const TokenMask> find(const std::string& key) const;
  void store(const std::string& key, std::shared_ptr<const TokenMask> m) const;
  json::Value cache_stats() const;

 private:
  std::shared_ptr<const Grammar> g_;
  std::shared_ptr<const Vocab> v_;
  Automaton a_;
  std::vector<int32_t> end_, esc_;
  std::vector<std::string> unenforced_;
  mutable std::mutex mu_;
  mutable std::shared_ptr<const TokenMask> think_, all_;
  struct Entry {
    std::shared_ptr<const TokenMask> mask;
    std::list<std::string>::iterator lru;
  };
  mutable std::unordered_map<std::string, Entry> cache_;
  mutable std::list<std::string> lru_;
  mutable size_t bytes_ = 0;
  static constexpr size_t kCacheBytes = size_t(24) << 20;
};

// A request's constraint: the compiled grammar and how the reply starts.
struct Constraint {
  std::shared_ptr<const Compiled> compiled;
  // >= 0: the reply opens in a reasoning block this token closes; the
  // grammar applies after it (before it, only the end tokens are masked).
  int32_t think_end = -1;
};

// One sequence's parse state (not thread-safe; one per sequence).
class Matcher {
 public:
  explicit Matcher(std::shared_ptr<const Constraint> c);
  // The token is allowed now (cheap: the cached mask, else its bytes through the automaton).
  bool allows(int32_t t);
  // Every allowed token (cached per state).
  std::shared_ptr<const TokenMask> mask();
  // Consume a token; false (state unchanged) when it is not allowed.
  bool advance(int32_t t);
  // How many of these tokens, in order, are allowed (drafts).
  size_t valid_prefix(const int32_t* t, size_t n);
  bool done() const { return st_.phase == Done; }
  bool in_grammar() const { return st_.phase == Json; }
  json::Value stats() const;
  const Constraint& constraint() const { return *c_; }
  // ESPETERO_GRAMMAR_CHECK: a masked argmax checked against a scan of the logits.
  void checked(bool ok) {
    ++checks_;
    check_bad_ += !ok;
  }

 private:
  enum Phase : uint8_t { Think, Json, Free, Done };
  struct State {
    Phase phase;
    std::vector<int32_t> tops;
  };
  // A token through a state (pool nodes appended); false when rejected.
  bool step_token(State& s, int32_t t);
  bool step_bytes(std::vector<int32_t>& tops, const std::string& bytes);
  std::string key(const std::vector<int32_t>& tops) const;
  std::shared_ptr<const TokenMask> compute(const std::vector<int32_t>& tops);
  void compact();

  std::shared_ptr<const Constraint> c_;
  const Compiled& cc_;
  const Automaton& a_;
  Pool pool_;
  State st_;
  std::shared_ptr<const TokenMask> cur_;
  std::vector<int32_t> scratch_;
  // Statistics: masks computed and found cached, time computing them, tokens checked alone.
  int computed_ = 0, cached_ = 0, quick_ = 0, checks_ = 0, check_bad_ = 0;
  double mask_ms_ = 0;
};

// ---------------------------------------------------------------- choice
// What the engine has of one readout: the GPU argmax, the GPU top-k
// candidates (ids and x sorted by x desc, id asc; x = logit / temperature
// when `scaled`, the request's own top-k head, else raw logits) and the full
// logits.
struct Readout {
  int32_t argmax = -1;
  const int32_t* cand = nullptr;
  const float* cand_x = nullptr;
  int n_cand = 0;
  bool scaled = false;
  const float* logits = nullptr;
  int vocab = 0;
};

// Greedy (temperature 0): the most likely allowed token of the logits
// (penalized when `sp` says so). The GPU argmax when it is allowed, else the
// first allowed GPU candidate (the list is the true top-k in (logit desc, id
// asc) order, so that is the masked argmax), else a scan of the allowed
// tokens' logits (the first index on ties). -1 if nothing is allowed.
int32_t masked_argmax(Matcher& m, const Readout& r, const SamplingParams& sp, const PenaltyState* pen);
// Sampling: the distribution of distribution() over the logits with every
// disallowed token at -inf: top-k, min-p and top-p among the allowed tokens.
// From the GPU candidates when they hold the masked head (enough allowed
// ones among them), else from the allowed tokens' logits.
Distribution masked_distribution(Matcher& m, const Readout& r, const SamplingParams& sp, const PenaltyState* pen);
// distribution() over a subset of the vocabulary: `ids` with logits `x`
// (penalties applied); explicit ids, zero-probability entries dropped.
Distribution subset_distribution(const std::vector<int32_t>& ids, const std::vector<float>& x, int vocab,
                                 const SamplingParams& sp);

}  // namespace grammar
}  // namespace espetero
