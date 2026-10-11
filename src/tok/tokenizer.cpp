// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <stdexcept>

#include "tok_internal.h"
#include "unicode.h"

namespace espetero {
namespace tok {

const ByteLevelMap& byte_level() {
  static const ByteLevelMap m;
  return m;
}

// A token text in SentencePiece-style BPE decoding (Replace U+2581 -> " ",
// ByteFallback): "<0xHH>" is that byte, otherwise U+2581 becomes a space.
std::string spm_decode(std::string_view text) {
  auto hex = [](char c) { return c >= '0' && c <= '9' ? c - '0' : (c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1); };
  if (text.size() == 6 && text.compare(0, 3, "<0x") == 0 && text[5] == '>' && hex(text[3]) >= 0 && hex(text[4]) >= 0)
    return std::string(1, char(hex(text[3]) * 16 + hex(text[4])));
  static const std::string_view sp = "\xE2\x96\x81";
  std::string out;
  out.reserve(text.size());
  for (size_t i = 0; i < text.size();) {
    if (text.compare(i, sp.size(), sp) == 0) {
      out.push_back(' ');
      i += sp.size();
    } else {
      out.push_back(text[i++]);
    }
  }
  return out;
}

std::string byte_level_decode(std::string_view text) {
  const ByteLevelMap& bl = byte_level();
  std::string out;
  out.reserve(text.size());
  const auto* p = reinterpret_cast<const unsigned char*>(text.data());
  const auto* e = p + text.size();
  while (p < e) {
    uint32_t cp;
    p += decode_utf8(p, e, cp);
    if (cp >= 324 || bl.cp_to_byte[cp] < 0) return std::string(text);
    out.push_back(char(bl.cp_to_byte[cp]));
  }
  return out;
}

// ------------------------------------------------------------------ WordCache
bool WordCache::get(std::string_view key, uint64_t h, std::vector<int32_t>& out) const {
  if (h == 0) h = 1;
  const Shard& sh = shards_[h >> 58];
  std::lock_guard<ShardLock> g(sh.lock);
  if (sh.slots.empty()) return false;
  for (size_t i = h & (kSlots - 1);; i = (i + 1) & (kSlots - 1)) {
    const Entry& en = sh.slots[i];
    if (en.hash == 0) return false;
    if (en.hash == h && en.key_len == key.size() && std::memcmp(sh.keys.data() + en.key_off, key.data(), key.size()) == 0) {
      out.insert(out.end(), sh.ids.begin() + en.ids_off, sh.ids.begin() + en.ids_off + en.n_ids);
      return true;
    }
  }
}

void WordCache::put(std::string_view key, uint64_t h, const int32_t* ids, size_t n) {
  if (key.size() > kMaxKeyLen || n > 0xFFFF) return;
  if (h == 0) h = 1;
  Shard& sh = shards_[h >> 58];
  std::lock_guard<ShardLock> g(sh.lock);
  if (sh.slots.empty() || sh.count >= kMaxEntries || sh.keys.size() + key.size() > kMaxKeyBytes) {
    // First use, or full: start over. Simple, bounded, and cheap to refill.
    sh.slots.assign(kSlots, Entry{0, 0, 0, 0, 0});
    sh.keys.clear();
    sh.ids.clear();
    sh.count = 0;
  }
  size_t i = h & (kSlots - 1);
  for (;; i = (i + 1) & (kSlots - 1)) {
    const Entry& en = sh.slots[i];
    if (en.hash == 0) break;
    if (en.hash == h && en.key_len == key.size() && std::memcmp(sh.keys.data() + en.key_off, key.data(), key.size()) == 0)
      return;  // another thread inserted it meanwhile
  }
  sh.slots[i] = Entry{h, uint32_t(sh.keys.size()), uint32_t(sh.ids.size()), uint16_t(key.size()), uint16_t(n)};
  sh.keys.append(key.data(), key.size());
  sh.ids.insert(sh.ids.end(), ids, ids + n);
  ++sh.count;
}

// ------------------------------------------------------------------ AddedMatcher
void AddedMatcher::build(const std::vector<AddedToken>& tokens, bool normalized) {
  nodes_.assign(1, Node{});
  std::fill(std::begin(first_byte_), std::end(first_byte_), false);
  for (int t = 0; t < int(tokens.size()); ++t) {
    if (tokens[size_t(t)].normalized != normalized) continue;
    const std::string& pat = tokens[size_t(t)].text;
    if (pat.empty()) continue;
    int32_t node = 0;
    for (unsigned char c : pat) {
      int32_t next = child(node, c);
      if (next < 0) {
        next = int32_t(nodes_.size());
        Node nn;
        nn.byte = c;
        nn.next_sibling = nodes_[size_t(node)].first_child;
        nodes_.push_back(nn);
        nodes_[size_t(node)].first_child = next;
      }
      node = next;
    }
    // A later token with identical content replaces the earlier one (same id in practice).
    nodes_[size_t(node)].token = t;
    first_byte_[static_cast<unsigned char>(pat[0])] = true;
  }
  single_first_byte_ = -1;
  int n_first = 0;
  for (int c = 0; c < 256; ++c)
    if (first_byte_[c]) {
      ++n_first;
      single_first_byte_ = c;
    }
  if (n_first != 1) single_first_byte_ = -1;
}

int AddedMatcher::find(std::string_view s, size_t from, size_t& mb, size_t& me) const {
  const auto* d = reinterpret_cast<const unsigned char*>(s.data());
  const size_t n = s.size();
  for (size_t i = from; i < n; ++i) {
    if (single_first_byte_ >= 0) {
      const void* hit = std::memchr(d + i, single_first_byte_, n - i);
      if (!hit) return -1;
      i = size_t(static_cast<const unsigned char*>(hit) - d);
    } else if (!first_byte_[d[i]]) {
      continue;
    }
    // Longest token starting at i.
    int best = -1;
    size_t best_end = 0;
    int32_t node = 0;
    for (size_t j = i; j < n; ++j) {
      node = child(node, d[j]);
      if (node < 0) break;
      if (nodes_[size_t(node)].token >= 0) {
        best = nodes_[size_t(node)].token;
        best_end = j + 1;
      }
    }
    if (best >= 0) {
      mb = i;
      me = best_end;
      return best;
    }
  }
  return -1;
}

// ------------------------------------------------------------------ build
std::unique_ptr<Tokenizer::Impl> build(BuildInput&& in) {
  auto im = std::make_unique<Tokenizer::Impl>();
  Tokenizer::Impl& t = *im;
  t.pattern = in.pattern;
  t.nfc = in.nfc;
  t.ignore_merges = in.ignore_merges;
  t.spm = in.spm;
  t.fuse_unk = in.fuse_unk;
  t.text_arena = std::move(in.arena);

  int32_t max_vocab_id = -1;
  for (const auto& v : in.vocab) {
    if (v.second < 0) throw std::runtime_error("tokenizer: negative token id in vocabulary");
    max_vocab_id = std::max(max_vocab_id, v.second);
  }
  if (max_vocab_id < 0) throw std::runtime_error("tokenizer: empty vocabulary");
  // Arrays are indexed by id: refuse absurdly sparse ids instead of allocating for them.
  if (size_t(max_vocab_id) > 4 * in.vocab.size() + 65536)
    throw std::runtime_error("tokenizer: token ids are too sparse (max id " + std::to_string(max_vocab_id) + " for " +
                             std::to_string(in.vocab.size()) + " tokens)");
  t.text_off.assign(size_t(max_vocab_id) + 1, 0);
  t.text_len.assign(size_t(max_vocab_id) + 1, 0);
  t.kind.assign(size_t(max_vocab_id) + 1, Tokenizer::Impl::kNone);

  // Model vocabulary index. Later duplicates of a text win, as in the reference's map.
  const size_t cap = pow2_at_least(in.vocab.size() * 2 + 1);
  t.vocab_slots.assign(cap, 0);
  t.vocab_mask = cap - 1;
  int64_t vocab_count = 0;  // distinct texts: the reference's model.get_vocab_size()
  for (const auto& [tx, id] : in.vocab) {
    const std::string_view s(t.text_arena.data() + tx.off, tx.len);
    if (s.empty()) throw std::runtime_error("tokenizer: empty token text for id " + std::to_string(id));
    const uint64_t h = hash_bytes(s.data(), s.size());
    size_t i = h & t.vocab_mask;
    bool dup = false;
    for (;; i = (i + 1) & t.vocab_mask) {
      const uint64_t sl = t.vocab_slots[i];
      if (sl == 0) break;
      if ((sl >> 32) == (h >> 32) && t.text(int32_t(uint32_t(sl) - 1)) == s) {
        t.text_len[uint32_t(sl) - 1] = 0;  // the earlier id loses its text
        t.kind[uint32_t(sl) - 1] = Tokenizer::Impl::kNone;
        dup = true;
        break;
      }
    }
    vocab_count += dup ? 0 : 1;
    t.vocab_slots[i] = ((h >> 32) << 32) | uint64_t(uint32_t(id) + 1);
    t.text_off[size_t(id)] = tx.off;
    t.text_len[size_t(id)] = tx.len;
    t.kind[size_t(id)] = Tokenizer::Impl::kNormal;
  }

  // Added tokens. Explicit ids (GGUF) are kept. Tokens without one (tokenizer.json,
  // whose "id" fields the reference ignores) get ids like AddedVocabulary::add_tokens:
  // known content keeps its id (added tokens first, then the vocabulary), new content
  // takes max(vocabulary size, largest added id + 1); an identical definition is
  // ignored and a redefinition replaces the flags.
  {
    std::vector<AddedToken> list;
    std::unordered_map<std::string, int32_t> ids;
    int64_t max_added = -1;
    for (AddedToken& a : in.added) {
      std::string normed;
      a.text = (a.normalized && in.nfc && nfc_normalize(a.content, normed)) ? std::move(normed) : a.content;
      if (a.content.empty()) continue;
      if (a.id < 0) {
        bool same = false;
        for (const AddedToken& o : list)
          same |= o.content == a.content && o.special == a.special && o.lstrip == a.lstrip && o.rstrip == a.rstrip &&
                  o.single_word == a.single_word && o.normalized == a.normalized;
        if (same) continue;
        int64_t id;
        if (auto it = ids.find(a.content); it != ids.end()) id = it->second;
        else if (const int32_t v = t.vocab_find(a.content); v >= 0) id = v;
        else if (max_added < 0) id = vocab_count;
        else id = (max_added >= vocab_count || vocab_count == 0) ? max_added + 1 : vocab_count;
        if (id > INT32_MAX) throw std::runtime_error("tokenizer: too many tokens");
        a.id = int32_t(id);
        bool replaced = false;
        for (AddedToken& o : list)
          if (o.content == a.content) {
            // The reference's special-token set only grows: a token once added as
            // special stays special (for parse_special=false and skip_special).
            a.special = a.special || o.special;
            o = a;
            replaced = true;
          }
        if (!replaced) list.push_back(a);
      } else {
        list.push_back(a);
      }
      ids[a.content] = a.id;
      max_added = std::max<int64_t>(max_added, a.id);
    }
    in.added = std::move(list);
  }
  int32_t max_id = max_vocab_id;
  for (const auto& a : in.added) max_id = std::max(max_id, a.id);
  const size_t n = size_t(max_id) + 1;
  t.n_vocab = max_id + 1;
  t.text_off.resize(n, 0);
  t.text_len.resize(n, 0);
  t.kind.resize(n, Tokenizer::Impl::kNone);

  if (t.spm) {
    // Byte-fallback tokens <0xHH> (-1: a byte without one becomes unk).
    for (int b = 0; b < 256; ++b) {
      char name[8];
      std::snprintf(name, sizeof(name), "<0x%02X>", b);
      t.byte_id[b] = t.vocab_find(name);
    }
    t.unk = in.unk_token.empty() ? -1 : t.vocab_find(in.unk_token);
    // The word / line boundaries no vocabulary token spans (see Impl::spm_before_sp).
    for (const auto& [tx, id] : in.vocab) {
      (void)id;
      const auto* p = reinterpret_cast<const unsigned char*>(t.text_arena.data() + tx.off);
      const auto* e = p + tx.len;
      uint32_t prev = kInvalidCp;
      while (p < e) {
        uint32_t cp;
        p += decode_utf8(p, e, cp);
        if (prev != kInvalidCp) {
          if (cp == 0x2581 && prev != 0x2581) t.spm_before_sp.push_back(prev);
          if (cp == '\n' && prev != '\n') t.spm_before_nl.push_back(prev);
          if (prev == '\n' && cp != '\n') t.spm_after_nl.push_back(cp);
        }
        prev = cp;
      }
    }
    for (auto* v : {&t.spm_before_sp, &t.spm_before_nl, &t.spm_after_nl}) {
      std::sort(v->begin(), v->end());
      v->erase(std::unique(v->begin(), v->end()), v->end());
    }
  } else {
    // Byte tokens: every byte must have a single-character token.
    const ByteLevelMap& bl = byte_level();
    for (int b = 0; b < 256; ++b) {
      std::string s;
      append_utf8(s, bl.byte_to_cp[b]);
      const int32_t id = t.vocab_find(s);
      if (id < 0) throw std::runtime_error("tokenizer: vocabulary lacks the byte-level token for byte " + std::to_string(b));
      t.byte_id[b] = id;
    }
  }

  // Merges: resolve texts to ids; the merged token is the concatenation.
  t.merges.init(in.merges.size());
  std::string joined;
  for (size_t r = 0; r < in.merges.size(); ++r) {
    const auto& [ta, tb] = in.merges[r];
    const std::string_view a(in.merge_arena.data() + ta.off, ta.len);
    const std::string_view b(in.merge_arena.data() + tb.off, tb.len);
    const int32_t ia = t.vocab_find(a), ib = t.vocab_find(b);
    joined.assign(a);
    joined.append(b);
    const int32_t im2 = t.vocab_find(joined);
    if (ia < 0 || ib < 0 || im2 < 0)
      throw std::runtime_error("tokenizer: merge #" + std::to_string(r) + " '" + std::string(a) + " " + std::string(b) +
                               "' uses a token missing from the vocabulary");
    t.merges.put(uint32_t(ia), uint32_t(ib), uint32_t(r), uint32_t(im2));
  }

  // Added tokens: their content is their text; decoded like any token.
  t.added = std::move(in.added);
  for (const AddedToken& a : t.added) {
    t.added_index[a.content] = a.id;
    t.text_off[size_t(a.id)] = uint32_t(t.text_arena.size());
    t.text_len[size_t(a.id)] = uint32_t(a.text.size());
    t.text_arena += a.text;
    t.kind[size_t(a.id)] = a.special ? Tokenizer::Impl::kSpecial : Tokenizer::Impl::kAdded;
  }
  t.raw_matcher.build(t.added, false);
  t.norm_matcher.build(t.added, true);

  // Decoded bytes of every token.
  t.pieces.resize(n);
  for (size_t id = 0; id < n; ++id)
    if (t.text_len[id]) t.pieces[id] = t.spm ? spm_decode(t.text(int32_t(id))) : byte_level_decode(t.text(int32_t(id)));

  return im;
}

}  // namespace tok

// ------------------------------------------------------------------ Impl
int32_t Tokenizer::Impl::vocab_find(std::string_view s) const {
  if (vocab_slots.empty()) return -1;
  const uint64_t h = tok::hash_bytes(s.data(), s.size());
  for (size_t i = h & vocab_mask;; i = (i + 1) & vocab_mask) {
    const uint64_t sl = vocab_slots[i];
    if (sl == 0) return -1;
    if ((sl >> 32) == (h >> 32)) {
      const int32_t id = int32_t(uint32_t(sl) - 1);
      if (text(id) == s) return id;
    }
  }
}

int32_t Tokenizer::Impl::find(std::string_view s) const {
  if (auto it = added_index.find(s); it != added_index.end()) return it->second;
  return vocab_find(s);
}

namespace {

// Words up to this many bytes use per-thread BPE scratch buffers.
constexpr size_t kScratchLimit = 4096;

// Rust's char::is_whitespace (the White_Space property) == our \s class.
inline bool is_ws(uint32_t cp) { return cp != tok::kInvalidCp && tok::cp_class(cp) == tok::ucd::kSpace; }

// Approximation of the reference's `\w` test for single_word added tokens
// (Alphabetic, marks, decimal digits, connector punctuation, join controls).
// It also counts No/Nl numbers as word characters; no Qwen token uses single_word.
inline bool is_word_char(uint32_t cp) {
  if (cp == '_' || cp == 0x200C || cp == 0x200D) return true;
  if (cp == tok::kInvalidCp) return false;
  const uint8_t c = tok::cp_class(cp);
  return c == tok::ucd::kLetter || c == tok::ucd::kMark || c == tok::ucd::kNumber;
}

inline uint32_t first_cp(std::string_view s) {
  uint32_t cp;
  tok::decode_utf8(reinterpret_cast<const unsigned char*>(s.data()),
                   reinterpret_cast<const unsigned char*>(s.data()) + s.size(), cp);
  return cp;
}

inline uint32_t last_cp(std::string_view s, size_t& start) {
  size_t i = s.size();
  while (i > 0 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80 && s.size() - i < 3) --i;
  if (i > 0) --i;
  uint32_t cp;
  const int len = tok::decode_utf8(reinterpret_cast<const unsigned char*>(s.data()) + i,
                                   reinterpret_cast<const unsigned char*>(s.data()) + s.size(), cp);
  if (i + size_t(len) != s.size()) {  // malformed tail: the last byte stands alone
    i = s.size() - 1;
    cp = tok::kInvalidCp;
  }
  start = i;
  return cp;
}

}  // namespace

// Splits s around the added tokens of matcher m, exactly like the reference's
// AddedVocabulary::find_matches: leftmost-longest matches, scanning resumes at
// a match's end even when the match is skipped (special tokens with
// parse_special=false, failed single_word checks); lstrip/rstrip absorb
// surrounding whitespace.
template <class OnText>
void Tokenizer::Impl::split_added(const tok::AddedMatcher& m, std::string_view s, bool parse_special,
                                  std::vector<int32_t>& out, OnText&& on_text) const {
  size_t start_offset = 0, pos = 0, mb = 0, me = 0;
  while (pos < s.size()) {
    const int k = m.find(s, pos, mb, me);
    if (k < 0) break;
    pos = me;
    const tok::AddedToken& a = added[size_t(k)];
    if (!parse_special && a.special) continue;
    size_t start = mb, stop = me;
    if (a.single_word) {
      size_t ls;
      const bool start_ok = start == 0 || !is_word_char(last_cp(s.substr(0, start), ls));
      const bool stop_ok = stop == s.size() || !is_word_char(first_cp(s.substr(stop)));
      if (!start_ok || !stop_ok) continue;
    }
    if (a.lstrip) {
      size_t i = start;
      while (i > 0) {
        size_t ls;
        const uint32_t cp = last_cp(s.substr(0, i), ls);
        if (!is_ws(cp)) break;
        i = ls;
      }
      start = std::max(i, start_offset);
    }
    if (a.rstrip) {
      while (stop < s.size()) {
        uint32_t cp;
        const int len = tok::decode_utf8(reinterpret_cast<const unsigned char*>(s.data()) + stop,
                                         reinterpret_cast<const unsigned char*>(s.data()) + s.size(), cp);
        if (!is_ws(cp)) break;
        stop += size_t(len);
      }
    }
    if (start > start_offset) on_text(s.substr(start_offset, start - start_offset));
    out.push_back(a.id);
    start_offset = stop;
  }
  if (start_offset < s.size()) on_text(s.substr(start_offset));
}

void Tokenizer::Impl::encode(std::string_view text, std::vector<int32_t>& out, bool parse_special) const {
  if (text.empty()) return;
  // ~3.5 bytes per token on typical text; grow geometrically so that many small
  // encode_append calls on one vector stay amortized O(1).
  const size_t need = out.size() + text.size() / 3 + 8;
  if (out.capacity() < need) out.reserve(std::max(need, out.capacity() * 2));
  if (raw_matcher.empty()) {
    encode_segment(text, out, parse_special);
    return;
  }
  split_added(raw_matcher, text, parse_special, out,
              [&](std::string_view seg) { encode_segment(seg, out, parse_special); });
}

// A stretch of text between non-normalized added tokens: normalize it, split
// out normalized added tokens, pre-tokenize the rest.
void Tokenizer::Impl::encode_segment(std::string_view seg, std::vector<int32_t>& out, bool parse_special) const {
  std::string normalized;
  if (spm) {  // the Replace normalizer: every space becomes U+2581
    normalized.reserve(seg.size() + seg.size() / 4);
    for (char ch : seg) {
      if (ch == ' ') normalized += "\xE2\x96\x81";
      else normalized.push_back(ch);
    }
    seg = normalized;
  } else if (nfc && tok::nfc_normalize(seg, normalized)) {
    seg = normalized;
  }
  if (norm_matcher.empty()) {
    encode_pieces(seg, out);
    return;
  }
  split_added(norm_matcher, seg, parse_special, out, [&](std::string_view s) { encode_pieces(s, out); });
}

void Tokenizer::Impl::encode_pieces(std::string_view s, std::vector<int32_t>& out) const {
  if (spm) {
    // No pre-tokenizer: the segment is one BPE word, encoded in pieces at the
    // boundaries no token spans (same ids; each piece is short and cached).
    auto in = [](const std::vector<uint32_t>& v, uint32_t cp) { return std::binary_search(v.begin(), v.end(), cp); };
    const auto* b = reinterpret_cast<const unsigned char*>(s.data());
    const auto* e = b + s.size();
    size_t start = 0, i = 0;
    uint32_t prev = tok::kInvalidCp;
    while (i < s.size()) {
      uint32_t cp;
      const int len = tok::decode_utf8(b + i, e, cp);
      if (i > start && prev != tok::kInvalidCp && cp != tok::kInvalidCp) {
        const bool cut = (cp == 0x2581 && prev != 0x2581 && !in(spm_before_sp, prev)) ||
                         (cp == '\n' && prev != '\n' && !in(spm_before_nl, prev)) ||
                         (prev == '\n' && cp != '\n' && !in(spm_after_nl, cp));
        if (cut) {
          encode_word(s.substr(start, i - start), out);
          start = i;
        }
      }
      prev = cp;
      i += size_t(len);
    }
    if (start < s.size()) encode_word(s.substr(start), out);
    return;
  }
  tok::PieceSplitter sp(pattern, s);
  size_t b, e;
  while (sp.next(b, e)) encode_word(s.substr(b, e - b), out);
}

void Tokenizer::Impl::encode_word(std::string_view word, std::vector<int32_t>& out) const {
  if (spm) {
    const uint64_t h = tok::hash_bytes(word.data(), word.size());
    if (cache.get(word, h, out)) return;
    const size_t before = out.size();
    bpe_spm(word, out);
    cache.put(word, h, out.data() + before, out.size() - before);
    return;
  }
  if (word.size() == 1) {  // a single byte cannot merge
    out.push_back(byte_id[static_cast<unsigned char>(word[0])]);
    return;
  }
  const uint64_t h = tok::hash_bytes(word.data(), word.size());
  if (cache.get(word, h, out)) return;
  const size_t before = out.size();
  int32_t whole = -1;
  if (ignore_merges) {
    // The whole word as one vocabulary token, before any merging (Llama-3 style).
    std::string text;
    text.reserve(word.size() * 2);
    for (unsigned char c : word) tok::append_utf8(text, tok::byte_level().byte_to_cp[c]);
    whole = vocab_find(text);
  }
  if (whole >= 0) out.push_back(whole);
  else bpe(word, out);
  cache.put(word, h, out.data() + before, out.size() - before);
}

// BPE exactly as the reference's Word::merge_all: a min-heap of candidate merges
// ordered by (rank, position); stale entries are skipped when the symbol was
// consumed, is last, or no longer forms a pair that merges into the same token.
void Tokenizer::Impl::bpe(std::string_view word, std::vector<int32_t>& out) const {
  struct Sym {
    int32_t id;
    int32_t prev;
    int32_t next;
    int32_t len;  // 0: merged into its left neighbor
  };
  struct Cand {
    uint32_t rank;
    uint32_t pos;
    uint32_t merged;
  };
  // Heap comparator: std heaps are max-heaps, so "less" means larger (rank, pos).
  const auto later = [](const Cand& x, const Cand& y) { return x.rank != y.rank ? x.rank > y.rank : x.pos > y.pos; };
  // Per-thread scratch for ordinary words; a huge word gets its own buffers so the
  // thread does not keep megabytes around afterwards.
  const size_t n = word.size();
  thread_local std::vector<Sym> tl_syms;
  thread_local std::vector<Cand> tl_heap;
  std::vector<Sym> big_syms;
  std::vector<Cand> big_heap;
  std::vector<Sym>& syms = n <= kScratchLimit ? tl_syms : big_syms;
  std::vector<Cand>& heap = n <= kScratchLimit ? tl_heap : big_heap;
  syms.resize(n);
  heap.clear();
  for (size_t i = 0; i < n; ++i)
    syms[i] = Sym{byte_id[static_cast<unsigned char>(word[i])], int32_t(i) - 1, i + 1 < n ? int32_t(i + 1) : -1, 1};
  uint32_t rank, merged;
  for (size_t i = 0; i + 1 < n; ++i)
    if (merges.get(uint32_t(syms[i].id), uint32_t(syms[i + 1].id), rank, merged))
      heap.push_back(Cand{rank, uint32_t(i), merged});
  std::make_heap(heap.begin(), heap.end(), later);
  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), later);
    const Cand top = heap.back();
    heap.pop_back();
    Sym& cur = syms[top.pos];
    if (cur.len == 0 || cur.next < 0) continue;
    const Sym right = syms[size_t(cur.next)];
    if (!merges.get(uint32_t(cur.id), uint32_t(right.id), rank, merged) || merged != top.merged) continue;
    syms[size_t(cur.next)].len = 0;
    cur.id = int32_t(top.merged);
    cur.len += right.len;
    cur.next = right.next;
    if (right.next >= 0) syms[size_t(right.next)].prev = int32_t(top.pos);
    if (cur.prev >= 0 && merges.get(uint32_t(syms[size_t(cur.prev)].id), uint32_t(cur.id), rank, merged)) {
      heap.push_back(Cand{rank, uint32_t(cur.prev), merged});
      std::push_heap(heap.begin(), heap.end(), later);
    }
    if (cur.next >= 0 && merges.get(uint32_t(cur.id), uint32_t(syms[size_t(cur.next)].id), rank, merged)) {
      heap.push_back(Cand{rank, top.pos, merged});
      std::push_heap(heap.begin(), heap.end(), later);
    }
  }
  for (size_t i = 0; i < n; ++i)
    if (syms[i].len) out.push_back(syms[i].id);
}

// SentencePiece-style BPE: the reference's BPE::merge_word with
// byte_fallback: one symbol per character (its vocabulary token), else one per
// byte (the byte-fallback tokens), else unk (consecutive unknown characters
// fused into one when fuse_unk); then the same merges by (rank, position) as
// bpe(). Invalid UTF-8 bytes are characters of their own (the reference only
// sees valid text).
void Tokenizer::Impl::bpe_spm(std::string_view word, std::vector<int32_t>& out) const {
  struct Sym {
    int32_t id;
    int32_t prev;
    int32_t next;
    int32_t len;  // 0: merged into its left neighbor
  };
  struct Cand {
    uint32_t rank;
    uint32_t pos;
    uint32_t merged;
  };
  const auto later = [](const Cand& x, const Cand& y) { return x.rank != y.rank ? x.rank > y.rank : x.pos > y.pos; };
  const size_t n = word.size();
  thread_local std::vector<Sym> tl_syms;
  thread_local std::vector<Cand> tl_heap;
  std::vector<Sym> big_syms;
  std::vector<Cand> big_heap;
  std::vector<Sym>& syms = n <= kScratchLimit ? tl_syms : big_syms;
  std::vector<Cand>& heap = n <= kScratchLimit ? tl_heap : big_heap;
  syms.clear();
  heap.clear();
  auto push = [&](int32_t id, int32_t len) {
    const int32_t i = int32_t(syms.size());
    syms.push_back(Sym{id, i - 1, -1, len});
    if (i > 0) syms[size_t(i - 1)].next = i;
  };
  const auto* p = reinterpret_cast<const unsigned char*>(word.data());
  const auto* e = p + n;
  bool in_unk = false;  // the last symbol is an unk that may absorb the next unknown character
  while (p < e) {
    uint32_t cp;
    int len = tok::decode_utf8(p, e, cp);
    if (len <= 0) len = 1;
    const std::string_view ch(reinterpret_cast<const char*>(p), size_t(len));
    const int32_t id = vocab_find(ch);
    if (id >= 0) {
      push(id, len);
      in_unk = false;
    } else {
      bool all = true;
      for (int k = 0; k < len; ++k) all = all && byte_id[p[k]] >= 0;
      if (all) {
        for (int k = 0; k < len; ++k) push(byte_id[p[k]], 1);
        in_unk = false;
      } else if (unk >= 0) {
        if (in_unk && fuse_unk) syms.back().len += len;
        else push(unk, len);
        in_unk = true;
      }
    }
    p += len;
  }
  const size_t ns = syms.size();
  uint32_t rank, merged;
  for (size_t i = 0; i + 1 < ns; ++i)
    if (merges.get(uint32_t(syms[i].id), uint32_t(syms[i + 1].id), rank, merged))
      heap.push_back(Cand{rank, uint32_t(i), merged});
  std::make_heap(heap.begin(), heap.end(), later);
  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), later);
    const Cand top = heap.back();
    heap.pop_back();
    Sym& cur = syms[top.pos];
    if (cur.len == 0 || cur.next < 0) continue;
    const Sym right = syms[size_t(cur.next)];
    if (!merges.get(uint32_t(cur.id), uint32_t(right.id), rank, merged) || merged != top.merged) continue;
    syms[size_t(cur.next)].len = 0;
    cur.id = int32_t(top.merged);
    cur.len += right.len;
    cur.next = right.next;
    if (right.next >= 0) syms[size_t(right.next)].prev = int32_t(top.pos);
    if (cur.prev >= 0 && merges.get(uint32_t(syms[size_t(cur.prev)].id), uint32_t(cur.id), rank, merged)) {
      heap.push_back(Cand{rank, uint32_t(cur.prev), merged});
      std::push_heap(heap.begin(), heap.end(), later);
    }
    if (cur.next >= 0 && merges.get(uint32_t(cur.id), uint32_t(syms[size_t(cur.next)].id), rank, merged)) {
      heap.push_back(Cand{rank, top.pos, merged});
      std::push_heap(heap.begin(), heap.end(), later);
    }
  }
  for (size_t i = 0; i < ns; ++i)
    if (syms[i].len) out.push_back(syms[i].id);
}

// ------------------------------------------------------------------ public API
Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Tokenizer::~Tokenizer() = default;

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool parse_special) const {
  std::vector<int32_t> out;
  impl_->encode(text, out, parse_special);
  return out;
}

void Tokenizer::encode_append(std::string_view text, std::vector<int32_t>& out, bool parse_special) const {
  impl_->encode(text, out, parse_special);
}

std::string Tokenizer::decode(const int32_t* ids, size_t n, bool skip_special) const {
  std::string out;
  const Impl& t = *impl_;
  for (size_t i = 0; i < n; ++i) {
    const int32_t id = ids[i];
    if (id < 0 || id >= t.n_vocab) continue;
    if (skip_special && t.kind[size_t(id)] == Impl::kSpecial) continue;
    out += t.pieces[size_t(id)];
  }
  return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
  return decode(ids.data(), ids.size(), skip_special);
}

const std::string& Tokenizer::piece(int32_t id) const {
  static const std::string empty;
  if (id < 0 || id >= impl_->n_vocab) return empty;
  return impl_->pieces[size_t(id)];
}

int32_t Tokenizer::token_id(std::string_view token_text) const { return impl_->find(token_text); }

bool Tokenizer::is_special(int32_t id) const {
  return id >= 0 && id < impl_->n_vocab && impl_->kind[size_t(id)] == Impl::kSpecial;
}

bool Tokenizer::is_added(int32_t id) const {
  return id >= 0 && id < impl_->n_vocab && impl_->kind[size_t(id)] >= Impl::kAdded;
}

std::string_view Tokenizer::token_text(int32_t id) const { return impl_->text(id); }
int32_t Tokenizer::vocab_size() const { return impl_->n_vocab; }
int32_t Tokenizer::eos_id() const { return impl_->eos; }
int32_t Tokenizer::bos_id() const { return impl_->bos; }
int32_t Tokenizer::pad_id() const { return impl_->pad; }
bool Tokenizer::add_bos() const { return impl_->add_bos && impl_->bos >= 0; }
std::vector<int32_t> Tokenizer::with_bos(std::vector<int32_t> ids) const {
  if (add_bos() && (ids.empty() || ids.front() != impl_->bos)) ids.insert(ids.begin(), impl_->bos);
  return ids;
}
const std::string& Tokenizer::chat_template() const { return impl_->chat_template; }
std::string_view Tokenizer::pre_tokenizer_name() const { return impl_->spm ? "spm" : tok::split_pattern_name(impl_->pattern); }

std::vector<int32_t> Tokenizer::special_ids() const {
  std::vector<int32_t> out;
  for (int32_t id = 0; id < impl_->n_vocab; ++id)
    if (impl_->kind[size_t(id)] == Impl::kSpecial) out.push_back(id);
  return out;
}

}  // namespace espetero
