// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Internal data structures of espetero::Tokenizer (shared by tokenizer.cpp,
// load_hf.cpp and gguf_vocab.cpp). Not part of the public API.
#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pretok.h"
#include "tokenizer.h"

#if defined(__APPLE__)
#include <os/lock.h>
#endif

namespace espetero::tok {

// 64-bit hash for short byte strings (vocabulary texts, cached words).
inline uint64_t hash_bytes(const void* data, size_t n) {
  const auto* p = static_cast<const unsigned char*>(data);
  uint64_t h = 0x9E3779B97F4A7C15ull ^ (uint64_t(n) * 0xC2B2AE3D27D4EB4Full);
  while (n >= 8) {
    uint64_t w;
    std::memcpy(&w, p, 8);
    h = (h ^ w) * 0xFF51AFD7ED558CCDull;
    h = (h << 31) | (h >> 33);
    p += 8;
    n -= 8;
  }
  if (n) {
    uint64_t w = 0;
    std::memcpy(&w, p, n);
    h = (h ^ w) * 0xC4CEB9FE1A85EC53ull;
  }
  // murmur3 finalizer: every input bit affects the low bits used for slots.
  h ^= h >> 33;
  h *= 0xFF51AFD7ED558CCDull;
  h ^= h >> 33;
  h *= 0xC4CEB9FE1A85EC53ull;
  h ^= h >> 33;
  return h;
}

inline size_t pow2_at_least(size_t n) {
  size_t p = 16;
  while (p < n) p <<= 1;
  return p;
}

// GPT-2 bytes_to_unicode: printable Latin-1 bytes map to themselves, the other
// 68 bytes to U+0100.. in byte order. Every byte-level token text is a string
// over these 256 code points.
struct ByteLevelMap {
  uint16_t byte_to_cp[256];
  int16_t cp_to_byte[324];  // -1: not in the alphabet
  ByteLevelMap() {
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      const bool keep = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
      byte_to_cp[b] = uint16_t(keep ? b : 256 + n++);
    }
    for (auto& v : cp_to_byte) v = -1;
    for (int b = 0; b < 256; ++b) cp_to_byte[byte_to_cp[b]] = int16_t(b);
  }
};
const ByteLevelMap& byte_level();

// Raw bytes of a token text under the ByteLevel decoder rule of the reference:
// if every code point is in the byte-level alphabet, map them back to bytes;
// otherwise the text's own UTF-8 is used unchanged.
std::string byte_level_decode(std::string_view text);

// Short lock for the cache shards (uncontended in practice).
class ShardLock {
 public:
#if defined(__APPLE__)
  void lock() { os_unfair_lock_lock(&l_); }
  void unlock() { os_unfair_lock_unlock(&l_); }

 private:
  os_unfair_lock l_ = OS_UNFAIR_LOCK_INIT;
#else
  void lock() { m_.lock(); }
  void unlock() { m_.unlock(); }

 private:
  std::mutex m_;
#endif
};

// (left id, right id) -> (rank, merged id), open addressing with linear probing.
class MergeTable {
 public:
  void init(size_t n_merges) {
    const size_t cap = pow2_at_least(n_merges * 2 + 1);
    slots_.assign(cap, Slot{kEmpty, 0, 0});
    mask_ = cap - 1;
  }
  // Inserts or overwrites (the reference keeps the last rank of a duplicated pair).
  void put(uint32_t a, uint32_t b, uint32_t rank, uint32_t merged) {
    const uint64_t key = (uint64_t(a) << 32) | b;
    for (size_t i = slot_of(key);; i = (i + 1) & mask_) {
      if (slots_[i].key == kEmpty || slots_[i].key == key) {
        slots_[i] = Slot{key, rank, merged};
        return;
      }
    }
  }
  bool get(uint32_t a, uint32_t b, uint32_t& rank, uint32_t& merged) const {
    const uint64_t key = (uint64_t(a) << 32) | b;
    for (size_t i = slot_of(key);; i = (i + 1) & mask_) {
      const Slot& s = slots_[i];
      if (s.key == key) {
        rank = s.rank;
        merged = s.merged;
        return true;
      }
      if (s.key == kEmpty) return false;
    }
  }

 private:
  static constexpr uint64_t kEmpty = ~0ull;
  struct Slot {
    uint64_t key;
    uint32_t rank;
    uint32_t merged;
  };
  size_t slot_of(uint64_t key) const { return size_t((key * 0x9E3779B97F4A7C15ull) >> 40) & mask_; }
  std::vector<Slot> slots_;
  size_t mask_ = 0;
};

// Thread-safe cache word bytes -> BPE ids. Sharded by hash so concurrent encoders
// rarely touch the same lock; each shard is a flat table over two arenas and is
// simply cleared when full (bounded memory, no per-entry allocation).
class WordCache {
 public:
  // Longer words are not cached (they are rare, and each would take a big share of a shard).
  static constexpr size_t kMaxKeyLen = 1024;

  WordCache() = default;
  bool get(std::string_view key, uint64_t h, std::vector<int32_t>& out) const;
  void put(std::string_view key, uint64_t h, const int32_t* ids, size_t n);

 private:
  // 64 shards x 1024 words; at most ~7 MB when every shard is full.
  static constexpr size_t kShards = 64;
  static constexpr size_t kSlots = 2048;  // per shard, power of two, load factor <= 0.5
  static constexpr size_t kMaxEntries = kSlots / 2;
  static constexpr size_t kMaxKeyBytes = 64 * 1024;
  struct Entry {
    uint64_t hash;  // 0: empty slot
    uint32_t key_off;
    uint32_t ids_off;
    uint16_t key_len;
    uint16_t n_ids;
  };
  struct alignas(64) Shard {
    mutable ShardLock lock;
    std::vector<Entry> slots;  // allocated on first insert
    std::string keys;
    std::vector<int32_t> ids;
    size_t count = 0;
  };
  mutable Shard shards_[kShards];
};

// Transparent hash so string_view lookups need no temporary std::string.
struct StringHash {
  using is_transparent = void;
  size_t operator()(std::string_view s) const { return size_t(hash_bytes(s.data(), s.size())); }
};

struct AddedToken {
  std::string content;  // as defined: what token_id() looks up
  // What is matched in the text and decoded: the content, NFC-normalized for
  // normalized=true tokens. The reference is asymmetric the same way (token_to_id
  // takes the raw content, id_to_token/decode give the normalized one).
  std::string text;
  int32_t id = -1;
  bool special = false;
  bool lstrip = false;
  bool rstrip = false;
  bool single_word = false;
  bool normalized = false;
};

// Leftmost-longest search over a set of added tokens (the Aho-Corasick
// MatchKind::LeftmostLongest semantics of the reference), using a byte trie.
class AddedMatcher {
 public:
  // Indexes the text of every token whose `normalized` flag equals the argument;
  // find() returns indexes into `tokens`.
  void build(const std::vector<AddedToken>& tokens, bool normalized);
  bool empty() const { return nodes_.size() <= 1; }
  // First match at or after `from`: returns the index into the token list (or -1)
  // and the match's byte range [mb, me).
  int find(std::string_view s, size_t from, size_t& mb, size_t& me) const;

 private:
  struct Node {
    int32_t first_child = -1;
    int32_t next_sibling = -1;
    int32_t token = -1;
    uint8_t byte = 0;
  };
  int32_t child(int32_t node, uint8_t byte) const {
    for (int32_t c = nodes_[size_t(node)].first_child; c >= 0; c = nodes_[size_t(c)].next_sibling)
      if (nodes_[size_t(c)].byte == byte) return c;
    return -1;
  }
  std::vector<Node> nodes_;
  bool first_byte_[256] = {};
  int single_first_byte_ = -1;  // >= 0 when all tokens share one first byte (memchr scan)
};

}  // namespace espetero::tok

namespace espetero {

struct Tokenizer::Impl {
  enum Kind : uint8_t { kNone = 0, kNormal = 1, kAdded = 2, kSpecial = 3 };

  // ---- vocabulary: model vocabulary and added tokens, indexed by id
  std::string text_arena;                    // token texts (byte-level form / added content)
  std::vector<uint32_t> text_off, text_len;  // text of an id; len 0: no token with this id
  std::vector<std::string> pieces;           // decoded bytes of each id
  std::vector<uint8_t> kind;
  std::vector<uint64_t> vocab_slots;  // model vocabulary index: (hash_hi32 << 32) | (id + 1)
  size_t vocab_mask = 0;
  std::unordered_map<std::string, int32_t, tok::StringHash, std::equal_to<>> added_index;  // content -> id
  std::vector<tok::AddedToken> added;
  tok::AddedMatcher raw_matcher;   // added tokens with normalized=false (matched before NFC)
  tok::AddedMatcher norm_matcher;  // normalized=true (matched on the normalized text)
  int32_t n_vocab = 0;             // max id + 1

  // ---- model
  int32_t byte_id[256] = {};  // byte-level: the token of each byte; spm: its byte-fallback token <0xHH>
  tok::MergeTable merges;
  bool ignore_merges = false;
  bool nfc = true;
  // SentencePiece-style BPE (Gemma 4's tokenizer.json): spaces are
  // normalized to U+2581, there is no pre-tokenizer split and no byte-level
  // alphabet (symbols start as characters; one missing from the vocabulary
  // becomes the byte-fallback tokens of its bytes, else unk), and decoding
  // turns U+2581 back into spaces and <0xHH> tokens into bytes.
  bool spm = false;
  int32_t unk = -1;
  bool fuse_unk = false;
  // spm: characters that some vocabulary token holds right before U+2581 or
  // a newline, or right after a newline (sorted). A text splits exactly at
  // every other word / line boundary: no token, hence no merge, spans it (no
  // pre-tokenizer in the pipeline, so this is only a speed-up: encoding a
  // segment as one word gives the same ids).
  std::vector<uint32_t> spm_before_sp, spm_before_nl, spm_after_nl;
  tok::SplitPattern pattern = tok::SplitPattern::kQwen35;
  mutable tok::WordCache cache;

  // ---- metadata
  int32_t eos = -1, bos = -1, pad = -1;
  // Raw text (completions, embeddings) starts with BOS: GGUF
  // tokenizer.ggml.add_bos_token, tokenizer_config.json add_bos_token; when
  // the files do not say, llama.cpp's default (SentencePiece: yes, BPE: no).
  bool add_bos = false;
  std::string chat_template;

  std::string_view text(int32_t id) const {
    if (id < 0 || size_t(id) >= text_len.size() || text_len[size_t(id)] == 0) return {};
    return std::string_view(text_arena).substr(text_off[size_t(id)], text_len[size_t(id)]);
  }
  int32_t vocab_find(std::string_view s) const;  // model vocabulary only
  int32_t find(std::string_view s) const;        // added tokens first, then the vocabulary

  void encode(std::string_view text, std::vector<int32_t>& out, bool parse_special) const;
  void encode_segment(std::string_view seg, std::vector<int32_t>& out, bool parse_special) const;
  void encode_pieces(std::string_view s, std::vector<int32_t>& out) const;
  void encode_word(std::string_view word, std::vector<int32_t>& out) const;
  void bpe(std::string_view word, std::vector<int32_t>& out) const;
  void bpe_spm(std::string_view word, std::vector<int32_t>& out) const;
  template <class OnText>
  void split_added(const tok::AddedMatcher& m, std::string_view s, bool parse_special, std::vector<int32_t>& out,
                   OnText&& on_text) const;
};

namespace tok {

// Everything a loader collects; build() validates it and produces the Impl.
struct BuildInput {
  std::string arena;  // model vocabulary texts
  struct Text {
    uint32_t off, len;
  };
  std::vector<std::pair<Text, int32_t>> vocab;  // (text, id), in file order (last duplicate wins)
  std::string merge_arena;
  std::vector<std::pair<Text, Text>> merges;  // by rank
  std::vector<AddedToken> added;              // ids already resolved
  SplitPattern pattern = SplitPattern::kQwen35;
  bool nfc = true;
  bool ignore_merges = false;
  bool spm = false;          // SentencePiece-style BPE (Impl::spm)
  std::string unk_token;     // spm: the model's unk_token ("" = none)
  bool fuse_unk = false;
};

std::unique_ptr<Tokenizer::Impl> build(BuildInput&& in);

}  // namespace tok
}  // namespace espetero
