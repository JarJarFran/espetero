// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Token selection from logits: penalties, temperature, top-k, min-p, top-p,
// sampling, the speculative accept/resample rule and prompt-lookup drafts.
#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace espetero {

struct SamplingParams {
  double temperature = 1.0;  // <= 0: greedy
  double top_p = 1.0;
  int top_k = 0;             // 0: off
  double min_p = 0.0;
  double repetition_penalty = 1.0;  // over prompt + output tokens
  double presence_penalty = 0.0;    // over output tokens
  double frequency_penalty = 0.0;
  int no_repeat_ngram_size = 0;     // > 0: no n-gram of the output twice (0: off)
  bool greedy() const { return temperature <= 0 && !penalized(); }
  bool penalized() const {
    return repetition_penalty != 1.0 || presence_penalty != 0 || frequency_penalty != 0 || no_repeat_ngram_size > 0;
  }
};

// Per-sequence penalty state: tokens seen (prompt + output), output counts
// and the output in order (no_repeat_ngram_size).
class PenaltyState {
 public:
  void add_prompt(const int32_t* ids, size_t n) {
    if (n == 0) return;
    reserve_seen(seen_size_ + n);
    for (size_t i = 0; i < n; ++i) find_or_insert(ids[i]);
  }
  void add_output(int32_t id) {
    ++seen_[find_or_insert(id)].count;
    out_.push_back(id);
  }
  // add_output that says whether the token was new, and its exact undo
  // (PenaltyScope).
  bool add_output_new(int32_t id) {
    bool fresh = false;
    const size_t slot = find_or_insert(id, &fresh);
    ++seen_[slot].count;
    out_.push_back(id);
    return fresh;
  }
  void undo_output(int32_t id, bool fresh) {
    if (!out_.empty() && out_.back() == id) out_.pop_back();
    const size_t slot = find_slot(id);
    if (slot == SIZE_MAX) return;
    if (fresh) {
      // Mark slot tombstoned (key = kTomb) so probing chains stay intact.
      seen_[slot] = {kTomb, 0};
      --seen_size_;
    } else {
      --seen_[slot].count;
    }
  }
  // Apply to a copy of `logits` (only the touched entries change). With
  // no_repeat_ngram_size n, every token that followed an earlier occurrence
  // of the output's last n - 1 tokens gets -inf (transformers'
  // NoRepeatNGramLogitsProcessor over the output tokens).
  void apply(float* logits, const SamplingParams& p) const;
  bool empty() const { return seen_size_ == 0 && out_.empty(); }

 private:
  static constexpr int32_t kEmpty = -1;
  static constexpr int32_t kTomb = -2;
  struct Entry {
    int32_t tok = kEmpty;
    int32_t count = 0;
  };
  static inline size_t hash_id(int32_t id) { return size_t(uint32_t(id) * 2654435761u); }
  void reserve_seen(size_t need) {
    if (seen_.size() * 2 >= need * 3 + 8) return;
    size_t cap = std::max<size_t>(32, seen_.size() * 2);
    while (cap * 2 < need * 3 + 8) cap *= 2;
    std::vector<Entry> next(cap);
    const size_t mask = cap - 1;
    for (const Entry& e : seen_) {
      if (e.tok < 0) continue;
      size_t i = hash_id(e.tok) & mask;
      while (next[i].tok != kEmpty) i = (i + 1) & mask;
      next[i] = e;
    }
    seen_ = std::move(next);
    seen_used_ = seen_size_;
  }
  size_t find_or_insert(int32_t id, bool* fresh = nullptr) {
    if ((seen_used_ + 1) * 3 >= seen_.size() * 2) reserve_seen(seen_size_ + 1);
    const size_t mask = seen_.size() - 1;
    size_t i = hash_id(id) & mask;
    size_t first_tomb = SIZE_MAX;
    for (;;) {
      const int32_t t = seen_[i].tok;
      if (t == id) {
        if (fresh) *fresh = false;
        return i;
      }
      if (t == kEmpty) {
        const size_t dst = first_tomb != SIZE_MAX ? first_tomb : i;
        if (dst == i) ++seen_used_;
        seen_[dst] = {id, 0};
        ++seen_size_;
        if (fresh) *fresh = true;
        return dst;
      }
      if (t == kTomb && first_tomb == SIZE_MAX) first_tomb = i;
      i = (i + 1) & mask;
    }
  }
  size_t find_slot(int32_t id) const {
    if (seen_.empty()) return SIZE_MAX;
    const size_t mask = seen_.size() - 1;
    size_t i = hash_id(id) & mask;
    for (;;) {
      const int32_t t = seen_[i].tok;
      if (t == id) return i;
      if (t == kEmpty) return SIZE_MAX;
      i = (i + 1) & mask;
    }
  }
  std::vector<Entry> seen_;  // open-addressed: token -> output count (0: prompt only)
  size_t seen_size_ = 0;
  size_t seen_used_ = 0;     // occupied + tombstones
  std::vector<int32_t> out_; // the output tokens, in order
};

// The tokens a verify pass produces, counted by the penalties of the
// readouts after them (depth d sees the drafts accepted at depths < d) and
// taken back when the scope ends: the pass's tokens are added for good when
// they are emitted. A null state (no penalties) does nothing.
class PenaltyScope {
 public:
  explicit PenaltyScope(PenaltyState* p) : p_(p) {}
  PenaltyScope(const PenaltyScope&) = delete;
  PenaltyScope& operator=(const PenaltyScope&) = delete;
  ~PenaltyScope() {
    for (auto it = added_.rbegin(); it != added_.rend(); ++it) p_->undo_output(it->first, it->second);
  }
  void add(int32_t id) {
    if (p_) added_.emplace_back(id, p_->add_output_new(id));
  }

 private:
  PenaltyState* p_;
  std::vector<std::pair<int32_t, bool>> added_;
};

// A sparse distribution: token ids (empty: the whole vocabulary in order)
// and their probabilities.
struct Distribution {
  std::vector<int32_t> ids;
  std::vector<double> p;
};

// Probabilities after temperature, then top-k, min-p and top-p. Top-p looks
// at the 2,048 most likely tokens first, normalized by the full
// distribution, and falls back to a full sort only if they do not reach p.
Distribution distribution(const float* logits, int vocab, const SamplingParams& sp);
// The same distribution from GPU top-k candidates: ids and scaled
// logits x sorted by (x desc, id asc), their maximum and the full-vocabulary
// sum of exp(x - max); valid for 1 <= top_k <= k (no penalties).
Distribution distribution_topk(const int32_t* ids, const float* x, int k, float max, float total, const SamplingParams& sp);

int32_t argmax(const float* logits, int vocab);

// Exact position-keyed sampling (the scheme of TensorFold, MIT): the
// token at absolute position `pos` is a fixed function of the distribution,
// the request's seed and the position. Head distributions (explicit ids):
// argmax_i log p_i + G(seed, pos, id_i), G a standard Gumbel variate from a
// splitmix64 hash of (seed, pos, id) (Gumbel-max). The whole vocabulary (no
// top-k/top-p/min-p): inverse CDF in id order at the hashed uniform U(seed,
// pos, -1). Either is an exact draw from d, and a verify row and the serial
// step at the same position give the same token (up to floating-point
// near-ties between the two rows' logits), so a draft is accepted exactly
// when it equals that token: the acceptance probability is p(draft), as with
// the random test.
double hashed_uniform(uint64_t seed, int64_t pos, int64_t id);
int32_t sample_at(const Distribution& d, uint64_t seed, int64_t pos);
int32_t sample(const Distribution& d, std::mt19937_64& rng);

// Speculative sampling with a deterministic (greedy) draft: accept it with
// probability p(draft), else sample from p without it. The emitted token is
// distributed exactly as p.
std::pair<bool, int32_t> verify_draft(const Distribution& d, int32_t draft, std::mt19937_64& rng);

// Prompt lookup: continuations of the context's last n tokens (n = 4, then 3)
// at their latest earlier occurrence. Incremental: O(1) per appended token.
class NgramIndex {
 public:
  void append(int32_t t);
  void extend(const int32_t* ids, size_t n) {
    for (size_t i = 0; i < n; ++i) append(ids[i]);
  }
  // The tokens that followed the latest earlier occurrence of the context's
  // last 4 (else 3) tokens, up to max_len, if that occurrence matches the
  // context for at least min_match tokens (the gram extended backwards, up
  // to 64): a copy "backed by min_match matching tokens".
  std::vector<int32_t> lookup(int max_len, int min_match = 3) const;
  size_t size() const { return ctx_.size(); }
  const std::vector<int32_t>& tokens() const { return ctx_; }

 private:
  static uint64_t key(const int32_t* t, int n);
  std::vector<int32_t> ctx_;
  std::unordered_map<uint64_t, int32_t> last3_, last4_;  // gram -> latest start with a follower
};

}  // namespace espetero
