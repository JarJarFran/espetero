// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "sampling.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace espetero {

void PenaltyState::apply(float* logits, const SamplingParams& p) const {
  for (const auto& [tok, count] : seen_) {
    float& x = logits[tok];
    if (p.repetition_penalty != 1.0) x = x > 0 ? float(x / p.repetition_penalty) : float(x * p.repetition_penalty);
    if (count > 0 && (p.presence_penalty != 0 || p.frequency_penalty != 0))
      x -= float(p.presence_penalty + p.frequency_penalty * count);
  }
}

int32_t argmax(const float* logits, int vocab) {
  // vDSP returns the first index of the maximum, like numpy.
  float m = 0;
  vDSP_Length i = 0;
  vDSP_maxvi(logits, 1, &m, &i, vDSP_Length(vocab));
  return int32_t(i);
}

Distribution distribution(const float* logits, int vocab, const SamplingParams& sp) {
  Distribution d;
  if (sp.temperature <= 0) {
    d.ids = {argmax(logits, vocab)};
    d.p = {1.0};
    return d;
  }
  // x = logits / T; total = sum exp(x - max) over the whole vocabulary.
  thread_local std::vector<float> x, e;
  x.resize(size_t(vocab));
  e.resize(size_t(vocab));
  const float inv_t = float(1.0 / sp.temperature);
  vDSP_vsmul(logits, 1, &inv_t, x.data(), 1, vDSP_Length(vocab));
  float m = 0;
  vDSP_maxv(x.data(), 1, &m, vDSP_Length(vocab));
  const float neg_m = -m;
  vDSP_vsadd(x.data(), 1, &neg_m, e.data(), 1, vDSP_Length(vocab));
  int n = vocab;
  vvexpf(e.data(), e.data(), &n);
  double total = 0;
  {
    float t = 0;
    vDSP_sve(e.data(), 1, &t, vDSP_Length(vocab));
    total = t;
  }
  if (!std::isfinite(total)) {
    // NaN (or infinite) logits break the ordering the sorts below rely
    // on (undefined behavior); they count as -inf, never drawn (+inf as the
    // largest float).
    for (float& v : x)
      if (std::isnan(v)) v = -std::numeric_limits<float>::infinity();
      else if (v == std::numeric_limits<float>::infinity()) v = std::numeric_limits<float>::max();
    vDSP_maxv(x.data(), 1, &m, vDSP_Length(vocab));
    const float nm = -m;
    vDSP_vsadd(x.data(), 1, &nm, e.data(), 1, vDSP_Length(vocab));
    vvexpf(e.data(), e.data(), &n);
    float t = 0;
    vDSP_sve(e.data(), 1, &t, vDSP_Length(vocab));
    total = t > 0 ? t : 1;
  }
  const int k = sp.top_k > 0 ? std::min(sp.top_k, vocab) : vocab;
  if (k < vocab || sp.top_p < 1.0 || sp.min_p > 0) {
    const int head = std::min(k, 2048);
    std::vector<int32_t> idx(static_cast<size_t>(vocab));
    std::iota(idx.begin(), idx.end(), 0);
    auto by_logit = [&](int32_t a, int32_t b) { return x[size_t(a)] > x[size_t(b)] || (x[size_t(a)] == x[size_t(b)] && a < b); };
    std::nth_element(idx.begin(), idx.begin() + head - 1, idx.end(), by_logit);
    idx.resize(size_t(head));
    std::sort(idx.begin(), idx.end(), by_logit);
    std::vector<double> p(idx.size());
    double mass = 0;
    for (size_t i = 0; i < idx.size(); ++i) mass += (p[i] = double(e[size_t(idx[i])]) / total);
    if (k == vocab && sp.top_p < 1.0 && mass < sp.top_p && head < vocab) {
      // Rare: a flat distribution; sort everything.
      idx.resize(size_t(vocab));
      std::iota(idx.begin(), idx.end(), 0);
      std::sort(idx.begin(), idx.end(), by_logit);
      p.resize(idx.size());
      for (size_t i = 0; i < idx.size(); ++i) p[i] = double(e[size_t(idx[i])]) / total;
    }
    if (sp.min_p > 0) {
      const double floor = sp.min_p * p[0];
      size_t keep = 0;
      while (keep < p.size() && p[keep] >= floor) ++keep;
      idx.resize(keep);
      p.resize(keep);
    }
    if (sp.top_p < 1.0) {
      double sum = 0;
      for (double v : p) sum += v;
      const double target = sp.top_p * sum;
      double acc = 0;
      size_t cut = p.size();
      for (size_t i = 0; i < p.size(); ++i) {
        acc += p[i];
        if (acc >= target) {  // numpy searchsorted(cumsum, target) + 1
          cut = i + 1;
          break;
        }
      }
      idx.resize(cut);
      p.resize(cut);
    }
    double s = 0;
    for (double v : p) s += v;
    for (double& v : p) v /= s;
    d.ids = std::move(idx);
    d.p = std::move(p);
    return d;
  }
  d.p.resize(size_t(vocab));
  for (int i = 0; i < vocab; ++i) d.p[size_t(i)] = double(e[size_t(i)]) / total;
  return d;
}

Distribution distribution_topk(const int32_t* ids, const float* x, int k, float max, float total, const SamplingParams& sp) {
  // Mirrors distribution()'s head branch: p = exp(x - max) / total over the
  // top-k head, then min_p, top_p and renormalization.
  Distribution d;
  const int head = std::min(k, sp.top_k);
  std::vector<int32_t> idx(ids, ids + head);
  std::vector<double> p(static_cast<size_t>(head));
  for (int i = 0; i < head; ++i) p[size_t(i)] = double(std::exp(x[i] - max)) / double(total);
  if (sp.min_p > 0) {
    const double floor = sp.min_p * p[0];
    size_t keep = 0;
    while (keep < p.size() && p[keep] >= floor) ++keep;
    idx.resize(keep);
    p.resize(keep);
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
    idx.resize(cut);
    p.resize(cut);
  }
  double s = 0;
  for (double v : p) s += v;
  for (double& v : p) v /= s;
  d.ids = std::move(idx);
  d.p = std::move(p);
  return d;
}

namespace {
inline uint64_t mix64(uint64_t x) {
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
}  // namespace

double hashed_uniform(uint64_t seed, int64_t pos, int64_t id) {
  uint64_t x = mix64(seed + 0x9E3779B97F4A7C15ull);
  x = mix64(x ^ (uint64_t(pos) * 0xD1B54A32D192ED03ull));
  x = mix64(x ^ uint64_t(id));
  return double(x >> 11) * 0x1.0p-53 + 0x1.0p-54;  // (0, 1)
}

int32_t sample_at(const Distribution& d, uint64_t seed, int64_t pos) {
  if (d.p.empty()) return 0;
  if (d.ids.empty()) {
    // Whole vocabulary: one uniform, inverse CDF in id order.
    double total = 0;
    for (double v : d.p) total += v;
    const double r = hashed_uniform(seed, pos, -1) * total;
    double acc = 0;
    size_t j = 0;
    for (; j + 1 < d.p.size(); ++j) {
      acc += d.p[j];
      if (r < acc) break;
    }
    return int32_t(j);
  }
  // Head: Gumbel-max, the first (lowest-rank) candidate on exact ties.
  double best = -std::numeric_limits<double>::infinity();
  int32_t tok = d.ids[0];
  for (size_t i = 0; i < d.ids.size(); ++i) {
    if (!(d.p[i] > 0)) continue;
    const double g = -std::log(-std::log(hashed_uniform(seed, pos, d.ids[i])));
    const double sc = std::log(d.p[i]) + g;
    if (sc > best) best = sc, tok = d.ids[i];
  }
  return tok;
}

namespace {
int32_t draw(const Distribution& d, const std::vector<double>& p, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double r = u(rng);
  double sum = 0;
  for (double v : p) sum += v;
  r *= sum;
  double acc = 0;
  size_t j = 0;
  for (; j + 1 < p.size(); ++j) {
    acc += p[j];
    if (r < acc) break;
  }
  return d.ids.empty() ? int32_t(j) : d.ids[j];
}
}  // namespace

int32_t sample(const Distribution& d, std::mt19937_64& rng) { return draw(d, d.p, rng); }

std::pair<bool, int32_t> verify_draft(const Distribution& d, int32_t draft, std::mt19937_64& rng) {
  size_t j = SIZE_MAX;
  if (d.ids.empty()) {
    if (draft >= 0 && size_t(draft) < d.p.size()) j = size_t(draft);
  } else {
    for (size_t i = 0; i < d.ids.size(); ++i)
      if (d.ids[i] == draft) {
        j = i;
        break;
      }
  }
  const double pd = j == SIZE_MAX ? 0.0 : d.p[j];
  std::uniform_real_distribution<double> u(0.0, 1.0);
  if (pd >= 1.0 || (pd > 0.0 && u(rng) < pd)) return {true, draft};
  std::vector<double> q = d.p;
  if (j != SIZE_MAX) q[j] = 0.0;
  return {false, draw(d, q, rng)};
}

uint64_t NgramIndex::key(const int32_t* t, int n) {
  uint64_t h = 1469598103934665603ull ^ uint64_t(n);
  for (int i = 0; i < n; ++i) h = (h ^ uint64_t(uint32_t(t[i]))) * 1099511628211ull;
  return h;
}

void NgramIndex::append(int32_t t) {
  ctx_.push_back(t);
  const int L = int(ctx_.size());
  // The gram ending just before the new token now has a follower.
  if (L - 1 >= 3) last3_[key(ctx_.data() + L - 4, 3)] = L - 4;
  if (L - 1 >= 4) last4_[key(ctx_.data() + L - 5, 4)] = L - 5;
}

std::vector<int32_t> NgramIndex::lookup(int max_len, int min_match) const {
  const int L = int(ctx_.size());
  if (max_len <= 0 || L <= 3) return {};
  for (int n = std::min(4, L - 1); n >= 3; --n) {
    const auto& map = n == 4 ? last4_ : last3_;
    auto it = map.find(key(ctx_.data() + L - n, n));
    if (it == map.end()) continue;
    const int start = it->second;
    if (!std::equal(ctx_.begin() + start, ctx_.begin() + start + n, ctx_.begin() + L - n)) continue;  // hash clash
    if (min_match > n) {
      // Match length: extend the gram backwards (the earlier occurrence ends before the context's).
      int m = n;
      while (m < std::min(min_match, 64) && start - (m - n) - 1 >= 0 &&
             ctx_[size_t(start - (m - n) - 1)] == ctx_[size_t(L - m - 1)])
        ++m;
      if (m < min_match) continue;
    }
    const int from = start + n;
    const int to = std::min(L, from + max_len);
    if (from < to) return std::vector<int32_t>(ctx_.begin() + from, ctx_.begin() + to);
  }
  return {};
}

}  // namespace espetero
