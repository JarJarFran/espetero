// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Exact position-keyed sampling: frequencies of sample_at over many
// positions match the distribution (chi-square), draws are deterministic in
// (seed, position), and the full-vocabulary path agrees with its inverse CDF.
//   build/sampling_test
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../src/engine/sampling.h"

using namespace espetero;

static double chi2(const std::vector<double>& p, const std::vector<long>& n, long N, int* dof) {
  double c = 0;
  *dof = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    const double e = p[i] * double(N);
    if (e < 5) continue;
    c += (double(n[i]) - e) * (double(n[i]) - e) / e;
    ++*dof;
  }
  --*dof;
  return c;
}

int main() {
  int fails = 0;
  std::mt19937 rng(3);
  // (1) head distribution: 24 candidates with ids scattered over the vocabulary
  {
    Distribution d;
    double s = 0;
    for (int i = 0; i < 24; ++i) {
      d.ids.push_back(int32_t(rng() % 248320));
      d.p.push_back(std::exp(-0.25 * i) * (0.5 + (rng() % 1000) / 1000.0));
      s += d.p.back();
    }
    for (double& v : d.p) v /= s;
    const long N = 200000;
    std::vector<long> n(d.p.size(), 0);
    for (long pos = 0; pos < N; ++pos) {
      const int32_t t = sample_at(d, 12345, pos);
      for (size_t i = 0; i < d.ids.size(); ++i)
        if (d.ids[i] == t) { ++n[i]; break; }
    }
    int dof;
    const double c = chi2(d.p, n, N, &dof);
    const double z = (c - dof) / std::sqrt(2.0 * dof);  // normal approximation
    std::printf("head: chi2 %.1f on %d dof (z %.2f)\n", c, dof, z);
    if (std::fabs(z) > 4) ++fails;
    // determinism and seed dependence
    int same = 0, differ = 0;
    for (long pos = 0; pos < 1000; ++pos) {
      same += sample_at(d, 7, pos) == sample_at(d, 7, pos);
      differ += sample_at(d, 7, pos) != sample_at(d, 8, pos);
    }
    std::printf("determinism: %d/1000 repeat, %d/1000 differ across seeds\n", same, differ);
    if (same != 1000 || differ < 300) ++fails;
  }
  // (2) whole vocabulary (ids empty): 1000-token distribution
  {
    Distribution d;
    double s = 0;
    for (int i = 0; i < 1000; ++i) {
      d.p.push_back(std::exp(-0.01 * (rng() % 1000)));
      s += d.p.back();
    }
    for (double& v : d.p) v /= s;
    const long N = 400000;
    std::vector<long> n(d.p.size(), 0);
    for (long pos = 0; pos < N; ++pos) ++n[size_t(sample_at(d, 99, pos))];
    int dof;
    const double c = chi2(d.p, n, N, &dof);
    const double z = (c - dof) / std::sqrt(2.0 * dof);
    std::printf("vocabulary: chi2 %.1f on %d dof (z %.2f)\n", c, dof, z);
    if (std::fabs(z) > 4) ++fails;
  }
  // (3) the hashed uniform is in (0, 1) and roughly uniform
  {
    double mean = 0, lo = 1, hi = 0;
    const int N = 1000000;
    for (int i = 0; i < N; ++i) {
      const double u = hashed_uniform(42, i, i % 97);
      mean += u, lo = std::min(lo, u), hi = std::max(hi, u);
    }
    mean /= N;
    std::printf("uniform: mean %.5f min %.2e max %.9f\n", mean, lo, hi);
    if (std::fabs(mean - 0.5) > 0.002 || lo <= 0 || hi >= 1) ++fails;
  }
  std::printf("%s (%d failures)\n", fails ? "FAIL" : "OK", fails);
  return fails ? 1 : 0;
}
