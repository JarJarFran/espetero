// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Quality cost of a kernel option on batched decode (Fase 3): the same
// real-text tokens run through B-row passes under option set A (the
// baseline) and under one or more option sets B, teacher-forced, and every
// readout's full-vocabulary distribution is compared: KLD(A || B) per token
// (mean, median, p99, max), top-1 agreement, max |d logprob|; then a
// lock-step greedy decode of the B sequences under each set, compared token
// by token (first divergence per sequence).
//
//   build/batch_kld <model> <windows.npy> --a "opt=v,opt=v" --b "opt=v,..." [--b "..."]...
//                   [--seqs 16] [--prompt 128] [--steps 192] [--greedy 64] [--max-ctx 2048] [--slots 24] [--ids out.json]
//
// windows.npy: int32 [n_windows][ctx + 1] token ids (the KLD harness's
// reference windows); sequence s is window s: its first `prompt` tokens are
// prefilled, the next `steps` tokens are the teacher-forced inputs, and the
// readout after each input is scored against the following token too (top-1
// accuracy, as a sanity check). Options are set with NativeModel::set_option
// (every value in A is set before the A runs, every value in B before the B
// runs, and A's values are restored before the next B).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../src/model/api.h"

using namespace espetero;

namespace {

std::vector<int32_t> read_npy_i32(const char* path, int& rows, int& cols) {
  std::ifstream f(path, std::ios::binary);
  char magic[8];
  f.read(magic, 8);
  if (!f || std::memcmp(magic, "\x93NUMPY", 6)) throw std::runtime_error("not a .npy file");
  uint32_t hlen = 0;
  if (magic[6] == 1) {
    uint16_t h;
    f.read(reinterpret_cast<char*>(&h), 2);
    hlen = h;
  } else {
    f.read(reinterpret_cast<char*>(&hlen), 4);
  }
  std::string h(hlen, ' ');
  f.read(h.data(), hlen);
  if (h.find("'<i4'") == std::string::npos || h.find("'fortran_order': False") == std::string::npos)
    throw std::runtime_error("windows.npy must be C-order int32");
  const size_t p = h.find("'shape': (");
  if (p == std::string::npos || std::sscanf(h.c_str() + p + 10, "%d, %d", &rows, &cols) != 2)
    throw std::runtime_error("windows.npy must be 2-D");
  std::vector<int32_t> v(size_t(rows) * size_t(cols));
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4));
  if (!f) throw std::runtime_error("short windows.npy");
  return v;
}

struct OptSet {
  std::string text;
  std::vector<std::pair<std::string, int>> kv;
};

OptSet parse_opts(const char* s) {
  OptSet o;
  o.text = s;
  std::string cur;
  auto flush = [&]() {
    if (cur.empty()) return;
    const size_t eq = cur.find('=');
    if (eq == std::string::npos) throw std::runtime_error("option without a value: " + cur);
    o.kv.push_back({cur.substr(0, eq), std::atoi(cur.c_str() + eq + 1)});
    cur.clear();
  };
  for (const char* p = s; *p; ++p) {
    if (*p == ',') flush();
    else if (*p != ' ') cur += *p;
  }
  flush();
  return o;
}

// "seqs=N" in a set is not a model option: the set's teacher-forced passes
// run N sequences at a time (N-row passes) instead of all of them together,
// so a set "seqs=1" measures the batch-shape drift of the engine itself (the
// 1-row kernels against the 16-row ones) with the same reference.
void apply(NativeModel& m, const OptSet& o) {
  for (const auto& [k, v] : o.kv) {
    if (k == "seqs") continue;
    if (!m.set_option(k, v)) throw std::runtime_error("unknown option " + k);
  }
}
int set_seqs(const OptSet& o, int dflt) {
  for (const auto& [k, v] : o.kv)
    if (k == "seqs") return std::max(1, std::min(dflt, v));
  return dflt;
}

// log-softmax of a logits row into out (double accumulation for the sum).
void log_softmax(const float* x, int V, float* out) {
  float mx = -INFINITY;
  for (int i = 0; i < V; ++i) mx = std::max(mx, x[i]);
  double s = 0.0;
  for (int i = 0; i < V; ++i) s += std::exp(double(x[i] - mx));
  const float lse = float(std::log(s)) + mx;
  for (int i = 0; i < V; ++i) out[i] = x[i] - lse;
}

double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double pos = q * double(v.size() - 1);
  const size_t i = size_t(pos);
  const double f = pos - double(i);
  return i + 1 < v.size() ? v[i] * (1 - f) + v[i + 1] * f : v[i];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model> <windows.npy> --a OPTS --b OPTS [--b OPTS]... [--seqs 16] [--prompt 128] [--steps 192] [--greedy 64] [--max-ctx 2048] [--slots 24] [--ids out.json]\n", argv[0]);
    return 2;
  }
  int seqs = 16, prompt = 128, steps = 192, greedy = 64, max_ctx = 2048, slots = 24;
  OptSet A;
  std::vector<OptSet> Bs;
  const char* ids_out = nullptr;
  for (int i = 3; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--a") && i + 1 < argc) A = parse_opts(argv[++i]);
    else if (!std::strcmp(argv[i], "--b") && i + 1 < argc) Bs.push_back(parse_opts(argv[++i]));
    else if (!std::strcmp(argv[i], "--seqs") && i + 1 < argc) seqs = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) steps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--greedy") && i + 1 < argc) greedy = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--max-ctx") && i + 1 < argc) max_ctx = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--slots") && i + 1 < argc) slots = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--ids") && i + 1 < argc) ids_out = argv[++i];
    else {
      std::fprintf(stderr, "unknown argument %s\n", argv[i]);
      return 2;
    }
  }
  if (Bs.empty()) {
    std::fprintf(stderr, "at least one --b\n");
    return 2;
  }
  int nwin = 0, wlen = 0;
  const std::vector<int32_t> win = read_npy_i32(argv[2], nwin, wlen);
  if (nwin < seqs || wlen < prompt + steps + 1)
    throw std::runtime_error("windows.npy too small: " + std::to_string(nwin) + " x " + std::to_string(wlen));
  LoadOptions o;
  o.slots = slots;
  o.max_ctx = max_ctx;
  o.kv_gb = std::max(4.0, (seqs + 1.0) * max_ctx * 72e3 / 1e9);
  auto m = NativeModel::load(argv[1], o);
  const ModelInfo& I = m->info();
  const int V = I.vocab;
  std::printf("# %s  %s  vocab %d  seqs %d  prompt %d  steps %d  greedy %d  (%d scored tokens per set)\n", argv[1],
              I.device.c_str(), V, seqs, prompt, steps, greedy, seqs * steps);
  auto tok = [&](int s, int t) { return win[size_t(s) * size_t(wlen) + size_t(t)]; };

  // Teacher-forced B-row passes under one option set: fills `lp` with the
  // log-softmax rows [step][seq][V] when given, else compares against it.
  struct Stats {
    std::vector<double> kld;
    long top1_agree = 0, top1_true_a = 0, top1_true_b = 0, n = 0;
    double max_dlp = 0.0, max_dlogit = 0.0;
    int dispatches = 0;
  };
  std::vector<float> ref;  // [steps][seqs][V] log-softmax of A
  std::vector<int32_t> ref_argmax;
  std::vector<float> row(static_cast<size_t>(V)), lsm(static_cast<size_t>(V));
  auto teacher = [&](const OptSet& set, bool fill, Stats& st) {
    apply(*m, set);
    const int group = set_seqs(set, seqs);  // sequences per pass
    for (int s0 = 0; s0 < seqs; s0 += group) {
      const int ns = std::min(group, seqs - s0);
      std::vector<int> sl(static_cast<size_t>(ns));
      for (int i = 0; i < ns; ++i) {
        sl[static_cast<size_t>(i)] = m->slot_alloc();
        if (sl[static_cast<size_t>(i)] < 0) throw std::runtime_error("no free slot");
        TreeSpec t;
        t.root = &win[size_t(s0 + i) * size_t(wlen)];
        t.root_len = prompt;
        t.save = sl[static_cast<size_t>(i)];
        m->run({t});
      }
      std::vector<int32_t> next(static_cast<size_t>(ns));
      for (int k = 0; k < steps; ++k) {
        std::vector<TreeSpec> ts;
        for (int i = 0; i < ns; ++i) {
          next[static_cast<size_t>(i)] = tok(s0 + i, prompt + k);
          TreeSpec d;
          d.parent = d.save = sl[static_cast<size_t>(i)];
          d.root = &next[static_cast<size_t>(i)];
          d.root_len = 1;
          d.score_root = true;
          d.want_logits = true;
          ts.push_back(d);
        }
        PassStats ps;
        m->run(ts, &ps);
        if (k == 1 && s0 == 0) st.dispatches = ps.dispatches;
        for (int i = 0; i < ns; ++i) {
          const int s = s0 + i;
          const float* lg = m->logits(i, 0);
          std::memcpy(row.data(), lg, size_t(V) * 4);
          log_softmax(row.data(), V, lsm.data());
          const int am = int(std::max_element(lsm.begin(), lsm.end()) - lsm.begin());
          const int32_t truth = tok(s, prompt + k + 1);
          const size_t base = (size_t(k) * size_t(seqs) + size_t(s)) * size_t(V);
          if (fill) {
            std::memcpy(ref.data() + base, lsm.data(), size_t(V) * 4);
            ref_argmax[size_t(k) * size_t(seqs) + size_t(s)] = am;
            st.top1_true_a += am == truth;
          } else {
            const float* la = ref.data() + base;
            double kl = 0.0, mdlp = 0.0;
            for (int j = 0; j < V; ++j) {
              const double d = double(la[j]) - double(lsm[size_t(j)]);
              kl += std::exp(double(la[j])) * d;
              mdlp = std::max(mdlp, std::fabs(d));
            }
            st.kld.push_back(kl);
            st.max_dlp = std::max(st.max_dlp, mdlp);
            const int ama = ref_argmax[size_t(k) * size_t(seqs) + size_t(s)];
            st.top1_agree += am == ama;
            st.top1_true_b += am == truth;
            st.top1_true_a += ama == truth;
          }
          ++st.n;
        }
      }
      for (int i = 0; i < ns; ++i) m->slot_free(sl[static_cast<size_t>(i)]);
    }
  };
  // Lock-step greedy decode under one option set: [seq][greedy] ids.
  auto greedy_run = [&](const OptSet& set) {
    apply(*m, set);
    std::vector<int> sl(static_cast<size_t>(seqs));
    std::vector<int32_t> out(size_t(seqs) * size_t(greedy));
    for (int s = 0; s < seqs; ++s) {
      sl[size_t(s)] = m->slot_alloc();
      TreeSpec t;
      t.root = &win[size_t(s) * size_t(wlen)];
      t.root_len = prompt;
      t.save = sl[size_t(s)];
      m->run({t});
    }
    std::vector<int32_t> cur(static_cast<size_t>(seqs)), nxt(static_cast<size_t>(seqs));
    for (int s = 0; s < seqs; ++s) cur[size_t(s)] = tok(s, prompt);
    for (int k = 0; k < greedy; ++k) {
      std::vector<TreeSpec> ts;
      for (int s = 0; s < seqs; ++s) {
        TreeSpec d;
        d.parent = d.save = sl[size_t(s)];
        d.root = &cur[size_t(s)];
        d.root_len = 1;
        d.score_root = true;
        d.out_argmax = &nxt[size_t(s)];
        ts.push_back(d);
      }
      m->run(ts);
      for (int s = 0; s < seqs; ++s) {
        out[size_t(s) * size_t(greedy) + size_t(k)] = nxt[size_t(s)];
        cur[size_t(s)] = nxt[size_t(s)];
      }
    }
    for (int s = 0; s < seqs; ++s) m->slot_free(sl[size_t(s)]);
    return out;
  };

  ref.assign(size_t(steps) * size_t(seqs) * size_t(V), 0.0f);
  ref_argmax.assign(size_t(steps) * size_t(seqs), 0);
  Stats sa;
  teacher(A, true, sa);
  std::printf("A (%s): %d dispatches per %d-row step, top-1 accuracy on the true next token %.2f %%\n", A.text.c_str(),
              sa.dispatches, seqs, 100.0 * double(sa.top1_true_a) / double(sa.n));
  std::printf("\n| set | dispatches/step | KLD(A||B) mean | median | p99 | max | tokens with KLD > 1e-3 | top-1 agreement | max |dlogprob| | top-1 accuracy A / B |\n|---|---|---|---|---|---|---|---|---|---|\n");
  for (const OptSet& B : Bs) {
    Stats sb;
    teacher(B, false, sb);
    apply(*m, A);
    double mean = 0.0;
    long over = 0;
    for (double k : sb.kld) mean += k, over += k > 1e-3;
    mean /= double(std::max<size_t>(1, sb.kld.size()));
    std::printf("| %s | %d | %.3g | %.3g | %.3g | %.3g | %ld of %ld | %.2f %% | %.3g | %.2f / %.2f %% |\n", B.text.c_str(), sb.dispatches,
                mean, percentile(sb.kld, 0.5), percentile(sb.kld, 0.99), percentile(sb.kld, 1.0), over, long(sb.kld.size()),
                100.0 * double(sb.top1_agree) / double(sb.n), sb.max_dlp, 100.0 * double(sb.top1_true_a) / double(sb.n),
                100.0 * double(sb.top1_true_b) / double(sb.n));
    std::fflush(stdout);
  }
  if (greedy > 0) {
    std::printf("\n## lock-step greedy decode of the %d sequences (%d tokens each)\n", seqs, greedy);
    const std::vector<int32_t> ga = greedy_run(A);
    std::string js = "{\"prompt\": " + std::to_string(prompt) + ", \"greedy\": " + std::to_string(greedy) + ", \"sets\": {";
    auto dump = [&](const std::string& name, const std::vector<int32_t>& g, bool first) {
      js += std::string(first ? "" : ", ") + "\"" + name + "\": [";
      for (int s = 0; s < seqs; ++s) {
        js += s ? ", [" : "[";
        for (int k = 0; k < greedy; ++k) js += (k ? "," : "") + std::to_string(g[size_t(s) * size_t(greedy) + size_t(k)]);
        js += "]";
      }
      js += "]";
    };
    dump(A.text, ga, true);
    for (const OptSet& B : Bs) {
      const std::vector<int32_t> gb = greedy_run(B);
      apply(*m, A);
      int same_seqs = 0;
      long same_tokens = 0;
      std::string firsts;
      for (int s = 0; s < seqs; ++s) {
        int first = greedy;
        for (int k = 0; k < greedy; ++k)
          if (ga[size_t(s) * size_t(greedy) + size_t(k)] != gb[size_t(s) * size_t(greedy) + size_t(k)]) {
            first = k;
            break;
          }
        same_seqs += first == greedy;
        same_tokens += first;
        firsts += (s ? " " : "") + (first == greedy ? std::string("=") : std::to_string(first));
      }
      std::printf("%s: %d of %d sequences identical; identical prefix %ld of %ld tokens; first divergence per sequence: %s\n",
                  B.text.c_str(), same_seqs, seqs, same_tokens, long(seqs) * greedy, firsts.c_str());
      dump(B.text, gb, false);
    }
    js += "}, \"prompts\": [";
    for (int s = 0; s < seqs; ++s) {
      js += s ? ", [" : "[";
      for (int t = 0; t < prompt; ++t) js += (t ? "," : "") + std::to_string(tok(s, t));
      js += "]";
    }
    js += "]}\n";
    if (ids_out) {
      std::FILE* f = std::fopen(ids_out, "wb");
      if (f) {
        std::fwrite(js.data(), 1, js.size(), f);
        std::fclose(f);
        std::printf("ids written to %s\n", ids_out);
      }
    }
  }
  return 0;
}
