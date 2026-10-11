// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Checks espetero::Tokenizer against Hugging Face `tokenizers` reference data
// produced by tests/ref/tok_ref.py, and reports load time and encode speed.
//
//   python3 tests/ref/tok_ref.py OUT_DIR
//   clang++ -std=c++20 -O3 -mcpu=apple-m1 -Wall -Wextra tests/tok_test.cpp src/tok/*.cpp \
//       src/util/json.cpp -framework CoreFoundation -o tok_test
//   ./tok_test OUT_DIR/tok_ref.bin
//
// Every case must match exactly: encode (parse_special true and false), decode
// with and without skip_special (compared after the reference's lossy UTF-8
// conversion), decoding arbitrary id sequences, and tokenizer metadata.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../src/tok/tokenizer.h"

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

class Reader {
 public:
  explicit Reader(std::string data) : d_(std::move(data)) {}
  uint8_t u8() { return uint8_t(take(1)[0]); }
  uint32_t u32() {
    uint32_t v;
    std::memcpy(&v, take(4), 4);
    return v;
  }
  int32_t i32() { return int32_t(u32()); }
  std::string str() {
    const uint32_t n = u32();
    return std::string(take(n), n);
  }
  std::vector<int32_t> ids() {
    const uint32_t n = u32();
    std::vector<int32_t> v(n);
    if (n) std::memcpy(v.data(), take(size_t(n) * 4), size_t(n) * 4);
    return v;
  }
  const char* take(size_t n) {
    if (n > d_.size() - pos_) throw std::runtime_error("reference file truncated");
    const char* p = d_.data() + pos_;
    pos_ += n;
    return p;
  }

 private:
  std::string d_;
  size_t pos_ = 0;
};

struct TokRef {
  std::string name, kind, path, chat_template;
  int32_t eos, bos, pad, vocab_size;
  std::vector<int32_t> special_ids;
  std::vector<std::pair<std::string, int32_t>> added;
  std::vector<int32_t> bench_ids;
};

struct Expect {
  uint8_t flags;
  std::vector<int32_t> ids, ids_ns;
  std::string dec, dec_skip;
};

struct Case {
  std::string text;
  std::vector<Expect> per;
};

struct DecodeCase {
  uint32_t tok;
  bool skip;
  std::vector<int32_t> ids;
  std::string expected;
};

// UTF-8 -> valid UTF-8, each maximal invalid subpart replaced by U+FFFD: the
// behavior of Rust's String::from_utf8_lossy and Python's errors="replace".
std::string lossy(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  const auto* p = reinterpret_cast<const unsigned char*>(s.data());
  const size_t n = s.size();
  size_t i = 0;
  while (i < n) {
    const unsigned c = p[i];
    if (c < 0x80) {
      out.push_back(char(c));
      ++i;
      continue;
    }
    int len = 0;
    unsigned lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) len = 2;
    else if (c == 0xE0) len = 3, lo = 0xA0;
    else if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF) len = 3;
    else if (c == 0xED) len = 3, hi = 0x9F;
    else if (c == 0xF0) len = 4, lo = 0x90;
    else if (c >= 0xF1 && c <= 0xF3) len = 4;
    else if (c == 0xF4) len = 4, hi = 0x8F;
    size_t j = 1;
    if (len) {
      if (i + 1 < n && p[i + 1] >= lo && p[i + 1] <= hi) {
        j = 2;
        while (j < size_t(len) && i + j < n && (p[i + j] & 0xC0) == 0x80) ++j;
      }
    }
    if (len && j == size_t(len)) {
      out.append(s.substr(i, j));
    } else {
      out += "\xEF\xBF\xBD";
    }
    i += j;
  }
  return out;
}

std::string escape(std::string_view s, size_t max = 160) {
  std::string out;
  for (unsigned char c : s.substr(0, max)) {
    if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else if (c < 0x20 || c == 0x7F) {
      char b[8];
      std::snprintf(b, sizeof b, "\\x%02x", c);
      out += b;
    } else out.push_back(char(c));
  }
  if (s.size() > max) out += "...";
  return out;
}

std::string ids_str(const std::vector<int32_t>& v, size_t from = 0, size_t max = 24) {
  std::string s = "[";
  if (from) s += "... ";
  for (size_t i = from; i < v.size() && i < from + max; ++i) s += std::to_string(v[i]) + (i + 1 < v.size() ? " " : "");
  if (v.size() > from + max) s += "...";
  return s + "] (" + std::to_string(v.size()) + ")";
}

struct Counter {
  const char* what;
  size_t ok = 0, bad = 0;
  void check(bool good, const std::function<void()>& report) {
    if (good) {
      ++ok;
    } else if (++bad <= 3) {
      report();
    }
  }
};

void report_ids(const char* tok, const char* what, const std::string& text, const std::vector<int32_t>& want,
                const std::vector<int32_t>& got) {
  size_t k = 0;
  while (k < want.size() && k < got.size() && want[k] == got[k]) ++k;
  const size_t from = k > 4 ? k - 4 : 0;
  std::fprintf(stderr, "  [%s] %s MISMATCH for \"%s\"\n    want %s\n    got  %s\n", tok, what, escape(text).c_str(),
               ids_str(want, from).c_str(), ids_str(got, from).c_str());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s tok_ref.bin\n", argv[0]);
    return 2;
  }
  std::ifstream f(argv[1], std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  Reader rd(ss.str());
  if (std::string(rd.take(8), 8) != "TOKREF03") {
    std::fprintf(stderr, "bad reference file (regenerate with tests/ref/tok_ref.py)\n");
    return 2;
  }
  std::vector<TokRef> toks(rd.u32());
  for (TokRef& t : toks) {
    t.name = rd.str();
    t.kind = rd.str();
    t.path = rd.str();
    t.eos = rd.i32();
    t.bos = rd.i32();
    t.pad = rd.i32();
    t.vocab_size = rd.i32();
    t.chat_template = rd.str();
    t.special_ids = rd.ids();
    t.added.resize(rd.u32());
    for (auto& a : t.added) {
      a.first = rd.str();
      a.second = rd.i32();
    }
  }
  std::vector<Case> cases(rd.u32());
  for (Case& c : cases) {
    c.text = rd.str();
    c.per.resize(toks.size());
    for (Expect& e : c.per) {
      e.flags = rd.u8();
      e.ids = rd.ids();
      e.ids_ns = (e.flags & 1) ? e.ids : rd.ids();
      e.dec = (e.flags & 2) ? c.text : rd.str();
      e.dec_skip = (e.flags & 4) ? e.dec : rd.str();
    }
  }
  std::vector<DecodeCase> dcases(rd.u32());
  for (DecodeCase& d : dcases) {
    d.tok = rd.u32();
    d.skip = rd.u8() != 0;
    d.ids = rd.ids();
    d.expected = rd.str();
  }
  const std::string bench = rd.str();
  for (TokRef& t : toks) t.bench_ids = rd.ids();
  size_t corpus_bytes = 0;
  for (const Case& c : cases) corpus_bytes += c.text.size();
  std::printf("reference: %zu tokenizers, %zu cases (%.1f MB), %zu decode cases, bench %.1f KB\n", toks.size(),
              cases.size(), corpus_bytes / 1e6, dcases.size(), bench.size() / 1024.0);

  size_t total_bad = 0;
  std::vector<std::unique_ptr<espetero::Tokenizer>> loaded(toks.size());
  for (size_t ti = 0; ti < toks.size(); ++ti) {
    const TokRef& R = toks[ti];
    const char* name = R.name.c_str();
    double load_ms = 1e9;
    try {
      for (int rep = 0; rep < 3; ++rep) {  // best of 3 (page cache warm)
        const auto t0 = Clock::now();
        loaded[ti] = R.kind == "gguf" ? espetero::Tokenizer::from_gguf(R.path) : espetero::Tokenizer::from_hf_dir(R.path);
        load_ms = std::min(load_ms, ms_since(t0));
      }
    } catch (const std::exception& ex) {
      std::printf("%-13s LOAD FAILED: %s\n", name, ex.what());
      ++total_bad;
      continue;
    }
    const espetero::Tokenizer& T = *loaded[ti];
    // BOS before raw text as llama.cpp does it: Gemma's GGUF says so
    // (tokenizer.ggml.add_bos_token) and its tokenizer.json is SentencePiece;
    // Qwen's files never ask for it.
    {
      const bool want = R.name.find("gemma") != std::string::npos;
      const std::vector<int32_t> ids = T.with_bos(T.encode("x"));
      const bool ok = T.add_bos() == want && (ids.front() == T.bos_id()) == want;
      std::printf("%-13s add_bos %s%s\n", name, T.add_bos() ? "yes" : "no", ok ? "" : " (WRONG)");
      total_bad += !ok;
    }

    // ---- first encode on a fresh tokenizer (empty word cache)
    const auto tc = Clock::now();
    std::vector<int32_t> got = T.encode(bench);
    const double cold_ms = ms_since(tc);
    Counter bn{"bench"};
    bn.check(got == R.bench_ids, [&] { report_ids(name, "bench", bench.substr(0, 60), R.bench_ids, got); });

    // ---- metadata
    Counter meta{"meta"};
    auto expect_eq = [&](const char* what, long want, long got) {
      meta.check(want == got, [&] { std::fprintf(stderr, "  [%s] %s: want %ld got %ld\n", name, what, want, got); });
    };
    expect_eq("vocab_size", R.vocab_size, T.vocab_size());
    expect_eq("eos", R.eos, T.eos_id());
    expect_eq("bos", R.bos, T.bos_id());
    expect_eq("pad", R.pad, T.pad_id());
    meta.check(T.chat_template() == R.chat_template, [&] {
      std::fprintf(stderr, "  [%s] chat_template differs (%zu vs %zu bytes)\n", name, R.chat_template.size(),
                   T.chat_template().size());
    });
    meta.check(T.special_ids() == R.special_ids, [&] {
      std::fprintf(stderr, "  [%s] special_ids want %s got %s\n", name, ids_str(R.special_ids).c_str(),
                   ids_str(T.special_ids()).c_str());
    });
    for (const auto& [content, id] : R.added) {
      expect_eq(("token_id(" + content + ")").c_str(), id, T.token_id(content));
      const bool sp = std::binary_search(R.special_ids.begin(), R.special_ids.end(), id);
      expect_eq(("is_special(" + content + ")").c_str(), sp, T.is_special(id));
      expect_eq(("is_added(" + content + ")").c_str(), 1, T.is_added(id));
    }
    size_t text_roundtrip_bad = 0;  // vocabulary tokens; added tokens are checked above
    for (int32_t id = 0; id < T.vocab_size(); ++id) {
      const std::string_view tx = T.token_text(id);
      if (!tx.empty() && !T.is_added(id) && T.token_id(tx) != id) ++text_roundtrip_bad;
    }
    expect_eq("token_id(token_text(id)) mismatches", 0, long(text_roundtrip_bad));

    // ---- corpus
    Counter enc{"encode"}, enc_ns{"nospecial"}, dec{"decode"}, dec_skip{"skip"}, pieces{"pieces"};
    for (const Case& c : cases) {
      const Expect& e = c.per[ti];
      got = T.encode(c.text);
      enc.check(got == e.ids, [&] { report_ids(name, "encode", c.text, e.ids, got); });
      const std::vector<int32_t> got_ns = T.encode(c.text, false);
      enc_ns.check(got_ns == e.ids_ns, [&] { report_ids(name, "encode(parse_special=false)", c.text, e.ids_ns, got_ns); });
      const std::string d = T.decode(e.ids);
      dec.check(lossy(d) == e.dec, [&] {
        std::fprintf(stderr, "  [%s] decode MISMATCH\n    want \"%s\"\n    got  \"%s\"\n", name, escape(e.dec).c_str(),
                     escape(lossy(d)).c_str());
      });
      const std::string ds = T.decode(e.ids, true);
      dec_skip.check(lossy(ds) == e.dec_skip, [&] {
        std::fprintf(stderr, "  [%s] decode(skip_special) MISMATCH\n    want \"%s\"\n    got  \"%s\"\n", name,
                     escape(e.dec_skip).c_str(), escape(lossy(ds)).c_str());
      });
      std::string joined;
      for (int32_t id : e.ids) joined += T.piece(id);
      pieces.check(joined == d, [&] { std::fprintf(stderr, "  [%s] pieces do not concatenate to decode()\n", name); });
    }
    Counter dcs{"decode-cases"};
    // SentencePiece-style tokenizers: the reference's ByteFallback
    // decoder turns a run of byte tokens that is not valid UTF-8 into one
    // U+FFFD per token; decode() gives the bytes (a documented difference),
    // so the expectation is built the reference's way from the pieces.
    const bool spm = T.pre_tokenizer_name() == "spm";
    auto hf_decode = [&](const std::vector<int32_t>& ids, bool skip) {
      std::string out, run;
      int n_run = 0;
      auto flush = [&]() {
        if (!n_run) return;
        if (lossy(run) == run) out += run;
        else
          for (int k = 0; k < n_run; ++k) out += "\xEF\xBF\xBD";
        run.clear();
        n_run = 0;
      };
      for (int32_t id : ids) {
        if (id < 0 || id >= T.vocab_size() || (skip && T.is_special(id))) continue;
        const std::string_view tx = T.token_text(id);
        if (tx.size() == 6 && tx.substr(0, 3) == "<0x" && tx[5] == '>' && !T.is_added(id)) {
          run += T.piece(id);
          ++n_run;
        } else {
          flush();
          out += T.piece(id);
        }
      }
      flush();
      return lossy(out);
    };
    for (const DecodeCase& d : dcases) {
      if (d.tok != ti) continue;
      const std::string out = spm ? hf_decode(d.ids, d.skip) : lossy(T.decode(d.ids, d.skip));
      dcs.check(out == d.expected, [&] {
        std::fprintf(stderr, "  [%s] decode case MISMATCH ids %s skip=%d\n    want \"%s\"\n    got  \"%s\"\n", name,
                     ids_str(d.ids).c_str(), int(d.skip), escape(d.expected).c_str(), escape(out).c_str());
      });
    }

    // ---- input that is not valid UTF-8 (the reference cannot take it): every
    // invalid byte becomes its byte token, so decode gives the input back.
    Counter inv{"invalid-utf8"};
    {
      static const char* const kBad[] = {"\xff", "\xfe\xff", "\xc0\xaf", "\xe0\x80\xaf", "\xed\xa0\x80", "\xf4\x90\x80\x80",
                                         "\xf8\x88\x80\x80\x80", "\xe4\xbd", "\xe4", "\x80", "\xbf\xbf", "\xc3", "\xf0\x9f\x98"};
      static const char* const kGood[] = {"h\xc3\xa9llo", " w\xc3\xb6rld", "\xe4\xb8\xad\xe6\x96\x87", " \xf0\x9f\x98\x80",
                                          "<|im_start|>", "<|im_end|>\n", "  ", "\n", "'s", "123", "x", "\t", "e\xcc\x81"};
      uint64_t rng = 0x9E3779B97F4A7C15ull + ti;
      auto next = [&rng] {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
      };
      for (int k = 0; k < 3000; ++k) {
        std::string t;
        const int parts = 1 + int(next() % 8);
        for (int j = 0; j < parts; ++j) {
          const uint64_t r = next();
          if (r % 3 == 0) t += kBad[(r >> 8) % (sizeof kBad / sizeof *kBad)];
          else if (r % 3 == 1) t += kGood[(r >> 8) % (sizeof kGood / sizeof *kGood)];
          // Lone continuation byte: it can only complete a truncated sequence from kBad
          // into a Latin-1, CJK or emoji character, none of which NFC changes.
          else t.push_back(char(0x80 | ((r >> 8) & 0x3F)));
        }
        // Text that NFC would change ("e" + U+0301) is not expected to round-trip.
        const bool nfc_stable = t.find("e\xcc\x81") == std::string::npos;
        for (bool ps : {true, false}) {
          const std::string back = T.decode(T.encode(t, ps));
          inv.check(!nfc_stable || back == t, [&] {
            std::fprintf(stderr, "  [%s] invalid UTF-8 round trip failed for \"%s\"\n", name, escape(t).c_str());
          });
        }
      }
    }

    // ---- speed: warm cache
    std::vector<double> warm;
    for (int rep = 0; rep < 30; ++rep) {
      const auto t0 = Clock::now();
      got.clear();
      T.encode_append(bench, got);
      warm.push_back(ms_since(t0));
    }
    std::sort(warm.begin(), warm.end());
    const auto t0 = Clock::now();
    size_t n_tok = 0;
    for (const Case& c : cases) n_tok += T.encode(c.text).size();
    const double corpus_ms = ms_since(t0);

    size_t bad = 0;
    for (const Counter* c : {&meta, &enc, &enc_ns, &dec, &dec_skip, &pieces, &dcs, &bn, &inv}) bad += c->bad;
    total_bad += bad;
    std::printf("%-13s load %6.1f ms | encode %zu/%zu  nospecial %zu/%zu  decode %zu/%zu  skip %zu/%zu  "
                "decode-ids %zu/%zu  meta %zu/%zu  invalid-utf8 %zu/%zu | %s\n",
                name, load_ms, enc.ok, enc.ok + enc.bad, enc_ns.ok, enc_ns.ok + enc_ns.bad, dec.ok, dec.ok + dec.bad,
                dec_skip.ok, dec_skip.ok + dec_skip.bad, dcs.ok, dcs.ok + dcs.bad, meta.ok, meta.ok + meta.bad, inv.ok,
                inv.ok + inv.bad, bad ? "FAIL" : "ok");
    std::printf("%-13s bench %.1f KB -> %zu tokens: cold %.2f ms, warm min %.2f / median %.2f ms (%.0f MB/s); "
                "whole corpus %.1f ms (%.1f MB/s, %zu tokens)\n",
                "", bench.size() / 1024.0, R.bench_ids.size(), cold_ms, warm.front(), warm[warm.size() / 2],
                bench.size() / 1e6 / (warm.front() / 1e3), corpus_ms, corpus_bytes / 1e6 / (corpus_ms / 1e3), n_tok);
  }

  // GGUF and tokenizer.json of the same model must agree token for token.
  for (size_t a = 0; a < toks.size(); ++a)
    for (size_t b = 0; b < toks.size(); ++b)
      if (loaded[a] && loaded[b] && toks[b].kind == "gguf" && toks[a].kind == "hf" && toks[b].name == toks[a].name + "-gguf") {
        size_t diff = 0;
        const int32_t n = std::max(loaded[a]->vocab_size(), loaded[b]->vocab_size());
        for (int32_t id = 0; id < n; ++id)
          diff += loaded[a]->token_text(id) != loaded[b]->token_text(id) || loaded[a]->piece(id) != loaded[b]->piece(id);
        std::printf("%s vs %s: %d ids, %zu differ in text/piece\n", toks[a].name.c_str(), toks[b].name.c_str(), n, diff);
        total_bad += diff;
      }

  // Concurrent use: several threads share one tokenizer (and its word cache).
  if (loaded[0]) {
    const espetero::Tokenizer& T = *loaded[0];
    std::atomic<size_t> bad{0};
    std::vector<std::thread> th;
    const auto t0 = Clock::now();
    for (int k = 0; k < 8; ++k)
      th.emplace_back([&, k] {
        for (size_t i = size_t(k); i < cases.size(); i += 3)
          if (T.encode(cases[i].text) != cases[i].per[0].ids) ++bad;
      });
    for (auto& t : th) t.join();
    std::printf("threads: 8 x %zu encodes on %s in %.1f ms, %zu mismatches\n", cases.size() / 3, toks[0].name.c_str(),
                ms_since(t0), bad.load());
    total_bad += bad;
  }

  std::printf("%s\n", total_bad ? "FAILED" : "ALL OK");
  return total_bad ? 1 : 0;
}
