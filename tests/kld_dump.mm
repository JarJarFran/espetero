// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Final-normed hidden states of the native engine over the windows of a KLD
// harness reference, for the harness to score Espetero itself:
//
//   build/kld_dump <model> <REF_DIR/windows.npy> <out.f16> [--skip 64] [--max-windows N] [--kv-gb G]
//
// windows.npy: int32 [n_windows][ctx + 1] (the harness's reference windows).
// Each window is one prefill pass over its first ctx tokens; rows
// [skip, ctx) of TreeSpec::out_hidden are appended to out.f16 as raw fp16
// [n_windows][ctx - skip][hidden] (the engine keeps them in fp16). The harness
// applies the checkpoint's LM head (kld_harness.py eval --fmt espetero:out.f16).
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../src/model/api.h"

using namespace espetero;

namespace {

// Minimal .npy reader for a C-order int32 2-D array.
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <model> <windows.npy> <out.f16> [--skip N] [--max-windows N] [--kv-gb G]\n", argv[0]);
    return 2;
  }
  int skip = 64, max_windows = 0;
  LoadOptions o;
  o.kv_gb = 2;
  o.slots = 2;
  for (int i = 4; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--skip") && i + 1 < argc) skip = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--max-windows") && i + 1 < argc) max_windows = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--kv-gb") && i + 1 < argc) o.kv_gb = std::atof(argv[++i]);
  }
  int nw = 0, cols = 0;
  const std::vector<int32_t> win = read_npy_i32(argv[2], nw, cols);
  const int ctx = cols - 1;
  if (max_windows > 0 && max_windows < nw) nw = max_windows;
  o.max_ctx = ctx + 64;
  o.max_rows = std::max(o.max_rows, ctx);
  auto model = NativeModel::load(argv[1], o);
  const ModelInfo& I = model->info();
  if (!I.hidden_states) {
    std::fprintf(stderr, "%s: this model has no hidden-state readouts\n", argv[1]);
    return 1;
  }
  const int D = I.hidden;
  std::fprintf(stderr, "loaded %s: %d layers, hidden %d, %d windows of %d, load %.0f ms\n", argv[1], I.layers, D, nw,
               ctx, I.load_ms);
  std::FILE* out = std::fopen(argv[3], "wb");
  if (!out) {
    std::perror(argv[3]);
    return 1;
  }
  std::vector<float> hid(size_t(ctx) * size_t(D));
  std::vector<__fp16> h16(size_t(ctx - skip) * size_t(D));
  double gpu_ms = 0;
  for (int w = 0; w < nw; ++w) {
    TreeSpec t;
    t.root = win.data() + size_t(w) * size_t(cols);
    t.root_len = ctx;
    t.score_root = true;
    t.out_hidden = hid.data();
    PassStats st;
    model->run({t}, &st);
    gpu_ms += st.gpu_ms;
    for (size_t i = 0; i < h16.size(); ++i) h16[i] = __fp16(hid[size_t(skip) * size_t(D) + i]);
    std::fwrite(h16.data(), sizeof(__fp16), h16.size(), out);
    std::fprintf(stderr, "\r[kld_dump] window %d/%d  %.0f tok/s", w + 1, nw, (w + 1) * ctx / (gpu_ms / 1000.0));
  }
  std::fprintf(stderr, "\n");
  std::fclose(out);
  return 0;
}
