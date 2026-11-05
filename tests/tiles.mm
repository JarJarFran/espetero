// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Writes a checkpoint's tile image (src/model/impl/tiles.h): one load with
// ESPETERO_TILES=write, with the default load options (an image matches
// later loads with the same options, this build and these files).
//
//   build/espetero_tiles <model> [--mtp none]

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../src/model/api.h"

using namespace espetero;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model> [--mtp none]\n", argv[0]);
    return 2;
  }
  LoadOptions o;
  o.kv_gb = 1;
  o.slots = 1;
  for (int i = 2; i < argc; ++i)
    if (!std::strcmp(argv[i], "--mtp") && i + 1 < argc) o.mtp = argv[++i];
  ::setenv("ESPETERO_TILES", "write", 1);
  auto model = NativeModel::load(argv[1], o);
  std::fprintf(stderr, "%s: loaded in %.0f ms; tile image written beside it\n", argv[1], model->info().load_ms);
  return 0;
}
