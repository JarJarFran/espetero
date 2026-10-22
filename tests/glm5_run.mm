// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// GLM-5.3-Flash development driver (src/model/impl/glm5_model.h): one prefill
// of the token ids tools/glm5_ref.py used (its case.json), every layer dumped
// for tests/ref/glm5_compare.py.
//
//   build/glm5_run MODEL_DIR REF_DIR OUT_DIR [layers]
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/model/impl/glm5_model.h"
#include "../src/util/json.h"
#include "kernels_src.h"

using namespace espetero;
using namespace espetero::native;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s MODEL_DIR REF_DIR OUT_DIR [layers]\n", argv[0]);
    return 2;
  }
  try {
    const json::Value cs = json::parse(read_file(std::string(argv[2]) + "/case.json"));
    std::vector<int32_t> ids;
    for (const json::Value& v : cs.get("ids")->as_array()) ids.push_back(int32_t(v.as_int()));
    const int layers = argc > 4 ? std::atoi(argv[4]) : int(cs.get_int("layers", 0));
    ::mkdir(argv[3], 0755);
    glm5::DevModel m(argv[1], std::string(espetero_kernels_metal, espetero_kernels_metal_len), layers,
                     std::max(4096, int(ids.size()) + 64));
    const auto t0 = Clock::now();
    m.prefill(ids, argv[3]);
    std::fprintf(stderr, "prefill of %zu tokens over %d layers: %.1f s (GPU %.1f ms)\n", ids.size(), m.layers(),
                 ms_since(t0) / 1e3, m.gpu_ms());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
