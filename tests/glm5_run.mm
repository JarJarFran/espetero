// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// GLM-5.3-Flash development driver (src/model/impl/glm5_model.h): one prefill
// of the token ids tools/glm5_ref.py used (its case.json), every layer dumped
// for tests/ref/glm5_compare.py.
//
//   build/glm5_run MODEL_DIR REF_DIR OUT_DIR [layers] [split]
//
// split P > 0: a pass over the first P ids, then one decode pass per remaining id (the state carried across
// passes: KDA and conv states, latent and pooled-key caches, pending pool rows); the passes' dumps are merged
// row by row into OUT_DIR, so the same comparison checks decoding.
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
    const int split = argc > 5 ? std::atoi(argv[5]) : 0;
    const std::string out = argv[3];
    ::mkdir(out.c_str(), 0755);
    const char* cslots = std::getenv("ESPETERO_GLM5_CACHE_SLOTS");  // a small expert cache (eviction tests)
    glm5::DevModel m(argv[1], std::string(espetero_kernels_metal, espetero_kernels_metal_len), layers,
                     std::max(4096, int(ids.size()) + 64), 1, 4096, 0, cslots ? std::atoi(cslots) : 0);
    struct Report {
      glm5::DevModel& m;
      ~Report() { std::fprintf(stderr, "expert cache: %s\n", m.cache_json().c_str()); }
    } report{m};
    const auto t0 = Clock::now();
    if (split <= 0 || split >= int(ids.size())) {
      m.prefill(ids, out);
      if (env_on("ESPETERO_GLM5_MTP")) {  // the MTP layer on the last layer's normed hidden (tools/glm5_ref.py --mtp)
        if (!m.has_mtp()) throw std::runtime_error("no MTP layer (or no head: ESPETERO_GLM5_HEAD_ANY=1 on a slice)");
        m.mtp_dev(ids, out);
        std::fprintf(stderr, "mtp: %zu rows\n", ids.size() - 1);
      }
      std::fprintf(stderr, "prefill of %zu tokens over %d layers: %.1f s (GPU %.1f ms)\n", ids.size(), m.layers(),
                   ms_since(t0) / 1e3, m.gpu_ms());
      return 0;
    }
    std::vector<std::string> dirs;
    auto pass = [&](const std::vector<int32_t>& v) {
      const std::string d = out + "/pass" + std::to_string(dirs.size());
      ::mkdir(d.c_str(), 0755);
      m.prefill(v, d);
      dirs.push_back(d);
    };
    pass(std::vector<int32_t>(ids.begin(), ids.begin() + split));
    const double pre_ms = m.gpu_ms();
    for (size_t i = size_t(split); i < ids.size(); ++i) pass({ids[i]});
    const int nd = int(ids.size()) - split;
    std::fprintf(stderr, "prefill of %d tokens, then %d decode steps over %d layers: GPU %.1f ms + %.2f ms per step\n", split, nd,
                 m.layers(), pre_ms, (m.gpu_ms() - pre_ms) / nd);
    // merge: every dump file of the passes, concatenated in pass order (rows); sel rows keep their width
    for (int l = 0; l < m.layers(); ++l)
      for (const char* kind : {"a", "f", "h", "route", "sel"}) {
        const std::string name = std::string(kind) + std::to_string(l) + (kind[0] == 'r' || kind[0] == 's' ? ".i32" : ".f32");
        std::string all;
        bool any = false;
        for (const std::string& d : dirs)
          if (file_exists(d + "/" + name)) any = true, all += read_file(d + "/" + name);
        if (!any) continue;
        if (FILE* f = std::fopen((out + "/" + name).c_str(), "wb")) {
          std::fwrite(all.data(), 1, all.size(), f);
          std::fclose(f);
        }
      }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
