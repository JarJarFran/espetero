// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Which models fit this Mac (models.h): a curated table of public
// repositories of the families Espetero runs, with their sizes on the Hub
// (2026-10-09), and the best file of each model that fits.
//
// Models are ranked by quality, the newest generation and the largest first:
// Qwen3.8-Flash-Next (125B-A6B; only where its experts mostly fit in memory,
// as they stream from the SSD), Qwen3.8-27B, Qwen3.6-35B-A3B (a MoE: lower
// quality than the 27B, several times its speed), then the small Qwen3.5
// models. A model's files go 8-bit, 6-bit, 4-bit (4-bit
// only when memory forces it); at one width a GGUF file with an MTP head
// (faster decoding, the same replies) comes before an MLX file without one,
// except for the MoE, whose MLX files are the measured ones. Older models of
// the same size (Qwen3.5-27B, Qwen3.6-27B, Qwen3.5-35B-A3B) are left out, and
// so are files the engine does not read (mxfp4/nvfp4, IQ1-IQ3 GGUF) and models
// it refuses (Qwen3.5-122B-A10B and 397B-A17B: GQA group 16).
#include <algorithm>
#include <cctype>
#include <cstdio>

#include "models.h"

namespace espetero::models {

namespace {

struct Base {
  const char* name;
  const char* kind;
  Geometry geo;
  bool tied;  // tie_word_embeddings: the engine keeps a second, GEMM-layout copy as the LM head
};

Geometry geometry(int D, int I, int L, int n_full, int H, int Hkv, int Hv, int E = 0, int k = 0, int Ie = 0, int hc = 1,
                  int hc_rank = 0) {
  Geometry g;
  g.known = true;
  g.D = D, g.I = I, g.L = L, g.n_full = n_full, g.n_linear = L - n_full, g.H = H, g.Hkv = Hkv, g.Hk = 16, g.Hv = Hv;
  g.E = E, g.k = k, g.Ie = Ie, g.I_shared = E ? Ie : 0, g.V = 248320, g.hc = hc, g.hc_rank = hc_rank, g.q4 = hc > 1;
  return g;
}

// Their config.json (Qwen/...): hidden size, MLP width, layers, full-attention layers, heads, KV heads, delta heads.
const Base kFlashNext = {"Qwen3.8-Flash-Next", "MoE 125B-A6B, experts read from the SSD",
                         geometry(2560, 0, 48, 12, 24, 2, 48, 512, 10, 640, 4, 320), false};
const Base k27B = {"Qwen3.8-27B", "dense", geometry(5120, 17408, 64, 16, 24, 4, 48), false};
const Base k35B = {"Qwen3.6-35B-A3B", "MoE: several times faster than the 27B, a little below it",
                   geometry(2048, 0, 40, 10, 16, 2, 32, 256, 8, 512), false};
const Base k9B = {"Qwen3.5-9B", "dense", geometry(4096, 12288, 32, 8, 16, 4, 32), false};
const Base k4B = {"Qwen3.5-4B", "dense", geometry(2560, 9216, 32, 8, 16, 4, 32), true};
const Base k2B = {"Qwen3.5-2B", "dense", geometry(2048, 6144, 24, 6, 8, 2, 16), true};
const Base k08B = {"Qwen3.5-0.8B", "dense", geometry(1024, 3584, 24, 6, 8, 2, 16), true};

int64_t bytes(double gb) { return int64_t(gb * 1e9); }
int64_t coarse_head(const Geometry& g) { return int64_t(g.V) * g.D * 5 / 16; }  // the 2-bit draft copy

CatalogEntry entry(const Base& b, const std::string& id, int bits, const char* format, const std::string& quant) {
  CatalogEntry e;
  e.model = b.name;
  e.kind = b.kind;
  e.id = id;
  e.bits = bits;
  e.footprint.format = format;
  e.footprint.quant = quant;
  e.footprint.geo = b.geo;
  e.footprint.has_vision = true;
  return e;
}

// An MLX repository: `gb` of safetensors (its weights as the engine holds
// them; 4- to 8-bit codes keep their size), plus a tied head's copy.
CatalogEntry mlx(const Base& b, const std::string& id, int bits, double gb, bool mtp = false) {
  CatalogEntry e = entry(b, id, bits, "MLX", std::to_string(bits) + "-bit");
  Footprint& f = e.footprint;
  f.disk = bytes(gb);
  f.base = f.disk + (b.tied ? int64_t(b.geo.V) * b.geo.D * (bits * 2 + 1) / 16 : 0);
  f.weights = f.base;
  if (mtp) {
    f.has_mtp = true;
    f.mtp = coarse_head(b.geo);
    f.weights += f.mtp;
  }
  f.what = "MLX " + f.quant;
  return e;
}

// A GGUF file of a repository: the model file (`file_gb`, its MTP head
// inside or not), the projector, and a separate MTP head file (its own
// tensors in GPU memory: `mtp_gpu_gb`). GPU bytes from the quantization's
// widening (gguf_quant_factor) and a tied head's copy.
CatalogEntry gguf(const Base& b, const std::string& repo, const std::string& quant, int bits, double file_gb,
                  double mmproj_gb, bool mtp_inside, double mtp_file_gb = 0, double mtp_gpu_gb = 0) {
  CatalogEntry e = entry(b, repo + ":" + quant, bits, "GGUF", quant);
  Footprint& f = e.footprint;
  const double head_bits = bits == 8 ? 8.5 : bits == 6 ? 6.5625 : 5;
  f.base = int64_t(double(bytes(file_gb)) * gguf_quant_factor(quant)) +
           (b.tied ? int64_t(double(b.geo.V) * b.geo.D * head_bits / 8) : 0);
  f.vision = bytes(mmproj_gb);
  f.has_mtp = mtp_inside || mtp_file_gb > 0;
  if (f.has_mtp) f.mtp = bytes(mtp_gpu_gb) + coarse_head(b.geo);
  f.weights = f.base + f.vision + f.mtp;
  f.disk = bytes(file_gb + mmproj_gb + mtp_file_gb);
  f.what = "GGUF " + quant;
  return e;
}

std::vector<CatalogEntry> build() {
  std::vector<CatalogEntry> c;
  // Flash-Next oQ4e with its MTP layer: 4.9 GB stay in memory (trunk 4.0, vision 0.9), the experts
  // (69.4 GB) and the n-gram table (32 GB) are read from the SSD; recommended where at least half
  // of the experts fit in the expert cache.
  {
    CatalogEntry e = mlx(kFlashNext, "Jundot/Qwen3.8-Flash-Next-oQ4e-mtp", 4, 4.90);
    e.footprint.quant = "mixed 4-bit (oQ4e)";
    e.footprint.disk = bytes(106.29);
    e.footprint.streamed = bytes(99.95);
    e.experts = bytes(69.36);
    e.cache_share = 0.5;
    c.push_back(e);
  }
  // Qwen3.8-27B: unsloth's Q8_0 holds its MTP head; UD-Q6_K and UD-Q4_K_XL use MTP/mtp-*-Q4_0.gguf
  // (1.37 GB, 0.35 GB of its own tensors).
  const std::string u27 = "unsloth/Qwen3.8-27B-GGUF";
  c.push_back(gguf(k27B, u27, "Q8_0", 8, 29.05, 0.93, true));
  c.push_back(mlx(k27B, "mlx-community/Qwen3.8-27B-8bit", 8, 29.50));
  c.push_back(gguf(k27B, u27, "UD-Q6_K", 6, 21.98, 0.93, false, 1.37, 0.35));
  c.push_back(mlx(k27B, "lmstudio-community/Qwen3.8-27B-MLX-6bit", 6, 22.78));
  c.push_back(gguf(k27B, u27, "UD-Q4_K_XL", 4, 17.56, 0.93, false, 1.37, 0.35));
  c.push_back(mlx(k27B, "mlx-community/Qwen3.8-27B-4bit", 4, 16.05));
  // Qwen3.6-35B-A3B
  const std::string u35 = "unsloth/Qwen3.6-35B-A3B-GGUF";
  c.push_back(mlx(k35B, "mlx-community/Qwen3.6-35B-A3B-8bit", 8, 37.72));
  c.push_back(gguf(k35B, u35, "Q8_0", 8, 36.90, 0.90, false));
  c.push_back(mlx(k35B, "mlx-community/Qwen3.6-35B-A3B-6bit", 6, 29.06));
  c.push_back(gguf(k35B, u35, "UD-Q6_K", 6, 29.31, 0.90, false));
  c.push_back(mlx(k35B, "mlx-community/Qwen3.6-35B-A3B-4bit", 4, 20.40));
  // Qwen3.5 small models: unsloth's MTP-GGUF files hold their MTP heads.
  struct Small {
    const Base* b;
    const char* size;
    double q8, q6, q4, mmproj, m8, m6, m4;
  };
  for (const Small& s : {Small{&k9B, "9B", 9.79, 7.68, 5.87, 0.92, 10.43, 8.19, 5.95},
                         Small{&k4B, "4B", 4.61, 3.64, 2.83, 0.67, 5.14, 4.09, 3.03},
                         Small{&k2B, "2B", 2.08, 1.63, 1.33, 0.67, 2.66, 2.19, 1.72},
                         Small{&k08B, "0.8B", 0.83, 0.66, 0.55, 0.20, 1.00, 0.81, 0.63}}) {
    const std::string u = std::string("unsloth/Qwen3.5-") + s.size + "-MTP-GGUF";
    const std::string m = std::string("mlx-community/Qwen3.5-") + s.size;
    c.push_back(gguf(*s.b, u, "Q8_0", 8, s.q8, s.mmproj, true));
    c.push_back(mlx(*s.b, m + "-8bit", 8, s.m8));
    c.push_back(gguf(*s.b, u, "Q6_K", 6, s.q6, s.mmproj, true));
    c.push_back(mlx(*s.b, m + "-6bit", 6, s.m6));
    c.push_back(gguf(*s.b, u, "Q4_K_M", 4, s.q4, s.mmproj, true));
    c.push_back(mlx(*s.b, m + "-4bit", 4, s.m4));
  }
  return c;
}

std::string mac_text(const Mac& m) {
  const std::string ram = std::to_string(int64_t(double(m.ram) / double(1ull << 30) + 0.5)) + " GB of RAM";
  return m.chip.empty() ? "a Mac with " + ram + " (" + gb(m.usable()) + " for the GPU)"
                        : "this Mac (" + m.chip + ", " + ram + ", " + gb(m.usable()) + " for the GPU)";
}

}  // namespace

const std::vector<CatalogEntry>& catalog() {
  static const std::vector<CatalogEntry> c = build();
  return c;
}

std::vector<Alternative> catalog_alternatives(const std::string& id_or_name) {
  std::string name = id_or_name.substr(id_or_name.find('/') == std::string::npos ? 0 : id_or_name.find('/') + 1);
  name = name.substr(0, name.find(':'));
  auto lower = [](std::string x) {
    for (char& ch : x) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    return x;
  };
  const std::string l = lower(name);
  const CatalogEntry* match = nullptr;
  for (const CatalogEntry& e : catalog())
    if (e.id == id_or_name || l == lower(e.model) || l.rfind(lower(e.model) + "-", 0) == 0) {
      match = &e;
      break;
    }
  std::vector<Alternative> out;
  if (match)
    for (const CatalogEntry& e : catalog())
      if (e.model == match->model && e.id != id_or_name) out.push_back({e.id, e.footprint});
  return out;
}

std::vector<Recommendation> recommend(const Mac& mac, const Plan& plan,
                                      const std::function<bool(const std::string&)>& cached, bool all) {
  std::vector<Recommendation> out;
  const std::vector<CatalogEntry>& cat = catalog();
  auto check = [&](const CatalogEntry& e, int ctx) {
    Plan q = plan;
    q.max_ctx = ctx;
    Recommendation r;
    r.entry = &e;
    const Verdict v = judge(e.footprint, q, mac);
    r.need = v.need;
    r.fits = v.fits;
    if (e.experts) {  // a streaming model: the rest of the memory caches its experts
      r.cache = std::max<int64_t>(0, v.usable - v.need.total());
      r.fits = r.fits && double(r.cache) >= e.cache_share * double(e.experts);
    }
    r.context = ctx == plan.max_ctx ? 0 : ctx;
    r.cached = cached && cached(e.id);
    return r;
  };
  // Precision first: each width (8, 6, then 4 bits) at the plan's --context,
  // else at 16384; 8192 only when nothing else fits.
  std::vector<int> wide = {plan.max_ctx}, last;
  if (plan.kv_gb <= 0) {
    if (16384 < plan.max_ctx) wide.push_back(16384);
    if (8192 < plan.max_ctx) last.push_back(8192);
  }
  for (size_t i = 0; i < cat.size();) {
    size_t j = i;
    while (j < cat.size() && cat[j].model == cat[i].model) ++j;  // [i, j): one model's files
    if (all) {
      for (size_t k = i; k < j; ++k) out.push_back(check(cat[k], plan.max_ctx));
    } else {
      bool done = false;
      for (const std::vector<int>* contexts : {&wide, &last})
        for (int bits : {8, 6, 4})
          for (int ctx : *contexts)
            for (size_t k = i; k < j && !done; ++k)
              if (cat[k].bits == bits)
                if (Recommendation r = check(cat[k], ctx); r.fits) out.push_back(r), done = true;
      // Other files of the model already downloaded that fit too.
      if (done)
        for (size_t k = i; k < j; ++k)
          if (&cat[k] != out.back().entry && cached && cached(cat[k].id) && check(cat[k], plan.max_ctx).fits)
            out.back().also.push_back(cat[k].id);
    }
    i = j;
  }
  return out;
}

std::string recommendations_text(const Mac& mac, const std::vector<Recommendation>& recs, bool all) {
  std::string out;
  bool any_fit = false, any_cached = false;
  size_t w = 8;
  for (const Recommendation& r : recs) {
    const std::string arg = r.entry->id + (r.context ? " --context " + std::to_string(r.context) : "");
    w = std::max(w, arg.size());
    any_fit = any_fit || r.fits;
    any_cached = any_cached || r.cached;
  }
  if (!all && !any_fit) return "No model of the table fits " + mac_text(mac) + ".\n";
  out += (all ? "Every file of the table, for " : "Models that fit ") + mac_text(mac) + (all ? ":\n\n" : ", best first:\n\n");
  char line[512];
  std::snprintf(line, sizeof line, "  %-*s  %8s  %9s  %s\n", int(w), "--model", "on disk", "needs", "model");
  out += line;
  std::string last;
  for (const Recommendation& r : recs) {
    const std::string arg = r.entry->id + (r.context ? " --context " + std::to_string(r.context) : "");
    std::string what = r.entry->model;
    if (r.entry->model != last || !all) what += ", " + r.entry->kind;
    if (r.entry->experts)
      what += r.cache >= r.entry->experts ? " (all its experts fit in memory)"
                                          : " (" + gb(r.cache) + " of its " + gb(r.entry->experts) + " of experts cached)";
    if (all && !r.fits) what = "does not fit; " + what;
    last = r.entry->model;
    std::snprintf(line, sizeof line, "%c %-*s  %8s  %9s  %s\n", r.cached ? '*' : ' ', int(w), arg.c_str(),
                  gb(r.entry->footprint.disk).c_str(), gb(r.need.total()).c_str(), what.c_str());
    out += line;
    for (const std::string& id : r.also) out += "  " + std::string(w, ' ') + "  also downloaded: " + id + "\n";
  }
  out += "\n";
  if (any_cached) out += "* in the Hugging Face cache already.\n";
  out += "\"needs\" is GPU memory: the weights, the conversation state for 2 x --context tokens and the scratch.\n";
  return out;
}

}  // namespace espetero::models
