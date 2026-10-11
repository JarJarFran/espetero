// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// What a model needs in GPU memory and what this Mac has, so a model
// that cannot run here is refused before its download and before its load,
// with the numbers and smaller files of the same model that fit.
//
// The need estimates what the engine allocates at its floor (src/model/model.mm):
//   weights   the bytes the loader keeps in GPU memory: the files' tensors as
//             the engine stores them (GGUF codes widen to 4, 5, 6 or 8 bits
//             with their own scales; MLX 2- and 3-bit codes to 8 bits), the
//             vision tower and MTP head unless turned off, a second copy of a
//             tied LM head and the 2-bit draft copy of the LM head;
//             Qwen3.8-Flash-Next's experts and n-gram table stay on the SSD
//   pools     the state slots and KV blocks at their floor: 8 user slots (or
//             --slots) plus 9 temporary and zero slots, and 2 x --context
//             tokens of KV (or --kv-gb); --memory-gb (3 GB for Flash-Next)
//             when larger
//   scratch   the pass buffers for --max-rows rows (ensure_scratch and its
//             MoE / Flash-Next parts), the readout logits and 0.5 GB for the
//             vision tower's passes and staging
// against the GPU working set (MTLDevice recommendedMaxWorkingSetSize, which
// iogpu.wired_limit_mb raises) and the RAM.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "../util/json.h"

namespace espetero::models {

// The geometry the pools and scratch depend on (the parts of the engine's
// Config, src/model/impl/config.h, that size memory).
struct Geometry {
  bool known = false;
  int D = 0, L = 0, n_full = 0, n_linear = 0, H = 0, Hkv = 0, HD = 256;
  int Hk = 0, Hv = 0, Dk = 128, Dv = 128, conv_k = 4, I = 0, V = 0;
  int E = 0, k = 0, Ie = 0, I_shared = 0;  // mixture of experts (E = 0: dense)
  int hc = 1, hc_rank = 0;                  // Flash-Next residual streams
  bool q4 = false;                          // qwen4_exp (Flash-Next): experts and n-gram table on the SSD
  // Gemma 4: n_full global layers (Hkv_g x HDg K/V per token in the
  // paged pool), n_swa sliding layers (a ring of `window` positions per slot).
  bool gemma = false;
  int HDg = 0, Hkv_g = 0, window = 0, n_swa = 0, ld = 0;
  std::string unsupported;                  // why the engine refuses this geometry ("" = it loads)

  int64_t kv_bytes_per_token(bool mtp) const;
  int64_t slot_bytes(bool mtp) const;
  int64_t row_bytes() const;      // scratch per pass row
  int64_t readout_bytes() const;  // logits and top-k scratch of the LM-head readouts
};

// MLX config.json (text_config first, as the loader reads it).
Geometry geometry_from_config(const json::Value& config);

// ---------------------------------------------------------------- headers
struct Tensor {
  std::string name;
  std::string dtype;           // safetensors dtype ("U32", "BF16", ...) or GGUF type name
  int gguf_type = -1;          // GGUF: ggml type
  std::vector<int64_t> shape;  // as stored (GGUF: ggml order, dims[0] contiguous)
  int64_t bytes = 0;           // safetensors: data bytes; GGUF: 0
  int64_t elements() const;
};

// A safetensors header: 8-byte length, then the JSON tensor table. Throws
// std::runtime_error when `bytes` does not hold a valid header; `need` gets the
// header's full size (8 + length) so a caller with too few bytes can read more.
std::vector<Tensor> parse_safetensors(const std::string& bytes, size_t* need = nullptr);

struct GgufHeader {
  std::vector<Tensor> tensors;
  std::vector<std::pair<std::string, json::Value>> meta;  // scalar keys (arrays skipped)
  const json::Value* get(const std::string& key) const;
  int64_t get_int(const std::string& key, int64_t def) const;
  std::string get_string(const std::string& key, const std::string& def) const;
};
// GGUF header (metadata and tensor table). Returns false when `bytes` ends
// before the tensor table does (read more); throws when it is not a GGUF file.
bool parse_gguf(const std::string& bytes, GgufHeader& out);
Geometry geometry_from_gguf(const GgufHeader& h);

// Bytes a GGUF tensor of ggml `type` takes in GPU memory once loaded
// (gguf.h / matrix.h: 4, 5 and 6-bit codes in planes, others as bytes, the
// native Q6_K and Q8_0 scales, affine half scale + bias per group elsewhere).
int64_t gguf_engine_bytes(int type, int64_t elements);
std::string ggml_type_name(int type);

// ---------------------------------------------------------------- footprint
// Reads `length` bytes at `offset` of a model file (relative path in the
// model's directory or repository). Throws on failure.
using ReadFn = std::function<std::string(const std::string& path, int64_t offset, int64_t length)>;

struct FileRef {
  std::string path;  // relative to the directory or repository
  int64_t size = 0;
};

// A model's files as the engine would hold them, by part.
struct Footprint {
  std::string format;   // "MLX" or "GGUF"
  std::string quant;    // "4-bit", "mixed 4/8-bit", "Q6_K", ...
  int64_t disk = 0;     // file bytes (weights, projector, MTP head)
  int64_t weights = 0;  // GPU bytes of everything below that is loaded
  int64_t base = 0;     // part of weights: the model's own weight files (GGUF: the model file)
  int64_t vision = 0;   // part of weights: vision tower or projector (--no-vision, --mmproj none)
  int64_t mtp = 0;      // part of weights: MTP head (--mtp none); the 2-bit draft head included
  int64_t streamed = 0; // bytes that stay on the SSD (Flash-Next experts and n-gram table)
  bool has_mtp = false, has_vision = false;
  Geometry geo;
  std::string what;     // the files, for messages
  std::string arch;     // GGUF general.architecture, MLX model_type (text_config's first)
  bool mlx_layout = false;  // MLX: converted for MLX (language_model.model.embed_tokens.weight), as the loader needs
};

// MLX: config.json and the repository's or directory's files (every
// *.safetensors is read by its header).
Footprint footprint_mlx(const json::Value& config, const std::vector<FileRef>& files, const ReadFn& read);
// GGUF: the model file, and the projector and MTP head when there are any.
Footprint footprint_gguf(const FileRef& model, const FileRef* mmproj, const FileRef* mtp, const ReadFn& read);
// A local model: an MLX directory (plus a Flash-Next drafter beside it) or a
// .gguf file (plus the projector and MTP head the engine finds next to it, or
// `mmproj` / `mtp` when given; "none" drops them). Throws std::runtime_error.
Footprint footprint_local(const std::string& path, const std::string& mmproj = "", const std::string& mtp = "");

// ---------------------------------------------------------------- need
// The options memory depends on (--context, --slots, --kv-gb, --memory-gb,
// --expert-cache-gb, --max-rows, --no-vision, --mtp none).
struct Plan {
  int max_ctx = 32768;
  int slots = 0;  // 0: automatic (8 at least)
  int max_rows = 4096;
  double kv_gb = 0, memory_gb = 0, expert_cache_gb = 0;
  bool vision = true, mtp = true;
};

struct Need {
  int64_t weights = 0, pools = 0, scratch = 0, cache = 0;  // cache: --expert-cache-gb
  int64_t total() const { return weights + pools + scratch + cache; }
};
Need need(const Footprint& f, const Plan& p);

// This Mac: RAM and the GPU working set.
struct Mac {
  int64_t ram = 0;  // hw.memsize
  int64_t gpu = 0;  // recommendedMaxWorkingSetSize
  std::string chip;
  int64_t usable() const;  // what one process can hold in GPU memory: min(gpu, ram)
};
Mac this_mac();  // mac.mm
// A Mac with `ram_gb` of RAM and macOS's default working set (2/3 of the RAM
// up to 36 GB, 3/4 above), for recommendations on other Macs.
Mac mac_with_ram(double ram_gb);

// ---------------------------------------------------------------- the check
// An alternative to suggest when a model does not fit: another file of the
// same model ("--model ID") and its footprint.
struct Alternative {
  std::string model;  // the --model value
  Footprint footprint;
};

struct Verdict {
  bool fits = true;
  Need need;
  int64_t usable = 0;
  int context_that_fits = 0;  // > 0: the largest --context (power of two >= 2048) that fits, when the plan's does not
};
Verdict judge(const Footprint& f, const Plan& p, const Mac& mac);

// Why `label` (a repo id or a path) does not fit: needed against usable GB
// by part, a smaller --context that fits, the alternatives that fit (largest
// first), then `ending` (what happens now and the way out).
std::string refusal(const std::string& label, const Footprint& f, const Plan& p, const Mac& mac, const Verdict& v,
                    const std::vector<Alternative>& alternatives, const std::string& ending);

// Other files of a local model: MLX directories beside it named with other
// bit widths (bit_siblings), other GGUF models in its directory.
std::vector<Alternative> local_alternatives(const std::string& path, const std::string& mmproj = "",
                                            const std::string& mtp = "");

// "12.3 GB"
std::string gb(int64_t bytes);
// GB, MB or KB, whichever reads best ("12.3 GB", "28.8 MB", "279 KB").
std::string size_text(int64_t bytes);
// A path for messages: the home directory as "~".
std::string tilde(const std::string& path);

// Other bit widths of a repository named by them: "-8bit" -> "-6bit", "-5bit",
// "-4bit"; "-oQ8e-mtp" -> "-oQ6e-mtp", ... (candidates; the caller checks
// which exist).
std::vector<std::string> bit_siblings(const std::string& name);

// The in-memory size of a GGUF file of quantization `name` ("UD-Q6_K_XL",
// "Q4_K_M", ...) relative to its file size, for files whose header is not read.
double gguf_quant_factor(const std::string& name);

// ---------------------------------------------------------------- recommendations
// A curated table of public repositories of the supported families
// (catalog.cpp; their sizes on the Hub, 2026-10-09): models best first and,
// for each, its files in the order they are preferred: 8-bit, then 6-bit,
// then 4-bit when memory forces it, a GGUF file with
// an MTP head before an MLX file without one at the same width.
struct CatalogEntry {
  std::string model;  // "Qwen3.8-27B"
  std::string kind;   // "dense, best quality", ...
  std::string id;     // the --model value
  int bits = 0;       // 8, 6 or 4
  Footprint footprint;
  double cache_share = 0;  // streaming models: the part of the experts the cache must hold to be recommended
  int64_t experts = 0;     // their bytes
};
const std::vector<CatalogEntry>& catalog();

struct Recommendation {
  const CatalogEntry* entry = nullptr;
  Need need;
  int context = 0;    // > 0: with --context this (the plan's does not fit)
  int64_t cache = 0;  // streaming models: GPU memory left for the expert cache
  bool fits = true;
  bool cached = false;
  std::vector<std::string> also;  // other files of the model in the cache that fit too
};

// The best file of each catalog model that fits `mac` with `plan`, best model
// first: the model's files in order at the plan's --context, else at 16384,
// else 8192. `all`: every file, those that do not fit too. `cached` tells
// which --model values are in the Hugging Face cache.
std::vector<Recommendation> recommend(const Mac& mac, const Plan& plan,
                                      const std::function<bool(const std::string&)>& cached, bool all = false);

// The catalog's other files of the model `id_or_name` names (a catalog id,
// or a repository or directory whose name starts with the model's, such as
// Qwen3.8-27B-8bit), as alternatives for a refusal.
std::vector<Alternative> catalog_alternatives(const std::string& id_or_name);

// The table for the terminal: the --model values, their sizes, what they
// need and which are downloaded already.
std::string recommendations_text(const Mac& mac, const std::vector<Recommendation>& recs, bool all);

}  // namespace espetero::models
