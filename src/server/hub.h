// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Hugging Face repo ids for --model: `ORG/NAME` (an MLX repository) or
// `ORG/NAME:QUANT` (a GGUF repository and the quantization to run). The model
// is the newest complete snapshot in the local hub cache (huggingface_hub's
// layout: models--ORG--NAME/snapshots/<revision>/); when none is there, it is
// downloaded into that cache first, natively (download.h).
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "models.h"

namespace espetero::hub {

// A failure for the user: what is missing and the commands that fix it.
struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Ref {
  std::string repo;   // ORG/NAME
  std::string quant;  // GGUF quantization after ':' ("" = none)
};

// Whether `model` is a repo id: not an existing path, exactly one '/',
// Hugging Face's characters ([A-Za-z0-9._-]) on both sides, then an optional
// ":QUANT".
bool parse_ref(const std::string& model, Ref& ref);

// The hub cache as huggingface_hub finds it: $HF_HUB_CACHE, else
// $HUGGINGFACE_HUB_CACHE, else $HF_HOME/hub, else
// $XDG_CACHE_HOME/huggingface/hub, else ~/.cache/huggingface/hub.
std::string cache_dir();

// HF_HUB_OFFLINE is 1, true, yes or on.
bool offline_env();

struct Model {
  std::string path;      // the MLX snapshot directory or the .gguf file
  std::string mmproj;    // GGUF: a projector the engine would not find next to `path` ("" = its own lookup)
  std::string mtp;       // GGUF: an MTP head the engine would not find next to `path` ("" = its own lookup)
  std::string revision;  // the snapshot's revision (its directory name, a commit hash)
};

// The GGUF files of a repository (paths relative to its root) that Espetero
// reads: the model file `ref.quant` names, matched case-insensitively against
// the end of the file name after a '-', '_' or '.' ("UD-Q6_K" selects
// X-UD-Q6_K.gguf, not X-UD-Q6_K_XL.gguf; the shortest name wins a tie), or
// the only model file when there is no selector; one projector (mmproj*, F16
// first) and one MTP head (MTP/mtp-*.gguf). imatrix files are not models.
struct GgufPick {
  std::string model, mmproj, mtp;
};
// false: no model file matches. Throws Error when the choice is not unique
// (no selector and several models, a tie) or the model is split in parts.
bool pick_gguf(const std::vector<std::string>& files, const Ref& ref, GgufPick& out);

// The newest complete snapshot of `ref` in `cache` (the revision refs/main
// names first, then the most recently modified), without the network. MLX:
// config.json, tokenizer.json and every weight file; GGUF: the selected
// file. false when none is usable.
bool find_cached(const std::string& cache, const Ref& ref, Model& out);

// What to do when the model does not fit this Mac's GPU memory
// (models.h): refuse before downloading it, warn, or not look.
enum class MemoryCheck { Refuse, Warn, Skip };

struct ResolveOptions {
  bool offline = false;
  MemoryCheck memory = MemoryCheck::Refuse;
  models::Plan plan;  // the settings the model will run with
};

// find_cached, else (unless `offline`) the files the engine reads from the
// Hub's listing, downloaded into that cache (download.h), then find_cached
// again. Before downloading, the memory the model needs (its files' sizes
// and headers on the Hub) is checked against this Mac. MLX repositories
// download config.json first, so a model of another family is refused before
// its weights. Throws Error with what to do when it cannot.
Model resolve(const Ref& ref, const ResolveOptions& opt);
// The same without the memory check.
Model resolve(const Ref& ref, bool offline);

// `espetero pull`: resolve without the cache's shortcut, so the
// repository's newest revision is fetched (files already there are not
// downloaded again).
Model pull(const Ref& ref, const ResolveOptions& opt);

// The supported models on this Mac, for `espetero list`: the newest
// complete snapshot of each repository in the hub cache `cache` (MLX, or
// each GGUF model in it, named ORG/NAME:QUANT) and the MLX directories and
// .gguf files under `models_dir` (~/models/<family>/<name>). `other` gets
// the number of models of other families (or geometries) left out.
struct Installed {
  std::string id;       // the --model value: a repo id or a path
  std::string path;     // the snapshot directory or the .gguf file
  bool in_cache = false;  // in the Hugging Face cache (else under models_dir)
  int64_t disk = 0;     // bytes of its files on disk (each blob once)
  models::Footprint footprint;
};
std::vector<Installed> installed(const std::string& cache, const std::string& models_dir, int* other = nullptr);

// `espetero rm`: what removing `ref` from the hub cache deletes, as
// huggingface_hub's `hf cache delete` does: ORG/NAME the whole repository
// (every revision, its blobs and refs); ORG/NAME:QUANT that GGUF model's
// files in every snapshot (and a snapshot's projector and MTP head when no
// other model is left in it), then the blobs no snapshot links to any more.
// Nothing outside models--ORG--NAME is touched and links are never followed.
// Throws Error when `ref` is not in the cache.
struct Removal {
  std::string repo_dir;
  bool whole = false;               // the repository's directory goes
  std::vector<std::string> links;   // snapshot entries (links) to delete
  std::vector<std::string> blobs;   // blobs nothing links to afterwards
  std::vector<std::string> files;   // what the user sees: the model files removed (relative to the snapshot)
  int64_t bytes = 0;                // freed
};
Removal plan_removal(const std::string& cache, const Ref& ref);
void remove(const std::string& cache, const Removal& r);

// Other files of the model `ref` names that a refusal can suggest: GGUF
// quantizations of its repository, MLX repositories named with other bit
// widths (models::bit_siblings) that exist. Empty when the Hub cannot be
// reached (or `offline`).
std::vector<models::Alternative> alternatives(const Ref& ref, const models::Footprint& fp, bool offline);

}  // namespace espetero::hub
