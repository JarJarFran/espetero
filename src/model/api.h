// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Native model interface (pure C++; the implementation is Objective-C++ over
// Metal). One NativeModel per loaded checkpoint; calls are not thread-safe —
// the engine's scheduler thread is the only caller.
//
// Work is expressed as a forest of trees run in one GPU pass:
//   * a tree's root continues a parent slot (or an empty context) and may be
//     checkpointed into `save` (first save_len root tokens) and `work` (whole
//     root) in the same pass;
//   * leaves continue the root (or the parent when there is no root) and never
//     see each other; they are read out at their last token (candidate
//     logits, log-sum-exp, argmax or full logits);
//   * projections and MLPs run over every row of every tree at once, sequence
//     mixing (delta rule, attention) per tree in the same dispatches.
// Slots are sequence states: recurrent + convolution state and a list of KV
// blocks. Slots that share a prefix share its KV blocks (copy on write).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../util/json.h"

namespace espetero {

struct LoadOptions {
  int slots = 0;            // user state slots (0: automatic from the memory budget)
  double kv_gb = 0;         // KV pool size in GB (0: automatic)
  double memory_gb = 0;     // budget for slots + KV (0: ~60% of free GPU memory)
  double expert_cache_gb = 0;  // models whose experts stay on the SSD (Qwen4-Exp): GB of expert slots in memory
                               // (0: automatic, < 0: none, every layer's experts read per pass)
  std::string expert_store;         // expert records (Qwen4-Exp): "" / "auto" = the default place if built, a directory, or "none"
  bool expert_store_build = false;  // build the records where they are missing (one-time: ~68 GB written for Flash-Next)
  int max_ctx = 32768;      // tokens per sequence
  int block_tokens = 64;    // KV block size
  int max_rows = 4096;      // rows per pass
  int max_readouts = 256;   // LM-head rows per pass
  bool vision = true;
  std::string mmproj;       // GGUF: "" = look next to the model, "none" = text only
  std::string mtp;          // "" = the model's own head, a path, or "none"
  bool mtp_head4 = true;    // 4-bit copy of the LM head for drafts
  bool mtp_q8 = true;       // dense (bf16) MTP-layer matrices as 8-bit affine copies (on by default; drafts are verified, output unchanged)
  std::string dense;        // decoder matrices served as fp16: gate_up,down,in_proj,out|all
  int tune_max_m = 1024;    // GEMM autotune up to this many rows (0: off)
  bool gemv = true, gemv8 = true;
  bool prune_last = true;
  int attn_split = 512;     // max cached keys per split-K attention tile (0: off); few-tile passes split down to 64
  int mtp_vocab = 0;        // MTP drafts score LM head rows [0, mtp_vocab) only (0: the whole vocabulary, opt-in)
  int mtp_coarse = 0;       // MTP drafts from a 2-bit copy of the LM head: this many 32-id column blocks per draft rescored
                            // with the LM head (0: off; drafts are verified, output unchanged)
  bool verbose = false;
};

struct ModelInfo {
  std::string device, format, path;
  int hidden = 0, intermediate = 0, layers = 0, full_layers = 0, linear_layers = 0;
  int heads = 0, kv_heads = 0, vocab = 0, eos_token_id = -1;
  int slots = 0, kv_blocks = 0, block_tokens = 0, max_ctx = 0, max_rows = 0, max_readouts = 0;
  size_t weight_bytes = 0, kv_bytes = 0, state_bytes = 0, slot_state_bytes = 0;
  bool vision = false, mtp = false;
  bool leaves = true;  // multi-branch passes (draft leaves, decision trees)
  bool commit = false;  // verify-commit: TreeSpec::commit_chain + NativeModel::commit
  bool hidden_states = true;  // TreeSpec::out_hidden readouts (not Qwen4-Exp yet)
  int draft_default = 0;  // MTP drafts per pass when --draft is not given (0: the engine's default, 5)
  std::string family = "qwen3_5";  // "qwen3_5" (also qwen3_5_moe, qwen4_exp) or "gemma4"
  std::vector<int> end_ids;        // end-of-turn ids the checkpoint's config names beside the tokenizer's (Gemma 4)
  int vision_patch = 0, vision_merge = 0, vision_patch_dim = 0, vision_temporal = 0;
  double load_ms = 0, tune_ms = 0;
  json::Value extra;        // format details (matrix widths, source types, ...)
  json::Value to_json() const;
};

// An image whose vision-tower embeddings replace root rows
// [row, row + grid_t * grid_h * grid_w / merge^2). `pixels`: the processor's
// patch rows (Qwen2/3-VL order), float32.
struct ImageRef {
  const float* pixels = nullptr;
  int grid_t = 1, grid_h = 0, grid_w = 0;
  int row = 0;
};

struct TreeSpec {
  int parent = -1;              // slot, -1: empty context
  int save = -1;                // checkpoint parent + root[:save_len] (may equal parent)
  int save_len = 0;             // 0: whole root
  int work = -1;                // also checkpoint parent + whole root (differs from save/parent)
  const int32_t* root = nullptr;
  int root_len = 0;
  const int32_t* root_pos = nullptr;  // [root_len][3] M-RoPE positions (images), or null
  const ImageRef* images = nullptr;
  int n_images = 0;
  int n_leaves = 0;
  const int32_t* leaf_offsets = nullptr;  // n_leaves + 1
  const int32_t* leaf_tokens = nullptr;
  const int32_t* cand_offsets = nullptr;  // n_leaves + 1, or null (vocabulary readouts only)
  const int32_t* cand_ids = nullptr;
  bool score_root = false;      // one more readout at the root's last row (after the leaves)
  bool mtp = false;             // keep the saved slots' MTP draft state
  // Verify-commit: leaves [0, commit_chain) are one chain (leaf i is the
  // first i + 1 tokens of leaf commit_chain - 1). The pass writes their K/V
  // (and MTP K/V) at the positions they would take and keeps their DeltaNet
  // inputs, so NativeModel::commit can make an accepted prefix part of the
  // save slot instead of a next pass recomputing it. In-place trees only
  // (save == parent, whole root, no work slot); ModelInfo::commit says if the
  // model supports it.
  int commit_chain = 0;
  // Outputs (null: not wanted). Readout i of the tree: leaves first, then the root.
  float* out_cand_logits = nullptr;  // [cand_offsets[n_leaves]]
  float* out_lse = nullptr;          // [readouts]
  int32_t* out_argmax = nullptr;     // [readouts]
  bool want_logits = false;          // full logits via NativeModel::logits(tree, i)
  // Final-normed hidden states (the LM head's input) of every root row,
  // float32 [root_len][hidden] (Clef joint heads). Disables last-layer
  // pruning for the pass; not supported for qwen4_exp.
  float* out_hidden = nullptr;
  // > 0: the top-`topk` (<= 32) scaled logits x = logit / temperature of every
  // readout, their ids, the maximum and the full-vocabulary sum of exp(x - max),
  // computed on the GPU (NativeModel::topk).
  int topk = 0;
  float temperature = 1.0f;
  // At least this many top-k candidates (<= 32) for this tree's readouts,
  // without changing the head the GPU draws from (`topk`); with topk 0 they
  // are raw logits (temperature 1). Logprobs of sampled requests.
  int topk_min = 0;
  // With topk: also draw each readout's token on the GPU: position-keyed
  // Gumbel-max over the top-k head cut by min_p and top_p (NativeModel::topk
  // returns it as `sampled`). Readout positions: draw_pos0 for the root's
  // readout, draw_pos0 + the leaf's length for a leaf's.
  bool gpu_draw = false;
  uint64_t seed = 0;
  int64_t draw_pos0 = 0;
  float top_p = 1.0f, min_p = 0.0f;
};

// GPU top-k candidates of one readout (valid until the next run): ids and
// scaled logits sorted by (x desc, id asc), the maximum scaled logit and the
// sum of exp(x - max) over the whole vocabulary.
struct TopK {
  const int32_t* ids = nullptr;
  const float* x = nullptr;
  int n = 0;
  float max = 0, total = 0;
  int32_t sampled = -1;  // the GPU draw (gpu_draw), else -1
};

struct PassStats {
  double gpu_ms = 0, encode_ms = 0, wall_ms = 0;
  int rows = 0, readouts = 0, dispatches = 0, barriers = 0;
};

struct DraftRequest {
  int slot = -1;                  // saved by a pass with mtp = true
  const int32_t* tokens = nullptr;  // accepted leaf tokens of that pass, then the next token
  int n_tokens = 0;
  int n_draft = 0;
  int32_t* out = nullptr;         // [n_draft]
  int32_t* out2 = nullptr;        // optional [n_draft]: each step's runner-up (sibling leaves)
  int lh_first = 0;               // saved leaf row of tokens[0]'s successor: the accepted leaf's first row
  // Draw the drafts as the target draws its tokens (the top-k head of
  // the draft logits at the temperature, top-p / min-p, Gumbel noise keyed by
  // seed, position and id: the GPU draw of TreeSpec::gpu_draw); draft i is
  // the guess for the token at position pos0 + i. Off: greedy drafts.
  bool draw = false;
  uint64_t seed = 0;
  int64_t pos0 = 0;
  float temperature = 1.0f, top_p = 1.0f, min_p = 0.0f;
  int top_k = 0;
};

// Verify-commit: the first n rows of the chain that slot's last pass
// kept (TreeSpec::commit_chain) become part of the slot.
struct CommitRequest {
  int slot = -1;
  int n = 0;
};

class NativeModel {
 public:
  static std::unique_ptr<NativeModel> load(const std::string& path, const LoadOptions& options);
  virtual ~NativeModel() = default;

  virtual const ModelInfo& info() const = 0;

  // Slots.
  virtual int slot_alloc() = 0;                 // -1 when none is free
  virtual void slot_free(int slot) = 0;         // releases its KV blocks
  virtual void slot_reset(int slot) = 0;        // empty (keeps the slot allocated)
  virtual void slot_fork(int dst, int src) = 0; // dst := src (KV shared, state copied)
  virtual int slot_len(int slot) const = 0;
  virtual int slot_next_pos(int slot) const = 0;
  virtual bool slot_mtp(int slot) const = 0;
  virtual int free_slots() const = 0;
  virtual int free_kv_tokens() const = 0;       // free KV blocks x block size
  // New KV blocks (x block size) a pass over `trees` takes, as run() counts them
  // before it starts (the engine makes room first).
  virtual int pass_kv_tokens(const std::vector<TreeSpec>& trees) const = 0;
  // KV blocks (x block size) that freeing every slot of `slots` would give
  // back: the blocks no other slot holds (a block they share counts once).
  // The engine weighs what evicting checkpoints would free before it
  // evicts any.
  virtual int freed_kv_tokens(const std::vector<int>& slots) const = 0;
  // Checkpoint persistence: a slot's KV blocks, recurrent state, MTP rows and
  // (Qwen4-Exp) n-gram mixing state written to a file, streamed from the
  // buffers (no copy in memory), and read back into a freshly allocated
  // slot. Valid for the same weights, layout and build only (the signature).
  // slot_load: the slot, -1 when the file is not a checkpoint of this model,
  // -2 when no slot or KV space is free.
  virtual bool slot_save(int slot, const std::string& path) = 0;
  virtual int slot_load(const std::string& path) = 0;
  virtual std::string state_signature() const = 0;  // changes when saved checkpoints stop being valid

  // One GPU pass over a forest; throws std::runtime_error on invalid input or
  // GPU failure (slots are left as before the pass on validation errors).
  virtual void run(const std::vector<TreeSpec>& trees, PassStats* stats = nullptr) = 0;
  // `steps` decode passes back to back: every tree is a plain decode
  // step (parent == save, one root token, score_root, no leaves, no MTP, no
  // images, no logits), greedy or drawn on the GPU (gpu_draw: draw_pos0
  // advances by one a step). Pass j > 0 embeds pass j - 1's token on the
  // GPU; the passes are separate command buffers ordered by a GPU event and
  // planned while the previous one runs, so the CPU waits once. out:
  // [steps][trees] argmax tokens (out[j][t] follows pass j's root).
  virtual void run_chain(const std::vector<TreeSpec>& first, int steps, int32_t* out, PassStats* stats = nullptr) = 0;
  // Full-vocabulary logits of readout i of tree t of the last run (valid until
  // the next call). Requires want_logits.
  virtual const float* logits(int tree, int readout) const = 0;
  // Top-k candidates of readout i of tree t of the last run (requires topk).
  virtual TopK topk(int tree, int readout) const = 0;

  // MTP drafts for several slots in one GPU pass (greedy, n_draft each).
  virtual void draft(const std::vector<DraftRequest>& requests, PassStats* stats = nullptr) = 0;

  // Verify-commit: append the accepted chain rows of each slot's last
  // pass (requests of the pass that just ran, before any other pass).
  virtual void commit(const std::vector<CommitRequest>& requests, PassStats* stats = nullptr) = 0;

  // Vision tower only: merged token embeddings of the images (float32
  // [sum tokens, hidden]).
  virtual void encode_images(const std::vector<ImageRef>& images, float* out, PassStats* stats = nullptr) = 0;

  // Development: per-kernel GPU time (every dispatch in its own command buffer).
  virtual void set_profile(bool on) = 0;
  virtual std::string profile_json() = 0;
  // Development: switch an engine option at run time for A/B tests
  // ("serial", "dn_chunk", "fuse_norm", "fuse_gated", "attn_split", "prune_last"). Returns false if unknown.
  // The engine itself sets "gemm_fixed" around embedding passes (GEMMs on one tile whatever their rows).
  virtual bool set_option(const std::string& name, int value) = 0;
  // Development: internal micro-benchmarks by name ("gap": dispatch-gap
  // characterization); returns JSON.
  virtual std::string bench_json(const std::string& what) = 0;
};

}  // namespace espetero
