// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Qwen3.5 model state on the GPU: decoder layers, vision tower, MTP head,
// the paged KV pool, the state pool (recurrent + conv + MTP rows per slot)
// and per-pass scratch.
//
// Memory layout
//  * KV pool: one buffer, one region per KV layer (full-attention layers,
//    then the MTP layer). A region holds n_blocks blocks of bt tokens, each
//    block [K rows][V rows] of kv_dim halves. Slots own lists of block ids;
//    blocks are reference counted, so checkpoints that share a prefix share
//    its blocks, and a partially filled shared block is copied on write.
//  * State pool: one buffer, one region per linear layer for conv states
//    (3 x C floats per slot) and one for recurrent states (Hv x Dv x Dk floats
//    per slot), then the MTP rows region. Region-per-layer keeps every
//    dispatch's footprint one contiguous range (cheap hazard tracking) and
//    lets one dispatch serve every tree of a forest by slot index.
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "config.h"
#include "gemm.h"

namespace espetero::native {

// Qwen4-Exp hyper-connection block (docs/qwen4_exp.md 2): a grouped RMSNorm
// of the hc streams (weight 1 + w folded into `norm`), the low-rank read
// A_down [r, W] (the hc injection rows appended as rows r..r+hc when they
// share its format, else `inj`), and the per-channel stream gates A_up [W, r].
// The final mixer has no injection rows.
struct HCBlock {
  Buf norm;  // fp32 [W]
  Mat down, inj, up;
  bool inj_fused = false;
  bool present() const { return bool(norm); }
};

struct Layer {
  bool linear = false;
  Buf in_norm, post_norm;
  int I = 0;     // MLP width of this layer (the shared expert's for MoE layers)
  Mat gate_up;   // gate/up rows interleaved in blocks of 8 (SwiGLU epilogue)
  Mat gate, up;  // instead, when the two keep different formats
  Mat down;
  // Mixture of experts: gate_up / down above are the shared expert. All
  // experts' gate/up rows (interleaved by 8 per expert) form one matrix of
  // E * 2 * Ie rows; all experts' down rows one of E * D rows.
  bool moe = false;
  Mat router, shared_gate;  // [E, D], [1, D]
  QW experts_gu, experts_down;
  // Gated DeltaNet: in_proj = [qkv; z; a; b]; full attention: [q|gate; k; v].
  std::vector<Mat> in_proj;
  Mat out;
  Buf conv_w, a_log, dt_bias, gnorm;
  Buf q_norm, k_norm;
  int index_in_kind = 0;  // index among full or linear layers
  // Qwen4-Exp: the two hyper-connection blocks (token mixer, MoE) and the
  // sparse-attention indexer ([q 4 x 128 | k 128] rows, zero-centred norms + 1).
  HCBlock hca, hcm;
  Mat index_qk;
  Buf iq_norm, ik_norm;
  Buf router_f16;  // Qwen4-Exp: the dense router row-major fp16 [E][D] for the 1-8-row GEMV
  // Gemma 4 (gemma.h): sliding-window or global attention; in_norm /
  // post_norm are input_layernorm / post_attention_layernorm, then the
  // feed-forward norms (MoE layers: _1 after the dense MLP, _2 around the
  // experts), the router's input norm weight (scale / sqrt(D)) and the
  // per-expert output scales, and the layer's output scalar.
  bool swa = false;
  Buf pre_ffn, post_ffn, post_ffn1, pre_ffn2, post_ffn2;
  Buf router_norm, expert_scale;
  float scalar = 1.0f;
};

// Qwen4-Exp n-gram embedding table, left in the checkpoint's safetensors
// (docs/qwen4_exp.md 5): 16 hashed rows of 160 values per token, 4-bit affine
// group 32 (codes, bf16 scales, bf16 biases in three tensors per shard) times
// one weight_scale. Rows are read with pread; the table never enters memory.
struct NgramTable {
  struct Shard { int file = -1; uint64_t w = 0, s = 0, b = 0; };
  std::vector<std::string> files;
  std::vector<int> fds;
  std::vector<Shard> shards;
  int64_t shard_rows = 0, rows = 0;
  int width = 160, group = 32;
  float scale = 1.0f;
  int64_t mult[3] = {0, 0, 0};
  std::vector<int64_t> vocab, offs;  // per hash head (2 x heads_per_ngram)
};

// Qwen4-Exp n-gram mix before layer ple_layer: Kp = norm(W_key e), Vp = W_val e,
// gates from the residual, a dilated causal conv over the gated values.
struct Ple {
  bool present = false;
  Mat key, value;                              // [W, D], [D, D]
  Buf norm_key, norm_query, norm_conv, conv_w;  // fp32 [W] (+1 folded), conv [W, 4]
  NgramTable table;
};

// Routed experts left on the SSD (Qwen4-Exp: 67.95 GB at 4-bit g64). Per layer
// the MLX switch_mlp tensors [E, out, in] of gate, up and down with their
// scales and biases: expert e's slice of a tensor starts at off + e * stride.
struct ExpertPiece {
  int file = -1;
  uint64_t off = 0, stride = 0;
};
struct ExpertLayer {
  ExpertPiece gw, gs, gb, uw, us, ub, dw, ds, db;
  int bits = 4, group = 64;
};
struct ExpertStore {
  bool present = false;
  std::vector<std::string> files;
  std::vector<int> fds;
  std::vector<ExpertLayer> layers;
  // Expert records: one file per layer holding expert e's bytes in the
  // pool's GPU layout (gate/up codes, their fp16 scale pairs, down codes and
  // scale pairs: a one-slot pool) at e * rec_stride, built once from the
  // checkpoint (build_expert_records). A load is one read and two copies.
  std::vector<int> rec_fds;  // empty: the checkpoint's 9 pieces per expert
  size_t rec_bytes = 0, rec_stride = 0;
  size_t rec_o[4] = {};  // gate/up codes, gate/up scale pairs, down codes, down scale pairs
};

// Expert cache: per MoE layer a pool of S expert slots in the engine's
// GPU layout (the matrix of all experts with S in place of E: gate/up rows
// interleaved by 8 [S * 2Ie, D], down [S * D, Ie]); the MoE of a pass runs on
// the pool after its routed ids are remapped to slots. LRU replacement.
struct ExpertPool {
  int S = 0;
  QW gu, dn;
  std::vector<int> slot_of;         // [E] -> slot, -1 when not resident
  std::vector<int> expert_of;       // [S] -> expert, -1 when free
  std::vector<uint64_t> last_use;   // [S]
  // Staged experts of the current long pass: st_count records at `st`,
  // st_stride apart, addressed by the MoE as experts S .. S + st_count - 1.
  int st_count = 0;
  Buf st;
  size_t st_stride = 0;
  // Handshake scratch: generation-stamped marks of the routed experts [E].
  std::vector<uint32_t> mark;
  uint32_t stamp = 0;
  // Use history of every expert of the layer, resident or not (LRU-2:
  // evict the oldest second most recent use): last and previous clock [E].
  std::vector<uint64_t> e_last, e_prev;
  // Online hot set (ESPETERO_XPIN=fraction, vision/classify traffic): use
  // counts of every expert [E], halved every 4096 handshakes of the layer; the
  // top fraction of S by count is pinned (never a victim while others exist).
  std::vector<uint32_t> use_cnt;
  std::vector<char> pinned;  // [E]
  uint32_t pin_age = 0;
  // Eviction key of slot `sl` (smaller goes first; free slots 0).
  uint64_t evict_key(int sl, int policy) const {
    const int x = expert_of[size_t(sl)];
    if (x < 0) return 0;
    if (!pinned.empty() && pinned[size_t(x)]) return ~0ull;
    if (policy == 2 && !e_prev.empty()) return (e_prev[size_t(x)] << 32) | (e_last[size_t(x)] & 0xffffffffull);
    return last_use[size_t(sl)] + 1;
  }
};
struct ExpertCacheStats {
  uint64_t picks = 0, hits = 0, loads = 0, bytes = 0, fallbacks = 0, admits = 0, handshakes = 0;
  double load_ms = 0, wait_ms = 0, gpu_ms = 0;
  // Prefetch: guessed experts queued, loads served by a guessed entry
  // (ready or still reading at the handshake), loads read on demand, bytes
  // read for guesses, host time waiting for entries at handshakes.
  uint64_t pf_issued = 0, pf_used = 0, pf_late = 0, pf_demand = 0, pf_bytes = 0;
  double pf_wait_ms = 0;
  // Split handshakes: resident tiles committed before the reads were
  // done, and the part of pf_wait_ms spent while they ran.
  uint64_t splits = 0;
  double split_ms = 0;
};

// Expert prefetch: entries of one expert each (the host layout of its
// 9 pieces, then gate and up interleaved by 8 rows) in one shared buffer,
// filled by a pool of reader threads. At a routing handshake a missing expert
// found in an entry is repacked into its pool slot from there; guessed
// experts of the next layer are queued behind the demand reads. An entry
// repacked at handshake h is read by GPU work that completes before handshake
// h + 1's flush returns, so it is reusable from then on.
struct ExpertPrefetch {
  struct Entry {
    int layer = -1, expert = -1;  // tag (-1: free)
    size_t off = 0;               // byte offset in `buf`
    std::atomic<int> pending{0};  // pieces still being read
    std::atomic<bool> done{false};  // ready or failed (for spin waits)
    bool loading = false;         // reads queued or running (guarded by mu)
    bool ready = false;           // read and interleaved (guarded by mu)
    bool failed = false;          // a read failed (guarded by mu)
    bool guessed = false;         // queued as a guess, not a demand read
    uint64_t stamp = 0;           // last handshake that tagged or used it
    uint64_t busy = 0;            // handshake whose GPU work reads it
  };
  struct Job {
    int fd = -1;
    uint8_t* dst = nullptr;
    size_t n = 0;
    uint64_t off = 0;
    int entry = -1;                     // prefetch entry the read fills, or -1
    std::atomic<int>* cnt = nullptr;    // decremented when the read completes (batch reads)
  };
  id<MTLBuffer> buf = nil;
  size_t entry_bytes = 0;
  // Byte offsets inside an entry: the 9 pieces (host layout), then the
  // interleaved gate/up codes, scales and biases.
  size_t o[12] = {};
  int Ie = 0, D = 0, group = 64;
  bool records = false;  // entries hold one expert record (record store), not the 9 pieces
  std::vector<std::unique_ptr<Entry>> entries;
  std::unordered_map<int64_t, int> where;  // layer * E + expert -> entry
  std::mutex mu;
  std::condition_variable cv_job, cv_ready;
  std::deque<Job> jobs;
  std::vector<std::thread> threads;
  bool stop = false;
  uint64_t hs = 1;  // handshake clock (after each flush)
  // Reads the next commit depends on: batch reads (counted) and entries whose
  // copies are encoded; the encoder waits for them, spinning, before it commits.
  std::atomic<int> outstanding{0};
  std::atomic<bool> io_failed{false};
  std::vector<int> await;  // entries (main thread only)
  // Long-pass pipeline: reads into the two staging buffers, counted
  // apart so a commit waits only for the buffer its MoE reads (st_need).
  std::atomic<int> st_pending[2];
  int st_need = -1;
  ~ExpertPrefetch() {
    {
      std::lock_guard<std::mutex> lk(mu);
      stop = true;
      jobs.clear();
    }
    cv_job.notify_all();
    for (std::thread& t : threads) t.join();
  }
};

// Token embedding table for the gather (and, when tied, the LM head).
struct Embed {
  int kind = 0;  // 0: 4-bit nibbles, 1: bytes, 2: fp16
  int group = 64;
  Buf codes, s, z;
};

struct VBlock {
  Buf n1w, n1b, n2w, n2b;
  HW qkv, proj, fc1, fc2;
};

struct Vision {
  VisionConfig c;
  HW patch, m1, m2;
  Buf pos;  // fp32 [n_pos, D]
  std::vector<VBlock> blocks;
  Buf mnw, mnb, inv_freq;
};

struct VScratch {
  int rows = 0;
  id<MTLBuffer> pix, x, xn, qkv, q, k, v, o, act, mact, out;
  id<MTLBuffer> pidx, pw, hw, seg, dst_rows;
};

// Multi-token prediction head (HF mtp.*, GGUF nextn): one full-attention
// decoder layer over fc([enorm(embed(t[i+1])) | hnorm(h[i])]) whose output
// predicts t[i+2]. It shares the embedding and LM head and keeps its own K/V
// (the last KV layer of the pool).
struct Mtp {
  bool present = false;
  Buf enorm, hnorm, norm;  // Qwen4-Exp: hnorm is the global norm over the W = hc * D residual streams
  Mat fc;       // [D] <- [2D]
  // Qwen4-Exp (docs/qwen4_exp.md 7): the streams of the MTP layer are
  // U_s = fc_emb . norm(embedding) + fc_hid . norm(H)_s (s < hc), and its
  // logits come from its own final mixer.
  Mat fc_emb, fc_hid;
  HCBlock mixer;
  Layer layer;  // full attention
  Mat head;     // 4-bit copy of the LM head for drafts (empty: the LM head itself)
  bool head4 = true;
  bool use_head4 = true;   // runtime: draft with `head` when built (else the LM head)
  // Coarse draft head: a 2-bit copy of the LM head scores the draft
  // vocabulary, the coarse_c column blocks (32 ids each) with the highest
  // coarse maxima are recomputed with the LM head, and the draft is their
  // exact argmax. Built at load when coarse_c > 0; coarse_c 0 at run time
  // falls back to `head`.
  Mat coarse;
  int coarse_c = 0;
  bool q8 = true;          // dense (bf16) MTP-layer matrices requantized to 8 bits at load
  int vocab = 0;           // drafts come from LM head rows [0, vocab)
  bool post_norm = true;   // decoder hidden rows pass the final norm first
  std::string source;
};

constexpr int kMaxDraft = 16;

// A checkpoint / sequence state. KV blocks are owned through `blocks`.
struct Slot {
  bool used = false;
  int len = 0;
  int next_pos = 0;           // next text position (M-RoPE grids advance less than tokens)
  std::vector<int> blocks;    // KV block ids covering positions [0, len)
  // MTP draft state: K/V for positions [0, len - 1) (in the MTP KV layer),
  // the decoder's final hidden row at len - 1 (mh) and at the last pass's
  // first leaf rows (lh), in the state pool's MTP region.
  bool mtp_ok = false;
  int mh_pos[3] = {0, 0, 0};
  int n_lh = 0;
  // Verify-commit: the chain rows of the slot's last pass (pass_seq
  // cm_pass), cm_k of them; their in_proj rows of every linear layer are in
  // Model::cm_x from row cm_row0, row j at position cm_pos0 + j; the DeltaNet
  // kernel and rows per simdgroup the pass ran them with.
  int cm_k = 0, cm_row0 = 0, cm_pos0 = 0, cm_R = 16;
  int cm_prow0 = 0;  // Qwen4-Exp: the pass row of chain row 0 (n-gram mix state at commit)
  bool cm_fused = false;
  uint64_t cm_pass = 0, cm_scratch = 0;  // the pass and the draft passes before it (Model::scratch_seq)
};

struct KVPool {
  id<MTLBuffer> buf = nil;
  int bt = 64;               // tokens per block
  int n_blocks = 0;
  int n_layers = 0;          // full layers (+1 with MTP)
  int kv_dim = 0;
  size_t region_halves = 0;  // halves per layer region
  std::vector<int> ref;      // reference count per block
  std::vector<int> free_list;
  size_t block_halves() const { return size_t(2) * size_t(bt) * size_t(kv_dim); }
  size_t layer_base(int l) const { return size_t(l) * region_halves; }
  // Qwen4-Exp sparse attention index, same block ids, one region per
  // decoder full layer: per block the raw indexer keys [bt][128] and the
  // pooled block keys [bt/4][128] (fp16 holding bf16 values), then the
  // tokens' M-RoPE positions [bt] (int4). kv_copy_blocks copies it as blocks
  // of 2 * bt * 84 halves.
  id<MTLBuffer> ibuf = nil;
  int i_layers = 0;
  static constexpr int kIdxHd = 128;
  size_t iblock_halves() const { return i_layers ? size_t(bt) * (kIdxHd + kIdxHd / 4 + 8) : 0; }
  size_t ipool_off() const { return size_t(bt) * kIdxHd; }
  size_t ipos_off() const { return size_t(bt) * (kIdxHd + kIdxHd / 4); }
  size_t iregion_halves() const { return size_t(n_blocks) * iblock_halves(); }
  size_t ilayer_base(int l) const { return size_t(l) * iregion_halves(); }
};

struct StatePool {
  id<MTLBuffer> buf = nil;
  int n_slots = 0;            // including the zero slot and temp slots
  int zero_slot = 0;          // all-zero state (empty context), never written
  int first_temp = 0;         // temp slots [first_temp, n_slots) for unsaved roots
  size_t conv_floats = 0;     // per slot, per layer
  size_t st_floats = 0;
  size_t mtp_floats = 0;      // per slot: (1 + kMaxDraft) * D
  std::vector<size_t> conv_base, st_base;  // float offsets per linear layer
  size_t mtp_base = 0;
  size_t total_floats = 0;
  size_t mh_off(int slot) const { return mtp_base + size_t(slot) * mtp_floats; }
  size_t lh_off(int slot, int j) const { return mh_off(slot) + size_t(1 + j) * (mtp_floats / (1 + kMaxDraft)); }
};

// Mixture-of-experts routing and expert outputs for up to `rows` rows.
struct MoeScratch {
  int rows = 0;
  id<MTLBuffer> logits, ids, w, counts, fill, offsets, tiles, ntiles, rowmap, slotpos;
  id<MTLBuffer> act, ys, shared, sgate;
  id<MTLBuffer> cparts, cctr;  // combine: per-row partial sums of squares and arrival counters
  id<MTLBuffer> rparts, rctr;  // parallel route: per-row slice partial logits and arrival counters
};

struct Scratch {
  int rows = 0, leaves = 0, cands = 0;
  id<MTLBuffer> g_xn2, g_xr;           // Gemma 4 MoE layers: the experts' and the router's normalized rows (fp16 [rows, D])
  id<MTLBuffer> tokens, h, xn, proj, qo, gb, y, yn;
  id<MTLBuffer> Q, Kb, Vb, O, act;
  id<MTLBuffer> hc, xc, mixc, actc;  // last-layer rows actually scored
  id<MTLBuffer> g32, u32;            // split gate/up outputs (fp32 [rows, I])
  id<MTLBuffer> hn, logits, lse, argmax, cand_out;
  id<MTLBuffer> tk_x, tk_i, tk_stats, tk_out;  // GPU top-k: per-part candidates, part stats, merged rows
  id<MTLBuffer> hid;                 // final-normed fp16 rows of every pass row (TreeSpec::out_hidden)
  int hid_rows = 0;
  int tk_k = 0;  // K of the last pass's top-k (0: none)
};

struct Model : Runtime {
  Config cfg;
  // Decode chains (NativeModel::run_chain): pass j of a chain reads its
  // tokens from chain_tok[(j - 1) * chain_n, j * chain_n) (the previous pass's
  // argmax) and writes its argmax to chain_tok[j * chain_n, ...); it commits
  // without waiting and records its command buffer in chain_cbs.
  bool chain_active = false;
  int chain_step = 0, chain_n = 0;
  id<MTLBuffer> chain_tok = nil;
  size_t chain_cap = 0;
  std::vector<id<MTLCommandBuffer>> chain_cbs;
  uint64_t chain_counter = 0;
  std::vector<Layer> layers;
  Embed embed;
  Mat lm_head;
  Buf final_norm, inv_freq, rope_sel;
  HCBlock mixer;     // Qwen4-Exp: the final mixer replaces final_norm
  Ple ple;
  ExpertStore xs;
  // Qwen4-Exp per-pass scratch (rows x the residual width W = hc * D) and the
  // n-gram mix state per state slot: the last 9 normalized conv inputs on the
  // GPU ([slots][9][W] fp32), the last two token ids on the host.
  struct Q4Scratch {
    int rows = 0;
    id<MTLBuffer> H, N16, dn, injr, inj, a16, mm, y, e16, kraw, vp, g, G, Gn, psrc, pslot, Hc;
    id<MTLBuffer> pfx, pflog, pfids, pfw;  // early guess of the next layer's experts
  } q4s;
  id<MTLBuffer> ple_hist = nil;
  std::vector<std::array<int32_t, 2>> ple_ctx;
  static constexpr int kPleHist = 9;
  // Expert cache: pools per layer (empty: every layer's experts are read
  // whole per pass, the test path), the LRU clock, the id -> slot map the
  // remap kernel reads, and the hook moe_mlp calls after routing.
  double expert_cache_gb = 0;  // GB of expert slots (0: whole layers per pass, the test path)
  std::vector<ExpertPool> xpool;
  std::unique_ptr<Arena> xstage;  // staging of expert reads, reused across handshakes (reset per handshake)
  std::unique_ptr<ExpertPrefetch> xpf;  // prefetch entries and reader pool, made at the first handshake
  // Admissions of a long pass: (staged record, pool slot) copies encoded after its MoE.
  std::vector<std::pair<int, int>> xadmit;
  Buf xadmit_src;
  // Split handshake: the copies of the experts this handshake reads,
  // encoded after the resident experts' tiles were committed (xsplit_cb); xld
  // flags the expert indices being read (slots loaded, staged records).
  std::vector<std::function<void(id<MTLComputeCommandEncoder>)>> xdefer;
  bool xsplit = false;  // this handshake splits its grouped GEMVs
  id<MTLBuffer> xld = nil;
  std::vector<int> xld_set;  // indices flagged in xld (cleared at the next handshake)
  id<MTLCommandBuffer> xsplit_cb = nil;    // the next commit waits for it (and for the reads)
  id<MTLCommandBuffer> xsplit_last = nil;  // its GPU time goes to xstats.gpu_ms at the next handshake
  // Handshake scratch (no allocation per handshake).
  std::vector<int> xneed, xcnt;
  std::vector<std::pair<uint64_t, int>> xcand;
  std::vector<std::pair<int, int>> xloads;
  std::vector<int> xgcnt, xgcand;  // guesses of many-row passes: rows guessing each expert, candidates
  // (layer, slot) admissions of the pass being run: their copies run in its
  // command buffers, so a pass that fails rolls them back.
  std::vector<std::pair<int, int>> xpass;
  uint64_t pass_count = 0;  // passes run (ESPETERO_FAIL_PASS)
  uint64_t pass_seq = 0;    // every run_forest pass (commit records belong to the last one)
  uint64_t scratch_seq = 0; // every draft pass: it reuses the pass arena and q4 scratch a commit reads
  // Verify-commit: the chain rows' in_proj rows of the last pass,
  // [linear layer][row][ldl] floats (cm_ld = ldl, cm_cap rows per layer).
  id<MTLBuffer> cm_x = nil;
  int cm_cap = 0, cm_ld = 0;
  std::vector<float> xmiss;  // per layer: EWMA of misses per decode-sized handshake (adaptive prefetch)
  // Long-pass pipeline (ESPETERO_XPIPE=1): two staging buffers of records;
  // while layer l's MoE runs, layer l + 1's non-resident experts are read into
  // the other one (layer[b] = the layer whose experts buffer b holds, idx[b]
  // [E] = record index of each expert or -1).
  struct XPipe {
    id<MTLBuffer> buf[2] = {nil, nil};
    size_t cap = 0;  // records per buffer
    int layer[2] = {-1, -1};
    std::vector<int> idx[2];
    int count[2] = {0, 0};
  } xpipe;
  // Decode passes (rows x k <= slots): the whole pass is one command buffer;
  // after each MoE router it signals xevA and waits for xevB, which the host
  // sets once the routed experts are resident (their repacks on xqueue).
  id<MTLSharedEvent> xevA = nil, xevB = nil;
  uint64_t xev_v = 0;
  id<MTLCommandQueue> xqueue = nil;
  uint64_t xclock = 0;
  id<MTLBuffer> xmap = nil;
  ExpertCacheStats xstats;
  std::function<const ExpertPool*(const Layer&, int)> moe_after_route;
  Vision vis;
  VScratch vs;
  Mtp mtp;
  KVPool kv;
  // Gemma 4 sliding-window layers: a ring of ring_blocks KV blocks
  // per user slot and sliding layer (position p in ring row p mod
  // ring_blocks * bt), slot s owning blocks [s * ring_blocks, (s + 1) *
  // ring_blocks); slot state like the recurrent state of the Qwen3.5 family,
  // copied on forks (no sharing). n_layers 0: no sliding layers.
  KVPool kvs;
  int ring_blocks = 0;
  Buf inv_freq_g;  // Gemma 4 global layers' RoPE frequencies ([HDg / 2], 0 past the rotated pairs)
  StatePool st;
  std::vector<Slot> slots;   // user slots [0, n_user); the pool also has zero/temp slots
  int n_user = 0;
  int max_ctx = 32768;
  int max_rows = 4096;       // rows per pass
  int max_readouts = 256;    // LM-head rows per pass
  Scratch sc;
  MoeScratch mz;
  Tuner tune;
  std::string format, path;
  std::set<std::string> dense_mats;
  int dn_rows = 0;           // 0: choose per dispatch
  int dn_chunk = 0;          // segments this long use the chunked delta rule (0: never; measured slower so far)
  bool prune_last = true;
  bool fuse_norm = true;     // decode passes: RMSNorm finalized by the producing GEMV (no norm dispatches)
  int fuse_norm_max_m = 2;   // ... for passes of at most this many rows: 3-8-row MMA GEMVs normalizing on load cost 2% on the 27B verify pass
  id<MTLBuffer> norm_ctr = nil;  // arrival counters of the fused-norm producers, one int per norm point
  bool fuse_gated = false;   // decode passes: gated per-head norm fused into the out projection (reads fp32 y and z: off)
  bool fuse_route = true;    // MoE: router + shared gate + top-k (+ single-row tile table) in one dispatch
  bool route_par = true;     // MoE: the fused route as partial-GEMV threadgroups + a finalize dispatch
  int route_nsl = 0;         // parallel route: K slices per column block (0: (K/64 + 7) / 8, at most 8)
  bool attn_vec = false;     // with attn_gqa off: decode attention of single-row trees on the GEMV-style kernel (measured neutral, off)
  bool attn_gqa = true;      // decode attention of single-row trees with the GQA group as the MMA rows (attn_decode_gqa, g4_attention_dec)
  bool route_fin1 = true;    // parallel route: barrier-free single-simdgroup finalize
  bool route_in_gemv = true; // parallel route: the finalize folded into the gate/up GEMV prologue (single row)
  int route_sync = 2;        // parallel route: 0 last arriver with fences, 1 without (diagnostic), 2 two dispatches
  bool route_notiles = false; // diagnostic only: skip the tile-table writes (stale routing, timing)
  bool route_seqexp = false;  // diagnostic only: route every row to experts 0..k-1 (timing)
  bool moe_fma = true;       // MoE: 1-2-row expert tiles use the FMA GEMV instead of the 8-row MMA one
  int moe_fma_rows = 2;      // MoE: largest pass (rows) whose expert tiles use the FMA GEMV (1-2; 3-4 measured slower)
  bool moe_down_comb = true; // single-row MoE: down projection and combine in one kernel
  int moe_comb_ks = 1;       // ... K slices per slot (1 or 2; 1 measured faster)
  int moe_ks = 4;            // FMA expert GEMV: simdgroups per threadgroup (K slices): 2, 4 or 8 (4 for gate/up)
  bool moe_direct = true;    // single-row passes: direct grids for the expert GEMVs (no indirect arguments)
  int moe_nbpt = 1;          // FMA expert GEMV: 32-column blocks per threadgroup (direct grids only)
  int moe_ks_dn = 4;         // as moe_ks / moe_nbpt, for the down projection (4 slices x 2 column blocks)
  int moe_nbpt_dn = 2;
  bool moe_swap = false;     // FMA expert GEMV: block-major grid (consecutive threadgroups stream one expert)
  int moe_stages = 15;       // diagnostic (bench_json "moe"): MoE layer stages encoded: 1 route, 2 gate/up, 4 down, 8 combine
  bool fuse_shared = true;   // MoE: the shared expert runs as one more tile of the grouped GEMVs
  bool fuse_gdn = true;      // decode passes: conv + gates + recurrence + gated norm in one dispatch per segment set
  bool gdn_conv_fold = true; // leaf-free fused passes also save the conv state (no conv_state_save dispatch)
  id<MTLBuffer> gdn_ctr = nil;  // arrival counters [segment][key head] (self-resetting)
  static constexpr int kGdnCtrSegs = 256;
  // A submission failed (or was refused) since the arrival counters
  // were last known to be zero: the next pass zeroes them first.
  bool ctr_dirty = false;
  bool all_quant = true;     // every decoder matrix quantized (fusion needs the GEMV kernels)
  int attn_split = 512;      // cached keys per split-K tile (0: never split)
  bool qsa = true;           // Qwen4-Exp: sparse attention past 2051 visible tokens (false: dense, the approximation before)
  int n_q4 = 0, n_q5 = 0, n_q6 = 0, n_q8 = 0, n_dense = 0;
  int n_dense_mtp = 0;       // dense matrices that never take a fused-norm link (the MTP layer's, MoE routers and shared gates)
  bool split_mlp = false;
  std::map<std::string, int> source_types;
  size_t weight_bytes = 0;

  // ---------------------------------------------------------------- KV
  int kv_free() const { return int(kv.free_list.size()); }
  int kv_alloc() {
    if (kv.free_list.empty()) fail("KV cache full (raise kv_bytes or lower concurrency)");
    const int b = kv.free_list.back();
    kv.free_list.pop_back();
    kv.ref[size_t(b)] = 1;
    return b;
  }
  void kv_retain(int b) { ++kv.ref[size_t(b)]; }
  void kv_release(int b) {
    if (--kv.ref[size_t(b)] == 0) kv.free_list.push_back(b);
  }
  void release_blocks(Slot& s) {
    for (int b : s.blocks) kv_release(b);
    s.blocks.clear();
  }
};

}  // namespace espetero::native
