// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The serving engine: one loaded model, one scheduler thread that owns the
// GPU, and three kinds of work sharing every pass:
//
//   * decisions (JEV-style structured scoring): one tree per request — the
//     uncached prompt prefix as the root, the field suffixes as leaves;
//   * generation sequences (chat, completions, batch jobs): chunked prefill,
//     then one tree per decode step (pending tokens as the root, MTP or
//     prompt-lookup drafts as a chain of leaves);
//   * everything of one scheduler step runs as ONE forest pass, so decode
//     steps of many sequences, decision requests and prefill chunks share
//     their weight reads (continuous batching).
//
// Prompt prefixes are checkpointed in an exact-token LRU of model slots
// (KV blocks shared by reference, recurrent state copied into the slot).
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../model/api.h"
#include "../util/json.h"
#include "model_meta.h"
#include "sampling.h"

namespace espetero {

class Tokenizer;
namespace tmpl {
class Template;
}
namespace clef {
class Head;
}
namespace vision {
struct Spec;
}
namespace grammar {
struct Constraint;
class Vocab;
}

// Errors carrying an HTTP status (422 invalid request, 503 overloaded, 502 backend).
struct RequestError : std::runtime_error {
  int status;
  RequestError(int s, const std::string& msg) : std::runtime_error(msg), status(s) {}
};

struct EngineOptions {
  std::string model_path;
  std::string model_name;          // reported in responses (default: last path component)
  std::string model_revision;      // a Hugging Face snapshot's revision (repo ids): part of the --prefix-dir key
  LoadOptions load;
  bool generation_config = true;   // requests that omit temperature / top_p / top_k / min_p get the model's
                                   // generation_config.json (GGUF: general.sampling.*) values (--no-generation-config: OpenAI's)
  int max_seqs = 64;               // concurrent generation sequences
  int prefill_chunk = 2048;        // prefill rows per pass while other sequences decode
  bool image_checkpoint = true;    // chats: a checkpoint after the last image, shared by questions about it
  bool instr_checkpoint = true;    // chats: a checkpoint after the user's instructions before the first image
  bool single_flight = true;       // a prompt waits for a shared prefix another request is computing
  bool keep_live = true;           // checkpoints running sequences continue or saved are evicted last, made recent when they end
  bool prompt_checkpoint = true;   // chats: a checkpoint at the prompt end for later turns (requests override with cache_prompt)
  bool decode_first = true;        // prompts wait (bounded) for decoding sequences; big prompts one per pass
  int prefill_wait_ms = 2500;      // ... longest wait of a prompt chunk while sequences decode
  int prefill_batch_rows = 512;    // ... prompt rows per pass beyond which no further prompt joins
  int draft_tokens = -1;           // MTP drafts per decode pass (0: plain decoding; -1: the model's default,
                                   // else 5; Qwen4-Exp 3)
  int sibling_drafts = 0;          // the MTP head's runner-ups as extra leaves (1: depth 1, 2: depths 1-2; greedy/exact only)
  bool draft_auto = false;         // MTP draft depth per pass from the request's per-depth acceptance and
                                   //      the measured verify/draft costs (single decoding sequence; up to draft_max)
  int draft_max = 7;
  int spec_max_seqs = 2;           // speculate only while at most this many sequences decode
  bool prompt_lookup = true;       // n-gram drafts from the context
  int lookup_min = 8;              // ... only copies backed by this many matching tokens
  bool adaptive_drafts = true;     // draft only while it beats plain decoding (single decoding sequence)
  int chain_steps = 8;             // greedy decode steps per GPU chain when no drafts are in play (0/1: off)
  int chain_after = 16;            // ... once a sequence has generated this many tokens (short answers never wait on a chain)
  bool gpu_topk = true;            // sampled requests with 1 <= top_k <= 32: candidates from the GPU
  bool exact_sampling = true;      // position-keyed draws, drafts verified by equality (--legacy-sampling: mt19937 + random test)
  bool gpu_draw = true;            // eligible sampled requests draw on the GPU and can chain (--no-gpu-draw)
  // Image budgets: the "standard" detail (~1 MP: a good balance of detail and
  // time) and "max", the processor's
  // own ceiling for Qwen-VL towers (mlx-vlm's default budget). Per request,
  // detail low = 512x512, high = 4x standard (at most max), max = image_detail_max.
  int64_t image_max_pixels = 1003520;
  int64_t image_detail_max = 16777216;
  // A request may ask for a detail above the server's own budget (high, max:
  // up to 16K image tokens and minutes of vision tower per image) only when the
  // operator allows it (--allow-image-detail-max); otherwise 422.
  bool allow_image_detail_max = false;
  // Http(s) image URLs in chat, Responses and Anthropic requests are
  // fetched by the server (--image-urls; off: a 422), never from loopback,
  // private or link-local addresses unless --image-urls-private (vision/fetch.h).
  bool image_urls = false;
  bool image_urls_private = false;
  // /v1/embeddings pools each input's final-normed hidden states:
  // "last" (the last token's, as Qwen3-Embedding) or "mean" (--embedding-pooling).
  std::string embedding_pooling = "last";
  int decision_batch = 8;          // decision requests per pass
  int decision_rows = 2048;        // decision rows per pass
  bool share_prefix = true;        // decisions: cached prefixes cost no rows; same-prefix jobs wait for the one computing it
  size_t memo = 0;                 // exact-repeat decision memo entries
  bool verbose = false;
  bool profile = false;            // accumulate per-kernel GPU times (GET /v1/profile)
  std::string prefix_dir;          // persist chat checkpoints here (empty: off)
  double prefix_dir_gb = 20;       // ... at most this many GB on disk (least recently used files go first)
};

class GenStream;

// Images prepared for the model (preprocessed pixels and their placement).
struct PreparedImage {
  std::vector<float> pixels;
  int grid_t = 1, grid_h = 0, grid_w = 0;
  std::string digest;  // content hash (hex)
  int tokens(int merge) const { return grid_t * grid_h * grid_w / (merge * merge); }
};

// A generation request in tokens.
struct GenRequest {
  std::vector<int32_t> prompt;          // image placeholders already expanded
  std::vector<int32_t> positions;       // [prompt][3] M-RoPE positions (images), or empty
  std::vector<PreparedImage> images;    // in prompt order
  std::vector<int> image_rows;          // first prompt row of each image
  int split = -1;                       // checkpoint boundary (chat: after the assistant header); -1: none
  int sys_split = -1;                   // second boundary: end of the system message (shared across conversations); -1: none
  int img_split = -1;                   // third boundary: right after the last image run (shared by questions about the same images); -1: none
  int pre_split = -1;                   // fourth boundary: end of the user's own text before the first image (instructions shared by a set of photos); -1: none
  uint64_t sys_hash = 0, pre_hash = 0;  // hashes of the prompt up to sys_split / pre_split (single-flight prefixes)
  int cache_prompt = -1;                // save the prompt-end checkpoint: 1 yes, 0 no (a flow never re-sends the prompt), -1 the server's default
  SamplingParams sampling;
  int max_tokens = 512;
  std::vector<int32_t> stop_ids;        // besides the model's end tokens
  bool ignore_eos = false;
  uint64_t seed = 0;
  bool has_seed = false;
  int priority = 0;                     // lower runs first (batch jobs: 1)
  bool speculate = true;
  // Teacher-forced scoring: these tokens are produced instead of the
  // model's choices, and each position records the log-probability of the
  // forced token, the model's argmax and its top `score_topk` alternatives
  // (returned in the stream's timings as "score").
  std::vector<int32_t> forced;
  int score_topk = 0;
  // Per scored position, token ids whose logprobs are returned too (e.g.
  // another model's top-k, to compare the two distributions).
  std::vector<std::vector<int32_t>> score_ids;
  // >= 0: each generated token's log-probability and its top `logprobs`
  // alternatives (OpenAI logprobs / top_logprobs, <= 20), read from the GPU
  // top-k of its readout (sampled requests too): the model's
  // own distribution, log-softmax of the raw logits (temperature 1, before
  // penalties, top-k, top-p and min-p), as vLLM's default raw logprobs.
  int logprobs = -1;
  // The other choices of an n > 1 request (Engine::generate_n): their
  // streams and seeds. They start from this request's prompt, computed once.
  std::vector<std::shared_ptr<GenStream>> group_out;
  std::vector<uint64_t> group_seeds;
  // Structured output: every token is chosen under this grammar's mask
  // (null: unconstrained). Such sequences pick their tokens on the CPU and run
  // no decode chains; their drafts are verified against the masked choice.
  std::shared_ptr<const grammar::Constraint> constraint;
};

// A generated token's log-probability and the most likely tokens at its
// position.
struct TokenLogprob {
  int32_t token = -1;
  float logprob = 0;
  std::vector<std::pair<int32_t, float>> top;
};

// Tokens of a running generation, filled by the scheduler thread and read
// by the caller's thread.
class GenStream {
 public:
  // Wait for new tokens (appended to `out`) or the end; returns false once
  // finished and fully drained. With GenRequest::logprobs >= 0, `lps` gets
  // the tokens' records, one per token in the same order (they travel
  // with the tokens, so a stream can attach them to its chunks).
  bool next(std::vector<int32_t>& out, int timeout_ms = -1, std::vector<TokenLogprob>* lps = nullptr);
  void cancel() { cancelled_.store(true); }
  bool cancelled() const { return cancelled_.load(); }
  bool finished() const;
  // Final state (valid once next() returned false).
  std::string finish_reason;   // stop | length | cancelled | error
  std::string error;
  int prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0;
  json::Value timings;

  // Scheduler side.
  void push(const int32_t* t, size_t n, std::vector<TokenLogprob>* lps = nullptr);
  void finish(const std::string& reason, const std::string& err = "");
  // Streams finished with this one while they stay linked (the other
  // choices of an n > 1 request, until its prompt is forked to them).
  void link(std::shared_ptr<GenStream> s);
  std::vector<std::shared_ptr<GenStream>> unlink();

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<int32_t> buf_;
  std::vector<TokenLogprob> lp_buf_;
  std::vector<std::shared_ptr<GenStream>> linked_;
  bool done_ = false;
  std::atomic<bool> cancelled_{false};
};

class Engine {
 public:
  explicit Engine(const EngineOptions& options);
  ~Engine();

  const ModelInfo& info() const { return model_->info(); }
  const std::string& model_name() const { return name_; }
  const Tokenizer& tokenizer() const { return *tok_; }
  const tmpl::Template* chat_template() const { return template_.get(); }
  const EngineOptions& options() const { return opt_; }
  // The model's sampling defaults for requests that omit them (none with --no-generation-config).
  const SamplingDefaults& sampling_defaults() const { return sampling_defaults_; }
  // Pixel budget for an image's detail level: "auto" / "standard" -> 0 (the
  // default budget), "low", "high", "max"; -1 for an unknown level.
  int64_t detail_pixels(const std::string& detail) const {
    if (detail.empty() || detail == "auto" || detail == "standard") return 0;
    if (detail == "low") return 262144;
    if (detail == "high") return std::min(opt_.image_max_pixels * 4, std::max(opt_.image_max_pixels, opt_.image_detail_max));
    if (detail == "max") return opt_.image_detail_max;
    return -1;
  }
  // The largest budget a request may ask for: the server's own (and "low"), or
  // every level with --allow-image-detail-max.
  int64_t detail_ceiling() const {
    return std::max(opt_.allow_image_detail_max ? opt_.image_detail_max : int64_t(262144), opt_.image_max_pixels);
  }
  // 422 naming --allow-image-detail-max when a detail_pixels() budget is above
  // `ceiling` (0: detail_ceiling()). `field` and `detail` word the message.
  void check_detail(int64_t pixels, const std::string& field, const std::string& detail, int64_t ceiling = 0) const;
  // Prompt text past 16 bytes per token of the context window cannot fit it
  // (real text averages 3 to 5): 413 before it is rendered or tokenized.
  void check_text_size(size_t bytes, const std::string& what) const;
  bool has_vision() const { return model_->info().vision; }
  // Image token ids (vision models).
  int32_t image_token = -1, vision_start = -1, vision_end = -1;
  std::vector<int32_t> end_ids;  // tokens that end a generation

  // ---- decisions (blocking; thread-safe)
  json::Value decide(const json::Value& request);

  // ---- Clef joint-head decisions (blocking, thread-safe): a Jev / SystemOne
  // /v1/systemone body (state, questions, images) -> answers. Only for Clef releases
  // (joint_head.safetensors next to the backbone); such models serve no chat.
  json::Value systemone(const json::Value& request);
  bool is_clef() const { return clef_ != nullptr; }
  // The JEVfire decision contract (context + boolean / enum fields) answered by a
  // Clef head: fields become noul / choice questions; the response keeps decide()'s format.
  json::Value clef_decide(const json::Value& request);

  // ---- generation (thread-safe): the stream starts filling immediately
  std::shared_ptr<GenStream> generate(GenRequest request);
  // N generations of one prompt (OpenAI's n), its prefill computed once:
  // the first stream's sequence computes the prompt, its final readout gives
  // every choice its first token, and the others continue from forks of its
  // state. Choice i draws with seed choice_seed(seed, i) (a request without a
  // seed: random ones); greedy choices are identical.
  std::vector<std::shared_ptr<GenStream>> generate_n(GenRequest request, int n);
  static uint64_t choice_seed(uint64_t seed, int i);

  // The model's vocabulary as structured output reads it (a byte trie of
  // the tokens), built at the first constrained request. Thread-safe.
  std::shared_ptr<const grammar::Vocab> grammar_vocab() const;

  // ---- embeddings (blocking, thread-safe): each input's final-normed
  // hidden states (the LM head's input, no LM head run) pooled as
  // EngineOptions::embedding_pooling says and L2-normalized; inputs share
  // passes when they fit, a long one spans several (its state in a slot).
  struct EmbedStats {
    int passes = 0, rows = 0;
    double gpu_ms = 0, queue_ms = 0, run_ms = 0;
  };
  std::vector<std::vector<float>> embed(std::vector<std::vector<int32_t>> inputs, EmbedStats* stats = nullptr);

  // ---- images
  // min_pixels > 0 overrides the processor's floor (min(65536, max)); Clef requests pass the release's own limits.
  // At most 4 images are decoded and preprocessed at once across request threads.
  PreparedImage prepare_image(const std::string& data_url, int64_t max_pixels = 0, int64_t min_pixels = 0) const;
  // The tokens prepare_image will give an image at this budget, from the file's
  // header alone (no pixel decoded): 422 for a bad data URL, an image over the
  // decode cap or an aspect ratio the processor rejects; -1 when the header
  // cannot be read (decoding then decides).
  int image_tokens(const std::string& data_url, int64_t max_pixels = 0, int64_t min_pixels = 0) const;
  // Length of `ids` once each image placeholder becomes its run of `runs[i]`
  // tokens (an unknown run, -1, counts as its placeholder); 422 like
  // expand_images when placeholders and images differ in number.
  size_t expanded_length(const std::vector<int32_t>& ids, const std::vector<int>& runs) const;
  // M-RoPE positions of a token list with expanded image runs.
  std::vector<int32_t> mrope_positions(const std::vector<int32_t>& ids, const std::vector<PreparedImage>& images,
                                       int* next_pos = nullptr) const;
  // Replace each image placeholder token by the image's token run.
  std::vector<int32_t> expand_images(const std::vector<int32_t>& ids, const std::vector<PreparedImage>& images) const;

  json::Value stats() const;
  void clear_cache();
  // Stops taking requests and ends what is queued or running: the
  // scheduler finishes its current pass, then fails every waiting and active
  // sequence and queued decision ("server shutting down"), so callers blocked
  // in their streams return at once instead of after their generations.
  // Thread-safe; the destructor joins the scheduler.
  void shutdown();
  // Per-kernel GPU times since load (--profile), as the model reports them.
  std::string profile_json() { return model_->profile_json(); }

 private:
  struct Seq;
  struct DecisionJob;
  struct ClefJob;
  void run_clef(ClefJob& job);
  struct EmbedJob;  // embed.cpp
  std::deque<std::shared_ptr<EmbedJob>> embed_jobs_;
  std::vector<float> embed_hidden_;  // a pass's hidden rows
  void embed_step();                                 // scheduler: one pass of the oldest embedding job
  void embed_fail_all(const std::string& why);       // scheduler: at shutdown
  // The processor geometry and token ids of an image budget (prepare_image, image_tokens).
  vision::Spec image_spec(int64_t max_pixels, int64_t min_pixels) const;
  struct CacheEntry {
    std::string key;           // namespace + salt + image digests + tokens
    std::vector<int32_t> tokens;
    std::vector<std::string> digests;
    std::string ns;            // "dec", "chat", "hdr"
    int slot = -1;
    bool mtp = false;
    uint64_t used = 0;
    int next_pos = 0;
    std::string parent;        // the checkpoint this one extends ("" for a root)
  };

  void loop();
  void warmup();
  void calibrate_drafts();  // draft and verify costs at load
  void step();
  int choose_depth(const Seq& s, int budget) const;
  // A token's raw log-probability and top alternatives at a readout of the last pass.
  TokenLogprob readout_logprob(int tree, int readout, int32_t tok, int n_top, const float* lse) const;
  // The sampling distribution of a readout of the last pass for these parameters and penalties.
  Distribution readout_dist(int tree, int readout, const SamplingParams& sp, const PenaltyState* pen) const;
  // An n > 1 request's leader just computed its prompt: its other choices start from it
  // (appended to the pass's sequences, tokens and logprob records).
  void fork_group(Seq& leader, std::vector<Seq*>& touched, std::vector<std::vector<int32_t>>& produced,
                  std::vector<std::vector<TokenLogprob>>& produced_lp);
  double mtp_accept(const Seq& s, int d) const;      // acceptance at depth d (Beta prior)
  double mtp_expect(const Seq& s, int k) const;      // expected tokens of a pass with k MTP drafts
  // Slots: a free model slot, evicting unpinned cache entries (LRU) if needed.
  int acquire_slot(const std::vector<int>& avoid);
  // `live_ok`: also checkpoints running sequences continue or saved (last);
  // false stops before them (admission and decode chains).
  int evict_lru(const std::vector<int>& avoid, bool live_ok = true);
  // The KV tokens that evicting every checkpoint evict_lru(avoid,
  // live_ok) may take would give back (blocks no other slot holds), so that
  // callers evict only when that makes their work fit.
  int evictable_kv_tokens(const std::vector<int>& avoid, bool live_ok) const;
  // KV tokens the active sequences may still take: their prompt rows
  // not computed yet and their answers up to max_tokens, with slack.
  int reserved_kv_tokens() const;
  // The part of an answer of `remaining` tokens that admission reserves: all
  // of it up to max(8K tokens, a quarter of the KV cache).
  int answer_reserve(int remaining) const;
  // Chat checkpoint keys name the MTP state (a draft-less checkpoint
  // cannot serve a request that drafts, and the two may coexist).
  static std::string chat_key(const std::vector<std::string>& digests, const std::vector<int32_t>& tokens, bool mtp);
  CacheEntry* cache_find(const std::string& key);
  // Disk-persisted checkpoints (text-only chat prefixes): an index of the
  // stored token vectors, written after a checkpoint becomes valid, read on
  // a cache miss for the longest stored strict prefix of a prompt.
  struct StoredPrefix {
    std::vector<int32_t> tokens;
    std::string path;  // the .ckpt file (the .tok file beside it holds the tokens)
    bool mtp = false;
    uint64_t bytes = 0;
    int64_t used = 0;  // last write or restore (file mtime, nanoseconds)
  };
  std::vector<StoredPrefix> stored_;
  std::string prefix_root_;
  double other_roots_bytes_ = 0;  // checkpoints of other builds / models kept in prefix_dir
  void prefix_store_init();
  void prefix_store_trim_roots();
  void prefix_store_trim();
  void prefix_store_put(const CacheEntry& e);
  CacheEntry* prefix_store_restore(const std::vector<int32_t>& prompt, size_t longer_than, size_t limit, bool want_mtp);
  void cache_put(const std::string& key, CacheEntry e);
  void cache_drop(const std::string& key);
  std::string decision_key(const std::string& salt, const std::vector<int32_t>& prefix, const std::string& digest) const;

  EngineOptions opt_;
  std::string name_;
  SamplingDefaults sampling_defaults_;
  std::unique_ptr<NativeModel> model_;
  std::unique_ptr<Tokenizer> tok_;
  std::shared_ptr<const tmpl::Template> template_;
  struct DecisionPrompts;
  std::unique_ptr<DecisionPrompts> dp_;
  mutable std::mutex grammar_mu_;
  mutable std::shared_ptr<const grammar::Vocab> grammar_vocab_;

  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::deque<std::shared_ptr<DecisionJob>> decisions_;
  std::deque<std::shared_ptr<ClefJob>> clef_jobs_;
  std::unique_ptr<clef::Head> clef_;
  // Clef image prefixes (scheduler thread): the system prompt and the media
  // block of recent requests with images, as a checkpoint slot plus their
  // final-normed hidden rows, so the next request about the same images
  // starts after them.
  struct ClefPrefix {
    std::vector<int32_t> tokens;
    std::string digest;
    int slot = -1;
    std::vector<float> hidden;
    uint64_t used = 0;
  };
  std::vector<ClefPrefix> clef_prefixes_;
  size_t clef_prefix_cap_ = 4;
  // The release's image processor limits (processor_config.json size.shortest_edge / longest_edge),
  // the default image budget of Clef requests (0: unknown, the engine's own default).
  int64_t clef_min_pixels_ = 0, clef_max_pixels_ = 0;
  // Frees the least recently used Clef prefix other than those in `keep`; false when none is left.
  bool clef_evict_prefix(const std::vector<int>& keep);
  std::deque<std::shared_ptr<Seq>> waiting_;
  std::vector<std::shared_ptr<Seq>> active_;
  std::thread worker_;

  // Scheduler-thread state.
  std::unordered_map<std::string, CacheEntry> cache_;
  std::vector<int> pinned_;  // slots never evicted (header, running sequences)
  uint64_t clock_ = 0;
  int header_slot_ = -1;
  std::vector<int32_t> header_;
  std::mutex memo_mu_;
  std::unordered_map<std::string, json::Value> memo_;
  std::deque<std::string> memo_order_;

  // Counters.
  std::atomic<bool> clear_requested_{false};
  std::atomic<uint64_t> passes_{0}, rows_{0}, gen_tokens_{0}, decisions_done_{0};
  std::atomic<uint64_t> live_evictions_{0};  // checkpoints evicted while a running sequence still used them
  // Sequences stopped because the KV cache could not hold every running
  // sequence (the newest go first), and prompt chunks deferred for it.
  std::atomic<uint64_t> kv_stopped_{0}, kv_deferred_{0};
  // Passes whose newest sequences' drafts or decode steps waited so that
  // a decision job that had waited for rows could run.
  std::atomic<uint64_t> decision_reservations_{0};
  // Optional checkpoints (prompt boundaries, decision prefixes) a pass
  // short of KV skipped so that its work could run.
  std::atomic<uint64_t> kv_skipped_{0};
  // Model counters published by the scheduler after every step:
  // stats() runs on HTTP threads and never calls the model, which is not
  // thread-safe.
  std::atomic<int> free_slots_pub_{0}, free_kv_pub_{0}, active_pub_{0}, waiting_pub_{0};
  void publish_counters();
  bool trace_evict_ = false;                 // ESPETERO_TRACE_EVICT=1 logs every checkpoint eviction
  // Expert-cache counters (models that stream experts), snapshot by the scheduler
  // after every pass for GET /v1/stats: stats() never touches the model.
  mutable std::mutex xsnap_mu_;
  std::string xsnap_;
  bool xstream_ = false;
  std::atomic<double> gpu_ms_{0};
  // Adaptive drafting: wall-time EWMAs of a plain decode step (per
  // token, chains included), a draft pass, and verify passes by draft count,
  // measured on passes with one decoding sequence and nothing else.
  struct DraftCost {
    double plain = 0, draft = 0, verify[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    int n_plain = 0, n_draft = 0, n_verify[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    double draftk[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // draft pass by drafted count
    int n_draftk[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    // MTP drafts' conditional acceptance by depth over all requests: a
    // running mean (window 500), the depth controller's prior.
    double acc_depth[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int n_acc_depth[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // Load-time costs (calibrate_drafts): plain step, verify and draft
    // passes by draft count, best of 3; the MTP head's verdicts use their ratios.
    bool calibrated = false;
    double cal_plain = 0, cal_verify[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, cal_draft[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    // Verify-commit: the commit after a verify pass (its own command
    // buffer), an EWMA seeded at load; part of the cost of drafting.
    double commit = 0;
    int n_commit = 0;
  } dcost_;
};

}  // namespace espetero
