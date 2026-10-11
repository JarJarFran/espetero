// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "engine.h"

#include <dirent.h>
#include <fstream>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <ctime>

#include <dispatch/dispatch.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

#include "../tmpl/tmpl.h"
#include "../tok/tokenizer.h"
#include "../util/paths.h"
#include "../vision/vision.h"
#include "clef.h"
#include "decisions.h"
#include "grammar.h"

namespace espetero {

using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

double round3(double x) { return std::round(x * 1000.0) / 1000.0; }

std::string read_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return "";
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool is_dir(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string hex(const uint8_t* d, size_t n) {
  static const char* h = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    s.push_back(h[d[i] >> 4]);
    s.push_back(h[d[i] & 15]);
  }
  return s;
}

std::string tokens_key(const std::vector<int32_t>& t, size_t n) {
  return std::string(reinterpret_cast<const char*>(t.data()), n * sizeof(int32_t));
}

template <class F>
void parallel(size_t n, F&& f) {
  if (n <= 1) {
    for (size_t i = 0; i < n; ++i) f(i);
    return;
  }
  struct Ctx {
    F* f;
  } ctx{&f};
  dispatch_apply_f(n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &ctx,
                   [](void* c, size_t i) { (*static_cast<Ctx*>(c)->f)(i); });
}

}  // namespace

// ---------------------------------------------------------------- GenStream
bool GenStream::next(std::vector<int32_t>& out, int timeout_ms, std::vector<TokenLogprob>* lps) {
  std::unique_lock<std::mutex> lk(mu_);
  auto ready = [&] { return !buf_.empty() || done_; };
  if (timeout_ms < 0) cv_.wait(lk, ready);
  else cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
  if (!buf_.empty()) {
    out.insert(out.end(), buf_.begin(), buf_.end());
    buf_.clear();
    if (lps) lps->insert(lps->end(), std::make_move_iterator(lp_buf_.begin()), std::make_move_iterator(lp_buf_.end()));
    lp_buf_.clear();
    return true;
  }
  return !done_;
}

bool GenStream::finished() const {
  std::lock_guard<std::mutex> lk(mu_);
  return done_ && buf_.empty();
}

void GenStream::push(const int32_t* t, size_t n, std::vector<TokenLogprob>* lps) {
  if (!n) return;
  {
    std::lock_guard<std::mutex> lk(mu_);
    buf_.insert(buf_.end(), t, t + n);
    if (lps) lp_buf_.insert(lp_buf_.end(), std::make_move_iterator(lps->begin()), std::make_move_iterator(lps->end()));
  }
  cv_.notify_all();
}

void GenStream::finish(const std::string& reason, const std::string& err) {
  std::vector<std::shared_ptr<GenStream>> linked;
  {
    std::lock_guard<std::mutex> lk(mu_);
    finish_reason = reason;
    error = err;
    done_ = true;
    linked.swap(linked_);
  }
  cv_.notify_all();
  for (auto& s : linked) s->finish(reason, err);  // choices not forked yet end with their leader
}

void GenStream::link(std::shared_ptr<GenStream> s) {
  std::lock_guard<std::mutex> lk(mu_);
  linked_.push_back(std::move(s));
}

std::vector<std::shared_ptr<GenStream>> GenStream::unlink() {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<std::shared_ptr<GenStream>> out;
  out.swap(linked_);
  return out;
}

// ---------------------------------------------------------------- internals
struct Engine::DecisionPrompts {
  std::vector<Label> labels;
  std::unique_ptr<DecisionPromptBuilder> builder;
};

struct Engine::DecisionJob {
  DecisionRequest req;
  DecisionPlan plan;
  std::vector<PreparedImage> images;
  std::vector<int32_t> positions;  // [prefix + stem][3] (images only)
  std::string key;
  std::string digest;
  Clock::time_point started;
  // Pass arrays.
  std::vector<int32_t> root, leaf_off, leaf_tok, cand_off, cand_ids, root_pos;
  std::vector<ImageRef> irefs;
  // Results.
  std::vector<float> cand_logits, lse;
  PassStats stats;
  std::string cache_state;
  int cached = 0, batched = 0, rows = 0;
  int waits = 0;  // Passes it was put back because the decode rows left too few for its tree
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  std::string error;
  int error_status = 502;
};

struct Engine::ClefJob {
  clef::Record rec;
  std::vector<PreparedImage> images;
  std::string digest;              // image digests (the prefix cache key with the prefix tokens)
  int cached = 0;                  // prefix rows served from the cache
  std::vector<int32_t> positions;  // [ids][3] M-RoPE positions (images only)
  std::vector<float> hidden;       // [ids][hidden] final-normed backbone states
  // Pass arrays.
  std::vector<int32_t> root_pos;
  std::vector<ImageRef> irefs;
  double gpu_ms = 0, run_ms = 0;
  int passes = 0;
  Clock::time_point queued, started;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  std::string error;
  int error_status = 502;
};

struct Engine::Seq {
  GenRequest req;
  std::shared_ptr<GenStream> out;
  enum Phase { Prefill, Decode } phase = Prefill;
  int work = -1;          // generation state
  int ckpt = -1;          // prompt checkpoint slot (cached for later turns)
  int sysckpt = -1;       // system-prompt checkpoint slot (shared across conversations)
  int imgckpt = -1;       // image checkpoint slot (shared by questions about the same images)
  int prectk = -1;        // instruction checkpoint slot (the user's text before the first image)
  std::string ckpt_key;
  int parent = -1;        // cached checkpoint the first prefill chunk continues
  // Cache keys of the checkpoints it continues or saved: evicted last while it
  // runs, made recent when it ends.
  std::vector<std::string> keep;
  std::string chain_key;  // the latest checkpoint on its path (restored or saved): its next one's parent
  int cached = 0;         // prompt tokens covered by it
  int done = 0;           // prompt tokens already in the work slot
  bool mtp = false;       // keep MTP draft state
  bool saved_ckpt = false;
  bool saved_sys = false;
  bool saved_img = false;
  bool saved_pre = false;
  std::vector<int32_t> pending;  // decode: tokens the next pass appends
  std::vector<int32_t> drafts;
  bool from_lookup = false;
  // Sibling drafts: each MTP step's runner-up, the sibling leaves of
  // this pass (1: [s1]; 2: also [d1, s2]), the saved leaf row the next draft
  // pass starts from (the accepted leaf's first row), and how often a miss at
  // depth d equalled the runner-up (whether or not its leaf was in the pass).
  std::vector<int32_t> drafts2;
  int sib = 0, lh_first = 0, sib_taken = 0;
  int commit_k = 0;  // the drafts of this pass form a commit chain (accepted ones become slot rows)
  int sib_opp[3] = {0, 0, 0}, sib_hit[3] = {0, 0, 0};
  // Per-depth draft statistics, [0] MTP drafts, [1] prompt lookup:
  // verify passes that reached depth d (every earlier draft accepted), and
  // those that accepted the draft there.
  int dep_reach[2][8] = {}, dep_acc[2][8] = {};
  int dep_k[9] = {};  // verify passes by MTP draft count
  int generated = 0;
  // Teacher-forced scoring: one record per scored position.
  struct ScoreRec {
    int32_t target = 0, argmax = 0;
    float logprob = 0;
    std::vector<std::pair<int32_t, float>> top;
    std::vector<float> ids_lp;  // logprobs of the request's ids for this position
  };
  std::vector<ScoreRec> scored;
  PenaltyState pen;
  std::mt19937_64 rng;
  uint64_t seed = 0;  // exact sampling: the request's seed, else a random 64-bit one
  // Adaptive drafting, per draft source: prompt lookup keeps its
  // accepted drafts per verify pass (the mean of the first 4 passes, then an
  // EWMA), the passes seen since the last decision and plain steps left
  // before it is tried again. The MTP head is judged on its per-depth
  // acceptance (mtp_expect) and never stops for good: a loss buys a plain
  // stretch of mtp_backoff steps (8, doubling to 128, back to 8 after a win),
  // then one probe pass decides again; lost_mtp counts the losses. Plain
  // stretches keep the head's state (no decode chains, which do not run it),
  // so only kMtpGiveUp losses in a row (120 plain steps of evidence) stop the
  // head for the request and let chains run.
  double acc_look = 0;
  int n_look = 0, skip_mtp = 0, skip_look = 0, lost_mtp = 0, mtp_backoff = 8, mtp_lost_run = 0;
  NgramIndex ngram;
  Clock::time_point t_start, t_first, t_decode;
  Clock::time_point t_wait;  // decode-first: since when the prompt's next chunk has been waiting
  int passes = 0, drafted = 0, accepted = 0, lookups = 0;
  int retries = 0;  // first prompt chunks run again after a GPU abort
  double gpu_ms = 0;
  // Per-pass arrays.
  std::vector<int32_t> root, leaf_off, leaf_tok, arg, root_pos;
  std::vector<float> lse;
  std::vector<float> lp_lse;  // raw log-normalizer of each readout (logprobs of GPU-sampled requests)
  std::vector<ImageRef> irefs;
  int tree = -1;
  int chunk_end = 0;
  bool final_chunk = false;
  // Structured output: the parse state of the reply (made at its first
  // token), and a token the grammar refused (an internal error: the sequence fails).
  std::unique_ptr<grammar::Matcher> gm;
  bool gm_error = false;
};

// ---------------------------------------------------------------- setup
namespace {
// ESPETERO_TRACE_LOAD=1: the time of each load phase on stderr (where a slow
// start goes: tokenizer, weights, kernels, pools, warm-up, draft costs).
struct LoadTrace {
  bool on = [] {
    const char* v = std::getenv("ESPETERO_TRACE_LOAD");
    return v && *v == '1';
  }();
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now(), last = t0;
  void mark(const char* what) {
    if (!on) return;
    const auto now = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[espetero] load: %-24s %8.0f ms (at %.2f s)\n", what,
                 std::chrono::duration<double, std::milli>(now - last).count(),
                 std::chrono::duration<double>(now - t0).count());
    last = now;
  }
};
}  // namespace

Engine::Engine(const EngineOptions& options) : opt_(options) {
  LoadTrace trace;
  if (const char* v = std::getenv("ESPETERO_TRACE_EVICT")) trace_evict_ = *v == '1';
  // No sequence could ever be admitted (every request would wait forever).
  if (opt_.max_seqs < 1) throw std::runtime_error("max_seqs (--max-seqs) must be at least 1");
  const std::string& path = opt_.model_path;
  name_ = opt_.model_name;
  if (name_.empty()) {
    std::string p = path;
    while (!p.empty() && p.back() == '/') p.pop_back();
    name_ = p.substr(p.find_last_of('/') + 1);
  }
  // Other model families fail here with one message, before their tokenizer
  // or geometry would fail with a less clear one.
  check_model_supported(path);
  if (opt_.generation_config) sampling_defaults_ = read_sampling_defaults(path);
  trace.mark("model checks");
  const bool gguf = path.size() > 5 && path.compare(path.size() - 5, 5, ".gguf") == 0;
  tok_ = gguf ? Tokenizer::from_gguf(path) : Tokenizer::from_hf_dir(path);
  if (!tok_->chat_template().empty()) template_ = tmpl::Template::parse(tok_->chat_template());
  trace.mark("tokenizer + template");
  model_ = NativeModel::load(path, opt_.load);
  trace.mark("model (weights, pools)");
  if (opt_.profile) model_->set_profile(true);
  const ModelInfo& I = model_->info();
  // Models without multi-branch passes (Qwen4-Exp so far): no prompt-lookup
  // drafts (verified as leaves) and no decision trees.
  if (!I.leaves) opt_.prompt_lookup = false;
  if (opt_.draft_tokens < 0) opt_.draft_tokens = I.draft_default > 0 ? I.draft_default : 5;

  for (const char* t : {"<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|eot_id|>"}) {
    const int32_t id = tok_->token_id(t);
    if (id >= 0 && std::find(end_ids.begin(), end_ids.end(), id) == end_ids.end()) end_ids.push_back(id);
  }
  if (tok_->eos_id() >= 0 && std::find(end_ids.begin(), end_ids.end(), tok_->eos_id()) == end_ids.end())
    end_ids.push_back(tok_->eos_id());
  for (const int id : I.end_ids)  // Gemma 4: the config's end-of-turn ids, e.g. <turn|>
    if (id >= 0 && std::find(end_ids.begin(), end_ids.end(), id) == end_ids.end()) end_ids.push_back(id);
  if (I.vision) {
    image_token = tok_->token_id("<|image_pad|>");
    vision_start = tok_->token_id("<|vision_start|>");
    vision_end = tok_->token_id("<|vision_end|>");
    if (image_token < 0 || vision_start < 0 || vision_end < 0)
      throw std::runtime_error("vision model without image tokens in its vocabulary");
  }

  // Clef releases: a joint schema head next to the backbone answers
  // /v1/systemone (and /v1/decisions) from the final hidden states.
  if (is_dir(path) && clef::is_release(path)) {
    if (I.format == "gguf" || !I.leaves) throw std::runtime_error("Clef heads need a Qwen3.5 MLX-layout backbone");
    clef_ = clef::Head::load(path);
    if (clef_->hidden() != I.hidden) throw std::runtime_error("Clef head width differs from the backbone's hidden size");
    // The release resizes images with its own processor limits (Qwen2VLImageProcessor: 65,536 to
    // 16,777,216 pixels for the Clef releases), not Espetero's ~1 MP chat budget.
    for (const char* name : {"/processor_config.json", "/preprocessor_config.json"}) {
      try {
        const std::string txt = read_text(path + name);
        if (txt.empty()) continue;
        json::Value pc = json::parse(txt);
        if (const json::Value* ip = pc.get("image_processor"); ip && ip->is_object()) pc = *ip;
        if (const json::Value* sz = pc.get("size"); sz && sz->is_object()) {
          clef_min_pixels_ = sz->get_int("shortest_edge", 0);
          clef_max_pixels_ = sz->get_int("longest_edge", 0);
        }
        if (!clef_min_pixels_) clef_min_pixels_ = pc.get_int("min_pixels", 0);
        if (!clef_max_pixels_) clef_max_pixels_ = pc.get_int("max_pixels", 0);
        if (clef_max_pixels_) break;
      } catch (...) {
      }
    }
  }

  // Decisions: labels, the prompt builder and the shared header checkpoint.
  if (template_ && I.leaves && !clef_) {
    dp_ = std::make_unique<DecisionPrompts>();
    dp_->labels = make_labels(*tok_);
    const std::string image_block = I.vision ? tok_->decode(std::vector<int32_t>{vision_start, image_token, vision_end}) : "";
    dp_->builder = std::make_unique<DecisionPromptBuilder>(*tok_, *template_, dp_->labels, image_block);
    header_ = dp_->builder->header();
    if (!header_.empty()) {
      header_slot_ = model_->slot_alloc();
      TreeSpec t;
      t.root = header_.data();
      t.root_len = int(header_.size());
      t.save = header_slot_;
      model_->run({t});
      pinned_.push_back(header_slot_);
    }
  }
  trace.mark("decision header");
  warmup();
  trace.mark("warm-up");
  calibrate_drafts();
  trace.mark("draft costs");
  publish_counters();
  worker_ = std::thread([this] { loop(); });
}

void Engine::publish_counters() {
  free_slots_pub_ = model_->free_slots();
  free_kv_pub_ = model_->free_kv_tokens();
  active_pub_ = int(active_.size());
  std::lock_guard<std::mutex> lk(mu_);
  waiting_pub_ = int(waiting_.size());
}

// Draft costs at load (after TensorFold's window timings): on a primed
// 64-token context with the MTP state, the best of 3 wall times of a plain
// step, of an MTP draft pass of k tokens and of a verify pass with k chained
// draft leaves (behind min(3, 8 - k) root rows, as a pass after accepted
// drafts carries), k = 1 .. kmax, after two throwaway rounds that build the
// pipelines. The MTP head's verdicts compare these costs' ratios with the
// request's expected tokens, so neither cold first passes nor noisy wall
// times decide them; the live EWMAs start from them too. Skipped for models
// that stream experts (their pass costs depend on the expert cache).
void Engine::calibrate_drafts() {
  const ModelInfo& I = model_->info();
  if (!I.mtp || !I.leaves || !opt_.adaptive_drafts || opt_.draft_tokens <= 0) return;
  if (clef_) return;  // Clef models serve no generation
  if (model_->bench_json("xcache").find("\"slots\": 0,") == std::string::npos) return;
  const auto t0 = Clock::now();
  const int kmax = std::min(7, std::max(opt_.draft_tokens, opt_.draft_auto ? opt_.draft_max : 0));
  std::vector<int32_t> ctx(64);
  for (size_t i = 0; i < ctx.size(); ++i) ctx[i] = int32_t(1000 + 37 * i);
  int32_t arg = 0;
  const int base = model_->slot_alloc(), work = model_->slot_alloc();
  {
    TreeSpec t;
    t.root = ctx.data();
    t.root_len = int(ctx.size());
    t.save = base;
    t.score_root = true;
    t.mtp = true;
    t.out_argmax = &arg;
    model_->run({t});
  }
  if (!model_->slot_mtp(base)) {
    model_->slot_free(work);
    model_->slot_free(base);
    return;
  }
  const int32_t tok = arg;
  double plain = 1e30, verify[9], draft[9], commit = 1e30;
  for (int k = 0; k <= 8; ++k) verify[k] = draft[k] = 1e30;
  std::vector<int32_t> out(8), roots(8), leaf_tok, leaf_off, args(16);
  for (int rep = 0; rep < 5; ++rep) {  // 2 throwaway rounds, then the best of 3
    PassStats ps;
    TreeSpec p;
    p.parent = base;
    p.save = work;
    p.root = &tok;
    p.root_len = 1;
    p.score_root = true;
    p.mtp = true;
    p.out_argmax = args.data();
    model_->run({p}, &ps);
    if (rep >= 2) plain = std::min(plain, ps.wall_ms);
    for (int k = 1; k <= kmax; ++k) {
      DraftRequest q;
      q.slot = base;
      q.tokens = &tok;
      q.n_tokens = 1;
      q.n_draft = k;
      q.out = out.data();
      PassStats ds;
      model_->draft({q}, &ds);
      if (rep >= 2) draft[k] = std::min(draft[k], ds.wall_ms);
      const int nroot = I.commit ? 1 : std::max(1, std::min(3, 8 - k));  // committed drafts leave one root row
      for (int j = 0; j < nroot; ++j) roots[size_t(j)] = j == 0 ? tok : ctx[size_t(j)];
      leaf_tok.clear();
      leaf_off.assign(1, 0);
      for (int i = 0; i < k; ++i) {
        leaf_tok.insert(leaf_tok.end(), out.begin(), out.begin() + i + 1);
        leaf_off.push_back(int32_t(leaf_tok.size()));
      }
      // With verify-commit the pass runs in place (a commit chain needs it):
      // work := base first, then the chain is committed in work.
      if (I.commit) model_->slot_fork(work, base);
      TreeSpec v;
      v.parent = I.commit ? work : base;
      v.save = work;
      v.root = roots.data();
      v.root_len = nroot;
      v.n_leaves = k;
      v.leaf_offsets = leaf_off.data();
      v.leaf_tokens = leaf_tok.data();
      v.score_root = true;
      v.mtp = true;
      v.commit_chain = I.commit ? k : 0;
      v.out_argmax = args.data();
      PassStats vs;
      model_->run({v}, &vs);
      if (rep >= 2) verify[k] = std::min(verify[k], vs.wall_ms);
      if (I.commit) {  // the commit of the accepted drafts is part of the cost
        PassStats cs;
        model_->commit({{work, k}}, &cs);
        if (rep >= 2) commit = std::min(commit, cs.wall_ms);
      }
    }
  }
  model_->slot_free(work);
  model_->slot_free(base);
  DraftCost& D = dcost_;
  D.cal_plain = plain;
  D.plain = plain, D.n_plain = 3;
  for (int k = 1; k <= kmax; ++k) {
    D.cal_verify[k] = verify[k], D.cal_draft[k] = draft[k];
    D.verify[k] = verify[k], D.n_verify[k] = 2;
    D.draftk[k] = draft[k], D.n_draftk[k] = 2;
  }
  D.draft = draft[std::min(opt_.draft_tokens, kmax)], D.n_draft = 2;
  if (I.commit && commit < 1e30) D.commit = commit, D.n_commit = 2;
  D.calibrated = true;
  std::string line;
  for (int k = 1; k <= kmax; ++k) {
    char b[48];
    std::snprintf(b, sizeof b, "%s%d: %.1f + %.1f", k > 1 ? ", " : "", k, verify[k], draft[k]);
    line += b;
  }
  if (D.n_commit) {
    char b[48];
    std::snprintf(b, sizeof b, "; commit %.1f", D.commit);
    line += b;
  }
  std::fprintf(stderr, "[espetero] draft costs (ms, verify + draft by drafts; plain step %.1f): %s (%.1f s)\n", plain,
               line.c_str(), ms_since(t0) / 1e3);
}

// One throwaway pass shaped like real traffic (prefill + decision leaves with
// candidate readouts + argmax) so first requests do not pay for first-use
// pipeline and memory setup.
void Engine::warmup() {
  const int n_root = 48, n_leaf = 4;
  std::vector<int32_t> root(static_cast<size_t>(n_root)), leaves(static_cast<size_t>(2 * n_leaf));
  for (size_t i = 0; i < root.size(); ++i) root[i] = int32_t(100 + 7 * i);
  for (size_t i = 0; i < leaves.size(); ++i) leaves[i] = int32_t(200 + 3 * i);
  const int32_t leaf_off[3] = {0, n_leaf, 2 * n_leaf};
  const int32_t cand_off[3] = {0, 2, 4};
  const int32_t cand_ids[4] = {1, 2, 3, 4};
  float cand_logits[4], lse[3];  // readouts: two leaves, then the root
  int32_t args[3] = {0, 0, 0};
  TreeSpec t;
  t.root = root.data();
  t.root_len = n_root;
  t.score_root = true;
  t.out_argmax = args;
  const bool leaves_ok = model_->info().leaves;  // Qwen4-Exp: plain roots only so far
  if (leaves_ok) {
    t.n_leaves = 2;
    t.leaf_offsets = leaf_off;
    t.leaf_tokens = leaves.data();
    t.cand_offsets = cand_off;
    t.cand_ids = cand_ids;
    t.out_cand_logits = cand_logits;
    t.out_lse = lse;
  }
  const int s = model_->slot_alloc();
  t.save = s;
  model_->run({t});
  int32_t arg = args[leaves_ok ? 2 : 0];
  TreeSpec d;
  d.parent = s;
  d.save = s;
  d.root = &arg;
  d.root_len = 1;
  d.score_root = true;
  d.out_argmax = &arg;
  model_->run({d});
  model_->slot_free(s);
}

Engine::~Engine() {
  shutdown();
  if (worker_.joinable()) worker_.join();
}

void Engine::shutdown() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
}

// ---------------------------------------------------------------- limits
void Engine::check_detail(int64_t pixels, const std::string& field, const std::string& detail, int64_t ceiling) const {
  if (ceiling <= 0) ceiling = detail_ceiling();
  if (pixels > ceiling)
    throw RequestError(422, field + ": \"" + detail + "\" asks for up to " + std::to_string(pixels) +
                                " pixels per image, above this server's limit of " + std::to_string(ceiling) +
                                " (its --image-detail level); the operator can allow it with --allow-image-detail-max");
}

void Engine::check_text_size(size_t bytes, const std::string& what) const {
  const int ctx = model_->info().max_ctx;
  const size_t limit = size_t(std::max(1, ctx)) * 16;
  if (bytes > limit)
    throw RequestError(413, what + ": " + std::to_string(bytes) + " bytes of text exceed the limit of " +
                                std::to_string(limit) + " (16 bytes per token of the " + std::to_string(ctx) +
                                "-token context window)");
}

// ---------------------------------------------------------------- images
namespace {

// Images decoded and preprocessed at once across request threads: each holds
// its decoded pixels (up to 192 MB at the 64 MP decode cap) and the resize
// buffers while it runs; others wait their turn.
class DecodeGate {
 public:
  static constexpr int kSlots = 4;
  DecodeGate() {
    std::unique_lock<std::mutex> lk(mu());
    cv().wait(lk, [] { return busy() < kSlots; });
    ++busy();
  }
  ~DecodeGate() {
    {
      std::lock_guard<std::mutex> lk(mu());
      --busy();
    }
    cv().notify_one();
  }
  DecodeGate(const DecodeGate&) = delete;
  DecodeGate& operator=(const DecodeGate&) = delete;

 private:
  static std::mutex& mu() {
    static std::mutex m;
    return m;
  }
  static std::condition_variable& cv() {
    static std::condition_variable c;
    return c;
  }
  static int& busy() {
    static int n = 0;
    return n;
  }
};

}  // namespace

// An image's token run is computed in one pass (prompt chunks never
// split it), so a run longer than max_rows can never run: before, the chunker
// put it alone in a pass, which failed with "pass exceeds max_rows" (a 12 MP
// photo at detail max is ~11,700 tokens against the default 4,096 rows).
static void check_image_rows(int tokens, int max_rows) {
  if (tokens > max_rows)
    throw RequestError(422, "image: its " + std::to_string(tokens) + " tokens do not fit in one pass of " +
                                std::to_string(max_rows) + " rows (an image is computed in one pass); send a smaller "
                                "image or a lower detail, or start the server with --max-rows " +
                                std::to_string(tokens) + " or more");
}

vision::Spec Engine::image_spec(int64_t max_pixels, int64_t min_pixels) const {
  vision::Spec spec;
  const ModelInfo& I = model_->info();
  spec.patch = I.vision_patch;
  spec.merge = I.vision_merge;
  spec.temporal = I.vision_temporal;
  spec.max_pixels = max_pixels > 0 ? max_pixels : opt_.image_max_pixels;
  spec.min_pixels = std::min<int64_t>(min_pixels > 0 ? min_pixels : 65536, spec.max_pixels);
  spec.image_token_id = image_token;
  spec.vision_start_id = vision_start;
  spec.vision_end_id = vision_end;
  return spec;
}

int Engine::image_tokens(const std::string& data_url, int64_t max_pixels, int64_t min_pixels) const {
  if (!has_vision()) throw RequestError(422, "This model has no vision tower; images are not supported");
  std::string bytes, err;
  if (!vision::decode_data_url(data_url, bytes, err)) throw RequestError(422, "image: " + err);
  int64_t w = 0, h = 0;
  if (!vision::probe_image(bytes, w, h, err)) return -1;
  if (w > vision::max_image_pixels() / h)
    throw RequestError(422, "image: " + std::to_string(w) + "x" + std::to_string(h) + " pixels is over the limit of " +
                                std::to_string(vision::max_image_pixels()) + " pixels per image (--image-max-mp)");
  const vision::Spec spec = image_spec(max_pixels, min_pixels);
  int n = 0;
  try {
    n = vision::image_tokens(vision::grid_for(int(h), int(w), spec), spec.merge);
  } catch (const std::exception& e) {
    throw RequestError(422, std::string("image: ") + e.what());
  }
  check_image_rows(n, model_->info().max_rows);
  return n;
}

size_t Engine::expanded_length(const std::vector<int32_t>& ids, const std::vector<int>& runs) const {
  size_t n = 0, g = 0;
  for (int32_t t : ids) {
    if (t == image_token && image_token >= 0) {
      if (g >= runs.size()) throw RequestError(422, "more image placeholders than images");
      n += size_t(std::max(1, runs[g++]));
    } else {
      ++n;
    }
  }
  if (g != runs.size()) throw RequestError(422, "image placeholders and images differ in number");
  return n;
}

PreparedImage Engine::prepare_image(const std::string& data_url, int64_t max_pixels, int64_t min_pixels) const {
  if (!has_vision()) throw RequestError(422, "This model has no vision tower; images are not supported");
  std::string bytes, err;
  if (!vision::decode_data_url(data_url, bytes, err)) throw RequestError(422, "image: " + err);
  DecodeGate gate;
  vision::RGBImage rgb;
  if (!vision::decode_image(bytes, rgb, err)) throw RequestError(422, "image: " + err);
  vision::Spec spec = image_spec(max_pixels, min_pixels);
  // Normalization from the processor config when the checkpoint has one.
  if (is_dir(opt_.model_path)) {
    for (const char* name : {"/preprocessor_config.json", "/processor_config.json"}) {
      const std::string txt = read_text(opt_.model_path + name);
      if (txt.empty()) continue;
      try {
        json::Value pc = json::parse(txt);
        if (const json::Value* ip = pc.get("image_processor"); ip && ip->is_object()) pc = *ip;
        if (const json::Value* m = pc.get("image_mean"); m && m->is_array() && m->size() == 3)
          for (int i = 0; i < 3; ++i) spec.mean[i] = float((*m)[size_t(i)].as_double());
        if (const json::Value* s = pc.get("image_std"); s && s->is_array() && s->size() == 3)
          for (int i = 0; i < 3; ++i) spec.std[i] = float((*s)[size_t(i)].as_double());
        break;
      } catch (...) {
      }
    }
  }
  vision::Prepared p;
  try {
    p = vision::preprocess(rgb, spec);
  } catch (const std::exception& e) {
    throw RequestError(422, std::string("image: ") + e.what());
  }
  PreparedImage out;
  out.pixels = std::move(p.pixels);
  out.grid_t = p.grid_t;
  out.grid_h = p.grid_h;
  out.grid_w = p.grid_w;
  out.digest = hex(p.digest.data(), p.digest.size());
  // Images whose header image_tokens() could not read are checked here.
  check_image_rows(out.tokens(model_->info().vision_merge), model_->info().max_rows);
  return out;
}

std::vector<int32_t> Engine::expand_images(const std::vector<int32_t>& ids, const std::vector<PreparedImage>& images) const {
  std::vector<int32_t> out;
  size_t g = 0;
  const int merge = model_->info().vision_merge;
  for (int32_t t : ids) {
    if (t == image_token && image_token >= 0) {
      if (g >= images.size()) throw RequestError(422, "more image placeholders than images");
      out.insert(out.end(), size_t(images[g].tokens(merge)), t);
      ++g;
    } else {
      out.push_back(t);
    }
  }
  if (g != images.size()) throw RequestError(422, "image placeholders and images differ in number");
  return out;
}

std::vector<int32_t> Engine::mrope_positions(const std::vector<int32_t>& ids, const std::vector<PreparedImage>& images,
                                             int* next_pos) const {
  std::vector<int32_t> pos(ids.size() * 3);
  const int merge = model_->info().vision_merge;
  int nxt = 0;
  size_t i = 0, g = 0;
  while (i < ids.size()) {
    if (ids[i] == image_token && image_token >= 0) {
      if (g >= images.size()) throw RequestError(422, "more image blocks than images");
      const int t = images[g].grid_t, gh = images[g].grid_h / merge, gw = images[g].grid_w / merge;
      const size_t count = size_t(t) * size_t(gh) * size_t(gw);
      int mx = nxt;
      size_t k = 0;
      for (int a = 0; a < t; ++a)
        for (int b = 0; b < gh; ++b)
          for (int c = 0; c < gw; ++c, ++k) {
            if (i + k >= ids.size() || ids[i + k] != image_token) throw RequestError(422, "image run does not match its grid");
            pos[3 * (i + k) + 0] = nxt + a;
            pos[3 * (i + k) + 1] = nxt + b;
            pos[3 * (i + k) + 2] = nxt + c;
            mx = std::max({mx, nxt + a, nxt + b, nxt + c});
          }
      nxt = mx + 1;
      i += count;
      ++g;
    } else {
      pos[3 * i + 0] = pos[3 * i + 1] = pos[3 * i + 2] = nxt++;
      ++i;
    }
  }
  if (next_pos) *next_pos = nxt;
  return pos;
}

// ---------------------------------------------------------------- cache
std::string Engine::decision_key(const std::string& salt, const std::vector<int32_t>& prefix,
                                 const std::string& digest) const {
  std::string k = "dec";
  k.push_back('\0');
  k += salt;
  k.push_back('\0');
  k += digest;
  k.push_back('\0');
  k += tokens_key(prefix, prefix.size());
  return k;
}

Engine::CacheEntry* Engine::cache_find(const std::string& key) {
  auto it = cache_.find(key);
  if (it == cache_.end()) return nullptr;
  it->second.used = ++clock_;
  return &it->second;
}

std::string Engine::chat_key(const std::vector<std::string>& digests, const std::vector<int32_t>& tokens, bool mtp) {
  std::string key = "chat";
  key.push_back('\0');
  key.push_back(mtp ? 'M' : 'P');
  for (const auto& d : digests) key += d;
  key.push_back('\0');
  key += tokens_key(tokens, tokens.size());
  return key;
}

// An entry put over another of the same key frees the slot it replaces
// (prefix_store_restore loaded a stored checkpoint over a same-key entry
// and leaked its slot and KV blocks). A replaced slot still pinned (a waiting
// prompt's parent) is kept instead and the new one freed.
void Engine::cache_put(const std::string& key, CacheEntry e) {
  e.key = key;
  e.used = ++clock_;
  auto it = cache_.find(key);
  if (it != cache_.end() && it->second.slot >= 0 && it->second.slot != e.slot) {
    if (std::find(pinned_.begin(), pinned_.end(), it->second.slot) != pinned_.end()) {
      if (e.slot >= 0) model_->slot_free(e.slot);
      it->second.used = e.used;
      return;
    }
    model_->slot_free(it->second.slot);
  }
  cache_[key] = std::move(e);
}

void Engine::cache_drop(const std::string& key) {
  auto it = cache_.find(key);
  if (it == cache_.end()) return;
  model_->slot_free(it->second.slot);
  cache_.erase(it);
}

int Engine::acquire_slot(const std::vector<int>& avoid) {
  const int s = model_->slot_alloc();
  if (s >= 0) return s;
  return evict_lru(avoid);
}

// Evicts the least recently used checkpoint that nobody needs now; its slot
// (reset, still allocated) or -1. Checkpoints a running sequence continues or
// saved go last: in a flow, a photo's image checkpoint was last used when
// its first request started and looks older than the checkpoints of finished
// photos while that request still decodes; plain LRU evicted it first, and the
// photo's next request computed the image again.
int Engine::evict_lru(const std::vector<int>& avoid, bool live_ok) {
  std::vector<int> live;
  for (const auto& s : active_)
    for (const std::string& k : s->keep)
      if (auto it = cache_.find(k); it != cache_.end()) live.push_back(it->second.slot);
  // Leaves go before the checkpoints others extend. A shared prefix (a
  // flow's long system prompt and instructions) costs its whole length to
  // compute again; an image checkpoint ~1K rows.
  std::vector<const std::string*> parents;
  for (const auto& [k, e] : cache_)
    if (!e.parent.empty() && cache_.count(e.parent)) parents.push_back(&e.parent);
  auto is_parent = [&](const std::string& k) {
    for (const std::string* p : parents)
      if (*p == k) return true;
    return false;
  };
  auto victim = cache_.end();
  // 0: free leaves, 1: free checkpoints others extend, 2: those of running requests.
  int pass = 0;
  for (; pass < (live_ok ? 3 : 2) && victim == cache_.end(); ++pass)
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
      const int sl = it->second.slot;
      if (std::find(pinned_.begin(), pinned_.end(), sl) != pinned_.end()) continue;
      if (std::find(avoid.begin(), avoid.end(), sl) != avoid.end()) continue;
      if (pass < 2 && opt_.keep_live && std::find(live.begin(), live.end(), sl) != live.end()) continue;
      if (pass == 0 && is_parent(it->first)) continue;
      if (victim == cache_.end() || it->second.used < victim->second.used) victim = it;
    }
  if (victim == cache_.end()) return -1;
  if (std::find(live.begin(), live.end(), victim->second.slot) != live.end()) ++live_evictions_;
  if (trace_evict_)
    std::fprintf(stderr, "[evict] %s %zu tokens (pass %d, %s, used %llu of %llu)\n", victim->second.ns.c_str(),
                 victim->second.tokens.size(), pass - 1, is_parent(victim->first) ? "extended" : "leaf",
                 (unsigned long long)victim->second.used, (unsigned long long)clock_);
  const int sl = victim->second.slot;
  cache_.erase(victim);
  model_->slot_reset(sl);
  return sl;
}

// The entries evict_lru(avoid, live_ok) chooses from, and what freeing
// all of them gives back. Blocks they share with a running sequence or a pinned
// slot stay; blocks shared among them come back once all of them are gone.
int Engine::evictable_kv_tokens(const std::vector<int>& avoid, bool live_ok) const {
  std::vector<int> live;
  if (!live_ok && opt_.keep_live)
    for (const auto& s : active_)
      for (const std::string& k : s->keep)
        if (auto it = cache_.find(k); it != cache_.end()) live.push_back(it->second.slot);
  auto has = [](const std::vector<int>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); };
  std::vector<int> slots;
  for (const auto& [k, e] : cache_)
    if (e.slot >= 0 && !has(pinned_, e.slot) && !has(avoid, e.slot) && !has(live, e.slot)) slots.push_back(e.slot);
  return slots.empty() ? 0 : model_->freed_kv_tokens(slots);
}

void Engine::clear_cache() {
  // Cache entries belong to the scheduler thread: ask it, and wake it. The
  // flag is set under the scheduler's lock, or the wake-up could land between
  // its check of the flag and its wait (and be lost until the next request).
  {
    std::lock_guard<std::mutex> lk(mu_);
    clear_requested_ = true;
  }
  cv_.notify_all();
}

int Engine::reserved_kv_tokens() const {
  const int bt = std::max(1, model_->info().block_tokens);
  int64_t r = 0;
  for (const auto& s : active_)
    r += std::max<int64_t>(0, int64_t(s->req.prompt.size()) - s->done) + answer_reserve(s->req.max_tokens - s->generated) +
         2 * bt;  // a copied partial block, verify rows past the slot's length
  return int(std::min<int64_t>(r, INT32_MAX));
}

// Clients often ask for far more than they get back: a whole context window
// as max_tokens, or Claude Code's 32K-64K for replies of a few thousand
// tokens. Reserving all of it let one such request hold every other back
// while it ran (a 128K max_tokens on a 64K-token cache admitted nothing
// else). An answer counts up to max(8K tokens, a quarter of the KV cache);
// one that really runs past that is what the pass-level handling covers
// (prompt chunks wait, then the newest sequences stop).
int Engine::answer_reserve(int remaining) const {
  const ModelInfo& I = model_->info();
  const int64_t pool = int64_t(I.kv_blocks) * int64_t(std::max(1, I.block_tokens));
  const int64_t cap = std::max<int64_t>(8192, pool / 4);
  return int(std::min<int64_t>(std::max(0, remaining), cap));
}

// ---------------------------------------------------------------- decisions
json::Value Engine::decide(const json::Value& body) {
  if (clef_) return clef_decide(body);
  const auto started = Clock::now();
  if (!dp_)
    throw RequestError(422, model_->info().leaves ? "this model has no chat template; decisions need one"
                                                   : "decisions are not supported for this model yet");
  auto job = std::make_shared<DecisionJob>();
  job->started = started;
  job->req = parse_decision_request(body);
  DecisionRequest& r = job->req;
  const int64_t px = detail_pixels(r.image_detail);
  check_detail(px, "image_detail", r.image_detail);
  // The plan needs only the number of images: with the images' token runs
  // read from their headers, a request that cannot fit the context window is
  // refused before any image is decoded.
  job->plan = dp_->builder->plan(r, int(r.images.size()));
  DecisionPlan& plan = job->plan;
  // A decision runs as one tree of one pass: its uncached prefix (less
  // the shared header checkpoint, when it can continue that), the stem and
  // every field's leaf. A tree larger than a pass failed the pass, and with it
  // every request batched into it. The scheduler refuses such a job alone
  // (422) when its prefix is not cached; one whose stem and fields alone
  // cannot fit a pass is refused here, before any image is decoded.
  {
    size_t rows = plan.stem.size();
    for (const auto& l : plan.leaves) rows += l.size();
    const int max_rows = model_->info().max_rows;
    if (rows > size_t(max_rows))
      throw RequestError(422, "the decision's fields need " + std::to_string(rows) + " rows in one pass, more than "
                              "the " + std::to_string(max_rows) + " of this server (--max-rows): use fewer or shorter "
                              "fields, or start the server with --max-rows " + std::to_string(rows) + " or more");
  }
  if (!r.images.empty()) {
    std::vector<int> runs;
    for (const std::string& url : r.images) runs.push_back(image_tokens(url, px > 0 ? px : 0));
    size_t longest = 0;
    for (const auto& s : plan.suffixes) longest = std::max(longest, s.size());
    if (expanded_length(plan.prefix, runs) + longest + 1 > size_t(model_->info().max_ctx))
      throw RequestError(422, "Request exceeds the model context window (" + std::to_string(model_->info().max_ctx) +
                                  " tokens)");
  }
  for (const std::string& url : r.images) job->images.push_back(prepare_image(url, px > 0 ? px : 0));
  if (!job->images.empty()) {
    const size_t before = plan.prefix.size();
    plan.prefix = expand_images(plan.prefix, job->images);
    plan.images = int(job->images.size());
    plan.image_tokens = int(plan.prefix.size() - before + job->images.size());
    std::vector<int32_t> ps = plan.prefix;
    ps.insert(ps.end(), plan.stem.begin(), plan.stem.end());
    job->positions = mrope_positions(ps, job->images);
    for (const auto& im : job->images) job->digest += im.digest;
  }
  if (plan.max_prompt_tokens() + 1 > size_t(model_->info().max_ctx))
    throw RequestError(422, "Request exceeds the model context window (" + std::to_string(model_->info().max_ctx) +
                                " tokens)");
  job->key = decision_key(r.has_salt ? "s:" + r.cache_salt : "", plan.prefix, job->digest);

  // Exact-repeat memo: identical tokens give identical scores.
  std::string memo_key;
  json::Value memo_hit;
  if (opt_.memo) {
    memo_key = job->key;
    memo_key.push_back('\0');
    memo_key += tokens_key(plan.stem, plan.stem.size());
    for (size_t i = 0; i < plan.leaves.size(); ++i) {
      memo_key.push_back('\1');
      memo_key += tokens_key(plan.leaves[i], plan.leaves[i].size());
      memo_key += std::to_string(r.fields[i].n_values());
    }
    std::lock_guard<std::mutex> lk(memo_mu_);
    auto it = memo_.find(memo_key);
    if (it != memo_.end()) memo_hit = it->second;
  }
  if (memo_hit.is_null()) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (stop_) throw RequestError(503, "engine is shutting down");
      decisions_.push_back(job);
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> lk(job->mu);
    job->cv.wait(lk, [&] { return job->done; });
    if (!job->error.empty()) throw RequestError(job->error_status, job->error);
  }

  // Response (the reference format).
  const bool repeat = !memo_hit.is_null();
  std::vector<float> cand = repeat ? std::vector<float>() : job->cand_logits;
  std::vector<float> lse = repeat ? std::vector<float>() : job->lse;
  if (repeat) {
    for (const auto& v : memo_hit.get("cand")->as_array()) cand.push_back(float(v.as_double()));
    for (const auto& v : memo_hit.get("lse")->as_array()) lse.push_back(float(v.as_double()));
  } else if (opt_.memo) {
    json::Value mv = json::Value::object();
    json::Value a = json::Value::array(), b = json::Value::array();
    for (float v : cand) a.push_back(double(v));
    for (float v : lse) b.push_back(double(v));
    mv.set("cand", a);
    mv.set("lse", b);
    std::lock_guard<std::mutex> lk(memo_mu_);
    if (!memo_.count(memo_key)) {
      memo_[memo_key] = mv;
      memo_order_.push_back(memo_key);
      while (memo_order_.size() > opt_.memo) {
        memo_.erase(memo_order_.front());
        memo_order_.pop_front();
      }
    }
  }
  json::Value fields = json::Value::object();
  json::Value parsed = json::Value::object();
  json::Value abstained_fields = json::Value::array();
  size_t off = 0;
  for (size_t i = 0; i < r.fields.size(); ++i) {
    const DecisionField& f = r.fields[i];
    bool abst = false;
    json::Value res = decode_field(f, dp_->labels, cand.data() + off, lse[i], r, &abst);
    off += f.n_values();
    parsed.set(f.name, *res.get("value"));
    if (abst) abstained_fields.push_back(f.name);
    fields.set(f.name, std::move(res));
  }
  json::Value out = json::Value::object();
  out.set("model", name_);
  out.set("mode", "metal_parallel_categorical_scoring");
  out.set("strategy", "shared_prefix_tree");
  out.set("requested_strategy", r.strategy);
  out.set("parsed_json", std::move(parsed));
  out.set("fields", std::move(fields));
  out.set("scores_are_calibrated", false);
  out.set("abstained_fields", std::move(abstained_fields));
  out.set("elapsed_ms", round3(ms_since(started)));
  out.set("backend_requests", repeat ? 0 : 1);
  out.set("scored_fields", int64_t(r.fields.size()));
  out.set("max_prompt_tokens", int64_t(plan.max_prompt_tokens()));
  out.set("cache_block_tokens", json::Value());
  json::Value usage = json::Value::object();
  usage.set("prompt_tokens", int64_t(plan.prompt_tokens()));
  usage.set("completion_tokens", 0);
  usage.set("total_tokens", int64_t(plan.prompt_tokens()));
  out.set("usage", std::move(usage));
  json::Value metal = json::Value::object();
  metal.set("cache", repeat ? "repeat" : job->cache_state);
  metal.set("prefix_tokens", int64_t(plan.prefix.size()));
  metal.set("cached_prefix_tokens", repeat ? int64_t(plan.prefix.size()) : int64_t(job->cached));
  metal.set("shared_stem_tokens", int64_t(plan.stem.size()));
  metal.set("computed_tokens", repeat ? 0 : job->rows);
  metal.set("gpu_ms", repeat ? 0.0 : round3(job->stats.gpu_ms));
  metal.set("encode_ms", repeat ? 0.0 : round3(job->stats.encode_ms));
  metal.set("tokenize_ms", round3(plan.tokenize_ms));
  metal.set("dispatches", repeat ? 0 : job->stats.dispatches);
  metal.set("batched_requests", repeat ? 1 : job->batched);
  metal.set("images", plan.images);
  metal.set("image_tokens", plan.image_tokens);
  out.set("metal", std::move(metal));
  decisions_done_ += 1;
  return out;
}

// ---------------------------------------------------------------- Clef
json::Value Engine::systemone(const json::Value& body) {
  const auto started = Clock::now();
  if (!clef_)
    throw RequestError(422, "this model has no Clef joint head (joint_head.safetensors); /v1/systemone serves Clef releases");
  if (!body.is_object()) throw RequestError(422, "request body must be a JSON object");
  const json::Value* model = body.get("model");
  if (model && !model->is_null() && !model->is_string()) throw RequestError(422, "model: a string");
  const json::Value* state = body.get("state");
  if (!state) throw RequestError(422, "model and state are required");
  const json::Value* questions = body.get("questions");
  if (const json::Value* v = body.get("videos");
      v && !v->is_null() && !(v->is_array() && v->size() == 0) && !(v->is_bool() && !v->as_bool()))
    throw RequestError(422, "videos are not supported");
  auto job = std::make_shared<ClefJob>();
  // Image budget: by default the release's own processor limits; image_detail picks Espetero's levels
  // (standard ~1 MP, low, high, max) instead; media_kwargs (min_pixels / max_pixels, or size
  // {shortest_edge, longest_edge}) overrides both, as the release passes it to its processor.
  const std::string detail = body.get_string("image_detail", "auto");
  int64_t px = detail_pixels(detail), min_px = 0;
  if (px < 0) throw RequestError(422, "image_detail: low, auto, standard, high or max");
  // This server's own level for Clef requests is the release's processor budget.
  const int64_t ceiling = std::max(detail_ceiling(), clef_max_pixels_);
  check_detail(px, "image_detail", detail, ceiling);
  if ((detail.empty() || detail == "auto") && clef_max_pixels_ > 0) px = clef_max_pixels_, min_px = clef_min_pixels_;
  if (const json::Value* mk = body.get("media_kwargs"); mk && !mk->is_null()) {
    if (!mk->is_object()) throw RequestError(422, "media_kwargs: an object");
    const json::Value* sz = mk->get("size");
    const int64_t mx = sz && sz->is_object() ? sz->get_int("longest_edge", 0) : mk->get_int("max_pixels", 0);
    const int64_t mn = sz && sz->is_object() ? sz->get_int("shortest_edge", 0) : mk->get_int("min_pixels", 0);
    // At most the release's own limit (16,777,216 for the Clef releases; the
    // operator's ceiling when it has none): a larger max_pixels only admits
    // bigger images, a larger min_pixels upscales small ones (4e9 made a 64x64
    // image 63,264 pixels wide). Smaller values pass as the processor takes them.
    const int64_t hi = clef_max_pixels_ > 0 ? clef_max_pixels_ : std::min(ceiling, opt_.image_detail_max);
    if (mx > 0) px = std::min(mx, hi);
    if (mn > 0) min_px = std::min(mn, hi);
  }
  std::vector<std::string> urls;
  if (const json::Value* ims = body.get("images"); ims && !ims->is_null()) {
    if (!ims->is_array()) throw RequestError(422, "images: a list of image URLs");
    if (ims->size() > 16) throw RequestError(422, "images: at most 16 per request");
    for (const json::Value& im : ims->as_array()) {
      std::string url;
      if (im.is_string()) {
        url = im.as_string();
      } else if (im.is_object()) {
        if (const json::Value* u = im.get("url"); u && u->is_string()) url = u->as_string();
        else if (const json::Value* iu = im.get("image_url"); iu && iu->is_object()) url = iu->get_string("url", "");
      }
      if (url.empty()) throw RequestError(422, "images: each image is a data URL string or {\"url\": ...}");
      urls.push_back(std::move(url));
    }
  }
  const ModelInfo& I = model_->info();
  const int max_length = int(std::min<int64_t>(body.get_int("max_length", 16384), I.max_ctx));
  if (!urls.empty()) {
    // The image runs alone, read from the headers, must leave room in
    // max_length (the record holds every one of them, plus the system prompt
    // and the schema): checked before any image is decoded.
    int64_t total = 0;
    for (const std::string& url : urls) total += std::max(1, image_tokens(url, px > 0 ? px : 0, min_px));
    if (total >= max_length)
      throw RequestError(422, "the images take " + std::to_string(total) + " tokens; maximum is " +
                                  std::to_string(max_length));
  }
  for (const std::string& url : urls) job->images.push_back(prepare_image(url, px > 0 ? px : 0, min_px));
  std::vector<int> itoks;
  for (const PreparedImage& im : job->images) itoks.push_back(im.tokens(I.vision_merge));
  // max_state_tokens as the release slices: state_ids[:n] (0 = no state, negative = drop from the end).
  const json::Value* ms = body.get("max_state_tokens");
  if (ms && !ms->is_null() && !ms->is_int()) throw RequestError(422, "max_state_tokens: an integer");
  const bool has_max_state = ms && ms->is_int();
  const int max_state = has_max_state ? int(ms->as_int()) : 0;
  const auto t_enc = Clock::now();
  job->rec = clef::encode(*tok_, *state, questions ? *questions : json::Value(), itoks, image_token, max_length,
                          has_max_state, max_state);
  const double encode_ms = ms_since(t_enc);
  if (!job->images.empty()) job->positions = mrope_positions(job->rec.ids, job->images);
  for (const PreparedImage& im : job->images) job->digest += im.digest;
  if (const json::Value* salt = body.get("cache_salt"); salt && salt->is_string()) {  // no sharing across salts
    job->digest.push_back('\0');
    job->digest += salt->as_string();
  }
  job->queued = Clock::now();
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (stop_) throw RequestError(503, "engine is shutting down");
    clef_jobs_.push_back(job);
  }
  cv_.notify_all();
  {
    std::unique_lock<std::mutex> lk(job->mu);
    job->cv.wait(lk, [&] { return job->done; });
    if (!job->error.empty()) throw RequestError(job->error_status, job->error);
  }
  const auto t_head = Clock::now();
  const std::vector<std::vector<float>> logits = clef_->forward(job->hidden.data(), job->rec);
  const double head_ms = ms_since(t_head);
  // Development: ESPETERO_CLEF_DUMP=dir writes each request's ids, hidden states
  // and logits (<dir>/clefNNNN.{ids,hidden}.bin, .logits.json) for parity checks.
  if (const char* dump = std::getenv("ESPETERO_CLEF_DUMP"); dump && *dump) {
    static std::atomic<int> counter{0};
    char base[64];
    std::snprintf(base, sizeof(base), "/clef%04d", counter++);
    const std::string stem = std::string(dump) + base;
    std::ofstream(stem + ".ids.bin", std::ios::binary)
        .write(reinterpret_cast<const char*>(job->rec.ids.data()), std::streamsize(job->rec.ids.size() * 4));
    std::ofstream(stem + ".hidden.bin", std::ios::binary)
        .write(reinterpret_cast<const char*>(job->hidden.data()), std::streamsize(job->hidden.size() * 4));
    json::Value lj = json::Value::object();
    for (size_t q = 0; q < logits.size(); ++q) {
      json::Value a = json::Value::array();
      for (float x : logits[q]) a.push_back(double(x));
      lj.set(job->rec.questions[q].id, std::move(a));
    }
    std::ofstream(stem + ".logits.json") << json::dump(lj);
  }
  job->hidden.clear();
  job->hidden.shrink_to_fit();

  const bool want_logits = body.get_bool("return_logits", false);
  json::Value answers = json::Value::object(), raw = json::Value::object();
  for (size_t q = 0; q < job->rec.questions.size(); ++q) {
    const clef::Question& Q = job->rec.questions[q];
    const std::vector<float>& l = logits[q];
    const float mx = *std::max_element(l.begin(), l.end());
    double sum = 0;
    std::vector<double> p(l.size());
    for (size_t k = 0; k < l.size(); ++k) sum += (p[k] = std::exp(double(l[k] - mx)));
    for (double& x : p) x /= sum;
    answers.set(Q.id, clef::answer(Q, p));
    if (want_logits) {
      json::Value o = json::Value::object();
      for (size_t k = 0; k < l.size(); ++k) o.set(Q.option_ids[k], double(l[k]));
      raw.set(Q.id, std::move(o));
    }
  }
  json::Value out = json::Value::object();
  out.set("model", model && model->is_string() ? model->as_string() : name_);
  out.set("answers", std::move(answers));
  if (want_logits) out.set("logits", std::move(raw));
  json::Value usage = json::Value::object();
  usage.set("input_tokens", int64_t(job->rec.ids.size()));
  usage.set("output_tokens", 0);
  out.set("usage", std::move(usage));
  json::Value metal = json::Value::object();
  metal.set("mode", "clef_joint_head");
  metal.set("elapsed_ms", round3(ms_since(started)));
  metal.set("queue_ms", round3(std::chrono::duration<double, std::milli>(job->started - job->queued).count()));
  metal.set("backbone_ms", round3(job->run_ms));
  metal.set("gpu_ms", round3(job->gpu_ms));
  metal.set("head_ms", round3(head_ms));
  metal.set("encode_ms", round3(encode_ms));
  metal.set("passes", job->passes);
  metal.set("cached_prefix_tokens", job->cached);
  metal.set("images", int64_t(job->images.size()));
  int64_t image_tokens = 0;
  for (int n : itoks) image_tokens += n;
  metal.set("image_tokens", image_tokens);
  metal.set("state_tokens", job->rec.state_tokens);
  if (job->rec.truncated_state_tokens) metal.set("truncated_state_tokens", job->rec.truncated_state_tokens);
  out.set("metal", std::move(metal));
  decisions_done_ += 1;
  return out;
}

json::Value Engine::clef_decide(const json::Value& body) {
  const auto started = Clock::now();
  const DecisionRequest r = parse_decision_request(body);
  json::Value sb = json::Value::object(), qs = json::Value::object();
  sb.set("model", name_);
  sb.set("state", r.context);
  for (const DecisionField& f : r.fields) {
    json::Value q = json::Value::object();
    q.set("type", f.boolean ? "noul" : "choice");
    q.set("instructions", f.description.empty() ? f.name : f.description);
    if (!f.boolean) {
      json::Value c = json::Value::object();
      for (const std::string& ch : f.choices) c.set(ch, json::Value());
      q.set("criteria", std::move(c));
    }
    qs.set(f.name, std::move(q));
  }
  sb.set("questions", std::move(qs));
  if (!r.images.empty()) {
    json::Value ims = json::Value::array();
    for (const std::string& u : r.images) ims.push_back(u);
    sb.set("images", std::move(ims));
  }
  sb.set("image_detail", r.image_detail);
  if (r.has_salt) sb.set("cache_salt", r.cache_salt);
  sb.set("return_logits", true);
  const json::Value res = systemone(sb);
  const json::Value& logits = *res.get("logits");
  json::Value fields = json::Value::object(), parsed = json::Value::object(), abstained = json::Value::array();
  for (const DecisionField& f : r.fields) {
    const json::Value& lf = *logits.get(f.name);
    const size_t n = f.n_values();
    std::vector<double> l(n), prob(n), lp(n);
    for (size_t i = 0; i < n; ++i) l[i] = lf.get(f.boolean ? (i == 0 ? "true" : "false") : f.choices[i])->as_double();
    const double mx = *std::max_element(l.begin(), l.end());
    double z = 0, zt = 0;
    for (size_t i = 0; i < n; ++i) z += std::exp(l[i] - mx), zt += (prob[i] = std::exp((l[i] - mx) / r.score_temperature));
    size_t win = 0;
    for (size_t i = 0; i < n; ++i) {
      prob[i] /= zt;
      lp[i] = l[i] - mx - std::log(z);
      if (prob[i] > prob[win]) win = i;
    }
    const bool abst = r.has_min_probability && prob[win] < r.min_probability;
    json::Value o = json::Value::object(), cands = json::Value::array();
    o.set("value", abst ? json::Value() : f.value(win));
    o.set("selected_value", f.value(win));
    o.set("probability", prob[win]);
    o.set("abstained", abst);
    o.set("candidate_probability_mass", 1.0);
    for (size_t i = 0; i < n; ++i) {
      json::Value c = json::Value::object();
      c.set("value", f.value(i));
      c.set("label", f.boolean ? (i == 0 ? "true" : "false") : f.choices[i]);
      c.set("probability", prob[i]);
      c.set("logprob", lp[i]);
      cands.push_back(std::move(c));
    }
    o.set("candidates", std::move(cands));
    parsed.set(f.name, *o.get("value"));
    if (abst) abstained.push_back(f.name);
    fields.set(f.name, std::move(o));
  }
  const int64_t tokens = res.get("usage")->get_int("input_tokens", 0);
  json::Value out = json::Value::object();
  out.set("model", name_);
  out.set("mode", "clef_joint_head");
  out.set("strategy", "joint_schema_head");
  out.set("requested_strategy", r.strategy);
  out.set("parsed_json", std::move(parsed));
  out.set("fields", std::move(fields));
  out.set("scores_are_calibrated", false);
  out.set("abstained_fields", std::move(abstained));
  out.set("elapsed_ms", round3(ms_since(started)));
  out.set("backend_requests", 1);
  out.set("scored_fields", int64_t(r.fields.size()));
  out.set("max_prompt_tokens", tokens);
  json::Value usage = json::Value::object();
  usage.set("prompt_tokens", tokens);
  usage.set("completion_tokens", 0);
  usage.set("total_tokens", tokens);
  out.set("usage", std::move(usage));
  out.set("metal", *res.get("metal"));
  return out;
}

bool Engine::clef_evict_prefix(const std::vector<int>& keep) {
  auto lru = clef_prefixes_.end();
  for (auto it = clef_prefixes_.begin(); it != clef_prefixes_.end(); ++it)
    if (std::find(keep.begin(), keep.end(), it->slot) == keep.end() && (lru == clef_prefixes_.end() || it->used < lru->used))
      lru = it;
  if (lru == clef_prefixes_.end()) return false;
  model_->slot_free(lru->slot);
  clef_prefixes_.erase(lru);
  return true;
}

// Scheduler thread: the whole sequence through the backbone, in passes of at
// most prefill_chunk rows (an image run never straddles two; one larger than
// that gets a pass of up to max_rows), each continuing the previous one's
// state; every row's final-normed state is read out.
void Engine::run_clef(ClefJob& j) {
  j.started = Clock::now();
  const ModelInfo& I = model_->info();
  const int D = I.hidden, L = int(j.rec.ids.size());
  const int chunk = std::max(64, std::min(I.max_rows, opt_.prefill_chunk));
  if (L > I.max_ctx)  // nothing is generated: the record itself may fill the window
    throw RequestError(422, "Request exceeds the model context window (" + std::to_string(I.max_ctx) + " tokens)");
  j.hidden.assign(size_t(L) * size_t(D), 0.0f);
  std::vector<std::pair<int, int>> runs;  // image rows [first, last) per image
  for (int i = 0; i < L && runs.size() < j.images.size();) {
    if (j.rec.ids[size_t(i)] == image_token) {
      const int n = j.images[runs.size()].tokens(I.vision_merge);
      runs.emplace_back(i, i + n);
      i += n;
    } else {
      ++i;
    }
  }
  if (runs.size() != j.images.size()) throw RequestError(422, "image runs do not match the images");
  int done = 0, parent = -1;
  size_t g = 0;
  // Image prefix cache: the system prompt + media block of a request with
  // images is checkpointed (its own pass ends there) and reused, with its
  // hidden rows, by later requests about the same images.
  // Only a prefix that fits one pass is cached (several images can exceed max_rows: they are then
  // computed in passes cut between image runs, uncached).
  const int P = j.images.empty() || clef_prefix_cap_ == 0 || j.rec.prefix_tokens > I.max_rows ? 0 : j.rec.prefix_tokens;
  int cache_slot = -1;  // the cached checkpoint this request continues (never freed here)
  if (P > 0) {
    for (ClefPrefix& c : clef_prefixes_)
      if (c.digest == j.digest && int(c.tokens.size()) == P && std::equal(c.tokens.begin(), c.tokens.end(), j.rec.ids.begin())) {
        std::copy(c.hidden.begin(), c.hidden.end(), j.hidden.begin());
        c.used = ++clock_;
        cache_slot = parent = c.slot;
        done = j.cached = P;
        g = j.images.size();  // every image lies inside the prefix
        break;
      }
  }
  const bool save_prefix = P > 0 && cache_slot < 0;
  try {
    while (done < L) {
      int end = std::min(L, done + chunk);
      if (save_prefix && done < P) end = P;  // the prefix gets a pass of its own (checkpointed)
      for (const auto& [a, b] : runs)
        if (a < end && end < b) {
          end = a > done ? a : b;  // stop before the image, or take the whole image when it starts the pass
          break;
        }
      if (end - done > I.max_rows) throw RequestError(422, "an image needs more rows than one pass holds (raise --max-rows)");
      TreeSpec t;
      t.parent = parent;
      int slot = -1;
      const bool prefix_pass = save_prefix && end == P;
      // KV room for this pass's rows: drop least recently used prefixes (never the ones in use).
      while (model_->free_kv_tokens() < (end - done) + 2 * I.block_tokens && clef_evict_prefix({parent, cache_slot})) {
      }
      if (end < L || prefix_pass) {
        std::vector<int> avoid;
        if (parent >= 0) avoid.push_back(parent);
        if (cache_slot >= 0) avoid.push_back(cache_slot);
        for (const ClefPrefix& c : clef_prefixes_) avoid.push_back(c.slot);  // evict our own LRU below instead
        slot = acquire_slot(avoid);
        while (slot < 0 && clef_evict_prefix({parent, cache_slot}))  // no room: drop least recently used prefixes
          slot = acquire_slot(avoid);
        if (slot < 0) throw RequestError(503, "no free model slot for a long Clef request");
        t.save = slot;
      }
      PassStats st;
      try {  // from the slot's allocation on: a failed pass leaves the slot unused
        t.root = j.rec.ids.data() + done;
        t.root_len = end - done;
        if (!j.positions.empty()) {
          j.root_pos.assign(j.positions.begin() + long(done) * 3, j.positions.begin() + long(end) * 3);
          t.root_pos = j.root_pos.data();
        }
        j.irefs.clear();
        for (; g < runs.size() && runs[g].first < end; ++g) {
          const PreparedImage& im = j.images[g];
          ImageRef ref;
          ref.pixels = im.pixels.data();
          ref.grid_t = im.grid_t;
          ref.grid_h = im.grid_h;
          ref.grid_w = im.grid_w;
          ref.row = runs[g].first - done;
          j.irefs.push_back(ref);
        }
        if (!j.irefs.empty()) {
          t.images = j.irefs.data();
          t.n_images = int(j.irefs.size());
        }
        t.out_hidden = j.hidden.data() + size_t(done) * size_t(D);
        model_->run({t}, &st);
      } catch (...) {
        if (slot >= 0) model_->slot_free(slot);
        throw;
      }
      j.gpu_ms += st.gpu_ms;
      ++j.passes;
      ++passes_;
      rows_ += uint64_t(st.rows);
      gpu_ms_ = gpu_ms_ + st.gpu_ms;
      if (parent >= 0 && parent != cache_slot) model_->slot_free(parent);
      if (prefix_pass) {  // the new checkpoint joins the cache (LRU beyond the capacity)
        ClefPrefix c;
        c.tokens.assign(j.rec.ids.begin(), j.rec.ids.begin() + P);
        c.digest = j.digest;
        c.slot = slot;
        c.hidden.assign(j.hidden.begin(), j.hidden.begin() + long(P) * D);
        c.used = ++clock_;
        clef_prefixes_.push_back(std::move(c));
        cache_slot = slot;
        while (clef_prefixes_.size() > clef_prefix_cap_ && clef_evict_prefix({cache_slot, parent})) {
        }
      }
      parent = slot;
      done = end;
    }
  } catch (...) {
    if (parent >= 0 && parent != cache_slot) model_->slot_free(parent);
    throw;
  }
  j.run_ms = ms_since(j.started);
}

// ---------------------------------------------------------------- generation
std::shared_ptr<GenStream> Engine::generate(GenRequest request) {
  auto seq = std::make_shared<Seq>();
  seq->out = std::make_shared<GenStream>();
  seq->t_start = Clock::now();
  seq->t_wait = seq->t_start;
  const ModelInfo& I = model_->info();
  if (request.prompt.empty()) throw RequestError(422, "empty prompt");
  if (request.constraint && !request.forced.empty())
    throw RequestError(422, "structured output: not with teacher-forced scoring");
  for (int32_t t : request.prompt)
    if (t < 0 || t >= I.vocab) throw RequestError(422, "token id out of range");
  const int room = I.max_ctx - int(request.prompt.size());
  if (room < 1)
    throw RequestError(422, "Prompt exceeds the model context window (" + std::to_string(I.max_ctx) + " tokens)");
  request.max_tokens = std::max(1, std::min(request.max_tokens, room));
  // A temperature this close to 0 is greedy: dividing logits by it
  // (float on the GPU) overflows to inf, and inf - inf is NaN.
  if (request.sampling.temperature > 0 && request.sampling.temperature < 1e-5) request.sampling.temperature = 0;
  seq->req = std::move(request);
  for (const auto& o : seq->req.group_out) seq->out->link(o);  // they end with it until forked
  seq->rng.seed(seq->req.has_seed ? seq->req.seed : std::random_device{}());
  if (seq->req.has_seed) {
    seq->seed = seq->req.seed;
  } else {
    std::random_device rd;
    seq->seed = (uint64_t(rd()) << 32) ^ uint64_t(rd());
  }
  seq->out->prompt_tokens = int(seq->req.prompt.size());
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (stop_) throw RequestError(503, "engine is shutting down");
    // Priority queue: stable insert behind requests of the same or higher priority.
    auto it = waiting_.begin();
    while (it != waiting_.end() && (*it)->req.priority <= seq->req.priority) ++it;
    waiting_.insert(it, seq);
  }
  cv_.notify_all();
  return seq->out;
}

// The vocabulary of structured output, built once (~0.1 s for 248K tokens).
std::shared_ptr<const grammar::Vocab> Engine::grammar_vocab() const {
  std::lock_guard<std::mutex> lk(grammar_mu_);
  if (!grammar_vocab_) {
    auto v = std::make_shared<grammar::Vocab>(*tok_, model_->info().vocab);
    if (opt_.verbose)
      std::fprintf(stderr, "[espetero] structured output: %d tokens, %zu trie nodes, built in %.0f ms\n", v->size(),
                   v->trie_nodes(), v->build_ms);
    grammar_vocab_ = std::move(v);
  }
  return grammar_vocab_;
}

// Choice i's seed: the request's own for choice 0 (n = 1 and the first
// of n choices draw alike), splitmix64 of (seed, i) for the others.
uint64_t Engine::choice_seed(uint64_t seed, int i) {
  if (i == 0) return seed;
  uint64_t x = seed + 0x9E3779B97F4A7C15ull * uint64_t(i);
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

std::vector<std::shared_ptr<GenStream>> Engine::generate_n(GenRequest request, int n) {
  if (n <= 1) return {generate(std::move(request))};
  std::random_device rd;
  const uint64_t seed = request.has_seed ? request.seed : (uint64_t(rd()) << 32) ^ uint64_t(rd());
  request.seed = seed;  // the leader is choice 0
  request.has_seed = true;
  request.group_out.clear();
  request.group_seeds.clear();
  for (int i = 1; i < n; ++i) {
    auto o = std::make_shared<GenStream>();
    o->prompt_tokens = int(request.prompt.size());
    request.group_out.push_back(o);
    request.group_seeds.push_back(choice_seed(seed, i));
  }
  std::vector<std::shared_ptr<GenStream>> out = request.group_out;
  out.insert(out.begin(), generate(std::move(request)));
  return out;
}

// ---------------------------------------------------------------- scheduler
// ---------------------------------------------------------------- persisted prefixes
// Wall clock in nanoseconds: the disk cap's LRU order (file times are kept
// in nanoseconds too; whole seconds tied files saved together).
static int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string hex_digest(const std::string& bytes) {
  uint64_t h1 = 1469598103934665603ull, h2 = 0x9e3779b97f4a7c15ull;
  for (unsigned char c : bytes) {
    h1 = (h1 ^ c) * 1099511628211ull;
    h2 = (h2 + c) * 0xbf58476d1ce4e5b9ull;
    h2 ^= h2 >> 31;
  }
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%016llx%016llx", (unsigned long long)h1, (unsigned long long)h2);
  return buf;
}

void Engine::prefix_store_init() {
  if (opt_.prefix_dir.empty()) return;
  // A repo id names every revision of a model: the revision keeps their checkpoints apart.
  const std::string rev = opt_.model_revision.empty() ? std::string() : "@" + opt_.model_revision;
  prefix_root_ = opt_.prefix_dir + "/" + hex_digest(name_ + rev + "|" + model_->state_signature());
  // Checkpoints hold users' prompts: 0700 / 0600 (the hashed directory is the
  // engine's own, so one made by an older version is tightened too).
  paths::make_private_dirs(prefix_root_);
  ::chmod(prefix_root_.c_str(), 0700);
  stored_.clear();
  DIR* d = ::opendir(prefix_root_.c_str());
  if (!d) return;
  while (dirent* e = ::readdir(d)) {
    std::string n = e->d_name;
    if (n.size() < 5 || n.compare(n.size() - 4, 4, ".tok") != 0) continue;
    const std::string base = prefix_root_ + "/" + n.substr(0, n.size() - 4);
    std::ifstream f(base + ".tok", std::ios::binary);
    if (!f) continue;
    std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (bytes.size() < 1 || bytes.size() % 4 != 1) continue;
    StoredPrefix sp;
    sp.mtp = bytes[0] == 'M';
    sp.tokens.resize((bytes.size() - 1) / 4);
    std::memcpy(sp.tokens.data(), bytes.data() + 1, bytes.size() - 1);
    sp.path = base + ".ckpt";
    struct stat sb;
    if (::stat(sp.path.c_str(), &sb) != 0) continue;
    sp.bytes = uint64_t(sb.st_size);
    sp.used = int64_t(sb.st_mtimespec.tv_sec) * 1000000000 + int64_t(sb.st_mtimespec.tv_nsec);
    stored_.push_back(std::move(sp));
  }
  ::closedir(d);
  prefix_store_trim_roots();
  prefix_store_trim();
  if (opt_.verbose) std::fprintf(stderr, "[espetero] %zu stored prefixes in %s\n", stored_.size(), prefix_root_.c_str());
}

// A root names the model and its build (state_signature), so every
// rebuild and every other model starts a new one, and the disk cap applied to
// the current root only: the roots of earlier builds stayed on disk, up to
// prefix_dir_gb each. The cap now covers the whole directory: other roots go
// whole, least recently used first, while the directory is over the cap, and
// any unused for 30 days; one used in the last hour (perhaps by another
// running server) is kept. Only the engine's own files are removed (regular
// .ckpt / .tok / .tmp files in a directory named by a 32-digit hash).
void Engine::prefix_store_trim_roots() {
  struct Root {
    std::string path;
    double bytes = 0;
    int64_t newest = 0;
  };
  auto mtime_ns = [](const struct stat& sb) {
    return int64_t(sb.st_mtimespec.tv_sec) * 1000000000 + int64_t(sb.st_mtimespec.tv_nsec);
  };
  auto ours = [](const std::string& n) {
    for (const char* x : {".ckpt", ".tok", ".tmp"}) {
      const size_t k = std::strlen(x);
      if (n.size() > k && n.compare(n.size() - k, k, x) == 0) return true;
    }
    return false;
  };
  const std::string self = prefix_root_.substr(prefix_root_.find_last_of('/') + 1);
  std::vector<Root> roots;
  if (DIR* d = ::opendir(opt_.prefix_dir.c_str())) {
    while (dirent* e = ::readdir(d)) {
      const std::string n = e->d_name;
      if (n.size() != 32 || n == self || n.find_first_not_of("0123456789abcdef") != std::string::npos) continue;
      Root r;
      r.path = opt_.prefix_dir + "/" + n;
      struct stat sb;
      if (::lstat(r.path.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) continue;
      r.newest = mtime_ns(sb);
      if (DIR* rd = ::opendir(r.path.c_str())) {
        while (dirent* f = ::readdir(rd)) {
          struct stat fb;
          if (::lstat((r.path + "/" + f->d_name).c_str(), &fb) != 0 || !S_ISREG(fb.st_mode)) continue;
          r.bytes += double(fb.st_size);
          r.newest = std::max(r.newest, mtime_ns(fb));
        }
        ::closedir(rd);
      }
      roots.push_back(std::move(r));
    }
    ::closedir(d);
  }
  double mine = 0;
  for (const StoredPrefix& sp : stored_) mine += double(sp.bytes);
  double others = 0;
  for (const Root& r : roots) others += r.bytes;
  std::sort(roots.begin(), roots.end(), [](const Root& a, const Root& b) { return a.newest < b.newest; });
  const int64_t now = now_ns(), hour = int64_t(3600) * 1000000000;
  int removed = 0;
  double freed = 0;
  for (const Root& r : roots) {
    const bool stale = now - r.newest > 30 * 24 * hour;
    if (!stale && mine + others <= opt_.prefix_dir_gb * 1e9) continue;
    if (now - r.newest < hour) continue;
    if (DIR* rd = ::opendir(r.path.c_str())) {
      std::vector<std::string> files;
      while (dirent* f = ::readdir(rd))
        if (ours(f->d_name)) files.push_back(r.path + "/" + f->d_name);
      ::closedir(rd);
      double gone = 0;
      for (const std::string& f : files) {
        struct stat fb;
        if (::lstat(f.c_str(), &fb) == 0 && S_ISREG(fb.st_mode) && ::unlink(f.c_str()) == 0) gone += double(fb.st_size);
      }
      ::rmdir(r.path.c_str());  // fails (and stays) if anything else is inside
      others -= gone;
      freed += gone;
      ++removed;
    }
  }
  other_roots_bytes_ = std::max(0.0, others);
  if (removed)
    std::fprintf(stderr, "[espetero] removed the stored prefixes of %d other build(s) or model(s) (%.1f GB) from %s\n",
                 removed, freed / 1e9, opt_.prefix_dir.c_str());
}

// At most prefix_dir_gb on disk, the other roots kept by prefix_store_trim_roots
// included: the least recently used files go first (a checkpoint of a
// prompt of tens of thousands of tokens takes gigabytes); the
// newest is always kept.
void Engine::prefix_store_trim() {
  const double cap = std::max(0.0, opt_.prefix_dir_gb * 1e9 - other_roots_bytes_);
  double total = 0;
  for (const StoredPrefix& sp : stored_) total += double(sp.bytes);
  while (total > cap && stored_.size() > 1) {
    size_t v = 0, newest = 0;
    for (size_t i = 1; i < stored_.size(); ++i) {
      if (stored_[i].used < stored_[v].used) v = i;
      if (stored_[i].used > stored_[newest].used) newest = i;
    }
    if (v == newest) break;
    const std::string base = stored_[v].path.substr(0, stored_[v].path.size() - 5);
    ::unlink(stored_[v].path.c_str());
    ::unlink((base + ".tok").c_str());
    total -= double(stored_[v].bytes);
    if (opt_.verbose) std::fprintf(stderr, "[espetero] dropped a stored prefix of %zu tokens (disk cap)\n", stored_[v].tokens.size());
    stored_.erase(stored_.begin() + long(v));
  }
}

void Engine::prefix_store_put(const CacheEntry& e) {
  if (prefix_root_.empty() || !e.digests.empty()) return;  // text prefixes only
  // One file per prefix: a checkpoint with the MTP state replaces one without
  // (it serves both kinds of requests; a draft-less copy kept instead would
  // keep every drafting request from ever restoring that prefix).
  for (size_t i = 0; i < stored_.size(); ++i)
    if (stored_[i].tokens == e.tokens) {
      if (stored_[i].mtp || !e.mtp) return;
      stored_.erase(stored_.begin() + long(i));  // the file is replaced below (same name)
      break;
    }
  const auto t0 = Clock::now();
  const std::string base = prefix_root_ + "/" + hex_digest(tokens_key(e.tokens, e.tokens.size()));
  // Streamed from the slot's buffers to the file (no copy in memory); the
  // token list is written last, so an interrupted save leaves no entry. Both
  // files are created 0600 first (the writers keep an existing file's mode).
  if (!paths::make_private_file(base + ".ckpt.tmp") || !model_->slot_save(e.slot, base + ".ckpt.tmp")) {
    ::unlink((base + ".ckpt.tmp").c_str());
    return;
  }
  std::rename((base + ".ckpt.tmp").c_str(), (base + ".ckpt").c_str());
  {
    if (!paths::make_private_file(base + ".tok.tmp")) return;
    std::ofstream f(base + ".tok.tmp", std::ios::binary);
    f.put(e.mtp ? 'M' : 'P');
    f.write(reinterpret_cast<const char*>(e.tokens.data()), std::streamsize(e.tokens.size() * sizeof(int32_t)));
    if (!f) return;
  }
  std::rename((base + ".tok.tmp").c_str(), (base + ".tok").c_str());
  StoredPrefix sp;
  sp.tokens = e.tokens;
  sp.path = base + ".ckpt";
  sp.mtp = e.mtp;
  struct stat sb;
  sp.bytes = ::stat(sp.path.c_str(), &sb) == 0 ? uint64_t(sb.st_size) : 0;
  sp.used = now_ns();
  stored_.push_back(std::move(sp));
  if (opt_.verbose)
    std::fprintf(stderr, "[espetero] stored a prefix of %zu tokens (%llu MB, %.2f s)\n", e.tokens.size(),
                 (unsigned long long)(stored_.back().bytes >> 20), ms_since(t0) / 1e3);
  prefix_store_trim();
}

Engine::CacheEntry* Engine::prefix_store_restore(const std::vector<int32_t>& prompt, size_t longer_than, size_t limit,
                                                 bool want_mtp) {
  if (prefix_root_.empty()) return nullptr;
  StoredPrefix* best = nullptr;
  for (StoredPrefix& sp : stored_) {
    if (sp.tokens.size() <= longer_than || sp.tokens.size() >= prompt.size() || sp.tokens.size() > limit) continue;
    if (want_mtp && !sp.mtp) continue;
    if (best && sp.tokens.size() <= best->tokens.size()) continue;
    if (std::equal(sp.tokens.begin(), sp.tokens.end(), prompt.begin())) best = &sp;
  }
  if (!best) return nullptr;
  const auto t0 = Clock::now();
  // Room for the slot: evict LRU entries that nobody pins, as acquire_slot does.
  int slot = model_->slot_load(best->path);
  if (slot == -2) {
    // Only when evicting makes room for its blocks; before, a prefix
    // that could not fit evicted every checkpoint and was not loaded either.
    const int64_t bt = std::max(1, model_->info().block_tokens);
    const int64_t kv = (int64_t(best->tokens.size()) + bt - 1) / bt * bt;
    if (int64_t(model_->free_kv_tokens()) + evictable_kv_tokens({}, true) < kv) return nullptr;
  }
  while (slot == -2) {  // no slot or KV room: evict checkpoints nobody pins (LRU)
    const int victim = evict_lru({});
    if (victim < 0) return nullptr;
    model_->slot_free(victim);
    slot = model_->slot_load(best->path);
  }
  if (slot < 0) {
    // Not a checkpoint of this model any more (deleted or replaced by another
    // server sharing the directory): forget it.
    stored_.erase(stored_.begin() + (best - stored_.data()));
    return nullptr;
  }
  ::utimes(best->path.c_str(), nullptr);  // most recently used (the disk cap keeps it longest)
  best->used = now_ns();
  CacheEntry e;
  e.ns = "chat";
  e.tokens = best->tokens;
  e.slot = slot;
  e.mtp = model_->slot_mtp(slot);
  const std::string key = chat_key({}, e.tokens, e.mtp);
  cache_put(key, e);
  if (opt_.verbose)
    std::fprintf(stderr, "[espetero] restored a %zu-token prefix from disk (%.2f s)\n", e.tokens.size(), ms_since(t0) / 1e3);
  return cache_find(key);
}

void Engine::loop() {
  prefix_store_init();
  {  // expert-streaming models expose their cache counters in /v1/stats
    const std::string x = model_->bench_json("xcache");
    xstream_ = x.find("\"slots\": 0,") == std::string::npos;
  }
  while (true) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] {
        return stop_ || clear_requested_ || !decisions_.empty() || !clef_jobs_.empty() || !waiting_.empty() ||
               !active_.empty() || !embed_jobs_.empty();
      });
      if (stop_) break;
    }
    try {
      step();
    } catch (const std::exception& e) {
      // A failed pass fails its work, never the server.
      std::fprintf(stderr, "[espetero] scheduler step failed: %s\n", e.what());
    }
    publish_counters();
  }
  // Fail whatever is left (shutdown: the pass that was running has finished).
  std::lock_guard<std::mutex> lk(mu_);
  const char* msg = "engine stopped: the server is shutting down";
  for (auto& j : decisions_) {
    std::lock_guard<std::mutex> jl(j->mu);
    j->error = msg;
    j->error_status = 503;
    j->done = true;
    j->cv.notify_all();
  }
  decisions_.clear();
  for (auto& j : clef_jobs_) {
    std::lock_guard<std::mutex> jl(j->mu);
    j->error = msg;
    j->error_status = 503;
    j->done = true;
    j->cv.notify_all();
  }
  clef_jobs_.clear();
  for (auto& s : waiting_) {
    s->out->completion_tokens = s->generated;
    s->out->finish("error", msg);
  }
  waiting_.clear();
  for (auto& s : active_) {
    s->out->completion_tokens = s->generated;
    s->out->cached_tokens = s->cached;
    s->out->finish("error", msg);
  }
  active_.clear();
  embed_fail_all(msg);
}

// Sampled requests whose distribution the GPU top-k candidates determine.
static bool gpu_topk_ok(const EngineOptions& o, const SamplingParams& sp) {
  return o.gpu_topk && sp.temperature > 0 && !sp.penalized() && sp.top_k >= 1 && sp.top_k <= 32;
}
// ... and whose token the GPU draws: no CPU work per sampled token, and
// decode chains can feed it to the next step.
static bool gpu_draw_ok(const EngineOptions& o, const SamplingParams& sp) {
  return gpu_topk_ok(o, sp) && o.exact_sampling && o.gpu_draw;
}
// A tree's GPU top-k / draw fields for sequence sampling `sp`; pos0: the
// absolute position of the token its root readout produces.
static void set_gpu_sampling(TreeSpec& t, const EngineOptions& o, const SamplingParams& sp, uint64_t seed, int64_t pos0) {
  if (!gpu_topk_ok(o, sp)) return;
  t.topk = sp.top_k;
  t.temperature = float(sp.temperature);
  if (!gpu_draw_ok(o, sp)) return;
  t.gpu_draw = true;
  t.seed = seed;
  t.draw_pos0 = pos0;
  t.top_p = float(sp.top_p);
  t.min_p = float(sp.min_p);
}

// Logprobs (sampled requests too): the readouts of a request that asks
// for them carry at least max(4, top_logprobs) GPU top-k candidates. A greedy
// request's candidates are raw logits. A request with its own GPU top-k keeps
// its head and temperature (its draws stay the same), so its candidates are
// scaled: the raw log-normalizer of every readout then comes from lse_rows
// (`lp_lse`, `readouts` of them). Call after set_gpu_sampling.
static void set_logprob_readouts(TreeSpec& t, const GenRequest& r, std::vector<float>& lp_lse, int readouts) {
  lp_lse.clear();
  if (r.logprobs < 0) return;
  t.topk_min = std::max(4, r.logprobs);
  if (t.topk > 0 && t.temperature != 1.0f) {
    lp_lse.assign(size_t(readouts), 0.0f);
    t.out_lse = lp_lse.data();
  }
}

// A constrained sequence picks its tokens on the CPU under its
// grammar's mask: every readout's full logits and top 32 candidates (raw, or
// the request's own scaled head), and no GPU draw. Call last.
static void set_constrained_readouts(TreeSpec& t, const GenRequest& r) {
  if (!r.constraint) return;
  t.gpu_draw = false;
  t.want_logits = true;
  t.topk_min = std::max(t.topk_min, 32);
}

// What a constrained choice reads of a readout of the last pass.
static grammar::Readout readout_view(const NativeModel& m, int tree, int readout, int32_t argmax, const EngineOptions& o,
                                     const SamplingParams& sp) {
  grammar::Readout r;
  r.argmax = argmax;
  const TopK tk = m.topk(tree, readout);
  r.cand = tk.ids;
  r.cand_x = tk.x;
  r.n_cand = tk.n;
  r.scaled = gpu_topk_ok(o, sp);
  r.logits = m.logits(tree, readout);
  r.vocab = m.info().vocab;
  return r;
}

// A token's log-probability and the top `n_top` alternatives at a readout of
// the last pass: the model's distribution, log-softmax of the raw
// logits (temperature 1, before penalties, top-k, top-p and min-p, as vLLM's
// default raw logprobs): the readout's logits minus their log-normalizer
// (`lse` when the request's candidates are scaled, else the GPU top-k's own).
// The alternatives are the GPU top-k candidates, most likely first.
TokenLogprob Engine::readout_logprob(int tree, int readout, int32_t tok, int n_top, const float* lse) const {
  TokenLogprob r;
  r.token = tok;
  r.logprob = -9999.0f;  // no candidates (OpenAI's value for "negligible")
  const TopK tk = model_->topk(tree, readout);
  double lz = 0;
  if (lse) lz = double(*lse);
  else if (tk.n > 0 && tk.total > 0) lz = double(tk.max) + std::log(double(tk.total));
  else return r;
  const float* lg = model_->logits(tree, readout);
  r.logprob = float(double(lg[tok]) - lz);
  for (int j = 0; j < tk.n && j < n_top; ++j) r.top.push_back({tk.ids[j], float(double(lg[tk.ids[j]]) - lz)});
  return r;
}

// The distribution a sequence samples a readout from (the step's `dist`): the
// GPU top-k's head, else the full logits with the penalties applied.
Distribution Engine::readout_dist(int tree, int readout, const SamplingParams& sp, const PenaltyState* pen) const {
  if (gpu_topk_ok(opt_, sp)) {
    const TopK tk = model_->topk(tree, readout);
    if (tk.n >= sp.top_k && tk.total > 0) return distribution_topk(tk.ids, tk.x, tk.n, tk.max, tk.total, sp);
  }
  const int V = model_->info().vocab;
  const float* lg = model_->logits(tree, readout);
  if (sp.penalized() && pen && !pen->empty()) {
    std::vector<float> buf(lg, lg + V);
    pen->apply(buf.data(), sp);
    return distribution(buf.data(), V, sp);
  }
  return distribution(lg, V, sp);
}

// Draft depth: the MTP draft count k that maximizes the expected tokens
// per millisecond of a round, E(k) / (verify(k) + draft(k)), where
// E(k) = 1 + sum_{d<k} prod_{i<=d} a_i with a_i the request's acceptance at
// depth i (verify passes that reached i and accepted it) under a Beta prior:
// the engine's acceptance at depth i over all requests once it has 20
// samples (weight 8; a depth not measured yet borrows the deepest measured
// one above it, times 0.9 per step, weight 4), else 0.6 * 0.9^i (weight 2),
// and the costs are the pass-cost EWMAs (a count not
// measured yet borrows the nearest measured one: verify cost is flat for
// 3..8 rows on the MMA GEMV, a draft pass grows by one MTP step per token).
// Every 16th MTP pass drafts one deeper than the optimum so the next depth's
// acceptance keeps being measured.
double Engine::mtp_accept(const Seq& s, int d) const {
  double prior = 0.6 * std::pow(0.9, double(d)), w = 2.0;
  for (int dd = d; dd >= 0; --dd)
    if (dcost_.n_acc_depth[dd] >= 20) {
      prior = dcost_.acc_depth[dd] * std::pow(0.9, double(d - dd));
      w = dd == d ? 8.0 : 4.0;
      break;
    }
  return (double(s.dep_acc[0][d]) + w * prior) / (double(s.dep_reach[0][d]) + w);
}

// Expected tokens of an MTP verify pass with k drafts: E(k) above.
double Engine::mtp_expect(const Seq& s, int k) const {
  double p = 1, tokens = 1;
  for (int d = 0; d < std::min(k, 8); ++d) {
    p *= mtp_accept(s, d);
    tokens += p;
  }
  return tokens;
}

int Engine::choose_depth(const Seq& s, int budget) const {
  const int kmax = std::min({budget, opt_.draft_max, 7});
  const int k0 = std::max(1, std::min(opt_.draft_tokens, kmax));
  if (kmax <= 0) return 0;
  auto nearest = [&](const double* v, const int* n, int k, bool scale) -> double {
    if (n[k] >= 2) return v[k];
    for (int dk = 1; dk < 8; ++dk)
      for (int kk : {k - dk, k + dk})
        if (kk >= 1 && kk <= 8 && n[kk] >= 2) return scale ? v[kk] * double(k + 1) / double(kk + 1) : v[kk];
    return 0.0;
  };
  if (nearest(dcost_.verify, dcost_.n_verify, k0, false) <= 0 || nearest(dcost_.draftk, dcost_.n_draftk, k0, true) <= 0)
    return k0;  // no costs measured yet: the configured depth
  double best = -1, p = 1, tokens = 1;
  int bk = k0;
  for (int k = 1; k <= kmax; ++k) {
    p *= mtp_accept(s, k - 1);
    tokens += p;
    const double cost = nearest(dcost_.verify, dcost_.n_verify, k, false) + nearest(dcost_.draftk, dcost_.n_draftk, k, true);
    if (cost > 0 && tokens / cost > best) best = tokens / cost, bk = k;
  }
  if (s.dep_reach[0][0] % 16 == 15 && bk < kmax) ++bk;
  return bk;
}

// Consecutive losses after which the MTP head stops for the request.
constexpr int kMtpGiveUp = 4;
// Passes a decision job waits for the rows decoding sequences leave
// before the next pass reserves them for it.
constexpr int kDecisionAging = 4;

static inline void ewma(double& v, int& n, double x) {
  v = n == 0 ? x : 0.8 * v + 0.2 * x;
  ++n;
}

// KV reservation at admission (and for forked choices); off with
// ESPETERO_KV_RESERVE=0 (development: the plain free-blocks check, so the
// pass-level shortage handling can be exercised).
static bool kv_reserve() {
  static const bool on = [] {
    const char* v = std::getenv("ESPETERO_KV_RESERVE");
    return !(v && *v == '0');
  }();
  return on;
}

// The other choices of an n > 1 request start from its leader's prompt.
// Once the leader's last prompt chunk has run, each choice gets a work slot
// forked from the leader's (KV blocks shared, recurrent and MTP state copied)
// and its first token from the leader's final readout, drawn with its own
// seed (greedy: the same token); then it decodes as any sequence. A choice
// that finds the server at --max-seqs, no free slot or no KV room for its
// answer waits as a request of its own instead (it computes the prompt again,
// or continues from a cached checkpoint of it).
void Engine::fork_group(Seq& L, std::vector<Seq*>& touched, std::vector<std::vector<int32_t>>& produced,
                        std::vector<std::vector<TokenLogprob>>& produced_lp) {
  std::vector<std::shared_ptr<GenStream>> outs = std::move(L.req.group_out);
  std::vector<uint64_t> seeds = std::move(L.req.group_seeds);
  L.req.group_out.clear();
  L.req.group_seeds.clear();
  L.out->unlink();
  const SamplingParams& sp = L.req.sampling;
  const int P = int(L.req.prompt.size());
  const int bt = model_->info().block_tokens > 0 ? model_->info().block_tokens : 64;
  // A forked choice shares the leader's prompt blocks: it needs room for its
  // answer beside what the running sequences (the leader among them) may
  // still take, as admission asks of any request, with checkpoints no running
  // request uses evicted for it; otherwise it waits as a request of its own.
  // (Weighing min(max_tokens, 1024) against the free blocks alone let a later
  // pass run out and stop the newest choices.)
  const bool reserve = kv_reserve();
  const int reserved = reserve ? reserved_kv_tokens() : 0;
  int taken = 0;  // the forks admitted so far
  const auto now = Clock::now();
  // A forked choice's prompt is computed: its images keep their digests, not their pixels.
  auto copy_req = [&](GenRequest& r, bool pixels) {
    if (pixels) {
      r = L.req;
      return;
    }
    std::vector<PreparedImage> images = std::move(L.req.images);
    L.req.images.clear();
    for (const PreparedImage& im : images) {
      PreparedImage p;
      p.grid_t = im.grid_t, p.grid_h = im.grid_h, p.grid_w = im.grid_w;
      p.digest = im.digest;
      L.req.images.push_back(std::move(p));
    }
    r = L.req;
    L.req.images = std::move(images);
  };
  for (size_t k = 0; k < outs.size() && k < seeds.size(); ++k) {
    auto f = std::make_shared<Seq>();
    f->out = outs[k];
    f->seed = seeds[k];
    f->rng.seed(seeds[k]);
    f->t_start = L.t_start;
    f->t_wait = now;
    if (f->out->cancelled()) {
      f->out->finish("cancelled");
      continue;
    }
    const int need = reserve ? answer_reserve(L.req.max_tokens) + 2 * bt : std::min(L.req.max_tokens, 1024) + 64 + bt;
    // Checkpoints go only when that makes the fork fit (else the choice
    // waits as a request of its own, and they stay cached).
    if (reserve && model_->free_kv_tokens() - reserved - taken < need &&
        model_->free_kv_tokens() + evictable_kv_tokens({}, false) - reserved - taken >= need)
      while (model_->free_kv_tokens() - reserved - taken < need) {
        const int victim = evict_lru({}, false);
        if (victim < 0) break;
        model_->slot_free(victim);
      }
    const bool room = model_->free_kv_tokens() - reserved - taken >= need;
    int work = int(active_.size()) < opt_.max_seqs && room ? acquire_slot({}) : -1;
    if (work >= 0) {
      try {
        model_->slot_fork(work, L.work);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[espetero] forking a choice failed: %s\n", e.what());
        model_->slot_free(work);
        work = -1;
      }
    }
    if (work < 0) {  // a request of its own, first in its priority class
      copy_req(f->req, true);
      f->req.seed = seeds[k];
      std::lock_guard<std::mutex> lk(mu_);
      auto it = waiting_.begin();
      while (it != waiting_.end() && (*it)->req.priority < f->req.priority) ++it;
      waiting_.insert(it, f);
      continue;
    }
    taken += need;
    copy_req(f->req, false);
    f->req.seed = seeds[k];
    f->work = work;
    pinned_.push_back(work);
    f->mtp = L.mtp;
    f->cached = L.cached;
    // Its prompt ends in this pass, as the leader's: the step's end-of-prompt bookkeeping follows.
    f->done = f->chunk_end = P;
    f->final_chunk = true;
    if (sp.penalized()) f->pen.add_prompt(f->req.prompt.data(), f->req.prompt.size());
    if (opt_.prompt_lookup) f->ngram.extend(f->req.prompt.data(), f->req.prompt.size());
    int32_t tok;
    if (L.req.constraint) {  // the choice's own parse state, its first token under the mask
      f->gm = std::make_unique<grammar::Matcher>(L.req.constraint);
      const grammar::Readout r = readout_view(*model_, L.tree, 0, L.arg[0], opt_, sp);
      if (sp.greedy()) tok = grammar::masked_argmax(*f->gm, r, sp, &L.pen);
      else if (opt_.exact_sampling) tok = sample_at(grammar::masked_distribution(*f->gm, r, sp, &L.pen), f->seed, P);
      else tok = sample(grammar::masked_distribution(*f->gm, r, sp, &L.pen), f->rng);
      if (tok < 0 || !f->gm->advance(tok)) f->gm_error = true;
    } else if (sp.greedy()) {
      tok = L.arg[0];
    } else if (opt_.exact_sampling) {
      tok = sample_at(readout_dist(L.tree, 0, sp, &L.pen), f->seed, P);
    } else {
      tok = sample(readout_dist(L.tree, 0, sp, &L.pen), f->rng);
    }
    std::vector<TokenLogprob> lp;
    if (f->req.logprobs >= 0)
      lp.push_back(readout_logprob(L.tree, 0, tok, f->req.logprobs, L.lp_lse.empty() ? nullptr : L.lp_lse.data()));
    active_.push_back(f);
    touched.push_back(f.get());
    produced.push_back({tok});
    produced_lp.push_back(std::move(lp));
  }
}

void Engine::step() {
  const ModelInfo& I = model_->info();
  if (clear_requested_.exchange(false)) {
    for (ClefPrefix& c : clef_prefixes_) model_->slot_free(c.slot);
    clef_prefixes_.clear();
    for (auto it = cache_.begin(); it != cache_.end();) {
      if (std::find(pinned_.begin(), pinned_.end(), it->second.slot) != pinned_.end()) {
        ++it;
        continue;
      }
      model_->slot_free(it->second.slot);
      it = cache_.erase(it);
    }
  }
  // ---- Clef requests: one at a time, alone in their passes (Clef
  // models serve nothing else). The head runs on the caller's thread.
  if (clef_) {
    std::shared_ptr<ClefJob> cj;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!clef_jobs_.empty()) {
        cj = clef_jobs_.front();
        clef_jobs_.pop_front();
      }
    }
    if (cj) {
      std::string err;
      int status = 502;
      try {
        run_clef(*cj);
      } catch (const RequestError& e) {
        err = e.what();
        status = e.status;
      } catch (const std::exception& e) {
        err = e.what();
      }
      if (!err.empty()) std::fprintf(stderr, "[espetero] clef request failed: %s\n", err.c_str());
      {
        std::lock_guard<std::mutex> lk(cj->mu);
        cj->error = err;
        cj->error_status = status;
        cj->done = true;
      }
      cj->cv.notify_all();
      return;
    }
  }
  // ---- embeddings: one pass of the oldest job, ahead of the step's own.
  embed_step();
  const int max_rows = I.max_rows, max_readouts = I.max_readouts;

  // ---- retire cancelled sequences
  static const bool trace = [] {
    const char* v = std::getenv("ESPETERO_TRACE");
    return v && *v && *v != '0';
  }();
  auto finish_seq = [&](Seq& s, const std::string& reason, const std::string& err = "") {
    if (reason == "error") std::fprintf(stderr, "[espetero] generation failed after %d tokens: %s\n", s.generated, err.c_str());
    s.out->completion_tokens = s.generated;
    s.out->cached_tokens = s.cached;
    json::Value t = json::Value::object();
    const double total = ms_since(s.t_start);
    t.set("prefill_ms", round3(s.t_first.time_since_epoch().count() ? std::chrono::duration<double, std::milli>(s.t_first - s.t_start).count() : total));
    const double dec = s.t_decode.time_since_epoch().count() ? ms_since(s.t_decode) : 0;
    t.set("decode_ms", round3(dec));
    t.set("decode_tokens_per_s", dec > 0 && s.generated > 1 ? round3((s.generated - 1) / (dec / 1000)) : json::Value());
    t.set("passes", s.passes);
    if (s.drafted) t.set("draft_acceptance", round3(double(s.accepted) / double(s.drafted)));
    if (s.lookups) t.set("lookup_passes", s.lookups);
    if (s.lost_mtp) t.set("mtp_plain_stretches", s.lost_mtp);  // losses of the MTP head (each a back-off)
    if (s.dep_reach[0][0] || s.dep_reach[1][0]) {
      json::Value dp = json::Value::object();
      for (int src = 0; src < 2; ++src) {
        if (!s.dep_reach[src][0]) continue;
        json::Value o = json::Value::object(), re = json::Value::array(), ac = json::Value::array();
        int n = 8;
        while (n > 0 && !s.dep_reach[src][n - 1]) --n;
        for (int d = 0; d < n; ++d) {
          re.push_back(int64_t(s.dep_reach[src][d]));
          ac.push_back(int64_t(s.dep_acc[src][d]));
        }
        o.set("reached", std::move(re));
        o.set("accepted", std::move(ac));
        dp.set(src ? "lookup" : "mtp", std::move(o));
      }
      json::Value ks = json::Value::array();
      for (int k = 0; k <= 8; ++k) ks.push_back(int64_t(s.dep_k[k]));
      dp.set("mtp_k", std::move(ks));
      if (opt_.draft_auto && dcost_.n_acc_depth[0]) {
        json::Value pr = json::Value::array();
        for (int d = 0; d < 8 && dcost_.n_acc_depth[d]; ++d) pr.push_back(round3(dcost_.acc_depth[d]));
        dp.set("engine_acceptance", std::move(pr));  // the controller's prior
      }
      t.set("depth", std::move(dp));
    }
    if (s.sib_opp[0]) {
      json::Value sb = json::Value::object(), op = json::Value::array(), hi = json::Value::array();
      for (int d = 0; d < 3; ++d) {
        op.push_back(int64_t(s.sib_opp[d]));
        hi.push_back(int64_t(s.sib_hit[d]));
      }
      sb.set("misses", std::move(op));
      sb.set("runner_up", std::move(hi));
      sb.set("taken", int64_t(s.sib_taken));
      t.set("siblings", std::move(sb));
    }
    t.set("gpu_ms", round3(s.gpu_ms));
    if (s.gm) t.set("structured", s.gm->stats());  // masks computed / cached and their time
    if (!s.req.forced.empty()) {
      json::Value sc = json::Value::object(), lp = json::Value::array(), am = json::Value::array(),
                  tg = json::Value::array(), tops = json::Value::array();
      int match = 0;
      double nll = 0;
      for (const auto& r : s.scored) {
        tg.push_back(int64_t(r.target));
        am.push_back(int64_t(r.argmax));
        lp.push_back(double(r.logprob));
        match += r.argmax == r.target;
        nll -= double(r.logprob);
        if (s.req.score_topk > 0) {
          json::Value row = json::Value::array();
          for (const auto& [id, l] : r.top) {
            json::Value pr = json::Value::array();
            pr.push_back(int64_t(id));
            pr.push_back(double(l));
            row.push_back(std::move(pr));
          }
          tops.push_back(std::move(row));
        }
      }
      const size_t n = s.scored.size();
      sc.set("n", int64_t(n));
      sc.set("argmax_match", n ? double(match) / double(n) : 0.0);
      sc.set("mean_nll", n ? nll / double(n) : 0.0);
      sc.set("tokens", std::move(tg));
      sc.set("argmax", std::move(am));
      sc.set("logprobs", std::move(lp));
      if (s.req.score_topk > 0) sc.set("top", std::move(tops));
      if (!s.req.score_ids.empty()) {
        json::Value il = json::Value::array();
        for (const auto& r : s.scored) {
          json::Value row = json::Value::array();
          for (float l : r.ids_lp) row.push_back(double(l));
          il.push_back(std::move(row));
        }
        sc.set("ids_logprobs", std::move(il));
      }
      t.set("score", std::move(sc));
    }
    s.out->timings = t;
    if (s.work >= 0) {
      model_->slot_free(s.work);
      pinned_.erase(std::remove(pinned_.begin(), pinned_.end(), s.work), pinned_.end());
      s.work = -1;
    }
    if (s.ckpt >= 0) {
      pinned_.erase(std::remove(pinned_.begin(), pinned_.end(), s.ckpt), pinned_.end());
      if (!s.saved_ckpt) {
        model_->slot_free(s.ckpt);
      }
      s.ckpt = -1;
    }
    if (s.imgckpt >= 0) {
      pinned_.erase(std::remove(pinned_.begin(), pinned_.end(), s.imgckpt), pinned_.end());
      if (!s.saved_img) model_->slot_free(s.imgckpt);
      s.imgckpt = -1;
    }
    if (s.sysckpt >= 0) {
      pinned_.erase(std::remove(pinned_.begin(), pinned_.end(), s.sysckpt), pinned_.end());
      if (!s.saved_sys) model_->slot_free(s.sysckpt);
      s.sysckpt = -1;
    }
    if (s.prectk >= 0) {
      pinned_.erase(std::remove(pinned_.begin(), pinned_.end(), s.prectk), pinned_.end());
      if (!s.saved_pre) model_->slot_free(s.prectk);
      s.prectk = -1;
    }
    // A sequence that ends before its first chunk ran (an error in that pass)
    // still holds its parent's pin: release it, or the parent is never evicted.
    if (s.parent >= 0) {
      auto it = std::find(pinned_.begin(), pinned_.end(), s.parent);
      if (it != pinned_.end()) pinned_.erase(it);
      s.parent = -1;
    }
    // Its checkpoints were last used now: a flow's next request about the
    // same photo arrives a moment later and finds them still cached.
    if (opt_.keep_live)
      for (const std::string& k : s.keep) cache_find(k);
    gen_tokens_ += uint64_t(s.generated);
    s.out->finish(reason, err);
  };
  for (auto it = active_.begin(); it != active_.end();) {
    if ((*it)->out->cancelled()) {
      finish_seq(**it, "cancelled");
      it = active_.erase(it);
    } else {
      ++it;
    }
  }

  // A chat checkpoint that is a strict prefix of the prompt, with the same images inside it.
  auto on_path = [&](const GenRequest& r, const CacheEntry& e) {
    if (e.ns != "chat" || e.tokens.size() >= r.prompt.size()) return false;
    if (!std::equal(e.tokens.begin(), e.tokens.end(), r.prompt.begin())) return false;
    size_t inside = 0;
    for (int row : r.image_rows)
      if (size_t(row) < e.tokens.size()) ++inside;
    if (inside != e.digests.size()) return false;
    for (size_t i = 0; i < inside; ++i)
      if (e.digests[i] != r.images[i].digest) return false;
    return true;
  };
  // Longest cached chat checkpoint that is a strict prefix of the prompt,
  // longer than `longer_than` tokens, with the same images inside it.
  auto best_chat_ckpt = [&](const GenRequest& r, bool want_mtp, size_t longer_than) -> CacheEntry* {
    CacheEntry* best = nullptr;
    for (auto& [k, e] : cache_) {
      if (want_mtp && !e.mtp) continue;
      if (e.tokens.size() <= longer_than || (best && e.tokens.size() <= best->tokens.size())) continue;
      if (on_path(r, e)) best = &e;
    }
    return best;
  };

  // ---- admit waiting sequences
  std::vector<std::shared_ptr<DecisionJob>> jobs;
  {
    std::lock_guard<std::mutex> lk(mu_);
    while (!waiting_.empty() && int(active_.size()) < opt_.max_seqs) {
      auto s = waiting_.front();
      if (s->out->cancelled()) {
        waiting_.pop_front();
        s->out->finish("cancelled");
        continue;
      }
      // Longest cached chat checkpoint that is a strict prefix of the prompt.
      const bool want_mtp = I.mtp && opt_.draft_tokens > 0 && s->req.speculate;
      CacheEntry* best = best_chat_ckpt(s->req, want_mtp, 0);
      // KV for the prompt's uncached part and the answer, with some headroom.
      // A cached prefix is shared, not copied: counting the whole prompt would
      // let only a few requests in at a time under a long shared prefix (tens
      // of thousands of tokens), the others waiting minutes for their first
      // token. A prefix still to be loaded from disk counts as uncached.
      // The answer (max_tokens, already cut to the context room by
      // generate, counted up to answer_reserve's cap) and what the running
      // sequences may still take are counted too. Weighing only part of the
      // new request's answer against the blocks free at that moment would let
      // running answers take the rest later, and the pass that ran out would
      // fail every request in it.
      // Checkpoints no running request uses are evicted (LRU) to make that
      // room; without that, once cached checkpoints held the pool, a request
      // would wait until nothing was active.
      // With nothing active the request is admitted whatever its need (it
      // could never be otherwise); a pass that runs out stops the newest.
      // ESPETERO_KV_RESERVE=0 (development, recovery tests): the plain
      // free-blocks check (no reservation, no eviction), so the pass-level
      // handling can be run.
      const bool reserve = kv_reserve();
      const int bt = I.block_tokens > 0 ? I.block_tokens : 64;
      const int uncached = int(s->req.prompt.size()) - (best ? int(best->tokens.size()) : 0);
      if (!reserve && !active_.empty() && model_->free_kv_tokens() < uncached + std::min(s->req.max_tokens, 1024) + 64 + bt)
        break;
      const int need = uncached + answer_reserve(s->req.max_tokens) + 2 * bt;
      if (reserve && !active_.empty()) {
        const int reserved = reserved_kv_tokens();
        std::vector<int> keep_best;
        if (best) keep_best.push_back(best->slot);
        // Checkpoints are evicted only when that makes the request fit: a
        // request that waits anyway leaves the cache alone, so the next request
        // with that prefix does not compute it again.
        if (model_->free_kv_tokens() - reserved < need &&
            model_->free_kv_tokens() + evictable_kv_tokens(keep_best, false) - reserved < need)
          break;
        while (model_->free_kv_tokens() - reserved < need) {
          const int victim = evict_lru(keep_best, false);
          if (victim < 0) break;
          model_->slot_free(victim);
        }
        if (model_->free_kv_tokens() - reserved < need) break;
      }
      if (!prefix_root_.empty()) {
        // Stored prefixes are text: valid up to the first image (a photo
        // request also loads the instructions before its image).
        const size_t limit = s->req.image_rows.empty() ? s->req.prompt.size() : size_t(s->req.image_rows[0]);
        // The restore may evict to make room: pin the memory hit meanwhile, or
        // `best` could point at an evicted entry.
        const int bslot = best ? best->slot : -1;
        if (bslot >= 0) pinned_.push_back(bslot);
        // A load is a full copy of its prefix (its own KV blocks): only when it
        // saves at least a 2048-row pass over the checkpoint in memory. Otherwise a
        // request whose instructions are on disk, just past a long system prompt
        // already in memory, would load a second copy of both (hundreds of KV
        // blocks) to save a few hundred rows.
        CacheEntry* disk = prefix_store_restore(s->req.prompt, best ? best->tokens.size() + 2047 : 0, limit, want_mtp);
        if (bslot >= 0) pinned_.erase(std::find(pinned_.begin(), pinned_.end(), bslot));
        if (disk) best = disk;
      }
      std::vector<int> avoid;
      if (best) avoid.push_back(best->slot);
      const int work = acquire_slot(avoid);
      if (work < 0) {
        if (active_.empty() && decisions_.empty()) {
          // Nothing running will free a slot: fail instead of waiting forever.
          waiting_.pop_front();
          s->out->finish("error", "no free state slot for this request (raise --slots)");
          continue;
        }
        break;
      }
      avoid.push_back(work);
      int ckpt = -1, sysckpt = -1, imgckpt = -1, prectk = -1;
      const int split = s->req.split;
      const int sys_split = s->req.sys_split;
      const int img_split = s->req.img_split;
      const int pre_split = s->req.pre_split;
      const int cached = best ? int(best->tokens.size()) : 0;
      // Checkpoint slots are optional: taken while free, in order of
      // value (system prompt, image, prompt end); a prompt without one is only
      // not cached at that boundary, so a server with few slots (--slots 4)
      // never waits for them.
      auto optional_slot = [&]() {
        const int sl = acquire_slot(avoid);
        if (sl >= 0) avoid.push_back(sl);
        return sl;
      };
      if (sys_split > cached && sys_split < int(s->req.prompt.size())) sysckpt = optional_slot();
      if (opt_.instr_checkpoint && pre_split > cached && pre_split < int(s->req.prompt.size())) prectk = optional_slot();
      if (opt_.image_checkpoint && img_split > cached && img_split < int(s->req.prompt.size())) imgckpt = optional_slot();
      const bool keep_prompt = s->req.cache_prompt < 0 ? opt_.prompt_checkpoint : s->req.cache_prompt > 0;
      if (keep_prompt && split > cached && split <= int(s->req.prompt.size())) ckpt = optional_slot();
      waiting_.pop_front();
      s->work = work;
      s->ckpt = ckpt;
      s->sysckpt = sysckpt;
      s->imgckpt = imgckpt;
      s->prectk = prectk;
      s->parent = best ? best->slot : -1;
      // Every cached checkpoint on the prompt's path is kept while
      // the request runs, not only the one it continues: a request restored
      // from its image's checkpoint still relies on the shared system prompt
      // and instructions under it. A flow that lost its long shared prefix
      // under slot pressure would restore or recompute it for every new image.
      // (only those it can continue: a request that drafts cannot use a
      // checkpoint without the MTP state, which then stays plain LRU.)
      for (auto& [k, e] : cache_)
        if (on_path(s->req, e) && (!want_mtp || e.mtp)) s->keep.push_back(k);
      s->chain_key = best ? best->key : std::string();
      s->cached = cached;
      s->done = cached;
      s->mtp = want_mtp;
      if (best) best->used = ++clock_;
      pinned_.push_back(work);
      if (ckpt >= 0) pinned_.push_back(ckpt);
      if (sysckpt >= 0) pinned_.push_back(sysckpt);
      if (imgckpt >= 0) pinned_.push_back(imgckpt);
      if (prectk >= 0) pinned_.push_back(prectk);
      if (s->parent >= 0) pinned_.push_back(s->parent);  // until the first chunk runs
      if (s->req.sampling.penalized()) s->pen.add_prompt(s->req.prompt.data(), s->req.prompt.size());
      if (opt_.prompt_lookup) s->ngram.extend(s->req.prompt.data(), s->req.prompt.size());
      active_.push_back(s);
    }
    // Decision jobs (FIFO) within the pass budget.
    // Shared prefixes: a job whose prefix is cached adds only its stem
    // and leaves to the pass; a job whose prefix another job of this pass is
    // computing (and will checkpoint) waits for the next pass instead of
    // computing it again.
    int rows = 0, leaves = 0;
    for (auto it = decisions_.begin(); it != decisions_.end() && int(jobs.size()) < opt_.decision_batch;) {
      auto& j = *it;
      int r = int(j->plan.prefix.size() + j->plan.stem.size());
      if (opt_.share_prefix) {
        if (cache_.count(j->key)) {
          r = int(j->plan.stem.size());
        } else if (std::any_of(jobs.begin(), jobs.end(), [&](const auto& o) { return o->key == j->key; })) {
          ++it;
          continue;
        }
      }
      int lr = 0;
      for (const auto& l : j->plan.leaves) lr += int(l.size());
      if (!jobs.empty() && (rows + r + lr > opt_.decision_rows || leaves + int(j->plan.leaves.size()) > max_readouts / 2)) break;
      rows += r + lr;
      leaves += int(j->plan.leaves.size());
      jobs.push_back(j);
      it = decisions_.erase(it);
    }
  }
  if (jobs.empty() && active_.empty()) return;

  // ---- drafts for decoding sequences
  std::vector<Seq*> decoding, prefilling;
  for (auto& s : active_) (s->phase == Seq::Decode ? decoding : prefilling).push_back(s.get());
  const bool speculate = opt_.draft_tokens > 0 && int(decoding.size()) <= opt_.spec_max_seqs;
  {
    std::vector<DraftRequest> reqs;
    std::vector<Seq*> who;
    std::vector<std::vector<int32_t>> outs, outs2;
    const bool adapt = opt_.adaptive_drafts && decoding.size() == 1 && jobs.empty() && prefilling.empty();
    for (Seq* s : decoding) {
      s->drafts.clear();
      s->drafts2.clear();
      s->sib = 0;
      s->from_lookup = false;
      if (!s->req.forced.empty()) {
        // Teacher-forced scoring: the next forced tokens are the drafts,
        // so one pass reads out every one of their positions.
        const int n = int(s->req.forced.size()), g0 = s->generated;
        const int k = std::max(0, std::min(8 - int(s->pending.size()), n - g0));
        for (int j = 0; j < k; ++j) s->drafts.push_back(s->req.forced[size_t(g0 + j)]);
        continue;
      }
      if (!speculate || !s->req.speculate) continue;
      if (adapt) {
        // Plain steps: a few once, to know what a plain step costs.
        if (dcost_.n_plain < 3 && s->generated >= 1) continue;
        if (s->skip_look > 0) --s->skip_look;
        if (s->skip_mtp > 0) --s->skip_mtp;
      }
      const int ctx_room = I.max_ctx - int(s->req.prompt.size()) - s->generated;
      const int room = std::min(s->req.max_tokens - s->generated - 1, ctx_room - 1);
      const int budget = std::min(room, 8 - int(s->pending.size()));
      if (budget <= 0) continue;
      if (opt_.prompt_lookup && !(adapt && s->skip_look > 0)) {
        std::vector<int32_t> look = s->ngram.lookup(budget, opt_.lookup_min);
        if (!look.empty()) {
          s->drafts = std::move(look);
          s->from_lookup = true;
          ++s->lookups;
          continue;
        }
      }
      if (s->mtp && model_->slot_mtp(s->work) && !(adapt && (s->skip_mtp > 0 || s->mtp_lost_run >= kMtpGiveUp))) {
        const int k = (opt_.draft_auto && adapt) ? choose_depth(*s, budget) : std::min(opt_.draft_tokens, budget);
        if (k <= 0) continue;
        who.push_back(s);
        outs.emplace_back(size_t(k));
        outs2.emplace_back(size_t(k));
      }
    }
    for (size_t i = 0; i < who.size(); ++i) {
      DraftRequest q;
      q.slot = who[i]->work;
      q.tokens = who[i]->pending.data();
      q.n_tokens = int(who[i]->pending.size());
      q.n_draft = int(outs[i].size());
      q.out = outs[i].data();
      q.out2 = outs2[i].data();
      q.lh_first = who[i]->pending.size() > 1 ? who[i]->lh_first : 0;
      // A sampled request's drafts are drawn with its own keyed noise at
      // the positions the verify pass will draw them (its draw_pos0 onwards).
      const SamplingParams& sp = who[i]->req.sampling;
      // (constrained requests too: their masked draw is this one whenever the mask keeps the head.)
      if (!sp.greedy() && gpu_draw_ok(opt_, sp)) {
        q.draw = true;
        q.seed = who[i]->seed;
        q.pos0 = int64_t(who[i]->req.prompt.size()) + int64_t(who[i]->generated);
        q.temperature = float(sp.temperature);
        q.top_p = float(sp.top_p);
        q.min_p = float(sp.min_p);
        q.top_k = sp.top_k;
      }
      reqs.push_back(q);
    }
    if (!reqs.empty()) {
      try {
        PassStats ds;
        model_->draft(reqs, &ds);
        gpu_ms_ = gpu_ms_ + ds.gpu_ms;
        if (adapt) {
          ewma(dcost_.draft, dcost_.n_draft, ds.wall_ms);
          if (who.size() == 1 && outs[0].size() <= 8) ewma(dcost_.draftk[outs[0].size()], dcost_.n_draftk[outs[0].size()], ds.wall_ms);
        }
        if (trace) std::fprintf(stderr, "[trace] draft seqs=%zu wall %.1f ms\n", reqs.size(), ds.wall_ms);
        for (size_t i = 0; i < who.size(); ++i) {
          Seq* w = who[i];
          w->drafts = outs[i];
          w->drafts2 = outs2[i];
          // Sibling leaves in the rows the MMA verify pass pays for anyway (<= 8).
          const int k = int(w->drafts.size());
          const int free_rows = 8 - int(w->pending.size()) - k;
          const bool eq_verified = w->req.sampling.greedy() || opt_.exact_sampling;
          if (opt_.sibling_drafts >= 1 && eq_verified && k >= 1 && free_rows >= 1) w->sib = 1;
          if (opt_.sibling_drafts >= 2 && eq_verified && k >= 2 && free_rows >= 3) w->sib = 2;
        }
      } catch (const std::exception& e) {
        // Drafting is an optimization: decode plainly this step.
        if (opt_.verbose) std::fprintf(stderr, "[espetero] draft failed: %s\n", e.what());
      }
    }
  }

  // A constrained sequence's drafts stop before the first one its
  // grammar rejects (it could never be accepted: verify picks under the mask).
  for (Seq* s : decoding)
    if (s->gm && !s->drafts.empty() && s->req.forced.empty()) {
      const size_t keep = s->gm->valid_prefix(s->drafts.data(), s->drafts.size());
      if (keep < s->drafts.size()) {
        s->drafts.resize(keep);
        if (s->drafts2.size() > keep) s->drafts2.resize(keep);
        s->sib = std::min(s->sib, int(keep));
      }
    }

  // ---- decode chains: when every decoding sequence is greedy, has no
  // drafts this step and one pending token, and nothing else needs the pass,
  // run K plain decode steps as one GPU chain (each step embeds the previous
  // step's argmax on the GPU; one CPU wait per chain). Tokens past an end
  // token or the length limit are computed and discarded.
  // Prompts that decode-first holds back this step do not block a
  // chain: its K steps cost the waiting prompt at most one chain's time.
  bool prompts_due = false;
  for (Seq* s : prefilling)
    prompts_due = prompts_due || !opt_.decode_first || decoding.empty() || ms_since(s->t_wait) >= double(opt_.prefill_wait_ms);
  if (opt_.chain_steps > 1 && jobs.empty() && !prompts_due && !decoding.empty()) {
    bool ok = true;
    int K = 1;
    for (Seq* s : decoding) {
      const int ctx_room = I.max_ctx - int(s->req.prompt.size()) - s->generated;
      // Only after chain_after generated tokens: a chain delays the end of an
      // answer by up to K - 1 steps, which short outputs (labels, decisions)
      // must not pay.
      // A chain would leave the MTP head's state behind: only once the head
      // has stopped for the request (kMtpGiveUp losses in a row).
      const bool mtp_live = speculate && s->req.speculate && s->mtp && s->mtp_lost_run < kMtpGiveUp && model_->slot_mtp(s->work);
      ok = ok && s->drafts.empty() && s->req.forced.empty() && !s->req.constraint /* structured output */ && !mtp_live &&
           (s->req.sampling.greedy() || gpu_draw_ok(opt_, s->req.sampling)) &&
           s->req.logprobs < 0 &&
           s->pending.size() == 1 && !s->out->cancelled() && ctx_room > opt_.chain_steps + 1 &&
           s->generated >= opt_.chain_after;
      K = std::max(K, std::min(opt_.chain_steps, s->req.max_tokens - s->generated));
    }
    if (ok && K > 1) {
      // KV for all K steps before the chain starts. Its passes are
      // planned while the earlier ones run and take blocks as they go: one that
      // ran out failed the chain after the slots had advanced, and every
      // decoding sequence with it. Each step takes at most one new block per
      // sequence: the block its row opens, or a copy of a shared partial one
      // (at the first step only). Together that is what one K-row append to the
      // slot takes, counted as the pass counts it.
      // Checkpoints no running request uses are evicted for it; if that is not
      // enough, this step is a plain pass (which evicts more, or stops the
      // newest sequences).
      int64_t need = 0;
      const std::vector<int32_t> k_rows(size_t(K), 0);  // pass_kv_tokens reads only the length
      for (Seq* s : decoding) {
        TreeSpec t;
        t.parent = t.save = s->work;
        t.root = k_rows.data();
        t.root_len = K;
        need += model_->pass_kv_tokens({t});
      }
      // Only when evicting makes the chain fit; else the plain pass
      // runs without losing any checkpoint for it.
      if (need > model_->free_kv_tokens() && need <= int64_t(model_->free_kv_tokens()) + evictable_kv_tokens({}, false))
        while (need > model_->free_kv_tokens()) {
          const int victim = evict_lru({}, false);
          if (victim < 0) break;
          model_->slot_free(victim);
        }
      ok = need <= model_->free_kv_tokens();
    }
    if (ok && K > 1) {
      std::vector<TreeSpec> ct;
      for (Seq* s : decoding) {
        TreeSpec t;
        t.parent = s->work;
        t.save = s->work;
        t.root = s->pending.data();
        t.root_len = 1;
        t.score_root = true;
        if (!s->req.sampling.greedy())  // drawn on the GPU; positions advance one a step inside run_chain
          set_gpu_sampling(t, opt_, s->req.sampling, s->seed, int64_t(s->req.prompt.size()) + int64_t(s->generated));
        ct.push_back(t);
      }
      const size_t n = decoding.size();
      std::vector<int32_t> out(size_t(K) * n);
      PassStats st;
      try {
        model_->run_chain(ct, K, out.data(), &st);
      } catch (const std::exception& e) {
        const std::string msg = e.what();
        for (Seq* s : decoding) {
          auto it = std::find_if(active_.begin(), active_.end(), [&](const auto& p) { return p.get() == s; });
          finish_seq(*s, "error", msg);
          if (it != active_.end()) active_.erase(it);
        }
        return;
      }
      passes_ += uint64_t(K);
      rows_ += uint64_t(st.rows);
      gpu_ms_ = gpu_ms_ + st.gpu_ms;
      if (trace)
        std::fprintf(stderr, "[trace] chain K=%d seqs=%zu wall %.1f ms gpu %.1f ms (%.1f ms/step) plain_ewma %.1f\n", K, n,
                     st.wall_ms, st.gpu_ms, st.wall_ms / K, dcost_.plain);
      for (size_t i = 0; i < n; ++i) {
        Seq* s = decoding[i];
        s->passes += K;
        s->gpu_ms += st.gpu_ms;
        // Plain stretches count steps (the step that started the chain already counted one).
        s->skip_mtp = std::max(0, s->skip_mtp - (K - 1));
        s->skip_look = std::max(0, s->skip_look - (K - 1));
        std::vector<int32_t> emit;
        std::string reason;
        int32_t last = -1;
        for (int j = 0; j < K; ++j) {
          const int32_t t = out[size_t(j) * n + i];
          last = t;
          if (!s->req.ignore_eos && (std::find(end_ids.begin(), end_ids.end(), t) != end_ids.end() ||
                                     std::find(s->req.stop_ids.begin(), s->req.stop_ids.end(), t) != s->req.stop_ids.end())) {
            reason = "stop";
            break;
          }
          emit.push_back(t);
          ++s->generated;
          if (opt_.prompt_lookup) s->ngram.append(t);
          if (s->generated >= s->req.max_tokens) {
            reason = "length";
            break;
          }
        }
        s->out->push(emit.data(), emit.size());
        s->pending.assign(1, last);
        if (!reason.empty()) s->pending.clear();
        if (!reason.empty() || s->out->cancelled()) {
          auto it = std::find_if(active_.begin(), active_.end(), [&](const auto& p) { return p.get() == s; });
          finish_seq(*s, reason.empty() ? "cancelled" : reason);
          if (it != active_.end()) active_.erase(it);
        }
      }
      return;
    }
  }

  // ---- forest
  std::vector<TreeSpec> trees;
  std::vector<std::pair<char, void*>> owners;  // 'd' decision job, 's' sequence
  int rows = 0, readouts = 0;
  auto use_header_for = [&](const DecisionJob& j) {
    const DecisionPlan& p = j.plan;
    bool use = !j.req.has_salt && !header_.empty() && p.prefix.size() >= header_.size() &&
               std::equal(header_.begin(), header_.end(), p.prefix.begin());
    if (use && !j.images.empty()) {
      const auto first = std::find(p.prefix.begin(), p.prefix.end(), image_token);
      use = size_t(first - p.prefix.begin()) >= header_.size();
    }
    return use;
  };
  // A decision job's rows in this pass: its uncached prefix (less the
  // shared header checkpoint, when it continues that), the stem and the leaves.
  auto job_rows = [&](const DecisionJob& j, bool hit) {
    const DecisionPlan& p = j.plan;
    int r = int(p.stem.size());
    if (!hit) r += int(p.prefix.size()) - (use_header_for(j) ? int(header_.size()) : 0);
    for (const auto& l : p.leaves) r += int(l.size());
    return r;
  };
  // Decode steps go first, and a decision job whose tree does not fit the
  // rows and readouts they leave waits for a later pass, but not without
  // bound: a job put back kDecisionAging times reserves its tree in this pass: the newest decoding sequences' drafts, then their
  // decode steps, wait for the next pass until it fits. Drafts are verified
  // and a step that waits runs unchanged one pass later, so no reply changes;
  // the sequences held back lose that pass.
  std::vector<Seq*> held;
  if (!jobs.empty() && jobs.front()->waits >= kDecisionAging) {
    const DecisionJob& j = *jobs.front();
    const int jr = job_rows(j, cache_.count(j.key) > 0), jn = int(j.plan.leaves.size());
    auto seq_rows = [](const Seq* s) {  // as the trees below count them
      const int k = int(s->drafts.size());
      return int(s->pending.size()) + k * (k + 1) / 2 + (s->sib >= 1 ? 1 : 0) + (s->sib >= 2 ? 2 : 0);
    };
    auto seq_reads = [](const Seq* s) { return 1 + int(s->drafts.size()) + std::min(s->sib, 2); };
    int dr = 0, dn = 0;
    for (const Seq* s : decoding) dr += seq_rows(s), dn += seq_reads(s);
    auto short_of = [&] { return dr + jr > max_rows || dn + jn > max_readouts; };
    if (jr <= max_rows && jn <= max_readouts && short_of()) {
      ++decision_reservations_;
      for (auto it = decoding.rbegin(); it != decoding.rend() && short_of(); ++it) {
        Seq* s = *it;
        if (s->drafts.empty()) continue;
        dr -= seq_rows(s), dn -= seq_reads(s);
        if (s->from_lookup) --s->lookups;
        s->drafts.clear();
        s->drafts2.clear();
        s->sib = 0;
        s->from_lookup = false;
        dr += seq_rows(s), dn += seq_reads(s);
      }
      for (auto it = decoding.rbegin(); it != decoding.rend() && short_of(); ++it) {
        held.push_back(*it);
        dr -= seq_rows(*it), dn -= seq_reads(*it);
      }
    }
  }
  // Decode steps first (latency).
  for (Seq* s : decoding) {
    if (std::find(held.begin(), held.end(), s) != held.end()) continue;
    s->root = s->pending;
    s->leaf_off.assign(1, 0);
    s->leaf_tok.clear();
    for (size_t i = 0; i < s->drafts.size(); ++i) {
      s->leaf_tok.insert(s->leaf_tok.end(), s->drafts.begin(), s->drafts.begin() + long(i) + 1);
      s->leaf_off.push_back(int32_t(s->leaf_tok.size()));
    }
    // Sibling leaves: [s1], then [d1, s2]; each hosts its own rows after the chain's.
    if (s->sib >= 1) {
      s->leaf_tok.push_back(s->drafts2[0]);
      s->leaf_off.push_back(int32_t(s->leaf_tok.size()));
    }
    if (s->sib >= 2) {
      s->leaf_tok.push_back(s->drafts[0]);
      s->leaf_tok.push_back(s->drafts2[1]);
      s->leaf_off.push_back(int32_t(s->leaf_tok.size()));
    }
    const int nr = int(s->leaf_off.size());  // leaves + the root
    s->arg.assign(size_t(nr), -1);
    TreeSpec t;
    t.parent = s->work;
    t.save = s->work;
    t.root = s->root.data();
    t.root_len = int(s->root.size());
    t.n_leaves = nr - 1;
    t.leaf_offsets = s->leaf_off.data();
    t.leaf_tokens = s->leaf_tok.data();
    t.score_root = true;
    t.mtp = s->mtp;
    // A single chain of drafts (no sibling leaves, not teacher-forced
    // scoring) is committed in place after the pass instead of re-run as roots.
    s->commit_k = (I.commit && !s->drafts.empty() && s->sib == 0 && s->req.forced.empty()) ? int(s->drafts.size()) : 0;
    t.commit_chain = s->commit_k;
    t.out_argmax = s->arg.data();
    t.want_logits = !s->req.sampling.greedy() || !s->req.forced.empty();
    set_gpu_sampling(t, opt_, s->req.sampling, s->seed, int64_t(s->req.prompt.size()) + int64_t(s->generated));
    set_logprob_readouts(t, s->req, s->lp_lse, nr);
    set_constrained_readouts(t, s->req);
    s->tree = int(trees.size());
    trees.push_back(t);
    owners.push_back({'s', s});
    rows += int(s->root.size() + s->leaf_tok.size());
    readouts += nr;
  }
  // Decisions.
  // A job joins the pass only if its tree fits the rows and readouts the
  // decode steps left; the others wait for the next pass (in order, at the
  // front of the queue, counting the passes they waited), so a job larger
  // than what was left cannot fail the pass and every sequence in it. A tree
  // larger than any pass (an uncached prefix too long for --max-rows) fails
  // alone.
  {
    std::vector<std::shared_ptr<DecisionJob>> fit, later;
    std::vector<std::string> computing_keys;
    int r_rows = rows, r_readouts = readouts;
    for (auto& j : jobs) {
      const DecisionPlan& p = j->plan;
      const bool hit = cache_.count(j->key) &&
                       std::find(computing_keys.begin(), computing_keys.end(), j->key) == computing_keys.end();
      const int jr = job_rows(*j, hit);
      const int jn = int(p.leaves.size());
      if (jr > max_rows || jn > max_readouts) {
        std::lock_guard<std::mutex> lk(j->mu);
        j->error = "the decision needs " + std::to_string(jr) + " rows in one pass (its context, images and fields), "
                   "more than the " + std::to_string(max_rows) + " of this server (--max-rows): shorten the context "
                   "or the fields, or start the server with --max-rows " + std::to_string(jr) + " or more";
        j->error_status = 422;
        j->done = true;
        j->cv.notify_all();
      } else if (r_rows + jr <= max_rows && r_readouts + jn <= max_readouts) {
        r_rows += jr;
        r_readouts += jn;
        if (!hit) computing_keys.push_back(j->key);
        fit.push_back(j);
      } else {
        ++j->waits;
        later.push_back(j);
      }
    }
    if (!later.empty()) {
      std::lock_guard<std::mutex> lk(mu_);
      decisions_.insert(decisions_.begin(), later.begin(), later.end());
    }
    jobs = std::move(fit);
  }
  std::vector<std::string> reserved_keys;
  std::vector<int> pass_parents;
  // The cached prefixes of the jobs in this pass: a save slot taken for an
  // earlier job must not evict them (the later job would then compute
  // its whole prefix, more rows than counted above).
  for (auto& j : jobs)
    if (auto it = cache_.find(j->key); it != cache_.end()) pass_parents.push_back(it->second.slot);
  for (auto& j : jobs) {
    DecisionPlan& p = j->plan;
    CacheEntry* hit = cache_find(j->key);
    bool reserved_here = std::find(reserved_keys.begin(), reserved_keys.end(), j->key) != reserved_keys.end();
    int parent = -1, save = -1, save_len = 0;
    if (hit && !reserved_here) {
      j->cache_state = "hit";
      j->cached = int(p.prefix.size());
      j->root = p.stem;
      parent = hit->slot;
    } else {
      const bool use_header = use_header_for(*j);
      parent = use_header ? header_slot_ : -1;
      j->cached = use_header ? int(header_.size()) : 0;
      j->cache_state = use_header ? "header" : "miss";
      j->root.assign(p.prefix.begin() + j->cached, p.prefix.end());
      const size_t uncached = j->root.size();
      j->root.insert(j->root.end(), p.stem.begin(), p.stem.end());
      if (uncached && !reserved_here) {
        std::vector<int> avoid = pass_parents;
        if (parent >= 0) avoid.push_back(parent);
        save = acquire_slot(avoid);
        if (save >= 0) {
          save_len = int(uncached);
          reserved_keys.push_back(j->key);
          CacheEntry e;
          e.ns = "dec";
          e.tokens = p.prefix;
          e.slot = save;
          cache_put(j->key, e);
          pinned_.push_back(save);
        }
      }
    }
    if (parent >= 0) pass_parents.push_back(parent);
    j->leaf_off.assign(1, 0);
    j->leaf_tok.clear();
    j->cand_off.assign(1, 0);
    j->cand_ids.clear();
    for (size_t i = 0; i < p.leaves.size(); ++i) {
      j->leaf_tok.insert(j->leaf_tok.end(), p.leaves[i].begin(), p.leaves[i].end());
      j->leaf_off.push_back(int32_t(j->leaf_tok.size()));
      for (size_t c = 0; c < j->req.fields[i].n_values(); ++c) j->cand_ids.push_back(dp_->labels[c].token);
      j->cand_off.push_back(int32_t(j->cand_ids.size()));
    }
    j->cand_logits.assign(j->cand_ids.size(), 0);
    j->lse.assign(p.leaves.size(), 0);
    TreeSpec t;
    t.parent = parent;
    t.save = save;
    t.save_len = save_len;
    t.root = j->root.data();
    t.root_len = int(j->root.size());
    t.n_leaves = int(p.leaves.size());
    t.leaf_offsets = j->leaf_off.data();
    t.leaf_tokens = j->leaf_tok.data();
    t.cand_offsets = j->cand_off.data();
    t.cand_ids = j->cand_ids.data();
    t.out_cand_logits = j->cand_logits.data();
    t.out_lse = j->lse.data();
    if (!j->positions.empty()) {
      // Explicit M-RoPE positions for the root; images only when their rows
      // are computed here (a checkpoint hit already holds them).
      const size_t start = size_t(j->cached);
      j->root_pos.assign(j->positions.begin() + long(start * 3), j->positions.begin() + long((start + j->root.size()) * 3));
      t.root_pos = j->root_pos.data();
      if (j->cache_state != "hit") {
        int g = 0;
        j->irefs.clear();
        for (size_t i = 0; i < j->root.size();) {
          if (j->root[i] == image_token) {
            const PreparedImage& im = j->images[size_t(g++)];
            ImageRef ref;
            ref.pixels = im.pixels.data();
            ref.grid_t = im.grid_t;
            ref.grid_h = im.grid_h;
            ref.grid_w = im.grid_w;
            ref.row = int(i);
            j->irefs.push_back(ref);
            i += size_t(im.tokens(I.vision_merge));
          } else {
            ++i;
          }
        }
        t.images = j->irefs.data();
        t.n_images = int(j->irefs.size());
      }
    }
    j->rows = int(j->root.size() + j->leaf_tok.size());
    j->batched = int(jobs.size());
    trees.push_back(t);
    owners.push_back({'d', j.get()});
    rows += j->rows;
    readouts += t.n_leaves;
  }
  // Prefill chunks (FIFO) in the remaining row budget. Decode-first:
  // a pass that runs a big prompt chunk takes seconds on the 27B (~14 s for
  // 2,048 rows), and every decoding sequence would wait for it on every step.
  // So while sequences decode, a prompt's next chunk waits up to
  // prefill_wait_ms for them to finish, and prompt rows stop once a pass holds
  // prefill_batch_rows of them: big prompts go one at a time (the shortest
  // remaining work finishes first), small ones still share a pass.
  int prefill_rows = 0;
  // Single-flight shared prefixes: the system / instruction prefixes
  // some started request is computing (it holds the checkpoint slot that will
  // cache them). A prompt that has not started and shares one of them waits
  // for that checkpoint (then continues from it by the late upgrade below)
  // instead of computing the same rows again in parallel: a batch of image
  // requests under a long shared prompt would otherwise prefill it several times.
  std::vector<std::pair<uint64_t, int>> computing;
  auto add_computing = [&](const Seq* q) {
    if (q->sysckpt >= 0 && !q->saved_sys && q->req.sys_split > q->done) computing.push_back({q->req.sys_hash, q->req.sys_split});
    if (q->prectk >= 0 && !q->saved_pre && q->req.pre_split > q->done) computing.push_back({q->req.pre_hash, q->req.pre_split});
  };
  if (opt_.single_flight)
    for (Seq* q : prefilling)
      if (q->passes > 0) add_computing(q);
  auto shares_computing = [&](const Seq* q) {
    for (const auto& [h, n] : computing)
      if ((q->req.sys_split == n && q->req.sys_hash == h && n > q->cached) ||
          (q->req.pre_split == n && q->req.pre_hash == h && n > q->cached))
        return true;
    return false;
  };
  for (Seq* s : prefilling) {
    if (opt_.decode_first) {
      if (!decoding.empty() && ms_since(s->t_wait) < double(opt_.prefill_wait_ms)) continue;
      if (prefill_rows >= opt_.prefill_batch_rows) break;
    }
    if (s->done == s->cached && s->passes == 0) {
      // Late parent upgrade: a longer checkpoint appeared after this
      // prompt was admitted (another request with the same system prompt and
      // image finished its prefill): continue from it instead.
      if (CacheEntry* b = best_chat_ckpt(s->req, s->mtp, size_t(s->cached))) {
        if (s->parent >= 0) {
          auto it = std::find(pinned_.begin(), pinned_.end(), s->parent);
          if (it != pinned_.end()) pinned_.erase(it);
        }
        s->parent = b->slot;
        pinned_.push_back(s->parent);
        b->used = ++clock_;
        s->cached = s->done = int(b->tokens.size());
        auto drop = [&](int& slot, int pos) {
          if (slot < 0 || pos > s->cached) return;
          pinned_.erase(std::remove(pinned_.begin(), pinned_.end(), slot), pinned_.end());
          model_->slot_free(slot);
          slot = -1;
        };
        drop(s->sysckpt, s->req.sys_split);
        drop(s->prectk, s->req.pre_split);
        drop(s->imgckpt, s->req.img_split);
        drop(s->ckpt, s->req.split);
      }
      if (opt_.single_flight && shares_computing(s)) continue;
    }
    const int total = int(s->req.prompt.size());
    int budget = max_rows - rows;
    if (!decoding.empty() || !jobs.empty()) budget = std::min(budget, opt_.prefill_chunk);
    if (budget <= 0 || readouts + 1 > max_readouts) break;
    int end = std::min(total, s->done + budget);
    // Chunks never split an image run: a chunk that would cut one ends before
    // it; a chunk that starts at it takes the whole run only if it fits in the
    // rows left in this pass (or the pass is still empty), otherwise the prompt
    // waits for the next pass. (taking the run regardless can push batched
    // image passes past max_rows, and those generations would fail.)
    for (size_t g = 0; g < s->req.image_rows.size(); ++g) {
      const int r0 = s->req.image_rows[g], r1 = r0 + s->req.images[g].tokens(I.vision_merge);
      if (r0 < end && r1 > end) end = r0 > s->done ? r0 : (rows == 0 || r1 - s->done <= max_rows - rows ? r1 : s->done);
    }
    // Checkpoint boundaries still ahead, in prompt order (system message,
    // last image, prompt end). A chunk writes at most one checkpoint: the
    // first chunk into `save` (a prefix of its rows), later chunks as their
    // `work` copy when they end on it. So a chunk that would also reach a
    // second boundary (its last row included) is cut at the first one. (Comparing
    // the second boundary with `<` would let a first chunk holding the system
    // boundary and ending at the prompt end cache a prompt checkpoint that was
    // never written.)
    const bool first = s->done == s->cached && s->passes == 0;
    const std::pair<int, int> bounds[4] = {
        {s->req.sys_split, s->sysckpt}, {s->req.pre_split, s->prectk}, {s->req.img_split, s->imgckpt}, {s->req.split, s->ckpt}};
    int bpos = -1, bslot = -1, more = 0;
    for (const auto& [pos, slot] : bounds) {
      if (slot < 0 || pos <= s->done) continue;
      if (bpos < 0) bpos = pos, bslot = slot;
      else if (pos <= end) ++more;
    }
    if (bpos > 0 && bpos < end && (!first || more > 0)) end = bpos;
    if (end <= s->done) continue;
    if (opt_.single_flight && first) add_computing(s);  // later prompts in this pass wait for its prefixes
    prefill_rows += end - s->done;
    s->root.assign(s->req.prompt.begin() + s->done, s->req.prompt.begin() + end);
    s->final_chunk = end == total;
    s->chunk_end = end;
    TreeSpec t;
    if (first) {
      t.parent = s->parent;
      t.work = s->work;
      if (bslot >= 0 && bpos > s->done && bpos <= end) {
        t.save = bslot;
        t.save_len = bpos - s->done;
      }
    } else {
      t.parent = s->work;
      t.save = s->work;
      if (bslot >= 0 && bpos == end) t.work = bslot;
    }
    t.root = s->root.data();
    t.root_len = int(s->root.size());
    t.mtp = s->mtp;
    if (!s->req.positions.empty()) {
      s->root_pos.assign(s->req.positions.begin() + long(size_t(s->done) * 3), s->req.positions.begin() + long(size_t(end) * 3));
      t.root_pos = s->root_pos.data();
      s->irefs.clear();
      for (size_t g = 0; g < s->req.image_rows.size(); ++g) {
        const int r0 = s->req.image_rows[g];
        if (r0 < s->done || r0 >= end) continue;
        const PreparedImage& im = s->req.images[g];
        ImageRef ref;
        ref.pixels = im.pixels.data();
        ref.grid_t = im.grid_t;
        ref.grid_h = im.grid_h;
        ref.grid_w = im.grid_w;
        ref.row = r0 - s->done;
        s->irefs.push_back(ref);
      }
      t.images = s->irefs.data();
      t.n_images = int(s->irefs.size());
    }
    if (s->final_chunk) {
      t.score_root = true;
      s->arg.assign(1, -1);
      t.out_argmax = s->arg.data();
      t.want_logits = !s->req.sampling.greedy() || !s->req.forced.empty();
      set_gpu_sampling(t, opt_, s->req.sampling, s->seed, int64_t(s->req.prompt.size()));
      set_logprob_readouts(t, s->req, s->lp_lse, 1);
      set_constrained_readouts(t, s->req);
    }
    s->tree = int(trees.size());
    trees.push_back(t);
    owners.push_back({'s', s});
    rows += t.root_len;
    readouts += s->final_chunk ? 1 : 0;
  }
  if (trees.empty()) return;

  // ---- KV room for the pass: evict checkpoints nobody pins (LRU) rather than
  // fail every sequence of the pass with "KV cache full", never a parent
  // this pass reads (a decision's cached prefix is not pinned, and its
  // eviction under the pass failed it).
  auto stop_seq = [&](Seq* s, const std::string& why) {
    auto it = std::find_if(active_.begin(), active_.end(), [&](const auto& p) { return p.get() == s; });
    decoding.erase(std::remove(decoding.begin(), decoding.end(), s), decoding.end());
    prefilling.erase(std::remove(prefilling.begin(), prefilling.end(), s), prefilling.end());
    finish_seq(*s, "error", why);
    if (it != active_.end()) active_.erase(it);
  };
  // The message says what would fit: a request alone gets the cache's
  // size and, for its prompt, the KV free for it (the free list, the
  // checkpoints it could evict and its own blocks); a stopped answer, the
  // tokens it reached.
  auto kv_full_msg = [&](const Seq& s) {
    const std::string pool = std::to_string(int64_t(I.kv_blocks) * std::max(1, I.block_tokens));
    const std::string gen = std::to_string(s.generated);
    if (active_.size() <= 1) {
      if (s.phase == Seq::Prefill) {
        std::vector<int> own;
        for (int sl : {s.work, s.ckpt, s.sysckpt, s.imgckpt, s.prectk})
          if (sl >= 0) own.push_back(sl);
        const int64_t avail = int64_t(model_->free_kv_tokens()) + evictable_kv_tokens({}, true) +
                              (own.empty() ? 0 : model_->freed_kv_tokens(own));
        return "KV cache full: this request's prompt does not fit: it needs " +
               std::to_string(int64_t(s.req.prompt.size()) - s.cached) + " tokens of KV cache beyond its cached "
               "prefix, and about " + std::to_string(avail) + " are free for it with nothing else running (the "
               "cache holds " + pool + "): send a shorter prompt, or raise --kv-gb or --memory-gb";
      }
      return "KV cache full after " + gen + " tokens: the KV cache (" + pool + " tokens) holds this request's " +
             std::to_string(s.req.prompt.size()) + "-token prompt and " + gen + " tokens of its answer, no more: ask "
             "for at most " + gen + " max_tokens with this prompt, or raise --kv-gb or --memory-gb";
    }
    return "KV cache full: the running requests hold all of it (" + pool + " tokens); the newest ones are stopped "
           "(this one after " + gen + " tokens) so that earlier ones can finish (raise --kv-gb or --memory-gb, or send "
           "fewer requests at once)";
  };
  // A decision job leaves the pass: the checkpoint it reserved for its prefix
  // (never written now) is dropped.
  auto unreserve = [&](const DecisionJob& j, int save) {
    if (save < 0) return;
    auto it = cache_.find(j.key);
    if (it == cache_.end() || it->second.slot != save) return;
    pinned_.erase(std::remove(pinned_.begin(), pinned_.end(), save), pinned_.end());
    cache_drop(j.key);
    reserved_keys.erase(std::remove(reserved_keys.begin(), reserved_keys.end(), j.key), reserved_keys.end());
  };
  {
    std::vector<int64_t> tneed(trees.size(), 0);
    int64_t need = 0;
    for (size_t i = 0; i < trees.size(); ++i) need += tneed[i] = model_->pass_kv_tokens({trees[i]});
    std::vector<int> reads;
    for (const TreeSpec& t : trees)
      if (t.parent >= 0) reads.push_back(t.parent);
    // What evicting every checkpoint the pass does not read would give
    // back, counted before any goes. When even that cannot cover the pass,
    // work waits or stops first (below), and checkpoints are evicted only for
    // the shortage that is left, so no checkpoint is lost for work that
    // waits anyway.
    auto room = [&]() { return int64_t(model_->free_kv_tokens()) + evictable_kv_tokens(reads, true); };
    // The optional checkpoint a tree writes, if any: a prompt's first
    // chunk saves its first boundary (system message, instructions, image or
    // prompt end) into `save` beside its `work` slot, a later chunk ending on
    // one copies the work slot into `work`, a decision saves its prefix into
    // `save`. A first chunk pays its rows twice for it: a 2,320-token prompt
    // on a 4,096-token cache needed 74 of the 62 free blocks, 37 without it.
    auto ckpt_slot = [&](size_t i) -> int {
      const TreeSpec& t = trees[i];
      if (owners[i].first == 'd') return t.save;
      const Seq* q = static_cast<const Seq*>(owners[i].second);
      if (q->phase != Seq::Prefill) return -1;
      return t.save == q->work ? t.work : t.save;
    };
    auto without_ckpt = [&](size_t i) {
      TreeSpec t = trees[i];
      if (owners[i].first == 's' && t.save == static_cast<const Seq*>(owners[i].second)->work) {
        t.work = -1;
      } else {
        t.save = -1;
        t.save_len = 0;
      }
      return t;
    };
    auto ckpt_kv = [&](size_t i) -> int64_t {
      return ckpt_slot(i) >= 0 ? tneed[i] - model_->pass_kv_tokens({without_ckpt(i)}) : 0;
    };
    // The tree runs without it: its slot goes back (unpinned), the request
    // plans no checkpoint there and the decision no cache entry (a later
    // request computes those rows again); replies do not change.
    auto skip_ckpt = [&](size_t i) {
      const int sl = ckpt_slot(i);
      const TreeSpec t = without_ckpt(i);
      if (owners[i].first == 'd') {
        unreserve(*static_cast<DecisionJob*>(owners[i].second), sl);
      } else {
        Seq* q = static_cast<Seq*>(owners[i].second);
        for (int* f : {&q->sysckpt, &q->prectk, &q->imgckpt, &q->ckpt})
          if (*f == sl) *f = -1;
        pinned_.erase(std::remove(pinned_.begin(), pinned_.end(), sl), pinned_.end());
        model_->slot_free(sl);
      }
      trees[i] = t;
      const int64_t n = model_->pass_kv_tokens({t});
      need -= tneed[i] - n;
      tneed[i] = n;
      ++kv_skipped_;
    };
    // A decode step's commit chain is optional as well: its draft
    // rows take blocks past the slot's length before any is verified, and a
    // request alone on a full cache stopped there, up to a block early, though
    // its next token still fit. Without it the drafts are verified as plain
    // leaves and the accepted ones run as roots next pass (replies do not
    // change).
    auto chain_kv = [&](size_t i) -> int64_t {
      if (trees[i].commit_chain <= 0) return 0;
      TreeSpec t = trees[i];
      t.commit_chain = 0;
      return tneed[i] - model_->pass_kv_tokens({t});
    };
    auto skip_chain = [&](size_t i) {
      trees[i].commit_chain = 0;
      static_cast<Seq*>(owners[i].second)->commit_k = 0;
      const int64_t n = model_->pass_kv_tokens({trees[i]});
      need -= tneed[i] - n;
      tneed[i] = n;
    };
    if (need > model_->free_kv_tokens() && need > room()) {
      // Short even with every checkpoint evicted: the running sequences
      // hold the pool. Rather than fail every request in the pass, in this
      // order until the rest fits: the trees' optional work is skipped
      // (newest first, once that covers the shortage: commit chains, then
      // checkpoints, so a lone prompt runs without its checkpoint); prompt
      // chunks wait (newest
      // first) while a decoding sequence stays in the pass (it frees its
      // blocks when it ends); decision jobs wait (newest first) while any
      // other tree stays (their checkpoints can be evicted next pass); then
      // the newest decoding sequences stop with an error, then the newest
      // prompts, then the last decision job fails. Only trees that take KV
      // wait (a decision without its prefix checkpoint takes none).
      std::vector<char> drop(trees.size(), 0);  // 1: waits for a later pass, 2: fails
      auto seq_of = [&](size_t i) { return owners[i].first == 's' ? static_cast<Seq*>(owners[i].second) : nullptr; };
      auto newest = [&](auto pred) -> long {
        for (size_t i = trees.size(); i-- > 0;)
          if (!drop[i] && pred(i)) return long(i);
        return -1;
      };
      auto decode_tree = [&](size_t i) { return seq_of(i) && seq_of(i)->phase == Seq::Decode; };
      auto prompt_tree = [&](size_t i) { return seq_of(i) && seq_of(i)->phase == Seq::Prefill; };
      auto decision_tree = [&](size_t i) { return owners[i].first == 'd'; };
      while (need > room()) {
        int64_t skippable = 0;
        for (size_t i = 0; i < trees.size(); ++i)
          if (!drop[i]) skippable += chain_kv(i) + ckpt_kv(i);
        if (skippable > 0 && need - skippable <= room()) {
          // Commit chains first (they cost a pass's speed), then checkpoints
          // (they cost a later request a recompute); newest first.
          for (size_t i = trees.size(); i-- > 0 && need > room();)
            if (!drop[i] && chain_kv(i) > 0) skip_chain(i);
          for (size_t i = trees.size(); i-- > 0 && need > room();)
            if (!drop[i] && ckpt_kv(i) > 0) skip_ckpt(i);
          continue;
        }
        long pick = -1;
        char how = 1;
        const long dec = newest([&](size_t i) { return decision_tree(i) && tneed[i] > 0; });
        const long prompt = newest([&](size_t i) { return prompt_tree(i) && tneed[i] > 0; });
        const bool decoders = newest(decode_tree) >= 0;
        const bool others = newest([&](size_t i) { return long(i) != dec; }) >= 0;
        if (decoders && prompt >= 0) {
          pick = prompt;
        } else if (dec >= 0 && others) {
          pick = dec;
        } else {
          how = 2;  // a stopped sequence frees its blocks too
          pick = decoders ? newest(decode_tree) : newest(prompt_tree) >= 0 ? newest(prompt_tree) : dec;
        }
        if (pick < 0) break;
        drop[size_t(pick)] = how;
        need -= tneed[size_t(pick)];
        if (Seq* s = seq_of(size_t(pick))) {
          if (how == 1) {
            ++kv_deferred_;
          } else {
            ++kv_stopped_;
            stop_seq(s, kv_full_msg(*s));
          }
        } else {
          DecisionJob& j = *static_cast<DecisionJob*>(owners[size_t(pick)].second);
          unreserve(j, trees[size_t(pick)].save);
          if (how == 2) {
            std::lock_guard<std::mutex> lk(j.mu);
            j.error = "KV cache full: no room for this decision's prefix (raise --kv-gb or --memory-gb)";
            j.error_status = 503;
            j.done = true;
            j.cv.notify_all();
          }
        }
      }
      std::vector<TreeSpec> kt;
      std::vector<std::pair<char, void*>> ko;
      std::vector<std::shared_ptr<DecisionJob>> keep, later;
      for (size_t i = 0; i < trees.size(); ++i) {
        if (drop[i]) continue;
        if (Seq* s = seq_of(i)) s->tree = int(kt.size());
        kt.push_back(trees[i]);
        ko.push_back(owners[i]);
      }
      for (auto& j : jobs) {
        size_t i = 0;
        while (i < owners.size() && owners[i].second != j.get()) ++i;
        if (i == owners.size() || !drop[i]) keep.push_back(j);
        else if (drop[i] == 1) later.push_back(j);
      }
      if (!later.empty()) {
        std::lock_guard<std::mutex> lk(mu_);
        decisions_.insert(decisions_.begin(), later.begin(), later.end());
      }
      trees = std::move(kt);
      owners = std::move(ko);
      jobs = std::move(keep);
      if (trees.empty()) return;
    }
    // Checkpoints nobody pins, least recently used first, for what is
    // still short (never for work that waits or stops anyway).
    while (need > model_->free_kv_tokens()) {
      const int victim = evict_lru(reads);
      if (victim < 0) break;
      model_->slot_free(victim);
    }
  }

  // ---- run
  PassStats st;
  try {
    model_->run(trees, &st);
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    for (const std::string& k : reserved_keys) {
      auto it = cache_.find(k);
      if (it != cache_.end()) {
        pinned_.erase(std::remove(pinned_.begin(), pinned_.end(), it->second.slot), pinned_.end());
        cache_drop(k);
      }
    }
    // A pass refused for KV changed no slot (the check runs before
    // anything is written, and blocks taken are given back); the room made
    // above should prevent it. Only the newest sequence of the pass stops and
    // the decision jobs wait for the next pass, so the others go on (with no
    // sequence in the pass, the jobs fail below, as before).
    Seq* newest_seq = nullptr;
    if (msg.find("KV cache full") != std::string::npos)
      for (auto it = active_.rbegin(); it != active_.rend() && !newest_seq; ++it)
        if (std::any_of(owners.begin(), owners.end(), [&](const auto& o) { return o.second == it->get(); }))
          newest_seq = it->get();
    if (newest_seq) {
      {
        std::lock_guard<std::mutex> lk(mu_);
        decisions_.insert(decisions_.begin(), jobs.begin(), jobs.end());
      }
      ++kv_stopped_;
      std::fprintf(stderr, "[espetero] pass refused (%s): the newest sequence stops\n", msg.c_str());
      stop_seq(newest_seq, kv_full_msg(*newest_seq));
      return;
    }
    for (auto& j : jobs) {
      std::lock_guard<std::mutex> lk(j->mu);
      j->error = "Metal scoring failed: " + msg;
      j->error_status = 502;
      j->done = true;
      j->cv.notify_all();
    }
    // A GPU command buffer that failed (macOS aborts long ones while the
    // display is in use) leaves every slot as it was before the pass. A first
    // prompt chunk read a cached checkpoint and wrote only slots of its own:
    // it runs again, at most twice. Later chunks and decode steps update
    // their slot's recurrent state in place, which a half-run pass broke.
    const bool gpu = msg.find("GPU command buffer failed") != std::string::npos;
    int again = 0;
    for (auto it = active_.begin(); it != active_.end();) {
      Seq* s = it->get();
      bool in = false;
      for (auto& o : owners) in = in || (o.first == 's' && o.second == s);
      if (in && gpu && s->passes == 0 && s->retries < 2) {
        ++s->retries;
        ++again;
        ++it;
      } else if (in) {
        finish_seq(*s, "error", msg);
        it = active_.erase(it);
      } else {
        ++it;
      }
    }
    if (again) std::fprintf(stderr, "[espetero] pass failed (%s): %d prompt(s) run again\n", msg.c_str(), again);
    return;
  }
  ++passes_;
  if (xstream_) {  // expert-cache counters for /v1/stats (models that stream experts)
    std::string x = model_->bench_json("xcache");
    std::lock_guard<std::mutex> lk(xsnap_mu_);
    xsnap_ = std::move(x);
  }
  rows_ += uint64_t(st.rows);
  gpu_ms_ = gpu_ms_ + st.gpu_ms;
  for (const std::string& k : reserved_keys) {
    auto it = cache_.find(k);
    if (it != cache_.end()) pinned_.erase(std::remove(pinned_.begin(), pinned_.end(), it->second.slot), pinned_.end());
  }

  // ---- decisions done
  for (auto& j : jobs) {
    std::lock_guard<std::mutex> lk(j->mu);
    j->stats = st;
    j->done = true;
    j->cv.notify_all();
  }

  // ---- sequences: sample, verify drafts, stream tokens
  std::vector<Seq*> touched;
  for (auto& o : owners)
    if (o.first == 's') touched.push_back(static_cast<Seq*>(o.second));
  // Token selection (parallel across sequences: sampling touches ~1 MB of logits each).
  std::vector<std::vector<int32_t>> produced(touched.size());
  std::vector<std::vector<TokenLogprob>> produced_lp(touched.size());
  parallel(touched.size(), [&](size_t i) {
    Seq* s = touched[i];
    const int V = I.vocab;
    // A constrained sequence picks every token under its grammar's mask
    // (masked argmax, or the masked distribution), and its parse state takes
    // each token it produces. Once it has produced an end token (an accepted
    // draft, then the pass's bonus readout), the rest of the pass is
    // discarded at emission, as for any sequence: picked without the mask.
    grammar::Matcher* gm = nullptr;
    if (s->req.constraint) {
      if (!s->gm) s->gm = std::make_unique<grammar::Matcher>(s->req.constraint);
      gm = s->gm.get();
    }
    auto masked = [&]() { return gm && !gm->done(); };
    auto view = [&](int readout) {
      return readout_view(*model_, s->tree, readout, s->arg[size_t(readout)], opt_, s->req.sampling);
    };
    auto argm = [&](int readout) {
      return masked() ? grammar::masked_argmax(*gm, view(readout), s->req.sampling, &s->pen) : s->arg[size_t(readout)];
    };
    auto dist = [&](int readout) {
      if (masked()) return grammar::masked_distribution(*gm, view(readout), s->req.sampling, &s->pen);
      if (gpu_topk_ok(opt_, s->req.sampling)) {
        const TopK tk = model_->topk(s->tree, readout);
        if (tk.n >= s->req.sampling.top_k && tk.total > 0)
          return distribution_topk(tk.ids, tk.x, tk.n, tk.max, tk.total, s->req.sampling);
      }
      const float* lg = model_->logits(s->tree, readout);
      if (s->req.sampling.penalized() && !s->pen.empty()) {
        thread_local std::vector<float> buf;
        buf.assign(lg, lg + V);
        s->pen.apply(buf.data(), s->req.sampling);
        return distribution(buf.data(), V, s->req.sampling);
      }
      return distribution(lg, V, s->req.sampling);
    };
    const bool greedy = s->req.sampling.greedy();
    const bool exact = opt_.exact_sampling;
    const bool gdraw = !greedy && !gm && gpu_draw_ok(opt_, s->req.sampling);
    // The sampled token of a readout: the GPU's draw, else the CPU's exact draw.
    auto pick = [&](int readout, int64_t pos) -> int32_t {
      if (gdraw) {
        const TopK tk = model_->topk(s->tree, readout);
        if (tk.sampled >= 0) return tk.sampled;
      }
      return sample_at(dist(readout), s->seed, pos);
    };
    // Absolute position of the next produced token (exact sampling keys on it).
    const int64_t pos0 = int64_t(s->req.prompt.size()) + int64_t(s->generated);
    std::vector<int32_t>& out = produced[i];
    auto emit = [&](int32_t t) {
      if (masked() && (t < 0 || !gm->advance(t))) s->gm_error = true;
      out.push_back(t < 0 ? 0 : t);
    };
    // An emitted token's log-probability and the top alternatives,
    // from the GPU top-k of the readout that produced it (exact: its sum covers
    // the whole vocabulary) and that readout's logits.
    auto lp_rec = [&](int readout, int32_t tok) {
      if (s->req.logprobs < 0) return;
      const float* lse = s->lp_lse.empty() ? nullptr : &s->lp_lse[size_t(readout)];
      produced_lp[i].push_back(readout_logprob(s->tree, readout, tok, s->req.logprobs, lse));
    };
    // Penalties at a readout count the tokens produced before it in this
    // pass; they are added for good when emitted.
    PenaltyScope pscope(s->req.sampling.penalized() ? &s->pen : nullptr);
    if (!s->req.forced.empty()) {
      // Teacher-forced scoring: record the readout that predicts each
      // forced token, then produce the forced tokens themselves.
      auto record = [&](int readout, int32_t target) {
        const float* lg = model_->logits(s->tree, readout);
        int am = 0;
        float mx = lg[0];
        for (int v = 1; v < V; ++v)
          if (lg[v] > mx) mx = lg[v], am = v;
        double sum = 0;
        for (int v = 0; v < V; ++v) sum += std::exp(double(lg[v]) - double(mx));
        const double lse = double(mx) + std::log(sum);
        Seq::ScoreRec r;
        r.target = target;
        r.argmax = am;
        r.logprob = float(double(lg[target]) - lse);
        if (s->scored.size() < s->req.score_ids.size())
          for (int32_t id : s->req.score_ids[s->scored.size()]) r.ids_lp.push_back(float(double(lg[id]) - lse));
        if (s->req.score_topk > 0) {
          std::vector<int32_t> idx(static_cast<size_t>(V));
          for (int v = 0; v < V; ++v) idx[size_t(v)] = v;
          const int kk = std::min(s->req.score_topk, V);
          std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int32_t a, int32_t b) { return lg[a] > lg[b]; });
          for (int j = 0; j < kk; ++j) r.top.push_back({idx[size_t(j)], float(double(lg[idx[size_t(j)]]) - lse)});
        }
        s->scored.push_back(std::move(r));
      };
      const auto& f = s->req.forced;
      const int n = int(f.size()), g0 = s->generated;
      if (s->phase == Seq::Prefill) {
        if (!s->final_chunk) return;
        record(0, f[0]);
        out.push_back(f[0]);
        return;
      }
      // Readouts: the root predicts forced[g0], leaf d-1 forced[g0 + d].
      const int k = int(s->drafts.size());
      for (int d = 0; d <= k && g0 + d < n; ++d) {
        record(d == 0 ? k : d - 1, f[size_t(g0 + d)]);
        out.push_back(f[size_t(g0 + d)]);
      }
      return;
    }
    if (s->phase == Seq::Prefill) {
      if (!s->final_chunk) return;
      emit(greedy ? argm(0) : exact ? pick(0, pos0) : sample(dist(0), s->rng));
      lp_rec(0, out.back());
      return;
    }
    // Readouts: leaf i (after draft i) at i, then the sibling leaves, the root last.
    const int k = int(s->drafts.size());
    const int nl = int(s->leaf_off.size()) - 1;
    s->lh_first = 0;
    const int dsrc = s->from_lookup ? 1 : 0;
    for (int d = 0; d < k; ++d) {
      const int readout = d == 0 ? nl : d - 1;
      int32_t tok;
      bool ok;
      if (d < 8) ++s->dep_reach[dsrc][d];
      if (greedy) {
        tok = argm(readout);
        ok = tok == s->drafts[size_t(d)];
      } else if (exact) {
        tok = pick(readout, pos0 + d);
        ok = tok == s->drafts[size_t(d)];
      } else {
        std::tie(ok, tok) = verify_draft(dist(readout), s->drafts[size_t(d)], s->rng);
      }
      emit(tok);
      lp_rec(readout, tok);
      pscope.add(tok);  // the next depth, the sibling leaf or the bonus readout follows it
      if (ok && d < 8) ++s->dep_acc[dsrc][d];
      if (!ok) {
        // A miss at depth d: was it the MTP head's runner-up there (whether or
        // not its sibling leaf is in this pass)?
        if (d < 3 && d < int(s->drafts2.size()) && (greedy || exact)) {
          ++s->sib_opp[d];
          s->sib_hit[d] += tok == s->drafts2[size_t(d)];
        }
        // The sibling leaf of depth d: [s1] is leaf k, [d1, s2] leaf k + 1.
        // Its readout gives the bonus token; the next draft pass starts from
        // its saved rows, which follow the chain's k rows.
        const int sl = (d == 0 && s->sib >= 1) ? k : (d == 1 && s->sib >= 2) ? k + 1 : -1;
        if (sl >= 0 && tok == s->drafts2[size_t(d)]) {
          emit(greedy ? argm(sl) : pick(sl, pos0 + d + 1));
          lp_rec(sl, out.back());
          s->lh_first = sl;
          ++s->sib_taken;
        }
        return;
      }
    }
    const int readout = k > 0 ? k - 1 : 0;
    emit(greedy ? argm(readout) : exact ? pick(readout, pos0 + k) : sample(dist(readout), s->rng));
    lp_rec(readout, out.back());
  });
  // N > 1 requests whose prompt just ended: their other choices start from it.
  for (size_t i = 0, n0 = touched.size(); i < n0; ++i)
    if (touched[i]->phase == Seq::Prefill && touched[i]->final_chunk && !touched[i]->req.group_out.empty())
      fork_group(*touched[i], touched, produced, produced_lp);
  std::vector<CommitRequest> commits;
  std::vector<Seq*> committing;
  // The cache key of a checkpoint a prompt chunk just wrote; a cached
  // copy of the same prefix with the MTP state serves a request without it
  // too (that copy is kept and this one freed, as for a same-key copy).
  auto saved_key = [&](const CacheEntry& e) {
    std::string key = chat_key(e.digests, e.tokens, e.mtp);
    if (!e.mtp && !cache_.count(key))
      if (std::string mk = chat_key(e.digests, e.tokens, true); cache_.count(mk)) return mk;
    return key;
  };
  for (size_t i = 0; i < touched.size(); ++i) {
    Seq* s = touched[i];
    s->passes += 1;
    s->gpu_ms += st.gpu_ms;
    if (s->phase == Seq::Prefill) {
      if (s->passes == 1 && s->parent >= 0) {
        pinned_.erase(std::find(pinned_.begin(), pinned_.end(), s->parent));
        s->parent = -1;
      }
      // The system-prompt checkpoint became valid: cache it for other conversations.
      if (s->sysckpt >= 0 && !s->saved_sys && s->req.sys_split <= s->chunk_end && s->req.sys_split > s->cached) {
        CacheEntry e;
        e.ns = "chat";
        e.tokens.assign(s->req.prompt.begin(), s->req.prompt.begin() + s->req.sys_split);
        for (size_t g = 0; g < s->req.image_rows.size(); ++g)
          if (s->req.image_rows[g] < s->req.sys_split) e.digests.push_back(s->req.images[g].digest);
        e.slot = s->sysckpt;
        e.mtp = s->mtp;
        e.parent = s->chain_key;
        const std::string key = saved_key(e);
        pinned_.erase(std::remove(pinned_.begin(), pinned_.end(), s->sysckpt), pinned_.end());
        if (cache_.count(key)) {
          model_->slot_free(s->sysckpt);  // keep the cached copy (it may be a pinned parent)
        } else {
          cache_put(key, e);
          prefix_store_put(e);
        }
        s->saved_sys = true;
        s->keep.push_back(key);
        s->chain_key = key;
        s->sysckpt = -1;
      }
      // The instruction checkpoint became valid: photos under the same
      // system prompt and instructions share it.
      if (s->prectk >= 0 && !s->saved_pre && s->req.pre_split <= s->chunk_end && s->req.pre_split > s->cached) {
        CacheEntry e;
        e.ns = "chat";
        e.tokens.assign(s->req.prompt.begin(), s->req.prompt.begin() + s->req.pre_split);
        for (size_t g = 0; g < s->req.image_rows.size(); ++g)
          if (s->req.image_rows[g] < s->req.pre_split) e.digests.push_back(s->req.images[g].digest);
        e.slot = s->prectk;
        e.mtp = s->mtp;
        e.parent = s->chain_key;
        const std::string key = saved_key(e);
        pinned_.erase(std::remove(pinned_.begin(), pinned_.end(), s->prectk), pinned_.end());
        if (cache_.count(key)) {
          model_->slot_free(s->prectk);
        } else {
          cache_put(key, e);
          if (e.digests.empty()) prefix_store_put(e);
        }
        s->saved_pre = true;
        s->keep.push_back(key);
        s->chain_key = key;
        s->prectk = -1;
      }
      // The image checkpoint became valid: questions about the same images share it.
      if (s->imgckpt >= 0 && !s->saved_img && s->req.img_split <= s->chunk_end && s->req.img_split > s->cached) {
        CacheEntry e;
        e.ns = "chat";
        e.tokens.assign(s->req.prompt.begin(), s->req.prompt.begin() + s->req.img_split);
        for (size_t g = 0; g < s->req.image_rows.size(); ++g)
          if (s->req.image_rows[g] < s->req.img_split) e.digests.push_back(s->req.images[g].digest);
        e.slot = s->imgckpt;
        e.mtp = s->mtp;
        e.parent = s->chain_key;
        const std::string key = saved_key(e);
        pinned_.erase(std::remove(pinned_.begin(), pinned_.end(), s->imgckpt), pinned_.end());
        if (cache_.count(key)) {
          // The same tokens and images are cached already (another request
          // computed them in this pass): keep that entry, which a waiting
          // prompt may have pinned as its parent, and free this copy.
          model_->slot_free(s->imgckpt);
        } else {
          cache_put(key, e);
        }
        s->saved_img = true;
        s->keep.push_back(key);
        s->chain_key = key;
        s->imgckpt = -1;
      }
      // The prompt checkpoint became valid: cache it for later turns.
      if (s->ckpt >= 0 && !s->saved_ckpt && s->req.split <= s->chunk_end && s->req.split > s->cached) {
        CacheEntry e;
        e.ns = "chat";
        e.tokens.assign(s->req.prompt.begin(), s->req.prompt.begin() + s->req.split);
        for (size_t g = 0; g < s->req.image_rows.size(); ++g)
          if (s->req.image_rows[g] < s->req.split) e.digests.push_back(s->req.images[g].digest);
        e.slot = s->ckpt;
        e.mtp = s->mtp;
        e.parent = s->chain_key;
        const std::string key = saved_key(e);
        pinned_.erase(std::remove(pinned_.begin(), pinned_.end(), s->ckpt), pinned_.end());
        if (cache_.count(key)) {
          // The same prompt is cached already (a concurrent identical request):
          // keep that entry, which a later turn may have pinned as its parent.
          model_->slot_free(s->ckpt);
        } else {
          cache_put(key, e);
          prefix_store_put(e);
        }
        s->saved_ckpt = true;
        s->keep.push_back(key);
        s->chain_key = key;
        s->ckpt = -1;
      }
      s->done = s->chunk_end;
      s->t_wait = Clock::now();
      if (!s->final_chunk) continue;
      s->phase = Seq::Decode;
      s->t_first = Clock::now();
      s->t_decode = s->t_first;
    } else {
      if (!s->drafts.empty() && !s->from_lookup && s->req.forced.empty()) {
        ++s->dep_k[std::min<size_t>(8, s->drafts.size())];
        // The engine's acceptance by depth: depths up to the first miss.
        const int k = int(s->drafts.size());
        int a = 0;
        while (a < k && a < int(produced[i].size()) && produced[i][size_t(a)] == s->drafts[size_t(a)]) ++a;
        for (int d = 0; d <= std::min({a, k - 1, 7}); ++d) {
          int& n = dcost_.n_acc_depth[d];
          dcost_.acc_depth[d] += ((d < a ? 1.0 : 0.0) - dcost_.acc_depth[d]) / double(std::min(n + 1, 500));
          ++n;
        }
      }
      s->drafted += int(s->drafts.size());
      s->accepted += std::max(0, int(produced[i].size()) - 1);
      if (opt_.adaptive_drafts && decoding.size() == 1 && jobs.empty() && prefilling.empty() && s->req.forced.empty()) {
        // Adaptive drafting: costs of this pass, then draft only while
        // (verify + draft) / (expected tokens) beats a plain step by 5 %.
        const int k = int(s->drafts.size());
        if (k == 0) {
          if (s->pending.size() <= 1) ewma(dcost_.plain, dcost_.n_plain, st.wall_ms);
        } else {
          const int kk = std::min(k, 8);
          ewma(dcost_.verify[kk], dcost_.n_verify[kk], st.wall_ms);
          if (s->from_lookup) {
            // Prompt lookup: accepted drafts per pass, the mean of the first 4
            // passes, then an EWMA; decide on 4+ passes (one unlucky pass is not
            // evidence), a loss skips lookups for 16 steps.
            const double acc = double(std::max(0, int(produced[i].size()) - 1));
            double& a = s->acc_look;
            int& na = s->n_look;
            a = na < 4 ? (a * na + acc) / (na + 1) : 0.7 * a + 0.3 * acc;
            ++na;
            if (na >= 4 && dcost_.n_plain >= 3 && dcost_.n_verify[kk] >= 2 && dcost_.verify[kk] / (1.0 + a) > 1.05 * dcost_.plain) {
              s->skip_look = 16;
              a = 0;
              na = 0;
            }
          } else if (s->dep_reach[0][0] >= 4 && dcost_.n_plain >= 3 && dcost_.n_verify[kk] >= 2) {
            // MTP head: the expected tokens of a pass at this depth from the
            // request's per-depth acceptance under the engine's prior (mtp_expect),
            // not a short EWMA of the last passes' outcomes, so a noisy stretch of
            // timings or misses does not switch the head off for the request. A
            // loss buys a plain stretch of mtp_backoff steps (8, 16, ..., 128; chains
            // may run in it), then one probe pass decides again; a win resets it to 8.
            // With load-time costs (calibrate_drafts) the verdict compares their
            // ratios, which noisy or cold wall times do not move; else the EWMAs.
            const bool cal = dcost_.calibrated && dcost_.cal_verify[kk] > 0 && dcost_.cal_draft[kk] > 0;
            const double cost = (cal ? dcost_.cal_verify[kk] + dcost_.cal_draft[kk]
                                     : dcost_.verify[kk] + (dcost_.n_draftk[kk] ? dcost_.draftk[kk] : dcost_.draft)) +
                                (s->commit_k > 0 && dcost_.n_commit ? dcost_.commit : 0.0);
            const double plain = cal ? dcost_.cal_plain : dcost_.plain;
            const double expect = mtp_expect(*s, k);
            const bool lost = cost / expect > 1.05 * plain;
            if (lost) {
              s->skip_mtp = s->mtp_backoff;
              s->mtp_backoff = std::min(2 * s->mtp_backoff, 128);
              ++s->lost_mtp;
              ++s->mtp_lost_run;
            } else {
              s->mtp_backoff = 8;
              s->mtp_lost_run = 0;
            }
            if (trace)
              std::fprintf(stderr, "[trace] mtp verdict k=%d cost %.1f expect %.2f plain %.1f%s -> %s (run %d, next stretch %d)\n", k, cost,
                           expect, plain, cal ? " (load-time costs)" : "", lost ? "plain" : "draft", s->mtp_lost_run, lost ? s->skip_mtp : 0);
          }
        }
      }
      if (trace)
        std::fprintf(stderr, "[trace] pass rows=%d seqs=%zu drafts=%d%s sib=%d produced=%zu wall %.1f ms gpu %.1f ms skip=%d mtp_ok=%d gen=%d lh=%d\n",
                     st.rows, decoding.size(), int(s->drafts.size()), s->drafts.empty() ? "" : (s->from_lookup ? "(lookup)" : "(mtp)"), s->sib,
                     produced[i].size(), st.wall_ms, st.gpu_ms, std::max(s->skip_mtp, s->skip_look), int(model_->slot_mtp(s->work)), s->generated,
                     s->lh_first);
    }
    if (s->gm_error) {  // never by design (every token is picked under the mask)
      auto it = std::find_if(active_.begin(), active_.end(), [&](const auto& p) { return p.get() == s; });
      finish_seq(*s, "error", "structured output: the grammar allows no token here (internal error)");
      if (it != active_.end()) active_.erase(it);
      continue;
    }
    // Emit tokens up to an end token / the limit.
    std::vector<int32_t> emit;
    std::string reason;
    for (int32_t t : produced[i]) {
      if (!s->req.ignore_eos && (std::find(end_ids.begin(), end_ids.end(), t) != end_ids.end() ||
                                 std::find(s->req.stop_ids.begin(), s->req.stop_ids.end(), t) != s->req.stop_ids.end())) {
        reason = "stop";
        break;
      }
      emit.push_back(t);
      ++s->generated;
      if (s->req.sampling.penalized()) s->pen.add_output(t);
      if (opt_.prompt_lookup) s->ngram.append(t);
      if (s->generated >= s->req.max_tokens) {
        reason = "length";
        break;
      }
    }
    // The records of the emitted tokens travel with them.
    if (produced_lp[i].size() > emit.size()) produced_lp[i].resize(emit.size());
    s->out->push(emit.data(), emit.size(), s->req.logprobs >= 0 ? &produced_lp[i] : nullptr);
    if (s->commit_k > 0 && !produced[i].empty()) {
      // The accepted drafts become rows of the slot (commit below); only
      // the new token is pending, so the next pass has room for a full chain.
      if (reason.empty() && !s->out->cancelled()) {
        commits.push_back({s->work, int(produced[i].size()) - 1});
        committing.push_back(s);
      }
      s->pending.assign(1, produced[i].back());
      s->commit_k = 0;
    } else {
      s->pending = produced[i];
    }
    if (!reason.empty()) s->pending.clear();
    if (!reason.empty() || s->out->cancelled()) {
      auto it = std::find_if(active_.begin(), active_.end(), [&](const auto& p) { return p.get() == s; });
      finish_seq(*s, reason.empty() ? "cancelled" : reason);
      if (it != active_.end()) active_.erase(it);
    }
  }
  if (!commits.empty()) {
    try {
      PassStats cs;
      model_->commit(commits, &cs);
      gpu_ms_ = gpu_ms_ + cs.gpu_ms;
      if (opt_.adaptive_drafts && commits.size() == 1) ewma(dcost_.commit, dcost_.n_commit, cs.wall_ms);
      if (trace) std::fprintf(stderr, "[trace] commit seqs=%zu rows=%d wall %.1f ms gpu %.1f ms\n", commits.size(), cs.rows, cs.wall_ms, cs.gpu_ms);
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      for (Seq* s : committing) {
        auto it = std::find_if(active_.begin(), active_.end(), [&](const auto& p) { return p.get() == s; });
        if (it == active_.end()) continue;
        finish_seq(*s, "error", msg);
        active_.erase(it);
      }
    }
  }
}

json::Value Engine::stats() const {
  json::Value o = json::Value::object();
  o.set("passes", int64_t(passes_.load()));
  o.set("rows", int64_t(rows_.load()));
  o.set("gpu_ms", round3(gpu_ms_.load()));
  o.set("generated_tokens", int64_t(gen_tokens_.load()));
  o.set("decisions", int64_t(decisions_done_.load()));
  o.set("free_slots", free_slots_pub_.load());  // published by the scheduler (the model is single-threaded)
  o.set("free_kv_tokens", free_kv_pub_.load());
  o.set("ckpt_live_evictions", int64_t(live_evictions_.load()));
  o.set("active_sequences", active_pub_.load());
  o.set("waiting_sequences", waiting_pub_.load());
  o.set("kv_stopped_sequences", int64_t(kv_stopped_.load()));
  o.set("kv_deferred_chunks", int64_t(kv_deferred_.load()));
  o.set("decision_row_reservations", int64_t(decision_reservations_.load()));
  o.set("kv_skipped_checkpoints", int64_t(kv_skipped_.load()));
  std::string x;
  {
    std::lock_guard<std::mutex> lk(xsnap_mu_);
    x = xsnap_;
  }
  if (!x.empty()) o.set("experts", json::parse(x));  // hit rate, GB read, load and wait ms, prefetch
  return o;
}

}  // namespace espetero
