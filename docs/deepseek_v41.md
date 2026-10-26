# DeepSeek-V4.1 (`deepseek_v41`) — architecture and Espetero's implementation

DeepSeek-V4.1-Flash and its finetunes (for example `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`):
552B parameters, 16B active per token, 510 GB on the SSD. Espetero runs it on a
128 GB Mac with the routed experts (289 GB) and the Engram n-gram tables (203 GB)
read from the SSD, the rest (≈ 11 GB) in memory.

Sources: DeepSeek's reference `inference/model.py`, `kernel.py` and `engram.py`
(deepseek-ai/DeepSeek-V4.1-Flash, `inference/`); the checkpoint's `config.json`
and safetensors headers. `tools/dsv4_ref.py` is a numpy transcription of
`model.py` that Espetero is checked against (§9).

## 1. Checkpoint

| Tensors | Format |
|:--|:--|
| attention `wq_a` [1280, 5120], `wq_b` [32768, 1280], `wkv` [512, 5120], `wo_a` [8192, 4096], `wo_b` [5120, 8192]; shared expert `w1`/`w3` [2304, 5120], `w2` [5120, 2304]; indexer `wq_b` [4096, 1280]; Engram `wkv` [25600, 6144] | FP8 e4m3 + one e8m0 scale per 32 × 32 block (`.scale` [N/32, K/32]) |
| routed experts `layers.l.ffn.experts.e.w{1,2,3}` | FP4 e2m1, two per byte (low nibble first) as I8 [N, K/2] + one e8m0 scale per 32 inputs of a row (`.scale` [N, K/32]) |
| Engram tables `layers.{1,14}.engram.embed` | FP8 [384 M rows, 256] + e8m0 [rows, 8] |
| embedding, head [129280, 5120], routers [384, 5120], compressor `wkv`/`wgate` [512, 5120], indexer `wk` [128, 512] and `weights_proj` [32, 5120], norms | bf16 |
| `hc_{attn,ffn}_{fn,base,scale}`, `attn_sink`, router biases | fp32 |

Every expert's three code tensors are contiguous in its file, and so are its three
scale tensors (18.8 MB per expert: two reads). Shards 3–42 hold one layer each,
47–48 the two Engram tables, 44–46 the DSpark draft layers (not used yet), 1 the
vision tower (not used yet).

## 2. Decoder (inference/model.py `Transformer.forward`)

```
h = [E(id)] * 4                                  # hc_mult = 4 copies of the residual
pre = one-hot(0)
for l in 0..39:
    if l in (1, 14): h = Engram_l(h, n-gram rows)
    (a_pre, a_post, a_comb) = mixes(h, hc_attn)  # used by THIS layer's FFN (pre) and attention (post, comb)
    x = RMSNorm(sum_j pre_j h_j);  y = Attention_l(x);  h_k = a_post_k y + sum_j a_comb[j][k] h_j
    (f_pre, f_post, f_comb) = mixes(h, hc_ffn)
    x = RMSNorm(sum_j a_pre_j h_j); y = MoE_l(x);       h_k = f_post_k y + sum_j f_comb[j][k] h_j
    pre = f_pre                                   # the next layer's attention input mix
logits = head(RMSNorm(sum_j pre_j h_j))
```

`mixes`: `m = fn · flat(h) · rsqrt(mean(flat(h)²) + 1e-20)` (24 outputs from the
20480-long flat residual), `pre = σ(m·s0 + b) + 1e-6`, `post = 2σ(m·s1 + b)`,
`comb` = 20 Sinkhorn iterations on `softmax_rows(m·s2 + b)` (+1e-6 and column
normalization first). All RMSNorm eps are 1e-20.

**Engram** (layers 1, 14): for each position, 2-, 3- and 4-grams of
*compressed* token ids (tokens that normalize alike share an id; 99,092 ids)
are hashed with per-layer multipliers (`(id_t·M0) ⊕ (id_{t−1}·M1) ⊕ …`) modulo
24 distinct primes (3 n-gram sizes × 8 heads) into disjoint ranges of the
table: 24 rows of 256 per position, `wkv` maps them to 4 keys and one value of
5120; each copy j is gated by `σ(signed_sqrt(⟨h_j, q_j k_j key_j⟩ / (rms h_j · rms key_j · √5120)))`.

**Attention** (MQA over one 512-wide latent per token, used as K and V):
`qr = RMSNorm(wq_a x)`, `q = wq_b qr` (64 heads × 512), `kv = RMSNorm(wkv x)`,
RoPE on the last 64 dims (adjacent pairs). Keys: a sliding window of the last
128 tokens, plus — for `compress_ratio` > 0 — compressed entries:

* ratio 2 (layers 2–19): pairs of tokens pooled by a learned per-channel
  softmax gate (`wkv`, `wgate`), ratio 1 (layers 20–39): one entry per token;
  RMSNorm, RoPE at the group's first position (YaRN, θ = 160,000, factor 16),
  FP4 with e4m3 scales per 16. Only `kv_source_layers` (2, 8, 14, 20) compute
  them; the layers after a source read its cache.
* the indexer (`index_source_layers` 2, 8, 14, 20, 24, 28, 32, 36) keeps, per
  query, the 512 compressed entries with the largest `Σ_h w_h · relu(q_h · k)`
  (32 heads of 128, FP4 queries and keys); the layers after a source reuse its
  selection. With ≤ 512 visible entries this selects everything.
* candidate blocks (layer 20 → 24…36): beyond 16,384 visible entries, the
  later indexers only look inside the 2048 best blocks of 8 entries chosen by
  layer 20.

Softmax with a learned per-head sink (`+ exp(sink)` in the denominator), output
un-rotated, then `wo_a` per group of 8 heads (block-diagonal) and `wo_b`.
Layers 0–1 (ratio 0) use plain RoPE θ = 10,000.

**MoE**: `s = sqrt(softplus(router x))`, experts = top-6 of `s + bias`, weights
= the six `s` normalized × 1.5; experts and the shared expert are SwiGLU with
the gate clamped ≤ 10 and the up branch to ±10, the routing weight applied
before `w2`.

**Quantized activations**: before every FP8/FP4 matmul the reference rounds the
input to e4m3 with one power-of-two scale per 32 values (`act_quant`); window KV
is stored the same way, compressed KV and index keys/queries on the FP4 grid.
Espetero reproduces these roundings bit for bit (`ds4_quant`, checked against
numpy on 2 M values); the matmuls then use exact products.

## 3. Espetero's implementation

Files: `kernels/80_dsv4.metal`, `src/model/impl/dsv4.h` (config, checkpoint
index, Engram constants, expert cache), `src/model/impl/dsv4_model.h` (the
`NativeModel`). `NativeModel::load` picks it for `model_type` `deepseek_v41`.

* **Weights** stay as stored. Decode GEMVs decode FP8 (`e4m3` bits shifted
  into a half = value·2⁻⁸) and FP4 (`e2m1` → value·2⁻¹⁴) in registers; prefill
  GEMMs (≥ 16 rows) decode into bfloat tiles (exact) for simdgroup matrices.
* **Experts**: an LRU cache of expert slots in ≤ 3 buffers of host memory
  locked with `mlock` and wrapped for the GPU without copies (by default what
  the GPU working set and the Mac's memory leave: ~97 GB, 5168 slots, a third
  of the experts; `--expert-cache-gb`). Locked pages are never compressed or
  paged out; GPU-private buffers were, whenever macOS unwired an idle GPU's
  memory (`iogpu.wired_lwm_mb` = 0), which cost minutes of compression storms.
  After each layer's routing the missing experts are read with parallel 1 MB
  `pread`s (`F_NOCACHE`, ≈ 8 GB/s from the internal SSD) straight into their
  slots; long passes pipeline reads with the GPU (shared events), decode
  prefetches the next layer's likely experts.
  **Streaming** (long passes): a layer over ≥ 256 rows uses nearly every
  expert, so its experts need no routing: a reader thread reads the coming
  layers' missing experts in layer order, two layers ahead of the GPU, while
  the GPU runs attention and the MoE of the layers before; the MoE takes them
  as they arrive. A pass needs more experts (~14 K) than the cache holds, so
  while streaming the cache evicts first what this pass streamed in for layers
  it is done with: the coming layers' cached experts and the decode working set
  stay (LRU evicted both, and decoding after a long prompt lost hits).
  `ESPETERO_DS4_STREAM=0`: off. A pass holds up to 32 K rows (~140 KB of
  scratch per row), so prompts up to the context run as one pass. The hottest slots are saved at
  exit and read back at start (`ESPETERO_DS4_WARM=0` skips it).
  Eviction: `ESPETERO_DS4_TRACE=FILE` records every access and
  `tools/dsv4_xsim.py` replays policies on it. On a mixed trace (decodes and a
  4.5 K-token prompt) LRU missed 19 experts per decoded token, ARC 19.4, SLRU
  18.6, decayed frequency 18.7–58 (the longer its memory the worse: new experts
  are evicted before they earn a count), and Belady's optimum, which knows the
  future, 9.2. The policy is not where the misses are; LRU stays.
* **Engram**: the 24 rows a position needs per Engram layer are hashed on the
  host and read with `pread` from the checkpoint (264 B each; never loaded).
  The compressed token map needs a Unicode normalizer Espetero does not have:
  `tools/dsv4_engram_prep.py MODEL_DIR` writes it once to
  `MODEL_DIR/espetero_engram.bin` (with the multipliers and primes).
* **Passes run layer by layer**: a prompt of up to 16,384 tokens is one pass;
  every layer runs over all its rows (sub-chunks of 1024 rows on the GPU)
  before the next layer, so a layer's experts are read from the SSD once per
  prompt, not once per chunk.
* **Sequences** are slots of their own (no paged KV): 40 window rings of 128
  latents, the four compressed caches and index-key caches for the context,
  the compressor's pending rows; ≈ 3.3 KB per token of context (~110 MB a slot
  at 32 K). The server gets up to 16 slots (the concurrent requests, `--users`;
  at least 6; before Fase 3 W7: 6), out of the expert cache. Slots save to
  and load from disk (`slot_save` / `slot_load`, the engine's prompt checkpoints).
  While other sequences decode, a prompt still runs as one pass (the engine's
  `ModelInfo::chunk_prefill` is off for this model): chunks of `--prefill-chunk`
  rows would read a layer's experts once per chunk.
* **Causal encoder-decoder** (CED, the V4.1 report §2.2 and §3.2.2): layers 0–19
  are the encoder; layer 20, the last KV source, compresses the encoder's output
  into the KV and index keys every decoder layer (20–39) reads. A decoder layer's
  only state of its own is its 128-row window ring, so for a prompt each decoder
  layer runs over its last rows only: layer 39 over the rows read out (the
  logits) plus 127 rows of window latents, each layer below over what the next
  one needs plus 127 (layer 20: the last ~2.5 K rows, and the compressor over all).
  Exact (bit-identical logits with the deterministic settings of §9), unlike
  DeepSeek's serving, which replays only the last 128 rows through the decoder
  and trained the model for that approximation. Prefill of 4.5 K tokens: 46.3 →
  38.4 s; of 19.8 K tokens: 150 → 108 s (GPU 118 → 65 s). `ESPETERO_DS4_CED=0`: off.
* **Several sequences per pass**: the trees of a step (users decoding at once,
  prompts) share each pass: a GPU block of rows may mix sequences, the dense
  projections and the experts serve all of them, attention runs per sequence.
  A layer's experts are the union of every row's.
* **Images** (`vision_layout` "deepseek"): the engine plans and pads each image
  as `inference/image_processor.py` does (PIL-exact bicubic, grey padding, bf16
  pixels: identical to PIL), the run is `[start] + (row + newline) x rows + [end]`
  of `<｜deepseek_image｜>`; the model runs the 32-layer ViT (2D RoPE, Espetero's
  ViT attention) and the 3 x 3 aligner, places the learned start / newline / end
  rows, routes image-span rows with the VL router bias and keeps them out of the
  n-grams. Embeddings match a numpy transcription to bf16 precision (the same
  distance a bf16 run of the reference has from fp32).
* **Tool calls**: replies in DeepSeek's DSML format (`<｜DSML｜ calls>` …) are parsed
  into OpenAI `tool_calls` (string parameters as strings, the rest as JSON);
  `tool_choice` "required" or a named function force the call's opening.
* **DSpark** (the draft head, `mtp.*`): implemented and exact (verified chains
  produce the plain chain's tokens), but off by default (`ESPETERO_DS4_SPEC=1`):
  its drafts were accepted ~1 in 3 (2.1 tokens per round), and verifying a block
  reads several times one token's experts from the SSD, so a chain decoded at
  1.45 tok/s against 3.1 tok/s plain (12 GB expert cache). The draft rings, the
  target-layer states and the rollback of rejected rows are in `dsv4_model.h`.
* **Leaves** (the decisions API, `/v1/decisions`): each leaf continues its
  tree's root in a scratch slot of its own (forked from the root's state), all
  leaves of a step in shared passes. Prompt-lookup drafts stay off for this
  model (`ModelInfo::prompt_lookup`): verifying them reads more experts.

## 4. Chat template

The dealignai checkpoint ships no template; Espetero reads `chat_template.jinja`
next to the weights (copy it from deepseek-ai/DeepSeek-V4.1-Flash). Prompts end
with `<｜Assistant｜><think>` (thinking, `reasoning_effort` low…max) or
`<｜Assistant｜></think>` (`enable_thinking: false`); the engine treats a prompt
ending in `<think>` as an open reasoning block.

## 5. What the V4.1 report changes for a single Mac, and what is left

The report (arXiv 2609.19969) describes V4.1-Flash as a causal encoder-decoder
with CSA2 (Full / Reindex / Reuse layers, a hierarchical indexer from layer 20),
Single-Pass mHC (each block mixes its input with the previous block's
coefficients, which is what `hc_pre` with the previous layer's `f_pre` does),
Engram at layers 1 and 14 and DSpark with a confidence head. Measured on the
M4 Max (128 GB, 97 GB expert cache, internal SSD at ~6–7 GB/s):

* **Prefill**: every layer of a long prompt touches all 384 experts. Measured
  (server, warm cache):

  | prompt | before | CED | + streaming, 32 K-row passes, pass-aware eviction |
  |:--|:--|:--|:--|
  | 4.5 K tokens | 46.3 s | 38.4 s | 30.6 s (146 tok/s; 202 GB read at 7.1 GB/s) |
  | 19.7 K tokens | 149.8 s | 107.5 s | 69.9 s (282 tok/s; GPU 57 s, reads 29 s) |

  Short prompts are now bound by the SSD, long ones by the GPU's prefill
  kernels.
  DeepSeek's decoder bounded replay (128 rows per decoder layer, approximate,
  trained for) would save ~24 K decoder rows more; not adopted.
* **Decode** (~10.2 tok/s): expert reads ~56 ms per token (20 misses of
  18.8 MB at ~6.7 GB/s) and GPU ~42 ms, one after the other. GPU per token,
  1285 dispatches: the dense fp8 GEMVs (~5.5 GB) and the experts (4.5 GB) run
  near the memory's bandwidth; attention took 8 ms because a decoded token gave
  it 8 threadgroups (one simdgroup walking a head's ~640 keys):
  `ds4_attn_split` (≤ 4 rows) gives each head a threadgroup whose simdgroups
  split the keys, 1.2 ms (GPU 46 → 42 ms per token). One-row fp8 GEMVs compute
  4 output rows per simdgroup (`ESPETERO_DS4_GEMV1=1`: one), ~0.5 ms.
  Overlapping a layer's miss reads with its cached experts'
  GPU work (the pipelined MoE for one row) changed nothing measurable.
* **Anatomy of a decoded token** (2026-10-22, `dsv4_run`, a 512-token prompt
  of source text then 128 greedy tokens, medians of the last 64, main dde1aa1):
  82 ms = GPU 40.0 + command-buffer launch and wake-up 5.6 (41 command buffers)
  + expert reads 34.0 (12 misses × 2.8 ms) + Engram rows 1.0 + host work 0.9.
  The 40 ms of GPU, every dispatch timed alone (1332 dispatches): dense fp8
  GEMVs 16.1 ms (~6.8 GB, ~420 GB/s), routed experts 12.0 (4.5 GB, ~375 GB/s),
  bf16 head and routers 3.5, the norm / RoPE epilogues 2.1 (120 dispatches),
  attention 1.6, the four hyper-connection kernels 3.0 over 321 dispatches, the
  rest ~900 tiny dispatches ≈ 5 ms. The misses depend on the history: the first
  32 tokens after the prompt missed 58 experts each (225 ms per token).
* **Reading a token's experts earlier is not possible without prediction.**
  Layer L's routing is the output of layer L's attention, which needs layer
  L − 1's complete output, experts included: once layer L's misses are known
  the only GPU work left that does not need them is layer L's shared expert
  and its cached experts (~0.3 ms), which is what the pipelined MoE overlapped
  with nothing to show. What a token decides for itself is read as soon as it
  is decided; everything earlier is prediction (above). What the host adds
  around the reads is what can go: three changes, each bit-identical to main
  (per-layer residual dumps of a 48-token prefill plus 4 decode steps, 800
  files; logits hashes of 16 greedy steps):
  the Engram rows gathered in a thread while the GPU runs layer 0
  (`ESPETERO_DS4_ENGRAM_BG=0`; 1.1 → 0.15 ms per token when it hides), the
  GPU-driven decode (the pass is one command buffer committed before any route
  is known, the GPU signals a shared event after each router and waits for the
  host's before each layer's experts, the host resolves the route, reads the
  misses and fills the expert kernels' tables while it waits; opt-in
  `ESPETERO_DS4_DRIVE=1`: it **lost** too, wall time minus reads +15 / +23 /
  +26 ms per token in 3 pairs against main's binary: the 40 waits on a
  host-signalled event cost ~0.4 ms each to resume (`drive_gap_ms`: 55.7 ms
  from host signal to the GPU's next signal for ~39 ms of GPU work, the same
  with a spinning or a blocking host), far more than the 41 command-buffer
  launches they replace, 5.6 ms in all), and one
  hyper-connection dispatch between sub-layers instead of four
  (`ds4_hc_fused`, opt-in `ESPETERO_DS4_HCFUSE=1`: it **lost**, GPU time
  40.1 → 44.2 ms per token in 3 of 3 pairs against main; one threadgroup per
  row serializes the 24 × 20480 mix that `ds4_hc_part` spreads over 20, and at
  1 row the 240 dispatches it saves cost nothing on the M4 Max).
* **The 41 command buffers per token** cost 5.6 ms: ~85 µs from commit to the
  GPU's start and ~51 µs from the GPU's end to the host noticing (spinning),
  per buffer. Fewer buffers need either a GPU-side wait on the host (above:
  ~0.4 ms each, lost) or speculation: encoding a layer's experts from a
  GPU-side slot map and letting the GPU run on, redoing the layers after a
  miss. With ~12 misses in ~10 of 40 layers per token a two-layer buffer has
  a miss 44 % of the time and the wasted layer (~1 ms) outweighs the launch
  it saved; longer buffers are worse. The route is the only thing the host
  needs, and nothing cheaper than a command buffer boundary carries it.
* **The expert cache's size** (the access trace of the run above replayed with
  `tools/dsv4_xsim.py`, which starts cold, so the counts are relative):
  LRU misses per decoded token 28.7 at 4745 slots (89 GB), 26.5 at 5168
  (97 GB, the default), 24.4 at 5600, 22.1 at 6000, 19.4 at 6500, 16.7 at
  7000 (132 GB): each 1000 slots (18.8 GB) cut the misses ~15 %, ARC 2–3 %
  more than LRU. The default cache already takes what the Mac leaves (~14 GB);
  the Engram tables are read from the SSD (not resident) and the KV of 6 slots
  is ~1 GB, so neither frees a slot's worth: the misses move only with memory
  the machine does not have, or with a smaller expert (lossy).
* **Predicting the next layer's experts** (to read them while the GPU works) does
  not pay here: `ESPETERO_DS4_DUMPX=dir` dumps each layer's MoE input and routes,
  `tools/dsv4_route_predict.py` measures predictors and `tools/dsv4_prefetch_sim.py`
  replays a per-layer LRU cache with prefetching. The next layer's router on this
  layer's MoE input finds 57–81 % of the routed experts in its top 6 (a ridge
  regression trained on 13 K rows does no better), but prefetching its top 6
  raises the experts read per token from 23.6 to 42.8 to cut those still
  missing to 13.8, and the SSD is the bottleneck: the GPU's ~42 ms per token
  hide only ~15 reads. One prefetch per layer at most: 27.4 read, 18.9 missing,
  and a prefetch has ~1 ms of GPU work to hide a 2.8 ms read.
* **DSpark** (`ESPETERO_DS4_SPEC=1`) with the 95 GB cache: lossless (same text).
  Verifying every draft: 46 % accepted, 7.1 tok/s against ~10 plain. The
  confidence head (`mtp.2.confidence_head.proj`: w · [the last DSpark layer's
  hc-mixed state, the Markov embedding of the previous token], `ds4_conf`) is
  well calibrated in Espetero (predicted 0.97 → 98 % accepted, 0.85 → 88 %,
  0.45 → 47 %), and drafts are verified only while their predicted survival
  (the product of the sigmoids) stays above `ESPETERO_DS4_SPEC_MIN` (0.8): 94 %
  of verified drafts are accepted, 1.6–1.9 tokens per round. Still about even
  with plain decoding (9.7 / 9.5 tok/s against 9.9 / 9.4): a round costs the
  draft pass (14 ms) and an accepted row saves only the dense part of a token
  (~30 ms) — its experts' SSD reads move to the verification pass, they are not
  saved. So it stays off; `ESPETERO_DS4_SPEC_LOG=file` logs each round's
  confidences and acceptances.
* **Engram** rows depend only on the tokens: they can be read at the start of a
  pass (the report prefetches them over RDMA); 1.1 ms per token today.

## 9. Parity

`tools/dsv4_ref.py MODEL --tokens … --layers N [--no-quant] [--force-routes DIR]`
writes the reference's per-layer residuals; `ESPETERO_DS4_DUMP=DIR` makes
Espetero write its own (`build/dsv4_run`). Router scores are close together
(top-6 margins of 1e-5 on scores near 10 occur), so the two pick different
experts at such near-ties: `--force-routes` gives the reference Espetero's
routing to compare everything else. Results (first 3 layers, 11- and 1300-token
prompts, prefill in sub-chunks plus decode steps, no activation rounding so
that tiny differences do not flip fp8 roundings): ≤ 3·10⁻⁴ relative per token
(half-precision KV storage); with the roundings, rows that hit no rounding
boundary agree to 10⁻⁶ and the rest differ by the reference's own rounding
chaos (10⁻²).

Development switches: `ESPETERO_DS4_LAYERS=n` (first n layers, no head),
`ESPETERO_DS4_NO_ENGRAM=1`, `ESPETERO_DS4_NOQUANT=1`, `ESPETERO_DS4_DUMP=dir`,
`ESPETERO_DS4_PROFILE=1` (per-kernel GPU times), `ESPETERO_DS4_GEMM_MIN`,
`ESPETERO_DS4_SUB`, `ESPETERO_DS4_ROWS`, `ESPETERO_DS4_PREFETCH=0`,
`ESPETERO_DS4_WARM=0`, `ESPETERO_DS4_TRACE=file`, `ESPETERO_DS4_CED=0`, `ESPETERO_DS4_STREAM=0`,
`ESPETERO_DS4_PIPE=0`, `ESPETERO_DS4_ATTN1=1`, `ESPETERO_DS4_GEMV1=1`, `ESPETERO_DS4_DUMPX=dir`,
`ESPETERO_DS4_HCFUSE=0`, `ESPETERO_DS4_ENGRAM_BG=0`, `ESPETERO_DS4_DRIVE=0`, `ESPETERO_EXPERT_CACHE_GB`;
`build/dsv4_run` with `DS4_RUN_HASH=1` prints a hash of every logit after the prefill and each step.

Determinism: a pass is bit-reproducible except that the pipelined MoE of long
passes adds a row's expert outputs in an order that depends on which experts
were cached, and the activation roundings amplify that last bit into ~0.5 on
the logits. Exact comparisons (e.g. a change that must not alter results) run
`build/dsv4_run` with `ESPETERO_DS4_PREFETCH=0 ESPETERO_DS4_PIPE=0
ESPETERO_DS4_GEMM_MIN=1000000`: CED, streaming, chunked prompts and two
sequences per pass all give bit-identical logits that way. (`make` builds only
`build/espetero`; `make build/dsv4_run` for the driver.)
