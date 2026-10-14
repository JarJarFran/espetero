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
  the compressor's pending rows; ≈ 3.3 KB per token of context. Slots save to
  and load from disk (`slot_save` / `slot_load`, the engine's prompt checkpoints).
* **Causal encoder-decoder** (CED, the V4.1 report §2.2 and §3.2.2): layers 0–19
  are the encoder; layer 20, the last KV source, compresses the encoder's output
  into the KV and index keys every decoder layer (20–39) reads. A decoder layer's
  only state of its own is its 128-row window ring, so for a prompt each decoder
  layer runs over its last rows only: layer 39 over the rows read out (the
  logits) plus 127 rows of window latents, each layer below over what the next
  one needs plus 127 (layer 20: the last ~2.5 K rows, and the compressor over all).
  Exact (with the activation roundings off, logits agree with a full pass to
  float noise; with them, both move by the roundings' own chaos), unlike
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
* **Decode** (~10 tok/s): GPU ~46 ms per token, expert reads ~53 ms (19 misses
  of 18.8 MB). Eviction policies cannot do much (LRU 19 misses per token,
  Belady 9.2, §3 Experts); fewer misses need knowing the next experts earlier
  (trained routing predictors, SpecPrefetch-style, keep the native router and
  so the output). The GPU part is ~2x a bandwidth bound of ~12 GB per token.
* **DSpark** (`ESPETERO_DS4_SPEC=1`) with the 97 GB cache: lossless (same text),
  46 % of drafts accepted, 2.6 tokens per round, but 7.1 tok/s against 10.1
  plain (history text) and 8.4 against 9.4 (code): each verified row costs
  ~82 ms, almost a token, mostly the experts rejected drafts bring in (misses
  per token 19 → 29). The checkpoint ships the confidence head
  (`mtp.N.confidence_head.proj`, sigmoid of the backbone state and the previous
  draft's Markov embedding, trained to predict each draft's acceptance) that
  the report uses to choose how many drafts to verify; Espetero does not use it
  yet. With it, a draft is worth verifying only when its predicted survival
  exceeds the row's relative cost (~0.8 today).
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
`ESPETERO_DS4_PIPE=0`, `ESPETERO_EXPERT_CACHE_GB`.

Determinism: a pass is bit-reproducible except that the pipelined MoE of long
passes adds a row's expert outputs in an order that depends on which experts
were cached, and the activation roundings amplify that last bit into ~0.5 on
the logits. Exact comparisons (e.g. a change that must not alter results) run
`build/dsv4_run` with `ESPETERO_DS4_PREFETCH=0 ESPETERO_DS4_PIPE=0
ESPETERO_DS4_GEMM_MIN=1000000`: CED, streaming, chunked prompts and two
sequences per pass all give bit-identical logits that way. (`make` builds only
`build/espetero`; `make build/dsv4_run` for the driver.)
