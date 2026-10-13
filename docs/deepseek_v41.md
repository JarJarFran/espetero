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
* **Experts**: an LRU cache of expert slots in ≤ 3 GPU-private buffers (by
  default what the GPU working set and the Mac's memory leave: ~95 GB, a third of
  the experts; `--expert-cache-gb`). After each layer's routing the pass waits
  for the GPU, reads the missing experts with parallel 1 MB `pread`s
  (`F_NOCACHE`, ≈ 8 GB/s from the internal SSD) into staging and copies them into
  their slots. One grouped dispatch computes all of a layer's experts.
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
* Not yet: leaves (decisions, prompt-lookup drafts).

## 4. Chat template

The dealignai checkpoint ships no template; Espetero reads `chat_template.jinja`
next to the weights (copy it from deepseek-ai/DeepSeek-V4.1-Flash). Prompts end
with `<｜Assistant｜><think>` (thinking, `reasoning_effort` low…max) or
`<｜Assistant｜></think>` (`enable_thinking: false`); the engine treats a prompt
ending in `<think>` as an open reasoning block.

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
`ESPETERO_DS4_SUB`, `ESPETERO_DS4_ROWS`, `ESPETERO_EXPERT_CACHE_GB`.
