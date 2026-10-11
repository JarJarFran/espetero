# Espetero — design

Espetero is a native inference engine for Apple silicon, written in C++20
with Objective-C++ only where it calls Metal. It generalizes the JEVfire
Metal engine (https://github.com/kikoncuo/jevfire) from "one decision
request per GPU pass" to one scheduler that runs three kinds of work together:

| Workload | API | Shape of the work |
|:--|:--|:--|
| Structured decisions (JEV-like) | `POST /v1/decisions` | one tree per request: the uncached prompt prefix as the root, field suffixes as leaves scored on label tokens |
| Chat / completions | `POST /v1/chat/completions`, `POST /v1/completions` | chunked prefill, then one small tree per decode step (pending tokens + draft leaves) |
| Batch jobs | `POST /v1/files`, `POST /v1/batches` (OpenAI Batch API) or `espetero batch` | many chat/decision requests at low priority, continuously batched with interactive traffic |

Every scheduler step builds **one forest** from all runnable work (decode
steps of every active sequence, pending decisions, prefill chunks) and runs it
as **one GPU pass**: projections and MLPs cover every row at once (weights are
read once per pass, whatever the number of sequences), sequence mixing runs
per tree inside the same dispatches.

Supported models: the **dense** Qwen3.5 family (`qwen3_5`; Qwen3.5,
Qwen3.6-27B, Qwen3.8-2B/27B…) and its **mixture-of-experts** variant
(`qwen3_5_moe`, Qwen3.6-35B-A3B; design in [moe.md](moe.md)) from MLX
safetensors or llama.cpp `qwen35` / `qwen35moe` GGUF, with the Qwen3-VL vision
tower and the MTP draft head; **Qwen3.8-Flash-Next** (`qwen4_exp`, architecture in
[qwen4_exp.md](qwen4_exp.md)) from MLX, with its experts streamed from the SSD
when the model does not fit in memory; and **Gemma 4** (`gemma4`, 31B dense
and 26B-A4B MoE, text only) from MLX or llama.cpp `gemma4` GGUF: an
attention-only decoder whose sliding-window layers keep their last 1024
positions in a ring of KV blocks per sequence slot (slot state, copied on
forks) while its global layers use the paged pool (`src/model/impl/gemma.h`,
`kernels/80_gemma.metal`).

## Layers

```
src/util/json.*        JSON values, parser, Python-exact json.dumps
src/tok/               byte-level BPE tokenizer (HF tokenizer.json or GGUF vocab), token-identical to HF
src/tmpl/              Jinja2 interpreter for the model's own chat_template.jinja
src/vision/            image decode (ImageIO), Pillow-exact bicubic, Qwen-VL patches, M-RoPE
src/http/              HTTP/1.1 server with SSE streaming
src/model/api.h        pure C++ interface of the native model (forests, slots, drafts)
src/model/model.mm     the only Objective-C++ translation unit: Metal runtime + model
src/model/impl/        runtime, weight readers (MLX safetensors, GGUF), matrix layouts,
                       GEMM dispatch + autotune, config, loader, vision tower, forest pass
kernels/*.metal        Metal kernels, embedded at build time, compiled by the driver at load
src/engine/            scheduler (engine.cpp), sampling, decisions, chat, batch API
src/server/            routes and the `espetero` CLI
tools/bench_compare.py the README's speed comparison against other servers
docs/                  this file; configuration.md; kernel-guidelines.md; moe.md;
                       qwen4_exp.md
```

No Python, no third-party code, no offline Metal compiler: the Command Line
Tools build everything, and the kernels are compiled from source when the
model loads (the OS caches the result).

`src/model/impl/` holds two kinds of code: a format-neutral base (runtime,
tensors, gguf, matrix, gemm) and the model families (config, load, vision,
forest). Keeping them apart makes a new family cheaper to add (see the kernel
tiers in
[kernel-guidelines.md](kernel-guidelines.md#2-three-tiers-of-kernels-general-family-fused)).

## Memory

* **Weights**: repacked at load into simdgroup-fragment order (the
  register-direct GEMM layout inherited from JEVfire); 4/5/6-bit codes as a
  nibble plane plus high-bit planes, 8-bit as bytes, native GGUF Q6_K and Q8_0
  scales. Qwen3.8-27B in MLX 4-bit group 32 with 8-bit out/MLP parts is
  18.2 GB of decoder weights (5.0 and 8.5 bits per weight with their bf16
  scales and biases).
* **KV cache, paged**: one buffer, one region per full-attention layer (plus
  the MTP layer), blocks of 64 tokens. A slot (sequence state or checkpoint)
  owns a list of block ids; blocks are reference counted, so a checkpoint and
  the sequences continuing it share the prefix blocks, and a partially filled
  shared block is copied on write. Forking a conversation or a decision
  context copies no KV. In-place appends keep a partial block the slot owns
  alone (no copy per decode step).
* **State pool**: one buffer, one region per linear-attention layer for
  convolution windows and one for the Gated-DeltaNet recurrent state
  (fp32, `Hv x 128 x 128` per slot), plus the MTP draft rows. Layer-major
  regions make each dispatch's footprint one contiguous range.
* **Per-pass arena**: metadata (tokens, positions, block tables, segment and
  tile tables) is bump-allocated per pass and never rewritten while a command
  buffer that reads it is in flight.

Budgets are automatic (at most 55% of the free working set and no more than
max(4 GB, 2× the weights)) or set with `--slots`, `--kv-gb`, `--memory-gb`.

## The forest pass

Per layer, one dispatch per kernel for the whole forest:

* `rmsnorm` — fp32 residual rows to fp16 normalized rows (one dispatch per
  norm; fused variants that normalize inside the consuming GEMV exist
  (`rgemv*_n`, `*_addp`) but are **off**: they read fp32 rows in every
  threadgroup and measured slower).
* `conv_prep` — causal depthwise conv + SiLU + q/k RMS norms + gates for every
  row; taps before a tree's first row come from its parent slot's conv window
  (the all-zero slot for an empty context).
* `deltanet_tr` — the gated delta rule over *segments* (each tree's root,
  then each leaf, with their own input/output state slots) as a per-token
  recurrence with transpose-reduce shuffles (R = 8 rows per simdgroup for a
  single root, 16 otherwise). `deltanet_chunk`, an 8-token chunked WY/UT
  formulation with fp32 8x8 MMAs (`(I+L)^-1` by repeated squaring, as in
  MLX's chunked kernel), is kept for long segments behind `ESPETERO_DN_CHUNK`
  (measured 2% faster than the recurrence on prefill, so not the default).
* `attn_prep` — per-head q/k RMS norm, partial M-RoPE, fp16 Q and batch K/V.
* `flash_attention_gG` — tree-masked flash attention over the paged cache.
  Work is a table of *tiles* (8 query rows of one tree); passes with few
  tiles split the cached context over several tiles (flash decoding: 32- to
  512-key chunks, for ~256 threadgroups) and `attention_combine` merges
  them. One threadgroup serves all query heads of a KV head (GQA) so K/V are
  read once per tile.
* `attn_decode_gqa_gG` — the same attention when every tree of the pass is
  a single row (plain decode steps of any number of sequences, MTP draft
  steps): the G query heads of a KV head are the rows of the 8×8 MMA tiles,
  where `flash_attention` has 8 query rows of which 7 are padding at one
  row; two pairs of simdgroups split the head dim and take turns over 8-key
  blocks, with K/V fragments loaded straight from the pool. It reads the
  cache at the bandwidth of the same loads alone (~350 GB/s): 0.38 ms per
  27B layer at a 32K context against 1.6 ms for `flash_attention_g6`.
  Gemma 4's layers have the same layout in `g4_attention_dec` (head_dim /
  128 simdgroups per key slice, the sliding layers' rings).
* `kv_save`, `conv_state_save` — the new rows' K/V and conv windows into the
  target slots (checkpoints) of every tree.
* GEMMs over all rows, with the autotuned tiles and split-K of the original
  engine (`rgemm`, 7.1–7.5 TFLOPS at 2048 rows on the 27B); 1–2-row GEMVs
  (`rgemv`, 290–360 GB/s per matrix, 329 GB/s over a decode step); 3–8-row
  MMA GEMVs (`rgemv8`) for speculative verification, 1.5× the cost of a
  1-row step.

Dispatches are encoded on a **serial** compute encoder by default: the
concurrent encoder with barriers placed from declared read/write spans
measured 4–20% slower on prefill passes and equal on decode steps on an
M1 Max (`ESPETERO_CONCURRENT=1` switches it on; `barrier_all` and `serial`
are A/B options). Measured costs: a dependent dispatch after a barrier
~4.5–5 µs of GPU time with small kernels; inside a real 27B decode step the
630 dispatches account for 14.4 ms between isolated kernel time (62.8 ms)
and in-pass time (77.2 ms), ~23 µs each; commit-to-completion round trip
250–350 µs; CPU encode of a step 0.2 ms.

## Scheduling

One scheduler thread owns the GPU. Each step:

1. retires cancelled sequences, admits waiting ones (priority order) when a
   slot and KV room are free, looking up the longest cached chat checkpoint
   that prefixes the prompt;
2. drafts for decoding sequences (prompt lookup from an incremental n-gram
   index, else the model's MTP head, batched across sequences) — only while
   few sequences decode (`--spec-max-seqs`, default 2): verification rows are
   free while a pass is memory-bound, not once it is compute-bound (the 27B
   crosses over at ~10 rows: 16 sequences run its GEMMs at 6.3 TFLOPS);
3. builds the forest: decode steps first, then decision requests, then
   prefill chunks in the remaining row budget (a prefill never splits an
   image, and a later chunk ends exactly at the checkpoint boundary);
4. runs it, then samples / verifies drafts per sequence in parallel on the CPU
   and streams tokens to the waiting request threads.

Prompt checkpoints are exact-token (decision prefixes keyed by salt, tokens
and image digests; chat prompts up to the final assistant header, where the
next turn still matches) in an LRU over model slots. The engine runs one
warm-up pass at start-up so first requests do not pay for pipeline setup.

Decode chains: when every decoding sequence is greedy, has no drafts
and one pending token, and no prefill or decision job shares the pass, the
scheduler runs K plain decode steps as one GPU chain (`NativeModel::run_chain`):
each step is its own command buffer, ordered after the previous one by an
`MTLEvent`, embeds the previous step's argmax from a device buffer, and is
planned and committed while the previous step runs; the CPU waits once per
chain. Tokens are identical to step-by-step decoding; the gain is the
commit/wake-up/readback gap between steps (MoE: 3–5 % per step). Chains start
after 16 generated tokens (`--chain-after`), so short answers never wait on a
chain; `--chain 0` disables them.

The rule that makes chains safe: once a pass is committed, the CPU never
writes a buffer that pass may still read or write. Per-pass metadata comes
from the pass arena (not reset inside a chain), and GPU counters are reset by
the kernels that use them (the last threadgroup stores 0). A CPU `memset` of
such a counter at encode time would race the earlier steps of a chain still
running (corrupted tokens, sampled chains diverging from plain decode), so
counters are cleared by the CPU only at allocation and on non-chained passes.

Sampling: for sampled requests with 1 ≤ top_k ≤ 32 and no penalties,
the pass computes each readout's top-k scaled logits, their maximum and the
full-vocabulary sum of exp on the GPU (`topk_part`, `topk_merge`), and the CPU
builds the same distribution from those 2k + 2 numbers instead of reading the
full logits (`--no-gpu-topk` restores the CPU path).

Exact sampling (the scheme of TensorFold): the sampled token at
absolute position p is a fixed function of the row's distribution, the
request's seed (given, else a random 64-bit one) and p: Gumbel-max over the
top-k/top-p/min-p head with noise from a splitmix64 hash of (seed, p, token),
or the inverse CDF at one hashed uniform over the whole vocabulary. The same
distribution as a random draw, reproducible for a fixed seed, and a draft is
accepted exactly when it equals the token the row samples (acceptance
probability p(draft), as with the classic random test), so drafted replies equal
serial ones except at floating-point near-ties between a verify row and a
single-row step. Eligible requests (1 ≤ top_k ≤ 32, no penalties) draw on the
GPU inside the top-k merge kernel, which lets them use decode chains;
`--legacy-sampling` (mt19937 draws and a random acceptance test) and
`--no-gpu-draw` (the draw on the CPU, no chains for sampled requests) are the
alternatives.

## Reference costs (Qwen3.8-27B, MLX 4/8-bit with an MTP head, M1 Max 64 GB)

Prefill 148 tokens/s at 2048 rows; decode 13 / 34 / 88 tokens/s for 1 / 4 / 16
sequences at a 512-token context; MTP draft of 3 tokens 15.7 ms; 4-row
verification pass 117 ms; a 540-row decision tree 3.57 s; GEMM autotune
13 s at the first load, cached afterwards. Parity against the JEVfire reference on
Qwen3.8-2B: 10/10 cases within 2.2·10⁻³.

## Kernel rules

The rules every kernel follows (memory layouts, tiers, fusion, how to
measure) are in [kernel-guidelines.md](kernel-guidelines.md).
