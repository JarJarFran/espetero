# Mixture-of-experts path (Qwen3.5-MoE: Qwen3.6-35B-A3B)

Espetero runs Qwen3.5-family mixture-of-experts models (`qwen3_5_moe`,
Qwen3.6-35B-A3B) from MLX checkpoints and from GGUF (`qwen35moe`). The expert
path started from JEVfire's MoE kernels and was reworked for decode. Parity
against mlx-vlm on a 6-bit MLX Qwen3.6-35B-A3B: argmax identical on raw
prompts through a prefill pass and through prefill + an in-place decode step,
mean |logit diff| 0.03–0.08, max 0.19–0.47 (`tests/ref/moe_ref.py` dump,
`build/parity_test <model> --ref <json>`).

Where the time goes (M1 Max, one sequence, 6-bit): the floor is 6.7 ms per
token (2.36 GB at 354 GB/s, §1) and the step does not throttle; what is left
above the floor is latency-bound kernels: the expert GEMVs at one row (each
simdgroup runs only a few K steps with one step's loads in flight), the GDN
in_proj 12352×2048, the out projection 2048×4096, attention at Hkv = 2 and
the LM head (1.1 ms).

Measuring expert GEMVs (`bench_native <moe> --micro moe`): a chain of
dispatches over one layer's experts reads 15 MB that the 48 MB system-level
cache serves at 385 GB/s, above DRAM; the harness rotates the chain over 24
layers (550 MB) and reports the per-dispatch time and bandwidth per case
(K slices, column blocks per threadgroup, direct vs indirect grids, a
diagnostic variant without the tile/rowmap prologue, the dense rgemv over the
same rows, and the whole `moe_mlp` per stage through `moe_stages`). The
in-pass `--profile` numbers run 1.4× above the isolated ones.

## 1. The model

Reference checkpoints: Qwen3.6-35B-A3B in MLX 6-bit and 8-bit
(MLX affine, group 64; router `mlp.gate` and `shared_expert_gate` 8-bit; 27 GB
and 35 GB on disk). `model_type: qwen3_5_moe`, `text_config.model_type:
qwen3_5_moe_text`.

| Geometry | Value | Note |
|:--|--:|:--|
| hidden D | 2048 | |
| layers L | 40 = 30 Gated DeltaNet + 10 full attention (every 4th) | same interleave as the dense models |
| attention H / Hkv / head dim | 16 / 2 / 256 | GQA group 8 (`flash_attention_g8` exists) |
| delta heads Hk / Hv / Dk = Dv | 16 / 32 / 128 | supported geometry |
| experts E, per token k | 256, 8 | softmax router, top-k renormalized (`norm_topk_prob` semantics as in mlx-lm) |
| expert width Ie | 512 | `moe_intermediate_size`; `Ie % 64 == 0` |
| shared expert width | 512 | `shared_expert_intermediate_size`, gated by `sigmoid(shared_expert_gate · x)` |
| vocabulary | 248,320 | LM head 508 M parameters |
| MTP head | absent in these checkpoints | drafts by prompt lookup only, or an external MTP file |

Tensor names and shapes (MLX, per layer `language_model.model.layers.<i>.`):

```
mlp.gate.{weight,scales,biases}                       [256, 2048]        router logits
mlp.shared_expert_gate.{weight,scales,biases}         [1, 2048]          scalar gate
mlp.shared_expert.{gate_proj,up_proj}.weight          [512, 2048]
mlp.shared_expert.down_proj.weight                    [2048, 512]
mlp.switch_mlp.{gate_proj,up_proj}.weight             [256, 512, 2048]   stacked experts [E, Ie, D]
mlp.switch_mlp.down_proj.weight                       [256, 2048, 512]   stacked experts [E, D, Ie]
```

`switch_mlp.*` tensors are read as `[E·Ie, D]` and `[E·D, Ie]` matrices:
expert `e` owns rows `[e·Ne, (e+1)·Ne)`, so the existing quantized layout,
repack and scale formats apply unchanged, and gate/up interleave by 8 rows
exactly as the dense MLP does (Ie is a multiple of 64).

Bytes per token at 6-bit group 64 (6.5 bits/weight):

| Per layer | Parameters | MB at 6.5 b/w |
|:--|--:|--:|
| 8 routed experts (gate, up, down) | 8 × 3.15 M = 25.2 M | 20.5 |
| shared expert + gate + router | 3.7 M | 3.0 |
| GDN layer: in_proj [qkv 8192; z 4096; a,b 64] + out_proj 2048×4096 | 33.7 M | 27.4 |
| full-attention layer: q 8192, k/v 512 each, o 4096 → 2048 | 27.3 M | 22.2 |
| **per token, 40 layers** | 2.3 B | **1.95 GB** + LM head 0.41 GB = **2.36 GB** |

At 354 GB/s the floor is **6.7 ms per token (150 tokens/s)** for one sequence;
at 8-bit (8.5 b/w) 3.1 GB and 8.8 ms (114 tokens/s). The LM head is 17% of
the bytes. KV cache: 10 layers × 2 × 512 × 2 B = 20 KB per token (164 MB read
per step at an 8K context, 0.5 ms). Recurrent state: 30 × 32 × 128 × 128 × 4 B
= 63 MB per sequence, read and written per step (0.36 ms per sequence).

## 2. Why decode is different for MoE, and what it implies

Dispatch cost dominates sooner. A dense 27B step is 77 ms and pays ~14 ms in
dispatch gaps (630 dispatches); a 35B-A3B step with the same kernel structure
would run ~480 dispatches against a 7–9 ms floor, so **the gaps would be a
quarter to a third of the step**. Every fusion that saves a dispatch per layer
is worth ~4× more here than on the 27B, and the MoE layer itself must not
become 6–9 dispatches (router GEMV, top-k, sort, gate/up, down, combine,
shared gate/up, shared down, shared gate).

Batching pays differently. With k = 8 of E = 256 experts per token, B tokens
touch on average `E · (1 − (1 − k/E)^B)` distinct experts per layer: 8, 31.5,
58, 102, 160, 226 for B = 1, 4, 8, 16, 32, 64. Expert bytes grow almost
linearly with B until ~32 sequences, while attention, GDN, shared-expert,
router and LM-head bytes (1.5 GB) are read once per pass. Bytes per token
fall from 2.36 GB (B = 1) to 1.2 (B = 4), 0.75 (B = 16) and 0.38 GB (B = 64),
and the pass stays bandwidth-bound far past the dense model's crossover
(B ≈ 10 for the 27B, where 16 rows already run its GEMMs at 6.3 TFLOPS).
The scheduler's speculation and batching policy should therefore be
architecture-aware: for MoE, batched decode is the throughput lever; for dense,
speculative rows are only free up to ~8 rows.

Few tokens per expert. At B = 1 every active expert receives exactly one
token; at B = 8, on average 1.1. A token-major grouped GEMM (rows of one
expert as the M dimension) has nothing to fill its tiles with; the expert
weight stripe must be the unit of work (MonoMoE's "weight-major" argument,
which measured 17–22% of peak bandwidth for token-major grouped GEMMs in
vLLM at B ≤ 8). JEVfire's `moe_gemv8` tiles by (expert, ≤ 8
sorted rows) × 32 columns, but always in the 8-row MMA layout, which costs
1.4× the bytes-per-time of the 1–2-row FMA GEMV.

## 3. Design

### 3.1 Loader and config (tier 0 host)

- `Config`: `E`, `k`, `Ie`, `I_shared`; `I` stays the dense MLP width and is
  0 for MoE layers; `validate()` accepts `E > 0` with `Ie % 64 == 0`,
  `k ≤ 16`, `E ≤ 1024`. Read `num_experts`, `num_experts_per_tok`,
  `moe_intermediate_size`, `shared_expert_intermediate_size`,
  `decoder_sparse_step` / `mlp_only_layers` (dense layers inside an MoE
  model) from MLX; `qwen35moe.*` (`expert_count`, `expert_used_count`,
  `expert_feed_forward_length`, `expert_shared_feed_forward_length`) from
  GGUF, with the stacked `ffn_*_exps` tensors.
- `Layer`: `moe` flag; `router` (E × D), `shared_gate` (1 × D), the shared
  expert as the existing `gate_up`/`down`; `experts_gu` (E·2Ie × D,
  interleaved by 8) and `experts_down` (E·D × Ie) as `QW`.
- Memory: experts are 26 GB of the 27 GB file; keep them quantized (no
  `dense` option), repacked at load with the existing GPU repack (the tensor
  is one matrix). Autotune covers the shared expert and attention matrices
  as for dense models; the grouped kernels have their own tiles.

### 3.2 Decode kernels (tier 0 general, one family instance)

Checkpoint formats: a MoE file may mix bit widths per tensor (oQ files:
4-bit routed experts, 8-bit shared expert, bf16 router, 5–8-bit attention).
The FMA expert GEMV takes the shared expert's width separately
(`moe_gemv{4,5,6}_*_s8`: the K loop `moe_kloop<BITSX>` is instantiated for
both widths), so the shared expert stays one tile of the grouped dispatch; a
dense router is read in fp16 by the parallel route kernel
(`moe_route_par{gate bits}h`, same fragment layout as the quantized
matrices), exactly. Converting the router to 8-bit is opt-in
(`ESPETERO_ROUTER_Q8=1`), since defaults are lossless.

Per MoE layer at one row, three dispatches:

1. the parallel router GEMV (`moe_route_par`, partial logits per K slice);
2. the gate/up expert GEMV (FMA body for 1–2-row tiles, SwiGLU epilogue),
   whose simdgroup 0 recomputes the row's softmax + top-k from the partials
   in every threadgroup (`route_topk_simd`: one simdgroup, k rounds of simd
   max / min, no barriers; threadgroup (0, 0) writes ids, weights, the shared
   gate and the tile table for the down projection);
3. `moe_down_comb`, which computes the down projections of the row's k
   experts and the shared expert (one threadgroup per 32-column block, one
   simdgroup per slot over K = Ie) and sums them in fixed order with the
   routing weights and the shared gate into the residual, finishing the next
   layer's RMSNorm in its last-arriving threadgroup (the structure of
   TensorFold's `expert_down`).

Knobs: `ESPETERO_ROUTE_IN_GEMV` (default on), `ESPETERO_ROUTE_FIN1` (a
separate barrier-free finalize when the fold is off).

Multi-row passes keep the separate steps: `moe_route` (one threadgroup per
row, sized at dispatch time from `maxTotalThreadsPerThreadgroup`: threadgroup
reduction, shared-gate dot product, softmax, top-k with lower-index
tie-break, renormalization, tile table + indirect arguments), the grouped
gate/up and down GEMVs over the tile table by indirect dispatch (tiles of an
expert and ≤ 8 sorted rows; 1–2 rows on the FMA body, 3–8 on the MMA body;
the shared expert as tile `E` reading its own matrices), and `moe_combine`
over column slices with the next RMSNorm finished by the last-arriving
threadgroup (relaxed atomics for everything another threadgroup reads),
summing in slot order. For more than 8 rows the route is a router GEMM +
`moe_topk` + `moe_sort` (threadgroup atomics) and the shared expert runs its
own GEMMs.

### 3.3 Prefill kernels

At M rows, `M·k/E` pairs land on each expert (64 per expert at M = 2048). Use
the sorted (row, expert) pairs as the M dimension of `rgemm` tiles: a tile
table of (expert, first pair, rows ≤ BM) with BM = 32 or 64, the existing
`rgemm` body with a gathered A row index (`rowmap`) and the expert's column
range; split-K as usual. This is MLX's `gather_qmm_rhs` shape (sorted
tokens, bm = 32 when `M/E < 64`, else 64). Running prefill through the
8-row tiles would give 2048 tiles of 8 rows at M = 2048 on the ALU-bound
MMA-GEMV path, several times slower than a 32-row tile. Both grouped paths
share the tile-table format; the host picks by `M`.

### 3.4 Forest integration

- MoE rows are the pass's rows; nothing changes in tree, segment or tile
  planning. The MoE MLP replaces `mlp_in` + `down` in `run_forest` and in the
  MTP layer when the MTP block is MoE.
- Last-layer pruning (`prune_last`) applies unchanged (the pruned rows are
  gathered before routing).
- Fused norms: the combine step produces the residual and its norm partials,
  so the next layer's fused-norm GEMV works as for dense layers.
- Scratch: `logits[R·E]`, `ids/w[R·k]`, `tiles`, `rowmap`, `slotpos`,
  `act[R·k·Ie]` (fp16), `ys[R·k·D]` (fp32), shared expert `act/ys`; sized
  with `ensure_scratch`.

### 3.5 Speculative decoding

Drafts are leaves of the decode tree (pending token + up to 4 drafts = 5–8
rows). For MoE those rows route independently, so a verification pass touches
up to 8·rows experts: expert bytes grow with the draft count while the
shared bytes do not. With 4 drafts the pass reads about 2.7× the expert bytes
of a plain step (E(1−(1−k/E)^5) = 39 experts vs 8) but yields 2.5–3.8 tokens
per pass at the dense model's acceptance; the break-even is measured, not
assumed (adaptive speculation times verify, draft and plain passes at load). Prompt-lookup drafts are free of the MTP head this checkpoint
lacks.

## 4. Validation

1. **Reference logits from mlx-lm** (`tests/ref/moe_ref.py`): a few prompts,
   fp32 evaluation where possible, logits of the last positions dumped to a
   file; `parity_test --ref` compares against dumped logits (the JEVfire
   dylib cannot be co-resident with a 27 GB model in 64 GB).
   Tolerance as for dense: 2·10⁻² on candidate logits after the same
   quantized weights; argmax identical on text prompts (near-ties, such as
   two candidates at probability 0.14, may swap).
2. **Routing unit test**: top-k with ties and renormalization against a
   NumPy reference; sort/tile table for random assignments (every pair
   appears once, tiles ≤ 8 rows, expert-contiguous).
3. **Determinism**: two identical passes give byte-identical logits (slot
   order in the combine; no float atomics).
4. **Layer-level check**: one MoE layer against the unbatched dense path
   (run each selected expert as a dense GEMV, sum with the routing weights)
   — an internal reference that needs no external framework.
