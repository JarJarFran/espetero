# espetero quantize

`espetero quantize` turns a Hugging Face BF16 checkpoint of the Qwen3.5 family
(Qwen3.5, Qwen3.6; dense and mixture-of-experts) into an EQ checkpoint, the
quantization format Espetero's kernels are shaped for, without Python:

    espetero quantize --model ~/models/Qwen3.6-35B-A3B --preset eq4dhb --out ~/models/Qwen3.6-35B-A3B-EQ4DHB

1. **Calibration.** Espetero loads the BF16 weights (dense matrices as fp16,
   experts as 8-bit copies) and runs prefill passes over the calibration text
   (`--calib-data`, default `file:~/.cache/mlx-lm/calibration_v5.txt`, windows
   of `--ctx` 2048 tokens, up to `--calib-tokens` 262144). Taps in the forward
   pass hand every matrix input to the quantizer, which accumulates E[x x^T]
   per input (the diagonal past 16384 inputs), per-expert diagonals for the
   experts' gate/up and down inputs and per-expert E[x x^T] for expert inputs
   up to 1024 wide. Matrices that read the same input share one statistic.
   The statistics are saved (`--calib-out`, default `OUT.calib.safetensors`)
   and can be reused with `--calib FILE`; `--calib-only` stops there.
2. **Encoding** (CPU: Accelerate BLAS/LAPACK, all cores). Each matrix gets the
   format of the first matching rule of the preset (or `--fmt`). EQ matrices
   are encoded in the column order that makes EQ's lane sub-blocks
   contiguous: GPTQ with dynamic group scales for dense matrices, the
   imatrix-weighted scale search for experts, a Lloyd fit of a 16-entry
   codebook per matrix or per expert for `eql:4:16:cb`.
3. **Output.** The MLX directory layout: per EQ matrix `NAME.weight` (packed
   codes), `.scales` (int8 sub-scales), `.super` (fp16 scale per 256) and for
   codebooks `.codebook`; MLX affine embedding / LM head; the MTP head as MLX
   affine (8-bit, experts 4-bit); routers, norms and convolutions as the
   source holds them; `config.json` with `quantization` and `eq_format`
   (version 2, preset, rules). `--tiles` also writes the tile image.

## Presets

| preset | experts | other matrices | embedding / LM head |
|---|---|---|---|
| eq4dhb | gate/up EQ4 codebook (EQ5 in layers 0-7, 36-39), down EQ5 (EQ4 codebook in layers 16-31) | EQ8 (EQ6 in layers 16-31) | MLX 8-bit |
| eq4dh / eq5dh / eq6h | EQ4 + EQ5 down / EQ5 + EQ6 down / EQ6 | EQ8 | MLX 8-bit |
| eq4x / eq5x, eq4d / eq5d | EQ4 / EQ5 (down one width up for d) | EQ6 | MLX 6 / 8-bit |
| eq3dhb | gate/up EQ3 codebook (EQ4 codebook in layers 0-3, 36-39), down EQ4 codebook | EQ8 | MLX 8-bit |
| eq3dh / eq3dx / eq3h | gate/up EQ3 codebook, down EQ4 codebook / the same / all EQ3 codebook | EQ8 / EQ6 / EQ8 | MLX 8-bit (eq3dx: LM head 6-bit) |
| fnx3 / fnx3m | EQ3 codebook (Qwen3.8-Flash-Next, below) | EQ6 / EQ5 | MLX 8-bit / 6-bit |
| fnxvq / fnxmix | EQV 13 / EQ3 codebook + EQV 13 (Flash-Next, below) | EQ5 | MLX 8-bit / 6-bit |
| eq4, eq5, eq6, eq4f, eq4s | dense-model presets (EQ4/5/6 body, sensitive projections EQ6) | | |

Dense presets for Qwen3.5 / Qwen3.6 dense models (linear attention, full
attention every 4th layer), by tensor class and layer position; "first" and
"last" are the first and last quarter of the layers (L/4, from config.json).
Every matrix is GPTQ-encoded; EQ4 means EQ4 with codebooks. Each is sized at or
under the GGUF named. Set on KL probes over calibration text (Qwen3.5-9B).

| preset | linear attention | attention | MLP | embedding / LM head | 9B bpw |
|---|---|---|---|---|---|
| eqd4m (Q4_K_M) | first: qkv/z EQ6, out EQ8; else EQ4 | k/v/o EQ6, q EQ4 | down EQ5 in first+last, else EQ4 | MLX 5-bit / EQ5 | 5.03 |
| eqd4l (UD-Q4_K_XL) | as eqd4m | as eqd4m | gate/up EQ5 in first+last, down EQ5, else EQ4 | MLX 5-bit / EQ5 | 5.30 |
| eqd5 (UD-Q5_K_XL) | first: EQ8; else EQ5 | k/v/o EQ6, q EQ4 | first+last: gate/up EQ5, down EQ6; else EQ5 | MLX 6-bit / EQ6 | 6.00 |
| eqd6 (Q6_K) | first: EQ8; else out EQ8, qkv EQ5, z EQ6 | EQ6 | EQ6 | MLX 6-bit / EQ6 | 6.65 |

in_proj_a/b are EQ8 in every dense preset. `--print-rules` takes the layer
count from `--model` or `--layers N` (default 32).

These widths do not carry over to Qwen3.8-27B: its own probes, on held-out
text (Wikipedia, C4, Python), gave `eqd6-27b`: every matrix EQ6, the MLPs of
the first 5/16 of the layers (L0-19) EQ5, v_proj, LM head and in_proj_a/b
EQ8, the embedding MLX 5-bit, the vision tower BF16. It needs a calibration
text with prose as well as code: with one of wikitext-103, C4 and PG19 train
plus calibration_v5, chat and Python (452k tokens), KLD 0.00132 at 6.417 bpw
against UD-Q6_K's 0.00142 at 6.431, both run by Espetero (−7 %, 95 % interval
−14 % to +1 %: lower but not significantly; fp32 BF16 reference, wikitext-2
test and C++/Metal code); with calibration_v5 alone 0.00149.

**Other families.** Dense Gemma 4 and Spark-X2.5 / X2.6 (`spark2_5`) convert
with `--fmt` rules. Spark-X2.6-4B's `c3` (5.21 bits per weight; KLD 0.026
against BF16, MLX 4-bit 0.238, the official Q4_K_M 0.071 at 6.19):

    espetero quantize --model Spark-X2.6-4B --calib calib.safetensors --fmt "$(cat rules_c3.txt)" --out Spark-X2.6-4B-EQ-c3

where `rules_c3.txt` holds (one line): `*g_proj=bf16;model.layers.0.*=eql:6:16+gptq;`
... the same for layers 1-3, `model.layers.4.*=eql:5:16+gptq;` ... for layers 5-7, then
`model.embedding=affine:6:64:f16;*q_k_v_proj=eql:5:16+gptq;*out_proj=eql:5:16+gptq;*=eql:4:16:cb+gptq`:
layers 0-3 EQ6, layers 4-7 and the attention projections EQ5, everything else EQ4 with codebooks,
the tied embedding / LM head MLX 6-bit. `c4` (5.67 bits per weight, KLD 0.018) raises layers 0-7 and
the attention projections to EQ6 and the down projections to EQ5.

A 4-bit preset with a trailing `c` (`eq4dhc`, `eq4xc`, ...) gives its EQ4
experts codebooks. `espetero quantize --preset NAME --print-rules` prints a
preset's rules. Rules: `GLOB=FMT;...` matched against module paths in order,
FMT one of `eql:B:16[+enc]` (B = 3, 4, 5, 6, 8), `eql:4:16:cb[:ROWS][+enc]`,
`eql:3:16:cb[:ROWS][+enc]` (8-entry codebooks; Espetero runs EQ3 for routed
expert stacks only),
`affine:B:G[:bf16|f16|f32][+enc]`, `bf16`; enc `rtn`, `mse`, `imat`, `gptq`.
MoE routers stay BF16 in every preset.

On Qwen3.6-35B-A3B (bits per weight without the MTP head): eq3dhb 4.35, KLD
0.0204 against BF16, vs llama.cpp IQ4_XS with an imatrix 0.0279 at 4.32 (both
run by Espetero; 0.0243 dequantized into the reference); eq3dx 4.12, 0.0241;
eq3h 3.92, 0.0316.

**Router-weighted expert statistics.** `--calib-expert-weight router` weights
each routed (token, expert) pair of an expert's statistics by its routing
weight instead of counting every pair once. With it and a broader calibration
text (calibration_v5 plus chat rendered with the chat template, multilingual
chat, tool-use conversations and C4 prose, 262K tokens; every routed expert sees at least 264 rows, against 19 with
the default 112K tokens), Qwen3.6-35B-A3B's
`eq6h` lowers its KLD by 3.8 % (95 % interval −6.9 % to −0.7 %; 0.00378), but
`eq4dhb` (−1.2 %, not significant) and `eq3dhb` (+0.5 %) do not change; the
default stays `count`.

## Qwen3.8-Flash-Next (qwen4_exp)

    espetero quantize --model Qwen3.8-Flash-Next --preset fnx3 --keep-experts keep.json \
        --calib-model Qwen3.8-Flash-Next-oQ4e-mtp --hessian-max 8192 --out Flash-Next-EQ3

* **Sources.** The BF16 checkpoint (transformers names; fused `experts.gate_up_proj` /
  `down_proj` split into `switch_mlp.*`, the MTP head kept as `mtp.*` with its experts split
  the same way, the vision tower left out, norm weights kept zero-centred: the loader adds 1),
  or an MLX checkpoint of the model such as oQ4e-mtp (its affine matrices dequantized; for
  tests while no BF16 copy is at hand). The output uses the oQ4e-mtp names.
* **Calibration.** The BF16 model (360 GB) does not fit in memory: `--calib-model` captures the
  statistics with the forward of another checkpoint of the model (the oQ4e-mtp file; its experts
  are read whole per pass from the SSD, no expert cache). The taps cover every matrix: attention /
  DeltaNet / indexer inputs, mixer outputs, router and shared-expert inputs, routed-expert rows,
  both hyper-connection blocks of every layer and the final mixer (normed streams, 10240 wide, and
  the rank-320 activations), the n-gram rows and the LM head input. `--hessian-max 8192` keeps only
  the diagonal of the 10240-wide inputs (97 matrices of 420 MB otherwise). Qwen4-Exp files also
  hold each expert's routed-row count (`N:` tensors) for pruning.
* **Pruning.** `--keep-experts FILE` (JSON `{"0": [ids], ..., "47": [ids], "mtp": [ids]}`; each
  count a multiple of 8, layers may differ; list order = new expert id; without `"mtp"` the MTP
  layer keeps its 512) slices the expert stacks, the routers' rows and the per-expert statistics;
  `num_experts` becomes the largest count and `num_experts_per_layer` lists them (the loader reads
  each layer's count from its stacks). Top-k routing over the kept experts with `norm_topk_prob` is
  exactly the original routing restricted to them.
* **Formats.** `fnx3`: routed experts (trunk and MTP) EQ3 codebooks, every other matrix EQ6 GPTQ
  (`fnx3m`: EQ5), LM head MLX 6-bit, embedding MLX 8-bit, routers BF16. The n-gram table (51.2 B
  values, read from disk at run time) stays MLX affine group 32 with bf16 scales, `--ple-bits 4`
  (the oQ4e layout, 100 B per row, 32 GB; an MLX source of that width is copied) or 8 (180 B per
  row). EQ checkpoints keep their routed experts resident (no streamed form), and load their own
  `mtp.*` head.
* **Widths not a multiple of 256.** EQ rows that are a multiple of 64 wide end in a partial
  super-block: `ceil(in / 256)` fp16 super-scales per row (the expert down projection 640 = 256 +
  256 + 128, the hyper-connections' up projection 320 = 256 + 64), the same scale-buffer layout the
  NVFP4 path uses; every width that is a multiple of 256 is unchanged.
* **EQV and two expert formats** (experimental, Macs with 48-64 GB; [eq.md](eq.md#eqv-vector-quantized-experts-experimental)).
  `fnxvq[K]`: every routed expert EQV with 2^K-entry codebooks (K 13 by default, ~1.69 bits per
  weight), the rest as `fnx3m`. `fnxmix` with `--expert-split H|FILE`: each layer's first H experts
  in the `--keep-experts` order (most routed first) EQ3 codebooks, the rest EQV 13 in a second stack
  (`mlp.switch_mlp_b.*`); `FILE` is JSON `{"<layer>": H, ...}`. The EQV down projections use each
  expert's own Hessian when the calibration has one (`--expert-hessian-max 640` at capture). Example
  (Flash-Next for 64 GB Macs, 3.31 bits per weight):

      espetero quantize --model Qwen3.8-Flash-Next --preset fnxmix --keep-experts order.json \
          --expert-split 408 --calib calib-x640.safetensors --no-mtp --ple-bits 4 --out Flash-Next-mix408

* `--only GLOB,...,!GLOB` writes part of a checkpoint; `tools/dev/fnx_overlay.py BASE PART VIEW` lays
  such a part over another checkpoint of the model (symlinks, merged index and config), e.g. a few
  layers converted to EQ over the oQ4e-mtp file, or a pruned conversion that reuses the file's
  n-gram table.

## Time and memory

The encoders run on the CPU (Accelerate, all cores); the calibration forward
runs on the GPU. Measured with `/usr/bin/time -l`:

| job | Mac | time | peak footprint |
|---|---|---|---|
| Qwen3.6-35B-A3B calibration capture (`--calib-only`, 112K tokens) | M4 Max | 4 min | 64 GB |
| Qwen3.6-35B-A3B calibration capture, 262K tokens, router-weighted | M4 Max | 36 min | 94 GB |
| Qwen3.6-35B-A3B `eq4dhb` / `eq6h` / `eq3dhb` (single-threaded BLAS) | M4 Max | 30 / 20 / 34 min | 6 GB |
| Qwen3.6-35B-A3B `eq4dhb` (all cores) | M4 Max | 8.5 min | 7.5 GB |
| Qwen3.8-27B `eqd6-27b` (single-threaded BLAS) | M4 Max | 43 min | 12 GB |
| Qwen3.5-9B `eqd4m` / `eqd4l` / `eqd5` / `eqd6` (single-threaded BLAS) | M1 Max | 34 / 31 / 23 / 19 min | 11 GB |
| Spark-X2.6-4B c3 / c4 (single-threaded BLAS) | M1 Max | 27 / 16 min | 3 GB |
| Qwen3.8-Flash-Next `fnx4h` rules / `fnxmix` / `fnxvq` (all cores) | M4 Max | 70 / 88 / 117 min | 12 / 15 / 12 GB |

The BF16 source is memory-mapped: the resident size grows with the pages the
encoder has read (56 GB for the 35B, 123 GB for Flash-Next) but those pages
are file cache that macOS reclaims; the footprint column is what the process
holds. Calibration statistics are large (the 35B's 15 GB, Flash-Next's with
per-expert down Hessians 50 GB): save them once with `--calib-out` and reuse
them. A model whose BF16 weights do not fit in memory is captured in layer
ranges (`--calib-layers A:B`, merged with `tools/calib_merge.py`) or, for
Flash-Next, with the forward of a smaller checkpoint (`--calib-model`).

GPTQ sums in the order the BLAS threads take: with
`VECLIB_MAXIMUM_THREADS=1` a conversion is byte-for-byte reproducible (every
0.8B preset and the 35B `eq4dhb` were checked), with all cores it differs in
the last bits of some codes (KLD within noise, about 2 %).

## Checking a conversion

`tools/dev/eq_compare.py A B` compares two checkpoints tensor by tensor
(names, dtypes, shapes, the share of identical codes). `tests/kld_dump`
writes the final-normed hidden states of a checkpoint over a KLD reference's
windows, for scoring against a BF16 reference outside the engine.

The capture tokenizes with `tokenizer.json` (as the `tokenizers` library and
Espetero's server do). transformers 5's tokenizer for this family applies an
older pre-tokenizer without the mark class (`\p{M}`), which on
calibration_v5.txt gives 1 % more tokens and shifts every later window; at
0.8B such a shift alone moves the KLD of a GPTQ EQ4 checkpoint by ~8 %, so
calibrations from the two tokenizers are different samples.
