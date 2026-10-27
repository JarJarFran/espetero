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

A 4-bit preset with a trailing `c` (`eq4dhc`, `eq4xc`, ...) gives its EQ4
experts codebooks. `espetero quantize --preset NAME --print-rules` prints a
preset's rules. Rules: `GLOB=FMT;...` matched against module paths in order,
FMT one of `eql:B:16[+enc]` (B = 3, 4, 5, 6, 8), `eql:4:16:cb[:ROWS][+enc]`,
`eql:3:16:cb[:ROWS][+enc]` (8-entry codebooks; Espetero runs EQ3 for routed
expert stacks only),
`affine:B:G[:bf16|f16|f32][+enc]`, `bf16`; enc `rtn`, `mse`, `imat`, `gptq`.
MoE routers stay BF16 in every preset.

On Qwen3.6-35B-A3B (bits per weight without the MTP head): eq3dhb 4.35, KLD
0.0204 against BF16, vs GGUF IQ4_XS-imat 0.0243 at 4.40; eq3dx 4.12, 0.0241;
eq3h 3.92, 0.0316. The C++ eq3dhb has the Python checkpoint's tensors and
size, with 99.99 % of the EQ3 expert codes identical.

## Relation to the Python converter

The encoders are ports of `kld/quant_lab.py` and `kld/eq_convert.py` with
the same float roundings (fp16 / bf16 stores, ties to even, MLX's own
affine quantize rounding half away from zero, no fused multiply-adds):
given the same statistics (`tools/dev/calib_npz.py` converts the Python
harness's `.npz`), Qwen3.5-0.8B checkpoints have the same tensors, the
non-EQ tensors bit for bit, 99.97-99.998 % of the imatrix and codebook codes
identical, and Espetero KLD within 0.5 % (GPTQ spreads float-order
differences: EQ4 -0.05 %). `tools/dev/eq_compare.py A B` compares two checkpoints.

The capture tokenizes with `tokenizer.json` (as the `tokenizers` library and
Espetero's server do). transformers 5's tokenizer for this family applies an
older pre-tokenizer without the mark class (`\p{M}`), which on
calibration_v5.txt gives 1 % more tokens and shifts every later window; at
0.8B such a shift alone moves the KLD of a GPTQ EQ4 checkpoint by ~8 %, so
calibrations from the two tokenizers are different samples, not a port error.
