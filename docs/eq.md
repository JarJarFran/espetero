# EQ: Espetero's quantization format

EQ is the weight format Espetero's Metal kernels are shaped for. It is a
symmetric, two-level-scaled integer format (EQ3, EQ4, EQ5, EQ6, EQ8), with
optional per-matrix codebooks, whose scale groups follow the way a simdgroup of
an Apple GPU reads a weight matrix. `espetero quantize` writes it
([quantize.md](quantize.md)); the paper's name for it is LaneQuant.

On Qwen3.6-35B-A3B, Qwen3.5-9B, Qwen3.8-27B and Spark-X2.6-4B it gives lower
KL divergence to BF16 than the GGUF and MLX files of the same size, and on
the mixture-of-experts model it also decodes faster (README, "EQ formats").

## Why another format

A batch-1 decode GEMV is bound by memory bandwidth: every weight and every
scale is read once per token. Espetero's GEMV gives each simdgroup 32 output
columns and walks the inputs in K steps of 64. Within a step, the lane with
fragment row `f_m` (0-7) multiplies the eight inputs `8j + f_m`, j = 0..7,
for the four column pairs it owns.

A format with contiguous 16-input scale groups (Q4_K, Q6_K, MLX at group 16)
puts those eight inputs in four different groups, so a lane loads and
multiplies four scales per column per step. EQ instead groups the inputs a
lane pair reads:

    sub-block t of K step s:  S(s, t) = { k = 64 s + i : floor((i mod 8) / 2) = t },  t = 0..3

Each sub-block holds 16 inputs: eight pairs of neighbouring inputs, one pair
every eight. Lanes `f_m = 2t` and `2t + 1` read exactly sub-block t, so every
lane needs one scale per column and step. On an M1 Max this brings a 4-bit
GEMV with 16-input scale granularity to MLX 4-bit's bandwidth (within 1 %),
where the same format with contiguous sub-blocks ran 5-11 % below it. The
5-, 6- and 8-bit widths run at the same bandwidth.

## Layout

A matrix has N outputs (rows) and K inputs. K is a multiple of 64; a K that is
not a multiple of 256 ends in a partial super-block (Flash-Next's expert down
projection: 640 = 256 + 256 + 128). For each matrix EQ stores:

| tensor | dtype | shape | contents |
|---|---|---|---|
| `NAME.weight` | U32 | [N, K * B / 32] | B-bit unsigned codes q, a bit stream per row |
| `NAME.scales` | I8 | [N, K / 16] | signed sub-scale c per row, K step and lane sub-block |
| `NAME.super` | F16 | [N, ceil(K / 256)] | super-scale d per row and 256 inputs |
| `NAME.codebook` | F16 | [2^B] or [E, 2^B] | codebook modes only: one table per matrix, or per expert of a stack |

Expert stacks add a leading expert dimension E to every tensor
(`switch_mlp.gate_proj.weight` is [E, N, K * B / 32]).

The weight of input k (in sub-block t of step s) and row n is

    w = half(d[n, k / 256] * c[n, s, t]) * (q - 2^(B-1))          uniform codes (EQ4, EQ5, EQ6, EQ8)
    w = half(d[n, k / 256] * c[n, s, t]) * cb[q]                   codebook codes (EQ4 cb, EQ3)

symmetric around zero, with no bias. The product `half(d * c)` is rounded to
fp16 exactly as the encoder rounds it, so kernel and encoder agree bit for bit.

Storage is B + 8/16 + 16/256 = B + 0.5625 bits per weight:

| width | bits per weight | compare |
|---|---|---|
| EQ3 (codebook) | 3.56 | |
| EQ4 | 4.56 | MLX 4-bit 4.5, Q4_K 4.5 |
| EQ5 | 5.56 | Q5_K 5.5 |
| EQ6 | 6.56 | Q6_K 6.5625, MLX 6-bit 6.5 |
| EQ8 | 8.56 | Q8_0 8.5, MLX 8-bit 8.5 |

### Scale modes

At load every EQ matrix becomes a quantized matrix of Espetero's own kernels
with a scale mode, a function constant of the pipelines that read it:

| mode | format | widths |
|---|---|---|
| 6 | EQ, uniform codes | 4, 5, 6, 8 (EQ3 uniform is not used) |
| 7 | EQ4 with a 16-entry codebook | 4 |
| 8 | EQ3 with an 8-entry codebook | 3; routed expert stacks only |

Modes 4 and 5 are reserved. GGUF files keep their own modes (Q6_K's
contiguous sub-blocks are a separate mode), so one engine runs both kinds of
file with the same scheduler, attention and MoE routing code.

### Load-time repack

The codes are repacked into the simdgroup fragment order of the width (the
MLX 4-/8-bit fragment layouts; EQ5 and EQ6 add a plane for their upper bits;
EQ3 is two planes, the low two bits in the layout of MLX's 2-bit format and
the high bit in the layout of EQ5's fifth bit). A repack kernel writes the
sub-scales of each 32-column block as [K step][sub-block][16 pair slots] of
int8 pairs and the super-scales as [K / 256][16 pair slots] of half pairs,
with column pair `4j + q` in slot `4q + j`. A lane then reads its four column
pairs' scales with one 8-byte load (sub-scales) and one 16-byte load
(super-scales) per step, forms four float2 scales and applies one per column
pair to the step's eight codes.

### Codebooks

The 16 codes of the 4-bit format are evenly spaced, while the weights of a
sub-block concentrate near zero. In codebook mode `(q - 8)` is replaced by
`cb[q]`, a 16-entry fp16 table per matrix, or per expert of an MoE stack
(32 bytes each). Each simdgroup copies the tables its 32-column block uses
into threadgroup memory before its K loop; each code then costs one
threadgroup load. The converter fits the table by Lloyd iterations on
importance-weighted errors, starting from the shape the 35B's expert tables
converge to (ends at -9.0 and +7.8). EQ3 always uses an 8-entry table, kept in
the 16-entry slots with the upper eight zero.

EQ3 exists only in the MoE expert kernels (1-2-row and 8/16/32-row expert
tiles, the fused down projection with combine, the shared-expert variants);
the engine refuses a dense EQ3 matrix at load.

## Container

An EQ checkpoint is an MLX-style safetensors directory. Matrices not listed
as EQ stay MLX affine (typically the embedding and LM head), so one file
mixes the two. `config.json` carries:

* `quantization`: the MLX per-module map, where EQ modules have
  `{"bits": B, "group_size": 16, "mode": "eq"}` (`"eqc"` for codebooks);
* `eq_format`: `{"version": 2, "preset": ..., "rules": ...}`. The version is
  2 (lane sub-blocks); this build refuses other versions and says to convert
  again.

Routers, norms, convolutions and gates stay in the source's dtype. The MTP
head, when the source has one, is written as MLX affine (8-bit, experts
4-bit) unless `--no-mtp`.

## Kernels that read EQ

Every family the engine uses for Qwen3.5 / 3.6 / 3.8, Gemma 4 and Spark:
the 1-2-row FMA GEMVs (decode), the 3-8-row MMA GEMVs (verify, small
batches), the prefill GEMMs, the MoE expert GEMVs and MMA tiles (8, 16 and
32 rows per expert tile; a row's sums equal the 8-row kernel's), the fused
MoE down projection with combine, and the dequantization paths of MTP
drafting. A shared expert of another EQ width runs beside narrower routed
experts in the same fused kernels.

## Tile images

`ESPETERO_TILES=write` (or `build/espetero_tiles`) stores the finished GPU
buffers of a Qwen3.5 / 3.6 MLX or EQ checkpoint (fragment-ordered codes,
lane-ordered scales, fused projections and expert stacks) page-aligned beside
it: `espetero-tiles.json` and `espetero-tiles.bin`. A later load whose build,
files and options match maps the segments without copying. On
Qwen3.6-35B-A3B EQ4DHB this cuts load time from 7.2 s to 0.47 s and the peak
memory footprint from 27.5 GB to 7.1 GB, with bit-identical logits. Images
depend on the build: they are not distributed with checkpoints.
`ESPETERO_TILES=0` ignores them.

## EQV (vector-quantized experts, experimental)

For Qwen3.8-Flash-Next on Macs that cannot hold its routed experts at 3 bits,
routed experts may be stored as EQV: each row of K inputs is K / 8 vectors of
eight weights, vector v = `scale[row, v / 32] * cb[code]`, with a K-bit code
(9 to 16; the presets use 13, about 1.69 bits per weight), one fp16 codebook
[2^K, 8] per stack and an fp16 scale per 256 inputs. Tensors per stack:
`NAME.vq_lo` (U8 [E, N, K/8], code bits 0-7), `NAME.vq_hi` (U32, the upper
bits), `NAME.vq_scale` (F16 [E, N, ceil(K/256)]) and `NAME.vq_cb`
(F16 [2^K, 8]). Kernels: `kernels/16_moe_vq.metal` (1-2-row FMA, 8-row MMA).

A layer may also hold its routed experts in two stacks of different formats
(`espetero quantize --expert-split`): experts [0, H) in `mlp.switch_mlp.*`
(any resident format, e.g. EQ3 codebooks) and [H, E) EQV in
`mlp.switch_mlp_b.*`. The expert sort writes the second stack's tiles, ids
rebased, to a table of their own, and each stack's kernels run on their part.
Such layers use the separate route and sort with indirect grids (no fused
route, direct single-row grids, shared-expert tiles or fused down + combine).
`tools/vq_check.mm` checks the EQV kernels and the split sort against CPU
references.

## Further reading

* [quantize.md](quantize.md): the converter, presets, calibration, time and
  memory.
* The LaneQuant paper (in preparation): the encoder, the allocation method
  and the full measurements.
