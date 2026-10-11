# Qwen4-Exp (`qwen4_exp`, Qwen3.8-Flash-Next) — architecture reference

The model's math as Espetero implements it, with the checkpoint facts of the
`Qwen3.8-Flash-Next-oQ4e-mtp` file (4-bit with an MTP head). The engine's
kernels and loader cite the sections below. Qwen3.8-Flash-Next is a 125B-A6B
mixture of experts built from Qwen3.5's blocks plus three new ones:
hyper-connections (a four-stream residual), Qwen sparse attention and a hashed
n-gram embedding.

Contents: 0 sources and conventions · 1 decoder structure · 2 hyper-connections ·
3 Gated DeltaNet · 4 Qwen Sparse Attention · 5 n-gram embedding + PLE ·
6 MoE · 7 MTP · 8 vision + M-RoPE · 9 memory traffic and state ·
10 reference discrepancies and decisions · A tensor inventory ·
B per-layer quantization.

## 0. Sources, conventions, checkpoint facts

| Tag | File (transformers 5.17 / mlx-vlm 0.7.3, paths relative to `site-packages/`) |
|:--|:--|
| **[T]** | `transformers/models/qwen4_exp/modeling_qwen4_exp.py` (canonical math) |
| **[TC]** | `transformers/models/qwen4_exp/configuration_qwen4_exp.py` |
| **[M]** | `mlx_vlm/models/qwen4_exp/language.py` |
| **[MC] [MQ] [MF]** | `mlx_vlm/models/qwen4_exp/config.py`, `qwen4_exp.py` (sanitize), `fp8.py` |
| **[MK]** | `mlx_vlm/models/qwen4_exp/qsa_kernel.py` |
| **[MP]** | `mlx_vlm/models/qwen4_exp/ple_storage.py` |
| **[MD]** | `mlx_vlm/speculative/drafters/qwen4_exp_mtp/qwen4_exp_mtp.py` |
| **[MB] [MS]** | `mlx_vlm/speculative/drafters/mtp_base.py`, `mlx_vlm/speculative/mtp.py` |
| **[M35] [M35M] [MG]** | `mlx_vlm/models/qwen3_5/language.py`, `qwen3_5_moe/language.py`, `qwen3_5/gated_delta.py` |

**Trust rule.** [T] defines the base model's math (upstream reference, class
by class a Qwen3.5 copy plus the new modules; a diff of the shared classes
against `qwen3_5`/`qwen3_5_moe` shows only the changes listed in §3–§6).
mlx-vlm is the only source for what [T] does not implement: the MTP head
([T]:1322 drops `mtp.*`), the external PLE storage and the QSA Metal kernel.
Where the two differ, §10 gives the decision.

**Notation.** `D = 2560` hidden, `S = 4` streams (`hc_count`), `W = S·D =
10240` hyper-connection width, `r = 320` (`hc_lowrank`). Token positions `t`
are 0-based; `n = t + 1` tokens are visible to token `t`. Linear layers are
`y = A x` with `A` stored `[out, in]`. `σ` = sigmoid, `silu(x) = x·σ(x)`.

* `RMSNorm₁₊w(x)` — the Qwen4 norm: `x · rsqrt(mean(x²) + 1e-6) · (1 + w)`,
  computed in fp32, result cast to the activation dtype ([T]:152-172,
  [M]:740-768). Checkpoint weights are **zero-centred**; Qwen3.5 MLX files
  fold the `+1` at conversion (mlx-vlm `models/qwen3_5/qwen3_5.py` sanitize
  adds 1.0 to `input_layernorm`, `post_attention_layernorm`, `model.norm`,
  `q_norm`, `k_norm`), Qwen4 files do not ([MQ]:22-68 never shifts, and
  `Qwen4ExpRMSNorm` adds 1 at run time, [M]:750). Used by `hc_norm`, `q_norm`,
  `k_norm`, the indexer's `q_layernorm`/`k_layernorm`, the PLE's three norms
  and the MTP's `pre_fc_norm_*`.
* *Grouped* `RMSNorm₁₊w` over a 10240-vector: the same, separately per
  2560-slice (per stream), with the full 10240-long weight ([T]:161-165).
* `RMSNorm_w` — plain weight, no `+1`: only the GDN gated norm
  (`linear_attn.norm.weight`, stored ≈ 1.0) ([T]:179-196).
* MLX affine quantization: `w = q·s + b` per group of `G` along `in`; codes
  U32 `[out, in·bits/32]`, BF16 `scales`/`biases` `[out, in/G]`.

### Checkpoint geometry (`config.json`, text_config unless noted)

| Field | Value | Field | Value |
|:--|:--|:--|:--|
| `hidden_size` D | 2560 | `num_hidden_layers` | 48 (36 GDN + 12 QSA) |
| `vocab_size` | 248,320 | `tie_word_embeddings` | false (top level and text) |
| `rms_norm_eps` | 1e-6 | `max_position_embeddings` | 262,144 |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 24 / 2 / 256 (GQA 12) | `rope_parameters` | default, θ = 1e7, `partial_rotary_factor` 0.25 (64 dims), `mrope_section` [11, 11, 10], `mrope_interleaved` true |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 48 | `linear_key_head_dim` / `linear_value_head_dim` / `linear_conv_kernel_dim` | 128 / 128 / 4 |
| `mamba_ssm_dtype` | float32 | `output_gate_type` | `sigmoid` (GDN gated norm) |
| `num_experts` / `num_experts_per_tok` / `norm_topk_prob` | 512 / 10 / true (default) | `moe_intermediate_size` / `shared_expert_intermediate_size` | 640 / 640 |
| `hc_count` / `hc_lowrank` | 4 / 320 | `indexer_n_heads` / `indexer_kv_heads` / `indexer_head_dim` | 4 / 1 / 128 |
| `indexer_budget` / `indexer_compress_ratio` | 2048 tokens / 4 (512 blocks) | `layer_types` | `linear_attention` ×3, `full_attention` (renamed `qwen_sparse_attention`, [TC]:180-184, [MC]:96-100) |
| `ple_layer_ids` (1-indexed) | [2] → layer index 1 | `ple_embed_dim` / `ple_conv_kernel_size` | 2560 / 4 |
| `ngram_size` / `heads_per_ngram` | 3 / 8 (16 heads) | `ngram_vocab_size_base` / `make_ngram_vocab_size_divisible_by` / `split_ngram_parts` | 20,000,000 / 128 / 128 |
| `seed` | 1234 (default, not in the file) | `eos_token_id` (text) | 248044 `<\|endoftext\|>` (top level: [248046, 248044]) |
| `bos_token_id` / pad (generation_config) | 248044 / 248044 | image / video / vision start / end ids (top level) | 248056 / 248057 / 248053 / 248054 |
| `mtp` | `{hybrid: true, layer_types: [full_attention], mtp_use_hidden_state_from_layer: null, num_hidden_layers: 1, rope_theta: 1e7}` | `mtp_num_hidden_layers` / `mtp_use_dedicated_embeddings` | 1 / false |
| vision_config | depth 27, hidden 1152, 16 heads, MLP 4304 `gelu_pytorch_tanh`, patch 16, temporal 2, merge 2, 2304 pos. embeddings, `out_hidden_size` 2560, no deepstack | quantization (top level) | default 4-bit g64 affine + 631 per-module overrides (Appendix B) |

Parameter counts (from the shapes): trunk 125.7 B (routed experts 120.8 B),
**6.6 B active per token**; n-gram table 51.2 B (320,001,536 rows × 160);
MTP head 2.6 B; vision 0.45 B. oQ4e file ≈ 106 GB: experts 67.95 GB (4-bit
g64, 2.76 MB each), PLE table 32.0 GB (Q4 g32, 100 B/row), MTP ≈ 1.5 GB,
vision ≈ 0.9 GB (bf16), rest of the trunk ≈ 4.0 GB (incl. the 675 MB
embedding table).

## 1. Overall decoder structure

```
e_t    = Embed(id_t)                          # [2560]; embed_tokens, 8-bit g64 (row gather)
H⁰_t   = [e_t, e_t, e_t, e_t]                 # [4, 2560] = 10240, stream-major: element (s, c) at s·2560 + c
for l in 0 … 47:
    if l == 1:  H ← H + PLE(H, ids)           # §5; ple_layer_ids = [2] is one-indexed
    x, inj = HC_attn^l(H)                     # §2; x [2560], inj [4]
    y      = GDN^l(x)   if l mod 4 ≠ 3        # §3; layers 0,1,2, 4,5,6, …
             QSA^l(x)   if l mod 4 = 3        # §4; layers 3, 7, 11, …, 47
    H_s   ← H_s + inj_s · y      (s = 0…3)
    x, inj = HC_mlp^l(H)
    y      = MoE^l(x)                         # §6; every one of the 48 layers
    H_s   ← H_s + inj_s · y
h      = Mixer(H)                             # §2; hyper_connection_mixer — the only "final norm"
logits = LMHead(h)                            # lm_head [248320, 2560], 8-bit g64, untied
MTP input = H (pre-mixer, [10240])            # §7
```

References: stream init `hidden_states.repeat(1, 1, hc_count)` [T]:1480 /
`mx.tile` [M]:1712; layer body [T]:1283-1310 / [M]:1540-1560; PLE index
[T]:1268-1269; mixer [T]:1393,1493 / [M]:1593,1737; head [T]:1604, 2414.

What is **absent** compared with Qwen3.5/3.6: `input_layernorm`,
`post_attention_layernorm` and `model.norm` (the `hc_norm` of each HC block
and of the mixer take their place); dense MLP layers (all 48 layers are MoE,
no `mlp_only_layers`/`decoder_sparse_step`). `lm_head` and `embed_tokens` are
separate tensors (`tie_word_embeddings: false`; [T]'s `_tied_weights_keys`
is inert). Top-level tensors: `language_model.model.embed_tokens.*`,
`language_model.lm_head.*`,
`language_model.model.hyper_connection_mixer.{hc_norm.weight,
input_mix_weight_down.*, input_mix_weight_up.*}`; per layer
`language_model.model.layers.<l>.…` (Appendix A).

Positions: text rows carry `(t, t, t)`; the model keeps the whole position
history because the QSA indexer ropes block keys at their block's first
token ([T]:1430-1453 caches `position_ids` in the cache; [M]:110-119 keeps
`index_position_ids`).

## 2. Hyper-connections ("gated residual", `hc_count` 4, `hc_lowrank` 320)

Per token the residual is **4 streams × 2560 = 10240 values** (bf16 in both
references, fp32 in Espetero). Each of the 2 × 48 blocks (token mixer, MoE) and
the final mixer is a `Qwen4ExpTextGatedResidual` ([T]:1003-1031,
[M]:1162-1197, pointwise parts [M]:1740-1758):

```
N      = GroupedRMSNorm₁₊w(H; hc_norm.weight [10240])          # per stream, fp32 math
a      = silu( (A_down · N) / 4 )          A_down = input_mix_weight_down [320, 10240]     → [320]
m      = σ( A_up · a )                     A_up   = input_mix_weight_up   [10240, 320]     → [4, 2560]
x[c]   = ¼ Σ_s m[s, c] · N[s, c]                                                            → [2560] block input
inj[s] = 2 · σ( (A_inj · N)[s] / 4 )       A_inj  = block_inject_weight   [4, 10240]       → [4], in (0, 2)
y      = block(x)                                                                           → [2560]
H'[s, c] = H[s, c] + inj[s] · y[c]         (H = the un-normalized input, returned as hyper_input)
```

* Order: normalize → (low-rank mix, injection gate) both from `N` → block on
  `x` → inject into the **raw** `H` ([T]:1021-1031, [T]:1302-1309).
* Gating: silu on the rank-320 bottleneck, sigmoid for the per-channel stream
  weights, `2·σ(·/4)` for the per-stream injection. **No softmax anywhere**,
  no static (input-independent) mixing parameters, no stream-to-stream
  residual matrix: each stream keeps an identity residual and streams only
  interact through `x` (mean over streams) and the gates.
* `/4` (= `hc_count`) is applied to the projections before silu and before
  the injection sigmoid (exact power of two).
* Mixer (`use_combine=False`): same `N, a, m, x`, returns `x` only; no
  `block_inject_weight` ([T]:1028-1029, [M]:1191-1192). `h = x` feeds the LM
  head directly.
* Weights per block: 3.28 M + 3.28 M + 41 K; 96 blocks + mixer = 640 M
  parameters, **405 MB read per token** in oQ4e (4–8-bit g64, Appendix B).
* Shapes for the GEMV kernels: `A_down` K = 10240 → 320 outputs (10 column
  blocks of 32: needs split-K); `A_up` K = 320 (5 groups of 64) → 10240
  outputs; `A_inj` K = 10240 → 4 outputs (same input as `A_down`: fuse as
  rows 320–323 when the widths match: 46 of 48 attention blocks and 45 of 48
  MoE blocks; layers 19, 21 (both) and 29 (MoE block) differ, Appendix B).

## 3. Gated DeltaNet layers (36 layers)

Projections split exactly like Qwen3.5 (not Qwen3-Next's fused `qkvz`/`ba`):

```
p = W_qkv x   [10240] = [q 16×128 | k 16×128 | v 48×128]     in_proj_qkv [10240, 2560]
z = W_z  x    [6144]  = 48×128                              in_proj_z   [6144, 2560]
b = W_b  x    [48],  a = W_a x  [48]                        in_proj_b / in_proj_a [48, 2560]
c_t[ch] = silu( Σ_{j=0..3} w[ch, j] · p_{t−3+j}[ch] )       conv1d.weight [10240, 4, 1] (MLX [out, k, in/groups]),
                                                             causal, zero history; state = last 3 pre-conv p rows
q̂ = q / √(Σ q² + 1e-6) / √128,   k̂ = k / √(Σ k² + 1e-6)     per key head (FLA l2norm, eps after the sum)
value head j uses key head ⌊j / 3⌋                          (repeat_interleave, [T]:582-584)
β_j = σ(b_j),   g_j = −exp(A_log_j) · softplus(a_j + dt_bias_j),   decay_j = exp(g_j)
S_j ∈ ℝ^{128_k × 128_v}, fp32:   S ← decay·S;  u = Sᵀk̂;  S ← S + k̂ ⊗ β(v − u);  o_j = Sᵀq̂
out_j = RMSNorm_w(o_j; norm.weight [128]) ⊙ σ(z_j)          ← sigmoid, "output_gate_type"
y = W_out · concat_j out_j                                  out_proj [2560, 6144]
```

Sources: module [T]:465-625 (projections 506-509, conv 483-490, gates 579-584,
`use_qk_l2norm_in_kernel=True` 596/609, l2norm 256-259, query scale 429-430,
recurrence 442-453, norm + out 619-624); mlx-vlm reuses the Qwen3.5 module,
overriding the gated norm and the q/k normalization (plus a row-invariant
`a`/`b` projection, numerics only) ([M]:789-810), and the Qwen3.5 recurrence
([MG]:126-169, g/β 262-268; state `[B, Hv, Dv, Dk]` fp32).

Differences from the Qwen3.5/3.6 GDN:

| Item | Qwen3.5/3.6 | Qwen4-Exp |
|:--|:--|:--|
| Heads | 35B-A3B 16 K / 32 V; 27B 16 K / 48 V | 16 K / 48 V (ratio 3, as the 27B); C = 10240, Vd = 6144 |
| Projection split | in_proj_qkv, in_proj_z, in_proj_b, in_proj_a | identical |
| Gates g, β | as left | identical ([T] class diff: only the norm activation changed) |
| q/k normalization | Espetero/mlx-vlm Qwen3.5: `x·rsqrt(Σx²/128 + 1e-6)` → effective eps 128·1e-6 ([M35]:1070-1074; `conv_prep` and `gdn_decode_fused` in `30_deltanet.metal`) | FLA l2norm, eps 1e-6 after the sum ([T]:256-259, [M]:798-804). Same kernel formula with **eps = 1e-6/128**; the fused kernel then needs a second eps (its gated norm keeps 1e-6) |
| Output gate | `silu(z)` ([T] Qwen3.5 RMSNormGated) | **`σ(z)`** (`output_gate_type: sigmoid`, [T]:499-501, [M]:771-786) |
| Norm weights | gated norm plain | gated norm plain (≈ 1.0 in the file) |
| State dtype | fp32 | fp32 (`mamba_ssm_dtype: float32`) |

Per sequence: recurrent state 48 × 128 × 128 × 4 B = 3.15 MB per layer, **113.2
MB for 36 layers**; conv state 3 × 10240 values per layer (bf16 in the
references, 123 KB/layer in Espetero's fp32).

## 4. Qwen Sparse Attention (12 layers: 3, 7, …, 47)

### 4.1 Main attention (identical to Qwen3.5's gated attention except for the token selection)

```
[q_h | gate_h] = (W_q x)[h]      q_proj [12288, 2560], per head 512 = [q 256 | gate 256] ([T]:867-870)
k = W_k x [2, 256],  v = W_v x [2, 256]                   k_proj / v_proj [512, 2560]
q_h ← RMSNorm₁₊w(q_h; q_norm [256]);  k ← RMSNorm₁₊w(k; k_norm [256])      (zero-centred weights)
q_h, k ← RoPE(·, pos_t)   dims 0..63 only, rotate_half pairs (i, i+32), inv_freq_i = 1e7^(−2i/64), i < 32,
                          interleaved M-RoPE: frequency i uses h-position if i%3=1 and i<33, w-position if
                          i%3=2 and i<30, else t-position ([T]:140-149; Espetero rope_sel, config.h:38-45)
cache K (normed, roped) and V (raw)
o_h = Σ_{j ∈ Sel(t)} softmax_j(q_h·k_j / 16) v_j          KV head ⌊h/12⌋ (GQA 12), scale 256^-½
y = W_o · concat_h (o_h ⊙ σ(gate_h))                       o_proj [2560, 6144]
```

[T]:819-901 (indexer mask combined with the causal mask 855-860).

### 4.2 The indexer

`self_attn.indexer.index_qk_proj` [640, 2560], `q_layernorm` / `k_layernorm`
[128] (zero-centred). [T]:673-779, [M]:813-1080.

```
[q̃_0 … q̃_3 | k̃raw] = W_iqk x                 4 query heads × 128, one key head × 128
q̃_h ← RoPE(RMSNorm₁₊w(q̃_h; q_layernorm), pos_t)     RoPE on dims 0..63 of 128, same cos/sin as 4.1
cache kraw_t (no norm, no RoPE)                 [T]:712-717, [M]:889-895
Blocks  B_b = {4b, 4b+1, 4b+2, 4b+3}           consecutive visible tokens from the first visible one
                                               (absolute position 0 for an unpadded sequence)
n = t+1,  nb = ⌊n/4⌋ complete blocks,  tail T_t = {4·nb, …, t}  (n mod 4 ≤ 3 tokens)
K̂_b = RoPE( RMSNorm₁₊w( round_bf16( mean_{i∈B_b} kraw_i ) ; k_layernorm ), pos_{4b} )   mean in fp32
score_b = (1/√128) Σ_{h=0..3} ReLU(q̃_h · K̂_b)          fp32 ([T]:752-755, [M]:1004-1010)
Sel(t) = ⋃_{b ∈ topk(score, min(512, nb))} B_b  ∪  T_t          |Sel(t)| ≤ 2048 + 3
```

Exact loop in [T]:729-764: visible indices from the mask row; pooled keys
741-744; block RoPE at `group_starts` (first token of the block) 745-750;
`scores.topk(min(block_topk, num_complete_blocks))` 757; token remap 759;
tail 762-763; boolean mask over the key axis 766-779 (width
`budget + ratio − 1` = 2051 index slots, 723-728).

Properties an implementation must reproduce:

* **No guaranteed local window and no sink.** Only the ≤ 3-token tail is
  always attended. If `n mod 4 = 0` the query's own token sits in a complete
  block that must win the top-k to be attended; token 0 has no special status.
  [M] agrees (`tail = [4·complete, query_end)`, [M]:1068-1072; kernel tail
  [MK]:111-140).
* Selection is per query row (prefill rows differ; drafts differ per branch).
* The `1/√128` does not change the ranking; ReLU and the head sum do.
* Ties: `torch.topk` and `mx.argpartition` break ties arbitrarily. Ties are
  realistic at score 0 (a block where all four heads' dots are ≤ 0): when
  fewer than 512 blocks score > 0, which zero-score blocks enter is
  implementation-defined in both references. Pick one deterministic rule and
  exclude such rows from exact selection parity.
* The main attention's KV cache is unchanged by the indexer; the indexer only
  masks which cached tokens a query reads.

### 4.3 Equivalence with dense attention

If `nb ≤ 512`, i.e. **n ≤ 2051 visible tokens**, top-k selects every complete
block and the tail completes the range: `Sel(t) = {0, …, t}`, exactly dense
causal attention. [T] does this implicitly (`topk(min(512, nb))`); [M] makes
it explicit: decode returns no selection when `key_len // 4 ≤ 512`
([M]:900-902) and `build_mask` falls back to the causal mask when
`complete_counts ≤ block_topk` ([M]:1076-1079). So **dense causal attention
is exact for every query with n ≤ 2051** (prompt + generated
tokens), including the MTP layer. An implementation must still cache `kraw` (and may build
the block keys) from the first token, or it cannot switch to QSA when the
context passes 2051. Beyond 2051, dense attention reads more tokens than the
model was trained to see: an approximation, not the model.

### 4.4 What is cached per token (per QSA layer, fp16/bf16)

| Item | Bytes / token | 12 layers | Note |
|:--|--:|--:|:--|
| K (2 heads × 256) + V (2 × 256) | 2,048 | 24,576 | as Qwen3.5 (kv_dim 512) |
| raw indexer key `kraw` (1 × 128) | 256 | 3,072 | [T] `DynamicIndexedLayer.indexer_keys` `[B, T, 128]`; [M] `index_keys` |
| pooled block key `K̂_b` (128 per 4 tokens) | 64 | 768 | [M] caches them (`index_block_keys`, extended block by block, [M]:913-998); [T] recomputes every step |
| block-start positions (3 × int32) | ≤ 12 | shared by all layers | needed to rope `K̂_b` when images give 3-D positions ([M] keeps `index_position_ids`) |
| **Total** | **2,368** | **28,416 (27.75 KiB)** | + one more QSA layer (2,368 B) for the MTP head |

Per decode step at context N: dense regime reads 24 KiB·N of K/V (50 MB at
N = 2048); sparse regime reads ≤ 2051 tokens of K/V (50 MB, constant) plus
all pooled keys, 768 B·N (25 MB at 32K, 101 MB at 128K). Dense attention at
32K would read 805 MB per step.

### 4.5 In Espetero

Per QSA layer and pass (`kernels/45_qsa.metal`): `qsa_prep` computes the
indexer queries and raw keys (rounded to bf16 as in the reference), `qsa_pool`
pools every block a pass row completes, `qsa_score_cached` / `qsa_score_batch`
score the blocks already in the cache (shared by a tree's rows) and the blocks
ending in pass rows (per row: drafts differ), `qsa_select` takes the top 512
per row by a radix select on the fp32 bits (ties at the threshold go to the
lowest blocks, a deterministic rule), `qsa_tiles` lists per 8-row tile the
cached blocks any of its rows reads, `flash_attention_qsa` attends over them,
and `qsa_save` stores the raw keys, pooled keys and positions in an index pool
that shares the KV pool's block ids. The index is kept from the first token,
so a context can cross 2051 tokens later; rows still under the budget attend
densely (exact, §4.3). `ESPETERO_QSA=0` forces dense attention everywhere, an
approximation past 2051 tokens kept for comparisons.

## 5. N-gram embedding + PLE (layer index 1)

### 5.1 Table constants (computed with [T]:1034-1113; the checkpoint also stores them as I64 buffers)

* 16 heads: h = 0…7 hash bigrams, h = 8…15 trigrams. Head `h` has a prime
  vocabulary `P_h` = the (g+1)-th prime ≥ 20,000,000 with `g = ple_layer_index·16
  + h` (`ple_layer_index` = 0 here, [T]:1099-1104):

| h | P_h | OFF_h | h | P_h | OFF_h |
|--:|--:|--:|--:|--:|--:|
| 0 | 20,000,003 | 0 | 8 | 20,000,081 | 160,000,374 |
| 1 | 20,000,023 | 20,000,003 | 9 | 20,000,093 | 180,000,455 |
| 2 | 20,000,033 | 40,000,026 | 10 | 20,000,107 | 200,000,548 |
| 3 | 20,000,047 | 60,000,059 | 11 | 20,000,147 | 220,000,655 |
| 4 | 20,000,059 | 80,000,106 | 12 | 20,000,153 | 240,000,802 |
| 5 | 20,000,063 | 100,000,165 | 13 | 20,000,159 | 260,000,955 |
| 6 | 20,000,069 | 120,000,228 | 14 | 20,000,161 | 280,001,114 |
| 7 | 20,000,077 | 140,000,297 | 15 | 20,000,171 | 300,001,275 |

* Σ P_h = 320,001,446 rows, padded to a multiple of 128: **320,001,536 rows**
  = 128 shards × 2,500,012 rows ([T]:1111-1113; shard shapes confirm).
* Multipliers (`_build_layer_multipliers(248320, 3, 0, 1234)`, splitmix64,
  odd, < `(2⁶³−1)/248320 = 37,143,089,710,272`; [T]:1041-1057):
  `M0 = 23,703,573,157,769`, `M1 = 20,109,073,645,365`,
  `M2 = 8,052,911,324,071`. `token·M < 0.64·2⁶³`: no int64 overflow, XOR
  of non-negative values stays non-negative. Stored as
  `ple.ple_embedding.layer_multipliers` I64[3], `ngram_heads_vocab_sizes`
  I64[16], `ngram_heads_offsets` I64[16] — load and assert equal.

### 5.2 Hashing (exact)

```
EOS = 248044 (text_config.eos_token_id; <|endoftext|>, not <|im_end|>)   [T]:1094, [M]:1322-1323
history before the first token: [EOS, EOS]                                [T]:1138, [M]:1400-1402
c0 = id_t
c1 = id_{t−1}
c2 = EOS if id_{t−1} == EOS else id_{t−2}          # _shift_right_ignore_eos, [T]:1115-1129 / [M]:1373-1392
X2 = (c0·M0) XOR (c1·M1)                           # int64
X3 = X2 XOR (c2·M2)
row_h = OFF_h + (X2 mod P_h)   h = 0…7
row_h = OFF_h + (X3 mod P_h)   h = 8…15             # [T]:1159-1174, [M]:1412-1427
```

`_shift_right_ignore_eos(ids, s)` returns `ids[t−s]` unless an EOS occurs in
positions `[t−s, t−1]`, where it returns EOS; for s ≤ 2 that is the rule above
(the EOS token itself belongs to the segment it ends). Images: the ids at
image rows are the placeholder id (248056) in both references ([T]:2335-2337
passes the original `input_ids`; [M] passes `inputs` to every layer), so runs
of image tokens hash to the same rows. Padding rows get EOS ([T]:1471-1476).

### 5.3 Rows: count, width, format

* **16 rows per token** (one PLE layer), each **160 wide** (`2560 / 16`,
  [T]:1091), concatenated head-major into `e_t ∈ ℝ²⁵⁶⁰`
  (`flatten(-2)`, [T]:1180).
* oQ4e storage: `…ngram_embedding.shards.<k>.{weight U32[2500012, 20],
  scales BF16[2500012, 5], biases BF16[2500012, 5]}` — **4-bit affine, group
  32**, 100 B per row (80 B codes + 10 B scales + 10 B biases). Row `r`
  lives in shard `r / 2,500,012` at local row `r mod 2,500,012` (equal
  shards). Group 32 because 160 is not a multiple of 64 ([M]:2237-2241).
* **`…ngram_embedding.weight_scale` BF16[1] must multiply every row.** The
  rows hold un-scaled FP8 values: 200 random rows of shard 21 dequantize to
  values in −192…176 with RMS 38 (and shard 0's biases range −240…208) —
  e4m3 magnitudes, not embedding magnitudes. The official FP8 layout is FP8
  shards × one per-tensor `weight_scale` ([MF]:106-122:
  `from_fp8(shard) * scale`). So `E[row] = dequant_q4g32(row) ×
  weight_scale`, with the scalar read at load.

### 5.4 The PLE layer ([T]:1183-1255, [M]:1432-1509)

```
Kp = GroupedRMSNorm₁₊w( W_key e ; norm_key [10240] )     key_proj   [10240, 2560]  → [4, 2560]
Vp = W_val e                                              value_proj [2560, 2560]   → [2560]
Q  = GroupedRMSNorm₁₊w( H ; norm_query [10240] )          H = residual entering layer 1  → [4, 2560]
g_s = (Kp_s · Q_s) / √2560 ;   g_s ← sign(g_s) · √max(|g_s|, 1e-6)       (sign(0) = 0 → σ = 0.5)
G_s = σ(g_s) · Vp                                          → G [10240]
Gn  = GroupedRMSNorm₁₊w( G ; norm_conv [10240] )
C_t[ch] = silu( Σ_{j=0..3} w[ch, j] · Gn_{t−9+3j}[ch] )   conv1d.weight [10240, 4, 1], depthwise,
                                                           dilation 3 (= ngram_size), taps t−9, t−6, t−3, t,
                                                           zero history at sequence start
PLE_t = G_t + C_t ;    H ← H + PLE                         added before layer 1's attention HC block
```

The key path is scale-invariant (RMSNorm after a linear map) up to eps; the
value path and `G` are not, so `weight_scale` matters.

### 5.5 Decode state

The last **2 token ids** (n-gram context, [T] conv state 2 / [M] cache[3])
and the last **9 normalized conv inputs** `Gn` (10240 each: 184 KB bf16,
369 KB fp32; [T] conv state 1 / [M] cache[2], window update [M]:1471-1482).
The GDN layer 1 cache therefore has 4 entries in [M] (`ArraysCache(size=4)`,
[M]:2225-2232) and 3 conv states in [T] (`number_of_conv_states`,
[TC]:172). No other state.

### 5.6 Serving the rows

mlx-vlm keeps the whole ~30 GiB table resident by default (128 embedding
shards), or reads rows from memory-mapped files with an optional LRU. Espetero
never loads the table into GPU memory: the rows of a pass are hashed on the
host (the ids are known as soon as a token is sampled or drafted; 16 int64
mods per token), read with `pread` from the checkpoint's own safetensors
(`src/model/impl/qwen4_io.h`), and dequantized and multiplied by
`weight_scale` on the GPU. A token needs 1.6 KB of row payload.

## 6. MoE (all 48 layers)

```
r = W_router x               mlp.gate.weight BF16 [512, 2560] (not quantized in oQ4e)
p = softmax(r) in fp32;  (I, P) = top-10(p);  w = P / Σ P          norm_topk_prob = true
y_e  = W_down,e ( silu(W_gate,e x) ⊙ W_up,e x )                  experts: 640 wide, 4-bit g64
y_sh = W_down,sh ( silu(W_gate,sh x) ⊙ W_up,sh x )               shared expert: 640 wide, 8-bit g128
y = Σ_i w_i · y_{I_i}  +  σ(w_sg · x) · y_sh                      shared_expert_gate [1, 2560], 8-bit g64
```

[T]:960-1000 (router 971-977, experts 920-957, shared gate 996); mlx-vlm
[M35M]:15-49 (`softmax(precise=True)`, `argpartition`, renormalize,
`SwitchGLU`), Qwen4 only changes the shared-gate projection's reduction
order ([M]:1200-1203). No routing bias, no routed scaling factor, no group
limit, no jitter. The block is identical to Qwen3.5-MoE except geometry: E =
512 (was 256), k = 10 (was 8), Ie = 640 (was 512), D = 2560. The MLX file
splits HF's fused `experts.gate_up_proj` [512, 1280, 2560] into
`switch_mlp.{gate,up}_proj` at the midpoint ([MQ]:39-59). 480 expert uses per
token, each 2.76 MB at 4-bit g64; all experts 67.95 GB.

## 7. MTP head

Only mlx-vlm implements it ([MD]; transformers ignores `mtp.*`). Tensors
(oQ4e prefix `mtp.`, not `language_model.mtp.`; the bf16 drafter repo
`Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16` has the same 31 tensors without
the prefix):

```
inputs: token id_{i+1} and the target's pre-mixer residual H_i [10240] (last layer, mtp_use_hidden_state_from_layer null)
ê  = RMSNorm₁₊w(Embed(id_{i+1}); pre_fc_norm_embedding [2560])         (target's embed_tokens)
Ĥ  = RMSNorm₁₊w(H_i; pre_fc_norm_hidden [10240])                      ONE global norm over all 10240 ([MD]:42-46)
U_s = fc_embedding · ê + fc_hidden · Ĥ_s     (s = 0…3)                 fc_* [2560, 2560], 4-bit g64 ([MD]:95-108)
H^m = [U_0, U_1, U_2, U_3]                                               new 4-stream state
H^m ← decoder layer (QSA attention with its own indexer + MoE, attn/mlp HC blocks, no PLE)   mtp.layers.0.*
logits = LMHead(Mixer_mtp(H^m))                                          mtp.hyper_connection_mixer; shared lm_head
next draft step: (argmax/sampled token, H^m pre-mixer) → same computation  ([MD]:110-130)
```

* Positions: the MTP layer has its own QSA cache (K/V + indexer) and its own
  positions, starting at 0 at prefill: the row pairing `id_{i+1}` with `H_i`
  sits at position i ([MB]:89-96, 122-146). This matches Espetero's Qwen3.5 MTP
  convention (`forest.h`:1685).
* Prefill: the MTP runs over the whole prompt (tokens shifted by one plus the
  first sampled token, paired with the target's `H_0…H_{L−1}`); its last row's
  logits give the first draft (the "seed") and its pre-mixer output the seed
  hidden ([MB]:122-146).
* Drafting ([MB]:264-305): the seed token plus autoregressive steps, each
  step feeding the MTP's own pre-mixer output `H^m` back as the hidden.
  `default_runtime_block_size = 4` (≤ 3 drafts), `prefer_requested_block_size
  = False`: the configured block (config `block_size` 4 in the bf16 drafter,
  `mtp_num_hidden_layers + 1` = 2 by default) is widened to the requested one
  only when ≥ 65 % of the last 32 rounds accepted the full configured prefix
  ([MS]:511-557).
* Verification ([MS]:587-742): the target runs `[bonus, drafts…]` in one
  pass (batch-invariant forward, [M]:2183-2217) returning pre-mixer states;
  the walk samples the target at each position and accepts while it equals the
  draft ([MS]:243-286; exact for any sampler, since it samples the target);
  GDN/conv states and caches roll back to the accepted length; the MTP cache is
  trimmed to the accepted drafts, then extended with the accepted drafts not
  yet appended and the bonus token, each paired with the verify pass's
  pre-mixer state at its position ([MB]:148-185).
* Weights (oQ4e): attention 5–6-bit, HC 5-bit, shared expert 8-bit g128,
  router bf16 (the splitter never quantizes `layers.0.mlp.gate`), fc 4-bit g64.
* `mtp.hybrid: true` is read by neither reference.

## 8. Vision tower and M-RoPE positions

**Same tower as Qwen3.5/3.6** (Qwen3-VL without deepstack): [T]'s vision
classes (`VisionRotaryEmbedding`, `MLP`, `PatchEmbed`, `PatchMerger`,
`Attention`, `Block`, `Model`) diff clean against `qwen3_5`, and mlx-vlm uses
`qwen3_vl.VisionModel` for both ([M]-side `vision.py`, `qwen3_5/vision.py`)
with `Qwen3VLProcessor` (`qwen4_exp/__init__.py`:8). Geometry = the 27B's:
depth 27, D 1152, 16 heads (head 72), MLP 4304 gelu-tanh, Conv3d patch
(2 × 16 × 16 × 3) + bias, 48 × 48 learned positions bilinearly resampled, 2-D
axial RoPE, merger LayerNorm(1152) → fc1 4608 → GELU → fc2 **2560** (the text
hidden). oQ4e keeps the tower in bf16 under `vision_tower.*` (MLX Conv3d
layout `[1152, 2, 16, 16, 3]`). Espetero's `VisionConfig::validate` accepts it
unchanged (head 72, `out == D`).

M-RoPE position ids: `get_vision_position_ids`, `get_rope_index`,
`compute_3d_position_ids` are byte-identical to Qwen3.5's ([T]:2042-2307):
text `(p, p, p)`; an image of merged grid `h × w` starting at p gets `t = p`,
`h = p + row`, `w = p + col`, and the next text position is `p + max(h, w)`;
videos are split into single frames (timestamps between frames). Two Qwen4
consumers of these positions are new: the indexer query and block keys
(3-D position of the block's first token), and the PLE sees the placeholder
ids at image rows (§5.2).

## 9. Per-token decode memory traffic and per-sequence state (oQ4e-mtp)

Weights read per decoded token, one sequence, from the per-module widths in
`config.json` (Appendix B), MLX file bytes (bits/8 + 4 B per group); the
Espetero column splits group 128 into 64 (exact, +0.25 bit/weight) and keeps
small vectors in fp32:

| Component | MB / token | Share | Espetero MB | Params (M) |
|:--|--:|--:|--:|--:|
| GDN projections (36 layers: qkv, z, a, b, out; 4–6-bit, z/out often 5-bit g128) | 1,357.2 | 29.4 % | 1,387.9 | 2,085 |
| Routed experts (10 of 512 × 48 layers, 4-bit g64) | 1,327.1 | 28.7 % | 1,327.1 | 2,359 |
| LM head (8-bit g64, 248,320 × 2560) | 675.4 | 14.6 % | 675.4 | 636 |
| QSA q/k/v/o (12 layers, 4–8-bit) | 441.0 | 9.6 % | 441.1 | 598 |
| Hyper-connections (96 blocks, 4–8-bit g64) | 404.8 | 8.8 % | 404.8 | 633 |
| Shared experts (8-bit g128) | 243.3 | 5.3 % | 250.7 | 236 |
| Routers (bf16) + shared gates | 126.0 | 2.7 % | 126.0 | 63 |
| PLE key/value projections (4-bit g64) | 18.4 | 0.4 % | 18.4 | 33 |
| Indexer `index_qk_proj` (12 layers) | 14.8 | 0.3 % | 14.8 | 20 |
| Final mixer | 3.7 | 0.1 % | 3.7 | 7 |
| Norm, conv, A_log, dt_bias vectors | 5.1 | 0.1 % | 9.9 | – |
| Embedding row (8-bit) | 0.003 | – | 0.003 | – |
| n-gram rows (16 × 100 B, from SSD) | 0.002 | – | 0.002 | – |
| **Total** | **4,617** | | **4,660** | 6,670 |

Around the weights, per step: GDN recurrent state read + write 2 × 113.2 MB;
KV and indexer reads per §4.4 (50 MB at 2 k context); GDN conv, PLE conv
states < 5 MB. At the 354 GB/s an M1 Max sustains (moe.md) the floor is
≈ 14 ms/token (≈ 70 tokens/s) **if every expert were resident** — on a 64 GB
Mac they are not (67.95 GB > 64 GB), so the real number depends on the expert
cache and the SSD. MTP draft step: 771 MB, of which 675 MB is the 8-bit LM head (Espetero's
4-bit draft head copy: 358 MB; coarse 2-bit head + shortlist: ~200 MB); the
rest is attention + indexer 39.8, experts 27.7, HC 12.8, shared + router 7.7,
fc 7.4 MB.

Per-sequence state (Espetero, fp32):

| State | Size | Note |
|:--|--:|:--|
| GDN recurrent, 36 × 48 × 128 × 128 | 113.2 MB | read + written every step |
| GDN conv, 36 × 3 × 10240 | 4.4 MB | 2.2 MB in bf16 |
| PLE conv, 9 × 10240 | 369 KB | 184 KB bf16 |
| n-gram context | 2 ids | |
| MTP hidden rows (Espetero: mh + 16 lh rows) × 10240 | 696 KB | 4× Qwen3.5's D-wide rows |
| KV + indexer | 27.75 KiB / token (+2.3 KiB MTP) | §4.4; 3.7 GB at 131,072 tokens |

## 10. Reference discrepancies and decisions

| Topic | transformers | mlx-vlm | Decision |
|:--|:--|:--|:--|
| GDN q/k normalization | FLA l2norm, eps 1e-6 after the sum | same for Qwen4 ([M]:798-804), unlike its Qwen3.5 (`rms_norm`, eps·128) | both agree; Espetero uses it for this model, not its Qwen3.5 convention |
| QSA below the budget | runs top-k (selects all) | skips the indexer for decode, causal mask otherwise | equivalent (§4.3) |
| QSA ties | `torch.topk` | `argpartition` | arbitrary in both; Espetero gives ties at the threshold to the lowest blocks |
| Pooled key rounding | fp32 mean → bf16 → norm | same | Espetero rounds to bf16 as the references do |
| Residual dtype | bf16 | bf16 | Espetero fp32 (deliberate, more precise) |
| Gated norm rounding | normalized value rounded to bf16 before × weight | fused `rms_norm(x, w)` in fp32 | negligible |
| PLE `weight_scale` | table assumed bf16 without scale (95 GiB comment, [T]:1327-1330) | FP8 conversion multiplies; 0.7.3 cannot load the oQ4e layout as is | **multiply** (FP8 magnitudes in the stored rows) |
| PLE EOS | text `eos_token_id` 248044 | same | 248044 only (not 248046) |
| `split_ngram_parts` | default 512 | default 128 | layout only; the file has 128 |
| MTP | not implemented | [MD] | trust mlx-vlm; tensors and shapes match. The one unverifiable choice is the *global* `pre_fc_norm_hidden` (the drafter comment says "the released head applies one global RMS normalization"); if acceptance is poor, try the grouped variant — drafts are verified, so output quality cannot suffer |
| Batched prefill | vectorized | per-row passes then merged caches ([M]:1619-1710, batch invariance) | numerics only |

## Appendix A. Tensor inventory (oQ4e names; logical shapes `[out, in]`)

Quantized matrices have `.weight` (U32 codes), `.scales`, `.biases` (BF16).

```
language_model.model.embed_tokens                     [248320, 2560]   8/64   (U32 [248320, 640], s/b [248320, 40])
language_model.lm_head                                [248320, 2560]   8/64
language_model.model.hyper_connection_mixer.hc_norm.weight                      BF16 [10240] (zero-centred)
language_model.model.hyper_connection_mixer.input_mix_weight_down               [320, 10240]  4/64
language_model.model.hyper_connection_mixer.input_mix_weight_up                 [10240, 320]  4/64

language_model.model.layers.<l>.
  {attn,mlp}_hyper_connection.hc_norm.weight          BF16 [10240] (zero-centred)
  {attn,mlp}_hyper_connection.input_mix_weight_down   [320, 10240]    4–8/64
  {attn,mlp}_hyper_connection.input_mix_weight_up     [10240, 320]    4–8/64 (5 groups per row)
  {attn,mlp}_hyper_connection.block_inject_weight     [4, 10240]      4–8/64
  mlp.gate.weight                                     BF16 [512, 2560]          (router, unquantized)
  mlp.shared_expert_gate                              [1, 2560]       8/64
  mlp.shared_expert.{gate_proj,up_proj}               [640, 2560]     8/128
  mlp.shared_expert.down_proj                         [2560, 640]     8/128
  mlp.switch_mlp.{gate_proj,up_proj}                  [512, 640, 2560] 4/64   (U32 [512, 640, 320], s/b [512, 640, 40])
  mlp.switch_mlp.down_proj                            [512, 2560, 640] 4/64   (U32 [512, 2560, 80], s/b [512, 2560, 10])
 GDN layers (l mod 4 ≠ 3):
  linear_attn.in_proj_qkv                             [10240, 2560]   4–6/64
  linear_attn.in_proj_z                               [6144, 2560]    5/128 or 6/64
  linear_attn.in_proj_a, in_proj_b                    [48, 2560]      5/128 or 6/64
  linear_attn.out_proj                                [2560, 6144]    5/128
  linear_attn.conv1d.weight                           BF16 [10240, 4, 1]
  linear_attn.A_log, linear_attn.dt_bias              BF16 [48]
  linear_attn.norm.weight                             BF16 [128]  (plain weight, ≈ 1)
 QSA layers (l mod 4 = 3):
  self_attn.q_proj                                    [12288, 2560]   4–8/64  (per head [q 256 | gate 256])
  self_attn.k_proj, v_proj                            [512, 2560]     4–8/64 (one v_proj 6/128)
  self_attn.o_proj                                    [2560, 6144]    4–8/64
  self_attn.q_norm.weight, k_norm.weight              BF16 [256] (zero-centred)
  self_attn.indexer.index_qk_proj                     [640, 2560]     4–8/64  ([q 4×128 | k 128])
  self_attn.indexer.q_layernorm.weight, k_layernorm.weight   BF16 [128] (zero-centred)
 PLE (l = 1 only):
  ple.key_proj                                        [10240, 2560]   4/64
  ple.value_proj                                      [2560, 2560]    4/64
  ple.norm_key.weight, norm_query.weight, norm_conv.weight   BF16 [10240] (zero-centred)
  ple.conv1d.weight                                   BF16 [10240, 4, 1]
  ple.ple_embedding.layer_multipliers                 I64 [3]
  ple.ple_embedding.ngram_heads_vocab_sizes           I64 [16]
  ple.ple_embedding.ngram_heads_offsets               I64 [16]
  ple.ple_embedding.ngram_embedding.shards.<0..127>   [2500012, 160]  4/32  (U32 [2500012, 20], s/b [2500012, 5])
  ple.ple_embedding.ngram_embedding.weight_scale      BF16 [1]

mtp.pre_fc_norm_embedding.weight                      BF16 [2560] (zero-centred)
mtp.pre_fc_norm_hidden.weight                         BF16 [10240] (zero-centred, global norm)
mtp.fc_embedding, mtp.fc_hidden                       [2560, 2560]    4/64
mtp.hyper_connection_mixer.{hc_norm.weight, input_mix_weight_down, input_mix_weight_up}   (as the trunk's; 4/64)
mtp.layers.0.{attn,mlp}_hyper_connection.*            5/64
mtp.layers.0.self_attn.{q_proj 6/64, k_proj 6/64, v_proj 6/128, o_proj 5/64, q_norm, k_norm,
                        indexer.index_qk_proj 6/64, indexer.q_layernorm, indexer.k_layernorm}
mtp.layers.0.mlp.{gate BF16, shared_expert_gate 8/64, shared_expert.* 8/128, switch_mlp.*}

vision_tower.patch_embed.proj.{weight BF16 [1152, 2, 16, 16, 3], bias [1152]}
vision_tower.pos_embed.weight                         BF16 [2304, 1152]
vision_tower.blocks.<0..26>.{norm1,norm2}.{weight,bias} [1152]
vision_tower.blocks.<i>.attn.qkv.{weight [3456, 1152], bias}; attn.proj.{weight [1152, 1152], bias}
vision_tower.blocks.<i>.mlp.linear_fc1.{weight [4304, 1152], bias}; mlp.linear_fc2.{weight [1152, 4304], bias}
vision_tower.merger.norm.{weight,bias} [1152]; merger.linear_fc1.{weight [4608, 4608], bias}; merger.linear_fc2.{weight [2560, 4608], bias}
```

Zero-centring evidence from the file's bytes: indexer norms mean −0.02 to
−0.04, `q_norm`/`k_norm` means 0.43–0.46 (min −1.0), `hc_norm` means −0.07 to
1.16 (min −2.1); `linear_attn.norm.weight` means 1.00–1.31 (plain weights).

## Appendix B. Per-layer quantization of the oQ4e-mtp file (bits/group)

Everything not listed is 4/64 except: `embed_tokens`, `lm_head` 8/64; every
`mlp.shared_expert.*` 8/128; every `shared_expert_gate` 8/64; routers bf16;
PLE shards 4/32; PLE key/value 4/64; final mixer 4/64; experts 4/64.

| l | type | attn HC down / up / inject | mlp HC down / up / inject | GDN qkv / z / a / b / out, or QSA q / k / v / o / index_qk |
|--:|:--|:--|:--|:--|
| 0 | GDN | 5/64 5/64 5/64 | 5/64 5/64 5/64 | 6/64 6/64 6/64 6/64 5/128 |
| 1 | GDN + PLE | 4/64 ×3 | 4/64 ×3 | 4/64 5/128 5/128 5/128 5/128 |
| 2, 4–6, 8–10, 12–14, 16–18, 20, 22, 24–26, 28, 38, 40–42, 45 | GDN | 4/64 ×3 | 4/64 ×3 | 4/64 5/128 5/128 5/128 5/128 |
| 3, 7, 11, 47 | QSA | 4/64 ×3 | 4/64 ×3 | 4/64 ×5 |
| 15, 27, 35 | QSA | 6/64 ×3 | 6/64 ×3 | 6/64 6/64 8/64 6/64 6/64 |
| 19 | QSA | 4/64 4/64 5/64 | 4/64 4/64 5/64 | 4/64 4/64 4/64 4/64 5/64 |
| 21 | GDN | 4/64 4/64 5/64 | 4/64 4/64 5/64 | 4/64 5/128 6/64 5/128 5/128 |
| 23 | QSA | 5/64 ×3 | 5/64 ×3 | 5/64 ×5 |
| 29 | GDN | 5/64 ×3 | 4/64 4/64 5/64 | 5/64 6/64 6/64 6/64 5/128 |
| 30, 32, 37 | GDN | 5/64 ×3 | 5/64 ×3 | 5/64 6/64 6/64 6/64 5/128 |
| 31, 39 | QSA | 8/64 ×3 | 8/64 ×3 | 8/64 ×5 |
| 33 | GDN | 5/64 ×3 | 5/64 ×3 | 6/64 6/64 6/64 6/64 5/128 |
| 34, 36, 44, 46 | GDN | 6/64 ×3 | 6/64 ×3 | 6/64 6/64 6/64 6/64 5/128 |
| 43 | QSA | 5/64 ×3 | 5/64 ×3 | 6/64 6/64 6/128 5/64 6/64 |
