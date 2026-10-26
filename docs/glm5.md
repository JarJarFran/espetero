# GLM-5.3-Flash (`glm5_next`) — architecture reference and port plan

The model's math as the references define it, the checkpoint's facts, and how
Espetero will run it on a 128 GB Mac (text only first). Written before the
kernels; the loader and kernels cite the sections below. Status: design note
under review; step 2 (tokenizer check, template fixtures, GLM tool calls,
loader skeleton) in progress on branch `arch/glm5`.

GLM-5.3-Flash (zai-org, 2026-09) is a 320B-A18B hybrid: 34 KDA linear-attention
layers (Kimi Delta Attention) and 11 DeepSeek-sparse-attention layers (nope-only
MLA with a k-pool indexer), mHC hyper-connections with Sinkhorn (DeepSeek-V4's),
3 dense MLP layers then 42 MoE layers of 288 experts (8 routed + 1 shared,
sigmoid "noaux_tc" router, routed scaling 2.5, SwiGLU clamped at 10), one
NextN/MTP layer, and a vision tower (not covered here). The HF checkpoint is
FP8 e4m3 with fp32 scales per 128 × 128 block for the MLPs and the DSA
projections, bf16 for everything else: **328.3 GB, of which the routed experts
are 311.7 GB.**

Contents: 0 sources and checkpoint facts · 1 decoder structure · 2 mHC ·
3 KDA layers · 4 DSA layers (nope MLA + k-pool indexer) · 5 dense MLP, MoE,
shared expert · 6 MTP layer · 7 tokenizer, template, tool calls · 8 memory
plan on 128 GB (and disk) · 9 per-token traffic, state, expected speed ·
10 mapping onto Espetero (what is reused, varied, new) · 11 parity plan and
results · 12 reference discrepancies, decisions, open questions · 13 status ·
A tensor inventory.

## 0. Sources, conventions, checkpoint facts

| Tag | Source |
|:--|:--|
| **[T]** | `transformers/models/glm5_next/modeling_glm5_next.py` (5.18.0; installed in `~/.cache/ultra-metal/tf-venv`, identical to `main` but for cache-API renames). Canonical math. |
| **[TC]** | `.../configuration_glm5_next.py` (defaults, `__post_init__`, `validate_architecture`) |
| **[L]** | llama.cpp `src/models/glm5-next.cpp` (PR 27773 merged 2026-09-30; MTP in PR 29928) — the only reference that runs the MTP layer and the absorbed MLA |
| **[LC]** | llama.cpp `gguf-py/gguf/constants.py`, `tensor_mapping.py` (GGUF names of unsloth's files) |
| **[M]** | `mlx_vlm/models/glm5_next/language.py` (0.7.4) and `mlx_vlm/speculative/drafters/glm5_next_mtp/` — a second opinion on the indexer and the MTP |
| **[C]** | `zai-org/GLM-5.3-Flash` `config.json`, `model.safetensors.index.json`, the 62 shard headers, `tokenizer.json`, `chat_template.jinja` (fetched 2026-10-21) |

**Trust rule.** [T] defines the math of layers 0–44 and the head. [T] ignores
`layers.45` (the MTP layer, `_keys_to_ignore_on_load_unexpected`); for it [L]
and [M] agree with each other and are followed. Where [T] and [L]/[M] differ,
§12 gives the decision.

**Notation.** `D = 4096`, `S = 4` streams (`hc_mult`), flat width `W = 16384`.
Linear layers are `y = A x`, `A` stored `[out, in]`. `σ` = sigmoid,
`silu(x) = x σ(x)`. `RMSNorm_w(x) = w ⊙ x · rsqrt(mean(x²) + 1e-5)` (fp32,
plain weight — **no `1 + w`**, unlike Qwen4-Exp). Token positions `t` are
0-based, `n = t + 1` tokens visible to `t`.

### config.json (text_config)

| Field | Value | Field | Value |
|:--|:--|:--|:--|
| `hidden_size` | 4096 | `num_hidden_layers` | 45 (+ 1 MTP: `num_nextn_predict_layers` 1) |
| `vocab_size` | 154,880 | `tie_word_embeddings` | false |
| `rms_norm_eps` | 1e-5 | `max_position_embeddings` | 1,048,576 |
| `layer_types` | `linear_attention` ×34, `deepseek_sparse_attention` at 3, 7, 11, …, 43 (every `l mod 4 = 3`) | `indexer_types` | `full` ×45 (no shared-indexer layers) |
| `linear_attn_config` | 64 heads × 128, `short_conv_kernel_size` 4, `gate_lower_bound` −5.0 | `mlp_layer_types` | dense ×3 (`first_k_dense_replace` 3), sparse ×42 |
| `intermediate_size` (dense) | 12,288 | `moe_intermediate_size` | 2048 (`n_shared_experts` 1 → shared width 2048) |
| `n_routed_experts` / `num_experts_per_tok` | 288 / 8 | `topk_method` / `scoring_func` | `noaux_tc` / `sigmoid`, `n_group` 1, `topk_group` 1 |
| `routed_scaling_factor` | 2.5 | `norm_topk_prob` | true |
| `moe_router_dtype` | float32 | `swiglu_limit` | 10.0 (dense, shared and routed) |
| `q_lora_rank` / `kv_lora_rank` | 1536 / 512 | `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim` | 256 / **0** / 256; `mla_use_nope` true; 64 heads (= kv heads) |
| `index_n_heads` / `index_head_dim` / `index_topk` | 32 / 128 / 2048 | `index_kpool` / `index_kpool_compress` / `..._always_select_tail` | 4 / true / true |
| `indexer_rope_interleave` | true (inert: no RoPE anywhere) | `index_share_for_mtp_iteration` | true (read by no reference) |
| `mhc` / `hc_mult` / `hc_sinkhorn_iters` / `hc_eps` | true / 4 / 20 / 1e-6 | `head_dim` | 0 (= `qk_rope_head_dim`) |
| `eos_token_id` | [154820 `<\|endoftext\|>`, 154827 `<\|user\|>`, 154829 `<\|observation\|>`] | `pad_token_id` | 154820 |
| top level | `image_token_id` 154854, `video_token_id` 154855, image/video start/end 154830–154833; `quantization_config`: fp8 e4m3, `weight_block_size` [128, 128], `activation_scheme` dynamic, 1509 `modules_to_not_convert` | generation_config | `do_sample` true, `temperature` 1.0, `top_p` 0.95 |

### Checkpoint facts ([C], all 62 headers read)

* 76,108 tensors, `total_size` 328,326,771,576 B: routed experts 311.73 GB
  (12,384 = 43 × 288 experts of 25.17 MB), the rest of layers 0–44 and the
  head 15.23 GB, the MTP layer without its experts 0.24 GB, vision 1.13 GB.
* FP8 tensors (`F8_E4M3`, with `.weight_scale_inv` F32 `[N/128, K/128]`):
  dense `mlp.{gate,up,down}_proj` of layers 0–2, `shared_experts.*` and
  `experts.E.*` of layers 3–45, DSA `q_a_proj`, `q_b_proj`,
  `kv_a_proj_with_mqa`, `o_proj`. **bf16:** everything in the KDA layers
  (q/k/v/o projections 67 MB each), `kv_b_proj`, the indexer, routers,
  embeddings, head, norms. **F32:** `A_log`, `dt_bias`, `hc_*_base/scale`,
  `e_score_correction_bias`, all scales.
* Shards group tensors **by dtype** (F32, then BF16, then F8), each group
  sorted by name. So each expert's three code tensors are contiguous in the
  order `down_proj | gate_proj | up_proj` (3 × 8,388,608 B) and its three scale
  tensors are contiguous in the same order (3 × 2048 B) elsewhere in the shard
  — the two-read layout DeepSeek-V4.1's expert cache already assumes
  (`XLoc{file, w, s}`). One exception: expert 197 of layer 45 (the MTP layer)
  has its tensors in shards 1 and 2 (a two-piece read).
* Layers are sharded in string order (1, 10, 11, …), a layer spans 2–3 shards.
* Unquantized `modules_to_not_convert` (1509 names) match the headers: attention
  of the KDA layers, hyper-connections, `lm_head`, embeddings, routers, the
  indexer, the vision tower.

## 1. Overall decoder structure ([T]:1426-1511, 1279-1350; [L]:289-410)

```
e_t  = Embed(id_t)                                   # bf16 [154880, 4096], no scaling
H⁰_t = [e_t, e_t, e_t, e_t]                          # [4, 4096] residual streams (fp32 in Espetero; bf16 in [T])
for l in 0 … 44:
    (pre, post, C) = HC(H; hc_attn_*)                # §2, all three from THIS block's input
    x = RMSNorm(Σ_s pre_s H_s; input_layernorm)
    y = KDA_l(x)     if l mod 4 ≠ 3                  # §3
        DSA_l(x)     if l mod 4 = 3                  # §4 (layers 3, 7, …, 43)
    H_k ← post_k · y + Σ_j C[j][k] H_j
    (pre, post, C) = HC(H; hc_ffn_*)
    x = RMSNorm(Σ_s pre_s H_s; post_attention_layernorm)
    y = MLP_l(x)     if l < 3   (dense, 12288 wide)  # §5
        MoE_l(x)     else      (288 experts + shared)
    H_k ← post_k · y + Σ_j C[j][k] H_j
h      = RMSNorm(mean_s H_s; norm.weight)            # hc_head = unweighted mean ([T]:332-337), then the final norm
logits = lm_head · h
MTP input for position t: h_t (the final-normed hidden)   # §6
```

No RoPE in any layer (NoPE DSA, recurrent KDA); positions only matter for the
causal structure and the k-pool boundaries. No `hidden_states` scaling, no
logit soft-capping.

## 2. mHC hyper-connections ([T]:218-330; [L]:60-145)

Per block (2 × 45 blocks), from the flat residual `h ∈ ℝ^16384`:

```
n = h · rsqrt(mean(h²) + 1e-5)                  unweighted RMSNorm over all 16384 values, fp32  (eps = rms_norm_eps)
m = fn · n                                       fn [24, 16384] (bf16 in the file, fp32 math) → 24 = (2 + 4) · 4 mixes
pre_s  = σ(m[s]   · scale[0] + base[s])   + 1e-6        s = 0..3        (hc_eps)
post_s = 2 σ(m[4+s] · scale[1] + base[4+s])
C[j][k] = softmax_k(m[8+4j+k] · scale[2] + base[8+4j+k]) + 1e-6          j = source stream, k = destination
C ← C / (Σ_j C[j][k] + 1e-6)                                           columns first,
repeat 19 ×:  C ← C / (Σ_k C[j][k] + 1e-6);  C ← C / (Σ_j C[j][k] + 1e-6)   then rows / columns  (20 Sinkhorn iterations)
x    = Σ_s pre_s H_s                              the block's input (before its RMSNorm)
H'_k = post_k · y + Σ_j C[j][k] H_j               [T]:1326-1328: post ⊗ y + Cᵀ H
```

Identical to DeepSeek-V4.1's `mixes` (docs/deepseek_v41.md §2, kernels
`ds4_hc_mix` / `ds4_hc_part` + `ds4_hc_fin`, `ds4_hc_pre`, `ds4_hc_post`),
with three differences:

| | DeepSeek-V4.1 | GLM-5.3-Flash |
|:--|:--|:--|
| flat-norm eps | 1e-20 | **1e-5** (`Ds4HcP.eps` already a parameter) |
| the `pre` of a block | the previous block's mix (single-pass mHC: `prev`) | **the same block's mix** ([T]:1303 returns `collapsed` from the same call) |
| head | `Σ pre_j h_j` with the last FFN's pre | **mean of the 4 streams** ([T]:335, [L]:392) |
| `fn` dtype | fp32 | bf16 (convert to fp32 at load: 1.57 MB per block) |
| dtype of `post`, `C`, `x` | bf16 streams | [T] casts `post`/`C`/`x` to bf16; Espetero keeps fp32 (as for DSv4) |

## 3. KDA layers (34 layers; [T]:339-393, 451-770; [L]:412-490)

Kimi Delta Attention: Gated DeltaNet with a **per-channel** (per key
dimension) forget gate, separate q/k/v convolutions and low-rank gates.

```
q = W_q x,  k = W_k x,  v = W_v x                            bf16 [8192, 4096] each → 64 heads × 128
[q|k|v]_t ← silu( Σ_{j=0..3} w_c[j] · [q|k|v]_{t−3+j} )      depthwise conv, kernel 4, no bias, zero history;
                                                             q_conv1d / k_conv1d / v_conv1d [8192, 1, 4] (HF tap order: index 3 = current token)
q̂_h = q_h / sqrt(Σ q_h² + 1e-6) / √128,   k̂_h = k_h / sqrt(Σ k_h² + 1e-6)      FLA l2norm (eps after the sum), query scale folded
g_t[h][c] = −5 · σ( exp(A_log[h]) · ( (W_fb W_fa x)[h·128 + c] + dt_bias[h·128 + c] ) )    ∈ (−5, 0)     ([T]:353-371, gate_lower_bound −5)
β_t[h]    = σ( (W_b x)[h] )                                                                 ([T]:720)
S_h ∈ ℝ^{128_k × 128_v}, fp32:   S ← diag(e^{g_h}) S;   u = S_hᵀ k̂_h;   S ← S + k̂_h ⊗ β_h (v_h − u);   o_h = S_hᵀ q̂_h    ([T]:498-511)
z = W_gb W_ga x                                              [8192] output gate (g_a_proj [128, 4096], g_b_proj [8192, 128])
out_h = RMSNorm(o_h; o_norm [128], eps 1e-5) ⊙ σ(z_h)        [T]:374-393 (fp32, weight shared by the 64 heads)
y = W_o · concat_h out_h                                     bf16 [4096, 8192]
```

`W_fa` is `f_a_proj [128, 4096]`, `W_fb` is `f_b_proj [8192, 128]`, `W_b` is
`b_proj [64, 4096]`; `A_log` F32 [64], `dt_bias` F32 [8192]. The gates `g`,
`β`, `z` are computed from the layer input `x`, not from the conv output. The
chunked form [T]:517-617 (chunk 64, `g` cumulative within the chunk, the decay
inside every dot product) is the same recurrence; [L] uses 16-token chunks for
KDA.

Against Espetero's Gated DeltaNet (Qwen3.5 / Qwen4-Exp, `kernels/30_deltanet.metal`):

| Item | Qwen GDN (Espetero) | KDA | Consequence |
|:--|:--|:--|:--|
| heads | Hk 16 / Hv 32–48 (GQA), Dk = Dv = 128 | **64 / 64**, 128 / 128 | `Hk = Hv`, `tiled` layout trivially |
| projections | fused `in_proj_qkv [10240, 2560]` + z, a, b | three bf16 matrices + 5 small ones | concatenate q/k/v (and `f_a`, `g_a`, `b`) at load: two GEMVs, `[24576, 4096]` and `[320, 4096]`, then `f_b`, `g_b` |
| conv | one `conv1d` over [q\|k\|v] | three tensors, same op | concatenate the weights: `conv_prep` as is (C = 24576, Hk = Hv = 64) |
| q/k norm | `x·rsqrt(Σx²/128 + eps)`, Qwen3.5 eps 1e-6 / Qwen4 1e-6/128 | FLA l2norm eps 1e-6 after the sum | `conv_prep` with eps = 1e-6/128 (the Qwen4-Exp setting) |
| decay | scalar per head: `exp(−exp(A_log) softplus(a + dt_bias))` | **vector per head**: `−5 σ(exp(A_log) (f + dt_bias))`, 128 values | new tiny kernel for `g`/`β`; the delta-rule kernels take a per-channel decay |
| output gate | `silu(z)` (Qwen3.5) / `σ(z)` (Qwen4-Exp) | `σ(z)`, z from a low-rank pair | `gated_rmsnorm_sig` / the fused decode's gate with σ; norm eps **1e-5** |
| state dtype | fp32 | fp32 | same |

Per-channel decay in the chunked kernel (`deltanet_chunk`, 8 tokens per step):
with `G_t[c] = Σ_{i≤t} g_i[c]` cumulative in the chunk, every `e^{G_r − G_c}`
of the scalar derivation becomes a per-channel factor inside the dot products:
`L[r][c] = β_r Σ_k k_r[k] k_c[k] e^{G_r[k] − G_c[k]}`. Factored as
`K̃ = K ⊙ e^{G}` and `K̂ = K ⊙ e^{−G}` (per row, per channel) the chunk's MMAs
keep their shape: `L = β ⊙ strictly-lower(K̃ K̂ᵀ)`, `QK = (Q ⊙ e^{G}) K̂ᵀ`,
`X = β (V − K̃ S)`, `O = (Q ⊙ e^{G}) S + QK · D`, and at the chunk end
`S' = diag(e^{G_C}) (S + K̂ᵀ D)` (a per-k-row scale instead of a scalar).
`|G| ≤ 8 · 5 = 40` inside an 8-token chunk, so `e^{±G}` stays within fp32
(e^{±87}); keys are unit vectors. The decode kernel's `st[r] *= g` becomes
`st[r] *= e^{g}[lane·4 .. +3]` (the state is `[Dv][Dk]`, k contiguous).

Per sequence: recurrent state 34 × 64 × 128 × 128 × 4 B = **142.6 MB**, conv
state 34 × 3 × 24576 × 4 B = 10.0 MB.

## 4. DSA layers (11 layers: 3, 7, …, 43; and the MTP layer)

### 4.1 Nope-only MLA ([T]:1101-1277; [L]:647-706)

```
qr  = RMSNorm(W_qa x; q_a_layernorm [1536])              fp8 q_a_proj [1536, 4096]
q_h = (W_qb qr)_h                                         fp8 q_b_proj [16384, 1536] → 64 heads × 256 (all "nope")
c   = RMSNorm(W_kva x; kv_a_layernorm [512])              fp8 kv_a_proj_with_mqa [512, 4096]; the only cached state (1 KB/token in fp16)
k_h = W_kb,h c,   v_h = W_vb,h c                          bf16 kv_b_proj [32768, 512]: rows h·512 .. +255 = W_kb,h [256, 512], rows h·512+256 .. +511 = W_vb,h
o_h = Σ_{j ∈ Sel(t)} softmax_j( q_h · k_{h,j} / 16 ) v_{h,j}      scale 256^−½; no RoPE, no sink, no gate
y   = W_o concat_h o_h                                    fp8 o_proj [4096, 16384]
```

**Absorbed form** (exact algebra, what [L] runs and Espetero will run; the
cache holds the 512-wide latent only): `q̃_h = W_kb,hᵀ q_h ∈ ℝ^512`,
`p_j = softmax_j(q̃_h · c_j / 16)`, `õ_h = Σ_j p_j c_j`, `o_h = W_vb,h õ_h`.
This is DeepSeek-V4.1's attention shape (MQA over one 512-wide latent per
token used as K and V; `ds4_attn`), with two per-head bf16 matvecs around it
(256 → 512 before, 512 → 256 after; 33.5 MB of `kv_b_proj` per layer per
token) and without window, RoPE or sink.

### 4.2 The k-pool indexer ([T]:773-1061; [M]:298-540; [L]:494-645)

```
q̃  = W_iqb qr                       bf16 indexer.wq_b [4096, 1536] → 32 heads × 128   (shares qr with the attention)
k̃_t = LayerNorm_{w,b}(W_ik x_t; eps 1e-6)       bf16 wk [128, 4096]; a true LayerNorm (mean subtracted, bias) — k_norm.{weight,bias} [128]
γ_t = W_g x_t                        bf16 index_kpool_compress_gate [128, 4096] → gate scores [128]
w_t = (W_wp x_t) · 32^−½             bf16 weights_proj [32, 4096] → per-head weights [32]
pools  P_b = {4b, 4b+1, 4b+2, 4b+3}  from position 0 (no padding in Espetero's sequences)
π_i[c] = softmax_i( γ_{4b+i}[c] + ape[i][c] )     i = 0..3, per channel c; ape = index_kpool_compress_ape [4, 128]
K_b[c] = Σ_i π_i[c] · k̃_{4b+i}[c]                 the pooled key, computed once when the pool completes
score_b(t) = Σ_h w_t[h] · relu( q̃_{t,h} · K_b / √128 )     over complete pools with 4b + 3 ≤ t        ([T]:853-857)
Sel(t) = ⋃_{b ∈ top-512(score)} P_b  ∪  tail(t),   tail(t) = {4⌊(t+1)/4⌋, …, t}  (0–3 tokens, always selected)
```

* `index_topk / index_kpool = 512` pools of 4 → ≤ 2048 + 3 keys per query.
* **Dense equivalence:** with `nb = ⌊(t+1)/4⌋ ≤ 512` complete pools, i.e.
  `n ≤ 2051`, every pool is selected and the tail completes `{0..t}`:
  **plain causal attention is exact for every query with ≤ 2051 visible
  tokens** (prompt + generation; [T] `min(select_k, pools)` → all, [M]
  short-circuits). Same threshold as Qwen4-Exp's QSA (512 blocks of 4).
* Selection per query row; ties arbitrary in both references (score 0 when
  all heads' products are ≤ 0): pick a deterministic rule (lowest pools) and
  exclude such rows from exact selection parity.
* Pool `b` is a candidate when its last token is visible (`4b+3 ≤ t`):
  `ds4_index_score`'s `nvis = (pos + 1) / ratio` with ratio 4 is exactly
  that count.
* The scores' constants (`128^−½ · 32^−½`) do not change the ranking;
  DeepSeek-V4.1's code folds the same two constants into the weights.

### 4.3 Cached per token per DSA layer (fp16)

| Item | Bytes | Note |
|:--|--:|:--|
| latent `c` (512) | 1024 | K and V |
| indexer key `k̃` (128) + gate scores `γ` (128) | 512 | kept until the pool completes (the pass may end mid-pool); [L] caches `key | gate | pooled` rows |
| pooled key `K_b` (128 per 4 tokens) | 64 | read by every later query's indexer |
| **total** | **1600** | 11 layers: 17.6 KB/token; + 1.6 KB for the MTP layer. 32 K tokens: 0.6 GB per sequence |

Per decode step at context N: ≤ 2051 latents (2 MB) + all pooled keys
(64 B · N: 2 MB at 32 K, 8 MB at 128 K); dense regime N KB.

## 5. Dense MLP, MoE, shared expert ([T]:85-207)

```
MLP(x)  = W_d ( silu(min(W_g x, 10)) ⊙ clamp(W_u x, −10, 10) )        swiglu_limit 10 in dense, shared and routed MLPs ([T]:97-103, 136-141)
layers 0–2:  MLP with 12288-wide fp8 gate/up/down
layers 3–44: l = W_r x   (fp32 math; router bf16 [288, 4096])            ([T]:157-159)
             s = σ(l);   I = top-8 of (s + bias)     bias = e_score_correction_bias F32 [288]  (n_group 1: the group step is a no-op)
             w_i = s_i / (Σ_{i∈I} s_i + 1e-20) · 2.5                      norm_topk_prob, routed_scaling_factor  ([T]:177-183)
             y = Σ_{i∈I} w_i · MLP_i(x) + MLP_shared(x)                   expert and shared width 2048, fp8
```

Against DeepSeek-V4.1's MoE (`ds4_route`, `ds4_swiglu*`, `ds4_xgemv`): scores
`σ(l)` instead of `sqrt(softplus(l))`, 8 of 288 instead of 6 of 384, scale 2.5
instead of 1.5, the same bias-for-choice / score-for-weight rule, the same
`+1e-20` normalization and the same clamps. No VL router bias.

## 6. The MTP layer (`layers.45`; [L]:151-287, [M] drafter)

A DeepSeek-V3-style NextN block: one DSA decoder layer **without
hyper-connections** (no `hc_*` tensors; plain residual adds), fed by the
trunk's final hidden and the next token:

```
inputs for position i: h_i = RMSNorm(mean_s H_i; norm.weight) (the trunk's final-normed hidden, [L]:396 "h_nextn"), id_{i+1}
u = W_eh [ RMSNorm(Embed(id_{i+1}); enorm) ; RMSNorm(h_i; hnorm) ]        eh_proj bf16 [4096, 8192]
r = u + DSA_45( RMSNorm(u; input_layernorm) )          own latent + indexer caches, a full indexer, positions from 0
r = r + MoE_45( RMSNorm(r; post_attention_layernorm) )  288 routed + shared experts (7.25 GB), the trunk's router rule
logits = lm_head · RMSNorm(r; shared_head.norm)         the trunk's head (no shared_head.head tensor)
next draft step: h ← RMSNorm(r; shared_head.norm), id ← the sampled draft     ([M] feeds its own normed output back)
```

Prefill runs the MTP over the prompt (rows pair `h_i` with `id_{i+1}` and the
first sampled token) to fill its caches; drafting then steps autoregressively
through the MTP layer only (1 DSA layer + 8 experts + the 1.27 GB head per
draft). Drafts are verified by the trunk, so the one unverifiable choice (the
post-norm `h_i`; [L] and [M] agree) can only cost acceptance, not correctness.
`index_share_for_mtp_iteration` is read by no reference; below 2051 tokens the
indexer selects everything anyway.

## 7. Tokenizer, chat template, tool calls

* **Tokenizer** (`tokenizer.json`): byte-level BPE, 154,820 merges-vocabulary
  + 36 added tokens (154820–154855), `ignore_merges: true`, no normalizer,
  pre-tokenizer `Split(regex, Isolated)` + `ByteLevel(add_prefix_space false,
  use_regex false)`, decoder ByteLevel. **The regex is Llama-3's, character
  for character** (`\p{N}{1,3}`): Espetero's `kLlama3Regex` matcher applies
  and `ignore_merges` is implemented (`tokenizer.cpp:546`). 18 added tokens are
  `special` (`<|endoftext|>`, `[gMASK]`, `<sop>`, `<|system|>`, `<|user|>`,
  `<|assistant|>`, `<|observation|>`, image/video/audio markers), 18 are
  **not** (`<think>`, `</think>`, `<tool_call>`, `<arg_key>`, …, `<|image|>`):
  they must still tokenize atomically (HF treats every added token as a unit;
  `special` only governs `skip_special_tokens`). Verify with `tok_ref.py`.
* **Chat template** (10.8 KB): `[gMASK]<sop>` prefix, `<|system|>Reasoning
  Effort: {Low|High|Max}` (any `reasoning_effort` other than low/high → Max),
  tools as JSON inside `<tools>` with `<tool_call>{name}<arg_key>k</arg_key>
  <arg_value>v</arg_value>…</tool_call>` calls, `<|observation|>` +
  `<tool_response>…</tool_response>` blocks (sorted to match the calls' ids),
  `<think>…</think>` kept only after the last user turn unless
  `clear_thinking`, generation prompt `<|assistant|><think>`. Uses
  `namespace`, `break`, `unique(case_sensitive)`, `sort(attribute,
  case_sensitive)`, `tojson(ensure_ascii=False)`, `capitalize`, `split`,
  `strip`, `is mapping / iterable / string`, `content.0.type` — all present in
  `src/tmpl`; add the template to `tests/ref/tmpl_fixtures` and compare with
  `tmpl_ref.py`. A prompt ending in `<think>` is an open reasoning block (as
  DSv4; `/nothink` is a vocabulary token the user may send).
* **Tool calls:** GLM's `<tool_call>NAME<arg_key>K</arg_key><arg_value>V
  </arg_value></tool_call>` is not parsed yet (`chat.cpp` knows Qwen's
  `<tool_call><function=` and DeepSeek's DSML): a new parser (values are JSON
  when the schema says so, else strings) and `tool_choice` forcing with
  `<tool_call>` / `<tool_call>NAME`.

## 8. Memory plan on 128 GB (and disk)

The Mac Studio has 128 GiB (137.4 GB) and, today, **61 GB free on its
3.6 TB disk**; the MacBook 64 GB / 68 GB free. The FP8 checkpoint is 328 GB.
**Nothing below starts before ~330 GB are freed on the Studio** (Fran's call
which models go); the conversions of §8.2 also start from the FP8 files.

### 8.1 First target: FP8 as stored, experts from the SSD (DeepSeek-V4.1's plan)

| What | Bytes | Where |
|:--|--:|:--|
| embeddings bf16 (row gather) | 1.27 GB | resident |
| LM head bf16 | 1.27 GB | resident (an 8-bit copy later: 0.63 GB) |
| 34 KDA layers, bf16 | 9.37 GB | resident (8-bit at load later: 4.7 GB) |
| 11 DSA layers (fp8 projections, bf16 `kv_b` + indexer) | 1.64 GB | resident |
| dense MLPs of layers 0–2, fp8 | 0.45 GB | resident |
| 43 shared experts, fp8 | 1.08 GB | resident |
| routers, biases, hc (fp32), norms, MTP `eh_proj` + DSA | ~0.5 GB | resident |
| **resident weights** | **≈ 15.6 GB** | |
| routed experts 12,384 × 25.17 MB (fp8 + scales) | 311.7 GB | **SSD**; LRU cache of ≈ 93–97 GB ≈ 3,700–3,850 slots = 30 % of the experts (DSv4: 34 %) |
| per sequence: KDA 152.6 MB + DSA caches 17.6 KB/token (+ MTP 1.6 KB) | 0.78 GB at 32 K | 6 slots ≈ 4.7 GB |
| pass scratch (1024-row blocks; 64 KB of streams per row, expert staging) | ~2 GB | |
| vision tower | 1.13 GB | not loaded (text only) |

Cache size as DSv4 computes it: `min(GPU working set − weights − 3, RAM −
weights − 24, available − 6)` ≈ 95 GB. The cache's host code (locked slots,
`pread` misses after each layer's routing, prefetch of the next layer, the
streaming reader for long passes, the warm file) is reused unchanged
(§10); only the slot layout (codes 25.17 MB `down|gate|up`, scales 6 KB) and
the GEMV format differ.

### 8.2 Later: smaller experts, so that more of them are resident

| Expert format | MB / expert | all experts | resident at 95 GB | GPU bytes / token (8-bit KDA + head) |
|:--|--:|--:|--:|--:|
| fp8 as stored | 25.2 | 312 GB | 30 % | 17.2 GB |
| 4-bit affine g64 (Espetero/MLX, scale + bias) | 14.2 | 175 GB | 54 % | 13.0 GB |
| Q4_K-class GGUF (unsloth UD-Q4_K_XL is 200 GB in all) | 13.4 | 166 GB | 57 % | 12.7 GB |
| 3-bit g64 (EQ3 class) | 11.0 | 136 GB | 70 % | 11.8 GB |
| 2.5-bit class (UD-IQ2_XXS 102 GB, UD-Q2_K_XL 109 GB in all) | 8.1 | 100 GB | ~95 % | 10.7 GB |

Nothing at 3 bits or more fits resident with the trunk; every route keeps the
expert cache and streams the rest, and a 2-bit model is a quality question
Fran decides. Two ways to get there: (a) `tools/glm5_quant.py`, numpy, FP8 →
Espetero 4-bit g64 affine experts in safetensors with the same two-read
layout (the trunk stays FP8/bf16), one more decode format in the expert
GEMV; (b) unsloth's GGUF files (`unsloth/GLM-5.3-Flash-GGUF`, 2026-10-05;
headers read, nothing downloaded):

| File | Size (experts) | Routed experts: gate/up, down | Dense matrices | Espetero's GGUF converter |
|:--|--:|:--|:--|:--|
| UD-Q4_K_XL | 199.7 GB (189.9) | Q4_K (one layer Q5_K), Q5_K (3 layers Q6_K) | Q8_0 everywhere (KDA q/k/v/o, DSA, `attn_k_b`/`attn_v_b`, indexer, hc `fn`, head, embeddings); routers, norms, convs F32 | **every type supported** (Q4_K, Q5_K, Q6_K, Q8_0, F32) |
| UD-Q3_K_XL | 147.5 GB (137.8) | IQ3_XXS, IQ4_XS | Q8_0; head Q6_K | IQ3_XXS missing |
| UD-IQ3_XXS | 120.4 GB (112.5) | IQ2_S, IQ3_S | Q6_K; `k_b`/`v_b`, indexer Q8_0 | IQ2_S, IQ3_S missing |
| UD-Q2_K_XL | 108.7 GB (101.7) | IQ2_XS, IQ3_XXS | Q5_K / Q6_K; head Q4_K | IQ2_XS, IQ3_XXS missing |

The tensor names map one-to-one onto the HF ones ([LC]: `blk.N.hc_attn_fn`,
`ssm_conv1d_q/k/v` [4, 1, 8192], `ssm_f_a/f_b/g_a/g_b/beta/a/dt`,
`attn_q/k/v` for the KDA projections, `attn_k_b`/`attn_v_b` = `kv_b_proj`
split and transposed per head (the absorbed layout), `indexer.attn_q_b`,
`indexer.attn_k`, `indexer.proj`, `indexer.k_norm.{weight,bias}`,
`indexer_compressor_gate/ape`, `ffn_{gate,up,down}_exps` as one 3-D tensor per
matrix (a per-expert slice is contiguous: three reads per expert for the SSD
cache), `exp_probs_b.bias`, `ffn_*_shexp`, `nextn.*`; 1412 tensors, 46
blocks). The metadata carries the config (`glm5-next.*`: `kda.head_dim`,
`kda.gate_lower_bound`, `attention.indexer.{head_count,key_length,top_k,
kpool,index_share_mtp}`, `hyper_connection.*`, `expert_weights_scale` 2.5,
`expert_gating_func` 2, `swiglu_clamp_{exp,shexp}` per layer,
`nextn_predict_layers` 1), the tokenizer (`gpt2` model, pre `glm4` =
Llama-3's regex) and the chat template. Only UD-Q4_K_XL is loadable with the
types Espetero has; the ≤ 150 GB files need the IQ2/IQ3 codebook
dequantizers first. (a) is the smaller step and keeps the FP8 reference as the
parity anchor; (b) is §12's route B.

## 9. Per-token traffic, state, expected speed (FP8 route, M4 Max)

Weights read per decoded token (one sequence; bytes of the checkpoint's
formats, scales included):

| Component | MB / token | Share |
|:--|--:|--:|
| KDA layers, 34 × 275.5 (q/k/v/o bf16 268.4, gates 6.8) | 9,367 | 41.6 % |
| routed experts, 42 × 8 × 25.17 (fp8) | 8,457 | 37.6 % |
| DSA layers, 11 × 149.2 (o_proj 67.1, kv_b 33.6, q_b 25.2, indexer 14.9, q_a/kv_a 8.4) | 1,641 | 7.3 % |
| LM head (bf16) | 1,269 | 5.6 % |
| shared experts, 42 × 25.17 | 1,057 | 4.7 % |
| dense MLPs, 3 × 151 | 453 | 2.0 % |
| hyper-connections, 90 × 1.57 (fp32 at run time) | 142 | 0.6 % |
| routers + biases | 99 | 0.4 % |
| **total** | **22,490** | at ~480 GB/s (DSv4's fp8 GEMVs run near the memory's bandwidth): **~47 ms**; 546 GB/s peak: 41 ms |
| with the KDA matrices and the head as 8-bit copies at load | 17,170 | ~36 ms |

Plus per step: KDA state read + written 2 × 142.6 MB, DSA latents ≤ 2 MB per
layer, pooled keys 64 B × N × 12. The MTP layer per draft: 149 MB + 8 experts
201 MB + the head 1,269 MB (the head dominates: a 4-bit draft copy, as the
forest's `mtp_head4`, would cut a draft to ~0.7 GB).

**Decode.** The SSD decides: a miss costs 25.17 MB / 6.7 GB/s = 3.8 ms, and a
token uses 336 routed experts (+ 8 for a draft). DeepSeek-V4.1 measured 20
misses of 240 uses (8.3 %) with 34 % of its experts cached; its routing rule
(sigmoid-class score + bias, aux-loss-free) is GLM's, its cache share is 4
points higher.

| Miss rate | Misses / token | SSD | GPU | tok/s |
|:--|--:|--:|--:|--:|
| 8.3 % (DSv4's) | 28 | 105 ms | 47 ms | **6.6** |
| 15 % | 50 | 189 ms | 47 ms | 4.2 |
| 30 % | 101 | 379 ms | 47 ms | 2.3 |
| 70 % (no locality) | 236 | 885 ms | 47 ms | 1.1 |

Expect 3–6 tok/s in the FP8 route (DSv4: 10.2), and the measurement to be the
first real number (`ESPETERO_DS4_TRACE`'s replay tools apply unchanged). The
4-bit route of §8.2 (54 % resident, ≤ 10 misses × 14.2 MB ≈ 20 ms, GPU 13 GB
≈ 27 ms) is where ~20 tok/s would come from; 3-bit ~25.

**Prefill.** Above ~256 rows every layer touches all 288 experts: a pass
streams 42 × 288 × 25.17 MB = **304 GB from the SSD, ≈ 43–45 s at 7 GB/s,
whatever the prompt length** (DSv4: 202 GB, 30.6 s for 4.5 K tokens), plus
the GPU's prefill (~150 TFLOP for 4 K tokens, ~10 s). The KDA chunked kernel
and `ds4_gemm_f8` carry the GPU side; the first token of a long prompt will
take about a minute.

## 10. Mapping onto Espetero

**Home:** a model of its own beside DeepSeek-V4.1, built from it:
`src/model/impl/glm5.h` (config, layer weights, sequence slots; reusing
`ds4::Ckpt`, `ExpertCache`, `Mat8`/`MatB`), `src/model/impl/glm5_model.h`
(`Glm5Model : NativeModel`, `model.mm` picks it for `model_type glm5_next` /
`glm5_next_text`), `kernels/86_glm5.metal` (the variants and new kernels
below). DeepSeek's model file minus Engram, the compressor / CED / candidate
blocks, window rings, YaRN RoPE, activation rounding, DSpark and vision; plus
KDA layers, the k-pool indexer, the absorbed MLA, dense MLP layers, the
sigmoid router, the MTP layer. The expert-cache host code (misses, prefetch,
streaming, eviction, warm file) is copied first and factored into a shared
header once both models run — a question for Fran (§12).

| Op / tensor | Espetero piece | Status |
|:--|:--|:--|
| safetensors index, `pread`, F_NOCACHE, partial loads (`ESPETERO_GLM5_LAYERS`) | `ds4::Ckpt`, `raw`/`vecf` | as is |
| fp8 GEMV / GEMM with **fp32 scales per 128 × 128 block** (`weight_scale_inv [N/128, K/128]`) | `ds4_gemv_impl` FMT 0 (32 × 32 e8m0), `ds4_gemm_f8`, the expert `ds4_xgemv` | **variant**: one more FMT (scale `S[(n/128)·(K/128) + k/128]` as fp32, no `· 256`), in the three kernels |
| bf16 GEMV / GEMM (KDA projections, `kv_b`, indexer, routers, head, `eh_proj`) | `ds4_gemv_bf16` (`gemvb`), `ds4_gemm_bf16` | as is; the three q/k/v matrices and (`f_a`, `g_a`, `b`) concatenated at load into one GEMV each |
| embeddings into 4 streams | `ds4_embed` | as is |
| hc coefficients, Sinkhorn | `ds4_hc_mix` / `ds4_hc_part` + `ds4_hc_fin` | as is (`eps` 1e-5, `fn` converted to fp32 at load) |
| `x = Σ pre_s H_s` + RMSNorm | `ds4_hc_pre` | as is, with the **same block's** mix row |
| `H_k = post_k y + Σ_j C[j][k] H_j` | `ds4_hc_post` | as is |
| head: mean of the streams, final norm | `ds4_hc_mean` (DSpark used it), `ds4_rmsnorm` | as is |
| KDA conv + SiLU + l2norm(q, k) | `conv_prep` (30_deltanet) | as is: C 24576, Hk = Hv 64, eps 1e-6/128; a flag skips its scalar-gate output |
| KDA gates `g` (per channel), `β` | — | **new** `glm5_kda_gate` (tiny: from the `f_b`, `b` GEMV outputs, `A_log`, `dt_bias`) |
| KDA decode (1 row per sequence) | `gdn_decode_fused_tr8` | **variant** `glm5_kda_decode`: per-channel decay vector, σ output gate, norm eps 1e-5, gates from buffers |
| KDA chunked prefill | `deltanet_chunk` | **variant** `glm5_kda_chunk`: staged `K̃`, `K̂`, `Q̃`, per-k-row state decay (§3) |
| KDA gated norm (prefill) | `gated_rmsnorm_sig` (70_qwen4) | as is (eps parameter) |
| RMSNorms (`q_a`, `kv_a`, input/post, MTP norms) | `ds4_rmsnorm` | as is |
| latent `c` into the slot's cache (fp16) | `ds4_store_half` | as is |
| `q̃_h = W_kb,hᵀ q_h`, `o_h = W_vb,h õ_h` | — | **new** `glm5_headmv`: per-head bf16 matvec over `kv_b_proj` slices (decode: a simdgroup per head and 16 outputs; prefill: `ds4_gemm_bf16` per head) |
| attention over the latents, top-k or dense | `ds4_attn` / `ds4_attn_split` | as is: `win` 0, `ratio` 1, sink buffer = −∞, `idx` rows of 2051 token ids (−1 padded) |
| indexer key LayerNorm (mean, bias) | — | **new** `glm5_layernorm` (tiny) |
| k-pool pooling (`softmax` of gate + ape over 4 tokens, per channel) | `ds4_compress` (ratio-2 pooling with pending rows) | **variant** `glm5_kpool`: 4 members, `+ ape[i][c]`, pending key + gate rows |
| pool scores `Σ_h w_h relu(q̃_h · K_b)` | `ds4_index_score` | as is (ratio 4; weights pre-scaled as DSv4 does) |
| top-512 pools | `ds4_topk` | as is (k 512) |
| pools → 2048 token ids + tail | — | **new** `glm5_expand_sel` (tiny) |
| router GEMV (fp32 math) | `ds4_gemv_bf16` into fp32 | as is |
| routing | `ds4_route` | **variant**: scoring switch (`σ` vs `sqrt(softplus)`), scale from config |
| SwiGLU with clamps | `ds4_swiglu`, `ds4_swiglu_w` (lim 10) | as is |
| routed experts | `ds4_xgemv` + `ds4_gemm_f8` (≥ 16 rows) | the FMT variant above; slot = `down|gate|up` codes + scales |
| expert cache: slots, misses, prefetch, streaming, eviction, warm file, `--expert-cache-gb` | `ExpertCache`, `read_slots`, `evict_slot`, `prefetch`, `stream_to`, `warm_*` | as is (a two-piece read for the one split expert) |
| MoE sum, copies | `ds4_moe_sum`, `ds4_copy` | as is |
| passes: 1024-row blocks, layer by layer, several sequences per pass, routing on the host between phases | `Dsv4Model::forward`, `run_trees`, segments | as is in structure |
| slots: `slot_save`/`slot_load`, fork, reset | `Dsv4Model` slot code | as is (new per-slot buffers: KDA state, latents, index keys/gates, pooled keys, MTP caches) |
| MTP drafts (`NativeModel::draft`, `info.mtp`) | DSv4's `spec_chain` / rollback pattern | **new** for this model: prefill through the MTP layer, greedy drafts, rollback of rejected rows |
| logits / top-k / GPU draw readouts | DSv4's readout code | as is |
| tokenizer | `tok` (Llama-3 split, `ignore_merges`) | as is; verify the non-special added tokens |
| chat template | `tmpl` | as is; fixtures |
| tool calls | `engine/chat.cpp` | **new** GLM parser + forcing |
| vision (`model.visual.*`, 24 blocks, SwiGLU MLP, 2 × 2 downsample conv, merger) | `vision.h` / `60_vision.metal` | **not now** (text only); inventory in Appendix A |

Dispatches per decoded token: ≈ 24 per KDA layer, ≈ 30 per DSA layer, ≈ 1,150
per token (DSv4: 1,285); one host round trip per MoE layer for the routing, as
DSv4.

Loader (`glm5.h`, written: the config with its validation, every layer's
tensor names, dtypes and shapes, the routed experts' byte locations in
cache-slot order; `tests/ref/glm5_ref.py fake-ckpt` writes a checkpoint
directory of sparse shards holding the real headers and `build/glm5_load_test`
checks the index on it, so the loader is exercised before any download): HF
names `model.language_model.layers.L.*`, `lm_head.weight`,
`model.language_model.{embed_tokens,norm}.weight`; concatenations at load:
q/k/v projections, the three conv weights (tap order kept), `f_a|g_a|b` rows;
`kv_b_proj` kept as stored (per-head slices by offset); `hc_*_fn` bf16 → fp32;
the expert index `XLoc{file, w, s}` from the headers (order `down|gate|up`,
checked contiguous, the split expert recorded as two pieces).

## 11. Parity plan

1. **Reference:** `tools/glm5_ref.py`, a numpy transcription of [T] in fp32
   (the `tools/dsv4_ref.py` pattern: memory-mapped safetensors, fp8 LUT ×
   `weight_scale_inv` per 128 × 128 block, KDA in its recurrent form, DSA
   dense for prompts ≤ 2051 tokens, the k-pool indexer transcribed and
   checked separately on a > 2051-token synthetic case), writing per-layer
   residual streams `h<l>.f32 [T, 4, 4096]` and the logits of the last rows;
   `tests/ref/glm5_ref.py` wraps it in the `gemma4_ref.py` case / compare
   format (teacher-forced rows: argmax, log-sum-exp, top 64) for
   `build/logits_dump`. CPU only, ~30–40 GB of transient fp32 experts, ~10
   minutes per 64-token prompt on the Studio (`gpuq submit -s --mem 40`).
   No activation rounding anywhere: [T] runs bf16/fp32 matmuls on dequantized
   weights (`activation_scheme: dynamic` is a serving choice of vLLM-class
   kernels, not the model's definition); Espetero multiplies exact fp8 weights
   by fp32 activations, so agreement is to half-precision storage noise
   (DSv4 without rounding: ≤ 3·10⁻⁴ relative per token).
2. **Layer dumps:** `ESPETERO_GLM5_DUMP=dir` (as `ESPETERO_DS4_DUMP`) and
   `ESPETERO_GLM5_LAYERS=n` to compare the first layers while the download
   finishes; routing near-ties forced as in DSv4 (`--force-routes`).
3. **Logits:** `logits_dump` cases (prefill, decode, tree, split, gen) on
   3–4 prompts against the reference's teacher-forced rows; greedy
   continuations identical for 64 tokens or the divergence explained by the
   reference's own top-2 margin.
4. **KDA kernels:** chunked prefill against the decode recurrence inside one
   process (the forest's `dn_chunk` A/B pattern): identical state and outputs
   to 1e-5 for 1–4 K-token prompts.
5. **Indexer:** selection sets of a 4 K-token synthetic prompt against the
   numpy indexer (rows without score ties exact); dense vs top-k attention
   equal below 2051 tokens.
6. **MTP:** verified chains produce the plain chain's tokens; acceptance
   rate logged (`ESPETERO_GLM5_SPEC_LOG`).
7. **Serving** (`espetero serve` on the Studio through `gpuq -x`, idle GPU):
   a chat, a tool call, an MTP-drafted decode; tok/s, first-token time at
   256 / 4 K tokens, memory (`--expert-cache-gb` default), cache hit rate.

**Results so far** (2026-10-22, layers 0–3 of the FP8 checkpoint fetched by
range: KDA, dense MLP, the first DSA layer with its indexer, the first MoE
layer; Mac Studio M4 Max):

| Check | 21-token chat | 2,138-token prompt |
|:--|--:|--:|
| `tools/glm5_ref.py` vs mlx-vlm 0.7.6's own layers (`tests/ref/glm5_mlx_check.py`), worst per-row relative L2 of a branch's update | 2.7e-4 | 3.1e-4 |
| Espetero tier 0 (`build/glm5_run`) vs `tools/glm5_ref.py`, worst per-row relative L2 of the streams | 1.8e-4 | 9.2e-3 (median 1.4e-5) |
| … the same, one prefill + decode steps (8 / 18 tokens) | identical bits | identical bits |
| indexer selections, Espetero vs reference | identical | identical (all rows, top-512 regime included) |
| routes (top-8 sets), Espetero vs reference | identical | 1 row differs (a near-tie: its row carries the 9.2e-3) |

| UD-IQ3_XXS slice (blocks 0–3, GGUF through the native block kernels) vs `tools/glm5_ref.py` on the same GGUF weights | 1.0e-4 | 1.5e-4 (prefill + 18 decode steps; routes identical, selections differ in 2 rows) |
| UD-IQ3_XXS's own distance from the FP8 weights (the two references) | 2–3 % in layers 0–2, 20 % after the IQ2_S / IQ3_S MoE (routes differ in 9 / 21 rows) | |

Layers 0–2 agree to 3e-6 (fp32 noise); layer 3's attention to 3.6e-4 (half
latents); mlx-vlm's layer-3 MLP differs by 2–3e-4 because the check holds its
expert stacks in float16. GPU time at tier 0: 55 ms for 21 tokens, 5.6 s for
2,138 (KDA token by token, GEMVs per row), 4.3–6.0 ms per decoded token for
these four layers. The native GGUF-block GEMVs (`tests/gguf_gemv_test.mm`)
match gguf-py on random and real blocks of all nine types to ~1e-8.

Determinism as DSv4: bit-reproducible passes except the pipelined MoE of long
passes (its row sums depend on which experts were cached); exact comparisons
run with `ESPETERO_*_PREFETCH=0 ESPETERO_*_PIPE=0`.

## 12. Reference discrepancies, decisions, open questions

| Topic | [T] | [L] / [M] | Decision |
|:--|:--|:--|:--|
| residual / hc dtype | bf16 streams, `post`/`C` cast to bf16 | fp32 graph in [L] | fp32 (as DSv4) |
| KDA chunk size | 64 (per-channel decay inside the products) | 16 ([L] `CS = kda ? 16 : 64`) | 8-token WY steps with the factored `K̃ / K̂` (§3); the recurrence is the tier-0 reference |
| KDA l2norm | eps 1e-6 after the sum | [M]: `rms_norm(eps 1e-6/128)` × scale, equal | `conv_prep` eps 1e-6/128 |
| indexer below the budget | runs top-k (selects all) | [M] skips, [L] runs | dense attention, exact (§4.2) |
| indexer ties | `torch.topk` | `argpartition` / `ggml_top_k` | lowest pools win (deterministic), such rows excluded from selection parity |
| indexer `k_norm` | `nn.LayerNorm(128, eps 1e-6)` with bias | same | a LayerNorm, not an RMSNorm |
| MTP input | not implemented | the trunk's **post-norm** hidden; next steps feed the MTP's own normed output | follow [L]/[M]; measure acceptance |
| MTP position-0 embedding | — | [M] zeroes the token embedding at position 0 (left-padding handling) | not done ([L] does not); positions start at 0 with the real token |
| `index_share_for_mtp_iteration` | ignored | ignored | ignored; the MTP layer runs its own indexer |
| activation rounding | none | none (weights dequantized to the compute type) | none |
| `hc_mult ≠ 4` | general | asserts 4 | assert 4 |
| head | `norm(mean(H))` | same | same |

### The two routes to a first serving model (Fran picks)

| | **A. FP8 checkpoint, experts from the SSD** | **B. GGUF first: unsloth UD-Q4_K_XL** |
|:--|:--|:--|
| Disk on the Studio (61 GB free today) | 328 GB (`zai-org/GLM-5.3-Flash`, 62 shards) | 200 GB (6 shards); the 120 GB UD-IQ3_XXS only after IQ2_S / IQ3_S dequantizers exist |
| Memory | 15.6 GB resident; expert cache ≈ 95 GB = 30 % of the experts | ≈ 10 GB resident (Q8_0 dense); 190 GB of experts, ≈ 50 % cached |
| Bytes per decoded token | 22.5 GB (bf16 KDA 9.4, fp8 experts 8.5) → ≈ 47 ms GPU | ≈ 14 GB (Q8_0 KDA 4.9, Q4_K/Q5_K experts 15.2 MB × 336 = 5.1) → ≈ 29 ms GPU |
| SSD per token (DSv4's locality, 8 %) | 28 misses × 25.2 MB ≈ 105 ms | ≈ 5–15 misses × 15.2 MB ≈ 11–34 ms |
| Expected decode | **3–6 tok/s** (SSD-bound; 1 tok/s without locality) | **12–20 tok/s** (3–5 without locality) |
| Prefill of a long prompt | 304 GB streamed per pass, ≈ 45 s + GPU | 190 GB, ≈ 28 s + GPU |
| Parity story | exact: `tools/glm5_ref.py` (fp32 numpy of [T] on the same weights), per-layer dumps, logits to half-precision noise; kernels proven against the model itself | against llama.cpp on the very same file (teacher-forced logits, `llama-cli`/`llama-perplexity` with mmap on the Studio: slow but complete) — the kernels are then checked against another engine's quantized arithmetic, not the model; the FP8 transcription stays the oracle for anything that disagrees, so route B still wants the FP8 files (or a smaller FP8 slice: the first layers) at some point |
| Engine work | §10 as written | §10 plus: the GGUF tensor map and metadata for this model (`gguf.h` has the `qwen35`/`gemma4` ones), per-head `k_b`/`v_b` slices, expert slices from 3-D tensors (3 reads), Q4_K/Q5_K/Q6_K experts in the DSv4-derived path (repack each expert into the forest's affine layout when it enters the cache, with the existing conversion kernels, or K-quant-native GEMVs), Q8_0 dense GEMVs (one more `ds4_gemv` format) |
| What it serves | the model as released, lossless; any later quantization is Espetero's own (EQ) | what M4 Max users download; unsloth's choices (Q8_0 dense, Q4_K/Q5_K experts) |
| Order | A first when the disk is freed; (a) of §8.2 (4-bit experts) for speed after parity | B first only if the 328 GB cannot be freed; A's kernels are the same (the cache is format-agnostic), the GGUF map and K-quant formats are additions |

Decisions taken with the parent session's advice (pending Fran's review):

1. Load-time 8-bit copies of the bf16 KDA matrices and the head are
   **opt-in** (`--dense-8bit`, measured both ways); bf16 by default (lossless
   by default; EQ covers the dense matrices properly later).
2. DeepSeek-V4.1's expert-cache host code is **copied** into the GLM model
   first and factored once both run (no premature abstraction).
3. Context default 32 K (0.78 GB per slot) and 6 slots, as DSv4, until
   measured.

Open for Fran: the disk (which models go; the download), and A or B.

## 13. Status (2026-10-26, branch arch/glm5)

**Verified on the GPU** (against `tools/glm5_ref.py`, the numpy transcription of transformers, itself checked
against mlx-vlm's own layers to 3e-4; slices of layers 0–3 on the Studio HDD, §11):

* every layer kind on the FP8 weights: KDA (token by token and chunked), mHC, dense MLP, nope MLA with the
  k-pool indexer (dense and top-512 regimes), the sigmoid router, routed and shared experts; prefill and decode
  (bit-identical); 1.8e-4 worst per-row on a 21-token chat, 9.2e-3 (one near-tie route) on 2,138 tokens;
* the same on unsloth's UD-IQ3_XXS GGUF (native IQ2_S / IQ3_S / Q6_K / Q8_0 kernels): 1.0e-4 / 1.5e-4 against
  the reference on the same weights; all nine GGUF block types of UD-IQ3_XXS and UD-Q4_K_XL to ~1e-8;
* the expert cache: a 40-slot pool (fewer than one layer's experts) bit-identical to a full one;
* the tokenizer (HF tokenizer.json and the GGUF's own vocabulary: all 25,026 reference cases), the chat
  template (564 cases byte-identical to transformers), the GLM tool-call parser (unit tests);
* `espetero serve` on the slice (4 layers + head, `ESPETERO_GLM5_HEAD_ANY=1`): chat, sampled chat, streaming,
  completions well formed.

**Written, not yet verified:** the MTP layer (model and reference; the slices lack layer 45's tensors),
checkpoint files (`slot_save` / `slot_load`), trees with leaves.

**Needs the full model** (a GGUF on the Studio SSD, Fran's call): logits and greedy text against llama.cpp on
the same file, perplexity, the forced tool call (GLM's template always opens `<think>`; forcing waits for the
model's `</think>`), MTP acceptance, decode / prefill speed with the expert cache, memory.

**Not started:** vision tower, several sequences per GPU pass, GPU top-k / draw, GEMMs for prefill (tier 0
runs every row through GEMVs), MTP drafting in the engine.

## Appendix A. Tensor inventory ([C]; logical shapes `[out, in]`)

```
lm_head.weight                                           BF16 [154880, 4096]       1.27 GB
model.language_model.embed_tokens.weight                 BF16 [154880, 4096]       1.27 GB
model.language_model.norm.weight                         BF16 [4096]

model.language_model.layers.<l>.                         l = 0 … 44
  input_layernorm.weight, post_attention_layernorm.weight   BF16 [4096]
  hc_attn_fn, hc_ffn_fn                                  BF16 [24, 16384]          786 KB each
  hc_attn_base, hc_ffn_base                              F32 [24]
  hc_attn_scale, hc_ffn_scale                            F32 [3]
 KDA layers (l mod 4 ≠ 3):
  self_attn.q_proj, k_proj, v_proj                       BF16 [8192, 4096]         67.1 MB each
  self_attn.q_conv1d, k_conv1d, v_conv1d                 BF16 [8192, 1, 4]
  self_attn.f_a_proj, g_a_proj                           BF16 [128, 4096]
  self_attn.f_b_proj, g_b_proj                           BF16 [8192, 128]
  self_attn.b_proj                                       BF16 [64, 4096]
  self_attn.A_log                                        F32 [64]
  self_attn.dt_bias                                      F32 [8192]
  self_attn.o_norm.weight                                BF16 [128]
  self_attn.o_proj                                       BF16 [4096, 8192]         67.1 MB
 DSA layers (l mod 4 = 3, and l = 45):
  self_attn.q_a_proj                                     F8 [1536, 4096]   + weight_scale_inv F32 [12, 32]
  self_attn.q_a_layernorm.weight                         BF16 [1536]
  self_attn.q_b_proj                                     F8 [16384, 1536]  + F32 [128, 12]
  self_attn.kv_a_proj_with_mqa                           F8 [512, 4096]    + F32 [4, 32]
  self_attn.kv_a_layernorm.weight                        BF16 [512]
  self_attn.kv_b_proj                                    BF16 [32768, 512]         33.6 MB  (per head h: rows h·512 .. +255 = W_kb,h, +256 .. +511 = W_vb,h)
  self_attn.o_proj                                       F8 [4096, 16384]  + F32 [32, 128]
  self_attn.indexer.wq_b                                 BF16 [4096, 1536]
  self_attn.indexer.wk                                   BF16 [128, 4096]
  self_attn.indexer.k_norm.weight, k_norm.bias           BF16 [128]
  self_attn.indexer.weights_proj                         BF16 [32, 4096]
  self_attn.indexer.index_kpool_compress_gate            BF16 [128, 4096]
  self_attn.indexer.index_kpool_compress_ape             BF16 [4, 128]
 dense layers (l = 0, 1, 2):
  mlp.gate_proj, up_proj                                 F8 [12288, 4096]  + F32 [96, 32]
  mlp.down_proj                                          F8 [4096, 12288]  + F32 [32, 96]
 MoE layers (l = 3 … 45):
  mlp.gate.weight                                        BF16 [288, 4096]
  mlp.gate.e_score_correction_bias                       F32 [288]
  mlp.shared_experts.gate_proj, up_proj                  F8 [2048, 4096]   + F32 [16, 32]
  mlp.shared_experts.down_proj                           F8 [4096, 2048]   + F32 [32, 16]
  mlp.experts.<0..287>.gate_proj, up_proj                F8 [2048, 4096]   + F32 [16, 32]      8.39 MB + 2 KB each
  mlp.experts.<0..287>.down_proj                         F8 [4096, 2048]   + F32 [32, 16]
 MTP layer (l = 45) also:
  eh_proj.weight                                         BF16 [4096, 8192]
  enorm.weight, hnorm.weight, shared_head.norm.weight    BF16 [4096]

model.visual.                                            (not loaded; for the vision follow-up)
  patch_embed.proj.{weight BF16 [1024, 3, 2, 14, 14], bias}
  blocks.<0..23>.{norm1, norm2}.weight [1024]; attn.qkv.{weight [3072, 1024], bias}; attn.{q_norm, k_norm}.weight [64];
                 attn.proj.{weight [1024, 1024], bias}; mlp.{gate_proj, up_proj}.{weight [4096, 1024], bias}; mlp.down_proj.{weight [1024, 4096], bias}
  post_layernorm.weight [1024]; downsample.{weight [4096, 1024, 2, 2], bias}
  merger.proj.weight [4096, 4096]; merger.post_projection_norm.{weight, bias} [4096];
  merger.{gate_proj, up_proj}.weight [10240, 4096]; merger.down_proj.weight [4096, 10240]
```

Parameter counts from the shapes: routed experts 311.6 B (12,384 × 25.17 M),
KDA layers 4.7 B, DSA layers 1.4 B, dense MLPs 0.45 B, shared experts 1.08 B,
embeddings + head 1.27 B, vision 0.56 B. Active per token by these shapes:
8 experts × 42 layers 8.46 B + KDA 4.7 B + DSA 1.4 B + shared 1.06 B + head
0.63 B + dense 0.45 B + routers and hc 0.09 B ≈ **16.8 B** (the model card's
18 B presumably counts the embeddings and the MTP layer).
