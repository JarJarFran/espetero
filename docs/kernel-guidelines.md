# Writing Metal kernels for Espetero

How to add or change a kernel so that it moves the engine toward the machine's
roofline, stays correct, and stays general where generality is free. The rules
come from three sources: what an M1 Max measures (numbers marked *measured*
are from `bench_native` / `ab_native` / `parity_test` runs on a 32-core M1 Max
with 64 GB; "baseline" is the engine's plain path before the optimizations a
section discusses), Apple's published guidance for Apple-silicon GPUs (WWDC
2020 "Optimize Metal Performance for Apple silicon Macs", WWDC 2022 "Scale
compute workloads across Apple GPUs", the M5/A19 tech talk on neural
accelerators), and the kernel studies behind Espetero and JEVfire, its
predecessor.

The companion documents: [design.md](design.md) (what the engine is) and
[moe.md](moe.md) (the mixture-of-experts path).

## 1. The machine, and what "100%" means

| Resource (Apple M1 Max, 32-core GPU, 64 GB) | Limit | Baseline reached |
|:--|--:|--:|
| Streaming read bandwidth | 354 GB/s *measured*, 400 nominal | decode GEMVs 290–360 GB/s in isolation; whole decode step 235 GB/s (18.2 GB of weights in 77.2 ms) |
| fp16 `simdgroup_matrix` 8×8 MMA | 10.1 TFLOPS *measured* (10.4 nominal FMA) | quantized GEMMs 7.1–7.5 TFLOPS at 2048 rows (72–74%) |
| Dependent dispatch (barrier, drain, refill) | ≥4.5–5 µs each *measured* with small kernels | decode step: 630 dispatches, 14.4 ms between isolated kernel time (62.8 ms) and in-pass time (77.2 ms), ~23 µs per dispatch |
| Threads per core for good occupancy | 1K–2K (Apple) | GEMV: 256-thread threadgroups, 500–1100 per dispatch |
| Threadgroup memory | 32 KB per threadgroup | attention tiles use ~17 KB |
| Registers | 128 × 32-bit per thread; occupancy is set by peak use | GEMM tiles were chosen for this; 64-row tiles lost to spills |
| Data types | fp16 at the fp32 rate; no int8/fp8 matrix units; 16-bit registers | operands fp16, accumulate fp32 |

Two regimes, two rooflines:

- **Decode (1–8 rows per pass) is bandwidth-bound.** Time = weight bytes /
  bandwidth + everything that is not streaming weights. For Qwen3.8-27B (4-bit
  group 32 with bf16 scale and bias = 5.0 bits/weight, 8-bit group 64 parts =
  8.5) the weights are 18.2 GB, so the floor is 51 ms per token at 354 GB/s
  (19.5 tokens/s). The baseline step is 77.2 ms (13 tokens/s). The 26 ms above
  the floor: 14.4 ms inter-dispatch gaps, 3.9 ms attention (4 threadgroups at
  a 512-token context), 3.4 ms norms/conv/delta rule, ~4 ms GEMV inefficiency.
- **Prefill (≥ 32 rows) is MMA-bound.** A 2048-row pass spends 96.4% of its
  GPU time in the quantized GEMMs at 7.3 TFLOPS; the rest is the delta rule
  (1.6%), attention (1.3%) and norms. The gap to the 10.1 TFLOPS peak is the
  cost of feeding 4/5/6/8-bit weights to the MMA units (bit-plane unpacking,
  scales, staged activations, barriers): about a fifth of the time, measured
  by removing each part in turn.

"100% of the machine" therefore means, for decode: every weight byte read
exactly once per pass at streaming bandwidth, and nothing else on the critical
path; for prefill: MMA units fed continuously. Every kernel decision below is
judged against those two statements.

## 2. Three tiers of kernels: general, family, fused

The engine serves the Qwen3.5 family (dense and MoE) and Qwen3.8-Flash-Next,
which adds a gated four-stream residual, sparse attention and n-gram
embeddings to the same building blocks ([qwen4_exp.md](qwen4_exp.md)). Kernels
are organized so that a new model touches the smallest possible set:

| Tier | What | Geometry | Examples | Where |
|:--|:--|:--|:--|:--|
| **0 · general** | linear algebra and reductions any transformer needs | runtime shapes (`GemmParams`), function constants for the quantization group and scale mode, templates for bit width and tile | `rgemm*`, `rgemv*`, `rgemv8*`, `hgemm*`, `splitk_reduce`, `rmsnorm`, `swiglu_f32`, `embed_*`, `gather/scatter`, `lse_rows`, `argmax_rows`, MoE routing, grouped GEMV/GEMM, combine | `10_gemm.metal`, `15_moe.metal`, `20_embed_norm.metal`, `50_heads.metal` |
| **1 · family** | sequence mixing of the Qwen3.5 family (and Flash-Next's sparse attention, hyper-connections and n-gram embedding) | compile-time constants: head dim 256, rotary 64, delta heads 128, conv width 4, GQA group ∈ {1,2,3,4,6,8} | `attn_prep`, `flash_attention_g*`, `attention_combine`, `kv_save`, `conv_prep`, `conv_state_save`, `deltanet_tr*`, `deltanet_chunk`, `gated_rmsnorm`, `vit_*` | `30_deltanet.metal`, `40_attention.metal`, `45_qsa.metal`, `60_vision.metal`, `70_qwen4.metal` |
| **2 · fused** | decode-step megakernels that exist only to remove dispatches and re-reads for one family and shape | hard-coded per family; always shadowed by a tier-0/1 reference path | fused RMSNorm GEMVs (`*_n`, `*_addp`), GDN decode (`gdn_decode_fused`: conv + gates + recurrence + gated norm), MoE routing and the down projection with the combine (`moe_route*`, `moe_down_comb`); candidates: attention with K/V written straight into the paged pool | `10_gemm.metal`, `30_deltanet.metal`, `15_moe.metal` |

Rules that keep the tiers honest:

1. **A tier-2 kernel never becomes the only implementation.** Its tier-0/1
   reference stays in the tree, selectable at run time (`set_option`), and the
   parity test compares the two on every model that runs.
2. **Geometry is a template parameter, not a literal.** `HD = 256`,
   `Dk = Dv = 128` are `constexpr` today inside kernels; when a kernel is
   touched, lift them to template parameters instantiated by
   `[[host_name(...)]]` lists, so the next family adds an instantiation, not
   a copy. Loops still unroll: the values are compile-time either way.
3. **The host mirrors the tiers.** Keep the format-neutral base (runtime,
   tensors, gguf, matrix layouts, GEMM dispatch, autotune) apart from the
   family code (config, loader, vision, forest): a new family adds family
   code, a new checkpoint format touches the base only.
4. **One representation for weights.** Every format maps exactly to
   `code × scale + bias` per group (MLX affine; GGUF Q4_0/1, Q5_0/1, Q8_0,
   Q2_K–Q6_K, IQ4_NL/XS). A new format is a converter in `gguf.h`/`matrix.h`,
   not a new GEMM.

## 3. Memory rules (decode is won or lost here)

1. **Read each weight once per pass.** A GEMV threadgroup owns 32 output
   columns and streams its columns' K range; its 8 simdgroups split K and
   reduce through threadgroup memory. Anything that reads a weight twice
   (two dispatches over the same matrix, an unfused gate/up pair with two
   passes over shared inputs) is visible directly as bytes.
2. **Coalesce 16-byte loads per lane.** The repacked fragment order
   (`Wr[N/32][K/64][lane][…]`) exists so every lane issues `uint4` loads and
   32 lanes cover a contiguous 1–2 KB block per K step. Keep that property in
   any new layout (expert weights included: experts are packed as one
   `[E·Ne, K]` matrix so the same repack and loads apply).
3. **Stage in threadgroup memory only what is reused by several simdgroups**
   (activations in `rgemm`, K/V tiles in attention, the chunk's K/Q in
   `deltanet_chunk`). Weights are never staged: register-direct beats a
   threadgroup-memory round trip on this GPU (measured on whole GEMMs).
4. **Do not multiply activation traffic to save a dispatch.** The fused
   RMSNorm GEMV variants read fp32 residual rows (20 KB per row) in every one
   of the ~1100 threadgroups of a wide GEMV instead of the fp16 normalized row
   (10 KB): that doubles a re-read that already rivals 12% of the weight
   bytes, and the variants measured slower than a separate `rmsnorm`
   dispatch. The fix that keeps the dispatch saving is for the producer's
   epilogue to also write an fp16 copy of its output, so the consumer loads
   fp16 and applies `rsqrt(ss/D + eps) · w[k]` on load.
5. **Small, hot data goes to `constant` or `setBytes`,** large or per-thread
   indexed data to `device`. Parameter structs are `constant&` via
   `setBytes` (≤ 4 KB); tables indexed per row/segment/tile are `device`.
6. **16-bit everywhere the numerics allow.** GEMM operands, K/V cache, the
   normalized activations and the attention output are fp16; accumulators,
   the residual stream, recurrent and convolution states are fp32. Use `h`
   literals in half expressions (a float literal promotes the whole
   expression).
7. **State traffic scales with sequences, not weights.** The Gated DeltaNet
   state (Hv × 128 × 128 fp32 per layer per sequence: 12.6 MB per layer for
   the 27B, 8.4 MB for the 35B-A3B) is read and written every step. At 16
   sequences that is 0.6 GB per step for the 27B, comparable to a fifth of
   its weights; fp16 storage or deferred commits are the levers.

## 4. Execution rules

1. **Occupancy first.** Threadgroups are multiples of 32; aim for 1K–2K
   resident threads per core; a dispatch should offer at least ~256
   threadgroups (the attention planner and split-K rules target that) or use
   persistent threadgroups that loop over work items. The baseline's worst
   offender is decode attention at a 512-token context: `plan_tiles` splits
   only when `P ≥ 2 × attn_split` (1024), so 4 KV heads × 1 tile = **4
   threadgroups** on 32 cores, 217 µs per layer, 3.5 ms per step.
2. **Registers set occupancy.** 128 registers per thread; fp32 accumulator
   tiles and unrolled fragment arrays are the usual budget. 64-row GEMM tiles
   spilled and lost 25%; `simdgroup_matrix` accumulators indexed by a runtime
   bound spill to memory (30× slower). Check the compiler statistics
   (Xcode GPU debugger, pipeline statistics) when a kernel is slower than its
   byte count predicts.
3. **Reduce in registers before memory.** `simd_sum` / `simd_shuffle_xor`
   inside a simdgroup, then one value per simdgroup through threadgroup
   memory, then (rarely) device atomics. The transpose-reduce in
   `deltanet_tr` ((R − 1) + (5 − log₂ R) shuffles for R row sums) is the
   pattern to reuse.
4. **Barriers are expensive in loops.** Attention's 16-key blocks cost three
   `threadgroup_barrier` per block; larger blocks do not fit as laid out.
   Count barriers per byte moved when designing a tile.
5. **Uniform branching is free, divergent branching is not.** Branch per
   threadgroup on a tile descriptor (expert tile with 1 row vs 8 rows, split
   vs full tile) — never per lane on data.
6. **Signed indices, no dynamically indexed stack arrays,** vector loads
   (`float4`/`half4`/`uint4`) over scalar ones; type conversions are free.
7. **Tail effects.** A dispatch of W waves loses part of its last wave;
   bandwidth-bound kernels lose more, because few outstanding requests do
   not saturate memory. Persistent GEMVs (grid = cores × resident
   threadgroups, an atomic work counter over column blocks) trade the tail
   for one atomic per block; measure it before adopting it, since the tail
   and the barriers share the same decode gap.

## 5. Dispatch rules

1. **Count dispatches per step; each one costs ~3–7 µs of boundary time.**
   The 27B step has 630 (10 per linear layer, 9–10 per full layer, head)
   and the "gap" between isolated and in-pass time turned out to be power
   throttling, not dispatch cost (see the measured facts below). Dispatch count still
   matters where steps are short: the 35B-A3B step (595 dispatches, 16.6 ms
   against a 6.7 ms floor) is dominated by latency-bound small kernels, and
   there fusion pays.
2. **Fuse in epilogues first.** Residual add, SwiGLU, norm statistics, gated
   output, argmax over a tile: all live in the GEMM/GEMV epilogue today. The
   next fusions are along the dependency chain of one layer (conv → gates →
   recurrence → gated norm; router → top-k → sort; attention prep → paged
   K/V write).
3. **Declare exact read/write spans** (`E.rd`, `E.wr`) for every dispatch,
   whole-buffer only when the kernel really touches everything. The hazard
   tracker turns them into barriers under the concurrent encoder; with exact
   spans, independent dispatches (state saves, two projection runs) overlap.
   Barrier bookkeeping is skipped under the serial encoder (default), so
   wrong spans hide until someone flips `ESPETERO_CONCURRENT=1`: keep them
   right anyway, and run the parity test in both modes.
4. **The encoder mode is a measurement, not a belief.** Serial dispatch was
   faster on an M1 Max (4–20% on prefill, equal on decode) than the
   concurrent encoder with tracked barriers; Metal 4 command queues and
   argument tables did not win either. Re-measure both on every new chip
   (M3+ dynamic caching, M5 tensor units change the balance).
5. **Command-buffer round trips cost 250–350 µs** and the CPU encodes a 630-
   dispatch step in 0.2 ms; both are hidden only by encoding step t+1 while
   step t runs (GPU-side token selection feeding the next embedding), which
   matters once a step is ~10 ms (MoE) rather than 77 ms.

### Measured facts: decode, memory model, launch limits

- **Sustained decode is power-limited, not dispatch-limited.** A 27B decode
  step runs at 61 ms (the isolated kernel sum) for the first ~0.4 s and
  climbs to 85–90 ms within half a second of back-to-back steps; the 4-bit
  `gate_up` GEMV falls from 343 GB/s isolated to ~210 GB/s when chained,
  the 8-bit LM head only to ~315 GB/s. Removing per-weight dequant
  instructions changed nothing; removing the CPU spin in `commit_wait` cut
  the sustained step 8% and prefill 3%: the package power budget is shared
  with the CPU. Judge decode kernels on the *sustained* number
  (`bench_native --sustain`, last-8 average) and on energy: a faster kernel
  that leaves cores idle (an attention split worth −2.7 ms isolated) does
  not move the sustained step.
- **Plain device stores are not visible to other threadgroups inside a
  dispatch, whatever fence you add.** `atomic_thread_fence(mem_device,
  seq_cst, device)` and `threadgroup_barrier(mem_device)` both left 300/300
  cross-threadgroup reads stale; relaxed **atomic stores and loads** were
  correct 300/300. Any last-threadgroup reduction (fused norm finalizer,
  future MoE combine) stores and loads the shared values atomically and
  signals with an atomic counter. Requires MSL 3.2 for the fence spelling.
- **Threadgroup size comes from the pipeline, not from the design.** A
  GEMV body with ~80 registers gets `maxTotalThreadsPerThreadgroup` of
  768–896 (`moe_route8`: 832, `rgemv8_1_store`: 896). A 1024-thread launch
  silently computes garbage (the fused route routed to wrong experts until
  the threadgroup was sized from `[pso maxTotalThreadsPerThreadgroup]`).
  Read the limit at dispatch time for any kernel that needs more than 256
  threads, and derive in-kernel work splits from `threads_per_threadgroup`.
- **Grids sized for a maximum do work for the maximum.** The grouped expert
  GEMVs launched 8224 / 16448 threadgroups per layer for 256 / 512 real
  tiles and ran at 112–153 GB/s; indirect dispatch with the exact count from
  the sort kernel is the fix. Threadgroup launch rate is a resource.
- **Consumer-side work in a GEMV loop costs more than the dispatch it
  saves.** Normalizing on load (a load and two multiplies per activation)
  was 12% slower per step on the 2B than a separate `rmsnorm`; the
  producer-side finalizer is 1% faster with 20% fewer dispatches.

### Measured facts: the MoE path (Qwen3.6-35B-A3B unless noted)

- **One-row tiles must not use the 8×8 MMA.** The grouped expert GEMV at
  batch 1 runs 8-row MMA tiles with one live row: 151 M MACs per gate/up
  layer (~28 µs of matrix-unit time) that are *not* hidden under the weight
  stream. The FMA GEMV (`moe_gemv<bits, MM, epi>`, `rgemv`'s loop with the
  tile table) is 14% faster per call and, with the prefetch removed, 30%:
  gate/up 84 → 60 µs, down 53 → 41 µs per layer (−1.9 ms per step).
- **Software-pipelining the K loop (next step's weights loaded before this
  step's FMAs) is slower here**: +1.3 ms per MoE step, no change on the
  dense 27B; the FMA expert kernel loses 17% with it. The extra
  registers cost occupancy, and the loop was not latency-bound in the way
  assumed. Compile with `ESPETERO_PF=1` to re-test; off by default.
- **Split-K with a last-arriver reduce loses at these sizes**: the
  2048×4096 out projection takes 46 µs split four ways versus 36 µs on 64
  threadgroups. The partial round trip and the finalizing threadgroup cost
  more than the idle cores. `gemv_splitk` is off by default.
- **Device-scope fences are not the cost of the last-arriver pattern.** The
  parallel route measured identically with and without
  `atomic_thread_fence(mem_device, seq_cst, device)` (40.7 vs 40.9 µs); the
  cost was the finalizing threadgroup's own work. Keep the fences (they are
  what the memory model requires) and make the finalizer small.
- **A rank-based top-k (each thread counts the entries above its own over
  E = 256 in threadgroup memory) is 2.5× slower than k rounds of
  `simd_max` + `simd_min` on the index**: 32 → 24 µs for the whole
  finalize kernel. Reductions through the SIMD intrinsics, not loops over
  threadgroup memory.
- **Indirect dispatch arguments, direct grids, expert layout locality and
  the number of simdgroups per threadgroup (2/4/8) of the expert GEMV made
  no measurable difference** (all within 1%). The expert GEMV streams at
  ~200 GB/s against 253 GB/s for a dense 6-bit GEMV with the same K: the
  remaining gap is not explained yet (tile → expert → address dependent
  loads before the stream starts are the next suspect).
- **The fused GDN decode step** replaces conv_prep + deltanet_tr8 +
  gated_rmsnorm (38.9 µs per linear layer at one sequence) with one
  32.9 µs kernel and 96 fewer dispatches per 27B step; the sustained step
  does not move (power-throttled regime). It runs one threadgroup per value
  head (Hv = 32 threadgroups on the 27B, 16 on the 2B): splitting a head
  over 2–4 threadgroups needs a cross-threadgroup gated norm.
- **Session-to-session numbers drift ±3–5% with the thermal state**; A/B
  inside one session, alternating, and confirm on whole-model runs.

## 6. Numerics and determinism

1. Operands fp16, accumulation fp32, residual stream fp32; dequantization is
   bit-identical to the host expansion (`dequant_hr` is the proof: dense and
   quantized paths give equal operands).
2. **Every reduction has a fixed order:** split-K partials summed in slot
   order, MoE experts added in slot order, attention partials merged by
   `attention_combine` in split order. No `atomic_fetch_add` on floats.
3. Parity tolerance against the reference engine: 2·10⁻² on candidate
   logits and log-sum-exp, argmax identical; the baseline measured 0 to
   2·10⁻³ across the ten parity cases (differences come only from summation
   order in split-K attention and the delta rule).
4. Softmax, log-sum-exp and gates in fp32 with the usual guards
   (`-INFINITY` masks, `m == -INFINITY ? 0 : exp(x - m)`, log-decay clamped
   at −80 in the chunked delta rule).

## 7. Correctness workflow for a new kernel

1. Write or keep the **reference path** (tier 0/1) and make it selectable
   (`set_option`, env). If the kernel replaces several dispatches, the
   reference is the old dispatch sequence.
2. **Parity**: `build/parity_test <model> <reference dylib>` on a small dense
   model (Qwen3.8-2B, 6 s to load) covers trees, checkpoints, in-place decode,
   forests, 4K contexts with split-K attention and MTP drafts. Extend the
   harness with an MLX-dumped logits reference for models the JEVfire dylib
   cannot run beside (a 27 GB MoE model does not fit twice in 64 GB). Add a
   case that toggles the new kernel against its reference inside one process.
3. **A/B in the same process, alternating**: `build/ab_native <model>
   <option> <a> <b> [rows] [iters] [decode_steps]` interleaves the two
   settings every iteration so clock and thermal drift hit both; report the
   medians it prints. Kernel-level changes without an option switch are A/B'd
   as two builds run alternately with the same command.
4. **Profile**: `bench_native --profile` runs every dispatch in its own
   command buffer and sums GPU time per kernel and GEMM shape. Read it as
   *isolated* kernel time: the in-pass step is longer by the dispatch gaps,
   and that difference is itself a result to track.
5. **Check the whole model**: build, parity, a dense-model benchmark and
   profile, and a load and decode of a MoE model. A kernel change is done
   when all of them pass and the numbers are written down with it.

## 8. Measurement protocol

- Make sure nothing else is using the GPU, then **warm the clock**:
  the GPU needs 20–30 ms of work after idle to reach full speed (the
  autotuner and bench do a warm-up pass).
- **Thermal drift is larger than most kernel effects** on a 14-inch M1
  Max: sustained prefill drops the clock within seconds; a 27B decode can
  fall from 14.7 to 11.5 tokens/s after three seconds.
  Alternate variants (A/B), take medians of short bursts, and never compare a
  run that started hot against one that started cold.
- Prefer **GPU time** (`GPUEndTime − GPUStartTime`) for kernels and passes,
  **wall time** for tokens/s as users see it; report both.
- Convert every kernel time to **GB/s** (bytes of weights + activations it
  must move) or **TFLOPS** (2·M·N·K) and compare with the table in §1. A
  number without its roofline share is not a result.
- Untracked buffers let two command buffers overlap: serialize timed passes.
- Isolated GEMM timings differ from in-pass timings (cold weights, other
  kernels, power state); validate tile choices on whole passes (the
  autotuner's 3% preference for taller tiles came from that).

## 9. Checklist before a kernel is merged

- [ ] Bytes read per pass equal the matrix (or tile) size once; no operand is
      re-read per threadgroup unless it is activations, and then in fp16.
- [ ] Loads are 16-byte per lane and contiguous across the simdgroup.
- [ ] Threadgroup = multiple of 32; ≥ ~256 threadgroups or persistent; the
      compiler reports no spills; threadgroup memory ≤ 32 KB.
- [ ] Reductions ordered; no float atomics; fp32 accumulation.
- [ ] Geometry as template parameters with a host-name instantiation list;
      quantization group and scale mode as function constants.
- [ ] `E.rd` / `E.wr` spans exact; hazard tracking correct under
      `ESPETERO_CONCURRENT=1`.
- [ ] Reference path kept and selectable; parity within 2·10⁻²; argmax equal.
- [ ] A/B medians (alternating) and a profile line with GB/s or TFLOPS and
      the roofline share; dispatch count per step before and after.
- [ ] Documented: what it fuses or replaces, why the layout, what it
      measured, in the kernel's header comment and the commit message.

## 10. Skeleton of a tier-0 kernel

```metal
// Function constants specialize the pipeline (one PSO per value); templates
// specialize code width and tile; runtime shapes go through a params struct.
constant int kGroup [[function_constant(0)]];       // codes per scale/bias
constant int kScaleMode [[function_constant(1)]];   // 0 affine, 1 Q6_K, 2 Q8_0, 3 Q4_K/Q5_K

struct MyParams { int M, N, K, lda, ldy; };

template <int BITS, int EPI>
kernel void my_kernel(device const half* A      [[buffer(0)]],
                      device const uint* Wr     [[buffer(1)]],   // fragment-ordered codes
                      device const half4* SB    [[buffer(2)]],   // scales/biases per K step
                      device const uint* Wh     [[buffer(3)]],   // high-bit plane (5/6-bit)
                      device void* Y            [[buffer(4)]],
                      constant MyParams& p      [[buffer(5)]],
                      uint3 tgid [[threadgroup_position_in_grid]],
                      uint sgid  [[simdgroup_index_in_threadgroup]],
                      uint lane  [[thread_index_in_simdgroup]]) {
  // 1. one threadgroup = one 32-column block; simdgroups split K
  // 2. per K step: two coalesced uint4 loads of codes per lane, scales via
  //    kstep_scales(), dequant_pair() (exponent-bias trick, one half2 FMA)
  // 3. accumulate fp32; reduce fm lanes with simd_shuffle_xor, then across
  //    simdgroups through threadgroup memory in a fixed order
  // 4. epilogue: store / residual add / SwiGLU / stats, vector stores
}
#define INSTANTIATE(PRE, BITS) \
  template [[host_name(PRE "_store")]] kernel void my_kernel<BITS, 0>(/* args */); \
  template [[host_name(PRE "_add")]]   kernel void my_kernel<BITS, 1>(/* args */);
INSTANTIATE("my4", 4) INSTANTIATE("my8", 8)
```

Host side (`gemm.h` style): pick the pipeline by name + `w.fc()`, bind, set
params with `E.bytes`, declare `E.rd(A, off, off + bytes)` and `E.wr(Y, …)`,
dispatch with `E.groups(...)`. Add the names to `gemm_pipelines()` so
`prewarm()` builds them at load.

## 11. Forward compatibility

- **Apple M5 (neural accelerators).** Metal 4 tensor ops (`mpp::matmul2d`
  over `cooperative_tensor`) run on dedicated matrix units on M5, 4–8× the
  GEMM rate; on M1–M4 they are emulated on the shader ALUs (fp8
  `matmul2d` has been measured at 0.94× fp16 on an M4 Max; MLX and llama.cpp found no gain
  on M2). Apple's guidance: prefill (compute-bound) gains up to 4×, decode
  (bandwidth-bound) ~25% from bandwidth/caches, and since macOS 26.3–26.4
  cooperative tensors accept in-kernel dequantized inputs and int4/int8
  tensors. Keep the GEMM dispatch interface (`gemm()`, `Mat`) format-neutral
  so an M5 prefill path is one more tile family behind `hgemm_tile`/`rgemm`
  selection, chosen by device family at load.
- **Metal 4 command API.** Measured as no win on M1 Max for many small
  dispatches and it loses `setBytes`; re-evaluate on M5 together with the
  tensor path.
- **Architectures.** Head dim, rotary width, conv width and delta-head dims
  as template parameters (§2); MLP kind (dense / MoE) and residual structure
  (plain / gated multi-branch) as layer-type flags in `Config`; attention
  kind (full / sparse-indexed / sliding) as a tile-table variant.
