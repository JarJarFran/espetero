// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ---------------------------------------------------------------------------
// Gated DeltaNet: causal depthwise conv (kernel 4) + SiLU, per-head q/k RMS
// normalisation and the scalar gates, one simdgroup per (row, head block).
// Head blocks: [0, Hk) q, [Hk, 2Hk) k, [2Hk, 2Hk + Hv) v (which also emit
// g = exp(-exp(A_log) * softplus(a + dt_bias)) and beta = sigmoid(b)).
// src[r*3 + j] is the row holding tap t-(j+1); negative -> parent state row
// (-1 - s) of a [3, C] conv state ordered oldest..newest.
// ---------------------------------------------------------------------------
struct ConvParams {
  int C;            // conv channels (2*Hk*Dk + Hv*Dv)
  int ldx;          // row stride of the fused projection
  int a_off;        // column of a within the fused projection (b follows at +Hv)
  int Hk;
  int Hv;
  float eps;
  ulong conv_base;  // float offset of this layer's conv-state region in the state pool
  ulong conv_stride;  // floats per slot in that region (3 * C)
};

// Forest-wide: rows of every tree in one dispatch. src[r*3 + j] is the
// absolute row holding tap t-(j+1), or -1 - s for row s of the parent's conv
// state; row_slot[r] names that parent slot (the all-zero slot for an empty
// context).
kernel void conv_prep(
    device const float* X [[buffer(0)]],
    device const float* pool [[buffer(1)]],
    device const int* src [[buffer(2)]],
    device const float* Wc [[buffer(3)]],
    device half* Qo [[buffer(4)]],
    device float* GB [[buffer(5)]],
    device const float* A_log [[buffer(6)]],
    device const float* dt_bias [[buffer(7)]],
    constant ConvParams& p [[buffer(8)]],
    device const int* row_slot [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int HD = 128;
  constexpr int NPT = HD / 32;
  const int head = tgpig.x * 4 + sg;
  const int r = tgpig.y;
  if (head >= 2 * p.Hk + p.Hv) return;
  const int c0 = head * HD + lane * NPT;
  int s[3];
  UNROLL for (int j = 0; j < 3; ++j) s[j] = src[r * 3 + j];
  device const float* state = pool + p.conv_base + ulong(row_slot[r]) * p.conv_stride;
  const float4 w0 = *((device const float4*)(Wc + (c0 + 0) * 4));
  const float4 w1 = *((device const float4*)(Wc + (c0 + 1) * 4));
  const float4 w2 = *((device const float4*)(Wc + (c0 + 2) * 4));
  const float4 w3 = *((device const float4*)(Wc + (c0 + 3) * 4));
  // taps: t (weight 3), t-1 (2), t-2 (1), t-3 (0)
  float4 acc = *((device const float4*)(X + ulong(r) * p.ldx + c0)) * float4(w0.w, w1.w, w2.w, w3.w);
  UNROLL for (int j = 0; j < 3; ++j) {
    const float4 v = (s[j] >= 0) ? *((device const float4*)(X + ulong(s[j]) * p.ldx + c0))
                                 : *((device const float4*)(state + ulong(-1 - s[j]) * p.C + c0));
    acc += v * float4(w0[2 - j], w1[2 - j], w2[2 - j], w3[2 - j]);
  }
  acc = acc / (1.0f + exp(-acc));
  if (head < 2 * p.Hk) {
    const float ss = simd_sum(dot(acc, acc));
    const float scale = head < p.Hk ? 1.0f / float(HD) : rsqrt(float(HD));
    acc *= rsqrt(ss / float(HD) + p.eps) * scale;
  } else if (lane == 0) {
    const int hv = head - 2 * p.Hk;
    const float x = X[ulong(r) * p.ldx + p.a_off + hv] + dt_bias[hv];
    const float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
    GB[ulong(r) * 2 * p.Hv + hv] = exp(-exp(A_log[hv]) * sp);
    GB[ulong(r) * 2 * p.Hv + p.Hv + hv] =
        1.0f / (1.0f + exp(-X[ulong(r) * p.ldx + p.a_off + p.Hv + hv]));
  }
  *((device half4*)(Qo + ulong(r) * p.C + c0)) = half4(acc);
}

// Conv state after the first T rows of a root starting at absolute row row0:
// reads the parent slot's state and writes the target slot's (they may be the
// same slot: each thread reads all three taps before writing). One job per
// (tree, saved slot); grid (C, jobs).
struct ConvSave {
  int row0;
  int T;
  int in_slot;
  int out_slot;
};

kernel void conv_state_save(
    device const float* X [[buffer(0)]],
    device float* pool [[buffer(1)]],
    device const ConvSave* jobs [[buffer(2)]],
    constant ConvParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x;
  if (c >= p.C) return;
  const ConvSave job = jobs[gid.y];
  device const float* src = pool + p.conv_base + ulong(job.in_slot) * p.conv_stride;
  device float* dst = pool + p.conv_base + ulong(job.out_slot) * p.conv_stride;
  float nv[3];
  UNROLL for (int j = 0; j < 3; ++j) {
    const int idx = job.T - 3 + j;
    nv[j] = (idx >= 0) ? X[ulong(job.row0 + idx) * p.ldx + c] : src[ulong(3 + idx) * p.C + c];
  }
  UNROLL for (int j = 0; j < 3; ++j) dst[ulong(j) * p.C + c] = nv[j];
}

// Gated delta rule over segments. Each segment is a run of consecutive rows
// (a tree's root, or one leaf) with its own initial state slot; roots write
// their final state (write = 1) and optionally the state after save_t rows
// (mid_slot), leaves only read. Each simdgroup owns R consecutive value rows
// of one head; lanes hold 4 key columns. grid (32, Dv / R, nseg * Hv),
// threadgroup (32, 4, 1).
struct DeltaSeg {
  int row0;
  int len;
  int in_slot;
  int out_slot;
  int mid_slot;
  int save_t;
  int write_state;
  int pad;
};

struct DeltaParams {
  int Hk;
  int Hv;
  int C;
  int tiled;  // value head v reads key head v % Hk (llama.cpp order), else v / (Hv / Hk)
  ulong st_base;    // float offset of this layer's state region in the pool
  ulong st_stride;  // floats per slot (Hv * Dv * Dk)
};

// Sum R per-lane partials across the simdgroup with halving exchanges:
// afterwards lane l holds the total of row l / (32 / R). Costs
// (R - 1) + (5 - log2 R) shuffles instead of 5 per row (R = 8: 9 vs 40).
template <int R>
static inline float transpose_reduce(thread float* v, ushort lane) {
  int count = R;
  ushort mask = 16;
  UNROLL for (int stage = 0; stage < 5; ++stage) {
    if (count > 1) {
      const int half_n = count / 2;
      const bool upper = (lane & mask) != 0;
      UNROLL for (int i = 0; i < R / 2; ++i) {
        if (i < half_n) {
          const float send = upper ? v[i] : v[i + half_n];
          const float keep = upper ? v[i + half_n] : v[i];
          v[i] = keep + simd_shuffle_xor(send, mask);
        }
      }
      count = half_n;
    } else {
      v[0] += simd_shuffle_xor(v[0], mask);
    }
    mask >>= 1;
  }
  return v[0];
}

template <int R>
kernel void deltanet_tr(
    device const half* Qo [[buffer(0)]],
    device const float* GB [[buffer(1)]],
    device float* pool [[buffer(2)]],
    device float* Yo [[buffer(4)]],
    device const DeltaSeg* segs [[buffer(5)]],
    constant DeltaParams& p [[buffer(6)]],
    uint3 tpig [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int Dk = 128;
  constexpr int Dv = 128;
  constexpr int LPR = 32 / R;  // lanes per row after the reduction
  const int dv0 = tpig.y * R;
  const int n = tpig.z;
  const int seg = n / p.Hv;
  const int hv = n % p.Hv;
  const int hk = p.tiled ? hv % p.Hk : hv / (p.Hv / p.Hk);
  const DeltaSeg sgm = segs[seg];
  const int k_off = p.Hk * Dk;
  const int v_off = 2 * p.Hk * Dk;
  const int my_row = lane / LPR;
  const ulong head_off = (ulong(hv) * Dv + dv0) * Dk + lane * 4;
  device float* state_in = pool + p.st_base + ulong(sgm.in_slot) * p.st_stride + head_off;

  float4 st[R];
  UNROLL for (int r = 0; r < R; ++r) st[r] = *((device const float4*)(state_in + ulong(r) * Dk));

  for (int t = 0; t < sgm.len; ++t) {
    const ulong row = ulong(sgm.row0 + t);
    device const half* xr = Qo + row * p.C;
    const float4 q = float4(*((device const half4*)(xr + hk * Dk + lane * 4)));
    const float4 k = float4(*((device const half4*)(xr + k_off + hk * Dk + lane * 4)));
    const float g = GB[row * 2 * p.Hv + hv];
    const float beta = GB[row * 2 * p.Hv + p.Hv + hv];
    const float v_mine = float(xr[v_off + hv * Dv + dv0 + my_row]);
    float part[R];
    UNROLL for (int r = 0; r < R; ++r) {
      st[r] *= g;
      part[r] = dot(st[r], k);
    }
    const float kv = transpose_reduce<R>(part, lane);
    const float delta_mine = (v_mine - kv) * beta;
    UNROLL for (int r = 0; r < R; ++r) {
      const float delta = simd_shuffle(delta_mine, ushort(r * LPR));
      st[r] += k * delta;
      part[r] = dot(st[r], q);
    }
    const float o = transpose_reduce<R>(part, lane);
    if (lane % LPR == 0) Yo[row * (p.Hv * Dv) + hv * Dv + dv0 + my_row] = o;
    if (t + 1 == sgm.save_t) {
      device float* mid = pool + p.st_base + ulong(sgm.mid_slot) * p.st_stride + head_off;
      UNROLL for (int r = 0; r < R; ++r) *((device float4*)(mid + ulong(r) * Dk)) = st[r];
    }
  }
  if (sgm.write_state) {
    device float* out = pool + p.st_base + ulong(sgm.out_slot) * p.st_stride + head_off;
    UNROLL for (int r = 0; r < R; ++r) *((device float4*)(out + ulong(r) * Dk)) = st[r];
  }
}

#define INSTANTIATE_DELTANET_TR(R)                                             \
  template [[host_name("deltanet_tr" #R)]] kernel void deltanet_tr<R>(       \
      device const half*, device const float*, device float*, device float*, \
      device const DeltaSeg*, constant DeltaParams&, uint3, ushort);

INSTANTIATE_DELTANET_TR(2)
INSTANTIATE_DELTANET_TR(4)
INSTANTIATE_DELTANET_TR(8)
INSTANTIATE_DELTANET_TR(16)

// Chunked gated delta rule: the same contract as deltanet_tr (segments with
// state slots), 8 tokens per step in the WY/UT form with fp32 8x8 MMAs. For a
// chunk with state S (before it), cumulative log-decay G_t and gates b_t:
//   L  = strictly-lower(b_t e^(G_t-G_i) k_t.k_i)      T = (I + L)^-1
//   D  = T diag(b) (V - diag(e^G) K S^T)               (the per-token deltas)
//   O  = diag(e^G) Q S^T + (e^(G_t-G_i) q_t.k_i, i<=t) D
//   S' = e^(G_C) S + D^T diag(e^(G_C-G_t)) K
// (I + L)^-1 = (I - L)(I + L^2)(I + L^4) exactly (L is nilpotent). Each
// simdgroup owns an 8-row slab of one head's state (16 fragments of S^T in
// registers) and walks the segment chunk by chunk: the per-token latency
// chain of the recurrence becomes one chunk step per 8 tokens.
// grid (32, Dv / 8, nseg * Hv), threadgroup (32, 4, 1).
kernel void deltanet_chunk(
    device const half* Qo [[buffer(0)]],
    device const float* GB [[buffer(1)]],
    device float* pool [[buffer(2)]],
    device float* Yo [[buffer(4)]],
    device const DeltaSeg* segs [[buffer(5)]],
    constant DeltaParams& p [[buffer(6)]],
    uint3 tpig [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int Dk = 128;
  constexpr int Dv = 128;
  constexpr int LDS = Dk + 4;  // padded rows of the staged chunk
  // The 4 simdgroups of a threadgroup are 4 slabs of the same head: the
  // chunk's K and Q (8 x 128) and gates are staged once for all of them.
  threadgroup float Ks[8 * LDS];
  threadgroup float Qs[8 * LDS];
  threadgroup float Gs[16];
  const int slab = tpig.y;
  const int n = tpig.z;
  const int seg = n / p.Hv;
  const int hv = n % p.Hv;
  const int hk = p.tiled ? hv % p.Hk : hv / (p.Hv / p.Hk);
  const DeltaSeg sgm = segs[seg];
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  const int k_off = p.Hk * Dk;
  const int v_off = 2 * p.Hk * Dk + hv * Dv + slab * 8;
  const ulong slab_off = (ulong(hv) * Dv + ulong(slab) * 8) * Dk;

  simdgroup_float8x8 ST[16];
  {
    device const float* src = pool + p.st_base + ulong(sgm.in_slot) * p.st_stride + slab_off;
    UNROLL for (int kk = 0; kk < 16; ++kk) simdgroup_load(ST[kk], src + kk * 8, Dk, ulong2(0, 0), true);
  }
  simdgroup_float8x8 I8;
  I8.thread_elements()[0] = fm == fn ? 1.0f : 0.0f;
  I8.thread_elements()[1] = fm == fn + 1 ? 1.0f : 0.0f;

  const int split = (sgm.save_t > 0 && sgm.save_t < sgm.len) ? sgm.save_t : sgm.len;
  for (int phase = 0; phase < 2; ++phase) {
    const int b0 = phase == 0 ? 0 : split;
    const int e0 = phase == 0 ? split : sgm.len;
    for (int t0 = b0; t0 < e0; t0 += 8) {
      const int nt = min(8, e0 - t0);
      const ulong base = ulong(sgm.row0 + t0);
      // Stage K and Q rows (zero past the segment) and the gates.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int c = tid; c < 8 * 32; c += 128) {
        const int r = c / 32, col = (c % 32) * 4;
        float4 kv = float4(0.0f), qv = float4(0.0f);
        if (r < nt) {
          device const half* xr = Qo + (base + r) * p.C;
          kv = float4(*((device const half4*)(xr + k_off + hk * Dk + col)));
          qv = float4(*((device const half4*)(xr + hk * Dk + col)));
        }
        *((threadgroup float4*)(Ks + r * LDS + col)) = kv;
        *((threadgroup float4*)(Qs + r * LDS + col)) = qv;
      }
      if (tid < 8) {
        const bool ok = int(tid) < nt;
        Gs[tid] = ok ? max(log(GB[(base + tid) * 2 * p.Hv + hv]), -80.0f) : 0.0f;
        Gs[8 + tid] = ok ? GB[(base + tid) * 2 * p.Hv + p.Hv + hv] : 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float acc = 0.0f, g_r = 0.0f, g_c0 = 0.0f, g_c1 = 0.0f;
      UNROLL for (int t = 0; t < 8; ++t) {
        acc += Gs[t];
        if (t == fm) g_r = acc;
        if (t == fn) g_c0 = acc;
        if (t == fn + 1) g_c1 = acc;
      }
      const float g_last = acc;
      const float b_r = Gs[8 + fm];
      const bool r_ok = fm < nt;
      simdgroup_float8x8 KK = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      simdgroup_float8x8 QK = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      simdgroup_float8x8 WS = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      simdgroup_float8x8 QS = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      UNROLL for (int kk = 0; kk < 16; ++kk) {
        simdgroup_float8x8 Kf, Kt, Qf;
        simdgroup_load(Kf, Ks + kk * 8, LDS);
        simdgroup_load(Kt, Ks + kk * 8, LDS, ulong2(0, 0), true);
        simdgroup_load(Qf, Qs + kk * 8, LDS);
        simdgroup_multiply_accumulate(KK, Kf, Kt, KK);
        simdgroup_multiply_accumulate(QK, Qf, Kt, QK);
        simdgroup_multiply_accumulate(WS, Kf, ST[kk], WS);
        simdgroup_multiply_accumulate(QS, Qf, ST[kk], QS);
      }
      simdgroup_float8x8 L, T, P, tmp;
      L.thread_elements()[0] = fn < fm ? b_r * exp(g_r - g_c0) * KK.thread_elements()[0] : 0.0f;
      L.thread_elements()[1] = fn + 1 < fm ? b_r * exp(g_r - g_c1) * KK.thread_elements()[1] : 0.0f;
      T.thread_elements()[0] = I8.thread_elements()[0] - L.thread_elements()[0];
      T.thread_elements()[1] = I8.thread_elements()[1] - L.thread_elements()[1];
      P = L;
      UNROLL for (int step = 0; step < 2; ++step) {
        simdgroup_multiply(tmp, P, P);
        P = tmp;
        simdgroup_multiply_accumulate(tmp, T, P, T);
        T = tmp;
      }
      simdgroup_float8x8 X, D;
      {
        device const half* qrow = Qo + (base + fm) * p.C;
        const float2 vv = r_ok ? float2(*((device const half2*)(qrow + v_off + fn))) : float2(0.0f);
        const float eg = exp(g_r);
        X.thread_elements()[0] = b_r * (vv.x - eg * WS.thread_elements()[0]);
        X.thread_elements()[1] = b_r * (vv.y - eg * WS.thread_elements()[1]);
      }
      simdgroup_multiply(D, T, X);
      simdgroup_float8x8 QKm, O;
      QKm.thread_elements()[0] = fn <= fm ? exp(g_r - g_c0) * QK.thread_elements()[0] : 0.0f;
      QKm.thread_elements()[1] = fn + 1 <= fm ? exp(g_r - g_c1) * QK.thread_elements()[1] : 0.0f;
      {
        const float eg = exp(g_r);
        O.thread_elements()[0] = eg * QS.thread_elements()[0];
        O.thread_elements()[1] = eg * QS.thread_elements()[1];
      }
      simdgroup_multiply_accumulate(O, QKm, D, O);
      if (r_ok) {
        device float* y = Yo + (base + fm) * (p.Hv * Dv) + hv * Dv + slab * 8 + fn;
        *((device float2*)y) = float2(O.thread_elements()[0], O.thread_elements()[1]);
      }
      simdgroup_float8x8 W;
      {
        const float w = r_ok ? exp(g_last - g_r) : 0.0f;
        W.thread_elements()[0] = w * D.thread_elements()[0];
        W.thread_elements()[1] = w * D.thread_elements()[1];
      }
      const float ec = exp(g_last);
      UNROLL for (int kk = 0; kk < 16; ++kk) {
        simdgroup_float8x8 Kt;
        simdgroup_load(Kt, Ks + kk * 8, LDS, ulong2(0, 0), true);
        ST[kk].thread_elements()[0] *= ec;
        ST[kk].thread_elements()[1] *= ec;
        simdgroup_multiply_accumulate(ST[kk], Kt, W, ST[kk]);
      }
    }
    if (phase == 0 && sgm.save_t > 0 && sgm.save_t <= sgm.len) {
      device float* mid = pool + p.st_base + ulong(sgm.mid_slot) * p.st_stride + slab_off;
      UNROLL for (int kk = 0; kk < 16; ++kk) simdgroup_store(ST[kk], mid + kk * 8, Dk, ulong2(0, 0), true);
    }
    if (split >= sgm.len) break;
  }
  if (sgm.write_state) {
    device float* out = pool + p.st_base + ulong(sgm.out_slot) * p.st_stride + slab_off;
    UNROLL for (int kk = 0; kk < 16; ++kk) simdgroup_store(ST[kk], out + kk * 8, Dk, ulong2(0, 0), true);
  }
}

// out = rms_norm(y) * w * silu(z) -> fp16, per (row, value head).
kernel void gated_rmsnorm(
    device const float* Yin [[buffer(0)]],
    device const float* Zg [[buffer(1)]],
    device const float* Wn [[buffer(2)]],
    device half* Out [[buffer(3)]],
    constant int& width [[buffer(4)]],
    constant int& z_ld [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int Dv = 128;
  const int head = tgpig.x * 4 + sg;
  const ulong row = tgpig.y;
  if (head * Dv >= width) return;
  const float4 v = *((device const float4*)(Yin + row * width + head * Dv + lane * 4));
  const float4 z = *((device const float4*)(Zg + row * z_ld + head * Dv + lane * 4));
  const float4 w = *((device const float4*)(Wn + lane * 4));
  const float ss = simd_sum(dot(v, v));
  const float inv = rsqrt(ss / float(Dv) + eps);
  *((device half4*)(Out + row * width + head * Dv + lane * 4)) =
      half4(v * inv * w * (z / (1.0f + exp(-z))));
}

// ---------------------------------------------------------------------------
// Fused Gated DeltaNet decode step (tier 2): for GEMV-sized passes, one
// threadgroup per (segment, value head) does what conv_prep, deltanet_tr
// and gated_rmsnorm do in three dispatches: the causal conv + SiLU + q/k
// norm + gates for its key/value head, the per-token recurrence over
// Dv/R simdgroups x R state rows (the whole 128-row head), the gated
// RMSNorm over the head's outputs reduced in threadgroup memory, fp16 yn,
// and the recurrent state saves. R = 8 (512 threads) hides the per-step
// latency better for one or two sequences, R = 16 (256 threads) for
// larger passes, mirroring deltanet_tr<R>. The conv state save is a
// separate dispatch after this one, since inside one dispatch a saving
// threadgroup could overwrite tap channels that a sibling value head still
// reads (in-place decode), except with conv_save (root passes without
// leaves, save slots read by no other segment): each threadgroup copies its
// value head's new taps after its token loop, and the last threadgroup of
// each key head to finish (a self-resetting arrival counter per segment and
// key head) copies the q/k taps.
// grid (32*Dv/R, 1, nseg * Hv), threadgroup (32*Dv/R, 1, 1).
// ---------------------------------------------------------------------------

struct DeltaFusedParams {
  int Hk;
  int Hv;
  int C;            // conv channels (2*Hk*Dk + Hv*Dv)
  int ldx;          // row stride of the fused projection X
  int a_off;        // column of a in X (b at + Hv)
  int tiled;
  float eps;
  int z_off;        // column of z in X (Vd wide)
  int yn_ld;        // row stride of yn (Vd)
  ulong conv_base;
  ulong conv_stride;
  ulong st_base;
  ulong st_stride;
  int conv_save;    // segment i's conv save job is cjobs[i] (T = 0: none)
};

// Conv tap: rows >= 0 index the fused projection X, negative ones the saved
// conv state (-1 - s taps back).
static inline float gdn_tap(device const float* X, device const float* cstate, int s, int c, int ldx, int C) {
  return s >= 0 ? X[ulong(s) * ldx + c] : cstate[ulong(-1 - s) * C + c];
}

template <int R>
kernel void gdn_decode_fused_impl(
    device const float* X [[buffer(0)]],
    device float* pool [[buffer(1)]],
    device const int* src [[buffer(2)]],
    device const float* Wc [[buffer(3)]],
    device const int* row_slot [[buffer(4)]],
    device const float* A_log [[buffer(5)]],
    device const float* dt_bias [[buffer(6)]],
    device const float* Wn [[buffer(7)]],
    device half* Yn [[buffer(8)]],
    device const DeltaSeg* segs [[buffer(9)]],
    constant DeltaFusedParams& p [[buffer(10)]],
    device const ConvSave* cjobs [[buffer(11)]],  // conv_save: one per segment
    device atomic_int* cctr [[buffer(12)]],       // conv_save: [segment][key head] arrivals
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int Dk = 128, Dv = 128, LPR = 32 / R;
  constexpr int NSG = Dv / R;   // simdgroups per threadgroup: 8 (R=16) or 16 (R=8)
  constexpr int NT = 32 * NSG;  // 256 or 512 threads
  threadgroup float qs[Dk], ks[Dk], vs[Dv], os[Dv], gs[2];
  threadgroup float red[NSG];
  const int sg = int(tid) / 32;   // this simdgroup's R state rows start at sg * R
  const int n = int(tgpig.z);
  const int seg = n / p.Hv;
  const int hv = n % p.Hv;
  const int hk = p.tiled ? hv % p.Hk : hv / (p.Hv / p.Hk);
  const DeltaSeg sgm = segs[seg];
  const int k_off = p.Hk * Dk;
  const int v_off = 2 * p.Hk * Dk;
  const int dv0 = sg * R;
  const int my_row = int(lane) / LPR;
  const ulong head_off = (ulong(hv) * Dv + dv0) * Dk + lane * 4;
  device const float* state_in = pool + p.st_base + ulong(sgm.in_slot) * p.st_stride + head_off;
  float4 st[R];
  UNROLL for (int r = 0; r < R; ++r) st[r] = *((device const float4*)(state_in + ulong(r) * Dk));
  // conv channels: q on threads 0..127, k on 128..255, v on 256..383 when the
  // threadgroup has them (R=8), else on the q threads as a second channel.
  const bool qk_thread = tid < 256;
  const bool v_thread = NT >= 384 ? (tid >= 256 && tid < 384) : tid < 128;
  const int cqk = (tid < 128) ? hk * Dk + int(tid) : k_off + hk * Dk + int(tid & 127);
  const int cv = v_off + hv * Dv + int(tid & 127);
  const float4 wqk = qk_thread ? *((device const float4*)(Wc + cqk * 4)) : float4(0.0f);
  const float4 wv = v_thread ? *((device const float4*)(Wc + cv * 4)) : float4(0.0f);
  device const float* cstate = pool + p.conv_base + ulong(row_slot[sgm.row0]) * p.conv_stride;

  for (int t = 0; t < sgm.len; ++t) {
    const int row = sgm.row0 + t;
    const int s0 = src[row * 3 + 0], s1 = src[row * 3 + 1], s2 = src[row * 3 + 2];
    // Causal conv + SiLU for the thread's channels.
    float aq = 0.0f;
    if (qk_thread) {
      aq = X[ulong(row) * p.ldx + cqk] * wqk.w + gdn_tap(X, cstate, s0, cqk, p.ldx, p.C) * wqk.z +
           gdn_tap(X, cstate, s1, cqk, p.ldx, p.C) * wqk.y + gdn_tap(X, cstate, s2, cqk, p.ldx, p.C) * wqk.x;
      aq = aq / (1.0f + exp(-aq));
    }
    float av = 0.0f;
    if (v_thread) {
      av = X[ulong(row) * p.ldx + cv] * wv.w + gdn_tap(X, cstate, s0, cv, p.ldx, p.C) * wv.z +
           gdn_tap(X, cstate, s1, cv, p.ldx, p.C) * wv.y + gdn_tap(X, cstate, s2, cv, p.ldx, p.C) * wv.x;
      av = av / (1.0f + exp(-av));
    }
    // q/k RMS norm over the head: simdgroups 0..3 hold q, 4..7 hold k.
    const float ss = simd_sum(aq * aq);
    if (lane == 0) red[sg] = ss;
    if (tid == 0) {
      const float x = X[ulong(row) * p.ldx + p.a_off + hv] + dt_bias[hv];
      const float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
      gs[0] = exp(-exp(A_log[hv]) * sp);
      gs[1] = 1.0f / (1.0f + exp(-X[ulong(row) * p.ldx + p.a_off + p.Hv + hv]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (qk_thread) {
      const float ssq = (tid < 128) ? red[0] + red[1] + red[2] + red[3] : red[4] + red[5] + red[6] + red[7];
      const float scale = (tid < 128) ? 1.0f / float(Dk) : rsqrt(float(Dk));
      const float normed = aq * rsqrt(ssq / float(Dk) + p.eps) * scale;
      if (tid < 128) qs[tid] = normed; else ks[tid - 128] = normed;
    }
    if (v_thread) vs[tid & 127] = av;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Recurrence for this simdgroup's R rows.
    const float4 q = *((threadgroup const float4*)(qs + lane * 4));
    const float4 k = *((threadgroup const float4*)(ks + lane * 4));
    const float g = gs[0], beta = gs[1];
    const float v_mine = vs[dv0 + my_row];
    float part[R];
    UNROLL for (int r = 0; r < R; ++r) {
      st[r] *= g;
      part[r] = dot(st[r], k);
    }
    const float kv = transpose_reduce<R>(part, lane);
    const float delta_mine = (v_mine - kv) * beta;
    UNROLL for (int r = 0; r < R; ++r) {
      const float delta = simd_shuffle(delta_mine, ushort(r * LPR));
      st[r] += k * delta;
      part[r] = dot(st[r], q);
    }
    const float o = transpose_reduce<R>(part, lane);
    if (lane % LPR == 0) os[dv0 + my_row] = o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Gated RMSNorm over the head's 128 outputs -> fp16 yn (simdgroups 0..3).
    if (tid < 128) {
      const float ov = os[tid];
      const float s2n = simd_sum(ov * ov);
      if (lane == 0) red[sg] = s2n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128) {
      const float inv = rsqrt((red[0] + red[1] + red[2] + red[3]) / float(Dv) + p.eps);
      const float z = X[ulong(row) * p.ldx + p.z_off + hv * Dv + tid];
      Yn[ulong(row) * p.yn_ld + hv * Dv + tid] = half(os[tid] * inv * Wn[tid] * (z / (1.0f + exp(-z))));
    }
    if (t + 1 == sgm.save_t) {
      device float* mid = pool + p.st_base + ulong(sgm.mid_slot) * p.st_stride + head_off;
      UNROLL for (int r = 0; r < R; ++r) *((device float4*)(mid + ulong(r) * Dk)) = st[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // qs/ks/vs/os/red reused by the next token
  }
  if (sgm.write_state) {
    device float* out = pool + p.st_base + ulong(sgm.out_slot) * p.st_stride + head_off;
    UNROLL for (int r = 0; r < R; ++r) *((device float4*)(out + ulong(r) * Dk)) = st[r];
  }
  if (p.conv_save) {
    const ConvSave job = cjobs[seg];
    if (job.T > 0) {
      // Every tap read of this threadgroup is done (the token loop ends on a barrier).
      device const float* cin = pool + p.conv_base + ulong(job.in_slot) * p.conv_stride;
      device float* cout = pool + p.conv_base + ulong(job.out_slot) * p.conv_stride;
      auto save = [&](int c) {  // conv_state_save for channel c
        float nv[3];
        UNROLL for (int j = 0; j < 3; ++j) {
          const int idx = job.T - 3 + j;
          nv[j] = (idx >= 0) ? X[ulong(job.row0 + idx) * p.ldx + c] : cin[ulong(3 + idx) * p.C + c];
        }
        UNROLL for (int j = 0; j < 3; ++j) cout[ulong(j) * p.C + c] = nv[j];
      };
      // The value head's channels: read by this threadgroup only.
      if (tid < 128) save(v_off + hv * Dv + int(tid));
      // The key head's q/k channels: its last threadgroup, after every
      // sibling value head has read them.
      threadgroup int last;
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0) last = atomic_fetch_add_explicit(cctr + seg * p.Hk + hk, 1, memory_order_relaxed) == p.Hv / p.Hk - 1;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (last) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
        if (tid < 256) save(tid < 128 ? hk * Dk + int(tid) : k_off + hk * Dk + int(tid - 128));
        if (tid == 0) atomic_store_explicit(cctr + seg * p.Hk + hk, 0, memory_order_relaxed);
      }
    }
  }
}

template [[host_name("gdn_decode_fused_tr8")]] kernel void gdn_decode_fused_impl<8>(
    device const float*, device float*, device const int*, device const float*, device const int*,
    device const float*, device const float*, device const float*, device half*, device const DeltaSeg*,
    constant DeltaFusedParams&, device const ConvSave*, device atomic_int*, uint3, uint, ushort);
template [[host_name("gdn_decode_fused_tr16")]] kernel void gdn_decode_fused_impl<16>(
    device const float*, device float*, device const int*, device const float*, device const int*,
    device const float*, device const float*, device const float*, device half*, device const DeltaSeg*,
    constant DeltaFusedParams&, device const ConvSave*, device atomic_int*, uint3, uint, ushort);
