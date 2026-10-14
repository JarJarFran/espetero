// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Gemma 4 vision tower (gemma4_vision) and its projection into the language
// model (embed_vision), with the semantics of mlx-vlm's models/gemma4/vision.py
// and transformers' Gemma4VisionModel (see THIRD_PARTY_NOTICES.md). Per image
// of P patches (rows in raster order of a gh x gw patch grid):
//
//   x  = input_proj(2 * (pixels - 0.5)) + pos[0][column] + pos[1][row]
//   per block:  x += rmsnorm(o_proj(attention(q, k, v))) * post_attention_layernorm
//               x += rmsnorm(down(gelu(gate(x2)) * up(x2))) * post_feedforward_layernorm,
//               x2 = rmsnorm(x) * pre_feedforward_layernorm
//     q, k = rope2d(rmsnorm(q_proj(x1)) * q_norm), rope2d(rmsnorm(k_proj(x1)) * k_norm),
//     v = rmsnorm(v_proj(x1)), x1 = rmsnorm(x) * input_layernorm; scale 1,
//     every patch of the image attends to every other (not across images)
//   soft token = mean of a pool x pool block of x * sqrt(D), then
//               (t - std_bias) * std_scale (standardize), raster order over the pooled grid
//   embedding  = embedding_projection(rmsnorm(soft token))  (no weight)
//
// The 2D RoPE rotates head dims [0, hd/2) by the patch column and
// [hd/2, hd) by the row, each half as rotate_half pairs (i, i + hd/4) at
// theta^(-4i / hd). Weights of the tower are fp16 (dense GEMMs, fp32
// accumulation, fp32 residual); the projection keeps the checkpoint's format.
// The embeddings overwrite the soft-token rows of the pass after the
// embedding scale (they are not multiplied by sqrt(D), as the reference
// scatters them into the scaled embeddings). The loader is
// load_gemma_vision (gemma_load.h).
#pragma once

namespace espetero::native {

// ---------------------------------------------------------------- pass
// Images of a pass start at patch rows that are multiples of this (the
// attention's 64-row query tiles: vit_attention2 aligns its key blocks to the
// first row of a tile), so each image's rows are computed exactly as when it
// is alone in the pass, whatever else the pass holds.
constexpr int kG4vAlign = 64;

// Per-patch and per-token metadata of the pass's images: patch (column, row),
// the image's patch range, and per soft token its pooling record
// (first patch, grid width, pooled row, pooled column) and decoder row;
// pixels -> fp16 of the reference's bf16 input 2 * (x - 0.5). The padding
// rows between images are zero and form segments of their own.
inline void prepare_images_g4(Model& m, const std::vector<ImageJob>& jobs) {
  VScratch& v = m.vs;
  const VisionConfig& c = m.vis.c;
  const int kp = m.vis.patch.kp, pd = c.patch_dim(), k = c.merge;
  int32_t* hw = static_cast<int32_t*>([v.hw contents]);
  int32_t* seg = static_cast<int32_t*>([v.seg contents]);
  int32_t* rec = static_cast<int32_t*>([v.pidx contents]);
  int32_t* dst = static_cast<int32_t*>([v.dst_rows contents]);
  __fp16* pix = static_cast<__fp16*>([v.pix contents]);
  int end = 0;
  for (const ImageJob& J : jobs) {
    for (int q = end; q < J.patch0; ++q) {
      hw[2 * q + 0] = hw[2 * q + 1] = 0;
      seg[2 * q + 0] = end;
      seg[2 * q + 1] = J.patch0;
    }
    if (J.patch0 > end) std::memset(pix + size_t(end) * size_t(kp), 0, size_t(J.patch0 - end) * size_t(kp) * 2);
    end = J.patch0 + J.n_patch;
  }
  for (const ImageJob& J : jobs) {
    const int gw = J.im->grid_w;
    for (int r = 0; r < J.n_patch; ++r) {
      const int q = J.patch0 + r;
      hw[2 * q + 0] = r % gw;
      hw[2 * q + 1] = r / gw;
      seg[2 * q + 0] = J.patch0;
      seg[2 * q + 1] = J.patch0 + J.n_patch;
    }
    for (int t = 0; t < J.n_tok; ++t) {
      const int u = J.tok0 + t;
      rec[4 * u + 0] = J.patch0;
      rec[4 * u + 1] = gw;
      rec[4 * u + 2] = t / (gw / k);
      rec[4 * u + 3] = t % (gw / k);
      dst[u] = J.dst_row0 + t;
    }
    const float* src = J.im->pixels;
    const int p0 = J.patch0;
    parallel_for(size_t(J.n_patch), [&](size_t q) {
      __fp16* d = pix + size_t(p0 + int(q)) * size_t(kp);
      const float* sp = src + q * size_t(pd);
      for (int e = 0; e < pd; ++e) {
        const float x = sp[e] - 0.5f;
        d[e] = __fp16(bf16_round(2.0f * x));
      }
      for (int e = pd; e < kp; ++e) d[e] = __fp16(0.0f);
    });
  }
}

struct G4vPrepParamsHost {
  int32_t N, H;
  float eps;
  int32_t pad;
};
struct G4vPoolParamsHost {
  int32_t D, k;
  float scale, eps;
  int32_t standardize, pad;
};

// The tower over every image of the pass (P patch rows, n_tok soft tokens);
// the projected embeddings overwrite their rows of `h`.
inline void encode_vision_g4(Model& m, Enc& E, int P, int n_tok, id<MTLBuffer> h) {
  const VisionConfig& c = m.vis.c;
  VScratch& v = m.vs;
  const int D = c.D, Ip = c.Ip();
  auto rmsnorm = [&](id<MTLBuffer> X, const Buf& w, id<MTLBuffer> Y, int rows) {
    E.pipe("rmsnorm");
    E.set(0, X);
    E.set(1, w);
    E.set(2, Y);
    E.bytes(3, D);
    E.bytes(4, c.eps);
    const int g = 0;
    E.set(5, X);
    E.bytes(6, g);
    E.rd(X);
    E.wr(Y);
    E.groups(MTLSizeMake(size_t(rows), 1, 1), MTLSizeMake(256, 1, 1));
  };
  gemm_h(E, m.vis.patch, v.pix, 0, m.vis.patch.kp, v.x, 0, D, P, 0);
  {
    const int32_t dp[2] = {D, c.n_pos};
    E.pipe("g4v_pos_add");
    E.set(0, v.x);
    E.set(1, m.vis.pos);
    E.set(2, v.hw);
    E.bytes(3, dp);
    E.rd(v.x);
    E.wr(v.x);
    E.threads(MTLSizeMake(size_t(D / 4), size_t(P), 1), MTLSizeMake(64, 1, 1));
  }
  // Long passes (several photos) commit every block on its own, as the
  // Qwen tower does (encode_vision): same work and order.
  static const int flush_from = [] {
    const char* e = std::getenv("ESPETERO_VIT_FLUSH");
    return e && *e ? std::atoi(e) : 2048;
  }();
  const bool flush_every = flush_from > 0 && P >= flush_from;
  std::snprintf(E.phase, sizeof(E.phase), "vision, %d patch rows", P);
  const std::string attn = "vit_attention2_" + std::to_string(c.hd);
  const G4vPrepParamsHost pp{P, c.heads, c.eps, 0};
  const struct {
    int N, H;
  } at{P, c.heads};
  rmsnorm(v.x, m.vis.blocks[0].n1w, v.xn, P);
  for (size_t bi = 0; bi < m.vis.blocks.size(); ++bi) {
    const VBlock& B = m.vis.blocks[bi];
    if (flush_every && bi > 0) E.flush();
    gemm_h(E, B.qkv, v.xn, 0, D, v.qkv, 0, 3 * D, P, 0);
    E.pipe("g4v_qkv_prep" + std::to_string(c.hd));
    E.set(0, v.qkv);
    E.set(1, v.hw);
    E.set(2, m.vis.inv_freq);
    E.set(3, B.qn);
    E.set(4, B.kn);
    E.set(5, v.q);
    E.set(6, v.k);
    E.set(7, v.v);
    E.bytes(8, pp);
    E.rd(v.qkv);
    E.wr(v.q);
    E.wr(v.k);
    E.wr(v.v);
    E.groups(MTLSizeMake(size_t(c.heads + 3) / 4, size_t(P), 1), MTLSizeMake(128, 1, 1));
    E.pipe(attn);
    E.set(0, v.q);
    E.set(1, v.k);
    E.set(2, v.v);
    E.set(3, v.seg);
    E.set(4, v.o);
    E.bytes(5, at);
    E.rd(v.q);
    E.rd(v.k);
    E.rd(v.v);
    E.wr(v.o);
    E.groups(MTLSizeMake(size_t(c.heads), size_t(P + 63) / 64, 1), MTLSizeMake(256, 1, 1));
    gemm_h(E, B.proj, v.o, 0, D, v.qkv, 0, D, P, 0);
    {
      const G4NormParamsHost np{D, c.eps, 1.0f, 1};
      E.pipe("g4_post_attn");
      E.set(0, v.qkv);
      E.set(1, v.x);
      E.set(2, B.post_attn);
      E.set(3, B.n2w);
      E.set(4, B.n2w);
      E.set(5, B.n2w);
      E.set(6, v.xn);
      E.set(7, v.xn);
      E.set(8, v.xn);
      E.bytes(9, np);
      E.rd(v.qkv);
      E.rd(v.x);
      E.wr(v.x);
      E.wr(v.xn);
      E.groups(MTLSizeMake(size_t(P), 1, 1), MTLSizeMake(256, 1, 1));
    }
    gemm_h(E, B.fc1, v.xn, 0, D, v.act, 0, Ip, P, 5);
    gemm_h(E, B.fc2, v.act, 0, Ip, v.qkv, 0, D, P, 0);
    {
      const bool next = bi + 1 < m.vis.blocks.size();
      const G4NormParamsHost np{D, c.eps, 1.0f, next ? 1 : 0};
      E.pipe("g4_post_ffn");
      E.set(0, v.qkv);
      E.set(1, v.x);
      E.set(2, B.post_ffn);
      E.set(3, next ? m.vis.blocks[bi + 1].n1w : B.post_ffn);
      E.set(4, v.xn);
      E.bytes(5, np);
      E.rd(v.qkv);
      E.rd(v.x);
      E.wr(v.x);
      if (next) E.wr(v.xn);
      E.groups(MTLSizeMake(size_t(P), 1, 1), MTLSizeMake(256, 1, 1));
    }
  }
  {
    const G4vPoolParamsHost pool{D, c.merge, float(std::sqrt(double(D))), c.eps, c.standardize ? 1 : 0, 0};
    E.pipe("g4v_pool");
    E.set(0, v.x);
    E.set(1, v.pidx);
    E.set(2, c.standardize ? m.vis.std_bias : m.vis.pos);
    E.set(3, c.standardize ? m.vis.std_scale : m.vis.pos);
    E.set(4, v.mact);
    E.bytes(5, pool);
    E.rd(v.x);
    E.wr(v.mact);
    E.groups(MTLSizeMake(size_t(n_tok), 1, 1), MTLSizeMake(256, 1, 1));
  }
  gemm(E, m.tune, m.vis.proj, v.mact, 0, D, v.out, 0, c.out, n_tok, 0);
  E.pipe("scatter_rows_f32");
  E.set(0, v.out);
  E.set(1, h);
  E.set(2, v.dst_rows);
  E.bytes(3, c.out);
  E.rd(v.out);
  E.wr(h);
  E.threads(MTLSizeMake(size_t(c.out / 4), size_t(n_tok), 1), MTLSizeMake(64, 1, 1));
  if (flush_every) E.flush();  // the language model's layers start a new command buffer
  E.phase[0] = 0;
}

}  // namespace espetero::native
