// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Qwen3-VL vision tower: host-side patch metadata and the GPU encode (patch
// embedding, interpolated position embedding, 2D-rotary blocks, 2x2 merger).
#pragma once

#include "../api.h"
#include "model.h"

namespace espetero::native {

inline void layernorm(Enc& E, id<MTLBuffer> X, Buf w, Buf b, id<MTLBuffer> Y, int rows, int D, float eps) {
  E.pipe("layernorm");
  E.set(0, X);
  E.set(1, w);
  E.set(2, b);
  E.set(3, Y);
  E.bytes(4, D);
  E.bytes(5, eps);
  E.rd(X);
  E.wr(Y);
  E.groups(MTLSizeMake(size_t(rows), 1, 1), MTLSizeMake(256, 1, 1));
}

inline void ensure_vscratch(Model& m, int rows) {
  VScratch& v = m.vs;
  if (rows <= v.rows) return;
  const VisionConfig& c = m.vis.c;
  const int R = std::max(rows, std::max(1024, v.rows * 3 / 2));
  v.pix = m.buffer(size_t(R) * size_t(m.vis.patch.kp) * 2);
  v.x = m.buffer(size_t(R) * size_t(c.D) * 4);
  v.xn = m.buffer(size_t(R) * size_t(c.D) * 2);
  v.qkv = m.buffer(size_t(R) * 3 * size_t(c.D) * 4);
  v.q = m.buffer(size_t(R) * size_t(c.D) * 2);
  v.k = m.buffer(size_t(R) * size_t(c.D) * 2);
  v.v = m.buffer(size_t(R) * size_t(c.D) * 2);
  v.o = m.buffer(size_t(R) * size_t(c.D) * 2);
  v.act = m.buffer(size_t(R) * size_t(c.Ip()) * 2);
  std::memset([v.act contents], 0, [v.act length]);  // K padding columns stay zero
  v.mact = m.buffer(size_t(R / 4) * 4 * size_t(c.D) * 2);
  v.out = m.buffer(size_t(R / 4) * size_t(c.out) * 4);
  v.pidx = m.buffer(size_t(R) * 16);
  v.pw = m.buffer(size_t(R) * 16);
  v.hw = m.buffer(size_t(R) * 8);
  v.seg = m.buffer(size_t(R) * 8);
  v.dst_rows = m.buffer(size_t(R / 4) * 4);
  v.rows = R;
}

// One image of a pass: its patch rows in the vision batch and the decoder rows
// its merged tokens replace.
struct ImageJob {
  const ImageRef* im = nullptr;
  int patch0 = 0, n_patch = 0, dst_row0 = 0;
};

// Per-patch metadata in the processor's merge-window order: bilinear taps of
// the learned position table (weights rounded to bf16 like the reference),
// 2D rotary coordinates and the image segment; pixels -> fp16.
inline void prepare_images(Model& m, const std::vector<ImageJob>& jobs) {
  const VisionConfig& c = m.vis.c;
  VScratch& v = m.vs;
  const int kp = m.vis.patch.kp, pd = c.patch_dim();
  int32_t* pidx = static_cast<int32_t*>([v.pidx contents]);
  float* pw = static_cast<float*>([v.pw contents]);
  int32_t* hw = static_cast<int32_t*>([v.hw contents]);
  int32_t* seg = static_cast<int32_t*>([v.seg contents]);
  int32_t* dst = static_cast<int32_t*>([v.dst_rows contents]);
  __fp16* pix = static_cast<__fp16*>([v.pix contents]);
  const int side = c.side;
  for (const ImageJob& J : jobs) {
    const int gt = J.im->grid_t, gh = J.im->grid_h, gw = J.im->grid_w;
    std::vector<float> hs(static_cast<size_t>(gh)), ws(static_cast<size_t>(gw));
    for (int i = 0; i < gh; ++i) hs[size_t(i)] = gh > 1 ? float(i) * (float(side - 1) / float(gh - 1)) : 0.0f;
    for (int j = 0; j < gw; ++j) ws[size_t(j)] = gw > 1 ? float(j) * (float(side - 1) / float(gw - 1)) : 0.0f;
    int r = J.patch0;
    for (int t = 0; t < gt; ++t)
      for (int bh = 0; bh < gh / 2; ++bh)
        for (int bw = 0; bw < gw / 2; ++bw)
          for (int mh = 0; mh < 2; ++mh)
            for (int mw = 0; mw < 2; ++mw, ++r) {
              const int i = bh * 2 + mh, j = bw * 2 + mw;
              const int fh = int(hs[size_t(i)]), fw = int(ws[size_t(j)]);
              const int ch = std::min(fh + 1, side - 1), cw = std::min(fw + 1, side - 1);
              const float dh = hs[size_t(i)] - float(fh), dw = ws[size_t(j)] - float(fw);
              pidx[r * 4 + 0] = fh * side + fw;
              pidx[r * 4 + 1] = fh * side + cw;
              pidx[r * 4 + 2] = ch * side + fw;
              pidx[r * 4 + 3] = ch * side + cw;
              pw[r * 4 + 0] = bf16_round((1.0f - dh) * (1.0f - dw));
              pw[r * 4 + 1] = bf16_round((1.0f - dh) * dw);
              pw[r * 4 + 2] = bf16_round(dh * (1.0f - dw));
              pw[r * 4 + 3] = bf16_round(dh * dw);
              hw[r * 2 + 0] = i;
              hw[r * 2 + 1] = j;
              seg[r * 2 + 0] = J.patch0;
              seg[r * 2 + 1] = J.patch0 + J.n_patch;
            }
    for (int k = 0; k < J.n_patch / 4; ++k) dst[J.patch0 / 4 + k] = J.dst_row0 + k;
    const float* src = J.im->pixels;
    const int p0 = J.patch0;
    parallel_for(size_t(J.n_patch), [&](size_t q) {
      __fp16* d = pix + size_t(p0 + int(q)) * size_t(kp);
      const float* sp = src + q * size_t(pd);
      for (int e = 0; e < pd; ++e) d[e] = __fp16(sp[e]);
      for (int e = pd; e < kp; ++e) d[e] = __fp16(0.0f);
    });
  }
}

// Vision tower over every image of the pass (patch rows concatenated); the
// merged tokens overwrite their placeholder rows of `h`.
inline void encode_vision(Model& m, Enc& E, int P, id<MTLBuffer> h) {
  const VisionConfig& c = m.vis.c;
  VScratch& v = m.vs;
  const int D = c.D, Dm = 4 * c.D;
  gemm_h(E, m.vis.patch, v.pix, 0, m.vis.patch.kp, v.x, 0, D, P, 0);
  E.pipe("vit_pos_add");
  E.set(0, v.x);
  E.set(1, m.vis.pos);
  E.set(2, v.pidx);
  E.set(3, v.pw);
  E.bytes(4, D);
  E.rd(v.x);
  E.wr(v.x);
  E.threads(MTLSizeMake(size_t(D / 4), size_t(P), 1), MTLSizeMake(64, 1, 1));
  struct {
    int N, H, HD;
    float scale;
  } prep{P, c.heads, c.hd, float(1.0 / std::sqrt(double(c.hd)))};
  struct {
    int N, H;
  } at{P, c.heads};
  // 64 query rows per threadgroup (bit-identical output, ESPETERO_VIT_ATTN=1
  // selects the 32-row kernel). Variants: 5 = 128 rows (two 8-row groups
  // per simdgroup) with the next K/V block prefetched, 6 = 128 rows, 7 = 64
  // rows prefetched; bit-identical to 2 when the rows share one segment, so
  // 5 and 6 serve single-image passes only (others fall back to 2).
  static const int attn_v = [] {
    const char* v = std::getenv("ESPETERO_VIT_ATTN");
    return v && *v >= '1' && *v <= '7' ? *v - '0' : 2;  // 3: 64-key blocks, 4: also K transposed (not bit-identical)
  }();
  const int32_t* segh = static_cast<const int32_t*>([v.seg contents]);
  const bool one_seg = P > 0 && segh[0] == 0 && segh[1] == P;
  const int av = ((attn_v == 5 || attn_v == 6) && !one_seg) ? 2 : attn_v;
  const bool attn2 = av != 1;
  const int attn_rows = av == 1 ? 32 : (av == 5 || av == 6) ? 128 : 64;
  static const char* attn_names[8] = {"", "vit_attention_", "vit_attention2_", "vit_attention3_", "vit_attention4_",
                                      "vit_attention5_", "vit_attention6_", "vit_attention7_"};
  const std::string attn = std::string(attn_names[av]) + std::to_string(c.hd);
  // A ~1 MP photo is ~3.2K patch rows, ~0.5 s for the tower on an M1 Max:
  // several photos in one pass made a command buffer of seconds, which macOS
  // aborts while the display is in use. From 2K rows on, every block is
  // committed on its own (~0.3 ms per commit); the work and its order are
  // unchanged (bit-identical).
  static const int flush_from = [] {  // ESPETERO_VIT_FLUSH=rows (0: one command buffer)
    const char* e = std::getenv("ESPETERO_VIT_FLUSH");
    return e && *e ? std::atoi(e) : 2048;
  }();
  const int flush_every = flush_from > 0 && P >= flush_from ? 1 : 0;
  int bi = 0;
  std::snprintf(E.phase, sizeof(E.phase), "vision, %d patch rows", P);
  for (const VBlock& B : m.vis.blocks) {
    if (flush_every && bi > 0 && bi % flush_every == 0) E.flush();
    ++bi;
    layernorm(E, v.x, B.n1w, B.n1b, v.xn, P, D, c.eps);
    gemm_h(E, B.qkv, v.xn, 0, D, v.qkv, 0, 3 * D, P, 0);
    E.pipe("vit_qkv_prep");
    E.set(0, v.qkv);
    E.set(1, v.hw);
    E.set(2, m.vis.inv_freq);
    E.set(3, v.q);
    E.set(4, v.k);
    E.set(5, v.v);
    E.bytes(6, prep);
    E.rd(v.qkv);
    E.wr(v.q);
    E.wr(v.k);
    E.wr(v.v);
    E.threads(MTLSizeMake(size_t(c.hd / 2), size_t(c.heads), size_t(P)), MTLSizeMake(size_t(c.hd / 2), 1, 1));
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
    E.groups(MTLSizeMake(size_t(c.heads), size_t(P + attn_rows - 1) / size_t(attn_rows), 1), MTLSizeMake(attn2 ? 256 : 128, 1, 1));
    gemm_h(E, B.proj, v.o, 0, D, v.x, 0, D, P, 1);
    layernorm(E, v.x, B.n2w, B.n2b, v.xn, P, D, c.eps);
    gemm_h(E, B.fc1, v.xn, 0, D, v.act, 0, c.Ip(), P, 2);
    gemm_h(E, B.fc2, v.act, 0, c.Ip(), v.x, 0, D, P, 1);
  }
  layernorm(E, v.x, m.vis.mnw, m.vis.mnb, v.xn, P, D, c.eps);
  gemm_h(E, m.vis.m1, v.xn, 0, Dm, v.mact, 0, Dm, P / 4, 3);
  gemm_h(E, m.vis.m2, v.mact, 0, Dm, v.out, 0, c.out, P / 4, 0);
  E.pipe("scatter_rows_f32");
  E.set(0, v.out);
  E.set(1, h);
  E.set(2, v.dst_rows);
  E.bytes(3, c.out);
  E.rd(v.out);
  E.wr(h);
  E.threads(MTLSizeMake(size_t(c.out / 4), size_t(P / 4), 1), MTLSizeMake(64, 1, 1));
  if (flush_every) E.flush();  // the language model's layers start a new command buffer
  E.phase[0] = 0;
}

// Validate the images of one tree whose root starts at absolute row `base`
// (root_len < 0: encode-only, tokens are laid out back to back).
inline void collect_images(const ImageRef* images, int n, int root_len, int base, int& n_patch,
                           std::vector<ImageJob>& jobs) {
  if (n < 0 || (n > 0 && !images)) fail("invalid images");
  for (int i = 0; i < n; ++i) {
    const ImageRef& im = images[i];
    if (!im.pixels || im.grid_t < 1 || im.grid_h < 2 || im.grid_w < 2 || im.grid_h % 2 || im.grid_w % 2)
      fail("invalid image grid");
    ImageJob J;
    J.im = &im;
    J.patch0 = n_patch;
    J.n_patch = im.grid_t * im.grid_h * im.grid_w;
    const int n_tok = J.n_patch / 4;
    if (root_len >= 0 && (im.row < 0 || im.row + n_tok > root_len))
      fail("image rows must lie inside the root segment");
    J.dst_row0 = root_len >= 0 ? base + im.row : n_patch / 4;
    n_patch += J.n_patch;
    jobs.push_back(J);
  }
}

}  // namespace espetero::native
