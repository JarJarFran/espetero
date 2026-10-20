// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Qwen3-VL vision tower: host-side patch metadata and the GPU encode (patch
// embedding, interpolated position embedding, 2D-rotary blocks, 2x2 merger).
// LFM2-VL's SigLIP2 NaFlex tower runs on the same structures: patch rows
// (py, px, channel) of one temporal patch, the learned position table resized
// to each image's patch grid on the host (torch's antialiased bilinear
// interpolation, siglip2's resize_positional_embeddings), no rotary (zero
// frequencies), and the 2x2 pixel unshuffle + projector as the merger.
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
  if (c.unified) {  // Gemma 4 unified: `rows` are soft tokens (gemma_vision.h)
    v.qkv = m.buffer(size_t(R) * size_t(c.unified_dim()) * 4);  // fp32 pixel rows
    v.pix = m.buffer(size_t(R) * size_t(m.vis.patch.kp) * 2);
    v.x = m.buffer(size_t(R) * size_t(c.D) * 4);
    v.xn = m.buffer(size_t(R) * size_t(c.D) * 2);
    v.out = m.buffer(size_t(R) * size_t(c.out) * 4);
    v.hw = m.buffer(size_t(R) * 8);
    v.dst_rows = m.buffer(size_t(R) * 4);
    v.rows = R;
    return;
  }
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
  if (!c.deepstack.empty()) v.ds = m.buffer(c.deepstack.size() * size_t(R / 4) * size_t(c.out) * 4);
  v.rows = R;
}

// torch's antialiased separable weights for one axis (bilinear,
// _compute_indices_min_size_weights_aa of aten's UpSampleKernel.cpp with the
// same float / double steps): output i reads inputs [x0, x0 + w.size()).
struct AaTaps {
  int x0 = 0;
  std::vector<float> w;
};
inline std::vector<AaTaps> aa_bilinear_taps(int n_in, int n_out) {
  const float scale = float(n_in) / float(n_out);
  const float support = scale >= 1.0f ? float(1.0 * double(scale)) : 1.0f;
  const float invscale = scale >= 1.0f ? float(1.0 / double(scale)) : 1.0f;
  const int max_taps = int(std::ceil(double(support))) * 2 + 1;
  std::vector<AaTaps> out(static_cast<size_t>(n_out));
  for (int i = 0; i < n_out; ++i) {
    const float center = float(double(scale) * (double(i) + 0.5));
    const int64_t xmin = std::max<int64_t>(int64_t(double(center - support) + 0.5), 0);
    int64_t xsize = std::min<int64_t>(int64_t(double(center + support) + 0.5), n_in) - xmin;
    xsize = std::clamp<int64_t>(xsize, 0, max_taps);
    AaTaps& t = out[size_t(i)];
    t.x0 = int(xmin);
    float total = 0.0f;
    for (int64_t j = 0; j < xsize; ++j) {
      float x = float((double(float(j + xmin) - center) + 0.5) * double(invscale));
      x = std::fabs(x);
      const float w = x < 1.0f ? float(1.0 - double(x)) : 0.0f;
      t.w.push_back(w);
      total += w;
    }
    if (total != 0.0f)
      for (float& w : t.w) w /= total;
  }
  return out;
}

// SigLIP2: the position table [side][side][D] resized to (gh, gw) as
// F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// does it in fp32 (the width first, then the height), row-major [gh * gw][D];
// kept for the 8 most recent grids.
inline const std::vector<float>& siglip_pos(Model& m, int gh, int gw) {
  auto& cache = m.vis.pos_grids;
  if (auto it = cache.find({gh, gw}); it != cache.end()) return it->second;
  if (cache.size() >= 8) cache.clear();
  const int side = m.vis.c.side, D = m.vis.c.D;
  const float* tab = static_cast<const float*>(ptr(m.vis.pos));
  std::vector<float> wide;  // [side][gw][D]
  if (gw != side) {
    const std::vector<AaTaps> tx = aa_bilinear_taps(side, gw);
    wide.assign(size_t(side) * size_t(gw) * size_t(D), 0.0f);
    parallel_for(size_t(side), [&](size_t a) {
      for (int j = 0; j < gw; ++j) {
        const AaTaps& t = tx[size_t(j)];
        float* o = &wide[(a * size_t(gw) + size_t(j)) * size_t(D)];
        for (int d = 0; d < D; ++d) {
          float acc = tab[(a * size_t(side) + size_t(t.x0)) * size_t(D) + size_t(d)] * t.w[0];
          for (size_t k = 1; k < t.w.size(); ++k) acc += tab[(a * size_t(side) + size_t(t.x0) + k) * size_t(D) + size_t(d)] * t.w[k];
          o[d] = acc;
        }
      }
    });
  } else {
    wide.assign(tab, tab + size_t(side) * size_t(side) * size_t(D));
  }
  std::vector<float> out;
  if (gh != side) {
    const std::vector<AaTaps> ty = aa_bilinear_taps(side, gh);
    out.assign(size_t(gh) * size_t(gw) * size_t(D), 0.0f);
    parallel_for(size_t(gh), [&](size_t i) {
      const AaTaps& t = ty[i];
      for (int j = 0; j < gw; ++j) {
        float* o = &out[(i * size_t(gw) + size_t(j)) * size_t(D)];
        for (int d = 0; d < D; ++d) {
          float acc = wide[(size_t(t.x0) * size_t(gw) + size_t(j)) * size_t(D) + size_t(d)] * t.w[0];
          for (size_t k = 1; k < t.w.size(); ++k)
            acc += wide[((size_t(t.x0) + k) * size_t(gw) + size_t(j)) * size_t(D) + size_t(d)] * t.w[k];
          o[d] = acc;
        }
      }
    });
  } else {
    out = std::move(wide);
  }
  return cache.emplace(std::make_pair(gh, gw), std::move(out)).first->second;
}

// One image of a pass: its patch rows in the vision batch, its merged tokens
// in the tower's output and the decoder rows they replace.
struct ImageJob {
  const ImageRef* im = nullptr;
  int patch0 = 0, n_patch = 0, dst_row0 = 0;
  int tok0 = 0, n_tok = 0;
};

// Per-patch metadata in the processor's merge-window order: bilinear taps of
// the learned position table (weights rounded to bf16 like the reference),
// 2D rotary coordinates and the image segment; pixels -> fp16.
//
// SigLIP2 (c.siglip): the processor's patch rows are row-major; they take
// the merge-window order here (patch (2 bh + mh, 2 bw + mw) at row
// ((bh * gw / 2 + bw) * 2 + mh) * 2 + mw), so the merged rows are the
// projector's 2x2 unshuffle; each row's resized position embedding goes into
// the tower's input rows, which the patch embedding then adds to.
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
  if (c.siglip) {
    float* X = static_cast<float*>([v.x contents]);
    const int D = c.D;
    for (const ImageJob& J : jobs) {
      const int gh = J.im->grid_h, gw = J.im->grid_w;
      const std::vector<float>& pos = siglip_pos(m, gh, gw);
      const float* src = J.im->pixels;
      const int p0 = J.patch0;
      parallel_for(size_t(J.n_patch), [&](size_t q) {
        const int r = int(q), mw = r % 2, mh = (r / 2) % 2, b = r / 4, bw = b % (gw / 2), bh = b / (gw / 2);
        const int i = bh * 2 + mh, j = bw * 2 + mw, sq = i * gw + j;
        const int row = p0 + r;
        std::memcpy(X + size_t(row) * size_t(D), pos.data() + size_t(sq) * size_t(D), size_t(D) * 4);
        __fp16* d = pix + size_t(row) * size_t(kp);
        const float* sp = src + size_t(sq) * size_t(pd);
        for (int e = 0; e < pd; ++e) d[e] = __fp16(sp[e]);
        for (int e = pd; e < kp; ++e) d[e] = __fp16(0.0f);
        hw[row * 2 + 0] = i;
        hw[row * 2 + 1] = j;
        seg[row * 2 + 0] = p0;
        seg[row * 2 + 1] = p0 + J.n_patch;
      });
      for (int k = 0; k < J.n_patch / 4; ++k) dst[J.patch0 / 4 + k] = J.dst_row0 + k;
    }
    return;
  }
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
// merged tokens overwrite their placeholder rows of `h`. DeepStack towers
// (Qwen3-Omni) also leave merger j's output of block c.deepstack[j] in
// v.ds ([j][P / 4][out], the decoder adds it to the same rows).
inline void encode_vision(Model& m, Enc& E, int P, id<MTLBuffer> h) {
  const VisionConfig& c = m.vis.c;
  VScratch& v = m.vs;
  const int D = c.D, Dm = 4 * c.D, Dh = c.merger_hidden();
  if (c.siglip) {  // the rows hold their position embeddings (prepare_images): add the patch embedding
    gemm_h(E, m.vis.patch, v.pix, 0, m.vis.patch.kp, v.x, 0, D, P, 1);
  } else {
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
  }
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
    // DeepStack merger: LayerNorm over each merged row (4 consecutive patch rows), fc1 + GELU, fc2.
    for (size_t j = 0; j < c.deepstack.size(); ++j) {
      if (c.deepstack[j] != bi - 1) continue;
      const VMerger& DM = m.vis.ds[j];
      layernorm(E, v.x, DM.nw, DM.nb, v.xn, P / 4, Dm, c.eps);
      gemm_h(E, DM.fc1, v.xn, 0, Dm, v.mact, 0, Dm, P / 4, 3);
      gemm_h(E, DM.fc2, v.mact, 0, Dm, v.ds, j * size_t(P / 4) * size_t(c.out) * 4, c.out, P / 4, 0);
    }
  }
  layernorm(E, v.x, m.vis.mnw, m.vis.mnb, v.xn, P, D, c.eps);
  gemm_h(E, m.vis.m1, v.xn, 0, Dm, v.mact, 0, Dh, P / 4, 3);
  gemm_h(E, m.vis.m2, v.mact, 0, Dh, v.out, 0, c.out, P / 4, 0);
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
// (root_len < 0: encode-only, tokens are laid out back to back). `merge`:
// the tower's merge window (Qwen3-VL 2) or pooling kernel (Gemma 4: 3, one
// frame per image); a grid's tokens are its patches / merge^2. `align`: the
// first patch row of each image is a multiple of it (Gemma 4: 64, the
// attention's query tile, so an image's rows are computed exactly as when it
// is alone in the pass; the rows between images are padding).
inline void collect_images(const ImageRef* images, int n, int root_len, int base, int& n_patch,
                           std::vector<ImageJob>& jobs, int merge = 2, int align = 1) {
  if (n < 0 || (n > 0 && !images)) fail("invalid images");
  int n_tok = 0;
  for (const ImageJob& J : jobs) n_tok += J.n_tok;
  for (int i = 0; i < n; ++i) {
    const ImageRef& im = images[i];
    if (!im.pixels || im.grid_t < 1 || im.grid_h < merge || im.grid_w < merge || im.grid_h % merge ||
        im.grid_w % merge || (merge != 2 && im.grid_t != 1))
      fail("invalid image grid");
    ImageJob J;
    J.im = &im;
    n_patch = (n_patch + align - 1) / align * align;
    J.patch0 = n_patch;
    J.n_patch = im.grid_t * im.grid_h * im.grid_w;
    J.n_tok = J.n_patch / (merge * merge);
    J.tok0 = n_tok;
    if (root_len >= 0 && (im.row < 0 || im.row + J.n_tok > root_len))
      fail("image rows must lie inside the root segment");
    J.dst_row0 = root_len >= 0 ? base + im.row : n_tok;
    n_patch += J.n_patch;
    n_tok += J.n_tok;
    jobs.push_back(J);
  }
}

}  // namespace espetero::native
