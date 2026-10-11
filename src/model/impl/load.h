// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Build a Qwen3.5 model from a weight source: an MLX directory (safetensors +
// config.json, uniform or mixed widths) or a GGUF file (qwen35) with an
// optional mmproj (qwen3vl_merger) for images.
#pragma once

#include <fstream>
#include <sstream>

#include <dirent.h>

#include "tensors.h"
#include "model.h"

namespace espetero::native {

// A matrix in host layout before the GPU repack: quantized or dense fp16.
struct HostMat {
  bool dense = false;
  QW q;
  Buf f16;
  int out = 0, in = 0;
};

inline HostMat host_mat(Model& m, Arena& tmp, const WeightSource& src, const std::string& name, int out, int in) {
  const TensorView t = src.tensor(name);
  HostMat h;
  h.out = out;
  h.in = in;
  if (t.kind == TensorView::Dense) {
    h.dense = true;
    h.f16 = f16_from_view(tmp, t, out, in);
    m.source_types[t.dtype] += 1;
  } else {
    h.q = qw_from_view(tmp, t, out, in);
    m.source_types[t.kind == TensorView::GGUFQuant ? ggml_type_info(t.gguf_type)->name
                                                    : "mlx" + std::to_string(t.bits) + "g" + std::to_string(t.group)] += 1;
  }
  return h;
}

// Dequantize a host QW to fp16 (identical to the GPU's half-precision fma).
inline Buf dequant_f16(Arena& ar, const QW& q_in) {
  const QW q0 = to_affine(ar, q_in);
  const QW q = to_half_scales(ar, q0.src_bits == 4 || q0.src_bits == 8 ? q0 : widen8(ar, q0));
  Buf out = ar.alloc(size_t(q.out) * q.in * 2);
  __fp16* dst = (__fp16*)ptr(out);
  const uint8_t* codes = (const uint8_t*)ptr(q.w);
  const uint16_t* s = (const uint16_t*)ptr(q.s);
  const uint16_t* z = (const uint16_t*)ptr(q.z);
  const int in = q.in, g = q.group, bits = q.src_bits;
  parallel_for(size_t(q.out), [&](size_t r) {
    for (int c = 0; c < in; ++c) {
      const uint32_t code = bits == 8 ? codes[r * size_t(in) + c]
                                      : (((const uint32_t*)codes)[(r * size_t(in) + c) / 8] >> (4 * (c % 8))) & 0xF;
      const size_t gi = r * size_t(in / g) + size_t(c / g);
      dst[r * size_t(in) + c] = __fp16(float(code) * f16_to_f32(s[gi]) + f16_to_f32(z[gi]));
    }
  });
  return out;
}

inline Buf as_f16(Arena& tmp, const HostMat& h) { return h.dense ? h.f16 : dequant_f16(tmp, h.q); }

// A BITS-bit (2, 4 or 8), group-64 affine copy of `h` (per-group min/max):
// 4-bit for a cheaper MTP draft head, 8-bit for a dense (bf16) MoE router or
// shared-expert gate, so the fused route kernel serves it (oQ4e-mtp files
// keep the router in bf16; 8 bits per 64-group loses ~0.2 % of the group
// range per weight, far below the routing margins), 2-bit for the coarse
// draft head (scale and bias refit by least squares, see below).
template <int BITS>
inline HostMat requant(Arena& tmp, const HostMat& h) {
  static_assert(BITS == 2 || BITS == 4 || BITS == 8, "requant: 2, 4 or 8 bits");
  HostMat o;
  o.out = h.out;
  o.in = h.in;
  QW& q = o.q;
  q.out = h.out;
  q.in = h.in;
  q.bits = q.src_bits = BITS;
  q.group = 64;
  q.half_scales = true;
  const int in = h.in, gpr = in / 64;
  q.w = tmp.alloc(size_t(h.out) * in * BITS / 8);
  q.s = tmp.alloc(size_t(h.out) * gpr * 2);
  q.z = tmp.alloc(size_t(h.out) * gpr * 2);
  QW src;
  if (!h.dense) {
    src = to_affine(tmp, h.q);
    if (src.src_bits != 4 && src.src_bits != 8) src = widen8(tmp, src);
    src = to_half_scales(tmp, src);
  }
  const __fp16* dense = h.dense ? (const __fp16*)ptr(h.f16) : nullptr;
  const uint8_t* codes = h.dense ? nullptr : (const uint8_t*)ptr(src.w);
  const uint16_t* ss = h.dense ? nullptr : (const uint16_t*)ptr(src.s);
  const uint16_t* zz = h.dense ? nullptr : (const uint16_t*)ptr(src.z);
  const int g = src.group, bits = src.src_bits;
  uint8_t* wo = (uint8_t*)ptr(q.w);
  uint16_t* so = (uint16_t*)ptr(q.s);
  uint16_t* zo = (uint16_t*)ptr(q.z);
  parallel_for(size_t(h.out), [&](size_t r) {
    std::vector<float> row(in);
    for (int c = 0; c < in; ++c) {
      if (dense) {
        row[c] = float(dense[r * size_t(in) + c]);
      } else {
        const uint32_t code = bits == 8 ? codes[r * size_t(in) + c]
                                        : (((const uint32_t*)codes)[(r * size_t(in) + c) / 8] >> (4 * (c % 8))) & 0xF;
        const size_t gi = r * size_t(in / g) + size_t(c / g);
        row[c] = float(code) * f16_to_f32(ss[gi]) + f16_to_f32(zz[gi]);
      }
    }
    for (int k = 0; k < gpr; ++k) {
      const float* v = &row[size_t(k) * 64];
      float mn = v[0], mx = mn;
      for (int c = 1; c < 64; ++c) {
        mn = std::min(mn, v[c]);
        mx = std::max(mx, v[c]);
      }
      constexpr float levels = float((1 << BITS) - 1);
      float s1 = mx > mn ? (mx - mn) / levels : 1.0f, z1 = mn;
      if (BITS == 2 && mx > mn) {
        // Four levels spread over min..max leave most weights far from any
        // level: alternate nearest codes and the least-squares scale and
        // bias for those codes (each step lowers the squared error).
        for (int it = 0; it < 6; ++it) {
          double sq = 0, sv = 0, sqq = 0, sqv = 0;
          for (int c = 0; c < 64; ++c) {
            const double qc = double(std::clamp(std::lrint((v[c] - z1) / s1), 0L, long(levels)));
            sq += qc, sv += v[c], sqq += qc * qc, sqv += qc * v[c];
          }
          const double var = sqq - sq * sq / 64.0;
          if (var <= 0) break;
          const double s2 = (sqv - sq * sv / 64.0) / var;
          if (!(s2 > 0)) break;
          s1 = float(s2);
          z1 = float((sv - s2 * sq) / 64.0);
        }
      }
      const uint16_t sh = f32_to_f16_bits(s1);
      const uint16_t zh = f32_to_f16_bits(z1);
      so[r * gpr + k] = sh;
      zo[r * gpr + k] = zh;
      const float sc = f16_to_f32(sh), z0 = f16_to_f32(zh);
      auto enc = [&](float x) { return uint8_t(std::clamp(std::lrint((x - z0) / sc), 0L, long(levels))); };
      if (BITS == 2) {
        for (int c = 0; c < 64; c += 4)
          wo[(r * size_t(in) + size_t(k * 64 + c)) / 4] =
              uint8_t(enc(v[c]) | (enc(v[c + 1]) << 2) | (enc(v[c + 2]) << 4) | (enc(v[c + 3]) << 6));
      } else if (BITS == 4) {
        for (int c = 0; c < 64; c += 2)
          wo[(r * size_t(in) + size_t(k * 64 + c)) / 2] = uint8_t(enc(row[k * 64 + c]) | (enc(row[k * 64 + c + 1]) << 4));
      } else {
        for (int c = 0; c < 64; ++c) wo[r * size_t(in) + size_t(k * 64 + c)] = enc(row[k * 64 + c]);
      }
    }
  });
  return o;
}
inline HostMat requant4(Arena& tmp, const HostMat& h) { return requant<4>(tmp, h); }

// A quantized matrix served densely: repacked into `tmp`, then expanded on
// the GPU with rgemm's own dequantization (bit-identical operands).
inline HW densify(Model& m, id<MTLComputeCommandEncoder> e, Arena& tmp, QW q) {
  repack(m, e, q, tmp, &tmp);
  HW d;
  d.out = q.out;
  d.in = d.kp = q.in;
  const int Np = (q.out + 31) / 32 * 32;
  d.wr = m.arena->alloc(size_t(Np) * q.in * 2);
  const int KG = q.in / 64;
  [e setComputePipelineState:m.kernel("dequant_hr" + std::to_string(q.bits), q.fc())];
  [e setBuffer:q.wr.b offset:q.wr.off atIndex:0];
  [e setBuffer:q.sb.b offset:q.sb.off atIndex:1];
  const Buf& hb = q.wh ? q.wh : q.wr;
  [e setBuffer:hb.b offset:hb.off atIndex:2];
  [e setBuffer:d.wr.b offset:d.wr.off atIndex:3];
  [e setBytes:&q.out length:4 atIndex:4];
  [e setBytes:&q.in length:4 atIndex:5];
  [e dispatchThreads:MTLSizeMake(size_t(Np / 32) * KG * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  return d;
}

inline Mat finish(Model& m, id<MTLComputeCommandEncoder> e, Arena& tmp, HostMat h, bool dense = false) {
  Mat out;
  if (h.dense) {
    out.dense = true;
    out.d = repack_dense(m, e, h.f16, h.out, h.in);
    ++m.n_dense;
  } else if (dense) {
    out.dense = true;
    out.d = densify(m, e, tmp, h.q);
    ++m.n_dense;
  } else {
    out.q = h.q;
    repack(m, e, out.q, tmp);
    out.q.w = out.q.s = out.q.z = Buf{};  // host staging stays with `tmp`
    ++(out.q.bits == 4 ? m.n_q4 : out.q.bits == 5 ? m.n_q5 : out.q.bits == 6 ? m.n_q6 : m.n_q8);
  }
  return out;
}

// Column runs for a fused projection: quantized parts fuse by GPU width,
// dense parts by concatenation.
inline std::vector<HostMat> fuse(Arena& tmp, const std::vector<HostMat>& parts) {
  std::vector<HostMat> runs;
  size_t i = 0;
  while (i < parts.size()) {
    size_t j = i;
    while (j < parts.size() && parts[j].dense == parts[i].dense) ++j;
    if (parts[i].dense) {
      HostMat h;
      h.dense = true;
      h.in = parts[i].in;
      for (size_t k = i; k < j; ++k) h.out += parts[k].out;
      h.f16 = tmp.alloc(size_t(h.out) * h.in * 2);
      size_t r = 0;
      for (size_t k = i; k < j; ++k) {
        std::memcpy((uint8_t*)ptr(h.f16) + r * h.in * 2, ptr(parts[k].f16), size_t(parts[k].out) * h.in * 2);
        r += size_t(parts[k].out);
      }
      runs.push_back(h);
    } else {
      std::vector<QW> qs;
      for (size_t k = i; k < j; ++k) qs.push_back(parts[k].q);
      for (const QW& q : fuse_rows(tmp, qs)) {
        HostMat h;
        h.q = q;
        h.out = q.out;
        h.in = q.in;
        runs.push_back(h);
      }
    }
    i = j;
  }
  return runs;
}

inline HostMat interleave_gate_up(Arena& tmp, const HostMat& gate, const HostMat& up) {
  HostMat h;
  h.out = 2 * gate.out;
  h.in = gate.in;
  if (!gate.dense && !up.dense) {
    h.q = interleave8(tmp, gate.q, up.q);
  } else {
    h.dense = true;
    h.f16 = interleave8_f16(tmp, as_f16(tmp, gate), as_f16(tmp, up), gate.out, gate.in);
  }
  return h;
}

inline Buf vec(Model& m, const WeightSource& src, const std::string& name, size_t n, float add = 0.0f) {
  std::vector<float> v = src.floats(name, n);
  if (add != 0.0f)
    for (float& x : v) x += add;
  return m.arena->copy(v.data(), v.size() * 4);
}

inline void load_embedding(Model& m, Arena& tmp, const WeightSource& src, const std::string& name) {
  const Config& c = m.cfg;
  const TensorView t = src.tensor(name);
  Embed& E = m.embed;
  if (t.kind == TensorView::Dense) {
    E.kind = 2;
    E.codes = m.arena->alloc(size_t(c.V) * c.D * 2);
    Buf h = f16_from_view(tmp, t, c.V, c.D);
    std::memcpy(ptr(E.codes), ptr(h), size_t(c.V) * c.D * 2);
    return;
  }
  QW q = to_affine(tmp, qw_from_view(tmp, t, c.V, c.D));  // rows are gathered: affine
  if (q.src_bits != 4 && q.src_bits != 8) q = widen8(tmp, q);
  q = to_half_scales(tmp, q);
  E.kind = q.src_bits == 4 ? 0 : 1;
  E.group = q.group;
  E.codes = m.arena->copy(ptr(q.w), q.row_bytes() * size_t(c.V));
  const size_t groups = size_t(c.V) * size_t(c.D / q.group);
  E.s = m.arena->copy(ptr(q.s), groups * 2);
  E.z = m.arena->copy(ptr(q.z), groups * 2);
}

// ---------------------------------------------------------------- vision
inline HW load_vdense(Model& m, id<MTLComputeCommandEncoder> e, Arena& tmp, const WeightSource& src,
               const std::string& base, int out, int in, const std::vector<int>* perm = nullptr) {
  const TensorView t = src.tensor(base + ".weight");
  Buf host = f16_from_view(tmp, t, out, in, perm);
  std::vector<float> bias = src.floats(base + ".bias", size_t(out));
  return repack_dense(m, e, host, out, in, &bias);
}

inline void load_vision(Model& m, const WeightSource& src, VisionConfig c) {
  c.validate(m.cfg.D);
  Arena tmp(m.dev);
  const std::string v = "vision_tower.";
  {
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    // Conv3d kernel: MLX [O, T, P, P, C] (or PyTorch [O, C, T, P, P]); the
    // processor's patch rows are ordered (C, T, P, P).
    const TensorView pw = src.tensor(v + "patch_embed.proj.weight");
    std::vector<int> perm(size_t(c.patch_dim()));
    const bool mlx_order = pw.shape.size() == 5 && pw.shape[4] == c.in_ch;
    for (int ch = 0; ch < c.in_ch; ++ch)
      for (int t = 0; t < c.tps; ++t)
        for (int y = 0; y < c.patch; ++y)
          for (int x = 0; x < c.patch; ++x) {
            const int dst = ((ch * c.tps + t) * c.patch + y) * c.patch + x;
            perm[dst] = mlx_order ? (((t * c.patch + y) * c.patch + x) * c.in_ch + ch) : dst;
          }
    m.vis.patch = load_vdense(m, e, tmp, src, v + "patch_embed.proj", c.D, c.patch_dim(), &perm);
    m.vis.m1 = load_vdense(m, e, tmp, src, v + "merger.linear_fc1", 4 * c.D, 4 * c.D);
    m.vis.m2 = load_vdense(m, e, tmp, src, v + "merger.linear_fc2", c.out, 4 * c.D);
    [e endEncoding];
    commit_wait(cb);
  }
  m.vis.pos = vec(m, src, v + "pos_embed.weight", size_t(c.n_pos) * c.D);
  m.vis.mnw = vec(m, src, v + "merger.norm.weight", c.D);
  m.vis.mnb = vec(m, src, v + "merger.norm.bias", c.D);
  for (int i = 0; i < c.depth; ++i) {
    Arena btmp(m.dev);
    const std::string b = v + "blocks." + std::to_string(i) + ".";
    VBlock B;
    B.n1w = vec(m, src, b + "norm1.weight", c.D);
    B.n1b = vec(m, src, b + "norm1.bias", c.D);
    B.n2w = vec(m, src, b + "norm2.weight", c.D);
    B.n2b = vec(m, src, b + "norm2.bias", c.D);
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    B.qkv = load_vdense(m, e, btmp, src, b + "attn.qkv", 3 * c.D, c.D);
    B.proj = load_vdense(m, e, btmp, src, b + "attn.proj", c.D, c.D);
    B.fc1 = load_vdense(m, e, btmp, src, b + "mlp.linear_fc1", c.I, c.D);
    B.fc2 = load_vdense(m, e, btmp, src, b + "mlp.linear_fc2", c.D, c.I);
    [e endEncoding];
    commit_wait(cb);
    m.vis.blocks.push_back(B);
  }
  m.vis.inv_freq = m.arena->alloc(sizeof(float) * (c.hd / 4));
  float* f = (float*)ptr(m.vis.inv_freq);
  const int dim = c.hd / 2;
  for (int i = 0; i < c.hd / 4; ++i) f[i] = float(1.0 / std::pow(10000.0, double(2 * i) / double(dim)));
  c.present = true;
  m.vis.c = c;
}

// ---------------------------------------------------------------- decoder
// A MoE layer's router, shared-expert gate and stacked experts (MLX
// switch_mlp tensors [E, out, in], read as [E * out, in]).
inline void load_experts(Model& m, const WeightSource& src, const std::string& p, Layer& L) {
  const Config& c = m.cfg;
  Arena tmp(m.dev);
  HostMat router = host_mat(m, tmp, src, p + "mlp.gate.weight", c.E, c.D);
  HostMat sgate = host_mat(m, tmp, src, p + "mlp.shared_expert_gate.weight", 1, c.D);
  // A dense (bf16) router stays dense: the parallel fused route reads it in
  // fp16. ESPETERO_ROUTER_Q8=1 converts dense routers and shared gates to
  // 8-bit group 64 instead (lossy; opt-in).
  static const bool router_q8 = [] {
    const char* v = std::getenv("ESPETERO_ROUTER_Q8");
    return v && *v == '1';
  }();
  if (router_q8 && router.dense) router = requant<8>(tmp, router);
  if (router_q8 && sgate.dense) sgate = requant<8>(tmp, sgate);
  HostMat eg = host_mat(m, tmp, src, p + "mlp.switch_mlp.gate_proj.weight", c.E * c.Ie, c.D);
  HostMat eu = host_mat(m, tmp, src, p + "mlp.switch_mlp.up_proj.weight", c.E * c.Ie, c.D);
  HostMat ed = host_mat(m, tmp, src, p + "mlp.switch_mlp.down_proj.weight", c.E * c.D, c.Ie);
  if (eg.dense || eu.dense || ed.dense) fail("dense expert weights are not supported");
  // Interleaving by 8 rows keeps every expert's 2 * Ie rows together (Ie % 64 == 0).
  const HostMat egu = interleave_gate_up(tmp, eg, eu);
  id<MTLCommandBuffer> cb = [m.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  L.router = finish(m, e, tmp, router);
  L.shared_gate = finish(m, e, tmp, sgate);
  // Routers and shared gates read the normalized rows through the route
  // kernels (or plain GEMMs without norm links): a dense one does not block
  // the fused RMSNorm of the decoder GEMVs.
  m.n_dense_mtp += (L.router.dense ? 1 : 0) + (L.shared_gate.dense ? 1 : 0);
  L.experts_gu = finish(m, e, tmp, egu).q;
  L.experts_down = finish(m, e, tmp, ed).q;
  [e endEncoding];
  commit_wait(cb);
  L.moe = true;
}

// One decoder layer (prefix `p`, e.g. "language_model.model.layers.3.").
// requant_bits (8): dense matrices of the layer become 8-bit affine copies
// (the MTP layer of bf16 files); quantized ones stay as they are.
inline Layer load_layer(Model& m, const WeightSource& src, const std::string& p, bool linear, int index_in_kind,
                        int requant_bits = 0) {
  const Config& c = m.cfg;
  Layer L;
  L.linear = linear;
  L.index_in_kind = index_in_kind;
  L.in_norm = vec(m, src, p + "input_layernorm.weight", c.D);
  L.post_norm = vec(m, src, p + "post_attention_layernorm.weight", c.D);
  Arena tmp(m.dev);
  // MoE layers (detected by their expert tensors): the shared expert takes the
  // dense MLP's place; dense layers inside a MoE model keep intermediate_size
  // (or their tensors' width when the config lacks it).
  const bool moe = c.E > 0 && src.has(p + "mlp.switch_mlp.gate_proj.weight");
  const std::string mp = moe ? p + "mlp.shared_expert." : p + "mlp.";
  int I = moe ? c.I_shared : c.I;
  if (!I) I = int(src.tensor(mp + "gate_proj.weight").rows());
  L.I = I;
  HostMat gate = host_mat(m, tmp, src, mp + "gate_proj.weight", I, c.D);
  HostMat up = host_mat(m, tmp, src, mp + "up_proj.weight", I, c.D);
  HostMat down = host_mat(m, tmp, src, mp + "down_proj.weight", c.D, I);
  std::vector<HostMat> parts;
  HostMat out;
  if (L.linear) {
    const std::string a = p + "linear_attn.";
    parts = {host_mat(m, tmp, src, a + "in_proj_qkv.weight", c.C(), c.D),
             host_mat(m, tmp, src, a + "in_proj_z.weight", c.Vd(), c.D),
             host_mat(m, tmp, src, a + "in_proj_a.weight", c.Hv, c.D),
             host_mat(m, tmp, src, a + "in_proj_b.weight", c.Hv, c.D)};
    out = host_mat(m, tmp, src, a + "out_proj.weight", c.D, c.Vd());
    L.conv_w = vec(m, src, a + "conv1d.weight", size_t(c.C()) * c.conv_k);
    L.a_log = vec(m, src, a + "A_log", c.Hv);
    L.dt_bias = vec(m, src, a + "dt_bias", c.Hv);
    L.gnorm = vec(m, src, a + "norm.weight", c.Dv);
  } else {
    const std::string a = p + "self_attn.";
    parts = {host_mat(m, tmp, src, a + "q_proj.weight", c.H * c.HD * 2, c.D),
             host_mat(m, tmp, src, a + "k_proj.weight", c.kv_dim(), c.D),
             host_mat(m, tmp, src, a + "v_proj.weight", c.kv_dim(), c.D)};
    out = host_mat(m, tmp, src, a + "o_proj.weight", c.D, c.H * c.HD);
    L.q_norm = vec(m, src, a + "q_norm.weight", c.HD);
    L.k_norm = vec(m, src, a + "k_norm.weight", c.HD);
  }
  if (requant_bits == 8) {
    for (HostMat* h : {&gate, &up, &down, &out})
      if (h->dense) *h = requant<8>(tmp, *h);
    for (HostMat& h : parts)
      if (h.dense) h = requant<8>(tmp, h);
  }
  // Gate and up fuse (rows interleaved for the SwiGLU epilogue) when they
  // share a layout; otherwise each keeps its own and they run as two GEMMs
  // (fusing would widen both, e.g. 8 bits per weight for Q5_K + Q6_K).
  const bool same_layout = gate.dense || up.dense ||
                           (gate.q.bits == up.q.bits && gate.q.src_bits == up.q.src_bits &&
                            gate.q.scale_mode == up.q.scale_mode && gate.q.group == up.q.group &&
                            gate.q.half_scales == up.q.half_scales);
  const HostMat gu = same_layout ? interleave_gate_up(tmp, gate, up) : HostMat{};
  const std::vector<HostMat> runs = fuse(tmp, parts);
  id<MTLCommandBuffer> cb = [m.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  auto dn = [&](const char* k) { return m.dense_mats.count(k) > 0; };
  if (same_layout) {
    L.gate_up = finish(m, e, tmp, gu, dn("gate_up"));
  } else {
    L.gate = finish(m, e, tmp, gate, dn("gate_up"));
    L.up = finish(m, e, tmp, up, dn("gate_up"));
    m.split_mlp = true;
  }
  L.down = finish(m, e, tmp, down, dn("down"));
  L.out = finish(m, e, tmp, out, dn("out"));
  for (const HostMat& r : runs) L.in_proj.push_back(finish(m, e, tmp, r, dn("in_proj")));
  [e endEncoding];
  commit_wait(cb);
  if (moe) load_experts(m, src, p, L);
  return L;
}

// draft_head: the model drafts with an MTP head; build its 4-bit LM head
// (mtp.head4) and its 2-bit coarse head (mtp.coarse_c > 0).
inline void load_decoder(Model& m, const WeightSource& src, bool draft_head = false) {
  Config& c = m.cfg;
  c.validate();
  const std::string pre = "language_model.model.";
  {
    Arena tmp(m.dev);
    load_embedding(m, tmp, src, pre + "embed_tokens.weight");
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    const std::string head = c.tie ? pre + "embed_tokens.weight" : "language_model.lm_head.weight";
    const HostMat h = host_mat(m, tmp, src, head, c.V, c.D);
    if (draft_head && m.mtp.head4) {
      m.mtp.head = finish(m, e, tmp, requant4(tmp, h));
      --m.n_q4;  // not a decoder matrix
    }
    if (draft_head && m.mtp.coarse_c > 0 && !h.dense) {
      m.mtp.coarse = finish(m, e, tmp, requant<2>(tmp, h));
      --m.n_q8;  // not a decoder matrix (finish counts widths without a counter as 8-bit)
    }
    m.lm_head = finish(m, e, tmp, h);
    [e endEncoding];
    commit_wait(cb);
  }
  m.final_norm = vec(m, src, pre + "norm.weight", c.D);
  int nf = 0, nl = 0;
  for (int i = 0; i < c.L; ++i)
    m.layers.push_back(load_layer(m, src, pre + "layers." + std::to_string(i) + ".", c.linear[i],
                                  c.linear[i] ? nl++ : nf++));
  m.inv_freq = m.arena->alloc(sizeof(float) * (c.rot / 2));
  float* f = (float*)ptr(m.inv_freq);
  for (int i = 0; i < c.rot / 2; ++i) f[i] = float(1.0 / std::pow(c.theta, double(2 * i) / double(c.rot)));
  m.rope_sel = m.arena->copy(c.rope_sel, sizeof(int32_t) * 32);
}

// The MTP head from `src` (HF mtp.* names; GGUF nextn block), sharing the
// decoder's embedding and LM head.
inline void load_mtp(Model& m, const WeightSource& src) {
  const Config& c = m.cfg;
  const std::string p = "language_model.mtp.";
  Mtp& T = m.mtp;
  const int dense0 = m.n_dense, exempt0 = m.n_dense_mtp;
  T.enorm = vec(m, src, p + "pre_fc_norm_embedding.weight", c.D);
  T.hnorm = vec(m, src, p + "pre_fc_norm_hidden.weight", c.D);
  T.norm = vec(m, src, p + "norm.weight", c.D);
  {
    Arena tmp(m.dev);
    HostMat fc = host_mat(m, tmp, src, p + "fc.weight", c.D, 2 * c.D);
    if (T.q8 && fc.dense) fc = requant<8>(tmp, fc);
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    T.fc = finish(m, e, tmp, fc);
    [e endEncoding];
    commit_wait(cb);
  }
  T.layer = load_layer(m, src, p + "layers.0.", false, -1, T.q8 ? 8 : 0);
  // Every dense matrix of the MTP layer is exempt; its router may already have
  // been counted as exempt by load_experts.
  m.n_dense_mtp += (m.n_dense - dense0) - (m.n_dense_mtp - exempt0);
  // The MTP input staging ([embedding, hidden] rows of 2 * D halves) uses the
  // pass scratch `act`, sized max(I, I_shared, 2 * D) (forest.h): fine for a
  // MoE MTP layer, whose MLP width is the shared expert's (oQ4e-mtp files).
  const std::string where = src.describe();  // "gguf:<path>": keep the file name only
  T.source = where.substr(where.find_last_of('/') + 1);
  T.present = true;
}

// Files in `dir` whose lower-cased names start with `prefix` and end with
// `suffix`, sorted.
inline std::vector<std::string> list_files(const std::string& dir, const std::string& prefix, const std::string& suffix) {
  std::vector<std::string> out;
  DIR* d = ::opendir(dir.c_str());
  if (!d) return out;
  while (dirent* e = ::readdir(d)) {
    std::string name = e->d_name, low = name;
    for (auto& ch : low) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    if (low.size() >= prefix.size() + suffix.size() && low.compare(0, prefix.size(), prefix) == 0 &&
        ends_with(low, suffix))
      out.push_back(dir + "/" + name);
  }
  ::closedir(d);
  std::sort(out.begin(), out.end());
  return out;
}

// An MTP GGUF next to the model (unsloth ships it as MTP/mtp-*.gguf).
inline std::string find_mtp(const std::string& gguf_path) {
  const std::string dir = dirname_of(gguf_path);
  for (const std::string& d : {dir, dir + "/MTP"}) {
    auto v = list_files(d, "mtp", ".gguf");
    if (!v.empty()) return v[0];
  }
  return "";
}

inline std::string find_mmproj(const std::string& gguf_path) {
  auto v = list_files(dirname_of(gguf_path), "mmproj", ".gguf");
  return v.empty() ? "" : v[0];
}

}  // namespace espetero::native

#include "gemma_load.h"

namespace espetero::native {

// Qwen4-Exp checkpoints (qwen4_load.h).
inline void load_q4(Model& m, const SafetensorsSource& src, bool want_vision);
inline void load_q4_mtp(Model& m, const std::string& dir);

// `path`: an MLX model directory or a .gguf file. `mmproj`: vision projector
// for GGUF ("" = look next to the model file, "none" = text only).
//
// `mtp`: "" = the model's own MTP head (GGUF nextn block or MLX
// language_model.mtp.*), else an MTP GGUF found next to a GGUF model; a path =
// that GGUF or MLX directory; "none" = no drafting head.
inline void load_qwen35(Model& m, const std::string& path, bool want_vision, std::string mmproj,
                 std::string mtp = "") {
  // An MTP head from another file: a GGUF with a nextn layer, or an MLX
  // directory whose safetensors carry language_model.mtp.* tensors.
  auto external_mtp = [&](const std::string& file) {
    auto check = [&](const Config& mc, const WeightSource& src) {
      if (mc.D != m.cfg.D || mc.V != m.cfg.V || mc.H != m.cfg.H || mc.Hkv != m.cfg.Hkv || mc.I != m.cfg.I)
        fail(file + ": MTP head does not match the model's dimensions");
      if (!src.has("language_model.mtp.fc.weight")) fail(file + ": no MTP layer");
    };
    if (ends_with(file, ".gguf")) {
      GGUFSource msrc(file);
      check(config_from_gguf(msrc.file()), msrc);
      load_mtp(m, msrc);
    } else {
      SafetensorsSource msrc(file);
      check(config_from_mlx(msrc.config()), msrc);
      load_mtp(m, msrc);
    }
  };
  if (ends_with(path, ".gguf")) {
    GGUFSource text(path);
    m.format = "gguf";
    if (text.file().get_string("general.architecture", "") == "gemma4") {  // Text only so far
      if (!mtp.empty() && mtp != "none") fail("--mtp: Gemma 4 has no drafting head (prompt lookup drafts only)");
      if (!mmproj.empty() && mmproj != "none") fail("--mmproj: Gemma 4 vision is not supported yet (text only)");
      m.cfg = config_from_gemma_gguf(text.file());
      load_gemma(m, text);
      return;
    }
    m.cfg = config_from_gguf(text.file());
    std::string mtp_file;
    const bool own = mtp.empty() && text.has("language_model.mtp.fc.weight");
    if (mtp != "none" && !own) mtp_file = mtp.empty() ? find_mtp(path) : mtp;
    load_decoder(m, text, (m.mtp.head4 || m.mtp.coarse_c > 0) && (own || !mtp_file.empty()));
    if (own) load_mtp(m, text);
    else if (!mtp_file.empty()) external_mtp(mtp_file);
    if (want_vision && mmproj != "none") {
      if (mmproj.empty()) mmproj = find_mmproj(path);
      if (!mmproj.empty()) {
        GGUFVisionSource vsrc(mmproj);
        const VisionConfig vc = vision_from_gguf(vsrc.file());
        if (vc.present) load_vision(m, vsrc, vc);
      }
    }
    return;
  }
  SafetensorsSource src(path);
  if (const std::string other = family::mlx_check(path, src.config()); !other.empty()) fail(other);
  {  // Gemma 4: text only so far (no drafting head; images are refused by the engine)
    const json::Value& cfg = src.config();
    const json::Value* tc = cfg.get("text_config");
    if (is_gemma4_type(cfg.get_string("model_type", "")) ||
        (tc && tc->is_object() && is_gemma4_type(tc->get_string("model_type", "")))) {
      if (!mtp.empty() && mtp != "none") fail("--mtp: Gemma 4 has no drafting head (prompt lookup drafts only)");
      m.format = "mlx";
      m.cfg = config_from_gemma(cfg);
      load_gemma(m, src);
      return;
    }
  }
  if (!src.has("language_model.model.embed_tokens.weight"))
    fail("expected an MLX-converted Qwen3.5 checkpoint (language_model.model.* tensors)");
  m.format = "mlx";
  m.cfg = config_from_mlx(src.config());
  if (m.cfg.q4) {
    load_q4(m, src, want_vision);
    // MTP: a drafter directory (--mtp PATH, or by default a directory
    // beside the model whose config says qwen4_exp_mtp, e.g.
    // Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16); the file's own mtp.* head
    // (4-bit experts, reported to accept 0-4 % of drafts) is not loaded.
    std::string dir = mtp;
    if (dir.empty()) {
      std::string base = path;
      while (base.size() > 1 && base.back() == '/') base.pop_back();
      const std::string parent = base.substr(0, base.find_last_of('/'));
      if (DIR* d = ::opendir(parent.c_str())) {
        std::vector<std::string> found;
        while (dirent* e = ::readdir(d)) {
          const std::string name = e->d_name;
          if (name.find("MTP") == std::string::npos && name.find("mtp") == std::string::npos) continue;
          std::ifstream cf(parent + "/" + name + "/config.json");
          std::stringstream cs;
          cs << cf.rdbuf();
          if (cs.str().find("\"qwen4_exp_mtp\"") != std::string::npos) found.push_back(parent + "/" + name);
        }
        ::closedir(d);
        std::sort(found.begin(), found.end());
        if (!found.empty()) dir = found[0];
      }
    }
    if (!dir.empty() && dir != "none") load_q4_mtp(m, dir);
    return;
  }
  const bool own = mtp.empty() && src.has("language_model.mtp.fc.weight");
  const bool other = mtp != "none" && !mtp.empty();
  load_decoder(m, src, (m.mtp.head4 || m.mtp.coarse_c > 0) && (own || other));
  if (own) load_mtp(m, src);
  else if (other) external_mtp(mtp);
  if (want_vision && src.has("vision_tower.patch_embed.proj.weight")) {
    const VisionConfig vc = vision_from_mlx(src.config());
    if (vc.present) load_vision(m, src, vc);
  }
}

}  // namespace espetero::native
