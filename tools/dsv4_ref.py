#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero
"""DeepSeek-V4.1 reference forward in numpy, for Espetero's parity checks.

A line-by-line transcription of DeepSeek's inference/model.py (Transformer.forward
with start_pos 0, batch 1), on the checkpoint's own FP8 / FP4 tensors, in
float32: the fp8 / fp4 activation rounding of the reference (act_quant,
fp4_act_quant) is reproduced, its bf16 storage of activations is not.

    python3 -I tools/dsv4_ref.py MODEL_DIR --tokens 0,1234,... [--layers N] [--out DIR]

Writes DIR/h<l>.f32 (the residual streams after layer l, [T, 4, D]) and, with
all layers, DIR/logits.f32 (last position). Needs numpy only.
"""
import argparse
import json
import math
import os
import struct
import sys

import numpy as np

# ---------------------------------------------------------------- checkpoint
class Ckpt:
    def __init__(self, d):
        self.d = d
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.loc = {}
        self.maps = {}
        for f in sorted(set(idx.values())):
            path = os.path.join(d, f)
            if not os.path.exists(path):
                continue
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            self.maps[f] = np.memmap(path, dtype=np.uint8, mode="r")
            for k, v in h.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.loc[k] = (f, 8 + n + a, 8 + n + b, v["dtype"], v["shape"])

    def raw(self, name):
        f, a, b, dt, shape = self.loc[name]
        return self.maps[f][a:b], dt, shape

    def get(self, name):
        r, dt, shape = self.raw(name)
        if dt == "BF16":
            return (r.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if dt == "F32":
            return r.view(np.float32).reshape(shape)
        if dt == "F8_E4M3":
            return E4M3[r].reshape(shape)
        if dt == "F8_E8M0":
            return E8M0[r].reshape(shape)
        if dt == "I8":
            return r.reshape(shape)
        raise ValueError(dt)


def _e4m3():
    t = np.zeros(256, np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        x = m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        if e == 15 and m == 7:
            x = float("nan")
        t[b] = -x if s else x
    return t


E4M3 = _e4m3()
E8M0 = np.array([2.0 ** (b - 127) for b in range(256)], np.float32)
FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)
FP4_PAIRS = np.stack([FP4[np.arange(256) & 15], FP4[np.arange(256) >> 4]], -1)  # byte -> (lo, hi)


# ---------------------------------------------------------------- numerics
def round_e4m3(v):
    a = np.minimum(np.abs(v), 448.0).astype(np.float32)
    sub = a < 2.0 ** -6
    e = np.floor(np.log2(np.where(sub, 1.0, a)))
    step = np.where(sub, 2.0 ** -9, 2.0 ** (e - 3)).astype(np.float32)
    return np.copysign(np.rint(a / step) * step, v).astype(np.float32)


def round_e2m1(v):
    a = np.minimum(np.abs(v), 6.0).astype(np.float32)
    sub = a < 1.0
    e = np.floor(np.log2(np.where(sub, 1.0, a)))
    step = np.where(sub, 0.5, 2.0 ** (e - 1)).astype(np.float32)
    return np.copysign(np.rint(a / step) * step, v).astype(np.float32)


def pow2_ceil(x):
    m, e = np.frexp(x)  # x = m * 2^e, m in [0.5, 1)
    e = np.where(m == 0.5, e - 1, e)
    return np.ldexp(np.float32(1.0), e).astype(np.float32)


NOQUANT = False


def act_quant(x, block=32):
    """kernel.py act_quant with scale_fmt ue8m0, inplace (quantize-dequantize)."""
    if NOQUANT:
        return x.astype(np.float32)
    s = x.reshape(*x.shape[:-1], -1, block)
    amax = np.maximum(np.abs(s).max(-1, keepdims=True), 1e-4)
    sc = pow2_ceil((amax * np.float32(1 / 448.0)).astype(np.float32))
    return (round_e4m3(np.clip(s / sc, -448, 448)) * sc).reshape(x.shape).astype(np.float32)


def fp4_act_quant(x, block=32, e4m3_scale=False):
    if NOQUANT:
        return x.astype(np.float32)
    s = x.reshape(*x.shape[:-1], -1, block)
    amax = np.abs(s).max(-1, keepdims=True)
    if e4m3_scale:
        amax = np.maximum(amax, 6 * 2.0 ** -9)
        sc = round_e4m3(amax / 6.0)
    else:
        amax = np.maximum(amax, 6 * 2.0 ** -126)
        sc = pow2_ceil((amax * np.float32(1 / 6.0)).astype(np.float32))
    return (round_e2m1(np.clip(s / sc, -6, 6)) * sc).reshape(x.shape).astype(np.float32)


class Lin:
    """A weight as the reference's linear() uses it."""

    def __init__(self, ck, prefix, fp32=False):
        w = ck.loc.get(prefix + ".weight")
        self.kind = w[3]
        self.ck, self.p = ck, prefix
        self.fp32 = fp32
        self._w = None

    def dense(self):
        if self._w is None:
            ck, p = self.ck, self.p
            if self.kind == "F8_E4M3":
                w = ck.get(p + ".weight")
                s = ck.get(p + ".scale")
                N, K = w.shape
                s = np.repeat(np.repeat(s, 32, 0)[:N], 32, 1)[:, :K]
                self._w = (w * s).astype(np.float32)
            elif self.kind == "I8":
                r, _, shape = ck.raw(p + ".weight")
                N, K2 = shape
                w = FP4_PAIRS[r.reshape(N, K2)].reshape(N, 2 * K2)
                s = ck.get(p + ".scale")
                self._w = (w * np.repeat(s, 32, 1)).astype(np.float32)
            else:
                self._w = ck.get(p + ".weight").astype(np.float32)
        return self._w

    def __call__(self, x):
        if self.kind in ("F8_E4M3", "I8"):
            x = act_quant(x)
        return (x.astype(np.float32) @ self.dense().T).astype(np.float32)


def rmsnorm(x, w, eps):
    x = x.astype(np.float32)
    return (w * (x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)))).astype(np.float32)


def freqs(dim, seqlen, orig, base, factor, bf, bs):
    f = (1.0 / (np.float32(base) ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)))).astype(np.float32)
    if orig > 0:
        corr = lambda r: dim * math.log(orig / (r * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corr(bf)), 0)
        high = min(math.ceil(corr(bs)), dim - 1)
        ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 1e-3), 0, 1).astype(np.float32)
        smooth = 1 - ramp
        f = (f / np.float32(factor) * (1 - smooth) + f * smooth).astype(np.float32)
    ang = np.outer(np.arange(seqlen, dtype=np.float32), f).astype(np.float32)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rope(x, cs, inverse=False):
    """x[..., T?, d] with adjacent pairs; cs = (cos, sin) rows matching x's position axis."""
    c, s = cs
    if inverse:
        s = -s
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- model
class Ref:
    def __init__(self, d):
        self.ck = ck = Ckpt(d)
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.c = c = cfg.get("text_config", cfg)
        self.D = c["hidden_size"]
        self.eps = c["rms_norm_eps"]
        self.ratios = c["compress_ratios"]
        self.kv_src = c["kv_source_layer_ids"]
        self.idx_src = c["index_source_layer_ids"]
        self.cand = c.get("candidate_source_layer_id", -1)
        eb = open(os.path.join(d, "espetero_engram.bin"), "rb").read()
        h = struct.unpack("<8i", eb[8:40])
        self.e_vocab, self.e_layers, self.e_n, self.e_heads, self.e_pad = h[1], h[3], h[4], h[5], h[6]
        o = 40
        nm = self.e_layers * self.e_n
        self.e_mult = np.frombuffer(eb, "<i8", nm, o).reshape(self.e_layers, self.e_n)
        o += 8 * nm
        npr = self.e_layers * (self.e_n - 1) * self.e_heads
        self.e_primes = np.frombuffer(eb, "<i8", npr, o).reshape(self.e_layers, -1)
        o += 8 * npr
        self.e_map = np.frombuffer(eb, "<u4", self.e_vocab, o).astype(np.int64)
        self.e_offs = np.concatenate([np.zeros((self.e_layers, 1), np.int64), np.cumsum(self.e_primes, 1)[:, :-1]], 1)

    def engram_hashes(self, ids, g):
        T = len(ids)
        comp = self.e_map[np.asarray(ids)]
        toks = []
        for sh in range(self.e_n):
            src = np.array([comp[i - sh] if i - sh >= 0 else self.e_pad for i in range(T)], np.int64)
            toks.append(src)
        prod = [t * self.e_mult[g, i] for i, t in enumerate(toks)]
        rolling = prod[0]
        cols = []
        for i in range(1, self.e_n):
            rolling = np.bitwise_xor(rolling, prod[i])
            for hh in range(self.e_heads):
                j = (i - 1) * self.e_heads + hh
                cols.append(rolling % self.e_primes[g, j] + self.e_offs[g, j])
        return np.stack(cols, -1)

    def engram(self, l, h, ids):
        ck, p = self.ck, f"layers.{l}.engram."
        g = self.c["engram_layer_ids"].index(l)
        rows = self.engram_hashes(ids, g)
        wraw, _, wshape = ck.raw(p + "embed.weight")
        sraw, _, sshape = ck.raw(p + "embed.scale")
        hd = wshape[1]
        e = np.zeros(rows.shape + (hd,), np.float32)
        for idx in np.ndindex(rows.shape):
            r = int(rows[idx])
            e[idx] = E4M3[np.asarray(wraw[r * hd:(r + 1) * hd])] * np.repeat(E8M0[np.asarray(sraw[r * (hd // 32):(r + 1) * (hd // 32)])], 32)
        kv = Lin(ck, p + "wkv")(e.reshape(len(ids), -1))
        D = self.D
        key = kv[:, : 4 * D].reshape(-1, 4, D)
        value = kv[:, 4 * D:]
        weight = ck.get(p + "q_weight") * ck.get(p + "k_weight")
        rstd = 1 / np.sqrt((h * h).mean(-1) + self.eps) / np.sqrt((key * key).mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * D ** -0.5
        gate = sigmoid(np.copysign(np.sqrt(np.maximum(np.abs(dot), 1e-6)), dot))
        return (h + gate[..., None] * value[:, None, :]).astype(np.float32)

    def hc_mixes(self, x, fn, scale, base):
        T = x.shape[0]
        xf = x.reshape(T, -1)
        rs = 1 / np.sqrt((xf * xf).mean(-1, keepdims=True) + self.eps)
        mixes = (xf @ fn.T) * rs
        hc, eps, iters = 4, self.c["hc_eps"], self.c["hc_sinkhorn_iters"]
        pre = sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
        post = 2 * sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
        comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).reshape(T, hc, hc)
        comb = np.exp(comb - comb.max(-1, keepdims=True))
        comb = comb / comb.sum(-1, keepdims=True) + eps
        comb = comb / (comb.sum(-2, keepdims=True) + eps)
        for _ in range(iters - 1):
            comb = comb / (comb.sum(-1, keepdims=True) + eps)
            comb = comb / (comb.sum(-2, keepdims=True) + eps)
        return pre.astype(np.float32), post.astype(np.float32), comb.astype(np.float32)

    def attention(self, l, x, shared):
        ck, c, p = self.ck, self.c, f"layers.{l}.attn."
        T = x.shape[0]
        H, HD, RD, win = c["num_attention_heads"], c["head_dim"], c["qk_rope_head_dim"], c["sliding_window"]
        ratio = self.ratios[l]
        if ratio:
            cs = freqs(RD, T, c["rope_scaling"]["original_max_position_embeddings"], c["compress_rope_theta"],
                       c["rope_scaling"]["factor"], c["rope_scaling"]["beta_fast"], c["rope_scaling"]["beta_slow"])
        else:
            cs = freqs(RD, T, 0, c["rope_theta"], 1, 32, 1)
        qr = rmsnorm(Lin(ck, p + "wq_a")(x), ck.get(p + "q_norm.weight"), self.eps)
        q = Lin(ck, p + "wq_b")(qr).reshape(T, H, HD)
        q[..., -RD:] = rope(q[..., -RD:], (cs[0][:, None, :], cs[1][:, None, :]))
        # window kv
        kv = rmsnorm(Lin(ck, p + "wkv")(x), ck.get(p + "kv_norm.weight"), self.eps)
        kv[..., -RD:] = rope(kv[..., -RD:], cs)
        kv = act_quant(kv)
        end = np.arange(T)[:, None]
        widx = np.clip(end - win + 1, 0, None) + np.arange(min(T, win))
        widx = np.where(widx > end, -1, widx)
        keys = kv
        idxs = widx
        if ratio:
            latent = None
            if l in self.kv_src:
                cp = p + "compressor."
                if ratio == 1:
                    latent = rmsnorm(x @ ck.get(cp + "wkv.weight").T, ck.get(cp + "norm.weight"), self.eps)
                else:
                    kvc = x @ ck.get(cp + "wkv.weight").T
                    sc = x @ ck.get(cp + "wgate.weight").T
                    cut = T - T % ratio
                    kvc = kvc[:cut].reshape(-1, ratio, HD)
                    sc = sc[:cut].reshape(-1, ratio, HD)
                    e = np.exp(sc - sc.max(1, keepdims=True))
                    latent = (kvc * (e / e.sum(1, keepdims=True))).sum(1)
                    latent = rmsnorm(latent, ck.get(cp + "norm.weight"), self.eps)
                ng = latent.shape[0]
                gcs = (cs[0][: T - T % ratio: ratio], cs[1][: T - T % ratio: ratio])
                if l in self.idx_src:
                    ip = p + "indexer."
                    k = rmsnorm(latent @ ck.get(ip + "wk.weight").T, ck.get(ip + "k_norm.weight"), self.eps)
                    k[..., -RD:] = rope(k[..., -RD:], gcs)
                    shared["index_k"] = fp4_act_quant(k)
                lat = latent.copy()
                lat[..., -RD:] = rope(lat[..., -RD:], gcs)
                shared["comp"] = fp4_act_quant(lat, 16, True)
            ncomp = T // ratio
            if l in self.idx_src:
                if ncomp == 0:
                    cidx = np.zeros((T, 0), np.int64)
                else:
                    ip = p + "indexer."
                    IH, IHD = c["index_n_heads"], c["index_head_dim"]
                    qi = Lin(ck, ip + "wq_b")(qr).reshape(T, IH, IHD)
                    qi[..., -RD:] = rope(qi[..., -RD:], (cs[0][:, None, :], cs[1][:, None, :]))
                    qi = fp4_act_quant(qi)
                    w = (x @ ck.get(ip + "weights_proj.weight").T) * (IHD ** -0.5 * IH ** -0.5)
                    ik = shared["index_k"][:ncomp]
                    score = np.einsum("shd,td->sht", qi, ik)
                    score = (np.maximum(score, 0) * w[:, :, None]).sum(1)
                    clen = (np.arange(1, T + 1) // ratio)[:, None]
                    score = np.where(np.arange(ncomp)[None, :] >= clen, -np.inf, score)
                    if l == self.cand:
                        pass  # candidates only restrict beyond candidate_topk_blocks * block_size positions
                    topk = min(c["index_topk"], ncomp)
                    part = np.argsort(-score, axis=-1, kind="stable")[:, :topk]
                    part = np.sort(part, -1)
                    cidx = np.where(part < clen, part + T, -1)
                shared["topk"] = cidx
            comp = shared["comp"][:ncomp]
            keys = np.concatenate([kv, comp], 0)
            idxs = np.concatenate([widx, shared["topk"]], -1)
        # sparse attention
        sink = ck.get(p + "attn_sink")
        o = np.zeros((T, H, HD), np.float32)
        scale = HD ** -0.5
        for t in range(T):
            ii = idxs[t][idxs[t] >= 0]
            kk = keys[ii]  # [n, HD]
            s = (q[t] @ kk.T) * scale  # [H, n]
            m = s.max(-1, keepdims=True)
            e = np.exp(s - m)
            den = e.sum(-1, keepdims=True) + np.exp(sink[:, None] - m)
            o[t] = (e / den) @ kk
        o[..., -RD:] = rope(o[..., -RD:], (cs[0][:, None, :], cs[1][:, None, :]), inverse=True)
        G, OR = c["o_groups"], c["o_lora_rank"]
        o = act_quant(o.reshape(T, G, -1))
        wo_a = Lin(ck, p + "wo_a").dense().reshape(G, OR, -1)
        o = np.einsum("sgd,grd->sgr", o, wo_a)
        return Lin(ck, p + "wo_b")(o.reshape(T, -1))

    def moe(self, l, x):
        ck, c, p = self.ck, self.c, f"layers.{l}.ffn."
        T = x.shape[0]
        lim = c["swiglu_limit"]
        scores = x @ ck.get(p + "gate.weight").T
        scores = np.where(scores > 20, scores, np.log1p(np.exp(np.minimum(scores, 20))))
        scores = np.sqrt(scores)
        bias = ck.get(p + "gate.bias")
        k = c["num_experts_per_tok"]
        idx = np.argsort(-(scores + bias), axis=-1, kind="stable")[:, :k]
        if self.force_routes is not None:
            forced = self.force_routes[l]
            b = scores + bias
            for t in range(T):
                if set(forced[t]) != set(idx[t]):
                    gap = np.sort(b[t])[::-1][:k + 2]
                    print(f"  route l{l} t{t}: forced {sorted(forced[t].tolist())} over {sorted(idx[t].tolist())}; "
                          f"margin k/k+1 {gap[k - 1] - gap[k]:.2e}")
            idx = forced
        w = np.take_along_axis(scores, idx, 1)
        w = w / (w.sum(-1, keepdims=True) + 1e-20) * c["routed_scaling_factor"]

        def expert(q, xs, wt=None):
            g = Lin(ck, q + "w1")(xs)
            u = Lin(ck, q + "w3")(xs)
            u = np.clip(u, -lim, lim)
            g = np.minimum(g, lim)
            h = g / (1 + np.exp(-g)) * u
            if wt is not None:
                h = wt * h
            return Lin(ck, q + "w2")(h)

        y = np.zeros_like(x)
        for e in np.unique(idx):
            rows, top = np.nonzero(idx == e)
            y[rows] += expert(p + f"experts.{e}.", x[rows], w[rows, top][:, None])
        y += expert(p + "shared_experts.", x)
        return y.astype(np.float32), idx

    def forward(self, ids, layers, out):
        ck, D = self.ck, self.D
        T = len(ids)
        h = np.repeat(ck.get("embed.weight")[np.asarray(ids)][:, None, :], 4, 1).astype(np.float32)
        pre_mix = np.zeros((T, 4), np.float32)
        pre_mix[:, 0] = 1
        shared = {}
        routes = {}
        for l in range(layers):
            p = f"layers.{l}."
            if l in self.c["engram_layer_ids"] and not self.no_engram:
                h = self.engram(l, h, ids)
            res = h
            ap, apost, acomb = self.hc_mixes(h, ck.get(p + "hc_attn_fn"), ck.get(p + "hc_attn_scale"), ck.get(p + "hc_attn_base"))
            x = (pre_mix[..., None] * h).sum(1)
            x = rmsnorm(x, ck.get(p + "attn_norm.weight"), self.eps)
            x = self.attention(l, x, shared)
            x.astype(np.float32).tofile(os.path.join(out, f"a{l}.f32"))
            h = apost[..., None] * x[:, None, :] + (acomb[..., None] * res[:, :, None, :]).sum(1)
            res = h
            fp, fpost, fcomb = self.hc_mixes(h, ck.get(p + "hc_ffn_fn"), ck.get(p + "hc_ffn_scale"), ck.get(p + "hc_ffn_base"))
            x = (ap[..., None] * h).sum(1)
            x = rmsnorm(x, ck.get(p + "ffn_norm.weight"), self.eps)
            x, routes[l] = self.moe(l, x)
            x.astype(np.float32).tofile(os.path.join(out, f"f{l}.f32"))
            h = (fpost[..., None] * x[:, None, :] + (fcomb[..., None] * res[:, :, None, :]).sum(1)).astype(np.float32)
            pre_mix = fp
            h.astype(np.float32).tofile(os.path.join(out, f"h{l}.f32"))
            np.asarray(routes[l], np.int32).tofile(os.path.join(out, f"route{l}.i32"))
            print(f"layer {l}: |h| {np.abs(h).mean():.4f}", flush=True)
        if layers == self.c["num_hidden_layers"]:
            x = (pre_mix[-1:, :, None] * h[-1:]).sum(1)
            x = rmsnorm(x, ck.get("norm.weight"), self.eps)
            logits = x @ ck.get("head.weight").T
            logits.astype(np.float32).tofile(os.path.join(out, "logits.f32"))
            top = np.argsort(-logits[0])[:5]
            print("top5", top.tolist(), logits[0][top].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--out", default="ds4ref")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--force-routes", default="", help="Espetero dump dir: use its routing (p*_route<l>.i32)")
    ap.add_argument("--no-quant", action="store_true", help="no activation rounding (parity without rounding flips)")
    a = ap.parse_args()
    ids = [int(t) for t in a.tokens.split(",")]
    os.makedirs(a.out, exist_ok=True)
    global NOQUANT
    NOQUANT = a.no_quant
    r = Ref(a.model)
    r.no_engram = a.no_engram
    r.force_routes = None
    if a.force_routes:
        import glob
        r.force_routes = {}
        for l in range(a.layers or r.c["num_hidden_layers"]):
            files = sorted(glob.glob(os.path.join(a.force_routes, f"p*_route{l}.i32")),
                           key=lambda f: int(os.path.basename(f).split("_")[0][1:]))
            r.force_routes[l] = np.concatenate([np.fromfile(f, np.int32) for f in files]).reshape(-1, r.c["num_experts_per_tok"])
    r.forward(ids, a.layers or r.c["num_hidden_layers"], a.out)


if __name__ == "__main__":
    main()
