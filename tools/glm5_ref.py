#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero
"""GLM-5.3-Flash (glm5_next) reference forward in numpy, for Espetero's parity checks.

A line-by-line transcription of transformers' modeling_glm5_next.py
(Glm5NextTextModel.forward, batch 1, no cache, no padding) on the checkpoint's
own FP8 (128 x 128 fp32 block scales) and bf16 tensors, in float32 throughout
(transformers runs the projections in bf16; docs/glm5.md 11). The KDA layers
use the recurrent form, the DSA layers the explicit k / v of kv_b_proj with
the k-pool indexer's selection (every token below 2052 visible ones).

    python3 -I tools/glm5_ref.py MODEL_DIR --tokens 154822,154824,... [--layers N] [--out DIR]
    python3 -I tools/glm5_ref.py MODEL_DIR --text "Hello" [--chat] ...

MODEL_DIR may also hold unsloth's GGUF split files (with the checkpoint's config.json copied next
to them): the reference then runs on their weights, dequantized with gguf-py (GGUF_PY=dir if it
is not installed). MODEL_DIR may be a partial checkpoint (tests/ref/glm5_ref.py fake-ckpt
--fill-layers 0-3): run --layers up to the layers present. Writes DIR/h<l>.f32
(the residual streams after layer l, [T, 4, D]), DIR/a<l>.f32 and DIR/f<l>.f32
(the attention / KDA block's and the MLP's outputs, [T, D]), DIR/route<l>.i32
([T, 8] experts, MoE layers), DIR/sel<l>.i32 ([T, 2051] selected token ids, -1
padded, DSA layers) and, with all layers, DIR/logits.f32 ([T, V]). Needs numpy.
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np


# ---------------------------------------------------------------- checkpoint
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


class Ckpt:
    def __init__(self, d):
        self.d = d
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.loc, self.maps = {}, {}
        for f in sorted(set(idx.values())):
            path = os.path.join(d, f)
            if not os.path.exists(path):
                continue
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            self.maps[f] = np.memmap(path, dtype=np.uint8, mode="r")
            for k, v in h.items():
                if k != "__metadata__":
                    a, b = v["data_offsets"]
                    self.loc[k] = (f, 8 + n + a, 8 + n + b, v["dtype"], v["shape"])
        filled = os.path.join(d, ".filled.json")  # a partial checkpoint: only these tensors hold data
        self.filled = set(json.load(open(filled))) if os.path.exists(filled) else None

    def has(self, name):
        return name in self.loc and (self.filled is None or name in self.filled)

    def get(self, name):
        if not self.has(name):
            raise KeyError(f"{name}: not in the checkpoint (or not filled in this partial one)")
        f, a, b, dt, shape = self.loc[name]
        r = self.maps[f][a:b]
        if dt == "BF16":
            return (r.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if dt == "F32":
            return np.array(r.view(np.float32).reshape(shape))
        if dt == "F8_E4M3":
            return E4M3[r].reshape(shape)
        raise ValueError(f"{name}: dtype {dt}")


class GgufCkpt:
    """unsloth's GGUF split files under the transformers names (docs/glm5.md 8.2), every tensor dequantized
    with gguf-py (`pip install gguf`, or GGUF_PY=dir holding its gguf package): the reference on the very
    weights a GGUF run uses."""

    BLOCK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144),
             13: (256, 176), 14: (256, 210), 17: (256, 74), 18: (256, 98), 20: (32, 18), 21: (256, 110), 22: (256, 82),
             23: (256, 136)}

    def __init__(self, d):
        if os.environ.get("GGUF_PY"):
            sys.path.insert(0, os.environ["GGUF_PY"])
        from gguf.quants import dequantize  # noqa: F401 (checked here, used in get)
        self.t, self.maps = {}, {}
        for f in sorted(x for x in os.listdir(d) if x.endswith(".gguf")):
            path = os.path.join(d, f)
            m = np.memmap(path, dtype=np.uint8, mode="r")
            self.maps[f] = m
            for name, (dims, typ, off, nb) in _gguf_tensors(m).items():
                self.t[name] = (f, dims, typ, off, nb)
        self.loc = {}
        self.filled = None

    @staticmethod
    def _map(name):
        """transformers name -> (GGUF name, transform)"""
        import re
        if name == "model.language_model.embed_tokens.weight":
            return "token_embd.weight", None
        if name == "model.language_model.norm.weight":
            return "output_norm.weight", None
        if name == "lm_head.weight":
            return "output.weight", None
        m = re.match(r"model\.language_model\.layers\.(\d+)\.(.*)$", name)
        b, rest = f"blk.{m.group(1)}.", m.group(2)
        e = re.match(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$", rest)
        if e:
            return b + f"ffn_{e.group(2)}_exps.weight", ("expert", int(e.group(1)))
        table = {
            "input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
            "hc_attn_fn": "hc_attn_fn.weight", "hc_attn_base": "hc_attn_base.weight", "hc_attn_scale": "hc_attn_scale.weight",
            "hc_ffn_fn": "hc_ffn_fn.weight", "hc_ffn_base": "hc_ffn_base.weight", "hc_ffn_scale": "hc_ffn_scale.weight",
            "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
            "self_attn.q_conv1d.weight": "ssm_conv1d_q.weight", "self_attn.k_conv1d.weight": "ssm_conv1d_k.weight",
            "self_attn.v_conv1d.weight": "ssm_conv1d_v.weight", "self_attn.f_a_proj.weight": "ssm_f_a.weight",
            "self_attn.f_b_proj.weight": "ssm_f_b.weight", "self_attn.g_a_proj.weight": "ssm_g_a.weight",
            "self_attn.g_b_proj.weight": "ssm_g_b.weight", "self_attn.b_proj.weight": "ssm_beta.weight",
            "self_attn.dt_bias": "ssm_dt.bias", "self_attn.o_norm.weight": "ssm_norm.weight",
            "self_attn.q_a_proj.weight": "attn_q_a.weight", "self_attn.q_a_layernorm.weight": "attn_q_a_norm.weight",
            "self_attn.q_b_proj.weight": "attn_q_b.weight", "self_attn.kv_a_proj_with_mqa.weight": "attn_kv_a_mqa.weight",
            "self_attn.kv_a_layernorm.weight": "attn_kv_a_norm.weight",
            "self_attn.indexer.wq_b.weight": "indexer.attn_q_b.weight", "self_attn.indexer.wk.weight": "indexer.attn_k.weight",
            "self_attn.indexer.k_norm.weight": "indexer.k_norm.weight", "self_attn.indexer.k_norm.bias": "indexer.k_norm.bias",
            "self_attn.indexer.weights_proj.weight": "indexer.proj.weight",
            "self_attn.indexer.index_kpool_compress_gate": "indexer_compressor_gate.weight",
            "self_attn.indexer.index_kpool_compress_ape": "indexer_compressor_ape.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight",
            "mlp.gate.weight": "ffn_gate_inp.weight", "mlp.gate.e_score_correction_bias": "exp_probs_b.bias",
            "mlp.shared_experts.gate_proj.weight": "ffn_gate_shexp.weight", "mlp.shared_experts.up_proj.weight": "ffn_up_shexp.weight",
            "mlp.shared_experts.down_proj.weight": "ffn_down_shexp.weight",
            "eh_proj.weight": "nextn.eh_proj.weight", "enorm.weight": "nextn.enorm.weight",
            "hnorm.weight": "nextn.hnorm.weight", "shared_head.norm.weight": "nextn.shared_head_norm.weight",
        }
        if rest == "self_attn.A_log":
            return b + "ssm_a", ("neglog",)
        if rest == "self_attn.kv_b_proj.weight":
            return b + "attn_k_b.weight", ("kv_b",)
        if rest in table:
            return b + table[rest], None
        raise KeyError(name)

    def _raw(self, gname, row0=0, rows=None):
        from gguf.quants import dequantize
        f, dims, typ, off, nb = self.t[gname]
        K = dims[0]
        n_rows = int(np.prod(dims[1:])) if len(dims) > 1 else 1
        bv, bb = self.BLOCK[typ]
        rb = K // bv * bb
        rows = n_rows - row0 if rows is None else rows
        raw = np.asarray(self.maps[f][off + row0 * rb: off + (row0 + rows) * rb]).reshape(rows, rb)
        if typ == 0:
            return raw.view(np.float32).reshape(rows, K).copy()
        return dequantize(raw, typ).astype(np.float32).reshape(rows, K)

    def has(self, name):
        try:
            g, _ = self._map(name)
        except (KeyError, AttributeError):
            return False
        return g in self.t

    def get(self, name):
        g, tr = self._map(name)
        f, dims, typ, off, nb = self.t[g]
        if tr and tr[0] == "expert":  # rows of expert e of a [K, N, E] tensor
            N = dims[1]
            return self._raw(g, tr[1] * N, N)
        if tr and tr[0] == "neglog":
            return np.log(-self._raw(g).reshape(-1)).astype(np.float32)
        if tr and tr[0] == "kv_b":  # rebuilt from the per-head halves: rows h*512+i = k_b[h][:, i], rows h*512+256+d = v_b[h][d]
            kb = self._raw(g).reshape(dims[2], dims[1], dims[0])                      # [H][512][256]
            vd = self.t[g.replace("attn_k_b", "attn_v_b")][1]
            vb = self._raw(g.replace("attn_k_b", "attn_v_b")).reshape(vd[2], vd[1], vd[0])  # [H][256][512]
            return np.concatenate([kb.transpose(0, 2, 1), vb], 1).reshape(-1, dims[1]).astype(np.float32)
        x = self._raw(g)
        if len(dims) == 1:
            return x.reshape(-1)
        if name.endswith("_conv1d.weight"):  # ggml [4, 1, C] -> [C, 1, 4]
            return x.reshape(dims[2], 1, dims[0])
        return x.reshape(int(np.prod(dims[1:])), dims[0])


def _gguf_tensors(m):
    """{name: (dims, type, absolute offset, nbytes)} of a mapped GGUF file."""
    p = [4]

    def take(fmt):
        v = struct.unpack_from(fmt, m, p[0])
        p[0] += struct.calcsize(fmt)
        return v[0]

    def string():
        n = take("<Q")
        v = bytes(m[p[0]:p[0] + n]).decode("utf-8", "replace")
        p[0] += n
        return v

    fmts = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = take("<I"), take("<Q")
            return [value(et) for _ in range(n)]
        return take(fmts[t])

    take("<I")
    nt, nkv = take("<Q"), take("<Q")
    align = 32
    for _ in range(nkv):
        k = string()
        v = value(take("<I"))
        if k == "general.alignment":
            align = v
    infos = []
    for _ in range(nt):
        name = string()
        dims = [take("<Q") for _ in range(take("<I"))]
        t = take("<I")
        infos.append((name, dims, t, take("<Q")))
    start = (p[0] + align - 1) // align * align
    out = {}
    for name, dims, t, off in infos:
        bv, bb = GgufCkpt.BLOCK[t]
        out[name] = (dims, t, start + off, int(np.prod(dims)) // bv * bb)
    return out


class Lin:
    """y = x W^T with W dequantized once (fp8: times its 128 x 128 block scale)."""

    def __init__(self, ck, name):
        self.ck, self.name, self._w = ck, name, None

    def dense(self):
        if self._w is None:
            w = self.ck.get(self.name + ".weight")
            if self.name + ".weight" in self.ck.loc and self.ck.loc[self.name + ".weight"][3] == "F8_E4M3":
                s = self.ck.get(self.name + ".weight_scale_inv")
                N, K = w.shape
                w = w * np.repeat(np.repeat(s, 128, 0)[:N], 128, 1)[:, :K]
            self._w = w.astype(np.float32)
        return self._w

    def __call__(self, x):
        return (x.astype(np.float32) @ self.dense().T).astype(np.float32)


# ---------------------------------------------------------------- pieces
def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def silu(x):
    return (x * sigmoid(x)).astype(np.float32)


def rmsnorm(x, w, eps):
    x = x.astype(np.float32)
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(eps))
    return (y if w is None else w * y).astype(np.float32)


def layernorm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + np.float32(eps)) * w + b).astype(np.float32)


def softmax(x, axis=-1):
    m = x.max(axis, keepdims=True)
    e = np.exp(x - m)
    return (e / e.sum(axis, keepdims=True)).astype(np.float32)


class Ref:
    def __init__(self, d):
        gguf = not os.path.exists(os.path.join(d, "model.safetensors.index.json"))
        self.ck = GgufCkpt(d) if gguf else Ckpt(d)
        root = json.load(open(os.path.join(d, "config.json")))
        tc = root.get("text_config", root)
        self.c = tc
        self.D = tc["hidden_size"]
        self.L = tc["num_hidden_layers"]
        self.eps = tc["rms_norm_eps"]
        la = tc.get("linear_attn_config", {})
        self.kH, self.kD = la.get("num_heads", 64), la.get("head_dim", 128)
        self.gate_lb = la.get("gate_lower_bound", -5.0)
        lt = tc["layer_types"]
        self.dsa = [t != "linear_attention" for t in lt]
        mt = tc.get("mlp_layer_types") or ["dense"] * tc.get("first_k_dense_replace", 3) + ["sparse"] * (self.L - 3)
        self.moe = [t == "sparse" for t in mt]
        self.H, self.NOPE, self.VD = tc["num_attention_heads"], tc["qk_nope_head_dim"], tc["v_head_dim"]
        self.IH, self.IHD, self.ITOPK, self.KPOOL = tc["index_n_heads"], tc["index_head_dim"], tc["index_topk"], tc["index_kpool"]
        self.E, self.K, self.rs = tc["n_routed_experts"], tc["num_experts_per_tok"], tc["routed_scaling_factor"]
        self.lim = tc["swiglu_limit"]
        self.hc, self.hc_iters, self.hc_eps = tc["hc_mult"], tc["hc_sinkhorn_iters"], tc["hc_eps"]

    def v(self, name):
        return self.ck.get(name).astype(np.float32)

    def P(self, l):
        return f"model.language_model.layers.{l}."

    # ---- mHC ([T] Glm5NextTextHyperConnection.forward)
    def hc_coeffs(self, h, p, site):
        T = h.shape[0]
        flat = h.reshape(T, -1)
        n = rmsnorm(flat, None, self.eps)
        m = (n @ self.v(p + f"hc_{site}_fn").T).astype(np.float32)  # [T, 24]
        base, sc = self.v(p + f"hc_{site}_base"), self.v(p + f"hc_{site}_scale")
        S = self.hc
        pre = sigmoid(m[:, :S] * sc[0] + base[:S]) + np.float32(self.hc_eps)
        post = 2 * sigmoid(m[:, S:2 * S] * sc[1] + base[S:2 * S])
        comb = softmax(m[:, 2 * S:].reshape(T, S, S) * sc[2] + base[2 * S:].reshape(S, S), -1) + np.float32(self.hc_eps)
        comb = comb / (comb.sum(-2, keepdims=True) + np.float32(self.hc_eps))
        for _ in range(self.hc_iters - 1):
            comb = comb / (comb.sum(-1, keepdims=True) + np.float32(self.hc_eps))
            comb = comb / (comb.sum(-2, keepdims=True) + np.float32(self.hc_eps))
        x = (pre[:, :, None] * h).sum(1).astype(np.float32)
        return post.astype(np.float32), comb.astype(np.float32), x

    @staticmethod
    def hc_post(y, h, post, comb):
        # post[k] y + sum_j comb[j][k] h_j    ([T]: post ⊗ y + comb^T @ residual)
        return (post[:, :, None] * y[:, None, :] + np.einsum("tjk,tjd->tkd", comb, h)).astype(np.float32)

    # ---- KDA ([T] Glm5NextTextLinearAttention, recurrent_kimi_delta_attention)
    def kda(self, x, l):
        p = self.P(l) + "self_attn."
        T = x.shape[0]
        H, Dk = self.kH, self.kD

        def conv(z, name):  # depthwise causal conv, kernel 4, then SiLU
            w = self.v(p + name + ".weight")[:, 0, :]  # [C, 4]
            zp = np.concatenate([np.zeros((3, z.shape[1]), np.float32), z], 0)
            out = sum(zp[i:i + T] * w[:, i] for i in range(4))
            return silu(out.astype(np.float32))

        q = conv(Lin(self.ck, p + "q_proj")(x), "q_conv1d").reshape(T, H, Dk)
        k = conv(Lin(self.ck, p + "k_proj")(x), "k_conv1d").reshape(T, H, Dk)
        v = conv(Lin(self.ck, p + "v_proj")(x), "v_conv1d").reshape(T, H, Dk)
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + np.float32(1e-6))
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + np.float32(1e-6))
        q = (q * np.float32(Dk ** -0.5)).astype(np.float32)
        f = Lin(self.ck, p + "f_b_proj")(Lin(self.ck, p + "f_a_proj")(x)) + self.v(p + "dt_bias")
        A = np.exp(self.v(p + "A_log"))
        g = (np.float32(self.gate_lb) * sigmoid(A[None, :, None] * f.reshape(T, H, Dk))).astype(np.float32)  # [T, H, Dk]
        beta = sigmoid(Lin(self.ck, p + "b_proj")(x))  # [T, H]
        S = np.zeros((H, Dk, Dk), np.float32)  # [head, k, v]
        o = np.zeros((T, H, Dk), np.float32)
        for t in range(T):
            S = S * np.exp(g[t])[:, :, None]
            kv = np.einsum("hkv,hk->hv", S, k[t])
            delta = (v[t] - kv) * beta[t][:, None]
            S = S + k[t][:, :, None] * delta[:, None, :]
            o[t] = np.einsum("hkv,hk->hv", S, q[t])
        z = Lin(self.ck, p + "g_b_proj")(Lin(self.ck, p + "g_a_proj")(x)).reshape(T, H, Dk)
        out = rmsnorm(o, self.v(p + "o_norm.weight"), self.eps) * sigmoid(z)
        return Lin(self.ck, p + "o_proj")(out.reshape(T, H * Dk))

    # ---- k-pool indexer ([T] Glm5NextTextIndexer.forward): per query, the selected token ids (-1 padded)
    def indexer(self, x, qr, l):
        p = self.P(l) + "self_attn.indexer."
        T = x.shape[0]
        q = Lin(self.ck, p + "wq_b")(qr).reshape(T, self.IH, self.IHD)
        k = layernorm(Lin(self.ck, p + "wk")(x), self.v(p + "k_norm.weight"), self.v(p + "k_norm.bias"), 1e-6)
        gate = (x @ self.v(p + "index_kpool_compress_gate").T).astype(np.float32)
        w = Lin(self.ck, p + "weights_proj")(x) * np.float32(self.IH ** -0.5)
        P = self.KPOOL
        npool = T // P  # complete pools (no padding: they start at token 0)
        width = self.ITOPK + (P - 1)
        sel = np.full((T, width), -1, np.int32)
        if npool:
            kk = k[:npool * P].reshape(npool, P, self.IHD)
            pr = softmax(gate[:npool * P].reshape(npool, P, self.IHD) + self.v(p + "index_kpool_compress_ape")[None], 1)
            K = (pr * kk).sum(1).astype(np.float32)  # [npool, 128]
            sc = np.maximum(np.einsum("thd,bd->thb", q, K) * np.float32(self.IHD ** -0.5), 0)
            score = np.einsum("th,thb->tb", w, sc).astype(np.float32)  # [T, npool]
        nsel = self.ITOPK // P
        for t in range(T):
            nvis = (t + 1) // P  # pools whose last token is visible
            ids = []
            if nvis:
                if nvis <= nsel:
                    chosen = np.arange(nvis)
                else:  # ties: the lowest pools (torch.topk's order is unspecified)
                    chosen = np.sort(np.argsort(-score[t, :nvis], kind="stable")[:nsel])
                ids = (chosen[:, None] * P + np.arange(P)).reshape(-1).tolist()
            ids += list(range(nvis * P, t + 1))  # the incomplete tail, always selected
            sel[t, :len(ids)] = ids
        return sel

    # ---- DSA ([T] Glm5NextTextAttention: nope MLA, dense over the selection)
    def dsa_attn(self, x, l):
        p = self.P(l) + "self_attn."
        T = x.shape[0]
        H, NOPE, VD = self.H, self.NOPE, self.VD
        qr = rmsnorm(Lin(self.ck, p + "q_a_proj")(x), self.v(p + "q_a_layernorm.weight"), self.eps)
        q = Lin(self.ck, p + "q_b_proj")(qr).reshape(T, H, NOPE)
        c = rmsnorm(Lin(self.ck, p + "kv_a_proj_with_mqa")(x), self.v(p + "kv_a_layernorm.weight"), self.eps)
        kv = (c @ self.v(p + "kv_b_proj.weight").T).reshape(T, H, NOPE + VD)
        k, vv = kv[..., :NOPE], kv[..., NOPE:]
        sel = self.indexer(x, qr, l)
        mask = np.full((T, T), -np.inf, np.float32)
        for t in range(T):
            ids = sel[t][sel[t] >= 0]
            mask[t, ids] = 0
        s = np.einsum("thd,jhd->htj", q, k) * np.float32(NOPE ** -0.5) + mask[None]
        a = softmax(s, -1)
        o = np.einsum("htj,jhd->thd", a, vv).reshape(T, H * VD)
        return Lin(self.ck, p + "o_proj")(o), sel

    # ---- MLPs ([T] Glm5NextTextMLP / MoE)
    def mlp(self, x, p):
        g = np.minimum(Lin(self.ck, p + "gate_proj")(x), np.float32(self.lim))
        u = np.clip(Lin(self.ck, p + "up_proj")(x), -self.lim, self.lim)
        return Lin(self.ck, p + "down_proj")(silu(g) * u)

    def moe_block(self, x, l):
        p = self.P(l) + "mlp."
        logits = (x.astype(np.float32) @ self.v(p + "gate.weight").T).astype(np.float32)
        s = sigmoid(logits)
        choice = s + self.v(p + "gate.e_score_correction_bias")
        route = np.argsort(-choice, axis=-1, kind="stable")[:, :self.K].astype(np.int32)
        w = np.take_along_axis(s, route, -1)
        w = w / (w.sum(-1, keepdims=True) + np.float32(1e-20)) * np.float32(self.rs)
        y = np.zeros_like(x)
        for e in np.unique(route):
            rows, slots = np.nonzero(route == e)
            y[rows] += self.mlp(x[rows], p + f"experts.{e}.") * w[rows, slots][:, None]
        return (y + self.mlp(x, p + "shared_experts.")).astype(np.float32), route

    # ---- the NextN / MTP layer (layers.<L>; llama.cpp glm5-next.cpp graph_mtp, mlx-vlm's glm5_next_mtp: docs/glm5.md 6)
    def mtp(self, h, next_ids, out):
        """h [T, D]: the trunk's final-normed hidden per row; next_ids [T]: the token after each row.
        Returns the MTP's logits [T, V] and its normed output (the next draft step's hidden)."""
        l = self.L
        p = self.P(l)
        emb = self.ck.get("model.language_model.embed_tokens.weight")
        e = emb[np.asarray(next_ids, np.int64)].astype(np.float32)
        u = np.concatenate([rmsnorm(e, self.v(p + "enorm.weight"), self.eps), rmsnorm(h, self.v(p + "hnorm.weight"), self.eps)], -1)
        u = Lin(self.ck, p + "eh_proj")(u)
        a, sel = self.dsa_attn(rmsnorm(u, self.v(p + "input_layernorm.weight"), self.eps), l)
        r = u + a
        f, route = self.moe_block(rmsnorm(r, self.v(p + "post_attention_layernorm.weight"), self.eps), l)
        r = r + f
        hn = rmsnorm(r, self.v(p + "shared_head.norm.weight"), self.eps)
        logits = (hn @ self.v("lm_head.weight").T).astype(np.float32)
        for name, x in (("mtp_u", u), ("mtp_a", a), ("mtp_f", f), ("mtp_h", hn), ("mtp_logits", logits)):
            x.tofile(os.path.join(out, name + ".f32"))
        route.tofile(os.path.join(out, "mtp_route.i32"))
        sel.tofile(os.path.join(out, "mtp_sel.i32"))
        return logits, hn

    # ---- forward
    def forward(self, ids, layers, out):
        ids = np.asarray(ids, np.int64)
        T = len(ids)
        emb = self.ck.get("model.language_model.embed_tokens.weight")
        h = np.repeat(emb[ids].astype(np.float32)[:, None, :], self.hc, 1)
        for l in range(layers):
            t0 = time.time()
            p = self.P(l)
            post, comb, x = self.hc_coeffs(h, p, "attn")
            x = rmsnorm(x, self.v(p + "input_layernorm.weight"), self.eps)
            if self.dsa[l]:
                y, sel = self.dsa_attn(x, l)
                sel.tofile(os.path.join(out, f"sel{l}.i32"))
            else:
                y = self.kda(x, l)
            y.tofile(os.path.join(out, f"a{l}.f32"))
            h = self.hc_post(y, h, post, comb)
            post, comb, x = self.hc_coeffs(h, p, "ffn")
            x = rmsnorm(x, self.v(p + "post_attention_layernorm.weight"), self.eps)
            if self.moe[l]:
                y, route = self.moe_block(x, l)
                route.tofile(os.path.join(out, f"route{l}.i32"))
            else:
                y = self.mlp(x, p + "mlp.")
            y.tofile(os.path.join(out, f"f{l}.f32"))
            h = self.hc_post(y, h, post, comb)
            h.tofile(os.path.join(out, f"h{l}.f32"))
            print(f"layer {l} ({'DSA' if self.dsa[l] else 'KDA'}, {'MoE' if self.moe[l] else 'dense'}): "
                  f"{time.time() - t0:.1f} s, |h| rms {np.sqrt((h * h).mean()):.4g}", file=sys.stderr)
        self.last_h = h
        if layers == self.L:
            hn = rmsnorm(h.mean(1), self.v("model.language_model.norm.weight"), self.eps)
            logits = (hn @ self.v("lm_head.weight").T).astype(np.float32)
            logits.tofile(os.path.join(out, "logits.f32"))
            return logits
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--tokens", help="comma-separated token ids")
    ap.add_argument("--text", help="text to tokenize (needs the tokenizers package)")
    ap.add_argument("--chat", action="store_true", help="wrap --text as one user turn of GLM's chat format")
    ap.add_argument("--layers", type=int, default=0, help="first N layers (default: all)")
    ap.add_argument("--out", default="glm5_ref_out")
    ap.add_argument("--mtp", action="store_true",
                    help="also run the MTP layer on the final-normed hidden of the last layer run (rows 0..T-2 paired "
                         "with the next token): with --layers < all, a check of the layer's math on a proxy hidden")
    a = ap.parse_args()
    r = Ref(a.model)
    if a.tokens:
        ids = [int(x) for x in a.tokens.split(",")]
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
        text = f"[gMASK]<sop><|user|>{a.text}<|assistant|><think>" if a.chat else a.text
        ids = tok.encode(text, add_special_tokens=False).ids
    os.makedirs(a.out, exist_ok=True)
    json.dump({"ids": ids, "layers": a.layers or r.L}, open(os.path.join(a.out, "case.json"), "w"))
    print(f"{len(ids)} tokens", file=sys.stderr)
    r.forward(ids, a.layers or r.L, a.out)
    if a.mtp:
        hn = rmsnorm(r.last_h.mean(1), r.v("model.language_model.norm.weight"), r.eps)
        r.mtp(hn[:-1], ids[1:], a.out)
        print("mtp: done", file=sys.stderr)


if __name__ == "__main__":
    main()
