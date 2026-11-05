#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero
"""Cross-check of tools/glm5_ref.py (Espetero's numpy transcription of transformers'
glm5_next) against an independent implementation: mlx-vlm's glm5_next decoder layers
(0.7.4), layer by layer on the same checkpoint tensors.

    python tools/glm5_ref.py MODEL --tokens ... --layers N --out REF     # first
    python tests/ref/glm5_mlx_check.py MODEL REF                        # then (MLX: through gpuq)

Each mlx-vlm layer l gets the reference's input streams (REF/h<l-1>.f32, the
embedding rows for l = 0); its mixing branch (KDA / DSA, whole sequence) is
compared with the reference's streams after that branch (rebuilt from REF/a<l>.f32),
its MLP branch (row-independent: run in chunks of 256 rows) is fed the reference's
streams and compared with REF/h<l>.f32, so a disagreement points at one branch
of one layer. Weights: the checkpoint's FP8 tensors
dequantized with their 128 x 128 block scales in float32; a MoE layer gets only
the experts the reference routed to (REF/route<l>.i32), its router's indices
remapped onto them, in float16 (mlx-vlm stacks all 288 experts: 29 GB per layer in float32).
"""
import argparse
import json
import os
import sys
from dataclasses import fields

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import glm5_ref  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx_vlm.models.glm5_next import language as lang  # noqa: E402
from mlx_vlm.models.glm5_next.config import TextConfig  # noqa: E402
from mlx_vlm.models.switch_layers import SwitchGLU  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ref")
    a = ap.parse_args()
    global REF
    REF = a.ref
    case = json.load(open(os.path.join(a.ref, "case.json")))
    ids, N = case["ids"], case["layers"]
    T = len(ids)
    r = glm5_ref.Ref(a.model)
    ck = r.ck
    tc = dict(r.c)
    tc["num_hidden_layers"] = N
    for k in ("layer_types", "mlp_layer_types", "indexer_types"):
        if tc.get(k):
            tc[k] = tc[k][:N]
    known = {f.name for f in fields(TextConfig)}
    cfg = TextConfig(**{k: v for k, v in tc.items() if k in known})
    model = lang.LanguageModel(cfg)
    D = r.D

    # HF tensors of the first N layers (dequantized), through mlx-vlm's own sanitize
    w = {}
    for name in ck.loc:
        if not name.startswith("model.language_model.layers."):
            continue
        l = int(name.split(".")[3])
        if l >= N or ".mlp.experts." in name or name.endswith("weight_scale_inv"):
            continue
        base = name[: -len(".weight")] if name.endswith(".weight") else None
        if base and ck.loc[name][3] == "F8_E4M3":
            arr = glm5_ref.Lin(ck, base).dense()
        else:
            arr = ck.get(name).astype(np.float32)
        w["language_model.model." + name[len("model.language_model."):]] = mx.array(arr)
    w = model.sanitize(w)
    # sanitize keeps the outer Model's "language_model." prefix; this is the LanguageModel itself
    w = {k[len("language_model."):] if k.startswith("language_model.") else k: v for k, v in w.items()}
    from mlx.utils import tree_flatten
    params = {k for k, _ in tree_flatten(model.parameters())}
    unknown = sorted(set(w) - params)
    if unknown:
        sys.exit(f"checkpoint tensors with no parameter: {unknown[:8]}")
    # every parameter of the layers must come from the checkpoint, but the routed experts (set below)
    missing = sorted(k for k in params if k.startswith("model.layers.") and k not in w and ".switch_mlp." not in k)
    if missing:
        sys.exit(f"parameters left at their random initialization: {missing[:8]}")
    model.load_weights(list(w.items()), strict=False)

    # MoE layers: only the routed experts, the router's indices remapped
    for l, layer in enumerate(model.model.layers):
        if not isinstance(layer.mlp, lang.Glm5NextMoE):
            continue
        route = np.fromfile(os.path.join(a.ref, f"route{l}.i32"), np.int32).reshape(T, -1)
        used = np.unique(route)
        lut = np.full(cfg.n_routed_experts, 0, np.int32)
        lut[used] = np.arange(len(used))
        sw = SwitchGLU(D, cfg.moe_intermediate_size, len(used), activation=lang.LimitedSwiGLU(cfg.swiglu_limit))
        p = f"model.language_model.layers.{l}.mlp.experts."
        for nm in ("gate_proj", "up_proj", "down_proj"):  # float16 stacks (2^-12 relative): ~5 GB per matrix for 288 experts
            first = glm5_ref.Lin(ck, p + f"{used[0]}.{nm}").dense()
            stack = np.empty((len(used),) + first.shape, np.float16)
            for i, e in enumerate(used):
                stack[i] = first if i == 0 else glm5_ref.Lin(ck, p + f"{e}.{nm}").dense()
            getattr(sw, nm).weight = mx.array(stack)
            mx.eval(getattr(sw, nm).weight)
            del stack
        layer.mlp.switch_mlp = sw
        gate = layer.mlp.gate
        lut_mx = mx.array(lut)

        class Remap:
            def __init__(self, g):
                self.g = g

            def __call__(self, x):
                inds, scores = self.g(x)
                return lut_mx[inds], scores

        layer.mlp.gate = Remap(gate)
        layer.mlp.gate_route = route  # kept for the comparison below

    emb = ck.get("model.language_model.embed_tokens.weight")[np.asarray(ids)].astype(np.float32)
    h_in = np.repeat(emb[:, None, :], cfg.hc_mult, 1)
    worst = 0.0
    topk = None
    CH = 256  # rows per chunk of the MLP branch (row-independent; mlx-vlm's MoE over all rows at once peaks at ~100 GB)

    def report(l, what, got, base, want):
        nonlocal worst
        dw, dg = want - base, got - base  # the branch's own contribution
        rows = np.sqrt(((dg - dw) ** 2).sum((1, 2)) / np.maximum((dw ** 2).sum((1, 2)), 1e-30))
        worst = max(worst, float(rows.max()))
        print(f"layer {l} {what}: per-row relative L2 of its update max {rows.max():.3g}, median {np.median(rows):.3g}")

    for l, layer in enumerate(model.model.layers):
        p = f"model.language_model.layers.{l}."
        # mixing branch (KDA / DSA) over the whole sequence, against the reference's streams after it
        if layer.is_linear:
            mid = layer.attn_hc.apply_branch(mx.array(h_in[None]), layer.input_layernorm, layer.self_attn, None, None)
        else:
            mid, topk = layer.attn_hc.apply_branch(mx.array(h_in[None]), layer.input_layernorm, layer.self_attn, None, None, topk)
        mx.eval(mid)
        got_mid = np.array(mid[0]).astype(np.float32)
        a_ref = np.fromfile(os.path.join(REF, f"a{l}.f32"), np.float32).reshape(T, D)
        post, comb, _ = r.hc_coeffs(h_in, p, "attn")
        want_mid = r.hc_post(a_ref, h_in, post, comb)
        report(l, "mixing", got_mid, h_in, want_mid)
        # MLP branch in row chunks, from the reference's streams
        outs = []
        for r0 in range(0, T, CH):
            o = layer.ffn_hc.apply_branch(mx.array(want_mid[None, r0:r0 + CH]), layer.post_attention_layernorm, layer.mlp)
            mx.eval(o)
            outs.append(np.array(o[0]).astype(np.float32))
        want = np.fromfile(os.path.join(REF, f"h{l}.f32"), np.float32).reshape(T, cfg.hc_mult, D)
        report(l, "MLP", np.concatenate(outs, 0), want_mid, want)
        h_in = want
    print("OK" if worst < 1e-2 else "MISMATCH", f"(worst per-row relative L2 {worst:.3g})")
    return 0 if worst < 1e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
