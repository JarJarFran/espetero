#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero
"""DeepSeek-V4.1 image path in numpy (inference/image_processor.py + vision.py),
for Espetero's parity checks: the patches and the run's embedding rows.

    python3 -I tools/dsv4_vision_ref.py MODEL_DIR IMAGE [pixels.f32 emb.f32]

Prints the planned grid; with Espetero's dumps (build/dsv4_run, DS4_RUN_IMAGE)
compares them. Needs numpy and PIL.
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv4_ref import Ckpt  # noqa: E402


def bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return u.view(np.float32)


def plan(width, height, p, r, max_tok, min_px):
    def grid(bh, bw):
        return math.ceil((bh // p) / r), math.ceil((bw // p) / r)

    def ntok(h, w):
        return h * (w + 1) + 2

    if 0 < width * height < min_px:
        ratio = (min_px / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    bw, bh = math.ceil(width / p) * p, math.ceil(height / p) * p
    lh, lw = grid(bh, bw)
    if ntok(lh, lw) > max_tok:
        rr = height / width
        mw = math.sqrt((max_tok - 2) / rr + 0.25) - 0.5
        mh = mw * rr
        cell = p * r
        if mw < 1.0:
            bh, bw = (max_tok - 2) // 2 * cell, cell
        elif mh < 1.0:
            bh, bw = cell, (max_tok - 3) * cell
        else:
            beta = min(math.floor(mw) * cell / width, math.floor(mh) * cell / height)
            bh, bw = math.floor(height * beta / p) * p, math.floor(width * beta / p) * p
        lh, lw = grid(bh, bw)
    return lh, lw, bh, bw


def main():
    d, path = sys.argv[1], sys.argv[2]
    vc = json.load(open(os.path.join(d, "config.json")))["vision_config"]
    p, r = vc["patch_size"], vc["downsample_ratio"]
    img = Image.open(path).convert("RGB")
    lh, lw, bh, bw = plan(img.width, img.height, p, r, vc["max_image_tokens"], vc["min_pixels"])
    nh, nw = bh // p, bw // p
    img = ImageOps.pad(img, (bw, bh), color=(127, 127, 127))
    x = np.asarray(img, np.float32).transpose(2, 0, 1) / 255
    x = bf16((x - 0.5) / 0.5)
    patches = x.reshape(3, nh, p, nw, p).transpose(1, 3, 0, 2, 4).reshape(nh * nw, -1)
    print(f"grid {nh}x{nw} patches, llm {lh}x{lw}, run {lh * (lw + 1) + 2} tokens")
    ck = Ckpt(d)
    g = lambda n: ck.get(n).astype(np.float32)
    # BF16_ACT=1: matmul inputs rounded to bf16 (the reference's dtype), to size the precision noise
    q16 = bf16 if os.environ.get("BF16_ACT") else (lambda t: t)
    VD, H = vc["hidden_size"], vc["num_attention_heads"]
    hd = VD // H
    x = q16(patches) @ g("vision.patch_embed.proj.weight").T + g("vision.patch_embed.proj.bias")
    rd = hd // 2
    inv = 1.0 / (vc.get("rope_theta", 10000) ** (np.arange(0, rd, 2, dtype=np.float32) / rd))
    hpos = np.repeat(np.arange(nh), nw).astype(np.float32)
    wpos = np.tile(np.arange(nw), nh).astype(np.float32)
    fr = np.concatenate([hpos[:, None] * inv, wpos[:, None] * inv], 1)
    cos, sin = np.cos(fr)[:, None, :], np.sin(fr)[:, None, :]

    def rms(v, w):
        return v / np.sqrt((v * v).mean(-1, keepdims=True) + 1e-6) * w

    def rot(t):
        a, b = t[..., : hd // 2], t[..., hd // 2:]
        return np.concatenate([a * cos - b * sin, b * cos + a * sin], -1)

    N = x.shape[0]
    for b in range(vc["num_hidden_layers"]):
        q = f"vision.blocks.{b}."
        h = rms(x, g(q + "norm1.weight"))
        qkv = q16(h) @ g(q + "attn.wqkv.weight").T + g(q + "attn.wqkv.bias")
        Q, K, V = (t.reshape(N, H, hd) for t in np.split(qkv, 3, -1))
        Q, K = rot(Q), rot(K)
        o = np.zeros((N, H, hd), np.float32)
        for hh in range(H):
            s = (Q[:, hh] @ K[:, hh].T) / math.sqrt(hd)
            s = np.exp(s - s.max(-1, keepdims=True))
            o[:, hh] = (s / s.sum(-1, keepdims=True)) @ V[:, hh]
        x = x + q16(o.reshape(N, -1)) @ g(q + "attn.wo.weight").T + g(q + "attn.wo.bias")
        h = rms(x, g(q + "norm2.weight"))
        gu = q16(h) @ g(q + "mlp.w1.weight").T
        gt, up = np.split(gu, 2, -1)
        x = x + q16(gt / (1 + np.exp(-gt)) * up) @ g(q + "mlp.w2.weight").T
        if b % 8 == 7:
            print(f"block {b}", flush=True)
    x = rms(x, g("vision.norm.weight"))
    grid = x.reshape(nh, nw, -1)
    ph, pw = -nh % r, -nw % r
    grid = np.pad(grid, ((0, ph), (0, pw), (0, 0)))
    cells = grid.reshape(lh, r, lw, r, VD).transpose(0, 2, 4, 1, 3).reshape(lh * lw, VD * r * r)
    a = cells @ g("aligner.w1.weight").T + g("aligner.w1.bias")
    from math import erf
    a = 0.5 * a * (1 + np.vectorize(erf)(a / math.sqrt(2)).astype(np.float32))
    a = a @ g("aligner.w2.weight").T + g("aligner.w2.bias")
    rows = [g("image_start")]
    for i in range(lh):
        rows += list(a[i * lw:(i + 1) * lw]) + [g("image_newline")]
    rows.append(g("image_end"))
    emb = np.stack(rows)
    if os.environ.get("SAVE_EMB"):
        emb.astype(np.float32).tofile(os.environ["SAVE_EMB"])
    if len(sys.argv) > 4:
        px = np.fromfile(sys.argv[3], np.float32).reshape(patches.shape)
        print("pixels: max |diff|", float(np.abs(px - patches).max()), "mismatches", int((px != patches).sum()))
        e = np.fromfile(sys.argv[4], np.float32).reshape(emb.shape)
        rel = np.linalg.norm(e - emb, axis=1) / np.linalg.norm(emb, axis=1)
        print("embeddings: rows", len(emb), "rel median %.2e max %.2e" % (np.median(rel), rel.max()))
        print("  percentiles 90/99: %.2e %.2e; cosine min %.5f" % (np.percentile(rel, 90), np.percentile(rel, 99),
              min(float(e[i] @ emb[i] / np.linalg.norm(e[i]) / np.linalg.norm(emb[i])) for i in range(len(emb)))))



if __name__ == "__main__":
    main()
