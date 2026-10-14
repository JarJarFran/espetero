#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Clef-Omni parity of the native engine: a reference Qwen3-Omni thinker in MLX (fp32
activations, the same 8-bit weights as the converted directory) and the joint schema head in
NumPy (fp32), both written from transformers' models/qwen3_omni_moe and the release's
joint_schema_model.py (read, never imported).

  # 1. synthetic records (drawn PNG images whose sizes need no resize, SystemOne bodies)
  python tests/ref/clef_omni_ref.py cases --out build/omni
  # 2. reference (needs mlx; the model's 32 GB must fit beside nothing else); --bf16: rounded
  #    like transformers' bf16 thinker, into ref_bf16/ (the release's own precision)
  python tests/ref/clef_omni_ref.py ref --model CONVERTED_DIR --cases build/omni [--bf16]
  # 3. the engine: vision features, then the records through /v1/systemone
  build/vision_dump CONVERTED_DIR build/omni/vision 1048576 build/omni/img*.png   (ESPETERO_NO_EXIF=1)
  ESPETERO_CLEF_DUMP=build/omni/dump espetero serve --model CONVERTED_DIR --max-ctx 65536 ...
  python tests/ref/clef_omni_ref.py post --cases build/omni --url http://127.0.0.1:8080
  # 4. comparison (no GPU)
  python tests/ref/clef_omni_ref.py compare --model CONVERTED_DIR --cases build/omni

`compare` reports: the encoded token ids against the reference encoder (identical or not);
the processor's patch rows (max |diff|); the merged vision features (per-token cosine); the
final hidden states of every record (per-row cosine, mean and min; after `ref --bf16`, also the
bf16 reference's own against the fp32 one: the precision the release runs at); the option
probabilities from the engine's head against the NumPy head on the reference states (max |dp|), and the NumPy
head on the engine's own states against the engine's logits (the C++ head alone).
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import math
import os
import sys
import urllib.request

import numpy as np

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)


# ---------------------------------------------------------------- cases
def draw(kind: str, w: int, h: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w, 3), 245, np.uint8)
    if kind == "circle":
        m = (xx - w * 0.5) ** 2 + (yy - h * 0.5) ** 2 < (min(w, h) * 0.3) ** 2
        img[m] = (210, 30, 30)
    elif kind == "shapes":
        img[:] = (30, 30, 60)
        img[h // 5 : h // 2, w // 8 : w // 3] = (40, 90, 220)
        tri = (yy > h * 0.55) & (yy < h * 0.9) & (np.abs(xx - w * 0.7) < (yy - h * 0.55) * 0.8)
        img[tri] = (240, 220, 40)
    elif kind == "stripes":
        img[(xx // 16) % 2 == 0] = (20, 140, 60)
    elif kind == "checker":
        img[((xx // 32) + (yy // 32)) % 2 == 0] = (10, 10, 10)
    elif kind == "gradient":
        img[..., 0] = (xx * 255 // max(w - 1, 1)).astype(np.uint8)
        img[..., 1] = (yy * 255 // max(h - 1, 1)).astype(np.uint8)
        img[..., 2] = 128
    elif kind == "bars":
        for i in range(5):
            x0 = w * (i + 1) // 7
            img[h - h * (i + 2) // 8 : h - h // 10, x0 : x0 + w // 12] = (50 + 40 * i, 80, 200 - 30 * i)
    elif kind == "noise":
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return img


IMAGES = [  # name, kind, width, height: multiples of 32 within 65,536 .. 1,048,576 pixels (no resize)
    ("img0", "circle", 512, 384),
    ("img1", "shapes", 640, 480),
    ("img2", "stripes", 320, 320),
    ("img3", "checker", 384, 256),
    ("img4", "gradient", 256, 256),
    ("img5", "bars", 1024, 1024),
    ("img6", "noise", 448, 320),
]


def records() -> list[dict]:
    yes_no = {"type": "noul"}
    return [
        {"state": {"note": "A photo of one drawn shape."}, "images": ["img0"], "questions": {
            "has_circle": {**yes_no, "instructions": "Is there a circle in the image?"},
            "color": {"type": "choice", "instructions": "Main color of the shape",
                      "criteria": {"red": "Mostly red", "green": "Mostly green", "blue": "Mostly blue"}},
            "count": {"type": "score", "instructions": "How many shapes are drawn?",
                      "criteria": ["none", "one", "two or more"]}}},
        {"state": "Describe the picture.", "images": ["img1"], "questions": {
            "dark_background": {**yes_no, "instructions": "Is the background dark?"},
            "shape": {"type": "choice", "instructions": "Which yellow shape is drawn?",
                      "criteria": {"triangle": None, "circle": None, "square": None}}}},
        {"state": {"task": "compare two images"}, "images": ["img2", "img3"], "questions": {
            "stripes": {"type": "choice", "instructions": "Which image shows stripes?",
                        "criteria": {"first": "The first image", "second": "The second image"}},
            "same": {**yes_no, "instructions": "Are both images identical?"}}},
        {"state": {"order": {"id": 1842, "items": [{"sku": "A-17", "qty": 2}, {"sku": "B-03", "qty": 1}],
                             "paid": True, "shipping": "express", "notes": "Leave at the front desk."}},
         "images": [], "questions": {
            "paid": {**yes_no, "instructions": "Has the order been paid?"},
            "shipping": {"type": "choice", "instructions": "Shipping method",
                         "criteria": {"standard": "Ground shipping", "express": "Next-day delivery", "pickup": "Collected at the shop"}},
            "size": {"type": "score", "instructions": "How large is the order?",
                     "criteria": ["one item", "a few items", "many items"]}}},
        {"state": "Heavy rain is expected tonight with strong wind gusts near the coast; the morning will be calm.",
         "images": [], "questions": {
            "rain": {**yes_no, "instructions": "Will it rain tonight?"},
            "severity": {"type": "score", "instructions": "How severe is the weather tonight?",
                         "criteria": ["calm", "mild", "rough", "dangerous"]}}},
        {"state": {"caption": "A color gradient."}, "images": ["img4"], "questions": {
            "bright_corner": {"type": "choice", "instructions": "Which corner is the brightest?",
                              "criteria": {"top_left": None, "bottom_right": None, "top_right": None, "bottom_left": None}},
            "uniform": {**yes_no, "instructions": "Is the image a single flat color?"}}},
        {"state": {"chart": "bar chart of five values"}, "images": ["img5"], "questions": {
            "rising": {**yes_no, "instructions": "Do the bars get taller from left to right?"},
            "bars": {"type": "score", "instructions": "How many bars are there?",
                     "criteria": ["fewer than three", "three to five", "more than five"]}}},
        {"state": {"source": "camera test"}, "images": ["img6"], "questions": {
            "structured": {**yes_no, "instructions": "Does the image show a recognizable object?"},
            "kind": {"type": "choice", "instructions": "What is it?",
                     "criteria": {"noise": "Random noise", "text": "A page of text", "landscape": "A landscape"}}}},
        {"state": {"recipe": "pancakes", "steps": ["mix flour and milk", "add eggs", "fry in a pan"], "servings": 4},
         "images": [], "questions": {
            "vegan": {**yes_no, "instructions": "Is the recipe vegan?"},
            "difficulty": {"type": "choice", "instructions": "",
                           "criteria": {"easy": "A beginner can do it", "hard": "Needs experience"}}}},
        {"state": {"pair": "shape and chart"}, "images": ["img0", "img5"], "questions": {
            "first_red": {**yes_no, "instructions": "Is the first image mostly red?"},
            "second": {"type": "choice", "instructions": "What does the second image show?",
                       "criteria": {"chart": "A bar chart", "face": "A face", "map": "A map"}}}},
    ]


def cmd_cases(args) -> None:
    from PIL import Image

    os.makedirs(args.out, exist_ok=True)
    for i, (name, kind, w, h) in enumerate(IMAGES):
        Image.fromarray(draw(kind, w, h, i)).save(os.path.join(args.out, name + ".png"))
    with open(os.path.join(args.out, "records.json"), "w") as f:
        json.dump(records(), f, indent=1)
    print(f"{len(IMAGES)} images, {len(records())} records -> {args.out}")


def load_png(cases: str, name: str) -> np.ndarray:
    from PIL import Image

    with Image.open(os.path.join(cases, name + ".png")) as im:
        return np.asarray(im.convert("RGB"))


# ---------------------------------------------------------------- the release's encoding
def render(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(q: dict) -> list:
    t = str(q["type"])
    if t == "noul":
        crit = {"true": "The proposition is true or the answer is yes.",
                "false": "The proposition is false or the answer is no."}
        crit.update(q.get("criteria") or {})
        return [(k, crit[k]) for k in ("true", "false")]
    if t == "choice":
        return sorted((str(k), v) for k, v in q["criteria"].items())
    return [(str(i), v) for i, v in enumerate(q["criteria"])]


def encode_record(tok, rec: dict, image_tokens: list[int], max_length: int = 64000) -> dict:
    """The release's encode_record for Clef-Omni (images only): ids, question and option spans."""
    T = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
    schema = T("\n\nSCHEMA FIELDS:\n")
    qs = []
    for qi, (qid, q) in enumerate(rec["questions"].items()):
        schema += T(f"\nFIELD {qi + 1}\nID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: ")
        q0 = len(schema)
        ins = q.get("instructions")
        if ins is None or ins == "":
            ins = str(qid)
        schema += T(render(ins))
        q1 = len(schema)
        schema += T("\nALLOWED OPTIONS:\n")
        spans, oids = [], []
        for oi, (oid, desc) in enumerate(question_options(q)):
            schema += T(f"OPTION {oi + 1}: ")
            s0 = len(schema)
            sem = {"option_id": oid}
            if desc is not None:
                sem["description"] = desc
            schema += T(render(sem))
            spans.append((s0, len(schema)))
            oids.append(oid)
            schema += T("\n")
        schema += T("END FIELD\n")
        qs.append({"id": qid, "type": {"noul": 0, "choice": 1, "score": 2}[q["type"]], "span": (q0, q1),
                   "spans": spans, "options": oids})
    prefix = T(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
    suffix = T("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    if image_tokens:  # the processor: IMAGE_PLACEHOLDER per image, <|image_pad|> expanded, one tokenizer call
        prefix = prefix + T("".join("<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>\n" for n in image_tokens))
    state = T(render(rec["state"]))
    fixed = len(prefix) + len(schema) + len(suffix)
    state = state[: max_length - fixed]
    off = len(prefix) + len(state)
    for q in qs:
        q["span"] = (q["span"][0] + off, q["span"][1] + off)
        q["spans"] = [(a + off, b + off) for a, b in q["spans"]]
    return {"ids": prefix + state + schema + suffix, "questions": qs}


def rope_positions(ids: list[int], grids: list[tuple[int, int, int]], cfg: dict) -> np.ndarray:
    """get_rope_index of Qwen3-Omni for text and images: (3, L)."""
    vs, img = cfg["vision_start_token_id"], cfg["image_token_id"]
    per_s = cfg.get("position_id_per_seconds", 13)
    out, st, g = [], 0, 0
    n_img = sum(1 for i, t in enumerate(ids[:-1]) if t == vs and ids[i + 1] == img)
    nxt = lambda: (max(int(p.max()) for p in out) + 1) if out else 0  # noqa: E731
    for _ in range(n_img):
        ed = ids.index(vs, st)
        st_idx = nxt()
        text_len = ed - st
        if text_len:
            out.append(np.arange(text_len)[None].repeat(3, 0) + st_idx)
            st_idx += text_len
        out.append(np.zeros((3, 1), np.int64) + st_idx)  # vision_start
        st_idx += 1
        t, h, w = grids[g]
        gh, gw = h // 2, w // 2
        ti = (np.arange(t) * per_s)[:, None].repeat(gh * gw, 1).reshape(-1)
        hi = np.arange(gh)[None, :, None].repeat(t, 0).repeat(gw, 2).reshape(-1)
        wi = np.arange(gw)[None, None, :].repeat(t, 0).repeat(gh, 1).reshape(-1)
        out.append(np.stack([ti, hi, wi]) + st_idx)
        st += text_len + 1 + t * gh * gw + 1
        out.append(np.zeros((3, 1), np.int64) + nxt())  # vision_end
        g += 1
    if st < len(ids):
        out.append(np.arange(len(ids) - st)[None].repeat(3, 0) + nxt())
    return np.concatenate(out, 1)


def patch_rows(img: np.ndarray) -> np.ndarray:
    """Qwen2-VL image processor rows for an image already at its target size (merge-window order)."""
    h, w, _ = img.shape
    x = (img.astype(np.float32) * np.float32(1 / 255) - np.float32(0.5)) / np.float32(0.5)
    x = x.transpose(2, 0, 1)  # C, H, W
    gh, gw = h // 16, w // 16
    p = x.reshape(3, gh // 2, 2, 16, gw // 2, 2, 16).transpose(1, 4, 2, 5, 0, 3, 6)  # bh, bw, mh, mw, C, py, px
    p = np.repeat(p[:, :, :, :, :, None], 2, axis=5)  # temporal pair
    return p.reshape(gh * gw, 3 * 2 * 16 * 16), (1, gh, gw)


# ---------------------------------------------------------------- MLX reference
class Ref:
    def __init__(self, model: str):
        import mlx.core as mx

        self.mx = mx
        self.cfg = json.load(open(os.path.join(model, "config.json")))
        idx = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
        self.W = {}
        for f in sorted(set(idx.values())):
            self.W.update(mx.load(os.path.join(model, f)))
        tc, vc = self.cfg["text_config"], self.cfg["vision_config"]
        self.tc, self.vc = tc, vc

    def f32(self, name):
        return self.W[name].astype(self.mx.float32)

    def deq(self, base, index=None):
        mx = self.mx
        w, s, b = self.W[base + ".weight"], self.W[base + ".scales"], self.W[base + ".biases"]
        if index is not None:
            w, s, b = w[index], s[index], b[index]
        return mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=8)

    # ---- vision tower (fp32; interpolation weights rounded to bf16 as mlx-vlm and Espetero do)
    def vision(self, pixels: list[np.ndarray], grids: list[tuple]):
        mx = self.mx
        vc = self.vc
        D, heads = vc["hidden_size"], vc["num_heads"]
        hd = D // heads
        side = int(round(math.sqrt(vc["num_position_embeddings"])))
        v = "vision_tower."
        x = mx.array(np.concatenate(pixels, 0))
        wp = self.f32(v + "patch_embed.proj.weight").reshape(D, -1)
        x = x @ wp.T + self.f32(v + "patch_embed.proj.bias")
        pos_tab = self.f32(v + "pos_embed.weight")
        pe, cosl, sinl, segs = [], [], [], []
        inv = 1.0 / (10000 ** (np.arange(0, hd // 2, 2, dtype=np.float32) / (hd // 2)))
        r0 = 0
        for (t, gh, gw) in grids:
            hs = np.linspace(0, side - 1, gh, dtype=np.float32)
            ws = np.linspace(0, side - 1, gw, dtype=np.float32)
            rows, cols = [], []
            for bh in range(gh // 2):
                for bw in range(gw // 2):
                    for mh in range(2):
                        for mw in range(2):
                            rows.append(bh * 2 + mh)
                            cols.append(bw * 2 + mw)
            rows, cols = np.array(rows), np.array(cols)
            fh, fw = hs[rows].astype(np.int64), ws[cols].astype(np.int64)
            ch, cw = np.minimum(fh + 1, side - 1), np.minimum(fw + 1, side - 1)
            dh, dw = hs[rows] - fh, ws[cols] - fw
            idx = [fh * side + fw, fh * side + cw, ch * side + fw, ch * side + cw]
            wts = [(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw]
            acc = 0
            for i_, w_ in zip(idx, wts):
                w_bf = mx.array(w_.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
                acc = acc + pos_tab[mx.array(i_)] * w_bf[:, None]
            pe.append(acc)
            emb = np.concatenate([rows[:, None] * inv[None], cols[:, None] * inv[None]], 1)
            emb = np.concatenate([emb, emb], 1)
            cosl.append(np.cos(emb))
            sinl.append(np.sin(emb))
            segs.append((r0, r0 + len(rows)))
            r0 += len(rows)
        x = x + mx.concatenate(pe, 0)
        cos, sin = mx.array(np.concatenate(cosl, 0))[:, None], mx.array(np.concatenate(sinl, 0))[:, None]

        def ln(x, p, eps=1e-6):
            m = mx.mean(x, -1, keepdims=True)
            var = mx.mean((x - m) ** 2, -1, keepdims=True)
            return (x - m) * mx.rsqrt(var + eps) * self.f32(p + ".weight") + self.f32(p + ".bias")

        def lin(x, p):
            return x @ self.f32(p + ".weight").T + self.f32(p + ".bias")

        def rot(x):
            h = x.shape[-1] // 2
            return mx.concatenate([-x[..., h:], x[..., :h]], -1)

        gelu_tanh = lambda z: 0.5 * z * (1 + mx.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))  # noqa: E731
        gelu_erf = lambda z: 0.5 * z * (1 + mx.erf(z / math.sqrt(2)))  # noqa: E731

        def merger(x, p, post):
            if post:
                x = ln(x.reshape(-1, 4 * D), p + ".norm")
            else:
                x = ln(x, p + ".norm").reshape(-1, 4 * D)
            return lin(gelu_erf(lin(x, p + ".linear_fc1")), p + ".linear_fc2")

        deep = []
        P = x.shape[0]
        for bi in range(vc["depth"]):
            b = f"{v}blocks.{bi}."
            qkv = lin(ln(x, b + "norm1"), b + "attn.qkv").reshape(P, 3, heads, hd)
            q, k, vv = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            q = q * cos + rot(q) * sin
            k = k * cos + rot(k) * sin
            outs = []
            for a, e in segs:
                qa, ka, va = (t[a:e].transpose(1, 0, 2) for t in (q, k, vv))
                s = (qa @ ka.transpose(0, 2, 1)) * (1.0 / math.sqrt(hd))
                outs.append((mx.softmax(s, axis=-1) @ va).transpose(1, 0, 2).reshape(e - a, D))
            x = x + lin(mx.concatenate(outs, 0), b + "attn.proj")
            x = x + lin(gelu_tanh(lin(ln(x, b + "norm2"), b + "mlp.linear_fc1")), b + "mlp.linear_fc2")
            if bi in vc["deepstack_visual_indexes"]:
                j = vc["deepstack_visual_indexes"].index(bi)
                deep.append(merger(x, f"{v}deepstack_merger_list.{j}", True))
            mx.eval(x, *deep)
        main = merger(x, v + "merger", False)
        mx.eval(main)
        return main, deep

    # ---- decoder (fp32)
    def decoder(self, ids: list[int], pos: np.ndarray, vis=None, deep=None, n_layers=None, bf16=False):
        """fp32 activations; bf16=True rounds to bf16 where transformers' bf16 thinker does (every
        linear, norm and residual output, cos / sin, router logits and weights), a measure of the
        precision of the release itself."""
        mx = self.mx
        Rb = (lambda x: x.astype(mx.bfloat16).astype(mx.float32)) if bf16 else (lambda x: x)  # noqa: E731
        tc = self.tc
        D, H, Hkv, HD = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
        E, k, eps = tc["num_experts"], tc["num_experts_per_tok"], tc["rms_norm_eps"]
        L = len(ids)
        pre = "language_model.model."
        h = self.f32(pre + "embed_tokens.weight")[mx.array(ids)]
        img_rows = np.nonzero(np.array(ids) == self.cfg["image_token_id"])[0]
        if vis is not None and len(img_rows):
            h[mx.array(img_rows)] = Rb(vis)
        rp = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
        sec = rp.get("mrope_section", [24, 20, 20])
        theta = rp.get("rope_theta", tc.get("rope_theta", 1e6))
        inv = 1.0 / (theta ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
        sel = np.array([1 if (d % 3 == 1 and d < 3 * sec[1]) else (2 if (d % 3 == 2 and d < 3 * sec[2]) else 0)
                        for d in range(HD // 2)])
        ang = pos[sel, :].T.astype(np.float64) * inv[None]  # [L, HD/2]
        ang = np.concatenate([ang, ang], 1)
        cos = Rb(mx.array(np.cos(ang).astype(np.float32)))[:, None]
        sin = Rb(mx.array(np.sin(ang).astype(np.float32)))[:, None]
        mask = mx.array(np.triu(np.full((L, L), -np.inf, np.float32), 1))

        def rms(x, w):
            return Rb(Rb(x * mx.rsqrt(mx.mean(x * x, -1, keepdims=True) + eps)) * w)

        def deq(*a):
            return Rb(self.deq(*a))

        def rot(x):
            hh = x.shape[-1] // 2
            return mx.concatenate([-x[..., hh:], x[..., :hh]], -1)

        for li in range(n_layers or tc["num_hidden_layers"]):
            p = f"{pre}layers.{li}."
            a = p + "self_attn."
            xn = rms(h, self.f32(p + "input_layernorm.weight"))
            q = Rb(xn @ deq(a + "q_proj").T).reshape(L, H, HD)
            kk = Rb(xn @ deq(a + "k_proj").T).reshape(L, Hkv, HD)
            vv = Rb(xn @ deq(a + "v_proj").T).reshape(L, Hkv, HD)
            q = rms(q, self.f32(a + "q_norm.weight"))
            kk = rms(kk, self.f32(a + "k_norm.weight"))
            q = Rb(Rb(q * cos) + Rb(rot(q) * sin))
            kk = Rb(Rb(kk * cos) + Rb(rot(kk) * sin))
            G = H // Hkv
            qh = q.transpose(1, 0, 2)
            kh = mx.repeat(kk.transpose(1, 0, 2), G, axis=0)
            vh = mx.repeat(vv.transpose(1, 0, 2), G, axis=0)
            s = (qh @ kh.transpose(0, 2, 1)) * (1.0 / math.sqrt(HD)) + mask
            o = Rb(Rb(mx.softmax(s, axis=-1)) @ vh).transpose(1, 0, 2).reshape(L, H * HD)
            h = Rb(h + Rb(o @ deq(a + "o_proj").T))
            x2 = rms(h, self.f32(p + "post_attention_layernorm.weight"))
            probs = mx.softmax(Rb(x2 @ self.f32(p + "mlp.gate.weight").T), axis=-1)
            mx.eval(h, x2, probs)
            pr = np.array(probs)
            top = np.argsort(-pr, axis=-1, kind="stable")[:, :k]
            tw = np.take_along_axis(pr, top, -1)
            tw = tw / tw.sum(-1, keepdims=True)
            if bf16:
                tw = np.array(Rb(mx.array(tw.astype(np.float32))))
            moe = mx.zeros((L, D), mx.float32)
            sm = p + "mlp.switch_mlp."
            for e in np.unique(top):
                rows, slot = np.nonzero(top == e)
                xe = x2[mx.array(rows)]
                g = Rb(xe @ deq(sm + "gate_proj", int(e)).T)
                u = Rb(xe @ deq(sm + "up_proj", int(e)).T)
                y = Rb(Rb(Rb(g * mx.sigmoid(g)) * u) @ deq(sm + "down_proj", int(e)).T)
                moe = Rb(moe.at[mx.array(rows)].add(Rb(y * mx.array(tw[rows, slot].astype(np.float32))[:, None])))
            h = Rb(h + moe)
            if deep is not None and li < len(deep) and len(img_rows):
                h = Rb(h.at[mx.array(img_rows)].add(Rb(deep[li])))
            mx.eval(h)
        h = rms(h, self.f32(pre + "norm.weight"))
        mx.eval(h)
        return np.array(h)


# ---------------------------------------------------------------- joint schema head (NumPy fp32)
class Head:
    def __init__(self, model: str):
        import mlx.core as mx

        t = mx.load(os.path.join(model, "joint_head.safetensors"))
        self.w = {k: np.array(v.astype(mx.float32)) for k, v in t.items()}
        self.cfg = json.load(open(os.path.join(model, "joint_head_config.json")))
        idx = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
        lm = mx.load(os.path.join(model, idx["language_model.lm_head.weight"]))["language_model.lm_head.weight"]
        self.lm = lm

    def lm_rows(self, ids):
        import mlx.core as mx

        return np.array(self.lm[mx.array(np.asarray(ids, np.int32))].astype(mx.float32))

    def ln(self, x, p, eps=1e-5):
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / np.sqrt(v + eps) * self.w[p + ".weight"] + self.w[p + ".bias"]

    def lin(self, x, p, bias=True):
        y = x @ self.w[p + ".weight"].T
        return y + self.w[p + ".bias"] if bias else y

    @staticmethod
    def gelu(x):
        from math import erf

        return 0.5 * x * (1 + np.vectorize(erf)(x / math.sqrt(2)))

    def mha(self, p, q, kv):
        W, b = self.w[p + ".in_proj_weight"], self.w[p + ".in_proj_bias"]
        E = q.shape[-1]
        nh = self.cfg["heads"]
        dh = E // nh
        Q, K, V = q @ W[:E].T + b[:E], kv @ W[E:2 * E].T + b[E:2 * E], kv @ W[2 * E:].T + b[2 * E:]
        out = np.zeros_like(Q)
        for i in range(nh):
            sl = slice(i * dh, (i + 1) * dh)
            s = Q[:, sl] @ K[:, sl].T / math.sqrt(dh)
            s = np.exp(s - s.max(-1, keepdims=True))
            out[:, sl] = (s / s.sum(-1, keepdims=True)) @ V[:, sl]
        return self.lin(out, p + ".out_proj")

    def forward(self, hidden, ids, questions):
        hn = self.ln(hidden, "hidden_norm")
        mem = self.lin(hn, "memory_projection", False)
        glob_ = hn[-1]
        qv = np.stack([hn[a:b].mean(0) for a, b in (q["span"] for q in questions)])
        types = np.array([q["type"] for q in questions])
        ctx, lex, counts = [], [], []
        for q in questions:
            ctx.append(np.stack([hn[a:b].mean(0) for a, b in q["spans"]]))
            lex.append(np.stack([self.lm_rows(ids[a:b]).mean(0) for a, b in q["spans"]]))
            counts.append(len(q["spans"]))
        opts = np.concatenate([self.lin(c, "option_context_projection", False) + self.lin(l, "option_lexical_projection", False)
                               + self.lin(qv[i], "option_question_projection", False)[None]
                               for i, (c, l) in enumerate(zip(ctx, lex))])
        for li in range(self.cfg["routing_layers"]):
            p = f"evidence_layers.{li}"
            opts = opts + self.mha(p + ".attention", self.ln(opts, p + ".query_norm"), self.ln(mem, p + ".memory_norm"))
            f = self.ln(opts, p + ".feedforward_norm")
            opts = opts + self.lin(self.gelu(self.lin(f, p + ".feedforward.0")), p + ".feedforward.3")
        split = np.split(opts, np.cumsum(counts)[:-1])
        base = self.lin(qv, "question_projection", False)
        summ = []
        for f, o in zip(base, split):
            s = o @ f / math.sqrt(o.shape[-1])
            s = np.exp(s - s.max())
            s /= s.sum()
            summ.append((s[:, None] * o).sum(0))
        fields = base + self.ln(np.stack(summ), "option_summary_norm") + self.lin(glob_, "global_projection", False)[None] \
            + self.w["type_embedding.weight"][types]
        for li in range(self.cfg["layers"]):
            p = f"layers.{li}"
            fields = fields + self.mha(p + ".self_attn", self.ln(fields, p + ".norm1"), self.ln(fields, p + ".norm1"))
            fields = fields + self.mha(p + ".multihead_attn", self.ln(fields, p + ".norm2"), mem)
            fields = fields + self.lin(self.gelu(self.lin(self.ln(fields, p + ".norm3"), p + ".linear1")), p + ".linear2")
        fields = self.ln(fields, "field_norm")
        cap = math.log(100.0)
        prior_s = math.exp(min(float(self.w["prior_logit_scale"].reshape(-1)[0]), cap))
        joint_s = math.exp(min(float(self.w["joint_logit_scale"].reshape(-1)[0]), cap))
        gate = 1 / (1 + math.exp(-float(self.w["residual_gate"].reshape(-1)[0])))
        out = []
        for i, (f, l, o) in enumerate(zip(fields, lex, split)):
            anchor = qv[i] + glob_
            anchor = anchor / max(np.linalg.norm(anchor), 1e-12)
            la = l / np.maximum(np.linalg.norm(l, axis=-1, keepdims=True), 1e-12)
            prior = prior_s * (la @ anchor)
            on = self.ln(o, "option_norm")
            rf = np.broadcast_to(f, on.shape)
            cosv = (rf * on).sum(-1) / np.maximum(np.linalg.norm(rf, axis=-1) * np.linalg.norm(on, axis=-1), 1e-8)
            feat = np.concatenate([rf, on, rf * on, np.abs(rf - on)], -1)
            res = self.lin(self.gelu(self.lin(feat, "residual_scorer.0")), "residual_scorer.3")[:, 0]
            out.append(prior + gate * (joint_s * cosv + res))
        return out


def softmax(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def image_tokens_of(img: np.ndarray) -> int:
    return (img.shape[0] // 32) * (img.shape[1] // 32)


def cmd_ref(args) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    recs = json.load(open(os.path.join(args.cases, "records.json")))
    ref = Ref(args.model)
    names = sorted({n for r in recs for n in r["images"]})
    imgs = {n: load_png(args.cases, n) for n in names}
    out_dir = os.path.join(args.cases, "ref_bf16" if args.bf16 else "ref")
    os.makedirs(out_dir, exist_ok=True)
    feats = {}
    for n in names:  # one image per tower call (as the engine's dump does)
        rows, grid = patch_rows(imgs[n])
        main, deep = ref.vision([rows], [grid])
        feats[n] = (np.array(main), [np.array(d) for d in deep])
        np.save(os.path.join(out_dir, f"{n}.pixels.npy"), rows)
        np.save(os.path.join(out_dir, f"{n}.emb.npy"), feats[n][0])
        np.save(os.path.join(out_dir, f"{n}.deep.npy"), np.stack(feats[n][1]))
        print(f"vision {n}: grid {grid}, {feats[n][0].shape[0]} tokens", flush=True)
    head = Head(args.model)
    for i, r in enumerate(recs):
        enc = encode_record(tok, r, [image_tokens_of(imgs[n]) for n in r["images"]])
        grids = [(1, imgs[n].shape[0] // 16, imgs[n].shape[1] // 16) for n in r["images"]]
        pos = rope_positions(enc["ids"], grids, ref.cfg) if grids else np.arange(len(enc["ids"]))[None].repeat(3, 0)
        vis = deep = None
        if r["images"]:
            vis = ref.mx.array(np.concatenate([feats[n][0] for n in r["images"]]))
            deep = [ref.mx.array(np.concatenate([feats[n][1][j] for n in r["images"]])) for j in range(len(feats[r["images"][0]][1]))]
        hidden = ref.decoder(enc["ids"], pos, vis, deep, bf16=args.bf16)
        logits = head.forward(hidden, enc["ids"], enc["questions"])
        np.save(os.path.join(out_dir, f"rec{i}.hidden.npy"), hidden.astype(np.float32))
        json.dump({"ids": enc["ids"], "questions": enc["questions"], "positions": pos.tolist(),
                   "logits": {q["id"]: lg.tolist() for q, lg in zip(enc["questions"], logits)}},
                  open(os.path.join(out_dir, f"rec{i}.json"), "w"))
        print(f"record {i}: {len(enc['ids'])} tokens; " +
              ", ".join(f"{q['id']}: {np.round(softmax(lg), 3).tolist()}" for q, lg in zip(enc["questions"], logits)), flush=True)


def cmd_post(args) -> None:
    recs = json.load(open(os.path.join(args.cases, "records.json")))
    outs = []
    for i, r in enumerate(recs):
        body = {"model": "clef-omni", "state": r["state"], "questions": r["questions"], "return_logits": True}
        if r["images"]:
            body["images"] = ["data:image/png;base64," + base64.b64encode(open(os.path.join(args.cases, n + ".png"), "rb").read()).decode()
                              for n in r["images"]]
        req = urllib.request.Request(args.url + "/v1/systemone", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        res = json.load(urllib.request.urlopen(req, timeout=600))
        outs.append(res)
        print(f"record {i}: {res['usage']['input_tokens']} tokens, {res['metal'].get('backbone_ms', 0):.0f} ms", flush=True)
    json.dump(outs, open(os.path.join(args.cases, "engine.json"), "w"))


def cos_rows(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-30)


def cmd_compare(args) -> None:
    recs = json.load(open(os.path.join(args.cases, "records.json")))
    rdir = os.path.join(args.cases, "ref")
    report = {"vision": {}, "records": []}
    vdir = os.path.join(args.cases, "vision")
    if os.path.exists(os.path.join(vdir, "index.json")):
        vidx = json.load(open(os.path.join(vdir, "index.json")))
        for e in vidx["images"]:
            n = os.path.splitext(os.path.basename(e["file"]))[0]
            px = np.fromfile(os.path.join(vdir, e["stem"] + ".pixels.f32"), np.float32).reshape(e["rows"], -1)
            emb = np.fromfile(os.path.join(vdir, e["stem"] + ".emb.f32"), np.float32).reshape(e["tokens"], -1)
            rpx, remb = np.load(os.path.join(rdir, n + ".pixels.npy")), np.load(os.path.join(rdir, n + ".emb.npy"))
            c = cos_rows(emb, remb)
            report["vision"][n] = {"pixels_max_abs": float(np.abs(px - rpx).max()), "cos_mean": float(c.mean()),
                                   "cos_min": float(c.min()), "tokens": int(c.size)}
    dumps = sorted(glob.glob(os.path.join(args.cases, "dump", "clef*.ids.bin")))[-len(recs):]  # the last post
    head = Head(args.model) if dumps else None
    for i, r in enumerate(recs):
        ref = json.load(open(os.path.join(rdir, f"rec{i}.json")))
        rh = np.load(os.path.join(rdir, f"rec{i}.hidden.npy"))
        row = {"record": i, "tokens": len(ref["ids"]), "images": len(r["images"])}
        if i < len(dumps):
            stem = dumps[i][: -len(".ids.bin")]
            ids = np.fromfile(stem + ".ids.bin", np.int32).tolist()
            row["ids_identical"] = ids == ref["ids"]
            hid = np.fromfile(stem + ".hidden.bin", np.float32).reshape(len(ids), -1)
            if row["ids_identical"]:
                c = cos_rows(hid, rh)
                row["hidden_cos_mean"], row["hidden_cos_min"] = float(c.mean()), float(c.min())
                el = json.load(open(stem + ".logits.json"))
                dp = dp_head = 0.0
                own = head.forward(hid, ids, ref["questions"])
                for q, lg in zip(ref["questions"], own):
                    pe, pr, po = softmax(el[q["id"]]), softmax(ref["logits"][q["id"]]), softmax(lg)
                    dp = max(dp, float(np.abs(pe - pr).max()))
                    dp_head = max(dp_head, float(np.abs(pe - po).max()))
                row["prob_max_abs_diff"], row["head_only_prob_max_abs_diff"] = dp, dp_head
        # The precision of the release itself: the bf16 reference against the fp32 one.
        bdir = os.path.join(args.cases, "ref_bf16")
        if os.path.exists(os.path.join(bdir, f"rec{i}.json")):
            bref = json.load(open(os.path.join(bdir, f"rec{i}.json")))
            c = cos_rows(np.load(os.path.join(bdir, f"rec{i}.hidden.npy")), rh)
            row["bf16_ref_hidden_cos_mean"], row["bf16_ref_hidden_cos_min"] = float(c.mean()), float(c.min())
            row["bf16_ref_prob_max_abs_diff"] = max(float(np.abs(softmax(bref["logits"][q["id"]]) - softmax(ref["logits"][q["id"]])).max())
                                                    for q in ref["questions"])
        report["records"].append(row)
        print(json.dumps(row), flush=True)
    for n, v in report["vision"].items():
        print(n, json.dumps(v))
    json.dump(report, open(os.path.join(args.cases, "report.json"), "w"), indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("cases")
    a.add_argument("--out", required=True)
    a = sub.add_parser("ref")
    a.add_argument("--model", required=True)
    a.add_argument("--cases", required=True)
    a.add_argument("--bf16", action="store_true", help="the decoder rounded like transformers' bf16 thinker (into ref_bf16/)")
    a = sub.add_parser("post")
    a.add_argument("--cases", required=True)
    a.add_argument("--url", default="http://127.0.0.1:8080")
    a = sub.add_parser("compare")
    a.add_argument("--model", required=True)
    a.add_argument("--cases", required=True)
    args = ap.parse_args()
    {"cases": cmd_cases, "ref": cmd_ref, "post": cmd_post, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
