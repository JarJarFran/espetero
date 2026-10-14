#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Gemma 4 vision parity of the native engine against mlx-vlm.

  # 0. synthetic test images (PIL only)
  python3 tests/ref/gemma4_vision_ref.py images --out build/g4v_img
  # 1. reference (mlx-vlm 0.7.1): processor, vision tower, and prompts with images
  python -I tests/ref/gemma4_vision_ref.py ref --model DIR --images build/g4v_img/*.png \\
      --out build/g4v_ref [--gen 64] [--bidi sliding|all|none] [--fp32]
  # 2. the engine on the same images and ids
  build/vision_dump DIR build/g4v_ours 0 build/g4v_img/*.png
  build/logits_dump DIR build/g4v_ref/cases.json build/g4v_ours.json
  # 3. comparison (no GPU)
  python3 tests/ref/gemma4_vision_ref.py compare --ref build/g4v_ref --vision build/g4v_ours \\
      --logits build/g4v_ours.json

`ref` writes, per image, the processor's patch rows (float32 [patches, 768]:
rows in raster order, columns (y, x, channel)) and the soft tokens the
language model receives (vision tower + embed_vision, float32 [tokens,
hidden]), then renders chat prompts with images through the model's
template, expands each image to <|image> + soft tokens + <image|> as the
processor does, and records the teacher-forced logits from the prompt's
last position on (argmax, log-sum-exp, top 64) and `--gen` greedy tokens.
`--fp32` runs mlx-vlm with fp32 activations (the precision reference); the
tower's input stays rounded to bf16 as the checkpoint's dtype makes every
engine round it (`--exact-pixels` drops that rounding).

Image tokens attend to each other within their image (Gemma 4's
use_bidirectional_attention "vision"). `--bidi` picks the layers that do:
"sliding" (transformers 5.17's create_masks_for_vision_model: sliding-window
layers only, global layers causal; the default), "all" (mlx-vlm 0.7.1's
_make_masks, which overlays every layer kind) or "none" (causal).
"""

import argparse
import json
import math
import os
import sys


def cmd_images(a):
    """Synthetic images of several sizes and aspect ratios (shapes, colours, digits, a noisy gradient)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(a.out, exist_ok=True)

    def font(size):
        for f in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"):
            if os.path.exists(f):
                return ImageFont.truetype(f, size)
        return ImageFont.load_default()

    im = Image.new("RGB", (640, 480), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse((220, 140, 420, 340), fill=(220, 30, 30))
    im.save(f"{a.out}/0_red_circle.png")

    im = Image.new("RGB", (800, 600), (235, 235, 240))
    d = ImageDraw.Draw(im)
    d.rectangle((80, 120, 330, 370), fill=(30, 60, 200))
    d.polygon([(500, 420), (720, 420), (610, 180)], fill=(30, 170, 60))
    im.save(f"{a.out}/1_square_triangle.png")

    im = Image.new("RGB", (512, 512), (250, 220, 60))
    d = ImageDraw.Draw(im)
    d.text((110, 120), "42", fill=(0, 0, 0), font=font(260))
    im.save(f"{a.out}/2_digits.png")

    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:750, 0:1000]
    g = np.stack([x / 1000 * 255, y / 750 * 255, (x + y) / 1750 * 255], -1) + rng.normal(0, 12, (750, 1000, 3))
    im = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle((600, 300, 900, 600), outline=(255, 255, 255), width=12)
    im.save(f"{a.out}/3_gradient_1000x750.png")

    im = Image.new("RGB", (300, 900), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for i, c in enumerate([(200, 0, 0), (0, 150, 0), (0, 0, 200)]):
        d.rectangle((0, i * 300, 299, i * 300 + 299), fill=c)
    im.save(f"{a.out}/4_stripes_tall.png")

    im = Image.new("RGB", (96, 72), (40, 40, 40))
    d = ImageDraw.Draw(im)
    d.rectangle((30, 20, 66, 52), fill=(255, 200, 0))
    im.save(f"{a.out}/5_small.png")
    print("\n".join(sorted(os.listdir(a.out))))


PROMPTS = [
    # (image indices, text, thinking)
    ([0], "What shape and colour is in this image? Answer in one sentence.", False),
    ([1], "Describe the shapes in this picture and their colours.", False),
    ([2], "Which number is written in the image?", False),
    ([3], "Describe this image briefly.", False),
    ([0, 4], "Compare the two images.", False),
    ([5], "What colour is the rectangle?", False),
    ([], "Explain in two sentences why the sky is blue.", False),  # text only: the same passes without images
]


def patches_of(pv):
    """[3, H, W] float32 -> [H/16 * W/16, 16*16*3] (the processor's convert_image_to_patches order)."""
    import numpy as np
    c, h, w = pv.shape
    p = pv.reshape(c, h // 16, 16, w // 16, 16).transpose(1, 3, 2, 4, 0)
    return np.ascontiguousarray(p.reshape((h // 16) * (w // 16), 16 * 16 * c), dtype=np.float32)


def bidi_masks(lm, mode):
    """Replace the text model's mask builder for passes that see image tokens."""
    import mlx.core as mx
    from mlx_vlm.models.base import create_causal_mask
    tm = lm.model
    orig = tm._make_masks

    def make(h, cache, mm_token_type_ids=None):
        if mode == "all" or h.shape[1] == 1 or mm_token_type_ids is None:
            return orig(h, cache, mm_token_type_ids if mode != "none" else None)
        if any(c is not None and int(getattr(c, "offset", 0) or 0) > 0 for c in cache):
            raise SystemExit("bidi_masks: a prefill over a cached prefix is not supported here")
        N = h.shape[1]
        ids = tm._block_sequence_ids_for_mask(mm_token_type_ids)
        same = (ids[:, :, None] != -1) & (ids[:, :, None] == ids[:, None, :])
        q = mx.arange(N)[:, None]
        k = mx.arange(N)[None]
        out, kinds = [], {}
        for l in tm.layers:
            t = l.layer_type
            if t not in kinds:
                if t == "sliding_attention":
                    m = q < k + tm.window_size  # AND(window, OR(causal, block))
                    m = m & ((q >= k) | same) if mode == "sliding" else m & (q >= k)
                    m = m[:, None] if m.ndim == 3 else m
                else:
                    m = create_causal_mask(N)
                kinds[t] = m
            out.append(kinds[t])
        return out

    tm._make_masks = make


def rows_of(logits, top):
    import mlx.core as mx
    x = logits.astype(mx.float32)
    lse = mx.logsumexp(x, axis=-1)
    idx = mx.argsort(-x, axis=-1)[:, :top]
    vals = mx.take_along_axis(x, idx, axis=-1)
    mx.eval(lse, idx, vals)
    out = []
    for r in range(x.shape[0]):
        ids = idx[r].tolist()
        out.append({"argmax": ids[0], "lse": float(lse[r]), "top": [[i, float(v)] for i, v in zip(ids, vals[r].tolist())]})
    return out


def cmd_ref(a):
    import mlx.core as mx
    import numpy as np
    from mlx_vlm import load
    from PIL import Image

    model, proc = load(a.model)
    if a.fp32:
        model.set_dtype(mx.float32)

    def tower_input(pv):
        # The tower casts 2 * (pixels - 0.5) to its weights' dtype (bf16 for these checkpoints).
        # With --fp32 the weights are fp32: keep that bf16 rounding of the input (the checkpoint's
        # own input precision, as transformers, vLLM and mlx-vlm run it) unless --exact-pixels.
        x = mx.array(pv)
        if a.fp32 and not a.exact_pixels:
            x = (2 * (x - 0.5)).astype(mx.bfloat16).astype(mx.float32) / 2 + 0.5
        return x
    lm = model.language_model
    bidi_masks(lm, a.bidi)
    tok = proc.tokenizer
    ip = proc.image_processor
    cfg = model.config
    os.makedirs(a.out, exist_ok=True)
    images = [Image.open(p).convert("RGB") for p in a.images]

    vis = []
    for i, (path, img) in enumerate(zip(a.images, images)):
        data, n_soft = ip([img])
        pv = data["pixel_values"]
        pv = pv[0] if isinstance(pv, list) else pv[0]
        pv = np.asarray(pv, dtype=np.float32)
        x = tower_input(pv[None])
        tower = model.vision_tower(x)
        soft = model.embed_vision(tower)
        mx.eval(tower, soft)
        stem = f"{a.out}/{i}"
        patches_of(pv).tofile(stem + ".pixels.f32")
        np.asarray(tower[0].astype(mx.float32)).tofile(stem + ".tower.f32")
        np.asarray(soft[0].astype(mx.float32)).tofile(stem + ".emb.f32")
        vis.append({"file": path, "width": img.width, "height": img.height, "resized": [int(pv.shape[1]), int(pv.shape[2])],
                    "patches": int(pv.shape[1] // 16 * pv.shape[2] // 16), "tokens": int(n_soft[0]),
                    "hidden": int(soft.shape[-1]), "stem": str(i)})
        print(f"{path}: {img.width}x{img.height} -> {pv.shape[2]}x{pv.shape[1]}, {n_soft[0]} soft tokens", file=sys.stderr)
    json.dump({"model": a.model, "fp32": a.fp32, "images": vis}, open(f"{a.out}/vision.json", "w"), indent=1)
    if a.vision_only:
        return

    feats = None
    if a.features:  # vision_dump's index.json: the stems of the same images, in order
        idx_json = json.load(open(f"{a.features}/index.json"))["images"]
        feats = [mx.array(np.fromfile(f"{a.features}/{e['stem']}.emb.f32", dtype=np.float32).reshape(e["tokens"], -1))
                 for e in idx_json]
        if a.fp32:
            feats = [f.astype(mx.float32) for f in feats]
    boi, eoi, img_id = cfg.boi_token_id, cfg.eoi_token_id, cfg.image_token_id
    cases, refs = [], []
    for idx, text, think in PROMPTS:
        if idx and max(idx) >= len(images):
            continue
        content = [{"type": "image"} for _ in idx] + [{"type": "text", "text": text}]
        rendered = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                           tokenize=False, enable_thinking=think)
        raw = tok.encode(rendered, add_special_tokens=False)
        ids, rows, k = [], [], 0
        for t in raw:  # the processor's expansion: <|image> + soft tokens + <image|>
            if t == img_id:
                n = vis[idx[k]]["tokens"]
                ids.append(boi)
                rows.append(len(ids))
                ids += [img_id] * n
                ids.append(eoi)
                k += 1
            else:
                ids.append(t)
        assert k == len(idx), "image placeholders and images differ"
        pvs = []
        for j in idx:
            data, _ = ip([images[j]])
            pv = data["pixel_values"]
            pvs.append(tower_input(np.asarray(pv[0] if isinstance(pv, list) else pv[0], dtype=np.float32)))
        tt = mx.array([[1 if t == img_id else 0 for t in ids]])
        P = len(ids)
        extra = {}
        if feats is not None:  # another tower's soft tokens (vision_dump's) instead of mlx-vlm's
            extra["cached_image_features"] = mx.concatenate([feats[j] for j in idx], axis=0)[None] if idx else None
        # Greedy generation over a KV cache (one prefill pass with the images, then decode steps).
        cache = model.make_cache()
        out = model(mx.array([ids]), pixel_values=pvs or None, cache=cache, mm_token_type_ids=tt, **extra)
        logits = out.logits if hasattr(out, "logits") else out
        gen, margins = [], []
        last = logits[0, -1].astype(mx.float32)
        for _ in range(a.gen):
            lp = last - mx.logsumexp(last)
            two = mx.topk(lp, 2)
            nxt = int(mx.argmax(lp).item())
            mx.eval(two)
            t2 = sorted(two.tolist(), reverse=True)
            margins.append(t2[0] - t2[1])
            gen.append(nxt)
            if nxt in (1, 106, 50):
                break
            o = lm(mx.array([[nxt]]), cache=cache)
            last = o.logits[0, -1].astype(mx.float32)
        # Teacher-forced: prompt + the first `forced` generated tokens in one pass, rows from P - 1 on.
        full = ids + gen[: a.forced]
        tt2 = mx.array([[1 if t == img_id else 0 for t in ids] + [0] * (len(full) - P)])
        out = model(mx.array([full]), pixel_values=pvs or None, mm_token_type_ids=tt2, **extra)
        logits = out.logits if hasattr(out, "logits") else out
        rws = rows_of(logits[0, P - 1:], 64)
        cases.append({"ids": full, "prompt_len": P, "gen": a.gen,
                      "images": [{"file": a.images[j], "row": r} for j, r in zip(idx, rows)]})
        refs.append({"rows": rws, "gen": gen, "gen_margin": margins, "prompt_len": P})
        print(f"prompt {P} tokens ({len(idx)} images), generated {len(gen)}: {tok.decode(gen)[:100]!r}", file=sys.stderr)
    json.dump({"cases": cases}, open(f"{a.out}/cases.json", "w"))
    json.dump({"model": a.model, "fp32": a.fp32, "bidi": a.bidi, "cases": refs}, open(f"{a.out}/ref.json", "w"))


def cos_stats(a, r):
    import numpy as np
    a = a.astype(np.float64)
    r = r.astype(np.float64)
    if a.shape != r.shape:
        return {"shape": [list(a.shape), list(r.shape)]}
    cos = (a * r).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(r, axis=-1), 1e-30)
    return {"tokens": int(cos.size), "cos_mean": round(float(cos.mean()), 6), "cos_min": round(float(cos.min()), 6),
            "below_0999": int((cos < 0.999).sum()),
            "rel_err": round(float(np.linalg.norm(a - r) / max(np.linalg.norm(r), 1e-30)), 6)}


def cmd_compare(a):
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gemma4_ref import logprob_stats

    report = {}
    if a.vision:
        ref = json.load(open(f"{a.ref}/vision.json"))
        ours = json.load(open(f"{a.vision}/index.json"))
        out = []
        for rv, ov in zip(ref["images"], ours["images"]):
            rp = np.fromfile(f"{a.ref}/{rv['stem']}.pixels.f32", dtype=np.float32)
            op = np.fromfile(f"{a.vision}/{ov['stem']}.pixels.f32", dtype=np.float32)
            re = np.fromfile(f"{a.ref}/{rv['stem']}.emb.f32", dtype=np.float32).reshape(rv["tokens"], -1)
            oe = np.fromfile(f"{a.vision}/{ov['stem']}.emb.f32", dtype=np.float32).reshape(ov["tokens"], -1)
            px = {"equal": bool(rp.shape == op.shape and np.array_equal(rp, op))}
            if rp.shape == op.shape:
                px["max_abs"] = float(np.abs(rp - op).max())
            out.append({"file": os.path.basename(rv["file"]), "tokens": [rv["tokens"], ov["tokens"]], "pixels": px,
                        "soft_tokens": cos_stats(oe, re)})
        report["vision"] = out
    if a.logits:
        ref = json.load(open(f"{a.ref}/ref.json"))
        ours = json.load(open(a.logits))
        summary, gens = {}, []

        def acc(kind, r, o):
            eq, d, dl = logprob_stats(r, o)
            s = summary.setdefault(kind, {"rows": 0, "top1": 0, "max_dlogp": 0.0, "max_dlse": 0.0, "sum": 0.0})
            s["rows"] += 1
            s["top1"] += int(eq)
            s["max_dlogp"] = max(s["max_dlogp"], d)
            s["max_dlse"] = max(s["max_dlse"], dl)
            s["sum"] += d if math.isfinite(d) else 0.0

        for ci, (rc, oc) in enumerate(zip(ref["cases"], ours["cases"])):
            rows = rc["rows"]
            acc("prefill", rows[0], oc["prefill"])
            for j, o in enumerate(oc.get("tree", [])):
                if j < len(rows):
                    acc("tree", rows[j], o)
            for j, o in enumerate(oc.get("decode", [])):
                if j + 1 < len(rows):
                    acc("decode", rows[j + 1], o)
            rg, og = rc["gen"], oc.get("gen", [])
            n = 0
            while n < min(len(rg), len(og)) and rg[n] == og[n]:
                n += 1
            part = None
            if n < min(len(rg), len(og)):
                rm = rc["gen_margin"][n] if n < len(rc["gen_margin"]) else None
                part = {"at": n, "ref_margin": None if rm is None else round(rm, 4),
                        "ours_margin": round(oc["gen_margin"][n], 4)}
            gens.append({"case": ci, "prompt": rc["prompt_len"], "equal": n, "ref_len": len(rg), "parted": part})
        for s in summary.values():
            s["mean_dlogp"] = round(s.pop("sum") / max(1, s["rows"]), 5)
            s["max_dlogp"] = round(s["max_dlogp"], 5)
            s["max_dlse"] = round(s["max_dlse"], 5)
        report["logits"] = {"rows": summary, "generations": gens}
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("images")
    i.add_argument("--out", required=True)
    r = sub.add_parser("ref")
    r.add_argument("--model", required=True)
    r.add_argument("--images", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gen", type=int, default=64)
    r.add_argument("--forced", type=int, default=48, help="generated tokens teacher-forced after the prompt")
    r.add_argument("--bidi", choices=["sliding", "all", "none"], default="sliding")
    r.add_argument("--fp32", action="store_true")
    r.add_argument("--exact-pixels", action="store_true", help="with --fp32: the tower's input not rounded to bf16")
    r.add_argument("--vision-only", action="store_true", help="the processor and the tower only (no prompts)")
    r.add_argument("--features", help="vision_dump output directory: its soft tokens replace mlx-vlm's tower in the prompts")
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--vision")
    c.add_argument("--logits")
    a = ap.parse_args()
    {"images": cmd_images, "ref": cmd_ref, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
