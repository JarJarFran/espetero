#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Vision fidelity of Espetero against mlx-vlm.

Runs mlx-vlm's image path on the files that `build/vision_dump` processed and
compares, per image:

  * the processor: grid and patch rows (mlx-vlm's AutoProcessor with the same
    pixel budget as Espetero, after mlx-vlm's load_image: EXIF transpose + RGB);
  * the vision tower: Espetero's merged embeddings and mlx-vlm's as shipped
    (pixels cast to the weights' dtype, bf16 for these checkpoints), both
    against an fp32 copy of mlx-vlm's tower (fp32 parameters, fp32 pixels),
    the accuracy reference. Per-token cosine (mean / min) and relative
    Frobenius error; "tower only" runs the fp32 reference on Espetero's own
    pixels, so it isolates the kernels from the preprocessing.

  build/vision_dump <MLX model dir> <dump dir> <max_pixels> <images...>
  uv run --isolated --no-project --with mlx-vlm python tools/vision_fidelity.py \\
      --model <MLX model dir> --espetero <dump dir> [--out report.json]
"""

import argparse
import json
import os
import sys

import numpy as np


def metrics(a, ref):
    a = np.asarray(a, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if a.shape != r.shape:
        return {"shape": [list(a.shape), list(r.shape)]}
    na = np.linalg.norm(a, axis=-1)
    nr = np.linalg.norm(r, axis=-1)
    cos = (a * r).sum(-1) / np.maximum(na * nr, 1e-30)
    return {
        "cos_mean": float(cos.mean()),
        "cos_min": float(cos.min()),
        "tokens": int(cos.size),
        "tokens_cos_below_0999": int((cos < 0.999).sum()),
        "tokens_cos_below_099": int((cos < 0.99).sum()),
        "worst_token": int(cos.argmin()),
        "worst_token_norm_ratio": float(nr[cos.argmin()] / max(np.median(nr), 1e-30)),
        "rel_err": float(np.linalg.norm(a - r) / max(np.linalg.norm(r), 1e-30)),
        "max_abs": float(np.abs(a - r).max()),
    }


def processor_call(ip, img, max_pixels):
    # Same budget as Espetero (min_pixels = min(65536, max_pixels)); the keyword
    # names moved between transformers versions.
    mn = min(65536, max_pixels)
    tries = [
        dict(max_pixels=max_pixels, min_pixels=mn),
        dict(size={"longest_edge": max_pixels, "shortest_edge": mn}),
    ]
    last = None
    for kw in tries:
        try:
            return ip(images=[img], return_tensors="np", **kw), kw
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--espetero", required=True, help="vision_dump output directory (index.json)")
    ap.add_argument("--out")
    a = ap.parse_args()

    import mlx.core as mx
    from mlx.utils import tree_map
    from mlx_vlm import load
    from mlx_vlm.utils import load_image

    model, processor = load(a.model)
    ip = getattr(processor, "image_processor", processor)
    idx = json.load(open(os.path.join(a.espetero, "index.json")))
    maxp = int(idx["max_pixels"])
    vt = model.vision_tower
    wdt = vt.patch_embed.proj.weight.dtype
    print(f"mlx-vlm tower dtype {wdt}, image processor {type(ip).__name__}", file=sys.stderr)

    staged = []
    for e in idx["images"]:
        img = load_image(e["file"])
        feats, kw = processor_call(ip, img, maxp)
        pv = np.asarray(feats["pixel_values"], dtype=np.float32)
        grid = [int(x) for x in np.asarray(feats["image_grid_thw"]).reshape(-1)[:3]]
        up = np.fromfile(os.path.join(a.espetero, e["stem"] + ".pixels.f32"), dtype=np.float32).reshape(e["rows"], e["row_dim"])
        ue = np.fromfile(os.path.join(a.espetero, e["stem"] + ".emb.f32"), dtype=np.float32).reshape(e["tokens"], e["hidden"])
        same = grid == list(e["grid"]) and pv.shape == up.shape
        rec = {
            "file": os.path.basename(e["file"]),
            "size": [e["width"], e["height"]],
            "grid_espetero": e["grid"],
            "grid_mlx": grid,
            "pixels_max_abs": float(np.abs(pv - up).max()) if same else None,
            "pixels_equal": bool(same and np.array_equal(pv, up)),
        }
        h, _ = vt(mx.array(pv).astype(wdt), mx.array([grid]))
        mx.eval(h)
        staged.append((rec, pv, up, ue, grid, np.array(h.astype(mx.float32)), same))

    # fp32 reference: the same tower with every floating parameter in fp32.
    vt.update(tree_map(lambda p: p.astype(mx.float32) if isinstance(p, mx.array) and mx.issubdtype(p.dtype, mx.floating) else p,
                       vt.parameters()))
    out = []
    for rec, pv, up, ue, grid, mlx_def, same in staged:
        h32, _ = vt(mx.array(pv), mx.array([grid]))
        mx.eval(h32)
        ref = np.array(h32)
        rec["mlx_shipped_vs_fp32"] = metrics(mlx_def, ref)
        if same:
            h32u, _ = vt(mx.array(up), mx.array([grid]))
            mx.eval(h32u)
            rec["espetero_vs_fp32_tower_only"] = metrics(ue, np.array(h32u))
            rec["espetero_vs_fp32"] = metrics(ue, ref)
        out.append(rec)
        u = rec.get("espetero_vs_fp32", {})
        m = rec["mlx_shipped_vs_fp32"]
        print(f"{rec['file']:28s} grid U{rec['grid_espetero']} M{rec['grid_mlx']} px max|d| {rec['pixels_max_abs']}  "
              f"vs fp32: Espetero rel {u.get('rel_err', float('nan')):.2e} cos_min {u.get('cos_min', float('nan')):.6f} | "
              f"mlx-vlm rel {m['rel_err']:.2e} cos_min {m['cos_min']:.6f}")
    if out:
        agg = lambda k, f: [r[k][f] for r in out if k in r and f in r[k]]  # noqa: E731
        print("\nmean over images (vs the fp32 reference):")
        for k in ("espetero_vs_fp32", "espetero_vs_fp32_tower_only", "mlx_shipped_vs_fp32"):
            rel, cmin = agg(k, "rel_err"), agg(k, "cos_min")
            if rel:
                print(f"  {k:26s} rel_err {np.mean(rel):.3e} (max {np.max(rel):.3e})  cos_min {np.min(cmin):.6f}")
    if a.out:
        json.dump({"model": a.model, "max_pixels": maxp, "images": out}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
