#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""A checkpoint view: the tensors of a partial espetero quantize output
(--only GLOBS) over another MLX checkpoint of the same model, without copying.

Every module the overlay writes replaces the base's module of that name (all
its tensors: .weight / .scales / .biases / .super / .codebook); the others
stay the base's. The view directory holds symlinks to both checkpoints'
safetensors files (prefixed base- / ovl-, as both are named model-*), a merged
model.safetensors.index.json (the base's files listed first: Espetero maps
every tensor of every listed file, later files winning, so the overlay's
replace the base's), and the overlay's config.json whose
"quantization" also lists the width of every module kept from the base (the
two files have different defaults). Used for Qwen3.8-Flash-Next smoke tests:
a few layers converted to EQ over the oQ4e-mtp file, or a pruned EQ conversion
that keeps the base's 32 GB n-gram table.

  python3 tools/dev/fnx_overlay.py BASE_DIR OVERLAY_DIR VIEW_DIR [--include REGEX]

--include: only the overlay's modules whose names match (bisecting which
converted matrices change the logits).
"""

import argparse
import json
import os
import re
import struct
from pathlib import Path

SUFFIXES = (".weight", ".scales", ".biases", ".super", ".codebook")


def module(name):
    for s in SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return name


def tensors(d):
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        out.update({k: f.name for k in h if k != "__metadata__"})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("overlay")
    ap.add_argument("view")
    ap.add_argument("--include", help="only the overlay modules matching this regex")
    a = ap.parse_args()
    base, ovl, view = (Path(x).resolve() for x in (a.base, a.overlay, a.view))
    view.mkdir(parents=True, exist_ok=True)
    bw, ow = tensors(base), tensors(ovl)
    if a.include:
        ow = {k: f for k, f in ow.items() if re.search(a.include, module(k))}
    replaced = {module(k) for k in ow}
    wm = {k: "base-" + f for k, f in bw.items() if module(k) not in replaced}
    wm.update({k: "ovl-" + f for k, f in ow.items()})
    for pre, d, files in (("base-", base, set(bw.values())), ("ovl-", ovl, set(ow.values()))):
        for f in files:
            dst = view / (pre + f)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(d / f, dst)
    for f in ovl.iterdir():  # tokenizer and the other small files
        if f.suffix in (".json", ".jinja", ".txt") and f.name not in ("config.json", "model.safetensors.index.json"):
            dst = view / f.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(f, dst)
    bcfg = json.loads((base / "config.json").read_text())
    cfg = json.loads((ovl / "config.json").read_text())
    bq = bcfg.get("quantization") or bcfg.get("quantization_config") or {}
    oq = cfg.get("quantization", {})
    q = {"bits": oq.get("bits", 8), "group_size": oq.get("group_size", 64), "mode": oq.get("mode", "affine")}
    for k in sorted({module(k) for k in wm if k.endswith(".weight")}):  # every quantized module, explicitly
        src = oq if k in replaced else bq
        if k in src and isinstance(src[k], dict):
            q[k] = src[k]
        elif k not in replaced and module(k) + ".scales" in bw:
            q[k] = {"bits": bq.get("bits", 4), "group_size": bq.get("group_size", 64)}
    cfg["quantization"] = cfg["quantization_config"] = q
    (view / "config.json").write_text(json.dumps(cfg, indent=1))
    total = sum(os.path.getsize(view / f) for f in set(wm.values()))
    (view / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": wm}, indent=1))
    print(f"{view}: {len(ow)} overlay tensors ({len(replaced)} modules), {len(wm) - len(ow)} from the base")


if __name__ == "__main__":
    main()
