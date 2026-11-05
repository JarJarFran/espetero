#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero
"""Espetero's GLM-5.3-Flash dumps (build/glm5_run) against tools/glm5_ref.py's, layer by layer.

    python3 tests/ref/glm5_compare.py REF_DIR OURS_DIR [--mtp]

Per layer: the attention / KDA block's output (a), the MLP's (f) and the streams
after the layer (h) as relative L2 errors per row (worst and median), the routed
experts (rows whose top-8 sets differ) and the DSA selections (rows whose selected
token sets differ). A layer's input is the previous layer's output in both runs,
so errors accumulate; the per-block numbers show where they enter.
"""
import json
import os
import sys

import numpy as np


def rel_rows(a, b):
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    return np.sqrt(((a - b) ** 2).sum(1) / np.maximum((a ** 2).sum(1), 1e-30))


def main():
    ref, ours = sys.argv[1], sys.argv[2]
    case = json.load(open(os.path.join(ref, "case.json")))
    T, D = len(case["ids"]), 4096
    worst = 0.0
    for l in range(case["layers"]):
        parts = []
        for name, shape in (("a", (T, D)), ("f", (T, D)), ("h", (T, 4, D))):
            pa, pb = os.path.join(ref, f"{name}{l}.f32"), os.path.join(ours, f"{name}{l}.f32")
            if not (os.path.exists(pa) and os.path.exists(pb)):
                parts.append(f"{name}: missing")
                continue
            r = rel_rows(np.fromfile(pa, np.float32).reshape(shape), np.fromfile(pb, np.float32).reshape(shape))
            if name == "h":
                worst = max(worst, float(r.max()))
            parts.append(f"{name} {r.max():.2e} (med {np.median(r):.1e})")
        pr, po = os.path.join(ref, f"route{l}.i32"), os.path.join(ours, f"route{l}.i32")
        if os.path.exists(pr) and os.path.exists(po):
            a = np.sort(np.fromfile(pr, np.int32).reshape(T, -1), 1)
            b = np.sort(np.fromfile(po, np.int32).reshape(T, -1), 1)
            parts.append(f"routes differ in {int((a != b).any(1).sum())}/{T} rows")
        sr, so = os.path.join(ref, f"sel{l}.i32"), os.path.join(ours, f"sel{l}.i32")
        if os.path.exists(sr) and os.path.exists(so):
            a, b = np.fromfile(sr, np.int32).reshape(T, -1), np.fromfile(so, np.int32).reshape(T, -1)
            diff = sum(set(a[t][a[t] >= 0]) != set(b[t][b[t] >= 0]) for t in range(T))
            parts.append(f"selections differ in {diff}/{T} rows")
        print(f"layer {l}: " + " | ".join(parts))
    print(f"worst per-row relative L2 of the streams: {worst:.3g}")
    if "--mtp" in sys.argv and os.path.exists(os.path.join(ref, "mtp_u.f32")):
        n = T - 1
        lid = case["layers"]  # the MTP layer's index in Espetero's dumps (after the loaded trunk layers)
        parts = []
        for name, w in (("mtp_u", D), ("mtp_a", D), ("mtp_f", D), ("mtp_h", D), ("mtp_logits", None)):
            a = np.fromfile(os.path.join(ref, name + ".f32"), np.float32)
            b = np.fromfile(os.path.join(ours, name + ".f32"), np.float32)
            r = rel_rows(a.reshape(n, -1), b.reshape(n, -1))
            parts.append(f"{name[4:]} {r.max():.2e}")
            if name == "mtp_logits":
                am_a, am_b = a.reshape(n, -1).argmax(1), b.reshape(n, -1).argmax(1)
                parts.append(f"argmax equal {int((am_a == am_b).sum())}/{n}")
        ra = np.sort(np.fromfile(os.path.join(ref, "mtp_route.i32"), np.int32).reshape(n, -1), 1)
        rb = np.sort(np.fromfile(os.path.join(ours, "mtp", f"route{lid}.i32"), np.int32).reshape(n, -1), 1)
        parts.append(f"routes differ in {int((ra != rb).any(1).sum())}/{n} rows")
        print("MTP: " + " | ".join(parts))


if __name__ == "__main__":
    main()
