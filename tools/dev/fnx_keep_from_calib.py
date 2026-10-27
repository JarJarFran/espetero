#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""A --keep-experts list from an espetero quantize calibration file of
Qwen3.8-Flash-Next (qwen4_exp): per layer the K experts that saw the most
routed rows (the file's N:...switch_mlp.gate_proj counts), ids ascending. A
frequency baseline for tests; REAP-style pruning scores experts by more than
their counts. The MTP layer has no calibration rows: its list is the K
experts of largest router row norm (from the checkpoint), or omitted with
--no-mtp.

  python3 tools/dev/fnx_keep_from_calib.py CALIB.safetensors K --model DIR --out keep.json [--no-mtp]
"""

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def read(path, base, e):
    dt = {"F32": np.float32, "BF16": np.uint16, "F16": np.float16}[e["dtype"]]
    a, b = e["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        v = np.frombuffer(f.read(b - a), dtype=dt).reshape(e["shape"])
    if e["dtype"] == "BF16":
        v = (v.astype(np.uint32) << 16).view(np.float32)
    return v.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib")
    ap.add_argument("k", type=int)
    ap.add_argument("--model", help="checkpoint directory (the MTP router)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-mtp", action="store_true")
    a = ap.parse_args()
    base, h = header(a.calib)
    keep = {}
    for name, e in h.items():
        m = re.match(r"N:language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.gate_proj$", name)
        if m:
            cnt = read(a.calib, base, e)
            keep[m.group(1)] = sorted(int(i) for i in np.argsort(-cnt, kind="stable")[: a.k])
    if not keep:
        raise SystemExit(a.calib + ": no N: expert counts (a Qwen4-Exp capture of this build?)")
    if not a.no_mtp:
        if not a.model:
            raise SystemExit("--model DIR for the MTP layer's list (or --no-mtp)")
        d = Path(a.model)
        idx = d / "model.safetensors.index.json"
        files = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
        for f in files:
            fb, fh = header(d / f)
            if "mtp.layers.0.mlp.gate.weight" in fh:
                r = read(d / f, fb, fh["mtp.layers.0.mlp.gate.weight"])
                keep["mtp"] = sorted(int(i) for i in np.argsort(-np.linalg.norm(r, axis=1), kind="stable")[: a.k])
                break
        else:
            raise SystemExit(a.model + ": no mtp.layers.0.mlp.gate.weight")
    out = {k: keep[k] for k in sorted(keep, key=lambda x: (x == "mtp", int(x) if x.isdigit() else 0))}
    Path(a.out).write_text(json.dumps(out))
    print(f"{a.out}: {len(out)} lists of {a.k}")


if __name__ == "__main__":
    main()
