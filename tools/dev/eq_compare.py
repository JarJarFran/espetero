#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Compare two EQ checkpoints tensor by tensor (espetero quantize against
eq_convert.py): names, dtypes and shapes must agree; per tensor the share of
identical elements (codes unpacked per matrix from the config's bits) and,
for float tensors, the largest difference. numpy only.

    tools/dev/eq_compare.py DIR_A DIR_B [--verbose]
"""

import json
import os
import struct
import sys

import numpy as np

DT = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "U32": np.uint32, "I8": np.int8}


def index(d):
    out = {}
    wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    for f in sorted(set(wm.values())):
        p = os.path.join(d, f)
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (p, 8 + n, v)
    return out


def read(e):
    p, base, v = e
    a, b = v["data_offsets"]
    with open(p, "rb") as f:
        f.seek(base + a)
        return np.frombuffer(f.read(b - a), dtype=DT[v["dtype"]]).reshape(v["shape"])


def unpack(w, bits):
    w = w.reshape(-1, w.shape[-1])
    by = w.view(np.uint8)
    bitsarr = np.unpackbits(by, axis=-1, bitorder="little")
    C = bitsarr.shape[-1] // bits
    return (bitsarr.reshape(w.shape[0], C, bits).astype(np.uint32) << np.arange(bits, dtype=np.uint32)).sum(-1)


def main():
    A, B = sys.argv[1], sys.argv[2]
    verbose = "--verbose" in sys.argv
    ia, ib = index(A), index(B)
    qa = json.load(open(os.path.join(A, "config.json")))["quantization"]
    qb = json.load(open(os.path.join(B, "config.json")))["quantization"]
    only = sorted(set(ia) ^ set(ib))
    if only:
        print("names in one checkpoint only:", only[:20], len(only))
    bad_meta = 0
    agg = {}
    worst = []
    for k in sorted(set(ia) & set(ib)):
        va, vb = ia[k][2], ib[k][2]
        if va["dtype"] != vb["dtype"] or va["shape"] != vb["shape"]:
            print("dtype/shape differ:", k, va["dtype"], va["shape"], vb["dtype"], vb["shape"])
            bad_meta += 1
            continue
        x, y = read(ia[k]), read(ib[k])
        base, kind = k.rsplit(".", 1)
        if kind == "weight" and va["dtype"] == "U32" and base in qa:
            if qa[base] != qb.get(base):
                print("quantization differs:", base, qa[base], qb.get(base))
            bits = qa[base]["bits"]
            x, y = unpack(x, bits), unpack(y, bits)
            kind = "codes"
        same = float(np.mean(x == y)) if x.size else 1.0
        if va["dtype"] in ("F16", "F32", "BF16"):
            fx = x.astype(np.float32) if va["dtype"] != "BF16" else (x.astype(np.uint32) << 16).view(np.float32)
            fy = y.astype(np.float32) if va["dtype"] != "BF16" else (y.astype(np.uint32) << 16).view(np.float32)
            md = float(np.max(np.abs(fx - fy))) if x.size else 0.0
        else:
            md = float(np.max(np.abs(x.astype(np.int64) - y.astype(np.int64)))) if x.size else 0
        t = agg.setdefault(kind, [0, 0])
        t[0] += int(np.sum(x == y))
        t[1] += x.size
        worst.append((same, k, md))
        if verbose:
            print(f"{same:.6f}  max|d| {md:.3g}  {k}")
    print(f"{len(set(ia) & set(ib))} common tensors, {bad_meta} dtype/shape mismatches")
    for kind, (s, n) in sorted(agg.items()):
        print(f"  {kind:10s} identical {s / max(n, 1) * 100:.4f}% of {n}")
    worst.sort()
    for s, k, md in worst[:8]:
        print(f"  worst {s:.6f}  max|d| {md:.3g}  {k}")


if __name__ == "__main__":
    main()
