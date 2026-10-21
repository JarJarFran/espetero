#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero
"""Reference data for tests/gguf_iq_test.mm: GGUF codebook blocks (IQ2_S, IQ3_S) and
their dequantized values from gguf-py (llama.cpp's numpy implementation of
ggml's dequantize_row_*), an implementation independent of Espetero's.

    python3 tests/ref/gguf_iq_ref.py OUT.bin [--real TYPE:ROWS:COLS:FILE ...]

Random blocks cover every grid index, sign pattern and scale nibble (the fp16
super-scale positive and of moderate size, as in real files); --real adds rows
cut from a real GGUF tensor (e.g. 64 rows of an expert matrix fetched with an
HTTP range). gguf-py: `pip install gguf`, or GGUF_PY=dir holding its `gguf`
package.

Format: "GIQREF01", u32 cases; per case: u32 ggml type, u32 rows, u32 cols,
u32 nbytes, the blocks, then rows * cols float32 values.
"""
import argparse
import os
import struct
import sys

import numpy as np

if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])
from gguf.constants import GGMLQuantizationType as T  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

BLOCK = {T.IQ2_S: 82, T.IQ3_S: 110}


def random_blocks(t, rows, cols, rng):
    nb = rows * cols // 256
    b = rng.integers(0, 256, size=(nb, BLOCK[t]), dtype=np.uint8)
    d = (rng.uniform(1e-4, 2e-2, size=nb)).astype(np.float16)
    b[:, 0:2] = d.view(np.uint8).reshape(nb, 2)
    return b.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--real", action="append", default=[])
    args = ap.parse_args()
    rng = np.random.default_rng(20261022)
    cases = []
    for t in (T.IQ2_S, T.IQ3_S):
        for rows, cols in ((7, 256), (33, 4096), (16, 2048)):
            cases.append((t, rows, cols, random_blocks(t, rows, cols, rng)))
    for spec in args.real:
        name, rows, cols, path = spec.split(":", 3)
        t = T[name]
        rows, cols = int(rows), int(cols)
        data = open(path, "rb").read()
        need = rows * cols // 256 * BLOCK[t]
        if len(data) < need:
            sys.exit(f"{path}: {len(data)} bytes, {need} needed")
        cases.append((t, rows, cols, data[:need]))
    with open(args.out, "wb") as f:
        f.write(b"GIQREF01")
        f.write(struct.pack("<I", len(cases)))
        for t, rows, cols, data in cases:
            raw = np.frombuffer(data, dtype=np.uint8).reshape(rows, -1)
            ref = dequantize(raw, t).astype(np.float32).reshape(rows, cols)
            f.write(struct.pack("<4I", int(t), rows, cols, len(data)))
            f.write(data)
            f.write(ref.tobytes())
            print(f"{t.name} {rows} x {cols}: max |w| {np.abs(ref).max():.4g}", file=sys.stderr)


if __name__ == "__main__":
    main()
