#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero
"""Reference data for tests/gguf_iq_test.mm (IQ2_S, IQ3_S: gguf.h's conversions) and
tests/gguf_gemv_test.mm (every block type the GLM-5.3-Flash GGUF files use: the
native GEMV kernels): GGUF blocks and their dequantized values from gguf-py
(llama.cpp's numpy implementation of ggml's dequantize_row_*), an implementation
independent of Espetero's.

    python3 tests/ref/gguf_iq_ref.py OUT.bin [--types IQ2_S,IQ3_S] [--real TYPE:ROWS:COLS:FILE ...]

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

# block size in values and bytes; byte offsets of the fp16 super-scale(s) in a block
BLOCK = {T.Q8_0: (32, 34, (0,)), T.Q2_K: (256, 84, (80, 82)), T.Q3_K: (256, 110, (108,)), T.Q4_K: (256, 144, (0, 2)),
         T.Q5_K: (256, 176, (0, 2)), T.Q6_K: (256, 210, (208,)), T.IQ4_XS: (256, 136, (0,)), T.IQ2_S: (256, 82, (0,)),
         T.IQ3_S: (256, 110, (0,))}


def random_blocks(t, rows, cols, rng):
    qk, nbytes, dpos = BLOCK[t]
    nb = rows * cols // qk
    b = rng.integers(0, 256, size=(nb, nbytes), dtype=np.uint8)
    for o in dpos:  # positive super-scales of moderate size, as in real files
        d = (rng.uniform(1e-4, 2e-2, size=nb)).astype(np.float16)
        b[:, o:o + 2] = d.view(np.uint8).reshape(nb, 2)
    return b.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--real", action="append", default=[])
    ap.add_argument("--types", default="IQ2_S,IQ3_S", help="block types of the random cases (comma-separated)")
    args = ap.parse_args()
    rng = np.random.default_rng(20261022)
    cases = []
    for t in (T[x] for x in args.types.split(",")):
        for rows, cols in ((7, 256), (33, 4096), (16, 2048)):
            cases.append((t, rows, cols, random_blocks(t, rows, cols, rng)))
    for spec in args.real:
        name, rows, cols, path = spec.split(":", 3)
        t = T[name]
        rows, cols = int(rows), int(cols)
        data = open(path, "rb").read()
        need = rows * cols // BLOCK[t][0] * BLOCK[t][1]
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
