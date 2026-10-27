#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""kld_harness.py calib statistics (.npz) -> the safetensors form espetero
quantize reads (--calib FILE): the same "H:" / "D:" / "E:" / "X:" float32
tensors, the ALIAS map in the metadata. numpy only; tensors are streamed one
at a time.

    tools/dev/calib_npz.py calib.npz calib.safetensors
"""

import json
import struct
import sys

import numpy as np


def main():
    src, dst = sys.argv[1], sys.argv[2]
    z = np.load(src)
    keys = sorted(k for k in z.files if k != "ALIAS")
    meta = {"source": src}
    if "ALIAS" in z.files:
        meta["ALIAS"] = str(z["ALIAS"])
    header, off, shapes = {"__metadata__": meta}, 0, {}
    for k in keys:  # shapes from the .npy headers (np.load reads them lazily per key)
        with z.zip.open(k + ".npy") as f:
            ver = np.lib.format.read_magic(f)
            rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = rd(f)
        if dtype != np.float32 or fortran:
            raise SystemExit(f"{k}: expected C-order float32, got {dtype}")
        n = int(np.prod(shape)) * 4
        header[k] = {"dtype": "F32", "shape": list(shape), "data_offsets": [off, off + n]}
        shapes[k] = shape
        off += n
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(dst + ".partial", "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for k in keys:
            a = np.ascontiguousarray(z[k], dtype=np.float32)
            assert a.shape == tuple(shapes[k])
            f.write(a.tobytes())
    import os
    os.replace(dst + ".partial", dst)
    print(f"{dst}: {len(keys)} tensors, {off / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
