"""Mix calibration captures of one model into a calibration set for --calib:
H = sum_j w_j H_j per matrix (weights normalized to sum 1; each H_j is E[x x^T] per token).
usage: python3 tools/calib_mix.py OUTDIR W1:FILES1 W2:FILES2 ...
FILESj: the comma-separated parts of one capture (any --calib-layers split). The output has the
first source's file names and partition (OUTDIR/<name>); per matrix, the full E[x x^T] (H:, or
packed U:) when every source has one, else the diagonals (D:). ALIAS from the first source.
Needs numpy. Prints the files written. The Qwen3.8-27B release file's calibration (v6a):
  python3 tools/calib_mix.py v6a 0.5:part-0-8.safetensors,...,part-56-64.safetensors \
      0.1667:wiki-0-32.safetensors,wiki-32-64.safetensors 0.1667:cc-0-32.safetensors,cc-32-64.safetensors \
      0.1667:chat-0-32.safetensors,chat-32-64.safetensors
"""
import json
import os
import struct
import sys

import numpy as np

DT = {"F32": np.float32}


class Src:
    def __init__(self, files):
        self.files = files
        self.where = {}  # key (module) -> (file, kind, tensor name)
        self.hdr = {}
        self.alias = {}
        self.tokens = 0
        for f in files:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            meta = h.pop("__metadata__", {}) or {}
            self.alias.update(json.loads(meta.get("ALIAS", "{}")))
            self.tokens = max(self.tokens, int(meta.get("tokens", "0")))
            self.hdr[f] = (h, 8 + n, meta)
            for k in h:
                kind, mod = k.split(":", 1)
                self.where[mod] = (f, kind, k)

    def read(self, f, k):
        h, base, _ = self.hdr[f]
        v = h[k]
        a, b = v["data_offsets"]
        return np.fromfile(f, dtype=np.float32, count=(b - a) // 4, offset=base + a).reshape(v["shape"])

    def get(self, mod):
        """('H', [C, C]) or ('D', [C]) for module key mod (its alias resolved)."""
        mod = self.alias.get(mod, mod)
        if mod not in self.where:
            return None
        f, kind, k = self.where[mod]
        x = self.read(f, k)
        if kind == "U":
            n = x.shape[0]
            C = int((np.sqrt(8.0 * n + 1.0) - 1.0) / 2.0 + 0.5)
            H = np.zeros((C, C), np.float32)
            iu = np.triu_indices(C)
            H[iu] = x
            H.T[iu] = x
            return "H", H
        return kind, x


def diag(t):
    return np.diagonal(t[1]).copy() if t[0] == "H" else t[1]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out = sys.argv[1]
    srcs, ws = [], []
    for s in sys.argv[2:]:
        w, files = s.split(":", 1)
        ws.append(float(w))
        srcs.append(Src(files.split(",")))
    ws = np.array(ws) / sum(ws)
    os.makedirs(out, exist_ok=True)
    s0 = srcs[0]
    for f in s0.files:
        h0, _, meta0 = s0.hdr[f]
        tensors = []
        for k in h0:
            kind0, mod = k.split(":", 1)
            got = [s.get(mod) for s in srcs]
            if any(g is None for g in got):
                print(f"skip {mod}: missing in a source", file=sys.stderr)
                continue
            if all(g[0] == "H" for g in got):
                M = sum(w * g[1].astype(np.float64) for w, g in zip(ws, got)).astype(np.float32)
                if kind0 == "U":
                    tensors.append(("U:" + mod, M[np.triu_indices(M.shape[0])]))
                else:
                    tensors.append(("H:" + mod, M))
            else:
                tensors.append(("D:" + mod, sum(w * diag(g).astype(np.float64) for w, g in zip(ws, got)).astype(np.float32)))
        hdr, off = {}, 0
        for name, a in tensors:
            hdr[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
            off += a.nbytes
        meta = {"ALIAS": meta0.get("ALIAS", "{}"), "tokens": str(int(sum(w * s.tokens for w, s in zip(ws, srcs)))),
                "source": "calib_mix " + " ".join(f"{w:.4f}:{','.join(os.path.basename(x) for x in s.files)}" for w, s in zip(ws, srcs))}
        hdr["__metadata__"] = meta
        hb = json.dumps(hdr).encode()
        hb += b" " * ((8 - len(hb) % 8) % 8)
        dst = os.path.join(out, os.path.basename(f))
        with open(dst + ".tmp", "wb") as fo:
            fo.write(struct.pack("<Q", len(hb)))
            fo.write(hb)
            for _, a in tensors:
                fo.write(np.ascontiguousarray(a).tobytes())
        os.replace(dst + ".tmp", dst)
        print(dst, len(tensors), "tensors", flush=True)


if __name__ == "__main__":
    main()
