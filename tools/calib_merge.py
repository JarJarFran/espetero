"""Merge calibration files captured over layer ranges (espetero quantize --calib-only
--calib-layers A:B) into one statistics file for --calib.

usage: python3 tools/calib_merge.py OUT.safetensors PART.safetensors [PART ...]

Tensors must be disjoint across parts and every part must cover the same tokens;
the ALIAS maps are merged. Data is streamed tensor by tensor (memory stays small).
"""
import json
import struct
import sys


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def main():
    out, parts = sys.argv[1], sys.argv[2:]
    if not parts:
        sys.exit(__doc__)
    tensors, alias, tokens, source = {}, {}, None, None
    for p in parts:
        h, base = header(p)
        meta = h.pop("__metadata__", {})
        t = meta.get("tokens")
        if tokens is not None and t != tokens:
            sys.exit(f"{p}: {t} tokens, other parts {tokens}")
        tokens, source = t, meta.get("source", source)
        alias.update(json.loads(meta.get("ALIAS", "{}")))
        for k, v in h.items():
            if k in tensors:
                sys.exit(f"{p}: {k} also in {tensors[k][0]}")
            tensors[k] = (p, base, v)
    names = sorted(tensors)
    hdr, off = {}, 0
    for k in names:
        _, _, v = tensors[k]
        size = v["data_offsets"][1] - v["data_offsets"][0]
        hdr[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + size]}
        off += size
    hdr["__metadata__"] = {"ALIAS": json.dumps(alias), "tokens": tokens, "source": source or ""}
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for k in names:
            p, base, v = tensors[k]
            a, b = v["data_offsets"]
            with open(p, "rb") as src:
                src.seek(base + a)
                left = b - a
                while left:
                    chunk = src.read(min(left, 64 << 20))
                    f.write(chunk)
                    left -= len(chunk)
    print(f"{out}: {len(names)} tensors from {len(parts)} parts, {off / 1e9:.2f} GB, {len(alias)} aliases, "
          f"{tokens} tokens")


main()
