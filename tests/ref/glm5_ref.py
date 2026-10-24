#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero
"""GLM-5.3-Flash (glm5_next) development helpers (docs/glm5.md).

  fake-ckpt OUT_DIR [--repo zai-org/GLM-5.3-Flash] [--revision main]
      A checkpoint directory without the weights: config.json, the safetensors
      index, the tokenizer files and every shard as a sparse file of its real
      size holding its real header (nothing else; APFS stores no data for the
      zeros). The loader's index and shape checks (build/glm5_load_test) run on
      it with no download. Needs the network for ~10 MB of headers.

  fake-gguf OUT_DIR --quant UD-IQ3_XXS [--fill-blocks 0-3]
      The same for unsloth's GGUF split files (unsloth/GLM-5.3-Flash-GGUF): every
      shard sparse with its real header; --fill-blocks fetches the tensors of those
      blocks plus token_embd and output_norm (UD-IQ3_XXS blocks 0-3: ~4 GB).

The numpy reference forward (teacher-forced logits for the parity checks of
docs/glm5.md 11) comes with the kernels.
"""
import argparse
import json
import os
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SMALL = ["config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json",
         "tokenizer_config.json", "chat_template.jinja"]


def _get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fake_ckpt(out, repo, revision):
    os.makedirs(out, exist_ok=True)
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    for f in SMALL:
        p = os.path.join(out, f)
        if not os.path.exists(p):
            open(p, "wb").write(_get(base + f))
    idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    for i, f in enumerate(shards):
        p = os.path.join(out, f)
        if os.path.exists(p):
            continue
        n = struct.unpack("<Q", _get(base + f, (0, 7)))[0]
        hdr = _get(base + f, (8, 8 + n - 1))
        h = json.loads(hdr)
        end = max(v["data_offsets"][1] for k, v in h.items() if k != "__metadata__")
        with open(p + ".part", "wb") as fh:
            fh.write(struct.pack("<Q", n))
            fh.write(hdr)
            fh.truncate(8 + n + end)  # sparse: the data region is never written
        os.rename(p + ".part", p)
        print(f"{f}: header {n} B, {8 + n + end} B logical ({i + 1}/{len(shards)})", file=sys.stderr)
    print(f"fake checkpoint in {out}: {len(shards)} sparse shards", file=sys.stderr)


def _layers(spec):
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def fill(out, repo, revision, layers, jobs=8, piece=64 << 20):
    """Real bytes for the tensors of `layers` and the top-level text tensors, in place."""
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    done_path = os.path.join(out, ".filled.json")
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    want = []
    for name, f in idx.items():
        if name.startswith("model.visual.") or name in done:
            continue
        parts = name.split(".")
        if "layers" in parts:
            if int(parts[parts.index("layers") + 1]) not in layers:
                continue
        want.append((name, f))
    headers = {}
    tasks = []  # (file, abs offset, nbytes, tensor)
    for name, f in want:
        if f not in headers:
            with open(os.path.join(out, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                headers[f] = (n, json.loads(fh.read(n)))
        n, h = headers[f]
        a, b = h[name]["data_offsets"]
        for o in range(a, b, piece):
            tasks.append((f, 8 + n + o, min(piece, b - o), name))
    total = sum(t[2] for t in tasks)
    print(f"fill: {len(want)} tensors, {total / 1e9:.2f} GB in {len(tasks)} ranges", file=sys.stderr)
    fds = {}
    for f, *_ in tasks:
        if f not in fds:
            fds[f] = os.open(os.path.join(out, f), os.O_WRONLY)
    left = {}
    for _, _, _, name in tasks:
        left[name] = left.get(name, 0) + 1
    got = [0]

    def one(t):
        f, off, nb, name = t
        for attempt in range(5):
            try:
                data = _get(base + f, (off, off + nb - 1))
                if len(data) != nb:
                    raise IOError(f"short read {len(data)} of {nb}")
                os.pwrite(fds[f], data, off)
                return t
            except Exception as e:  # noqa: BLE001 (retry any network failure)
                if attempt == 4:
                    raise
                print(f"retry {f}@{off}: {e}", file=sys.stderr)

    with ThreadPoolExecutor(jobs) as ex:
        for t in ex.map(one, tasks):
            got[0] += t[2]
            left[t[3]] -= 1
            if left[t[3]] == 0:
                done.add(t[3])
            if len(done) % 200 == 0 or got[0] == total:
                json.dump(sorted(done), open(done_path, "w"))
                print(f"  {got[0] / 1e9:.2f} / {total / 1e9:.2f} GB", file=sys.stderr)
    for fd in fds.values():
        os.close(fd)
    json.dump(sorted(done), open(done_path, "w"))
    print(f"filled layers {sorted(layers)} + top-level tensors in {out}", file=sys.stderr)


# GGUF: (values, bytes) per block of each ggml type
GGML_BLOCK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144),
              13: (256, 176), 14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50),
              20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136), 29: (256, 56)}


def gguf_header(data):
    """(tensors {name: (dims, type, absolute offset, nbytes)}, header length) of a GGUF file's first bytes."""
    p = [4]

    def take(fmt):
        v = struct.unpack_from(fmt, data, p[0])
        p[0] += struct.calcsize(fmt)
        return v[0]

    def string():
        n = take("<Q")
        v = data[p[0]:p[0] + n].decode("utf-8", "replace")
        p[0] += n
        return v

    sizes = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = take("<I"), take("<Q")
            return [value(et) for _ in range(n)]
        return take(sizes[t])

    assert data[:4] == b"GGUF"
    take("<I")
    nt, nkv = take("<Q"), take("<Q")
    align = 32
    for _ in range(nkv):
        k = string()
        v = value(take("<I"))
        if k == "general.alignment":
            align = v
    infos = []
    for _ in range(nt):
        name = string()
        dims = [take("<Q") for _ in range(take("<I"))]
        t = take("<I")
        infos.append((name, dims, t, take("<Q")))
    start = (p[0] + align - 1) // align * align
    out = {}
    for name, dims, t, off in infos:
        n = 1
        for d in dims:
            n *= d
        bv, bb = GGML_BLOCK[t]
        out[name] = (dims, t, start + off, n // bv * bb)
    return out, start


def fake_gguf(out, quant, blocks, repo="unsloth/GLM-5.3-Flash-GGUF", jobs=8, piece=64 << 20, extra=()):
    os.makedirs(out, exist_ok=True)
    api = json.loads(_get(f"https://huggingface.co/api/models/{repo}"))
    files = sorted(s["rfilename"] for s in api["siblings"] if s["rfilename"].startswith(quant + "/") and s["rfilename"].endswith(".gguf"))
    base = f"https://huggingface.co/{repo}/resolve/main/"
    tasks = []
    for f in files:
        p = os.path.join(out, os.path.basename(f))
        hdr_n = 1 << 20
        while True:
            data = _get(base + f, (0, hdr_n - 1))
            try:
                ts, start = gguf_header(data)
                break
            except (struct.error, IndexError, AssertionError):
                hdr_n *= 4
                if hdr_n > (64 << 20):
                    raise
        end = max([o + n for _, _, o, n in ts.values()] + [start])
        if not os.path.exists(p):
            with open(p + ".part", "wb") as fh:
                fh.write(data[:start])
                fh.truncate(end)
            os.rename(p + ".part", p)
        for name, (dims, t, o, n) in ts.items():
            keep = name in ("token_embd.weight", "output_norm.weight") or name in extra
            if name.startswith("blk."):
                keep = int(name.split(".")[1]) in blocks
            if keep:
                for a in range(0, n, piece):
                    tasks.append((f, os.path.basename(f), o + a, min(piece, n - a)))
        print(f"{os.path.basename(f)}: header {start} B, {end} B logical, {len(ts)} tensors", file=sys.stderr)
    done_path = os.path.join(out, ".filled.json")
    done = set(tuple(x) for x in json.load(open(done_path))) if os.path.exists(done_path) else set()
    tasks = [t for t in tasks if (t[1], t[2]) not in done]
    total = sum(t[3] for t in tasks)
    print(f"fill: {total / 1e9:.2f} GB in {len(tasks)} ranges", file=sys.stderr)
    fds = {}
    for _, b, _, _ in tasks:
        if b not in fds:
            fds[b] = os.open(os.path.join(out, b), os.O_WRONLY)

    def one(t):
        f, b, off, nb = t
        for attempt in range(5):
            try:
                data = _get(base + f, (off, off + nb - 1))
                if len(data) != nb:
                    raise IOError("short read")
                os.pwrite(fds[b], data, off)
                return t
            except Exception as e:  # noqa: BLE001
                if attempt == 4:
                    raise
                print(f"retry {b}@{off}: {e}", file=sys.stderr)

    got = 0
    with ThreadPoolExecutor(jobs) as ex:
        for t in ex.map(one, tasks):
            got += t[3]
            done.add((t[1], t[2]))
            if len(done) % 20 == 0:
                json.dump(sorted(done), open(done_path, "w"))
                print(f"  {got / 1e9:.2f} / {total / 1e9:.2f} GB", file=sys.stderr)
    json.dump(sorted(done), open(done_path, "w"))
    json.dump({"quant": quant, "blocks": sorted(blocks)}, open(os.path.join(out, ".slice.json"), "w"))
    print(f"filled blocks {sorted(blocks)} of {quant} in {out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fc = sub.add_parser("fake-ckpt")
    fc.add_argument("out_dir")
    fc.add_argument("--repo", default="zai-org/GLM-5.3-Flash")
    fc.add_argument("--revision", default="main")
    fc.add_argument("--fill-layers", default="", help="e.g. 0-3: fetch these layers' tensors (and embed/norm/head)")
    fg = sub.add_parser("fake-gguf")
    fg.add_argument("out_dir")
    fg.add_argument("--quant", default="UD-IQ3_XXS")
    fg.add_argument("--fill-blocks", default="0-3")
    fg.add_argument("--extra", default="", help="more tensors to fetch, comma-separated (e.g. output.weight)")
    args = ap.parse_args()
    if args.cmd == "fake-gguf":
        fake_gguf(args.out_dir, args.quant, _layers(args.fill_blocks), extra=tuple(x for x in args.extra.split(",") if x))
        return
    if args.cmd == "fake-ckpt":
        fake_ckpt(args.out_dir, args.repo, args.revision)
        if args.fill_layers:
            fill(args.out_dir, args.repo, args.revision, _layers(args.fill_layers))


if __name__ == "__main__":
    main()
