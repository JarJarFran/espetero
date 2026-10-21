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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fc = sub.add_parser("fake-ckpt")
    fc.add_argument("out_dir")
    fc.add_argument("--repo", default="zai-org/GLM-5.3-Flash")
    fc.add_argument("--revision", default="main")
    fc.add_argument("--fill-layers", default="", help="e.g. 0-3: fetch these layers' tensors (and embed/norm/head)")
    args = ap.parse_args()
    if args.cmd == "fake-ckpt":
        fake_ckpt(args.out_dir, args.repo, args.revision)
        if args.fill_layers:
            fill(args.out_dir, args.repo, args.revision, _layers(args.fill_layers))


if __name__ == "__main__":
    main()
