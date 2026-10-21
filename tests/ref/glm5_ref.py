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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fc = sub.add_parser("fake-ckpt")
    fc.add_argument("out_dir")
    fc.add_argument("--repo", default="zai-org/GLM-5.3-Flash")
    fc.add_argument("--revision", default="main")
    args = ap.parse_args()
    if args.cmd == "fake-ckpt":
        fake_ckpt(args.out_dir, args.repo, args.revision)


if __name__ == "__main__":
    main()
