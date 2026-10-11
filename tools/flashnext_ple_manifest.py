#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Manifest for mlx-vlm's SSD-served n-gram table (qwen4_exp `ple_storage`,
QuantizedMMapNGramEmbedding, layout "safetensors_ranges", version 2).

Qwen3.8-Flash-Next's n-gram embedding is 51B parameters (32 GB at 4-bit g32)
of which a token reads a few rows, so the table stays in the model's own
safetensors files and rows are read by byte range. This writes the manifest
from the files' headers (absolute offsets of each shard's weight / scales /
biases), and optionally a model view directory: symlinks to every file of the
model plus a config.json whose text_config.ple_storage points at the manifest.

  python3 tools/flashnext_ple_manifest.py MODEL_DIR --out ple-store.json [--view VIEW_DIR]
"""

import argparse
import json
import os
import re
import struct
from pathlib import Path

SHARD = re.compile(r"ngram_embedding\.shards?[._](\d+)\.(weight|scales|biases)$")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", help="directory for a model view that uses the manifest")
    a = ap.parse_args()
    model = Path(a.model).resolve()
    index = json.load(open(model / "model.safetensors.index.json"))["weight_map"]
    cfg = json.load(open(model / "config.json"))
    shards = {}
    headers = {}
    for name, fname in index.items():
        m = SHARD.search(name)
        if not m or ".mtp." in name or name.startswith("mtp."):
            continue
        if fname not in headers:
            headers[fname] = header(model / fname)
        base, h = headers[fname]
        e = h[name]
        shards.setdefault(int(m.group(1)), {})[m.group(2)] = {
            "file": fname, "offset": base + e["data_offsets"][0], "dtype": e["dtype"], "shape": e["shape"]}
    if not shards:
        raise SystemExit("no n-gram shard tensors in the index")
    q = cfg.get("quantization", {})
    qn = next((v for k, v in q.items() if isinstance(v, dict) and "ngram_embedding" in k), None) or \
        {"bits": q.get("bits", 4), "group_size": q.get("group_size", 32)}
    entries, start = [], 0
    for i in sorted(shards):
        s = shards[i]
        count = s["weight"]["shape"][0]
        entries.append({"row_start": start, "row_count": count, **{k: s[k] for k in ("weight", "scales", "biases")}})
        start += count
    width = entries[0]["weight"]["shape"][1] * 32 // int(qn["bits"])
    manifest = {"version": 2, "source_root": str(model), "layout": "safetensors_ranges", "row_width": width,
                "row_count": start, "quantization": {"bits": int(qn["bits"]), "group_size": int(qn["group_size"]),
                                                     "mode": "affine"}, "cache_rows": 0, "shards": entries}
    Path(a.out).write_text(json.dumps(manifest, indent=1))
    print(f"{len(entries)} shards, {start} rows x {width}, {len(headers)} files -> {a.out}")
    if a.view:
        view = Path(a.view)
        view.mkdir(parents=True, exist_ok=True)
        for f in model.iterdir():
            if f.name == "config.json" or f.name.startswith("."):
                continue
            dst = view / f.name
            if not dst.exists():
                os.symlink(f, dst)
        cfg.setdefault("text_config", {})["ple_storage"] = {"manifest": str(Path(a.out).resolve()), "cache_rows": 0}
        (view / "config.json").write_text(json.dumps(cfg, indent=1))
        print(f"view -> {view}")


if __name__ == "__main__":
    main()
