#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Convert an LFM2 / LFM2-VL checkpoint (Hugging Face layout: LiquidAI/LFM2.5-VL-*,
the d1 decision models built on it, LFM2 text models) into a directory
`espetero serve` loads.

    python tools/lfm2_convert.py HF_SNAPSHOT_DIR OUT_DIR [--bits 8]

What changes (nothing else is touched):
  * tensor names: model.language_model.* -> language_model.model.*,
    model.vision_tower.vision_model.* -> vision_tower.*, model.multi_modal_projector.* ->
    multi_modal_projector.* (an lfm2 text model: model.* -> language_model.model.*,
    lm_head.weight -> language_model.lm_head.weight);
  * --bits 8 (or 4): the decoder matrices (conv in/out projections, q/k/v/out projections,
    the MLP's w1/w2/w3) quantized to MLX affine, group 64 (mx.quantize); default: bf16 kept
    (Espetero serves it as dense fp16). The embedding (tied LM head), the norms, the conv
    taps, the vision tower and the projector always stay bf16.

The tokenizer, the chat template and the processor and generation configurations are
hard-linked unchanged (copied across file systems; config.json is rewritten when
quantizing). Python files a release ships are not copied: Espetero runs none of them.
Espetero also loads the Hugging Face snapshot itself (bf16); converting is needed for
the quantized widths. Needs `mlx` (any venv with it).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import mlx.core as mx

COPY = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "README.md",
    "LICENSE",
]
QUANTIZED = re.compile(
    r"^language_model\.model\.layers\.\d+\.(conv\.(in_proj|out_proj)|self_attn\.(q_proj|k_proj|v_proj|out_proj)|"
    r"feed_forward\.(w1|w2|w3))\.weight$"
)


def rename(name: str, vl: bool) -> str:
    if vl:
        for a, b in (("model.language_model.", "language_model.model."),
                     ("model.vision_tower.vision_model.", "vision_tower."),
                     ("model.multi_modal_projector.", "multi_modal_projector.")):
            if name.startswith(a):
                return b + name[len(a):]
        if name == "lm_head.weight":
            return "language_model.lm_head.weight"
    else:
        if name.startswith("model."):
            return "language_model.model." + name[len("model."):]
        if name == "lm_head.weight":
            return "language_model.lm_head.weight"
    raise SystemExit(f"unexpected tensor {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=0, choices=[0, 4, 8], help="0: keep bf16 (default); 8 or 4: quantized decoder")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    cfg = json.loads((src / "config.json").read_text())
    kind = cfg.get("model_type")
    if kind not in ("lfm2_vl", "lfm2"):
        raise SystemExit(f"{src}: model_type {kind!r} is not lfm2_vl or lfm2")
    vl = kind == "lfm2_vl"
    out.mkdir(parents=True, exist_ok=True)
    index_file = src / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index_file.read_text())["weight_map"].values())) if index_file.exists() else ["model.safetensors"]
    weight_map: dict[str, str] = {}
    n_quant = 0
    for i, shard in enumerate(shards):
        tensors = mx.load(str(src / shard))
        dst: dict[str, mx.array] = {}
        for name, value in tensors.items():
            new = rename(name, vl)
            if args.bits and QUANTIZED.match(new):
                w, scales, biases = mx.quantize(value, group_size=64, bits=args.bits)
                base = new[: -len(".weight")]
                dst[new], dst[base + ".scales"], dst[base + ".biases"] = w, scales, biases
                n_quant += 1
            else:
                dst[new] = value
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(out / name), dst, metadata={"format": "mlx"})
        weight_map.update({k: name for k in dst})
        print(f"{shard} -> {name}: {len(dst)} tensors", flush=True)
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, indent=2)
    )
    for f in COPY:
        if (src / f).exists():
            dst_file = out / f
            if dst_file.exists():
                dst_file.unlink()
            try:  # one physical copy: hard-link the (HF cache) blob, copy across file systems
                os.link(os.path.realpath(src / f), dst_file)
            except OSError:
                shutil.copyfile(src / f, dst_file)
    if args.bits:
        (out / "config.json").unlink()  # a hard link to the release's file: write a new one
        q = {"group_size": 64, "bits": args.bits, "mode": "affine"}
        cfg["quantization"] = q
        cfg["quantization_config"] = q
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"done: {len(weight_map)} tensors, {n_quant} matrices quantized -> {out}")


if __name__ == "__main__":
    main()
