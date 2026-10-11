#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Convert a Cloudflare Clef release (Hugging Face layout: Qwen3_5ForConditionalGeneration
backbone + joint schema head) into a directory `espetero serve` loads.

    python tools/clef_convert.py HF_SNAPSHOT_DIR OUT_DIR [--bits 8]

What changes (nothing else is touched):
  * tensor names: model.language_model.* -> language_model.model.*, lm_head.weight ->
    language_model.lm_head.weight, model.visual.* -> vision_tower.* (the MLX names Espetero reads);
  * the zero-centred RMSNorm weights (input/post-attention, q/k norms, final norm) are stored as
    1 + w in float32 (HF Qwen3.5 computes x * (1 + w); Espetero, MLX and llama.cpp store 1 + w;
    float32 avoids the bf16 rounding of the sum). The DeltaNet output norm stays as it is;
  * --bits 8: decoder matrices (attention, DeltaNet and MLP projections) quantized to MLX
    affine 8-bit, group 64 (mx.quantize); default: bf16 kept (Espetero serves it as dense fp16).
    The embedding, the LM head and the vision tower always stay bf16 (the joint head reads
    LM head rows on the host, exactly as the release does).

The joint head (joint_head.safetensors, joint_head_config.json), the tokenizer, the chat
template and the processor configuration are hard-linked unchanged (copied across file
systems; config.json is rewritten when quantizing). Needs `mlx` (any venv with it).
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
    "joint_head.safetensors",
    "joint_head_config.json",
    "joint_schema_model.py",
    "README.md",
    "LICENSE",
]
SHIFTED = re.compile(r"(input_layernorm|post_attention_layernorm|q_norm|k_norm)\.weight$")
QUANTIZED = re.compile(
    r"^language_model\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|"
    r"in_proj_b|out_proj)|mlp\.(gate|up|down)_proj)\.weight$"
)


def rename(name: str) -> str | None:
    if name.startswith("mtp."):
        return None  # Clef releases have no MTP layer; never serve a stray one
    if name == "lm_head.weight":
        return "language_model.lm_head.weight"
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model.") :]
    if name.startswith("model.visual."):
        return "vision_tower." + name[len("model.visual.") :]
    raise SystemExit(f"unexpected tensor {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=0, choices=[0, 8], help="0: keep bf16 (default); 8: 8-bit decoder")
    ap.add_argument("--backbone-only", action="store_true", help="a plain Qwen3.5 checkpoint (no joint head; tests)")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    if not args.backbone_only and not (src / "joint_head_config.json").exists():
        raise SystemExit(f"{src}: not a Clef release (joint_head_config.json missing)")
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))
    weight_map: dict[str, str] = {}
    n_shift = n_quant = 0
    for i, shard in enumerate(shards):
        tensors = mx.load(str(src / shard))
        dst: dict[str, mx.array] = {}
        for name, value in tensors.items():
            new = rename(name)
            if new is None:
                continue
            if SHIFTED.search(new) or new == "language_model.model.norm.weight":
                dst[new] = value.astype(mx.float32) + 1.0
                n_shift += 1
            elif args.bits and QUANTIZED.match(new):
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
        cfg = json.loads((out / "config.json").read_text())
        (out / "config.json").unlink()  # a hard link to the release's file: write a new one
        q = {"group_size": 64, "bits": args.bits, "mode": "affine"}
        cfg["quantization"] = q
        cfg["quantization_config"] = q
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"done: {len(weight_map)} tensors, {n_shift} norms shifted, {n_quant} matrices quantized -> {out}")


if __name__ == "__main__":
    main()
