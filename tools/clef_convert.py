#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Convert a Cloudflare Clef release (Hugging Face layout: a Qwen3_5ForConditionalGeneration
backbone, or the Qwen3-Omni thinker of Clef-Omni, + joint schema head) into a directory
`espetero serve` loads.

    python tools/clef_convert.py HF_SNAPSHOT_DIR OUT_DIR [--bits 8]

Qwen3.5 backbones (Clef, Clef-Flash). What changes (nothing else is touched):
  * tensor names: model.language_model.* -> language_model.model.*, lm_head.weight ->
    language_model.lm_head.weight, model.visual.* -> vision_tower.* (the MLX names Espetero reads);
  * the zero-centred RMSNorm weights (input/post-attention, q/k norms, final norm) are stored as
    1 + w in float32 (HF Qwen3.5 computes x * (1 + w); Espetero, MLX and llama.cpp store 1 + w;
    float32 avoids the bf16 rounding of the sum). The DeltaNet output norm stays as it is;
  * --bits 8: decoder matrices (attention, DeltaNet and MLP projections) quantized to MLX
    affine 8-bit, group 64 (mx.quantize); default: bf16 kept (Espetero serves it as dense fp16).
    The embedding, the LM head and the vision tower always stay bf16 (the joint head reads
    LM head rows on the host, exactly as the release does).

Qwen3-Omni thinker (Clef-Omni, model_type qwen3_omni_moe): only the thinker's text decoder and
vision encoder are converted (the audio encoder, the talker and code2wav are left out: Espetero
reads images and text). Tensor names follow mlx-vlm's qwen3_omni_moe sanitize:
thinker.model.* -> language_model.model.*, thinker.lm_head.weight -> language_model.lm_head.weight,
thinker.visual.* -> vision_tower.* (merger ln_q / mlp.0 / mlp.2 -> norm / linear_fc1 / linear_fc2,
merger_list -> deepstack_merger_list); each layer's 128 experts are stacked into
mlp.switch_mlp.{gate,up,down}_proj [E, out, in]. The attention and expert matrices are quantized
to 8-bit affine group 64 (the default and the only width: the bf16 thinker does not fit 64 GB
with its experts dense); RMSNorm weights (x * w, no shift here), the router (mlp.gate), the
embedding, the LM head and the vision encoder stay bf16. config.json is rewritten in the flat
MLX form (text_config, vision_config and the media token ids of the thinker at the top level).

The joint head (joint_head.safetensors, joint_head_config.json), the tokenizer, the chat
template and the processor configuration are hard-linked unchanged (copied across file
systems; config.json is rewritten when quantizing). Needs `mlx` (any venv with it); runs on
the CPU.
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
# Qwen3-Omni thinker: the attention projections (experts are quantized when stacked).
OMNI_QUANTIZED = re.compile(r"^language_model\.model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj\.weight$")
OMNI_EXPERT = re.compile(r"^thinker\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
OMNI_TOKEN_KEYS = (
    "image_token_id",
    "video_token_id",
    "audio_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "audio_start_token_id",
    "audio_end_token_id",
    "position_id_per_seconds",
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


def rename_omni(name: str) -> str | None:
    """Thinker tensors -> Espetero / mlx-vlm names; None for what is not converted."""
    if name.startswith(("talker.", "code2wav.", "thinker.audio_tower.")):
        return None
    if name == "thinker.lm_head.weight":
        return "language_model.lm_head.weight"
    if name.startswith("thinker.model."):
        return "language_model.model." + name[len("thinker.model.") :]
    if name.startswith("thinker.visual."):
        new = "vision_tower." + name[len("thinker.visual.") :]
        if ".merger" in new:
            new = new.replace("merger_list.", "deepstack_merger_list.")
            new = new.replace(".ln_q.", ".norm.").replace(".mlp.0.", ".linear_fc1.").replace(".mlp.2.", ".linear_fc2.")
        return new
    raise SystemExit(f"unexpected tensor {name}")


def link_files(src: Path, out: Path, skip: tuple[str, ...] = ()) -> None:
    for f in COPY:
        if f in skip or not (src / f).exists():
            continue
        dst_file = out / f
        if dst_file.exists():
            dst_file.unlink()
        try:  # one physical copy: hard-link the (HF cache) blob, copy across file systems
            os.link(os.path.realpath(src / f), dst_file)
        except OSError:
            shutil.copyfile(src / f, dst_file)


def write_index(out: Path, weight_map: dict[str, str]) -> None:
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, indent=2)
    )


def convert_qwen35(src: Path, out: Path, bits: int) -> None:
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
            elif bits and QUANTIZED.match(new):
                w, scales, biases = mx.quantize(value, group_size=64, bits=bits)
                base = new[: -len(".weight")]
                dst[new], dst[base + ".scales"], dst[base + ".biases"] = w, scales, biases
                n_quant += 1
            else:
                dst[new] = value
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(out / name), dst, metadata={"format": "mlx"})
        weight_map.update({k: name for k in dst})
        print(f"{shard} -> {name}: {len(dst)} tensors", flush=True)
    write_index(out, weight_map)
    link_files(src, out)
    if bits:
        cfg = json.loads((out / "config.json").read_text())
        (out / "config.json").unlink()  # a hard link to the release's file: write a new one
        q = {"group_size": 64, "bits": bits, "mode": "affine"}
        cfg["quantization"] = q
        cfg["quantization_config"] = q
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"done: {len(weight_map)} tensors, {n_shift} norms shifted, {n_quant} matrices quantized -> {out}")


def omni_config(cfg: dict, bits: int) -> dict:
    """The thinker's configuration in the flat form Espetero (and mlx-vlm's loaders) read."""
    th = cfg["thinker_config"]
    vc = dict(th["vision_config"])
    # transformers' Qwen3OmniMoeVisionEncoderConfig default (a 48 x 48 learned position grid).
    vc.setdefault("num_position_embeddings", 2304)
    out = {
        "model_type": "qwen3_omni_moe",
        "architectures": ["Qwen3OmniMoeThinkerForConditionalGeneration"],
        "text_config": dict(th["text_config"]),
        "vision_config": vc,
        "tie_word_embeddings": bool(th["text_config"].get("tie_word_embeddings", False)),
        "converted_parts": ["thinker.model", "thinker.lm_head", "thinker.visual"],
    }
    for k in OMNI_TOKEN_KEYS:
        if k in th:
            out[k] = th[k]
    q = {"group_size": 64, "bits": bits, "mode": "affine"}
    out["quantization"] = q
    out["quantization_config"] = q
    return out


def convert_omni(src: Path, out: Path, bits: int) -> None:
    if bits != 8:
        raise SystemExit("Qwen3-Omni thinker: the experts are converted to 8 bits (--bits 8, the default)")
    mx.set_default_device(mx.cpu)  # quantization and stacking need no GPU
    cfg = json.loads((src / "config.json").read_text())
    tc = cfg["thinker_config"]["text_config"]
    n_layers, n_experts = int(tc["num_hidden_layers"]), int(tc["num_experts"])
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    # Output groups: the embedding and LM head, the vision encoder, then one per decoder layer
    # (a layer's experts are stacked, whatever source shards hold them).
    groups: dict[str, list[str]] = {}
    for name in sorted(index):
        if rename_omni(name) is None:
            continue
        m = re.match(r"^thinker\.model\.layers\.(\d+)\.", name)
        key = f"layer{int(m.group(1)):03d}" if m else ("vision" if name.startswith("thinker.visual.") else "embed")
        groups.setdefault(key, []).append(name)
    order = ["embed", "vision"] + [f"layer{i:03d}" for i in range(n_layers)]
    missing = [k for k in order if k not in groups]
    if missing:
        raise SystemExit(f"{src}: no tensors for {missing} (incomplete download?)")
    # Every tensor converted must be there (a shard still downloading fails here); shards that
    # hold only what is left out (talker, code2wav, audio encoder) are not read.
    needed = sorted({index[n] for k in order for n in groups[k]})
    for shard in needed:
        if not (src / shard).exists():
            raise SystemExit(f"{src}: {shard} is missing (it holds thinker tensors)")
    unread = sorted(set(index.values()) - set(needed))
    if unread:
        print(f"not read (no text or vision tensors): {', '.join(unread)}", flush=True)
    opened: dict[str, dict[str, mx.array]] = {}

    def tensor(name: str) -> mx.array:
        shard = index[name]
        if shard not in opened:
            if len(opened) >= 3:  # lazy arrays: dropping a shard's dict releases nothing read yet
                opened.pop(next(iter(opened)))
            opened[shard] = mx.load(str(src / shard))
        return opened[shard][name]

    def quantize(dst: dict[str, mx.array], new: str, value: mx.array) -> None:
        w, scales, biases = mx.quantize(value, group_size=64, bits=bits)
        base = new[: -len(".weight")]
        dst[new], dst[base + ".scales"], dst[base + ".biases"] = w, scales, biases

    weight_map: dict[str, str] = {}
    n_quant = n_stack = 0
    names: list[str] = []
    pending: dict[str, mx.array] = {}
    pending_bytes = 0

    def flush(final: bool = False) -> None:
        nonlocal pending, pending_bytes
        if pending and (final or pending_bytes >= 4 << 30):
            name = f"part-{len(names) + 1:05d}.safetensors"
            mx.save_safetensors(str(out / name), pending, metadata={"format": "mlx"})
            weight_map.update({k: name for k in pending})
            print(f"{name}: {len(pending)} tensors, {pending_bytes / 1e9:.2f} GB", flush=True)
            names.append(name)
            pending, pending_bytes = {}, 0

    for key in order:
        experts: dict[tuple[int, str], mx.array] = {}
        dst: dict[str, mx.array] = {}
        for name in groups[key]:
            m = OMNI_EXPERT.match(name)
            if m:
                experts[(int(m.group(2)), m.group(3))] = tensor(name)
                continue
            new = rename_omni(name)
            value = tensor(name)
            if OMNI_QUANTIZED.match(new):
                quantize(dst, new, value)
                n_quant += 1
            else:
                dst[new] = value
        if experts:
            li = int(key[len("layer") :])
            for part in ("gate", "up", "down"):
                stack = mx.stack([experts[(e, part)] for e in range(n_experts)])
                quantize(dst, f"language_model.model.layers.{li}.mlp.switch_mlp.{part}_proj.weight", stack)
                n_stack += 1
        mx.eval(list(dst.values()))
        pending.update(dst)
        pending_bytes += sum(v.nbytes for v in dst.values())
        flush()
    flush(final=True)
    # MLX shard names (with the count, known at the end).
    renamed = {n: f"model-{i + 1:05d}-of-{len(names):05d}.safetensors" for i, n in enumerate(names)}
    for a, b in renamed.items():
        os.replace(out / a, out / b)
    write_index(out, {k: renamed[v] for k, v in weight_map.items()})
    link_files(src, out, skip=("config.json",))
    if (out / "config.json").exists():
        (out / "config.json").unlink()
    (out / "config.json").write_text(json.dumps(omni_config(cfg, bits), indent=2))
    print(f"done: {len(weight_map)} tensors, {n_quant} attention matrices and {n_stack} expert stacks quantized -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=None, choices=[0, 8],
                    help="0: keep bf16 (the Qwen3.5 default); 8: 8-bit decoder (the Qwen3-Omni default and only choice)")
    ap.add_argument("--backbone-only", action="store_true", help="a plain Qwen3.5 checkpoint (no joint head; tests)")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    if not args.backbone_only and not (src / "joint_head_config.json").exists():
        raise SystemExit(f"{src}: not a Clef release (joint_head_config.json missing)")
    out.mkdir(parents=True, exist_ok=True)
    model_type = json.loads((src / "config.json").read_text()).get("model_type", "")
    if model_type == "qwen3_omni_moe":
        convert_omni(src, out, 8 if args.bits is None else args.bits)
    else:
        convert_qwen35(src, out, args.bits or 0)


if __name__ == "__main__":
    main()
