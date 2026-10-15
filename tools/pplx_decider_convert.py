#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""A native decision checkpoint (perplexity-ai/pplx-decider-v1.1-27b) -> an Espetero-loadable MLX directory.

  python -I tools/pplx_decider_convert.py SNAPSHOT OUT [--bits 8] [--group 64]

SNAPSHOT is the release's "native decision-checkpoint layout": a transformers
Qwen3_5Model backbone (language_model.* and visual.* tensors, no LM head),
readout.safetensors (the decision head, [255, hidden], no bias) and
decision_config.json (answer codes, calibration temperature, attention mode,
pooling). The output is mlx-vlm 0.7.1's qwen3_5 layout:

  * language_model.* -> language_model.model.*, visual.* -> vision_tower.*;
  * the zero-centred RMSNorm weights (input / post-attention, q / k norms,
    the final norm) stored as 1 + w in float32 (transformers computes
    x * (1 + w) in float32); the DeltaNet output norm unchanged;
  * conv1d weights [C, 1, k] -> [C, k, 1] and the Conv3d patch embedding
    [O, C, T, P, P] -> [O, T, P, P, C] (MLX order: mlx-vlm then loads the
    directory as it is, Espetero reads either);
  * the decoder matrices (attention, DeltaNet and MLP projections) affine
    `--bits` / `--group` (8 / 64); the embedding and the vision tower bf16;
  * language_model.lm_head: a placeholder of zeros (4-bit; the release has no
    LM head and Espetero's loader wants one). hidden_head.json says so
    ("lm_head": false) and the server then answers only head requests;
  * hidden_head.safetensors: the readout as an fp32 linear head without bias
    over the last row's final-normed state (pooling "last");
    hidden_head.json: its slots, the release's attention_mode and
    temperature (Espetero runs the prompt with bidirectional full attention
    and reports softmax(logits / temperature)), and the answer codes of the
    slots (A, B, ..., Z, AA, ...).

decision_config.json, the tokenizer, chat template, processor configuration,
README, LICENSE and NOTICE are copied unchanged; config.json gets the
quantization. Runs on the CPU (mlx), a shard at a time.
"""

import argparse
import json
import os
import re
import shutil
import sys

COPY = ["decision_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "processor_config.json", "preprocessor_config.json", "README.md", "LICENSE", "NOTICE"]
SHIFTED = re.compile(r"(input_layernorm|post_attention_layernorm|q_norm|k_norm)\.weight$")
QUANTIZED = re.compile(
    r"^language_model\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|"
    r"in_proj_b|out_proj)|mlp\.(gate|up|down)_proj)\.weight$"
)
HEAD_BITS = 4  # the placeholder LM head


def rename(name: str) -> str | None:
    """Backbone tensor names (Qwen3_5Model, or with a ForConditionalGeneration's model. prefix) -> mlx-vlm's."""
    if name.startswith("mtp."):
        return None  # no drafting head is served for a decision model
    if name.startswith("model."):
        name = name[len("model."):]
    if name == "lm_head.weight":
        return "language_model.lm_head.weight"
    if name.startswith("language_model."):
        return "language_model.model." + name[len("language_model."):]
    if name.startswith("visual."):
        return "vision_tower." + name[len("visual."):]
    sys.exit(f"unexpected tensor {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
    ap.add_argument("--group", type=int, default=64, choices=[32, 64, 128])
    a = ap.parse_args()

    import mlx.core as mx
    mx.set_default_device(mx.cpu)

    src = a.src.rstrip("/")
    cfg = json.load(open(f"{src}/config.json"))
    dc = json.load(open(f"{src}/decision_config.json"))
    tc = cfg.get("text_config") or {}
    vc = cfg.get("vision_config") or {}
    if cfg.get("model_type") != "qwen3_5" or tc.get("model_type", "qwen3_5_text") != "qwen3_5_text":
        sys.exit(f"{src}: not a qwen3_5 checkpoint (model_type {cfg.get('model_type')!r})")
    if dc.get("format_version") != 1:
        sys.exit(f"{src}/decision_config.json: unsupported format_version {dc.get('format_version')!r}")
    if dc.get("pooling", "last") != "last":
        sys.exit(f"{src}: pooling {dc.get('pooling')!r} (only the last token's state is read out)")
    mode = dc.get("attention_mode", "causal")
    if mode not in ("causal", "noncausal_full_attention"):
        sys.exit(f"{src}: unknown attention_mode {mode!r}")
    temperature = float(dc.get("temperature", 1.0))
    if not (temperature > 0 and temperature < float("inf")):
        sys.exit(f"{src}: temperature {temperature!r}")
    D, V = int(tc["hidden_size"]), int(tc["vocab_size"])
    codes = list(dc.get("codes") or [])

    readout = mx.load(f"{src}/readout.safetensors")
    if set(readout) != {"weight"} or readout["weight"].ndim != 2 or readout["weight"].shape[1] != D:
        sys.exit(f"{src}/readout.safetensors: expected one weight [slots, {D}], got "
                 f"{ {k: v.shape for k, v in readout.items()} }")
    W = readout["weight"].astype(mx.float32)
    S = int(W.shape[0])
    if codes and len(codes) != S:
        sys.exit(f"{src}: {len(codes)} answer codes for {S} readout rows")

    index = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    shards = sorted(set(index.values()))
    for sh in shards:
        if not os.path.exists(f"{src}/{sh}"):
            sys.exit(f"{src}/{sh} is missing (incomplete download?)")
    os.makedirs(a.out, exist_ok=True)
    has_head = any(rename(n) == "language_model.lm_head.weight" for n in index)
    n_out = len(shards) + (0 if has_head else 1)  # + the placeholder LM head
    weight_map: dict[str, str] = {}
    n_shift = n_quant = 0
    in_ch = int(vc.get("in_channels", 3))
    for i, sh in enumerate(shards):
        tensors = mx.load(f"{src}/{sh}")
        dst = {}
        for name, value in tensors.items():
            new = rename(name)
            if new is None:
                continue
            if SHIFTED.search(new) or new == "language_model.model.norm.weight":
                dst[new] = value.astype(mx.float32) + 1.0
                n_shift += 1
            elif new.endswith("conv1d.weight") and value.ndim == 3 and value.shape[-1] != 1:
                dst[new] = mx.moveaxis(value, 2, 1)
            elif new.endswith("patch_embed.proj.weight") and value.ndim == 5 and value.shape[1] == in_ch:
                dst[new] = value.transpose(0, 2, 3, 4, 1)
            elif QUANTIZED.match(new):
                w, scales, biases = mx.quantize(value, group_size=a.group, bits=a.bits)
                base = new[: -len(".weight")]
                dst[new], dst[base + ".scales"], dst[base + ".biases"] = w, scales, biases
                n_quant += 1
            else:
                dst[new] = value
        name = f"model-{i + 1:05d}-of-{n_out:05d}.safetensors"
        mx.eval(list(dst.values()))
        mx.save_safetensors(f"{a.out}/{name}", dst, metadata={"format": "mlx"})
        weight_map.update({k: name for k in dst})
        print(f"{sh} -> {name}: {len(dst)} tensors", file=sys.stderr, flush=True)
        del tensors, dst
    quant = {"group_size": a.group, "bits": a.bits, "mode": "affine"}
    if not has_head:
        # Zero codes, scales and biases: every logit is 0 (affine code x scale + bias).
        name = f"model-{n_out:05d}-of-{n_out:05d}.safetensors"
        g = 64
        head = {"language_model.lm_head.weight": mx.zeros((V, D * HEAD_BITS // 32), dtype=mx.uint32),
                "language_model.lm_head.scales": mx.zeros((V, D // g), dtype=mx.bfloat16),
                "language_model.lm_head.biases": mx.zeros((V, D // g), dtype=mx.bfloat16)}
        mx.save_safetensors(f"{a.out}/{name}", head, metadata={"format": "mlx"})
        weight_map.update({k: name for k in head})
        quant["language_model.lm_head"] = {"group_size": g, "bits": HEAD_BITS}
    json.dump({"metadata": {"format": "mlx"}, "weight_map": dict(sorted(weight_map.items()))},
              open(f"{a.out}/model.safetensors.index.json", "w"), indent=1)

    oc = dict(cfg)
    oc["quantization"] = quant
    oc["quantization_config"] = quant
    json.dump(oc, open(f"{a.out}/config.json", "w"), indent=1)
    for f in COPY:
        if os.path.exists(f"{src}/{f}"):
            shutil.copyfile(f"{src}/{f}", f"{a.out}/{f}")

    mx.save_safetensors(f"{a.out}/hidden_head.safetensors", {"weight": W, "bias": mx.zeros((S,), dtype=mx.float32)},
                        metadata={"format": "mlx"})
    hh = {"type": "linear", "input": "final_norm_last_row", "standardize": False, "slots": S, "hidden_size": D,
          "attention_mode": mode, "temperature": temperature, "lm_head": has_head}
    if codes:
        hh["codes"] = codes
    json.dump(hh, open(f"{a.out}/hidden_head.json", "w"), indent=1)
    print(f"done: {len(weight_map)} tensors, {n_shift} norms shifted, {n_quant} matrices quantized; head {S} slots over "
          f"{D}, attention {mode}, temperature {temperature} -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
