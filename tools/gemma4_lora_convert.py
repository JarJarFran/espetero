#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Gemma 4 checkpoint + PEFT LoRA -> an Espetero-loadable MLX directory.

For decision models published as a Gemma 4 base plus adapters (for example
autotrust/GEV-26B-Decide or autotrust/JEV-Gemma4-26B-A4B): the backbone LoRA
is merged into the weights, the decision head becomes the LM head's rows of
its verbalizer tokens (the rest of the LM head is the tied embedding), and the
decoder matrices are quantized to affine `--bits` / `--group` (8 / 64 by
default) while the embedding, the LM head and the vision tower stay bf16.

  python -I tools/gemma4_lora_convert.py SRC OUT [--adapter adapter] [--head head.safetensors]
      [--lm-head-lora adapter_vllm] [--bits 8] [--group 64] [--template bare|keep]

SRC is a Hugging Face Gemma4ForConditionalGeneration directory (model.*
safetensors, config.json, tokenizer files). The LoRA comes from
SRC/adapter (PEFT: lora_A, lora_B, lora_alpha / r) unless --adapter names
another directory. The head: SRC/head.safetensors (proj.weight [slots,
hidden]) with the verbalizer ids of SRC/judge_config.json, else the lm_head
LoRA of --lm-head-lora (vLLM layout: the head as a LoRA on the tied LM head).
The head's bias (|b| ~ 1e-2 or less) and the per-kind temperatures are the
caller's read-out: decision_head.json and calibration*.json are copied.

The output is mlx-vlm's gemma4 layout (language_model.model.*,
language_model.lm_head.weight untied, experts as switch_glu gate / up / down,
vision_tower.*, embed_vision.*); audio towers and multi-token-prediction
heads are dropped. `--template bare` writes the decision prompt's raw
template as chat_template.jinja (the message contents concatenated, an image
part as <|image|>, <bos> first) and keeps the model's own as
chat_template.orig.jinja. Runs on the CPU (mlx), a tensor at a time; peak
memory is a shard being written (--shard-gb) plus the embedding and LM head
(~5 GB for 26B-A4B).
"""

import argparse
import json
import os
import re
import shutil
import sys

BARE_TEMPLATE = (
    "{%- set c0 = messages[0]['content'] if messages else '' -%}"
    "{%- if c0 is not string -%}{%- set c0 = c0[0]['text'] if c0 and c0[0]['type'] == 'text' else '' -%}{%- endif -%}"
    "{%- if not c0.startswith(bos_token) -%}{{ bos_token }}{%- endif -%}"
    "{%- for m in messages -%}{%- if m['content'] is string -%}{{ m['content'] }}{%- else -%}"
    "{%- for c in m['content'] -%}{%- if c['type'] == 'text' -%}{{ c['text'] }}{%- else -%}<|image|>{%- endif -%}"
    "{%- endfor -%}{%- endif -%}{%- endfor -%}"
)

QUANT = re.compile(r"^language_model\.model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj|"
                   r"experts\.switch_glu\.(gate|up|down)_proj|router\.proj)\.weight$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--adapter", help="PEFT LoRA directory (default SRC/adapter)")
    ap.add_argument("--head", help="decision head safetensors (default SRC/head.safetensors when present)")
    ap.add_argument("--lm-head-lora", help="a PEFT directory whose lm_head LoRA carries the head (default SRC/adapter_vllm)")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--template", choices=["bare", "keep"], default="bare")
    ap.add_argument("--shard-gb", type=float, default=5.0)
    a = ap.parse_args()

    import mlx.core as mx
    mx.set_default_device(mx.cpu)

    src = a.src.rstrip("/")
    cfg = json.load(open(f"{src}/config.json"))
    if cfg.get("model_type") != "gemma4":
        sys.exit(f"{src}: model_type {cfg.get('model_type')!r}, expected gemma4")
    tc = cfg["text_config"]
    D = tc["hidden_size"]

    # ---- LoRA
    ad = a.adapter or f"{src}/adapter"
    acfg = json.load(open(f"{ad}/adapter_config.json"))
    scale = float(acfg["lora_alpha"]) / float(acfg["r"])
    if acfg.get("use_dora") or acfg.get("use_rslora"):
        sys.exit("DoRA / rsLoRA adapters are not supported")
    lora = mx.load(f"{ad}/adapter_model.safetensors")
    pairs = {}  # module path (HF, without .weight) -> (A, B)
    for k in lora:
        m = re.match(r"^base_model\.model\.(.+)\.lora_([AB])\.weight$", k)
        if not m:
            sys.exit(f"{ad}: unexpected tensor {k}")
        pairs.setdefault(m.group(1), {})[m.group(2)] = lora[k]
    print(f"LoRA: {len(pairs)} modules, scale {scale} ({ad})", file=sys.stderr)

    # ---- decision head: rows of the LM head
    head_rows = None  # (ids, [S, D] float32)
    hp = a.head or (f"{src}/head.safetensors" if os.path.exists(f"{src}/head.safetensors") else None)
    jc = json.load(open(f"{src}/judge_config.json")) if os.path.exists(f"{src}/judge_config.json") else None
    if hp:
        if not jc or "verbalizer_ids" not in jc:
            sys.exit("--head needs judge_config.json with verbalizer_ids beside the checkpoint")
        h = mx.load(hp)
        W = h["proj.weight"].astype(mx.float32)
        ids = jc["verbalizer_ids"]
        if W.shape != (len(ids), D):
            sys.exit(f"{hp}: proj.weight {W.shape}, expected {(len(ids), D)}")
        head_rows = (ids, W)
        print(f"head: {len(ids)} rows from {hp}", file=sys.stderr)
    lm_lora = pairs.pop("lm_head", None)
    if head_rows is None:
        lp = a.lm_head_lora or f"{src}/adapter_vllm"
        if lm_lora is None and os.path.exists(f"{lp}/adapter_model.safetensors"):
            v = mx.load(f"{lp}/adapter_model.safetensors")
            vc = json.load(open(f"{lp}/adapter_config.json"))
            if "base_model.model.lm_head.lora_A.weight" in v:
                lm_lora = {"A": v["base_model.model.lm_head.lora_A.weight"], "B": v["base_model.model.lm_head.lora_B.weight"],
                           "scale": float(vc["lora_alpha"]) / float(vc["r"])}
        if lm_lora is None:
            sys.exit("no decision head: give --head (with judge_config.json) or an lm_head LoRA")
    elif lm_lora is not None:
        lm_lora = None  # the head itself is exact; its LoRA form is a bf16 approximation of it

    # ---- source tensors
    idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    shards = sorted(set(idx.values()))
    os.makedirs(a.out, exist_ok=True)
    out_map, buf, buf_bytes, n_out = {}, {}, 0, [0]
    limit = int(a.shard_gb * 1e9)

    def flush():
        nonlocal buf, buf_bytes
        if not buf:
            return
        n_out[0] += 1
        name = f"model-{n_out[0]:05d}.safetensors"
        mx.save_safetensors(f"{a.out}/{name}", buf, metadata={"format": "mlx"})
        for k in buf:
            out_map[k] = name
        print(f"  wrote {name} ({buf_bytes / 1e9:.2f} GB, {len(buf)} tensors)", file=sys.stderr)
        buf, buf_bytes = {}, 0

    def put(k, v):
        nonlocal buf_bytes
        mx.eval(v)
        buf[k] = v
        buf_bytes += v.nbytes
        if buf_bytes >= limit:
            flush()

    def emit(k, w):
        """bf16 tensor `w` under MLX name `k`, quantized when it is a decoder matrix."""
        if QUANT.match(k):
            q, s, b = mx.quantize(w.astype(mx.bfloat16), group_size=a.group, bits=a.bits)
            base = k[: -len(".weight")]
            put(k, q)
            put(base + ".scales", s.astype(mx.bfloat16))
            put(base + ".biases", b.astype(mx.bfloat16))
        else:
            put(k, w.astype(mx.bfloat16) if w.dtype in (mx.float32, mx.float16, mx.bfloat16) else w)

    used = set()

    def merged(hf_module, w):
        p = pairs.get(hf_module)
        if p is None:
            return w
        used.add(hf_module)
        delta = (p["B"].astype(mx.float32) @ p["A"].astype(mx.float32)) * scale
        if delta.shape != w.shape:
            sys.exit(f"LoRA {hf_module}: delta {delta.shape} != weight {w.shape}")
        return w.astype(mx.float32) + delta

    embed = None
    dropped = []
    for sh in shards:
        t = mx.load(f"{src}/{sh}")  # lazy: a tensor is read when it is evaluated
        for k in sorted(list(t)):
            v = t.pop(k)  # not kept once written (the shards are larger than memory)
            if k.startswith(("model.audio_tower.", "model.embed_audio.")) or "rotary_emb" in k or ".mtp." in k:
                dropped.append(k)
                continue
            if k.startswith("model.language_model."):
                rest = k[len("model.language_model."):]
                nk = "language_model.model." + rest
            elif k.startswith(("model.vision_tower.", "model.embed_vision.")):
                nk = k[len("model."):]
                if k.endswith(("input_max", "input_min", "output_max", "output_min")):
                    dropped.append(k)  # clip bounds (use_clipped_linears false)
                    continue
            elif k == "lm_head.weight":
                dropped.append(k)  # tied: rebuilt below
                continue
            else:
                sys.exit(f"unexpected tensor {k}")
            hf_mod = k[: -len(".weight")] if k.endswith(".weight") else None
            if nk.endswith(".experts.gate_up_proj"):
                # [E, 2 Ie, D] -> switch_glu gate / up [E, Ie, D] (gate rows first, as mlx-vlm splits them)
                if any(m.startswith(k) for m in pairs):
                    sys.exit("LoRA on the experts is not supported")
                half = v.shape[1] // 2
                base = nk[: -len(".gate_up_proj")]
                emit(base + ".switch_glu.gate_proj.weight", v[:, :half, :])
                emit(base + ".switch_glu.up_proj.weight", v[:, half:, :])
                continue
            if nk.endswith(".experts.down_proj"):
                emit(nk[: -len(".down_proj")] + ".switch_glu.down_proj.weight", v)
                continue
            if nk == "language_model.model.embed_tokens.weight":
                embed = v
            if hf_mod is not None:
                v = merged(hf_mod, v)
            emit(nk, v)
        del t
    missing = sorted(set(pairs) - used)
    if missing:
        sys.exit(f"LoRA modules without a weight: {missing[:5]} ...")
    if embed is None:
        sys.exit("no embedding")

    # ---- LM head: the embedding with the head's rows (untied)
    lm = embed.astype(mx.float32)
    if head_rows is not None:
        ids, W = head_rows
        lm[mx.array(ids)] = W
    else:
        delta = (lm_lora["B"].astype(mx.float32) @ lm_lora["A"].astype(mx.float32)) * lm_lora["scale"]
        lm = lm + delta
    put("language_model.lm_head.weight", lm.astype(mx.bfloat16))
    del lm
    flush()
    json.dump({"metadata": {"format": "mlx"}, "weight_map": out_map}, open(f"{a.out}/model.safetensors.index.json", "w"),
              indent=1)

    # ---- config and side files
    oc = dict(cfg)
    oc.pop("audio_config", None)
    oc["tie_word_embeddings"] = False
    oc["text_config"] = dict(tc, tie_word_embeddings=False)
    oc["quantization"] = {"group_size": a.group, "bits": a.bits, "mode": "affine"}
    oc["quantization_config"] = oc["quantization"]
    json.dump(oc, open(f"{a.out}/config.json", "w"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "processor_config.json", "generation_config.json",
              "judge_config.json", "calibration.json", "calibration_gold.json"):
        if os.path.exists(f"{src}/{f}"):
            shutil.copyfile(f"{src}/{f}", f"{a.out}/{f}")
    for f in (f"{src}/adapter_vllm/decision_head.json", f"{src}/decision_head.json"):
        if os.path.exists(f):
            shutil.copyfile(f, f"{a.out}/decision_head.json")
            break
    else:
        if jc and hp:  # the head's bias and slot layout for the read-out
            hb = mx.load(hp)["proj.bias"].astype(mx.float32).tolist()
            json.dump({"bias": hb, "verbalizer_ids": jc["verbalizer_ids"], "slots": jc.get("slots"),
                       "softcap": jc.get("softcap"), "prefix_token": jc.get("readout", {}).get("prefix_token")},
                      open(f"{a.out}/decision_head.json", "w"), indent=1)
    if os.path.exists(f"{src}/chat_template.jinja"):
        if a.template == "bare":
            shutil.copyfile(f"{src}/chat_template.jinja", f"{a.out}/chat_template.orig.jinja")
            open(f"{a.out}/chat_template.jinja", "w").write(BARE_TEMPLATE + "\n")
        else:
            shutil.copyfile(f"{src}/chat_template.jinja", f"{a.out}/chat_template.jinja")
    print(f"done: {len(out_map)} tensors in {n_out[0]} shards; dropped {len(dropped)} ({', '.join(dropped[:3])}...)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
