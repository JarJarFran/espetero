#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""A Kev decision model (jaredpalmer/kev-27b) -> an Espetero-loadable MLX directory.

  python -I tools/kev_convert.py SNAPSHOT OUT [--bits 8] [--group 64] [--vision-from MLX_DIR]

SNAPSHOT is the release: the whole fine-tuned backbone as transformers'
Qwen3_5TextModel (embed_tokens, layers, norm: no vision tower, LM head or
MTP layers), head.pt (the pointer head: two hidden -> head_dim projections q
and k, and the calibration temperature) and interpolation.json. The output
is mlx-vlm 0.7.1's qwen3_5 layout:

  * every backbone tensor -> language_model.model.*;
  * the zero-centred RMSNorm weights (input / post-attention, q / k norms,
    the final norm) stored as 1 + w in float32 (transformers computes
    x * (1 + w) in float32); the DeltaNet output norm unchanged; conv1d
    weights [C, 1, k] -> [C, k, 1];
  * the decoder matrices (attention, DeltaNet and MLP projections) affine
    `--bits` / `--group` (8 / 64); the embedding bf16;
  * language_model.lm_head: a placeholder of zeros (4-bit; the release has
    no LM head and Espetero's loader wants one);
  * pointer_head.safetensors: q_weight, q_bias, k_weight, k_bias in fp32 as
    head.pt holds them; pointer_head.json: their sizes, the temperature, the
    five delimiter tokens of the release's encoding and "lm_head": false (the
    server answers /v1/systemone only).

head.pt is never unpickled: its opcodes are read with pickletools, only a
few globals are accepted (OrderedDict, torch's tensor rebuild and float
storages; nothing is imported or called) and the tensors' bytes come from
the archive's raw storages.

`--vision-from MLX_DIR`: graft the vision tower of a base model's MLX
directory (vision_tower.*, e.g. mlx-community/Qwen3.8-27B-8bit, bf16) with its
vision_config, image token ids and processor configuration, so the model
takes images. The tower is the base model's, unchanged: Kev was fine-tuned
on text only and its backbone never saw image tokens, so its answers about
images are an experiment, not the release's evaluated setup
(pointer_head.json records the graft).

The tokenizer, chat template, README and interpolation.json are copied
unchanged; config.json is the qwen3_5 layout with Kev's text config and the
quantization. Runs on the CPU (mlx), a shard at a time.
"""

import argparse
import hashlib
import json
import os
import pickletools
import re
import shutil
import sys
import zipfile

COPY = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md", "interpolation.json"]
VISION_COPY = ["preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"]
SHIFTED = re.compile(r"(input_layernorm|post_attention_layernorm|q_norm|k_norm)\.weight$")
QUANTIZED = re.compile(
    r"^language_model\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|"
    r"in_proj_b|out_proj)|mlp\.(gate|up|down)_proj)\.weight$"
)
HEAD_BITS = 4  # the placeholder LM head
# kev/model.py SPECIAL: the state, question, option, option-end and decide delimiters.
TOKENS = {"state": "<|fim_prefix|>", "question": "<|fim_middle|>", "option": "<|box_start|>",
          "option_end": "<|box_end|>", "decide": "<|fim_suffix|>"}
HEAD_TENSORS = {"q.weight": "q_weight", "q.bias": "q_bias", "k.weight": "k_weight", "k.bias": "k_bias"}


# ---------------------------------------------------------------- head.pt (no unpickling)
def read_torch_pickle(path):
    """The object torch.save wrote to `path` (dicts, lists, tuples, strings, numbers; tensors as float32 ndarrays).

    The pickle is interpreted symbolically from pickletools.genops, as
    tools/gemma4_unified_convert.py reads a state dict: GLOBAL only names
    collections.OrderedDict, torch._utils._rebuild_tensor_v2 and torch
    storage classes (nothing is imported or called), and each tensor's
    bytes are read from <archive>/data/<key> with numpy.
    """
    import numpy as np

    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
    if len(pkl) != 1:
        sys.exit(f"{path}: not a torch zip archive (data.pkl)")
    root = pkl[0][: -len("data.pkl")]
    order = z.read(root + "byteorder").decode().strip() if root + "byteorder" in z.namelist() else "little"
    if order != "little":
        sys.exit(f"{path}: byte order {order}")
    dtypes = {"torch FloatStorage": np.float32, "torch HalfStorage": np.float16, "torch DoubleStorage": np.float64}
    ok_globals = {"collections OrderedDict", "torch._utils _rebuild_tensor_v2"} | set(dtypes)

    class Global:
        def __init__(self, name):
            self.name = name

    class Pid:
        def __init__(self, t):
            self.t = t

    def tensor(args):
        pid, offset, size, stride = args[:4]
        tag, storage, key, _location, numel = pid.t
        if tag != "storage" or storage.name not in dtypes:
            sys.exit(f"{path}: unsupported storage {getattr(storage, 'name', storage)}")
        raw = np.frombuffer(z.read(f"{root}data/{key}"), dtype=dtypes[storage.name])
        if raw.size != numel:
            sys.exit(f"{path}: storage {key}: {raw.size} elements, the pickle says {numel}")
        want, s = [], 1
        for d in reversed(size):
            want.insert(0, s)
            s *= d
        if list(stride) != want:
            sys.exit(f"{path}: storage {key}: non-contiguous strides {stride}")
        if offset + s > raw.size:
            sys.exit(f"{path}: storage {key}: out of range")
        return raw[offset:offset + s].reshape(size).astype(np.float32)

    MARK = object()
    stack, memo, result = [], {}, None

    def pop_mark():
        i = len(stack) - 1
        while stack[i] is not MARK:
            i -= 1
        items = stack[i + 1:]
        del stack[i:]
        return items

    for op, arg, _ in pickletools.genops(z.read(pkl[0])):
        n = op.name
        if n in ("PROTO", "FRAME"):
            continue
        if n == "GLOBAL":
            if arg not in ok_globals:
                sys.exit(f"{path}: refusing pickle global {arg!r}")
            stack.append(Global(arg))
        elif n == "MARK":
            stack.append(MARK)
        elif n in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE", "BININT", "BININT1", "BININT2", "LONG1", "BINFLOAT"):
            stack.append(arg)
        elif n == "NEWFALSE":
            stack.append(False)
        elif n == "NEWTRUE":
            stack.append(True)
        elif n == "NONE":
            stack.append(None)
        elif n == "EMPTY_TUPLE":
            stack.append(())
        elif n == "EMPTY_DICT":
            stack.append({})
        elif n == "EMPTY_LIST":
            stack.append([])
        elif n == "TUPLE":
            stack.append(tuple(pop_mark()))
        elif n in ("TUPLE1", "TUPLE2", "TUPLE3"):
            k = int(n[-1])
            items = tuple(stack[-k:])
            del stack[-k:]
            stack.append(items)
        elif n in ("BINPUT", "LONG_BINPUT"):
            memo[arg] = stack[-1]
        elif n == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif n in ("BINGET", "LONG_BINGET"):
            stack.append(memo[arg])
        elif n == "BINPERSID":
            stack.append(Pid(stack.pop()))
        elif n == "REDUCE":
            args = stack.pop()
            fn = stack.pop()
            if not isinstance(fn, Global):
                sys.exit(f"{path}: REDUCE of a non-global")
            if fn.name == "collections OrderedDict":
                stack.append({})
            elif fn.name == "torch._utils _rebuild_tensor_v2":
                stack.append(tensor(args))
            else:
                sys.exit(f"{path}: REDUCE of {fn.name}")
        elif n == "SETITEMS":
            items = pop_mark()
            d = stack[-1]
            for k in range(0, len(items), 2):
                d[items[k]] = items[k + 1]
        elif n == "SETITEM":
            v = stack.pop()
            k = stack.pop()
            stack[-1][k] = v
        elif n == "APPEND":
            v = stack.pop()
            stack[-1].append(v)
        elif n == "APPENDS":
            items = pop_mark()
            stack[-1].extend(items)
        elif n == "BUILD":
            stack.pop()  # an OrderedDict's attribute state (state_dict _metadata)
        elif n == "STOP":
            result = stack.pop()
            break
        else:
            sys.exit(f"{path}: unsupported pickle opcode {n}")
    return result


def read_head(path, hidden):
    """(tensors {q_weight, q_bias, k_weight, k_bias}, temperature, head_dim) of a Kev head.pt (kev/checkpoint.py Meta)."""
    import numpy as np

    meta = read_torch_pickle(path)
    if not isinstance(meta, dict) or not isinstance(meta.get("head"), dict):
        sys.exit(f"{path}: not a Kev head.pt (a dict with a head state dict)")
    if meta.get("option_isolation", False):
        sys.exit(f"{path}: option_isolation checkpoints need the packed mask (not a hybrid Qwen3.5 backbone)")
    if meta.get("special_embeddings", False):
        sys.exit(f"{path}: trained delimiter embeddings are not part of this layout")
    if meta.get("weights") != "full":
        sys.exit(f"{path}: weights {meta.get('weights')!r}: only full-weight checkpoints (the whole backbone saved)")
    head = meta["head"]
    if set(head) != set(HEAD_TENSORS):
        sys.exit(f"{path}: head tensors {sorted(head)}, expected {sorted(HEAD_TENSORS)}")
    dp = int(meta.get("head_dim", 256))
    want = {"q.weight": (dp, hidden), "q.bias": (dp,), "k.weight": (dp, hidden), "k.bias": (dp,)}
    out = {}
    for name, new in HEAD_TENSORS.items():
        v = head[name]
        if not isinstance(v, np.ndarray) or tuple(v.shape) != want[name]:
            sys.exit(f"{path}: {name} has shape {getattr(v, 'shape', None)}, expected {want[name]}")
        out[new] = v
    t = float(meta.get("temperature", 1.0))
    if not (t > 0 and t < float("inf")):
        sys.exit(f"{path}: temperature {t!r}")
    return out, t, dp


def rename(name: str) -> str:
    """Qwen3_5TextModel tensor names (embed_tokens, layers.*, norm) -> mlx-vlm's."""
    if name.startswith(("embed_tokens.", "layers.", "norm.")):
        return "language_model.model." + name
    sys.exit(f"unexpected tensor {name} (a Qwen3_5TextModel checkpoint holds embed_tokens, layers and norm only)")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
    ap.add_argument("--group", type=int, default=64, choices=[32, 64, 128])
    ap.add_argument("--vision-from", default="", help="an MLX Qwen3.5 directory whose vision tower is grafted")
    a = ap.parse_args()

    import mlx.core as mx
    mx.set_default_device(mx.cpu)

    src = a.src.rstrip("/")
    tc = json.load(open(f"{src}/config.json"))
    if tc.get("model_type") != "qwen3_5_text" or "text_config" in tc:
        sys.exit(f"{src}: not a Qwen3_5TextModel checkpoint (model_type {tc.get('model_type')!r})")
    D, V = int(tc["hidden_size"]), int(tc["vocab_size"])
    tensors_head, temperature, dp = read_head(f"{src}/head.pt", D)

    vcfg, vdir = None, a.vision_from.rstrip("/")
    if vdir:
        vcfg = json.load(open(f"{vdir}/config.json"))
        vtc = vcfg.get("text_config") or {}
        if vcfg.get("model_type") != "qwen3_5" or not isinstance(vcfg.get("vision_config"), dict):
            sys.exit(f"{vdir}: not a qwen3_5 MLX directory with a vision_config")
        if int(vcfg["vision_config"].get("out_hidden_size", 0)) != D or int(vtc.get("hidden_size", D)) != D:
            sys.exit(f"{vdir}: its vision tower projects to {vcfg['vision_config'].get('out_hidden_size')}, the backbone is {D} wide")
        vindex = json.load(open(f"{vdir}/model.safetensors.index.json"))["weight_map"]
        vshards = sorted({f for k, f in vindex.items() if k.startswith("vision_tower.")})
        if not vshards:
            sys.exit(f"{vdir}: no vision_tower.* tensors")

    index = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    shards = sorted(set(index.values()))
    for sh in shards:
        if not os.path.exists(f"{src}/{sh}"):
            sys.exit(f"{src}/{sh} is missing (incomplete download?)")
    os.makedirs(a.out, exist_ok=True)
    n_out = len(shards) + 1 + (1 if vdir else 0)  # + the placeholder LM head (+ the vision tower)
    weight_map: dict[str, str] = {}
    n_shift = n_quant = 0
    for i, sh in enumerate(shards):
        tensors = mx.load(f"{src}/{sh}")
        dst = {}
        for name, value in tensors.items():
            new = rename(name)
            if SHIFTED.search(new) or new == "language_model.model.norm.weight":
                dst[new] = value.astype(mx.float32) + 1.0
                n_shift += 1
            elif new.endswith("conv1d.weight") and value.ndim == 3 and value.shape[-1] != 1:
                dst[new] = mx.moveaxis(value, 2, 1)
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
    # Zero codes, scales and biases: every logit is 0 (affine code x scale + bias).
    name = f"model-{len(shards) + 1:05d}-of-{n_out:05d}.safetensors"
    g = 64
    head = {"language_model.lm_head.weight": mx.zeros((V, D * HEAD_BITS // 32), dtype=mx.uint32),
            "language_model.lm_head.scales": mx.zeros((V, D // g), dtype=mx.bfloat16),
            "language_model.lm_head.biases": mx.zeros((V, D // g), dtype=mx.bfloat16)}
    mx.save_safetensors(f"{a.out}/{name}", head, metadata={"format": "mlx"})
    weight_map.update({k: name for k in head})
    quant["language_model.lm_head"] = {"group_size": g, "bits": HEAD_BITS}
    n_vision = 0
    if vdir:
        vision = {}
        for sh in vshards:
            for k, v in mx.load(f"{vdir}/{sh}").items():
                if k.startswith("vision_tower."):
                    vision[k] = v
        if any(k.endswith((".scales", ".biases")) and k[: k.rfind(".")] + ".weight" in vision for k in vision):
            sys.exit(f"{vdir}: a quantized vision tower (Espetero reads it in bf16 or fp16)")
        name = f"model-{n_out:05d}-of-{n_out:05d}.safetensors"
        mx.eval(list(vision.values()))
        mx.save_safetensors(f"{a.out}/{name}", vision, metadata={"format": "mlx"})
        weight_map.update({k: name for k in vision})
        n_vision = len(vision)
        print(f"vision tower from {vdir}: {n_vision} tensors -> {name}", file=sys.stderr, flush=True)
    json.dump({"metadata": {"format": "mlx"}, "weight_map": dict(sorted(weight_map.items()))},
              open(f"{a.out}/model.safetensors.index.json", "w"), indent=1)

    oc = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"}
    if vcfg:
        for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id", "vision_config"):
            if k in vcfg:
                oc[k] = vcfg[k]
    oc["language_model_only"] = not vcfg
    oc["text_config"] = {k: v for k, v in tc.items() if k not in ("architectures", "transformers_version")}
    oc["tie_word_embeddings"] = False
    oc["eos_token_id"] = tc.get("eos_token_id")
    oc["quantization"] = quant
    oc["quantization_config"] = quant
    json.dump(oc, open(f"{a.out}/config.json", "w"), indent=1)
    for f in COPY:
        if os.path.exists(f"{src}/{f}"):
            shutil.copyfile(f"{src}/{f}", f"{a.out}/{f}")
    for f in VISION_COPY if vdir else []:
        if os.path.exists(f"{vdir}/{f}"):
            shutil.copyfile(f"{vdir}/{f}", f"{a.out}/{f}")

    mx.save_safetensors(f"{a.out}/pointer_head.safetensors", {k: mx.array(v) for k, v in tensors_head.items()},
                        metadata={"format": "mlx"})
    ph = {"type": "kev_pointer", "hidden_size": D, "head_dim": dp, "temperature": temperature, "tokens": TOKENS,
          "max_state_tokens": 65536, "max_question_tokens": 8192, "lm_head": False,
          "head_sha256": sha256(f"{src}/head.pt")}
    if vdir:
        ph["vision"] = {"grafted_from": os.path.basename(vdir.rstrip("/")) if "/snapshots/" not in vdir else
                        vdir.split("/snapshots/")[0].rsplit("/", 1)[-1].removeprefix("models--").replace("--", "/"),
                        "note": "the base model's vision tower, unchanged: the release was fine-tuned on text only"}
    json.dump(ph, open(f"{a.out}/pointer_head.json", "w"), indent=1)
    print(f"done: {len(weight_map)} tensors, {n_shift} norms shifted, {n_quant} matrices quantized, {n_vision} vision "
          f"tensors; pointer head {dp} over {D}, temperature {temperature} -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
