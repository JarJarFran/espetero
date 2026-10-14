#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Gemma 4 unified (gemma4_unified, 12B) -> an Espetero-loadable MLX directory.

  python -I tools/gemma4_unified_convert.py SRC OUT [--head head.pt] [--raw-switch]
      [--bits 8] [--group 64]
  python -I tools/gemma4_unified_convert.py MODEL.gguf OUT --mmproj MMPROJ.gguf --hf-dir DIR [--raw-switch]

SRC is a Hugging Face Gemma4UnifiedForConditionalGeneration directory
(model*.safetensors, config.json, tokenizer files), or a llama.cpp gemma4
GGUF (F32 / F16 / BF16 tensors) with its gemma4uv projector (`--mmproj`) and a
directory with the model's config.json and tokenizer files (`--hf-dir`, for
example the GGUF release's own repository).

The output is mlx-vlm 0.7.1's gemma4_unified layout: language_model.model.*
(the decoder matrices quantized to affine `--bits` / `--group`, 8 / 64 by
default; the tied embedding, norms and layer scalars bf16), vision_embedder.*
(patch LayerNorms, the 6912 -> 3840 patch projection, the factorized position
table [mm_posemb_size, 2, mm_embed_dim], pos_norm) and embed_vision.* /
embed_audio.* projections, all bf16. config.json names the global layers'
geometry explicitly (global_head_dim, num_global_key_value_heads, as
transformers' per_layer_config gives it).

`--head head.pt`: a linear head over the final-normed hidden state of the
prompt's last token (a PyTorch state dict with mu, sd, linear.weight,
linear.bias: z = linear((h - mu) / sd)) becomes hidden_head.safetensors (fp32)
and hidden_head.json; Espetero applies it to chat requests that ask for it
("head" in the body). The pickle is never executed: its opcodes are read with
pickletools and only the four tensors' storage records are accepted.

`--raw-switch`: chat_template.jinja renders the message contents concatenated
(an image part as <|image|>, <bos> first unless the text starts with it) when
the request sets chat_template_kwargs {"raw_prompt": true}, and the model's
own template otherwise; for decision protocols that write their own prompt.

Runs on the CPU (mlx), a tensor at a time; peak memory is a shard being
written (--shard-gb) plus one source tensor.
"""

import argparse
import json
import os
import pickletools
import re
import shutil
import struct
import sys
import zipfile

RAW_TEMPLATE = (
    "{%- set c0 = messages[0]['content'] if messages else '' -%}"
    "{%- if c0 is not string -%}{%- set c0 = c0[0]['text'] if c0 and c0[0]['type'] == 'text' else '' -%}{%- endif -%}"
    "{%- if not c0.startswith(bos_token) -%}{{ bos_token }}{%- endif -%}"
    "{%- for m in messages -%}{%- if m['content'] is string -%}{{ m['content'] }}{%- else -%}"
    "{%- for c in m['content'] -%}{%- if c['type'] == 'text' -%}{{ c['text'] }}{%- else -%}<|image|>{%- endif -%}"
    "{%- endfor -%}{%- endif -%}{%- endfor -%}"
)

QUANT = re.compile(r"^language_model\.model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$")


# ---------------------------------------------------------------- PyTorch state dict (no unpickling)
def read_torch_state(path, allowed):
    """{name: float32 ndarray} of a torch.save'd state dict whose values are plain tensors.

    The pickle is interpreted symbolically from pickletools.genops: GLOBAL only
    names collections.OrderedDict, torch._utils._rebuild_tensor_v2 and torch
    storage classes (nothing is imported or called), and each tensor's bytes
    are read from <archive>/data/<key> with numpy.
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

    class Call:
        def __init__(self, fn, args):
            self.fn, self.args = fn, args

    class Pid:
        def __init__(self, t):
            self.t = t

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
            stack.append({} if fn.name == "collections OrderedDict" else Call(fn.name, args))
        elif n == "SETITEMS":
            items = pop_mark()
            d = stack[-1]
            for k in range(0, len(items), 2):
                d[items[k]] = items[k + 1]
        elif n == "SETITEM":
            v = stack.pop()
            k = stack.pop()
            stack[-1][k] = v
        elif n == "BUILD":
            stack.pop()  # an OrderedDict's (empty) state
        elif n == "STOP":
            result = stack.pop()
            break
        else:
            sys.exit(f"{path}: unsupported pickle opcode {n}")
    if not isinstance(result, dict):
        sys.exit(f"{path}: not a state dict")
    out = {}
    for name, v in result.items():
        if name not in allowed:
            sys.exit(f"{path}: unexpected entry {name!r}")
        if not (isinstance(v, Call) and v.fn == "torch._utils _rebuild_tensor_v2"):
            sys.exit(f"{path}: {name} is not a tensor")
        pid, offset, size, stride = v.args[:4]
        tag, storage, key, _location, numel = pid.t
        if tag != "storage" or storage.name not in dtypes:
            sys.exit(f"{path}: {name}: unsupported storage {getattr(storage, 'name', storage)}")
        dt = dtypes[storage.name]
        raw = np.frombuffer(z.read(f"{root}data/{key}"), dtype=dt)
        if raw.size != numel:
            sys.exit(f"{path}: {name}: storage size {raw.size} != {numel}")
        want, s = [], 1
        for d in reversed(size):
            want.insert(0, s)
            s *= d
        if list(stride) != want:
            sys.exit(f"{path}: {name}: non-contiguous strides {stride}")
        count = 1
        for d in size:
            count *= d
        out[name] = raw[offset:offset + count].reshape(size).astype(np.float32)
    missing = set(allowed) - set(out)
    if missing:
        sys.exit(f"{path}: missing {sorted(missing)}")
    return out


# ---------------------------------------------------------------- GGUF (F32 / F16 / BF16 tensors)
class GGUF:
    TYPES = {0: ("float32", 4), 1: ("float16", 2), 30: ("bfloat16", 2)}

    def __init__(self, path):
        self.path = path
        f = open(path, "rb")
        r = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]
        s = lambda: f.read(r("Q")).decode("utf-8", "replace")
        if f.read(4) != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        r("I")
        nt, nkv = r("Q"), r("Q")
        fmts = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

        def val(t):
            if t == 8:
                return s()
            if t == 9:
                et, n = r("I"), r("Q")
                return [val(et) for _ in range(n)]
            return r(fmts[t])

        self.kv = {}
        for _ in range(nkv):
            k = s()
            self.kv[k] = val(r("I"))
        self.tensors = {}
        for _ in range(nt):
            name = s()
            dims = [r("Q") for _ in range(r("I"))]
            typ, off = r("I"), r("Q")
            self.tensors[name] = (dims, typ, off)
        align = self.kv.get("general.alignment", 32)
        self.base = (f.tell() + align - 1) // align * align
        f.close()

    def get(self, name):
        """The tensor as an mx.array of numpy shape reversed(dims) (rows = the last GGUF dim)."""
        import mlx.core as mx
        import numpy as np
        dims, typ, off = self.tensors[name]
        if typ not in self.TYPES:
            sys.exit(f"{self.path}: {name}: GGUF type {typ} (only F32, F16 and BF16 are converted)")
        kind, size = self.TYPES[typ]
        n = 1
        for d in dims:
            n *= d
        raw = np.fromfile(self.path, dtype=np.uint16 if kind == "bfloat16" else kind, count=n, offset=self.base + off)
        shape = list(reversed(dims))
        if kind == "bfloat16":
            return mx.array(raw.reshape(shape)).view(mx.bfloat16)
        return mx.array(raw.reshape(shape))


GGUF_TEXT = [
    (r"^token_embd\.weight$", "language_model.model.embed_tokens.weight"),
    (r"^output_norm\.weight$", "language_model.model.norm.weight"),
    (r"^blk\.(\d+)\.attn_norm\.weight$", "input_layernorm.weight"),
    (r"^blk\.(\d+)\.post_attention_norm\.weight$", "post_attention_layernorm.weight"),
    (r"^blk\.(\d+)\.ffn_norm\.weight$", "pre_feedforward_layernorm.weight"),
    (r"^blk\.(\d+)\.post_ffw_norm\.weight$", "post_feedforward_layernorm.weight"),
    (r"^blk\.(\d+)\.layer_output_scale\.weight$", "layer_scalar"),
    (r"^blk\.(\d+)\.attn_q\.weight$", "self_attn.q_proj.weight"),
    (r"^blk\.(\d+)\.attn_k\.weight$", "self_attn.k_proj.weight"),
    (r"^blk\.(\d+)\.attn_v\.weight$", "self_attn.v_proj.weight"),
    (r"^blk\.(\d+)\.attn_output\.weight$", "self_attn.o_proj.weight"),
    (r"^blk\.(\d+)\.attn_q_norm\.weight$", "self_attn.q_norm.weight"),
    (r"^blk\.(\d+)\.attn_k_norm\.weight$", "self_attn.k_norm.weight"),
    (r"^blk\.(\d+)\.ffn_gate\.weight$", "mlp.gate_proj.weight"),
    (r"^blk\.(\d+)\.ffn_up\.weight$", "mlp.up_proj.weight"),
    (r"^blk\.(\d+)\.ffn_down\.weight$", "mlp.down_proj.weight"),
]

GGUF_MMPROJ = {
    "v.patch_norm.1.weight": "vision_embedder.patch_ln1.weight",
    "v.patch_norm.1.bias": "vision_embedder.patch_ln1.bias",
    "v.patch_embd.weight": "vision_embedder.patch_dense.weight",
    "v.patch_embd.bias": "vision_embedder.patch_dense.bias",
    "v.patch_norm.2.weight": "vision_embedder.patch_ln2.weight",
    "v.patch_norm.2.bias": "vision_embedder.patch_ln2.bias",
    "v.position_embd.weight": "vision_embedder.pos_embedding",
    "v.patch_norm.3.weight": "vision_embedder.pos_norm.weight",
    "v.patch_norm.3.bias": "vision_embedder.pos_norm.bias",
    "mm.input_projection.weight": "embed_vision.embedding_projection.weight",
    "mm.a.input_projection.weight": "embed_audio.embedding_projection.weight",
}


def gguf_tensors(text, mmproj):
    """(MLX name, loader) pairs of a gemma4 GGUF and its gemma4uv projector.

    llama.cpp's gemma4uv projector orders a patch's 3 * 48 * 48 inputs by
    (channel, y, x); transformers (and the processor's patches) by (y, x,
    channel): patch_ln1 and the columns of patch_dense are put back in that
    order. Its position table is [2][n_pos][dim] (transformers [n_pos][2][dim]).
    """
    import mlx.core as mx
    import numpy as np
    if text.kv.get("general.architecture") != "gemma4":
        sys.exit(f"{text.path}: architecture {text.kv.get('general.architecture')!r}, expected gemma4")
    if mmproj.kv.get("clip.vision.projector_type") != "gemma4uv":
        sys.exit(f"{mmproj.path}: projector {mmproj.kv.get('clip.vision.projector_type')!r}, expected gemma4uv")
    out = []
    for g in text.tensors:
        if g == "rope_freqs.weight":
            continue  # the proportional RoPE comes from the config
        for pat, name in GGUF_TEXT:
            m = re.match(pat, g)
            if m:
                full = name if not m.groups() else f"language_model.model.layers.{m.group(1)}.{name}"
                out.append((full, (lambda t=g: text.get(t))))
                break
        else:
            sys.exit(f"{text.path}: unexpected tensor {g}")
    side = int(mmproj.kv.get("clip.vision.patch_size", 16)) * 3
    # (y, x, c) index -> its place in llama.cpp's (c, y, x) order
    perm = mx.array(np.arange(3 * side * side).reshape(3, side, side).transpose(1, 2, 0).reshape(-1))
    for g in mmproj.tensors:
        if g not in GGUF_MMPROJ:
            sys.exit(f"{mmproj.path}: unexpected tensor {g}")
        if g == "v.position_embd.weight":  # GGUF [2][n_pos][dim] -> [n_pos][2][dim]
            out.append((GGUF_MMPROJ[g], (lambda t=g: mx.transpose(mmproj.get(t), (1, 0, 2)))))
        elif g in ("v.patch_norm.1.weight", "v.patch_norm.1.bias"):
            out.append((GGUF_MMPROJ[g], (lambda t=g: mmproj.get(t)[perm])))
        elif g == "v.patch_embd.weight":
            out.append((GGUF_MMPROJ[g], (lambda t=g: mmproj.get(t)[:, perm])))
        else:
            out.append((GGUF_MMPROJ[g], (lambda t=g: mmproj.get(t))))
    return out


def hf_tensors(src):
    """(MLX name, loader) pairs of a Hugging Face gemma4_unified directory (lazy mx.load per shard)."""
    import mlx.core as mx
    idx = f"{src}/model.safetensors.index.json"
    shards = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    out = []
    for sh in shards:
        t = mx.load(f"{src}/{sh}")
        for k in sorted(t):
            if k == "lm_head.weight" or "rotary_emb" in k:
                continue  # tied / derived
            if k.startswith("model.language_model."):
                nk = "language_model.model." + k[len("model.language_model."):]
            elif k.startswith(("model.vision_embedder.", "model.embed_vision.", "model.embed_audio.")):
                nk = k[len("model."):]
            else:
                sys.exit(f"{src}/{sh}: unexpected tensor {k}")
            out.append((nk, (lambda t=t, k=k: t[k])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Hugging Face directory or a gemma4 GGUF file")
    ap.add_argument("out")
    ap.add_argument("--mmproj", help="GGUF input: the gemma4uv projector")
    ap.add_argument("--hf-dir", help="GGUF input: the directory with config.json and the tokenizer files")
    ap.add_argument("--head", help="a hidden-state decision head (PyTorch state dict: mu, sd, linear.weight, linear.bias)")
    ap.add_argument("--raw-switch", action="store_true", help="chat_template_kwargs raw_prompt renders the raw contents")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--shard-gb", type=float, default=5.0)
    a = ap.parse_args()

    import mlx.core as mx
    import numpy as np
    mx.set_default_device(mx.cpu)

    gguf_in = a.src.endswith(".gguf")
    if gguf_in and (not a.mmproj or not a.hf_dir):
        sys.exit("a GGUF input needs --mmproj and --hf-dir")
    meta = (a.hf_dir if gguf_in else a.src).rstrip("/")
    cfg = json.load(open(f"{meta}/config.json"))
    if cfg.get("model_type") != "gemma4_unified":
        sys.exit(f"{meta}/config.json: model_type {cfg.get('model_type')!r}, expected gemma4_unified")
    tc = dict(cfg["text_config"])
    if tc.get("enable_moe_block") or tc.get("hidden_size_per_layer_input") or tc.get("num_kv_shared_layers"):
        sys.exit("only the dense gemma4_unified decoder is converted")
    D = tc["hidden_size"]

    # The global layers' geometry, explicit (transformers 5.17 writes per_layer_config instead).
    plc = tc.pop("per_layer_config", None)
    if plc:
        geo = {(int(v.get("head_dim", tc.get("head_dim"))), int(v.get("num_key_value_heads", tc["num_key_value_heads"])))
               for v in plc.values()}
        types = tc.get("layer_types") or []
        if len(geo) != 1 or any(types and types[int(k)] != "full_attention" for k in plc):
            sys.exit("per_layer_config: only one geometry for the full-attention layers is supported")
        tc["global_head_dim"], tc["num_global_key_value_heads"] = geo.pop()
    tc.setdefault("global_head_dim", 512)

    pairs = gguf_tensors(GGUF(a.src), GGUF(a.mmproj)) if gguf_in else hf_tensors(a.src)
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

    seen = set()
    for k, load in pairs:
        if k in seen:
            sys.exit(f"duplicate tensor {k}")
        seen.add(k)
        w = load()
        if QUANT.match(k):
            q, s, b = mx.quantize(w.astype(mx.bfloat16), group_size=a.group, bits=a.bits)
            base = k[: -len(".weight")]
            put(k, q)
            put(base + ".scales", s.astype(mx.bfloat16))
            put(base + ".biases", b.astype(mx.bfloat16))
        else:
            if k.endswith("layer_scalar") and w.ndim == 0:
                w = w.reshape(1)
            put(k, w.astype(mx.bfloat16))
    flush()
    for need in ("language_model.model.embed_tokens.weight", "vision_embedder.patch_dense.weight",
                 "vision_embedder.pos_embedding", "embed_vision.embedding_projection.weight"):
        if need not in seen:
            sys.exit(f"missing {need}")
    json.dump({"metadata": {"format": "mlx"}, "weight_map": out_map}, open(f"{a.out}/model.safetensors.index.json", "w"),
              indent=1)

    oc = dict(cfg)
    oc["text_config"] = tc
    vc = dict(oc.get("vision_config") or {})
    vc.setdefault("model_patch_size", vc.get("patch_size", 16) * vc.get("pooling_kernel_size", 3))
    oc["vision_config"] = vc
    oc["quantization"] = {"group_size": a.group, "bits": a.bits, "mode": "affine"}
    oc["quantization_config"] = oc["quantization"]
    json.dump(oc, open(f"{a.out}/config.json", "w"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "processor_config.json", "generation_config.json"):
        if os.path.exists(f"{meta}/{f}"):
            shutil.copyfile(f"{meta}/{f}", f"{a.out}/{f}")
    tmpl = f"{meta}/chat_template.jinja"
    if os.path.exists(tmpl):
        if a.raw_switch:
            src = open(tmpl).read()
            open(f"{a.out}/chat_template.jinja", "w").write(
                "{%- if raw_prompt | default(false) -%}" + RAW_TEMPLATE + "{%- else -%}" + src + "{%- endif -%}\n")
        else:
            shutil.copyfile(tmpl, f"{a.out}/chat_template.jinja")

    if a.head:
        t = read_torch_state(a.head, {"mu", "sd", "linear.weight", "linear.bias"})
        W, b = t["linear.weight"], t["linear.bias"]
        mu, sd = t["mu"].reshape(-1), t["sd"].reshape(-1)
        if W.ndim != 2 or W.shape[1] != D or b.shape != (W.shape[0],) or mu.shape != (D,) or sd.shape != (D,):
            sys.exit(f"{a.head}: shapes {W.shape} {b.shape} {mu.shape} {sd.shape} do not fit hidden size {D}")
        if not (np.isfinite(W).all() and np.isfinite(b).all() and np.isfinite(mu).all() and (sd > 0).all()):
            sys.exit(f"{a.head}: non-finite values or a non-positive sd")
        mx.save_safetensors(f"{a.out}/hidden_head.safetensors",
                            {"weight": mx.array(W), "bias": mx.array(b), "mu": mx.array(mu), "sd": mx.array(sd)},
                            metadata={"format": "mlx"})
        json.dump({"type": "linear", "input": "final_norm_last_row", "standardize": True, "slots": int(W.shape[0]),
                   "hidden_size": D}, open(f"{a.out}/hidden_head.json", "w"), indent=1)
        print(f"head: {W.shape[0]} slots over {D} (hidden_head.safetensors)", file=sys.stderr)
    print(f"done: {len(out_map)} tensors in {n_out[0]} shards", file=sys.stderr)


if __name__ == "__main__":
    main()
