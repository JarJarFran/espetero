# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Reference logits for a Qwen3.5-family checkpoint from MLX (mlx-vlm or mlx-lm),
for parity checks of the native engine on models the JEVfire dylib cannot serve
(e.g. the 27 GB MoE checkpoints). Writes a JSON file with, per prompt, the token
ids and the fp32 logits of the last position (optionally the top-k ids and
values of every position).

  uv run --isolated --no-project --with mlx-vlm python tests/ref/moe_ref.py \
      --model /path/to/mlx-checkpoint --out build/moe_ref.json \
      --prompt "The capital of France is" --prompt "def fib(n):" --topk 16

The native side (`build/parity_test ... --ref build/moe_ref.json`, see
parity_test.mm) runs the same token ids through the engine and compares the
last-position logits (max abs diff after mean-centering; argmax equality).
"""

import argparse
import json
import sys

import mlx.core as mx


def load_model(path):
    # mlx-vlm for vision-language checkpoints (Qwen3.5/3.6 ship a vision tower),
    # mlx-lm as a fallback for text-only conversions.
    try:
        from mlx_vlm import load as vlm_load

        model, processor = vlm_load(path)
        tok = getattr(processor, "tokenizer", processor)
        lm = model.language_model if hasattr(model, "language_model") else model
        return lm, tok, "mlx_vlm"
    except Exception as e:  # noqa: BLE001
        print(f"mlx_vlm load failed ({e}); trying mlx_lm", file=sys.stderr)
        from mlx_lm import load as lm_load

        model, tok = lm_load(path)
        return model, tok, "mlx_lm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--topk", type=int, default=0)
    args = ap.parse_args()
    prompts = args.prompt or ["The capital of France is", "def fibonacci(n):\n    ", "Once upon a time in a small village,"]
    lm, tok, backend = load_model(args.model)
    out = {"model": args.model, "backend": backend, "cases": []}
    for text in prompts:
        ids = tok.encode(text)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        x = mx.array([ids], dtype=mx.int32)
        logits = lm(x)
        if hasattr(logits, "logits"):
            logits = logits.logits
        logits = logits.astype(mx.float32)
        mx.eval(logits)
        last = logits[0, -1, :]
        case = {"text": text, "ids": ids, "last_logits": last.tolist()}
        if args.topk:
            k = args.topk
            top_ids = mx.argsort(-logits[0], axis=-1)[:, :k]
            mx.eval(top_ids)
            case["topk_ids"] = top_ids.tolist()
        out["cases"].append(case)
        print(f"{backend}: {len(ids)} tokens, argmax {int(mx.argmax(last).item())}", file=sys.stderr)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
