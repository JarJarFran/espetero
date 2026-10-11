#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Gemma 4 parity of the native engine against mlx-lm.

  # 1. reference (mlx-lm 0.32): prompts -> greedy generations and teacher-forced rows
  uv run --offline --no-project --with mlx-lm==0.32.0 python tests/ref/gemma4_ref.py ref \\
      --model DIR --cases build/g4_cases.json --out build/g4_ref.json [--gen 64] [--fp32]
  # 2. the engine on the same ids
  build/logits_dump DIR build/g4_cases.json build/g4_ours.json
  # 3. comparison (no GPU)
  python3 tests/ref/gemma4_ref.py compare build/g4_ref.json build/g4_ours.json

`ref` renders ~10 chat prompts with the model's template (two longer than the
sliding window), generates `--gen` greedy tokens with mlx-lm's stream_generate
(bf16 activations, the KV-cached path), and computes the teacher-forced
logits of prompt + generation in one forward pass (rows from the prompt's
last position on: argmax, log-sum-exp and the top 64). `--fp32` runs the
model with fp32 activations instead (an upper bound on the precision a
reference can have for these quantized weights). `compare` reports, per
pass shape of the engine (prefill, tree, decode, batch), the top-1
agreement, the largest log-probability difference over the reference's top
16 and the log-sum-exp difference, and for the generations the number of
equal leading tokens (and the reference's top-2 margin where they part).
"""

import argparse
import inspect
import json
import math
import sys
import textwrap


def prompts():
    long_text = inspect.getsource(textwrap)
    story = (" ".join(["The lighthouse keeper counted the ships that passed each night, writing their names in a "
                       "worn notebook while the wind pushed against the glass."] * 40))
    return [
        ([{"role": "user", "content": "Explain in three sentences why the sky is blue."}], False),
        ([{"role": "user", "content": "Write a Python function that checks whether a number is prime, with a docstring."}], False),
        ([{"role": "user", "content": "What is 17 * 23? Think step by step."}], True),
        ([{"role": "system", "content": "You are a concise assistant that answers in French."},
          {"role": "user", "content": "Describe the water cycle."}], False),
        ([{"role": "user", "content": "Name five countries in Africa and their capitals, as a numbered list."}], False),
        ([{"role": "user", "content": "Give me a JSON object with the fields name, age and city for a fictional person."}], False),
        ([{"role": "user", "content": "Hi! Who are you?"}, {"role": "assistant", "content": "I am Gemma, a helpful assistant."},
          {"role": "user", "content": "Tell me a fun fact about octopuses."}], False),
        ([{"role": "user", "content": long_text[:5200] + "\n\nWhat does the function dedent do? Answer briefly."}], False),
        ([{"role": "user", "content": story + "\n\nWhat did the keeper write down?"}], False),
        ([{"role": "user", "content": "List the first ten prime numbers and then explain the sieve of Eratosthenes."}], False),
    ]


def load(path, fp32):
    import mlx.core as mx
    from mlx_lm import load as lm_load
    model, tok = lm_load(path)
    if fp32:
        model.set_dtype(mx.float32)
    lm = model.language_model if hasattr(model, "language_model") else model
    return model, lm, tok


def rows_of(logits, top):
    import mlx.core as mx
    x = logits.astype(mx.float32)
    lse = mx.logsumexp(x, axis=-1)
    idx = mx.argsort(-x, axis=-1)[:, :top]
    vals = mx.take_along_axis(x, idx, axis=-1)
    mx.eval(lse, idx, vals)
    out = []
    for r in range(x.shape[0]):
        ids = idx[r].tolist()
        out.append({"argmax": ids[0], "lse": float(lse[r]), "top": [[i, float(v)] for i, v in zip(ids, vals[r].tolist())]})
    return out


def cmd_ref(a):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, lm, tok = load(a.model, a.fp32)
    cases, refs = [], []
    if a.reuse:  # the prompts of an earlier cases file (same ids: e.g. an fp32 reference beside a bf16 one)
        todo = [(c["ids"][: c["prompt_len"]], c["ids"]) for c in json.load(open(a.reuse))["cases"]]
    else:
        todo = []
        for msgs, think in prompts():
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=think)
            todo.append((tok.encode(text, add_special_tokens=False), None))
    for ids, given in todo:
        gen, margins = [], []
        for r in stream_generate(model, tok, prompt=ids, max_tokens=a.gen, sampler=make_sampler(temp=0.0)):
            gen.append(int(r.token))
            lp = r.logprobs.astype(mx.float32)
            two = mx.topk(lp, 2)
            mx.eval(two)
            t = sorted(two.tolist(), reverse=True)
            margins.append(t[0] - t[1])
        full = given if given else ids + gen[: a.forced]  # reuse: the same teacher-forced ids
        x = mx.array([full])
        h = lm.model(x)
        h = h[:, len(ids) - 1:, :]
        logits = lm.model.embed_tokens.as_linear(h) if lm.tie_word_embeddings else lm.lm_head(h)
        if lm.final_logit_softcapping is not None:
            c = lm.final_logit_softcapping
            logits = mx.tanh(logits / c) * c
        rows = rows_of(logits[0], 64)
        cases.append({"ids": full, "prompt_len": len(ids), "gen": a.gen})
        refs.append({"rows": rows, "gen": gen, "gen_margin": margins, "prompt_len": len(ids)})
        print(f"prompt {len(ids)} tokens, generated {len(gen)}: {tok.decode(gen)[:80]!r}", file=sys.stderr)
    json.dump({"cases": cases}, open(a.cases, "w"))
    json.dump({"model": a.model, "fp32": a.fp32, "cases": refs}, open(a.out, "w"))


def cmd_llamacpp(a):
    """Greedy generations of a running llama-server (GGUF reference) for the cases' prompts."""
    import urllib.request
    cases = json.load(open(a.cases))["cases"]
    refs = []
    for c in cases:
        P = c["prompt_len"]
        body = {"prompt": c["ids"][:P], "n_predict": c.get("gen", 64), "temperature": 0, "top_k": 1, "n_probs": 2,
                "cache_prompt": False, "return_tokens": True, "post_sampling_probs": False}
        req = urllib.request.Request(a.url + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        gen = r.get("tokens") or []
        margins = []
        for p in r.get("completion_probabilities", []):
            t = sorted((x["logprob"] for x in p.get("top_logprobs", [])), reverse=True)
            margins.append(t[0] - t[1] if len(t) > 1 else float("inf"))
        refs.append({"rows": [], "gen": gen, "gen_margin": margins, "prompt_len": P})
        print(f"prompt {P} tokens, generated {len(gen)}", file=sys.stderr)
    json.dump({"model": a.url, "engine": "llama.cpp", "cases": refs}, open(a.out, "w"))


def logprob_stats(ref, ours):
    """(top-1 equal, max |dlogprob| over the reference's top 16 found in ours, |dlse|)."""
    lr, lo = ref["lse"], ours["lse"]
    ov = {i: v for i, v in ours["top"]}
    worst = 0.0
    for i, v in ref["top"][:16]:
        if i in ov:
            worst = max(worst, abs((v - lr) - (ov[i] - lo)))
        else:
            worst = max(worst, float("inf"))
    return ref["argmax"] == ours["argmax"], worst, abs(lr - lo)


def cmd_compare(a):
    ref = json.load(open(a.ref))
    ours = json.load(open(a.ours))
    summary = {}

    def acc(kind, r, o):
        eq, d, dl = logprob_stats(r, o)
        s = summary.setdefault(kind, {"rows": 0, "top1": 0, "max_dlogp": 0.0, "max_dlse": 0.0, "sum_dlogp": 0.0})
        s["rows"] += 1
        s["top1"] += int(eq)
        s["max_dlogp"] = max(s["max_dlogp"], d)
        s["sum_dlogp"] += d if math.isfinite(d) else 0.0
        s["max_dlse"] = max(s["max_dlse"], dl)

    gens = []
    for ci, (rc, oc) in enumerate(zip(ref["cases"], ours["cases"])):
        rows = rc["rows"]  # positions prompt_len - 1 .. (none from llama.cpp: generations only)
        if rows:
            acc("prefill", rows[0], oc["prefill"])
        for j, o in enumerate(oc.get("tree", [])):
            if j < len(rows):
                acc("tree", rows[j], o)
        for j, o in enumerate(oc.get("decode", [])):
            if j + 1 < len(rows):
                acc("decode", rows[j + 1], o)
        if rows:
            acc("batch", rows[0], ours["batch"][ci])
        for j, o in enumerate(oc.get("verify", [])):  # positions prompt_len ..
            if j + 1 < len(rows):
                acc("verify", rows[j + 1], o)
        for j, o in enumerate(oc.get("split", [])):  # positions prompt_len - 1, prompt_len
            if j < len(rows):
                acc("split", rows[j], o)
        rg, og = rc["gen"], oc.get("gen", [])
        n = 0
        while n < min(len(rg), len(og)) and rg[n] == og[n]:
            n += 1
        chain_same = oc.get("gen_chain") == og[: len(oc.get("gen_chain", []))] if "gen_chain" in oc else None
        part = None
        if n < min(len(rg), len(og)):
            rm = rc["gen_margin"][n] if n < len(rc["gen_margin"]) else None
            part = {"at": n, "ref_margin": None if rm is None else round(rm, 4), "ours_margin": round(oc["gen_margin"][n], 4)}
        gens.append({"case": ci, "prompt": rc["prompt_len"], "equal": n, "ref_len": len(rg), "chain_equal_plain": chain_same,
                     "parted": part})
    for k, s in summary.items():
        s["mean_dlogp"] = round(s["sum_dlogp"] / max(1, s["rows"]), 5)
        del s["sum_dlogp"]
        s["max_dlogp"] = round(s["max_dlogp"], 5)
        s["max_dlse"] = round(s["max_dlse"], 5)
    print(json.dumps({"rows": summary, "generations": gens}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ref")
    r.add_argument("--model", required=True)
    r.add_argument("--cases", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gen", type=int, default=64)
    r.add_argument("--forced", type=int, default=48, help="generated tokens teacher-forced after the prompt")
    r.add_argument("--fp32", action="store_true")
    r.add_argument("--reuse", help="take the prompts of this cases file instead of the built-in ones")
    l = sub.add_parser("llamacpp")
    l.add_argument("--url", required=True)
    l.add_argument("--cases", required=True)
    l.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("ours")
    a = ap.parse_args()
    {"ref": cmd_ref, "llamacpp": cmd_llamacpp, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
