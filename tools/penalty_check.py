#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Penalized replies with and without speculation. Drafts are
verified against the target's own penalized distribution, so a server
with drafts must reply exactly as one without (`--draft 0 --no-lookup
--chain 0`), up to float near-ties.

  python3 tools/penalty_check.py --base http://127.0.0.1:8080 --out drafts.json
  python3 tools/penalty_check.py --compare plain.json drafts.json
"""
import argparse
import hashlib
import json
import sys
import urllib.request

PROMPTS = [
    "Write a Python function that parses a CSV file and sums the second column. Explain it briefly.",
    "List twenty different fruits, one per line, with a one-sentence description of each.",
    "Explain how a hash map works, then give a small example in C.",
    "Describe a walk through an old city at night, in detail.",
]
# (label, sampling): greedy and sampled requests, each penalty kind.
CASES = [
    ("greedy+presence", {"temperature": 0, "presence_penalty": 1.5}),
    ("greedy+repetition", {"temperature": 0, "repetition_penalty": 1.15}),
    ("T0.7+frequency", {"temperature": 0.7, "top_p": 0.95, "frequency_penalty": 0.8, "seed": 7}),
    ("T1+all", {"temperature": 1.0, "top_k": 20, "presence_penalty": 1.0, "frequency_penalty": 0.3,
                "repetition_penalty": 1.05, "seed": 11}),
]


def run(base, key):
    out = []
    for p in PROMPTS:
        for label, samp in CASES:
            body = {"model": "bench", "max_tokens": 256, "ignore_eos": True, "enable_thinking": False,
                    "messages": [{"role": "user", "content": p}], **samp}
            h = {"content-type": "application/json"}
            if key:
                h["authorization"] = f"Bearer {key}"
            req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers=h)
            with urllib.request.urlopen(req, timeout=900) as r:
                j = json.loads(r.read())
            text = j["choices"][0]["message"].get("content") or ""
            t = j.get("timings", {})
            out.append({"prompt": p[:40], "case": label, "text": text, "sha": hashlib.sha256(text.encode()).hexdigest()[:12],
                        "tokens": j["usage"]["completion_tokens"], "drafts": t.get("drafts") or t.get("draft")})
            print(f"{label:18s} {out[-1]['sha']}  {p[:40]}", flush=True)
    return out


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    same = 0
    for x, y in zip(a, b):
        if x["text"] == y["text"]:
            same += 1
            continue
        i = next((k for k, (u, v) in enumerate(zip(x["text"], y["text"])) if u != v), min(len(x["text"]), len(y["text"])))
        print(f"DIFF {x['case']:18s} {x['prompt']}: first difference at char {i}: "
              f"{x['text'][max(0, i - 30):i + 30]!r} vs {y['text'][max(0, i - 30):i + 30]!r}")
    print(f"{same} of {len(a)} replies identical")
    return same == len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default="")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        sys.exit(0 if compare(*a.compare) else 1)
    json.dump(run(a.base, a.key), open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
