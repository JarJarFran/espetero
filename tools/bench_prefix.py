#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Prefix-cache benchmark: N chat requests that share one long system prompt
and differ in the user turn. Reports, per request, the prompt tokens, the
tokens the server reports as cached, the time to first token and the wall
time. The first request is the cold one; the rest should prefill only the
user turn once the system prefix is checkpointed.

  tools/bench_prefix.py --url http://127.0.0.1:8080 --n 6 [--system-words 800] [--model NAME]
"""

import argparse
import json
import random
import time
import urllib.request

WORDS = ("library catalogue shelf author title edition chapter index page volume series reader loan "
         "return date member card branch archive record genre topic summary review note reference "
         "format language publisher year copy request reservation rule condition priority").split()


def system_prompt(n_words, seed=1):
    rng = random.Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return ("You are a librarian assistant. Apply the following rules strictly and answer in one short "
            "paragraph. Rules and vocabulary: " + body)


def ask(url, model, system, user, max_tokens, api_key=None):
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          **({"authorization": "Bearer " + api_key} if api_key else {})})
    t0 = time.perf_counter()
    first = None
    usage = None
    n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                obj = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    first = first or time.perf_counter()
                    n += 1
    t1 = time.perf_counter()
    u = usage or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", u.get("cached_tokens"))
    return {"prompt_tokens": u.get("prompt_tokens"), "cached_tokens": cached, "completion_tokens": u.get("completion_tokens") or n,
            "ttft_s": (first - t0) if first else None, "wall_s": t1 - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="default")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--system-words", type=int, default=800)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()
    system = system_prompt(args.system_words)
    questions = ["Can a reference book be loaned? Answer briefly.", "What does a reservation do?",
                 "Name one rule for late returns.", "Is a member card needed to read on site?",
                 "When is a copy request accepted?", "What must a catalogue record show?",
                 "Describe how series are shelved.", "What is an edition number?"]
    rows = []
    for i in range(args.n):
        r = ask(args.url, args.model, system, questions[i % len(questions)], args.max_tokens, args.api_key)
        rows.append(r)
        print(f"[{i+1}/{args.n}] prompt={r['prompt_tokens']} cached={r['cached_tokens']} ttft={r['ttft_s'] or 0:.2f}s "
              f"wall={r['wall_s']:.2f}s completion={r['completion_tokens']}", flush=True)
    warm = [r for r in rows[1:] if r["ttft_s"]]
    if warm:
        print(f"cold TTFT {rows[0]['ttft_s']:.2f}s; warm TTFT mean {sum(r['ttft_s'] for r in warm)/len(warm):.2f}s "
              f"(cached tokens {[r['cached_tokens'] for r in warm]})")


if __name__ == "__main__":
    main()
