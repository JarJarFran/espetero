#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Engine invariance checks: serving paths that must not change the model's
function, measured with teacher-forced scoring rather than by comparing
generated text. Every check scores the same continuation under the same
messages through two paths and reports max |delta logprob| over its tokens:
float summation order moves them by ~1e-4..1e-3, a stale or wrong state (such
as a lost K/V entry: 0.13 nats and more) moves them by tenths.

Start two fresh servers with the same model and flags (B only serves cold
references; it may differ by one flag to test that flag):

  invariance_check.py --url http://127.0.0.1:8080 --ref-url http://127.0.0.1:8081 [--image PATH]

Checks, in order (server state matters, so run them on fresh servers):
  prompt-cache      the same request twice on A (the second restores the
                    prompt checkpoint) vs its first run
  system-ckpt       A: two chats sharing a long system prompt (the second
                    starts from the system checkpoint) vs B: the second alone
  image-ckpt        A: two questions about one image (the second starts from
                    the image checkpoint) vs B: the second alone
  concurrent        A: four requests in flight together vs B: one at a time
  generation        greedy replies of A vs B (reported, first divergence
                    explained by the logprob gap there)

Loopback servers only.
"""

import argparse
import base64
import io
import json
import sys
import threading
import urllib.request

LONG_SYSTEM = (
    "You are a meticulous assistant for a software documentation team. "
    + " ".join(f"Rule {i}: state facts precisely, keep units, and never guess version numbers." for i in range(1, 40))
)
CONTINUATION = (
    "A release must be tagged before it is published: freeze the main branch, run the full test suite, "
    "and verify the build from a clean checkout. Every public function needs a docstring and an example, "
    "and breaking changes must be listed in the changelog with a migration note. Record every step in "
    "the release log with the time, the person and any problem found."
)
PROMPTS = [
    "Explain how a hash map handles collisions, in about 120 words.",
    "List five checks to run before publishing a software release.",
    "Write a Python function that merges two sorted lists, with a docstring.",
    "¿Qué diferencia hay entre el tiempo y el clima? Responde en un párrafo.",
]


def post(url, body, timeout=1800):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def score(url, messages, text=CONTINUATION):
    body = {"messages": messages, "max_tokens": 8, "temperature": 0, "enable_thinking": False,
            "score": {"continuation": text, "top_k": 2}}
    return post(url, body)["timings"]["score"]


def chat(url, messages, max_tokens):
    o = post(url, {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "enable_thinking": False})
    return o["choices"][0]["message"]["content"]


def image_url(path):
    if path:
        from PIL import Image
        im = Image.open(path).convert("RGB")
    else:  # a synthetic scene: shapes and stripes
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (640, 480), (235, 235, 225))
        d = ImageDraw.Draw(im)
        d.rectangle([60, 60, 300, 260], fill=(200, 40, 40))
        d.ellipse([340, 120, 580, 360], fill=(40, 90, 200))
        for y in range(300, 440, 14):
            d.line([60, y, 300 + (y % 50), y], fill=(20, 20, 20), width=5)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--ref-url", required=True)
    ap.add_argument("--image")
    ap.add_argument("--tol", type=float, default=0.02, help="max |delta logprob| accepted as summation order")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--out")
    a = ap.parse_args()
    A, B = a.url.rstrip("/"), a.ref_url.rstrip("/")
    results = []

    def compare(name, sa, sb):
        if sa["tokens"] != sb["tokens"]:
            rec = {"check": name, "ok": False, "note": "tokenization differs"}
        else:
            d = [abs(x - y) for x, y in zip(sa["logprobs"], sb["logprobs"])]
            worst = max(d) if d else 0.0
            first = next((i for i, v in enumerate(d) if v > a.tol), None)
            am = sum(x != y for x, y in zip(sa["argmax"], sb["argmax"]))
            rec = {"check": name, "ok": worst <= a.tol, "tokens": len(d), "max_dlogprob": round(worst, 5),
                   "first_over_tol": first, "argmax_differ": am}
        results.append(rec)
        print(f"{name:22s} {'ok  ' if rec['ok'] else 'FAIL'} " + json.dumps({k: v for k, v in rec.items() if k not in ('check', 'ok')}),
              flush=True)

    # prompt checkpoint restore (A twice)
    m = [{"role": "user", "content": PROMPTS[0]}]
    first = score(A, m)
    again = score(A, m)
    compare("prompt-cache", again, first)
    # system checkpoint: A warm (after a chat with the same system prompt) vs B cold
    m1 = [{"role": "system", "content": LONG_SYSTEM}, {"role": "user", "content": PROMPTS[1]}]
    m2 = [{"role": "system", "content": LONG_SYSTEM}, {"role": "user", "content": PROMPTS[2]}]
    score(A, m1)
    compare("system-ckpt", score(A, m2), score(B, m2))
    # image checkpoint
    img = image_url(a.image)
    q1 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}},
                                       {"type": "text", "text": "Describe the shapes in this picture."}]}]
    q2 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}},
                                       {"type": "text", "text": "What colors appear in this picture?"}]}]
    try:
        score(A, q1)
        compare("image-ckpt", score(A, q2), score(B, q2))
    except urllib.error.HTTPError as e:
        print(f"image-ckpt             skipped (HTTP {e.code}: no vision tower?)")
    # concurrent (A) vs one at a time (B), fresh prompts for both servers
    msgs = [[{"role": "user", "content": p + " (v2)"}] for p in PROMPTS]
    seq = [score(B, mm) for mm in msgs]
    par = [None] * len(msgs)

    def run(k):
        par[k] = score(A, msgs[k])

    th = [threading.Thread(target=run, args=(k,)) for k in range(len(msgs))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    for k in range(len(msgs)):
        compare(f"concurrent/{k}", par[k], seq[k])
    # generation: greedy replies A vs B, explained at the first difference
    for k, p in enumerate(PROMPTS):
        mm = [{"role": "user", "content": p + " (v3)"}]
        ra, rb = chat(A, mm, a.max_tokens), chat(B, mm, a.max_tokens)
        rec = {"check": f"generation/{k}", "ok": True, "identical": ra == rb}
        if ra != rb:
            sa, sb = score(B, mm, ra), score(B, mm, rb)
            j = 0
            while j < min(len(sa["tokens"]), len(sb["tokens"])) and sa["tokens"][j] == sb["tokens"][j]:
                j += 1
            if j < min(len(sa["tokens"]), len(sb["tokens"])):
                gap = abs(sa["logprobs"][j] - sb["logprobs"][j])
                rec.update({"first_difference": j, "gap": round(gap, 4)})
                rec["ok"] = gap <= 0.05
        results.append(rec)
        print(f"{rec['check']:22s} {'ok  ' if rec['ok'] else 'FAIL'} " + json.dumps({k2: v for k2, v in rec.items() if k2 not in ('check', 'ok')}),
              flush=True)
    bad = [r for r in results if not r["ok"]]
    print(f"== {len(results)} checks, {len(bad)} failed")
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
