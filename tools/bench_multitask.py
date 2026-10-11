#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Several classification tasks over the same image and prompt: one request
with N fields against N separate requests, sequential and concurrent, through
POST /v1/decisions (label scoring, no decoding) and through chat completions
(short generated answers).

  python3 tools/bench_multitask.py --url http://127.0.0.1:8080 --images a.jpg b.jpg --out /tmp/multitask.jsonl

Cold runs use a fresh cache_salt (decisions) or a fresh system-prompt nonce
(chat), so the image prefix is prefilled again; warm runs repeat the previous
request. Answers are compared across modes (the per-field prompt is the same
in every decision mode, so the labels should agree).
"""

import argparse
import base64
import io
import json
import statistics
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FIELDS = {
    "indoor": {"type": "boolean", "description": "The photo was taken indoors."},
    "people": {"type": "boolean", "description": "At least one person is visible."},
    "text_visible": {"type": "boolean", "description": "Readable text (signs, labels, documents) is visible."},
    "image_kind": {"type": "enum", "description": "What kind of image this is.",
                   "choices": ["photograph", "screenshot", "scanned document", "drawing or diagram"]},
    "lighting": {"type": "enum", "description": "The dominant lighting.",
                 "choices": ["daylight", "artificial light", "dark or night"]},
    "text": {"type": "boolean", "description": "Visible text, a sign or a label in the image."},
}
QUESTIONS = {
    "indoor": "Was this photo taken indoors? Answer yes or no.",
    "people": "Is at least one person visible? Answer yes or no.",
    "text_visible": "Is readable text visible? Answer yes or no.",
    "image_kind": "Is this a photograph, a screenshot, a scanned document or a drawing? Answer with one of those.",
    "lighting": "Is the dominant lighting daylight, artificial light or dark? Answer with one of those.",
    "text": "Is there visible text, a sign or a label in the image? Answer yes or no.",
}
CONTEXT = "A photo. Classify it."


def data_url(path, max_side=1280):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def post(url, path, body, timeout=600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out, time.perf_counter() - t0


def decide(url, img, fields, salt):
    body = {"context": CONTEXT, "fields": {k: FIELDS[k] for k in fields}, "images": [img], "cache_salt": salt}
    out, dt = post(url, "/v1/decisions", body)
    vals = out.get("decisions") or out.get("values") or out
    return {k: (vals.get(k) if isinstance(vals, dict) else None) for k in fields}, dt, out


def chat(url, img, field, nonce):
    body = {"model": "m", "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": f"You classify photos. Request {nonce}."},
                         {"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}},
                                                      {"type": "text", "text": QUESTIONS[field]}]}]}
    out, dt = post(url, "/v1/chat/completions", body)
    return (out["choices"][0]["message"].get("content") or "").strip(), dt


def run_modes(url, img, names):
    rec = {}
    n = len(names)
    salt = uuid.uuid4().hex
    t0 = time.perf_counter()
    rec["dec_one_cold_vals"], rec["dec_one_cold_s"], raw = decide(url, img, names, salt)
    rec["dec_one_warm_vals"], rec["dec_one_warm_s"], _ = decide(url, img, names, salt)
    rec["raw_keys"] = sorted(raw.keys()) if isinstance(raw, dict) else None
    # N single-field requests, one after another (the first one prefills the image).
    salt = uuid.uuid4().hex
    t0 = time.perf_counter()
    vals = {}
    for k in names:
        v, _, _ = decide(url, img, [k], salt)
        vals.update(v)
    rec["dec_split_seq_s"], rec["dec_split_seq_vals"] = time.perf_counter() - t0, vals
    # N single-field requests at once.
    salt = uuid.uuid4().hex
    t0 = time.perf_counter()
    with ThreadPoolExecutor(n) as ex:
        res = list(ex.map(lambda k: decide(url, img, [k], salt), names))
    rec["dec_split_par_s"] = time.perf_counter() - t0
    rec["dec_split_par_vals"] = {k: v for r in res for k, v in r[0].items()}
    # Chat: N short answers, one after another, then at once.
    nonce = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    rec["chat_seq_vals"] = {k: chat(url, img, k, nonce)[0] for k in names}
    rec["chat_seq_s"] = time.perf_counter() - t0
    nonce = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(n) as ex:
        res = list(ex.map(lambda k: chat(url, img, k, nonce), names))
    rec["chat_par_s"] = time.perf_counter() - t0
    rec["chat_par_vals"] = {k: r[0] for k, r in zip(names, res)}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--fields", type=int, default=6)
    ap.add_argument("--out")
    a = ap.parse_args()
    names = list(FIELDS)[: a.fields]
    rows = []
    for p in a.images:
        img = data_url(p)
        rec = {"image": Path(p).name, "fields": len(names), **run_modes(a.url, img, names)}
        rows.append(rec)
        print(f"{rec['image']:28s} one cold {rec['dec_one_cold_s']:5.2f}s warm {rec['dec_one_warm_s']:5.2f}s | "
              f"split seq {rec['dec_split_seq_s']:5.2f}s par {rec['dec_split_par_s']:5.2f}s | "
              f"chat seq {rec['chat_seq_s']:5.2f}s par {rec['chat_par_s']:5.2f}s", flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    keys = ["dec_one_cold_s", "dec_one_warm_s", "dec_split_seq_s", "dec_split_par_s", "chat_seq_s", "chat_par_s"]
    print("\nmean over images (s):", {k: round(statistics.mean(r[k] for r in rows), 2) for k in keys})
    agree = sum(r["dec_one_cold_vals"] == r["dec_split_seq_vals"] == r["dec_split_par_vals"] for r in rows)
    print(f"decision labels identical across one/split-seq/split-par: {agree}/{len(rows)} images")
    print("raw decision response keys:", rows[0].get("raw_keys"))


if __name__ == "__main__":
    main()
