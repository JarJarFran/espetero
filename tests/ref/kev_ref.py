#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Kev pointer-head decisions (jaredpalmer/kev-27b): parity of the engine against mlx-vlm.

  # 0. synthetic test images (PIL only)
  python3 tests/ref/kev_ref.py images --out build/kev_img
  # 1. the encoder fixtures of tests/kev_test.cpp (transformers' tokenizer of the converted directory)
  python -I tests/ref/kev_ref.py tokens --model DIR --images build/kev_img/*.png --out tests/ref/kev_fixtures.json
  # 2. reference probabilities (mlx-vlm 0.7.1 modules, the converted directory) and the Espetero requests
  python -I tests/ref/kev_ref.py decisions --model DIR --images build/kev_img/*.png --out build/kev_dec \\
      [--head SNAPSHOT/head.pt] [--bf16] [--cases text|images|all]
  # 3. a running server (espetero serve --model DIR) against them
  python3 tests/ref/kev_ref.py check --dec build/kev_dec --url http://127.0.0.1:8090

The prompt is the release's encoding (kev/api.py to_record and render,
kev/model.py encode and user_tokens, re-implemented here): the state
`<|fim_prefix|> render(state)`, then per question `<|fim_middle|>
render(instructions)`, each option as `<|box_start|> option <|box_end|>`
(noul: `no` / `yes` with the false / true descriptions as `no: ...`; choice:
`key` or `key: description`; score: the rendered level) and `<|fim_suffix|>`;
caller text is tokenized with `<|name|>` rewritten to `<¦name¦>`. Images
(not part of the release) open the state as the base model's chat template
places an image part before the text, `<|vision_start|>` + the image's
`<|image_pad|>` run + `<|vision_end|>`, resized by transformers' Qwen2-VL
image processor (its PIL backend: bicubic, sides multiples of 32, 65,536 to
--max-pixels pixels, Espetero's standard budget by default).

The forward pass is mlx-vlm's Qwen3.5 language model with fp32 activations
(the vision tower, M-RoPE positions from get_rope_index), one causal row per
question (the state, then the question: the release's row form); the head
reads the final-normed states of the question's <|fim_suffix|> row and of
each option's <|box_end|> row, in float64:
z_k = (W_k h_k + b_k) . (W_q h_d + b_q) / sqrt(head_dim), p = softmax(z / T),
with the head and T from the release's head.pt (`--head`, read without
unpickling by tools/kev_convert.py's reader) or the converted
pointer_head.safetensors.
"""

import argparse
import base64
import json
import math
import os
import re
import sys

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- images
def cmd_images(a):
    from PIL import Image, ImageDraw
    os.makedirs(a.out, exist_ok=True)

    def canvas(w, h, bg=(255, 255, 255)):
        im = Image.new("RGB", (w, h), bg)
        return im, ImageDraw.Draw(im)

    im, d = canvas(512, 512)
    d.rectangle([156, 156, 356, 356], fill=(20, 60, 220))
    im.save(f"{a.out}/0_blue_square.png")
    im, d = canvas(640, 480)
    d.ellipse([220, 140, 420, 340], fill=(220, 30, 30))
    im.save(f"{a.out}/1_red_circle.png")
    im, d = canvas(800, 600, (40, 40, 40))
    d.polygon([(400, 100), (650, 500), (150, 500)], fill=(30, 200, 60))
    im.save(f"{a.out}/2_green_triangle.png")
    im, d = canvas(1000, 400)
    d.rectangle([120, 100, 320, 300], fill=(20, 60, 220))
    d.ellipse([680, 100, 880, 300], fill=(220, 30, 30))
    im.save(f"{a.out}/3_square_left_circle_right.png")
    im, d = canvas(360, 900)
    for i, c in enumerate([(220, 30, 30), (30, 160, 60), (20, 60, 220)]):
        d.rectangle([0, i * 300, 359, i * 300 + 299], fill=c)
    im.save(f"{a.out}/4_stripes_red_green_blue.png")
    im, d = canvas(600, 600, (0, 0, 0))
    d.ellipse([150, 150, 450, 450], fill=(255, 255, 255))
    im.save(f"{a.out}/5_white_circle_on_black.png")
    im, d = canvas(500, 500)
    d.rectangle([100, 100, 400, 400], fill=(130, 40, 170))
    im.save(f"{a.out}/6_purple_square.png")
    im, d = canvas(700, 500)
    for x in (150, 350, 550):
        d.ellipse([x - 60, 190, x + 60, 310], fill=(240, 200, 20))
    im.save(f"{a.out}/7_three_yellow_circles.png")
    print(f"images in {a.out}", file=sys.stderr)


# ---------------------------------------------------------------- the release's encoding
def render(v, indent=0):
    """kev/api.py render: str | object | array -> text, field names kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
                     for k, x in v.items())


def option_text(name, desc):
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def question_keys(qtype, criteria):
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def to_record(body):
    """kev/api.py to_record: (state text, [(instructions text, option texts)]) and per-question metadata."""
    qs, meta = [], []
    for qid, q in body["questions"].items():
        m = {"id": qid, "type": q["type"], "keys": question_keys(q["type"], q.get("criteria"))}
        if q["type"] == "noul":
            c = q.get("criteria") or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        elif q["type"] == "choice":
            opts = [option_text(k, v) for k, v in q["criteria"].items()]
        else:
            opts = [render(x) for x in q["criteria"]]
            m["legend"] = dict(zip(m["keys"], opts))
        qs.append((render(q.get("instructions")), opts))
        meta.append(m)
    return render(body["state"]), qs, meta


def user_tokens(tok, text):
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode(tok, body, image_tokens=()):
    """kev/model.py encode (strict, no option isolation) plus the image blocks: the state ids and per question its
    branch ids with the read-out rows (<decide>, then each option's </opt>) relative to the branch."""
    state, qs, meta = to_record(body)
    st, q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL)
    vs, pad, ve = (tok.convert_tokens_to_ids(t) for t in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"))
    S = [st]
    for n in image_tokens:
        S += [vs] + [pad] * n + [ve]
    S += user_tokens(tok, state)
    branches = []
    for instr, opts in qs:
        br = [q_id] + user_tokens(tok, instr)
        ends = []
        for o in opts:
            br += [o_id] + user_tokens(tok, o) + [c_id]
            ends.append(len(br) - 1)
        br.append(d_id)
        branches.append({"ids": br, "rows": [len(br) - 1] + ends})
    return S, branches, meta


def round_prob(x):
    return round(float(x), 4)


def _normalize(p):
    t = sum(p)
    return [1 / len(p)] * len(p) if t == 0 else [x / t for x in p]


def to_answers(probs, meta):
    """kev/api.py to_answers (the confidence formulas of TypeSafe's system-one-adapter 0.2.1)."""
    out = {}
    for p, m in zip(probs, meta):
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": round_prob(p[1])}
        elif m["type"] == "choice":
            K = len(p)
            conf = 1.0 if K == 1 else (max(_normalize(p)) - 1 / K) / (1 - 1 / K)
            out[m["id"]] = {"type": "choice", "choice": m["keys"][max(range(K), key=lambda i: p[i])],
                            "confidence": round_prob(conf), "probabilities": {k: round_prob(v) for k, v in zip(m["keys"], p)}}
        else:
            L = len(p)
            if L == 1:
                conf = 1.0
            else:
                q = _normalize(p)
                mode = max(range(L), key=q.__getitem__)
                D = sum(abs(i - (L - 1) / 2) for i in range(L)) / L
                conf = max(0.0, 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(q)) / D)
            score = sum(i * pi for i, pi in enumerate(p))
            out[m["id"]] = {"type": "score", "score": round_prob(score), "legend": m["legend"],
                            "probabilities": {str(i): round_prob(v) for i, v in enumerate(p)}, "confidence": round_prob(conf)}
    return out


# ---------------------------------------------------------------- cases
DEPARTMENT = {"returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages",
              "billing": "Charges, invoices, payment problems"}

# (image indices, request body without images). Images are cmd_images'.
CASES = [
    ([], {"state": "The meeting starts at 10 AM. It is now 9 AM.",
          "questions": {"started": {"type": "noul", "instructions": "Has the meeting started?"}}}),
    ([], {"state": "Water boils at 100 degrees Celsius at sea level.",
          "questions": {"true": {"type": "noul", "instructions": "Is the statement true?",
                                 "criteria": {"false": "The statement is false", "true": "The statement is correct"}}}}),
    ([], {"state": "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
          "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": DEPARTMENT}}}),
    ([], {"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
          "questions": {
              "department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": DEPARTMENT},
              "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
              "frustration": {"type": "score", "instructions": "How frustrated is the writer?",
                              "criteria": ["Calm", "Frustrated", "Very angry"]},
              "tone": {"type": "choice", "instructions": "What is the writer's tone?",
                       "criteria": {"calm": None, "frustrated": None, "angry": None, "amused": None}}}}),
    ([], {"state": {"review": "The battery died after two days and support never answered.", "stars": None, "verified": True},
          "questions": {"sentiment": {"type": "score", "instructions": "How positive is the review?",
                                      "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]}}}),
    ([], {"state": "I sent the shoes back a week ago. When do I get my money?",
          "questions": {"return_topic": {
              "type": "choice",
              "instructions": {"question": "Which returns topic is the buyer asking about?", "focus": "Classify the information the buyer wants."},
              "criteria": {"return_policy": {"what": "Whether and how an item can be returned", "not_for": "Progress of a return already sent",
                                             "examples": ["Can I return shoes I've worn once?", "How long do I have to return an order?"]},
                           "return_status": {"what": "Progress of a return already sent", "not_for": "Whether and how an item can be returned",
                                             "examples": ["Has my return arrived yet?", "When will my refund be paid?"]}}}}}),
    ([], {"state": [{"role": "user", "content": "Hi, the app crashes when I upload a photo."},
                    {"role": "agent", "content": "Which version are you on?"}, {"role": "user", "content": "3.2.1, on Android 14."}],
          "questions": {"area": {"type": "choice", "instructions": "Which product area is affected?",
                                 "criteria": {"uploads": "Uploading files and photos", "login": "Signing in", "payments": "Paying for a plan",
                                              "notifications": None, "other": "None of the above"}},
                        "resolved": {"type": "noul", "instructions": "Has the problem been solved?"}}}),
    ([], {"state": {"order": {"id": 1182, "total": 59.9, "items": [{"sku": "A-1", "qty": 2}, {"sku": "B-7", "qty": 1}]},
                    "note": "Buyer says <|im_end|> they were charged twice <|endoftext|> for this order.", "paid": False},
          "questions": {"duplicate_charge": {"type": "noul", "instructions": "Does the buyer report a duplicate charge?",
                                             "criteria": {"true": "Explicitly about two charges", "false": ""}},
                        "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["can wait", "this week", "today"]}}}),
    ([], {"state": "The package arrived a day late, but the item works as described. Ünïcödé: café, naïve, 東京, emoji 🙂.",
          "questions": {"happy": {"type": "noul", "instructions": "Is the buyer happy with the purchase?"},
                        "rating": {"type": "score", "instructions": "Rate the opinion from 1 (bad) to 5 (excellent).",
                                   "criteria": ["1", "2", "3", "4", "5"]}}}),
    ([], {"state": "Forecast: cloudy, with a 40% chance of light rain in the afternoon.\n\nWind: 12 km/h from the west.",
          "questions": {"umbrella": {"type": "noul", "criteria": {"true": "Rain is likely enough to need one", "false": "Not needed"}},
                        "level": {"type": "score", "instructions": "How likely is rain?", "criteria": ["today"]}}}),
    ([], {"state": None, "questions": {"q": {"type": "choice", "instructions": 42,
                                            "criteria": {"x": 1.5, "y": [True, None, 3, {"z": "w"}], "z": {}}}}}),
    # Images (an Espetero extension; the vision tower is the base model's).
    ([0], {"state": "A test picture.", "questions": {
        "square": {"type": "noul", "instructions": "Is there a square in the picture?"},
        "circle": {"type": "noul", "instructions": "Is there a circle in the picture?"},
        "blue": {"type": "noul", "instructions": "Is the shape blue?"},
        "red": {"type": "noul", "instructions": "Is the shape red?"},
        "colour": {"type": "choice", "instructions": "What colour is the shape?", "criteria": {"red": None, "blue": None, "green": None}}}}),
    ([1], {"state": "A test picture.", "questions": {
        "circle": {"type": "noul", "instructions": "Is the shape a circle?"},
        "blue": {"type": "noul", "instructions": "Is the circle blue?"},
        "red": {"type": "noul", "instructions": "Is the shape red?"}}}),
    ([2], {"state": "A test picture.", "questions": {
        "triangle": {"type": "noul", "instructions": "Is there a triangle in the picture?"},
        "white_bg": {"type": "noul", "instructions": "Is the background white?"},
        "green": {"type": "noul", "instructions": "Is the triangle green?"}}}),
    ([3], {"state": "A wide test picture with two shapes.", "questions": {
        "circle_right": {"type": "noul", "instructions": "Is the circle on the right side of the picture?"},
        "square_right": {"type": "noul", "instructions": "Is the square on the right side of the picture?"},
        "two": {"type": "noul", "instructions": "Are there exactly two shapes in the picture?"}}}),
    ([4], {"state": "A tall picture with three horizontal bands.", "questions": {
        "top_red": {"type": "noul", "instructions": "Is the top band red?"},
        "bottom_green": {"type": "noul", "instructions": "Is the bottom band green?"}}}),
    ([5], {"state": "A test picture.", "questions": {
        "black_bg": {"type": "noul", "instructions": "Is the background black?"},
        "black_circle": {"type": "noul", "instructions": "Is the circle black?"}}}),
    ([6], {"state": "A test picture.", "questions": {
        "purple": {"type": "noul", "instructions": "Is the square purple?"},
        "triangle": {"type": "noul", "instructions": "Is there a triangle in the picture?"}}}),
    ([7], {"state": "A test picture.", "questions": {
        "three": {"type": "noul", "instructions": "Are there three circles in the picture?"},
        "yellow": {"type": "noul", "instructions": "Are the circles yellow?"},
        "blue": {"type": "noul", "instructions": "Are the circles blue?"}}}),
    ([0, 1], {"state": "Two test pictures.", "questions": {
        "same": {"type": "noul", "instructions": "Do both pictures show a shape of the same colour?"},
        "first_square": {"type": "noul", "instructions": "Is the shape in the first picture a square?"}}}),
]
# The drawn images' yes / no truth (the graft's sanity check, not a parity target).
TRUTH = {11: {"square": True, "circle": False, "blue": True, "red": False}, 12: {"circle": True, "blue": False, "red": True},
         13: {"triangle": True, "white_bg": False, "green": True}, 14: {"circle_right": True, "square_right": False, "two": True},
         15: {"top_red": True, "bottom_green": False}, 16: {"black_bg": True, "black_circle": False},
         17: {"purple": True, "triangle": False}, 18: {"three": True, "yellow": True, "blue": False},
         19: {"same": False, "first_square": True}}


def selected(which):
    return [(ci, idx, body) for ci, (idx, body) in enumerate(CASES)
            if which == "all" or (which == "images") == bool(idx)]


def data_url(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


class Images:
    """transformers' Qwen2-VL image processor (PIL backend) at Espetero's chat budget."""

    def __init__(self, path, max_pixels):
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        self.ip = Qwen2VLImageProcessorPil.from_pretrained(path)
        self.ip.size = {"shortest_edge": min(65536, max_pixels), "longest_edge": max_pixels}

    def __call__(self, images):
        import numpy as np
        if not images:
            return None, []
        out = self.ip(images=images, return_tensors="np")
        return np.asarray(out["pixel_values"], dtype=np.float32), [list(map(int, g)) for g in out["image_grid_thw"]]


# ---------------------------------------------------------------- 1. encoder fixtures
def cmd_tokens(a):
    from PIL import Image
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    ims = Images(a.model, a.max_pixels)
    images = [Image.open(p).convert("RGB") for p in a.images]
    out = []
    for ci, idx, body in selected("all"):
        if idx and max(idx) >= len(images):
            continue
        _, grids = ims([images[j] for j in idx])
        itoks = [g[0] * g[1] * g[2] // 4 for g in grids]
        S, branches, meta = encode(tok, body, itoks)
        # Answers of fixed probabilities (the release's rounding and confidences, json.dumps of the answers).
        probs = []
        for q, m in zip(branches, meta):
            k = len(q["rows"]) - 1
            w = [math.exp(0.37 * ((3 * j + ci) % 7) - 0.11 * j) for j in range(k)]
            probs.append([x / sum(w) for x in w])
        answers = to_answers(probs, meta)
        out.append({"case": ci, "body": body, "image_tokens": itoks, "state": S,
                    "questions": [{"id": m["id"], "keys": m["keys"], "ids": q["ids"], "rows": q["rows"]} for q, m in zip(branches, meta)],
                    "input_tokens": len(S) + sum(len(q["ids"]) for q in branches),
                    "probs": probs, "answers_json": json.dumps(answers),
                    "output_tokens": len(tok(json.dumps(answers), add_special_tokens=False).input_ids)})
    render_cases = [None, "text", 7, -0.0, 1e-05, 1e16, 123456.789, True, False, [], {}, ["  lead", {"a": [1, {"b": None}]}],
                    {"k": [], "m": {}, "n": [[1, 2], [3]], "o": " spaced"}, [None, "", "\n x"]]
    json.dump({"cases": out, "render": [{"value": v, "text": render(v)} for v in render_cases]},
              open(a.out, "w"), ensure_ascii=False, indent=None, separators=(",", ":"))
    print(f"{len(out)} cases -> {a.out}", file=sys.stderr)


# ---------------------------------------------------------------- 2. reference
def load_head(model, head_pt):
    """(Wq, bq, Wk, bk, T) float64 from pointer_head.safetensors, checked against the release's head.pt when given."""
    import mlx.core as mx
    import numpy as np
    ph = mx.load(os.path.join(model, "pointer_head.safetensors"))
    cfg = json.load(open(os.path.join(model, "pointer_head.json")))
    W = {k: np.asarray(v, dtype=np.float32) for k, v in ph.items()}
    T = float(cfg["temperature"])
    if head_pt:
        import importlib.util
        spec = importlib.util.spec_from_file_location("kev_convert", os.path.join(HERE, "..", "..", "tools", "kev_convert.py"))
        kc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kc)
        rel, t_rel, _ = kc.read_head(head_pt, W["q_weight"].shape[1])
        for k in rel:
            if not np.array_equal(rel[k], W[k]):
                sys.exit(f"pointer_head.safetensors {k} differs from {head_pt}")
        if t_rel != T:
            sys.exit(f"temperature {T} differs from head.pt's {t_rel}")
        print(f"head: {head_pt} (bit-identical to pointer_head.safetensors), T = {T}", file=sys.stderr)
    f = lambda k: W[k].astype(np.float64)
    return f("q_weight"), f("q_bias"), f("k_weight"), f("k_bias"), T


def cmd_decisions(a):
    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from mlx_vlm.utils import load_model
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    ims = Images(a.model, a.max_pixels)
    Wq, bq, Wk, bk, T = load_head(a.model, a.head)
    model = load_model(__import__("pathlib").Path(a.model))
    # The converted norms are fp32 (1 + w), so without a cast the activations are fp32 after the first norm.
    model.set_dtype(mx.bfloat16 if a.bf16 else mx.float32)
    images = [Image.open(p).convert("RGB") for p in a.images]
    os.makedirs(a.out, exist_ok=True)
    out = []
    for ci, idx, body in selected(a.cases):
        if idx and max(idx) >= len(images):
            continue
        pv, grids = ims([images[j] for j in idx])
        itoks = [g[0] * g[1] * g[2] // 4 for g in grids]
        S, branches, meta = encode(tok, body, itoks)
        logits, probs = [], []
        for q in branches:
            ids = S + q["ids"]  # the release's row form: the state, then this question
            kw = {} if pv is None else {"pixel_values": mx.array(pv), "image_grid_thw": mx.array(grids)}
            e = model.get_input_embeddings(mx.array([ids]), **kw)
            h = model.language_model.model(None, inputs_embeds=e.inputs_embeds, position_ids=e.position_ids)
            rows = mx.array([len(S) + r for r in q["rows"]])
            hr = np.asarray(h[0][rows].astype(mx.float32), dtype=np.float64)
            qv = Wq @ hr[0] + bq
            z = (hr[1:] @ Wk.T + bk) @ qv / math.sqrt(len(bq))
            p = np.exp(z / T - (z / T).max())
            p /= p.sum()
            logits.append(z.tolist())
            probs.append(p.tolist())
        req = dict(body)
        if idx:
            req["images"] = [data_url(a.images[j]) for j in idx]
        req["return_logits"] = True
        out.append({"case": ci, "images": [os.path.basename(a.images[j]) for j in idx], "grids": grids,
                    "input_tokens": len(S) + sum(len(q["ids"]) for q in branches), "state_tokens": len(S),
                    "keys": [m["keys"] for m in meta], "ids": [m["id"] for m in meta], "logits": logits, "probs": probs,
                    "answers": to_answers(probs, meta), "truth": TRUTH.get(ci), "body": req})
        best = [f"{m['id']}={m['keys'][int(np.argmax(p))]}({max(p):.3f})" for m, p in zip(meta, probs)]
        print(f"{ci}: {len(S)} state tokens, grids {grids}: {' '.join(best)}", file=sys.stderr, flush=True)
    json.dump({"model": a.model, "fp32": not a.bf16, "max_pixels": a.max_pixels, "temperature": T, "cases": out},
              open(f"{a.out}/decisions.json", "w"), indent=1)


# ---------------------------------------------------------------- 3. the engine
def cmd_check(a):
    """Post every case's body to a running Espetero server and compare its logits and probabilities."""
    import urllib.request
    d = json.load(open(f"{a.dec}/decisions.json"))

    def post(body):
        req = urllib.request.Request(a.url + "/v1/systemone", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=600))

    import urllib.error
    worst_p, worst_z, rows, graft = 0.0, 0.0, [], {"n": 0, "correct": 0, "conf_correct": [], "conf_wrong": []}
    errors, flips = [], 0
    for c in d["cases"]:
        try:
            r = post(c["body"])
        except urllib.error.HTTPError as e:
            errors.append({"case": c["case"], "status": e.code, "detail": e.read().decode(errors="replace")[:300]})
            continue
        T = r["metal"]["temperature"]
        case = {"case": c["case"], "images": c["images"], "same_tokens": r["usage"]["input_tokens"] == c["input_tokens"],
                "passes": r["metal"]["passes"], "cached": r["metal"]["cached_state_tokens"], "ms": r["metal"]["elapsed_ms"],
                "same_answers": True, "questions": []}
        for qid, keys, z_ref, p_ref in zip(c["ids"], c["keys"], c["logits"], c["probs"]):
            z = [r["logits"][qid][k] for k in keys]
            m = max(x / T for x in z)
            e = [math.exp(x / T - m) for x in z]
            p = [x / sum(e) for x in e]
            dp = max(abs(x - y) for x, y in zip(p, p_ref))
            dz = max(abs(x - y) for x, y in zip(z, z_ref))
            worst_p, worst_z = max(worst_p, dp), max(worst_z, dz)
            flips += max(range(len(p)), key=p.__getitem__) != max(range(len(p_ref)), key=p_ref.__getitem__)
            case["questions"].append({"id": qid, "max_dp": round(dp, 6), "max_dlogit": round(dz, 5),
                                      "ref": [round(x, 4) for x in p_ref], "ours": [round(x, 4) for x in p]})
            if c.get("truth") and qid in c["truth"]:
                py = p[1]
                ok = (py > 0.5) == c["truth"][qid]
                graft["n"] += 1
                graft["correct"] += ok
                graft["conf_correct" if ok else "conf_wrong"].append(round(max(py, 1 - py), 4))
        case["same_answers"] = r["answers"] == c["answers"]
        rows.append(case)
    if graft["n"]:
        for k in ("conf_correct", "conf_wrong"):
            v = graft[k]
            graft[k] = {"n": len(v), "mean": round(sum(v) / len(v), 4) if v else None, "min": min(v) if v else None}
    print(json.dumps({"cases": len(rows), "max_dp": round(worst_p, 6), "max_dlogit": round(worst_z, 5), "argmax_flips": flips,
                      "same_tokens": sum(r["same_tokens"] for r in rows),
                      "same_rounded_answers": sum(r["same_answers"] for r in rows),
                      "graft": graft if graft["n"] else None, "errors": errors, "rows": rows}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("images")
    i.add_argument("--out", required=True)
    t = sub.add_parser("tokens")
    t.add_argument("--model", required=True)
    t.add_argument("--images", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--max-pixels", type=int, default=1003520)
    d = sub.add_parser("decisions")
    d.add_argument("--model", required=True)
    d.add_argument("--images", nargs="+", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--head", default="", help="the release's head.pt, checked against pointer_head.safetensors")
    d.add_argument("--bf16", action="store_true", help="bf16 activations and norms (default fp32)")
    d.add_argument("--cases", choices=["text", "images", "all"], default="all")
    d.add_argument("--max-pixels", type=int, default=1003520, help="Espetero's standard image budget")
    c = sub.add_parser("check")
    c.add_argument("--dec", required=True)
    c.add_argument("--url", default="http://127.0.0.1:8090")
    a = ap.parse_args()
    {"images": cmd_images, "tokens": cmd_tokens, "decisions": cmd_decisions, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
