#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Structured output against a running server.

  suite:      every schema greedy and sampled, thinking off and on, streamed
              and not, n > 1; json_object; Responses text.format; completions;
              tool calls beside a format; refusals (422). Each reply must parse
              and (with Python's jsonschema installed) validate.
  speed:      decode tok/s of constrained replies against unconstrained ones
              asking for the same JSON, and the masks' own time.
  identity:   unconstrained replies (greedy, seeded samples, n, logprobs,
              stream, completions) saved, to compare a build with the base
              (--compare a.json b.json: token-identical replies expected).

  python3 tools/structured_check.py --base http://127.0.0.1:8080 suite
  python3 tools/structured_check.py --base http://127.0.0.1:8080 speed
  python3 tools/structured_check.py --base http://127.0.0.1:8080 identity --out new.json
  python3 tools/structured_check.py compare base.json new.json
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

try:
    import jsonschema
except ImportError:  # parse-only checks
    jsonschema = None

SCHEMAS = {
    "person": ({"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0, "maximum": 150},
                                                 "email": {"type": "string"}},
                "required": ["name", "age"], "additionalProperties": False},
               "Invent a person: give their name, age and maybe an email."),
    "nested": ({"type": "object", "properties": {
        "id": {"type": "integer", "minimum": 1},
        "address": {"type": "object", "properties": {"street": {"type": "string"}, "city": {"type": "string", "minLength": 1},
                                                     "zip": {"type": ["string", "null"]}},
                    "required": ["street", "city", "zip"], "additionalProperties": False},
        "tags": {"type": "array", "items": {"type": "string", "maxLength": 20}, "minItems": 1, "maxItems": 4}},
        "required": ["id", "address", "tags"], "additionalProperties": False},
        "Describe a customer record for a shop in Madrid, with a few tags."),
    "enum": ({"type": "object", "properties": {"sentiment": {"enum": ["positive", "negative", "neutral"]},
                                               "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
              "required": ["sentiment", "confidence"], "additionalProperties": False},
             "Classify the sentiment of: 'The food was cold and the waiter ignored us.'"),
    "anyof": ({"type": "object", "properties": {"answer": {"anyOf": [{"type": "integer"}, {"type": "string", "maxLength": 30},
                                                                      {"type": "null"}]}},
               "required": ["answer"], "additionalProperties": False},
              "How many legs does a spider have?"),
    "ref": ({"$defs": {"point": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                                 "required": ["x", "y"], "additionalProperties": False}},
             "type": "object", "properties": {"start": {"$ref": "#/$defs/point"}, "end": {"$ref": "#/$defs/point"}},
             "required": ["start", "end"], "additionalProperties": False},
            "Give the start and end points of a line segment from the origin to (3, 4)."),
    "tree": ({"$defs": {"node": {"type": "object", "properties": {
        "name": {"type": "string"}, "children": {"type": "array", "items": {"$ref": "#/$defs/node"}, "maxItems": 3}},
        "required": ["name", "children"], "additionalProperties": False}}, "$ref": "#/$defs/node"},
        "Write a small file system tree: a root folder with two subfolders, one of which has a file."),
    "optional": ({"type": "object", "properties": {"title": {"type": "string"}, "year": {"type": "integer"},
                                                   "director": {"type": "string"}, "rating": {"type": "number"}},
                  "required": ["title"], "additionalProperties": False},
                 "Tell me about a famous film."),
    "nullable": ({"type": "object", "properties": {"capital": {"type": "string"}, "population": {"type": ["integer", "null"]},
                                                   "motto": {"type": "string", "nullable": True}},
                  "required": ["capital", "population", "motto"], "additionalProperties": False},
                 "Give facts about France: its capital, its population and its motto."),
    "const": ({"type": "object", "properties": {"type": {"const": "event"}, "name": {"type": "string"},
                                                "day": {"type": "integer", "minimum": 1, "maximum": 31}},
               "required": ["type", "name", "day"], "additionalProperties": False},
              "Create a calendar event for a dentist appointment on the 14th."),
    "oneof": ({"oneOf": [
        {"type": "object", "properties": {"shape": {"const": "circle"}, "radius": {"type": "number", "exclusiveMinimum": 0}},
         "required": ["shape", "radius"], "additionalProperties": False},
        {"type": "object", "properties": {"shape": {"const": "rectangle"}, "width": {"type": "number", "exclusiveMinimum": 0},
                                          "height": {"type": "number", "exclusiveMinimum": 0}},
         "required": ["shape", "width", "height"], "additionalProperties": False}]},
        "Describe a shape that is 4 wide and 2 high."),
    "array_of_objects": ({"type": "object", "properties": {"items": {"type": "array", "minItems": 2, "maxItems": 4, "items": {
        "type": "object", "properties": {"product": {"type": "string"}, "quantity": {"type": "integer", "minimum": 1},
                                         "price": {"type": "number", "minimum": 0}},
        "required": ["product", "quantity", "price"], "additionalProperties": False}}},
        "required": ["items"], "additionalProperties": False},
        "List a grocery order of three products with quantities and prices."),
    "lengths": ({"type": "object", "properties": {"code": {"type": "string", "minLength": 3, "maxLength": 3},
                                                  "country": {"type": "string", "minLength": 2, "maxLength": 40}},
                 "required": ["code", "country"], "additionalProperties": False},
                "Give the three-letter ISO code and the name of the country whose capital is Lisbon."),
    "allof": ({"allOf": [{"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
                         {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}]},
              "Make a labelled item with a numeric id."),
    "dictionary": ({"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}, "minProperties": 2, "maxProperties": 6},
                   "Count the vowels in the word 'education': one key per vowel."),
    "tuple": ({"type": "object", "properties": {"pair": {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}],
                                                         "items": False, "minItems": 2}},
               "required": ["pair"], "additionalProperties": False},
              "Give a city and its approximate population in millions as a pair."),
    "invoice": ({"type": "object", "properties": {
        "invoice_number": {"type": "string"}, "date": {"type": "string"},
        "customer": {"type": "object", "properties": {"name": {"type": "string"}, "vat_id": {"type": ["string", "null"]}},
                     "required": ["name", "vat_id"], "additionalProperties": False},
        "lines": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
            "description": {"type": "string"}, "amount": {"type": "number"}}, "required": ["description", "amount"],
            "additionalProperties": False}},
        "total": {"type": "number"}, "paid": {"type": "boolean"}},
        "required": ["invoice_number", "date", "customer", "lines", "total", "paid"], "additionalProperties": False},
        "Extract the invoice: 'Invoice 2026-118, 3 March 2026, to Acme S.L. (VAT ESB12345678): design work 1200.00, "
        "hosting 80.50; total 1280.50, not paid yet.'"),
}


def nullable(s):  # OpenAPI's nullable as a null type, for jsonschema
    if isinstance(s, dict):
        s = {k: nullable(v) for k, v in s.items()}
        if s.pop("nullable", False) is True and "type" in s:
            t = s["type"] if isinstance(s["type"], list) else [s["type"]]
            s["type"] = t if "null" in t else t + ["null"]
        return s
    if isinstance(s, list):
        return [nullable(x) for x in s]
    return s


def valid(schema, text):
    try:
        v = json.loads(text)
    except ValueError as e:
        return f"not JSON: {e}"
    if schema is None:
        return None if isinstance(v, dict) else "not an object"
    if jsonschema is None:
        return None
    s = nullable(schema)
    errs = list(jsonschema.validators.validator_for(s, default=jsonschema.Draft202012Validator)(s).iter_errors(v))
    return errs[0].message if errs else None


class Client:
    def __init__(self, base, key=None):
        self.base = base.rstrip("/")
        self.key = key

    def post(self, path, body, stream=False):
        h = {"content-type": "application/json"}
        if self.key:
            h["authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers=h)
        try:
            r = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")
        if not stream:
            with r:
                return r.status, json.loads(r.read())
        chunks = []
        with r:
            for line in r:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(json.loads(line[6:]))
        return r.status, chunks


def chat_body(schema, prompt, **kw):
    rf = {"type": "json_object"} if schema is None else {
        "type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}}
    body = {"model": "m", "messages": [{"role": "user", "content": prompt}], "response_format": rf, "max_tokens": 600}
    body.update(kw)
    return body


def stream_text(chunks, n=1):
    content = [""] * n
    reasoning = [""] * n
    finish = [None] * n
    for c in chunks:
        for ch in c.get("choices", []):
            i = ch.get("index", 0)
            d = ch.get("delta", {})
            content[i] += d.get("content") or ""
            reasoning[i] += d.get("reasoning_content") or ""
            if ch.get("finish_reason"):
                finish[i] = ch["finish_reason"]
    return content, reasoning, finish


def suite(c, args):
    rows, bad, cut = [], 0, 0
    configs = [
        ("greedy", {"temperature": 0, "enable_thinking": False}, False, 1),
        ("sampled", {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "seed": 3, "enable_thinking": False}, False, 1),
        ("greedy/stream", {"temperature": 0, "enable_thinking": False}, True, 1),
        ("sampled/stream", {"temperature": 1.0, "seed": 5, "enable_thinking": False}, True, 1),
        ("n=3", {"temperature": 0.8, "seed": 9, "n": 3, "enable_thinking": False}, False, 3),
        ("think/greedy", {"temperature": 0, "enable_thinking": True, "max_tokens": 1500}, False, 1),
        ("think/sampled/stream", {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 4, "enable_thinking": True,
                                  "max_tokens": 1500}, True, 1),
    ]
    if args.quick:
        configs = configs[:5]
    items = list(SCHEMAS.items()) + [("json_object", (None, "Describe your favourite book as a JSON object."))]
    if args.schemas:
        items = [x for x in items if x[0] in args.schemas.split(",")]
    for name, (schema, prompt) in items:
        for label, kw, stream, n in configs:
            if (label.startswith("think") and args.no_think) or (args.only and args.only not in label):
                continue
            body = chat_body(schema, prompt, stream=stream, **kw)
            t0 = time.time()
            st, j = c.post("/v1/chat/completions", body, stream)
            dt = time.time() - t0
            if st != 200:
                rows.append((name, label, f"HTTP {st}: {j}"))
                bad += 1
                print(f"{name:18s} {label:22s} HTTP {st} {j}", flush=True)
                continue
            if stream:
                contents, reasonings, finishes = stream_text(j, n)
            else:
                contents = [ch["message"].get("content") or "" for ch in j["choices"]]
                reasonings = [ch["message"].get("reasoning_content") or "" for ch in j["choices"]]
                finishes = [ch["finish_reason"] for ch in j["choices"]]
            for i in range(n):
                err = valid(schema, contents[i])
                if finishes[i] == "length":  # max_tokens cut the value: not a violation, counted apart
                    cut += 1
                    rows.append((name, label, "length"))
                    print(f"{name:18s} {label:22s} cut {dt:5.1f}s (max_tokens) {contents[i][:70]!r}", flush=True)
                    continue
                ok = err is None
                bad += not ok
                rows.append((name, label, "ok" if ok else err))
                think = f" think {len(reasonings[i])}c" if reasonings[i] else ""
                print(f"{name:18s} {label:22s} {'ok ' if ok else 'BAD'} {dt:5.1f}s{think} {contents[i][:90]!r}"
                      + ("" if ok else f"  <- {err}"), flush=True)
    if args.no_extras:
        print(f"suite: {len(rows)} constrained replies: {len(rows) - bad - cut} valid, {cut} cut by max_tokens, {bad} failures")
        return bad
    # Responses API (text.format) and completions (response_format)
    schema, prompt = SCHEMAS["person"]
    st, j = c.post("/v1/responses", {"model": "m", "input": prompt, "max_output_tokens": 300, "temperature": 0,
                                     "reasoning": {"effort": "none"},
                                     "text": {"format": {"type": "json_schema", "name": "person", "schema": schema, "strict": True}}})
    text = "".join(p.get("text", "") for o in j.get("output", []) if o.get("type") == "message" for p in o.get("content", []))
    err = valid(schema, text) if st == 200 else f"HTTP {st} {j}"
    echo = j.get("text", {}).get("format", {}).get("type") if st == 200 else None
    bad += err is not None or echo != "json_schema"
    print(f"{'responses':18s} {'text.format':22s} {'ok ' if err is None and echo == 'json_schema' else 'BAD'} {text[:90]!r} {err or ''}")
    st, j = c.post("/v1/completions", {"model": "m", "prompt": "A JSON object describing a cat:", "max_tokens": 600, "temperature": 0,
                                       "response_format": {"type": "json_object"}})
    text = j["choices"][0]["text"] if st == 200 else ""
    err = valid(None, text) if st == 200 else f"HTTP {st} {j}"
    if st == 200 and j["choices"][0]["finish_reason"] == "length":
        cut += 1
        print(f"{'completions':18s} {'json_object':22s} cut (max_tokens) {text[:70]!r}")
    else:
        bad += err is not None
        print(f"{'completions':18s} {'json_object':22s} {'ok ' if err is None else 'BAD'} {text[:90]!r} {err or ''}")
    # Anthropic Messages: output_config.format (strict)
    st, j = c.post("/v1/messages", {"model": "m", "max_tokens": 300, "temperature": 0,
                                    "messages": [{"role": "user", "content": prompt}],
                                    "output_config": {"format": {"type": "json_schema", "schema": schema}}})
    text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text") if st == 200 else ""
    err = valid(schema, text) if st == 200 else f"HTTP {st} {j}"
    bad += err is not None
    print(f"{'anthropic':18s} {'output_config.format':22s} {'ok ' if err is None else 'BAD'} {text[:90]!r} {err or ''}")
    # Tools beside a format: a call or the JSON
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
                                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                              "required": ["city"]}}}]
    for q in ["What's the weather in Paris right now? Use the tool.", "Give me a person record for Ada Lovelace."]:
        st, j = c.post("/v1/chat/completions", chat_body(SCHEMAS["person"][0], q, temperature=0, enable_thinking=False, tools=tools))
        m = j["choices"][0]["message"] if st == 200 else {}
        calls = m.get("tool_calls")
        err = None if calls else valid(SCHEMAS["person"][0], m.get("content") or "")
        if st != 200:
            err = f"HTTP {st} {j}"
        bad += err is not None
        print(f"{'tools+format':18s} {('call ' + calls[0]['function']['name']) if calls else 'json':22s} {'ok ' if err is None else 'BAD'} "
              f"{(m.get('content') or '')[:80]!r} {err or ''}")
    # Non-strict: an unenforced keyword is reported, not refused
    s2 = {"type": "object", "properties": {"zip": {"type": "string", "pattern": "^[0-9]{5}$"}}, "required": ["zip"]}
    body = chat_body(s2, "Give a Spanish postal code.", temperature=0, enable_thinking=False)
    body["response_format"]["json_schema"]["strict"] = False
    st, j = c.post("/v1/chat/completions", body)
    rep = j.get("structured_output", {}).get("unenforced") if st == 200 else None
    ok = st == 200 and rep == ["pattern at #/properties/zip"] and valid({"type": "object", "properties": {"zip": {"type": "string"}},
                                                                         "required": ["zip"]}, j["choices"][0]["message"]["content"]) is None
    bad += not ok
    print(f"{'non-strict':18s} {'pattern reported':22s} {'ok ' if ok else 'BAD'} {rep}")
    # Refusals
    refusals = [
        ("strict pattern", chat_body({"type": "string", "pattern": "^a"}, "x"), "pattern"),
        ("strict format", chat_body({"type": "object", "properties": {"d": {"type": "string", "format": "date"}}}, "x"), "format"),
        ("strict multipleOf", chat_body({"type": "integer", "multipleOf": 5}, "x"), "multipleOf"),
        ("unknown type", chat_body({"type": "color"}, "x"), "unknown type"),
        ("unsatisfiable", chat_body({"type": "string", "minLength": 3, "maxLength": 1}, "x"), "admits no value"),
        ("remote $ref", chat_body({"$ref": "https://example.com/s.json"}, "x"), "$ref"),
        ("no schema", {"model": "m", "messages": [{"role": "user", "content": "x"}],
                       "response_format": {"type": "json_schema", "json_schema": {"name": "a"}}}, "schema"),
        ("bad type", {"model": "m", "messages": [{"role": "user", "content": "x"}], "response_format": {"type": "xml"}}, "json_schema"),
        ("forced call", chat_body(None, "x", tools=tools, tool_choice="required"), "forced tool call"),
        ("ignore_eos", chat_body(None, "x", ignore_eos=True), "ignore_eos"),
        ("responses strict", None, "pattern"),
    ]
    for label, body, word in refusals:
        if body is None:
            st, j = c.post("/v1/responses", {"model": "m", "input": "x", "text": {"format": {
                "type": "json_schema", "name": "a", "schema": {"type": "string", "pattern": "a"}, "strict": True}}})
        else:
            st, j = c.post("/v1/chat/completions", body)
        msg = json.dumps(j)
        ok = st == 422 and word in msg
        bad += not ok
        print(f"{'refusal':18s} {label:22s} {'ok ' if ok else 'BAD'} {st} {j.get('error', {}).get('message', msg)[:110]}")
    print(f"suite: {len(rows)} constrained replies: {len(rows) - bad - cut} valid, {cut} cut by max_tokens, {bad} failures")
    return bad


def speed(c, args):
    # The same requests without and with the format: decode tok/s, and the masks' time.
    names = ["invoice", "nested", "array_of_objects", "tree", "person"]
    res = {"plain": [], "schema": [], "json_object": []}
    mask_ms = masks = 0
    for rep in range(args.reps):
        for name in names:
            schema, prompt = SCHEMAS[name]
            p2 = prompt + " Answer with a JSON object only."
            for mode in ["plain", "schema", "json_object"]:
                body = {"model": "m", "messages": [{"role": "user", "content": p2}], "max_tokens": 400,
                        "temperature": args.temperature, "enable_thinking": False, "seed": 1 + rep}
                if args.temperature > 0:
                    body.update({"top_p": 0.95, "top_k": 20})
                if mode == "schema":
                    body["response_format"] = {"type": "json_schema", "json_schema": {"name": "a", "schema": schema, "strict": True}}
                elif mode == "json_object":
                    body["response_format"] = {"type": "json_object"}
                st, j = c.post("/v1/chat/completions", body)
                if st != 200:
                    print("HTTP", st, j)
                    continue
                t = j.get("timings", {})
                n = j["usage"]["completion_tokens"]
                dec = t.get("decode_ms", 0)
                res[mode].append((n, dec))
                s = t.get("structured")
                if s:
                    mask_ms += s.get("mask_ms", 0)
                    masks += s.get("masks_computed", 0)
                print(f"{name:18s} {mode:12s} {n:4d} tok {(n - 1) / dec * 1000 if dec else 0:7.1f} tok/s"
                      + (f"  masks {s['masks_computed']} new / {s['masks_cached']} cached, {s['mask_ms']:.1f} ms" if s else ""), flush=True)
    for mode, v in res.items():
        toks = sum(n - 1 for n, _ in v)
        ms = sum(d for _, d in v)
        print(f"{mode:12s} {len(v)} replies, {toks} tokens, {toks / ms * 1000 if ms else 0:.1f} tok/s decode")
    print(f"masks computed: {masks}, {mask_ms:.1f} ms in all")


IDENTITY = [
    ("chat greedy", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Explain how a hash map works, with a short C example."}],
                                             "temperature": 0, "max_tokens": 200, "enable_thinking": False}),
    ("chat greedy think", "/v1/chat/completions", {"messages": [{"role": "user", "content": "What is 17 * 23? Think it through."}],
                                                   "temperature": 0, "max_tokens": 300, "enable_thinking": True}),
    ("chat sampled", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Write a short poem about the sea."}],
                                              "temperature": 0.8, "top_p": 0.95, "top_k": 20, "seed": 42, "max_tokens": 200,
                                              "enable_thinking": False}),
    ("chat sampled topk0", "/v1/chat/completions", {"messages": [{"role": "user", "content": "List ten animals and a fact about each."}],
                                                    "temperature": 0.7, "top_p": 0.9, "top_k": 0, "seed": 7, "max_tokens": 200,
                                                    "enable_thinking": False}),
    ("chat n=3", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Name a colour and say why you like it."}],
                                          "temperature": 1.0, "seed": 11, "n": 3, "max_tokens": 80, "enable_thinking": False}),
    ("chat logprobs", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hello in five languages."}],
                                               "temperature": 0, "max_tokens": 80, "logprobs": True, "top_logprobs": 3,
                                               "enable_thinking": False}),
    ("chat json in prompt", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Return a JSON object with a name and an age."}],
                                                     "temperature": 0, "max_tokens": 120, "enable_thinking": False}),
    ("chat stream", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Count from one to twenty in words."}],
                                             "temperature": 0, "max_tokens": 150, "stream": True, "enable_thinking": False}),
    ("completions", "/v1/completions", {"prompt": "The three laws of thermodynamics are", "temperature": 0, "max_tokens": 150}),
    ("completions sampled", "/v1/completions", {"prompt": "Once upon a time", "temperature": 0.9, "seed": 3, "max_tokens": 150}),
]


def identity(c, args):
    out = []
    for label, path, body in IDENTITY:
        b = {"model": "m", **body}
        st, j = c.post(path, b, b.get("stream", False))
        if b.get("stream"):
            texts = stream_text(j)[0]
            toks = None
        else:
            texts = [(ch.get("message", {}).get("content") or "") + "|" + (ch.get("message", {}).get("reasoning_content") or "")
                     if "message" in ch else ch.get("text", "") for ch in j["choices"]]
            toks = j["usage"]["completion_tokens"]
        ids = None
        if label == "chat logprobs":
            ids = [e.get("id") for e in j["choices"][0]["logprobs"]["content"]]
        out.append({"label": label, "texts": texts, "tokens": toks, "ids": ids})
        print(f"{label:22s} {toks} tokens  {texts[0][:70]!r}", flush=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)


def greedy(c, args):
    # Greedy constrained replies (with timings.structured), comparable with compare.
    out, mism = [], 0
    for name, (schema, prompt) in list(SCHEMAS.items()) + [("json_object", (None, "Describe your favourite book as a JSON object."))]:
        st, j = c.post("/v1/chat/completions", chat_body(schema, prompt, temperature=0, enable_thinking=args.think,
                                                        max_tokens=2000 if args.think else 600))
        text = j["choices"][0]["message"].get("content") or "" if st == 200 else f"HTTP {st}"
        s = j.get("timings", {}).get("structured", {}) if st == 200 else {}
        mism += s.get("argmax_mismatches", 0)
        out.append({"label": name, "texts": [text], "tokens": j["usage"]["completion_tokens"] if st == 200 else None, "ids": None})
        err = valid(schema, text)
        print(f"{name:18s} {'ok ' if err is None else 'BAD'} {j['choices'][0]['finish_reason'] if st == 200 else ''} "
              f"checks {s.get('argmax_checks', '-')} mismatches {s.get('argmax_mismatches', '-')} {text[:60]!r}", flush=True)
    print(f"argmax mismatches: {mism}")
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    same = 0
    for x, y in zip(a, b):
        eq = x["texts"] == y["texts"] and x["tokens"] == y["tokens"] and x["ids"] == y["ids"]
        same += eq
        print(f"{x['label']:22s} {'identical' if eq else 'DIFFERENT'}")
    print(f"{same}/{len(a)} identical")
    return 0 if same == len(a) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--key")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--out", default="identity.json")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--only", help="suite: only the configurations whose label contains this")
    ap.add_argument("--schemas", help="suite: only these schemas (comma-separated names)")
    ap.add_argument("--no-extras", action="store_true", help="suite: skip the Responses / completions / tools / refusal checks")
    ap.add_argument("cmd", choices=["suite", "speed", "identity", "greedy", "compare"])
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()
    if args.cmd == "compare":
        return compare(*args.files)
    c = Client(args.base, args.key)
    if args.cmd == "suite":
        return 1 if suite(c, args) else 0
    if args.cmd == "speed":
        speed(c, args)
    if args.cmd == "identity":
        identity(c, args)
    if args.cmd == "greedy":
        greedy(c, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
