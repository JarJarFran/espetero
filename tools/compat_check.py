#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Checks the other client APIs against a running server: Anthropic
Messages, OpenAI Responses and Prometheus /metrics. Each generation is
compared with /v1/chat/completions on the same conversation (greedy, no
drafts): the adapters must give the same prompt, so the same text. The
chat route's own features are checked too: n choices from one prefill,
logprobs of sampled and streamed replies, forced tool calls, image URLs
refused without --image-urls, embeddings. The Anthropic and
OpenAI SDKs are used too when they are installed. Parameters the server does
not implement must be refused (422), not ignored.

  python3 tools/compat_check.py --base http://127.0.0.1:8080 [--key KEY]
"""
import argparse
import base64
import json
import re
import struct
import sys
import urllib.error
import urllib.request
import zlib

FAILS = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def info(name, detail):
    print(f"info {name}  ({detail})", flush=True)


class Client:
    def __init__(self, base, key):
        self.base, self.key = base.rstrip("/"), key

    def open(self, path, body=None, method=None, headers=None, auth=True):
        h = {"content-type": "application/json"}
        if self.key and auth:
            h["authorization"] = f"Bearer {self.key}"
        h.update(headers or {})
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(self.base + path, data=data, headers=h,
                                   method=method or ("POST" if body is not None else "GET"))
        try:
            return urllib.request.urlopen(r, timeout=900)
        except urllib.error.HTTPError as e:
            return e

    def call(self, path, body=None, method=None, headers=None, auth=True):
        resp = self.open(path, body, method, headers, auth)
        text = resp.read().decode()
        try:
            return resp.status if hasattr(resp, "status") else resp.code, json.loads(text)
        except ValueError:
            return resp.status if hasattr(resp, "status") else resp.code, text

    def stream(self, path, body):
        """(event, data) pairs of a Server-Sent Events response."""
        resp = self.open(path, body)
        ev, data = None, []
        for raw in resp:
            line = raw.decode().rstrip("\r\n")
            if not line:
                if data and data != ["[DONE]"]:  # an OpenAI chat stream ends with [DONE]
                    yield ev, json.loads("\n".join(data))
                ev, data = None, []
            elif line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if data and data != ["[DONE]"]:
            yield ev, json.loads("\n".join(data))


def png(w=48, h=32):
    """A small PNG (red left half, blue right half), stdlib only."""
    rows = b"".join(b"\0" + b"".join(b"\xff\x20\x20" if x < w // 2 else b"\x20\x20\xff" for x in range(w))
                    for _ in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


SYS = "You are a terse assistant. Answer in one short sentence."
Q = "Name three primary colors."
GREEDY = {"temperature": 0, "speculative": False}


def chat(c, messages, max_tokens=48, **extra):
    body = {"messages": messages, "max_tokens": max_tokens, "enable_thinking": False, **GREEDY, **extra}
    st, r = c.call("/v1/chat/completions", body)
    assert st == 200, (st, r)
    return r["choices"][0]["message"], r["usage"]


def anthropic_checks(c):
    ref, usage = chat(c, [{"role": "system", "content": SYS}, {"role": "user", "content": Q}])
    body = {"model": "any", "max_tokens": 48, "system": SYS, **GREEDY,
            "messages": [{"role": "user", "content": [{"type": "text", "text": Q}]}]}
    st, m = c.call("/v1/messages", body)
    check("messages: 200", st == 200, str(m)[:200] if st != 200 else "")
    if st != 200:
        return
    text = "".join(b.get("text", "") for b in m["content"] if b["type"] == "text")
    check("messages: shape", m.get("type") == "message" and m.get("role") == "assistant" and m["id"].startswith("msg_"))
    check("messages: text equals chat", text == ref["content"], f"{text!r} vs {ref['content']!r}")
    check("messages: usage", m["usage"]["output_tokens"] == usage["completion_tokens"]
          and m["usage"]["input_tokens"] + m["usage"]["cache_read_input_tokens"] == usage["prompt_tokens"], str(m["usage"]))
    check("messages: stop_reason", m["stop_reason"] in ("end_turn", "max_tokens"), m["stop_reason"])

    events = list(c.stream("/v1/messages", {**body, "stream": True}))
    names = [e for e, _ in events]
    stext = "".join(d["delta"].get("text", "") for e, d in events
                    if e == "content_block_delta" and d["delta"]["type"] == "text_delta")
    check("messages stream: event order", names[:1] == ["message_start"] and names[-2:] == ["message_delta", "message_stop"]
          and names.count("content_block_start") == names.count("content_block_stop"), " ".join(names[:4]) + " ...")
    check("messages stream: text equals chat", stext == ref["content"], f"{stext!r}")
    check("messages stream: event names match types", all(e == d.get("type") for e, d in events))

    st, ct = c.call("/v1/messages/count_tokens", {"model": "any", "system": SYS,
                                                  "messages": [{"role": "user", "content": Q}]})
    check("count_tokens equals chat prompt_tokens", st == 200 and ct.get("input_tokens") == usage["prompt_tokens"], str(ct))

    # An image block is the chat route's data URL (models with a vision tower).
    st, h = c.call("/health")
    if not (st == 200 and h.get("images")):
        info("messages: image", "model without images, skipped")
        return anthropic_tool_checks(c)
    b64 = base64.b64encode(png()).decode()
    q2 = "What two colors does this image show? One short sentence."
    ref2, _ = chat(c, [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                                                     {"type": "text", "text": q2}]}])
    st, m2 = c.call("/v1/messages", {"model": "any", "max_tokens": 48, **GREEDY, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}, {"type": "text", "text": q2}]}]})
    t2 = "".join(b.get("text", "") for b in m2.get("content", []) if b.get("type") == "text") if st == 200 else str(m2)
    check("messages: image equals chat", st == 200 and t2 == ref2["content"], f"{t2!r}")
    anthropic_tool_checks(c)


def anthropic_tool_checks(c):
    # Tools: a call, then its result.
    tools = [{"name": "get_weather", "description": "Current weather of a city",
              "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
    ask = [{"role": "user", "content": "What is the weather in Madrid right now? Use the tool."}]
    st, m3 = c.call("/v1/messages", {"model": "any", "max_tokens": 256, **GREEDY, "tools": tools, "messages": ask})
    uses = [b for b in m3.get("content", []) if b.get("type") == "tool_use"] if st == 200 else []
    check("messages tools: 200", st == 200, str(m3)[:200] if st != 200 else "")
    if uses:
        check("messages tools: stop_reason tool_use", m3["stop_reason"] == "tool_use", m3["stop_reason"])
        check("messages tools: input is an object", isinstance(uses[0]["input"], dict), str(uses[0]))
        follow = ask + [{"role": "assistant", "content": m3["content"]},
                        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0]["id"],
                                                      "content": "22 C, sunny"}]}]
        st, m4 = c.call("/v1/messages", {"model": "any", "max_tokens": 64, **GREEDY, "tools": tools, "messages": follow})
        check("messages tools: result turn", st == 200, str(m4)[:200] if st != 200 else "")
        if st == 200:
            info("messages tools: result turn", f"blocks {[b.get('type') for b in m4['content']]}")
    else:
        info("messages tools", f"no tool_use block (model's choice): {str(m3)[:160]}")

    st, m5 = c.call("/v1/messages", {"model": "any", "max_tokens": 200, **GREEDY, "thinking": {"type": "enabled", "budget_tokens": 128},
                                     "messages": [{"role": "user", "content": "What is 17 * 23?"}]})
    kinds = [b.get("type") for b in m5.get("content", [])] if st == 200 else []
    info("messages thinking", f"status {st}, blocks {kinds}")


def responses_checks(c):
    ref, usage = chat(c, [{"role": "system", "content": SYS}, {"role": "user", "content": Q}])
    body = {"model": "any", "instructions": SYS, "input": Q, "max_output_tokens": 48, "reasoning": {"effort": "none"}, **GREEDY}
    st, r = c.call("/v1/responses", body)
    check("responses: 200", st == 200, str(r)[:200] if st != 200 else "")
    if st != 200:
        return
    msgs = [o for o in r["output"] if o["type"] == "message"]
    text = "".join(p["text"] for m in msgs for p in m["content"] if p["type"] == "output_text")
    check("responses: shape", r.get("object") == "response" and r["id"].startswith("resp_") and r["status"] in ("completed", "incomplete"))
    check("responses: text equals chat", text == ref["content"], f"{text!r} vs {ref['content']!r}")
    check("responses: usage", r["usage"]["input_tokens"] == usage["prompt_tokens"]
          and r["usage"]["output_tokens"] == usage["completion_tokens"], str(r["usage"]))

    events = list(c.stream("/v1/responses", {**body, "stream": True}))
    if not events:
        check("responses stream: events", False, "no events")
        return
    seqs = [d.get("sequence_number") for _, d in events]
    check("responses stream: sequence numbers", seqs == list(range(len(events))))
    check("responses stream: event names match types", all(e == d.get("type") for e, d in events))
    names = [e for e, _ in events]
    check("responses stream: first and last", names[:2] == ["response.created", "response.in_progress"]
          and names[-1] in ("response.completed", "response.incomplete"), f"{names[:2]} ... {names[-1:]}")
    stext = "".join(d["delta"] for e, d in events if e == "response.output_text.delta")
    if "response" not in events[-1][1]:
        check("responses stream: ends with the response", False, str(events[-1][1])[:200])
        return
    final = events[-1][1]["response"]
    ftext = "".join(p["text"] for o in final["output"] if o["type"] == "message" for p in o["content"])
    check("responses stream: deltas equal final text and chat", stext == ftext == ref["content"], f"{stext!r}")
    added = [d["output_index"] for e, d in events if e == "response.output_item.added"]
    check("responses stream: output indexes in order", added == list(range(len(final["output"]))), str(added))

    # Continuation by previous_response_id = the explicit conversation.
    q2 = "Now name three secondary colors."
    st, r2 = c.call("/v1/responses", {"model": "any", "instructions": SYS, "previous_response_id": r["id"], "input": q2,
                                      "max_output_tokens": 48, "reasoning": {"effort": "none"}, **GREEDY})
    ref2, _ = chat(c, [{"role": "system", "content": SYS}, {"role": "user", "content": Q},
                       {"role": "assistant", "content": text}, {"role": "user", "content": q2}])
    t2 = "".join(p["text"] for o in r2.get("output", []) if o.get("type") == "message" for p in o["content"]) if st == 200 else str(r2)
    check("responses: continuation equals explicit conversation", st == 200 and t2 == ref2["content"], f"{t2!r}")

    st, g = c.call(f"/v1/responses/{r['id']}")
    check("responses: GET stored", st == 200 and g.get("id") == r["id"])
    st, d = c.call(f"/v1/responses/{r['id']}", method="DELETE")
    check("responses: DELETE", st == 200 and d.get("deleted") is True)
    st, _ = c.call(f"/v1/responses/{r['id']}")
    check("responses: GET after DELETE is 404", st == 404)
    st, _ = c.call("/v1/responses", {**body, "previous_response_id": "resp_missing"})
    check("responses: unknown previous_response_id is 404", st == 404)
    # instructions + developer item (Codex-style): one system message, not two (templates refuse a second)
    st, r5 = c.call("/v1/responses", {**body, "input": [{"role": "developer", "content": "Answer in English."},
                                                       {"role": "user", "content": Q}]})
    check("responses: instructions + developer item", st == 200, str(r5)[:200] if st != 200 else "")
    # a stateless client echoing [message, function_call] then the output
    echo = [{"role": "user", "content": "What is the weather in Madrid?"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me check."}]},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\": \"Madrid\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "22 C, sunny"}]
    st, r6 = c.call("/v1/responses", {**body, "instructions": None, "input": echo, "tools": [
        {"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]})
    check("responses: stateless tool echo", st == 200, str(r6)[:200] if st != 200 else "")
    st, _ = c.call("/v1/responses", {**body, "text": {"format": {"type": "json_object"}}})
    check("responses: structured output refused (422)", st == 422)
    st, _ = c.call("/v1/responses", {**body, "tools": [{"type": "web_search"}]})
    check("responses: built-in tools refused (422)", st == 422)

    tools = [{"type": "function", "name": "get_weather", "description": "Current weather of a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
    st, r3 = c.call("/v1/responses", {"model": "any", "input": "What is the weather in Madrid right now? Use the tool.",
                                      "tools": tools, "max_output_tokens": 256, "reasoning": {"effort": "none"}, **GREEDY})
    calls = [o for o in r3.get("output", []) if o.get("type") == "function_call"] if st == 200 else []
    if calls:
        check("responses tools: arguments are JSON", isinstance(json.loads(calls[0]["arguments"]), dict), calls[0]["arguments"])
        st, r4 = c.call("/v1/responses", {"model": "any", "previous_response_id": r3["id"], "tools": tools,
                                          "input": [{"type": "function_call_output", "call_id": calls[0]["call_id"], "output": "22 C, sunny"}],
                                          "max_output_tokens": 64, "reasoning": {"effort": "none"}, **GREEDY})
        check("responses tools: output turn", st == 200, str(r4)[:200] if st != 200 else "")
        if st == 200:
            info("responses tools: output turn", f"items {[o['type'] for o in r4['output']]}")
    else:
        info("responses tools", f"no function_call (model's choice): {str(r3)[:160]}")


def params_checks(c):
    """Parameters refused (422) instead of ignored, tool_choice none, Anthropic's stop_sequence, sampling defaults."""
    msgs = [{"role": "user", "content": "Say hi."}]
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
                                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    schema = {"name": "greeting", "schema": {"type": "object", "properties": {"text": {"type": "string"}}}}
    for name, extra, want in [
            ("response_format json_object", {"response_format": {"type": "json_object"}}, 422),
            ("response_format json_schema", {"response_format": {"type": "json_schema", "json_schema": schema}}, 422),
            ("tool_choice unknown function", {"tools": tools, "tool_choice": {"type": "function", "function": {"name": "nope"}}}, 422),
            ("tool_choice required without tools", {"tool_choice": "required"}, 422),
            ("logit_bias", {"logit_bias": {"15": -100}}, 422),
            ("response_format text", {"response_format": {"type": "text"}}, 200),
            ("logit_bias {}", {"logit_bias": {}}, 200),
            ("tool_choice auto", {"tools": tools, "tool_choice": "auto"}, 200),
            ("tool_choice none", {"tools": tools, "tool_choice": "none"}, 200)]:
        st, r = c.call("/v1/chat/completions", {"messages": msgs, "max_tokens": 8, "enable_thinking": False, **GREEDY, **extra})
        check(f"chat: {name} -> {want}", st == want, f"{st} {str(r)[:160]}")
    st, a = c.call("/v1/tokenize", {"messages": msgs, "tools": tools, "tool_choice": "none"})
    st2, b = c.call("/v1/tokenize", {"messages": msgs})
    check("chat: tool_choice none renders no tools", st == st2 == 200 and a["tokens"] == b["tokens"])

    atools = [{"name": "get_weather", "description": "Current weather of a city",
               "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]
    st, ct = c.call("/v1/messages/count_tokens", {"model": "any", "tools": atools, "tool_choice": {"type": "any"}, "messages": msgs})
    check("count_tokens: tool_choice any counted", st == 200 and ct.get("input_tokens", 0) > 0, str(ct)[:160])
    # Images by URL are fetched only with --image-urls.
    st, cfg = c.call("/v1/config")
    if st == 200 and not cfg.get("effective", {}).get("image-urls"):
        st, r = c.call("/v1/chat/completions", {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}, {"type": "text", "text": "What is it?"}]}],
            "max_tokens": 4})
        check("image URL without --image-urls refused (422, naming the flag)", st == 422 and "--image-urls" in str(r), f"{st} {str(r)[:160]}")

    count = [{"role": "user", "content": "Count from 1 to 10, separated by spaces. Only the numbers."}]
    body = {"model": "any", "max_tokens": 64, **GREEDY, "stop_sequences": ["5"], "messages": count}
    st, m = c.call("/v1/messages", body)
    check("messages: stop_reason stop_sequence", st == 200 and m.get("stop_reason") == "stop_sequence" and m.get("stop_sequence") == "5",
          f"{m.get('stop_reason')} {m.get('stop_sequence')!r} {str(m.get('content'))[:120]}")
    deltas = [d for e, d in c.stream("/v1/messages", {**body, "stream": True}) if e == "message_delta"]
    check("messages stream: stop_reason stop_sequence", len(deltas) == 1 and deltas[0]["delta"].get("stop_reason") == "stop_sequence"
          and deltas[0]["delta"].get("stop_sequence") == "5", str(deltas)[:200])

    st, cfg = c.call("/v1/config")
    sd = cfg.get("effective", {}).get("sampling-defaults", {}) if st == 200 else {}
    check("config: sampling defaults", all(k in sd for k in ("temperature", "top_p", "top_k", "min_p", "source")), str(sd))
    info("config: sampling defaults", str(sd))


def generation_checks(c):
    """N choices from one prefill; logprobs of sampled and streamed replies."""
    base = {"messages": [{"role": "user", "content": Q}], "max_tokens": 24, "enable_thinking": False}
    st, g = c.call("/v1/chat/completions", {**base, **GREEDY, "n": 2})
    texts = [ch["message"]["content"] for ch in g.get("choices", [])] if st == 200 else []
    check("n=2 greedy: identical choices", st == 200 and len(texts) == 2 and texts[0] == texts[1], str(texts)[:160])
    samp = {**base, "temperature": 0.8, "top_k": 20, "seed": 11}
    st1, s1 = c.call("/v1/chat/completions", {**samp, "n": 3})
    st2, s2 = c.call("/v1/chat/completions", {**samp, "n": 3})
    t1 = [ch["message"]["content"] for ch in s1.get("choices", [])] if st1 == 200 else []
    t2 = [ch["message"]["content"] for ch in s2.get("choices", [])] if st2 == 200 else []
    check("n=3 sampled: indexes 0-2, reproducible with the seed", len(t1) == 3 and t1 == t2
          and [ch["index"] for ch in s1["choices"]] == [0, 1, 2], str(t1)[:160])
    if st == 200 and st1 == 200:
        check("n=3: usage counts the prompt once", s1["usage"]["prompt_tokens"] == g["usage"]["prompt_tokens"]
              and s1["usage"]["total_tokens"] == s1["usage"]["prompt_tokens"] + s1["usage"]["completion_tokens"], str(s1["usage"]))
    idx, fin = set(), set()
    for _, d in c.stream("/v1/chat/completions", {**base, **GREEDY, "n": 2, "stream": True}):
        for ch in d.get("choices", []):
            idx.add(ch["index"])
            if ch.get("finish_reason"):
                fin.add(ch["index"])
    check("n=2 stream: chunks and finish of both indexes", idx == fin == {0, 1}, f"{idx} {fin}")
    st, r = c.call("/v1/chat/completions", {**base, "n": 17})
    check("n=17 refused (422)", st == 422, str(st))

    st, a = c.call("/v1/chat/completions", samp)
    st2, b = c.call("/v1/chat/completions", {**samp, "logprobs": True, "top_logprobs": 3})
    if st != 200 or st2 != 200:
        check("logprobs sampled: 200", False, f"{st} {st2} {str(b)[:160]}")
        return
    entries = (b["choices"][0].get("logprobs") or {}).get("content") or []
    check("logprobs sampled: same text as without logprobs", a["choices"][0]["message"]["content"] == b["choices"][0]["message"]["content"])
    check("logprobs sampled: one finite entry per token, 3 alternatives",
          len(entries) == b["usage"]["completion_tokens"] and all(e["logprob"] <= 0 and len(e["top_logprobs"]) == 3 for e in entries))
    mine = [e for e in entries if any(t["id"] == e["id"] for t in e["top_logprobs"])]
    check("logprobs sampled: a listed sampled token has its own value",
          all(abs(next(t["logprob"] for t in e["top_logprobs"] if t["id"] == e["id"]) - e["logprob"]) < 1e-5 for e in mine))
    streamed = []
    for _, d in c.stream("/v1/chat/completions", {**samp, "logprobs": True, "top_logprobs": 3, "stream": True}):
        for ch in d.get("choices", []):
            streamed += (ch.get("logprobs") or {}).get("content") or []
    check("logprobs stream: chunk entries concatenate to the reply's", [e["id"] for e in streamed] == [e["id"] for e in entries])
    st, co = c.call("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 6, "temperature": 0.8, "seed": 3,
                                        "logprobs": 2})
    lp = co["choices"][0].get("logprobs") if st == 200 else None
    check("completions logprobs: legacy format", bool(lp) and len(lp["tokens"]) == len(lp["token_logprobs"]) == len(lp["text_offset"])
          and "".join(lp["tokens"]) == co["choices"][0]["text"], str(co)[:200])


def forced_tool_checks(c):
    """tool_choice required / a named function (Anthropic any / tool) make the reply a parsed call."""
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
                                                "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                               "required": ["city"]}}},
             {"type": "function", "function": {"name": "get_time", "description": "Current time in a timezone",
                                                "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}}}}]
    hello = [{"role": "user", "content": "Hello! How are you?"}]
    for choice, want in (("required", None), ({"type": "function", "function": {"name": "get_time"}}, "get_time")):
        st, r = c.call("/v1/chat/completions", {"messages": hello, "tools": tools, "tool_choice": choice, "max_tokens": 96,
                                                "enable_thinking": False, **GREEDY})
        calls = r["choices"][0]["message"].get("tool_calls") or [] if st == 200 else []
        ok = bool(calls) and r["choices"][0]["finish_reason"] == "tool_calls"
        if ok:
            f = calls[0]["function"]
            ok = f["name"] in ("get_weather", "get_time") and (want is None or f["name"] == want) and isinstance(json.loads(f["arguments"]), dict)
        check(f"chat tool_choice {'required' if want is None else 'function'}: a parsed call", ok, str(r)[:200])
    atools = [{"name": t["function"]["name"], "description": t["function"]["description"],
               "input_schema": t["function"]["parameters"]} for t in tools]
    for choice in ({"type": "any"}, {"type": "tool", "name": "get_weather"}):
        st, m = c.call("/v1/messages", {"model": "any", "max_tokens": 96, **GREEDY, "tools": atools, "tool_choice": choice,
                                        "messages": hello})
        uses = [b for b in m.get("content", []) if b.get("type") == "tool_use"] if st == 200 else []
        check(f"messages tool_choice {choice['type']}: tool_use", bool(uses) and m["stop_reason"] == "tool_use"
              and isinstance(uses[0]["input"], dict) and (choice["type"] == "any" or uses[0]["name"] == "get_weather"), str(m)[:200])
    st, r = c.call("/v1/responses", {"model": "any", "input": "Hello! How are you?", "max_output_tokens": 96, **GREEDY,
                                     "reasoning": {"effort": "none"}, "tool_choice": {"type": "function", "name": "get_time"},
                                     "tools": [{"type": "function", "name": t["function"]["name"],
                                                "parameters": t["function"]["parameters"]} for t in tools]})
    fc = [o for o in r.get("output", []) if o.get("type") == "function_call"] if st == 200 else []
    check("responses tool_choice function: function_call", bool(fc) and fc[0]["name"] == "get_time", str(r)[:200])


def embeddings_checks(c):
    """/v1/embeddings of the loaded model: unit vectors of the model's width, batch equal to single."""
    texts = ["The cat sits on the mat.", "A cat is sitting on a rug.", "Quarterly revenue grew by eight percent."]
    st, r = c.call("/v1/embeddings", {"model": "any", "input": texts})
    if st != 200:
        check("embeddings: 200", False, f"{st} {str(r)[:200]}")
        return
    vecs = [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
    width = len(vecs[0])
    check("embeddings: one unit vector per input", len(vecs) == 3 and all(len(v) == width and abs(sum(x * x for x in v) - 1) < 1e-3 for v in vecs))
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    info("embeddings: cosines", f"paraphrase {dot(vecs[0], vecs[1]):.3f}, unrelated {dot(vecs[0], vecs[2]):.3f} / {dot(vecs[1], vecs[2]):.3f}")
    st, one = c.call("/v1/embeddings", {"model": "any", "input": texts[1]})
    diff = max(abs(x - y) for x, y in zip(one["data"][0]["embedding"], vecs[1])) if st == 200 else 1
    check("embeddings: an input alone equals it in a batch", diff < 1e-3, f"max |diff| {diff:.2e}")
    st, b = c.call("/v1/embeddings", {"model": "any", "input": texts[0], "encoding_format": "base64"})
    if st == 200:
        raw = base64.b64decode(b["data"][0]["embedding"])
        fl = struct.unpack(f"<{len(raw) // 4}f", raw)
        check("embeddings: base64 is the float32 vector", len(fl) == width and max(abs(x - y) for x, y in zip(fl, vecs[0])) < 1e-6)
    st, _ = c.call("/v1/embeddings", {"model": "any", "input": "x", "dimensions": width // 2})
    check("embeddings: dimensions below the width refused (422)", st == 422, str(st))
    check("embeddings: usage", r["usage"]["prompt_tokens"] == r["usage"]["total_tokens"] > 0, str(r["usage"]))


def metrics_checks(c):
    resp = c.open("/metrics")
    text = resp.read().decode()
    check("metrics: 200", resp.status == 200)
    sample = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? (NaN|[+-]Inf|[-+0-9.eE]+)$')
    bad = [l for l in text.splitlines() if l and not l.startswith("#") and not sample.match(l)]
    check("metrics: exposition format", not bad, str(bad[:3]))
    check("metrics: counters and info", "espetero_passes_total" in text and "espetero_info{" in text)


def auth_checks(c):
    body = {"model": "any", "messages": [{"role": "user", "content": "hi"}]}
    st, _ = c.call("/v1/messages/count_tokens", body, headers={"x-api-key": c.key}, auth=False)
    check("auth: x-api-key accepted", st == 200, str(st))
    st, _ = c.call("/v1/messages/count_tokens", body, headers={"x-api-key": c.key + "x"}, auth=False)
    check("auth: wrong x-api-key refused", st == 401, str(st))
    st, _ = c.call("/v1/messages/count_tokens", body, auth=False)
    check("auth: no key refused", st == 401, str(st))


def sdk_checks(c):
    try:
        import anthropic
    except ImportError:
        info("anthropic SDK", "not installed")
    else:
        a = anthropic.Anthropic(base_url=c.base, api_key=c.key or "none")
        ref, _ = chat(c, [{"role": "system", "content": SYS}, {"role": "user", "content": Q}], max_tokens=96)
        # newer SDKs dropped the temperature keyword: send it in the body
        kw = dict(model="any", max_tokens=96, system=SYS, messages=[{"role": "user", "content": Q}],
                  extra_body={"temperature": 0, "speculative": False})
        try:
            m = a.messages.create(**kw)
            check("anthropic SDK: create", m.content[0].text == ref["content"], repr(m.content[0].text))
            with a.messages.stream(**kw) as s:
                fm = s.get_final_message()
            check("anthropic SDK: stream", fm.content[0].text == ref["content"], repr(fm.content[0].text))
            tm = a.messages.create(model="any", max_tokens=64, messages=[{"role": "user", "content": "Hello!"}],
                                   tools=[{"name": "get_time", "input_schema": {"type": "object", "properties": {"tz": {"type": "string"}}}}],
                                   tool_choice={"type": "tool", "name": "get_time"}, extra_body={"temperature": 0})
            check("anthropic SDK: forced tool_use", any(b.type == "tool_use" and b.name == "get_time" for b in tm.content))
        except Exception as e:  # an SDK error is a failed check, not the end of the run
            check("anthropic SDK", False, f"{type(e).__name__}: {e}"[:200])
    try:
        import openai
    except ImportError:
        info("openai SDK", "not installed")
    else:
        o = openai.OpenAI(base_url=c.base + "/v1", api_key=c.key or "none")
        try:  # n choices and logprobs through the SDK
            cc = o.chat.completions.create(model="any", messages=[{"role": "user", "content": Q}], max_tokens=16, n=2,
                                           temperature=0.8, seed=5, logprobs=True, top_logprobs=2,
                                           extra_body={"enable_thinking": False})
            check("openai SDK: chat n=2 with logprobs", len(cc.choices) == 2 and all(ch.logprobs and ch.logprobs.content for ch in cc.choices))
        except Exception as e:
            check("openai SDK: chat n=2 with logprobs", False, f"{type(e).__name__}: {e}"[:200])
        try:  # embeddings through the SDK (it asks for base64 and decodes it)
            em = o.embeddings.create(model="any", input=["The cat sits on the mat.", "A cat is sitting on a rug."])
            check("openai SDK: embeddings", len(em.data) == 2 and len(em.data[0].embedding) > 0)
        except Exception as e:
            check("openai SDK: embeddings", False, f"{type(e).__name__}: {e}"[:200])
        try:  # forced tool calls through the SDK
            cc = o.chat.completions.create(
                model="any", messages=[{"role": "user", "content": "Hello!"}], max_tokens=64, temperature=0,
                tools=[{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}}}}],
                tool_choice={"type": "function", "function": {"name": "get_time"}}, extra_body={"enable_thinking": False})
            check("openai SDK: forced tool call", cc.choices[0].message.tool_calls[0].function.name == "get_time")
        except Exception as e:
            check("openai SDK: forced tool call", False, f"{type(e).__name__}: {e}"[:200])
        ref, _ = chat(c, [{"role": "system", "content": SYS}, {"role": "user", "content": Q}], max_tokens=96)
        kw = dict(model="any", instructions=SYS, input=Q, max_output_tokens=96, temperature=0, reasoning={"effort": "none"},
                  extra_body={"speculative": False})
        try:
            r = o.responses.create(**kw)
            check("openai SDK: responses.create", r.output_text == ref["content"], repr(r.output_text))
            # get_final_response() needs response.completed: a reply cut at the limit ends with response.incomplete
            with o.responses.stream(**kw) as s:
                last = None
                for ev in s:
                    if ev.type in ("response.completed", "response.incomplete"):
                        last = ev.response
            text = last.output_text if last is not None else None
            check("openai SDK: responses.stream", text == ref["content"], repr(text))
        except Exception as e:
            check("openai SDK", False, f"{type(e).__name__}: {e}"[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default="")
    a = ap.parse_args()
    c = Client(a.base, a.key)
    anthropic_checks(c)
    responses_checks(c)
    params_checks(c)
    generation_checks(c)
    forced_tool_checks(c)
    embeddings_checks(c)
    metrics_checks(c)
    if a.key:
        auth_checks(c)
    sdk_checks(c)
    print("ALL OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
