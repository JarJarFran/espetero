#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Multi-turn tool-calling benchmark: one agent conversation against a
server, greedy, with scripted tools whose results depend only on their
arguments. Each turn re-sends the whole conversation (as agents do); per
turn it reports the prompt tokens, the tokens the server reports as cached,
the tokens it prefilled, the server's time to first token (timings.prefill_ms)
and the reply's length, and writes every reply to --out (JSON) so two runs
can be compared token for token:

  tools/bench_turns.py --url http://127.0.0.1:8080 --turns 8 --out a.json
  tools/bench_turns.py --compare a.json b.json

Replies end in tool calls while the model makes them; a reply without one
gets a scripted follow-up user message, so every run has --turns turns. The
assistant messages are sent back as the server returned them (content,
reasoning_content when --send-reasoning, tool_calls).
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request

SYSTEM = (
    "You are a coding agent working in a Python repository. Use the tools to inspect the code before you "
    "answer: search for files, read the relevant ones, run the tests. Call one tool at a time and keep "
    "your text between tool calls short. When you have what you need, answer in a few sentences."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "search_files", "description": "Find files in the repository whose path or content matches a pattern.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "a regular expression"}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file of the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the test suite (or the tests matching a filter) and return its output.",
        "parameters": {"type": "object", "properties": {"filter": {"type": "string"}}}}},
]

FOLLOW_UPS = [
    "Thanks. Now check whether the parser handles time zone offsets like +05:30, and show me where.",
    "Run the tests for the date parser and tell me which ones fail, if any.",
    "Propose a fix for the failing case and explain it briefly.",
    "Is there any other caller of parse_date that would be affected by the fix? Search for it.",
    "Summarize everything you found in five bullet points.",
]

FILE_BODY = '''"""Date parsing helpers."""
import re
from datetime import datetime, timedelta, timezone

_ISO = re.compile(r"(\\d{4})-(\\d{2})-(\\d{2})(?:[T ](\\d{2}):(\\d{2})(?::(\\d{2}))?)?(Z|[+-]\\d{2}:?\\d{2})?$")


def parse_date(text, default_tz=None):
    """Parse an ISO-8601 date or date-time; returns an aware datetime when an offset is given."""
    m = _ISO.match(text.strip())
    if not m:
        raise ValueError(f"not an ISO date: {text!r}")
    y, mo, d, h, mi, s, off = m.groups()
    dt = datetime(int(y), int(mo), int(d), int(h or 0), int(mi or 0), int(s or 0))
    if off == "Z":
        return dt.replace(tzinfo=timezone.utc)
    if off:
        sign = 1 if off[0] == "+" else -1
        hh, mm = int(off[1:3]), int(off[-2:])
        return dt.replace(tzinfo=timezone(sign * timedelta(hours=hh)))  # minutes are dropped
    return dt.replace(tzinfo=default_tz) if default_tz else dt


def parse_range(text):
    start, _, end = text.partition("/")
    return parse_date(start), parse_date(end)
'''


def tool_result(name, args):
    """Deterministic results: a function of the call alone."""
    if name == "search_files":
        pat = str(args.get("pattern", ""))
        h = int(hashlib.sha1(pat.encode()).hexdigest(), 16)
        files = ["src/dates/parse.py", "src/dates/__init__.py", "src/report/build.py", "tests/test_parse.py",
                 "src/cli/main.py", "docs/dates.md"]
        hits = [files[i] for i in range(len(files)) if (h >> i) & 1] or files[:2]
        lines = [f"{f}:{10 + (h >> (3 * i)) % 80}: ... {pat} ..." for i, f in enumerate(hits)]
        return "\n".join(lines)
    if name == "read_file":
        path = str(args.get("path", ""))
        if path.endswith("parse.py"):
            return FILE_BODY
        if path.startswith("tests"):
            return ("from dates.parse import parse_date\n\n\ndef test_utc():\n    assert parse_date('2024-01-02T03:04Z').hour == 3\n\n\n"
                    "def test_offset_minutes():\n    d = parse_date('2024-01-02T03:04+05:30')\n"
                    "    assert d.utcoffset().total_seconds() == 19800\n")
        return f"# {path}\nfrom dates.parse import parse_date, parse_range\n\n\ndef main(argv):\n    return parse_range(argv[1])\n"
    if name == "run_tests":
        flt = str(args.get("filter", ""))
        return ("============ test session starts ============\ncollected 2 items" + (f" (filter: {flt})" if flt else "") +
                "\n\ntests/test_parse.py .F\n\n================= FAILURES =================\n"
                "___________ test_offset_minutes ___________\n    assert d.utcoffset().total_seconds() == 19800\n"
                "E   assert 18000.0 == 19800\n========= 1 failed, 1 passed in 0.03s =========")
    return f"unknown tool {name}"


def post(url, body, timeout=1800):
    return call(url, "/v1/chat/completions", body, timeout)


def call(url, path, body, timeout=1800):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run(a):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Find the function that parses dates in this repo, read it, and tell me whether "
                                        "it handles time zone offsets correctly."}]
    turns = []
    prompts = []
    follow = 0
    for t in range(a.turns):
        body = {"messages": msgs, "tools": TOOLS, "temperature": 0, "max_tokens": a.max_tokens, "stream": False}
        if a.model:
            body["model"] = a.model
        if a.logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 2
        if a.thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": a.thinking}
        if a.dump_prompts:
            ids = call(a.url, "/v1/tokenize", body)["tokens"]
            if prompts:
                prev = prompts[-1]
                n = 0
                while n < min(len(prev), len(ids)) and prev[n] == ids[n]:
                    n += 1
                show = lambda x: json.dumps(call(a.url, "/v1/detokenize", {"tokens": x})["text"])
                print(f"  turn {t}'s prompt ({len(prev)} tokens) is a prefix of turn {t + 1}'s for {n} tokens; "
                      f"turn {t}'s rest {show(prev[n:n + 12])}, turn {t + 1}'s {show(ids[max(0, n - 8):n + 24])} "
                      f"ids after turn {t}'s prompt: {ids[len(prev):len(prev) + 40] if n == len(prev) else ids[n:n + 40]}")
            prompts.append(ids)
        t0 = time.perf_counter()
        r = post(a.url, body)
        wall = (time.perf_counter() - t0) * 1000
        ch = r["choices"][0]
        m = ch["message"]
        u = r.get("usage", {})
        tm = r.get("timings") or {}
        pt = int(u.get("prompt_tokens", 0))
        cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        rec = {"turn": t + 1, "prompt_tokens": pt, "cached": cached, "prefilled": pt - cached,
               "ttft_ms": tm.get("prefill_ms"), "wall_ms": round(wall, 1), "completion_tokens": u.get("completion_tokens"),
               "finish": ch.get("finish_reason"), "content": m.get("content"), "reasoning": m.get("reasoning_content"),
               "tool_calls": m.get("tool_calls"), "logprobs": (ch.get("logprobs") or {}).get("content")}
        turns.append(rec)
        calls = m.get("tool_calls") or []
        print(f"turn {t + 1:2d}: prompt {pt:6d}  cached {cached:6d}  prefilled {pt - cached:6d}  "
              f"ttft {tm.get('prefill_ms', 0):9.1f} ms  reply {u.get('completion_tokens', 0):4d} tok  "
              f"{ch.get('finish_reason')}{'  calls ' + ','.join(c['function']['name'] for c in calls) if calls else ''}",
              flush=True)
        am = {"role": "assistant", "content": m.get("content") or ""}
        if a.send_reasoning and m.get("reasoning_content"):
            am["reasoning_content"] = m["reasoning_content"]
        if calls:
            am["tool_calls"] = calls
        msgs.append(am)
        if calls:
            for c in calls:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": tool_result(c["function"]["name"], args)})
        else:
            msgs.append({"role": "user", "content": FOLLOW_UPS[follow % len(FOLLOW_UPS)]})
            follow += 1
    later = [x for x in turns if x["turn"] >= 2]
    summary = {"turns": len(turns),
               "prefilled_turn2plus": sum(x["prefilled"] for x in later),
               "prompt_turn2plus": sum(x["prompt_tokens"] for x in later),
               "ttft_ms_turn2plus": round(sum(x["ttft_ms"] or 0 for x in later), 1),
               "mean_ttft_ms_turn2plus": round(sum(x["ttft_ms"] or 0 for x in later) / max(1, len(later)), 1)}
    print(json.dumps(summary))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "turns": turns}, f, indent=1)


def compare(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))
    same = True
    for x, y in zip(A["turns"], B["turns"]):
        keys = ("content", "reasoning", "tool_calls", "completion_tokens", "finish")
        diff = [k for k in keys if (x.get(k) if k != "tool_calls" else [(c["function"]["name"], c["function"]["arguments"]) for c in x.get(k) or []])
                != (y.get(k) if k != "tool_calls" else [(c["function"]["name"], c["function"]["arguments"]) for c in y.get(k) or []])]
        if diff and x.get("logprobs") and y.get("logprobs"):
            # Where the greedy replies part: the logprob deltas before it (summation
            # order moves them by ~1e-3; a wrong state by tenths) and the top-2 gap there.
            la, lb = x["logprobs"], y["logprobs"]
            k = 0
            while k < min(len(la), len(lb)) and la[k]["token"] == lb[k]["token"]:
                k += 1
            dmax = max([abs(la[i]["logprob"] - lb[i]["logprob"]) for i in range(k)] or [0])
            gap = lambda r: (r["top_logprobs"][0]["logprob"] - r["top_logprobs"][1]["logprob"]) if len(r.get("top_logprobs") or []) > 1 else None
            if k < min(len(la), len(lb)):
                print(f"    replies part at token {k} ({la[k]['token']!r} vs {lb[k]['token']!r}); max |delta logprob| before it "
                      f"{dmax:.4f}; top-2 gap there {gap(la[k])} vs {gap(lb[k])}")
        print(f"turn {x['turn']:2d}: prefilled {x['prefilled']:6d} -> {y['prefilled']:6d}   ttft {x['ttft_ms'] or 0:9.1f} -> "
              f"{y['ttft_ms'] or 0:9.1f} ms   {'same reply' if not diff else 'DIFFERENT ' + ','.join(diff)}")
        same = same and not diff
    sa, sb = A["summary"], B["summary"]
    print(f"turns 2+: prefilled {sa['prefilled_turn2plus']} -> {sb['prefilled_turn2plus']} tokens, "
          f"ttft {sa['ttft_ms_turn2plus']} -> {sb['ttft_ms_turn2plus']} ms; replies {'identical' if same else 'differ'}")
    return 0 if same else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="")
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--thinking", type=lambda s: s.lower() in ("1", "true", "yes"), default=None,
                   help="chat_template_kwargs.enable_thinking (default: the template's)")
    p.add_argument("--no-send-reasoning", dest="send_reasoning", action="store_false",
                   help="drop reasoning_content from the assistant messages sent back")
    p.add_argument("--dump-prompts", action="store_true",
                   help="tokenize each turn's prompt and show where it stops extending the previous one")
    p.add_argument("--logprobs", action="store_true", help="record each reply's token logprobs (top 2) for --compare")
    p.add_argument("--out", default="")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = p.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    run(a)


if __name__ == "__main__":
    main()
