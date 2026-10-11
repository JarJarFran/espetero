#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Speed of OpenAI-compatible LLM servers on one Mac (the README's table).

Every engine gets the same requests on the same model file: greedy decoding,
thinking off (chat_template_kwargs), streamed, with the token counts the
server reports. Workloads, after one unrecorded warm-up request:

  decode    8 prompts that ask for long answers, one at a time, up to 512 new
            tokens: time to first token and decode speed per request
            ((tokens - 1) / (last token time - first token time))
  code      6 programming prompts, one at a time, up to 512 new tokens: the
            same measures on code (where speculative decoding helps most)
  prefill   2 prompts of each --prefill-tokens size (default ~6.5K tokens; a
            different first line each, so no server can reuse a cached
            prefix), 16 new tokens: prompt tokens / time to first token
  context   1 prompt of each --context-tokens size followed by a request for a
            long answer, up to 256 new tokens: decode speed with that much
            context in the KV cache (and the prefill speed of the prompt)
  parallel  the 8 decode prompts sent at once, up to 256 new tokens each:
            all new tokens / wall time
  users     for each --users count N, N decode prompts sent at once, up to
            256 new tokens each: all new tokens / wall time, and each user's
            own decode speed and time to first token
  vision    6 synthetic 1024x768 PNG images (drawn here, under every engine's
            default pixel budget, so all of them see the same pixels), one at
            a time, up to 128 new tokens: seconds per image and the prompt
            tokens the server counted
  quality   20 short questions with one right answer (arithmetic, unit
            conversions, capitals) and 12 Python functions checked by unit
            tests (run in a sandbox without network): a sanity check that a
            fast engine still answers like the model

  python3 tools/bench_compare.py --url http://127.0.0.1:8080 --model NAME \\
      --engine espetero --version 0.1.0 --label "Qwen3.8-27B MLX 4-bit" \\
      --workloads decode,prefill,parallel,vision --out results.jsonl
  python3 tools/bench_compare.py --table results.jsonl      # markdown table
  python3 tools/bench_compare.py --report results.jsonl     # every measure

With --started-at (the time the server was launched, in seconds since the
epoch) the run first waits for the model to answer and records the load time.
--watch PIDFILE samples the memory of the server whose pid the file holds
(and of its child processes) until it exits; start it before the server so
the GPU baseline is taken without the model:

  python3 tools/bench_compare.py --watch server.pid --engine NAME --label L --out results.jsonl &

Only the standard library is used.
"""

import argparse
import base64
import ctypes
import ctypes.util
import json
import os
import platform
import random
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zlib

PROMPTS = [
    "Explain step by step how a transformer language model turns a prompt into generated text, "
    "for a programmer who knows Python but not machine learning.",
    "Write a Python module that implements an LRU cache with a time-to-live per entry, with "
    "docstrings and unit tests.",
    "Tell the history of the city of Málaga, from the Phoenicians to today.",
    "Compare TCP and QUIC in depth: handshakes, congestion control, head-of-line blocking and "
    "deployment.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the SQL transaction isolation levels with a concrete example of each anomaly they "
    "prevent.",
    "Give a detailed eight-week plan to learn linear algebra, with weekly goals and exercises.",
    "Explain how speculative decoding speeds up language model inference and when it does not "
    "help.",
]
LONG_ANSWER = " Answer in detail, in about 600 words."
CODE_PROMPTS = [
    "Write a Python class for a thread-safe bounded queue with blocking put and get, timeouts and "
    "unit tests using unittest.",
    "Write a TypeScript module that parses and evaluates arithmetic expressions with + - * / and "
    "parentheses, with a tokenizer, a recursive descent parser and tests.",
    "Write a Rust function that reads a CSV file and computes the mean, median and standard "
    "deviation of each numeric column, with error handling and tests.",
    "Refactor this into idiomatic Python with type hints and docstrings, then add pytest tests:\n\n"
    "def f(l):\n    r=[]\n    for i in range(len(l)):\n        if l[i]%2==0:\n            r.append(l[i]*l[i])\n"
    "    return r\n\ndef g(d):\n    k=list(d.keys())\n    k.sort()\n    o={}\n    for x in k:\n        o[x]=d[x]\n"
    "    return o",
    "Write a Go HTTP server with two JSON endpoints (create and list todo items) kept in memory "
    "behind a mutex, with handler tests using httptest.",
    "Write a SQL schema for a library (books, authors, members, loans) with constraints and "
    "indexes, then five queries with explanations: overdue loans, most borrowed books, members "
    "with no loans, authors by number of books, and loans per month.",
]
WORDS = ("the of and to in is was for on that with as by at from it an be this which or are "
         "have had not but were one all their there been has more when will would who so no "
         "time river city market engine harbour mountain light signal winter garden letter "
         "machine voice station paper window bridge morning number system water street music "
         "evening question history theory island forest silver road animal century science "
         "measure travel quiet bright sudden ancient careful distant gentle narrow simple").split()
TOKENS_PER_WORD = 1.125  # long_document() text on the Qwen3.5 tokenizer (1,068 tokens for 949 words)


# ---------------------------------------------------------------- HTTP
class Client:
    def __init__(self, url, model, api_key=None, timeout=1800):
        self.url = url.rstrip("/")
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key
        self.timeout = timeout

    def chat(self, content, max_tokens):
        """One streamed chat request; returns timings and the server's token counts."""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     self.headers, method="POST")
        t0 = time.perf_counter()
        first = last = None
        chunks = 0
        usage = None
        server = {}
        text = []
        with urllib.request.urlopen(req, timeout=self.timeout) as res:
            for raw in res:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    usage = ev["usage"]
                for k in ("timings", "stats", "metrics"):  # engine-specific counters (drafts, cache)
                    if isinstance(ev.get(k), dict):
                        server[k] = ev[k]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                    if piece:
                        now = time.perf_counter()
                        first = first if first is not None else now
                        last = now
                        chunks += 1
                        text.append(piece)
        t1 = time.perf_counter()
        if first is None:
            raise RuntimeError("no tokens streamed")
        out = {
            "ttft": first - t0,
            "total": t1 - t0,
            "gen_span": last - first,
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "chunks": chunks,
            "chars": sum(len(t) for t in text),
            "text": "".join(text),
        }
        if usage:
            extra = {k: v for k, v in usage.items() if k not in ("completion_tokens", "prompt_tokens", "total_tokens")}
            if extra:
                server["usage"] = extra
        if server:
            out["server"] = server
        if out["completion_tokens"] is None:  # no usage in the stream: count chunks (approximate)
            out["completion_tokens"] = chunks
            out["approx_tokens"] = True
        n = out["completion_tokens"]
        out["decode_tps"] = (n - 1) / out["gen_span"] if n > 1 and out["gen_span"] > 0 else None
        return out


def drop_text(r):
    """A result row without the reply text (kept only where it is checked)."""
    return {k: v for k, v in r.items() if k != "text"}


# ---------------------------------------------------------------- inputs
def long_document(seed, n_words=6000):
    rng = random.Random(seed)
    sentences, words = [], 0
    while words < n_words:
        k = rng.randint(8, 22)
        s = " ".join(rng.choice(WORDS) for _ in range(k))
        sentences.append(s[0].upper() + s[1:] + ".")
        words += k
    paragraphs = [" ".join(sentences[i:i + 6]) for i in range(0, len(sentences), 6)]
    return "\n\n".join(paragraphs)


def nonce_line():
    return f"Document {random.Random(time.time_ns()).randrange(1 << 30):x}, read carefully.\n\n"


def sized_document(seed, tokens):
    return long_document(seed, max(50, int(tokens / TOKENS_PER_WORD)))


def png(width, height, pixel):
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixel(x, y))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


def synthetic_image(seed, width=1024, height=768):
    """A sky-to-ground gradient with a sun, hills and coloured boxes: something to describe."""
    rng = random.Random(seed)
    sky = [rng.randint(60, 140), rng.randint(110, 190), rng.randint(190, 255)]
    ground = [rng.randint(40, 110), rng.randint(90, 160), rng.randint(30, 80)]
    sun = (rng.randint(100, width - 100), rng.randint(60, 220), rng.randint(40, 80))
    hills = [(rng.randint(0, width), rng.randint(380, 520), rng.randint(150, 330)) for _ in range(3)]
    boxes = [(rng.randint(0, width - 160), rng.randint(450, height - 120), rng.randint(60, 160),
              rng.randint(50, 110), (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
             for _ in range(4)]
    def pixel(x, y):
        for bx, by, bw, bh, col in boxes:
            if bx <= x < bx + bw and by <= y < by + bh:
                return col
        if (x - sun[0]) ** 2 + (y - sun[1]) ** 2 < sun[2] ** 2:
            return (250, 220, 90)
        horizon = min(hy + ((x - hx) / hr) ** 2 * 60 for hx, hy, hr in hills)
        if y > horizon:
            t = (y - horizon) / height
            return tuple(max(0, min(255, int(c * (1 - t)))) for c in ground)
        t = y / height
        return tuple(int(c * (1 - 0.4 * t)) for c in sky)
    return png(width, height, pixel)


# ---------------------------------------------------------------- quality checks
CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
            ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Norway", "Oslo")]


def short_questions():
    """20 questions with one right answer: (question, answer, kind)."""
    rng = random.Random(2026)
    qs = []
    for _ in range(6):
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
        qs.append((f"What is {a} + {b}?", str(a + b), "number"))
    for _ in range(4):
        a, b = rng.randint(5000, 9999), rng.randint(1000, 4999)
        qs.append((f"What is {a} - {b}?", str(a - b), "number"))
    for _ in range(4):
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        qs.append((f"What is {a} times {b}?", str(a * b), "number"))
    h, w, km = rng.randint(3, 19), rng.randint(2, 9), rng.randint(2, 40)
    qs += [(f"How many minutes are there in {h} hours?", str(h * 60), "number"),
           (f"How many days are there in {w} weeks?", str(w * 7), "number"),
           (f"How many meters are there in {km} kilometers?", str(km * 1000), "number")]
    for country, city in rng.sample(CAPITALS, 3):
        qs.append((f"What is the capital of {country}?", city, "word"))
    return qs


# (function name, task, tests): the reply's code is run with the tests appended.
CODE_TASKS = [
    ("is_palindrome", "is_palindrome(s) that returns True when the string reads the same backwards, ignoring case "
     "and every character that is not a letter or a digit",
     "assert is_palindrome('A man, a plan, a canal: Panama')\nassert not is_palindrome('race a car')\n"
     "assert is_palindrome('')"),
    ("fizzbuzz", "fizzbuzz(n) that returns a list of strings for 1..n: 'Fizz' for multiples of 3, 'Buzz' for "
     "multiples of 5, 'FizzBuzz' for both, otherwise the number",
     "assert fizzbuzz(15)[-1] == 'FizzBuzz' and fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']\n"
     "assert fizzbuzz(0) == []"),
    ("rle_encode", "rle_encode(s) that run-length encodes a string as each character followed by its count, "
     "e.g. 'aaabcc' -> 'a3b1c2'",
     "assert rle_encode('aaabccdddd') == 'a3b1c2d4'\nassert rle_encode('') == ''\nassert rle_encode('x') == 'x1'"),
    ("to_roman", "to_roman(n) that converts an integer from 1 to 3999 to a Roman numeral",
     "assert to_roman(1994) == 'MCMXCIV'\nassert to_roman(4) == 'IV'\nassert to_roman(3999) == 'MMMCMXCIX'"),
    ("merge_sorted", "merge_sorted(a, b) that merges two sorted lists into one sorted list without calling "
     "sort or sorted",
     "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_sorted([], [1]) == [1]\n"
     "assert merge_sorted([1, 1], [1]) == [1, 1, 1]"),
    ("flatten", "flatten(xs) that flattens a list nested to any depth",
     "assert flatten([1, [2, [3, [4]], 5]]) == [1, 2, 3, 4, 5]\nassert flatten([]) == []\n"
     "assert flatten([[[]]]) == []"),
    ("word_freq", "word_freq(text) that returns a dict counting the lowercase words of a text, where a word is "
     "a run of letters",
     "assert word_freq('The cat and the hat.') == {'the': 2, 'cat': 1, 'and': 1, 'hat': 1}\n"
     "assert word_freq('') == {}"),
    ("transpose", "transpose(m) that transposes a matrix given as a list of rows",
     "assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]\nassert transpose([]) == []"),
    ("valid_parens", "valid_parens(s) that returns True when the brackets ()[]{} in s are balanced and properly "
     "nested",
     "assert valid_parens('([]{})')\nassert not valid_parens('([)]')\nassert valid_parens('')\n"
     "assert not valid_parens('((')"),
    ("binary_search", "binary_search(xs, x) that returns the index of x in the sorted list xs, or -1",
     "assert binary_search([1, 3, 5, 7, 9], 7) == 3\nassert binary_search([1, 3, 5, 7, 9], 2) == -1\n"
     "assert binary_search([], 1) == -1"),
    ("second_largest", "second_largest(xs) that returns the second largest distinct value of a list, or None "
     "when there is none",
     "assert second_largest([4, 1, 4, 3]) == 3\nassert second_largest([5, 5]) is None\n"
     "assert second_largest([]) is None"),
    ("parse_duration", "parse_duration(s) that converts a duration like '1h30m15s' (each part optional) to "
     "seconds",
     "assert parse_duration('1h30m15s') == 5415\nassert parse_duration('45s') == 45\n"
     "assert parse_duration('2h') == 7200"),
]

SANDBOX = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{tmp}") (literal "/dev/null"))
"""


def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def grade_short(reply, answer, kind):
    reply = strip_thinking(reply)
    if kind == "number":
        nums = re.findall(r"-?\d[\d,]*", reply)
        return bool(nums) and nums[0].replace(",", "") == answer
    return answer.lower() in reply.lower()


def extract_code(reply):
    reply = strip_thinking(reply)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, flags=re.S)
    return max(blocks, key=len) if blocks else reply


def run_tests(code, tests, timeout=20):
    """Runs code + tests in a temporary directory (no network, writes only there); True if they pass."""
    tmp = tempfile.mkdtemp(prefix="bench-quality-")
    try:
        path = os.path.join(tmp, "check.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code + "\n\n" + tests + "\nprint('PASS')\n")
        cmd = [sys.executable, "-I", path]
        if shutil.which("sandbox-exec"):
            cmd = ["sandbox-exec", "-p", SANDBOX.format(tmp=os.path.realpath(tmp))] + cmd
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0 and "PASS" in r.stdout
    except subprocess.TimeoutExpired:
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- workloads
def run_decode(c, rows, a):
    for i, p in enumerate(PROMPTS[:a.decode_prompts]):
        r = drop_text(c.chat(p + LONG_ANSWER, 512))
        rows.append({"workload": "decode", "i": i, **r})
        print(f"  decode {i}: {r['completion_tokens']} tok, ttft {r['ttft']:.2f} s, "
              f"{r['decode_tps'] or 0:.1f} tok/s", file=sys.stderr)


def run_code(c, rows, a):
    for i, p in enumerate(CODE_PROMPTS):
        r = drop_text(c.chat(p, 512))
        rows.append({"workload": "code", "i": i, **r})
        print(f"  code {i}: {r['completion_tokens']} tok, ttft {r['ttft']:.2f} s, "
              f"{r['decode_tps'] or 0:.1f} tok/s", file=sys.stderr)


def run_prefill(c, rows, a):
    for size in a.prefill_tokens:
        for i in range(2):
            doc = long_document(1234 + i) if size == 6500 else sized_document(1234 + i, size)
            r = drop_text(c.chat(nonce_line() + doc + "\n\nIn one sentence, what is this document about?", 16))
            r["prefill_tps"] = r["prompt_tokens"] / r["ttft"] if r.get("prompt_tokens") else None
            rows.append({"workload": "prefill", "target_tokens": size, "i": i, **r})
            print(f"  prefill ~{size}: {r.get('prompt_tokens')} tok in {r['ttft']:.2f} s", file=sys.stderr)


def run_context(c, rows, a):
    for size in a.context_tokens:
        r = drop_text(c.chat(nonce_line() + sized_document(4321, size) +
                             "\n\nWrite a long, detailed essay inspired by this document, in about 600 words.", 256))
        r["prefill_tps"] = r["prompt_tokens"] / r["ttft"] if r.get("prompt_tokens") else None
        rows.append({"workload": "context", "target_tokens": size, **r})
        print(f"  context ~{size}: {r.get('prompt_tokens')} prompt tok, ttft {r['ttft']:.2f} s, "
              f"decode {r['decode_tps'] or 0:.1f} tok/s", file=sys.stderr)


def concurrent(c, n, prompts, max_tokens):
    results = [None] * n
    def one(i):
        try:
            results[i] = drop_text(c.chat(prompts[i % len(prompts)], max_tokens))
        except Exception as e:  # recorded, not raised: one failure must not hide the others
            results[i] = {"error": str(e)}
    t0 = time.perf_counter()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, time.perf_counter() - t0


def run_parallel(c, rows, a, n=8):
    results, wall = concurrent(c, n, [p + LONG_ANSWER for p in PROMPTS], 256)
    ok = [r for r in results if r and "error" not in r]
    total = sum(r["completion_tokens"] for r in ok)
    rows.append({"workload": "parallel", "n": n, "ok": len(ok), "wall": wall, "completion_tokens": total,
                 "aggregate_tps": total / wall, "errors": [r["error"] for r in results if r and "error" in r]})
    print(f"  parallel x{n}: {total} tok in {wall:.1f} s = {total / wall:.1f} tok/s ({len(ok)} ok)", file=sys.stderr)


def run_users(c, rows, a):
    # A different first line per user and level, so no level reuses another's prompt cache.
    for n in a.users:
        prompts = [f"(User {n}.{i}) " + PROMPTS[i % len(PROMPTS)] + LONG_ANSWER for i in range(n)]
        results, wall = concurrent(c, n, prompts, 256)
        ok = [r for r in results if r and "error" not in r]
        total = sum(r["completion_tokens"] for r in ok)
        med = lambda f: statistics.median([r[f] for r in ok if r.get(f) is not None]) if ok else None
        rows.append({"workload": "users", "n": n, "ok": len(ok), "wall": wall, "completion_tokens": total,
                     "aggregate_tps": total / wall, "user_tps": med("decode_tps"), "ttft": med("ttft"),
                     "ttft_max": max((r["ttft"] for r in ok), default=None),
                     "errors": [r["error"] for r in results if r and "error" in r]})
        print(f"  users x{n}: {total / wall:.1f} tok/s total, {med('decode_tps') or 0:.1f} per user, "
              f"ttft {med('ttft') or 0:.2f} s ({len(ok)} ok)", file=sys.stderr)


def run_vision(c, rows, a, n=6):
    for i in range(n):
        url = "data:image/png;base64," + base64.b64encode(synthetic_image(100 + i)).decode()
        content = [{"type": "image_url", "image_url": {"url": url}},
                   {"type": "text", "text": "Describe this image in detail."}]
        r = drop_text(c.chat(content, 128))
        rows.append({"workload": "vision", "i": i, **r})
        print(f"  vision {i}: {r['total']:.2f} s, ttft {r['ttft']:.2f} s, prompt {r.get('prompt_tokens')} tok",
              file=sys.stderr)


def run_quality(c, rows, a):
    right = 0
    qs = short_questions()
    for i, (q, answer, kind) in enumerate(qs):
        r = c.chat(q + " Reply with only the answer.", 48)
        ok = grade_short(r["text"], answer, kind)
        right += ok
        rows.append({"workload": "quality", "kind": "short", "i": i, "ok": ok, "answer": answer,
                     "reply": r["text"][:200], **drop_text(r)})
    passed = 0
    for i, (name, task, tests) in enumerate(CODE_TASKS):
        r = c.chat(f"Write a Python function {task}. Use only the standard library. Reply with only the code "
                   "in one ```python block.", 512)
        ok = run_tests(extract_code(r["text"]), tests)
        passed += ok
        rows.append({"workload": "quality", "kind": "code", "i": i, "name": name, "ok": ok,
                     "reply": r["text"][:4000], **drop_text(r)})
    print(f"  quality: short {right}/{len(qs)}, code {passed}/{len(CODE_TASKS)}", file=sys.stderr)


WORKLOADS = {"decode": run_decode, "code": run_code, "prefill": run_prefill, "context": run_context,
             "parallel": run_parallel, "users": run_users, "vision": run_vision, "quality": run_quality}


# ---------------------------------------------------------------- memory (macOS)
class RusageInfoV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size phys_footprint "
        "proc_start_abstime proc_exit_abstime child_user_time child_system_time child_pkg_idle_wkups "
        "child_interrupt_wkups child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten "
        "cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background cpu_time_qos_utility "
        "cpu_time_qos_legacy cpu_time_qos_user_initiated cpu_time_qos_user_interactive billed_system_time "
        "serviced_system_time logical_writes lifetime_max_phys_footprint instructions cycles billed_energy "
        "serviced_energy interval_max_phys_footprint runnable_time").split()]


_libproc = None


def libproc():
    global _libproc
    if _libproc is None:
        _libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
    return _libproc


def process_tree(pid):
    pids, todo = [], [pid]
    buf = (ctypes.c_int * 4096)()
    while todo:
        p = todo.pop()
        pids.append(p)
        n = libproc().proc_listchildpids(p, buf, ctypes.sizeof(buf))
        todo.extend(buf[i] for i in range(max(0, n)) if buf[i] > 0)
    return pids


def rusage(pid):
    r = RusageInfoV4()
    return r if libproc().proc_pid_rusage(pid, 4, ctypes.byref(r)) == 0 else None


def gpu_in_use():
    """The GPU driver's "In use system memory" in bytes (all processes), or None."""
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                             capture_output=True, text=True, timeout=10).stdout
        m = re.search(r'"In use system memory"=(\d+)', out)
        return int(m.group(1)) if m else None
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_times():
    """{pid: [name, accumulated GPU nanoseconds]} of every process with a Metal client."""
    try:
        out = subprocess.run(["ioreg", "-r", "-c", "AGXDeviceUserClient", "-w", "0"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    res = {}
    for block in out.split("+-o ")[1:]:
        m = re.search(r'"IOUserClientCreator" = "pid (\d+), ([^"]*)"', block)
        if m:
            entry = res.setdefault(int(m.group(1)), [m.group(2), 0])
            entry[1] += sum(int(x) for x in re.findall(r'"accumulatedGPUTime"=(\d+)', block))
    return res


def watch_memory(a):
    """Samples the memory of the process in a.watch (a pid file) and its children until it exits."""
    gpu0 = gpu_in_use()
    gpu_first = {p: v[1] for p, v in gpu_times().items()}  # GPU time per process before the server starts
    gpu_last, names, ours = dict(gpu_first), {}, set()
    t_start = time.time()
    pid = None
    while pid is None:
        try:
            pid = int(open(a.watch).read().strip())
        except (OSError, ValueError):
            if time.time() - t_start > 600:
                sys.exit("no pid in " + a.watch)
            time.sleep(0.2)
    peak = {"footprint": 0, "resident": 0, "gpu": 0, "lifetime_max_footprint": 0}
    last_gpu = 0.0
    timeline = []
    while True:
        tree = process_tree(pid)
        infos = [x for x in (rusage(p) for p in tree) if x]
        if not infos:
            break
        ours.update(tree)
        fp = sum(x.phys_footprint for x in infos)
        rss = sum(x.resident_size for x in infos)
        peak["footprint"] = max(peak["footprint"], fp)
        peak["resident"] = max(peak["resident"], rss)
        peak["lifetime_max_footprint"] = max(peak["lifetime_max_footprint"],
                                             sum(x.lifetime_max_phys_footprint for x in infos))
        now = time.time()
        if now - last_gpu >= 3.0:  # ioreg is a process: sampled less often
            for p, (name, ns) in gpu_times().items():
                names[p] = name
                gpu_first.setdefault(p, 0)
                gpu_last[p] = max(gpu_last.get(p, 0), ns)
            g = gpu_in_use()
            if g is not None and gpu0 is not None:
                peak["gpu"] = max(peak["gpu"], g - gpu0)
            last_gpu = now
            timeline.append([round(now - t_start, 1), round(fp / 2**30, 2), round(rss / 2**30, 2),
                             None if g is None or gpu0 is None else round((g - gpu0) / 2**30, 2)])
        time.sleep(0.5)
    spent = {p: (gpu_last.get(p, 0) - gpu_first.get(p, 0)) / 1e9 for p in gpu_last}
    foreign = {}
    for p, sec in spent.items():
        if p not in ours and sec > 0.05:
            foreign[names.get(p, str(p))] = foreign.get(names.get(p, str(p)), 0) + sec
    wall = time.time() - t_start
    other = max([v for k, v in foreign.items() if k != "WindowServer"], default=0)  # one other GPU job
    screen = foreign.get("WindowServer", 0)  # the desktop in use (drawing the screen)
    row = {"workload": "memory", "engine": a.engine, "version": a.version, "label": a.label,
           "wall_s": wall, "gpu_busy_s": sum(spent.get(p, 0) for p in ours),
           "foreign_gpu_s": dict(sorted(foreign.items(), key=lambda kv: -kv[1])[:6]),
           "contended": other > 0.03 * wall or screen > 0.05 * wall,
           "peak_footprint_gb": peak["footprint"] / 2**30, "peak_resident_gb": peak["resident"] / 2**30,
           "lifetime_max_footprint_gb": peak["lifetime_max_footprint"] / 2**30,
           "peak_gpu_in_use_gb": peak["gpu"] / 2**30 if gpu0 is not None else None,
           "gpu_baseline_gb": gpu0 / 2**30 if gpu0 is not None else None, "timeline": timeline,
           "time": time.strftime("%Y-%m-%d %H:%M")}
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"memory: peak footprint {row['peak_footprint_gb']:.1f} GB, resident {row['peak_resident_gb']:.1f} GB, "
          f"GPU in use +{row['peak_gpu_in_use_gb'] or 0:.1f} GB; GPU busy {row['gpu_busy_s']:.0f} s of {wall:.0f} s; "
          f"other GPU users {row['foreign_gpu_s']}{' (CONTENDED)' if row['contended'] else ''}", file=sys.stderr)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wait_ready(c, started_at, server_pid=None, timeout=3600):
    """Seconds from the server's launch until it answers a request."""
    failures = 0  # answers that are errors other than "loading" (503)
    while True:
        try:
            c.chat("Say hello in five words.", 8)
            return time.time() - started_at
        except Exception as e:  # loading, port not open yet
            if server_pid and not alive(server_pid):
                raise RuntimeError("the server exited before it answered") from e
            if isinstance(e, urllib.error.HTTPError) and e.code != 503:
                failures += 1
                if failures >= 20:  # it is up but cannot answer this request
                    body = ""
                    try:
                        body = e.read().decode("utf-8", "replace")[:300]
                    except Exception:
                        pass
                    raise RuntimeError(f"the server answers HTTP {e.code}: {body}") from e
            if time.time() - started_at > timeout:
                raise
            time.sleep(0.25)


# ---------------------------------------------------------------- tables
def summarize(rows):
    """{(label, engine): metrics} with medians over the requests of each workload."""
    out = {}
    for r in rows:
        key = (r["label"], r["engine"])
        out.setdefault(key, {"version": r.get("version", ""), "rows": []})["rows"].append(r)
    table = {}
    for key, v in out.items():
        rs = v["rows"]
        def med(workload, field, **match):
            xs = [r[field] for r in rs if r["workload"] == workload and r.get(field) is not None
                  and all(r.get(k) == x for k, x in match.items())]
            return statistics.median(xs) if xs else None
        def biggest(field):
            xs = [r[field] for r in rs if r["workload"] == "memory" and r.get(field) is not None]
            return max(xs) if xs else None
        m = {
            "version": v["version"],
            "decode_tps": med("decode", "decode_tps"),
            "ttft": med("decode", "ttft"),
            "code_tps": med("code", "decode_tps"),
            "prefill_tps": med("prefill", "prefill_tps", target_tokens=6500) or med("prefill", "prefill_tps"),
            "parallel_tps": med("parallel", "aggregate_tps"),
            "image_s": med("vision", "total"),
            "image_tokens": med("vision", "prompt_tokens"),
            "load_s": med("load", "load_s"),
            "memory_gb": biggest("peak_gpu_in_use_gb"),
            "footprint_gb": biggest("lifetime_max_footprint_gb"),
            "resident_gb": biggest("peak_resident_gb"),
        }
        for size in sorted({r["target_tokens"] for r in rs if r["workload"] == "prefill" and "target_tokens" in r}):
            m[f"prefill_tps@{size}"] = med("prefill", "prefill_tps", target_tokens=size)
        for size in sorted({r["target_tokens"] for r in rs if r["workload"] == "context" and "target_tokens" in r}):
            m[f"context_tps@{size}"] = med("context", "decode_tps", target_tokens=size)
            if m.get(f"prefill_tps@{size}") is None:  # the context prompt's own prefill
                m[f"prefill_tps@{size}"] = med("context", "prefill_tps", target_tokens=size)
        for n in sorted({r["n"] for r in rs if r["workload"] == "users" and "n" in r}):
            m[f"users_tps@{n}"] = med("users", "aggregate_tps", n=n)
            m[f"user_tps@{n}"] = med("users", "user_tps", n=n)
            m[f"users_ttft@{n}"] = med("users", "ttft", n=n)
        q = [r for r in rs if r["workload"] == "quality" and "kind" in r]
        if q:
            m["quality_short"] = f"{sum(r['ok'] for r in q if r['kind'] == 'short')}/{sum(r['kind'] == 'short' for r in q)}"
            m["quality_code"] = f"{sum(r['ok'] for r in q if r['kind'] == 'code')}/{sum(r['kind'] == 'code' for r in q)}"
        table[key] = m
    return table


def load_rows(paths):
    return [json.loads(l) for p in paths for l in open(p, encoding="utf-8") if l.strip()]


def fmt(x, d=1):
    return "–" if x is None else f"{x:.{d}f}"


def print_table(paths):
    table = summarize(load_rows(paths))
    print("| Model | Engine | Decode tok/s | Time to first token (s) | Prefill tok/s (6.5K) | 8 users, tok/s | Image (s) |")
    print("|:--|:--|--:|--:|--:|--:|--:|")
    for (label, engine), m in table.items():
        img = fmt(m["image_s"], 2) if m["image_s"] is None else f"{m['image_s']:.2f} ({m['image_tokens'] or '?'} tok)"
        name = f"{engine} {m['version']}".strip()
        print(f"| {label} | {name} | {fmt(m['decode_tps'])} | {fmt(m['ttft'], 2)} | {fmt(m['prefill_tps'], 0)} | "
              f"{fmt(m['parallel_tps'])} | {img} |")


def print_report(paths):
    """Every measure, one column per engine, one table per model."""
    table = summarize(load_rows(paths))
    labels = list(dict.fromkeys(label for label, _ in table))
    for label in labels:
        engines = [(e, m) for (l, e), m in table.items() if l == label]
        keys = []
        for _, m in engines:
            keys += [k for k in m if k != "version" and k not in keys and m[k] is not None]
        print(f"\n### {label}\n")
        print("| Measure | " + " | ".join(f"{e} {m['version']}".strip() for e, m in engines) + " |")
        print("|:--|" + "--:|" * len(engines))
        for k in keys:
            cells = []
            for _, m in engines:
                x = m.get(k)
                cells.append(x if isinstance(x, str) else fmt(x, 2 if "ttft" in k or k.endswith("_s") else 1))
            print(f"| {k} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url")
    ap.add_argument("--model", help="model name sent in requests")
    ap.add_argument("--engine")
    ap.add_argument("--version", default="")
    ap.add_argument("--label", help="model/file label for the table")
    ap.add_argument("--api-key", help="bearer key (default: $BENCH_API_KEY)")
    ap.add_argument("--workloads", default="decode,prefill,parallel,vision")
    ap.add_argument("--prefill-tokens", default="6500", help="prompt sizes of the prefill workload")
    ap.add_argument("--context-tokens", default="1024,8192,32768", help="context sizes of the context workload")
    ap.add_argument("--users", default="1,4,8,16", help="concurrent users of the users workload")
    ap.add_argument("--decode-prompts", type=int, default=len(PROMPTS), help="prompts of the decode workload")
    ap.add_argument("--started-at", type=float, help="server launch time (epoch seconds): record the load time")
    ap.add_argument("--server-pid", type=int, help="with --started-at: stop waiting if this process exits")
    ap.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE", help="extra field on every row")
    ap.add_argument("--watch", metavar="PIDFILE", help="sample the memory of this server until it exits")
    ap.add_argument("--out")
    ap.add_argument("--table", nargs="+", metavar="JSONL")
    ap.add_argument("--report", nargs="+", metavar="JSONL")
    a = ap.parse_args()
    if a.table:
        print_table(a.table)
        return
    if a.report:
        print_report(a.report)
        return
    if a.watch:
        if not (a.engine and a.label and a.out):
            ap.error("--watch needs --engine, --label and --out")
        watch_memory(a)
        return
    if not (a.url and a.model and a.engine and a.label and a.out):
        ap.error("--url, --model, --engine, --label and --out are required to run")
    a.prefill_tokens = [int(x) for x in a.prefill_tokens.split(",") if x]
    a.context_tokens = [int(x) for x in a.context_tokens.split(",") if x]
    a.users = [int(x) for x in a.users.split(",") if x]
    c = Client(a.url, a.model, a.api_key or os.environ.get("BENCH_API_KEY"))
    work = a.workloads.split(",")
    rows = []
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout or 0) >> 30
    meta = {"engine": a.engine, "version": a.version, "label": a.label, "model": a.model, "chip": chip,
            "memory_gb": mem, "macos": platform.mac_ver()[0], "time": time.strftime("%Y-%m-%d %H:%M")}
    for kv in a.meta:
        k, _, v = kv.partition("=")
        meta[k] = v
    saved = [0]
    def save():  # appends the rows not written yet: a run cut short keeps its finished workloads
        with open(a.out, "a", encoding="utf-8") as f:
            for r in rows[saved[0]:]:
                f.write(json.dumps({**meta, **r}, ensure_ascii=False) + "\n")
        saved[0] = len(rows)
    if a.started_at:
        try:
            load_s = wait_ready(c, a.started_at, a.server_pid)
        except Exception as e:  # recorded: an engine that cannot load this model is a result too
            rows.append({"workload": "load", "error": str(e)[:500]})
            save()
            sys.exit(f"{a.engine} on {a.label}: not ready: {e}")
        rows.append({"workload": "load", "load_s": load_s})
        print(f"{a.engine} {a.version} on {a.label}: ready {load_s:.1f} s after launch", file=sys.stderr)
    print(f"{a.engine} {a.version} on {a.label}: warm-up", file=sys.stderr)
    c.chat("Say hello in five words.", 16)
    if "vision" in work:
        try:
            c.chat([{"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                     base64.b64encode(synthetic_image(7)).decode()}}, {"type": "text", "text": "What is this?"}], 8)
        except Exception as e:  # no images on this engine or model: recorded, the rest still runs
            rows.append({"workload": "vision", "error": str(e)[:500]})
            work = [w for w in work if w != "vision"]
            print(f"  vision: not available: {e}", file=sys.stderr)
    for w in work:
        try:
            WORKLOADS[w](c, rows, a)
        except Exception as e:  # an engine without images or long contexts: recorded, the rest still runs
            rows.append({"workload": w, "error": str(e)[:500]})
            print(f"  {w}: failed: {e}", file=sys.stderr)
        save()
    save()
    print_table([a.out])


if __name__ == "__main__":
    main()
