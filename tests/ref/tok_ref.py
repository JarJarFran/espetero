#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Reference data for tests/tok_test.cpp, computed with Hugging Face `tokenizers`.

    python3 tests/ref/tok_ref.py OUT_DIR

Writes OUT_DIR/tok_ref.bin (little-endian binary, format below) and two synthetic
tokenizer directories under OUT_DIR/variants/ that exercise code paths the real
models do not use (Llama-3 regex + ignore_merges; added tokens with
lstrip/rstrip/normalized/single_word flags, tokens inside the vocabulary,
redefinitions).

The corpus (>= 20,000 strings, fixed seed): English prose and code from the
Python standard library, JSON, ~40 languages (also in NFD/NFKD), emoji and ZWJ
sequences, whitespace torture (mixed \\n \\r \\t \\v \\f and Unicode spaces), digit
runs in many scripts, contractions in all cases, chat transcripts with special
tokens (also malformed/overlapping ones), random code points from all planes,
NFC stress (combining marks, Hangul jamo, composition exclusions, code points the
reference normalizer does not know), and very long words.

Format ("str" = u32 length + bytes, "ids" = u32 count + i32 values):
  magic "TOKREF03"
  u32 n_tokenizers; per tokenizer:
    str name, str kind ("hf" | "gguf"), str path,
    i32 eos, i32 bos, i32 pad, i32 vocab_size, str chat_template,
    ids special_ids, u32 n_added + n_added * (str content, i32 id)
  u32 n_cases; per case: str text; per tokenizer:
    u8 flags (1: nospecial ids equal default ids, 2: decode equals text,
              4: decode_skip equals decode), ids default, [ids nospecial],
    [str decode], [str decode_skip]
  u32 n_decode_cases; per case: u32 tokenizer, u8 skip_special, ids, str expected
  str bench_text; per tokenizer: ids bench
Decode strings are the reference's `decode(ids, skip_special_tokens=...)` (UTF-8,
invalid sequences already replaced by U+FFFD).
"""
import argparse
import inspect
import glob
import json
import mmap
import os
import random
import re
import struct
import sys
import unicodedata

from tokenizers import Tokenizer

def _hub(repo, pattern=""):
    """Newest snapshot of a Hugging Face repo in the local hub cache (or a file in it), else None."""
    cache = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    snaps = sorted(glob.glob(os.path.join(cache, "models--" + repo.replace("/", "--"), "snapshots", "*")),
                   key=os.path.getmtime)
    if not snaps:
        return None
    if not pattern:
        return snaps[-1]
    files = sorted(glob.glob(os.path.join(snaps[-1], pattern)))
    return files[0] if files else None


# Tokenizers to compare against (TOK_REF_* overrides a path).
HF_QWEN35 = os.environ.get("TOK_REF_QWEN35") or _hub("mlx-community/Qwen3.5-0.8B-4bit")
HF_QWEN3 = os.environ.get("TOK_REF_QWEN3") or _hub("Qwen/Qwen3-4B")
HF_QWEN38 = os.environ.get("TOK_REF_QWEN38") or _hub("mlx-community/Qwen3.8-27B-4bit")
GGUF_QWEN35 = os.environ.get("TOK_REF_GGUF") or _hub("unsloth/Qwen3.5-0.8B-GGUF", "*Q4_K_M.gguf")
# Gemma 4: SentencePiece-style BPE with byte fallback; compared when cached.
HF_GEMMA4 = os.environ.get("TOK_REF_GEMMA4") or _hub("mlx-community/gemma-4-31b-it-4bit")
GGUF_GEMMA4 = os.environ.get("TOK_REF_GGUF_GEMMA4") or _hub("unsloth/gemma-4-26B-A4B-it-GGUF", "*Q4_K_M.gguf")
LLAMA3_RE = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
             r"\s*[\r\n]+|\s+(?!\S)|\s+")
HERE = os.path.dirname(os.path.abspath(__file__))
UNICODE_DATA_H = os.path.join(HERE, "..", "..", "src", "tok", "unicode_data.h")

R = random.Random(20260924)
C = chr


# ============================================================================ GGUF metadata
def read_gguf_tokenizer(path):
    with open(path, "rb") as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmts = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    off = 24
    _, _, _, n_kv = struct.unpack_from("<4sIQQ", m, 0)

    def rstr(o):
        n, = struct.unpack_from("<Q", m, o)
        return m[o + 8:o + 8 + n].decode("utf-8"), o + 8 + n

    def rval(t, o):
        if t == 8:
            return rstr(o)
        if t == 9:
            et, n = struct.unpack_from("<IQ", m, o)
            o += 12
            out = []
            for _ in range(n):
                v, o = rval(et, o)
                out.append(v)
            return out, o
        return struct.unpack_from("<" + fmts[t], m, o)[0], o + sizes[t]

    kv = {}
    for _ in range(n_kv):
        k, off = rstr(off)
        t, = struct.unpack_from("<I", m, off)
        off += 4
        if k.startswith("tokenizer."):
            kv[k], off = rval(t, off)
        else:
            _, off = rval(t, off)
    return kv


# ============================================================================ corpus
def stdlib_dir():
    return os.path.dirname(os.__file__)


def english_docs():
    """Paragraphs of English prose: docstrings of standard library modules."""
    import importlib
    names = ["os", "sys", "json", "re", "argparse", "collections", "itertools", "functools", "pathlib", "typing",
             "asyncio", "subprocess", "threading", "logging", "unittest", "datetime", "decimal", "fractions", "random",
             "statistics", "string", "textwrap", "difflib", "csv", "email", "http.client", "urllib.parse", "socket",
             "ssl", "sqlite3", "zipfile", "tarfile", "shutil", "tempfile", "glob", "fnmatch", "pickle", "copy",
             "heapq", "bisect", "array", "struct", "codecs", "unicodedata", "locale", "gettext", "inspect", "dis",
             "ast", "tokenize", "pdb", "timeit", "trace", "contextlib", "abc", "dataclasses", "enum", "numbers",
             "math", "cmath", "operator", "queue", "sched", "selectors", "signal", "mmap", "hashlib", "hmac",
             "secrets", "base64", "binascii", "html", "html.parser", "xml.dom.minidom", "configparser", "calendar",
             "time", "weakref", "gc", "io", "platform", "warnings", "traceback", "shlex", "optparse", "getopt"]
    paras = []
    for name in names:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        objs = [mod] + [v for k, v in sorted(vars(mod).items()) if not k.startswith("_") and
                        (inspect.isclass(v) or inspect.isfunction(v))]
        for o in objs:
            doc = inspect.getdoc(o)
            if doc:
                for p in re.split(r"\n\s*\n", doc):
                    if len(p.strip()) > 20:
                        paras.append(p)
    return paras


def code_chunks():
    files = ["argparse.py", "json/decoder.py", "json/encoder.py", "textwrap.py", "difflib.py", "heapq.py",
             "bisect.py", "functools.py", "dataclasses.py", "enum.py", "string.py", "shlex.py", "calendar.py",
             "fractions.py", "statistics.py", "base64.py", "csv.py", "configparser.py", "tokenize.py", "ast.py",
             "inspect.py", "pathlib/_local.py", "typing.py", "contextlib.py", "copy.py", "pprint.py", "random.py",
             "tempfile.py", "shutil.py", "glob.py", "fnmatch.py", "gettext.py", "locale.py", "operator.py",
             "queue.py", "sched.py", "selectors.py", "subprocess.py", "threading.py", "timeit.py", "trace.py"]
    chunks = []
    for f in files:
        p = os.path.join(stdlib_dir(), f)
        if not os.path.exists(p):
            continue
        lines = open(p, encoding="utf-8").read().split("\n")
        i = 0
        while i < len(lines):
            n = R.choice([1, 2, 3, 5, 8, 13, 21, 40])
            chunks.append("\n".join(lines[i:i + n]))
            i += n
    return chunks


OTHER_CODE = [
    '#include <vector>\n#include <cstdio>\n\nint main(int argc, char** argv) {\n  std::vector<int> v{1, 2, 3};\n'
    '  for (auto& x : v) x *= 2;  // double\n  printf("%d\\n", v[0]);\n  return 0;\n}\n',
    'template <typename T>\nconstexpr T clamp(T x, T lo, T hi) noexcept { return x < lo ? lo : (hi < x ? hi : x); }\n',
    'fn main() {\n    let v: Vec<u32> = (0..10).map(|x| x * x).collect();\n    println!("{:?}", v);\n}\n',
    'impl<\'a> Iterator for Tokens<\'a> {\n    type Item = &\'a str;\n    fn next(&mut self) -> Option<Self::Item> { None }\n}\n',
    'package main\n\nimport "fmt"\n\nfunc main() {\n\tch := make(chan int)\n\tgo func() { ch <- 42 }()\n\tfmt.Println(<-ch)\n}\n',
    'const add = (a, b) => a + b;\nexport default async function fetchAll(urls) {\n  return Promise.all(urls.map(u => fetch(u).then(r => r.json())));\n}\n',
    'SELECT u.id, u.name, COUNT(o.id) AS orders\nFROM users u LEFT JOIN orders o ON o.user_id = u.id\n'
    'WHERE u.created_at >= \'2024-01-01\'\nGROUP BY u.id, u.name HAVING COUNT(o.id) > 3\nORDER BY orders DESC;\n',
    '<!DOCTYPE html>\n<html lang="en">\n<head><meta charset="utf-8"><title>Test &amp; demo</title></head>\n'
    '<body>\n  <div class="card" id=\'main\'>\n    <p>Hello, <b>world</b>!</p>\n  </div>\n</body>\n</html>\n',
    '.card > p:first-child { margin: 0 auto; color: #ff00aa; }\n@media (max-width: 600px) { .card { padding: 4px; } }\n',
    '#!/usr/bin/env bash\nset -euo pipefail\nfor f in *.txt; do\n  wc -l "$f" | awk \'{print $1}\'\ndone\n'
    'echo "${HOME:-/tmp}" >&2 && exit 0\n',
    '# Title\n\nSome *emphasis* and **bold** text with `code` and a [link](https://example.com).\n\n'
    '- item one\n- item two\n  1. nested\n\n```python\nprint("hi")\n```\n',
    '\\documentclass{article}\n\\begin{document}\n$E = mc^2$ and $\\int_0^\\infty e^{-x^2}\\,dx = '
    '\\frac{\\sqrt{\\pi}}{2}$\n\\end{document}\n',
    'public class Main {\n    public static void main(String[] args) {\n        System.out.println("Hello");\n    }\n}\n',
    'def f(x: int) -> int:\r\n    return x ** 2\r\n\r\nclass A:\r\n\tpass\r\n',
    '{\n\t"name": "espetero",\n\t"version": "0.1.0",\n\t"deps": ["metal", "json"],\n\t"ok": true,\n\t"n": null\n}\n',
    'x = [1,2,3];y={"a":1,"b":[true,false,null]};z=f(x,y)if x else g(**y)',
    '    if (a && b || !c) { return a->b.c[d] >> 2 <<= 3; } else if (x != y) {}\n',
    'diff --git a/x.py b/x.py\n@@ -1,3 +1,4 @@\n-old line\n+new line\n+another\n context\n',
]

LANG_SAMPLES = [
    "¿Dónde está la biblioteca? ¡Qué día tan bonito! El niño comió pingüinos y jalapeños.",
    "L'été dernier, nous sommes allés à la plage. Où est-ce que tu as mis les clés ? C'est déjà l'heure !",
    "Die Straße ist zu schmal für große Lastwagen. Größenwahn, Übermut und Ärger — ein Fußballspiel.",
    "A ação não é fácil, mas o coração está em paz. Você já viu o pôr do sol em São Paulo?",
    "Perché non vieni con noi? È già tardi e la città è piena di gente.",
    "Tiếng Việt có nhiều dấu thanh: sắc, huyền, hỏi, ngã, nặng. Người Việt Nam rất hiếu khách.",
    "我爱自然语言处理。今天天气很好，我们去公园散步吧！机器学习模型需要大量的数据。",
    "繁體中文與簡體中文的差異在於字形。臺灣、香港、澳門使用繁體字。",
    "日本語のテキストを処理します。ひらがな、カタカナ、漢字が混在しています。「こんにちは」と彼は言った。",
    "ｶﾀｶﾅ（半角）とカタカナ（全角）、ＡＢＣ１２３、ガギグゲゴ、パピプペポ。",
    "안녕하세요! 한국어 텍스트를 토큰화합니다. 저는 학생입니다. 값을 읽었습니다.",
    "مرحبا بالعالم! اللغة العربية جميلة جداً. كَتَبَ الطَّالِبُ الدَّرْسَ.",
    "שלום עולם! עִבְרִית עם ניקוד: בְּרֵאשִׁית בָּרָא אֱלֹהִים.",
    "नमस्ते दुनिया! हिन्दी भाषा में क़लम और ज़मीन जैसे शब्द हैं। मैं ठीक हूँ।",
    "আমি বাংলায় গান গাই। বাংলা ভাষা খুব সুন্দর।",
    "வணக்கம் உலகம்! தமிழ் மொழி மிகவும் பழமையானது.",
    "สวัสดีครับ ภาษาไทยไม่มีการเว้นวรรคระหว่างคำ",
    "Привет, мир! Съешь же ещё этих мягких французских булок, да выпей чаю.",
    "Γειά σου κόσμε! Η ελληνική γλώσσα έχει τόνους: ά, έ, ή, ί, ό, ύ, ώ, ΐ, ΰ.",
    "İstanbul'da ılık bir gün. Iğdır'lı öğrenciler şarkı söylüyor.",
    "Zażółć gęślą jaźń. Pchnąć w tę łódź jeża lub ośm skrzyń fig.",
    "سلام دنیا! زبان فارسی زیباست. می" + C(0x200C) + "خواهم کتاب بخوانم.",
    "اردو ایک خوبصورت زبان ہے۔",
    "བོད་ཡིག་ནི་སྐད་ཡིག་ཞིག་ཡིན། ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ",
    "ሰላም ለዓለም! ኢትዮጵያ",
    "Բարեւ աշխարհ", "გამარჯობა მსოფლიო", "မင်္ဂလာပါ",
    "សួស្តី" + C(0x200B) + "ពិភពលោក",
    "ආයුබෝවන්", "నమస్కారం ప్రపంచం", "ನಮಸ್ಕಾರ ಜಗತ್ತು ಕೊಡು ಕೋಟೆ", "നമസ്കാരം ലോകം", "નમસ્તે દુનિયા", "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ",
    "Сайн байна уу ᠮᠣᠩᠭᠣᠯ", "Ẓ̌ṹṩ ǅ ǈ ǋ ﬁ ﬂ Ⅻ ⅷ ① ⑳ ½ ¾ ² ³ ℃ ℉ № ™ Å Ω",
    "Čeština: příliš žluťoučký kůň úpěl ďábelské ódy.", "Hrvatski: Đakovo, ćevapi, čaša, šuma, žaba.",
    "Íslenska: Þórður og Ægir fóru í ferð.", "Română: știință, țară, înțelegere.",
    "Magyar: árvíztűrő tükörfúrógép.", "Suomi: Äänekoski ja Öljy.",
    "Kiswahili: Habari ya asubuhi? Nzuri sana, asante.", "Yorùbá: Ẹ káàárọ̀, ṣé dáadáa ni?",
    "ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ", "ᐃᓄᒃᑎᑐᑦ", "ꦧꦱꦗꦮ", "𐌰𐌱𐌲 Gothic", "𓀀𓀁𓀂 hieroglyphs", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕞𝕒𝕥𝕙 𝟘𝟙𝟚",
]

EMOJI = ["😀", "😂", "🥲", "👍", "👍🏽", "👩‍💻", "👨‍👩‍👧‍👦", "🏳️‍🌈", "🇪🇸", "🇯🇵", "🇺🇸", "1️⃣", "#️⃣",
         "❤️", "❤", "✨", "🔥", "🚀", "🧑🏿‍🚀", "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "👁️‍🗨️", "🤷‍♀️", "🙆🏻‍♂️", "☺️", "✌🏾", "🫠", "🪿"]

WS = [" ", "  ", "\t", "\n", "\r", "\r\n", "\x0b", "\x0c", C(0x85), C(0xA0), C(0x1680), C(0x2000), C(0x2001),
      C(0x2002), C(0x2003), C(0x2004), C(0x2005), C(0x2006), C(0x2007), C(0x2008), C(0x2009), C(0x200A), C(0x2028),
      C(0x2029), C(0x202F), C(0x205F), C(0x3000), C(0x180E), C(0x200B), C(0xFEFF), "\n\n", "    ", "\t\t", "\x00"]

QWEN_SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
                "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>",
                "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>", "<|fim_prefix|>",
                "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<tool_response>",
                "</tool_response>", "<think>", "</think>", "<|audio_start|>", "<|audio_end|>", "<tts_pad>",
                "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>", "<|audio_pad|>"]
BROKEN_SPECIAL = ["<|im_start", "im_start|>", "<|im_end|", "<think", "</think", "<|", "|>", "<<|im_start|>|>",
                  "<|im_start|im_end|>", "<|im_start|><|im_end|>", "<|IM_START|>", "< |im_start| >", "<tool_call",
                  "<|endoftext|><|endoftext|>", "<<think>>", "</think></think>", "<|fim_prefix|><|fim_suffix|>"]
FLAG_TOKENS = ["zqxj", "<ls>", "bar" + C(0x301), "bar" + C(0x155), "Special!", " spaced ", C(0x120) + "foo", "hello"]

CONTRACTIONS = ["'s", "'S", "'" + C(0x17F), "'t", "'T", "'re", "'RE", "'Re", "'rE", "'ve", "'VE", "'vE", "'m", "'M",
                "'ll", "'LL", "'lL", "'Ll", "'d", "'D", "'", "''", "'x", "'lx", "'rx", C(0x2019) + "s", "`s"]

DIGITS = ["0123456789", "٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹", "०१२३४५६७८९", "０１２３４５６７８９", "¹²³⁴⁵⁶⁷⁸⁹⁰", "₀₁₂₃",
          "①②③④⑤", "ⅠⅡⅢⅣⅤⅫ", "½¼¾⅓", "𝟎𝟏𝟐𝟑", "〇一二三", "๐๑๒๓", "߀߁߂"]


def load_barriers():
    """NFC barrier code points from the generated table (flag 16)."""
    txt = open(UNICODE_DATA_H).read()
    body = txt[txt.index("kRanges[kNumRanges] = {"):txt.index("};", txt.index("kRanges[kNumRanges] = {"))]
    vals = [int(x, 16) for x in re.findall(r"0x([0-9a-f]{8})", body)]
    out = []
    for i, v in enumerate(vals):
        start, flags = v >> 8, v & 0xFF
        end = (vals[i + 1] >> 8) if i + 1 < len(vals) else 0x110000
        if flags & 16:
            out.extend(range(start, end))
    return out


BARRIERS = load_barriers()
MARKS = [c for c in range(0x300, 0x110000) if not 0xD800 <= c <= 0xDFFF and unicodedata.combining(C(c))]
NFC_ODD = [0x0F73, 0x0F75, 0x0F81, 0x0344, 0x0340, 0x0341, 0x0343, 0x0374, 0x037E, 0x0387, 0x0958, 0x0959, 0x2126,
           0x212A, 0x212B, 0x2329, 0x232A, 0xF900, 0xFA10, 0x2F800, 0x1D15E, 0x1D160, 0x1D1BB, 0x0CC0, 0x0CCA, 0x0CD5,
           0x0CC2, 0x0B3E, 0x0B57, 0x0BBE, 0x0DCF, 0x102E, 0x1B35, 0x3099, 0x309A, 0x11930, 0x11935, 0x11938,
           0x113C2, 0x113B8, 0x113C5, 0x113C7, 0x105D2, 0x16D67, 0x1611E, 0x1611F, 0x0345, 0x0334, 0x0316, 0xFEFF]
BASES = list("aeiouyAEIOUYnNcCsSzZgGkKwWx<>=") + [C(c) for c in [0x391, 0x3B1, 0x3C9, 0x415, 0x438, 0x5D0, 0x627,
                                                                0x915, 0x995, 0xB95, 0xE01, 0xF40, 0x1100, 0xAC00,
                                                                0x1E00, 0x4E00, 0x3042, 0x30AB]]


def rand_cp():
    x = R.random()
    if x < 0.25:
        c = R.randint(0x20, 0x7E)
    elif x < 0.28:
        c = R.randint(0, 0x1F)
    elif x < 0.42:
        c = R.randint(0x80, 0x7FF)
    elif x < 0.65:
        c = R.randint(0x800, 0xFFFF)
    elif x < 0.8:
        c = R.randint(0x10000, 0x2FFFF)
    elif x < 0.86:
        c = R.randint(0x30000, 0x10FFFF)
    elif x < 0.9:
        c = R.choice(MARKS)
    elif x < 0.93:
        c = R.choice(BARRIERS)
    elif x < 0.96:
        c = R.choice(NFC_ODD)
    else:
        c = ord(R.choice(WS)[0])
    if 0xD800 <= c <= 0xDFFF:
        c = 0xFFFD
    return C(c)


def nfc_stress():
    out = []
    for _ in range(R.randint(1, 6)):
        out.append(R.choice(BASES))
        for _ in range(R.randint(0, 4)):
            r = R.random()
            if r < 0.55:
                out.append(C(R.choice(MARKS)))
            elif r < 0.75:
                out.append(C(R.choice(BARRIERS)))
            else:
                out.append(C(R.choice(NFC_ODD)))
    return "".join(out)


def hangul():
    out = []
    for _ in range(R.randint(1, 8)):
        r = R.random()
        if r < 0.4:
            out.append(C(R.randint(0x1100, 0x1112)) + C(R.randint(0x1161, 0x1175)) +
                       (C(R.randint(0x11A8, 0x11C2)) if R.random() < 0.5 else ""))
        elif r < 0.6:
            out.append(C(0xAC00 + 28 * R.randint(0, 398)) + C(R.randint(0x11A8, 0x11C2)))  # LV + T composes
        elif r < 0.8:
            out.append(C(R.randint(0xAC00, 0xD7A3)))
        else:
            out.append(C(R.randint(0x1161, 0x1175)) + C(R.randint(0x11A8, 0x11C2)))
    return "".join(out)


def mutate(s):
    """Random transforms: normalization forms, case, inserted whitespace/specials."""
    r = R.random()
    if r < 0.15:
        s = unicodedata.normalize("NFD", s)
    elif r < 0.2:
        s = unicodedata.normalize("NFKD", s)
    elif r < 0.25:
        s = s.upper()
    elif r < 0.3:
        s = s.swapcase()
    if R.random() < 0.2:
        i = R.randint(0, len(s))
        s = s[:i] + R.choice(WS) * R.randint(1, 3) + s[i:]
    return s


def insert_tokens(s, pool):
    for _ in range(R.randint(1, 3)):
        i = R.randint(0, len(s))
        tok = R.choice(pool)
        pre = R.choice(["", " ", "\n", "  ", "x", "_", "1", ".", C(0x3000)])
        post = R.choice(["", " ", "\n", "  ", "y", "_", "2", "!", C(0x2028)])
        s = s[:i] + pre + tok + post + s[i:]
    return s


def chat(eng):
    turns = []
    for _ in range(R.randint(1, 4)):
        role = R.choice(["system", "user", "assistant", "tool"])
        body = R.choice(eng)[:R.randint(10, 300)]
        if role == "assistant" and R.random() < 0.5:
            body = "<think>\n" + R.choice(eng)[:200] + "\n</think>\n\n" + body
        if R.random() < 0.3:
            body += '\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        turns.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
    s = "".join(turns)
    if R.random() < 0.5:
        s += "<|im_start|>assistant\n"
    return s


def build_corpus():
    eng = english_docs()
    code = code_chunks()
    corpus = ["", " ", "\n", "a", "'", "'s", "  \n", "\r\n", C(0xFEFF) + "hi", "\x00", "\x7f", C(0xFFFD)]
    corpus += [C(c) for c in range(0x80)]
    corpus += [C(c) for c in range(0x80, 0x250)]
    # English prose and code
    corpus += R.sample(eng, min(len(eng), 3500))
    corpus += R.sample(code, min(len(code), 2500))
    corpus += [mutate(R.choice(OTHER_CODE)) for _ in range(300)]
    corpus += OTHER_CODE
    # JSON
    for _ in range(800):
        obj = {R.choice(["id", "name", "text", "ok", "values", "nested", "ünï", "键"]): R.choice(
            [R.randint(-10**9, 10**9), R.random(), R.choice(eng)[:40], [1, 2.5, None, True], {"a": {"b": [1, {"c": "d"}]}},
             R.choice(LANG_SAMPLES)[:30]]) for _ in range(R.randint(1, 6))}
        corpus.append(json.dumps(obj, ensure_ascii=R.random() < 0.5, indent=R.choice([None, 2, 4, "\t"])))
    # languages
    corpus += LANG_SAMPLES
    for _ in range(3500):
        parts = [R.choice(LANG_SAMPLES) for _ in range(R.randint(1, 3))]
        s = R.choice([" ", "\n", "", "  "]).join(parts)
        if R.random() < 0.5:
            a = R.randint(0, len(s))
            s = s[a:a + R.randint(1, 80)]
        corpus.append(mutate(s))
    # emoji
    for _ in range(900):
        s = "".join(R.choice(EMOJI + [" ", "", "a", "!", "\n", C(0xFE0F), C(0x200D), C(0x1F3FB)]) for _ in range(R.randint(1, 12)))
        corpus.append(s)
    # whitespace torture
    for _ in range(2200):
        s = "".join(R.choice(WS + ["a", "word", ".", "1", "#", "x y", "é", "中", C(0x5D0)] if R.random() < 0.4 else WS)
                    for _ in range(R.randint(1, 30)))
        corpus.append(s)
    for n in (50, 200, 1000, 5000):
        corpus.append(" " * n)
        corpus.append("\n" * n)
        corpus.append((" \t\r\n" + C(0x3000)) * (n // 5) + "end")
        corpus.append("x" + " " * n + "y")
    # digits
    for _ in range(1000):
        s = "".join(R.choice(DIGITS)[R.randint(0, 3):][:R.randint(1, 8)] + R.choice(["", " ", ".", ",", "e-", "x", "%"])
                    for _ in range(R.randint(1, 6)))
        corpus.append(s)
    corpus += ["".join(R.choice("0123456789") for _ in range(n)) for n in (10, 11, 99, 1000, 4096)]
    # contractions
    words = ["it", "IT", "he", "We", "they", "you", "I", "Y", "o", "rock", "n", "don", "can", "y'all"]
    for _ in range(1200):
        corpus.append(" ".join(R.choice(words) + R.choice(CONTRACTIONS) * R.randint(1, 2) + R.choice(["", "x", " ", "e", "l"])
                               for _ in range(R.randint(1, 5))))
    # special tokens: chat transcripts, broken tokens, tokens glued to text
    for _ in range(900):
        corpus.append(chat(eng))
    for _ in range(900):
        base = R.choice(eng + LANG_SAMPLES + code)[:R.randint(0, 200)]
        corpus.append(insert_tokens(base, QWEN_SPECIAL + BROKEN_SPECIAL))
    for _ in range(500):
        base = R.choice(eng + code)[:R.randint(0, 150)]
        corpus.append(insert_tokens(base, FLAG_TOKENS + QWEN_SPECIAL[:5]))
    corpus += ["<|im_end|>" + C(0x338), "a<|im_start|>" + C(0x338) + "b", "=" + C(0x338), "<" + C(0x338) + "|im_start|>",
               "<|im_start|>" * 3, "<think></think>" * 2, "<|endoftext|>\n\n  <|endoftext|>", "x  <ls>  y",
               "<ls>" + C(0x3000) + C(0x2028) + "z", "bar" + C(0x301) + "x", "bare" + C(0x301), "zqxj", "a zqxj b",
               "afoo zqxjb zqxj_", "Special!Special!", " spaced  spaced ", C(0x120) + "foo" + C(0x120), "hello world"]
    # random code points
    for _ in range(3500):
        corpus.append("".join(rand_cp() for _ in range(R.randint(1, 40))))
    # NFC stress
    for _ in range(2200):
        corpus.append(nfc_stress())
    for _ in range(500):
        corpus.append(hangul())
    for c in BARRIERS + NFC_ODD:
        corpus.append("a" + C(c) + C(0x301) + C(0x316))
        corpus.append(C(0x301) + C(c) + "e" + C(0x301))
    # very long words and runs
    corpus += ["a" * 5000, "ab" * 3000, "x" * 20000, "".join(R.choice("abcdefghij") for _ in range(10000)),
               "😀" * 2000, "中" * 5000, "!" * 3000, "-=" * 2000, "é" * 3000, "e" + C(0x301) * 500,
               "".join(R.choice(LANG_SAMPLES) for _ in range(40)).replace(" ", ""),
               "Supercalifragilisticexpialidocious" * 100, "ـ" * 1000, C(0xFFFD) * 1000]
    return corpus


def bench_text():
    """~100 KB: English prose followed by Python code (fixed selection)."""
    rr = random.Random(7)
    eng = english_docs()
    rr.shuffle(eng)
    prose = ""
    for p in eng:
        if len(prose) > 50000:
            break
        prose += p + "\n\n"
    code = open(os.path.join(stdlib_dir(), "argparse.py"), encoding="utf-8").read()[:50000]
    return prose + code


# ============================================================================ tokenizers
def make_variants(out_dir):
    """Synthetic tokenizer.json files for code paths the real models do not use."""
    base3 = json.load(open(os.path.join(HF_QWEN3, "tokenizer.json")))
    v = json.loads(json.dumps(base3))
    v["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = LLAMA3_RE
    v["model"]["ignore_merges"] = True
    v["model"]["merges"] = [" ".join(m) if isinstance(m, list) else m for m in v["model"]["merges"]]  # string form
    llama_dir = os.path.join(out_dir, "variants", "llama3")
    os.makedirs(llama_dir, exist_ok=True)
    json.dump(v, open(os.path.join(llama_dir, "tokenizer.json"), "w"), ensure_ascii=True)  # \\u escapes

    base35 = json.load(open(os.path.join(HF_QWEN35, "tokenizer.json")))
    v = json.loads(json.dumps(base35))
    nid = max(t["id"] for t in v["added_tokens"]) + 1

    def add(content, **kw):
        nonlocal nid
        t = {"id": nid, "content": content, "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False, "special": False}
        t.update(kw)
        v["added_tokens"].append(t)
        nid += 1

    add("zqxj", single_word=True)
    add("<ls>", lstrip=True, rstrip=True)
    add("bar" + C(0x301), normalized=True)
    add("Special!", special=True)
    add(" spaced ")
    add(C(0x120) + "foo")
    add("hello")                      # already in the vocabulary: keeps the vocabulary id
    add("<|im_end|>", special=False)  # redefinition: the reference keeps it special
    flags_dir = os.path.join(out_dir, "variants", "flags")
    os.makedirs(flags_dir, exist_ok=True)
    json.dump(v, open(os.path.join(flags_dir, "tokenizer.json"), "w"), ensure_ascii=False)
    json.dump({"eos_token": "<|im_end|>", "pad_token": {"content": "<|endoftext|>"}, "bos_token": None,
               "chat_template": [{"name": "tool_use", "template": "T"}, {"name": "default", "template": "D"}]},
              open(os.path.join(flags_dir, "tokenizer_config.json"), "w"))
    return llama_dir, flags_dir


class Ref:
    def __init__(self, name, kind, path, tok, tok_ns, eos, bos, pad, template, special_ids, added):
        self.name, self.kind, self.path = name, kind, path
        self.tok, self.tok_ns = tok, tok_ns  # tok_ns: same, with encode_special_tokens=True
        self.eos, self.bos, self.pad, self.template = eos, bos, pad, template
        self.special_ids, self.added = special_ids, added


def hf_ref(name, d, config=True):
    tj = os.path.join(d, "tokenizer.json")
    tok = Tokenizer.from_file(tj)
    tok_ns = Tokenizer.from_file(tj)
    tok_ns.encode_special_tokens = True
    added = [(t.content, i) for i, t in sorted(tok.get_added_tokens_decoder().items())]
    # Special = skipped by decode(skip_special_tokens=True): the reference's special set,
    # which a later non-special redefinition of the same content does not shrink.
    special = sorted(i for _, i in added if tok.decode([i], skip_special_tokens=True) == "")
    eos = bos = pad = -1
    template = ""
    cfgp = os.path.join(d, "tokenizer_config.json")
    if config and os.path.exists(cfgp):
        cfg = json.load(open(cfgp))

        def tid(k):
            v = cfg.get(k)
            if isinstance(v, dict):
                v = v.get("content")
            return tok.token_to_id(v) if v else -1

        eos, bos, pad = tid("eos_token"), tid("bos_token"), tid("pad_token")
        ct = cfg.get("chat_template")
        if isinstance(ct, str):
            template = ct
        elif isinstance(ct, list):
            template = next((t["template"] for t in ct if t.get("name") == "default"), ct[0]["template"])
    jp = os.path.join(d, "chat_template.jinja")
    if os.path.exists(jp):
        template = open(jp, encoding="utf-8").read()
    eos, bos, pad = [x if x is not None else -1 for x in (eos, bos, pad)]
    return Ref(name, "hf", d, tok, tok_ns, eos, bos, pad, template, special, added)


def gguf_ref(name, path, hf_dir, special_extra=()):
    """Default ids: the HF tokenizer of the same model. Special flags come from the GGUF
    token types (llama.cpp marks <|fim_*|> & co. as control tokens), so the parse_special=false
    and skip_special references use a copy of tokenizer.json with those flags."""
    kv = read_gguf_tokenizer(path)
    types = kv["tokenizer.ggml.token_type"]
    toks = kv["tokenizer.ggml.tokens"]
    d = json.load(open(os.path.join(hf_dir, "tokenizer.json")))
    for t in d["added_tokens"]:
        t["special"] = types[t["id"]] == 3 or t["content"] in special_extra
        assert toks[t["id"]] == t["content"]
    s = json.dumps(d)
    tok = Tokenizer.from_file(os.path.join(hf_dir, "tokenizer.json"))
    tok_flags = Tokenizer.from_str(s)
    tok_ns = Tokenizer.from_str(s)
    tok_ns.encode_special_tokens = True
    special = [i for i, t in enumerate(types) if t == 3 or toks[i] in special_extra]
    added = [(toks[i], i) for i, t in enumerate(types) if t in (3, 4) or toks[i] in special_extra]
    ref = Ref(name, "gguf", path, tok, tok_ns, kv.get("tokenizer.ggml.eos_token_id", -1),
              kv.get("tokenizer.ggml.bos_token_id", -1), kv.get("tokenizer.ggml.padding_token_id", -1),
              kv.get("tokenizer.chat_template", ""), special, added)
    ref.tok_decode = tok_flags
    return ref


# ============================================================================ output
class Writer:
    def __init__(self, f):
        self.f = f

    def u8(self, v):
        self.f.write(struct.pack("<B", v))

    def u32(self, v):
        self.f.write(struct.pack("<I", v))

    def i32(self, v):
        self.f.write(struct.pack("<i", v))

    def str(self, s):
        b = s.encode("utf-8") if isinstance(s, str) else s
        self.u32(len(b))
        self.f.write(b)

    def ids(self, ids):
        self.u32(len(ids))
        self.f.write(struct.pack(f"<{len(ids)}i", *ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    llama_dir, flags_dir = make_variants(args.out_dir)
    makers = [("qwen35", lambda: hf_ref("qwen35", HF_QWEN35)), ("qwen3", lambda: hf_ref("qwen3", HF_QWEN3)),
              ("qwen38", lambda: hf_ref("qwen38", HF_QWEN38)),
              ("qwen35-gguf", lambda: gguf_ref("qwen35-gguf", GGUF_QWEN35, HF_QWEN35)),
              ("llama3-regex", lambda: hf_ref("llama3-regex", llama_dir)), ("flags", lambda: hf_ref("flags", flags_dir))]
    if HF_GEMMA4:
        makers.append(("gemma4", lambda: hf_ref("gemma4", HF_GEMMA4)))
    if HF_GEMMA4 and GGUF_GEMMA4:  # <eos> is a normal token in the GGUF; the engine keeps it special, as tokenizer.json does
        makers.append(("gemma4-gguf", lambda: gguf_ref("gemma4-gguf", GGUF_GEMMA4, HF_GEMMA4, special_extra=("<eos>",))))
    keep = os.environ.get("TOK_REF_ONLY", "").split(",") if os.environ.get("TOK_REF_ONLY") else None  # e.g. gemma4,qwen35
    refs = [make() for name, make in makers if keep is None or name in keep]
    corpus = build_corpus()
    print(f"corpus: {len(corpus)} strings, {sum(len(s) for s in corpus)} code points", file=sys.stderr)
    bench = bench_text()

    with open(os.path.join(args.out_dir, "tok_ref.bin"), "wb") as f:
        w = Writer(f)
        f.write(b"TOKREF03")
        w.u32(len(refs))
        for r in refs:
            w.str(r.name)
            w.str(r.kind)
            w.str(r.path)
            w.i32(r.eos)
            w.i32(r.bos)
            w.i32(r.pad)
            w.i32(r.tok.get_vocab_size(with_added_tokens=True))
            w.str(r.template)
            w.ids(r.special_ids)
            w.u32(len(r.added))
            for content, i in r.added:
                w.str(content)
                w.i32(i)
        per_tok = []
        for r in refs:
            dec = getattr(r, "tok_decode", r.tok)
            ids = [e.ids for e in r.tok.encode_batch(corpus)]
            ids_ns = [e.ids for e in r.tok_ns.encode_batch(corpus)]
            d0 = dec.decode_batch(ids, skip_special_tokens=False)
            d1 = dec.decode_batch(ids, skip_special_tokens=True)
            per_tok.append((ids, ids_ns, d0, d1))
            print(f"  {r.name}: {sum(len(x) for x in ids)} tokens", file=sys.stderr)
        w.u32(len(corpus))
        for k, text in enumerate(corpus):
            w.str(text)
            for ids, ids_ns, d0, d1 in per_tok:
                fl = (1 if ids_ns[k] == ids[k] else 0) | (2 if d0[k] == text else 0) | (4 if d1[k] == d0[k] else 0)
                w.u8(fl)
                w.ids(ids[k])
                if not fl & 1:
                    w.ids(ids_ns[k])
                if not fl & 2:
                    w.str(d0[k])
                if not fl & 4:
                    w.str(d1[k])
        # decode cases: every id of every tokenizer in chunks, plus random id sequences
        cases = []
        for ti, r in enumerate(refs):
            dec = getattr(r, "tok_decode", r.tok)
            n = r.tok.get_vocab_size(with_added_tokens=True)
            chunks = [list(range(i, min(n, i + 64))) for i in range(0, n, 64)]
            for _ in range(300):
                chunks.append([R.randrange(n) for _ in range(R.randint(1, 30))] + [R.randrange(n + 50)])
            for skip in (0, 1):
                outs = dec.decode_batch(chunks, skip_special_tokens=bool(skip))
                for c, o in zip(chunks, outs):
                    cases.append((ti, skip, c, o))
        w.u32(len(cases))
        for ti, skip, ids, o in cases:
            w.u32(ti)
            w.u8(skip)
            w.ids(ids)
            w.str(o)
        w.str(bench)
        for r in refs:
            w.ids(r.tok.encode(bench).ids)
    print(f"wrote {os.path.join(args.out_dir, 'tok_ref.bin')}: {len(corpus)} cases, {len(cases)} decode cases",
          file=sys.stderr)


if __name__ == "__main__":
    main()
