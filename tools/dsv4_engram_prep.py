#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero
"""DeepSeek-V4.1 Engram constants for Espetero (docs/deepseek_v41.md, Engram).

Writes <model>/espetero_engram.bin: the compressed token map (every token id
onto the normalized id space the n-gram hashes use), the hash multipliers and
the prime bucket sizes, exactly as DeepSeek's inference/engram.py builds them.
The map needs a Unicode normalizer Espetero does not carry (NFKC, NFD, strip
accents, lowercase), hence this one-time step.

    python3 tools/dsv4_engram_prep.py /path/to/DeepSeek-V4.1-Flash

Needs numpy and tokenizers.
"""
import json
import struct
import sys

import numpy as np
from tokenizers import Regex, Tokenizer, normalizers


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def token_map(tok):
    sentinel = ""
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    n = tok.get_vocab_size(with_added_tokens=True)
    key_to_new, lookup = {}, [0] * n
    for t in range(n):
        text = tok.decode([t], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(t)
        else:
            s = norm.normalize_str(text)
            key = s if s else text
        lookup[t] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)


def main(model_dir):
    cfg = json.load(open(model_dir + "/config.json"))
    tc = cfg.get("text_config", cfg)
    layers = tc["engram_layer_ids"]
    max_n, heads = tc["engram_max_ngram_size"], tc["engram_n_heads"]
    tok = Tokenizer.from_file(model_dir + "/tokenizer.json")
    lookup, cvocab = token_map(tok)
    if cvocab != tc["engram_compressed_vocab_size"]:
        sys.exit(f"compressed vocabulary {cvocab} != config {tc['engram_compressed_vocab_size']}")
    bound = max(1, (np.iinfo(np.int64).max // cvocab) // 2)
    mults = []
    for l in layers:
        g = np.random.default_rng(10007 * l)
        mults.append([int(v) * 2 + 1 for v in g.integers(low=0, high=bound, size=(max_n,), dtype=np.int64)])
    primes, seen = [], set()
    for _ in layers:
        per = []
        for _ in range(max_n - 1):
            cur = tc["engram_vocab_size"] - 1
            for _ in range(heads):
                cur += 1
                while not is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                per.append(cur)
        primes.append(per)
    pad = lookup[tc.get("engram_pad_token_id", 2)]
    with open(model_dir + "/espetero_engram.bin", "wb") as f:
        f.write(b"ESPDSV4E")
        f.write(struct.pack("<8i", 1, len(lookup), cvocab, len(layers), max_n, heads, pad, 0))
        for row in mults:
            f.write(struct.pack(f"<{max_n}q", *row))
        for row in primes:
            f.write(struct.pack(f"<{len(row)}q", *row))
        f.write(np.asarray(lookup, dtype="<u4").tobytes())
    print(f"vocab {len(lookup)}, compressed {cvocab}, pad {pad}, multipliers {mults}, primes[0][:3] {primes[0][:3]}")


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
