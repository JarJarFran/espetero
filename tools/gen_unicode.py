#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Generate src/tok/unicode_data.h: the Unicode tables used by the BPE tokenizer.

Usage (run with a Python that has the Hugging Face `tokenizers` package, on macOS):
    python3 tools/gen_unicode.py src/tok/unicode_data.h
    python3 tools/gen_unicode.py --no-reference src/tok/unicode_data.h   # unicodedata only

The table maps every code point to a flags byte:
  bits 0-2  regex class: 0 other, 1 \\p{L}, 2 \\p{M}, 3 \\p{N}, 4 \\s (White_Space)
  bit 3     NFC_UNSAFE: NFC may change text around this code point (canonical
            combining class != 0, or NFC_Quick_Check No/Maybe). Text without such
            code points is already NFC and skips normalization entirely.
  bit 4     NFC_BARRIER: code point the reference normalizer does not know but
            CoreFoundation does (see below). Kept verbatim and never passed to CF.

Why probe the reference instead of trusting unicodedata alone:
  * The regex classes must match Oniguruma inside `tokenizers` (Unicode 16.0 as of
    tokenizers 0.23), which is newer than Python 3.13's unicodedata (15.1). The base
    sets come from unicodedata; code points the reference classifies differently are
    added/removed and reported.
  * The reference NFC normalizer (the `unicode-normalization-alignments` crate) uses
    old Unicode tables (~9.0), while CoreFoundation is at Unicode 16+. For code points
    that CF normalizes but the reference treats as unassigned (inert starters), the
    runtime splits the text around them so CF never sees them: this reproduces the
    reference exactly, because an inert starter is a normalization boundary.
    The unsafe set is likewise taken from the reference's behavior.
  * CoreFoundation's FormC skips some compositions when the first element arrives
    precomposed (U+AC00 U+11A8 stays, U+0CCA U+0CD5 stays) and does not reorder the
    decomposition of U+0F73/U+0F75/U+0F81 against preceding marks. The runtime
    therefore normalizes each span to FormD and then FormC; every probe below is
    checked against the reference with exactly that call sequence.
"""
import argparse
import ctypes
import ctypes.util
import sys
import unicodedata

MAX_CP = 0x110000
CLASS_O, CLASS_L, CLASS_M, CLASS_N, CLASS_WS = 0, 1, 2, 3, 4
F_UNSAFE, F_BARRIER = 8, 16
# Probe marks with canonical combining classes 1, 240 and 220.
MARK_1, MARK_240, MARK_220 = chr(0x0334), chr(0x0345), chr(0x0316)
# Unicode White_Space, which is what Oniguruma's \s matches for UTF-8 subjects.
WHITE_SPACE = {0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029,
               0x202F, 0x205F, 0x3000}


def all_cps(start=0):
    return [c for c in range(start, MAX_CP) if not (0xD800 <= c <= 0xDFFF)]


# ----------------------------------------------------------------------------- unicodedata base
def base_classes():
    cls = bytearray(MAX_CP)
    for c in range(MAX_CP):
        if 0xD800 <= c <= 0xDFFF:
            continue
        if c in WHITE_SPACE:
            cls[c] = CLASS_WS
            continue
        cat = unicodedata.category(chr(c))[0]
        cls[c] = {"L": CLASS_L, "M": CLASS_M, "N": CLASS_N}.get(cat, CLASS_O)
    return cls


def base_unsafe():
    """ccc != 0 or NFC_Quick_Check != Yes, from unicodedata (used without the reference)."""
    unsafe = set()
    for c in all_cps(0x80):
        s = chr(c)
        if unicodedata.combining(s) or unicodedata.normalize("NFC", s) != s:
            unsafe.add(c)
        d = unicodedata.normalize("NFD", s)
        if d != s:
            unsafe.update(ord(x) for x in d[1:])  # second elements of compositions: Maybe
    return unsafe


# ----------------------------------------------------------------------------- reference probing
def probe_regex_class(pattern):
    """Code points matched by a one-character regex, as seen by the reference (Oniguruma)."""
    from tokenizers import Regex, pre_tokenizers
    pt = pre_tokenizers.Split(Regex(pattern), behavior="isolated")
    cps = all_cps(1)
    matched = set()
    for i in range(0, len(cps), 50000):
        s = "\x00".join(chr(c) for c in cps[i:i + 50000])
        for piece, _ in pt.pre_tokenize_str(s):
            if "\x00" not in piece:
                matched.add(ord(piece))
    return matched


class CF:
    """CoreFoundation's CFStringNormalize through ctypes, called exactly like the C++ runtime
    does: UTF-16 in via CFStringAppendCharacters (CFStringCreateWithBytes would strip a
    leading U+FEFF as a byte order mark), UTF-16 out via CFStringGetCharacters."""

    def __init__(self):
        lib = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

        class CFRange(ctypes.Structure):
            _fields_ = [("location", ctypes.c_long), ("length", ctypes.c_long)]

        self.CFRange = CFRange
        lib.CFStringCreateMutable.restype = ctypes.c_void_p
        lib.CFStringCreateMutable.argtypes = [ctypes.c_void_p, ctypes.c_long]
        lib.CFStringAppendCharacters.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long]
        lib.CFStringNormalize.argtypes = [ctypes.c_void_p, ctypes.c_long]
        lib.CFStringGetLength.restype = ctypes.c_long
        lib.CFStringGetLength.argtypes = [ctypes.c_void_p]
        lib.CFStringGetCharacters.argtypes = [ctypes.c_void_p, CFRange, ctypes.c_char_p]
        lib.CFRelease.argtypes = [ctypes.c_void_p]
        self.lib = lib

    def normalize(self, text, *forms):  # 0 = NFD, 2 = NFC; applied in order
        lib = self.lib
        u16 = text.encode("utf-16-le")
        m = lib.CFStringCreateMutable(None, 0)
        lib.CFStringAppendCharacters(m, u16, len(u16) // 2)
        for form in forms:
            lib.CFStringNormalize(m, form)
        n = lib.CFStringGetLength(m)
        buf = ctypes.create_string_buffer(n * 2 + 2)
        lib.CFStringGetCharacters(m, self.CFRange(0, n), buf)
        lib.CFRelease(m)
        return buf.raw[:n * 2].decode("utf-16-le")


def batch(fn, strings, sep="\n", size=20000):
    """Apply a string->string normalizer to many probes at once. '\\n' is an inert
    starter, so probes cannot interact across it."""
    out = []
    for i in range(0, len(strings), size):
        chunk = strings[i:i + size]
        parts = fn(sep.join(chunk)).split(sep)
        assert len(parts) == len(chunk)
        out.extend(parts)
    return out


def nfc_sets(cf):
    """Returns (unsafe, barrier) code point sets, derived from reference + CF behavior."""
    from tokenizers import normalizers
    hf_nfc = normalizers.NFC().normalize_str
    hf_nfd = normalizers.NFD().normalize_str
    cf_nfc = lambda s: cf.normalize(s, 0, 2)  # the runtime sequence: FormD, then FormC
    cf_nfd = lambda s: cf.normalize(s, 0)
    cps = [c for c in all_cps(0x80)]
    # Probes around each code point: alone (NFC_QC=No), between marks of combining class 1
    # and 240 (reveals any other nonzero class through reordering), and before a class-220 mark.
    probes = []
    for c in cps:
        ch = chr(c)
        probes += [ch, "a" + ch + MARK_1, "a" + MARK_240 + ch, ch + MARK_220]
    hf_out = batch(hf_nfc, probes)
    cf_out = batch(cf_nfc, probes)
    singles = [chr(c) for c in cps]
    hf_d = batch(hf_nfd, singles)
    cf_d = batch(cf_nfd, singles)

    hf_active, cf_active, unsafe = set(), set(), set()
    for k, c in enumerate(cps):
        p = probes[4 * k:4 * k + 4]
        h = hf_out[4 * k:4 * k + 4]
        f = cf_out[4 * k:4 * k + 4]
        if any(h[j] != p[j] for j in range(4)) or hf_d[k] != singles[k]:
            hf_active.add(c)
        if any(f[j] != p[j] for j in range(4)) or cf_d[k] != singles[k]:
            cf_active.add(c)
        if h[0] != p[0] or h[1] != p[1] or h[2] != p[2]:
            unsafe.add(c)  # NFC_QC=No, or nonzero combining class (reordered)
        if hf_d[k] != singles[k]:
            unsafe.update(ord(x) for x in hf_d[k][1:])  # composition second elements (Maybe)
    # Composition probes: every canonical decomposition CF knows, recomposed by both.
    decomp = [cf_d[k] for k, c in enumerate(cps) if cf_d[k] != singles[k]]
    hf_comp = batch(hf_nfc, decomp)
    cf_comp = batch(cf_nfc, decomp)
    for d, h, f in zip(decomp, hf_comp, cf_comp):
        members = {ord(x) for x in d}
        if h != d:
            hf_active |= members
        if f != d:
            cf_active |= members
    barrier = {c for c in cf_active if c not in hf_active}
    assert not (barrier & unsafe)
    # Validation-only probes: a precomposed first element followed by the second element of
    # every canonical composition pair (e.g. U+00C5 U+0301 -> U+01FA), Hangul LV + T and LVT + T.
    pairs = []
    for c in cps:
        d = unicodedata.decomposition(chr(c))
        if d and not d.startswith("<") and len(d.split()) == 2:
            x, y = (chr(int(h, 16)) for h in d.split())
            pairs += [unicodedata.normalize("NFC", x) + y, "a" + unicodedata.normalize("NFC", x) + MARK_220 + y]
    for lv in range(0xAC00, 0xD7A4, 28):
        for t in range(0x11A8, 0x11C3):
            pairs += [chr(lv) + chr(t), chr(lv + 1) + chr(t), chr(lv) + chr(t) + chr(t)]
    return unsafe, barrier, (hf_nfc, lambda t: cf.normalize(t, 0, 2), probes + decomp + pairs)


def emulate_nfc(s, unsafe, barrier, cf_dc):
    """Python model of the C++ runtime: CF (FormD then FormC) on barrier-free runs that
    contain unsafe code points; barriers stay verbatim."""
    out, run = [], []

    def flush():
        if run:
            t = "".join(run)
            if any(ord(x) in unsafe for x in t):
                t = cf_dc(t)
            out.append(t)
            run.clear()

    for ch in s:
        if ord(ch) in barrier:
            flush()
            out.append(ch)
        else:
            run.append(ch)
    flush()
    return "".join(out)


# ----------------------------------------------------------------------------- output
def emit(path, flags, meta):
    ranges = []
    prev = None
    for c in range(MAX_CP):
        f = flags[c]
        if f != prev:
            ranges.append((c, f))
            prev = f
    lines = [
        "// SPDX-License-Identifier: Unicode-3.0",
        "// Derived from the Unicode Character Database, Copyright (c) 1991-2026 Unicode, Inc.,",
        "// Unicode License v3: see THIRD_PARTY_NOTICES.md.",
    ]
    lines.append("// Generated by tools/gen_unicode.py -- do not edit by hand.")
    for m in meta:
        lines.append("// " + m)
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("namespace espetero::tok::ucd {")
    lines.append("")
    lines.append("// Flags byte: bits 0-2 regex class, bit 3 NFC unsafe, bit 4 NFC barrier.")
    lines.append("enum : uint8_t { kOther = 0, kLetter = 1, kMark = 2, kNumber = 3, kSpace = 4, kClassMask = 7,")
    lines.append("               kNfcUnsafe = 8, kNfcBarrier = 16 };")
    lines.append("")
    lines.append("// Sorted runs: (first_code_point << 8) | flags. A run extends to the next entry.")
    lines.append(f"inline constexpr uint32_t kNumRanges = {len(ranges)};")
    lines.append("inline constexpr uint32_t kRanges[kNumRanges] = {")
    row = []
    for c, f in ranges:
        row.append(f"0x{(c << 8) | f:08x}")
        if len(row) == 8:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    lines.append("};")
    lines.append("")
    lines.append("// Flags of U+0000..U+007F (the ASCII fast path).")
    lines.append("inline constexpr uint8_t kAscii[128] = {")
    for i in range(0, 128, 16):
        lines.append("    " + ", ".join(str(flags[c]) for c in range(i, i + 16)) + ",")
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace espetero::tok::ucd")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(ranges)


def ranges_str(cps, limit=12):
    cps = sorted(cps)
    out = []
    i = 0
    while i < len(cps):
        j = i
        while j + 1 < len(cps) and cps[j + 1] == cps[j] + 1:
            j += 1
        out.append(f"{cps[i]:04X}" if i == j else f"{cps[i]:04X}..{cps[j]:04X}")
        i = j + 1
    return ", ".join(out[:limit]) + (f", ... ({len(out)} runs)" if len(out) > limit else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--no-reference", action="store_true",
                    help="use unicodedata only (no tokenizers/CoreFoundation probing)")
    args = ap.parse_args()

    cls = base_classes()
    meta = [f"Base: Python {sys.version.split()[0]} unicodedata {unicodedata.unidata_version}."]
    if args.no_reference:
        unsafe, barrier = base_unsafe(), set()
        meta.append("No reference probing: NFC/regex behavior may differ from HF for newer code points.")
    else:
        import tokenizers
        meta.append(f"Reconciled with the reference: tokenizers {tokenizers.__version__} (Oniguruma classes, NFC).")
        for name, pat, code in (("L", r"\p{L}", CLASS_L), ("M", r"\p{M}", CLASS_M), ("N", r"\p{N}", CLASS_N),
                                ("WS", r"\s", CLASS_WS)):
            ref = probe_regex_class(pat)
            mine = {c for c in range(MAX_CP) if cls[c] == code and not (0xD800 <= c <= 0xDFFF) and c != 0}
            add, rem = ref - mine, mine - ref
            for c in add:
                assert cls[c] in (CLASS_O, code), hex(c)
                cls[c] = code
            for c in rem:
                cls[c] = CLASS_O
            msg = f"\\p{{{name}}}: +{len(add)} -{len(rem)} code points vs unicodedata"
            if add:
                msg += f" (added: {ranges_str(add, 6)})"
            meta.append(msg)
            print(msg, file=sys.stderr)
        cf = CF()
        unsafe, barrier, (hf_nfc, cf_dc, probes) = nfc_sets(cf)
        base_u = base_unsafe()
        meta.append(f"NFC unsafe: {len(unsafe)} code points (unicodedata would give {len(base_u)}; "
                    f"reference-only {len(unsafe - base_u)}, unicodedata-only {len(base_u - unsafe)}).")
        meta.append(f"NFC barriers (CF normalizes, reference does not): {len(barrier)} code points: "
                    f"{ranges_str(barrier, 10)}")
        print(meta[-2], file=sys.stderr)
        print(meta[-1], file=sys.stderr)
        # Validate the runtime model against the reference on every probe.
        hf_out = batch(hf_nfc, probes)
        bad = [p for p, h in zip(probes, hf_out) if emulate_nfc(p, unsafe, barrier, cf_dc) != h]
        print(f"NFC emulation mismatches on {len(probes)} probes: {len(bad)}", file=sys.stderr)
        for p in bad[:10]:
            print("   ", [hex(ord(x)) for x in p], file=sys.stderr)
        if bad:
            sys.exit("NFC emulation does not match the reference; refusing to write tables")

    flags = bytearray(MAX_CP)
    for c in range(MAX_CP):
        f = cls[c]
        if c in unsafe:
            f |= F_UNSAFE
        if c in barrier:
            f |= F_BARRIER
        flags[c] = f
    # Surrogates never occur in valid UTF-8; give them their neighbor's flags so runs stay long.
    for c in range(0xD800, 0xE000):
        flags[c] = flags[0xD7FF] if flags[0xD7FF] == flags[0xE000] else 0
    n = emit(args.out, flags, meta)
    print(f"wrote {args.out}: {n} ranges", file=sys.stderr)


if __name__ == "__main__":
    main()
