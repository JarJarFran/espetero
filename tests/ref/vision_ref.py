# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Reference data for tests/vision_test.cpp (image preprocessing).

Run with a Python that has PIL 12.3 and numpy, and JEVFIRE set to a JEVfire
checkout (its metal package is the preprocessing reference):
  JEVFIRE=/path/to/jevfire python tests/ref/vision_ref.py OUT_DIR

Writes OUT_DIR/manifest.json and OUT_DIR/data.bin (all arrays, referenced by
[offset, length]). Cases:
  resize      random RGB images resized with PIL Image.resize(BICUBIC)
  preprocess  jevfire.metal.vision.preprocess on decoded images (float32 rows,
              grid) + the digest construction of src/vision/digest.cpp
  decode      encoded files (PNG/GIF/JPEG/BMP/TIFF/WebP/...) and PIL's
              Image.open(...).convert("RGB")
  smart_resize, mrope, expand, rows, dataurl  small exact cases
"""

import base64
import hashlib
import io
import glob
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np
from PIL import Image

_JEVFIRE = os.environ.get("JEVFIRE")
if not _JEVFIRE:
    sys.exit("set JEVFIRE to a JEVfire checkout (https://github.com/kikoncuo/jevfire)")
sys.path.insert(0, _JEVFIRE)
from jevfire.metal import vision as V  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260924)
random.seed(20260924)
blob = open(os.path.join(OUT, "data.bin"), "wb")
offset = 0


def put(b: bytes):
    global offset
    blob.write(b)
    ref = [offset, len(b)]
    offset += len(b)
    return ref


# ---------------------------------------------------------------- images


def photo(h, w, seed=0, noise=10.0):
    """Photo-like content: gradients, waves, hard edges, saturated patches, noise."""
    r = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    fx, fy = r.uniform(5, 40, 2)
    img = np.stack(
        [
            128 + 110 * np.sin(x / fx + r.uniform(0, 6)) * np.cos(y / fy),
            x * 255.0 / max(w - 1, 1),
            (y * 255.0 / max(h - 1, 1)) * 0.6 + 70 * np.cos((x + y) / (fx + fy)),
        ],
        -1,
    )
    for _ in range(4):  # hard-edged rectangles, some fully saturated
        y0, x0 = r.integers(0, max(h, 1)), r.integers(0, max(w, 1))
        y1, x1 = y0 + r.integers(1, max(h // 3, 2)), x0 + r.integers(1, max(w // 3, 2))
        img[y0:y1, x0:x1] = r.choice([0, 255, 128, 30], 3)
    img += r.normal(0, noise, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def noise_img(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def make(h, w, seed, kind=None):
    kind = kind or ("noise" if seed % 5 == 0 else "photo")
    return noise_img(h, w, seed) if kind == "noise" else photo(h, w, seed)


manifest = {k: [] for k in ["resize", "preprocess", "decode", "smart_resize", "mrope", "expand", "rows", "dataurl", "digest",
                            "pipeline", "spec"]}

# ---------------------------------------------------------------- resize

resize_pairs = []
for _ in range(160):  # random up/down scaling, both axes
    resize_pairs.append(((random.randint(1, 300), random.randint(1, 300)), (random.randint(1, 300), random.randint(1, 300))))
for _ in range(25):  # one axis unchanged (single pass)
    w, h = random.randint(1, 300), random.randint(1, 300)
    resize_pairs.append(((w, h), (w, random.randint(1, 300))))
    resize_pairs.append(((w, h), (random.randint(1, 300), h)))
resize_pairs += [
    ((1, 1), (5, 7)), ((7, 5), (1, 1)), ((1, 300), (1, 50)), ((300, 1), (40, 1)), ((1, 1), (1, 9)),
    ((2, 2), (300, 3)), ((640, 480), (32, 32)), ((33, 33), (32, 32)), ((31, 31), (32, 32)),
    # tall images: Pillow 12 resizes vertically first when h > 100 * w and h shrinks
    ((2, 201), (2, 100)), ((2, 201), (4, 100)), ((3, 1000), (5, 37)), ((1, 150), (1, 20)),
    ((5, 600), (3, 400)), ((4, 900), (32, 448)), ((2, 250), (3, 300)),  # last one grows: normal order
    ((3, 300), (4, 100)), ((3, 301), (4, 100)),  # h == 100 w (normal) vs just above (vertical first)
    ((1000, 3), (37, 5)), ((900, 4), (448, 32)),  # wide images: normal order
    ((1920, 1080), (480, 256)), ((1920, 1080), (672, 384)), ((1080, 1920), (256, 480)),
    ((2000, 1500), (416, 320)), ((1280, 720), (1344, 768)), ((200, 150), (320, 224)),
    ((50, 6000), (32, 3936)),  # smart_resize of a very tall image (vertical-first path)
]
for i, ((w, h), (ow, oh)) in enumerate(resize_pairs):
    a = make(h, w, 1000 + i)
    out = np.asarray(Image.fromarray(a).resize((ow, oh), resample=Image.BICUBIC))
    manifest["resize"].append({"in": [w, h], "out": [ow, oh], "src": put(a.tobytes()), "exp": put(out.tobytes())})

# ---------------------------------------------------------------- digest (python mirror of digest.cpp)


def digest(pixels: np.ndarray, grid) -> str:
    buf = np.ascontiguousarray(pixels, dtype="<f4").tobytes()
    C = 256 * 1024
    leaves = b"".join(hashlib.blake2b(buf[i : i + C], digest_size=32).digest() for i in range(0, len(buf), C))
    head = b"espetero.vision.px1" + struct.pack("<3iQ", *grid, len(buf))
    return hashlib.blake2b(head + leaves, digest_size=16).hexdigest()


# ---------------------------------------------------------------- preprocess


def spec_json(s: V.VisionSpec):
    return {"patch": s.patch, "temporal": s.temporal, "merge": s.merge, "mean": list(s.mean), "std": list(s.std),
            "min_pixels": s.min_pixels, "max_pixels": s.max_pixels}


CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
pre_cases = [
    ((1080, 1920), dict(max_pixels=131072)), ((1080, 1920), dict(max_pixels=262144)),
    ((1920, 1080), dict(max_pixels=131072)), ((600, 800), dict(max_pixels=65536)),
    ((100, 100), dict(max_pixels=131072)),  # upscaled to min_pixels
    ((256, 512), dict(max_pixels=131072)),  # already a multiple of 32 within budget: no resize
    ((6000, 50), dict(max_pixels=131072)),  # tall: vertical-first resize
    ((37, 3000), dict(max_pixels=131072)),  # wide
    ((480, 640), dict(max_pixels=1003520, patch=14)),  # Qwen2-VL: factor 28, upscale
    ((720, 1280), dict(max_pixels=262144, mean=CLIP_MEAN, std=CLIP_STD)),
    ((333, 517), dict(max_pixels=131072, mean=CLIP_MEAN, std=CLIP_STD, patch=14)),
    ((1500, 2000), dict(max_pixels=802816)),
    ((64, 64), dict(max_pixels=4096)),  # tiny budget: min_pixels follows it down
]
for i, ((h, w), kw) in enumerate(pre_cases):
    s = V.VisionSpec(min_pixels=min(65536, kw.get("max_pixels", 131072)), **kw)
    a = make(h, w, 5000 + i, "photo")
    pixels, grid = V.preprocess(Image.fromarray(a), s)
    assert pixels.dtype == np.float32
    manifest["preprocess"].append({"size": [w, h], "rgb": put(a.tobytes()), "spec": spec_json(s), "grid": list(grid),
                                   "pixels": put(pixels.tobytes()), "digest": digest(pixels, grid)})

# digest of odd sizes (chunk boundaries)
for n in [0, 1, 31, 32, 65535, 65536, 65537, 200000]:
    x = rng.standard_normal(n).astype(np.float32)
    g = (1, n % 7, n % 5)
    manifest["digest"].append({"pixels": put(x.tobytes()), "grid": list(g), "digest": digest(x, g)})

# ---------------------------------------------------------------- decode


def png_chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)


def raw_png(arr, color_type, bit_depth, extra=b"", interlace=False, width=None):
    """Handwritten PNG (bit depths / chunks PIL cannot write). `arr` rows are packed samples."""
    h = arr.shape[0]
    w = width if width is not None else arr.shape[1]
    ihdr = struct.pack(">IIBBBBB", w, h, bit_depth, color_type, 0, 0, 1 if interlace else 0)
    if not interlace:
        raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    else:  # Adam7 over whole-byte samples (8/16-bit)
        bpp = arr.shape[2] * arr.dtype.itemsize if arr.ndim == 3 else arr.dtype.itemsize
        raw = b""
        for x0, y0, dx, dy in [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]:
            sub = arr[y0::dy, x0::dx]
            if sub.size == 0:
                continue
            raw += b"".join(b"\x00" + sub[y].tobytes() for y in range(sub.shape[0]))
        del bpp
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + extra + png_chunk(b"IDAT", zlib.compress(raw)) + png_chunk(b"IEND", b"")


def pil_bytes(img, fmt, **kw):
    b = io.BytesIO()
    img.save(b, fmt, **kw)
    return b.getvalue()


def gif_raw(width, height, frames, global_pal=None, gce=None):
    """Handwritten GIF: frames = [(x0, y0, w, h, indices(h,w), local_pal or None, interlace, min_code)]."""
    out = b"GIF89a" + struct.pack("<HH", width, height)
    if global_pal is not None:
        bits = max(1, math.ceil(math.log2(max(2, len(global_pal) // 3))))
        pal = global_pal + b"\x00" * (3 * (1 << bits) - len(global_pal))
        out += bytes([0x80 | (bits - 1), 0, 0]) + pal
    else:
        out += bytes([0, 0, 0])
    for x0, y0, w, h, idx, local, interlace, min_code in frames:
        if gce is not None:
            out += b"\x21\xf9\x04" + bytes([gce[0], 0, 0, gce[1]]) + b"\x00"
        flags = 0x40 if interlace else 0
        lp = b""
        if local is not None:
            bits = max(1, math.ceil(math.log2(max(2, len(local) // 3))))
            lp = local + b"\x00" * (3 * (1 << bits) - len(local))
            flags |= 0x80 | (bits - 1)
        out += b"\x2c" + struct.pack("<HHHH", x0, y0, w, h) + bytes([flags]) + lp
        rows = idx
        if interlace:
            order = list(range(0, h, 8)) + list(range(4, h, 8)) + list(range(2, h, 4)) + list(range(1, h, 2))
            rows = idx[order]
        out += bytes([min_code]) + lzw_encode(rows.ravel().tolist(), min_code) + b"\x00"
    return out + b"\x3b"


def lzw_encode(data, min_code):
    """Plain GIF LZW (clear at start, table reset when full)."""
    clear, end = 1 << min_code, (1 << min_code) + 1
    bits, buf, nbits, out = min_code + 1, 0, 0, bytearray()

    def emit(c):
        nonlocal buf, nbits
        buf |= c << nbits
        nbits += bits
        while nbits >= 8:
            out.append(buf & 255)
            buf >>= 8
            nbits -= 8

    table = {(i,): i for i in range(clear)}
    nxt = end + 1
    emit(clear)
    cur = ()
    for v in data:
        cand = cur + (v,)
        if cand in table:
            cur = cand
            continue
        emit(table[cur])
        if nxt < 4096:
            table[cand] = nxt
            nxt += 1
        else:
            emit(clear)
            table = {(i,): i for i in range(clear)}
            nxt = end + 1
            bits = min_code + 1
        cur = (v,)
        if nxt > (1 << bits) and bits < 12:  # the decoder widens when its table reaches 1 << bits
            bits += 1
    if cur:
        emit(table[cur])
    emit(end)
    if nbits:
        out.append(buf & 255)
    blocks = b"".join(bytes([len(out[i : i + 255])]) + bytes(out[i : i + 255]) for i in range(0, len(out), 255))
    return blocks


def cjpeg(arr, *args):
    """JPEG via IJG cjpeg (restart markers, sampling factors, arithmetic coding, RGB)."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.ppm")
        Image.fromarray(arr).save(src)
        return subprocess.run(["cjpeg", *args, src], check=True, capture_output=True).stdout


def add_decode(tag, data, fmt, expect_exact=True):
    try:
        im = Image.open(io.BytesIO(data))
        mode = im.mode
        rgb = np.asarray(im.convert("RGB"))
        manifest["decode"].append({"tag": tag, "format": fmt, "file": put(data), "mode": mode, "size": [rgb.shape[1], rgb.shape[0]],
                                   "exp": put(rgb.tobytes()), "exact": expect_exact, "error": None})
    except Exception as e:  # PIL rejects it: record the file anyway
        manifest["decode"].append({"tag": tag, "format": fmt, "file": put(data), "mode": None, "size": None, "exp": None,
                                   "exact": False, "error": f"{type(e).__name__}: {e}"})


a = photo(97, 131, 1)
alpha = rng.integers(0, 256, a.shape[:2], dtype=np.uint8)
alpha[:10, :10] = 0
rgba = np.dstack([a, alpha])
P = Image.fromarray(a).quantize(200)
# -- PNG
add_decode("png rgb8", pil_bytes(Image.fromarray(a), "PNG"), "png")
add_decode("png rgba8", pil_bytes(Image.fromarray(rgba, "RGBA"), "PNG"), "png")
add_decode("png l8", pil_bytes(Image.fromarray(a[..., 0]), "PNG"), "png")
add_decode("png la8", pil_bytes(Image.fromarray(rgba[..., [0, 3]], "LA"), "PNG"), "png")
add_decode("png p8", pil_bytes(P, "PNG"), "png")
add_decode("png p8 trns", pil_bytes(P, "PNG", transparency=5), "png")
add_decode("png p8 trns table", pil_bytes(P, "PNG", transparency=bytes(range(0, 200, 2))), "png")
for bits, ncol in [(1, 2), (2, 4), (4, 16)]:
    add_decode(f"png p{bits}", pil_bytes(Image.fromarray(a).quantize(ncol), "PNG", bits=bits), "png")
add_decode("png 1bit", pil_bytes(Image.fromarray(a).convert("1"), "PNG"), "png")
add_decode("png l2", pil_bytes(Image.fromarray((a[..., 1] // 64 * 85).astype(np.uint8)), "PNG", bits=2), "png")
add_decode("png l4", pil_bytes(Image.fromarray((a[..., 1] // 16 * 17).astype(np.uint8)), "PNG", bits=4), "png")
add_decode("png rgb8 trns", pil_bytes(Image.fromarray(a), "PNG", transparency=tuple(int(v) for v in a[0, 0])), "png")
add_decode("png l8 trns", pil_bytes(Image.fromarray(a[..., 0]), "PNG", transparency=int(a[0, 0, 0])), "png")
a16 = rng.integers(0, 65536, (61, 77, 3)).astype(">u2")
add_decode("png rgb16", raw_png(a16, 2, 16), "png")
add_decode("png rgba16", raw_png(rng.integers(0, 65536, (61, 77, 4)).astype(">u2"), 6, 16), "png")
g16 = rng.integers(0, 65536, (61, 77)).astype(">u2")
g16[:20] = rng.integers(0, 300, (20, 77))  # I;16 -> RGB clips at 255
add_decode("png l16", raw_png(g16, 0, 16), "png")
add_decode("png l16 trns", raw_png(g16, 0, 16, png_chunk(b"tRNS", struct.pack(">H", int(g16[0, 0])))), "png")
add_decode("png la16", raw_png(rng.integers(0, 65536, (61, 77, 2)).astype(">u2"), 4, 16), "png")
add_decode("png rgb16 trns", raw_png(a16, 2, 16, png_chunk(b"tRNS", a16[0, 0].astype(">u2").tobytes())), "png")
add_decode("png gAMA", raw_png(a, 2, 8, png_chunk(b"gAMA", struct.pack(">I", 100000))), "png")
add_decode("png sRGB", raw_png(a, 2, 8, png_chunk(b"sRGB", b"\x00")), "png")
add_decode("png cHRM", raw_png(a, 2, 8, png_chunk(b"cHRM", struct.pack(">8I", 64000, 33000, 30000, 60000, 15000, 6000, 31270, 32900))), "png")
for prof in ["Display P3.icc", "AdobeRGB1998.icc", "Generic Gray Gamma 2.2 Profile.icc"]:
    path = "/System/Library/ColorSync/Profiles/" + prof
    if os.path.exists(path):
        icc = open(path, "rb").read()
        src = Image.fromarray(a[..., 0]) if "Gray" in prof else Image.fromarray(a)
        add_decode(f"png icc {prof}", pil_bytes(src, "PNG", icc_profile=icc), "png")
add_decode("png interlaced rgb8", raw_png(a, 2, 8, interlace=True), "png")
add_decode("png interlaced rgba16", raw_png(rng.integers(0, 65536, (13, 11, 4)).astype(">u2"), 6, 16, interlace=True), "png")
add_decode("png 1x1", pil_bytes(Image.fromarray(a[:1, :1]), "PNG"), "png")
add_decode("png 1x300", pil_bytes(Image.fromarray(photo(300, 1, 3)), "PNG"), "png")
add_decode("png big", pil_bytes(Image.fromarray(photo(1080, 1920, 4)), "PNG"), "png")
# Palette of 2 entries, index 7 used: PIL reads it (black), libpng-based readers may not.
ihdr = struct.pack(">IIBBBBB", 4, 1, 8, 3, 0, 0, 0)
short = (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + png_chunk(b"PLTE", bytes([10, 20, 30, 40, 50, 60]))
         + png_chunk(b"IDAT", zlib.compress(b"\x00" + bytes([0, 1, 2, 7]))) + png_chunk(b"IEND", b""))
add_decode("png short palette", short, "png", expect_exact=False)
# -- GIF
add_decode("gif p256", pil_bytes(Image.fromarray(a).quantize(256), "GIF"), "gif")
add_decode("gif p256 noninterlaced", pil_bytes(Image.fromarray(a).quantize(256), "GIF", interlace=False), "gif")
add_decode("gif trns", pil_bytes(Image.fromarray(a).quantize(128), "GIF", transparency=3), "gif")
add_decode("gif gray", pil_bytes(Image.fromarray(a[..., 1]), "GIF"), "gif")
add_decode("gif 2 colours", pil_bytes(Image.fromarray(a).quantize(2), "GIF"), "gif")
frames = [Image.fromarray(photo(50, 60, 10 + k)).quantize(64) for k in range(3)]
add_decode("gif animated", pil_bytes(frames[0], "GIF", save_all=True, append_images=frames[1:], duration=100, loop=0), "gif")
idx = rng.integers(0, 16, (23, 31)).astype(np.uint8)
pal16 = bytes(rng.integers(0, 256, 48, dtype=np.uint8))
pal4 = bytes(rng.integers(0, 256, 12, dtype=np.uint8))
add_decode("gif offset frame", gif_raw(40, 30, [(5, 3, 31, 23, idx, None, False, 4)], global_pal=pal16), "gif")
add_decode("gif offset frame trns", gif_raw(40, 30, [(5, 3, 31, 23, idx, None, False, 4)], global_pal=pal16, gce=(1, 7)), "gif")
add_decode("gif frame past screen", gif_raw(20, 10, [(5, 3, 31, 23, idx, None, True, 4)], global_pal=pal16), "gif")
add_decode("gif local palette", gif_raw(31, 23, [(0, 0, 31, 23, idx, pal16[::-1], False, 4)], global_pal=pal16), "gif")
add_decode("gif no palette", gif_raw(31, 23, [(0, 0, 31, 23, idx * 13, None, False, 8)]), "gif")
add_decode("gif short palette", gif_raw(31, 23, [(0, 0, 31, 23, idx * 13, None, False, 8)], global_pal=pal4), "gif")
big_idx = rng.integers(0, 256, (200, 300)).astype(np.uint8)  # table resets (4096 codes)
add_decode("gif lzw reset", gif_raw(300, 200, [(0, 0, 300, 200, big_idx, None, True, 8)],
                                    global_pal=bytes(rng.integers(0, 256, 768, dtype=np.uint8))), "gif")
gray_pal = bytes(v for i in range(256) for v in (i, i, i))
add_decode("gif identity grey palette", gif_raw(31, 23, [(0, 0, 31, 23, idx * 13, None, False, 8)], global_pal=gray_pal), "gif")
# -- JPEG (libjpeg-turbo in PIL; our decoder must match)
jp = photo(97, 131, 7)
for q, sub in [(75, 2), (90, 0), (90, 1), (95, 2), (50, 2), (100, 0)]:
    add_decode(f"jpeg q{q} sub{sub}", pil_bytes(Image.fromarray(jp), "JPEG", quality=q, subsampling=sub), "jpeg")
add_decode("jpeg gray", pil_bytes(Image.fromarray(jp[..., 0]), "JPEG", quality=85), "jpeg")
add_decode("jpeg progressive 420", pil_bytes(Image.fromarray(jp), "JPEG", quality=80, progressive=True), "jpeg")
add_decode("jpeg progressive 444", pil_bytes(Image.fromarray(jp), "JPEG", quality=92, progressive=True, subsampling=0), "jpeg")
add_decode("jpeg progressive gray", pil_bytes(Image.fromarray(jp[..., 1]), "JPEG", quality=70, progressive=True), "jpeg")
add_decode("jpeg optimized", pil_bytes(Image.fromarray(jp), "JPEG", quality=85, optimize=True), "jpeg")
add_decode("jpeg cmyk", pil_bytes(Image.fromarray(jp).convert("CMYK"), "JPEG", quality=90), "jpeg")
for (h, w) in [(1, 1), (7, 9), (8, 8), (16, 16), (17, 33), (1, 200), (200, 1), (15, 17)]:
    add_decode(f"jpeg {w}x{h}", pil_bytes(Image.fromarray(photo(h, w, h * 7 + w)), "JPEG", quality=88), "jpeg")
icc = open("/System/Library/ColorSync/Profiles/Display P3.icc", "rb").read()
add_decode("jpeg icc", pil_bytes(Image.fromarray(jp), "JPEG", quality=90, icc_profile=icc), "jpeg")
exif = Image.Exif()
exif[0x0112] = 6  # orientation: rotate 90 (PIL ignores it)
add_decode("jpeg exif orientation", pil_bytes(Image.fromarray(jp), "JPEG", quality=90, exif=exif), "jpeg")
bigj = photo(1080, 1920, 11)
add_decode("jpeg 1080p q90", pil_bytes(Image.fromarray(bigj), "JPEG", quality=90), "jpeg")
add_decode("jpeg 1080p q75 progressive", pil_bytes(Image.fromarray(bigj), "JPEG", quality=75, progressive=True), "jpeg")
# Scan scripts: non-interleaved sequential scans, and a progressive file whose
# AC coefficients are never refined to full precision (libjpeg then applies
# inter-block smoothing, which we leave to ImageIO: reported, not exact).
SCANS_SEQ = "0;\n1;\n2;\n"
SCANS_INCOMPLETE = "0,1,2: 0-0, 0, 1;\n0: 1-5, 0, 2;\n2: 1-63, 0, 1;\n1: 1-63, 0, 1;\n0: 6-63, 0, 2;\n0: 1-63, 2, 1;\n0,1,2: 0-0, 1, 0;\n"


def cjpeg_scans(arr, script, *args):
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "scans.txt")
        open(sp, "w").write(script)
        return cjpeg(arr, "-scans", sp, *args)


add_decode("jpeg cjpeg sequential multi-scan", cjpeg_scans(jp, SCANS_SEQ), "jpeg")
add_decode("jpeg cjpeg progressive unrefined (smoothing)", cjpeg_scans(jp, SCANS_INCOMPLETE), "jpeg", expect_exact=False)
add_decode("jpeg cjpeg quality 1 (16-bit DQT)", cjpeg(jp, "-quality", "1"), "jpeg")
base = pil_bytes(Image.fromarray(jp), "JPEG", quality=85)
sos = base.index(b"\xff\xda")
add_decode("jpeg fill bytes before markers", base[:sos] + b"\xff\xff\xff" + base[sos:].replace(b"\xff\xd9", b"\xff\xff\xd9"), "jpeg")
add_decode("jpeg missing EOI", base[:-2], "jpeg")
add_decode("jpeg trailing garbage", base + b"garbage after EOI" * 10, "jpeg")
add_decode("jpeg junk before marker", base[:sos] + b"\x12\x34" + base[sos:], "jpeg")
for tag, args in [("restart rows", ["-restart", "1"]), ("restart blocks", ["-restart", "5B"]),
                  ("sample 1x2", ["-sample", "1x2,1x1,1x1"]), ("sample 2x1", ["-sample", "2x1,1x1,1x1"]),
                  ("sample 4x1", ["-sample", "4x1,1x1,1x1"]), ("sample 2x2 1x2", ["-sample", "2x2,1x2,1x1"]),
                  ("sample 1x1 1x1 2x2", ["-sample", "1x1,1x1,2x2"]), ("rgb", ["-rgb"]),
                  ("progressive restart", ["-progressive", "-restart", "3B"]),
                  ("quality 97 444", ["-quality", "97", "-sample", "1x1,1x1,1x1"]), ("baseline q20", ["-quality", "20"])]:
    try:
        add_decode(f"jpeg cjpeg {tag}", cjpeg(jp, *args), "jpeg")
    except Exception as e:  # noqa: BLE001
        print("cjpeg failed:", tag, e, file=sys.stderr)
# Large images exercise the fast paths: band-pipelined reconstruction
# (single-scan files >= 256K pixels, every upsampling method), parallel
# entropy decoding of restart intervals (>= 2048 MCUs), Apple's encoder.
lj = photo(771, 1027, 21)
add_decode("jpeg large 420 pipelined", pil_bytes(Image.fromarray(lj), "JPEG", quality=90), "jpeg")
add_decode("jpeg large 444 pipelined", pil_bytes(Image.fromarray(lj[:768, :1024]), "JPEG", quality=85, subsampling=0), "jpeg")
add_decode("jpeg large 422 pipelined", pil_bytes(Image.fromarray(lj), "JPEG", quality=80, subsampling=1), "jpeg")
add_decode("jpeg large gray pipelined", pil_bytes(Image.fromarray(lj[..., 1]), "JPEG", quality=90), "jpeg")
add_decode("jpeg large cmyk pipelined", pil_bytes(Image.fromarray(lj).convert("CMYK"), "JPEG", quality=90), "jpeg")
add_decode("jpeg large progressive", pil_bytes(Image.fromarray(lj), "JPEG", quality=85, progressive=True), "jpeg")
for tag, args in [("sample 1x2", ["-sample", "1x2,1x1,1x1"]), ("sample 4x1", ["-sample", "4x1,1x1,1x1"]),
                  ("restart rows", ["-restart", "1"]), ("restart 7 blocks 444", ["-restart", "7B", "-sample", "1x1,1x1,1x1"]),
                  ("progressive restart", ["-progressive", "-restart", "2"]), ("restart 1 block gray", ["-grayscale", "-restart", "1B"]),
                  ("gray sampled 2x2", ["-grayscale", "-sample", "2x2"]), ("gray sampled 2x2 progressive", ["-grayscale", "-sample", "2x2", "-progressive"])]:
    add_decode(f"jpeg large cjpeg {tag}", cjpeg(lj, *args), "jpeg")
with tempfile.TemporaryDirectory() as d:  # Apple's encoder: a restart marker per MCU row
    Image.fromarray(lj).save(os.path.join(d, "a.png"))
    r = subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "85", os.path.join(d, "a.png"), "--out", os.path.join(d, "a.jpg")], capture_output=True)
    if r.returncode == 0:
        add_decode("jpeg large apple encoder (sips)", open(os.path.join(d, "a.jpg"), "rb").read(), "jpeg")
# Arithmetic coding (never produced by cameras or browsers): ImageIO fallback.
add_decode("jpeg cjpeg arithmetic", cjpeg(jp, "-arithmetic"), "jpeg", expect_exact=False)
add_decode("jpeg cjpeg arithmetic progressive", cjpeg(jp, "-arithmetic", "-progressive"), "jpeg", expect_exact=False)
# -- other formats (ImageIO)
add_decode("bmp rgb", pil_bytes(Image.fromarray(a), "BMP"), "bmp")
add_decode("bmp p8", pil_bytes(P, "BMP"), "bmp")
add_decode("bmp 1bit", pil_bytes(Image.fromarray(a).convert("1"), "BMP"), "bmp")
add_decode("bmp rgba", pil_bytes(Image.fromarray(rgba, "RGBA"), "BMP"), "bmp")
add_decode("tiff rgb", pil_bytes(Image.fromarray(a), "TIFF"), "tiff")
add_decode("tiff rgb lzw", pil_bytes(Image.fromarray(a), "TIFF", compression="tiff_lzw"), "tiff")
add_decode("tiff rgba", pil_bytes(Image.fromarray(rgba, "RGBA"), "TIFF"), "tiff")
add_decode("tiff gray", pil_bytes(Image.fromarray(a[..., 2]), "TIFF"), "tiff")
add_decode("tiff gray16", pil_bytes(Image.fromarray(g16.astype(np.uint16), "I;16"), "TIFF"), "tiff")
add_decode("tiff cmyk", pil_bytes(Image.fromarray(a).convert("CMYK"), "TIFF"), "tiff")
add_decode("tiff palette", pil_bytes(P, "TIFF"), "tiff")
add_decode("webp lossless", pil_bytes(Image.fromarray(a), "WEBP", lossless=True), "webp")
add_decode("webp lossy", pil_bytes(Image.fromarray(a), "WEBP", quality=80), "webp")
add_decode("webp lossless rgba", pil_bytes(Image.fromarray(rgba, "RGBA"), "WEBP", lossless=True), "webp", expect_exact=False)
add_decode("webp lossy rgba", pil_bytes(Image.fromarray(rgba, "RGBA"), "WEBP", quality=80), "webp", expect_exact=False)
try:
    add_decode("avif", pil_bytes(Image.fromarray(a), "AVIF", quality=80), "avif", expect_exact=False)
except Exception as e:  # noqa: BLE001
    print("avif failed:", e, file=sys.stderr)
with tempfile.TemporaryDirectory() as d:  # HEIC: PIL cannot read it; ImageIO must decode it
    Image.fromarray(a).save(os.path.join(d, "a.png"))
    r = subprocess.run(["sips", "-s", "format", "heic", os.path.join(d, "a.png"), "--out", os.path.join(d, "a.heic")], capture_output=True)
    if r.returncode == 0:
        add_decode("heic (no PIL reference)", open(os.path.join(d, "a.heic"), "rb").read(), "heic", expect_exact=False)

# ---------------------------------------------------------------- smart_resize

sr = [(16, 16, 32), (48, 80, 32), (80, 48, 32), (1, 1, 32), (1, 200, 32), (1, 201, 32), (200, 1, 32)]
for _ in range(3000):
    h = random.choice([random.randint(1, 64), random.randint(1, 5000), random.randint(1, 200) * 16])
    w = random.choice([random.randint(1, 64), random.randint(1, 5000), h * random.randint(1, 250) // 7 + 1])
    sr.append((h, w, random.choice([28, 32])))
for h, w, f in sr:
    for mx in [131072, 262144, 65536, 4096, 1003520, 16384 * 28 * 28]:
        mn = min(65536, mx)
        try:
            exp = list(V.smart_resize(h, w, f, mn, mx))
        except ValueError:
            exp = None
        manifest["smart_resize"].append([h, w, f, mn, mx, exp])

# ---------------------------------------------------------------- mrope / expand / rows

spec = V.VisionSpec()


def expand(prefix, grids):  # jevfire.metal.engine.Engine._expand, verbatim logic
    out, g = [], 0
    for t in prefix:
        if t == spec.image_token_id:
            out += [t] * V.image_tokens(grids[g], spec.merge)
            g += 1
        else:
            out.append(t)
    if g != len(grids):
        raise ValueError("Image placeholders and images differ in number")
    return out


for case in range(300):
    n_img = case % 4
    grids = [(1, 2 * random.randint(1, 24), 2 * random.randint(1, 24)) for _ in range(n_img)]
    prefix = [random.randint(0, 5000) for _ in range(random.randint(0, 6))]
    for g in range(n_img):
        prefix += [spec.vision_start_id, spec.image_token_id, spec.vision_end_id]
        prefix += [random.randint(0, 5000) for _ in range(random.randint(0, 5))]
    ids = expand(prefix, grids)
    start = random.choice([0, 0, random.randint(1, 1000)])
    kind = "ok"
    g_used = grids
    if case % 23 == 5 and n_img:
        g_used = grids[:-1]  # more image blocks than grids
        kind = "fewer grids"
    elif case % 23 == 7:
        g_used = grids + [(1, 4, 4)]  # grids left over
        kind = "extra grid"
    elif case % 23 == 11 and n_img:
        g_used = [(1, grids[0][1] + 2, grids[0][2])] + grids[1:]  # run shorter than grid
        kind = "run mismatch"
    try:
        pos, nxt = V.mrope_positions(ids, g_used, spec, start)
        manifest["mrope"].append({"ids": ids, "grids": [list(g) for g in g_used], "start": start, "pos": pos.ravel().tolist(),
                                  "next": nxt, "error": None, "kind": kind})
    except ValueError as e:
        manifest["mrope"].append({"ids": ids, "grids": [list(g) for g in g_used], "start": start, "pos": None, "next": None,
                                  "error": str(e), "kind": kind})
    try:
        exp_ids = expand(prefix, g_used)
        manifest["expand"].append({"prefix": prefix, "grids": [list(g) for g in g_used], "ids": exp_ids, "error": None})
    except (ValueError, IndexError) as e:
        manifest["expand"].append({"prefix": prefix, "grids": [list(g) for g in g_used], "ids": None, "error": str(e)})
    try:
        imgs = [(None, g) for g in g_used]
        rows = [im.row for im in V.native_images(ids, imgs, spec)]
        manifest["rows"].append({"ids": ids, "grids": [list(g) for g in g_used], "rows": rows, "error": None})
    except (ValueError, IndexError) as e:
        manifest["rows"].append({"ids": ids, "grids": [list(g) for g in g_used], "rows": None, "error": str(e)})

# ---------------------------------------------------------------- data URLs

png = pil_bytes(Image.fromarray(a[:8, :8]), "PNG")
b64 = base64.b64encode(png).decode()
urls = [
    "data:image/png;base64," + b64,
    "data:image/png;base64," + "\n".join(b64[i : i + 76] for i in range(0, len(b64), 76)),  # MIME line breaks
    "data:image/png;base64," + b64.rstrip("="),  # missing padding
    "data:image/png;base64," + b64 + "trailing junk after padding" if b64.endswith("=") else "data:image/png;base64," + b64 + "==",
    "data:image/png;base64," + b64.replace("+", "-").replace("/", "_"),  # url-safe alphabet is skipped
    "data:image/png," + b64,  # not base64
    "data:image/png;base64",  # no comma
    "data:;base64," + b64,
    "data:image/png;charset=utf-8;base64," + b64,
    "data:image/png;base64, " + b64 + " ",
    "data:image/png;base64,QUJD=",  # pad after full quad (ignored)
    "data:image/png;base64,QQ",  # 1 byte, missing padding
    "data:image/png;base64,QQ==",
    "data:image/png;base64,Q===",
    "data:image/png;base64,QUI=",
    "data:image/png;base64,Q=U=I",
    "data:image/png;base64,é" + b64,
]
for u in urls:
    head, _, payload = u.partition(",")
    if ";base64" not in head:
        manifest["dataurl"].append({"url": u, "bytes": None, "error": "only base64 data URLs are supported"})
        continue
    try:
        manifest["dataurl"].append({"url": u, "bytes": put(base64.b64decode(payload)), "error": None})
    except (ValueError, Exception) as e:  # binascii.Error is a ValueError
        manifest["dataurl"].append({"url": u, "bytes": None, "error": str(e)})

# ---------------------------------------------------------------- end to end

# V.preprocess straight from encoded bytes / data URLs (PIL decode + resize +
# normalise + patchify) against load_rgb + preprocess.
e2e = [
    ("jpeg 1080p", pil_bytes(Image.fromarray(photo(1080, 1920, 31)), "JPEG", quality=90), dict(max_pixels=131072)),
    ("jpeg 1080p 262144", pil_bytes(Image.fromarray(photo(1080, 1920, 31)), "JPEG", quality=90), dict(max_pixels=262144)),
    ("png rgba", pil_bytes(Image.fromarray(np.dstack([photo(300, 400, 32), rng.integers(0, 256, (300, 400), dtype=np.uint8)]), "RGBA"), "PNG"), dict()),
    ("gif transparency", pil_bytes(Image.fromarray(photo(200, 150, 33)).quantize(64), "GIF", transparency=7), dict()),
    ("png l16", raw_png(g16, 0, 16), dict()),
    ("jpeg cmyk", pil_bytes(Image.fromarray(photo(333, 222, 34)).convert("CMYK"), "JPEG", quality=90), dict(patch=14)),
    ("jpeg progressive tall", pil_bytes(Image.fromarray(photo(3000, 40, 35)), "JPEG", quality=85, progressive=True), dict()),
]
for tag, data, kw in e2e:
    sp = V.VisionSpec(min_pixels=min(65536, kw.get("max_pixels", 131072)), **kw)
    for src_kind in (["bytes", "dataurl"] if tag == "jpeg 1080p" else ["bytes"]):
        src = data if src_kind == "bytes" else "data:image/jpeg;base64," + base64.b64encode(data).decode()
        pixels, grid = V.preprocess(src, sp)
        manifest["pipeline"].append({"tag": f"{tag} ({src_kind})", "src": put(src if isinstance(src, bytes) else src.encode()),
                                     "spec": spec_json(sp), "grid": list(grid), "pixels": put(pixels.tobytes()),
                                     "digest": digest(pixels, grid)})

# VisionSpec.from_model on the Qwen checkpoints listed in $VISION_REF_MODELS
# (os.pathsep-separated directories), else on public ones in the HF hub cache.
def _hub(repo):
    cache = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    snaps = sorted(glob.glob(os.path.join(cache, "models--" + repo.replace("/", "--"), "snapshots", "*")))
    return snaps[-1] if snaps else ""


_dirs = [d for d in os.environ.get("VISION_REF_MODELS", "").split(os.pathsep) if d] or [
    _hub(r) for r in ("mlx-community/Qwen3.5-0.8B-4bit", "mlx-community/Qwen3.8-27B-4bit",
                      "mlx-community/Qwen3-VL-2B-Instruct-4bit", "mlx-community/Qwen3.6-35B-A3B-8bit")]
for d in _dirs:
    if os.path.exists(os.path.join(d, "config.json")):
        for mx in (131072, 50000):
            sp = V.VisionSpec.from_model(d, mx)
            manifest["spec"].append({"dir": d, "max_pixels": mx, **spec_json(sp), "image_token_id": sp.image_token_id,
                                     "vision_start_id": sp.vision_start_id, "vision_end_id": sp.vision_end_id})

blob.close()
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f)
print({k: len(v) for k, v in manifest.items()}, f"{offset / 1e6:.1f} MB")
