// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Image limits without a model (make test):
//
//  * crafted headers that declare huge images (the 26-byte GIF of 65535 x
//    2730, a GIF frame far outside its screen, a 200000^2 PNG, a JPEG SOF of
//    65000^2) are measured by probe_image and refused by decode_image without
//    allocating their pixels (peak memory checked);
//  * the JPEG marker walk skips a SOF hidden inside another segment, and EXIF
//    orientations 5-8 swap the probed sides as decode_image's rotation does;
//  * real files of every decoder path (encoded here with ImageIO: PNG, JPEG,
//    GIF, TIFF, BMP, HEIC when available) probe to exactly the size
//    decode_image returns, and grid_for() of that size is preprocess()'s grid,
//    so the engine's token count from headers is the processor's own.
//
//   build/vision_limits_test
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <sys/resource.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/vision/vision.h"

using namespace espetero::vision;

namespace {

int checks = 0, failures = 0;

void check(bool ok, const std::string& what) {
  ++checks;
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
  }
}

long peak_mb() {
  rusage u{};
  ::getrusage(RUSAGE_SELF, &u);
  return long(u.ru_maxrss >> 20);  // bytes on macOS
}

void le16(std::string& b, int v) {
  b.push_back(char(v & 255));
  b.push_back(char((v >> 8) & 255));
}
void be16(std::string& b, int v) {
  b.push_back(char((v >> 8) & 255));
  b.push_back(char(v & 255));
}
void be32(std::string& b, uint32_t v) {
  for (int s = 24; s >= 0; s -= 8) b.push_back(char((v >> s) & 255));
}

// GIF89a: logical screen sw x sh, one frame (x0, y0, fw, fh) with no LZW data.
std::string gif(int sw, int sh, int x0, int y0, int fw, int fh) {
  std::string b = "GIF89a";
  le16(b, sw);
  le16(b, sh);
  b += std::string("\0\0\0", 3);  // no global table, background, aspect
  b += ',';
  le16(b, x0);
  le16(b, y0);
  le16(b, fw);
  le16(b, fh);
  b += '\0';      // no local table, not interlaced
  b += '\x02';    // LZW minimum code size
  b += '\0';      // no data sub-blocks
  b += ';';
  return b;
}

// SOI, optional APP1 payload, SOF0 (3 components) of w x h, EOI.
std::string jpeg(int w, int h, const std::string& app1 = "") {
  std::string b = "\xFF\xD8";
  if (!app1.empty()) {
    b += "\xFF\xE1";
    be16(b, int(app1.size()) + 2);
    b += app1;
  }
  b += "\xFF\xC0";
  be16(b, 17);
  b += '\x08';
  be16(b, h);
  be16(b, w);
  b += '\x03';
  for (int c = 1; c <= 3; ++c) b += std::string{char(c), '\x11', '\0'};
  b += "\xFF\xD9";
  return b;
}

// APP1 "Exif" payload with IFD0 holding only the orientation tag.
std::string exif(int orientation) {
  std::string t = "Exif";
  t += std::string("\0\0", 2);
  t += "MM";
  be16(t, 42);
  be32(t, 8);
  be16(t, 1);  // one entry
  be16(t, 0x0112);
  be16(t, 3);  // SHORT
  be32(t, 1);
  be16(t, orientation);
  be16(t, 0);
  be32(t, 0);  // no next IFD
  return t;
}

std::string png_header(uint32_t w, uint32_t h) {
  std::string b = "\x89PNG\r\n\x1a\n";
  be32(b, 13);
  b += "IHDR";
  be32(b, w);
  be32(b, h);
  b += std::string("\x08\x02\0\0\0", 5);
  be32(b, 0);  // CRC (wrong on purpose: nothing may get far enough to care)
  return b;
}

std::string size_str(int64_t w, int64_t h) { return std::to_string(w) + "x" + std::to_string(h); }

// Declared sizes probe as declared; decoding them fails at once.
void test_bombs() {
  struct Case {
    const char* what;
    std::string bytes;
    int64_t w, h;
  };
  const std::vector<Case> cases = {
      {"GIF screen 65535x2730 (179 MP from 26 bytes)", gif(65535, 2730, 0, 0, 1, 1), 65535, 2730},
      {"GIF frame 65000x65000 on a 1x1 screen", gif(1, 1, 0, 0, 65000, 65000), 65000, 65000},
      {"GIF frame placed at 60000,60000", gif(10, 10, 60000, 60000, 5000, 5000), 65000, 65000},
      {"PNG IHDR 200000x200000", png_header(200000, 200000), 200000, 200000},
      {"PNG IHDR 4294967295x1", png_header(0xFFFFFFFFu, 1), 0xFFFFFFFFll, 1},
      {"JPEG SOF 65000x65000", jpeg(65000, 65000), 65000, 65000},
      {"JPEG SOF 9000x8000 (72 MP, under PIL's limit)", jpeg(9000, 8000), 9000, 8000},
  };
  const long before = peak_mb();
  for (const Case& c : cases) {
    int64_t w = 0, h = 0;
    std::string err;
    check(probe_image(c.bytes, w, h, err) && w == c.w && h == c.h,
          std::string(c.what) + ": probed " + size_str(w, h) + " " + err);
    RGBImage img;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = decode_image(c.bytes, img, err);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    check(!ok && img.rgb.empty(), std::string(c.what) + ": decode must fail");
    check(ms < 200, std::string(c.what) + ": refused in " + std::to_string(ms) + " ms");
    std::printf("  %-48s probed %-18s decode: %s (%.2f ms)\n", c.what, size_str(w, h).c_str(), err.c_str(), ms);
  }
  const long grown = peak_mb() - before;
  check(grown < 64, "peak memory grew by " + std::to_string(grown) + " MB refusing the bombs");
  std::printf("  peak memory while refusing them: +%ld MB\n", grown);
}

void test_headers() {
  int64_t w = 0, h = 0;
  std::string err;
  // A SOF-like byte run inside an APP1 payload is not the frame: segments are
  // skipped by their length, as jpeg.cpp walks them.
  std::string fake = "XMP\0";
  fake += "\xFF\xC0";
  be16(fake, 17);
  fake += '\x08';
  be16(fake, 60000);
  be16(fake, 60000);
  check(probe_image(jpeg(100, 50, fake), w, h, err) && w == 100 && h == 50, "JPEG: SOF inside APP1 skipped, got " +
                                                                                size_str(w, h));
  // EXIF orientation: 5-8 transpose (decode_image rotates), 1-4 keep the sides.
  for (int o = 1; o <= 8; ++o) {
    const bool swap = o >= 5;
    check(probe_image(jpeg(300, 200, exif(o)), w, h, err) && w == (swap ? 200 : 300) && h == (swap ? 300 : 200),
          "JPEG orientation " + std::to_string(o) + ": got " + size_str(w, h));
  }
  set_exif_orientation(false);
  check(probe_image(jpeg(300, 200, exif(6)), w, h, err) && w == 300 && h == 200, "--no-exif-rotate keeps the sides");
  set_exif_orientation(true);
  // Unreadable headers: no size (decoding then gives its own verdict).
  for (const std::string& b : {std::string("GIF89a\x01\0", 8), std::string("\x89PNG\r\n\x1a\n", 8),
                               std::string("\xFF\xD8\xFF\xE0\0\x10JFIF", 10), std::string(64, '\x55'),
                               std::string("abc")})
    check(!probe_image(b, w, h, err), "garbage of " + std::to_string(b.size()) + " bytes has no size");
  // A DNL-sized JPEG (height 0) has no size from its SOF.
  check(!probe_image(jpeg(64, 0), w, h, err) || (w >= 1 && h >= 1), "JPEG with SOF height 0");
}

void test_cap() {
  check(max_image_pixels() == kDefaultMaxImagePixels, "default cap");
  set_max_image_pixels(int64_t(1) << 40);
  check(max_image_pixels() == 178956970, "the cap never exceeds PIL's limit");
  set_max_image_pixels(kDefaultMaxImagePixels);
}

// ---- real files through every decoder path

struct CF {
  CFTypeRef ref;
  explicit CF(CFTypeRef r) : ref(r) {}
  ~CF() {
    if (ref) CFRelease(ref);
  }
};

bool can_encode(CFStringRef uti) {
  CF types(CGImageDestinationCopyTypeIdentifiers());
  const CFArrayRef a = static_cast<CFArrayRef>(types.ref);
  return CFArrayContainsValue(a, CFRangeMake(0, CFArrayGetCount(a)), uti);
}

// A w x h gradient encoded as `uti`, with an orientation tag when o > 0.
std::string encode(CFStringRef uti, int w, int h, int o = 0) {
  std::vector<uint8_t> px(size_t(w) * h * 4);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      uint8_t* p = &px[(size_t(y) * w + x) * 4];
      p[0] = uint8_t(x * 255 / w);
      p[1] = uint8_t(y * 255 / h);
      p[2] = uint8_t((x ^ y) & 255);
      p[3] = 255;
    }
  CF cs(CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
  CF ctx(CGBitmapContextCreate(px.data(), size_t(w), size_t(h), 8, size_t(w) * 4, (CGColorSpaceRef)cs.ref,
                               kCGImageAlphaNoneSkipLast));
  CF img(CGBitmapContextCreateImage((CGContextRef)ctx.ref));
  CF data(CFDataCreateMutable(nullptr, 0));
  CF dst(CGImageDestinationCreateWithData((CFMutableDataRef)data.ref, uti, 1, nullptr));
  if (!dst.ref) return {};
  CF props(CFDictionaryCreateMutable(nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
  if (o > 0) {
    CF n(CFNumberCreate(nullptr, kCFNumberIntType, &o));
    CFDictionarySetValue((CFMutableDictionaryRef)props.ref, kCGImagePropertyOrientation, n.ref);
  }
  CGImageDestinationAddImage((CGImageDestinationRef)dst.ref, (CGImageRef)img.ref, (CFDictionaryRef)props.ref);
  if (!CGImageDestinationFinalize((CGImageDestinationRef)dst.ref)) return {};
  const CFDataRef d = (CFDataRef)data.ref;
  return std::string(reinterpret_cast<const char*>(CFDataGetBytePtr(d)), size_t(CFDataGetLength(d)));
}

void test_real_files() {
  struct Fmt {
    const char* name;
    CFStringRef uti;
    bool orientation;  // also written with orientation 6 (decode_image rotates what exif_orientation reads)
  };
  const Fmt fmts[] = {
      {"PNG", CFSTR("public.png"), true},          {"JPEG", CFSTR("public.jpeg"), true},
      {"GIF", CFSTR("com.compuserve.gif"), false},  {"TIFF", CFSTR("public.tiff"), true},
      {"BMP", CFSTR("com.microsoft.bmp"), false},   {"HEIC", CFSTR("public.heic"), true},
  };
  const int sizes[][2] = {{123, 45}, {40, 300}, {641, 479}, {600, 4}};  // aspect ratios below 200
  Spec small;  // 131072-pixel budget
  Spec standard;
  standard.set_budget(1003520);
  int files = 0;
  for (const Fmt& f : fmts) {
    if (!can_encode(f.uti)) {
      std::printf("  %s: no encoder on this Mac, skipped\n", f.name);
      continue;
    }
    for (const auto& sz : sizes) {
      for (int o : {0, 6}) {
        if (o && !f.orientation) continue;
        const std::string bytes = encode(f.uti, sz[0], sz[1], o);
        const std::string what = std::string(f.name) + " " + size_str(sz[0], sz[1]) + " orientation " + std::to_string(o);
        if (bytes.empty()) {
          check(false, what + ": cannot encode");
          continue;
        }
        int64_t w = 0, h = 0;
        std::string err;
        RGBImage img;
        const bool probed = probe_image(bytes, w, h, err);
        if (!decode_image(bytes, img, err)) {
          check(false, what + ": decode failed: " + err);
          continue;
        }
        check(probed && w == img.width && h == img.height,
              what + ": probed " + size_str(w, h) + ", decoded " + size_str(img.width, img.height));
        for (const Spec* sp : {&small, &standard}) {
          const Prepared p = preprocess(img, *sp);
          check(grid_for(int(h), int(w), *sp) == p.grid(), what + ": grid_for differs from preprocess");
        }
        ++files;
      }
    }
  }
  std::printf("  %d encoded files: probed size = decoded size, grid_for = preprocess grid\n", files);
  // The cap applies to real files as well, before decoding.
  set_max_image_pixels(10000);
  RGBImage img;
  std::string err;
  check(!decode_image(encode(CFSTR("public.png"), 123, 100), img, err), "a 12300-pixel PNG over a 10000-pixel cap");
  check(!decode_image(encode(CFSTR("public.jpeg"), 123, 100), img, err), "a 12300-pixel JPEG over a 10000-pixel cap");
  check(!decode_image(encode(CFSTR("com.compuserve.gif"), 123, 100), img, err), "a 12300-pixel GIF over a 10000-pixel cap");
  check(decode_image(encode(CFSTR("public.png"), 100, 100), img, err), "a 10000-pixel PNG at the cap: " + err);
  set_max_image_pixels(kDefaultMaxImagePixels);
}

}  // namespace

int main() {
  std::printf("image bombs\n");
  test_bombs();  // first: the peak-memory check needs a low peak to start from
  std::printf("headers\n");
  test_headers();
  test_cap();
  std::printf("real files\n");
  test_real_files();
  std::printf("%d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
