// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Image content digest: BLAKE2b (RFC 7693) in a two-level tree so large
// pixel buffers hash on all cores. BLAKE2b is a cryptographic hash, so the
// digest is a safe exact cache key even for adversarial inputs (a crafted
// image cannot collide with another user's image).
//
//   leaf_i = BLAKE2b-256(chunk_i)            chunk_i = 256 KiB of the float32 bytes
//   digest = BLAKE2b-128("espetero.vision.px1" || le32 t,h,w || le64 nbytes || leaf_0 || ...)
//
// Python equivalent (tests/ref/vision_ref.py checks it):
//   leaves = b"".join(blake2b(buf[i:i+C], digest_size=32).digest() for i in range(0, len(buf), C))
//   blake2b(b"espetero.vision.px1" + struct.pack("<3iQ", t, h, w, len(buf)) + leaves, digest_size=16)
#include <dispatch/dispatch.h>

#include <cstring>
#include <vector>

#include "vision.h"

namespace espetero::vision {
namespace {

class Blake2b {
 public:
  explicit Blake2b(size_t outlen) : outlen_(outlen) {
    for (int i = 0; i < 8; i++) h_[i] = kIV[i];
    h_[0] ^= 0x01010000u ^ uint64_t(outlen);  // no key, fanout 1, depth 1
  }
  void update(const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    while (n > 0) {
      if (buflen_ == 128) {  // keep the last block for final()
        count(128);
        compress(buf_, false);
        buflen_ = 0;
      }
      if (buflen_ == 0) {
        // Whole blocks straight from the input, always leaving at least one byte.
        while (n > 128) {
          count(128);
          compress(p, false);
          p += 128;
          n -= 128;
        }
      }
      const size_t take = std::min(n, 128 - buflen_);
      std::memcpy(buf_ + buflen_, p, take);
      buflen_ += take;
      p += take;
      n -= take;
    }
  }
  void final(uint8_t* out) {
    count(buflen_);
    std::memset(buf_ + buflen_, 0, 128 - buflen_);
    compress(buf_, true);
    for (size_t i = 0; i < outlen_; i++) out[i] = uint8_t(h_[i / 8] >> (8 * (i % 8)));
  }

 private:
  static constexpr uint64_t kIV[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull,
                                      0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                                      0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
                                      0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  static constexpr uint8_t kSigma[12][16] = {
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
      {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
      {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
      {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
      {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
      {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
      {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
      {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
      {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
      {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
      {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

  static uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }
  void count(size_t n) {
    t_[0] += n;
    if (t_[0] < n) t_[1]++;
  }
  void compress(const uint8_t* block, bool last) {
    uint64_t m[16], v[16];
    std::memcpy(m, block, 128);  // little-endian host (arm64 / x86)
    for (int i = 0; i < 8; i++) {
      v[i] = h_[i];
      v[i + 8] = kIV[i];
    }
    v[12] ^= t_[0];
    v[13] ^= t_[1];
    if (last) v[14] = ~v[14];
    auto g = [&](int a, int b, int c, int d, uint64_t x, uint64_t y) {
      v[a] = v[a] + v[b] + x;
      v[d] = rotr(v[d] ^ v[a], 32);
      v[c] = v[c] + v[d];
      v[b] = rotr(v[b] ^ v[c], 24);
      v[a] = v[a] + v[b] + y;
      v[d] = rotr(v[d] ^ v[a], 16);
      v[c] = v[c] + v[d];
      v[b] = rotr(v[b] ^ v[c], 63);
    };
    for (int r = 0; r < 12; r++) {
      const uint8_t* s = kSigma[r];
      g(0, 4, 8, 12, m[s[0]], m[s[1]]);
      g(1, 5, 9, 13, m[s[2]], m[s[3]]);
      g(2, 6, 10, 14, m[s[4]], m[s[5]]);
      g(3, 7, 11, 15, m[s[6]], m[s[7]]);
      g(0, 5, 10, 15, m[s[8]], m[s[9]]);
      g(1, 6, 11, 12, m[s[10]], m[s[11]]);
      g(2, 7, 8, 13, m[s[12]], m[s[13]]);
      g(3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) h_[i] ^= v[i] ^ v[i + 8];
  }

  uint64_t h_[8], t_[2] = {0, 0};
  uint8_t buf_[128];
  size_t buflen_ = 0, outlen_;
};

constexpr size_t kChunk = 256 * 1024;

void put_le(std::vector<uint8_t>& out, uint64_t v, int bytes) {
  for (int i = 0; i < bytes; i++) out.push_back(uint8_t(v >> (8 * i)));
}

}  // namespace

std::array<uint8_t, 16> digest(const float* pixels, size_t count, std::array<int, 3> grid) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(pixels);
  const size_t nbytes = count * sizeof(float);
  const size_t nchunks = (nbytes + kChunk - 1) / kChunk;
  std::vector<uint8_t> leaves(nchunks * 32);
  struct Ctx {
    const uint8_t* bytes;
    size_t nbytes;
    uint8_t* leaves;
  } ctx{bytes, nbytes, leaves.data()};
  auto leaf = [](void* p, size_t i) {
    auto* c = static_cast<Ctx*>(p);
    const size_t off = i * kChunk, len = std::min(kChunk, c->nbytes - off);
    Blake2b h(32);
    h.update(c->bytes + off, len);
    h.final(c->leaves + 32 * i);
  };
  if (nchunks > 1)
    dispatch_apply_f(nchunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &ctx, leaf);
  else if (nchunks == 1)
    leaf(&ctx, 0);

  std::vector<uint8_t> head;
  const char* tag = "espetero.vision.px1";
  head.insert(head.end(), tag, tag + 16);
  for (int g : grid) put_le(head, uint32_t(g), 4);
  put_le(head, nbytes, 8);
  Blake2b root(16);
  root.update(head.data(), head.size());
  root.update(leaves.data(), leaves.size());
  std::array<uint8_t, 16> out;
  root.final(out.data());
  return out;
}

}  // namespace espetero::vision
