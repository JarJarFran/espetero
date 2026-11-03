// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// GLM-5.3-Flash loader skeleton without a GPU (src/model/impl/glm5.h): the
// config, the per-layer tensor expectations (names, dtypes, shapes) and the
// routed experts' locations, checked against a checkpoint directory. Works on
// the sparse fake checkpoint tests/ref/glm5_ref.py writes from the shard
// headers (no weights, no download of them), or on the real files.
//
//   python3 -I tests/ref/glm5_ref.py fake-ckpt /tmp/glm5-fake
//   build/glm5_load_test /tmp/glm5-fake [--partial]
#include <cstdio>
#include <cstring>
#include <string>

#include "../src/model/impl/glm5.h"

using namespace espetero::native;

namespace {
int checks = 0, failures = 0;
void check(bool ok, const std::string& what) {
  ++checks;
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s CKPT_DIR [--partial]\n", argv[0]);
    return 2;
  }
  const bool partial = argc > 2 && std::strcmp(argv[2], "--partial") == 0;
  glm5::Index idx;
  try {
    idx.open(argv[1], partial);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "open failed: %s\n", e.what());
    return 1;
  }
  const glm5::Cfg& c = idx.c;
  std::printf("%s\n", idx.summary().c_str());
  // The GLM-5.3-Flash geometry (docs/glm5.md 0); another glm5_next checkpoint would differ here only.
  check(c.D == 4096 && c.L == 45 && c.V == 154880 && c.n_mtp == 1, "config: D 4096, 45 layers, vocab 154880, 1 MTP layer");
  check(c.n_kda() == 34 && c.n_dsa() == 11 && c.n_moe() == 42, "config: 34 KDA, 11 DSA, 42 MoE layers");
  bool sched = true;
  for (int l = 0; l < c.L; ++l) sched &= c.is_dsa[size_t(l)] == (l % 4 == 3) && c.is_moe[size_t(l)] == (l >= 3) && c.idx_full[size_t(l)];
  check(sched, "config: DSA at l mod 4 = 3, dense MLP in layers 0-2, every indexer full");
  check(c.kH == 64 && c.kD == 128 && c.conv == 4 && c.gate_lb == -5.0f, "config: KDA 64 x 128, conv 4, gate lower bound -5");
  check(c.H == 64 && c.QR == 1536 && c.KVR == 512 && c.NOPE == 256 && c.ROPE == 0 && c.VD == 256, "config: nope MLA geometry");
  check(c.IH == 32 && c.IHD == 128 && c.ITOPK == 2048 && c.KPOOL == 4 && c.tail, "config: indexer 32 x 128, top 2048, k-pool 4 + tail");
  check(c.E == 288 && c.K == 8 && c.IE == 2048 && c.NSH == 1 && c.I_dense == 12288, "config: 288 experts x 8, 2048 wide, dense 12288");
  check(c.route_scale == 2.5f && c.swiglu == 10.0f && c.norm_topk && c.hc == 4 && c.hc_iters == 20 && c.block == 128,
        "config: routed scale 2.5, swiglu limit 10, mHC 4 x 20, fp8 blocks of 128");
  check(c.eos.size() == 3 && c.eos[0] == 154820 && c.eos[1] == 154827 && c.eos[2] == 154829 && c.pad == 154820, "config: eos ids");
  check(idx.layers == (partial ? idx.layers : 46), "index: 46 layers in the file (45 + MTP)");
  // Expert slots: 3 fp8 matrices of 2048 x 4096 and their 128 x 128 fp32 scales, two contiguous runs per expert.
  check(idx.codes_bytes == 3ull * 2048 * 4096 && idx.scales_bytes == 3ull * 16 * 32 * 4, "experts: 25,165,824 B of codes + 6,144 B of scales");
  size_t n_experts = 0, two_pieces = 0;
  bool dst_ok = true;
  for (size_t l = 0; l < idx.experts.size(); ++l)
    for (const glm5::ExpertLoc& x : idx.experts[l]) {
      ++n_experts;
      two_pieces += x.pieces.size() == 2;
      uint64_t covered = 0;
      for (const glm5::Piece& pc : x.pieces) {
        dst_ok &= pc.dst == covered && pc.file >= 0;
        covered += pc.nbytes;
      }
      dst_ok &= covered == idx.codes_bytes + idx.scales_bytes;
      if (x.pieces.size() == 2) dst_ok &= x.pieces[0].nbytes == idx.codes_bytes && x.pieces[1].nbytes == idx.scales_bytes;
    }
  if (!partial) {
    check(n_experts == 43u * 288u, "experts: 43 layers x 288 indexed (" + std::to_string(n_experts) + ")");
    check(idx.bytes_experts == 42ull * 288 * (idx.codes_bytes + idx.scales_bytes), "experts: the trunk's bytes add up");
  }
  check(dst_ok, "experts: every slot covered once, in order (codes then scales)");
  check(two_pieces + size_t(idx.split_experts) == n_experts, "experts: two reads each but the split ones");
  std::printf("experts: %zu indexed, %zu in two contiguous runs, %d split across shards\n", n_experts, two_pieces, idx.split_experts);
  check(idx.bytes_resident > 14e9 && idx.bytes_resident < 17e9, "resident text weights ~15 GB (" + std::to_string(idx.bytes_resident) + ")");
  check(idx.bytes_vision > 1e9 && idx.bytes_vision < 1.3e9 && idx.bytes_other == 0, "vision ~1.1 GB, nothing unaccounted");
  std::printf("%d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
