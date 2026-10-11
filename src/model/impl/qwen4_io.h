// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Qwen4-Exp host I/O (docs/qwen4_exp.md 5): the n-gram rows of a pass and,
// until the expert cache, whole expert stacks of one layer, read with
// pread from the checkpoint files.
#pragma once

#include <unistd.h>

#include <sys/mount.h>
#include <sys/stat.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "../../util/paths.h"
#include "model.h"

namespace espetero::native {

inline void pread_all(int fd, void* dst, size_t n, uint64_t off) {
  uint8_t* p = static_cast<uint8_t*>(dst);
  while (n) {
    const ssize_t r = ::pread(fd, p, n, off_t(off));
    if (r <= 0) fail("pread failed at offset " + std::to_string(off));
    p += r;
    n -= size_t(r);
    off += uint64_t(r);
  }
}

// The table rows of a token from its id c0, the previous id c1 and c2 (the id
// before c1, or EOS when c1 is EOS): bigram heads hash (c0 M0) ^ (c1 M1),
// trigram heads also ^ (c2 M2); row = offset_h + hash mod vocab_h.
inline void ngram_rows(const NgramTable& T, int per_ngram, int64_t c0, int64_t c1, int64_t c2, int64_t* rows) {
  const int64_t x2 = (c0 * T.mult[0]) ^ (c1 * T.mult[1]);
  const int64_t x3 = x2 ^ (c2 * T.mult[2]);
  for (int h = 0; h < 2 * per_ngram; ++h) rows[h] = T.offs[size_t(h)] + (h < per_ngram ? x2 : x3) % T.vocab[size_t(h)];
}

// One table row dequantized (4-bit affine, bf16 scales and biases) times the
// table's weight_scale, as fp16.
inline void ngram_read_row(const NgramTable& T, int64_t row, __fp16* out) {
  const int64_t sh = row / T.shard_rows, r = row % T.shard_rows;
  const NgramTable::Shard& S = T.shards[size_t(sh)];
  const int fd = T.fds[size_t(S.file)];
  const int ng = T.width / T.group;
  uint32_t codes[64];
  uint16_t sc[16], bi[16];
  pread_all(fd, codes, size_t(T.width / 2), S.w + uint64_t(r) * uint64_t(T.width / 2));
  pread_all(fd, sc, size_t(ng) * 2, S.s + uint64_t(r) * uint64_t(ng) * 2);
  pread_all(fd, bi, size_t(ng) * 2, S.b + uint64_t(r) * uint64_t(ng) * 2);
  auto bf = [](uint16_t x) {
    uint32_t u = uint32_t(x) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  for (int j = 0; j < T.width; ++j) {
    const uint32_t q = (codes[j / 8] >> (4 * (j % 8))) & 0xFu;
    const int g = j / T.group;
    out[j] = __fp16((float(q) * bf(sc[g]) + bf(bi[g])) * T.scale);
  }
}

// e [rows][heads * width] (fp16) for the pass: each row's heads from `rows`
// ([n][heads]); distinct rows are read once, in parallel.
inline void ngram_gather(const NgramTable& T, int heads, const std::vector<int64_t>& rows, size_t n, __fp16* e) {
  std::unordered_map<int64_t, int> uniq;
  std::vector<int64_t> order;
  std::vector<int> slot(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    auto [it, fresh] = uniq.try_emplace(rows[i], int(order.size()));
    if (fresh) order.push_back(rows[i]);
    slot[i] = it->second;
  }
  std::vector<__fp16> vals(order.size() * size_t(T.width));
  parallel_for(order.size(), [&](size_t k) { ngram_read_row(T, order[k], vals.data() + k * size_t(T.width)); });
  for (size_t r = 0; r < n; ++r)
    for (int h = 0; h < heads; ++h)
      std::memcpy(e + (r * size_t(heads) + size_t(h)) * size_t(T.width),
                  vals.data() + size_t(slot[r * size_t(heads) + size_t(h)]) * size_t(T.width), size_t(T.width) * 2);
}

// Test path until the expert cache: every expert of layer `li` as the
// MLX host layout the loader would build ([E * Ie, D] gate and up interleaved
// by 8 rows, [E * D, Ie] down), read with pread into `tmp`.
inline void read_layer_experts(Model& m, int li, Arena& tmp, QW& gu, QW& down) {
  const Config& c = m.cfg;
  const ExpertLayer& X = m.xs.layers[size_t(li)];
  auto stack = [&](const ExpertPiece& w, const ExpertPiece& s, const ExpertPiece& z, int out, int in) {
    QW q;
    q.out = c.E * out;
    q.in = in;
    q.src_bits = X.bits;
    q.bits = gpu_bits_for(X.bits);
    q.group = X.group;
    q.half_scales = false;  // bf16 in MLX files
    if (q.group != 64 && q.group != 32) fail("expert group size " + std::to_string(q.group) + " is not supported yet");
    const size_t wb = q.row_bytes() * size_t(q.out), sb = size_t(q.out) * size_t(in / q.group) * 2;
    if (w.stride * uint64_t(c.E) != wb || s.stride * uint64_t(c.E) != sb || z.stride * uint64_t(c.E) != sb)
      fail("layer " + std::to_string(li) + ": expert stack sizes do not match the configuration");
    q.w = tmp.alloc(wb);
    q.s = tmp.alloc(sb);
    q.z = tmp.alloc(sb);
    struct Job { int fd; uint8_t* dst; size_t n; uint64_t off; };
    std::vector<Job> jobs;
    auto split = [&](const ExpertPiece& p, const Buf& b, size_t n) {
      const size_t step = size_t(64) << 20;  // 64 MB reads
      for (size_t o = 0; o < n; o += step)
        jobs.push_back({m.xs.fds[size_t(p.file)], static_cast<uint8_t*>(ptr(b)) + o, std::min(step, n - o), p.off + o});
    };
    split(w, q.w, wb);
    split(s, q.s, sb);
    split(z, q.z, sb);
    parallel_for(jobs.size(), [&](size_t i) { pread_all(jobs[i].fd, jobs[i].dst, jobs[i].n, jobs[i].off); });
    return q;
  };
  const QW g = stack(X.gw, X.gs, X.gb, c.Ie, c.D);
  const QW u = stack(X.uw, X.us, X.ub, c.Ie, c.D);
  gu = interleave8(tmp, g, u);
  down = stack(X.dw, X.ds, X.db, c.D, c.Ie);
}

}  // namespace espetero::native

namespace espetero::native {

// ---------------------------------------------------------------- expert cache
// Codes and scales of one expert (host layout) into slot `slot` of a pool:
// codes are block-major (the slot's rows are one contiguous range of the
// pool's codes), scales group-major at the pool's pair stride.
inline void repack_slot(Runtime& rt, id<MTLComputeCommandEncoder> e, const QW& q, const QW& pool, int slot) {
  if (q.bits != 4 || q.src_bits != 4 || q.out % 32 || q.in % 64) fail("expert cache: 4-bit experts with 32-row blocks only");
  const int KG = q.in / 64;
  const size_t slot_w = size_t(q.out) * size_t(q.in) / 2;
  [e setComputePipelineState:rt.kernel("repack_weights")];
  [e setBuffer:q.w.b offset:q.w.off atIndex:0];
  [e setBuffer:pool.wr.b offset:pool.wr.off + size_t(slot) * slot_w atIndex:1];
  [e setBytes:&q.out length:4 atIndex:2];
  [e setBytes:&q.in length:4 atIndex:3];
  [e dispatchThreads:MTLSizeMake(size_t(q.out / 32) * size_t(KG) * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  const int half_src = q.half_scales ? 1 : 0;
  const int pair0 = slot * (q.out / 2), pool_pairs = pool.out / 2;
  [e setComputePipelineState:rt.kernel("repack_scales_at")];
  [e setBuffer:q.s.b offset:q.s.off atIndex:0];
  [e setBuffer:q.z.b offset:q.z.off atIndex:1];
  [e setBuffer:pool.sb.b offset:pool.sb.off atIndex:2];
  [e setBytes:&q.out length:4 atIndex:3];
  [e setBytes:&q.in length:4 atIndex:4];
  [e setBytes:&q.group length:4 atIndex:5];
  [e setBytes:&half_src length:4 atIndex:6];
  [e setBytes:&pair0 length:4 atIndex:7];
  [e setBytes:&pool_pairs length:4 atIndex:8];
  [e dispatchThreads:MTLSizeMake(size_t(q.out / 2), size_t(q.in / q.group), 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

// ---------------------------------------------------------------- expert records
// One record = a one-slot pool in the GPU layout (the q4_pool_admit source):
// gate/up codes (block-major), their scale pairs [groups][Ie] (half4), down
// codes, down scale pairs [groups][D / 2]. Returns the record's bytes.
inline size_t record_layout(const Config& c, int group, size_t o[4]) {
  const size_t gw = size_t(2 * c.Ie) * size_t(c.D) / 2, gs = size_t(c.Ie) * size_t(c.D / group) * 8;
  const size_t dw = size_t(c.D) * size_t(c.Ie) / 2, ds = size_t(c.D / 2) * size_t(c.Ie / group) * 8;
  o[0] = 0;
  o[1] = gw;
  o[2] = gw + gs;
  o[3] = gw + gs + dw;
  return gw + gs + dw + ds;
}

// The store's identity: its layout and the checkpoint files (name, size, mtime).
inline std::string record_meta(Model& m) {
  const Config& c = m.cfg;
  const ExpertLayer& X0 = m.xs.layers[0];
  size_t o[4];
  const size_t bytes = record_layout(c, X0.group, o);
  std::string s = "espetero expert records v1\n" + std::to_string(c.E) + " " + std::to_string(c.L) + " " +
                  std::to_string(c.Ie) + " " + std::to_string(c.D) + " " + std::to_string(X0.bits) + " " +
                  std::to_string(X0.group) + " " + std::to_string(bytes) + "\n";
  for (const std::string& f : m.xs.files) {
    struct stat st {};
    if (::stat(f.c_str(), &st) != 0) fail("cannot stat " + f);
    s += f.substr(f.find_last_of('/') + 1) + " " + std::to_string(st.st_size) + " " + std::to_string(st.st_mtimespec.tv_sec) + "\n";
  }
  return s;
}

inline std::string record_file(const std::string& dir, int li) {
  char b[16];
  std::snprintf(b, sizeof(b), "/L%02d.bin", li);
  return dir + b;
}

// A 16 KiB-aligned staging range (direct reads want page-aligned buffers).
inline Buf alloc_aligned(Arena& a, size_t n, size_t align = 16384) {
  Buf b = a.alloc(n + align);
  const size_t addr = reinterpret_cast<size_t>(ptr(b));
  b.off += (align - addr % align) % align;
  return b;
}

// Copy one record (at `b` + `off`) into slot `slot` of pool P.
inline void admit_record(Runtime& rt, id<MTLComputeCommandEncoder> e, id<MTLBuffer> b, size_t off, const size_t o[4],
                         ExpertPool& P, int slot, const Config& c, int group) {
  auto one = [&](size_t ow, size_t os, const QW& dst, int rows, int in) {
    const int row_bytes = in / 2, groups = in / group, src_pairs = rows / 2, dst_pairs = dst.out / 2, src_e = 0;
    [e setComputePipelineState:rt.kernel("q4_pool_admit")];
    [e setBuffer:b offset:off + ow atIndex:0];
    [e setBuffer:b offset:off + os atIndex:1];
    [e setBuffer:dst.wr.b offset:dst.wr.off atIndex:2];
    [e setBuffer:dst.sb.b offset:dst.sb.off atIndex:3];
    [e setBytes:&rows length:4 atIndex:4];
    [e setBytes:&row_bytes length:4 atIndex:5];
    [e setBytes:&groups length:4 atIndex:6];
    [e setBytes:&src_pairs length:4 atIndex:7];
    [e setBytes:&dst_pairs length:4 atIndex:8];
    [e setBytes:&src_e length:4 atIndex:9];
    [e setBytes:&slot length:4 atIndex:10];
    const size_t words = size_t(rows) * size_t(row_bytes) / 4, sp = size_t(rows / 2) * size_t(groups);
    [e dispatchThreads:MTLSizeMake(std::max(words, sp), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  };
  one(o[0], o[1], P.gu, 2 * c.Ie, c.D);
  one(o[2], o[3], P.dn, c.D, c.Ie);
}

// Read into GPU-visible memory. ESPETERO_XS_BOUNCE (probe): 1 reads into a
// page-aligned host buffer first and copies, 2 shifts the destination off the
// page grid (the kernel then cannot DMA straight into the Metal pages).
inline void pread_gpu(int fd, void* dst, size_t n, uint64_t off) {
  static const int mode = [] {
    const char* v = std::getenv("ESPETERO_XS_BOUNCE");
    return v && *v ? std::atoi(v) : 0;
  }();
  if (mode == 1) {
    thread_local uint8_t* hb = nullptr;
    thread_local size_t hn = 0;
    if (hn < n) {
      std::free(hb);
      hb = nullptr;
      hn = align_up(n, 16384);
      if (::posix_memalign(reinterpret_cast<void**>(&hb), 16384, hn) != 0) fail("bounce buffer");
    }
    pread_all(fd, hb, n, off);
    std::memcpy(dst, hb, n);
  } else {
    pread_all(fd, dst, n, off);
  }
}

// Batch reads on the reader pool (front of the queue); the next commit waits for them.
inline void xio_submit(ExpertPrefetch& X, std::vector<ExpertPrefetch::Job>& js, std::atomic<int>* counter = nullptr,
                       bool front = true) {
  if (js.empty()) return;
  std::atomic<int>* cnt = counter ? counter : &X.outstanding;
  cnt->fetch_add(int(js.size()), std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lk(X.mu);
    if (front)
      for (auto it = js.rbegin(); it != js.rend(); ++it) {
        it->cnt = cnt;
        it->entry = -1;
        X.jobs.push_front(*it);
      }
    else
      for (auto& j : js) {
        j.cnt = cnt;
        j.entry = -1;
        X.jobs.push_back(j);
      }
  }
  X.cv_job.notify_all();
}

// Spin (never sleep: the GPU is idle here and a sleeping host lets the
// performance controller lower its clock) until every read the encoded work
// depends on has completed.
inline void xio_wait(ExpertPrefetch& X) {
  while (X.outstanding.load(std::memory_order_acquire) > 0) {
  }
  if (X.st_need >= 0) {  // the staging buffer the committed MoE reads
    while (X.st_pending[X.st_need].load(std::memory_order_acquire) > 0) {
    }
    X.st_need = -1;
  }
  for (int idx : X.await)
    while (!X.entries[size_t(idx)]->done.load(std::memory_order_acquire)) {
    }
  bool failed = X.io_failed.exchange(false);
  if (!X.await.empty()) {
    std::lock_guard<std::mutex> lk(X.mu);
    for (int idx : X.await) failed = failed || X.entries[size_t(idx)]->failed;
  }
  X.await.clear();
  if (failed) fail("expert read failed");
}

// Reads of one record at file offset `off` into `dst`, in `chunks` pieces
// (demand loads split so the SSD serves one expert with several reads in flight).
inline int record_chunks() {
  static const int n = [] {
    const char* v = std::getenv("ESPETERO_XS_CHUNKS");
    return v && *v ? std::max(1, std::atoi(v)) : 4;
  }();
  return n;
}

// Experts of layer `li` from the checkpoint (9 pieces each) into pool slots:
// all pieces are read in parallel (the SSD needs reads in flight), gate and up
// interleaved on the host, then each expert repacked into its slot. Staging
// lives in `tmp` until the encoder's work ran. Returns the bytes read.
// With `defer` (records with the reader pool), the record -> slot copies
// are appended there to be encoded later instead of into `e`.
inline size_t load_expert_slots(Model& m, id<MTLComputeCommandEncoder> e, int li,
                                const std::vector<std::pair<int, int>>& ex_slot, ExpertPool& P, Arena& tmp,
                                std::vector<std::function<void(id<MTLComputeCommandEncoder>)>>* defer = nullptr) {
  const Config& c = m.cfg;
  const ExpertLayer& X = m.xs.layers[size_t(li)];
  struct Job {
    int fd;
    void* dst;
    size_t n;
    uint64_t off;
  };
  std::vector<Job> jobs;
  size_t bytes = 0;
  if (!m.xs.rec_fds.empty()) {  // records: one read per expert, split in chunks, then two copies
    const size_t R = m.xs.rec_bytes, stride = m.xs.rec_stride;
    const size_t per = align_up((R + size_t(record_chunks()) - 1) / size_t(record_chunks()), 16384);
    std::vector<Buf> st;
    static const bool shift = [] {
      const char* v = std::getenv("ESPETERO_XS_BOUNCE");
      return v && *v == '2';
    }();
    for (auto [ex, slot] : ex_slot) {
      st.push_back(alloc_aligned(tmp, R + 256));
      if (shift) st.back().off += 256;
      for (size_t o = 0; o < R; o += per)
        jobs.push_back({m.xs.rec_fds[size_t(li)], static_cast<uint8_t*>(ptr(st.back())) + o, std::min(per, R - o),
                        uint64_t(ex) * stride + o});
      bytes += R;
    }
    if (m.xpf) {  // asynchronous: the copies are encoded now, the commit waits for the reads
      std::vector<ExpertPrefetch::Job> js;
      for (const Job& j : jobs) js.push_back({j.fd, static_cast<uint8_t*>(j.dst), j.n, j.off, -1, nullptr});
      xio_submit(*m.xpf, js);
    } else {
      parallel_for(jobs.size(), [&](size_t i) { pread_gpu(jobs[i].fd, jobs[i].dst, jobs[i].n, jobs[i].off); });
    }
    for (size_t i = 0; i < ex_slot.size(); ++i) {
      if (defer && m.xpf) {
        const Buf b = st[i];
        const int slot = ex_slot[i].second, group = X.group;
        ExpertPool* PP = &P;
        Model* mm = &m;
        defer->push_back([mm, b, slot, group, PP](id<MTLComputeCommandEncoder> enc) {
          admit_record(*mm, enc, b.b, b.off, mm->xs.rec_o, *PP, slot, mm->cfg, group);
        });
      } else {
        admit_record(m, e, st[i].b, st[i].off, m.xs.rec_o, P, ex_slot[i].second, c, X.group);
      }
    }
    return bytes;
  }
  auto host = [&](int ex, const ExpertPiece& w, const ExpertPiece& s, const ExpertPiece& z, int out, int in) {
    QW q;
    q.out = out;
    q.in = in;
    q.src_bits = X.bits;
    q.bits = gpu_bits_for(X.bits);
    q.group = X.group;
    q.half_scales = false;
    const size_t wb = q.row_bytes() * size_t(out), sb = size_t(out) * size_t(in / q.group) * 2;
    q.w = tmp.alloc(wb);
    q.s = tmp.alloc(sb);
    q.z = tmp.alloc(sb);
    jobs.push_back({m.xs.fds[size_t(w.file)], ptr(q.w), wb, w.off + uint64_t(ex) * w.stride});
    jobs.push_back({m.xs.fds[size_t(s.file)], ptr(q.s), sb, s.off + uint64_t(ex) * s.stride});
    jobs.push_back({m.xs.fds[size_t(z.file)], ptr(q.z), sb, z.off + uint64_t(ex) * z.stride});
    bytes += wb + 2 * sb;
    return q;
  };
  struct Staged {
    QW g, u, d;
    int slot;
  };
  std::vector<Staged> st;
  for (auto [ex, slot] : ex_slot)
    st.push_back({host(ex, X.gw, X.gs, X.gb, c.Ie, c.D), host(ex, X.uw, X.us, X.ub, c.Ie, c.D),
                  host(ex, X.dw, X.ds, X.db, c.D, c.Ie), slot});
  if (m.xpf) {  // the reader pool, waiting by spinning (the host interleaves next)
    std::vector<ExpertPrefetch::Job> js;
    for (const Job& j : jobs) js.push_back({j.fd, static_cast<uint8_t*>(j.dst), j.n, j.off, -1, nullptr});
    xio_submit(*m.xpf, js);
    while (m.xpf->outstanding.load(std::memory_order_acquire) > 0) {
    }
    if (m.xpf->io_failed.exchange(false)) fail("expert read failed");
  } else {
    parallel_for(jobs.size(), [&](size_t i) { pread_all(jobs[i].fd, jobs[i].dst, jobs[i].n, jobs[i].off); });
  }
  for (Staged& x : st) {
    const QW gu = interleave8(tmp, x.g, x.u);
    repack_slot(m, e, gu, P.gu, x.slot);
    repack_slot(m, e, x.d, P.dn, x.slot);
  }
  return bytes;
}

// ---------------------------------------------------------------- expert prefetch
// Early guess of the next layer's experts (ESPETERO_Q4_PF): 0 off, 1 its router
// on this layer's MoE input, 2 (default) its MLP hyper-connection read and
// router on the current streams (the more accurate guess).
inline int q4_pf_mode() {
  static const int v = [] {
    const char* e = std::getenv("ESPETERO_Q4_PF");
    return e && *e ? std::atoi(e) : 2;
  }();
  return v;
}
// Gate and up rows interleaved by 8 (interleave8 of two same-format 4-bit
// MLX matrices is exactly this copy) into the entry's interleaved region.
inline void xpf_interleave(ExpertPrefetch& X, ExpertPrefetch::Entry& e) {
  uint8_t* base = static_cast<uint8_t*>([X.buf contents]) + e.off;
  const size_t wrow = size_t(X.D) / 2, srow = size_t(X.D / X.group) * 2;
  auto rows8 = [&](size_t dst, size_t a, size_t b, size_t row) {
    for (int blk = 0; blk < X.Ie / 8; ++blk) {
      std::memcpy(base + dst + size_t(blk) * 16 * row, base + a + size_t(blk) * 8 * row, row * 8);
      std::memcpy(base + dst + (size_t(blk) * 16 + 8) * row, base + b + size_t(blk) * 8 * row, row * 8);
    }
  };
  rows8(X.o[9], X.o[0], X.o[3], wrow);
  rows8(X.o[10], X.o[1], X.o[4], srow);
  rows8(X.o[11], X.o[2], X.o[5], srow);
}

inline void xpf_worker(ExpertPrefetch* X) {
  for (;;) {
    ExpertPrefetch::Job j;
    {
      std::unique_lock<std::mutex> lk(X->mu);
      X->cv_job.wait(lk, [&] { return X->stop || !X->jobs.empty(); });
      if (X->stop) return;
      j = X->jobs.front();
      X->jobs.pop_front();
    }
    bool ok = true;
    // Timing probes: ESPETERO_Q4_PF_NOREAD=1 skips the reads (wrong logits),
    // ESPETERO_Q4_PF_BOUNCE=1 reads into host memory first, then copies.
    static const bool noread = env_on("ESPETERO_Q4_PF_NOREAD");
    static const int bounce = [] {
      const char* v = std::getenv("ESPETERO_Q4_PF_BOUNCE");
      return v ? std::atoi(v) : 0;
    }();
    thread_local std::vector<uint8_t> tmp;
    try {
      if (noread) {
      } else if (bounce) {
        if (tmp.size() < j.n) tmp.resize(j.n);
        pread_all(j.fd, tmp.data(), j.n, j.off);
        if (bounce == 1) std::memcpy(j.dst, tmp.data(), j.n);
      } else {
        pread_gpu(j.fd, j.dst, j.n, j.off);
      }
    } catch (...) {
      ok = false;
    }
    if (j.entry < 0) {  // batch read
      if (!ok) X->io_failed.store(true);
      j.cnt->fetch_sub(1, std::memory_order_release);
      continue;
    }
    ExpertPrefetch::Entry& e = *X->entries[size_t(j.entry)];
    if (!ok) {
      std::lock_guard<std::mutex> lk(X->mu);
      e.failed = true;
    }
    if (e.pending.fetch_sub(1) == 1) {
      bool failed;
      {
        std::lock_guard<std::mutex> lk(X->mu);
        failed = e.failed;
      }
      static const bool noil = env_on("ESPETERO_Q4_PF_NOIL");  // timing probe (wrong logits)
      if (!failed && !noil && !X->records) xpf_interleave(*X, e);
      {
        std::lock_guard<std::mutex> lk(X->mu);
        e.ready = true;
        e.loading = false;
      }
      e.done.store(true, std::memory_order_release);
      X->cv_ready.notify_all();
    }
  }
}

// The prefetch entries (ESPETERO_Q4_PF_ENTRIES, default 64: 4.6 MB each for
// Flash-Next) and the reader pool (ESPETERO_Q4_PF_THREADS, default 8).
inline ExpertPrefetch& ensure_prefetch(Model& m) {
  if (m.xpf) return *m.xpf;
  const Config& c = m.cfg;
  const ExpertLayer& L0 = m.xs.layers[0];
  if (L0.bits != 4) fail("expert prefetch: only 4-bit experts so far");
  auto env_n = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return v && *v ? std::max(1, std::atoi(v)) : d;
  };
  auto X = std::make_unique<ExpertPrefetch>();
  X->st_pending[0].store(0);
  X->st_pending[1].store(0);
  X->Ie = c.Ie;
  X->D = c.D;
  X->group = L0.group;
  X->records = !m.xs.rec_fds.empty();
  const size_t w = size_t(c.Ie) * size_t(c.D) / 2;                 // codes of one matrix
  const size_t sz = size_t(c.Ie) * size_t(c.D / L0.group) * 2;    // bf16 scales (or biases) of one matrix
  const size_t sizes[12] = {w, sz, sz, w, sz, sz, w, sz, sz, 2 * w, 2 * sz, 2 * sz};
  size_t off = 0;
  for (int i = 0; i < 12; ++i) {
    X->o[i] = off;
    off += align_up(sizes[i], 256);
  }
  X->entry_bytes = X->records ? m.xs.rec_stride : align_up(off, 16384);
  const int n = q4_pf_mode() > 0 ? env_n("ESPETERO_Q4_PF_ENTRIES", 64) : 0;  // entries only for prefetch
  X->buf = m.buffer(std::max<size_t>(X->entry_bytes * size_t(n), 16384));
  for (int i = 0; i < n; ++i) {
    X->entries.push_back(std::make_unique<ExpertPrefetch::Entry>());
    X->entries.back()->off = X->entry_bytes * size_t(i);
  }
  const int nt = env_n("ESPETERO_Q4_PF_THREADS", 8);
  for (int i = 0; i < nt; ++i) X->threads.emplace_back(xpf_worker, X.get());
  std::fprintf(stderr, "expert prefetch: %d entries (%.0f MB), %d readers\n", n,
               double(X->entry_bytes) * n / 1e6, nt);
  m.xpf = std::move(X);
  return *m.xpf;
}

// Entry tagged (li, ex), or -1. Caller holds X.mu.
inline int xpf_find(ExpertPrefetch& X, int E, int li, int ex) {
  auto it = X.where.find(int64_t(li) * E + ex);
  return it == X.where.end() ? -1 : it->second;
}

// A reusable entry (not reading, not read by this handshake's GPU work),
// free ones first, then the least recently used tagged one; re-tagged
// (li, ex) with its reads queued (front: demand reads before guesses).
// Returns -1 when every entry is busy. Caller holds X.mu.
inline int xpf_start(Model& m, ExpertPrefetch& X, int li, int ex, bool front, bool guessed) {
  const Config& c = m.cfg;
  int best = -1;
  for (int i = 0; i < int(X.entries.size()); ++i) {
    const ExpertPrefetch::Entry& e = *X.entries[size_t(i)];
    if (e.loading || e.busy >= X.hs) continue;
    if (best < 0) {
      best = i;
      continue;
    }
    const ExpertPrefetch::Entry& b = *X.entries[size_t(best)];
    const bool ef = e.layer < 0, bf = b.layer < 0;
    if ((ef && !bf) || (ef == bf && e.stamp < b.stamp)) best = i;
  }
  if (best < 0) return -1;
  ExpertPrefetch::Entry& e = *X.entries[size_t(best)];
  if (e.layer >= 0) X.where.erase(int64_t(e.layer) * c.E + e.expert);
  e.layer = li;
  e.expert = ex;
  e.loading = true;
  e.ready = false;
  e.failed = false;
  e.done.store(false, std::memory_order_relaxed);
  e.guessed = guessed;
  e.stamp = X.hs;
  X.where[int64_t(li) * c.E + ex] = best;
  uint8_t* base = static_cast<uint8_t*>([X.buf contents]) + e.off;
  if (X.records) {  // one record: demand reads in chunks, guesses in one read (fewer syscalls while the GPU runs)
    const size_t R = m.xs.rec_bytes;
    const size_t per = front ? align_up((R + size_t(record_chunks()) - 1) / size_t(record_chunks()), 16384) : R;
    std::vector<ExpertPrefetch::Job> js;
    for (size_t o = 0; o < R; o += per)
      js.push_back({m.xs.rec_fds[size_t(li)], base + o, std::min(per, R - o), uint64_t(ex) * m.xs.rec_stride + o, best});
    e.pending.store(int(js.size()));
    if (front)
      for (auto it = js.rbegin(); it != js.rend(); ++it) X.jobs.push_front(*it);
    else
      for (const auto& j : js) X.jobs.push_back(j);
    return best;
  }
  e.pending.store(9);
  const ExpertLayer& L = m.xs.layers[size_t(li)];
  const ExpertPiece* pieces[9] = {&L.gw, &L.gs, &L.gb, &L.uw, &L.us, &L.ub, &L.dw, &L.ds, &L.db};
  const size_t w = size_t(c.Ie) * size_t(c.D) / 2, sz = size_t(c.Ie) * size_t(c.D / X.group) * 2;
  ExpertPrefetch::Job jobs[9];
  for (int p = 0; p < 9; ++p) {
    const ExpertPiece& P = *pieces[p];
    jobs[p] = {m.xs.fds[size_t(P.file)], base + X.o[p], p % 3 == 0 ? w : sz, P.off + uint64_t(ex) * P.stride, best};
  }
  if (front)
    for (int p = 8; p >= 0; --p) X.jobs.push_front(jobs[p]);
  else
    for (int p = 0; p < 9; ++p) X.jobs.push_back(jobs[p]);
  return best;
}

// Repack a ready entry into slot `slot` of pool P (encoded into `enc`).
inline void xpf_repack(Model& m, id<MTLComputeCommandEncoder> enc, ExpertPrefetch& X, int idx, ExpertPool& P, int slot) {
  const Config& c = m.cfg;
  const ExpertPrefetch::Entry& e = *X.entries[size_t(idx)];
  if (X.records) {
    admit_record(m, enc, X.buf, e.off, m.xs.rec_o, P, slot, c, X.group);
    return;
  }
  auto qw = [&](int out, int in, size_t ow, size_t os, size_t oz) {
    QW q;
    q.out = out;
    q.in = in;
    q.src_bits = 4;
    q.bits = gpu_bits_for(4);
    q.group = X.group;
    q.half_scales = false;
    q.w = Buf(X.buf, e.off + ow);
    q.s = Buf(X.buf, e.off + os);
    q.z = Buf(X.buf, e.off + oz);
    return q;
  };
  repack_slot(m, enc, qw(2 * c.Ie, c.D, X.o[9], X.o[10], X.o[11]), P.gu, slot);
  repack_slot(m, enc, qw(c.D, c.Ie, X.o[6], X.o[7], X.o[8]), P.dn, slot);
}

// Default place of a model's record store: ~/Library/Caches (outside Time
// Machine backups), a .noindex directory (outside Spotlight).
inline std::string default_record_dir(const std::string& model_path) {
  const char* home = std::getenv("HOME");
  if (!home || !*home) return "";
  std::string base = model_path;
  while (base.size() > 1 && base.back() == '/') base.pop_back();
  base = base.substr(base.find_last_of('/') + 1);
  return paths::user_library_cache_dir() + "/xstore.noindex/" + base;
}

inline double record_store_gb(Model& m) {
  size_t o[4];
  return double(align_up(record_layout(m.cfg, m.xs.layers[0].group, o), 16384)) * m.cfg.E * m.cfg.L / 1e9;
}

inline void mkdir_p(const std::string& dir) { paths::make_private_dirs(dir); }  // 0700 when created

// Open the record store in `dir` if it matches this checkpoint.
inline bool open_expert_records(Model& m, const std::string& dir) {
  const Config& c = m.cfg;
  ExpertStore& X = m.xs;
  std::ifstream in(dir + "/meta.txt");
  if (!in) return false;
  std::stringstream ss;
  ss << in.rdbuf();
  if (ss.str() != record_meta(m)) {
    std::fprintf(stderr, "expert records in %s were built from other files; rebuilding\n", dir.c_str());
    return false;
  }
  X.rec_bytes = record_layout(c, X.layers[0].group, X.rec_o);
  X.rec_stride = align_up(X.rec_bytes, 16384);
  std::vector<int> fds;
  for (int li = 0; li < c.L; ++li) {
    const int fd = ::open(record_file(dir, li).c_str(), O_RDONLY);
    struct stat st {};
    if (fd < 0 || ::fstat(fd, &st) != 0 || uint64_t(st.st_size) < uint64_t(c.E) * X.rec_stride) {
      if (fd >= 0) ::close(fd);
      for (int f : fds) ::close(f);
      return false;
    }
    ::fcntl(fd, F_NOCACHE, 1);
    fds.push_back(fd);
  }
  X.rec_fds = fds;
  std::fprintf(stderr, "expert records: %s (%.3f MB per expert)\n", dir.c_str(), double(X.rec_bytes) / 1e6);
  return true;
}

// Build the record store in `dir` (one-time: reads every expert of the
// checkpoint once and writes E x L records, ~68 GB for Qwen3.8-Flash-Next).
// Each layer is read whole, repacked on the GPU by the loader's kernels and
// cut into records; the first layer is checked byte for byte against the
// per-expert load path. meta.txt is written last (a partial store is rebuilt).
inline void build_expert_records(Model& m, const std::string& dir) {
  const Config& c = m.cfg;
  mkdir_p(dir);
  struct statfs fs {};
  if (::statfs(dir.c_str(), &fs) == 0 && double(fs.f_bavail) * double(fs.f_bsize) < record_store_gb(m) * 1e9 + 10e9)
    fail("expert records: " + std::to_string(int(record_store_gb(m))) + " GB needed plus 10 GB of margin in " + dir);
  std::fprintf(stderr, "expert records: writing %.0f GB to %s (one-time)\n", record_store_gb(m), dir.c_str());
  ::unlink((dir + "/meta.txt").c_str());
  const int group = m.xs.layers[0].group;
  size_t o[4];
  const size_t R = record_layout(c, group, o), stride = align_up(R, 16384);
  const int batch = 32;
  uint8_t* buf = nullptr;
  if (::posix_memalign(reinterpret_cast<void**>(&buf), 16384, stride * batch) != 0) fail("expert records: out of memory");
  std::unique_ptr<uint8_t, decltype(&std::free)> hold(buf, &std::free);
  const auto t0 = Clock::now();
  const size_t gwe = o[1] - o[0], dwe = o[3] - o[2];
  const size_t gp = size_t(c.Ie), gg = size_t(c.D / group), dp = size_t(c.D / 2), dg = size_t(c.Ie / group);
  for (int li = 0; li < c.L; ++li) {
    Arena tmp(m.dev), gpu(m.dev);
    QW gu, dn;
    read_layer_experts(m, li, tmp, gu, dn);
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    repack(m, e, gu, tmp, &gpu);
    repack(m, e, dn, tmp, &gpu);
    [e endEncoding];
    commit_wait(cb);
    if (gu.bits != 4 || dn.bits != 4 || gu.out != c.E * 2 * c.Ie || dn.out != c.E * c.D)
      fail("expert records: unexpected GPU layout");
    const uint8_t *gw = static_cast<const uint8_t*>(ptr(gu.wr)), *gs = static_cast<const uint8_t*>(ptr(gu.sb));
    const uint8_t *dw = static_cast<const uint8_t*>(ptr(dn.wr)), *ds = static_cast<const uint8_t*>(ptr(dn.sb));
    const std::string f = record_file(dir, li);
    const int fd = ::open(f.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) fail("cannot create " + f);
    ::fchmod(fd, 0600);
    ::fcntl(fd, F_NOCACHE, 1);
    for (int e0 = 0; e0 < c.E; e0 += batch) {
      const int n = std::min(batch, c.E - e0);
      parallel_for(size_t(n), [&](size_t i) {
        const size_t ex = size_t(e0) + i;
        uint8_t* r = buf + i * stride;
        std::memset(r + R, 0, stride - R);
        std::memcpy(r + o[0], gw + ex * gwe, gwe);
        for (size_t g = 0; g < gg; ++g) std::memcpy(r + o[1] + g * gp * 8, gs + (g * size_t(c.E) * gp + ex * gp) * 8, gp * 8);
        std::memcpy(r + o[2], dw + ex * dwe, dwe);
        for (size_t g = 0; g < dg; ++g) std::memcpy(r + o[3] + g * dp * 8, ds + (g * size_t(c.E) * dp + ex * dp) * 8, dp * 8);
      });
      const uint8_t* p = buf;
      size_t left = size_t(n) * stride;
      uint64_t off = uint64_t(e0) * stride;
      while (left) {
        const ssize_t w = ::pwrite(fd, p, left, off_t(off));
        if (w <= 0) fail("expert records: write failed (" + f + ")");
        p += w;
        left -= size_t(w);
        off += uint64_t(w);
      }
    }
    if (::fsync(fd) != 0 || ::close(fd) != 0) fail("expert records: cannot finish " + f);
    if (li == 0) {  // check: records of layer 0 equal the per-expert load path's slots
      ExpertPool P;
      P.S = 3;
      auto pool = [&](int out, int in) {
        QW q;
        q.out = out;
        q.in = in;
        q.src_bits = q.bits = 4;
        q.group = group;
        q.wr = Buf(m.buffer(size_t(out) * size_t(in) / 2));
        q.sb = Buf(m.buffer(size_t(out / 2) * size_t(in / group) * 8));
        return q;
      };
      P.gu = pool(P.S * 2 * c.Ie, c.D);
      P.dn = pool(P.S * c.D, c.Ie);
      const std::vector<std::pair<int, int>> probe = {{0, 0}, {c.E / 2 + 1, 1}, {c.E - 1, 2}};
      Arena st(m.dev);
      id<MTLCommandBuffer> cb2 = [m.queue commandBuffer];
      id<MTLComputeCommandEncoder> e2 = [cb2 computeCommandEncoder];
      load_expert_slots(m, e2, 0, probe, P, st);  // the 9-piece path (no records open yet)
      [e2 endEncoding];
      commit_wait(cb2);
      const int rfd = ::open(f.c_str(), O_RDONLY);
      std::vector<uint8_t> rec(R);
      for (auto [ex, sl] : probe) {
        pread_all(rfd, rec.data(), R, uint64_t(ex) * stride);
        const uint8_t *pw = static_cast<const uint8_t*>(ptr(P.gu.wr)), *ps = static_cast<const uint8_t*>(ptr(P.gu.sb));
        const uint8_t *qw = static_cast<const uint8_t*>(ptr(P.dn.wr)), *qs = static_cast<const uint8_t*>(ptr(P.dn.sb));
        bool same = std::memcmp(rec.data() + o[0], pw + size_t(sl) * gwe, gwe) == 0 &&
                    std::memcmp(rec.data() + o[2], qw + size_t(sl) * dwe, dwe) == 0;
        for (size_t g = 0; g < gg; ++g)
          same = same && std::memcmp(rec.data() + o[1] + g * gp * 8, ps + (g * size_t(P.S) * gp + size_t(sl) * gp) * 8, gp * 8) == 0;
        for (size_t g = 0; g < dg; ++g)
          same = same && std::memcmp(rec.data() + o[3] + g * dp * 8, qs + (g * size_t(P.S) * dp + size_t(sl) * dp) * 8, dp * 8) == 0;
        if (!same) {
          ::close(rfd);
          fail("expert records: layer 0 expert " + std::to_string(ex) + " differs from the per-expert load path");
        }
      }
      ::close(rfd);
    }
    std::fprintf(stderr, "expert records: layer %d/%d written (%.0f s)\n", li + 1, c.L, ms_since(t0) / 1000.0);
  }
  paths::make_private_file(dir + "/meta.txt");
  std::ofstream out(dir + "/meta.txt");
  out << record_meta(m);
  if (!out) fail("expert records: cannot write meta.txt");
}

// Slots per layer for `gb` of experts (0: no cache).
inline void ensure_expert_pools(Model& m, double gb) {
  const Config& c = m.cfg;
  if (!m.xpool.empty() || gb <= 0 || !m.xs.present) return;
  const ExpertLayer& X0 = m.xs.layers[0];
  if (X0.bits != 4) fail("expert cache: only 4-bit experts so far");
  const size_t per_slot = size_t(2 * c.Ie) * size_t(c.D) / 2 + size_t(c.Ie) * size_t(c.D / X0.group) * 8 +
                          size_t(c.D) * size_t(c.Ie) / 2 + size_t(c.D / 2) * size_t(c.Ie / X0.group) * 8;
  const int S = int(std::min<double>(c.E, std::floor(gb * 1e9 / double(per_slot) / double(c.L))));
  if (S < 16) fail("expert cache: " + std::to_string(gb) + " GB is too small (fewer than 16 experts per layer)");
  m.xpool.resize(size_t(c.L));
  for (int li = 0; li < c.L; ++li) {
    ExpertPool& P = m.xpool[size_t(li)];
    P.S = S;
    auto pool = [&](int out, int in) {
      QW q;
      q.out = out;
      q.in = in;
      q.src_bits = 4;
      q.bits = 4;
      q.group = X0.group;
      static const bool priv = !env_on("ESPETERO_XPOOL_SHARED");  // pools in GPU-only memory unless =1
      q.wr = Buf(priv ? m.private_buffer(size_t(out) * size_t(in) / 2) : m.buffer(size_t(out) * size_t(in) / 2));
      q.sb = Buf(priv ? m.private_buffer(size_t(out / 2) * size_t(in / q.group) * 8)
                      : m.buffer(size_t(out / 2) * size_t(in / q.group) * 8));
      return q;
    };
    P.gu = pool(S * 2 * c.Ie, c.D);
    P.dn = pool(S * c.D, c.Ie);
    P.slot_of.assign(size_t(c.E), -1);
    P.expert_of.assign(size_t(S), -1);
    P.last_use.assign(size_t(S), 0);
  }
  m.xmap = m.buffer(sizeof(int32_t) * size_t(c.E));
  std::fprintf(stderr, "expert cache: %d of %d experts per layer (%.1f GB)\n", S, c.E,
               double(per_slot) * S * c.L / 1e9);
}

}  // namespace espetero::native
