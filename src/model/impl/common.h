// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Small shared helpers for the native model (errors, timing, parallel loops,
// float formats, memory-mapped files). Pure C++; included by the model's
// Objective-C++ translation unit.
#pragma once

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/storage/IOStorageDeviceCharacteristics.h>
#include <IOKit/storage/IOStorageProtocolCharacteristics.h>
#include <execinfo.h>

#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace espetero::native {

using Clock = std::chrono::steady_clock;

// ESPETERO_FAIL_BT=1: the call stack of every engine failure on stderr before
// it is thrown (symbol names as the binary carries them; a diagnostic for
// failures in harnesses that cannot run under a debugger).
[[noreturn]] inline void fail(const std::string& msg) {
  static const bool bt = [] {
    const char* v = std::getenv("ESPETERO_FAIL_BT");
    return v && *v == '1';
  }();
  if (bt) {
    void* frames[64];
    const int n = backtrace(frames, 64);
    std::fprintf(stderr, "[espetero] fail: %s\n", msg.c_str());
    backtrace_symbols_fd(frames, n, 2);
  }
  throw std::runtime_error(msg);
}

inline double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

inline size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Parallel loop over [0, n) on the global concurrent queue (GCD).
template <class F>
void parallel_for(size_t n, F&& f) {
  if (n == 0) return;
  if (n == 1) {
    f(size_t(0));
    return;
  }
  struct Ctx {
    F* f;
  } ctx{&f};
  dispatch_apply_f(n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &ctx,
                   [](void* c, size_t i) { (*static_cast<Ctx*>(c)->f)(i); });
}

// Weight files mapped for loading (MappedFile, below). Code that copies or
// converts a mapped range reads it with large parallel preads instead of
// page faults: a cold load read ~0.7 GB/s through page faults and ~6.5 GB/s
// with 8 MB preads on all cores (M1 Max SSD). The bytes are the file's
// either way.
struct MappedRange {
  const uint8_t* base;
  size_t size;
  int fd;
};
inline std::mutex& mapped_mu() {
  static std::mutex m;
  return m;
}
inline std::vector<MappedRange>& mapped_ranges() {
  static std::vector<MappedRange> v;
  return v;
}
// The file and offset of [p, p + n) when it lies in one mapped file.
inline bool mapped_at(const void* p, size_t n, int& fd, size_t& off) {
  const uint8_t* q = static_cast<const uint8_t*>(p);
  std::lock_guard<std::mutex> lk(mapped_mu());
  for (const MappedRange& r : mapped_ranges())
    if (q >= r.base && q + n <= r.base + r.size) {
      fd = r.fd;
      off = size_t(q - r.base);
      return true;
    }
  return false;
}
// n bytes at `off` into dst; false on an error or a short file.
inline bool pread_all(int fd, void* dst, size_t n, size_t off) {
  size_t got = 0;
  while (got < n) {
    const ssize_t r = ::pread(fd, static_cast<uint8_t*>(dst) + got, n - got, off_t(off + got));
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    got += size_t(r);
  }
  return true;
}

constexpr size_t kCopyPiece = size_t(8) << 20;

// ---------------------------------------------------------------- slow disks
// A spinning disk reads a weight file fastest front to back on one thread:
// a USB HDD read 264-278 MB/s that way and 176 MB/s in the loader's
// parallel pieces, which make the head seek between them.
// load_readers(fd): the threads that read weights from fd's volume,
// 0 = all cores in parallel pieces (SSDs). ESPETERO_LOAD_THREADS=N sets it
// for every file; otherwise it is 1 on a rotational or USB disk, a network
// share, or a volume listed in ESPETERO_VAULT_ROOTS (mount points, colon
// separated; none by default).
constexpr size_t kSeqPiece = size_t(64) << 20;

// Rotational medium or USB interconnect anywhere above the BSD disk `bsd`
// (an APFS volume reaches its physical store through its container).
inline bool slow_disk(const char* bsd) {
  io_service_t s = IOServiceGetMatchingService(kIOMainPortDefault, IOBSDNameMatching(kIOMainPortDefault, 0, bsd));
  if (!s) return false;
  auto prop = [&](CFStringRef dict, CFStringRef key, CFStringRef want) {
    const IOOptionBits up = kIORegistryIterateRecursively | kIORegistryIterateParents;
    CFTypeRef d = IORegistryEntrySearchCFProperty(s, kIOServicePlane, dict, kCFAllocatorDefault, up);
    bool hit = false;
    if (d && CFGetTypeID(d) == CFDictionaryGetTypeID()) {
      CFTypeRef v = CFDictionaryGetValue(CFDictionaryRef(d), key);
      hit = v && CFEqual(v, want);
    }
    if (d) CFRelease(d);
    return hit;
  };
  const bool slow =
      prop(CFSTR(kIOPropertyDeviceCharacteristicsKey), CFSTR(kIOPropertyMediumTypeKey),
           CFSTR(kIOPropertyMediumTypeRotationalKey)) ||
      prop(CFSTR(kIOPropertyProtocolCharacteristicsKey), CFSTR(kIOPropertyPhysicalInterconnectTypeKey),
           CFSTR(kIOPropertyPhysicalInterconnectTypeUSB));
  IOObjectRelease(s);
  return slow;
}

// Why reads from the volume of `sf` go one at a time ("" when they need not).
inline std::string slow_volume(const struct statfs& sf) {
  const std::string type = sf.f_fstypename, mnt = sf.f_mntonname, from = sf.f_mntfromname;
  if (type == "smbfs" || type == "nfs" || type == "afpfs" || type == "webdav") return "network share";
  const char* rv = std::getenv("ESPETERO_VAULT_ROOTS");
  const std::string roots = rv ? rv : "";
  for (size_t a = 0; a <= roots.size();) {
    size_t b = roots.find(':', a);
    if (b == std::string::npos) b = roots.size();
    std::string r = roots.substr(a, b - a);
    while (r.size() > 1 && r.back() == '/') r.pop_back();
    if (!r.empty() && r == mnt) return "vault";
    a = b + 1;
  }
  if (from.rfind("/dev/", 0) == 0 && slow_disk(from.c_str() + 5)) return "rotational or USB disk";
  return "";
}

inline int load_readers(int fd) {
  static const int forced = [] {
    const char* v = std::getenv("ESPETERO_LOAD_THREADS");
    return v && *v ? std::max(0, std::atoi(v)) : -1;
  }();
  if (forced >= 0) return forced;
  struct stat st;
  if (::fstat(fd, &st) != 0) return 0;
  static std::mutex mu;
  static std::map<dev_t, int> seen;  // per volume, decided once
  std::lock_guard<std::mutex> lk(mu);
  if (auto it = seen.find(st.st_dev); it != seen.end()) return it->second;
  struct statfs sf;
  std::string why;
  if (::fstatfs(fd, &sf) == 0) why = slow_volume(sf);
  if (!why.empty()) std::fprintf(stderr, "[espetero] weights on %s (%s): one reader, file order\n", sf.f_mntonname, why.c_str());
  return seen[st.st_dev] = why.empty() ? 0 : 1;
}

// n bytes of fd at `off` into dst: `piece`-sized preads on all cores, or on
// a slow volume kSeqPiece preads in file order on load_readers(fd) threads.
// False on an error or a short file.
inline bool read_pieces(int fd, void* dst, size_t n, size_t off, size_t piece) {
  uint8_t* d = static_cast<uint8_t*>(dst);
  const int rd = load_readers(fd);
  if (rd > 0) piece = std::max(piece, kSeqPiece);
  const size_t np = (n + piece - 1) / piece;
  std::atomic<size_t> next{0};
  std::atomic<bool> ok{true};
  auto run = [&](size_t) {  // the next piece in file order, until none is left
    for (size_t i; ok && (i = next++) < np;)
      if (!pread_all(fd, d + i * piece, std::min(piece, n - i * piece), off + i * piece)) ok = false;
  };
  parallel_for(rd > 0 ? std::min(size_t(rd), np) : np, [&](size_t w) { run(w); });
  return ok;
}

// A large copy on several threads, in 8 MB pieces; from a mapped weight file
// it reads the file instead of faulting the mapping in. Loading copies every
// weight: on one thread, through page faults, a 28 GB MoE took 37-52 s.
inline void parallel_copy(void* dst, const void* src, size_t n) {
  int fd = -1;
  size_t base = 0;
  if (mapped_at(src, n, fd, base) && load_readers(fd) > 0) {  // every size: a page fault is a seek too
    if (!read_pieces(fd, dst, n, base, kSeqPiece)) std::memcpy(dst, src, n);
    return;
  }
  if (n < 2 * kCopyPiece) {
    std::memcpy(dst, src, n);
    return;
  }
  const bool file = fd >= 0;
  parallel_for((n + kCopyPiece - 1) / kCopyPiece, [&](size_t i) {
    const size_t off = i * kCopyPiece, m = std::min(kCopyPiece, n - off);
    uint8_t* d = static_cast<uint8_t*>(dst) + off;
    if (!file || !pread_all(fd, d, m, base + off)) std::memcpy(d, static_cast<const uint8_t*>(src) + off, m);
  });
}

// Brings a mapped range into the file cache with parallel preads before code
// reads it through the mapping (GGUF rows converted on many threads, each
// faulting its own pages in). Nothing to do when the range is resident.
inline void prefetch_mapped(const void* src, size_t n) {
  int fd = -1;
  size_t base = 0;
  if (n < 2 * kCopyPiece || !mapped_at(src, n, fd, base)) return;
  {
    const size_t pg = size_t(::getpagesize());
    const uintptr_t a = reinterpret_cast<uintptr_t>(src) / pg * pg;
    const size_t len = reinterpret_cast<uintptr_t>(src) + n - a, pages = (len + pg - 1) / pg;
    std::vector<char> in(pages);
    if (::mincore(reinterpret_cast<void*>(a), len, in.data()) == 0) {
      size_t resident = 0;
      for (char c : in) resident += (c & 1);
      if (resident * 10 >= pages * 9) return;  // already (almost) all in memory
    }
  }
  if (const int rd = load_readers(fd); rd > 0) {  // file order, kSeqPiece at a time
    std::vector<uint8_t> scratch(std::min(n, kSeqPiece));
    for (size_t off = 0; off < n; off += kSeqPiece)
      if (!pread_all(fd, scratch.data(), std::min(kSeqPiece, n - off), base + off)) break;
    return;
  }
  parallel_for((n + kCopyPiece - 1) / kCopyPiece, [&](size_t i) {
    thread_local std::vector<uint8_t> scratch;
    if (scratch.size() < kCopyPiece) scratch.resize(kCopyPiece);
    const size_t off = i * kCopyPiece;
    pread_all(fd, scratch.data(), std::min(kCopyPiece, n - off), base + off);
  });
}

inline bool env_on(const char* name) {
  const char* v = std::getenv(name);
  return v && *v == '1';
}

inline bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline bool file_exists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}

inline bool is_dir(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

inline std::string read_file(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) fail("cannot open " + path);
  struct stat st;
  ::fstat(fd, &st);
  std::string out(size_t(st.st_size), '\0');
  size_t got = 0;
  while (got < out.size()) {
    const ssize_t r = ::read(fd, out.data() + got, out.size() - got);
    if (r <= 0) break;
    got += size_t(r);
  }
  ::close(fd);
  if (got != out.size()) fail("short read on " + path);
  return out;
}

inline std::string dirname_of(const std::string& path) {
  const size_t p = path.find_last_of('/');
  return p == std::string::npos ? std::string(".") : path.substr(0, p);
}

inline std::string basename_of(const std::string& path) {
  const size_t p = path.find_last_of('/');
  return p == std::string::npos ? path : path.substr(p + 1);
}

// ---------------------------------------------------------------- floats
inline float bf16_to_f32(uint16_t v) {
  const uint32_t u = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline uint16_t f32_to_bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);  // round to nearest even
  return uint16_t(u >> 16);
}

inline float bf16_round(float f) { return bf16_to_f32(f32_to_bf16_bits(f)); }

inline uint16_t f32_to_f16_bits(float f) {
  const __fp16 h = __fp16(f);
  uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}

inline float f16_to_f32(uint16_t v) {
  __fp16 h;
  std::memcpy(&h, &v, 2);
  return float(h);
}

// Element i of a dense tensor stored as F32, F16 or BF16.
inline float dense_at(const uint8_t* src, const std::string& dtype, size_t i) {
  if (dtype == "BF16") return bf16_to_f32(reinterpret_cast<const uint16_t*>(src)[i]);
  if (dtype == "F16") return float(reinterpret_cast<const __fp16*>(src)[i]);
  if (dtype == "F32") return reinterpret_cast<const float*>(src)[i];
  fail("unsupported dtype " + dtype);
}

// ---------------------------------------------------------------- mmap
struct MappedFile {
  void* base = nullptr;
  size_t size = 0;
  int fd = -1;  // kept open: parallel_copy and prefetch_mapped read through it
  std::string path;
  explicit MappedFile(const std::string& p) : path(p) {
    fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open " + p);
    struct stat st;
    ::fstat(fd, &st);
    size = size_t(st.st_size);
    base = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
      base = nullptr;
      ::close(fd);
      fd = -1;
      fail("cannot mmap " + p);
    }
    // Weights are read once, front to back, during the load.
    ::madvise(base, size, MADV_SEQUENTIAL);
    std::lock_guard<std::mutex> lk(mapped_mu());
    mapped_ranges().push_back({static_cast<const uint8_t*>(base), size, fd});
  }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  ~MappedFile() {
    {
      std::lock_guard<std::mutex> lk(mapped_mu());
      auto& v = mapped_ranges();
      v.erase(std::remove_if(v.begin(), v.end(), [&](const MappedRange& r) { return r.base == base; }), v.end());
    }
    if (base) ::munmap(base, size);
    if (fd >= 0) ::close(fd);
  }
  const uint8_t* at(size_t off) const { return static_cast<const uint8_t*>(base) + off; }
};

}  // namespace espetero::native
