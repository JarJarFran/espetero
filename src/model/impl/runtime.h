// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Metal runtime: device, kernel library (compiled at load from embedded
// source), pipelines specialized by function constants, arenas of shared
// untracked buffers, and an encoder that runs dispatches concurrently and
// inserts memory barriers only where a dispatch depends on an earlier one.
//
// Why concurrent dispatch: a serial compute encoder makes every dispatch wait
// for the previous one to drain. A decode step is a few hundred short kernels,
// so those bubbles add up; with concurrent dispatch the GPU overlaps
// independent kernels and the tail of one kernel with the head of the next.
// Every dispatch declares what it reads and writes (weights excluded: they are
// immutable after load) and the encoder places barriers from that.
#pragma once

#include <functional>

#include <dlfcn.h>
#include <mach/mach_time.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <mutex>
#include <thread>

#include "common.h"

namespace espetero::native {

struct Buf {
  id<MTLBuffer> b = nil;
  size_t off = 0;
  Buf() = default;
  Buf(id<MTLBuffer> x, size_t o = 0) : b(x), off(o) {}  // NOLINT: implicit on purpose
  explicit operator bool() const { return b != nil; }
};

inline void* ptr(const Buf& b) { return static_cast<uint8_t*>([b.b contents]) + b.off; }

constexpr MTLResourceOptions kShared = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;

// ESPETERO_ALLOCLOG=MB: logs every Metal allocation of at least MB
// megabytes with the device's running total and the calling function (to
// account for the memory around the expert cache).
inline void alloc_log(id<MTLDevice> dev, const char* kind, size_t nbytes, void* ra) {
  static const long thr = [] {
    const char* v = std::getenv("ESPETERO_ALLOCLOG");
    return v ? std::atol(v) : -1L;
  }();
  if (thr < 0 || nbytes < (size_t(thr) << 20)) return;
  Dl_info di{};
  const char* fn = dladdr(ra, &di) && di.dli_sname ? di.dli_sname : "?";
  std::fprintf(stderr, "[alloc] %s %.1f MB (device total %.2f GB) from %s\n", kind, double(nbytes) / 1e6,
               double([dev currentAllocatedSize]) / 1e9, fn);
}

// ---- Residency (macOS 15+): the model's weights and the expert pools in
// one residency set attached to the command queues, made resident when they are
// added rather than by the first command buffer that uses them (Flash-Next's
// first pass after loading: ~320 -> 58-70 ms). It does not keep macOS from
// compressing shared buffers (shared pools at 42 GB thrashed with it); the
// pools are GPU-private for that. ESPETERO_RESIDENCY=0 turns it off.
struct Residency {
  id<MTLResidencySet> set = nil;
  bool dirty = false;
  std::mutex mu;
  static Residency& get() {
    static Residency r;
    return r;
  }
  void attach(id<MTLDevice> dev, id<MTLCommandQueue> q) {
    std::lock_guard<std::mutex> lk(mu);
    if (!set) {
      const char* v = std::getenv("ESPETERO_RESIDENCY");
      if (v && *v == '0') return;
      if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
        d.label = @"espetero";
        d.initialCapacity = 1024;
        NSError* err = nil;
        set = [dev newResidencySetWithDescriptor:d error:&err];
      }
      if (!set) return;
    }
    [q addResidencySet:set];
  }
  void add(id<MTLBuffer> b) {
    if (!set || !b) return;
    std::lock_guard<std::mutex> lk(mu);
    [set addAllocation:b];
    dirty = true;
  }
  void remove(id<MTLBuffer> b) {
    if (!set || !b) return;
    std::lock_guard<std::mutex> lk(mu);
    [set removeAllocation:b];
    [set commit];  // drop the set's reference now, so the memory is released with the arena
  }
  // Apply additions and removals, and make the allocations resident.
  void commit() {
    if (!set) return;
    std::lock_guard<std::mutex> lk(mu);
    if (!dirty) return;
    [set commit];
    [set requestResidency];
    dirty = false;
  }
};

// Bump allocator over large shared buffers (weights, host staging).
class Arena {
 public:
  // resident: the model's weights join the residency set; scratch and
  // staging arenas do not.
  explicit Arena(id<MTLDevice> dev, size_t chunk = size_t(1) << 30, bool resident = false)
      : dev_(dev), chunk_(chunk), resident_(resident) {}
  ~Arena() {
    if (resident_)
      for (id<MTLBuffer> b : buffers_) Residency::get().remove(b);
  }
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  Buf alloc(size_t nbytes) {
    nbytes = align_up(std::max<size_t>(nbytes, 16), 256);
    // First fit over the free tails of the buffers, in allocation order: after
    // reset() the buffers are reused before anything new is allocated (a fresh
    // buffer costs the GPU a mapping of its pages at the first submission that
    // uses it: ~30-50 ms per GB), and a tensor too large for the current chunk
    // no longer strands that chunk's tail (Flash-Next's weights, 7.8 GB,
    // took 9.19 GB of 1 GB chunks when only the last chunk took allocations).
    for (size_t i = 0; i < buffers_.size(); ++i) {
      if (fill_[i] + nbytes <= len_[i]) {
        Buf out{buffers_[i], fill_[i]};
        fill_[i] += nbytes;
        total_ += nbytes;
        return out;
      }
    }
    const size_t cap = std::max(nbytes, chunk_);
    id<MTLBuffer> b = [dev_ newBufferWithLength:cap options:kShared];
    if (!b) fail("Metal allocation of " + std::to_string(cap >> 20) + " MB failed");
    if (resident_) Residency::get().add(b);
    alloc_log(dev_, resident_ ? "arena (resident)" : "arena", cap, __builtin_return_address(0));
    buffers_.push_back(b);
    len_.push_back(cap);
    fill_.push_back(nbytes);
    held_ += cap;
    total_ += nbytes;
    return Buf{b, 0};
  }

  Buf copy(const void* src, size_t n) {
    Buf b = alloc(n);
    parallel_copy(ptr(b), src, n);
    return b;
  }

  size_t total() const { return total_; }  // bytes handed out
  size_t held() const { return held_; }    // bytes of the buffers behind them
  // Forget every allocation but keep the buffers for reuse (callers make sure
  // the GPU no longer reads what was allocated).
  void reset() { std::fill(fill_.begin(), fill_.end(), size_t(0)); }

 private:
  id<MTLDevice> dev_;
  size_t chunk_;
  size_t total_ = 0, held_ = 0;
  bool resident_ = false;
  std::vector<id<MTLBuffer>> buffers_;
  std::vector<size_t> len_, fill_;
};

struct Runtime {
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> lib = nil;
  std::unique_ptr<Arena> arena;
  std::mutex pso_mu;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pso;
  int dispatches = 0;
  int barriers = 0;
  bool profile = false;
  bool serial = false;  // serial dispatch, no barrier tracking
  bool fma34 = false;   // the library has the 3-4-row JIT FMA GEMVs (ESPETERO_FMA34=1)
  bool barrier_all = false;  // experiment: concurrent encoder with a barrier before every dispatch
  // Decode chains: consecutive passes of a chain are separate command
  // buffers ordered on the GPU by chain_ev: a pass waits for value chain_wait
  // before its encoder and signals chain_signal after it (0: none).
  id<MTLEvent> chain_ev = nil;
  uint64_t chain_wait = 0, chain_signal = 0;
  std::map<std::string, std::pair<double, int>> prof;

  void init(const std::string& kernel_source) {
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) fail("no Metal device");
    queue = [dev newCommandQueueWithMaxCommandBufferCount:64];
    Residency::get().attach(dev, queue);
    // Serial dispatch measured faster than concurrent dispatch with tracked
    // barriers on an M1 Max (4-20% on prefill passes, equal on decode steps);
    // ESPETERO_CONCURRENT=1 switches to the concurrent encoder.
    serial = !env_on("ESPETERO_CONCURRENT");
    NSString* src = [[NSString alloc] initWithBytes:kernel_source.data()
                                             length:kernel_source.size()
                                           encoding:NSUTF8StringEncoding];
    MTLCompileOptions* opts = [MTLCompileOptions new];
    // 3.2 for device-scope atomic fences (last-threadgroup finalization of
    // fused norms); every kernel also compiles under 3.1 otherwise.
    opts.languageVersion = MTLLanguageVersion3_2;
    // ESPETERO_PF=1 compiles the GEMV K loops with the next step's weights prefetched (measured slower).
    // ESPETERO_DIAG=1 adds the diagnostic GEMV instantiations of `bench_native --micro gemv`.
    const char* pf = std::getenv("ESPETERO_PF");
    const char* dg = std::getenv("ESPETERO_DIAG");
    // ESPETERO_FMA34=1 adds the 3-4-row JIT FMA GEMVs (served when gemv_max_m >= 3).
    const char* f34 = std::getenv("ESPETERO_FMA34");
    fma34 = f34 && *f34 == '1';
    opts.preprocessorMacros = @{@"ESPETERO_PF" : @(pf && *pf == '1' ? 1 : 0), @"ESPETERO_DIAG" : @(dg && *dg == '1' ? 1 : 0),
                                @"ESPETERO_FMA34" : @(fma34 ? 1 : 0)};
    NSError* error = nil;
    lib = [dev newLibraryWithSource:src options:opts error:&error];
    if (!lib) fail(std::string("kernel compile failed: ") + [[error localizedDescription] UTF8String]);
    arena = std::make_unique<Arena>(dev, size_t(1) << 30, true);
  }

  static std::string key_of(const std::string& name, int fc) { return fc ? name + "#" + std::to_string(fc) : name; }

  id<MTLFunction> function(const std::string& name, int fc) {
    NSString* fname = [NSString stringWithUTF8String:name.c_str()];
    NSError* error = nil;
    id<MTLFunction> fn = nil;
    if (fc) {
      // Quantized kernels read (group, scale mode) as function constants 0/1;
      // a gate/up matrix with a GELU gate (Gemma 4) also sets constant 2
      // (left undefined otherwise: the SwiGLU kernels compile as before).
      MTLFunctionConstantValues* v = [MTLFunctionConstantValues new];
      const int g = fc & 0xFFFF, mode = (fc >> 16) & 0xFF, act = fc >> 24;
      [v setConstantValue:&g type:MTLDataTypeInt atIndex:0];
      [v setConstantValue:&mode type:MTLDataTypeInt atIndex:1];
      if (act) [v setConstantValue:&act type:MTLDataTypeInt atIndex:2];
      fn = [lib newFunctionWithName:fname constantValues:v error:&error];
    } else {
      fn = [lib newFunctionWithName:fname];
    }
    if (!fn) fail("missing kernel " + name);
    return fn;
  }

  // Pipeline for `name` specialized by `fc` (0: none). Thread-safe.
  id<MTLComputePipelineState> kernel(const std::string& name, int fc = 0) {
    const std::string key = key_of(name, fc);
    {
      std::lock_guard<std::mutex> lk(pso_mu);
      auto it = pso.find(key);
      if (it != pso.end()) return it->second;
    }
    id<MTLFunction> fn = function(name, fc);
    NSError* error = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&error];
    if (!p) fail("pipeline " + name + ": " + [[error localizedDescription] UTF8String]);
    std::lock_guard<std::mutex> lk(pso_mu);
    return pso.emplace(key, p).first->second;
  }

  // Build many pipelines in parallel (first requests would otherwise pay for
  // each one on the critical path).
  void prewarm(const std::vector<std::pair<std::string, int>>& list) {
    std::vector<std::pair<std::string, int>> todo;
    {
      std::lock_guard<std::mutex> lk(pso_mu);
      for (const auto& [n, fc] : list)
        if (!pso.count(key_of(n, fc))) todo.push_back({n, fc});
    }
    std::sort(todo.begin(), todo.end());
    todo.erase(std::unique(todo.begin(), todo.end()), todo.end());
    std::atomic<int> errors{0};
    std::string first_error;
    std::mutex err_mu;
    parallel_for(todo.size(), [&](size_t i) {
      try {
        kernel(todo[i].first, todo[i].second);
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(err_mu);
        if (!errors++) first_error = e.what();
      }
    });
    if (errors) fail(first_error);
  }

  id<MTLBuffer> buffer(size_t nbytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:std::max<size_t>(nbytes, 16) options:kShared];
    if (!b) fail("Metal allocation of " + std::to_string(nbytes >> 20) + " MB failed");
    alloc_log(dev, "buffer", nbytes, __builtin_return_address(0));
    return b;
  }
  // GPU-only memory: written and read by kernels only (expert pools).
  id<MTLBuffer> private_buffer(size_t nbytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:std::max<size_t>(nbytes, 16)
                                       options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked];
    if (!b) fail("Metal allocation of " + std::to_string(nbytes >> 20) + " MB (private) failed");
    Residency::get().add(b);
    alloc_log(dev, "private", nbytes, __builtin_return_address(0));
    return b;
  }
};

// A byte range of a buffer for hazard tracking.
struct Span {
  const void* buf = nullptr;
  size_t lo = 0, hi = SIZE_MAX;
};

inline bool overlaps(const Span& a, const Span& b) { return a.buf == b.buf && a.lo < b.hi && b.lo < a.hi; }

// Compute encoder over one command buffer (see the header comment).
// ESPETERO_CBLOG=ms: log every command buffer whose GPU time exceeds ms,
// with the phase that encoded it. macOS aborts long command buffers while
// the display is in use (kIOGPUCommandBufferCallbackErrorImpactingInteractivity).
inline double cb_log_ms() {
  static const double v = [] {
    const char* e = std::getenv("ESPETERO_CBLOG");
    return e && *e ? std::atof(e) : 0.0;
  }();
  return v;
}
inline void cb_log(id<MTLCommandBuffer> cb, const char* what) {
  const double lim = cb_log_ms();
  if (lim <= 0) return;
  const double ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
  if (ms > lim) std::fprintf(stderr, "[espetero] command buffer %.0f ms (%s)\n", ms, what && *what ? what : "pass");
}

class Enc {
 public:
  Runtime& rt;
  id<MTLCommandBuffer> cb;
  id<MTLComputeCommandEncoder> e;
  std::string cur;  // profiling label

  Enc(Runtime& r, id<MTLCommandBuffer> c) : rt(r), cb(c) {
    Residency::get().commit();
    if (rt.chain_wait) [cb encodeWaitForEvent:rt.chain_ev value:rt.chain_wait];
    open();
  }
  ~Enc() {
    for (id<MTLCommandBuffer> s : submitted_) [s waitUntilCompleted];  // a failed pass's submitted work
  }

  void pipe(const std::string& name, int fc = 0) {
    // ESPETERO_PIPE_LOG=1: every pipeline as it is encoded (with profiling on, each
    // dispatch runs and is waited for alone: the last line names a hung kernel).
    static const bool pipe_log = env_on("ESPETERO_PIPE_LOG");
    if (pipe_log) std::fprintf(stderr, "[pipe] %s\n", name.c_str());
    [e setComputePipelineState:rt.kernel(name, fc)];
    if (rt.profile) cur = name;
    if (trace_) cur_name_ = name;
  }
  void label(const std::string& s) {
    if (rt.profile) cur = s;
  }
  void set(int i, id<MTLBuffer> b, size_t off = 0) { [e setBuffer:b offset:off atIndex:i]; }
  void set(int i, const Buf& b, size_t extra = 0) { [e setBuffer:b.b offset:b.off + extra atIndex:i]; }
  template <typename T>
  void bytes(int i, const T& v) {
    [e setBytes:&v length:sizeof(T) atIndex:i];
  }
  void raw(int i, const void* p, size_t n) { [e setBytes:p length:n atIndex:i]; }

  // Declare the next dispatch's reads and writes (whole buffer by default).
  void rd(id<MTLBuffer> b, size_t lo = 0, size_t hi = SIZE_MAX) { reads_.push_back({(__bridge const void*)b, lo, hi}); }
  void wr(id<MTLBuffer> b, size_t lo = 0, size_t hi = SIZE_MAX) { writes_.push_back({(__bridge const void*)b, lo, hi}); }
  void rd(const Buf& b, size_t n = SIZE_MAX) { rd(b.b, b.off, n == SIZE_MAX ? SIZE_MAX : b.off + n); }
  void wr(const Buf& b, size_t n = SIZE_MAX) { wr(b.b, b.off, n == SIZE_MAX ? SIZE_MAX : b.off + n); }

  void groups(MTLSize grid, MTLSize tg) {
    if (grid.width * grid.height * grid.depth == 0) return drop();
    hazards();
    [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    done();
  }
  void threads(MTLSize grid, MTLSize tg) {
    if (grid.width * grid.height * grid.depth == 0) return drop();
    hazards();
    [e dispatchThreads:grid threadsPerThreadgroup:tg];
    done();
  }
  // Threadgroup counts written by an earlier dispatch (MTLDispatchThreadgroupsIndirectArguments at `off`).
  void groups_indirect(id<MTLBuffer> args, size_t off, MTLSize tg) {
    rd(args, off, off + 12);
    hazards();
    [e dispatchThreadgroupsWithIndirectBuffer:args indirectBufferOffset:off threadsPerThreadgroup:tg];
    done();
  }
  // Full barrier (e.g. after blits into buffers the next kernels read).
  void barrier() {
    if (!rt.serial) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    ++rt.barriers;
    since_r_.clear();
    since_w_.clear();
  }
  // Commit what is encoded so far, wait for it, and continue in a new command
  // buffer: host work between GPU phases of one pass (Qwen4-Exp experts read
  // from the SSD once the routing is known). Not for chained passes.
  // Returns the flushed command buffer's GPU time (ms).
  // Called before every commit of this encoder's command buffer (the expert
  // cache waits there for the reads its encoded copies depend on).
  std::function<void()> before_commit;

  double flush() {
    [e endEncoding];
    if (before_commit) before_commit();
    const double t_commit = host_seconds();
    [cb commit];
    [cb waitUntilCompleted];
    const double t_done = host_seconds();
    if (flush_log) {
      flush_log->queue += ([cb GPUStartTime] - t_commit) * 1000.0;
      flush_log->exec += ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
      flush_log->wake += (t_done - [cb GPUEndTime]) * 1000.0;
      ++flush_log->n;
    }
    cb_log(cb, phase);
    if ([cb status] != MTLCommandBufferStatusCompleted) {
      NSError* err = [cb error];
      fail(std::string("GPU command buffer failed: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
    }
    check_submitted();
    const double gpu = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
    cb = [rt.queue commandBufferWithUnretainedReferences];
    open();
    since_r_.clear();
    since_w_.clear();
    return gpu;
  }
  // Commit what is encoded so far without waiting and without before_commit
  // (the work must not depend on reads still in flight), and continue in a
  // new command buffer (the resident experts' tiles run while the SSD
  // reads the missing ones). The caller commits the next one only after this
  // one completed (the expert pools are untracked). Its status is checked at
  // the next flush or check_submitted().
  id<MTLCommandBuffer> submit() {
    [e endEncoding];
    [cb commit];
    id<MTLCommandBuffer> sent = cb;
    submitted_.push_back(sent);
    cb = [rt.queue commandBufferWithUnretainedReferences];
    open();
    since_r_.clear();
    since_w_.clear();
    return sent;
  }
  // Fail like flush() if a submitted command buffer did not complete (call
  // after a later command buffer of the queue completed).
  void check_submitted() {
    std::vector<id<MTLCommandBuffer>> sent;
    sent.swap(submitted_);
    for (id<MTLCommandBuffer> s : sent) {
      [s waitUntilCompleted];
      if ([s status] != MTLCommandBufferStatusCompleted) {
        NSError* err = [s error];
        fail(std::string("GPU command buffer failed: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
      }
    }
  }
  // GPU-side handshake inside one command buffer: signal `a` = va when the
  // work so far is done, then wait until `b` >= vb (set by the host or by
  // another queue) before the next dispatch (Qwen4-Exp expert cache).
  void event_signal_wait(id<MTLSharedEvent> a, uint64_t va, id<MTLSharedEvent> b, uint64_t vb) {
    [e endEncoding];
    [cb encodeSignalEvent:a value:va];
    [cb encodeWaitForEvent:b value:vb];
    open();
    since_r_.clear();
    since_w_.clear();
  }
  id<MTLCommandBuffer> finish() {
    [e endEncoding];
    e = nil;
    if (before_commit) before_commit();
    if (rt.chain_signal) [cb encodeSignalEvent:rt.chain_ev value:rt.chain_signal];
    return cb;
  }

 private:
  void open() {
    e = rt.serial ? [cb computeCommandEncoder] : [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  }
  void drop() {
    reads_.clear();
    writes_.clear();
  }
  void hazards() {
    if (rt.serial || rt.profile) {
      drop();
      return;
    }
    bool need = rt.barrier_all;
    for (const Span& w : writes_) {
      for (const Span& s : since_r_) need = need || overlaps(w, s);
      for (const Span& s : since_w_) need = need || overlaps(w, s);
    }
    for (const Span& r : reads_)
      for (const Span& s : since_w_) need = need || overlaps(r, s);
    if (need) barrier();
    if (trace_) std::fprintf(stderr, "%s%s\n", need ? "| " : "  ", cur_name_.c_str());
    since_r_.insert(since_r_.end(), reads_.begin(), reads_.end());
    since_w_.insert(since_w_.end(), writes_.begin(), writes_.end());
    drop();
  }
  void done() {
    ++rt.dispatches;
    if (!rt.profile) return;
    // Profiling: every dispatch in its own command buffer, timed.
    [e endEncoding];
    if (before_commit) before_commit();
    [cb commit];
    [cb waitUntilCompleted];
    auto& slot = rt.prof[cur];
    slot.first += ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
    slot.second += 1;
    cb = [rt.queue commandBuffer];
    open();
  }

 public:
  // Host time in the GPUStartTime / GPUEndTime timebase (seconds).
  static double host_seconds() {
    static mach_timebase_info_data_t tb = [] {
      mach_timebase_info_data_t t;
      mach_timebase_info(&t);
      return t;
    }();
    return double(mach_absolute_time()) * double(tb.numer) / double(tb.denom) * 1e-9;
  }
  struct FlushLog {
    double queue = 0, exec = 0, wake = 0;
    int n = 0;
  };
  FlushLog* flush_log = nullptr;  // diagnostics: where flush() time goes
  char phase[48] = {0};            // phase being encoded (ESPETERO_CBLOG)

 private:
  std::vector<Span> reads_, writes_, since_r_, since_w_;
  std::vector<id<MTLCommandBuffer>> submitted_;
  bool trace_ = env_on("ESPETERO_TRACE_BARRIERS");
  std::string cur_name_;
};

// Commit and wait. Very short passes spin on the status for a lower wake-up
// latency than waitUntilCompleted's; everything else blocks. A spinning core
// draws package power that the (power-limited) GPU could use: a 27B decode
// step is 60-90 ms, and the spin's ~0.1-0.3 ms of wake-up latency is not
// worth it there.
inline void commit_wait(id<MTLCommandBuffer> cb, double expect_ms = 0) {
  Residency::get().commit();
  [cb commit];
  if (expect_ms > 0 && expect_ms < 3) {
    const auto t0 = Clock::now();
    while ([cb status] < MTLCommandBufferStatusCompleted) {
      if (ms_since(t0) > 4 * expect_ms + 2) {
        [cb waitUntilCompleted];
        break;
      }
      std::this_thread::yield();
    }
  } else {
    [cb waitUntilCompleted];
  }
  cb_log(cb, "pass end");
  if ([cb status] != MTLCommandBufferStatusCompleted) {
    NSError* err = [cb error];
    fail(std::string("GPU command buffer failed: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
  }
}

inline double gpu_ms(id<MTLCommandBuffer> cb) { return ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0; }

}  // namespace espetero::native
