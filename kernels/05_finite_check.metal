// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// ---- Debug: NaN / Inf / outlier detector (ESPETERO_CHECK_FINITE; src/model/impl/finite_check.h).
// After a chosen dispatch, finite_check scans its destination region (rows x
// cols, row stride ld, fp32 / fp16 / bf16) and records the first element with
// |x| >= threshold (NaN and Inf always count) in its own record: the number
// of offending elements, the smallest offending index, the threadgroups done
// and the first offender's value. Bits, not float compares: the library
// compiles with fast math, under which NaN tests may fold away. Buffer
// indices 28-30 are used by no other kernel, so the bindings of the dispatch
// being checked stay as they were.

struct FiniteCheckP {
  uint rows, cols, ld, dtype;  // dtype: 0 fp32, 1 fp16, 2 bf16
  uint thr_bits;               // |x| bits >= thr_bits offend (0x7F800000: NaN and Inf only)
  uint elem;                   // finite_poison: the element (row * ld + col) to overwrite
  uint pad0, pad1;
};

// The fp32 bits of element o (half NaN / Inf mapped by hand: exact under fast math).
static inline uint fc_bits(device const uchar* x, ulong o, uint dtype) {
  if (dtype == 0) return ((device const uint*)x)[o];
  const uint h = ((device const ushort*)x)[o];
  if (dtype == 2) return h << 16;
  if ((h & 0x7C00u) == 0x7C00u) return ((h & 0x8000u) << 16) | 0x7F800000u | ((h & 0x3FFu) << 13);
  return as_type<uint>(float(as_type<half>(ushort(h))));
}

kernel void finite_check(device const uchar* x [[buffer(28)]], device atomic_uint* rec [[buffer(29)]],
                         constant FiniteCheckP& p [[buffer(30)]], uint tid [[thread_position_in_grid]],
                         uint nthr [[threads_per_grid]], uint lid [[thread_index_in_threadgroup]],
                         uint ntg [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint n = p.rows * p.cols;
  uint first = 0xFFFFFFFFu, cnt = 0;
  for (uint i = tid; i < n; i += nthr) {
    const uint r = i / p.cols, c = i - r * p.cols;
    const uint b = fc_bits(x, ulong(r) * ulong(p.ld) + c, p.dtype) & 0x7FFFFFFFu;
    if (b >= p.thr_bits) {
      first = min(first, i);
      ++cnt;
    }
  }
  first = simd_min(first);
  cnt = simd_sum(cnt);
  threadgroup uint tf[32], tc[32];
  threadgroup bool last;
  if (lane == 0) tf[sg] = first, tc[sg] = cnt;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    const uint nsg = (256 + 31) / 32;
    uint f = 0xFFFFFFFFu, k = 0;
    for (uint s = 0; s < nsg; ++s) f = min(f, tf[s]), k += tc[s];
    if (k) {
      atomic_fetch_add_explicit(rec + 0, k, memory_order_relaxed);
      atomic_fetch_min_explicit(rec + 1, f, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
    last = atomic_fetch_add_explicit(rec + 2, 1u, memory_order_relaxed) == ntg - 1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!last || lid != 0) return;
  // The last threadgroup: the first offender's value (the region was written by an earlier dispatch).
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
  const uint i = atomic_load_explicit(rec + 1, memory_order_relaxed);
  if (i == 0xFFFFFFFFu) return;
  const uint r = i / p.cols, c = i - r * p.cols;
  atomic_store_explicit(rec + 3, fc_bits(x, ulong(r) * ulong(p.ld) + c, p.dtype), memory_order_relaxed);
}

// ESPETERO_CHECK_FINITE_POISON (tests): a NaN into one element of a checked region, before its check.
kernel void finite_poison(device uchar* x [[buffer(28)]], constant FiniteCheckP& p [[buffer(30)]]) {
  if (p.dtype == 0) ((device uint*)x)[p.elem] = 0x7FC00000u;
  else ((device ushort*)x)[p.elem] = p.dtype == 2 ? ushort(0x7FC0) : ushort(0x7E00);
}
