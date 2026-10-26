// ---------------------------------------------------------------------------
// EQV: vector-quantized routed experts (Qwen3.8-Flash-Next route B, ~1.7 bpw).
//
// A row of K inputs is K / 8 vectors of 8 weights. Vector v of row r is
// scale(r, v / 32) * cb[code(r, v)], with a KB-bit code (9..16) and an fp16
// scale per 256 inputs (the last one partial when K is not a multiple of 256,
// e.g. 640 = 256 + 256 + 128). Codes are two planes: lo [rows][nv] bytes (bits
// 0-7) and hi [rows][hw] words holding the KB - 8 upper bits of every vector as
// one bit stream per row (vector v at bits (KB - 8) v ...). Codebooks: fp16
// [2^KB][8] per matrix; the fused gate/up stack (gate and up rows interleaved by
// 8, as every MoE matrix here) keeps gate's codebook and then up's, and row r
// reads codebook (r / 8) & 1.
//
// moe_vq_gemv: the grouped GEMV of the MoE tiles (moe_sort's table, as
// moe_gemv): one threadgroup per (tile, 32-row block), 8 simdgroups of 4 rows,
// each lane walks the vectors v = lane + 32 i of its rows, MM tile rows at once.

struct MoeVqParams {
  int Ne;      // output rows per expert
  int K;       // inputs per row
  int lda;
  int ldy;
  int gather;  // 1: tile row r reads A row rowmap[r]
  int direct;  // 1: exact direct grid, the tile count is not read
  int kbits;   // code bits per vector
  int nv;      // vectors per row (K / 8)
  int hw;      // high-bit words per row
  int ns;      // scales per row (ceil(K / 256))
  int cbsel;   // 1: two codebooks, row r uses (r / 8) & 1 (fused gate/up)
};

static inline uint vq_code(device const uchar* lo, device const uint* hi, ulong r, int v, constant MoeVqParams& p) {
  const int kb = p.kbits - 8;
  const uint bit = uint(v) * uint(kb);
  device const uint* hr = hi + r * ulong(p.hw);
  const uint w = bit / 32, sh = bit % 32;
  uint h = hr[w] >> sh;
  if (sh + uint(kb) > 32) h |= hr[w + 1] << (32 - sh);
  return uint(lo[r * ulong(p.nv) + ulong(v)]) | ((h & ((1u << kb) - 1u)) << 8);
}

template <int MM, int EPI>
kernel void moe_vq_gemv(
    device const half* A [[buffer(0)]],
    device const uchar* lo [[buffer(1)]],
    device const uint* hi [[buffer(2)]],
    device const half* sc [[buffer(3)]],
    device void* Yv [[buffer(4)]],
    constant MoeVqParams& p [[buffer(5)]],
    device const int4* tiles [[buffer(6)]],
    device const uint* ntiles [[buffer(7)]],
    device const int* rowmap [[buffer(8)]],
    device const half4* cb [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint tix = tgid.x, bix = tgid.y;
  if (!p.direct && tix >= ntiles[0]) return;
  const int4 ti = tiles[tix];
  const int e = ti.x, r0 = ti.y, rows = ti.z;
  const int nloc = int(bix) * 32;
  device const half* arow[MM];
  UNROLL for (int m = 0; m < MM; ++m) {
    const int ri = m < rows ? (p.gather ? rowmap[r0 + m] : r0 + m) : 0;
    arow[m] = A + ulong(ri) * ulong(p.lda);
  }
  threadgroup float part[MM][32];
  const int rl0 = nloc + int(sgid) * 4;
  float acc[4][MM];
  UNROLL for (int rr = 0; rr < 4; ++rr)
    UNROLL for (int m = 0; m < MM; ++m) acc[rr][m] = 0.0f;
  const ulong rbase = ulong(e) * ulong(p.Ne);
  for (int i = 0; i * 32 < p.nv; ++i) {
    const int v = i * 32 + int(lane);
    if (v >= p.nv) break;
    float4 xa[MM], xb[MM];
    UNROLL for (int m = 0; m < MM; ++m) {
      if (m < rows) {
        device const half4* xp = (device const half4*)(arow[m] + v * 8);
        xa[m] = float4(xp[0]);
        xb[m] = float4(xp[1]);
      } else {
        xa[m] = xb[m] = float4(0.0f);
      }
    }
    UNROLL for (int rr = 0; rr < 4; ++rr) {
      const int rl = rl0 + rr;
      const ulong r = rbase + ulong(rl);
      const uint code = vq_code(lo, hi, r, v, p);
      const uint cbi = p.cbsel ? uint((rl / 8) & 1) : 0u;
      device const half4* c = cb + ((ulong(cbi) << p.kbits) + ulong(code)) * 2;
      const float4 ca = float4(c[0]), cc = float4(c[1]);
      const float s = float(sc[r * ulong(p.ns) + ulong(i)]);
      UNROLL for (int m = 0; m < MM; ++m) acc[rr][m] += s * (dot(ca, xa[m]) + dot(cc, xb[m]));
    }
  }
  UNROLL for (int rr = 0; rr < 4; ++rr)
    UNROLL for (int m = 0; m < MM; ++m) {
      const float t = simd_sum(acc[rr][m]);
      if (lane == 0) part[m][int(sgid) * 4 + rr] = t;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgid == 0) {
    const int c = int(lane);
    UNROLL for (int m = 0; m < MM; ++m) {
      if (m >= rows) break;
      const float v = part[m][c];
      const ulong orow = ulong(r0 + m) * ulong(p.ldy);
      if (EPI == EPI_SWIGLU) {
        if ((c / 8) % 2 == 0 && nloc + c < p.Ne) {
          const float u = part[m][c + 8];
          const int out = ((nloc + (c / 8) * 8) / 16) * 8 + (c % 8);
          ((device half*)Yv)[orow + out] = half(glu_gate(v) * u);
        }
      } else if (nloc + c < p.Ne) {
        device float* y = (device float*)Yv + orow + nloc + c;
        *y = (EPI == EPI_ADD) ? *y + v : v;
      }
    }
  }
}

#define INSTANTIATE_MOE_VQ(MM)                                                                         \
  template [[host_name("moe_vq_gemv" #MM "_store")]] kernel void moe_vq_gemv<MM, EPI_STORE>(            \
      device const half*, device const uchar*, device const uint*, device const half*, device void*,     \
      constant MoeVqParams&, device const int4*, device const uint*, device const int*, device const half4*, \
      uint3, uint, uint);                                                                                \
  template [[host_name("moe_vq_gemv" #MM "_swiglu")]] kernel void moe_vq_gemv<MM, EPI_SWIGLU>(          \
      device const half*, device const uchar*, device const uint*, device const half*, device void*,     \
      constant MoeVqParams&, device const int4*, device const uint*, device const int*, device const half4*, \
      uint3, uint, uint);

INSTANTIATE_MOE_VQ(1)
INSTANTIATE_MOE_VQ(2)
INSTANTIATE_MOE_VQ(4)
INSTANTIATE_MOE_VQ(8)

// Dequantize rows [r0, r0 + n) of an EQV stack to fp16 (checks, MTP / dense fallbacks):
// one thread per vector.
kernel void vq_dequant(
    device const uchar* lo [[buffer(0)]],
    device const uint* hi [[buffer(1)]],
    device const half* sc [[buffer(2)]],
    device const half4* cb [[buffer(3)]],
    device half4* out [[buffer(4)]],
    constant MoeVqParams& p [[buffer(5)]],
    constant int& r0 [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int v = int(gid.x), rl = int(gid.y);
  if (v >= p.nv) return;
  const ulong r = ulong(r0 + rl);
  const uint code = vq_code(lo, hi, r, v, p);
  const uint cbi = p.cbsel ? uint((int(r) / 8) & 1) : 0u;
  device const half4* c = cb + ((ulong(cbi) << p.kbits) + ulong(code)) * 2;
  const half s = sc[r * ulong(p.ns) + ulong(v / 32)];
  out[(ulong(rl) * ulong(p.nv) + ulong(v)) * 2] = c[0] * s;
  out[(ulong(rl) * ulong(p.nv) + ulong(v)) * 2 + 1] = c[1] * s;
}
