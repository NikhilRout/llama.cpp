// Device-side building blocks shared by the e2e application kernels:
// DXA-staged CTA GEMM mainloops (TCU and SIMT/FPU), quantizing stores and
// small math helpers.
#pragma once

#include "e2e_fmt.h"
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_dxa.h>
#if E2E_TCU
#include <vx_tensor.h>
#endif

namespace e2e {

constexpr uint32_t NT = VX_CFG_NUM_THREADS;

// ---------------------------------------------------------------------------
// math
// ---------------------------------------------------------------------------
E2E_INL int32_t fcvt_rne(float x) {
  int32_t r;
  __asm__ volatile("fcvt.w.s %0, %1, rne" : "=r"(r) : "f"(x));
  return r;
}

// 2^x, clamped to [2^-126, 2^126].
E2E_INL float exp2_dev(float x) {
  x = x < -126.0f ? -126.0f : (x > 126.0f ? 126.0f : x);
  int32_t i = fcvt_rne(x);
  float f = x - (float)i;
  return u2f(f2u(exp2_poly(f)) + ((uint32_t)i << 23));
}

// Single fmadd.s; __builtin_fmaf lowers to a libm call on this target.
E2E_INL float fma_dev(float a, float b, float c) {
  float d;
  __asm__("fmadd.s %0, %1, %2, %3" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}

E2E_INL float shfl_bfly(float v, int b) {
  return u2f((uint32_t)vx_shfl_bfly(f2u(v), b, NT - 1, 0));
}
E2E_INL float warp_max(float v) {
  for (int b = NT / 2; b > 0; b >>= 1) v = fmax_(v, shfl_bfly(v, b));
  return v;
}
E2E_INL float warp_sum(float v) {
  for (int b = NT / 2; b > 0; b >>= 1) v += shfl_bfly(v, b);
  return v;
}
// Quantizes a run of QRUN values into operand words. MX/NV: a run is one
// word; the WPB words of a scale block must sit on adjacent, WPB-aligned
// lanes (callers map consecutive runs of a row to consecutive threads), so
// the block amax is a lane butterfly and every warp load stays coalesced.
// The block's first lane writes the scale byte at sc_dst.
E2E_INL void quant_store_run(const float* x, uint32_t n, float tscale,
                             uint32_t* dst_words, uint8_t* sc_dst) {
  if constexpr (IS_MX) {
    (void)n;
    constexpr uint32_t WPB = F::blk / F::epw;
    float v[F::epw];
    float amax = 0.0f;
    for (uint32_t e = 0; e < F::epw; ++e) { v[e] = x[e]; amax = fmax_(amax, fabs_(v[e])); }
    for (uint32_t b = WPB / 2; b > 0; b >>= 1)
      amax = fmax_(amax, shfl_bfly(amax, b));
    float mult;
    uint32_t sf;
    if constexpr (F::scale == SCALE_MX) {
      sf = e8m0_select(amax, 1.0f / (F::bits == 8 ? E4M3_MAX : E2M1_MAX));
      mult = e8m0_inv(sf);
    } else if constexpr (F::scale == SCALE_LNS) {
      sf = lns_select(amax, 1.0f / (E2M1_MAX * tscale));
      mult = 1.0f / (lns_val(sf) * tscale);
    } else {
      sf = nv_select(amax, 1.0f / (E2M1_MAX * tscale));
      mult = 1.0f / (e4m3_to_f32(sf) * tscale);
    }
    if (threadIdx.x % WPB == 0) *sc_dst = (uint8_t)sf;
    uint32_t word = 0;
    for (uint32_t e = 0; e < F::epw; ++e) word |= encode(v[e], mult) << (e * F::bits);
    *dst_words = word;
  } else {
    (void)sc_dst;
    float mult = (F::scale == SCALE_TENSOR) ? 1.0f / tscale : 1.0f;
    for (uint32_t w = 0; w < n / F::epw; ++w) {
      uint32_t word = 0;
      for (uint32_t e = 0; e < F::epw; ++e)
        word |= encode(x[w * F::epw + e], mult) << (e * F::bits);
      dst_words[w] = word;
    }
  }
}

// Elements per quantization run owned by one thread when writing operands.
constexpr uint32_t QRUN = IS_MX ? F::epw : (F::epw > 4 ? F::epw : 4);

// Pair of CTA barriers selected by a runtime parity without materializing
// barrier objects (an indexed array of them lands on the stack, i.e. in
// global memory, on every K-tile's critical path).
struct stage_bars {
  uint32_t id0, nw;
  E2E_INL stage_bars make(uint32_t bar0) {
    return {(bar0 << 8) + get_local_group_id(), get_num_sub_groups()};
  }
  E2E_INL_M uint32_t id(uint32_t s) const { return id0 + (s << 8); }
  E2E_INL_M void sync(uint32_t s) const { vx_barrier(id(s), nw); }
  E2E_INL_M void expect(uint32_t s, uint32_t n) const { vx_barrier_expect_tx(id(s), n); }
};

// ---------------------------------------------------------------------------
// GEMM operand in the CTA-blocked layout, blocked by the CTA tile rows of the
// side it feeds (BM for A, BN for B). `dslot`/`sslot` are the DXA descriptor
// slots used when the operand is staged by DXA.
// ---------------------------------------------------------------------------
struct operand_t {
  const uint32_t* data;     // CTA-blocked words (see blk_word)
  const uint32_t* scales;   // CTA-blocked scale tiles (see blk_sc)
  uint32_t row0;            // first row of this CTA's tile (multiple of the block)
  uint32_t dslot, sslot;
};

// Cooperative LSU copy of a contiguous [ROWS x COLS] word block into padded
// smem rows, split in two halves (global->regs, regs->smem) so the loads of
// the next stage overlap the compute of the current one.
// Every address is derived from one base pointer laundered through an empty
// asm, so the compiler cannot hoist PER separate loop-invariant addresses out
// of the K-loop (it otherwise spills them to the stack, i.e. to DRAM).
template <uint32_t ROWS, uint32_t COLS, uint32_t PITCH, uint32_t THREADS>
struct tile_copy {
  static constexpr uint32_t WORDS = ROWS * COLS;
  static constexpr uint32_t PER = (WORDS + THREADS - 1) / THREADS;
  static_assert(THREADS % COLS == 0, "a thread keeps its column across iterations");
  E2E_INL void load(uint32_t (&r)[PER], const uint32_t* g, uint32_t tid) {
    const uint32_t* p = g + tid;
    __asm__ volatile("" : "+r"(p));
#pragma unroll
    for (uint32_t i = 0; i < PER; ++i) {
      if (WORDS % THREADS == 0 || tid + i * THREADS < WORDS)
        r[i] = p[i * THREADS];
    }
  }
  E2E_INL void store(const uint32_t (&r)[PER], uint32_t* s, uint32_t tid) {
    uint32_t* p = s + (tid / COLS) * PITCH + tid % COLS;
    __asm__ volatile("" : "+r"(p));
#pragma unroll
    for (uint32_t i = 0; i < PER; ++i) {
      if (WORDS % THREADS == 0 || tid + i * THREADS < WORDS)
        p[i * (THREADS / COLS) * PITCH] = r[i];
    }
  }
};

#if E2E_TCU
// ---------------------------------------------------------------------------
// TCU GEMM: C[BM x BN] += A[BM x K] . B[BN x K]^T over CTA-blocked operands.
// All 32 warps stage the next K-tile (a few words each, global->regs->smem)
// and each computes one 16x16 tile. Rows in smem are padded to 80 B, which
// makes the WMMA fragment loads bank-conflict free. Two smem stages, one CTA
// barrier per K-tile.
//
// Codegen constraints (each was measured to cost 2-10x in SimX):
//  - global and smem requests of a warp share one in-order LSU queue, so the
//    tile's fragment loads are fenced ahead of the next tile's global loads;
//  - any stack spill is a DRAM round trip on the critical path, so addresses
//    come from laundered base pointers and barrier ids are computed, never
//    indexed from an array;
//  - one register set for the prefetch (two sets spill).
// ---------------------------------------------------------------------------
using ctx = vt::wmma_context<NT, IT, vt::fp32>;
constexpr uint32_t LDW = LDW_TCU;
constexpr uint32_t LDM = LDW * (4 / sizeof(ctx::input_t));
constexpr uint32_t CTA_WARPS = CTA_NW;

// Staging assignment: the TCU's MX metadata loads (TCU_LD) are forwarded
// through issue lane 0's LSU queue, and a DRAM miss queued there holds every
// TCU block waiting on its metadata. So warps on lane 0 never touch global
// memory: the 24 warps on lanes 1..3 each stream one contiguous 128-word slice
// of the K-tile (A: 16 slices, B: 8 slices; MX scale tiles: 12 more 32-word
// slices), and lane-0 warps run the same code against a scratch line in smem.
template <uint32_t BM, uint32_t BN>
struct tcu_gemm {
  static constexpr uint32_t WM = BM / TILE, WN = BN / TILE;
  static_assert(WM * WN == CTA_WARPS, "one TILE x TILE tile per warp");
  static constexpr uint32_t A_WORDS = BM * LDW, B_WORDS = BN * LDW;
  static constexpr uint32_t STAGE_WORDS = A_WORDS + B_WORDS;
  static constexpr uint32_t SCRATCH = 2 * STAGE_WORDS;
  static constexpr uint32_t SMEM_BYTES = (2 * STAGE_WORDS + 128) * 4;
  static_assert(SMEM_BYTES == tcu_gemm_smem(BM, BN, 2), "host/device smem mismatch");
  static constexpr uint32_t ISSUE_W = VX_CFG_ISSUE_WIDTH;
  // With one issue lane there is no separate lane-0 LSU queue to protect, so
  // every warp fetches.
  static constexpr uint32_t FETCH_WARPS = (ISSUE_W == 1) ? CTA_WARPS : CTA_WARPS - CTA_WARPS / ISSUE_W;
  // A slice is the 4 * NT contiguous words one warp moves per K-tile (four
  // warp-wide loads), i.e. SLICE_ROWS rows of the blocked K-tile.
  static constexpr uint32_t SLICE_WORDS = 4 * NT;
  static constexpr uint32_t CHUNK_ROWS = NT / KT_WORDS;   // rows per warp-wide load
  static constexpr uint32_t SLICE_ROWS = 4 * CHUNK_ROWS;
  static_assert(NT % KT_WORDS == 0, "a warp-wide load covers whole K-tile rows");
  static_assert(4 * NT <= 128, "non-fetch warps stage against the 128-word scratch line");
  static constexpr uint32_t A_SLICES = BM * KT_WORDS / SLICE_WORDS, B_SLICES = BN * KT_WORDS / SLICE_WORDS;
  static constexpr uint32_t SLICES = A_SLICES + B_SLICES;
  static constexpr uint32_t NS = (SLICES + FETCH_WARPS - 1) / FETCH_WARPS;   // slices per fetch warp

  struct regs_t { uint32_t d[4 * NS]; };
  struct cursor_t {
    const uint32_t* src[NS];   // this thread's word of each slice (global, or scratch)
    uint32_t stride[NS];       // words per K-tile step
    uint32_t* dst0[NS];        // destination in stage 0 / stage 1 (or scratch)
    uint32_t* dst1[NS];
    E2E_INL_M void advance() {
#pragma unroll
      for (uint32_t j = 0; j < NS; ++j) src[j] += stride[j];
    }
  };

  static inline __attribute__((always_inline)) cursor_t make_cursor(uint32_t* smem, const operand_t& A,
                                                                    const operand_t& B, uint32_t num_kt, uint32_t wid) {
    const uint32_t lane = threadIdx.x % NT;
    const uint32_t ab = (A.row0 / BM) * num_kt, bb = (B.row0 / BN) * num_kt;
    cursor_t c;
    for (uint32_t j = 0; j < NS; ++j) {
      c.src[j] = smem + SCRATCH + lane; c.stride[j] = 0;
      c.dst0[j] = c.dst1[j] = smem + SCRATCH + lane;
    }
    if (ISSUE_W == 1 || wid % ISSUE_W != 0) {
      const uint32_t f = (ISSUE_W == 1) ? wid : (wid / ISSUE_W) * (ISSUE_W - 1) + wid % ISSUE_W - 1;
      for (uint32_t j = 0; j < NS; ++j) {
        const uint32_t sl = f + j * FETCH_WARPS;
        if (sl >= SLICES) continue;
        uint32_t off;
        if (sl < A_SLICES) {
          c.src[j] = A.data + ab * (BM * KT_WORDS) + sl * SLICE_WORDS + lane;
          c.stride[j] = BM * KT_WORDS;
          off = (sl * SLICE_ROWS + lane / KT_WORDS) * LDW + lane % KT_WORDS;
        } else {
          c.src[j] = B.data + bb * (BN * KT_WORDS) + (sl - A_SLICES) * SLICE_WORDS + lane;
          c.stride[j] = BN * KT_WORDS;
          off = A_WORDS + ((sl - A_SLICES) * SLICE_ROWS + lane / KT_WORDS) * LDW + lane % KT_WORDS;
        }
        c.dst0[j] = smem + off;
        c.dst1[j] = smem + STAGE_WORDS + off;
      }
    }
    return c;
  }

  static inline __attribute__((always_inline)) void fetch(regs_t& r, const cursor_t& c) {
#pragma unroll
    for (uint32_t j = 0; j < NS; ++j) {
      const uint32_t* p = c.src[j];
      __asm__ volatile("" : "+r"(p));
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) r.d[4 * j + i] = p[i * NT];
    }
  }
  static inline __attribute__((always_inline)) void commit(const regs_t& r, uint32_t* const (&dst)[NS]) {
#pragma unroll
    for (uint32_t j = 0; j < NS; ++j) {
      uint32_t* p = dst[j];
      __asm__ volatile("" : "+r"(p));
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) p[i * CHUNK_ROWS * LDW] = r.d[4 * j + i];
    }
  }

  // MX: TCU_LD reads the warp's A/B scale tiles straight from the CTA-blocked
  // scale tensors in global memory (L1-resident: 4-8 warps share each tile).
  // Staging them through smem as well needs extra cursor registers that push
  // this loop into stack spills. The last step prefetches one tile past the
  // end (operand buffers carry slack).
  static inline __attribute__((always_inline)) void step(ctx::fragment_acc& acc, uint32_t* st, uint32_t* const (&dst)[NS],
      cursor_t& cur, regs_t& r, bool more, uint32_t wm, uint32_t wn, const stage_bars& bars,
      uint32_t nxt, const uint32_t*& gsa, const uint32_t*& gsb) {
    ctx::fragment_a fa;
    ctx::fragment_b fb;
    ctx::load_matrix_sync(fa, st + wm * TILE * LDW, LDM);
    ctx::load_matrix_sync<vt::col_major>(fb, st + A_WORDS + wn * TILE * LDW, LDM);
    __asm__ volatile("" ::: "memory");
#ifdef E2E_GEMM_NOFETCH
    more = false;  // profiling only: MMA from resident smem tiles
#endif
    if (more) { cur.advance(); fetch(r, cur); }
    ctx::mma_sync(acc, fa, fb, acc);
    // Next K-tile's scales: the TCU runs this warp's uops in order, so the
    // load lands after this MMA consumed the current ones and overlaps the
    // barrier wait instead of stalling the next MMA.
    if constexpr (IS_MX) {
      gsa += WM * SC_TILE_WORDS;
      gsb += WN * SC_TILE_WORDS;
      ctx::load_mx_metadata(fa, gsa);
      ctx::load_mx_metadata(fb, gsb);
    } else {
      (void)gsa; (void)gsb;
    }
    if (more) commit(r, dst);
    bars.sync(nxt);
  }

  // Leaves the fp32 C tile in smem at `cs` (pitch cp) and returns synchronized.
  // TRANSPOSE_C parks C^T instead (row = B row, col = A row).
  template <bool TRANSPOSE_C = false>
  static inline __attribute__((always_inline)) void run(uint32_t* smem, const operand_t& A, const operand_t& B,
                                                        uint32_t num_kt, float* cs, uint32_t cp, uint32_t bar0 = 0) {
    const uint32_t wid = get_sub_group_id();
    const uint32_t wm = wid / WN, wn = wid % WN;
    const stage_bars bars = stage_bars::make(bar0);
    cursor_t cur = make_cursor(smem, A, B, num_kt, wid);
    ctx::fragment_acc acc;
    ctx::fill_fragment(acc, 0);
    regs_t r;
    const uint32_t* gsa = IS_MX ? A.scales + (A.row0 / BM) * num_kt * (WM * SC_TILE_WORDS) + wm * SC_TILE_WORDS : nullptr;
    const uint32_t* gsb = IS_MX ? B.scales + (B.row0 / BN) * num_kt * (WN * SC_TILE_WORDS) + wn * SC_TILE_WORDS : nullptr;
    fetch(r, cur);
    if constexpr (IS_MX) {   // scales of K-tile 0; step() prefetches the rest
      ctx::fragment_a fa0;
      ctx::fragment_b fb0;
      ctx::load_mx_metadata(fa0, gsa);
      ctx::load_mx_metadata(fb0, gsb);
    }
    commit(r, cur.dst0);
    bars.sync(0);
    for (uint32_t kt = 0; kt < num_kt; kt += 2) {
      step(acc, smem, cur.dst1, cur, r, kt + 1 < num_kt, wm, wn, bars, 1, gsa, gsb);
      if (kt + 1 < num_kt)
        step(acc, smem + STAGE_WORDS, cur.dst0, cur, r, kt + 2 < num_kt, wm, wn, bars, 0, gsa, gsb);
    }
    if constexpr (TRANSPOSE_C)
      ctx::store_matrix_sync<vt::col_major>(cs + wn * TILE * cp + wm * TILE, acc, cp);
    else
      ctx::store_matrix_sync(cs + wm * TILE * cp + wn * TILE, acc, cp);
    bars.sync(0);
  }
};

// Accumulator fragment element (r) -> (row, col) inside the warp's 16x16 tile.
E2E_INL uint32_t acc_row(uint32_t lane, uint32_t r) { return lane / 4 + 8 * (r / 4); }
E2E_INL uint32_t acc_col(uint32_t lane, uint32_t r) { return lane % 4 + 4 * (r % 4); }
#endif

// ---------------------------------------------------------------------------
// SIMT GEMM (FPU baseline): fp32 C[BM x BN] += A[BM x K] . B[BN x K]^T over a
// GY x GX thread grid (rows ty+GY*i, cols tx+GX*j). Staged
// like tcu_gemm (all threads copy a few words of the next K-tile), but
// transposed into K-major smem so a thread's operands are small immediate
// offsets from one pointer and each warp load hits 32 consecutive banks.
// ---------------------------------------------------------------------------
constexpr uint32_t SBK = SBK_SIMT;

// K-major staging copy of a [ROWS x 16] word block (pitch ROWS + 1).
template <uint32_t ROWS, uint32_t THREADS>
struct tile_copy_t {
  static constexpr uint32_t WORDS = ROWS * SBK;
  static constexpr uint32_t PITCH = ROWS + 1;
  static constexpr uint32_t PER = (WORDS + THREADS - 1) / THREADS;
  static_assert(THREADS % SBK == 0, "a thread keeps its K column");
  E2E_INL void load(uint32_t (&r)[PER], const uint32_t* g, uint32_t tid) {
    const uint32_t* p = g + tid;
    __asm__ volatile("" : "+r"(p));
#pragma unroll
    for (uint32_t i = 0; i < PER; ++i)
      if (WORDS % THREADS == 0 || tid + i * THREADS < WORDS) r[i] = p[i * THREADS];
  }
  E2E_INL void store(const uint32_t (&r)[PER], uint32_t* s, uint32_t tid) {
    uint32_t* p = s + (tid % SBK) * PITCH + tid / SBK;
    __asm__ volatile("" : "+r"(p));
#pragma unroll
    for (uint32_t i = 0; i < PER; ++i)
      if (WORDS % THREADS == 0 || tid + i * THREADS < WORDS) p[i * (THREADS / SBK)] = r[i];
  }
};

// acc[i][j] covers row ty()+GY*i (i < BM/GY), col tx()+GX*j (j < BN/GX).
template <uint32_t BM, uint32_t BN>
struct simt_gemm {
  static constexpr uint32_t GY = SIMT_GY, GX = SIMT_GX;
  static_assert(BM % GY == 0 && BN % GX == 0, "GY x GX thread grid");
  static constexpr uint32_t TM = BM / GY, TN = BN / GX;
  static constexpr uint32_t THREADS = CTA_THREADS;
  using copy_a = tile_copy_t<BM, THREADS>;
  using copy_b = tile_copy_t<BN, THREADS>;
  static constexpr uint32_t A_WORDS = SBK * copy_a::PITCH, B_WORDS = SBK * copy_b::PITCH;
  static constexpr uint32_t STAGE_WORDS = A_WORDS + B_WORDS;
  static constexpr uint32_t SMEM_BYTES = STAGE_WORDS * 4;
  static_assert(SMEM_BYTES <= simt_gemm_smem(BM, BN, 2), "host reserves enough smem");
  struct regs_t { uint32_t a[copy_a::PER]; uint32_t b[copy_b::PER]; };

  E2E_INL uint32_t ty() { return threadIdx.x % GY; }
  E2E_INL uint32_t tx() { return (threadIdx.x % GY + threadIdx.x / GY) % GX; }

  // Classic single-stage loop (stage, sync, FMA over the tile, sync). The
  // pipelined variants spill address registers to the stack (DRAM) here.
  static inline __attribute__((always_inline)) void run(float (&acc)[TM][TN], float* smem, const operand_t& A,
                                                        const operand_t& B, uint32_t num_kt, uint32_t bar0 = 0) {
    const uint32_t tid = threadIdx.x, r0 = ty(), c0 = tx();
    const stage_bars bars = stage_bars::make(bar0);
    const uint32_t* pa = A.data + (A.row0 / BM) * num_kt * (BM * SBK);
    const uint32_t* pb = B.data + (B.row0 / BN) * num_kt * (BN * SBK);
    regs_t r;
    for (uint32_t kt = 0; kt < num_kt; ++kt) {
      copy_a::load(r.a, pa, tid);
      copy_b::load(r.b, pb, tid);
      copy_a::store(r.a, reinterpret_cast<uint32_t*>(smem), tid);
      copy_b::store(r.b, reinterpret_cast<uint32_t*>(smem + A_WORDS), tid);
      pa += BM * SBK;
      pb += BN * SBK;
      bars.sync(0);
      const float* As = smem + r0;
      const float* Bs = smem + A_WORDS + c0;
#pragma unroll 4
      for (uint32_t k = 0; k < SBK; ++k) {
        float a[TM], b[TN];
#pragma unroll
        for (uint32_t i = 0; i < TM; ++i) a[i] = As[i * GY];
#pragma unroll
        for (uint32_t j = 0; j < TN; ++j) b[j] = Bs[j * GX];
#pragma unroll
        for (uint32_t i = 0; i < TM; ++i)
#pragma unroll
          for (uint32_t j = 0; j < TN; ++j)
            acc[i][j] = fma_dev(a[i], b[j], acc[i][j]);
        As += copy_a::PITCH;
        Bs += copy_b::PITCH;
      }
      bars.sync(1);
    }
  }
};

} // namespace e2e
