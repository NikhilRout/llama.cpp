// Shared by device kernels and host references: format traits and the
// bit-exact fp32 -> {tf32, fp16, bf16, e4m3, e2m1, mx/nv/lns scale} quantizers, so the
// host can emulate exactly what the device produced.
#pragma once

#include <stdint.h>
#include <tensor_cfg.h>

#ifdef E2E_FPU
#define E2E_TCU 0
#else
#define E2E_TCU 1
#ifndef ITYPE
#error "TCU builds need -DITYPE=<tf32|fp16|bf16|fp8|mxfp8|mxfp4|nvfp4|lnsf4>"
#endif
#endif

#if defined(__VORTEX__)
#define E2E_INL static inline __attribute__((always_inline))
#define E2E_INL_M inline __attribute__((always_inline))
#else
#define E2E_INL static inline
#define E2E_INL_M inline
#endif

namespace e2e {

namespace vt = vortex::tensor;

enum scale_kind_t : uint32_t {
  SCALE_NONE   = 0,  // tf32, fp16
  SCALE_TENSOR = 1,  // fp8 e4m3 with a per-tensor fp32 scale
  SCALE_MX     = 2,  // OCP MX: e8m0 scale per 32 K-elements
  SCALE_NV     = 3,  // NVFP4: ue4m3 scale per 16 K-elements + per-tensor fp32 scale
  SCALE_LNS    = 4,  // LNSF4: e2m1 with an LNS8 (Q4.3 log2) scale per 16 K-elements + per-tensor fp32 scale
};

template <typename F> struct fmt_t;

template <> struct fmt_t<vt::fp32> {   // FPU baseline
  static constexpr uint32_t bits = 32, epw = 1, blk = 0;
  static constexpr scale_kind_t scale = SCALE_NONE;
  static constexpr const char* name = "fpu_fp32";
};
template <> struct fmt_t<vt::tf32> {
  static constexpr uint32_t bits = 32, epw = 1, blk = 0;
  static constexpr scale_kind_t scale = SCALE_NONE;
  static constexpr const char* name = "tf32";
};
template <> struct fmt_t<vt::fp16> {
  static constexpr uint32_t bits = 16, epw = 2, blk = 0;
  static constexpr scale_kind_t scale = SCALE_NONE;
  static constexpr const char* name = "fp16";
};
template <> struct fmt_t<vt::bf16> {
  static constexpr uint32_t bits = 16, epw = 2, blk = 0;
  static constexpr scale_kind_t scale = SCALE_NONE;
  static constexpr const char* name = "bf16";
};
template <> struct fmt_t<vt::fp8> {
  static constexpr uint32_t bits = 8, epw = 4, blk = 0;
  static constexpr scale_kind_t scale = SCALE_TENSOR;
  static constexpr const char* name = "fp8";
};
template <> struct fmt_t<vt::mxfp8> {
  static constexpr uint32_t bits = 8, epw = 4, blk = 32;
  static constexpr scale_kind_t scale = SCALE_MX;
  static constexpr const char* name = "mxfp8";
};
template <> struct fmt_t<vt::mxfp4> {
  static constexpr uint32_t bits = 4, epw = 8, blk = 32;
  static constexpr scale_kind_t scale = SCALE_MX;
  static constexpr const char* name = "mxfp4";
};
template <> struct fmt_t<vt::nvfp4> {
  static constexpr uint32_t bits = 4, epw = 8, blk = 16;
  static constexpr scale_kind_t scale = SCALE_NV;
  static constexpr const char* name = "nvfp4";
};
#ifdef VX_CFG_TCU_LNSF4_ENABLE
template <> struct fmt_t<vt::lnsf4> {
  static constexpr uint32_t bits = 4, epw = 8, blk = 16;
  static constexpr scale_kind_t scale = SCALE_LNS;
  static constexpr const char* name = "lnsf4";
};
#endif

#if E2E_TCU
using IT = vt::ITYPE;
#else
using IT = vt::fp32;
#endif
using F = fmt_t<IT>;

// WMMA tile geometry follows the warp size (tensor_cfg.h): a tile holds
// NT * 8 words, M = N = TILE and the K-tile is KT_WORDS 32-bit words per row
// (NT=32: 16x16, 16 words; NT=8: 8x8, 8 words).
constexpr uint32_t clog2_(uint32_t x) { return x < 2 ? 0 : 1 + clog2_(x / 2); }
constexpr uint32_t WARP_NT = VX_CFG_NUM_THREADS;
constexpr uint32_t TILE = 1u << (clog2_(WARP_NT * 8) - clog2_(WARP_NT * 8) / 2);
constexpr uint32_t KT_WORDS = WARP_NT * 8 / TILE;
// One CTA per core: every warp, every thread.
constexpr uint32_t CTA_NW = VX_CFG_NUM_WARPS;
constexpr uint32_t CTA_THREADS = WARP_NT * CTA_NW;
// SIMT GEMM thread grid (GY rows x GX columns): 32x32 at 1024 threads.
constexpr uint32_t SIMT_GX = 1u << (clog2_(CTA_THREADS) / 2);
constexpr uint32_t SIMT_GY = CTA_THREADS / SIMT_GX;
constexpr uint32_t KT = KT_WORDS * F::epw;       // logical K per WMMA tile
constexpr uint32_t ROW_BYTES(uint32_t k) { return k * F::bits / 8; }
constexpr bool IS_MX = (F::scale == SCALE_MX || F::scale == SCALE_NV || F::scale == SCALE_LNS);
constexpr uint32_t BLK = F::blk ? F::blk : 1;   // scale block (1 when unscaled)
// Scale code meaning 1.0 (fill value for padded blocks).
constexpr uint8_t SC_ONE = (F::scale == SCALE_NV) ? 0x38 : (F::scale == SCALE_LNS) ? 0 : 127;
constexpr bool HAS_TSCALE = (F::scale == SCALE_TENSOR || F::scale == SCALE_NV || F::scale == SCALE_LNS);
// Scale bytes per row inside one K-tile, and per TILE-row tile (TCU_LD loads
// one NT-word slot regardless; the tail is zero).
constexpr uint32_t SC_PER_ROW = IS_MX ? KT / F::blk : 0;
constexpr uint32_t SC_TILE_WORDS = WARP_NT;
constexpr uint32_t SC_TILE_BYTES = SC_TILE_WORDS * 4;

// Staging layouts shared by host DXA descriptors and device mainloops.
constexpr uint32_t LDW_TCU = KT_WORDS + 4;   // padded smem row (words) for WMMA tiles
constexpr uint32_t SLDW_SIMT = 17;   // padded smem row (words) for the SIMT GEMM
constexpr uint32_t SBK_SIMT = KT_WORDS;   // K words per SIMT stage (= blocked operand K-tile)
// Two-or-more staging buffers plus a 128-word scratch line (see tcu_gemm).
constexpr uint32_t tcu_gemm_smem(uint32_t bm, uint32_t bn, uint32_t stages) {
  return stages * (bm + bn) * LDW_TCU * 4 + 128 * 4;
}
// Staging smem plus room to park the fp32 C tile (pitch BN+1) for epilogues.
constexpr uint32_t tcu_gemm_lmem(uint32_t bm, uint32_t bn) {
  return tcu_gemm_smem(bm, bn, 2) > bm * (bn + 1) * 4 ? tcu_gemm_smem(bm, bn, 2) : bm * (bn + 1) * 4;
}
constexpr uint32_t simt_gemm_smem(uint32_t bm, uint32_t bn, uint32_t stages) {
  return stages * (bm + bn) * SLDW_SIMT * 4;
}

// GEMM operands are stored CTA-blocked so every K-tile stage of a CTA is one
// contiguous DRAM burst: [rows/RB][num_kt][RB rows][16 words]. Scales use the
// TCU tile layout grouped the same way: [rows/RB][num_kt][RB/16][128 B].
E2E_INL uint32_t blk_word(uint32_t row, uint32_t kw, uint32_t rb, uint32_t num_kt) {
  return ((row / rb) * num_kt + kw / KT_WORDS) * (rb * KT_WORDS) + (row % rb) * KT_WORDS + kw % KT_WORDS;
}
E2E_INL uint32_t blk_sc(uint32_t row, uint32_t k, uint32_t rb, uint32_t num_kt) {
  return ((row / rb) * num_kt + k / KT) * (rb / TILE * SC_TILE_BYTES)
       + ((row % rb) / TILE) * SC_TILE_BYTES + (row % TILE) * SC_PER_ROW + (k % KT) / (F::blk ? F::blk : 1);
}

E2E_INL uint32_t f2u(float f) { union { float f; uint32_t u; } c; c.f = f; return c.u; }
E2E_INL float u2f(uint32_t u) { union { float f; uint32_t u; } c; c.u = u; return c.f; }
E2E_INL float fabs_(float x) { return u2f(f2u(x) & 0x7fffffffu); }
E2E_INL float fmax_(float a, float b) { return a > b ? a : b; }

// ---- fp32 -> fp16 (RNE, saturating to inf like IEEE) ----
E2E_INL uint32_t f32_to_f16(float f) {
  uint32_t x = f2u(f);
  uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t ax = x & 0x7fffffffu;
  if (ax >= 0x47800000u)
    return sign | (ax > 0x7f800000u ? 0x7e00u : 0x7c00u);
  if (ax < 0x38800000u) {
    float t = u2f(ax) + 0.5f;            // ulp(0.5) == fp16 subnormal unit
    return sign | (f2u(t) - 0x3f000000u);
  }
  ax += 0xc8000fffu + ((ax >> 13) & 1u);
  return sign | (ax >> 13);
}

// ---- fp32 -> bf16 (RNE, quiet NaN) ----
E2E_INL uint32_t f32_to_bf16(float f) {
  uint32_t u = f2u(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40u;
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

// ---- fp32 -> OCP e4m3fn (RNE, satfinite at 448) ----
E2E_INL uint32_t f32_to_e4m3(float f) {
  uint32_t x = f2u(f);
  uint32_t sign = (x >> 24) & 0x80u;
  uint32_t ax = x & 0x7fffffffu;
  if (ax >= 0x43e00000u) return sign | 0x7eu;
  if (ax < 0x3c800000u) {
    float t = u2f(ax) + 16384.0f;        // ulp(2^14) == e4m3 subnormal unit 2^-9
    return sign | (f2u(t) - 0x46800000u);
  }
  ax += 0xc407ffffu + ((ax >> 20) & 1u);  // (7-127)<<23 + 0x7ffff
  return sign | (ax >> 20);
}

E2E_INL float e4m3_to_f32(uint32_t v) {
  uint32_t s = (v & 0x80u) << 24, e = (v >> 3) & 0xfu, m = v & 7u;
  if (e == 0) return u2f(s | f2u((float)m * 0.001953125f));
  return u2f(s | ((e + 120u) << 23) | (m << 20));
}

// ---- fp32 -> e2m1 (RNE, satfinite at 6) ----
E2E_INL uint32_t f32_to_e2m1(float f) {
  float a = fabs_(f);
  uint32_t c = (uint32_t)(a > 0.25f) + (uint32_t)(a >= 0.75f) + (uint32_t)(a > 1.25f)
             + (uint32_t)(a >= 1.75f) + (uint32_t)(a > 2.5f) + (uint32_t)(a >= 3.5f)
             + (uint32_t)(a > 5.0f);
  return c | ((f2u(f) >> 28) & 0x8u & (0u - (c != 0)));
}

E2E_INL float e2m1_to_f32(uint32_t v) {
  static const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float r = lut[v & 7u];
  return (v & 8u) ? -r : r;
}

// ---- MX e8m0 shared scale: 2^ceil(log2(amax / elem_max)), sf in [1,253] ----
E2E_INL uint32_t e8m0_select(float amax, float inv_elem_max) {
  uint32_t b = f2u(amax * inv_elem_max);
  int32_t e = (int32_t)((b >> 23) & 0xffu) + (int32_t)((b & 0x7fffffu) != 0);
  if (amax == 0.0f) e = 127;
  e = e < 1 ? 1 : (e > 253 ? 253 : e);
  return (uint32_t)e;
}
// Multiplier 2^-(sf-127) applied before element conversion.
E2E_INL float e8m0_inv(uint32_t sf) { return u2f((254u - sf) << 23); }
E2E_INL float e8m0_val(uint32_t sf) { return u2f(sf << 23); }

// ---- NVFP4 ue4m3 block scale: e4m3(amax / (6 * tensor_scale)) ----
E2E_INL uint32_t nv_select(float amax, float inv_6ts) {
  uint32_t sf = f32_to_e4m3(amax * inv_6ts) & 0x7fu;
  return sf == 0 ? 0x38u : sf;   // all-zero block: any nonzero scale works
}

// ---- LNSF4 block scale: raw = round(8 * log2(amax / (6 * tensor_scale))) in
// [-64, 63], stored as 7-bit two's complement (Q4.3 log2). The log2 is taken
// from the float's exponent plus a threshold search over its mantissa, so
// device and host agree bit for bit without libm.
E2E_INL uint32_t lns_select(float amax, float inv_6ts) {
  const float v = amax * inv_6ts;
  if (!(v > 0.0f)) return 0;
  const uint32_t b = f2u(v);
  const float m = u2f((b & 0x7fffffu) | 0x3f800000u);
  // 2^((k + 0.5) / 8): the mantissa bounds where 8*log2(m) rounds up to k+1.
  const float th[8] = {1.04427378f, 1.13878863f, 1.24185781f, 1.35425555f,
                       1.47682615f, 1.61049033f, 1.75625216f, 1.91520656f};
  int32_t frac = 0;
  for (uint32_t k = 0; k < 8; ++k) frac += (m >= th[k]) ? 1 : 0;
  int32_t raw = ((int32_t)((b >> 23) & 0xffu) - 127) * 8 + frac;
  raw = raw < -64 ? -64 : (raw > 63 ? 63 : raw);
  return (uint32_t)raw & 0x7fu;
}
// 2^(raw/8) for a 7-bit two's-complement LNS8 scale code.
E2E_INL float lns_val(uint32_t sf) {
  const int32_t raw = (int32_t)(sf << 25) >> 25;
  const float pk[8] = {1.0f, 1.09050773f, 1.18920712f, 1.29683955f,
                       1.41421356f, 1.54221083f, 1.68179283f, 1.83400809f};
  return pk[raw & 7] * u2f((uint32_t)((raw >> 3) + 127) << 23);
}

// Per-format element maxima used when choosing scales.
constexpr float E4M3_MAX = 448.0f;
constexpr float E2M1_MAX = 6.0f;

// Encode one element given the per-block multiplier (MX/NV) or the per-tensor
// inverse scale (fp8). Returns the raw element code (unshifted).
E2E_INL uint32_t encode(float x, float mult) {
  if constexpr (std::is_same<IT, vt::tf32>::value) {
    // TCU tf32 operand: 19-bit {sign, e8, m10} in the low bits (RNE).
    (void)mult;
    uint32_t u = f2u(x);
    return (u + 0xfffu + ((u >> 13) & 1u)) >> 13;
  } else if constexpr (F::bits == 32) {
    (void)mult;
    return f2u(x);
  } else if constexpr (std::is_same<IT, vt::bf16>::value) {
    (void)mult;
    return f32_to_bf16(x);
  } else if constexpr (F::bits == 16) {
    (void)mult;
    return f32_to_f16(x);
  } else if constexpr (F::bits == 8) {
    return f32_to_e4m3(x * mult);
  } else {
    return f32_to_e2m1(x * mult);
  }
}

// Decode a raw element code with its scale for host references.
E2E_INL float decode(uint32_t code, uint32_t sf, float tscale) {
  if constexpr (std::is_same<IT, vt::tf32>::value) {
    (void)sf; (void)tscale;
    return u2f(code << 13);
  } else if constexpr (F::bits == 32) {
    (void)sf; (void)tscale;
    return u2f(code);
  } else if constexpr (std::is_same<IT, vt::bf16>::value) {
    (void)sf; (void)tscale;
    return u2f(code << 16);
  } else if constexpr (F::bits == 16) {
    (void)sf; (void)tscale;
    uint32_t s = (code & 0x8000u) << 16, e = (code >> 10) & 0x1fu, m = code & 0x3ffu;
    if (e == 0) return u2f(s | f2u((float)m * 5.9604644775390625e-8f));
    if (e == 31) return u2f(s | 0x7f800000u | (m << 13));
    return u2f(s | ((e + 112u) << 23) | (m << 13));
  } else if constexpr (F::scale == SCALE_TENSOR) {
    (void)sf;
    return e4m3_to_f32(code) * tscale;
  } else if constexpr (F::scale == SCALE_MX) {
    (void)tscale;
    float v = (F::bits == 8) ? e4m3_to_f32(code) : e2m1_to_f32(code);
    return v * e8m0_val(sf);
  } else if constexpr (F::scale == SCALE_LNS) {
    return e2m1_to_f32(code) * lns_val(sf) * tscale;
  } else {
    return e2m1_to_f32(code) * e4m3_to_f32(sf) * tscale;
  }
}

// Quantize one K-run of `n` fp32 values (n a multiple of the scale block, or
// of epw for unscaled formats) into packed words + per-block scale codes.
// `tscale` is the per-tensor scale (fp8 / nvfp4); ignored otherwise.
// Returns words written; scales go to sc_out[i] for block i.
E2E_INL uint32_t quantize_run(const float* x, uint32_t n, float tscale,
                              uint32_t* words, uint8_t* sc_out) {
  uint32_t nw = 0;
  if constexpr (IS_MX) {
    for (uint32_t b = 0; b < n / F::blk; ++b) {
      const float* xb = x + b * F::blk;
      float amax = 0.0f;
      for (uint32_t i = 0; i < F::blk; ++i) amax = fmax_(amax, fabs_(xb[i]));
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
      sc_out[b] = (uint8_t)sf;
      for (uint32_t w = 0; w < F::blk / F::epw; ++w) {
        uint32_t word = 0;
        for (uint32_t e = 0; e < F::epw; ++e)
          word |= encode(xb[w * F::epw + e], mult) << (e * F::bits);
        words[nw++] = word;
      }
    }
  } else {
    float mult = (F::scale == SCALE_TENSOR) ? 1.0f / tscale : 1.0f;
    (void)sc_out;
    for (uint32_t w = 0; w < n / F::epw; ++w) {
      uint32_t word = 0;
      for (uint32_t e = 0; e < F::epw; ++e)
        word |= encode(x[w * F::epw + e], mult) << (e * F::bits);
      words[nw++] = word;
    }
  }
  return nw;
}

// Per-tensor scale for fp8 / nvfp4 from a calibrated absolute maximum.
E2E_INL float tensor_scale_for(float amax) {
  if (!(amax > 0.0f)) return 1.0f;
  if constexpr (F::scale == SCALE_TENSOR) return amax / E4M3_MAX;
  if constexpr (F::scale == SCALE_NV) return amax / (E4M3_MAX * E2M1_MAX);
  // Anchors the tensor's largest block at the top LNS8 scale, 2^7.875.
  if constexpr (F::scale == SCALE_LNS) return amax / (E2M1_MAX * 234.753035f);
  return 1.0f;
}

// exp2 with round-to-nearest range reduction and a degree-5 polynomial on
// [-0.5, 0.5]; shared so host references reproduce device softmax exactly.
E2E_INL float exp2_poly(float f) {
  return 1.0f + f * (0.693147182f + f * (0.240226507f + f * (0.0555041087f
       + f * (0.00961812911f + f * 0.00133335581f))));
}

// sin/cos in float with 2*pi range reduction (|x| < 2^20), shared by host
// references and kernels (libm's single-precision versions differ between the
// two and newlib's go through soft-float double on RV32F).
E2E_INL float sin_cos_reduce(float x) {
  const float k = (x * 0.159154943f + 12582912.0f) - 12582912.0f;   // round(x / 2pi)
  return (x - k * 6.28318548f) + k * 1.74845553e-7f;
}
E2E_INL float sin_poly(float x) {
  float r = sin_cos_reduce(x);
  if (r > 1.57079637f) r = 3.14159274f - r;
  if (r < -1.57079637f) r = -3.14159274f - r;
  const float r2 = r * r;
  return r * (1.0f + r2 * (-0.166666672f + r2 * (8.33333377e-3f + r2 * (-1.98412701e-4f
       + r2 * (2.75573188e-6f - r2 * 2.50521080e-8f)))));
}
E2E_INL float cos_poly(float x) { return sin_poly(x + 1.57079637f); }

// 2^x with round-to-nearest range reduction (magic-number rounding, so host
// and device agree bit for bit), clamped to [2^-126, 2^126].
E2E_INL float exp2_shared(float x) {
  x = x < -126.0f ? -126.0f : (x > 126.0f ? 126.0f : x);
  const float k = (x + 12582912.0f) - 12582912.0f;
  return u2f(f2u(exp2_poly(x - k)) + ((uint32_t)(int32_t)k << 23));
}

} // namespace e2e
