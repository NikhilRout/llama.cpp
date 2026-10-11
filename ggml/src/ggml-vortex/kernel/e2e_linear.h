// Fused linear layer shared by the MLP-style applications:
//   Y[M x N] = act(scale * X[M x K] . W[N x K]^T + bias) (* row_scale) (+ resid)
// stored either as the next layer's quantized, CTA-blocked activation operand
// (at a column offset, so layers can write into concatenated inputs) or as
// fp32. `linear_body` puts the activations on the TCU A side (M rows per CTA
// tile = 128); `linear_swap_body` puts the weights there instead (swap-AB, for
// skinny M such as decode batches: 32 activation rows per CTA tile).
#pragma once

#include <stdint.h>

enum : uint32_t {
  ACT_NONE = 0,
  ACT_RELU = 1,
  ACT_SWIGLU = 2,   // columns come in (gate, up) groups of SWIGLU_G; output N/2 columns
};

enum : uint32_t {
  OUT_FP32 = 0,     // fp32 row-major [M][ldo] at column col0
  OUT_QUANT = 1,    // quantized blocked operand, K' = ldo, at column col0
};

typedef struct {
  uint64_t a, a_sc;       // A operand (blocked by 128 rows)
  uint64_t w, w_sc;       // weights (blocked by the CTA tile N)
  uint64_t bias;          // fp32 [N] or 0
  uint64_t row_scale;     // fp32 [M] or 0 (e.g. MoE gate weights)
  uint64_t out, out_sc;   // see out_mode
  uint64_t aux;           // fp32 [M] receives column 0 before the activation, or 0
  float    ts_a, ts_w, ts_out;
  uint32_t M, N, K, num_kt;
  uint32_t act, out_mode;
  uint32_t ldo, col0;
  uint32_t w_row0;        // first weight row (grouped GEMM: expert offset)
  uint32_t tile_w_row0;   // per-M-tile weight row table (fp32 bits as uint32 [M/128]), 0 = use w_row0
  uint64_t resid;         // fp32 [M][ldo] added to the fp32 output, or 0
} linear_args_t;

#include "e2e_fmt.h"

namespace e2e {
// Interleave width of gate/up weight rows for ACT_SWIGLU: wide enough that one
// quantization run of outputs maps onto contiguous gate columns.
constexpr uint32_t SWIGLU_G_MAX = 32;
constexpr uint32_t SWIGLU_G = (IS_MX && F::blk > 16) ? SWIGLU_G_MAX : 16;
// CTA tiles derived from the core geometry (NT=32/NW=32 gives the original
// 128x64 TCU / 128x128 SIMT linear tiles and 256x32 / 128x32 swap-AB tiles).
// TCU: one TILE x TILE tile per warp, 4 warps across N (2 for swap-AB).
// SIMT: 4x4 outputs per thread on the GY x GX grid (TN = 1 for swap-AB).
#if E2E_TCU
// Wide enough for a SwiGLU tile to emit whole scale blocks in every format, so
// all formats share one tile shape.
constexpr uint32_t LIN_BN = (4 * TILE > 2 * SWIGLU_G_MAX) ? 4 * TILE : 2 * SWIGLU_G_MAX;
constexpr uint32_t LIN_BM = CTA_NW * TILE * TILE / LIN_BN;
constexpr uint32_t LSW_BM = (CTA_NW / 2) * TILE;
constexpr uint32_t LSW_BN = 2 * TILE;
#else
constexpr uint32_t LIN_BM = 4 * SIMT_GY;
constexpr uint32_t LIN_BN = 4 * SIMT_GX;
constexpr uint32_t LSW_BM = 4 * SIMT_GY;
constexpr uint32_t LSW_BN = SIMT_GX;
#endif
constexpr uint32_t LIN_CP = LIN_BN + 1;
constexpr uint32_t LSW_CP = LSW_BM + 1;
// smem needed by the two linear kernels.
constexpr uint32_t lin_lmem() {
#if E2E_TCU
  return tcu_gemm_lmem(LIN_BM, LIN_BN);
#else
  return simt_gemm_smem(LIN_BM, LIN_BN, 2);
#endif
}
constexpr uint32_t lin_swap_lmem() {
#if E2E_TCU
  return tcu_gemm_smem(LSW_BM, LSW_BN, 2) > LSW_BN * LSW_CP * 4 ? tcu_gemm_smem(LSW_BM, LSW_BN, 2) : LSW_BN * LSW_CP * 4;
#else
  return simt_gemm_smem(LSW_BM, LSW_BN, 2) > LSW_BN * LSW_CP * 4 ? simt_gemm_smem(LSW_BM, LSW_BN, 2) : LSW_BN * LSW_CP * 4;
#endif
}
}

#ifdef __VORTEX__
#include "e2e_dev.h"

namespace e2e {

// Epilogue over rows [r_lo, r_hi) of the fp32 C tile parked in smem as
// [activation row][output column] (pitch cp, `cs` pointing at row r_lo).
// tc = output columns in the tile; arb = row blocking of the output operand.
E2E_INL void linear_epilogue(const linear_args_t* arg, float* cs, uint32_t m0, uint32_t n0,
                             uint32_t r_lo, uint32_t r_hi, uint32_t tc = LIN_BN, uint32_t cp = LIN_CP,
                             uint32_t arb = LIN_BM) {
  const float* bias = reinterpret_cast<const float*>(arg->bias);
  const float* rsc = reinterpret_cast<const float*>(arg->row_scale);
  float* aux = reinterpret_cast<float*>(arg->aux);
  const float* resid = reinterpret_cast<const float*>(arg->resid);
  const float scale = arg->ts_a * arg->ts_w;
  const uint32_t act = arg->act;
  const uint32_t ocols = (act == ACT_SWIGLU) ? tc / 2 : tc;
  const uint32_t on0 = (act == ACT_SWIGLU) ? n0 / 2 : n0;
  const uint32_t rpr = ocols / QRUN;
  for (uint32_t run = threadIdx.x; run < (r_hi - r_lo) * rpr; run += blockDim.x) {
    const uint32_t r = run / rpr, c = (run % rpr) * QRUN;
    const uint32_t gr = m0 + r_lo + r;
    if (gr >= arg->M) continue;
    float* v = cs + r * cp + c;
    const float rs = rsc ? rsc[gr] : 1.0f;
    if (act == ACT_SWIGLU) {
      // Outputs c..c+QRUN of this tile live in gate columns of one group;
      // compute silu(gate) * up in place over those gate slots.
      const uint32_t gcol = (c / SWIGLU_G) * 2 * SWIGLU_G + c % SWIGLU_G;
      v = cs + r * cp + gcol;
      for (uint32_t i = 0; i < QRUN; ++i) {
        float gv = v[i] * scale + (bias ? bias[n0 + gcol + i] : 0.0f);
        float uv = v[i + SWIGLU_G] * scale + (bias ? bias[n0 + gcol + i + SWIGLU_G] : 0.0f);
        float sg = gv / (1.0f + exp2_dev(-1.44269504f * gv));
        v[i] = sg * uv * rs;
      }
    } else {
      for (uint32_t i = 0; i < QRUN; ++i) {
        float t = v[i] * scale + (bias ? bias[n0 + c + i] : 0.0f);
        if (aux && n0 + c + i == 0) aux[gr] = t;
        if (act == ACT_RELU) t = t > 0.0f ? t : 0.0f;
        v[i] = t * rs;
      }
    }
    const uint32_t gc = arg->col0 + on0 + c;
    if (resid)
      for (uint32_t i = 0; i < QRUN; ++i) v[i] += resid[gr * arg->ldo + gc + i];
    if (arg->out_mode == OUT_QUANT) {
      const uint32_t nkt = arg->ldo / KT;
      quant_store_run(v, QRUN, arg->ts_out,
                      reinterpret_cast<uint32_t*>(arg->out) + blk_word(gr, gc / F::epw, arb, nkt),
                      IS_MX ? reinterpret_cast<uint8_t*>(arg->out_sc) + blk_sc(gr, gc, arb, nkt) : nullptr);
    } else {
      float* o = reinterpret_cast<float*>(arg->out) + gr * arg->ldo + gc;
      for (uint32_t i = 0; i < QRUN; ++i) o[i] = v[i];
    }
  }
}

E2E_INL uint32_t linear_w_row0(const linear_args_t* arg, uint32_t mtile) {
  if (arg->tile_w_row0) return reinterpret_cast<const uint32_t*>(arg->tile_w_row0)[mtile];
  return arg->w_row0;
}

#if E2E_TCU
E2E_INL void linear_body(const linear_args_t* arg) {
  using G = tcu_gemm<LIN_BM, LIN_BN>;
  auto smem = reinterpret_cast<uint32_t*>(__local_mem());
  float* cs = reinterpret_cast<float*>(smem);
  const uint32_t m0 = blockIdx.y * LIN_BM, n0 = blockIdx.x * LIN_BN;
  operand_t A{reinterpret_cast<const uint32_t*>(arg->a), reinterpret_cast<const uint32_t*>(arg->a_sc), m0, 0, 0};
  operand_t B{reinterpret_cast<const uint32_t*>(arg->w), reinterpret_cast<const uint32_t*>(arg->w_sc),
              linear_w_row0(arg, blockIdx.y) + n0, 0, 0};
  G::run(smem, A, B, arg->num_kt, cs, LIN_CP);
  linear_epilogue(arg, cs, m0, n0, 0, LIN_BM);
}

// Swap-AB: weights are the TCU A operand (LSW_BM output rows per CTA, weights
// blocked by LSW_BM), activations the B operand (LSW_BN rows, blocked by
// LSW_BN); C is parked transposed so the shared epilogue sees [act row][col].
// Grid: x = N / LSW_BM, y = M / LSW_BN.
E2E_INL void linear_swap_body(const linear_args_t* arg) {
  using G = tcu_gemm<LSW_BM, LSW_BN>;
  auto smem = reinterpret_cast<uint32_t*>(__local_mem());
  float* cs = reinterpret_cast<float*>(smem);
  const uint32_t n0 = blockIdx.x * LSW_BM, m0 = blockIdx.y * LSW_BN;
  operand_t A{reinterpret_cast<const uint32_t*>(arg->w), reinterpret_cast<const uint32_t*>(arg->w_sc),
              arg->w_row0 + n0, 0, 0};
  operand_t B{reinterpret_cast<const uint32_t*>(arg->a), reinterpret_cast<const uint32_t*>(arg->a_sc), m0, 0, 0};
  G::template run<true>(smem, A, B, arg->num_kt, cs, LSW_CP);
  linear_epilogue(arg, cs, m0, n0, 0, LSW_BN, LSW_BM, LSW_CP, LSW_BN);
}
#else
E2E_INL void linear_body(const linear_args_t* arg) {
  using S = simt_gemm<LIN_BM, LIN_BN>;
  auto smem = reinterpret_cast<float*>(__local_mem());
  const uint32_t m0 = blockIdx.y * LIN_BM, n0 = blockIdx.x * LIN_BN;
  float acc[S::TM][S::TN] = {};
  operand_t A{reinterpret_cast<const uint32_t*>(arg->a), nullptr, m0, 0, 0};
  operand_t B{reinterpret_cast<const uint32_t*>(arg->w), nullptr, linear_w_row0(arg, blockIdx.y) + n0, 0, 0};
  S::run(acc, smem, A, B, arg->num_kt);
  // Park the tile in smem (in halves when a full fp32 tile exceeds the local
  // memory) and share the TCU epilogue.
  constexpr uint32_t H = (LIN_BM * LIN_CP * 4 > (1u << VX_CFG_LMEM_LOG_SIZE)) ? 2 : 1;
  constexpr uint32_t HR = LIN_BM / H, HT = S::TM / H;
  float* cs = smem;
  for (uint32_t h = 0; h < H; ++h) {
    vortex::barrier(2).arrive_and_wait();
    for (uint32_t i = HT * h; i < HT * h + HT; ++i)
      for (uint32_t j = 0; j < S::TN; ++j)
        cs[(S::ty() + S::GY * (i - HT * h)) * LIN_CP + S::tx() + S::GX * j] = acc[i][j];
    vortex::barrier(2).arrive_and_wait();
    linear_epilogue(arg, cs, m0, n0, HR * h, HR * h + HR);
  }
}

E2E_INL void linear_swap_body(const linear_args_t* arg) {
  using S = simt_gemm<LSW_BM, LSW_BN>;
  auto smem = reinterpret_cast<float*>(__local_mem());
  const uint32_t n0 = blockIdx.x * LSW_BM, m0 = blockIdx.y * LSW_BN;
  float acc[S::TM][S::TN] = {};
  operand_t A{reinterpret_cast<const uint32_t*>(arg->w), nullptr, arg->w_row0 + n0, 0, 0};
  operand_t B{reinterpret_cast<const uint32_t*>(arg->a), nullptr, m0, 0, 0};
  S::run(acc, smem, A, B, arg->num_kt);
  float* cs = smem;
  for (uint32_t i = 0; i < S::TM; ++i)
    for (uint32_t j = 0; j < S::TN; ++j)
      cs[(S::tx() + S::GX * j) * LSW_CP + S::ty() + S::GY * i] = acc[i][j];
  vortex::barrier(2).arrive_and_wait();
  linear_epilogue(arg, cs, m0, n0, 0, LSW_BN, LSW_BM, LSW_CP, LSW_BN);
}
#endif

} // namespace e2e
#endif
