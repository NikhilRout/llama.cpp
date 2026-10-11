// Kernel argument blocks shared by the Vortex backend (host) and its device
// kernels. Tensors are passed as a device address plus ggml's ne/nb (element
// counts and byte strides); every block is a plain struct of 32/64-bit fields
// so host and RV32 device agree on its layout.
#pragma once

#include <stdint.h>

#define VX_MAX_DIMS 4

// Division by a launch-invariant d as a multiply-shift (the device's integer
// divider is serial per lane): n / d == (mulhi(n, m) + n) >> s for n < 2^31.
typedef struct {
    uint32_t m, s;
} vx_div_t;

typedef struct {
    uint64_t addr;
    int32_t  ne[VX_MAX_DIMS];
    int32_t  nb[VX_MAX_DIMS];   // bytes
    vx_div_t dv[VX_MAX_DIMS];   // divisors for ne[]
} vx_tensor_t;

static inline vx_div_t vx_div_make(uint32_t d) {
    uint32_t s = 0;
    while ((1ull << s) < d) ++s;
    vx_div_t r = { (uint32_t)(((1ull << 32) * ((1ull << s) - d)) / d + 1), s };
    return r;
}

// f32 activation rows -> CTA-blocked operand for the TCU GEMM (zero-padded to
// Tp rows). One tensor scale per row lands in row_ts (1 for unscaled formats).
typedef struct {
    vx_tensor_t src;            // f32 [K, T]
    uint64_t    dst;            // blocked words
    uint64_t    dst_sc;         // blocked scales (MX/NV/LNS), or 0
    uint64_t    row_ts;         // f32 [Tp]
    uint32_t    T, Tp, K, rb;
    vx_div_t    runs;           // K / QRUN
} vx_act_args_t;

// Decode-time y[T][N] = x[T][K] . W[N][K]^T on the SIMT lanes for 16-bit
// unscaled formats, reading W in the TCU's CTA-blocked layout (row block rb).
#define VX_GEMV_TMAX 2
typedef struct {
    uint64_t    w, out;
    vx_tensor_t x;              // f32 [K, T]
    uint32_t    N, K, nkt, rb, T, ldo;
} vx_gemv_args_t;

// dst = a OP b with ggml broadcasting of b over a (b's dims divide a's).
enum { VX_BIN_ADD = 0, VX_BIN_MUL = 1 };
// Contiguous fast paths: b same shape as a, or b one row broadcast over a's rows.
enum { VX_BCAST_GENERAL = 0, VX_BCAST_NONE = 1, VX_BCAST_ROW = 2 };
typedef struct {
    vx_tensor_t a, b, dst;
    uint32_t    op, bcast;
} vx_bin_args_t;

typedef struct {
    vx_tensor_t src, dst;
    float       eps;
    uint64_t    gamma;          // fused MUL by a [ne0] f32 vector, or 0
} vx_rms_args_t;

// ggml rope (NORM mode: adjacent pairs) with YaRN parameters and optional
// per-dimension frequency factors.
typedef struct {
    vx_tensor_t src, dst;
    uint64_t    pos;            // int32 [ne2]
    uint64_t    freq_factors;   // f32 [n_dims/2] or 0
    int32_t     n_dims, mode;
    float       freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    float       corr_lo, corr_hi;
    vx_div_t    half;           // ne0 / 2
} vx_rope_args_t;

// dst = silu(gate) * up
typedef struct {
    vx_tensor_t gate, up, dst;
} vx_glu_args_t;

// Flash attention, one warp per (query row, head); online softmax over the
// keys. q/k/v/mask are read through their strides; k/v/mask may be f16.
typedef struct {
    vx_tensor_t q, k, v, mask, dst;
    uint32_t    k_f16, v_f16, mask_f16, has_mask;
    float       scale, softcap;
    vx_div_t    gqa;            // q heads per kv head
} vx_fa_args_t;

// dst rows [idx[i]] = src rows [i]   (KV-cache append); f32 -> f32 or f16.
typedef struct {
    vx_tensor_t src, idx, dst;
    uint32_t    idx_i64, dst_f16;
} vx_set_rows_args_t;

// dst rows = src rows [idx]   (f32 source)
typedef struct {
    vx_tensor_t src, idx, dst;
    uint32_t    idx_i64;
} vx_get_rows_args_t;

// Element-wise strided copy (cpy / dup / cont), f32 or f16 on either side.
typedef struct {
    vx_tensor_t src, dst;
    uint32_t    src_f16, dst_f16;
} vx_cpy_args_t;
