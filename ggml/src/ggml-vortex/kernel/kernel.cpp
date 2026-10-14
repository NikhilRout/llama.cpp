#include "e2e_linear.h"
#include "vx_ops.h"

using namespace e2e;

// ---------------------------------------------------------------------------
// TCU GEMM: y[M][N] = x[M][K] . W[N][K]^T with the weights on the TCU A side
// (swap-AB), so a decode step pads its activation rows only up to LSW_BN.
// ---------------------------------------------------------------------------
__kernel void linear_swap(linear_args_t* __UNIFORM__ arg) { linear_swap_body(arg); }

// Skinny (M <= TT * TILE) swap-AB GEMM, one CTA. Each TILE x KT weight tile
// is contiguous in the blocked layout, so a warp loads its A fragments
// straight from DRAM. The activation rows and their scales are first copied
// to local memory: as D-cache hits they would queue behind the weight misses
// in the cache's MSHRs. With TT = 2, warp pairs take the same weight rows for
// the two activation tiles, so the second warp's weight loads hit the lines
// the first one fetched and DRAM streams the weights once per 2 * TILE rows.
// One lane per output row stores (needs NT == TILE).
#if E2E_TCU
constexpr uint32_t DEC_CS_WORDS = CTA_NW * TILE * TILE;

// NT=8 B fragment: register r, lane l hold column (l%4)/2 + 2*(l/4) + 4*(r%2),
// K word l%2 + 2*(r/2). With M <= 4 the odd registers are all padding, so
// HALF loads only the even ones and keeps the odd ones zero.
template <bool HALF>
E2E_INL void dec_load_b(ctx::fragment_b& fb, const uint32_t* xp, uint32_t lane) {
  if constexpr (HALF && NT == 8) {
    const float* p = reinterpret_cast<const float*>(xp) + ((lane % 4) / 2 + 2 * (lane / 4)) * KT_WORDS + lane % 2;
#pragma unroll
    for (uint32_t r = 0; r < ctx::fragment_b::NR; r += 2) fb.data[r] = p[(r / 2) * 2];
  } else {
    ctx::load_matrix_sync<vt::col_major>(fb, xp, KT_WORDS * (4 / sizeof(ctx::input_t)));
  }
}

template <bool HALF, uint32_t TT>
E2E_INL void linear_dec_body(const linear_args_t* arg) {
  constexpr uint32_t DLDM = KT_WORDS * (4 / sizeof(ctx::input_t));
  constexpr uint32_t SCW = LSW_BM / TILE * SC_TILE_WORDS;
  constexpr uint32_t XT = TT * TILE * KT_WORDS;    // activation words per K-tile in lmem
  constexpr uint32_t ST = TT * SC_TILE_WORDS;      // activation scale words per K-tile in lmem
  constexpr uint32_t RW = CTA_NW / TT;             // warps across weight rows
  static_assert(TT * TILE <= LSW_BN, "the activation tiles share one LSW_BN row block");
  const uint32_t wid = threadIdx.x / NT, lane = threadIdx.x % NT;
  const uint32_t tt = wid % TT, rw = wid / TT;
  const uint32_t nkt = arg->num_kt;
  // Activation rows of this warp's tile.
  const uint32_t T = arg->M > tt * TILE ? (arg->M - tt * TILE < TILE ? arg->M - tt * TILE : TILE) : 0;
  uint32_t* lmem = reinterpret_cast<uint32_t*>(__local_mem());
  float* cs = reinterpret_cast<float*>(lmem) + wid * TILE * TILE;
  uint32_t* xs = lmem + DEC_CS_WORDS;
  uint32_t* ss = xs + nkt * XT;
  {
    const uint32_t* X = reinterpret_cast<const uint32_t*>(arg->a);
    for (uint32_t i = threadIdx.x; i < nkt * XT; i += CTA_THREADS)
      xs[i] = X[(i / XT) * (LSW_BN * KT_WORDS) + i % XT];
    if constexpr (IS_MX) {
      const uint32_t* XS = reinterpret_cast<const uint32_t*>(arg->a_sc);
      for (uint32_t i = threadIdx.x; i < nkt * ST; i += CTA_THREADS)
        ss[i] = XS[(i / ST) * (LSW_BN / TILE * SC_TILE_WORDS) + i % ST];
    }
    vortex::barrier(0).arrive_and_wait();
  }
  const float* rts = reinterpret_cast<const float*>(arg->row_scale);
  if (rts) rts += tt * TILE;
  const float scale = arg->ts_a * arg->ts_w;
  float* out = reinterpret_cast<float*>(arg->out) + tt * TILE * arg->ldo;
  for (uint32_t r0 = rw * TILE; r0 < arg->N; r0 += RW * TILE) {
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(arg->w) + blk_word(r0, 0, LSW_BM, nkt);
    const uint32_t* xp = xs + tt * TILE * KT_WORDS;
    ctx::fragment_a fa;
    ctx::fragment_b fb;
    ctx::fragment_acc acc;
    ctx::fill_fragment(acc, 0);
    if constexpr (HALF) ctx::fill_fragment(fb, 0);
    const uint32_t* gsa = nullptr;
    const uint32_t* gsb = ss + tt * SC_TILE_WORDS;
    if constexpr (IS_MX) {
      gsa = reinterpret_cast<const uint32_t*>(arg->w_sc) + (r0 / LSW_BM) * nkt * SCW + (r0 % LSW_BM) / TILE * SC_TILE_WORDS;
      ctx::load_mx_metadata(fa, gsa);
      ctx::load_mx_metadata(fb, gsb);
    }
    for (uint32_t kt = 0; kt < nkt; ++kt) {
      dec_load_b<HALF>(fb, xp, lane);
      ctx::load_matrix_sync(fa, wp, DLDM);
      ctx::mma_sync(acc, fa, fb, acc);
      // Next K-tile's scales: the TCU consumes uops in order, so these land
      // after this MMA read the current ones. The last prefetch reads one
      // tile past the end (operand buffers carry slack).
      if constexpr (IS_MX) {
        gsa += SCW;
        gsb += ST;
        ctx::load_mx_metadata(fa, gsa);
        ctx::load_mx_metadata(fb, gsb);
      }
      wp += LSW_BM * KT_WORDS;
      xp += XT;
    }
    ctx::store_matrix_sync<vt::col_major>(cs, acc, TILE);
    __asm__ volatile("" ::: "memory");
    for (uint32_t t = 0; t < T; ++t)
      out[t * arg->ldo + r0 + lane] = cs[t * TILE + lane] * scale * (rts ? rts[t] : 1.0f);
  }
}
#endif

__kernel void linear_dec(linear_args_t* __UNIFORM__ arg) {
#if E2E_TCU
  if constexpr (NT == TILE) {
    if (arg->M <= 4)         linear_dec_body<true, 1>(arg);
    else if (arg->M <= TILE) linear_dec_body<false, 1>(arg);
    else                     linear_dec_body<false, 2>(arg);
  }
#else
  (void)arg;
#endif
}

// ---------------------------------------------------------------------------
// Element helpers
// ---------------------------------------------------------------------------
E2E_INL uint32_t gtid() { return blockIdx.x * blockDim.x + threadIdx.x; }
E2E_INL uint32_t gthreads() { return gridDim.x * blockDim.x; }

E2E_INL float f16_to_f32(uint32_t h) {
    const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    if (e == 0) return u2f(s | f2u((float)m * 5.9604644775390625e-8f));
    if (e == 31) return u2f(s | 0x7f800000u | (m << 13));
    return u2f(s | ((e + 112u) << 23) | (m << 13));
}

E2E_INL uint32_t fdiv(uint32_t n, vx_div_t d) {
    uint32_t hi;
    __asm__("mulhu %0, %1, %2" : "=r"(hi) : "r"(n), "r"(d.m));
    return (hi + n) >> d.s;
}

// n % ne[k] of t
E2E_INL uint32_t fmod_dim(uint32_t n, const vx_tensor_t& t, int k) { return n - fdiv(n, t.dv[k]) * t.ne[k]; }

struct idx4 { uint32_t i0, i1, i2, i3; };

E2E_INL idx4 unravel(const vx_tensor_t& t, uint32_t i) {
    const uint32_t q0 = fdiv(i, t.dv[0]), q1 = fdiv(q0, t.dv[1]), q2 = fdiv(q1, t.dv[2]);
    return {i - q0 * t.ne[0], q0 - q1 * t.ne[1], q1 - q2 * t.ne[2], q2};
}

// Row r of t (dims 1..3) -> (i1, i2, i3)
E2E_INL idx4 unravel_row(const vx_tensor_t& t, uint32_t r) {
    const uint32_t q1 = fdiv(r, t.dv[1]), q2 = fdiv(q1, t.dv[2]);
    return {0, r - q1 * t.ne[1], q1 - q2 * t.ne[2], q2};
}

template <typename T>
E2E_INL T* at(const vx_tensor_t& t, uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
    return reinterpret_cast<T*>((uintptr_t)t.addr + i0 * t.nb[0] + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3]);
}

E2E_INL float load(const vx_tensor_t& t, bool f16, uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
    return f16 ? f16_to_f32(*at<const uint16_t>(t, i0, i1, i2, i3)) : *at<const float>(t, i0, i1, i2, i3);
}

E2E_INL float silu(float x) { return x / (1.0f + exp2_shared(-1.44269504f * x)); }

E2E_INL float rsqrt_dev(float x) {
    float r;
    __asm__("fsqrt.s %0, %1" : "=f"(r) : "f"(x));
    return 1.0f / r;
}

// ---------------------------------------------------------------------------
// f32 activation rows -> blocked TCU operand, one thread per quantization run.
// ---------------------------------------------------------------------------
__kernel void act_pack(vx_act_args_t* __UNIFORM__ arg) {
    const uint32_t K = arg->K, runs = K / QRUN, nkt = K / KT;
    const float* ts = reinterpret_cast<const float*>(arg->row_ts);
    for (uint32_t g = gtid(); g < arg->Tp * runs; g += gthreads()) {
        const uint32_t r = fdiv(g, arg->runs), c0 = (g - r * runs) * QRUN;
        float x[QRUN];
        for (uint32_t i = 0; i < QRUN; ++i)
            x[i] = r < arg->T ? *at<const float>(arg->src, c0 + i, r, 0, 0) : 0.0f;
        quant_store_run(x, QRUN, HAS_TSCALE ? ts[r] : 1.0f,
                        reinterpret_cast<uint32_t*>(arg->dst) + blk_word(r, c0 / F::epw, arg->rb, nkt),
                        IS_MX ? reinterpret_cast<uint8_t*>(arg->dst_sc) + blk_sc(r, c0, arg->rb, nkt) : nullptr);
    }
}

// ---------------------------------------------------------------------------
// GEMV for decode: a warp owns RR consecutive weight rows; lane l reads word l
// of each row's K-tile slice (one contiguous run per row) and the warp reduces
// across lanes once per row.
// ---------------------------------------------------------------------------
static_assert(KT_WORDS == NT, "one K-tile word per lane");
static_assert(std::is_same<IT, vt::bf16>::value || !(F::bits == 16), "GEMV decodes bf16 words");

// The high bf16 of each word is read with an FP load: the low element rides
// along as extra mantissa bits (< 1 bf16 ulp), which saves an ALU and an FPU
// op per word. The core fetches ~1 warp instruction per cycle, so instruction
// count is the GEMV's cost.
template <uint32_t TT, uint32_t RR>
E2E_INL void gemv_body(const vx_gemv_args_t* arg) {
    const uint32_t lane = threadIdx.x % NT, nw = gthreads() / NT;
    const uint32_t nkt = arg->nkt, rb = arg->rb;
    const uint32_t* W = reinterpret_cast<const uint32_t*>(arg->w);
    float* out = reinterpret_cast<float*>(arg->out);
    for (uint32_t r0 = (gtid() / NT) * RR; r0 < arg->N; r0 += nw * RR) {
        const uint32_t* wp = W + blk_word(r0, 0, rb, nkt) + lane;
        const uint32_t step = rb * KT_WORDS;
        float acc[TT][RR];
        const float* xp[TT];
#pragma unroll
        for (uint32_t t = 0; t < TT; ++t) {
            xp[t] = at<const float>(arg->x, 2 * lane, t, 0, 0);
#pragma unroll
            for (uint32_t j = 0; j < RR; ++j) acc[t][j] = 0.0f;
        }
        for (uint32_t kt = 0; kt < nkt; ++kt) {
            float x0[TT], x1[TT];
#pragma unroll
            for (uint32_t t = 0; t < TT; ++t) {
                x0[t] = xp[t][0];
                x1[t] = xp[t][1];
                xp[t] += KT;
            }
#pragma unroll
            for (uint32_t j = 0; j < RR; ++j) {
                const uint32_t w = wp[j * KT_WORDS];
                const float hi = reinterpret_cast<const float*>(wp)[j * KT_WORDS];
                const float lo = u2f(w << 16);
#pragma unroll
                for (uint32_t t = 0; t < TT; ++t) acc[t][j] = fma_dev(hi, x1[t], fma_dev(lo, x0[t], acc[t][j]));
            }
            wp += step;
        }
#pragma unroll
        for (uint32_t t = 0; t < TT; ++t)
#pragma unroll
            for (uint32_t j = 0; j < RR; ++j) out[t * arg->ldo + r0 + j] = warp_sum(acc[t][j]);
    }
}

__kernel void gemv(vx_gemv_args_t* __UNIFORM__ arg) {
    switch (arg->T) {
        case 1:  gemv_body<1, 16>(arg); break;
        default: gemv_body<2, 8>(arg); break;
    }
}

// Per-row tensor scale for the scaled formats: one warp per row.
__kernel void act_scale(vx_act_args_t* __UNIFORM__ arg) {
    float* ts = reinterpret_cast<float*>(arg->row_ts);
    const uint32_t lane = threadIdx.x % NT, nw = gthreads() / NT;
    for (uint32_t r = gtid() / NT; r < arg->Tp; r += nw) {
        float amax = 0.0f;
        if (r < arg->T)
            for (uint32_t k = lane; k < arg->K; k += NT) amax = fmax_(amax, fabs_(*at<const float>(arg->src, k, r, 0, 0)));
        amax = warp_max(amax);
        ts[r] = tensor_scale_for(amax);
    }
}

// ---------------------------------------------------------------------------
// Element-wise ops
// ---------------------------------------------------------------------------
__kernel void binary(vx_bin_args_t* __UNIFORM__ arg) {
    const vx_tensor_t& d = arg->dst;
    const uint32_t n = d.ne[0] * d.ne[1] * d.ne[2] * d.ne[3];
    if (arg->bcast != VX_BCAST_GENERAL) {
        const float* pa = reinterpret_cast<const float*>(arg->a.addr);
        const float* pb = reinterpret_cast<const float*>(arg->b.addr);
        float* pd = reinterpret_cast<float*>(d.addr);
        const bool row = arg->bcast == VX_BCAST_ROW, add = arg->op == VX_BIN_ADD;
        for (uint32_t i = gtid(); i < n; i += gthreads()) {
            const float b = pb[row ? fmod_dim(i, d, 0) : i];
            pd[i] = add ? pa[i] + b : pa[i] * b;
        }
        return;
    }
    for (uint32_t i = gtid(); i < n; i += gthreads()) {
        const idx4 x = unravel(d, i);
        const vx_tensor_t& bt = arg->b;
        const float a = *at<const float>(arg->a, x.i0, x.i1, x.i2, x.i3);
        const float b = *at<const float>(bt, fmod_dim(x.i0, bt, 0), fmod_dim(x.i1, bt, 1), fmod_dim(x.i2, bt, 2), fmod_dim(x.i3, bt, 3));
        *at<float>(d, x.i0, x.i1, x.i2, x.i3) = arg->op == VX_BIN_ADD ? a + b : a * b;
    }
}

__kernel void glu_swiglu(vx_glu_args_t* __UNIFORM__ arg) {
    const vx_tensor_t& d = arg->dst;
    const uint32_t n = d.ne[0] * d.ne[1] * d.ne[2] * d.ne[3];
    for (uint32_t i = gtid(); i < n; i += gthreads()) {
        const idx4 x = unravel(d, i);
        *at<float>(d, x.i0, x.i1, x.i2, x.i3) =
            silu(*at<const float>(arg->gate, x.i0, x.i1, x.i2, x.i3)) * *at<const float>(arg->up, x.i0, x.i1, x.i2, x.i3);
    }
}

__kernel void cpy(vx_cpy_args_t* __UNIFORM__ arg) {
    const vx_tensor_t &s = arg->src, &d = arg->dst;
    const uint32_t n = s.ne[0] * s.ne[1] * s.ne[2] * s.ne[3];
    for (uint32_t i = gtid(); i < n; i += gthreads()) {
        const idx4 x = unravel(s, i), y = unravel(d, i);
        const float v = load(s, arg->src_f16, x.i0, x.i1, x.i2, x.i3);
        if (arg->dst_f16) *at<uint16_t>(d, y.i0, y.i1, y.i2, y.i3) = (uint16_t)f32_to_f16(v);
        else              *at<float>(d, y.i0, y.i1, y.i2, y.i3) = v;
    }
}

// ---------------------------------------------------------------------------
// RMS norm (optionally fused with the following MUL by a gamma vector),
// one warp per row.
// ---------------------------------------------------------------------------
__kernel void rms_norm(vx_rms_args_t* __UNIFORM__ arg) {
    const vx_tensor_t &s = arg->src, &d = arg->dst;
    const uint32_t lane = threadIdx.x % NT, nw = gthreads() / NT;
    const uint32_t rows = s.ne[1] * s.ne[2] * s.ne[3], K = s.ne[0];
    const float* g = reinterpret_cast<const float*>(arg->gamma);
    for (uint32_t r = gtid() / NT; r < rows; r += nw) {
        const idx4 ri = unravel_row(s, r);
        const float* x = at<const float>(s, 0, ri.i1, ri.i2, ri.i3);
        float ss = 0.0f;
        for (uint32_t k = lane; k < K; k += NT) ss = fma_dev(x[k], x[k], ss);
        ss = warp_sum(ss);
        const float inv = rsqrt_dev(ss / (float)K + arg->eps);
        float* y = at<float>(d, 0, ri.i1, ri.i2, ri.i3);
        for (uint32_t k = lane; k < K; k += NT) y[k] = x[k] * inv * (g ? g[k] : 1.0f);
    }
}

// ---------------------------------------------------------------------------
// RoPE, one thread per rotated pair. src/dst [head_dim, n_head, T, 1].
// ---------------------------------------------------------------------------
__kernel void rope(vx_rope_args_t* __UNIFORM__ arg) {
    const vx_tensor_t &s = arg->src, &d = arg->dst;
    const int32_t* pos = reinterpret_cast<const int32_t*>(arg->pos);
    const float* ff = reinterpret_cast<const float*>(arg->freq_factors);
    const uint32_t D = s.ne[0], H = s.ne[1], T = s.ne[2], half = D / 2;
    const uint32_t nd = (uint32_t)arg->n_dims;
    // log2(freq_base^(-2/n_dims)) is folded into freq_base on the host.
    const float lg_theta_scale = arg->freq_base;
    for (uint32_t g = gtid(); g < half * H * T; g += gthreads()) {
        const uint32_t q = fdiv(g, arg->half), ip = g - q * half, t = fdiv(q, s.dv[1]), h = q - t * H;
        float* y = at<float>(d, 0, h, t, 0);
        const float* x = at<const float>(s, 0, h, t, 0);
        const uint32_t i0 = 2 * ip;
        if (i0 >= nd) {   // past the rotated dims: copy through
            y[i0] = x[i0];
            y[i0 + 1] = x[i0 + 1];
            continue;
        }
        float theta_extrap = (float)pos[t] * exp2_shared(lg_theta_scale * (float)ip);
        if (ff) theta_extrap /= ff[ip];
        float theta = arg->freq_scale * theta_extrap;
        if (arg->ext_factor != 0.0f) {
            float yr = ((float)ip - arg->corr_lo) / fmax_(0.001f, arg->corr_hi - arg->corr_lo);
            yr = yr < 0.0f ? 0.0f : (yr > 1.0f ? 1.0f : yr);
            const float mix = (1.0f - yr) * arg->ext_factor;
            theta = theta * (1.0f - mix) + theta_extrap * mix;
        }
        const float c = cos_poly(theta) * arg->attn_factor, sn = sin_poly(theta) * arg->attn_factor;
        // NORM pairs adjacent dims; NEOX pairs dim i with i + n_dims/2.
        const uint32_t a = arg->mode == 2 ? ip : i0, b = arg->mode == 2 ? ip + nd / 2 : i0 + 1;
        const float x0 = x[a], x1 = x[b];
        y[a] = x0 * c - x1 * sn;
        y[b] = x0 * sn + x1 * c;
    }
}

// ---------------------------------------------------------------------------
// Flash attention: one warp per (query row, head), online softmax over keys.
// q [D, nq, H, 1], k/v [D, nkv, Hkv, 1], mask [nkv, nq_pad, ...], dst [D, H, nq, 1].
// ---------------------------------------------------------------------------
__kernel void flash_attn(vx_fa_args_t* __UNIFORM__ arg) {
    const vx_tensor_t &q = arg->q, &k = arg->k, &v = arg->v, &mk = arg->mask, &d = arg->dst;
    const uint32_t D = q.ne[0], nq = q.ne[1], H = q.ne[2], nkv = k.ne[1];
    const uint32_t lane = threadIdx.x % NT, nw = gthreads() / NT;
    constexpr uint32_t DMAX = 256 / NT;   // head dims per lane (D <= 256)
    for (uint32_t w = gtid() / NT; w < nq * H; w += nw) {
        const uint32_t h = fdiv(w, q.dv[1]), iq = w - h * nq, hk = fdiv(h, arg->gqa);
        const uint32_t mh = arg->has_mask ? fmod_dim(h, mk, 2) : 0;
        float qr[DMAX], o[DMAX];
        const uint32_t dpl = D / NT;
        for (uint32_t i = 0; i < dpl; ++i) {
            qr[i] = *at<const float>(q, lane + i * NT, iq, h, 0) * arg->scale;
            o[i] = 0.0f;
        }
        float m = -1e30f, l = 0.0f;
        for (uint32_t j = 0; j < nkv; ++j) {
            float mv = 0.0f;
            if (arg->has_mask) {
                mv = load(mk, arg->mask_f16, j, iq, mh, 0);
                if (mv < -1e30f) continue;   // -inf: key not visible
            }
            float s = 0.0f;
            for (uint32_t i = 0; i < dpl; ++i) s = fma_dev(qr[i], load(k, arg->k_f16, lane + i * NT, j, hk, 0), s);
            s = warp_sum(s);
            if (arg->softcap != 0.0f) {
                // tanh(x) = 1 - 2 / (exp(2x) + 1)
                const float x = s / arg->softcap;
                s = arg->softcap * (1.0f - 2.0f / (exp2_shared(2.88539008f * x) + 1.0f));
            }
            s += mv;
            const float mn = fmax_(m, s);
            const float a = exp2_shared((m - mn) * 1.44269504f), p = exp2_shared((s - mn) * 1.44269504f);
            l = l * a + p;
            for (uint32_t i = 0; i < dpl; ++i) o[i] = fma_dev(p, load(v, arg->v_f16, lane + i * NT, j, hk, 0), o[i] * a);
            m = mn;
        }
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (uint32_t i = 0; i < dpl; ++i) *at<float>(d, lane + i * NT, h, iq, 0) = o[i] * inv;
    }
}

// ---------------------------------------------------------------------------
// Row gather / scatter.
// ---------------------------------------------------------------------------
E2E_INL uint32_t row_index(const vx_tensor_t& idx, bool i64, uint32_t i0, uint32_t i1, uint32_t i2) {
    return i64 ? (uint32_t)*at<const int64_t>(idx, i0, i1, i2, 0) : (uint32_t)*at<const int32_t>(idx, i0, i1, i2, 0);
}

// dst row (i1, i2, i3) <- src row idx[i1, i2, i3] (f32)
__kernel void get_rows(vx_get_rows_args_t* __UNIFORM__ arg) {
    const vx_tensor_t &s = arg->src, &d = arg->dst;
    const uint32_t n = d.ne[0] * d.ne[1] * d.ne[2] * d.ne[3];
    for (uint32_t g = gtid(); g < n; g += gthreads()) {
        const idx4 x = unravel(d, g);
        const uint32_t sr = row_index(arg->idx, arg->idx_i64, x.i1, x.i2, x.i3);
        *at<float>(d, x.i0, x.i1, x.i2, x.i3) = *at<const float>(s, x.i0, sr, x.i2, x.i3);
    }
}

// dst row idx[i1, i2 % idx.ne1, i3 % idx.ne2] (at i2, i3) <- src row (i1, i2, i3)
__kernel void set_rows(vx_set_rows_args_t* __UNIFORM__ arg) {
    const vx_tensor_t &s = arg->src, &d = arg->dst, &ix = arg->idx;
    const uint32_t n = s.ne[0] * s.ne[1] * s.ne[2] * s.ne[3];
    for (uint32_t g = gtid(); g < n; g += gthreads()) {
        const idx4 x = unravel(s, g);
        const uint32_t dr = row_index(ix, arg->idx_i64, x.i1, fmod_dim(x.i2, ix, 1), fmod_dim(x.i3, ix, 2));
        const float v = *at<const float>(s, x.i0, x.i1, x.i2, x.i3);
        if (arg->dst_f16) *at<uint16_t>(d, x.i0, dr, x.i2, x.i3) = (uint16_t)f32_to_f16(v);
        else              *at<float>(d, x.i0, dr, x.i2, x.i3) = v;
    }
}
