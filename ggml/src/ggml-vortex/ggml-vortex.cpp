// Vortex GPGPU backend. Weights, KV cache and activations live in device
// memory; a graph is recorded into one Command Processor batch (one doorbell,
// one completion wait) so the host only moves graph inputs and outputs.
// MUL_MAT runs on the TCU (weights repacked into its CTA-blocked operand
// layout, quantized to ITYPE, on upload); the other ops run as SIMT kernels on
// the ALU/FPU lanes.

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-vortex.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if __has_include(<vortex2.h>)
#include <vortex2.h>
#else
#include <vortex.h>
#endif
#include "e2e_linear.h"
#include "vx_ops.h"

#ifndef GGML_VORTEX_KERNEL_PATH
#define GGML_VORTEX_KERNEL_PATH "kernel.vxbin"
#endif

using namespace e2e;

#define VX_CHECK(expr) do {                                                        \
        vx_result_t _r = (vx_result_t)(expr);                                      \
        if (_r != VX_SUCCESS) {                                                    \
            GGML_ABORT("%s: '%s' returned %d", __func__, #expr, (int)_r);          \
        }                                                                          \
    } while (0)

namespace {

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// Slack past the end of every device allocation: the GEMM mainloop may
// prefetch one K-tile beyond the operand it streams.
constexpr uint64_t VX_SLACK = 64 * 1024;

// A weight stored in the TCU's blocked operand layout (set by set_tensor).
struct vx_weight_info {
    float    ts = 1.0f;     // per-tensor scale (scaled formats)
    uint64_t scales = 0;    // device address of the blocked scale tiles, or 0
    std::vector<uint8_t> host;  // original bytes, kept for GGML_VORTEX_ANY_WEIGHT read-back / GGML_VORTEX_CHECK
};

// A launch recorded for the batch: the args blob must outlive the submission.
struct vx_rec {
    vx_launch_info_t     li;
    std::vector<uint8_t> args;
    const char *         name;
    std::vector<std::pair<uint32_t, uint32_t>> dcrs;   // DCR writes issued just before the launch
    const ggml_tensor *  node = nullptr;                // set on the launch that completes a MUL_MAT
};

struct vx_device_ctx {
    std::mutex  mtx;
    bool        ready = false;
    vx_device_h dev   = nullptr;
    vx_queue_h  q     = nullptr;
    vx_module_h mod   = nullptr;
    std::map<std::string, vx_kernel_h> kernels;

    // Activation scratch for MUL_MAT (packed operand, scales, row scales).
    vx_buffer_h scr_h = nullptr;
    uint64_t    scr_addr = 0, scr_size = 0;

    bool verbose = false, profile = false, fuse = true;
    std::map<std::string, std::pair<uint64_t, double>> prof;   // name -> (launches, ms)
    double t_submit_ms = 0;
    uint64_t n_graphs = 0, n_launches = 0;

    void open() {
        if (ready) {
            return;
        }
        verbose = getenv("GGML_VORTEX_VERBOSE") != nullptr;
        profile = getenv("GGML_VORTEX_PROFILE") != nullptr;
        fuse    = getenv("GGML_VORTEX_NOFUSE") == nullptr;
        const char * kernel = getenv("GGML_VORTEX_KERNEL");
        if (!kernel) {
            kernel = GGML_VORTEX_KERNEL_PATH;
        }
        VX_CHECK(vx_device_open(0, &dev));
        vx_queue_info_t qi = {};
        qi.struct_size = sizeof(qi);
        qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
        VX_CHECK(vx_queue_create(dev, &qi, &q));
        VX_CHECK(vx_module_load_file(dev, kernel, &mod));
        for (const char * k : {"linear_swap", "linear_dec", "gemv", "act_pack", "act_scale", "binary", "glu_swiglu", "cpy",
                               "rms_norm", "rope", "flash_attn", "get_rows", "set_rows"}) {
            vx_kernel_h h = nullptr;
            VX_CHECK(vx_module_get_kernel(mod, k, &h));
            kernels[k] = h;
        }
        uint64_t nt = 0, nw = 0, nc = 0;
        VX_CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
        VX_CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
        VX_CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &nc));
        if (nt != VX_CFG_NUM_THREADS || nw != VX_CFG_NUM_WARPS) {
            GGML_ABORT("Vortex device has %lu threads x %lu warps, backend built for %d x %d",
                       (unsigned long)nt, (unsigned long)nw, VX_CFG_NUM_THREADS, VX_CFG_NUM_WARPS);
        }
        GGML_LOG_INFO("%s: Vortex device: %lu core(s) x %lu warps x %lu threads, %s GEMM operands, kernel %s\n",
                      __func__, (unsigned long)nc, (unsigned long)nw, (unsigned long)nt, F::name, kernel);
        ready = true;
    }

    void alloc(uint64_t bytes, vx_buffer_h * h, uint64_t * addr) {
        VX_CHECK(vx_buffer_create(dev, bytes + VX_SLACK, VX_MEM_READ_WRITE, h));
        VX_CHECK(vx_buffer_address(*h, addr));
    }

    void wait(vx_event_h ev) {
        VX_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
        vx_event_release(ev);
    }

    void write(vx_buffer_h h, uint64_t off, const void * src, uint64_t n) {
        vx_event_h ev = nullptr;
        VX_CHECK(vx_enqueue_write(q, h, off, src, n, 0, nullptr, &ev));
        wait(ev);
    }

    void read(void * dst, vx_buffer_h h, uint64_t off, uint64_t n) {
        vx_event_h ev = nullptr;
        VX_CHECK(vx_enqueue_read(q, dst, h, off, n, 0, nullptr, &ev));
        wait(ev);
    }

    void ensure_scratch(uint64_t bytes) {
        if (scr_h && scr_size >= bytes) {
            return;
        }
        if (scr_h) {
            vx_buffer_release(scr_h);
        }
        scr_size = std::max<uint64_t>(bytes, 1 << 20);
        alloc(scr_size, &scr_h, &scr_addr);
    }

    void print_stats() {
        if (!ready) {
            return;
        }
        GGML_LOG_INFO("%s: Vortex: %lu graphs, %lu launches, %.1f ms submitted\n", __func__,
                      (unsigned long)n_graphs, (unsigned long)n_launches, t_submit_ms);
        for (auto & [k, v] : prof) {
            GGML_LOG_INFO("%s:   %-12s %8lu launches %10.1f ms\n", __func__, k.c_str(), (unsigned long)v.first, v.second);
        }
    }
};

vx_device_ctx & vx_dev() {
    static vx_device_ctx ctx;
    return ctx;
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

uint64_t dev_addr(const ggml_tensor * t) {
    return (uint64_t)(uintptr_t)t->data;
}

vx_tensor_t vt_of(const ggml_tensor * t) {
    vx_tensor_t v = {};
    v.addr = t ? dev_addr(t) : 0;
    for (int i = 0; i < VX_MAX_DIMS; ++i) {
        v.ne[i] = t ? (int32_t)t->ne[i] : 1;
        v.nb[i] = t ? (int32_t)t->nb[i] : 0;
        v.dv[i] = vx_div_make((uint32_t)v.ne[i]);
    }
    return v;
}

float bf16_to_f32(uint16_t h) {
    return u2f((uint32_t)h << 16);
}

// 2D BF16 weights the TCU GEMM can stream: stored in the blocked layout, so
// only whole contiguous tensors (a view would index the packed bytes as rows).
bool is_gemm_weight(const ggml_tensor * t) {
    return t->type == GGML_TYPE_BF16 && ggml_n_dims(t) == 2 && t->view_src == nullptr && ggml_is_contiguous(t) &&
           t->ne[0] % KT == 0 && t->ne[1] % LSW_BM == 0 &&
           (strstr(t->name, "weight") != nullptr || getenv("GGML_VORTEX_ANY_WEIGHT") != nullptr);
}

constexpr uint32_t NT = WARP_NT;
// Values per act_pack work item (matches QRUN in e2e_dev.h).
constexpr uint32_t QRUN = IS_MX ? F::epw : (F::epw > 4 ? F::epw : 4);

// The SIMT kernels grid-stride; a few CTAs fill the core, more only add CTA dispatch.
uint32_t grid_for(uint64_t threads) {
    uint64_t g = (threads + CTA_THREADS - 1) / CTA_THREADS;
    static const uint64_t cap = getenv("GGML_VORTEX_GRID") ? atoi(getenv("GGML_VORTEX_GRID")) : 4;
    return (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(g, cap));
}

// ---------------------------------------------------------------------------
// Graph recording
// ---------------------------------------------------------------------------

void check_mul_mat(vx_device_ctx & d, const ggml_tensor * dst);

struct vx_recorder {
    vx_device_ctx &     d;
    std::vector<vx_rec> recs;

    template <typename A>
    void launch(const char * name, const A & args, uint32_t grid, uint32_t lmem = 0, uint32_t gy = 1) {
        vx_rec r = {};
        r.name = name;
        r.args.resize(sizeof(A));
        memcpy(r.args.data(), &args, sizeof(A));
        r.li.struct_size  = sizeof(r.li);
        r.li.kernel       = d.kernels.at(name);
        r.li.args_size    = sizeof(A);
        r.li.ndim         = gy > 1 ? 2 : 1;
        r.li.grid_dim[0]  = grid;
        r.li.grid_dim[1]  = gy;
        r.li.grid_dim[2]  = 1;
        r.li.block_dim[0] = CTA_THREADS;
        r.li.block_dim[1] = 1;
        r.li.block_dim[2] = 1;
        r.li.lmem_size    = lmem;
        recs.push_back(std::move(r));
    }

    void submit() {
        if (recs.empty()) {
            return;
        }
        const auto t0 = clk::now();
        for (auto & r : recs) {
            r.li.args_host = r.args.data();
        }
        auto emit = [](std::vector<vx_command_t> & cmds, vx_rec & r) {
            for (auto & [a, v] : r.dcrs) {
                vx_command_t c = {};
                c.type = VX_COMMAND_DCR_WRITE;
                c.data.dcr.addr = a;
                c.data.dcr.value = v;
                cmds.push_back(c);
            }
            vx_command_t c = {};
            c.type = VX_COMMAND_LAUNCH;
            c.data.launch = &r.li;
            cmds.push_back(c);
        };
        if (d.profile) {
            for (auto & r : recs) {
                const auto t1 = clk::now();
                std::vector<vx_command_t> cmds;
                emit(cmds, r);
                vx_event_h ev = nullptr;
                VX_CHECK(vx_enqueue_commands(d.q, cmds.data(), (uint32_t)cmds.size(), 0, nullptr, &ev));
                d.wait(ev);
                auto & p = d.prof[r.name];
                p.first++;
                p.second += ms_since(t1);
                static const bool check = getenv("GGML_VORTEX_CHECK") != nullptr;
                if (check && r.node) {
                    check_mul_mat(d, r.node);
                }
                if (d.verbose) {
                    fprintf(stderr, "vortex: launch %-11s grid=%u,%u %8.2f ms\n", r.name, r.li.grid_dim[0], r.li.grid_dim[1], ms_since(t1));
                }
            }
        } else {
            // The runtime stages a whole batch into the 64 KiB CP ring without
            // backpressure (~14 CLs per launch, one per DCR write), so a batch must fit in it.
            static const size_t chunk = getenv("GGML_VORTEX_BATCH") ? atoi(getenv("GGML_VORTEX_BATCH")) : 32;
            for (size_t i = 0; i < recs.size(); i += chunk) {
                std::vector<vx_command_t> cmds;
                for (size_t j = i; j < std::min(recs.size(), i + chunk); ++j) {
                    emit(cmds, recs[j]);
                }
                vx_event_h ev = nullptr;
                VX_CHECK(vx_enqueue_commands(d.q, cmds.data(), (uint32_t)cmds.size(), 0, nullptr, &ev));
                d.wait(ev);
            }
        }
        d.t_submit_ms += ms_since(t0);
        d.n_launches += recs.size();
        recs.clear();
    }
};

// Scratch layout for one MUL_MAT's activation operand.
struct act_layout {
    uint64_t words, scales, row_ts, total;
};

act_layout act_layout_for(uint32_t T, uint32_t K) {
    const uint32_t Tp = (T + LSW_BN - 1) / LSW_BN * LSW_BN;
    act_layout l;
    l.words  = 0;
    l.scales = ((uint64_t)Tp * K / F::epw * 4 + 63) & ~63ull;
    l.row_ts = l.scales + (IS_MX ? (((uint64_t)Tp / TILE * (K / KT) * SC_TILE_BYTES + 63) & ~63ull) : 0);
    l.total  = l.row_ts + (uint64_t)Tp * 4 + 64;
    return l;
}

// linear_dec local memory: per-warp C tiles, then tt * TILE activation rows
// and their scale tiles (plus the one-past-the-end scale prefetch).
uint32_t dec_lmem(uint32_t nkt, uint32_t tt = 1) {
    return (CTA_NW * TILE * TILE + nkt * tt * TILE * KT_WORDS + (nkt + 1) * tt * SC_TILE_WORDS) * 4;
}

void rec_mul_mat(vx_recorder & R, const ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const auto * wi = (const vx_weight_info *)w->extra;
    const uint32_t K = (uint32_t)w->ne[0], N = (uint32_t)w->ne[1], T = (uint32_t)x->ne[1];
    const uint32_t Tp = (T + LSW_BN - 1) / LSW_BN * LSW_BN, nkt = K / KT;
    const act_layout L = act_layout_for(T, K);
    const uint64_t base = R.d.scr_addr;

    // The SIMT GEMV streams W once per VX_GEMV_TMAX activation rows; above
    // GGML_VORTEX_GEMV_T rows the TCU GEMM takes over.
    // linear_dec takes over wherever its activation tile fits in local memory:
    // about half the GEMV's instructions per weight byte. GGML_VORTEX_NO_GEMV_DEC
    // keeps the GEMV on every shape.
    static const uint32_t gemv_t = getenv("GGML_VORTEX_GEMV_T") ? atoi(getenv("GGML_VORTEX_GEMV_T")) : VX_GEMV_TMAX;
    static const bool gemv_dec = getenv("GGML_VORTEX_NO_GEMV_DEC") == nullptr;
    const bool dec_fits = NT == TILE && dec_lmem(nkt) <= (1u << VX_CFG_LMEM_LOG_SIZE);
    if (!IS_MX && !HAS_TSCALE && F::bits == 16 && T <= gemv_t && !(gemv_dec && dec_fits)) {
        for (uint32_t t0 = 0; t0 < T; t0 += VX_GEMV_TMAX) {
            vx_gemv_args_t g = {};
            g.w = dev_addr(w);
            g.out = dev_addr(dst) + (uint64_t)t0 * dst->nb[1];
            g.x = vt_of(x);
            g.x.addr += (uint64_t)t0 * x->nb[1];
            g.x.ne[1] = std::min<uint32_t>(VX_GEMV_TMAX, T - t0);
            g.N = N; g.K = K; g.nkt = nkt; g.rb = LSW_BM; g.T = g.x.ne[1]; g.ldo = N;
            R.launch("gemv", g, grid_for((uint64_t)N / 16 * NT));
        }
        R.recs.back().node = dst;
        return;
    }

    // linear_dec runs TILE activation rows per launch; up to GGML_VORTEX_DEC_T
    // rows it beats linear_swap, which re-stages the weights per LSW_BN rows.
    // It stores only real rows and MMA columns are independent, so padding
    // rows need not be packed.
    static const uint32_t dec_t = getenv("GGML_VORTEX_DEC_T") ? atoi(getenv("GGML_VORTEX_DEC_T")) : 64;
    const bool dec = NT == TILE && T <= dec_t && dec_lmem(nkt) <= (1u << VX_CFG_LMEM_LOG_SIZE);
    vx_act_args_t a = {};
    a.src    = vt_of(x);
    a.dst    = base + L.words;
    a.dst_sc = IS_MX ? base + L.scales : 0;
    a.row_ts = base + L.row_ts;
    a.T = T; a.Tp = dec ? T : Tp; a.K = K; a.rb = LSW_BN;
    a.runs = vx_div_make(K / QRUN);
    if (HAS_TSCALE) {
        R.launch("act_scale", a, grid_for((uint64_t)a.Tp * NT));
    }
    R.launch("act_pack", a, grid_for((uint64_t)a.Tp * (K / QRUN)));

    linear_args_t l = {};
    l.a = a.dst;
    l.a_sc = a.dst_sc;
    l.w = dev_addr(w);
    l.w_sc = wi->scales;
    l.out = dev_addr(dst);
    l.ts_a = 1.0f;
    l.ts_w = wi->ts;
    l.ts_out = 1.0f;
    l.row_scale = HAS_TSCALE ? a.row_ts : 0;
    l.M = T; l.N = N; l.K = K; l.num_kt = nkt;
    l.act = ACT_NONE;
    l.out_mode = OUT_FP32;
    l.ldo = N;
    if (dec) {
        // Two TILE-row tiles per launch stream the weights once per 2 * TILE rows.
        static const bool dec2_off = getenv("GGML_VORTEX_NO_DEC2") != nullptr;
        const bool fits2 = !dec2_off && 2 * TILE <= LSW_BN && dec_lmem(nkt, 2) <= (1u << VX_CFG_LMEM_LOG_SIZE);
        for (uint32_t t0 = 0, step; t0 < T; t0 += step) {
            const uint32_t tt = (fits2 && T - t0 > TILE) ? 2 : 1;
            step = tt * TILE;
            linear_args_t c = l;
            c.a += 4ull * ((t0 / LSW_BN) * nkt * LSW_BN * KT_WORDS + (t0 % LSW_BN) * KT_WORDS);
            if (IS_MX) {
                c.a_sc += (uint64_t)(t0 / LSW_BN) * nkt * (LSW_BN / TILE) * SC_TILE_BYTES + (t0 % LSW_BN) / TILE * SC_TILE_BYTES;
            }
            if (HAS_TSCALE) {
                c.row_scale += 4ull * t0;
            }
            c.out += (uint64_t)t0 * dst->nb[1];
            c.M = std::min<uint32_t>(step, T - t0);
            R.launch("linear_dec", c, 1, dec_lmem(nkt, tt));
        }
    } else {
        R.launch("linear_swap", l, N / LSW_BM, lin_swap_lmem(), Tp / LSW_BN);
    }
    R.recs.back().node = dst;
}

void rec_rope(vx_recorder & R, const ggml_tensor * dst) {
    const int32_t * op = (const int32_t *)dst->op_params;
    vx_rope_args_t a = {};
    a.src = vt_of(dst->src[0]);
    a.dst = vt_of(dst);
    a.pos = dev_addr(dst->src[1]);
    a.freq_factors = dst->src[2] ? dev_addr(dst->src[2]) : 0;
    a.n_dims = op[1];
    a.mode   = op[2];
    const int n_ctx_orig = op[4];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   op + 5,  sizeof(float));
    memcpy(&freq_scale,  op + 6,  sizeof(float));
    memcpy(&ext_factor,  op + 7,  sizeof(float));
    memcpy(&attn_factor, op + 8,  sizeof(float));
    memcpy(&beta_fast,   op + 9,  sizeof(float));
    memcpy(&beta_slow,   op + 10, sizeof(float));
    float corr[2];
    ggml_rope_yarn_corr_dims(a.n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr);
    // The device evaluates theta_scale^i as exp2(i * log2(theta_scale)).
    a.freq_base   = log2f(powf(freq_base, -2.0f / a.n_dims));
    a.freq_scale  = freq_scale;
    a.ext_factor  = ext_factor;
    a.attn_factor = ext_factor != 0.0f ? attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale)) : attn_factor;
    a.corr_lo = corr[0];
    a.corr_hi = corr[1];
    a.half = vx_div_make((uint32_t)dst->src[0]->ne[0] / 2);
    R.launch("rope", a, grid_for((uint64_t)dst->ne[0] / 2 * dst->ne[1] * dst->ne[2]));
}

void rec_flash_attn(vx_recorder & R, const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * m = dst->src[3];
    vx_fa_args_t a = {};
    a.q = vt_of(q); a.k = vt_of(k); a.v = vt_of(v); a.dst = vt_of(dst);
    a.mask = vt_of(m);
    a.has_mask = m != nullptr;
    a.gqa = vx_div_make((uint32_t)(q->ne[2] / k->ne[2]));
    a.k_f16 = k->type == GGML_TYPE_F16;
    a.v_f16 = v->type == GGML_TYPE_F16;
    a.mask_f16 = m && m->type == GGML_TYPE_F16;
    memcpy(&a.scale,   (const float *)dst->op_params + 0, sizeof(float));
    memcpy(&a.softcap, (const float *)dst->op_params + 2, sizeof(float));
    R.launch("flash_attn", a, grid_for((uint64_t)q->ne[1] * q->ne[2] * NT));
}

bool is_noop(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

// RMS_NORM immediately consumed by a MUL with a [ne0] f32 gamma: one kernel.
const ggml_tensor * fused_gamma(const ggml_cgraph * g, int i) {
    const ggml_tensor * n = g->nodes[i];
    if (i + 1 >= g->n_nodes) {
        return nullptr;
    }
    const ggml_tensor * m = g->nodes[i + 1];
    if (m->op != GGML_OP_MUL || m->src[0] != n || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return nullptr;
    }
    const ggml_tensor * gm = m->src[1];
    if (gm->type != GGML_TYPE_F32 || ggml_nelements(gm) != n->ne[0] || !ggml_is_contiguous(gm) || !ggml_are_same_shape(m, n)) {
        return nullptr;
    }
    for (int j = i + 2; j < g->n_nodes; ++j) {
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (g->nodes[j]->src[s] == n) {
                return nullptr;
            }
        }
    }
    return gm;
}

} // namespace

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------

struct ggml_backend_vortex_buffer_context {
    vx_buffer_h h    = nullptr;
    uint64_t    base = 0;
    uint64_t    size = 0;
    std::vector<vx_weight_info *> infos;

    ~ggml_backend_vortex_buffer_context() {
        for (auto * i : infos) {
            delete i;
        }
        if (h) {
            vx_buffer_release(h);
        }
    }
};

namespace {

// GGML_VORTEX_CHECK: recompute a finished MUL_MAT on the host and report its error.
void check_mul_mat(vx_device_ctx & d, const ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const auto * wi = (const vx_weight_info *)w->extra;
    const int64_t K = w->ne[0], N = w->ne[1], T = x->ne[1];
    auto rd = [&](const ggml_tensor * t, void * out, size_t n) {
        auto * c = (ggml_backend_vortex_buffer_context *)t->buffer->context;
        d.read(out, c->h, dev_addr(t) - c->base, n);
    };
    std::vector<float> xs(x->nb[1] / 4 * (T - 1) + K), ys(N * T);
    rd(x, xs.data(), xs.size() * 4);
    rd(dst, ys.data(), ys.size() * 4);
    const uint16_t * wh = (const uint16_t *)wi->host.data();
    {
        std::vector<uint32_t> wd(N * K / 2);
        rd(w, wd.data(), wd.size() * 4);
        int64_t bad = 0, first = -1;
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t kw = 0; kw < K / 2; ++kw) {
                const uint32_t want = wh[n * K + 2 * kw] | ((uint32_t)wh[n * K + 2 * kw + 1] << 16);
                if (wd[blk_word((uint32_t)n, (uint32_t)kw, LSW_BM, (uint32_t)(K / KT))] != want) {
                    if (first < 0) first = n * K / 2 + kw;
                    ++bad;
                }
            }
        }
        fprintf(stderr, "check %-24s weight words @0x%llx: %ld/%ld wrong (first word %ld)\n", w->name,
                (unsigned long long)dev_addr(w), (long)bad, (long)(N * K / 2), (long)first);
    }
    double emax = 0, rmax = 0;
    int64_t at_n = 0, at_t = 0;
    std::string per_t;
    for (int64_t t = 0; t < T; ++t) {
        double et = 0;
        const float * xr = xs.data() + t * (x->nb[1] / 4);
        for (int64_t n = 0; n < N; ++n) {
            double ref = 0;
            for (int64_t k = 0; k < K; ++k) ref += (double)bf16_to_f32(wh[n * K + k]) * bf16_to_f32(f32_to_bf16(xr[k]));
            const double e = fabs(ref - ys[t * N + n]);
            rmax = std::max(rmax, fabs(ref));
            et = std::max(et, e);
            if (e > emax) { emax = e; at_n = n; at_t = t; }
        }
        char b[32];
        snprintf(b, sizeof(b), " %.2g", et);
        per_t += b;
    }
    fprintf(stderr, "check %-24s K=%5ld N=%6ld T=%3ld  max|err|=%.4g (at n=%ld t=%ld) max|ref|=%.4g%s  per-t:%s\n", dst->name, (long)K, (long)N,
                  (long)T, emax, (long)at_n, (long)at_t, rmax, emax > 1e-2 * std::max(1.0, rmax) ? "  <-- BAD" : "", per_t.c_str());
}

} // namespace

static void ggml_backend_vortex_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    std::lock_guard<std::mutex> lock(vx_dev().mtx);
    delete (ggml_backend_vortex_buffer_context *)buffer->context;
}

static void * ggml_backend_vortex_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *)(uintptr_t)((ggml_backend_vortex_buffer_context *)buffer->context)->base;
}

static uint64_t buf_off(ggml_backend_buffer_t buffer, const ggml_tensor * t, size_t offset) {
    return dev_addr(t) + offset - ((ggml_backend_vortex_buffer_context *)buffer->context)->base;
}

static void ggml_backend_vortex_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * c = (ggml_backend_vortex_buffer_context *)buffer->context;
    std::lock_guard<std::mutex> lock(vx_dev().mtx);
    std::vector<uint8_t> v(size, value);
    vx_dev().write(c->h, buf_off(buffer, tensor, offset), v.data(), size);
}

// Weights go to the device in the TCU's blocked operand layout, quantized to
// ITYPE with a per-tensor scale; the scale tiles follow the words in place.
static void pack_weight(ggml_backend_buffer_t buffer, ggml_tensor * t, const void * data) {
    auto * c = (ggml_backend_vortex_buffer_context *)buffer->context;
    const uint32_t K = (uint32_t)t->ne[0], N = (uint32_t)t->ne[1], nkt = K / KT, rw = K / F::epw;
    const uint16_t * h = (const uint16_t *)data;
    float amax = 0.0f;
    if (HAS_TSCALE) {
#pragma omp parallel for reduction(max:amax)
        for (int64_t i = 0; i < (int64_t)N * K; ++i) {
            amax = std::max(amax, std::fabs(bf16_to_f32(h[i])));
        }
    }
    auto * wi = new vx_weight_info;
    c->infos.push_back(wi);
    wi->ts = tensor_scale_for(amax);
    if (getenv("GGML_VORTEX_ANY_WEIGHT") || getenv("GGML_VORTEX_CHECK")) {
        wi->host.assign((const uint8_t *)data, (const uint8_t *)data + ggml_nbytes(t));
    }
    std::vector<uint32_t> words((size_t)N * rw);
    std::vector<uint8_t>  scales(IS_MX ? (size_t)N / TILE * nkt * SC_TILE_BYTES : 0);
#pragma omp parallel
    {
        std::vector<float>    x(K);
        std::vector<uint32_t> w(rw);
        std::vector<uint8_t>  sc(K / BLK + 1);
#pragma omp for schedule(static)
        for (int64_t r = 0; r < (int64_t)N; ++r) {
            for (uint32_t k = 0; k < K; ++k) {
                x[k] = bf16_to_f32(h[(size_t)r * K + k]);
            }
            quantize_run(x.data(), K, wi->ts, w.data(), sc.data());
            for (uint32_t kw = 0; kw < rw; ++kw) {
                words[blk_word((uint32_t)r, kw, LSW_BM, nkt)] = w[kw];
            }
            if (IS_MX) {
                for (uint32_t b = 0; b < K / BLK; ++b) {
                    scales[blk_sc((uint32_t)r, b * BLK, LSW_BM, nkt)] = sc[b];
                }
            }
        }
    }
    const uint64_t off = buf_off(buffer, t, 0);
    GGML_ASSERT(words.size() * 4 + scales.size() <= ggml_nbytes(t) + VX_SLACK / 2);
    vx_dev().write(c->h, off, words.data(), words.size() * 4);
    if (IS_MX) {
        const uint64_t so = (off + words.size() * 4 + 63) & ~63ull;
        vx_dev().write(c->h, so, scales.data(), scales.size());
        wi->scales = c->base + so;
    }
    t->extra = wi;
}

static void ggml_backend_vortex_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * c = (ggml_backend_vortex_buffer_context *)buffer->context;
    std::lock_guard<std::mutex> lock(vx_dev().mtx);
    vx_dev().open();
    if (is_gemm_weight(tensor)) {
        GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor) && "GEMM weights are uploaded whole");
        pack_weight(buffer, tensor, data);
        return;
    }
    vx_dev().write(c->h, buf_off(buffer, tensor, offset), data, size);
}

static void ggml_backend_vortex_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * c = (ggml_backend_vortex_buffer_context *)buffer->context;
    if (tensor->extra) {
        const auto & h = ((const vx_weight_info *)tensor->extra)->host;
        GGML_ASSERT(offset + size <= h.size() && "blocked GEMM weights cannot be read back");
        memcpy(data, h.data() + offset, size);
        return;
    }
    std::lock_guard<std::mutex> lock(vx_dev().mtx);
    vx_dev().read(data, c->h, buf_off(buffer, tensor, offset), size);
}

static bool ggml_backend_vortex_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (src->buffer && src->buffer->buft == buffer->buft && src->extra == nullptr) {
        auto * dc = (ggml_backend_vortex_buffer_context *)buffer->context;
        auto * sc = (ggml_backend_vortex_buffer_context *)src->buffer->context;
        std::lock_guard<std::mutex> lock(vx_dev().mtx);
        vx_event_h ev = nullptr;
        VX_CHECK(vx_enqueue_copy(vx_dev().q, dc->h, buf_off(buffer, dst, 0), sc->h, buf_off(src->buffer, src, 0),
                                 ggml_nbytes(src), 0, nullptr, &ev));
        vx_dev().wait(ev);
        return true;
    }
    return false;
}

static void ggml_backend_vortex_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * c = (ggml_backend_vortex_buffer_context *)buffer->context;
    std::lock_guard<std::mutex> lock(vx_dev().mtx);
    const uint32_t pattern = value * 0x01010101u;
    vx_event_h ev = nullptr;
    VX_CHECK(vx_enqueue_fill_buffer(vx_dev().q, c->h, 0, c->size & ~3ull, &pattern, 4, 0, nullptr, &ev));
    vx_dev().wait(ev);
}

static const ggml_backend_buffer_i ggml_backend_vortex_buffer_i = {
    /* .free_buffer     = */ ggml_backend_vortex_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vortex_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_vortex_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vortex_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vortex_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_vortex_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vortex_buffer_clear,
    /* .reset           = */ NULL,
};

// ---------------------------------------------------------------------------
// Buffer type
// ---------------------------------------------------------------------------

static const char * ggml_backend_vortex_buft_get_name(ggml_backend_buffer_type_t buft) {
    return "Vortex";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_vortex_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    std::lock_guard<std::mutex> lock(vx_dev().mtx);
    vx_dev().open();
    auto * c = new ggml_backend_vortex_buffer_context;
    c->size = std::max<size_t>(size, 64);
    vx_dev().alloc(c->size, &c->h, &c->base);
    return ggml_backend_buffer_init(buft, ggml_backend_vortex_buffer_i, c, size);
}

static size_t ggml_backend_vortex_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_vortex_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    return (size_t)1 << 31;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_vortex_buft_i = {
    /* .get_name       = */ ggml_backend_vortex_buft_get_name,
    /* .alloc_buffer     = */ ggml_backend_vortex_buft_alloc_buffer,
    /* .alloc_buffer_n   = */ NULL,
    /* .get_alignment    = */ ggml_backend_vortex_buft_get_alignment,
    /* .get_max_size     = */ ggml_backend_vortex_buft_get_max_size,
    /* .get_alloc_size   = */ NULL,
    /* .get_alloc_size_n = */ NULL,
    /* .is_host          = */ NULL,
};

static ggml_backend_buffer_type_t ggml_backend_vortex_buffer_type(void) {
    static ggml_backend_buffer_type buft = {
        /* .iface   = */ ggml_backend_vortex_buft_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_vortex_reg(), 0),
        /* .context = */ nullptr,
    };
    return &buft;
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

static const char * ggml_backend_vortex_get_name(ggml_backend_t backend) {
    return "Vortex";

    GGML_UNUSED(backend);
}

static void ggml_backend_vortex_free(ggml_backend_t backend) {
    vx_dev().print_stats();
    delete backend;
}

static enum ggml_status ggml_backend_vortex_graph_compute(ggml_backend_t backend, ggml_cgraph * g) {
    vx_device_ctx & d = vx_dev();
    std::lock_guard<std::mutex> lock(d.mtx);
    d.open();

    // The MUL_MATs run back to back and share one activation scratch.
    uint64_t scratch = 0;
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor * n = g->nodes[i];
        if (n->op == GGML_OP_MUL_MAT) {
            scratch = std::max(scratch, act_layout_for((uint32_t)n->src[1]->ne[1], (uint32_t)n->src[0]->ne[0]).total);
        }
    }
    d.ensure_scratch(scratch);

    vx_recorder R{d, {}};
    for (int i = 0; i < g->n_nodes; ++i) {
        ggml_tensor * n = g->nodes[i];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || is_noop(n)) {
            continue;
        }
        switch (n->op) {
            case GGML_OP_MUL_MAT:
                rec_mul_mat(R, n);
                break;
            case GGML_OP_RMS_NORM: {
                vx_rms_args_t a = {};
                a.src = vt_of(n->src[0]);
                memcpy(&a.eps, n->op_params, sizeof(float));
                const ggml_tensor * gm = d.fuse ? fused_gamma(g, i) : nullptr;
                a.dst = vt_of(gm ? g->nodes[i + 1] : n);
                a.gamma = gm ? dev_addr(gm) : 0;
                R.launch("rms_norm", a, grid_for((uint64_t)ggml_nrows(n) * NT));
                if (gm) {
                    ++i;
                }
                break;
            }
            case GGML_OP_ADD:
            case GGML_OP_MUL: {
                vx_bin_args_t a = {};
                a.a = vt_of(n->src[0]);
                a.b = vt_of(n->src[1]);
                a.dst = vt_of(n);
                a.op = n->op == GGML_OP_ADD ? VX_BIN_ADD : VX_BIN_MUL;
                if (ggml_is_contiguous(n) && ggml_is_contiguous(n->src[0]) && ggml_is_contiguous(n->src[1])) {
                    a.bcast = ggml_are_same_shape(n, n->src[1])            ? VX_BCAST_NONE
                            : ggml_nelements(n->src[1]) == n->ne[0]        ? VX_BCAST_ROW
                                                                           : VX_BCAST_GENERAL;
                }
                R.launch("binary", a, grid_for(ggml_nelements(n)));
                break;
            }
            case GGML_OP_GLU: {
                vx_glu_args_t a = {};
                a.gate = vt_of(n->src[0]);
                a.up = vt_of(n->src[1]);
                a.dst = vt_of(n);
                R.launch("glu_swiglu", a, grid_for(ggml_nelements(n)));
                break;
            }
            case GGML_OP_ROPE:
                rec_rope(R, n);
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                rec_flash_attn(R, n);
                break;
            case GGML_OP_SET_ROWS: {
                vx_set_rows_args_t a = {};
                a.src = vt_of(n->src[0]);
                a.idx = vt_of(n->src[1]);
                a.dst = vt_of(n);
                a.idx_i64 = n->src[1]->type == GGML_TYPE_I64;
                a.dst_f16 = n->type == GGML_TYPE_F16;
                R.launch("set_rows", a, grid_for(ggml_nelements(n->src[0])));
                break;
            }
            case GGML_OP_GET_ROWS: {
                vx_get_rows_args_t a = {};
                a.src = vt_of(n->src[0]);
                a.idx = vt_of(n->src[1]);
                a.dst = vt_of(n);
                a.idx_i64 = n->src[1]->type == GGML_TYPE_I64;
                R.launch("get_rows", a, grid_for(ggml_nelements(n)));
                break;
            }
            case GGML_OP_CPY:
            case GGML_OP_DUP:
            case GGML_OP_CONT: {
                vx_cpy_args_t a = {};
                a.src = vt_of(n->src[0]);
                a.dst = vt_of(n->op == GGML_OP_CPY ? n->src[1] : n);
                a.src_f16 = n->src[0]->type == GGML_TYPE_F16;
                a.dst_f16 = n->type == GGML_TYPE_F16;
                R.launch("cpy", a, grid_for(ggml_nelements(n->src[0])));
                break;
            }
            default:
                GGML_ABORT("%s: unsupported op %s", __func__, ggml_op_desc(n));
        }
    }
    R.submit();
    d.n_graphs++;
    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static const ggml_backend_i vortex_backend_i = {
    /* .get_name                = */ ggml_backend_vortex_get_name,
    /* .free                    = */ ggml_backend_vortex_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_vortex_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_vortex_guid(void) {
    static ggml_guid guid = { 0x56, 0x6f, 0x72, 0x74, 0x65, 0x78, 0x47, 0x50, 0x47, 0x50, 0x55, 0x2d, 0x74, 0x63, 0x75, 0x02 };
    return &guid;
}

ggml_backend_t ggml_backend_vortex_init(void) {
    return new ggml_backend {
        /* .guid    = */ ggml_backend_vortex_guid(),
        /* .iface   = */ vortex_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_vortex_reg(), 0),
        /* .context = */ nullptr,
    };
}

bool ggml_backend_is_vortex(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_vortex_guid());
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

static const char * ggml_backend_vortex_device_get_name(ggml_backend_dev_t dev) {
    return "Vortex";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_vortex_device_get_description(ggml_backend_dev_t dev) {
    return "Vortex GPGPU (U55C)";

    GGML_UNUSED(dev);
}

static void ggml_backend_vortex_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // 32-bit device address space, less the stack/IO windows.
    *total = (size_t)3800 << 20;
    *free  = *total;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_vortex_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_vortex_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_vortex_device_get_name(dev);
    props->description = ggml_backend_vortex_device_get_description(dev);
    props->type        = ggml_backend_vortex_device_get_type(dev);
    ggml_backend_vortex_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
}

static ggml_backend_t ggml_backend_vortex_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_vortex_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_vortex_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_vortex_buffer_type();

    GGML_UNUSED(dev);
}

static bool f32_or_f16(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16;
}

static bool ggml_backend_vortex_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    static const char * disabled = getenv("GGML_VORTEX_DISABLE");   // comma list of op names kept off the device
    if (disabled && !is_noop(op)) {
        const std::string list = std::string(",") + disabled + ",";
        if (list.find(std::string(",") + ggml_op_name(op->op) + ",") != std::string::npos) {
            return false;
        }
    }
    const ggml_tensor * s0 = op->src[0];
    const ggml_tensor * s1 = op->src[1];
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT:
            return is_gemm_weight(s0) && s1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   s1->ne[2] == 1 && s1->ne[3] == 1 && s1->nb[0] == 4 && ggml_is_contiguous(op);
        case GGML_OP_RMS_NORM:
            return s0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && s0->nb[0] == 4 && op->nb[0] == 4;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return op->src[2] == nullptr && s0->type == GGML_TYPE_F32 && s1->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 && ggml_can_repeat(s1, s0) && ggml_are_same_shape(op, s0);
        case GGML_OP_GLU:
            return ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && s1 != nullptr && ggml_get_op_params_i32(op, 1) == 0 &&
                   s0->type == GGML_TYPE_F32 && s1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_are_same_shape(s0, s1);
        case GGML_OP_ROPE: {
            const int mode = ggml_get_op_params_i32(op, 2);
            return (mode == 0 || mode == 2) && s0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   s0->ne[3] == 1 && s0->nb[0] == 4 && op->nb[0] == 4 && s0->ne[0] % 2 == 0 &&
                   (op->src[2] == nullptr || op->src[2]->type == GGML_TYPE_F32);
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            float max_bias;
            memcpy(&max_bias, (const float *)op->op_params + 1, sizeof(float));
            const ggml_tensor * k = op->src[1];
            const ggml_tensor * v = op->src[2];
            const ggml_tensor * m = op->src[3];
            return op->src[4] == nullptr && max_bias == 0.0f && s0->type == GGML_TYPE_F32 &&
                   f32_or_f16(k) && f32_or_f16(v) && (m == nullptr || f32_or_f16(m)) &&
                   s0->ne[0] % NT == 0 && s0->ne[0] <= 256 && s0->ne[0] == v->ne[0] &&
                   s0->ne[3] == 1 && k->ne[3] == 1 && s0->ne[2] % k->ne[2] == 0 && op->type == GGML_TYPE_F32;
        }
        case GGML_OP_SET_ROWS:
            return s0->type == GGML_TYPE_F32 && (s1->type == GGML_TYPE_I64 || s1->type == GGML_TYPE_I32) && f32_or_f16(op);
        case GGML_OP_GET_ROWS:
            return s0->type == GGML_TYPE_F32 && (s1->type == GGML_TYPE_I64 || s1->type == GGML_TYPE_I32) &&
                   op->type == GGML_TYPE_F32;
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            return f32_or_f16(s0) && f32_or_f16(op);
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_vortex_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_vortex_buffer_type();

    GGML_UNUSED(dev);
}

static bool ggml_backend_vortex_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return false;

    GGML_UNUSED(dev);
    GGML_UNUSED(op);
}

static const ggml_backend_device_i ggml_backend_vortex_device_i = {
    /* .get_name             = */ ggml_backend_vortex_device_get_name,
    /* .get_description      = */ ggml_backend_vortex_device_get_description,
    /* .get_memory           = */ ggml_backend_vortex_device_get_memory,
    /* .get_type             = */ ggml_backend_vortex_device_get_type,
    /* .get_props            = */ ggml_backend_vortex_device_get_props,
    /* .init_backend         = */ ggml_backend_vortex_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_vortex_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_vortex_device_supports_op,
    /* .supports_buft        = */ ggml_backend_vortex_device_supports_buft,
    /* .offload_op           = */ ggml_backend_vortex_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

static const char * ggml_backend_vortex_reg_get_name(ggml_backend_reg_t reg) {
    return "Vortex";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_vortex_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_vortex_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device ggml_backend_vortex_device = {
        /* .iface   = */ ggml_backend_vortex_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &ggml_backend_vortex_device;

    GGML_UNUSED(index);
}

static void * ggml_backend_vortex_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const ggml_backend_reg_i ggml_backend_vortex_reg_i = {
    /* .get_name         = */ ggml_backend_vortex_reg_get_name,
    /* .get_device_count = */ ggml_backend_vortex_reg_get_device_count,
    /* .get_device       = */ ggml_backend_vortex_reg_get_device,
    /* .get_proc_address = */ ggml_backend_vortex_get_proc_address,
};

ggml_backend_reg_t ggml_backend_vortex_reg(void) {
    static ggml_backend_reg ggml_backend_vortex_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_vortex_reg_i,
        /* .context     = */ NULL,
    };
    return &ggml_backend_vortex_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_vortex_reg)
