// Vortex emulation backend: runs the MUL_MATs a Vortex TCU would run, with the
// TCU's numerics (both operands quantized to ITYPE exactly as the device
// kernels do, fp32 accumulate), on the host CPU, and paces each graph to the
// time a projected Vortex configuration would take. Everything else stays on
// the CPU backend.
//
// Projection (per MUL_MAT, compute- or bandwidth-bound, whichever is slower):
//   compute   = N*K*ceil(T/TPAD)*TPAD / (CORES * MACPC * (16/bits) * MHZ * EFF)
//   bandwidth = weight bytes / (CORES * 64 B * MHZ * BWEFF)
// TPAD is the TCU tile's token dimension (decode pads one token up to it) and
// MACPC the per-core bf16 MAC/cycle; 4-bit formats pack 4x the MACs per word.
// Every knob is a GGML_VORTEX_EMU_* environment variable; GGML_VORTEX_EMU_PACE=0
// runs unpaced.

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-vortex.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "e2e_fmt.h"

using namespace e2e;

namespace {

using clk = std::chrono::steady_clock;

double env_f(const char * name, double def) {
    const char * v = getenv(name);
    return v ? atof(v) : def;
}

float bf16_to_f32(uint16_t h) {
    return u2f((uint32_t)h << 16);
}

constexpr bool FMT4 = F::bits == 4;
constexpr bool FMT_BF16 = std::is_same<IT, vt::bf16>::value;

// A weight in the device format, stored compactly for the host GEMV:
// 4-bit formats as int8 q = 2*e2m1 plus one fp32 S/2 per scale block (value =
// q * S/2); bf16 as bf16; anything else as dequantized fp32.
struct emu_weight {
    std::vector<int8_t>   q;
    std::vector<float>    s;
    std::vector<uint16_t> h;
    std::vector<float>    f;
    double                bytes = 0;   // device bytes (words + scales)
};

struct emu_cfg {
    double cores, macpc, mhz, eff, bw_gbs, bweff, tpad, launch_us, sleep_min_s;
    bool   pace;

    void load() {
        cores     = env_f("GGML_VORTEX_EMU_CORES", 4);
        macpc     = env_f("GGML_VORTEX_EMU_MACPC", 135);
        mhz       = env_f("GGML_VORTEX_EMU_MHZ", 200);
        eff       = env_f("GGML_VORTEX_EMU_EFF", 0.6);
        bw_gbs    = env_f("GGML_VORTEX_EMU_BW", cores * 64 * mhz * 1e6 / 1e9);
        bweff     = env_f("GGML_VORTEX_EMU_BWEFF", 0.6);
        tpad      = env_f("GGML_VORTEX_EMU_TPAD", 16);
        launch_us = env_f("GGML_VORTEX_EMU_LAUNCH_US", 10);
        pace      = env_f("GGML_VORTEX_EMU_PACE", 1) != 0;
        sleep_min_s = env_f("GGML_VORTEX_EMU_SLEEP_MS", 25) * 1e-3;
    }

    double mul_mat_s(double N, double K, double T, double wbytes) const {
        const double tp = std::ceil(T / tpad) * tpad;
        const double fmtx = 16.0 / std::min<double>(16, F::bits);
        const double t_c = N * K * tp / (cores * macpc * fmtx * mhz * 1e6 * eff);
        const double t_m = wbytes / (bw_gbs * 1e9 * bweff);
        return std::max(t_c, t_m) + launch_us * 1e-6;
    }
};

struct emu_ctx {
    std::mutex mtx;
    bool       ready = false;
    emu_cfg    cfg;
    std::unordered_map<const void *, emu_weight> weights;

    uint64_t n_mul_mat = 0;
    double   t_proj_s = 0, t_host_s = 0, t_sleep_s = 0;
    // When the projected device finishes the work submitted so far. Sleeping
    // only once it runs ahead by sleep_min_s keeps the many short graph splits
    // of a token from each paying a wake-up.
    clk::time_point dev_done{};

    void open() {
        if (ready) {
            return;
        }
        cfg.load();
        fprintf(stderr, "%s: Vortex emulation (%s): projecting %g cores x %g bf16 MAC/cycle @ %g MHz (eff %.2f), "
                      "%.1f GB/s (eff %.2f), token tile %g%s\n", __func__, F::name, cfg.cores, cfg.macpc, cfg.mhz,
                      cfg.eff, cfg.bw_gbs, cfg.bweff, cfg.tpad, cfg.pace ? "" : ", unpaced");
        ready = true;
    }

    const emu_weight & weight(const ggml_tensor * w) {
        auto it = weights.find(w->data);
        if (it != weights.end()) {
            return it->second;
        }
        const int64_t K = w->ne[0], N = w->ne[1];
        const uint16_t * src = (const uint16_t *)w->data;
        float amax = 0.0f;
        if (HAS_TSCALE) {
#pragma omp parallel for reduction(max:amax)
            for (int64_t i = 0; i < N * K; ++i) {
                amax = std::max(amax, std::fabs(bf16_to_f32(src[i])));
            }
        }
        const float ts = tensor_scale_for(amax);
        const uint32_t mask = F::bits == 32 ? 0xffffffffu : ((1u << F::bits) - 1);
        emu_weight ew;
        if (FMT4) {
            ew.q.resize(N * K);
            ew.s.resize(N * K / BLK);
        } else if (FMT_BF16) {
            ew.h.resize(N * K);
        } else {
            ew.f.resize(N * K);
        }
#pragma omp parallel
        {
            std::vector<float>    x(K);
            std::vector<uint32_t> wd(K / F::epw);
            std::vector<uint8_t>  sc(K / BLK + 1);
#pragma omp for schedule(static)
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t k = 0; k < K; ++k) {
                    x[k] = bf16_to_f32(src[n * K + k]);
                }
                quantize_run(x.data(), (uint32_t)K, ts, wd.data(), sc.data());
                for (int64_t k = 0; k < K; ++k) {
                    const uint32_t code = (wd[k / F::epw] >> ((k % F::epw) * F::bits)) & mask;
                    const uint32_t sf = IS_MX ? sc[k / BLK] : 0;
                    if (FMT4) {
                        if (k % BLK == 0) {
                            // e2m1 code 2 is 1.0, so this is the block's value scale.
                            ew.s[(n * K + k) / BLK] = 0.5f * decode(2, sf, ts);
                        }
                        const float v = e2m1_to_f32(code & 0x7u) * 2.0f;
                        ew.q[n * K + k] = (int8_t)((code & 0x8u) ? -v : v);
                    } else if (FMT_BF16) {
                        ew.h[n * K + k] = (uint16_t)code;
                    } else {
                        ew.f[n * K + k] = decode(code, sf, ts);
                    }
                }
            }
        }
        ew.bytes = (double)N * K * F::bits / 8 + (IS_MX ? (double)N * K / BLK : 0);
        return weights.emplace(w->data, std::move(ew)).first->second;
    }

    // dst[T][N] = q(src1)[T][K] . q(src0)[N][K]^T; returns the projected device seconds
    double mul_mat(ggml_tensor * dst) {
        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];
        const int64_t K = src0->ne[0], N = src0->ne[1], T = src1->ne[1];
        const emu_weight & ew = weight(src0);

        // Activations: quantized per row with their own tensor scale, then dequantized.
        std::vector<float> xa(T * K);
        const uint32_t mask = F::bits == 32 ? 0xffffffffu : ((1u << F::bits) - 1);
        for (int64_t t = 0; t < T; ++t) {
            const float * x = (const float *)((const char *)src1->data + t * src1->nb[1]);
            float amax = 0.0f;
            if (HAS_TSCALE) {
                for (int64_t k = 0; k < K; ++k) {
                    amax = std::max(amax, std::fabs(x[k]));
                }
            }
            const float ts = tensor_scale_for(amax);
            std::vector<uint32_t> wd(K / F::epw);
            std::vector<uint8_t>  sc(K / BLK + 1);
            quantize_run(x, (uint32_t)K, ts, wd.data(), sc.data());
            for (int64_t k = 0; k < K; ++k) {
                xa[t * K + k] = decode((wd[k / F::epw] >> ((k % F::epw) * F::bits)) & mask, IS_MX ? sc[k / BLK] : 0, ts);
            }
        }

#pragma omp parallel
        {
            std::vector<float> wr(K);
#pragma omp for schedule(static)
            for (int64_t n = 0; n < N; ++n) {
                if (FMT4) {
                    const int8_t * q = ew.q.data() + n * K;
                    const float *  s = ew.s.data() + n * K / BLK;
                    for (int64_t b = 0; b < K / BLK; ++b) {
                        const float sb = s[b];
#pragma omp simd
                        for (int64_t i = 0; i < BLK; ++i) {
                            wr[b * BLK + i] = (float)q[b * BLK + i] * sb;
                        }
                    }
                } else if (FMT_BF16) {
                    const uint16_t * h = ew.h.data() + n * K;
#pragma omp simd
                    for (int64_t k = 0; k < K; ++k) {
                        wr[k] = u2f((uint32_t)h[k] << 16);
                    }
                } else {
                    std::memcpy(wr.data(), ew.f.data() + n * K, K * 4);
                }
                for (int64_t t = 0; t < T; ++t) {
                    const float * xr = xa.data() + t * K;
                    float acc = 0.0f;
#pragma omp simd reduction(+:acc)
                    for (int64_t k = 0; k < K; ++k) {
                        acc += wr[k] * xr[k];
                    }
                    ((float *)((char *)dst->data + t * dst->nb[1]))[n] = acc;
                }
            }
        }
        n_mul_mat++;
        const double t = cfg.mul_mat_s((double)N, (double)K, (double)T, ew.bytes);
        if (getenv("GGML_VORTEX_EMU_TRACE")) {
            fprintf(stderr, "emu %-28s N=%6ld K=%5ld T=%3ld  %.3f ms\n", src0->name, (long)N, (long)K, (long)T, t * 1e3);
        }
        return t;
    }

    void print_stats() {
        if (!ready) {
            return;
        }
        fprintf(stderr, "%s: Vortex emulation (%s): %lu MUL_MATs, projected device time %.2f s, host compute %.2f s, "
                      "paced %.2f s\n", __func__, F::name, (unsigned long)n_mul_mat, t_proj_s, t_host_s, t_sleep_s);
    }
};

emu_ctx & emu() {
    static emu_ctx ctx;
    return ctx;
}

bool emu_can_mul_mat(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(op) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           src0->ne[0] % 32 == 0 && strstr(src0->name, "weight") != nullptr;
}

} // namespace

static const char * ggml_backend_vortex_get_name(ggml_backend_t backend) {
    return "Vortex-emu";

    GGML_UNUSED(backend);
}

static void ggml_backend_vortex_free(ggml_backend_t backend) {
    emu().print_stats();
    delete backend;
}

static enum ggml_status ggml_backend_vortex_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    emu_ctx & e = emu();
    std::lock_guard<std::mutex> lock(e.mtx);
    e.open();
    const auto t0 = clk::now();
    double proj = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                proj += e.mul_mat(node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                GGML_ABORT("%s: unsupported op %s", __func__, ggml_op_desc(node));
        }
    }
    const double host = std::chrono::duration<double>(clk::now() - t0).count();
    e.t_proj_s += proj;
    e.t_host_s += host;
    if (e.cfg.pace) {
        if (e.dev_done < t0) {
            e.dev_done = t0;
        }
        e.dev_done += std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(proj));
        const auto now = clk::now();
        const double ahead = std::chrono::duration<double>(e.dev_done - now).count();
        if (ahead > e.cfg.sleep_min_s) {
            std::this_thread::sleep_until(e.dev_done);
            e.t_sleep_s += ahead;
        }
    }
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
    static ggml_guid guid = { 0x56, 0x6f, 0x72, 0x74, 0x65, 0x78, 0x47, 0x50, 0x47, 0x50, 0x55, 0x2d, 0x65, 0x6d, 0x75, 0x01 };
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

static const char * ggml_backend_vortex_device_get_name(ggml_backend_dev_t dev) {
    return "Vortex-emu";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_vortex_device_get_description(ggml_backend_dev_t dev) {
    return "Vortex GPGPU TCU (host emulation, paced to a projected configuration)";

    GGML_UNUSED(dev);
}

static void ggml_backend_vortex_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_vortex_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_vortex_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_vortex_device_get_name(dev);
    props->description = ggml_backend_vortex_device_get_description(dev);
    props->type        = ggml_backend_vortex_device_get_type(dev);
    ggml_backend_vortex_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_vortex_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_vortex_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_vortex_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_vortex_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_vortex_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT:
            return emu_can_mul_mat(op);
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_vortex_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_vortex_device_i = {
    /* .get_name             = */ ggml_backend_vortex_device_get_name,
    /* .get_description      = */ ggml_backend_vortex_device_get_description,
    /* .get_memory           = */ ggml_backend_vortex_device_get_memory,
    /* .get_type             = */ ggml_backend_vortex_device_get_type,
    /* .get_props            = */ ggml_backend_vortex_device_get_props,
    /* .init_backend         = */ ggml_backend_vortex_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_vortex_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_vortex_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_vortex_device_supports_op,
    /* .supports_buft        = */ ggml_backend_vortex_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static const char * ggml_backend_vortex_reg_get_name(ggml_backend_reg_t reg) {
    return "Vortex-emu";

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

static const struct ggml_backend_reg_i ggml_backend_vortex_reg_i = {
    /* .get_name         = */ ggml_backend_vortex_reg_get_name,
    /* .get_device_count = */ ggml_backend_vortex_reg_get_device_count,
    /* .get_device       = */ ggml_backend_vortex_reg_get_device,
    /* .get_proc_address = */ ggml_backend_vortex_get_proc_address,
};

ggml_backend_reg_t ggml_backend_vortex_reg(void) {
    static struct ggml_backend_reg ggml_backend_vortex_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_vortex_reg_i,
        /* .context     = */ NULL,
    };
    return &ggml_backend_vortex_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_vortex_reg)
