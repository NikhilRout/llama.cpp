set(GGML_VORTEX_HOME   "$ENV{HOME}/vortex" CACHE PATH   "Vortex source tree (runtime, kernel library, toolchain config)")
option(GGML_VORTEX_EMU "Build the host emulation backend (Vortex TCU numerics, paced to a projected configuration) instead of the device backend" OFF)
set(GGML_VORTEX_TARGET "H"                 CACHE STRING "Vortex bitstream the backend is built for: H, L1, L2, J or W8")
set(GGML_VORTEX_ITYPE  "bf16"              CACHE STRING "TCU operand format (bf16, fp16, tf32, fp8, mxfp8, mxfp4, nvfp4, lnsf4)")

# Hardware configurations of the U55C llama bitstreams; the kernel and the host
# tiling must be compiled for exactly the configuration the bitstream was built with.
set(_VX_FMTS "-DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_NVFP4_ENABLE -DVX_CFG_TCU_LNSF4_ENABLE -DVX_CFG_TCU_LNSF4_ANTILOG_BITS=6")
set(_VX_BASE "-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_USE_DSP=1 -DVX_CFG_TCU_LATENCY=5")
if (GGML_VORTEX_TARGET STREQUAL "H")
    set(_VX_NT 16)
    set(_VX_LMEM_LOG 16)
    set(_VX_HW "-DVX_CFG_NUM_THREADS=16 ${_VX_BASE} -DVX_CFG_LMEM_LOG_SIZE=16 -DVX_CFG_NUM_DXA_CORES=2 -DVX_CFG_DCACHE_NUM_BANKS=2 ${_VX_FMTS}")
elseif (GGML_VORTEX_TARGET STREQUAL "L1")
    set(_VX_NT 8)
    set(_VX_LMEM_LOG 14)
    set(_VX_HW "-DVX_CFG_NUM_THREADS=8 ${_VX_BASE} -DVX_CFG_TCU_FEDP2K -DVX_CFG_LMEM_LOG_SIZE=14 -DVX_CFG_NUM_DXA_CORES=1 ${_VX_FMTS}")
elseif (GGML_VORTEX_TARGET STREQUAL "W8")
    # llama_wmma_nt8_nw8_iw1_lmem64 (200 MHz): WMMA-only TCU, FP16/BF16/NVFP4/LNSF4, 64 KB LMEM
    set(_VX_NT 8)
    set(_VX_NW 8)
    set(_VX_LMEM_LOG 16)
    set(_VX_HW "-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_ISSUE_WIDTH=1 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_USE_DSP=1 -DVX_CFG_LMEM_LOG_SIZE=16 -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_NVFP4_ENABLE -DVX_CFG_TCU_LNSF4_ENABLE -DVX_CFG_TCU_LNSF4_ANTILOG_BITS=6")
elseif (GGML_VORTEX_TARGET STREQUAL "J")
    # Jacob's wgdxa1c8t16w bitstream (FP16/BF16 only, default LMEM)
    set(_VX_NT 8)
    set(_VX_LMEM_LOG 14)
    set(_VX_HW "-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_FEDP2K -DVX_CFG_TCU_USE_DSP=1")
elseif (GGML_VORTEX_TARGET STREQUAL "L2")
    set(_VX_NT 16)
    set(_VX_LMEM_LOG 14)
    set(_VX_HW "-DVX_CFG_NUM_THREADS=16 ${_VX_BASE} -DVX_CFG_LMEM_LOG_SIZE=14 -DVX_CFG_NUM_DXA_CORES=1 ${_VX_FMTS}")
else()
    message(FATAL_ERROR "GGML_VORTEX_TARGET must be H, L1, L2, J or W8 (got ${GGML_VORTEX_TARGET})")
endif()

if (NOT DEFINED _VX_NW)
    set(_VX_NW 16)
endif()

set(GGML_VORTEX_BUILD "${GGML_VORTEX_HOME}/build_llama_${GGML_VORTEX_TARGET}" CACHE PATH "Configured Vortex build directory for the target bitstream")

set(GGML_VORTEX_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/kernel")

if (GGML_VORTEX_EMU)
    ggml_add_backend_library(ggml-vortex ggml-vortex-emu.cpp)
    target_include_directories(ggml-vortex PRIVATE ${GGML_VORTEX_HOME}/sw/common ${GGML_VORTEX_KERNEL_SRC})
    # The projected device: NT=32 tiles; every TCU format the emulator can model.
    target_compile_definitions(ggml-vortex PRIVATE
        VX_CFG_NUM_THREADS=32 VX_CFG_NUM_WARPS=32
        VX_CFG_TCU_MX_ENABLE VX_CFG_TCU_FP4_ENABLE VX_CFG_TCU_NVFP4_ENABLE VX_CFG_TCU_LNSF4_ENABLE
        VX_CFG_TCU_LNSF4_ANTILOG_BITS=6
        ITYPE=${GGML_VORTEX_ITYPE} OTYPE=fp32)
    find_package(OpenMP REQUIRED)
    target_link_libraries(ggml-vortex PRIVATE OpenMP::OpenMP_CXX)
    return()
endif()
set(GGML_VORTEX_KERNEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/kernel")
set(GGML_VORTEX_KERNEL "${GGML_VORTEX_KERNEL_DIR}/kernel.vxbin")

file(MAKE_DIRECTORY ${GGML_VORTEX_KERNEL_DIR})
file(GLOB GGML_VORTEX_KERNEL_HDRS "${GGML_VORTEX_KERNEL_SRC}/*.h")

add_custom_command(
    OUTPUT  ${GGML_VORTEX_KERNEL}
    COMMAND make -s -f ${GGML_VORTEX_KERNEL_SRC}/Makefile -C ${GGML_VORTEX_KERNEL_DIR}
            VORTEX_BUILD=${GGML_VORTEX_BUILD}
            "CONFIGS=${_VX_HW} -DITYPE=${GGML_VORTEX_ITYPE} -DOTYPE=fp32"
            kernel.vxbin
    DEPENDS ${GGML_VORTEX_KERNEL_SRC}/kernel.cpp ${GGML_VORTEX_KERNEL_SRC}/Makefile ${GGML_VORTEX_KERNEL_HDRS}
    COMMENT "Building Vortex kernel ${GGML_VORTEX_KERNEL} (${GGML_VORTEX_TARGET}, ${GGML_VORTEX_ITYPE})"
    VERBATIM)
add_custom_target(ggml-vortex-kernel DEPENDS ${GGML_VORTEX_KERNEL})

ggml_add_backend_library(ggml-vortex
                         ggml-vortex.cpp
                        )

add_dependencies(ggml-vortex ggml-vortex-kernel)

target_include_directories(ggml-vortex PRIVATE
    ${GGML_VORTEX_HOME}/sw/runtime/include
    ${GGML_VORTEX_HOME}/sw/common
    ${GGML_VORTEX_BUILD}/sw
    ${GGML_VORTEX_KERNEL_SRC})

# The host packs and quantizes operands for the same hardware configuration the
# kernel is built for (thread/warp counts, LMEM, enabled TCU formats).
separate_arguments(_VX_HW_LIST UNIX_COMMAND "${_VX_HW}")
target_compile_options(ggml-vortex PRIVATE ${_VX_HW_LIST})

target_compile_definitions(ggml-vortex PRIVATE
    VX_CFG_NUM_THREADS=${_VX_NT}
    VX_CFG_NUM_WARPS=${_VX_NW}
    VX_CFG_LMEM_LOG_SIZE=${_VX_LMEM_LOG}
    ITYPE=${GGML_VORTEX_ITYPE}
    OTYPE=fp32
    GGML_VORTEX_KERNEL_PATH="${GGML_VORTEX_KERNEL}")

find_package(OpenMP REQUIRED)
target_link_libraries(ggml-vortex PRIVATE ${GGML_VORTEX_BUILD}/sw/runtime/libvortex.so OpenMP::OpenMP_CXX)
set_target_properties(ggml-vortex PROPERTIES BUILD_RPATH "${GGML_VORTEX_BUILD}/sw/runtime")
