# Builds kernel.vxbin for the Vortex backend with the Vortex regression build
# rules. Invoked from CMake as:
#   make -f <this file> -C <out dir> VORTEX_BUILD=<vortex build dir> CONFIGS="<hw + ITYPE flags>" kernel.vxbin

KERNEL_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

VORTEX_BUILD ?= $(HOME)/vortex/build_llama_H
override ROOT_DIR := $(VORTEX_BUILD)

include $(VORTEX_BUILD)/config.mk

PROJECT := ggml_vortex_kernel

VX_SRCS := $(KERNEL_DIR)kernel.cpp
VX_CFLAGS += -I$(KERNEL_DIR)
VX_HDRS := $(wildcard $(KERNEL_DIR)*.h)

KERNEL_LIB := vortex2

include $(VORTEX_BUILD)/tests/regression/common.mk
