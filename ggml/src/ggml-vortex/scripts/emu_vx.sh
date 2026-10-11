#!/bin/bash
# usage: emu_vx.sh <bf16|lnsf4|nvfp4> <llama binary> [args...]
#   e.g. emu_vx.sh lnsf4 llama-cli -m ~/models/Llama-3.2-1B-Instruct-BF16.gguf
# Runs a llama.cpp binary from build-emu-<fmt> with the Vortex emulation backend:
# TCU numerics on the host, paced to the projected device (GGML_VORTEX_EMU_* knobs,
# default 4 cores x NT32 @ 200 MHz; GGML_VORTEX_EMU_PACE=0 for raw host speed).
# The host splits its cores between llama.cpp's CPU ops and the emulated GEMMs;
# --poll 0 keeps idle llama.cpp threads from starving the GEMM threads.
set -e
F=$1; B=$2; shift 2
DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
T=${THREADS:-32}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$T}
exec "$DIR/build-emu-$F/bin/$B" -t $T --poll 0 "$@"
