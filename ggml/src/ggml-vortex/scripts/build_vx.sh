#!/bin/bash
# usage: build_vx.sh [targets] [formats]  -- e.g. build_vx.sh "H L1" "bf16 lnsf4"
# Builds llama.cpp with the Vortex backend into build-vx-<T>-<fmt>.
TARGETS=${1:-"H L1 L2"}; FMTS=${2:-"bf16 nvfp4 lnsf4"}
cd "$(dirname "$0")/../../../.."
for T in $TARGETS; do for F in $FMTS; do
  B=build-vx-$T-$F
  cmake -B $B -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DGGML_VORTEX=ON \
        -DGGML_VORTEX_TARGET=$T -DGGML_VORTEX_ITYPE=$F > $B.cmake.log 2>&1 &&
  nice -n 10 cmake --build $B -j ${JOBS:-8} --target llama-completion llama-cli llama-bench > $B.log 2>&1
  echo "$B rc=$?"
done; done
