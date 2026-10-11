#!/bin/bash
# usage: run_vx.sh <H|L1|L2> <command> [args...]
# Runs a llama.cpp binary against the target's U55C bitstream (one board user at a time).
set -e
T=$1; shift
source "$(dirname "$0")/vx_env.sh"
BIN=$VX_BITSTREAMS/$(vx_bitstream $T)
RT=$VX_HOME/build_llama_$T/sw/runtime
[ -f $BIN/vortex_afu.xclbin ] || { echo "no bitstream at $BIN" >&2; exit 1; }
source /opt/xilinx/xrt/setup.sh > /dev/null 2>&1
export VORTEX_DRIVER=xrt XRT_DEVICE_INDEX=0 XCLBIN_PATH=$BIN/vortex_afu.xclbin EMCONFIG_PATH=$BIN \
       XRT_INI_PATH=$RT/xrt/xrt.ini SCOPE_JSON_PATH=$BIN/scope.json \
       LD_LIBRARY_PATH=$XILINX_XRT/lib:$RT:$LD_LIBRARY_PATH
exec 9> /tmp/vortex_u55c_board.lock; flock 9
exec "$@"
