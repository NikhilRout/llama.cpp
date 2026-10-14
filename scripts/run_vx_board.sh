#!/bin/bash
# usage: [BIN=<bitstream dir>] run_vx_board.sh <cmd...>
# Runs a host program on the U55C with a Vortex bitstream (default: the W8 build), holding the board lock.
BIN=${BIN:-$HOME/vortex_bitstreams/llama_wmma_nt8_nw8_iw1_lmem64_axi32}
RT=${VX_RT:-$HOME/vortex/build/sw/runtime}
source /opt/xilinx/xrt/setup.sh > /dev/null 2>&1
exec 9> $HOME/.vortex_u55c_board.lock; flock 9
export XRT_INI_PATH=$RT/xrt/xrt.ini EMCONFIG_PATH=$BIN XRT_DEVICE_INDEX=0 XCLBIN_PATH=$BIN/vortex_afu.xclbin \
       LD_LIBRARY_PATH=$XILINX_XRT/lib:$RT:$LD_LIBRARY_PATH VORTEX_DRIVER=xrt
"$@"
