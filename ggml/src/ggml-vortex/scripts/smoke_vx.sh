#!/bin/bash
# usage: smoke_vx.sh <H|L1|L2> [fmt]  -- first-light check of a new bitstream: 3 greedy tokens of a
# fixed prompt with every weight MUL_MAT on the device, plus the backend's timing summary.
T=$1; F=${2:-bf16}
DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
P=$(mktemp); printf '%s' '<|start_header_id|>user<|end_header_id|>

can you explain what quantization means?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

' > $P
GGML_VORTEX_VERBOSE=${GGML_VORTEX_VERBOSE:-0} "$(dirname "$0")/run_vx.sh" $T \
  $DIR/build-vx-$T-$F/bin/llama-completion -m ${MODEL:-$HOME/models/Llama-3.2-1B-Instruct-BF16.gguf} \
  -t 32 -c 2048 -nr -n ${N:-3} --temp 0 -no-cnv -f $P -v 2>&1 | grep -a -E 'Vortex|assistant|Quantization|eval time|load time|ABORT|error'
rm -f $P
