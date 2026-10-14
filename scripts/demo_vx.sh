#!/bin/bash
# Llama-3.2-1B on the U55C Vortex demo bitstream: runs one question through each
# requested format and prints the answer with prefill/decode tok/s.
#
#   demo_vx.sh [--prompt="what is an FPGA?"] [--format=bf16,lnsf4] [--N=64]
#
#   --prompt  the question (default: "can you explain what quantization means?")
#   --format  comma-separated formats, run in that order: bf16, fp16, nvfp4, lnsf4,
#             or all (default: bf16,lnsf4). A format's backend is built on first use.
#   --N       tokens to generate (default: 64)

# The demo bitstream (fixed for the demo).
BIN=$HOME/vortex_bitstreams/llama_wmma_nt8_nw8_iw1_lmem64_axi32M

FORMATS="bf16 fp16 nvfp4 lnsf4"
Q="can you explain what quantization means?"
FMTS="bf16,lnsf4"
N=64
for arg in "$@"; do
  case "$arg" in
    --prompt=*) Q=${arg#*=} ;;
    --format=*) FMTS=${arg#*=} ;;
    --N=*|--n=*) N=${arg#*=} ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (see --help)"; exit 1 ;;
  esac
done
[ "$FMTS" = all ] && FMTS=${FORMATS// /,}
for F in ${FMTS//,/ }; do
  [[ " $FORMATS " == *" $F "* ]] || { echo "unknown format: $F (one of: $FORMATS)"; exit 1; }
done

cd "$(dirname "$0")/.." || exit 1
MODEL=$HOME/models/Llama-3.2-1B-Instruct-BF16.gguf
LOGS=$HOME/llama_runs/demo
mkdir -p "$LOGS"
P=$LOGS/prompt.txt
printf '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' "$Q" > "$P"

for F in ${FMTS//,/ }; do
  B=build-vx-W8-$F
  if [ ! -x $B/bin/llama-completion ]; then
    echo "(building the $F backend, first use only)"
    { cmake -B $B -DCMAKE_BUILD_TYPE=Release -DGGML_VORTEX=ON -DGGML_VORTEX_TARGET=W8 -DGGML_VORTEX_ITYPE=$F \
            -DGGML_VORTEX_HOME=$HOME/vortex -DGGML_VORTEX_BUILD=$HOME/vortex/build &&
      cmake --build $B -j32 --target llama-completion; } > $LOGS/build_$F.log 2>&1 ||
      { echo "build failed, see $LOGS/build_$F.log"; exit 1; }
  fi
  L=$LOGS/$F.log; A=$LOGS/$F.out
  echo "=== $F"
  BIN=$BIN GGML_VORTEX_BATCH=1 scripts/run_vx_board.sh $B/bin/llama-completion -m "$MODEL" -t 32 -c 2048 -nr -ngl 99 \
    -fa on -ot "token_embd\.weight=CPU" -n "$N" --temp 0 -no-cnv -f "$P" > "$A" 2> "$L"
  grep -av VXDRV "$A" | python3 -c "
import sys
t = sys.stdin.read()
i = t.rfind('assistant')
print(t[i + 9:].strip() if i >= 0 else t.strip())"
  grep -a 'eval time' "$L" | python3 -c "
import re, sys
t = sys.stdin.read()
pf = re.search(r'prompt eval time .*?([0-9.]+) tokens per second', t)
dc = re.search(r'\n.*?\beval time .*?runs.*?([0-9.]+) tokens per second', '\n' + t)
print(f'--- {sys.argv[1]}: prefill {pf.group(1) if pf else \"?\"} tok/s, decode {dc.group(1) if dc else \"?\"} tok/s')
" "$F"
done
