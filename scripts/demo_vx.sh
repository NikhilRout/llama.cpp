#!/bin/bash
# Llama-3.2-1B on Vortex: the U55C demo bitstream, or with --emu a CPU emulation
# paced to the projected Vortex (4 cores, NT=NW=32, 200 MHz).
#
#   demo_vx.sh [--prompt="what is an FPGA?"] [--format=bf16,lnsf4] [--N=64] [--emu]
#   demo_vx.sh --chat [--format=lnsf4] [--emu]
#
#   --prompt  the question (default: "can you explain what quantization means?")
#   --format  comma-separated formats, run in that order: bf16, fp16, nvfp4, lnsf4,
#             or all (default: bf16,lnsf4). A format's backend is built on first use.
#   --N       tokens to generate (default: 64)
#   --chat    interactive chat in llama.cpp's own frontend (llama-cli), first format
#             only, exactly as llama-cli prints it. Ctrl+C or /exit to quit.
#   --emu     run on the CPU emulation of the projected Vortex instead of the board

# The demo bitstream (fixed for the demo).
BIN=$HOME/vortex_bitstreams/llama_wmma_nt8_nw8_iw1_lmem64_axi32M

FORMATS="bf16 fp16 nvfp4 lnsf4"
Q="can you explain what quantization means?"
FMTS="bf16,lnsf4"
N=64
CHAT=0
EMU=0
for arg in "$@"; do
  case "$arg" in
    --prompt=*) Q=${arg#*=} ;;
    --format=*) FMTS=${arg#*=} ;;
    --N=*|--n=*) N=${arg#*=} ;;
    --chat) CHAT=1 ;;
    --emu) EMU=1 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
# The emulation runs 16 threads on socket 1's physical cores: every op waits for
# its slowest thread, so more threads on this shared server only add stragglers.
EMU_CPUS=16-31
[ $EMU = 1 ] && T=16 || T=32
COMMON=(-m "$MODEL" -t $T -c 2048 -fa on -ot "token_embd\.weight=CPU" --temp 0)
# The board backend is a GPU-type device; the emulation is an accelerator ggml uses as is.
[ $EMU = 1 ] || COMMON+=(-ngl 99)

# Backend build directory for format $1, built on first use.
backend() {
  local F=$1 B
  if [ $EMU = 1 ]; then
    B=build-emu-$F
    local OPTS=(-DGGML_VORTEX_EMU=ON -DGGML_VORTEX_TARGET=J)
  else
    B=build-vx-W8-$F
    local OPTS=(-DGGML_VORTEX_TARGET=W8 -DGGML_VORTEX_BUILD=$HOME/vortex/build)
  fi
  if [ ! -x $B/bin/llama-completion ] || [ ! -x $B/bin/llama-cli ]; then
    echo "(building the $F backend, first use only)" >&2
    { cmake -B $B -DCMAKE_BUILD_TYPE=Release -DGGML_VORTEX=ON -DGGML_VORTEX_ITYPE=$F \
            -DGGML_VORTEX_HOME=$HOME/vortex "${OPTS[@]}" &&
      cmake --build $B -j32 --target llama-completion llama-cli; } > $LOGS/build_$B.log 2>&1 ||
      { echo "build failed, see $LOGS/build_$B.log" >&2; return 1; }
  fi
  echo $B
}

# Runs a llama.cpp binary on the board, or on the paced CPU emulation.
run() {
  if [ $EMU = 1 ]; then
    # --poll 0: the emulator's OpenMP workers need the cores ggml would spin on.
    OMP_NUM_THREADS=$T taskset -c $EMU_CPUS "$@" --poll 0
  else
    BIN=$BIN GGML_VORTEX_BATCH=1 scripts/run_vx_board.sh "$@"
  fi
}

if [ $EMU = 1 ]; then
  echo "[CPU emulation of the projected Vortex: 4 cores, NT=NW=32, 200 MHz]"
else
  echo "[U55C FPGA: $(basename "$BIN")]"
fi

if [ $CHAT = 1 ]; then
  F=${FMTS%%,*}
  B=$(backend $F) || exit 1
  echo "=== $F chat"
  run $B/bin/llama-cli "${COMMON[@]}"
  exit
fi

P=$LOGS/prompt.txt
printf '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' "$Q" > "$P"
echo "Q: $Q"
for F in ${FMTS//,/ }; do
  B=$(backend $F) || exit 1
  L=$LOGS/$F.log; A=$LOGS/$F.out
  echo "=== $F"
  # The answer streams token by token; llama's log goes to $L.
  run $B/bin/llama-completion "${COMMON[@]}" -nr -n "$N" -no-cnv --no-display-prompt -f "$P" 2> "$L" | tee "$A"
  echo
  grep -a 'eval time' "$L" | python3 -c "
import re, sys
t = sys.stdin.read()
pf = re.search(r'prompt eval time .*?([0-9.]+) tokens per second', t)
dc = re.search(r'\n.*?\beval time .*?runs.*?([0-9.]+) tokens per second', '\n' + t)
print(f'--- {sys.argv[1]}: prefill {pf.group(1) if pf else \"?\"} tok/s, decode {dc.group(1) if dc else \"?\"} tok/s')
" "$F"
done
