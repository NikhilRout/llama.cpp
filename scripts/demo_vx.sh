#!/bin/bash
# usage: [BIN=<bitstream dir>] [N=<tokens>] demo_vx.sh "<question>" [formats...]
# Llama-3.2-1B on the U55C Vortex bitstream: runs the question through each
# format (default: bf16 lnsf4) and prints the answer plus prefill/decode tok/s.
cd "$(dirname "$0")/.." || exit 1
Q=${1:-"can you explain what quantization means?"}; shift
FMTS=${*:-bf16 lnsf4}
N=${N:-64}
MODEL=${MODEL:-$HOME/models/Llama-3.2-1B-Instruct-BF16.gguf}
LOGS=${LOGS:-$HOME/llama_runs/demo}
mkdir -p "$LOGS"
P=$LOGS/prompt.txt
printf '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' "$Q" > "$P"
for F in $FMTS; do
  L=$LOGS/$F.log
  echo "=== $F"
  GGML_VORTEX_BATCH=1 scripts/run_vx_board.sh build-vx-W8-$F/bin/llama-completion -m "$MODEL" -t 32 -c 2048 -nr -ngl 99 \
    -fa on -ot "token_embd\.weight=CPU" -n "$N" --temp 0 -no-cnv -f "$P" > "$L" 2>&1
  grep -av VXDRV "$L" | python3 -c "
import re, sys
t = sys.stdin.read()
i = t.rfind('assistant')
ans = t[i + 9:].split('\n\n', 2)[1] if i >= 0 and t[i + 9:].count('\n\n') >= 2 else ''
print(ans.strip())
pf = re.search(r'prompt eval time .*?([0-9.]+) tokens per second', t)
dc = re.search(r'\beval time .*?runs.*?([0-9.]+) tokens per second', t)
print(f'--- {sys.argv[1]}: prefill {pf.group(1) if pf else \"?\"} tok/s, decode {dc.group(1) if dc else \"?\"} tok/s')
" "$F"
done
