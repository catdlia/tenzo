#!/bin/bash
# Teacher-forced WikiText-2 scoring per KV mode (Block C). Sequential, inside the dev container, ctx 512 (windows are 512 tokens).
# Usage: scripts/run_kv_ppl.sh <out_dir> [windows_file] [model_dir] [modes...]
OUT=${1:?out dir}; WIN=${2:-results/reference/windows.ids}; MODEL=${3:-models/bitnet-1.58b}
shift 3 2>/dev/null; MODES=${*:-fp32 int8_fused tl1_fused popcount_fused}
mkdir -p "$OUT"
for kv in $MODES; do
  echo "$(date -Is) kv=$kv MemAvailable=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB" >> "$OUT/progress.txt"
  docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev \
    /app/cmake-build-debug/tenzo-inference -m /app/$MODEL --kv-quant $kv -c 512 \
    --score /app/$WIN --score-out /app/$OUT/$kv.txt > "$OUT/$kv.log" 2>&1
  echo "$kv exit=$? $(date +%T)" >> "$OUT/progress.txt"
done
