#!/bin/bash
# Teacher-forced WikiText-2 scoring per KV mode (Block C). Sequential, inside the dev container, ctx 512 (windows are 512 tokens).
# Usage: scripts/run_kv_ppl.sh <out_dir> [windows_file]
OUT=${1:?out dir}; WIN=${2:-results/reference/windows.ids}
mkdir -p "$OUT"
for kv in fp32 int8_fused tl1_fused popcount_fused; do
  docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev \
    /app/cmake-build-debug/tenzo-inference -m /app/models/bitnet-1.58b --kv-quant $kv -c 512 \
    --score /app/$WIN --score-out /app/$OUT/$kv.txt > "$OUT/$kv.log" 2>&1
  echo "$kv exit=$? $(date +%T)" >> "$OUT/progress.txt"
done
