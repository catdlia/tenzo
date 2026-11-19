#!/bin/bash
# G6 regression check: greedy generation must be unchanged by any engine change.
# Usage: scripts/g6_check.sh <out_dir> [model_dir]   (run on host; executes inside docker)
set -u
OUT=${1:?out dir}; MODEL=${2:-/app/models/bitnet-1.58b-v2}
mkdir -p "$OUT"
for kv in fp32 int8_fused tl1_fused popcount_fused ternary_rot; do
  docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev \
    /app/cmake-build-debug/tenzo-inference -p "The capital of France is" -n 24 -t 0 --rep-penalty 1.0 \
    -m "$MODEL" --kv-quant "$kv" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -a -A3 'Output:' | grep -a -v 'Time\|Decode\|╔\|╠\|║' > "$OUT/g6_$kv.txt"
  echo "$kv: $(sha256sum < $OUT/g6_$kv.txt | cut -c1-12) $(head -c 100 $OUT/g6_$kv.txt | tr '\n' ' ')"
done
