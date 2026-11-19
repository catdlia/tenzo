#!/bin/bash
# Interleaved A/B (ABBA x reps) of --prefill sequential vs batched in the same binary; CSV with the 1-min load average per run.
# Usage: scripts/ab_prefill_speed.sh <out.csv> <reps> <kv> <pp> [chunk]
OUT=$1; REPS=$2; KV=$3; PP=$4; CH=${5:-32}
echo "rep,variant,kv,pp,chunk,pp_ms,pp_tps,tg_tps,load1" > "$OUT"
for r in $(seq 1 "$REPS"); do
  for v in sequential batched batched sequential; do
    load=$(cut -d' ' -f1 /proc/loadavg)
    line=$(docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-inference \
      -m /app/models/bitnet-1.58b-v2 --kv-quant "$KV" -c 1024 --ids-file /app/results/2026-11-18_overnight/e2e_ids.txt \
      --bench-pp "$PP" --bench-tg 16 --prefill $v --prefill-chunk $CH 2>/dev/null | grep -a BENCH_FAIR)
    ppms=$(echo "$line" | grep -o 'pp_ms=[0-9.]*' | cut -d= -f2); pptps=$(echo "$line" | grep -o 'pp_tps=[0-9.]*' | cut -d= -f2); tgtps=$(echo "$line" | grep -o 'tg_tps=[0-9.]*' | cut -d= -f2)
    echo "$r,$v,$KV,$PP,$CH,$ppms,$pptps,$tgtps,$load" >> "$OUT"
  done
done
