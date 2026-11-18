#!/bin/bash
# Interleaved A/B decode/prefill speed: old kernel (c0db56f, 6-bit LUT) vs new (HEAD, 8-bit two-table LUT), same config
# (cmake-build-debug/ab/build_{old,new}), same model dir, same moment: ABBA order repeated, so machine drift cancels.
# Usage: scripts/ab_kernel_speed.sh <out.csv> <reps> <kv> <pp> <tg>
OUT=$1; REPS=$2; KV=$3; PP=$4; TG=$5
echo "rep,variant,kv,pp,tg,pp_ms,tg_ms,pp_tps,tg_tps" > "$OUT"
for r in $(seq 1 "$REPS"); do
  for v in old new new old; do
    line=$(docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/ab/build_$v/tenzo-inference \
      -m /app/models/bitnet-1.58b-v2 --kv-quant "$KV" -c 1024 --ids-file /app/results/2026-11-18_overnight/e2e_ids.txt \
      --bench-pp "$PP" --bench-tg "$TG" 2>/dev/null | grep -a BENCH_FAIR)
    pp_ms=$(echo "$line" | grep -o 'pp_ms=[0-9.]*' | cut -d= -f2); tg_ms=$(echo "$line" | grep -o 'tg_ms=[0-9.]*' | cut -d= -f2)
    echo "$r,$v,$KV,$PP,$TG,$pp_ms,$tg_ms,$(python3 -c "print($PP/($pp_ms/1e3))"),$(python3 -c "print($TG/($tg_ms/1e3))")" >> "$OUT"
  done
done
