#!/bin/bash
# Decode speed of the KV modes (same binary, i2s GEMV): CSV rows per (kv, prompt, rep) with the 1-min load average.
# Usage: scripts/bench_decode_modes.sh <out.csv> <reps>
OUT=$1; REPS=$2
echo "kv,pp,rep,pp_tps,tg_tps,load1" > "$OUT"
for r in $(seq 1 "$REPS"); do for pp in 128 512; do for kv in int8_fused tl1_fused ternary_rot ternary_rot_alpha; do
  load=$(cut -d' ' -f1 /proc/loadavg)
  extra_args=""
  kv_flag="$kv"
  if [ "$kv" = "ternary_rot_alpha" ]; then
    kv_flag="ternary_rot"
    extra_args="--kv-alpha /app/results/2026-11-23_agent2/H1/alpha_mix.txt"
  fi
  line=$(docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-inference \
    -m /app/models/bitnet-1.58b-v2 --kv-quant "$kv_flag" $extra_args -c 1024 --ids-file /app/results/2026-11-18_overnight/e2e_ids.txt \
    --bench-pp $pp --bench-tg 128 --prefill batched 2>/dev/null | grep -a BENCH_FAIR)
  echo "$kv,$pp,$r,$(echo "$line" | grep -o 'pp_tps=[0-9.]*' | cut -d= -f2),$(echo "$line" | grep -o 'tg_tps=[0-9.]*' | cut -d= -f2),$load" >> "$OUT"
done; done; done
