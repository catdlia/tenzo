#!/bin/bash
# Deferred speed measurements (each line self-contained, output CSV with load average). Run ONLY on a clean machine:
# 1-min load average < 1.0 and no firefox/zen/neko/steam. See results/2026-11-20_overnight/part2/pending_benchmarks.md.
set -u
cd "$(dirname "$0")/.."
OUT=results/2026-11-20_overnight/part2/bench; mkdir -p "$OUT"
clean() { awk '{exit !($1<1.0)}' /proc/loadavg && ! ps -eo comm | grep -qE 'firefox|zen-bin|neko|steam'; }
clean || { echo "machine not clean: $(cat /proc/loadavg)"; exit 3; }
# 1. I2_S vs LUT GEMV, same binary, interleaved ABBA x5, int8_fused, pp 128 / tg 128 and pp 512 / tg 128
scripts/ab_gemv_speed.sh $OUT/ab_gemv_pp128.csv 5 int8_fused 128 128
scripts/ab_gemv_speed.sh $OUT/ab_gemv_pp512.csv 5 int8_fused 512 128
# 2. e2e v3: Tenzo (i2s) fp32/int8_fused vs release bitnet.cpp (c17d1c5), prompts 16/128/512, gen 128, 5 reps
python3 scripts/run_e2e_fair.py --out $OUT/e2e_fair_v3.csv --reps 5 --gen 128 --prompts 16,128,512 --tenzo-kv fp32,int8_fused \
  --model models/bitnet-1.58b-v2 --llama-ctk f16,q8_0 --bench ~/tenzo-refs/BitNet/build/bin/llama-bench --bench-in-container
# 3. batched vs sequential prefill (TTFT and prefill tok/s), interleaved ABBA x5, int8_fused, chunk 32
scripts/ab_prefill_speed.sh $OUT/ab_prefill_pp128.csv 5 int8_fused 128 32
scripts/ab_prefill_speed.sh $OUT/ab_prefill_pp512.csv 5 int8_fused 512 32
# 4. decode speed of int8_fused / tl1_fused / ternary_rot / ternary_rot_alpha at prompt 128 and 512, 3 reps
scripts/bench_decode_modes.sh $OUT/decode_kv_modes.csv 3
