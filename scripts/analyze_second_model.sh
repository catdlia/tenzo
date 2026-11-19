#!/bin/bash
# dNLL / regression analysis of the Qwen2.5-1.5B KV study (scheme files in part2/second_model). Golden = the same model with fp KV.
set -u
D=${1:-results/2026-11-20_overnight/part2/second_model}; OUT=${2:-results/2026-11-20_overnight/part2/second_model}
args=()
for s in tenzo_int8 S2 S3 S3_s4r32; do
  [ -s "$D/${s}_en.txt" ] && [ -s "$D/${s}_uk.txt" ] || continue
  [ "$(grep -c '^w ' "$D/${s}_en.txt")" -ge 16 ] && [ "$(grep -c '^w ' "$D/${s}_uk.txt")" -ge 16 ] || continue
  case $s in tenzo_int8) n="INT8 per token (engine int8_fused rule)";; S2) n="ternary g32";; S3) n="ternary g32 + Hadamard";; S3_s4r32) n="ternary g32 + Hadamard + 4 sink / 32 recent INT8";; esac
  args+=("$n=$D/${s}_en.txt,$D/${s}_uk.txt")
done
python3 scripts/analyze_kv_calib.py --golden-en "$D/fp_en.txt" --golden-uk "$D/fp_uk.txt" --out "$OUT" "${args[@]}"
