#!/bin/bash
# Block D driver: golden fp32 + KV fake-quant scheme over a window file, one scheme at a time (never parallel).
# Usage: scripts/run_kv_ptq.sh <out_dir> <windows_file> <max_windows> <scheme> [<scheme> ...]
set -u
OUT=${1:?out}; WIN=${2:?windows}; NW=${3:?max windows}; shift 3
mkdir -p "$OUT"
for sc in "$@"; do
  f="$OUT/$(echo "$sc" | tr ':+' '__').txt"
  [ -s "$f" ] && grep -q "^lp" "$f" && [ "$(grep -c '^w ' "$f")" -ge "$NW" ] && { echo "skip $sc (done)"; continue; }
  echo "$(date -Is) scheme=$sc MemAvailable=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB" >> "$OUT/progress.txt"
  GOLDEN_QUIET=1 systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 -- \
    python3 tools/reference/golden_bitnet.py nll --dtype float32 --batch 4 --max-windows "$NW" --windows "$WIN" \
    --kv-scheme "$sc" --out "$f" > "${f%.txt}.log" 2>&1
  echo "$sc exit=$? $(date +%T)" >> "$OUT/progress.txt"
done
