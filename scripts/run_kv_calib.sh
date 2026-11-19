#!/bin/bash
# Round 3 pipeline (quiet mode friendly: nice, MemoryMax 3G): calibrate on EN / UK / EN+UK, then evaluate each on 16 EN + 32 UK.
# Usage: scripts/run_kv_calib.sh <out_dir> [steps]
set -u
OUT=${1:?out}; STEPS=${2:-100}; mkdir -p "$OUT/eval_en16" "$OUT/eval_uk32"
RUN="systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 -- nice -n 19 ionice -c3"
cal() { # name, windows..., per-file in $PF
  name=$1; shift
  [ -s "$OUT/alpha_$name.json" ] && grep -q '"layer": 29' "$OUT/alpha_$name.json" && { echo "skip calib $name"; return; }
  echo "$(date -Is) calibrate $name MemAvailable=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB" >> "$OUT/progress.txt"
  $RUN python3 tools/reference/kv_calib.py calibrate --windows "$@" --per-file $PF --steps $STEPS --mb 2 --lr 0.02 --out "$OUT/alpha_$name.json" > "$OUT/calib_$name.log" 2>&1
  echo "calib $name exit=$? $(date +%T)" >> "$OUT/progress.txt"
}
ev() { # name, windows file, count, subdir
  f="$OUT/$4/calib_$1.txt"
  [ -s "$f" ] && [ "$(grep -c '^w ' "$f")" -ge "$3" ] && { echo "skip eval $1 $4"; return; }
  echo "$(date -Is) eval $1 $4 MemAvailable=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB" >> "$OUT/progress.txt"
  GOLDEN_QUIET=1 $RUN python3 tools/reference/golden_bitnet.py nll --dtype float32 --batch 4 --max-windows $3 --windows "$2" \
    --kv-scheme "calib:$OUT/alpha_$1.json" --out "$f" > "${f%.txt}.log" 2>&1
  echo "eval $1 $4 exit=$? $(date +%T)" >> "$OUT/progress.txt"
}
PF=8; cal en results/reference/calib_en8.ids
PF=8; cal uk results/reference/calib_uk8.ids
PF=4; cal mix results/reference/calib_en8.ids results/reference/calib_uk8.ids
for n in en uk mix; do ev $n results/reference/windows.ids 16 eval_en16; ev $n results/reference/windows_uk32.ids 32 eval_uk32; done
echo PIPELINEDONE >> "$OUT/progress.txt"
