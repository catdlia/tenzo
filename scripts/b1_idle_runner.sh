#!/bin/bash
# Self-gated idle benchmark runner for B1 (T38).
# Waits until AC power online, 20 consecutive 30s checks with load1 < LOAD_THRESH (10 min),
# and no browser/VM/IDE/Steam processes; then runs pending_benchmarks items sequentially with
# live 5s load1 sampling (invalidating if load1 > DURING_LOAD_THRESH, up to 2 retries per item).
# Never commits or pushes. Resumes cleanly across restarts via done marker files.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

OUT_DIR="$ROOT/results/2026-11-24_B1"
DRY_RUN=0
LOAD_THRESH=1.0
DURING_LOAD_THRESH=1.5
CONSEC_REQ=20
INTERVAL_SEC=30
TIMEOUT_HOURS=48
FAKE_CMD=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; DURING_LOAD_THRESH=99.0; shift ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --load-threshold) LOAD_THRESH="$2"; shift 2 ;;
    --during-threshold) DURING_LOAD_THRESH="$2"; shift 2 ;;
    --consec) CONSEC_REQ="$2"; shift 2 ;;
    --interval) INTERVAL_SEC="$2"; shift 2 ;;
    --timeout-hours) TIMEOUT_HOURS="$2"; shift 2 ;;
    --fake-cmd) FAKE_CMD="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"
STATUS_FILE="$OUT_DIR/STATUS"
LOG_FILE="$OUT_DIR/runner.log"
SUMMARY_CSV="$OUT_DIR/b1_summary.csv"

if [ ! -f "$SUMMARY_CSV" ]; then
  echo "item,status,attempt,load_before,load_max_during,load_after,governor,ac_online,temp_c,out_csv,rc" > "$SUMMARY_CSV"
fi

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG_FILE"
}

get_ac() {
  local ac="1"
  for f in /sys/class/power_supply/A*/online; do
    [ -r "$f" ] || continue
    if [ "$(cat "$f" 2>/dev/null)" != "1" ]; then
      ac="0"
    fi
  done
  echo "$ac"
}

get_gov() {
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown"
}

get_load1() {
  cut -d' ' -f1 /proc/loadavg
}

get_temp_c() {
  local max_mc=0 z
  for z in /sys/class/thermal/thermal_zone*/temp; do
    [ -r "$z" ] || continue
    local v
    v=$(cat "$z" 2>/dev/null || echo 0)
    if [ "$v" -gt "$max_mc" ] 2>/dev/null; then
      max_mc=$v
    fi
  done
  awk -v m="$max_mc" 'BEGIN { printf "%.1f", m / 1000.0 }'
}

has_forbidden_procs() {
  if [ "$DRY_RUN" -eq 1 ]; then
    return 1
  fi
  ps -eo comm= | grep -qE '^(firefox|zen-bin|neko|steam|qemu-system|clion|idea|java)'
}

wait_until_idle() {
  local consec=0
  local start_ts
  start_ts=$(date +%s)
  local max_wait_sec=$((TIMEOUT_HOURS * 3600))
  echo "waiting $(date -Is)" > "$STATUS_FILE"
  while true; do
    local now_ts
    now_ts=$(date +%s)
    if [ $((now_ts - start_ts)) -ge "$max_wait_sec" ]; then
      echo "gave_up $(date -Is) timeout=${TIMEOUT_HOURS}h" > "$STATUS_FILE"
      log "GAVE UP waiting for idle state after ${TIMEOUT_HOURS}h"
      return 1
    fi
    local ac l1 ok_l1=0 ok_proc=0
    ac=$(get_ac)
    l1=$(get_load1)
    if awk -v l="$l1" -v t="$LOAD_THRESH" 'BEGIN { exit !(l < t) }'; then
      ok_l1=1
    fi
    if ! has_forbidden_procs; then
      ok_proc=1
    fi
    if [ "$ac" = "1" ] && [ "$ok_l1" -eq 1 ] && [ "$ok_proc" -eq 1 ]; then
      consec=$((consec + 1))
      log "idle check $consec/$CONSEC_REQ OK (load1=$l1 < $LOAD_THRESH, ac=$ac)"
      if [ "$consec" -ge "$CONSEC_REQ" ]; then
        return 0
      fi
    else
      if [ "$consec" -gt 0 ]; then
        log "idle streak reset ($consec -> 0): ac=$ac load1=$l1 ok_proc=$ok_proc"
      fi
      consec=0
    fi
    sleep "$INTERVAL_SEC"
  done
}

run_item() {
  local item="$1"
  local out_csv="$2"
  local cmd_str="$3"
  local done_flag="$OUT_DIR/item_done_${item}.ok"
  if [ -f "$done_flag" ]; then
    log "SKIP $item (already done)"
    return 0
  fi

  local attempt=1
  while [ "$attempt" -le 3 ]; do
    wait_until_idle || return 1
    echo "running $item attempt=$attempt $(date -Is)" > "$STATUS_FILE"
    local lb gov ac tc max_l_file
    lb=$(get_load1)
    gov=$(get_gov)
    ac=$(get_ac)
    tc=$(get_temp_c)
    max_l_file="$OUT_DIR/.maxload_${item}_${attempt}"
    echo "$lb" > "$max_l_file"

    # Background 5s load sampler
    (
      while true; do
        sleep 5
        local cl cur_max
        cl=$(get_load1)
        cur_max=$(cat "$max_l_file" 2>/dev/null || echo "$cl")
        if awk -v a="$cl" -v b="$cur_max" 'BEGIN { exit !(a > b) }'; then
          echo "$cl" > "$max_l_file"
        fi
        echo "[monitor $item#$attempt $(date -Is)] load1=$cl temp_c=$(get_temp_c)" >> "$LOG_FILE"
      done
    ) &
    local mon_pid=$!

    log "RUN $item attempt=$attempt cmd: $cmd_str"
    local rc=0
    if command -v systemd-inhibit >/dev/null 2>&1 && [ "$DRY_RUN" -eq 0 ]; then
      systemd-inhibit --what=idle:sleep:handle-lid-switch --why="tenzo B1" bash -c "$cmd_str" >> "$LOG_FILE" 2>&1 || rc=$?
    else
      bash -c "$cmd_str" >> "$LOG_FILE" 2>&1 || rc=$?
    fi

    kill "$mon_pid" 2>/dev/null || true
    wait "$mon_pid" 2>/dev/null || true

    local lmax la valid=1
    lmax=$(cat "$max_l_file" 2>/dev/null || get_load1)
    rm -f "$max_l_file"
    la=$(get_load1)
    if awk -v lm="$lmax" -v dt="$DURING_LOAD_THRESH" 'BEGIN { exit !(lm > dt) }'; then
      valid=0
    fi
    if [ "$rc" -eq 0 ] && [ "$valid" -eq 1 ]; then
      echo "$item,valid,$attempt,$lb,$lmax,$la,$gov,$ac,$tc,$out_csv,$rc" >> "$SUMMARY_CSV"
      touch "$done_flag"
      log "OK $item attempt=$attempt load_before=$lb load_max=$lmax load_after=$la temp=${tc}C"
      return 0
    else
      local st="failed"
      [ "$valid" -eq 0 ] && st="invalid_load"
      echo "$item,$st,$attempt,$lb,$lmax,$la,$gov,$ac,$tc,$out_csv,$rc" >> "$SUMMARY_CSV"
      log "RETRY $item attempt=$attempt status=$st rc=$rc load_max=$lmax (> $DURING_LOAD_THRESH)"
      attempt=$((attempt + 1))
    fi
  done
  log "FAILED $item after 3 attempts"
  return 1
}

log "=== B1 runner starting (dry_run=$DRY_RUN load_thresh=$LOAD_THRESH during_thresh=$DURING_LOAD_THRESH consec=$CONSEC_REQ interval=${INTERVAL_SEC}s) ==="

if [ "$DRY_RUN" -eq 1 ]; then
  FAKE="${FAKE_CMD:-echo DRY_RUN_BENCH_OK; sleep 2}"
  run_item "dryrun_item" "$OUT_DIR/dryrun.csv" "$FAKE"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "done $(date -Is) dry_run=1" > "$STATUS_FILE"
  else
    echo "gave_up $(date -Is) dry_run=1" > "$STATUS_FILE"
  fi
  exit "$rc"
fi

# Real B1 benchmark queue (pending_benchmarks.sh items + ternary_rot_alpha decode)
run_item "ab_gemv_pp128"   "$OUT_DIR/ab_gemv_pp128.csv"   "scripts/ab_gemv_speed.sh '$OUT_DIR/ab_gemv_pp128.csv' 5 int8_fused 128 128" || { echo "gave_up $(date -Is)" > "$STATUS_FILE"; exit 1; }
run_item "ab_gemv_pp512"   "$OUT_DIR/ab_gemv_pp512.csv"   "scripts/ab_gemv_speed.sh '$OUT_DIR/ab_gemv_pp512.csv' 5 int8_fused 512 128" || { echo "gave_up $(date -Is)" > "$STATUS_FILE"; exit 1; }
run_item "e2e_fair_v3"     "$OUT_DIR/e2e_fair_v3.csv"     "python3 scripts/run_e2e_fair.py --out '$OUT_DIR/e2e_fair_v3.csv' --reps 5 --gen 128 --prompts 16,128,512 --tenzo-kv fp32,int8_fused --model models/bitnet-1.58b-v2 --llama-ctk f16,q8_0 --bench /home/illia/tenzo-refs/BitNet/build/bin/llama-bench --bench-in-container" || { echo "gave_up $(date -Is)" > "$STATUS_FILE"; exit 1; }
run_item "ab_prefill_pp128" "$OUT_DIR/ab_prefill_pp128.csv" "scripts/ab_prefill_speed.sh '$OUT_DIR/ab_prefill_pp128.csv' 5 int8_fused 128 32" || { echo "gave_up $(date -Is)" > "$STATUS_FILE"; exit 1; }
run_item "ab_prefill_pp512" "$OUT_DIR/ab_prefill_pp512.csv" "scripts/ab_prefill_speed.sh '$OUT_DIR/ab_prefill_pp512.csv' 5 int8_fused 512 32" || { echo "gave_up $(date -Is)" > "$STATUS_FILE"; exit 1; }
run_item "decode_kv_modes" "$OUT_DIR/decode_kv_modes.csv" "scripts/bench_decode_modes.sh '$OUT_DIR/decode_kv_modes.csv' 3" || { echo "gave_up $(date -Is)" > "$STATUS_FILE"; exit 1; }

echo "done $(date -Is)" > "$STATUS_FILE"
log "=== B1 runner COMPLETE ==="
exit 0
