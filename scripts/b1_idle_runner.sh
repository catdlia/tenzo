#!/bin/bash
# Self-gated idle benchmark runner for B1 (T38).
# Pre-gate: waits until AC power online, 20 consecutive 30s checks with load1 < LOAD_THRESH (10 min),
# and zero forbidden processes (firefox|zen-bin|neko|steam|qemu-system|clion|idea|java).
# During each item (pinned to P-cores cpu 0-3): samples E-cores (cpu4..7) via /proc/stat every 2s.
# Valid iff rc == 0 AND foreign_cpu_mean < FOREIGN_THRESH (0.15) AND no forbidden procs before/after.
# load1_before/after are logged for provenance only. Invalid runs wait for idle again (up to 3 tries/item);
# exhausting 3 tries marks the item failed and advances to the next queue item (final STATUS: done or done_with_failures).
# Never commits or pushes. Resumes cleanly via item_done_<item>.ok markers.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

OUT_DIR="$ROOT/results/2026-11-24-b_B1"
DRY_RUN=0
SMOKE=0
LOAD_THRESH=1.0
FOREIGN_THRESH=0.15
SAMPLE_SEC=2
CONSEC_REQ=20
INTERVAL_SEC=30
TIMEOUT_HOURS=48
FAKE_CMD=""
PIN_CPUS="0-3"
FOREIGN_CPUS_REGEX="^cpu[4-7] "

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; FOREIGN_THRESH=99.0; shift ;;
    --smoke) SMOKE=1; shift ;;
    --out|--out-dir) OUT_DIR="$2"; shift 2 ;;
    --load-threshold) LOAD_THRESH="$2"; shift 2 ;;
    --foreign-threshold) FOREIGN_THRESH="$2"; shift 2 ;;
    --sample-sec) SAMPLE_SEC="$2"; shift 2 ;;
    --during-threshold) shift 2 ;; # legacy compatibility no-op
    --consec) CONSEC_REQ="$2"; shift 2 ;;
    --interval) INTERVAL_SEC="$2"; shift 2 ;;
    --timeout-hours) TIMEOUT_HOURS="$2"; shift 2 ;;
    --fake-cmd) FAKE_CMD="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$OUT_DIR" in
  /*) ;;
  *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac

mkdir -p "$OUT_DIR"
STATUS_FILE="$OUT_DIR/STATUS"
LOG_FILE="$OUT_DIR/runner.log"
SUMMARY_CSV="$OUT_DIR/b1_summary.csv"

if [ ! -f "$SUMMARY_CSV" ]; then
  echo "item,status,attempt,load1_before,load1_after,foreign_cpu_mean,foreign_cpu_max,governor,ac_online,temp_c,out_csv,rc" > "$SUMMARY_CSV"
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
  local max_mc=0 z v
  for z in /sys/class/thermal/thermal_zone*/temp; do
    [ -r "$z" ] || continue
    v=$(cat "$z" 2>/dev/null || echo 0)
    if [ "$v" -gt "$max_mc" ] 2>/dev/null; then
      max_mc=$v
    fi
  done
  awk -v m="$max_mc" 'BEGIN { printf "%.1f", m / 1000.0 }'
}

# Print "total_jiffies idle_jiffies" summed over E-cores (cpu4..7) from /proc/stat.
read_ecore_stat() {
  awk -v re="$FOREIGN_CPUS_REGEX" '
    $0 ~ re {
      tot += $2 + $3 + $4 + $5 + $6 + $7 + $8 + $9
      idl += $5 + $6
    }
    END { printf "%.0f %.0f\n", tot, idl }
  ' /proc/stat
}

compute_ecore_util() {
  local prev_t="$1" prev_i="$2" cur_t="$3" cur_i="$4"
  awk -v pt="$prev_t" -v pi="$prev_i" -v ct="$cur_t" -v ci="$cur_i" '
    BEGIN {
      dt = ct - pt
      di = ci - pi
      if (dt <= 0) { printf "0.0000\n" }
      else {
        u = (dt - di) / dt
        if (u < 0) u = 0
        if (u > 1) u = 1
        printf "%.4f\n", u
      }
    }
  '
}

has_forbidden_procs() {
  if [ "$DRY_RUN" -eq 1 ]; then
    return 1
  fi
  ps -eo comm= | grep -qE '^(firefox|zen-bin|neko|steam|qemu-system|clion|idea|java)'
}

forbidden_procs_list() {
  ps -eo comm= | grep -E '^(firefox|zen-bin|neko|steam|qemu-system|clion|idea|java)' | sort -u | tr '\n' ',' | sed 's/,$//'
}

# Returns 0 when idle gate met; 1 on timeout (gave_up).
wait_until_idle() {
  if [ "$SMOKE" -eq 1 ]; then
    log "SMOKE mode: skipping wait_until_idle pre-gate"
    return 0
  fi
  local consec=0
  local start_ts now_ts max_wait_sec
  start_ts=$(date +%s)
  max_wait_sec=$((TIMEOUT_HOURS * 3600))
  echo "waiting $(date -Is)" > "$STATUS_FILE"
  while true; do
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

ANY_FAILED=0

# Return value: 0 = item succeeded (valid) or skipped; 1 = item failed max attempts (continue queue); 2 = idle timeout (gave_up).
run_item() {
  local item="$1"
  local out_csv="$2"
  local cmd_str="$3"
  local done_flag="$OUT_DIR/item_done_${item}.ok"
  local fail_flag="$OUT_DIR/item_failed_${item}.fail"
  if [ -f "$done_flag" ]; then
    log "SKIP $item (already done)"
    return 0
  fi

  local max_attempts=3
  if [ "$SMOKE" -eq 1 ]; then
    max_attempts=1
  fi

  local attempt=1
  while [ "$attempt" -le "$max_attempts" ]; do
    wait_until_idle || return 2
    echo "running $item attempt=$attempt $(date -Is)" > "$STATUS_FILE"

    local lb proc_before=0 procs_b_str=""
    lb=$(get_load1)
    if has_forbidden_procs; then
      proc_before=1
      procs_b_str=$(forbidden_procs_list)
    fi
    local gov ac tc
    gov=$(get_gov)
    ac=$(get_ac)
    tc=$(get_temp_c)

    local util_file="$OUT_DIR/.fcpu_${item}_${attempt}"
    : > "$util_file"
    read -r prev_t prev_i < <(read_ecore_stat)

    # Background 2s E-core (/proc/stat cpu4..7) foreign CPU sampler
    (
      pt="$prev_t"; pi="$prev_i"
      while true; do
        sleep "$SAMPLE_SEC"
        read -r ct ci < <(read_ecore_stat)
        u=$(compute_ecore_util "$pt" "$pi" "$ct" "$ci")
        echo "$u" >> "$util_file"
        pt="$ct"; pi="$ci"
        echo "[monitor $item#$attempt $(date -Is)] foreign_cpu=$u load1=$(get_load1) temp_c=$(get_temp_c)" >> "$LOG_FILE"
      done
    ) &
    local mon_pid=$!

    log "RUN $item attempt=$attempt pinned=$PIN_CPUS proc_before=$proc_before($procs_b_str) cmd: $cmd_str"
    local rc=0
    if command -v systemd-inhibit >/dev/null 2>&1 && [ "$DRY_RUN" -eq 0 ] && [ "$SMOKE" -eq 0 ]; then
      systemd-inhibit --what=idle:sleep:handle-lid-switch --why="tenzo B1" taskset -c "$PIN_CPUS" bash -c "$cmd_str" >> "$LOG_FILE" 2>&1 || rc=$?
    else
      taskset -c "$PIN_CPUS" bash -c "$cmd_str" >> "$LOG_FILE" 2>&1 || rc=$?
    fi

    kill "$mon_pid" 2>/dev/null || true
    wait "$mon_pid" 2>/dev/null || true

    # Fallback single sample across whole command if run took < SAMPLE_SEC
    if [ ! -s "$util_file" ]; then
      read -r end_t end_i < <(read_ecore_stat)
      compute_ecore_util "$prev_t" "$prev_i" "$end_t" "$end_i" >> "$util_file"
    fi

    local f_stats fmean fmax
    f_stats=$(awk '
      BEGIN { sum=0; n=0; mx=0 }
      { v=$1+0; sum+=v; n++; if (v>mx) mx=v }
      END { if (n==0) printf "0.0000 0.0000\n"; else printf "%.4f %.4f\n", sum/n, mx }
    ' "$util_file")
    rm -f "$util_file"
    fmean="${f_stats%% *}"
    fmax="${f_stats##* }"

    local la proc_after=0 procs_a_str=""
    la=$(get_load1)
    if has_forbidden_procs; then
      proc_after=1
      procs_a_str=$(forbidden_procs_list)
    fi

    local ok_fcpu=0
    if awk -v m="$fmean" -v t="$FOREIGN_THRESH" 'BEGIN { exit !(m < t) }'; then
      ok_fcpu=1
    fi

    if [ "$rc" -eq 0 ] && [ "$ok_fcpu" -eq 1 ] && [ "$proc_before" -eq 0 ] && [ "$proc_after" -eq 0 ]; then
      echo "$item,valid,$attempt,$lb,$la,$fmean,$fmax,$gov,$ac,$tc,$out_csv,$rc" >> "$SUMMARY_CSV"
      touch "$done_flag"
      log "OK $item attempt=$attempt foreign_cpu_mean=$fmean foreign_cpu_max=$fmax load1=$lb->$la temp=${tc}C"
      return 0
    else
      local st="failed"
      if [ "$ok_fcpu" -eq 0 ]; then
        st="invalid_foreign_cpu"
      elif [ "$proc_before" -ne 0 ] || [ "$proc_after" -ne 0 ]; then
        st="invalid_proc"
      fi
      echo "$item,$st,$attempt,$lb,$la,$fmean,$fmax,$gov,$ac,$tc,$out_csv,$rc" >> "$SUMMARY_CSV"
      log "RETRY $item attempt=$attempt status=$st rc=$rc foreign_cpu_mean=$fmean (max=$fmax, thresh=$FOREIGN_THRESH) proc_b=$proc_before($procs_b_str) proc_a=$proc_after($procs_a_str) load1=$lb->$la"
      attempt=$((attempt + 1))
    fi
  done

  log "FAILED $item after $max_attempts attempts; continuing queue"
  touch "$fail_flag"
  ANY_FAILED=1
  return 1
}

log "=== B1 runner starting (dry_run=$DRY_RUN smoke=$SMOKE pin=$PIN_CPUS foreign_thresh=$FOREIGN_THRESH sample=${SAMPLE_SEC}s load_thresh=$LOAD_THRESH consec=$CONSEC_REQ interval=${INTERVAL_SEC}s) ==="

if [ "$DRY_RUN" -eq 1 ]; then
  FAKE="${FAKE_CMD:-echo DRY_RUN_BENCH_OK; sleep 2}"
  run_item "dryrun_item" "$OUT_DIR/dryrun.csv" "$FAKE"
  rc=$?
  echo "done $(date -Is) dry_run=1 rc=$rc" > "$STATUS_FILE"
  exit "$rc"
fi

if [ "$SMOKE" -eq 1 ]; then
  run_item "smoke_ab_gemv_pp16_tg16" "$OUT_DIR/smoke_ab_gemv.csv" "scripts/ab_gemv_speed.sh '$OUT_DIR/smoke_ab_gemv.csv' 1 int8_fused 16 16"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "done $(date -Is) smoke=1 valid=1" > "$STATUS_FILE"
  else
    echo "done_with_failures $(date -Is) smoke=1 valid=0" > "$STATUS_FILE"
  fi
  exit 0
fi

# Real B1 benchmark queue (all runs confined to P-cores 0-3)
run_item "ab_gemv_pp128"   "$OUT_DIR/ab_gemv_pp128.csv"   "scripts/ab_gemv_speed.sh '$OUT_DIR/ab_gemv_pp128.csv' 5 int8_fused 128 128"
[ $? -eq 2 ] && exit 1
run_item "ab_gemv_pp512"   "$OUT_DIR/ab_gemv_pp512.csv"   "scripts/ab_gemv_speed.sh '$OUT_DIR/ab_gemv_pp512.csv' 5 int8_fused 512 128"
[ $? -eq 2 ] && exit 1
run_item "e2e_fair_v3"     "$OUT_DIR/e2e_fair_v3.csv"     "python3 scripts/run_e2e_fair.py --out '$OUT_DIR/e2e_fair_v3.csv' --reps 5 --gen 128 --prompts 16,128,512 --tenzo-kv fp32,int8_fused --model models/bitnet-1.58b-v2 --llama-ctk f16,q8_0 --bench /home/illia/tenzo-refs/BitNet/build/bin/llama-bench --bench-in-container"
[ $? -eq 2 ] && exit 1
run_item "ab_prefill_pp128" "$OUT_DIR/ab_prefill_pp128.csv" "scripts/ab_prefill_speed.sh '$OUT_DIR/ab_prefill_pp128.csv' 5 int8_fused 128 32"
[ $? -eq 2 ] && exit 1
run_item "ab_prefill_pp512" "$OUT_DIR/ab_prefill_pp512.csv" "scripts/ab_prefill_speed.sh '$OUT_DIR/ab_prefill_pp512.csv' 5 int8_fused 512 32"
[ $? -eq 2 ] && exit 1
run_item "decode_kv_modes" "$OUT_DIR/decode_kv_modes.csv" "scripts/bench_decode_modes.sh '$OUT_DIR/decode_kv_modes.csv' 3"
[ $? -eq 2 ] && exit 1

if [ "$ANY_FAILED" -eq 1 ]; then
  echo "done_with_failures $(date -Is)" > "$STATUS_FILE"
  log "=== B1 runner COMPLETE (with failures) ==="
else
  echo "done $(date -Is)" > "$STATUS_FILE"
  log "=== B1 runner COMPLETE ==="
fi
exit 0
