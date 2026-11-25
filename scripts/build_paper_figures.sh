#!/usr/bin/env bash
# Rebuild the paper's figures and tables from committed score files / CSVs (T04). No model is run.
#
#   scripts/build_paper_figures.sh            # run everything, report failures, exit 0
#   scripts/build_paper_figures.sh --strict   # exit 1 if any step failed
#   scripts/build_paper_figures.sh --dry-run  # only print the steps
#
# Notes
# - Steps that rebuild windows or character offsets need the raw corpora under ~/tenzo-refs/datasets
#   (FLORES-200, NTREX-128; not redistributed). Without them those steps fail and are reported; the
#   others still run.
# - Re-running the agent's analysis scripts rewrites their md/csv files. If the inputs did not change,
#   only provenance lines (commit hash, timestamps) differ: check with `git diff --stat` before committing.
# - Everything runs under `nice -n 19`; total runtime is a few minutes on the laptop.
set -u
cd "$(dirname "$0")/.." || exit 1
STRICT=0; DRY=0
for a in "$@"; do
  case "$a" in
    --strict) STRICT=1 ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done
LOG=results/paper/figures_build.log
mkdir -p results/paper
[ "$DRY" -eq 1 ] || : > "$LOG"
fail=0
run() {
  if [ "$DRY" -eq 1 ]; then echo "$*"; return; fi
  echo "== $*" | tee -a "$LOG"
  if nice -n 19 "$@" >> "$LOG" 2>&1; then
    echo "   ok" | tee -a "$LOG"
  else
    echo "   FAILED (see $LOG)" | tee -a "$LOG"
    fail=$((fail + 1))
  fi
}

# --- coordinator re-analyses (score files only) ---
run python3 scripts/analyze_matched_difficulty.py          # F1, F2 preview
run python3 scripts/plot_matched_difficulty.py
run python3 scripts/analyze_disparity_decomposition.py     # F2a
run python3 scripts/plot_disparity_decomposition.py
run python3 scripts/analyze_iso_loss_bits.py               # F5a
run python3 scripts/plot_iso_loss_bits.py
run python3 scripts/analyze_ntrex_slot_ratio.py
run python3 scripts/analyze_fertility_vs_competence.py --set flores5 --response R
run python3 scripts/analyze_fertility_vs_competence.py --set flores5 --response ratio_lt4
run python3 scripts/analyze_fertility_vs_competence.py --set flores5 --response fragility

# --- agent analyses (score files + window meta; some need ~/tenzo-refs texts) ---
run python3 scripts/analyze_h3_flores.py                   # T1 (FLORES-32, 3 models)
run python3 scripts/analyze_h3_multilang.py                # 5 languages
run python3 scripts/analyze_h3b_char_window.py             # F3 (5 languages)
run python3 scripts/analyze_h4_ntrex.py                    # T1 (NTREX)
run python3 scripts/analyze_h1b_ctx2048.py                 # F6
run python3 scripts/run_h1_char_window.py --analyze-only   # F3 (FLORES-32 coverage curves)
run python3 scripts/analyze_disparity_bpc.py               # T2
run python3 scripts/build_paper_tables.py                  # results/paper/tables.{md,tex}

if [ "$DRY" -eq 0 ]; then
  echo "failures: $fail" | tee -a "$LOG"
  if [ "$STRICT" -eq 1 ] && [ "$fail" -gt 0 ]; then exit 1; fi
fi
exit 0
