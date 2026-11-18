#!/bin/bash
# Four-variant attention ablation grid. Run INSIDE the dev container:
#   docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev bash /app/scripts/run_attn_ablation.sh <out.csv> [governor]
# One process per (geometry, T, threads, variant); data is seeded Gaussian; no other heavy process may run meanwhile.
set -u
OUT=${1:?csv}; GOV=${2:-unknown}; BIN=/app/cmake-build-debug/paper_benchmark
WARM=${WARM:-5}; ITERS=${ITERS:-30}
for geom in "20 5" "32 8"; do
  set -- $geom; HQ=$1; HKV=$2
  for T in 1024 4096 8192 16384 32768; do
    # untimed error measurement (all variants in one process)
    OMP_NUM_THREADS=1 taskset -c 0 $BIN --error-only --hq $HQ --hkv $HKV --dim 128 --context $T --threads 1 --csv "$OUT" --governor "$GOV" --label "laptop-i3-1215U"
    for thr in 1 4; do
      if [ $thr -eq 1 ]; then PIN="taskset -c 0"; else PIN="taskset -c 0-3"; fi
      for v in a b c d; do
        OMP_NUM_THREADS=$thr $PIN $BIN --variant $v --hq $HQ --hkv $HKV --dim 128 --context $T --threads $thr \
            --warmup $WARM --iters $ITERS --csv "$OUT" --governor "$GOV" --label "laptop-i3-1215U"
      done
    done
  done
done
echo "ablation done: $(grep -c '^timing' "$OUT") timing rows, $(grep -c '^error' "$OUT") error rows"
