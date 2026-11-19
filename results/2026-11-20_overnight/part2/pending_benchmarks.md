# Pending speed measurements (queue)

Run `scripts/pending_benchmarks.sh` on a clean machine (load average < 1.0, no browsers/steam). It refuses otherwise.

| # | measurement | why | parameters | output |
|---|---|---|---|---|
| 1 | I2_S vs LUT GEMV, interleaved A/B in one binary (`TENZO_GEMV=i2s|lut`) | block A target: recover the ~19% decode loss of the two-table LUT kernel | int8_fused, prompt 128 and 512, gen 128, ABBA ×5, 4 threads cpus 0-3, models/bitnet-1.58b-v2 | `part2/bench/ab_gemv_pp{128,512}.csv` |
| 2 | e2e v3: Tenzo (i2s) vs the valid bitnet.cpp release build | the 2026-11-18 e2e used a broken bitnet.cpp and the wrong Tenzo kernel | prompts 16/128/512, gen 128, 5 reps; Tenzo fp32/int8_fused; bitnet.cpp f16/q8_0 KV | `part2/bench/e2e_fair_v3.csv` |
