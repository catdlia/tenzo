# Speed after fix 2 (8-bit activations, two-table LUT kernel)

**Caveat:** the machine was heavily loaded by processes outside this run during all of tonight's speed runs (D11). The day-to-day comparison below is dominated by that drift (control point bitnet.cpp decode 0.38x); see the A/B section at the end for the kernel effect.

Medians of 5 repetitions, 4 threads on cpus 0-3, gen 128 (scripts/run_e2e_fair.py). before = 2026-11-18 CSV (6-bit kernel), after = tonight. Control point bitnet.cpp f16 prompt 128 measures machine drift between the days.

| control point bitnet.cpp f16, prompt 128 | before | after | after/before (drift) |
|---|---|---|---|
| pp tok/s | 70.03 | 46.07 | 0.658 |
| tg tok/s | 19.01 | 7.31 | 0.384 |
| pg tok/s | 29.11 | 12.21 | 0.419 |

| engine | kv | prompt | metric | before tok/s | after tok/s | after/before | drift-corrected |
|---|---|---|---|---|---|---|---|
| tenzo | fp32 | 16 | pp | 35.24 | 12.27 | 0.348 | 0.529 |
| tenzo | fp32 | 16 | tg | 23.63 | 7.32 | 0.310 | 0.806 |
| tenzo | fp32 | 16 | pg | 24.52 | 7.67 | 0.313 | 0.746 |
| tenzo | fp32 | 128 | pp | 35.16 | 10.55 | 0.300 | 0.456 |
| tenzo | fp32 | 128 | tg | 22.90 | 8.26 | 0.361 | 0.938 |
| tenzo | fp32 | 128 | pg | 27.71 | 9.88 | 0.357 | 0.850 |
| tenzo | fp32 | 512 | pp | 33.43 | 7.94 | 0.238 | 0.361 |
| tenzo | fp32 | 512 | tg | 21.40 | 6.33 | 0.296 | 0.769 |
| tenzo | fp32 | 512 | pg | 30.04 | 7.76 | 0.258 | 0.616 |
| tenzo | int8_fused | 16 | pp | 35.03 | 10.26 | 0.293 | 0.445 |
| tenzo | int8_fused | 16 | tg | 23.56 | 8.11 | 0.344 | 0.895 |
| tenzo | int8_fused | 16 | pg | 24.46 | 7.16 | 0.293 | 0.698 |
| tenzo | int8_fused | 128 | pp | 35.47 | 10.70 | 0.302 | 0.458 |
| tenzo | int8_fused | 128 | tg | 23.24 | 7.18 | 0.309 | 0.803 |
| tenzo | int8_fused | 128 | pg | 28.06 | 9.49 | 0.338 | 0.807 |
| tenzo | int8_fused | 512 | pp | 34.49 | 9.87 | 0.286 | 0.435 |
| tenzo | int8_fused | 512 | tg | 22.36 | 7.07 | 0.316 | 0.822 |
| tenzo | int8_fused | 512 | pg | 31.09 | 9.10 | 0.293 | 0.698 |

| engine | kv | prompt | RSS MB before | RSS MB after |
|---|---|---|---|---|
| bitnet.cpp | f16 | 128 | 1228 | 1227 |
| tenzo | fp32 | 16 | 2778 | 2779 |
| tenzo | fp32 | 128 | 2778 | 2779 |
| tenzo | fp32 | 512 | 2778 | 2778 |
| tenzo | int8_fused | 16 | 2667 | 2667 |
| tenzo | int8_fused | 128 | 2667 | 2667 |
| tenzo | int8_fused | 512 | 2667 | 2667 |

## A/B on the same machine at the same time (the number to use)

Old kernel = commit c0db56f (theta fix only, 6-bit LUT), new = HEAD (8-bit two-table LUT). Both built from `git archive` with
the same CMake config (Debug, which compiles the runtime with -O3; native; fast-math) into cmake-build-debug/ab/, model dir
models/bitnet-1.58b-v2, int8_fused, prompt 128, gen 128, 4 threads on cpus 0-3. Order ABBA x 5 (`scripts/ab_kernel_speed.sh`),
ratio per block = mean(new)/mean(old), summary = median over the 5 blocks. Raw rows: `ab_kernel_speed.csv`.

| metric | old median tok/s | new median tok/s | new/old (median of paired) | min | max |
|---|---|---|---|---|---|
| decode | 9.94 | 7.35 | 0.807 | 0.537 | 1.064 |
| prefill (token by token) | 16.02 | 9.81 | 0.684 | 0.417 | 0.952 |

**Verdict:** decode is about 19% slower and prefill about 32% slower with the 8-bit kernel. That exceeds the 10% bar, so per the task
the kernel is NOT rewritten tonight; design note: `design_i2s_kernel.md` (maddubs on u = w+1 in {0,1,2}, subtract sum(a)).
The spread is large (min 0.54 / max 1.06 for decode) because of background load not owned by this run (firefox / neko / a
claude process, load average 8-14, `logs/load_*.txt`). Absolute tok/s tonight are 2-3x below 2026-11-18 for BOTH engines
(the bitnet.cpp control point fell to 0.38x in decode), so the absolute numbers above must not be compared with 2026-11-18;
only the paired ratio is meaningful.
