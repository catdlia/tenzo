# B1 Clean Idle Speed Benchmarks (`2026-11-26`)

- **Provenance:** `results/2026-11-26_agent2/B1/b1_summary.csv` (`tenzo-b1.service`, `scripts/b1_idle_runner.sh` v2, pinned CPUs `0-3`, AC online `1`, governor `powersave`, E-core `cpu4..7` `foreign_cpu_mean < 0.15`, `0` forbidden processes before/after).
- **Completed valid items:** `6/6` (all on attempt `1`, `rc=0`).

## 1. Runner Gate & Telemetry (`b1_summary.csv`, `valid` only)

| Item | Status | Attempt | `load1` (`before→after`) | `foreign_cpu_mean` | `foreign_cpu_max` | Gov | AC | Temp (°C) |
|---|---|---:|---|---:|---:|---|---:|---:|
| `ab_gemv_pp128` | `valid` | 1 | `0.90→4.89` | `0.1274` | `0.2313` | `powersave` | 1 | 50.0 |
| `ab_gemv_pp512` | `valid` | 1 | `0.92→4.43` | `0.1197` | `0.3420` | `powersave` | 1 | 50.0 |
| `e2e_fair_v3` | `valid` | 1 | `0.79→4.65` | `0.1399` | `0.5482` | `powersave` | 1 | 51.0 |
| `ab_prefill_pp128` | `valid` | 1 | `0.87→3.38` | `0.1199` | `0.2268` | `powersave` | 1 | 49.0 |
| `ab_prefill_pp512` | `valid` | 1 | `0.90→4.23` | `0.1190` | `0.3474` | `powersave` | 1 | 51.0 |
| `decode_kv_modes` | `valid` | 1 | `0.87→4.09` | `0.1140` | `0.2802` | `powersave` | 1 | 50.0 |

## 2. Kernel A/B Benchmarks (`i2s` vs `lut`, `int8_fused`, `tg=128`, `5` ABBA reps = `10` runs/variant)

| Prompt | Variant | `n` | `pp_tps` median [IQR] | `tg_tps` median [IQR] |
|---|---|---:|---:|---:|
| `pp=128` | `i2s` | 10 | `34.24` [`0.37`] | `22.19` [`0.18`] |
| `pp=128` | `lut` | 10 | `23.11` [`0.27`] | `16.97` [`0.14`] |
| `pp=128` | **`i2s / lut` speedup** | — | **`1.48×`** | **`1.31×`** |
| `pp=512` | `i2s` | 10 | `33.36` [`0.25`] | `21.26` [`0.20`] |
| `pp=512` | `lut` | 10 | `22.52` [`0.24`] | `16.38` [`0.04`] |
| `pp=512` | **`i2s / lut` speedup** | — | **`1.48×`** | **`1.30×`** |

## 3. Prefill Schedule A/B (`batched chunk=32` vs `sequential`, `int8_fused`, `5` ABBA reps = `10` runs/variant)

| Prompt | Variant | `n` | `pp_tps` median [IQR] | `tg_tps` median [IQR] |
|---|---|---:|---:|---:|
| `pp=128` | `sequential` | 10 | `36.85` [`0.35`] | `24.80` [`0.90`] |
| `pp=128` | `batched` | 10 | `54.20` [`1.70`] | `24.60` [`1.40`] |
| `pp=128` | **`batched / sequential` speedup** | — | **`1.47×`** | `0.99×` |
| `pp=512` | `sequential` | 10 | `35.00` [`0.85`] | `23.25` [`0.43`] |
| `pp=512` | `batched` | 10 | `50.75` [`0.55`] | `22.90` [`0.43`] |
| `pp=512` | **`batched / sequential` speedup** | — | **`1.45×`** | `0.98×` |

## 4. End-to-End Fair Comparison (`tenzo` token-by-token prefill vs `bitnet.cpp`, CPUs `0-3`, `threads=4`, `gen=128`, `n=5`)

| `prompt_len` | Engine | KV | `n` | `pp_tps` med [IQR] | `tg_tps` med [IQR] | `pg_tps` med [IQR] | RSS (MB) med |
|---:|---|---|---:|---:|---:|---:|---:|
| 16 | `tenzo` | `fp32` | 5 | `34.01` [`1.77`] | `22.41` [`0.84`] | `23.19` [`0.96`] | `1022.4` |
| 16 | `tenzo` | `int8_fused` | 5 | `34.71` [`0.50`] | `22.55` [`0.09`] | `23.43` [`0.06`] | `911.5` |
| 16 | `bitnet.cpp` | `f16` | 5 | `45.01` [`3.51`] | `14.58` [`0.22`] | `15.74` [`0.15`] | — |
| 16 | `bitnet.cpp` | `q8_0` | 5 | `43.27` [`4.39`] | `14.99` [`0.08`] | `16.26` [`0.10`] | — |
| 128 | `tenzo` | `fp32` | 5 | `36.42` [`0.33`] | `24.67` [`0.04`] | `29.41` [`0.20`] | `1022.1` |
| 128 | `tenzo` | `int8_fused` | 5 | `36.92` [`0.22`] | `24.50` [`0.07`] | `29.45` [`0.04`] | `911.2` |
| 128 | `bitnet.cpp` | `f16` | 5 | `53.01` [`0.96`] | `14.73` [`0.05`] | `23.20` [`0.06`] | — |
| 128 | `bitnet.cpp` | `q8_0` | 5 | `53.95` [`0.98`] | `14.86` [`0.09`] | `23.44` [`0.15`] | — |
| 512 | `tenzo` | `fp32` | 5 | `34.11` [`0.32`] | `22.53` [`0.32`] | `30.93` [`0.60`] | `1023.0` |
| 512 | `tenzo` | `int8_fused` | 5 | `35.53` [`0.27`] | `23.14` [`0.17`] | `32.09` [`0.11`] | `911.3` |
| 512 | `bitnet.cpp` | `f16` | 5 | `48.44` [`0.85`] | `14.62` [`0.16`] | `32.65` [`0.46`] | — |
| 512 | `bitnet.cpp` | `q8_0` | 5 | `47.29` [`0.79`] | `14.76` [`0.19`] | `32.31` [`0.06`] | — |

- *Note:* In `e2e_fair_v3`, `tenzo` uses token-by-token prefill (`prefill` default) while `--prefill batched` raises `pp_tps` to `~50–54 tok/s` (Table 3 above); `bitnet.cpp` `tg_tps` is measured from empty context (`llama-bench -n`), and `pg_tps` is combined prompt+generation (`-pg`).

## 5. KV-Mode Throughput Comparison (`tenzo-inference --prefill batched -c 1024`, `tg=128`, `n=3` reps)

| `pp` | KV Mode | `n` | `pp_tps` med [IQR] | `tg_tps` med [IQR] |
|---:|---|---:|---:|---:|
| 128 | `int8_fused` | 3 | `54.80` [`9.25`] | `24.60` [`2.25`] |
| 128 | `tl1_fused` | 3 | `54.50` [`9.65`] | `24.30` [`2.85`] |
| 128 | `ternary_rot` | 3 | `53.70` [`0.90`] | `24.20` [`0.10`] |
| 128 | `ternary_rot_alpha` | 3 | `51.90` [`0.35`] | `24.20` [`0.15`] |
| 512 | `int8_fused` | 3 | `52.20` [`0.25`] | `23.30` [`0.05`] |
| 512 | `tl1_fused` | 3 | `51.00` [`0.00`] | `22.70` [`0.05`] |
| 512 | `ternary_rot` | 3 | `50.10` [`0.30`] | `22.20` [`0.25`] |
| 512 | `ternary_rot_alpha` | 3 | `50.00` [`0.35`] | `22.20` [`0.00`] |
