# H3b (Conditional): 5-Language Equal-Character Recent Window (`S3+s4c120`)

## Plan

- **Trigger:** Conditionally activated per coordinator protocol because `H1` (`T47`) concluded **`PARTIAL`** (`~22%` reduction in UK-minus-EN `R`, leaving `R = +0.98% [+0.55%, +1.40%] > 0`).
- **Machine state (`2026-11-25T03:26:16+02:00`):** `loadavg 2.69 3.81 4.01`, `MemAvailable 9.58 GiB`, `/` `8.4G` free, `governor=powersave`, `AC=1`, active `zen-bin` (`tenzo-b1` waiting cleanly -> no idle-yield).
- **Dataset & Reused Scores (`N = 48` parallel 5-language `FLORES-200` windows from `H3`):**
  - Token IDs: `results/2026-11-24-b_agent2/H3/flores5_{bitnet,eurollm}_{en,uk,pl,bg,de}.ids` (`48` windows, `0` 32-gram overlap).
  - Raw character offsets verified `48/48` across all `5` languages × `2` models (`480/480` exact match against `~/tenzo-refs/datasets/flores200/flores5_texts.json`).
  - Baselines reused directly from `results/2026-11-24-b_agent2/H3/scores/{bitnet,eurollm}/` (`fp32`/`fp` and `S3_s4r32`).
- **Character-Window Scheme (`S3+s4c120`, $C^* = 120$):**
  - Keep `4` sink tokens (`0..3`), current query `t`, and all prior non-sink keys/values `s` with character span `E_t - c_s <= 120` in `int8_absmax`; all older non-sink keys/values in `tern_g32_rot`.
  - Run on both `BitNet-2B-4T` (`MemoryMax=4G`, `MemAvailable >= 6 GiB`) and `EuroLLM-1.7B` (`MemoryMax=6G`, `MemAvailable >= 8 GiB`) after B=1 parity gate (`< 1e-5`).
- **Metrics & Bootstrap (`10,000` paired window samples, `seed = 0`):**
  - Effective coverage/budget: `cov_chars_exact`, `r_eff`, `bits_win`, `bits_at_512`.
  - Quality parity per `(model, L)`: `rel_loss`, $D_L(C^*=120)$, $R_L(C^*=120)$ vs $R_L(r=32)$, paired reduction $R_L(r=32) - R_L(C^*=120)$ with 95% CI, fraction reduced `1 - R_L(C*) / R_L(r=32)`, and across-language Spearman $\rho(\text{tok}_L / \text{tok}_{\text{EN}}, R_L)$.
- **Pre-registered Decision Rule:**
  - **`explains`:** Mean non-EN `R_L(C*)` drops by $> 2/3$ relative to `r=32` AND `95% CI(R_L(C*))` spans zero on $\ge 3/4$ non-EN languages in both models.
  - **`partially_explains`:** `C*=120` significantly reduces `R_L` (`R_L(r=32) - R_L(C*) > 0`) on non-EN languages across models (`~15–65%` reduction) while substantial excess non-EN disparity (`R_L > 0`) persists.
  - **`does_not_explain`:** Equal-character coverage fails to consistently reduce `R_L` across the 4 non-EN languages (or degrades overall disparity).

## Results (`N = 48` Parallel Windows, `S3+s4c120` vs `S3+s4r32`)

### BitNet-2B-4T

| Lang | `tok/EN` | `cov_chars` | `r_eff` | `bits_win` | `bits@512` | `R(r=32)` (pp) | `R(C*=120)` (pp) [95% CI] | `ΔR = r32 - c120` [95% CI] | `frac_red` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `en` (Lat) | 1.00× | 117.3 | 23.0 | 3.136 | 2.796 | `+0.00` | `+0.00` [`+0.00`, `+0.00`] | `+0.00` [`+0.00`, `+0.00`] | — |
| `de` (Lat) | 1.58× | 118.2 | 31.5 | 3.029 | 2.890 | `+1.48` | `+1.28` [`+0.64`, `+2.23`] | `+0.19` [`+0.06`, `+0.32`] | 12.9% |
| `uk` (Cyr) | 1.64× | 118.5 | 37.3 | 3.098 | 2.954 | `+1.91` | `+1.51` [`+0.84`, `+2.48`] | `+0.40` [`+0.27`, `+0.54`] | 21.0% |
| `pl` (Lat) | 1.88× | 118.8 | 41.3 | 3.070 | 2.998 | `+1.93` | `+1.50` [`+0.90`, `+2.44`] | `+0.43` [`+0.29`, `+0.56`] | 22.3% |
| `bg` (Cyr) | 2.03× | 118.9 | 45.2 | 3.074 | 3.041 | `+2.32` | `+1.66` [`+1.00`, `+2.61`] | `+0.66` [`+0.54`, `+0.78`] | 28.4% |

- **Aggregate non-EN parity (`de, pl, uk, bg`):** Mean `R_L` shifts `+1.91` -> `+1.49` pp (**22.0%** mean reduction; `ΔR > 0` significantly on `4/4` languages; `95% CI(R_c120)` spans 0 on `0/4`; Spearman `ρ(tok/EN, R_L)`: `1.000` -> `0.900`).

### EuroLLM-1.7B

| Lang | `tok/EN` | `cov_chars` | `r_eff` | `bits_win` | `bits@512` | `R(r=32)` (pp) | `R(C*=120)` (pp) [95% CI] | `ΔR = r32 - c120` [95% CI] | `frac_red` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `en` (Lat) | 1.00× | 117.5 | 25.4 | 3.132 | 2.824 | `+0.00` | `+0.00` [`+0.00`, `+0.00`] | `+0.00` [`+0.00`, `+0.00`] | — |
| `pl` (Lat) | 1.21× | 117.8 | 29.5 | 3.094 | 2.868 | `+0.27` | `+0.17` [`-0.12`, `+0.46`] | `+0.10` [`-0.03`, `+0.25`] | 38.2% |
| `de` (Lat) | 1.22× | 117.7 | 26.8 | 3.043 | 2.838 | `+0.29` | `+0.38` [`+0.16`, `+0.61`] | `-0.09` [`-0.23`, `+0.05`] | -32.2% |
| `bg` (Cyr) | 1.39× | 118.2 | 34.2 | 3.092 | 2.919 | `+1.00` | `+0.67` [`+0.40`, `+0.95`] | `+0.33` [`+0.19`, `+0.46`] | 32.6% |
| `uk` (Cyr) | 1.42× | 118.4 | 35.9 | 3.102 | 2.938 | `+0.56` | `+0.16` [`-0.10`, `+0.39`] | `+0.40` [`+0.23`, `+0.57`] | 71.1% |

- **Aggregate non-EN parity (`de, pl, uk, bg`):** Mean `R_L` shifts `+0.53` -> `+0.35` pp (**34.7%** mean reduction; `ΔR > 0` significantly on `2/4` languages; `95% CI(R_c120)` spans 0 on `2/4`; Spearman `ρ(tok/EN, R_L)`: `0.900` -> `0.400`).

## Decision

- **Pre-registered verdict:** **`partially_explains`**
- **Interpretation:** Equalizing physical character coverage (`~118` chars across all 5 languages) reduces non-EN excess relative loss by **22.0%** (BitNet) and **34.7%** (EuroLLM-1.7B), showing token-count windows account for part of the cross-lingual disparity while residual gaps persist at ternary middle positions.

## Provenance

- Run: `scripts/run_h3b_char_window.py` (`BitNet` & `EuroLLM` B=1 parity verified)
- Analysis: `scripts/analyze_h3b_char_window.py` -> `multilang_char_window.{csv,png}`, `multilang_char_window_windows.csv`
