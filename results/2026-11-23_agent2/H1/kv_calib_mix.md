# H1.2 — EN+UK Mixed KV Calibration (`S3+s4r32` / `ternary_rot`)

Calibrated per-`(layer, head, group)` scale multipliers `(alpha_k, alpha_v)` in `[0.5, 1.5]` via `tools/reference/kv_calib.py` (`100` Adam steps/layer, matching attention output MSE) on three disjoint calibration sets (verified `0` shared 32-grams against `EN16` and `UK32`):

- `alpha_en`: `results/reference/calib_en8.ids` (`8 × 512`, WikiText-2 train)
- `alpha_uk`: `results/reference/calib_uk8.ids` (`8 × 512`, Kobza train rows `100..199`)
- `alpha_mix`: `calib_en8.ids` + `calib_uk8.ids` (`16 × 512` mixed)

Run log: `results/2026-11-23_agent2/H1/calib/calib_mix.log` (`1414s`, peak `2.25 GB` RSS under `MemoryMax=3G`, `nice -n 19 ionice -c3`), exported to `results/2026-11-23_agent2/H1/alpha_{en,uk,mix}.txt` (`600` `(l, h, g)` rows each).

## Alpha Distribution & Cross-Language Alignment

| Calibration | `alpha_k` mean ± std [min, max] (`>1`) | `alpha_v` mean ± std [min, max] (`>1`) |
| --- | --- | --- |
| `en` | `1.0821 ± 0.1255` `[0.5801, 1.5000]` (`78.5%`) | `1.0154 ± 0.1235` `[0.5000, 1.5000]` (`57.5%`) |
| `uk` | `1.0948 ± 0.1269` `[0.5708, 1.5000]` (`80.3%`) | `0.9767 ± 0.1225` `[0.5000, 1.4910]` (`40.3%`) |
| `mix` | `1.0915 ± 0.1227` `[0.5601, 1.5000]` (`80.8%`) | `1.0013 ± 0.1209` `[0.5000, 1.5000]` (`53.2%`) |

Cross-language Pearson correlations across the `600` `(l, h, g)` groups:
- `corr(en, uk)`: `r = 0.6737` (`alpha_k`), `r = 0.8547` (`alpha_v`)
- `corr(mix, en)`: `r = 0.8555` (`alpha_k`), `r = 0.9481` (`alpha_v`)
- `corr(mix, uk)`: `r = 0.8788` (`alpha_k`), `r = 0.9455` (`alpha_v`)

## Simulator PPL (`golden_bitnet.py`, fp32) — EN16 & UK32

| Scheme | EN16 PPL | EN16 Ratio vs Golden [95% CI] | EN16 dNLL [95% CI] | UK32 PPL | UK32 Ratio vs Golden [95% CI] | UK32 dNLL [95% CI] | Pooled `c_UK` [95% CI] |
| --- | ---: | --- | --- | ---: | --- | --- | --- |
| `golden fp32` | `19.066` | `1.0000` (ref) | `—` | `118.217` | `1.0000` (ref) | `—` | `—` |
| `S3+s4r32` (uncal) | `20.419` | `1.0709 [1.0547, 1.0883]` | `+0.0685 [+0.0531, +0.0843]` | `136.094` | `1.1512 [1.1281, 1.1749]` | `+0.1408 [+0.1207, +0.1610]` | `+0.1126 [+0.0476, +0.1901]` |
| `calib EN` | `19.828` | `1.0399 [1.0296, 1.0508]` | `+0.0392 [+0.0290, +0.0495]` | `133.124` | `1.1261 [1.1072, 1.1452]` | `+0.1188 [+0.1022, +0.1359]` | `+0.1070 [+0.0505, +0.1637]` |
| `calib UK` | `19.997` | `1.0488 [1.0367, 1.0619]` | `+0.0476 [+0.0359, +0.0601]` | `131.997` | `1.1166 [1.0970, 1.1360]` | `+0.1103 [+0.0930, +0.1272]` | `+0.1096 [+0.0550, +0.1676]` |
| `calib EN+UK mix` | `19.837` | `1.0404 [1.0292, 1.0522]` | `+0.0396 [+0.0286, +0.0507]` | `131.891` | `1.1157 [1.0982, 1.1332]` | `+0.1094 [+0.0940, +0.1251]` | `+0.1132 [+0.0678, +0.1637]` |

## Findings
- Mixed `EN+UK` calibration achieves **nearly identical EN PPL to EN-only calibration** (`×1.0404` vs `×1.0399`, eliminating the `+0.0089` EN regression of UK-only calibration) and **matches or slightly improves UK PPL relative to UK-only calibration** (`×1.1157` vs `×1.1166`).
- Relative to uncalibrated `S3+s4r32`, `alpha_mix` reduces KV degradation (`dNLL`) by **42.2%** on EN16 (`+0.0685 → +0.0396` nats/tok) and **22.3%** on UK32 (`+0.1408 → +0.1094` nats/tok), while preserving the positive baseline-adjusted language disparity `c_UK > 0`.
