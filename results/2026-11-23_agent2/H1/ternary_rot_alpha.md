# H1.3 — `ternary_rot` with Calibrated Alphas (`TENZO_KV_ALPHA`) in TenzoEngine

Evaluated `tenzo-inference` (`commit 11d3bac`, CPU I2_S, `governor=powersave`) with `TENZO_KV_ALPHA=/app/results/2026-11-23_agent2/H1/alpha_mix.txt` across `EN16` (`results/reference/windows.ids`) and `UK32` (`results/reference/windows_uk32.ids`).

## Summary Table

| Eval Set | Mode | Engine PPL [95% CI] | Engine Ratio vs `int8_fused` [95% CI] | Sim PPL (`calib_mix`) | Raw `eng/sim` Ratio [95% CI] | Baseline-Adj `eng/sim` Ratio [95% CI] | Peak RSS |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| **EN16** (`16×512`) | `int8_fused` | `19.141 [16.231, 22.510]` | `1.0000` (ref) | `—` | `—` | `—` | `—` |
| **EN16** (`16×512`) | `ternary_rot` (uncal) | `20.502 [17.508, 23.962]` | `1.0711 [1.0547, 1.0879]` | `20.419` | `1.0041 [1.0007, 1.0077]` | `1.0002 [0.9957, 1.0046]` (`True`) | `886.3 MB` |
| **EN16** (`16×512`) | **`ternary_rot + alpha_mix`** | **`19.839 [16.892, 23.217]`** | **`1.0365 [1.0256, 1.0477]`** | **`19.837`** | **`1.0001 [0.9963, 1.0043]` (`True`)** | **`0.9962 [0.9912, 1.0013]` (`True`)** | **`886.4 MB`** |
| **UK32** (`32×512`) | `int8_fused` | `118.557 [102.629, 136.704]` | `1.0000` (ref) | `—` | `—` | `—` | `897.6 MB` |
| **UK32** (`32×512`) | **`ternary_rot + alpha_mix`** | **`132.327 [114.928, 152.113]`** | **`1.1162 [1.0988, 1.1333]`** | **`131.891`** | **`1.0033 [0.9988, 1.0081]` (`True`)** | **`1.0004 [0.9953, 1.0059]` (`True`)** | **`886.5 MB`** |
| **UK8** (`8×512`) | `int8_fused` | `142.196 [111.974, 176.319]` | `1.0000` (ref) | `—` | `—` | `—` | `—` |
| **UK8** (`8×512`) | `ternary_rot` (uncal) | `165.634 [130.755, 202.617]` | `1.1648 [1.1288, 1.2008]` | `164.766` | `1.0053 [0.9966, 1.0144]` | `1.0033 [0.9930, 1.0124]` (`True`) | `—` |
| **UK8** (`8×512`) | **`ternary_rot + alpha_mix`** | **`160.159 [126.663, 196.382]`** | **`1.1263 [1.1051, 1.1476]`** | **`159.267`** | **`1.0056 [0.9990, 1.0130]` (`True`)** | **`1.0036 [0.9942, 1.0123]` (`True`)** | **`—`** |

## Key Takeaways
- Loading `TENZO_KV_ALPHA=.../alpha_mix.txt` into `TenzoEngine` reduces KV-induced PPL degradation (`vs int8_fused`) from **`×1.0711` → `×1.0365`** on EN16 (**-48.6%** relative excess PPL) and from **`×1.1648` → `×1.1263`** on UK8 (and reaches **`×1.1162`** on full UK32).
- Engine/simulator parity holds strictly across both languages: both raw (`1.0001`, `1.0033`, `1.0056`) and baseline-adjusted (`0.9962`, `1.0004`, `1.0036`) 95% bootstrap CIs contain `1.0000`.
- Peak RSS stays flat at **`886.5 MB`** (`~865 MiB`).

## Reproduction
```bash
docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread \
  -e TENZO_KV_ALPHA=/app/results/2026-11-23_agent2/H1/alpha_mix.txt dev \
  nice -n 19 ionice -c3 /app/cmake-build-debug/tenzo-inference \
  -m /app/models/bitnet-1.58b-v2 --kv-quant ternary_rot -c 512 \
  --score /app/results/reference/windows.ids \
  --score-out /app/results/2026-11-23_agent2/H1/engine/ternary_rot_alpha_mix_en16.txt
```
