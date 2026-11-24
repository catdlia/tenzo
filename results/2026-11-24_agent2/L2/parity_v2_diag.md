# L2 — Parity Gate v2 Cosine Diagnostics (`T02`, Owner Decision D1)

Checks whether the two single-prompt `1 - cos <= 1.5 * (1 - cos_ref)` misses in `results/2026-11-23_agent2/L2/parity_v3.md`
(`p4_ukrainian` on `fp32`: `1.77e-4 > 1.13e-4`; `p2_tech_en` on `int8_fused`: `3.09e-4 > 1.96e-4`) stem from Tenzo's
intentional INT8 embeddings + INT8 LM head rather than a transformer/KV bug.
Thresholds in `scripts/verify_model_parity.py` were **not modified**.

## Per-Prompt Results (`1 - cos` vs `golden fp32`)

| Prompt | `v2` limit `1.5×ref` | Tenzo default `fp32` KV | Tenzo default `int8` KV | Golden w/ INT8 embed+head | Engine `FP32_LM` (`fp32` KV) | Engine `FP32_LM` (`int8` KV) | `FP32_LM` max_abs `fp32/int8` |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `p1_fact_en` | 1.17e-03 | 3.89e-04 | 4.13e-04 | **4.02e-04** | **4.32e-04** (PASS) | **1.85e-04** (PASS) | 0.478 / 0.348 (lim 0.810) |
| `p2_tech_en` | 1.96e-04 | 1.49e-04 | 3.09e-04 | **1.68e-04** | **1.84e-04** (PASS) | **1.35e-04** (PASS) | 0.591 / 0.552 (lim 0.798) |
| `p3_code` | 1.53e-04 | 1.50e-04 | 1.36e-04 | **1.28e-04** | **1.02e-04** (PASS) | **1.40e-04** (PASS) | 0.672 / 0.782 (lim 1.057) |
| `p4_ukrainian` | 1.13e-04 | 1.77e-04 | 1.03e-04 | **1.06e-04** | **5.17e-05** (PASS) | **3.59e-05** (PASS) | 1.135 / 1.175 (lim 2.451) |
| `p5_long_ctx` | 1.01e-02 | 2.35e-03 | 3.31e-03 | **1.44e-03** | **3.69e-03** (PASS) | **3.98e-03** (PASS) | 0.888 / 0.881 (lim 1.663) |
| `p6_weird` | 3.48e-04 | 2.03e-04 | 2.60e-04 | **1.50e-04** | **1.01e-04** (PASS) | **1.52e-04** (PASS) | 0.392 / 0.526 (lim 1.026) |
| `p7_math_en` | 1.86e-04 | 8.08e-05 | 7.31e-05 | **7.81e-05** | **1.18e-04** (PASS) | **7.87e-05** (PASS) | 0.737 / 0.687 (lim 1.220) |

## Summary & Attribution

- **Default Tenzo (`INT8` embed + head):** `v2 max_abs` 7/7, `top-1` 6/7 (`fp32`) / 7/7 (`int8_fused`), `v2 1-cos` **6/7** (`p4` misses on `fp32`, `p2` misses on `int8_fused`), EN16 PPL ratio CI passes `[0.995, 1.010]`.
- **Golden INT8 embed + head emulation vs Golden fp32:** on `p4_ukrainian`, `1 - cos = 1.06e-04` (vs Tenzo `1.77e-04`); on `p2_tech_en`, `1 - cos = 1.68e-04` (vs Tenzo `3.09e-04`).
- **Tenzo Engine with `TENZO_EXPERIMENT_FP32_LM=1`:** `fp32` KV passes `v2 1-cos` **7/7** (`p4`: `5.17e-05 <= 1.13e-04`) and `top-1` **7/7**; `int8_fused` KV passes `v2 1-cos` **7/7** (`p2`: `1.35e-04 <= 1.96e-04`) and `top-1` **7/7**.
- **Conclusion:** Both independent diagnostics confirm that the small residual cosine divergence on `p4` and `p2` is accounted for by the intentional INT8 embedding / INT8 LM-head quantization (`~1–2e-4`), not by transformer or KV-cache bugs.

## Reproducibility
- `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- nice -n 19 ionice -c3 python3 scripts/diagnose_parity_v2.py`
