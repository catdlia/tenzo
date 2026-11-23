# Parity Evaluation v3: Legacy Gate (`v1`) vs Reference-Calibrated Gate (`v2`)

Evaluated on `models/bitnet-1.58b-v2` after L1 (`T42`) fixes (`tenzo-inference`, 7 prompts from `tools/reference/prompts.json`, reference dumps `results/reference/{golden,hf}`), 2026-11-23, governor `powersave`.

## 1. Gate Definitions

- **Legacy Gate `v1` (default, unchanged):** `cos > 0.999`, `max_abs <= 0.25`, `top1_match == True` on every prompt.
- **Reference-Calibrated Gate `v2` (`--gate v2`):** calibrated to the empirical fp32-vs-bf16 noise floor between `tools/reference/golden_bitnet.py` (`fp32`) and HF `transformers` (`bf16`) (`max_abs = 0.532–1.634`, `cos = 0.9932–0.99992`, `1 - cos = 7.56e-5–6.76e-3`):
  1. Per prompt $p$: $\text{max\_abs}(\text{Tenzo}, \text{golden}) \le 1.5 \cdot \text{max\_abs}(\text{HF bf16}, \text{golden})$ and $(1 - \cos(\text{Tenzo}, \text{golden})) \le 1.5 \cdot (1 - \cos(\text{HF bf16}, \text{golden}))$.
  2. Aggregate top-1: $\text{top-1}(\text{Tenzo}) == \text{top-1}(\text{golden})$ on $\ge 6/7$ prompts.
  3. 16-window WikiText-2 EN paired PPL ratio $\text{PPL}(\text{Tenzo}) / \text{PPL}(\text{golden})$: 95% bootstrap CI (`10,000` draws, seed 0) inside `[0.995, 1.010]`.

## 2. Summary Across KV Modes

| KV mode | `v1` vs golden | `v1` vs HF | `v2` max_abs (7 prompts) | `v2` 1−cos (7 prompts) | `v2` top-1 | `v2` EN16 PPL ratio [95% CI] | `v2` overall |
|---|---|---|---|---|---|---|---|
| `fp32` | **FAIL** (0/7) | **FAIL** (0/7) | **7/7 PASS** | **6/7** (`p4_ukrainian` fails) | **6/7 PASS** | **1.0030** `[1.0003, 1.0057]` (**PASS**) | **FAIL** (1 prompt `1−cos`) |
| `int8_fused` | **FAIL** (0/7) | **FAIL** (0/7) | **7/7 PASS** | **6/7** (`p2_tech_en` fails) | **7/7 PASS** | **1.0039** `[1.0009, 1.0070]` (**PASS**) | **FAIL** (1 prompt `1−cos`) |
| `ternary_rot` | **FAIL** (0/7) | **FAIL** (0/7) | **5/7** (`p5`, `p6` fail) | **3/7** (`p1`, `p2`, `p5`, `p6` fail) | **7/7 PASS** | **1.0753** `[1.0593, 1.0921]` (**FAIL**) | **FAIL** (expected) |

> **Honesty note:** No thresholds in `scripts/verify_model_parity.py` were relaxed. Both `fp32` and `int8_fused` satisfy the `v2` `max_abs` check (`7/7`), the `top-1` check (`6/7` and `7/7`), and the `EN16 PPL` CI check (`[0.995, 1.010]`), yet each misses the `1.5 × (1 − cos_ref)` bound on one prompt where the `golden` vs `hf` baseline cosine happens to be very tight (`1 − cos_ref = 7.56e-5` on `p4_ukrainian` and `1.31e-4` on `p2_tech_en`). As expected, `ternary_rot` fails `v2` on both logits (`p5_long_ctx` has `cos = 0.9596`, `max_abs = 2.761` because length > 36 activates the ternary KV cache outside the 4-sink / 32-recent INT8 window) and on EN16 PPL (`1.0753` vs `<= 1.010`).

## 3. Per-Prompt `v2` Breakdown vs `golden` (`fp32`)

### 3.1 `fp32` KV (`parity_v2_fp32.log`)
| prompt | `max_abs` | limit `1.5×ref` | `1−cos` | limit `1.5×ref` | top-1 | KL | greedy | `v2` prompt |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `p1_fact_en` | 0.5378 | 0.8100 | 3.895e-04 | 1.167e-03 | True | 0.0004 | 32/32 | PASS |
| `p2_tech_en` | 0.5548 | 0.7984 | 1.490e-04 | 1.963e-04 | True | 0.0010 | 32/32 | PASS |
| `p3_code` | 0.7598 | 1.0574 | 1.496e-04 | 1.532e-04 | True | 0.0001 | 32/32 | PASS |
| `p4_ukrainian` | 2.2709 | 2.4512 | **1.773e-04** | **1.133e-04** | False | 0.0764 | 0/32 | **FAIL** |
| `p5_long_ctx` | 0.6614 | 1.6634 | 2.353e-03 | 1.013e-02 | True | 0.0042 | 32/32 | PASS |
| `p6_weird` | 0.5950 | 1.0259 | 2.034e-04 | 3.483e-04 | True | 0.0047 | 18/32 | PASS |
| `p7_math_en` | 0.5921 | 1.2202 | 8.076e-05 | 1.855e-04 | True | 0.0000 | 32/32 | PASS |

### 3.2 `int8_fused` KV (`parity_v2_int8_fused.log`)
| prompt | `max_abs` | limit `1.5×ref` | `1−cos` | limit `1.5×ref` | top-1 | KL | greedy | `v2` prompt |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `p1_fact_en` | 0.6233 | 0.8100 | 4.128e-04 | 1.167e-03 | True | 0.0003 | 23/32 | PASS |
| `p2_tech_en` | 0.7279 | 0.7984 | **3.090e-04** | **1.963e-04** | True | 0.0025 | 32/32 | **FAIL** |
| `p3_code` | 0.7871 | 1.0574 | 1.362e-04 | 1.532e-04 | True | 0.0002 | 32/32 | PASS |
| `p4_ukrainian` | 1.9497 | 2.4512 | 1.031e-04 | 1.133e-04 | True | 0.0277 | 14/32 | PASS |
| `p5_long_ctx` | 0.8819 | 1.6634 | 3.314e-03 | 1.013e-02 | True | 0.0030 | 32/32 | PASS |
| `p6_weird` | 0.6917 | 1.0259 | 2.605e-04 | 3.483e-04 | True | 0.0054 | 26/32 | PASS |
| `p7_math_en` | 0.5746 | 1.2202 | 7.314e-05 | 1.855e-04 | True | 0.0000 | 7/32 | PASS |

### 3.3 `ternary_rot` KV (`parity_v2_ternary_rot.log`)
| prompt | `max_abs` | limit `1.5×ref` | `1−cos` | limit `1.5×ref` | top-1 | KL | greedy | `v2` prompt |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `p1_fact_en` | 0.7290 | 0.8100 | **1.408e-03** | **1.167e-03** | True | 0.0003 | 23/32 | **FAIL** |
| `p2_tech_en` | 0.7144 | 0.7984 | **3.235e-04** | **1.963e-04** | True | 0.0022 | 32/32 | **FAIL** |
| `p3_code` | 0.7192 | 1.0574 | 1.199e-04 | 1.532e-04 | True | 0.0001 | 32/32 | PASS |
| `p4_ukrainian` | 1.4232 | 2.4512 | 5.981e-05 | 1.133e-04 | True | 0.0174 | 2/32 | PASS |
| `p5_long_ctx` | **2.7614** | **1.6634** | **4.039e-02** | **1.013e-02** | True | 0.0951 | 10/32 | **FAIL** |
| `p6_weird` | **1.0808** | **1.0259** | **8.117e-04** | **3.483e-04** | True | 0.0207 | 1/32 | **FAIL** |
| `p7_math_en` | 0.6513 | 1.2202 | 8.495e-05 | 1.855e-04 | True | 0.0000 | 7/32 | PASS |

## 4. Reproduction Commands
```bash
python3 scripts/verify_model_parity.py --self-test
python3 scripts/verify_model_parity.py -m models/bitnet-1.58b-v2 --sources golden,hf --gate-source golden --kv-quant <mode> --gate v1
python3 scripts/verify_model_parity.py -m models/bitnet-1.58b-v2 --sources golden,hf --gate-source golden --kv-quant <mode> --gate v2 --score-file <en16_score_txt>
```
Logs and JSONs: `results/2026-11-23_agent2/L2/parity_{v1,v2}_{fp32,int8_fused,ternary_rot}.{json,log}`.
