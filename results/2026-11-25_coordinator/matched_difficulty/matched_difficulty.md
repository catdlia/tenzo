# Matched-difficulty analysis of KV-quantization damage (T50, part 1)

_Coordinator re-analysis, 2026-11-25. No new model runs: every number comes from score files already committed by the agent (`lp` lines). Reproduce with:_

```
python3 scripts/analyze_matched_difficulty.py      # CSVs + meta.json (10,000 paired bootstrap reps, seed 0)
python3 scripts/plot_matched_difficulty.py         # matched_bins.png, matched_ratio_vs_kappa.png
```

Inputs (sha256 prefixes in `meta.json`):
- FLORES-32: `results/2026-11-24_agent2/H3/scores/{bitnet,eurollm,qwen}`;
- FLORES-5: `results/2026-11-24-b_agent2/H3/scores/{bitnet,eurollm}`;
- NTREX-24: `results/2026-11-24-b_agent2/H4/scores/{bitnet,eurollm}`.

## Plan

Written in `prompt agent2_2026-11-26` before the full computation. A coarse 3-bin preview had been seen, so treat the result as confirmatory for the direction and exploratory for the size.

- **Per token:**
  - b = −lp_base — base NLL, i.e. difficulty before KV quantization;
  - d = lp_base − lp_scheme — extra NLL from the KV scheme.
- **Bins of b:** [0, 0.05), [0.05, 0.3), [0.3, 1), [1, 2), [2, 4), [4, 7), [7, ∞).
- **Matched mean.** For a language L, matched_L = Σ_k w_EN,k · mean_d_L,k: the per-bin means of L, reweighted to the EN bin shares of the same corpus and model. matched_L − matched_EN is the extra damage at equal difficulty. matched_L / matched_EN is the ratio, reported only where the EN matched mean has CI > 0.
- **`lt4` variant:** only b < 4. In the high-surprise tail, ternary noise *lowers* NLL, because it flattens the predictive distribution and the mass leaks to surprising tokens. That tail dilutes and can flip the "all bins" mean, especially for EN on FLORES.
- **Bootstrap:** windows are aligned across languages (parallel corpora) and across models within a corpus, so the bootstrap resamples window ids jointly (paired).
- **Decision rule (from the prompt):** "competence matters" if
  1. for BitNet the matched UK − EN difference has CI > 0, **and**
  2. the BitNet UK/EN matched ratio is significantly larger than EuroLLM's (bootstrap of the difference of ratios).

## Result: rule satisfied → **competence matters**

### Per-bin extra NLL, S3+s4r32 (`matched_bins.png`)

Ukrainian tokens lose more than English tokens **in every difficulty bin below b = 4**.
- **BitNet:** the gap is large — 6–24× in the bins below b = 4 on FLORES, 4–10× on NTREX.
- **EuroLLM:** the gap is ~1.4–1.7×.
- **High-NLL tail:** both languages get *negative* d (flattening).

### UK vs EN at equal difficulty, S3+s4r32

| Corpus | Model | Δ UK − EN, all bins (nats/tok) | Ratio UK/EN, b<4 | R (relative loss, pp) |
|---|---|---|---|---|
| FLORES-32 | BitNet | +0.106 [+0.083, +0.131] | 7.5 [5.5, 11.2] | +1.22 [+0.83, +1.64] |
| FLORES-32 | EuroLLM | +0.011 [−0.001, +0.023] | 1.60 [1.17, 2.41] | +0.47 [+0.04, +0.91] |
| FLORES-32 | Qwen2.5 | +0.411 [+0.288, +0.528] | 1.79 [1.56, 2.08] | +31.6 [+27.1, +36.0] |
| NTREX-24 | BitNet | +0.195 [+0.173, +0.221] | 4.77 [4.08, 5.64] (all bins: 4.79) | +1.51 [+1.06, +2.00] |
| NTREX-24 | EuroLLM | +0.031 [+0.021, +0.041] | 1.47 [1.30, 1.66] (all bins: 1.51) | +1.38 [+0.96, +1.85] |

- **EN matched means:**
  - all bins — BitNet FLORES-32 +0.0013 [−0.0065, +0.0094]; NTREX +0.051 (BitNet) and +0.060 (EuroLLM);
  - b < 4 — +0.019 / +0.024 on FLORES-32, +0.063 / +0.090 on NTREX.
- **Cross-model test** (BitNet ratio − EuroLLM ratio, joint bootstrap):
  - NTREX all bins: +3.28 [+2.57, +4.22];
  - NTREX b < 4: +3.30 [+2.64, +4.12];
  - FLORES-32 b < 4: +5.89 [+3.86, +9.45].
- **INT8 KV (`tenzo_int8`) control:** matched UK − EN ≈ 0 for BitNet and EuroLLM (|Δ| ≤ 0.004). Qwen is the exception, with ratio ~2.5–3.0 — consistent with its known K outliers.

### Five languages, FLORES-5 (N = 48), S3+s4r32 (`matched_ratio_vs_kappa.png`)

- φ — tokens of L / tokens of EN on the same content (fertility).
- κ — base nats of L / base nats of EN on the same content: how much more the model "pays" for the same meaning, our competence proxy.

| Model | Lang | φ | κ | Ratio vs EN at equal difficulty (b<4) | R (pp) |
|---|---|---|---|---|---|
| BitNet | de | 1.58 | 2.01 | 5.1 [3.2, 10.7] | +1.48 [+0.80, +2.46] |
| BitNet | pl | 1.88 | 2.64 | 7.6 [4.8, 15.9] | +1.93 [+1.29, +2.89] |
| BitNet | uk | 1.64 | 2.86 | 10.8 [6.6, 22.9] | +1.91 [+1.21, +2.91] |
| BitNet | bg | 2.03 | 2.98 | 10.9 [6.9, 22.9] | +2.32 [+1.68, +3.25] |
| EuroLLM | de | 1.22 | 1.14 | 1.31 [1.12, 1.55] | +0.29 [+0.08, +0.49] |
| EuroLLM | pl | 1.21 | 1.14 | 1.42 [1.19, 1.72] | +0.27 [−0.01, +0.56] |
| EuroLLM | uk | 1.42 | 1.26 | 1.71 [1.45, 2.03] | +0.56 [+0.33, +0.79] |
| EuroLLM | bg | 1.39 | 1.20 | 1.68 [1.45, 1.97] | +1.00 [+0.72, +1.28] |

**Within BitNet**, the per-token matched ratio follows κ (de < pl < uk < bg, Spearman 1.0) better than φ (Spearman 0.8: uk has lower fertility than pl, yet higher κ and a higher ratio). With 4 languages this is suggestive, not a test. The 11-language run (T50 part 2) is what can separate them.

## Interpretation

1. **Two components.** The UK/EN disparity has two separable components:
   - (a) more tokens per unit of content, combined with token-defined cache policies (T47: a character-defined window removes 22–35%);
   - (b) **higher per-token fragility at equal difficulty**, which scales with how poorly the model knows the language: BitNet κ ≈ 2.9 → ratio 5–11×; EuroLLM κ ≈ 1.2 → 1.3–1.7×.
2. **Fertility alone does not explain (b) across models.** Qwen has the highest UK fertility (φ = 2.53) but a matched ratio of only 1.6–1.8, similar to EuroLLM's.
3. **Confound.** BitNet differs from EuroLLM not only in Ukrainian competence but also in architecture: native 1.58-bit weights, a different training mix. Cross-model ratios therefore cannot attribute (b) to competence alone. The within-model language ordering in BitNet (κ-ordered) is the cleaner evidence; T50 part 2 extends it to 10 non-EN languages.

## Caveats

- **Not comparable across models:** per-token units differ between tokenizers. Only within-model differences and unitless ratios are compared.
- **Ratios on FLORES "all bins" are undefined** for BitNet and EuroLLM because EN damage ≈ 0 there (short multi-article windows). The b < 4 variant and NTREX are the informative ones.
- **Statistics:** N = 24–48 windows, a single scheme family (S3 / S13 / r128). Bootstrap CIs on ratios are wide and right-skewed.
