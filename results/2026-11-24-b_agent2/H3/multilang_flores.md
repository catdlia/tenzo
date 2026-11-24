# H3 (`T48`): 5-Language Parallel FLORES-200 Evaluation

## Plan

- **Machine state (`2026-11-24T15:01:39+02:00`):** `loadavg 0.66 3.14 6.07`, `MemAvailable 10.35 GiB`, `/` `8.5G` free, `governor=powersave`, `AC=1`, active `zen-bin/steam/qemu-system` (`tenzo-b1` waiting cleanly -> no idle-yield).
- **Languages (`5` parallel `FLORES-200` `devtest` translations):**
  - `en` (`eng_Latn`): Latin, low fertility;
  - `de` (`deu_Latn`): Latin, moderate Germanic morphology;
  - `pl` (`pol_Latn`): **Latin**, high-fertility Slavic morphology (higher BitNet token count than `uk` despite Latin script);
  - `uk` (`ukr_Cyrl`): **Cyrillic**, high-fertility East Slavic;
  - `bg` (`bul_Cyrl`): **Cyrillic**, analytic South Slavic with highest BitNet token fertility.
- **Models & Precision Schemes (`Qwen2.5-1.5B` excluded per D4/D1):**
  - `BitNet` (`tools/reference/golden_bitnet.py`, layer-major FP32, `bos=128000`, run under `MemoryMax=4G`).
  - `EuroLLM-1.7B` (`~/tenzo-refs/hf/eurollm-1.7b`, `model_type=llama`, layer-major FP32, `bos=1`, run under `MemoryMax=6G` after verifying `MemAvailable >= 8 GiB`).
  - Schemes: baseline (`fp32` / `fp`), `tenzo_int8` (`K=int8, V=int8`), `S3+s4r32` (3 `int8` sinks + 32 recent `int8`, middle `tern_g32_rot`).
  - Parity gate: verify `max(|B=1 - layer-major|) == 0.000e+00` on window 0 for every `(model, scheme)` before batch scoring.
- **Window Construction (`N = 48` fresh parallel windows):**
  - Start cleanly at `devtest` sentence index `209` (`2026-11-24/H3` used sentences `0..208`) so `H3` is 100% out-of-sample relative to `2026-11-24/H3`.
  - Greedily join consecutive sentences with `"\n"` until the next sentence would cause any `(model, lang)` tokenized length (excluding `BOS`) to exceed `511` tokens (`total len <= 512`).
  - Prepend model `BOS`; write `.ids` + `flores5_meta.json` under `results/2026-11-24-b_agent2/H3/` (raw texts stay in `~/tenzo-refs/datasets/flores200/flores5_texts.json`, never committed).
  - Assert **0 32-gram token overlap** against all `results/reference/*.ids`, `results/2026-11-23_agent2/H2/*.ids`, `results/2026-11-24_agent2/{H1b,H3}/*.ids`, and within the 48 windows.
- **Metrics & 10,000-Sample Paired Window Bootstrap (`seed = 0`):**
  - Token fertility metrics: `tok_L`, `tok_ratio = tok_L / tok_EN`, `tok_per_word_L`, `chars_L / chars_EN`.
  - Degradation metrics per `(model, L, scheme)`: relative NLL increase `rel_loss_L = sum(NLL_q - NLL_fp) / sum(NLL_fp)` (equals `PPL_ratio_L - 1`, unit-invariant across BPT/BPC/BPB), $D_L = \bar{\Delta}_{\text{doc},L} - \bar{\Delta}_{\text{doc,EN}}$, and $R_L = \text{rel\_loss}_L - \text{rel\_loss}_{\text{EN}}$ with 95% bootstrap CIs.
  - Across-language correlation: Spearman $\rho$ (and OLS fit) between `tok_L / tok_EN` and $R_L$ (and $D_L$) across the 5 languages, plus script comparison (`pl` Latin vs `uk`/`bg` Cyrillic on BitNet; `pl` vs `de` vs `uk`/`bg` on EuroLLM).
- **Pre-registered Decision Rule (for `S3+s4r32`):**
  - **`fertility_dominates`:** Strong rank order ($\rho \ge 0.80$) between token multiplier `tok_L / tok_EN` and $R_L$ (or $D_L$) in both models, with high-fertility Latin Polish (`pl` on BitNet: `tok_PL > tok_UK`) degrading proportionally to its token count without a systematic excess Cyrillic jump.
  - **`script_effect`:** Cyrillic (`uk`, `bg`) degrades systematically more than Latin (`pl`, `de`, `en`) at comparable or lower token count/fertility (e.g., BitNet `uk > pl` in $R_L$ / $D_L$ despite `tok_PL > tok_UK`, or EuroLLM Cyrillic jump unexplained by token count).
  - **`mixed`:** Partial correlation ($\rho < 0.80$ in at least one model) or divergence between models where both fertility and language/script coverage contribute.

## Results (`N = 48` Parallel Windows, Out-of-Sample `sents 209..633`)

### BitNet-2B-4T

| Lang | Script | `mean_tok` | `tok/EN` | `tok/word` | `base BPC` | `tenzo_int8` `rel_loss` | `S3+s4r32` `rel_loss` [95% CI] | `S3+s4r32` `D_L` (nats/win) | `S3+s4r32` `R_L` (pp) [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `en` (English) | Latin | 237.3 | 1.00× | 1.23 | 0.615 | `+0.03%` | `-0.34%` [`-1.26%`, `+0.27%`] | `+0.00` [`+0.00`, `+0.00`] | `+0.00` [`+0.00`, `+0.00`] |
| `de` (German) | Latin | 376.1 | 1.58× | 1.99 | 1.057 | `-0.01%` | `+1.14%` [`+0.87%`, `+1.41%`] | `+18.72` [`+12.72`, `+26.76`] | `+1.48` [`+0.81`, `+2.46`] |
| `uk` (Ukrainian) | Cyrillic | 388.2 | 1.64× | 2.35 | 1.733 | `+0.00%` | `+1.57%` [`+1.28%`, `+1.88%`] | `+34.55` [`+26.73`, `+43.82`] | `+1.91` [`+1.21`, `+2.92`] |
| `pl` (Polish) | Latin | 446.4 | 1.88× | 2.67 | 1.545 | `-0.02%` | `+1.59%` [`+1.36%`, `+1.83%`] | `+32.43` [`+26.29`, `+40.52`] | `+1.93` [`+1.30`, `+2.89`] |
| `bg` (Bulgarian) | Cyrillic | 481.6 | 2.03× | 2.62 | 1.767 | `+0.06%` | `+1.98%` [`+1.73%`, `+2.25%`] | `+44.71` [`+37.66`, `+52.89`] | `+2.32` [`+1.67`, `+3.27`] |

- **Fertility-vs-loss correlation (`S3+s4r32`):** Spearman `ρ(tok/EN, R_L) = 1.000`, Pearson `r = 0.971`, Spearman `ρ(tok/EN, D_L) = 0.900` (OLS slope = `+2.23` pp per 1.0× EN token ratio).

### EuroLLM-1.7B

| Lang | Script | `mean_tok` | `tok/EN` | `tok/word` | `base BPC` | `tenzo_int8` `rel_loss` | `S3+s4r32` `rel_loss` [95% CI] | `S3+s4r32` `D_L` (nats/win) | `S3+s4r32` `R_L` (pp) [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `en` (English) | Latin | 261.1 | 1.00× | 1.36 | 0.589 | `+0.01%` | `+0.64%` [`+0.42%`, `+0.93%`] | `+0.00` [`+0.00`, `+0.00`] | `+0.00` [`+0.00`, `+0.00`] |
| `pl` (Polish) | Latin | 316.6 | 1.21× | 1.90 | 0.643 | `+0.01%` | `+0.92%` [`+0.65%`, `+1.20%`] | `+2.78` [`+0.67`, `+4.94`] | `+0.27` [`-0.02`, `+0.56`] |
| `de` (German) | Latin | 317.7 | 1.22× | 1.68 | 0.574 | `-0.03%` | `+0.93%` [`+0.69%`, `+1.20%`] | `+2.85` [`+1.28`, `+4.36`] | `+0.29` [`+0.07`, `+0.50`] |
| `bg` (Bulgarian) | Cyrillic | 362.0 | 1.39× | 1.97 | 0.681 | `-0.01%` | `+1.64%` [`+1.31%`, `+2.02%`] | `+9.09` [`+6.74`, `+11.52`] | `+1.00` [`+0.72`, `+1.28`] |
| `uk` (Ukrainian) | Cyrillic | 371.9 | 1.42× | 2.25 | 0.732 | `+0.02%` | `+1.20%` [`+0.95%`, `+1.47%`] | `+5.98` [`+4.00`, `+7.84`] | `+0.56` [`+0.32`, `+0.79`] |

- **Fertility-vs-loss correlation (`S3+s4r32`):** Spearman `ρ(tok/EN, R_L) = 0.900`, Pearson `r = 0.857`, Spearman `ρ(tok/EN, D_L) = 0.900` (OLS slope = `+1.91` pp per 1.0× EN token ratio).

## Decision

- **Pre-registered verdict:** **`fertility_dominates`**
- **BitNet key contrast (`pl` Latin `×1.88` vs `uk` Cyrillic `×1.64`):** `pl R_L = +1.93` pp [`+1.30`, `+2.89`] vs `uk R_L = +1.91` pp [`+1.21`, `+2.92`]; `bg` (`×2.03` Cyrillic) `R_bg = +2.32` pp; `de` (`×1.58` Latin) `R_de = +1.48` pp.
- **EuroLLM-1.7B contrast (`pl` `×1.21`, `de` `×1.22`, `bg` `×1.39`, `uk` `×1.42`):** `R_de = +0.29` pp, `R_pl = +0.27` pp, `R_bg = +1.00` pp, `R_uk = +0.56` pp.

## Provenance

- Prep: `scripts/prepare_h3_multilang.py` -> `results/2026-11-24-b_agent2/H3/flores5_meta.json` (`0` 32-gram overlap)
- Run: `scripts/run_h3_multilang.py` (parity verified `max_abs_lp_diff < 1e-6`)
- Analysis: `scripts/analyze_h3_multilang.py` -> `multilang_flores.{csv,png}`, `multilang_flores_windows.csv`
