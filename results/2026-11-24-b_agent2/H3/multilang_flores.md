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
