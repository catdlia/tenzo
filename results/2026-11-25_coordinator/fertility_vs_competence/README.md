# Fertility vs competence — smoke run on FLORES-5 (coordinator, 2026-11-25)

`scripts/analyze_fertility_vs_competence.py` was written **before** the 11-language data exist, so the T50 part 2 analysis is fixed in advance. These files are a smoke test of the script on the existing FLORES-5 scores (`results/2026-11-24-b_agent2/H3/scores`). With 4 non-EN languages per model the regression has one residual degree of freedom, so the script prints `too_few_languages` instead of a verdict.

**Do not interpret the coefficients.** Only two descriptive observations are worth carrying forward:

- **BitNet: the two responses order the languages differently.**
  - The relative-loss disparity R is ordered by fertility φ: Spearman 1.0 (de < uk ≈ pl < bg). Against κ it is 0.8.
  - The matched-difficulty ratio (b < 4) is ordered by κ: Spearman 1.0 (de 5.1 < pl 7.6 < uk 10.8 ≈ bg 10.9). Against φ it is 0.8.
  - So "how much more a language loses in total" and "how fragile its tokens are at equal difficulty" can have different drivers. H2 must report both responses.
- **BitNet: the fragility term of the exact decomposition** (`--response fragility`, nats per EN token of the same content; `scripts/analyze_disparity_decomposition.py`) is also ordered by κ, with Spearman 1.0: de 0.106 < pl 0.182 < uk 0.219 < bg 0.255. Against φ the Spearman is 0.8.
- **EuroLLM: log φ and log κ correlate at 0.95 (VIF 11) on these four languages**, so no regression can separate them there. The 11-language set is needed for that.

Files:
- `regression_flores5_S3_s4r32_{R,ratio_lt4,fragility}.csv`
- `languages_flores5_S3_s4r32_{R,ratio_lt4,fragility}.csv`
- `meta_*.json`
