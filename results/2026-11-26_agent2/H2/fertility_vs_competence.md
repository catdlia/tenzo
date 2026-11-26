# H2 (T50 Part 2): Fertility vs Competence Across 11 Parallel FLORES-200 Languages

## Plan

### Research Question
On fixed parallel content where both tokenizer fertility (`φ_L`) and model content cost (`κ_L`, inverse competence) vary across languages, which factor drives KV-cache quantization disparity (`S3+s4r32`) in `BitNet-b1.58-2B-4T` (`golden_bitnet.py`) and `EuroLLM-1.7B`?

### Data & Window Definition (`N = 48`)
- **Corpus:** `FLORES-200 devtest` (`~/tenzo-refs/datasets/flores200/flores200_dataset.tar.gz`).
- **Windows:** Exact `N = 48` windows from `results/2026-11-24-b_agent2/H3/flores5_meta.json` using `windows[i].sent_start .. windows[i].sent_end` (sentences `209..633`), **without** re-running the greedy length packer so boundaries remain fixed and all 5 prior languages are reused without recomputation.
- **11 Languages:**
  - **5 reused from `2026-11-24-b` H3 (`flores5`):** `en` (`eng_Latn`), `de` (`deu_Latn`), `pl` (`pol_Latn`), `uk` (`ukr_Cyrl`), `bg` (`bul_Cyrl`).
  - **6 new:**
    - `fr` (`fra_Latn`), `es` (`spa_Latn`) — high-resource Latin
    - `cs` (`ces_Latn`), `hu` (`hun_Latn`), `fi` (`fin_Latn`) — lower-resource Latin (`hu`, `fi` agglutinative)
    - `el` (`ell_Grek`) — non-Latin Greek script
- **Invariants verified prior to evaluation:**
  - Reconstructed `.ids` for `{en,de,pl,uk,bg} × {bitnet,eurollm}` MUST be byte-for-byte identical to `results/2026-11-24-b_agent2/H3/flores5_{model}_{lang}.ids`.
  - Context limit `ctx = 1024`; any window with `> 1023` scored tokens is dropped for that language and reported explicitly (`max_scored_tokens <= 605` expected -> `0` dropped).
  - `0` 32-gram overlap against calibration sets, prior non-FLORES5 eval windows, and within-language self-windows. No raw FLORES text committed.

### Models, Schemes & Execution Protocol
- **Models (`D4`):** `BitNet-b1.58-2B-4T` (`golden_bitnet.py`, `MemoryMax=4G`) and `EuroLLM-1.7B` (`MemoryMax=6G`), executed sequentially under `systemd-run --user --scope -p MemoryMax=... -p MemorySwapMax=0 -- nice -n 19 ionice -c3` (verifying `MemAvailable >= MemoryMax + 2 GB` before each run).
- **Schemes:** baseline (`fp32` for BitNet, `fp` for EuroLLM) and `S3+s4r32` (`tenzo_int8` omitted as near-zero control; `Qwen2.5-1.5B` excluded per `D4`).
- **Parity check:** B=1 logit/log-prob parity on first window (`< 1e-6` BitNet, `< 1e-5` EuroLLM).
- **Staged run order (`D5` hard cutoff: start nothing after `08:30` or finishing past `08:45`):**
  1. **Stage 1 (`fr, cs, hu, el`):** BitNet (`fr,cs,hu,el`), then EuroLLM-1.7B (`fr,cs,hu,el`). Yields `8` non-EN languages (`>= 6` minimum for OLS verdict).
  2. **Stage 2 (`es`, then `fi`, clock-gated):** each language run sequentially across BitNet then EuroLLM ONLY if estimated finish is `<= 08:45` and start is `< 08:30`, ensuring no half-model language enters `flores11`.

### Metrics & Statistical Analysis
- **Statistical convention:** All pooled `ΣΔ/ΣB`; `10,000` paired window bootstrap samples (`seed = 0`).
- **Predictors & Metrics (per language `L` and model):**
  - `φ_L = tok_L / tok_EN` (tokenizer fertility)
  - `κ_L = B_L / B_EN` (content cost in nats / inverse model competence)
  - Matched-difficulty metrics via `scripts/analyze_matched_difficulty.py` (after adding `"flores11"` to `SETS`)
  - Exact 3-way disparity decomposition (`fertility`, `composition`, `fragility`) via `scripts/analyze_disparity_decomposition.py` (after adding `"flores11"` to `SETS`)
- **Primary & Secondary Regressions (`scripts/analyze_fertility_vs_competence.py --set flores11` unchanged):**
  - **Primary:** `--response R` (`R_L` excess relative loss vs EN, pp)
  - **Secondary:** `--response ratio_lt4` (matched-difficulty ratio on `b < 4`)
  - **Third:** `--response fragility` (decomposition fragility term in nats per EN token)
  - Per model, fits OLS `y_L ~ log φ_L + log κ_L` across non-EN languages with standardized coefficients, `95%` paired window bootstrap CIs, `VIF`, and Spearman rank correlations `ρ`.

### Decision Rule (Pre-registered)
Applied separately per model and response (`N_non_en < 6` -> `too_few_languages`):
- **`tokenization`:** `95% CI(β_logφ) > 0` AND `0 ∈ 95% CI(β_logκ)` (with `VIF <= 5`)
- **`competence`:** `95% CI(β_logκ) > 0` AND `0 ∈ 95% CI(β_logφ)` (with `VIF <= 5`)
- **`both`:** both `95% CI(β_logφ) > 0` AND `95% CI(β_logκ) > 0` (with `VIF <= 5`)
- **`collinear`:** `VIF > 5` OR both CIs include `0`
- All three responses (`R`, `ratio_lt4`, `fragility`) will be reported transparently regardless of whether they agree.
