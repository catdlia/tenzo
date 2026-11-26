# Results — draft 0 (coordinator, 2026-11-25)

_Prose draft of the Results section for the KV-disparity paper. Numbers come from the run reports and the coordinator's re-analyses. **Audit status (2026-11-26):** `[check]` flags pooled-vs-mean or draft-vs-source differences resolved into `results/paper/tables.{md,tex}` (`2026-11-24-H3` and `2026-11-24-b-H1` used per-window means; `2026-11-24-b-H3`, `2026-11-24-b-H3b`, `2026-11-24-b-H4`, and `2026-11-26-H2` use pooled `ΣΔ/ΣB`). `[TODO]` = experiment pending; `[claim …]` = source._

## 4.1 The gap exists on content-matched text

**INT8 KV is essentially lossless in every model and language we test**, with one exception: Qwen2.5-1.5B, which shows a small INT8 penalty [claims 2026-11-24-H3, 2026-11-24-b-H4].

**With a ternary cache (S3+s4r32, 2.9 bits per element at 512 tokens), Ukrainian loses more than English on identical content in all three models.**
- FLORES-200, 32 windows: `[check]` draft quotes window-mean R = +1.26 pp [+0.85, +1.68] for BitNet and +0.53 pp for EuroLLM; unified pooled `ΣΔ/ΣB` values in `tables.md` are **+1.22 [+0.83, +1.64]** (BitNet) and **+0.47 [+0.04, +0.91]** (EuroLLM). For Qwen the relative loss explodes in both languages, and much more in Ukrainian (`R_pooled = +31.56 [+27.12, +35.95]` vs `R_mean = +31.99`, diff `0.43 pp > 0.1 pp`) [claim 2026-11-24-H3].
- NTREX-128, 24 complete articles at up to 1024 tokens: English is no longer spared (+1.90% for BitNet, +2.51% for EuroLLM). The gap persists: `[check]` draft quotes window-mean R = +1.53 pp [+1.09, +2.00] for BitNet and +1.43 pp for EuroLLM, whereas `claim 2026-11-24-b-H4` / `tables.md` pooled values are **+1.51 [+1.06, +2.01]** (BitNet) and **+1.38 [+0.96, +1.85]** (EuroLLM).
- Over five (`-> 10` in `[2026-11-26-H2]`) languages the loss is ordered en < de < pl ≈ uk < bg for BitNet [claim 2026-11-24-b-H3]. Latin-script Polish matches Cyrillic Ukrainian, so script is not the driver.

## 4.2 Values and the recent window carry the gap

Ablations on 12 paired FLORES windows (BitNet) [claim 2026-11-24-H4]:
- **Values carry the gap.** Quantizing only V reproduces 90% of D; only K, 39%.
- **Layers.** The first and last thirds of the network contribute 47% and 36%, the middle third 4%.
- **The recent INT8 window controls the gap almost entirely.** Without it D grows 4.5-fold (`[check]` baseline without window/r256 from `mechanism.md`, not tabulated verbatim in `claims.md` row `2026-11-24-H4`); with r = 128 it falls to 17% (CI includes 0), with r = 256 to 2%.
- **What is ruled out.**
  - The relative L2 error of quantized K and V is identical for the two languages (0.426 vs 0.426).
  - The attention mass on positions outside the sinks and the window is the same as well (44.4% vs 43.7% `[check]` order in draft is UK vs EN: `claims.md` `2026-11-24-H4` reports EN `43.65%` vs UK `44.38%`).
  - So neither noisier Ukrainian keys and values nor more long-range attention explains the gap.

## 4.3 A window defined in tokens is not language-neutral

A 32-token window covers fewer characters of Ukrainian text than of English.
- **Defining it in characters closes part of the gap.**
  - FLORES-32, BitNet: `[check]` draft quotes window-mean R falls from +1.26 to +0.98 pp at C = 120 and to +0.37 pp [+0.13, +0.62] at C = 240; unified pooled values in `tables.md` are **+1.22 -> +0.96 [+0.52, +1.39] -> +0.35 [+0.11, +0.59]** (`-21.9%` at `C=120`) [claim 2026-11-24-b-H1].
  - Five languages: C = 120 reduces the mean non-English R by 22% for BitNet and 35% for EuroLLM; for Ukrainian in EuroLLM by 71%, to +0.16 pp [−0.10, +0.39] [claim 2026-11-24-b-H3b].
- **Equal text coverage is still not enough for BitNet.** To reach English's relative loss, Ukrainian needs about 4.2× as many INT8 recent slots: 16 → 65 at a 0.5% target, i.e. +0.54 [+0.34, +0.70] bits per element at ctx 512. That is also 2.7–2.8× as many covered characters, whereas fertility alone predicts ×1.6–1.7 (`[check]` iso-loss FLORES section uses `mean(Δ/B)`, while NTREX cross-check uses pooled `ΣΔ/ΣB`) [claim 2026-11-25-coord-ISO, exploratory].
- **On NTREX the same comparison separates the models:**
  - BitNet: ×3.35 [1.70, ≥ 4];
  - EuroLLM: ×1.40 [1.17, 1.79], which is its fertility [claim 2026-11-25-coord-ISO, exploratory].
- On the bits axis, the character window and a larger token window lie on the same loss curve for Ukrainian. The character window is an allocation rule, not a better quantizer.

## 4.4 Fertility or fragility?

The extra nats on identical content decompose exactly into three terms [claim 2026-11-25-coord-DEC, exploratory]:
- **fertility:** more tokens at English's damage rate;
- **composition:** a different mix of token difficulty;
- **fragility:** more damage at equal difficulty.

The three models differ in kind:

| Model | Fertility | Fragility | Note |
|---|---|---|---|
| BitNet | ≈ 0 (`-9%..+15%`) | 122–158% of D | every language, both corpora; 97–103% on tokens with base NLL < 4 |
| EuroLLM | 18–46% | 73–115% (`-36%..+115%` in 10 langs) | a character window removes its Ukrainian fragility |
| Qwen | 42% | 47% | |

- **Composition offsets part of D in BitNet and EuroLLM.** Their non-English tokens fall into difficulty bins that lose less on average.
- **Matched difficulty gives the same picture.** At equal base NLL, a Ukrainian token in BitNet loses 4.77× `[check]` (draft rounded `4.8×` and CI `[4.1, 5.8]`; verified `matched_difficulty.md` gives **`4.77× [4.08, 5.64]`** `b<4`, all bins `4.79`) as much as an English one on NTREX; in EuroLLM `1.47× [1.30, 1.66]` (`b<4`, all bins `1.51`; draft rounded `1.5× [1.3, 1.7]`); cross-model difference `+3.30 [+2.64, +4.12]` (`b<4`; all bins `+3.28 [+2.57, +4.22]`; draft wrote `+3.3 [+2.6, +4.2]`) [claim 2026-11-25-coord-MD].
  - Within BitNet, across de, pl, uk, bg, this ratio follows the cost of content κ (Spearman 1.0) rather than fertility (0.8).
  - Qwen has the highest fertility (`φ=2.53`) and an intermediate κ, yet a ratio of only `1.79 [1.56, 2.08]` on FLORES-32 `b<4` (`[check]` draft wrote `1.6–1.8`).
- **Across 10 FLORES languages (`9` non-EN, `fi` pending) [claim 2026-11-26-H2]:**
  - Expanding from 4 to 9 non-EN languages reduces predictor collinearity to `VIF = 1.5` (BitNet) and `VIF = 2.4` (EuroLLM).
  - In BitNet, **relative loss `R`** maps cleanly to **`tokenization`** (`β_logφ = +0.67 [+0.32, +0.93]`, `β_logκ = +0.32 [-0.06, +0.62]`), **`ratio_lt4`** maps to **`competence`** (`β_logφ = -0.11 [-0.27, +0.07]`, `β_logκ = +0.98 [+0.85, +1.06]`), and **`fragility`** receives **`both`** (`β_logφ = +0.21 [+0.05, +0.38]`, `β_logκ = +0.83 [+0.69, +0.93]`, `κ` ~4× `φ`; BitNet `es` `R_pooled = +1.43` vs `R_mean = +1.32`).
  - In EuroLLM (`κ ∈ [1.08, 1.32]`, `|R| <= 1.00 pp`), `R` and `ratio_lt4` remain **`collinear`**, while **`fragility`** isolates **`competence`** (`β_logκ = +0.75 [+0.38, +0.97]`).
- **Open questions.**
  - Whether BitNet's fragility comes from weak Ukrainian competence or from ternary weights: `[TODO H4, SmolLM2 weights pending]` (`[2026-11-26-H4-tok]` tokenizer check done: `SmolLM2-1.7B` `φ_UK = 3.92×` FLORES / `3.70×` NTREX; `24/24` NTREX articles exceed `2047` UK tokens).
  - Finnish (`fi`) completion for 11-language H2 (`10/11` done `[2026-11-26-H2]`).

## 4.5 Context length: concatenation, not language

At 2048 tokens, the ternary penalty grew with position for Ukrainian but not for English. The Ukrainian windows were concatenations of short documents.
- With equally long single documents the growth disappears in both languages.
- With concatenated documents it appears in both.

So the positional effect belongs to multi-document (RAG-like) contexts, not to a language [claim 2026-11-24-H1b].

## 4.6 Practice

- **Bits.** `[TODO T51 — deferred to next night]` bits per element vs R for V with 2 or 4 bits under a ternary K, against r = 128 and C = 120 / 240; iso-loss budget for each family.
- **Calibration.** One clip-calibration table fitted on mixed EN+UK data matches language-specific tables and transfers out of domain [claims 2026-11-23-H1-sim, 2026-11-24-H2].
- **Runtime.** The C++ runtime reproduces the simulator: ternary + α gives ×1.037 (EN) and ×1.116 (UK) at 512 tokens (`[check]` verified claim `2026-11-23-H1-eng` reports `1.0365` and `1.1162`), ×1.038 (`1.0381`) and ×1.164 (`1.1643`) at 2048; CIs against the simulator contain 1 [claims 2026-11-23-H1-eng, 2026-11-23-H3]. Under clean idle conditions (`6/6 valid`, `i3-1215U powersave`), `ternary_rot` / `ternary_rot_alpha` decode at `24.20` (`pp128`) and `22.20` (`pp512`) tok/s (within `1.6–4.7%` of `int8_fused`), while `Tenzo` decodes faster than `bitnet.cpp q8_0` (`24.50` vs `14.86` tok/s at `pp128`) [claim 2026-11-26-B1].
