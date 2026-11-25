# Results — draft 0 (coordinator, 2026-11-25)

_Prose draft of the Results section for the KV-disparity paper. Numbers come from the run reports and the coordinator's re-analyses. **Before submission, every number must be re-checked against `results/paper/claims.md`**, after L3 has unified the `rel` convention to pooled (H1 and H3-multilang currently average per-window ratios). `[TODO]` = experiment pending; `[claim …]` = source._

## 4.1 The gap exists on content-matched text

**INT8 KV is essentially lossless in every model and language we test**, with one exception: Qwen2.5-1.5B, which shows a small INT8 penalty [claims 2026-11-24-H3, 2026-11-24-b-H4].

**With a ternary cache (S3+s4r32, 2.9 bits per element at 512 tokens), Ukrainian loses more than English on identical content in all three models.**
- FLORES-200, 32 windows: R = +1.26 pp [+0.85, +1.68] for BitNet and +0.53 pp for EuroLLM. For Qwen the relative loss explodes in both languages, and much more in Ukrainian [claim 2026-11-24-H3].
- NTREX-128, 24 complete articles at up to 1024 tokens: English is no longer spared (+1.90% for BitNet, +2.51% for EuroLLM). The gap persists: R = +1.53 pp [+1.09, +2.00] for BitNet and +1.43 pp for EuroLLM [claim 2026-11-24-b-H4].
- Over five languages the loss is ordered en < de < pl ≈ uk < bg for BitNet [claim 2026-11-24-b-H3]. Latin-script Polish matches Cyrillic Ukrainian, so script is not the driver.

## 4.2 Values and the recent window carry the gap

Ablations on 12 paired FLORES windows (BitNet) [claim 2026-11-24-H4]:
- **Values carry the gap.** Quantizing only V reproduces 90% of D; only K, 39%.
- **Layers.** The first and last thirds of the network contribute 47% and 36%, the middle third 4%.
- **The recent INT8 window controls the gap almost entirely.** Without it D grows 4.5-fold; with r = 128 it falls to 17% (CI includes 0), with r = 256 to 2%.
- **What is ruled out.**
  - The relative L2 error of quantized K and V is identical for the two languages (0.426 vs 0.426).
  - The attention mass on positions outside the sinks and the window is the same as well (44.4% vs 43.7%).
  - So neither noisier Ukrainian keys and values nor more long-range attention explains the gap.

## 4.3 A window defined in tokens is not language-neutral

A 32-token window covers fewer characters of Ukrainian text than of English.
- **Defining it in characters closes part of the gap.**
  - FLORES-32, BitNet: R falls from +1.26 to +0.98 pp at C = 120 and to +0.37 pp [+0.13, +0.62] at C = 240 [claim 2026-11-24-b-H1].
  - Five languages: C = 120 reduces the mean non-English R by 22% for BitNet and 35% for EuroLLM; for Ukrainian in EuroLLM by 71%, to +0.16 pp [−0.10, +0.39] [claim 2026-11-24-b-H3b].
- **Equal text coverage is still not enough for BitNet.** To reach English's relative loss, Ukrainian needs about 4.2× as many INT8 recent slots: 16 → 65 at a 0.5% target, i.e. +0.54 [+0.34, +0.70] bits per element at ctx 512. That is also 2.7–2.8× as many covered characters, whereas fertility alone predicts ×1.6–1.7 [coordinator iso-loss, exploratory].
- **On NTREX the same comparison separates the models:**
  - BitNet: ×3.35 [1.70, ≥ 4];
  - EuroLLM: ×1.40 [1.17, 1.79], which is its fertility [coordinator NTREX slot ratio, exploratory].
- On the bits axis, the character window and a larger token window lie on the same loss curve for Ukrainian. The character window is an allocation rule, not a better quantizer.

## 4.4 Fertility or fragility?

The extra nats on identical content decompose exactly into three terms [coordinator decomposition, exploratory]:
- **fertility:** more tokens at English's damage rate;
- **composition:** a different mix of token difficulty;
- **fragility:** more damage at equal difficulty.

The three models differ in kind:

| Model | Fertility | Fragility | Note |
|---|---|---|---|
| BitNet | ≈ 0 | 122–150% of D | every language, both corpora; ≈ 100% on tokens with base NLL < 4 |
| EuroLLM | 18–46% | 73–115% | a character window removes its Ukrainian fragility |
| Qwen | 42% | 47% | |

- **Composition offsets part of D in BitNet and EuroLLM.** Their non-English tokens fall into difficulty bins that lose less on average.
- **Matched difficulty gives the same picture.** At equal base NLL, a Ukrainian token in BitNet loses 4.8× [4.1, 5.8] as much as an English one on NTREX; in EuroLLM 1.5× [1.3, 1.7]; difference +3.3 [+2.6, +4.2] [coordinator matched difficulty].
  - Within BitNet, across de, pl, uk, bg, this ratio follows the cost of content κ (Spearman 1.0) rather than fertility (0.8).
  - Qwen has the highest fertility and an intermediate κ, yet a ratio of only 1.6–1.8.
- **Open questions.**
  - Whether BitNet's fragility comes from weak Ukrainian competence or from ternary weights: `[TODO H4, SmolLM2 control]`.
  - Which language property predicts fragility across 11 languages: `[TODO H2]`.

## 4.5 Context length: concatenation, not language

At 2048 tokens, the ternary penalty grew with position for Ukrainian but not for English. The Ukrainian windows were concatenations of short documents.
- With equally long single documents the growth disappears in both languages.
- With concatenated documents it appears in both.

So the positional effect belongs to multi-document (RAG-like) contexts, not to a language [claim 2026-11-24-H1b].

## 4.6 Practice

- **Bits.** `[TODO T51]` bits per element vs R for V with 2 or 4 bits under a ternary K, against r = 128 and C = 120 / 240; iso-loss budget for each family.
- **Calibration.** One clip-calibration table fitted on mixed EN+UK data matches language-specific tables and transfers out of domain [claims 2026-11-23-H1-sim, 2026-11-24-H2].
- **Runtime.** The C++ runtime reproduces the simulator: ternary + α gives ×1.037 (EN) and ×1.116 (UK) at 512 tokens, ×1.038 and ×1.164 at 2048; CIs against the simulator contain 1 [claims 2026-11-23-H1-eng, 2026-11-23-H3].
