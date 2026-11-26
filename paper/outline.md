# Paper outline (draft 0, 2026-11-25)

_Skeleton only: no prose yet. Every number must come from `results/paper/claims.md` (IDs in brackets). "TODO" means the experiment is not done yet. The owner decides the framing and the venue._

## Working title options

1. **KV-Cache Quantization Is Not Language-Neutral: Evidence from Parallel Corpora on a Native 1.58-bit LLM**
2. Same Content, Different Cost: Language Disparity in Low-Bit KV Caches
3. Lost in the Cache: Token-Defined KV Policies Penalize High-Fertility Languages

## Thesis (one sentence)

Low-bit KV-cache quantization that is lossless "on average" costs high-fertility languages (Ukrainian, Polish, Bulgarian) disproportionately more information on *identical* content. Two components compound:
- (a) cache policies defined in tokens;
- (b) higher per-token fragility in languages the model knows less well.

Simple, measurable remedies shrink the gap.

**Thesis v2 (coordinator, after the 25.11 decomposition; owner decides).** The same gap has a different anatomy depending on how well the model knows the language:
- in a model that knows it (EuroLLM), the gap is the extra tokens plus a token-defined cache window, and a window defined in characters largely removes the second part;
- in a model that does not (BitNet), it is per-token fragility at equal difficulty that no window definition removes and only more bits reduce.

Pending: whether that fragility is about competence or about ternary weights (H4, SmolLM2 — tokenizer fertility check done `[2026-11-26-H4-tok]`, weights pending), while H2 (`10` FLORES languages, `9` non-EN; `fi` pending) already separates `R -> tokenization` from `ratio_lt4 -> competence` and `fragility -> both` (`κ` ~4× `φ`) in BitNet `[2026-11-26-H2]`.

## Contributions

1. **Protocol.** A paired, information-equivalent way to measure KV-quantization disparity on parallel corpora — D in nats per identical content, R as relative loss, and a matched-difficulty ratio. It avoids the BPC-vs-BPB ambiguity; BPB can hide a real gap [2026-11-24-H3 §T45; 2026-11-23-L3; 2026-11-23-H2-eurollm].
2. **Finding.** Ternary (rotated, g32, 4 sinks + 32 recent INT8) KV loses more on Ukrainian than on English on the same content:
   - in three models (BitNet-b1.58-2B-4T, EuroLLM-1.7B, Qwen2.5-1.5B) and two parallel corpora (FLORES-200, NTREX-128) [2026-11-24-H3, 2026-11-24-b-H4];
   - in fertility order across de < pl ≈ uk < bg [2026-11-24-b-H3];
   - INT8 KV is neutral, except for Qwen.
3. **Mechanism.**
   - The disparity is carried by **V**-quantization and by the **token-defined recent window** [2026-11-24-H4, 2026-11-24-b-H1, 2026-11-24-b-H3b].
   - The residual is a **per-token fragility at equal difficulty** that is ×5–11 for BitNet (`×4.5–10.9` across 9 non-EN FLORES languages `[2026-11-26-H2]`) but only ×1.3–1.7 for EuroLLM (`×1.2–1.7`) and tracks the model's cost-of-content κ [2026-11-25-coord-MD; 2026-11-26-H2 separates `ratio_lt4 -> competence` (`β_logκ=+0.98`) vs `R -> tokenization` (`β_logφ=+0.67`) at `VIF=1.5` in BitNet; `fi` pending].
   - Exact decomposition of the extra nats into fertility, difficulty composition and fragility: for BitNet 122–158% fragility across all tested languages and both corpora (fertility ≈ 0); for EuroLLM 18–46% fertility (up to 140% in `es`, where fragility vanishes), and a character window removes its Ukrainian fragility; Qwen 42% / 47% [2026-11-25-coord-DEC, 2026-11-26-H2].
   - Positional degradation appears only in multi-document contexts, in any language [2026-11-24-H1b].
4. **Practice.**
   - Character-defined recent windows; spending bits on V [TODO T51].
   - Iso-loss budget: for equal relative loss UK needs ~4.2× the INT8 recent slots of EN (16 → 65 at 0.5%; +0.54 [0.34, 0.70] bits/elem at ctx 512), and ×2.7–2.8 the covered characters, so equal text coverage is not enough for BitNet. On NTREX the same comparison gives ×3.35 for BitNet but ×1.40 for EuroLLM, i.e. its plain fertility: a character-defined window equalizes only when the model knows the language [2026-11-25-coord-ISO].
   - Mixed-language clip calibration that transfers out of domain [2026-11-23-H1-sim, 2026-11-24-H2].
   - All of it validated in a CPU runtime whose engine matches the simulator [2026-11-23-H1-eng, 2026-11-23-H3, T10-03] and achieves 22.2–24.2 tok/s decode (`i3-1215U`, `powersave`, within ~1.6–4.7% of `int8_fused`, `6/6` clean `B1` items) [2026-11-26-B1].

## Sections (1–2 sentences each)

1. **Introduction.** Edge inference compresses the KV cache, and quality is reported on English perplexity. We ask who pays for "lossless on average", and show that the bill depends on the language even when the content is identical.
2. **Background and related work.** KV-cache quantization (KIVI, KVQuant, rotation-based, 1-bit CQ); multilingual effects of weight quantization (Marchisio et al. 2024; LCD 2026); tokenizer unfairness (Petrov et al. 2023); native ternary LLMs (BitNet b1.58). No prior work, to our knowledge, measures KV-cache quantization disparity on parallel content — see `paper/related_work.md`; verify before submission.
3. **Setup.**
   - Models; KV schemes (INT8, S3 = rotated ternary g32, S13 = K INT4 + V ternary, sinks/recent windows, α calibration).
   - fp32 golden reference and engine parity.
   - Corpora: FLORES-200 (5 → 11 languages), NTREX-128, Wikipedia/FineWeb/Kobza 2×2.
   - Metrics: D, R, matched difficulty, BPC/BPB; paired window bootstrap.
4. **Results.**
   - 4.1 Disparity on identical content (FLORES-32, NTREX-24, 3 models).
   - 4.2 Why per-character and per-byte normalizations disagree, and why parallel data settles it.
   - 4.3 Across languages: fertility vs competence (FLORES 5 → 11 languages).
   - 4.4 Mechanism: K vs V, layer groups, recent-window size, token- vs character-defined windows, matched difficulty.
   - 4.5 Long and multi-document contexts.
   - 4.6 Mitigations: bits vs disparity Pareto, calibration transfer.
5. **Discussion.**
   - Token-defined cache policies as a fairness issue.
   - Weak-language fragility in native-ternary models.
   - What a practitioner should do.
6. **Limitations.** See below.
7. **Conclusion.**
8. **Appendix.**
   - Engine/simulator parity; reproducibility (one command per figure).
   - The two silent semantic bugs caught by the golden reference (experience note).

## Figures and tables (claim ID → script)

| # | What | Claim ID | Script / file |
|---|---|---|---|
| T1 | D and R on FLORES-32, NTREX-24, and FLORES-10/H3b (pooled convention) | 2026-11-24-H3, 2026-11-24-b-H4, 2026-11-24-b-H3b, 2026-11-26-H2 | `scripts/build_paper_tables.py` → `results/paper/tables.md` |
| F1 | Extra NLL per base-difficulty bin, EN vs UK, BitNet vs EuroLLM | 2026-11-25-coord-MD | `scripts/plot_matched_difficulty.py` → `matched_bins.png` |
| F2 | R, ratio_lt4, and fragility vs fertility φ and cost κ, 10 FLORES languages (`fi` pending) | 2026-11-26-H2 | `scripts/plot_h2_flores11.py` → `results/2026-11-26_agent2/H2/fertility_vs_competence.png` |
| F3 | Token- vs character-defined recent window (coverage curves) | 2026-11-24-b-H1, 2026-11-24-b-H3b | `char_window_coverage.png`, `multilang_char_window.png` |
| F4 | Mechanism ablations (K/V, layers, r) | 2026-11-24-H4 | `mechanism.png` |
| F5 | Bits per element vs disparity, Pareto | TODO T51 | TODO |
| F2a | Decomposition of D: fertility / composition / fragility, 3 models | 2026-11-25-coord-DEC | `scripts/plot_disparity_decomposition.py` → `decomposition.png` |
| F5a | Iso-loss budget EN vs UK (token r, character C) | 2026-11-25-coord-ISO | `scripts/plot_iso_loss_bits.py` → `iso_loss_bits.png` |
| F6 | Long vs concatenated context at 2048 | 2026-11-24-H1b | `long_vs_concat_2048.png` |
| T2 | BPC vs BPB vs parallel nats (EuroLLM sign flip) | 2026-11-23-L3, 2026-11-23-H2-eurollm, 2026-11-24-H3 | existing analyses |
| T3 | α calibration in- and out-of-domain | 2026-11-23-H1-sim, 2026-11-24-H2 | existing |
| T4 | Clean idle CPU speed benchmarks (`B1`, 6/6 valid on i3-1215U `powersave`) | 2026-11-26-B1 | `scripts/analyze_b1_speed.py`, `scripts/build_paper_tables.py` |
| A1 | Engine ↔ simulator parity at 512 and 2048 | 2026-11-23-H1-eng, 2026-11-23-H3, T10-03 | existing |

## Limitations (to state plainly)

- **Data size:** 24–48 windows per corpus; FLORES windows are short (~180–605 tokens).
- **Models:** three models ≤ 2B parameters; only one of them is native-ternary. The BitNet vs EuroLLM contrast mixes competence with architecture (`H4` SmolLM2 tokenizer check done `[2026-11-26-H4-tok]`, weights pending).
- **Method:** PTQ only, no KV-aware fine-tuning; perplexity/NLL only, no downstream task or generation-quality evaluation yet.
- **Competence proxy:** κ is a proxy measured on the same data.
- **Speed:** `6/6` clean idle items completed on a single Intel i3-1215U laptop under `powersave` governor [2026-11-26-B1]; not yet benchmarked across multiple CPUs or governors.

## TODO before submission

1. T50 part 2 finish: run `fi` (11th language; `10` languages done in `[2026-11-26-H2]`).
2. `H4`: run `SmolLM2-1.7B` weights (`FLORES-32` + trimmed `NTREX-24`, tokenizer pre-check done `[2026-11-26-H4-tok]`).
3. T51: bits vs disparity Pareto (V INT4 / V 2-bit, r128, character window).
4. One downstream check on parallel data, e.g. a small multilingual reading-comprehension or QA set with en/uk/pl/bg, to show NLL gaps matter.
5. Ideally a second ternary-native model, to separate "ternary weights" from "weak language competence".
6. Related work: read and verify every entry in `paper/related_work.md`.
7. Venue decision. Candidates (owner decides):
   - SlavicNLP 2027 @ EACL (direct submission 9 Jan 2027) — the language set is Slavic-heavy;
   - UNLP 2027 (~March);
   - ACL 2027 SRW;
   - arXiv preprint.
