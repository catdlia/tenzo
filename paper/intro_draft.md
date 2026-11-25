# Abstract, Introduction, Related Work — draft 0 (coordinator, 2026-11-25)

_Prose draft. It follows "Thesis v2" in `paper/outline.md`; if the owner picks thesis v1, only the last two sentences of the abstract and contribution 3 change. Numbers come from the run reports and must be re-checked against `results/paper/claims.md` before submission. Citations are `[key]` placeholders; `paper/related_work.md` maps them to arXiv IDs and records what has been verified. `[TODO]` = result pending (H2: 11 languages; H4: SmolLM2 control; T51: V bits)._

## Working title

KV-Cache Quantization Is Not Language-Neutral: Evidence from Parallel Corpora

## Abstract

Low-bit KV-cache quantization is usually validated on English perplexity. We ask what it costs other languages when the content is held fixed. We use two parallel corpora (FLORES-200 and NTREX-128) and three models: the natively ternary BitNet b1.58 2B4T and the full-precision EuroLLM-1.7B and Qwen2.5-1.5B.

An INT8 cache is essentially lossless. A 2.9-bit ternary cache with Hadamard rotation and an INT8 recent window is not language-neutral: on identical content, Ukrainian loses more relative log-likelihood than English in all three models. For BitNet and EuroLLM the gap is 0.5–1.5 percentage points; Qwen degrades severely in both languages. The gap is larger for languages that need more tokens for the same text.

Ablations locate the gap in the value cache and in the recent window. An exact decomposition of the extra loss shows that what drives it depends on the model:
- **EuroLLM:** a third to a half of the gap is simply the larger number of tokens, and defining the recent window in characters rather than tokens removes most of the rest.
- **BitNet**, which models Ukrainian far less well: nearly all of the gap is per-token fragility at equal difficulty. No window definition removes it; reaching English's loss takes about four times as many full-precision cache slots.

We release the evaluation code and per-token scores.

## 1. Introduction

**Context.**
- On edge devices, decoding a language model is limited by memory traffic, and at long context the key-value (KV) cache becomes the dominant consumer of both memory and bandwidth.
- Quantizing the cache to a few bits per element is therefore one of the most effective levers for local inference, and many methods now reach 2–4 bits with small perplexity changes [KIVI; KVQuant; CQ; KVLinC].
- **These evaluations are almost entirely English:** WikiText-2, C4 and English LongBench tasks. A cache policy that is "lossless on average" may not be lossless for every language that uses it.

**Why the gap is plausible, and what is missing.**
- *Weight* quantization is already known to hurt non-English text more, especially in non-Latin scripts [Marchisio et al. 2024; LCD 2026], although not universally [k-quant study 2025].
- Tokenizers split many languages into far more tokens than English for the same content [Petrov et al. 2023; Arnett et al. 2024].
- KV-cache policies are usually defined in tokens: a fixed number of sink tokens, a fixed window of recent tokens kept at high precision, budgets and evictions counted in tokens. A token-defined policy therefore gives a high-fertility language less text at high precision.
- Whether KV-cache quantization is language-neutral has, to our knowledge, not been measured on content-matched text.

**Approach.** We measure it with a simple protocol.
- Every comparison is made on **parallel text**, so both languages carry the same content.
- We report the extra negative log-likelihood (in nats) that the cache causes for that content, the relative loss, and their differences between languages. This sidesteps the units problem of comparing per-token, per-character or per-byte perplexities across tokenizers: for one of our models, bits-per-character and bits-per-byte disagree on the sign of the gap.
- We use a natively ternary model, BitNet b1.58 2B4T [BitNet 2B4T], whose weights are already ~1.6 bits, so the KV cache is the main remaining target for compression. It is evaluated through an independent fp32 re-implementation. EuroLLM-1.7B and Qwen2.5-1.5B serve as full-precision comparisons.

**Findings.**
1. An INT8 cache is lossless in every model and language we test (with a small exception for Qwen). The best ternary scheme we found (2.9 bits per element) costs Ukrainian more than English on identical content, in all three models and on both corpora. Across English, German, Polish, Ukrainian and Bulgarian the loss follows fertility.
2. The gap lives in the **value** cache and in the **recent window**. Quantizing only values reproduces 90% of it. A 128-token INT8 window removes most of it on short windows, and about half of it on 1,000-token articles.
3. Its anatomy differs by model:
   - **EuroLLM** knows Ukrainian well. Part of its gap is the larger token count, and the rest disappears when the recent window is defined in characters.
   - **BitNet** knows Ukrainian much less well (it spends 2.9× as many nats as on English for the same content). Nearly all of its gap is fragility at equal per-token difficulty, which a character-defined window barely touches.
4. `[TODO H2]` Across 11 languages, … `[TODO H4]` A full-precision model with similarly weak Ukrainian …

**Contributions.**
- (i) A paired, content-matched protocol for measuring language disparity of cache compression, including an exact decomposition of the extra loss into fertility, difficulty composition and per-token fragility.
- (ii) Evidence of the disparity across three models, two corpora and five (`[TODO]` eleven) languages.
- (iii) A mechanism: values and token-defined windows, with a model-dependent share of fragility.
- (iv) Practical guidance: define cache budgets per language or in characters; spend extra bits on values; evaluate cache quantization on parallel multilingual text. All of it is reproduced in a CPU runtime whose ternary cache matches the simulator.

## 2. Related work

**KV-cache quantization.**
- Methods quantize keys and values to 1–4 bits with per-channel or per-token groups and outlier handling [KIVI; KVQuant], use rotations to spread outliers [KVLinC; QuaRot `[check]`], and keep a small high-precision window of recent tokens [KIVI; CQ].
- Ternary-native models motivate even lower budgets; BitNet a4.8 supports a 3-bit cache [BitNet a4.8].
- Evaluation is dominated by English perplexity and English long-context suites.
- Closest to our question:
  - KV-CoRE [KV-CoRE] measures the low-rank compressibility of KV caches across languages and finds strong cross-lingual variation. It studies rank, not quantization error, and does not hold content fixed.
  - Work on eviction shows that a uniform cache budget is not neutral across the parts of a prompt [Pitfalls of KV compression]. We show the analogous effect across languages.

**Multilingual effects of quantization.**
- Quantizing weights and activations degrades non-English and especially non-Latin-script languages more [Marchisio et al. 2024]. The same holds for low-bit weight PTQ in machine translation [Uneven impact of PTQ in MT].
- At 3 bits, the degradation for non-English text in sub-4B models is two to four times larger than for English [LCD 2026].
- Calibrating on the target language helps [Calibrating beyond English].
- Moderate k-quantization of a 70B model shows no disproportionate harm [k-quant study 2025]. This is a reminder that the effect depends on bit width and model.
- Our work moves the question from weights to the cache. There the relevant budgets (windows, sinks, groups) are defined in tokens.

**Tokenization disparity.**
- The same content can cost several times more tokens in one language than in another [Petrov et al. 2023], and byte counts differ as well [Arnett et al. 2024].
- These premiums are usually discussed as cost and context-length problems. We show they also change how much of a cache policy's high-precision budget a language receives.

**Ternary LLMs and runtimes.**
- BitNet b1.58 and its 2B4T release [BitNet b1.58; BitNet 2B4T] and open ternary suites [Spectra/TriLM] make ~1.6-bit weights practical; CPU runtimes execute them efficiently [bitnet.cpp].
- A recent runtime consumes signed-digit KV caches with the same lookup tables as the ternary weights [LUT signed-digit KV 2026], which makes a ternary cache a natural next step. Our results describe what such a cache costs per language.
