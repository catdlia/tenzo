# Related work notes (2026-11-25, verified pass in the evening)

_Status: **[S]** = seen in a search result on 2026-11-25 (title/abstract level only); **[K]** = known, not re-checked this session. Read every primary source before citing._

## Quantization × languages (closest)

- **[S] Marchisio et al., "How Does Quantization Affect Multilingual LLMs?"** — EMNLP Findings 2024, arXiv:2407.03211. Weight/activation quantization; non-Latin-script languages are hurt most. *Ours:* KV cache, parallel content, mechanism.
- **[S] "Language-Conditional Dequantization: Recovering What Quantization Steals from Non-English Languages"** — arXiv:2608.11786 (Aug 2026). INT3 GPTQ **weights**, sub-4B models; 2–4× larger perplexity degradation on non-English; a per-language post-hoc fix. *Ours:* KV cache, not weights; token-defined cache policies; ternary-native model.
- **[S] "The Uneven Impact of Post-Training Quantization in Machine Translation"** — arXiv:2508.20893. Weight PTQ, many languages.
- **[S] "Calibrating Beyond English"** — arXiv:2601.18306. Calibration-language effects. Relevant to our α calibration (EN+UK mix).
- **[S] KV-CoRE** — arXiv:2602.05929 (v2, Feb 2026). SVD-based low-rank compressibility (normalized effective rank) of KV caches; English datasets plus the multilingual split of VisR-Bench (15 languages, questions, not parallel content). Reports that cross-lingual variation outweighs cross-domain variation and a rank collapse in low-resource languages. *Closest KV-side work; low-rank, not quantization; no parallel content, no Ukrainian.* Must read in full.
- **[S] "English K_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance"** — arXiv:2503.03592. Llama-3.3-70B, llama.cpp k-quants with EN / Norwegian / Malayalam importance matrices; finds no disproportionate harm. **Counterpoint to cite** (weights, large model, mild bit widths).

## Tokenization disparity

- **[S] Petrov, La Malfa, Torr, Bibi, "Language Model Tokenizers Introduce Unfairness Between Languages"** — NeurIPS 2023, arXiv:2305.15425. Up to 15× tokenization-length differences on FLORES-200. *Our "token-defined cache policy" argument builds on this.*
- **[S] Arnett, Chang, Bergen, "A Bit of a Problem: Measurement Disparities in Dataset Sizes Across Languages"** — SIGUL @ LREC-COLING 2024, arXiv:2403.00686. Byte premiums for 1155 languages on content-matched text. Relevant to why BPB hides the gap.
- **[S] "Measuring the Tokenization Premium: A Cost Audit for Underserved Language Communities"** — arXiv:2608.09046. Tokenization premiums → cost, latency, usable context (memory pressure). Motivation only.

## KV-cache quantization methods (background; all [K] or [S] titles only)

- **[S]** KIVI (2-bit, per-channel K / per-token V);
- **[S]** KVQuant (non-uniform, outliers);
- **[S]** Coupled Quantization (1 bit/channel; uses a 128-token sliding window, analogous to our recent window);
- **[S]** AQUA-KV;
- **[S]** AnTKV;
- **[S]** GSRQ;
- **[S]** E8 lattice key quantization.

- **[S]** KVLinC (arXiv:2510.05373; Hadamard rotation + linear correction) — e.g. its LongBench evaluation is 14 English tasks.
- **[S]** "The Pitfalls of KV Cache Compression" (Chen, Geh, Grover; arXiv:2510.00231) — eviction bias in multi-instruction prompts and a *fair eviction* fix. Not about languages, but the same shape of argument: a uniform budget is not neutral across inputs.
- **[S]** DynamicKV (EMNLP Findings 2025) — eviction; claims generalization to multilingual scenarios.
- **[S, via survey arXiv:2412.19442]** OneRuler — multilingual long-context benchmark, 26 languages. Candidate downstream check; title and language list not yet verified.

These evaluate essentially on English (WikiText-2, C4, LongBench). That is the gap we point at; cite a few as examples, do not overclaim.

**Novelty check (2026-11-25, four web searches):** no work found that measures KV-cache *quantization* loss across languages on parallel content, or ties cache policies defined in tokens to fertility. Not exhaustive; repeat before submission (Semantic Scholar / ACL Anthology, terms: KV cache quantization multilingual / cross-lingual / language disparity).

## Ternary-native LLMs and runtimes

- **[K]** BitNet b1.58 (arXiv:2402.17764);
- **[K]** BitNet b1.58 2B4T (arXiv:2504.12285);
- **[S]** Bitnet.cpp: Efficient Edge Inference for Ternary LLMs (Wang et al., ACL 2025, arXiv:2502.11880) — TL and I2_S kernels;
- **[S]** BitNet a4.8 (Wang, Ma, Wei; arXiv:2411.04965) — 4-bit activations; reports support for a 3-bit KV cache, evaluated on English zero-shot tasks;
- **[K]** Spectra/TriLM (arXiv:2407.12327);
- **[S]** arXiv:2608.03229, "Unified Lookup-Table Inference with Signed-Digit K/V Caches for Ternary LLMs" (Duan et al., Aug 2026) — the systems-side competitor.

## Data and models

- **[K]** FLORES-200 (NLLB team, CC-BY-SA 4.0);
- **[K]** NTREX-128 (Federmann et al. 2022);
- **[K]** EuroLLM-1.7B;
- **[K]** Qwen2.5;
- **[K]** Kobza (UNLP 2025).
