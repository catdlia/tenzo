# H4 (`T49`): Coherent Longer Parallel `NTREX-128` Evaluation (`ctx = 1024`)

## Plan

- **Machine state (`2026-11-24T21:30:31+02:00`):** `loadavg 1.08 2.52 3.09`, `MemAvailable 10.28 GiB`, `/` `8.5G` free, `governor=powersave`, `AC=1`, active `zen-bin` (`tenzo-b1` waiting cleanly -> no idle-yield).
- **Dataset & Document-Respecting Window Selection (`N = 24` coherent articles):**
  - Source files cached locally under `~/tenzo-refs/datasets/ntrex128/` (never committed): `DOCUMENT_IDS.tsv`, `newstest2019-src.eng.txt`, `newstest2019-ref.ukr.txt` (`1997` lines across `123` distinct news articles).
  - For each distinct `doc_id`, take the maximal contiguous leading sentence prefix `0..k` whose tokenized length (excluding `BOS`) satisfies `max(tok_BitNet_EN, tok_BitNet_UK, tok_EuroLLM_EN, tok_EuroLLM_UK) <= 1023` (`total <= 1024`). Zero cross-document splicing.
  - Rank all coherent articles by `min(tok_BitNet_EN, tok_BitNet_UK, tok_EuroLLM_EN, tok_EuroLLM_UK)` descending and take the top `N = 24` articles (deterministic tie-break by original document order).
  - Prepend `BOS = 128000` (`BitNet`) / `BOS = 1` (`EuroLLM-1.7B`), write `.ids` + `ntrex_meta.json` under `results/2026-11-24-b_agent2/H4/`, and assert **0 32-gram overlap** against all `results/reference/*.ids`, `2026-11-23_agent2/H2/*.ids`, `2026-11-24_agent2/{H1b,H3}/*.ids`, `2026-11-24-b_agent2/H3/*.ids`, and self.
- **Models & Schemes:**
  - `BitNet-2B-4T` (`MemoryMax=4G`, `MemAvailable >= 6 GiB`) and `EuroLLM-1.7B` (`MemoryMax=6G`, `MemAvailable >= 8 GiB`).
  - Schemes: baseline (`fp32` / `fp`), `tenzo_int8`, `S3+s4r32`, `S3+s4r128` (recent 128 INT8).
  - Gate: `max(|B=1 - layer-major|) < 1e-6` on window 0 for every model/scheme check before full run.
- **Metrics & Bootstrap (`10,000` paired document samples, `seed = 0`):**
  - Window-level: `delta_doc`, `rel_loss = sum(NLL_q - NLL_fp) / sum(NLL_fp)`, `D = delta_doc_UK - delta_doc_EN`, `R = rel_loss_UK - rel_loss_EN` with 95% CIs, and `S3+s4r128` recovery fraction `1 - R(r=128) / R(r=32)`.
  - 128-token positional slices (`1..128`, `129..256`, ..., `897..1023` in 1-based scored target token position): mean per-token `ΔNLL/tok` in each slice for `S3+s4r32` (and `S3+s4r128`) with 95% CI, plus UK positional trend contrast `slice(897..1023) minus slice(129..256)` and Spearman $\rho$ across slices `[129..1023]`.
- **Pre-registered Decision / Interpretation Rules:**
  - **Gap replication & `r=128` mitigation:** Does `S3+s4r32` still yield `R > 0` (`95% CI > 0`) on longer coherent parallel documents, and does `S3+s4r128` recover $\ge 70\%$ of $R$?
  - **Positional trend on coherent single documents (`S3+s4r32`, UK `129..1023`):** Classify as `monotonic_growth` iff Spearman $\rho \ge 0.80$ across slices `129..1023` AND `95% CI(Δ/tok[897..1023] - Δ/tok[129..256]) > 0`; otherwise `flat_or_bounded` (confirming `H1b`'s finding that unbound monotonic degradation was driven by spliced short documents rather than long coherent context).
