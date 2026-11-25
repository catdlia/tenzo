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

## Results (`N = 24` Single-Document Parallel Articles, `ctx = 1024`)

### BitNet-2B-4T (`mean_tok`: EN `600.3`, UK `986.4`)

| Scheme | EN `rel_loss` [95% CI] | UK `rel_loss` [95% CI] | `D` (`UK-EN`, nats/doc) | `R` (pp) [95% CI] |
|---|---:|---:|---:|---:|
| `tenzo_int8` | `+0.01%` [`-0.03%`, `+0.05%`] | `+0.02%` [`-0.02%`, `+0.05%`] | `+0.73` [`-1.15`, `+2.71`] | `+0.01` [`-0.05`, `+0.07`] |
| `S3+s4r32` | `+1.90%` [`+1.61%`, `+2.19%`] | `+3.41%` [`+3.05%`, `+3.83%`] | `+133.55` [`+115.78`, `+152.98`] | `+1.51` [`+1.06`, `+2.01`] |
| `S3+s4r128` | `+1.02%` [`+0.78%`, `+1.25%`] | `+1.85%` [`+1.60%`, `+2.13%`] | `+72.50` [`+60.25`, `+85.86`] | `+0.83` [`+0.49`, `+1.19`] |

- **`S3+s4r128` recovery:** `R` drops `+1.51` -> `+0.83` pp (**45.0%** reduction; `D`: `+133.55` -> `+72.50`, **45.7%** reduction).
- **Positional trend (`S3+s4r32`, UK `129..1023`):** Spearman `ρ = 0.429`, `897..1023 minus 129..256 = +0.06606` [`+0.01467`, `+0.11170`] -> **`flat_or_bounded`**.

| Pos Slice | `n_docs` (EN / UK) | EN `S3+s4r32` `Δ/tok` [95% CI] | UK `S3+s4r32` `Δ/tok` [95% CI] | UK `S3+s4r128` `Δ/tok` [95% CI] |
|---|---:|---:|---:|---:|
| `1..128` | 24 / 24 | `+0.03655` [`+0.02125`, `+0.05244`] | `+0.09682` [`+0.07265`, `+0.12186`] | `+0.00321` [`-0.00147`, `+0.00782`] |
| `129..256` | 24 / 24 | `+0.05368` [`+0.03711`, `+0.07290`] | `+0.14225` [`+0.11177`, `+0.17349`] | `+0.03750` [`+0.02175`, `+0.05261`] |
| `257..384` | 24 / 24 | `+0.05542` [`+0.03373`, `+0.07777`] | `+0.20175` [`+0.15831`, `+0.24993`] | `+0.10178` [`+0.06952`, `+0.14175`] |
| `385..512` | 24 / 24 | `+0.06281` [`+0.04440`, `+0.08089`] | `+0.19283` [`+0.15729`, `+0.22765`] | `+0.12454` [`+0.09193`, `+0.16383`] |
| `513..640` | 24 / 24 | `+0.04942` [`+0.03379`, `+0.06487`] | `+0.15274` [`+0.11532`, `+0.19182`] | `+0.07799` [`+0.05170`, `+0.10464`] |
| `641..768` | 2 / 24 | — (n=2) | `+0.17512` [`+0.12652`, `+0.22250`] | `+0.12350` [`+0.08408`, `+0.16218`] |
| `769..896` | 0 / 24 | — (n=0) | `+0.17573` [`+0.14311`, `+0.20855`] | `+0.12468` [`+0.09634`, `+0.15579`] |
| `897..1023` | 0 / 24 | — (n=0) | `+0.20831` [`+0.16056`, `+0.25131`] | `+0.14417` [`+0.10232`, `+0.18503`] |

### EuroLLM-1.7B (`mean_tok`: EN `672.8`, UK `962.3`)

| Scheme | EN `rel_loss` [95% CI] | UK `rel_loss` [95% CI] | `D` (`UK-EN`, nats/doc) | `R` (pp) [95% CI] |
|---|---:|---:|---:|---:|
| `tenzo_int8` | `+0.00%` [`-0.02%`, `+0.03%`] | `+0.01%` [`-0.01%`, `+0.03%`] | `+0.21` [`-0.31`, `+0.73`] | `+0.01` [`-0.02`, `+0.04`] |
| `S3+s4r32` | `+2.51%` [`+2.14%`, `+2.90%`] | `+3.90%` [`+3.39%`, `+4.47%`] | `+37.59` [`+29.58`, `+46.15`] | `+1.38` [`+0.96`, `+1.85`] |
| `S3+s4r128` | `+1.11%` [`+0.85%`, `+1.37%`] | `+1.65%` [`+1.43%`, `+1.92%`] | `+15.06` [`+10.34`, `+20.08`] | `+0.53` [`+0.27`, `+0.83`] |

- **`S3+s4r128` recovery:** `R` drops `+1.38` -> `+0.53` pp (**61.4%** reduction; `D`: `+37.59` -> `+15.06`, **59.9%** reduction).
- **Positional trend (`S3+s4r32`, UK `129..1023`):** Spearman `ρ = 0.857`, `897..1023 minus 129..256 = +0.02103` [`-0.00931`, `+0.05021`] -> **`flat_or_bounded`**.

| Pos Slice | `n_docs` (EN / UK) | EN `S3+s4r32` `Δ/tok` [95% CI] | UK `S3+s4r32` `Δ/tok` [95% CI] | UK `S3+s4r128` `Δ/tok` [95% CI] |
|---|---:|---:|---:|---:|
| `1..128` | 24 / 24 | `+0.02393` [`+0.01618`, `+0.03185`] | `+0.03032` [`+0.02287`, `+0.03803`] | `-0.00133` [`-0.00293`, `+0.00021`] |
| `129..256` | 24 / 24 | `+0.05599` [`+0.03908`, `+0.07280`] | `+0.07328` [`+0.05672`, `+0.09057`] | `+0.01217` [`+0.00594`, `+0.01874`] |
| `257..384` | 24 / 24 | `+0.06309` [`+0.04914`, `+0.07757`] | `+0.07452` [`+0.04960`, `+0.09865`] | `+0.03251` [`+0.01953`, `+0.04541`] |
| `385..512` | 24 / 24 | `+0.06590` [`+0.04755`, `+0.08586`] | `+0.09224` [`+0.06926`, `+0.11653`] | `+0.03988` [`+0.02729`, `+0.05253`] |
| `513..640` | 24 / 24 | `+0.08554` [`+0.06288`, `+0.10906`] | `+0.08047` [`+0.05901`, `+0.10224`] | `+0.03934` [`+0.02583`, `+0.05384`] |
| `641..768` | 19 / 24 | `+0.08889` [`+0.04905`, `+0.12015`] | `+0.09710` [`+0.08251`, `+0.11333`] | `+0.05294` [`+0.04287`, `+0.06337`] |
| `769..896` | 1 / 24 | — (n=1) | `+0.11405` [`+0.09126`, `+0.14111`] | `+0.05582` [`+0.03940`, `+0.07522`] |
| `897..1023` | 0 / 23 | — (n=0) | `+0.09431` [`+0.07015`, `+0.11868`] | `+0.05204` [`+0.03469`, `+0.06985`] |

## Provenance

- Prep: `scripts/prepare_h4_ntrex.py` -> `results/2026-11-24-b_agent2/H4/ntrex_meta.json` (`0` 32-gram overlap)
- Run: `scripts/run_h4_ntrex.py` (parity verified `0.000e+00` BitNet, `<= 5.96e-06 < 1e-5` EuroLLM)
- Analysis: `scripts/analyze_h4_ntrex.py` -> `ntrex_long_parallel.{csv,png}`, `ntrex_windows.csv`, `ntrex_slices.csv`
