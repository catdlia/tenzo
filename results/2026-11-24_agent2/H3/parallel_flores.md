# H3 — Parallel Corpus EN/UK KV Degradation (`FLORES-200`, `T45`)

## Plan

- **Research question:** Does Ukrainian suffer greater KV-cache quantization loss than English when semantic content, difficulty context, and domain are **strictly matched** sentence-for-sentence on a parallel corpus (`FLORES-200` `devtest`), removing any confound from comparing different corpora or normalizing per character (`BPC`) vs per byte (`BPB`)?
- **Corpus & window construction (`N = 32` aligned pairs):**
  - Source: original `FLORES-200` tarball (`https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`, ~25 MB, CC BY-SA 4.0) stored outside git in `~/tenzo-refs/datasets/flores200/`; archive `sha256` recorded in metadata (no raw text committed).
  - Pairs: consecutive sentences from `devtest/eng_Latn.devtest` and `devtest/ukr_Cyrl.devtest`, joined with `" "`.
  - Packing: greedy sentence-order packing from index `0` so each window $w \in \{0,\dots,31\}$ satisfies `n_tokens(UK) <= 511` AND `n_tokens(EN) <= 511` across all three model tokenizers (`BitNet`, `EuroLLM-1.7B`, `Qwen2.5-1.5B`). Notice EN windows cover the exact same sentence indices $[i_w, j_w)$ and thus contain identical semantic content while being shorter in tokens.
  - Leakage check: verify `0` shared 32-grams between packed BitNet windows and `calib_en8.ids`, `calib_uk8.ids`, and all prior evaluation `.ids` files.
- **Models & KV schemes evaluated:**
  1. `BitNet-b1.58-2B-4T` (`tools/reference/golden_bitnet.py`, `fp32` base + `tenzo_int8`, `S3+s4r32`, `S13+s4r32`).
  2. `utter-project/EuroLLM-1.7B` (`fp` base + `tenzo_int8`, `S3+s4r32`, `S13+s4r32`).
  3. `Qwen/Qwen2.5-1.5B` (`fp` base + `tenzo_int8`, `S3+s4r32`; optional `S13+s4r32` if time permits, noting S13 collapse observed in 2026-11-23 `H2`).
- **Metrics (per window $w$ and language $L \in \{\text{EN}, \text{UK}\}$):**
  - $B_{wL} = -\sum_{t} \text{lp}_{\text{base}}(w, L, t)$: total NLL (nats) of the unquantized baseline on window $w$.
  - $S_{wL} = -\sum_{t} \text{lp}_{\text{scheme}}(w, L, t)$: total NLL (nats) under KV quantization.
  - $\Delta_{wL} = S_{wL} - B_{wL}$: quantization-induced information loss in **nats on identical content**.
  - **Primary metric:** Paired window difference $D_w = \Delta_{w,\text{UK}} - \Delta_{w,\text{EN}}$ (nats on identical meaning), summarized by its mean $\bar{D}$ and **10,000-rep paired window bootstrap 95% CI** (`seed=0`).
  - **Secondary metrics (all with 10,000-rep paired bootstrap CIs):**
    - Relative loss $\text{rel}_{wL} = \Delta_{wL} / B_{wL}$ and paired difference $R_w = \text{rel}_{w,\text{UK}} - \text{rel}_{w,\text{EN}}$ (plus total corpus ratio $\sum_w \Delta_{wL} / \sum_w B_{wL}$);
    - Normalized losses $\Delta\text{BPC}_{wL} = \Delta_{wL} / (\ln(2)\cdot\text{chars}_{wL})$ and $\Delta\text{BPB}_{wL} = \Delta_{wL} / (\ln(2)\cdot\text{bytes}_{wL})$ and their UK − EN paired differences (explicitly checking whether unit dependence persists on parallel content);
    - Baseline gap $B_{w,\text{UK}} - B_{w,\text{EN}}$ (nats).
- **Pre-registered decision rules:**
  1. *"UK penalty on identical content exists"* for `(model, scheme)` iff $95\%\ \text{CI}(\bar{D}) > 0$.
  2. *"Penalty is proportional to baseline complexity"* iff $95\%\ \text{CI}(\bar{R}) \ni 0$ even when $95\%\ \text{CI}(\bar{D}) > 0$ (i.e., extra nats track extra baseline bits/nats rather than an excess fractional rate). Conversely, $95\%\ \text{CI}(\bar{R}) > 0$ establishes super-proportional UK sensitivity even at fixed semantic content.
