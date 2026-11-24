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

## Results

Dataset: `FLORES-200 devtest` (`N = 32` paired windows, sentences `0..208`, `max_scored_tokens <= 511` across all 3 tokenizers, archive `sha256: b8b0b76783024b85...`, `0` shared 32-grams vs all existing calibration/eval files).

### 1. Unquantized baseline on identical semantic content

| Model | Mean tok EN | Mean tok UK | Mean B_EN (nats) | Mean B_UK (nats) | Paired gap B_UK−B_EN [95% CI] | Base BPC EN / UK | Base BPB EN / UK |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| BitNet-b1.58-2B | 181.6 | 306.7 | 577.60 | 1670.30 | +1092.70 [+1050.34, +1136.57] | 0.9411 / 2.6420 | 0.9406 / 1.4511 |
| EuroLLM-1.7B | 196.8 | 290.5 | 557.80 | 707.25 | +149.45 [+133.03, +166.83] | 0.9089 / 1.1187 | 0.9083 / 0.6144 |
| Qwen2.5-1.5B | 186.1 | 471.5 | 550.74 | 1066.35 | +515.61 [+493.51, +537.88] | 0.8973 / 1.6867 | 0.8968 / 0.9264 |

### 2. Primary table: Matched-content quantization disparity (`D`, `R`, `ΔBPC`, `ΔBPB`)

| Model | Scheme | Mean Δ_EN (nats) | Mean Δ_UK (nats) | Primary D = Δ_UK−Δ_EN [95% CI] | Rel loss EN / UK | Rel diff R = UK−EN [95% CI] | ΔBPC (UK−EN) [95% CI] | ΔBPB (UK−EN) [95% CI] | Decision |
| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- | --- |
| BitNet-b1.58-2B | `tenzo_int8` | +0.176 | +0.439 | **+0.263 [-0.474, +1.099]** | +0.03% / +0.03% | -0.00% [-0.07%, +0.07%] | +0.0004 [-0.0007, +0.0017] | +0.0001 [-0.0007, +0.0009] | `no clear diff (D~0)` |
| BitNet-b1.58-2B | `S3+s4r32` | +0.232 | +21.081 | **+20.849 [+14.817, +26.898]** | +0.02% / +1.28% | +1.26% [+0.85%, +1.68%] | +0.0332 [+0.0235, +0.0430] | +0.0180 [+0.0127, +0.0235] | `super-prop (D>0, R>0)` |
| BitNet-b1.58-2B | `S13+s4r32` | +0.196 | +21.145 | **+20.949 [+17.430, +24.514]** | +0.03% / +1.28% | +1.26% [+1.02%, +1.50%] | +0.0332 [+0.0276, +0.0389] | +0.0180 [+0.0150, +0.0212] | `super-prop (D>0, R>0)` |
| EuroLLM-1.7B | `tenzo_int8` | +0.138 | -0.113 | **-0.251 [-0.512, +0.007]** | +0.02% / -0.02% | -0.04% [-0.09%, -0.00%] | -0.0004 [-0.0009, +0.0000] | -0.0003 [-0.0006, -0.0000] | `no clear diff (D~0)` |
| EuroLLM-1.7B | `S3+s4r32` | +1.349 | +5.010 | **+3.661 [+0.942, +6.467]** | +0.21% / +0.74% | +0.53% [+0.10%, +0.97%] | +0.0057 [+0.0014, +0.0102] | +0.0022 [-0.0011, +0.0054] | `super-prop (D>0, R>0)` |
| EuroLLM-1.7B | `S13+s4r32` | +0.465 | +1.303 | **+0.838 [-0.761, +2.469]** | +0.08% / +0.19% | +0.11% [-0.13%, +0.35%] | +0.0012 [-0.0013, +0.0037] | +0.0003 [-0.0015, +0.0020] | `no clear diff (D~0)` |
| Qwen2.5-1.5B | `tenzo_int8` | +7.507 | +53.845 | **+46.337 [+38.773, +54.068]** | +1.38% / +5.04% | +3.66% [+2.87%, +4.45%] | +0.0729 [+0.0609, +0.0852] | +0.0346 [+0.0272, +0.0421] | `super-prop (D>0, R>0)` |
| Qwen2.5-1.5B | `S3+s4r32` | +125.009 | +578.634 | **+453.625 [+423.184, +485.495]** | +22.53% / +54.52% | +31.99% [+27.55%, +36.38%] | +0.7138 [+0.6661, +0.7626] | +0.3013 [+0.2636, +0.3388] | `super-prop (D>0, R>0)` |
| Qwen2.5-1.5B | `S13+s4r32` | +598.750 | +1554.548 | **+955.798 [+881.499, +1036.605]** | +109.47% / +147.03% | +37.56% [+27.81%, +47.48%] | +1.4830 [+1.3696, +1.6041] | +0.3759 [+0.2988, +0.4554] | `super-prop (D>0, R>0)` |

### 3. Per-unit breakdown (`ΔBPC`, `ΔBPB`, corpus-weighted `R`)

| Model | Scheme | ΔBPC EN | ΔBPC UK | ΔBPB EN | ΔBPB UK | Corpus-weighted Δ_UK/B_UK − Δ_EN/B_EN [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| BitNet-b1.58-2B | `tenzo_int8` | +0.0003 | +0.0007 | +0.0003 | +0.0004 | -0.00% [-0.08%, +0.07%] |
| BitNet-b1.58-2B | `S3+s4r32` | +0.0004 | +0.0336 | +0.0004 | +0.0184 | +1.22% [+0.82%, +1.63%] |
| BitNet-b1.58-2B | `S13+s4r32` | +0.0004 | +0.0336 | +0.0004 | +0.0184 | +1.23% [+1.00%, +1.48%] |
| EuroLLM-1.7B | `tenzo_int8` | +0.0002 | -0.0002 | +0.0002 | -0.0001 | -0.04% [-0.08%, -0.00%] |
| EuroLLM-1.7B | `S3+s4r32` | +0.0022 | +0.0079 | +0.0022 | +0.0043 | +0.47% [+0.04%, +0.92%] |
| EuroLLM-1.7B | `S13+s4r32` | +0.0009 | +0.0021 | +0.0009 | +0.0011 | +0.10% [-0.14%, +0.35%] |
| Qwen2.5-1.5B | `tenzo_int8` | +0.0122 | +0.0851 | +0.0122 | +0.0468 | +3.69% [+2.89%, +4.48%] |
| Qwen2.5-1.5B | `S3+s4r32` | +0.2013 | +0.9152 | +0.2012 | +0.5025 | +31.56% [+27.12%, +36.01%] |
| Qwen2.5-1.5B | `S13+s4r32` | +0.9756 | +2.4585 | +0.9750 | +1.3509 | +37.07% [+27.18%, +47.15%] |

### 4. Conclusions & Caveats

- **Decision Rule 1 (`95% CI(D) > 0`, UK penalty at matched semantic content):** Holds across **all three model families** on `S3+s4r32` — BitNet-b1.58-2B `D=+20.849 [+14.817, +26.898]` nats, EuroLLM-1.7B `D=+3.661 [+0.942, +6.467]` nats, Qwen2.5-1.5B `D=+453.625 [+423.184, +485.495]` nats. On `S13+s4r32` (`K INT4`), BitNet (`+20.949 [+17.430, +24.514]`) and Qwen2.5 (`+955.798`) retain strong UK penalty while EuroLLM-1.7B crosses zero (`+0.838 [-0.761, +2.469]`). Under `tenzo_int8`, BitNet and EuroLLM show `D ~ 0` while Qwen2.5 degrades UK (`+46.337 [+38.773, +54.068]`).
- **Decision Rule 2 (`95% CI(R) > 0`, super-proportional vs baseline complexity):** Under `S3+s4r32`, **all three models** exhibit `95% CI(R) > 0` (`BitNet` `+1.26% [+0.85%, +1.68%]`, `EuroLLM-1.7B` `+0.53% [+0.10%, +0.97%]`, `Qwen2.5-1.5B` `+31.99% [+27.55%, +36.38%]`), showing UK sensitivity rises super-proportionally faster than base NLL.
- **Resolution of `T45` (`ΔBPC` vs `ΔBPB`):** On `EuroLLM-1.7B S3+s4r32`, UK loses `+5.010` vs EN `+1.349` nats on the exact same meaning (`ΔBPC diff +0.0057 [+0.0014, +0.0102]`), yet `ΔBPB diff` crosses zero (`+0.0022 [-0.0011, +0.0054]`) solely because UTF-8 Ukrainian has `1.82×` more bytes/char than English ASCII. Parallel content removes denominator artifacts.
- **Caveat — Sequence token-length & `s4r32` protection fraction (tested next in `H4`):** Matched-content UK strings produce more tokens than EN (`BitNet` 306.7 vs 181.6; `EuroLLM` 290.5 vs 196.8; `Qwen2.5` 471.5 vs 186.1), so the fixed 36-token INT8 protection (`sinks=4, recent=32`) protects a slightly larger fraction of EN tokens on shorter windows. `H4` isolates `r ∈ {0, 128, 256}` on these same FLORES windows.

### Reproducibility & Metadata
- Prepare windows: `python3 scripts/prepare_h3_flores.py` -> `results/2026-11-24_agent2/H3/flores_windows_meta.json`.
- Score models: `python3 scripts/run_h3_flores.py --model {bitnet,eurollm,qwen} --batch 32` (layer-major unpadded `0.000e+00` vs `B=1`, under `MemoryMax=4G` BitNet / `6G` HF).
- Analyze: `python3 scripts/analyze_h3_flores.py` (`commit 1cdb632`, governor `powersave`, `n_boot=10000`, paired by window index, `seed=0`).
