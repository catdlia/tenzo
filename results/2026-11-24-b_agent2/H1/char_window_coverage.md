# H1 — Recent Window with Equal Content Coverage (`T47`, FLORES-200)

## Plan

- **Hypothesis (`T47`):** Because Ukrainian has higher subword fertility than English on matched FLORES-200 content (`306.7` vs `181.6` BitNet tokens/window; `~1.69×`), a token-defined recent INT8 window ($r = 32$ tokens) protects roughly `~60%` as much textual content (`characters`) on UK as on EN. If content coverage drives the quantization disparity under `S3+s4r32`, switching to a character-defined recent INT8 window of fixed character width $C$ (with sinks $= 4$ tokens unchanged) will equalize relative loss and collapse the excess rate disparity $R$.
- **Dataset (`N = 32` paired FLORES-200 BitNet windows):**
  - `results/2026-11-24_agent2/H3/flores_bitnet_{en,uk}.ids` (`0` shared 32-grams vs `calib_*8.ids` and prior eval sets), paired with exact character offsets from `~/tenzo-refs/datasets/flores200/windows_flores_texts.json` (`models/bitnet-1.58b-v2/tokenizer.json`, verified `32/32` exact ID match on EN and UK).
  - Baseline `fp32` and `S3+s4r32` scores are reused verbatim from `results/2026-11-24_agent2/H3/scores/bitnet/`.
- **Simulator Runs (`GB.Golden` `fp32` layer-major, `MemoryMax=4G`, `S3` = Hadamard `D=128` + `TWN g=32`):**
  1. **(a) Token-window sweep (`S3+s4r<r>`):**
     - **EN:** $r \in \{8, 12, 16, 19, 24\}$ (+ reused $r = 32$).
     - **UK:** $r \in \{40, 48, 54, 64, 96\}$ (+ reused $r = 32$).
     - **Covered characters per `(window, r)`:** Compute BOTH:
       - (i) **exact position-averaged character span** across query steps $t \ge r$: $\frac{1}{T - r}\sum_{t=r}^{T-1}(E_t - c_{\max(4, t - r + 1)})$ (reporting exact sum of character lengths of the $r$ recent tokens $\frac{1}{T-r}\sum_{t=r}^{T-1}\sum_{s=t-r+1}^t (E_s - c_s)$), and
       - (ii) **linear approximation** $r \times (\text{total\_chars}_w / (T_w - 1))$.
  2. **(a2) Character-window policy (`S3+s4c<C>` for $C \in \{80, 120, 160, 240\}$ on both EN and UK):**
     - At query step $t$, key token $s \le t$ stays in INT8 iff $s < 4$ (`sinks = 4`) OR $s == t$ OR $(E_t - c_s) \le C$ (i.e., token $s$'s character start offset $c_s$ lies within the last $C$ characters before current position end $E_t$).
     - Also run **V-only (`K8V:S3+s4c120`)** at $C^* = 120$ on EN and UK to verify consistency with `H4` (that ternary `V` dominates the residual penalty).
     - Record physical mean INT8 ring slots (`r_eff`, mean recent INT8 tokens at $t \ge 4$) and effective KV bits/element per `(lang, C)` at both actual mean window length (`bits_win`) and standardized `ctx = 512` (`bits_at_512`) using `kv_ptq.bits` constants (`8.125` b/elem INT8, `2.500` b/elem `S3` ternary):
       $$\text{bits}_{@512} = \frac{\min(512, 4 + \bar{r}_{\text{eff}})\cdot 8.125 + (512 - \min(512, 4 + \bar{r}_{\text{eff}}))\cdot 2.500}{512}.$$
- **Metrics & Pre-Registered Decision Rules (`10,000`-rep paired window bootstrap, `seed = 0`):**
  - Per language $L \in \{\text{EN}, \text{UK}\}$: $\Delta_L = \text{NLL}_L - B_L$ (nats/window), relative loss $\text{rel}_L = \Delta_L / B_L$.
  - Paired cross-language contrasts for equal-token ($r=32$) and equal-character ($C \in \{80, 120, 160, 240\}$, with primary $C^* = 120$):
    $$D = \Delta_{\text{UK}} - \Delta_{\text{EN}}, \qquad R = \text{rel}_{\text{UK}} - \text{rel}_{\text{EN}}.$$
  - **Decision rule (`C* = 120` primary, checked across all `C`):**
    - **"Coverage explains the penalty"** iff at $C^* = 120$, $95\%\ \text{CI}(R)$ contains `0` AND $|R(C^*)| < \frac{1}{3}|R(r=32)|$.
    - **"Partially explains"** iff $R(C^*)$ is significantly reduced relative to $R(r=32)$ ($95\%\ \text{CI}(R_{r=32} - R_{C^*}) > 0$) while $95\%\ \text{CI}(R(C^*)) > 0$.
    - **"Does not explain"** iff $R(C^*)$ at equal characters is not smaller than $R(r=32)$ at equal tokens ($95\%\ \text{CI}(R_{r=32} - R_{C^*}) \le 0$).
- **Outputs:** `results/2026-11-24-b_agent2/H1/char_window_coverage.{md,csv,png}` (two-panel plot: `rel_L` vs covered characters, and `rel_L` vs effective bits/elem).

## Results

### 1. Part (a) — Token-Window Sweep (`S3+s4r<r>`, BitNet `FLORES-200`, `N = 32`)

| Lang | Policy | Exact covered chars | Linear approx `r × ch/tok` | Mean active `r_eff` | Bits (`win` / `@512`) | `Δ` (nats) [95% CI] | `rel_L` [95% CI] |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `EN` | `S3+s4r8` | 39.3 | 39.2 | 7.8 | 2.865 / 2.630 | +6.421 [+4.189, +8.660] | **+1.12% [+0.73%, +1.51%]** |
| `EN` | `S3+s4r12` | 58.9 | 58.8 | 11.6 | 2.981 / 2.672 | +4.605 [+2.195, +7.171] | **+0.76% [+0.37%, +1.19%]** |
| `EN` | `S3+s4r16` | 78.5 | 78.4 | 15.3 | 3.095 / 2.712 | +3.076 [+0.932, +5.337] | **+0.52% [+0.16%, +0.90%]** |
| `EN` | `S3+s4r19` | 93.2 | 93.1 | 18.0 | 3.179 / 2.742 | +1.921 [-0.040, +3.832] | **+0.32% [-0.00%, +0.63%]** |
| `EN` | `S3+s4r24` | 117.7 | 117.6 | 22.4 | 3.315 / 2.790 | +1.030 [-0.839, +2.885] | **+0.16% [-0.14%, +0.47%]** |
| `EN` | `S3+s4r32` | 156.9 | 156.8 | 29.2 | 3.523 / 2.865 | +0.232 [-1.197, +1.704] | **+0.02% [-0.22%, +0.28%]** |
| `UK` | `S3+s4r32` | 95.6 | 95.4 | 30.4 | 3.128 / 2.877 | +21.081 [+14.841, +27.342] | **+1.28% [+0.90%, +1.68%]** |
| `UK` | `S3+s4r40` | 119.5 | 119.2 | 37.4 | 3.257 / 2.955 | +16.786 [+11.312, +22.472] | **+1.03% [+0.69%, +1.40%]** |
| `UK` | `S3+s4r48` | 143.5 | 143.1 | 44.3 | 3.382 / 3.030 | +12.945 [+8.402, +17.481] | **+0.79% [+0.50%, +1.08%]** |
| `UK` | `S3+s4r54` | 161.4 | 161.0 | 49.3 | 3.474 / 3.085 | +12.593 [+8.397, +16.668] | **+0.76% [+0.50%, +1.02%]** |
| `UK` | `S3+s4r64` | 191.4 | 190.8 | 57.3 | 3.621 / 3.174 | +9.590 [+5.367, +13.782] | **+0.58% [+0.33%, +0.84%]** |
| `UK` | `S3+s4r96` | 287.0 | 286.2 | 80.9 | 4.052 / 3.433 | +5.007 [+2.085, +7.759] | **+0.31% [+0.14%, +0.48%]** |

### 2. Part (a2) — Equal-Character Policy (`S3+s4c<C>`) & Contrast vs `S3+s4r32`

| Policy | Cov chars EN/UK | Mean `r_eff` EN/UK | Bits `@512` EN/UK | `rel_EN` | `rel_UK` | `D` (nats) [95% CI] | `R = UK − EN` [95% CI] | `R(r32) − R(C)` [95% CI] | Verdict |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| `S3+s4r32` | 156.9 / 95.6 | 29.2 / 30.4 | 2.865 / 2.877 | +0.02% | +1.28% | +20.849 [+14.817, +26.898] | **+1.26% [+0.85%, +1.68%]** | +0.00% [+0.00%, +0.00%] | baseline |
| `S3+s4c80` | 77.3 / 78.5 | 15.4 / 25.6 | 2.714 / 2.825 | +0.43% | +1.47% | +21.487 [+14.574, +28.165] | **+1.03% [+0.55%, +1.48%]** | +0.23% [-0.11%, +0.56%] | DOES NOT EXPLAIN (R not smaller at equal chars) |
| `S3+s4c120` | 117.2 / 118.5 | 22.7 / 37.5 | 2.793 / 2.956 | +0.09% | +1.08% | +17.128 [+11.501, +22.361] | **+0.98% [+0.55%, +1.40%]** | +0.28% [+0.02%, +0.52%] | PARTIAL (R significantly reduced, CI(R) > 0) |
| `S3+s4c160` | 157.3 / 158.6 | 29.6 / 48.8 | 2.869 / 3.081 | -0.02% | +0.74% | +12.372 [+7.858, +16.760] | **+0.76% [+0.42%, +1.11%]** | +0.49% [+0.28%, +0.71%] | PARTIAL (R significantly reduced, CI(R) > 0) |
| `S3+s4c240` | 237.2 / 238.5 | 42.3 / 69.7 | 3.008 / 3.310 | +0.04% | +0.41% | +6.397 [+3.062, +9.587] | **+0.37% [+0.13%, +0.62%]** | +0.89% [+0.55%, +1.24%] | PARTIAL (R significantly reduced, CI(R) > 0) |
| `K8V:S3+s4c120` | 117.2 / 118.5 | 22.7 / 37.5 | 2.793 / 2.956 | +0.06% | +1.03% | +16.324 [+11.981, +20.655] | **+0.97% [+0.66%, +1.29%]** | +0.29% [-0.04%, +0.66%] | DOES NOT EXPLAIN (R not smaller at equal chars) |

### 3. Pre-Registered Decision Outcome

- **Baseline `S3+s4r32` (equal tokens):** `EN` covers `156.9` chars/step vs `UK` `95.6` chars/step (`60.97%` of EN coverage); `D = +20.849` nats, **`R = +1.26% [+0.85%, +1.68%]`**.
- **Primary test at `C* = 120` (`S3+s4c120`):** **`PARTIAL (R significantly reduced, CI(R) > 0)`**. At `C = 120`, `R = +0.98% [+0.55%, +1.40%]` (vs threshold `|R| < 0.42%`); UK requires `37.5` INT8 recent slots (`2.956` bits `@512`, `3.259` bits on FLORES window) vs `22.7` EN slots (`2.793` bits `@512`).

### Reproducibility & Metadata
- Command: `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- nice -n 19 ionice -c3 python3 scripts/run_h1_char_window.py`
- `commit 82eabdc`, governor `powersave`, `n_boot=10000`, paired by window index, `seed=0`.
