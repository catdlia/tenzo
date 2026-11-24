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
