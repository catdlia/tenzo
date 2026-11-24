# H1a — Positional UK Degradation & Boundary Analysis on H3 (`ctx = 2048`) Data

## Plan

- **Research question:** On the existing 4 EN + 4 UK `ctx = 2048` engine runs (`results/2026-11-23_agent2/H3/scores/`), is the positional increase in Ukrainian KV quantization loss driven solely by document concatenation boundaries (`\n\n`) in `windows_uk_2048.ids`, or does position-dependent degradation also persist within long spans and inside boundary-free windows?
- **Data:**
  - Per-token log-probabilities (`lp` lines, 2047 tokens/window across $N=4$ windows per language) from:
    - `results/2026-11-23_agent2/H3/scores/engine_int8_fused_{en,uk}.txt` (reference baseline)
    - `results/2026-11-23_agent2/H3/scores/engine_ternary_rot_{en,uk}.txt`
    - `results/2026-11-23_agent2/H3/scores/engine_ternary_rot_alpha_{en,uk}.txt`
  - Input token IDs (`2048` tokens/window): `results/reference/windows_{en,uk}_2048.ids`.
  - Document boundary tokens: any vocab ID in `models/bitnet-1.58b-v2/tokenizer.json` whose token string contains `ĊĊ` (`\n\n`).
- **Metrics & models:**
  - Per-token NLL loss vs `int8_fused`: $\Delta\text{NLL}_{w,t} = \text{lp}_{\text{int8\_fused}}(w,t) - \text{lp}_{\text{mode}}(w,t)$ for step $t \in \{0,\dots,2046\}$ (input token `w[t]`, target token `w[t+1]`).
  - Document distance tracking at step $t$: when $t=0$ triggers if `w[0] in b_ids` (`last_b=0`, `n_prev_docs=1`), and for $t>0$ when `w[t] in b_ids`, `last_b = t`, `n_prev_docs += 1`; `dist = t - last_b`.
  - Block $\Delta\text{NLL}$ over `[0,512), [512,1024), [1024,1536), [1536,2047)` (and equal quarters `[0,511), [511,1023), [1023,1535), [1535,2047)` to cross-check exact boundary split), both overall and restricted to tokens `dist <= 256` after a boundary.
  - Separate per-window row for boundary-free UK window (`w=2`, `0` boundaries).
  - Primary OLS regression (pooled across the $N=4$ windows per language/mode):
    $$\Delta\text{NLL}_{w,t} = \beta_0 + \beta_{\text{pos}} \frac{t}{1000} + \beta_{\text{dist}} \log(1 + \text{dist}) + \beta_{\text{docs}} n_{\text{prev\_docs}} + \varepsilon_{w,t}$$
    plus simple slope $\Delta\text{NLL}_{w,t} \sim t/1000$.
  - Inference: **10,000-rep bootstrap by resampling the $N=4$ windows with replacement** (`seed=0`) for $\beta_{\text{pos}}$, simple slope, and last-minus-first block difference.
- **Decision / interpretation rule:**
  - Check exact replication of coordinator numbers (`boundaries=[2,2,0,57]`, `pos/1000` slope `≈+0.083`, `n_prev_docs ≈ 0`, boundary-free window rising `0.159 → 0.240`).
  - Because $N=4$ windows per language limits bootstrap power (only $4^4 = 256$ resamples), treat `H1a` findings as **exploratory/suggestive** rather than conclusive proof until tested at $N=12$ in `H1b`.
