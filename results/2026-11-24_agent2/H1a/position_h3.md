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

## Results

### 1. Boundary counts & replication of coordinator checks
- `uk` windows (`windows_uk_2048.ids`, `sha256: 403a0f9a71daa621...`): boundary counts per window = `[2, 2, 0, 57]`
- `en` windows (`windows_en_2048.ids`, `sha256: 66bc4ca09d6e49d3...`): boundary counts per window = `[0, 0, 0, 0]`

| Lang | Mode | B1 (1..512) | B2 (513..1024) | B3 (1025..1536) | B4 (1537..2047) | Eq B1..511 | Eq B512..1023 | Eq B1024..1535 | Eq B1536..2047 | Δ(B4−B1) [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| UK | `ternary_rot` | 0.135 | 0.153 | 0.218 | 0.266 | 0.131 | 0.157 | 0.216 | 0.267 | +0.132 [+0.068, +0.198] |
| UK | `ternary_rot+alpha_mix` | 0.111 | 0.106 | 0.168 | 0.224 | 0.107 | 0.110 | 0.167 | 0.225 | +0.113 [+0.060, +0.167] |
| EN | `ternary_rot` | 0.065 | 0.054 | 0.082 | 0.064 | 0.065 | 0.054 | 0.081 | 0.065 | -0.001 [-0.033, +0.030] |
| EN | `ternary_rot+alpha_mix` | 0.035 | 0.025 | 0.056 | 0.033 | 0.036 | 0.025 | 0.054 | 0.034 | -0.002 [-0.038, +0.033] |

### 2. Controlling for document boundaries (`dist <= 256` after `\n\n`)

| Lang | Subset / Mode | B1 (1..512) | B2 (513..1024) | B3 (1025..1536) | B4 (1537..2047) | Δ(B4−B1) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| UK | `ternary_rot` (`dist<=256`) | 0.096 | 0.117 | 0.138 | 0.175 | +0.080 |
| UK | `ternary_rot+alpha_mix` (`dist<=256`) | 0.085 | 0.079 | 0.097 | 0.153 | +0.068 |
| EN | `ternary_rot` (`dist<=256`) | 0.056 | — | — | — | — |
| EN | `ternary_rot+alpha_mix` (`dist<=256`) | 0.026 | — | — | — | — |

### 3. Per-window breakdown (highlighting boundary-free UK `w=2`)

| Window | Mode | Boundaries | B1 | B2 | B3 | B4 | B4−B1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| UK w=0 | `ternary_rot` | 2 | 0.187 | 0.142 | 0.221 | 0.425 | +0.238 |
| UK w=1 | `ternary_rot` | 2 | 0.074 | 0.186 | 0.339 | 0.226 | +0.153 |
| UK w=2 | `ternary_rot` | 0 **(0 boundaries)** | 0.159 | 0.152 | 0.136 | 0.240 | +0.081 |
| UK w=3 | `ternary_rot` | 57 | 0.119 | 0.131 | 0.174 | 0.174 | +0.055 |
| UK w=0 | `ternary_rot+alpha_mix` | 2 | 0.139 | 0.099 | 0.167 | 0.335 | +0.197 |
| UK w=1 | `ternary_rot+alpha_mix` | 2 | 0.079 | 0.145 | 0.275 | 0.216 | +0.137 |
| UK w=2 | `ternary_rot+alpha_mix` | 0 **(0 boundaries)** | 0.122 | 0.089 | 0.096 | 0.191 | +0.069 |
| UK w=3 | `ternary_rot+alpha_mix` | 57 | 0.103 | 0.091 | 0.133 | 0.154 | +0.051 |
| EN w=0 | `ternary_rot` | 0 **(0 boundaries)** | 0.097 | 0.152 | 0.100 | 0.046 | -0.051 |
| EN w=1 | `ternary_rot` | 0 **(0 boundaries)** | 0.053 | 0.018 | 0.059 | 0.079 | +0.026 |
| EN w=2 | `ternary_rot` | 0 **(0 boundaries)** | 0.082 | 0.039 | 0.070 | 0.068 | -0.014 |
| EN w=3 | `ternary_rot` | 0 **(0 boundaries)** | 0.029 | 0.007 | 0.101 | 0.063 | +0.034 |
| EN w=0 | `ternary_rot+alpha_mix` | 0 **(0 boundaries)** | 0.060 | 0.082 | 0.099 | 0.014 | -0.046 |
| EN w=1 | `ternary_rot+alpha_mix` | 0 **(0 boundaries)** | 0.018 | -0.002 | 0.026 | 0.038 | +0.020 |
| EN w=2 | `ternary_rot+alpha_mix` | 0 **(0 boundaries)** | 0.060 | 0.034 | 0.027 | 0.030 | -0.030 |
| EN w=3 | `ternary_rot+alpha_mix` | 0 **(0 boundaries)** | 0.004 | -0.012 | 0.070 | 0.050 | +0.046 |

### 4. Per-token OLS regression (`10,000`-rep window bootstrap `95% CI`, `seed=0`)

| Lang | Mode | Simple `pos/1000` [95% CI] | Full `pos/1000` [95% CI] | Full `log1p(dist)` [95% CI] | Full `n_prev_docs` [95% CI] |
| --- | --- | --- | --- | --- | --- |
| UK | `ternary_rot` | +0.0936 [+0.0623, +0.1249] | +0.0831 [-0.0319, +0.1317] | +0.0174 [+0.0115, +0.0602] | -0.0002 [-0.0014, +0.1672] |
| UK | `ternary_rot+alpha_mix` | +0.0832 [+0.0537, +0.1126] | +0.0671 [-0.0196, +0.1047] | +0.0181 [+0.0111, +0.0469] | +0.0004 [-0.0003, +0.1350] |
| EN | `ternary_rot` | +0.0118 [-0.0183, +0.0377] | -0.0092 [-0.1081, +0.0677] | +0.0144 [-0.0206, +0.0632] | +0.0000 [+0.0000, +0.0000] |
| EN | `ternary_rot+alpha_mix` | +0.0097 [-0.0162, +0.0356] | -0.0018 [-0.0699, +0.0663] | +0.0079 [-0.0210, +0.0401] | +0.0000 [+0.0000, +0.0000] |

### Observations (cautious — exploratory `N = 4`)

- Coordinator numbers replicate for UK `ternary_rot`: boundary counts `[2, 2, 0, 57]`, equal-quarter blocks `0.131, 0.157, 0.216, 0.267`, `dist <= 256` blocks `0.096, 0.117, 0.138, 0.175`, and full OLS `pos/1000` slope `+0.0831` (`[-0.0319, +0.1317]`) with `n_prev_docs = -0.0002` (`[-0.0014, +0.1672]`).
- Comparing `ternary_rot` vs `ternary_rot+alpha_mix`: full OLS `pos/1000` on UK changes from `+0.0831` to `+0.0671` (`[-0.0196, +0.1047]`), while EN `ternary_rot` remains near zero/negative (`-0.0092 [-0.1081, +0.0677]`).
- Because `N = 4` windows per language provides only 4 independent resampling units, bootstrap CIs remain wide/exploratory even when all 4 UK windows show `B4 > B1`. `H1b` (`N = 12` clean `UK-long` vs `UK-concat`) tests whether `pos/1000` CI > 0 without boundaries.

### Reproducibility & Metadata
- Command: `python3 scripts/analyze_h3_position.py` (`commit 77edea0`, governor `powersave`, `N=4` windows/lang, `n_boot=10000`, `seed=0`).
- Engine score files in `results/2026-11-23_agent2/H3/scores/` (`engine_{int8_fused,ternary_rot,ternary_rot_alpha}_{en,uk}.txt`).
