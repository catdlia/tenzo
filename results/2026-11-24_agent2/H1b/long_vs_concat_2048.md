# H1b — Long Document vs Concatenation at `ctx = 2048` (`T44`, TenzoEngine Only)

## Plan

- **Research question:** Is the monotonic positional rise in Ukrainian KV quantization loss across `ctx = 2048` (`H3` `2026-11-23` and `H1a` `2026-11-24`) caused by concatenating short documents (`\n\n` boundaries in Kobza), or does it persist inside single contiguous long Ukrainian documents?
- **Experimental design (`4 cells × N = 12 windows`, `[BOS] + 2047` BitNet tokens per window):**
  1. `EN-long`: Continuous WikiText-2 test text (`~/tenzo-refs/hf/wt2_test.raw`). Reuses the `4` windows `w=0..3` from `results/reference/windows_en_2048.ids` plus `8` new non-overlapping `2048`-token windows (`0` shared 32-grams vs `calib_en8.ids`, `windows.ids`, `windows_en_2048.ids`, `H2/bitnet_en_wiki.ids`).
  2. `EN-concat`: Short English Wikipedia articles (`wikimedia/wikipedia` `20231101.en`, `300 <= chars < 1500`, starting from row offset `200` to remain disjoint from `2026-11-23 H2` rows `0..99`), joined with `"\n\n"` and sliced into `12` `[BOS] + 2047` windows.
  3. `UK-long`: Single long Ukrainian Wikipedia articles (`wikimedia/wikipedia` `20231101.uk`, starting from row offset `200` to remain disjoint from `2026-11-23 H2` rows `0..99`), selecting `12` individual articles each having `>= 2047` BitNet tokens (1 article = 1 window: `[BOS] + first 2047 tokens`, expected `0` inter-document concatenation boundaries).
  4. `UK-concat`: Ukrainian web documents from `Goader/kobza` (`default`, starting from row offset `300` to remain disjoint from prior `0..38`, `100..199`, `200..299` slices), joined with `"\n\n"` and sliced into `12` `[BOS] + 2047` windows.
  - All new windows (`w=4..11` in `EN-long` and `w=0..11` in `EN-concat`, `UK-long`, `UK-concat`) are verified to have `0` shared 32-grams with `calib_en8.ids`, `calib_uk8.ids`, and all existing evaluation `.ids` files.
- **Engine evaluation (`TenzoEngine` only — validated `0.999–1.003` vs golden at `ctx=2048`):**
  - Modes: `int8_fused` (baseline), `ternary_rot`, `ternary_rot --kv-alpha /app/results/2026-11-23_agent2/H1/alpha_mix.txt` (`-c 2048 --prefill batched`).
  - `EN-long w=0..3` reuses exact existing `results/2026-11-23_agent2/H3/scores/engine_*_en.txt` scores (checked byte/format consistent); remaining `EN-long w=4..11` and all `EN-concat`, `UK-long`, `UK-concat` run incrementally with checkpointing every 4 windows.
- **Primary & secondary metrics (`10,000`-rep window bootstrap `seed=0`):**
  - Full-context and 512-block PPL ratios vs `int8_fused` (`1..512`, `513..1024`, `1025..1536`, `1537..2047`) + $\Delta(\text{B4}-\text{B1})$ $\Delta\text{NLL}$.
  - Per-token OLS regression from `H1a` (for $\Delta\text{NLL}_{w,t} = \text{lp}_{\text{int8\_fused}} - \text{lp}_{\text{mode}}$):
    - Simple slope $\Delta\text{NLL}_{w,t} \sim t/1000$,
    - Full model $\Delta\text{NLL}_{w,t} \sim \beta_0 + \beta_{\text{pos}} (t/1000) + \beta_{\text{dist}} \log(1+\text{dist}) + \beta_{\text{docs}} n_{\text{prev\_docs}}$,
    reporting **95% window-bootstrap CIs** (`10,000` reps, `seed=0`) for $\beta_{\text{pos}}$ and $\beta_{\text{docs}}$ per cell and mode (and simple `pos/1000` where `n_prev_docs == 0` in single-document cells).
- **Pre-registered decision rules (`H1b`):**
  1. **"Positional effect exists in cell X"** iff the 95% window-bootstrap CI of $\beta_{\text{pos}}$ (`pos/1000`) is strictly $> 0$. (In boundary-free `UK-long` / `EN-long`, `n_prev_docs` is constant `0` if 0 `ĊĊ` boundaries exist or tracked via actual `ĊĊ` paragraph tokens if internal `\n\n` breaks exist; wait — we explicitly inspect BOTH paragraph `\n\n` counts and true inter-document boundaries via article counts (`= 1` for `UK-long`!) — reporting both simple `pos/1000` and full OLS with `\n\n` boundary indicators!).
  2. **"Document concatenation explains the UK positional effect"** iff `UK-long` `pos/1000` 95% CI contains `0` while `UK-concat` `pos/1000` 95% CI is strictly $> 0$.
  3. **"Positional effect is language-specific"** iff `UK-long` `pos/1000` 95% CI is strictly $> 0$ while both `EN-long` and `EN-concat` 95% CIs contain `0`.

## Results

### Cell Structure
- `en_long` (`N=12`): source docs/win = `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` (mean `1.00`); `ĊĊ` breaks/win = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` (mean `0.0`)
- `en_concat` (`N=12`): source docs/win = `[8, 11, 10, 12, 10, 11, 9, 10, 9, 10, 12, 13]` (mean `10.42`); `ĊĊ` breaks/win = `[64, 46, 51, 47, 43, 48, 58, 53, 62, 54, 52, 60]` (mean `53.2`)
- `uk_long` (`N=12`): source docs/win = `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` (mean `1.00`); `ĊĊ` breaks/win = `[12, 18, 18, 20, 18, 18, 17, 21, 18, 19, 18, 27]` (mean `18.7`)
- `uk_concat` (`N=12`): source docs/win = `[4, 3, 1, 5, 3, 4, 1, 1, 5, 6, 4, 2]` (mean `3.25`); `ĊĊ` breaks/win = `[24, 15, 0, 4, 2, 4, 0, 0, 4, 13, 3, 1]` (mean `5.8`)

### 1. PPL & Block PPL Ratios vs `int8_fused` (`ctx = 2048`, `10,000`-rep window bootstrap)

| Cell | Mode | N | int8 PPL | Mode PPL | Full Ratio [95% CI] | B1 (1..512) | B2 (513..1024) | B3 (1025..1536) | B4 (1537..2047) | Δ(B4−B1) dNLL [95% CI] |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `en_long` | `ternary_rot` | 12 | 13.424 | 14.416 | ×1.0739 [1.0636, 1.0854] | ×1.0804 | ×1.0739 | ×1.0750 | ×1.0664 | -0.0130 [-0.0331, +0.0073] |
| `en_long` | `ternary_rot+alpha_mix` | 12 | 13.424 | 13.950 | ×1.0392 [1.0296, 1.0498] | ×1.0461 | ×1.0353 | ×1.0455 | ×1.0300 | -0.0154 [-0.0342, +0.0063] |
| `en_concat` | `ternary_rot` | 12 | 11.404 | 12.108 | ×1.0618 [1.0512, 1.0727] | ×1.0344 | ×1.0552 | ×1.0672 | ×1.0913 | +0.0535 [+0.0330, +0.0741] |
| `en_concat` | `ternary_rot+alpha_mix` | 12 | 11.404 | 11.912 | ×1.0446 [1.0355, 1.0540] | ×1.0244 | ×1.0416 | ×1.0459 | ×1.0670 | +0.0407 [+0.0222, +0.0577] |
| `uk_long` | `ternary_rot` | 12 | 114.332 | 131.094 | ×1.1466 [1.1305, 1.1630] | ×1.1935 | ×1.0918 | ×1.1897 | ×1.1150 | -0.0681 [-0.1034, -0.0347] |
| `uk_long` | `ternary_rot+alpha_mix` | 12 | 114.332 | 128.662 | ×1.1253 [1.1158, 1.1353] | ×1.1466 | ×1.0916 | ×1.1587 | ×1.1058 | -0.0362 [-0.0630, -0.0103] |
| `uk_concat` | `ternary_rot` | 12 | 126.804 | 149.743 | ×1.1809 [1.1495, 1.2108] | ×1.1511 | ×1.1688 | ×1.2080 | ×1.1966 | +0.0387 [-0.0026, +0.0820] |
| `uk_concat` | `ternary_rot+alpha_mix` | 12 | 126.804 | 144.488 | ×1.1395 [1.1097, 1.1641] | ×1.1038 | ×1.1374 | ×1.1597 | ×1.1579 | +0.0478 [+0.0107, +0.0850] |

### 2. Per-Token OLS Positional Slopes (`10,000`-rep window bootstrap, `seed=0`)

| Cell | Mode | N | Simple `pos/1000` [95% CI] | Full `pos/1000` [95% CI] | Full `log1p(dist)` [95% CI] | Full `n_prev_docs` [95% CI] | Pos Effect |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `en_long` | `ternary_rot` | 12 | -0.0019 [-0.0188, +0.0150] | **-0.0323 [-0.0727, +0.0074]** | +0.0208 [+0.0046, +0.0384] | +0.0000 [+0.0000, +0.0000] | no (spans 0) |
| `en_long` | `ternary_rot+alpha_mix` | 12 | -0.0020 [-0.0158, +0.0138] | **-0.0218 [-0.0514, +0.0099]** | +0.0135 [+0.0014, +0.0258] | +0.0000 [+0.0000, +0.0000] | no (spans 0) |
| `en_concat` | `ternary_rot` | 12 | +0.0340 [+0.0182, +0.0487] | **+0.0458 [+0.0073, +0.0997]** | -0.0203 [-0.0293, -0.0120] | -0.0004 [-0.0026, +0.0011] | YES (>0) |
| `en_concat` | `ternary_rot+alpha_mix` | 12 | +0.0245 [+0.0106, +0.0368] | **+0.0364 [+0.0067, +0.0795]** | -0.0162 [-0.0241, -0.0093] | -0.0004 [-0.0020, +0.0006] | YES (>0) |
| `uk_long` | `ternary_rot` | 12 | -0.0234 [-0.0427, -0.0056] | **-0.0488 [-0.1187, +0.0610]** | +0.0043 [-0.0028, +0.0104] | +0.0023 [-0.0095, +0.0079] | no (spans 0) |
| `uk_long` | `ternary_rot+alpha_mix` | 12 | -0.0113 [-0.0258, +0.0038] | **-0.0209 [-0.0697, +0.0517]** | -0.0041 [-0.0122, +0.0029] | +0.0012 [-0.0066, +0.0052] | no (spans 0) |
| `uk_concat` | `ternary_rot` | 12 | +0.0274 [-0.0009, +0.0550] | **-0.0003 [-0.0592, +0.0479]** | +0.0194 [+0.0027, +0.0410] | +0.0030 [-0.0024, +0.0098] | no (spans 0) |
| `uk_concat` | `ternary_rot+alpha_mix` | 12 | +0.0293 [+0.0055, +0.0521] | **+0.0099 [-0.0533, +0.0594]** | +0.0101 [-0.0095, +0.0366] | +0.0033 [-0.0015, +0.0107] | SIMPLE>0 |

### Pre-registered Decision Evaluation

- **`ternary_rot`:** Mixed / nuanced pattern (UK-long full_CI>0=False [simple_CI>0=False], UK-concat=False, EN-long=False, EN-concat=True). (`UK-long` full `pos/1000 = -0.0488 [-0.1187, +0.0610]`, simple `-0.0234 [-0.0427, -0.0056]`; `UK-concat` full `-0.0003 [-0.0592, +0.0479]`; `EN-long` `-0.0323 [-0.0727, +0.0074]`; `EN-concat` `+0.0458 [+0.0073, +0.0997]`).
- **`ternary_rot+alpha_mix`:** Mixed / nuanced pattern (UK-long full_CI>0=False [simple_CI>0=False], UK-concat=False, EN-long=False, EN-concat=True). (`UK-long` full `pos/1000 = -0.0209 [-0.0697, +0.0517]`, simple `-0.0113 [-0.0258, +0.0038]`; `UK-concat` full `+0.0099 [-0.0533, +0.0594]`; `EN-long` `-0.0218 [-0.0514, +0.0099]`; `EN-concat` `+0.0364 [+0.0067, +0.0795]`).

### Reproducibility & Metadata
- Prepare windows: `python3 scripts/prepare_h1b_windows.py` -> `results/2026-11-24_agent2/H1b/windows_h1b.json`.
- Evaluate engine: `python3 scripts/run_h1b_ctx2048.py` (`tenzo-inference -c 2048 --prefill batched`, `--kv-alpha /app/results/2026-11-23_agent2/H1/alpha_mix.txt`).
- Analyze: `python3 scripts/analyze_h1b_ctx2048.py` (`commit a0ecc28`, governor `powersave`, `n_boot=10000`, `seed=0`).
