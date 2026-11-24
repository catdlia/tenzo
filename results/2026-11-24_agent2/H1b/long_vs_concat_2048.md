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
