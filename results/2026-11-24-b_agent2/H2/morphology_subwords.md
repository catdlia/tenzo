# H2 (`T47b`): Morphological / Subword Position Degradation Analysis (Exploratory)

## Plan

- **Status:** Exploratory diagnostic analysis on existing `H3` (`results/2026-11-24_agent2/H3/`) score files; does not gate downstream tasks.
- **Machine state (`2026-11-24T14:56:09+02:00`):** `loadavg 3.66 8.17 8.39`, `MemAvailable 10.35 GiB`, `/` `8.6G` free, `governor=powersave`, `AC=1`, `zen-bin/steam/qemu-system` active (`tenzo-b1` waiting cleanly).
- **Inputs (`N = 32` parallel FLORES-200 windows):**
  - Tokenizers: `models/bitnet-1.58b-v2/tokenizer.json` (`BitNet`) and `~/tenzo-refs/hf/eurollm-1.7b/tokenizer.json` (`EuroLLM-1.7B`).
  - Raw parallel texts (used locally via character offsets only, never committed): `~/tenzo-refs/datasets/flores200/windows_flores_texts.json`.
  - Token IDs: `results/2026-11-24_agent2/H3/flores_{bitnet,eurollm}_{en,uk}.ids`.
  - Score files (`S3+s4r32` vs baseline):
    - BitNet: `results/2026-11-24_agent2/H3/scores/bitnet/{fp32,S3_s4r32}_{en,uk}.txt`
    - EuroLLM: `results/2026-11-24_agent2/H3/scores/eurollm/{fp,S3_s4r32}_{en,uk}.txt`
- **Target token classification rule:**
  - In window $w$, each log-probability `lp[t]` ($t = 0 \dots T-2$) predicts target token `ids[t+1]`, matching `tok.encode(text, add_special_tokens=False)` offset `(a, b)` at index `t`.
  - Extract alphabetic word spans using Unicode regex `[^\W\d_]+`.
  - Assign each token overlapping word $w$ to that word; group contiguous tokens belonging to the same word (handles rare byte-level sub-character splits cleanly):
    - `single_token_word`: only 1 token overlaps the word.
    - `word_prefix`: first token of a $\ge 2$-token word.
    - `word_middle`: any interior token ($0 < i < k-1$) of a $\ge 3$-token word.
    - `word_suffix`: last token of a $\ge 2$-token word (morphological endings / inflections typically reside here).
    - `punct_space_num`: overlaps zero alphabetic words (whitespace, punctuation, digits, symbols).
- **Metrics & Bootstrap:**
  - Reported both across **all scored tokens** ($t \ge 0$, exact additive decomposition of window $\Delta\text{NLL}$ and $D$) and across **compressed positions only** ($t \ge 35$, where `S3+s4r32` evicts older KV states beyond 3 sinks + 32 recent tokens).
  - For each `(model, lang, subset, class)`: token count, token share (%), mean per-token delta $\Delta\text{NLL}/\text{tok} = \sum \Delta_{w,c} / \sum n_{w,c}$ with **10,000 paired window-cluster bootstrap 95% CI** (`seed = 0`), class mean window sum $\bar{\Delta}_{w,c}$, and exact share of total $\bar{\Delta}_{\text{doc}}$ / $D = \bar{\Delta}_{\text{doc,UK}} - \bar{\Delta}_{\text{doc,EN}}$.
- **Decision / Interpretation rule (exploratory):**
  - Check whether `word_suffix` (inflectional endings) exhibits substantially higher per-token $\Delta\text{NLL}$ than `word_prefix` / `single_token_word`, OR whether per-token $\Delta\text{NLL}$ is roughly uniform across word-position classes (meaning UK's higher $D$ arises primarily from **having many more split tokens / higher token density per sentence** rather than disproportionately corrupting inflections, or specifically hits inflections/continuations).

## Results (Exploratory, `N = 32`)

- Note on early positions (`t < 35`): in whole-window prefill `S3+s4r32` replaces FP32 with `int8_absmax` on sinks+recent tail (`[0..2] ∪ [T-32..T-1]`) and `tern_g32_rot` on `[3..T-33]`, so `max(|ΔNLL|)` over `t < 35` vs FP32 is non-zero (`BitNet_en=0.980`, `BitNet_uk=0.840`, `EuroLLM-1.7B_en=0.436`, `EuroLLM-1.7B_uk=0.567`). Primary decomposition uses **all scored tokens** (`t >= 0`), matching `D` exactly; `t >= 35` sensitivity is shown right below.

### BitNet (`D = +20.849` nats/win; `Δ_UK = +21.081`, `Δ_EN = +0.232`)

**All scored tokens (`t >= 0`, exact additive decomposition of `D`):**

| Class | EN tok (%) | EN `base/tok` | EN `ΔNLL/tok` [95% CI] | UK tok (%) | UK `base/tok` | UK `ΔNLL/tok` [95% CI] | `D_c` (nats/win) | Share of `D` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_token_word` | 4335 (74.6%) | 3.432 | `+0.00142` [`-0.00848`, `+0.01154`] | 1631 (16.6%) | 5.377 | `+0.06358` [`+0.03249`, `+0.09393`] | `+3.048` [`+1.220`, `+4.926`] | 14.6% |
| `word_prefix` | 276 (4.8%) | 6.164 | `-0.04060` [`-0.08576`, `+0.00568`] | 2468 (25.2%) | 8.361 | `+0.04495` [`+0.01191`, `+0.07775`] | `+3.817` [`+1.213`, `+6.396`] | 18.3% |
| `word_middle` | 103 (1.8%) | 1.794 | `+0.01396` [`-0.03336`, `+0.05970`] | 2255 (23.0%) | 4.930 | `+0.09605` [`+0.05926`, `+0.13530`] | `+6.723` [`+4.123`, `+9.638`] | 32.2% |
| `word_suffix` | 276 (4.8%) | 1.145 | `-0.02671` [`-0.05777`, `+0.00003`] | 2468 (25.2%) | 3.994 | `+0.05213` [`+0.01639`, `+0.08909`] | `+4.251` [`+1.486`, `+7.146`] | 20.4% |
| `punct_space_num` | 820 (14.1%) | 1.712 | `+0.02246` [`+0.01160`, `+0.03477`] | 991 (10.1%) | 3.098 | `+0.11574` [`+0.07671`, `+0.15852`] | `+3.009` [`+1.851`, `+4.158`] | 14.4% |

**Paired contrasts (`all`, 10k window bootstrap):**

| Contrast | Mean (`ΔNLL/tok`) | 95% CI |
|---|---:|---:|
| `en_suffix_minus_prefix` | `+0.01389` | [`-0.03986`, `+0.06828`] |
| `en_suffix_minus_single` | `-0.02813` | [`-0.05974`, `+0.00042`] |
| `en_middle_minus_prefix` | `+0.05456` | [`-0.01714`, `+0.12441`] |
| `UK_minus_EN_tok[single_token_word]` | `+0.06216` | [`+0.03205`, `+0.09147`] |
| `UK_minus_EN_tok[word_prefix]` | `+0.08556` | [`+0.02676`, `+0.14178`] |
| `UK_minus_EN_tok[word_middle]` | `+0.08208` | [`+0.02759`, `+0.13520`] |
| `UK_minus_EN_tok[word_suffix]` | `+0.07884` | [`+0.03391`, `+0.12461`] |
| `UK_minus_EN_tok[punct_space_num]` | `+0.09328` | [`+0.05488`, `+0.13448`] |
| `uk_suffix_minus_prefix` | `+0.00718` | [`-0.03008`, `+0.04625`] |
| `uk_suffix_minus_single` | `-0.01145` | [`-0.05473`, `+0.03623`] |
| `uk_middle_minus_prefix` | `+0.05109` | [`+0.01011`, `+0.09253`] |

**`t >= 35` sensitivity check (`UK ΔNLL/tok`):** `single_token_word=+0.07109`, `word_prefix=+0.04804`, `word_middle=+0.10677`, `word_suffix=+0.05660`, `punct_space_num=+0.13485` (same qualitative pattern; full breakdown in `morphology_subwords.csv`).

### EuroLLM-1.7B (`D = +3.661` nats/win; `Δ_UK = +5.010`, `Δ_EN = +1.349`)

**All scored tokens (`t >= 0`, exact additive decomposition of `D`):**

| Class | EN tok (%) | EN `base/tok` | EN `ΔNLL/tok` [95% CI] | UK tok (%) | UK `base/tok` | UK `ΔNLL/tok` [95% CI] | `D_c` (nats/win) | Share of `D` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_token_word` | 4121 (65.4%) | 3.200 | `+0.00163` [`-0.00878`, `+0.01160`] | 1733 (18.6%) | 3.047 | `+0.02114` [`+0.00638`, `+0.03671`] | `+0.934` [`-0.580`, `+2.493`] | 25.5% |
| `word_prefix` | 490 (7.8%) | 5.590 | `+0.02389` [`-0.01185`, `+0.05914`] | 2366 (25.4%) | 4.839 | `+0.01372` [`-0.00576`, `+0.03658`] | `+0.648` [`-1.022`, `+2.557`] | 17.7% |
| `word_middle` | 140 (2.2%) | 1.173 | `+0.04888` [`-0.00714`, `+0.11190`] | 1645 (17.7%) | 1.476 | `+0.00814` [`-0.00655`, `+0.02244`] | `+0.204` [`-0.632`, `+1.029`] | 5.6% |
| `word_suffix` | 490 (7.8%) | 0.665 | `-0.00421` [`-0.02479`, `+0.01507`] | 2366 (25.4%) | 0.753 | `+0.01506` [`+0.00706`, `+0.02293`] | `+1.178` [`+0.533`, `+1.818`] | 32.2% |
| `punct_space_num` | 1058 (16.8%) | 1.356 | `+0.01886` [`+0.01050`, `+0.02669`] | 1187 (12.8%) | 1.427 | `+0.03555` [`+0.02394`, `+0.04656`] | `+0.695` [`+0.240`, `+1.133`] | 19.0% |

**Paired contrasts (`all`, 10k window bootstrap):**

| Contrast | Mean (`ΔNLL/tok`) | 95% CI |
|---|---:|---:|
| `en_suffix_minus_prefix` | `-0.02810` | [`-0.06458`, `+0.01028`] |
| `en_suffix_minus_single` | `-0.00584` | [`-0.02845`, `+0.01536`] |
| `en_middle_minus_prefix` | `+0.02498` | [`-0.03870`, `+0.08826`] |
| `UK_minus_EN_tok[single_token_word]` | `+0.01950` | [`+0.00147`, `+0.03773`] |
| `UK_minus_EN_tok[word_prefix]` | `-0.01017` | [`-0.05494`, `+0.03682`] |
| `UK_minus_EN_tok[word_middle]` | `-0.04074` | [`-0.10696`, `+0.01938`] |
| `UK_minus_EN_tok[word_suffix]` | `+0.01927` | [`-0.00122`, `+0.04114`] |
| `UK_minus_EN_tok[punct_space_num]` | `+0.01669` | [`+0.00398`, `+0.02898`] |
| `uk_suffix_minus_prefix` | `+0.00134` | [`-0.02497`, `+0.02475`] |
| `uk_suffix_minus_single` | `-0.00607` | [`-0.02183`, `+0.00876`] |
| `uk_middle_minus_prefix` | `-0.00558` | [`-0.03140`, `+0.01895`] |

**`t >= 35` sensitivity check (`UK ΔNLL/tok`):** `single_token_word=+0.02512`, `word_prefix=+0.01626`, `word_middle=+0.00917`, `word_suffix=+0.01729`, `punct_space_num=+0.04009` (same qualitative pattern; full breakdown in `morphology_subwords.csv`).

## Key Findings (Exploratory)

- **BitNet:** Across all UK tokens, per-token error is `prefix=+0.0450`, `middle=+0.0960`, `suffix=+0.0521`, `single=+0.0636`, `punct=+0.1157` nats/tok. `word_suffix` is **not** worse than `word_prefix` (`+0.00718`, 95% CI [`-0.03008`, `+0.04625`]) or `single_token_word` (`-0.01145`, 95% CI [`-0.05473`, `+0.03623`]); on BitNet `word_middle` exceeds `word_prefix` (`+0.05109` [`+0.01011`, `+0.09253`]). Multi-token words generate **70.9%** of $D$ (`middle + suffix` = **52.6%**) reflecting UK's **73.3%** split-word token share, while even `single_token_word` and `punct_space_num` show significant UK-minus-EN per-token penalties in both models.
- **EuroLLM-1.7B:** Across all UK tokens, per-token error is `prefix=+0.0137`, `middle=+0.0081`, `suffix=+0.0151`, `single=+0.0211`, `punct=+0.0356` nats/tok. `word_suffix` is **not** worse than `word_prefix` (`+0.00134`, 95% CI [`-0.02497`, `+0.02475`]) or `single_token_word` (`-0.00607`, 95% CI [`-0.02183`, `+0.00876`]); `word_middle` minus `word_prefix` is null (`-0.00558` [`-0.03140`, `+0.01895`]). Multi-token words generate **55.5%** of $D$ (`middle + suffix` = **37.8%**) reflecting UK's **68.6%** split-word token share, while even `single_token_word` and `punct_space_num` show significant UK-minus-EN per-token penalties in both models.

## Provenance

- Script: `scripts/analyze_h2_morphology.py`
- Tables: `results/2026-11-24-b_agent2/H2/morphology_subwords.csv`, `results/2026-11-24-b_agent2/H2/morphology_contrasts.csv`
- Plot: `results/2026-11-24-b_agent2/H2/morphology_subwords.png`
