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
