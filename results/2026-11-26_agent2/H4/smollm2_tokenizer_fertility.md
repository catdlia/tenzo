# H4 Tokenizer Pre-Check: `HuggingFaceTB/SmolLM2-1.7B` Fertility (`2026-11-26`)

- **Scope (D3 / H4 cheap step):** Downloaded ONLY `tokenizer.json` and `tokenizer_config.json` from `HuggingFaceTB/SmolLM2-1.7B` into `~/tenzo-refs/hf/smollm2-1.7b/` (no model weights downloaded tonight; status `exploratory` tokenizer check only).
- **Disk check (`/` free):** `11.45 GiB` free -> **1.7B feasible (>= 9.5 GB)** (rule: `>= 9.5 GB` for `1.7B`, else `SmolLM2-360M`).

## `φ_UK = tokens_UK / tokens_EN` Across Models (Identical Content)

| Corpus | `N` | Tokenizer / Model | Mean tok EN | Mean tok UK | Max tok UK | `φ_UK` | `UK > 1023` | `UK > 2047` |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `flores32` | 32 | `bitnet` | 181.56 | 306.66 | 349 | **1.6890×** | 0 | 0 |
| `flores32` | 32 | `eurollm` | 196.84 | 290.53 | 343 | **1.4759×** | 0 | 0 |
| `flores32` | 32 | `qwen` | 186.06 | 471.53 | 511 | **2.5343×** | 0 | 0 |
| `flores32` | 32 | `smollm2` | 188.50 | 739.19 | 819 | **3.9214×** | 0 | 0 |
| `ntrex24` | 24 | `bitnet` | 600.33 | 986.38 | 1017 | **1.6430×** | 0 | 0 |
| `ntrex24` | 24 | `eurollm` | 672.75 | 962.33 | 1021 | **1.4304×** | 0 | 0 |
| `ntrex24` | 24 | `smollm2` | 644.54 | 2387.21 | 2667 | **3.7037×** | 24 | 24 |

## Implications for Next-Night Full `H4` Run
- `FLORES-32` fits cleanly within `ctx = 1024` if `uk_gt_1023 == 0` (otherwise report exact drops/trimming).
- For `NTREX-24` (`ctx = 2048` budget in full `H4`), any UK article exceeding `2047` scored tokens under SmolLM2 (`uk_gt_2047` above) must be truncated cleanly at an aligned EN/UK sentence boundary and reported explicitly so content remains parallel.
