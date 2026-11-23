# L4 — Full Llama-3 Byte-Level BPE Tokenizer (`tenzo::runtime::Tokenizer`)

Replaced the greedy prefix matcher (`BpeTokenizer` in `src/main_inference.cpp` and greedy legacy lookup in `src/runtime/Tokenizer.cpp`) with a dependency-free C++17 HuggingFace `tokenizer.json` Llama-3 BPE implementation wired into `tenzo-inference` and `tenzo-cli`:

- **Pre-tokenizer (`src/runtime/UnicodeTables.h`, generated via `tools/tokenizer/gen_unicode_tables.py`):** Exact Unicode 16.0.0 `\p{L}` (`677` ranges), `\p{N}` (`144` ranges), and HF `White_Space` (`25` codepoints, verified `0` mismatches against HF `tokenizers` `Split` across `293,963` BMP + assigned astral codepoints), implementing all 7 Llama-3 regex branches including contractions (`(?i:'s|'t|'re|'ve|'m|'ll|'d)`), 1–3 digit chunking (`\p{N}{1,3}`), newline runs (`\s*[\r\n]+`), and trailing-whitespace lookahead (`\s+(?!\S)`).
- **`added_tokens` & BPE merges (`src/runtime/Tokenizer.{h,cpp}`):** Leftmost-longest splitting for `256` special tokens (`128000..128255`), GPT-2 `ByteLevel` bijection (`128,000` vocab entries mapped 1-to-1 to raw bytes), `ignore_merges=True` fast path, and rank-ordered merge table (`280,147` merges).
- **UTF-8 decode & asset resolution:** `decode()` / `decode_token()` apply the inverse GPT-2 byte-to-unicode map so multi-byte sequences (e.g., `Île-de-France`, Ukrainian Cyrillic) reconstruct exact raw UTF-8 without `Ãİ` mojibake. Loading `models/bitnet-1.58b*/tokenizer.vocab` automatically upgrades `>=128k` vocabs using `TENZO_TOKENIZER_JSON` or `tools/tokenizer/bitnet_b158_tokenizer.json` without modifying `models/`, while preserving legacy fallback for small mock fixtures. Added `--tokenizer` and `--tokenize-batch` CLI flags to `src/main_inference.cpp`.

## Verification Results

1. **Build:** `make build-local JOBS=1` succeeded (`exit=0`) with **`0` compiler warnings** (`results/2026-11-23_agent2/L4/build.log`).
2. **Parity Corpus (`scripts/test_tokenizer_parity.py`):**
   - **`3,200 / 3,200`** exact match (`100.0%`) with `add_special_tokens=True` (`bos_exact=3200`)
   - **`3,200 / 3,200`** exact match (`100.0%`) with `add_special_tokens=False` (`raw_exact=3200`)
   - **`3,200 / 3,200`** exact UTF-8 round-trip (`decode(encode(s, false)) == s`)
   - Breakdown: `adversarial` (`34/34`), `wikitext2_en` (`900/900`), `kobza_uk` (`1306/1306`), `code` (`960/960`) (`results/2026-11-23_agent2/L4/tokenizer_parity.json`).
3. **Golden Prompts (`scripts/verify_model_parity.py --only-tokenization`):** **`7 / 7` prompts `identical=True`** vs HF (`p1_fact_en` `6/6`, `p2_tech_en` `18/18`, `p3_code` `18/18`, `p4_ukrainian` `18/18`, `p5_long_ctx` `343/343`, `p6_weird` `50/50`, `p7_math_en` `19/19`).
4. **C++ & Regression Gates (`results/2026-11-23_agent2/L4/`):**
   - `tenzo-cli test` (including new `runTokenizerTests()` checking `" Île-de-France"`, Ukrainian, contractions, whitespace, added tokens): `exit=0`.
   - 4 Python suites (`test_fixture_regressions.py`, `test_compiler_regressions.py`, `verify_references.py --all`, `test_pybind_hf_semantics.py`): all `exit=0`.
   - `G6` greedy check across all 5 KV modes (`fp32, int8_fused, tl1_fused, popcount_fused, ternary_rot`): **5/5 byte-identical** to `results/2026-11-20_overnight/part2/g6/after_*.txt`.
   - `PPL4 int8_fused`: **`16.700700`**, **byte-identical** to `results/2026-11-23_init/logs/live_ppl4_int8_fused.txt`.
   - Single-shot generation on French (`Paris is the capital of the Île-de-France`) and Ukrainian (`Київ — столиця`) produces clean UTF-8 output (`results/2026-11-23_agent2/L4/utf8_decode_{fr,uk}.log`).
