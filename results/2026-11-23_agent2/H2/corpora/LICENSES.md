# H2 Corpora Licenses and Attribution (`results/2026-11-23_agent2/H2/corpora/`)

Fetched via the HuggingFace Datasets Server API on `2026-11-23` solely for local research evaluation (`2×2` EN/UK × Wiki/Web perplexity & KV-quantization disparity experiments).

## 1. `en_wiki_0_99.json` and `uk_wiki_0_99.json`

- **Source dataset:** `wikimedia/wikipedia` (snapshot configs `20231101.en`, `20231101.uk`; commit `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`)
- **Rows fetched:** `train[0:100]` (`offset=0, length=100`) for each language
- **License:** Creative Commons Attribution-ShareAlike 3.0 Unported (**CC BY-SA 3.0**) and GNU Free Documentation License (**GFDL**), as listed on the Wikimedia Wikipedia HF dataset card (`license: [cc-by-sa-3.0, gfdl]`).
- **Attribution:** Wikipedia contributors / Wikimedia Foundation (`wikimedia/wikipedia`). Derived evaluation token windows are accompanied by attribution to the Wikipedia contributors under CC BY-SA 3.0.

## 2. `en_web_0_99.json`

- **Source dataset:** `HuggingFaceFW/fineweb-edu` (config `sample-10BT`; commit `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`)
- **Rows fetched:** `train[0:100]` (`offset=0, length=100`)
- **License:** Open Data Commons Attribution License v1.0 (**ODC-By 1.0**), as listed on the HF dataset card (`license: odc-by`).
- **Attribution:** Penedo et al. / HuggingFaceFW (`HuggingFaceFW/fineweb-edu`, CommonCrawl educational web subset under CommonCrawl Terms of Use and ODC-By 1.0).

## 3. `uk_web_200_299.json`

- **Source dataset:** `Goader/kobza` (default config; commit `5ec24e168e06ed0e7e0ef70fd734f610ac9ebf93`)
- **Rows fetched:** `train[200:300]` (`offset=200, length=100`, disjoint from `results/reference/kobza_first_rows.json` rows `0..38` and `kobza_rows_100_199.json` rows `100..199`)
- **License (verified directly from `Goader/kobza` dataset card & API):**
  - Dataset card tag: **`cc0-1.0`**.
  - Explicit wording in `Goader/kobza` `README.md` (`📜 License`): the authors state they *"do not own any of the text from which these data has been extracted"* and license *"the actual packaging, the metadata and the annotations of these data under Creative Commons CC0 1.0"*, while providing a **Notice and Take Down policy** (`kaljandlux@gmail.com`) and referencing upstream subcorpus licenses: **CulturaX**, **FineWeb 2** (`ODC-By 1.0`), **HPLT 2.0**, **UberText 2.0**, and **Ukrainian News**.
- **Attribution / Citation:** Mykola Haltiuk & Aleksander Smywiński-Pohl (UNLP 2025, *"On the Path to Make Ukrainian a High-Resource Language"*, `aclanthology.org/2025.unlp-1.14/`). Raw text files remain local/research-only and comply with upstream notice-and-takedown obligations.
