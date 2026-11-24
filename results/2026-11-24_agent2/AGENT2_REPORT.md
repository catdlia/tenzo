# Tenzo Agent 2 Progress Report (2026-11-24)

- **Branch:** `agent2/2026-11-24` (branched from `agent2/2026-11-23@42a8d4cb0275ebdaf31985932f44dcd7671691d1`)
- **Start weights checksum (`models/bitnet-1.58b/weights.bin`):** `68c8f32b7f5d905ed7b3aaaf688e8913ed12227c7d9afc1bf3f429e3479400d1`
- **Initial machine state (`2026-11-24T02:03:38+02:00`):** **User at laptop ("власник за ноутом")** — `loadavg` `2.33 2.09 2.22`, `MemAvailable` `9.5 GiB`, `/` `8.5G` free (`99%` used), governor `powersave`, active `antigravity`, `Telegram`, `Hyprland`, `zen-bin`, `qemu-system-x86`, `claude-desktop`, `steam`, `jetbrains-toolbox`. Mode: `nice -n 19 ionice -c3`, builds `-j1`, Python under `systemd-run --user --scope -p MemoryMax=... -p MemorySwapMax=0`, `B1` via self-gated idle runner at end of session.

## 1. Tasks

| Task | Class | Status | Time | Key Result | Files |
|---|---|---|---|---|---|
| **L0** (Start, snapshot & L4 baseline gates) | L | **done** | ~5 min | Branched `agent2/2026-11-24` from `42a8d4c`; `weights.bin` sha256 matches `68c8f32b…00d1`; G6 (5 KV modes) & PPL4 (`16.7`, `PROC_HWM_KB=913864`) byte-identical to `results/2026-11-23_agent2/L4/` (`sha256: 46489eb1…b6eb5`) | `results/2026-11-24_agent2/L0/` |
| **L1** (T43: Repo hygiene, licenses, `kv_alpha` header) | L + build | **done** | ~10 min | Untracked `tools/tokenizer/bitnet_b158_tokenizer.json` (`git rm --cached`, `.gitignore`), copied to `models/bitnet-1.58b-v2/tokenizer.json` (`sha256: e134af98…6cc7`), added [tools/tokenizer/README.md](tools/tokenizer/README.md) & [LICENSES.md](../2026-11-23_agent2/H2/corpora/LICENSES.md) (`kobza`: packaging `CC0-1.0`, underlying text subcorpora w/ takedown notice); added `--kv-alpha` & `# kv_alpha=<path> sha256=<hex>` header; `make build-local JOBS=1` `0` warnings; `tenzo-cli test`, `3200/3200` tokenizer parity, G6 5/5 & PPL4 byte-identical, `--kv-alpha` == `TENZO_KV_ALPHA` (`sha256: d67ed67b…3eeb`) | `.gitignore`, `models/bitnet-1.58b-v2/tokenizer.json`, `tools/tokenizer/README.md`, `results/2026-11-23_agent2/H2/corpora/LICENSES.md`, `src/main_inference.cpp`, `src/runtime/Tokenizer.{h,cpp}`, `src/tests/TestEngineAPI.cpp`, `scripts/{test_tokenizer_parity,verify_model_parity}.py`, `results/2026-11-24_agent2/L1/` |
| **H1a** (Positional UK degradation on `H3 ctx=2048`) | L (analysis) | **done** | ~6 min | Replicated coordinator numbers on UK `ternary_rot` (boundaries `[2,2,0,57]`, eq blocks `0.131,0.157,0.216,0.267`, `dist<=256` `0.096..0.175`, boundary-free `w=2` `0.159→0.240`, OLS `pos/1000=+0.0831`, `n_prev_docs=-0.0002`); simple UK `pos/1000` CI > 0 (`+0.0936 [+0.062,+0.125]`; `alpha_mix` `+0.0832 [+0.054,+0.113]`), full OLS `pos/1000` CIs span 0 at `N=4` (`+0.0831 [-0.032,+0.132]`); EN flat (`-0.0092 [-0.108,+0.068]`) | `scripts/analyze_h3_position.py`, `results/2026-11-24_agent2/H1a/position_h3.{md,csv,png}` |
| **H3** (T45: Parallel corpus FLORES-200 EN/UK, 3 models) | H | **done** | ~2.7 h | `N=32` aligned FLORES-200 pairs across **BitNet-2B**, **EuroLLM-1.7B**, **Qwen2.5-1.5B** (`0.000e+00` layer-major parity vs `B=1`). On `S3+s4r32`, **all 3 models** show significant super-proportional UK penalty at identical meaning: BitNet `D=+20.849 [+14.82,+26.90]` nats (`R=+1.26% [+0.85,+1.68%]`), EuroLLM `D=+3.661 [+0.94,+6.47]` (`R=+0.53% [+0.10,+0.97%]`), Qwen2.5 `D=+453.63 [+423.18,+485.49]` (`R=+31.99% [+27.55,+36.38%]`). BitNet `S13+s4r32` `D=+20.949 [+17.43,+24.51]`, EuroLLM `S13` `D=+0.838 [-0.76,+2.47]`. Resolves `T45`: `EuroLLM S3 ΔBPB` crossed 0 (`+0.0022`) only from UTF-8's `~1.82×` byte/char ratio | `scripts/{prepare,run,analyze}_h3_flores.py`, `results/2026-11-24_agent2/H3/parallel_flores.{md,csv,png}`, `flores_*.ids` |
| **H1b** (T44: Long document vs concat on `ctx=2048`) | H | pending | — | — | — |
| **H2** (Cross-domain transfer of `alpha_mix` on BitNet `2×2`) | H | pending | — | — | — |
| **H4** (T46: Mechanism ablations & diagnostics on FLORES-12) | H | pending | — | — | — |
| **L2** (T02: Parity gate `v2` INT8 embeddings/head diag) | L | pending | — | — | — |
| **L3** (Paper claims registry) | L | pending | — | — | — |
| **L4** (End-of-day wrap-up, tables, changelog, report) | L | pending | — | — | — |
| **B1** (T38: Self-gated idle speed runner `tenzo-b1`) | B (daemon) | pending | — | — | — |

## 2. Decision Log

- **D1 (`2026-11-24T01:57` — Owner D1: Parity gate `v2`):** Threshold `1 - cos <= 1.5 * (1 - cos_ref)` stays strict (no relaxation, no noise floor such as `5e-4`). Report `v2` honestly as `max_abs 7/7, PPL ratio CI OK, cos 6/7` and diagnose whether INT8 embeddings + INT8 LM head account for `p4`/`p2` (`L2`).
- **D2 (`2026-11-24T01:57` — Owner D2: `alpha_mix.txt` & provenance):** Default `--kv-quant ternary_rot` behavior stays unchanged; calibrated alpha remains opt-in (`TENZO_KV_ALPHA` or `--kv-alpha <path>`). Engine must record `# kv_alpha=<path> sha256=<hex>` in score file headers (`L1`). Default decision awaits `H2` domain-transfer results.
- **D3 (`2026-11-24T01:57` — Owner D3: Downloads & disk):** No new models downloaded; only text corpora totaling `<= 200 MB` allowed while keeping `>= 6 GB` free on `/` (`df -h /` checked before each download).
- **D4 (`2026-11-24T01:57` — Owner D4: Tokenizer tree hygiene):** Remove `tools/tokenizer/bitnet_b158_tokenizer.json` from git index (`git rm --cached`) without deleting files on disk (`L1`).
- **D5 (`2026-11-24T01:57` — Owner D5: Benchmarks):** Speed benchmarks run via self-gated background runner `scripts/b1_idle_runner.sh` (`tenzo-b1` user unit) launched at the end of session (`B1`). Runner neither commits nor pushes.
- **D6 (`2026-11-24T02:03` — `L0` Machine State):** `loadavg 2.33 2.09 2.22`, `MemAvailable 9.5 GiB`, `/` `8.5G` free (`99%`), governor `powersave`, active desktop apps (`zen-bin`, `Telegram`, `steam`, `qemu-system-x86`, `claude-desktop`, `jetbrains-toolbox`). G6 (all 5 modes) and PPL4 match `results/2026-11-23_agent2/L4/` byte-for-byte.
- **D7 (`2026-11-24T02:14` — `H1a` Pre-registration Machine State):** `loadavg 2.71 3.62 3.09`, `MemAvailable 10.0 GiB`, `/` `8.5G` free, governor `powersave`. CPU-light Python OLS/bootstrap analysis over existing `ctx=2048` score files.
- **D8 (`2026-11-24T02:16` — `H3` Pre-registration Machine State):** `loadavg 2.42 3.23 3.00`, `MemAvailable 10.0 GiB` (`MemAvailable - 2 GB = 8.0 GiB`, fits `4G` Golden and `6G` HF scopes sequentially), `/` `8.5G` free (`>= 6G` limit under D3), governor `powersave`. Pre-registering `H3/parallel_flores.md` before downloading `flores200_dataset.tar.gz`.
- **D9 (`2026-11-24T05:07` — `H1b` Pre-registration Machine State):** `loadavg 2.55 4.34 5.30`, `MemAvailable 10.0 GiB`, `/` `8.7G` free (`>= 6G` limit under D3), governor `powersave`. Engine-only `ctx=2048` runs (`~950 MB` peak RSS) run sequentially (`1` container at a time) under `nice -n 19 ionice -c3`. Pre-registering `H1b/long_vs_concat_2048.md` before fetching Wikipedia/Kobza slices.

## 3. Blockers & Disk Watch

- **Disk space on `/`:** `8.5G` free (`464G / 476G`, `99%` used) at `L0` — above `6 GB` floor for text corpora (`H3` FLORES-200 ~25 MB and `H1b` Wikipedia/Kobza slices).
- **RAM (`MemAvailable 9.5 GiB`):** Fits `4G` golden (`<= 6G` needed) and `6G` HF models (`<= 8G` needed) sequentially under `systemd-run` memory scopes; never run two heavy processes concurrently.

## 4. What Must NOT Be Claimed

- No post-fix speed numbers until `B1` (`tenzo-b1`) completes on an idle machine and the next session verifies `STATUS` and CSVs.
- No `7/7` pass claim for parity gate `v2`.
- No claim that UK positional degradation (`ctx=2048`) is a language property unless `H1b` `UK-long` slope 95% CI exceeds `0` (with `N >= 8`).
- On `H3` (`FLORES-200`), `CI(D) > 0` holds across all three models under `S3+s4r32`; do NOT overclaim `S13+s4r32` (`EuroLLM S13` `D` crosses zero) or `tenzo_int8` on BitNet/EuroLLM (`D ~ 0`), and keep the token-length / `s4r32` protected-fraction caveat explicit pending `H4`.
- No claim that `alpha_mix` is the default for `ternary_rot`.

## 5. Questions to Owner

*(To be finalized with evidence and recommendations at `L4`.)*
