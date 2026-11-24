# Tenzo Agent 2 Progress Report (2026-11-24-b)

- **Branch:** `agent2/2026-11-24-b` (branched from `agent2/2026-11-24@b683412fbe47221263c21ed5f9b73933f3364de1`)
- **Start weights checksum (`models/bitnet-1.58b/weights.bin`):** `68c8f32b7f5d905ed7b3aaaf688e8913ed12227c7d9afc1bf3f429e3479400d1`
- **Initial machine state (`2026-11-24T12:36:07+02:00`):** **User at laptop ("власник за ноутом")** — `loadavg` `1.40 1.41 1.46`, `MemAvailable` `7.9 GiB` (`8,318,296 kB`), `/` `8.5G` free (`99%` used), governor `powersave`, AC online `1`, active desktop processes (`zen-bin`, `steam`, `steamwebhelper`, `qemu-system-x86`, `Telegram`, `antigravity`, `claude-desktop`, `jetbrains-toolbox`). Policy: `nice -n 19 ionice -c3`, builds `-j1`, Python under `systemd-run --user --scope -p MemoryMax=... -p MemorySwapMax=0`, checking for idle gaps between tasks for `tenzo-b1`.

## 1. Tasks

| Task | Class | Status | Time | Key Result | Files |
|---|---|---|---|---|---|
| **L0** (Start, snapshot, B1 archive & baseline gates) | L | **done** | ~5 min | Verified `origin/agent2/2026-11-24 == b683412`, branched `agent2/2026-11-24-b`; `weights.bin` sha256 `68c8f32b…00d1`; `G6` (5 KV modes) & `PPL4` (`16.7`, `PROC_HWM_KB=914384`, `sha256: 46489eb1…b6eb5`) byte-identical to `results/2026-11-23_agent2/L4/`. Archived active `tenzo-b1` (`waiting` since `11:10:01`, 0 summary rows) to `B0/previous_run/` | `results/2026-11-24-b_agent2/L0/`, `results/2026-11-24-b_agent2/B0/previous_run/` |
| **B0** (Fix B1 idle runner: E-core `foreign_cpu_mean < 0.15`) | L + harness | **done** | ~10 min | Stopped `tenzo-b1`; verified `lscpu -e` (`0–3` P-cores 4.4 GHz, `4–7` E-cores 3.3 GHz) & container/host `0–3` pinning (`cpuset: "0-3"` / `taskset -c 0-3`); rewrote `scripts/b1_idle_runner.sh` to sample `/proc/stat` `cpu4..7` every `2s` (`foreign_cpu_mean < 0.15` + no forbidden procs before/after, `3` attempts/item returning to idle gate between retries, `failed` advances queue to `done` / `done_with_failures`); real `--smoke` (`pp=16, tg=16`) measured `foreign_cpu_mean=0.1195` (`max=0.1831`, `load1=1.73->2.13`) and marked `invalid_proc` (`qemu-system-x86,steam,zen-bin`, `valid=0`); restarted `tenzo-b1` (`--out results/2026-11-24-b_B1`, live `waiting`) | `scripts/b1_idle_runner.sh`, `.gitignore`, `results/2026-11-24-b_agent2/B0/{runner_fix.md,smoke.log}` |
| **H1** (T47: Equal-character recent window on FLORES BitNet `N=32`) | H | **done** | ~2.2 h | **Verdict at `C*=120`: `PARTIAL`** (CI(R) remains `> 0`, explaining `~22%` of `R`). Equal-token `S3+s4r32` covers `156.9` EN vs `95.6` UK chars (`60.97%`), `R=+1.26% [+0.85%, +1.68%]`. Switching to equal-char `S3+s4c120` (`117.2` EN / `118.5` UK chars; `r_eff` `22.7 / 37.5`; `bits@512` `2.793 / 2.956`) shrinks `R` to **`+0.98% [+0.55%, +1.40%]`** (`R_red=+0.28% [+0.02%, +0.52%]`), leaving **78%** of `R` unexplained; even `S3+s4c240` (`237.2 / 238.5` chars, `3.310` b `@512` UK) leaves `R=+0.37% [+0.13%, +0.62%] > 0`. Equal-char token sweeps confirm gap at fixed chars (`~160` chars: `EN r=32 +0.02%` vs `UK r=54 +0.76%`). `K8V:S3+s4c120` retains `95.3%` of `C=120` `D` (`R=+0.97% [+0.66%, +1.29%]`) | `scripts/run_h1_char_window.py`, `results/2026-11-24-b_agent2/H1/char_window_{coverage,paired}.csv`, `char_window_coverage.{md,png}` |
| **H2** (T47b: Morphological token-class analysis on FLORES BitNet/EuroLLM) | L / exploratory | pending | — | — | — |
| **H3** (T48: 5-language parallel FLORES-200 `N=48`, BitNet + EuroLLM) | H | pending | — | — | — |
| **H4** (T49: Coherent longer parallel `NTREX-128` `ctx=1024`, `N=24`) | H | pending | — | — | — |
| **H3b** (Conditional: Character window $C^*$ across 5 FLORES languages) | H (cond) | pending | — | — | — |
| **L2** (`D1`: `--kv-alpha auto` & `install_kv_alpha_mix.sh`) | L + build | pending | — | — | — |
| **L3** (`B1` speed summary & tables if valid rows exist) | L | pending | — | — | — |
| **L4** (Claims registry, tables, changelog, report & handoff) | L | pending | — | — | — |

## 2. Decision Log

- **D1 (`2026-11-24T12:31` — Coordinator / Session Decisions):** Keep raw `--kv-quant ternary_rot` unchanged; add `--kv-alpha auto` loading `kv_alpha_mix.txt` from model directory (copied by script, not manually); prioritize fixing & restarting `B1` (`B0`) first; no new model downloads, text data `<= 300 MB`, keep `>= 6 GB` free on `/`; exclude `Qwen2.5-1.5B` from `T48` multilingual `H3` (use `BitNet` and `EuroLLM-1.7B`); yield up to 90 min if machine becomes idle between tasks so `tenzo-b1` can run.
- **D2 (`2026-11-24T12:36` — `L0` Branch & Machine State):** Branched `agent2/2026-11-24-b` from `agent2/2026-11-24@b683412`; excluded `Claude outputs/` in `.git/info/exclude`; archived `tenzo-b1` (`Active: active (running)`, `STATUS: waiting 2026-11-24T11:10:01+02:00`, zero benchmark runs attempted yet). Machine: `loadavg 1.40 1.41 1.46`, `MemAvailable 7.9 GiB`, `/` `8.5G` free (`99%`), `governor=powersave`, AC online. Note: `MemAvailable` (`7.9 GiB`) fits `MemoryMax=4G` (`BitNet`) right now (`>= 6 GiB` required) and is close to `8 GiB` needed for `EuroLLM` `MemoryMax=6G` later (`H3`/`H4` will verify `MemAvailable >= 8 GB` before running EuroLLM).
- **D3 (`2026-11-24T12:41` — `H1` Pre-registration Machine State):** `loadavg 1.63 2.10 1.82`, `MemAvailable 8.05 GiB` (`8,438,880 kB >= 4G + 2 GB`), `/` `8.5G` free (`>= 6 GB`), `governor=powersave`, `AC=1`, active `qemu-system-x86,steam,zen-bin` (machine NOT idle -> do not pause for `tenzo-b1`; run `H1` under `MemoryMax=4G -- nice -n 19 ionice -c3`). Pre-registering `H1/char_window_coverage.md` before execution.
- **D4 (`2026-11-24T14:56` — `H2` Pre-registration Machine State):** `loadavg 3.66 8.17 8.39`, `MemAvailable 10.35 GiB` (`10,849,628 kB`), `/` `8.6G` free (`>= 6 GB`), `governor=powersave`, `AC=1`, active `zen-bin,steam,qemu-system-x86` (`tenzo-b1` cleanly `waiting 2026-11-24T12:41:18+02:00`, no idle-yield needed). Running lightweight CPU-only `H2` exploratory analysis (`scripts/analyze_h2_morphology.py`) on existing `H3` (`2026-11-24`) score files (`N=32`).

## 3. Blockers & Disk Watch

- **Disk space on `/`:** `8.5G` free (`>= 6 GB` floor). `FLORES-200` tarball is already cached locally (`~/tenzo-refs/datasets/flores200/`), and `NTREX-128` (`H4`) is a small text download (<10 MB).
- **RAM (`MemAvailable 7.9 GiB` at `L0`):** Fits `4G` Golden runs comfortably; verify `>= 8 GiB` before `6G` EuroLLM runs (`H3`, `H4`).

## 4. What Must NOT Be Claimed

- No speed claims without `valid` rows in `B1` (`foreign_cpu_mean < 0.15`, pinned CPUs `0-3`, no forbidden processes before/after) and accompanying machine metadata.
- No claim that "tokenization explains the EN/UK disparity" unless `H1` passes the `"explains"` decision rule (`95% CI(R) \ni 0` and $|R(C^*)| < |R(r=32)|/3$).
- No universal law on "all high-fertility languages" from `H3` (`2 models × 4 non-EN languages` is exploratory).
- No claim that English "never suffers from ternary KV": on WikiText-2 EN16 `ternary_rot` degrades `×1.071` (`~1.07x`), demonstrating corpus sensitivity.

## 5. Questions to Owner

*(To be finalized with evidence and recommendations at `L4`.)*
