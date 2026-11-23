# Tenzo Agent 2 Initialization Report (2026-11-23)

- **Branch:** `agent2/2026-11-23` (cut from `overnight/2026-11-20@8fe1947`)
- **Target weights checksum (`models/bitnet-1.58b/weights.bin`):** `68c8f32b7f5d905ed7b3aaaf688e8913ed12227c7d9afc1bf3f429e3479400d1` (identical at start and end; `results/2026-11-23_init/weights_sha256.txt`)

---

## 1. Agent and Capabilities

- **Identity:** Antigravity engineering agent, running on **Gemini 4 Argon (Medium)** inside the Antigravity IDE/runtime on Linux.
- **Command execution & background tasks:**
  - Synchronous shell wait per command is up to `10,000 ms` (`WaitMsBeforeAsync`); commands exceeding 10 s automatically continue in the background (`task-*`) and wake the agent via reactive notification upon completion.
  - Background tasks and daemons (`IsDaemon=true`) survive the end of a turn and can notify an active conversation (they do not survive a host shutdown/reboot).
- **Multi-hour autonomous operation:** Supported via reactive wakeup on long background jobs, timers/schedules, and plan-review checkpoints (~30 execution-model steps per cadence), provided the host remains powered and `MemAvailable`/disk limits are respected.
- **Network & Git:** Network access is available; `git push` to `origin` (`https://github.com/catdlia/tenzo.git`) works and was verified on both `overnight/2026-11-20` and `agent2/2026-11-23`.
- **Parallelism & subagents:** Subagents (`research`, `browser`, `self`, custom subagents), workflow scripts (`run_workflow`), and swarms are supported, though **concurrent heavy CPU/memory jobs are strictly avoided** per project memory rules.
- **Known session limits:** `10 s` max synchronous command blocking before backgrounding; `800 lines / 46 KB` per `read_file` slice; ~30 execution steps between planning-model reviews; host RAM/disk constraints documented below.

---

## 2. Environment and Machine Health

| Metric | Observed Value (2026-11-23 ~13:33–13:40 +02:00) |
|---|---|
| OS / Kernel | `Linux illia 7.2.11-arch1-1 #1 SMP PREEMPT_DYNAMIC Fri, 16 Oct 2026 00:28:58 +0000 x86_64 GNU/Linux` |
| CPU (`lscpu`) | `12th Gen Intel(R) Core(TM) i3-1215U`, 6 cores (2P+4E), 8 logical CPUs (`0-7`), 400–4400 MHz, AVX2 / AVX-VNNI / FMA / F16C |
| Power & Governor | AC online (`1`), `scaling_governor = powersave` on `cpu0..cpu7` |
| RAM (`free -m`, start) | `19679` MiB total, `8360` MiB used, `5592` MiB free, `2070` MiB shared, `8137` MiB buff/cache, **`11318` MiB available**; Swap `9839` MiB (`0` used) |
| RAM (`free -m`, end) | `19679` MiB total, `8308` MiB used, `1860` MiB free, `1919` MiB shared, `11842` MiB buff/cache, **`11370` MiB available**; Swap `0` used |
| Load & Active Processes | Start loadavg: `2.27 2.56 2.00 3/1572 7240` (end: `2.22 3.25 2.66`). Active desktop processes: `zen-bin` (35.3% / 16.2% CPU), `antigravity` (~44% combined), `claude-desktop` (11.2%), `Telegram` (8.7%), `jetbrains-toolbox` (4.2%), `qemu-system-x86` (3.9% MEM) |
| Disk (`/`, `/home`) | `/dev/mapper/ArchinstallVg-root`: `476G` size, `468G` used, **`5.6G` avail (`99%` used)** |
| Disk (`/tmp`) | `tmpfs`: `9.7G` size, `776K` used at start (`3.1M` at end), `9.7G` avail (`1%` used) |
| Docker (`docker system df`) | Images: `25` (`5` active, `21.69GB`, **`14.94GB` (68%) reclaimable**); Containers: `5` (`1` active, `0B`); Volumes: `19` (`2` active, `5.337GB`, **`2.642GB` (49%) reclaimable**); Build Cache: `49` (`14.58GB`, **`6.614GB` reclaimable**) |
| Key Directories | `results/` `47M`, `cmake-build-debug/` `995M`, `cmake-build-release/` `79M`, `cmake-build-release-docker/` `19M`, `~/tenzo-refs` `546M` |
| `/tmp/qwen2.5-1.5b` | **Does not exist** (`/tmp` is `tmpfs` and was cleared on reboot after 2026-11-19 shutdown) |

### Risks and Cleanup Proposals (nothing deleted)
1. **Critical `/` disk pressure (`99%` full, `5.6G` free):** Any Docker layer pull, core dump, or HF cache write risks `ENOSPC`.
   - *Proposal:* Reclaim unused Docker build cache (`6.61 GB`) and dangling/unused Docker images (`14.94 GB`) while explicitly preserving `tenzo-dev:latest` and the `ccache_data` volume.
2. **Secondary build directories:** `cmake-build-release/` (`79M`) and `cmake-build-release-docker/` (`19M`) duplicate parts of `cmake-build-debug/` (`995M`).
   - *Proposal:* Confirm whether `cmake-build-release*` are needed; otherwise remove them to recover ~`98M`.
3. **Host RAM contention & `/tmp` (`tmpfs`):** Desktop + `zen-bin` + `claude-desktop` + `qemu-system-x86` already use ~`8.3 GB`. Storing multi-GB models in `/tmp` directly reduces available RAM.
   - *Proposal:* Keep `systemd-run --user --scope -p MemoryMax=<N>G -p MemorySwapMax=0` for all Python jobs > 1 GB and never run two heavy tasks concurrently.
4. **Clean-machine state for benchmarks:** Desktop apps and `load > 2.0` keep `scripts/pending_benchmarks.sh` blocked (`load1 < 1.0`, no browsers/Steam).

---

## 3. Repository State

### Branches and HEADs
| Branch | Local HEAD | Origin HEAD | Notes |
|---|---|---|---|
| `master` | `e075145` | `e075145` | Untouched |
| `experimental` | `b83b77f` | `b83b77f` | Untouched |
| `recovery` | `de66443` | `de66443` | Untouched |
| `overnight/2026-11-18` | `aad222d` | `aad222d` | Untouched |
| `overnight/2026-11-19` | `5e7e235` | `5e7e235` | Untouched |
| `overnight/2026-11-20` | `8fe1947` | `8fe1947` | Closed leftover files from `3dc02f8` -> `8fe1947` |
| `agent2/2026-11-23` *(HEAD)* | `988d662` (+ this init commit) | `988d662` | Branched from `overnight/2026-11-20@8fe1947`; holds Part 2 report compilation and init artifacts |

### Closed Leftovers on `overnight/2026-11-20` (`8fe1947`)
Committed as-is with message `results: interrupted EN+UK mix calibration (incomplete, killed at layer 4) and leftover logs` and pushed:
- `results/2026-11-20_overnight/part2/calib/progress.txt`
- `results/2026-11-20_overnight/part2/calib/alpha_mix.json` (layers 0–3 only)
- `results/2026-11-20_overnight/part2/calib/calib_mix.log` (printed layer 4 before `SIGTERM` / exit 143)
- `results/2026-11-20_overnight/part2/calib/eval_en16/calib_mix.log`
- `results/2026-11-20_overnight/part2/calib/eval_uk32/calib_mix.log`
- `results/2026-11-20_overnight/part2/logs/mem.log`
- `results/2026-11-19_overnight/logs/build_test2.log`

Also appended `## Part 2 (compiled after the fact by the next agent)` to `results/2026-11-20_overnight/OVERNIGHT_REPORT.md` on branch `agent2/2026-11-23` (`988d662`).

---

## 4. Build and Quality Gates

All commands ran inside the `dev` Docker container under `nice -n 19 ionice -c3` (build with `-j1`).

| Gate / Target | Exit Code | Key Result | Log |
|---|---|---|---|
| `ninja -C /app/cmake-build-debug -j1 tenzo-cli tenzo-inference` | `0` | `ninja: no work to do.` (binaries up to date) | `results/2026-11-23_init/logs/build.log` |
| `tenzo-cli test` | `0` | Passed `[1/7]`–`[7/7]` incl. Audit JIT (8/8 kernels), HF semantics, Engine API validation, `ternary_rot` KV (FWHT `2.76e-8`, attn `1.78e-6`), batched prefill bit-identity (12/12 cases) | `results/2026-11-23_init/logs/tenzo_cli_test.log` |
| `scripts/test_fixture_regressions.py` | `0` | All 7 TL1 fixture ref + CLI corruption/acceptance checks passed | `results/2026-11-23_init/logs/test_fixture_regressions.log` |
| `scripts/test_compiler_regressions.py` | `0` | All 5 compiler regression suites passed | `results/2026-11-23_init/logs/test_compiler_regressions.log` |
| `scripts/verify_references.py --all` | `0` | All 6 independent reference verification suites passed | `results/2026-11-23_init/logs/verify_references.log` |
| `scripts/test_pybind_hf_semantics.py` | `0` | `classic_tl1` rel L2 `5.259e-08` vs HF ref; `rope_theta=0` rejected (`ValueError`) | `results/2026-11-23_init/logs/test_pybind_hf_semantics.log` |
| `scripts/g6_check.sh` (5 KV modes) | `0` | **Byte-for-byte MATCH** vs `results/2026-11-20_overnight/part2/g6/after_*.txt` across `fp32`, `int8_fused`, `tl1_fused`, `popcount_fused`, `ternary_rot` | `results/2026-11-23_init/logs/g6_check.log` |

---

## 5. Independent Number Verification and Live Checks

### 5.1 Score-File Recomputations (`results/2026-11-23_init/verify_numbers.py`)
Computed independently from raw `w <id> <n_tokens> <sum_nll>` lines (`PPL = exp(sum_nll / sum_n_tokens)`) without importing `ppl_ci.py` (`results/2026-11-23_init/logs/verify_numbers.txt`).

| Item / File | Metric | Claimed | Recomputed | Rel. Diff | Match (≤ 0.1%) |
|---|---|---|---|---|---|
| `results/reference/golden/nll_golden_float32.txt` | PPL (16 win) | `19.066` | `19.066456` | `+0.0024%` | Yes |
| `results/reference/golden/nll_golden_float32_uk.txt` | PPL (8 win) | `141.911` | `141.910613` | `-0.0003%` | Yes |
| `results/2026-11-19_overnight/ppl_kv_v2/fp32.txt` | PPL (16 win) | `19.124` | `19.124496` | `+0.0026%` | Yes |
| `results/2026-11-19_overnight/ppl_kv_v2/int8_fused.txt` | PPL (16 win) | `19.141` | `19.140996` | `-0.0000%` | Yes |
| `results/2026-11-19_overnight/ppl_kv_v2/tl1_fused.txt` | PPL (16 win) | `54.94` | `54.934731` | `-0.0096%` | Yes |
| `results/2026-11-19_overnight/ppl_kv_v2/popcount_fused.txt` | PPL (16 win) | `207.7` | `207.711832` | `+0.0057%` | Yes |
| `results/2026-11-20_overnight/bitnet_ref/nll_bitnet_rel_en.txt` | PPL (16 win) | `19.059` | `19.058811` | `-0.0010%` | Yes |
| `results/2026-11-20_overnight/part2/ppl_ternary_rot/ternary_rot.txt` | PPL (16 win) | `20.502` | `20.502103` | `+0.0005%` | Yes |
| `part2/kv_final/en16/S13_s4r32.txt` vs golden EN16 | Paired ratio | `1.0104` | `1.010396` | `-0.0004%` | Yes |
| `part2/kv_final/uk32/S3_s4r32.txt` vs `uk32/fp32.txt` | Paired ratio | `1.1512` | `1.151222` | `+0.0019%` | Yes |
| `part2/kv_final/uk32/S13_s4r32.txt` vs `uk32/fp32.txt` | Paired ratio | `1.0960` | `1.096017` | `+0.0015%` | Yes |
| `part2/calib/eval_en16/calib_en.txt` vs golden EN16 | Paired ratio | `1.0399` | `1.039928` | `+0.0027%` | Yes |
| `part2/calib/eval_en16/calib_uk.txt` vs golden EN16 | Paired ratio | `1.0488` | `1.048781` | `-0.0018%` | Yes |
| `part2/calib/eval_uk32/calib_en.txt` vs `uk32/fp32.txt` | Paired ratio | `1.1261` | `1.126099` | `-0.0001%` | Yes |
| `part2/calib/eval_uk32/calib_uk.txt` vs `uk32/fp32.txt` | Paired ratio | `1.1166` | `1.116563` | `-0.0033%` | Yes |
| `part2/second_model/tenzo_int8_en.txt` vs `fp_en.txt` | Paired ratio | `1.016` | `1.015565` | `-0.0428%` | Yes |
| `part2/second_model/S3_s4r32_en.txt` vs `fp_en.txt` | Paired ratio | `1.956` | `1.955917` | `-0.0042%` | Yes |
| `part2/second_model/tenzo_int8_uk.txt` vs `fp_uk.txt` | Paired ratio | `1.041` | `1.040788` | `-0.0204%` | Yes |
| `part2/second_model/S3_s4r32_uk.txt` vs `fp_uk.txt` | Paired ratio | `2.774` | `2.773724` | `-0.0100%` | Yes |

All 19 checks match within `0.043%` (no discrepancies `> 0.1%`).

### 5.2 Live Verification Checks
1. **`tenzo-inference --score` (`int8_fused`, EN windows 0–3, `-c 512`, `models/bitnet-1.58b-v2`):**
   - `MemAvailable` at start: `11,141 MB` (≥ 5 GB).
   - Exit `0`; `PROC_RSS_KB=897612`, `PROC_HWM_KB=897612` (`876.6 MiB`).
   - Recomputed PPL over `2,044` tokens: **`16.700700`**, **byte-identical** (`cmp -s`) to `results/2026-11-20_overnight/part2/rss/ppl4_int8_fused.txt` (`results/2026-11-23_init/logs/live_ppl4_int8_fused.{txt,log}`).
2. **Release `bitnet.cpp` (`~/tenzo-refs/BitNet/build/bin/llama-cli`, greedy 24 tokens on `"The capital of France is"`):**
   - `MemAvailable` at start: `11,404.3 MB` (≥ 5 GB).
   - Exit `0`; output: `" Paris. Paris is a city that is known for its rich history, culture, and architecture. It is also a major"` (`results/2026-11-23_init/logs/live_bitnet_cpp_g6.log`).

---

## 6. Understanding of Project State

The numerical correctness and memory footprint of `TenzoEngine` for `BitNet-b1.58-2B-4T` are now well established: after the `rope_theta` (`500000`) and 8-bit `ActQuant` fixes, `fp32` (`19.124`) and `int8_fused` (`19.141`) PPL are within `0.4%` of both the independent `golden_bitnet.py` fp32 reference (`19.066`) and the rebuilt release `bitnet.cpp` (`19.059`), peak RSS at `ctx 2048` dropped from `2761 MB` to `935 MB`, and the new `gemv_i2s` and batched prefill paths are bit-identical to their baselines. However, **all post-fix speed claims remain unverified** because no clean-machine run (`load1 < 1.0`, no browsers/Steam) has occurred yet. On the KV compression track, `<= 2.5`-bit ternary PTQ still misses the `<= 5%` PPL target (`ternary_rot` / `S3+s4r32` is `+7.5%` EN / `+16.7%` UK in-engine, `+4.0%` EN / `+11.7%` UK when calibrated in simulation, and collapses to `1.96x`–`2.77x` on `Qwen2.5-1.5B`), and the EN+UK mix calibration was cut short at layer 4 by host shutdown. Remaining structural gaps are the unattainable `max_abs <= 0.25` logit parity gate, the lack of `ctx 2048` PPL measurements, and the greedy longest-match `Tokenizer.cpp` replacing real BPE.

---

## 7. Open Tasks (Prioritized)

| Priority | Task | Scope / Why | Est. Time |
|---|---|---|---|
| **1** | **Clean-machine speed benchmark queue** | Execute `scripts/pending_benchmarks.sh` (`ab_gemv_pp{128,512}`, `e2e_fair_v3`, `ab_prefill_pp{128,512}`, `decode_kv_modes`) once the laptop is idle (`load1 < 1.0`, no browsers/Steam), then regenerate `results/paper/tables.{md,tex}` | 60–90 min |
| **2** | **Complete EN+UK calibration & wire `TENZO_KV_ALPHA`** | Resume `scripts/run_kv_calib.sh results/2026-11-20_overnight/part2/calib 100` (from scratch for `mix`, since `alpha_mix.json` stopped at layer 3), evaluate `eval_en16`/`eval_uk32`, export via `export_kv_alpha.py`, and measure `tenzo-inference --kv-quant ternary_rot` with `TENZO_KV_ALPHA` | 60–90 min |
| **3** | **Language disparity normalized per byte / char** | Test whether Ukrainian's larger KV degradation persists on equal UTF-8 byte / character counts across `BitNet-2B-4T` and `Qwen2.5-1.5B` (since per-token NLL differs across tokenizers) | 45–60 min |
| **4** | **Parity gate v2** | Define and implement a realistic multi-tier logit parity gate once the owner decides thresholds (replacing `max_abs <= 0.25`, which fails even between golden fp32 and HF bf16) | 30–45 min |
| **5** | **Context 2048 PTQ & engine quality checks** | Run `S3+s4r32`/`S13+s4r32` and `ternary_rot` vs `int8_fused` at `ctx 2048` (currently only `ctx 512` is measured; `32` recent INT8 tokens may represent a much smaller fraction at longer contexts) | 60–90 min |
| **6** | **True byte-level BPE tokenizer (`Tokenizer.cpp`)** | Replace greedy longest-match in `Tokenizer::encode` with real BPE merges (`bpe_ranks_` is already loaded but unused!) and add full GPT-2 byte-to-UTF-8 decoding in `Tokenizer::decode` | 90–120 min |

---

## 8. Discrepancies and Suspicions

1. **Mix calibration kill cause (`progress.txt` & `eval_{en16,uk32}/calib_mix.log`):** The previous session recorded `calib mix exit=143 16:30:04` and immediate `exit=1` on `eval_en16` / `eval_uk32`. Reading `eval_en16/calib_mix.log` and `eval_uk32/calib_mix.log` reveals `systemd-run` failed with `Transaction for run-...scope/start is destructive (shutdown.target has 'start' job queued, ...)` — i.e., a **host shutdown/reboot** killed `kv_calib.py` via `SIGTERM` right after layer 4 printed to `calib_mix.log` (before `kv_calib.py:132` saved layer 4 to `alpha_mix.json`, leaving only layers 0–3 in `alpha_mix.json`).
2. **Latent `q_dim != hidden_size` bug in `TenzoEngine.cpp` (`prefill_tokens_batched` line 2198 & decode line 2102):** `rms_norm_raw(Ab.data() + size_t(m) * QD, lw.attn_sub_norm_ptr, An.data() + size_t(m) * QD, h);` normalizes `h` (`hidden_size`) elements instead of `QD` (`q_dim`). This is harmless for `BitNet-b1.58-2B-4T` (`20 * 128 == 2560 == hidden_size`), but would mis-normalize if a future model has `q_dim != hidden_size`. There is also an unused `Sc(C)` vector on line 2161.
3. **`-Waggressive-loop-optimizations` compiler warning (`part2/logs/build_t34.log`, `TenzoEngine.cpp:226,236`):** `gemv_i2s_rows` triggers GCC/Clang warning `iteration 274877906883 invokes undefined behavior` on `(row0 + r) * row_bytes` pointer arithmetic; likely benign for `nrows = 64`, but worth cleaning up.
4. **Stale timestamp on `cmake-build-debug/tenzo_runtime.cpython-312-x86_64-linux-gnu.so`:** Timestamp is `Nov 18 21:16` (vs `Nov 19 14:43` for `tenzo-cli` / `tenzo-inference`). `test_pybind_hf_semantics.py` passes, but `tenzo_runtime` was not rebuilt by `build_main_targets` or `make build-local` in Part 2.
5. **Paper tables metadata lag (`results/paper/tables.md`):** Footer states `Commit 33f6281 (+uncommitted changes)` (`2026-11-19 16:23`), predating commits `65c9ea8`, `3dc02f8`, `8fe1947`, and `988d662`. Re-running `scripts/build_paper_tables.py` is trivial once pending runs finish.
6. **Tokenizer implementation gap (`src/runtime/Tokenizer.cpp`):** `Tokenizer::load` populates `bpe_ranks_` (`line 40`) which `Tokenizer::encode` never uses (using greedy longest-match instead), and `Tokenizer::load` only converts `"Ġ"` (`" "`) without handling `"Ċ"` (`"\n"`) or full GPT-2 byte-unicode decoding — explaining both Ukrainian over-segmentation and `"Ãİle-de-France"` output.
7. **Mislabelled commit `e367633` (`decision_log.md` D14):** Message reads `results: EN-calibrated ternary KV (interim)` while containing initial unbuilt engine changes (fixed and verified in `545d24a`, `e095a78`, `8e3fcf5`). Preserved untouched per no-history-rewrite rule.
8. **Shared GGUF dependency across `BitNet/` and `~/tenzo-refs/BitNet`:** Although `BitNet/` in the repo is broken for execution, `scripts/run_e2e_fair.py` and `~/tenzo-refs/BitNet` still read `BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf` (`1.1 GiB`). Do not delete `BitNet/models/` if cleaning up `BitNet/`.

---

## 9. Questions to the Owner

1. **Parity gate v2:** Since `max_abs <= 0.25` fails even between `golden_bitnet.py` (`fp32`) and HF `transformers` (`bf16`) (`0.53–1.63`), what exact replacement gate should we adopt (e.g., `cos > 0.999` + KL/PPL tolerance, or an optional FP16/FP32 LM head)?
2. **Clean-machine benchmark window:** When can the laptop be left idle (`load1 < 1.0`, no browsers/Steam) for 60–90 min to run `scripts/pending_benchmarks.sh`, and do you want the governor left at `powersave` or manually set to `performance`?
3. **Disk space cleanup:** `/` is at `99%` (`5.6G` free). May we prune unused Docker build cache / old images (~`24.2 GB` reclaimable in `docker system df`, keeping `tenzo-dev:latest` and `ccache_data`) and/or `cmake-build-release*` (`98M`)?
4. **Mix calibration & second model:** Should we complete `alpha_mix.json` from scratch (starting fresh rather than resuming the 4-layer file) and wire calibrated alphas into `ternary_rot`? Also, since `/tmp/qwen2.5-1.5b` disappeared on reboot, do you want `Qwen2.5-1.5B` re-downloaded later for byte-normalized disparity checks?
5. **Tokenizer priority:** Where does implementing true byte-level BPE + UTF-8 decoding in `src/runtime/Tokenizer.cpp` sit relative to `ctx 2048` KV evaluation and speed benchmarking?
