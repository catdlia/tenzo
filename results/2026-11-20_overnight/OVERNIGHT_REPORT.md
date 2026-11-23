# Overnight report 2026-11-20 (branch overnight/2026-11-20)

## Part 1 (01:00–02:30)

The task text of part 1 was cut off mid-sentence in block A (decision log D1). Blocks were therefore done as follows:
- **A:** I2_S GEMV with a bit-identity gate.
- **B:** the bitnet.cpp baseline per the previous night's spec.
- **C:** ternary-KV round 2, following kv_ptq.md.

### 1. Blocks

| block | status | key result | files |
|---|---|---|---|
| start | done | build ok, `tenzo-cli test` (incl. hf-semantics) exit 0, G6 baseline = 2026-11-19 after-fix outputs | `meta.txt`, `g6/before_*`, `logs/start_tenzo_cli_test.log` |
| A: I2_S kernel | **correctness done, speed not measured** | maddubs GEMV (u = w + 1, −Σq); outputs **bit-identical** to the LUT kernel: logits and 32-token greedy on w0/p5 in fp32/int8 KV, 2-window scores, G6 in 4 modes. hf-semantics passes with both kernels. Default `TENZO_GEMV=i2s`, `lut` kept as fallback, other values rejected. TL1 copies freed on the CPU i2s path (steady RSS i2s = lut ≈ 0.9 GB) | `logs/i2s_bitwise_identity.txt`, `logs/hf_semantics_*.log`, `logs/gates_i2s/`, `logs/rss_lut_vs_i2s.txt` |
| A: speed A/B | **skipped** | clean-machine rule failed 3/3 (firefox/zen/neko open, load 4–13) | `logs/clean_machine_checks.log` |
| B: bitnet.cpp baseline | **done** | microsoft/BitNet@c17d1c5 + llama.cpp 5eb47b7, clang 21: greedy == HF; PPL 19.059 EN (golden 19.066), 141.8 UK (golden 141.9) | `bitnet_ref.md`, `bitnet_ref/` |
| B: clean e2e v3 | **skipped** | same clean-machine blocker | – |
| C: KV PTQ round 2 | **screening done, finalists not run** | best: K INT4 + V ternary(rot) + INT8 sinks4/recent32: +1.7% EN / +7.0% UK at ~4 b/elem; no ≤2.5-bit scheme below +9.9% EN / +17% UK; ternary V and missing INT8 windows drive the UK loss | `kv_ptq2.md`, `kv_ptq2/` |

### 2. Negative results

- **Speed.** No speed numbers were taken tonight: the machine was never clean.
- **Round-2 changes give small gains.** The LS-optimal ternary threshold, g=16 and rotated per-channel K all give small or no gains on EN.
- **Ternary V is the UK problem.** It costs about 30% on UK even with INT4 K.

### 3. Commits (part 1)

```
9d3d0ab results: ternary KV PTQ round 2 screening (EN + UK)
(perf) free the TL1 weight copies on the CPU i2s path
(feat) KV PTQ round-2 schemes
0b78839 results: bitnet.cpp reference rebuild
5e1bff0 feat(engine): I2_S-style maddubs ternary GEMV (TENZO_GEMV=i2s|lut)
```

---

## Part 2 (compiled after the fact by the next agent)

> **Reconstruction notice.** This Part 2 summary was compiled after the fact (2026-11-23) by the succeeding engineering agent strictly from existing Part 2 artifacts (`results/2026-11-20_overnight/part2/*.md`), `results/2026-11-20_overnight/decision_log.md`, `results/2026-11-20_overnight/part2/calib/progress.txt`, and `git log 00fe9e3..8fe1947`, with headline PPL and ratio numbers restricted to those independently recomputed from raw score files during initialization.

Part 2 ran in quiet mode (`nice -n 19 ionice -c3`, `-j1` builds, Python `MemoryMax=3G` or `4G` for Qwen2.5-1.5B) starting at `02:26` (2026-11-19), paused from `03:15` to `11:18`, and ended at `16:30:04` when the host initiated shutdown (`shutdown.target`), sending SIGTERM (`exit 143`) to the resumed EN+UK mix calibration at layer 4. Weights `models/bitnet-1.58b/weights.bin` sha256 remained `68c8f32b7f5d905ed7b3aaaf688e8913ed12227c7d9afc1bf3f429e3479400d1` (`part2/weights_sha256_start.txt` and post-run check).

### 1. Blocks

| block | status | key result | files |
|---|---|---|---|
| Triage & clean-machine speed queue | **done (runs queued, not executed)** | Deferred clean-machine benchmarks collected into `scripts/pending_benchmarks.sh`; machine remained loaded (`load > 1` / user active) | `part2/triage.md`, `part2/pending_benchmarks.md`, `scripts/pending_benchmarks.sh` |
| PTQ round-2 finalists (0c) | **done at ctx 512 (ctx 2048 skipped)** | Full-set golden fp32 + fake-quant evaluation on 16 EN + 32 UK windows (`windows_uk32.ids`); UK32 windows 0–7 reproduce UK8 golden | `part2/kv_final.md`, `part2/kv_final/{en16,uk32}/`, `results/reference/windows_uk32.ids` |
| Standalone build, JIT dump & Engine API (T34/T37) | **done** | Dropped MLIR-JIT executor from standalone runtime libs (`-DTENZO_STANDALONE_RUNTIME=ON` builds); `tenzo_create_engine` rejects unknown `kv_mode` and invalid `rope_theta`/`TENZO_GEMV`; `TENZO_DUMP_JIT_ASM` dumps all 8 audit objects | `part2/logs/standalone_build.log`, `part2/logs/engine_api.log`, `part2/logs/jit_dump_list.txt` |
| Calibrated ternary KV (round 3, T30) | **EN-only & UK-only done; EN+UK mix interrupted** | Layer-wise Adam calibration of clip multipliers $\alpha \in [0.5, 1.5]$ (`S3+s4r32`) on 8 disjoint EN and 8 UK windows; EN and UK runs evaluated on EN16 + UK32; mix run killed at layer 4 (`exit=143`) on host shutdown; Cayley rotation not attempted | `part2/kv_calib.md`, `part2/kv_calib.csv`, `part2/kv_calib_scatter.png`, `part2/calib/`, `tools/reference/kv_calib.py`, `scripts/run_kv_calib.sh` |
| `ternary_rot` KV mode in engine (block 4) | **correctness & quality done; speed queued** | AVX2 FWHT ($D=128$, seed-0 signs) + ternary g32 + 4 sinks / 32 recent INT8; FWHT round-trip rel L2 `8.9e-8`, 80-step cache attention rel L2 $\le 1.8 \times 10^{-6}$, G6 matches greedy 24-token prefix, KV 14,400 B/tok + 1.4 MB ring (`971` MB RSS at ctx 8192) | `part2/ternary_rot_report.md`, `part2/kv_bytes_per_token.md`, `part2/ppl_ternary_rot/`, `part2/ppl_ternary_rot_uk/`, `part2/g6/after_ternary_rot.txt`, `src/runtime/Hadamard.h`, `src/runtime/TenzoEngine.{h,cpp}` |
| Batched prefill (T32) | **correctness done; speed queued** | `prefill_tokens_batched` + `gemm_i2s_rows` + `noinline` `rms_norm_raw`; bit-identical (`max_abs=0`) vs sequential across 21 (7 prompts $\times$ 3 KV modes) pairs and tiny 2-layer tests | `part2/design_batched_prefill.md`, `part2/batched_prefill_report.md`, `part2/prefill/batched_vs_sequential.md`, `part2/logs/engine_api_full.log` |
| Engine RSS reduction (T31) | **done** | Progressive read-only `mmap` + `madvise(MADV_DONTNEED)`, per-layer I2_S conversion freeing TL1 copies, and 1-block stub for unused `PagedKVCache`; `int8_fused` ctx 2048 VmHWM `2761 -> 935` MB (ctx 8192: `3225 -> 1166` MB) | `part2/rss_report.md`, `part2/rss/rss_summary.md`, `part2/rss/rss_breakdown.md`, `part2/rss/ppl4_int8_fused.txt`, `scripts/analyze_smaps.py` |
| Second model language disparity (`Qwen2.5-1.5B`) | **done** | Patched eager attention (`mean/max |dlogp| = 0` check on 4+4 windows), evaluated 16 EN + 16 UK windows; Ukrainian degrades worse under INT8 (`1.041` vs `1.016`) and `S3+s4r32` (`2.774` vs `1.956`) despite lower UK base NLL (`2.00` vs `2.41`) | `part2/second_model_report.md`, `part2/second_model.md`, `part2/second_model.csv`, `part2/second_model_scatter.png`, `part2/second_model/patch_check.md`, `tools/reference/second_model_kv.py` |
| Paper tables & docs sync | **done (Part 2 report missing)** | Added automated table generator (`results/paper/tables.{md,tex}`), updated `README.md`, `CHANGELOG.md`, `PROJECT_STATUS.md`, `docs/ROADMAP.md`, and `decision_log.md` (D7–D17) | `scripts/build_paper_tables.py`, `results/paper/tables.{md,tex}`, `decision_log.md` |

### 2. Main numbers (verified from score files)

All PPL values and paired ratios below were independently recomputed from the raw score files (`PPL = exp(sum_nll / sum_tokens)` over matched windows) and verified within `< 0.05%` of the reported figures:

**PTQ finalists & calibrated `S3+s4r32` (BitNet-b1.58-2B-4T, golden fp32 + fake-quant, ctx 512):**
- Reference baselines: golden EN16 `19.066` (`19.0665` recomputed, `results/reference/golden/nll_golden_float32.txt`); golden UK32 `118.217` (`118.2172` recomputed, `part2/kv_final/uk32/fp32.txt`).
- `S13+s4r32` (K INT4 asym g32, V S10 ternary + 4 sinks / 32 recent INT8): EN16 ratio to golden `1.0104` (PPL `19.2647`); UK32 ratio to `uk32/fp32.txt` `1.0960` (PPL `129.5680`).
- `S3+s4r32` uncalibrated (UK32): ratio to `uk32/fp32.txt` `1.1512` (PPL `136.0942`; EN16 baseline `1.0709` from `2026-11-19_overnight`).
- `S3+s4r32` calibrated on EN (`alpha_en.json`): EN16 ratio `1.0399` (PPL `19.8277`); UK32 ratio `1.1261` (PPL `133.1243`).
- `S3+s4r32` calibrated on UK (`alpha_uk.json`): EN16 ratio `1.0488` (PPL `19.9965`); UK32 ratio `1.1166` (PPL `131.9969`).

**Engine `ternary_rot` mode (`models/bitnet-1.58b-v2`, ctx 512):**
- WikiText-2 EN16 (`part2/ppl_ternary_rot/ternary_rot.txt`): PPL `20.502` (`20.5021` recomputed), ratio `1.0753` to golden EN16 (`19.0665`). *(Note: raw engine/sim ratio is `1.0041` [1.0007, 1.0077], baseline-adjusted against `int8_fused` is `1.0002` [0.9957, 1.0046]; UK8 `165.634` vs golden `141.911` in `part2/ternary_rot_report.md`.)*

**Second model (`Qwen2.5-1.5B`, bf16 weights / fp32 compute, 16 EN + 16 UK windows):**
- `tenzo_int8` vs `fp`: EN ratio `1.016` (`1.0156` recomputed, PPL `11.3560` vs `11.1820`); UK ratio `1.041` (`1.0408` recomputed, PPL `7.7091` vs `7.4070`).
- `S3+s4r32` vs `fp`: EN ratio `1.956` (`1.9559` recomputed, PPL `21.8710` vs `11.1820`); UK ratio `2.774` (`2.7737` recomputed, PPL `20.5450` vs `7.4070`).

### 3. Not completed
- **Clean-machine speed measurements** (`scripts/pending_benchmarks.sh`): I2_S vs LUT GEMV A/B (`ab_gemv_pp{128,512}`), e2e v3 vs `bitnet.cpp` (`e2e_fair_v3`), batched vs sequential prefill A/B (`ab_prefill_pp{128,512}`), and KV-mode decode comparison (`decode_kv_modes`).
- **EN+UK mix calibration** (`alpha_mix.json`): interrupted by `SIGTERM` (`exit 143`) at layer 4 during host shutdown; only layers 0–3 were saved to `alpha_mix.json`, and both `eval_en16`/`eval_uk32` failed (`exit 1`) due to `systemd-run` rejecting unit creation during shutdown.
- **Cayley rotation optimization** (`decision_log.md` D10): omitted for time.
- **Context 2048 PTQ evaluations** (`part2/kv_final.md`): omitted for time.
- **Parity gate v2 (`B`) and BPE tokenizer (`F`)** (`part2/triage.md`): skipped (`B` lacked spec from truncated part-1 prompt; `F` lower priority).
- **Wiring calibrated `TENZO_KV_ALPHA` into engine default / eval**: `TENZO_KV_ALPHA` loading exists in `FusedKVCache::load_kv_alpha`, but `ternary_rot` PPL runs were executed without calibrated alphas (`part2/ternary_rot_report.md`).

### 4. Blockers
- **Host load / clean-machine condition:** browsers, Steam/gaming (`load` 4–13) persisted throughout Part 2, preventing clean speed benchmarks (`load1 < 1.0`, no browsers/steam).
- **Disk space & RAM ceiling:** `/` at ~99% full (~5.6–6.9 GB free) forced temporary `Qwen2.5-1.5B` download into `/tmp` (`tmpfs`, lost on reboot); quiet-mode `MemoryMax=3G` (4G for Qwen) slowed calibration/eval runs.
- **Missing parity gate v2 spec:** `max_abs <= 0.25` fails even between golden fp32 and HF bf16 (`0.53–1.63`), requiring an owner decision.
- **Host shutdown at `16:30:04`:** aborted mix calibration and prevented Part 2 report compilation on 2026-11-20.

### 5. What must NOT be claimed from these data
- **No speed claims** for I2_S GEMV, batched prefill, `ternary_rot` decode latency, or Tenzo vs `bitnet.cpp` v3 (zero clean-machine runs executed).
- **No EN+UK mix calibration result** (`alpha_mix.json` is incomplete through layer 3/4 and never evaluated).
- **No <= 2.5-bit lossless ternary KV claim**: uncalibrated `S3+s4r32` (2.90 b/elem @512) is `+7.1%` EN / `+15.1%` UK (32 win), and even calibrated `S3+s4r32` remains `+4.0%` EN / `+11.7–12.6%` UK; on `Qwen2.5-1.5B`, `S3+s4r32` still degrades PPL by `1.96x` EN / `2.77x` UK.
- **No per-character / per-byte cross-tokenizer claim** for language disparity, and no causal claim across non-overlapping base NLL ranges (`part2/second_model_report.md`, `results/paper/tables.md`).
- **No ctx 2048 or ctx 8192 PPL claim** for PTQ or `ternary_rot` (all PPL runs were at ctx 512).
- **Do not treat commit `e367633` by its message alone** (`decision_log.md` D14): it also staged initial unbuilt `ternary_rot`/batched-prefill/mmap code, later fixed and verified in `545d24a`, `e095a78`, `8e3fcf5`.

### 6. Commits (Part 2, `00fe9e3..8fe1947`)

```text
d91479e chore(bench): pending speed queue, in-binary GEMV A/B, e2e load column
fb68f68 fix(bench): do not report docker-client RSS for in-container llama-bench
1f16a28 results: UK32 eval windows and disjoint EN/UK calibration windows
38deddc docs: supersede invalid 2026-11-18 measurements, changelog for semantic fixes
1cb84e9 results: PTQ finalists on 16 EN + 32 UK windows
b62e8fa fix(results): correct S13+s4r32 bits at ctx 512 to 4.058 (earlier text said 4.271)
e57b75c build: drop the MLIR-JIT executor from the standalone runtime libraries
9a69dc8 feat(api): reject unknown kv_mode in tenzo_create_engine, with tests
e5db27e fix(test): TENZO_DUMP_JIT_ASM dumps after the lazy JIT compile
a96667d feat(ref): layer-wise calibrated ternary KV (clip multipliers)
d4c7b04 feat(paper): tables generated from CSVs
e367633 results: EN-calibrated ternary KV (interim) [also contains initial unbuilt engine changes; see D14]
02770bb docs: decision log D13 (pipeline pause)
84ff44b results: UK-calibrated ternary KV; calibration pipeline paused before the EN+UK mix
545d24a fix(runtime): token-by-token and batched prefill give bit-identical results
e095a78 test: ternary_rot KV and batched-prefill tests (FWHT, attention reference, 2-layer bit-identity)
8e3fcf5 perf(runtime): reduce peak RSS at load
64ab729 chore(bench): queue prefill A/B and KV-mode decode benchmarks for a clean machine
e1eec58 results: batched prefill equivalence (bit-identical on 21 prompt/mode pairs), decision log D14-D15
e87d0c1 results: ternary_rot quality, memory, speed
357b7f7 results: calibrated ternary KV and calibration-language study (EN-only, UK-only; EN+UK mix pending)
33f6281 docs: README measured table - ternary_rot, calibration, RSS, batched prefill
65c9ea8 results: KV quantization language disparity on a second model
3dc02f8 docs: changelog, status and roadmap for part 2 (ternary_rot, batched prefill, RSS, API validation)
8fe1947 results: interrupted EN+UK mix calibration (incomplete, killed at layer 4) and leftover logs
```
