# Overnight report 2026-11-19 (branch overnight/2026-11-19)

The night ran as two sessions.
- **Session 1:** 12:46–14:03. It was interrupted (D6), and on resume the owner was at the laptop, so the memory rule blocked further work.
- **Session 2:** 19:26–00:40, with a new 6 h budget after the owner said the laptop was free.

Weights `models/bitnet-1.58b/weights.bin` sha256 `68c8f32b…00d1`, identical at start and end (`weights_sha256_start.txt`, `weights_sha256_end.txt`). Both sleep inhibitors were stopped at the end.

## 1. Blocks

| block | status | key result | files |
|---|---|---|---|
| start | done | build ok, `tenzo-cli test` exit 0, G6 baseline | `meta.txt`, `g6/before_*` |
| A golden fp32 | done | golden bf16 == HF bf16 (eager) bit-for-bit; golden fp32 PPL 19.066 vs HF bf16 19.091 (EN), 141.9 vs 142.2 (UK) | `golden_validation.md`, `tools/reference/golden_bitnet.py` |
| B gap cause | done (2 causes) | fp32-KV PPL 23.13 → 19.55 (rope_theta) → **19.12** (8-bit activations); golden 19.07 | `fix1_rope_theta.md`, `fix2_act_quant.md`, `divergence/`, `attribution/` |
| C re-measure | done | KV modes v2 EN+UK, parity v2, G6 after, gates all exit 0 | `ppl_kv_v2*.md`, `parity_v2.md`, `g6/`, `logs/gate*` |
| speed after fix 2 | done | same-time A/B: decode ×0.81, prefill ×0.68 (old 6-bit kernel = 1). Kernel not rewritten (>10% rule), design note written | `speed_after_fix2.md`, `ab_kernel_speed.csv`, `design_i2s_kernel.md` |
| same bugs in other paths | done | Pybind (theta 10000, ±31), MLIR `tenzo.rope` (10000), `bitlinear_tl1_pack` (31 levels, truncation), `bitlinear_tl1` (63 levels, truncation) fixed; AuditJIT / reference scripts encoded neither | commits 5b100cc, 1fd1aa2 |
| tests | done | `tenzo-cli hf-semantics` (in `test`/`all`) + `scripts/test_pybind_hf_semantics.py`: every path matches the HF reference (rel ≤ 4e-7) and the old semantics would fail | `logs/test_hf_semantics.log` |
| D ternary KV PTQ | done | no scheme with ΔPPL ≤ 5% at ≤ 2.5 bits; best: +7.1% EN / +16.1% UK at 2.9 b/elem | `kv_ptq.md`, `kv_ptq.csv`, `kv_ptq.png` |
| v2 model config | done | `models/bitnet-1.58b-v2/` (config.json with rope_theta from HF, weights symlinked) | `tools/reference/make_config_v2.py` |
| E bitnet.cpp baseline | **skipped** (time) | – | – |
| F tokenizer, G small items | skipped | – | – |

## 2. Main numbers

**Golden vs HF** (`golden_validation.md`)
- Golden bf16 vs HF bf16 eager: rel L2 = 0 on 60 hooked tensors and on the logits of 3 prompts.
- Golden fp32 vs HF bf16 PPL ratio: 1.0013 [0.998, 1.005] (EN), 1.0023 (UK).
- Golden fp32 greedy vs HF greedy: 32/32 on 5 of 7 prompts.

**Causes of the gap.** Both were found by layer-by-layer dumps against the golden, then attributed by emulation inside the golden (8 EN windows).

| cause | emulated in golden | PPL fp32-KV EN before → after fix |
|---|---|---|
| RoPE base 10000 instead of config 500000 (both KV caches) | ×1.205 | 23.131 → 19.545 |
| activations ±31 levels (~6 bit) instead of HF Qp=127 | ×1.037 | 19.545 → 19.124 |
| residual: INT8 embeddings / LM head (design choice, not changed) | – | diagnostic FP32 head: 19.088, ratio CI includes 1 |

**KV modes v2** (`ppl_kv_v2.md`, `ppl_kv_v2_uk.md`). Ratio vs golden fp32, with the 2026-11-18 value in parentheses:

| kv mode | EN 16×512 PPL | EN ratio | UK 8×512 PPL | UK ratio |
|---|---|---|---|---|
| fp32 | 19.124 | 1.0030 (1.213) | 142.0 | 1.0007 (1.475) |
| int8_fused | 19.141 | 1.0039 (1.218) | 142.2 | 1.0020 (1.457) |
| tl1_fused | 54.94 | 2.88 (3.12) | 1258 | 8.87 (15.4) |
| popcount_fused | 207.7 | 10.9 (9.18) | 4162 | 29.3 (64.0) |

**Parity v2** (`parity_v2.md`, gates unchanged: cos > 0.999, max_abs ≤ 0.25, top-1)
- fp32/int8 vs golden fp32: cos > 0.999 on 6/7 prompts (it was 0/7), top-1 6/7 / 7/7, greedy up to 32/32.
- max_abs is 0.54–2.27, so **0/7 pass**.
- tl1: cos gate 0/7.

**Speed** (`speed_after_fix2.md`)
- Interleaved A/B (ABBA ×5, int8_fused, pp 128 / tg 128): decode ×0.807 (range 0.54–1.06), prefill ×0.684.
- Tonight's day-to-day e2e numbers are dominated by background load (bitnet.cpp control point decode ×0.38), so they are only reported with that caveat.

**Ternary KV PTQ** (`kv_ptq.md`). Screening of 12 configs on 4 EN windows, then 2 finalists on 16 EN + 8 UK with CI.

| scheme | bits @512 (asym.) | EN ΔPPL [CI] | UK ΔPPL |
|---|---|---|---|
| engine tl1 replica (TWN 0.5, per token) | 2.125 | +194% | – |
| S3: ternary g32 + Hadamard | 2.5 | +23% (4 win) | – |
| S3 + 4 sinks + 32 recent INT8 | 2.9 (2.5) | +7.1% [5.5, 8.8] | +16.1% |
| K INT8 + V S3 | 5.3 | +5.6% [3.7, 7.6] | +36.5% |
| comparators: 2-bit g32+rot / INT4 g32 | 3.0 / 5.0 | +12.9% / −0.4% (4 win) | – |

Conclusion:
- No scheme reaches ≤ 5% at ≤ 2.5 bits.
- What helped: the Hadamard rotation (+44% → +23%) and INT8 sink/recent windows.
- KIVI-style per-channel K did not beat rotation.
- K is more sensitive than V.
- Ukrainian degrades 2–6× more than English.
- In the engine the rotation would need an explicit FWHT per head (it cannot be folded into ternary o_proj).

**bitnet.cpp baseline / e2e v2:** not done (block E skipped).

## 3. What changed in the conclusions relative to 2026-11-18 (every table from that night)

| 18.11 table | then | now |
|---|---|---|
| `ppl_kv.md` / `ppl_kv_uk.md` | Tenzo fp32 +21% EN / +47% UK vs HF; int8 ≈ fp32; tl1 ×2.6 / ×10 | measured on an engine with broken RoPE and 6-bit activations, **invalid**. Now fp32/int8 within 0.4% EN and within CI UK; tl1 ×2.88 / ×8.9; popcount ×10.9 / ×29 |
| `parity_summary.md` | cos 0.91–0.999, 0/7 | cos > 0.999 on 6/7, still 0/7 because of max_abs |
| "INT8 LM head is not the cause" | true for the big gap | the INT8 embedding/LM head explains the remaining 0.3% |
| "HF bf16 is a noisy reference" (hypothesis) | – | rejected: golden fp32 ≈ HF bf16 within 0.13% |
| `e2e_fair.md` (decode 1.1–1.2× bitnet.cpp) | old 6-bit kernel | the correct 8-bit kernel is ~19% slower in decode; the old comparison favoured Tenzo by an incorrect kernel. Not re-measured against bitnet.cpp under clean conditions |
| `attn_ablation.md` | synthetic attention microbenchmark | unaffected (no RoPE / activation quant inside) |
| `spill_report.md`, tokenization | – | unaffected |
| bitnet.cpp PPL 102 | broken local build | still unresolved (block E skipped) |

## 4. Negative and unexpected results

- The first golden did not match HF: HF quantizes activations in fp32 (D3). The fix went into the golden, not into a threshold.
- popcount_fused EN got **worse** after the fixes (175 → 208). popcount G6 also degraded ("the capital of the United States…").
- The correct activation semantics cost ~19% decode speed with the current LUT design.
- No ternary-KV PTQ scheme is usable at ≤ 2.5 bits. Ukrainian suffers far more.
- The parity gate max_abs ≤ 0.25 is not reachable with INT8 embeddings/LM head.
- Compiler TL1 lowerings truncated activations (FPToSI) and used 31 or 63 levels: a third and fourth instance of the same bug class. No test covered them before tonight.
- The standalone runtime build (`-DTENZO_STANDALONE_RUNTIME=ON`) fails: ExecutionContext.cpp includes MLIR headers. This is a pre-existing bug and was not fixed.
- Golden greedy is slow (~10 s/step even with the int8 weight cache).

## 5. Decision log

`decision_log.md` (D1–D12). Additional decisions in session 2:
- `test_fixture_regressions.py` is Docker-only (D12).
- Speed judged by a same-time A/B instead of the cross-day comparison.
- `fx_to_mlir.py` falls back to theta 10000 with a warning only for the in-repo Llama module that hard-codes it.
- Block D finalists limited to 2 for time.
- Block E skipped.

## 6. Blockers / what the owner needs to decide

- **Kernel.** Implement the I2_S-style maddubs kernel (`design_i2s_kernel.md`) to recover the ~19% decode speed while keeping 8-bit activations.
- **Embeddings/LM head.** Keep INT8 (+0.3% PPL, parity max_abs gate fails) or offer FP16/FP32 as an option.
- **Block E.** Clean rebuild of microsoft/BitNet at the 2B-4T release commit to get a valid speed baseline.
- **Background load.** Speed measurements need the laptop without browser/neko load.
- **Tokenizer (block F).** Still greedy longest match, not BPE.

## 7. What these data do NOT support

- "Tenzo == HF": fp32-KV is +0.3% EN (CI excludes 1). The parity gates fail on max_abs.
- Any absolute speed claim from tonight, or any Tenzo-vs-bitnet.cpp speed claim after the fixes.
- "Ternary KV cache at ~2 bits is viable without retraining": every tested scheme ≥ +7% EN / +16% UK.
- PTQ numbers for contexts other than 512 (ctx 2048 not run). Screening numbers come from only 4 windows.
- Last night's KV-mode PPL tables (invalid) and last night's e2e speed advantage (wrong kernel).

## 8. Commits (aad222d..HEAD)

```
09d2458 results: ternary KV PTQ study
42e843c results: KV PTQ simulator validation and screening (12 configs)
5a9f526 results: report update for the second session (C, speed, other paths, tests; D partial)
c03ee42 feat(ref): KV fake-quant simulator
1702c8a results: speed after the 8-bit activation kernel (A/B) and I2_S design note
9da05b8 results: parity vs golden fp32
f02ba8b test: HF-semantics regressions for rope_theta and 8-bit ActQuant
1fd1aa2 fix(compiler): tenzo.rope theta attribute and HF ActQuant in TL1 lowerings
5b100cc fix(bindings): rope_theta parameter and 8-bit ActQuant in the Python runtime
55dff79 results: KV-mode perplexity after engine fix
c08843c results: interim overnight report (blocks A-B done, C-E blocked on memory)
5012150 fix(engine): quantize activations to 8 bit (Qp=127) as in HF ActQuant
c0db56f fix(engine): use the model's rope_theta (500000) instead of hard-coded 10000
f33b6b7 feat(debug): opt-in intermediate tensor dumps
a03a8cb results: golden reference validation
fb0bfee feat(ref): layer-major fp32 golden reference for BitNet-2B-4T
```
(plus the commit containing this report)
