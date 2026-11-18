# Overnight report 2026-11-19 (branch overnight/2026-11-19)

Status: **interim**. The session ran 12:46–14:03 (1 h 17 min of active time). It stopped, and the owner resumed it at 18:59 while still using the laptop. Free memory was then 3.7 GB, and no heavy step (≥ 1 GB) fits within the rule-0.3 margins, so blocks C (except PPL fp32), D and E have not been done yet. See D6/D9 in `decision_log.md`.

## 1. Blocks

| block | status | time | key result | files |
|---|---|---|---|---|
| start | done | 10 min | build ok, `tenzo-cli test` exit 0, G6 baseline | `meta.txt`, `g6/before_*`, `logs/start_tenzo_cli_test.log` |
| A golden fp32 | done | 25 min | golden bf16 == HF bf16 (eager) **bit-for-bit**; golden fp32 PPL 19.066 vs HF bf16 19.091 | `golden_validation.md`, `results/reference/golden/` |
| B gap cause | done (2 causes fixed) | 50 min | fp32-KV PPL 23.13 → 19.55 → **19.12** (golden 19.07) | `fix1_rope_theta.md`, `fix2_act_quant.md`, `divergence/`, `attribution/` |
| C re-measure | partial | – | only fp32-KV EN; G6-after for fix 2, the speed check, the other KV modes, UK, parity and gates are not done | `ppl_actquant/` |
| D ternary KV PTQ | not started (blocked: memory) | – | simulator written (`tools/reference/kv_ptq.py`), not run and not committed | – |
| E bitnet.cpp baseline | not started (blocked: memory) | – | – | – |
| F, G | skipped | – | – | – |

## 2. Main numbers

**Golden vs HF** (`golden_validation.md`). Golden bf16 matches HF bf16 eager with rel L2 = 0 on 60 tensors (L0 and L29, pos 0 and 5) and on the logits of p1, p3 and p5.

| WikiText-2, 16 × 512 | PPL | ratio vs golden fp32 [95% CI] |
|---|---|---|
| golden fp32 | 19.066 | 1 |
| HF bf16 (sdpa) | 19.091 | 1.0013 [0.998, 1.005] |
| Tenzo fp32-KV, 18.11 (before fixes) | 23.131 | 1.2132 [1.179, 1.255] |
| Tenzo fp32-KV + fix 1 (rope_theta) | 19.545 | 1.0251 [1.013, 1.038] |
| Tenzo fp32-KV + fix 1 + fix 2 (8-bit activations) | **19.124** | 1.0030 [1.0003, 1.0057] |
| same with diagnostic `TENZO_EXPERIMENT_FP32_LM=1` | 19.088 | 1.0011 [0.998, 1.005] |

On UK (Kobza, 8 windows), golden fp32 gives 141.9 and HF bf16 gives 142.2. Tenzo UK after the fixes has not been measured yet.

**Causes of the gap.** Both are semantic deviations from HF, and each was confirmed on tensors.

1. **RoPE theta.** Both KV caches hard-coded `theta_base = 10000.0f`, while config.json has `rope_theta = 500000`.
   - Signature: q_rope == q at pos 0, and the error grows with position. At pos 511, L0.q_rope had cos 0.27 against golden.
   - After the fix, the q_rope error equals the pre-RoPE q error at every position.
   - Golden attribution: emulating theta 10000 gives ×1.205 PPL.
2. **Activation quantization.** `build_lut` used ±31 levels (about 6 bits) instead of Qp=127.
   - The jump showed at `down` (input dimension 6912 with outliers): L0.down rel L2 0.16 → 0.05 after the fix.
   - Golden attribution: emulating 31 levels gives ×1.037.
3. **Residual +0.3%.** It comes from the INT8 embeddings and INT8 LM head (D8). The default is unchanged because this is a memory trade-off.

**KV modes v2, parity v2, PTQ, bitnet.cpp, e2e v2:** not done (see §6).

## 3. What changed in the conclusions compared with 2026-11-18

- "Tenzo is +21% worse than HF, and the cause is not the LM head": the cause was **two engine bugs** (RoPE theta, 6-bit activations). After the fixes, fp32-KV is within 0.3% of golden.
- "HF bf16 is a noisy reference": it is not. Golden fp32 and HF bf16 agree within 0.13%.
- Last night's PPL numbers for all KV modes (int8 23.22, tl1 59.42, popcount 174.98, UK 207/2186/9076) were measured on an engine with broken RoPE. **They are not valid for the current engine.** In particular, the tl1/popcount deterioration ratios have to be re-measured.
- The speed figures from 18.11 (decode 1.1–1.2× bitnet.cpp) were taken with the 6-bit kernel. Fix 2 doubles the shuffles in the GEMV, so the numbers may get worse. They have not been re-measured.

## 4. Negative and unexpected results

- The first golden version did not match HF. The cause was in the golden itself: HF quantizes activations in fp32 (D3). It was fixed in the golden; no threshold was changed.
- G6 after fix 1 has changed. fp32: "Paris. Paris is a city in the north of France…". popcount_fused now starts with "the capital of the United States…", which is worse than before (`g6/after_theta_*`).
- The 10000 constant is also present in `src/bindings/PybindModule.cpp:587` (the separate Python engine) and in `LinalgLowering.cpp` (MLIR `tenzo.rope`). Those paths are not fixed.

## 5. Decision log

See `decision_log.md` (D1–D9).

## 6. Blockers and what I need from you

- **Free memory (≥ 5–9 GB) and the laptop left alone.** The remaining work needs it:
  - G6 after fix 2 in 4 modes;
  - the decode-speed check (int8_fused, prompt 128 / gen 128, 3 repeats) — a regression is likely;
  - PPL of all KV modes on EN and UK;
  - parity v2 and the gates (`tenzo-cli test`/`audit`, Python suites);
  - block D (PTQ in golden, ~3.1 GB per run);
  - block E (bitnet.cpp rebuild).
- `models/bitnet-1.58b/config.json` lacks `rope_theta`. The engine warns and uses the default of 500000. A re-export with the updated `export_bitnet.py` would fix this permanently.

## 7. What these data do NOT support

- Any claim about speed after fix 2 (not measured).
- Any PPL for int8/tl1/popcount after the fixes (not measured). Last night's numbers are invalid.
- "Tenzo == HF". fp32-KV is +0.3% (CI excludes 1) with the default INT8 embeddings/head. Only the diagnostic FP32 head lands within CI.
- Parity per the verify_model_parity.py thresholds (not re-run).

## 8. Commits

```
5012150 fix(engine): quantize activations to 8 bit (Qp=127) as in HF ActQuant
c0db56f fix(engine): use the model's rope_theta (500000) instead of hard-coded 10000
f33b6b7 feat(debug): opt-in intermediate tensor dumps
a03a8cb results: golden reference validation
fb0bfee feat(ref): layer-major fp32 golden reference for BitNet-2B-4T
```

Weights sha256 is the same at start and end: `68c8f32b…00d1` (`weights_sha256_start.txt`, `weights_sha256_end.txt`).

---

# Update: second session (resumed 19:26, laptop free; cut off at ~23:15 by the agent's usage limit)

| step | status | key result | files |
|---|---|---|---|
| C: G6 after fix 2 | done | fp32 and int8_fused now generate the same text as HF ("Paris. Paris is a city that is known for its rich history…") | `g6/after_actquant_*` |
| C: gates | done | tenzo-cli test, audit, fixture (in Docker), compiler, references: all exit 0, also after the compiler fixes | `logs/gate_*`, `logs/gates_after_compilerfix/` |
| C: PPL, 4 KV modes, EN + UK | done | EN: fp32 19.124 (×1.0030 vs golden), int8 19.141 (×1.0039), tl1 54.9 (×2.88), popcount 207.7 (×10.9). UK: fp32 ×1.0007, int8 ×1.0020 (both within CI), tl1 ×8.87, popcount ×29.3 | `ppl_kv_v2.md`, `ppl_kv_v2_uk.md` |
| C: parity v2 | done | cosine > 0.999 on 6/7 prompts (fp32, int8), top-1 6/7 (fp32) and 7/7 (int8), greedy up to 32/32; max_abs 0.54–2.27 > 0.25, so **0/7 pass the unchanged gates** | `parity_v2.md` |
| 2: speed | done | same-time A/B (old vs new kernel): **decode ×0.81, prefill ×0.68**. Kernel not rewritten (>10% rule) | `speed_after_fix2.md`, `design_i2s_kernel.md` |
| 3: same bugs elsewhere | done | Pybind (theta + 31 levels), MLIR `tenzo.rope` (theta is now a required attribute), `bitlinear_tl1_pack` (31 levels, truncation) and `bitlinear_tl1` (63 levels, truncation) fixed; AuditJIT/refs encoded neither 31 nor 10000 | commits 5b100cc, 1fd1aa2 |
| 3: tests | done | `tenzo-cli hf-semantics` (compiler + engine via the C API, also in `test`/`all`) and `scripts/test_pybind_hf_semantics.py`. Every path matches HF (rel ≤ 4e-7), and the old semantics would fail | commit f02ba8b |
| 6: v2 config | done | `models/bitnet-1.58b-v2/config.json` (rope_theta, rms_norm_eps from HF; weights symlinked; `tools/reference/make_config_v2.py`); every v2 run uses it | – |
| D: PTQ | **partial** | simulator committed; replica runs (int8 / tl1 on 16 EN) and screening of S1–S4, S8, S9 on 4 windows were still running in the background when the session was cut off (`kv_ptq/`); results not analysed or committed | `kv_ptq/` (uncommitted) |
| E: bitnet.cpp | not started | – | – |

## What changed relative to the 2026-11-18 tables

- **ppl_kv / ppl_kv_uk.** All of them were measured with broken RoPE and 6-bit activations. Current numbers are in `ppl_kv_v2*.md`.
  - fp32/int8: from ×1.21 to ×1.003 (EN) and from ×1.47 to ×1.001 (UK).
  - tl1: ×3.12 → ×2.88 (EN), ×15.4 → ×8.9 (UK).
  - popcount got **worse** on EN (175 → 208) and better on UK (9076 → 4162).
- **parity_summary.** cosine went from 0.91–0.999 to > 0.999 on 6/7. The max_abs gate still fails on every prompt.
- **e2e_fair.** These numbers are for the old (6-bit) kernel. The new kernel is ~19% slower in decode (A/B). Tonight's absolute e2e numbers are unusable because of background load: the bitnet.cpp control point fell to 0.38×.
- **attn_ablation, spill_report, tokenization.** Not affected by the fixes. These are synthetic microbenchmarks and the tokenizer, which did not change.

## Still to do

Analyse block D (finalists on 16 EN + 8 UK, Pareto, conclusion), block E, the final weights hash, and stopping the inhibitor.
