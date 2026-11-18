# Overnight report: 2026-11-18 (branch `overnight/2026-11-18`)

Machine: i3-1215U laptop on AC, governor `powersave` throughout (sudo needs a password, so it was never changed; every CSV records it).
weights.bin sha256 is identical before and after: `68c8f32b…79400d1` (`weights_before.sha256`, `weights_after.sha256`).

**Clock note (read first).** Work started at 02:05. At 02:36 **I caused an OOM**: I started a Docker build while the HF reference process
(5 GB) was running with about 2.5 GB free. The kernel killed the HF process, `claude-desktop`, the user's Hyprland session and the inhibitor.
The run then sat dead until the user typed "continue" at 08:33, and ran from 08:33 to about 10:50. Active work time is about 2 h 50 min.
I treated the 08:33 restart as a continuation (decision log D8, D9). No data was lost. After the incident, HF jobs ran under a
`systemd-run MemoryMax=7G` cap and builds used `-j2`.

## 1. Blocks

| Block | Status | Active time (approx.) | Key result | Files |
|---|---|---|---|---|
| Start | done | 15 min | rebuild OK; `tenzo-cli test` exit 0 | meta.txt, startup_*.log |
| A T01 logs | done | 3 min | `!results/**/*.log` exception, logs committed | .gitignore |
| B reference + M4 parity | done (gates fail) | 75 min | HF transformers = ground truth; the local bitnet.cpp build is broken; Tenzo top-1 = HF on 7/7 prompts, but cos/max_abs gates fail | parity_summary.md, parity/, provenance.md, tokenization.json |
| C KV perplexity | done | 50 min | int8 KV is free; tl1 KV gives 2.6x PPL (EN) and 10x (UK); Tenzo fp32-KV is +21% PPL over HF | ppl_kv.md/.csv, ppl_kv_uk.*, ppl_experiment/ |
| D attention ablation | done | 15 min | zero-copy ~3x; ternary packed KV is **slower** than FP32-direct and INT8 | attn_ablation.md/.csv/.png (+ run2) |
| E fair e2e | done | 35 min (mostly waiting) | decode 1.1-1.2x vs bitnet.cpp; prefill ~0.5x; prompt+gen at 512: 0.75x; RSS 2.2x | e2e_fair.md/.csv |
| F T01 tails | done | 25 min | TL1 test fixed per op contract; fixture regressions now strict; tenzo-diag SKIP/FAIL semantics | logs/F_* |
| G spills | done (report only) | 15 min | fused_pack hot loop 0 spills; engine V-loops ≤0.5 spill/reload per FMA | spill_report.md, spill/ |
| H README honesty | done | 10 min | historical 20.32 tok/s / 1.65x / "Production" marked unverified; "Measured (2026-11-18)" section added | README.md, PROJECT_STATUS.md, DOCUMENTATION.md, docs/ROADMAP.md |
| Final gates | done | 5 min | `tenzo-cli test`, `tenzo-cli audit`, test_fixture_regressions, test_compiler_regressions, verify_references --all: all exit 0 | final_*.log |

## 2. Key numbers (each with its source)

**Reference validity** (ppl_kv.md, parity_summary.md)
- WikiText-2 PPL (16 windows x 511 tokens): HF bf16 **19.09**; local bitnet.cpp i2_s **102.05**.
- bitnet.cpp first-step logits vs HF: cosine −0.46 (p1) / −0.57 (p2). It answers "The capital of France is" with " a small city…".
- The GGUF holds the **same ternary weights as HF, bit for bit** (provenance.md). The fault is therefore in that fork's graph or kernels, not in the weights.

**M4 parity, Tenzo vs HF** (parity_summary.md, gates unchanged: cos > 0.999, max_abs ≤ 0.25, top-1)
- int8_fused KV: top-1 matches on 7/7 prompts; cosine 0.914–0.999; max_abs 2.74–6.26; KL 0.0001–0.41; greedy prefix 3–32 of 32. **0/7 pass.**
- fp32 KV: similar (max_abs 2.6–4.5), and top-1 differs on 2/7. The gap therefore does not come from KV quantisation.
- Negative control (truncated weights must exit non-zero): PASS.

**KV-mode quality** (ppl_kv.md; per-window bootstrap CIs are in the file)

| | WikiText-2 PPL | Ukrainian (Kobza, 8 windows) PPL | KV bytes/token | RSS at ctx 8192 |
|---|---|---|---|---|
| HF bf16 | 19.09 | 142.2 | - | - |
| Tenzo fp32 | 23.13 | 209.4 | 153,600 | 2.41 GB |
| Tenzo int8_fused | 23.22 (Δ +0.09, CI [−0.09, +0.30]) | 206.8 | 39,600 | 1.49 GB |
| Tenzo tl1_fused | 59.42 | 2186 | 10,800 | 1.26 GB |
| Tenzo popcount_fused | 174.98 | 9076 | 10,800 | 1.26 GB |

- Diagnostic (ppl_experiment/README.md): an FP32 LM head and embeddings give 23.26. The INT8 head is **not** the cause of the +21% gap to HF.

**Attention ablation** (attn_ablation.md; 1 thread, synthetic Gaussian KV)
- b/a (zero-copy GQA): **0.26–0.38**.
- d/b (ternary vs FP32 direct): **1.9–4.8** (slower).
- d/c (ternary vs INT8): **4.4–6.3** (slower).
- c is the fastest variant everywhere.
- Ternary-KV output vs FP32: rel L2 ≈ 0.69, cosine 0.77–0.79. INT8: rel L2 ≈ 0.009.
- Peak RSS at 32K, 32Q/8KV: a 293 MB vs d 54 MB.
- A repeat run agrees: median ratio 0.999, range 0.80–1.27.

**E2E** (e2e_fair.md; 4 threads, cpus 0-3, 5 reps, medians)
- prompt+gen tok/s at prompt 16 / 128 / 512: Tenzo int8 24.5 / 28.1 / 31.1; bitnet.cpp f16 20.6 / 29.1 / 41.5.
- Decode: Tenzo 21–24 tok/s vs bitnet.cpp 19 tok/s.
- Prefill: Tenzo ~35 tok/s (token by token) vs bitnet.cpp 54–70 tok/s.
- Peak RSS: Tenzo ~2.7 GB vs bitnet.cpp ~1.2 GB.

## 3. Negative and unexpected results
1. **The local bitnet.cpp is not a valid reference** (PPL 102). The old script's "expected first token 264 (' a')" came from that broken output.
2. **Tenzo is +21% PPL over HF on English and +47% on Ukrainian, even with fp32 KV.** INT8 LM head is ruled out. The cause is somewhere in the transformer body, or partly the bf16 reference itself.
3. **Ternary KV badly damages quality.** tl1_fused gives 2.6x (EN) and 10x (UK) PPL; popcount_fused 7.6x / 43x. G6 shows popcount degenerating into repetition.
4. **Ternary packed attention is slower** than FP32-direct and INT8 in the micro-benchmark. Its only win is memory.
5. With 4 threads (2 P-cores with HT), variant a is *slower* than with 1 thread at 4K–8K.
6. Tenzo's tokenizer (greedy longest match) differs from HF on 4/7 prompts. It mangles Ukrainian (10 vs 18 tokens) and code. Its decode prints raw byte-level BPE ("Ãİle-de-France").
7. Engine bug (now fixed in the CLI): an unknown `--kv-quant` silently ran the fp32 path.
8. tenzo-diag printed `[FAIL]` and then "ALL SYSTEM CHECKS PASSED"; skipped checks were shown as `[PASS]` (fixed).
9. The TL1 negative fixture tests passed vacuously, because MLIR compile failed for every case (fixed).
10. My own mistake: I briefly concluded that `restore_checkpoint.py` permuted the weights. The cause was a bug in my check (`stack(axis=2)`). Rebuilding with the "fixed" decoder gave the same SHA, so the claim was withdrawn before it was reported anywhere.

## 4. Decision log
Full text is in `decision_log.md` (D1–D9). In short:
- D1: branch cut from `recovery`.
- D2: Debug build dir, targets built with `-O3`.
- D3: governor left at powersave.
- D4: the libllama dumper is built and run on the host (BitNet was built against host glibc).
- D5: HF transformers is the ground truth.
- D6: `*.f32` dumps are gitignored.
- D7: the HF reference is bf16 (two fp32 attempts failed: OOM, and `.float()` is unsupported on quantized models).
- D8: the OOM incident.
- D9: active-time accounting.

Also:
- Bootstrap: 10,000 draws, seed 0.
- Ukrainian set: 8 windows from the first 39 Kobza rows via datasets-server, because the `datasets` library is broken locally.
- PPL runs used ctx 512.
- The tenzo-diag success banner now requires zero skips.
- The `TENZO_EXPERIMENT_FP32_LM` env switch was added, opt-in only. G6 output is byte-identical before and after every engine change, in all 4 KV modes.

## 5. Blockers and what the owner needs to do
- Decide what to do about the local BitNet tree. It is a modified fork (isHuangXin llama.cpp, plus local commit 0b6459f, plus a 1-line diff). Rebuilding upstream Microsoft BitNet would give a second trustworthy reference and an honest speed baseline.
- The +21% PPL gap to HF needs a layer-by-layer hidden-state comparison against HF. It was not done (2-approach limit).
- An fp32 HF reference needs more than ~7 GB of free RAM. bf16 noise in the reference is unquantified.
- Governor `performance` needs sudo. All timings are under powersave.
- `TENZO_DUMP_JIT_ASM` dumps objects before lazy JIT compilation, so only `fused_pack.o` appears. The spill analysis therefore used offline llc.

## 6. What must NOT be claimed from these data
- No speedup over a *correct* bitnet.cpp. The comparison build computes a different (broken) model.
- Not "1.65x faster" or "20.32 tok/s". The like-for-like decode advantage measured here is 1.1–1.2x, and prefill and long-prompt throughput are worse.
- Not "M4 parity achieved". Gates fail; only top-1 agreement holds.
- Not "ternary KV is lossless or cheap". It costs 2.6x–10x PPL, and its attention kernel is slower.
- No memory-reduction claim for the whole process: Tenzo peaks at ~2.2x bitnet.cpp's RSS.
- Nothing about other CPUs, about formats other than BitNet TL1, or about performance under the `performance` governor.
- The spill counts are static, from toy-shape kernels. They say nothing about runtime impact.

## 7. Commits (`git log recovery..overnight/2026-11-18`)
- fe63aa8 docs: mark unverified claims, link measured results
- 82ec6e7 results: fair e2e comparison vs bitnet.cpp (laptop)
- fac6109 results: attention ablation laptop (4 variants, 2 geometries, 5 contexts, 1 and 4 threads, two repeat runs)
- c132d30 results: Ukrainian slice PPL, FP32-LM diagnostic, combined ppl_kv.md, parity vs both references
- ead6af6 results: KV-mode perplexity (WikiText-2, 16x512) with HF and bitnet.cpp references
- 8568e8f results: M4 parity run vs HF + weights provenance
- 15e82be fix(diag): unavailable backends are SKIP, real failures fail the run
- f4f296f results: F1 logs (TL1 fixture, test, audit)
- 3ad4430 test(fixture): negative TL1 cases must fail via fixture validation
- dede560 fix(test): TL1 fixture shape follows the bitlinear_tl1 contract [N/64, K/2, 32]
- 675a94f results: spill/reload static analysis of fused kernels (report only)
- 04aebf3 feat(bench): four-variant attention ablation with RSS, pack timer and output error
- 7cc9202 fix(parity): replace stub reference in verify_model_parity
- 0ff2b94 feat(parity): real reference dumpers (HF transformers + libllama) and fixed eval windows
- 1ee5212 feat(eval): teacher-forced scoring, logits dump and fair-bench modes
- 0e8c258 chore(results): track T01 logs
- (this report) results: overnight report
