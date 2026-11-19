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
