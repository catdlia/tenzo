# Block G: spill / reload report for the fused kernels (static analysis only; no code changed)

## What was analysed
1. **JIT audit kernels** (`ternary_pack` = `fused_pack`/`standalone_pack`, packed attention = `explicit_attention_*`, `explicit_gqa_*`).
   The existing hook `TENZO_DUMP_JIT_ASM=1` only produced `fused_pack.o` (3 KB; `dumpToObjectFile` is called before the lazily-compiled
   symbols are materialised), so the kernels were re-lowered offline exactly as the JIT does: `TENZO_DUMP_JIT_IR=1` -> `*-llvm.mlir`
   -> `mlir-translate` -> `opt -O3 -mcpu=alderlake` -> `llc -O3 -mcpu=alderlake` (scripts/spill_asm.sh). This replicates, but is not
   byte-identical to, the JIT output. **Shapes are the audit shapes (T=8, D=32, Hq/Hkv=1/1 and 20/5), not model shapes.**
2. **Engine hand-written attention** (`FusedKVCache::forward_attention_raw`, OpenMP-outlined body `_omp_fn.0`, object linked into
   `tenzo-inference`, built `-O3 -mavx2 -mfma`), disassembled with `llvm-objdump -d`. This is the code that actually serves the
   `acc[16]` YMM accumulator V-loops.

A spill is a vector store to an `%rsp/%rbp` slot; a reload is a vector load from one (scripts/analyze_spills.py). Loops are regions
closed by a backward branch; "innermost" = no other backward-branch region inside. Counts are of static instructions in the loop body.

## Counts
| file | whole-function vec spills (st/ld) | innermost loop | instrs | vec instrs | FMA/madd | spill st | reload ld | spill+reload per FMA |
|---|---|---|---|---|---|---|---|---|
| explicit_attention_H1_KV1_T8_D32.s | 26/24 | loop 0 [362:426] | 62 | 57 | 4 | 1 | 4 | 1.25 |
| explicit_attention_H1_KV1_T8_D32.s | 26/24 | loop 1 [367:483] | 114 | 109 | 8 | 1 | 8 | 1.12 |
| explicit_gqa_H20_KV5_T8_D32.s | 30/27 | loop 0 [34:443] | 406 | 371 | 0 | 16 | 10 | - |
| explicit_gqa_H20_KV5_T8_D32.s | 30/27 | loop 1 [783:847] | 62 | 57 | 4 | 1 | 4 | 1.25 |
| explicit_gqa_H20_KV5_T8_D32.s | 30/27 | loop 2 [788:904] | 114 | 109 | 8 | 1 | 8 | 1.12 |
| fused_pack.s | 3/1 | loop 0 [28:96] | 68 | 56 | 32 | 0 | 0 | 0.00 |
| standalone_pack.s | 0/1 | (no loop found) | 283 | 276 | 0 | 0 | 1 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 0 [13:16] | 4 | 0 | 0 | 0 | 0 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 1 [180:198] | 19 | 16 | 8 | 0 | 0 | 0.00 |
| engine_forward_attention_raw.dis | 111/120 | loop 2 [308:359] | 52 | 41 | 0 | 8 | 1 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 3 [487:505] | 19 | 16 | 0 | 0 | 0 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 4 [586:632] | 47 | 42 | 32 | 4 | 4 | 0.25 |
| engine_forward_attention_raw.dis | 111/120 | loop 5 [672:697] | 26 | 0 | 0 | 0 | 3 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 6 [753:809] | 57 | 53 | 16 | 2 | 0 | 0.12 |
| engine_forward_attention_raw.dis | 111/120 | loop 7 [884:994] | 111 | 107 | 16 | 6 | 2 | 0.50 |
| engine_forward_attention_raw.dis | 111/120 | loop 8 [1095:1121] | 27 | 24 | 8 | 0 | 0 | 0.00 |
| engine_forward_attention_raw.dis | 111/120 | loop 9 [1129:1139] | 11 | 4 | 0 | 0 | 0 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 10 [1293:1311] | 19 | 16 | 0 | 0 | 0 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 11 [1688:1768] | 81 | 77 | 8 | 0 | 0 | 0.00 |
| engine_forward_attention_raw.dis | 111/120 | loop 12 [1833:1851] | 19 | 16 | 0 | 0 | 0 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 13 [1975:2017] | 43 | 40 | 0 | 0 | 0 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 14 [2106:2174] | 69 | 17 | 0 | 0 | 0 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 15 [2112:2178] | 67 | 16 | 0 | 0 | 0 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 16 [2171:2188] | 18 | 3 | 0 | 0 | 0 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 17 [2196:2258] | 63 | 17 | 0 | 0 | 0 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 18 [2202:2262] | 61 | 16 | 0 | 0 | 0 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 19 [2255:2320] | 66 | 9 | 0 | 0 | 0 | - |
| engine_forward_attention_raw.dis | 111/120 | loop 20 [2396:2487] | 92 | 11 | 0 | 0 | 4 | - |

## llvm-mca (alderlake model; hottest innermost loop = most FMA/madd)
```
== engine_forward_attention_raw.dis.loop4.s
Iterations:        200
Instructions:      9200
Total Cycles:      3226
IPC:               2.85
Block RThroughput: 16.0
== explicit_attention_H1_KV1_T8_D32.s.loop1.s
Iterations:        200
Instructions:      22200
Total Cycles:      7252
IPC:               3.06
Block RThroughput: 23.0
== explicit_gqa_H20_KV5_T8_D32.s.loop2.s
Iterations:        200
Instructions:      22200
Total Cycles:      7252
IPC:               3.06
Block RThroughput: 23.0
== fused_pack.s.loop0.s
Iterations:        200
Instructions:      13400
Total Cycles:      3215
IPC:               4.17
Block RThroughput: 16.0
```
(`llvm-mca` models the core only: no cache/memory effects, so "Block RThroughput" is a lower bound in cycles per loop iteration.)

## Reading
* `fused_pack` hot loop: 32 FMA/madd-class instructions, **0 spills, 0 reloads** -> no register pressure problem in the fused ternary pack at this shape.
* `explicit_attention`/`explicit_gqa` innermost loops: 1 spill store and 4-8 reloads per 4-8 FMA (~1.1-1.25 spill-or-reload instructions per FMA).
  These are loop-invariant reloads (the loop body re-reads a small set of vectors from the stack each iteration) rather than accumulator ping-pong,
  judged from instruction counts only (not inspected per register).
* Engine attention: the 16-accumulator V loops (16 FMA per body) show 2 stores / 0 reloads and 6 stores / 2 reloads per iteration (<= 0.5 per FMA),
  and the 32-FMA loop shows 4/4 (0.25 per FMA). 16 YMM accumulators exhaust the AVX2 register file, so some spilling is expected; the measured amount is small.
* Whole-function totals (111 stores / 120 reloads in the engine function) include prologue/epilogue and cold paths and say nothing about the hot loop.

## What this does NOT show
* No claim about wall-clock impact: static counts and a core-only model, no hardware counters.
* The audit kernels are toy shapes; the model-shape JIT kernels (D=128, long T) were not generated here.
* The loop recogniser is a backward-branch heuristic; nested/overlapping regions in the engine function (e.g. loops 14-20) are not individually interpreted.
