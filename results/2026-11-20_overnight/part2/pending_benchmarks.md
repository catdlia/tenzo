# Pending speed measurements (queue)

Run `scripts/pending_benchmarks.sh` on a clean machine (load average < 1.0, no browsers/steam). It refuses otherwise.

| # | measurement | why | parameters | output |
|---|---|---|---|---|
| 1 | I2_S vs LUT GEMV, interleaved A/B in one binary (`TENZO_GEMV=i2s|lut`) | block A target: recover the ~19% decode loss of the two-table LUT kernel | int8_fused, prompt 128 and 512, gen 128, ABBA ×5, 4 threads cpus 0-3, models/bitnet-1.58b-v2 | `part2/bench/ab_gemv_pp{128,512}.csv` |
| 2 | e2e v3: Tenzo (i2s) vs the valid bitnet.cpp release build | the 2026-11-18 e2e used a broken bitnet.cpp and the wrong Tenzo kernel | prompts 16/128/512, gen 128, 5 reps; Tenzo fp32/int8_fused; bitnet.cpp f16/q8_0 KV | `part2/bench/e2e_fair_v3.csv` |
| 3 | batched vs sequential prefill (TTFT, prefill tok/s), same binary, interleaved ABBA | block 5 target: prefill is weight-bandwidth bound token by token; one weight unpack per chunk should help | int8_fused, prompt 128 and 512, chunk 32, 5 blocks, 4 threads | `part2/bench/ab_prefill_pp{128,512}.csv` |
| 4 | decode speed int8_fused vs tl1_fused vs ternary_rot | block 4: cost of the Hadamard rotation / ternary attention vs the other modes | prompt 128 and 512, gen 128, 3 reps | `part2/bench/decode_kv_modes.csv` |
