# bitnet.cpp reference rebuild (block B)

Tree: microsoft/BitNet@c17d1c5 (2025-04-23, first commit after the 2B-4T release fix 488dc1e), submodule llama.cpp 5eb47b72106e3b35f10e8befa616a9241242b226 from Eddie-Wang1120/llama.cpp (merge-dev). Clone in ~/tenzo-refs/BitNet (BitNet/ in the repo untouched).

Build (inside the dev container, clang 21.1.8): `python3 utils/codegen_tl2.py --model bitnet_b1_58-3B --BM 160,320,320 --BK 96,96,96 --bm 32,32,32` (setup_env.py recipe for BitNet-b1.58-2B-4T on x86_64), `cmake -B build -DBITNET_X86_TL2=ON -DCMAKE_BUILD_TYPE=Release`, `cmake --build build -j2`. Model: the existing BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf (nothing downloaded).

Greedy "The capital of France is" (24 tokens, temp 0):

```
The capital of France is Paris. Paris is a city that is known for its rich history, culture, and architecture. It is also a major
```
(identical to the HF bf16 greedy and to Tenzo fp32/int8 after the 2026-11-19 fixes)

NLL with tools/reference/bitnet_ref_dump.cpp built against this tree (-DTENZO_LEGACY_LLAMA_API), same windows/protocol:

| file | windows | PPL | 95% CI | ratio vs nll_golden_float32.txt | 95% CI of ratio |
|---|---|---|---|---|---|
| nll_golden_float32.txt | 16 | 19.066 | [16.163, 22.448] | 1.0000 | [1.0000, 1.0000] |
| nll_bitnet_rel_en.txt | 16 | 19.059 | [16.163, 22.427] | 0.9996 | [0.9965, 1.0027] |
| nll_llama_i2s.txt | 16 | 102.051 | [86.759, 117.960] | 5.3524 | [4.8195, 5.9912] |
| int8_fused.txt | 16 | 19.141 | [16.231, 22.510] | 1.0039 | [1.0009, 1.0070] |

| file | windows | PPL | 95% CI | ratio vs nll_golden_float32_uk.txt | 95% CI of ratio |
|---|---|---|---|---|---|
| nll_golden_float32_uk.txt | 8 | 141.911 | [111.691, 176.099] | 1.0000 | [1.0000, 1.0000] |
| nll_bitnet_rel_uk.txt | 8 | 141.801 | [111.457, 175.769] | 0.9992 | [0.9940, 1.0043] |
| int8_fused.txt | 8 | 142.196 | [111.974, 176.319] | 1.0020 | [0.9959, 1.0084] |

**Valid baseline.** PPL within CI of golden fp32 on EN and UK. The broken local tree (BitNet/, upstream HEAD + isHuangXin/llama.cpp submodule + 1-line local patch, host clang 23) gave PPL 102 (nll_llama_i2s.txt row). Its cause was not isolated further: the release-era tree with container clang works, which answers the question the baseline was needed for.
