# M4 parity v2: Tenzo (after fixes 1+2) vs golden fp32 and HF bf16

First-step logits on 7 prompts, Tenzo driven with the reference token ids. Gates unchanged (scripts/verify_model_parity.py):
cosine > 0.999, max_abs <= 0.25, top-1 equal. greedy = matching prefix length of the 32-token greedy continuation.

| kv mode | prompt | ref | cosine | max_abs | top-1 | top-5 | KL(ref||tenzo) | greedy | gate |
|---|---|---|---|---|---|---|---|---|---|
| fp32 | p1_fact_en | golden | 0.99961 | 0.538 | True | 1.0 | 0.00044 | 32/32 | FAIL |
| fp32 | p1_fact_en | hf | 0.99927 | 0.716 | True | 1.0 | 0.00020 | 32/32 | FAIL |
| fp32 | p2_tech_en | golden | 0.99985 | 0.555 | True | 1.0 | 0.00102 | 32/32 | FAIL |
| fp32 | p2_tech_en | hf | 0.99982 | 0.536 | True | 1.0 | 0.00399 | 32/32 | FAIL |
| fp32 | p3_code | golden | 0.99985 | 0.760 | True | 0.8 | 0.00009 | 32/32 | FAIL |
| fp32 | p3_code | hf | 0.99988 | 0.658 | True | 1.0 | 0.00013 | 32/32 | FAIL |
| fp32 | p4_ukrainian | golden | 0.99982 | 2.271 | False | 1.0 | 0.07638 | 0/32 | FAIL |
| fp32 | p4_ukrainian | hf | 0.99992 | 1.459 | False | 1.0 | 0.03640 | 0/32 | FAIL |
| fp32 | p5_long_ctx | golden | 0.99765 | 0.661 | True | 1.0 | 0.00421 | 32/32 | FAIL |
| fp32 | p5_long_ctx | hf | 0.99811 | 0.757 | True | 1.0 | 0.01564 | 32/32 | FAIL |
| fp32 | p6_weird | golden | 0.99980 | 0.595 | True | 0.8 | 0.00468 | 18/32 | FAIL |
| fp32 | p6_weird | hf | 0.99979 | 0.681 | True | 0.8 | 0.00301 | 17/32 | FAIL |
| fp32 | p7_math_en | golden | 0.99992 | 0.592 | True | 1.0 | 0.00000 | 32/32 | FAIL |
| fp32 | p7_math_en | hf | 0.99989 | 0.771 | True | 1.0 | 0.00000 | 32/32 | FAIL |
| int8_fused | p1_fact_en | golden | 0.99959 | 0.623 | True | 1.0 | 0.00028 | 23/32 | FAIL |
| int8_fused | p1_fact_en | hf | 0.99921 | 0.614 | True | 1.0 | 0.00086 | 23/32 | FAIL |
| int8_fused | p2_tech_en | golden | 0.99969 | 0.728 | True | 1.0 | 0.00253 | 32/32 | FAIL |
| int8_fused | p2_tech_en | hf | 0.99965 | 0.759 | True | 1.0 | 0.00580 | 32/32 | FAIL |
| int8_fused | p3_code | golden | 0.99986 | 0.787 | True | 0.8 | 0.00020 | 32/32 | FAIL |
| int8_fused | p3_code | hf | 0.99987 | 0.753 | True | 1.0 | 0.00005 | 32/32 | FAIL |
| int8_fused | p4_ukrainian | golden | 0.99990 | 1.950 | True | 1.0 | 0.02767 | 14/32 | FAIL |
| int8_fused | p4_ukrainian | hf | 0.99996 | 0.987 | True | 1.0 | 0.01292 | 14/32 | FAIL |
| int8_fused | p5_long_ctx | golden | 0.99669 | 0.882 | True | 1.0 | 0.00301 | 32/32 | FAIL |
| int8_fused | p5_long_ctx | hf | 0.99885 | 0.652 | True | 1.0 | 0.00707 | 32/32 | FAIL |
| int8_fused | p6_weird | golden | 0.99974 | 0.692 | True | 0.8 | 0.00540 | 26/32 | FAIL |
| int8_fused | p6_weird | hf | 0.99982 | 0.495 | True | 0.8 | 0.00381 | 17/32 | FAIL |
| int8_fused | p7_math_en | golden | 0.99993 | 0.575 | True | 0.8 | 0.00000 | 7/32 | FAIL |
| int8_fused | p7_math_en | hf | 0.99990 | 0.720 | True | 0.8 | 0.00000 | 7/32 | FAIL |
| tl1_fused | p1_fact_en | golden | 0.88543 | 6.516 | True | 0.6 | 0.08216 | 4/32 | FAIL |
| tl1_fused | p1_fact_en | hf | 0.89169 | 6.461 | True | 0.6 | 0.08628 | 4/32 | FAIL |
| tl1_fused | p2_tech_en | golden | 0.97721 | 5.993 | True | 0.8 | 0.16641 | 3/32 | FAIL |
| tl1_fused | p2_tech_en | hf | 0.97704 | 6.013 | True | 0.8 | 0.17455 | 3/32 | FAIL |
| tl1_fused | p3_code | golden | 0.97172 | 9.159 | True | 0.6 | 0.03196 | 11/32 | FAIL |
| tl1_fused | p3_code | hf | 0.97130 | 9.166 | True | 0.8 | 0.03031 | 11/32 | FAIL |
| tl1_fused | p4_ukrainian | golden | 0.98993 | 18.114 | False | 0.2 | 3.22402 | 0/32 | FAIL |
| tl1_fused | p4_ukrainian | hf | 0.99009 | 18.724 | False | 0.2 | 3.25686 | 0/32 | FAIL |
| tl1_fused | p5_long_ctx | golden | -0.40482 | 13.984 | False | 0.8 | 1.11517 | 0/32 | FAIL |
| tl1_fused | p5_long_ctx | hf | -0.30361 | 13.477 | False | 0.8 | 0.97149 | 0/32 | FAIL |
| tl1_fused | p6_weird | golden | 0.98183 | 5.569 | True | 0.2 | 0.40356 | 1/32 | FAIL |
| tl1_fused | p6_weird | hf | 0.98233 | 5.515 | True | 0.2 | 0.40322 | 1/32 | FAIL |
| tl1_fused | p7_math_en | golden | 0.97621 | 7.965 | True | 0.8 | 0.00383 | 3/32 | FAIL |
| tl1_fused | p7_math_en | hf | 0.97495 | 8.346 | True | 0.8 | 0.00393 | 3/32 | FAIL |

| kv mode | ref | prompts | gate PASS | top-1 match | cosine gate |
|---|---|---|---|---|---|
| fp32 | golden | 7 | 0 | 6 | 6 |
| fp32 | hf | 7 | 0 | 6 | 6 |
| int8_fused | golden | 7 | 0 | 7 | 6 |
| int8_fused | hf | 7 | 0 | 7 | 6 |
| tl1_fused | golden | 7 | 0 | 5 | 0 |
| tl1_fused | hf | 7 | 0 | 5 | 0 |

## Notes

- Reference "golden" = tools/reference/golden_bitnet.py --dtype float32 (results/reference/golden/, logits are not committed;
  sha256 in results/reference/golden/SHA256SUMS). Golden fp32 greedy vs HF bf16 greedy: 32/32 on 5 prompts, 26/32 (p4)
  and 17/32 (p6) (`logs/golden_vs_hf_greedy.txt`), so the two references diverge on long greedy continuations too.
- **No prompt passes the unchanged gates.** cosine > 0.999 holds on 6/7 prompts for fp32 and int8_fused (was 0/7 on 2026-11-18,
  cos 0.91-0.999), top-1 matches 6/7 (fp32) and 7/7 (int8_fused), but max_abs is 0.54-2.27 against the 0.25 gate.
  The fp32-KV PPL residual (+0.3%) was attributed to the INT8 embedding table / LM head (decision log D8); max_abs on
  the full 128k-vocab logit vector is dominated by that quantization. Not re-run with TENZO_EXPERIMENT_FP32_LM here.
- p4_ukrainian fp32: top-1 differs from golden (max_abs 2.27); int8_fused matches top-1 on the same prompt. Both are
  near-ties in the reference (KL 0.03-0.08).
