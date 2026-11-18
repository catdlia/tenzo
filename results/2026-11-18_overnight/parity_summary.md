# M4 parity: Tenzo vs real references (first-step logits, 7 prompts; greedy = matching prefix length of 32)

Gates (unchanged): cosine > 0.999, max_abs <= 0.25, top-1 equal. Reference "hf" = HF transformers bf16 (ground truth), "llama" = local bitnet.cpp i2_s build.

| kv mode | prompt | ref | cosine | max_abs | top-1 | top-5 | KL(ref||tenzo) | greedy | gate |
|---|---|---|---|---|---|---|---|---|---|
| int8_fused | p1_fact_en | hf | 0.9407 | 4.41 | True | 0.8 | 0.0051 | 6/32 | FAIL |
| int8_fused | p1_fact_en | llama | -0.7200 | 18.15 | False | 0.6 | 3.9663 | 0/32 | FAIL |
| int8_fused | p2_tech_en | hf | 0.9943 | 3.63 | True | 0.8 | 0.0892 | 10/32 | FAIL |
| int8_fused | p2_tech_en | llama | -0.6314 | 19.14 | False | 0.6 | 1.3013 | 0/32 | FAIL |
| int8_fused | p3_code | hf | 0.9984 | 3.05 | True | 1.0 | 0.0021 | 32/32 | FAIL |
| int8_fused | p3_code | llama | 0.9037 | 15.50 | False | 0.6 | 5.6569 | 0/32 | FAIL |
| int8_fused | p4_ukrainian | hf | 0.9991 | 4.49 | True | 0.6 | 0.2163 | 7/32 | FAIL |
| int8_fused | p4_ukrainian | llama | 0.9807 | 29.22 | False | 0.0 | 12.5021 | 0/32 | FAIL |
| int8_fused | p5_long_ctx | hf | 0.9136 | 4.26 | True | 0.8 | 0.4143 | 10/32 | FAIL |
| int8_fused | p5_long_ctx | llama | -0.2778 | 17.83 | False | 0.4 | 6.2278 | 0/32 | FAIL |
| int8_fused | p6_weird | hf | 0.9802 | 6.26 | True | 0.6 | 0.2717 | 3/32 | FAIL |
| int8_fused | p6_weird | llama | 0.3827 | 10.31 | False | 0.6 | 1.1952 | 0/32 | FAIL |
| int8_fused | p7_math_en | hf | 0.9983 | 2.74 | True | 0.8 | 0.0001 | 7/32 | FAIL |
| int8_fused | p7_math_en | llama | 0.4487 | 14.60 | True | 0.4 | 0.2093 | 1/32 | FAIL |
| fp32 | p1_fact_en | hf | 0.9620 | 3.51 | True | 1.0 | 0.0067 | 6/32 | FAIL |
| fp32 | p2_tech_en | hf | 0.9966 | 2.65 | False | 1.0 | 0.1104 | 0/32 | FAIL |
| fp32 | p3_code | hf | 0.9982 | 4.29 | True | 0.8 | 0.0003 | 32/32 | FAIL |
| fp32 | p4_ukrainian | hf | 0.9994 | 3.89 | False | 0.8 | 0.2002 | 0/32 | FAIL |
| fp32 | p5_long_ctx | hf | 0.9253 | 4.53 | True | 1.0 | 0.4737 | 9/32 | FAIL |
| fp32 | p6_weird | hf | 0.9927 | 3.83 | True | 0.6 | 0.1242 | 3/32 | FAIL |
| fp32 | p7_math_en | hf | 0.9984 | 3.32 | True | 0.8 | 0.0001 | 3/32 | FAIL |
| tl1_fused | p1_fact_en | hf | 0.9518 | 6.26 | True | 0.4 | 0.0518 | 4/32 | FAIL |
| tl1_fused | p2_tech_en | hf | 0.9690 | 7.66 | True | 0.6 | 0.3195 | 10/32 | FAIL |
| tl1_fused | p3_code | hf | 0.8939 | 12.99 | True | 0.6 | 0.0399 | 11/32 | FAIL |
| tl1_fused | p4_ukrainian | hf | 0.9848 | 22.64 | False | 0.2 | 3.5869 | 0/32 | FAIL |
| tl1_fused | p5_long_ctx | hf | -0.0697 | 10.27 | True | 0.6 | 0.2641 | 10/32 | FAIL |
| tl1_fused | p6_weird | hf | 0.9761 | 6.61 | True | 0.8 | 0.4853 | 1/32 | FAIL |
| tl1_fused | p7_math_en | hf | 0.9774 | 8.16 | True | 0.8 | 0.0020 | 3/32 | FAIL |

Negative control (truncated weights must exit non-zero): {'name': 'truncated_weights', 'rc': 1, 'pass': True}

## Tokenisation (Tenzo greedy-longest-match tokenizer vs references; identical = same ids incl. BOS)

| prompt | tenzo len | vs HF | vs llama.cpp (native) |
|---|---|---|---|
| p1_fact_en | 6 | identical | identical |
| p2_tech_en | 18 | identical | differs from idx 14 (ref 19 tokens) |
| p3_code | 18 | differs from idx 5 (ref 18 tokens) | differs from idx 3 (ref 22 tokens) |
| p4_ukrainian | 10 | differs from idx 1 (ref 18 tokens) | differs from idx 1 (ref 18 tokens) |
| p5_long_ctx | 340 | differs from idx 23 (ref 343 tokens) | differs from idx 23 (ref 345 tokens) |
| p6_weird | 51 | differs from idx 43 (ref 50 tokens) | differs from idx 2 (ref 59 tokens) |
| p7_math_en | 19 | identical | differs from idx 7 (ref 23 tokens) |
