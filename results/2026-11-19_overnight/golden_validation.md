# Golden reference validation (block A)

Golden: `tools/reference/golden_bitnet.py` (layer-major, written from HF transformers 4.57.6 code + config.json; engine code not consulted).

## A.6a golden --dtype bfloat16 vs HF transformers bf16

HF run: `tools/reference/hf_validate_golden.py --attn eager` (forward hooks), prompts p1, p3, p5; tensors at layers 0 and 29, positions 0 and 5. Log: `logs/golden_bf16_vs_hf_eager.txt`.

Result: **bit-identical** (rel L2 = 0, max_abs = 0, cos = 1.000000) on all 60 hooked tensors and on first-step logits of all 3 prompts. This required one golden fix (decision D3: HF ActQuant quantizes in fp32).

```
logits p1_fact_en.logits.f32 cos 1.000000 max_abs 0.000 top1 12366/12366
logits p3_code.logits.f32 cos 1.000000 max_abs 0.000 top1 76798/76798
logits p5_long_ctx.logits.f32 cos 1.000000 max_abs 0.000 top1 578/578
```

Windows 0-1 (512 tokens) vs last night's HF NLL, which used HF's default **sdpa** attention backend (not eager):

```
0 ppl_a 12.331 ppl_b 12.264 mean|dlogp| 0.0536 max 0.609
1 ppl_a 20.687 ppl_b 20.677 mean|dlogp| 0.0605 max 0.512
```
The window-level |Δlogp| (~0.05) is the eager-vs-sdpa rounding difference in bf16 (the eager comparison above is exact).

## A.6b golden fp32 PPL (same windows and protocol as 2026-11-18)

WikiText-2, 16 x 512:

| file | windows | PPL | 95% CI | ratio vs nll_golden_float32.txt | 95% CI of ratio |
|---|---|---|---|---|---|
| nll_golden_float32.txt | 16 | 19.066 | [16.163, 22.448] | 1.0000 | [1.0000, 1.0000] |
| nll_hf_bfloat16.txt | 16 | 19.091 | [16.178, 22.467] | 1.0013 | [0.9984, 1.0049] |
| fp32.txt | 16 | 23.131 | [19.555, 27.219] | 1.2132 | [1.1790, 1.2552] |
| int8_fused.txt | 16 | 23.221 | [19.580, 27.403] | 1.2179 | [1.1827, 1.2598] |

Kobza UK, 8 x 512:

| file | windows | PPL | 95% CI | ratio vs nll_golden_float32_uk.txt | 95% CI of ratio |
|---|---|---|---|---|---|
| nll_golden_float32_uk.txt | 8 | 141.911 | [111.691, 176.099] | 1.0000 | [1.0000, 1.0000] |
| nll_hf_bfloat16.txt | 8 | 142.232 | [112.080, 176.042] | 1.0023 | [0.9972, 1.0075] |
| fp32.txt | 8 | 209.357 | [159.924, 266.216] | 1.4753 | [1.4258, 1.5340] |
| int8_fused.txt | 8 | 206.815 | [158.412, 262.867] | 1.4574 | [1.4078, 1.5140] |

**Finding:** golden fp32 and HF bf16 agree within +0.13% (EN) / +0.23% (UK), CIs of the paired ratio include 1. The bf16 arithmetic of the HF reference is NOT a material part of Tenzo's gap: Tenzo fp32-KV is +21.3% (EN) / +47.5% (UK) above golden fp32, i.e. the gap is in the engine. (Tenzo rows are last night's score files, engine unchanged since.)

Peak RSS: golden fp32 batch 8: 3.04 GB (EN), 3.15 GB (UK) (`logs/golden_fp32_*.log`); target was <= 4 GB.
