# PTQ finalists on the full eval sets (part 2, item 0c)

Golden fp32 + fake-quant (tools/reference/kv_ptq.py), quiet mode (nice, MemoryMax=3G). UK = 32 windows (windows_uk32.ids), EN = 16 windows. 95% bootstrap CIs, paired ratio vs golden fp32-KV on the same windows. S3+s4r32 EN row is the 2026-11-19 file (same protocol).

UK32:

| file | windows | PPL | 95% CI | ratio vs fp32.txt | 95% CI of ratio |
|---|---|---|---|---|---|
| fp32.txt | 32 | 118.217 | [102.417, 136.219] | 1.0000 | [1.0000, 1.0000] |
| S3_s4r32.txt | 32 | 136.094 | [118.057, 156.553] | 1.1512 | [1.1281, 1.1749] |
| S13_s4r32.txt | 32 | 129.568 | [112.082, 149.490] | 1.0960 | [1.0791, 1.1140] |

EN16:

| file | windows | PPL | 95% CI | ratio vs nll_golden_float32.txt | 95% CI of ratio |
|---|---|---|---|---|---|
| nll_golden_float32.txt | 16 | 19.066 | [16.163, 22.448] | 1.0000 | [1.0000, 1.0000] |
| S3_s4r32.txt | 16 | 20.419 | [17.453, 23.855] | 1.0709 | [1.0547, 1.0883] |
| S13_s4r32.txt | 16 | 19.265 | [16.388, 22.618] | 1.0104 | [1.0009, 1.0201] |

Consistency: golden fp32 on UK32 windows 0-7 reproduces the 2026-11-19 UK8 golden file exactly:

| nll_golden_float32_uk.txt | 8 | 141.911 | [109.281, 178.420] | 1.0000 | [1.0000, 1.0000] |

Bits/element at ctx 512: S3+s4r32 2.896, S13+s4r32 4.058 (asymptotic 2.5 / 3.75; kv_ptq.bits). ctx-2048 runs: not done (time).
