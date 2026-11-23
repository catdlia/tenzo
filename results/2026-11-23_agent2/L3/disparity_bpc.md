# Language Disparity Normalized per Byte and Character (L3 / T40 part 1)

Metadata: commit `fadfa21`, CPU `12th Gen Intel(R) Core(TM) i3-1215U`, governor `powersave`, load `5.43 5.15 4.48 7/1660 23189`, offline score analysis (no model runs).

## 1. BPC and BPB Disparity (`dBPC = BPC(scheme) - BPC(fp)`)

| model | scheme | lang | win | fertility (tok/char) | base BPC | dBPC [95% CI] | pooled c_UK(BPC) [95% CI] | base BPB | dBPB [95% CI] | pooled c_UK(BPB) [95% CI] |
|---|---|---|---:|---:|---:|---|---|---|---|---|
| BitNet-b1.58-2B-4T | int8_fused (engine vs fp32) | en | 16 | 0.2264 | 0.9612 | +0.0003 [-0.0006, +0.0013] | +0.0020 [-0.0124, +0.0177] | 0.9580 | +0.0003 [-0.0006, +0.0013] | -0.0003 [-0.0040, +0.0036] |
| BitNet-b1.58-2B-4T | int8_fused (engine vs fp32) | uk | 8 | 0.3463 | 2.4677 | +0.0007 [-0.0021, +0.0033] | +0.0020 [-0.0124, +0.0177] | 1.3640 | +0.0004 [-0.0011, +0.0018] | -0.0003 [-0.0040, +0.0036] |
| BitNet-b1.58-2B-4T | S3+s4r32 (uncalibrated) | en | 16 | 0.2264 | 0.9602 | +0.0228 [+0.0173, +0.0285] | +0.0684 [+0.0371, +0.1073] | 0.9570 | +0.0227 [+0.0173, +0.0284] | +0.0176 [+0.0082, +0.0286] |
| BitNet-b1.58-2B-4T | S3+s4r32 (uncalibrated) | uk | 32 | 0.3296 | 2.2675 | +0.0660 [+0.0571, +0.0748] | +0.0684 [+0.0371, +0.1073] | 1.2803 | +0.0371 [+0.0323, +0.0419] | +0.0176 [+0.0082, +0.0286] |
| BitNet-b1.58-2B-4T | S13+s4r32 (K INT4, V ternary) | en | 16 | 0.2264 | 0.9602 | +0.0036 [+0.0005, +0.0067] | +0.0520 [+0.0239, +0.0803] | 0.9570 | +0.0036 [+0.0005, +0.0067] | +0.0247 [+0.0187, +0.0314] |
| BitNet-b1.58-2B-4T | S13+s4r32 (K INT4, V ternary) | uk | 32 | 0.3296 | 2.2675 | +0.0427 [+0.0360, +0.0498] | +0.0520 [+0.0239, +0.0803] | 1.2803 | +0.0237 [+0.0203, +0.0275] | +0.0247 [+0.0187, +0.0314] |
| BitNet-b1.58-2B-4T | S3+s4r32 calibrated on EN | en | 16 | 0.2264 | 0.9602 | +0.0130 [+0.0094, +0.0167] | +0.0624 [+0.0356, +0.0932] | 0.9570 | +0.0130 [+0.0094, +0.0166] | +0.0201 [+0.0120, +0.0289] |
| BitNet-b1.58-2B-4T | S3+s4r32 calibrated on EN | uk | 32 | 0.3296 | 2.2675 | +0.0556 [+0.0483, +0.0628] | +0.0624 [+0.0356, +0.0932] | 1.2803 | +0.0314 [+0.0272, +0.0354] | +0.0201 [+0.0120, +0.0289] |
| BitNet-b1.58-2B-4T | S3+s4r32 calibrated on UK | en | 16 | 0.2264 | 0.9602 | +0.0159 [+0.0117, +0.0203] | +0.0630 [+0.0330, +0.0945] | 0.9570 | +0.0158 [+0.0116, +0.0202] | +0.0149 [+0.0063, +0.0239] |
| BitNet-b1.58-2B-4T | S3+s4r32 calibrated on UK | uk | 32 | 0.3296 | 2.2675 | +0.0516 [+0.0441, +0.0591] | +0.0630 [+0.0330, +0.0945] | 1.2803 | +0.0292 [+0.0248, +0.0335] | +0.0149 [+0.0063, +0.0239] |
| Qwen2.5-1.5B | INT8 per token (tenzo_int8) | en | 16 | 0.2353 | 0.8167 | +0.0052 [+0.0033, +0.0071] | +0.0310 [-0.0151, +0.0777] | 0.8155 | +0.0052 [+0.0033, +0.0071] | +0.0109 [+0.0054, +0.0163] |
| Qwen2.5-1.5B | INT8 per token (tenzo_int8) | uk | 16 | 0.5123 | 1.4758 | +0.0292 [+0.0198, +0.0382] | +0.0310 [-0.0151, +0.0777] | 0.8113 | +0.0161 [+0.0109, +0.0210] | +0.0109 [+0.0054, +0.0163] |
| Qwen2.5-1.5B | S3+s4r32 | en | 16 | 0.2353 | 0.8167 | +0.2267 [+0.1933, +0.2770] | +0.7363 [+0.4592, +0.8967] | 0.8155 | +0.2264 [+0.1931, +0.2765] | +0.1852 [+0.1191, +0.2372] |
| Qwen2.5-1.5B | S3+s4r32 | uk | 16 | 0.5123 | 1.4758 | +0.7500 [+0.6852, +0.8180] | +0.7363 [+0.4592, +0.8967] | 0.8113 | +0.4121 [+0.3760, +0.4499] | +0.1852 [+0.1191, +0.2372] |

## 2. Within-Language Fertility Correlation (`dNLL/char` vs `tokens/char`)

| model | scheme | lang | win | fertility (tok/char) | r(dNLL/char, fertility) [95% CI] | slope (nats/char per tok/char) | R2 |
|---|---|---|---:|---:|---|---|---:|
| BitNet-b1.58-2B-4T | int8_fused (engine vs fp32) | en | 16 | 0.2264 | +0.116 [-0.371, +0.494] | +0.0109 | 0.013 |
| BitNet-b1.58-2B-4T | int8_fused (engine vs fp32) | uk | 8 | 0.3463 | +0.347 [-0.499, +0.913] | +0.0365 | 0.121 |
| BitNet-b1.58-2B-4T | S3+s4r32 (uncalibrated) | en | 16 | 0.2264 | +0.680 [+0.304, +0.903] | +0.3763 | 0.462 |
| BitNet-b1.58-2B-4T | S3+s4r32 (uncalibrated) | uk | 32 | 0.3296 | -0.154 [-0.439, +0.127] | -0.0894 | 0.024 |
| BitNet-b1.58-2B-4T | S13+s4r32 (K INT4, V ternary) | en | 16 | 0.2264 | +0.664 [+0.309, +0.885] | +0.2036 | 0.441 |
| BitNet-b1.58-2B-4T | S13+s4r32 (K INT4, V ternary) | uk | 32 | 0.3296 | -0.280 [-0.501, -0.070] | -0.1272 | 0.078 |
| BitNet-b1.58-2B-4T | S3+s4r32 calibrated on EN | en | 16 | 0.2264 | +0.616 [+0.294, +0.828] | +0.2181 | 0.380 |
| BitNet-b1.58-2B-4T | S3+s4r32 calibrated on EN | uk | 32 | 0.3296 | -0.166 [-0.452, +0.109] | -0.0789 | 0.028 |
| BitNet-b1.58-2B-4T | S3+s4r32 calibrated on UK | en | 16 | 0.2264 | +0.706 [+0.358, +0.913] | +0.2971 | 0.498 |
| BitNet-b1.58-2B-4T | S3+s4r32 calibrated on UK | uk | 32 | 0.3296 | -0.146 [-0.454, +0.163] | -0.0720 | 0.021 |
| Qwen2.5-1.5B | INT8 per token (tenzo_int8) | en | 16 | 0.2353 | -0.022 [-0.502, +0.361] | -0.0043 | 0.000 |
| Qwen2.5-1.5B | INT8 per token (tenzo_int8) | uk | 16 | 0.5123 | -0.326 [-0.766, +0.171] | -0.1765 | 0.106 |
| Qwen2.5-1.5B | S3+s4r32 | en | 16 | 0.2353 | -0.045 [-0.352, +0.475] | -0.1986 | 0.002 |
| Qwen2.5-1.5B | S3+s4r32 | uk | 16 | 0.5123 | -0.351 [-0.751, +0.174] | -1.3905 | 0.123 |

## 3. Conclusion

After normalizing for UTF-8 byte and Unicode character length, Ukrainian's excess KV-quantization degradation (**$c_{\text{UK}} > 0$**) holds across **BitNet-b1.58-2B-4T** ternary schemes (`S3+s4r32`, `S13+s4r32`, EN-calibrated, and UK-calibrated: `c_UK(BPC) = +0.0520` to `+0.0684`, `c_UK(BPB) = +0.0149` to `+0.0247`, all 95% CIs strictly $> 0$; engine `int8_fused` vs `fp32` is near zero in both languages, CI includes `0`) and across **Qwen2.5-1.5B** under byte normalization (`BPB`) for both `tenzo_int8` (`c_UK(BPB) = +0.0109 [+0.0054, +0.0163]`, where EN and UK have virtually identical `base BPB` `0.8155` vs `0.8113`) and `S3+s4r32` (`c_UK(BPB) = +0.1852 [+0.1191, +0.2372]`). Under character normalization (`BPC`) on Qwen2.5-1.5B, `c_UK(BPC) = +0.7363 [+0.4592, +0.8967]` for `S3+s4r32`, while for `tenzo_int8` the `mean dBPC` is `5.6x` larger on UK (`+0.0292` vs `+0.0052`) yet the pooled regression CI (`+0.0310 [-0.0151, +0.0777]`) crosses `0` due to collinearity between the `UK` indicator and non-overlapping `base BPC` (`1.4758` UK vs `0.8167` EN, driven by `2.18x` higher UK fertility `0.5123` vs `0.2353`). Within-language fertility (`tokens/char`) correlates positively with `dNLL/char` inside BitNet EN (`r = +0.62` to `+0.71`), but shows weak or negative correlation inside BitNet UK (`r = -0.15` to `-0.28`) and Qwen2.5 (`r = -0.02` to `-0.35`), showing that higher subword fertility alone does not explain Ukrainian's larger degradation.

## 4. Artifacts & Reproduction
- Script: `scripts/analyze_disparity_bpc.py`
- Data: `results/2026-11-23_agent2/L3/disparity_bpc.csv`, `results/2026-11-23_agent2/L3/disparity_bpc_windows.csv`
- Plots: `results/2026-11-23_agent2/L3/disparity_bpc_scatter.png`, `results/2026-11-23_agent2/L3/fertility_corr.png`
