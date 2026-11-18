# Fix 2: activation quantization (engine used +-31 levels, HF uses Qp=127)

**Cause.** `build_lut` (src/runtime/TenzoEngine.cpp) scaled activations by `31/max_abs` so that the pair sum w0*a0 + w1*a1 fits one int8 LUT entry (~6-bit activations), with std::round (half away) and a 1e-6 floor. HF `integrations/bitnet.py ActQuant.forward`: `scale = 127 / x.abs().max(-1).clamp_(min=1e-5)`, `(x*scale).round().clamp(-128,127) / scale`, computed in fp32 (torch.round = half to even).

**Fix.** Two int8 tables per activation pair (w0*a0 and w1*a1, each |.| <= 127), two shuffles per nibble, sum in int16 (<= 64*254 per chunk). Scale 127/max(|x|, 1e-5), std::nearbyint (half to even). Weight layout and weights.bin unchanged.

**Tensors, WikiText-2 window 0, fp32 KV, vs golden fp32** (`divergence/cmp_theta_w0.md` -> `divergence/cmp_aq_w0.md`):

| tensor | pos | rel L2 before (theta fix only) | rel L2 after (8-bit act) | cos after |
|---|---|---|---|---|
| L0.q | 0 | 5.133e-02 | 3.468e-02 | 0.99940 |
| L0.q | 5 | 2.719e-02 | 1.166e-02 | 0.99993 |
| L0.q | 200 | 2.476e-02 | 1.112e-02 | 0.99994 |
| L0.q | 511 | 2.864e-02 | 1.153e-02 | 0.99993 |
| L0.v | 0 | 2.065e-02 | 1.313e-02 | 0.99992 |
| L0.v | 5 | 1.244e-02 | 5.845e-03 | 0.99998 |
| L0.v | 200 | 1.144e-02 | 5.480e-03 | 0.99999 |
| L0.v | 511 | 1.605e-02 | 7.458e-03 | 0.99998 |
| L0.o_proj | 0 | 7.369e-02 | 4.244e-02 | 0.99916 |
| L0.o_proj | 5 | 3.441e-02 | 2.936e-02 | 0.99970 |
| L0.o_proj | 200 | 3.743e-02 | 1.580e-02 | 0.99988 |
| L0.o_proj | 511 | 3.892e-02 | 1.835e-02 | 0.99986 |
| L0.down | 0 | 1.589e-01 | 5.443e-02 | 0.99880 |
| L0.down | 5 | 1.052e-01 | 3.410e-02 | 0.99953 |
| L0.down | 200 | 9.177e-02 | 2.802e-02 | 0.99962 |
| L0.down | 511 | 9.330e-02 | 2.649e-02 | 0.99970 |
| L0.resid_ffn | 0 | 1.579e-01 | 5.415e-02 | 0.99881 |
| L0.resid_ffn | 5 | 1.030e-01 | 3.345e-02 | 0.99955 |
| L0.resid_ffn | 200 | 9.150e-02 | 2.794e-02 | 0.99962 |
| L0.resid_ffn | 511 | 9.304e-02 | 2.642e-02 | 0.99971 |
| L1.down | 0 | 1.241e-01 | 4.299e-02 | 0.99913 |
| L1.down | 5 | 1.523e-01 | 4.209e-02 | 0.99912 |
| L1.down | 200 | 1.363e-01 | 3.875e-02 | 0.99925 |
| L1.down | 511 | 1.510e-01 | 3.490e-02 | 0.99939 |
| L15.attn_out | 0 | 9.146e-02 | 1.156e-02 | 0.99993 |
| L15.attn_out | 5 | 1.917e-01 | 5.468e-02 | 0.99850 |
| L15.attn_out | 200 | 1.955e-01 | 5.353e-02 | 0.99863 |
| L15.attn_out | 511 | 2.160e-01 | 4.192e-02 | 0.99916 |
| L29.down | 0 | 1.851e-01 | 4.892e-02 | 0.99884 |
| L29.down | 5 | 1.759e-01 | 3.868e-02 | 0.99929 |
| L29.down | 200 | 1.831e-01 | 3.444e-02 | 0.99947 |
| L29.down | 511 | 1.479e-01 | 3.986e-02 | 0.99930 |
| L29.resid_ffn | 0 | 1.217e-01 | 2.396e-02 | 0.99971 |
| L29.resid_ffn | 5 | 6.456e-02 | 1.576e-02 | 0.99988 |
| L29.resid_ffn | 200 | 7.571e-02 | 1.768e-02 | 0.99986 |
| L29.resid_ffn | 511 | 7.654e-02 | 2.049e-02 | 0.99979 |
| final_norm | 0 | 3.416e-01 | 8.802e-02 | 0.99628 |
| final_norm | 5 | 1.576e-01 | 3.443e-02 | 0.99941 |
| final_norm | 200 | 2.008e-01 | 4.747e-02 | 0.99888 |
| final_norm | 511 | 1.662e-01 | 4.147e-02 | 0.99914 |
| logits | 0 | 2.638e-01 | 7.060e-02 | 0.99809 |
| logits | 5 | 1.480e-01 | 1.717e-02 | 0.99991 |
| logits | 200 | 8.920e-02 | 1.516e-02 | 0.99993 |
| logits | 511 | 9.096e-02 | 2.053e-02 | 0.99986 |

**PPL, fp32 KV, WikiText-2 16 x 512:**

| file | windows | PPL | 95% CI | ratio vs nll_golden_float32.txt | 95% CI of ratio |
|---|---|---|---|---|---|
| nll_golden_float32.txt | 16 | 19.066 | [16.163, 22.448] | 1.0000 | [1.0000, 1.0000] |
| fp32.txt | 16 | 23.131 | [19.555, 27.219] | 1.2132 | [1.1790, 1.2552] |
| fp32.txt | 16 | 19.545 | [16.647, 22.873] | 1.0251 | [1.0129, 1.0376] |
| fp32.txt | 16 | 19.124 | [16.229, 22.492] | 1.0030 | [1.0003, 1.0057] |
| fp32_fp32lm.txt | 16 | 19.088 | [16.185, 22.457] | 1.0011 | [0.9980, 1.0046] |

Rows: before both fixes (2026-11-18), after theta fix, after theta + act-quant fix (default INT8 embeddings/LM head), and the same with the diagnostic `TENZO_EXPERIMENT_FP32_LM=1`. The remaining +0.3% is the INT8 embedding/LM-head quantization (the FP32-LM run's CI includes 1).

**NOT yet done for this fix (blocker, decision log D6/D9):** G6 after the fix in 4 KV modes and the decode-speed regression check. After the session resumed at 18:59 the owner was using the laptop (MemAvailable 3.7 GB < 2.8 GB + 2 GB needed; rule 0.5 forbids speed benchmarks while the owner is at the laptop). The kernel does twice the shuffles per weight byte, so a decode slowdown is expected and must be measured before this is called a free fix.
