# Fix 1: RoPE theta (engine used 10000, model uses 500000)

**Cause.** `FusedKVCache::forward_attention_raw` and `PagedKVCache::forward_attention_raw` hard-coded `theta_base = 10000.0f`. HF: `BitNetRotaryEmbedding` -> `ROPE_INIT_FUNCTIONS['default']` -> `inv_freq = 1/base^(arange(0,dim,2)/dim)` with `base = config.rope_theta`; config.json of microsoft/bitnet-b1.58-2B-4T: `"rope_theta": 500000.0`.

**Fix.** `tenzo_config_t.rope_theta` (default 500000 = model config; `tenzo_create_engine` rejects <= 0); `tenzo-inference` reads `rope_theta` from `<model>/config.json` when present and warns otherwise (the existing `models/bitnet-1.58b/config.json` lacks it, so the run logs the warning and uses the default). `export_bitnet.py` now refuses to export without rope_theta / rms_norm_eps. Weights untouched.

**Signature (before).** At pos 0 the rotation angle is 0, so q_rope == q; error grows with position. After the fix q_rope/k_rope error equals the pre-RoPE q/k error at every position.

Window 0 of WikiText-2 (fp32 KV), engine vs golden fp32 (`divergence/cmp_before_w0.md`, `divergence/cmp_theta_w0.md`):

| tensor | pos | rel L2 before | cos before | rel L2 after | cos after |
|---|---|---|---|---|---|
| L0.q | 0 | 5.133e-02 | 0.9987 | 5.133e-02 | 0.9987 |
| L0.q | 1 | 1.992e-02 | 0.9998 | 1.992e-02 | 0.9998 |
| L0.q | 5 | 2.719e-02 | 0.9996 | 2.719e-02 | 0.9996 |
| L0.q | 50 | 2.933e-02 | 0.9996 | 2.933e-02 | 0.9996 |
| L0.q | 200 | 2.476e-02 | 0.9997 | 2.476e-02 | 0.9997 |
| L0.q | 511 | 2.864e-02 | 0.9996 | 2.864e-02 | 0.9996 |
| L0.q_rope | 0 | 5.133e-02 | 0.9987 | 5.133e-02 | 0.9987 |
| L0.q_rope | 1 | 6.128e-02 | 0.9981 | 1.992e-02 | 0.9998 |
| L0.q_rope | 5 | 2.566e-01 | 0.9671 | 2.719e-02 | 0.9996 |
| L0.q_rope | 50 | 9.106e-01 | 0.5859 | 2.933e-02 | 0.9996 |
| L0.q_rope | 200 | 1.076e+00 | 0.4214 | 2.476e-02 | 0.9997 |
| L0.q_rope | 511 | 1.209e+00 | 0.2676 | 2.864e-02 | 0.9996 |
| L0.k_rope | 0 | 2.296e-02 | 0.9998 | 2.296e-02 | 0.9998 |
| L0.k_rope | 1 | 3.517e-02 | 0.9994 | 1.380e-02 | 0.9999 |
| L0.k_rope | 5 | 1.964e-01 | 0.9807 | 1.973e-02 | 0.9998 |
| L0.k_rope | 50 | 6.516e-01 | 0.7884 | 2.140e-02 | 0.9998 |
| L0.k_rope | 200 | 9.744e-01 | 0.5254 | 1.873e-02 | 0.9998 |
| L0.k_rope | 511 | 1.324e+00 | 0.1228 | 2.043e-02 | 0.9998 |
| L0.attn_out | 0 | 2.065e-02 | 0.9998 | 2.065e-02 | 0.9998 |
| L0.attn_out | 1 | 2.662e-02 | 0.9997 | 2.044e-02 | 0.9998 |
| L0.attn_out | 5 | 7.635e-02 | 0.9973 | 4.154e-02 | 0.9993 |
| L0.attn_out | 50 | 1.534e-01 | 0.9882 | 1.564e-02 | 0.9999 |
| L0.attn_out | 200 | 1.308e-01 | 0.9920 | 1.554e-02 | 0.9999 |
| L0.attn_out | 511 | 2.519e-01 | 0.9704 | 1.327e-02 | 0.9999 |
| L0.resid_ffn | 0 | 1.579e-01 | 0.9963 | 1.579e-01 | 0.9963 |
| L0.resid_ffn | 1 | 9.177e-02 | 0.9958 | 8.475e-02 | 0.9964 |
| L0.resid_ffn | 5 | 1.336e-01 | 0.9919 | 1.030e-01 | 0.9948 |
| L0.resid_ffn | 50 | 1.221e-01 | 0.9925 | 6.796e-02 | 0.9977 |
| L0.resid_ffn | 200 | 1.410e-01 | 0.9906 | 9.150e-02 | 0.9958 |
| L0.resid_ffn | 511 | 2.107e-01 | 0.9810 | 9.304e-02 | 0.9957 |
| L15.attn_out | 0 | 9.146e-02 | 0.9990 | 9.146e-02 | 0.9990 |
| L15.attn_out | 1 | 1.318e-01 | 0.9959 | 1.247e-01 | 0.9966 |
| L15.attn_out | 5 | 2.792e-01 | 0.9736 | 1.917e-01 | 0.9887 |
| L15.attn_out | 50 | 3.315e-01 | 0.9516 | 1.763e-01 | 0.9904 |
| L15.attn_out | 200 | 9.969e-01 | 0.7267 | 1.955e-01 | 0.9807 |
| L15.attn_out | 511 | 7.763e-01 | 0.8275 | 2.160e-01 | 0.9771 |
| L29.resid_ffn | 0 | 1.217e-01 | 0.9947 | 1.217e-01 | 0.9947 |
| L29.resid_ffn | 1 | 1.555e-01 | 0.9929 | 1.191e-01 | 0.9948 |
| L29.resid_ffn | 5 | 1.092e-01 | 0.9962 | 6.456e-02 | 0.9980 |
| L29.resid_ffn | 50 | 1.472e-01 | 0.9914 | 8.573e-02 | 0.9970 |
| L29.resid_ffn | 200 | 1.918e-01 | 0.9915 | 7.571e-02 | 0.9975 |
| L29.resid_ffn | 511 | 1.954e-01 | 0.9808 | 7.654e-02 | 0.9971 |
| final_norm | 0 | 3.416e-01 | 0.9472 | 3.416e-01 | 0.9472 |
| final_norm | 1 | 2.523e-01 | 0.9677 | 2.094e-01 | 0.9778 |
| final_norm | 5 | 2.113e-01 | 0.9781 | 1.576e-01 | 0.9885 |
| final_norm | 50 | 2.337e-01 | 0.9753 | 1.347e-01 | 0.9911 |
| final_norm | 200 | 3.629e-01 | 0.9408 | 2.008e-01 | 0.9796 |
| final_norm | 511 | 3.971e-01 | 0.9216 | 1.662e-01 | 0.9866 |
| logits | 0 | 2.638e-01 | 0.9885 | 2.638e-01 | 0.9885 |
| logits | 1 | 1.099e-01 | 0.9954 | 1.053e-01 | 0.9970 |
| logits | 5 | 1.766e-01 | 0.9957 | 1.480e-01 | 0.9975 |
| logits | 50 | 1.392e-01 | 0.9960 | 1.197e-01 | 0.9958 |
| logits | 200 | 1.171e-01 | 0.9971 | 8.920e-02 | 0.9991 |
| logits | 511 | 3.864e-01 | 0.9553 | 9.096e-02 | 0.9979 |

**PPL (fp32 KV, WikiText-2 16 x 512)**:

| file | windows | PPL | 95% CI | ratio vs nll_golden_float32.txt | 95% CI of ratio |
|---|---|---|---|---|---|
| nll_golden_float32.txt | 16 | 19.066 | [16.163, 22.448] | 1.0000 | [1.0000, 1.0000] |
| fp32.txt | 16 | 23.131 | [19.555, 27.219] | 1.2132 | [1.1790, 1.2552] |
| fp32.txt | 16 | 19.545 | [16.647, 22.873] | 1.0251 | [1.0129, 1.0376] |

(second row = before, last night's file; third row = after, `ppl_theta/fp32.txt`).

**Attribution in the golden (8 EN windows, diagnostic emulation flags, `attribution/`)**: theta 10000 alone x1.205, Tenzo activation quant (qp 31, half-away) alone x1.037, both x1.232; Tenzo before fix x1.213.

**G6 before -> after** (`g6/before_*.txt`, `g6/after_theta_*.txt`):

- fp32 before: ` Paris. Paris is a city in the Ãİle-de-France region of France. It is the capital city of France`
- fp32 after:  ` Paris. Paris is a city in the north of France. It is a very important city in the world. It is`
- int8_fused before: ` Paris. Paris is a city in the Ãİle-de-France region of France. It is the capital city of France`
- int8_fused after:  ` Paris. Paris is a city in the Ãİle-de-France region of France. It is located in the north-central`
- tl1_fused before: ` Paris. Paris is the capital of France. It is also the capital of the European Union, which is also the capital`
- tl1_fused after:  ` Paris. Paris is the capital of France. Paris is the capital of France. Paris is the capital of France. Paris`
- popcount_fused before: ` Paris. Paris is the capital of france. Paris is the capital of france. Paris is the the the.    The`
- popcount_fused after:  ` the capital of the United States of America is the capital of the United Kingdom of the United Kingdom of the United Kingdom of`

**Not fixed (same 10000 constant, other paths, not used by tenzo-inference):** `src/bindings/PybindModule.cpp:587` (separate Python-binding KV cache), `src/passes/LinalgLowering.cpp:207-218` (MLIR `tenzo.rope` lowering; the op has no theta attribute).
