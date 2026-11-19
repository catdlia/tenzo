# KV quantization language disparity on a second model (Qwen2.5-1.5B)

Setup (`tools/reference/second_model_kv.py`, HF transformers 4.57.6): Qwen2.5-1.5B (Qwen2, 12 q / 2 kv heads, head_dim 128, SwiGLU, QKV bias, rope_theta 1e6,
151,936-token vocabulary; downloaded once from HF to /tmp, decision log D12) with bf16 weights and exact fp32 compute (every weight upcast at use); the
attention function is `modeling_qwen2.eager_attention_forward` replaced by one that fake-quantizes post-RoPE K and V with the same hooks as the BitNet study
(`tools/reference/kv_ptq.py`; Hadamard schemes simulated as deq(Q(k R)) R^T). Windows: 512 tokens of the Qwen tokenizer, 16 EN (WikiText-2 test, RandomState(0)
offsets) and 16 UK (the Kobza first-rows documents, consecutive windows); 511 scored tokens per window. Reference ("golden") = the same model with fp KV.

**Patch check:** HF eager attention vs the patched attention without quantization: identical log-probabilities on the first 4 windows of both languages
(mean and max |dlogp| = 0; `second_model/patch_check.md`).

Results (`second_model.md`, `second_model.csv`; PPL ratio to the fp-KV model, 95% paired bootstrap CI; dNLL in nats/token):

| scheme | EN ratio | UK ratio | EN dNLL | UK dNLL | c_UK at equal difficulty [95% CI] |
|---|---|---|---|---|---|
| INT8 per token | 1.016 [1.010, 1.021] | 1.041 [1.028, 1.054] | +0.015 | +0.040 | +0.030 [+0.015, +0.046] |
| ternary g32 | 1828 | 4841 | +7.5 | +8.5 | +0.51 [-0.004, +0.91] |
| ternary g32 + Hadamard | 1922 | 933 | +7.6 | +6.8 | -0.79 [-1.62, +0.19] |
| ternary g32 + Hadamard + 4 sink / 32 recent INT8 | 1.956 [1.77, 2.28] | 2.774 [2.50, 3.08] | +0.67 | +1.02 | +0.37 [+0.16, +0.53] |

Base NLL (fp KV): EN 2.41, UK 2.00 nats/token: **for this model Ukrainian text is the easier one**, yet it loses more under INT8 and under windowed ternary.

Conclusions (second model):
- "Ukrainian suffers more" holds for INT8 per-token KV (4.1% vs 1.6% PPL; c_UK > 0, CI excludes 0) and for ternary + INT8 windows (2.77x vs 1.96x). Because UK is easier than EN for
  Qwen, text difficulty cannot be the explanation here; it is not a BitNet-only or Llama-3-tokenizer-only effect.
- For the two ternary schemes without INT8 windows the model is destroyed in both languages (PPL in the thousands to tens of thousands); the sign of the difference is
  not meaningful there (CIs of c_UK contain 0).
- Ternary KV is far worse on Qwen2.5-1.5B than on BitNet (S3 + windows: x1.96 / x2.77 vs x1.07 / x1.15). Plausible cause (not tested): Qwen2.5 has QKV biases and strong
  outlier channels in K; the Hadamard rotation did not rescue the unwindowed scheme.
- Limits: one UK corpus (Kobza first rows), 16 windows per language (16 EN, 16 UK), per-token NLL is not comparable across tokenizers, only one second model, no per-character normalization.
