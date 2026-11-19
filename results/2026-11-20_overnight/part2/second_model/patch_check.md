# Patch-equivalence check (Qwen2.5-1.5B): HF eager attention vs this module's attention without quantization

| language | windows | PPL eager | PPL patched (fp KV) | mean abs dlogp | max abs dlogp |
|---|---|---|---|---|---|
| en | 4 | 12.237449 | 12.237449 | 0.00e+00 | 0.00e+00 |
| uk | 4 | 7.810895 | 7.810895 | 0.00e+00 | 0.00e+00 |
