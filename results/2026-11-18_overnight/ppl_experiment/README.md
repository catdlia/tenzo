# Diagnostic: is the INT8 embedding / LM head the cause of the PPL gap vs HF?

`TENZO_EXPERIMENT_FP32_LM=1` (opt-in env switch added in TenzoEngine.cpp; default behaviour unchanged, G6 byte-identical) keeps token embeddings and the
LM head in float32 instead of per-row INT8. Same 16 WikiText-2 windows, fp32 KV cache (`fp32kv_fp32lm.txt`, log next to it).

| config | PPL (16 x 511 tokens) |
|---|---|
| HF transformers bf16 (reference) | 19.091 |
| Tenzo, fp32 KV, INT8 embeddings + INT8 LM head (default) | 23.131 |
| Tenzo, fp32 KV, FP32 embeddings + FP32 LM head | 23.257 |

Result: removing the INT8 head/embedding quantisation does **not** close the gap (23.26 vs 23.13, i.e. not better). The ~+21% PPL gap to HF
must come from the transformer body (activation quantisation details, RoPE/attention/norm differences, bf16 scale rounding of <= 0.4%) or from the
bf16 reference itself. Not investigated further (2-approach limit): see OVERNIGHT_REPORT.md.
