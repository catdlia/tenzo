# Batched prefill (T32)

Design: `design_batched_prefill.md`. Implementation: `TenzoEngineImpl::prefill_tokens_batched` (CPU + I2_S GEMV + non-paged KV),
`gemm_i2s_rows` (weight tile of 2 rows unpacked once and applied to 4 tokens at a time), API `tenzo_prefill_tokens(engine, tokens, n, chunk)`,
CLI `--prefill sequential|batched` (default sequential) and `--prefill-chunk N` (default 32). Unsupported configurations
(GPU, TENZO_GEMV=lut, paged KV) return TENZO_ERROR_UNSUPPORTED and the CLI falls back to sequential with a warning.

Gates:
- Tiny 2-layer model with ffn != hidden, chunks 8/16/32/64, kv fp32 / int8_fused / ternary_rot: next-token logits and every byte of
  the KV cache identical (`tenzo-cli engine-api`, `logs/engine_api_full.log`, `logs/gates_features/`).
- Real model, 7 prompts x {fp32, int8_fused, ternary_rot}, chunk 32: **max_abs of the next-token logits 0 (bit-identical), top-1 equal 21/21,
  8-token greedy equal 21/21** (`prefill/batched_vs_sequential.md`). The first attempt failed (decision log D15) and was fixed by making
  `rms_norm_raw` noinline.
- Required thresholds were max_abs <= 1e-3 and KV max_abs <= 1e-5; the result is exact equality.

`--score` stays sequential: scoring needs the logits of every position (LM head per token), so a batched prefill does not apply; the
PPL runs were therefore not made faster.

Speed (TTFT, prefill tok/s) is not measured: the machine is not clean (decision log D7/D13). It is queued: `scripts/ab_prefill_speed.sh`
in `scripts/pending_benchmarks.sh`.
