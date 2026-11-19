# Design: batched prefill (T32)

Today `tenzo-inference` feeds the prompt token by token through the decode path. Each token re-reads every weight byte
(2.4 B ternary weights ≈ 0.6 GB at 2 bit) and runs 7 GEMVs per layer, so prefill speed is close to decode speed.
llama.cpp processes the prompt as one batch.

## BitLinear for M > 1 tokens (on the I2_S layout)

- **Activations.** Quantize each of the M token rows separately (HF ActQuant is per token). This gives int8 codes
  `Q[M][K]`, one sum `Σq_m` per token and one `inv_scale_m` per token.
- **Weights.** For each output-row tile (e.g. 4 rows × K) and each 128-input block, load and unpack the 32 weight bytes
  into u0..u3 **once**. Then loop over the M tokens: `maddubs(u_s, q_m,s)` → `madd` → accumulate `acc[row][m]`.
  The unpacked tile stays in registers, so weight traffic drops by a factor of M.
  - Register budget: 4 u-vectors + 4 q-vectors + 4×(rows) accumulators.
  - Use 2 rows × 4 tokens, or 4 rows × 2 tokens, per inner kernel.
- **Output.** `y[m][n] = (acc[n][m] − Σq_m) · inv_scale_m · w_scale`. These are the same integer sums as the GEMV path,
  so each token's outputs are **bit-identical** to the sequential path, as long as RMSNorm and the residual adds run
  per row in the same order.

## Attention inside a chunk (causal)

Chunk of C tokens at positions p0..p0+C−1, processed per layer:
1. Run QKV GEMMs for all C tokens.
2. Apply RoPE per token at its own position (the same per-position code as today).
3. Append all C keys/values to the KV cache using the existing per-token quantization, so int8/tl1 scales are identical.
4. For token i, attend over cache positions [0, p0 + i]. This is the same loop as decode with `seq_len = p0 + i + 1`,
   so per-token results are identical to sequential.
   - Parallelize over (token, head) pairs, instead of heads only.
5. Run attn_sub_norm, o_proj (GEMM), residual, post-norm, gate/up GEMM, relu², ffn_sub_norm, down GEMM and residual,
   all row-wise.

## KV cache writes

`FusedKVCache::forward_attention_raw` currently does RoPE, append and attend for one position, then the caller
increments `cur_seq_len`. The batched version needs two separate operations:
- `append(layer, pos, k, v)`: RoPE plus quantize-and-store for one position.
- `attend(layer, pos, q)`: attention for one query over [0, pos].

The cache length advances by C after the whole chunk has gone through all layers. The fp32 and int8 contents must
equal the sequential path bit for bit; this is the gate (max_abs ≤ 1e-5, expected 0).

## Interface and gates

- **Interface.** `--prefill sequential|batched` (default sequential until the gates pass) and `--prefill-chunk 32|64`.
  `--score` switches to batched only if 16-window EN PPL matches sequential to the 3rd decimal.
- **Gates.**
  - Next-token logits after the prompt: max_abs ≤ 1e-3 and the same top-1 on 7 prompts (expected exact equality,
    because the math is the same per token).
  - KV contents after prefill: fp32 and int8_fused identical.

## Expected gain

Prefill becomes compute-bound instead of weight-bandwidth-bound: one weight unpack serves C tokens.
llama.cpp reaches about 2× Tenzo's token-by-token prefill on this laptop (2026-11-18 e2e). Speed is measured only on a clean machine.
