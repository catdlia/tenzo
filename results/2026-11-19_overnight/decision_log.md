# Decision log — 2026-11-19 overnight (branch overnight/2026-11-19)

Times are local (EET). Memory readings before each heavy step: `logs/mem.log`.

- **D1 (12:46)** Branch cut from origin/overnight/2026-11-18 (aad222d). The auto-set upstream (overnight/2026-11-18) was
  unset so that a bare push can never target the old branch; pushes always name `origin overnight/2026-11-19`.
- **D2 (12:50)** Golden reference written from HF code only (transformers 4.57.6 modeling_bitnet.py,
  integrations/bitnet.py, config.json). The engine sources were not opened until the golden was validated.
- **D3 (12:56)** First golden-bf16 vs HF-bf16 comparison diverged from L0.q (rel L2 8e-3) while L0.input_norm was
  bit-identical. Cause was in the golden, not HF: HF `ActQuant.forward` casts to fp32 before quantizing
  (`activation = activation.float()`), my first version quantized in bf16. Fixed the golden (not a threshold);
  after the fix golden-bf16 == HF-bf16-eager bit-for-bit on every hooked tensor (L0, L29, pos 0 and 5) and on the
  first-step logits of p1, p3, p5 (`golden_validation.md`).
- **D4** Validation backend labels: the prompt comparison used HF **eager** attention (run tonight); last night's
  window NLL file `results/reference/hf/nll_hf_bfloat16.txt` used the HF default (**sdpa**). The residual
  window-level difference golden-bf16 vs HF-bf16-sdpa is therefore an attention-backend rounding difference in bf16,
  not a golden bug.
- **D5 (13:05)** Engine read after golden validation. Two semantic deviations from HF seen in code before any dump:
  (a) RoPE `theta_base = 10000.0f` hard-coded in FusedKVCache and PagedKVCache, config.json says
  `rope_theta = 500000`; (b) `build_lut` quantizes activations to ±31 (`31/max_abs`, ~6 bit) where HF uses
  Qp=127 (8 bit), plus half-away rounding (std::round) vs torch half-to-even, floor 1e-6 vs 1e-5.
  GQA mapping is correct (`kv_h = qh / gqa_ratio`, matches repeat_kv). Decision (advisor-reviewed): measure each
  deviation's share with opt-in DIAGNOSTIC emulation flags in the golden (`--emulate-rope-theta`,
  `--emulate-act-qp`; default = HF semantics) before rewriting any kernel.
