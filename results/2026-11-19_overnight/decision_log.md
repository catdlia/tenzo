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
- **D6 (13:47 -> 18:59) Session interruption.** The FP32-LM diagnostic run (started 13:47) finished all 16 windows
  at 14:03 (score file complete), but the tool call returned exit 137 and the agent session stopped; the owner
  wrote "продовжуй" at 18:59. No OOM entry found in the kernel log for that window. Active time counted:
  12:46-14:03 = 1 h 17 min. On resume the owner is using the laptop (a game holds ~2 GB, MemAvailable 3.7 GB,
  swap 6.7 GB used): per rule 0.5 no speed benchmarks while the owner is at the laptop, and per 0.3 heavy steps
  only when MemAvailable >= need + 2 GB.
- **D7** Act-quant fix: two int8 LUTs per activation pair (w0*a0, w1*a1) so activations can use the full HF range
  (Qp=127) instead of +-31; half-to-even rounding (std::nearbyint) and the 1e-5 floor as in HF ActQuant.
  fp32-KV PPL 19.545 -> 19.124 (golden 19.066, paired ratio 1.0030 [1.0003, 1.0057]).
- **D8** Remaining 0.3%: the diagnostic TENZO_EXPERIMENT_FP32_LM=1 run gives 19.088, ratio 1.0011 [0.9980, 1.0046]
  -> the residual is the INT8 embedding table / INT8 LM head (a deliberate memory trade-off; default unchanged).
- **D9 (19:05)** MemAvailable 3.73 GB after the 5-minute wait (needed ~4.8 GB for tenzo-inference). Per rule 0.3 the
  G6-after and speed check for fix 2 are skipped for now and recorded as a blocker; fix 2 is committed with the tensor
  and PPL evidence only.
- **D10 (19:26) Second resume** ("ноут вільний"): new 6 h budget from 19:26 (last block start 00:46, end 01:26).
  MemAvailable 9.9 GB. Inhibitor restarted (the first one was still alive). governor powersave, EPP performance, AC on.
- **D11 (19:58) Background CPU load not owned by this run**: firefox, `neko` and a browser isolate process use
  ~100% CPU in total (load average 8-12 while tenzo-inference uses ~3 cores). Each fp32 PPL pass took 25 min instead of 10 min at 13:37.
  PPL is deterministic and unaffected. Speed numbers ARE affected; the bitnet.cpp f16 control point is re-run
  in the same session to expose machine drift, and the speed section carries this caveat. These processes
  were not touched (they are the owner's).
- **D12** `test_fixture_regressions.py` run on the host fails with rc=127 (`libMLIR.so.21.1` missing on the host);
  it is a Docker-only suite: the gate was re-run inside the dev container -> exit 0. The host log is kept as
  `logs/gate_test_fixture_regressions_host_wrongenv.log`.
