# Decision log — 2026-11-20 overnight (branch overnight/2026-11-20, cut from overnight/2026-11-19@5e7e235)

- **D1 (01:00)** The pasted task text ends mid-sentence in block A ("Гейти коректності (обидва ядра рахують точні…").
  Interpretation: both kernels compute exact integer dot products with the same activation codes and the same float
  scaling, so the gate is BIT-IDENTITY of outputs (logits, G6, scores) between `lut` and `i2s`, plus hf-semantics and
  the existing gates. Blocks B (bitnet.cpp baseline + clean e2e) and C (ternary-KV round 2) have no detailed spec in
  the text; they follow last night's block E spec and the conclusions of kv_ptq.md.
- **D2 (01:00)** Clean-machine rule for speed: at start load average 4.3 and firefox/neko/zen running -> no speed
  runs now; correctness work first, the condition is re-checked before each speed series.
- **D3** scripts/g6_check.sh default model dir switched to models/bitnet-1.58b-v2 (config with rope_theta; same weights).
- **D4** I2_S kernel: weights re-packed at load from the TL1 buffers (exact re-encoding), TL1 buffers kept for the
  `lut` fallback and the GPU/RVV paths -> +~600 MB resident (not freed tonight). TENZO_GEMV=i2s (default) | lut; any
  other value makes tenzo_create_engine fail.
- **D5 (01:12-01:37) Clean-machine rule failed 3/3** (`logs/clean_machine_checks.log`): firefox/zen/neko (16 processes)
  stay open while the owner sleeps; load average 4-13. Per the rule all speed series are skipped (I2_S A/B, clean
  e2e vs bitnet.cpp). One extra check is made at the end of the night; if it passes, the A/B is run then.
- **D6 (01:15) bitnet.cpp baseline**: microsoft/BitNet@c17d1c5 (2025-04-23, right after the 2B-4T release) +
  Eddie-Wang1120/llama.cpp@5eb47b7, clang 21 in the dev container, setup_env.py codegen recipe. First approach
  worked (greedy == HF, PPL 19.059 EN / 141.8 UK, within CI of golden), so the second approach (upstream HEAD) was
  not needed. tools/reference/bitnet_ref_dump.cpp gained a TENZO_LEGACY_LLAMA_API shim (old libllama names).
