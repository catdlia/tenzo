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
- **D7 (02:27) Part 2 starts.** Quiet mode ON: steam/steamwebhelper appeared (12 processes) at 02:27 while firefox/neko
  closed -> the owner may be back. Quiet-mode limits applied: nice -n 19 ionice -c3 for heavy work, Python MemoryMax=3G,
  builds -j1, nothing > 4 GB, speed runs only queued (scripts/pending_benchmarks.sh).
- **D8** Only 8 UK eval windows exist (39 Kobza first-rows documents). Part 2 asks for 32 UK eval windows plus 8
  disjoint UK calibration windows: more Kobza text is fetched from the datasets-server rows API (text, < 50 MB).
- **D9 (part 2, 11:18) Time accounting.** Part 2 was active 02:26-03:15 (49 min); then the session sat idle until the owner
  typed "Продовжуй" at ~11:18 (about 8 h of wall clock). The idle gap is NOT counted against the 8 h active budget
  (active so far: ~1 h 10 min at 11:25). Quiet mode stays ON: zen-bin and steam are running (12 processes) and the owner
  is at the machine; heavy jobs run niced, Python MemoryMax=3G, builds -j1, speed runs only queued.
- **D10 (11:22) Calibration pipeline** (`scripts/run_kv_calib.sh`): steps=100 (limit 300), minibatch 2 windows, Adam lr 0.02,
  alphas clamped to [0.5, 1.5] per layer x KV head x 32-group, K and V separately. The ternary codes and group scales are
  recomputed from the data at every forward (constants for autograd), so gradients reach alpha only (straight-through
  as in the spec); rounding-free ternary thresholds are not learned. Cayley rotation (optional) is only attempted if time allows.
- **D11** For `ternary_rot` the ternary encoding of a token is computed at APPEND time from the exact fp32 k/v (so it matches the
  simulator, which quantizes the original vectors); the INT8 copy lives in the 32-token ring and 4 sink slots. The spec's
  "token leaving the window is quantized to ternary" cannot be done from the INT8 copy without changing the numbers.
