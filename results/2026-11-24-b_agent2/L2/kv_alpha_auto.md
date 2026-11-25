# L2 (`D1`): `--kv-alpha auto` & `scripts/install_kv_alpha_mix.sh`

## Changes Implemented

1. **Installer script (`scripts/install_kv_alpha_mix.sh`):**
   - Verifies source `results/2026-11-23_agent2/H1/alpha_mix.txt` against expected `sha256 = d67ed67b412ab136c63a75cc7239769726b48717ef8b48fd8b5f4e00ca043eeb`.
   - Copies it to `models/bitnet-1.58b-v2/kv_alpha_mix.txt` and re-verifies the installed `sha256`.
   - Executed cleanly: `Installed /home/illia/CLionProjects/untitled/models/bitnet-1.58b-v2/kv_alpha_mix.txt (sha256=d67ed67b412ab136c63a75cc7239769726b48717ef8b48fd8b5f4e00ca043eeb)`.

2. **Runtime CLI & Env Resolution (`src/main_inference.cpp`):**
   - Added `--kv-alpha auto` (and `TENZO_KV_ALPHA=auto` when `--kv-alpha` is omitted) resolving `<model_dir>/kv_alpha_mix.txt` after `opt.model_dir` auto-selection.
   - If readable, sets `TENZO_KV_ALPHA` to the resolved path so both `tenzo_create_engine` (`FusedKVCache`) and `--score-out` (`# kv_alpha=<path> sha256=<hex>`) pick it up automatically.
   - If missing, emits `[warn] --kv-alpha auto: <candidate> not found; using default scale` to `std::cerr` and unsets `TENZO_KV_ALPHA`.
   - Default behavior without `--kv-alpha` or `TENZO_KV_ALPHA` remains **100% unchanged** (`D1`: raw `--kv-quant ternary_rot` uncalibrated by default).

## Verification Results

- **Docker build (`make build-local JOBS=1`):** Compiled & linked `tenzo-inference` with **0 warnings, 0 errors**.
- **`tenzo-cli test` (`docker compose run --rm dev /app/cmake-build-debug/tenzo-cli test`):** All tests passed (`--- All Tests Complete ---`, including CPU MatMul, Audit JIT, HF semantics, Engine API, `ternary_rot`, batched prefill bit-identity, `q_dim != hidden_size`, and Llama-3 BPE tokenizer).
- **`G6` determinism check (`scripts/g6_check.sh results/2026-11-24-b_agent2/L2/g6`):** `5/5` KV modes byte-identical (`diff -r` against `results/2026-11-24-b_agent2/L0/g6` returned `G6_DIFF_OK`: `fp32: 37b6e02003fd`, `int8_fused: 2dfd79933b0e`, `tl1_fused: e9e7e2edf465`, `popcount_fused: dd40633550d9`, `ternary_rot: 2dfd79933b0e`).
- **`PPL4` regression (`tenzo-inference --kv-quant int8_fused --ctx-size 512 --prefill batched` on `windows4.ids`):**
  - `ppl=16.7007`, `nll=5754.78110`, `PROC_HWM_KB=913560`.
  - Byte-identical (`sha256 = 46489eb141d47e0c1610e71a4133f3ee2681cd62728fec532cdf50b5a1ab6eb5`) to `results/2026-11-24-b_agent2/L0/ppl4_int8_fused.txt`.
- **`--kv-quant ternary_rot` (`windows4.ids`, `batched`):**
  - Default (no `--kv-alpha`): `ppl=18.3939`, `nll=5952.16462` (no `# kv_alpha=` header, unchanged).
  - Explicit `--kv-alpha /app/results/2026-11-23_agent2/H1/alpha_mix.txt`: `ppl=17.4338`, `nll=5842.59428`, `# kv_alpha=/app/results/2026-11-23_agent2/H1/alpha_mix.txt sha256=d67ed67b412ab136c63a75cc7239769726b48717ef8b48fd8b5f4e00ca043eeb`.
  - `--kv-alpha auto`: **exact `diff -u` parity** on all `w`/`lp` lines (`ppl=17.4338`, `nll=5842.59428`), header `# kv_alpha=/app/models/bitnet-1.58b-v2/kv_alpha_mix.txt sha256=d67ed67b412ab136c63a75cc7239769726b48717ef8b48fd8b5f4e00ca043eeb`.
  - `TENZO_KV_ALPHA=auto` (without CLI flag): **byte-identical** (`diff -u` empty) to `--kv-alpha auto`.
- **Missing-file fallback (`--model /app/models/bitnet-1.58b --kv-quant ternary_rot --kv-alpha auto --prompt "The" --max-tokens 1`):** Emits `[warn] --kv-alpha auto: /app/models/bitnet-1.58b/kv_alpha_mix.txt not found; using default scale` and completes inference cleanly.
