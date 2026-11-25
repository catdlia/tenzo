# Changelog

All notable changes to Tenzo Compiler & Inference Engine are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — 2026-11-24-b

### Added
- **`--kv-alpha auto` & `TENZO_KV_ALPHA=auto` resolution (`src/main_inference.cpp`) + `scripts/install_kv_alpha_mix.sh` (`L2`, `D1`).**
  Added installer script copying `alpha_mix.txt` (`sha256: d67ed67b...043eeb`) into `models/bitnet-1.58b-v2/kv_alpha_mix.txt` with checksum checks; `--kv-alpha auto` / `TENZO_KV_ALPHA=auto` resolves `<model_dir>/kv_alpha_mix.txt` with fallback warning if absent, preserving default uncalibrated `ternary_rot` behavior when omitted. Verified `JOBS=1` Docker build (`0` warnings), `tenzo-cli test`, `G6 5/5`, `PPL4` byte-identity (`46489eb1...ab6eb5`), and `--kv-alpha auto` exact score parity (`ppl=17.4338` vs `18.3939` default).
- **Equal-character recent-window causality test (`H1`, `T47`, `results/2026-11-24-b_agent2/H1/`).**
  Tested `S3+s4r<r>` token sweeps and equal-character `S3+s4c<C>` masks on `FLORES-200` `N=32` BitNet (`0.000e+00` B=1 parity). Equalizing character span at `C*=120` (`117.2` EN vs `118.5` UK chars; `r_eff` `22.7/37.5`) lowers UK-minus-EN `R` from `+1.26%` to `+0.98% [+0.55%, +1.40%]` (`~22%` reduction -> **`PARTIAL`**, triggering `H3b`), leaving `78%` of `R > 0`.
- **Exploratory morphological / subword position breakdown (`H2`, `T47b`, `results/2026-11-24-b_agent2/H2/`).**
  Aligned FLORES `N=32` target offsets to alphabetic word positions on BitNet and EuroLLM-1.7B: `word_suffix` shows no excess per-token penalty over `word_prefix` or `single_token_word` (`uk_suffix_minus_prefix`: `+0.0072` BitNet, `+0.0013` EuroLLM), while split words carry `70.9%` / `55.5%` of `D` due to UK's `73.3%` / `68.6%` multi-token share.
- **5-language parallel `FLORES-200` evaluation (`H3`, `T48`, `results/2026-11-24-b_agent2/H3/`).**
  Packed `N=48` out-of-sample parallel windows (`sents 209..633`, `0` 32-gram overlap) across `en, de, pl, uk, bg`: `S3+s4r32` loss correlates strongly with token multiplier across Latin and Cyrillic scripts (`BitNet ρ=1.000, r=0.971` where Latin Polish `×1.88` loses `+1.59%` vs Cyrillic Ukrainian `×1.64` `+1.57%`; `EuroLLM ρ=0.900, r=0.857`) -> pre-registered verdict **`fertility_dominates`**.
- **Coherent single-document parallel `NTREX-128` at `ctx=1024` (`H4`, `T49`, `results/2026-11-24-b_agent2/H4/`).**
  Evaluated `N=24` single news articles (`DOCUMENT_IDS.tsv`, `0` splicing, `0` 32-gram overlap): replicates EN/UK gap under `S3+s4r32` (`BitNet R=+1.51 pp`, `EuroLLM R=+1.38 pp`), recovers `45.0%` / `61.4%` with `S3+s4r128`, and shows **`flat_or_bounded`** positional `Δ/tok` across `129..1023`.
- **Conditional 5-language equal-character window (`H3b`, `results/2026-11-24-b_agent2/H3b/`).**
  Triggered by `H1=PARTIAL`: `S3+s4c120` reduces mean non-EN `R_L` across `{de, pl, uk, bg}` by `22.0%` on BitNet (`4/4` significant) and `34.7%` on EuroLLM-1.7B (`uk` `-71.1%` to `R=+0.16 [-0.10, +0.39]`) -> **`partially_explains`**.

### Fixed
- **E-core `/proc/stat` idle gating & `taskset -c 0-3` pinning in `scripts/b1_idle_runner.sh` (`B0`, `T38`).**
  Replaced broken whole-system `load1` invalidation (which falsely rejected clean 4-thread P-core benchmarks) with `cpu4..7` `/proc/stat` sampling every `2s` (`foreign_cpu_mean < 0.15` + process checks + 3 retries/item), verified in `--smoke` (`foreign_cpu_mean=0.1195`, `invalid_proc` caught), and restarted live `tenzo-b1` into `results/2026-11-24-b_B1/`.

---

## [Unreleased] — 2026-11-24

### Added
- **KV alpha provenance (`--kv-alpha <path>` & `# kv_alpha=<path> sha256=<hex>` in `src/main_inference.cpp`).**
  Added FIPS 180-4 `sha256_file_hex()` logging into score-file headers whenever `TENZO_KV_ALPHA` or `--kv-alpha` is set (`d67ed67b...` for `alpha_mix.txt`), preserving bit-identical output against `TENZO_KV_ALPHA` and unchanged `ternary_rot` default behavior (`D2`).
- **Parallel EN/UK evaluation on FLORES-200 (`H3`, `results/2026-11-24_agent2/H3/`).**
  Built `N=32` sentence-matched aligned pairs (`flores200_dataset.tar.gz`, `sha256: b8b0b767...`, `0` shared 32-grams vs all calib/eval sets) and evaluated BitNet-b1.58-2B, EuroLLM-1.7B, and Qwen2.5-1.5B with exact `0.000e+00` layer-major unpadded parity vs `B=1`. Under `S3+s4r32`, all three models show super-proportional UK degradation at matched semantic content (`95% CI(D) > 0`, `95% CI(R) > 0`), resolving the `ΔBPC`/`ΔBPB` normalization discrepancy (`T45`).
- **Controlled `ctx = 2048` document-length vs concatenation study (`H1b`, `T44`, `results/2026-11-24_agent2/H1b/`).**
  Tested `4 × 12` `ctx=2048` windows (`en_long`, `en_concat`, `uk_long`, `uk_concat`): single contiguous UK Wikipedia articles (`UK-long`) show no positional increase (`full pos/1000 = -0.0488 [-0.1187, +0.0610]`, `Δ(B4−B1) = -0.0681`), whereas multi-article `EN-concat` rises significantly (`+0.0458 [+0.0073, +0.0997]`), attributing positional drift to concatenation while confirming the overall `ctx=2048` UK gap (`+14.7%` vs `+7.4%`).
- **Cross-domain transfer check for `alpha_mix.txt` (`H2`, `results/2026-11-24_agent2/H2/`).**
  `alpha_mix.txt` improves `ternary_rot` strictly (`95% CI(gain) > 1.0`) across all four `2×2` cells in `TenzoEngine`, including out-of-domain `en_web` (`×1.0113 [1.0052, 1.0179]`) and `uk_wiki` (`×1.0217 [1.0121, 1.0308]`).
- **Mechanism ablations & probes on `FLORES-12` (`H4`, `T46`, `results/2026-11-24_agent2/H4/`).**
  Showed ternary Value quantization drives `90.3%` of UK penalty `D` (`K8V:S3+s4r32` `+19.865` vs `KV8` `+8.571`), concentrated in early (`L0..9` `47.2%`) and late (`L20..29` `35.9%`) layers (`L10..19` `4.3%`) and `82.6%` mitigated at `r=128`, while single-pass probes rule out intrinsic `K/V` S3 error differences, higher `fp32` far-attention mass, or `S3-K` tail leakage.
- **Parity gate `v2` diagnostics & paper claims registry (`L2` `parity_v2_diag.md`, `L3` `results/paper/claims.md`).**
  Confirmed `TENZO_EXPERIMENT_FP32_LM=1` passes `v2 1-cos`, `max_abs`, and `top-1` `7/7` for both `fp32` and `int8_fused` KV, attributing the two `6/7` cosine misses to intentional INT8 embedding/LM-head quantization.
- **Self-gated idle benchmark runner (`scripts/b1_idle_runner.sh`, `B1`, `T38`).**
  Added AC-power + 10-minute idle (`load1 < 1.0`) + process-gating + `systemd-inhibit` runner for `pending_benchmarks.sh` (`bench_decode_modes.sh` updated to include `ternary_rot_alpha`); validated in `--dry-run` (`results/2026-11-24_agent2/B1_dryrun.log`).

### Changed / Hygiene
- Removed `tools/tokenizer/bitnet_b158_tokenizer.json` (`9 MB`) from git tracking (`git rm --cached`, `.gitignore`), added `tools/tokenizer/README.md`, copied HF cache `tokenizer.json` to `models/bitnet-1.58b-v2/tokenizer.json`, updated tokenizer fallback order, and documented dataset licenses (`kobza` `CC0-1.0` packaging + notice/takedown, Wikipedia `CC BY-SA 3.0`/`GFDL`, FineWeb-Edu `ODC-By`) in `results/2026-11-23_agent2/H2/corpora/LICENSES.md` (`T43`).

---

## [Unreleased] — 2026-11-23

### Added
- **Native Llama-3 byte-level BPE tokenizer (`src/runtime/Tokenizer.{h,cpp}`, `src/runtime/UnicodeTables.h`).**
  Dependency-free C++17 HuggingFace `tokenizer.json` loader supporting `added_tokens`, 7-branch regex-free Llama-3 pre-tokenizer over generated Unicode property tables (`tools/tokenizer/gen_unicode_tables.py`), 280,147 BPE merges (`ignore_merges=True`), and inverse GPT-2 byte-to-UTF-8 decoding; auto-upgrades vocabs `>= 128k`. Integrated into `tenzo-inference` (`--tokenizer`, `--tokenize-batch`) and `tenzo-cli test` (`runTokenizerTests`). Verified `3200/3200` exact ID & round-trip parity vs HF (`scripts/test_tokenizer_parity.py`) and `7/7 identical=True` via `verify_model_parity.py --only-tokenization`.
- **Parity gate v2 (`scripts/verify_model_parity.py --gate v2`).**
  Added prompt-normalized logit/NLL parity checks (`max_abs <= 1.5 * max_abs_ref`, `1 - cos <= 1.5 * (1 - cos_ref)`, `top-1 >= 6/7`, EN16 PPL CI containment) while leaving the `v1` gate unchanged (`results/2026-11-23_agent2/L2/parity_v3.md`).
- **EN+UK mixed KV calibration (`results/2026-11-23_agent2/H1/`).**
  Calibrated per-layer `ternary_rot` clip multipliers (`alpha_mix.txt`) on `calib_en8 + calib_uk8`; wired into `TenzoEngine` via `TENZO_KV_ALPHA`. Improved `ternary_rot` PPL ratio vs `int8_fused` from `×1.0711` to `×1.0365` (`19.839`) on EN16 and from `×1.1648` (`UK8`) to `×1.1263` (`×1.1162` / `132.327` on UK32), while `eng/sim` 95% CIs contain `1.0`.
- **Cross-family `2×2` disparity & long-context (`ctx = 2048`) evaluation suites (`results/2026-11-23_agent2/{L3,H2,H3}/`).**
  - `L3`: Confirmed byte/character normalization (`c_UK > 0` in `BPC` and `BPB`) across BitNet ternary KV and `Qwen2.5-1.5B` `BPB` (`scripts/analyze_disparity_bpc.py`).
  - `H2`: Evaluated `2×2` (`EN/UK × Wiki/Web`, 16×512 windows each) across `BitNet-b1.58-2B-4T`, `Qwen2.5-1.5B`, and `utter-project/EuroLLM-1.7B`; in `EuroLLM-1.7B` (`uk_wiki` base BPC `0.802 ≈ en_web 0.803`), `S3+s4r32` and `S13+s4r32` retain a significant UK penalty after adjusting for `Web` and `base_BPC`.
  - `H3`: Evaluated `ctx = 2048` (4 EN + 4 UK windows) in golden and `TenzoEngine`; engine peak RSS stayed `< 1 GB` (`950.7 MB` `int8_fused`, `902.8 MB` `ternary_rot`). Engine `ternary_rot` degradation stays bounded across context blocks on EN (`×1.067..×1.066`, `×1.0381` full w/ `alpha_mix`) but increases monotonically with context position on UK (`×1.144 → ×1.165 → ×1.243 → ×1.305`; `×1.117 → ×1.251` w/ `alpha_mix`).

### Fixed
- **Attention `q_dim` vs `hidden_size` mismatch (`src/runtime/TenzoEngine.cpp`, `src/bindings/PybindModule.cpp`).**
  Fixed `attn_sub_norm` input size and `out_proj` weight/GEMV dimension from `hidden_size` to `q_dim` across CPU, GPU, CUDA, ROCm, RVV paths, `PybindModule`, and helpers (`gemv_i2s_rows`, `gemm_i2s_rows`); added `runQDimNeHiddenTests()` (`rel_l2 = 9.91e-8` vs `0.493` pre-fix on `q_dim=256, hidden=128`) and eliminated compiler warnings (`JOBS=1` Docker build, `0` warnings).

---

## [Unreleased] — 2026-11-19/20

### Fixed — semantic fixes: **outputs change; previous quality numbers invalid**
- **RoPE base.** The runtime engine (`FusedKVCache`, `PagedKVCache`) hard-coded 10000. It now uses the model's
  `rope_theta` (500000 for BitNet-b1.58-2B-4T) via `tenzo_config_t.rope_theta`, which is read from the model's
  `config.json` by `tenzo-inference`. The Python bindings take `rope_theta`. The MLIR op `tenzo.rope` now requires a
  `theta` attribute: lowering fails without it.
- **Activation quantization.** The BitLinear activations now follow HF `ActQuant`: per-token
  `127 / max(|x|, 1e-5)`, round half to even, clamp [-128, 127]. Before, they used ±31 levels (engine, Python
  bindings, `bitlinear_tl1_pack` lowering) or 63 levels with truncation (`bitlinear_tl1` lowering).
- **Effect.** WikiText-2 PPL with fp32 KV went from 23.13 to 19.12 against a golden fp32 of 19.07. All PPL, parity
  and generation numbers measured before 2026-11-19 are invalid.

### Added (2026-11-20)
- `ternary_rot` KV mode (`--kv-quant ternary_rot`): Hadamard-rotated ternary g32 K/V with INT8 sinks (4) and recent window (32); 14,400 B/token
  vs 39,600 for `int8_fused`; quality equals the simulator's S3+s4r32 (WikiText-2 PPL ratio to the fp32 reference 1.075, Kobza-UK 1.167).
  Optional calibrated clip multipliers via `TENZO_KV_ALPHA`.
- Batched prefill (`--prefill batched`, `tenzo_prefill_tokens`): CPU + I2_S, bit-identical to token-by-token prefill.
- `tenzo_create_engine` rejects unknown `kv_mode` values (`tenzo-cli engine-api` test); the standalone runtime build
  (`-DTENZO_STANDALONE_RUNTIME=ON`) builds again; `TENZO_DUMP_JIT_ASM` dumps all audit kernels (it dumped only one).
- `TENZO_SMAPS_DIR` (smaps dumps), `scripts/analyze_smaps.py`.

### Changed (2026-11-20)
- Peak RSS at load: weights.bin is mmapped and released progressively, the I2_S layout is built per layer, the unused paged KV pool is no longer
  allocated: `int8_fused`, ctx 2048: VmHWM 2761 -> 935 MB. Outputs unchanged (G6 and per-window PPL identical).
- `rms_norm_raw` is `noinline` so that the token-by-token and batched paths execute the same code (results of the existing paths unchanged).

### Added (2026-11-19)
- I2_S-style maddubs GEMV (`TENZO_GEMV=i2s`, default; `lut` fallback), bit-identical to the LUT kernel.
- `tenzo-cli hf-semantics` regression tests (compiler and engine) and `scripts/test_pybind_hf_semantics.py`.
- Opt-in intermediate tensor dumps (`TENZO_DUMP_DIR`, `TENZO_DUMP_POS`, `TENZO_DUMP_LAYERS`).
- `tools/reference/golden_bitnet.py`: an independent layer-major fp32 reference.

---

## [v1.1.0-beta] — 2026-08-18

### Highlights

**Heterogeneous Multi-Backend GPU Runtime** — Tenzo now ships a complete Vulkan compute pipeline with SPIR-V shaders for BitLinear TL1, GEMM FP32, and RMSNorm, plus CUDA and ROCm translation layers that fall back to Vulkan when native drivers are absent. A hardware-aware `MicroarchProfiler` auto-detects CPU microarchitecture, SIMD capabilities, cache hierarchy, and GPU availability at runtime.

**Verified Cross-Architecture Deployment** — The engine has been built and tested on x86_64 (Intel AVX2/VNNI, Docker), ARM64 AArch64 (Samsung Galaxy Tab S11 / MediaTek Dimensity 9400+ via Termux), and Android Snapdragon (Poco X5 Pro 5G / Snapdragon 778G via Termux). Inference speed on the Tablet reached **23.83 tok/sec** (ARM NEON DotProd, 8 cores).

**Multi-Precision HuggingFace Exporter** — A new Python exporter supports exporting models in 1.58-bit (TL1), FP32, FP16, INT8, INT4, and INT3 formats from any HuggingFace-compatible checkpoint.

---

### Added

#### Heterogeneous GPU Backends
- **Vulkan Compute Runtime** (`VulkanRuntime`): Full Vulkan 1.3 pipeline with instance/device creation, compute queue dispatch, push constants, and automatic memory management. Includes pre-compiled SPIR-V shaders for:
  - `bitlinear_tl1.comp` — Ternary 1.58-bit GPU matrix-vector multiply
  - `gemm_f32.comp` — Dense FP32 GPU GEMV
  - `rmsnorm.comp` — Root Mean Square Normalization on GPU
- **CUDA Translation Layer** (`CUDARuntime`): Detects native CUDA drivers (`libcuda.so`); when absent, transparently falls back to Vulkan compute via the translation layer.
- **ROCm Translation Layer** (`ROCmRuntime`): Detects native AMD ROCm/HIP drivers (`libamdhip64.so`); when absent, transparently falls back to Vulkan compute.
- **RISC-V RVV Kernels** (`src/runtime/arch/RISCV_RVV.h`): Scalar-fallback implementations for `gemv_bitlinear_tl1_rvv`, `rmsnorm_rvv`, and `gemv_f32_rvv` targeting future RISC-V Vector Extension hardware.

#### Runtime & Profiling
- **MicroarchProfiler** (`MicroarchProfiler.h/.cpp`): Detects CPU vendor, architecture, core/thread counts, SIMD capabilities (AVX2/AVX-512/NEON/SVE2/DotProd/I8MM/RVV), L1/L2/L3 cache sizes, GPU name, and auto-calibrates optimal tiling parameters (`tile_M`, `tile_N`, `tile_K`) for the detected microarchitecture.
- **BpeTokenizer::decode()**: Added `decode(const std::vector<int>&)` method to the inline BPE tokenizer for streaming token-to-text conversion during inference.
- **TenzoEngine** enhancements: Integrated `MicroarchProfiler` for hardware-aware kernel dispatch. Engine startup banner now reports detected architecture, SIMD level, and backend.

#### Multi-Precision Model Export
- `tenzo-frontend/export_bitnet.py`: Complete multi-precision HuggingFace model exporter supporting `1.58b` (TL1), `fp32`, `fp16`, `int8`, `int4`, and `int3` quantization formats with vectorized BF16-to-FP32 conversion.

#### Cross-Device Deployment Tooling
- `scripts/sync_to_target.py`: Efficient source tree packer and deployer for remote devices (phone/tablet) via tmate SSH tunnels. Packs only essential source files (~220 KB compressed).
- `scripts/device_runner.py`: Remote command executor for Termux devices with base64-encoded payload transport and automatic session management.
- `scripts/compile_shaders.py`: SPIR-V shader compilation script using `glslangValidator`.

#### Diagnostic Suite
- **DeviceDiagnosticTest** (`tenzo-diag`): Comprehensive post-compilation hardware diagnostic binary testing:
  - SIMD intrinsic correctness (AVX2 shuffle, shift, convert, madd, FMA)
  - Heterogeneous backend initialization (Vulkan, CUDA, ROCm translation layers)
  - RISC-V RVV kernel mathematical correctness
  - MicroarchProfiler telemetry validation
  - Model weight integrity and vocabulary verification
  - Single-token generation validation

#### Build System
- **CMakeLists.txt**: Added `TENZO_STANDALONE_RUNTIME` option for building without LLVM/MLIR. Added architecture-specific compiler flags for ARM64 (`-O3 -funroll-loops -fopenmp`) and x86_64 (`-mavx2 -mavxvnni -mfma -mf16c`). Android Vulkan fallback detection via `find_library(vulkan /system/lib64)`.
- **Makefile**: Added `build-sdk` target for fast standalone SDK compilation. Added `diag` target for running hardware diagnostics. Added `inference` target for CLI inference.

---

### Changed
- **VulkanRuntime**: Refactored from singleton instance pattern to static API (`VulkanRuntime::initialize()`, `VulkanRuntime::executeShader()`, etc.) for cleaner cross-backend usage.
- **CUDARuntime / ROCmRuntime**: Updated to use `runtime::VulkanRuntime` static methods with proper `uint32_t` casts instead of `size_t` for Vulkan API compatibility.
- **DeviceDiagnosticTest**: Moved `#include` directives for backend headers from mid-file (inside anonymous namespace) to file scope for proper linkage.
- **README.md**: Version badge updated to reflect current release.
- **tenzo_cli.py**: Updated startup banner, added backend device selection (`/device cpu|gpu|vulkan|cuda|rocm`).
- **include/tenzo.h**: Version bumped to `1.1.0-beta`.

---

### Fixed
- **Namespace resolution**: Fixed `CUDARuntime.cpp` and `ROCmRuntime.cpp` to correctly reference `runtime::VulkanRuntime` static methods instead of non-existent `VulkanRuntime::getInstance()`.
- **Vulkan header guard**: Added `check_include_file_cxx("vulkan/vulkan.h")` before enabling `TENZO_HAS_VULKAN` to prevent build failures on systems without Vulkan SDK headers.
- **ARM NEON compatibility**: Fixed `simd_arm_compat.h` to provide complete 128-bit and 256-bit SIMD intrinsic mappings from AVX2 to NEON, preventing compilation errors on AArch64 Termux.
- **Model path resolution**: Fixed dynamic model path resolution with auto-fallback on mobile/Termux environments.
- **Weight export**: Fixed row-packed `[N/4, K]` layout in `pack_ternary_weights_raw` and ensured exact per-layer scale export.

---

### Tested On

| Device | Architecture | SIMD | Inference Speed | Status |
|:-------|:-------------|:-----|:----------------|:-------|
| Intel Core i3-1215U (Docker) | x86_64 | AVX2 / VNNI / FMA | 14.01 tok/sec | ✅ PASS |
| Samsung Galaxy Tab S11 (Termux) | ARM64 / ARMv9.2 | NEON / DotProd | 23.83 tok/sec | ✅ PASS |
| Poco X5 Pro 5G (Termux) | ARM64 / ARMv8.2 | NEON / DotProd | — | 🔌 Disconnected |

---

## [v1.0.0-beta] — 2026-08-17

### Highlights
First public beta release of the Tenzo AI Compiler & Inference Engine.

### Added
- Universal quantization support: BitNet TL1 (1.58-bit), GGUF (Q4_0/Q8_0), AutoGPTQ, AutoAWQ, ExLlamaV2 (EXL2), INT4, INT3.
- Zero-allocation C++ runtime with fused INT8/TL1 KV-Cache (314 MB for 8K context).
- **20.32 tok/sec** on Intel Core i3-1215U — **1.65x faster** than Microsoft BitNet.cpp.
- MLIR compiler pipeline: Tenzo dialect → Linalg → Vectorization → LLVM/SPIR-V code generation.
- C/C++ SDK with static and shared libraries (`libtenzo_runtime.a`, `libtenzo_runtime.so`).
- Interactive CLI console (`tenzo_cli.py`) with model management, hot-swap, live telemetry.
- Python frontend exporters for all supported quantization formats.
- ARM NEON / SVE2 / I8MM compatibility layer (`simd_arm_compat.h`).
- Android Termux and NDK build scripts.
- Community standards: LICENSE (Apache 2.0), CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md.
- Documentation: ARCHITECTURE.md, QUANTIZATION.md, MOBILE_ARM_VULKAN.md, ROADMAP.md, VERTICAL_SCALING_RESEARCH.md.

[v1.1.0-beta]: https://github.com/catdlia/tenzo/compare/v1.0.0-beta...v1.1.0-beta
[v1.0.0-beta]: https://github.com/catdlia/tenzo/releases/tag/v1.0.0-beta
