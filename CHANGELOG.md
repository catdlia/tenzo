# Changelog

All notable changes to Tenzo Compiler & Inference Engine are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **include/tenzo.h**: Version bumped to `1.0.0-beta.1`.

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
