# Tenzo Project Roadmap & Release Milestones

## Release Versions

### 🚀 v1.0.0-beta (Universal Low-Bit Compiler & Engine)
- [x] **Universal Quantization Matrix:** Native MLIR operations, lowerings, and micro-kernels for TL1 (1.58b), INT4, INT3, GGUF (Q4_0, Q8_0), AutoGPTQ, AutoAWQ, and ExLlamaV2 (EXL2).
- [x] **Frontend Exporter Suite:** Standalone Python exporters generating MLIR and binary weights for all formats.
- [x] **Fused Zero-Allocation KV Cache:** `popcount_fused`, `tl1_fused`, `int8_fused` caches with 14.2x memory footprint reduction.
- [x] **C/C++ Standalone Engine & SDK:** Pure C ABI (`libtenzo_runtime.a` / `libtenzo_runtime.so` / `include/tenzo.h`).
- [x] **Tenzo Production CLI:** Model hub, model pulling, stats inspection, memory telemetry, and graceful interrupt handling.
- [x] **Repository Health Standards:** Complete GitHub community profile, security policy, and contributing guidelines.

---

### ⚡ v1.1.0-beta (Heterogeneous Multi-Backend & Cross-Device Engine)
- [x] **Vulkan 1.3 GPU Compute Runtime:** Pure C++ SPIR-V compute shader dispatch for BitLinear TL1, GEMM FP32, and RMSNorm.
- [x] **CUDA & ROCm Translation Layers:** Dynamic driver probing with automatic fallback to Vulkan compute.
- [x] **MicroarchProfiler:** Runtime CPU microarchitecture detection, cache hierarchy discovery, SIMD capability probing, and dynamic GEMV tile auto-tuning.
- [x] **RISC-V RVV 1.0 Vector Extension Backend:** Vectorized micro-kernels with scalar fallbacks.
- [x] **Verified Mobile ARM64 Execution:** Tested and operational on Samsung Galaxy Tab S11 (MediaTek Dimensity 9400+, **23.83 tok/sec**) and Android Termux.
- [x] **Multi-Precision Hugging Face Exporter:** Export models to 1.58b, FP32, FP16, INT8, INT4, and INT3.
- [x] **Comprehensive Hardware Diagnostic Suite:** `tenzo-diag` hardware and SIMD verification binary.
- [x] **PagedAttention & Virtual Memory Management:** Block-based KV cache allocation (`BLOCK_SIZE = 16`).

---

### 🌟 v1.2.0 (Target: Speculative Decoding & Continuous Serving)
- [ ] **Speculative Verification Engine:** Draft-target verification using 1-layer ternary draft models.
- [ ] **FlashDecoding GPU Pipeline:** Parallelized sequence reduction across Vulkan compute workgroups.
- [ ] **Continuous Batching & OpenAI-Compatible REST Server:** Asynchronous streaming HTTP endpoint in C++.
- [ ] **ARM SVE2 / I8MM Direct Assembly Micro-Kernels:** Handwritten assembly kernels for Cortex-X925 and Dimensity 9400+.
