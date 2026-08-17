# Tenzo Project Roadmap & Release Milestones

## Release Versions

### 🚀 v0.3.0-alpha (Current Release)
- [x] **Universal Quantization Matrix:** Native MLIR operations, lowerings, and micro-kernels for TL1 (1.58b), INT4, INT3, GGUF (Q4_0, Q8_0), AutoGPTQ, AutoAWQ, and ExLlamaV2 (EXL2).
- [x] **Frontend Exporter Suite:** Standalone Python exporters generating MLIR and binary weights for all formats.
- [x] **Fused Zero-Allocation KV Cache:** `popcount_fused`, `tl1_fused`, `int8_fused` caches with 15x memory footprint reduction.
- [x] **C/C++ Standalone Engine & SDK:** Pure C++ API (`libtenzo_runtime.a` / `libtenzo_runtime.so`).
- [x] **Tenzo Production CLI:** Model hub, model pulling, stats inspection, memory telemetry, and graceful interrupt handling.
- [x] **Repository Health Standards:** Complete GitHub community profile, security policy, and contributing guidelines.

---

### 🌟 v0.4.0 (Target: Mobile & ARM Optimization)
- [ ] **Native ARM NEON / SVE2 / I8MM Micro-Kernels:** `sdot`, `udot`, and `smmla` kernel paths for Snapdragon and MediaTek Dimensity processors.
- [ ] **Android Termux One-Click Build Package:** Prebuilt aarch64 binaries and installation script.
- [ ] **PagedAttention Memory Manager:** Virtual block table allocation for zero-fragmentation 32k context.
- [ ] **Speculative Verification Engine:** Draft-target verification using 1-layer ternary draft models.

---

### 🔮 v0.5.0 (Target: Vulkan GPU Compute & Continuous Serving)
- [ ] **Vulkan 1.3 Mobile GPU Pipeline:** Subgroup cooperative matrix shaders for Adreno and Immortalis GPUs.
- [ ] **FlashDecoding Kernel Pass:** Parallelized sequence reduction across GPU workgroups.
- [ ] **Continuous Batching & OpenAI-Compatible REST Server:** Asynchronous streaming endpoint in C++.
