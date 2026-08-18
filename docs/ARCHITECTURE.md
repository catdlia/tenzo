# Tenzo AI Compiler Architecture

## Overview

**Tenzo** is a domain-specific, high-performance compiler and inference runtime for low-bit Large Language Models (BitNet 1.58-bit, 3-bit, 4-bit, GGUF, GPTQ, AWQ, EXL2). Built on the **MLIR (Multi-Level Intermediate Representation)** framework and LLVM 21 infrastructure, Tenzo provides ahead-of-time (AOT) and just-in-time (JIT) compilation tailored for extreme latency reduction and minimal memory footprints.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Tenzo Frontend Exporters                        │
│   (PyTorch FX / SafeTensors / GGUF Binaries / AutoGPTQ / AutoAWQ / EXL2)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       Tenzo MLIR Dialect (tenzo)                       │
│  tenzo.bitlinear_tl1 | tenzo.bitlinear_gguf | tenzo.bitlinear_gptq     │
│  tenzo.bitlinear_awq | tenzo.bitlinear_exl2 | tenzo.rmsnorm            │
│  tenzo.attention     | tenzo.rope           | tenzo.kv_update          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼ [Linalg Lowering Pass]
┌────────────────────────────────────────────────────────────────────────┐
│                       Structured Linalg & SCF IR                       │
│   (linalg.generic / linalg.matmul / scf.parallel / scf.for / tensor)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼ [Vectorization & Bufferization]
┌────────────────────────────────────────────────────────────────────────┐
│                        Vector & MemRef Dialects                        │
│   (vector.contract / vector.transfer_read / vector.fma / memref.view)  │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼ [LLVM Lowering]                ▼ [SPIR-V Lowering]
┌───────────────────────────────────────┐ ┌──────────────────────────────┐
│             LLVM Dialect              │ │        SPIR-V Dialect        │
│   (x86_64 AVX2 / ARM NEON / SVE / RVV)│ │  (Vulkan 1.3 Compute Shader) │
└───────────────────┬───────────────────┘ └──────────────┬───────────────┘
                    │                                    │
                    ▼                                    ▼
┌───────────────────────────────────────┐ ┌──────────────────────────────┐
│       Tenzo Multi-Backend CPU HAL     │ │    Tenzo Heterogeneous GPU   │
│   - AVX2 / VNNI Micro-Kernels         │ │  - Native Vulkan 1.3 Compute │
│   - ARM NEON / DotProd / SVE2 / I8MM  │ │  - CUDA Translation Layer    │
│   - RISC-V RVV 1.0 Vector Kernels     │ │  - ROCm Translation Layer    │
│   - MicroarchProfiler (Auto-Tiling)   │ │  - Push Constants Dispatch   │
│   - Fused Zero-Copy KV-Cache          │ │  - Async Compute Pipelines   │
└───────────────────────────────────────┘ └──────────────────────────────┘
```

---

## 1. Dialect Specification (`tenzo`)

The Tenzo dialect encapsulates specialized tensor operators for quantized and sub-byte neural representations:

### Linear Transformations
- `tenzo.bitlinear_tl1`: Microsoft TL1 dense 2-bit ternary mapping $\{-1, 0, 1\}$.
- `tenzo.bitlinear_tl1_pack`: Dual-element nibble packed lookup matmul (64 channels per YMM register).
- `tenzo.bitlinear_elut`: ELUT exponent-separated LUT matmul.
- `tenzo.bitlinear_int4`: 4-bit symmetric linear layer.
- `tenzo.bitlinear_int3`: 3-bit packed linear layer.
- `tenzo.bitlinear_gguf`: GGUF block-quantized matmul (`Q4_0`, `Q8_0`).
- `tenzo.bitlinear_gptq`: AutoGPTQ 4-bit groupwise matmul with zero-point offset.
- `tenzo.bitlinear_awq`: AutoAWQ column-interleaved 4-bit matmul.
- `tenzo.bitlinear_exl2`: ExLlamaV2 variable bitrate group descriptor matmul.

### Normalization & Attention
- `tenzo.rmsnorm`: Root Mean Square Normalization with custom epsilon.
- `tenzo.rope`: Rotary Positional Embeddings with rotary frequency cache.
- `tenzo.attention`: Scaled dot-product multi-head and grouped-query attention (GQA).
- `tenzo.kv_update`: In-place KV-cache token insertion and memory barrier.

---

## 2. Compilation Pipeline

1. **Frontend Lowering:** Python exporters parse weight files and construct high-level SSA `tenzo.*` operations.
2. **Tenzo-to-Linalg Pass:** Rewrites domain-specific ops into structured loops (`linalg.generic`, `scf.parallel`) with loop tiling and vector-friendly access strides.
3. **Bufferization & Memory Planning:** Replaces abstract `tensor` values with explicit `memref` allocations, views, and subviews, eliminating unnecessary allocations.
4. **Vectorization Pass:** Tiles compute loops to vector sizes matching hardware registers (`vector<8xf32>`, `vector<32xi8>`).
5. **Target Code Generation:**
   - **CPU:** Lowered via LLVM dialect to native machine code with `-O3`, vectorizing to AVX2/FMA/VNNI, ARM NEON, or RISC-V RVV instructions.
   - **GPU:** Lowered to SPIR-V dialect, compiled to `.spv` compute shaders executed via Vulkan runtime pipelines.

---

## 3. Runtime Engine (`TenzoEngine`) & HAL

The runtime architecture is designed for zero heap allocation during token generation:
- **Preallocated Scratchpads:** Activations, norms, logits, and attention caches are pre-allocated during engine initialization.
- **Fused KV-Cache:** Supports `popcount_fused`, `tl1_fused` (14.2x compression), `int8_fused` (4x compression), and `fp32` representations.
- **PagedAttention:** Virtual block table allocation (`BLOCK_SIZE = 16`) to eliminate dynamic memory fragmentation during 32k+ context scaling.
- **High-Performance Sampling:** Hardware-accelerated Top-K, Top-P (nucleus), Temperature, and Repetition Penalty filtering.
- **Hardware Autotuning (`MicroarchProfiler`):** Probes CPU L1/L2/L3 cache capacities and dynamically selects optimal $(M, N, K)$ GEMV tiling parameters.
- **Heterogeneous GPU Dispatch:**
  - **Vulkan 1.3:** Direct execution of SPIR-V compute shaders (`bitlinear_tl1.comp`, `gemm_f32.comp`, `rmsnorm.comp`).
  - **CUDA & ROCm Translation Layers:** Dynamic driver detection with seamless Vulkan fallback.
- **C/C++ Foreign Function Interface (FFI):** Clean C ABI exposed via `libtenzo_runtime.so` / `tenzo.h` for integration into Python, Rust, Go, or mobile apps.
