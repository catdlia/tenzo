# 📋 Tenzo Compiler — Project Status (v0.3.0)

> **Last updated:** August 2026

---

## 💡 Vision

Tenzo is a high-performance **heterogeneous compiler** designed specifically for running ultra-efficient, heavily quantized Large Language Models (LLMs) on consumer and edge hardware. 

While existing solutions (TensorRT, CoreML, OpenVINO, llama.cpp) are either vendor-locked or rely on hand-written assembly libraries for every new architecture, **Tenzo takes a compiler-first approach**. By building on top of MLIR/LLVM, Tenzo detects the hardware at compile-time, applies aggressive graph-level fusions, and statically lowers operations to optimally tiled and unrolled vector micro-kernels (e.g. AVX2, NEON) — completely eliminating dynamic memory allocation on the hot path.

Our primary focus is the **1.58-bit (ternary) BitNet architecture**, enabling models like `BitNet-b1.58-2B-4T` to run with minimal memory footprints and blazing fast decode speeds on constrained edge devices like low-power Intel Core i3 CPUs.

---

## 📍 Current State (Honest Assessment)

### ✅ What Works
- **Custom MLIR Dialect for LLMs**:
  - `tenzo.bitlinear_tl1`: Native support for 1.58-bit ternary weights packed into 2-bit formats.
  - `tenzo.matmul_q8`: Native `i8` quantized linear layers (used for `lm_head`).
  - `tenzo.rope`, `tenzo.rms_norm`, `tenzo.attention`: Specialized LLM operations.
- **End-to-End LLM Generation Pipeline**:
  - Full support for 30-layer LLaMA-based BitNet architectures.
  - Generates perfectly coherent text with strict token-to-token accuracy matching PyTorch reference implementations.
- **Micro-Architectural Optimizations (x86 AVX2)**:
  - **AVX2 256-bit PSHUFB Micro-kernels**: Decodes 64 output channels per instruction using duplicated 16-byte LUTs across 128-bit lanes.
  - **16-Bit Intermediate SIMD Accumulation**: Vectorized `_mm256_add_epi16` accumulation with chunked 32-bit promotion every 64 iterations.
  - **Fused Multi-Projection OpenMP Loops**: Fused QKV (60 parallel blocks) and Fused Gate-Up (216 parallel blocks) eliminating OpenMP thread starvation and barrier overhead.
  - **Zero-Allocation ExecutionContext**: Completely eliminated all dynamic `malloc`/`free`/`py::array_t` overhead during autoregressive decoding.
  - **Fused INT8 In-Register KV-Cache Compression**: 4x smaller memory footprint (314 MB for 8192 context length) with online register-level symmetric quantization.
  - **C++ Top-K / Top-P Min-Heap Sampler**: Min-heap sampler completing in 0.01 ms per token.

### ⚠️ What's Missing (Next Milestones)

| Gap | Description | Priority |
|-----|-------------|----------|
| **Multi-target Backends** | Micro-kernels are explicitly tuned for x86 AVX2. Need equivalent paths for ARM NEON/SVE and RISC-V RVV. | 🔴 Critical |
| **GPU/Vulkan Support** | The SPIR-V pipeline is a basic proof-of-concept. It needs to support `tl1_pack` and `q8` tensor layouts. | 🟡 High |
| **Attention + RoPE Fusion** | RoPE and SDPA in MLIR dialect level can be fused into a single unified FlashAttention MLIR pattern. | 🟡 High |
| **Autoregressive Speculation** | Speculative decoding / multi-token drafting to achieve 40+ tok/sec. | 🔵 Future |

---

## 📊 Performance Metrics

**Hardware:** Intel Core i3-1215U (Alder Lake, 15W, 2 P-Cores, 4 E-Cores, 20 GB RAM)  
**Model:** `microsoft/BitNet-b1.58-2B-4T` (30 layers)

| Engine / Configuration | Model Format | KV-Cache | Decode Speed | Latency / Token | Note |
|---|---|---|---|---|---|
| **Microsoft BitNet.cpp** | TL1 + INT8 LM | FP32 | 12.31 tok/sec | 81.23 ms | Official Microsoft C++ Reference Engine |
| **Tenzo Native Engine (v0.3.0)** | **TL1 + INT8 LM** | **INT8 Fused** | **20.32 tok/sec** | **49.21 ms** | **1.65x FASTER than BitNet.cpp 🚀** |

---

## 🗺️ Roadmap

### Phase 1: Edge CPU Dominance (Completed ✅)
- [x] End-to-end BitNet 1.58B generation.
- [x] Zero-Allocation MLIR Bufferization.
- [x] Hand-tuned AVX2 micro-kernels (Zero-Spill).
- [x] Fused OpenMP Loops & 16-Bit SIMD Accumulation.
- [x] Fused in-register quantized INT8 KV-Cache compression.
- [x] Beat Microsoft `BitNet.cpp` reference performance on edge CPUs (**20.32 tok/sec**).

### Phase 2: Kernel Fusion & GPU (Next)
- [ ] **ARM NEON Backend**: Expand auto-vectorizer to target Apple Silicon and Snapdragon.
- [ ] **FlashAttention MLIR Kernel**: Fuse SDPA, RoPE, and causal masking at dialect level.
- [ ] **Vulkan SPIR-V Compute**: Bring 1.58-bit decompression to iGPUs via Vulkan.

### Phase 3: Developer Experience & Deployment
- [ ] **Standalone Inference Engine**: Decouple the JIT execution engine from `tenzo-cli` for embedding in iOS/Android apps.
- [ ] **Multi-Model Support**: Support modern architectures beyond LLaMA (Mistral, MoE).

---

## 📦 Component Inventory

- `src/dialect/` — MLIR definitions (`tenzo.bitlinear_tl1`, `tenzo.rope`).
- `src/passes/` — Compiler passes (Linalg Lowering, Bufferization, Micro-Kernel generation).
- `src/context/` — Hardware detection and topology profiling.
- `src/runtime/` — Execution environment (Tokenizer, KVCacheManager, OpenMP executor).
- `src/bindings/` — Pybind11 native execution engine (`tenzo_runtime`).
- `tenzo-frontend/` — PyTorch-to-MLIR exporter (`export_bitnet.py`).
