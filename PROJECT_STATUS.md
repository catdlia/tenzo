# 📋 Tenzo Compiler — Project Status

> **Last updated:** August 2026

---

## 💡 Vision

Tenzo is a high-performance **heterogeneous compiler** designed specifically for running ultra-efficient, heavily quantized Large Language Models (LLMs) on consumer and edge hardware. 

While existing solutions (TensorRT, CoreML, OpenVINO, llama.cpp) are either vendor-locked or rely on hand-written assembly libraries for every new architecture, **Tenzo takes a compiler-first approach**. By building on top of MLIR/LLVM, Tenzo detects the hardware at compile-time, applies aggressive graph-level fusions, and statically lowers operations to optimally tiled and unrolled vector micro-kernels (e.g. AVX2, NEON) — completely eliminating the need for dynamic memory allocation on the hot path.

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
  - **Zero-Spill Unroll x2**: Hand-tuned inner loop unrolling that strictly fits within the 16 YMM registers of AVX2, preventing LLVM from spilling to the stack.
  - **Zero-Allocation Bufferization**: Replaced all dynamic `memref.alloc` calls with `tensor.empty` + `OneShotBufferize`, keeping the inference hot-loop 100% free of memory allocations (`malloc`/`free`).
- **Runtime Components**:
  - OpenMP-based multithreading (Optimal on P-Cores).
  - Built-in BPE Tokenizer and dynamic `KVCacheManager`.

### ⚠️ What's Missing (Critical Gaps)

| Gap | Description | Priority |
|-----|-------------|----------|
| **K/V Cache Quantization** | K/V cache currently consumes FP32/FP16 memory. Needs `i8` or `i4` quantization to support large context windows on edge devices. | 🔴 Critical |
| **Attention + RoPE Fusion** | RoPE and SDPA are currently separate. Fusing them into a FlashAttention-style MLIR kernel will reduce memory bandwidth. | 🟡 High |
| **Multi-target Backends** | Micro-kernels are explicitly tuned for x86 AVX2. Need equivalent paths for ARM NEON/SVE and RISC-V RVV. | 🔴 Critical |
| **GPU/Vulkan Support** | The SPIR-V pipeline is a basic proof-of-concept. It needs to support `tl1_pack` and `q8` tensor layouts. | 🟡 High |
| **Autoregressive Speculation** | Needs speculative decoding or Medusa-heads to break the memory bandwidth wall. | 🔵 Future |

---

## 📊 Performance Metrics

**Hardware:** Intel Core i3-1215U (Alder Lake, 15W, 2 P-Cores, 4 E-Cores)
**Model:** `microsoft/BitNet-b1.58-2B-4T` (30 layers)

| Configuration | Threads | Decode Speed | Memory Footprint | Note |
|---------------|---------|--------------|------------------|------|
| **Baseline (F32/F16)** | N/A | OOM | ~4.0 GB | Exceeds device memory |
| **Tenzo (Q8 + TL1 Unroll x4)** | 2 (P-Cores) | 1.99 tok/sec | 851 MB | Register spilling due to >16 YMM limit |
| **Tenzo (Q8 + TL1 Unroll x2)** | 4 (SMT) | 1.91 tok/sec | 851 MB | SMT cache contention degrades perf |
| **Tenzo (Q8 + TL1 Unroll x2)** | **2 (P-Cores)** | **2.55 tok/sec** | **851 MB** | **Optimal: Zero-Spill Fused FMA/PSHUFB** |

---

## 🗺️ Roadmap

### Phase 1: Edge CPU Dominance (Ongoing)
- [x] End-to-end BitNet 1.58B generation.
- [x] Zero-Allocation MLIR Bufferization.
- [x] Hand-tuned AVX2 micro-kernels (Zero-Spill).
- [ ] **K/V Cache Quantization**: Implement INT8/INT4 dynamic caching.
- [ ] **ARM NEON Backend**: Expand auto-vectorizer to target Apple Silicon and Snapdragon.

### Phase 2: Kernel Fusion & GPU
- [ ] **FlashAttention MLIR Kernel**: Fuse SDPA, RoPE, and causal masking.
- [ ] **Vulkan SPIR-V Compute**: Bring 1.58-bit decompression to iGPUs via Vulkan.

### Phase 3: Developer Experience
- [ ] **Standalone Inference Engine**: Decouple the JIT execution engine from `tenzo-cli` for embedding in iOS/Android apps.
- [ ] **Multi-Model Support**: Support modern architectures beyond LLaMA (Mistral, MoE).

---

## 📦 Component Inventory

- `src/dialect/` — MLIR definitions (`tenzo.bitlinear_tl1`, `tenzo.rope`).
- `src/passes/` — Compiler passes (Linalg Lowering, Bufferization, Micro-Kernel generation).
- `src/context/` — Hardware detection and topology profiling.
- `src/runtime/` — Execution environment (Tokenizer, KVCacheManager, OpenMP executor).
- `tenzo-frontend/` — PyTorch-to-MLIR exporter (`export_bitnet.py`).
