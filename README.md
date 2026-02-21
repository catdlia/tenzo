# ⚡ Tenzo Compiler

> **Heterogeneous AI Compiler built on MLIR/LLVM** - Write once, vectorize everywhere, dispatch anywhere.(https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

Tenzo is an experimental, high-performance Deep Learning compiler leveraging the **MLIR (Multi-Level Intermediate Representation)** framework. It bridges the gap between high-level neural network mathematics and raw hardware execution (CPU/iGPU).

Unlike standard interpreters, Tenzo actively transforms compute graphs into highly optimized machine code by utilizing **GotoBLAS/BLIS-style cache blocking**, explicit AVX2/FMA vectorization, and Vulkan SPIR-V generation.

### ✨ Key Technical Features
- **Custom MLIR Dialect (`tenzo`)**: High-level graph representation (`tenzo.matmul`, `tenzo.conv2d`, `tenzo.relu`).
- **Aggressive CPU Optimization**: 5-loop GotoBLAS cache-blocking architecture with memory packing for L1/L2/L3 cache efficiency.
- **Hardware-Aware Vectorization**: Explicit 6x16 micro-kernels targeting AVX2 YMM registers with zero spilling.
- **Operator Fusion**: Fuses bias and activation functions (e.g., MatMul + Bias + ReLU) directly into the AVX2 micro-kernel registers, eliminating memory roundtrips.
- **Heterogeneous Execution**: Path to lower Linalg operations to GPU Dialect and serialize to `spirv.module` for Vulkan execution.

## 📊 Performance (MatMul 512x512)

Tested on Intel Alder Lake (i3-1215U) under ideal thermal conditions.

| Approach | Environment | Throughput | Note |
|----------|-------------|------------|------|
| **LLVM -O3 (Scalar)** | Single-core | ~2.3 GFLOPS | Standard loop compilation |
| **OpenBLAS (Numpy)** | Single-core | ~45.7 GFLOPS | Highly optimized assembly |
| **Tenzo (MLIR E2E)** | Single-core | **~60.3 GFLOPS** | Auto-generated AVX2 micro-kernel + Cache Blocking |
| **Tenzo (Isolated)** | L1 Cache | **124.4 GFLOPS** | Peak micro-kernel efficiency |
| **Tenzo Parallel** | 8 Instances | **~225.2 GFLOPS** | Thread-bound parallel scaling |

*Note: Tenzo outperforms OpenBLAS (+31.9% in single-threaded) by exploiting perfect register allocation, zero-overhead packing, and exact L1/L2 tile sizing.*

## 🏗️ Architecture Stack

```text
   tenzo.matmul, tenzo.conv2d, tenzo.add
        │
        ▼   Fusion Pass (e.g., MatMul + ReLU -> FusedMatMulRelu)
        │
        ▼   Linalg Lowering -> Bufferization -> MemRef
        │
    ┌───┴──────────────────────────────┐
    ▼                                  ▼
1. GotoBLAS Packing               1. Linalg to Parallel Loops
2. 5-Loop Cache Blocking          2. SCF to GPU Dialect
3. 6x16 Micro-kernel unrolling    3. GPU to SPIR-V Dialect
4. LLVM IR Lowering               4. Binary Serialization
    │                                  │
    ▼                                  ▼
LLVM JIT Engine                   Vulkan Compute Pipeline
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose (Highly recommended for reproducible LLVM 21 environments)
- *Optional:* Native LLVM/MLIR 21 and Vulkan SDK.

### Build & Run via Docker

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/tenzo.git
cd tenzo

# 2. Enter the development container
docker compose run --rm -it -v $(pwd):/app -w /app dev bash

# 3. Build the compiler inside the container
mkdir -p build_e2e && cd build_e2e
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
ninja tenzo-cli

# 4. Run the End-to-End GEMM Benchmark
./tenzo-cli gemm-e2e
```

### Command Line Interface
```bash
./tenzo-cli cpu           # Run standard CPU MatMul benchmark
./tenzo-cli microkernel   # Run pure 6x16 micro-kernel performance test
./tenzo-cli gemm-e2e      # Run full 5-loop packed GEMM benchmark
./tenzo-cli conv2d        # Run Conv2D (NHWC) test
./tenzo-cli gpu           # Run GPU/SPIR-V pipeline test
```

## 📂 Project Structure

- `src/dialect/`: Definition of the `tenzo` MLIR dialect (ODS/TableGen).
- `src/passes/`: Core compiler optimizations.
  - `PackingKernels.cpp`: Memory layout transformations (Row-major to Block-panel).
  - `MacroKernelPass.cpp`: Cache-blocking loop nest generation.
  - `ExplicitMicroKernelPass.cpp`: AVX2 FMA instruction generation.
  - `gpu/GPULowering.cpp`: SPIR-V conversion pipeline.
- `src/runtime/`: Minimal Vulkan compute wrapper.
- `src/tests/`: Benchmarks and E2E validation.

## 🗺️ Roadmap

- End-to-End Linalg to LLVM pipeline.
- GotoBLAS micro-kernel architecture & packing.
- AVX2 explicit vectorization (124 GFLOPS peak).
- Intra-operation Multithreading (OpenMP parallelization of macro-kernel).
- Conv2D optimization via `im2col` + GEMM.
- Complete Vulkan Runtime integration for MatMul.
- Quantization (INT8) using AVX-VNNI.

## 📜 License
MIT License
