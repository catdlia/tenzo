# 📋 Tenzo Compiler — Project Status

> **Last updated:** 2026-03-14

---

## 💡 Problem Statement

There is no open-source, truly **heterogeneous compiler** for tensor computations (inference and training) that targets **consumer and edge hardware**. The existing landscape is:

| Solution | Problem |
|----------|---------|
| TensorRT, CoreML, OpenVINO | Proprietary, vendor-locked |
| XLA (Google) | Tightly coupled to TPU/CUDA server stacks |
| TVM | Complex, still mostly server-oriented |
| Triton | NVIDIA GPUs only |
| IREE (Google) | Closest competitor, but immature on consumer hardware |

**Tenzo's thesis:** LLVM/MLIR already provides backends for every major ISA (x86, ARM, RISC-V, AMDGPU, NVPTX, SPIR-V). By building a compiler *on top of MLIR*, we can detect the hardware at compile-time, select the optimal lowering path, and generate native code — **for any target** — without vendor lock-in.

---

## 🎯 Vision

```text
                      Tensor IR (tenzo dialect)
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
           Operator Fusion         Graph Optimization
                    │
                    ▼
         Linalg / Tensor / MemRef (MLIR mid-level)
                    │
          ┌────┬────┼────┬────────┐
          ▼    ▼    ▼    ▼        ▼
        x86  ARM  RISC-V  CUDA   Vulkan/SPIR-V
       AVX2  NEON  V-ext  PTX    (iGPU/AMD)
       AVX512 SVE
          │    │    │      │        │
          ▼    ▼    ▼      ▼        ▼
       ┌──────────────────────────────┐
       │  Hardware Auto-Detection     │
       │  → Optimal tiling, blocking  │
       │  → Register mapping          │
       │  → Cache hierarchy tuning    │
       └──────────────────────────────┘
```

**Goal:** Write a neural network once → Tenzo detects the target hardware → generates optimal machine code → runs inference (and eventually training).

---

## 📍 Current State (honest assessment)

### ✅ What Works
- **Custom MLIR Dialect** (`tenzo.matmul`, `tenzo.conv2d`, `tenzo.relu`, `tenzo.add`)
- **Full CPU compilation pipeline** (x86 AVX2 only):
  - Tenzo DSL → Linalg → Bufferization → GotoBLAS packing → 6×16 micro-kernel → LLVM JIT
  - Operator Fusion (MatMul + Bias + ReLU)
  - 60.3 GFLOPS on 512×512 matmul (beats OpenBLAS by 31.9%)
- **Basic GPU path** (Vulkan/SPIR-V only, proof of concept)
- **Hardware detection** (x86 P/E-core topology via sysfs, cache sizes)

### ⚠️ What's Missing (Critical Gaps)

| Gap | Description | Priority |
|-----|-------------|----------|
| **No inference runtime** | No model loading, no graph execution engine, no memory management | 🔴 Critical |
| **Single-target only** | Only x86 AVX2 works; ARM/RISC-V/CUDA not implemented | 🔴 Critical |
| **Primitive HW detection** | Only reads x86 sysfs; needs a proper hardware abstraction layer | 🟡 High |
| **No model ingestion** | Can't load ONNX/SafeTensors/other model formats | 🟡 High |
| **No training** | Only forward pass; no autograd, no backward pass, no optimizer | 🔵 Future |
| **No memory planning** | No tensor lifetime analysis, no memory pool allocation | 🟡 High |
| **No quantization** | INT8/INT4 paths not implemented (despite VNNI support on hardware) | 🟡 High |
| **No multi-threading** | Macro-kernel runs single-threaded; `ThreadPool::parallelFor()` API added but not yet wired into GEMM | 🟡 High |

---

## 📊 Performance (what's proven)

MatMul 512×512, Intel i3-1215U (Alder Lake):

| Approach | GFLOPS | vs OpenBLAS |
|----------|--------|-------------|
| LLVM -O3 Scalar | ~2.3 | — |
| OpenBLAS (NumPy) | ~45.7 | baseline |
| **Tenzo MLIR E2E** | **~60.3** | **+31.9%** |
| **Tenzo Isolated Kernel** | **124.4** | **+172%** |
| **Tenzo 8-instance Parallel** | **~225.2** | **+393%** |

This proves the compilation approach works well for x86 AVX2. The challenge is generalizing this to other targets.

---

## 🗺️ Roadmap

### Phase 1: Inference Engine (next milestone)
- [x] **Hardware Abstraction Layer (HAL)**: Abstract hardware detection beyond x86 sysfs — need a unified `HardwareProfile` that describes any target (ISA, SIMD width, cache hierarchy, num cores, GPU capabilities)
- [x] **Model Ingestion**: ONNX parser → Tenzo dialect graph
- [x] **Dynamic Shape Support**: Support for `?` batch sizes in MLIR and C++ Runtime
- [x] **Inference Runtime**: `ExecutionContext`, `Tensor`, and `MemRefDescriptor` for JIT execution
- [ ] **Memory Planning**: Tensor lifetime analysis, memory pool allocation
- [ ] **Intra-kernel Threading**: Parallelize macro-kernel M-loop via OpenMP

### Phase 2: Multi-Target Backends
- [ ] **ARM/NEON backend**: Micro-kernel generation for NEON/SVE (Apple Silicon, Snapdragon, server ARM)
- [ ] **RISC-V Vector extension**: Micro-kernel for RVV 1.0
- [ ] **CUDA/NVPTX path**: MLIR GPU dialect → NVPTX backend, cuBLAS-competitive kernels
- [ ] **AMD/ROCm path**: MLIR → AMDGPU backend or HIP-compatible SPIR-V

### Phase 3: Production Readiness
- [ ] **Quantization**: INT8 (AVX-VNNI `vpdpbusd`), INT4, mixed-precision
- [ ] **Conv2D via im2col + GEMM**: Reuse GEMM infra for convolutions
- [ ] **Operator library expansion**: Softmax, LayerNorm, Attention, Embedding
- [ ] **Benchmark CI**: Automated perf regression tracking across targets

### Phase 4: Training (distant future)
- [ ] **Autograd**: Reverse-mode automatic differentiation on the Tenzo IR
- [ ] **Backward pass generation**: Adjoint of each operation
- [ ] **Optimizer kernels**: SGD, Adam, AdamW as fused operations
- [ ] **Distributed training**: Multi-device data/model parallelism

---

## 📦 Component Inventory

### `src/dialect/` — Tenzo MLIR Dialect
| File | Status | Description |
|------|--------|-------------|
| `TenzoOps.td` | ✅ Stable | ODS: `matmul`, `conv2d`, `relu`, `add` |
| `TenzoDialect.cpp/h` | ✅ Stable | Dialect registration |

### `src/passes/` — Compiler Passes (CPU path)
| File | Status | Description |
|------|--------|-------------|
| `Lowering.cpp` | ✅ Stable | Tenzo → Arith patterns |
| `FusionPass.cpp` | ✅ Stable | Operator fusion |
| `LinalgLowering.cpp` | ✅ Stable | Tenzo → Linalg |
| `Bufferization.cpp` | ✅ Stable | One-shot bufferization |
| `OptimalVectorization.cpp` | ✅ Stable | GotoBLAS-style vectorization |
| `ExplicitMicroKernel.cpp` | ✅ Stable | 6×16 AVX2 FMA kernel |
| `ExplicitMicroKernelPass.cpp` | ✅ Stable | Pass wrapper |
| `PackingPass.cpp` | ✅ Stable | BLIS-style packing |
| `PackingKernels.cpp` | ✅ Stable | Row-major → Block-panel |
| `MacroKernelPass.cpp` | ✅ Stable | 5-loop cache-blocking |
| `TransformStrategy.cpp` | ✅ Stable | Transform Dialect strategy |
| `LLVMLowering.cpp` | ✅ Stable | LLVM IR lowering |

### `src/passes/gpu/` — GPU/SPIR-V Path
| File | Status | Description |
|------|--------|-------------|
| `GPULowering.cpp/h` | 🧪 PoC | Linalg → GPU → SPIR-V (proof of concept) |

### `src/context/` — Hardware Abstraction Layer
| File | Status | Description |
|------|--------|--------------|
| `HardwareProfile.h` | ✅ Stable | Abstract interface: ISA, topology, cache, micro-kernel params |
| `HardwareProfile.cpp` | ✅ Stable | Platform-aware factory (`#ifdef __aarch64__`) |
| `X86HardwareProfile.cpp/h` | ✅ Stable | x86 sysfs P/E-core topology, CPUID, cache detection |
| `ARMHardwareProfile.cpp/h` | 🧪 Stub | ARM NEON/SVE, big.LITTLE topology detection |
| `TenzoContext.cpp/h` | ✅ Stable | MLIR context + dialect registration |
| `AutoTuner.cpp/h` | ✅ Stable | Runtime parameter tuning |

### `src/runtime/` — Execution
| File | Status | Description |
|------|--------|-------------|
| `VulkanRuntime.cpp/h` | 🧪 PoC | Minimal Vulkan compute wrapper |
| `ThreadPool.cpp/h` | ✅ Stable | CPU-pinned thread pool |

---

## 🏗️ Build Infrastructure

```text
 Compilation:  make build → remote_build.sh → Hetzner cpx62 (16C) + distcc
 Execution:    docker compose run → Docker container (Vulkan SDK, LLVM 21)
```

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Build config (tenzo-cli + micro_bench targets) |
| `Makefile` | Convenience targets: `make build`, `cpu`, `test`, `gpu`, etc. |
| `remote_build.sh` | Hetzner cloud compilation orchestrator |
| `Dockerfile` | Ubuntu 24.04 + LLVM 21 + Vulkan SDK |
| `docker-compose.yml` | Dev container with Intel iGPU passthrough |
