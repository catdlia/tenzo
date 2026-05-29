# ⚡ Tenzo Compiler

> **Open-source heterogeneous AI compiler built on MLIR/LLVM** — Write once, compile for any hardware.

## 💡 Why Tenzo?

There is no open-source compiler for tensor computations that truly targets **all hardware**. TensorRT is NVIDIA-only. CoreML is Apple-only. XLA is married to TPUs. OpenVINO is Intel-only. Every solution is either proprietary, vendor-locked, or server-oriented.

**Tenzo** uses MLIR/LLVM as a universal backend to generate optimal native code for **any target**: x86 (AVX2/AVX-512), ARM (NEON/SVE), RISC-V (V-extension), CUDA, AMD, and Vulkan GPUs. The compiler detects the hardware, selects the optimal lowering strategy (tiling, blocking, register allocation, SIMD width), and generates code that rivals hand-tuned libraries.

### ✨ Key Features
- **Custom MLIR Dialect** (`tenzo.matmul`, `tenzo.conv2d`, `tenzo.relu`) — high-level tensor IR
- **GotoBLAS/BLIS-style CPU optimization** — 5-loop cache-blocking with memory packing
- **Explicit AVX2 micro-kernels** — 6×16 register-tiled FMA with zero spilling
- **Operator Fusion** — MatMul + Bias + ReLU fused in-register
- **Hardware auto-detection** — CPU topology, cache hierarchy, SIMD capabilities
- **Vulkan/SPIR-V path** — proof-of-concept GPU pipeline

## 📊 Performance (MatMul 512×512, i3-1215U)

| Approach | GFLOPS | Note |
|----------|--------|------|
| LLVM -O3 (Scalar) | ~2.3 | Standard loop compilation |
| OpenBLAS (NumPy) | ~45.7 | Hand-optimized assembly |
| **Tenzo (MLIR E2E)** | **~60.3** | **+31.9% vs OpenBLAS** |
| **Tenzo (Isolated kernel)** | **124.4** | Peak micro-kernel efficiency |
| **Tenzo Parallel (8T)** | **~225.2** | Multi-instance scaling |

## 🏗️ Architecture

```text
   tenzo.matmul / tenzo.conv2d / tenzo.relu
                 │
         Fusion + Graph Opt
                 │
    Linalg → Bufferization → MemRef
                 │
    ┌────┬───────┼───────┬──────────┐
    ▼    ▼       ▼       ▼          ▼
  x86  ARM   RISC-V   CUDA    Vulkan/SPIR-V
  AVX2 NEON  V-ext    PTX     (iGPU/AMD)
    │    │       │       │          │
    └────┴───────┴───────┴──────────┘
         Hardware Auto-Detection
         → Optimal tiling, blocking
         → Register mapping
         → Cache hierarchy tuning
```

## 📍 Current State

**Working:** Full x86 AVX2 pipeline (DSL → LLVM JIT), operator fusion, basic Vulkan/SPIR-V.
**Not yet:** Multi-target backends, inference runtime, model loading, training.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed component inventory and gap analysis.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Hetzner Cloud CLI (`hcloud`) for remote compilation

### Build & Run
```bash
make build         # Compile remotely (Hetzner cloud server)
make test          # Quick validation (CPU + Conv2D + GPU)
make cpu           # CPU MatMul benchmark (512×512)
make gpu           # GPU/Vulkan pipeline test
make bench         # All benchmarks
```

> Compilation via `distcc` on a Hetzner server. Execution inside Docker with Vulkan SDK + LLVM 21.

## 📂 Structure

| Directory | Purpose |
|-----------|---------|
| `src/dialect/` | Tenzo MLIR dialect (ODS/TableGen) |
| `src/passes/` | Compiler passes: fusion, lowering, vectorization, packing, micro-kernel |
| `src/passes/gpu/` | SPIR-V conversion pipeline |
| `src/context/` | Hardware detection, auto-tuning |
| `src/runtime/` | Vulkan compute wrapper, thread pool |
| `src/tests/` | Benchmarks and E2E validation |

## 🗺️ Roadmap

- [x] End-to-end MLIR compilation pipeline (x86 AVX2)
- [x] GotoBLAS micro-kernel architecture & packing (124 GFLOPS peak)
- [x] Operator fusion (MatMul + ReLU)
- [ ] **Inference runtime** — model loading, graph executor, memory planner
- [ ] **Multi-target backends** — ARM/NEON, RISC-V, CUDA/NVPTX, AMD/ROCm
- [ ] **Hardware Abstraction Layer** — unified hardware profiling beyond x86
- [ ] **Quantization** — INT8/INT4 via AVX-VNNI
- [ ] **Training** — autograd, backward pass, optimizer kernels

## 📜 License
MIT License
