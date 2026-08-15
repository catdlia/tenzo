# ⚡ Tenzo Compiler

> **Open-source heterogeneous AI compiler built on MLIR/LLVM** — Write once, compile for any hardware.

## 💡 Why Tenzo?

There is no open-source compiler for tensor computations that truly targets **all hardware**. TensorRT is NVIDIA-only. CoreML is Apple-only. XLA is married to TPUs. OpenVINO is Intel-only. Every solution is either proprietary, vendor-locked, or server-oriented.

**Tenzo** uses MLIR/LLVM as a universal backend to generate optimal native code for **any target**: x86 (AVX2/AVX-512), ARM (NEON/SVE), RISC-V (V-extension), CUDA, AMD, and Vulkan GPUs. The compiler detects the hardware, selects the optimal lowering strategy (tiling, blocking, register allocation, SIMD width), and generates code that rivals hand-tuned libraries.

### ✨ Key Features
- **Custom MLIR Dialect** (`tenzo.matmul`, `tenzo.bitlinear_tl1`, `tenzo.rope`) — high-level tensor IR
- **Zero-Allocation Bufferization** — in-place memory reuse keeping allocations completely out of the hot loop
- **Explicit AVX2 micro-kernels** — e.g. `vpshufb` 256-bit SIMD kernel for BitNet with unroll x2 and zero register spilling
- **Quantization Support** — BitNet 1.58b ternary weights (`tl1_pack`), `lm_head` Q8 (`i8`) quantization
- **Operator Fusion & Hardware auto-detection** — CPU topology, cache hierarchy, SIMD capabilities
- **Vulkan/SPIR-V path** — proof-of-concept GPU pipeline

## 📊 Performance (BitNet 1.58B 2B model, 30 Layers)
*Benchmarked on Intel Core i3-1215U (Alder Lake, 15W, 2 P-Cores).*

| Configuration | Model Size | Decode Speed | Note |
|---------------|------------|--------------|------|
| **Baseline (F32/F16)** | ~4.0 GB | OOM / N/A | Exceeds device memory |
| **BitNet Q8 + TL1 (2 Threads)** | **851 MB** | **~2.55 tok/sec** | **Fully in-register AVX2 `vpshufb` decoding (Optimal)** |
| BitNet Q8 + TL1 (4 Threads SMT) | 851 MB | ~1.91 tok/sec | SMT degrades performance due to L1/L2 cache contention |

## 🏗️ Architecture

```text
   tenzo.matmul / tenzo.bitlinear_tl1 / tenzo.rope
                 │
         Fusion + Graph Opt
                 │
    Linalg → Zero-Alloc Bufferization → MemRef
                 │
    ┌────┬───────┼───────┬──────────┐
    ▼    ▼       ▼       ▼          ▼
  x86  ARM   RISC-V   CUDA    Vulkan/SPIR-V
  AVX2 NEON  V-ext    PTX     (iGPU/AMD)
    │    │       │       │          │
    └────┴───────┴───────┴──────────┘
         Hardware Auto-Detection
         → Optimal tiling, blocking
         → Register mapping (Zero-spill)
```

## 📍 Current State

**Working:** Full x86 AVX2 pipeline (DSL → LLVM JIT), operator fusion, BitNet 1.58B generation (Llama architecture), zero-allocation bufferization.
**Not yet:** Multi-target backends, standalone inference runtime beyond CLI.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed component inventory and gap analysis.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Hetzner Cloud CLI (`hcloud`) for remote compilation

### Build & Run
```bash
make build         # Compile remotely (Hetzner cloud server)
make build-local   # Compile locally in Docker
make test          # Quick validation (CPU + Conv2D + GPU)
make cpu           # CPU MatMul benchmark
make bench         # All benchmarks
```

> Compilation via `distcc` on a Hetzner server (or local Docker). Execution inside Docker with Vulkan SDK + LLVM 21.

## 📂 Structure

| Directory | Purpose |
|-----------|---------|
| `src/dialect/` | Tenzo MLIR dialect (ODS/TableGen) |
| `src/passes/` | Compiler passes: fusion, lowering, zero-alloc bufferize, AVX2 micro-kernels |
| `src/passes/gpu/` | SPIR-V conversion pipeline |
| `src/context/` | Hardware detection, auto-tuning |
| `src/runtime/` | Vulkan compute wrapper, OpenMP execution |
| `src/tests/` | Benchmarks and E2E validation |
| `tenzo-frontend/` | PyTorch exporters (`export_bitnet.py`) |

## 🗺️ Roadmap

- [x] End-to-end MLIR compilation pipeline (x86 AVX2)
- [x] GotoBLAS micro-kernel architecture & packing
- [x] BitNet 1.58B LLM Generation with `vpshufb` AVX2 kernel (Zero Spill)
- [x] Zero-Allocation MLIR Bufferization
- [ ] **Multi-target backends** — ARM/NEON, RISC-V, CUDA/NVPTX, AMD/ROCm
- [ ] **Hardware Abstraction Layer** — unified hardware profiling beyond x86
- [ ] **Training** — autograd, backward pass, optimizer kernels

## 📜 License
MIT License
