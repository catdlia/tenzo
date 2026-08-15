# ⚡ Tenzo Compiler

> **Open-source heterogeneous AI compiler built on MLIR/LLVM** — Write once, compile for any hardware.

## 💡 Why Tenzo?

There is no open-source compiler for tensor computations that truly targets **all hardware**. TensorRT is NVIDIA-only. CoreML is Apple-only. XLA is married to TPUs. OpenVINO is Intel-only. Every solution is either proprietary, vendor-locked, or server-oriented.

**Tenzo** uses MLIR/LLVM as a universal backend to generate optimal native code for **any target**, with a specialized focus on heavily quantized Large Language Models (like the 1.58-bit BitNet architectures). The compiler detects the hardware, selects the optimal lowering strategy (register allocation, SIMD width), and generates code that rivals hand-tuned libraries.

### ✨ Key Features
- **Custom MLIR Dialect** (`tenzo.bitlinear_tl1`, `tenzo.rope`, `tenzo.rms_norm`) — high-level LLM tensor IR.
- **Zero-Allocation Bufferization** — in-place memory reuse keeping dynamic memory allocations completely out of the inference hot loop.
- **Explicit AVX2 micro-kernels** — e.g. `vpshufb` 256-bit SIMD kernel for BitNet with unroll x2 and **zero register spilling**.
- **Quantization Support** — BitNet 1.58B ternary weights (`tl1_pack`), `lm_head` Q8 (`i8`) quantization.
- **Operator Fusion & Hardware auto-detection** — CPU topology, cache hierarchy, SIMD capabilities.

## 📊 Performance (BitNet 1.58B 2B model, 30 Layers)
*Benchmarked on Intel Core i3-1215U (Alder Lake, 15W, 2 P-Cores).*

| Configuration | Model Size | Decode Speed | Note |
|---------------|------------|--------------|------|
| **Baseline (F32/F16)** | ~4.0 GB | OOM / N/A | Exceeds device memory |
| **BitNet Q8 + TL1 (2 Threads)** | **851 MB** | **~2.55 tok/sec** | **Fully in-register AVX2 `vpshufb` decoding (Optimal)** |
| BitNet Q8 + TL1 (4 Threads SMT) | 851 MB | ~1.91 tok/sec | SMT degrades performance due to L1/L2 cache contention |

## 🏗️ Architecture

```text
       PyTorch / HuggingFace Model
                 │
   tenzo-frontend (Weight Extraction & tl1_pack)
                 │
   tenzo.bitlinear_tl1 / tenzo.rope / tenzo.matmul_q8
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

**Working:** Full x86 AVX2 pipeline (DSL → LLVM JIT), operator fusion, BitNet 1.58B autoregressive generation, zero-allocation bufferization, OpenMP multithreading.
**Not yet:** Multi-target backends, standalone inference runtime beyond CLI, K/V Cache quantization.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed component inventory and gap analysis.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- `uv` (Python package manager)

### 1. Build the Compiler
```bash
make build-local   # Compile locally in Docker
```

### 2. Export Model (BitNet 1.58B)
```bash
uv run python3 tenzo-frontend/export_bitnet.py --quant-mode tl1_pack --num-layers 30 --output-dir export_output_bitnet
```

### 3. Run Inference
```bash
docker compose run --rm -e OMP_NUM_THREADS=2 -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-cli generate -m /app/export_output_bitnet -n 30
```

## 📂 Structure

| Directory | Purpose |
|-----------|---------|
| `src/dialect/` | Tenzo MLIR dialect (ODS/TableGen) |
| `src/passes/` | Compiler passes: fusion, lowering, zero-alloc bufferize, AVX2 micro-kernels |
| `src/context/` | Hardware detection, auto-tuning |
| `src/runtime/` | Tokenizer, KVCacheManager, OpenMP execution |
| `src/tests/` | Benchmarks and E2E validation |
| `tenzo-frontend/` | PyTorch exporters (`export_bitnet.py`) |

## 🗺️ Roadmap

- [x] End-to-end MLIR compilation pipeline (x86 AVX2)
- [x] BitNet 1.58B LLM Generation with `vpshufb` AVX2 kernel (Zero Spill)
- [x] Zero-Allocation MLIR Bufferization
- [ ] **K/V Cache Quantization** — INT8/INT4 context window
- [ ] **Multi-target backends** — ARM/NEON, RISC-V, CUDA/NVPTX, AMD/ROCm

## 📜 License
MIT License
