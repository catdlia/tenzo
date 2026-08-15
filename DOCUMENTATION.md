# 📖 Tenzo Compiler: Comprehensive Documentation

Tenzo is a high-performance, MLIR-based compiler designed for heterogeneous tensor computations on consumer and edge hardware. It aims to bridge the gap between high-level machine learning frameworks and low-level hardware optimizations, with a specific focus on quantized Large Language Models (LLMs) like BitNet 1.58B.

---

## 🏗 1. System Architecture

The compilation pipeline follows a multi-stage transformation:
1.  **Frontend (Python/PyTorch):** `export_bitnet.py` extracts HuggingFace models → `tenzo.mlir` (Tenzo Dialect) + `weights.bin`.
2.  **Middle-end (C++):** `tenzo` → `linalg` → `vector` → `llvm`.
3.  **Backend (JIT/LLVM):** LLVM IR → Native Machine Code (AVX2).

---

## 📂 2. Project Structure & Components

### 🛰 `tenzo-frontend/` (PyTorch Exporter)
The Python-based entry point for the compiler.
*   **`export_bitnet.py`**: The core converter. It loads a HuggingFace model (e.g., `BitNet-b1.58-2B-4T`), extracts ternary weights, packs them into 2-bit formats (`tl1_pack`), and generates the MLIR representation of the network.

### 🧩 `src/dialect/` (Tenzo MLIR Dialect)
Defines the high-level intermediate representation for LLMs.
*   **`TenzoOps.td`**: ODS (Operation Definition Specification) file defining ops like `tenzo.bitlinear_tl1`, `tenzo.matmul_q8`, `tenzo.rope`, and `tenzo.rms_norm`.
*   **`TenzoDialect.cpp/h`**: C++ registration and implementation of the dialect.

### ⚙️ `src/passes/` (Transformation Passes)
The "brains" of the compiler, responsible for optimization.
*   **`LinalgLowering.cpp`**: Lowers high-level ops to `linalg.generic` and specific hardware-intrinsics.
*   **`Bufferization.cpp`**: Performs **Zero-Allocation Bufferization** using `OneShotBufferize`, preventing dynamic heap allocations (`malloc`/`free`) during inference.
*   **`ExplicitMicroKernel.cpp`** / AVX2 CodeGen: Generates highly optimized inner loops (e.g., `vpshufb` 256-bit SIMD kernels) that perfectly fit inside CPU registers without spilling.

### 💻 `src/context/` (Hardware Abstraction Layer)
Ensures portability across different CPUs.
*   **`HardwareProfile.h`**: Abstract interface for hardware capabilities (ISA, SIMD width, Cache topology).
*   **`X86HardwareProfile.cpp`**: Concrete implementation for x86, detecting P/E-cores and AVX capabilities.

### 🚀 `src/runtime/` (Execution)
*   **`ExecutionContext.cpp/h`**: The primary C++ API for running inference. Manages JIT compilation.
*   **`Tokenizer.cpp/h`**: Implements BPE tokenization decoding for LLaMA-based vocabularies.
*   **`KVCacheManager.cpp/h`**: Dynamically manages the Key/Value cache tensors during autoregressive generation.
*   **`ThreadPool.cpp/h`**: OpenMP-powered thread pool for parallelizing the batch/sequence loops across P-Cores.

---

## 🛠 3. Detailed Component Documentation

### 🐍 HuggingFace Exporter (`export_bitnet.py`)
*   **`--quant-mode tl1_pack`**: 
    Extracts weights $w \in \{-1, 0, 1\}$, shifts them to $\{0, 1, 2\}$, and bitwise-packs them into 4-bit nibbles: `(w_even+1) | ((w_odd+1)<<2)`.
    This brings the footprint of a 2B model down to ~851 MB.
*   **`--num-layers`**: Configures how many transformer layers to export.

### 🏗 Compiler Micro-Kernels
*   **`BitLinearTL1PackLoweringToLinalg`**: 
    Converts the `tl1_pack` representation into a highly optimized unrolled AVX2 loop. The kernel duplicates the activation LUT across 128-bit lanes using `vinserti128`, processing 64 channels per instruction.
*   **RoPE (Rotary Positional Embeddings)**:
    Fully vectorized inline using MLIR complex numbers and `tensor.empty` to avoid `libm` fallback.

---

## 🔄 4. The Pipeline Walkthrough

### Step 1: Exporting
The user runs `uv run python3 tenzo-frontend/export_bitnet.py`.
The script writes the packed weights to `weights.bin` and the computation graph to `model.mlir`.

### Step 2: High-Level Optimization
The C++ CLI loads `model.mlir`. High-level ops like `tenzo.rms_norm` are broken down into arithmetic operations or vectorized directly.

### Step 3: Zero-Allocation Bufferization
Instead of allocating memory dynamically, ops use `tensor.empty`. The `OneShotBufferize` pass analyzes data flow and reuses previously allocated I/O buffers for every single generation step.

### Step 4: Micro-Kernel Generation & JIT
The compiler lowers the operations to LLVM IR, specifically mapping the `tl1_pack` loop to `llvm.x86.avx2.pshuf.b`. The LLVM ORC JIT compiles it to native x86 machine code.

---

## ⌨️ 5. User Commands Reference

| Command | Action |
| :--- | :--- |
| `make build` | Compile the C++ core remotely (Hetzner). |
| `make build-local` | Compile the C++ core locally inside Docker. |
| `make bench` | Run local tests and benchmarks. |
| `tenzo-cli generate` | Run autoregressive LLM generation on the exported MLIR model. |
