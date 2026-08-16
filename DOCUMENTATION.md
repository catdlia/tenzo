# 📖 Tenzo Compiler: Comprehensive Documentation (v0.3.0)

Tenzo is a high-performance, MLIR-based compiler and native execution runtime designed for heterogeneous tensor computations on consumer and edge hardware. It bridges the gap between high-level machine learning frameworks and low-level hardware optimizations, with a specialized focus on quantized Large Language Models (LLMs) like BitNet 1.58B.

---

## 🏗 1. System Architecture

The compilation and execution pipeline follows a multi-stage transformation:
1.  **Frontend (Python/PyTorch):** `export_bitnet.py` extracts HuggingFace models → `model.mlir` (Tenzo Dialect) + `weights.bin`.
2.  **Middle-end (MLIR/C++):** `tenzo` → `linalg` → `vector` → `llvm`.
3.  **Backend (JIT/LLVM):** LLVM IR → Native Machine Code with hand-tuned AVX2 micro-kernels.
4.  **Native Runtime (`tenzo_runtime`):** Pure C++ Zero-Allocation `ExecutionContext` with fused multi-projection OpenMP execution, in-register INT8 KV-Cache compression, and native Top-K / Top-P min-heap sampling.

---

## 📂 2. Project Structure & Components

### 🛰 `tenzo-frontend/` (PyTorch Exporter)
The Python-based entry point for the compiler.
*   **`export_bitnet.py`**: Loads a HuggingFace model (e.g., `BitNet-b1.58-2B-4T`), extracts ternary weights, packs them into 2-bit formats (`tl1_pack`), and generates the MLIR representation of the network.

### 🧩 `src/dialect/` (Tenzo MLIR Dialect)
Defines the high-level intermediate representation for LLMs.
*   **`TenzoOps.td`**: ODS (Operation Definition Specification) file defining ops like `tenzo.bitlinear_tl1`, `tenzo.matmul_q8`, `tenzo.rope`, and `tenzo.rms_norm`.
*   **`TenzoDialect.cpp/h`**: C++ registration and implementation of the dialect.

### ⚙️ `src/passes/` (Transformation Passes)
The compiler optimization engine.
*   **`LinalgLowering.cpp`**: Lowers high-level ops to `linalg.generic` and hardware intrinsics.
*   **`Bufferization.cpp`**: Performs **Zero-Allocation Bufferization** using `OneShotBufferize`, preventing dynamic heap allocations (`malloc`/`free`) during inference.
*   **`ExplicitMicroKernel.cpp`** / AVX2 CodeGen: Generates highly optimized inner loops (`vpshufb` 256-bit SIMD kernels) that fit inside CPU registers without stack spilling.

### 💻 `src/context/` (Hardware Abstraction Layer)
*   **`HardwareProfile.h`**: Abstract interface for hardware capabilities (ISA, SIMD width, Cache topology).
*   **`X86HardwareProfile.cpp`**: Concrete implementation for x86, detecting P/E-cores and AVX2/AVX-VNNI capabilities.

### ⚡ `src/bindings/PybindModule.cpp` (`tenzo_runtime`)
The high-performance C++ execution engine that powers ultra-fast autoregressive decoding.
*   **`ExecutionContext`**: Maintains static, pre-allocated scratch buffers for all 30 layers with **zero memory allocations on the hot path**.
*   **`generate_step_cxx(...)`**: Runs embedding lookup, all 30 transformer layers, INT8 LM-head projection, and sampling entirely in native C++.
*   **`Fused OpenMP Regions`**:
    *   *Fused QKV*: Computes 40 Q blocks + 10 K blocks + 10 V blocks in a single 60-block OpenMP parallel loop.
    *   *Fused Gate-Up*: Computes 108 Gate blocks + 108 Up blocks in a single 216-block OpenMP parallel loop.
*   **`16-Bit SIMD Accumulation`**: Uses `_mm256_add_epi16` in the inner BitLinear loop, accumulating in 16-bit registers and reducing to 32-bit only once every 64 iterations ($64 \times 62 = 3968 \ll 32767$).
*   **`Fused INT8 KV-Cache Compression`**:
    *   Quantizes Key and Value tensors into `int8_t` online with channel-wise scales.
    *   Dequantizes during SDPA directly into 16 YMM registers with sign-extended `_mm256_cvtepi8_epi32`.
    *   Reduces KV memory consumption by **4x** (314 MB for 8192 context).
*   **`Top-K / Top-P Min-Heap Sampler`**: Fast 40-element min-heap in C++ completing sampling in **0.01 ms** (1800x faster than Python sorting).

---

## 🛠 3. User Commands & Makefile Reference

| Command | Description |
| :--- | :--- |
| `make build-local` | Compiles `tenzo-cli` and `tenzo_runtime` locally inside Docker. |
| `make build` | Triggers remote cloud compilation (via Hetzner / DigitalOcean). |
| `make run-fast [PROMPT="..."] [TOKENS=50]` | Executes ultra-fast native C++ autoregressive text generation. |
| `make compare [PROMPT="..."] [TOKENS=50]` | Runs a direct side-by-side benchmark against Microsoft `BitNet.cpp`. |
| `make cpu` | Runs CPU matrix multiplication benchmarks. |
| `make test` | Runs the full compiler regression test suite. |

---

## 📊 4. Performance Summary (Alder Lake Core i3-1215U)

| Engine | Model Format | KV-Cache | Decode Throughput | Per-Token Latency |
|---|---|---|---|---|
| **Microsoft BitNet.cpp** | TL1 + INT8 LM | FP32 | 12.31 tok/sec | 81.23 ms |
| **Tenzo Native Engine (v0.3.0)** | **TL1 + INT8 LM** | **INT8 Fused** | **20.32 tok/sec** | **49.21 ms (1.65x faster 🚀)** |

