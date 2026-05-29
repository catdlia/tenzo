# 📖 Tenzo Compiler: Comprehensive Documentation

Tenzo is a high-performance, MLIR-based compiler designed for heterogeneous tensor computations on consumer and edge hardware. It aims to bridge the gap between high-level machine learning frameworks and low-level hardware optimizations.

---

## 🏗 1. System Architecture

The compilation pipeline follows a multi-stage transformation:
1.  **Frontend (Python):** `model.onnx` → `tenzo.mlir` (Tenzo Dialect).
2.  **Middle-end (C++):** `tenzo` → `linalg` → `vector` → `llvm`.
3.  **Backend (JIT/LLVM):** LLVM IR → Native Machine Code (AVX2/Vulkan).

---

## 📂 2. Project Structure & Components

### 🛰 `tenzo-frontend/` (ONNX Ingestion)
The Python-based entry point for the compiler.
*   **`onnx_to_mlir.py`**: The core converter. It parses ONNX models, performs shape inference, and maps ONNX nodes to Tenzo MLIR operations.
*   **`weight_utils.py`**: Manages model weights. It extracts initializers from ONNX and saves them as `.bin` files while generating `memref.global` declarations in MLIR.
*   **`test_simple_mlp.py`**: A validation script that creates a synthetic 2-layer MLP to test the frontend pipeline.

### 🧩 `src/dialect/` (Tenzo MLIR Dialect)
Defines the high-level intermediate representation.
*   **`TenzoOps.td`**: ODS (Operation Definition Specification) file defining ops like `tenzo.matmul`, `tenzo.conv2d`, `tenzo.add`, and `tenzo.relu`.
*   **`TenzoDialect.cpp/h`**: C++ registration and implementation of the dialect.

### ⚙️ `src/passes/` (Transformation Passes)
The "brains" of the compiler, responsible for optimization.
*   **`LinalgLowering.cpp`**: Lowers high-level ops to `linalg.generic` for tiling and fusion.
*   **`FusionPass.cpp`**: Implements operator fusion (e.g., MatMul + Bias + ReLU).
*   **`OptimalVectorization.cpp`**: Implements GotoBLAS-style vectorization for x86 AVX2.
*   **`ExplicitMicroKernel.cpp`**: Generates highly optimized 6x16 FMA kernels.
*   **`PackingPass.cpp`**: Implements memory layout transformations (packing) for cache efficiency.

### 💻 `src/context/` (Hardware Abstraction Layer)
Ensures portability across different CPUs and GPUs.
*   **`HardwareProfile.h`**: Abstract interface for hardware capabilities (ISA, SIMD width, Cache topology).
*   **`X86HardwareProfile.cpp`**: Concrete implementation for x86, detecting P/E-cores and AVX capabilities.
*   **`TenzoContext.cpp`**: Manages the MLIR context and dialect registration.

### 🚀 `src/runtime/` (Execution)
*   **`ExecutionContext.cpp/h`**: The primary C++ API for running inference. Manages JIT compilation and `Tensor` input/output.
*   **`MemRefUtils.h`**: Low-level MLIR MemRef descriptor mapping for C++ interop. Supports dynamic shapes.
*   **`ThreadPool.cpp/h`**: A CPU-pinned thread pool for parallel execution of macro-kernels.
*   **`VulkanRuntime.cpp/h`**: Minimal runtime for executing SPIR-V kernels on GPUs.


---

## 🛠 3. Detailed Component Documentation

### 🐍 ONNX Frontend API (Python)
*   **`onnx_to_mlir(onnx_path, output_path)`**:
    *   **Input**: Path to `.onnx` file.
    *   **Output**: Path to `.mlir` file + `weights/` directory.
    *   **Logic**: Performs a topological sort of the ONNX graph, identifies `Add + Relu` sequences for fusion, and emits Tenzo dialect text.
*   **`save_initializer_as_bin(init, dir)`**:
    *   Converts ONNX `TensorProto` to `numpy` array, then to a raw `float32` binary file.

### 🏗 C++ Pipeline Functions
*   **`tenzo::runFullPipelineTest(context)`**:
    *   Compiles a MatMul operation through the entire CPU stack (Fusion -> Tiling -> Vectorization -> JIT).
*   **`tenzo::runGEMMEndToEndBenchmark(context)`**:
    *   Runs the most optimized CPU path including BLIS-style packing and explicit micro-kernels.
*   **`tenzo::gpu::runGPUPipelineTest(context)`**:
    *   Lowers MLIR to SPIR-V and executes it via the Vulkan runtime.

---

## 🔄 4. The Pipeline Walkthrough

### Step 1: Frontend Ingestion
The user runs `make onnx-convert MODEL=model.onnx`.
The Python script extracts weights and creates a `module { ... }` in MLIR. Heavy ops like `Gemm` are decomposed or fused into `tenzo.matmul` and `tenzo.fused_add_relu`.

### Step 2: High-Level Optimization
The C++ compiler applies `FusionPass`. It looks for patterns like:
`tenzo.matmul` → `tenzo.add` → `tenzo.relu`
and replaces them with a single fused operation to minimize memory traffic.

### Step 3: Tiling & Packing
`MacroKernelPass` tiles the operation based on the detected CPU cache sizes (L1/L2/L3). `PackingPass` reorders matrix data into "blocks" and "panels" to ensure contiguous memory access during computation.

### Step 4: Micro-Kernel Generation & JIT
The `ExplicitMicroKernelPass` generates a nested loop that fits exactly into the CPU registers (e.g., 6 rows x 16 columns for AVX2). This is then converted to LLVM IR and JIT-compiled into machine code.

---

## ⌨️ 5. User Commands Reference

| Command | Action |
| :--- | :--- |
| `make frontend-setup` | Install Python dependencies. |
| `make onnx-convert MODEL=m.onnx` | Convert ONNX to MLIR. |
| `make build` | Compile the C++ core (remote via distcc). |
| `make cpu` | Run standard CPU benchmarks. |
| `make gpu` | Run Vulkan/SPIR-V tests. |
| `make bench` | Run all performance tests. |

---

## 🗺 6. Roadmap
1.  **Phase 1 (Current):** Static ONNX conversion and optimized AVX2 kernels.
2.  **Phase 2:** ARM NEON support and dynamic shape handling.
3.  **Phase 3:** Full Inference Runtime with memory planning.
4.  **Phase 4:** Training support (Autograd on Tenzo IR).
