# 🛠️ Tenzo CLI & Testing Guide

This document describes the recommended developer workflows for building, testing, and debugging the Tenzo compiler and runtime.

---

## ⚡ 1. Standard Workflow (Makefile & Docker Compose)

In accordance with project rules, all binaries are compiled remotely or via `make build` and run locally inside the Docker container environment.

### 1.1 Running Tests
```bash
# Run full compiler regression test suite
make test

# Run CPU MatMul benchmark
make cpu

# Run large matrix benchmark (768x768)
make large

# Run hardware diagnostics
make diag
```

### 1.2 Interactive Development Shell
To avoid spinning up a new container for each command, open an interactive bash shell in the development container:
```bash
make dev
# or:
docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev bash
```

Inside the container, binaries in `/app/cmake-build-debug` can be run directly:
```bash
/app/cmake-build-debug/tenzo-cli test
/app/cmake-build-debug/tenzo-cli cpu
/app/cmake-build-debug/tenzo-cli version
```

---

## 🎮 2. Main Test Commands (`tenzo-cli`)

All commands are available via the `tenzo-cli` binary:

### 🏆 `test` (Core Regression Suite)
Runs all 5 core validation suites:
1. CPU MatMul (512x512 with FMA/AVX2)
2. Conv2D vectorization
3. GPU / Vulkan SPIR-V pipeline
4. Ternary Pack (1.58-bit AVX2 micro-kernel verification)
5. Bitwise Attention (SIMD packed attention micro-kernel)

```bash
docker compose run --rm dev /app/cmake-build-debug/tenzo-cli test
```

### 🔬 `cpu` (CPU MatMul Benchmark)
Evaluates standard matrix multiplication performance with LLVM O3 baseline vs Tenzo vector optimization.
```bash
docker compose run --rm dev /app/cmake-build-debug/tenzo-cli cpu
```

### 📦 `packing` (Packing Kernels Benchmark)
Benchmarks matrix data reordering and packing bandwidth.
```bash
docker compose run --rm dev /app/cmake-build-debug/tenzo-cli packing
```

### 🧠 `attention` (Bitwise Attention Suite)
Tests SIMD packed attention kernels, zero-copy GQA routing, and numerical parity against FP32 reference.
```bash
docker compose run --rm dev /app/cmake-build-debug/tenzo-cli attention
```

---

## 🎛️ 3. Debugging with GDB

If an operation encounters a fault or unexpected latency, debug it inside the Docker container:

```bash
# Start dev container
make dev

# Run under GDB
gdb --args /app/cmake-build-debug/tenzo-cli test
(gdb) run
# When halted or on error:
(gdb) bt
```
