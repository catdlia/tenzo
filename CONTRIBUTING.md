# Contributing to Tenzo

Thank you for your interest in contributing to **Tenzo**! We welcome contributions ranging from MLIR passes and SIMD micro-kernels to documentation and bug fixes.

---

## 1. Development Setup & Build Instructions

Tenzo utilizes a hybrid build environment:
* **Compilation:** Managed via CMake, Ninja, and Clang in Docker container or local environment (`cmake-build-debug`).
* **Execution & Testing:** Executed inside Docker or natively with AVX2/NEON.

### Quick Build Command
```bash
# Build standalone C++ SDK and inference CLI
make build-sdk
```

---

## 2. Code Organization

* `src/dialect/`: Tenzo MLIR dialect definition (`TenzoOps.td`), TableGen rules, and type definitions.
* `src/passes/`: Transformation and lowering passes (`LinalgLowering.cpp`, `BitLinearToLinalg`, `Bufferization`).
* `src/runtime/`: High-performance execution engine (`TenzoEngine.cpp`, `TenzoEngine.h`), zero-allocation KV-cache, and AVX2/NEON micro-kernels.
* `tenzo-frontend/`: Python exporters for GGUF, SafeTensors, PyTorch FX, GPTQ, AWQ, and EXL2.
* `scripts/`: CLI application (`tenzo_cli.py`), benchmark suites, and download utilities.
* `docs/`: In-depth architecture and quantization documentation.

---

## 3. Pull Request Guidelines

1. **Create a Feature Branch:** Branch off `master` with a descriptive name (e.g., `feature/arm-neon-kernels` or `fix/paged-attn-index`).
2. **Code Style & Formatting:** Follow LLVM coding conventions for C++ and PEP 8 for Python. Use `clang-format` on C++ files.
3. **Tests:** Ensure all unit tests and benchmark suites pass:
   ```bash
   python3 scripts/tenzo_cli.py list
   make test
   ```
4. **Documentation:** Add clear comments for non-trivial optimizations, memory layouts, or vector intrinsics in English.

---

## 4. Community & Support

* For bugs and feature requests, please open an issue using the provided templates.
* Discussions take place on GitHub Discussions and issues.
