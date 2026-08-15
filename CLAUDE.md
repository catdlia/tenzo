# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Tenzo is an open-source heterogeneous tensor compiler built on MLIR/LLVM 21. A high-level Tenzo MLIR dialect (`tenzo.matmul`, `tenzo.conv2d`, `tenzo.attention`, `tenzo.quantize`, …) is lowered through MLIR (Linalg → Bufferization → MemRef → LLVM) and JIT-compiled to native code for x86 AVX2 (production) and Vulkan/SPIR-V (proof-of-concept). The active workstream is a **1.58-bit quantized LLM inference path** (BitNet-b1.58-2B): PyTorch FX export → Tenzo MLIR → C++ runtime generation. See `README.md`, `PROJECT_STATUS.md`, `DOCUMENTATION.md`, and `implementation_plan.md` for the vision and honest gap analysis.

## Build & run (everything goes through Docker)

The canonical environment is the `dev` Docker container (Ubuntu 24.04 + LLVM/MLIR 21 + Vulkan SDK, Intel iGPU passthrough, `cpuset 0-3`). **Compile inside the container; never assume a host toolchain works.** The host (Arch) is only used to drive Docker/Hetzner.

```bash
make build          # remote compile on a Hetzner cloud box via remote_build.sh + distcc (default)
make build-local    # fallback: docker compose run --rm dev ninja -C /app/cmake-build-debug tenzo-cli
make dev            # interactive bash shell inside the dev container

# From inside the container (or via `docker compose run --rm dev ...`):
cmake -B/app/cmake-build-debug -S/app -GNinja -DCMAKE_BUILD_TYPE=Debug   # one-time configure (make configure)
ninja tenzo-cli                                                         # incremental build
ninja micro_bench                                                       # isolated AVX2 micro-kernel target (separate executable)
```

Two CMake targets: `tenzo-cli` (the compiler + runtime driver) and `micro_bench` (standalone AVX2/FMA kernel benchmark, built with `-mavx2 -mfma -ffast-math -fno-lto`, optionally links an MLIR-compiled kernel via `-DUSE_MLIR_KERNEL=ON`).

Build flags are aggressive and host-specific: `-march=native -ffast-math -funroll-loops`, LLD linker, ccache. LTO is **explicitly disabled** (caused an undefined-symbol:main linker error) — do not re-enable without checking.

Optional deps auto-detected at configure: Vulkan (`-DTENZO_HAS_VULKAN`), OpenMP, OpenBLAS (`TENZO_HAS_OPENBLAS` define enables GEMM comparison benchmarks).

## Running tenzo-cli (single entry point, subcommand dispatch)

`src/main.cpp` dispatches on `argv[1]` (mode). There is no unit-test framework — "tests" are subcommands that compile an MLIR pipeline and report GFLOPS/correctness. Key modes:

| Mode | Purpose |
|------|---------|
| `cpu`, `large`, `parallel` | MatMul benchmarks (512² / 1024² / multi-threaded) |
| `gemm-e2e` | Full optimized CPU path: packing + 5-loop macro-kernel + explicit AVX2 micro-kernel (primary perf test) |
| `microkernel`, `blis`, `packing` | Isolated stages (packing must finish in <1s or there's an infinite loop in `PackingPass.cpp`) |
| `conv2d` | Conv2D pipeline test |
| `gpu`, `gpu-bench`, `hybrid-bench` | Vulkan/SPIR-V path (Intel iGPU) |
| `dynamic`, `bridge`, `validate` | Dynamic shapes, zero-copy bridge, end-to-end math validation |
| `generate` | **LLM inference**: `-m <model_dir> -p "<prompt>" -n <max_tokens> -t/--temp <temp> --top-p <p>`. Default model format consumed by the runtime lives in `export_output_bitnet/` (`model.mlir`, `weights.bin`, `tokenizer.vocab`). |
| `test` / `all` | Run CPU + Conv2D + GPU smoke tests |

Run a single mode: `docker compose run --rm dev /app/cmake-build-debug/tenzo-cli <mode> [args]` — or build then invoke the binary directly inside `make dev`. `scripts/run_tests.sh` is a thin wrapper for the CI-style suite.

## Python frontend (`tenzo-frontend/`)

```bash
make frontend-setup           # create .venv, pip install -r tenzo-frontend/requirements.txt
make onnx-convert MODEL=m.onnx OUT=out.mlir
.venv/bin/python3 tenzo-frontend/export_bitnet.py            # exports BitNet-b1.58-2B → export_output_bitnet/
```

Two ingestion paths share an `mmap`-able weight format (`weights.bin` + offset/stride pointers in `model.mlir`, deliberately zero-copy into the C++ runtime):
- **ONNX path** (`onnx_to_mlir.py` + `weight_utils.py`) — generic ONNX → Tenzo dialect.
- **PyTorch FX path** (`fx_to_mlir.py`, `export_llama.py`, `export_bitnet.py`, `qat.py`) — the active direction for QAT 1.58-bit models; emits `tenzo.quantize`/`dequantize`, `tenzo.attention`, `tenzo.rotary`, `tenzo.kv_cache`, `tenzo.rms_norm`, `tenzo.embedding` ops and a BPE `tokenizer.vocab`.

## Architecture (the parts that span multiple files)

**Compilation pipeline (CPU):** `src/passes/` — `FusionPass` (fuses MatMul+Bias+ReLU) → `LinalgLowering` (Tenzo→Linalg, the biggest file ~58k and central lowering rules for quantize/attention/kv-cache) → `Bufferization` (one-shot) → `TransformStrategy` / `MacroKernelPass` (5-loop GotoBLAS cache-blocking) → `PackingPass`+`PackingKernels` (BLIS-style packing, needs `AffineMap` support for stride views) → `ExplicitMicroKernelPass` (6×16 register-tiled FMA micro-kernel; currently `MR=6, NR=16`, with a `FusedTernaryKernel` path for 2-bit) → `LLVMLowering` → JIT via `MLIRExecutionEngine`. Passes are registered in `Passes.h` and added to the `OpPassManager` in pipeline functions (`runGEMMEndToEndBenchmark`, etc.).

**Hardware HAL:** `src/context/` — `HardwareProfile` (abstract ISA/SIMD-width/cache/micro-kernel params) with `X86HardwareProfile` (sysfs P/E-core topology, CPUID, AVX detection — production) and `ARMHardwareProfile` (stub). `AutoTuner` tunes `MR`/`NR`/tiling. `MicroKernelParams` from the profile flow into `addPackingPass`/`addExplicitMicroKernelPass` — tiling choices belong in the hardware profile, not hardcoded in passes. `TenzoContext` owns the `MLIRContext` + dialect registration.

**Runtime (inference/generation):** `src/runtime/` — `ExecutionContext` (JIT + Tensor I/O), `MemRefUtils.h` (C++↔MLIR MemRef descriptor, dynamic shapes, stride views), `ArenaPlanner.h` (memory planning, still incomplete), `ThreadPool`, `Sampler` (`runGenerationTest`), `Tokenizer` (BPE), `KVCacheManager`, `VulkanRuntime`. The `generate` path ties these together end-to-end.

**GPU path:** `src/passes/gpu/GPULowering.cpp` lowers Linalg → GPU → SPIR-V; `VulkanRuntime` executes it. Proof-of-concept only.

**Dialect:** `src/dialect/TenzoOps.td` is ODS/TableGen. Changing ops requires regenerating `TenzoOps.h.inc`/ `.cpp.inc` / `TenzoDialect.*.inc` (handled by the `TenzoOpsIncGen` custom target; build depends on it). Keep the TD, the lowering rules in `LinalgLowering.cpp`, and the runtime's expectations in sync when adding ops.

## Conventions

- Comments, README, and many docs are **mixed Ukrainian/English** (e.g. `CLI_GUIDE.md`, `implementation_plan.md`). Match the language of the file you're editing when adding comments.
- The repo has many build dirs (`build`, `build_docker`, `build_e2e`, `build_micro`, `build_mlir`, `cmake-build-*`); only `cmake-build-debug` is wired through the Makefile/Docker defaults, others are scratch.
- `simple_mlp.mlir` / `simple_mlp.onnx` are sample inputs; `simple_mlp.onnx` is a 2-layer MLP used by `test_simple_mlp.py`.
- The `packed`/`gemm` modes hang indefinitely if a packing loop is wrong — always `timeout` long-running modes first when iterating.