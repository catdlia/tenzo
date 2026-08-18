# ⚡ Tenzo Compiler & AI Inference Engine

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-v1.1.0--beta-green.svg)](https://github.com/catdlia/tenzo/releases)
[![Build Status](https://img.shields.io/badge/Build-Hybrid_Remote_%2B_Local_Docker-brightgreen.svg)](#hybrid-build-environment)
[![Target Arch](https://img.shields.io/badge/Targets-x86__64_AVX2_%7C_ARM_NEON_%7C_Vulkan_%7C_CUDA_%7C_ROCm_%7C_RISC--V_RVV-orange.svg)](#hardware-support)

> **High-Performance Heterogeneous MLIR Compiler & Zero-Allocation Inference Engine for Sub-Byte and Quantized Large Language Models (BitNet 1.58-bit, GGUF, GPTQ, AWQ, EXL2).**

---

## 💡 Why Tenzo?

Modern quantized LLMs (1-bit, 3-bit, 4-bit) require specialized memory representations and micro-kernels that standard general-purpose compilers (XLA, PyTorch Inductor) cannot efficiently represent or vectorize. Proprietary runtimes (TensorRT-LLM, CoreML, OpenVINO) lock developers into single-vendor ecosystems.

**Tenzo** bridges this divide by providing:
1. **Universal Quantization Matrix:** Native MLIR dialect operations and micro-kernels for **BitNet 1.58b (TL1)**, **GGUF (Q4_0 / Q8_0)**, **AutoGPTQ**, **AutoAWQ**, **ExLlamaV2 (EXL2)**, **INT4**, and **INT3**.
2. **Zero-Allocation Execution:** The entire autoregressive decode loop across all transformer layers runs in pure C++ with zero intermediate heap allocations.
3. **Official Reference Outperformance:** Tenzo delivers **20.32 tok/sec** on consumer CPUs (Intel Core i3-1215U, 15W TDP), **1.65x faster than Microsoft BitNet.cpp**.
4. **Heterogeneous Multi-Backend Runtime:** Compiles and runs natively on x86_64 (AVX2/VNNI), ARM64 (NEON/DotProd/SVE2/I8MM), RISC-V (RVV), with GPU acceleration via Vulkan compute, CUDA, and ROCm translation layers.

---

## 🏆 Benchmark: Tenzo Native Engine vs Microsoft BitNet.cpp

*Benchmarked on an ultra-low-power consumer CPU: **Intel Core i3-1215U** (2 P-Cores + 4 E-Cores, 15W TDP).*  
*Model:* `BitNet-b1.58-2B-4T` *(30 Layers, Hidden Size 2560, Vocab Size 128,256).*

```
══════════════════════════════════════════════════════════════════════════════
📊 BENCHMARK COMPARISON: Microsoft BitNet.cpp vs Tenzo Native Engine
══════════════════════════════════════════════════════════════════════════════
Prompt: "In computer science, a compiler translates source code written in a high-level programming language into"

Metric                         | Microsoft BitNet.cpp | Tenzo Native Engine (v1.1.0-beta)
-------------------------------------------------------------------------------
Model Weights Format           | TL1 + INT8 LM        | TL1 + INT8 LM (328 MB)
KV-Cache Architecture          | Standard FP32/FP16   | Fused INT8 (314 MB for 8k ctx)
Generated Tokens               | 49                   | 50                  
Time To First Token (TTFT)     | 367.77 ms            | 660.92 ms           
Decode Speed (tok/sec)         | 12.31 tok/sec        | 20.32 tok/sec (🚀 1.65x FASTER)
Per-Token Decode Latency       | 81.23 ms             | 49.21 ms            
Effective Memory Bandwidth     | ~10.5 GB/s           | ~17.3 GB/s          
═══════════════════════════════════════════════════════════════════════════════
```

### Cross-Architecture Performance

| Device | Architecture | SIMD | Decode Speed |
|:-------|:-------------|:-----|:-------------|
| Intel Core i3-1215U (Docker) | x86_64 | AVX2 / VNNI / FMA | 14.01 tok/sec |
| Samsung Galaxy Tab S11 (Termux) | ARM64 / ARMv9.2 | NEON / DotProd | **23.83 tok/sec** |
| Poco X5 Pro 5G (Termux) | ARM64 / ARMv8.2 | NEON / DotProd | *TBD* |

---

## 📦 Supported Models & Quantization Formats

| Format / Scheme | Bit Width | Description | Status |
| :--- | :--- | :--- | :--- |
| **BitNet TL1** | **1.58-bit** | Ternary weights $\{-1, 0, 1\}$ with dual-element 4-bit LUT packing | ✅ Production |
| **GGUF (llama.cpp)** | **4-bit / 8-bit** | Block quantization (`Q4_0`, `Q8_0`) with FP16 scale factors | ✅ Production |
| **AutoGPTQ** | **4-bit** | Groupwise symmetric/asymmetric quantization with `qzeros` | ✅ Production |
| **AutoAWQ** | **4-bit** | Activation-aware column-interleaved salient channel packing | ✅ Production |
| **ExLlamaV2 (EXL2)**| **2.0–8.0 bpw**| Variable bitrate group dispatch with multi-rate masks | ✅ Production |
| **INT4 / INT3** | **3-bit / 4-bit** | Dense symmetric integer packing for ultra-low memory footprints | ✅ Production |

---

## 🏗️ Compiler Architecture

```text
               Hugging Face / GGUF / Safetensors Weights
                                   │
                         tenzo-frontend Exporters
                                   │
              tenzo.bitlinear_* / tenzo.rmsnorm / tenzo.attention
                                   │
                          Tenzo-to-Linalg Pass
                                   │
                  Zero-Allocation Bufferization (MemRef)
                                   │
                    Vectorization (vector.contract)
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
       LLVM Dialect                                 SPIR-V Dialect
   (x86 AVX2 / ARM NEON)                        (Vulkan 1.3 Compute)
            │                                             │
      Native Machine Code                           Compute Shaders
            │                                             │
      ┌─────┴─────┐                          ┌────────────┴────────────┐
      ▼           ▼                          ▼            ▼            ▼
   x86_64     ARM NEON                   Vulkan       CUDA TL       ROCm TL
  AVX2/VNNI  DotProd/SVE2             (native)    (translation)  (translation)
```

For complete architectural details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/QUANTIZATION.md](docs/QUANTIZATION.md).

---

## 🚀 Quick Start

### 1. Build the Project
```bash
# Build C++ Standalone SDK and Inference Binaries
make build-sdk
```

### 2. Interactive CLI Console
Launch the unified Tenzo AI console:
```bash
python3 scripts/tenzo_cli.py chat
```

Inside the console, use interactive slash commands:
- `/models` or `/list`: Show all available local models and formats.
- `/pull <hf_repo>`: Download and compile a model from Hugging Face.
- `/load <model_alias>`: Switch active model on the fly.
- `/kv <popcount_fused|tl1_fused|int8_fused|fp32>`: Switch KV-cache quantization mode.
- `/device <cpu|gpu|vulkan|cuda|rocm>`: Select execution backend.
- `/stats`: Display live memory and execution telemetry.
- `Ctrl+C`: Gracefully stop token generation or exit cleanly.

### 3. Running Single-Shot Inference
```bash
python3 scripts/tenzo_cli.py run -p "Explain quantum computing in three sentences:" -n 64
```

### 4. Hardware Diagnostics
Run the comprehensive hardware verification suite:
```bash
make diag
# or directly:
docker compose run --rm dev /app/cmake-build-debug/tenzo-diag /app/models/bitnet-1.58b
```

---

## 📱 Mobile & Edge Devices (ARM / Android / Termux)

Tenzo runs seamlessly on mobile devices via **Termux** or Android NDK:
- **Snapdragon 778G / 8 Gen 3 (Adreno GPU + ARM NEON):** Native `sdot`/`udot` integer pipelines.
- **MediaTek Dimensity 9400+ (Immortalis-G925 GPU + Cortex-X925):** ARM SVE2 & I8MM matrix instructions.

### Quick Mobile Setup
```bash
# In Termux:
pkg update && pkg install -y git clang cmake ninja make python vulkan-loader-generic
git clone https://github.com/catdlia/tenzo.git && cd tenzo
cmake -B build-termux -G Ninja -DCMAKE_BUILD_TYPE=Release -DTENZO_STANDALONE_RUNTIME=ON
ninja -C build-termux tenzo-inference tenzo-diag
```

See [docs/MOBILE_ARM_VULKAN.md](docs/MOBILE_ARM_VULKAN.md) for detailed mobile setup.

---

## 🔧 Heterogeneous Backend Architecture

Tenzo v1.1.0-beta introduces a multi-backend runtime HAL:

| Backend | Detection | Fallback | Operators |
|:--------|:----------|:---------|:----------|
| **x86_64 AVX2** | Native | — | BitLinear TL1, GEMM, RMSNorm, Attention |
| **ARM NEON/DotProd** | Native | — | BitLinear TL1, GEMM, RMSNorm, Attention |
| **RISC-V RVV** | `riscv_vector.h` | Scalar emulation | BitLinear TL1, GEMV, RMSNorm |
| **Vulkan Compute** | `libvulkan.so` | — | BitLinear TL1, GEMM FP32, RMSNorm |
| **CUDA** | `libcuda.so.1` | Vulkan Translation Layer | BitLinear TL1, GEMM FP32, RMSNorm |
| **ROCm** | `libamdhip64.so` | Vulkan Translation Layer | BitLinear TL1, GEMM FP32, RMSNorm |

The `MicroarchProfiler` automatically detects all available backends and SIMD capabilities at startup.

---

## 📚 Documentation & Deep Dives

- [Compiler Architecture](docs/ARCHITECTURE.md)
- [Quantization Matrix & Bit-Packing](docs/QUANTIZATION.md)
- [Vertical Scaling Research](docs/VERTICAL_SCALING_RESEARCH.md)
- [Mobile & Vulkan Guide](docs/MOBILE_ARM_VULKAN.md)
- [Project Roadmap](docs/ROADMAP.md)
- [Changelog](CHANGELOG.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)

---

## 📄 License

Tenzo is released under the [Apache 2.0 License](LICENSE).
