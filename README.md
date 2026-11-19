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
3. **Official Reference Outperformance (UNVERIFIED, historical):** an earlier run reported **20.32 tok/sec** on an Intel Core i3-1215U (15W TDP) and **1.65x** over Microsoft BitNet.cpp. It was not reproduced under controlled conditions and the BitNet.cpp build used for comparison is not numerically trustworthy; see [Measured (2026-11-19/20)](#measured-2026-11-1920) for verified numbers (no clean speed comparison yet).
4. **Heterogeneous Multi-Backend Runtime:** Compiles and runs natively on x86_64 (AVX2/VNNI), ARM64 (NEON/DotProd/SVE2/I8MM), RISC-V (RVV), with GPU acceleration via Vulkan compute, CUDA, and ROCm translation layers.

---

## 🏆 Benchmark: Tenzo Native Engine vs Microsoft BitNet.cpp

> **UNVERIFIED / historical.** The block below is a pre-recovery measurement with no raw logs; do not cite it. Verified numbers: [Measured (2026-11-19/20)](#measured-2026-11-1920).

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
| **BitNet TL1** | **1.58-bit** | Ternary weights $\{-1, 0, 1\}$ with dual-element 4-bit LUT packing | ⚠️ Runs end-to-end; PPL within 0.4% of an fp32 HF-semantics reference; strict logit parity gate not met (see Measured 2026-11-19/20) |
| **GGUF (llama.cpp)** | **4-bit / 8-bit** | Block quantization (`Q4_0`, `Q8_0`) with FP16 scale factors | ⚠️ Unverified (no end-to-end test) |
| **AutoGPTQ** | **4-bit** | Groupwise symmetric/asymmetric quantization with `qzeros` | ⚠️ Unverified (no end-to-end test) |
| **AutoAWQ** | **4-bit** | Activation-aware column-interleaved salient channel packing | ⚠️ Unverified (no end-to-end test) |
| **ExLlamaV2 (EXL2)**| **2.0–8.0 bpw**| Variable bitrate group dispatch with multi-rate masks | ⚠️ Unverified (no end-to-end test) |
| **INT4 / INT3** | **3-bit / 4-bit** | Dense symmetric integer packing for ultra-low memory footprints | ⚠️ Unverified (no end-to-end test) |

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

<a id="hybrid-build-environment"></a>
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

<a id="hardware-support"></a>
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

---

<a id="measured-2026-11-18"></a>
## Measured (2026-11-19/20)

Verified numbers only, after the semantic fixes listed in [CHANGELOG.md](CHANGELOG.md) (RoPE base from the model config,
8-bit HF ActQuant activations; commits `c0db56f`, `5012150`, `5b100cc`, `1fd1aa2`). One laptop (Intel Core i3-1215U, governor `powersave`).
Quality ground truth: an independent fp32 "golden" PyTorch forward written from the HF code
(`tools/reference/golden_bitnet.py`; bit-identical to HF transformers in bf16) and HF transformers bf16 itself.

| What | Result | Source |
|---|---|---|
| WikiText-2 PPL (16 x 511 tokens), ratio to golden fp32 (19.066) | fp32 KV 19.124 (1.0030); int8_fused 19.141 (1.0039); tl1_fused 54.94 (2.88); popcount_fused 207.7 (10.9) | [ppl_kv_v2.md](results/2026-11-19_overnight/ppl_kv_v2.md) |
| Kobza Ukrainian PPL (8 x 511), ratio to golden fp32 (141.9) | fp32 KV 142.0 (1.0007); int8_fused 142.2 (1.0020); tl1_fused 1258 (8.87); popcount_fused 4162 (29.3) | [ppl_kv_v2_uk.md](results/2026-11-19_overnight/ppl_kv_v2_uk.md) |
| Valid bitnet.cpp reference (microsoft/BitNet@c17d1c5) | PPL 19.059 EN / 141.8 UK, greedy equal to HF | [bitnet_ref.md](results/2026-11-20_overnight/bitnet_ref.md) |
| Best ternary-KV PTQ scheme (no retraining, simulated) | ternary g32 + Hadamard + 4 sink + 32 recent tokens in INT8: +7.1% EN / +16.1% UK at 2.9 bits/element; no scheme <= 2.5 bits within 5% | [kv_ptq.md](results/2026-11-19_overnight/kv_ptq.md), [kv_ptq2.md](results/2026-11-20_overnight/kv_ptq2.md) |
| Parity, first-step logits, 7 prompts (gates cos > 0.999 and max_abs <= 0.25, unchanged) | vs golden fp32: cos gate 6/7, top-1 6/7 (fp32 KV) and 7/7 (int8), **max_abs gate 0/7** (0.54-2.27). The two references fail the same gate against each other: golden vs HF max_abs 0.53-1.63 | [parity_v2.md](results/2026-11-19_overnight/parity_v2.md), [golden_vs_hf_logits.txt](results/2026-11-20_overnight/part2/logs/golden_vs_hf_logits.txt) |
| GEMV kernels | I2_S (maddubs) and two-table LUT produce bit-identical outputs; speed not yet measured on a clean machine | [i2s_bitwise_identity.txt](results/2026-11-20_overnight/logs/i2s_bitwise_identity.txt) |
| `ternary_rot` KV mode (Hadamard rotation + ternary g32 + 4 sink / 32 recent tokens in INT8), engine | WikiText-2 PPL ratio to golden 1.075 [1.059, 1.092], Kobza-UK 1.167 [1.126, 1.210]; matches the simulator's S3+s4r32 (KV-induced degradation 1.0711 vs 1.0709); KV 14,400 B/token (+1.4 MB) vs 39,600 for int8_fused | [ternary_rot_report.md](results/2026-11-20_overnight/part2/ternary_rot_report.md) |
| Calibrated ternary KV (learned clip multipliers, no weight change; simulated) | S3+s4r32 PPL ratio to golden EN / UK: no calibration 1.071 / 1.151; calibrated on EN 1.040 / 1.126; on UK 1.049 / 1.117. The Ukrainian excess at equal window difficulty (c_UK ~ +0.11 nats/token) is unchanged | [kv_calib.md](results/2026-11-20_overnight/part2/kv_calib.md) |
| Engine memory (BitNet-2B-4T, int8_fused, ctx 2048) | VmHWM 2761 -> 935 MB (peak == steady RSS); ctx 8192: 3225 -> 1166 MB | [rss_report.md](results/2026-11-20_overnight/part2/rss_report.md) |
| Batched prefill (`--prefill batched`, CPU + I2_S) | next-token logits bit-identical to token-by-token on 7 prompts x 3 KV modes; speed not measured (queued for a clean machine) | [batched_prefill_report.md](results/2026-11-20_overnight/part2/batched_prefill_report.md) |

No speed claim is made here: tonight's machines were never idle (see the deferred queue `scripts/pending_benchmarks.sh`).

## Measured (2026-11-18) — SUPERSEDED

> **Superseded, do not cite.** These numbers were produced by an engine with two semantic bugs: RoPE base hard-coded to 10000 instead of the
> model's 500000 (fixed in `c0db56f`) and activations quantized to +-31 levels instead of HF's 127 (fixed in `5012150`). All quality numbers
> below (PPL, parity) are invalid; the speed comparison used the old 6-bit kernel and a broken bitnet.cpp build. Current numbers:
> [Measured (2026-11-19/20)](#measured-2026-11-1920).


Everything below was produced on one laptop (Intel Core i3-1215U, governor `powersave`, no sudo to change it) by the scripts and logs under
[`results/2026-11-18_overnight/`](results/2026-11-18_overnight/). It is a snapshot of branch `overnight/2026-11-18`, not a release claim.
The ground truth for quality is HF transformers bf16 on the same checkpoint; the local bitnet.cpp build turned out not to be a valid reference.

| What | Result | Source |
|---|---|---|
| Parity vs HF, first-step logits (7 prompts, kv int8_fused) | top-1 equal on 7/7, cosine 0.91-0.999, max abs diff 2.7-6.3. **Gates (cos > 0.999, max_abs <= 0.25) are NOT met** | [parity_summary.md](results/2026-11-18_overnight/parity_summary.md) |
| WikiText-2 perplexity (16 x 511 tokens) | HF bf16 **19.09**; Tenzo fp32 KV **23.13** (+21%); int8_fused **23.22** (CI of difference contains 0); tl1_fused **59.42**; popcount_fused **174.98** | [ppl_kv.md](results/2026-11-18_overnight/ppl_kv.md) |
| Ukrainian slice perplexity (8 x 511 tokens) | HF 142; Tenzo fp32 209; int8_fused 207; tl1_fused 2186; popcount_fused 9076 | [ppl_kv.md](results/2026-11-18_overnight/ppl_kv.md) |
| Local bitnet.cpp i2_s build | WikiText-2 PPL 102, first-step logits uncorrelated with HF: not usable as a reference | [ppl_kv.md](results/2026-11-18_overnight/ppl_kv.md), [parity_summary.md](results/2026-11-18_overnight/parity_summary.md) |
| Attention micro-benchmark (synthetic Gaussian KV, 1 thread) | zero-copy GQA (b vs a): 0.26-0.38x time; ternary packed KV vs FP32 direct (d vs b): 1.9-4.8x **slower**; vs INT8 KV (d vs c): 4.4-6.3x slower; ternary output cosine to FP32 about 0.77-0.79 | [attn_ablation.md](results/2026-11-18_overnight/attn_ablation.md) |
| End-to-end vs bitnet.cpp (4 threads, prompt 16/128/512 + 128 tokens) | prompt+gen tok/s: Tenzo 24.5 / 28.1 / 31.1 (int8 KV), bitnet.cpp f16 20.6 / 29.1 / 41.5; decode after prompt 21-24 vs 19 tok/s; Tenzo prefill ~35 vs ~66-70 tok/s; peak RSS ~2.7 GB vs ~1.2 GB | [e2e_fair.md](results/2026-11-18_overnight/e2e_fair.md) |

The historical figures above (20.32 tok/s, 1.65x) are **not** supported by these runs: Tenzo's decode is about 1.1-1.2x of this bitnet.cpp build, prefill is about half, and
the longer the prompt the more the batched prefill of llama.cpp wins. `laptop_metrics.txt` (the 80x "memory reduction") comes from a formula that was later found to be wrong and is kept only as history.
Full list of what these data do and do not support: [OVERNIGHT_REPORT.md](results/2026-11-18_overnight/OVERNIGHT_REPORT.md).

## 📚 Documentation & Deep Dives

- [Comprehensive Documentation](DOCUMENTATION.md)
- [Compiler Architecture](docs/ARCHITECTURE.md)
- [Quantization Matrix & Bit-Packing](docs/QUANTIZATION.md)
- [Developer CLI & Testing Guide](src/tests/CLI_GUIDE.md)
- [Vertical Scaling Research](docs/VERTICAL_SCALING_RESEARCH.md)
- [Mobile & Vulkan Guide](docs/MOBILE_ARM_VULKAN.md)
- [CPU Optimization Pipeline](documentation/CPU_PIPELINE.md)
- [Frontend Exporter Ecosystem](documentation/FRONTEND.md)
- [Hardware Abstraction Layer (HAL)](documentation/HARDWARE_HAL.md)
- [Runtime & Execution Engine](documentation/RUNTIME.md)
- [Project Roadmap](docs/ROADMAP.md)
- [Changelog](CHANGELOG.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)

---

## 📄 License

Tenzo is released under the [Apache 2.0 License](LICENSE).
