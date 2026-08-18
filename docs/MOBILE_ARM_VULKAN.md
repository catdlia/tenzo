# Mobile & Edge Deployment Guide: ARM NEON, Android, Termux & Vulkan

## 1. Overview

Tenzo is designed from the ground up for edge devices. By compiling models to pure C++ standalone binaries with **zero runtime dynamic allocations**, Tenzo runs natively on Android devices via **Termux** or direct Native Development Kit (NDK) builds.

---

## 2. Target Hardware Capabilities & Benchmarks

### 2.1 MediaTek Dimensity 9400+ (e.g., Samsung Galaxy Tab S11)
* **CPU:** ARMv9.2 Architecture
  - 1x Cortex-X925 (Super core @ 3.63 GHz)
  - 3x Cortex-X4 (Performance cores @ 3.3 GHz)
  - 4x Cortex-A720 (Efficiency cores @ 2.4 GHz)
  - **Vector Extensions:** ARM NEON, SVE2 (Scalable Vector Extension 2), DotProd (`sdot`, `udot`), I8MM (`vmmlaq_s32`).
* **GPU:** ARM Immortalis-G925 (12 cores)
  - **Vulkan Support:** Vulkan 1.3 with Subgroup Operations (`VK_KHR_shader_subgroup_extended_types`).
* **Verified Benchmark Speed:** **23.83 tok/sec** on 30-layer BitNet-b1.58-2B-4T (8 threads, NEON DotProd).

### 2.2 Qualcomm Snapdragon 778G (e.g., Poco X5 Pro 5G)
* **CPU:** ARMv8.2-A Architecture
  - 4x Cortex-A78 (Kryo 670 Gold @ 2.4 GHz)
  - 4x Cortex-A55 (Kryo 670 Silver @ 1.8 GHz)
  - **Vector Extensions:** ARM NEON, `dotprod` (`sdot`, `udot`).
* **GPU:** Qualcomm Adreno 642L
  - **Vulkan Support:** Vulkan 1.1 / 1.2 with 16-bit float storage and push constants.

---

## 3. One-Command Running via Termux (Android)

You can build and run Tenzo directly inside Termux on any ARM64 Android device without root.

### 3.1 Termux Setup (Run Once)
```bash
pkg update && pkg install -y git clang cmake ninja make python vulkan-loader-generic
```

### 3.2 Clone & Build Native Runtime
```bash
git clone https://github.com/catdlia/tenzo.git && cd tenzo
cmake -B build-termux -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DTENZO_STANDALONE_RUNTIME=ON
ninja -C build-termux tenzo-inference tenzo-diag
```

### 3.3 Run Hardware Diagnostic Suite
```bash
./build-termux/tenzo-diag models/bitnet-1.58b
```

### 3.4 Single-Command Interactive Chat
```bash
./build-termux/tenzo-inference -m models/bitnet-1.58b --kv-quant int8_fused
```

---

## 4. Mobile Hardware Acceleration Strategy

### 4.1 ARM NEON / SVE2 / I8MM Vector Micro-Kernels
* **NEON SDOT Fast Path:** 4-way INT8 dot product accumulation using `vdotq_s32`:
  ```c
  int32x4_t acc = vdotq_s32(vdupq_n_s32(0), act_vec_i8, weight_vec_i8);
  ```
* **I8MM Matrix Multiplication:** Matrix tile compute on Dimensity 9400+ via `vmmlaq_s32` computing a $2 \times 8 \times 2$ matrix multiply in a single cycle.
* **SIMD Compatibility Layer (`simd_arm_compat.h`):** 460+ lines of vector mappings translating x86 AVX2 intrinsics directly into ARM NEON assembly.

### 4.2 Vulkan 1.3 Compute on Mobile GPUs
* Tenzo's Vulkan backend utilizes 16-bit Push Constants and Subgroup Shuffles (`gl_SubgroupInvocationID`) to eliminate global memory barriers on mobile unified memory (UMA) architectures.
