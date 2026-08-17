# Mobile & Edge Deployment Guide: ARM NEON, Android, Termux & Vulkan

## 1. Overview

Tenzo is designed from the ground up for edge devices. By compiling models to pure C++ standalone binaries with **zero runtime dynamic allocations**, Tenzo can run natively on Android devices via **Termux** or direct Native Development Kit (NDK) builds.

---

## 2. Target Hardware Capabilities

### 2.1 MediaTek Dimensity 9400+ (e.g., Galaxy Tab S11)
* **CPU:** ARMv9.2 Architecture
  - 1x Cortex-X925 (Super core @ 3.63 GHz)
  - 3x Cortex-X4 (Performance cores @ 3.3 GHz)
  - 4x Cortex-A720 (Efficiency cores @ 2.4 GHz)
  - **Vector Extensions:** ARM NEON, SVE2 (Scalable Vector Extension 2), I8MM (Matrix Multiplication instructions: `smmla`, `ummla`, `usmmla`).
* **GPU:** ARM Immortalis-G925 (12 cores)
  - **Vulkan Support:** Vulkan 1.3 with Ray Tracing & Subgroup Operations (`VK_KHR_shader_subgroup_extended_types`).

### 2.2 Qualcomm Snapdragon 778G (e.g., Poco X5 Pro 5G)
* **CPU:** ARMv8.2-A Architecture
  - 4x Cortex-A78 (Kryo 670 Gold @ 2.4 GHz)
  - 4x Cortex-A55 (Kryo 670 Silver @ 1.8 GHz)
  - **Vector Extensions:** ARM NEON, `dotprod` (`sdot`, `udot`).
* **GPU:** Qualcomm Adreno 642L
  - **Vulkan Support:** Vulkan 1.1 / 1.2 with 16-bit float storage and push constants.

---

## 3. One-Command Running via Termux (Android)

You can build and run Tenzo directly inside Termux on any ARM64 Android device.

### 3.1 Termux Setup (Run Once)
```bash
pkg update && pkg install -y git clang cmake ninja make python ndk-sysroot vulkan-loader-generic libomp
```

### 3.2 Clone & Build Native Runtime
```bash
git clone https://github.com/catdlia/tenzo.git && cd tenzo
cmake -B build-termux -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DTENZO_ENABLE_ARM_NEON=ON
ninja -C build-termux tenzo-inference
```

### 3.3 Single-Command Interactive Chat
```bash
python3 scripts/tenzo_cli.py chat -m models/gguf_tinyllama/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
```

---

## 4. Mobile Hardware Acceleration Strategy

### 4.1 ARM NEON / SVE2 / I8MM Vector Micro-Kernels
* **NEON SDOT Fast Path:** 4-way INT8 dot product accumulation using `vdotq_s32`:
  ```c
  int32x4_t acc = vdotq_s32(vdupq_n_s32(0), act_vec_i8, weight_vec_i8);
  ```
* **I8MM Matrix Multiplication:** Matrix tile compute on Dimensity 9400+ via `vmmlaq_s32` computing a $2 \times 8 \times 2$ matrix multiply in a single cycle.

### 4.2 Vulkan 1.3 Compute on Mobile GPUs
* Tenzo's Vulkan backend utilizes 16-bit Push Constants and Subgroup Shuffles (`gl_SubgroupInvocationID`) to eliminate global memory barriers on mobile unified memory (UMA) architectures.
