#!/bin/bash
# ==============================================================================
# Tenzo Compiler: Automated Build & Setup for Android (Termux)
# Target Architectures: ARMv8.2+ (Snapdragon 778G) / ARMv9.2 (Dimensity 9400+)
# ==============================================================================

set -e

ANSI_GREEN="\033[1;32m"
ANSI_CYAN="\033[1;36m"
ANSI_YELLOW="\033[1;33m"
ANSI_RESET="\033[0m"

echo -e "${ANSI_CYAN}"
echo "  _____ _____ _   _ _____ ____  "
echo " |_   _| ____| \ | |__  / __ \ "
echo "   | | |  _| |  \| | / / |  | |"
echo "   | | | |___| |\  |/ /| |__| |"
echo "   |_| |_____|_| \_/____\____/ "
echo -e " ⚡ Tenzo Android Native Build Tool (Termux ARM64)${ANSI_RESET}\n"

# 1. Update and install required dependencies in Termux
echo -e "${ANSI_YELLOW}📦 [1/4] Checking and installing Termux build tools...${ANSI_RESET}"
pkg update -y
pkg install -y git clang cmake ninja make python ndk-sysroot libomp vulkan-loader-generic libllvm-static

# 2. Setup Build Directory
BUILD_DIR="build-termux"
echo -e "${ANSI_YELLOW}⚙️  [2/4] Configuring CMake for ARM NEON / DotProd...${ANSI_RESET}"

mkdir -p "${BUILD_DIR}"

# Detect CPU features for optimal compilation
ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"
if grep -q "sve2" /proc/cpuinfo 2>/dev/null; then
    echo -e "${ANSI_GREEN}🚀 SVE2 detected (Dimensity 9400+ / Cortex-X925)! Enabling ARMv9 flags.${ANSI_RESET}"
    ARCH_FLAGS="-march=armv9-a+sve2+i8mm+fp16"
elif grep -q "asimddp" /proc/cpuinfo 2>/dev/null; then
    echo -e "${ANSI_GREEN}🚀 ASIMD Dot Product detected (Snapdragon 778G / Cortex-A78)!${ANSI_RESET}"
    ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"
fi

cmake -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_FLAGS="${ARCH_FLAGS} -O3 -ffast-math -fopenmp" \
    -DCMAKE_C_FLAGS="${ARCH_FLAGS} -O3 -ffast-math -fopenmp"

# 3. Compile Standalone Inference Engine
echo -e "${ANSI_YELLOW}🔨 [3/4] Compiling tenzo-inference with Ninja...${ANSI_RESET}"
ninja -C "${BUILD_DIR}" tenzo_runtime_static tenzo-inference

echo -e "${ANSI_GREEN}✅ [4/4] Build complete! Binary created at: ${BUILD_DIR}/tenzo-inference${ANSI_RESET}\n"

echo -e "${ANSI_CYAN}🚀 To start interactive chat:${ANSI_RESET}"
echo -e "   python3 scripts/tenzo_cli.py chat\n"
