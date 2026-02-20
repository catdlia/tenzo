#!/bin/bash
# generate_and_compile_microkernel.sh - Complete pipeline to generate MLIR micro-kernel
#
# This script:
# 1. Generates MLIR micro-kernel using tenzo-cli
# 2. Lowers it to LLVM IR
# 3. Compiles to object file
# 4. Checks the assembly
# 5. Rebuilds micro_bench with the MLIR kernel

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════╗"
echo "║  🚀 MLIR MICRO-KERNEL COMPILATION PIPELINE             ║"
echo "║  Goal: Match 124 GFLOPS from hand-written intrinsics   ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build_microkernel"
SRC_TESTS="${PROJECT_ROOT}/src/tests"

echo "📁 Project root: $PROJECT_ROOT"
echo "📁 Build dir: $BUILD_DIR"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ============================================================
# STEP 1: Generate MLIR micro-kernel using tenzo-cli
# ============================================================
echo "╔════════════════════════════════════════════════════════╗"
echo "║  STEP 1: Generate MLIR Micro-Kernel                   ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

if [ ! -f "$PROJECT_ROOT/cmake-build-release-docker/tenzo-cli" ]; then
    echo "❌ tenzo-cli not found. Please build it first:"
    echo "   docker compose run --rm dev ninja -C /app/cmake-build-debug tenzo-cli"
    exit 1
fi

echo "🔧 Running: tenzo-cli generate-microkernel"
"$PROJECT_ROOT/cmake-build-release-docker/tenzo-cli" generate-microkernel > micro_kernel_generated.log 2>&1 || true

# Check if file was generated
if [ -f "generated_micro_kernel.mlir" ]; then
    echo "✅ Generated: generated_micro_kernel.mlir"
    echo ""
    echo "📝 First 30 lines of generated MLIR:"
    head -30 generated_micro_kernel.mlir
    echo "..."
    echo ""
else
    echo "⚠️  No MLIR file generated, using pre-written micro_kernel.mlir"
    cp "$SRC_TESTS/micro_kernel.mlir" generated_micro_kernel.mlir
fi

# ============================================================
# STEP 2: Lower MLIR to LLVM IR
# ============================================================
echo "╔════════════════════════════════════════════════════════╗"
echo "║  STEP 2: MLIR → LLVM IR                                ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

if ! command -v mlir-opt &> /dev/null; then
    echo "❌ mlir-opt not found. Using Docker..."
    exit 1
fi

echo "🔄 Lowering Vector Dialect → LLVM..."
mlir-opt generated_micro_kernel.mlir \
    --convert-vector-to-llvm \
    --convert-memref-to-llvm \
    --convert-func-to-llvm \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-arith-to-llvm \
    --convert-index-to-llvm \
    --reconcile-unrealized-casts \
    -o micro_kernel_llvm.mlir

echo "✅ Generated: micro_kernel_llvm.mlir"
echo ""

echo "🔄 MLIR LLVM Dialect → LLVM IR..."
mlir-translate --mlir-to-llvmir micro_kernel_llvm.mlir -o micro_kernel.ll

echo "✅ Generated: micro_kernel.ll"
echo ""

# ============================================================
# STEP 3: Compile LLVM IR to Object File
# ============================================================
echo "╔════════════════════════════════════════════════════════╗"
echo "║  STEP 3: LLVM IR → Object File                         ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

if ! command -v llc &> /dev/null; then
    echo "❌ llc not found"
    exit 1
fi

echo "🔧 Compiling with aggressive optimizations..."
llc micro_kernel.ll \
    -O3 \
    -mcpu=native \
    -mattr=+avx2,+fma \
    -filetype=obj \
    -o micro_kernel.o

echo "✅ Generated: micro_kernel.o"
echo ""

# ============================================================
# STEP 4: Inspect Assembly
# ============================================================
echo "╔════════════════════════════════════════════════════════╗"
echo "║  STEP 4: Assembly Inspection                           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Generate disassembly
objdump -d micro_kernel.o > micro_kernel.asm

echo "🔍 Checking for FMA instructions..."
FMA_COUNT=$(grep -c "vfmadd" micro_kernel.asm || echo "0")
BROADCAST_COUNT=$(grep -c "vbroadcastss" micro_kernel.asm || echo "0")
YMM_COUNT=$(grep -c "ymm" micro_kernel.asm || echo "0")

echo "   - vfmadd instructions: $FMA_COUNT"
echo "   - vbroadcastss instructions: $BROADCAST_COUNT"
echo "   - YMM register usage: $YMM_COUNT occurrences"
echo ""

if [ "$FMA_COUNT" -lt 100 ]; then
    echo "⚠️  WARNING: Only $FMA_COUNT FMA instructions found!"
    echo "   Expected: ~3000+ (6 rows × 2 vecs × 256 K)"
fi

if [ "$YMM_COUNT" -lt 10 ]; then
    echo "⚠️  WARNING: Very low YMM register usage!"
    echo "   Expected: Heavy use of ymm0-ymm15"
fi

echo "📝 Sample assembly (first 50 FMAs):"
grep -A2 "vfmadd" micro_kernel.asm | head -150
echo "..."
echo ""

# ============================================================
# STEP 5: Statistics
# ============================================================
echo "╔════════════════════════════════════════════════════════╗"
echo "║  SUMMARY                                               ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

ls -lh micro_kernel.o
echo ""
echo "📊 Object file size: $(stat -f%z micro_kernel.o 2>/dev/null || stat -c%s micro_kernel.o) bytes"
echo "📊 FMA instructions: $FMA_COUNT"
echo "📊 Expected for 6x16x256: 3072 FMAs"
echo ""

if [ "$FMA_COUNT" -ge 2500 ]; then
    echo "✅ EXCELLENT! Kernel looks well optimized."
elif [ "$FMA_COUNT" -ge 1000 ]; then
    echo "✅ GOOD! Kernel is reasonably optimized."
else
    echo "⚠️  Optimization may be suboptimal."
fi

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  NEXT STEPS                                            ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "1. Rebuild micro_bench with MLIR kernel:"
echo "   cd build && cmake .. -DUSE_MLIR_KERNEL=ON && make micro_bench"
echo ""
echo "2. Run benchmark:"
echo "   ./build/micro_bench"
echo ""
echo "3. Compare with intrinsics baseline (124 GFLOPS)"
echo ""

