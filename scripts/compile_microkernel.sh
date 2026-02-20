#!/bin/bash
# compile_microkernel.sh - Compile MLIR micro-kernel to object file
#
# This script:
# 1. Compiles micro_kernel.mlir → micro_kernel.ll (LLVM IR)
# 2. Compiles micro_kernel.ll → micro_kernel.o (object file)
# 3. Generates disassembly for inspection
#
# Usage: ./scripts/compile_microkernel.sh
#
# Expected assembly:
#   - Long chains of vfmadd231ps
#   - Minimal mov instructions inside loop
#   - Uses ymm0-ymm15 registers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/src/tests"
BUILD_DIR="$PROJECT_DIR/build"

echo "╔════════════════════════════════════════════════════════╗"
echo "║  🔧 MICRO-KERNEL COMPILATION                           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Check tools
if ! command -v mlir-opt &> /dev/null; then
    echo "❌ mlir-opt not found. Please ensure MLIR is installed."
    exit 1
fi

if ! command -v mlir-translate &> /dev/null; then
    echo "❌ mlir-translate not found. Please ensure MLIR is installed."
    exit 1
fi

if ! command -v llc &> /dev/null; then
    echo "❌ llc not found. Please ensure LLVM is installed."
    exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "📁 Source: $SRC_DIR/micro_kernel.mlir"
echo "📁 Output: $BUILD_DIR/micro_kernel.o"
echo ""

# ============================================================
# STEP 1: MLIR → LLVM IR (lower all dialects)
# ============================================================
echo "🔄 Step 1: MLIR → LLVM IR..."

mlir-opt "$SRC_DIR/micro_kernel.mlir" \
    --convert-vector-to-llvm \
    --convert-memref-to-llvm \
    --convert-func-to-llvm \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-arith-to-llvm \
    --convert-index-to-llvm \
    --reconcile-unrealized-casts \
    -o micro_kernel_llvm.mlir

echo "   ✅ Generated micro_kernel_llvm.mlir"

# ============================================================
# STEP 2: MLIR LLVM → LLVM IR (.ll)
# ============================================================
echo "🔄 Step 2: MLIR LLVM → LLVM IR..."

mlir-translate --mlir-to-llvmir micro_kernel_llvm.mlir -o micro_kernel.ll

echo "   ✅ Generated micro_kernel.ll"

# ============================================================
# STEP 3: LLVM IR → Object File (.o)
# ============================================================
echo "🔄 Step 3: LLVM IR → Object file..."

# Use aggressive optimizations for maximum FMA throughput
llc micro_kernel.ll \
    -O3 \
    -mcpu=native \
    -mattr=+avx2,+fma \
    -filetype=obj \
    -o micro_kernel.o

echo "   ✅ Generated micro_kernel.o"

# ============================================================
# STEP 4: Generate disassembly for inspection
# ============================================================
echo ""
echo "🔍 Step 4: Generating disassembly..."

objdump -d -M intel micro_kernel.o > micro_kernel.asm

echo "   ✅ Generated micro_kernel.asm"
echo ""

# ============================================================
# STEP 5: Assembly analysis
# ============================================================
echo "════════════════════════════════════════════════════════"
echo "📊 ASSEMBLY ANALYSIS (micro_kernel.asm)"
echo "════════════════════════════════════════════════════════"
echo ""

# Count key instructions
FMA_COUNT=$(grep -c "vfmadd" micro_kernel.asm 2>/dev/null || echo "0")
BROADCAST_COUNT=$(grep -c "vbroadcast" micro_kernel.asm 2>/dev/null || echo "0")
MOV_COUNT=$(grep -c "vmov" micro_kernel.asm 2>/dev/null || echo "0")
LOAD_COUNT=$(grep -c "vmovups\|vmovaps" micro_kernel.asm 2>/dev/null || echo "0")

echo "  vfmadd* instructions: $FMA_COUNT (want 12+ per K iteration)"
echo "  vbroadcast instructions: $BROADCAST_COUNT (want 6 per K iteration)"
echo "  vmov* instructions: $MOV_COUNT (fewer is better)"
echo "  Vector loads (vmovups/vmovaps): $LOAD_COUNT"
echo ""

# Check for register spilling (stack operations inside loop)
SPILL_COUNT=$(grep -c "rsp\|rbp" micro_kernel.asm 2>/dev/null || echo "0")
echo "  Stack operations: $SPILL_COUNT (0 inside loop = no spilling)"
echo ""

# Check YMM register usage
YMM_REGS=$(grep -o "ymm[0-9]*" micro_kernel.asm | sort -u | wc -l)
echo "  Unique YMM registers used: $YMM_REGS (want 15-16)"
echo ""

# Show first few lines of the main loop
echo "════════════════════════════════════════════════════════"
echo "📜 MAIN LOOP SNIPPET:"
echo "════════════════════════════════════════════════════════"
# Find the loop (look for jmp or je instructions after FMA chain)
grep -A5 "vfmadd" micro_kernel.asm | head -30
echo "..."
echo ""

echo "════════════════════════════════════════════════════════"
echo "✅ COMPILATION COMPLETE"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Review micro_kernel.asm for register spilling"
echo "  2. Run: cmake -DUSE_MLIR_KERNEL=ON .. && make micro_bench"
echo "  3. Execute: ./micro_bench"
echo ""
echo "Expected performance: 45-55 GFLOPS"
echo ""

