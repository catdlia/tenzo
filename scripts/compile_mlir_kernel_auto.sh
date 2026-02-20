#!/bin/bash
set -x  # Debug mode

echo "====== MLIR MICRO-KERNEL COMPILATION ======"
echo "Starting at: $(date)"
echo ""

# Create build directory
mkdir -p /tmp/mlir_build
cd /tmp/mlir_build

echo "Step 1: MLIR -> LLVM Dialect"
mlir-opt /app/src/tests/micro_kernel.mlir \
    --convert-vector-to-llvm \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --memref-expand \
    --finalize-memref-to-llvm \
    --reconcile-unrealized-casts \
    -o micro_kernel_llvm.mlir 2>&1 | tee step1.log

if [ -f micro_kernel_llvm.mlir ]; then
    echo "✅ Step 1 SUCCESS: $(wc -l micro_kernel_llvm.mlir)"
else
    echo "❌ Step 1 FAILED"
    cat step1.log
    exit 1
fi

echo ""
echo "Step 2: LLVM Dialect -> LLVM IR"
mlir-translate --mlir-to-llvmir micro_kernel_llvm.mlir -o micro_kernel.ll 2>&1 | tee step2.log

if [ -f micro_kernel.ll ]; then
    echo "✅ Step 2 SUCCESS: $(wc -l micro_kernel.ll)"
else
    echo "❌ Step 2 FAILED"
    cat step2.log
    exit 1
fi

echo ""
echo "Step 3: LLVM IR -> Object File"
llc micro_kernel.ll \
    -O3 \
    -mcpu=native \
    -mattr=+avx2,+fma \
    -filetype=obj \
    -o micro_kernel.o 2>&1 | tee step3.log

if [ -f micro_kernel.o ]; then
    echo "✅ Step 3 SUCCESS: $(ls -lh micro_kernel.o)"
else
    echo "❌ Step 3 FAILED"
    cat step3.log
    exit 1
fi

echo ""
echo "Step 4: Analyze Assembly"
objdump -d micro_kernel.o > micro_kernel.asm
FMA_COUNT=$(grep -c "vfmadd" micro_kernel.asm || echo 0)
BROADCAST_COUNT=$(grep -c "vbroadcastss" micro_kernel.asm || echo 0)
YMM_COUNT=$(grep "ymm" micro_kernel.asm | wc -l || echo 0)

echo "📊 Assembly Statistics:"
echo "   FMA instructions: $FMA_COUNT (expected: 3000+)"
echo "   Broadcast instructions: $BROADCAST_COUNT (expected: 1500+)"
echo "   YMM register uses: $YMM_COUNT"

if [ $FMA_COUNT -gt 2500 ]; then
    echo "✅ EXCELLENT FMA count!"
elif [ $FMA_COUNT -gt 1000 ]; then
    echo "⚠️  Good FMA count, but could be better"
else
    echo "❌ LOW FMA count - optimization problem!"
fi

echo ""
echo "Step 5: Copy to project"
cp micro_kernel.o /app/micro_kernel_mlir.o
echo "✅ Copied to /app/micro_kernel_mlir.o"

echo ""
echo "Step 6: Show first 50 lines of assembly"
head -50 micro_kernel.asm

echo ""
echo "====== COMPILATION COMPLETE ======"
echo "Finished at: $(date)"



