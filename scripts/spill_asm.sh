#!/bin/bash
# Block G helper (run INSIDE the dev container): turn the audit JIT kernels' LLVM-dialect MLIR into x86 assembly the way the JIT does
# (mlir-translate -> opt -O3 -> llc -O3 -mcpu=alderlake) and disassemble the engine's hand-written attention.
# Usage: bash /app/scripts/spill_asm.sh <ir_dir> <out_dir>
set -eu
IR=${1:?ir dir}; OUT=${2:?out dir}; B=/usr/lib/llvm-21/bin
mkdir -p "$OUT"
for f in "$IR"/*-llvm.mlir; do
  n=$(basename "$f" -llvm.mlir)
  $B/mlir-translate --mlir-to-llvmir "$f" -o "$OUT/$n.ll"
  $B/opt -O3 -mcpu=alderlake -S "$OUT/$n.ll" -o "$OUT/$n.opt.ll"
  $B/llc -O3 -mcpu=alderlake -filetype=asm "$OUT/$n.opt.ll" -o "$OUT/$n.s"
  echo "asm: $OUT/$n.s ($(wc -l < "$OUT/$n.s") lines)"
done
# Engine hand-written kernels: objdump of the object the Docker build produced
# (the loops live in the OpenMP-outlined body `..._omp_fn.0`; object = the one linked into tenzo-inference)
OBJ=/app/cmake-build-debug/CMakeFiles/tenzo_runtime_static.dir/src/runtime/TenzoEngine.cpp.o
echo "engine object: $OBJ"
$B/llvm-objdump -d --no-show-raw-insn "$OBJ" \
  --disassemble-symbols='_ZN5tenzo12FusedKVCache21forward_attention_rawEiPfS1_S1_S1_._omp_fn.0' > "$OUT/engine_forward_attention_raw.dis"
wc -l "$OUT/engine_forward_attention_raw.dis"
