# Handoff - Reviewer 2

## Status
Reviewer 2 adversarial audit and defect fixing complete. All tests pass with zero regressions.

## Summary of Defects Found & Fixed
1. **Broken Fusion across Bufferization Boundaries**: When `linalg.matmul` produced an FP32 memref and passed it to `tenzo.ternary_pack` via `bufferization.to_tensor`, the matmul pattern only checked direct memref users, failed to match `tenzo.ternary_pack`, and fell back to standard FP32 GEMM (MR=6), causing shape mismatches (4 != 6). Fixed by inspecting recursive users through `to_tensor` and cast operations.
2. **Rewriter Pattern Preemption**: Both `ExplicitMicroKernelPattern` and `TernaryPackMicroKernelPattern` had equal benefit (1). Greedy pattern application matched standalone `tenzo.ternary_pack` first, detaching it from producer matmuls and preventing fused micro-kernel generation. Fixed by assigning benefit 2 to the fused GEMM pattern and guarding the standalone pattern against unlowered matmul producers.
3. **Dead `to_tensor` LLVM Legalization Failure**: `TernaryPackMicroKernelPattern` lowered the tensor pack operation but left dead `bufferization.to_tensor` ops in the IR, causing downstream LLVM lowering to fail with `failed to legalize operation 'bufferization.to_tensor'`. Fixed by rewiring downstream transfer reads directly to the packed memref and erasing dead to_tensor operations.
4. **Register Budget Violation**: R2 allocated 4 constant registers (`vScale`, `vOne`, `shift2Const`, `shift4Const`), exceeding the 3-constant AVX2 budget (total 17 registers vs 16 available YMM registers). Fixed by allocating strictly 3 constants (`vScale`, `vOne`, `vMultipliers`) and dynamically computing shift-by-4 in scratchpad YMM15 via `arith.addi`.
5. **Illegal Cross-Type Reinterpret Cast**: Downstream replaceOp attempted `memref.reinterpret_cast` from `f32` to `i8`. Fixed by creating a dedicated `memref.alloc` for `memref<4x4xi8>`.

## Verification
- `docker compose run --rm dev /app/cmake-build-debug/tenzo-cli ternary`: 8/8 tests PASSED (including round-trip MLIR parsing, fused pass verification, standalone lowering, FP32 isolation, AVX2 bit-exact arithmetic, end-to-end JIT fused execution, pipeline fusion, standalone JIT execution with positive and negative scales).
- `docker compose run --rm dev /app/cmake-build-debug/tenzo-cli test`: All 4 test suites PASSED (CPU, GPU, Hetero, Ternary).

