# Original User Request

## 2026-09-17T21:49:01Z

This is a single self-contained fix; keep it small and focused. Integrate the `tenzo.ternary_pack` MLIR operation and its corresponding AVX2 fused micro-kernel generation into the Tenzo compiler.

Working directory: `/home/illia/CLionProjects/untitled`
Integrity mode: benchmark

## Requirements

### R1. Update MLIR Dialect
In `src/dialect/TenzoOps.td`, add the `TernaryPackOp` (`tenzo.ternary_pack`) with the `[Pure]` trait. It must accept `AnyTensor:$input`, `Optional<AnyType>:$scale`, `DefaultValuedAttr<I32Attr, "4">:$values_per_byte` and return `AnyTensor:$result` (packed uint8 tensor).

### R2. Modify Microkernel Generation Pass
In `src/passes/ExplicitMicroKernelPass.cpp`, add handling for `tenzo.ternary_pack`. Hardcode the tiling to `MR=4, NR=16` specifically for this operation to preserve YMM registers. Do not break the existing FP32 pipeline for other operations. 
Allocate registers: 8 for accumulators, 2 for B matrix, 1 for A broadcast, 3 for packing constants (`v_scale`, `v_one`, multipliers), and 2 as scratchpads for packing instructions.

### R3. Inject Fused AVX2 Logic
Inside the inner loop generation for `tenzo.ternary_pack` (after the K-loop), implement the C++ AST generation for: multiplying accumulators by scale, converting to 32-bit ints (`_mm256_cvtps_epi32`), shifting values by +1 (domain `{-1, 0, 1}` to `{0, 1, 2}`), narrowing to 16-bit and 8-bit (`_mm256_packs_epi32`, `_mm256_packs_epi16`), cross-lane permute (`_mm256_permute4x64_epi64`), final packing of 4 bytes into 1 (`_mm_maddubs_epi16`, `_mm_madd_epi16`, `_mm_shuffle_epi8`), and writing the final 32-bit blocks to the uint8_t output memref.

## Verification Resources
Verify that the generated code builds successfully via the project's standard build process (e.g., `make build` / `./remote_build.sh` as per global rules) and that any relevant tests pass. 

## Acceptance Criteria

### Implementation
- [ ] `TenzoOps.td` compiles with tablegen and correctly defines `TernaryPackOp`.
- [ ] `ExplicitMicroKernelPass.cpp` successfully generates the `MR=4, NR=16` micro-kernel AST only when `tenzo.ternary_pack` is encountered.
- [ ] The existing FP32 micro-kernel generation for other operations (using `MR=6, NR=16`) remains intact and unmodified.
- [ ] The generated C++ AVX2 code builds without syntax errors and correctly handles 1.58-bit packing.
