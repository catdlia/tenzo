# Progress - Adversarial Review of Ternary Pack Integration

## Step 1: Independent Understanding of Requirements
- R1: Define `TernaryPackOp` (`tenzo.ternary_pack`) with `[Pure]` trait in `src/dialect/TenzoOps.td`.
  - Arguments: `AnyTensor:$input`, `Optional<AnyType>:$scale`, `DefaultValuedAttr<I32Attr, "4">:$values_per_byte`.
  - Results: `AnyTensor:$result` (packed uint8 tensor).
- R2: Modify `src/passes/ExplicitMicroKernelPass.cpp` for `tenzo.ternary_pack`:
  - Hardcode tiling to `MR=4, NR=16` specifically for `tenzo.ternary_pack`.
  - Preserve FP32 pipeline (`MR=6, NR=16`) intact.
  - Register budget: 8 accums, 2 B matrix, 1 A broadcast, 3 packing constants (`v_scale`, `v_one`, multipliers), 2 scratchpads.
- R3: Inject fused AVX2 packing logic after K-loop:
  - Scale accumulators, convert to i32 (`_mm256_cvtps_epi32`), shift +1 (`{-1,0,1}` -> `{0,1,2}`), narrow to i16/i8 (`_mm256_packs_epi32`, `_mm256_packs_epi16`), cross-lane permute (`_mm256_permute4x64_epi64`), 4-to-1 packing (`_mm_maddubs_epi16`, `_mm_madd_epi16`, `_mm_shuffle_epi8`), write 32-bit blocks (4 bytes per row) to uint8 memref.

## Step 2: Adversarial Audit of Prior Attempt
Found the following defects:
1. **Fatal Functional Bug: Invalid `FloatAttr` on `VectorType` in `ExplicitMicroKernelPass.cpp`**:
   `rewriter.getFloatAttr(vecType, 1.0f)` was called in 10 places.
   MLIR verifier rejects FloatAttr with VectorType (`error: floating point value not valid for specified type`).
   Root cause: FloatAttr must have a scalar FloatType, not a VectorType.
2. **Fatal Functional Bug: Invalid `memref.reinterpret_cast` across element types**:
   In `ExplicitMicroKernelPass.cpp`, line 428 created `memref.reinterpret_cast` when `outMemref` was not i8 (e.g. `memref<4x16xf32>`).
   MLIR verifier strictly requires identical element types for `memref.reinterpret_cast`.
   Root cause: Mismatched element types (`f32` to `i8`) trigger `error: different element types specified for source type 'memref<4x16xf32>' and result memref type 'memref<4x4xi8>'`.
3. **Fatal Functional Bug: Unsafe `rewriter.eraseOp(ternaryPackConsumer)`**:
   If `ternaryPackConsumer` had users, erasing it triggered an MLIR assertion failure (`"operation destroyed but still has uses"`).
   Root cause: Operations with remaining uses cannot be erased; their uses must be replaced with the produced memref or tensor.
4. **Shallow Verification in Prior Test Suite**:
   Prior test 2 never called `module.verify()`, hiding the invalid IR bugs.
   Prior test 4 only executed a handwritten C++ reference function, NEVER testing the generated MLIR AST end-to-end via execution engine.
   Optional scale was never tested when omitted in parser/printer round-trip.
   Standalone `tenzo.ternary_pack` pattern was never tested.

## Step 3: Fix Applied
- Fixed `ExplicitMicroKernelPass.cpp`:
  - Replaced all 10 occurrences of `rewriter.getFloatAttr(vecType, ...)` with `DenseElementsAttr::get(vecType, ...)`.
  - Replaced invalid `memref.reinterpret_cast` across element types with `UnrealizedConversionCastOp` (preserving `ReinterpretCastOp` for same-element-type reinterpretation).
  - Replaced unsafe `rewriter.eraseOp(ternaryPackConsumer)` with safe `rewriter.replaceOp(ternaryPackConsumer, ...)` when uses exist.
- Enhanced `src/tests/TernaryPackTest.cpp`:
  - Added strict `assert(succeeded(module.verify()))` to every test.
  - Added Test 1: TableGen dialect definition & textual MLIR parsing round-trip with and without scale.
  - Added Test 2: Fused microkernel generation on matmul with `tenzo.ternary_pack` attribute.
  - Added Test 3: Standalone `tenzo.ternary_pack` lowering and consumer replacement verification.
  - Added Test 4: FP32 pipeline isolation (`MR=6, NR=16` producing exactly 12 transfer writes).
  - Added Test 5: AVX2 bit-exact packing arithmetic across ternary domains and scales (1.0f and 0.5f).
  - Added Test 6: True end-to-end JIT execution of the generated fused microkernel via `ExecutionEngine`, verifying bit-exact packed byte outputs (`0xAA`, `0x55`, `0x00`, `0x24`).

## Step 4: Re-verification
- Built cleanly in Docker: `docker compose run --rm dev ninja -C /app/cmake-build-debug tenzo-cli` (exit code 0).
- Ran `tenzo-cli ternary` (all 6/6 tests passed).
- Ran `tenzo-cli test` (CPU MatMul, Conv2D, GPU Pipeline, and Ternary Pack all passed).
- Ran `tenzo-cli explicit` (512x512 microkernel benchmark passed, 25.4 GFLOPS).
