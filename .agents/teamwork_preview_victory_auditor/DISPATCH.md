## 2026-09-18T09:48:28Z

You are teamwork_preview_victory_auditor.
Your assigned working directory is: /home/illia/CLionProjects/untitled/.agents/teamwork_preview_victory_auditor/
Write your audit report to: /home/illia/CLionProjects/untitled/.agents/teamwork_preview_victory_auditor/handoff.md

<original_task>
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
</original_task>

CRITICAL PROJECT CONSTRAINTS:
1. NEVER compile locally by default. ALWAYS use `make build` (which triggers `./remote_build.sh` remotely) if compilation is needed.
2. Execution/Testing is Docker-isolated: run via Docker compose / Makefile targets (e.g., `docker compose run --rm dev /app/cmake-build-debug/tenzo-cli ternary`, `docker compose run --rm dev /app/cmake-build-debug/tenzo-cli test`). Do not execute binaries directly on host.
3. The build directory is strictly `cmake-build-debug`. Do not create or use `build/` or `build_mlir/`.

Perform a comprehensive, independent 3-phase victory audit:
Phase 1: Timeline & Diff inspection against the requirements R1, R2, R3 and acceptance criteria.
Phase 2: Cheating & Anti-pattern detection (verify that tests are not mocked, no hardcoded cheating bypasses exist, traits and register budgets are genuinely respected).
Phase 3: Independent test execution via Docker (`docker compose run --rm dev /app/cmake-build-debug/tenzo-cli ternary` and `docker compose run --rm dev /app/cmake-build-debug/tenzo-cli test`).

Deliver a structured verdict: CONFIRMED or REJECTED, with full rationale and evidence.
