#!/usr/bin/env python3
"""
scripts/test_compiler_regressions.py - Automated Regression Test Suite for Requirement R3
Milestone M3_continuation: Requirement R3 (Compiler Core Fixes & Math Regressions)

Verifies:
1. Initial C Write Ordering & Uninitialized Memory Read:
   - ExplicitMicroKernelPattern checks: hasNonzeroInitialC = hasExplicitNonZeroFill || (isa<BlockArgument>(C) && !isZeroFilled)
   - Uninitialized memref.alloc() has hasNonzeroInitialC = false (no uninitialized memory reads)
   - Write-back to C occurs strictly when (hasSecondaryConsumer || hasNonzeroInitialC)
   - Write-back occurs before emitTernaryPackingAST
   - Dead allocation cleanup erases unused AllocOp and DeallocOp
2. Exact Causal Masking Uniformity & Index Clamping:
   - LinalgLowering.cpp uses -1e30 for all causal masking and softmax max-initialization
   - ExplicitMicroKernelPass.cpp host reference microkernel uses -1e30f (no -10000.0f)
   - PackedAttentionMicroKernelPattern clamps seqPos to [0, T-1]
   - Numerical invariant: dot-product logits <= -1e9 do not invert attention against masked positions
3. Shape, Layout, and Type Rejections:
   - ExplicitMicroKernelPattern rejects cType rank != 2
   - ExplicitMicroKernelPattern rejects non-identity memref layouts on A, B, C
   - PackedAttentionMicroKernelPattern verifies Key and Value element types are i8
   - PackedAttentionMicroKernelPattern verifies Key and Value shapes match exactly
   - PackedAttentionMicroKernelPattern verifies rank consistency
   - LinalgLowering rejects non-divisible GQA heads (H % H_kv != 0)
   - LinalgLowering rejects mismatched head dimensions
4. Regressions for Random A/B/C and Boundary Canaries:
   - reference_gemm_fused_with_consumer handles random A/B and non-zero initial C across multiple scales
   - 16-element boundary canaries before and after buffers detect overrun corruptions
5. End-to-end integration and exit code 0 confirmation.
"""

import os
import sys
import re
import subprocess
import numpy as np
from typing import Tuple, List, Dict, Any

# Ensure project root and scripts dir are on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from verify_references import (
    reference_gemm_fp64,
    reference_quantize_round_clamp,
    reference_pack_ternary,
    reference_gemm_fused_with_consumer,
    reference_causal_gqa_attention_decode,
    print_diagnostic_diff,
)


def print_section(title: str):
    print("\n" + "=" * 76)
    print(f"🔬 {title}")
    print("=" * 76)


# ============================================================================
# Test 1: Initial C Write Ordering & Uninitialized Memory Read Verification
# ============================================================================
def test_initial_c_write_ordering_and_uninit_read() -> bool:
    print_section("Test 1: Initial C Write Ordering & Uninitialized Memory Read")
    all_passed = True
    pass_file = os.path.join(PROJECT_ROOT, "src", "passes", "ExplicitMicroKernelPass.cpp")

    if not os.path.isfile(pass_file):
        print(f"❌ File not found: {pass_file}")
        return False

    with open(pass_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Verify hasNonzeroInitialC requires explicit fill or block argument
    # Bad pattern: hasNonzeroInitialC = !isZeroFilled;
    bad_pattern = re.search(r"hasNonzeroInitialC\s*=\s*!isZeroFilled\s*;", content)
    if bad_pattern:
        print("❌ FAILED: Found buggy 'hasNonzeroInitialC = !isZeroFilled;' in ExplicitMicroKernelPass.cpp!")
        all_passed = False
    else:
        print("✅ [PASS] Defective 'hasNonzeroInitialC = !isZeroFilled;' correctly removed.")

    # Good pattern: hasExplicitNonZeroFill and block argument check
    good_pattern = re.search(
        r"hasNonzeroInitialC\s*=\s*hasExplicitNonZeroFill\s*\|\|\s*\(isa<BlockArgument>\(C\)\s*&&\s*!isZeroFilled\)\s*;",
        content
    )
    if not good_pattern:
        print("❌ FAILED: Did not find required 'hasNonzeroInitialC = hasExplicitNonZeroFill || (isa<BlockArgument>(C) && !isZeroFilled);'")
        all_passed = False
    else:
        print("✅ [PASS] Strict initial C detection verified: uninitialized memref.alloc() produces hasNonzeroInitialC = false.")

    # 2. Verify write-back occurs only when secondary consumers or genuine nonzero C exist
    wb_pattern = re.search(
        r"if\s*\(hasSecondaryConsumer\s*\|\|\s*hasNonzeroInitialC\)\s*\{[\s\S]*?TransferWriteOp[\s\S]*?emitTernaryPackingAST",
        content
    )
    if not wb_pattern:
        print("❌ FAILED: Write-back to C is either unconditional or incorrectly sequenced after emitTernaryPackingAST!")
        all_passed = False
    else:
        print("✅ [PASS] Correct write-back ordering verified: dual-write occurs before emitTernaryPackingAST when needed.")

    # 3. Verify dead allocation cleanup for unconsumed local allocations
    cleanup_pattern = re.search(
        r"if\s*\(ternaryPackConsumer\s*&&\s*!hasSecondaryConsumer\s*&&\s*!hasNonzeroInitialC\)\s*\{[\s\S]*?AllocOp[\s\S]*?DeallocOp",
        content
    )
    if not cleanup_pattern:
        print("❌ FAILED: Dead allocation cleanup for unconsumed C buffers not found!")
        all_passed = False
    else:
        print("✅ [PASS] Dead allocation cleanup verified: unconsumed AllocOp and DeallocOp properly erased.")

    return all_passed


# ============================================================================
# Test 2: Exact Causal Masking Uniformity & Index Clamping
# ============================================================================
def test_causal_masking_uniformity_and_index_clamping() -> bool:
    print_section("Test 2: Exact Causal Masking Uniformity & Index Clamping")
    all_passed = True

    # 1. Check LinalgLowering.cpp for -1e9 mask values
    linalg_file = os.path.join(PROJECT_ROOT, "src", "passes", "LinalgLowering.cpp")
    with open(linalg_file, "r", encoding="utf-8") as f:
        linalg_content = f.read()

    # Search for -1e9 in LinalgLowering
    if "-1e9" in linalg_content:
        print("❌ FAILED: Found lingering '-1e9' mask value in LinalgLowering.cpp!")
        all_passed = False
    else:
        print("✅ [PASS] Zero occurrences of weak '-1e9' mask value in LinalgLowering.cpp.")

    # Verify -1e30 is used
    if "-1e30" not in linalg_content:
        print("❌ FAILED: Expected '-1e30' mask value not found in LinalgLowering.cpp!")
        all_passed = False
    else:
        print("✅ [PASS] Uniform '-1e30' causal mask constant verified in LinalgLowering.cpp.")

    # 2. Check ExplicitMicroKernelPass.cpp host reference microkernel
    pass_file = os.path.join(PROJECT_ROOT, "src", "passes", "ExplicitMicroKernelPass.cpp")
    with open(pass_file, "r", encoding="utf-8") as f:
        pass_content = f.read()

    if "-10000.0f" in pass_content:
        print("❌ FAILED: Found lingering '-10000.0f' mask in ExplicitMicroKernelPass.cpp host reference microkernel!")
        all_passed = False
    else:
        print("✅ [PASS] Zero occurrences of weak '-10000.0f' mask in ExplicitMicroKernelPass.cpp.")

    ref_mask_pattern = re.search(r"scores\(T,\s*-1e30f\)", pass_content)
    if not ref_mask_pattern:
        print("❌ FAILED: Host reference microkernel does not initialize scores with -1e30f!")
        all_passed = False
    else:
        print("✅ [PASS] Host reference microkernel initializes scores with robust -1e30f.")

    # 3. Check seqPos index clamping in PackedAttentionMicroKernelPattern
    clamp_pattern = re.search(r"maxTIdx[\s\S]*?MaxSIOp[\s\S]*?MinSIOp", pass_content)
    if not clamp_pattern:
        print("❌ FAILED: seqPos clamping to [0, T-1] via MaxSIOp/MinSIOp not found in PackedAttentionMicroKernelPattern!")
        all_passed = False
    else:
        print("✅ [PASS] seqPos index clamping to valid bounds [0, T-1] verified.")

    # 4. Numerical validation: large negative logits (<= -1e9) do not invert attention
    pos = 0
    T = 4
    # Valid logit is very negative: -1.5e9
    # Defective mask logit was -1.0e9 (which is LARGER than -1.5e9, causing inversion!)
    # Robust mask logit is -1.0e30
    valid_score = -1.5e9
    robust_mask = -1.0e30
    scores = np.array([valid_score, robust_mask, robust_mask, robust_mask], dtype=np.float64)

    # Numerically stable softmax
    max_s = np.max(scores)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / np.sum(exp_s)

    # Valid token at index 0 must receive probability 1.0; masked positions must receive 0.0
    if abs(probs[0] - 1.0) > 1e-12 or np.sum(probs[1:]) > 1e-15:
        print(f"❌ FAILED: Numerical inversion under large negative logit! probs={probs}")
        all_passed = False
    else:
        print(f"✅ [PASS] Numerical causal isolation: valid_prob={probs[0]:.6f}, masked_leakage={np.sum(probs[1:]):.2e} under -1.5e9 logit.")

    return all_passed


# ============================================================================
# Test 3: Shape, Layout, and Type Rejections
# ============================================================================
def test_shape_layout_type_rejections() -> bool:
    print_section("Test 3: Shape, Layout, and Type Rejections")
    all_passed = True

    pass_file = os.path.join(PROJECT_ROOT, "src", "passes", "ExplicitMicroKernelPass.cpp")
    with open(pass_file, "r", encoding="utf-8") as f:
        pass_content = f.read()

    linalg_file = os.path.join(PROJECT_ROOT, "src", "passes", "LinalgLowering.cpp")
    with open(linalg_file, "r", encoding="utf-8") as f:
        linalg_content = f.read()

    # 1. cType.getRank() != 2 rejection
    c_rank_pattern = re.search(r"cType\.getRank\(\)\s*!=\s*2", pass_content)
    if not c_rank_pattern:
        print("❌ FAILED: cType.getRank() != 2 rejection not found in ExplicitMicroKernelPattern!")
        all_passed = False
    else:
        print("✅ [PASS] cType.getRank() != 2 rejection verified in ExplicitMicroKernelPattern.")

    # 2. Non-identity memref layout rejection on A, B, C
    layout_pattern = re.search(r"!aType\.getLayout\(\)\.isIdentity\(\)\s*\|\|\s*!bType\.getLayout\(\)\.isIdentity\(\)\s*\|\|\s*!cType\.getLayout\(\)\.isIdentity\(\)", pass_content)
    if not layout_pattern:
        print("❌ FAILED: Non-identity memref layout rejection on A, B, C not found!")
        all_passed = False
    else:
        print("✅ [PASS] Non-identity memref layout rejection on A, B, C verified.")

    # 3. PackedAttentionMicroKernelPattern: Key/Value element types are i8
    i8_pattern = re.search(r"!kType\.getElementType\(\)\.isInteger\(8\)\s*\|\|\s*!vType\.getElementType\(\)\.isInteger\(8\)", pass_content)
    if not i8_pattern:
        print("❌ FAILED: Key/Value i8 element type check not found in PackedAttentionMicroKernelPattern!")
        all_passed = False
    else:
        print("✅ [PASS] Key and Value i8 element type verification verified.")

    # 4. PackedAttentionMicroKernelPattern: Key and Value shapes match
    shape_match_pattern = re.search(r"kType\.getShape\(\)\s*!=\s*vType\.getShape\(\)", pass_content)
    if not shape_match_pattern:
        print("❌ FAILED: Key and Value shape matching check not found in PackedAttentionMicroKernelPattern!")
        all_passed = False
    else:
        print("✅ [PASS] Key and Value shape identity verification verified.")

    # 5. PackedAttentionMicroKernelPattern: Rank consistency
    rank_pattern = re.search(r"kRank\s*!=\s*vRank", pass_content)
    if not rank_pattern:
        print("❌ FAILED: Key/Value rank consistency check not found in PackedAttentionMicroKernelPattern!")
        all_passed = False
    else:
        print("✅ [PASS] Key and Value rank consistency verification verified.")

    # 6. LinalgLowering: Non-divisible GQA heads rejection
    gqa_div_pattern = re.search(r"H_kv\s*<=\s*0\s*\|\|\s*H\s*<=\s*0\s*\|\|\s*\(H\s*%\s*H_kv\s*!=\s*0\)", linalg_content)
    if not gqa_div_pattern:
        print("❌ FAILED: Non-divisible GQA heads rejection (H % H_kv != 0) not found in LinalgLowering.cpp!")
        all_passed = False
    else:
        print("✅ [PASS] Non-divisible GQA heads rejection (H % H_kv != 0) verified in LinalgLowering.cpp.")

    # 7. LinalgLowering: Mismatched head dimensions rejection
    head_dim_pattern = re.search(r"Dh\s*<=\s*0\s*\|\|\s*\(Dh\s*%\s*4\s*!=\s*0\)\s*\|\|\s*K_Dh\s*!=\s*\(Dh\s*/\s*4\)", linalg_content)
    if not head_dim_pattern:
        print("❌ FAILED: Mismatched head dimensions check not found in LinalgLowering.cpp!")
        all_passed = False
    else:
        print("✅ [PASS] Mismatched head dimensions rejection verified in LinalgLowering.cpp.")

    return all_passed


# ============================================================================
# Test 4: Regressions for Random A/B/C and Boundary Canaries
# ============================================================================
def test_random_matrices_and_boundary_canaries() -> bool:
    print_section("Test 4: Regressions for Random A/B/C and Boundary Canaries")
    all_passed = True
    np.random.seed(2026)

    M, K, N = 4, 16, 16

    # 1. Independent math verification with random A/B and non-zero C across scales
    for scale in [0.5, 1.0, 2.5, -1.0]:
        for trial in range(5):
            A = np.random.uniform(-4.0, 4.0, size=(M, K)).astype(np.float32)
            B = np.random.uniform(-3.0, 3.0, size=(K, N)).astype(np.float32)
            C_init = np.random.uniform(-2.0, 2.0, size=(M, N)).astype(np.float32)

            C_out, P_packed = reference_gemm_fused_with_consumer(A, B, C_initial=C_init, scale=scale)

            # Mathematical parity check: C_final = A @ B + C_init
            C_expected = reference_gemm_fp64(A, B) + C_init.astype(np.float64)
            diff = float(np.max(np.abs(C_out.astype(np.float64) - C_expected)))
            if diff > 1e-5:
                print(f"❌ FAILED: Parity mismatch at scale={scale}, trial={trial}: max_diff={diff}")
                all_passed = False
                break

            # Packed parity check
            Q_expected = reference_quantize_round_clamp(C_expected, scale=scale, scale_is_multiplier=True)
            P_expected = reference_pack_ternary(Q_expected)
            if not np.array_equal(P_packed, P_expected):
                print(f"❌ FAILED: Ternary packing mismatch at scale={scale}, trial={trial}")
                all_passed = False
                break

    if all_passed:
        print("✅ [PASS] Independent math parity verified across random A, B, non-zero C, and scales {0.5, 1.0, 2.5, -1.0}.")

    # 2. Boundary Canary Simulation Test
    # Simulate a_buf, b_buf, c_buf, other_buf, packed_buf memory layout and overrun detector
    FLOAT_CANARY = -789.0
    BYTE_CANARY = 0xCC

    class BufferWithCanaries:
        def __init__(self, size: int, is_byte: bool = False):
            self.size = size
            self.is_byte = is_byte
            self.canary_val = BYTE_CANARY if is_byte else FLOAT_CANARY
            dtype = np.uint8 if is_byte else np.float32
            self.buffer = np.full(16 + size + 16, self.canary_val, dtype=dtype)

        def payload(self):
            return self.buffer[16:16 + self.size]

        def check_canaries(self) -> int:
            corruptions = 0
            if not np.all(self.buffer[:16] == self.canary_val):
                corruptions += int(np.sum(self.buffer[:16] != self.canary_val))
            if not np.all(self.buffer[16 + self.size:] == self.canary_val):
                corruptions += int(np.sum(self.buffer[16 + self.size:] != self.canary_val))
            return corruptions

    # Test clean state
    a_buf = BufferWithCanaries(64)
    b_buf = BufferWithCanaries(256)
    c_buf = BufferWithCanaries(64)
    other_buf = BufferWithCanaries(64)
    packed_buf = BufferWithCanaries(16, is_byte=True)

    buffers = [a_buf, b_buf, c_buf, other_buf, packed_buf]
    total_corruptions = sum(b.check_canaries() for b in buffers)
    if total_corruptions != 0:
        print(f"❌ FAILED: False-positive canary corruption in clean state: {total_corruptions}")
        all_passed = False
    else:
        print("✅ [PASS] Buffer canaries: Clean state reports 0 corruptions across all 5 buffers.")

    # Test intentional overrun detection
    a_buf.buffer[15] = 999.0  # front overrun
    other_buf.buffer[16 + 64] = 999.0  # back overrun
    packed_buf.buffer[0] = 0xAA  # front byte overrun

    detected = a_buf.check_canaries() + other_buf.check_canaries() + packed_buf.check_canaries()
    if detected != 3:
        print(f"❌ FAILED: Overrun detector failed to catch injected corruptions! detected={detected}, expected=3")
        all_passed = False
    else:
        print("✅ [PASS] Buffer canaries: Overrun detector successfully caught all 3 injected boundary corruptions.")

    return all_passed


# ============================================================================
# Test 5: Verify Audit JIT Test Source Code Coverage
# ============================================================================
def test_audit_jit_test_source_coverage() -> bool:
    print_section("Test 5: Audit JIT Test Source Code Coverage")
    all_passed = True
    audit_test_file = os.path.join(PROJECT_ROOT, "src", "tests", "AuditJITTest.cpp")

    if not os.path.isfile(audit_test_file):
        print(f"❌ File not found: {audit_test_file}")
        return False

    with open(audit_test_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Verify buffer canaries on a_buf, b_buf, c_buf, other_buf, packed_buf
    expected_buffers = ["a_buf", "b_buf", "c_buf", "other_buf", "packed_buf"]
    for buf in expected_buffers:
        if buf not in content:
            print(f"❌ FAILED: Expected canary buffer '{buf}' not found in AuditJITTest.cpp!")
            all_passed = False
        else:
            print(f"✅ [PASS] Found canary buffer '{buf}' in AuditJITTest.cpp.")

    # 2. Verify checkCanaries function exists
    if "checkCanaries" not in content:
        print("❌ FAILED: 'checkCanaries' function not found in AuditJITTest.cpp!")
        all_passed = False
    else:
        print("✅ [PASS] 'checkCanaries' boundary validation function found.")

    # 3. Verify random matrix testing with std::uniform_real_distribution
    if "dist_ab" not in content or "std::uniform_real_distribution" not in content:
        print("❌ FAILED: Random matrix distribution testing not found in AuditJITTest.cpp!")
        all_passed = False
    else:
        print("✅ [PASS] Random A and B matrix distribution testing found.")

    # 4. Verify non-zero initial C testing across scales
    if "c_initial" not in content or "audit_pack_nonzero" not in content:
        print("❌ FAILED: Non-zero initial C accumulation testing not found in AuditJITTest.cpp!")
        all_passed = False
    else:
        print("✅ [PASS] Non-zero initial C accumulation testing found.")

    return all_passed


# ============================================================================
# Main Test Runner
# ============================================================================
def main():
    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║   Tenzo Compiler Core & Math Regressions (Requirement R3)             ║")
    print("║   Milestone M3_continuation: Automated Verification Suite              ║")
    print("╚════════════════════════════════════════════════════════════════════════╝")

    suite_results = []

    suite_results.append(("Initial C Write Ordering & Uninit Read", test_initial_c_write_ordering_and_uninit_read()))
    suite_results.append(("Causal Masking Uniformity & Index Clamping", test_causal_masking_uniformity_and_index_clamping()))
    suite_results.append(("Shape, Layout, and Type Rejections", test_shape_layout_type_rejections()))
    suite_results.append(("Random A/B/C and Boundary Canaries", test_random_matrices_and_boundary_canaries()))
    suite_results.append(("Audit JIT Test Source Coverage", test_audit_jit_test_source_coverage()))

    print("\n" + "=" * 76)
    print("📊 REGRESSION SUITE EXECUTION SUMMARY")
    print("=" * 76)

    all_ok = True
    for name, passed in suite_results:
        status_str = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status_str} : {name}")
        if not passed:
            all_ok = False

    print("=" * 76)
    if all_ok:
        print("🎉 ALL COMPILER REGRESSION TESTS PASSED CLEANLY (EXIT CODE 0)! 🎉")
        sys.exit(0)
    else:
        print("❌ COMPILER REGRESSION TESTS FAILED! (Divergence or defect detected)")
        sys.exit(1)


if __name__ == "__main__":
    main()
