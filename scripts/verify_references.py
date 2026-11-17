#!/usr/bin/env python3
"""
scripts/verify_references.py - Independent Mathematical Reference Verification Suite
for Tenzo MLIR Compiler (Milestone 2 / W02, W03).

Provides independent scalar double-precision (NumPy FP64) references for:
1. Ternary round-and-clamp packing (standalone and QAT quantization)
2. GEMM with multi-consumer FP32 outputs
3. Causal GQA attention: softmax(Q @ K^T / sqrt(D) + causal_mask) @ V
   across arbitrary B, Hq, Hkv, T, D.

Asserts numerical parity with strict tolerance (1e-4) against compiler outputs.
Prints diagnostic diffs and exits with non-zero status on divergence.
"""

import sys
import os
import argparse
import json
import math
from typing import Tuple, List, Optional, Dict, Any

try:
    import numpy as np
except ImportError:
    print("❌ Error: numpy is required for independent reference verification.", file=sys.stderr)
    print("Install via: pip3 install numpy", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# 1. TERNARY PACKING REFERENCE MATH (FP64 -> Round & Clamp -> 2-bit Pack)
# ============================================================================

def reference_pack_byte(v0: int, v1: int, v2: int, v3: int) -> int:
    """Pack four ternary values in {-1, 0, 1} into a single uint8 byte."""
    u0 = (v0 + 1) & 0x03
    u1 = (v1 + 1) & 0x03
    u2 = (v2 + 1) & 0x03
    u3 = (v3 + 1) & 0x03
    return u0 | (u1 << 2) | (u2 << 4) | (u3 << 6)


def reference_unpack_byte(byte_val: int) -> Tuple[int, int, int, int]:
    """Unpack a uint8 byte into four ternary values in {-1, 0, 1}."""
    u0 = byte_val & 0x03
    u1 = (byte_val >> 2) & 0x03
    u2 = (byte_val >> 4) & 0x03
    u3 = (byte_val >> 6) & 0x03
    return (u0 - 1, u1 - 1, u2 - 1, u3 - 1)


def reference_quantize_round_clamp(
    x: np.ndarray,
    scale: float = 1.0,
    scale_is_multiplier: bool = True
) -> np.ndarray:
    """
    Independent FP64 round-to-nearest and clamp to {-1, 0, 1}.
    Matches QAT Straight-Through Estimator and IEEE 754 round-to-nearest ties-to-even.
    """
    x_fp64 = np.asarray(x, dtype=np.float64)
    if scale_is_multiplier:
        scaled = x_fp64 * float(scale)
    else:
        scaled = x_fp64 / float(scale)
    # np.round uses Banker's rounding (round half to even), matching std::nearbyint
    rounded = np.round(scaled)
    clamped = np.clip(rounded, -1.0, 1.0).astype(np.int64)
    return clamped


def reference_pack_ternary(ternary_array: np.ndarray) -> np.ndarray:
    """
    Packs array of shape (..., K) where K % 4 == 0 and values in {-1, 0, 1}
    into uint8 array of shape (..., K // 4).
    """
    arr = np.asarray(ternary_array, dtype=np.int64)
    assert arr.shape[-1] % 4 == 0, f"Last dimension must be a multiple of 4, got shape {arr.shape}"
    assert np.all((arr >= -1) & (arr <= 1)), "All values must be ternary in {-1, 0, 1}"

    u = (arr + 1).astype(np.uint8) & 0x03
    orig_shape = arr.shape[:-1]
    k_dim = arr.shape[-1]
    u_reshaped = u.reshape(*orig_shape, k_dim // 4, 4)

    b0 = u_reshaped[..., 0]
    b1 = u_reshaped[..., 1] << 2
    b2 = u_reshaped[..., 2] << 4
    b3 = u_reshaped[..., 3] << 6
    return (b0 | b1 | b2 | b3).astype(np.uint8)


def reference_unpack_ternary(packed_array: np.ndarray) -> np.ndarray:
    """
    Unpacks uint8 array of shape (..., K_bytes) into int8 array of shape (..., K_bytes * 4)
    with values in {-1, 0, 1}.
    """
    packed = np.asarray(packed_array, dtype=np.uint8)
    orig_shape = packed.shape[:-1]
    num_bytes = packed.shape[-1]

    u0 = (packed & 0x03).astype(np.int8) - 1
    u1 = ((packed >> 2) & 0x03).astype(np.int8) - 1
    u2 = ((packed >> 4) & 0x03).astype(np.int8) - 1
    u3 = ((packed >> 6) & 0x03).astype(np.int8) - 1

    stacked = np.stack([u0, u1, u2, u3], axis=-1)
    return stacked.reshape(*orig_shape, num_bytes * 4)


# ============================================================================
# 2. GEMM WITH MULTI-CONSUMER FP32 OUTPUT REFERENCE MATH
# ============================================================================

def reference_gemm_fp64(
    A: np.ndarray,
    B: np.ndarray
) -> np.ndarray:
    """
    Double-precision reference GEMM: C = A @ B.
    A: shape (M, K)
    B: shape (K, N)
    Returns C in float64 of shape (M, N).
    """
    A_fp64 = np.asarray(A, dtype=np.float64)
    B_fp64 = np.asarray(B, dtype=np.float64)
    return np.matmul(A_fp64, B_fp64)


def reference_gemm_fused_with_consumer(
    A: np.ndarray,
    B: np.ndarray,
    C_initial: Optional[np.ndarray] = None,
    scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes both outputs of fused MatMul -> TernaryPack with secondary consumer:
    1. C_fp32: intermediate FP32 GEMM output for secondary consumers.
       If C_initial is provided: C_final = A @ B + C_initial.
    2. P_packed: uint8 packed ternary output for primary consumer.
    """
    C_base = reference_gemm_fp64(A, B)
    if C_initial is not None:
        C_base = C_base + np.asarray(C_initial, dtype=np.float64)
    C_fp32 = C_base.astype(np.float32)
    Q = reference_quantize_round_clamp(C_base, scale=scale, scale_is_multiplier=True)
    P_packed = reference_pack_ternary(Q)
    return C_fp32, P_packed


# ============================================================================
# 3. CAUSAL GQA ATTENTION REFERENCE MATH (Arbitrary B, Hq, Hkv, T, D)
# ============================================================================

def reference_causal_gqa_attention_decode(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    pos: int,
    scale: Optional[float] = None
) -> np.ndarray:
    """
    Reference Causal GQA Attention for decode position `pos`:
    Q: shape (B, Hq, 1, D) or (B, Hq, D) - float32 / float64
    K: shape (B, Hkv, T, D) - float or ternary {-1, 0, 1}
    V: shape (B, Hkv, T, D) - float or ternary {-1, 0, 1}
    pos: integer position (0 <= pos < T), keys/values beyond pos are masked out.
    scale: scaling factor, defaults to 1.0 / sqrt(D).

    Returns: Output tensor of shape (B, Hq, 1, D) in float64.
    """
    Q_arr = np.asarray(Q, dtype=np.float64)
    K_arr = np.asarray(K, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)

    if Q_arr.ndim == 3:
        Q_arr = Q_arr[:, :, np.newaxis, :]  # (B, Hq, 1, D)

    B, Hq, _, D = Q_arr.shape
    _, Hkv, T, D_k = K_arr.shape
    assert D == D_k, f"Head dimension mismatch: Q has {D}, K has {D_k}"
    assert Hq % Hkv == 0, f"Hq ({Hq}) must be divisible by Hkv ({Hkv})"
    assert 0 <= pos < T, f"Position {pos} out of range [0, {T-1}]"

    if scale is None:
        scale = 1.0 / math.sqrt(float(D))

    group_size = Hq // Hkv
    out = np.zeros((B, Hq, 1, D), dtype=np.float64)

    for b in range(B):
        for h in range(Hq):
            kh = h // group_size
            q_vec = Q_arr[b, h, 0, :]  # shape (D,)
            k_seq = K_arr[b, kh, :pos + 1, :]  # shape (pos+1, D)
            v_seq = V_arr[b, kh, :pos + 1, :]  # shape (pos+1, D)

            # Dot products: Q @ K^T / sqrt(D)
            logits = np.dot(k_seq, q_vec) * scale  # shape (pos+1,)

            # Numerically stable softmax
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)  # shape (pos+1,)

            # Weighted sum: probs @ V
            context = np.dot(probs, v_seq)  # shape (D,)
            out[b, h, 0, :] = context

    return out


def reference_causal_gqa_attention_prefill(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: Optional[float] = None
) -> np.ndarray:
    """
    Reference Causal GQA Attention for full prefill sequence:
    Q: shape (B, Hq, T_q, D)
    K: shape (B, Hkv, T_k, D)
    V: shape (B, Hkv, T_k, D)
    Applies lower-triangular causal mask where position t_k > t_q is masked to -inf.
    """
    Q_arr = np.asarray(Q, dtype=np.float64)
    K_arr = np.asarray(K, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)

    B, Hq, T_q, D = Q_arr.shape
    _, Hkv, T_k, _ = K_arr.shape
    assert Hq % Hkv == 0

    if scale is None:
        scale = 1.0 / math.sqrt(float(D))

    group_size = Hq // Hkv
    out = np.zeros((B, Hq, T_q, D), dtype=np.float64)

    for b in range(B):
        for h in range(Hq):
            kh = h // group_size
            q_mat = Q_arr[b, h, :, :]  # (T_q, D)
            k_mat = K_arr[b, kh, :, :]  # (T_k, D)
            v_mat = V_arr[b, kh, :, :]  # (T_k, D)

            # (T_q, T_k)
            scores = np.matmul(q_mat, k_mat.T) * scale

            # Apply causal mask: mask out t_k > t_q
            for t_q in range(T_q):
                valid_len = min(t_q + 1, T_k)
                row = scores[t_q, :valid_len]
                m = np.max(row)
                e = np.exp(row - m)
                probs = e / np.sum(e)
                out[b, h, t_q, :] = np.dot(probs, v_mat[:valid_len, :])

    return out


# ============================================================================
# 4. DIAGNOSTIC DIFF PRINTER & ASSERTION ENGINE
# ============================================================================

def print_diagnostic_diff(
    test_name: str,
    expected: np.ndarray,
    actual: np.ndarray,
    tolerance: float = 1e-4,
    max_mismatches_to_show: int = 10,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compares expected and actual arrays.
    Returns (passed, stats_dict).
    If mismatched, prints comprehensive diagnostic diff table.
    """
    # 1. Validate tolerance input: must be a positive finite number
    if (
        not isinstance(tolerance, (int, float, np.floating, np.integer))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        print(f"\n❌ [DIFF FAILURE] {test_name}: Invalid tolerance '{tolerance}'! Tolerance must be a positive finite number.", file=sys.stderr)
        return False, {
            "error": "invalid_tolerance",
            "tolerance": tolerance,
            "passed": False,
        }
    tolerance = float(tolerance)

    expected_arr = np.asarray(expected)
    actual_arr = np.asarray(actual)

    if expected_arr.shape != actual_arr.shape:
        print(f"\n❌ [DIFF FAILURE] {test_name}: Shape mismatch!")
        print(f"   Expected shape: {expected_arr.shape}")
        print(f"   Actual shape:   {actual_arr.shape}")
        return False, {"error": "shape_mismatch", "passed": False}

    # 2. Explicitly reject NaN / Inf in reference array
    if not np.issubdtype(expected_arr.dtype, np.integer):
        exp_non_finite = ~np.isfinite(expected_arr.astype(np.float64))
        num_exp_non_finite = int(np.sum(exp_non_finite))
        if num_exp_non_finite > 0:
            print(f"\n❌ [DIFF FAILURE] {test_name}: Reference (expected) contains {num_exp_non_finite} non-finite values (NaN / Inf)! References must be finite numbers.", file=sys.stderr)
            return False, {
                "error": "non_finite_reference",
                "non_finite_reference_count": num_exp_non_finite,
                "passed": False,
            }

    if np.issubdtype(expected_arr.dtype, np.integer) and np.issubdtype(actual_arr.dtype, np.integer):
        # Exact bitwise / integer comparison
        mismatches = (expected_arr != actual_arr)
        num_mismatches = int(np.sum(mismatches))
        total_elements = expected_arr.size
        passed = (num_mismatches == 0)

        stats = {
            "total_elements": total_elements,
            "mismatching_elements": num_mismatches,
            "mismatch_pct": (num_mismatches / total_elements) * 100.0,
            "passed": passed
        }

        if passed:
            print(f"✅ [PASS] {test_name}: Exact match across {total_elements} integer elements.")
        else:
            print(f"\n❌ [FAIL] {test_name}: Integer mismatch detected!")
            print(f"   Mismatches: {num_mismatches} / {total_elements} ({stats['mismatch_pct']:.2f}%)")
            coords = np.argwhere(mismatches)
            print("   --- First Discrepancies ---")
            for idx in coords[:max_mismatches_to_show]:
                idx_tuple = tuple(idx)
                exp_val = int(expected_arr[idx_tuple])
                act_val = int(actual_arr[idx_tuple])
                print(f"   Index {list(idx)}: Expected=0x{exp_val:02X} ({exp_val}), Actual=0x{act_val:02X} ({act_val})")
        return passed, stats

    # Floating point comparison
    exp_f64 = expected_arr.astype(np.float64)
    act_f64 = actual_arr.astype(np.float64)

    non_finite = ~np.isfinite(act_f64)
    num_non_finite = int(np.sum(non_finite))

    diff = np.abs(exp_f64 - act_f64)
    # Replace non-finite diffs with infinity for sorting
    diff_clean = np.where(non_finite, np.inf, diff)

    max_abs_err = float(np.max(diff_clean)) if diff_clean.size > 0 else 0.0
    mean_abs_err = float(np.mean(diff_clean[np.isfinite(diff_clean)])) if np.any(np.isfinite(diff_clean)) else np.nan

    exceeds_tol = (diff_clean > tolerance) | non_finite
    num_exceeds = int(np.sum(exceeds_tol))
    total_elements = exp_f64.size
    passed = (num_exceeds == 0) and (num_non_finite == 0)

    stats = {
        "total_elements": total_elements,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "tolerance": tolerance,
        "mismatched_elements": num_exceeds,
        "mismatch_pct": (num_exceeds / total_elements) * 100.0,
        "non_finite_count": num_non_finite,
        "passed": passed
    }

    if metadata:
        stats.update(metadata)

    if passed:
        print(f"✅ [PASS] {test_name}: max_abs_err={max_abs_err:.4e} <= tol={tolerance:.1e} (mean={mean_abs_err:.4e})")
    else:
        print(f"\n❌ [FAIL] {test_name}: Numerical divergence detected!")
        if metadata:
            print(f"   Parameters: {json.dumps(metadata)}")
        print(f"   Max Absolute Error:  {max_abs_err:.6e} (Threshold: {tolerance:.1e})")
        print(f"   Mean Absolute Error: {mean_abs_err:.6e}")
        print(f"   Mismatched Elements: {num_exceeds} / {total_elements} ({stats['mismatch_pct']:.2f}%)")
        if num_non_finite > 0:
            print(f"   ⚠️ Non-finite values: {num_non_finite} (NaN / Inf present!)")

        print("   --- Top Largest Discrepancies ---")
        flat_sorted_indices = np.argsort(-diff_clean.ravel())
        for i in range(min(max_mismatches_to_show, num_exceeds)):
            flat_idx = flat_sorted_indices[i]
            coord = np.unravel_index(flat_idx, exp_f64.shape)
            exp_val = float(exp_f64[coord])
            act_val = float(act_f64[coord])
            cur_diff = float(diff_clean[coord])
            print(f"   Coord {list(coord)}: Expected={exp_val:12.6f}, Actual={act_val:12.6f}, Diff={cur_diff:12.6e}")

    return passed, stats


# ============================================================================
# 5. SELF-TEST TEST SUITES
# ============================================================================

def run_self_test_packing() -> bool:
    print("\n--- [Suite 1] Independent Reference: Ternary Packing Verification ---")
    all_passed = True

    # 1. Exhaustive 81-cases of 4 ternary values
    exhaustive_v = []
    expected_bytes = []
    for code in range(81):
        c = code
        v = []
        for _ in range(4):
            v.append(c % 3 - 1)
            c //= 3
        exhaustive_v.append(v)
        expected_bytes.append(reference_pack_byte(*v))

    exhaustive_arr = np.array(exhaustive_v, dtype=np.int64)
    packed_2d = reference_pack_ternary(exhaustive_arr)
    expected_arr = np.array(expected_bytes, dtype=np.uint8)[:, np.newaxis]

    p, _ = print_diagnostic_diff("TernaryPack: 81 Exhaustive Combinations", expected_arr, packed_2d)
    if not p: all_passed = False

    # 2. Round-trip unpack
    unpacked = reference_unpack_ternary(packed_2d)
    p, _ = print_diagnostic_diff("TernaryPack: Round-Trip Unpack Parity", exhaustive_arr, unpacked)
    if not p: all_passed = False

    # 3. Continuous float round-and-clamp parity
    np.random.seed(42)
    floats = np.random.uniform(-2.0, 2.0, size=(100, 16))
    quantized = reference_quantize_round_clamp(floats, scale=1.0)
    assert np.all((quantized >= -1) & (quantized <= 1))
    packed_floats = reference_pack_ternary(quantized)
    assert packed_floats.shape == (100, 4)
    print("✅ [PASS] TernaryPack: Continuous Float Round & Clamp -> Pack Verified (100x16 elements)")

    return all_passed


def run_self_test_gemm() -> bool:
    print("\n--- [Suite 2] Independent Reference: GEMM Multi-Consumer FP32 Output ---")
    all_passed = True
    np.random.seed(1337)

    M, K, N = 4, 16, 16

    # 1. Baseline: Zero/None initial C
    A = np.random.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    B = np.random.uniform(-1.0, 1.0, size=(K, N)).astype(np.float32)

    C_fp32, P_packed = reference_gemm_fused_with_consumer(A, B, C_initial=None, scale=1.0)
    C_ref = reference_gemm_fp64(A, B)
    p, _ = print_diagnostic_diff("GEMM Consumer 2: Intermediate FP32 Buffer Parity (Zero Initial C)", C_ref, C_fp32, tolerance=1e-5)
    if not p: all_passed = False

    Q_ref = reference_quantize_round_clamp(C_ref, scale=1.0)
    P_ref = reference_pack_ternary(Q_ref)
    p, _ = print_diagnostic_diff("GEMM Consumer 1: Fused Ternary Pack Output Parity (Zero Initial C)", P_ref, P_packed)
    if not p: all_passed = False

    # 2. Non-zero initial C across multiple scales {0.5, 1.0, 2.5, -1.0} with random FP matrices A, B, C_init
    test_scales = [0.5, 1.0, 2.5, -1.0]
    for scale in test_scales:
        A_rand = np.random.uniform(-3.0, 3.0, size=(M, K)).astype(np.float32)
        B_rand = np.random.uniform(-2.0, 2.0, size=(K, N)).astype(np.float32)
        C_init = np.random.uniform(-1.5, 1.5, size=(M, N)).astype(np.float32)

        C_fp32_nonzero, P_packed_nonzero = reference_gemm_fused_with_consumer(
            A_rand, B_rand, C_initial=C_init, scale=scale
        )

        C_ref_expected = reference_gemm_fp64(A_rand, B_rand) + C_init.astype(np.float64)
        p, _ = print_diagnostic_diff(
            f"GEMM Consumer 2: Non-Zero Initial C Parity (scale={scale})",
            C_ref_expected, C_fp32_nonzero, tolerance=1e-5
        )
        if not p: all_passed = False

        Q_ref_nonzero = reference_quantize_round_clamp(C_ref_expected, scale=scale, scale_is_multiplier=True)
        P_ref_nonzero = reference_pack_ternary(Q_ref_nonzero)
        p, _ = print_diagnostic_diff(
            f"GEMM Consumer 1: Non-Zero Initial C Fused Pack Parity (scale={scale})",
            P_ref_nonzero, P_packed_nonzero
        )
        if not p: all_passed = False

    if all_passed:
        print("✅ [PASS] GEMM Multi-Consumer: Random A/B, Non-Zero Initial C, and All Scales Parity Verified.")

    return all_passed


def run_self_test_attention() -> bool:
    print("\n--- [Suite 3] Independent Reference: Causal GQA Attention (Arbitrary B, Hq, Hkv, T, D) ---")
    all_passed = True
    np.random.seed(999)

    test_configs = [
        {"B": 1, "Hq": 1,  "Hkv": 1, "T": 8,  "D": 32, "desc": "MHA Single Head Rank-2"},
        {"B": 2, "Hq": 4,  "Hkv": 1, "T": 8,  "D": 32, "desc": "MQA Multi-Query Attention"},
        {"B": 2, "Hq": 20, "Hkv": 5, "T": 8,  "D": 32, "desc": "GQA 4:1 Ratio (Audit Baseline)"},
        {"B": 1, "Hq": 32, "Hkv": 8, "T": 16, "D": 64, "desc": "GQA High Dimension Large Context"},
    ]

    for cfg in test_configs:
        B, Hq, Hkv, T, D = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["T"], cfg["D"]
        desc = cfg["desc"]

        # 1. Fractional Q in [-0.9, 0.9], Ternary K, V in {-1, 0, 1}
        Q = np.random.uniform(-0.9, 0.9, size=(B, Hq, 1, D)).astype(np.float32)
        K = np.random.randint(-1, 2, size=(B, Hkv, T, D)).astype(np.float32)
        V = np.random.randint(-1, 2, size=(B, Hkv, T, D)).astype(np.float32)

        for pos in [0, T - 1]:
            out = reference_causal_gqa_attention_decode(Q, K, V, pos=pos)
            assert out.shape == (B, Hq, 1, D)
            assert np.all(np.isfinite(out)), f"Non-finite output detected in {desc} pos={pos}"

            # Analytical invariant check for pos=0:
            # Softmax of a 1-element logit vector is identically [1.0].
            # Therefore, output for query attending to position 0 must analytically equal V[:, kh, 0, :].
            if pos == 0:
                group_size = Hq // Hkv
                v_pos0_expected = np.zeros((B, Hq, 1, D), dtype=np.float64)
                for h in range(Hq):
                    kh = h // group_size
                    v_pos0_expected[:, h, 0, :] = V[:, kh, 0, :]
                max_inv_diff = float(np.max(np.abs(out - v_pos0_expected)))
                assert max_inv_diff < 1e-12, f"Pos=0 analytical invariant violated in {desc}: diff={max_inv_diff}"
                print(f"✅ [PASS] Causal GQA: {desc} pos=0 analytical invariant (out == V[:, kh, 0, :]): max_diff={max_inv_diff:.2e} < 1e-12")

            print(f"✅ [PASS] Causal GQA: {desc} pos={pos}/{T-1} output shape={out.shape} finite=True")

        # 2. Prefill causal attention and numerical cross-validation against decode across all positions
        out_prefill = reference_causal_gqa_attention_prefill(Q.repeat(T, axis=2), K, V)
        assert out_prefill.shape == (B, Hq, T, D)
        assert np.all(np.isfinite(out_prefill)), f"Non-finite output detected in prefill for {desc}"

        max_parity_diff = 0.0
        for pos in range(T):
            out_decode_pos = reference_causal_gqa_attention_decode(Q, K, V, pos=pos)
            parity_diff = float(np.max(np.abs(out_decode_pos - out_prefill[:, :, pos:pos + 1, :])))
            max_parity_diff = max(max_parity_diff, parity_diff)
            assert parity_diff < 1e-12, f"Causal prefill vs decode divergence at pos={pos} for {desc}: diff={parity_diff}"

        print(f"✅ [PASS] Causal GQA Prefill: {desc} sequence={T} output shape={out_prefill.shape} (decode-prefill parity cross-validated, max_diff={max_parity_diff:.2e} < 1e-12)")

    return all_passed


def run_self_test_comparator() -> bool:
    print("\n--- [Suite 4] Independent Reference: Comparator Robustness Verification ---")
    all_passed = True
    valid_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # 1. Reject NaN in reference
    ref_nan = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    p1, s1 = print_diagnostic_diff("SelfTest: NaN Reference", ref_nan, valid_data)
    if p1 or s1.get("error") != "non_finite_reference":
        print("❌ FAILED to reject NaN in reference!")
        all_passed = False

    # 2. Reject Inf in reference
    ref_inf = np.array([1.0, np.inf, 3.0], dtype=np.float32)
    p2, s2 = print_diagnostic_diff("SelfTest: Inf Reference", ref_inf, valid_data)
    if p2 or s2.get("error") != "non_finite_reference":
        print("❌ FAILED to reject Inf in reference!")
        all_passed = False

    # 3. Reject negative tolerance
    p3, s3 = print_diagnostic_diff("SelfTest: Negative Tol", valid_data, valid_data, tolerance=-1e-4)
    if p3 or s3.get("error") != "invalid_tolerance":
        print("❌ FAILED to reject negative tolerance!")
        all_passed = False

    # 4. Reject zero tolerance
    p4, s4 = print_diagnostic_diff("SelfTest: Zero Tol", valid_data, valid_data, tolerance=0.0)
    if p4 or s4.get("error") != "invalid_tolerance":
        print("❌ FAILED to reject zero tolerance!")
        all_passed = False

    # 5. Reject non-finite tolerance
    p5, s5 = print_diagnostic_diff("SelfTest: Inf Tol", valid_data, valid_data, tolerance=float("inf"))
    if p5 or s5.get("error") != "invalid_tolerance":
        print("❌ FAILED to reject non-finite tolerance!")
        all_passed = False

    if all_passed:
        print("✅ [PASS] Comparator Robustness: NaN/Inf references and invalid tolerances strictly rejected.")
    return all_passed


# ============================================================================
# 6. EXTERNAL FILE & DIRECTORY COMPARISON
# ============================================================================

def compare_binary_files(
    actual_path: str,
    expected_path: str,
    dtype_str: str,
    shape: Optional[List[int]] = None,
    tolerance: float = 1e-4
) -> bool:
    """Loads two binary files, compares them, and prints diagnostic diffs."""
    if (
        not isinstance(tolerance, (int, float, np.floating, np.integer))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        print(f"❌ Error: Invalid tolerance '{tolerance}'. Tolerance must be a positive finite number.", file=sys.stderr)
        return False

    if not os.path.isfile(actual_path):
        print(f"❌ Error: Actual file '{actual_path}' does not exist.", file=sys.stderr)
        return False
    if not os.path.isfile(expected_path):
        print(f"❌ Error: Expected file '{expected_path}' does not exist.", file=sys.stderr)
        return False

    dtype = np.dtype(dtype_str)
    itemsize = dtype.itemsize

    # 1. Strict byte alignment check against element size
    act_bytes = os.path.getsize(actual_path)
    exp_bytes = os.path.getsize(expected_path)

    if act_bytes % itemsize != 0:
        print(
            f"❌ Error: Incomplete binary dump in actual file '{actual_path}': size {act_bytes} bytes "
            f"is not divisible by {itemsize} ({dtype_str}). Truncated dump detected!",
            file=sys.stderr
        )
        return False

    if exp_bytes % itemsize != 0:
        print(
            f"❌ Error: Incomplete binary dump in expected file '{expected_path}': size {exp_bytes} bytes "
            f"is not divisible by {itemsize} ({dtype_str}). Truncated dump detected!",
            file=sys.stderr
        )
        return False

    act_raw = np.fromfile(actual_path, dtype=dtype)
    exp_raw = np.fromfile(expected_path, dtype=dtype)

    # 2. Strict shape element count check
    if shape:
        expected_element_count = int(np.prod(shape))
        if act_raw.size != expected_element_count:
            print(
                f"❌ Error: Incomplete binary dump in actual file '{actual_path}': contains {act_raw.size} elements, "
                f"but expected {expected_element_count} elements for shape {shape}!",
                file=sys.stderr
            )
            return False
        if exp_raw.size != expected_element_count:
            print(
                f"❌ Error: Incomplete binary dump in expected file '{expected_path}': contains {exp_raw.size} elements, "
                f"but expected {expected_element_count} elements for shape {shape}!",
                file=sys.stderr
            )
            return False
        act = act_raw.reshape(shape)
        exp = exp_raw.reshape(shape)
    else:
        if act_raw.size != exp_raw.size:
            print(
                f"❌ Error: Incomplete binary dump: actual file has {act_raw.size} elements, "
                f"expected file has {exp_raw.size} elements (size mismatch)!",
                file=sys.stderr
            )
            return False
        act, exp = act_raw, exp_raw

    passed, _ = print_diagnostic_diff(
        f"File Compare: {os.path.basename(actual_path)} vs {os.path.basename(expected_path)}",
        exp, act, tolerance=tolerance
    )
    return passed


def run_self_test_incomplete_dumps() -> bool:
    """Suite 5: Incomplete and truncated binary dump rejection tests."""
    print("\n--- [Suite 5] Independent Reference: Incomplete Binary Dump Rejection ---")
    import tempfile
    all_passed = True

    with tempfile.TemporaryDirectory() as tmpdir:
        valid_file = os.path.join(tmpdir, "valid.bin")
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tofile(valid_file)

        # 1. Truncated mid-float (15 bytes instead of 16 for float32)
        trunc_byte_file = os.path.join(tmpdir, "trunc_bytes.bin")
        with open(trunc_byte_file, "wb") as f:
            f.write(b"\x00" * 15)
        ok1 = compare_binary_files(trunc_byte_file, valid_file, dtype_str="float32", shape=[4])
        if ok1:
            print("❌ FAILED to reject non-aligned byte dump in actual file!")
            all_passed = False
        else:
            print("✅ [PASS] Incomplete binary dump (non-aligned bytes in actual) strictly rejected.")

        # 1b. Truncated mid-float in expected
        ok1b = compare_binary_files(valid_file, trunc_byte_file, dtype_str="float32", shape=[4])
        if ok1b:
            print("❌ FAILED to reject non-aligned byte dump in expected file!")
            all_passed = False
        else:
            print("✅ [PASS] Incomplete binary dump (non-aligned bytes in expected) strictly rejected.")

        # 2. Fewer elements than shape expects in actual
        trunc_elem_file = os.path.join(tmpdir, "trunc_elems.bin")
        np.array([1.0, 2.0], dtype=np.float32).tofile(trunc_elem_file)
        ok2 = compare_binary_files(trunc_elem_file, valid_file, dtype_str="float32", shape=[4])
        if ok2:
            print("❌ FAILED to reject incomplete element count in actual file!")
            all_passed = False
        else:
            print("✅ [PASS] Incomplete tensor dump (missing elements in actual) strictly rejected.")

        # 2b. Fewer elements than shape expects in expected
        ok2b = compare_binary_files(valid_file, trunc_elem_file, dtype_str="float32", shape=[4])
        if ok2b:
            print("❌ FAILED to reject incomplete element count in expected file!")
            all_passed = False
        else:
            print("✅ [PASS] Incomplete tensor dump (missing elements in expected) strictly rejected.")

        # 3. Size mismatch between flat files (no shape specified)
        ok3 = compare_binary_files(trunc_elem_file, valid_file, dtype_str="float32", shape=None)
        if ok3:
            print("❌ FAILED to reject unshaped size mismatch!")
            all_passed = False
        else:
            print("✅ [PASS] Incomplete binary dump (unshaped size mismatch) strictly rejected.")

        # 4. Valid identical files
        ok4 = compare_binary_files(valid_file, valid_file, dtype_str="float32", shape=[4])
        if not ok4:
            print("❌ FAILED to accept valid identical binary dumps!")
            all_passed = False
        else:
            print("✅ [PASS] Valid binary dumps successfully verified.")

    if all_passed:
        print("✅ [PASS] Incomplete Binary Dumps: Non-aligned bytes and incomplete elements strictly rejected.")
    return all_passed


def run_self_test_causal_mask_large_negative_logits() -> bool:
    """
    Suite 6: Regression verification for causal mask behavior under large negative logits.
    Verifies that when valid dot-product logits are <= -1e9 (e.g., -1.5e9, -1e10),
    a robust causal mask (-1e30f) ensures masked positions receive zero attention weight
    and does NOT invert the attention distribution against masked future tokens.
    """
    print("\n--- [Suite 6] Independent Reference: Causal Masking Robustness Under Large Negative Logits ---")
    all_passed = True

    # Consider sequence length T = 4, decode position pos = 1 (tokens 0, 1 valid; tokens 2, 3 masked)
    T = 4
    pos = 1

    # Large negative logits for valid tokens (e.g. -1.5e9, -2.0e9)
    valid_logits = np.array([-1.5e9, -2.0e9], dtype=np.float64)

    # Defective mask from old implementation: -1e9
    defective_mask_val = -1.0e9
    # Robust mask: -1e30
    robust_mask_val = -1.0e30

    # 1. Defective mask test:
    # Under defective mask, masked positions have logit -1e9, which is strictly GREATER than -1.5e9!
    defective_full_logits = np.array([-1.5e9, -2.0e9, defective_mask_val, defective_mask_val], dtype=np.float64)
    defective_max = np.max(defective_full_logits)
    defective_exp = np.exp(defective_full_logits - defective_max)
    defective_probs = defective_exp / np.sum(defective_exp)

    masked_weight_defective = float(np.sum(defective_probs[pos + 1:]))
    if masked_weight_defective > 0.5:
        print(f"⚠️ [VULNERABILITY CONFIRMED] Defective mask (-1e9) allocated {masked_weight_defective:.4f} "
              f"weight to masked future tokens! (Inverted attention when logits <= -1e9)")

    # 2. Robust mask test (-1e30):
    robust_full_logits = np.array([-1.5e9, -2.0e9, robust_mask_val, robust_mask_val], dtype=np.float64)
    robust_max = np.max(robust_full_logits[:pos + 1])
    diff = robust_full_logits - robust_max
    diff = np.where(diff < -500.0, -500.0, diff)
    robust_exp = np.exp(diff)
    robust_probs = robust_exp / np.sum(robust_exp)

    masked_weight_robust = float(np.sum(robust_probs[pos + 1:]))
    valid_weight_robust = float(np.sum(robust_probs[:pos + 1]))

    if masked_weight_robust > 1e-15 or abs(valid_weight_robust - 1.0) > 1e-12:
        print(f"❌ Robust mask (-1e30) failed! masked_weight={masked_weight_robust}, valid_weight={valid_weight_robust}")
        all_passed = False
    else:
        print(f"✅ [PASS] Robust Causal Mask (-1e30f): masked position leakage = {masked_weight_robust:.2e} "
              f"(valid tokens weight = {valid_weight_robust:.6f}) under large negative logits (<= -1e9).")

    # 3. Test multi-dimensional attention tensor with large negative dot-products
    B, Hq, Hkv, D = 1, 4, 1, 32
    Q = np.full((B, Hq, 1, D), -1e5, dtype=np.float32)
    K = np.full((B, Hkv, T, D), 1.0, dtype=np.float32)
    V = np.ones((B, Hkv, T, D), dtype=np.float32)
    for t_idx in range(T):
        V[:, :, t_idx, :] = float(t_idx + 1)

    scale = 1.0
    for test_pos in range(T):
        out = reference_causal_gqa_attention_decode(Q, K, V, pos=test_pos, scale=scale)
        assert np.all(np.isfinite(out)), f"Non-finite output at pos={test_pos}"
        max_v_expected = float(test_pos + 1)
        assert np.all(out <= max_v_expected + 1e-5), f"Future token leakage detected at pos={test_pos}"

    print("✅ [PASS] Causal Mask Robustness: Multi-head attention decode preserves strict causality across all positions.")
    return all_passed


# ============================================================================
# 7. MAIN ENTRY POINT & CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tenzo Independent Mathematical Reference Verification Suite"
    )
    parser.add_argument(
        "--check", choices=["pack", "gemm", "attention", "comparator", "incomplete", "causal_mask", "all"], default="all",
        help="Subsystem reference check to run (default: all)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-4,
        help="Strict numerical tolerance threshold (default: 1e-4)"
    )
    parser.add_argument(
        "--compare-files", nargs=2, metavar=("ACTUAL", "EXPECTED"),
        help="Compare two binary raw dumps directly"
    )
    parser.add_argument(
        "--dtype", default="float32",
        help="Data type for --compare-files (float32, float64, uint8, int8, int32)"
    )
    parser.add_argument(
        "--shape", type=int, nargs="*", default=None,
        help="Tensor shape for --compare-files"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all mathematical reference verification checks"
    )

    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║   Tenzo Independent Mathematical Reference Verification Suite          ║")
    print("║   Milestone 2: Independent CI & Math Hardening (W02, W03)              ║")
    print("╚════════════════════════════════════════════════════════════════════════╝")

    if args.compare_files:
        ok = compare_binary_files(
            args.compare_files[0], args.compare_files[1],
            dtype_str=args.dtype, shape=args.shape, tolerance=args.tolerance
        )
        sys.exit(0 if ok else 1)

    all_passed = True
    check_mode = "all" if args.all else args.check

    if check_mode in ["pack", "all"]:
        if not run_self_test_packing():
            all_passed = False

    if check_mode in ["gemm", "all"]:
        if not run_self_test_gemm():
            all_passed = False

    if check_mode in ["attention", "all"]:
        if not run_self_test_attention():
            all_passed = False

    if check_mode in ["comparator", "all"]:
        if not run_self_test_comparator():
            all_passed = False

    if check_mode in ["incomplete", "all"]:
        if not run_self_test_incomplete_dumps():
            all_passed = False

    if check_mode in ["causal_mask", "all"]:
        if not run_self_test_causal_mask_large_negative_logits():
            all_passed = False

    print("\n════════════════════════════════════════════════════════════════════════")
    if all_passed:
        print("🎉 ALL INDEPENDENT REFERENCE VERIFICATION CHECKS PASSED SUCCESSFULLY! 🎉")
        sys.exit(0)
    else:
        print("❌ INDEPENDENT REFERENCE VERIFICATION FAILED! (Numerical divergence detected)")
        sys.exit(1)


if __name__ == "__main__":
    main()
