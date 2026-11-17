#pragma once

#include <cstdint>
#include <string>

namespace mlir {
class MLIRContext;
}

namespace tenzo {

/// Test execution modes for the Audit JIT test harness
enum class AuditTestMode {
    All,
    PackStandalone,
    PackFused,
    FusionConsumer,
    AttentionExplicit,
    AttentionExplicitInteger,
    AttentionExplicitGQA,
    AttentionLinalg,
    AttentionLinalgGQA
};

/// 1. Standalone / Fused Ternary Pack Audit
/// - Tests 81 exhaustive ternary combinations with scaling factors {1.0, 0.5, -1.0}.
/// - Tests semantic parity against FP64 round-and-clamp: std::clamp(int(std::nearbyint(x)), -1, 1).
/// - Verifies output buffer canaries (0xCC) for overrun detection.
/// Returns true on PASS, false on FAIL (any mismatch).
bool auditTernaryPack(mlir::MLIRContext &context, bool fused = false);

/// 2. Matmul + Pack Fusion Multi-Consumer Audit
/// - Verifies that fusing Matmul + TernaryPack preserves intermediate FP32 buffer
///   for secondary consumers, supporting nonzero initial C and random A/B matrices.
/// - Compares secondary consumer and nonzero C output against independent double-precision GEMM reference.
/// - Verifies 16-element boundary canaries before and after buffers (a_buf, b_buf, c_buf, other_buf, packed_buf)
///   for overrun/corruption detection.
/// - Tolerance: 1e-4.
/// Returns true on PASS, false on FAIL.
bool auditFusionMultiConsumer(mlir::MLIRContext &context);

/// 3. Attention JIT Audit (Explicit SIMD Micro-Kernel or Linalg Lowering)
/// - Generates tenzo.packed_attention op.
/// - Compiles via MLIR ExecutionEngine.
/// - Computes independent causal double-precision attention reference:
///     s = (Q @ K^T) / sqrt(D), softmax(s), O = s @ V.
/// - Compares JIT output against reference with strict absolute tolerance 1e-4.
/// - Verifies output canary values (-789.0f) guarding against buffer overruns.
/// Returns true on PASS, false on FAIL.
bool auditAttention(mlir::MLIRContext &context,
                    bool linalg,
                    bool fractional,
                    int B,
                    int Hq,
                    int Hkv,
                    int T = 8,
                    int D = 32);

/// Convenience overload mapping boolean gqa flag to canonical dimensions (MHA vs GQA 2x20x5)
inline bool auditAttention(mlir::MLIRContext &context,
                           bool linalg,
                           bool fractional,
                           bool gqa,
                           int T = 8,
                           int D = 32) {
    int B = gqa ? 2 : 1;
    int Hq = gqa ? 20 : 1;
    int Hkv = gqa ? 5 : 1;
    return auditAttention(context, linalg, fractional, B, Hq, Hkv, T, D);
}

/// Convenience wrapper for Explicit Attention (GQA or MHA)
bool auditExplicitAttention(mlir::MLIRContext &context,
                           int B, int Hq, int Hkv, int T, int D,
                           bool fractional = true);

/// Convenience wrapper for Linalg Lowering Attention (GQA or MHA)
bool auditLinalgAttention(mlir::MLIRContext &context,
                          int B, int Hq, int Hkv, int T, int D,
                          bool fractional = true);

/// Master Audit JIT Runner
/// Executes all 8 audit test cases:
///   [1] Standalone Packing (81 exhaustive + round-clamp)
///   [2] Fused Matmul + Pack (81 exhaustive + round-clamp)
///   [3] Fusion Multi-Consumer (FP32 output preservation)
///   [4] Explicit Attention (Rank-2, MHA, fractional Q)
///   [5] Explicit Attention (Rank-2, MHA, integer Q)
///   [6] Explicit Attention (Rank-4, GQA, fractional Q)
///   [7] Linalg Attention Fallback (Rank-2, MHA)
///   [8] Linalg Attention GQA Fallback (Rank-4)
///
/// Returns true if ALL passed, false if ANY failed.
bool runAuditJITTests(mlir::MLIRContext &context, AuditTestMode mode = AuditTestMode::All);

} // namespace tenzo
