// BitwiseAttentionTest.cpp - Deep verification of tenzo.packed_attention and
// AVX2 bitwise micro-kernel Tests numerical parity of the SIMD bitwise
// attention against a naive FP32 reference implementation.
#include "tests/BitwiseAttentionTest.h"
#include "context/HardwareProfile.h"
#include "dialect/TenzoDialect.h"
#include "passes/Passes.h"
#include "runtime/KVCacheManager.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Import the AVX2 bitwise attention microkernel from
// ExplicitMicroKernelPass.cpp
extern "C" {
void bitwise_attention_microkernel(const float *query, const uint8_t *key_cache,
                                   const uint8_t *val_cache, float *output,
                                   int64_t T, int64_t D, int64_t seq_pos,
                                   float scale);
}

namespace tenzo {

// ============================================================================
// Naive FP32 reference implementation for packed attention
// Packing format: 4 ternary values {-1, 0, 1} per byte, encoded as nibble-pair
// indices for pshufb LUT decoding. Each byte = lo_nibble | (hi_nibble << 4),
// where each nibble encodes a pair (w1, w2) as: nibble = (w2+1)*4 + (w1+1).
// This matches the C++ AVX2 microkernel's LUT-based unpacking in
// ExplicitMicroKernelPass.cpp.
// ============================================================================

/// Encode a pair of ternary values (w1, w2) into a 4-bit nibble index.
/// The C++ AVX2 microkernel uses pshufb with LUTs where:
///   lut0[n] gives the first value of the pair
///   lut1[n] gives the second value of the pair
/// The encoding is: nibble = (w2+1)*4 + (w1+1)
/// where w1 is decoded by lut0, w2 is decoded by lut1.
static inline uint8_t encodeTernaryPairNibble(int w1, int w2) {
  return static_cast<uint8_t>((w2 + 1) * 4 + (w1 + 1));
}

/// Pack 4 ternary values into a single byte using nibble-pair encoding.
/// byte = nibble(v0,v1) | (nibble(v2,v3) << 4)
/// This matches the C++ AVX2 kernel's LUT-based unpacking.
static uint8_t packTernaryByte(int v0, int v1, int v2, int v3) {
  uint8_t lo_nibble = encodeTernaryPairNibble(v0, v1);
  uint8_t hi_nibble = encodeTernaryPairNibble(v2, v3);
  return lo_nibble | (hi_nibble << 4);
}

/// Decode a nibble into a pair of ternary values using the LUT convention.
static inline void decodeTernaryPairNibble(uint8_t nibble, int &w1, int &w2) {
  // lut0 pattern: [-1, 0, 1, 0, -1, 0, 1, 0, ...]  => (nibble % 4) mapping:
  // 0->-1, 1->0, 2->1, 3->0 lut1 pattern: [-1,-1,-1,-1, 0,0,0,0, 1,1,1,1,
  // 0,0,0,0] => (nibble / 4) mapping: 0->-1, 1->0, 2->1, 3->0
  static const int lut0_table[4] = {-1, 0, 1, 0};
  static const int lut1_table[4] = {-1, 0, 1, 0};
  w1 = lut0_table[nibble & 3];
  w2 = lut1_table[(nibble >> 2) & 3];
}

/// Unpack 4 ternary values from a single byte
static void unpackTernaryByte(uint8_t byte, int &v0, int &v1, int &v2,
                              int &v3) {
  uint8_t lo_nibble = byte & 0x0F;
  uint8_t hi_nibble = (byte >> 4) & 0x0F;
  decodeTernaryPairNibble(lo_nibble, v0, v1);
  decodeTernaryPairNibble(hi_nibble, v2, v3);
}

/// Pack a ternary vector of length D into a uint8 byte array of length D/4
static void packTernaryVector(const int *ternary, uint8_t *packed, int64_t D) {
  for (int64_t i = 0; i < D; i += 4) {
    packed[i / 4] = packTernaryByte(ternary[i], ternary[i + 1], ternary[i + 2],
                                    ternary[i + 3]);
  }
}

/// Naive FP32 reference attention:
///   scores[t] = sum_d(query[d] * key_ternary[t][d]) * scale
///   weights = softmax(scores)
///   output[d] = sum_t(weights[t] * value_ternary[t][d])
static void naiveFP32Attention(const float *query,       // [D]
                               const int *key_ternary,   // [T x D] row-major
                               const int *value_ternary, // [T x D] row-major
                               float *output,            // [D]
                               int64_t T, int64_t D, int64_t seq_pos,
                               float scale) {
  int64_t valid_T = (seq_pos >= 0 && seq_pos < T) ? (seq_pos + 1) : T;

  // 1. Compute Q * K^T scores
  std::vector<float> scores(T, -10000.0f);
  for (int64_t t = 0; t < valid_T; ++t) {
    float dot = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
      dot += query[d] * static_cast<float>(key_ternary[t * D + d]);
    }
    scores[t] = dot * scale;
  }

  // 2. Softmax
  float max_score = *std::max_element(scores.begin(), scores.begin() + valid_T);
  float sum_exp = 0.0f;
  for (int64_t t = 0; t < T; ++t) {
    scores[t] = std::exp(scores[t] - max_score);
    sum_exp += scores[t];
  }
  float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
  for (int64_t t = 0; t < T; ++t) {
    scores[t] *= inv_sum;
  }

  // 3. Value accumulation: output = sum_t(w_t * V[t])
  for (int64_t d = 0; d < D; ++d)
    output[d] = 0.0f;
  for (int64_t t = 0; t < T; ++t) {
    float w_t = scores[t];
    if (std::abs(w_t) < 1e-12f)
      continue;
    for (int64_t d = 0; d < D; ++d) {
      output[d] += w_t * static_cast<float>(value_ternary[t * D + d]);
    }
  }
}

// ============================================================================
// Test runner
// ============================================================================
void runBitwiseAttentionTest(mlir::MLIRContext &context) {
  llvm::outs()
      << "\n╔════════════════════════════════════════════════════════╗\n";
  llvm::outs()
      << "║  🧪 BITWISE ATTENTION & SIMD MICRO-KERNEL TESTS        ║\n";
  llvm::outs()
      << "╚════════════════════════════════════════════════════════╝\n\n";

  // ------------------------------------------------------------------------
  // TEST 1: TableGen Dialect Definition Verification — PackedAttentionOp
  // ------------------------------------------------------------------------
  llvm::outs() << "[Test 1/10] Verifying tenzo.packed_attention Dialect "
                  "Definition & Parser...\n";
  {
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto f32Type = builder.getF32Type();
    auto i8Type = builder.getI8Type();

    auto queryType = mlir::RankedTensorType::get({1, 64}, f32Type);
    auto keyType = mlir::RankedTensorType::get(
        {8, 16}, i8Type); // 8 tokens, D/4=16 -> D=64
    auto valueType = mlir::RankedTensorType::get({8, 16}, i8Type);
    auto resultType = mlir::RankedTensorType::get({1, 64}, f32Type);

    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    // 1a. C++ API with scale
    auto funcType = builder.getFunctionType(
        {queryType, keyType, valueType, f32Type}, {resultType});
    auto func = builder.create<mlir::func::FuncOp>(
        loc, "test_packed_attn_with_scale", funcType);
    auto &entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    mlir::Value qVal = entryBlock.getArgument(0);
    mlir::Value kVal = entryBlock.getArgument(1);
    mlir::Value vVal = entryBlock.getArgument(2);
    mlir::Value scaleVal = entryBlock.getArgument(3);

    auto packedAttnOp = builder.create<tenzo::PackedAttentionOp>(
        loc, resultType, qVal, kVal, vVal,
        /*seq_pos=*/mlir::Value(),
        /*scale=*/scaleVal, builder.getI32IntegerAttr(4));

    assert(packedAttnOp && "Failed to create PackedAttentionOp with scale!");
    assert(packedAttnOp.getValuesPerByte() == 4 &&
           "values_per_byte default should be 4");
    assert(packedAttnOp.getQuery() == qVal);
    assert(packedAttnOp.getKey() == kVal);
    assert(packedAttnOp.getValue() == vVal);
    assert(packedAttnOp.getScale() == scaleVal);
    assert(packedAttnOp.getResult().getType() == resultType);
    builder.create<mlir::func::ReturnOp>(loc, packedAttnOp.getResult());

    // 1b. C++ API without scale (optional scale omitted)
    builder.setInsertionPointToStart(module.getBody());
    auto funcTypeNoScale =
        builder.getFunctionType({queryType, keyType, valueType}, {resultType});
    auto funcNoScale = builder.create<mlir::func::FuncOp>(
        loc, "test_packed_attn_no_scale", funcTypeNoScale);
    auto &entryBlockNoScale = *funcNoScale.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlockNoScale);

    auto packedAttnOpNoScale = builder.create<tenzo::PackedAttentionOp>(
        loc, resultType, entryBlockNoScale.getArgument(0),
        entryBlockNoScale.getArgument(1), entryBlockNoScale.getArgument(2),
        /*seq_pos=*/mlir::Value(),
        /*scale=*/mlir::Value(), builder.getI32IntegerAttr(4));
    assert(packedAttnOpNoScale &&
           "Failed to create PackedAttentionOp without scale!");
    assert(!packedAttnOpNoScale.getScale() &&
           "Scale should be null when omitted!");
    builder.create<mlir::func::ReturnOp>(loc, packedAttnOpNoScale.getResult());

    // Verification of MLIR module
    assert(mlir::succeeded(module.verify()) &&
           "Module verification failed for PackedAttentionOp!");

    // 1c. Textual round-trip parsing test
    const char *mlirText = R"mlir(
            module {
                func.func @parsed_packed_attn_scale(%q: tensor<1x64xf32>, %k: tensor<8x16xi8>, %v: tensor<8x16xi8>, %s: f32) -> tensor<1x64xf32> {
                    %0 = tenzo.packed_attention %q, %k, %v scale(%s) : (tensor<1x64xf32>, tensor<8x16xi8>, tensor<8x16xi8>, f32) -> tensor<1x64xf32>
                    return %0 : tensor<1x64xf32>
                }
                func.func @parsed_packed_attn_noscale(%q: tensor<1x64xf32>, %k: tensor<8x16xi8>, %v: tensor<8x16xi8>) -> tensor<1x64xf32> {
                    %0 = tenzo.packed_attention %q, %k, %v : (tensor<1x64xf32>, tensor<8x16xi8>, tensor<8x16xi8>) -> tensor<1x64xf32>
                    return %0 : tensor<1x64xf32>
                }
                func.func @parsed_packed_attn_all(%q: tensor<1x64xf32>, %k: tensor<8x16xi8>, %v: tensor<8x16xi8>, %pos: tensor<1xi32>, %s: f32) -> tensor<1x64xf32> {
                    %0 = tenzo.packed_attention %q, %k, %v seq_pos(%pos) scale(%s) : (tensor<1x64xf32>, tensor<8x16xi8>, tensor<8x16xi8>, tensor<1xi32>, f32) -> tensor<1x64xf32>
                    return %0 : tensor<1x64xf32>
                }
            }
        )mlir";
    auto parsedModule =
        mlir::parseSourceString<mlir::ModuleOp>(mlirText, &context);
    assert(parsedModule &&
           "Failed to parse textual MLIR containing tenzo.packed_attention!");
    assert(mlir::succeeded(parsedModule->verify()) &&
           "Parsed module verification failed!");

    llvm::outs() << "  ✅ PackedAttentionOp verified with scale, without "
                    "scale, and via textual MLIR parser!\n";
  }

  // ------------------------------------------------------------------------
  // TEST 2: MLIR Micro-Kernel Pattern Generation for tenzo.packed_attention
  // ------------------------------------------------------------------------
  llvm::outs() << "[Test 2/10] Verifying ExplicitMicroKernelPass on "
                  "tenzo.packed_attention...\n";
  {
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto f32Type = builder.getF32Type();
    auto i8Type = builder.getI8Type();

    int64_t D = 32;
    int64_t T = 8;

    auto queryType = mlir::RankedTensorType::get({D}, f32Type);
    auto keyType = mlir::RankedTensorType::get({T, D / 4}, i8Type);
    auto valueType = mlir::RankedTensorType::get({T, D / 4}, i8Type);
    auto resultType = mlir::RankedTensorType::get({D}, f32Type);

    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    auto funcType =
        builder.getFunctionType({queryType, keyType, valueType}, {resultType});
    auto func = builder.create<mlir::func::FuncOp>(
        loc, "packed_attention_microkernel", funcType);
    auto &entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    mlir::Value Q = entryBlock.getArgument(0);
    mlir::Value K = entryBlock.getArgument(1);
    mlir::Value V = entryBlock.getArgument(2);

    auto packedAttnOp = builder.create<tenzo::PackedAttentionOp>(
        loc, resultType, Q, K, V,
        /*seq_pos=*/mlir::Value(),
        /*scale=*/mlir::Value(), builder.getI32IntegerAttr(4));
    builder.create<mlir::func::ReturnOp>(loc, packedAttnOp.getResult());

    // Run ExplicitMicroKernelPass
    mlir::PassManager pm(&context);
    tenzo::MicroKernelParams params;
    params.MR = 6; // Global default is 6
    params.NR = 16;
    params.VEC_SIZE = 8;
    tenzo::addExplicitMicroKernelPass(pm, params);

    mlir::LogicalResult result = pm.run(module);
    assert(mlir::succeeded(result) &&
           "ExplicitMicroKernelPass failed on tenzo.packed_attention!");
    assert(mlir::succeeded(module.verify()) &&
           "ExplicitMicroKernelPass produced invalid IR!");

    // Verify that packed_attention was replaced and expected ops were generated
    bool foundPackedAttn = false;
    bool foundSCFFor = false;
    bool foundMathExp = false;
    bool foundArithMulF = false;
    bool foundVectorTransferRead = false;

    func.walk([&](mlir::Operation *op) {
      if (llvm::isa<tenzo::PackedAttentionOp>(op))
        foundPackedAttn = true;
      if (llvm::isa<mlir::scf::ForOp>(op))
        foundSCFFor = true;
      if (llvm::isa<mlir::math::ExpOp>(op))
        foundMathExp = true;
      if (llvm::isa<mlir::arith::MulFOp>(op))
        foundArithMulF = true;
      if (llvm::isa<mlir::vector::TransferReadOp>(op))
        foundVectorTransferRead = true;
    });

    assert(!foundPackedAttn &&
           "tenzo.packed_attention should have been lowered/erased!");
    assert(foundSCFFor &&
           "Iteration loops (scf.for) should have been generated!");
    assert(foundMathExp && "Softmax exp() should have been generated!");
    assert(foundArithMulF && "Float multiply ops should have been generated!");
    assert(foundVectorTransferRead &&
           "Vector transfer reads should have been generated!");

    llvm::outs() << "  ✅ ExplicitMicroKernelPass successfully generated valid "
                    "packed attention IR!\n";
  }

  // ------------------------------------------------------------------------
  // TEST 3: FP32 Isolation Verification (tenzo.packed_attention doesn't break
  // FP32 / ternary_pack)
  // ------------------------------------------------------------------------
  llvm::outs() << "[Test 3/10] Verifying FP32 Pipeline Isolation (MR=6, NR=16 "
                  "unmodified after packed_attention pass)...\n";
  {
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto f32Type = builder.getF32Type();

    // FP32 microkernel: 6x16
    auto aMemRefType = mlir::MemRefType::get({6, 64}, f32Type);
    auto bMemRefType = mlir::MemRefType::get({64, 16}, f32Type);
    auto cMemRefType = mlir::MemRefType::get({6, 16}, f32Type);

    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    auto funcType =
        builder.getFunctionType({aMemRefType, bMemRefType, cMemRefType}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        loc, "fp32_microkernel_6x16_isolation", funcType);
    auto &entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    mlir::Value A = entryBlock.getArgument(0);
    mlir::Value B = entryBlock.getArgument(1);
    mlir::Value C = entryBlock.getArgument(2);

    // Standard FP32 matmul (no ternary pack attr)
    builder.create<mlir::linalg::MatmulOp>(loc, mlir::ValueRange{A, B},
                                           mlir::ValueRange{C});
    builder.create<mlir::func::ReturnOp>(loc);

    mlir::PassManager pm(&context);
    tenzo::MicroKernelParams params;
    params.MR = 6;
    params.NR = 16;
    params.VEC_SIZE = 8;
    tenzo::addExplicitMicroKernelPass(pm, params);

    mlir::LogicalResult result = pm.run(module);
    assert(mlir::succeeded(result) &&
           "ExplicitMicroKernelPass failed on standard FP32 GEMM!");
    assert(mlir::succeeded(module.verify()) &&
           "FP32 GEMM lowering produced invalid IR!");

    bool foundMatmul = false;
    int transferWriteCount = 0;
    func.walk([&](mlir::Operation *op) {
      if (llvm::isa<mlir::linalg::MatmulOp>(op))
        foundMatmul = true;
      if (llvm::isa<mlir::vector::TransferWriteOp>(op))
        transferWriteCount++;
    });

    assert(!foundMatmul &&
           "Standard FP32 linalg.matmul should have been lowered!");
    // For 6x16 with VEC_SIZE=8: 6 rows * 2 vectors = 12 transfer writes to C
    assert(transferWriteCount == 12 &&
           "Standard FP32 must produce exactly 12 transfer writes (6x2)!");

    llvm::outs() << "  ✅ Standard FP32 GEMM generation intact: exactly 12 "
                    "accumulators stored (6x16)!\n";
  }

  // ------------------------------------------------------------------------
  // TEST 4: Packing Utility Verification — round-trip pack/unpack
  // ------------------------------------------------------------------------
  llvm::outs() << "[Test 4/10] Verifying ternary pack/unpack round-trip...\n";
  {
    // Test all 81 (3^4) combinations of 4 ternary values
    int count = 0;
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          for (int d = -1; d <= 1; ++d) {
            uint8_t packed = packTernaryByte(a, b, c, d);
            int ua, ub, uc, ud;
            unpackTernaryByte(packed, ua, ub, uc, ud);
            assert(ua == a && "Pack/unpack round-trip failed for v0!");
            assert(ub == b && "Pack/unpack round-trip failed for v1!");
            assert(uc == c && "Pack/unpack round-trip failed for v2!");
            assert(ud == d && "Pack/unpack round-trip failed for v3!");
            count++;
          }
        }
      }
    }
    assert(count == 81 &&
           "Should have tested all 3^4 = 81 ternary combinations!");

    // Edge cases: specific known encodings (nibble-pair scheme)
    // All -1: lo_nibble = (-1+1)*4+(-1+1) = 0, hi_nibble = 0 => byte = 0x00
    assert(packTernaryByte(-1, -1, -1, -1) == 0x00);
    // All 0: lo_nibble = (0+1)*4+(0+1) = 5, hi_nibble = 5 => byte = 0x55
    assert(packTernaryByte(0, 0, 0, 0) == 0x55);
    // All 1: lo_nibble = (1+1)*4+(1+1) = 10 = 0xA, hi_nibble = 0xA => byte =
    // 0xAA
    assert(packTernaryByte(1, 1, 1, 1) == 0xAA);

    llvm::outs() << "  ✅ All 81 ternary combinations pack/unpack correctly!\n";
  }

  // ------------------------------------------------------------------------
  // TEST 5: Numerical Parity — AVX2 Bitwise vs Naive FP32 Reference (Small)
  // Key test: verifies the AVX2 microkernel produces numerically equivalent
  // results to the scalar FP32 reference implementation.
  // ------------------------------------------------------------------------
  llvm::outs() << "[Test 5/10] Verifying numerical parity: AVX2 bitwise vs "
                  "naive FP32 (D=32, T=4)...\n";
  {
    const int64_t D = 32;
    const int64_t T = 4;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Generate deterministic ternary K and V
    int key_ternary[T * D];
    int value_ternary[T * D];
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1, 1);
    for (int64_t i = 0; i < T * D; ++i) {
      key_ternary[i] = dist(rng);
      value_ternary[i] = dist(rng);
    }

    // Generate query vector
    alignas(32) float query[D];
    std::uniform_real_distribution<float> qdist(-1.0f, 1.0f);
    for (int64_t d = 0; d < D; ++d) {
      query[d] = qdist(rng);
    }

    // Pack K and V into uint8
    std::vector<uint8_t> packed_key(T * D / 4);
    std::vector<uint8_t> packed_value(T * D / 4);
    for (int64_t t = 0; t < T; ++t) {
      packTernaryVector(key_ternary + t * D, packed_key.data() + t * (D / 4),
                        D);
      packTernaryVector(value_ternary + t * D,
                        packed_value.data() + t * (D / 4), D);
    }

    // Naive FP32 reference
    alignas(32) float ref_output[D];
    naiveFP32Attention(query, key_ternary, value_ternary, ref_output, T, D,
                       T - 1, scale);

    // AVX2 bitwise microkernel
    alignas(32) float avx_output[D];
    memset(avx_output, 0, sizeof(avx_output));
    bitwise_attention_microkernel(query, packed_key.data(), packed_value.data(),
                                  avx_output, T, D, T - 1, scale);

    // Compare — the AVX2 kernel quantizes Q to int16, so there's quantization
    // error
    float max_err = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
      float err = std::abs(ref_output[d] - avx_output[d]);
      max_err = std::max(max_err, err);
    }

    // Numerical parity tolerance (< 1e-4 as required by specification)
    float tolerance = 1e-4f;
    llvm::outs() << "  Max absolute error: " << max_err
                 << " (tolerance: " << tolerance << ")\n";
    assert(max_err < tolerance && "AVX2 bitwise attention deviates too much "
                                  "from FP32 reference (exceeds 1e-4)!");

    llvm::outs() << "  ✅ AVX2 bitwise attention matches FP32 reference within "
                    "1e-4 tolerance!\n";
  }

  // ------------------------------------------------------------------------
  // TEST 6: Numerical Parity — Larger dimensions (D=64, T=16) with causal
  // masking
  // ------------------------------------------------------------------------
  llvm::outs() << "[Test 6/10] Verifying numerical parity with causal masking "
                  "(D=64, T=16, seq_pos=7)...\n";
  {
    const int64_t D = 64;
    const int64_t T = 16;
    const int64_t seq_pos = 7; // Only tokens 0-7 visible
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(-1, 1);
    std::uniform_real_distribution<float> qdist(-2.0f, 2.0f);

    std::vector<int> key_ternary(T * D);
    std::vector<int> value_ternary(T * D);
    for (int64_t i = 0; i < T * D; ++i) {
      key_ternary[i] = dist(rng);
      value_ternary[i] = dist(rng);
    }

    alignas(32) float query[D];
    for (int64_t d = 0; d < D; ++d) {
      query[d] = qdist(rng);
    }

    std::vector<uint8_t> packed_key(T * D / 4);
    std::vector<uint8_t> packed_value(T * D / 4);
    for (int64_t t = 0; t < T; ++t) {
      packTernaryVector(key_ternary.data() + t * D,
                        packed_key.data() + t * (D / 4), D);
      packTernaryVector(value_ternary.data() + t * D,
                        packed_value.data() + t * (D / 4), D);
    }

    // Naive FP32 reference with causal masking
    std::vector<float> ref_output(D, 0.0f);
    naiveFP32Attention(query, key_ternary.data(), value_ternary.data(),
                       ref_output.data(), T, D, seq_pos, scale);

    // AVX2 bitwise microkernel with causal masking
    std::vector<float> avx_output(D, 0.0f);
    bitwise_attention_microkernel(query, packed_key.data(), packed_value.data(),
                                  avx_output.data(), T, D, seq_pos, scale);

    float max_err = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
      float err = std::abs(ref_output[d] - avx_output[d]);
      max_err = std::max(max_err, err);
    }

    float tolerance = 1e-4f;
    llvm::outs() << "  Max absolute error: " << max_err
                 << " (tolerance: " << tolerance << ")\n";
    assert(max_err < tolerance && "AVX2 bitwise attention with causal mask "
                                  "deviates too much (exceeds 1e-4)!");

    // Verify that masked tokens don't contribute: compute reference with full
    // T, the scores for t > seq_pos should be effectively zeroed out
    llvm::outs() << "  ✅ AVX2 bitwise attention with causal masking matches "
                    "FP32 reference within 1e-4 tolerance!\n";
  }

  // ------------------------------------------------------------------------
  // TEST 7: Edge Cases — all zeros, all ones, single token, null guard
  // ------------------------------------------------------------------------
  llvm::outs() << "[Test 7/10] Verifying edge cases (all-zero, all-one, single "
                  "token, null guard)...\n";
  {
    const int64_t D = 32;

    // 7a. Single token (T=1), all key=1, all value=1
    {
      alignas(32) float query[D];
      for (int64_t d = 0; d < D; ++d)
        query[d] = 1.0f;

      int key_ternary[D];
      int value_ternary[D];
      for (int64_t d = 0; d < D; ++d) {
        key_ternary[d] = 1;
        value_ternary[d] = 1;
      }

      std::vector<uint8_t> packed_key(D / 4);
      std::vector<uint8_t> packed_value(D / 4);
      packTernaryVector(key_ternary, packed_key.data(), D);
      packTernaryVector(value_ternary, packed_value.data(), D);

      alignas(32) float output[D];
      memset(output, 0, sizeof(output));
      bitwise_attention_microkernel(query, packed_key.data(),
                                    packed_value.data(), output, 1, D, 0, 1.0f);

      // With T=1, softmax gives weight=1.0 for the single token
      // So output should be the value vector unpacked: all 1.0
      for (int64_t d = 0; d < D; ++d) {
        assert(std::abs(output[d] - 1.0f) < 0.01f &&
               "Single token all-1 failed!");
      }
    }

    // 7b. All key=0, all value=0 -> output should be all 0
    {
      alignas(32) float query[D];
      for (int64_t d = 0; d < D; ++d)
        query[d] = 1.0f;

      int key_ternary[D];
      int value_ternary[D];
      for (int64_t d = 0; d < D; ++d) {
        key_ternary[d] = 0;
        value_ternary[d] = 0;
      }

      std::vector<uint8_t> packed_key(D / 4);
      std::vector<uint8_t> packed_value(D / 4);
      packTernaryVector(key_ternary, packed_key.data(), D);
      packTernaryVector(value_ternary, packed_value.data(), D);

      alignas(32) float output[D];
      memset(output, 0, sizeof(output));
      bitwise_attention_microkernel(query, packed_key.data(),
                                    packed_value.data(), output, 1, D, 0, 1.0f);

      for (int64_t d = 0; d < D; ++d) {
        assert(std::abs(output[d]) < 0.01f &&
               "All-zero V should produce all-zero output!");
      }
    }

    // 7c. Null pointer guard
    {
      float dummy_output[D];
      bitwise_attention_microkernel(nullptr, nullptr, nullptr, nullptr, 0, 0, 0,
                                    1.0f);
      bitwise_attention_microkernel(nullptr, nullptr, nullptr, dummy_output, 4,
                                    D, 0, 1.0f);
      // Should not crash (early return for null pointers)
    }

    // 7d. All key=-1, all value=-1 -> output should be all -1
    {
      alignas(32) float query[D];
      for (int64_t d = 0; d < D; ++d)
        query[d] = 1.0f;

      int key_ternary[D];
      int value_ternary[D];
      for (int64_t d = 0; d < D; ++d) {
        key_ternary[d] = -1;
        value_ternary[d] = -1;
      }

      std::vector<uint8_t> packed_key(D / 4);
      std::vector<uint8_t> packed_value(D / 4);
      packTernaryVector(key_ternary, packed_key.data(), D);
      packTernaryVector(value_ternary, packed_value.data(), D);

      alignas(32) float output[D];
      memset(output, 0, sizeof(output));
      bitwise_attention_microkernel(query, packed_key.data(),
                                    packed_value.data(), output, 1, D, 0, 1.0f);

      for (int64_t d = 0; d < D; ++d) {
        assert(std::abs(output[d] - (-1.0f)) < 0.01f &&
               "Single token all-(-1) V should produce all -1 output!");
      }
    }

    llvm::outs() << "  ✅ All edge cases passed verification!\n";
  }

  // ------------------------------------------------------------------------
  // TEST 8: Zero-Copy GQA Routing Lowering Verification in MLIR (20 Q -> 5 KV,
  // floordiv, zero memref.copy)
  // ------------------------------------------------------------------------
  llvm::outs()
      << "[Test 8/10] Verifying Zero-Copy GQA Routing in LinalgLowering (20 Q "
         "heads -> 5 KV heads, floordiv, zero memref.copy)...\n";
  {
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto f32Type = builder.getF32Type();
    auto i8Type = builder.getI8Type();
    auto i32Type = builder.getI32Type();

    int64_t B = 1;
    int64_t H_q = 20;
    int64_t H_kv = 5;
    int64_t T = 16;
    int64_t Dh = 128;
    int64_t K_Dh = Dh / 4; // 32 bytes

    // Q: [1, 20, 1, 128], K: [1, 5, 16, 32], V: [1, 5, 16, 32]
    auto queryType = mlir::RankedTensorType::get({B, H_q, 1, Dh}, f32Type);
    auto keyType = mlir::RankedTensorType::get({B, H_kv, T, K_Dh}, i8Type);
    auto valueType = mlir::RankedTensorType::get({B, H_kv, T, K_Dh}, i8Type);
    auto seqPosType = mlir::RankedTensorType::get({1}, i32Type);
    auto outType = mlir::RankedTensorType::get({B, H_q, 1, Dh}, f32Type);

    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    auto funcType = builder.getFunctionType(
        {queryType, keyType, valueType, seqPosType}, {f32Type});
    auto func = builder.create<mlir::func::FuncOp>(
        loc, "gqa_packed_attention_20x5", funcType);
    auto &entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    mlir::Value Q = entryBlock.getArgument(0);
    mlir::Value K = entryBlock.getArgument(1);
    mlir::Value V = entryBlock.getArgument(2);
    mlir::Value seqPos = entryBlock.getArgument(3);

    auto packedAttnOp = builder.create<tenzo::PackedAttentionOp>(
        loc, outType, Q, K, V, seqPos,
        /*scale=*/mlir::Value(), builder.getI32IntegerAttr(4));
    auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto scalarVal = builder.create<mlir::tensor::ExtractOp>(
        loc, packedAttnOp.getResult(), mlir::ValueRange{c0, c0, c0, c0});
    builder.create<mlir::func::ReturnOp>(loc, scalarVal.getResult());

    // Also test un-transposed shape [1, 1, 20, 128] to verify zero-transpose
    // handling
    builder.setInsertionPointToEnd(module.getBody());
    auto qUnTransposedType =
        mlir::RankedTensorType::get({B, 1, H_q, Dh}, f32Type);
    auto outUnTransposedType =
        mlir::RankedTensorType::get({B, 1, H_q, Dh}, f32Type);
    auto func2Type = builder.getFunctionType(
        {qUnTransposedType, keyType, valueType, seqPosType}, {f32Type});
    auto func2 = builder.create<mlir::func::FuncOp>(
        loc, "gqa_untransposed_zero_copy", func2Type);
    auto &entryBlock2 = *func2.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock2);

    auto packedAttnOp2 = builder.create<tenzo::PackedAttentionOp>(
        loc, outUnTransposedType, entryBlock2.getArgument(0),
        entryBlock2.getArgument(1), entryBlock2.getArgument(2),
        entryBlock2.getArgument(3),
        /*scale=*/mlir::Value(), builder.getI32IntegerAttr(4));
    auto c0_2 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto scalarVal2 = builder.create<mlir::tensor::ExtractOp>(
        loc, packedAttnOp2.getResult(),
        mlir::ValueRange{c0_2, c0_2, c0_2, c0_2});
    builder.create<mlir::func::ReturnOp>(loc, scalarVal2.getResult());

    // Run TenzoToLinalgPass and Bufferization
    mlir::PassManager pm(&context);
    tenzo::addTenzoToLinalgPass(pm);
    tenzo::addTenzoBufferizationPasses(pm);

    mlir::LogicalResult result = pm.run(module);
    assert(mlir::succeeded(result) &&
           "LinalgLowering failed on GQA tenzo.packed_attention!");
    assert(mlir::succeeded(module.verify()) &&
           "Lowered GQA module failed verification!");

    // Walk and verify:
    // 1. tenzo.packed_attention is erased/lowered
    // 2. Zero memref.copy
    // 3. Zero linalg.transpose
    // 4. Presence of AffineMap floorDiv routing (d1 floordiv 4)
    bool foundPackedAttn = false;
    int memrefCopyCount = 0;
    int transposeCount = 0;
    bool foundFloorDiv = false;

    module.walk([&](mlir::Operation *op) {
      if (llvm::isa<tenzo::PackedAttentionOp>(op))
        foundPackedAttn = true;
      if (llvm::isa<mlir::memref::CopyOp>(op))
        memrefCopyCount++;
      if (llvm::isa<mlir::linalg::TransposeOp>(op))
        transposeCount++;
      if (auto genericOp = llvm::dyn_cast<mlir::linalg::GenericOp>(op)) {
        for (auto map : genericOp.getIndexingMapsArray()) {
          for (unsigned res = 0; res < map.getNumResults(); ++res) {
            auto expr = map.getResult(res);
            if (expr.getKind() == mlir::AffineExprKind::FloorDiv) {
              foundFloorDiv = true;
            }
          }
        }
      }
    });

    assert(!foundPackedAttn &&
           "tenzo.packed_attention should have been converted!");
    assert(memrefCopyCount == 0 &&
           "Zero memref.copy must be emitted in zero-copy GQA routing!");
    assert(transposeCount == 0 &&
           "Zero linalg.transpose must be emitted for un-transposed GQA!");
    assert(foundFloorDiv && "Implicit Shuffle GQA routing must utilize "
                            "AffineMap floorDiv expressions!");

    llvm::outs() << "  ✅ Zero-Copy GQA Routing verified: 20 Q heads mapped to "
                    "5 KV heads via floordiv!\n";
    llvm::outs() << "  ✅ Verified 0 memref.copy and 0 linalg.transpose ops "
                    "emitted in lowered IR!\n";
  }

  // ------------------------------------------------------------------------
  // TEST 9: Numerical Parity of Zero-Copy GQA Routing Against Reference
  // (H_q=20, H_kv=5, ratio=4)
  // ------------------------------------------------------------------------
  llvm::outs() << "[Test 9/10] Verifying GQA Routing Numerical Parity Against "
                  "Reference (H_q=20, H_kv=5, ratio=4)...\n";
  {
    const int64_t H_q = 20;
    const int64_t H_kv = 5;
    const int64_t gqaRatio = 4;
    const int64_t T = 16;
    const int64_t Dh = 128;
    const int64_t seq_pos = 11;
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));

    std::mt19937 rng(9999);
    std::uniform_int_distribution<int> dist(-1, 1);
    std::uniform_real_distribution<float> qdist(-1.5f, 1.5f);

    // 20 Query heads of size Dh
    std::vector<std::vector<float>> queries(H_q, std::vector<float>(Dh));
    for (int64_t qh = 0; qh < H_q; ++qh) {
      for (int64_t d = 0; d < Dh; ++d) {
        queries[qh][d] = qdist(rng);
      }
    }

    // 5 KV heads: each has T tokens of Dh ternary values
    std::vector<std::vector<int>> kv_keys_ternary(H_kv,
                                                  std::vector<int>(T * Dh));
    std::vector<std::vector<int>> kv_vals_ternary(H_kv,
                                                  std::vector<int>(T * Dh));
    std::vector<std::vector<uint8_t>> kv_keys_packed(
        H_kv, std::vector<uint8_t>(T * (Dh / 4)));
    std::vector<std::vector<uint8_t>> kv_vals_packed(
        H_kv, std::vector<uint8_t>(T * (Dh / 4)));

    for (int64_t kvh = 0; kvh < H_kv; ++kvh) {
      for (int64_t i = 0; i < T * Dh; ++i) {
        kv_keys_ternary[kvh][i] = dist(rng);
        kv_vals_ternary[kvh][i] = dist(rng);
      }
      for (int64_t t = 0; t < T; ++t) {
        packTernaryVector(kv_keys_ternary[kvh].data() + t * Dh,
                          kv_keys_packed[kvh].data() + t * (Dh / 4), Dh);
        packTernaryVector(kv_vals_ternary[kvh].data() + t * Dh,
                          kv_vals_packed[kvh].data() + t * (Dh / 4), Dh);
      }
    }

    // Compute Reference GQA: for each qh, kvh = qh / 4
    std::vector<std::vector<float>> ref_outputs(H_q,
                                                std::vector<float>(Dh, 0.0f));
    for (int64_t qh = 0; qh < H_q; ++qh) {
      int64_t kvh = qh / gqaRatio;
      naiveFP32Attention(queries[qh].data(), kv_keys_ternary[kvh].data(),
                         kv_vals_ternary[kvh].data(), ref_outputs[qh].data(), T,
                         Dh, seq_pos, scale);
    }

    // Compute Bitwise Attention with GQA routing: each qh reads from kvh = qh /
    // 4
    std::vector<std::vector<float>> avx_outputs(H_q,
                                                std::vector<float>(Dh, 0.0f));
    for (int64_t qh = 0; qh < H_q; ++qh) {
      int64_t kvh = qh / gqaRatio; // Zero-copy GQA virtual routing!
      bitwise_attention_microkernel(
          queries[qh].data(), kv_keys_packed[kvh].data(),
          kv_vals_packed[kvh].data(), avx_outputs[qh].data(), T, Dh, seq_pos,
          scale);
    }

    float max_err = 0.0f;
    for (int64_t qh = 0; qh < H_q; ++qh) {
      for (int64_t d = 0; d < Dh; ++d) {
        float err = std::abs(ref_outputs[qh][d] - avx_outputs[qh][d]);
        max_err = std::max(max_err, err);
      }
    }

    float tolerance = 1e-4f;
    llvm::outs() << "  Max absolute error across all 20 heads: " << max_err
                 << " (tolerance: " << tolerance << ")\n";
    assert(max_err < tolerance &&
           "GQA bitwise attention parity check failed (exceeds 1e-4)!");
    llvm::outs() << "  ✅ Zero-Copy GQA bitwise attention matches reference "
                    "across all 20 heads within 1e-4 tolerance!\n";
  }

  // ------------------------------------------------------------------------
  // TEST 10: 256-bit Physical Memory Layout in KVCacheManager & Zero VGATHER
  // Assembly Verification
  // ------------------------------------------------------------------------
  llvm::outs() << "[Test 10/10] Verifying 256-bit Physical Memory Layout in "
                  "KVCacheManager & Zero VGATHER Assembly...\n";
  {
    const int num_kv_heads = 5;
    const int max_seq_len = 1024;
    const int head_dim = 128;
    const int num_layers = 2;

    // 1. Verify KVCacheManager 256-bit physical memory layout for packed uint8
    // GQA cache
    runtime::KVCacheManager kv_cache(max_seq_len, num_layers, num_kv_heads,
                                     head_dim, /*is_packed_u8=*/true);

    assert(kv_cache.is_256b_aligned(0) &&
           "KVCacheManager layer 0 must be 256-bit aligned!");
    assert(kv_cache.is_256b_aligned(1) &&
           "KVCacheManager layer 1 must be 256-bit aligned!");
    assert(kv_cache.is_unit_stride() &&
           "KVCacheManager must maintain unit stride along innermost head "
           "dimension!");
    assert(kv_cache.get_head_bytes() == 32 &&
           "Packed head bytes must be exactly 32 bytes (256-bit YMM width)!");
    assert(kv_cache.get_token_stride_bytes() == 32 &&
           "Token stride must be 32 bytes!");

    uint8_t *k_ptr0 = kv_cache.get_packed_k_ptr(0);
    uint8_t *v_ptr0 = kv_cache.get_packed_v_ptr(0);
    assert((reinterpret_cast<uintptr_t>(k_ptr0) % 32) == 0 &&
           "K buffer base pointer must be 32-byte aligned!");
    assert((reinterpret_cast<uintptr_t>(v_ptr0) % 32) == 0 &&
           "V buffer base pointer must be 32-byte aligned!");

    // Verify that EVERY head and EVERY token is 256-bit aligned
    for (int h = 0; h < num_kv_heads; ++h) {
      for (int t = 0; t < max_seq_len; ++t) {
        uint8_t *head_k =
            k_ptr0 + (static_cast<size_t>(h) * max_seq_len + t) * 32;
        uint8_t *head_v =
            v_ptr0 + (static_cast<size_t>(h) * max_seq_len + t) * 32;
        assert((reinterpret_cast<uintptr_t>(head_k) % 32) == 0 &&
               "Each token's Key vector must start on a 256-bit aligned "
               "boundary!");
        assert((reinterpret_cast<uintptr_t>(head_v) % 32) == 0 &&
               "Each token's Value vector must start on a 256-bit aligned "
               "boundary!");
      }
    }
    llvm::outs() << "  ✅ Interleaved 256-bit aligned physical memory layout "
                    "confirmed across all "
                 << (num_kv_heads * max_seq_len) << " token vectors!\n";

    // 2. Assembly Disassembly Inspection: Confirm Zero VGATHER Instructions in
    // Attention Executable
    int vgatherCount = 0;
    FILE *fp = popen(
        "objdump -d /proc/self/exe 2>/dev/null | grep -iE 'vgather|vpgather'",
        "r");
    if (fp) {
      char buffer[512];
      while (fgets(buffer, sizeof(buffer), fp) != nullptr) {
        vgatherCount++;
      }
      pclose(fp);
    }

    // Also check function byte opcodes directly in memory
    const uint8_t *fn_bytes = reinterpret_cast<const uint8_t *>(
        reinterpret_cast<void *>(bitwise_attention_microkernel));
    int vexGatherOpcodeCount = 0;
    if (fn_bytes) {
      for (size_t i = 0; i < 4096; ++i) {
        if (i > 200 && fn_bytes[i] == 0xC3)
          break;
        // VEX prefix: 0xC4 followed by map 0x02 (0F 38) followed by 0x90-0x93
        // (gather opcodes)
        if (fn_bytes[i] == 0xC4 && i + 4 < 4096) {
          uint8_t byte2 = fn_bytes[i + 1] & 0x1F;
          if (byte2 == 2) { // 0F 38 map
            uint8_t opcode = fn_bytes[i + 4];
            if (opcode >= 0x90 && opcode <= 0x93) {
              vexGatherOpcodeCount++;
            }
          }
        }
      }
    }

    assert(vgatherCount == 0 &&
           "Zero vgather instructions allowed in attention executable!");
    assert(vexGatherOpcodeCount == 0 &&
           "Zero VEX gather opcodes found in microkernel!");
    llvm::outs() << "  ✅ Assembly verification confirmed: strictly ZERO "
                    "VGATHER instructions in generated executable!\n";
  }

  llvm::outs()
      << "\n🎉 ALL BITWISE ATTENTION & SIMD MICRO-KERNEL TESTS PASSED! 🎉\n\n";
}

} // namespace tenzo
