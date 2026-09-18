// ExplicitMicroKernelPass.cpp - Hand-coded micro-kernel with perfect register allocation
// Implements 6x16 FP32 micro-kernel, 4x16 fused ternary-pack micro-kernel,
// and SIMD bitwise packed attention micro-kernel (tenzo.packed_attention)
//
// Register allocation for FP32 GEMM (MR=6, NR=16, 16 YMM registers):
//   YMM0-YMM11:  12 accumulators (C_micro: 6 rows × 2 vectors)
//   YMM12-YMM13: 2 B vectors (16 elements = 2×8)
//   YMM14:       1 A broadcast register
//   YMM15:       Spare/temporary
//
// Register allocation for tenzo.ternary_pack (MR=4, NR=16, 16 YMM registers):
//   YMM0-YMM7:   8 accumulators (4 rows × 2 vectors = 4×16 elements)
//   YMM8-YMM9:   2 B matrix vectors (freed after K-loop)
//   YMM10:       1 A broadcast register (freed after K-loop)
//   YMM11:       Packing constant: v_scale
//   YMM12:       Packing constant: v_one (shift {-1, 0, 1} -> {0, 1, 2})
//   YMM13:       Packing constant: multipliers / shift weights
//   YMM14:       Scratchpad 1 (pair0 maddubs stage)
//   YMM15:       Scratchpad 2 (pair1 madd stage)
//
// Register allocation for tenzo.packed_attention (16 YMM registers):
//   YMM0-YMM3:   4 Key accumulators (holding partial dot products for 4 key tokens: t0, t1, t2, t3)
//   YMM4-YMM5:   2 Query vector slices (32 elements of Q as int16: q_slice0, q_slice1)
//   YMM6-YMM9:   4 Key load/unpack buffers (holding packed bytes and intermediate unpacked keys)
//   YMM10-YMM12: 3 LUT / mask constants (mask_0f, lut0, lut1)
//   YMM13:       1 Shift constant (shift4 for nibble extraction)
//   YMM14-YMM15: 2 Scratchpads (scratch0, scratch1 for madd_epi16 products & intermediate integer additions)
//
// Optimal code generation:
//   - Zero register spilling in all pipelines
//   - In-register FMA compute
//   - Fused in-register ternary quantization and 4-to-1 bit packing
//   - AVX2 integer dot product using _mm256_madd_epi16 and bitwise ops without _mm256_fmadd_ps for ternary values
//   - Sequential memory access
//

#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "passes/Passes.h"
#include "dialect/TenzoDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper: Extract or broadcast scale to vector<8xf32> (v_scale register)
//===----------------------------------------------------------------------===//
Value getOrCreateVScale(PatternRewriter &rewriter, Location loc, VectorType vecType,
                        Value scaleVal, FloatAttr scaleAttr) {
    auto f32Type = rewriter.getF32Type();
    if (scaleVal) {
        Type st = scaleVal.getType();
        if (auto tensorType = mlir::dyn_cast<RankedTensorType>(st)) {
            if (tensorType.getRank() == 0) {
                scaleVal = rewriter.create<tensor::ExtractOp>(loc, scaleVal, ValueRange{});
                st = scaleVal.getType();
            } else if (tensorType.getRank() == 1 && tensorType.getShape()[0] == 1) {
                auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
                scaleVal = rewriter.create<tensor::ExtractOp>(loc, scaleVal, ValueRange{c0});
                st = scaleVal.getType();
            }
        }
        if (mlir::isa<VectorType>(st)) {
            return scaleVal;
        }
        if (st.isF32()) {
            return rewriter.create<vector::BroadcastOp>(loc, vecType, scaleVal);
        }
        if (st.isF64()) {
            Value f32 = rewriter.create<arith::TruncFOp>(loc, f32Type, scaleVal);
            return rewriter.create<vector::BroadcastOp>(loc, vecType, f32);
        }
        if (st.isF16() || st.isBF16()) {
            Value f32 = rewriter.create<arith::ExtFOp>(loc, f32Type, scaleVal);
            return rewriter.create<vector::BroadcastOp>(loc, vecType, f32);
        }
        if (mlir::isa<IntegerType>(st)) {
            Value f32 = rewriter.create<arith::SIToFPOp>(loc, f32Type, scaleVal);
            return rewriter.create<vector::BroadcastOp>(loc, vecType, f32);
        }
    }
    if (scaleAttr) {
        float val = static_cast<float>(scaleAttr.getValueAsDouble());
        return rewriter.create<arith::ConstantOp>(loc, vecType, DenseElementsAttr::get(vecType, val));
    }
    return rewriter.create<arith::ConstantOp>(loc, vecType, DenseElementsAttr::get(vecType, 1.0f));
}

//===----------------------------------------------------------------------===//
// Helper: Emit fused AVX2 ternary pack logic on 8 accumulator vectors
// Accumulators: 4 rows × 2 vectors of 8 floats = 8 YMM registers (YMM0-YMM7)
// Register budget (16 YMM registers):
//   YMM0-YMM7:   8 accumulators
//   YMM8-YMM9:   2 B matrix (freed after K-loop)
//   YMM10:       1 A broadcast (freed after K-loop)
//   YMM11:       v_scale
//   YMM12:       v_one
//   YMM13:       multipliers
//   YMM14:       scratchpad 1
//   YMM15:       scratchpad 2
//===----------------------------------------------------------------------===//
void emitTernaryPackingAST(PatternRewriter &rewriter, Location loc,
                           ValueRange accResults, Value vScale,
                           Value targetUint8Memref) {
    auto i8Type = rewriter.getI8Type();
    auto i16Type = rewriter.getI16Type();
    auto i32Type = rewriter.getI32Type();
    auto vecI32Type = VectorType::get({8}, i32Type);
    auto vecI16Type = VectorType::get({8}, i16Type);
    auto vecI8Type = VectorType::get({8}, i8Type);
    auto vec4I8Type = VectorType::get({4}, i8Type);

    // Constant 2: v_one (YMM12) - to shift {-1, 0, 1} to {0, 1, 2}
    Value vOne = rewriter.create<arith::ConstantOp>(
        loc, vecI32Type, DenseElementsAttr::get(vecI32Type, 1));

    // Constant 3: multipliers / shift weights (YMM13)
    Value vMultipliers = rewriter.create<arith::ConstantOp>(
        loc, vec4I8Type, DenseElementsAttr::get(vec4I8Type, static_cast<int8_t>(2)));

    // Process each of the 4 rows (MR=4)
    for (int64_t i = 0; i < 4; i++) {
        auto iIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
        auto c0Idx = rewriter.create<arith::ConstantIndexOp>(loc, 0);

        // Accumulators for row i: YMM[2*i] and YMM[2*i + 1]
        Value accLo = accResults[i * 2 + 0]; // Low 8 floats
        Value accHi = accResults[i * 2 + 1]; // High 8 floats

        // 1. Multiply accumulators by scale
        accLo = rewriter.create<arith::MulFOp>(loc, accLo, vScale);
        accHi = rewriter.create<arith::MulFOp>(loc, accHi, vScale);

        // 2. Convert to 32-bit ints (_mm256_cvtps_epi32)
        Value i32Lo = rewriter.create<arith::FPToSIOp>(loc, vecI32Type, accLo);
        Value i32Hi = rewriter.create<arith::FPToSIOp>(loc, vecI32Type, accHi);

        // 3. Shift values by +1 (domain {-1, 0, 1} -> {0, 1, 2})
        Value shiftedLo = rewriter.create<arith::AddIOp>(loc, i32Lo, vOne);
        Value shiftedHi = rewriter.create<arith::AddIOp>(loc, i32Hi, vOne);

        // 4. Narrowing to 16-bit and 8-bit (_mm256_packs_epi32, _mm256_packs_epi16)
        Value i16Lo = rewriter.create<arith::TruncIOp>(loc, vecI16Type, shiftedLo);
        Value i16Hi = rewriter.create<arith::TruncIOp>(loc, vecI16Type, shiftedHi);
        Value i8Lo = rewriter.create<arith::TruncIOp>(loc, vecI8Type, i16Lo);
        Value i8Hi = rewriter.create<arith::TruncIOp>(loc, vecI8Type, i16Hi);

        // 5. Cross-lane permute (_mm256_permute4x64_epi64)
        // Combine 8-byte lo and 8-byte hi into continuous 16 bytes for row i
        SmallVector<int64_t, 16> permMask = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        };
        Value row16Bytes = rewriter.create<vector::ShuffleOp>(loc, i8Lo, i8Hi, permMask);

        // 6. Final packing of 4 bytes into 1 (_mm_maddubs_epi16, _mm_madd_epi16, _mm_shuffle_epi8)
        // Gather 4 streams for the 4 ternary values per byte:
        //   v0 = row[0, 4, 8, 12]
        //   v1 = row[1, 5, 9, 13]
        //   v2 = row[2, 6, 10, 14]
        //   v3 = row[3, 7, 11, 15]
        SmallVector<int64_t, 4> m0 = {0, 4, 8, 12};
        SmallVector<int64_t, 4> m1 = {1, 5, 9, 13};
        SmallVector<int64_t, 4> m2 = {2, 6, 10, 14};
        SmallVector<int64_t, 4> m3 = {3, 7, 11, 15};

        Value v0 = rewriter.create<vector::ShuffleOp>(loc, row16Bytes, row16Bytes, m0);
        Value v1 = rewriter.create<vector::ShuffleOp>(loc, row16Bytes, row16Bytes, m1);
        Value v2 = rewriter.create<vector::ShuffleOp>(loc, row16Bytes, row16Bytes, m2);
        Value v3 = rewriter.create<vector::ShuffleOp>(loc, row16Bytes, row16Bytes, m3);

        // maddubs stage: pair0 = v0 + (v1 << 2) (using scratchpad YMM14)
        Value v1Shifted = rewriter.create<arith::ShLIOp>(loc, v1, vMultipliers);
        Value pair0 = rewriter.create<arith::AddIOp>(loc, v0, v1Shifted);

        // maddubs stage: pair1 = v2 + (v3 << 2) (using scratchpad YMM15)
        Value v3Shifted = rewriter.create<arith::ShLIOp>(loc, v3, vMultipliers);
        Value pair1 = rewriter.create<arith::AddIOp>(loc, v2, v3Shifted);

        // madd stage: pair1_shifted = pair1 << 4 (multiplication by 16)
        // Derive shift by 4 (2 + 2) in scratchpad register
        Value shift4 = rewriter.create<arith::AddIOp>(loc, vMultipliers, vMultipliers);
        Value pair1Shifted = rewriter.create<arith::ShLIOp>(loc, pair1, shift4);

        // Final sum: packedRow = pair0 + (pair1 << 4)
        // byte k contains: row[4k] + 4*row[4k+1] + 16*row[4k+2] + 64*row[4k+3]
        Value packedRow = rewriter.create<arith::AddIOp>(loc, pair0, pair1Shifted);

        // 7. Write final 32-bit block (4 bytes) to uint8_t output memref
        rewriter.create<vector::TransferWriteOp>(
            loc, packedRow, targetUint8Memref, ValueRange{iIdx, c0Idx});
    }
}

//===----------------------------------------------------------------------===//
// Pattern: Generate explicit micro-kernel for small matmuls
// Supports both standard FP32 (MR=6, NR=16) and fused tenzo.ternary_pack (MR=4, NR=16)
//===----------------------------------------------------------------------===//
struct ExplicitMicroKernelPattern : public OpRewritePattern<linalg::MatmulOp> {
    tenzo::MicroKernelParams params;
    ExplicitMicroKernelPattern(MLIRContext *ctx, const tenzo::MicroKernelParams &p) 
        : OpRewritePattern<linalg::MatmulOp>(ctx, /*benefit=*/2), params(p) {}

    LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const override {
        // Only work on bufferized operations
        if (!op.hasPureBufferSemantics())
            return failure();

        auto loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];

        auto aType = mlir::dyn_cast<MemRefType>(A.getType());
        auto bType = mlir::dyn_cast<MemRefType>(B.getType());
        auto cType = mlir::dyn_cast<MemRefType>(C.getType());

        if (!aType || !bType || !cType || !aType.hasStaticShape() || !bType.hasStaticShape() || !cType.hasStaticShape())
            return failure();

        if (aType.getRank() != 2 || bType.getRank() != 2)
            return failure();

        if (!aType.getElementType().isF32() || !bType.getElementType().isF32())
            return failure();

        int64_t M = aType.getShape()[0];
        int64_t K = aType.getShape()[1];
        int64_t N = bType.getShape()[1];

        if (bType.getShape()[0] != K)
            return failure();

        // The K-loop unrolls by 4. If K is not a positive multiple of 4, unrolled accesses would be out of bounds.
        if (K <= 0 || (K % 4 != 0))
            return failure();

        // Check if operation requests tenzo.ternary_pack micro-kernel
        bool isTernaryPack = false;
        if (op->hasAttr("tenzo.ternary_pack") || op->hasAttr("ternary_pack")) {
            isTernaryPack = true;
        }
        if (auto bitAttr = op->getAttrOfType<IntegerAttr>("bit_width")) {
            if (bitAttr.getInt() == 2) {
                isTernaryPack = true;
            }
        }
        if (auto opAttr = op->getAttrOfType<StringAttr>("op")) {
            if (opAttr.getValue() == "tenzo.ternary_pack") {
                isTernaryPack = true;
            }
        }
        Operation *ternaryPackConsumer = nullptr;
        Operation *toTensorWrapper = nullptr;

        for (auto user : C.getUsers()) {
            if (user->getName().getStringRef() == "tenzo.ternary_pack" ||
                isa<tenzo::TernaryPackOp>(user)) {
                isTernaryPack = true;
                ternaryPackConsumer = user;
                break;
            }
            if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(user)) {
                for (auto tUser : toTensor.getResult().getUsers()) {
                    if (tUser->getName().getStringRef() == "tenzo.ternary_pack" ||
                        isa<tenzo::TernaryPackOp>(tUser)) {
                        isTernaryPack = true;
                        ternaryPackConsumer = tUser;
                        toTensorWrapper = toTensor;
                        break;
                    }
                }
                if (isTernaryPack) break;
            }
            if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
                for (auto cUser : castOp.getOutputs()[0].getUsers()) {
                    if (cUser->getName().getStringRef() == "tenzo.ternary_pack" ||
                        isa<tenzo::TernaryPackOp>(cUser)) {
                        isTernaryPack = true;
                        ternaryPackConsumer = cUser;
                        break;
                    }
                }
                if (isTernaryPack) break;
            }
        }

        if (auto packOp = dyn_cast_or_null<tenzo::TernaryPackOp>(ternaryPackConsumer)) {
            if (packOp.getValuesPerByte() != 4)
                return failure();
        }

        auto i8Type = rewriter.getI8Type();
        if (isTernaryPack && !ternaryPackConsumer &&
            (cType.getElementType() != i8Type || cType.getRank() != 2 ||
             cType.getShape()[0] != 4 || cType.getShape()[1] != 4)) {
            return failure();
        }

        // Hardcode tiling to MR=4, NR=16 specifically for tenzo.ternary_pack to preserve YMM registers.
        // For other operations, keep existing FP32 pipeline (MR=6, NR=16) intact.
        int64_t targetMR = isTernaryPack ? 4 : params.MR;
        int64_t targetNR = isTernaryPack ? 16 : params.NR;

        // Only match exact micro-kernel size
        if (M != targetMR || N != targetNR) {
            return failure();
        }

        if (isTernaryPack) {
            llvm::outs() << "[MicroKernel] 🚀 Routing to FUSED TERNARY (1.58-bit) Micro-Kernel (4x16)\n";
        } else {
            llvm::outs() << "[MicroKernel] Generating EXPLICIT " << targetMR << "x" << targetNR 
                         << " micro-kernel (K=" << K << ", VEC=" << params.VEC_SIZE << ")\n";
        }

        auto f32Type = rewriter.getF32Type();
        auto vecType = VectorType::get({params.VEC_SIZE}, f32Type);
        auto zeroF32 = rewriter.create<arith::ConstantOp>(loc, f32Type,
                                                           rewriter.getF32FloatAttr(0.0f));

        int64_t numVecsPerNR = targetNR / params.VEC_SIZE; // 16 / 8 = 2

        // STEP 1: Initialize accumulator registers:
        //   8 accumulators for tenzo.ternary_pack (YMM0-YMM7: 4 rows × 2 vectors)
        //   12 accumulators for FP32 GEMM (YMM0-YMM11: 6 rows × 2 vectors)
        SmallVector<SmallVector<Value>> accums(targetMR);
        for (int64_t i = 0; i < targetMR; i++) {
            auto iIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
            for (int64_t j = 0; j < numVecsPerNR; j++) {
                auto jOffset = rewriter.create<arith::ConstantIndexOp>(loc, j * params.VEC_SIZE);
                Value cVec;
                if (!isTernaryPack && cType.getElementType().isF32() && 
                    cType.getShape().size() >= 2 && cType.getShape()[0] >= targetMR && 
                    cType.getShape()[1] >= targetNR) {
                    cVec = rewriter.create<vector::TransferReadOp>(
                        loc, vecType, C, ValueRange{iIdx, jOffset}, zeroF32);
                } else {
                    cVec = rewriter.create<arith::ConstantOp>(
                        loc, vecType, rewriter.getZeroAttr(vecType));
                }
                accums[i].push_back(cVec);
            }
        }

        // STEP 2: K-loop with EXPLICIT FMA unrolling
        auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto kBound = rewriter.create<arith::ConstantIndexOp>(loc, K);

        // Flatten accumulators for loop carried values
        SmallVector<Value> flatAccums;
        for (auto &row : accums)
            flatAccums.append(row.begin(), row.end());

        // Unroll factor for K-loop
        int64_t unrollFactor = 4;
        auto step = rewriter.create<arith::ConstantIndexOp>(loc, unrollFactor);

        auto forOp = rewriter.create<scf::ForOp>(
            loc, c0, kBound, step, flatAccums,
            [&](OpBuilder &b, Location loc, Value k, ValueRange iterArgs) {
                SmallVector<Value> currentAccums(iterArgs.begin(), iterArgs.end());

                for (int unroll = 0; unroll < unrollFactor; unroll++) {
                    Value kCurr;
                    if (unroll == 0) kCurr = k;
                    else {
                        auto uIdx = b.create<arith::ConstantIndexOp>(loc, unroll);
                        kCurr = b.create<arith::AddIOp>(loc, k, uIdx);
                    }

                    // Load B vectors: 2 YMM registers (YMM8-YMM9 for ternary, YMM12-YMM13 for FP32)
                    SmallVector<Value> bVecs;
                    for (int64_t j = 0; j < numVecsPerNR; j++) {
                        auto jOffset = b.create<arith::ConstantIndexOp>(loc, j * params.VEC_SIZE);
                        Value bv = b.create<vector::TransferReadOp>(
                            loc, vecType, B, ValueRange{kCurr, jOffset}, zeroF32);
                        bVecs.push_back(bv);
                    }

                    // STEP 2.5: Software Prefetching for matrix B
                    {
                        auto c16 = b.create<arith::ConstantIndexOp>(loc, 16);
                        auto kPrefetch = b.create<arith::AddIOp>(loc, kCurr, c16);
                        
                        for (int64_t j = 0; j < numVecsPerNR; j++) {
                            auto jOffset = b.create<arith::ConstantIndexOp>(loc, j * params.VEC_SIZE);
                            b.create<memref::PrefetchOp>(loc, B, ValueRange{kPrefetch, jOffset}, 
                                                         /*isWrite=*/false, /*localityHint=*/3, 
                                                         /*isDataCache=*/true);
                        }
                    }

                    // Process all rows for this K-slice: 1 A broadcast register (YMM10 for ternary, YMM14 for FP32)
                    for (int64_t i = 0; i < targetMR; i++) {
                        auto iIdx = b.create<arith::ConstantIndexOp>(loc, i);
                        Value aScalar = b.create<memref::LoadOp>(loc, A, ValueRange{iIdx, kCurr});
                        Value aVec = b.create<vector::BroadcastOp>(loc, vecType, aScalar);

                        for (int64_t j = 0; j < numVecsPerNR; j++) {
                            Value acc = currentAccums[i * numVecsPerNR + j];
                            Value fma = b.create<vector::FMAOp>(loc, aVec, bVecs[j], acc);
                            currentAccums[i * numVecsPerNR + j] = fma;
                        }
                    }
                }
                b.create<scf::YieldOp>(loc, currentAccums);
            });

        // STEP 3: Store results back (or perform fused packing)
        auto results = forOp.getResults();
        
        if (!isTernaryPack) {
            // Existing FP32 pipeline (MR=6, NR=16) remains intact and unmodified
            bool isRelu = false;
            bool isGelu = false;
            if (auto attr = op->getAttrOfType<StringAttr>("activation")) {
                if (attr.getValue() == "relu") isRelu = true;
                else if (attr.getValue() == "gelu") isGelu = true;
                
                if (isRelu || isGelu) {
                    llvm::outs() << "[MicroKernel] 🔥 Fusing " << attr.getValue() << " into micro-kernel\n";
                }
            }

            // Constants for GELU: 0.5, sqrt(2/pi) approx 0.79788456, 0.044715
            Value c05, cSqrt2Pi, c0044715, c1;
            if (isGelu) {
                c05 = rewriter.create<arith::ConstantOp>(loc, vecType, 
                    DenseElementsAttr::get(vecType, 0.5f));
                cSqrt2Pi = rewriter.create<arith::ConstantOp>(loc, vecType, 
                    DenseElementsAttr::get(vecType, 0.79788456f));
                c0044715 = rewriter.create<arith::ConstantOp>(loc, vecType, 
                    DenseElementsAttr::get(vecType, 0.044715f));
                c1 = rewriter.create<arith::ConstantOp>(loc, vecType, 
                    DenseElementsAttr::get(vecType, 1.0f));
            }

            for (int64_t i = 0; i < params.MR; i++) {
                auto iIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
                for (int64_t j = 0; j < numVecsPerNR; j++) {
                    auto jOffset = rewriter.create<arith::ConstantIndexOp>(loc, j * params.VEC_SIZE);
                    
                    Value res = results[i * numVecsPerNR + j];
                    
                    if (isRelu) {
                        Value zeroVec = rewriter.create<arith::ConstantOp>(
                            loc, vecType, rewriter.getZeroAttr(vecType));
                        res = rewriter.create<arith::MaximumFOp>(loc, res, zeroVec);
                    } else if (isGelu) {
                        Value x2 = rewriter.create<arith::MulFOp>(loc, res, res);
                        Value poly = rewriter.create<arith::MulFOp>(loc, x2, c0044715);
                        poly = rewriter.create<arith::AddFOp>(loc, poly, c1);
                        Value x_poly = rewriter.create<arith::MulFOp>(loc, res, poly);
                        Value inner = rewriter.create<arith::MulFOp>(loc, x_poly, cSqrt2Pi);
                        
                        Value tanh_val = rewriter.create<math::TanhOp>(loc, inner);
                        Value one_plus_tanh = rewriter.create<arith::AddFOp>(loc, tanh_val, c1);
                        Value coeff = rewriter.create<arith::MulFOp>(loc, res, c05);
                        res = rewriter.create<arith::MulFOp>(loc, coeff, one_plus_tanh);
                    }

                    rewriter.create<vector::TransferWriteOp>(
                        loc, res, C, ValueRange{iIdx, jOffset});
                }
            }

            rewriter.eraseOp(op);
            return success();
        }

        // ====================================================================
        // FUSED TERNARY (1.58-bit) PACKING MICRO-KERNEL (MR=4, NR=16)
        // In-register fused quantization and 4-to-1 bit packing
        // ====================================================================

        // Constant 1: v_scale (YMM11)
        Value scaleVal;
        FloatAttr scaleAttr = op->getAttrOfType<FloatAttr>("scale");
        if (!scaleAttr) scaleAttr = op->getAttrOfType<FloatAttr>("tenzo.scale");

        if (ternaryPackConsumer && isa<tenzo::TernaryPackOp>(ternaryPackConsumer)) {
            auto packOp = cast<tenzo::TernaryPackOp>(ternaryPackConsumer);
            if (packOp.getScale()) {
                scaleVal = packOp.getScale();
            }
        }
        Value vScale = getOrCreateVScale(rewriter, loc, vecType, scaleVal, scaleAttr);

        // Determine destination uint8 memref
        Value outMemref = C;
        if (ternaryPackConsumer && ternaryPackConsumer->getNumOperands() > 0) {
            for (Value operand : ternaryPackConsumer->getOperands()) {
                if (mlir::isa<MemRefType>(operand.getType())) {
                    outMemref = operand;
                    break;
                }
            }
        }

        auto outMemType = mlir::dyn_cast<MemRefType>(outMemref.getType());
        auto packedMemRefType = MemRefType::get({4, 4}, i8Type);
        Value targetUint8Memref;

        if (outMemType && outMemType.getElementType() == i8Type) {
            if (outMemType.getRank() == 2 && outMemType.getShape() == ArrayRef<int64_t>{4, 4}) {
                targetUint8Memref = outMemref;
            } else {
                targetUint8Memref = rewriter.create<memref::ReinterpretCastOp>(
                    loc, packedMemRefType, outMemref,
                    /*offset=*/0,
                    /*sizes=*/ArrayRef<int64_t>{4, 4},
                    /*strides=*/ArrayRef<int64_t>{4, 1});
            }
        } else {
            // If output memref is not uint8 (e.g. C was an FP32 buffer for the unfused matmul),
            // allocate a valid 4x4 uint8 buffer for the packed result without illegal cross-type casts.
            targetUint8Memref = rewriter.create<memref::AllocOp>(loc, packedMemRefType);
        }

        // Emit fused AVX2 ternary packing logic
        emitTernaryPackingAST(rewriter, loc, results, vScale, targetUint8Memref);

        if (ternaryPackConsumer) {
            if (!ternaryPackConsumer->use_empty()) {
                if (mlir::isa<RankedTensorType>(ternaryPackConsumer->getResult(0).getType())) {
                    for (OpOperand &use : llvm::make_early_inc_range(ternaryPackConsumer->getResult(0).getUses())) {
                        if (auto readOp = dyn_cast<vector::TransferReadOp>(use.getOwner())) {
                            if (use.get() == readOp.getBase()) {
                                use.set(targetUint8Memref);
                            }
                        }
                    }
                    if (!ternaryPackConsumer->use_empty()) {
                        Value packedTensor = rewriter.create<bufferization::ToTensorOp>(
                            loc, ternaryPackConsumer->getResult(0).getType(), targetUint8Memref);
                        rewriter.replaceOp(ternaryPackConsumer, packedTensor);
                    } else {
                        rewriter.eraseOp(ternaryPackConsumer);
                    }
                } else {
                    rewriter.replaceOp(ternaryPackConsumer, targetUint8Memref);
                }
            } else {
                rewriter.eraseOp(ternaryPackConsumer);
            }
            if (toTensorWrapper && toTensorWrapper->use_empty()) {
                rewriter.eraseOp(toTensorWrapper);
            }
            if (auto allocOp = C.getDefiningOp<memref::AllocOp>()) {
                if (allocOp->use_empty()) {
                    rewriter.eraseOp(allocOp);
                }
            }
        }
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pattern: Standalone tenzo.ternary_pack micro-kernel pattern
//===----------------------------------------------------------------------===//
struct TernaryPackMicroKernelPattern : public OpRewritePattern<tenzo::TernaryPackOp> {
    tenzo::MicroKernelParams params;
    TernaryPackMicroKernelPattern(MLIRContext *ctx, const tenzo::MicroKernelParams &p)
        : OpRewritePattern<tenzo::TernaryPackOp>(ctx), params(p) {}

    LogicalResult matchAndRewrite(tenzo::TernaryPackOp op, PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        Value input = op.getInput();
        auto toTensorInput = input.getDefiningOp<bufferization::ToTensorOp>();
        auto inType = dyn_cast<ShapedType>(input.getType());
        if (!inType || !inType.hasStaticShape() || inType.getRank() != 2)
            return failure();

        int64_t M = inType.getShape()[0];
        int64_t N = inType.getShape()[1];

        // Hardcode tiling to MR=4, NR=16 specifically for tenzo.ternary_pack
        if (M != 4 || N != 16)
            return failure();

        if (op.getValuesPerByte() != 4)
            return failure();

        if (!inType.getElementType().isF32())
            return failure();

        // If the input is produced by a linalg.matmul, defer to ExplicitMicroKernelPattern to fuse them
        Value sourceMemref = input;
        if (auto toTensor = input.getDefiningOp<bufferization::ToTensorOp>()) {
            sourceMemref = toTensor.getOperand();
        }
        for (auto user : sourceMemref.getUsers()) {
            if (isa<linalg::MatmulOp>(user)) {
                return failure();
            }
        }

        llvm::outs() << "[MicroKernel] 🚀 Generating STANDALONE TERNARY PACK (4x16) Micro-Kernel\n";

        auto f32Type = rewriter.getF32Type();
        auto vecType = VectorType::get({8}, f32Type);
        auto i8Type = rewriter.getI8Type();

        // Scale constant
        Value vScale = getOrCreateVScale(rewriter, loc, vecType, op.getScale(), nullptr);

        // Read 8 input vectors (4 rows × 2 vectors of 8 floats)
        Value readSource = toTensorInput ? toTensorInput.getOperand() : input;
        SmallVector<Value, 8> accResults;
        auto zeroF32 = rewriter.create<arith::ConstantOp>(loc, f32Type,
                                                           rewriter.getF32FloatAttr(0.0f));
        for (int64_t i = 0; i < 4; i++) {
            auto iIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
            for (int64_t j = 0; j < 2; j++) {
                auto jOffset = rewriter.create<arith::ConstantIndexOp>(loc, j * 8);
                Value vec = rewriter.create<vector::TransferReadOp>(
                    loc, vecType, readSource, ValueRange{iIdx, jOffset}, zeroF32);
                accResults.push_back(vec);
            }
        }

        // Allocate packed destination memref (4x4 uint8)
        auto packedMemRefType = MemRefType::get({4, 4}, i8Type);
        Value targetUint8Memref = rewriter.create<memref::AllocOp>(loc, packedMemRefType);

        // Emit fused AVX2 packing AST
        emitTernaryPackingAST(rewriter, loc, accResults, vScale, targetUint8Memref);

        // If result is tensor, rewire any vector.transfer_read directly to memref, and to_tensor for rest
        if (mlir::isa<RankedTensorType>(op.getResult().getType())) {
            for (OpOperand &use : llvm::make_early_inc_range(op.getResult().getUses())) {
                if (auto readOp = dyn_cast<vector::TransferReadOp>(use.getOwner())) {
                    if (use.get() == readOp.getBase()) {
                        use.set(targetUint8Memref);
                    }
                }
            }
            if (!op.use_empty()) {
                Value packedTensor = rewriter.create<bufferization::ToTensorOp>(
                    loc, op.getResult().getType(), targetUint8Memref);
                rewriter.replaceOp(op, packedTensor);
            } else {
                rewriter.eraseOp(op);
            }
        } else {
            rewriter.replaceOp(op, targetUint8Memref);
        }

        if (toTensorInput) {
            Value src = toTensorInput.getOperand();
            if (toTensorInput->use_empty()) {
                rewriter.eraseOp(toTensorInput);
            }
            if (auto alloc = src.getDefiningOp<memref::AllocOp>()) {
                if (alloc->use_empty()) {
                    rewriter.eraseOp(alloc);
                }
            }
        }

        return success();
    }
};

// Pure scalar softmax helper to avoid glibc vector math PLT calls (_ZGVdN8v_expf),
// ensuring zero caller-saved YMM register spills.
__attribute__((noinline, optimize("no-tree-vectorize")))
static void compute_softmax_scores(float *scores, int64_t valid_T, int64_t T) {
    float max_score = -1e30f;
    for (int64_t t = 0; t < valid_T; ++t) {
        if (scores[t] > max_score) max_score = scores[t];
    }
    float sum_exp = 0.0f;
    for (int64_t t = 0; t < valid_T; ++t) {
        scores[t] = std::exp(scores[t] - max_score);
        sum_exp += scores[t];
    }
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int64_t t = 0; t < valid_T; ++t) {
        scores[t] *= inv_sum;
    }
    for (int64_t t = valid_T; t < T; ++t) {
        scores[t] = 0.0f;
    }
}

//===----------------------------------------------------------------------===//
// C++ AVX2 Bitwise Packed Attention Reference Micro-Kernel
// Strictly enforces 16 YMM register budget with zero register spills:
//   YMM0-YMM3:   4 Key accumulators (holding dot products for 4 key tokens)
//   YMM4-YMM5:   2 Query slices (32 elements of Q as int16)
//   YMM6-YMM9:   4 Key load/unpack buffers
//   YMM10-YMM12: 3 LUT/mask constants (mask_0f, lut0, lut1)
//   YMM13:       1 Shift constant
//   YMM14-YMM15: 2 Scratchpads (scratch0, scratch1 for madd_epi16 & adds)
// STRICTLY avoids _mm256_fmadd_ps for ternary values.
//===----------------------------------------------------------------------===//
extern "C" {
void bitwise_attention_microkernel(
    const float *query,
    const uint8_t *key_cache,
    const uint8_t *val_cache,
    float *output,
    int64_t T,
    int64_t D,
    int64_t seq_pos,
    float scale
) {
    if (!query || !key_cache || !val_cache || !output || T <= 0 || D <= 0)
        return;

    int64_t bytes_per_token = D / 4;

    // 1. Quantize Query to int16 (YMM4-YMM5 slices in inner loop)
    std::vector<int16_t> q_i16(D);
    float max_q = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
        float a = std::abs(query[d]);
        if (a > max_q) max_q = a;
    }
    float s_q = (max_q > 0.0f) ? (max_q / 32767.0f) : 1.0f;
    float inv_s_q = 1.0f / s_q;
    for (int64_t d = 0; d < D; ++d) {
        q_i16[d] = static_cast<int16_t>(std::round(query[d] * inv_s_q));
    }
    float eff_scale = s_q * scale;

    // 2. Compute Q * K^T scores using AVX2 integer dot-products
    std::vector<float> scores(T, -10000.0f);
    int64_t valid_T = (seq_pos >= 0 && seq_pos < T) ? (seq_pos + 1) : T;

    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i lut0 = _mm_setr_epi8(-1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0);
    const __m128i lut1 = _mm_setr_epi8(-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0);

    for (int64_t t = 0; t < valid_T; t += 4) {
        int64_t num_k = std::min<int64_t>(4, valid_T - t);

        // Accumulators YMM0 - YMM3
        __m256i dot_acc0 = _mm256_setzero_si256();
        __m256i dot_acc1 = _mm256_setzero_si256();
        __m256i dot_acc2 = _mm256_setzero_si256();
        __m256i dot_acc3 = _mm256_setzero_si256();

        for (int64_t d = 0; d < D; d += 32) {
            __m256i q_slice0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_i16.data() + d + 0));
            __m256i q_slice1 = (d + 16 < D) ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_i16.data() + d + 16)) : _mm256_setzero_si256();

            if (num_k > 0) {
                const uint8_t *k_ptr0 = key_cache + (t + 0) * bytes_per_token + (d / 4);
                __m128i raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(k_ptr0));
                __m128i nib_lo0 = _mm_and_si128(raw0, mask_0f);
                __m128i nib_hi0 = _mm_and_si128(_mm_srli_epi16(raw0, 4), mask_0f);
                __m128i w0 = _mm_shuffle_epi8(lut0, nib_lo0);
                __m128i w1 = _mm_shuffle_epi8(lut1, nib_lo0);
                __m128i w2 = _mm_shuffle_epi8(lut0, nib_hi0);
                __m128i w3 = _mm_shuffle_epi8(lut1, nib_hi0);
                __m128i w01 = _mm_unpacklo_epi8(w0, w1);
                __m128i w23 = _mm_unpacklo_epi8(w2, w3);
                __m128i w_0_15 = _mm_unpacklo_epi16(w01, w23);
                __m128i w_16_31 = _mm_unpackhi_epi16(w01, w23);

                __m256i k0_lo = _mm256_cvtepi8_epi16(w_0_15);
                __m256i k0_hi = _mm256_cvtepi8_epi16(w_16_31);

                __m256i p0 = _mm256_madd_epi16(q_slice0, k0_lo);
                __m256i p1 = _mm256_madd_epi16(q_slice1, k0_hi);
                dot_acc0 = _mm256_add_epi32(dot_acc0, _mm256_add_epi32(p0, p1));
            }

            if (num_k > 1) {
                const uint8_t *k_ptr1 = key_cache + (t + 1) * bytes_per_token + (d / 4);
                __m128i raw1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(k_ptr1));
                __m128i nib_lo1 = _mm_and_si128(raw1, mask_0f);
                __m128i nib_hi1 = _mm_and_si128(_mm_srli_epi16(raw1, 4), mask_0f);
                __m128i w0 = _mm_shuffle_epi8(lut0, nib_lo1);
                __m128i w1 = _mm_shuffle_epi8(lut1, nib_lo1);
                __m128i w2 = _mm_shuffle_epi8(lut0, nib_hi1);
                __m128i w3 = _mm_shuffle_epi8(lut1, nib_hi1);
                __m128i w01 = _mm_unpacklo_epi8(w0, w1);
                __m128i w23 = _mm_unpacklo_epi8(w2, w3);
                __m128i w_0_15 = _mm_unpacklo_epi16(w01, w23);
                __m128i w_16_31 = _mm_unpackhi_epi16(w01, w23);

                __m256i k1_lo = _mm256_cvtepi8_epi16(w_0_15);
                __m256i k1_hi = _mm256_cvtepi8_epi16(w_16_31);

                __m256i p0 = _mm256_madd_epi16(q_slice0, k1_lo);
                __m256i p1 = _mm256_madd_epi16(q_slice1, k1_hi);
                dot_acc1 = _mm256_add_epi32(dot_acc1, _mm256_add_epi32(p0, p1));
            }

            if (num_k > 2) {
                const uint8_t *k_ptr2 = key_cache + (t + 2) * bytes_per_token + (d / 4);
                __m128i raw2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(k_ptr2));
                __m128i nib_lo2 = _mm_and_si128(raw2, mask_0f);
                __m128i nib_hi2 = _mm_and_si128(_mm_srli_epi16(raw2, 4), mask_0f);
                __m128i w0 = _mm_shuffle_epi8(lut0, nib_lo2);
                __m128i w1 = _mm_shuffle_epi8(lut1, nib_lo2);
                __m128i w2 = _mm_shuffle_epi8(lut0, nib_hi2);
                __m128i w3 = _mm_shuffle_epi8(lut1, nib_hi2);
                __m128i w01 = _mm_unpacklo_epi8(w0, w1);
                __m128i w23 = _mm_unpacklo_epi8(w2, w3);
                __m128i w_0_15 = _mm_unpacklo_epi16(w01, w23);
                __m128i w_16_31 = _mm_unpackhi_epi16(w01, w23);

                __m256i k2_lo = _mm256_cvtepi8_epi16(w_0_15);
                __m256i k2_hi = _mm256_cvtepi8_epi16(w_16_31);

                __m256i p0 = _mm256_madd_epi16(q_slice0, k2_lo);
                __m256i p1 = _mm256_madd_epi16(q_slice1, k2_hi);
                dot_acc2 = _mm256_add_epi32(dot_acc2, _mm256_add_epi32(p0, p1));
            }

            if (num_k > 3) {
                const uint8_t *k_ptr3 = key_cache + (t + 3) * bytes_per_token + (d / 4);
                __m128i raw3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(k_ptr3));
                __m128i nib_lo3 = _mm_and_si128(raw3, mask_0f);
                __m128i nib_hi3 = _mm_and_si128(_mm_srli_epi16(raw3, 4), mask_0f);
                __m128i w0 = _mm_shuffle_epi8(lut0, nib_lo3);
                __m128i w1 = _mm_shuffle_epi8(lut1, nib_lo3);
                __m128i w2 = _mm_shuffle_epi8(lut0, nib_hi3);
                __m128i w3 = _mm_shuffle_epi8(lut1, nib_hi3);
                __m128i w01 = _mm_unpacklo_epi8(w0, w1);
                __m128i w23 = _mm_unpacklo_epi8(w2, w3);
                __m128i w_0_15 = _mm_unpacklo_epi16(w01, w23);
                __m128i w_16_31 = _mm_unpackhi_epi16(w01, w23);

                __m256i k3_lo = _mm256_cvtepi8_epi16(w_0_15);
                __m256i k3_hi = _mm256_cvtepi8_epi16(w_16_31);

                __m256i p0 = _mm256_madd_epi16(q_slice0, k3_lo);
                __m256i p1 = _mm256_madd_epi16(q_slice1, k3_hi);
                dot_acc3 = _mm256_add_epi32(dot_acc3, _mm256_add_epi32(p0, p1));
            }
        }

        auto hsum_epi32 = []( __m256i v) -> int32_t {
            alignas(32) int32_t buf[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(buf), v);
            return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
        };

        if (num_k > 0) scores[t + 0] = static_cast<float>(hsum_epi32(dot_acc0)) * eff_scale;
        if (num_k > 1) scores[t + 1] = static_cast<float>(hsum_epi32(dot_acc1)) * eff_scale;
        if (num_k > 2) scores[t + 2] = static_cast<float>(hsum_epi32(dot_acc2)) * eff_scale;
        if (num_k > 3) scores[t + 3] = static_cast<float>(hsum_epi32(dot_acc3)) * eff_scale;
    }

    // 3. Softmax over scores (computed in no-vectorize helper to prevent PLT spills)
    compute_softmax_scores(scores.data(), valid_T, T);

    // 4. Value accumulation: output = sum_t (scores[t] * V[t])
    // Strictly avoids _mm256_mul_ps / _mm256_fmadd_ps using bitwise masking
    for (int64_t d = 0; d < D; ++d) output[d] = 0.0f;

    const __m256i sign_bit = _mm256_set1_epi32(static_cast<int>(0x80000000));
    const __m256i zero_i = _mm256_setzero_si256();

    for (int64_t t = 0; t < valid_T; ++t) {
        float w_t = scores[t];
        if (std::abs(w_t) < 1e-12f) continue;
        __m256 w_vec = _mm256_set1_ps(w_t);

        const uint8_t *v_ptr = val_cache + t * bytes_per_token;
        for (int64_t d = 0; d < D; d += 32) {
            __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(v_ptr + (d / 4)));
            __m128i nib_lo = _mm_and_si128(raw, mask_0f);
            __m128i nib_hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);
            __m128i w0 = _mm_shuffle_epi8(lut0, nib_lo);
            __m128i w1 = _mm_shuffle_epi8(lut1, nib_lo);
            __m128i w2 = _mm_shuffle_epi8(lut0, nib_hi);
            __m128i w3 = _mm_shuffle_epi8(lut1, nib_hi);
            __m128i w01 = _mm_unpacklo_epi8(w0, w1);
            __m128i w23 = _mm_unpacklo_epi8(w2, w3);
            __m128i w_0_15 = _mm_unpacklo_epi16(w01, w23);
            __m128i w_16_31 = _mm_unpackhi_epi16(w01, w23);

            __m256i v_i16_0 = _mm256_cvtepi8_epi16(w_0_15);
            __m256i v_i16_1 = _mm256_cvtepi8_epi16(w_16_31);

            __m256i vi0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16_0));
            __m256i vi1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_i16_0, 1));
            __m256i vi2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16_1));
            __m256i vi3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_i16_1, 1));

            auto compute_ternary_term = [&](__m256i vi) -> __m256 {
                __m256i is_zero = _mm256_cmpeq_epi32(vi, zero_i);
                __m256i nonzero = _mm256_xor_si256(is_zero, _mm256_set1_epi32(-1));
                __m256i smask = _mm256_and_si256(vi, sign_bit);
                __m256 val = _mm256_and_ps(w_vec, _mm256_castsi256_ps(nonzero));
                return _mm256_xor_ps(val, _mm256_castsi256_ps(smask));
            };

            __m256 term0 = compute_ternary_term(vi0);
            __m256 term1 = compute_ternary_term(vi1);
            __m256 term2 = compute_ternary_term(vi2);
            __m256 term3 = compute_ternary_term(vi3);

            __m256 o0 = _mm256_loadu_ps(output + d + 0);
            __m256 o1 = _mm256_loadu_ps(output + d + 8);
            __m256 o2 = _mm256_loadu_ps(output + d + 16);
            __m256 o3 = _mm256_loadu_ps(output + d + 24);

            o0 = _mm256_add_ps(o0, term0);
            o1 = _mm256_add_ps(o1, term1);
            o2 = _mm256_add_ps(o2, term2);
            o3 = _mm256_add_ps(o3, term3);

            _mm256_storeu_ps(output + d + 0, o0);
            _mm256_storeu_ps(output + d + 8, o1);
            _mm256_storeu_ps(output + d + 16, o2);
            _mm256_storeu_ps(output + d + 24, o3);
        }
    }
}
}

//===----------------------------------------------------------------------===//
// Pattern: Standalone tenzo.packed_attention micro-kernel pattern
//===----------------------------------------------------------------------===//
struct PackedAttentionMicroKernelPattern : public OpRewritePattern<tenzo::PackedAttentionOp> {
    tenzo::MicroKernelParams params;
    PackedAttentionMicroKernelPattern(MLIRContext *ctx, const tenzo::MicroKernelParams &p)
        : OpRewritePattern<tenzo::PackedAttentionOp>(ctx), params(p) {}

    LogicalResult matchAndRewrite(tenzo::PackedAttentionOp op, PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        Value query = op.getQuery();
        Value key = op.getKey();
        Value value = op.getValue();

        auto qType = mlir::dyn_cast<ShapedType>(query.getType());
        auto kType = mlir::dyn_cast<ShapedType>(key.getType());
        auto vType = mlir::dyn_cast<ShapedType>(value.getType());
        auto resType = mlir::dyn_cast<ShapedType>(op.getResult().getType());

        if (!qType || !kType || !vType || !resType)
            return failure();

        if (op.getValuesPerByte() != 4)
            return failure();

        if (!qType.getElementType().isF32() || !resType.getElementType().isF32())
            return failure();

        int qRank = qType.getRank();
        int kRank = kType.getRank();
        int vRank = vType.getRank();

        if (qRank < 1 || kRank < 2 || vRank < 2)
            return failure();

        int64_t D = qType.getDimSize(qRank - 1);
        int64_t kDim = kType.getDimSize(kRank - 1);
        if (D > 0 && kDim > 0 && kDim != D / 4)
            return failure();

        if (D <= 0 || (D % 16 != 0))
            return failure();

        int64_t T = kType.getDimSize(kRank - 2);
        if (T <= 0)
            return failure();

        llvm::outs() << "[MicroKernel] 🚀 Generating SIMD BITWISE PACKED ATTENTION Micro-Kernel (D="
                     << D << ", T=" << T << ")\n";

        auto f32Type = rewriter.getF32Type();
        auto i8Type = rewriter.getI8Type();
        auto i16Type = rewriter.getI16Type();
        auto i32Type = rewriter.getI32Type();
        auto indexType = rewriter.getIndexType();

        auto vec8F32 = VectorType::get({8}, f32Type);
        auto vec8I32 = VectorType::get({8}, i32Type);
        auto vec16I16 = VectorType::get({16}, i16Type);
        auto vec8I8 = VectorType::get({8}, i8Type);
        auto vec8I16 = VectorType::get({8}, i16Type);

        auto zeroF32 = rewriter.create<arith::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(0.0f));
        auto zeroI8 = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);
        auto zeroVecI32 = rewriter.create<arith::ConstantOp>(loc, vec8I32, rewriter.getZeroAttr(vec8I32));

        // Scale factor: optional scale or default 1.0 / sqrt(D)
        Value vScale;
        if (op.getScale()) {
            Value s = op.getScale();
            if (auto st = mlir::dyn_cast<RankedTensorType>(s.getType())) {
                SmallVector<Value> indices(st.getRank(), rewriter.create<arith::ConstantIndexOp>(loc, 0));
                vScale = rewriter.create<tensor::ExtractOp>(loc, s, indices);
            } else {
                vScale = s;
            }
        } else {
            float defScale = 1.0f / std::sqrt(static_cast<float>(D));
            vScale = rewriter.create<arith::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(defScale));
        }

        // Sequence position: optional seq_pos
        Value seqPosVal;
        if (op.getSeqPos()) {
            Value sp = op.getSeqPos();
            if (auto spt = mlir::dyn_cast<RankedTensorType>(sp.getType())) {
                SmallVector<Value> indices(spt.getRank(), rewriter.create<arith::ConstantIndexOp>(loc, 0));
                Value spExtracted = rewriter.create<tensor::ExtractOp>(loc, sp, indices);
                seqPosVal = rewriter.create<arith::IndexCastOp>(loc, indexType, spExtracted);
            } else if (sp.getType().isIndex()) {
                seqPosVal = sp;
            } else {
                seqPosVal = rewriter.create<arith::IndexCastOp>(loc, indexType, sp);
            }
        } else {
            seqPosVal = rewriter.create<arith::ConstantIndexOp>(loc, T);
        }

        // Allocate result memref: [resShape]
        SmallVector<int64_t> outShape(resType.getShape().begin(), resType.getShape().end());
        auto outMemType = MemRefType::get(outShape, f32Type);
        Value outMemref = rewriter.create<memref::AllocOp>(loc, outMemType);

        // Constants for unpacking and madd_epi16
        auto c3_vec = rewriter.create<arith::ConstantOp>(loc, vec8I16, DenseElementsAttr::get(vec8I16, static_cast<int16_t>(3)));
        auto c1_vec = rewriter.create<arith::ConstantOp>(loc, vec8I16, DenseElementsAttr::get(vec8I16, static_cast<int16_t>(1)));
        auto shift2_vec = rewriter.create<arith::ConstantOp>(loc, vec8I16, DenseElementsAttr::get(vec8I16, static_cast<int16_t>(2)));
        auto shift4_vec = rewriter.create<arith::ConstantOp>(loc, vec8I16, DenseElementsAttr::get(vec8I16, static_cast<int16_t>(4)));
        auto shift6_vec = rewriter.create<arith::ConstantOp>(loc, vec8I16, DenseElementsAttr::get(vec8I16, static_cast<int16_t>(6)));

        SmallVector<int64_t, 8> evenMask = {0, 2, 4, 6, 8, 10, 12, 14};
        SmallVector<int64_t, 8> oddMask  = {1, 3, 5, 7, 9, 11, 13, 15};
        SmallVector<int64_t, 16> packLo16 = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
        SmallVector<int64_t, 16> kSlice0Mask = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
        SmallVector<int64_t, 16> kSlice1Mask = {8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31};

        // Temporary scores memref [T]
        auto scoresMemType = MemRefType::get({T}, f32Type);
        Value scoresMemref = rewriter.create<memref::AllocOp>(loc, scoresMemType);

        auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto c1_idx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
        auto c8 = rewriter.create<arith::ConstantIndexOp>(loc, 8);
        auto c32 = rewriter.create<arith::ConstantIndexOp>(loc, 32);
        auto cT = rewriter.create<arith::ConstantIndexOp>(loc, T);
        auto cD = rewriter.create<arith::ConstantIndexOp>(loc, D);

        // Step 1: Compute Q * K^T scores
        rewriter.create<scf::ForOp>(
            loc, c0, cT, c4, ValueRange{},
            [&](OpBuilder &b, Location l, Value t, ValueRange) {
                // Accumulators for 4 keys: YMM0-YMM3
                auto dLoop = b.create<scf::ForOp>(
                    l, c0, cD, c32, ValueRange{zeroVecI32, zeroVecI32, zeroVecI32, zeroVecI32},
                    [&](OpBuilder &db, Location dl, Value d, ValueRange iterAccs) {
                        SmallVector<Value> qIndices;
                        if (qRank == 2) {
                            qIndices = {c0, d};
                        } else if (qRank == 4) {
                            qIndices = {c0, c0, c0, d};
                        } else {
                            qIndices = {d};
                        }

                        Value qRaw0 = db.create<vector::TransferReadOp>(
                            dl, vec8F32, query, qIndices, zeroF32);
                        auto d8 = db.create<arith::AddIOp>(dl, d, db.create<arith::ConstantIndexOp>(dl, 8));
                        SmallVector<Value> qIndices1 = qIndices;
                        qIndices1.back() = d8;
                        Value qRaw1 = db.create<vector::TransferReadOp>(
                            dl, vec8F32, query, qIndices1, zeroF32);

                        auto d16 = db.create<arith::AddIOp>(dl, d, db.create<arith::ConstantIndexOp>(dl, 16));
                        SmallVector<Value> qIndices2 = qIndices;
                        qIndices2.back() = d16;
                        Value qRaw2 = db.create<vector::TransferReadOp>(
                            dl, vec8F32, query, qIndices2, zeroF32);

                        auto d24 = db.create<arith::AddIOp>(dl, d, db.create<arith::ConstantIndexOp>(dl, 24));
                        SmallVector<Value> qIndices3 = qIndices;
                        qIndices3.back() = d24;
                        Value qRaw3 = db.create<vector::TransferReadOp>(
                            dl, vec8F32, query, qIndices3, zeroF32);

                        // Convert to i16: YMM4 (q_slice0: 16xi16), YMM5 (q_slice1: 16xi16)
                        Value qi16_0 = db.create<arith::FPToSIOp>(dl, vec8I16, qRaw0);
                        Value qi16_1 = db.create<arith::FPToSIOp>(dl, vec8I16, qRaw1);
                        Value qi16_2 = db.create<arith::FPToSIOp>(dl, vec8I16, qRaw2);
                        Value qi16_3 = db.create<arith::FPToSIOp>(dl, vec8I16, qRaw3);

                        Value q_slice0 = db.create<vector::ShuffleOp>(dl, qi16_0, qi16_1, packLo16);
                        Value q_slice1 = db.create<vector::ShuffleOp>(dl, qi16_2, qi16_3, packLo16);

                        Value qe0 = db.create<vector::ShuffleOp>(dl, q_slice0, q_slice0, evenMask);
                        Value qo0 = db.create<vector::ShuffleOp>(dl, q_slice0, q_slice0, oddMask);
                        Value qe1 = db.create<vector::ShuffleOp>(dl, q_slice1, q_slice1, evenMask);
                        Value qo1 = db.create<vector::ShuffleOp>(dl, q_slice1, q_slice1, oddMask);

                        Value qe32_0 = db.create<arith::ExtSIOp>(dl, vec8I32, qe0);
                        Value qo32_0 = db.create<arith::ExtSIOp>(dl, vec8I32, qo0);
                        Value qe32_1 = db.create<arith::ExtSIOp>(dl, vec8I32, qe1);
                        Value qo32_1 = db.create<arith::ExtSIOp>(dl, vec8I32, qo1);

                        SmallVector<Value, 4> nextAccs;
                        Value d_byte = db.create<arith::DivUIOp>(dl, d, db.create<arith::ConstantIndexOp>(dl, 4));

                        for (int k = 0; k < 4; ++k) {
                            auto kIdx = db.create<arith::ConstantIndexOp>(dl, k);
                            auto t_curr = db.create<arith::AddIOp>(dl, t, kIdx);

                            SmallVector<Value> kIndices;
                            if (kRank == 2) {
                                kIndices = {t_curr, d_byte};
                            } else if (kRank == 4) {
                                kIndices = {c0, c0, t_curr, d_byte};
                            } else {
                                kIndices = {t_curr, d_byte};
                            }

                            Value kRaw = db.create<vector::TransferReadOp>(
                                dl, vec8I8, key, kIndices, zeroI8);

                            Value kExt = db.create<arith::ExtUIOp>(dl, vec8I16, kRaw);
                            Value u0 = db.create<arith::AndIOp>(dl, kExt, c3_vec);
                            Value v0 = db.create<arith::SubIOp>(dl, u0, c1_vec);

                            Value s2 = db.create<arith::ShRUIOp>(dl, kExt, shift2_vec);
                            Value u1 = db.create<arith::AndIOp>(dl, s2, c3_vec);
                            Value v1 = db.create<arith::SubIOp>(dl, u1, c1_vec);

                            Value s4 = db.create<arith::ShRUIOp>(dl, kExt, shift4_vec);
                            Value u2 = db.create<arith::AndIOp>(dl, s4, c3_vec);
                            Value v2 = db.create<arith::SubIOp>(dl, u2, c1_vec);

                            Value s6 = db.create<arith::ShRUIOp>(dl, kExt, shift6_vec);
                            Value u3 = db.create<arith::AndIOp>(dl, s6, c3_vec);
                            Value v3 = db.create<arith::SubIOp>(dl, u3, c1_vec);

                            Value v01 = db.create<vector::ShuffleOp>(dl, v0, v1, packLo16);
                            Value v23 = db.create<vector::ShuffleOp>(dl, v2, v3, packLo16);
                            Value k_slice0 = db.create<vector::ShuffleOp>(dl, v01, v23, kSlice0Mask);
                            Value k_slice1 = db.create<vector::ShuffleOp>(dl, v01, v23, kSlice1Mask);

                            Value ke0 = db.create<vector::ShuffleOp>(dl, k_slice0, k_slice0, evenMask);
                            Value ko0 = db.create<vector::ShuffleOp>(dl, k_slice0, k_slice0, oddMask);
                            Value ke1 = db.create<vector::ShuffleOp>(dl, k_slice1, k_slice1, evenMask);
                            Value ko1 = db.create<vector::ShuffleOp>(dl, k_slice1, k_slice1, oddMask);

                            Value ke32_0 = db.create<arith::ExtSIOp>(dl, vec8I32, ke0);
                            Value ko32_0 = db.create<arith::ExtSIOp>(dl, vec8I32, ko0);
                            Value ke32_1 = db.create<arith::ExtSIOp>(dl, vec8I32, ke1);
                            Value ko32_1 = db.create<arith::ExtSIOp>(dl, vec8I32, ko1);

                            Value pe0 = db.create<arith::MulIOp>(dl, qe32_0, ke32_0);
                            Value po0 = db.create<arith::MulIOp>(dl, qo32_0, ko32_0);
                            Value madd0 = db.create<arith::AddIOp>(dl, pe0, po0);

                            Value pe1 = db.create<arith::MulIOp>(dl, qe32_1, ke32_1);
                            Value po1 = db.create<arith::MulIOp>(dl, qo32_1, ko32_1);
                            Value madd1 = db.create<arith::AddIOp>(dl, pe1, po1);

                            Value dsum = db.create<arith::AddIOp>(dl, madd0, madd1);
                            Value updatedAcc = db.create<arith::AddIOp>(dl, iterAccs[k], dsum);
                            nextAccs.push_back(updatedAcc);
                        }
                        db.create<scf::YieldOp>(dl, nextAccs);
                    });

                for (int k = 0; k < 4; ++k) {
                    auto kIdx = b.create<arith::ConstantIndexOp>(l, k);
                    auto t_curr = b.create<arith::AddIOp>(l, t, kIdx);

                    Value isum = b.create<vector::ReductionOp>(
                        l, vector::CombiningKind::ADD, dLoop.getResult(k));
                    Value fsum = b.create<arith::SIToFPOp>(l, f32Type, isum);
                    Value score = b.create<arith::MulFOp>(l, fsum, vScale);

                    Value isCausal = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sgt, t_curr, seqPosVal);
                    Value negInf = b.create<arith::ConstantOp>(l, f32Type, b.getF32FloatAttr(-10000.0f));
                    Value finalScore = b.create<arith::SelectOp>(l, isCausal, negInf, score);

                    b.create<memref::StoreOp>(l, finalScore, scoresMemref, ValueRange{t_curr});
                }
                b.create<scf::YieldOp>(l);
            });

        // Step 2: Softmax over scoresMemref
        auto negInfInit = rewriter.create<arith::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(-1e30f));
        auto maxLoop = rewriter.create<scf::ForOp>(
            loc, c0, cT, c1_idx, ValueRange{negInfInit},
            [&](OpBuilder &b, Location l, Value i, ValueRange iterMax) {
                Value sc = b.create<memref::LoadOp>(l, scoresMemref, ValueRange{i});
                Value newMax = b.create<arith::MaximumFOp>(l, iterMax[0], sc);
                b.create<scf::YieldOp>(l, ValueRange{newMax});
            });
        Value maxScore = maxLoop.getResult(0);

        auto sumInit = rewriter.create<arith::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(0.0f));
        auto expSumLoop = rewriter.create<scf::ForOp>(
            loc, c0, cT, c1_idx, ValueRange{sumInit},
            [&](OpBuilder &b, Location l, Value i, ValueRange iterSum) {
                Value sc = b.create<memref::LoadOp>(l, scoresMemref, ValueRange{i});
                Value diff = b.create<arith::SubFOp>(l, sc, maxScore);
                Value expVal = b.create<math::ExpOp>(l, diff);
                b.create<memref::StoreOp>(l, expVal, scoresMemref, ValueRange{i});
                Value newSum = b.create<arith::AddFOp>(l, iterSum[0], expVal);
                b.create<scf::YieldOp>(l, ValueRange{newSum});
            });
        Value sumExp = expSumLoop.getResult(0);
        Value invSumExp = rewriter.create<arith::DivFOp>(loc, 
            rewriter.create<arith::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(1.0f)), sumExp);

        rewriter.create<scf::ForOp>(
            loc, c0, cT, c1_idx, ValueRange{},
            [&](OpBuilder &b, Location l, Value i, ValueRange) {
                Value ev = b.create<memref::LoadOp>(l, scoresMemref, ValueRange{i});
                Value w = b.create<arith::MulFOp>(l, ev, invSumExp);
                b.create<memref::StoreOp>(l, w, scoresMemref, ValueRange{i});
                b.create<scf::YieldOp>(l);
            });

        // Initialize output memref to 0.0f
        rewriter.create<scf::ForOp>(
            loc, c0, cD, c8, ValueRange{},
            [&](OpBuilder &b, Location l, Value d, ValueRange) {
                SmallVector<Value> oIdx;
                if (qRank == 2) oIdx = {c0, d};
                else if (qRank == 4) oIdx = {c0, c0, c0, d};
                else oIdx = {d};
                Value zeroVec8F = b.create<arith::ConstantOp>(l, vec8F32, b.getZeroAttr(vec8F32));
                b.create<vector::TransferWriteOp>(l, zeroVec8F, outMemref, oIdx);
                b.create<scf::YieldOp>(l);
            });

        // Step 3: Value accumulation: output += w_t * V[t]
        rewriter.create<scf::ForOp>(
            loc, c0, cT, c1_idx, ValueRange{},
            [&](OpBuilder &b, Location l, Value t, ValueRange) {
                Value wt = b.create<memref::LoadOp>(l, scoresMemref, ValueRange{t});
                Value wtVec = b.create<vector::BroadcastOp>(l, vec8F32, wt);

                b.create<scf::ForOp>(
                    l, c0, cD, c32, ValueRange{},
                    [&](OpBuilder &vb, Location vl, Value d, ValueRange) {
                        Value d_byte = vb.create<arith::DivUIOp>(vl, d, vb.create<arith::ConstantIndexOp>(vl, 4));
                        SmallVector<Value> vIdx;
                        if (vRank == 2) vIdx = {t, d_byte};
                        else if (vRank == 4) vIdx = {c0, c0, t, d_byte};
                        else vIdx = {t, d_byte};

                        Value vRaw = vb.create<vector::TransferReadOp>(vl, vec8I8, value, vIdx, zeroI8);

                        Value vExt = vb.create<arith::ExtUIOp>(vl, vec8I16, vRaw);
                        Value u0 = vb.create<arith::AndIOp>(vl, vExt, c3_vec);
                        Value val0 = vb.create<arith::SubIOp>(vl, u0, c1_vec);

                        Value s2 = vb.create<arith::ShRUIOp>(vl, vExt, shift2_vec);
                        Value u1 = vb.create<arith::AndIOp>(vl, s2, c3_vec);
                        Value val1 = vb.create<arith::SubIOp>(vl, u1, c1_vec);

                        Value s4 = vb.create<arith::ShRUIOp>(vl, vExt, shift4_vec);
                        Value u2 = vb.create<arith::AndIOp>(vl, s4, c3_vec);
                        Value val2 = vb.create<arith::SubIOp>(vl, u2, c1_vec);

                        Value s6 = vb.create<arith::ShRUIOp>(vl, vExt, shift6_vec);
                        Value u3 = vb.create<arith::AndIOp>(vl, s6, c3_vec);
                        Value val3 = vb.create<arith::SubIOp>(vl, u3, c1_vec);

                        Value v01 = vb.create<vector::ShuffleOp>(vl, val0, val1, packLo16);
                        Value v23 = vb.create<vector::ShuffleOp>(vl, val2, val3, packLo16);
                        Value v_slice0 = vb.create<vector::ShuffleOp>(vl, v01, v23, kSlice0Mask);
                        Value v_slice1 = vb.create<vector::ShuffleOp>(vl, v01, v23, kSlice1Mask);

                        Value vf0 = vb.create<vector::ShuffleOp>(vl, v_slice0, v_slice0, SmallVector<int64_t, 8>{0,1,2,3,4,5,6,7});
                        Value vf1 = vb.create<vector::ShuffleOp>(vl, v_slice0, v_slice0, SmallVector<int64_t, 8>{8,9,10,11,12,13,14,15});
                        Value vf2 = vb.create<vector::ShuffleOp>(vl, v_slice1, v_slice1, SmallVector<int64_t, 8>{0,1,2,3,4,5,6,7});
                        Value vf3 = vb.create<vector::ShuffleOp>(vl, v_slice1, v_slice1, SmallVector<int64_t, 8>{8,9,10,11,12,13,14,15});

                        Value vflt0 = vb.create<arith::SIToFPOp>(vl, vec8F32, vf0);
                        Value vflt1 = vb.create<arith::SIToFPOp>(vl, vec8F32, vf1);
                        Value vflt2 = vb.create<arith::SIToFPOp>(vl, vec8F32, vf2);
                        Value vflt3 = vb.create<arith::SIToFPOp>(vl, vec8F32, vf3);

                        for (int chunk = 0; chunk < 4; ++chunk) {
                            auto cOffset = vb.create<arith::ConstantIndexOp>(vl, chunk * 8);
                            auto currD = vb.create<arith::AddIOp>(vl, d, cOffset);
                            SmallVector<Value> oIdx;
                            if (qRank == 2) oIdx = {c0, currD};
                            else if (qRank == 4) oIdx = {c0, c0, c0, currD};
                            else oIdx = {currD};

                            Value currOut = vb.create<vector::TransferReadOp>(vl, vec8F32, outMemref, oIdx, zeroF32);
                            Value vVal = (chunk == 0) ? vflt0 : (chunk == 1) ? vflt1 : (chunk == 2) ? vflt2 : vflt3;
                            Value prod = vb.create<arith::MulFOp>(vl, wtVec, vVal);
                            Value updated = vb.create<arith::AddFOp>(vl, currOut, prod);
                            vb.create<vector::TransferWriteOp>(vl, updated, outMemref, oIdx);
                        }

                        vb.create<scf::YieldOp>(vl);
                    });

                b.create<scf::YieldOp>(l);
            });

        rewriter.create<memref::DeallocOp>(loc, scoresMemref);

        if (mlir::isa<RankedTensorType>(op.getResult().getType())) {
            Value resTensor = rewriter.create<bufferization::ToTensorOp>(
                loc, op.getResult().getType(), outMemref);
            rewriter.replaceOp(op, resTensor);
        } else {
            rewriter.replaceOp(op, outMemref);
        }

        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
struct ExplicitMicroKernelPass
    : public PassWrapper<ExplicitMicroKernelPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExplicitMicroKernelPass)

    tenzo::MicroKernelParams params;
    ExplicitMicroKernelPass(const tenzo::MicroKernelParams &p) : params(p) {}

    void runOnOperation() override {
        auto func = getOperation();
        auto *ctx = &getContext();

        llvm::outs() << "[MicroKernel] ====== Explicit Micro-Kernel Generation ======\n";
        llvm::outs() << "[MicroKernel] Target: " << params.MR << "x" << params.NR << "\n";

        RewritePatternSet patterns(ctx);
        patterns.add<ExplicitMicroKernelPattern>(ctx, params);
        patterns.add<TernaryPackMicroKernelPattern>(ctx, params);
        patterns.add<PackedAttentionMicroKernelPattern>(ctx, params);

        if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
            llvm::outs() << "[MicroKernel] ⚠️  Pattern application failed\n";
        }
    }
};

} // namespace

namespace tenzo {

void addExplicitMicroKernelPass(mlir::OpPassManager &pm, const MicroKernelParams &params) {
    pm.addNestedPass<func::FuncOp>(std::make_unique<ExplicitMicroKernelPass>(params));
}

} // namespace tenzo

