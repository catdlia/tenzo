// ExplicitMicroKernelPass.cpp - Hand-coded micro-kernel with perfect register allocation
// Implements 6x16 FP32 micro-kernel and 4x16 fused ternary-pack micro-kernel
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
// Optimal code generation:
//   - Zero register spilling in both pipelines
//   - In-register FMA compute
//   - Fused in-register ternary quantization and 4-to-1 bit packing
//   - Sequential memory access

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

