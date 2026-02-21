// ExplicitMicroKernelPass.cpp - Hand-coded micro-kernel with perfect register allocation
// Implements 6x16 micro-kernel with explicit FMA operations
//
// Register allocation (16 YMM registers):
//   YMM0-YMM11:  12 accumulators (C_micro: 6 rows × 2 vectors)
//   YMM12-YMM13: 2 B vectors (16 elements = 2×8)
//   YMM14:       1 A broadcast register
//   YMM15:       Spare/temporary
//
// This generates optimal code:
//   - Zero register spilling
//   - All FMAs in-register
//   - Sequential memory access (thanks to packing)

#include "passes/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Pattern: Generate explicit micro-kernel for small matmuls
//===----------------------------------------------------------------------===//
struct ExplicitMicroKernelPattern : public OpRewritePattern<linalg::MatmulOp> {
    tenzo::MicroKernelParams params;
    ExplicitMicroKernelPattern(MLIRContext *ctx, const tenzo::MicroKernelParams &p) 
        : OpRewritePattern<linalg::MatmulOp>(ctx), params(p) {}

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

        if (!aType || !bType || !cType || !aType.hasStaticShape())
            return failure();

        int64_t M = aType.getShape()[0];
        int64_t K = aType.getShape()[1];
        int64_t N = bType.getShape()[1];

        // Only match exact micro-kernel size
        if (M != params.MR || N != params.NR) {
            return failure();
        }

        llvm::outs() << "[MicroKernel] Generating EXPLICIT " << params.MR << "x" << params.NR 
                     << " micro-kernel (K=" << K << ", VEC=" << params.VEC_SIZE << ")\n";

        auto f32Type = rewriter.getF32Type();
        auto vecType = VectorType::get({params.VEC_SIZE}, f32Type);
        auto zeroF32 = rewriter.create<arith::ConstantOp>(loc, f32Type,
                                                           rewriter.getF32FloatAttr(0.0f));

        int64_t numVecsPerNR = params.NR / params.VEC_SIZE;

        // STEP 1: Initialize accumulator registers
        SmallVector<SmallVector<Value>> accums(params.MR);
        for (int64_t i = 0; i < params.MR; i++) {
            auto iIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
            for (int64_t j = 0; j < numVecsPerNR; j++) {
                auto jOffset = rewriter.create<arith::ConstantIndexOp>(loc, j * params.VEC_SIZE);
                Value cVec = rewriter.create<vector::TransferReadOp>(
                    loc, vecType, C, ValueRange{iIdx, jOffset}, zeroF32);
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

                    // Load B vectors
                    SmallVector<Value> bVecs;
                    for (int64_t j = 0; j < numVecsPerNR; j++) {
                        auto jOffset = b.create<arith::ConstantIndexOp>(loc, j * params.VEC_SIZE);
                        Value bv = b.create<vector::TransferReadOp>(
                            loc, vecType, B, ValueRange{kCurr, jOffset}, zeroF32);
                        bVecs.push_back(bv);
                    }

                    // STEP 2.5: Software Prefetching for matrix B
                    // Prefetch B elements for k + 16 (future iterations)
                    {
                        auto c16 = b.create<arith::ConstantIndexOp>(loc, 16);
                        auto kPrefetch = b.create<arith::AddIOp>(loc, kCurr, c16);
                        
                        // Prefetch each vector block of B
                        for (int64_t j = 0; j < numVecsPerNR; j++) {
                            auto jOffset = b.create<arith::ConstantIndexOp>(loc, j * params.VEC_SIZE);
                            // memref.prefetch: B[k+16, j*vec], read=false, locality=3, type=data
                            b.create<memref::PrefetchOp>(loc, B, ValueRange{kPrefetch, jOffset}, 
                                                         /*isWrite=*/false, /*localityHint=*/3, 
                                                         /*isDataCache=*/true);
                        }
                    }

                    // Process all rows for this K-slice
                    for (int64_t i = 0; i < params.MR; i++) {
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

        // STEP 3: Store results back
        auto results = forOp.getResults();
        
        // --- ACTIVATION FUSION (ReLU / GELU) ---
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
                rewriter.getFloatAttr(vecType, 0.5f));
            cSqrt2Pi = rewriter.create<arith::ConstantOp>(loc, vecType, 
                rewriter.getFloatAttr(vecType, 0.79788456f));
            c0044715 = rewriter.create<arith::ConstantOp>(loc, vecType, 
                rewriter.getFloatAttr(vecType, 0.044715f));
            c1 = rewriter.create<arith::ConstantOp>(loc, vecType, 
                rewriter.getFloatAttr(vecType, 1.0f));
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
                    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    // Optimized: inner = sqrt(2/pi) * x * (1 + 0.044715 * x^2)
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

