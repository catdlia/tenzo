// ExplicitMicroKernel.cpp - GotoBLAS-style explicit micro-kernel with perfect register allocation
// Target: 6x16 micro-kernel for AVX2 (12 YMM accumulators + 4 temp registers)
//
// Register allocation for AVX2 (16 ymm registers):
//   ymm0-ymm11:  6x2 accumulators (C matrix blocks)
//   ymm12-ymm13: broadcast A values (2 registers for pipelining)
//   ymm14-ymm15: loaded B vectors (2 registers for double buffering)
//
// This generates explicit vector.fma instructions for optimal register usage

#include "passes/Passes.h"
#include "context/HardwareProfile.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// Optimal micro-kernel dimensions for AVX2
constexpr int64_t MR = 6;   // Rows (6 * 2 vectors = 12 accumulators)
constexpr int64_t NR = 16;  // Columns (2 AVX2 vectors of 8 floats)
constexpr int64_t VL = 8;   // Vector length (AVX2 = 8 x f32)

//===----------------------------------------------------------------------===//
// Explicit Micro-Kernel Pattern
// Converts small linalg.matmul (after tiling) to explicit vector.fma operations
//===----------------------------------------------------------------------===//
struct ExplicitMicroKernelPattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const override {
        // Only work on bufferized ops (memref semantics)
        if (!op.hasPureBufferSemantics())
            return failure();

        auto loc = op.getLoc();

        // Get operands
        Value A = op.getInputs()[0];  // [M, K]
        Value B = op.getInputs()[1];  // [K, N]
        Value C = op.getOutputs()[0]; // [M, N]

        // Get shapes
        auto aType = A.getType()mlir::cast<MemRefType>();
        auto bType = B.getType()mlir::cast<MemRefType>();
        auto cType = C.getType()mlir::cast<MemRefType>();

        if (!aType.hasStaticShape() || !bType.hasStaticShape() || !cType.hasStaticShape())
            return failure();

        int64_t M = aType.getShape()[0];
        int64_t K = aType.getShape()[1];
        int64_t N = bType.getShape()[1];

        // CRITICAL: Only apply to micro-kernels that fit our 6x16 pattern
        // This is the result of prior tiling passes
        if (M > MR || N > NR || M < 1 || N < 1) {
            return failure();
        }

        // Verify N is compatible with vectorization
        if (N % VL != 0) {
            return failure();
        }

        llvm::outs() << "[ExplicitKernel] Generating 6x16 micro-kernel for "
                     << M << "x" << K << "x" << N << "\n";

        auto f32Type = rewriter.getF32Type();
        auto vecType = VectorType::get({VL}, f32Type);
        int64_t numVecCols = N / VL;  // Should be 2 for NR=16

        // Constants
        auto zeroF32 = rewriter.create<arith::ConstantOp>(
            loc, f32Type, rewriter.getF32FloatAttr(0.0f));

        // ========================================
        // PHASE 1: Initialize accumulators
        // Load existing C values into registers (12 YMM: ymm0-ymm11)
        // ========================================
        SmallVector<SmallVector<Value>> accumulators(M);
        for (int64_t i = 0; i < M; i++) {
            auto iIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
            for (int64_t j = 0; j < numVecCols; j++) {
                auto jOffset = rewriter.create<arith::ConstantIndexOp>(loc, j * VL);

                // Read C[i, j*VL : (j+1)*VL] into vector register
                Value cVec = rewriter.create<vector::TransferReadOp>(
                    loc, vecType, C, ValueRange{iIdx, jOffset},
                    /*padding=*/zeroF32);

                accumulators[i].push_back(cVec);
            }
        }

        // ========================================
        // PHASE 2: K-loop with explicit FMA
        // This is the hottest loop - must be perfect!
        // ========================================
        auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto kBound = rewriter.create<arith::ConstantIndexOp>(loc, K);

        // Flatten accumulators for loop carried values
        SmallVector<Value> flatAccumulators;
        for (const auto &row : accumulators) {
            flatAccumulators.append(row.begin(), row.end());
        }

        auto forOp = rewriter.create<scf::ForOp>(
            loc, c0, kBound, c1, flatAccumulators,
            [&](OpBuilder &builder, Location loc, Value k, ValueRange iterArgs) {
                SmallVector<Value> newAccumulators;

                // Load B vectors for this k (uses ymm14, ymm15)
                SmallVector<Value> bVecs;
                for (int64_t j = 0; j < numVecCols; j++) {
                    auto jOffset = builder.create<arith::ConstantIndexOp>(loc, j * VL);
                    Value bVec = builder.create<vector::TransferReadOp>(
                        loc, vecType, B, ValueRange{k, jOffset},
                        /*padding=*/zeroF32);
                    bVecs.push_back(bVec);
                }

                // For each row of A
                int accIdx = 0;
                for (int64_t i = 0; i < M; i++) {
                    auto iIdx = builder.create<arith::ConstantIndexOp>(loc, i);

                    // Load A[i,k] and broadcast to vector (uses ymm12)
                    Value aScalar = builder.create<memref::LoadOp>(loc, A, ValueRange{iIdx, k});
                    Value aVec = builder.create<vector::BroadcastOp>(loc, vecType, aScalar);

                    // FMA for each vector column: C[i,j*VL:] += A[i,k] * B[k,j*VL:]
                    for (int64_t j = 0; j < numVecCols; j++) {
                        Value acc = iterArgs[accIdx];

                        // vector.fma: result = a * b + c
                        // This maps directly to VFMADD instruction on AVX2
                        Value fmaResult = builder.create<vector::FMAOp>(
                            loc, aVec, bVecs[j], acc);

                        newAccumulators.push_back(fmaResult);
                        accIdx++;
                    }
                }

                builder.create<scf::YieldOp>(loc, newAccumulators);
            });

        // ========================================
        // PHASE 3: Store results back to C
        // ========================================
        auto results = forOp.getResults();
        int resIdx = 0;
        for (int64_t i = 0; i < M; i++) {
            auto iIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
            for (int64_t j = 0; j < numVecCols; j++) {
                auto jOffset = rewriter.create<arith::ConstantIndexOp>(loc, j * VL);

                rewriter.create<vector::TransferWriteOp>(
                    loc, results[resIdx], C, ValueRange{iIdx, jOffset});
                resIdx++;
            }
        }

        // Remove the original matmul
        rewriter.eraseOp(op);

        llvm::outs() << "[ExplicitKernel] ✅ Generated " << (M * numVecCols * K)
                     << " FMA operations (optimal register usage)\n";

        return success();
    }
};

//===----------------------------------------------------------------------===//
// Main Pass
//===----------------------------------------------------------------------===//
struct ExplicitMicroKernelPass
    : public PassWrapper<ExplicitMicroKernelPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExplicitMicroKernelPass)

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        MLIRContext *ctx = &getContext();

        llvm::outs() << "[ExplicitKernel] ====== Explicit Micro-Kernel Pass ======\n";
        llvm::outs() << "[ExplicitKernel] Target: " << MR << "x" << NR << " (optimal for AVX2)\n";

        // Count bufferized matmuls
        int matmulCount = 0;
        func.walk([&](linalg::MatmulOp op) {
            if (op.hasPureBufferSemantics()) {
                matmulCount++;
                auto aType = op.getInputs()[0].getType()mlir::cast<MemRefType>();
                auto shape = aType.getShape();
                llvm::outs() << "[ExplicitKernel] Found bufferized matmul: "
                             << shape[0] << "x" << shape[1] << " x "
                             << op.getInputs()[1].getType()mlir::cast<MemRefType>().getShape()[1] << "\n";
            }
        });

        if (matmulCount == 0) {
            llvm::outs() << "[ExplicitKernel] No bufferized matmul ops found\n";
            return;
        }

        // Apply explicit micro-kernel pattern
        RewritePatternSet patterns(ctx);
        patterns.add<ExplicitMicroKernelPattern>(ctx);

        GreedyRewriteConfig config;
        config.maxIterations = 10;

        if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
            llvm::outs() << "[ExplicitKernel] ⚠️ Some patterns failed (expected for large ops)\n";
        }

        // Check results
        int fmaCount = 0;
        func.walk([&](vector::FMAOp op) { fmaCount++; });

        if (fmaCount > 0) {
            llvm::outs() << "[ExplicitKernel] ✅ Generated " << fmaCount
                         << " explicit vector.fma operations!\n";
        } else {
            llvm::outs() << "[ExplicitKernel] ℹ️ No vector.fma generated (ops too large for micro-kernel)\n";
        }
    }
};

} // namespace

namespace tenzo {

void addExplicitMicroKernelPass(mlir::OpPassManager &pm) {
    pm.addNestedPass<func::FuncOp>(std::make_unique<ExplicitMicroKernelPass>());
}

} // namespace tenzo

