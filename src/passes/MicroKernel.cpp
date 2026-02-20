// MicroKernel.cpp - GotoBLAS-style explicit micro-kernel generation
// Target: 6x16 micro-kernel for AVX2 (12 accumulators + 4 temp registers)
//
// Register allocation for AVX2 (16 ymm registers):
//   ymm0-ymm11: 6x2 accumulators (C matrix)
//   ymm12-ymm13: broadcast A values
//   ymm14-ymm15: loaded B vectors
//
// This generates explicit FMA instructions without relying on LLVM autovectorization

#include "passes/Passes.h"
#include "context/HardwareInfo.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// Micro-kernel dimensions for AVX2
// 6x16 gives us: 6*2 = 12 accumulator registers + 4 for loading
constexpr int64_t MR = 6;   // Rows of C computed per micro-kernel
constexpr int64_t NR = 16;  // Columns of C (2 AVX2 vectors)
constexpr int64_t VL = 8;   // Vector length (AVX2 = 8 floats)

//===----------------------------------------------------------------------===//
// GotoBLAS-style Micro-Kernel Pattern
// Converts linalg.matmul on memrefs to explicit vector.outerproduct
//===----------------------------------------------------------------------===//
struct GotoBLASMicroKernelPattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                  PatternRewriter &rewriter) const override {
        // Only handle buffer semantics (after bufferization)
        if (!op.hasPureBufferSemantics())
            return failure();

        auto loc = op.getLoc();

        // Get operands
        Value A = op.getInputs()[0];  // [M, K]
        Value B = op.getInputs()[1];  // [K, N]
        Value C = op.getOutputs()[0]; // [M, N]

        // Get shapes
        auto aType = A.getType().cast<MemRefType>();
        auto bType = B.getType().cast<MemRefType>();
        auto cType = C.getType().cast<MemRefType>();

        if (!aType.hasStaticShape() || !bType.hasStaticShape())
            return failure();

        int64_t M = aType.getShape()[0];
        int64_t K = aType.getShape()[1];
        int64_t N = bType.getShape()[1];

        // Only apply to micro-kernels that fit our pattern
        // We want exactly MR x NR or smaller
        if (M > MR || N > NR) {
            return failure();
        }

        // Check if N is compatible with our vector length
        if (N % VL != 0 && N != NR) {
            return failure();
        }

        llvm::outs() << "[MicroKernel] Generating GotoBLAS kernel for "
                     << M << "x" << K << "x" << N << "\n";

        auto f32Type = rewriter.getF32Type();
        auto vecType = VectorType::get({VL}, f32Type);

        // Number of vector columns
        int64_t numVecCols = (N + VL - 1) / VL;

        // Create zero constant for initialization
        auto zeroF32 = rewriter.create<arith::ConstantOp>(
            loc, f32Type, rewriter.getF32FloatAttr(0.0f));
        auto zeroVec = rewriter.create<vector::BroadcastOp>(loc, vecType, zeroF32);

        // Create accumulator registers (M x numVecCols vectors)
        SmallVector<Value> accumulators;
        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < numVecCols; j++) {
                // Load initial C values if C is not zero-initialized
                auto jOffset = rewriter.create<arith::ConstantIndexOp>(loc, j * VL);
                auto iIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);

                // Read existing C value
                Value cVec = rewriter.create<vector::TransferReadOp>(
                    loc, vecType, C, ValueRange{iIdx, jOffset},
                    /*padding=*/zeroF32);
                accumulators.push_back(cVec);
            }
        }

        // Main K loop - FMA operations
        auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto kBound = rewriter.create<arith::ConstantIndexOp>(loc, K);

        auto forOp = rewriter.create<scf::ForOp>(
            loc, c0, kBound, c1, accumulators,
            [&](OpBuilder &builder, Location loc, Value k, ValueRange iterArgs) {
                SmallVector<Value> newAccumulators;

                // Load B vectors for this k iteration
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

                    // Load and broadcast A[i,k]
                    Value aScalar = builder.create<memref::LoadOp>(loc, A, ValueRange{iIdx, k});
                    Value aVec = builder.create<vector::BroadcastOp>(loc, vecType, aScalar);

                    // FMA: C[i,:] += A[i,k] * B[k,:]
                    for (int64_t j = 0; j < numVecCols; j++) {
                        Value acc = iterArgs[accIdx];

                        // vector.fma: acc = a * b + acc
                        Value fmaResult = builder.create<vector::FMAOp>(
                            loc, aVec, bVecs[j], acc);

                        newAccumulators.push_back(fmaResult);
                        accIdx++;
                    }
                }

                builder.create<scf::YieldOp>(loc, newAccumulators);
            });

        // Store results back to C
        auto results = forOp.getResults();
        int resIdx = 0;
        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < numVecCols; j++) {
                auto iIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
                auto jOffset = rewriter.create<arith::ConstantIndexOp>(loc, j * VL);

                rewriter.create<vector::TransferWriteOp>(
                    loc, results[resIdx], C, ValueRange{iIdx, jOffset});
                resIdx++;
            }
        }

        // Erase the original matmul
        rewriter.eraseOp(op);

        llvm::outs() << "[MicroKernel] ✅ Generated " << M * numVecCols
                     << " FMA chains with " << K << " iterations\n";

        return success();
    }
};

//===----------------------------------------------------------------------===//
// Main Pass: Apply GotoBLAS micro-kernel transformation
//===----------------------------------------------------------------------===//
struct TenzoMicroKernelPass
    : public PassWrapper<TenzoMicroKernelPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TenzoMicroKernelPass)

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        MLIRContext *ctx = &getContext();

        llvm::outs() << "[MicroKernel] ====== GotoBLAS Micro-Kernel Pass ======\n";
        llvm::outs() << "[MicroKernel] Target: " << MR << "x" << NR << " (AVX2 optimal)\n";

        // Count matmuls
        int matmulCount = 0;
        func.walk([&](linalg::MatmulOp op) {
            if (op.hasPureBufferSemantics()) {
                matmulCount++;
                auto aType = op.getInputs()[0].getType().cast<MemRefType>();
                llvm::outs() << "[MicroKernel] Found bufferized matmul: "
                             << aType.getShape()[0] << "x" << aType.getShape()[1] << "\n";
            }
        });

        if (matmulCount == 0) {
            llvm::outs() << "[MicroKernel] No bufferized matmul ops found\n";
            return;
        }

        // Apply micro-kernel pattern
        RewritePatternSet patterns(ctx);
        patterns.add<GotoBLASMicroKernelPattern>(ctx);

        GreedyRewriteConfig config;
        config.maxIterations = 10;

        if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
            llvm::outs() << "[MicroKernel] ⚠️ Pattern application had failures\n";
        }

        // Check results
        int fmaCount = 0;
        func.walk([&](vector::FMAOp op) { fmaCount++; });

        if (fmaCount > 0) {
            llvm::outs() << "[MicroKernel] ✅ Generated " << fmaCount << " vector.fma operations!\n";
        } else {
            llvm::outs() << "[MicroKernel] ⚠️ No vector.fma generated (fallback to other path)\n";
        }
    }
};

} // namespace

namespace tenzo {

void addMicroKernelPass(mlir::OpPassManager &pm) {
    pm.addNestedPass<func::FuncOp>(std::make_unique<TenzoMicroKernelPass>());
}

} // namespace tenzo

