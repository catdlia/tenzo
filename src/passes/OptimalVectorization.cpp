// OptimalVectorization.cpp - GotoBLAS-style explicit micro-kernel
// Target: 6x16 tiles with perfect register allocation for AVX2
//
// Strategy:
// 1. Work AFTER bufferization (on memref, not tensors)
// 2. Find small matmul ops (result of tiling)
// 3. Generate explicit vector.fma operations
//
// Register allocation:
//   ymm0-ymm11:  12 accumulators (6 rows × 2 vectors)
//   ymm12-ymm13: 2 A broadcast registers
//   ymm14-ymm15: 2 B load registers

#include "passes/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// Perfect AVX2 micro-kernel size
constexpr int64_t MICRO_M = 6;   // 6 rows
constexpr int64_t MICRO_N = 16;  // 16 cols (2 AVX vectors)
constexpr int64_t VEC_SIZE = 8;  // AVX2 vector length

//===----------------------------------------------------------------------===//
// Pattern: Replace small matmul with explicit vector operations
//===----------------------------------------------------------------------===//
struct OptimalMicroKernelPattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const override {
        // Only work on bufferized operations
        if (!op.hasPureBufferSemantics())
            return failure();

        auto loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];

        auto aType = A.getType().dyn_cast<MemRefType>();
        auto bType = B.getType().dyn_cast<MemRefType>();
        auto cType = C.getType().dyn_cast<MemRefType>();

        if (!aType || !bType || !cType || !aType.hasStaticShape())
            return failure();

        int64_t M = aType.getShape()[0];
        int64_t K = aType.getShape()[1];
        int64_t N = bType.getShape()[1];

        // Only match our exact micro-kernel size
        if (M != MICRO_M || N != MICRO_N) {
            return failure();
        }

        llvm::outs() << "[OptimalKernel] Generating 6x16 micro-kernel (K=" << K << ")\n";

        auto f32Type = rewriter.getF32Type();
        auto vecType = VectorType::get({VEC_SIZE}, f32Type);
        auto zeroF32 = rewriter.create<arith::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(0.0f));

        // Create 12 accumulator registers: c[6][2]
        SmallVector<SmallVector<Value>> accums(MICRO_M);
        for (int64_t i = 0; i < MICRO_M; i++) {
            auto iIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
            for (int64_t j = 0; j < 2; j++) {
                auto jOffset = rewriter.create<arith::ConstantIndexOp>(loc, j * VEC_SIZE);
                Value cVec = rewriter.create<vector::TransferReadOp>(
                    loc, vecType, C, ValueRange{iIdx, jOffset}, zeroF32);
                accums[i].push_back(cVec);
            }
        }

        // K loop with explicit FMA
        auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto kBound = rewriter.create<arith::ConstantIndexOp>(loc, K);

        // Flatten accumulators for loop
        SmallVector<Value> flatAccums;
        for (auto &row : accums) flatAccums.append(row.begin(), row.end());

        auto forOp = rewriter.create<scf::ForOp>(
            loc, c0, kBound, c1, flatAccums,
            [&](OpBuilder &b, Location loc, Value k, ValueRange iterArgs) {
                SmallVector<Value> newAccums;

                // Load B vectors (2 vectors, use ymm14-ymm15)
                Value b0 = b.create<vector::TransferReadOp>(
                    loc, vecType, B, ValueRange{k, c0}, zeroF32);
                auto c8 = b.create<arith::ConstantIndexOp>(loc, 8);
                Value b1 = b.create<vector::TransferReadOp>(
                    loc, vecType, B, ValueRange{k, c8}, zeroF32);

                // Process all 6 rows
                for (int64_t i = 0; i < MICRO_M; i++) {
                    auto iIdx = b.create<arith::ConstantIndexOp>(loc, i);

                    // Load and broadcast A[i,k] (use ymm12)
                    Value aScalar = b.create<memref::LoadOp>(loc, A, ValueRange{iIdx, k});
                    Value aVec = b.create<vector::BroadcastOp>(loc, vecType, aScalar);

                    // FMA for both vector columns
                    Value acc0 = iterArgs[i * 2];
                    Value acc1 = iterArgs[i * 2 + 1];

                    // c[i][0] += a[i] * b[0]
                    Value fma0 = b.create<vector::FMAOp>(loc, aVec, b0, acc0);
                    // c[i][1] += a[i] * b[1]
                    Value fma1 = b.create<vector::FMAOp>(loc, aVec, b1, acc1);

                    newAccums.push_back(fma0);
                    newAccums.push_back(fma1);
                }

                b.create<scf::YieldOp>(loc, newAccums);
            });

        // Store results
        auto results = forOp.getResults();
        for (int64_t i = 0; i < MICRO_M; i++) {
            auto iIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
            for (int64_t j = 0; j < 2; j++) {
                auto jOffset = rewriter.create<arith::ConstantIndexOp>(loc, j * VEC_SIZE);
                rewriter.create<vector::TransferWriteOp>(
                    loc, results[i * 2 + j], C, ValueRange{iIdx, jOffset});
            }
        }

        rewriter.eraseOp(op);
        llvm::outs() << "[OptimalKernel] ✅ Generated " << (MICRO_M * 2 * K) << " FMA ops\n";
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
struct OptimalVectorizationPass
    : public PassWrapper<OptimalVectorizationPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimalVectorizationPass)

    void runOnOperation() override {
        auto func = getOperation();
        auto *ctx = &getContext();

        llvm::outs() << "[OptimalVectorization] Looking for 6x16 micro-kernels...\n";

        // Count candidates
        int count = 0;
        func.walk([&](linalg::MatmulOp op) {
            if (op.hasPureBufferSemantics()) {
                auto aType = op.getInputs()[0].getType().dyn_cast<MemRefType>();
                if (aType && aType.hasStaticShape()) {
                    auto shape = aType.getShape();
                    llvm::outs() << "  Found: " << shape[0] << "x" << shape[1] << "\n";
                    if (shape[0] == MICRO_M) count++;
                }
            }
        });

        if (count == 0) {
            llvm::outs() << "[OptimalVectorization] No 6x16 micro-kernels found\n";
            return;
        }

        llvm::outs() << "[OptimalVectorization] Found " << count << " candidate(s)\n";

        // Apply pattern
        RewritePatternSet patterns(ctx);
        patterns.add<OptimalMicroKernelPattern>(ctx);

        if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
            llvm::outs() << "[OptimalVectorization] Some patterns failed\n";
        }

        // Check results
        int fmaCount = 0;
        func.walk([&](vector::FMAOp) { fmaCount++; });

        if (fmaCount > 0) {
            llvm::outs() << "[OptimalVectorization] ✅ SUCCESS: " << fmaCount << " vector.fma ops\n";
        }
    }
};

} // namespace

namespace tenzo {

void addOptimalVectorizationPass(mlir::OpPassManager &pm) {
    pm.addNestedPass<func::FuncOp>(std::make_unique<OptimalVectorizationPass>());
}

} // namespace tenzo

