#include "passes/Passes.h"
#include "context/HardwareInfo.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Pattern 1: Vectorize Fill operations (for initialization)
//===----------------------------------------------------------------------===//
struct VectorizeFillPattern : public OpRewritePattern<linalg::FillOp> {
    using OpRewritePattern<linalg::FillOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(linalg::FillOp op, PatternRewriter &rewriter) const override {
        return linalg::vectorize(rewriter, op);
    }
};

//===----------------------------------------------------------------------===//
// Pattern 2: Vectorize micro-kernels (small MatMul after tiling)
// This converts linalg.matmul -> vector.contract
//===----------------------------------------------------------------------===//
struct VectorizeMicroKernelPattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const override {
        // Get result shape
        auto resultType = op.getResult(0).getType().dyn_cast<ShapedType>();
        if (!resultType || !resultType.hasStaticShape())
            return failure();

        auto shape = resultType.getShape(); // [M, N]
        int64_t m = shape[0];
        int64_t n = shape[1];

        // Get K from input A: [M, K]
        auto inputType = op.getInputs()[0].getType().dyn_cast<ShapedType>();
        if (!inputType || !inputType.hasStaticShape())
            return failure();
        int64_t k = inputType.getShape()[1];

        // FILTER: Only vectorize micro-kernels!
        // For AVX2, we target tiles like 8x16x4 (or similar)
        // If matrix is huge (512x512), skip it - Affine tiling will handle it
        int64_t totalOps = m * n * k;

        // Vectorize matrices up to 64x64x64 (262K ops)
        // This is safe and generates good vector.contract code
        if (totalOps > 262144) {
            // Too big - skip, Affine path handles this
            return failure();
        }

        llvm::outs() << "[Vectorize] Vectorizing matmul " << m << "x" << n << "x" << k
                     << " (" << totalOps << " ops)\n";

        // Call standard linalg vectorization
        // This transforms linalg.matmul -> vector.contract
        return linalg::vectorize(rewriter, op);
    }
};

//===----------------------------------------------------------------------===//
// Pattern 3: Tile large MatMul into micro-kernels
// DISABLED: SCF tiling on tensors causes bufferization issues
// Using Affine tiling in LLVMLowering.cpp instead
//===----------------------------------------------------------------------===//
struct TileMacroKernelPattern : public OpRewritePattern<linalg::MatmulOp> {
    int64_t tileM, tileN, tileK;

    TileMacroKernelPattern(MLIRContext *ctx, int64_t m, int64_t n, int64_t k)
        : OpRewritePattern<linalg::MatmulOp>(ctx), tileM(m), tileN(n), tileK(k) {}

    LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const override {
        // DISABLED: SCF tiling on tensor semantics causes issues with bufferization
        // The Affine tiling path in LLVMLowering.cpp handles this correctly
        return failure();
    }
};

//===----------------------------------------------------------------------===//
// Main Pass: Tile + Vectorize MatMul for AVX2
//===----------------------------------------------------------------------===//
struct TenzoExplicitVectorizePass
    : public PassWrapper<TenzoExplicitVectorizePass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TenzoExplicitVectorizePass)

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        MLIRContext *ctx = &getContext();

        // Get optimal tile sizes from hardware
        auto hwInfo = tenzo::HardwareInfo::detect();
        auto tiles = hwInfo.getOptimalTileSizes();

        llvm::outs() << "[Vectorize] ====== Explicit Vectorization Pass ======\n";
        llvm::outs() << "[Vectorize] Hardware tiles: " << tiles.M << "x" << tiles.N << "x" << tiles.K << "\n";

        // Count ops before
        int matmulCount = 0;
        func.walk([&](linalg::MatmulOp op) { matmulCount++; });
        llvm::outs() << "[Vectorize] Found " << matmulCount << " matmul(s)\n";

        // ===== PHASE 1: TILING =====
        // Cut large MatMul (512x512) into micro-kernels (8x16)
        {
            llvm::outs() << "[Vectorize] Phase 1: Tiling large matrices...\n";
            RewritePatternSet patterns(ctx);
            patterns.add<TileMacroKernelPattern>(ctx, tiles.M, tiles.N, tiles.K);

            GreedyRewriteConfig config;
            config.maxIterations = 100; // Allow many iterations for nested tiling

            (void)applyPatternsAndFoldGreedily(func, std::move(patterns), config);
        }

        // ===== PHASE 2: VECTORIZATION =====
        // Transform micro-kernels into vector.contract
        {
            llvm::outs() << "[Vectorize] Phase 2: Vectorizing micro-kernels...\n";
            RewritePatternSet patterns(ctx);
            patterns.add<VectorizeFillPattern>(ctx);
            patterns.add<VectorizeMicroKernelPattern>(ctx);

            GreedyRewriteConfig config;
            config.maxIterations = 50;

            (void)applyPatternsAndFoldGreedily(func, std::move(patterns), config);
        }

        // ===== PHASE 3: VECTOR OPTIMIZATIONS =====
        // Optimize vector operations
        {
            llvm::outs() << "[Vectorize] Phase 3: Optimizing vector ops...\n";
            RewritePatternSet patterns(ctx);

            // Add vector optimization patterns
            vector::populateVectorToVectorCanonicalizationPatterns(patterns);
            vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

            (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
        }

        // Check what we generated
        bool hasVectorContract = false;
        bool hasVectorTransfer = false;
        bool hasScfFor = false;
        int finalMatmulCount = 0;
        int vectorContractCount = 0;

        func.walk([&](Operation *op) {
            if (isa<vector::ContractionOp>(op)) {
                hasVectorContract = true;
                vectorContractCount++;
            }
            if (isa<vector::TransferReadOp>(op) || isa<vector::TransferWriteOp>(op))
                hasVectorTransfer = true;
            if (isa<scf::ForOp>(op)) hasScfFor = true;
            if (isa<linalg::MatmulOp>(op)) finalMatmulCount++;
        });

        llvm::outs() << "[Vectorize] ====== Results ======\n";
        if (hasVectorContract) {
            llvm::outs() << "[Vectorize] ✅ Generated " << vectorContractCount << " vector.contract (explicit FMA)!\n";
        }
        if (hasVectorTransfer) {
            llvm::outs() << "[Vectorize] ✅ Generated vector.transfer operations\n";
        }
        if (hasScfFor) {
            llvm::outs() << "[Vectorize] ✅ Generated scf.for loops (from tiling)\n";
        }
        if (finalMatmulCount > 0) {
            llvm::outs() << "[Vectorize] ⚠️  " << finalMatmulCount << " matmul(s) remaining (will use Affine path)\n";
        }
        if (!hasVectorContract && !hasVectorTransfer && !hasScfFor) {
            llvm::outs() << "[Vectorize] ⚠️  No explicit vectorization (fallback to Affine)\n";
        }
    }
};

} // namespace

void tenzo::addTenzoVectorizationPass(mlir::OpPassManager &pm) {
    // Explicit vectorization pass with tiling + vectorization
    pm.addNestedPass<func::FuncOp>(std::make_unique<TenzoExplicitVectorizePass>());

    // Cleanup
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
}
