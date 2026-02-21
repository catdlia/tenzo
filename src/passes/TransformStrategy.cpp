//===----------------------------------------------------------------------===//
// TransformStrategy.cpp - Transform Dialect based optimization
//===----------------------------------------------------------------------===//
// Uses Transform operations programmatically for tiling and vectorization.
//===----------------------------------------------------------------------===//

#include "passes/Passes.h"
#include "context/HardwareInfo.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Transform Strategy Pass - Programmatic Tiling + Vectorization
// Works on TENSOR level (before bufferization)
//===----------------------------------------------------------------------===//
struct TransformStrategyPass
    : public PassWrapper<TransformStrategyPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformStrategyPass)

    int64_t tileM, tileN, tileK;

    TransformStrategyPass(int64_t m = 8, int64_t n = 16, int64_t k = 4)
        : tileM(m), tileN(n), tileK(k) {}

    TransformStrategyPass(const TransformStrategyPass &other)
        : PassWrapper(other), tileM(other.tileM), tileN(other.tileN), tileK(other.tileK) {}

    StringRef getArgument() const final { return "tenzo-transform-strategy"; }
    StringRef getDescription() const final {
        return "Apply programmatic tiling + vectorization strategy";
    }

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        MLIRContext *ctx = &getContext();

        llvm::outs() << "[Transform] ====== Programmatic Tiling + Vectorization ======\n";
        llvm::outs() << "[Transform] Tile sizes: " << tileM << "x" << tileN << "x" << tileK << "\n";

        // Count matmuls
        SmallVector<linalg::MatmulOp> matmulOps;
        func.walk([&](linalg::MatmulOp op) {
            matmulOps.push_back(op);
        });

        llvm::outs() << "[Transform] Found " << matmulOps.size() << " matmul(s)\n";

        if (matmulOps.empty()) return;

        // ===== PHASE 1: TILING using SCF =====
        // Only tile if matrix is larger than tile size
        llvm::outs() << "[Transform] Phase 1: Tiling...\n";

        for (auto matmul : matmulOps) {
            // Check if this is on tensors and large enough to tile
            if (matmul.hasPureTensorSemantics()) {
                auto resultType = mlir::dyn_cast<ShapedType>(matmul.getResult(0).getType());
                if (!resultType || !resultType.hasStaticShape()) continue;

                auto shape = resultType.getShape();
                // Only tile if matrix is larger than tile size
                if (shape[0] <= tileM && shape[1] <= tileN) {
                    llvm::outs() << "[Transform] Matrix " << shape[0] << "x" << shape[1]
                                 << " is small, skipping tiling\n";
                    continue;
                }

                IRRewriter rewriter(ctx);
                rewriter.setInsertionPoint(matmul);

                // Set up tiling options
                scf::SCFTilingOptions tilingOptions;
                SmallVector<OpFoldResult> tileSizes = {
                    rewriter.getIndexAttr(tileM),
                    rewriter.getIndexAttr(tileN),
                    rewriter.getIndexAttr(tileK)
                };
                tilingOptions.setTileSizes(tileSizes);

                // Tile the operation
                FailureOr<scf::SCFTilingResult> tilingResult =
                    scf::tileUsingSCF(rewriter,
                        cast<TilingInterface>(matmul.getOperation()),
                        tilingOptions);

                if (succeeded(tilingResult)) {
                    llvm::outs() << "[Transform] ✅ Tiled " << shape[0] << "x" << shape[1] << " matmul!\n";
                    rewriter.replaceOp(matmul, tilingResult->replacements);
                } else {
                    llvm::outs() << "[Transform] ⚠️ Tiling failed\n";
                }
            }
        }

        // ===== PHASE 2: VECTORIZATION of matmuls =====
        llvm::outs() << "[Transform] Phase 2: Vectorizing...\n";

        // Vectorize original matmuls (must be small enough - up to 64x64)
        SmallVector<linalg::MatmulOp> toVectorize;
        func.walk([&](linalg::MatmulOp op) {
            auto resultType = mlir::dyn_cast<ShapedType>(op.getResult(0).getType());
            if (resultType && resultType.hasStaticShape()) {
                int64_t totalElements = resultType.getNumElements();
                // Only vectorize matrices up to 64x64 (4096 elements)
                if (totalElements <= 4096) {
                    toVectorize.push_back(op);
                }
            }
        });

        llvm::outs() << "[Transform] Found " << toVectorize.size() << " matmul(s) to vectorize\n";

        for (auto matmul : toVectorize) {
            IRRewriter rewriter(ctx);
            rewriter.setInsertionPoint(matmul);

            if (succeeded(linalg::vectorize(rewriter, matmul))) {
                llvm::outs() << "[Transform] ✅ Vectorized matmul!\n";
            }
        }

        // ===== PHASE 3: VECTOR LOWERING =====
        llvm::outs() << "[Transform] Phase 3: Vector lowering...\n";
        {
            RewritePatternSet patterns(ctx);

            // Lower vector.contract to OuterProduct (FMA)
            vector::populateVectorContractLoweringPatterns(patterns, 
                vector::VectorContractLowering::OuterProduct);

            // Lower reductions
            vector::populateVectorMultiReductionLoweringPatterns(
                patterns, vector::VectorMultiReductionLowering::InnerReduction);

            // Cleanup
            vector::populateVectorToVectorCanonicalizationPatterns(patterns);
            vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

            (void)applyPatternsGreedily(func, std::move(patterns));
        }

        // Check results
        bool hasScfFor = false;
        bool hasVectorContract = false;
        bool hasVectorOps = false;
        int remainingMatmul = 0;

        func.walk([&](Operation *op) {
            if (isa<scf::ForOp>(op)) hasScfFor = true;
            if (isa<vector::ContractionOp>(op)) hasVectorContract = true;
            if (isa<vector::TransferReadOp>(op) || isa<vector::OuterProductOp>(op))
                hasVectorOps = true;
            if (isa<linalg::MatmulOp>(op)) remainingMatmul++;
        });

        llvm::outs() << "[Transform] ====== Results ======\n";
        if (hasScfFor) llvm::outs() << "[Transform] ✅ Generated scf.for loops\n";
        if (hasVectorContract) llvm::outs() << "[Transform] ✅ Generated vector.contract\n";
        if (hasVectorOps) llvm::outs() << "[Transform] ✅ Generated vector ops (FMA ready)\n";
        if (remainingMatmul > 0) {
            llvm::outs() << "[Transform] ⚠️ " << remainingMatmul << " matmul(s) remaining\n";
        } else {
            llvm::outs() << "[Transform] ✅ All matmuls processed!\n";
        }
    }
};

} // namespace

namespace tenzo {

void addTransformStrategyPass(mlir::OpPassManager &pm,
                               HardwareInfo::TileSizes tiles) {
    pm.addNestedPass<func::FuncOp>(
        std::make_unique<TransformStrategyPass>(tiles.M, tiles.N, tiles.K));
}

} // namespace tenzo

