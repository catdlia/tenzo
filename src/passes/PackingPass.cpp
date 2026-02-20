// PackingPass.cpp - Data layout transformation for optimal memory access
// Implements BLIS-style packing: transform row-major to panel layout
//
// Strategy:
// 1. Pack matrix A: [M, K] -> [M/Mr, K, Mr] (panel of Mr rows)
// 2. Pack matrix B: [K, N] -> [K, N/Nr, Nr] (panel of Nr cols)
//
// This ensures:
// - A reads are sequential within Mr rows
// - B reads are sequential within Nr cols
// - Perfect for AVX2 vectorization (no strided access!)

#include "passes/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// Micro-kernel dimensions (must match register allocation strategy)
constexpr int64_t MR = 6;   // Panel height for A
constexpr int64_t NR = 16;  // Panel width for B

//===----------------------------------------------------------------------===//
// Pattern: Pack Matrix A into panels
//===----------------------------------------------------------------------===//
struct PackMatrixAPattern : public OpRewritePattern<linalg::FillOp> {
    using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::FillOp op, PatternRewriter &rewriter) const override {
        auto outputType = op.getOutputs()[0].getType().dyn_cast<RankedTensorType>();
        if (!outputType || outputType.getRank() != 2)
            return failure();

        // Only pack if this looks like matrix A (M x K where K is reduction dim)
        // We identify this heuristically by checking usage
        // TODO: Better pattern matching

        return failure(); // Skip for now - will implement after testing B packing
    }
};

//===----------------------------------------------------------------------===//
// Pattern: Pack Matrix B into panels for optimal column access
//===----------------------------------------------------------------------===//
struct PackMatrixBPattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();

        // Get operands
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];

        auto bType = B.getType().dyn_cast<RankedTensorType>();
        if (!bType || !bType.hasStaticShape())
            return failure();

        int64_t K = bType.getShape()[0];
        int64_t N = bType.getShape()[1];

        // Support ANY N - we'll pad to the nearest multiple of NR
        int64_t paddedN = ((N + NR - 1) / NR) * NR;

        llvm::outs() << "[Packing] Packing matrix B: [" << K << "x" << N << "]";
        if (N != paddedN) {
            llvm::outs() << " (padded to " << paddedN << " cols)";
        }
        llvm::outs() << " -> [" << K << "x" << (paddedN/NR) << "x" << NR << "]\n";

        // Create packed shape: [K, paddedN/Nr, Nr]
        SmallVector<int64_t> packedShape = {K, paddedN / NR, NR};
        auto packedType = RankedTensorType::get(packedShape, bType.getElementType());

        // Create empty tensor for packed output
        auto emptyPacked = rewriter.create<tensor::EmptyOp>(
            loc, packedType.getShape(), bType.getElementType());

        // Use tensor.pack to transform layout with PADDING
        // MLIR 18 API: pack(source, dest, inner_dims_pos, inner_tiles, outer_dims_perm, padding_value)
        SmallVector<int64_t> innerDimsPos = {1}; // Pack dimension 1 (N)
        SmallVector<OpFoldResult> innerTiles = {rewriter.getIndexAttr(NR)};
        SmallVector<int64_t> outerDimsPerm = {}; // Keep natural order

        // Create zero padding value for incomplete tiles
        Value zeroPadding = rewriter.create<arith::ConstantOp>(
            loc, bType.getElementType(),
            rewriter.getFloatAttr(bType.getElementType(), 0.0));

        auto packOp = rewriter.create<tensor::PackOp>(
            loc, B, emptyPacked,
            innerDimsPos, innerTiles,
            /*padding_value=*/zeroPadding,
            outerDimsPerm);

        // Now we need to create a new matmul that works with packed B
        // For now, just mark that packing happened
        llvm::outs() << "[Packing] ✅ Created packed B layout with zero-padding\n";

        // Don't replace yet - we need to also handle the matmul differently
        // This will be done in ExplicitMicroKernelPass
        op->setAttr("packed_b", rewriter.getUnitAttr());

        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
struct PackingPass : public PassWrapper<PackingPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackingPass)

    void runOnOperation() override {
        auto func = getOperation();
        auto *ctx = &getContext();

        llvm::outs() << "[Packing] ====== Data Layout Transformation ======\n";
        llvm::outs() << "[Packing] Target: Panel layout for BLIS-style micro-kernel\n";
        llvm::outs() << "[Packing] Mr=" << MR << " (rows), Nr=" << NR << " (cols)\n\n";

        // Count matmul operations
        int matmulCount = 0;
        func.walk([&](linalg::MatmulOp) { matmulCount++; });

        if (matmulCount == 0) {
            llvm::outs() << "[Packing] No matmul ops found\n";
            return;
        }

        llvm::outs() << "[Packing] Found " << matmulCount << " matmul op(s)\n";

        // Apply packing patterns
        RewritePatternSet patterns(ctx);
        patterns.add<PackMatrixBPattern>(ctx);

        if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
            llvm::outs() << "[Packing] ⚠️  Pattern application failed\n";
        }

        // Check results
        int packedCount = 0;
        func.walk([&](linalg::MatmulOp op) {
            if (op->hasAttr("packed_b")) packedCount++;
        });

        if (packedCount > 0) {
            llvm::outs() << "[Packing] ✅ Successfully marked " << packedCount << " matmul(s) for packing\n";
        } else {
            llvm::outs() << "[Packing] ⚠️  No operations were packed (constraints not met?)\n";
        }

        llvm::outs() << "[Packing] ======================================\n";
    }
};

} // namespace

namespace tenzo {

void addPackingPass(mlir::OpPassManager &pm) {
    pm.addNestedPass<func::FuncOp>(std::make_unique<PackingPass>());
}

} // namespace tenzo

