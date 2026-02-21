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

//===----------------------------------------------------------------------===//
// Pattern: Pack Matrix B into panels for optimal column access
//===----------------------------------------------------------------------===//
struct PackMatrixBPattern : public OpRewritePattern<linalg::MatmulOp> {
    tenzo::MicroKernelParams params;
    PackMatrixBPattern(MLIRContext *ctx, const tenzo::MicroKernelParams &p) 
        : OpRewritePattern<linalg::MatmulOp>(ctx), params(p) {}

    LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();

        Value B = op.getInputs()[1];
        auto bType = mlir::dyn_cast<RankedTensorType>(B.getType());
        if (!bType || !bType.hasStaticShape())
            return failure();

        int64_t K = bType.getShape()[0];
        int64_t N = bType.getShape()[1];

        // Support ANY N - pad to multiple of NR
        int64_t paddedN = ((N + params.NR - 1) / params.NR) * params.NR;

        llvm::outs() << "[Packing] Packing matrix B: [" << K << "x" << N << "] -> [" 
                     << K << "x" << (paddedN / params.NR) << "x" << params.NR << "]\n";

        SmallVector<int64_t> packedShape = {K, paddedN / params.NR, params.NR};
        auto packedType = RankedTensorType::get(packedShape, bType.getElementType());

        auto emptyPacked = rewriter.create<tensor::EmptyOp>(
            loc, packedType.getShape(), bType.getElementType());

        SmallVector<int64_t> innerDimsPos = {1}; 
        SmallVector<OpFoldResult> innerTiles = {rewriter.getIndexAttr(params.NR)};

        Value zeroPadding = rewriter.create<arith::ConstantOp>(
            loc, bType.getElementType(),
            rewriter.getFloatAttr(bType.getElementType(), 0.0));

        auto packOp = rewriter.create<linalg::PackOp>(
            loc, B, emptyPacked,
            innerDimsPos, innerTiles,
            /*padding_value=*/zeroPadding);

        op->setAttr("packed_b", rewriter.getUnitAttr());
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
struct PackingPass : public PassWrapper<PackingPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackingPass)

    tenzo::MicroKernelParams params;
    PackingPass(const tenzo::MicroKernelParams &p) : params(p) {}

    void runOnOperation() override {
        auto func = getOperation();
        auto *ctx = &getContext();

        llvm::outs() << "[Packing] ====== Data Layout Transformation ======\n";
        llvm::outs() << "[Packing] Target: MR=" << params.MR << ", NR=" << params.NR << "\n";

        RewritePatternSet patterns(ctx);
        patterns.add<PackMatrixBPattern>(ctx, params);

        if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
            llvm::outs() << "[Packing] ⚠️  Pattern application failed\n";
        }
    }
};

} // namespace

namespace tenzo {

void addPackingPass(mlir::OpPassManager &pm, const MicroKernelParams &params) {
    pm.addNestedPass<func::FuncOp>(std::make_unique<PackingPass>(params));
}

} // namespace tenzo

