#include "passes/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

/// Pattern: linalg.matmul -> linalg.generic (GELU) => linalg.matmul {activation="gelu"}
struct MatmulGeluFusionPattern : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
        // GELU detection logic (simplified: check for math.tanh in region)
        bool isGelu = false;
        genericOp.getRegion().walk([&](math::TanhOp) {
            isGelu = true;
        });

        if (!isGelu) return failure();

        Value input = genericOp.getInputs()[0];
        auto matmulOp = input.getDefiningOp<linalg::MatmulOp>();
        if (!matmulOp) return failure();

        llvm::outs() << "[Fusion] Fusing MatMul + GELU into single operation\n";
        matmulOp->setAttr("activation", rewriter.getStringAttr("gelu"));
        rewriter.replaceOp(genericOp, matmulOp.getResults());

        return success();
    }
};

} // namespace

namespace tenzo {
    void populateFusionPatterns(RewritePatternSet &patterns) {
        patterns.add<MatmulGeluFusionPattern>(patterns.getContext());
    }
}
