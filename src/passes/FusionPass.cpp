#include "passes/Passes.h"
#include "dialect/TenzoDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
    // Патерн: шукаємо ReLU(Add(a, b)) -> FusedAddRelu(a, b)
    struct FuseAddRelu : public OpRewritePattern<tenzo::ReluOp> {
        using OpRewritePattern<tenzo::ReluOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(tenzo::ReluOp relu,
                                    PatternRewriter &rewriter) const override {
            // Отримуємо операцію, що створила вхід для ReLU
            auto addOp = relu.getInput().getDefiningOp<tenzo::AddOp>();

            // Якщо це не Add — ми не можемо фузити
            if (!addOp) return failure();

            // Замінюємо ReLU на нову операцію FusedAddRelu
            rewriter.replaceOpWithNewOp<tenzo::FusedAddReluOp>(
                relu,
                addOp.getLhs(),
                addOp.getRhs()
            );
            return success();
        }
    };
} // namespace

namespace tenzo {
    void populateFusionPatterns(RewritePatternSet &patterns) {
        patterns.add<FuseAddRelu>(patterns.getContext());
    }
}