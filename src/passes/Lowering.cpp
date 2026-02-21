#include "passes/Passes.h"
#include "dialect/TenzoDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace tenzo {

// 1. Патерн для звичайного Add
struct AddLowering : public OpConversionPattern<tenzo::AddOp> {
    using OpConversionPattern<tenzo::AddOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::AddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::AddFOp>(op, adaptor.getLhs(), adaptor.getRhs());
        return success();
    }
};

// 2. Патерн для звичайного ReLU
struct ReluLowering : public OpConversionPattern<tenzo::ReluOp> {
    using OpConversionPattern<tenzo::ReluOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::ReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto type = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto zeroAttr = rewriter.getFloatAttr(type.getElementType(), 0.0);
        auto zero = rewriter.create<arith::ConstantOp>(op.getLoc(), zeroAttr);

        rewriter.replaceOpWithNewOp<arith::MaximumFOp>(op, adaptor.getInput(), zero);
        return success();
    }
};

// 3. Патерн для Fused Add + ReLU (НОВЕ)
struct FusedAddReluLowering : public OpConversionPattern<tenzo::FusedAddReluOp> {
    using OpConversionPattern<tenzo::FusedAddReluOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::FusedAddReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto type = mlir::cast<RankedTensorType>(op.getResult().getType());

        // Спершу робимо додавання
        auto add = rewriter.create<arith::AddFOp>(loc, adaptor.getLhs(), adaptor.getRhs());

        // Потім робимо ReLU (max з 0.0)
        auto zeroAttr = rewriter.getFloatAttr(type.getElementType(), 0.0);
        auto zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

        rewriter.replaceOpWithNewOp<arith::MaximumFOp>(op, add, zero);
        return success();
    }
};

// Реєстрація всіх патернів
void populateTenzoToArithConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<AddLowering, ReluLowering, FusedAddReluLowering>(patterns.getContext());
}

} // namespace tenzo