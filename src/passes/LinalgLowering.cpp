#include "passes/Passes.h"
#include "dialect/TenzoDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// FusedAddRelu -> Linalg.generic
//===----------------------------------------------------------------------===//
struct FusedLoweringToLinalg : public OpConversionPattern<tenzo::FusedAddReluOp> {
    using OpConversionPattern<tenzo::FusedAddReluOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::FusedAddReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();

        SmallVector<AffineMap, 3> indexingMaps(
            3, rewriter.getMultiDimIdentityMap(resultType.getRank()));

        SmallVector<utils::IteratorType, 1> iteratorTypes(
            resultType.getRank(), utils::IteratorType::parallel);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            resultType,
            ValueRange{adaptor.getLhs(), adaptor.getRhs()},
            ValueRange{rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType)},
            indexingMaps,
            iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
                auto sum = nestedBuilder.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
                auto zeroAttr = nestedBuilder.getF32FloatAttr(0.0f);
                auto zero = nestedBuilder.create<arith::ConstantOp>(nestedLoc, zeroAttr);
                auto res = nestedBuilder.create<arith::MaximumFOp>(nestedLoc, sum, zero);
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, res.getResult());
            });
        return success();
    }
};

//===----------------------------------------------------------------------===//
// tenzo.matmul -> linalg.matmul
//===----------------------------------------------------------------------===//
struct MatMulLoweringToLinalg : public OpConversionPattern<tenzo::MatMulOp> {
    using OpConversionPattern<tenzo::MatMulOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::MatMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();

        // Create zero-initialized output tensor
        auto zero = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(elemType));
        auto emptyTensor = rewriter.create<tensor::EmptyOp>(
            loc, resultType.getShape(), elemType);
        auto filledTensor = rewriter.create<linalg::FillOp>(
            loc, ValueRange{zero}, ValueRange{emptyTensor});

        // Create linalg.matmul
        rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
            op,
            resultType,
            ValueRange{adaptor.getLhs(), adaptor.getRhs()},
            ValueRange{filledTensor.getResult(0)});

        return success();
    }
};

//===----------------------------------------------------------------------===//
// tenzo.conv2d -> linalg.conv_2d_nhwc_hwcf
//===----------------------------------------------------------------------===//
struct Conv2DLoweringToLinalg : public OpConversionPattern<tenzo::Conv2DOp> {
    using OpConversionPattern<tenzo::Conv2DOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::Conv2DOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();

        // Create output tensor initialized to zero
        auto zero = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(elemType));
        auto emptyTensor = rewriter.create<tensor::EmptyOp>(
            loc, resultType.getShape(), elemType);
        auto filledTensor = rewriter.create<linalg::FillOp>(
            loc, ValueRange{zero}, ValueRange{emptyTensor});

        // Default strides and dilations (1x1)
        auto stridesAttr = rewriter.getDenseI64ArrayAttr({1, 1});
        auto dilationsAttr = rewriter.getDenseI64ArrayAttr({1, 1});

        // Create linalg.conv_2d_nhwc_hwcf
        rewriter.replaceOpWithNewOp<linalg::Conv2DNhwcHwcfOp>(
            op,
            resultType,
            ValueRange{adaptor.getInput(), adaptor.getFilter()},
            ValueRange{filledTensor.getResult(0)},
            stridesAttr,
            dilationsAttr);

        return success();
    }
};

} // namespace

void tenzo::populateTenzoToLinalgConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<FusedLoweringToLinalg>(patterns.getContext());
    patterns.add<MatMulLoweringToLinalg>(patterns.getContext());
    patterns.add<Conv2DLoweringToLinalg>(patterns.getContext());
}