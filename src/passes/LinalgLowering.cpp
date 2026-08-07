#include "passes/Passes.h"
#include "dialect/TenzoDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

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

        SmallVector<Value> dynamicSizes;
        for (int i = 0; i < resultType.getRank(); ++i) {
            if (resultType.isDynamicDim(i)) {
                dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, adaptor.getLhs(), i));
            }
        }

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            resultType,
            ValueRange{adaptor.getLhs(), adaptor.getRhs()},
            ValueRange{rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType, dynamicSizes)},
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
        SmallVector<Value> dynamicSizes;
        for (int i = 0; i < resultType.getRank(); ++i) {
            if (resultType.isDynamicDim(i)) {
                if (i == 0) dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, adaptor.getLhs(), 0));
                else if (i == 1) dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, adaptor.getRhs(), 1));
            }
        }

        auto emptyTensor = rewriter.create<tensor::EmptyOp>(
            loc, resultType.getShape(), elemType, dynamicSizes);
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
        SmallVector<Value> dynamicSizes;
        for (int i = 0; i < resultType.getRank(); ++i) {
            if (resultType.isDynamicDim(i)) {
                if (i == 0) dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, adaptor.getInput(), 0)); // N
                else if (i == 3) dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, adaptor.getFilter(), 3)); // C
                else {
                    // Out = In - Filter + 1 for strides=1 padding=0
                    auto inDim = rewriter.create<tensor::DimOp>(loc, adaptor.getInput(), i);
                    auto filterDim = rewriter.create<tensor::DimOp>(loc, adaptor.getFilter(), i-1); // HWIO
                    auto sub = rewriter.create<arith::SubIOp>(loc, inDim, filterDim);
                    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
                    auto add = rewriter.create<arith::AddIOp>(loc, sub, c1);
                    dynamicSizes.push_back(add);
                }
            }
        }

        auto emptyTensor = rewriter.create<tensor::EmptyOp>(
            loc, resultType.getShape(), elemType, dynamicSizes);
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

//===----------------------------------------------------------------------===//
// tenzo.quantize -> linalg.generic
//===----------------------------------------------------------------------===//
struct QuantizeLoweringToLinalg : public OpConversionPattern<tenzo::QuantizeOp> {
    using OpConversionPattern<tenzo::QuantizeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::QuantizeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();

        SmallVector<AffineMap, 2> indexingMaps(
            2, rewriter.getMultiDimIdentityMap(resultType.getRank()));

        SmallVector<utils::IteratorType, 1> iteratorTypes(
            resultType.getRank(), utils::IteratorType::parallel);

        SmallVector<Value> dynamicSizes;
        for (int i = 0; i < resultType.getRank(); ++i) {
            if (resultType.isDynamicDim(i)) {
                dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, adaptor.getInput(), i));
            }
        }

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            resultType,
            ValueRange{adaptor.getInput()},
            ValueRange{rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType, dynamicSizes)},
            indexingMaps,
            iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
                // Dummy lowering to just pass through for now, real implementation depends on scheme
                // Usually this would involve scaling, rounding, clamping
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
            });
        return success();
    }
};

//===----------------------------------------------------------------------===//
// tenzo.dequantize -> linalg.generic
//===----------------------------------------------------------------------===//
struct DequantizeLoweringToLinalg : public OpConversionPattern<tenzo::DequantizeOp> {
    using OpConversionPattern<tenzo::DequantizeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::DequantizeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();

        SmallVector<AffineMap, 2> indexingMaps(
            2, rewriter.getMultiDimIdentityMap(resultType.getRank()));

        SmallVector<utils::IteratorType, 1> iteratorTypes(
            resultType.getRank(), utils::IteratorType::parallel);

        SmallVector<Value> dynamicSizes;
        for (int i = 0; i < resultType.getRank(); ++i) {
            if (resultType.isDynamicDim(i)) {
                dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, adaptor.getInput(), i));
            }
        }

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            resultType,
            ValueRange{adaptor.getInput()},
            ValueRange{rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType, dynamicSizes)},
            indexingMaps,
            iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
                // Dummy pass through
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
            });
        return success();
    }
};

} // namespace

void tenzo::populateTenzoToLinalgConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<FusedLoweringToLinalg>(patterns.getContext());
    patterns.add<MatMulLoweringToLinalg>(patterns.getContext());
    patterns.add<Conv2DLoweringToLinalg>(patterns.getContext());
    patterns.add<QuantizeLoweringToLinalg>(patterns.getContext());
    patterns.add<DequantizeLoweringToLinalg>(patterns.getContext());
}

namespace {
struct TenzoToLinalgPass : public PassWrapper<TenzoToLinalgPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TenzoToLinalgPass)

    void runOnOperation() override {
        auto func = getOperation();
        auto *ctx = &getContext();

        RewritePatternSet patterns(ctx);
        tenzo::populateTenzoToLinalgConversionPatterns(patterns);

        ConversionTarget target(*ctx);
        target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect, tensor::TensorDialect>();
        target.addIllegalDialect<tenzo::TenzoDialect>();

        if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // namespace

void tenzo::addTenzoToLinalgPass(mlir::OpPassManager &pm) {
    pm.addNestedPass<func::FuncOp>(std::make_unique<TenzoToLinalgPass>());
}