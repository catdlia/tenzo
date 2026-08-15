import re

cxx_code = r"""
//===----------------------------------------------------------------------===//
// tenzo.relu2 -> Linalg
//===----------------------------------------------------------------------===//
struct Relu2LoweringToLinalg : public OpConversionPattern<tenzo::Relu2Op> {
    using OpConversionPattern<tenzo::Relu2Op>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::Relu2Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();
        Value input = adaptor.getInput();

        Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);
        
        SmallVector<AffineMap, 2> maps(2, rewriter.getMultiDimIdentityMap(resultType.getRank()));
        SmallVector<utils::IteratorType, 3> iters(resultType.getRank(), utils::IteratorType::parallel);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op, resultType, ValueRange{input}, ValueRange{emptyTensor},
            maps, iters,
            [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
                Value zero = b.create<arith::ConstantOp>(nestedLoc, b.getFloatAttr(elemType, 0.0));
                Value max = b.create<arith::MaximumFOp>(nestedLoc, args[0], zero);
                Value mul = b.create<arith::MulFOp>(nestedLoc, max, max);
                b.create<linalg::YieldOp>(nestedLoc, mul);
            }
        );
        return success();
    }
};

//===----------------------------------------------------------------------===//
// tenzo.mul -> Linalg
//===----------------------------------------------------------------------===//
struct MulLoweringToLinalg : public OpConversionPattern<tenzo::MulOp> {
    using OpConversionPattern<tenzo::MulOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::MulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();
        Value lhs = adaptor.getLhs();
        Value rhs = adaptor.getRhs();

        Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);
        
        SmallVector<AffineMap, 3> maps(3, rewriter.getMultiDimIdentityMap(resultType.getRank()));
        SmallVector<utils::IteratorType, 3> iters(resultType.getRank(), utils::IteratorType::parallel);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op, resultType, ValueRange{lhs, rhs}, ValueRange{emptyTensor},
            maps, iters,
            [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
                Value mul = b.create<arith::MulFOp>(nestedLoc, args[0], args[1]);
                b.create<linalg::YieldOp>(nestedLoc, mul);
            }
        );
        return success();
    }
};
"""

with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

if "struct Relu2LoweringToLinalg" not in content:
    content = content.replace('//===----------------------------------------------------------------------===//\n// tenzo.conv2d', cxx_code + '\n//===----------------------------------------------------------------------===//\n// tenzo.conv2d')

if "patterns.add<Relu2LoweringToLinalg>" not in content:
    content = content.replace(
        "patterns.add<MatMulLoweringToLinalg>(patterns.getContext());",
        "patterns.add<MatMulLoweringToLinalg>(patterns.getContext());\n    patterns.add<Relu2LoweringToLinalg>(patterns.getContext());\n    patterns.add<MulLoweringToLinalg>(patterns.getContext());"
    )

with open("src/passes/LinalgLowering.cpp", "w") as f:
    f.write(content)

print("Injected Relu2 and Mul ops")
