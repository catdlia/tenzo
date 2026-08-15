//===----------------------------------------------------------------------===//
// tenzo.silu -> SiLU/Swish activation: x * sigmoid(x)
//===----------------------------------------------------------------------===//
struct SiLuLoweringToLinalg : public OpConversionPattern<tenzo::SiLuOp> {
    using OpConversionPattern<tenzo::SiLuOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::SiLuOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();

        SmallVector<AffineMap, 2> indexingMaps(
            2, rewriter.getMultiDimIdentityMap(resultType.getRank()));

        SmallVector<utils::IteratorType, 1> iteratorTypes(
            resultType.getRank(), utils::IteratorType::parallel);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            resultType,
            ValueRange{adaptor.getInput()},
            ValueRange{rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType)},
            indexingMaps,
            iteratorTypes,
            [&](OpBuilder &b, Location l, ValueRange args) {
                // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                Value x = args[0];
                Value negX = b.create<arith::NegFOp>(l, x);
                Value expNegX = b.create<math::ExpOp>(l, negX);
                Value one = b.create<arith::ConstantOp>(l, b.getFloatAttr(elemType, 1.0));
                Value denom = b.create<arith::AddFOp>(l, one, expNegX);
                Value res = b.create<arith::DivFOp>(l, x, denom);
                b.create<linalg::YieldOp>(l, res);
            });
        return success();
    }
};

//===----------------------------------------------------------------------===//
// tenzo.mul -> Element-wise multiplication
//===----------------------------------------------------------------------===//
struct MulLoweringToLinalg : public OpConversionPattern<tenzo::MulOp> {
    using OpConversionPattern<tenzo::MulOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::MulOp op, OpAdaptor adaptor,
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
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value res = b.create<arith::MulFOp>(l, args[0], args[1]);
                b.create<linalg::YieldOp>(l, res);
            });
        return success();
    }
};

//===----------------------------------------------------------------------===//
// tenzo.rope -> Rotary Position Embedding
//===----------------------------------------------------------------------===//