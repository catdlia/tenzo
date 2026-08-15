struct RopeLoweringToLinalg : public OpConversionPattern<tenzo::RopeOp> {
    using OpConversionPattern<tenzo::RopeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::RopeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        Value input = adaptor.getInput();
        Value seqPos = adaptor.getSeqPos();

        auto inType = mlir::cast<RankedTensorType>(input.getType());
        auto elemType = inType.getElementType();
        int rank = inType.getRank();
        int64_t D = inType.getDimSize(rank - 1);
        int64_t halfD = D / 2;

        Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value seqPosI32;
        if (auto spType = mlir::dyn_cast<RankedTensorType>(seqPos.getType())) {
            SmallVector<Value> indices(spType.getRank(), zeroIdx);
            seqPosI32 = rewriter.create<tensor::ExtractOp>(loc, seqPos, indices);
        } else {
            seqPosI32 = seqPos;
        }
        Value seqPosF = rewriter.create<arith::SIToFPOp>(loc, elemType, seqPosI32);

        Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, inType.getShape(), elemType);

        SmallVector<AffineMap, 2> maps(2, rewriter.getMultiDimIdentityMap(rank));
        SmallVector<utils::IteratorType, 4> iterators(rank, utils::IteratorType::parallel);

        Value halfD_Val = rewriter.create<arith::ConstantIndexOp>(loc, halfD);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op, inType,
            ValueRange{input}, ValueRange{emptyOut},
            maps, iterators,
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value curD = b.create<linalg::IndexOp>(l, rank - 1);

                Value isFirstHalf = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::slt, curD, halfD_Val);
                Value partnerD = b.create<arith::SelectOp>(
                    l, isFirstHalf,
                    b.create<arith::AddIOp>(l, curD, halfD_Val),
                    b.create<arith::SubIOp>(l, curD, halfD_Val)
                );
                Value freqIdx = b.create<arith::SelectOp>(
                    l, isFirstHalf, curD, b.create<arith::SubIOp>(l, curD, halfD_Val)
                );

                // Build index array for extracting partner element
                SmallVector<Value> partnerIndices;
                for (int i = 0; i < rank - 1; ++i) {
                    partnerIndices.push_back(b.create<linalg::IndexOp>(l, i));
                }
                partnerIndices.push_back(partnerD);

                Value partnerVal = b.create<tensor::ExtractOp>(l, input, partnerIndices);

                // theta_i = 10000^(-2 * freqIdx / D)
                Value freqIdxF = b.create<arith::IndexCastOp>(l, b.getI64Type(), freqIdx);
                Value freqIdxFloat = b.create<arith::SIToFPOp>(l, elemType, freqIdxF);
                Value factor = b.create<arith::ConstantOp>(l, rewriter.getFloatAttr(elemType, -2.0 / static_cast<double>(D)));
                Value exponent = b.create<arith::MulFOp>(l, freqIdxFloat, factor);
                
                // pow(10000, exponent) = exp(exponent * ln(10000))
                Value lnBase = b.create<arith::ConstantOp>(l, rewriter.getFloatAttr(elemType, std::log(10000.0)));
                Value expArg = b.create<arith::MulFOp>(l, exponent, lnBase);
                Value invFreq = b.create<math::ExpOp>(l, expArg);

                Value angle = b.create<arith::MulFOp>(l, seqPosF, invFreq);
                Value cosVal = b.create<math::CosOp>(l, angle);
                Value sinVal = b.create<math::SinOp>(l, angle);

                // sign: -1.0 for first half, +1.0 for second half
                Value negOne = b.create<arith::ConstantOp>(l, rewriter.getFloatAttr(elemType, -1.0));
                Value posOne = b.create<arith::ConstantOp>(l, rewriter.getFloatAttr(elemType, 1.0));
                Value sign = b.create<arith::SelectOp>(l, isFirstHalf, negOne, posOne);

                Value term1 = b.create<arith::MulFOp>(l, args[0], cosVal);
                Value signedPartner = b.create<arith::MulFOp>(l, partnerVal, sign);
                Value term2 = b.create<arith::MulFOp>(l, signedPartner, sinVal);
                Value res = b.create<arith::AddFOp>(l, term1, term2);

                b.create<linalg::YieldOp>(l, res);
            }
        );
        return success();
    }
};

//===----------------------------------------------------------------------===//
// tenzo.attention -> Scaled Dot-Product Attention (Q @ K^T -> Softmax -> @ V)
//===----------------------------------------------------------------------===//

