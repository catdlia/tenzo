        b.create<linalg::YieldOp>(l, embVal);
            }
        );
        return success();
    }
};

struct BitLinearTL1LoweringToLinalg
    : public OpConversionPattern<tenzo::BitLinearTL1Op> {
    using OpConversionPattern<tenzo::BitLinearTL1Op>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::BitLinearTL1Op op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        Value act = adaptor.getInput();
        Value weights = adaptor.getWeights();
        auto actType = mlir::cast<RankedTensorType>(act.getType());
        auto wType = mlir::cast<RankedTensorType>(weights.getType());
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();

        if (actType.getRank() != 3 || actType.getShape()[0] != 1 ||
            actType.getShape()[1] != 1)
            return failure();
        if (wType.getRank() != 2 || !wType.getElementType().isInteger(8))
            return failure();
        if (resultType.getRank() != 3 || resultType.getShape()[0] != 1 ||
            resultType.getShape()[1] != 1)
            return failure();

        int64_t K = actType.getShape()[2];
        int64_t N = resultType.getShape()[2];
        int64_t Kpack = wType.getShape()[0]; // K/2
        int64_t Npack = wType.getShape()[1]; // N/2
        if (K % 2 != 0 || Kpack * 2 != K || N % 2 != 0 || Npack * 2 != N)
            return failure();

        // Resolve FP32 scale
        Value scaleF;
        if (op.getScale()) {
            scaleF = adaptor.getScale();
            if (auto sType = mlir::dyn_cast<RankedTensorType>(scaleF.getType())) {
                Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
                SmallVector<Value> sIdxs(sType.getRank(), zeroIdx);
                scaleF = rewriter.create<tensor::ExtractOp>(loc, scaleF, sIdxs);
            }
        } else {
            scaleF = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getFloatAttr(elemType, 1.0));
        }

        // Initialize output tensor [1, 1, N]
        Value zeroF = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0));
        Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);
        Value initOut = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF}, ValueRange{emptyOut}).getResult(0);

        Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        Value ubN = rewriter.create<arith::ConstantIndexOp>(loc, N);
        Value ubKp = rewriter.create<arith::ConstantIndexOp>(loc, Kpack);

        // Collapse activation from 3D [1, 1, K] to 1D [K]
        auto act1DType = RankedTensorType::get({K}, elemType);
        Value act1D = rewriter.create<tensor::CollapseShapeOp>(
            loc, act1DType, act, SmallVector<ReassociationIndices>{{0, 1, 2}});

        // Dynamic Activation Quantization to i16 (same exact lossless setup as I2_S)
        auto vector128f32 = VectorType::get({128}, rewriter.getF32Type());
        Value zero128f32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vector128f32, rewriter.getF32FloatAttr(0.0f)));
        Value ubK = rewriter.create<arith::ConstantIndexOp>(loc, K);
        Value c128Idx = rewriter.create<arith::ConstantIndexOp>(loc, 128);

        Value maxVec = rewriter.create<scf::ForOp>(
            loc, zeroIdx, ubK, c128Idx, ValueRange{zero128f32},
            [&](OpBuilder &b, Location l, Value kIV, ValueRange iterK) {
                Value padF32 = b.create<arith::ConstantOp>(l, b.getF32FloatAttr(0.0f));
                Value v = b.create<vector::TransferReadOp>(l, vector128f32, act1D, ValueRange{kIV}, padF32);
                Value absV = b.create<math::AbsFOp>(l, v);
                Value newMax = b.create<arith::MaximumFOp>(l, absV, iterK[0]);
                b.create<scf::YieldOp>(l, ValueRange{newMax});
            }).getResult(0);

        Value maxAbs = rewriter.create<vector::ReductionOp>(loc, vector::CombiningKind::MAXIMUMF, maxVec);
        Value eps = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 1e-5f));
        Value safeMaxAbs = rewriter.create<arith::MaximumFOp>(loc, maxAbs, eps);

        Value c127F = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 127.0f));
        Value scaleAct = rewriter.create<arith::DivFOp>(loc, c127F, safeMaxAbs);
        Value invScaleAct = rewriter.create<arith::DivFOp>(loc, safeMaxAbs, c127F);
        Value totalScale = rewriter.create<arith::MulFOp>(loc, invScaleAct, scaleF);

        auto act1DQuantType = RankedTensorType::get({K}, rewriter.getI16Type());
        Value emptyQuant = rewriter.create<tensor::EmptyOp>(loc, act1DQuantType.getShape(), rewriter.getI16Type());

        auto vector128i16 = VectorType::get({128}, rewriter.getI16Type());
        Value scaleActVec = rewriter.create<vector::SplatOp>(loc, vector128f32, scaleAct);

        Value act1D_q = rewriter.create<scf::ForOp>(
            loc, zeroIdx, ubK, c128Idx, ValueRange{emptyQuant},
            [&](OpBuilder &b, Location l, Value kIV, ValueRange iterK) {
                Value padF32 = b.create<arith::ConstantOp>(l, b.getF32FloatAttr(0.0f));
                Value vF32 = b.create<vector::TransferReadOp>(l, vector128f32, act1D, ValueRange{kIV}, padF32);
                Value vScaled = b.create<arith::MulFOp>(l, vF32, scaleActVec);
                auto vector128i32 = VectorType::get({128}, b.getI32Type());
                Value vI32 = b.create<arith::FPToSIOp>(l, vector128i32, vScaled);
                Value vI16 = b.create<arith::TruncIOp>(l, vector128i16, vI32);

                Value updatedTensor = b.create<vector::TransferWriteOp>(l, vI16, iterK[0], ValueRange{kIV}).getResult();
                b.create<scf::YieldOp>(l, ValueRange{updatedTensor});
            }).getResult(0);

        // Outer-product loop over N by tiles of 128 channels (64 bytes in packed weights)
        Value c128IdxN = rewriter.create<arith::ConstantIndexOp>(loc, 128);
        auto outerFor = rewriter.create<scf::ForOp>(
            loc, zeroIdx, ubN, c128IdxN, ValueRange{initOut},
            [&](OpBuilder &b, Location l, Value nIV, ValueRange iterN) {
                // Initialize 128 accumulators as 4x vector<32xi32>
                auto vec32i32 = VectorType::get({32}, b.getI32Type());
                Value zeroVec32i32 = b.create<arith::ConstantOp>(l, DenseElementsAttr::get(vec32i32, b.getI32IntegerAttr(0)));
                
                Value acc0 = zeroVec32i32;
                Value acc1 = zeroVec32i32;
                Value acc2 = zeroVec32i32;
                Value acc3 = zeroVec32i32;

                // Inner reduction loop over K/2 (k_pair_idx)
                Value twoIdx = b.create<arith::ConstantIndexOp>(l, 2);
                auto kForResults = b.create<scf::ForOp>(
                    l, zeroIdx, ubKp, oneIdx, ValueRange{acc0, acc1, acc2, acc3},
                    [&](OpBuilder &b2, Location l2, Value kpIV, ValueRange iterAcc) {
                        // 1. Read pair of quantized activations: a0 = act[2*kpIV], a1 = act[2*kpIV + 1]
                        Value k0Idx = b2.create<arith::MulIOp>(l2, kpIV, twoIdx);
                        Value k1Idx = b2.create<arith::AddIOp>(l2, k0Idx, oneIdx);

                        Value a0 = b2.create<tensor::ExtractOp>(l2, act1D_q, ValueRange{k0Idx});
                        Value a1 = b2.create<tensor::ExtractOp>(l2, act1D_q, ValueRange{k1Idx});

                        // 2. Precompute 9 LUT sum values (i16) for this pair
                        Value zeroI16 = b2.create<arith::ConstantOp>(l2, b2.getI16IntegerAttr(0));
                        Value negA0 = b2.create<arith::SubIOp>(l2, zeroI16, a0);
                        Value negA1 = b2.create<arith::SubIOp>(l2, zeroI16, a1);

                        Value s0 = b2.create<arith::AddIOp>(l2, negA0, negA1);
                        Value s1 = negA0;
                        Value s2 = b2.create<arith::AddIOp>(l2, negA0, a1);
                        Value s3 = negA1;
                        Value s4 = zeroI16;
                        Value s5 = a1;
                        Value s6 = b2.create<arith::AddIOp>(l2, a0, negA1);
                        Value s7 = a0;
                        Value s8 = b2.create<arith::AddIOp>(l2, a0, a1);

                        SmallVector<Value> lut16_i16 = {s0, s1, s2, s3, s4, s5, s6, s7, s8, zeroI16, zeroI16, zeroI16, zeroI16, zeroI16, zeroI16, zeroI16};
                        
                        // Load 64 packed weight bytes (128 N channels) for current kpIV
                        Value npIV = b2.create<arith::DivUIOp>(l2, nIV, twoIdx);
                        auto vec64i8 = VectorType::get({64}, b2.getI8Type());
                        Value padI8 = b2.create<arith::ConstantOp>(l2, b2.getI8IntegerAttr(0));
                        Value wVec64 = b2.create<vector::TransferReadOp>(l2, vec64i8, weights, ValueRange{kpIV, npIV}, padI8);

                        Value curAcc0 = iterAcc[0];
                        Value curAcc1 = iterAcc[1];
                        Value curAcc2 = iterAcc[2];
                        Value curAcc3 = iterAcc[3];

                        for (int i = 0; i < 128; ++i) {
                            int byteIdx = i / 2;
                            int nibblePos = (i % 2) * 4;

                            Value bIdxVal = b2.create<arith::ConstantIndexOp>(l2, byteIdx);
                            Value wByte = b2.create<vector::ExtractElementOp>(l2, wVec64, bIdxVal);
                            
                            Value nibbleMask = b2.create<arith::ConstantOp>(l2, b2.getI8IntegerAttr(0x0F));
                            Value shiftVal = b2.create<arith::ConstantOp>(l2, b2.getI8IntegerAttr(nibblePos));
                            Value shifted = b2.create<arith::ShRUIOp>(l2, wByte, shiftVal);
                            Value idxI8 = b2.create<arith::AndIOp>(l2, shifted, nibbleMask);
                            Value idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), idxI8);

                            Value lutVal = zeroI16;
                            for (int lutIdx = 0; lutIdx < 9; ++lutIdx) {
                                Value matchC = b2.create<arith::ConstantIndexOp>(l2, lutIdx);
                                Value isEq = b2.create<arith::CmpIOp>(l2, arith::CmpIPredicate::eq, idx, matchC);
                                lutVal = b2.create<arith::SelectOp>(l2, isEq, lut16_i16[lutIdx], lutVal);
                            }

                            Value lutI32 = b2.create<arith::ExtSIOp>(l2, b2.getI32Type(), lutVal);

                            int accIdx = i / 32;
                            int subIdx = i % 32;
                            Value subIdxVal = b2.create<arith::ConstantIndexOp>(l2, subIdx);

                            if (accIdx == 0) curAcc0 = b2.create<vector::InsertElementOp>(l2, b2.create<arith::AddIOp>(l2, b2.create<vector::ExtractElementOp>(l2, curAcc0, subIdxVal), lutI32), curAcc0, subIdxVal);
                            else if (accIdx == 1) curAcc1 = b2.create<vector::InsertElementOp>(l2, b2.create<arith::AddIOp>(l2, b2.create<vector::ExtractElementOp>(l2, curAcc1, subIdxVal), lutI32), curAcc1, subIdxVal);
                            else if (accIdx == 2) curAcc2 = b2.create<vector::InsertElementOp>(l2, b2.create<arith::AddIOp>(l2, b2.create<vector::ExtractElementOp>(l2, curAcc2, subIdxVal), lutI32), curAcc2, subIdxVal);
                            else curAcc3 = b2.create<vector::InsertElementOp>(l2, b2.create<arith::AddIOp>(l2, b2.create<vector::ExtractElementOp>(l2, curAcc3, subIdxVal), lutI32), curAcc3, subIdxVal);
                        }

                        b2.create<scf::YieldOp>(l2, ValueRange{curAcc0, curAcc1, curAcc2, curAcc3});
                    }).getResults();

                Value finalAcc0 = kForResults[0];
                Value finalAcc1 = kForResults[1];
                Value finalAcc2 = kForResults[2];
                Value finalAcc3 = kForResults[3];

                auto vec32f32 = VectorType::get({32}, b.getF32Type());
                Value totalScaleVec = b.create<vector::SplatOp>(l, vec32f32, totalScale);

                Value outTensor = iterN[0];
                for (int c = 0; c < 4; ++c) {
                    Value accVec = (c == 0) ? finalAcc0 : (c == 1) ? finalAcc1 : (c == 2) ? finalAcc2 : finalAcc3;
                    Value fVec = b.create<arith::SIToFPOp>(l, vec32f32, accVec);
                    Value scaledF = b.create<arith::MulFOp>(l, fVec, totalScaleVec);

                    Value offsetN = b.create<arith::AddIOp>(l, nIV, b.create<arith::ConstantIndexOp>(l, c * 32));
                    Value zeroIdxLocal = b.create<arith::ConstantIndexOp>(l, 0);
                    outTensor = b.create<vector::TransferWriteOp>(
                        l, scaledF, outTensor, ValueRange{zeroIdxLocal, zeroIdxLocal, offsetN}).getResult();
                }

                b.create<scf::YieldOp>(l, ValueRange{outTensor});
            });

        rewriter.replaceOp(op, outerFor.getResult(0));
        return success();
    }
};

} // namespace

void tenzo::populateTenzoToLinalgConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<FusedLoweringToLinalg>(patterns.getContext());
    patterns.add<AddLoweringToLinalg>(patterns.getContext());
    patterns.add<ReluLoweringToLinalg>(patterns.getContext());
    patterns.add<SiLuLoweringToLinalg>(patterns.getContext());
    patterns.add<Relu2LoweringToLinalg>(patterns.getContext());
    patterns.add<MulLoweringToLinalg>(patterns.getContext());
    patterns.add<AttentionLoweringToLinalg>(patterns.getContext());
    patterns.add<RopeLoweringToLinalg>(patterns.getContext());
    patterns.add<MatMulLoweringToLinalg>(patterns.getContext());
    patterns.add<Conv2DLoweringToLinalg>(patterns.getContext());
    patterns.add<QuantizeLoweringToLinalg>(patterns.getContext());
    patterns.add<DequantizeLoweringToLinalg>(patterns.getContext());
    patterns.add<BitLinearElutLoweringToLinalg>(patterns.getContext());
    patterns.add<BitLinearTL1LoweringToLinalg>(patterns.getContext());
    patterns.add<KVCacheUpdateLowering>(patterns.getContext());
    patterns.add<RMSNormLoweringToLinalg>(patterns.getContext());
    patterns.add<EmbeddingLoweringToLinalg>(patterns.getContext());
}