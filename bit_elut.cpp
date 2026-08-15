struct BitLinearElutLoweringToLinalg
    : public OpConversionPattern<tenzo::BitLinearElutOp> {
    using OpConversionPattern<tenzo::BitLinearElutOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::BitLinearElutOp op, OpAdaptor adaptor,
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
        int64_t Kpack = wType.getShape()[1];
        if (K % 4 != 0 || Kpack * 4 != K || wType.getShape()[0] != N)
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
        Value fourIdx = rewriter.create<arith::ConstantIndexOp>(loc, 4);

        Value ubN = rewriter.create<arith::ConstantIndexOp>(loc, N);
        Value ubKp = rewriter.create<arith::ConstantIndexOp>(loc, Kpack);

        // We process 8 bytes (32 weights) per inner loop iteration
        int64_t vectorBytes = 8;
        Value vectorBytesVal = rewriter.create<arith::ConstantIndexOp>(loc, vectorBytes);

        // Collapse activation from 3D [1, 1, K] to 1D [K] to eliminate 3D stride math
        auto act1DType = RankedTensorType::get({K}, elemType);
        Value act1D = rewriter.create<tensor::CollapseShapeOp>(
            loc, act1DType, act, SmallVector<ReassociationIndices>{{0, 1, 2}});

        // Outer loop over output features N
        Value out = rewriter.create<scf::ForOp>(
            loc, zeroIdx, ubN, oneIdx, ValueRange{initOut},
            [&](OpBuilder &b, Location l, Value nIV, ValueRange iterN) {
                Value accTensor = iterN[0];

                // Inner reduction over Kpack bytes using Vector dialect
                Value sum = b.create<scf::ForOp>(
                    l, zeroIdx, ubKp, vectorBytesVal, ValueRange{zeroF},
                    [&](OpBuilder &bb, Location bl, Value kpIV, ValueRange iterKp) {
                        Value sAcc = iterKp[0];

                        // 1. Vector load 8 bytes from weights tensor
                        auto vector8i8 = VectorType::get({vectorBytes}, bb.getI8Type());
                        Value padI8 = bb.create<arith::ConstantOp>(bl, bb.getIntegerAttr(bb.getI8Type(), 0));
                        Value wVec = bb.create<vector::TransferReadOp>(bl, vector8i8, weights, ValueRange{nIV, kpIV}, padI8);

                        // 2. Vector unpacking
                        auto vector8i32 = VectorType::get({vectorBytes}, bb.getI32Type());
                        Value wVecI32 = bb.create<arith::ExtUIOp>(bl, vector8i32, wVec);

                        Value mask3 = bb.create<arith::ConstantOp>(bl, DenseElementsAttr::get(vector8i32, bb.getI32IntegerAttr(3)));
                        Value sh2 = bb.create<arith::ConstantOp>(bl, DenseElementsAttr::get(vector8i32, bb.getI32IntegerAttr(2)));
                        Value sh4 = bb.create<arith::ConstantOp>(bl, DenseElementsAttr::get(vector8i32, bb.getI32IntegerAttr(4)));
                        Value sh6 = bb.create<arith::ConstantOp>(bl, DenseElementsAttr::get(vector8i32, bb.getI32IntegerAttr(6)));

                        Value c0 = bb.create<arith::AndIOp>(bl, wVecI32, mask3);
                        Value c1 = bb.create<arith::AndIOp>(bl, bb.create<arith::ShRUIOp>(bl, wVecI32, sh2), mask3);
                        Value c2 = bb.create<arith::AndIOp>(bl, bb.create<arith::ShRUIOp>(bl, wVecI32, sh4), mask3);
                        Value c3 = bb.create<arith::AndIOp>(bl, bb.create<arith::ShRUIOp>(bl, wVecI32, sh6), mask3);

                        auto vector8f32 = VectorType::get({vectorBytes}, bb.getF32Type());
                        Value oneFVec = bb.create<arith::ConstantOp>(bl, DenseElementsAttr::get(vector8f32, bb.getF32FloatAttr(1.0f)));

                        auto toFloat = [&](Value c) {
                            Value f = bb.create<arith::SIToFPOp>(bl, vector8f32, c);
                            return bb.create<arith::SubFOp>(bl, f, oneFVec);
                        };

                        Value w0f = toFloat(c0);
                        Value w1f = toFloat(c1);
                        Value w2f = toFloat(c2);
                        Value w3f = toFloat(c3);

                        // Shuffle into proper order
                        SmallVector<int64_t> idx1(16);
                        for(int i=0; i<8; i++) { idx1[i*2] = i; idx1[i*2+1] = i+8; }
                        Value w01 = bb.create<vector::ShuffleOp>(bl, w0f, w1f, idx1);
                        Value w23 = bb.create<vector::ShuffleOp>(bl, w2f, w3f, idx1);

                        SmallVector<int64_t> idx2(32);
                        for(int i=0; i<16; i++) { idx2[i*2] = i; idx2[i*2+1] = i+16; }
                        Value w0123 = bb.create<vector::ShuffleOp>(bl, w01, w23, idx2);

                        // 3. Vector Math
                        auto vector32f32 = VectorType::get({32}, bb.getF32Type());
                        Value padF32 = bb.create<arith::ConstantOp>(bl, bb.getF32FloatAttr(0.0f));
                        Value actIdx = bb.create<arith::MulIOp>(bl, kpIV, fourIdx);
                        Value actVec = bb.create<vector::TransferReadOp>(bl, vector32f32, act1D, ValueRange{actIdx}, padF32);

                        Value mulVec = bb.create<arith::MulFOp>(bl, actVec, w0123);
                        Value blockSum = bb.create<vector::ReductionOp>(bl, vector::CombiningKind::ADD, mulVec);

                        Value newSAcc = bb.create<arith::AddFOp>(bl, sAcc, blockSum);
                        bb.create<scf::YieldOp>(bl, ValueRange{newSAcc});
                    }).getResult(0);

                // Scale result and insert into output tensor
                Value scaled = b.create<arith::MulFOp>(l, sum, scaleF);
                Value newAccTensor = b.create<tensor::InsertOp>(l, scaled, accTensor, ValueRange{zeroIdx, zeroIdx, nIV});
                b.create<scf::YieldOp>(l, ValueRange{newAccTensor});
            }).getResult(0);

        rewriter.replaceOp(op, out);
        return success();
    }
};