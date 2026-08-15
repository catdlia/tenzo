struct AttentionLoweringToLinalg : public OpConversionPattern<tenzo::AttentionOp> {
    using OpConversionPattern<tenzo::AttentionOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::AttentionOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        Value query = adaptor.getQuery();
        Value keyCache = adaptor.getKey();
        Value valCache = adaptor.getValue();
        Value seqPos = adaptor.getSeqPos();

        auto queryType = mlir::cast<RankedTensorType>(query.getType());
        auto keyType   = mlir::cast<RankedTensorType>(keyCache.getType());
        auto elemType  = queryType.getElementType();
        int rank       = queryType.getRank();

        Value zeroF = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));
        Value negInfF = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, -1e9));

        int64_t D = queryType.getDimSize(rank - 1);
        double scaleVal = 1.0 / std::sqrt(static_cast<double>(D));
        Value scaleF = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, scaleVal));

        // Extract seqPos index
        Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value seqPosI32;
        if (auto spType = mlir::dyn_cast<RankedTensorType>(seqPos.getType())) {
            SmallVector<Value> indices(spType.getRank(), zeroIdx);
            seqPosI32 = rewriter.create<tensor::ExtractOp>(loc, seqPos, indices);
        } else {
            seqPosI32 = seqPos;
        }
        Value seqPosIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), seqPosI32);

        if (rank == 4) {
            int64_t B = queryType.getDimSize(0);
            int64_t H = queryType.getDimSize(1);
            int64_t MaxSeq = keyType.getDimSize(2);
            int64_t Dh = queryType.getDimSize(3);

            int64_t H_kv = keyType.getDimSize(1);
            int64_t gqaRatio = (H_kv > 0) ? (H / H_kv) : 1;

            // --- Step 1: Compute Q @ K^T -> scores [B, H, MaxSeq] ---
            RankedTensorType scoresType = RankedTensorType::get({B, H, MaxSeq}, elemType);
            Value emptyScores = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{B, H, MaxSeq}, elemType);
            Value filledScores = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF}, ValueRange{emptyScores}).getResult(0);

            auto d0 = rewriter.getAffineDimExpr(0);
            auto d1 = rewriter.getAffineDimExpr(1);
            auto d2 = rewriter.getAffineDimExpr(2);
            auto d3 = rewriter.getAffineDimExpr(3);

            AffineExpr kvHeadExpr = (gqaRatio > 1) ? d1.floorDiv(gqaRatio) : d1;

            AffineMap qMap = AffineMap::get(4, 0, {d0, d1, rewriter.getAffineConstantExpr(0), d3}, rewriter.getContext());
            AffineMap kMap = AffineMap::get(4, 0, {d0, kvHeadExpr, d2, d3}, rewriter.getContext());
            AffineMap scoresMap = AffineMap::get(4, 0, {d0, d1, d2}, rewriter.getContext());

            SmallVector<utils::IteratorType, 4> qkIterators = {
                utils::IteratorType::parallel,
                utils::IteratorType::parallel,
                utils::IteratorType::parallel,
                utils::IteratorType::reduction
            };

            Value rawScores = rewriter.create<linalg::GenericOp>(
                loc, scoresType,
                ValueRange{query, keyCache}, ValueRange{filledScores},
                ArrayRef<AffineMap>{qMap, kMap, scoresMap},
                qkIterators,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    Value mul = b.create<arith::MulFOp>(l, args[0], args[1]);
                    Value add = b.create<arith::AddFOp>(l, args[2], mul);
                    b.create<linalg::YieldOp>(l, add);
                }
            ).getResult(0);

            // --- Step 2: Causal Masking & Safe Softmax over [B, H, MaxSeq] ---
            RankedTensorType scalarType = RankedTensorType::get({B, H}, elemType);
            Value emptyMax = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{B, H}, elemType);
            Value filledMax = rewriter.create<linalg::FillOp>(loc, ValueRange{negInfF}, ValueRange{emptyMax}).getResult(0);

            AffineMap scoreReduceMap = AffineMap::get(3, 0, {d0, d1, d2}, rewriter.getContext());
            AffineMap maxOutMap = AffineMap::get(3, 0, {d0, d1}, rewriter.getContext());

            Value maxScores = rewriter.create<linalg::GenericOp>(
                loc, scalarType,
                ValueRange{rawScores}, ValueRange{filledMax},
                ArrayRef<AffineMap>{scoreReduceMap, maxOutMap},
                SmallVector<utils::IteratorType, 3>{utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction},
                [&](OpBuilder &b, Location l, ValueRange args) {
                    Value curS = b.create<linalg::IndexOp>(l, 2);
                    Value isvalid = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sle, curS, seqPosIdx);
                    Value scaledScore = b.create<arith::MulFOp>(l, args[0], scaleF);
                    Value valToUse = b.create<arith::SelectOp>(l, isvalid, scaledScore, negInfF);
                    Value maxVal = b.create<arith::MaximumFOp>(l, args[1], valToUse);
                    b.create<linalg::YieldOp>(l, maxVal);
                }
            ).getResult(0);

            Value emptySum = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{B, H}, elemType);
            Value filledSum = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF}, ValueRange{emptySum}).getResult(0);

            Value sumExp = rewriter.create<linalg::GenericOp>(
                loc, scalarType,
                ValueRange{rawScores, maxScores}, ValueRange{filledSum},
                ArrayRef<AffineMap>{scoreReduceMap, maxOutMap, maxOutMap},
                SmallVector<utils::IteratorType, 3>{utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction},
                [&](OpBuilder &b, Location l, ValueRange args) {
                    Value curS = b.create<linalg::IndexOp>(l, 2);
                    Value isvalid = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sle, curS, seqPosIdx);
                    Value scaledScore = b.create<arith::MulFOp>(l, args[0], scaleF);
                    Value diff = b.create<arith::SubFOp>(l, scaledScore, args[1]);
                    Value expVal = b.create<math::ExpOp>(l, diff);
                    Value validExp = b.create<arith::SelectOp>(l, isvalid, expVal, zeroF);
                    Value acc = b.create<arith::AddFOp>(l, args[2], validExp);
                    b.create<linalg::YieldOp>(l, acc);
                }
            ).getResult(0);

            Value probs = rewriter.create<linalg::GenericOp>(
                loc, scoresType,
                ValueRange{rawScores, maxScores, sumExp}, ValueRange{filledScores},
                ArrayRef<AffineMap>{scoreReduceMap, maxOutMap, maxOutMap, scoreReduceMap},
                SmallVector<utils::IteratorType, 3>{utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel},
                [&](OpBuilder &b, Location l, ValueRange args) {
                    Value curS = b.create<linalg::IndexOp>(l, 2);
                    Value isvalid = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sle, curS, seqPosIdx);
                    Value scaledScore = b.create<arith::MulFOp>(l, args[0], scaleF);
                    Value diff = b.create<arith::SubFOp>(l, scaledScore, args[1]);
                    Value expVal = b.create<math::ExpOp>(l, diff);
                    Value prob = b.create<arith::DivFOp>(l, expVal, args[2]);
                    Value validProb = b.create<arith::SelectOp>(l, isvalid, prob, zeroF);
                    b.create<linalg::YieldOp>(l, validProb);
                }
            ).getResult(0);

            // --- Step 3: Compute probs @ V -> out [B, H, 1, Dh] ---
            auto outType = mlir::cast<RankedTensorType>(op.getResult().getType());
            Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{B, H, 1, Dh}, elemType);
            Value filledOut = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF}, ValueRange{emptyOut}).getResult(0);

            AffineMap probMap = AffineMap::get(4, 0, {d0, d1, d3}, rewriter.getContext());
            AffineMap vMap = AffineMap::get(4, 0, {d0, kvHeadExpr, d3, d2}, rewriter.getContext());
            AffineMap outMap = AffineMap::get(4, 0, {d0, d1, rewriter.getAffineConstantExpr(0), d2}, rewriter.getContext());

            SmallVector<utils::IteratorType, 4> pvIterators = {
                utils::IteratorType::parallel,
                utils::IteratorType::parallel,
                utils::IteratorType::parallel,
                utils::IteratorType::reduction
            };

            rewriter.replaceOpWithNewOp<linalg::GenericOp>(
                op, outType,
                ValueRange{probs, valCache}, ValueRange{filledOut},
                ArrayRef<AffineMap>{probMap, vMap, outMap},
                pvIterators,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    Value mul = b.create<arith::MulFOp>(l, args[0], args[1]);
                    Value add = b.create<arith::AddFOp>(l, args[2], mul);
                    b.create<linalg::YieldOp>(l, add);
                }
            );
            return success();
        }

        // --- 3D Tensor Fallback [B, 1, D] ---
        int64_t B = queryType.getDimSize(0);
        int64_t MaxSeq = keyType.getDimSize(1);

        RankedTensorType scoresType = RankedTensorType::get({B, MaxSeq}, elemType);
        Value emptyScores = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{B, MaxSeq}, elemType);
        Value filledScores = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF}, ValueRange{emptyScores}).getResult(0);

        auto d0 = rewriter.getAffineDimExpr(0);
        auto d1 = rewriter.getAffineDimExpr(1);
        auto d2 = rewriter.getAffineDimExpr(2);

        AffineMap qMap = AffineMap::get(3, 0, {d0, rewriter.getAffineConstantExpr(0), d2}, rewriter.getContext());
        AffineMap kMap = AffineMap::get(3, 0, {d0, d1, d2}, rewriter.getContext());
        AffineMap scoresMap = AffineMap::get(3, 0, {d0, d1}, rewriter.getContext());

        SmallVector<utils::IteratorType, 3> qkIterators = {
            utils::IteratorType::parallel,
            utils::IteratorType::parallel,
            utils::IteratorType::reduction
        };

        Value rawScores = rewriter.create<linalg::GenericOp>(
            loc, scoresType,
            ValueRange{query, keyCache}, ValueRange{filledScores},
            ArrayRef<AffineMap>{qMap, kMap, scoresMap},
            qkIterators,
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value mul = b.create<arith::MulFOp>(l, args[0], args[1]);
                Value add = b.create<arith::AddFOp>(l, args[2], mul);
                b.create<linalg::YieldOp>(l, add);
            }
        ).getResult(0);

        RankedTensorType scalarType = RankedTensorType::get({B}, elemType);
        Value emptyMax = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{B}, elemType);
        Value filledMax = rewriter.create<linalg::FillOp>(loc, ValueRange{negInfF}, ValueRange{emptyMax}).getResult(0);

        AffineMap scoreReduceMap = AffineMap::get(2, 0, {d0, d1}, rewriter.getContext());
        AffineMap maxOutMap = AffineMap::get(2, 0, {d0}, rewriter.getContext());

        Value maxScores = rewriter.create<linalg::GenericOp>(
            loc, scalarType,
            ValueRange{rawScores}, ValueRange{filledMax},
            ArrayRef<AffineMap>{scoreReduceMap, maxOutMap},
            SmallVector<utils::IteratorType, 2>{utils::IteratorType::parallel, utils::IteratorType::reduction},
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value curS = b.create<linalg::IndexOp>(l, 1);
                Value isvalid = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sle, curS, seqPosIdx);
                Value scaledScore = b.create<arith::MulFOp>(l, args[0], scaleF);
                Value valToUse = b.create<arith::SelectOp>(l, isvalid, scaledScore, negInfF);
                Value maxVal = b.create<arith::MaximumFOp>(l, args[1], valToUse);
                b.create<linalg::YieldOp>(l, maxVal);
            }
        ).getResult(0);

        Value emptySum = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{B}, elemType);
        Value filledSum = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF}, ValueRange{emptySum}).getResult(0);

        Value sumExp = rewriter.create<linalg::GenericOp>(
            loc, scalarType,
            ValueRange{rawScores, maxScores}, ValueRange{filledSum},
            ArrayRef<AffineMap>{scoreReduceMap, maxOutMap, maxOutMap},
            SmallVector<utils::IteratorType, 2>{utils::IteratorType::parallel, utils::IteratorType::reduction},
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value curS = b.create<linalg::IndexOp>(l, 1);
                Value isvalid = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sle, curS, seqPosIdx);
                Value scaledScore = b.create<arith::MulFOp>(l, args[0], scaleF);
                Value diff = b.create<arith::SubFOp>(l, scaledScore, args[1]);
                Value expVal = b.create<math::ExpOp>(l, diff);
                Value validExp = b.create<arith::SelectOp>(l, isvalid, expVal, zeroF);
                Value acc = b.create<arith::AddFOp>(l, args[2], validExp);
                b.create<linalg::YieldOp>(l, acc);
            }
        ).getResult(0);

        Value probs = rewriter.create<linalg::GenericOp>(
            loc, scoresType,
            ValueRange{rawScores, maxScores, sumExp}, ValueRange{filledScores},
            ArrayRef<AffineMap>{scoreReduceMap, maxOutMap, maxOutMap, scoreReduceMap},
            SmallVector<utils::IteratorType, 2>{utils::IteratorType::parallel, utils::IteratorType::parallel},
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value curS = b.create<linalg::IndexOp>(l, 1);
                Value isvalid = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sle, curS, seqPosIdx);
                Value scaledScore = b.create<arith::MulFOp>(l, args[0], scaleF);
                Value diff = b.create<arith::SubFOp>(l, scaledScore, args[1]);
                Value expVal = b.create<math::ExpOp>(l, diff);
                Value prob = b.create<arith::DivFOp>(l, expVal, args[2]);
                Value validProb = b.create<arith::SelectOp>(l, isvalid, prob, zeroF);
                b.create<linalg::YieldOp>(l, validProb);
            }
        ).getResult(0);

        auto outType = mlir::cast<RankedTensorType>(op.getResult().getType());
        Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{B, 1, D}, elemType);
        Value filledOut = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF}, ValueRange{emptyOut}).getResult(0);

        AffineMap probMap = AffineMap::get(3, 0, {d0, d2}, rewriter.getContext());
        AffineMap vMap = AffineMap::get(3, 0, {d0, d2, d1}, rewriter.getContext());
        AffineMap outMap = AffineMap::get(3, 0, {d0, rewriter.getAffineConstantExpr(0), d1}, rewriter.getContext());

        SmallVector<utils::IteratorType, 3> pvIterators = {
            utils::IteratorType::parallel,
            utils::IteratorType::parallel,
            utils::IteratorType::reduction
        };

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op, outType,
            ValueRange{probs, valCache}, ValueRange{filledOut},
            ArrayRef<AffineMap>{probMap, vMap, outMap},
            pvIterators,
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value mul = b.create<arith::MulFOp>(l, args[0], args[1]);
                Value add = b.create<arith::AddFOp>(l, args[2], mul);
                b.create<linalg::YieldOp>(l, add);
            }
        );
        return success();
    }
};

//===----------------------------------------------------------------------===//
// tenzo.matmul -> linalg.matmul
//===----------------------------------------------------------------------===//

