#include "passes/Passes.h"
#include "dialect/TenzoDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <cmath>

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

struct AddLoweringToLinalg : public OpConversionPattern<tenzo::AddOp> {
    using OpConversionPattern<tenzo::AddOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::AddOp op, OpAdaptor adaptor,
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
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum.getResult());
            });
        return success();
    }
};

struct ReluLoweringToLinalg : public OpConversionPattern<tenzo::ReluOp> {
    using OpConversionPattern<tenzo::ReluOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::ReluOp op, OpAdaptor adaptor,
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
                auto zeroAttr = nestedBuilder.getF32FloatAttr(0.0f);
                auto zero = nestedBuilder.create<arith::ConstantOp>(nestedLoc, zeroAttr);
                auto res = nestedBuilder.create<arith::MaximumFOp>(nestedLoc, args[0], zero);
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, res.getResult());
            });
        return success();
    }
};

//===----------------------------------------------------------------------===//
// tenzo.rope -> Rotary Position Embedding
//===----------------------------------------------------------------------===//
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

            // --- Step 1: Compute Q @ K^T -> scores [B, H, MaxSeq] ---
            RankedTensorType scoresType = RankedTensorType::get({B, H, MaxSeq}, elemType);
            Value emptyScores = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{B, H, MaxSeq}, elemType);
            Value filledScores = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF}, ValueRange{emptyScores}).getResult(0);

            auto d0 = rewriter.getAffineDimExpr(0);
            auto d1 = rewriter.getAffineDimExpr(1);
            auto d2 = rewriter.getAffineDimExpr(2);
            auto d3 = rewriter.getAffineDimExpr(3);

            AffineMap qMap = AffineMap::get(4, 0, {d0, d1, rewriter.getAffineConstantExpr(0), d3}, rewriter.getContext());
            AffineMap kMap = AffineMap::get(4, 0, {d0, d1, d2, d3}, rewriter.getContext());
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
            AffineMap vMap = AffineMap::get(4, 0, {d0, d1, d3, d2}, rewriter.getContext());
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
struct MatMulLoweringToLinalg : public OpConversionPattern<tenzo::MatMulOp> {
    using OpConversionPattern<tenzo::MatMulOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::MatMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();
        auto lhsType = mlir::cast<RankedTensorType>(adaptor.getLhs().getType());
        auto rhsType = mlir::cast<RankedTensorType>(adaptor.getRhs().getType());

        // Create zero-initialized output tensor
        auto zero = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(elemType));
        SmallVector<Value> dynamicSizes;
        for (int i = 0; i < resultType.getRank(); ++i) {
            if (resultType.isDynamicDim(i)) {
                dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, adaptor.getLhs(), i));
            }
        }

        auto emptyTensor = rewriter.create<tensor::EmptyOp>(
            loc, resultType.getShape(), elemType, dynamicSizes);
        auto filledTensor = rewriter.create<linalg::FillOp>(
            loc, ValueRange{zero}, ValueRange{emptyTensor});

        if (lhsType.getRank() == 3 && rhsType.getRank() == 2) {
            // 3D x 2D MatMul: [B, S, K] x [N, K] -> [B, S, N]
            // Iterators: d0=B (parallel), d1=S (parallel), d2=N (parallel), d3=K (reduction)
            auto d0 = rewriter.getAffineDimExpr(0);
            auto d1 = rewriter.getAffineDimExpr(1);
            auto d2 = rewriter.getAffineDimExpr(2);
            auto d3 = rewriter.getAffineDimExpr(3);

            AffineMap lhsMap = AffineMap::get(4, 0, {d0, d1, d3}, rewriter.getContext());
            AffineMap rhsMap = AffineMap::get(4, 0, {d2, d3}, rewriter.getContext());
            AffineMap outMap = AffineMap::get(4, 0, {d0, d1, d2}, rewriter.getContext());

            SmallVector<utils::IteratorType, 4> iterators = {
                utils::IteratorType::parallel,
                utils::IteratorType::parallel,
                utils::IteratorType::parallel,
                utils::IteratorType::reduction
            };

            rewriter.replaceOpWithNewOp<linalg::GenericOp>(
                op,
                resultType,
                ValueRange{adaptor.getLhs(), adaptor.getRhs()},
                ValueRange{filledTensor.getResult(0)},
                ArrayRef<AffineMap>{lhsMap, rhsMap, outMap},
                iterators,
                [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
                    auto mul = b.create<arith::MulFOp>(nestedLoc, args[0], args[1]);
                    auto add = b.create<arith::AddFOp>(nestedLoc, args[2], mul);
                    b.create<linalg::YieldOp>(nestedLoc, add.getResult());
                }
            );
            return success();
        }

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
// tenzo.quantize -> linalg.generic (ternary: round(x/scale).clamp(-1,1)*scale)
//===----------------------------------------------------------------------===//
struct QuantizeLoweringToLinalg : public OpConversionPattern<tenzo::QuantizeOp> {
    using OpConversionPattern<tenzo::QuantizeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::QuantizeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        Value input = adaptor.getInput();
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();
        int64_t rank = resultType.getRank();

        SmallVector<Value> dynamicSizes;
        for (int i = 0; i < rank; ++i) {
            if (resultType.isDynamicDim(i)) {
                dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
            }
        }

        // --- Stage 1: compute sum(|x|) over all elements using a reduction ---
        // We reduce to a scalar tensor<f32> (rank-0 via linalg.generic with all-reduction iterators)
        Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));
        Value one  = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 1.0));
        Value neg1 = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, -1.0));
        Value eps  = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 1e-5));

        // Scalar output for sum and count
        RankedTensorType scalarType = RankedTensorType::get({}, elemType);
        Value scalarEmpty = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{}, elemType);
        Value scalarZero  = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{scalarEmpty}).getResult(0);

        // Build identity map for input (all dims), empty map for scalar output
        AffineMap inputMap = rewriter.getMultiDimIdentityMap(rank);
        AffineMap scalarMap = AffineMap::get(rank, 0, {}, rewriter.getContext());

        SmallVector<utils::IteratorType> allReduction(rank, utils::IteratorType::reduction);

        // sum_abs = reduce sum(|x|)
        Value sumAbs = rewriter.create<linalg::GenericOp>(
            loc, scalarType,
            ValueRange{input}, ValueRange{scalarZero},
            ArrayRef<AffineMap>{inputMap, scalarMap},
            allReduction,
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value absVal = b.create<math::AbsFOp>(l, args[0]);
                Value acc    = b.create<arith::AddFOp>(l, args[1], absVal);
                b.create<linalg::YieldOp>(l, acc);
            }
        ).getResult(0);

        // count = total number of elements (product of static dims)
        int64_t numElems = 1;
        for (int i = 0; i < rank; ++i) {
            numElems *= resultType.getDimSize(i);
        }
        Value countF = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, static_cast<double>(numElems)));

        // extract scalar from tensor
        Value sumAbsScalar = rewriter.create<tensor::ExtractOp>(loc, sumAbs, ValueRange{});

        // scale = max(sum_abs / count, eps)
        Value meanAbs  = rewriter.create<arith::DivFOp>(loc, sumAbsScalar, countF);
        Value scale    = rewriter.create<arith::MaximumFOp>(loc, meanAbs, eps);

        // --- Stage 2: elementwise: round(x / scale).clamp(-1,1) * scale ---
        Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType, dynamicSizes);
        Value filledOut = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyOut}).getResult(0);

        SmallVector<AffineMap, 2> identityMaps(2, inputMap);
        SmallVector<utils::IteratorType> allParallel(rank, utils::IteratorType::parallel);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op, resultType,
            ValueRange{input}, ValueRange{filledOut},
            identityMaps, allParallel,
            [&](OpBuilder &b, Location l, ValueRange args) {
                // x / scale
                Value divided  = b.create<arith::DivFOp>(l, args[0], scale);
                // round (nearest)
                Value rounded  = b.create<math::RoundOp>(l, divided);
                // clamp to [-1, 1]
                Value clamped  = b.create<arith::MaximumFOp>(l, rounded, neg1);
                clamped        = b.create<arith::MinimumFOp>(l, clamped,  one);
                // rescale
                Value rescaled = b.create<arith::MulFOp>(l, clamped, scale);
                b.create<linalg::YieldOp>(l, rescaled);
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
        Value input = adaptor.getInput();
        auto inType = mlir::cast<RankedTensorType>(input.getType());
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();
        int rank = resultType.getRank();

        Value scaleF;
        if (op.getScale()) {
            scaleF = adaptor.getScale();
            if (auto sType = mlir::dyn_cast<RankedTensorType>(scaleF.getType())) {
                Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
                SmallVector<Value> indices(sType.getRank(), zeroIdx);
                scaleF = rewriter.create<tensor::ExtractOp>(loc, scaleF, indices);
            }
        } else {
            scaleF = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 1.0));
        }

        if (inType.getElementType().isInteger(8)) {
            Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);

            SmallVector<AffineMap, 1> indexingMaps = {
                rewriter.getMultiDimIdentityMap(rank)
            };
            SmallVector<utils::IteratorType, 2> iterators(rank, utils::IteratorType::parallel);

            Value zeroF = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0));
            Value posOneF = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 1.0));
            Value negOneF = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, -1.0));
            Value fourVal = rewriter.create<arith::ConstantIndexOp>(loc, 4);
            Value twoVal = rewriter.create<arith::ConstantIndexOp>(loc, 2);
            Value threeI8 = rewriter.create<arith::ConstantIntOp>(loc, 3, 8);
            Value oneI8 = rewriter.create<arith::ConstantIntOp>(loc, 1, 8);
            Value twoI8 = rewriter.create<arith::ConstantIntOp>(loc, 2, 8);

            rewriter.replaceOpWithNewOp<linalg::GenericOp>(
                op, resultType,
                ValueRange{}, ValueRange{emptyOut},
                indexingMaps,
                iterators,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    Value rIdx = b.create<linalg::IndexOp>(l, 0);
                    Value cIdx = b.create<linalg::IndexOp>(l, 1);

                    Value byteIdx = b.create<arith::DivUIOp>(l, cIdx, fourVal);
                    Value remIdx = b.create<arith::RemUIOp>(l, cIdx, fourVal);
                    Value shiftIdx = b.create<arith::MulIOp>(l, remIdx, twoVal);
                    Value shiftI8 = b.create<arith::IndexCastOp>(l, b.getI8Type(), shiftIdx);

                    Value byteVal = b.create<tensor::ExtractOp>(l, input, ValueRange{rIdx, byteIdx});
                    Value shifted = b.create<arith::ShRUIOp>(l, byteVal, shiftI8);
                    Value code = b.create<arith::AndIOp>(l, shifted, threeI8);

                    Value isPos = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::eq, code, oneI8);
                    Value isNeg = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::eq, code, twoI8);

                    Value val = b.create<arith::SelectOp>(l, isPos, posOneF, zeroF);
                    val = b.create<arith::SelectOp>(l, isNeg, negOneF, val);

                    Value res = b.create<arith::MulFOp>(l, val, scaleF);
                    b.create<linalg::YieldOp>(l, res);
                }
            );
            return success();
        }

        SmallVector<AffineMap, 2> indexingMaps(2, rewriter.getMultiDimIdentityMap(rank));
        SmallVector<utils::IteratorType, 1> iteratorTypes(rank, utils::IteratorType::parallel);
        Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op, resultType,
            ValueRange{input}, ValueRange{emptyOut},
            indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location l, ValueRange args) {
                b.create<linalg::YieldOp>(l, args[0]);
            }
        );
        return success();
    }
};

//===----------------------------------------------------------------------===//
// tenzo.kv_cache_update -> tensor.insert_slice
//===----------------------------------------------------------------------===//
struct KVCacheUpdateLowering : public OpConversionPattern<tenzo::KVCacheUpdateOp> {
    using OpConversionPattern<tenzo::KVCacheUpdateOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::KVCacheUpdateOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        auto cacheKType = mlir::cast<RankedTensorType>(adaptor.getCacheK().getType());
        auto cacheVType = mlir::cast<RankedTensorType>(adaptor.getCacheV().getType());
        auto newKType = mlir::cast<RankedTensorType>(adaptor.getNewK().getType());
        auto newVType = mlir::cast<RankedTensorType>(adaptor.getNewV().getType());

        // Extract seq_pos (assumed tensor<1xi32> or similar)
        Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value seqPosI32;
        if (auto seqPosType = mlir::dyn_cast<RankedTensorType>(adaptor.getSeqPos().getType())) {
            SmallVector<Value> indices(seqPosType.getRank(), zeroIdx);
            seqPosI32 = rewriter.create<tensor::ExtractOp>(loc, adaptor.getSeqPos(), indices);
        } else {
            // Fallback if not a tensor
            seqPosI32 = adaptor.getSeqPos();
        }
        Value seqPosIndex = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), seqPosI32);

        auto createInsertSlice = [&](Value source, Value dest, RankedTensorType sourceType) -> Value {
            SmallVector<OpFoldResult> offsets, sizes, strides;
            int rank = sourceType.getRank();
            for (int i = 0; i < rank; ++i) {
                if (i == rank - 2) { // Sequence dimension is usually the second to last (e.g. batch, seq, dim)
                    offsets.push_back(seqPosIndex);
                } else {
                    offsets.push_back(rewriter.getIndexAttr(0));
                }
                
                if (sourceType.isDynamicDim(i)) {
                    sizes.push_back(rewriter.create<tensor::DimOp>(loc, source, i).getResult());
                } else {
                    sizes.push_back(rewriter.getIndexAttr(sourceType.getDimSize(i)));
                }
                
                strides.push_back(rewriter.getIndexAttr(1));
            }
            return rewriter.create<tensor::InsertSliceOp>(loc, source, dest, offsets, sizes, strides);
        };

        Value updatedK = createInsertSlice(adaptor.getNewK(), adaptor.getCacheK(), newKType);
        Value updatedV = createInsertSlice(adaptor.getNewV(), adaptor.getCacheV(), newVType);

        rewriter.replaceOp(op, {updatedK, updatedV});
        return success();
    }
};

} // namespace

void tenzo::populateTenzoToLinalgConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<FusedLoweringToLinalg>(patterns.getContext());
    patterns.add<AddLoweringToLinalg>(patterns.getContext());
    patterns.add<ReluLoweringToLinalg>(patterns.getContext());
    patterns.add<AttentionLoweringToLinalg>(patterns.getContext());
    patterns.add<RopeLoweringToLinalg>(patterns.getContext());
    patterns.add<MatMulLoweringToLinalg>(patterns.getContext());
    patterns.add<Conv2DLoweringToLinalg>(patterns.getContext());
    patterns.add<QuantizeLoweringToLinalg>(patterns.getContext());
    patterns.add<DequantizeLoweringToLinalg>(patterns.getContext());
    patterns.add<KVCacheUpdateLowering>(patterns.getContext());
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
        target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect, math::MathDialect, tensor::TensorDialect>();
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