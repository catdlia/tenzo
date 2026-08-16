#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
// tenzo.matmul_q8 -> scf.parallel
//===----------------------------------------------------------------------===//
struct MatMulQ8Lowering : public OpConversionPattern<tenzo::MatMulQ8Op> {
    using OpConversionPattern<tenzo::MatMulQ8Op>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::MatMulQ8Op op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();
        auto rhsType = mlir::cast<RankedTensorType>(adaptor.getRhsQ().getType());

        // Create zero-initialized output tensor
        auto zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));
        auto emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);
        auto filledTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor});
        
        Value resultMemref = rewriter.create<bufferization::ToBufferOp>(
            loc, MemRefType::get(resultType.getShape(), rewriter.getF32Type()), filledTensor.getResult(0));

        int64_t N = rhsType.getShape()[0];
        int64_t K = rhsType.getShape()[1];

        auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto cN = rewriter.create<arith::ConstantIndexOp>(loc, N);
        auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto cK = rewriter.create<arith::ConstantIndexOp>(loc, K);

        rewriter.create<scf::ParallelOp>(loc, ValueRange{c0}, ValueRange{cN}, ValueRange{c1}, ValueRange{},
            [&](OpBuilder &b, Location loc, ValueRange ivs, ValueRange regionArgs) {
                Value n = ivs[0];
                auto c0_f32 = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0));
                
                auto kLoop = b.create<scf::ForOp>(
                    loc, c0, cK, c1, ValueRange{c0_f32},
                    [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
                        Value k = iv;
                        Value sum = iterArgs[0];

                        // Load activation f32
                        Value actVal = b.create<tensor::ExtractOp>(loc, adaptor.getLhs(), ValueRange{c0, c0, k});
                        
                        // Load weight i8
                        Value wVal = b.create<tensor::ExtractOp>(loc, adaptor.getRhsQ(), ValueRange{n, k});
                        
                        // Convert i8 to f32
                        Value wExt = b.create<arith::ExtSIOp>(loc, b.getI32Type(), wVal);
                        Value wF32 = b.create<arith::SIToFPOp>(loc, b.getF32Type(), wExt);

                        Value mul = b.create<arith::MulFOp>(loc, actVal, wF32);
                        Value newSum = b.create<arith::AddFOp>(loc, sum, mul);

                        b.create<scf::YieldOp>(loc, newSum);
                    });

                Value finalSum = kLoop.getResult(0);

                // Load scale
                Value scale = b.create<tensor::ExtractOp>(loc, adaptor.getRhsScales(), ValueRange{n});
                Value scaledSum = b.create<arith::MulFOp>(loc, finalSum, scale);

                b.create<memref::StoreOp>(loc, scaledSum, resultMemref, ValueRange{c0, c0, n});
                b.create<scf::ReduceOp>(loc);
            });

        Value newResultTensor = rewriter.create<bufferization::ToTensorOp>(
            loc, resultType, resultMemref, /*restrict=*/true);
        rewriter.replaceOp(op, newResultTensor);
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

                    Value rowByteIdx = b.create<arith::DivUIOp>(l, rIdx, fourVal);
                    Value remIdx = b.create<arith::RemUIOp>(l, rIdx, fourVal);
                    Value shiftIdx = b.create<arith::MulIOp>(l, remIdx, twoVal);
                    Value shiftI8 = b.create<arith::IndexCastOp>(l, b.getI8Type(), shiftIdx);

                    Value byteVal = b.create<tensor::ExtractOp>(l, input, ValueRange{rowByteIdx, cIdx});
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

        auto reshapeTo4D = [&](Value val, RankedTensorType valType, RankedTensorType targetKVType) -> Value {
            if (valType.getRank() == 4) return val;
            int64_t B = targetKVType.getDimSize(0);
            int64_t H = targetKVType.getDimSize(1);
            int64_t Dh = targetKVType.getDimSize(3);
            auto targetType = RankedTensorType::get({B, H, 1, Dh}, valType.getElementType());
            
            Value empty = rewriter.create<tensor::EmptyOp>(loc, targetType.getShape(), valType.getElementType());
            auto d0 = rewriter.getAffineDimExpr(0);
            auto d1 = rewriter.getAffineDimExpr(1);
            auto d2 = rewriter.getAffineDimExpr(2);
            auto d3 = rewriter.getAffineDimExpr(3);
            
            AffineExpr flatIdx = d1 * Dh + d3;
            AffineMap inMap = AffineMap::get(4, 0, {d0, rewriter.getAffineConstantExpr(0), flatIdx}, rewriter.getContext());
            AffineMap outMap = rewriter.getMultiDimIdentityMap(4);
            
            return rewriter.create<linalg::GenericOp>(
                loc, targetType, ValueRange{val}, ValueRange{empty},
                ArrayRef<AffineMap>{inMap, outMap},
                SmallVector<utils::IteratorType, 4>(4, utils::IteratorType::parallel),
                [&](OpBuilder &b, Location l, ValueRange args) {
                    b.create<linalg::YieldOp>(l, args[0]);
                }
            ).getResult(0);
        };

        Value newK4D = reshapeTo4D(adaptor.getNewK(), newKType, cacheKType);
        Value newV4D = reshapeTo4D(adaptor.getNewV(), newVType, cacheVType);
        auto newK4DType = mlir::cast<RankedTensorType>(newK4D.getType());
        auto newV4DType = mlir::cast<RankedTensorType>(newV4D.getType());

        auto createInsertSlice = [&](Value source, Value dest, RankedTensorType sourceType, RankedTensorType destType) -> Value {
            SmallVector<OpFoldResult> offsets, sizes, strides;
            int rank = destType.getRank();
            for (int i = 0; i < rank; ++i) {
                if (i == rank - 2) { // Sequence dimension (e.g. [B, H, Seq, Dim])
                    offsets.push_back(seqPosIndex);
                    sizes.push_back(rewriter.getIndexAttr(1));
                } else {
                    offsets.push_back(rewriter.getIndexAttr(0));
                    sizes.push_back(rewriter.getIndexAttr(sourceType.getDimSize(i)));
                }
                strides.push_back(rewriter.getIndexAttr(1));
            }
            return rewriter.create<tensor::InsertSliceOp>(loc, source, dest, offsets, sizes, strides);
        };

        Value updatedK = createInsertSlice(newK4D, adaptor.getCacheK(), newK4DType, cacheKType);
        Value updatedV = createInsertSlice(newV4D, adaptor.getCacheV(), newV4DType, cacheVType);

        rewriter.replaceOp(op, {updatedK, updatedV});
        return success();
    }
};

struct RMSNormLoweringToLinalg : public OpConversionPattern<tenzo::RMSNormOp> {
    using OpConversionPattern<tenzo::RMSNormOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::RMSNormOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        Value input = adaptor.getInput();
        Value weight = adaptor.getWeight();
        float eps = op.getEps().convertToFloat();

        auto inType = mlir::cast<RankedTensorType>(input.getType());
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();
        int rank = resultType.getRank();
        int64_t dSize = inType.getDimSize(rank - 1);

        Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);

        Value dSizeConst = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, static_cast<double>(dSize)));
        Value epsConst = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, static_cast<double>(eps)));

        SmallVector<AffineMap, 1> indexingMaps = {
            rewriter.getMultiDimIdentityMap(rank)
        };
        SmallVector<utils::IteratorType, 3> iterators(rank, utils::IteratorType::parallel);

        // Step 1: Compute sum of squares along last dimension (reduction)
        SmallVector<int64_t> redShape(resultType.getShape().begin(), resultType.getShape().end() - 1);
        auto redType = RankedTensorType::get(redShape, elemType);

        Value zeroF = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0));
        Value emptyRed = rewriter.create<tensor::EmptyOp>(loc, redShape, elemType);
        Value initRed = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF}, ValueRange{emptyRed}).getResult(0);

        SmallVector<AffineExpr, 4> redExprs;
        for (int i = 0; i < rank - 1; ++i) {
            redExprs.push_back(rewriter.getAffineDimExpr(i));
        }
        AffineMap redOutMap = AffineMap::get(rank, 0, redExprs, rewriter.getContext());

        SmallVector<AffineMap, 2> redMaps = {
            rewriter.getMultiDimIdentityMap(rank),
            redOutMap
        };

        SmallVector<utils::IteratorType> redIterators(rank - 1, utils::IteratorType::parallel);
        redIterators.push_back(utils::IteratorType::reduction);

        Value sumSq = rewriter.create<linalg::GenericOp>(
            loc, redType, ValueRange{input}, ValueRange{initRed},
            redMaps, redIterators,
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value x = args[0];
                Value acc = args[1];
                Value sq = b.create<arith::MulFOp>(l, x, x);
                Value newAcc = b.create<arith::AddFOp>(l, acc, sq);
                b.create<linalg::YieldOp>(l, newAcc);
            }
        ).getResult(0);

        // Step 2: Compute RMS per row (var = sSq / dSize + eps, rms = 1 / sqrt(var))
        Value emptyRms = rewriter.create<tensor::EmptyOp>(loc, redShape, elemType);
        SmallVector<AffineMap, 2> rmsMaps = {
            rewriter.getMultiDimIdentityMap(rank - 1),
            rewriter.getMultiDimIdentityMap(rank - 1)
        };
        SmallVector<utils::IteratorType> rmsIterators(rank - 1, utils::IteratorType::parallel);

        Value rmsTensor = rewriter.create<linalg::GenericOp>(
            loc, redType, ValueRange{sumSq}, ValueRange{emptyRms},
            rmsMaps, rmsIterators,
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value sSq = args[0];
                Value meanSq = b.create<arith::DivFOp>(l, sSq, dSizeConst);
                Value var = b.create<arith::AddFOp>(l, meanSq, epsConst);
                Value sqrtVar = b.create<math::SqrtOp>(l, var);
                Value one = b.create<arith::ConstantOp>(l, rewriter.getFloatAttr(elemType, 1.0));
                Value rms = b.create<arith::DivFOp>(l, one, sqrtVar);
                b.create<linalg::YieldOp>(l, rms);
            }
        ).getResult(0);

        // Step 3: Normalize and scale elementwise
        AffineMap weightMap = AffineMap::get(rank, 0, {rewriter.getAffineDimExpr(rank - 1)}, rewriter.getContext());

        SmallVector<AffineMap, 4> normMaps = {
            rewriter.getMultiDimIdentityMap(rank),
            redOutMap, // this maps [B, Seq, D] -> [B, Seq] for rmsTensor
            weightMap,
            rewriter.getMultiDimIdentityMap(rank)
        };

        SmallVector<utils::IteratorType> normIterators(rank, utils::IteratorType::parallel);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op, resultType,
            ValueRange{input, rmsTensor, weight}, ValueRange{emptyOut},
            normMaps, normIterators,
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value x = args[0];
                Value rms = args[1];
                Value w = args[2];

                Value normX = b.create<arith::MulFOp>(l, x, rms);
                Value res = b.create<arith::MulFOp>(l, normX, w);

                b.create<linalg::YieldOp>(l, res);
            }
        );
        return success();
    }
};

struct EmbeddingLoweringToLinalg : public OpConversionPattern<tenzo::EmbeddingOp> {
    using OpConversionPattern<tenzo::EmbeddingOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::EmbeddingOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        Value indices = adaptor.getIndices();
        Value weight_q = adaptor.getWeightQ();
        Value scales = adaptor.getScales();

        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();
        int rank = resultType.getRank();

        Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);

        SmallVector<AffineMap, 1> indexingMaps = {
            rewriter.getMultiDimIdentityMap(rank)
        };
        SmallVector<utils::IteratorType, 3> iterators(rank, utils::IteratorType::parallel);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op, resultType,
            ValueRange{}, ValueRange{emptyOut},
            indexingMaps, iterators,
            [&](OpBuilder &b, Location l, ValueRange args) {
                SmallVector<Value> idxIndices;
                for (int r = 0; r < rank - 1; ++r) {
                    idxIndices.push_back(b.create<linalg::IndexOp>(l, r));
                }
                Value tokenI32 = b.create<tensor::ExtractOp>(l, indices, idxIndices);
                Value tokenIdx = b.create<arith::IndexCastOp>(l, b.getIndexType(), tokenI32);
                Value dIdx = b.create<linalg::IndexOp>(l, rank - 1);

                Value embVal;
                if (scales) {
                    Value embValQ = b.create<tensor::ExtractOp>(l, weight_q, ValueRange{tokenIdx, dIdx});
                    Value embValQExt = b.create<arith::ExtSIOp>(l, b.getI32Type(), embValQ);
                    Value embValQF32 = b.create<arith::SIToFPOp>(l, b.getF32Type(), embValQExt);
                    Value scale = b.create<tensor::ExtractOp>(l, scales, ValueRange{tokenIdx});
                    embVal = b.create<arith::MulFOp>(l, embValQF32, scale);
                } else {
                    embVal = b.create<tensor::ExtractOp>(l, weight_q, ValueRange{tokenIdx, dIdx});
                }
                
                b.create<linalg::YieldOp>(l, embVal);
            }
        );
        return success();
    }
};

struct BitLinearElutLoweringToLinalg : public OpConversionPattern<tenzo::BitLinearElutOp> {
    using OpConversionPattern<tenzo::BitLinearElutOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::BitLinearElutOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        Value input = adaptor.getInput();
        Value weights = adaptor.getWeights();
        Value scale = adaptor.getScale();

        auto inType = mlir::cast<RankedTensorType>(input.getType());
        auto wType = mlir::cast<RankedTensorType>(weights.getType());
        auto resType = mlir::cast<RankedTensorType>(op.getResult().getType());

        int64_t K_packed = wType.getShape()[1];

        Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(), resType.getElementType());

        SmallVector<AffineMap, 1> indexingMaps = {
            rewriter.getMultiDimIdentityMap(3)
        };
        SmallVector<utils::IteratorType, 3> iterators = {
            utils::IteratorType::parallel,
            utils::IteratorType::parallel,
            utils::IteratorType::parallel
        };

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op, resType,
            ValueRange{}, ValueRange{emptyOut},
            indexingMaps, iterators,
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value bIdx = b.create<linalg::IndexOp>(l, 0);
                Value sIdx = b.create<linalg::IndexOp>(l, 1);
                Value nIdx = b.create<linalg::IndexOp>(l, 2);

                Value zeroF32 = b.create<arith::ConstantOp>(l, b.getF32FloatAttr(0.0f));
                Value zeroIdx = b.create<arith::ConstantIndexOp>(l, 0);
                Value oneIdx = b.create<arith::ConstantIndexOp>(l, 1);
                Value kPackedIdx = b.create<arith::ConstantIndexOp>(l, K_packed);

                Value sum = b.create<scf::ForOp>(
                    l, zeroIdx, kPackedIdx, oneIdx, ValueRange{zeroF32},
                    [&](OpBuilder &bLoop, Location lLoop, Value kIdx, ValueRange accRange) {
                        Value curAcc = accRange[0];
                        Value wByte = bLoop.create<tensor::ExtractOp>(lLoop, weights, ValueRange{nIdx, kIdx});
                        Value wByteI32 = bLoop.create<arith::ExtUIOp>(lLoop, bLoop.getI32Type(), wByte);

                        Value c0 = bLoop.create<arith::ConstantIndexOp>(lLoop, 0);
                        Value c1 = bLoop.create<arith::ConstantIndexOp>(lLoop, 1);
                        Value c2 = bLoop.create<arith::ConstantIndexOp>(lLoop, 2);
                        Value c3 = bLoop.create<arith::ConstantIndexOp>(lLoop, 3);
                        Value c4 = bLoop.create<arith::ConstantIndexOp>(lLoop, 4);

                        Value kBase = bLoop.create<arith::MulIOp>(lLoop, kIdx, c4);

                        Value c0I32 = bLoop.create<arith::ConstantOp>(lLoop, bLoop.getI32IntegerAttr(0));
                        Value c2I32 = bLoop.create<arith::ConstantOp>(lLoop, bLoop.getI32IntegerAttr(2));
                        Value c4I32 = bLoop.create<arith::ConstantOp>(lLoop, bLoop.getI32IntegerAttr(4));
                        Value c6I32 = bLoop.create<arith::ConstantOp>(lLoop, bLoop.getI32IntegerAttr(6));
                        Value mask3 = bLoop.create<arith::ConstantOp>(lLoop, bLoop.getI32IntegerAttr(3));
                        Value oneI32 = bLoop.create<arith::ConstantOp>(lLoop, bLoop.getI32IntegerAttr(1));

                        auto add_slot = [&](Value shiftVal, Value kOffset, Value accIn) {
                            Value shifted = (shiftVal == c0I32) ? wByteI32 : bLoop.create<arith::ShRUIOp>(lLoop, wByteI32, shiftVal);
                            Value code = bLoop.create<arith::AndIOp>(lLoop, shifted, mask3);
                            Value ternI32 = bLoop.create<arith::SubIOp>(lLoop, code, oneI32);
                            Value ternF32 = bLoop.create<arith::SIToFPOp>(lLoop, bLoop.getF32Type(), ternI32);

                            Value inKIdx = bLoop.create<arith::AddIOp>(lLoop, kBase, kOffset);
                            Value inVal = bLoop.create<tensor::ExtractOp>(lLoop, input, ValueRange{bIdx, sIdx, inKIdx});
                            Value prod = bLoop.create<arith::MulFOp>(lLoop, inVal, ternF32);
                            return bLoop.create<arith::AddFOp>(lLoop, accIn, prod);
                        };

                        Value acc0 = add_slot(c0I32, c0, curAcc);
                        Value acc1 = add_slot(c2I32, c1, acc0);
                        Value acc2 = add_slot(c4I32, c2, acc1);
                        Value acc3 = add_slot(c6I32, c3, acc2);

                        bLoop.create<scf::YieldOp>(lLoop, ValueRange{acc3});
                    }
                ).getResult(0);

                Value finalVal = b.create<arith::MulFOp>(l, sum, scale);
                b.create<linalg::YieldOp>(l, finalVal);
            }
        );
        return success();
    }
};

} // namespace


struct BitLinearTL1PackLoweringToLinalg : public OpConversionPattern<tenzo::BitLinearTL1PackOp> {
    using OpConversionPattern<tenzo::BitLinearTL1PackOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::BitLinearTL1PackOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        Value act = adaptor.getInput();
        Value weights = adaptor.getWeights(); // [N_blocks, K/2, 32]
        auto actType = mlir::cast<RankedTensorType>(act.getType());
        auto wType = mlir::cast<RankedTensorType>(weights.getType());
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();

        if (actType.getRank() != 3 || wType.getRank() != 3 || resultType.getRank() != 3) return failure();

        int64_t K = actType.getShape()[2];
        int64_t N = resultType.getShape()[2];
        int64_t n_blocks = wType.getShape()[0];
        int64_t K_half = wType.getShape()[1];
        if (K % 2 != 0 || K_half * 2 != K || wType.getShape()[2] != 32 || n_blocks * 64 != N) return failure();

        Value scaleF;
        if (op.getScale()) {
            scaleF = adaptor.getScale();
            if (auto sType = mlir::dyn_cast<RankedTensorType>(scaleF.getType())) {
                Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
                SmallVector<Value> sIdxs(sType.getRank(), zeroIdx);
                scaleF = rewriter.create<tensor::ExtractOp>(loc, scaleF, sIdxs);
            }
        } else {
            scaleF = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 1.0));
        }

        Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value ubK = rewriter.create<arith::ConstantIndexOp>(loc, K);
        Value c128Idx = rewriter.create<arith::ConstantIndexOp>(loc, 128);

        Value dimB = rewriter.create<tensor::DimOp>(loc, act, 0);
        Value dimS = rewriter.create<tensor::DimOp>(loc, act, 1);

        auto vector128f32 = VectorType::get({128}, rewriter.getF32Type());
        Value zero128f32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vector128f32, rewriter.getF32FloatAttr(0.0f)));

        Value rhsMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get(wType.getShape(), rewriter.getI8Type()), weights);
        Value outMemref = rewriter.create<memref::AllocOp>(loc, MemRefType::get(resultType.getShape(), elemType));
        
        // LUT memory
        auto vec32i8 = VectorType::get({32}, rewriter.getI8Type());
        Value lutMemref = rewriter.create<memref::AllocOp>(loc, MemRefType::get({K_half}, vec32i8));
        Value zeros32i8 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec32i8, rewriter.getI8IntegerAttr(0)));
        Value lhsMemref = rewriter.create<memref::AllocOp>(loc, MemRefType::get({K}, rewriter.getI16Type()));

        auto module = op->getParentOfType<ModuleOp>();
        if (!module.lookupSymbol("llvm.x86.avx2.pshuf.b")) {
            OpBuilder moduleBuilder(module.getBodyRegion());
            auto funcOp = moduleBuilder.create<func::FuncOp>(loc, "llvm.x86.avx2.pshuf.b", 
                rewriter.getFunctionType({vec32i8, vec32i8}, {vec32i8}));
            funcOp.setPrivate();
        }

        auto vec32i16 = VectorType::get({32}, rewriter.getI16Type());
        auto vec16i16 = VectorType::get({16}, rewriter.getI16Type());
        auto vec32i32 = VectorType::get({32}, rewriter.getI32Type());
        auto vec32f32 = VectorType::get({32}, rewriter.getF32Type());

        Value zero32i32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec32i32, rewriter.getI32IntegerAttr(0)));
        Value zero32i16 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec32i16, rewriter.getI16IntegerAttr(0)));
        Value zeroI8Scalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getI8IntegerAttr(0));

        Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
        Value cn_blocks = rewriter.create<arith::ConstantIndexOp>(loc, n_blocks);
        Value ck_half = rewriter.create<arith::ConstantIndexOp>(loc, K_half);

        rewriter.create<scf::ForOp>(loc, zeroIdx, dimB, oneIdx, ValueRange{}, [&](OpBuilder &bB, Location locB, Value bIdx, ValueRange argsB) {
            bB.create<scf::ForOp>(locB, zeroIdx, dimS, oneIdx, ValueRange{}, [&](OpBuilder &bS, Location locS, Value sIdx, ValueRange argsS) {

                // 1. Max Abs
                Value maxVec = bS.create<scf::ForOp>(
                    locS, zeroIdx, ubK, c128Idx, ValueRange{zero128f32},
                    [&](OpBuilder &b, Location l, Value kIV, ValueRange iterK) {
                        Value padF32 = b.create<arith::ConstantOp>(l, b.getF32FloatAttr(0.0f));
                        bool inBounds[] = {true};
                        Value v = b.create<vector::TransferReadOp>(l, vector128f32, act, ValueRange{bIdx, sIdx, kIV}, padF32, inBounds);
                        Value absV = b.create<math::AbsFOp>(l, v);
                        Value newMax = b.create<arith::MaximumFOp>(l, absV, iterK[0]);
                        b.create<scf::YieldOp>(l, ValueRange{newMax});
                    }).getResult(0);

                Value maxAbs = bS.create<vector::ReductionOp>(locS, vector::CombiningKind::MAXIMUMF, maxVec);
                Value eps = bS.create<arith::ConstantOp>(locS, bS.getFloatAttr(elemType, 1e-5f));
                Value safeMaxAbs = bS.create<arith::MaximumFOp>(locS, maxAbs, eps);

                Value c31F = bS.create<arith::ConstantOp>(locS, bS.getFloatAttr(elemType, 31.0f));
                Value scaleAct = bS.create<arith::DivFOp>(locS, c31F, safeMaxAbs);
                Value invScaleAct = bS.create<arith::DivFOp>(locS, safeMaxAbs, c31F);
                Value totalScale = bS.create<arith::MulFOp>(locS, invScaleAct, scaleF);

                Value scaleActVec = bS.create<vector::SplatOp>(locS, vector128f32, scaleAct);
                auto vector128i16 = VectorType::get({128}, bS.getI16Type());

                // 2. Quantize
                bS.create<scf::ForOp>(
                    locS, zeroIdx, ubK, c128Idx, ValueRange{},
                    [&](OpBuilder &b, Location l, Value kIV, ValueRange iterK) {
                        Value padF32 = b.create<arith::ConstantOp>(l, b.getF32FloatAttr(0.0f));
                        bool inBounds[] = {true};
                        Value vF32 = b.create<vector::TransferReadOp>(l, vector128f32, act, ValueRange{bIdx, sIdx, kIV}, padF32, inBounds);
                        Value vScaled = b.create<arith::MulFOp>(l, vF32, scaleActVec);
                        auto vector128i32 = VectorType::get({128}, b.getI32Type());
                        Value vI32 = b.create<arith::FPToSIOp>(l, vector128i32, vScaled);
                        Value vI16 = b.create<arith::TruncIOp>(l, vector128i16, vI32);
                        b.create<vector::TransferWriteOp>(l, vI16, lhsMemref, ValueRange{kIV});
                        b.create<scf::YieldOp>(l);
                    });

                // 3. LUT Gen
                bS.create<scf::ForOp>(locS, zeroIdx, ck_half, oneIdx, ValueRange{}, 
                    [&](OpBuilder &b, Location l, Value k_half_idx, ValueRange args) {
                        Value k_even_idx = b.create<arith::MulIOp>(l, k_half_idx, c2);
                        Value k_odd_idx = b.create<arith::AddIOp>(l, k_even_idx, oneIdx);
                        
                        Value a0_i16 = b.create<memref::LoadOp>(l, lhsMemref, ValueRange{k_even_idx});
                        Value a1_i16 = b.create<memref::LoadOp>(l, lhsMemref, ValueRange{k_odd_idx});
                        
                        Value a0_i32 = b.create<arith::ExtSIOp>(l, b.getI32Type(), a0_i16);
                        Value a1_i32 = b.create<arith::ExtSIOp>(l, b.getI32Type(), a1_i16);
                        
                        Value lut_vec = zeros32i8;
                        int w0_vals[] = {-1, 0, 1};
                        int w1_vals[] = {-1, 0, 1};
                        
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                int w0 = w0_vals[i];
                                int w1 = w1_vals[j];
                                int idx = (w0 + 1) | ((w1 + 1) << 2);

                                Value sum_i32;
                                if (w0 == 0 && w1 == 0) {
                                    sum_i32 = b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0));
                                } else if (w0 == 0) {
                                    sum_i32 = (w1 == 1) ? a1_i32 : b.create<arith::SubIOp>(l, b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0)), a1_i32);
                                } else if (w1 == 0) {
                                    sum_i32 = (w0 == 1) ? a0_i32 : b.create<arith::SubIOp>(l, b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0)), a0_i32);
                                } else {
                                    Value term0 = (w0 == 1) ? a0_i32 : b.create<arith::SubIOp>(l, b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0)), a0_i32);
                                    Value term1 = (w1 == 1) ? a1_i32 : b.create<arith::SubIOp>(l, b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0)), a1_i32);
                                    sum_i32 = b.create<arith::AddIOp>(l, term0, term1);
                                }
                                
                                Value val_i8 = b.create<arith::TruncIOp>(l, b.getI8Type(), sum_i32);
                                lut_vec = b.create<vector::InsertElementOp>(l, val_i8, lut_vec, b.create<arith::ConstantIndexOp>(l, idx));
                                lut_vec = b.create<vector::InsertElementOp>(l, val_i8, lut_vec, b.create<arith::ConstantIndexOp>(l, idx + 16));
                            }
                        }
                        b.create<memref::StoreOp>(l, lut_vec, lutMemref, ValueRange{k_half_idx});
                        b.create<scf::YieldOp>(l);
                    });

                // 4. Dot Product
                Value c0F_i8_vec = bS.create<arith::ConstantOp>(locS, DenseElementsAttr::get(vec32i8, bS.getI8IntegerAttr(0x0F)));
                Value c4_i16_vec = bS.create<arith::ConstantOp>(locS, DenseElementsAttr::get(vec16i16, bS.getI16IntegerAttr(4)));

                bS.create<scf::ParallelOp>(locS, ValueRange{zeroIdx}, ValueRange{cn_blocks}, ValueRange{oneIdx}, ValueRange{},
                    [&](OpBuilder &b, Location loc2, ValueRange ivs, ValueRange nArgs) {
                        Value n_block_idx = ivs[0];
                        Value c256 = b.create<arith::ConstantIndexOp>(loc2, 256);

                        auto kLoop = b.create<scf::ForOp>(loc2, zeroIdx, ck_half, c256, ValueRange{zero32i32, zero32i32},
                            [&](OpBuilder &b2, Location loc3, Value k_outer, ValueRange accs_i32) {
                                Value nextK = b2.create<arith::AddIOp>(loc3, k_outer, c256);
                                Value isEnd = b2.create<arith::CmpIOp>(loc3, arith::CmpIPredicate::sgt, nextK, ck_half);
                                Value k_end = b2.create<arith::SelectOp>(loc3, isEnd, ck_half, nextK);
                                
                                Value c2 = b2.create<arith::ConstantIndexOp>(loc3, 2);
                                auto kInnerLoop = b2.create<scf::ForOp>(loc3, k_outer, k_end, c2, ValueRange{zero32i16, zero32i16},
                                    [&](OpBuilder &b3, Location loc4, Value k_micro_idx, ValueRange accs_i16) {
                                        Value acc_low_i16 = accs_i16[0];
                                        Value acc_high_i16 = accs_i16[1];
                                        
                                        Value k_micro_plus_1 = b3.create<arith::AddIOp>(loc4, k_micro_idx, oneIdx);
                                        
                                        Value lut_k0 = b3.create<memref::LoadOp>(loc4, lutMemref, ValueRange{k_micro_idx});
                                        Value lut_k1 = b3.create<memref::LoadOp>(loc4, lutMemref, ValueRange{k_micro_plus_1});

                                        bool inB[] = {true};
                                        ArrayRef<bool> inBounds1D(inB);
                                        Value w_k0 = b3.create<vector::TransferReadOp>(loc4, vec32i8, rhsMemref, ValueRange{n_block_idx, k_micro_idx, zeroIdx}, zeroI8Scalar, inBounds1D);
                                        Value w_k1 = b3.create<vector::TransferReadOp>(loc4, vec32i8, rhsMemref, ValueRange{n_block_idx, k_micro_plus_1, zeroIdx}, zeroI8Scalar, inBounds1D);

                                        auto process_w = [&](Value w_bytes, Value lut_vec, Value &res_low_16, Value &res_high_16) {
                                            Value idx_low = b3.create<arith::AndIOp>(loc4, w_bytes, c0F_i8_vec);
                                            Value w_bytes_as_i16 = b3.create<vector::BitCastOp>(loc4, vec16i16, w_bytes);
                                            Value w_bytes_shr4_i16 = b3.create<arith::ShRUIOp>(loc4, w_bytes_as_i16, c4_i16_vec);
                                            Value w_bytes_shr4 = b3.create<vector::BitCastOp>(loc4, vec32i8, w_bytes_shr4_i16);
                                            Value idx_high = b3.create<arith::AndIOp>(loc4, w_bytes_shr4, c0F_i8_vec);

                                            Value res_low_w = b3.create<func::CallOp>(loc4, "llvm.x86.avx2.pshuf.b", TypeRange{vec32i8}, ValueRange{lut_vec, idx_low}).getResult(0);
                                            Value res_high_w = b3.create<func::CallOp>(loc4, "llvm.x86.avx2.pshuf.b", TypeRange{vec32i8}, ValueRange{lut_vec, idx_high}).getResult(0);
                                            
                                            res_low_16 = b3.create<arith::ExtSIOp>(loc4, vec32i16, res_low_w);
                                            res_high_16 = b3.create<arith::ExtSIOp>(loc4, vec32i16, res_high_w);
                                        };

                                        Value res_low_k0_16, res_high_k0_16;
                                        process_w(w_k0, lut_k0, res_low_k0_16, res_high_k0_16);
                                        
                                        Value res_low_k1_16, res_high_k1_16;
                                        process_w(w_k1, lut_k1, res_low_k1_16, res_high_k1_16);

                                        Value sum_low_16 = b3.create<arith::AddIOp>(loc4, res_low_k0_16, res_low_k1_16);
                                        Value sum_high_16 = b3.create<arith::AddIOp>(loc4, res_high_k0_16, res_high_k1_16);

                                        Value next_acc_low_i16 = b3.create<arith::AddIOp>(loc4, acc_low_i16, sum_low_16);
                                        Value next_acc_high_i16 = b3.create<arith::AddIOp>(loc4, acc_high_i16, sum_high_16);

                                        b3.create<scf::YieldOp>(loc4, ValueRange{next_acc_low_i16, next_acc_high_i16});
                                    });
                                
                                Value block_acc_low_i32 = b2.create<arith::ExtSIOp>(loc3, vec32i32, kInnerLoop.getResult(0));
                                Value block_acc_high_i32 = b2.create<arith::ExtSIOp>(loc3, vec32i32, kInnerLoop.getResult(1));

                                Value next_acc_low = b2.create<arith::AddIOp>(loc3, accs_i32[0], block_acc_low_i32);
                                Value next_acc_high = b2.create<arith::AddIOp>(loc3, accs_i32[1], block_acc_high_i32);

                                b2.create<scf::YieldOp>(loc3, ValueRange{next_acc_low, next_acc_high});
                            });

                        Value totalScaleVec = b.create<vector::BroadcastOp>(loc2, vec32f32, totalScale);
                        Value c32Idx = b.create<arith::ConstantIndexOp>(loc2, 32);
                        Value c64Idx = b.create<arith::ConstantIndexOp>(loc2, 64);

                        auto write_out = [&](OpBuilder &b1, Location l1, Value acc_low_i32, Value acc_high_i32, int output_offset_within_block) {
                            Value out_offset = b1.create<arith::AddIOp>(l1, b1.create<arith::MulIOp>(l1, n_block_idx, c64Idx), b1.create<arith::ConstantIndexOp>(l1, output_offset_within_block));
                            Value sum_val_f32 = b1.create<arith::SIToFPOp>(l1, vec32f32, acc_low_i32);
                            Value final_f32 = b1.create<arith::MulFOp>(l1, sum_val_f32, totalScaleVec);
                            b1.create<vector::TransferWriteOp>(l1, final_f32, outMemref, ValueRange{bIdx, sIdx, out_offset});

                            Value out_offset_high = b1.create<arith::AddIOp>(l1, out_offset, c32Idx);
                            Value sum_val_high_f32 = b1.create<arith::SIToFPOp>(l1, vec32f32, acc_high_i32);
                            Value final_f32_high = b1.create<arith::MulFOp>(l1, sum_val_high_f32, totalScaleVec);
                            b1.create<vector::TransferWriteOp>(l1, final_f32_high, outMemref, ValueRange{bIdx, sIdx, out_offset_high});
                        };

                        write_out(b, loc2, kLoop.getResult(0), kLoop.getResult(1), 0);

                        b.create<scf::ReduceOp>(loc2);
                    });
                bS.create<scf::YieldOp>(locS);
            });
            bB.create<scf::YieldOp>(locB);
        });

        rewriter.create<memref::DeallocOp>(loc, lutMemref);
        
        Value finalTensor = rewriter.create<bufferization::ToTensorOp>(loc, resultType, outMemref);
        rewriter.replaceOp(op, finalTensor);

        rewriter.create<memref::DeallocOp>(loc, lhsMemref);

        return success();
    }
};

struct BitLinearTL1LoweringToLinalg : public OpConversionPattern<tenzo::BitLinearTL1Op> {
    using OpConversionPattern<tenzo::BitLinearTL1Op>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::BitLinearTL1Op op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Fallback for non-packed version, implemented exactly the same but using different unpacking logic.
        // It's not the primary execution path in the current model but kept for compatibility.
        auto loc = op.getLoc();
        Value act = adaptor.getInput();
        Value weights = adaptor.getWeights(); // [N_blocks, K/2, 32]
        auto actType = mlir::cast<RankedTensorType>(act.getType());
        auto wType = mlir::cast<RankedTensorType>(weights.getType());
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();

        if (actType.getRank() != 3 || wType.getRank() != 3 || resultType.getRank() != 3) return failure();

        int64_t K = actType.getShape()[2];
        int64_t N = resultType.getShape()[2];
        int64_t n_blocks = wType.getShape()[0];
        int64_t K_half = wType.getShape()[1];
        if (K % 2 != 0 || K_half * 2 != K || wType.getShape()[2] != 32 || n_blocks * 64 != N) return failure();

        Value scaleF;
        if (op.getScale()) {
            scaleF = adaptor.getScale();
            if (auto sType = mlir::dyn_cast<RankedTensorType>(scaleF.getType())) {
                Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
                SmallVector<Value> sIdxs(sType.getRank(), zeroIdx);
                scaleF = rewriter.create<tensor::ExtractOp>(loc, scaleF, sIdxs);
            }
        } else {
            scaleF = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 1.0));
        }

        Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value ubK = rewriter.create<arith::ConstantIndexOp>(loc, K);
        Value c128Idx = rewriter.create<arith::ConstantIndexOp>(loc, 128);

        Value dimB = rewriter.create<tensor::DimOp>(loc, act, 0);
        Value dimS = rewriter.create<tensor::DimOp>(loc, act, 1);

        auto vector128f32 = VectorType::get({128}, rewriter.getF32Type());
        Value zero128f32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vector128f32, rewriter.getF32FloatAttr(0.0f)));

        Value rhsMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get(wType.getShape(), rewriter.getI8Type()), weights);
        Value outMemref = rewriter.create<memref::AllocOp>(loc, MemRefType::get(resultType.getShape(), elemType));
        
        // LUT memory
        auto vec32i8 = VectorType::get({32}, rewriter.getI8Type());
        Value lutMemref = rewriter.create<memref::AllocOp>(loc, MemRefType::get({K_half}, vec32i8));
        Value zeros32i8 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec32i8, rewriter.getI8IntegerAttr(0)));
        Value lhsMemref = rewriter.create<memref::AllocOp>(loc, MemRefType::get({K}, rewriter.getI16Type()));

        auto module = op->getParentOfType<ModuleOp>();
        if (!module.lookupSymbol("llvm.x86.avx2.pshuf.b")) {
            OpBuilder moduleBuilder(module.getBodyRegion());
            auto funcOp = moduleBuilder.create<func::FuncOp>(loc, "llvm.x86.avx2.pshuf.b", 
                rewriter.getFunctionType({vec32i8, vec32i8}, {vec32i8}));
            funcOp.setPrivate();
        }

        auto vec32i16 = VectorType::get({32}, rewriter.getI16Type());
        auto vec16i16 = VectorType::get({16}, rewriter.getI16Type());
        auto vec32i32 = VectorType::get({32}, rewriter.getI32Type());
        auto vec32f32 = VectorType::get({32}, rewriter.getF32Type());

        Value zero32i32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec32i32, rewriter.getI32IntegerAttr(0)));
        Value zero32i16 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec32i16, rewriter.getI16IntegerAttr(0)));
        Value zeroI8Scalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getI8IntegerAttr(0));

        Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
        Value cn_blocks = rewriter.create<arith::ConstantIndexOp>(loc, n_blocks);
        Value ck_half = rewriter.create<arith::ConstantIndexOp>(loc, K_half);

        rewriter.create<scf::ForOp>(loc, zeroIdx, dimB, oneIdx, ValueRange{}, [&](OpBuilder &bB, Location locB, Value bIdx, ValueRange argsB) {
            bB.create<scf::ForOp>(locB, zeroIdx, dimS, oneIdx, ValueRange{}, [&](OpBuilder &bS, Location locS, Value sIdx, ValueRange argsS) {

                // 1. Max Abs
                Value maxVec = bS.create<scf::ForOp>(
                    locS, zeroIdx, ubK, c128Idx, ValueRange{zero128f32},
                    [&](OpBuilder &b, Location l, Value kIV, ValueRange iterK) {
                        Value padF32 = b.create<arith::ConstantOp>(l, b.getF32FloatAttr(0.0f));
                        bool inBounds[] = {true};
                        Value v = b.create<vector::TransferReadOp>(l, vector128f32, act, ValueRange{bIdx, sIdx, kIV}, padF32, inBounds);
                        Value absV = b.create<math::AbsFOp>(l, v);
                        Value newMax = b.create<arith::MaximumFOp>(l, absV, iterK[0]);
                        b.create<scf::YieldOp>(l, ValueRange{newMax});
                    }).getResult(0);

                Value maxAbs = bS.create<vector::ReductionOp>(locS, vector::CombiningKind::MAXIMUMF, maxVec);
                Value eps = bS.create<arith::ConstantOp>(locS, bS.getFloatAttr(elemType, 1e-5f));
                Value safeMaxAbs = bS.create<arith::MaximumFOp>(locS, maxAbs, eps);

                Value c63F = bS.create<arith::ConstantOp>(locS, bS.getFloatAttr(elemType, 63.0f));
                Value scaleAct = bS.create<arith::DivFOp>(locS, c63F, safeMaxAbs);
                Value invScaleAct = bS.create<arith::DivFOp>(locS, safeMaxAbs, c63F);
                Value totalScale = bS.create<arith::MulFOp>(locS, invScaleAct, scaleF);

                Value scaleActVec = bS.create<vector::SplatOp>(locS, vector128f32, scaleAct);
                auto vector128i16 = VectorType::get({128}, bS.getI16Type());

                // 2. Quantize
                bS.create<scf::ForOp>(
                    locS, zeroIdx, ubK, c128Idx, ValueRange{},
                    [&](OpBuilder &b, Location l, Value kIV, ValueRange iterK) {
                        Value padF32 = b.create<arith::ConstantOp>(l, b.getF32FloatAttr(0.0f));
                        bool inBounds[] = {true};
                        Value vF32 = b.create<vector::TransferReadOp>(l, vector128f32, act, ValueRange{bIdx, sIdx, kIV}, padF32, inBounds);
                        Value vScaled = b.create<arith::MulFOp>(l, vF32, scaleActVec);
                        auto vector128i32 = VectorType::get({128}, b.getI32Type());
                        Value vI32 = b.create<arith::FPToSIOp>(l, vector128i32, vScaled);
                        Value vI16 = b.create<arith::TruncIOp>(l, vector128i16, vI32);
                        b.create<vector::TransferWriteOp>(l, vI16, lhsMemref, ValueRange{kIV});
                        b.create<scf::YieldOp>(l);
                    });

                // 3. LUT Gen
                bS.create<scf::ForOp>(locS, zeroIdx, ck_half, oneIdx, ValueRange{}, 
                    [&](OpBuilder &b, Location l, Value k_half_idx, ValueRange args) {
                        Value k_even_idx = b.create<arith::MulIOp>(l, k_half_idx, c2);
                        Value k_odd_idx = b.create<arith::AddIOp>(l, k_even_idx, oneIdx);
                        
                        Value a0_i16 = b.create<memref::LoadOp>(l, lhsMemref, ValueRange{k_even_idx});
                        Value a1_i16 = b.create<memref::LoadOp>(l, lhsMemref, ValueRange{k_odd_idx});
                        
                        Value a0_i32 = b.create<arith::ExtSIOp>(l, b.getI32Type(), a0_i16);
                        Value a1_i32 = b.create<arith::ExtSIOp>(l, b.getI32Type(), a1_i16);
                        
                        Value lut_vec = zeros32i8;
                        int w_vals[] = {-1, 0, 1};
                        
                        for (int i = 0; i < 3; ++i) {
                            int w = w_vals[i];
                            
                            Value term_a0;
                            if (w == 1) term_a0 = a0_i32;
                            else if (w == 0) term_a0 = b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0));
                            else term_a0 = b.create<arith::SubIOp>(l, b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0)), a0_i32);
                            Value val_a0_i8 = b.create<arith::TruncIOp>(l, b.getI8Type(), term_a0);
                            
                            Value term_a1;
                            if (w == 1) term_a1 = a1_i32;
                            else if (w == 0) term_a1 = b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0));
                            else term_a1 = b.create<arith::SubIOp>(l, b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0)), a1_i32);
                            Value val_a1_i8 = b.create<arith::TruncIOp>(l, b.getI8Type(), term_a1);
                            
                            int idx_a0 = w + 1;
                            int idx_a1 = (w + 1) + 8;
                            
                            lut_vec = b.create<vector::InsertElementOp>(l, val_a0_i8, lut_vec, b.create<arith::ConstantIndexOp>(l, idx_a0));
                            lut_vec = b.create<vector::InsertElementOp>(l, val_a0_i8, lut_vec, b.create<arith::ConstantIndexOp>(l, idx_a0 + 16));
                            
                            lut_vec = b.create<vector::InsertElementOp>(l, val_a1_i8, lut_vec, b.create<arith::ConstantIndexOp>(l, idx_a1));
                            lut_vec = b.create<vector::InsertElementOp>(l, val_a1_i8, lut_vec, b.create<arith::ConstantIndexOp>(l, idx_a1 + 16));
                        }
                        b.create<memref::StoreOp>(l, lut_vec, lutMemref, ValueRange{k_half_idx});
                        b.create<scf::YieldOp>(l);
                    });

                // 4. Dot Product
                Value c03_i8_vec = bS.create<arith::ConstantOp>(locS, DenseElementsAttr::get(vec32i8, bS.getI8IntegerAttr(0x03)));
                Value c08_i8_vec = bS.create<arith::ConstantOp>(locS, DenseElementsAttr::get(vec32i8, bS.getI8IntegerAttr(0x08)));
                Value c2_i8_vec = bS.create<arith::ConstantOp>(locS, DenseElementsAttr::get(vec32i8, bS.getI8IntegerAttr(2)));
                Value c4_i8_vec = bS.create<arith::ConstantOp>(locS, DenseElementsAttr::get(vec32i8, bS.getI8IntegerAttr(4)));
                Value c6_i8_vec = bS.create<arith::ConstantOp>(locS, DenseElementsAttr::get(vec32i8, bS.getI8IntegerAttr(6)));

                bS.create<scf::ParallelOp>(locS, ValueRange{zeroIdx}, ValueRange{cn_blocks}, ValueRange{oneIdx}, ValueRange{},
                    [&](OpBuilder &b, Location loc2, ValueRange ivs, ValueRange nArgs) {
                        Value n_block_idx = ivs[0];
                        Value c256 = b.create<arith::ConstantIndexOp>(loc2, 256);

                        auto kLoop = b.create<scf::ForOp>(loc2, zeroIdx, ck_half, c256, ValueRange{zero32i32, zero32i32},
                            [&](OpBuilder &b2, Location loc3, Value k_outer, ValueRange accs_i32) {
                                Value nextK = b2.create<arith::AddIOp>(loc3, k_outer, c256);
                                Value isEnd = b2.create<arith::CmpIOp>(loc3, arith::CmpIPredicate::sgt, nextK, ck_half);
                                Value k_end = b2.create<arith::SelectOp>(loc3, isEnd, ck_half, nextK);
                                
                                auto kInnerLoop = b2.create<scf::ForOp>(loc3, k_outer, k_end, c2, ValueRange{zero32i16, zero32i16},
                                    [&](OpBuilder &b3, Location loc4, Value k_micro_idx, ValueRange accs_i16) {
                                        Value acc_low_i16 = accs_i16[0];
                                        Value acc_high_i16 = accs_i16[1];
                                        
                                        Value k_micro_plus_1 = b3.create<arith::AddIOp>(loc4, k_micro_idx, oneIdx);
                                        
                                        Value lut_k0 = b3.create<memref::LoadOp>(loc4, lutMemref, ValueRange{k_micro_idx});
                                        Value lut_k1 = b3.create<memref::LoadOp>(loc4, lutMemref, ValueRange{k_micro_plus_1});

                                        bool inB[] = {true};
                                        ArrayRef<bool> inBounds1D(inB);
                                        Value w_k0 = b3.create<vector::TransferReadOp>(loc4, vec32i8, rhsMemref, ValueRange{n_block_idx, k_micro_idx, zeroIdx}, zeroI8Scalar, inBounds1D);
                                        Value w_k1 = b3.create<vector::TransferReadOp>(loc4, vec32i8, rhsMemref, ValueRange{n_block_idx, k_micro_plus_1, zeroIdx}, zeroI8Scalar, inBounds1D);

                                        auto process_w = [&](Value w_bytes, Value lut_vec, Value &res_low_16, Value &res_high_16) {
                                            Value idx_low_w0 = b3.create<arith::AndIOp>(loc4, w_bytes, c03_i8_vec);
                                            Value w_bytes_shr2 = b3.create<arith::ShRUIOp>(loc4, w_bytes, c2_i8_vec);
                                            Value idx_low_w1 = b3.create<arith::OrIOp>(loc4, b3.create<arith::AndIOp>(loc4, w_bytes_shr2, c03_i8_vec), c08_i8_vec);
                                            
                                            Value w_bytes_shr4 = b3.create<arith::ShRUIOp>(loc4, w_bytes, c4_i8_vec);
                                            Value idx_high_w0 = b3.create<arith::AndIOp>(loc4, w_bytes_shr4, c03_i8_vec);
                                            Value w_bytes_shr6 = b3.create<arith::ShRUIOp>(loc4, w_bytes, c6_i8_vec);
                                            Value idx_high_w1 = b3.create<arith::OrIOp>(loc4, b3.create<arith::AndIOp>(loc4, w_bytes_shr6, c03_i8_vec), c08_i8_vec);

                                            Value res_low_w0 = b3.create<func::CallOp>(loc4, "llvm.x86.avx2.pshuf.b", TypeRange{vec32i8}, ValueRange{lut_vec, idx_low_w0}).getResult(0);
                                            Value res_low_w1 = b3.create<func::CallOp>(loc4, "llvm.x86.avx2.pshuf.b", TypeRange{vec32i8}, ValueRange{lut_vec, idx_low_w1}).getResult(0);
                                            
                                            Value res_high_w0 = b3.create<func::CallOp>(loc4, "llvm.x86.avx2.pshuf.b", TypeRange{vec32i8}, ValueRange{lut_vec, idx_high_w0}).getResult(0);
                                            Value res_high_w1 = b3.create<func::CallOp>(loc4, "llvm.x86.avx2.pshuf.b", TypeRange{vec32i8}, ValueRange{lut_vec, idx_high_w1}).getResult(0);
                                            
                                            Value res_low_w0_16 = b3.create<arith::ExtSIOp>(loc4, vec32i16, res_low_w0);
                                            Value res_low_w1_16 = b3.create<arith::ExtSIOp>(loc4, vec32i16, res_low_w1);
                                            res_low_16 = b3.create<arith::AddIOp>(loc4, res_low_w0_16, res_low_w1_16);
                                            
                                            Value res_high_w0_16 = b3.create<arith::ExtSIOp>(loc4, vec32i16, res_high_w0);
                                            Value res_high_w1_16 = b3.create<arith::ExtSIOp>(loc4, vec32i16, res_high_w1);
                                            res_high_16 = b3.create<arith::AddIOp>(loc4, res_high_w0_16, res_high_w1_16);
                                        };

                                        Value res_low_k0_16, res_high_k0_16;
                                        process_w(w_k0, lut_k0, res_low_k0_16, res_high_k0_16);
                                        
                                        Value res_low_k1_16, res_high_k1_16;
                                        process_w(w_k1, lut_k1, res_low_k1_16, res_high_k1_16);

                                        Value sum_low_16 = b3.create<arith::AddIOp>(loc4, res_low_k0_16, res_low_k1_16);
                                        Value sum_high_16 = b3.create<arith::AddIOp>(loc4, res_high_k0_16, res_high_k1_16);

                                        Value next_acc_low_i16 = b3.create<arith::AddIOp>(loc4, acc_low_i16, sum_low_16);
                                        Value next_acc_high_i16 = b3.create<arith::AddIOp>(loc4, acc_high_i16, sum_high_16);

                                        b3.create<scf::YieldOp>(loc4, ValueRange{next_acc_low_i16, next_acc_high_i16});
                                    });
                                
                                Value block_acc_low_i32 = b2.create<arith::ExtSIOp>(loc3, vec32i32, kInnerLoop.getResult(0));
                                Value block_acc_high_i32 = b2.create<arith::ExtSIOp>(loc3, vec32i32, kInnerLoop.getResult(1));

                                Value next_acc_low = b2.create<arith::AddIOp>(loc3, accs_i32[0], block_acc_low_i32);
                                Value next_acc_high = b2.create<arith::AddIOp>(loc3, accs_i32[1], block_acc_high_i32);

                                b2.create<scf::YieldOp>(loc3, ValueRange{next_acc_low, next_acc_high});
                            });

                        Value totalScaleVec = b.create<vector::BroadcastOp>(loc2, vec32f32, totalScale);
                        Value c32Idx = b.create<arith::ConstantIndexOp>(loc2, 32);
                        Value c64Idx = b.create<arith::ConstantIndexOp>(loc2, 64);

                        auto write_out = [&](OpBuilder &b1, Location l1, Value acc_low_i32, Value acc_high_i32, int output_offset_within_block) {
                            Value out_offset = b1.create<arith::AddIOp>(l1, b1.create<arith::MulIOp>(l1, n_block_idx, c64Idx), b1.create<arith::ConstantIndexOp>(l1, output_offset_within_block));
                            Value sum_val_f32 = b1.create<arith::SIToFPOp>(l1, vec32f32, acc_low_i32);
                            Value final_f32 = b1.create<arith::MulFOp>(l1, sum_val_f32, totalScaleVec);
                            b1.create<vector::TransferWriteOp>(l1, final_f32, outMemref, ValueRange{bIdx, sIdx, out_offset});

                            Value out_offset_high = b1.create<arith::AddIOp>(l1, out_offset, c32Idx);
                            Value sum_val_high_f32 = b1.create<arith::SIToFPOp>(l1, vec32f32, acc_high_i32);
                            Value final_f32_high = b1.create<arith::MulFOp>(l1, sum_val_high_f32, totalScaleVec);
                            b1.create<vector::TransferWriteOp>(l1, final_f32_high, outMemref, ValueRange{bIdx, sIdx, out_offset_high});
                        };

                        write_out(b, loc2, kLoop.getResult(0), kLoop.getResult(1), 0);

                        b.create<scf::ReduceOp>(loc2);
                    });
                bS.create<scf::YieldOp>(locS);
            });
            bB.create<scf::YieldOp>(locB);
        });

        rewriter.create<memref::DeallocOp>(loc, lutMemref);
        
        Value finalTensor = rewriter.create<bufferization::ToTensorOp>(loc, resultType, outMemref);
        rewriter.replaceOp(op, finalTensor);

        rewriter.create<memref::DeallocOp>(loc, lhsMemref);

        return success();
    }
};
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
// tenzo.relu2 -> ReLU squared activation: max(0, x)^2
//===----------------------------------------------------------------------===//
struct Relu2LoweringToLinalg : public OpConversionPattern<tenzo::Relu2Op> {
    using OpConversionPattern<tenzo::Relu2Op>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::Relu2Op op, OpAdaptor adaptor,
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
                // relu2(x) = max(0, x)^2
                Value x = args[0];
                Value zero = b.create<arith::ConstantOp>(l, b.getZeroAttr(elemType));
                Value maxVal = b.create<arith::MaximumFOp>(l, x, zero);
                Value res = b.create<arith::MulFOp>(l, maxVal, maxVal);
                b.create<linalg::YieldOp>(l, res);
            });
        return success();
    }
};

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

void tenzo::populateTenzoToLinalgConversionPatterns(RewritePatternSet &patterns) {


    patterns.add<FusedLoweringToLinalg>(patterns.getContext());
    patterns.add<AddLoweringToLinalg>(patterns.getContext());
    patterns.add<ReluLoweringToLinalg>(patterns.getContext());
    patterns.add<Relu2LoweringToLinalg>(patterns.getContext());
    patterns.add<SiLuLoweringToLinalg>(patterns.getContext());
    patterns.add<MulLoweringToLinalg>(patterns.getContext());
    patterns.add<AttentionLoweringToLinalg>(patterns.getContext());
    patterns.add<RopeLoweringToLinalg>(patterns.getContext());
    patterns.add<MatMulLoweringToLinalg>(patterns.getContext());
    patterns.add<MatMulQ8Lowering>(patterns.getContext());
    patterns.add<BitLinearTL1LoweringToLinalg>(patterns.getContext());
    patterns.add<BitLinearTL1PackLoweringToLinalg>(patterns.getContext());
    patterns.add<BitLinearElutLoweringToLinalg>(patterns.getContext());
    patterns.add<Conv2DLoweringToLinalg>(patterns.getContext());
    patterns.add<QuantizeLoweringToLinalg>(patterns.getContext());
    patterns.add<DequantizeLoweringToLinalg>(patterns.getContext());
    patterns.add<KVCacheUpdateLowering>(patterns.getContext());
    patterns.add<RMSNormLoweringToLinalg>(patterns.getContext());
    patterns.add<EmbeddingLoweringToLinalg>(patterns.getContext());
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
        target.addLegalDialect<func::FuncDialect, linalg::LinalgDialect, arith::ArithDialect, math::MathDialect, tensor::TensorDialect, scf::SCFDialect, vector::VectorDialect, bufferization::BufferizationDialect, memref::MemRefDialect>();
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