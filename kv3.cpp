tenzo.kv_cache_update -> tensor.insert_slice
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

        // Step 2: Normalize and scale elementwise
        AffineMap weightMap = AffineMap::get(rank, 0, {rewriter.getAffineDimExpr(rank - 1)}, rewriter.getContext());

        SmallVector<AffineMap, 4> normMaps = {
            rewriter.getMultiDimIdentityMap(rank),
            redOutMap,
            weightMap,
            rewriter.getMultiDimIdentityMap(rank)
        };

        SmallVector<utils::IteratorType> normIterators(rank, utils::IteratorType::parallel);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op, resultType,
            ValueRange{input, sumSq, weight}, ValueRange{emptyOut},
            normMaps, normIterators,
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value x = args[0];
                Value sSq = args[1];
                Value w = args[2];

                Value meanSq = b.create<arith::DivFOp>(l, sSq, dSizeConst);
                Value meanSqEps = b.create<arith::AddFOp>(l, meanSq, epsConst);
                Value rsqrt = b.create<math::RsqrtOp>(l, meanSqEps);

                Value normX = b.create<arith::MulFOp>(l, x, rsqrt);
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
        Value weight = adaptor.getWeight();

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

                Value embVal = b.create<tensor::ExtractOp>(l, weight, ValueRange{tokenIdx, dIdx});
                b.create<linalg::YieldOp>(l, embVal);
            }
        );
        return success();
    }
};

} // namespace



struct BitLinearTL1LoweringToLinalg : public OpConversionPattern<tenzo::BitLinearTL1Op> {
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

        if (actType.getRank() != 3 || wType.getRank() != 3 || resultType.getRank() != 3) return failure();

        int64_t K = actType.getShape()[2];
        int64_t N = resultType.getShape()[2];
        int64_t n_blocks = wType.getShape()[0];
        int64_t K_half = wType.getShape()[1];
        if (K % 2 != 0 || K_half * 2 != K || wType.getShape()[2] != 64 || n_blocks * 128 != N) return failure();

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

        Value zeroF = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0));
        Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);

        Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value ubK = rewriter.create<arith::ConstantIndexOp>(loc, K);
        Value c128Idx = rewriter.create<arith::ConstantIndexOp>(loc, 128);

        auto act1DType = RankedTensorType::get({K}, elemType);
        Value act1D = rewriter.create<tensor::CollapseShapeOp>(
            loc, act1DType, act, SmallVector<ReassociationIndices>{{0, 1, 2}});

        auto vector128f32 = VectorType::get({128}, rewriter.getF32Type());
        Value zero128f32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vector128f32, rewriter.getF32FloatAttr(0.0f)));

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
        Value scaleActVec = rewriter.create<vector::SplatOp>(loc, vector128f32, scaleAct);
        auto vector128i16 = VectorType::get({128}, rewriter.getI16Type());

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

        // BUFFERIZATION
        Value lhsMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get({K}, rewriter.getI16Type()), act1D_q);
        Value rhsMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get(wType.getShape(), rewriter.getI8Type()), weights);
        Value outMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get(resultType.getShape(), elemType), emptyOut);

        Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
        Value cn_blocks = rewriter.create<arith::ConstantIndexOp>(loc, n_blocks);
        Value ck_half = rewriter.create<arith::ConstantIndexOp>(loc, K_half);

        auto vec64i8 = VectorType::get({64}, rewriter.getI8Type());
        auto vec64i32 = VectorType::get({64}, rewriter.getI32Type());
        auto vec64f32 = VectorType::get({64}, rewriter.getF32Type());

        Value zero64i32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec64i32, rewriter.getI32IntegerAttr(0)));
        Value zeroI8Scalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getI8IntegerAttr(0));

        Value c15_i8 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec64i8, rewriter.getI8IntegerAttr(15)));
        Value c4_i8 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec64i8, rewriter.getI8IntegerAttr(4)));
        Value c3_i8 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec64i8, rewriter.getI8IntegerAttr(3)));
        Value c2_i8 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec64i8, rewriter.getI8IntegerAttr(2)));
        Value c1_i8 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec64i8, rewriter.getI8IntegerAttr(1)));

        auto nLoop = rewriter.create<scf::ParallelOp>(loc, ValueRange{c0}, ValueRange{cn_blocks}, ValueRange{c1}, ValueRange{},
            [&](OpBuilder &b, Location loc2, ValueRange ivs, ValueRange nArgs) {
                Value n_block_idx = ivs[0];
                
                auto kLoop = b.create<scf::ForOp>(loc2, c0, ck_half, c1, ValueRange{zero64i32, zero64i32},
                    [&](OpBuilder &b3, Location loc3, Value k_half_idx, ValueRange accs) {
                        Value acc_low = accs[0];
                        Value acc_high = accs[1];

                        // Load Act[k*2] and Act[k*2+1]
                        Value k_even_idx = b3.create<arith::MulIOp>(loc3, k_half_idx, c2);
                        Value k_odd_idx = b3.create<arith::AddIOp>(loc3, k_even_idx, c1);
                        Value act_even_i16 = b3.create<memref::LoadOp>(loc3, lhsMemref, ValueRange{k_even_idx});
                        Value act_odd_i16 = b3.create<memref::LoadOp>(loc3, lhsMemref, ValueRange{k_odd_idx});
                        Value act_even_i32 = b3.create<arith::ExtSIOp>(loc3, b3.getI32Type(), act_even_i16);
                        Value act_odd_i32 = b3.create<arith::ExtSIOp>(loc3, b3.getI32Type(), act_odd_i16);
                        Value act_even_vec = b3.create<vector::BroadcastOp>(loc3, vec64i32, act_even_i32);
                        Value act_odd_vec = b3.create<vector::BroadcastOp>(loc3, vec64i32, act_odd_i32);

                        // Load weights
                        Value w_bytes = b3.create<vector::TransferReadOp>(loc3, vec64i8, rhsMemref, ValueRange{n_block_idx, k_half_idx, c0}, zeroI8Scalar);

                        // Extract low and high nibbles
                        Value low_nibbles = b3.create<arith::AndIOp>(loc3, w_bytes, c15_i8);
                        Value w_shifted = b3.create<arith::ShRUIOp>(loc3, w_bytes, c4_i8);
                        Value high_nibbles = b3.create<arith::AndIOp>(loc3, w_shifted, c15_i8);

                        // Decode low nibble: w_even = (val & 3) - 1, w_odd = ((val >> 2) & 3) - 1
                        auto decode = [&](Value nibbles) {
                            Value w_even_val = b3.create<arith::AndIOp>(loc3, nibbles, c3_i8);
                            Value w_even_i8 = b3.create<arith::SubIOp>(loc3, w_even_val, c1_i8);
                            Value w_odd_val_shift = b3.create<arith::ShRUIOp>(loc3, nibbles, c2_i8);
                            Value w_odd_val = b3.create<arith::AndIOp>(loc3, w_odd_val_shift, c3_i8);
                            Value w_odd_i8 = b3.create<arith::SubIOp>(loc3, w_odd_val, c1_i8);
                            Value w_even_i32 = b3.create<arith::ExtSIOp>(loc3, vec64i32, w_even_i8);
                            Value w_odd_i32 = b3.create<arith::ExtSIOp>(loc3, vec64i32, w_odd_i8);
                            return std::make_pair(w_even_i32, w_odd_i32);
                        };

                        auto [w_even_low, w_odd_low] = decode(low_nibbles);
                        auto [w_even_high, w_odd_high] = decode(high_nibbles);

                        // Multiply and Accumulate
                        Value m_even_low = b3.create<arith::MulIOp>(loc3, act_even_vec, w_even_low);
                        Value m_odd_low = b3.create<arith::MulIOp>(loc3, act_odd_vec, w_odd_low);
                        Value sum_low = b3.create<arith::AddIOp>(loc3, m_even_low, m_odd_low);
                        Value next_acc_low = b3.create<arith::AddIOp>(loc3, acc_low, sum_low);

                        Value m_even_high = b3.create<arith::MulIOp>(loc3, act_even_vec, w_even_high);
                        Value m_odd_high = b3.create<arith::MulIOp>(loc3, act_odd_vec, w_odd_high);
                        Value sum_high = b3.create<arith::AddIOp>(loc3, m_even_high, m_odd_high);
                        Value next_acc_high = b3.create<arith::AddIOp>(loc3, acc_high, sum_high);

                        b3.create<scf::YieldOp>(loc3, ValueRange{next_acc_low, next_acc_high});
                    });

                Value totalScaleVec = b.create<vector::BroadcastOp>(loc2, vec64f32, totalScale);

                auto write_out = [&](Value acc, int offset) {
                    Value acc_f32 = b.create<arith::SIToFPOp>(loc2, vec64f32, acc);
                    Value final_f32 = b.create<arith::MulFOp>(loc2, acc_f32, totalScaleVec);
                    Value out_offset = b.create<arith::AddIOp>(loc2, b.create<arith::MulIOp>(loc2, n_block_idx, b.create<arith::ConstantIndexOp>(loc2, 128)), b.create<arith::ConstantIndexOp>(loc2, offset));
                    b.create<vector::TransferWriteOp>(loc2, final_f32, outMemref, ValueRange{c0, c0, out_offset});
                };

                write_out(kLoop.getResult(0), 0);
                write_out(kLoop.getResult(1), 64);

                b.create<scf::ReduceOp>(loc2);
            });

        Value finalTensor = rewriter.create<bufferization::ToTensorOp>(loc, resultType, outMemref);
        rewriter.replaceOp(op, finalTensor);

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


