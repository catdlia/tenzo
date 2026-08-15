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

