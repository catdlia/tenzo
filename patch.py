with open('src/passes/LinalgLowering.cpp', 'r') as f:
    content = f.read()

# Get bit_elut
with open('bit_elut.cpp', 'r') as f:
    bit_elut = f.read()

# Make the optimized TL1 pass
bit_tl1 = """
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

        int64_t K = actType.getShape()[2];
        int64_t N = resultType.getShape()[2];
        int64_t Kpack = wType.getShape()[0]; // K/2
        int64_t Npack = wType.getShape()[1]; // N/2

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
        Value rhsMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get({Kpack, Npack}, rewriter.getI8Type()), weights);
        Value outMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get(resultType.getShape(), elemType), emptyOut);

        Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
        Value cn_blocks = rewriter.create<arith::ConstantIndexOp>(loc, N / 128);

        auto vec8i32Type = VectorType::get({8}, rewriter.getI32Type());
        Value zero8i32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec8i32Type, rewriter.getI32IntegerAttr(0)));
        auto vec32i8Type = VectorType::get({32}, rewriter.getI8Type());
        Value zeroI8Scalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getI8IntegerAttr(0));
        
        Value c0f_vec32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec32i8Type, rewriter.getI8IntegerAttr(15)));
        Value c4_vec32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec32i8Type, rewriter.getI8IntegerAttr(4)));
        Value c3_vec32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec32i8Type, rewriter.getI8IntegerAttr(3)));
        Value c1_vec32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec32i8Type, rewriter.getI8IntegerAttr(1)));
        Value c2_vec32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec32i8Type, rewriter.getI8IntegerAttr(2)));

        auto vec8f32Type = VectorType::get({8}, rewriter.getF32Type());
        Value ck_half = rewriter.create<arith::ConstantIndexOp>(loc, Kpack);

        auto nLoop = rewriter.create<scf::ParallelOp>(loc, ValueRange{c0}, ValueRange{cn_blocks}, ValueRange{c1}, ValueRange{},
            [&](OpBuilder &b, Location loc2, ValueRange ivs, ValueRange nArgs) {
                Value n_idx = b.create<arith::MulIOp>(loc2, ivs[0], b.create<arith::ConstantIndexOp>(loc2, 64));
                Value c128_idx = b.create<arith::ConstantIndexOp>(loc2, 64);
                
                SmallVector<Value, 16> outerAccs(16, zero8i32);
                
                auto kOuterLoop = b.create<scf::ForOp>(loc2, c0, ck_half, c128_idx, outerAccs,
                    [&](OpBuilder &b2, Location locOuter, Value k_outer_idx, ValueRange curOuterAccs) {
                        Value k_outer_end = b2.create<arith::AddIOp>(locOuter, k_outer_idx, c128_idx);
                        
                        auto kInnerLoop = b2.create<scf::ForOp>(locOuter, k_outer_idx, k_outer_end, c1, curOuterAccs,
                            [&](OpBuilder &b3, Location loc3, Value k_half_idx, ValueRange curAccs) {
                                
                                Value k_idx = b3.create<arith::MulIOp>(loc3, k_half_idx, c2);
                                Value k_idx_plus_1 = b3.create<arith::AddIOp>(loc3, k_idx, c1);
                                
                                Value a_even = b3.create<memref::LoadOp>(loc3, lhsMemref, ValueRange{k_idx});
                                Value a_odd = b3.create<memref::LoadOp>(loc3, lhsMemref, ValueRange{k_idx_plus_1});
                                
                                Value a_even_i32 = b3.create<arith::ExtSIOp>(loc3, b3.getI32Type(), a_even);
                                Value a_odd_i32 = b3.create<arith::ExtSIOp>(loc3, b3.getI32Type(), a_odd);
                                
                                Value a_even_vec = b3.create<vector::BroadcastOp>(loc3, vec8i32Type, a_even_i32);
                                Value a_odd_vec = b3.create<vector::BroadcastOp>(loc3, vec8i32Type, a_odd_i32);
                                
                                SmallVector<Value, 16> nextAccs(16);
                                
                                for (int chunk = 0; chunk < 4; ++chunk) {
                                    Value byteOffset = b3.create<arith::AddIOp>(loc3, n_idx, b3.create<arith::ConstantIndexOp>(loc3, chunk * 16));
                                    Value wLoad_chunk = b3.create<vector::TransferReadOp>(loc3, vec32i8Type, rhsMemref, ValueRange{k_half_idx, byteOffset}, zeroI8Scalar);
                                    
                                    Value nibble;
                                    if (chunk % 2 == 0) {
                                        nibble = b3.create<arith::AndIOp>(loc3, wLoad_chunk, c0f_vec32);
                                    } else {
                                        Value w_shifted = b3.create<arith::ShRUIOp>(loc3, wLoad_chunk, c4_vec32);
                                        nibble = b3.create<arith::AndIOp>(loc3, w_shifted, c0f_vec32);
                                    }
                                    
                                    Value w_even_plus_1 = b3.create<arith::AndIOp>(loc3, nibble, c3_vec32);
                                    Value w_even = b3.create<arith::SubIOp>(loc3, w_even_plus_1, c1_vec32);
                                    
                                    Value w_odd_shifted = b3.create<arith::ShRUIOp>(loc3, nibble, c2_vec32);
                                    Value w_odd_plus_1 = b3.create<arith::AndIOp>(loc3, w_odd_shifted, c3_vec32);
                                    Value w_odd = b3.create<arith::SubIOp>(loc3, w_odd_plus_1, c1_vec32);
                                    
                                    for (int sub = 0; sub < 4; ++sub) {
                                        Value w_even_sub = b3.create<vector::ExtractStridedSliceOp>(loc3, w_even, ArrayRef<int64_t>{sub * 8}, ArrayRef<int64_t>{8}, ArrayRef<int64_t>{1});
                                        Value w_odd_sub = b3.create<vector::ExtractStridedSliceOp>(loc3, w_odd, ArrayRef<int64_t>{sub * 8}, ArrayRef<int64_t>{8}, ArrayRef<int64_t>{1});
                                        
                                        Value w_even_32 = b3.create<arith::ExtSIOp>(loc3, vec8i32Type, w_even_sub);
                                        Value w_odd_32 = b3.create<arith::ExtSIOp>(loc3, vec8i32Type, w_odd_sub);
                                        
                                        Value m_even = b3.create<arith::MulIOp>(loc3, a_even_vec, w_even_32);
                                        Value m_odd = b3.create<arith::MulIOp>(loc3, a_odd_vec, w_odd_32);
                                        Value m_sum = b3.create<arith::AddIOp>(loc3, m_even, m_odd);
                                        
                                        int acc_idx = chunk * 4 + sub;
                                        nextAccs[acc_idx] = b3.create<arith::AddIOp>(loc3, curAccs[acc_idx], m_sum);
                                    }
                                }
                                b3.create<scf::YieldOp>(loc3, nextAccs);
                            });
                        b2.create<scf::YieldOp>(locOuter, kInnerLoop.getResults());
                    });
                
                SmallVector<Value, 16> outAccs = kOuterLoop.getResults();
                Value totalScaleVec = b.create<vector::BroadcastOp>(loc2, vec8f32Type, totalScale);
                
                for (int chunk = 0; chunk < 4; ++chunk) {
                    for (int sub = 0; sub < 4; ++sub) {
                        int acc_idx = chunk * 4 + sub;
                        Value accFinal = b.create<arith::SIToFPOp>(loc2, vec8f32Type, outAccs[acc_idx]);
                        Value accScaled = b.create<arith::MulFOp>(loc2, accFinal, totalScaleVec);
                        
                        Value n_base = b.create<arith::MulIOp>(loc2, ivs[0], b.create<arith::ConstantIndexOp>(loc2, 128));
                        Value n_offset = b.create<arith::ConstantIndexOp>(loc2, chunk * 32 + sub * 8);
                        Value n_actual = b.create<arith::AddIOp>(loc2, n_base, n_offset);
                        b.create<vector::TransferWriteOp>(loc2, accScaled, outMemref, ValueRange{c0, c0, n_actual});
                    }
                }
                b.create<scf::ReduceOp>(loc2);
            });
        
        nLoop->setAttr("tenzo.parallelize", rewriter.getUnitAttr());
        Value finalOutTensor = rewriter.create<bufferization::ToTensorOp>(loc, resultType, outMemref);
        rewriter.replaceOp(op, finalOutTensor);
        return success();
    }
};
"""

populate_idx = content.find('void tenzo::populateTenzoToLinalgConversionPatterns(RewritePatternSet &patterns) {')
content = content[:populate_idx] + bit_elut + '\n' + bit_tl1 + '\n' + content[populate_idx:]

reg_str = """void tenzo::populateTenzoToLinalgConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<FusedLoweringToLinalg>(patterns.getContext());
    patterns.add<AddLoweringToLinalg>(patterns.getContext());
    patterns.add<ReluLoweringToLinalg>(patterns.getContext());
    patterns.add<AttentionLoweringToLinalg>(patterns.getContext());
    patterns.add<RopeLoweringToLinalg>(patterns.getContext());
    patterns.add<MatMulLoweringToLinalg>(patterns.getContext());
    patterns.add<Conv2DLoweringToLinalg>(patterns.getContext());
    patterns.add<QuantizeLoweringToLinalg>(patterns.getContext());
    patterns.add<DequantizeLoweringToLinalg>(patterns.getContext());
    patterns.add<BitLinearElutLoweringToLinalg>(patterns.getContext());
    patterns.add<BitLinearTL1LoweringToLinalg>(patterns.getContext());
"""
content = content.replace('void tenzo::populateTenzoToLinalgConversionPatterns(RewritePatternSet &patterns) {\n    patterns.add<FusedLoweringToLinalg>(patterns.getContext());\n    patterns.add<AddLoweringToLinalg>(patterns.getContext());\n    patterns.add<ReluLoweringToLinalg>(patterns.getContext());\n    patterns.add<AttentionLoweringToLinalg>(patterns.getContext());\n    patterns.add<RopeLoweringToLinalg>(patterns.getContext());\n    patterns.add<MatMulLoweringToLinalg>(patterns.getContext());\n    patterns.add<Conv2DLoweringToLinalg>(patterns.getContext());\n    patterns.add<QuantizeLoweringToLinalg>(patterns.getContext());\n    patterns.add<DequantizeLoweringToLinalg>(patterns.getContext());\n', reg_str)

with open('src/passes/LinalgLowering.cpp', 'w') as f:
    f.write(content)
