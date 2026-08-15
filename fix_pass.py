import re

with open('src/passes/LinalgLowering.cpp', 'r') as f:
    content = f.read()

# Remove the old BitLinearTL1LoweringToLinalg
start = content.find('struct BitLinearTL1LoweringToLinalg')
if start != -1:
    end = content.find('};', start) + 2
    content = content[:start] + content[end:]

new_pass = """
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
        
        if (wType.getRank() != 3 || !wType.getElementType().isInteger(8))
            return failure();
            
        int64_t n_blocks = wType.getShape()[0];
        int64_t Kpack = wType.getShape()[1]; // K/2
        int64_t Npack = n_blocks * 64; // N/2
        
        if (K % 2 != 0 || Kpack * 2 != K || N % 2 != 0 || Npack * 2 != N)
            return failure();

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
        Value rhsMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get({n_blocks, Kpack, 64}, rewriter.getI8Type()), weights);
        Value outMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get(resultType.getShape(), elemType), emptyOut);

        Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
        Value cn_blocks_idx = rewriter.create<arith::ConstantIndexOp>(loc, n_blocks);

        auto vec8i32Type = VectorType::get({8}, rewriter.getI32Type());
        Value zero8i32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec8i32Type, rewriter.getI32IntegerAttr(0)));
        auto vec16i8Type = VectorType::get({16}, rewriter.getI8Type());
        Value zeroI8Scalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getI8IntegerAttr(0));
        
        Value c0f_vec16 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i8Type, rewriter.getI8IntegerAttr(15)));
        Value c4_vec16 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i8Type, rewriter.getI8IntegerAttr(4)));
        Value c3_vec16 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i8Type, rewriter.getI8IntegerAttr(3)));
        Value c1_vec16 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i8Type, rewriter.getI8IntegerAttr(1)));
        Value c2_vec16 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i8Type, rewriter.getI8IntegerAttr(2)));

        auto vec8f32Type = VectorType::get({8}, rewriter.getF32Type());
        Value ck_half = rewriter.create<arith::ConstantIndexOp>(loc, Kpack);

        auto nLoop = rewriter.create<scf::ParallelOp>(loc, ValueRange{c0}, ValueRange{cn_blocks_idx}, ValueRange{c1}, ValueRange{},
            [&](OpBuilder &b, Location loc2, ValueRange ivs, ValueRange nArgs) {
                Value n_block = ivs[0];
                Value c128_idx = b.create<arith::ConstantIndexOp>(loc2, 64);
                
                // 128 channels = 16 accumulators of vector<8xi32>
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
                                    Value byteOffset = b3.create<arith::ConstantIndexOp>(loc3, chunk * 16);
                                    Value wLoad_chunk = b3.create<vector::TransferReadOp>(loc3, vec16i8Type, rhsMemref, ValueRange{n_block, k_half_idx, byteOffset}, zeroI8Scalar);
                                    
                                    // Process lower nibble (N = 0, 2, 4...)
                                    Value nibble_lower = b3.create<arith::AndIOp>(loc3, wLoad_chunk, c0f_vec16);
                                    Value w_even_plus_1_l = b3.create<arith::AndIOp>(loc3, nibble_lower, c3_vec16);
                                    Value w_even_l = b3.create<arith::SubIOp>(loc3, w_even_plus_1_l, c1_vec16);
                                    Value w_odd_shifted_l = b3.create<arith::ShRUIOp>(loc3, nibble_lower, c2_vec16);
                                    Value w_odd_plus_1_l = b3.create<arith::AndIOp>(loc3, w_odd_shifted_l, c3_vec16);
                                    Value w_odd_l = b3.create<arith::SubIOp>(loc3, w_odd_plus_1_l, c1_vec16);
                                    
                                    // Process upper nibble (N = 1, 3, 5...)
                                    Value nibble_upper = b3.create<arith::ShRUIOp>(loc3, wLoad_chunk, c4_vec16);
                                    Value nibble_upper_masked = b3.create<arith::AndIOp>(loc3, nibble_upper, c0f_vec16);
                                    Value w_even_plus_1_u = b3.create<arith::AndIOp>(loc3, nibble_upper_masked, c3_vec16);
                                    Value w_even_u = b3.create<arith::SubIOp>(loc3, w_even_plus_1_u, c1_vec16);
                                    Value w_odd_shifted_u = b3.create<arith::ShRUIOp>(loc3, nibble_upper_masked, c2_vec16);
                                    Value w_odd_plus_1_u = b3.create<arith::AndIOp>(loc3, w_odd_shifted_u, c3_vec16);
                                    Value w_odd_u = b3.create<arith::SubIOp>(loc3, w_odd_plus_1_u, c1_vec16);
                                    
                                    for (int sub = 0; sub < 2; ++sub) {
                                        // sub=0 processes elements 0..7
                                        // sub=1 processes elements 8..15
                                        Value w_even_sub_l = b3.create<vector::ExtractStridedSliceOp>(loc3, w_even_l, ArrayRef<int64_t>{sub * 8}, ArrayRef<int64_t>{8}, ArrayRef<int64_t>{1});
                                        Value w_odd_sub_l = b3.create<vector::ExtractStridedSliceOp>(loc3, w_odd_l, ArrayRef<int64_t>{sub * 8}, ArrayRef<int64_t>{8}, ArrayRef<int64_t>{1});
                                        
                                        Value w_even_32_l = b3.create<arith::ExtSIOp>(loc3, vec8i32Type, w_even_sub_l);
                                        Value w_odd_32_l = b3.create<arith::ExtSIOp>(loc3, vec8i32Type, w_odd_sub_l);
                                        Value m_even_l = b3.create<arith::MulIOp>(loc3, a_even_vec, w_even_32_l);
                                        Value m_odd_l = b3.create<arith::MulIOp>(loc3, a_odd_vec, w_odd_32_l);
                                        Value m_sum_l = b3.create<arith::AddIOp>(loc3, m_even_l, m_odd_l);
                                        
                                        Value w_even_sub_u = b3.create<vector::ExtractStridedSliceOp>(loc3, w_even_u, ArrayRef<int64_t>{sub * 8}, ArrayRef<int64_t>{8}, ArrayRef<int64_t>{1});
                                        Value w_odd_sub_u = b3.create<vector::ExtractStridedSliceOp>(loc3, w_odd_u, ArrayRef<int64_t>{sub * 8}, ArrayRef<int64_t>{8}, ArrayRef<int64_t>{1});
                                        
                                        Value w_even_32_u = b3.create<arith::ExtSIOp>(loc3, vec8i32Type, w_even_sub_u);
                                        Value w_odd_32_u = b3.create<arith::ExtSIOp>(loc3, vec8i32Type, w_odd_sub_u);
                                        Value m_even_u = b3.create<arith::MulIOp>(loc3, a_even_vec, w_even_32_u);
                                        Value m_odd_u = b3.create<arith::MulIOp>(loc3, a_odd_vec, w_odd_32_u);
                                        Value m_sum_u = b3.create<arith::AddIOp>(loc3, m_even_u, m_odd_u);
                                        
                                        int acc_idx_l = chunk * 4 + sub * 2;
                                        int acc_idx_u = chunk * 4 + sub * 2 + 1;
                                        
                                        nextAccs[acc_idx_l] = b3.create<arith::AddIOp>(loc3, curAccs[acc_idx_l], m_sum_l);
                                        nextAccs[acc_idx_u] = b3.create<arith::AddIOp>(loc3, curAccs[acc_idx_u], m_sum_u);
                                    }
                                }
                                b3.create<scf::YieldOp>(loc3, nextAccs);
                            });
                        b2.create<scf::YieldOp>(locOuter, kInnerLoop.getResults());
                    });
                
                SmallVector<Value, 16> outAccs = kOuterLoop.getResults();
                Value totalScaleVec = b.create<vector::BroadcastOp>(loc2, vec8f32Type, totalScale);
                
                for (int chunk = 0; chunk < 4; ++chunk) {
                    for (int sub = 0; sub < 2; ++sub) {
                        int acc_idx_l = chunk * 4 + sub * 2;
                        int acc_idx_u = chunk * 4 + sub * 2 + 1;
                        
                        Value accF32_l = b.create<arith::SIToFPOp>(loc2, vec8f32Type, outAccs[acc_idx_l]);
                        Value accScaled_l = b.create<arith::MulFOp>(loc2, accF32_l, totalScaleVec);
                        
                        Value accF32_u = b.create<arith::SIToFPOp>(loc2, vec8f32Type, outAccs[acc_idx_u]);
                        Value accScaled_u = b.create<arith::MulFOp>(loc2, accF32_u, totalScaleVec);
                        
                        // We need to write them back interleaved!
                        // But wait! We can just extract elements and write them.
                        Value n_base = b.create<arith::MulIOp>(loc2, n_block, b.create<arith::ConstantIndexOp>(loc2, 128));
                        Value chunk_offset = b.create<arith::AddIOp>(loc2, n_base, b.create<arith::ConstantIndexOp>(loc2, chunk * 32 + sub * 16));
                        
                        for (int i = 0; i < 8; ++i) {
                            Value iVal = b.create<arith::ConstantIndexOp>(loc2, i);
                            Value val_l = b.create<vector::ExtractElementOp>(loc2, accScaled_l, iVal);
                            Value val_u = b.create<vector::ExtractElementOp>(loc2, accScaled_u, iVal);
                            
                            Value dest_l = b.create<arith::AddIOp>(loc2, chunk_offset, b.create<arith::ConstantIndexOp>(loc2, i * 2));
                            Value dest_u = b.create<arith::AddIOp>(loc2, chunk_offset, b.create<arith::ConstantIndexOp>(loc2, i * 2 + 1));
                            
                            b.create<memref::StoreOp>(loc2, val_l, outMemref, ValueRange{c0, c0, dest_l});
                            b.create<memref::StoreOp>(loc2, val_u, outMemref, ValueRange{c0, c0, dest_u});
                        }
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
content = content[:populate_idx] + new_pass + '\n' + content[populate_idx:]

with open('src/passes/LinalgLowering.cpp', 'w') as f:
    f.write(content)
