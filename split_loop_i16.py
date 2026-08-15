import re

with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

target = r"""                for (int half = 0; half < 2; ++half) {
                    SmallVector<Value, 4> accs(4, zero16i32);
                    auto kLoop = b.create<scf::ForOp>(loc2, c0, ck_half, c1, accs,
                        [&](OpBuilder &b3, Location loc3, Value k_half_idx, ValueRange kArgs) {
                            SmallVector<Value, 4> curAccs(kArgs.begin(), kArgs.end());
                            
                            Value k_idx = b3.create<arith::MulIOp>(loc3, k_half_idx, c2);
                            Value k_idx_plus_1 = b3.create<arith::AddIOp>(loc3, k_idx, c1);
                            Value a_even = b3.create<tensor::ExtractOp>(loc3, lhsI16, ValueRange{c0, c0, k_idx});
                            Value a_odd = b3.create<tensor::ExtractOp>(loc3, lhsI16, ValueRange{c0, c0, k_idx_plus_1});
                            Value a_even_vec = b3.create<vector::BroadcastOp>(loc3, vec16i16Type, a_even);
                            Value a_odd_vec = b3.create<vector::BroadcastOp>(loc3, vec16i16Type, a_odd);
                            
                            for(int chunk = 0; chunk < 2; ++chunk) {
                                Value chunkOffset = chunk == 0 ? c0 : c32;
                                Value zeroI8Scalar = b3.create<arith::ConstantOp>(loc3, b3.getI8IntegerAttr(0));
                                Value wLoad = b3.create<vector::TransferReadOp>(loc3, vec32i8Type, rhs, ValueRange{n_idx, k_half_idx, chunkOffset}, zeroI8Scalar);
                                
                                Value c15_vec32 = b3.create<arith::ConstantOp>(loc3, DenseElementsAttr::get(vec32i8Type, b3.getI8IntegerAttr(15)));
                                Value c4_vec32 = b3.create<arith::ConstantOp>(loc3, DenseElementsAttr::get(vec32i8Type, b3.getI8IntegerAttr(4)));
                                Value c3_vec32 = b3.create<arith::ConstantOp>(loc3, DenseElementsAttr::get(vec32i8Type, b3.getI8IntegerAttr(3)));
                                Value c1_vec32 = b3.create<arith::ConstantOp>(loc3, DenseElementsAttr::get(vec32i8Type, b3.getI8IntegerAttr(1)));
                                Value c2_vec32 = b3.create<arith::ConstantOp>(loc3, DenseElementsAttr::get(vec32i8Type, b3.getI8IntegerAttr(2)));
                                
                                Value idx_low32 = b3.create<arith::AndIOp>(loc3, wLoad, c15_vec32);
                                Value idx_high32 = b3.create<arith::ShRUIOp>(loc3, wLoad, c4_vec32);
                                
                                Value idx32;
                                if (half == 0) idx32 = idx_low32;
                                else idx32 = idx_high32;
                                
                                Value wk_plus1_32 = b3.create<arith::AndIOp>(loc3, idx32, c3_vec32);
                                Value wk_32 = b3.create<arith::SubIOp>(loc3, wk_plus1_32, c1_vec32);
                                
                                Value idx_sh32 = b3.create<arith::ShRUIOp>(loc3, idx32, c2_vec32);
                                Value wk1_plus1_32 = b3.create<arith::AndIOp>(loc3, idx_sh32, c3_vec32);
                                Value wk1_32 = b3.create<arith::SubIOp>(loc3, wk1_plus1_32, c1_vec32);
                                
                                Value wk_low16 = b3.create<vector::ExtractStridedSliceOp>(loc3, wk_32, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
                                Value wk_high16 = b3.create<vector::ExtractStridedSliceOp>(loc3, wk_32, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
                                
                                Value wk1_low16 = b3.create<vector::ExtractStridedSliceOp>(loc3, wk1_32, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
                                Value wk1_high16 = b3.create<vector::ExtractStridedSliceOp>(loc3, wk1_32, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
                                
                                SmallVector<Value, 2> wk_halves = {wk_low16, wk_high16};
                                SmallVector<Value, 2> wk1_halves = {wk1_low16, wk1_high16};
                                
                                for (int h = 0; h < 2; ++h) {
                                    Value wk = wk_halves[h];
                                    Value wk1 = wk1_halves[h];
                                    
                                    Value wk_16 = b3.create<arith::ExtSIOp>(loc3, vec16i16Type, wk);
                                    Value wk1_16 = b3.create<arith::ExtSIOp>(loc3, vec16i16Type, wk1);
                                    
                                    Value m0 = b3.create<arith::MulIOp>(loc3, a_even_vec, wk_16);
                                    Value m1 = b3.create<arith::MulIOp>(loc3, a_odd_vec, wk1_16);
                                    Value m_sum = b3.create<arith::AddIOp>(loc3, m0, m1);
                                    
                                    Value m_sum32 = b3.create<arith::ExtSIOp>(loc3, vec16i32Type, m_sum);
                                    
                                    int acc_idx = (chunk * 2) + h;
                                    curAccs[acc_idx] = b3.create<arith::AddIOp>(loc3, curAccs[acc_idx], m_sum32);
                                }
                            }
                            b3.create<scf::YieldOp>(loc3, curAccs);
                        });
                        
                    SmallVector<Value, 4> outAccs = kLoop.getResults();"""

replacement = r"""                Value c128 = b.create<arith::ConstantIndexOp>(loc2, 128);
                Value zero16i16 = b.create<arith::ConstantOp>(loc2, DenseElementsAttr::get(vec16i16Type, b.getI16IntegerAttr(0)));
                
                for (int half = 0; half < 2; ++half) {
                    SmallVector<Value, 4> accs(4, zero16i32);
                    auto kOuterLoop = b.create<scf::ForOp>(loc2, c0, ck_half, c128, accs,
                        [&](OpBuilder &b3, Location loc3, Value k_outer_idx, ValueRange kOuterArgs) {
                            
                            Value k_end_unclamped = b3.create<arith::AddIOp>(loc3, k_outer_idx, c128);
                            Value is_less = b3.create<arith::CmpIOp>(loc3, arith::CmpIPredicate::slt, k_end_unclamped, ck_half);
                            Value k_end = b3.create<arith::SelectOp>(loc3, is_less, k_end_unclamped, ck_half);
                            
                            SmallVector<Value, 4> innerAccs(4, zero16i16);
                            
                            auto kInnerLoop = b3.create<scf::ForOp>(loc3, k_outer_idx, k_end, c1, innerAccs,
                                [&](OpBuilder &b4, Location loc4, Value k_half_idx, ValueRange kInnerArgs) {
                                    SmallVector<Value, 4> curAccs(kInnerArgs.begin(), kInnerArgs.end());
                                    
                                    Value k_idx = b4.create<arith::MulIOp>(loc4, k_half_idx, c2);
                                    Value k_idx_plus_1 = b4.create<arith::AddIOp>(loc4, k_idx, c1);
                                    Value a_even = b4.create<tensor::ExtractOp>(loc4, lhsI16, ValueRange{c0, c0, k_idx});
                                    Value a_odd = b4.create<tensor::ExtractOp>(loc4, lhsI16, ValueRange{c0, c0, k_idx_plus_1});
                                    Value a_even_vec = b4.create<vector::BroadcastOp>(loc4, vec16i16Type, a_even);
                                    Value a_odd_vec = b4.create<vector::BroadcastOp>(loc4, vec16i16Type, a_odd);
                                    
                                    for(int chunk = 0; chunk < 2; ++chunk) {
                                        Value chunkOffset = chunk == 0 ? c0 : c32;
                                        Value zeroI8Scalar = b4.create<arith::ConstantOp>(loc4, b4.getI8IntegerAttr(0));
                                        Value wLoad = b4.create<vector::TransferReadOp>(loc4, vec32i8Type, rhs, ValueRange{n_idx, k_half_idx, chunkOffset}, zeroI8Scalar);
                                        
                                        Value c15_vec32 = b4.create<arith::ConstantOp>(loc4, DenseElementsAttr::get(vec32i8Type, b4.getI8IntegerAttr(15)));
                                        Value c4_vec32 = b4.create<arith::ConstantOp>(loc4, DenseElementsAttr::get(vec32i8Type, b4.getI8IntegerAttr(4)));
                                        Value c3_vec32 = b4.create<arith::ConstantOp>(loc4, DenseElementsAttr::get(vec32i8Type, b4.getI8IntegerAttr(3)));
                                        Value c1_vec32 = b4.create<arith::ConstantOp>(loc4, DenseElementsAttr::get(vec32i8Type, b4.getI8IntegerAttr(1)));
                                        Value c2_vec32 = b4.create<arith::ConstantOp>(loc4, DenseElementsAttr::get(vec32i8Type, b4.getI8IntegerAttr(2)));
                                        
                                        Value idx_low32 = b4.create<arith::AndIOp>(loc4, wLoad, c15_vec32);
                                        Value idx_high32 = b4.create<arith::ShRUIOp>(loc4, wLoad, c4_vec32);
                                        
                                        Value idx32;
                                        if (half == 0) idx32 = idx_low32;
                                        else idx32 = idx_high32;
                                        
                                        Value wk_plus1_32 = b4.create<arith::AndIOp>(loc4, idx32, c3_vec32);
                                        Value wk_32 = b4.create<arith::SubIOp>(loc4, wk_plus1_32, c1_vec32);
                                        
                                        Value idx_sh32 = b4.create<arith::ShRUIOp>(loc4, idx32, c2_vec32);
                                        Value wk1_plus1_32 = b4.create<arith::AndIOp>(loc4, idx_sh32, c3_vec32);
                                        Value wk1_32 = b4.create<arith::SubIOp>(loc4, wk1_plus1_32, c1_vec32);
                                        
                                        Value wk_low16 = b4.create<vector::ExtractStridedSliceOp>(loc4, wk_32, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
                                        Value wk_high16 = b4.create<vector::ExtractStridedSliceOp>(loc4, wk_32, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
                                        
                                        Value wk1_low16 = b4.create<vector::ExtractStridedSliceOp>(loc4, wk1_32, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
                                        Value wk1_high16 = b4.create<vector::ExtractStridedSliceOp>(loc4, wk1_32, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
                                        
                                        SmallVector<Value, 2> wk_halves = {wk_low16, wk_high16};
                                        SmallVector<Value, 2> wk1_halves = {wk1_low16, wk1_high16};
                                        
                                        for (int h = 0; h < 2; ++h) {
                                            Value wk = wk_halves[h];
                                            Value wk1 = wk1_halves[h];
                                            
                                            Value wk_16 = b4.create<arith::ExtSIOp>(loc4, vec16i16Type, wk);
                                            Value wk1_16 = b4.create<arith::ExtSIOp>(loc4, vec16i16Type, wk1);
                                            
                                            Value m0 = b4.create<arith::MulIOp>(loc4, a_even_vec, wk_16);
                                            Value m1 = b4.create<arith::MulIOp>(loc4, a_odd_vec, wk1_16);
                                            Value m_sum = b4.create<arith::AddIOp>(loc4, m0, m1);
                                            
                                            int acc_idx = (chunk * 2) + h;
                                            curAccs[acc_idx] = b4.create<arith::AddIOp>(loc4, curAccs[acc_idx], m_sum);
                                        }
                                    }
                                    b4.create<scf::YieldOp>(loc4, curAccs);
                                });
                            
                            SmallVector<Value, 4> res16 = kInnerLoop.getResults();
                            SmallVector<Value, 4> nextOuterAccs(4);
                            for (int i=0; i<4; ++i) {
                                Value ext32 = b3.create<arith::ExtSIOp>(loc3, vec16i32Type, res16[i]);
                                nextOuterAccs[i] = b3.create<arith::AddIOp>(loc3, kOuterArgs[i], ext32);
                            }
                            b3.create<scf::YieldOp>(loc3, nextOuterAccs);
                        });
                        
                    SmallVector<Value, 4> outAccs = kOuterLoop.getResults();"""

content = content.replace(target, replacement)
with open("src/passes/LinalgLowering.cpp", "w") as f:
    f.write(content)

