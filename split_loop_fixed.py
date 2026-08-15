import re

with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

target = r"""                // Initialize 8 accumulators for 128 channels
                SmallVector<Value, 8> accs(8, zero16i32);
                
                auto kLoop = b.create<scf::ForOp>(loc2, c0, ck_half, c1, accs,
                    [&](OpBuilder &b3, Location loc3, Value k_half_idx, ValueRange kArgs) {
                        SmallVector<Value, 8> curAccs(kArgs.begin(), kArgs.end());
                        
                        Value k_idx = b3.create<arith::MulIOp>(loc3, k_half_idx, c2);
                        Value k_idx_plus_1 = b3.create<arith::AddIOp>(loc3, k_idx, c1);
                        Value a_even = b3.create<tensor::ExtractOp>(loc3, lhsI16, ValueRange{c0, c0, k_idx});
                        Value a_odd = b3.create<tensor::ExtractOp>(loc3, lhsI16, ValueRange{c0, c0, k_idx_plus_1});
                        Value a_even_vec = b3.create<vector::BroadcastOp>(loc3, vec16i16Type, a_even);
                        Value a_odd_vec = b3.create<vector::BroadcastOp>(loc3, vec16i16Type, a_odd);
                        
                        // Process chunk 0 (bytes 0..31) and chunk 1 (bytes 32..63)
                        for(int chunk = 0; chunk < 2; ++chunk) {
                            Value chunkOffset = chunk == 0 ? c0 : c32;
                            Value zeroI8Scalar = b3.create<arith::ConstantOp>(loc3, b3.getI8IntegerAttr(0));
                            Value wLoad = b3.create<vector::TransferReadOp>(loc3, vec32i8Type, rhs, ValueRange{n_idx, k_half_idx, chunkOffset}, zeroI8Scalar);
                            
                            Value c15_vec32 = b3.create<arith::ConstantOp>(loc3, DenseElementsAttr::get(vec32i8Type, b3.getI8IntegerAttr(15)));
                            Value c4_vec32 = b3.create<arith::ConstantOp>(loc3, DenseElementsAttr::get(vec32i8Type, b3.getI8IntegerAttr(4)));
                            Value c3_vec32 = b3.create<arith::ConstantOp>(loc3, DenseElementsAttr::get(vec32i8Type, b3.getI8IntegerAttr(3)));
                            Value c1_vec32 = b3.create<arith::ConstantOp>(loc3, DenseElementsAttr::get(vec32i8Type, b3.getI8IntegerAttr(1)));
                            Value c2_vec32 = b3.create<arith::ConstantOp>(loc3, DenseElementsAttr::get(vec32i8Type, b3.getI8IntegerAttr(2)));
                            
                            // Low nibble (channels 0..63) for 32 bytes
                            Value idx_low32 = b3.create<arith::AndIOp>(loc3, wLoad, c15_vec32);
                            // High nibble (channels 64..127) for 32 bytes
                            Value idx_high32 = b3.create<arith::ShRUIOp>(loc3, wLoad, c4_vec32);
                            
                            SmallVector<Value, 2> nibbles = {idx_low32, idx_high32};
                            for (int nib = 0; nib < 2; ++nib) {
                                Value idx32 = nibbles[nib];
                                
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
                                    
                                    int acc_idx = (nib * 4) + (chunk * 2) + h;
                                    curAccs[acc_idx] = b3.create<arith::AddIOp>(loc3, curAccs[acc_idx], m_sum32);
                                }
                            }
                        }
                        b3.create<scf::YieldOp>(loc3, curAccs);
                    });
                    
                // Dequantize and store
                SmallVector<Value, 8> outAccs = kLoop.getResults();
                Value totalScaleVec = b.create<vector::BroadcastOp>(loc2, vec16f32Type, totalScale);
                
                Value newOutTensor = outTensor;
                for (int acc_idx = 0; acc_idx < 8; ++acc_idx) {
                    Value acc32 = outAccs[acc_idx];
                    Value accF32 = b.create<arith::SIToFPOp>(loc2, vec16f32Type, acc32);
                    Value accFinal = b.create<arith::MulFOp>(loc2, accF32, totalScaleVec);
                    
                    int ch_offset = acc_idx * 16;
                    
                    Value n_offset_val = b.create<arith::ConstantIndexOp>(loc2, ch_offset);
                    Value n_start = b.create<arith::MulIOp>(loc2, n_idx, b.create<arith::ConstantIndexOp>(loc2, 128));
                    Value n_actual = b.create<arith::AddIOp>(loc2, n_start, n_offset_val);
                    
                    // vector.transfer_write
                    newOutTensor = b.create<vector::TransferWriteOp>(loc2, accFinal, newOutTensor, ValueRange{c0, c0, n_actual}).getResult();
                }
                
                b.create<scf::YieldOp>(loc2, newOutTensor);"""

replacement = r"""                Value newOutTensor = outTensor;
                Value totalScaleVec = b.create<vector::BroadcastOp>(loc2, vec16f32Type, totalScale);
                
                for (int half = 0; half < 2; ++half) {
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
                        
                    SmallVector<Value, 4> outAccs = kLoop.getResults();
                    for (int acc_idx = 0; acc_idx < 4; ++acc_idx) {
                        Value acc32 = outAccs[acc_idx];
                        Value accF32 = b.create<arith::SIToFPOp>(loc2, vec16f32Type, acc32);
                        Value accFinal = b.create<arith::MulFOp>(loc2, accF32, totalScaleVec);
                        
                        int ch_offset = (half * 64) + (acc_idx * 16);
                        
                        Value n_offset_val = b.create<arith::ConstantIndexOp>(loc2, ch_offset);
                        Value n_start = b.create<arith::MulIOp>(loc2, n_idx, b.create<arith::ConstantIndexOp>(loc2, 128));
                        Value n_actual = b.create<arith::AddIOp>(loc2, n_start, n_offset_val);
                        
                        newOutTensor = b.create<vector::TransferWriteOp>(loc2, accFinal, newOutTensor, ValueRange{c0, c0, n_actual}).getResult();
                    }
                }
                
                b.create<scf::YieldOp>(loc2, newOutTensor);"""

content = content.replace(target, replacement)
with open("src/passes/LinalgLowering.cpp", "w") as f:
    f.write(content)

