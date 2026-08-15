import re

with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

target = r"""                        // Process chunk 0 (bytes 0..31) and chunk 1 (bytes 32..63)
                        for(int chunk = 0; chunk < 2; ++chunk) {
                            Value chunkOffset = chunk == 0 ? c0 : c32;
                            Value zeroI8Scalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getI8IntegerAttr(0));
                            Value wLoad = b3.create<vector::TransferReadOp>(loc3, vec32i8Type, rhs, ValueRange{n_idx, k_half_idx, chunkOffset}, zeroI8Scalar);
                            
                            // Split wLoad into wLow (0..15) and wHigh (16..31)
                            Value wLow = b3.create<vector::ExtractStridedSliceOp>(loc3, wLoad, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
                            Value wHigh = b3.create<vector::ExtractStridedSliceOp>(loc3, wLoad, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
                            
                            SmallVector<Value, 2> halves = {wLow, wHigh};
                            
                            for (int h = 0; h < 2; ++h) {
                                Value wPart = halves[h];
                                
                                // Low nibble (channels 0..63)
                                Value idx_low = b3.create<arith::AndIOp>(loc3, wPart, c15_vec);
                                // High nibble (channels 64..127)
                                Value idx_high = b3.create<arith::ShRUIOp>(loc3, wPart, c4_vec);
                                
                                SmallVector<Value, 2> nibbles = {idx_low, idx_high};
                                for (int nib = 0; nib < 2; ++nib) {
                                    Value idx = nibbles[nib];
                                    
                                    // wk = (idx & 3) - 1
                                    Value wk_plus1 = b3.create<arith::AndIOp>(loc3, idx, c3_vec);
                                    Value wk = b3.create<arith::SubIOp>(loc3, wk_plus1, c1_vec);
                                    
                                    // wk1 = ((idx >> 2) & 3) - 1
                                    Value idx_sh = b3.create<arith::ShRUIOp>(loc3, idx, c2_vec);
                                    Value wk1_plus1 = b3.create<arith::AndIOp>(loc3, idx_sh, c3_vec);
                                    Value wk1 = b3.create<arith::SubIOp>(loc3, wk1_plus1, c1_vec);
                                    
                                    Value wk_16 = b3.create<arith::ExtSIOp>(loc3, vec16i16Type, wk);
                                    Value wk1_16 = b3.create<arith::ExtSIOp>(loc3, vec16i16Type, wk1);
                                    
                                    Value m0 = b3.create<arith::MulIOp>(loc3, a_even_vec, wk_16);
                                    Value m1 = b3.create<arith::MulIOp>(loc3, a_odd_vec, wk1_16);
                                    Value m_sum = b3.create<arith::AddIOp>(loc3, m0, m1);
                                    
                                    Value m_sum32 = b3.create<arith::ExtSIOp>(loc3, vec16i32Type, m_sum);
                                    
                                    // Accumulator index
                                    // chunk 0 -> bytes 0..31 -> offsets 0, 16
                                    // chunk 1 -> bytes 32..63 -> offsets 32, 48
                                    // nibble 0 -> 0..63
                                    // nibble 1 -> 64..127
                                    int acc_idx = (nib * 4) + (chunk * 2) + h;
                                    curAccs[acc_idx] = b3.create<arith::AddIOp>(loc3, curAccs[acc_idx], m_sum32);
                                }
                            }
                        }"""

replacement = r"""                        // Process chunk 0 (bytes 0..31) and chunk 1 (bytes 32..63)
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
                        }"""

content = content.replace(target, replacement)
with open("src/passes/LinalgLowering.cpp", "w") as f:
    f.write(content)
