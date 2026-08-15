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
                        }
                        b3.create<scf::YieldOp>(loc3, curAccs);
                    });
                    
                // Dequantize and store
                SmallVector<Value, 8> outAccs = kLoop.getResults();"""

replacement = r"""                SmallVector<Value, 8> outAccs(8);
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
                                Value wLoad = b3.create<vector::TransferReadOp>(loc3, vec32i8Type, rhs, ValueRange{n_idx, k_half_idx, chunkOffset}, zeroI8Scalar);
                                
                                Value wLow = b3.create<vector::ExtractStridedSliceOp>(loc3, wLoad, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
                                Value wHigh = b3.create<vector::ExtractStridedSliceOp>(loc3, wLoad, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
                                
                                SmallVector<Value, 2> halves = {wLow, wHigh};
                                
                                for (int h = 0; h < 2; ++h) {
                                    Value wPart = halves[h];
                                    Value idx = (half == 0) ? b3.create<arith::AndIOp>(loc3, wPart, c15_vec) : b3.create<arith::ShRUIOp>(loc3, wPart, c4_vec);
                                    
                                    Value wk_plus1 = b3.create<arith::AndIOp>(loc3, idx, c3_vec);
                                    Value wk = b3.create<arith::SubIOp>(loc3, wk_plus1, c1_vec);
                                    
                                    Value idx_sh = b3.create<arith::ShRUIOp>(loc3, idx, c2_vec);
                                    Value wk1_plus1 = b3.create<arith::AndIOp>(loc3, idx_sh, c3_vec);
                                    Value wk1 = b3.create<arith::SubIOp>(loc3, wk1_plus1, c1_vec);
                                    
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
                    SmallVector<Value, 4> halfAccs = kLoop.getResults();
                    for(int i=0; i<4; ++i) outAccs[half * 4 + i] = halfAccs[i];
                }
                
                // Dequantize and store"""

content = content.replace(target, replacement)

with open("src/passes/LinalgLowering.cpp", "w") as f:
    f.write(content)
