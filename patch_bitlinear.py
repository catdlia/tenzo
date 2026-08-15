import re

with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

# Replace the inner loop generation
target = r"""        auto nLoop = rewriter.create<scf::ForOp>(loc, c0, cn_blocks, c1, ValueRange{emptyOut},
            \[&\]\(OpBuilder &b, Location loc2, Value n_idx, ValueRange args\) \{
                Value outTensor = args\[0\];
                
                SmallVector<Value, 8> innerAccs\(8, zero16i32\);
                
                auto kLoop = b.create<scf::ForOp>\(loc2, c0, k_half, c1, innerAccs,
                    \[&\]\(OpBuilder &b3, Location loc3, Value k_half_idx, ValueRange curAccs\) \{
                        
                        Value k_idx = b3.create<arith::MulIOp>\(loc3, k_half_idx, c2_idx\);
                        Value k_idx_plus_1 = b3.create<arith::AddIOp>\(loc3, k_idx, c1_idx\);
                        
                        Value a_even = b3.create<tensor::ExtractOp>\(loc3, lhsI16, ValueRange\{c0, c0, k_idx\}\);
                        Value a_odd = b3.create<tensor::ExtractOp>\(loc3, lhsI16, ValueRange\{c0, c0, k_idx_plus_1\}\);
                        
                        Value a_even_vec = b3.create<vector::BroadcastOp>\(loc3, vec16i16Type, a_even\);
                        Value a_odd_vec = b3.create<vector::BroadcastOp>\(loc3, vec16i16Type, a_odd\);
                        
                        Value chunkOffset = b3.create<arith::ConstantIndexOp>\(loc3, 0\);
                        Value wLoad = b3.create<vector::TransferReadOp>\(loc3, vec32i8Type, rhs, ValueRange\{n_idx, k_half_idx, chunkOffset\}, zeroI8Scalar\);
                        
                        for \(int chunk = 0; chunk < 2; \+\+chunk\) \{
                            Value c_off = \(chunk == 0\) \? c0 : c16;
                            
                            Value wLoad_chunk = b3.create<vector::ExtractStridedSliceOp>\(loc3, wLoad, ArrayRef<int64_t>\{chunk \* 16\}, ArrayRef<int64_t>\{16\}, ArrayRef<int64_t>\{1\}\);
                            
                            Value idx = b3.create<arith::AndIOp>\(loc3, wLoad_chunk, c15_vec\);
                            Value idx_sh = b3.create<arith::ShRUIOp>\(loc3, wLoad_chunk, c4_vec\);
                            
                            SmallVector<Value, 2> nibbles = \{idx, idx_sh\};
                            
                            for \(int h = 0; h < 2; \+\+h\) \{
                                for \(int nib = 0; nib < 2; \+\+nib\) \{
                                    Value idx = nibbles\[nib\];
                                    
                                    // wk = \(idx & 3\) - 1
                                    Value wk_plus1 = b3.create<arith::AndIOp>\(loc3, idx, c3_vec\);
                                    Value wk = b3.create<arith::SubIOp>\(loc3, wk_plus1, c1_vec\);
                                    
                                    // wk1 = \(\(idx >> 2\) & 3\) - 1
                                    Value idx_sh = b3.create<arith::ShRUIOp>\(loc3, idx, c2_vec\);
                                    Value wk1_plus1 = b3.create<arith::AndIOp>\(loc3, idx_sh, c3_vec\);
                                    Value wk1 = b3.create<arith::SubIOp>\(loc3, wk1_plus1, c1_vec\);
                                    
                                    Value wk_16 = b3.create<arith::ExtSIOp>\(loc3, vec16i16Type, wk\);
                                    Value wk1_16 = b3.create<arith::ExtSIOp>\(loc3, vec16i16Type, wk1\);
                                    
                                    Value m0 = b3.create<arith::MulIOp>\(loc3, a_even_vec, wk_16\);
                                    Value m1 = b3.create<arith::MulIOp>\(loc3, a_odd_vec, wk1_16\);
                                    Value m_sum = b3.create<arith::AddIOp>\(loc3, m0, m1\);
                                    
                                    Value m_sum32 = b3.create<arith::ExtSIOp>\(loc3, vec16i32Type, m_sum\);
                                    
                                    int acc_idx = \(nib \* 4\) \+ \(chunk \* 2\) \+ h;
                                    curAccs\[acc_idx\] = b3.create<arith::AddIOp>\(loc3, curAccs\[acc_idx\], m_sum32\);
                                \}
                            \}
                        \}
                        b3.create<scf::YieldOp>\(loc3, curAccs\);
                    \}\);
                    
                // Dequantize and store
                SmallVector<Value, 8> outAccs = kLoop.getResults\(\);"""

replacement = """        auto nLoop = rewriter.create<scf::ForOp>(loc, c0, cn_blocks, c1, ValueRange{emptyOut},
            [&](OpBuilder &b, Location loc2, Value n_idx, ValueRange args) {
                Value outTensor = args[0];
                
                SmallVector<Value, 8> outerAccs(8, zero16i32);
                Value c128_idx = b.create<arith::ConstantIndexOp>(loc2, 128);
                
                auto kOuterLoop = b.create<scf::ForOp>(loc2, c0, k_half, c128_idx, outerAccs,
                    [&](OpBuilder &b2, Location locOuter, Value k_outer_idx, ValueRange curOuterAccs) {
                        
                        Value zero16i16 = b2.create<arith::ConstantOp>(locOuter, DenseElementsAttr::get(vec16i16Type, b2.getI16IntegerAttr(0)));
                        SmallVector<Value, 8> innerAccs(8, zero16i16);
                        
                        Value k_end = b2.create<arith::AddIOp>(locOuter, k_outer_idx, c128_idx);
                        
                        auto kInnerLoop = b2.create<scf::ForOp>(locOuter, k_outer_idx, k_end, c1, innerAccs,
                            [&](OpBuilder &b3, Location loc3, Value k_half_idx, ValueRange curAccs) {
                                
                                Value k_idx = b3.create<arith::MulIOp>(loc3, k_half_idx, c2_idx);
                                Value k_idx_plus_1 = b3.create<arith::AddIOp>(loc3, k_idx, c1_idx);
                                
                                Value a_even = b3.create<tensor::ExtractOp>(loc3, lhsI16, ValueRange{c0, c0, k_idx});
                                Value a_odd = b3.create<tensor::ExtractOp>(loc3, lhsI16, ValueRange{c0, c0, k_idx_plus_1});
                                
                                Value a_even_vec = b3.create<vector::BroadcastOp>(loc3, vec16i16Type, a_even);
                                Value a_odd_vec = b3.create<vector::BroadcastOp>(loc3, vec16i16Type, a_odd);
                                
                                Value chunkOffset = b3.create<arith::ConstantIndexOp>(loc3, 0);
                                Value wLoad = b3.create<vector::TransferReadOp>(loc3, vec32i8Type, rhs, ValueRange{n_idx, k_half_idx, chunkOffset}, zeroI8Scalar);
                                
                                for (int chunk = 0; chunk < 2; ++chunk) {
                                    Value wLoad_chunk = b3.create<vector::ExtractStridedSliceOp>(loc3, wLoad, ArrayRef<int64_t>{chunk * 16}, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
                                    
                                    Value idx = b3.create<arith::AndIOp>(loc3, wLoad_chunk, c15_vec);
                                    Value idx_sh = b3.create<arith::ShRUIOp>(loc3, wLoad_chunk, c4_vec);
                                    
                                    SmallVector<Value, 2> nibbles = {idx, idx_sh};
                                    
                                    for (int h = 0; h < 2; ++h) {
                                        for (int nib = 0; nib < 2; ++nib) {
                                            Value idx_val = nibbles[nib];
                                            
                                            // wk = (idx_val & 3) - 1
                                            Value wk_plus1 = b3.create<arith::AndIOp>(loc3, idx_val, c3_vec);
                                            Value wk = b3.create<arith::SubIOp>(loc3, wk_plus1, c1_vec);
                                            
                                            // wk1 = ((idx_val >> 2) & 3) - 1
                                            Value idx_val_sh = b3.create<arith::ShRUIOp>(loc3, idx_val, c2_vec);
                                            Value wk1_plus1 = b3.create<arith::AndIOp>(loc3, idx_val_sh, c3_vec);
                                            Value wk1 = b3.create<arith::SubIOp>(loc3, wk1_plus1, c1_vec);
                                            
                                            Value wk_16 = b3.create<arith::ExtSIOp>(loc3, vec16i16Type, wk);
                                            Value wk1_16 = b3.create<arith::ExtSIOp>(loc3, vec16i16Type, wk1);
                                            
                                            Value m0 = b3.create<arith::MulIOp>(loc3, a_even_vec, wk_16);
                                            Value m1 = b3.create<arith::MulIOp>(loc3, a_odd_vec, wk1_16);
                                            Value m_sum = b3.create<arith::AddIOp>(loc3, m0, m1);
                                            
                                            int acc_idx = (nib * 4) + (chunk * 2) + h;
                                            curAccs[acc_idx] = b3.create<arith::AddIOp>(loc3, curAccs[acc_idx], m_sum);
                                        }
                                    }
                                }
                                b3.create<scf::YieldOp>(loc3, curAccs);
                            });
                        
                        SmallVector<Value, 8> newOuterAccs(8);
                        for(int i = 0; i < 8; ++i) {
                            Value ext = b2.create<arith::ExtSIOp>(locOuter, vec16i32Type, kInnerLoop.getResult(i));
                            newOuterAccs[i] = b2.create<arith::AddIOp>(locOuter, curOuterAccs[i], ext);
                        }
                        b2.create<scf::YieldOp>(locOuter, newOuterAccs);
                    });
                    
                // Dequantize and store
                SmallVector<Value, 8> outAccs = kOuterLoop.getResults();"""

new_content = re.sub(target, replacement, content, count=1)
if new_content == content:
    print("Failed to replace!")
else:
    with open("src/passes/LinalgLowering.cpp", "w") as f:
        f.write(new_content)
    print("Success!")
