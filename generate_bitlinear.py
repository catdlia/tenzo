import re

cxx_code = r"""
//===----------------------------------------------------------------------===//
// tenzo.bitlinear_tl1 -> manual AVX2 vectorization with vector<16>
//===----------------------------------------------------------------------===//
struct BitLinearTL1LoweringToLinalg : public OpConversionPattern<tenzo::BitLinearTL1Op> {
    using OpConversionPattern<tenzo::BitLinearTL1Op>::OpConversionPattern;

    LogicalResult matchAndRewrite(tenzo::BitLinearTL1Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
        auto elemType = resultType.getElementType();
        auto lhsType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
        
        int64_t B = lhsType.getShape()[0];
        int64_t S = lhsType.getShape()[1];
        int64_t K = lhsType.getShape()[2];
        int64_t N = resultType.getShape()[2];

        Value lhs = adaptor.getInput();
        Value rhs = adaptor.getWeights(); // [N/128, K/2, 64]
        Value scale = adaptor.getScale();
        
        auto f32Type = rewriter.getF32Type();
        auto i32Type = rewriter.getI32Type();
        auto i16Type = rewriter.getI16Type();
        auto i8Type = rewriter.getI8Type();
        
        Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
        Value c16 = rewriter.create<arith::ConstantIndexOp>(loc, 16);
        Value c32 = rewriter.create<arith::ConstantIndexOp>(loc, 32);
        Value cn_blocks = rewriter.create<arith::ConstantIndexOp>(loc, N / 128);
        Value ck_half = rewriter.create<arith::ConstantIndexOp>(loc, K / 2);
        
        Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);
        
        // Fast quantization (dynamic max per token)
        Value c0_f32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, 0.0));
        Value maxAbsInitial = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{B, S}, f32Type);
        Value maxAbsFilled = rewriter.create<linalg::FillOp>(loc, ValueRange{c0_f32}, ValueRange{maxAbsInitial}).getResult(0);
        
        SmallVector<AffineMap, 2> maps = {
            AffineMap::get(3, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(2)}, rewriter.getContext()),
            AffineMap::get(3, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)}, rewriter.getContext())
        };
        SmallVector<utils::IteratorType, 3> iters = {utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction};
        
        auto maxAbsOp = rewriter.create<linalg::GenericOp>(
            loc, TypeRange{maxAbsFilled.getType()}, ValueRange{lhs}, ValueRange{maxAbsFilled},
            maps, iters,
            [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
                Value abs = b.create<math::AbsFOp>(nestedLoc, args[0]);
                Value max = b.create<arith::MaximumFOp>(nestedLoc, abs, args[1]);
                b.create<linalg::YieldOp>(nestedLoc, max);
            });
        Value maxAbsVal = rewriter.create<tensor::ExtractOp>(loc, maxAbsOp.getResult(0), ValueRange{c0, c0});
        
        Value c127_f32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, 127.0));
        Value quantScale = rewriter.create<arith::DivFOp>(loc, c127_f32, maxAbsVal);
        Value totalScale = rewriter.create<arith::DivFOp>(loc, maxAbsVal, c127_f32);
        if (scale) {
            totalScale = rewriter.create<arith::MulFOp>(loc, totalScale, scale);
        }
        
        Value emptyLhsI16 = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{B, S, K}, i16Type);
        auto quantLhsOp = rewriter.create<linalg::GenericOp>(
            loc, TypeRange{emptyLhsI16.getType()}, ValueRange{lhs}, ValueRange{emptyLhsI16},
            SmallVector<AffineMap, 2>{maps[0], maps[0]}, 
            SmallVector<utils::IteratorType, 3>{utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel},
            [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
                Value scaled = b.create<arith::MulFOp>(nestedLoc, args[0], quantScale);
                Value intVal = b.create<arith::FPToSIOp>(nestedLoc, i16Type, scaled);
                b.create<linalg::YieldOp>(nestedLoc, intVal);
            });
        Value lhsI16 = quantLhsOp.getResult(0);

        // Types
        auto vec32i8Type = VectorType::get({32}, i8Type);
        auto vec16i8Type = VectorType::get({16}, i8Type);
        auto vec16i16Type = VectorType::get({16}, i16Type);
        auto vec16i32Type = VectorType::get({16}, i32Type);
        auto vec16f32Type = VectorType::get({16}, f32Type);
        Value zero32i8 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec32i8Type, rewriter.getI8IntegerAttr(0)));
        Value zero16i32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i32Type, rewriter.getI32IntegerAttr(0)));
        
        Value c15_vec = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i8Type, rewriter.getI8IntegerAttr(15)));
        Value c4_vec = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i8Type, rewriter.getI8IntegerAttr(4)));
        Value c3_vec = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i8Type, rewriter.getI8IntegerAttr(3)));
        Value c1_vec = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i8Type, rewriter.getI8IntegerAttr(1)));
        Value c2_vec = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i8Type, rewriter.getI8IntegerAttr(2)));
        
        // Main Loop
        auto nLoop = rewriter.create<scf::ForOp>(loc, c0, cn_blocks, c1, ValueRange{emptyOut},
            [&](OpBuilder &b, Location loc2, Value n_idx, ValueRange nArgs) {
                Value outTensor = nArgs[0];
                
                // Initialize 8 accumulators for 128 channels
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
                            Value wLoad = b3.create<vector::TransferReadOp>(loc3, vec32i8Type, rhs, ValueRange{n_idx, k_half_idx, chunkOffset}, zero32i8);
                            
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
                
                b.create<scf::YieldOp>(loc2, newOutTensor);
            });

        rewriter.replaceOp(op, nLoop.getResult(0));
        return success();
    }
};
"""

with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

# Add includes
if '#include "mlir/Dialect/Vector/IR/VectorOps.h"' not in content:
    content = content.replace('#include "mlir/Dialect/Func/IR/FuncOps.h"',
"""#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
""")

# Add dialect registration
if 'scf::SCFDialect' not in content:
    content = content.replace(
        'target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect, math::MathDialect, tensor::TensorDialect>();',
        'target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect, math::MathDialect, tensor::TensorDialect, scf::SCFDialect, vector::VectorDialect>();'
    )

# Insert the code
if "struct BitLinearTL1LoweringToLinalg" not in content:
    content = content.replace('//===----------------------------------------------------------------------===//\n// tenzo.conv2d', cxx_code + '\n//===----------------------------------------------------------------------===//\n// tenzo.conv2d')
else:
    content = re.sub(r'struct BitLinearTL1LoweringToLinalg : public OpConversionPattern<tenzo::BitLinearTL1Op> \{.*?\n//===----------------------------------------------------------------------===//\n// tenzo.conv2d', cxx_code + '\n//===----------------------------------------------------------------------===//\n// tenzo.conv2d', content, flags=re.DOTALL)

# Insert pattern registration
if "BitLinearTL1LoweringToLinalg" not in content:
    content = content.replace('patterns.add<MatMulLoweringToLinalg>(patterns.getContext());', 
                              'patterns.add<MatMulLoweringToLinalg>(patterns.getContext());\n    patterns.add<BitLinearTL1LoweringToLinalg>(patterns.getContext());')

with open("src/passes/LinalgLowering.cpp", "w") as f:
    f.write(content)

print("Injected BitLinearTL1LoweringToLinalg")
