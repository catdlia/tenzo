import os

cpp_code = """
LogicalResult matchAndRewrite(tenzo::BitLinearTL1PackOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value act = adaptor.getInput();
    Value weights = adaptor.getWeights(); // [N_blocks, K/2, 16]
    auto actType = mlir::cast<RankedTensorType>(act.getType());
    auto wType = mlir::cast<RankedTensorType>(weights.getType());
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto elemType = resultType.getElementType();

    if (actType.getRank() != 3 || wType.getRank() != 3 || resultType.getRank() != 3) return failure();

    int64_t K = actType.getShape()[2];
    int64_t N = resultType.getShape()[2];
    int64_t n_blocks = wType.getShape()[0];
    int64_t K_half = wType.getShape()[1];
    
    if (K % 2 != 0 || K_half * 2 != K || wType.getShape()[2] != 16 || n_blocks * 32 != N) return failure();

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

    Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c32Idx = rewriter.create<arith::ConstantIndexOp>(loc, 32);
    Value ubK = rewriter.create<arith::ConstantIndexOp>(loc, K);
    Value c128Idx = rewriter.create<arith::ConstantIndexOp>(loc, 128);

    auto act1DType = RankedTensorType::get({K}, elemType);
    Value act1D = rewriter.create<tensor::CollapseShapeOp>(
        loc, act1DType, act, SmallVector<ReassociationIndices>{{0, 1, 2}});

    // Quantize activations to int16 
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

    // Bufferize
    Value actMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get({K}, rewriter.getI16Type()), act1D_q);
    Value weightsMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get(wType.getShape(), rewriter.getI8Type()), weights);
    Value outMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get(resultType.getShape(), elemType), emptyOut);

    // Ensure pshufb function exists
    auto module = op->getParentOfType<ModuleOp>();
    auto vec16i8 = VectorType::get({16}, rewriter.getI8Type());
    if (!module.lookupSymbol("llvm.x86.ssse3.pshuf.b.128")) {
        OpBuilder moduleBuilder(module.getBodyRegion());
        moduleBuilder.create<func::FuncOp>(loc, "llvm.x86.ssse3.pshuf.b.128", 
            rewriter.getFunctionType({vec16i8, vec16i8}, {vec16i8})).setPrivate();
    }

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value ck_half = rewriter.create<arith::ConstantIndexOp>(loc, K_half);
    Value cn_blocks = rewriter.create<arith::ConstantIndexOp>(loc, n_blocks);

    // Allocate LUT buffers
    auto lutLowMemrefType = MemRefType::get({K_half}, vec16i8);
    auto lutHighMemrefType = MemRefType::get({K_half}, vec16i8);
    Value lutLowMemref = rewriter.create<memref::AllocOp>(loc, lutLowMemrefType);
    Value lutHighMemref = rewriter.create<memref::AllocOp>(loc, lutHighMemrefType);

    Value zeros16i8 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i8, rewriter.getI8IntegerAttr(0)));

    // Precompute LUTs
    rewriter.create<scf::ForOp>(loc, c0, ck_half, c1, ValueRange{}, 
        [&](OpBuilder &b, Location l, Value k_half_idx, ValueRange args) {
            Value k_even_idx = b.create<arith::MulIOp>(l, k_half_idx, c2);
            Value k_odd_idx = b.create<arith::AddIOp>(l, k_even_idx, c1);
            Value a0 = b.create<memref::LoadOp>(l, actMemref, ValueRange{k_even_idx});
            Value a1 = b.create<memref::LoadOp>(l, actMemref, ValueRange{k_odd_idx});

            Value a0_i32 = b.create<arith::ExtSIOp>(l, b.getI32Type(), a0);
            Value a1_i32 = b.create<arith::ExtSIOp>(l, b.getI32Type(), a1);

            Value lut_low_vec = zeros16i8;
            Value lut_high_vec = zeros16i8;

            int w0_vals[] = {-1, 0, 1};
            int w1_vals[] = {-1, 0, 1};

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    int w0 = w0_vals[i];
                    int w1 = w1_vals[j];
                    int idx = (w0 + 1) * 3 + (w1 + 1); // 0 to 8

                    Value sum_i32;
                    if (w0 == 0 && w1 == 0) {
                        sum_i32 = b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0));
                    } else if (w0 == 0) {
                        sum_i32 = (w1 == 1) ? a1_i32 : b.create<arith::SubIOp>(l, b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0)), a1_i32);
                    } else if (w1 == 0) {
                        sum_i32 = (w0 == 1) ? a0_i32 : b.create<arith::SubIOp>(l, b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0)), a0_i32);
                    } else {
                        Value term0 = (w0 == 1) ? a0_i32 : b.create<arith::SubIOp>(l, b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0)), a0_i32);
                        Value term1 = (w1 == 1) ? a1_i32 : b.create<arith::SubIOp>(l, b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0)), a1_i32);
                        sum_i32 = b.create<arith::AddIOp>(l, term0, term1);
                    }
                    
                    Value val_low_i32 = b.create<arith::AndIOp>(l, sum_i32, b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0xFF)));
                    Value val_low_i8 = b.create<arith::TruncIOp>(l, b.getI8Type(), val_low_i32);

                    Value c8 = b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(8));
                    Value val_high_i32_shr = b.create<arith::ShRSIOp>(l, sum_i32, c8);
                    Value val_high_i32 = b.create<arith::AndIOp>(l, val_high_i32_shr, b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(0xFF)));
                    Value val_high_i8 = b.create<arith::TruncIOp>(l, b.getI8Type(), val_high_i32);

                    Value idx_val = b.create<arith::ConstantOp>(l, b.getI32IntegerAttr(idx));
                    lut_low_vec = b.create<vector::InsertElementOp>(l, val_low_i8, lut_low_vec, idx_val);
                    lut_high_vec = b.create<vector::InsertElementOp>(l, val_high_i8, lut_high_vec, idx_val);
                }
            }

            b.create<memref::StoreOp>(l, lut_low_vec, lutLowMemref, ValueRange{k_half_idx});
            b.create<memref::StoreOp>(l, lut_high_vec, lutHighMemref, ValueRange{k_half_idx});
            b.create<scf::YieldOp>(l);
        });

    // Execute lookup
    auto vec16i16 = VectorType::get({16}, rewriter.getI16Type());
    auto vec16i32 = VectorType::get({16}, rewriter.getI32Type());
    Value zeros16i32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i32, rewriter.getI32IntegerAttr(0)));
    Value c4_i8_vec = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i8, rewriter.getI8IntegerAttr(4)));
    Value c0F_i8_vec = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec16i8, rewriter.getI8IntegerAttr(0x0F)));

    rewriter.create<scf::ParallelOp>(loc, ValueRange{c0}, ValueRange{cn_blocks}, ValueRange{c1}, ValueRange{},
        [&](OpBuilder &b, Location l, ValueRange ivs, ValueRange nArgs) {
            Value n_block_idx = ivs[0];

            auto kLoop = b.create<scf::ForOp>(l, c0, ck_half, c1, ValueRange{zeros16i32, zeros16i32},
                [&](OpBuilder &b2, Location l2, Value k_half_idx, ValueRange accs_i32) {
                    Value w_bytes = b2.create<vector::TransferReadOp>(l2, vec16i8, weightsMemref, ValueRange{n_block_idx, k_half_idx, c0}, rewriter.create<arith::ConstantOp>(l2, rewriter.getI8IntegerAttr(0)));
                    
                    Value idx_low = b2.create<arith::AndIOp>(l2, w_bytes, c0F_i8_vec);
                    Value w_bytes_shr4 = b2.create<arith::ShRUIOp>(l2, w_bytes, c4_i8_vec);
                    Value idx_high = b2.create<arith::AndIOp>(l2, w_bytes_shr4, c0F_i8_vec);

                    Value lut_low_vec = b2.create<memref::LoadOp>(l2, lutLowMemref, ValueRange{k_half_idx});
                    Value lut_high_vec = b2.create<memref::LoadOp>(l2, lutHighMemref, ValueRange{k_half_idx});

                    // Call pshufb for low and high channels
                    auto callPshufbLowChannelLowByte = b2.create<func::CallOp>(l2, "llvm.x86.ssse3.pshuf.b.128", TypeRange{vec16i8}, ValueRange{lut_low_vec, idx_low});
                    auto callPshufbLowChannelHighByte = b2.create<func::CallOp>(l2, "llvm.x86.ssse3.pshuf.b.128", TypeRange{vec16i8}, ValueRange{lut_high_vec, idx_low});

                    auto callPshufbHighChannelLowByte = b2.create<func::CallOp>(l2, "llvm.x86.ssse3.pshuf.b.128", TypeRange{vec16i8}, ValueRange{lut_low_vec, idx_high});
                    auto callPshufbHighChannelHighByte = b2.create<func::CallOp>(l2, "llvm.x86.ssse3.pshuf.b.128", TypeRange{vec16i8}, ValueRange{lut_high_vec, idx_high});

                    // Combine low and high bytes into i16
                    auto combineBytes = [&](Value lowByte, Value highByte) -> Value {
                        Value low16 = b2.create<arith::ExtUIOp>(l2, vec16i16, lowByte);
                        Value high16 = b2.create<arith::ExtSIOp>(l2, vec16i16, highByte); // signed extension for high byte
                        Value c8 = b2.create<arith::ConstantOp>(l2, DenseElementsAttr::get(vec16i16, b2.getI16IntegerAttr(8)));
                        Value high16_shl = b2.create<arith::ShLIOp>(l2, high16, c8);
                        return b2.create<arith::OrIOp>(l2, low16, high16_shl);
                    };

                    Value vals_low_i16 = combineBytes(callPshufbLowChannelLowByte.getResult(0), callPshufbLowChannelHighByte.getResult(0));
                    Value vals_high_i16 = combineBytes(callPshufbHighChannelLowByte.getResult(0), callPshufbHighChannelHighByte.getResult(0));

                    Value vals_low_i32 = b2.create<arith::ExtSIOp>(l2, vec16i32, vals_low_i16);
                    Value vals_high_i32 = b2.create<arith::ExtSIOp>(l2, vec16i32, vals_high_i16);

                    Value next_acc_low = b2.create<arith::AddIOp>(l2, accs_i32[0], vals_low_i32);
                    Value next_acc_high = b2.create<arith::AddIOp>(l2, accs_i32[1], vals_high_i32);

                    b2.create<scf::YieldOp>(l2, ValueRange{next_acc_low, next_acc_high});
                });

            auto vec16f32 = VectorType::get({16}, rewriter.getF32Type());
            Value totalScaleVec = b.create<vector::BroadcastOp>(l, vec16f32, totalScale);

            auto write_out = [&](Value acc, int offset) {
                Value acc_f32 = b.create<arith::SIToFPOp>(l, vec16f32, acc);
                Value final_f32 = b.create<arith::MulFOp>(l, acc_f32, totalScaleVec);
                Value out_offset = b.create<arith::AddIOp>(l, b.create<arith::MulIOp>(l, n_block_idx, c32Idx), b.create<arith::ConstantIndexOp>(l, offset));
                b.create<vector::TransferWriteOp>(l, final_f32, outMemref, ValueRange{c0, c0, out_offset});
            };

            write_out(kLoop.getResult(0), 0);
            write_out(kLoop.getResult(1), 16);

            b.create<scf::ReduceOp>(l);
        });

    rewriter.create<memref::DeallocOp>(loc, lutLowMemref);
    rewriter.create<memref::DeallocOp>(loc, lutHighMemref);

    Value finalTensor = rewriter.create<bufferization::ToTensorOp>(loc, resultType, outMemref);
    rewriter.replaceOp(op, finalTensor);

    return success();
}
"""

with open("/home/illia/CLionProjects/untitled/src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()
    
# Find BitLinearTL1PackLoweringToLinalg
start_str = "struct BitLinearTL1PackLoweringToLinalg : public OpConversionPattern<tenzo::BitLinearTL1PackOp> {\n    using OpConversionPattern<tenzo::BitLinearTL1PackOp>::OpConversionPattern;"
end_str = "};"

start_idx = content.find(start_str)
if start_idx != -1:
    end_idx = content.find(end_str, start_idx) + len(end_str)
    
    new_struct = start_str + "\n" + cpp_code + "};\n"
    
    new_content = content[:start_idx] + new_struct + content[end_idx:]
    with open("/home/illia/CLionProjects/untitled/src/passes/LinalgLowering.cpp", "w") as f:
        f.write(new_content)
    print("Successfully updated LinalgLowering.cpp")
else:
    print("Could not find struct BitLinearTL1PackLoweringToLinalg in LinalgLowering.cpp")
