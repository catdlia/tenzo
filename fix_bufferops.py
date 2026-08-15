with open('src/passes/LinalgLowering.cpp', 'r') as f:
    content = f.read()

# 1. Remove lhsMemref and rhsMemref
lhs_rhs = """        auto act1DQuantTypeMem = RankedTensorType::get({K}, rewriter.getI16Type());
        Value lhsMemref = rewriter.create<tensor::CollapseShapeOp>(loc, MemRefType::get({K}, rewriter.getI16Type()), act1D_q, SmallVector<ReassociationIndices>{{0}});
        Value rhsMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get({n_blocks, Kpack, 64}, rewriter.getI8Type()), weights);"""
content = content.replace(lhs_rhs, "")

# 2. Fix weights read
old_w_read = "Value wLoad_chunk = b3.create<vector::TransferReadOp>(loc3, vec16i8Type, rhsMemref, ValueRange{n_block, k_half_idx, byteOffset}, zeroI8Scalar);"
new_w_read = "Value wLoad_chunk = b3.create<vector::TransferReadOp>(loc3, vec16i8Type, weights, ValueRange{n_block, k_half_idx, byteOffset}, zeroI8Scalar);"
content = content.replace(old_w_read, new_w_read)

# 3. Fix act reads
old_act_read = """                                Value a_even = b3.create<memref::LoadOp>(loc3, lhsMemref, ValueRange{k_idx});
                                Value a_odd = b3.create<memref::LoadOp>(loc3, lhsMemref, ValueRange{k_idx_plus_1});"""
new_act_read = """                                Value a_even = b3.create<tensor::ExtractOp>(loc3, act1D_q, ValueRange{k_idx});
                                Value a_odd = b3.create<tensor::ExtractOp>(loc3, act1D_q, ValueRange{k_idx_plus_1});"""
content = content.replace(old_act_read, new_act_read)

with open('src/passes/LinalgLowering.cpp', 'w') as f:
    f.write(content)
