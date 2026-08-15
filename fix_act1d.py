with open('src/passes/LinalgLowering.cpp', 'r') as f:
    content = f.read()

# Fix the broken lhsMemref replacement
broken = """        auto act1DQuantTypeMem = RankedTensorType::get({K}, rewriter.getI16Type());
        Value lhsMemref = rewriter.create<tensor::CollapseShapeOp>(loc, MemRefType::get({K}, rewriter.getI16Type()), act1D_q, SmallVector<ReassociationIndices>{{0}});"""
fixed = "        Value lhsMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get({K}, rewriter.getI16Type()), act1D_q);"
content = content.replace(broken, fixed)

# Now fix act1D correctly
broken_act1d = "Value act1D = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get({K}, rewriter.getF32Type()), act);"
fixed_act1d = """auto act1DType = RankedTensorType::get({K}, elemType);
        Value act1D = rewriter.create<tensor::CollapseShapeOp>(
            loc, act1DType, act, SmallVector<ReassociationIndices>{{0, 1, 2}});"""
content = content.replace(broken_act1d, fixed_act1d)

with open('src/passes/LinalgLowering.cpp', 'w') as f:
    f.write(content)
