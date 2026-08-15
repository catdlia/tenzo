with open('src/passes/LinalgLowering.cpp', 'r') as f:
    content = f.read()

# 1. Replace ParallelOp with ForOp and remove outMemref entirely
old_memref = "Value outMemref = rewriter.create<bufferization::ToBufferOp>(loc, MemRefType::get(resultType.getShape(), elemType), emptyOut);"
content = content.replace(old_memref, "")

old_loop_start = """auto nLoop = rewriter.create<scf::ParallelOp>(loc, ValueRange{c0}, ValueRange{cn_blocks_idx}, ValueRange{c1}, ValueRange{},
            [&](OpBuilder &b, Location loc2, ValueRange ivs, ValueRange nArgs) {
                Value n_block = ivs[0];"""
new_loop_start = """auto nLoop = rewriter.create<scf::ForOp>(loc, c0, cn_blocks_idx, c1, ValueRange{emptyOut},
            [&](OpBuilder &b, Location loc2, Value n_block, ValueRange nArgs) {
                Value currentOut = nArgs[0];"""
content = content.replace(old_loop_start, new_loop_start)

# 2. Replace memref::StoreOp with tensor::InsertOp
old_store = """                            b.create<memref::StoreOp>(loc2, val_l, outMemref, ValueRange{c0, c0, dest_l});
                            b.create<memref::StoreOp>(loc2, val_u, outMemref, ValueRange{c0, c0, dest_u});"""
new_store = """                            currentOut = b.create<tensor::InsertOp>(loc2, val_l, currentOut, ValueRange{c0, c0, dest_l});
                            currentOut = b.create<tensor::InsertOp>(loc2, val_u, currentOut, ValueRange{c0, c0, dest_u});"""
content = content.replace(old_store, new_store)

# 3. Fix the end of the loop (yield currentOut) and the return value
old_loop_end = """                b.create<scf::ReduceOp>(loc2);
            });
        
        nLoop->setAttr("tenzo.parallelize", rewriter.getUnitAttr());
        Value finalOutTensor = rewriter.create<bufferization::ToTensorOp>(loc, resultType, outMemref);
        rewriter.replaceOp(op, finalOutTensor);"""
new_loop_end = """                b.create<scf::YieldOp>(loc2, currentOut);
            });
        
        nLoop->setAttr("tenzo.parallelize", rewriter.getUnitAttr());
        Value finalOutTensor = nLoop.getResult(0);
        rewriter.replaceOp(op, finalOutTensor);"""
content = content.replace(old_loop_end, new_loop_end)

with open('src/passes/LinalgLowering.cpp', 'w') as f:
    f.write(content)
