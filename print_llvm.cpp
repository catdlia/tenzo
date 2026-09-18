
#include "dialect/TenzoDialect.h"
#include "passes/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/PassManager.h"
#include <iostream>

using namespace mlir;

int main() {
    MLIRContext ctx;
    tenzo::registerAllDialects(ctx);
    OpBuilder builder(&ctx);
    auto loc = builder.getUnknownLoc();
    auto f32Type = builder.getF32Type();
    auto i8Type = builder.getI8Type();

    auto aMemRefType = MemRefType::get({4, 16}, f32Type);
    auto bMemRefType = MemRefType::get({16, 16}, f32Type);
    auto cMemRefType = MemRefType::get({4, 4}, i8Type);

    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    auto funcType = builder.getFunctionType({aMemRefType, bMemRefType, cMemRefType}, {});
    auto func = builder.create<func::FuncOp>(loc, "jit_fused_kernel", funcType);
    func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    auto &block = *func.addEntryBlock();
    builder.setInsertionPointToStart(&block);

    auto matmulOp = builder.create<linalg::MatmulOp>(
        loc, ValueRange{block.getArgument(0), block.getArgument(1)},
        ValueRange{block.getArgument(2)});
    matmulOp->setAttr("tenzo.ternary_pack", builder.getUnitAttr());
    builder.create<func::ReturnOp>(loc);

    PassManager pm(&ctx);
    tenzo::MicroKernelParams params;
    params.MR = 6;
    params.NR = 16;
    params.VEC_SIZE = 8;
    tenzo::addExplicitMicroKernelPass(pm, params);
    pm.run(module);

    PassManager llvmPM(&ctx);
    tenzo::TileSizes tiles;
    tiles.M = 4;
    tiles.N = 16;
    tiles.K = 16;
    tenzo::addTenzoToLLVMPasses(llvmPM, false, tiles, false, false);
    llvmPM.run(module);

    module.print(llvm::outs());
    return 0;
}
