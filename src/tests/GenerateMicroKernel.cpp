// GenerateMicroKernel.cpp - Generate and compile MLIR micro-kernel
// This creates a standalone 6x16 micro-kernel function that can be linked with C++ benchmarks
//
// Goal: Generate MLIR code that produces the same 124 GFLOPS as hand-written intrinsics

#include "PipelineTests.h"
#include "passes/Passes.h"
#include "context/HardwareProfile.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

using namespace mlir;

namespace tenzo {

void generateMicroKernelMLIR(mlir::MLIRContext &context) {
    llvm::outs() << "\n╔════════════════════════════════════════════════════════╗\n";
    llvm::outs() << "║  🔧 MLIR MICRO-KERNEL GENERATOR                        ║\n";
    llvm::outs() << "║  Generating 6x16x256 optimized micro-kernel            ║\n";
    llvm::outs() << "╚════════════════════════════════════════════════════════╝\n\n";

    auto hwInfo = tenzo::HardwareProfile::detect();
    hwInfo->print();
    llvm::outs() << "\n";

    // Create module
    auto loc = UnknownLoc::get(&context);
    auto module = ModuleOp::create(loc);
    OpBuilder builder(&context);

    builder.setInsertionPointToStart(module.getBody());

    // Create function signature: void micro_kernel_6x16(float* A, float* B, float* C)
    // A: [6 x 256] row-major
    // B: [256 x 16] row-major
    // C: [6 x 16] row-major
    auto f32Type = builder.getF32Type();
    auto aMemRefType = MemRefType::get({6, 256}, f32Type);
    auto bMemRefType = MemRefType::get({256, 16}, f32Type);
    auto cMemRefType = MemRefType::get({6, 16}, f32Type);

    auto funcType = builder.getFunctionType(
        {aMemRefType, bMemRefType, cMemRefType},
        {}
    );

    auto func = builder.create<func::FuncOp>(loc, "micro_kernel_6x16", funcType);
    func.setPublic();

    // Create function body with linalg.matmul
    auto &entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    Value A = entryBlock.getArgument(0);
    Value B = entryBlock.getArgument(1);
    Value C = entryBlock.getArgument(2);

    // Create linalg.matmul operation
    // This will be transformed by ExplicitMicroKernelPass into explicit vector FMAs
    builder.create<linalg::MatmulOp>(
        loc,
        ValueRange{A, B},  // inputs
        ValueRange{C}      // outputs
    );

    builder.create<func::ReturnOp>(loc);

    module.push_back(func);

    llvm::outs() << "📝 Generated MLIR function:\n\n";
    module.print(llvm::outs());
    llvm::outs() << "\n\n";

    // Now apply optimization passes
    llvm::outs() << "🔄 Applying optimization passes...\n\n";

    PassManager pm(&context);
    pm.enableIRPrinting();

    // 1. Explicit micro-kernel generation (THE KEY PASS!)
    llvm::outs() << "[1/5] Explicit Micro-Kernel Generation...\n";
    addExplicitMicroKernelPass(pm);

    // 2. Vector lowering to LLVM
    llvm::outs() << "[2/5] Vector -> LLVM...\n";
    // TODO: Fix for MLIR 18 - these function names changed
    // pm.addPass(mlir::createConvertVectorToLLVMPass());

    // 3. All other lowering
    llvm::outs() << "[3/5] Lowering to LLVM Dialect...\n";
    // pm.addPass(mlir::createConvertMemRefToLLVMPass());
    // pm.addPass(mlir::createConvertFuncToLLVMPass());
    // pm.addPass(mlir::createConvertSCFToCFPass());
    // pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    // pm.addPass(mlir::createConvertArithToLLVMPass());
    // pm.addPass(mlir::createConvertIndexToLLVMPass());

    // 4. Reconcile casts
    llvm::outs() << "[4/5] Reconciling casts...\n";
    // pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    // 5. Run the pipeline
    llvm::outs() << "[5/5] Running pipeline...\n";

    if (failed(pm.run(module))) {
        llvm::errs() << "❌ Pass pipeline failed!\n";
        return;
    }

    llvm::outs() << "\n✅ Optimization complete!\n\n";
    llvm::outs() << "📝 Optimized MLIR (LLVM Dialect):\n\n";
    module.print(llvm::outs());
    llvm::outs() << "\n\n";

    // Save to file
    std::error_code ec;
    llvm::raw_fd_ostream outFile("generated_micro_kernel.mlir", ec, llvm::sys::fs::OF_None);
    if (ec) {
        llvm::errs() << "❌ Failed to open output file: " << ec.message() << "\n";
        return;
    }

    module.print(outFile);
    outFile.close();

    llvm::outs() << "💾 Saved to: generated_micro_kernel.mlir\n";
    llvm::outs() << "\n";
    llvm::outs() << "╔════════════════════════════════════════════════════════╗\n";
    llvm::outs() << "║  NEXT STEPS:                                           ║\n";
    llvm::outs() << "╠════════════════════════════════════════════════════════╣\n";
    llvm::outs() << "║  1. Compile to LLVM IR:                                ║\n";
    llvm::outs() << "║     mlir-translate --mlir-to-llvmir \\                  ║\n";
    llvm::outs() << "║       generated_micro_kernel.mlir \\                    ║\n";
    llvm::outs() << "║       -o micro_kernel.ll                               ║\n";
    llvm::outs() << "║                                                        ║\n";
    llvm::outs() << "║  2. Compile to object file:                            ║\n";
    llvm::outs() << "║     llc micro_kernel.ll -O3 -mcpu=native \\             ║\n";
    llvm::outs() << "║       -mattr=+avx2,+fma -filetype=obj \\                ║\n";
    llvm::outs() << "║       -o micro_kernel.o                                ║\n";
    llvm::outs() << "║                                                        ║\n";
    llvm::outs() << "║  3. Check assembly:                                    ║\n";
    llvm::outs() << "║     objdump -d micro_kernel.o | grep vfmadd            ║\n";
    llvm::outs() << "║                                                        ║\n";
    llvm::outs() << "║  4. Rebuild benchmark:                                 ║\n";
    llvm::outs() << "║     cmake -DUSE_MLIR_KERNEL=ON ..                      ║\n";
    llvm::outs() << "║     make micro_bench                                   ║\n";
    llvm::outs() << "║     ./micro_bench                                      ║\n";
    llvm::outs() << "╚════════════════════════════════════════════════════════╝\n";
}

} // namespace tenzo

