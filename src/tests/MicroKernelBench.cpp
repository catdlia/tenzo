// MicroKernelBench.cpp - Isolated 6x16 micro-kernel benchmark
// Goal: Measure pure computational performance without cache/tiling overhead

#include "PipelineTests.h"
#include "passes/Passes.h"
#include "context/HardwareInfo.h"
#include "dialect/TenzoDialect.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>

namespace tenzo {

void runMicroKernelBenchmark(mlir::MLIRContext &context) {
    llvm::outs() << "\n╔════════════════════════════════════════════════╗\n";
    llvm::outs() << "║  🔬 MICRO-KERNEL BENCHMARK (6x16)              ║\n";
    llvm::outs() << "║  Pure computational performance test           ║\n";
    llvm::outs() << "╚════════════════════════════════════════════════╝\n\n";

    auto hwInfo = HardwareInfo::detect();
    hwInfo.print();
    llvm::outs() << "\n";

    // Test exact micro-kernel size: 6x16 with K=32
    // This should fit perfectly in registers:
    // - 12 YMM regs for C (6x2 vectors)
    // - 2 YMM regs for B
    // - 1 YMM reg for A (broadcast)
    // - 1 YMM spare
    const char* mlirSource = R"(
        module {
            func.func @main() -> f32 {
                %c2 = arith.constant 2.0 : f32
                %c3 = arith.constant 3.0 : f32
                %c0 = arith.constant 0.0 : f32

                // Exact micro-kernel size
                %A_empty = tensor.empty() : tensor<6x32xf32>
                %B_empty = tensor.empty() : tensor<32x16xf32>
                %C_empty = tensor.empty() : tensor<6x16xf32>

                %A = linalg.fill ins(%c2 : f32) outs(%A_empty : tensor<6x32xf32>) -> tensor<6x32xf32>
                %B = linalg.fill ins(%c3 : f32) outs(%B_empty : tensor<32x16xf32>) -> tensor<32x16xf32>
                %C = linalg.fill ins(%c0 : f32) outs(%C_empty : tensor<6x16xf32>) -> tensor<6x16xf32>

                // Single matmul - should vectorize perfectly
                %result = linalg.matmul
                    ins(%A, %B : tensor<6x32xf32>, tensor<32x16xf32>)
                    outs(%C : tensor<6x16xf32>) -> tensor<6x16xf32>

                %idx0 = arith.constant 0 : index
                %val = tensor.extract %result[%idx0, %idx0] : tensor<6x16xf32>
                return %val : f32
            }
        }
    )";

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(mlirSource), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

    if (!module) {
        llvm::errs() << "Failed to parse MLIR\n";
        return;
    }

    mlir::PassManager pm(&context);

    // No fusion needed (single op)

    // Linalg lowering (no-op for already linalg)

    // Bufferization
    llvm::outs() << "Pipeline: Bufferize -> Affine (6x unroll) -> Vectorize\n";
    addTenzoBufferizationPasses(pm);

    // CRITICAL: Use Affine path with perfect unrolling for 6x16
    HardwareInfo::TileSizes microTiles;
    microTiles.M = 6;
    microTiles.N = 16;
    microTiles.K = 4;

    addTenzoToLLVMPasses(pm, /*enableVectorization=*/true, microTiles,
                         /*enableParallel=*/false, /*useExplicitKernel=*/false);

    if (mlir::failed(pm.run(*module))) {
        llvm::errs() << "Pipeline failed\n";
        return;
    }

    // JIT Setup
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerLLVMDialectTranslation(context);
    mlir::registerBuiltinDialectTranslation(context);

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
    if (!maybeEngine) {
        llvm::errs() << "Failed to create execution engine\n";
        return;
    }
    auto engine = std::move(maybeEngine.get());

    float result = 0.0f;
    void* args[] = {&result};

    // Warmup
    llvm::outs() << "Warming up...\n";
    for (int i = 0; i < 100; i++) {
        (void)engine->invokePacked("main", args);
    }

    // Benchmark: Run micro-kernel millions of times
    const int ITERATIONS = 1000000; // 1M iterations
    llvm::outs() << "Running " << ITERATIONS << " iterations of 6x16x32 micro-kernel...\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        (void)engine->invokePacked("main", args);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Each iteration: 6*16*32 muls + 6*16*32 adds = 6144 + 6144 = 12288 ops
    // Or 6*16*2*32 = 6144 FMAs (count as 2 ops each)
    double totalOps = 2.0 * 6 * 16 * 32 * ITERATIONS;
    double gflops = (totalOps / (totalMs / 1000.0)) / 1e9;

    double theoretical = hwInfo.getTheoreticalPeakGFLOPS();
    double efficiency = (gflops / theoretical) * 100.0;

    llvm::outs() << "\n╔════════════════════════════════════════════════╗\n";
    llvm::outs() << "║  📊 MICRO-KERNEL RESULTS                       ║\n";
    llvm::outs() << "╠════════════════════════════════════════════════╣\n";
    llvm::outs() << "║  Size: 6x16x32                                 ║\n";
    llvm::outs() << "║  Iterations: " << ITERATIONS << "                             ║\n";
    llvm::outs() << "║  Time: " << totalMs << " ms                               ║\n";
    llvm::outs() << "║  Throughput: " << gflops << " GFLOPS              ║\n";
    llvm::outs() << "║  Theoretical Peak: " << theoretical << " GFLOPS       ║\n";
    llvm::outs() << "║  Efficiency: " << efficiency << "%                     ║\n";
    llvm::outs() << "╠════════════════════════════════════════════════╣\n";

    double expected = 32.0 * 2.0 * 3.0; // K * A * B = 192.0
    if (std::abs(result - expected) < 0.001) {
        llvm::outs() << "║  ✅ Result: " << result << " (correct)             ║\n";
    } else {
        llvm::outs() << "║  ❌ Result: " << result << " (expected " << expected << ")    ║\n";
    }
    llvm::outs() << "╚════════════════════════════════════════════════╝\n";

    // Interpretation
    if (efficiency > 80) {
        llvm::outs() << "\n🎉 EXCELLENT! Register allocation is perfect!\n";
        llvm::outs() << "   This is the target performance for the full matmul.\n";
    } else if (efficiency > 50) {
        llvm::outs() << "\n✅ GOOD! Close to optimal, minor improvements possible.\n";
    } else if (efficiency > 20) {
        llvm::outs() << "\n⚠️  MODERATE. Check for register spilling or memory stalls.\n";
    } else {
        llvm::outs() << "\n❌ LOW EFFICIENCY! Something is very wrong:\n";
        llvm::outs() << "   - Register spilling?\n";
        llvm::outs() << "   - Not vectorized?\n";
        llvm::outs() << "   - Memory bandwidth bottleneck?\n";
    }
}

} // namespace tenzo

