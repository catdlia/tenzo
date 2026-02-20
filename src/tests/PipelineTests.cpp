#include "PipelineTests.h"
#include "passes/Passes.h"
#include "dialect/TenzoDialect.h"
#include "context/HardwareInfo.h"
#include "context/TenzoContext.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cmath>
#include <thread>
#include <atomic>
#include <vector>

namespace tenzo {

//===----------------------------------------------------------------------===//
// Compile with OpenMP parallelism enabled
// This generates code that uses OpenMP for parallel loops
//===----------------------------------------------------------------------===//
std::unique_ptr<mlir::ExecutionEngine> compileWithOpenMP(
    mlir::MLIRContext &context,
    bool enableParallel = true) {

    const char* mlirSource = R"(
        module {
            func.func @main() -> f32 {
                %c2 = arith.constant 2.0 : f32
                %c3 = arith.constant 3.0 : f32
                %c0 = arith.constant 0.0 : f32
                %A_empty = tensor.empty() : tensor<512x512xf32>
                %B_empty = tensor.empty() : tensor<512x512xf32>
                %C_empty = tensor.empty() : tensor<512x512xf32>
                %A = linalg.fill ins(%c2 : f32) outs(%A_empty : tensor<512x512xf32>) -> tensor<512x512xf32>
                %B = linalg.fill ins(%c3 : f32) outs(%B_empty : tensor<512x512xf32>) -> tensor<512x512xf32>
                %C = linalg.fill ins(%c0 : f32) outs(%C_empty : tensor<512x512xf32>) -> tensor<512x512xf32>
                %result = linalg.matmul
                    ins(%A, %B : tensor<512x512xf32>, tensor<512x512xf32>)
                    outs(%C : tensor<512x512xf32>) -> tensor<512x512xf32>
                %idx0 = arith.constant 0 : index
                %val = tensor.extract %result[%idx0, %idx0] : tensor<512x512xf32>
                return %val : f32
            }
        }
    )";

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(mlirSource), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) return nullptr;

    mlir::PassManager pm(&context);

    // 1. Fusion
    {
        mlir::RewritePatternSet patterns(&context);
        populateFusionPatterns(patterns);
        (void)mlir::applyPatternsAndFoldGreedily(module.get(), std::move(patterns));
    }

    // 2. Linalg Lowering
    {
        mlir::RewritePatternSet patterns(&context);
        populateTenzoToLinalgConversionPatterns(patterns);
        mlir::ConversionTarget target(context);
        target.addLegalDialect<mlir::linalg::LinalgDialect, mlir::arith::ArithDialect,
                               mlir::tensor::TensorDialect, mlir::func::FuncDialect>();
        target.addIllegalDialect<tenzo::TenzoDialect>();
        (void)mlir::applyPartialConversion(*module, target, std::move(patterns));
    }

    // 3. Vectorization - Skip for parallel path (handled in LLVMLowering)
    // The parallel path in LLVMLowering does its own Affine vectorization
    if (!enableParallel) {
        addTenzoVectorizationPass(pm);
    }

    // 4. Bufferization
    addTenzoBufferizationPasses(pm);

    // 5. LLVM Lowering
    // For parallel path: enableVectorization=true because LLVMLowering handles it
    auto hwInfo = HardwareInfo::detect();
    auto tiles = hwInfo.getOptimalTileSizes();
    addTenzoToLLVMPasses(pm, /*enableVectorization=*/true, tiles, /*enableParallel=*/enableParallel);

    if (mlir::failed(pm.run(*module))) {
        llvm::errs() << "Pipeline failed\n";
        return nullptr;
    }

    // JIT Setup with OpenMP support
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerLLVMDialectTranslation(context);
    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerOpenMPDialectTranslation(context);  // Register OpenMP translation!

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

    // Link with OpenMP runtime (for parallel execution)
    // Note: The generated code will use OpenMP pragmas that link at runtime

    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
    if (!maybeEngine) {
        llvm::errs() << "Failed to create execution engine: "
                     << llvm::toString(maybeEngine.takeError()) << "\n";
        return nullptr;
    }

    return std::move(maybeEngine.get());
}

std::unique_ptr<mlir::ExecutionEngine> compileAndGetEngine(mlir::MLIRContext &context, bool enableVectorization) {
    // SCALE UP: Матриці 512x512
    // Очікуваний результат: 512 * 2.0 * 3.0 = 3072.0
    const char* mlirSource = R"(
        module {
            func.func @main() -> f32 {
                %c2 = arith.constant 2.0 : f32
                %c3 = arith.constant 3.0 : f32
                %c0 = arith.constant 0.0 : f32
                %A_empty = tensor.empty() : tensor<512x512xf32>
                %B_empty = tensor.empty() : tensor<512x512xf32>
                %C_empty = tensor.empty() : tensor<512x512xf32>
                %A = linalg.fill ins(%c2 : f32) outs(%A_empty : tensor<512x512xf32>) -> tensor<512x512xf32>
                %B = linalg.fill ins(%c3 : f32) outs(%B_empty : tensor<512x512xf32>) -> tensor<512x512xf32>
                %C = linalg.fill ins(%c0 : f32) outs(%C_empty : tensor<512x512xf32>) -> tensor<512x512xf32>
                %result = linalg.matmul
                    ins(%A, %B : tensor<512x512xf32>, tensor<512x512xf32>)
                    outs(%C : tensor<512x512xf32>) -> tensor<512x512xf32>
                %idx0 = arith.constant 0 : index
                %val = tensor.extract %result[%idx0, %idx0] : tensor<512x512xf32>
                return %val : f32
            }
        }
    )";
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(mlirSource), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) return nullptr;
    mlir::PassManager pm(&context);
    // 1. Fusion
    {
        mlir::RewritePatternSet patterns(&context);
        populateFusionPatterns(patterns);
        (void)mlir::applyPatternsAndFoldGreedily(module.get(), std::move(patterns));
    }
    // 2. Linalg Lowering (Tenzo -> Linalg)
    {
        mlir::RewritePatternSet patterns(&context);
        populateTenzoToLinalgConversionPatterns(patterns);
        mlir::ConversionTarget target(context);
        target.addLegalDialect<mlir::linalg::LinalgDialect, mlir::arith::ArithDialect, mlir::tensor::TensorDialect, mlir::func::FuncDialect>();
        target.addIllegalDialect<tenzo::TenzoDialect>();
        (void)mlir::applyPartialConversion(*module, target, std::move(patterns));
    }
    // 3. Vectorization / Optimization Strategy
    if (enableVectorization) {
        addTenzoVectorizationPass(pm);
    }
    // 4. Bufferization & Backend
    addTenzoBufferizationPasses(pm);

    // Get hardware-optimal tile sizes and pass to LLVM lowering
    auto hwInfo = HardwareInfo::detect();
    auto tiles = hwInfo.getOptimalTileSizes();
    addTenzoToLLVMPasses(pm, enableVectorization, tiles, /*enableParallel=*/false);

    if (mlir::failed(pm.run(*module))) {
        llvm::errs() << "Pipeline failed\n";
        return nullptr;
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
        return nullptr;
    }
    return std::move(maybeEngine.get());
}
void runBenchmark(mlir::MLIRContext &context) {
    llvm::outs() << "\n=========================================\n";
    llvm::outs() << "🚀 BENCHMARK: 512x512 MatMul\n";
    llvm::outs() << "=========================================\n\n";
    auto hwInfo = HardwareInfo::detect();
    hwInfo.print();
    llvm::outs() << "\n";
    const int ITERATIONS = 100;
    llvm::outs() << "Compiling Scalar version...\n";
    auto scalarEngine = compileAndGetEngine(context, false);
    llvm::outs() << "Compiling Vector version...\n";
    auto vectorEngine = compileAndGetEngine(context, true);
    if (!scalarEngine || !vectorEngine) {
        llvm::errs() << "Engine creation failed.\n";
        return;
    }
    float scalarRes = 0.0f, vectorRes = 0.0f;
    void* scalarArgs[] = { &scalarRes };
    void* vectorArgs[] = { &vectorRes };
    // Warmup (multiple iterations for stable results)
    llvm::outs() << "Warming up...\n";
    for(int i = 0; i < 10; ++i) {
        (void)scalarEngine->invokePacked("main", scalarArgs);
        (void)vectorEngine->invokePacked("main", vectorArgs);
    }
    // Benchmark Scalar
    llvm::outs() << "Running Scalar (LLVM O3)...\n";
    auto startScalar = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < ITERATIONS; ++i) {
        if(scalarEngine->invokePacked("main", scalarArgs)) break;
    }
    auto endScalar = std::chrono::high_resolution_clock::now();
    // Benchmark Vector
    llvm::outs() << "Running Vector (Tenzo Optimized)...\n";
    auto startVector = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < ITERATIONS; ++i) {
        if(vectorEngine->invokePacked("main", vectorArgs)) break;
    }
    auto endVector = std::chrono::high_resolution_clock::now();
    auto scalarDur = std::chrono::duration_cast<std::chrono::milliseconds>(endScalar - startScalar).count();
    auto vectorDur = std::chrono::duration_cast<std::chrono::milliseconds>(endVector - startVector).count();

    // Calculate GFLOPS: 2 * N^3 * iterations
    double totalOps = 2.0 * 512 * 512 * 512 * ITERATIONS;
    double scalarGflops = scalarDur > 0 ? (totalOps / (scalarDur / 1000.0)) / 1e9 : 0;
    double vectorGflops = vectorDur > 0 ? (totalOps / (vectorDur / 1000.0)) / 1e9 : 0;

    llvm::outs() << "\n📊 RESULTS (512x512, " << ITERATIONS << " runs):\n";
    llvm::outs() << "Scalar: " << scalarDur << " ms (" << scalarGflops << " GFLOPS)\n";
    llvm::outs() << "Vector: " << vectorDur << " ms (" << vectorGflops << " GFLOPS)\n";
    if (vectorDur > 0 && vectorDur < scalarDur) {
        double speedup = (double)scalarDur / vectorDur;
        llvm::outs() << "⚡ Speedup: " << speedup << "x faster!\n";
    } else {
        llvm::outs() << "🐢 No speedup yet (Need Affine Tiling!)\n";
    }
    llvm::outs() << "Check: " << vectorRes << " (Expected 3072.0)\n";
}

void runConv2DTest(mlir::MLIRContext &context) {
    llvm::outs() << "\n=========================================\n";
    llvm::outs() << "🎯 Conv2D Benchmark (NHWC format)\n";
    llvm::outs() << "=========================================\n\n";

    // Conv2D: Input[1,32,32,3] * Filter[3,3,3,64] -> Output[1,30,30,64]
    // More realistic: 32x32 RGB image with 64 output channels
    const char* mlirSource = R"(
        module {
            func.func @main() -> f32 {
                %c1 = arith.constant 1.0 : f32
                %c0 = arith.constant 0.0 : f32

                // Input: [1, 32, 32, 3]
                %input_empty = tensor.empty() : tensor<1x32x32x3xf32>
                %input = linalg.fill ins(%c1 : f32) outs(%input_empty : tensor<1x32x32x3xf32>) -> tensor<1x32x32x3xf32>

                // Filter: [3, 3, 3, 64]
                %filter_empty = tensor.empty() : tensor<3x3x3x64xf32>
                %filter = linalg.fill ins(%c1 : f32) outs(%filter_empty : tensor<3x3x3x64xf32>) -> tensor<3x3x3x64xf32>

                // Output: [1, 30, 30, 64]
                %output_empty = tensor.empty() : tensor<1x30x30x64xf32>
                %output = linalg.fill ins(%c0 : f32) outs(%output_empty : tensor<1x30x30x64xf32>) -> tensor<1x30x30x64xf32>

                %result = linalg.conv_2d_nhwc_hwcf {
                    dilations = dense<1> : tensor<2xi64>,
                    strides = dense<1> : tensor<2xi64>
                } ins(%input, %filter : tensor<1x32x32x3xf32>, tensor<3x3x3x64xf32>)
                  outs(%output : tensor<1x30x30x64xf32>) -> tensor<1x30x30x64xf32>

                %idx0 = arith.constant 0 : index
                %val = tensor.extract %result[%idx0, %idx0, %idx0, %idx0] : tensor<1x30x30x64xf32>
                return %val : f32
            }
        }
    )";

    llvm::outs() << "Input: [1, 32, 32, 3] (NHWC)\n";
    llvm::outs() << "Filter: [3, 3, 3, 64] (HWIO)\n";
    llvm::outs() << "Output: [1, 30, 30, 64] (NHWC)\n";
    llvm::outs() << "Operations: ~50M MACs\n\n";

    auto hwInfo = HardwareInfo::detect();
    hwInfo.print();
    llvm::outs() << "\n";

    // Compile optimized version
    llvm::outs() << "Compiling optimized Conv2D...\n";

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(mlirSource), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

    if (!module) {
        llvm::errs() << "Failed to parse MLIR\n";
        return;
    }

    mlir::PassManager pm(&context);
    addTenzoVectorizationPass(pm);
    addTenzoBufferizationPasses(pm);
    auto tiles = hwInfo.getOptimalTileSizes();
    addTenzoToLLVMPasses(pm, true, tiles);

    if (mlir::failed(pm.run(*module))) {
        llvm::errs() << "Pipeline failed\n";
        return;
    }

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
    for (int i = 0; i < 5; i++) {
        (void)engine->invokePacked("main", args);
    }

    // Benchmark
    const int ITERATIONS = 100;
    llvm::outs() << "Running benchmark (" << ITERATIONS << " iterations)...\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        (void)engine->invokePacked("main", args);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double msPerIter = (double)totalMs / ITERATIONS;

    // Calculate GFLOPS: Conv2D FLOPs = 2 * Out_H * Out_W * K_H * K_W * In_C * Out_C
    // = 2 * 30 * 30 * 3 * 3 * 3 * 64 = 31,104,000 FLOPs per conv
    double flops = 2.0 * 30 * 30 * 3 * 3 * 3 * 64;
    double gflops = (flops / msPerIter) / 1e6;  // GFLOPS

    llvm::outs() << "\n📊 RESULTS:\n";
    llvm::outs() << "Total time: " << totalMs << " ms\n";
    llvm::outs() << "Per iteration: " << msPerIter << " ms\n";
    llvm::outs() << "Throughput: " << gflops << " GFLOPS\n";
    llvm::outs() << "Result: " << result << " (Expected 27.0)\n";

    if (std::abs(result - 27.0f) < 0.001f) {
        llvm::outs() << "✅ Conv2D correct!\n";
    } else {
        llvm::outs() << "❌ Result mismatch!\n";
    }
}

void runSmallMatMulTest(mlir::MLIRContext &context) {
    llvm::outs() << "\n=========================================\n";
    llvm::outs() << "🔬 Small MatMul Test (64x64) - Explicit Vectorization\n";
    llvm::outs() << "=========================================\n\n";

    // 64x64 matrix - small enough for explicit vectorization
    // Expected result: 64 * 2.0 * 3.0 = 384.0
    const char* mlirSource = R"(
        module {
            func.func @main() -> f32 {
                %c2 = arith.constant 2.0 : f32
                %c3 = arith.constant 3.0 : f32
                %c0 = arith.constant 0.0 : f32
                %A_empty = tensor.empty() : tensor<64x64xf32>
                %B_empty = tensor.empty() : tensor<64x64xf32>
                %C_empty = tensor.empty() : tensor<64x64xf32>
                %A = linalg.fill ins(%c2 : f32) outs(%A_empty : tensor<64x64xf32>) -> tensor<64x64xf32>
                %B = linalg.fill ins(%c3 : f32) outs(%B_empty : tensor<64x64xf32>) -> tensor<64x64xf32>
                %C = linalg.fill ins(%c0 : f32) outs(%C_empty : tensor<64x64xf32>) -> tensor<64x64xf32>
                %result = linalg.matmul
                    ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>)
                    outs(%C : tensor<64x64xf32>) -> tensor<64x64xf32>
                %idx0 = arith.constant 0 : index
                %val = tensor.extract %result[%idx0, %idx0] : tensor<64x64xf32>
                return %val : f32
            }
        }
    )";

    auto hwInfo = HardwareInfo::detect();
    hwInfo.print();
    llvm::outs() << "\n";

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(mlirSource), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) { llvm::errs() << "Parse failed\n"; return; }

    // Compile with vectorization
    mlir::PassManager pm(&context);

    // Fusion
    {
        mlir::RewritePatternSet patterns(&context);
        populateFusionPatterns(patterns);
        (void)mlir::applyPatternsAndFoldGreedily(module.get(), std::move(patterns));
    }

    // Linalg lowering
    {
        mlir::RewritePatternSet patterns(&context);
        populateTenzoToLinalgConversionPatterns(patterns);
        mlir::ConversionTarget target(context);
        target.addLegalDialect<mlir::linalg::LinalgDialect, mlir::arith::ArithDialect,
                               mlir::tensor::TensorDialect, mlir::func::FuncDialect>();
        target.addIllegalDialect<tenzo::TenzoDialect>();
        (void)mlir::applyPartialConversion(*module, target, std::move(patterns));
    }

    // Vectorization
    addTenzoVectorizationPass(pm);

    // Bufferization
    addTenzoBufferizationPasses(pm);

    // LLVM lowering
    auto tiles = hwInfo.getOptimalTileSizes();
    addTenzoToLLVMPasses(pm, true, tiles);

    if (mlir::failed(pm.run(*module))) {
        llvm::errs() << "Pipeline failed\n";
        return;
    }

    // JIT
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerLLVMDialectTranslation(context);
    mlir::registerBuiltinDialectTranslation(context);

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
    if (!maybeEngine) { llvm::errs() << "Engine failed\n"; return; }
    auto engine = std::move(maybeEngine.get());

    // Benchmark
    float result = 0.0f;
    void* args[] = {&result};

    // Warmup
    for (int i = 0; i < 10; i++) {
        (void)engine->invokePacked("main", args);
    }

    const int ITERATIONS = 10000;
    llvm::outs() << "Running " << ITERATIONS << " iterations...\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        (void)engine->invokePacked("main", args);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double msPerIter = (double)totalMs / ITERATIONS;

    // Calculate GFLOPS: 2*M*N*K = 2*64*64*64 = 524288
    double flops = 2.0 * 64 * 64 * 64;
    double gflops = (flops / msPerIter) / 1e6;

    llvm::outs() << "\n📊 RESULTS (64x64 MatMul):\n";
    llvm::outs() << "Total: " << totalMs << " ms\n";
    llvm::outs() << "Per iteration: " << msPerIter << " ms\n";
    llvm::outs() << "Throughput: " << gflops << " GFLOPS\n";
    llvm::outs() << "Result: " << result << " (Expected 384.0)\n";

    if (std::abs(result - 384.0f) < 0.1f) {
        llvm::outs() << "✅ Result correct!\n";
    } else {
        llvm::outs() << "❌ Result incorrect!\n";
    }
}

void runTransformDialectTest(mlir::MLIRContext &context) {
    llvm::outs() << "\n=========================================\n";
    llvm::outs() << "🔄 Transform Dialect Test (8x8 MatMul)\n";
    llvm::outs() << "=========================================\n\n";

    // Use 8x8 matrix - small enough for direct vectorization
    const char* mlirSource = R"(
        module {
            func.func @main() -> f32 {
                %c2 = arith.constant 2.0 : f32
                %c3 = arith.constant 3.0 : f32
                %c0 = arith.constant 0.0 : f32
                %A_empty = tensor.empty() : tensor<8x8xf32>
                %B_empty = tensor.empty() : tensor<8x8xf32>
                %C_empty = tensor.empty() : tensor<8x8xf32>
                %A = linalg.fill ins(%c2 : f32) outs(%A_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
                %B = linalg.fill ins(%c3 : f32) outs(%B_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
                %C = linalg.fill ins(%c0 : f32) outs(%C_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
                %result = linalg.matmul
                    ins(%A, %B : tensor<8x8xf32>, tensor<8x8xf32>)
                    outs(%C : tensor<8x8xf32>) -> tensor<8x8xf32>
                %idx0 = arith.constant 0 : index
                %val = tensor.extract %result[%idx0, %idx0] : tensor<8x8xf32>
                return %val : f32
            }
        }
    )";

    // Expected: 8 * 2.0 * 3.0 = 48.0
    auto hwInfo = HardwareInfo::detect();
    auto tiles = hwInfo.getOptimalTileSizes();
    hwInfo.print();
    llvm::outs() << "\n";

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(mlirSource), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) { llvm::errs() << "Parse failed\n"; return; }

    // Apply transform dialect strategy BEFORE bufferization
    mlir::PassManager pm(&context);

    // Use Transform Dialect for tiling + vectorization
    addTransformStrategyPass(pm, tiles);

    // Standard cleanup
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // Bufferization
    addTenzoBufferizationPasses(pm);

    // LLVM lowering (without affine tiling since transform already tiled)
    addTenzoToLLVMPasses(pm, false, tiles);

    if (mlir::failed(pm.run(*module))) {
        llvm::errs() << "Transform pipeline failed\n";
        return;
    }

    // JIT
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerLLVMDialectTranslation(context);
    mlir::registerBuiltinDialectTranslation(context);

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
    if (!maybeEngine) {
        llvm::errs() << "Engine creation failed\n";
        return;
    }
    auto engine = std::move(maybeEngine.get());

    // Benchmark
    float result = 0.0f;
    void* args[] = {&result};

    // Warmup
    for (int i = 0; i < 10; i++) {
        (void)engine->invokePacked("main", args);
    }

    const int ITERATIONS = 10000;
    llvm::outs() << "Running " << ITERATIONS << " iterations...\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        (void)engine->invokePacked("main", args);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double msPerIter = (double)totalMs / ITERATIONS;

    // Calculate GFLOPS: 2*M*N*K = 2*8*8*8 = 1024
    double flops = 2.0 * 8 * 8 * 8;
    double gflops = (flops / msPerIter) / 1e6;

    llvm::outs() << "\n📊 RESULTS (Transform Dialect, 8x8):\n";
    llvm::outs() << "Total: " << totalMs << " ms\n";
    llvm::outs() << "Per iteration: " << msPerIter << " ms\n";
    llvm::outs() << "Throughput: " << gflops << " GFLOPS\n";
    llvm::outs() << "Result: " << result << " (Expected 48.0)\n";

    if (std::abs(result - 48.0f) < 0.1f) {
        llvm::outs() << "✅ Transform Dialect test passed!\n";
    } else {
        llvm::outs() << "❌ Result incorrect!\n";
    }
}

//===----------------------------------------------------------------------===//
// Parallel (Multithreaded) Benchmark using OpenMP
// Compiles with parallel loops that execute on multiple cores
//===----------------------------------------------------------------------===//
void runParallelBenchmark(mlir::MLIRContext &context) {
    llvm::outs() << "\n=========================================\n";
    llvm::outs() << "🧵 PARALLEL BENCHMARK: 512x512 MatMul\n";
    llvm::outs() << "=========================================\n\n";

    auto hwInfo = HardwareInfo::detect();
    hwInfo.print();

    auto strategy = hwInfo.getMicroKernelStrategy();
    llvm::outs() << "\n🎯 Parallel Strategy:\n";
    llvm::outs() << "   Threads: " << strategy.numThreads << "\n";
    llvm::outs() << "   Tile: " << strategy.tileM << "x" << strategy.tileN << "x" << strategy.tileK << "\n\n";

    const int ITERATIONS = 100;

    // ========================================
    // 1. Compile SINGLE-THREADED version
    // ========================================
    llvm::outs() << "Compiling Single-threaded version (Affine Tiling)...\n";
    auto singleEngine = compileWithOpenMP(context, /*enableParallel=*/false);

    // ========================================
    // 2. Compile PARALLEL version (OpenMP)
    // ========================================
    llvm::outs() << "Compiling Parallel version (OpenMP)...\n";
    auto parallelEngine = compileWithOpenMP(context, /*enableParallel=*/true);

    if (!singleEngine) {
        llvm::errs() << "Single-threaded compilation failed\n";
        return;
    }

    float singleResult = 0.0f, parallelResult = 0.0f;
    void* singleArgs[] = {&singleResult};
    void* parallelArgs[] = {&parallelResult};

    // Warmup
    llvm::outs() << "Warming up...\n";
    for (int i = 0; i < 10; i++) {
        (void)singleEngine->invokePacked("main", singleArgs);
        if (parallelEngine) {
            (void)parallelEngine->invokePacked("main", parallelArgs);
        }
    }

    // ========================================
    // 3. Benchmark SINGLE-THREADED
    // ========================================
    llvm::outs() << "Running Single-threaded (" << ITERATIONS << " iterations)...\n";
    auto startSingle = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        (void)singleEngine->invokePacked("main", singleArgs);
    }
    auto endSingle = std::chrono::high_resolution_clock::now();

    // ========================================
    // 4. Benchmark PARALLEL (OpenMP)
    // ========================================
    long parallelMs = 0;
    if (parallelEngine) {
        llvm::outs() << "Running Parallel/OpenMP (" << ITERATIONS << " iterations)...\n";
        auto startParallel = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            (void)parallelEngine->invokePacked("main", parallelArgs);
        }
        auto endParallel = std::chrono::high_resolution_clock::now();
        parallelMs = std::chrono::duration_cast<std::chrono::milliseconds>(endParallel - startParallel).count();
    } else {
        llvm::outs() << "⚠️ OpenMP compilation failed, skipping parallel benchmark\n";
    }

    auto singleMs = std::chrono::duration_cast<std::chrono::milliseconds>(endSingle - startSingle).count();

    // Calculate GFLOPS: 2 * N^3 * iterations
    double totalOps = 2.0 * 512 * 512 * 512 * ITERATIONS;
    double singleGflops = (totalOps / (singleMs / 1000.0)) / 1e9;
    double parallelGflops = parallelMs > 0 ? (totalOps / (parallelMs / 1000.0)) / 1e9 : 0;

    // Peak efficiency calculation
    float peakGflops = hwInfo.getTheoreticalPeakGFLOPS();

    llvm::outs() << "\n" << std::string(50, '=') << "\n";
    llvm::outs() << "📊 RESULTS (512x512 MatMul, " << ITERATIONS << " runs)\n";
    llvm::outs() << std::string(50, '=') << "\n\n";

    llvm::outs() << "Single-threaded (Affine Tiling):\n";
    llvm::outs() << "   Time: " << singleMs << " ms\n";
    llvm::outs() << "   Throughput: " << singleGflops << " GFLOPS\n";
    llvm::outs() << "   Efficiency: " << (singleGflops / peakGflops * 100) << "% of peak\n\n";

    if (parallelMs > 0) {
        llvm::outs() << "Parallel (OpenMP, " << strategy.numThreads << " threads):\n";
        llvm::outs() << "   Time: " << parallelMs << " ms\n";
        llvm::outs() << "   Throughput: " << parallelGflops << " GFLOPS\n";
        llvm::outs() << "   Efficiency: " << (parallelGflops / peakGflops * 100) << "% of peak\n\n";

        if (parallelMs < singleMs) {
            double speedup = (double)singleMs / parallelMs;
            llvm::outs() << "⚡ Parallel Speedup: " << speedup << "x faster!\n";
        } else {
            llvm::outs() << "⚠️ No parallel speedup (check OpenMP runtime)\n";
        }
    }

    llvm::outs() << "\nTheoretical Peak: " << peakGflops << " GFLOPS\n";
    llvm::outs() << "Verification: Single=" << singleResult << ", Parallel=" << parallelResult
                 << " (Expected 3072.0)\n";
}

//===----------------------------------------------------------------------===//
// Large Matrix Benchmark (768x768) - Tests scaling behavior
//===----------------------------------------------------------------------===//
std::unique_ptr<mlir::ExecutionEngine> compileLargeMatrix(
    mlir::MLIRContext &context,
    bool enableVectorization) {

    // 768x768 MatMul - larger work than 512x512
    // Expected result: 768 * 2.0 * 3.0 = 4608.0
    const char* mlirSource = R"(
        module {
            func.func @main() -> f32 {
                %c2 = arith.constant 2.0 : f32
                %c3 = arith.constant 3.0 : f32
                %c0 = arith.constant 0.0 : f32
                %A_empty = tensor.empty() : tensor<768x768xf32>
                %B_empty = tensor.empty() : tensor<768x768xf32>
                %C_empty = tensor.empty() : tensor<768x768xf32>
                %A = linalg.fill ins(%c2 : f32) outs(%A_empty : tensor<768x768xf32>) -> tensor<768x768xf32>
                %B = linalg.fill ins(%c3 : f32) outs(%B_empty : tensor<768x768xf32>) -> tensor<768x768xf32>
                %C = linalg.fill ins(%c0 : f32) outs(%C_empty : tensor<768x768xf32>) -> tensor<768x768xf32>
                %result = linalg.matmul
                    ins(%A, %B : tensor<768x768xf32>, tensor<768x768xf32>)
                    outs(%C : tensor<768x768xf32>) -> tensor<768x768xf32>
                %idx0 = arith.constant 0 : index
                %val = tensor.extract %result[%idx0, %idx0] : tensor<768x768xf32>
                return %val : f32
            }
        }
    )";

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(mlirSource), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) return nullptr;

    mlir::PassManager pm(&context);

    // Standard pipeline
    {
        mlir::RewritePatternSet patterns(&context);
        populateFusionPatterns(patterns);
        (void)mlir::applyPatternsAndFoldGreedily(module.get(), std::move(patterns));
    }

    {
        mlir::RewritePatternSet patterns(&context);
        populateTenzoToLinalgConversionPatterns(patterns);
        mlir::ConversionTarget target(context);
        target.addLegalDialect<mlir::linalg::LinalgDialect, mlir::arith::ArithDialect,
                               mlir::tensor::TensorDialect, mlir::func::FuncDialect>();
        target.addIllegalDialect<tenzo::TenzoDialect>();
        (void)mlir::applyPartialConversion(*module, target, std::move(patterns));
    }

    if (enableVectorization) {
        addTenzoVectorizationPass(pm);
    }

    addTenzoBufferizationPasses(pm);

    auto hwInfo = HardwareInfo::detect();
    auto tiles = hwInfo.getOptimalTileSizes();
    addTenzoToLLVMPasses(pm, enableVectorization, tiles, false);

    if (mlir::failed(pm.run(*module))) {
        return nullptr;
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerLLVMDialectTranslation(context);
    mlir::registerBuiltinDialectTranslation(context);

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
    if (!maybeEngine) return nullptr;

    return std::move(maybeEngine.get());
}

void runLargeMatrixBenchmark(mlir::MLIRContext &context) {
    llvm::outs() << "\n" << std::string(50, '=') << "\n";
    llvm::outs() << "🔥 LARGE MATRIX BENCHMARK: 768x768 MatMul\n";
    llvm::outs() << std::string(50, '=') << "\n\n";

    auto hwInfo = HardwareInfo::detect();
    hwInfo.print();

    const int ITERATIONS = 30;  // Fewer iterations for larger matrix

    llvm::outs() << "\nCompiling Scalar version (768x768)...\n";
    auto scalarEngine = compileLargeMatrix(context, false);

    llvm::outs() << "Compiling Vector version (768x768)...\n";
    auto vectorEngine = compileLargeMatrix(context, true);

    if (!scalarEngine || !vectorEngine) {
        llvm::errs() << "Compilation failed\n";
        return;
    }

    float scalarRes = 0.0f, vectorRes = 0.0f;
    void* scalarArgs[] = {&scalarRes};
    void* vectorArgs[] = {&vectorRes};

    // Warmup
    llvm::outs() << "Warming up...\n";
    for (int i = 0; i < 3; i++) {
        (void)scalarEngine->invokePacked("main", scalarArgs);
        (void)vectorEngine->invokePacked("main", vectorArgs);
    }

    // Benchmark
    llvm::outs() << "Running Scalar (" << ITERATIONS << " iterations)...\n";
    auto startScalar = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        (void)scalarEngine->invokePacked("main", scalarArgs);
    }
    auto endScalar = std::chrono::high_resolution_clock::now();

    llvm::outs() << "Running Vector (" << ITERATIONS << " iterations)...\n";
    auto startVector = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        (void)vectorEngine->invokePacked("main", vectorArgs);
    }
    auto endVector = std::chrono::high_resolution_clock::now();

    auto scalarMs = std::chrono::duration_cast<std::chrono::milliseconds>(endScalar - startScalar).count();
    auto vectorMs = std::chrono::duration_cast<std::chrono::milliseconds>(endVector - startVector).count();

    // GFLOPS: 2 * N^3 * iterations
    double totalOps = 2.0 * 768 * 768 * 768 * ITERATIONS;
    double scalarGflops = (totalOps / (scalarMs / 1000.0)) / 1e9;
    double vectorGflops = vectorMs > 0 ? (totalOps / (vectorMs / 1000.0)) / 1e9 : 0;

    llvm::outs() << "\n" << std::string(50, '=') << "\n";
    llvm::outs() << "📊 RESULTS (768x768, " << ITERATIONS << " runs)\n";
    llvm::outs() << std::string(50, '=') << "\n\n";

    llvm::outs() << "Scalar (LLVM -O3):\n";
    llvm::outs() << "   Time: " << scalarMs << " ms\n";
    llvm::outs() << "   Throughput: " << scalarGflops << " GFLOPS\n\n";

    llvm::outs() << "Vector (Tenzo Optimized):\n";
    llvm::outs() << "   Time: " << vectorMs << " ms\n";
    llvm::outs() << "   Throughput: " << vectorGflops << " GFLOPS\n\n";

    if (vectorMs > 0 && vectorMs < scalarMs) {
        double speedup = (double)scalarMs / vectorMs;
        llvm::outs() << "⚡ Speedup: " << speedup << "x faster!\n";
    }

    // Single-core efficiency
    double singleCorePeak = 4.4 * 16;  // P-core turbo * AVX2 FMA ops
    double efficiency = (vectorGflops / singleCorePeak) * 100;
    llvm::outs() << "\n📈 Single-core Efficiency: " << efficiency << "% of " << singleCorePeak << " GFLOPS peak\n";

    llvm::outs() << "\nVerification: Scalar=" << scalarRes << ", Vector=" << vectorRes
                 << " (Expected 4608.0)\n";
}

// NEW: Explicit Micro-Kernel Benchmark
void runExplicitKernelBenchmark(mlir::MLIRContext &context) {
    llvm::outs() << "\n=========================================\n";
    llvm::outs() << "🔥 EXPLICIT MICRO-KERNEL: 512x512 MatMul\n";
    llvm::outs() << "=========================================\n\n";

    auto hwInfo = HardwareInfo::detect();
    hwInfo.print();
    llvm::outs() << "\n";

    const char* mlirSource = R"(
        module {
            func.func @main() -> f32 {
                %c2 = arith.constant 2.0 : f32
                %c3 = arith.constant 3.0 : f32
                %c0 = arith.constant 0.0 : f32
                %A_empty = tensor.empty() : tensor<512x512xf32>
                %B_empty = tensor.empty() : tensor<512x512xf32>
                %C_empty = tensor.empty() : tensor<512x512xf32>
                %A = linalg.fill ins(%c2 : f32) outs(%A_empty : tensor<512x512xf32>) -> tensor<512x512xf32>
                %B = linalg.fill ins(%c3 : f32) outs(%B_empty : tensor<512x512xf32>) -> tensor<512x512xf32>
                %C = linalg.fill ins(%c0 : f32) outs(%C_empty : tensor<512x512xf32>) -> tensor<512x512xf32>
                %result = linalg.matmul
                    ins(%A, %B : tensor<512x512xf32>, tensor<512x512xf32>)
                    outs(%C : tensor<512x512xf32>) -> tensor<512x512xf32>
                %idx0 = arith.constant 0 : index
                %val = tensor.extract %result[%idx0, %idx0] : tensor<512x512xf32>
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

    // Simple strategy: Use standard pipeline but with 6x16 micro-kernel tiles
    llvm::outs() << "Strategy: Standard Affine optimization with 6x16 micro-kernels\n";

    mlir::PassManager pm(&context);

    // 1. Fusion
    {
        mlir::RewritePatternSet patterns(&context);
        populateFusionPatterns(patterns);
        (void)mlir::applyPatternsAndFoldGreedily(module.get(), std::move(patterns));
    }

    // 2. Linalg Lowering (Tenzo -> Linalg)
    {
        mlir::RewritePatternSet patterns(&context);
        populateTenzoToLinalgConversionPatterns(patterns);
        mlir::ConversionTarget target(context);
        target.addLegalDialect<mlir::linalg::LinalgDialect, mlir::arith::ArithDialect,
                               mlir::tensor::TensorDialect, mlir::func::FuncDialect>();
        target.addIllegalDialect<tenzo::TenzoDialect>();
        (void)mlir::applyPartialConversion(*module, target, std::move(patterns));
    }

    // 3. Bufferization
    llvm::outs() << "Bufferizing...\n";
    addTenzoBufferizationPasses(pm);

    // Use LLVM lowering with CUSTOM 24x32 tile sizes (4x2 blocks of 6x16)
    HardwareInfo::TileSizes customTiles;
    customTiles.M = 24;  // 4 * 6 = 24 rows (4 micro-kernels)
    customTiles.N = 32;  // 2 * 16 = 32 cols (2 micro-kernels)
    customTiles.K = 32;  // K dimension

    llvm::outs() << "Using 24x32 tiles (4x2 blocks of 6x16 micro-kernels)\n";
    addTenzoToLLVMPasses(pm, /*enableVectorization=*/true, customTiles,
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
    for (int i = 0; i < 10; i++) {
        (void)engine->invokePacked("main", args);
    }

    // Benchmark
    const int ITERATIONS = 100;
    llvm::outs() << "Running benchmark (" << ITERATIONS << " iterations)...\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        (void)engine->invokePacked("main", args);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double totalOps = 2.0 * 512 * 512 * 512 * ITERATIONS;
    double gflops = (totalOps / (totalMs / 1000.0)) / 1e9;

    llvm::outs() << "\n🔥 EXPLICIT MICRO-KERNEL RESULTS:\n";
    llvm::outs() << "Total time: " << totalMs << " ms\n";
    llvm::outs() << "Throughput: " << gflops << " GFLOPS\n";
    llvm::outs() << "Result: " << result << " (Expected 3072.0)\n";

    if (std::abs(result - 3072.0f) < 0.001f) {
        llvm::outs() << "✅ Correctness verified!\n";
    } else {
        llvm::outs() << "❌ Result mismatch!\n";
    }
}

void runFullPipelineTest(mlir::MLIRContext &context) {
    runBenchmark(context);
}
}
