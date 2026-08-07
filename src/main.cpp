#include "context/TenzoContext.h"
#include "tests/PipelineTests.h"
#include "tests/GPUTests.h"
#include "tests/DynamicInferenceTest.h"
#include "tests/ZeroCopyBridgeTest.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>

void printUsage() {
    llvm::outs() << "Usage: tenzo-cli [command]\n";
    llvm::outs() << "Commands:\n";
    llvm::outs() << "  cpu       Run CPU MatMul benchmark (512x512, default)\n";
    llvm::outs() << "  explicit  Run Explicit Micro-Kernel benchmark (GotoBLAS-style)\n";
    llvm::outs() << "  large     Run large MatMul benchmark (1024x1024)\n";
    llvm::outs() << "  parallel  Run parallel (multithreaded) benchmark\n";
    llvm::outs() << "  small     Run small MatMul test (64x64, explicit vectorization)\n";
    llvm::outs() << "  microkernel Run micro-kernel benchmark (6x16, pure performance)\n";
    llvm::outs() << "  blis      Run BLIS-style benchmark (packing + explicit kernel, 40+ GFLOPS target)\n";
    llvm::outs() << "  generate-microkernel Generate MLIR micro-kernel source code\n";
    llvm::outs() << "  packing   Run packing kernels bandwidth benchmark (Etap 3)\n";
    llvm::outs() << "  gemm      Run full GEMM with 5-loop macro-kernel (Etap 3 Day 2)\n";
    llvm::outs() << "  gemm-e2e  Run native AVX2 GEMM end-to-end benchmark (Etap 3 - Packing)\n";
    llvm::outs() << "  linear    Run Linear Layer inference benchmark (Etap 3 Day 3 - Real NN)\n";
    llvm::outs() << "  transform Run Transform Dialect test (64x64)\n";
    llvm::outs() << "  conv2d    Run Conv2D benchmark\n";
    llvm::outs() << "  dynamic   Run Dynamic Shape Inference test\n";
    llvm::outs() << "  gpu       Run GPU pipeline test\n";
    llvm::outs() << "  gpu-bench Run GPU vs CPU benchmark\n";
    llvm::outs() << "  hybrid-bench Run A/B benchmark for Hybrid Affinity and Heterogeneous Split\n";
    llvm::outs() << "  test      Run quick validation tests\n";
    llvm::outs() << "  all       Run all tests\n";
    llvm::outs() << "  version   Show version info\n";
}

void printVersion() {
    llvm::outs() << "╔════════════════════════════════════════╗\n";
    llvm::outs() << "║  Tenzo Compiler v0.3.0                 ║\n";
    llvm::outs() << "║  Heterogeneous AI Compiler             ║\n";
    llvm::outs() << "╠════════════════════════════════════════╣\n";
    llvm::outs() << "║  Targets: CPU (AVX2), GPU (Vulkan)     ║\n";
    llvm::outs() << "║  Backend: MLIR/LLVM 21                 ║\n";
    llvm::outs() << "╚════════════════════════════════════════╝\n";
}

int main(int argc, char* argv[]) {
    mlir::MLIRContext context;
    tenzo::registerAllDialects(context);

    const char* mode = "cpu";
    if (argc > 1) {
        mode = argv[1];
    }

    if (strcmp(mode, "version") == 0 || strcmp(mode, "-v") == 0 || strcmp(mode, "--version") == 0) {
        printVersion();
        return 0;
    }

    if (strcmp(mode, "cpu") == 0) {
        llvm::outs() << "--- Running CPU Pipeline Test ---\n";
        tenzo::runFullPipelineTest(context);
    } else if (strcmp(mode, "explicit") == 0) {
        llvm::outs() << "--- Running Explicit Micro-Kernel Benchmark ---\n";
        tenzo::runExplicitKernelBenchmark(context);
    } else if (strcmp(mode, "large") == 0) {
        llvm::outs() << "--- Running Large Matrix Benchmark (1024x1024) ---\n";
        tenzo::runLargeMatrixBenchmark(context);
    } else if (strcmp(mode, "parallel") == 0) {
        llvm::outs() << "--- Running Parallel (Multithreaded) Benchmark ---\n";
        tenzo::runParallelBenchmark(context);
    } else if (strcmp(mode, "small") == 0) {
        tenzo::runSmallMatMulTest(context);
    } else if (strcmp(mode, "microkernel") == 0) {
        tenzo::runMicroKernelBenchmark(context);
    } else if (strcmp(mode, "blis") == 0) {
        tenzo::runBLISBenchmark(context);
    } else if (strcmp(mode, "generate-microkernel") == 0) {
        tenzo::generateMicroKernelMLIR(context);
    } else if (strcmp(mode, "packing") == 0) {
        llvm::outs() << "--- Running Packing Kernels Benchmark ---\n";
        tenzo::runPackingBenchmark(context);
    } else if (strcmp(mode, "gemm") == 0) {
        llvm::outs() << "--- Running Full GEMM Benchmark ---\n";
        tenzo::runFullGEMMBenchmark(context);
    } else if (strcmp(mode, "gemm-e2e") == 0) {
        llvm::outs() << "--- Running GEMM End-to-End Benchmark (Native AVX2) ---\n";
        tenzo::runGEMMEndToEndBenchmark(context);
    } else if (strcmp(mode, "linear") == 0) {
        llvm::outs() << "--- Running Linear Layer Inference Benchmark ---\n";
        tenzo::runLinearLayerBenchmark();
    } else if (strcmp(mode, "transform") == 0) {
        tenzo::runTransformDialectTest(context);
    } else if (strcmp(mode, "conv2d") == 0) {
        llvm::outs() << "--- Running Conv2D Benchmark ---\n";
        tenzo::runConv2DTest(context);
    } else if (strcmp(mode, "gpu") == 0) {
        tenzo::gpu::runGPUPipelineTest(context);
    } else if (strcmp(mode, "gpu-bench") == 0) {
        tenzo::gpu::runGPUBenchmark(context);
    } else if (strcmp(mode, "hybrid-bench") == 0) {
        llvm::outs() << "--- Running Hybrid Affinity A/B Benchmark ---\n";
        tenzo::runHybridAffinityBenchmark(context);
    } else if (strcmp(mode, "fusion-bench") == 0) {
        llvm::outs() << "--- Running Final Operator Fusion Benchmark ---\n";
        tenzo::runFusionBenchmark(context);
    } else if (strcmp(mode, "dynamic") == 0) {
        tenzo::runDynamicInferenceTest(context);
    } else if (strcmp(mode, "bridge") == 0) {
        tenzo::runZeroCopyBridgeTest(context);
    } else if (strcmp(mode, "test") == 0) {
        // Quick validation tests
        llvm::outs() << "--- Running Quick Validation Tests ---\n\n";
        bool allPassed = true;

        // Test 1: CPU pipeline
        llvm::outs() << "[1/3] CPU MatMul... ";
        tenzo::runFullPipelineTest(context);

        // Test 2: Conv2D
        llvm::outs() << "\n[2/3] Conv2D... ";
        tenzo::runConv2DTest(context);

        // Test 3: GPU pipeline
        llvm::outs() << "\n[3/3] GPU Pipeline... ";
        tenzo::gpu::runGPUPipelineTest(context);

        llvm::outs() << "\n--- All Tests Complete ---\n";
        return allPassed ? 0 : 1;
    } else if (strcmp(mode, "all") == 0) {
        llvm::outs() << "--- Running All Tests ---\n\n";
        llvm::outs() << "=== CPU Tests ===\n";
        tenzo::runFullPipelineTest(context);
        llvm::outs() << "\n=== Conv2D Tests ===\n";
        tenzo::runConv2DTest(context);
        llvm::outs() << "\n=== GPU Tests ===\n";
        tenzo::gpu::runGPUPipelineTest(context);
    } else if (strcmp(mode, "help") == 0 || strcmp(mode, "-h") == 0 || strcmp(mode, "--help") == 0) {
        printUsage();
        return 0;
    } else {
        llvm::errs() << "Unknown command: " << mode << "\n\n";
        printUsage();
        return 1;
    }
    return 0;
}
