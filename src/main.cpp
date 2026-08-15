#include "context/TenzoContext.h"
#include "tests/PipelineTests.h"
#include "tests/GPUTests.h"
#include "tests/DynamicInferenceTest.h"
#include "tests/ZeroCopyBridgeTest.h"
#include "tests/EndToEndMathTest.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <cstdlib>

void printUsage() {
    llvm::outs() << "Usage: tenzo-cli [command] [options]\n\n";
    llvm::outs() << "Commands:\n";
    llvm::outs() << "  generate  Run autoregressive text generation loop\n";
    llvm::outs() << "            Options:\n";
    llvm::outs() << "              -p, --prompt <str>       Prompt string (default: \"Tenzo Edge AI\")\n";
    llvm::outs() << "              -n, --max-tokens <int>   Max generated tokens (default: 20)\n";
    llvm::outs() << "              -t, --temp <float>       Temperature (default: 0.7, 0.0 = Greedy/ArgMax)\n";
    llvm::outs() << "                  --top-p <float>      Nucleus sampling top-p (default: 0.9)\n";
    llvm::outs() << "              -m, --model-dir <path>   Directory with model.mlir, weights.bin, etc.\n";
    llvm::outs() << "  cpu       Run CPU MatMul benchmark (512x512, default)\n";
    llvm::outs() << "  explicit  Run Explicit Micro-Kernel benchmark (GotoBLAS-style)\n";
    llvm::outs() << "  large     Run large MatMul benchmark (1024x1024)\n";
    llvm::outs() << "  parallel  Run parallel (multithreaded) benchmark\n";
    llvm::outs() << "  microkernel Run micro-kernel benchmark (6x16, pure performance)\n";
    llvm::outs() << "  blis      Run BLIS-style benchmark (packing + explicit kernel, 40+ GFLOPS target)\n";
    llvm::outs() << "  validate  Run single-pass E2E math validation\n";
    llvm::outs() << "  gpu       Run GPU pipeline test\n";
    llvm::outs() << "  gpu-bench Run GPU vs CPU benchmark\n";
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
    } else if (strcmp(mode, "tl1") == 0) { tenzo::runTestTL1(context); } else if (strcmp(mode, "validate") == 0) {
        tenzo::runEndToEndMathTest(context);
    } else if (strcmp(mode, "generate") == 0) {
        tenzo::GenerationConfig config;
        
        int i = 2;
        while (i < argc) {
            std::string arg = argv[i];
            if (arg == "-p" || arg == "--prompt") {
                if (i + 1 < argc) { config.prompt = argv[++i]; }
            } else if (arg == "-n" || arg == "--max-tokens") {
                if (i + 1 < argc) { config.max_tokens = std::atoi(argv[++i]); }
            } else if (arg == "-t" || arg == "--temp" || arg == "--temperature") {
                if (i + 1 < argc) { config.temperature = std::atof(argv[++i]); }
            } else if (arg == "--top-p") {
                if (i + 1 < argc) { config.top_p = std::atof(argv[++i]); }
            } else if (arg == "-m" || arg == "--model-dir") {
                if (i + 1 < argc) { config.model_dir = argv[++i]; }
            } else if (arg == "-h" || arg == "--help") {
                printUsage();
                return 0;
            } else if (arg[0] != '-') {
                // Positional arguments fallback: 1st is prompt, 2nd is max_tokens
                if (i == 2) config.prompt = arg;
                else if (i == 3) config.max_tokens = std::atoi(arg.c_str());
            }
            i++;
        }

        tenzo::runGenerationTest(context, config);

    } else if (strcmp(mode, "test") == 0) {
        llvm::outs() << "--- Running Quick Validation Tests ---\n\n";
        bool allPassed = true;

        llvm::outs() << "[1/3] CPU MatMul... ";
        tenzo::runFullPipelineTest(context);

        llvm::outs() << "\n[2/3] Conv2D... ";
        tenzo::runConv2DTest(context);

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
