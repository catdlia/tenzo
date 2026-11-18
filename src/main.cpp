#include "context/TenzoContext.h"
#include "tests/PipelineTests.h"
#include "tests/GPUTests.h"
#include "tests/DynamicInferenceTest.h"
#include "tests/ZeroCopyBridgeTest.h"
#include "tests/EndToEndMathTest.h"
#include "tests/HeteroTests.h"
#include "tests/TernaryPackTest.h"
#include "tests/BitwiseAttentionTest.h"
#include "tests/AuditJITTest.h"
#include "runtime/TenzoEngine.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <cstdlib>
#include <thread>
#include <chrono>

namespace tenzo {
bool runTestTL1(mlir::MLIRContext& context, const std::string& custom_weights_path = "");
bool runHFSemanticsTests(mlir::MLIRContext& context);
}

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
    llvm::outs() << "  hetero    Run heterogeneous pipeline & distributed network tests\n";
    llvm::outs() << "  hetero-bench Run heterogeneous pipeline partition benchmark\n";
    llvm::outs() << "  worker    Run distributed cluster worker node (e.g. worker [port])\n";
    llvm::outs() << "  cpu       Run CPU MatMul benchmark (512x512, default)\n";
    llvm::outs() << "  explicit  Run Explicit Micro-Kernel benchmark (GotoBLAS-style)\n";
    llvm::outs() << "  large     Run large MatMul benchmark (1024x1024)\n";
    llvm::outs() << "  parallel  Run parallel (multithreaded) benchmark\n";
    llvm::outs() << "  microkernel Run micro-kernel benchmark (6x16, pure performance)\n";
    llvm::outs() << "  blis      Run BLIS-style benchmark (packing + explicit kernel, 40+ GFLOPS target)\n";
    llvm::outs() << "  validate  Run single-pass E2E math validation\n";
    llvm::outs() << "  gpu       Run GPU pipeline test\n";
    llvm::outs() << "  gpu-bench Run GPU vs CPU benchmark\n";
    llvm::outs() << "  gpu-nan-test Run GPU NaN/Inf detection regression test\n";
    llvm::outs() << "  tl1 [path] Run TL1 fixture test (optional custom weights fixture path)\n";
    llvm::outs() << "  hf-semantics Regression: rope_theta and 8-bit ActQuant vs HF (compiler + runtime)\n";
    llvm::outs() << "  packing   Run packing kernels bandwidth benchmark\n";
    llvm::outs() << "  gemm      Run GotoBLAS 5-loop GEMM benchmark\n";
    llvm::outs() << "  gemm-e2e  Run GEMM end-to-end native AVX2 benchmark\n";
    llvm::outs() << "  conv2d    Run Conv2D benchmark\n";
    llvm::outs() << "  dynamic   Run dynamic inference test\n";
    llvm::outs() << "  ternary   Run ternary pack (1.58-bit) test\n";
    llvm::outs() << "  generate-microkernel Generate MLIR micro-kernel\n";
    llvm::outs() << "  test      Run quick validation tests\n";
    llvm::outs() << "  attention Run bitwise attention (tenzo.packed_attention) tests\n";
    llvm::outs() << "  audit     Run Audit JIT regression tests against independent references\n";
    llvm::outs() << "            Subcommands: [all|pack|fused|consumer|explicit|explicit-integer|explicit-gqa|linalg|linalg-gqa]\n";
    llvm::outs() << "  all       Run all tests\n";
    llvm::outs() << "  version   Show version info\n";
}

void printVersion() {
    llvm::outs() << "╔════════════════════════════════════════╗\n";
    llvm::outs() << "║  Tenzo Compiler v1.1.0                 ║\n";
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
        if (!tenzo::runFullPipelineTest(context)) return 1;
    } else if (strcmp(mode, "explicit") == 0) {
        llvm::outs() << "--- Running Explicit Micro-Kernel Benchmark ---\n";
        if (!tenzo::runExplicitKernelBenchmark(context)) return 1;
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
        if (!tenzo::runConv2DTest(context)) return 1;
    } else if (strcmp(mode, "gpu") == 0) {
        if (!tenzo::gpu::runGPUPipelineTest(context)) return 1;
    } else if (strcmp(mode, "gpu-bench") == 0) {
        tenzo::gpu::runGPUBenchmark(context);
    } else if (strcmp(mode, "gpu-nan-test") == 0) {
        return tenzo::gpu::runGPUNaNRegressionTest() ? 0 : 1;
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
    } else if (strcmp(mode, "hf-semantics") == 0) {
        if (!tenzo::runHFSemanticsTests(context)) return 1;
    } else if (strcmp(mode, "tl1") == 0) {
        std::string custom_w = (argc > 2) ? argv[2] : "";
        if (!tenzo::runTestTL1(context, custom_w)) return 1;
    } else if (strcmp(mode, "hetero") == 0 || strcmp(mode, "hetero-test") == 0) {
        tenzo::runHeteroPipelineTests();
        tenzo::runDistributedNetworkTests();
    } else if (strcmp(mode, "hetero-bench") == 0) {
        tenzo::runHeteroBenchmark();
    } else if (strcmp(mode, "worker") == 0) {
        int port = 9000;
        if (argc > 2) port = std::atoi(argv[2]);
        const char* weights = (argc > 3) ? argv[3] : nullptr;
        const char* mlir = (argc > 4) ? argv[4] : nullptr;
        llvm::outs() << "=== Starting Tenzo Distributed Cluster Worker on port " << port << " ===\n";
        tenzo_start_network_worker(port, weights, mlir);
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    } else if (strcmp(mode, "ternary") == 0 || strcmp(mode, "ternary-pack") == 0) {
        if (!tenzo::runTernaryPackTest(context)) return 1;
    } else if (strcmp(mode, "attention") == 0 || strcmp(mode, "bitwise-attention") == 0) {
        if (!tenzo::runBitwiseAttentionTest(context)) return 1;
    } else if (strcmp(mode, "audit") == 0) {
        llvm::outs() << "--- Running Audit JIT Regression Tests ---\n\n";
        std::string submode = (argc > 2) ? argv[2] : "all";
        bool ok = false;
        if (submode == "all") {
            ok = tenzo::runAuditJITTests(context, tenzo::AuditTestMode::All);
        } else if (submode == "pack") {
            ok = tenzo::auditTernaryPack(context, /*fused=*/false);
        } else if (submode == "fused") {
            ok = tenzo::auditTernaryPack(context, /*fused=*/true);
        } else if (submode == "consumer" || submode == "fusion-consumer") {
            ok = tenzo::auditFusionMultiConsumer(context);
        } else if (submode == "explicit") {
            ok = tenzo::auditExplicitAttention(context, /*B=*/1, /*Hq=*/1, /*Hkv=*/1, /*T=*/8, /*D=*/32, /*fractional=*/true);
        } else if (submode == "explicit-integer") {
            ok = tenzo::auditExplicitAttention(context, /*B=*/1, /*Hq=*/1, /*Hkv=*/1, /*T=*/8, /*D=*/32, /*fractional=*/false);
        } else if (submode == "explicit-gqa") {
            ok = tenzo::auditExplicitAttention(context, /*B=*/2, /*Hq=*/20, /*Hkv=*/5, /*T=*/8, /*D=*/32, /*fractional=*/true);
        } else if (submode == "linalg") {
            ok = tenzo::auditLinalgAttention(context, /*B=*/1, /*Hq=*/1, /*Hkv=*/1, /*T=*/8, /*D=*/32, /*fractional=*/true);
        } else if (submode == "linalg-gqa") {
            ok = tenzo::auditLinalgAttention(context, /*B=*/2, /*Hq=*/20, /*Hkv=*/5, /*T=*/8, /*D=*/32, /*fractional=*/true);
        } else {
            llvm::errs() << "Unknown audit submode: " << submode << "\n";
            return 1;
        }
        return ok ? 0 : 1;
    } else if (strcmp(mode, "validate") == 0) {
        tenzo::runEndToEndMathTest(context);
        return 0;
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
        return 0;

    } else if (strcmp(mode, "test") == 0) {
        llvm::outs() << "--- Running Quick Validation Tests ---\n\n";
        bool allPassed = true;

        llvm::outs() << "[1/7] CPU MatMul... ";
        if (!tenzo::runFullPipelineTest(context)) allPassed = false;

        llvm::outs() << "\n[2/7] Conv2D... ";
        if (!tenzo::runConv2DTest(context)) allPassed = false;

        llvm::outs() << "\n[3/7] GPU Pipeline... ";
        if (!tenzo::gpu::runGPUPipelineTest(context)) allPassed = false;

        llvm::outs() << "\n[4/7] Ternary Pack (1.58-bit AVX2 Micro-Kernel)... ";
        if (!tenzo::runTernaryPackTest(context)) allPassed = false;

        llvm::outs() << "\n[5/7] Bitwise Attention (SIMD Packed Attention Micro-Kernel)... ";
        if (!tenzo::runBitwiseAttentionTest(context)) allPassed = false;

        llvm::outs() << "\n[6/7] Audit JIT (Independent Reference Regressions)... \n";
        if (!tenzo::runAuditJITTests(context)) allPassed = false;

        llvm::outs() << "\n[7/7] HF semantics (rope_theta, 8-bit ActQuant)... \n";
        if (!tenzo::runHFSemanticsTests(context)) allPassed = false;

        llvm::outs() << "\n--- All Tests Complete ---\n";
        return allPassed ? 0 : 1;
    } else if (strcmp(mode, "all") == 0) {
        llvm::outs() << "--- Running All Tests ---\n\n";
        bool allPassed = true;
        llvm::outs() << "=== CPU Tests ===\n";
        if (!tenzo::runFullPipelineTest(context)) allPassed = false;
        llvm::outs() << "\n=== Conv2D Tests ===\n";
        if (!tenzo::runConv2DTest(context)) allPassed = false;
        llvm::outs() << "\n=== GPU Tests ===\n";
        if (!tenzo::gpu::runGPUPipelineTest(context)) allPassed = false;
        llvm::outs() << "\n=== Ternary Pack Tests ===\n";
        if (!tenzo::runTernaryPackTest(context)) allPassed = false;
        llvm::outs() << "\n=== Bitwise Attention Tests ===\n";
        if (!tenzo::runBitwiseAttentionTest(context)) allPassed = false;
        llvm::outs() << "\n=== Audit JIT Tests ===\n";
        if (!tenzo::runAuditJITTests(context)) allPassed = false;
        llvm::outs() << "\n=== HF Semantics Tests ===\n";
        if (!tenzo::runHFSemanticsTests(context)) allPassed = false;
        return allPassed ? 0 : 1;
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
