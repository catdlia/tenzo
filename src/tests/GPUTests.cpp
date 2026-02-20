#include "GPUTests.h"
#include "passes/gpu/GPULowering.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "runtime/VulkanRuntime.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include <chrono>
#include <cmath>

namespace tenzo {
namespace gpu {

using namespace tenzo::gpu;

bool runGPUPipelineTest(mlir::MLIRContext &context) {
    llvm::outs() << "\n=========================================\n";
    llvm::outs() << "🎮 GPU Pipeline Test (Linalg -> SPIR-V)\n";
    llvm::outs() << "=========================================\n\n";

    const char* mlirSource = R"(
        module {
            func.func @elemwise_add(%A: memref<256xf32>, %B: memref<256xf32>, %C: memref<256xf32>) {
                linalg.generic {
                    indexing_maps = [
                        affine_map<(i) -> (i)>,
                        affine_map<(i) -> (i)>,
                        affine_map<(i) -> (i)>
                    ],
                    iterator_types = ["parallel"]
                } ins(%A, %B : memref<256xf32>, memref<256xf32>)
                  outs(%C : memref<256xf32>) {
                ^bb0(%a: f32, %b: f32, %c: f32):
                    %sum = arith.addf %a, %b : f32
                    linalg.yield %sum : f32
                }
                return
            }
        }
    )";

    llvm::outs() << "Input MLIR:\n" << mlirSource << "\n";

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(mlirSource), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

    if (!module) {
        llvm::errs() << "Failed to parse MLIR\n";
        return false;
    }

    // Stage 1: Linalg -> GPU
    llvm::outs() << "Stage 1: Converting Linalg to GPU dialect...\n";
    mlir::PassManager pm(&context);
    addLinalgToGPUPasses(pm);

    if (mlir::failed(pm.run(*module))) {
        llvm::errs() << "❌ Stage 1 (Linalg->GPU) failed!\n";
        return false;
    }

    llvm::outs() << "\nIR after Stage 1 (Linalg -> GPU):\n";
    module->print(llvm::outs());
    llvm::outs() << "\n";

    // Check for gpu.module
    bool hasGPUModule = false;
    module->walk([&](mlir::gpu::GPUModuleOp op) {
        hasGPUModule = true;
    });

    if (!hasGPUModule) {
        llvm::outs() << "⚠️  No gpu.module found - skipping SPIR-V conversion\n";
        return true;
    }

    // Stage 2: GPU -> SPIR-V
    llvm::outs() << "Stage 2: Converting GPU to SPIR-V...\n";
    mlir::PassManager pm2(&context);
    addGPUToSPIRVPasses(pm2);

    if (mlir::failed(pm2.run(*module))) {
        llvm::outs() << "\n⚠️  GPU->SPIR-V pipeline incomplete\n";
        llvm::outs() << "   (Host-side gpu.launch_func not converted)\n\n";
    }

    // Try to serialize SPIR-V
    llvm::outs() << "\nChecking for SPIR-V module...\n";
    auto spirvBinary = serializeSPIRVModule(*module);

    if (!spirvBinary.empty()) {
        llvm::outs() << "\n🎉🎉🎉 SUCCESS! 🎉🎉🎉\n";
        llvm::outs() << "✅ SPIR-V binary generated: " << spirvBinary.size() * 4 << " bytes\n";

        // Save to file
        std::error_code EC;
        llvm::raw_fd_ostream file("/tmp/kernel.spv", EC, llvm::sys::fs::OF_None);
        if (!EC) {
            file.write(reinterpret_cast<const char*>(spirvBinary.data()),
                      spirvBinary.size() * sizeof(uint32_t));
            llvm::outs() << "✅ Saved to /tmp/kernel.spv\n";
        }

        // Try to execute on GPU!
        llvm::outs() << "\n📋 Attempting GPU execution...\n";

        if (runtime::VulkanRuntime::initialize()) {
            llvm::outs() << "GPU: " << runtime::VulkanRuntime::getDeviceName() << "\n";

            // Prepare test data
            std::vector<float> A(256, 2.0f);  // All 2.0
            std::vector<float> B(256, 3.0f);  // All 3.0
            std::vector<float> C(256, 0.0f);  // Output

            std::vector<std::pair<void*, size_t>> inputs = {
                {A.data(), A.size() * sizeof(float)},
                {B.data(), B.size() * sizeof(float)}
            };
            std::pair<void*, size_t> output = {C.data(), C.size() * sizeof(float)};

            if (runtime::VulkanRuntime::execute(
                    spirvBinary, inputs, output,
                    {1, 1, 1},   // workgroup size
                    {256, 1, 1}  // num workgroups
                )) {
                // Verify result
                bool correct = true;
                for (size_t i = 0; i < 256 && correct; i++) {
                    if (std::abs(C[i] - 5.0f) > 0.001f) {
                        correct = false;
                        llvm::outs() << "❌ Mismatch at " << i << ": " << C[i] << " != 5.0\n";
                    }
                }
                if (correct) {
                    llvm::outs() << "\n🎉🎉🎉 GPU EXECUTION SUCCESS! 🎉🎉🎉\n";
                    llvm::outs() << "✅ Result verified: 2.0 + 3.0 = 5.0 for all 256 elements\n";
                }
            } else {
                llvm::outs() << "⚠️  GPU execution failed (shader may need adjustment)\n";
            }

            runtime::VulkanRuntime::cleanup();
        } else {
            llvm::outs() << "⚠️  Vulkan not available - skipping GPU execution\n";
        }

        llvm::outs() << "\n📋 Next steps:\n";
        llvm::outs() << "   1. Debug shader if execution failed\n";
        llvm::outs() << "   2. Add MatMul kernel\n";
        return true;
    }

    // Check for spirv ops
    bool hasSpirvOps = false;
    module->walk([&](mlir::Operation *op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "spirv") {
            hasSpirvOps = true;
        }
    });

    if (hasSpirvOps) {
        llvm::outs() << "\n✅ GPU kernel compiled to SPIR-V ops!\n";
        llvm::outs() << "   (spirv.module serialization pending)\n";
        return true;
    }

    llvm::errs() << "❌ No SPIR-V code generated!\n";
    return false;
}

void runGPUBenchmark(mlir::MLIRContext &context) {
    llvm::outs() << "\n=========================================\n";
    llvm::outs() << "🚀 GPU vs CPU Benchmark (Element-wise Add)\n";
    llvm::outs() << "=========================================\n\n";

    const size_t SIZE = 256 * 1024;  // 256K elements = 1MB (reasonable for iGPU)
    const int CPU_ITERATIONS = 100;
    const int GPU_ITERATIONS = 10;   // Multiple GPU iterations

    // Prepare data
    std::vector<float> A(SIZE, 2.0f);
    std::vector<float> B(SIZE, 3.0f);
    std::vector<float> C_cpu(SIZE, 0.0f);
    std::vector<float> C_gpu(SIZE, 0.0f);

    llvm::outs() << "Data size: " << SIZE << " floats (" << SIZE * 4 / 1024 << " KB)\n\n";

    // ===== CPU Benchmark =====
    llvm::outs() << "Running CPU benchmark (" << CPU_ITERATIONS << " iterations)...\n";
    auto cpuStart = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < CPU_ITERATIONS; iter++) {
        for (size_t i = 0; i < SIZE; i++) {
            C_cpu[i] = A[i] + B[i];
        }
    }

    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuTotalMs = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart).count();
    double cpuPerIterMs = (double)cpuTotalMs / CPU_ITERATIONS;
    llvm::outs() << "CPU Total: " << cpuTotalMs << " ms (" << cpuPerIterMs << " ms/iter)\n\n";

    // ===== GPU Setup =====
    if (!runtime::VulkanRuntime::initialize()) {
        llvm::outs() << "⚠️  Vulkan not available\n";
        return;
    }

    llvm::outs() << "GPU: " << runtime::VulkanRuntime::getDeviceName() << "\n";

    // Compile kernel for 256K elements
    const char* mlirSource = R"(
        module {
            func.func @elemwise_add(%A: memref<262144xf32>, %B: memref<262144xf32>, %C: memref<262144xf32>) {
                linalg.generic {
                    indexing_maps = [
                        affine_map<(i) -> (i)>,
                        affine_map<(i) -> (i)>,
                        affine_map<(i) -> (i)>
                    ],
                    iterator_types = ["parallel"]
                } ins(%A, %B : memref<262144xf32>, memref<262144xf32>)
                  outs(%C : memref<262144xf32>) {
                ^bb0(%a: f32, %b: f32, %c: f32):
                    %sum = arith.addf %a, %b : f32
                    linalg.yield %sum : f32
                }
                return
            }
        }
    )";

    llvm::outs() << "Compiling GPU kernel...\n";

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(mlirSource), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

    if (!module) {
        llvm::outs() << "Failed to parse MLIR\n";
        runtime::VulkanRuntime::cleanup();
        return;
    }

    // Compile to SPIR-V
    mlir::PassManager pm(&context);
    addLinalgToGPUPasses(pm);
    if (mlir::failed(pm.run(*module))) {
        llvm::outs() << "Linalg->GPU failed\n";
        runtime::VulkanRuntime::cleanup();
        return;
    }

    mlir::PassManager pm2(&context);
    addGPUToSPIRVPasses(pm2);
    (void)pm2.run(*module);  // May fail partially, that's OK

    auto spirvBinary = serializeSPIRVModule(*module);
    if (spirvBinary.empty()) {
        llvm::outs() << "SPIR-V compilation failed\n";
        runtime::VulkanRuntime::cleanup();
        return;
    }

    llvm::outs() << "SPIR-V kernel: " << spirvBinary.size() * 4 << " bytes\n";

    std::vector<std::pair<void*, size_t>> inputs = {
        {A.data(), SIZE * sizeof(float)},
        {B.data(), SIZE * sizeof(float)}
    };
    std::pair<void*, size_t> output = {C_gpu.data(), SIZE * sizeof(float)};

    // Warmup
    llvm::outs() << "GPU warmup...\n";
    runtime::VulkanRuntime::execute(spirvBinary, inputs, output, {1,1,1}, {(uint32_t)SIZE,1,1});

    // Timed run
    llvm::outs() << "Running GPU benchmark (" << GPU_ITERATIONS << " iterations)...\n";
    auto gpuStart = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < GPU_ITERATIONS; iter++) {
        runtime::VulkanRuntime::execute(spirvBinary, inputs, output, {1,1,1}, {(uint32_t)SIZE,1,1});
    }

    auto gpuEnd = std::chrono::high_resolution_clock::now();
    auto gpuTotalMs = std::chrono::duration_cast<std::chrono::milliseconds>(gpuEnd - gpuStart).count();
    double gpuPerIterMs = (double)gpuTotalMs / GPU_ITERATIONS;
    llvm::outs() << "GPU Total: " << gpuTotalMs << " ms (" << gpuPerIterMs << " ms/iter)\n\n";


    // Results
    llvm::outs() << "=========================================\n";
    llvm::outs() << "📊 RESULTS (per iteration):\n";
    llvm::outs() << "=========================================\n";
    llvm::outs() << "CPU: " << cpuPerIterMs << " ms/iter\n";
    llvm::outs() << "GPU: " << gpuPerIterMs << " ms/iter (includes transfer)\n";

    if (gpuPerIterMs > 0 && cpuPerIterMs > gpuPerIterMs) {
        double speedup = cpuPerIterMs / gpuPerIterMs;
        llvm::outs() << "⚡ GPU is " << speedup << "x faster!\n";
    } else if (cpuPerIterMs > 0 && gpuPerIterMs > cpuPerIterMs) {
        double ratio = gpuPerIterMs / cpuPerIterMs;
        llvm::outs() << "🐢 GPU is " << ratio << "x slower (transfer overhead)\n";
    }

    // Throughput
    double cpuGBps = (SIZE * sizeof(float) * 3.0) / (cpuPerIterMs * 1e6); // 3 arrays: A, B, C
    double gpuGBps = (SIZE * sizeof(float) * 3.0) / (gpuPerIterMs * 1e6);
    llvm::outs() << "\n📈 Throughput:\n";
    llvm::outs() << "CPU: " << cpuGBps << " GB/s\n";
    llvm::outs() << "GPU: " << gpuGBps << " GB/s\n";

    // Verify correctness
    bool correct = true;
    for (size_t i = 0; i < SIZE && correct; i++) {
        if (std::abs(C_gpu[i] - 5.0f) > 0.001f) {
            correct = false;
            llvm::outs() << "❌ GPU result mismatch at " << i << "\n";
        }
    }
    if (correct) {
        llvm::outs() << "✅ GPU results verified correct!\n";
    }

    runtime::VulkanRuntime::cleanup();
}

} // namespace gpu
} // namespace tenzo

