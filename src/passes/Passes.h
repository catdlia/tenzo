#pragma once
#include <memory>
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "context/HardwareInfo.h"

namespace tenzo {
    struct MicroKernelParams {
        int64_t MR = 6;
        int64_t NR = 16;
        int64_t KC = 256;
        int64_t MC = 96;
        int64_t NC = 256;
        int64_t VEC_SIZE = 8; // AVX2 = 8, AVX512 = 16
    };

    void populateTenzoToArithConversionPatterns(mlir::RewritePatternSet &patterns);
    void populateFusionPatterns(mlir::RewritePatternSet &patterns);
    void populateTenzoToLinalgConversionPatterns(mlir::RewritePatternSet &patterns);

    // Векторизація
    void addTenzoVectorizationPass(mlir::OpPassManager &pm);

    // GotoBLAS-style Micro-kernel (explicit FMA generation)
    void addMicroKernelPass(mlir::OpPassManager &pm);

    // Explicit Micro-kernel (NEW - optimal register allocation)
    void addExplicitMicroKernelPass(mlir::OpPassManager &pm, 
                                     const MicroKernelParams &params = {});

    // Optimal Vectorization (GotoBLAS-style, FINAL VERSION)
    void addOptimalVectorizationPass(mlir::OpPassManager &pm);

    // BLIS-Style Optimization Passes (NEW - for 40+ GFLOPS)
    void addPackingPass(mlir::OpPassManager &pm,
                        const MicroKernelParams &params = {});

    // GotoBLAS-style Packing Kernels (Etap 3 - The Packing)
    std::unique_ptr<mlir::Pass> createGeneratePackingKernelsPass();

    // GotoBLAS-style Macro-Kernel with 5-Loop Nest (Etap 3 Day 2)
    std::unique_ptr<mlir::Pass> createGenerateMacroKernelPass(
        const MicroKernelParams &params = {});

    // Буферизація
    void addTenzoBufferizationPasses(mlir::OpPassManager &pm);

    // Transform Dialect Strategy (NEW)
    void addTransformStrategyPass(mlir::OpPassManager &pm,
                                   HardwareInfo::TileSizes tiles = {8, 16, 4});

    // LLVM Lowering з hardware-aware тайлінгом та паралельністю
    void addTenzoToLLVMPasses(mlir::OpPassManager &pm,
                              bool enableAffineTiling = false,
                              HardwareInfo::TileSizes tiles = {8, 16, 4},
                              bool enableParallel = false,
                              bool useExplicitKernel = false);

    // Register all translation interfaces for ExecutionEngine in MLIR 21
    void registerAllTenzoDialectTranslations(mlir::MLIRContext &context);
}