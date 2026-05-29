#pragma once
#include "context/HardwareProfile.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace tenzo {

void populateTenzoToArithConversionPatterns(mlir::RewritePatternSet &patterns);
void populateFusionPatterns(mlir::RewritePatternSet &patterns);
void populateTenzoToLinalgConversionPatterns(mlir::RewritePatternSet &patterns);

// Lower Tenzo dialect to Linalg dialect
void addTenzoToLinalgPass(mlir::OpPassManager &pm);

// Optimal Vectorization (GotoBLAS-style, FINAL VERSION)
void addOptimalVectorizationPass(mlir::OpPassManager &pm);

// BLIS-Style Optimization Passes (NEW - for 40+ GFLOPS)
void addPackingPass(mlir::OpPassManager &pm,
                    const MicroKernelParams &params = {});
                    
void addExplicitMicroKernelPass(mlir::OpPassManager &pm, const MicroKernelParams &params = {});

// GotoBLAS-style Packing Kernels (Etap 3 - The Packing)
std::unique_ptr<mlir::Pass> createGeneratePackingKernelsPass();

// GotoBLAS-style Macro-Kernel with 5-Loop Nest (Etap 3 Day 2)
std::unique_ptr<mlir::Pass>
createGenerateMacroKernelPass(const MicroKernelParams &params = {});

// Буферизація
void addTenzoBufferizationPasses(mlir::OpPassManager &pm);

// Transform Dialect Strategy (NEW)
void addTransformStrategyPass(
    mlir::OpPassManager &pm,
    tenzo::TileSizes tiles = {8, 16, 4});

// LLVM Lowering з hardware-aware тайлінгом та паралельністю
void addTenzoToLLVMPasses(
    mlir::OpPassManager &pm, bool enableAffineTiling = false,
    tenzo::TileSizes tiles = {8, 16, 4},
    bool enableParallel = false, bool useExplicitKernel = false);

// Register all translation interfaces for ExecutionEngine in MLIR 21
void registerAllTenzoDialectTranslations(mlir::MLIRContext &context);
} // namespace tenzo