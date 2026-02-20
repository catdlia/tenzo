#pragma once
#include "mlir/IR/MLIRContext.h"

namespace tenzo {
    // Основна точка входу для тестів
    void runFullPipelineTest(mlir::MLIRContext &context);

    // Нова функція для бенчмаркінгу
    void runBenchmark(mlir::MLIRContext &context);

    // Large matrix benchmark (1024x1024)
    void runLargeMatrixBenchmark(mlir::MLIRContext &context);

    // Conv2D test
    void runConv2DTest(mlir::MLIRContext &context);

    // Small matrix test (64x64) for explicit vectorization
    void runSmallMatMulTest(mlir::MLIRContext &context);

    // Transform Dialect test
    void runTransformDialectTest(mlir::MLIRContext &context);

    // Parallel (multithreaded) benchmark test
    void runParallelBenchmark(mlir::MLIRContext &context);
    void runExplicitKernelBenchmark(mlir::MLIRContext &context);

    // Micro-kernel benchmark (6x16) - pure computational performance
    void runMicroKernelBenchmark(mlir::MLIRContext &context);

    // BLIS-style benchmark (packing + explicit micro-kernel)
    void runBLISBenchmark(mlir::MLIRContext &context);

    // Generate MLIR micro-kernel and save to file
    void generateMicroKernelMLIR(mlir::MLIRContext &context);

    // Packing kernels bandwidth benchmark (Etap 3)
    void runPackingBenchmark(mlir::MLIRContext &context);

    // Full GEMM with 5-loop macro-kernel (Etap 3 Day 2)
    void runFullGEMMBenchmark(mlir::MLIRContext &context);

    // Native AVX2 GEMM End-to-End benchmark (Etap 3 - Packing validation)
    void runGEMMEndToEndBenchmark(mlir::MLIRContext &context);

    // Linear Layer Inference benchmark (Etap 3 Day 3 - Real NN inference)
    void runLinearLayerBenchmark();
}