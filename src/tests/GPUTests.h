#pragma once
#include "mlir/IR/MLIRContext.h"

namespace tenzo {
namespace gpu {

/// Test GPU pipeline: Linalg -> GPU -> SPIR-V
/// Returns true if test passes
bool runGPUPipelineTest(mlir::MLIRContext &context);

/// Benchmark GPU vs CPU MatMul
void runGPUBenchmark(mlir::MLIRContext &context);

} // namespace gpu
} // namespace tenzo

