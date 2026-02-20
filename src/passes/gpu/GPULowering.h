#pragma once
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"
#include <vector>
#include <cstdint>

namespace tenzo {
namespace gpu {

/// Adds passes to lower Linalg ops to GPU dialect
void addLinalgToGPUPasses(mlir::OpPassManager &pm);

/// Adds passes to lower GPU dialect to SPIR-V
void addGPUToSPIRVPasses(mlir::OpPassManager &pm);

/// Serializes SPIR-V module to binary
/// Returns empty vector on failure
std::vector<uint32_t> serializeSPIRVModule(mlir::ModuleOp module);

} // namespace gpu
} // namespace tenzo

