#pragma once
#include "mlir/IR/MLIRContext.h"

namespace tenzo {

/// Runs the Zero-Copy Bridge Test loading weights via mmap and executing MLIR
void runZeroCopyBridgeTest(mlir::MLIRContext& context);

} // namespace tenzo
