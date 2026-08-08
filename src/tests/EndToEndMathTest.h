#pragma once

#include "mlir/IR/MLIRContext.h"

namespace tenzo {

void runEndToEndMathTest(mlir::MLIRContext& context);
void runGenerationTest(mlir::MLIRContext& context);

} // namespace tenzo
