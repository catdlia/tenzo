#pragma once

#include "mlir/IR/MLIRContext.h"

namespace tenzo {

void runEndToEndMathTest(mlir::MLIRContext& context);
void runGenerationTest(mlir::MLIRContext& context, const std::string& prompt = "Tenzo Edge AI", int max_tokens = 20);

} // namespace tenzo
