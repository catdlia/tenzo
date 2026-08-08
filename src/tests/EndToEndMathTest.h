#pragma once

#include "mlir/IR/MLIRContext.h"
#include <string>

namespace tenzo {

struct GenerationConfig {
    std::string prompt = "Tenzo Edge AI";
    int max_tokens = 20;
    float temperature = 0.7f;
    float top_p = 0.9f;
    std::string model_dir = "tenzo-frontend/export_output";
};

void runEndToEndMathTest(mlir::MLIRContext& context);
void runGenerationTest(mlir::MLIRContext& context, const GenerationConfig& config = GenerationConfig());

} // namespace tenzo
