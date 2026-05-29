#pragma once
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "context/HardwareProfile.h"

namespace tenzo {
// Налаштовує контекст: завантажує Tenzo, Func та Arith діалекти
void registerAllDialects(mlir::MLIRContext &context);

// Додає апаратно-залежний пайплайн для оптимізації GEMM
void addHardwareAwareGEMMPipeline(
    mlir::OpPassManager &pm,
    const std::shared_ptr<tenzo::HardwareProfile> &hwInfo);
} // namespace tenzo