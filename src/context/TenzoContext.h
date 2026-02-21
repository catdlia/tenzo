#pragma once
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "HardwareInfo.h"

namespace tenzo {
    // Налаштовує контекст: завантажує Tenzo, Func та Arith діалекти
    void registerAllDialects(mlir::MLIRContext &context);

    // Додає апаратно-залежний пайплайн для оптимізації GEMM
    void addHardwareAwareGEMMPipeline(mlir::OpPassManager &pm, 
                                       const HardwareInfo &hwInfo);
}