#pragma once
#include "mlir/IR/MLIRContext.h"
#include "HardwareInfo.h"

namespace tenzo {
    // Налаштовує контекст: завантажує Tenzo, Func та Arith діалекти
    void registerAllDialects(mlir::MLIRContext &context);
}