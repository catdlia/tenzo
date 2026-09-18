#pragma once

#include "mlir/IR/MLIRContext.h"

namespace tenzo {

void runHeteroPipelineTests();
void runDistributedNetworkTests();
void runHeteroBenchmark();

} // namespace tenzo
