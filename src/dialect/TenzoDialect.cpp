#include "TenzoDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace tenzo;

// Підключаємо згенеровану реалізацію діалекту
#include "TenzoDialect.cpp.inc"

// Підключаємо згенеровану реалізацію операцій
#define GET_OP_CLASSES
#include "TenzoOps.cpp.inc"

// Ініціалізація діалекту
void TenzoDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "TenzoOps.cpp.inc"
    >();
}