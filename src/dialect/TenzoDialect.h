#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"            // Щоб знати, що таке TensorType
#include "mlir/Interfaces/SideEffectInterfaces.h" // Щоб знати про Pure/SideEffects
#include "mlir/Interfaces/InferTypeOpInterface.h" // Щоб знати про InferTypeOpInterface
// ---------------------------

#include "TenzoDialect.h.inc"

#define GET_OP_CLASSES
#include "TenzoOps.h.inc"