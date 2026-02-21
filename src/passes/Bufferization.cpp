#include "passes/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
// Bufferizable interface implementations
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
void tenzo::addTenzoBufferizationPasses(mlir::OpPassManager &pm) {
    // 1. Підготовка: empty -> alloc_tensor
    pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
    // 2. OneShot Bufferization
    mlir::bufferization::OneShotBufferizePassOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.copyBeforeWrite = true;
    // Options allowUnknownOps no longer exists / is true by default
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));

    // 3. Конвертуємо bufferization ops -> memref ops
    pm.addPass(mlir::createConvertBufferizationToMemRefPass());
    // 4. Очищення
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
}
