#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/UB/UBToLLVMIRTranslation.h"
int main() {
    mlir::MLIRContext context;
    mlir::ub::registerUBToLLVMIRTranslation(context);
    return 0;
}
