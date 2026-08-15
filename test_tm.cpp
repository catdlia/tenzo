#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include <iostream>

int main() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    std::string error;
    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target) {
        std::cerr << "Error: " << error << "\n";
        return 1;
    }
    llvm::TargetOptions opt;
    auto tm = target->createTargetMachine(targetTriple, llvm::sys::getHostCPUName(), llvm::sys::getHostCPUFeatures().getString(), opt, llvm::Reloc::PIC_);
    auto transformer = mlir::makeOptimizingTransformer(3, 0, tm);
    std::cout << "Success!" << std::endl;
    return 0;
}
