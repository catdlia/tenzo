#include "ExecutionContext.h"
#include "MemRefUtils.h"
#include "passes/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"

namespace tenzo {
namespace runtime {

ExecutionContext::ExecutionContext(mlir::MLIRContext& context, mlir::ModuleOp module) {
    // 1. Initialize LLVM
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // 2. Register translations
    tenzo::registerAllTenzoDialectTranslations(context);
    mlir::registerLLVMDialectTranslation(context);
    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerOpenMPDialectTranslation(context);

    // 3. Setup JIT options
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

    // 4. Create Engine
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    if (!maybeEngine) {
        throw std::runtime_error("Failed to create ExecutionEngine");
    }
    engine = std::move(maybeEngine.get());
}

void ExecutionContext::forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    std::vector<MemRef2D> inputDescs;
    std::vector<MemRef2D> outputDescs;
    std::vector<void*> args;

    for (auto* in : inputs) {
        inputDescs.push_back(MemRef2D::create(in->data, in->shape));
    }
    for (auto* out : outputs) {
        outputDescs.push_back(MemRef2D::create(out->data, out->shape));
    }

    for (auto& d : inputDescs) args.push_back(&d);
    for (auto& d : outputDescs) args.push_back(&d);
    
    auto error = engine->invokePacked("main", llvm::MutableArrayRef<void*>(args));
    if (error) {
        throw std::runtime_error("JIT invocation failed");
    }
}

} // namespace runtime
} // namespace tenzo
