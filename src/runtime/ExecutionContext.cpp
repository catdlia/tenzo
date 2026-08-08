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
    // Removed makeOptimizingTransformer to prevent segfault (target machine is nullptr)
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
    
    // Add runtime libraries to resolve symbols like memrefCopy
    llvm::SmallVector<llvm::StringRef, 2> sharedLibs = {
        "/usr/lib/llvm-21/lib/libmlir_c_runner_utils.so.21.1",
        "/usr/lib/llvm-21/lib/libmlir_runner_utils.so.21.1"
    };
    engineOptions.sharedLibPaths = sharedLibs;

    // 4. Create Engine
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    if (!maybeEngine) {
        llvm::errs() << "❌ Failed to create ExecutionEngine: " << llvm::toString(maybeEngine.takeError()) << "\n";
        throw std::runtime_error("Failed to create ExecutionEngine");
    }
    engine = std::move(maybeEngine.get());
}

void ExecutionContext::forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    std::vector<void*> args;
    std::vector<std::unique_ptr<void, void(*)(void*)>> memrefs;
    std::vector<void*> desc_ptrs;

    auto push_tensor = [&](Tensor* t) {
        size_t rank = t->shape.size();
        void* desc = nullptr;
        if (rank == 1) {
            auto* d = new MemRef1D(MemRef1D::create((float*)t->data, t->shape));
            memrefs.emplace_back(d, [](void* ptr) { delete static_cast<MemRef1D*>(ptr); });
            desc = d;
        } else if (rank == 2) {
            auto* d = new MemRef2D(MemRef2D::create((float*)t->data, t->shape));
            memrefs.emplace_back(d, [](void* ptr) { delete static_cast<MemRef2D*>(ptr); });
            desc = d;
        } else if (rank == 3) {
            auto* d = new MemRef3D(MemRef3D::create((float*)t->data, t->shape));
            memrefs.emplace_back(d, [](void* ptr) { delete static_cast<MemRef3D*>(ptr); });
            desc = d;
        } else if (rank == 4) {
            auto* d = new MemRef4D(MemRef4D::create((float*)t->data, t->shape));
            memrefs.emplace_back(d, [](void* ptr) { delete static_cast<MemRef4D*>(ptr); });
            desc = d;
        } else {
            throw std::runtime_error("Unsupported tensor rank: " + std::to_string(rank));
        }
        desc_ptrs.push_back(desc);
    };

    for (auto* in : inputs) push_tensor(in);
    for (auto* out : outputs) push_tensor(out);
    
    for (size_t i = 0; i < desc_ptrs.size(); ++i) {
        args.push_back(&desc_ptrs[i]);
    }
    
    auto error = engine->invokePacked("_mlir_ciface_main", llvm::MutableArrayRef<void*>(args));
    if (error) {
        llvm::consumeError(std::move(error));
        // Fallback for tests that don't have _mlir_ciface_main
        error = engine->invokePacked("main", llvm::MutableArrayRef<void*>(args));
        if (error) {
            llvm::consumeError(std::move(error));
            throw std::runtime_error("JIT invocation failed");
        }
    }
}

} // namespace runtime
} // namespace tenzo
