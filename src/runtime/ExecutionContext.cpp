#include "ExecutionContext.h"
#include "MemRefUtils.h"
#include "passes/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/DynamicLibrary.h"

namespace tenzo {
namespace runtime {

ExecutionContext::ExecutionContext(mlir::MLIRContext& context, mlir::ModuleOp module) {
    // 1. Initialize LLVM
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);

    // 2. Register translations
    tenzo::registerAllTenzoDialectTranslations(context);
    mlir::registerLLVMDialectTranslation(context);
    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerOpenMPDialectTranslation(context);

    // 3. Setup JIT options
    mlir::ExecutionEngineOptions engineOptions;
    
    std::string error;
    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (target) {
        llvm::TargetOptions opt;
        tm = std::unique_ptr<llvm::TargetMachine>(
            target->createTargetMachine(targetTriple, llvm::sys::getHostCPUName(), "", opt, llvm::Reloc::PIC_)
        );
    } else {
        llvm::errs() << "Warning: Could not create TargetMachine for optimizing transformer: " << error << "\n";
    }

    auto transformer = tm ? mlir::makeOptimizingTransformer(3, 0, tm.get()) : nullptr;
    if (transformer) {
        engineOptions.transformer = transformer;
    }
    engineOptions.enableObjectDump = true;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
    
    // Add runtime libraries to resolve symbols like memrefCopy and OpenMP kmpc symbols
    llvm::SmallVector<llvm::StringRef, 4> sharedLibs = {
        "/usr/lib/llvm-21/lib/libmlir_c_runner_utils.so.21.1",
        "/usr/lib/llvm-21/lib/libmlir_runner_utils.so.21.1",
        "/usr/lib/llvm-21/lib/libomp.so",
        "/usr/lib/x86_64-linux-gnu/libgomp.so.1"
    };
    engineOptions.sharedLibPaths = sharedLibs;

    std::error_code ec;
    llvm::raw_fd_ostream os("/app/optimized.mlir", ec, llvm::sys::fs::OF_None);
    module->print(os);
    os.close();

    // 4. Create Engine
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    if (!maybeEngine) {
        llvm::errs() << "❌ Failed to create ExecutionEngine: " << llvm::toString(maybeEngine.takeError()) << "\n";
        throw std::runtime_error("Failed to create ExecutionEngine");
    }
    engine = std::move(maybeEngine.get());
}

ExecutionContext::~ExecutionContext() = default;

#include <deque>

void ExecutionContext::forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    std::vector<void*> args;
    std::vector<void*> desc_ptrs;
    std::deque<MemRef1D> memrefs1D;
    std::deque<MemRef2D> memrefs2D;
    std::deque<MemRef3D> memrefs3D;
    std::deque<MemRef4D> memrefs4D;

    auto push_tensor = [&](Tensor* t) {
        size_t rank = t->shape.size();
        if (rank == 1) {
            memrefs1D.push_back(MemRef1D::create((float*)t->data, t->shape));
            desc_ptrs.push_back(&memrefs1D.back());
        } else if (rank == 2) {
            memrefs2D.push_back(MemRef2D::create((float*)t->data, t->shape));
            desc_ptrs.push_back(&memrefs2D.back());
        } else if (rank == 3) {
            memrefs3D.push_back(MemRef3D::create((float*)t->data, t->shape));
            desc_ptrs.push_back(&memrefs3D.back());
        } else if (rank == 4) {
            memrefs4D.push_back(MemRef4D::create((float*)t->data, t->shape));
            desc_ptrs.push_back(&memrefs4D.back());
        } else {
            throw std::runtime_error("Unsupported tensor rank: " + std::to_string(rank));
        }
    };

    for (auto* in : inputs) push_tensor(in);
    for (auto* out : outputs) push_tensor(out);
    
    for (size_t i = 0; i < desc_ptrs.size(); ++i) {
        args.push_back(&desc_ptrs[i]);
    }
    
    auto error = engine->invokePacked("_mlir_ciface_main", llvm::MutableArrayRef<void*>(args));
    if (error) {
        llvm::consumeError(std::move(error));
        error = engine->invokePacked("main", llvm::MutableArrayRef<void*>(args));
        if (error) {
            llvm::consumeError(std::move(error));
            throw std::runtime_error("JIT invocation failed");
        }
    }

    // Update output tensor pointers from descriptor aligned pointers
    for (size_t i = 0; i < outputs.size(); ++i) {
        void* desc = desc_ptrs[inputs.size() + i];
        size_t rank = outputs[i]->shape.size();
        float* new_ptr = nullptr;
        if (rank == 1) new_ptr = static_cast<MemRef1D*>(desc)->aligned;
        else if (rank == 2) new_ptr = static_cast<MemRef2D*>(desc)->aligned;
        else if (rank == 3) new_ptr = static_cast<MemRef3D*>(desc)->aligned;
        else if (rank == 4) new_ptr = static_cast<MemRef4D*>(desc)->aligned;

        if (new_ptr) {
            outputs[i]->data = new_ptr;
            outputs[i]->is_owned = false;
        }
    }
}

} // namespace runtime
} // namespace tenzo
