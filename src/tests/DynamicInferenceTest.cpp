#include "DynamicInferenceTest.h"
#include "runtime/ExecutionContext.h"
#include "runtime/MemRefUtils.h"
#include "context/TenzoContext.h"
#include "passes/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "dialect/TenzoDialect.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace tenzo::runtime;

namespace tenzo {

static ModuleOp createDynamicMatMulModule(MLIRContext& context) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // tensor<?x128xf32> * tensor<128x64xf32> -> tensor<?x64xf32>
    SmallVector<int64_t, 2> lhsShape = {ShapedType::kDynamic, 128};
    SmallVector<int64_t, 2> rhsShape = {128, 64};
    SmallVector<int64_t, 2> outShape = {ShapedType::kDynamic, 64};

    auto lhsType = RankedTensorType::get(lhsShape, builder.getF32Type());
    auto rhsType = RankedTensorType::get(rhsShape, builder.getF32Type());
    auto outType = RankedTensorType::get(outShape, builder.getF32Type());

    auto funcType = builder.getFunctionType({lhsType, rhsType}, {outType});
    auto func = builder.create<func::FuncOp>(loc, "main", funcType);
    auto block = func.addEntryBlock();
    builder.setInsertionPointToEnd(block);

    // Create tenzo.matmul
    auto matmulOp = builder.create<tenzo::MatMulOp>(loc, outType, 
        block->getArgument(0), block->getArgument(1));
    
    builder.create<func::ReturnOp>(loc, matmulOp.getResult());
    return module;
}

void runDynamicInferenceTest(MLIRContext& context) {
    llvm::outs() << "\n=== [Phase 5] Dynamic Inference Test ===\n";

    auto module = createDynamicMatMulModule(context);
    
    // Compile using OpPassManager
    PassManager pm(&context);
    
    addTenzoToLinalgPass(pm);
    addTenzoBufferizationPasses(pm);
    addTenzoToLLVMPasses(pm);

    if (failed(pm.run(module))) {
        llvm::errs() << "Compilation failed\n";
        module.dump();
        return;
    }

    llvm::outs() << "module after compilation:\n";
    module.dump();

    ExecutionContext engine(context, module);

    auto test_batch = [&](int batch) {
        llvm::outs() << "Running with batch=" << batch << "...\n";
        Tensor lhs({batch, 128});
        Tensor rhs({128, 64});
        Tensor out({batch, 64});

        // Fill with 1s
        for (int i = 0; i < batch * 128; ++i) lhs.data[i] = 1.0f;
        for (int i = 0; i < 128 * 64; ++i) rhs.data[i] = 1.0f;

        std::vector<Tensor*> inputs = {&lhs, &rhs};
        std::vector<Tensor*> outputs = {&out};

        engine.forward(inputs, outputs);

        // Verify: each element should be 128.0
        bool ok = true;
        for (int i = 0; i < batch * 64; ++i) {
            if (std::abs(out.data[i] - 128.0f) > 1e-5) {
                ok = false;
                break;
            }
        }
        llvm::outs() << "  Status: " << (ok ? "✅ PASSED" : "❌ FAILED") << "\n";
    };

    test_batch(1);
    test_batch(4);
    test_batch(8);
}

} // namespace tenzo
