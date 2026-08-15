#include "context/TenzoContext.h"
#include "runtime/ExecutionContext.h"
#include "runtime/MemRefUtils.h"
#include "passes/Passes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include <fstream>
#include <vector>
#include <iostream>

using namespace mlir;

namespace tenzo {

void runTestTL1(MLIRContext& context) {
    llvm::outs() << "--- Running Test TL1 ---\n";
    auto module = mlir::parseSourceFile<mlir::ModuleOp>("/app/test_tl1.mlir", &context);
    if (!module) {
        llvm::errs() << "Failed to load test_tl1.mlir\n";
        return;
    }

    PassManager pm(&context);
    addTenzoToLinalgPass(pm);
    addTenzoBufferizationPasses(pm);
    addTenzoToLLVMPasses(pm, false, {8, 16, 4}, true);

    if (failed(pm.run(module.get()))) {
        llvm::errs() << "❌ Compilation failed\n";
        return;
    }

    std::vector<int8_t> weights;
    std::ifstream file("/app/weights_tl1_test.bin", std::ios::binary);
    if (file) {
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        weights.resize(size);
        file.read(reinterpret_cast<char*>(weights.data()), size);
    } else {
        llvm::errs() << "Failed to load weights\n";
        return;
    }

    runtime::ExecutionContext engine(context, module.get());

    std::vector<int64_t> in_shape = {1, 1, 128};
    runtime::Tensor input(in_shape);
    for (int i = 0; i < 128; ++i) {
        input.data[i] = 2.0f; // Fill with 2.0
    }

    std::vector<int64_t> w_shape = {1, 64, 64}; // n_blocks=1, k_half=64, 64 bytes
    runtime::Tensor w_tensor(w_shape); // Use default float Tensor but we'll cast its memory
    std::memcpy(reinterpret_cast<int8_t*>(w_tensor.data), weights.data(), weights.size());

    std::vector<int64_t> out_shape = {1, 1, 128};
    runtime::Tensor output(out_shape);

    std::vector<runtime::Tensor*> inputs = {&input, &w_tensor};
    std::vector<runtime::Tensor*> outputs = {&output};

    engine.forward(inputs, outputs);

    llvm::outs() << "Output: ";
    for (int i = 0; i < 10; ++i) {
        llvm::outs() << output.data[i] << " ";
    }
    llvm::outs() << "\n";
}

}
