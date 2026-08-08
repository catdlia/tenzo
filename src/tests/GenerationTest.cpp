#include "EndToEndMathTest.h"
#include "runtime/ArenaPlanner.h"
#include "runtime/MemRefUtils.h"
#include "runtime/ExecutionContext.h"
#include "runtime/KVCacheManager.h"
#include "passes/Passes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <fstream>
#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>

using namespace mlir;

namespace tenzo {

static std::vector<float> readFloatBinaryFile(const std::vector<std::string>& candidate_paths) {
    for (const auto& path : candidate_paths) {
        std::ifstream file(path, std::ios::binary);
        if (file.is_open()) {
            file.seekg(0, std::ios::end);
            size_t size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<float> data(size / sizeof(float));
            file.read(reinterpret_cast<char*>(data.data()), size);
            llvm::outs() << "✅ Successfully loaded " << path << " (" << data.size() << " floats, " << size << " bytes)\n";
            return data;
        }
    }
    return {};
}

void runGenerationTest(MLIRContext& context) {
    llvm::outs() << "\n=== [Phase 5] Autoregressive Generation Test ===\n";

    std::vector<std::string> mlir_candidates = {
        "/app/tenzo-frontend/export_output/model.mlir",
        "../tenzo-frontend/export_output/model.mlir",
        "tenzo-frontend/export_output/model.mlir",
        "model.mlir"
    };

    std::string errorMessage;
    std::unique_ptr<llvm::MemoryBuffer> file;
    for (const auto& candidate : mlir_candidates) {
        file = mlir::openInputFile(candidate, &errorMessage);
        if (file) {
            llvm::outs() << "✅ Successfully loaded model from " << candidate << "\n";
            break;
        }
    }

    if (!file) {
        llvm::errs() << "❌ Failed to open model.mlir: " << errorMessage << "\n";
        return;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

    OwningOpRef<ModuleOp> module = mlir::parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "❌ Failed to parse MLIR module\n";
        return;
    }

    PassManager pm(&context);
    addTenzoToLinalgPass(pm);
    addTenzoBufferizationPasses(pm);
    addTenzoToLLVMPasses(pm);

    if (failed(pm.run(module.get()))) {
        llvm::errs() << "❌ Compilation failed\n";
        module->dump();
        return;
    }

    llvm::outs() << "=== MODULE AFTER COMPILATION PASSES ===\n";
    module->dump();
    llvm::outs() << "=======================================\n";

    llvm::outs() << "✅ Model compiled successfully!\n";

    std::vector<std::string> input_candidates = {
        "tenzo-frontend/export_output/input.bin",
        "export_output/input.bin",
        "/app/tenzo-frontend/export_output/input.bin",
        "input.bin"
    };

    std::vector<std::string> expected_candidates = {
        "tenzo-frontend/export_output/expected.bin",
        "export_output/expected.bin",
        "/app/tenzo-frontend/export_output/expected.bin",
        "expected.bin"
    };

    std::vector<float> input_data = readFloatBinaryFile(input_candidates);
    std::vector<float> expected_data = readFloatBinaryFile(expected_candidates);

    if (input_data.empty() || expected_data.empty()) {
        llvm::errs() << "❌ Failed to load input.bin or expected.bin!\n";
        return;
    }

    std::vector<std::string> weights_candidates = {
        "tenzo-frontend/export_output/weights.bin",
        "export_output/weights.bin",
        "/app/tenzo-frontend/export_output/weights.bin",
        "weights.bin"
    };

    std::vector<int8_t> weights_bytes;
    for (const auto& candidate : weights_candidates) {
        std::ifstream file(candidate, std::ios::binary);
        if (file.is_open()) {
            file.seekg(0, std::ios::end);
            size_t size = file.tellg();
            file.seekg(0, std::ios::beg);
            weights_bytes.resize(size);
            file.read(reinterpret_cast<char*>(weights_bytes.data()), size);
            llvm::outs() << "✅ Successfully loaded weights from " << candidate << " (" << size << " bytes)\n";
            break;
        }
    }

    if (weights_bytes.empty()) {
        llvm::errs() << "❌ Failed to load weights.bin!\n";
        return;
    }

    runtime::Tensor weights_tensor({static_cast<int64_t>(weights_bytes.size())}, reinterpret_cast<float*>(weights_bytes.data()));

    runtime::ExecutionContext engine(context, module.get());
    
    // Create KVCacheManager: 1 layer, max_seq_len = 1024, embed_dim = 128, num_heads = 4
    runtime::KVCacheManager kv_cache(1024, 1, 128, 4);
    kv_cache.reset();
    
    // Simulate autoregressive loop (5 tokens)
    llvm::outs() << "🔄 Starting Autoregressive Generation Loop (5 tokens)...\n";
    
    for (int i = 0; i < 5; ++i) {
        llvm::outs() << "  -> Generating Token " << (i + 1) << " (seq_pos = " << kv_cache.get_current_seq_len() << ")\n";
        
        std::vector<int64_t> in_shape = {1, 1, 128};
        runtime::Tensor input(in_shape);
        // Copy the i-th token from input_data
        std::memcpy(input.data, input_data.data() + i * 128, 128 * sizeof(float));

        // Output tensor [1, 1, 128]
        runtime::Tensor output(in_shape);
        
        // seq_pos scalar [1]
        std::vector<int64_t> seq_pos_shape = {1};
        runtime::Tensor seq_pos_t(seq_pos_shape);
        int32_t pos = kv_cache.get_current_seq_len();
        std::memcpy(seq_pos_t.data, &pos, sizeof(int32_t)); 

        std::vector<runtime::Tensor*> inputs = {&input, &weights_tensor, kv_cache.get_k_cache(), kv_cache.get_v_cache(), &seq_pos_t};
        
        // Allocate fresh buffers for outputs to prevent memcpy UB (src == dst) during BufferResultsToOutParams
        std::vector<int64_t> kv_shape = kv_cache.get_k_cache()->shape;
        runtime::Tensor out_k_cache(kv_shape);
        runtime::Tensor out_v_cache(kv_shape);
        
        // The outputs are: output tensor, updated k cache, updated v cache
        std::vector<runtime::Tensor*> outputs = {&output, &out_k_cache, &out_v_cache};

        try {
            engine.forward(inputs, outputs);
        } catch (const std::exception& e) {
            llvm::errs() << "❌ Runtime Error during forward pass: " << e.what() << "\n";
            return;
        }

        // Copy the updated K and V caches back to the manager
        std::memcpy(kv_cache.get_k_cache()->data, out_k_cache.data, 1024 * 128 * sizeof(float));
        std::memcpy(kv_cache.get_v_cache()->data, out_v_cache.data, 1024 * 128 * sizeof(float));


        // Verify output token
        float max_err = 0;
        for (int j = 0; j < 128; ++j) {
            float py_val = expected_data[i * 128 + j];
            float cpp_val = output.data[j];
            float err = std::abs(py_val - cpp_val);
            if (err > max_err) max_err = err;
        }
        
        if (max_err > 1e-4) {
            llvm::errs() << "❌ FAILED: Token " << (i + 1) << " diverged! Max err: " << max_err << "\n";
            return;
        } else {
            llvm::outs() << "    ✅ Token " << (i + 1) << " matches PyTorch (max err: " << max_err << ")\n";
        }

        kv_cache.increment_seq_len(1);
    }
    
    llvm::outs() << "🎉 Generation loop completed successfully!\n";
}

} // namespace tenzo
