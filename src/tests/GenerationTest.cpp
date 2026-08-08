#include "EndToEndMathTest.h"
#include "runtime/ArenaPlanner.h"
#include "runtime/MemRefUtils.h"
#include "runtime/ExecutionContext.h"
#include "runtime/KVCacheManager.h"
#include "runtime/Sampler.h"
#include "runtime/Tokenizer.h"
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
#include <cstring>

using namespace mlir;

namespace tenzo {

void runGenerationTest(MLIRContext& context, const std::string& prompt, int max_tokens) {
    llvm::outs() << "\n=== [Phase 8] Real LLM Generation Pipeline ===\n";
    llvm::outs() << "Prompt: \"" << prompt << "\"\n";
    llvm::outs() << "Max Tokens: " << max_tokens << "\n\n";

    std::vector<std::string> mlir_candidates = {
        "/app/tenzo-frontend/export_output/model.mlir",
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

    llvm::outs() << "✅ Model compiled successfully!\n";

    std::vector<std::string> weights_candidates = {
        "tenzo-frontend/export_output/weights.bin",
        "/app/tenzo-frontend/export_output/weights.bin",
        "weights.bin"
    };

    std::vector<int8_t> weights_bytes;
    for (const auto& candidate : weights_candidates) {
        std::ifstream f(candidate, std::ios::binary);
        if (f.is_open()) {
            f.seekg(0, std::ios::end);
            size_t size = f.tellg();
            f.seekg(0, std::ios::beg);
            weights_bytes.resize(size);
            f.read(reinterpret_cast<char*>(weights_bytes.data()), size);
            llvm::outs() << "✅ Successfully loaded weights (" << size << " bytes)\n";
            break;
        }
    }

    if (weights_bytes.empty()) {
        llvm::errs() << "❌ Failed to load weights.bin!\n";
        return;
    }

    runtime::Tensor weights_tensor({static_cast<int64_t>(weights_bytes.size())}, reinterpret_cast<float*>(weights_bytes.data()));

    runtime::Tokenizer tokenizer;
    std::vector<std::string> vocab_candidates = {
        "tenzo-frontend/export_output/tokenizer.vocab",
        "/app/tenzo-frontend/export_output/tokenizer.vocab",
        "tokenizer.vocab"
    };
    bool loaded_vocab = false;
    for (const auto& path : vocab_candidates) {
        if (tokenizer.load(path)) {
            llvm::outs() << "✅ Loaded Tokenizer (" << tokenizer.vocab_size() << " tokens)\n";
            loaded_vocab = true;
            break;
        }
    }

    runtime::SamplerConfig s_cfg;
    s_cfg.temperature = 0.7f;
    s_cfg.top_p = 0.9f;
    runtime::Sampler sampler(s_cfg);

    runtime::ExecutionContext engine(context, module.get());
    runtime::KVCacheManager kv_cache(1024, 1, 128, 4);
    kv_cache.reset();

    std::vector<int32_t> prompt_tokens;
    if (loaded_vocab) {
        prompt_tokens = tokenizer.encode(prompt);
    }
    if (prompt_tokens.empty()) {
        prompt_tokens = {1, 2, 3};
    }

    llvm::outs() << "\n💬 [Tenzo Engine] Output Stream: " << prompt;
    llvm::outs().flush();

    int32_t current_token = prompt_tokens[0];

    for (int step = 0; step < max_tokens; ++step) {
        if (step < static_cast<int>(prompt_tokens.size())) {
            current_token = prompt_tokens[step];
        }

        std::vector<int64_t> in_shape = {1, 1};
        runtime::Tensor input(in_shape);
        std::memcpy(input.data, &current_token, sizeof(int32_t));

        size_t v_size = loaded_vocab ? tokenizer.vocab_size() : 128;
        std::vector<int64_t> out_shape = {1, 1, static_cast<int64_t>(v_size)};
        runtime::Tensor output(out_shape);

        std::vector<int64_t> seq_pos_shape = {1};
        runtime::Tensor seq_pos_t(seq_pos_shape);
        int32_t pos = kv_cache.get_current_seq_len();
        std::memcpy(seq_pos_t.data, &pos, sizeof(int32_t));

        std::vector<runtime::Tensor*> inputs = {&input, &weights_tensor, kv_cache.get_k_cache(), kv_cache.get_v_cache(), &seq_pos_t};

        std::vector<int64_t> kv_shape = kv_cache.get_k_cache()->shape;
        runtime::Tensor out_k_cache(kv_shape);
        runtime::Tensor out_v_cache(kv_shape);

        std::vector<runtime::Tensor*> outputs = {&output, &out_k_cache, &out_v_cache};

        try {
            engine.forward(inputs, outputs);
        } catch (const std::exception& e) {
            llvm::errs() << "\n❌ Runtime Error during forward pass: " << e.what() << "\n";
            return;
        }

        std::memcpy(kv_cache.get_k_cache()->data, out_k_cache.data, 1024 * 128 * sizeof(float));
        std::memcpy(kv_cache.get_v_cache()->data, out_v_cache.data, 1024 * 128 * sizeof(float));

        if (step >= static_cast<int>(prompt_tokens.size()) - 1) {
            current_token = sampler.sample(output.data, v_size);
            if (loaded_vocab) {
                std::string token_str = tokenizer.decode({current_token});
                llvm::outs() << token_str;
                llvm::outs().flush();
            }
        }

        kv_cache.increment_seq_len(1);
    }
    llvm::outs() << "\n\n🎉 Real LLM Generation Completed Successfully!\n";
}

} // namespace tenzo
