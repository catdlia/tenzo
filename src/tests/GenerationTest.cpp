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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <fstream>
#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>
#include <cstring>
#include <chrono>
#include <iomanip>

using namespace mlir;

namespace tenzo {

void runGenerationTest(MLIRContext& context, const GenerationConfig& config) {
    llvm::outs() << "\n=== [Tenzo Engine] Inference Pipeline ===\n";
    llvm::outs() << "Prompt:      \"" << config.prompt << "\"\n";
    llvm::outs() << "Max Tokens:  " << config.max_tokens << "\n";
    llvm::outs() << "Temperature: " << config.temperature << (config.temperature <= 0.0f ? " (Greedy / ArgMax)" : "") << "\n";
    llvm::outs() << "Top-P:       " << config.top_p << "\n";
    llvm::outs() << "Model Dir:   " << config.model_dir << "\n\n";

    std::vector<std::string> mlir_candidates = {
        config.model_dir + "/model.mlir",
        "/app/" + config.model_dir + "/model.mlir",
        "tenzo-frontend/export_output/model.mlir",
        "model.mlir"
    };

    std::string errorMessage;
    std::unique_ptr<llvm::MemoryBuffer> file;
    for (const auto& candidate : mlir_candidates) {
        file = mlir::openInputFile(candidate, &errorMessage);
        if (file) {
            llvm::outs() << "✅ Loaded MLIR model from: " << candidate << "\n";
            break;
        }
    }

    if (!file) {
        llvm::errs() << "❌ Failed to open model.mlir in: " << config.model_dir << "\n";
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
    addTenzoToLLVMPasses(pm, /*enableVectorization=*/false);

    if (failed(pm.run(module.get()))) {
        llvm::errs() << "❌ Compilation failed\n";
        return;
    }

    llvm::outs() << "✅ Model compiled successfully!\n";
    llvm::outs().flush();

    std::vector<std::string> weights_candidates = {
        config.model_dir + "/weights.bin",
        "/app/" + config.model_dir + "/weights.bin",
        "tenzo-frontend/export_output/weights.bin",
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
            llvm::outs() << "✅ Loaded 2-bit weights: " << size << " bytes\n";
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
        config.model_dir + "/tokenizer.vocab",
        "/app/" + config.model_dir + "/tokenizer.vocab",
        "tenzo-frontend/export_output/tokenizer.vocab",
        "tokenizer.vocab"
    };
    bool loaded_vocab = false;
    for (const auto& path : vocab_candidates) {
        if (tokenizer.load(path)) {
            llvm::outs() << "✅ Loaded Tokenizer: " << tokenizer.vocab_size() << " tokens\n";
            loaded_vocab = true;
            break;
        }
    }

    runtime::SamplerConfig s_cfg;
    s_cfg.temperature = config.temperature;
    s_cfg.top_p = config.top_p;
    runtime::Sampler sampler(s_cfg);

    int num_heads = 4;
    auto mainFunc = module->lookupSymbol<func::FuncOp>("main");
    if (mainFunc && mainFunc.getFunctionType().getNumInputs() >= 3) {
        if (auto kType = mlir::dyn_cast<RankedTensorType>(mainFunc.getFunctionType().getInput(2))) {
            if (kType.getRank() == 4) {
                num_heads = kType.getDimSize(1);
            }
        }
    }

    runtime::ExecutionContext engine(context, module.get());
    runtime::KVCacheManager kv_cache(1024, 1, 128, num_heads);
    kv_cache.reset();

    std::vector<int32_t> prompt_tokens;
    if (loaded_vocab) {
        prompt_tokens = tokenizer.encode(config.prompt);
    }
    if (prompt_tokens.empty()) {
        prompt_tokens = {1, 2, 3};
    }

    llvm::outs() << "\n💬 [Tenzo Engine] Output Stream: " << config.prompt;
    llvm::outs().flush();

    int32_t current_token = prompt_tokens[0];

    auto start_time = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point ttft_time;
    bool ttft_captured = false;
    int generated_tokens_count = 0;

    int prefill_token_count = static_cast<int>(prompt_tokens.size());
    int total_steps = prefill_token_count + config.max_tokens;

    std::vector<int64_t> in_shape = {1, 1};
    runtime::Tensor input(in_shape);
    size_t v_size = loaded_vocab ? tokenizer.vocab_size() : 128;
    std::vector<int64_t> out_shape = {1, 1, static_cast<int64_t>(v_size)};
    runtime::Tensor output(out_shape);
    std::vector<int64_t> seq_pos_shape = {1};
    runtime::Tensor seq_pos_t(seq_pos_shape);
    std::vector<int64_t> kv_shape = kv_cache.get_k_cache()->shape;
    runtime::Tensor out_k_cache(kv_shape);
    runtime::Tensor out_v_cache(kv_shape);
    std::vector<runtime::Tensor*> inputs = {&input, &weights_tensor, kv_cache.get_k_cache(), kv_cache.get_v_cache(), &seq_pos_t};
    std::vector<runtime::Tensor*> outputs = {&output, &out_k_cache, &out_v_cache};

    for (int step = 0; step < total_steps; ++step) {
        if (step < prefill_token_count) {
            current_token = prompt_tokens[step];
        }

        std::memcpy(input.data, &current_token, sizeof(int32_t));

        int32_t pos = kv_cache.get_current_seq_len();
        std::memcpy(seq_pos_t.data, &pos, sizeof(int32_t));

        try {
            engine.forward(inputs, outputs);
        } catch (const std::exception& e) {
            llvm::errs() << "\n❌ Runtime Error during forward pass: " << e.what() << "\n";
            return;
        }

        size_t kv_num_floats = 1;
        for (auto dim : kv_cache.get_k_cache()->shape) kv_num_floats *= dim;
        std::memcpy(kv_cache.get_k_cache()->data, out_k_cache.data, kv_num_floats * sizeof(float));
        std::memcpy(kv_cache.get_v_cache()->data, out_v_cache.data, kv_num_floats * sizeof(float));

        // When prefill completes, start decoding and sample tokens
        if (step >= prefill_token_count - 1) {
            current_token = sampler.sample(output.data, v_size);
            
            if (!ttft_captured) {
                ttft_time = std::chrono::high_resolution_clock::now();
                ttft_captured = true;
            }

            if (step >= prefill_token_count) {
                generated_tokens_count++;
                if (loaded_vocab) {
                    std::string token_str = tokenizer.decode({current_token});
                    llvm::outs() << token_str;
                    llvm::outs().flush();
                }
            }
        }

        kv_cache.increment_seq_len(1);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(ttft_time - start_time).count();
    double decode_ms = std::chrono::duration<double, std::milli>(end_time - ttft_time).count();
    double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double tok_per_sec = (generated_tokens_count > 0 && decode_ms > 0) ? (generated_tokens_count / (decode_ms / 1000.0)) : 0.0;

    llvm::outs() << "\n\n";
    llvm::outs() << "╔════════════════════════════════════════════════════════╗\n";
    llvm::outs() << "║             Tenzo Engine Profiling Summary             ║\n";
    llvm::outs() << "╠════════════════════════════════════════════════════════╣\n";
    llvm::outs() << "║ Prompt Tokens:           " << prefill_token_count << "\n";
    llvm::outs() << "║ Generated Tokens:        " << generated_tokens_count << "\n";
    llvm::outs() << "║ Time To First Token:     " << ttft_ms << " ms\n";
    llvm::outs() << "║ Total Decode Time:       " << decode_ms << " ms\n";
    llvm::outs() << "║ Total Elapsed Time:      " << total_ms << " ms\n";
    llvm::outs() << "║ Decode Speed:            " << tok_per_sec << " tok/sec\n";
    llvm::outs() << "╚════════════════════════════════════════════════════════╝\n";
}

} // namespace tenzo
