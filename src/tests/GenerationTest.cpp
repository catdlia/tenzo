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

    // Extract model topology metadata BEFORE passes transform the function signature
    int num_heads = 5;
    int num_layers = 1;
    int64_t vocab_size_i64 = 128;
    {
        auto mainFunc = module->lookupSymbol<func::FuncOp>("main");
        if (mainFunc) {
            size_t num_inputs = mainFunc.getFunctionType().getNumInputs();
            // Signature: (token, weights, k0, v0, ..., kN, vN, seq_pos)
            // num_inputs = 3 + 2*num_layers
            if (num_inputs >= 5) {
                num_layers = static_cast<int>((num_inputs - 3) / 2);
            }
            if (auto kType = mlir::dyn_cast<RankedTensorType>(mainFunc.getFunctionType().getInput(2))) {
                if (kType.getRank() == 4) {
                    num_heads = kType.getDimSize(1);
                }
            }
            if (mainFunc.getFunctionType().getNumResults() > 0) {
                if (auto outType = mlir::dyn_cast<RankedTensorType>(mainFunc.getFunctionType().getResult(0))) {
                    if (outType.getRank() == 3) {
                        vocab_size_i64 = outType.getDimSize(2);
                    }
                }
            }
            llvm::outs() << "  Model: " << num_layers << " layers, " << num_heads << " KV heads, vocab " << vocab_size_i64 << "\n";
        }
    }

    PassManager pm(&context);
    addTenzoToLinalgPass(pm);
    addTenzoBufferizationPasses(pm);
    addTenzoToLLVMPasses(pm, /*enableVectorization=*/false, {8, 16, 4}, /*enableParallel=*/true);

    if (failed(pm.run(module.get()))) {
        llvm::errs() << "❌ Compilation failed\n";
        return;
    }

    llvm::outs() << "✅ Model compiled successfully!\n";
    module->dump();
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

    std::vector<int32_t> prompt_tokens;
    if (loaded_vocab) {
        prompt_tokens = tokenizer.encode(config.prompt);
    }
    if (prompt_tokens.empty()) {
        prompt_tokens = {1, 2, 3};
    }

    int prefill_token_count = static_cast<int>(prompt_tokens.size());
    int total_steps = prefill_token_count + config.max_tokens;

    runtime::ExecutionContext engine(context, module.get());
    runtime::KVCacheManager kv_cache(std::max(4096, total_steps + 128), num_layers, 128 * num_heads, num_heads);
    kv_cache.reset();

    llvm::outs() << "\n💬 [Tenzo Engine] Output Stream: " << config.prompt;
    llvm::outs().flush();

    int32_t current_token = prompt_tokens[0];
    std::vector<int32_t> all_tokens = prompt_tokens;

    auto start_time = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point ttft_time;
    bool ttft_captured = false;
    int generated_tokens_count = 0;

    std::vector<int64_t> in_shape = {1, 1};
    runtime::Tensor input(in_shape);
    size_t v_size = vocab_size_i64; // Use model's vocab size!
    std::vector<int64_t> out_shape = {1, 1, static_cast<int64_t>(v_size)};
    runtime::Tensor output(out_shape);
    std::vector<int64_t> seq_pos_shape = {1};
    runtime::Tensor seq_pos_t(seq_pos_shape);

    std::vector<runtime::Tensor*> inputs = {&input, &weights_tensor};
    std::vector<runtime::Tensor*> outputs = {&output};

    for (int l = 0; l < num_layers; ++l) {
        inputs.push_back(kv_cache.get_k_cache(l));
        inputs.push_back(kv_cache.get_v_cache(l));
    }
    inputs.push_back(&seq_pos_t);

    for (int l = 0; l < num_layers; ++l) {
        outputs.push_back(kv_cache.get_k_cache(l));
        outputs.push_back(kv_cache.get_v_cache(l));
    }

    for (int step = 0; step < total_steps; ++step) {
        if (step < prefill_token_count) {
            current_token = prompt_tokens[step];
        }

        std::memcpy(input.data, &current_token, sizeof(int32_t));

        int32_t pos = kv_cache.get_current_seq_len();
        std::memcpy(seq_pos_t.data, &pos, sizeof(int32_t));

        try {
            engine.forward(inputs, outputs);
            if (step == prefill_token_count - 1) {
                llvm::outs() << "\nDEBUG: First 10 logits: ";
                for (int i = 0; i < 10; ++i) {
                    llvm::outs() << output.data[i] << " ";
                }
                llvm::outs() << "\n";
            }
        } catch (const std::exception& e) {
            llvm::errs() << "\n❌ Runtime Error during forward pass: " << e.what() << "\n";
            return;
        }

        // When prefill completes, start decoding and sample tokens
        if (step >= prefill_token_count - 1) {
            current_token = sampler.sample(output.data, v_size, all_tokens);
            all_tokens.push_back(current_token);
            
            if (!ttft_captured) {
                ttft_time = std::chrono::high_resolution_clock::now();
                ttft_captured = true;
            }

            if (step >= prefill_token_count) {
                generated_tokens_count++;
                if (loaded_vocab) {
                    std::string token_str = tokenizer.decode({current_token});
                    if (token_str == "<|endoftext|>" || token_str == "<|eot_id|>" || 
                        token_str == "</s>" || token_str == "<eos>") {
                        llvm::outs() << "\n[EOS reached]\n";
                        break;
                    }
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
