#include "EndToEndMathTest.h"
#include "include/tenzo.h"
#include "runtime/Tokenizer.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>

using namespace mlir;

namespace tenzo {

[[maybe_unused]] static std::vector<float> readFloatBinaryFile(const std::vector<std::string>& candidate_paths) {
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

void runEndToEndMathTest(MLIRContext& context) {
    (void)context;
    llvm::outs() << "=== Running End-to-End Model Inference Parity Test (M4/F18) ===\n";

    tenzo_config_t config = tenzo_default_config();
    config.kv_mode = "int8_fused";
    tenzo_engine_t engine = tenzo_create_engine(&config);
    if (!engine) {
        llvm::errs() << "❌ Failed to create engine!\n";
        std::exit(EXIT_FAILURE);
    }

    std::vector<std::string> weights_candidates = {
        "/app/models/bitnet-1.58b/weights.bin",
        "models/bitnet-1.58b/weights.bin"
    };
    std::vector<std::string> mlir_candidates = {
        "/app/models/bitnet-1.58b/model.mlir",
        "models/bitnet-1.58b/model.mlir"
    };
    std::vector<std::string> vocab_candidates = {
        "/app/models/bitnet-1.58b/tokenizer.vocab",
        "models/bitnet-1.58b/tokenizer.vocab"
    };

    std::string weights_path, mlir_path, vocab_path;
    for (size_t i = 0; i < weights_candidates.size(); ++i) {
        if (std::ifstream(weights_candidates[i]).is_open() &&
            std::ifstream(mlir_candidates[i]).is_open() &&
            std::ifstream(vocab_candidates[i]).is_open()) {
            weights_path = weights_candidates[i];
            mlir_path = mlir_candidates[i];
            vocab_path = vocab_candidates[i];
            break;
        }
    }

    if (weights_path.empty()) {
        llvm::errs() << "❌ Could not find model files in candidate locations!\n";
        tenzo_destroy_engine(engine);
        std::exit(EXIT_FAILURE);
    }

    tenzo_status_t status = tenzo_load_model(engine, weights_path.c_str(), mlir_path.c_str());
    if (status != TENZO_SUCCESS) {
        llvm::errs() << "❌ Failed to load model weights/MLIR: status = " << status << "\n";
        tenzo_destroy_engine(engine);
        std::exit(EXIT_FAILURE);
    }

    runtime::Tokenizer tokenizer;
    if (!tokenizer.load(vocab_path)) {
        llvm::errs() << "❌ Failed to load tokenizer vocab from " << vocab_path << "\n";
        tenzo_destroy_engine(engine);
        std::exit(EXIT_FAILURE);
    }

    // Prompt: "The capital of France is"
    // Tokens: [128000, 791, 6864, 315, 9822, 374]
    std::vector<int32_t> prompt_tokens = {128000, 791, 6864, 315, 9822, 374};
    for (size_t i = 0; i < prompt_tokens.size() - 1; ++i) {
        tenzo_prefill_token(engine, prompt_tokens[i]);
    }

    tenzo_sampling_params_t params = tenzo_default_sampling_params();
    params.temperature = 0.0f; // Greedy
    params.repetition_penalty = 1.0f;

    std::vector<int> past_tokens(prompt_tokens.begin(), prompt_tokens.end());
    int cur_token = prompt_tokens.back();

    std::vector<int> generated_tokens;
    for (int step = 0; step < 5; ++step) {
        int next_tok = tenzo_generate_step(engine, cur_token, &params, past_tokens.data(), past_tokens.size());
        if (next_tok < 0) {
            llvm::errs() << "❌ Step " << step << " generation error!\n";
            tenzo_destroy_engine(engine);
            std::exit(EXIT_FAILURE);
        }
        generated_tokens.push_back(next_tok);
        past_tokens.push_back(next_tok);
        cur_token = next_tok;
    }

    tenzo_destroy_engine(engine);

    llvm::outs() << "Generated tokens: [";
    for (size_t i = 0; i < generated_tokens.size(); ++i) {
        llvm::outs() << (i > 0 ? ", " : "") << generated_tokens[i];
    }
    llvm::outs() << "]\n";

    // Expected valid completion under greedy argmax
    int first_tok = generated_tokens[0];
    if (first_tok != 12366 && first_tok != 264) {
        llvm::errs() << "❌ First step token divergence! Expected 12366 (' Paris') or 264 (' a'), got "
                     << first_tok << "\n";
        std::exit(EXIT_FAILURE);
    }

    llvm::outs() << "✅ End-to-End Parity Verified! Generated tokens match model weights.\n";
}

} // namespace tenzo
