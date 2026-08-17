/**
 * @file basic_inference.cpp
 * @brief Standalone C++ Inference Example using Tenzo SDK (Phase E)
 * 
 * Demonstrates embedding the Tenzo Native AI Engine in pure C++ with ZERO Python dependencies.
 */

#include "tenzo.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <chrono>
#include <cstring>

class SimpleTokenizer {
public:
    std::unordered_map<int, std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;

    bool load_vocab(const std::string& vocab_path) {
        std::ifstream f(vocab_path);
        if (!f.is_open()) return false;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                int tok_id = std::stoi(line.substr(0, space_pos));
                std::string tok_str = line.substr(space_pos + 1);
                // Replace Ġ with space
                size_t pos = 0;
                while ((pos = tok_str.find("\xc4\xa0", pos)) != std::string::npos) {
                    tok_str.replace(pos, 2, " ");
                    pos += 1;
                }
                id_to_token[tok_id] = tok_str;
                token_to_id[tok_str] = tok_id;
            }
        }
        return true;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> tokens;
        if (token_to_id.count("<|begin_of_text|>")) {
            tokens.push_back(token_to_id.at("<|begin_of_text|>"));
        } else if (id_to_token.count(128000)) {
            tokens.push_back(128000);
        }

        size_t i = 0;
        while (i < text.length()) {
            bool matched = false;
            size_t max_len = std::min(static_cast<size_t>(32), text.length() - i);
            for (size_t len = max_len; len > 0; --len) {
                std::string sub = text.substr(i, len);
                auto it = token_to_id.find(sub);
                if (it != token_to_id.end()) {
                    tokens.push_back(it->second);
                    i += len;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                i += 1;
            }
        }
        return tokens;
    }

    std::string decode(const std::vector<int>& token_ids) const {
        std::string res;
        for (int tid : token_ids) {
            auto it = id_to_token.find(tid);
            if (it != id_to_token.end()) {
                res += it->second;
            }
        }
        return res;
    }
};

int main(int argc, char** argv) {
    std::string prompt = "In computer science, a compiler translates source code written in a high-level programming language into";
    int max_tokens = 50;
    float temp = 0.7f;
    std::string model_dir = "/app/tenzo-frontend/export_output";
    std::string kv_mode = "int8_fused";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temp = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--kv-quant") == 0 && i + 1 < argc) {
            kv_mode = argv[++i];
        }
    }

    std::cout << "============================================================" << std::endl;
    std::cout << "🚀 Tenzo Native C++ SDK Standalone Inference Engine (v0.3.0)" << std::endl;
    std::cout << "============================================================" << std::endl;

    std::string vocab_path = model_dir + "/tokenizer.vocab";
    std::string mlir_path = model_dir + "/model.mlir";
    std::string weights_path = model_dir + "/weights.bin";

    SimpleTokenizer tokenizer;
    if (!tokenizer.load_vocab(vocab_path)) {
        std::cerr << "❌ Failed to load tokenizer vocab from " << vocab_path << std::endl;
        return 1;
    }
    std::cout << "📖 Loaded vocabulary (" << tokenizer.id_to_token.size() << " tokens)" << std::endl;

    tenzo_config_t config = tenzo_default_config();
    config.kv_mode = kv_mode.c_str();

    std::cout << "⚙️ Initializing Tenzo Engine (KV Mode: " << kv_mode << ")..." << std::endl;
    tenzo::Engine engine(config);

    std::cout << "📦 Loading model weights & MLIR graph from " << model_dir << "..." << std::endl;
    auto t_load_start = std::chrono::high_resolution_clock::now();
    engine.load_model(weights_path, mlir_path);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_time_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    std::cout << "✅ Model loaded in " << load_time_ms << " ms" << std::endl;

    std::vector<int> prompt_tokens = tokenizer.encode(prompt);
    if (prompt_tokens.empty()) {
        prompt_tokens = {128000, 1, 2};
    }

    std::cout << "\n💬 [Tenzo C++ Engine] Output Stream: " << prompt << std::flush;

    // 1. Prefill Prompt Tokens
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < prompt_tokens.size() - 1; ++i) {
        engine.prefill_token(prompt_tokens[i]);
    }
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

    // 2. Autoregressive Decode Loop
    tenzo_sampling_params_t params = tenzo_default_sampling_params();
    params.temperature = temp;

    std::vector<int> all_tokens = prompt_tokens;
    std::vector<int> gen_tokens;
    int cur_token = prompt_tokens.back();

    auto t_decode_start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < max_tokens; ++step) {
        int next_tok = engine.generate_step(cur_token, params, all_tokens);
        all_tokens.push_back(next_tok);
        gen_tokens.push_back(next_tok);

        std::string tok_str = tokenizer.decode({next_tok});
        if (tok_str == "<|endoftext|>" || tok_str == "<|eot_id|>" || tok_str == "</s>" || tok_str == "<eos>") {
            std::cout << "\n[EOS reached]" << std::endl;
            break;
        }

        std::cout << tok_str << std::flush;
        cur_token = next_tok;
    }
    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_time_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

    double decode_speed = (gen_tokens.size() / (decode_time_ms / 1000.0));

    std::cout << "\n\n╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║        Tenzo Standalone C++ Profiling Summary          ║" << std::endl;
    std::cout << "╠════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║ KV-Cache Architecture:   " << kv_mode << std::string(30 - kv_mode.length(), ' ') << "║" << std::endl;
    std::cout << "║ Prompt Tokens:           " << prompt_tokens.size() << std::string(30 - std::to_string(prompt_tokens.size()).length(), ' ') << "║" << std::endl;
    std::cout << "║ Generated Tokens:        " << gen_tokens.size() << std::string(30 - std::to_string(gen_tokens.size()).length(), ' ') << "║" << std::endl;
    std::cout << "║ Time To First Token:     " << ttft_ms << " ms" << std::string(27 - std::to_string((int)ttft_ms).length(), ' ') << "║" << std::endl;
    std::cout << "║ Total Decode Time:       " << decode_time_ms << " ms" << std::string(27 - std::to_string((int)decode_time_ms).length(), ' ') << "║" << std::endl;
    std::cout << "║ Decode Speed:            " << decode_speed << " tok/sec" << std::string(22 - std::to_string((int)decode_speed).length(), ' ') << "║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;

    return 0;
}
