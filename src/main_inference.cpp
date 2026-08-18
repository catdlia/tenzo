/**
 * @file main_inference.cpp
 * @brief Tenzo CLI — Full-Featured Native LLM Inference Engine (v0.3.0)
 * 
 * Provides production-grade command-line interface for 1.58-bit BitNet LLM inference:
 * - Single-shot generation & Streaming token decode
 * - Interactive multi-turn REPL chat mode with KV-Cache memory preservation
 * - Configurable Model & KV-Cache Quantization (INT8, Ternary TL1, FP32)
 * - Sampling hyperparameter controls (Temp, Top-P, Top-K, Repetition Penalty)
 * - Zero Python / PyTorch dependencies
 */

#include "tenzo.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <chrono>
#include <cstring>
#include <iomanip>

// ANSI Color Codes
#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_DIM     "\033[2m"
#define ANSI_RED     "\033[31m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_YELLOW  "\033[33m"
#define ANSI_BLUE    "\033[34m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_WHITE   "\033[37m"

class BpeTokenizer {
public:
    std::unordered_map<int, std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    int bos_token = 128000;
    int eos_token = 128001;
    int eot_token = 128009;

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
                // Handle GPT-2 / LLaMA space replacement (Ġ -> " ")
                size_t pos = 0;
                while ((pos = tok_str.find("\xc4\xa0", pos)) != std::string::npos) {
                    tok_str.replace(pos, 2, " ");
                    pos += 1;
                }
                // Handle newline replacement (Ċ -> "\n")
                pos = 0;
                while ((pos = tok_str.find("\xc4\x8a", pos)) != std::string::npos) {
                    tok_str.replace(pos, 2, "\n");
                    pos += 1;
                }
                id_to_token[tok_id] = tok_str;
                token_to_id[tok_str] = tok_id;
            }
        }
        if (token_to_id.count("<|begin_of_text|>")) bos_token = token_to_id["<|begin_of_text|>"];
        if (token_to_id.count("<|endoftext|>")) eos_token = token_to_id["<|endoftext|>"];
        if (token_to_id.count("<|eot_id|>")) eot_token = token_to_id["<|eot_id|>"];
        return true;
    }

    std::vector<int> encode(const std::string& text, bool add_bos = true) const {
        std::vector<int> tokens;
        if (add_bos) {
            tokens.push_back(bos_token);
        }

        size_t i = 0;
        while (i < text.length()) {
            bool matched = false;
            size_t max_len = std::min(static_cast<size_t>(64), text.length() - i);
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

    bool is_stop_token(int tid) const {
        if (tid >= 128000 && tid <= 128256) return true;
        if (tid == eos_token || tid == eot_token) return true;
        auto it = id_to_token.find(tid);
        if (it != id_to_token.end()) {
            const std::string& s = it->second;
            if (s == "<|end_of_text|>" || s == "<|endoftext|>" || s == "<|eot_id|>" || s == "</s>" || s == "<eos>") return true;
            if (s.find("endoftext") != std::string::npos || s.find("eot_id") != std::string::npos || 
                s.find("end_of_eot") != std::string::npos || s.find("end_of_text") != std::string::npos ||
                s.find("end_header_id") != std::string::npos) return true;
        }
        return false;
    }
};

struct CliOptions {
    std::string model_dir = "/app/tenzo-frontend/export_output";
    std::string prompt = "";
    std::string system_prompt = "You are Tenzo, a fast and helpful AI assistant powered by 1.58-bit native MLIR execution.";
    std::string kv_mode = "int8_fused";
    int max_tokens = 128;
    int ctx_size = 8192;
    float temp = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float rep_penalty = 1.15f;
    bool chat_mode = false;
    bool show_banner = true;
    bool benchmark = false;
};

void print_banner() {
    std::cout << ANSI_CYAN << ANSI_BOLD;
    std::cout << "  _____ _____ _   _ _____ ____  \n";
    std::cout << " |_   _| ____| \\ | |__  / __ \\ \n";
    std::cout << "   | | |  _| |  \\| | / / |  | |\n";
    std::cout << "   | | | |___| |\\  |/ /| |__| |\n";
    std::cout << "   |_| |_____|_| \\_/____\\____/ \n" << ANSI_RESET;
    std::cout << ANSI_BOLD << " ⚡ Tenzo Native LLM Inference Engine " << ANSI_GREEN << "v0.3.0" << ANSI_RESET << "\n";
    std::cout << ANSI_DIM << " High-performance MLIR/AVX2 runtime for 1.58-bit BitNet architectures\n" << ANSI_RESET;
    std::cout << "────────────────────────────────────────────────────────────────────────────\n" << std::endl;
}

void print_help(const char* prog_name) {
    print_banner();
    std::cout << ANSI_BOLD << "USAGE:" << ANSI_RESET << "\n";
    std::cout << "  " << prog_name << " [OPTIONS]\n\n";
    std::cout << ANSI_BOLD << "INFERENCE OPTIONS:" << ANSI_RESET << "\n";
    std::cout << "  -p, --prompt <str>          Prompt text for single-shot generation\n";
    std::cout << "  -n, --max-tokens <int>      Maximum number of tokens to generate (default: 128)\n";
    std::cout << "  -m, --model <path>          Path to exported model folder (default: /app/tenzo-frontend/export_output)\n";
    std::cout << "  --chat                      Enter interactive multi-turn REPL chat mode\n";
    std::cout << "  --system <str>              Custom system prompt (default: Tenzo Assistant)\n\n";
    std::cout << ANSI_BOLD << "QUANTIZATION & MEMORY:" << ANSI_RESET << "\n";
    std::cout << "  --kv-quant <mode>           KV-Cache Quantization mode (default: int8_fused)\n";
    std::cout << "                              Modes: " << ANSI_GREEN << "tl1_fused" << ANSI_RESET << " (14.2x comp, 84MB), " 
              << ANSI_GREEN << "int8_fused" << ANSI_RESET << " (4x comp, 309MB), " 
              << ANSI_GREEN << "fp32" << ANSI_RESET << " (1.2GB)\n";
    std::cout << "  -c, --ctx-size <int>        Maximum context sequence length (default: 8192)\n\n";
    std::cout << ANSI_BOLD << "SAMPLING HYPERPARAMETERS:" << ANSI_RESET << "\n";
    std::cout << "  -t, --temp <float>          Temperature (default: 0.7, 0.0 = Greedy ArgMax)\n";
    std::cout << "  --top-p <float>             Top-P / Nucleus sampling cutoff (default: 0.9)\n";
    std::cout << "  --top-k <int>               Top-K candidate filter (default: 40)\n";
    std::cout << "  --rep-penalty <float>       Repetition penalty multiplier (default: 1.15)\n\n";
    std::cout << ANSI_BOLD << "UTILITIES:" << ANSI_RESET << "\n";
    std::cout << "  -b, --benchmark             Run automated performance benchmark across sequence lengths\n";
    std::cout << "  -h, --help                  Show this help message and exit\n";
    std::cout << "  -v, --version               Show version information\n\n";
    std::cout << ANSI_BOLD << "EXAMPLES:" << ANSI_RESET << "\n";
    std::cout << "  # Interactive chat:\n";
    std::cout << "  " << prog_name << " --chat --kv-quant tl1_fused\n\n";
    std::cout << "  # Single prompt generation:\n";
    std::cout << "  " << prog_name << " -p \"Explain the importance of compilers\" -n 50 -t 0.7\n\n";
}

int main(int argc, char** argv) {
    CliOptions opt;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--version") {
            print_banner();
            return 0;
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            opt.prompt = argv[++i];
        } else if ((arg == "-n" || arg == "--max-tokens") && i + 1 < argc) {
            opt.max_tokens = std::atoi(argv[++i]);
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            opt.model_dir = argv[++i];
        } else if ((arg == "-t" || arg == "--temp" || arg == "--temperature") && i + 1 < argc) {
            opt.temp = std::atof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            opt.top_p = std::atof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            opt.top_k = std::atoi(argv[++i]);
        } else if ((arg == "--rep-penalty" || arg == "--repetition-penalty") && i + 1 < argc) {
            opt.rep_penalty = std::atof(argv[++i]);
        } else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) {
            opt.ctx_size = std::atoi(argv[++i]);
        } else if (arg == "--kv-quant" && i + 1 < argc) {
            opt.kv_mode = argv[++i];
        } else if (arg == "--system" && i + 1 < argc) {
            opt.system_prompt = argv[++i];
        } else if (arg == "--chat") {
            opt.chat_mode = true;
        } else if (arg == "-b" || arg == "--benchmark") {
            opt.benchmark = true;
        }
    }

    if (opt.show_banner) {
        print_banner();
    }

    // Load Tokenizer
    std::string vocab_path = opt.model_dir + "/tokenizer.vocab";
    std::string mlir_path = opt.model_dir + "/model.mlir";
    std::string weights_path = opt.model_dir + "/weights.bin";

    BpeTokenizer tokenizer;
    std::cout << ANSI_CYAN << "📖 Loading vocabulary..." << ANSI_RESET;
    if (!tokenizer.load_vocab(vocab_path)) {
        std::cout << ANSI_RED << " [FAILED]" << ANSI_RESET << std::endl;
        std::cerr << "❌ Could not open vocab file: " << vocab_path << std::endl;
        return 1;
    }
    std::cout << ANSI_GREEN << " [OK] " << ANSI_RESET << "(" << tokenizer.id_to_token.size() << " tokens)\n";

    // Initialize Tenzo Engine
    tenzo_config_t config = tenzo_default_config();
    config.kv_mode = opt.kv_mode.c_str();
    config.max_seq_len = opt.ctx_size;

    std::cout << ANSI_CYAN << "⚙️  Initializing Execution Engine (" 
              << ANSI_BOLD << "KV-Cache: " << opt.kv_mode << ANSI_RESET << ANSI_CYAN << ", Max Context: " 
              << opt.ctx_size << ")..." << ANSI_RESET;
    tenzo::Engine engine(config);
    std::cout << ANSI_GREEN << " [OK]\n" << ANSI_RESET;

    std::cout << ANSI_CYAN << "📦 Mapping weights & MLIR graph from " << opt.model_dir << "..." << ANSI_RESET;
    auto t_load_start = std::chrono::high_resolution_clock::now();
    try {
        engine.load_model(weights_path, mlir_path);
    } catch (const std::exception& e) {
        std::cout << ANSI_RED << " [FAILED]" << ANSI_RESET << std::endl;
        std::cerr << "❌ Error loading model: " << e.what() << std::endl;
        return 1;
    }
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    std::cout << ANSI_GREEN << " [OK] " << ANSI_RESET << "(" << std::fixed << std::setprecision(1) << load_ms << " ms)\n\n";

    tenzo_sampling_params_t params = tenzo_default_sampling_params();
    params.temperature = opt.temp;
    params.top_p = opt.top_p;
    params.top_k = opt.top_k;
    params.repetition_penalty = opt.rep_penalty;

    // --- REPL / Interactive Chat Mode ---
    if (opt.chat_mode) {
        std::cout << ANSI_BOLD << ANSI_MAGENTA << "💬 Tenzo Interactive Chat REPL" << ANSI_RESET << "\n";
        std::cout << ANSI_DIM << " Commands: /reset (clear history), /stats (view context), /exit (quit)\n" << ANSI_RESET;
        std::cout << "────────────────────────────────────────────────────────────────────────────\n";

        std::vector<int> conversation_tokens;
        // Apply System Prompt
        if (!opt.system_prompt.empty()) {
            std::string sys_formatted = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + opt.system_prompt + "<|eot_id|>";
            std::vector<int> sys_tokens = tokenizer.encode(sys_formatted, false);
            for (int t : sys_tokens) {
                engine.prefill_token(t);
                conversation_tokens.push_back(t);
            }
        }

        while (true) {
            std::cout << "\n" << ANSI_BOLD << ANSI_GREEN << "User > " << ANSI_RESET;
            std::string user_input;
            if (!std::getline(std::cin, user_input)) break;
            if (user_input.empty()) continue;

            if (user_input[0] == '/') {
                std::istringstream iss(user_input);
                std::string cmd;
                iss >> cmd;

                if (cmd == "/exit" || cmd == "/quit" || cmd == "/bye") {
                    std::cout << ANSI_DIM << "Exiting Tenzo REPL. Goodbye!\n" << ANSI_RESET;
                    break;
                } else if (cmd == "/help") {
                    std::cout << ANSI_BOLD << "\n📖 Interactive Chat Commands:\n" << ANSI_RESET;
                    std::cout << "  /help                       Show this help message\n";
                    std::cout << "  /stats                      Display active token count & KV-cache RAM\n";
                    std::cout << "  /reset, /clear              Clear conversation history & reset KV-Cache\n";
                    std::cout << "  /temp <float>               Set sampling temperature (e.g. /temp 0.7)\n";
                    std::cout << "  /top_p <float>              Set top-p nucleus sampling (e.g. /top_p 0.9)\n";
                    std::cout << "  /top_k <int>                Set top-k candidates (e.g. /top_k 40)\n";
                    std::cout << "  /rep <float>                Set repetition penalty (e.g. /rep 1.15)\n";
                    std::cout << "  /max_tokens <int>           Set max generation tokens per turn\n";
                    std::cout << "  /exit, /quit                Exit chat session\n\n";
                    continue;
                } else if (cmd == "/reset" || cmd == "/clear") {
                    engine.reset();
                    conversation_tokens.clear();
                    std::cout << ANSI_YELLOW << "🧹 Conversation context & KV-cache reset.\n" << ANSI_RESET;
                    if (!opt.system_prompt.empty()) {
                        std::string sys_formatted = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + opt.system_prompt + "<|eot_id|>";
                        std::vector<int> sys_tokens = tokenizer.encode(sys_formatted, false);
                        for (int t : sys_tokens) {
                            engine.prefill_token(t);
                            conversation_tokens.push_back(t);
                        }
                    }
                    continue;
                } else if (cmd == "/stats") {
                    int cur_len = engine.get_seq_len();
                    double kv_mb = (30.0 * cur_len * 5 * 128 * (opt.kv_mode == "tl1_fused" ? 0.25 : (opt.kv_mode == "int8_fused" ? 1.0 : 4.0)) * 2) / (1024.0 * 1024.0);
                    std::cout << ANSI_CYAN << "📊 Stats: Active Tokens: " << cur_len << " / " << opt.ctx_size 
                              << " | KV-Cache RAM: " << std::fixed << std::setprecision(2) << kv_mb << " MB"
                              << " | Temp: " << params.temperature << " | Top-P: " << params.top_p 
                              << " | Top-K: " << params.top_k << "\n" << ANSI_RESET;
                    continue;
                } else if (cmd == "/temp" || cmd == "/temperature") {
                    float v;
                    if (iss >> v) { params.temperature = v; std::cout << ANSI_GREEN << "✅ Temperature set to " << v << "\n" << ANSI_RESET; }
                    else std::cout << ANSI_RED << "Usage: /temp <float>\n" << ANSI_RESET;
                    continue;
                } else if (cmd == "/top_p" || cmd == "/topp") {
                    float v;
                    if (iss >> v) { params.top_p = v; std::cout << ANSI_GREEN << "✅ Top-P set to " << v << "\n" << ANSI_RESET; }
                    else std::cout << ANSI_RED << "Usage: /top_p <float>\n" << ANSI_RESET;
                    continue;
                } else if (cmd == "/top_k" || cmd == "/topk") {
                    int v;
                    if (iss >> v) { params.top_k = v; std::cout << ANSI_GREEN << "✅ Top-K set to " << v << "\n" << ANSI_RESET; }
                    else std::cout << ANSI_RED << "Usage: /top_k <int>\n" << ANSI_RESET;
                    continue;
                } else if (cmd == "/rep" || cmd == "/repetition_penalty") {
                    float v;
                    if (iss >> v) { params.repetition_penalty = v; std::cout << ANSI_GREEN << "✅ Repetition penalty set to " << v << "\n" << ANSI_RESET; }
                    else std::cout << ANSI_RED << "Usage: /rep <float>\n" << ANSI_RESET;
                    continue;
                } else if (cmd == "/max_tokens" || cmd == "/tokens") {
                    int v;
                    if (iss >> v) { opt.max_tokens = v; std::cout << ANSI_GREEN << "✅ Max tokens set to " << v << "\n" << ANSI_RESET; }
                    else std::cout << ANSI_RED << "Usage: /max_tokens <int>\n" << ANSI_RESET;
                    continue;
                } else {
                    std::cout << ANSI_RED << "Unknown command: " << cmd << ". Type /help for available commands.\n" << ANSI_RESET;
                    continue;
                }
            }

            std::string turn_formatted = "<|start_header_id|>user<|end_header_id|>\n\n" + user_input + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
            std::vector<int> turn_tokens = tokenizer.encode(turn_formatted, false);

            auto t_prefill_0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < turn_tokens.size() - 1; ++i) {
                engine.prefill_token(turn_tokens[i]);
                conversation_tokens.push_back(turn_tokens[i]);
            }
            auto t_prefill_1 = std::chrono::high_resolution_clock::now();
            double ttft_ms = std::chrono::duration<double, std::milli>(t_prefill_1 - t_prefill_0).count();

            std::cout << ANSI_BOLD << ANSI_CYAN << "Tenzo > " << ANSI_RESET << std::flush;

            int cur_token = turn_tokens.back();
            conversation_tokens.push_back(cur_token);

            auto t_dec_0 = std::chrono::high_resolution_clock::now();
            int gen_count = 0;

            for (int s = 0; s < opt.max_tokens; ++s) {
                int next_tok = engine.generate_step(cur_token, params, conversation_tokens);
                conversation_tokens.push_back(next_tok);
                gen_count++;

                if (tokenizer.is_stop_token(next_tok)) {
                    break;
                }

                std::cout << tokenizer.decode({next_tok}) << std::flush;
                cur_token = next_tok;
            }
            auto t_dec_1 = std::chrono::high_resolution_clock::now();
            double dec_ms = std::chrono::duration<double, std::milli>(t_dec_1 - t_dec_0).count();
            double tok_per_sec = (gen_count > 0 && dec_ms > 0) ? (gen_count / (dec_ms / 1000.0)) : 0.0;

            std::cout << "\n" << ANSI_DIM << "[" << gen_count << " tok | " 
                      << std::fixed << std::setprecision(1) << tok_per_sec << " tok/s | TTFT: " 
                      << (int)ttft_ms << "ms]\n" << ANSI_RESET;
        }
        return 0;
    }

    // --- Single-Shot Generation Mode ---
    std::string prompt_text = opt.prompt.empty() ? "In computer science, a compiler translates source code written in a high-level programming language into" : opt.prompt;

    std::vector<int> prompt_tokens = tokenizer.encode(prompt_text, true);

    std::cout << ANSI_BOLD << "💬 Prompt: " << ANSI_RESET << prompt_text << "\n";
    std::cout << ANSI_BOLD << ANSI_CYAN << "⚡ Output: " << ANSI_RESET << std::flush;

    auto t_prefill_0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < prompt_tokens.size() - 1; ++i) {
        engine.prefill_token(prompt_tokens[i]);
    }
    auto t_prefill_1 = std::chrono::high_resolution_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(t_prefill_1 - t_prefill_0).count();

    std::vector<int> all_tokens = prompt_tokens;
    std::vector<int> gen_tokens;
    int cur_token = prompt_tokens.back();

    auto t_dec_0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < opt.max_tokens; ++step) {
        int next_tok = engine.generate_step(cur_token, params, all_tokens);
        all_tokens.push_back(next_tok);
        gen_tokens.push_back(next_tok);

        if (tokenizer.is_stop_token(next_tok)) {
            break;
        }

        std::cout << tokenizer.decode({next_tok}) << std::flush;
        cur_token = next_tok;
    }
    auto t_dec_1 = std::chrono::high_resolution_clock::now();
    double dec_ms = std::chrono::duration<double, std::milli>(t_dec_1 - t_dec_0).count();
    double tok_per_sec = (gen_tokens.size() > 0 && dec_ms > 0) ? (gen_tokens.size() / (dec_ms / 1000.0)) : 0.0;
    double bytes_per_weight = (opt.kv_mode == "tl1_fused" || opt.kv_mode == "popcount_fused") ? 0.25 : (opt.kv_mode == "int8_fused" ? 1.0 : 4.0);
    double kv_mem_mb = (30.0 * 8192 * 5 * 128 * bytes_per_weight * 2) / (1024.0 * 1024.0);

    std::cout << "\n\n" << ANSI_BOLD << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║             Tenzo Engine Inference Profile             ║\n";
    std::cout << "╠════════════════════════════════════════════════════════╣\n" << ANSI_RESET;
    std::cout << "║ Model Architecture:      BitNet-b1.58-2B-4T (30 Layers)║\n";
    std::cout << "║ KV-Cache Quantization:   " << std::left << std::setw(30) << (opt.kv_mode + " (" + std::to_string((int)kv_mem_mb) + " MB for 8K)") << "║\n";
    std::cout << "║ Prompt Context Length:   " << std::left << std::setw(30) << (std::to_string(prompt_tokens.size()) + " tokens") << "║\n";
    std::cout << "║ Generated Tokens:        " << std::left << std::setw(30) << (std::to_string(gen_tokens.size()) + " tokens") << "║\n";
    std::cout << "║ Time To First Token:     " << std::left << std::setw(30) << (std::to_string((int)ttft_ms) + " ms") << "║\n";
    std::cout << "║ Total Decode Time:       " << std::left << std::setw(30) << (std::to_string((int)dec_ms) + " ms") << "║\n";
    std::cout << "║ " << ANSI_BOLD << ANSI_GREEN << "Decode Speed:            " << std::fixed << std::setprecision(2) << std::left << std::setw(23) << (std::to_string(tok_per_sec).substr(0, 5) + " tok/sec") << ANSI_RESET << "║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    return 0;
}
