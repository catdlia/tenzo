/**
 * @file main_inference.cpp
 * @brief Tenzo CLI — Full-Featured Native LLM Inference Engine (v1.1.0)
 * 
 * Provides production-grade command-line interface for 1.58-bit BitNet LLM inference:
 * - Single-shot generation & Streaming token decode
 * - Interactive multi-turn REPL chat mode with KV-Cache memory preservation
 * - Configurable Model & KV-Cache Quantization (INT8, Ternary TL1, FP32)
 * - Sampling hyperparameter controls (Temp, Top-P, Top-K, Repetition Penalty)
 * - Zero Python / PyTorch dependencies
 */

#include "tenzo.hpp"
#include "runtime/MicroarchProfiler.h"
#include "runtime/Tokenizer.h"
#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>

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

// ---------------------------------------------------------------------------
// Base64 helpers for --tokenize-batch parity testing (preserves binary/newlines)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
static std::vector<std::vector<int>> read_id_lines(const std::string& path) {
    std::vector<std::vector<int>> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        std::vector<int> ids;
        int v;
        while (iss >> v) ids.push_back(v);
        if (!ids.empty()) out.push_back(std::move(ids));
    }
    return out;
}

static void read_rss_kb(long& rss_kb, long& hwm_kb) {
    rss_kb = hwm_kb = -1;
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) rss_kb = std::atol(line.c_str() + 6);
        else if (line.rfind("VmHWM:", 0) == 0) hwm_kb = std::atol(line.c_str() + 6);
    }
}

static int argmax_f(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[best]) best = (int)i;
    return best;
}

// log P(target) under softmax(logits), accumulated in double precision.
static double log_prob_of(const std::vector<float>& logits, int target) {
    float mx = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (float l : logits) sum += std::exp((double)l - (double)mx);
    return ((double)logits[target] - (double)mx) - std::log(sum);
}

static bool valid_kv_mode(const std::string& m) {
    return m == "fp32" || m == "int8_fused" || m == "tl1_fused" || m == "popcount_fused" || m == "paged_int8" || m == "ternary_rot";
}

static std::string sha256_file_hex(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return "unreadable";
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };
    auto rotr = [](uint32_t x, uint32_t n) -> uint32_t { return (x >> n) | (x << (32u - n)); };

    const uint64_t bit_len = static_cast<uint64_t>(data.size()) * 8ULL;
    data.push_back(0x80u);
    while ((data.size() % 64u) != 56u) data.push_back(0u);
    for (int s = 56; s >= 0; s -= 8) {
        data.push_back(static_cast<uint8_t>((bit_len >> s) & 0xffu));
    }

    uint32_t H[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    for (size_t off = 0; off < data.size(); off += 64u) {
        uint32_t w[64];
        for (int t = 0; t < 16; ++t) {
            const size_t idx = off + static_cast<size_t>(t) * 4u;
            w[t] = (static_cast<uint32_t>(data[idx]) << 24) | (static_cast<uint32_t>(data[idx + 1]) << 16) |
                   (static_cast<uint32_t>(data[idx + 2]) << 8) | static_cast<uint32_t>(data[idx + 3]);
        }
        for (int t = 16; t < 64; ++t) {
            uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
            uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16] + s0 + w[t - 7] + s1;
        }
        uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
        for (int t = 0; t < 64; ++t) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t t1 = h + S1 + ch + K[t] + w[t];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        H[0] += a; H[1] += b; H[2] += c; H[3] += d;
        H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }

    static const char HEX[] = "0123456789abcdef";
    std::string out(64, '0');
    for (int i = 0; i < 8; ++i) {
        for (int b = 0; b < 8; ++b) {
            out[static_cast<size_t>(i) * 8u + static_cast<size_t>(b)] = HEX[(H[i] >> ((7 - b) * 4)) & 0xfu];
        }
    }
    return out;
}

static std::string base64_decode(const std::string& in) {
    static const int T[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
    };
    std::string out;
    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (c == '\r' || c == '\n' || c == ' ') continue;
        int d = T[c];
        if (d == -1) break;
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

static std::string base64_encode(const std::string& in) {
    static const char* B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    int val = 0, valb = -6;
    for (unsigned char c : in) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back(B64[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) out.push_back(B64[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}

struct CliOptions {
    std::string model_dir = "/app/tenzo-frontend/export_output";
    std::string tokenizer_path = "";
    std::string prompt = "";
    std::string system_prompt = "You are Tenzo, a fast and helpful AI assistant powered by 1.58-bit native MLIR execution.";
    std::string kv_mode = "int8_fused";
    std::string kv_alpha_path = "";
    std::string device = "cpu";
    int max_tokens = 128;
    int ctx_size = 8192;
    float temp = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float rep_penalty = 1.15f;
    bool chat_mode = false;
    bool show_banner = true;
    bool benchmark = false;
    bool profile = false;
    // --- evaluation / measurement modes (token-id driven) ---
    bool tokenize_only = false;
    bool tokenize_batch = false;
    std::string ids_file;        // whitespace-separated token ids, one sequence per line
    std::string dump_prefix;     // writes <prefix>.logits.f32 and <prefix>.greedy.txt
    int greedy_steps = 32;
    std::string score_file;      // windows of token ids, one per line (teacher forcing)
    std::string score_out;
    std::string prefill_mode = "sequential";   // --prefill sequential|batched
    int prefill_chunk = 32;                    // --prefill-chunk
    int bench_pp = 0;            // prompt tokens for the fair e2e benchmark
    int bench_tg = 0;            // generated tokens for the fair e2e benchmark
};

void print_banner() {
    std::cout << ANSI_CYAN << ANSI_BOLD;
    std::cout << "  _____ _____ _   _ _____ ____  \n";
    std::cout << " |_   _| ____| \\ | |__  / __ \\ \n";
    std::cout << "   | | |  _| |  \\| | / / |  | |\n";
    std::cout << "   | | | |___| |\\  |/ /| |__| |\n";
    std::cout << "   |_| |_____|_| \\_/____\\____/ \n" << ANSI_RESET;
    std::cout << ANSI_BOLD << " ⚡ Tenzo Native LLM Inference Engine " << ANSI_GREEN << "v1.1.0-beta (Release)" << ANSI_RESET << "\n";
    std::cout << ANSI_DIM << " Heterogeneous MLIR runtime: CPU (AVX2/NEON/RVV), GPU (Vulkan/CUDA/ROCm)\n" << ANSI_RESET;
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
    std::cout << ANSI_BOLD << "HETEROGENEOUS COMPUTE & BACKENDS:" << ANSI_RESET << "\n";
    std::cout << "  -d, --device <backend>      Compute backend: " << ANSI_GREEN << "cpu" << ANSI_RESET << " (AVX2/NEON), "
              << ANSI_GREEN << "gpu / vulkan" << ANSI_RESET << " (Vulkan SPIR-V), "
              << ANSI_GREEN << "cuda" << ANSI_RESET << " (NVPTX/Vulkan), "
              << ANSI_GREEN << "rocm" << ANSI_RESET << " (AMDGCN/Vulkan), "
              << ANSI_GREEN << "riscv" << ANSI_RESET << " (RVV 1.0)\n";
    std::cout << "  --kv-quant <mode>           KV-Cache Quantization mode (default: int8_fused)\n";
    std::cout << "  --kv-alpha <path|auto>      Per-layer ternary_rot clip multiplier file or 'auto' (<model_dir>/kv_alpha_mix.txt)\n";
    std::cout << "  --prefill <sequential|batched>  prompt prefill: token by token (default) or in chunks (CPU + I2_S)\n";
    std::cout << "  --prefill-chunk <n>         chunk size for --prefill batched (default 32)\n";
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
    std::cout << "  --profile                   Print hardware microarchitecture topology & cache analysis\n";
    std::cout << "  -b, --benchmark             Run automated performance benchmark across sequence lengths\n";
    std::cout << "  -h, --help                  Show this help message and exit\n";
    std::cout << "  -v, --version               Show version information\n\n";
    std::cout << ANSI_BOLD << "EXAMPLES:" << ANSI_RESET << "\n";
    std::cout << "  # Interactive chat on Vulkan GPU:\n";
    std::cout << "  " << prog_name << " --chat --device vulkan\n\n";
    std::cout << "  # CUDA translation mode:\n";
    std::cout << "  " << prog_name << " -p \"Hello CUDA\" --device cuda\n\n";
    std::cout << "  # RISC-V RVV vector emulation:\n";
    std::cout << "  " << prog_name << " -p \"Hello RISC-V\" --device riscv\n\n";
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
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            opt.device = argv[++i];
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
        } else if (arg == "--prefill" && i + 1 < argc) {
            opt.prefill_mode = argv[++i];
            if (opt.prefill_mode != "sequential" && opt.prefill_mode != "batched") {
                std::cerr << "error: --prefill must be sequential or batched\n"; return 2;
            }
        } else if (arg == "--prefill-chunk" && i + 1 < argc) {
            opt.prefill_chunk = std::atoi(argv[++i]);
            if (opt.prefill_chunk < 1) { std::cerr << "error: --prefill-chunk must be >= 1\n"; return 2; }
        } else if (arg == "--kv-quant" && i + 1 < argc) {
            opt.kv_mode = argv[++i];
        } else if (arg == "--kv-alpha" && i + 1 < argc) {
            opt.kv_alpha_path = argv[++i];
        } else if (arg == "--system" && i + 1 < argc) {
            opt.system_prompt = argv[++i];
        } else if (arg == "--chat") {
            opt.chat_mode = true;
        } else if (arg == "-b" || arg == "--benchmark") {
            opt.benchmark = true;
        } else if (arg == "--profile") {
            opt.profile = true;
        } else if (arg == "--tokenize-only") {
            opt.tokenize_only = true;
        } else if (arg == "--tokenize-batch") {
            opt.tokenize_batch = true;
            opt.show_banner = false;
        } else if (arg == "--tokenizer" && i + 1 < argc) {
            opt.tokenizer_path = argv[++i];
        } else if (arg == "--ids-file" && i + 1 < argc) {
            opt.ids_file = argv[++i];
        } else if (arg == "--dump-prefix" && i + 1 < argc) {
            opt.dump_prefix = argv[++i];
        } else if (arg == "--greedy-steps" && i + 1 < argc) {
            opt.greedy_steps = std::atoi(argv[++i]);
        } else if (arg == "--score" && i + 1 < argc) {
            opt.score_file = argv[++i];
        } else if (arg == "--score-out" && i + 1 < argc) {
            opt.score_out = argv[++i];
        } else if (arg == "--bench-pp" && i + 1 < argc) {
            opt.bench_pp = std::atoi(argv[++i]);
        } else if (arg == "--bench-tg" && i + 1 < argc) {
            opt.bench_tg = std::atoi(argv[++i]);
        }
    }

    if (!valid_kv_mode(opt.kv_mode)) {
        std::cerr << "error: unknown --kv-quant mode '" << opt.kv_mode
                  << "' (valid: fp32, int8_fused, tl1_fused, popcount_fused, paged_int8, ternary_rot). "
                     "Previously unknown modes silently ran the fp32 path.\n";
        return 2;
    }

    if (opt.show_banner) {
        print_banner();
    }

    if (opt.profile) {
        auto& prof = tenzo::MicroarchProfiler::getProfile();
        tenzo::MicroarchProfiler::printReport(prof);
        if (argc == 2) {
            return 0;
        }
    }

    // Load Tokenizer
    std::string vocab_path = opt.model_dir + "/tokenizer.vocab";
    std::string mlir_path = opt.model_dir + "/model.mlir";
    std::string weights_path = opt.model_dir + "/weights.bin";

    // Auto-fallback search if model path is not directly accessible
    if (opt.tokenizer_path.empty() && !std::ifstream(vocab_path).is_open() &&
        !std::ifstream(opt.model_dir + "/tokenizer.json").is_open()) {
        std::vector<std::string> candidates = {
            "tenzo-frontend/export_output",
            "./tenzo-frontend/export_output",
            "/app/tenzo-frontend/export_output"
        };
        for (const auto& c : candidates) {
            std::string test_v = c + "/tokenizer.vocab";
            std::string test_j = c + "/tokenizer.json";
            if (std::ifstream(test_v).is_open() || std::ifstream(test_j).is_open()) {
                opt.model_dir = c;
                vocab_path = c + "/tokenizer.vocab";
                mlir_path = c + "/model.mlir";
                weights_path = c + "/weights.bin";
                break;
            }
        }
    }

    const std::string tok_load_path = opt.tokenizer_path.empty() ? vocab_path : opt.tokenizer_path;
    tenzo::runtime::Tokenizer tokenizer;
    if (!opt.tokenize_batch) {
        std::cout << ANSI_CYAN << "📖 Loading vocabulary..." << ANSI_RESET;
    }
    if (!tokenizer.load(tok_load_path)) {
        if (!opt.tokenize_batch) {
            std::cout << ANSI_RED << " [FAILED]" << ANSI_RESET << std::endl;
        }
        std::cerr << "❌ Could not load tokenizer from: " << tok_load_path << std::endl;
        return 1;
    }
    if (!opt.tokenize_batch) {
        std::cout << ANSI_GREEN << " [OK] " << ANSI_RESET << "(" << tokenizer.vocab_size() << " tokens"
                  << (tokenizer.uses_bpe() ? ", BPE" : ", legacy") << ")\n";
    }

    if (opt.tokenize_batch) {
        std::string b64_line;
        while (std::getline(std::cin, b64_line)) {
            if (!b64_line.empty() && b64_line.back() == '\r') b64_line.pop_back();
            if (b64_line == ".") break;
            std::string raw = base64_decode(b64_line);
            auto ids_bos = tokenizer.encode(raw, true);
            auto ids_raw = tokenizer.encode(raw, false);
            std::string rt = tokenizer.decode(ids_raw);
            std::cout << "IDS_BOS:";
            for (int32_t t : ids_bos) std::cout << " " << t;
            std::cout << "\tIDS_RAW:";
            for (int32_t t : ids_raw) std::cout << " " << t;
            std::cout << "\tRT:" << base64_encode(rt) << "\n";
        }
        return 0;
    }

    if (opt.tokenize_only) {
        // Machine-readable: "TOKENS: id id id ..." (BOS included, as in single-shot generation)
        auto ids = tokenizer.encode(opt.prompt, true);
        std::cout << "TOKENS:";
        for (int32_t t : ids) std::cout << " " << t;
        std::cout << "\n";
        return 0;
    }

    std::string kv_alpha_req = opt.kv_alpha_path;
    if (kv_alpha_req.empty()) {
        if (const char* env_alpha = std::getenv("TENZO_KV_ALPHA")) {
            kv_alpha_req = env_alpha;
        }
    }
    if (kv_alpha_req == "auto") {
        const std::string sep = (!opt.model_dir.empty() && opt.model_dir.back() == '/') ? "" : "/";
        const std::string candidate = opt.model_dir + sep + "kv_alpha_mix.txt";
        std::ifstream kf(candidate);
        if (kf.good()) {
            setenv("TENZO_KV_ALPHA", candidate.c_str(), 1);
        } else {
            std::cerr << "[warn] --kv-alpha auto: " << candidate << " not found; using default scale\n";
            unsetenv("TENZO_KV_ALPHA");
        }
    } else if (!opt.kv_alpha_path.empty()) {
        setenv("TENZO_KV_ALPHA", opt.kv_alpha_path.c_str(), 1);
    }

    // Initialize Tenzo Engine
    tenzo_config_t config = tenzo_default_config();
    config.kv_mode = opt.kv_mode.c_str();
    config.max_seq_len = opt.ctx_size;
    config.device = opt.device.c_str();
    {
        // rope_theta comes from the model's config.json when present (HF field name). Exports made before
        // 2026-11-19 lack it; then the API default (BitNet-b1.58-2B-4T value) is used and this is said loudly.
        std::ifstream cf(opt.model_dir + "/config.json");
        std::stringstream cs; cs << cf.rdbuf();
        const std::string ctext = cs.str();
        const size_t kp = ctext.find("\"rope_theta\"");
        if (kp != std::string::npos) {
            const size_t colon = ctext.find(':', kp);
            config.rope_theta = std::strtof(ctext.c_str() + colon + 1, nullptr);
            if (!(config.rope_theta > 0.0f)) {
                std::cerr << "error: invalid rope_theta in " << opt.model_dir << "/config.json\n";
                return 2;
            }
        } else {
            std::cerr << "warning: " << opt.model_dir << "/config.json has no rope_theta; using default "
                      << config.rope_theta << " (microsoft/bitnet-b1.58-2B-4T)\n";
        }
    }

    std::cout << ANSI_CYAN << "⚙️  Initializing Execution Engine (" 
              << ANSI_BOLD << "Backend: " << opt.device 
              << ", KV-Cache: " << opt.kv_mode << ANSI_RESET << ANSI_CYAN << ", Max Context: " 
              << opt.ctx_size << ")..." << ANSI_RESET;
    std::unique_ptr<tenzo::Engine> engine_ptr;
    try {
        engine_ptr = std::make_unique<tenzo::Engine>(config);
    } catch (const std::exception& e) {
        std::cout << ANSI_RED << " [FAILED]" << ANSI_RESET << std::endl;
        std::cerr << "❌ Engine initialization failed: " << e.what() << std::endl;
        return 1;
    }
    tenzo::Engine& engine = *engine_ptr;
    std::cout << ANSI_GREEN << " [OK]\n" << ANSI_RESET;

    // Prompt prefill: --prefill batched processes the prompt in chunks (CPU + I2_S + non-paged KV), bit-identical to token by token.
    auto prefill_prompt = [&](const int* toks, int n) {
        if (n <= 0) return;
        if (opt.prefill_mode == "batched") {
            if (engine.prefill_tokens(toks, n, opt.prefill_chunk)) return;
            static bool warned = false;
            if (!warned) {
                std::cerr << "warning: --prefill batched is unsupported for this configuration (needs CPU, TENZO_GEMV=i2s, non-paged KV); using sequential\n";
                warned = true;
            }
        }
        for (int i = 0; i < n; ++i) engine.prefill_token(toks[i]);
    };

    // TENZO_SMAPS_DIR=<dir>: copy /proc/self/smaps_rollup and /proc/self/smaps to <dir>/<tag>.{rollup,smaps} (RSS breakdown by mapping)
    auto dump_smaps = [&](const char* tag) {
        const char* dir = std::getenv("TENZO_SMAPS_DIR");
        if (!dir || !*dir) return;
        for (const char* what : {"smaps_rollup", "smaps"}) {
            std::ifstream in(std::string("/proc/self/") + what);
            std::ofstream out(std::string(dir) + "/" + tag + (std::string(what) == "smaps" ? ".smaps" : ".rollup"));
            out << in.rdbuf();
        }
    };
    dump_smaps("after_engine_create");

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

    dump_smaps("after_load");
    tenzo_sampling_params_t params = tenzo_default_sampling_params();
    params.temperature = opt.temp;
    params.top_p = opt.top_p;
    params.top_k = opt.top_k;
    params.repetition_penalty = opt.rep_penalty;

    // ======================= Evaluation / measurement modes =======================
    if (!opt.dump_prefix.empty() || !opt.score_file.empty() || opt.bench_pp > 0) {
        const int V = config.vocab_size;
        std::vector<float> logits(V);

        if (!opt.dump_prefix.empty()) {
            auto seqs = read_id_lines(opt.ids_file);
            if (seqs.empty()) { std::cerr << "error: --dump-prefix needs --ids-file with one sequence\n"; return 2; }
            const std::vector<int>& ids = seqs[0];
            if ((int)ids.size() + opt.greedy_steps >= opt.ctx_size) { std::cerr << "error: ids + greedy_steps exceed --ctx-size\n"; return 2; }
            engine.reset();
            prefill_prompt(ids.data(), static_cast<int>(ids.size()) - 1);   // --prefill sequential|batched
            engine.forward_logits(ids.back(), logits);                      // last prompt token: logits of the next-token distribution
            {
                std::ofstream lf(opt.dump_prefix + ".logits.f32", std::ios::binary);
                lf.write(reinterpret_cast<const char*>(logits.data()), sizeof(float) * V);
            }
            std::ofstream gf(opt.dump_prefix + ".greedy.txt");
            for (int k = 0; k < opt.greedy_steps; ++k) {
                int nxt = argmax_f(logits);
                gf << nxt << (k + 1 < opt.greedy_steps ? " " : "\n");
                if (k + 1 < opt.greedy_steps) engine.forward_logits(nxt, logits);
            }
            std::cout << "DUMP_OK prefix=" << opt.dump_prefix << " prompt_tokens=" << ids.size() << " kv=" << opt.kv_mode << "\n";
        }

        if (!opt.score_file.empty()) {
            auto windows = read_id_lines(opt.score_file);
            if (windows.empty() || opt.score_out.empty()) { std::cerr << "error: --score needs windows and --score-out\n"; return 2; }
            std::ofstream of(opt.score_out);
            of << std::setprecision(9);
            of << "# kv_mode=" << opt.kv_mode << " ctx=" << opt.ctx_size << " windows=" << windows.size() << "\n";
            if (const char* alpha_env = std::getenv("TENZO_KV_ALPHA")) {
                if (*alpha_env) {
                    of << "# kv_alpha=" << alpha_env << " sha256=" << sha256_file_hex(alpha_env) << "\n";
                }
            }
            double tot_nll = 0.0; long tot_n = 0;
            for (size_t w = 0; w < windows.size(); ++w) {
                const auto& ids = windows[w];
                if ((int)ids.size() > opt.ctx_size) { std::cerr << "error: window longer than --ctx-size\n"; return 2; }
                engine.reset();
                std::vector<double> lps;
                lps.reserve(ids.size());
                for (size_t i = 0; i + 1 < ids.size(); ++i) {
                    engine.forward_logits(ids[i], logits);
                    lps.push_back(log_prob_of(logits, ids[i + 1]));
                }
                double nll = 0.0;
                for (double lp : lps) nll -= lp;
                tot_nll += nll; tot_n += (long)lps.size();
                of << "w " << w << " " << lps.size() << " " << nll << "\n";
                of << "lp";
                for (double lp : lps) of << " " << lp;
                of << "\n";
                of.flush();
                std::cout << "score window " << w << " tokens=" << lps.size() << " ppl=" << std::exp(nll / lps.size())
                          << " running_ppl=" << std::exp(tot_nll / tot_n) << std::endl;
            }
        }

        if (opt.bench_pp > 0) {
            auto seqs = read_id_lines(opt.ids_file);
            if (seqs.empty() || (int)seqs[0].size() < opt.bench_pp) { std::cerr << "error: --ids-file must hold >= --bench-pp ids\n"; return 2; }
            if (opt.bench_pp + opt.bench_tg >= opt.ctx_size) { std::cerr << "error: pp+tg exceed --ctx-size\n"; return 2; }
            const std::vector<int>& ids = seqs[0];
            using clk = std::chrono::steady_clock;
            auto run_once = [&](int pp, int tg, double& pp_ms, double& tg_ms) {
                engine.reset();
                auto t0 = clk::now();
                prefill_prompt(ids.data(), pp - 1);
                engine.forward_logits(ids[pp - 1], logits);          // last prompt token: logits (LM head) computed once
                auto t1 = clk::now();
                int cur = argmax_f(logits);
                tenzo_sampling_params_t gp = tenzo_default_sampling_params();
                gp.temperature = 0.0f; gp.repetition_penalty = 1.0f;
                std::vector<int> none;
                for (int s2 = 0; s2 < tg; ++s2) cur = engine.generate_step(cur, gp, none);
                auto t2 = clk::now();
                pp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                tg_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            };
            double a, b;
            run_once(std::min(16, opt.bench_pp), 4, a, b);           // warm-up (not reported)
            double pp_ms = 0, tg_ms = 0;
            run_once(opt.bench_pp, opt.bench_tg, pp_ms, tg_ms);
            dump_smaps("after_bench");
            long rss, hwm; read_rss_kb(rss, hwm);
            std::cout << "BENCH_FAIR,kv=" << opt.kv_mode << ",pp_tokens=" << opt.bench_pp << ",pp_ms=" << pp_ms
                      << ",tg_tokens=" << opt.bench_tg << ",tg_ms=" << tg_ms
                      << ",pp_tps=" << (pp_ms > 0 ? opt.bench_pp / (pp_ms / 1000.0) : 0.0)
                      << ",tg_tps=" << (tg_ms > 0 ? opt.bench_tg / (tg_ms / 1000.0) : 0.0)
                      << ",rss_kb=" << rss << ",hwm_kb=" << hwm << "\n";
        }
        long rss, hwm; read_rss_kb(rss, hwm);
        std::cout << "PROC_RSS_KB=" << rss << " PROC_HWM_KB=" << hwm << "\n";
        return 0;
    }

    // --- REPL / Interactive Chat Mode ---
    if (opt.chat_mode) {
        std::cout << ANSI_BOLD << ANSI_MAGENTA << "💬 Tenzo Interactive Chat REPL" << ANSI_RESET << "\n";
        std::cout << ANSI_DIM << " Commands: /reset (clear history), /stats (view context), /exit (quit)\n" << ANSI_RESET;
        std::cout << "────────────────────────────────────────────────────────────────────────────\n";

        std::vector<int> conversation_tokens;
        // Apply System Prompt
        if (!opt.system_prompt.empty()) {
            std::string sys_formatted = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + opt.system_prompt + "<|eot_id|>";
            auto sys_tokens = tokenizer.encode(sys_formatted, false);
            for (int32_t t : sys_tokens) {
                engine.prefill_token(static_cast<int>(t));
                conversation_tokens.push_back(static_cast<int>(t));
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
                        auto sys_tokens = tokenizer.encode(sys_formatted, false);
                        for (int32_t t : sys_tokens) {
                            engine.prefill_token(static_cast<int>(t));
                            conversation_tokens.push_back(static_cast<int>(t));
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
            auto turn_tokens = tokenizer.encode(turn_formatted, false);

            auto t_prefill_0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i + 1 < turn_tokens.size(); ++i) {
                engine.prefill_token(static_cast<int>(turn_tokens[i]));
                conversation_tokens.push_back(static_cast<int>(turn_tokens[i]));
            }
            auto t_prefill_1 = std::chrono::high_resolution_clock::now();
            double ttft_ms = std::chrono::duration<double, std::milli>(t_prefill_1 - t_prefill_0).count();

            std::cout << ANSI_BOLD << ANSI_CYAN << "Tenzo > " << ANSI_RESET << std::flush;

            int cur_token = static_cast<int>(turn_tokens.back());
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

    auto ptoks32 = tokenizer.encode(prompt_text, true);
    std::vector<int> prompt_tokens(ptoks32.begin(), ptoks32.end());

    std::cout << ANSI_BOLD << "💬 Prompt: " << ANSI_RESET << prompt_text << "\n";
    std::cout << ANSI_BOLD << ANSI_CYAN << "⚡ Output: " << ANSI_RESET << std::flush;

    auto t_prefill_0 = std::chrono::high_resolution_clock::now();
    prefill_prompt(prompt_tokens.data(), static_cast<int>(prompt_tokens.size()) - 1);
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
    // ternary_rot: 96 B per kv head and token (K and V: 32 B codes + 4 fp32 scales each) vs 128 * bytes_per_weight * 2
    double bytes_per_weight = (opt.kv_mode == "tl1_fused" || opt.kv_mode == "popcount_fused") ? 0.25 : (opt.kv_mode == "int8_fused" ? 1.0 : (opt.kv_mode == "ternary_rot" ? 0.375 : 4.0));
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
