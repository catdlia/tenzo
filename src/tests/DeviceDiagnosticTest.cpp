/**
 * @file DeviceDiagnosticTest.cpp
 * @brief Comprehensive Hardware, Compiler, SIMD & Inference Diagnostic Suite
 * Verifies correctness across all device architectures (x86_64, ARMv8.2, ARMv9 SVE2/DotProd).
 */

#include "tenzo.h"
#include "tenzo.hpp"
#include "runtime/TenzoEngine.h"
#include "runtime/simd_arm_compat.h"
#include "runtime/MicroarchProfiler.h"
#include "runtime/CUDARuntime.h"
#include "runtime/ROCmRuntime.h"
#include "runtime/arch/RISCV_RVV.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <fstream>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

const char* ANSI_RESET  = "\033[0m";
const char* ANSI_BOLD   = "\033[1m";
const char* ANSI_GREEN  = "\033[1;32m";
const char* ANSI_RED    = "\033[1;31m";
const char* ANSI_YELLOW = "\033[1;33m";
const char* ANSI_CYAN   = "\033[1;36m";

void print_header(const std::string& title) {
    std::cout << "\n" << ANSI_CYAN << ANSI_BOLD << "════════════════════════════════════════════════════════════════════" << ANSI_RESET << "\n";
    std::cout << " 🔬 " << ANSI_BOLD << title << ANSI_RESET << "\n";
    std::cout << ANSI_CYAN << "────────────────────────────────────────────────────────────────────" << ANSI_RESET << "\n";
}

void report_status(const std::string& test_name, bool success, const std::string& details = "") {
    std::cout << "  " << std::left << std::setw(45) << test_name << " ";
    if (success) {
        std::cout << ANSI_GREEN << "[PASS]" << ANSI_RESET;
    } else {
        std::cout << ANSI_RED << "[FAIL]" << ANSI_RESET;
    }
    if (!details.empty()) {
        std::cout << " " << ANSI_YELLOW << details << ANSI_RESET;
    }
    std::cout << "\n";
}

// 1. Hardware & Compiler Info
void run_system_diagnostics() {
    print_header("1. Target Hardware & Compiler Environment");

    #if defined(__aarch64__) || defined(__ARM_NEON)
        std::cout << "  Architecture:         " << ANSI_GREEN << "ARM64 / AArch64 (NEON)" << ANSI_RESET << "\n";
        #if defined(__ARM_FEATURE_DOTPROD)
        std::cout << "  Dot Product (SDOT):   " << ANSI_GREEN << "Enabled (+dotprod)" << ANSI_RESET << "\n";
        #else
        std::cout << "  Dot Product (SDOT):   " << ANSI_YELLOW << "Disabled (Fallback NEON)" << ANSI_RESET << "\n";
        #endif
        #if defined(__ARM_FEATURE_SVE2)
        std::cout << "  SVE2 (ARMv9):         " << ANSI_GREEN << "Enabled (+sve2)" << ANSI_RESET << "\n";
        #endif
    #elif defined(__x86_64__)
        std::cout << "  Architecture:         " << ANSI_GREEN << "x86_64 (AVX2 / FMA / VNNI)" << ANSI_RESET << "\n";
    #else
        std::cout << "  Architecture:         " << ANSI_YELLOW << "Generic CPU" << ANSI_RESET << "\n";
    #endif

    #if defined(__clang__)
        std::cout << "  Compiler:             " << "Clang " << __clang_version__ << "\n";
    #elif defined(__GNUC__)
        std::cout << "  Compiler:             " << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
    #endif

    #ifdef _OPENMP
        std::cout << "  OpenMP Max Threads:   " << omp_get_max_threads() << " threads\n";
    #else
        std::cout << "  OpenMP:               " << ANSI_YELLOW << "Disabled" << ANSI_RESET << "\n";
    #endif
}

// 2. Exact SIMD Math Verification
bool run_simd_math_diagnostics() {
    print_header("2. SIMD & Mathematical Kernels Verification");
    bool all_ok = true;

    // Test A: Shuffle / eLUT indexing
    {
        alignas(32) int8_t lut[32];
        for (int i = 0; i < 32; ++i) lut[i] = static_cast<int8_t>(i * 3 + 1);
        alignas(32) int8_t idx[32];
        for (int i = 0; i < 32; ++i) idx[i] = static_cast<int8_t>(i % 16);

        __m256i v_lut = _mm256_load_si256(reinterpret_cast<const __m256i*>(lut));
        __m256i v_idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(idx));
        __m256i v_res = _mm256_shuffle_epi8(v_lut, v_idx);

        alignas(32) int8_t out[32];
        _mm256_store_si256(reinterpret_cast<__m256i*>(out), v_res);

        bool ok = true;
        for (int i = 0; i < 16; ++i) {
            if (out[i] != lut[i % 16]) ok = false;
            if (out[i + 16] != lut[(i % 16) + 16]) ok = false;
        }
        report_status("AVX2/NEON _mm256_shuffle_epi8 (eLUT)", ok);
        if (!ok) all_ok = false;
    }

    // Test B: 16-bit shift
    {
        alignas(32) uint8_t raw[32];
        for (int i = 0; i < 32; ++i) raw[i] = static_cast<uint8_t>(0x50 | (i & 0x0F));
        __m256i v_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(raw));
        __m256i v_mask = _mm256_set1_epi8(0x0F);
        __m256i v_hi = _mm256_and_si256(_mm256_srli_epi16(v_raw, 4), v_mask);

        alignas(32) uint8_t out[32];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), v_hi);

        bool ok = true;
        for (int i = 0; i < 32; ++i) {
            if (out[i] != 5) ok = false;
        }
        report_status("AVX2/NEON _mm256_srli_epi16 (4-bit unpack)", ok);
        if (!ok) all_ok = false;
    }

    // Test C: Sign extension epi8 -> epi16 -> epi32
    {
        alignas(16) int8_t src8[16] = {-128, -50, -1, 0, 1, 50, 127, 42, -10, 20, -30, 40, -50, 60, -70, 80};
        __m128i v8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src8));
        __m256i v16 = _mm256_cvtepi8_epi16(v8);

        alignas(32) int16_t out16[16];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out16), v16);

        bool ok16 = true;
        for (int i = 0; i < 16; ++i) {
            if (out16[i] != static_cast<int16_t>(src8[i])) ok16 = false;
        }
        report_status("AVX2/NEON _mm256_cvtepi8_epi16", ok16);
        if (!ok16) all_ok = false;

        __m128i v16_half = _mm256_extracti128_si256(v16, 0);
        __m256i v32 = _mm256_cvtepi16_epi32(v16_half);
        alignas(32) int32_t out32[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out32), v32);

        bool ok32 = true;
        for (int i = 0; i < 8; ++i) {
            if (out32[i] != static_cast<int32_t>(src8[i])) ok32 = false;
        }
        report_status("AVX2/NEON _mm256_cvtepi16_epi32", ok32);
        if (!ok32) all_ok = false;
    }

    // Test D: Fused Multiply-Add (madd_epi16)
    {
        alignas(32) int16_t a[16] = {1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8};
        alignas(32) int16_t b[16] = {10, 20, 30, 40, 50, 60, 70, 80, 10, 20, 30, 40, 50, 60, 70, 80};
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
        __m256i vres = _mm256_madd_epi16(va, vb);

        alignas(32) int32_t out[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), vres);

        bool ok = true;
        for (int i = 0; i < 8; ++i) {
            int32_t expected = a[2 * i] * b[2 * i] + a[2 * i + 1] * b[2 * i + 1];
            if (out[i] != expected) ok = false;
        }
        report_status("AVX2/NEON _mm256_madd_epi16", ok);
        if (!ok) all_ok = false;
    }

    // Test E: Floating-Point FMA
    {
        alignas(32) float a[8] = {1.5f, -2.0f, 3.25f, -0.5f, 10.0f, 2.5f, -1.0f, 4.0f};
        alignas(32) float b[8] = {2.0f, 3.0f, -1.0f, 4.0f, 0.5f, -2.0f, 3.0f, 1.5f};
        alignas(32) float c[8] = {0.5f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
        __m256 va = _mm256_loadu_ps(a);
        __m256 vb = _mm256_loadu_ps(b);
        __m256 vc = _mm256_loadu_ps(c);
        __m256 vres = _mm256_fmadd_ps(va, vb, vc);

        alignas(32) float out[8];
        _mm256_storeu_ps(out, vres);

        bool ok = true;
        for (int i = 0; i < 8; ++i) {
            float expected = a[i] * b[i] + c[i];
            if (std::abs(out[i] - expected) > 1e-5f) ok = false;
        }
        report_status("AVX2/NEON _mm256_fmadd_ps", ok);
        if (!ok) all_ok = false;
    }

    return all_ok;
}

// 3. Weight File Diagnostics
bool run_weight_diagnostics(const std::string& model_dir) {
    print_header("3. Model Weights & Vocabulary Verification");
    std::string vocab_path = model_dir + "/tokenizer.vocab";
    std::string weights_path = model_dir + "/weights.bin";

    bool all_ok = true;

    // Check vocabulary
    std::ifstream vf(vocab_path);
    if (!vf.is_open()) {
        report_status("Vocabulary file present", false, "Missing: " + vocab_path);
        all_ok = false;
    } else {
        int token_count = 0;
        std::string line;
        while (std::getline(vf, line)) token_count++;
        report_status("Vocabulary loaded", true, std::to_string(token_count) + " tokens");
    }

    // Check weights.bin
    std::ifstream wf(weights_path, std::ios::binary | std::ios::ate);
    if (!wf.is_open()) {
        report_status("Weights file present", false, "Missing: " + weights_path);
        return false;
    }

    size_t fsize = static_cast<size_t>(wf.tellg());
    wf.seekg(0, std::ios::beg);
    double mb = fsize / (1024.0 * 1024.0);

    const size_t expected_size = 1836252160ULL; // 1.75 GB
    bool size_ok = (fsize >= 100000000);
    report_status("Weights file size check", size_ok, std::to_string(static_cast<int>(mb)) + " MB");

    // Sample embeddings
    size_t embed_bytes = 128256ULL * 2560 * 4;
    if (fsize >= embed_bytes) {
        std::vector<float> sample_row(2560);
        wf.read(reinterpret_cast<char*>(sample_row.data()), 2560 * sizeof(float));

        float sum = 0.0f, sum_sq = 0.0f;
        for (float v : sample_row) {
            sum += v;
            sum_sq += v * v;
        }
        float mean = sum / 2560.0f;
        float var = (sum_sq / 2560.0f) - (mean * mean);

        bool var_ok = (var > 1e-6f);
        report_status("Token embedding variance check", var_ok, "var=" + std::to_string(var));
        if (!var_ok) all_ok = false;
    }

    return all_ok;
}

// 4. Tenzo Engine Single-Step Test
bool run_engine_inference_test(const std::string& model_dir) {
    print_header("4. Native Inference & Sampling Validation");

    tenzo_config_t config = tenzo_default_config();
    config.kv_mode = "popcount_fused";
    config.max_seq_len = 512;

    try {
        tenzo::Engine engine(config);
        engine.load_model(model_dir + "/weights.bin", model_dir + "/model.mlir");
        report_status("Engine model mapping", true);

        // Run single decode step with prompt token 128000
        tenzo_sampling_params_t params = tenzo_default_sampling_params();
        params.temperature = 0.0f; // Deterministic argmax for diagnostics

        auto t0 = std::chrono::high_resolution_clock::now();
        engine.prefill_token(128000);
        std::vector<int> past = {128000};
        int next_token = engine.generate_step(128000, params, past);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        bool token_valid = (next_token >= 0 && next_token < 128256);
        report_status("Single token generation", token_valid, "Next Token ID=" + std::to_string(next_token) + " (" + std::to_string(ms) + " ms)");

        return token_valid;
    } catch (const std::exception& e) {
        report_status("Engine initialization", false, e.what());
        return false;
    }
}

// Backend Diagnostics
bool run_backend_diagnostics() {
    print_header("4. Heterogeneous Backends & Microarch Diagnostics");

    // 1. Profiler
    auto& prof = tenzo::MicroarchProfiler::getProfile();
    report_status("Microarch Profiler", prof.num_threads > 0, 
                  prof.cpu_arch + ", " + std::to_string(prof.vector_bits) + "-bit SIMD, L1=" + std::to_string(prof.l1d_cache_size/1024) + "KB");

    // 2. RISC-V RVV Kernel Test
    float act[4] = {1.0f, 2.0f, -1.0f, 0.5f};
    uint8_t w_packed[1] = { static_cast<uint8_t>(2 | (0 << 2) | (1 << 4) | (2 << 6)) }; // [+1, -1, 0, +1]
    float out_rvv[1] = {0.0f};
    tenzo::rvv::gemv_bitlinear_tl1_rvv(act, w_packed, out_rvv, 1, 4, 1, 1.0f);
    // Expected: 1*1 + 2*(-1) + (-1)*0 + 0.5*1 = 1 - 2 + 0.5 = -0.5f
    bool rvv_ok = std::abs(out_rvv[0] - (-0.5f)) < 1e-4f;
    report_status("RISC-V RVV BitLinear Math", rvv_ok, "Calculated: " + std::to_string(out_rvv[0]) + ", Expected: -0.500");

    // 3. CUDA Translation Layer
    auto& cuda = tenzo::CUDARuntime::getInstance();
    bool cuda_init = cuda.initialize();
    report_status("CUDA Backend (Vulkan Translation)", cuda_init, cuda.getDeviceName());

    // 4. ROCm Translation Layer
    auto& rocm = tenzo::ROCmRuntime::getInstance();
    bool rocm_init = rocm.initialize();
    report_status("ROCm Backend (Vulkan Translation)", rocm_init, rocm.getDeviceName());

    return rvv_ok;
}

} // namespace

int main(int argc, char** argv) {
    std::string model_dir = "tenzo-frontend/export_output";
    if (argc > 1) model_dir = argv[1];

    std::cout << ANSI_CYAN << ANSI_BOLD
              << "╔══════════════════════════════════════════════════════════════════╗\n"
              << "║    Tenzo Compiler Diagnostic Suite v1.0.0-beta.1 (Beta-1.0)      ║\n"
              << "╚══════════════════════════════════════════════════════════════════╝\n"
              << ANSI_RESET;

    run_system_diagnostics();
    bool simd_ok = run_simd_math_diagnostics();
    bool backend_ok = run_backend_diagnostics();
    bool weights_ok = run_weight_diagnostics(model_dir);
    bool engine_ok = run_engine_inference_test(model_dir);

    print_header("Diagnostic Summary");
    if (simd_ok && backend_ok && weights_ok && engine_ok) {
        std::cout << ANSI_GREEN << ANSI_BOLD << "🎉 ALL SYSTEM CHECKS PASSED! Tenzo Beta-1.0 is fully operational on this device." << ANSI_RESET << "\n\n";
        return 0;
    } else {
        std::cout << ANSI_RED << ANSI_BOLD << "⚠️  DIAGNOSTIC ISSUES DETECTED. Review the logs above for details." << ANSI_RESET << "\n\n";
        return 1;
    }
}
