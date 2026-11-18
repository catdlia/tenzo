// ==============================================================================
// Native C++ Synthetic Attention Micro-Benchmark
// File: src/tests/PaperMicroBenchmark.cpp
//
// Comparing:
//   1. BaselineAttention: FP32 Attention + Explicit Per-Head Copy Ablation
//   2. TenzoPackedAttention: 2-bit Packed Ternary Attention + Direct GQA Indexing
//
// Metrics Measured:
//   - Latency (ms) & Throughput (invocations/sec)
//   - Memory Footprint (Allocated bytes & Active working set in MiB)
//   - Analytical Footprint Divided By Time (not physical bandwidth) (GB/s (footprint/time; not measured traffic))
//
// Cross-platform support:
//   - x86_64: AVX2 + FMA intrinsics
//   - ARM64: NEON FP32 / scalar packed fallback
// ==============================================================================

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <stdexcept>
#include <sstream>

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #define TENZO_ARCH_X86 1
#elif defined(__aarch64__) || defined(__arm__)
  #include <arm_neon.h>
  #define TENZO_ARCH_ARM 1
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

// ==============================================================================
// 1. Benchmark Configuration & Constants
// ==============================================================================
struct BenchmarkConfig {
    int64_t num_q_heads = 32;       // H_q (Query heads)
    int64_t num_kv_heads = 8;       // H_kv (Key/Value heads, GQA Ratio = 4)
    int64_t head_dim = 128;         // D (Head dimension)
    int warmup_iters = 5;           // Warm-up runs before timing
    int timed_iters = 25;           // Measurement iterations
    bool run_baseline = true;
    bool run_tenzo = true;
    std::string csv_file = "";
};

struct BenchmarkResult {
    int64_t context_length;
    double baseline_latency_ms;
    double tenzo_latency_ms;
    double speedup;
    double baseline_tokens_per_sec;
    double tenzo_tokens_per_sec;
    double baseline_mem_mb;
    double tenzo_mem_mb;
    double mem_reduction_ratio;
    double baseline_bw_gbps;
    double tenzo_bw_gbps;
};

// ==============================================================================
// 2. Packing & Encoding Utilities (1.58-bit Ternary {-1, 0, 1})
// ==============================================================================
// Encodes two ternary values (w1, w2) into a 4-bit nibble: nibble = (w2+1)*4 + (w1+1)
// 4 ternary values packed per 8-bit byte.
static inline uint8_t pack_ternary_byte(int v0, int v1, int v2, int v3) {
    uint8_t lo = static_cast<uint8_t>(((v1 + 1) * 4) + (v0 + 1));
    uint8_t hi = static_cast<uint8_t>(((v3 + 1) * 4) + (v2 + 1));
    return static_cast<uint8_t>((lo & 0x0F) | ((hi & 0x0F) << 4));
}

// Decode 4-bit nibble into pair of {-1, 0, 1}
static inline void decode_ternary_nibble(uint8_t nibble, int &w1, int &w2) {
    static const int lut[4] = {-1, 0, 1, 0};
    w1 = lut[nibble & 3];
    w2 = lut[(nibble >> 2) & 3];
}

static inline void unpack_ternary_byte(uint8_t byte, int &v0, int &v1, int &v2, int &v3) {
    decode_ternary_nibble(byte & 0x0F, v0, v1);
    decode_ternary_nibble((byte >> 4) & 0x0F, v2, v3);
}

// Numerically stable softmax
static void compute_softmax(float *scores, int64_t T) {
    float max_s = scores[0];
    for (int64_t t = 1; t < T; ++t) {
        if (scores[t] > max_s) max_s = scores[t];
    }
    float sum_exp = 0.0f;
    for (int64_t t = 0; t < T; ++t) {
        scores[t] = std::exp(scores[t] - max_s);
        sum_exp += scores[t];
    }
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int64_t t = 0; t < T; ++t) {
        scores[t] *= inv_sum;
    }
}

// ==============================================================================
// 3. Baseline Implementation: Classical FP32 Attention + GQA Physical Copy
// ==============================================================================
void run_baseline_attention(
    const float *queries,           // [H_q, D]
    const float *key_cache,         // [H_kv, T, D]
    const float *val_cache,         // [H_kv, T, D]
    float *output,                  // [H_q, D]
    int64_t H_q, int64_t H_kv, int64_t T, int64_t D,
    float scale
) {
    if (H_q <= 0 || H_kv <= 0 || H_q % H_kv || T <= 0 || D <= 0 || D % 32)
        throw std::invalid_argument("attention requires positive sizes, Hq divisible by Hkv, D divisible by 32");
    int64_t gqa_ratio = H_q / H_kv;

    // Allocate intermediate copy buffers simulating physical GQA memory copy / reshape / transpose
    // In naive pipelines (vLLM unoptimized / PyTorch eager), KV heads are broadcast-copied to Q heads:
    std::vector<float> k_replicated(T * D);
    std::vector<float> v_replicated(T * D);
    std::vector<float> scores(T);

    for (int64_t h_q = 0; h_q < H_q; ++h_q) {
        int64_t h_kv = h_q / gqa_ratio;

        // 1. Memory Wall Bottleneck: Physical memory copy of KV head data into replicated buffer
        const float *k_src = key_cache + h_kv * T * D;
        const float *v_src = val_cache + h_kv * T * D;
        std::memcpy(k_replicated.data(), k_src, T * D * sizeof(float));
        std::memcpy(v_replicated.data(), v_src, T * D * sizeof(float));

        const float *q_ptr = queries + h_q * D;

        // 2. Q * K^T Score computation with FP32 FMADD
        for (int64_t t = 0; t < T; ++t) {
            const float *k_ptr = k_replicated.data() + t * D;
            float dot = 0.0f;

#if defined(TENZO_ARCH_X86)
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (int64_t d = 0; d < D; d += 16) {
                __m256 q0 = _mm256_loadu_ps(q_ptr + d);
                __m256 k0 = _mm256_loadu_ps(k_ptr + d);
                acc0 = _mm256_fmadd_ps(q0, k0, acc0);

                __m256 q1 = _mm256_loadu_ps(q_ptr + d + 8);
                __m256 k1 = _mm256_loadu_ps(k_ptr + d + 8);
                acc1 = _mm256_fmadd_ps(q1, k1, acc1);
            }
            __m256 sum256 = _mm256_add_ps(acc0, acc1);
            alignas(32) float buf[8];
            _mm256_store_ps(buf, sum256);
            dot = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
#elif defined(TENZO_ARCH_ARM)
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            for (int64_t d = 0; d < D; d += 8) {
                float32x4_t q0 = vld1q_f32(q_ptr + d);
                float32x4_t k0 = vld1q_f32(k_ptr + d);
                acc0 = vfmaq_f32(acc0, q0, k0);

                float32x4_t q1 = vld1q_f32(q_ptr + d + 4);
                float32x4_t k1 = vld1q_f32(k_ptr + d + 4);
                acc1 = vfmaq_f32(acc1, q1, k1);
            }
            dot = vaddvq_f32(vaddq_f32(acc0, acc1));
#else
            for (int64_t d = 0; d < D; ++d) {
                dot += q_ptr[d] * k_ptr[d];
            }
#endif
            scores[t] = dot * scale;
        }

        // 3. Softmax
        compute_softmax(scores.data(), T);

        // 4. Value accumulation: Out = sum_t (Score[t] * V[t]) with FP32 FMADD
        float *out_head = output + h_q * D;
        for (int64_t d = 0; d < D; ++d) out_head[d] = 0.0f;

        for (int64_t t = 0; t < T; ++t) {
            float w = scores[t];
            if (std::abs(w) < 1e-9f) continue;
            const float *v_ptr = v_replicated.data() + t * D;

#if defined(TENZO_ARCH_X86)
            __m256 w_vec = _mm256_set1_ps(w);
            for (int64_t d = 0; d < D; d += 8) {
                __m256 o = _mm256_loadu_ps(out_head + d);
                __m256 v = _mm256_loadu_ps(v_ptr + d);
                o = _mm256_fmadd_ps(w_vec, v, o);
                _mm256_storeu_ps(out_head + d, o);
            }
#elif defined(TENZO_ARCH_ARM)
            float32x4_t w_vec = vdupq_n_f32(w);
            for (int64_t d = 0; d < D; d += 4) {
                float32x4_t o = vld1q_f32(out_head + d);
                float32x4_t v = vld1q_f32(v_ptr + d);
                o = vfmaq_f32(o, w_vec, v);
                vst1q_f32(out_head + d, o);
            }
#else
            for (int64_t d = 0; d < D; ++d) {
                out_head[d] += w * v_ptr[d];
            }
#endif
        }
    }
}

// ==============================================================================
// 4. Tenzo Implementation: Fused 1.58-bit Bitwise Attention + Implicit Shuffle
// ==============================================================================
// Eliminates memory wall:
//   - Zero-Copy GQA: Virtual routing via h_q / gqa_ratio directly to packed KV cache
//   - Bit-Packed Cache: 4 ternary values {-1, 0, 1} per uint8_t byte (16x smaller)
//   - Bitwise Dot-Product: Shifts, masks, and integer multiply-accumulate (Zero FP32 FMA)
void run_tenzo_packed_attention(
    const float *queries,           // [H_q, D]
    const uint8_t *packed_key_cache,// [H_kv, T, D / 4]
    const uint8_t *packed_val_cache,// [H_kv, T, D / 4]
    float *output,                  // [H_q, D]
    int64_t H_q, int64_t H_kv, int64_t T, int64_t D,
    float scale
) {
    if (H_q <= 0 || H_kv <= 0 || H_q % H_kv || T <= 0 || D <= 0 || D % 32)
        throw std::invalid_argument("attention requires positive sizes, Hq divisible by Hkv, D divisible by 32");
    int64_t gqa_ratio = H_q / H_kv;
    int64_t bytes_per_token = D / 4;
    std::vector<float> scores(T);

#if defined(TENZO_ARCH_X86)
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i lut0 = _mm_setr_epi8(-1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0);
    const __m128i lut1 = _mm_setr_epi8(-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0);
#endif

    // Quantize query heads to int16 (one-time preparation per token)
    std::vector<int16_t> q_i16(H_q * D);
    std::vector<float> q_eff_scales(H_q);

    for (int64_t h_q = 0; h_q < H_q; ++h_q) {
        const float *q_ptr = queries + h_q * D;
        float max_q = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            float a = std::abs(q_ptr[d]);
            if (a > max_q) max_q = a;
        }
        float s_q = (max_q > 0.0f) ? (max_q / 32767.0f) : 1.0f;
        float inv_s_q = 1.0f / s_q;
        for (int64_t d = 0; d < D; ++d) {
            q_i16[h_q * D + d] = static_cast<int16_t>(std::round(q_ptr[d] * inv_s_q));
        }
        q_eff_scales[h_q] = s_q * scale;
    }

    for (int64_t h_q = 0; h_q < H_q; ++h_q) {
        // Zero-Copy Implicit Shuffle: Virtual routing directly into packed KV memory
        // No memory copy, no reshape, no transpose buffer!
        int64_t h_kv = h_q / gqa_ratio;
        const uint8_t *k_head = packed_key_cache + h_kv * T * bytes_per_token;
        const uint8_t *v_head = packed_val_cache + h_kv * T * bytes_per_token;

        const int16_t *q_head = q_i16.data() + h_q * D;
        float eff_scale = q_eff_scales[h_q];

        // 1. Bitwise Dot-Product Q * K^T
        for (int64_t t = 0; t < T; ++t) {
            const uint8_t *k_token = k_head + t * bytes_per_token;

#if defined(TENZO_ARCH_X86)
            __m256i dot_acc = _mm256_setzero_si256();
            for (int64_t d = 0; d < D; d += 32) {
                __m256i q_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_head + d));
                __m256i q_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_head + d + 16));

                __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(k_token + (d / 4)));
                __m128i nib_lo = _mm_and_si128(raw, mask_0f);
                __m128i nib_hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);

                __m128i w0 = _mm_shuffle_epi8(lut0, nib_lo);
                __m128i w1 = _mm_shuffle_epi8(lut1, nib_lo);
                __m128i w2 = _mm_shuffle_epi8(lut0, nib_hi);
                __m128i w3 = _mm_shuffle_epi8(lut1, nib_hi);

                __m128i w01 = _mm_unpacklo_epi8(w0, w1);
                __m128i w23 = _mm_unpacklo_epi8(w2, w3);
                __m128i w_0_15 = _mm_unpacklo_epi16(w01, w23);
                __m128i w_16_31 = _mm_unpackhi_epi16(w01, w23);

                __m256i k_lo = _mm256_cvtepi8_epi16(w_0_15);
                __m256i k_hi = _mm256_cvtepi8_epi16(w_16_31);

                __m256i p0 = _mm256_madd_epi16(q_lo, k_lo);
                __m256i p1 = _mm256_madd_epi16(q_hi, k_hi);
                dot_acc = _mm256_add_epi32(dot_acc, _mm256_add_epi32(p0, p1));
            }

            alignas(32) int32_t buf[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(buf), dot_acc);
            int32_t total_dot = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
            scores[t] = static_cast<float>(total_dot) * eff_scale;
#else
            int32_t total_dot = 0;
            static const int8_t nibble_w0[16] = {-1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0};
            static const int8_t nibble_w1[16] = {-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0};

            for (int64_t b = 0; b < bytes_per_token; ++b) {
                uint8_t byte = k_token[b];
                uint8_t lo = byte & 0x0F;
                uint8_t hi = byte >> 4;
                int64_t d = b * 4;
                total_dot += q_head[d + 0] * nibble_w0[lo] +
                             q_head[d + 1] * nibble_w1[lo] +
                             q_head[d + 2] * nibble_w0[hi] +
                             q_head[d + 3] * nibble_w1[hi];
            }
            scores[t] = static_cast<float>(total_dot) * eff_scale;
#endif
        }

        // 2. Softmax
        compute_softmax(scores.data(), T);

        // 3. Value accumulation with Branchless Ternary Addition/Subtraction
        float *out_head = output + h_q * D;
        for (int64_t d = 0; d < D; ++d) out_head[d] = 0.0f;

        for (int64_t t = 0; t < T; ++t) {
            float w = scores[t];
            if (std::abs(w) < 1e-9f) continue;
            const uint8_t *v_token = v_head + t * bytes_per_token;
            float delta[4] = {-w, 0.0f, w, 0.0f};

            for (int64_t b = 0; b < bytes_per_token; ++b) {
                uint8_t byte = v_token[b];
                uint8_t lo = byte & 0x0F;
                uint8_t hi = byte >> 4;
                int64_t d = b * 4;
                out_head[d + 0] += delta[lo & 3];
                out_head[d + 1] += delta[(lo >> 2) & 3];
                out_head[d + 2] += delta[hi & 3];
                out_head[d + 3] += delta[(hi >> 2) & 3];
            }
        }
    }
}

// ==============================================================================
// 5. Benchmark Suite Execution & Reporting
// ==============================================================================
void run_context_benchmark(int64_t T, const BenchmarkConfig &cfg, std::vector<BenchmarkResult> &results) {
    int64_t H_q = cfg.num_q_heads;
    int64_t H_kv = cfg.num_kv_heads;
    int64_t D = cfg.head_dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::cout << "\n══════════════════════════════════════════════════════════════════════\n";
    std::cout << "  📊 Benchmarking Context Length T = " << T << " tokens (H_q=" << H_q 
              << ", H_kv=" << H_kv << ", D=" << D << ")\n";
    std::cout << "══════════════════════════════════════════════════════════════════════\n";

    std::cout << "Native C++ synthetic attention; analytical buffer sizes are NOT process RSS.\n";
    // Memory Footprint calculation:
    // Baseline: Key cache [H_kv, T, D] + Val cache [H_kv, T, D] in FP32 (4 bytes)
    //           Plus intermediate GQA replicated working buffer [H_q, T, D] in FP32
    double base_kv_bytes = 2.0 * H_kv * T * D * sizeof(float);
    double base_working_bytes = base_kv_bytes + (2.0 * T * D * sizeof(float)); // KV copy buffer
    double base_mem_mb = base_working_bytes / (1024.0 * 1024.0);

    // Tenzo: Key cache [H_kv, T, D/4] + Val cache [H_kv, T, D/4] in uint8_t (1 byte)
    //        Zero intermediate GQA buffer (Implicit Shuffle)
    double tenzo_kv_bytes = 2.0 * H_kv * T * (D / 4) * sizeof(uint8_t);
    double tenzo_working_bytes = tenzo_kv_bytes; // Zero copy buffers
    double tenzo_mem_mb = tenzo_working_bytes / (1024.0 * 1024.0);

    double mem_reduction = base_working_bytes / tenzo_working_bytes;

    std::cout << "  [Memory Footprint]\n";
    std::cout << "    • Baseline Allocated (KV + Copy Buffer): " << std::fixed << std::setprecision(2) 
              << base_mem_mb << " MiB\n";
    std::cout << "    • Tenzo Allocated (Packed KV, Zero Copy): " << std::fixed << std::setprecision(2) 
              << tenzo_mem_mb << " MiB (" << std::setprecision(1) << mem_reduction << "x reduction)\n";

    // Allocate and populate test vectors
    std::vector<float> queries(H_q * D);
    std::vector<float> base_out(H_q * D, 0.0f);
    std::vector<float> tenzo_out(H_q * D, 0.0f);

    std::vector<float> key_cache_fp32(H_kv * T * D);
    std::vector<float> val_cache_fp32(H_kv * T * D);
    std::vector<uint8_t> key_cache_packed(H_kv * T * (D / 4));
    std::vector<uint8_t> val_cache_packed(H_kv * T * (D / 4));

    // Populate with deterministic ternary values
    for (int64_t i = 0; i < H_q * D; ++i) {
        queries[i] = static_cast<float>((i % 5) - 2) * 0.25f;
    }
    for (int64_t h = 0; h < H_kv; ++h) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t b = 0; b < D / 4; ++b) {
                int v0 = ((t + b + 0) % 3) - 1;
                int v1 = ((t + b + 1) % 3) - 1;
                int v2 = ((t + b + 2) % 3) - 1;
                int v3 = ((t + b + 3) % 3) - 1;

                int64_t d = b * 4;
                int64_t base_idx = h * T * D + t * D + d;
                key_cache_fp32[base_idx + 0] = static_cast<float>(v0);
                key_cache_fp32[base_idx + 1] = static_cast<float>(v1);
                key_cache_fp32[base_idx + 2] = static_cast<float>(v2);
                key_cache_fp32[base_idx + 3] = static_cast<float>(v3);

                val_cache_fp32[base_idx + 0] = static_cast<float>(v0);
                val_cache_fp32[base_idx + 1] = static_cast<float>(v1);
                val_cache_fp32[base_idx + 2] = static_cast<float>(v2);
                val_cache_fp32[base_idx + 3] = static_cast<float>(v3);

                int64_t packed_idx = h * T * (D / 4) + t * (D / 4) + b;
                key_cache_packed[packed_idx] = pack_ternary_byte(v0, v1, v2, v3);
                val_cache_packed[packed_idx] = pack_ternary_byte(v0, v1, v2, v3);
            }
        }
    }

    double base_latency_ms = 0.0;
    if (cfg.run_baseline) {
        // Warm-up
        for (int i = 0; i < cfg.warmup_iters; ++i) {
            run_baseline_attention(queries.data(), key_cache_fp32.data(), val_cache_fp32.data(),
                                  base_out.data(), H_q, H_kv, T, D, scale);
        }
        // Timed runs
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < cfg.timed_iters; ++i) {
            run_baseline_attention(queries.data(), key_cache_fp32.data(), val_cache_fp32.data(),
                                  base_out.data(), H_q, H_kv, T, D, scale);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        base_latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / cfg.timed_iters;
    }

    double tenzo_latency_ms = 0.0;
    if (cfg.run_tenzo) {
        // Warm-up
        for (int i = 0; i < cfg.warmup_iters; ++i) {
            run_tenzo_packed_attention(queries.data(), key_cache_packed.data(), val_cache_packed.data(),
                                      tenzo_out.data(), H_q, H_kv, T, D, scale);
        }
        // Timed runs
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < cfg.timed_iters; ++i) {
            run_tenzo_packed_attention(queries.data(), key_cache_packed.data(), val_cache_packed.data(),
                                      tenzo_out.data(), H_q, H_kv, T, D, scale);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        tenzo_latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / cfg.timed_iters;
    }

    double speedup = (tenzo_latency_ms > 0.0) ? (base_latency_ms / tenzo_latency_ms) : 0.0;
    double base_tokens_sec = (base_latency_ms > 0.0) ? (1000.0 / base_latency_ms) : 0.0;
    double tenzo_tokens_sec = (tenzo_latency_ms > 0.0) ? (1000.0 / tenzo_latency_ms) : 0.0;

    // Effective I/O Bandwidth calculation: total bytes touched / latency
    double base_bw_gbps = (base_working_bytes / (base_latency_ms * 1e-3)) / 1e9;
    double tenzo_bw_gbps = (tenzo_working_bytes / (tenzo_latency_ms * 1e-3)) / 1e9;

    std::cout << "  [Performance & Throughput]\n";
    if (cfg.run_baseline) {
        std::cout << "    • Baseline FP32 Attention Latency : " << std::fixed << std::setprecision(3) 
                  << base_latency_ms << " ms  (" << std::setprecision(1) << base_tokens_sec << " invocations/sec, "
                  << std::setprecision(2) << base_bw_gbps << " GB/s (footprint/time; not measured traffic))\n";
    }
    if (cfg.run_tenzo) {
        std::cout << "    • Tenzo Packed Attention Latency  : " << std::fixed << std::setprecision(3) 
                  << tenzo_latency_ms << " ms  (" << std::setprecision(1) << tenzo_tokens_sec << " invocations/sec, "
                  << std::setprecision(2) << tenzo_bw_gbps << " GB/s (footprint/time; not measured traffic))\n";
    }
    if (cfg.run_baseline && cfg.run_tenzo) {
        std::cout << "    ⚡ Speedup: " << std::setprecision(2) << speedup << "x faster\n";
    }

    results.push_back({
        T, base_latency_ms, tenzo_latency_ms, speedup,
        base_tokens_sec, tenzo_tokens_sec,
        base_mem_mb, tenzo_mem_mb, mem_reduction,
        base_bw_gbps, tenzo_bw_gbps
    });
}

// Print formatted summary table suitable for the paper
void print_paper_summary_table(const std::vector<BenchmarkResult> &results) {
    std::cout << "\n\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                     🔬 TENZO VS. BASELINE PAPER EVALUATION SUMMARY TABLE                          ║\n";
    std::cout << "╠═════════════╦═══════════════════════╦═══════════════════════╦══════════╦══════════════════════════╣\n";
    std::cout << "║ Context (T) ║ Baseline (FP32+Copy)  ║ Tenzo (Packed+Zero-Cp)║ Speedup  ║ Memory Reduction         ║\n";
    std::cout << "║             ║ Latency  | Memory     ║ Latency  | Memory     ║          ║ Working Set Ratio        ║\n";
    std::cout << "╠═════════════╬══════════╪════════════╬══════════╪════════════╬══════════╬══════════════════════════╣\n";

    for (const auto &r : results) {
        std::cout << "║ " << std::setw(11) << r.context_length << " ║ "
                  << std::setw(6) << std::fixed << std::setprecision(2) << r.baseline_latency_ms << "ms | "
                  << std::setw(7) << std::setprecision(1) << r.baseline_mem_mb << "MB ║ "
                  << std::setw(6) << std::setprecision(2) << r.tenzo_latency_ms << "ms | "
                  << std::setw(7) << std::setprecision(1) << r.tenzo_mem_mb << "MB ║ "
                  << std::setw(7) << std::setprecision(2) << r.speedup << "x ║ "
                  << std::setw(15) << std::setprecision(1) << r.mem_reduction_ratio << "x smaller     ║\n";
    }
    std::cout << "╚═════════════╩═══════════════════════╩═══════════════════════╩══════════╩══════════════════════════╝\n";
}

void export_csv(const std::string &path, const std::vector<BenchmarkResult> &results) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open CSV output: " << path << "\n";
        return;
    }
    ofs << "context_length,baseline_latency_ms,tenzo_latency_ms,speedup,"
        << "baseline_invocations_sec,tenzo_invocations_sec,baseline_mem_mb,tenzo_mem_mb,mem_reduction_ratio,"
        << "baseline_bw_gbps,tenzo_bw_gbps\n";
    for (const auto &r : results) {
        ofs << r.context_length << ","
            << r.baseline_latency_ms << ","
            << r.tenzo_latency_ms << ","
            << r.speedup << ","
            << r.baseline_tokens_per_sec << ","
            << r.tenzo_tokens_per_sec << ","
            << r.baseline_mem_mb << ","
            << r.tenzo_mem_mb << ","
            << r.mem_reduction_ratio << ","
            << r.baseline_bw_gbps << ","
            << r.tenzo_bw_gbps << "\n";
    }
    std::cout << "📁 Exported benchmark metrics to CSV: " << path << "\n";
}

// ==============================================================================
// 6. Main Entry Point
// ==============================================================================

// ==============================================================================
// 6. Four-variant attention ablation (--variant a|b|c|d), one process per variant
//    a) FP32 KV + physical per-query-head GQA copy     (the "baseline" of the paper)
//    b) FP32 KV + direct GQA indexing                  (a minus the copy  -> zero-copy contribution)
//    c) INT8 KV (per-token scale) + direct indexing    (representation contribution of 8-bit)
//    d) Ternary packed KV (per-token scale) + direct   (representation contribution of 2-bit)
//  Data: seeded Gaussian K/V/Q (NOT ternary-valued: a ternary-valued input would make d lossless by construction).
//  Packing KV is timed separately (pack_ms_total, pack_us_per_token). RSS comes from /proc/self/status.
//  --error-only builds every variant in one process and reports the output error of c and d relative to a (untimed).
// ==============================================================================
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static inline float gauss_at(uint64_t seed, uint64_t stream, uint64_t idx) {
    uint64_t h1 = splitmix64(seed * 0x100000001B3ULL + (stream << 56) + idx * 2);
    uint64_t h2 = splitmix64(seed * 0x100000001B3ULL + (stream << 56) + idx * 2 + 1);
    double u1 = ((h1 >> 11) + 1.0) * (1.0 / 9007199254740993.0);   // (0,1]
    double u2 = (h2 >> 11) * (1.0 / 9007199254740992.0);           // [0,1)
    return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
}
static void gen_head_fp32(float *dst, uint64_t seed, uint64_t stream, int64_t h, int64_t T, int64_t D) {
    uint64_t base = static_cast<uint64_t>(h) * T * D;
    for (int64_t i = 0; i < T * D; ++i) dst[i] = gauss_at(seed, stream, base + i);
}

static void read_proc_kb(long &rss, long &hwm) {
    rss = hwm = -1;
    std::ifstream f("/proc/self/status"); std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) rss = std::atol(line.c_str() + 6);
        else if (line.rfind("VmHWM:", 0) == 0) hwm = std::atol(line.c_str() + 6);
    }
}

static inline float dot_f32(const float *a, const float *b, int64_t D) {
#if defined(TENZO_ARCH_X86)
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int64_t d = 0; d < D; d += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d + 8), _mm256_loadu_ps(b + d + 8), acc1);
    }
    alignas(32) float t[8]; _mm256_store_ps(t, _mm256_add_ps(acc0, acc1));
    return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
#else
    float s = 0.f; for (int64_t d = 0; d < D; ++d) s += a[d] * b[d]; return s;
#endif
}
static inline float dot_f32_i8(const float *q, const int8_t *k, int64_t D) {
#if defined(TENZO_ARCH_X86)
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int64_t d = 0; d < D; d += 16) {
        __m256 k0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(k + d))));
        __m256 k1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(k + d + 8))));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d), k0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 8), k1, acc1);
    }
    alignas(32) float t[8]; _mm256_store_ps(t, _mm256_add_ps(acc0, acc1));
    return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
#else
    float s = 0.f; for (int64_t d = 0; d < D; ++d) s += q[d] * static_cast<float>(k[d]); return s;
#endif
}
static inline void axpy_f32(float *out, const float *v, float w, int64_t D) {
#if defined(TENZO_ARCH_X86)
    __m256 wv = _mm256_set1_ps(w);
    for (int64_t d = 0; d < D; d += 8)
        _mm256_storeu_ps(out + d, _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + d), _mm256_loadu_ps(out + d)));
#else
    for (int64_t d = 0; d < D; ++d) out[d] += w * v[d];
#endif
}
static inline void axpy_i8(float *out, const int8_t *v, float w, int64_t D) {
#if defined(TENZO_ARCH_X86)
    __m256 wv = _mm256_set1_ps(w);
    for (int64_t d = 0; d < D; d += 8) {
        __m256 x = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(v + d))));
        _mm256_storeu_ps(out + d, _mm256_fmadd_ps(wv, x, _mm256_loadu_ps(out + d)));
    }
#else
    for (int64_t d = 0; d < D; ++d) out[d] += w * static_cast<float>(v[d]);
#endif
}

// a/b: FP32 attention for one query head over K/V pointers [T, D] (a passes the replicated copy, b the cache itself)
static void attn_fp32_head(const float *q, const float *K, const float *V, float *scores, float *out,
                           int64_t T, int64_t D, float scale) {
    for (int64_t t = 0; t < T; ++t) scores[t] = dot_f32(q, K + t * D, D) * scale;
    compute_softmax(scores, T);
    for (int64_t d = 0; d < D; ++d) out[d] = 0.f;
    for (int64_t t = 0; t < T; ++t) { float w = scores[t]; if (std::abs(w) < 1e-9f) continue; axpy_f32(out, V + t * D, w, D); }
}
static void attn_int8_head(const float *q, const int8_t *K, const float *ks, const int8_t *V, const float *vs,
                           float *scores, float *out, int64_t T, int64_t D, float scale) {
    for (int64_t t = 0; t < T; ++t) scores[t] = dot_f32_i8(q, K + t * D, D) * (ks[t] * scale);
    compute_softmax(scores, T);
    for (int64_t d = 0; d < D; ++d) out[d] = 0.f;
    for (int64_t t = 0; t < T; ++t) { float w = scores[t]; if (std::abs(w) < 1e-9f) continue; axpy_i8(out, V + t * D, w * vs[t], D); }
}
// d: packed ternary KV + per-token scale; query quantised to int16 as in run_tenzo_packed_attention
static void attn_tern_head(const float *q, const uint8_t *K, const float *ks, const uint8_t *V, const float *vs,
                           float *scores, float *out, int16_t *q_i16, int64_t T, int64_t D, float scale) {
    const int64_t bpt = D / 4;
    float mq = 0.f; for (int64_t d = 0; d < D; ++d) mq = std::max(mq, std::abs(q[d]));
    float sq = mq > 0.f ? mq / 32767.f : 1.f, inv = 1.f / sq;
    for (int64_t d = 0; d < D; ++d) q_i16[d] = static_cast<int16_t>(std::round(q[d] * inv));
    const float eff = sq * scale;
#if defined(TENZO_ARCH_X86)
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i lut0 = _mm_setr_epi8(-1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0);
    const __m128i lut1 = _mm_setr_epi8(-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0);
#endif
    for (int64_t t = 0; t < T; ++t) {
        const uint8_t *kt = K + t * bpt;
#if defined(TENZO_ARCH_X86)
        __m256i acc = _mm256_setzero_si256();
        for (int64_t d = 0; d < D; d += 32) {
            __m256i q_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q_i16 + d));
            __m256i q_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q_i16 + d + 16));
            __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(kt + (d / 4)));
            __m128i nl = _mm_and_si128(raw, mask_0f), nh = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);
            __m128i w01 = _mm_unpacklo_epi8(_mm_shuffle_epi8(lut0, nl), _mm_shuffle_epi8(lut1, nl));
            __m128i w23 = _mm_unpacklo_epi8(_mm_shuffle_epi8(lut0, nh), _mm_shuffle_epi8(lut1, nh));
            __m256i k_lo = _mm256_cvtepi8_epi16(_mm_unpacklo_epi16(w01, w23));
            __m256i k_hi = _mm256_cvtepi8_epi16(_mm_unpackhi_epi16(w01, w23));
            acc = _mm256_add_epi32(acc, _mm256_add_epi32(_mm256_madd_epi16(q_lo, k_lo), _mm256_madd_epi16(q_hi, k_hi)));
        }
        alignas(32) int32_t b[8]; _mm256_store_si256(reinterpret_cast<__m256i *>(b), acc);
        scores[t] = static_cast<float>(b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + b[7]) * eff * ks[t];
#else
        static const int8_t w0[16] = {-1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0};
        static const int8_t w1[16] = {-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0};
        int32_t tot = 0;
        for (int64_t b = 0; b < bpt; ++b) {
            uint8_t by = kt[b], lo = by & 0x0F, hi = by >> 4; int64_t d = b * 4;
            tot += q_i16[d] * w0[lo] + q_i16[d + 1] * w1[lo] + q_i16[d + 2] * w0[hi] + q_i16[d + 3] * w1[hi];
        }
        scores[t] = static_cast<float>(tot) * eff * ks[t];
#endif
    }
    compute_softmax(scores, T);
    for (int64_t d = 0; d < D; ++d) out[d] = 0.f;
    for (int64_t t = 0; t < T; ++t) {
        float w = scores[t] * vs[t];
        if (std::abs(w) < 1e-12f) continue;
        const uint8_t *vt = V + t * bpt;
        float delta[4] = {-w, 0.f, w, 0.f};
        for (int64_t b = 0; b < bpt; ++b) {
            uint8_t by = vt[b], lo = by & 0x0F, hi = by >> 4; int64_t d = b * 4;
            out[d] += delta[lo & 3]; out[d + 1] += delta[(lo >> 2) & 3]; out[d + 2] += delta[hi & 3]; out[d + 3] += delta[(hi >> 2) & 3];
        }
    }
}

struct AblOpts {
    char variant = 'b'; int64_t Hq = 20, Hkv = 5, D = 128, T = 4096; int threads = 1; int warmup = 5; int iters = 30;
    uint64_t seed = 0; std::string csv, governor = "unknown", label; bool error_only = false;
};
// Per-(kv head, token) quantisers; scales are stored [Hkv*T]
static void quant_int8_head(const float *x, int8_t *dst, float *sc, int64_t T, int64_t D) {
    for (int64_t t = 0; t < T; ++t) {
        float m = 0.f; for (int64_t d = 0; d < D; ++d) m = std::max(m, std::abs(x[t * D + d]));
        float s = m > 0.f ? m / 127.f : 1.f; sc[t] = s; float inv = 1.f / s;
        for (int64_t d = 0; d < D; ++d) dst[t * D + d] = static_cast<int8_t>(std::lrintf(x[t * D + d] * inv));
    }
}
static void quant_tern_head(const float *x, uint8_t *dst, float *sc, int64_t T, int64_t D) {
    for (int64_t t = 0; t < T; ++t) {
        float m = 0.f; for (int64_t d = 0; d < D; ++d) m += std::abs(x[t * D + d]);
        float s = m / static_cast<float>(D); if (s <= 0.f) s = 1.f; sc[t] = s; float inv = 1.f / s;   // absmean scale (BitNet-style)
        for (int64_t b = 0; b < D / 4; ++b) {
            int v[4];
            for (int j = 0; j < 4; ++j) v[j] = std::max(-1, std::min(1, static_cast<int>(std::lrintf(x[t * D + b * 4 + j] * inv))));
            dst[t * (D / 4) + b] = pack_ternary_byte(v[0], v[1], v[2], v[3]);
        }
    }
}

struct AblData {                       // everything one variant needs; only the needed members are allocated
    std::vector<float> q, k32, v32, ks, vs;
    std::vector<int8_t> k8, v8; std::vector<uint8_t> kt, vt;
    double pack_ms = 0.0; double kv_bytes = 0.0;
};
static void build_data(const AblOpts &o, char variant, AblData &d, bool want_fp32_too) {
    const int64_t Hq = o.Hq, Hkv = o.Hkv, T = o.T, D = o.D;
    d.q.resize(Hq * D);
    for (int64_t i = 0; i < Hq * D; ++i) d.q[i] = gauss_at(o.seed, 0, i);
    bool need32 = (variant == 'a' || variant == 'b' || want_fp32_too);
    bool needq8 = (variant == 'c'), needt = (variant == 'd');
    if (need32) { d.k32.resize(Hkv * T * D); d.v32.resize(Hkv * T * D); }
    if (needq8) { d.k8.resize(Hkv * T * D); d.v8.resize(Hkv * T * D); }
    if (needt)  { d.kt.resize(Hkv * T * (D / 4)); d.vt.resize(Hkv * T * (D / 4)); }
    if (needq8 || needt) { d.ks.resize(Hkv * T); d.vs.resize(Hkv * T); }
    std::vector<float> chunk_k(T * D), chunk_v(T * D);          // one KV head at a time -> no full FP32 cache for c/d
    double pack_total = 0.0;
    for (int64_t h = 0; h < Hkv; ++h) {
        float *kk = need32 ? d.k32.data() + h * T * D : chunk_k.data();
        float *vv = need32 ? d.v32.data() + h * T * D : chunk_v.data();
        gen_head_fp32(kk, o.seed, 1, h, T, D); gen_head_fp32(vv, o.seed, 2, h, T, D);
        if (needq8 || needt) {
            auto t0 = std::chrono::steady_clock::now();
            if (needq8) { quant_int8_head(kk, d.k8.data() + h * T * D, d.ks.data() + h * T, T, D);
                          quant_int8_head(vv, d.v8.data() + h * T * D, d.vs.data() + h * T, T, D); }
            else        { quant_tern_head(kk, d.kt.data() + h * T * (D / 4), d.ks.data() + h * T, T, D);
                          quant_tern_head(vv, d.vt.data() + h * T * (D / 4), d.vs.data() + h * T, T, D); }
            pack_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
    }
    d.pack_ms = pack_total;
}
// One full attention call (all query heads, parallel over heads). `scratch` holds per-thread buffers.
struct AblScratch { std::vector<float> scores, kcopy, vcopy; std::vector<int16_t> qi; };
static void run_variant(const AblOpts &o, char variant, const AblData &d, std::vector<AblScratch> &sc, float *out) {
    const int64_t Hq = o.Hq, Hkv = o.Hkv, T = o.T, D = o.D, ratio = Hq / Hkv; const float scale = 1.0f / std::sqrt(static_cast<float>(D));
#pragma omp parallel for schedule(static) num_threads(o.threads)
    for (int64_t hq = 0; hq < Hq; ++hq) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        AblScratch &s = sc[tid]; int64_t hk = hq / ratio; const float *q = d.q.data() + hq * D; float *op = out + hq * D;
        if (variant == 'a') {
            std::memcpy(s.kcopy.data(), d.k32.data() + hk * T * D, T * D * sizeof(float));      // physical GQA broadcast copy
            std::memcpy(s.vcopy.data(), d.v32.data() + hk * T * D, T * D * sizeof(float));
            attn_fp32_head(q, s.kcopy.data(), s.vcopy.data(), s.scores.data(), op, T, D, scale);
        } else if (variant == 'b') {
            attn_fp32_head(q, d.k32.data() + hk * T * D, d.v32.data() + hk * T * D, s.scores.data(), op, T, D, scale);
        } else if (variant == 'c') {
            attn_int8_head(q, d.k8.data() + hk * T * D, d.ks.data() + hk * T, d.v8.data() + hk * T * D, d.vs.data() + hk * T, s.scores.data(), op, T, D, scale);
        } else {
            attn_tern_head(q, d.kt.data() + hk * T * (D / 4), d.ks.data() + hk * T, d.vt.data() + hk * T * (D / 4), d.vs.data() + hk * T,
                           s.scores.data(), op, s.qi.data(), T, D, scale);
        }
    }
}
static std::vector<AblScratch> make_scratch(const AblOpts &o, char variant) {
    std::vector<AblScratch> sc(std::max(1, o.threads));
    for (auto &s : sc) { s.scores.assign(o.T, 0.f); s.qi.assign(o.D, 0);
        if (variant == 'a') { s.kcopy.assign(o.T * o.D, 0.f); s.vcopy.assign(o.T * o.D, 0.f); } }
    return sc;
}
static double kv_bytes_of(const AblOpts &o, char v) {
    double e = 2.0 * o.Hkv * o.T;                                   // K and V tokens
    if (v == 'a' || v == 'b') return e * o.D * 4.0;
    if (v == 'c') return e * o.D + e * 4.0;                          // int8 + one fp32 scale per token
    return e * (o.D / 4.0) + e * 4.0;                                // 2-bit packed + one fp32 scale per token
}
static void csv_row(const AblOpts &o, const std::string &line, const std::string &header) {
    if (o.csv.empty()) { std::cout << line << "\n"; return; }
    bool exists = std::ifstream(o.csv).good();
    std::ofstream f(o.csv, std::ios::app); if (!exists) f << header << "\n"; f << line << "\n";
}
static const char *ABL_HEADER =
    "kind,variant,Hq,Hkv,D,T,threads,warmup,iters,median_ms,q1_ms,q3_ms,iqr_ms,min_ms,pack_ms_total,pack_us_per_kv_token,kv_bytes,rss_kb,hwm_kb,"
    "err_max_abs,err_rel_l2,err_cosine,seed,data,governor,label";

static int run_ablation(const AblOpts &o) {
    if (o.Hq % o.Hkv || o.D % 32) { std::cerr << "bad geometry\n"; return 2; }
    std::ostringstream pre;
    if (o.error_only) {
        AblData da; build_data(o, 'a', da, false);
        auto sa = make_scratch(o, 'a'); std::vector<float> ref(o.Hq * o.D);
        run_variant(o, 'a', da, sa, ref.data());
        for (char v : {'b', 'c', 'd'}) {
            AblData dv; build_data(o, v, dv, v == 'b');             // 'b' reuses fp32 (regenerated, identical values)
            auto sv = make_scratch(o, v); std::vector<float> out(o.Hq * o.D);
            run_variant(o, v, dv, sv, out.data());
            double dmax = 0, num = 0, den = 0, dot = 0, nu = 0, nv = 0;
            for (size_t i = 0; i < out.size(); ++i) {
                double e = (double)out[i] - (double)ref[i]; dmax = std::max(dmax, std::abs(e)); num += e * e; den += (double)ref[i] * ref[i];
                dot += (double)out[i] * ref[i]; nu += (double)out[i] * out[i]; nv += (double)ref[i] * ref[i];
            }
            std::ostringstream l;
            l << "error," << v << "," << o.Hq << "," << o.Hkv << "," << o.D << "," << o.T << "," << o.threads << ",,,,,,,,,,,,,"
              << dmax << "," << std::sqrt(num / den) << "," << dot / std::sqrt(nu * nv) << "," << o.seed << ",gaussian," << o.governor << "," << o.label;
            csv_row(o, l.str(), ABL_HEADER);
        }
        return 0;
    }
    AblData d; build_data(o, o.variant, d, false);
    auto sc = make_scratch(o, o.variant); std::vector<float> out(o.Hq * o.D);
    for (int i = 0; i < o.warmup; ++i) run_variant(o, o.variant, d, sc, out.data());
    std::vector<double> ms(o.iters);
    for (int i = 0; i < o.iters; ++i) {
        auto t0 = std::chrono::steady_clock::now(); run_variant(o, o.variant, d, sc, out.data());
        ms[i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    std::sort(ms.begin(), ms.end());
    auto pct = [&](double p) { double x = p * (ms.size() - 1); size_t i = (size_t)x; double f = x - i; return i + 1 < ms.size() ? ms[i] * (1 - f) + ms[i + 1] * f : ms[i]; };
    long rss, hwm; read_proc_kb(rss, hwm);
    double pack_us = (o.variant == 'c' || o.variant == 'd') ? d.pack_ms * 1000.0 / (2.0 * o.Hkv * o.T) : 0.0;
    std::ostringstream l; l << std::setprecision(9);
    l << "timing," << o.variant << "," << o.Hq << "," << o.Hkv << "," << o.D << "," << o.T << "," << o.threads << "," << o.warmup << "," << o.iters << ","
      << pct(0.5) << "," << pct(0.25) << "," << pct(0.75) << "," << (pct(0.75) - pct(0.25)) << "," << ms.front() << ","
      << d.pack_ms << "," << pack_us << "," << kv_bytes_of(o, o.variant) << "," << rss << "," << hwm << ",,,," << o.seed << ",gaussian," << o.governor << "," << o.label;
    csv_row(o, l.str(), ABL_HEADER);
    return 0;
}
static bool parse_ablation(int argc, char **argv, AblOpts &o) {
    bool any = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&](int64_t &x) { x = std::stoll(argv[++i]); };
        if (a == "--variant" && i + 1 < argc) { o.variant = argv[++i][0]; any = true; }
        else if (a == "--error-only") { o.error_only = true; any = true; }
        else if (a == "--hq" && i + 1 < argc) nx(o.Hq);
        else if (a == "--hkv" && i + 1 < argc) nx(o.Hkv);
        else if (a == "--dim" && i + 1 < argc) nx(o.D);
        else if (a == "--context" && i + 1 < argc) nx(o.T);
        else if (a == "--threads" && i + 1 < argc) o.threads = std::stoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) o.warmup = std::stoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) o.iters = std::stoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) o.seed = std::stoull(argv[++i]);
        else if (a == "--csv" && i + 1 < argc) o.csv = argv[++i];
        else if (a == "--governor" && i + 1 < argc) o.governor = argv[++i];
        else if (a == "--label" && i + 1 < argc) o.label = argv[++i];
    }
    return any;
}

int legacy_main(int argc, char **argv);
int main(int argc, char **argv) {
    AblOpts o;
    if (parse_ablation(argc, argv, o)) {
        if (!o.error_only && !(o.variant == 'a' || o.variant == 'b' || o.variant == 'c' || o.variant == 'd')) { std::cerr << "--variant must be a|b|c|d\n"; return 2; }
        return run_ablation(o);
    }
    return legacy_main(argc, argv);
}

int legacy_main(int argc, char **argv) {
    BenchmarkConfig cfg;
    std::vector<int64_t> contexts = {1024, 4096, 8192, 32768};

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--baseline") {
            cfg.run_baseline = true;
            cfg.run_tenzo = false;
        } else if (arg == "--tenzo") {
            cfg.run_baseline = false;
            cfg.run_tenzo = true;
        } else if (arg == "--context" && i + 1 < argc) {
            contexts = {std::stoll(argv[++i])};
        } else if (arg == "--csv" && i + 1 < argc) {
            cfg.csv_file = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            cfg.timed_iters = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: paper_benchmark [options]\n"
                      << "  --baseline       Run only Baseline Attention (useful for isolated perf profiling)\n"
                      << "  --tenzo          Run only Tenzo Packed Attention (useful for isolated perf profiling)\n"
                      << "  --context <N>    Run single context length (e.g. 1024, 4096, 8192, 32768)\n"
                      << "  --csv <file>     Export results to CSV for plotting\n"
                      << "  --iters <N>      Number of timed iterations (default: 25)\n";
            return 0;
        }
    }

    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  🚀 TENZO COMPILER: PAPER MICRO-BENCHMARK (GQA ATTENTION)            ║\n";
#if defined(TENZO_ARCH_X86)
    std::cout << "║  Architecture: x86_64 AVX2 + FMA SIMD Micro-Kernel                   ║\n";
#elif defined(TENZO_ARCH_ARM)
    std::cout << "║  Architecture: ARM64 NEON / DotProd SIMD                             ║\n";
#else
    std::cout << "║  Architecture: Generic Portable SIMD / Scalar Fallback               ║\n";
#endif
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";

    std::vector<BenchmarkResult> results;
    for (int64_t T : contexts) {
        run_context_benchmark(T, cfg, results);
    }

    if (cfg.run_baseline && cfg.run_tenzo) {
        print_paper_summary_table(results);
    }

    if (!cfg.csv_file.empty()) {
        export_csv(cfg.csv_file, results);
    }

    return 0;
}
