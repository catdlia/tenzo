/**
 * @file TenzoEngine.cpp
 * @brief Tenzo Native Inference Engine & C-API Implementation (Phase E)
 */

#include "TenzoEngine.h"
#include "VulkanRuntime.h"
#include "CUDARuntime.h"
#include "ROCmRuntime.h"
#include "MicroarchProfiler.h"
#include "arch/RISCV_RVV.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <chrono>

namespace tenzo {

namespace {

inline float build_lut(
    const float* __restrict__ act,
    int64_t K,
    __m256i* __restrict__ lut_vecs
) {
    const int64_t K_half = K / 2;
    float max_abs = 1e-6f;
    for (int64_t k = 0; k < K; ++k) {
        float val = std::abs(act[k]);
        if (val > max_abs) max_abs = val;
    }

    const float scale_act = 31.0f / max_abs;
    const float inv_scale_act = max_abs / 31.0f;

    static const int w0_vals[3] = {-1, 0, 1};
    static const int w1_vals[3] = {-1, 0, 1};

    for (int64_t k = 0; k < K_half; ++k) {
        int16_t a0 = static_cast<int16_t>(std::round(act[2 * k] * scale_act));
        int16_t a1 = static_cast<int16_t>(std::round(act[2 * k + 1] * scale_act));

        alignas(32) int8_t lut_bytes[32] = {0};

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                int w0 = w0_vals[i];
                int w1 = w1_vals[j];
                int idx = (w0 + 1) | ((w1 + 1) << 2);

                int32_t sum = w0 * a0 + w1 * a1;
                int8_t val_i8 = static_cast<int8_t>(std::clamp(sum, -128, 127));

                lut_bytes[idx] = val_i8;
                lut_bytes[idx + 16] = val_i8; // Duplicate for top 128-bit lane
            }
        }
        lut_vecs[k] = _mm256_load_si256(reinterpret_cast<const __m256i*>(lut_bytes));
    }
    return inv_scale_act;
}

inline void compute_single_block(
    const __m256i* __restrict__ lut_vecs,
    const int8_t* __restrict__ block_w,
    int64_t K_half,
    float* __restrict__ out_block,
    __m256 scale_vec,
    __m256i mask_low
) {
    __m256i acc_low_32_0 = _mm256_setzero_si256();
    __m256i acc_low_32_1 = _mm256_setzero_si256();
    __m256i acc_low_32_2 = _mm256_setzero_si256();
    __m256i acc_low_32_3 = _mm256_setzero_si256();

    __m256i acc_high_32_0 = _mm256_setzero_si256();
    __m256i acc_high_32_1 = _mm256_setzero_si256();
    __m256i acc_high_32_2 = _mm256_setzero_si256();
    __m256i acc_high_32_3 = _mm256_setzero_si256();

    int64_t k = 0;
    while (k < K_half) {
        int64_t chunk = std::min<int64_t>(64, K_half - k);
        __m256i acc_low_16_0 = _mm256_setzero_si256();
        __m256i acc_low_16_1 = _mm256_setzero_si256();
        __m256i acc_high_16_0 = _mm256_setzero_si256();
        __m256i acc_high_16_1 = _mm256_setzero_si256();

        for (int64_t c = 0; c < chunk; ++c, ++k) {
            __m256i w_bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block_w + k * 32));
            __m256i lut = lut_vecs[k];

            __m256i idx_low = _mm256_and_si256(w_bytes, mask_low);
            __m256i idx_high = _mm256_and_si256(_mm256_srli_epi16(w_bytes, 4), mask_low);

            __m256i res_low = _mm256_shuffle_epi8(lut, idx_low);
            __m256i res_high = _mm256_shuffle_epi8(lut, idx_high);

            acc_low_16_0 = _mm256_add_epi16(acc_low_16_0, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(res_low, 0)));
            acc_low_16_1 = _mm256_add_epi16(acc_low_16_1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(res_low, 1)));

            acc_high_16_0 = _mm256_add_epi16(acc_high_16_0, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(res_high, 0)));
            acc_high_16_1 = _mm256_add_epi16(acc_high_16_1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(res_high, 1)));
        }

        acc_low_32_0 = _mm256_add_epi32(acc_low_32_0, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_low_16_0, 0)));
        acc_low_32_1 = _mm256_add_epi32(acc_low_32_1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_low_16_0, 1)));
        acc_low_32_2 = _mm256_add_epi32(acc_low_32_2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_low_16_1, 0)));
        acc_low_32_3 = _mm256_add_epi32(acc_low_32_3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_low_16_1, 1)));

        acc_high_32_0 = _mm256_add_epi32(acc_high_32_0, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_high_16_0, 0)));
        acc_high_32_1 = _mm256_add_epi32(acc_high_32_1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_high_16_0, 1)));
        acc_high_32_2 = _mm256_add_epi32(acc_high_32_2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_high_16_1, 0)));
        acc_high_32_3 = _mm256_add_epi32(acc_high_32_3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_high_16_1, 1)));
    }

    _mm256_storeu_ps(out_block + 0, _mm256_mul_ps(_mm256_cvtepi32_ps(acc_low_32_0), scale_vec));
    _mm256_storeu_ps(out_block + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(acc_low_32_1), scale_vec));
    _mm256_storeu_ps(out_block + 16, _mm256_mul_ps(_mm256_cvtepi32_ps(acc_low_32_2), scale_vec));
    _mm256_storeu_ps(out_block + 24, _mm256_mul_ps(_mm256_cvtepi32_ps(acc_low_32_3), scale_vec));

    _mm256_storeu_ps(out_block + 32, _mm256_mul_ps(_mm256_cvtepi32_ps(acc_high_32_0), scale_vec));
    _mm256_storeu_ps(out_block + 40, _mm256_mul_ps(_mm256_cvtepi32_ps(acc_high_32_1), scale_vec));
    _mm256_storeu_ps(out_block + 48, _mm256_mul_ps(_mm256_cvtepi32_ps(acc_high_32_2), scale_vec));
    _mm256_storeu_ps(out_block + 56, _mm256_mul_ps(_mm256_cvtepi32_ps(acc_high_32_3), scale_vec));
}

inline void compute_from_lut(
    const __m256i* __restrict__ lut_vecs,
    const int8_t* __restrict__ packed_w,
    int64_t n_blocks,
    int64_t K_half,
    float* __restrict__ out,
    float total_scale
) {
    if (!packed_w || n_blocks <= 0) return;
    const __m256i mask_low = _mm256_set1_epi8(0x0F);
    const __m256 scale_vec = _mm256_set1_ps(total_scale);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int64_t b = 0; b < n_blocks; ++b) {
        const int8_t* block_w = packed_w + b * K_half * 32;
        compute_single_block(lut_vecs, block_w, K_half, out + b * 64, scale_vec, mask_low);
    }
}

inline void rms_norm_raw(const float* src, const float* w, float* dst, int dim, float eps = 1e-5f) {
    float sum_sq = 0.0f;
    #pragma omp simd reduction(+:sum_sq)
    for (int i = 0; i < dim; ++i) {
        sum_sq += src[i] * src[i];
    }
    float inv_rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(dim)) + eps);
    if (w) {
        #pragma omp simd
        for (int i = 0; i < dim; ++i) {
            dst[i] = src[i] * inv_rms * w[i];
        }
    } else {
        #pragma omp simd
        for (int i = 0; i < dim; ++i) {
            dst[i] = src[i] * inv_rms;
        }
    }
}

} // namespace

//===----------------------------------------------------------------------===//
// FusedKVCache Implementation (INT8, Ternary TL1, FP32)
//===----------------------------------------------------------------------===//

FusedKVCache::FusedKVCache(
    int num_layers,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    const std::string& kv_mode
) : num_layers(num_layers), num_q_heads(num_q_heads), num_kv_heads(num_kv_heads),
    head_dim(head_dim), max_seq_len(max_seq_len), kv_mode(kv_mode) {
    
    size_t total_elements = static_cast<size_t>(num_layers) * max_seq_len * num_kv_heads * head_dim;
    size_t total_scales = static_cast<size_t>(num_layers) * max_seq_len * num_kv_heads;

    if (kv_mode == "int8_fused") {
        int8_k.resize(total_elements);
        int8_v.resize(total_elements);
        int8_k_scales.resize(total_scales);
        int8_v_scales.resize(total_scales);
    } else if (kv_mode == "tl1_fused") {
        // TL1 packs 4 channels into 1 byte (2 bits each)
        size_t total_tl1_bytes = total_elements / 4;
        tl1_k.resize(total_tl1_bytes);
        tl1_v.resize(total_tl1_bytes);
        tl1_k_scales.resize(total_scales);
        tl1_v_scales.resize(total_scales);
    } else if (kv_mode == "popcount_fused") {
        // Popcount bitplanes: 32 bytes per token head (contiguous struct)
        size_t total_heads_tokens = static_cast<size_t>(num_layers) * num_kv_heads * max_seq_len;
        size_t total_tl1_v_bytes = total_elements / 4;
        popcount_k.resize(total_heads_tokens);
        popcount_k_scales.resize(total_scales);
        popcount_v.resize(total_tl1_v_bytes);
        popcount_v_scales.resize(total_scales);
    } else {
        fp32_k.resize(total_elements);
        fp32_v.resize(total_elements);
    }
}

void FusedKVCache::reset() {
    cur_seq_len = 0;
}

//===----------------------------------------------------------------------===//
// Paged Attention Dynamic KV-Cache Implementation
//===----------------------------------------------------------------------===//
PagedKVCache::PagedKVCache(
    int num_layers,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int total_blocks
) : num_layers(num_layers),
    num_q_heads(num_q_heads),
    num_kv_heads(num_kv_heads),
    head_dim(head_dim),
    total_blocks(total_blocks),
    block_mgr(total_blocks, BLOCK_SIZE) {

    size_t total_elements = static_cast<size_t>(total_blocks) * num_layers * num_kv_heads * BLOCK_SIZE * head_dim;
    paged_k_pool.resize(total_elements, 0);
    paged_v_pool.resize(total_elements, 0);

    size_t total_scales = static_cast<size_t>(total_blocks) * num_layers * num_kv_heads * BLOCK_SIZE;
    paged_k_scales.resize(total_scales, 1.0f);
    paged_v_scales.resize(total_scales, 1.0f);
}

void PagedKVCache::reset() {
    cur_seq_len = 0;
    block_table.clear();
    block_mgr.reset();
}

void PagedKVCache::forward_attention_raw(
    int layer_idx,
    float* __restrict__ q_ptr,
    float* __restrict__ k_ptr,
    float* __restrict__ v_ptr,
    float* __restrict__ out_ptr
) {
    const int pos = cur_seq_len;
    const int half_dim = head_dim / 2;
    const float theta_base = 10000.0f;

    // 1. In-place AVX2 RoPE on Q
    for (int h = 0; h < num_q_heads; ++h) {
        float* q_head = q_ptr + h * head_dim;
        for (int i = 0; i < half_dim; i += 8) {
            alignas(32) float cos_arr[8];
            alignas(32) float sin_arr[8];
            for (int j = 0; j < 8; ++j) {
                int idx = i + j;
                float freq = 1.0f / std::pow(theta_base, static_cast<float>(2 * idx) / head_dim);
                float angle = static_cast<float>(pos) * freq;
                cos_arr[j] = std::cos(angle);
                sin_arr[j] = std::sin(angle);
            }
            __m256 cos_vec = _mm256_load_ps(cos_arr);
            __m256 sin_vec = _mm256_load_ps(sin_arr);
            __m256 x0 = _mm256_loadu_ps(q_head + i);
            __m256 x1 = _mm256_loadu_ps(q_head + i + half_dim);
            __m256 r0 = _mm256_fmsub_ps(x0, cos_vec, _mm256_mul_ps(x1, sin_vec));
            __m256 r1 = _mm256_fmadd_ps(x0, sin_vec, _mm256_mul_ps(x1, cos_vec));
            _mm256_storeu_ps(q_head + i, r0);
            _mm256_storeu_ps(q_head + i + half_dim, r1);
        }
    }

    // 2. In-place AVX2 RoPE on K
    for (int h = 0; h < num_kv_heads; ++h) {
        float* k_head = k_ptr + h * head_dim;
        for (int i = 0; i < half_dim; i += 8) {
            alignas(32) float cos_arr[8];
            alignas(32) float sin_arr[8];
            for (int j = 0; j < 8; ++j) {
                int idx = i + j;
                float freq = 1.0f / std::pow(theta_base, static_cast<float>(2 * idx) / head_dim);
                float angle = static_cast<float>(pos) * freq;
                cos_arr[j] = std::cos(angle);
                sin_arr[j] = std::sin(angle);
            }
            __m256 cos_vec = _mm256_load_ps(cos_arr);
            __m256 sin_vec = _mm256_load_ps(sin_arr);
            __m256 x0 = _mm256_loadu_ps(k_head + i);
            __m256 x1 = _mm256_loadu_ps(k_head + i + half_dim);
            __m256 r0 = _mm256_fmsub_ps(x0, cos_vec, _mm256_mul_ps(x1, sin_vec));
            __m256 r1 = _mm256_fmadd_ps(x0, sin_vec, _mm256_mul_ps(x1, cos_vec));
            _mm256_storeu_ps(k_head + i, r0);
            _mm256_storeu_ps(k_head + i + half_dim, r1);
        }
    }

    // 3. Dynamic Page Allocation
    int logical_block_idx = pos / BLOCK_SIZE;
    int slot_in_block = pos % BLOCK_SIZE;
    if (logical_block_idx >= static_cast<int>(block_table.size())) {
        int32_t new_blk = block_mgr.allocate_block();
        if (new_blk >= 0) {
            block_table.push_back(new_blk);
        }
    }

    int32_t phys_block = (logical_block_idx < static_cast<int>(block_table.size())) ? block_table[logical_block_idx] : 0;

    // 4. Quantize and write current token K and V into physical block page
    for (int h = 0; h < num_kv_heads; ++h) {
        size_t block_head_off = ((static_cast<size_t>(phys_block) * num_layers + layer_idx) * num_kv_heads + h) * BLOCK_SIZE * head_dim + slot_in_block * head_dim;
        size_t scale_off = ((static_cast<size_t>(phys_block) * num_layers + layer_idx) * num_kv_heads + h) * BLOCK_SIZE + slot_in_block;

        const float* k_src = k_ptr + h * head_dim;
        const float* v_src = v_ptr + h * head_dim;
        int8_t* k_dst = paged_k_pool.data() + block_head_off;
        int8_t* v_dst = paged_v_pool.data() + block_head_off;

        float amax_k = 1e-8f, amax_v = 1e-8f;
        for (int d = 0; d < head_dim; ++d) {
            amax_k = std::max(amax_k, std::abs(k_src[d]));
            amax_v = std::max(amax_v, std::abs(v_src[d]));
        }
        float s_k = amax_k / 127.0f;
        float s_v = amax_v / 127.0f;
        paged_k_scales[scale_off] = s_k;
        paged_v_scales[scale_off] = s_v;

        float inv_sk = 1.0f / s_k;
        float inv_sv = 1.0f / s_v;
        for (int d = 0; d < head_dim; ++d) {
            k_dst[d] = static_cast<int8_t>(std::clamp(std::round(k_src[d] * inv_sk), -128.0f, 127.0f));
            v_dst[d] = static_cast<int8_t>(std::clamp(std::round(v_src[d] * inv_sv), -128.0f, 127.0f));
        }
    }

    // 5. Paged Scaled Dot-Product Attention
    const int total_tokens = pos + 1;
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const float qk_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> scores(total_tokens);

    for (int qh = 0; qh < num_q_heads; ++qh) {
        int kv_h = qh / gqa_ratio;
        const float* q_head = q_ptr + qh * head_dim;

        float max_score = -1e30f;
        for (int b = 0; b <= logical_block_idx; ++b) {
            int32_t blk = block_table[b];
            int tokens_in_this_block = (b == logical_block_idx) ? (slot_in_block + 1) : BLOCK_SIZE;

            for (int s = 0; s < tokens_in_this_block; ++s) {
                int t = b * BLOCK_SIZE + s;
                size_t k_off = ((static_cast<size_t>(blk) * num_layers + layer_idx) * num_kv_heads + kv_h) * BLOCK_SIZE * head_dim + s * head_dim;
                size_t scale_off = ((static_cast<size_t>(blk) * num_layers + layer_idx) * num_kv_heads + kv_h) * BLOCK_SIZE + s;

                const int8_t* k_page = paged_k_pool.data() + k_off;
                float s_k = paged_k_scales[scale_off];

                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                for (int d = 0; d < head_dim; d += 16) {
                    __m256 q_v0 = _mm256_loadu_ps(q_head + d);
                    __m256 q_v1 = _mm256_loadu_ps(q_head + d + 8);

                    __m128i k_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(k_page + d));
                    __m256i k_epi16 = _mm256_cvtepi8_epi16(k_raw);
                    __m256 k_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(k_epi16)));
                    __m256 k_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(k_epi16, 1)));

                    acc0 = _mm256_fmadd_ps(q_v0, k_f0, acc0);
                    acc1 = _mm256_fmadd_ps(q_v1, k_f1, acc1);
                }
                __m256 acc = _mm256_add_ps(acc0, acc1);
                alignas(32) float acc_arr[8];
                _mm256_store_ps(acc_arr, acc);
                float dot = 0.0f;
                for (int i = 0; i < 8; ++i) dot += acc_arr[i];

                float score = dot * s_k * qk_scale;
                scores[t] = score;
                max_score = std::max(max_score, score);
            }
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int t = 0; t < total_tokens; ++t) {
            float exp_val = std::exp(scores[t] - max_score);
            scores[t] = exp_val;
            sum_exp += exp_val;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (int t = 0; t < total_tokens; ++t) {
            scores[t] *= inv_sum;
        }

        // Accumulate Weighted Values
        float* out_head = out_ptr + qh * head_dim;
        std::memset(out_head, 0, head_dim * sizeof(float));

        for (int b = 0; b <= logical_block_idx; ++b) {
            int32_t blk = block_table[b];
            int tokens_in_this_block = (b == logical_block_idx) ? (slot_in_block + 1) : BLOCK_SIZE;

            for (int s = 0; s < tokens_in_this_block; ++s) {
                int t = b * BLOCK_SIZE + s;
                float alpha = scores[t];
                if (alpha < 1e-7f) continue;

                size_t v_off = ((static_cast<size_t>(blk) * num_layers + layer_idx) * num_kv_heads + kv_h) * BLOCK_SIZE * head_dim + s * head_dim;
                size_t scale_off = ((static_cast<size_t>(blk) * num_layers + layer_idx) * num_kv_heads + kv_h) * BLOCK_SIZE + s;

                const int8_t* v_page = paged_v_pool.data() + v_off;
                float s_v = paged_v_scales[scale_off];
                float weight = alpha * s_v;
                __m256 w_vec = _mm256_set1_ps(weight);

                for (int d = 0; d < head_dim; d += 16) {
                    __m128i v_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(v_page + d));
                    __m256i v_epi16 = _mm256_cvtepi8_epi16(v_raw);
                    __m256 v_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_epi16)));
                    __m256 v_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_epi16, 1)));

                    __m256 cur_out0 = _mm256_loadu_ps(out_head + d);
                    __m256 cur_out1 = _mm256_loadu_ps(out_head + d + 8);

                    _mm256_storeu_ps(out_head + d, _mm256_fmadd_ps(w_vec, v_f0, cur_out0));
                    _mm256_storeu_ps(out_head + d + 8, _mm256_fmadd_ps(w_vec, v_f1, cur_out1));
                }
            }
        }
    }
}

void FusedKVCache::forward_attention_raw(
    int layer_idx,
    float* __restrict__ q_ptr,
    float* __restrict__ k_ptr,
    float* __restrict__ v_ptr,
    float* __restrict__ out_ptr
) {
    const int pos = cur_seq_len;
    const int half_dim = head_dim / 2;
    const float theta_base = 10000.0f;

    // 1. In-place AVX2 RoPE on Q
    for (int h = 0; h < num_q_heads; ++h) {
        float* q_head = q_ptr + h * head_dim;
        for (int i = 0; i < half_dim; i += 8) {
            alignas(32) float cos_arr[8];
            alignas(32) float sin_arr[8];
            for (int j = 0; j < 8; ++j) {
                int idx = i + j;
                float freq = 1.0f / std::pow(theta_base, static_cast<float>(2 * idx) / head_dim);
                float angle = static_cast<float>(pos) * freq;
                cos_arr[j] = std::cos(angle);
                sin_arr[j] = std::sin(angle);
            }
            __m256 cos_vec = _mm256_load_ps(cos_arr);
            __m256 sin_vec = _mm256_load_ps(sin_arr);
            __m256 x0 = _mm256_loadu_ps(q_head + i);
            __m256 x1 = _mm256_loadu_ps(q_head + i + half_dim);
            __m256 r0 = _mm256_fmsub_ps(x0, cos_vec, _mm256_mul_ps(x1, sin_vec));
            __m256 r1 = _mm256_fmadd_ps(x0, sin_vec, _mm256_mul_ps(x1, cos_vec));
            _mm256_storeu_ps(q_head + i, r0);
            _mm256_storeu_ps(q_head + i + half_dim, r1);
        }
    }

    // 2. In-place AVX2 RoPE on K
    for (int h = 0; h < num_kv_heads; ++h) {
        float* k_head = k_ptr + h * head_dim;
        for (int i = 0; i < half_dim; i += 8) {
            alignas(32) float cos_arr[8];
            alignas(32) float sin_arr[8];
            for (int j = 0; j < 8; ++j) {
                int idx = i + j;
                float freq = 1.0f / std::pow(theta_base, static_cast<float>(2 * idx) / head_dim);
                float angle = static_cast<float>(pos) * freq;
                cos_arr[j] = std::cos(angle);
                sin_arr[j] = std::sin(angle);
            }
            __m256 cos_vec = _mm256_load_ps(cos_arr);
            __m256 sin_vec = _mm256_load_ps(sin_arr);
            __m256 x0 = _mm256_loadu_ps(k_head + i);
            __m256 x1 = _mm256_loadu_ps(k_head + i + half_dim);
            __m256 r0 = _mm256_fmsub_ps(x0, cos_vec, _mm256_mul_ps(x1, sin_vec));
            __m256 r1 = _mm256_fmadd_ps(x0, sin_vec, _mm256_mul_ps(x1, cos_vec));
            _mm256_storeu_ps(k_head + i, r0);
            _mm256_storeu_ps(k_head + i + half_dim, r1);
        }
    }

    // 3. Append to KV-Cache in contiguous [layer][head][time][dim] layout
    if (kv_mode == "int8_fused") {
        for (int h = 0; h < num_kv_heads; ++h) {
            size_t head_off = (static_cast<size_t>(layer_idx) * num_kv_heads + h) * max_seq_len * head_dim + pos * head_dim;
            size_t scale_off = (static_cast<size_t>(layer_idx) * num_kv_heads + h) * max_seq_len + pos;

            const float* k_src = k_ptr + h * head_dim;
            const float* v_src = v_ptr + h * head_dim;
            int8_t* k_dst = int8_k.data() + head_off;
            int8_t* v_dst = int8_v.data() + head_off;

            float amax_k = 1e-8f, amax_v = 1e-8f;
            for (int d = 0; d < head_dim; ++d) {
                amax_k = std::max(amax_k, std::abs(k_src[d]));
                amax_v = std::max(amax_v, std::abs(v_src[d]));
            }
            float s_k = amax_k / 127.0f;
            float s_v = amax_v / 127.0f;
            int8_k_scales[scale_off] = s_k;
            int8_v_scales[scale_off] = s_v;

            float inv_sk = 1.0f / s_k;
            float inv_sv = 1.0f / s_v;
            for (int d = 0; d < head_dim; ++d) {
                k_dst[d] = static_cast<int8_t>(std::clamp(std::round(k_src[d] * inv_sk), -128.0f, 127.0f));
                v_dst[d] = static_cast<int8_t>(std::clamp(std::round(v_src[d] * inv_sv), -128.0f, 127.0f));
            }
        }
    } else if (kv_mode == "tl1_fused") {
        for (int h = 0; h < num_kv_heads; ++h) {
            size_t head_off = (static_cast<size_t>(layer_idx) * num_kv_heads + h) * max_seq_len * (head_dim / 4) + pos * (head_dim / 4);
            size_t scale_off = (static_cast<size_t>(layer_idx) * num_kv_heads + h) * max_seq_len + pos;

            const float* k_src = k_ptr + h * head_dim;
            const float* v_src = v_ptr + h * head_dim;
            uint8_t* k_dst = tl1_k.data() + head_off;
            uint8_t* v_dst = tl1_v.data() + head_off;

            float mean_k = 0.0f, mean_v = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                mean_k += std::abs(k_src[d]);
                mean_v += std::abs(v_src[d]);
            }
            mean_k /= static_cast<float>(head_dim);
            mean_v /= static_cast<float>(head_dim);

            float th_k = 0.5f * mean_k;
            float th_v = 0.5f * mean_v;

            float sum_k_nz = 0.0f; int cnt_k = 0;
            float sum_v_nz = 0.0f; int cnt_v = 0;
            for (int d = 0; d < head_dim; ++d) {
                float ak = std::abs(k_src[d]);
                if (ak > th_k) { sum_k_nz += ak; cnt_k++; }
                float av = std::abs(v_src[d]);
                if (av > th_v) { sum_v_nz += av; cnt_v++; }
            }
            float s_k = (cnt_k > 0) ? (sum_k_nz / static_cast<float>(cnt_k)) : (mean_k + 1e-6f);
            float s_v = (cnt_v > 0) ? (sum_v_nz / static_cast<float>(cnt_v)) : (mean_v + 1e-6f);

            tl1_k_scales[scale_off] = s_k;
            tl1_v_scales[scale_off] = s_v;

            for (int d = 0; d < head_dim; d += 4) {
                auto quant_tern = [](float val, float th) -> int {
                    if (val > th) return 1;
                    if (val < -th) return -1;
                    return 0;
                };
                int wk0 = quant_tern(k_src[d + 0], th_k);
                int wk1 = quant_tern(k_src[d + 1], th_k);
                int wk2 = quant_tern(k_src[d + 2], th_k);
                int wk3 = quant_tern(k_src[d + 3], th_k);
                k_dst[d / 4] = static_cast<uint8_t>((wk0 + 1) | ((wk1 + 1) << 2) | ((wk2 + 1) << 4) | ((wk3 + 1) << 6));

                int wv0 = quant_tern(v_src[d + 0], th_v);
                int wv1 = quant_tern(v_src[d + 1], th_v);
                int wv2 = quant_tern(v_src[d + 2], th_v);
                int wv3 = quant_tern(v_src[d + 3], th_v);
                v_dst[d / 4] = static_cast<uint8_t>((wv0 + 1) | ((wv1 + 1) << 2) | ((wv2 + 1) << 4) | ((wv3 + 1) << 6));
            }
        }
    } else if (kv_mode == "popcount_fused") {
        for (int h = 0; h < num_kv_heads; ++h) {
            size_t head_k_off = (static_cast<size_t>(layer_idx) * num_kv_heads + h) * max_seq_len + pos;
            size_t head_v_off = (static_cast<size_t>(layer_idx) * num_kv_heads + h) * max_seq_len * (head_dim / 4) + pos * (head_dim / 4);
            size_t scale_off = (static_cast<size_t>(layer_idx) * num_kv_heads + h) * max_seq_len + pos;

            const float* k_src = k_ptr + h * head_dim;
            const float* v_src = v_ptr + h * head_dim;

            float mean_k = 0.0f, mean_v = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                mean_k += std::abs(k_src[d]);
                mean_v += std::abs(v_src[d]);
            }
            mean_k /= static_cast<float>(head_dim);
            mean_v /= static_cast<float>(head_dim);

            float th_k = 0.5f * mean_k;
            float th_v = 0.5f * mean_v;

            float sum_k_nz = 0.0f; int cnt_k = 0;
            float sum_v_nz = 0.0f; int cnt_v = 0;
            for (int d = 0; d < head_dim; ++d) {
                float ak = std::abs(k_src[d]);
                if (ak > th_k) { sum_k_nz += ak; cnt_k++; }
                float av = std::abs(v_src[d]);
                if (av > th_v) { sum_v_nz += av; cnt_v++; }
            }
            float s_k = (cnt_k > 0) ? (sum_k_nz / static_cast<float>(cnt_k)) : (mean_k + 1e-6f);
            float s_v = (cnt_v > 0) ? (sum_v_nz / static_cast<float>(cnt_v)) : (mean_v + 1e-6f);

            popcount_k_scales[scale_off] = s_k;
            popcount_v_scales[scale_off] = s_v;

            // Pack K into positive and negative bitplanes (64 bits per uint64)
            uint64_t kp0 = 0, kn0 = 0, kp1 = 0, kn1 = 0;
            for (int d = 0; d < 64; ++d) {
                if (k_src[d] > th_k) kp0 |= (1ULL << d);
                else if (k_src[d] < -th_k) kn0 |= (1ULL << d);
            }
            for (int d = 0; d < 64; ++d) {
                if (k_src[64 + d] > th_k) kp1 |= (1ULL << d);
                else if (k_src[64 + d] < -th_k) kn1 |= (1ULL << d);
            }
            popcount_k[head_k_off] = {kp0, kp1, kn0, kn1};

            // Pack V into TL1 format (32 bytes)
            uint8_t* v_dst = popcount_v.data() + head_v_off;
            for (int d = 0; d < head_dim; d += 4) {
                auto quant_tern = [](float val, float th) -> int {
                    if (val > th) return 1;
                    if (val < -th) return -1;
                    return 0;
                };
                int wv0 = quant_tern(v_src[d + 0], th_v);
                int wv1 = quant_tern(v_src[d + 1], th_v);
                int wv2 = quant_tern(v_src[d + 2], th_v);
                int wv3 = quant_tern(v_src[d + 3], th_v);
                v_dst[d / 4] = static_cast<uint8_t>((wv0 + 1) | ((wv1 + 1) << 2) | ((wv2 + 1) << 4) | ((wv3 + 1) << 6));
            }
        }
    } else {
        for (int h = 0; h < num_kv_heads; ++h) {
            size_t head_off = (static_cast<size_t>(layer_idx) * num_kv_heads + h) * max_seq_len * head_dim + pos * head_dim;
            std::memcpy(fp32_k.data() + head_off, k_ptr + h * head_dim, head_dim * sizeof(float));
            std::memcpy(fp32_v.data() + head_off, v_ptr + h * head_dim, head_dim * sizeof(float));
        }
    }

    // 4. Scaled Dot-Product Attention
    const int seq_len = pos + 1;
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));

    #pragma omp parallel for schedule(dynamic, 2)
    for (int qh = 0; qh < num_q_heads; ++qh) {
        int kv_h = qh / gqa_ratio;
        const float* q_vec = q_ptr + qh * head_dim;
        float* out_vec = out_ptr + qh * head_dim;

        alignas(32) float scores_buf[8192];
        float* scores = (seq_len <= 8192) ? scores_buf : new float[seq_len];
        float max_score = -1e30f;

        if (kv_mode == "int8_fused") {
            size_t base_k_off = (static_cast<size_t>(layer_idx) * num_kv_heads + kv_h) * max_seq_len * head_dim;
            size_t base_s_off = (static_cast<size_t>(layer_idx) * num_kv_heads + kv_h) * max_seq_len;
            const int8_t* k_base = int8_k.data() + base_k_off;
            const float* s_base = int8_k_scales.data() + base_s_off;

            for (int t = 0; t < seq_len; ++t) {
                const int8_t* k_cached = k_base + t * head_dim;
                float s_k = s_base[t];

                __m256 dot_acc = _mm256_setzero_ps();
                for (int d = 0; d < head_dim; d += 8) {
                    __m128i k_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(k_cached + d));
                    __m256 k_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(k_bytes));
                    __m256 q_f32 = _mm256_loadu_ps(q_vec + d);
                    dot_acc = _mm256_fmadd_ps(q_f32, k_f32, dot_acc);
                }
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, dot_acc);
                float dot = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]) * (s_k * scale_factor);
                scores[t] = dot;
                if (dot > max_score) max_score = dot;
            }
        } else if (kv_mode == "tl1_fused") {
            alignas(16) static const int8_t LUT_W0[16] = { -1,  0,  1, 0,  -1,  0,  1, 0,  -1,  0,  1, 0,  -1,  0,  1, 0 };
            alignas(16) static const int8_t LUT_W1[16] = { -1, -1, -1, 0,   0,  0,  0, 0,   1,  1,  1, 0,   0,  0,  0, 0 };
            const __m128i lut0 = _mm_load_si128(reinterpret_cast<const __m128i*>(LUT_W0));
            const __m128i lut1 = _mm_load_si128(reinterpret_cast<const __m128i*>(LUT_W1));
            const __m128i mask_0f = _mm_set1_epi8(0x0F);

            // Pre-quantize Q to int16 for pure integer SIMD madd_epi16 dot-product
            alignas(32) int16_t q_i16[128];
            float max_q = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float a = std::abs(q_vec[d]);
                if (a > max_q) max_q = a;
            }
            float s_q = (max_q > 0.0f) ? (max_q / 127.0f) : 1.0f;
            float inv_s_q = 1.0f / s_q;
            for (int d = 0; d < head_dim; ++d) {
                q_i16[d] = static_cast<int16_t>(std::round(q_vec[d] * inv_s_q));
            }
            const __m256i* q_i16_vec = reinterpret_cast<const __m256i*>(q_i16);
            float q_eff_scale = s_q * scale_factor;

            size_t base_k_off = (static_cast<size_t>(layer_idx) * num_kv_heads + kv_h) * max_seq_len * (head_dim / 4);
            size_t base_s_off = (static_cast<size_t>(layer_idx) * num_kv_heads + kv_h) * max_seq_len;
            const uint8_t* k_base = tl1_k.data() + base_k_off;
            const float* s_base = tl1_k_scales.data() + base_s_off;

            for (int t = 0; t < seq_len; ++t) {
                const uint8_t* k_cached = k_base + t * (head_dim / 4);
                float s_k = s_base[t];

                __m256i dot_acc = _mm256_setzero_si256();

                for (int chunk = 0; chunk < 2; ++chunk) {
                    __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(k_cached + chunk * 16));
                    __m128i nib_lo = _mm_and_si128(raw16, mask_0f);
                    __m128i nib_hi = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask_0f);

                    __m128i w0 = _mm_shuffle_epi8(lut0, nib_lo);
                    __m128i w1 = _mm_shuffle_epi8(lut1, nib_lo);
                    __m128i w2 = _mm_shuffle_epi8(lut0, nib_hi);
                    __m128i w3 = _mm_shuffle_epi8(lut1, nib_hi);

                    __m128i w01_lo = _mm_unpacklo_epi8(w0, w1);
                    __m128i w01_hi = _mm_unpackhi_epi8(w0, w1);
                    __m128i w23_lo = _mm_unpacklo_epi8(w2, w3);
                    __m128i w23_hi = _mm_unpackhi_epi8(w2, w3);

                    __m128i w_0_15  = _mm_unpacklo_epi16(w01_lo, w23_lo);
                    __m128i w_16_31 = _mm_unpackhi_epi16(w01_lo, w23_lo);
                    __m128i w_32_47 = _mm_unpacklo_epi16(w01_hi, w23_hi);
                    __m128i w_48_63 = _mm_unpackhi_epi16(w01_hi, w23_hi);

                    int base_v = chunk * 4;
                    __m256i d0 = _mm256_madd_epi16(q_i16_vec[base_v + 0], _mm256_cvtepi8_epi16(w_0_15));
                    __m256i d1 = _mm256_madd_epi16(q_i16_vec[base_v + 1], _mm256_cvtepi8_epi16(w_16_31));
                    __m256i d2 = _mm256_madd_epi16(q_i16_vec[base_v + 2], _mm256_cvtepi8_epi16(w_32_47));
                    __m256i d3 = _mm256_madd_epi16(q_i16_vec[base_v + 3], _mm256_cvtepi8_epi16(w_48_63));

                    dot_acc = _mm256_add_epi32(dot_acc, _mm256_add_epi32(_mm256_add_epi32(d0, d1), _mm256_add_epi32(d2, d3)));
                }

                alignas(32) int32_t tmp[8];
                _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), dot_acc);
                int32_t isum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                float dot = static_cast<float>(isum) * (q_eff_scale * s_k);
                scores[t] = dot;
                if (dot > max_score) max_score = dot;
            }
        } else if (kv_mode == "popcount_fused") {
            // SIMD Popcount Attention: 0 FMA multiplications, pure bitwise AND + POPCNT!
            float mean_q = 0.0f;
            for (int d = 0; d < head_dim; ++d) mean_q += std::abs(q_vec[d]);
            mean_q /= static_cast<float>(head_dim);
            float th_q = 0.5f * mean_q;

            float sum_q_nz = 0.0f; int cnt_q = 0;
            for (int d = 0; d < head_dim; ++d) {
                float aq = std::abs(q_vec[d]);
                if (aq > th_q) { sum_q_nz += aq; cnt_q++; }
            }
            float s_q = (cnt_q > 0) ? (sum_q_nz / static_cast<float>(cnt_q)) : (mean_q + 1e-6f);
            float q_eff_scale = s_q * scale_factor;

            uint64_t q_pos0 = 0, q_neg0 = 0, q_pos1 = 0, q_neg1 = 0;
            for (int d = 0; d < 64; ++d) {
                if (q_vec[d] > th_q) q_pos0 |= (1ULL << d);
                else if (q_vec[d] < -th_q) q_neg0 |= (1ULL << d);
            }
            for (int d = 0; d < 64; ++d) {
                if (q_vec[64 + d] > th_q) q_pos1 |= (1ULL << d);
                else if (q_vec[64 + d] < -th_q) q_neg1 |= (1ULL << d);
            }

            size_t base_k_off = (static_cast<size_t>(layer_idx) * num_kv_heads + kv_h) * max_seq_len;
            size_t base_s_off = (static_cast<size_t>(layer_idx) * num_kv_heads + kv_h) * max_seq_len;
            const PopcountBitplanes* k_base = popcount_k.data() + base_k_off;
            const float* s_base = popcount_k_scales.data() + base_s_off;

            for (int t = 0; t < seq_len; ++t) {
                const PopcountBitplanes& bp = k_base[t];

                int pos_match = __builtin_popcountll(q_pos0 & bp.kp0) + __builtin_popcountll(q_pos1 & bp.kp1)
                              + __builtin_popcountll(q_neg0 & bp.kn0) + __builtin_popcountll(q_neg1 & bp.kn1);
                int neg_match = __builtin_popcountll(q_pos0 & bp.kn0) + __builtin_popcountll(q_pos1 & bp.kn1)
                              + __builtin_popcountll(q_neg0 & bp.kp0) + __builtin_popcountll(q_neg1 & bp.kp1);
                int dot_int = pos_match - neg_match;
                float dot = static_cast<float>(dot_int) * (q_eff_scale * s_base[t]);
                scores[t] = dot;
                if (dot > max_score) max_score = dot;
            }
        } else {
            size_t base_k_off = (static_cast<size_t>(layer_idx) * num_kv_heads + kv_h) * max_seq_len * head_dim;
            const float* k_base = fp32_k.data() + base_k_off;
            for (int t = 0; t < seq_len; ++t) {
                const float* k_cached = k_base + t * head_dim;
                __m256 dot_acc = _mm256_setzero_ps();
                for (int d = 0; d < head_dim; d += 8) {
                    dot_acc = _mm256_fmadd_ps(_mm256_loadu_ps(q_vec + d), _mm256_loadu_ps(k_cached + d), dot_acc);
                }
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, dot_acc);
                float dot = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]) * scale_factor;
                scores[t] = dot;
                if (dot > max_score) max_score = dot;
            }
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            scores[t] = std::exp(scores[t] - max_score);
            sum_exp += scores[t];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int t = 0; t < seq_len; ++t) scores[t] *= inv_sum;

        // Attention weighted sum V in 16 YMM registers
        __m256 acc[16];
        for (int i = 0; i < 16; ++i) acc[i] = _mm256_setzero_ps();

        if (kv_mode == "int8_fused") {
            size_t base_v_off = (static_cast<size_t>(layer_idx) * num_kv_heads + kv_h) * max_seq_len * head_dim;
            size_t base_sv_off = (static_cast<size_t>(layer_idx) * num_kv_heads + kv_h) * max_seq_len;
            const int8_t* v_base = int8_v.data() + base_v_off;
            const float* sv_base = int8_v_scales.data() + base_sv_off;

            for (int t = 0; t < seq_len; ++t) {
                const int8_t* v_cached = v_base + t * head_dim;
                __m256 weight_scale = _mm256_set1_ps(scores[t] * sv_base[t]);

                for (int i = 0; i < 16; ++i) {
                    __m128i v_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(v_cached + i * 8));
                    __m256 v_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_bytes));
                    acc[i] = _mm256_fmadd_ps(v_f32, weight_scale, acc[i]);
                }
            }
        } else if (kv_mode == "tl1_fused" || kv_mode == "popcount_fused") {
            alignas(16) static const int8_t LUT_W0[16] = { -1,  0,  1, 0,  -1,  0,  1, 0,  -1,  0,  1, 0,  -1,  0,  1, 0 };
            alignas(16) static const int8_t LUT_W1[16] = { -1, -1, -1, 0,   0,  0,  0, 0,   1,  1,  1, 0,   0,  0,  0, 0 };
            const __m128i lut0 = _mm_load_si128(reinterpret_cast<const __m128i*>(LUT_W0));
            const __m128i lut1 = _mm_load_si128(reinterpret_cast<const __m128i*>(LUT_W1));
            const __m128i mask_0f = _mm_set1_epi8(0x0F);

            size_t base_v_off = (static_cast<size_t>(layer_idx) * num_kv_heads + kv_h) * max_seq_len * (head_dim / 4);
            size_t base_sv_off = (static_cast<size_t>(layer_idx) * num_kv_heads + kv_h) * max_seq_len;
            const uint8_t* v_base = (kv_mode == "popcount_fused") ? (popcount_v.data() + base_v_off) : (tl1_v.data() + base_v_off);
            const float* sv_base = (kv_mode == "popcount_fused") ? (popcount_v_scales.data() + base_sv_off) : (tl1_v_scales.data() + base_sv_off);

            for (int t = 0; t < seq_len; ++t) {
                const uint8_t* v_cached = v_base + t * (head_dim / 4);
                __m256 weight_scale = _mm256_set1_ps(scores[t] * sv_base[t]);

                for (int chunk = 0; chunk < 2; ++chunk) {
                    __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(v_cached + chunk * 16));
                    __m128i nib_lo = _mm_and_si128(raw16, mask_0f);
                    __m128i nib_hi = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask_0f);

                    __m128i w0 = _mm_shuffle_epi8(lut0, nib_lo);
                    __m128i w1 = _mm_shuffle_epi8(lut1, nib_lo);
                    __m128i w2 = _mm_shuffle_epi8(lut0, nib_hi);
                    __m128i w3 = _mm_shuffle_epi8(lut1, nib_hi);

                    __m128i w01_lo = _mm_unpacklo_epi8(w0, w1);
                    __m128i w01_hi = _mm_unpackhi_epi8(w0, w1);
                    __m128i w23_lo = _mm_unpacklo_epi8(w2, w3);
                    __m128i w23_hi = _mm_unpackhi_epi8(w2, w3);

                    __m128i w_0_15  = _mm_unpacklo_epi16(w01_lo, w23_lo);
                    __m128i w_16_31 = _mm_unpackhi_epi16(w01_lo, w23_lo);
                    __m128i w_32_47 = _mm_unpacklo_epi16(w01_hi, w23_hi);
                    __m128i w_48_63 = _mm_unpackhi_epi16(w01_hi, w23_hi);

                    int base_i = chunk * 8;
                    acc[base_i + 0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_0_15)), weight_scale, acc[base_i + 0]);
                    acc[base_i + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w_0_15, 8))), weight_scale, acc[base_i + 1]);
                    acc[base_i + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_16_31)), weight_scale, acc[base_i + 2]);
                    acc[base_i + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w_16_31, 8))), weight_scale, acc[base_i + 3]);
                    acc[base_i + 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_32_47)), weight_scale, acc[base_i + 4]);
                    acc[base_i + 5] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w_32_47, 8))), weight_scale, acc[base_i + 5]);
                    acc[base_i + 6] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_48_63)), weight_scale, acc[base_i + 6]);
                    acc[base_i + 7] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w_48_63, 8))), weight_scale, acc[base_i + 7]);
                }
            }
        } else {
            size_t base_v_off = (static_cast<size_t>(layer_idx) * num_kv_heads + kv_h) * max_seq_len * head_dim;
            const float* v_base = fp32_v.data() + base_v_off;
            for (int t = 0; t < seq_len; ++t) {
                const float* v_cached = v_base + t * head_dim;
                __m256 weight_vec = _mm256_set1_ps(scores[t]);
                for (int i = 0; i < 16; ++i) {
                    acc[i] = _mm256_fmadd_ps(_mm256_loadu_ps(v_cached + i * 8), weight_vec, acc[i]);
                }
            }
        }

        for (int i = 0; i < 16; ++i) {
            _mm256_storeu_ps(out_vec + i * 8, acc[i]);
        }

        if (scores != scores_buf) {
            delete[] scores;
        }
    }
}

//===----------------------------------------------------------------------===//
// INT4 and INT3 AVX2 Matrix-Multiplication Micro-Kernels
//===----------------------------------------------------------------------===//

void compute_linear_int4(
    const float* act_f32,
    const uint8_t* w_int4,
    const float* scales,
    float* out_f32,
    int K, int N
) {
    alignas(32) std::vector<int16_t> act_i16(K);
    float amax = 1e-8f;
    for (int k = 0; k < K; ++k) amax = std::max(amax, std::abs(act_f32[k]));
    float act_scale = amax / 127.0f;
    float inv_act_scale = 1.0f / act_scale;
    for (int k = 0; k < K; ++k) act_i16[k] = static_cast<int16_t>(std::round(act_f32[k] * inv_act_scale));
    const __m256i* act_i16_vec = reinterpret_cast<const __m256i*>(act_i16.data());

    #pragma omp parallel for schedule(dynamic, 8)
    for (int n = 0; n < N; ++n) {
        const uint8_t* row_w = w_int4 + n * (K / 2);
        float s_w = scales[n];
        float eff_scale = s_w * act_scale;

        __m256i acc = _mm256_setzero_si256();
        for (int k = 0; k < K; k += 32) {
            __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_w + k / 2));
            __m128i lo_nib = _mm_and_si128(raw16, _mm_set1_epi8(0x0F));
            __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(raw16, 4), _mm_set1_epi8(0x0F));

            __m128i unpacked_lo = _mm_unpacklo_epi8(lo_nib, hi_nib);
            __m128i unpacked_hi = _mm_unpackhi_epi8(lo_nib, hi_nib);

            __m256i w_i16_0 = _mm256_cvtepi8_epi16(unpacked_lo);
            __m256i w_i16_1 = _mm256_cvtepi8_epi16(unpacked_hi);

            __m256i prod0 = _mm256_madd_epi16(act_i16_vec[(k + 0) / 16], w_i16_0);
            __m256i prod1 = _mm256_madd_epi16(act_i16_vec[(k + 16) / 16], w_i16_1);

            acc = _mm256_add_epi32(acc, _mm256_add_epi32(prod0, prod1));
        }
        alignas(32) int32_t tmp[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc);
        int32_t isum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        out_f32[n] = static_cast<float>(isum) * eff_scale;
    }
}

void compute_linear_int3(
    const float* act_f32,
    const uint8_t* w_int3,
    const float* scales,
    float* out_f32,
    int K, int N
) {
    alignas(32) std::vector<int16_t> act_i16(K);
    float amax = 1e-8f;
    for (int k = 0; k < K; ++k) amax = std::max(amax, std::abs(act_f32[k]));
    float act_scale = amax / 127.0f;
    float inv_act_scale = 1.0f / act_scale;
    for (int k = 0; k < K; ++k) act_i16[k] = static_cast<int16_t>(std::round(act_f32[k] * inv_act_scale));
    const __m256i* act_i16_vec = reinterpret_cast<const __m256i*>(act_i16.data());

    #pragma omp parallel for schedule(dynamic, 8)
    for (int n = 0; n < N; ++n) {
        const uint8_t* row_w = w_int3 + n * (3 * K / 8);
        float s_w = scales[n];
        float eff_scale = s_w * act_scale;

        int32_t total_sum = 0;
        for (int k = 0; k < K; k += 8) {
            int byte_idx = (k / 8) * 3;
            uint32_t b0 = row_w[byte_idx + 0];
            uint32_t b1 = row_w[byte_idx + 1];
            uint32_t b2 = row_w[byte_idx + 2];
            uint32_t packed24 = b0 | (b1 << 8) | (b2 << 16);

            int32_t w0 = (packed24 >> 0) & 0x07;
            int32_t w1 = (packed24 >> 3) & 0x07;
            int32_t w2 = (packed24 >> 6) & 0x07;
            int32_t w3 = (packed24 >> 9) & 0x07;
            int32_t w4 = (packed24 >> 12) & 0x07;
            int32_t w5 = (packed24 >> 15) & 0x07;
            int32_t w6 = (packed24 >> 18) & 0x07;
            int32_t w7 = (packed24 >> 21) & 0x07;

            total_sum += act_i16[k + 0] * (w0 - 4);
            total_sum += act_i16[k + 1] * (w1 - 4);
            total_sum += act_i16[k + 2] * (w2 - 4);
            total_sum += act_i16[k + 3] * (w3 - 4);
            total_sum += act_i16[k + 4] * (w4 - 4);
            total_sum += act_i16[k + 5] * (w5 - 4);
            total_sum += act_i16[k + 6] * (w6 - 4);
            total_sum += act_i16[k + 7] * (w7 - 4);
        }
        out_f32[n] = static_cast<float>(total_sum) * eff_scale;
    }
}

// GGUF Q4_0 Block: 32 weights per block. 2 bytes FP16 scale 'd' + 16 bytes packed nibbles
void compute_linear_gguf_q4_0(
    const float* act_f32,
    const uint8_t* w_gguf_q4_0,
    float* out_f32,
    int K, int N
) {
    const int block_size = 32;
    const int bytes_per_block = 18; // 2 bytes fp16 + 16 bytes nibbles
    int blocks_per_row = K / block_size;

    #pragma omp parallel for schedule(dynamic, 8)
    for (int n = 0; n < N; ++n) {
        const uint8_t* row_base = w_gguf_q4_0 + n * (blocks_per_row * bytes_per_block);
        __m256 sum_acc = _mm256_setzero_ps();

        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_base + b * bytes_per_block;
            uint16_t fp16_scale = *reinterpret_cast<const uint16_t*>(blk);
            __m128 d_f32_vec = _mm_cvtph_ps(_mm_cvtsi32_si128(fp16_scale));
            float d = _mm_cvtss_f32(d_f32_vec);
            __m256 d_vec = _mm256_set1_ps(d);

            const uint8_t* qs = blk + 2;
            __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
            __m128i lo = _mm_and_si128(raw16, _mm_set1_epi8(0x0F));
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw16, 4), _mm_set1_epi8(0x0F));

            __m128i offset = _mm_set1_epi8(8);
            lo = _mm_sub_epi8(lo, offset);
            hi = _mm_sub_epi8(hi, offset);

            const float* act_blk = act_f32 + b * block_size;
            __m256 w0_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
            __m256 w1_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
            __m256 w2_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
            __m256 w3_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));

            __m256 a0 = _mm256_loadu_ps(act_blk + 0);
            __m256 a1 = _mm256_loadu_ps(act_blk + 8);
            __m256 a2 = _mm256_loadu_ps(act_blk + 16);
            __m256 a3 = _mm256_loadu_ps(act_blk + 24);

            __m256 blk_dot = _mm256_fmadd_ps(a0, w0_f32, _mm256_setzero_ps());
            blk_dot = _mm256_fmadd_ps(a1, w1_f32, blk_dot);
            blk_dot = _mm256_fmadd_ps(a2, w2_f32, blk_dot);
            blk_dot = _mm256_fmadd_ps(a3, w3_f32, blk_dot);

            sum_acc = _mm256_fmadd_ps(blk_dot, d_vec, sum_acc);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, sum_acc);
        out_f32[n] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    }
}

// GGUF Q8_0 Block: 32 weights per block. 2 bytes FP16 scale 'd' + 32 bytes int8
void compute_linear_gguf_q8_0(
    const float* act_f32,
    const uint8_t* w_gguf_q8_0,
    float* out_f32,
    int K, int N
) {
    const int block_size = 32;
    const int bytes_per_block = 34;
    int blocks_per_row = K / block_size;

    #pragma omp parallel for schedule(dynamic, 8)
    for (int n = 0; n < N; ++n) {
        const uint8_t* row_base = w_gguf_q8_0 + n * (blocks_per_row * bytes_per_block);
        __m256 sum_acc = _mm256_setzero_ps();

        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_base + b * bytes_per_block;
            uint16_t fp16_scale = *reinterpret_cast<const uint16_t*>(blk);
            __m128 d_f32_vec = _mm_cvtph_ps(_mm_cvtsi32_si128(fp16_scale));
            float d = _mm_cvtss_f32(d_f32_vec);
            __m256 d_vec = _mm256_set1_ps(d);

            const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
            const float* act_blk = act_f32 + b * block_size;

            for (int i = 0; i < 4; ++i) {
                __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + i * 8));
                __m256 w_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8));
                __m256 a = _mm256_loadu_ps(act_blk + i * 8);
                sum_acc = _mm256_fmadd_ps(_mm256_mul_ps(a, w_f32), d_vec, sum_acc);
            }
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, sum_acc);
        out_f32[n] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    }
}

// GPTQ 4-bit Groupwise GEMV
void compute_linear_gptq(
    const float* act_f32,
    const uint32_t* qweight,
    const float* scales,
    const uint32_t* qzeros,
    float* out_f32,
    int K, int N, int group_size = 128
) {
    #pragma omp parallel for schedule(dynamic, 8)
    for (int n = 0; n < N; ++n) {
        float total_sum = 0.0f;
        for (int k = 0; k < K; k += 8) {
            int group_idx = k / group_size;
            float s = scales[group_idx * N + n];
            uint32_t z_pack = qzeros[(group_idx / 8) * N + n];
            int z = ((z_pack >> ((group_idx % 8) * 4)) & 0x0F) + 1;

            uint32_t qw = qweight[(k / 8) * N + n];
            for (int elem = 0; elem < 8; ++elem) {
                int w = (qw >> (elem * 4)) & 0x0F;
                float w_deq = static_cast<float>(w - z) * s;
                total_sum += act_f32[k + elem] * w_deq;
            }
        }
        out_f32[n] = total_sum;
    }
}

// AWQ 4-bit Column-Interleaved GEMV
void compute_linear_awq(
    const float* act_f32,
    const uint32_t* qweight,
    const float* scales,
    const uint32_t* qzeros,
    float* out_f32,
    int K, int N, int group_size = 128
) {
    static const int awq_order[8] = {0, 2, 4, 6, 1, 3, 5, 7};

    #pragma omp parallel for schedule(dynamic, 8)
    for (int n = 0; n < N; ++n) {
        float total_sum = 0.0f;
        for (int k = 0; k < K; k += 8) {
            int group_idx = k / group_size;
            float s = scales[group_idx * N + n];
            uint32_t z_pack = qzeros[group_idx * (N / 8) + (n / 8)];
            int z = (z_pack >> ((n % 8) * 4)) & 0x0F;

            uint32_t qw = qweight[(k / 8) * N + n];
            for (int elem = 0; elem < 8; ++elem) {
                int orig_idx = awq_order[elem];
                int w = (qw >> (elem * 4)) & 0x0F;
                float w_deq = static_cast<float>(w - z) * s;
                total_sum += act_f32[k + orig_idx] * w_deq;
            }
        }
        out_f32[n] = total_sum;
    }
}

// EXL2 Variable Bitrate GEMV
void compute_linear_exl2(
    const float* act_f32,
    const uint32_t* qweight,
    const float* scales,
    const uint16_t* qgroups,
    float* out_f32,
    int K, int N, int num_groups
) {
    #pragma omp parallel for schedule(dynamic, 8)
    for (int n = 0; n < N; ++n) {
        float total_sum = 0.0f;
        for (int g = 0; g < num_groups; ++g) {
            int bits = qgroups[g * 4 + 0];
            int k_start = qgroups[g * 4 + 1];
            int k_len = qgroups[g * 4 + 2];
            float s = scales[g * N + n];

            uint32_t mask = (1U << bits) - 1;
            int mid = 1 << (bits - 1);

            for (int k = 0; k < k_len; ++k) {
                int bit_pos = (k_start + k) * bits;
                int word_idx = (bit_pos / 32) * N + n;
                int shift = bit_pos % 32;
                uint32_t raw_w = (qweight[word_idx] >> shift) & mask;
                float w_deq = static_cast<float>(static_cast<int>(raw_w) - mid) * s;
                total_sum += act_f32[k_start + k] * w_deq;
            }
        }
        out_f32[n] = total_sum;
    }
}

//===----------------------------------------------------------------------===//
// TenzoEngineImpl Implementation
//===----------------------------------------------------------------------===//

TenzoEngineImpl::TenzoEngineImpl(const tenzo_config_t& cfg)
    : config(cfg),
      kv_cache(cfg.num_layers, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len, cfg.kv_mode ? cfg.kv_mode : "int8_fused"),
      paged_kv_cache(cfg.num_layers, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, (cfg.max_seq_len + 15) / 16) {
    
    use_paged_kv = (cfg.kv_mode && std::string(cfg.kv_mode) == "paged_int8");
    q_dim = cfg.num_q_heads * cfg.head_dim;
    kv_dim = cfg.num_kv_heads * cfg.head_dim;
    ffn_dim = cfg.ffn_dim;

    layers.resize(cfg.num_layers);

    int max_dim = 16384;
    buf_x.resize(max_dim);
    buf_norm_x.resize(max_dim);
    buf_q.resize(max_dim);
    buf_k.resize(max_dim);
    buf_v.resize(max_dim);
    buf_attn_out.resize(max_dim);
    buf_attn_sub.resize(max_dim);
    buf_out.resize(max_dim);
    buf_h1.resize(max_dim);
    buf_post_norm.resize(max_dim);
    buf_gate.resize(max_dim);
    buf_up.resize(max_dim);
    buf_act.resize(max_dim);
    buf_ffn_norm.resize(max_dim);
    buf_down.resize(max_dim);
    lut_buf.resize(max_dim / 2);
    logits_scratch.resize(cfg.vocab_size);
    buf_x_i8.resize(max_dim);
    buf_x_expanded.resize(max_dim / 16);
}

void TenzoEngineImpl::reset() {
    if (use_paged_kv) {
        paged_kv_cache.reset();
    } else {
        kv_cache.reset();
    }
}

void TenzoEngineImpl::set_layer_weights(int layer_idx, const tenzo_layer_weights_t* w) {
    if (layer_idx < 0 || layer_idx >= config.num_layers || !w) return;
    auto& lw = layers[layer_idx];

    int k_half = config.hidden_size / 2;
    int f_half = ffn_dim / 2;

    // Direct pointers (already packed in TL1)
    lw.q_w_ptr = w->q_w; lw.q_scale = w->q_scale;
    lw.k_w_ptr = w->k_w; lw.k_scale = w->k_scale;
    lw.v_w_ptr = w->v_w; lw.v_scale = w->v_scale;
    lw.out_w_ptr = w->out_w; lw.out_scale = w->out_scale;
    lw.gate_w_ptr = w->gate_w; lw.gate_scale = w->gate_scale;
    lw.up_w_ptr = w->up_w; lw.up_scale = w->up_scale;
    lw.down_w_ptr = w->down_w; lw.down_scale = w->down_scale;

    lw.in_norm_ptr = w->in_norm;
    lw.attn_sub_norm_ptr = w->attn_sub_norm;
    lw.post_norm_ptr = w->post_norm;
    lw.ffn_sub_norm_ptr = w->ffn_sub_norm;
}

void TenzoEngineImpl::set_final_norm(const float* norm_w) {
    final_norm_ptr = norm_w;
}

void TenzoEngineImpl::set_lm_head_f32(const float* embed_w, int vocab_size, int hidden_size) {
    is_lm_i8 = false;
    embed_w_f32.assign(embed_w, embed_w + static_cast<size_t>(vocab_size) * hidden_size);
}

void TenzoEngineImpl::set_lm_head_i8(const int8_t* embed_w, const float* scales, int vocab_size, int hidden_size) {
    is_lm_i8 = true;
    embed_w_i8.assign(embed_w, embed_w + static_cast<size_t>(vocab_size) * hidden_size);
    embed_scales.assign(scales, scales + vocab_size);
}

void TenzoEngineImpl::embedding_lookup(int token_id) {
    const int h_dim = config.hidden_size;
    if (!is_lm_i8) {
        const float* row = embed_w_f32.data() + token_id * h_dim;
        std::memcpy(buf_x.data(), row, h_dim * sizeof(float));
    } else {
        const int8_t* row = embed_w_i8.data() + token_id * h_dim;
        float scale = embed_scales[token_id];
        __m256 s_vec = _mm256_set1_ps(scale);
        int i = 0;
        for (; i + 16 <= h_dim; i += 16) {
            __m128i r_lo = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + i));
            __m128i r_hi = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + i + 8));
            __m256 f_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r_lo));
            __m256 f_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r_hi));
            _mm256_storeu_ps(buf_x.data() + i, _mm256_mul_ps(f_lo, s_vec));
            _mm256_storeu_ps(buf_x.data() + i + 8, _mm256_mul_ps(f_hi, s_vec));
        }
        for (; i < h_dim; ++i) {
            buf_x[i] = static_cast<float>(row[i]) * scale;
        }
    }
}

void TenzoEngineImpl::forward_layer_gpu(int layer_idx) {
    const auto& lw = layers[layer_idx];
    const int h_dim = config.hidden_size;

    // 1. GPU Pre-RMSNorm
    runtime::VulkanRuntime::executeRMSNorm(buf_x.data(), lw.in_norm_ptr, buf_norm_x.data(), h_dim);

    // 2. GPU BitLinear Projections Q, K, V
    runtime::VulkanRuntime::executeBitLinearTL1(buf_norm_x.data(), reinterpret_cast<const uint8_t*>(lw.q_w_ptr), buf_q.data(), q_dim, h_dim, lw.q_scale);
    runtime::VulkanRuntime::executeBitLinearTL1(buf_norm_x.data(), reinterpret_cast<const uint8_t*>(lw.k_w_ptr), buf_k.data(), kv_dim, h_dim, lw.k_scale);
    runtime::VulkanRuntime::executeBitLinearTL1(buf_norm_x.data(), reinterpret_cast<const uint8_t*>(lw.v_w_ptr), buf_v.data(), kv_dim, h_dim, lw.v_scale);

    // 3. Attention
    if (use_paged_kv) {
        paged_kv_cache.forward_attention_raw(layer_idx, buf_q.data(), buf_k.data(), buf_v.data(), buf_attn_out.data());
    } else {
        kv_cache.forward_attention_raw(layer_idx, buf_q.data(), buf_k.data(), buf_v.data(), buf_attn_out.data());
    }

    // 4. GPU Attn Sub-Norm
    runtime::VulkanRuntime::executeRMSNorm(buf_attn_out.data(), lw.attn_sub_norm_ptr, buf_attn_sub.data(), h_dim);

    // 5. GPU Out Projection
    runtime::VulkanRuntime::executeBitLinearTL1(buf_attn_sub.data(), reinterpret_cast<const uint8_t*>(lw.out_w_ptr), buf_out.data(), h_dim, h_dim, lw.out_scale);

    // 6. Residual 1
    #pragma omp simd
    for (int i = 0; i < h_dim; ++i) buf_h1[i] = buf_x[i] + buf_out[i];

    // 7. GPU MLP Block
    if (lw.gate_w_ptr && lw.up_w_ptr && lw.down_w_ptr) {
        runtime::VulkanRuntime::executeRMSNorm(buf_h1.data(), lw.post_norm_ptr, buf_post_norm.data(), h_dim);
        runtime::VulkanRuntime::executeBitLinearTL1(buf_post_norm.data(), reinterpret_cast<const uint8_t*>(lw.gate_w_ptr), buf_gate.data(), ffn_dim, h_dim, lw.gate_scale);
        runtime::VulkanRuntime::executeBitLinearTL1(buf_post_norm.data(), reinterpret_cast<const uint8_t*>(lw.up_w_ptr), buf_up.data(), ffn_dim, h_dim, lw.up_scale);

        #pragma omp simd
        for (int i = 0; i < ffn_dim; ++i) {
            float r = std::max(0.0f, buf_gate[i]);
            buf_act[i] = (r * r) * buf_up[i];
        }

        runtime::VulkanRuntime::executeRMSNorm(buf_act.data(), lw.ffn_sub_norm_ptr, buf_ffn_norm.data(), ffn_dim);
        runtime::VulkanRuntime::executeBitLinearTL1(buf_ffn_norm.data(), reinterpret_cast<const uint8_t*>(lw.down_w_ptr), buf_down.data(), h_dim, ffn_dim, lw.down_scale);

        #pragma omp simd
        for (int i = 0; i < h_dim; ++i) buf_x[i] = buf_h1[i] + buf_down[i];
    } else {
        std::memcpy(buf_x.data(), buf_h1.data(), h_dim * sizeof(float));
    }
}

void TenzoEngineImpl::forward_layer_cuda(int layer_idx) {
    const auto& lw = layers[layer_idx];
    const int h_dim = config.hidden_size;

    // 1. CUDA Pre-RMSNorm
    CUDARuntime::getInstance().executeRMSNorm(buf_x.data(), lw.in_norm_ptr, buf_norm_x.data(), h_dim);

    // 2. CUDA BitLinear Projections Q, K, V
    CUDARuntime::getInstance().executeBitLinearTL1(buf_norm_x.data(), reinterpret_cast<const uint8_t*>(lw.q_w_ptr), buf_q.data(), 1, h_dim, q_dim, lw.q_scale);
    CUDARuntime::getInstance().executeBitLinearTL1(buf_norm_x.data(), reinterpret_cast<const uint8_t*>(lw.k_w_ptr), buf_k.data(), 1, h_dim, kv_dim, lw.k_scale);
    CUDARuntime::getInstance().executeBitLinearTL1(buf_norm_x.data(), reinterpret_cast<const uint8_t*>(lw.v_w_ptr), buf_v.data(), 1, h_dim, kv_dim, lw.v_scale);

    // 3. Attention
    if (use_paged_kv) {
        paged_kv_cache.forward_attention_raw(layer_idx, buf_q.data(), buf_k.data(), buf_v.data(), buf_attn_out.data());
    } else {
        kv_cache.forward_attention_raw(layer_idx, buf_q.data(), buf_k.data(), buf_v.data(), buf_attn_out.data());
    }

    // 4. CUDA Attn Sub-Norm
    CUDARuntime::getInstance().executeRMSNorm(buf_attn_out.data(), lw.attn_sub_norm_ptr, buf_attn_sub.data(), h_dim);

    // 5. CUDA Out Projection
    CUDARuntime::getInstance().executeBitLinearTL1(buf_attn_sub.data(), reinterpret_cast<const uint8_t*>(lw.out_w_ptr), buf_out.data(), 1, h_dim, h_dim, lw.out_scale);

    // 6. Residual 1
    #pragma omp simd
    for (int i = 0; i < h_dim; ++i) buf_h1[i] = buf_x[i] + buf_out[i];

    // 7. CUDA MLP Block
    if (lw.gate_w_ptr && lw.up_w_ptr && lw.down_w_ptr) {
        CUDARuntime::getInstance().executeRMSNorm(buf_h1.data(), lw.post_norm_ptr, buf_post_norm.data(), h_dim);
        CUDARuntime::getInstance().executeBitLinearTL1(buf_post_norm.data(), reinterpret_cast<const uint8_t*>(lw.gate_w_ptr), buf_gate.data(), 1, h_dim, ffn_dim, lw.gate_scale);
        CUDARuntime::getInstance().executeBitLinearTL1(buf_post_norm.data(), reinterpret_cast<const uint8_t*>(lw.up_w_ptr), buf_up.data(), 1, h_dim, ffn_dim, lw.up_scale);

        #pragma omp simd
        for (int i = 0; i < ffn_dim; ++i) {
            float r = std::max(0.0f, buf_gate[i]);
            buf_act[i] = (r * r) * buf_up[i];
        }

        CUDARuntime::getInstance().executeRMSNorm(buf_act.data(), lw.ffn_sub_norm_ptr, buf_ffn_norm.data(), ffn_dim);
        CUDARuntime::getInstance().executeBitLinearTL1(buf_ffn_norm.data(), reinterpret_cast<const uint8_t*>(lw.down_w_ptr), buf_down.data(), 1, ffn_dim, h_dim, lw.down_scale);

        #pragma omp simd
        for (int i = 0; i < h_dim; ++i) buf_x[i] = buf_h1[i] + buf_down[i];
    } else {
        std::memcpy(buf_x.data(), buf_h1.data(), h_dim * sizeof(float));
    }
}

void TenzoEngineImpl::forward_layer_rocm(int layer_idx) {
    const auto& lw = layers[layer_idx];
    const int h_dim = config.hidden_size;

    // 1. ROCm Pre-RMSNorm
    ROCmRuntime::getInstance().executeRMSNorm(buf_x.data(), lw.in_norm_ptr, buf_norm_x.data(), h_dim);

    // 2. ROCm BitLinear Projections Q, K, V
    ROCmRuntime::getInstance().executeBitLinearTL1(buf_norm_x.data(), reinterpret_cast<const uint8_t*>(lw.q_w_ptr), buf_q.data(), 1, h_dim, q_dim, lw.q_scale);
    ROCmRuntime::getInstance().executeBitLinearTL1(buf_norm_x.data(), reinterpret_cast<const uint8_t*>(lw.k_w_ptr), buf_k.data(), 1, h_dim, kv_dim, lw.k_scale);
    ROCmRuntime::getInstance().executeBitLinearTL1(buf_norm_x.data(), reinterpret_cast<const uint8_t*>(lw.v_w_ptr), buf_v.data(), 1, h_dim, kv_dim, lw.v_scale);

    // 3. Attention
    if (use_paged_kv) {
        paged_kv_cache.forward_attention_raw(layer_idx, buf_q.data(), buf_k.data(), buf_v.data(), buf_attn_out.data());
    } else {
        kv_cache.forward_attention_raw(layer_idx, buf_q.data(), buf_k.data(), buf_v.data(), buf_attn_out.data());
    }

    // 4. ROCm Attn Sub-Norm
    ROCmRuntime::getInstance().executeRMSNorm(buf_attn_out.data(), lw.attn_sub_norm_ptr, buf_attn_sub.data(), h_dim);

    // 5. ROCm Out Projection
    ROCmRuntime::getInstance().executeBitLinearTL1(buf_attn_sub.data(), reinterpret_cast<const uint8_t*>(lw.out_w_ptr), buf_out.data(), 1, h_dim, h_dim, lw.out_scale);

    // 6. Residual 1
    #pragma omp simd
    for (int i = 0; i < h_dim; ++i) buf_h1[i] = buf_x[i] + buf_out[i];

    // 7. ROCm MLP Block
    if (lw.gate_w_ptr && lw.up_w_ptr && lw.down_w_ptr) {
        ROCmRuntime::getInstance().executeRMSNorm(buf_h1.data(), lw.post_norm_ptr, buf_post_norm.data(), h_dim);
        ROCmRuntime::getInstance().executeBitLinearTL1(buf_post_norm.data(), reinterpret_cast<const uint8_t*>(lw.gate_w_ptr), buf_gate.data(), 1, h_dim, ffn_dim, lw.gate_scale);
        ROCmRuntime::getInstance().executeBitLinearTL1(buf_post_norm.data(), reinterpret_cast<const uint8_t*>(lw.up_w_ptr), buf_up.data(), 1, h_dim, ffn_dim, lw.up_scale);

        #pragma omp simd
        for (int i = 0; i < ffn_dim; ++i) {
            float r = std::max(0.0f, buf_gate[i]);
            buf_act[i] = (r * r) * buf_up[i];
        }

        ROCmRuntime::getInstance().executeRMSNorm(buf_act.data(), lw.ffn_sub_norm_ptr, buf_ffn_norm.data(), ffn_dim);
        ROCmRuntime::getInstance().executeBitLinearTL1(buf_ffn_norm.data(), reinterpret_cast<const uint8_t*>(lw.down_w_ptr), buf_down.data(), 1, ffn_dim, h_dim, lw.down_scale);

        #pragma omp simd
        for (int i = 0; i < h_dim; ++i) buf_x[i] = buf_h1[i] + buf_down[i];
    } else {
        std::memcpy(buf_x.data(), buf_h1.data(), h_dim * sizeof(float));
    }
}

void TenzoEngineImpl::forward_layer_riscv(int layer_idx) {
    const auto& lw = layers[layer_idx];
    const int h_dim = config.hidden_size;

    // 1. RISC-V RVV Pre-RMSNorm
    rvv::rms_norm_rvv(buf_x.data(), lw.in_norm_ptr, buf_norm_x.data(), h_dim);

    // 2. RISC-V RVV BitLinear Projections Q, K, V
    rvv::gemv_bitlinear_tl1_rvv(buf_norm_x.data(), reinterpret_cast<const uint8_t*>(lw.q_w_ptr), buf_q.data(), 1, h_dim, q_dim, lw.q_scale);
    rvv::gemv_bitlinear_tl1_rvv(buf_norm_x.data(), reinterpret_cast<const uint8_t*>(lw.k_w_ptr), buf_k.data(), 1, h_dim, kv_dim, lw.k_scale);
    rvv::gemv_bitlinear_tl1_rvv(buf_norm_x.data(), reinterpret_cast<const uint8_t*>(lw.v_w_ptr), buf_v.data(), 1, h_dim, kv_dim, lw.v_scale);

    // 3. Attention
    if (use_paged_kv) {
        paged_kv_cache.forward_attention_raw(layer_idx, buf_q.data(), buf_k.data(), buf_v.data(), buf_attn_out.data());
    } else {
        kv_cache.forward_attention_raw(layer_idx, buf_q.data(), buf_k.data(), buf_v.data(), buf_attn_out.data());
    }

    // 4. RISC-V RVV Attn Sub-Norm
    rvv::rms_norm_rvv(buf_attn_out.data(), lw.attn_sub_norm_ptr, buf_attn_sub.data(), h_dim);

    // 5. RISC-V RVV Out Projection
    rvv::gemv_bitlinear_tl1_rvv(buf_attn_sub.data(), reinterpret_cast<const uint8_t*>(lw.out_w_ptr), buf_out.data(), 1, h_dim, h_dim, lw.out_scale);

    // 6. Residual 1
    #pragma omp simd
    for (int i = 0; i < h_dim; ++i) buf_h1[i] = buf_x[i] + buf_out[i];

    // 7. RISC-V RVV MLP Block
    if (lw.gate_w_ptr && lw.up_w_ptr && lw.down_w_ptr) {
        rvv::rms_norm_rvv(buf_h1.data(), lw.post_norm_ptr, buf_post_norm.data(), h_dim);
        rvv::gemv_bitlinear_tl1_rvv(buf_post_norm.data(), reinterpret_cast<const uint8_t*>(lw.gate_w_ptr), buf_gate.data(), 1, h_dim, ffn_dim, lw.gate_scale);
        rvv::gemv_bitlinear_tl1_rvv(buf_post_norm.data(), reinterpret_cast<const uint8_t*>(lw.up_w_ptr), buf_up.data(), 1, h_dim, ffn_dim, lw.up_scale);

        #pragma omp simd
        for (int i = 0; i < ffn_dim; ++i) {
            float r = std::max(0.0f, buf_gate[i]);
            buf_act[i] = (r * r) * buf_up[i];
        }

        rvv::rms_norm_rvv(buf_act.data(), lw.ffn_sub_norm_ptr, buf_ffn_norm.data(), ffn_dim);
        rvv::gemv_bitlinear_tl1_rvv(buf_ffn_norm.data(), reinterpret_cast<const uint8_t*>(lw.down_w_ptr), buf_down.data(), 1, ffn_dim, h_dim, lw.down_scale);

        #pragma omp simd
        for (int i = 0; i < h_dim; ++i) buf_x[i] = buf_h1[i] + buf_down[i];
    } else {
        std::memcpy(buf_x.data(), buf_h1.data(), h_dim * sizeof(float));
    }
}

void TenzoEngineImpl::forward_layer_raw(int layer_idx) {
    std::string dev = config.device ? config.device : "cpu";
    if (dev == "gpu" || dev == "vulkan") {
        forward_layer_gpu(layer_idx);
        return;
    } else if (dev == "cuda") {
        forward_layer_cuda(layer_idx);
        return;
    } else if (dev == "rocm") {
        forward_layer_rocm(layer_idx);
        return;
    } else if (dev == "riscv") {
        forward_layer_riscv(layer_idx);
        return;
    }

    const auto& lw = layers[layer_idx];
    const int h_dim = config.hidden_size;

    // 1. Pre-RMSNorm
    rms_norm_raw(buf_x.data(), lw.in_norm_ptr, buf_norm_x.data(), h_dim);

    // 2. Build LUT for norm_x ONCE (reused across Q, K, V)
    float inv_scale_act = build_lut(buf_norm_x.data(), h_dim, lut_ptr());

    // 3. Fused Q, K, V parallel loop (60 blocks)
    int q_blocks = q_dim / 64;   // 40
    int kv_blocks = kv_dim / 64; // 10
    int total_qkv = q_blocks + 2 * kv_blocks; // 60
    int k_half = h_dim / 2;
    __m256i mask_low = _mm256_set1_epi8(0x0F);

    __m256 s_q = _mm256_set1_ps(inv_scale_act * lw.q_scale);
    __m256 s_k = _mm256_set1_ps(inv_scale_act * lw.k_scale);
    __m256 s_v = _mm256_set1_ps(inv_scale_act * lw.v_scale);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int b = 0; b < total_qkv; ++b) {
        if (b < q_blocks) {
            const int8_t* bw = lw.q_w_ptr + b * k_half * 32;
            compute_single_block(lut_ptr(), bw, k_half, buf_q.data() + b * 64, s_q, mask_low);
        } else if (b < q_blocks + kv_blocks) {
            int kb = b - q_blocks;
            const int8_t* bw = lw.k_w_ptr + kb * k_half * 32;
            compute_single_block(lut_ptr(), bw, k_half, buf_k.data() + kb * 64, s_k, mask_low);
        } else {
            int vb = b - (q_blocks + kv_blocks);
            const int8_t* bw = lw.v_w_ptr + vb * k_half * 32;
            compute_single_block(lut_ptr(), bw, k_half, buf_v.data() + vb * 64, s_v, mask_low);
        }
    }

    // 4. In-place Zero-Copy Attention with RoPE and Compressed/Paged KV-Cache
    if (use_paged_kv) {
        paged_kv_cache.forward_attention_raw(layer_idx, buf_q.data(), buf_k.data(), buf_v.data(), buf_attn_out.data());
    } else {
        kv_cache.forward_attention_raw(layer_idx, buf_q.data(), buf_k.data(), buf_v.data(), buf_attn_out.data());
    }

    // 5. Attn Sub-Norm
    rms_norm_raw(buf_attn_out.data(), lw.attn_sub_norm_ptr, buf_attn_sub.data(), h_dim);

    // 6. Out Projection
    float inv_s_out = build_lut(buf_attn_sub.data(), h_dim, lut_ptr());
    compute_from_lut(lut_ptr(), lw.out_w_ptr, h_dim / 64, k_half, buf_out.data(), inv_s_out * lw.out_scale);

    // 7. Residual 1
    #pragma omp simd
    for (int i = 0; i < h_dim; ++i) buf_h1[i] = buf_x[i] + buf_out[i];

    // 8. Fused MLP Block
    if (lw.gate_w_ptr && lw.up_w_ptr && lw.down_w_ptr) {
        rms_norm_raw(buf_h1.data(), lw.post_norm_ptr, buf_post_norm.data(), h_dim);

        float inv_s_mlp = build_lut(buf_post_norm.data(), h_dim, lut_ptr());
        int cur_ffn = ffn_dim;
        int ffn_blocks = cur_ffn / 64; // 108
        int total_mlp = 2 * ffn_blocks; // 216

        __m256 s_gate = _mm256_set1_ps(inv_s_mlp * lw.gate_scale);
        __m256 s_up = _mm256_set1_ps(inv_s_mlp * lw.up_scale);

        #pragma omp parallel for schedule(dynamic, 8)
        for (int b = 0; b < total_mlp; ++b) {
            if (b < ffn_blocks) {
                const int8_t* bw = lw.gate_w_ptr + b * k_half * 32;
                compute_single_block(lut_ptr(), bw, k_half, buf_gate.data() + b * 64, s_gate, mask_low);
            } else {
                int ub = b - ffn_blocks;
                const int8_t* bw = lw.up_w_ptr + ub * k_half * 32;
                compute_single_block(lut_ptr(), bw, k_half, buf_up.data() + ub * 64, s_up, mask_low);
            }
        }

        #pragma omp simd
        for (int i = 0; i < cur_ffn; ++i) {
            float r = std::max(0.0f, buf_gate[i]);
            buf_act[i] = (r * r) * buf_up[i];
        }

        rms_norm_raw(buf_act.data(), lw.ffn_sub_norm_ptr, buf_ffn_norm.data(), cur_ffn);

        float inv_s_down = build_lut(buf_ffn_norm.data(), cur_ffn, lut_ptr());
        compute_from_lut(lut_ptr(), lw.down_w_ptr, h_dim / 64, cur_ffn / 2, buf_down.data(), inv_s_down * lw.down_scale);

        #pragma omp simd
        for (int i = 0; i < h_dim; ++i) buf_x[i] = buf_h1[i] + buf_down[i];
    } else {
        std::memcpy(buf_x.data(), buf_h1.data(), h_dim * sizeof(float));
    }
}

void TenzoEngineImpl::compute_logits() {
    const int h_dim = config.hidden_size;
    const int vocab_size = config.vocab_size;

    rms_norm_raw(buf_x.data(), final_norm_ptr, buf_norm_x.data(), h_dim);

    if (!is_lm_i8) {
        const float* x_ptr = buf_norm_x.data();
        const float* w_ptr = embed_w_f32.data();

        #pragma omp parallel for schedule(dynamic, 1024)
        for (int v = 0; v < vocab_size; ++v) {
            const float* row = w_ptr + v * h_dim;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            int i = 0;
            for (; i + 16 <= h_dim; i += 16) {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + i), _mm256_loadu_ps(row + i), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + i + 8), _mm256_loadu_ps(row + i + 8), acc1);
            }
            __m256 acc = _mm256_add_ps(acc0, acc1);
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, acc);
            float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
            for (; i < h_dim; ++i) sum += x_ptr[i] * row[i];
            logits_scratch[v] = sum;
        }
    } else {
        const float* x_ptr = buf_norm_x.data();
        const int8_t* w_ptr = embed_w_i8.data();
        const float* scales_ptr = embed_scales.data();

        float amax_x = 1e-8f;
        for (int i = 0; i < h_dim; ++i) {
            float val = std::abs(x_ptr[i]);
            if (val > amax_x) amax_x = val;
        }
        float s_x = amax_x / 127.0f;
        float inv_s_x = 1.0f / s_x;
        int8_t* x_i8_ptr = buf_x_i8.data();
        #pragma omp simd
        for (int i = 0; i < h_dim; ++i) {
            float q = std::round(x_ptr[i] * inv_s_x);
            x_i8_ptr[i] = static_cast<int8_t>(std::clamp(q, -128.0f, 127.0f));
        }

        int num_vecs = h_dim / 16;
        Vec256i* x_exp_arr = buf_x_expanded.data();
        for (int j = 0; j < num_vecs; ++j) {
            __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x_i8_ptr + j * 16));
            x_exp_arr[j] = _mm256_cvtepi8_epi16(a);
        }
        const __m256i* x_exp_ptr = reinterpret_cast<const __m256i*>(x_exp_arr);

        #pragma omp parallel for schedule(dynamic, 1024)
        for (int v = 0; v < vocab_size; ++v) {
            const int8_t* row = w_ptr + v * h_dim;
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();

            int j = 0;
            for (; j + 4 <= num_vecs; j += 4) {
                __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j * 16));
                __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + (j + 1) * 16));
                __m128i b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + (j + 2) * 16));
                __m128i b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + (j + 3) * 16));

                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(x_exp_ptr[j], _mm256_cvtepi8_epi16(b0)));
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(x_exp_ptr[j + 1], _mm256_cvtepi8_epi16(b1)));
                acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(x_exp_ptr[j + 2], _mm256_cvtepi8_epi16(b2)));
                acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(x_exp_ptr[j + 3], _mm256_cvtepi8_epi16(b3)));
            }

            __m256i total_acc = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
            alignas(32) int32_t tmp[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), total_acc);
            int32_t sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

            for (int rem = j * 16; rem < h_dim; ++rem) {
                sum += static_cast<int32_t>(x_i8_ptr[rem]) * static_cast<int32_t>(row[rem]);
            }
            logits_scratch[v] = static_cast<float>(sum) * (s_x * scales_ptr[v]);
        }
    }
}

int TenzoEngineImpl::sample_top_k_top_p(
    const tenzo_sampling_params_t* params,
    const int* past_tokens,
    int past_tokens_len
) {
    const int vocab_size = config.vocab_size;
    float* logits = logits_scratch.data();
    float temperature = params ? params->temperature : 0.7f;
    float top_p = params ? params->top_p : 0.9f;
    int top_k = std::clamp(params ? params->top_k : 40, 1, 64);
    float repetition_penalty = params ? params->repetition_penalty : 1.15f;

    // Apply repetition penalty directly to past tokens in O(past_tokens_len)
    if (repetition_penalty != 1.0f && past_tokens && past_tokens_len > 0) {
        for (int i = 0; i < past_tokens_len; ++i) {
            int tok = past_tokens[i];
            if (tok >= 0 && tok < vocab_size) {
                float l = logits[tok];
                logits[tok] = (l > 0.0f) ? (l / repetition_penalty) : (l * repetition_penalty);
            }
        }
    }

    if (temperature <= 0.0f) {
        int best_idx = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    using Pair = std::pair<float, int>;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> min_heap;

    for (int i = 0; i < vocab_size; ++i) {
        float l = logits[i];
        if (static_cast<int>(min_heap.size()) < top_k) {
            min_heap.push({l, i});
        } else if (l > min_heap.top().first) {
            min_heap.pop();
            min_heap.push({l, i});
        }
    }

    int k_size = min_heap.size();
    Pair candidates[64];
    for (int i = k_size - 1; i >= 0; --i) {
        candidates[i] = min_heap.top();
        min_heap.pop();
    }

    float max_l = candidates[0].first;
    float sum_exp = 0.0f;
    float probs[64];
    for (int i = 0; i < k_size; ++i) {
        probs[i] = std::exp((candidates[i].first - max_l) / temperature);
        sum_exp += probs[i];
    }
    for (int i = 0; i < k_size; ++i) probs[i] /= sum_exp;

    float cum_p = 0.0f;
    int cutoff = k_size;
    for (int i = 0; i < k_size; ++i) {
        cum_p += probs[i];
        if (cum_p >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    float sub_sum = 0.0f;
    for (int i = 0; i < cutoff; ++i) sub_sum += probs[i];
    for (int i = 0; i < cutoff; ++i) probs[i] /= sub_sum;

    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float acc = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
        acc += probs[i];
        if (r <= acc) return candidates[i].second;
    }
    return candidates[0].second;
}

void TenzoEngineImpl::prefill_token(int token_id) {
    embedding_lookup(token_id);
    for (int l = 0; l < config.num_layers; ++l) {
        forward_layer_raw(l);
    }
    if (use_paged_kv) {
        paged_kv_cache.increment_seq_len(1);
    } else {
        kv_cache.increment_seq_len(1);
    }
}

int TenzoEngineImpl::generate_step(
    int cur_token,
    const tenzo_sampling_params_t* params,
    const int* past_tokens,
    int past_tokens_len
) {
    embedding_lookup(cur_token);
    for (int l = 0; l < config.num_layers; ++l) {
        forward_layer_raw(l);
    }
    if (use_paged_kv) {
        paged_kv_cache.increment_seq_len(1);
    } else {
        kv_cache.increment_seq_len(1);
    }
    compute_logits();
    return sample_top_k_top_p(params, past_tokens, past_tokens_len);
}

// Helpers for model loading
namespace {
std::vector<int8_t> pack_ternary_weights_raw(const uint8_t* raw_bytes, int64_t N, int64_t K) {
    int64_t n_blocks = N / 64;
    int64_t K_half = K / 2;
    std::vector<int8_t> packed(n_blocks * K_half * 32);

    // In SafeTensors from Microsoft BitNet-b1.58-2B-4T:
    // Shape is [N / 4, K] where each byte at (r_packed, c) contains 4 weights for rows:
    //   4 * r_packed + 0 (shift 0)
    //   4 * r_packed + 1 (shift 2)
    //   4 * r_packed + 2 (shift 4)
    //   4 * r_packed + 3 (shift 6)
    // with ternary value = ((byte >> shift) & 3) - 1.
    auto get_tern = [&](int r, int c) -> int {
        int r_packed = r / 4;
        int shift = (r % 4) * 2;
        uint8_t byte_val = raw_bytes[r_packed * K + c];
        return static_cast<int>((byte_val >> shift) & 0x03) - 1;
    };

    for (int64_t b = 0; b < n_blocks; ++b) {
        for (int64_t k = 0; k < K_half; ++k) {
            int8_t* dst = packed.data() + (b * K_half + k) * 32;
            for (int ch = 0; ch < 32; ++ch) {
                int row0 = b * 64 + ch;
                int row1 = b * 64 + ch + 32;

                int col0 = 2 * k;
                int col1 = 2 * k + 1;

                int w0 = get_tern(row0, col0);
                int w1 = get_tern(row0, col1);
                int nibble0 = (w0 + 1) | ((w1 + 1) << 2);

                int w2 = get_tern(row1, col0);
                int w3 = get_tern(row1, col1);
                int nibble1 = (w2 + 1) | ((w3 + 1) << 2);

                dst[ch] = static_cast<int8_t>((nibble0 & 0x0F) | ((nibble1 & 0x0F) << 4));
            }
        }
    }
    return packed;
}
} // namespace

tenzo_status_t TenzoEngineImpl::load_model_from_files(const char* weights_path, const char* mlir_path) {
    std::ifstream mf(mlir_path);
    if (!mf.is_open()) return TENZO_ERROR_IO;
    std::stringstream ss; ss << mf.rdbuf();
    std::string mlir_text = ss.str();

    std::ifstream wf(weights_path, std::ios::binary | std::ios::ate);
    if (!wf.is_open()) return TENZO_ERROR_IO;
    std::streamsize fsize = wf.tellg();
    wf.seekg(0, std::ios::beg);
    std::vector<uint8_t> raw_bytes(fsize);
    if (!wf.read(reinterpret_cast<char*>(raw_bytes.data()), fsize)) return TENZO_ERROR_IO;

    const int vocab_size = config.vocab_size;
    const int hidden_size = config.hidden_size;
    const int num_layers = config.num_layers;
    const int ff_dim = config.ffn_dim;

    // Weight Integrity & Diagnostic Check
    if (fsize < 100000000) {
        std::cerr << "\n⚠️  [Tenzo Diagnostic] Warning: Model file (" << (fsize / (1024*1024)) 
                  << " MB) is smaller than the expected ~1.8 GB for 30-layer BitNet-2B.\n";
    }

    // 1. Embeddings
    size_t embed_bytes = static_cast<size_t>(vocab_size) * hidden_size * 4;
    const float* embed_f32 = reinterpret_cast<const float*>(raw_bytes.data());

    // Check for uniform synthetic placeholder weights
    bool is_dummy_data = true;
    for (int i = 1; i < std::min(200, vocab_size); ++i) {
        if (std::abs(embed_f32[i * hidden_size] - embed_f32[0]) > 1e-6f) {
            is_dummy_data = false;
            break;
        }
    }
    if (is_dummy_data && vocab_size > 200) {
        std::cerr << "\n⚠️  [Tenzo Diagnostic] Notice: Running on synthetic/placeholder model weights.\n"
                  << "   To download genuine pre-trained weights from Hugging Face, run:\n"
                  << "   python3 scripts/download_model.py\n\n";
    }

    // INT8 LM Head quantization
    is_lm_i8 = true;
    embed_w_i8.resize(static_cast<size_t>(vocab_size) * hidden_size);
    embed_scales.resize(vocab_size);

    for (int v = 0; v < vocab_size; ++v) {
        const float* row = embed_f32 + v * hidden_size;
        float amax = 1e-8f;
        for (int i = 0; i < hidden_size; ++i) {
            amax = std::max(amax, std::abs(row[i]));
        }
        float scale = amax / 127.0f;
        embed_scales[v] = scale;
        float inv_s = 1.0f / scale;
        for (int i = 0; i < hidden_size; ++i) {
            embed_w_i8[v * hidden_size + i] = static_cast<int8_t>(std::clamp(std::round(row[i] * inv_s), -127.0f, 127.0f));
        }
    }

    // 2. Parse views and scales from MLIR or calculate sequential offsets safely
    std::regex view_regex(R"((%\w+)\s*=\s*arith\.constant\s+(\d+)\s*:\s*index\s*\r?\n\s*(%\w+)\s*=\s*memref\.view\s+%\w+\[\1\]\[\]\s*:\s*memref<\?xi8>\s+to\s+memref<([^>]+)>)");
    std::vector<size_t> all_views;
    auto v_begin = std::sregex_iterator(mlir_text.begin(), mlir_text.end(), view_regex);
    auto v_end = std::sregex_iterator();
    for (auto it = v_begin; it != v_end; ++it) {
        all_views.push_back(std::stoull((*it)[2].str()));
    }

    std::regex scale_regex(R"((%\w+)\s*=\s*arith\.constant\s+([0-9\.e\+\-]+)\s*:\s*f32\s*\r?\n\s*%\w+\s*=\s*"tenzo\.bitlinear_elut"\([^,]+,\s*[^,]+,\s*\1\))");
    std::vector<float> all_scales;
    auto s_begin = std::sregex_iterator(mlir_text.begin(), mlir_text.end(), scale_regex);
    auto s_end = std::sregex_iterator();
    for (auto it = s_begin; it != s_end; ++it) {
        all_scales.push_back(std::stof((*it)[2].str()));
    }

    // Lambda helpers with strict bounds checking
    auto safe_slice_f32 = [&](size_t off, size_t count) -> std::vector<float> {
        std::vector<float> vec(count, 1.0f);
        if (off + count * sizeof(float) <= raw_bytes.size()) {
            const float* p = reinterpret_cast<const float*>(raw_bytes.data() + off);
            vec.assign(p, p + count);
        }
        return vec;
    };

    auto safe_pack_ternary = [&](size_t off, int64_t N, int64_t K) -> std::vector<int8_t> {
        size_t needed = static_cast<size_t>(N) * (K / 4);
        if (off + needed <= raw_bytes.size()) {
            return pack_ternary_weights_raw(raw_bytes.data() + off, N, K);
        }
        int64_t n_blocks = N / 64;
        int64_t K_half = K / 2;
        return std::vector<int8_t>(n_blocks * K_half * 32, 0x55);
    };

    size_t v_idx = 1;
    size_t s_idx = 0;
    size_t seq_offset = embed_bytes;

    for (int l = 0; l < num_layers; ++l) {
        auto& lw = layers[l];

        auto get_offset = [&](size_t sz_bytes) -> size_t {
            size_t off;
            if (v_idx < all_views.size()) {
                off = all_views[v_idx++];
            } else {
                off = seq_offset;
            }
            seq_offset += sz_bytes;
            return off;
        };

        auto get_scale = [&]() -> float {
            if (s_idx < all_scales.size()) {
                return all_scales[s_idx++];
            }
            return 1.0f;
        };

        // 1. in_norm
        size_t off = get_offset(hidden_size * sizeof(float));
        lw.in_norm = safe_slice_f32(off, hidden_size);
        lw.in_norm_ptr = lw.in_norm.data();

        // 2. q_proj
        off = get_offset(2560 * (2560 / 4));
        lw.q_w = safe_pack_ternary(off, 2560, 2560);
        lw.q_w_ptr = lw.q_w.data();
        lw.q_scale = get_scale();

        // 3. k_proj
        off = get_offset(640 * (2560 / 4));
        lw.k_w = safe_pack_ternary(off, 640, 2560);
        lw.k_w_ptr = lw.k_w.data();
        lw.k_scale = get_scale();

        // 4. v_proj
        off = get_offset(640 * (2560 / 4));
        lw.v_w = safe_pack_ternary(off, 640, 2560);
        lw.v_w_ptr = lw.v_w.data();
        lw.v_scale = get_scale();

        // 5. attn_sub_norm
        off = get_offset(hidden_size * sizeof(float));
        lw.attn_sub_norm = safe_slice_f32(off, hidden_size);
        lw.attn_sub_norm_ptr = lw.attn_sub_norm.data();

        // 6. out_proj
        off = get_offset(2560 * (2560 / 4));
        lw.out_w = safe_pack_ternary(off, 2560, 2560);
        lw.out_w_ptr = lw.out_w.data();
        lw.out_scale = get_scale();

        // 7. post_norm
        off = get_offset(hidden_size * sizeof(float));
        lw.post_norm = safe_slice_f32(off, hidden_size);
        lw.post_norm_ptr = lw.post_norm.data();

        // 8. gate_proj
        off = get_offset(ff_dim * (2560 / 4));
        lw.gate_w = safe_pack_ternary(off, ff_dim, 2560);
        lw.gate_w_ptr = lw.gate_w.data();
        lw.gate_scale = get_scale();

        // 9. up_proj
        off = get_offset(ff_dim * (2560 / 4));
        lw.up_w = safe_pack_ternary(off, ff_dim, 2560);
        lw.up_w_ptr = lw.up_w.data();
        lw.up_scale = get_scale();

        // 10. ffn_sub_norm
        off = get_offset(ff_dim * sizeof(float));
        lw.ffn_sub_norm = safe_slice_f32(off, ff_dim);
        lw.ffn_sub_norm_ptr = lw.ffn_sub_norm.data();

        // 11. down_proj
        off = get_offset(2560 * (ff_dim / 4));
        lw.down_w = safe_pack_ternary(off, 2560, ff_dim);
        lw.down_w_ptr = lw.down_w.data();
        lw.down_scale = get_scale();
    }

    size_t final_off = (v_idx < all_views.size()) ? all_views[v_idx] : seq_offset;
    final_norm_w = safe_slice_f32(final_off, hidden_size);
    final_norm_ptr = final_norm_w.data();

    return TENZO_SUCCESS;
}

} // namespace tenzo

//===----------------------------------------------------------------------===//
// C-API Export Implementation
//===----------------------------------------------------------------------===//

extern "C" {

const char* tenzo_get_version(void) {
    return TENZO_VERSION_STRING;
}

tenzo_config_t tenzo_default_config(void) {
    tenzo_config_t cfg;
    cfg.hidden_size = 2560;
    cfg.num_q_heads = 20;
    cfg.num_kv_heads = 5;
    cfg.head_dim = 128;
    cfg.num_layers = 30;
    cfg.ffn_dim = 6912;
    cfg.vocab_size = 128256;
    cfg.max_seq_len = 8192;
    cfg.kv_mode = "int8_fused";
    cfg.device = "cpu";
    return cfg;
}

tenzo_sampling_params_t tenzo_default_sampling_params(void) {
    tenzo_sampling_params_t p;
    p.temperature = 0.7f;
    p.top_p = 0.9f;
    p.top_k = 40;
    p.repetition_penalty = 1.15f;
    return p;
}

tenzo_engine_t tenzo_create_engine(const tenzo_config_t* config) {
    tenzo_config_t cfg = config ? *config : tenzo_default_config();
    auto handle = new tenzo_engine_opaque();
    handle->impl = std::make_unique<tenzo::TenzoEngineImpl>(cfg);
    return handle;
}

void tenzo_destroy_engine(tenzo_engine_t engine) {
    if (engine) {
        delete engine;
    }
}

void tenzo_reset(tenzo_engine_t engine) {
    if (engine && engine->impl) {
        engine->impl->reset();
    }
}

int tenzo_get_seq_len(tenzo_engine_t engine) {
    return (engine && engine->impl) ? engine->impl->get_seq_len() : 0;
}

tenzo_status_t tenzo_set_layer_weights(
    tenzo_engine_t engine,
    int layer_idx,
    const tenzo_layer_weights_t* weights
) {
    if (!engine || !engine->impl || !weights) return TENZO_ERROR_INVALID_ARGUMENT;
    engine->impl->set_layer_weights(layer_idx, weights);
    return TENZO_SUCCESS;
}

tenzo_status_t tenzo_set_final_norm(tenzo_engine_t engine, const float* norm_w) {
    if (!engine || !engine->impl || !norm_w) return TENZO_ERROR_INVALID_ARGUMENT;
    engine->impl->set_final_norm(norm_w);
    return TENZO_SUCCESS;
}

tenzo_status_t tenzo_set_lm_head_f32(
    tenzo_engine_t engine,
    const float* embed_w,
    int vocab_size,
    int hidden_size
) {
    if (!engine || !engine->impl || !embed_w) return TENZO_ERROR_INVALID_ARGUMENT;
    engine->impl->set_lm_head_f32(embed_w, vocab_size, hidden_size);
    return TENZO_SUCCESS;
}

tenzo_status_t tenzo_set_lm_head_i8(
    tenzo_engine_t engine,
    const int8_t* embed_w,
    const float* embed_scales,
    int vocab_size,
    int hidden_size
) {
    if (!engine || !engine->impl || !embed_w || !embed_scales) return TENZO_ERROR_INVALID_ARGUMENT;
    engine->impl->set_lm_head_i8(embed_w, embed_scales, vocab_size, hidden_size);
    return TENZO_SUCCESS;
}

tenzo_status_t tenzo_load_model(
    tenzo_engine_t engine,
    const char* weights_path,
    const char* mlir_path
) {
    if (!engine || !engine->impl || !weights_path || !mlir_path) return TENZO_ERROR_INVALID_ARGUMENT;
    return engine->impl->load_model_from_files(weights_path, mlir_path);
}

tenzo_status_t tenzo_prefill_token(tenzo_engine_t engine, int token_id) {
    if (!engine || !engine->impl) return TENZO_ERROR_INVALID_ARGUMENT;
    engine->impl->prefill_token(token_id);
    return TENZO_SUCCESS;
}

int tenzo_generate_step(
    tenzo_engine_t engine,
    int cur_token,
    const tenzo_sampling_params_t* params,
    const int* past_tokens,
    int past_tokens_len
) {
    if (!engine || !engine->impl) return -1;
    return engine->impl->generate_step(cur_token, params, past_tokens, past_tokens_len);
}

} // extern "C"
