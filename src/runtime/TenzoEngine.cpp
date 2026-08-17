/**
 * @file TenzoEngine.cpp
 * @brief Tenzo Native Inference Engine & C-API Implementation (Phase E)
 */

#include "TenzoEngine.h"
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
    } else {
        fp32_k.resize(total_elements);
        fp32_v.resize(total_elements);
    }
}

void FusedKVCache::reset() {
    cur_seq_len = 0;
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

    // 3. Append to KV-Cache
    size_t layer_kv_offset = (static_cast<size_t>(layer_idx) * max_seq_len + pos) * num_kv_heads * head_dim;
    size_t layer_scale_offset = (static_cast<size_t>(layer_idx) * max_seq_len + pos) * num_kv_heads;

    if (kv_mode == "int8_fused") {
        for (int h = 0; h < num_kv_heads; ++h) {
            const float* k_src = k_ptr + h * head_dim;
            const float* v_src = v_ptr + h * head_dim;
            int8_t* k_dst = int8_k.data() + layer_kv_offset + h * head_dim;
            int8_t* v_dst = int8_v.data() + layer_kv_offset + h * head_dim;

            float amax_k = 1e-8f, amax_v = 1e-8f;
            for (int d = 0; d < head_dim; ++d) {
                amax_k = std::max(amax_k, std::abs(k_src[d]));
                amax_v = std::max(amax_v, std::abs(v_src[d]));
            }
            float s_k = amax_k / 127.0f;
            float s_v = amax_v / 127.0f;
            int8_k_scales[layer_scale_offset + h] = s_k;
            int8_v_scales[layer_scale_offset + h] = s_v;

            float inv_sk = 1.0f / s_k;
            float inv_sv = 1.0f / s_v;
            for (int d = 0; d < head_dim; ++d) {
                k_dst[d] = static_cast<int8_t>(std::clamp(std::round(k_src[d] * inv_sk), -128.0f, 127.0f));
                v_dst[d] = static_cast<int8_t>(std::clamp(std::round(v_src[d] * inv_sv), -128.0f, 127.0f));
            }
        }
    } else if (kv_mode == "tl1_fused") {
        size_t tl1_layer_offset = (static_cast<size_t>(layer_idx) * max_seq_len + pos) * num_kv_heads * (head_dim / 4);
        for (int h = 0; h < num_kv_heads; ++h) {
            const float* k_src = k_ptr + h * head_dim;
            const float* v_src = v_ptr + h * head_dim;
            uint8_t* k_dst = tl1_k.data() + tl1_layer_offset + h * (head_dim / 4);
            uint8_t* v_dst = tl1_v.data() + tl1_layer_offset + h * (head_dim / 4);

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

            tl1_k_scales[layer_scale_offset + h] = s_k;
            tl1_v_scales[layer_scale_offset + h] = s_v;

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
    } else {
        std::memcpy(fp32_k.data() + layer_kv_offset, k_ptr, num_kv_heads * head_dim * sizeof(float));
        std::memcpy(fp32_v.data() + layer_kv_offset, v_ptr, num_kv_heads * head_dim * sizeof(float));
    }

    // 4. Scaled Dot-Product Attention
    const int seq_len = pos + 1;
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));

    #pragma omp parallel for schedule(static)
    for (int qh = 0; qh < num_q_heads; ++qh) {
        int kv_h = qh / gqa_ratio;
        const float* q_vec = q_ptr + qh * head_dim;
        float* out_vec = out_ptr + qh * head_dim;

        alignas(32) float scores_buf[8192];
        float* scores = (seq_len <= 8192) ? scores_buf : new float[seq_len];
        float max_score = -1e30f;

        if (kv_mode == "int8_fused") {
            for (int t = 0; t < seq_len; ++t) {
                size_t offset_k = (static_cast<size_t>(layer_idx) * max_seq_len + t) * num_kv_heads * head_dim + kv_h * head_dim;
                size_t offset_s = (static_cast<size_t>(layer_idx) * max_seq_len + t) * num_kv_heads + kv_h;
                const int8_t* k_cached = int8_k.data() + offset_k;
                float s_k = int8_k_scales[offset_s];

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
            for (int t = 0; t < seq_len; ++t) {
                size_t offset_k = (static_cast<size_t>(layer_idx) * max_seq_len + t) * num_kv_heads * (head_dim / 4) + kv_h * (head_dim / 4);
                size_t offset_s = (static_cast<size_t>(layer_idx) * max_seq_len + t) * num_kv_heads + kv_h;
                const uint8_t* k_cached = tl1_k.data() + offset_k;
                float amax_k = tl1_k_scales[offset_s];

                float dot = 0.0f;
                for (int d = 0; d < head_dim; d += 4) {
                    uint8_t packed = k_cached[d / 4];
                    int w0 = static_cast<int>(packed & 0x03) - 1;
                    int w1 = static_cast<int>((packed >> 2) & 0x03) - 1;
                    int w2 = static_cast<int>((packed >> 4) & 0x03) - 1;
                    int w3 = static_cast<int>((packed >> 6) & 0x03) - 1;
                    dot += q_vec[d + 0] * w0 + q_vec[d + 1] * w1 + q_vec[d + 2] * w2 + q_vec[d + 3] * w3;
                }
                dot *= (amax_k * scale_factor);
                scores[t] = dot;
                if (dot > max_score) max_score = dot;
            }
        } else {
            for (int t = 0; t < seq_len; ++t) {
                size_t offset_k = (static_cast<size_t>(layer_idx) * max_seq_len + t) * num_kv_heads * head_dim + kv_h * head_dim;
                const float* k_cached = fp32_k.data() + offset_k;
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
            for (int t = 0; t < seq_len; ++t) {
                size_t offset_v = (static_cast<size_t>(layer_idx) * max_seq_len + t) * num_kv_heads * head_dim + kv_h * head_dim;
                size_t offset_s = (static_cast<size_t>(layer_idx) * max_seq_len + t) * num_kv_heads + kv_h;
                const int8_t* v_cached = int8_v.data() + offset_v;
                __m256 weight_scale = _mm256_set1_ps(scores[t] * int8_v_scales[offset_s]);

                for (int i = 0; i < 16; ++i) {
                    __m128i v_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(v_cached + i * 8));
                    __m256 v_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_bytes));
                    acc[i] = _mm256_fmadd_ps(v_f32, weight_scale, acc[i]);
                }
            }
        } else if (kv_mode == "tl1_fused") {
            std::memset(out_vec, 0, head_dim * sizeof(float));
            for (int t = 0; t < seq_len; ++t) {
                size_t offset_v = (static_cast<size_t>(layer_idx) * max_seq_len + t) * num_kv_heads * (head_dim / 4) + kv_h * (head_dim / 4);
                size_t offset_s = (static_cast<size_t>(layer_idx) * max_seq_len + t) * num_kv_heads + kv_h;
                const uint8_t* v_cached = tl1_v.data() + offset_v;
                float eff_w = scores[t] * tl1_v_scales[offset_s];

                for (int d = 0; d < head_dim; d += 4) {
                    uint8_t packed = v_cached[d / 4];
                    int w0 = static_cast<int>(packed & 0x03) - 1;
                    int w1 = static_cast<int>((packed >> 2) & 0x03) - 1;
                    int w2 = static_cast<int>((packed >> 4) & 0x03) - 1;
                    int w3 = static_cast<int>((packed >> 6) & 0x03) - 1;
                    out_vec[d + 0] += w0 * eff_w;
                    out_vec[d + 1] += w1 * eff_w;
                    out_vec[d + 2] += w2 * eff_w;
                    out_vec[d + 3] += w3 * eff_w;
                }
            }
        } else {
            for (int t = 0; t < seq_len; ++t) {
                size_t offset_v = (static_cast<size_t>(layer_idx) * max_seq_len + t) * num_kv_heads * head_dim + kv_h * head_dim;
                const float* v_cached = fp32_v.data() + offset_v;
                __m256 weight_vec = _mm256_set1_ps(scores[t]);
                for (int i = 0; i < 16; ++i) {
                    acc[i] = _mm256_fmadd_ps(_mm256_loadu_ps(v_cached + i * 8), weight_vec, acc[i]);
                }
            }
        }

        if (kv_mode != "tl1_fused") {
            for (int i = 0; i < 16; ++i) {
                _mm256_storeu_ps(out_vec + i * 8, acc[i]);
            }
        }

        if (scores != scores_buf) {
            delete[] scores;
        }
    }
}

//===----------------------------------------------------------------------===//
// TenzoEngineImpl Implementation
//===----------------------------------------------------------------------===//

TenzoEngineImpl::TenzoEngineImpl(const tenzo_config_t& cfg)
    : config(cfg),
      kv_cache(cfg.num_layers, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len, cfg.kv_mode ? cfg.kv_mode : "int8_fused") {
    
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
    kv_cache.reset();
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

void TenzoEngineImpl::forward_layer_raw(int layer_idx) {
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

    // 4. In-place Zero-Copy Attention with RoPE and Compressed KV-Cache
    kv_cache.forward_attention_raw(layer_idx, buf_q.data(), buf_k.data(), buf_v.data(), buf_attn_out.data());

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
    int top_k = params ? params->top_k : 40;
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
    kv_cache.increment_seq_len(1);
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
    kv_cache.increment_seq_len(1);
    compute_logits();
    return sample_top_k_top_p(params, past_tokens, past_tokens_len);
}

// Helpers for model loading
namespace {
std::vector<int8_t> pack_ternary_weights_raw(const uint8_t* raw_bytes, int64_t N, int64_t K) {
    int64_t n_blocks = N / 64;
    int64_t K_half = K / 2;
    std::vector<int8_t> packed(n_blocks * K_half * 32);

    for (int64_t b = 0; b < n_blocks; ++b) {
        for (int64_t k = 0; k < K_half; ++k) {
            int8_t* dst = packed.data() + (b * K_half + k) * 32;
            for (int ch = 0; ch < 32; ++ch) {
                int row0 = b * 64 + ch;
                int row1 = b * 64 + ch + 32;

                int col0 = 2 * k;
                int col1 = 2 * k + 1;

                // Read 2-bit values from raw_bytes [N, K/4]
                auto get_tern = [&](int r, int c) -> int {
                    uint8_t byte_val = raw_bytes[r * (K / 4) + (c / 4)];
                    int shift = (c % 4) * 2;
                    return static_cast<int>((byte_val >> shift) & 0x03) - 1;
                };

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

    // 1. Embeddings
    size_t embed_bytes = static_cast<size_t>(vocab_size) * hidden_size * 4;
    const float* embed_f32 = reinterpret_cast<const float*>(raw_bytes.data());

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

    // 2. Parse views and scales from MLIR
    std::regex view_regex(R"((%\w+)\s*=\s*arith\.constant\s+(\d+)\s*:\s*index\s*\n\s*(%\w+)\s*=\s*memref\.view\s+%\w+\[\1\]\[\]\s*:\s*memref<\?xi8>\s+to\s+memref<([^>]+)>)");
    std::vector<size_t> all_views;
    auto v_begin = std::sregex_iterator(mlir_text.begin(), mlir_text.end(), view_regex);
    auto v_end = std::sregex_iterator();
    for (auto it = v_begin; it != v_end; ++it) {
        all_views.push_back(std::stoull((*it)[2].str()));
    }

    std::regex scale_regex(R"((%\w+)\s*=\s*arith\.constant\s+([0-9\.e\+\-]+)\s*:\s*f32\s*\n\s*%\w+\s*=\s*"tenzo\.bitlinear_elut"\([^,]+,\s*[^,]+,\s*\1\))");
    std::vector<float> all_scales;
    auto s_begin = std::sregex_iterator(mlir_text.begin(), mlir_text.end(), scale_regex);
    auto s_end = std::sregex_iterator();
    for (auto it = s_begin; it != s_end; ++it) {
        all_scales.push_back(std::stof((*it)[2].str()));
    }

    size_t v_idx = 1; // 0 is embed_tokens
    size_t s_idx = 0;

    for (int l = 0; l < num_layers; ++l) {
        auto& lw = layers[l];

        // 1. in_norm
        size_t off = all_views[v_idx++];
        lw.in_norm.assign(reinterpret_cast<const float*>(raw_bytes.data() + off),
                          reinterpret_cast<const float*>(raw_bytes.data() + off) + hidden_size);
        lw.in_norm_ptr = lw.in_norm.data();

        // 2. q_proj
        off = all_views[v_idx++];
        lw.q_w = pack_ternary_weights_raw(raw_bytes.data() + off, 2560, 2560);
        lw.q_w_ptr = lw.q_w.data();
        lw.q_scale = all_scales[s_idx++];

        // 3. k_proj
        off = all_views[v_idx++];
        lw.k_w = pack_ternary_weights_raw(raw_bytes.data() + off, 640, 2560);
        lw.k_w_ptr = lw.k_w.data();
        lw.k_scale = all_scales[s_idx++];

        // 4. v_proj
        off = all_views[v_idx++];
        lw.v_w = pack_ternary_weights_raw(raw_bytes.data() + off, 640, 2560);
        lw.v_w_ptr = lw.v_w.data();
        lw.v_scale = all_scales[s_idx++];

        // 5. attn_sub_norm
        off = all_views[v_idx++];
        lw.attn_sub_norm.assign(reinterpret_cast<const float*>(raw_bytes.data() + off),
                                reinterpret_cast<const float*>(raw_bytes.data() + off) + hidden_size);
        lw.attn_sub_norm_ptr = lw.attn_sub_norm.data();

        // 6. out_proj
        off = all_views[v_idx++];
        lw.out_w = pack_ternary_weights_raw(raw_bytes.data() + off, 2560, 2560);
        lw.out_w_ptr = lw.out_w.data();
        lw.out_scale = all_scales[s_idx++];

        // 7. post_norm
        off = all_views[v_idx++];
        lw.post_norm.assign(reinterpret_cast<const float*>(raw_bytes.data() + off),
                            reinterpret_cast<const float*>(raw_bytes.data() + off) + hidden_size);
        lw.post_norm_ptr = lw.post_norm.data();

        // 8. gate_proj
        off = all_views[v_idx++];
        lw.gate_w = pack_ternary_weights_raw(raw_bytes.data() + off, ff_dim, 2560);
        lw.gate_w_ptr = lw.gate_w.data();
        lw.gate_scale = all_scales[s_idx++];

        // 9. up_proj
        off = all_views[v_idx++];
        lw.up_w = pack_ternary_weights_raw(raw_bytes.data() + off, ff_dim, 2560);
        lw.up_w_ptr = lw.up_w.data();
        lw.up_scale = all_scales[s_idx++];

        // 10. ffn_sub_norm
        off = all_views[v_idx++];
        lw.ffn_sub_norm.assign(reinterpret_cast<const float*>(raw_bytes.data() + off),
                               reinterpret_cast<const float*>(raw_bytes.data() + off) + ff_dim);
        lw.ffn_sub_norm_ptr = lw.ffn_sub_norm.data();

        // 11. down_proj
        off = all_views[v_idx++];
        lw.down_w = pack_ternary_weights_raw(raw_bytes.data() + off, 2560, ff_dim);
        lw.down_w_ptr = lw.down_w.data();
        lw.down_scale = all_scales[s_idx++];
    }

    if (v_idx < all_views.size()) {
        size_t off = all_views[v_idx];
        final_norm_w.assign(reinterpret_cast<const float*>(raw_bytes.data() + off),
                            reinterpret_cast<const float*>(raw_bytes.data() + off) + hidden_size);
        final_norm_ptr = final_norm_w.data();
    }

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
