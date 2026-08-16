#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <immintrin.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <optional>

namespace py = pybind11;

//===----------------------------------------------------------------------===//
// 1. AVX2 High-Performance FP32 GEMM (BLIS 6x16 Micro-Kernel)
//===----------------------------------------------------------------------===//

namespace {

constexpr int MR = 6;
constexpr int NR = 16;

inline void micro_kernel_6x16_avx2(
    const float* __restrict__ A_packed,
    const float* __restrict__ B_packed,
    float* __restrict__ C,
    int ldc,
    int K,
    const float* __restrict__ bias,
    bool apply_relu
) {
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        __m256 b0 = _mm256_loadu_ps(B_packed + k * NR);
        __m256 b1 = _mm256_loadu_ps(B_packed + k * NR + 8);

        __m256 a0 = _mm256_broadcast_ss(A_packed + k * MR + 0);
        __m256 a1 = _mm256_broadcast_ss(A_packed + k * MR + 1);
        __m256 a2 = _mm256_broadcast_ss(A_packed + k * MR + 2);
        __m256 a3 = _mm256_broadcast_ss(A_packed + k * MR + 3);
        __m256 a4 = _mm256_broadcast_ss(A_packed + k * MR + 4);
        __m256 a5 = _mm256_broadcast_ss(A_packed + k * MR + 5);

        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    __m256 bias0 = bias ? _mm256_loadu_ps(bias) : _mm256_setzero_ps();
    __m256 bias1 = bias ? _mm256_loadu_ps(bias + 8) : _mm256_setzero_ps();
    __m256 zero = _mm256_setzero_ps();

    auto store_row = [&](int r, __m256 r0, __m256 r1) {
        if (bias) {
            r0 = _mm256_add_ps(r0, bias0);
            r1 = _mm256_add_ps(r1, bias1);
        }
        if (apply_relu) {
            r0 = _mm256_max_ps(r0, zero);
            r1 = _mm256_max_ps(r1, zero);
        }
        _mm256_storeu_ps(C + r * ldc, r0);
        _mm256_storeu_ps(C + r * ldc + 8, r1);
    };

    store_row(0, c00, c01);
    store_row(1, c10, c11);
    store_row(2, c20, c21);
    store_row(3, c30, c31);
    store_row(4, c40, c41);
    store_row(5, c50, c51);
}

void gemm_avx2_dense(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    const float* bias = nullptr,
    bool apply_relu = false
) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i += MR) {
        for (int j = 0; j < N; j += NR) {
            int ib = std::min(MR, M - i);
            int jb = std::min(NR, N - j);

            if (ib == MR && jb == NR) {
                float A_tile[MR * K] __attribute__((aligned(32)));
                float B_tile[K * NR] __attribute__((aligned(32)));

                for (int k = 0; k < K; ++k) {
                    for (int r = 0; r < MR; ++r) {
                        A_tile[k * MR + r] = A[(i + r) * K + k];
                    }
                    for (int c = 0; c < NR; ++c) {
                        // B is [N, K], so B(j+c, k) = B[(j + c) * K + k]
                        B_tile[k * NR + c] = B[(j + c) * K + k];
                    }
                }

                micro_kernel_6x16_avx2(
                    A_tile, B_tile,
                    C + i * N + j, N,
                    K,
                    bias ? bias + j : nullptr,
                    apply_relu
                );
            } else {
                // Scalar edge fallback
                for (int ii = 0; ii < ib; ++ii) {
                    for (int jj = 0; jj < jb; ++jj) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k) {
                            sum += A[(i + ii) * K + k] * B[(j + jj) * K + k];
                        }
                        if (bias) sum += bias[j + jj];
                        if (apply_relu && sum < 0.0f) sum = 0.0f;
                        C[(i + ii) * N + (j + jj)] = sum;
                    }
                }
            }
        }
    }
}

} // namespace

//===----------------------------------------------------------------------===//
// 2. AVX2 256-bit BitLinear TL1 Micro-Kernel (vpshufb LUT SIMD)
//===----------------------------------------------------------------------===//

namespace {

void bitlinear_tl1_avx2_single(
    const float* __restrict__ act,
    const int8_t* __restrict__ packed_w, // [n_blocks, K/2, 32]
    float* __restrict__ out,
    int64_t K,
    int64_t N,
    float weight_scale
) {
    const int64_t K_half = K / 2;
    const int64_t n_blocks = N / 64;

    // 1. Find max abs for activation quantization
    float max_abs = 1e-6f;
    for (int64_t k = 0; k < K; ++k) {
        float val = std::abs(act[k]);
        if (val > max_abs) max_abs = val;
    }

    const float scale_act = 31.0f / max_abs;
    const float inv_scale_act = max_abs / 31.0f;
    const float total_scale = inv_scale_act * weight_scale;

    // 2. Quantize activations to int16
    std::vector<int16_t> act_i16(K);
    for (int64_t k = 0; k < K; ++k) {
        act_i16[k] = static_cast<int16_t>(std::round(act[k] * scale_act));
    }

    // 3. Build 256-bit LUTs (16 bytes per 128-bit lane duplicated)
    std::vector<__m256i> lut_vecs(K_half);
    static const int w0_vals[3] = {-1, 0, 1};
    static const int w1_vals[3] = {-1, 0, 1};

    for (int64_t k = 0; k < K_half; ++k) {
        int16_t a0 = act_i16[2 * k];
        int16_t a1 = act_i16[2 * k + 1];

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

    const __m256i mask_low = _mm256_set1_epi8(0x0F);
    const __m256 scale_vec = _mm256_set1_ps(total_scale);

    // 4. Dot Product across N blocks of 64 channels
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < n_blocks; ++b) {
        const int8_t* block_w = packed_w + b * K_half * 32;

        __m256i acc_low_0 = _mm256_setzero_si256(); // ch 0..7
        __m256i acc_low_1 = _mm256_setzero_si256(); // ch 8..15
        __m256i acc_low_2 = _mm256_setzero_si256(); // ch 16..23
        __m256i acc_low_3 = _mm256_setzero_si256(); // ch 24..31

        __m256i acc_high_0 = _mm256_setzero_si256(); // ch 32..39
        __m256i acc_high_1 = _mm256_setzero_si256(); // ch 40..47
        __m256i acc_high_2 = _mm256_setzero_si256(); // ch 48..55
        __m256i acc_high_3 = _mm256_setzero_si256(); // ch 56..63

        for (int64_t k = 0; k < K_half; ++k) {
            __m256i w_bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block_w + k * 32));
            __m256i lut = lut_vecs[k];

            // Extract nibbles
            __m256i idx_low = _mm256_and_si256(w_bytes, mask_low);
            __m256i idx_high = _mm256_and_si256(_mm256_srli_epi16(w_bytes, 4), mask_low);

            // Parallel lookup with vpshufb
            __m256i res_low = _mm256_shuffle_epi8(lut, idx_low);
            __m256i res_high = _mm256_shuffle_epi8(lut, idx_high);

            // Low 32 channels: sign extend 8-bit to 16-bit
            __m256i res_l16_0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(res_low, 0)); // bytes 0..15 -> 16 int16
            __m256i res_l16_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(res_low, 1)); // bytes 16..31 -> 16 int16

            acc_low_0 = _mm256_add_epi32(acc_low_0, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res_l16_0, 0)));
            acc_low_1 = _mm256_add_epi32(acc_low_1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res_l16_0, 1)));
            acc_low_2 = _mm256_add_epi32(acc_low_2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res_l16_1, 0)));
            acc_low_3 = _mm256_add_epi32(acc_low_3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res_l16_1, 1)));

            // High 32 channels: sign extend 8-bit to 16-bit
            __m256i res_h16_0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(res_high, 0)); // bytes 0..15 -> 16 int16
            __m256i res_h16_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(res_high, 1)); // bytes 16..31 -> 16 int16

            acc_high_0 = _mm256_add_epi32(acc_high_0, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res_h16_0, 0)));
            acc_high_1 = _mm256_add_epi32(acc_high_1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res_h16_0, 1)));
            acc_high_2 = _mm256_add_epi32(acc_high_2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res_h16_1, 0)));
            acc_high_3 = _mm256_add_epi32(acc_high_3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res_h16_1, 1)));
        }

        // Store out all 64 channels
        auto store_8f = [&](__m256i acc, float* dst) {
            __m256 f = _mm256_cvtepi32_ps(acc);
            f = _mm256_mul_ps(f, scale_vec);
            _mm256_storeu_ps(dst, f);
        };

        float* out_block = out + b * 64;
        store_8f(acc_low_0, out_block + 0);
        store_8f(acc_low_1, out_block + 8);
        store_8f(acc_low_2, out_block + 16);
        store_8f(acc_low_3, out_block + 24);
        store_8f(acc_high_0, out_block + 32);
        store_8f(acc_high_1, out_block + 40);
        store_8f(acc_high_2, out_block + 48);
        store_8f(acc_high_3, out_block + 56);
    }
}

} // namespace

//===----------------------------------------------------------------------===//
// 3. Pybind11 Export Functions
//===----------------------------------------------------------------------===//

// Helper: Pack ternary weight matrix [N, K] into Microsoft TL1 [N/64, K/2, 32] format
py::array_t<int8_t> pack_ternary_weights(py::array_t<int8_t> tern_weights) {
    auto buf = tern_weights.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Ternary weights must be 2D [N, K]");
    }
    int64_t N = buf.shape[0];
    int64_t K = buf.shape[1];
    if (N % 64 != 0 || K % 2 != 0) {
        throw std::runtime_error("N must be multiple of 64 and K must be multiple of 2");
    }

    int64_t n_blocks = N / 64;
    int64_t K_half = K / 2;

    auto packed = py::array_t<int8_t>({n_blocks, K_half, (int64_t)32});
    auto p_buf = packed.request();

    const int8_t* src = static_cast<const int8_t*>(buf.ptr);
    int8_t* dst = static_cast<int8_t*>(p_buf.ptr);

    for (int64_t b = 0; b < n_blocks; ++b) {
        for (int64_t k = 0; k < K_half; ++k) {
            for (int64_t c = 0; c < 32; ++c) {
                // Low channel (0..31)
                int64_t n_low = b * 64 + c;
                int8_t w0_low = src[n_low * K + 2 * k];
                int8_t w1_low = src[n_low * K + 2 * k + 1];
                uint8_t idx_low = (w0_low + 1) | ((w1_low + 1) << 2);

                // High channel (32..63)
                int64_t n_high = b * 64 + 32 + c;
                int8_t w0_high = src[n_high * K + 2 * k];
                int8_t w1_high = src[n_high * K + 2 * k + 1];
                uint8_t idx_high = (w0_high + 1) | ((w1_high + 1) << 2);

                dst[b * K_half * 32 + k * 32 + c] = static_cast<int8_t>((idx_low & 0x0F) | ((idx_high & 0x0F) << 4));
            }
        }
    }
    return packed;
}

// 1. FP32 High-Performance GEMM Binding
py::array_t<float> dispatch_linear_fp32(
    py::array_t<float> input,
    py::array_t<float> weight,
    std::optional<py::array_t<float>> bias = std::nullopt,
    bool relu = false
) {
    auto buf_in = input.request();
    auto buf_w = weight.request();

    if (buf_in.ndim != 2 && buf_in.ndim != 3) {
        throw std::runtime_error("Input must be 2D [M, K] or 3D [Batch, Seq, K]");
    }
    if (buf_w.ndim != 2) {
        throw std::runtime_error("Weight must be 2D [N, K]");
    }

    int64_t batch = (buf_in.ndim == 3) ? buf_in.shape[0] : 1;
    int64_t seq = (buf_in.ndim == 3) ? buf_in.shape[1] : buf_in.shape[0];
    int64_t M = batch * seq;
    int64_t K = buf_in.shape[buf_in.ndim - 1];
    int64_t N = buf_w.shape[0];

    if (buf_w.shape[1] != K) {
        throw std::runtime_error("Weight shape mismatch: expected [N, K]");
    }

    py::array_t<float> result;
    if (buf_in.ndim == 3) {
        result = py::array_t<float>({batch, seq, N});
    } else {
        result = py::array_t<float>({M, N});
    }

    const float* ptr_in = static_cast<const float*>(buf_in.ptr);
    const float* ptr_w = static_cast<const float*>(buf_w.ptr);
    float* ptr_out = static_cast<float*>(result.request().ptr);

    const float* ptr_b = nullptr;
    if (bias.has_value()) {
        auto buf_b = bias->request();
        if (buf_b.size != N) throw std::runtime_error("Bias size mismatch");
        ptr_b = static_cast<const float*>(buf_b.ptr);
    }

    gemm_avx2_dense(ptr_in, ptr_w, ptr_out, M, N, K, ptr_b, relu);

    return result;
}

// 2. BitLinear TL1 SIMD Binding
py::array_t<float> dispatch_bitlinear_tl1(
    py::array_t<float> input,
    py::array_t<int8_t> packed_weight,
    float scale = 1.0f,
    std::optional<py::array_t<float>> bias = std::nullopt
) {
    auto buf_in = input.request();
    auto buf_w = packed_weight.request();

    if (buf_in.ndim != 2 && buf_in.ndim != 3) {
        throw std::runtime_error("Input must be 2D [M, K] or 3D [Batch, Seq, K]");
    }
    if (buf_w.ndim != 3) {
        throw std::runtime_error("Packed weight must be 3D [N/64, K/2, 32]");
    }

    int64_t batch = (buf_in.ndim == 3) ? buf_in.shape[0] : 1;
    int64_t seq = (buf_in.ndim == 3) ? buf_in.shape[1] : buf_in.shape[0];
    int64_t M = batch * seq;
    int64_t K = buf_in.shape[buf_in.ndim - 1];

    int64_t n_blocks = buf_w.shape[0];
    int64_t N = n_blocks * 64;

    if (buf_w.shape[1] * 2 != K || buf_w.shape[2] != 32) {
        throw std::runtime_error("Packed weight shape mismatch: expected [N/64, K/2, 32]");
    }

    py::array_t<float> result;
    if (buf_in.ndim == 3) {
        result = py::array_t<float>({batch, seq, N});
    } else {
        result = py::array_t<float>({M, N});
    }

    const float* ptr_in = static_cast<const float*>(buf_in.ptr);
    const int8_t* ptr_w = static_cast<const int8_t*>(buf_w.ptr);
    float* ptr_out = static_cast<float*>(result.request().ptr);

    for (int64_t m = 0; m < M; ++m) {
        bitlinear_tl1_avx2_single(
            ptr_in + m * K,
            ptr_w,
            ptr_out + m * N,
            K, N, scale
        );
    }

    if (bias.has_value()) {
        auto buf_b = bias->request();
        const float* ptr_b = static_cast<const float*>(buf_b.ptr);
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                ptr_out[m * N + n] += ptr_b[n];
            }
        }
    }

    return result;
}

// 3. Universal Linear Dispatch
py::array_t<float> dispatch_linear_cxx(
    py::array_t<float> input,
    py::object weight,
    std::optional<py::array_t<float>> bias = std::nullopt,
    const std::string& quant_scheme = "fp32",
    float scale = 1.0f
) {
    if (quant_scheme == "fp32" || quant_scheme == "fp16") {
        py::array_t<float> w_f32 = weight.cast<py::array_t<float>>();
        return dispatch_linear_fp32(input, w_f32, bias);
    } else if (quant_scheme == "classic_tl1" || quant_scheme == "classic_tl2") {
        py::array_t<int8_t> packed_w;
        if (py::isinstance<py::array_t<int8_t>>(weight)) {
            auto w_arr = weight.cast<py::array_t<int8_t>>();
            if (w_arr.ndim() == 2) {
                packed_w = pack_ternary_weights(w_arr);
            } else {
                packed_w = w_arr;
            }
        } else {
            // Convert float/double/int array to int8 ternary
            py::array_t<int8_t> w_tern = weight.cast<py::array_t<int8_t>>();
            packed_w = pack_ternary_weights(w_tern);
        }
        return dispatch_bitlinear_tl1(input, packed_w, scale, bias);
    } else {
        throw std::runtime_error("Unsupported quant_scheme: " + quant_scheme);
    }
}

//===----------------------------------------------------------------------===//
// 4. C++ KV-Cache & High-Performance Scaled Dot-Product Attention
//===----------------------------------------------------------------------===//

class KVCache {
public:
    int num_layers;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
    int q_dim;
    int kv_dim;
    int current_seq_len;
    std::string kv_mode; // "fp32" or "int8_fused"

    // FP32 Cache: k_cache_fp32[layer_idx], v_cache_fp32[layer_idx]
    std::vector<std::vector<float>> k_cache_fp32;
    std::vector<std::vector<float>> v_cache_fp32;

    // Fused INT8 Compressed Cache: 4x memory savings & bandwidth reduction
    std::vector<std::vector<int8_t>> k_cache_i8;
    std::vector<std::vector<int8_t>> v_cache_i8;
    std::vector<std::vector<float>> k_scales;
    std::vector<std::vector<float>> v_scales;

    // Preallocated per-head scores buffer
    std::vector<float> scores_buffer;

    KVCache(
        int num_layers,
        int num_q_heads,
        int num_kv_heads,
        int head_dim,
        int max_seq_len = 8192,
        const std::string& kv_mode = "int8_fused"
    ) : num_layers(num_layers), num_q_heads(num_q_heads), num_kv_heads(num_kv_heads),
        head_dim(head_dim), max_seq_len(max_seq_len),
        q_dim(num_q_heads * head_dim), kv_dim(num_kv_heads * head_dim),
        current_seq_len(0), kv_mode(kv_mode) {
        
        size_t cache_elements = static_cast<size_t>(num_kv_heads) * max_seq_len * head_dim;
        size_t scale_elements = static_cast<size_t>(num_kv_heads) * max_seq_len;

        if (kv_mode == "fp32") {
            k_cache_fp32.resize(num_layers, std::vector<float>(cache_elements, 0.0f));
            v_cache_fp32.resize(num_layers, std::vector<float>(cache_elements, 0.0f));
        } else {
            // int8_fused
            k_cache_i8.resize(num_layers, std::vector<int8_t>(cache_elements, 0));
            v_cache_i8.resize(num_layers, std::vector<int8_t>(cache_elements, 0));
            k_scales.resize(num_layers, std::vector<float>(scale_elements, 1.0f));
            v_scales.resize(num_layers, std::vector<float>(scale_elements, 1.0f));
        }

        scores_buffer.resize(static_cast<size_t>(num_q_heads) * max_seq_len, 0.0f);
    }

    void reset() {
        current_seq_len = 0;
        if (kv_mode == "fp32") {
            for (int l = 0; l < num_layers; ++l) {
                std::fill(k_cache_fp32[l].begin(), k_cache_fp32[l].end(), 0.0f);
                std::fill(v_cache_fp32[l].begin(), v_cache_fp32[l].end(), 0.0f);
            }
        } else {
            for (int l = 0; l < num_layers; ++l) {
                std::fill(k_cache_i8[l].begin(), k_cache_i8[l].end(), 0);
                std::fill(v_cache_i8[l].begin(), v_cache_i8[l].end(), 0);
                std::fill(k_scales[l].begin(), k_scales[l].end(), 1.0f);
                std::fill(v_scales[l].begin(), v_scales[l].end(), 1.0f);
            }
        }
    }

    int get_seq_len() const { return current_seq_len; }

    void increment_seq_len(int n = 1) {
        current_seq_len += n;
        if (current_seq_len >= max_seq_len) {
            throw std::runtime_error("KV-Cache overflow! Exceeded max_seq_len.");
        }
    }

    static void apply_rope(float* head_ptr, int head_dim, int pos) {
        int half_dim = head_dim / 2;
        for (int i = 0; i < half_dim; ++i) {
            float freq = 1.0f / std::pow(10000.0f, (2.0f * i) / static_cast<float>(head_dim));
            float angle = static_cast<float>(pos) * freq;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            float x1 = head_ptr[i];
            float x2 = head_ptr[i + half_dim];
            head_ptr[i] = x1 * cos_val - x2 * sin_val;
            head_ptr[i + half_dim] = x1 * sin_val + x2 * cos_val;
        }
    }

    // Appends new K and V (with in-place RoPE & fused quantization), then computes GQA Attention
    py::array_t<float> forward_attention(
        int layer_idx,
        py::array_t<float> q_arr,
        py::array_t<float> k_arr,
        py::array_t<float> v_arr
    ) {
        if (layer_idx < 0 || layer_idx >= num_layers) {
            throw std::out_of_range("Invalid layer index");
        }

        auto buf_q = q_arr.request();
        auto buf_k = k_arr.request();
        auto buf_v = v_arr.request();

        if (buf_q.size != q_dim || buf_k.size != kv_dim || buf_v.size != kv_dim) {
            throw std::runtime_error("Tensor dimension mismatch in forward_attention");
        }

        float* q_ptr = static_cast<float*>(buf_q.ptr);
        float* k_ptr = static_cast<float*>(buf_k.ptr);
        const float* v_src = static_cast<const float*>(buf_v.ptr);

        // In-place zero-copy RoPE
        for (int h = 0; h < num_q_heads; ++h) {
            apply_rope(q_ptr + h * head_dim, head_dim, current_seq_len);
        }
        for (int h = 0; h < num_kv_heads; ++h) {
            apply_rope(k_ptr + h * head_dim, head_dim, current_seq_len);
        }

        int t_idx = current_seq_len;
        int seq_len = current_seq_len + 1;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        int group_size = num_q_heads / num_kv_heads;

        auto out_arr = py::array_t<float>({1, 1, q_dim});
        float* out_ptr = static_cast<float*>(out_arr.request().ptr);
        float* scores_base = scores_buffer.data();

        if (kv_mode == "fp32") {
            float* layer_k = k_cache_fp32[layer_idx].data();
            float* layer_v = v_cache_fp32[layer_idx].data();

            for (int h = 0; h < num_kv_heads; ++h) {
                float* k_dst = layer_k + (h * max_seq_len + t_idx) * head_dim;
                float* v_dst = layer_v + (h * max_seq_len + t_idx) * head_dim;
                const float* k_val = k_ptr + h * head_dim;
                const float* v_val = v_src + h * head_dim;
                std::memcpy(k_dst, k_val, head_dim * sizeof(float));
                std::memcpy(v_dst, v_val, head_dim * sizeof(float));
            }

            #pragma omp parallel for schedule(static)
            for (int q_h = 0; q_h < num_q_heads; ++q_h) {
                int kv_h = q_h / group_size;
                const float* q_head = q_ptr + q_h * head_dim;
                const float* k_head_base = layer_k + kv_h * max_seq_len * head_dim;
                const float* v_head_base = layer_v + kv_h * max_seq_len * head_dim;
                float* out_head = out_ptr + q_h * head_dim;
                float* scores = scores_base + q_h * max_seq_len;
                float max_score = -1e9f;

                for (int t = 0; t < seq_len; ++t) {
                    const float* k_t = k_head_base + t * head_dim;
                    __m256 dot_vec = _mm256_setzero_ps();
                    int d = 0;
                    for (; d + 8 <= head_dim; d += 8) {
                        __m256 q_v = _mm256_loadu_ps(q_head + d);
                        __m256 k_v = _mm256_loadu_ps(k_t + d);
                        dot_vec = _mm256_fmadd_ps(q_v, k_v, dot_vec);
                    }
                    alignas(32) float temp[8];
                    _mm256_store_ps(temp, dot_vec);
                    float dot = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
                    for (; d < head_dim; ++d) {
                        dot += q_head[d] * k_t[d];
                    }
                    float s = dot * scale;
                    scores[t] = s;
                    if (s > max_score) max_score = s;
                }

                float sum_exp = 0.0f;
                for (int t = 0; t < seq_len; ++t) {
                    scores[t] = std::exp(scores[t] - max_score);
                    sum_exp += scores[t];
                }
                float inv_sum = 1.0f / (sum_exp > 1e-9f ? sum_exp : 1.0f);
                for (int t = 0; t < seq_len; ++t) {
                    scores[t] *= inv_sum;
                }

                // Out = Score * V (Accumulated in 16 YMM registers)
                __m256 o_acc[16];
                for (int i = 0; i < 16; ++i) o_acc[i] = _mm256_setzero_ps();

                for (int t = 0; t < seq_len; ++t) {
                    __m256 w_v = _mm256_set1_ps(scores[t]);
                    const float* v_t = v_head_base + t * head_dim;
                    for (int i = 0; i < 16; ++i) {
                        __m256 v_val = _mm256_loadu_ps(v_t + i * 8);
                        o_acc[i] = _mm256_fmadd_ps(w_v, v_val, o_acc[i]);
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    _mm256_storeu_ps(out_head + i * 8, o_acc[i]);
                }
            }
        } else {
            // === Fused INT8 Compressed KV-Cache ===
            int8_t* layer_k = k_cache_i8[layer_idx].data();
            int8_t* layer_v = v_cache_i8[layer_idx].data();
            float* l_k_scales = k_scales[layer_idx].data();
            float* l_v_scales = v_scales[layer_idx].data();

            // 1. In-Register Fused Symmetric Quantization for K and V
            for (int h = 0; h < num_kv_heads; ++h) {
                const float* k_src = k_ptr + h * head_dim;
                const float* v_src_h = v_src + h * head_dim;
                int8_t* k_dst = layer_k + (h * max_seq_len + t_idx) * head_dim;
                int8_t* v_dst = layer_v + (h * max_seq_len + t_idx) * head_dim;

                float amax_k = 0.0f;
                float amax_v = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    amax_k = std::max(amax_k, std::abs(k_src[d]));
                    amax_v = std::max(amax_v, std::abs(v_src_h[d]));
                }
                float s_k = std::max(amax_k / 127.0f, 1e-8f);
                float s_v = std::max(amax_v / 127.0f, 1e-8f);
                float inv_s_k = 1.0f / s_k;
                float inv_s_v = 1.0f / s_v;

                l_k_scales[h * max_seq_len + t_idx] = s_k;
                l_v_scales[h * max_seq_len + t_idx] = s_v;

                for (int d = 0; d < head_dim; ++d) {
                    float q_k = std::round(k_src[d] * inv_s_k);
                    float q_v = std::round(v_src_h[d] * inv_s_v);
                    k_dst[d] = static_cast<int8_t>(std::clamp(q_k, -127.0f, 127.0f));
                    v_dst[d] = static_cast<int8_t>(std::clamp(q_v, -127.0f, 127.0f));
                }
            }

            // 2. High-Performance AVX2 Fused Dequantized Attention
            #pragma omp parallel for schedule(static)
            for (int q_h = 0; q_h < num_q_heads; ++q_h) {
                int kv_h = q_h / group_size;
                const float* q_head = q_ptr + q_h * head_dim;
                const int8_t* k_head_base = layer_k + kv_h * max_seq_len * head_dim;
                const int8_t* v_head_base = layer_v + kv_h * max_seq_len * head_dim;
                const float* k_head_scales = l_k_scales + kv_h * max_seq_len;
                const float* v_head_scales = l_v_scales + kv_h * max_seq_len;
                float* out_head = out_ptr + q_h * head_dim;
                float* scores = scores_base + q_h * max_seq_len;
                float max_score = -1e9f;

                // Scaled Dot Product (Q @ K^T)
                for (int t = 0; t < seq_len; ++t) {
                    const int8_t* k_t = k_head_base + t * head_dim;
                    float s_k = k_head_scales[t];

                    __m256 dot_vec0 = _mm256_setzero_ps();
                    __m256 dot_vec1 = _mm256_setzero_ps();
                    int d = 0;
                    for (; d + 16 <= head_dim; d += 16) {
                        __m256 q0 = _mm256_loadu_ps(q_head + d);
                        __m128i k_raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(k_t + d));
                        __m256 k_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(k_raw0));
                        dot_vec0 = _mm256_fmadd_ps(q0, k_f0, dot_vec0);

                        __m256 q1 = _mm256_loadu_ps(q_head + d + 8);
                        __m128i k_raw1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(k_t + d + 8));
                        __m256 k_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(k_raw1));
                        dot_vec1 = _mm256_fmadd_ps(q1, k_f1, dot_vec1);
                    }
                    __m256 dot_vec = _mm256_add_ps(dot_vec0, dot_vec1);
                    alignas(32) float temp[8];
                    _mm256_store_ps(temp, dot_vec);
                    float raw_dot = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
                    for (; d < head_dim; ++d) {
                        raw_dot += q_head[d] * static_cast<float>(k_t[d]);
                    }

                    float s = raw_dot * (s_k * scale);
                    scores[t] = s;
                    if (s > max_score) max_score = s;
                }

                // Softmax
                float sum_exp = 0.0f;
                for (int t = 0; t < seq_len; ++t) {
                    scores[t] = std::exp(scores[t] - max_score);
                    sum_exp += scores[t];
                }
                float inv_sum = 1.0f / (sum_exp > 1e-9f ? sum_exp : 1.0f);
                for (int t = 0; t < seq_len; ++t) {
                    scores[t] *= inv_sum;
                }

                // Out = Scores @ V (Accumulated in 16 YMM registers)
                __m256 o_acc[16];
                for (int i = 0; i < 16; ++i) o_acc[i] = _mm256_setzero_ps();

                for (int t = 0; t < seq_len; ++t) {
                    float s_v = v_head_scales[t];
                    float eff_w = scores[t] * s_v;
                    __m256 w_v = _mm256_set1_ps(eff_w);
                    const int8_t* v_t = v_head_base + t * head_dim;

                    for (int i = 0; i < 16; ++i) {
                        __m128i v_raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(v_t + i * 8));
                        __m256 v_val = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_raw));
                        o_acc[i] = _mm256_fmadd_ps(w_v, v_val, o_acc[i]);
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    _mm256_storeu_ps(out_head + i * 8, o_acc[i]);
                }
            }
        }

        return out_arr;
    }
};

//===----------------------------------------------------------------------===//
// 5. ExecutionContext Class for Full-Model Autoregressive Inference
//===----------------------------------------------------------------------===//

struct LayerWeights {
    py::object q_w;
    py::object k_w;
    py::object v_w;
    py::object out_w;
    py::object gate_w;
    py::object up_w;
    py::object down_w;

    py::object in_norm_w;
    py::object attn_sub_norm_w;
    py::object post_norm_w;
    py::object ffn_sub_norm_w;

    float q_scale = 1.0f;
    float k_scale = 1.0f;
    float v_scale = 1.0f;
    float out_scale = 1.0f;
    float gate_scale = 1.0f;
    float up_scale = 1.0f;
    float down_scale = 1.0f;
};

class ExecutionContext {
public:
    int hidden_size;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int num_layers;
    int max_seq_len;
    std::string quant_scheme;
    KVCache kv_cache;
    std::vector<LayerWeights> layers;
    py::object final_norm_w;

    ExecutionContext(
        int hidden_size,
        int num_q_heads,
        int num_kv_heads,
        int head_dim,
        int num_layers,
        int max_seq_len = 8192,
        const std::string& quant_scheme = "classic_tl1",
        const std::string& kv_mode = "int8_fused"
    ) : hidden_size(hidden_size), num_q_heads(num_q_heads), num_kv_heads(num_kv_heads),
        head_dim(head_dim), num_layers(num_layers), max_seq_len(max_seq_len),
        quant_scheme(quant_scheme),
        kv_cache(num_layers, num_q_heads, num_kv_heads, head_dim, max_seq_len, kv_mode) {
        
        layers.resize(num_layers);
    }

    void reset() {
        kv_cache.reset();
    }

    int get_seq_len() const {
        return kv_cache.get_seq_len();
    }

    void set_final_norm(py::object w) {
        final_norm_w = w;
    }

    void set_layer_full(
        int layer_idx,
        py::object q_w, float q_scale,
        py::object k_w, float k_scale,
        py::object v_w, float v_scale,
        py::object out_w, float out_scale,
        py::object gate_w, float gate_scale,
        py::object up_w, float up_scale,
        py::object down_w, float down_scale,
        py::object in_norm_w = py::none(),
        py::object attn_sub_norm_w = py::none(),
        py::object post_norm_w = py::none(),
        py::object ffn_sub_norm_w = py::none()
    ) {
        if (layer_idx < 0 || layer_idx >= num_layers) {
            throw std::out_of_range("Invalid layer index");
        }

        auto pack_if_tl1 = [&](py::object w) -> py::object {
            if (w.is_none()) return w;
            if (quant_scheme == "classic_tl1" || quant_scheme == "classic_tl2") {
                if (py::isinstance<py::array_t<int8_t>>(w)) {
                    auto arr = w.cast<py::array_t<int8_t>>();
                    if (arr.ndim() == 2) {
                        return pack_ternary_weights(arr);
                    }
                }
            }
            return w;
        };

        layers[layer_idx].q_w = pack_if_tl1(q_w);
        layers[layer_idx].k_w = pack_if_tl1(k_w);
        layers[layer_idx].v_w = pack_if_tl1(v_w);
        layers[layer_idx].out_w = pack_if_tl1(out_w);
        layers[layer_idx].gate_w = pack_if_tl1(gate_w);
        layers[layer_idx].up_w = pack_if_tl1(up_w);
        layers[layer_idx].down_w = pack_if_tl1(down_w);

        layers[layer_idx].in_norm_w = in_norm_w;
        layers[layer_idx].attn_sub_norm_w = attn_sub_norm_w;
        layers[layer_idx].post_norm_w = post_norm_w;
        layers[layer_idx].ffn_sub_norm_w = ffn_sub_norm_w;

        layers[layer_idx].q_scale = q_scale;
        layers[layer_idx].k_scale = k_scale;
        layers[layer_idx].v_scale = v_scale;
        layers[layer_idx].out_scale = out_scale;
        layers[layer_idx].gate_scale = gate_scale;
        layers[layer_idx].up_scale = up_scale;
        layers[layer_idx].down_scale = down_scale;
    }

    // Weighted RMSNorm helper
    py::array_t<float> rms_norm_weighted(py::array_t<float> x, py::object weight_obj, float eps = 1e-5f) {
        auto buf = x.request();
        const float* src = static_cast<const float*>(buf.ptr);
        py::ssize_t dim = buf.size;
        auto out = py::array_t<float>(std::vector<py::ssize_t>{1, 1, dim});
        float* dst = static_cast<float*>(out.request().ptr);

        float sum_sq = 0.0f;
        #pragma omp simd reduction(+:sum_sq)
        for (py::ssize_t i = 0; i < dim; ++i) {
            sum_sq += src[i] * src[i];
        }
        float inv_rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(dim)) + eps);

        if (!weight_obj.is_none()) {
            auto w_buf = weight_obj.cast<py::array_t<float>>().request();
            const float* w_ptr = static_cast<const float*>(w_buf.ptr);
            #pragma omp simd
            for (py::ssize_t i = 0; i < dim; ++i) {
                dst[i] = src[i] * inv_rms * w_ptr[i];
            }
        } else {
            #pragma omp simd
            for (py::ssize_t i = 0; i < dim; ++i) {
                dst[i] = src[i] * inv_rms;
            }
        }
        return out;
    }

    // Runs a full transformer layer in C++ without Python overhead
    py::array_t<float> forward_layer(int layer_idx, py::array_t<float> x) {
        if (layer_idx < 0 || layer_idx >= num_layers) {
            throw std::out_of_range("Invalid layer index");
        }
        const auto& lw = layers[layer_idx];

        // 1. Pre-Layer RMSNorm (input_layernorm)
        auto norm_x = rms_norm_weighted(x, lw.in_norm_w);

        // 2. Compute Q, K, V projections using native AVX2 SIMD
        auto q = dispatch_linear_cxx(norm_x, lw.q_w, std::nullopt, quant_scheme, lw.q_scale);
        auto k = dispatch_linear_cxx(norm_x, lw.k_w, std::nullopt, quant_scheme, lw.k_scale);
        auto v = dispatch_linear_cxx(norm_x, lw.v_w, std::nullopt, quant_scheme, lw.v_scale);

        // 3. Fused Scaled Dot-Product Attention & KV Cache Update in C++
        auto attn_raw = kv_cache.forward_attention(layer_idx, q, k, v);

        // 4. Attention Sub-LayerNorm
        auto attn_sub_norm = rms_norm_weighted(attn_raw, lw.attn_sub_norm_w);

        // 5. Output Projection
        auto out = dispatch_linear_cxx(attn_sub_norm, lw.out_w, std::nullopt, quant_scheme, lw.out_scale);

        // 6. Residual Connection (h1 = x + out)
        auto h1 = py::array_t<float>(std::vector<py::ssize_t>{1, 1, static_cast<py::ssize_t>(hidden_size)});
        const float* x_ptr = static_cast<const float*>(x.request().ptr);
        const float* out_ptr = static_cast<const float*>(out.request().ptr);
        float* h1_ptr = static_cast<float*>(h1.request().ptr);

        #pragma omp simd
        for (int i = 0; i < hidden_size; ++i) {
            h1_ptr[i] = x_ptr[i] + out_ptr[i];
        }

        // 7. MLP Block (if present)
        if (!lw.gate_w.is_none() && !lw.up_w.is_none() && !lw.down_w.is_none()) {
            // Post-Attention RMSNorm
            auto post_norm_h = rms_norm_weighted(h1, lw.post_norm_w);

            auto gate = dispatch_linear_cxx(post_norm_h, lw.gate_w, std::nullopt, quant_scheme, lw.gate_scale);
            auto up = dispatch_linear_cxx(post_norm_h, lw.up_w, std::nullopt, quant_scheme, lw.up_scale);

            auto g_buf = gate.request();
            auto u_buf = up.request();
            py::ssize_t ffn_dim = g_buf.size;
            const float* g_ptr = static_cast<const float*>(g_buf.ptr);
            const float* u_ptr = static_cast<const float*>(u_buf.ptr);

            auto act_mult = py::array_t<float>(std::vector<py::ssize_t>{1, 1, ffn_dim});
            float* m_ptr = static_cast<float*>(act_mult.request().ptr);

            // Gated activation: relu(gate)^2 * up
            #pragma omp simd
            for (py::ssize_t i = 0; i < ffn_dim; ++i) {
                float r = std::max(0.0f, g_ptr[i]);
                m_ptr[i] = (r * r) * u_ptr[i];
            }

            // FFN Sub-LayerNorm
            auto ffn_sub_norm = rms_norm_weighted(act_mult, lw.ffn_sub_norm_w);

            // Down projection
            auto down = dispatch_linear_cxx(ffn_sub_norm, lw.down_w, std::nullopt, quant_scheme, lw.down_scale);
            const float* down_ptr = static_cast<const float*>(down.request().ptr);

            // Residual connection (h1 + down)
            auto res = py::array_t<float>(std::vector<py::ssize_t>{1, 1, static_cast<py::ssize_t>(hidden_size)});
            float* res_ptr = static_cast<float*>(res.request().ptr);
            #pragma omp simd
            for (int i = 0; i < hidden_size; ++i) {
                res_ptr[i] = h1_ptr[i] + down_ptr[i];
            }
            return res;
        }

        return h1;
    }

    // Runs a full token generation step across all layers
    py::array_t<float> forward_step(py::array_t<float> x) {
        py::array_t<float> cur_x = x;
        for (int l = 0; l < num_layers; ++l) {
            cur_x = forward_layer(l, cur_x);
        }
        kv_cache.increment_seq_len(1);
        return cur_x;
    }

    // Fast LM Head Logits calculation using AVX2 SIMD dot products: FP32 or INT8
    py::array_t<float> compute_logits(
        py::array_t<float> x,
        py::object embed_weights_obj,
        py::object embed_scales_obj = py::none()
    ) {
        auto final_x = rms_norm_weighted(x, final_norm_w);
        const float* x_ptr = static_cast<const float*>(final_x.request().ptr);

        if (py::isinstance<py::array_t<float>>(embed_weights_obj)) {
            auto ew_buf = embed_weights_obj.cast<py::array_t<float>>().request();
            py::ssize_t vocab_size = ew_buf.shape[0];
            py::ssize_t h_dim = ew_buf.shape[1];
            const float* w_ptr = static_cast<const float*>(ew_buf.ptr);

            auto logits_arr = py::array_t<float>(std::vector<py::ssize_t>{1, 1, vocab_size});
            float* logits_ptr = static_cast<float*>(logits_arr.request().ptr);

            #pragma omp parallel for schedule(static)
            for (py::ssize_t v = 0; v < vocab_size; ++v) {
                const float* row = w_ptr + v * h_dim;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                py::ssize_t i = 0;
                for (; i + 32 <= h_dim; i += 32) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + i), _mm256_loadu_ps(row + i), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + i + 8), _mm256_loadu_ps(row + i + 8), acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + i + 16), _mm256_loadu_ps(row + i + 16), acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + i + 24), _mm256_loadu_ps(row + i + 24), acc3);
                }
                __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                for (; i + 8 <= h_dim; i += 8) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + i), _mm256_loadu_ps(row + i), acc);
                }
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, acc);
                float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

                for (; i < h_dim; ++i) {
                    sum += x_ptr[i] * row[i];
                }
                logits_ptr[v] = sum;
            }
            return logits_arr;
        }

        auto ew_buf = embed_weights_obj.cast<py::array_t<int8_t>>().request();
        auto es_buf = embed_scales_obj.cast<py::array_t<float>>().request();

        py::ssize_t vocab_size = ew_buf.shape[0];
        py::ssize_t h_dim = ew_buf.shape[1];
        const int8_t* w_ptr = static_cast<const int8_t*>(ew_buf.ptr);
        const float* scales_ptr = static_cast<const float*>(es_buf.ptr);

        auto logits_arr = py::array_t<float>(std::vector<py::ssize_t>{1, 1, vocab_size});
        float* logits_ptr = static_cast<float*>(logits_arr.request().ptr);

        #pragma omp parallel for schedule(static)
        for (py::ssize_t v = 0; v < vocab_size; ++v) {
            const int8_t* row = w_ptr + v * h_dim;
            __m256 acc = _mm256_setzero_ps();
            py::ssize_t i = 0;
            for (; i + 8 <= h_dim; i += 8) {
                __m256 va = _mm256_loadu_ps(x_ptr + i);
                __m128i vb_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + i));
                __m256i vb_i32 = _mm256_cvtepi8_epi32(vb_i8);
                __m256 vb_f32 = _mm256_cvtepi32_ps(vb_i32);
                acc = _mm256_fmadd_ps(va, vb_f32, acc);
            }
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, acc);
            float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

            for (; i < h_dim; ++i) {
                sum += x_ptr[i] * static_cast<float>(row[i]);
            }
            logits_ptr[v] = sum * scales_ptr[v];
        }

        return logits_arr;
    }

    // Fast Embedding Lookup
    py::array_t<float> embedding_lookup(
        int token_id,
        py::object embed_weights_obj,
        py::object embed_scales_obj = py::none()
    ) {
        if (py::isinstance<py::array_t<float>>(embed_weights_obj)) {
            auto ew_buf = embed_weights_obj.cast<py::array_t<float>>().request();
            py::ssize_t h_dim = ew_buf.shape[1];
            const float* row = static_cast<const float*>(ew_buf.ptr) + token_id * h_dim;

            auto out = py::array_t<float>(std::vector<py::ssize_t>{1, 1, h_dim});
            float* out_ptr = static_cast<float*>(out.request().ptr);
            std::memcpy(out_ptr, row, h_dim * sizeof(float));
            return out;
        }

        auto ew_buf = embed_weights_obj.cast<py::array_t<int8_t>>().request();
        auto es_buf = embed_scales_obj.cast<py::array_t<float>>().request();

        py::ssize_t h_dim = ew_buf.shape[1];
        const int8_t* row = static_cast<const int8_t*>(ew_buf.ptr) + token_id * h_dim;
        float scale = static_cast<const float*>(es_buf.ptr)[token_id];

        auto out = py::array_t<float>(std::vector<py::ssize_t>{1, 1, h_dim});
        float* out_ptr = static_cast<float*>(out.request().ptr);

        #pragma omp simd
        for (py::ssize_t i = 0; i < h_dim; ++i) {
            out_ptr[i] = static_cast<float>(row[i]) * scale;
        }
        return out;
    }
};

//===----------------------------------------------------------------------===//
// 6. Pybind11 Module Definitions
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(tenzo_runtime, m) {
    m.doc() = "Tenzo C++ Runtime Native AVX2 Engine & KV-Cache Module";

    m.def("pack_ternary_weights", &pack_ternary_weights, "Pack [N, K] ternary weights into [N/64, K/2, 32] TL1 format",
          py::arg("tern_weights"));

    m.def("dispatch_linear_fp32", &dispatch_linear_fp32, "Dispatch high-performance AVX2 FP32 GEMM",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("relu") = false);

    m.def("dispatch_bitlinear_tl1", &dispatch_bitlinear_tl1, "Dispatch AVX2 TL1 256-bit BitLinear SIMD micro-kernel",
          py::arg("input"), py::arg("packed_weight"), py::arg("scale") = 1.0f, py::arg("bias") = py::none());

    m.def("dispatch_linear_cxx", &dispatch_linear_cxx, "Universal Linear Dispatcher (FP32, BitLinear TL1)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("quant_scheme") = "fp32", py::arg("scale") = 1.0f);

    py::class_<KVCache>(m, "KVCache")
        .def(py::init<int, int, int, int, int, const std::string&>(),
             py::arg("num_layers"), py::arg("num_q_heads"), py::arg("num_kv_heads"), py::arg("head_dim"),
             py::arg("max_seq_len") = 8192, py::arg("kv_mode") = "int8_fused")
        .def_readonly("max_seq_len", &KVCache::max_seq_len)
        .def("reset", &KVCache::reset)
        .def("get_seq_len", &KVCache::get_seq_len)
        .def("increment_seq_len", &KVCache::increment_seq_len, py::arg("n") = 1)
        .def("forward_attention", &KVCache::forward_attention,
             py::arg("layer_idx"), py::arg("q"), py::arg("k"), py::arg("v"));

    py::class_<ExecutionContext>(m, "ExecutionContext")
        .def(py::init<int, int, int, int, int, int, const std::string&, const std::string&>(),
             py::arg("hidden_size"), py::arg("num_q_heads"), py::arg("num_kv_heads"), py::arg("head_dim"),
             py::arg("num_layers"), py::arg("max_seq_len") = 8192, py::arg("quant_scheme") = "classic_tl1",
             py::arg("kv_mode") = "int8_fused")
        .def_readonly("max_seq_len", &ExecutionContext::max_seq_len)
        .def("reset", &ExecutionContext::reset)
        .def("get_seq_len", &ExecutionContext::get_seq_len)
        .def("set_final_norm", &ExecutionContext::set_final_norm, py::arg("w"))
        .def("set_layer_full", &ExecutionContext::set_layer_full,
             py::arg("layer_idx"),
             py::arg("q_w"), py::arg("q_scale"),
             py::arg("k_w"), py::arg("k_scale"),
             py::arg("v_w"), py::arg("v_scale"),
             py::arg("out_w"), py::arg("out_scale"),
             py::arg("gate_w"), py::arg("gate_scale"),
             py::arg("up_w"), py::arg("up_scale"),
             py::arg("down_w"), py::arg("down_scale"),
             py::arg("in_norm_w") = py::none(),
             py::arg("attn_sub_norm_w") = py::none(),
             py::arg("post_norm_w") = py::none(),
             py::arg("ffn_sub_norm_w") = py::none())
        .def("forward_layer", &ExecutionContext::forward_layer, py::arg("layer_idx"), py::arg("x"))
        .def("forward_step", &ExecutionContext::forward_step, py::arg("x"))
        .def("compute_logits", &ExecutionContext::compute_logits,
             py::arg("x"), py::arg("embed_weights"), py::arg("embed_scales") = py::none())
        .def("embedding_lookup", &ExecutionContext::embedding_lookup,
             py::arg("token_id"), py::arg("embed_weights"), py::arg("embed_scales") = py::none());
}

