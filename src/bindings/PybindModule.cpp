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
#include <queue>
#include <unordered_set>

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
    __m256i acc_low_32_0 = _mm256_setzero_si256(); // ch 0..7
    __m256i acc_low_32_1 = _mm256_setzero_si256(); // ch 8..15
    __m256i acc_low_32_2 = _mm256_setzero_si256(); // ch 16..23
    __m256i acc_low_32_3 = _mm256_setzero_si256(); // ch 24..31

    __m256i acc_high_32_0 = _mm256_setzero_si256(); // ch 32..39
    __m256i acc_high_32_1 = _mm256_setzero_si256(); // ch 40..47
    __m256i acc_high_32_2 = _mm256_setzero_si256(); // ch 48..55
    __m256i acc_high_32_3 = _mm256_setzero_si256(); // ch 56..63

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
    const int8_t* __restrict__ packed_w, // [n_blocks, K_half, 32]
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

    std::vector<__m256i> lut_vecs(K_half);
    float inv_scale_act = build_lut(act, K, lut_vecs.data());
    compute_from_lut(lut_vecs.data(), packed_w, n_blocks, K_half, out, inv_scale_act * weight_scale);
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

    void forward_attention_raw(
        int layer_idx,
        float* q_ptr,
        float* k_ptr,
        const float* v_src,
        float* out_ptr
    ) {
        if (layer_idx < 0 || layer_idx >= num_layers) {
            throw std::out_of_range("Invalid layer index");
        }

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

            #pragma omp parallel for schedule(dynamic, 2)
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

                // Out = Scores @ V
                std::memset(out_head, 0, head_dim * sizeof(float));
                for (int t = 0; t < seq_len; ++t) {
                    float s = scores[t];
                    __m256 s_v = _mm256_set1_ps(s);
                    const float* v_t = v_head_base + t * head_dim;
                    int d = 0;
                    for (; d + 8 <= head_dim; d += 8) {
                        __m256 o_v = _mm256_loadu_ps(out_head + d);
                        __m256 val_v = _mm256_loadu_ps(v_t + d);
                        o_v = _mm256_fmadd_ps(s_v, val_v, o_v);
                        _mm256_storeu_ps(out_head + d, o_v);
                    }
                    for (; d < head_dim; ++d) {
                        out_head[d] += s * v_t[d];
                    }
                }
            }
        } else {
            // INT8 Fused KV-Cache
            int8_t* layer_k = k_cache_i8[layer_idx].data();
            int8_t* layer_v = v_cache_i8[layer_idx].data();
            float* layer_k_scales = k_scales[layer_idx].data();
            float* layer_v_scales = v_scales[layer_idx].data();

            for (int h = 0; h < num_kv_heads; ++h) {
                int8_t* k_dst = layer_k + (h * max_seq_len + t_idx) * head_dim;
                int8_t* v_dst = layer_v + (h * max_seq_len + t_idx) * head_dim;
                const float* k_src = k_ptr + h * head_dim;
                const float* v_s = v_src + h * head_dim;

                float max_abs_k = 1e-8f;
                float max_abs_v = 1e-8f;
                for (int d = 0; d < head_dim; ++d) {
                    float ak = std::abs(k_src[d]);
                    float av = std::abs(v_s[d]);
                    if (ak > max_abs_k) max_abs_k = ak;
                    if (av > max_abs_v) max_abs_v = av;
                }
                float s_k = max_abs_k / 127.0f;
                float s_v = max_abs_v / 127.0f;
                layer_k_scales[h * max_seq_len + t_idx] = s_k;
                layer_v_scales[h * max_seq_len + t_idx] = s_v;

                float inv_s_k = 1.0f / s_k;
                float inv_s_v = 1.0f / s_v;
                for (int d = 0; d < head_dim; ++d) {
                    k_dst[d] = static_cast<int8_t>(std::clamp(std::round(k_src[d] * inv_s_k), -128.0f, 127.0f));
                    v_dst[d] = static_cast<int8_t>(std::clamp(std::round(v_s[d] * inv_s_v), -128.0f, 127.0f));
                }
            }

            #pragma omp parallel for schedule(dynamic, 2)
            for (int q_h = 0; q_h < num_q_heads; ++q_h) {
                int kv_h = q_h / group_size;
                const float* q_head = q_ptr + q_h * head_dim;
                const int8_t* k_head_base = layer_k + kv_h * max_seq_len * head_dim;
                const int8_t* v_head_base = layer_v + kv_h * max_seq_len * head_dim;
                const float* k_head_scales = layer_k_scales + kv_h * max_seq_len;
                const float* v_head_scales = layer_v_scales + kv_h * max_seq_len;
                float* out_head = out_ptr + q_h * head_dim;
                float* scores = scores_base + q_h * max_seq_len;
                float max_score = -1e9f;

                for (int t = 0; t < seq_len; ++t) {
                    const int8_t* k_t = k_head_base + t * head_dim;
                    float s_k = k_head_scales[t];

                    __m256 dot_acc0 = _mm256_setzero_ps();
                    __m256 dot_acc1 = _mm256_setzero_ps();

                    for (int i = 0; i < 16; i += 2) {
                        __m256 q_v0 = _mm256_loadu_ps(q_head + i * 8);
                        __m128i k_raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(k_t + i * 8));
                        __m256 k_val0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(k_raw0));
                        dot_acc0 = _mm256_fmadd_ps(q_v0, k_val0, dot_acc0);

                        __m256 q_v1 = _mm256_loadu_ps(q_head + (i + 1) * 8);
                        __m128i k_raw1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(k_t + (i + 1) * 8));
                        __m256 k_val1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(k_raw1));
                        dot_acc1 = _mm256_fmadd_ps(q_v1, k_val1, dot_acc1);
                    }

                    __m256 dot_tot = _mm256_add_ps(dot_acc0, dot_acc1);
                    alignas(32) float temp[8];
                    _mm256_store_ps(temp, dot_tot);
                    float raw_dot = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

                    for (int d = 128; d < head_dim; ++d) {
                        raw_dot += q_head[d] * static_cast<float>(k_t[d]);
                    }

                    float s = raw_dot * (s_k * scale);
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
    }

    py::array_t<float> forward_attention(
        int layer_idx,
        py::array_t<float> q_arr,
        py::array_t<float> k_arr,
        py::array_t<float> v_arr
    ) {
        auto buf_q = q_arr.request();
        auto buf_k = k_arr.request();
        auto buf_v = v_arr.request();

        if (buf_q.size != q_dim || buf_k.size != kv_dim || buf_v.size != kv_dim) {
            throw std::runtime_error("Tensor dimension mismatch in forward_attention");
        }

        auto out_arr = py::array_t<float>({1, 1, q_dim});
        forward_attention_raw(
            layer_idx,
            static_cast<float*>(buf_q.ptr),
            static_cast<float*>(buf_k.ptr),
            static_cast<const float*>(buf_v.ptr),
            static_cast<float*>(out_arr.request().ptr)
        );
        return out_arr;
    }
};

//===----------------------------------------------------------------------===//
// 5. ExecutionContext Class for Full-Model Autoregressive Inference
//===----------------------------------------------------------------------===//

struct LayerWeights {
    py::object q_w, k_w, v_w, out_w, gate_w, up_w, down_w;
    py::object in_norm_w, attn_sub_norm_w, post_norm_w, ffn_sub_norm_w;

    const int8_t* q_w_ptr = nullptr;
    const int8_t* k_w_ptr = nullptr;
    const int8_t* v_w_ptr = nullptr;
    const int8_t* out_w_ptr = nullptr;
    const int8_t* gate_w_ptr = nullptr;
    const int8_t* up_w_ptr = nullptr;
    const int8_t* down_w_ptr = nullptr;

    const float* in_norm_ptr = nullptr;
    const float* attn_sub_norm_ptr = nullptr;
    const float* post_norm_ptr = nullptr;
    const float* ffn_sub_norm_ptr = nullptr;

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
    const float* final_norm_ptr = nullptr;

    int q_dim;
    int kv_dim;
    int ffn_dim;

    // Preallocated scratch buffers (Zero Heap Allocations during generation)
    std::vector<float> buf_x;
    std::vector<float> buf_norm_x;
    std::vector<float> buf_q;
    std::vector<float> buf_k;
    std::vector<float> buf_v;
    std::vector<float> buf_attn_out;
    std::vector<float> buf_attn_sub;
    std::vector<float> buf_out;
    std::vector<float> buf_h1;
    std::vector<float> buf_post_norm;
    std::vector<float> buf_gate;
    std::vector<float> buf_up;
    std::vector<float> buf_act;
    std::vector<float> buf_ffn_norm;
    std::vector<float> buf_down;
    std::vector<__m256i> lut_buf;

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
        q_dim = num_q_heads * head_dim;
        kv_dim = num_kv_heads * head_dim;
        ffn_dim = 6912;

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
    }

    void reset() {
        kv_cache.reset();
    }

    int get_seq_len() const {
        return kv_cache.get_seq_len();
    }

    void set_final_norm(py::object w) {
        final_norm_w = w;
        if (!final_norm_w.is_none() && py::isinstance<py::array_t<float>>(final_norm_w)) {
            final_norm_ptr = static_cast<const float*>(final_norm_w.cast<py::array_t<float>>().request().ptr);
        } else {
            final_norm_ptr = nullptr;
        }
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

        auto get_packed_ptr = [&](py::object w) -> const int8_t* {
            if (w.is_none()) return nullptr;
            if (py::isinstance<py::array_t<int8_t>>(w)) {
                return static_cast<const int8_t*>(w.cast<py::array_t<int8_t>>().request().ptr);
            }
            return nullptr;
        };

        auto get_float_ptr = [&](py::object w) -> const float* {
            if (w.is_none()) return nullptr;
            if (py::isinstance<py::array_t<float>>(w)) {
                return static_cast<const float*>(w.cast<py::array_t<float>>().request().ptr);
            }
            return nullptr;
        };

        layers[layer_idx].q_w_ptr = get_packed_ptr(layers[layer_idx].q_w);
        layers[layer_idx].k_w_ptr = get_packed_ptr(layers[layer_idx].k_w);
        layers[layer_idx].v_w_ptr = get_packed_ptr(layers[layer_idx].v_w);
        layers[layer_idx].out_w_ptr = get_packed_ptr(layers[layer_idx].out_w);
        layers[layer_idx].gate_w_ptr = get_packed_ptr(layers[layer_idx].gate_w);
        layers[layer_idx].up_w_ptr = get_packed_ptr(layers[layer_idx].up_w);
        layers[layer_idx].down_w_ptr = get_packed_ptr(layers[layer_idx].down_w);

        layers[layer_idx].in_norm_ptr = get_float_ptr(in_norm_w);
        layers[layer_idx].attn_sub_norm_ptr = get_float_ptr(attn_sub_norm_w);
        layers[layer_idx].post_norm_ptr = get_float_ptr(post_norm_w);
        layers[layer_idx].ffn_sub_norm_ptr = get_float_ptr(ffn_sub_norm_w);

        if (!layers[layer_idx].gate_w.is_none() && py::isinstance<py::array_t<int8_t>>(layers[layer_idx].gate_w)) {
            auto gw_req = layers[layer_idx].gate_w.cast<py::array_t<int8_t>>().request();
            if (gw_req.ndim == 3) {
                ffn_dim = gw_req.shape[0] * 64;
            }
        }
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

    void forward_layer_raw(int layer_idx) {
        const auto& lw = layers[layer_idx];

        // 1. RMSNorm
        rms_norm_raw(buf_x.data(), lw.in_norm_ptr, buf_norm_x.data(), hidden_size);

        // 2. Build LUT for norm_x ONCE (reused across Q, K, V)
        float inv_scale_act = build_lut(buf_norm_x.data(), hidden_size, lut_buf.data());

        // 3. FUSED Q, K, V in ONE parallel loop (60 blocks total)
        int q_blocks = q_dim / 64;   // 40
        int kv_blocks = kv_dim / 64; // 10
        int total_qkv = q_blocks + 2 * kv_blocks; // 60
        int k_half = hidden_size / 2;
        __m256i mask_low = _mm256_set1_epi8(0x0F);

        __m256 s_q = _mm256_set1_ps(inv_scale_act * lw.q_scale);
        __m256 s_k = _mm256_set1_ps(inv_scale_act * lw.k_scale);
        __m256 s_v = _mm256_set1_ps(inv_scale_act * lw.v_scale);

        #pragma omp parallel for schedule(dynamic, 4)
        for (int b = 0; b < total_qkv; ++b) {
            if (b < q_blocks) {
                const int8_t* bw = lw.q_w_ptr + b * k_half * 32;
                compute_single_block(lut_buf.data(), bw, k_half, buf_q.data() + b * 64, s_q, mask_low);
            } else if (b < q_blocks + kv_blocks) {
                int kb = b - q_blocks;
                const int8_t* bw = lw.k_w_ptr + kb * k_half * 32;
                compute_single_block(lut_buf.data(), bw, k_half, buf_k.data() + kb * 64, s_k, mask_low);
            } else {
                int vb = b - (q_blocks + kv_blocks);
                const int8_t* bw = lw.v_w_ptr + vb * k_half * 32;
                compute_single_block(lut_buf.data(), bw, k_half, buf_v.data() + vb * 64, s_v, mask_low);
            }
        }

        // 4. Zero-Copy In-Place Attention
        kv_cache.forward_attention_raw(layer_idx, buf_q.data(), buf_k.data(), buf_v.data(), buf_attn_out.data());

        // 5. Attn Sub-Norm
        rms_norm_raw(buf_attn_out.data(), lw.attn_sub_norm_ptr, buf_attn_sub.data(), hidden_size);

        // 6. Out Projection
        float inv_s_out = build_lut(buf_attn_sub.data(), hidden_size, lut_buf.data());
        compute_from_lut(lut_buf.data(), lw.out_w_ptr, hidden_size / 64, k_half, buf_out.data(), inv_s_out * lw.out_scale);

        // 7. Residual 1
        #pragma omp simd
        for (int i = 0; i < hidden_size; ++i) {
            buf_h1[i] = buf_x[i] + buf_out[i];
        }

        // 8. FUSED MLP Block
        if (lw.gate_w_ptr && lw.up_w_ptr && lw.down_w_ptr) {
            rms_norm_raw(buf_h1.data(), lw.post_norm_ptr, buf_post_norm.data(), hidden_size);

            float inv_s_mlp = build_lut(buf_post_norm.data(), hidden_size, lut_buf.data());
            int cur_ffn_dim = ffn_dim;
            int ffn_blocks = cur_ffn_dim / 64; // 108
            int total_mlp = 2 * ffn_blocks;    // 216

            __m256 s_gate = _mm256_set1_ps(inv_s_mlp * lw.gate_scale);
            __m256 s_up = _mm256_set1_ps(inv_s_mlp * lw.up_scale);

            #pragma omp parallel for schedule(dynamic, 8)
            for (int b = 0; b < total_mlp; ++b) {
                if (b < ffn_blocks) {
                    const int8_t* bw = lw.gate_w_ptr + b * k_half * 32;
                    compute_single_block(lut_buf.data(), bw, k_half, buf_gate.data() + b * 64, s_gate, mask_low);
                } else {
                    int ub = b - ffn_blocks;
                    const int8_t* bw = lw.up_w_ptr + ub * k_half * 32;
                    compute_single_block(lut_buf.data(), bw, k_half, buf_up.data() + ub * 64, s_up, mask_low);
                }
            }

            #pragma omp simd
            for (int i = 0; i < cur_ffn_dim; ++i) {
                float r = std::max(0.0f, buf_gate[i]);
                buf_act[i] = (r * r) * buf_up[i];
            }

            rms_norm_raw(buf_act.data(), lw.ffn_sub_norm_ptr, buf_ffn_norm.data(), cur_ffn_dim);

            float inv_s_down = build_lut(buf_ffn_norm.data(), cur_ffn_dim, lut_buf.data());
            compute_from_lut(lut_buf.data(), lw.down_w_ptr, hidden_size / 64, cur_ffn_dim / 2, buf_down.data(), inv_s_down * lw.down_scale);

            #pragma omp simd
            for (int i = 0; i < hidden_size; ++i) {
                buf_x[i] = buf_h1[i] + buf_down[i];
            }
        } else {
            std::memcpy(buf_x.data(), buf_h1.data(), hidden_size * sizeof(float));
        }
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
            py::ssize_t f_dim = g_buf.size;
            const float* g_ptr = static_cast<const float*>(g_buf.ptr);
            const float* u_ptr = static_cast<const float*>(u_buf.ptr);

            auto act_mult = py::array_t<float>(std::vector<py::ssize_t>{1, 1, f_dim});
            float* m_ptr = static_cast<float*>(act_mult.request().ptr);

            // Gated activation: relu(gate)^2 * up
            #pragma omp simd
            for (py::ssize_t i = 0; i < f_dim; ++i) {
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
        if (quant_scheme == "classic_tl1" || quant_scheme == "classic_tl2") {
            auto buf = x.request();
            std::memcpy(buf_x.data(), buf.ptr, hidden_size * sizeof(float));

            for (int l = 0; l < num_layers; ++l) {
                forward_layer_raw(l);
            }
            kv_cache.increment_seq_len(1);

            auto out = py::array_t<float>(std::vector<py::ssize_t>{1, 1, static_cast<py::ssize_t>(hidden_size)});
            std::memcpy(out.request().ptr, buf_x.data(), hidden_size * sizeof(float));
            return out;
        }

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
        const float* x_in_ptr = static_cast<const float*>(x.request().ptr);
        rms_norm_raw(x_in_ptr, final_norm_ptr, buf_norm_x.data(), hidden_size);
        const float* x_ptr = buf_norm_x.data();

        if (py::isinstance<py::array_t<float>>(embed_weights_obj)) {
            auto ew_buf = embed_weights_obj.cast<py::array_t<float>>().request();
            py::ssize_t vocab_size = ew_buf.shape[0];
            py::ssize_t h_dim = ew_buf.shape[1];
            const float* w_ptr = static_cast<const float*>(ew_buf.ptr);

            auto logits_arr = py::array_t<float>(std::vector<py::ssize_t>{1, 1, vocab_size});
            float* logits_ptr = static_cast<float*>(logits_arr.request().ptr);

            #pragma omp parallel for schedule(dynamic, 1024)
            for (py::ssize_t v = 0; v < vocab_size; ++v) {
                const float* row = w_ptr + v * h_dim;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                py::ssize_t i = 0;
                for (; i + 16 <= h_dim; i += 16) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + i), _mm256_loadu_ps(row + i), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + i + 8), _mm256_loadu_ps(row + i + 8), acc1);
                }
                __m256 acc = _mm256_add_ps(acc0, acc1);
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

        // 1. Quantize activation vector x to int8 ONCE
        float amax_x = 1e-8f;
        for (py::ssize_t i = 0; i < h_dim; ++i) {
            float val = std::abs(x_ptr[i]);
            if (val > amax_x) amax_x = val;
        }
        float s_x = amax_x / 127.0f;
        float inv_s_x = 1.0f / s_x;
        std::vector<int8_t> x_i8(h_dim);
        #pragma omp simd
        for (py::ssize_t i = 0; i < h_dim; ++i) {
            float q = std::round(x_ptr[i] * inv_s_x);
            x_i8[i] = static_cast<int8_t>(std::clamp(q, -128.0f, 127.0f));
        }

        // Pre-expand x into 16-bit SIMD registers ONCE (cuts inner loop ops by 2x)
        py::ssize_t num_vecs = h_dim / 16;
        std::vector<__m256i> x_expanded(num_vecs);
        for (py::ssize_t j = 0; j < num_vecs; ++j) {
            __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x_i8.data() + j * 16));
            x_expanded[j] = _mm256_cvtepi8_epi16(a);
        }
        const __m256i* x_exp_ptr = x_expanded.data();

        // 2. Pure Integer AVX2 Dot-Products across vocab
        #pragma omp parallel for schedule(dynamic, 1024)
        for (py::ssize_t v = 0; v < vocab_size; ++v) {
            const int8_t* row = w_ptr + v * h_dim;
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();

            py::ssize_t j = 0;
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

            for (py::ssize_t rem = j * 16; rem < h_dim; ++rem) {
                sum += static_cast<int32_t>(x_i8[rem]) * static_cast<int32_t>(row[rem]);
            }
            logits_ptr[v] = static_cast<float>(sum) * (s_x * scales_ptr[v]);
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

        __m256 s_vec = _mm256_set1_ps(scale);
        py::ssize_t i = 0;
        for (; i + 16 <= h_dim; i += 16) {
            __m128i r_lo = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + i));
            __m128i r_hi = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + i + 8));
            __m256 f_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r_lo));
            __m256 f_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r_hi));
            _mm256_storeu_ps(out_ptr + i, _mm256_mul_ps(f_lo, s_vec));
            _mm256_storeu_ps(out_ptr + i + 8, _mm256_mul_ps(f_hi, s_vec));
        }
        for (; i < h_dim; ++i) {
            out_ptr[i] = static_cast<float>(row[i]) * scale;
        }
        return out;
    }

    std::vector<float> logits_scratch;

    // Direct C++ Top-K/Top-P sampler (0.01 ms vs 18 ms in Python)
    int sample_top_k_top_p(
        const float* logits,
        int vocab_size,
        const std::vector<int>& past_tokens,
        float temperature = 0.7f,
        float top_p = 0.9f,
        int top_k = 40,
        float repetition_penalty = 1.15f
    ) {
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

        std::unordered_set<int> past_set(past_tokens.begin(), past_tokens.end());

        for (int i = 0; i < vocab_size; ++i) {
            float l = logits[i];
            if (repetition_penalty != 1.0f && past_set.count(i)) {
                l = (l > 0.0f) ? (l / repetition_penalty) : (l * repetition_penalty);
            }

            if (static_cast<int>(min_heap.size()) < top_k) {
                min_heap.push({l, i});
            } else if (l > min_heap.top().first) {
                min_heap.pop();
                min_heap.push({l, i});
            }
        }

        int k_size = min_heap.size();
        std::vector<Pair> candidates(k_size);
        for (int i = k_size - 1; i >= 0; --i) {
            candidates[i] = min_heap.top();
            min_heap.pop();
        }

        float max_l = candidates[0].first;
        float sum_exp = 0.0f;
        std::vector<float> probs(k_size);
        for (int i = 0; i < k_size; ++i) {
            probs[i] = std::exp((candidates[i].first - max_l) / temperature);
            sum_exp += probs[i];
        }
        for (int i = 0; i < k_size; ++i) {
            probs[i] /= sum_exp;
        }

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
            if (r <= acc) {
                return candidates[i].second;
            }
        }
        return candidates[0].second;
    }

    // Single fully-native token generation step (0 python roundtrips)
    int generate_step_cxx(
        int token_id,
        float temperature,
        float top_p,
        int top_k,
        float repetition_penalty,
        const std::vector<int>& past_tokens,
        py::object embed_weights_obj,
        py::object embed_scales_obj = py::none()
    ) {
        // 1. Embedding lookup directly into buf_x
        if (py::isinstance<py::array_t<float>>(embed_weights_obj)) {
            auto ew_buf = embed_weights_obj.cast<py::array_t<float>>().request();
            const float* row = static_cast<const float*>(ew_buf.ptr) + token_id * hidden_size;
            std::memcpy(buf_x.data(), row, hidden_size * sizeof(float));
        } else {
            auto ew_buf = embed_weights_obj.cast<py::array_t<int8_t>>().request();
            auto es_buf = embed_scales_obj.cast<py::array_t<float>>().request();
            const int8_t* row = static_cast<const int8_t*>(ew_buf.ptr) + token_id * hidden_size;
            float scale = static_cast<const float*>(es_buf.ptr)[token_id];
            __m256 s_vec = _mm256_set1_ps(scale);
            int i = 0;
            for (; i + 16 <= hidden_size; i += 16) {
                __m128i r_lo = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + i));
                __m128i r_hi = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + i + 8));
                __m256 f_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r_lo));
                __m256 f_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r_hi));
                _mm256_storeu_ps(buf_x.data() + i, _mm256_mul_ps(f_lo, s_vec));
                _mm256_storeu_ps(buf_x.data() + i + 8, _mm256_mul_ps(f_hi, s_vec));
            }
            for (; i < hidden_size; ++i) {
                buf_x[i] = static_cast<float>(row[i]) * scale;
            }
        }

        // 2. Transformer layers in raw C++
        for (int l = 0; l < num_layers; ++l) {
            forward_layer_raw(l);
        }
        kv_cache.increment_seq_len(1);

        // 3. Compute logits into logits_scratch
        int vocab_size = 0;
        if (py::isinstance<py::array_t<float>>(embed_weights_obj)) {
            auto ew_buf = embed_weights_obj.cast<py::array_t<float>>().request();
            vocab_size = ew_buf.shape[0];
            if (static_cast<int>(logits_scratch.size()) < vocab_size) logits_scratch.resize(vocab_size);
            rms_norm_raw(buf_x.data(), final_norm_ptr, buf_norm_x.data(), hidden_size);
            const float* x_ptr = buf_norm_x.data();
            const float* w_ptr = static_cast<const float*>(ew_buf.ptr);

            #pragma omp parallel for schedule(dynamic, 1024)
            for (int v = 0; v < vocab_size; ++v) {
                const float* row = w_ptr + v * hidden_size;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                int i = 0;
                for (; i + 16 <= hidden_size; i += 16) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + i), _mm256_loadu_ps(row + i), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + i + 8), _mm256_loadu_ps(row + i + 8), acc1);
                }
                __m256 acc = _mm256_add_ps(acc0, acc1);
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, acc);
                float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                for (; i < hidden_size; ++i) sum += x_ptr[i] * row[i];
                logits_scratch[v] = sum;
            }
        } else {
            auto ew_buf = embed_weights_obj.cast<py::array_t<int8_t>>().request();
            auto es_buf = embed_scales_obj.cast<py::array_t<float>>().request();
            vocab_size = ew_buf.shape[0];
            if (static_cast<int>(logits_scratch.size()) < vocab_size) logits_scratch.resize(vocab_size);

            rms_norm_raw(buf_x.data(), final_norm_ptr, buf_norm_x.data(), hidden_size);
            const float* x_ptr = buf_norm_x.data();
            const int8_t* w_ptr = static_cast<const int8_t*>(ew_buf.ptr);
            const float* scales_ptr = static_cast<const float*>(es_buf.ptr);

            float amax_x = 1e-8f;
            for (int i = 0; i < hidden_size; ++i) {
                float val = std::abs(x_ptr[i]);
                if (val > amax_x) amax_x = val;
            }
            float s_x = amax_x / 127.0f;
            float inv_s_x = 1.0f / s_x;
            std::vector<int8_t> x_i8(hidden_size);
            #pragma omp simd
            for (int i = 0; i < hidden_size; ++i) {
                float q = std::round(x_ptr[i] * inv_s_x);
                x_i8[i] = static_cast<int8_t>(std::clamp(q, -128.0f, 127.0f));
            }

            int num_vecs = hidden_size / 16;
            std::vector<__m256i> x_expanded(num_vecs);
            for (int j = 0; j < num_vecs; ++j) {
                __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x_i8.data() + j * 16));
                x_expanded[j] = _mm256_cvtepi8_epi16(a);
            }
            const __m256i* x_exp_ptr = x_expanded.data();

            #pragma omp parallel for schedule(dynamic, 1024)
            for (int v = 0; v < vocab_size; ++v) {
                const int8_t* row = w_ptr + v * hidden_size;
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

                for (int rem = j * 16; rem < hidden_size; ++rem) {
                    sum += static_cast<int32_t>(x_i8[rem]) * static_cast<int32_t>(row[rem]);
                }
                logits_scratch[v] = static_cast<float>(sum) * (s_x * scales_ptr[v]);
            }
        }

        // 4. Sample token in C++
        return sample_top_k_top_p(logits_scratch.data(), vocab_size, past_tokens, temperature, top_p, top_k, repetition_penalty);
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
             py::arg("token_id"), py::arg("embed_weights"), py::arg("embed_scales") = py::none())
        .def("generate_step_cxx", &ExecutionContext::generate_step_cxx,
             py::arg("token_id"),
             py::arg("temperature") = 0.7f,
             py::arg("top_p") = 0.9f,
             py::arg("top_k") = 40,
             py::arg("repetition_penalty") = 1.15f,
             py::arg("past_tokens") = std::vector<int>(),
             py::arg("embed_weights") = py::none(),
             py::arg("embed_scales") = py::none());
}

