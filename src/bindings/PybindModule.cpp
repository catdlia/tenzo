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

PYBIND11_MODULE(tenzo_runtime, m) {
    m.doc() = "Tenzo C++ Runtime Native AVX2 Module";

    m.def("pack_ternary_weights", &pack_ternary_weights, "Pack [N, K] ternary weights into [N/64, K/2, 32] TL1 format",
          py::arg("tern_weights"));

    m.def("dispatch_linear_fp32", &dispatch_linear_fp32, "Dispatch high-performance AVX2 FP32 GEMM",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("relu") = false);

    m.def("dispatch_bitlinear_tl1", &dispatch_bitlinear_tl1, "Dispatch AVX2 TL1 256-bit BitLinear SIMD micro-kernel",
          py::arg("input"), py::arg("packed_weight"), py::arg("scale") = 1.0f, py::arg("bias") = py::none());

    m.def("dispatch_linear_cxx", &dispatch_linear_cxx, "Universal Linear Dispatcher (FP32, BitLinear TL1)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("quant_scheme") = "fp32", py::arg("scale") = 1.0f);
}
