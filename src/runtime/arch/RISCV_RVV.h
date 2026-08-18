#ifndef TENZO_ARCH_RISCV_RVV_H
#define TENZO_ARCH_RISCV_RVV_H

#include <cstdint>
#include <cstddef>

#if defined(__riscv) && defined(__riscv_vector)
#include <riscv_vector.h>

namespace tenzo {
namespace rvv {

// RISC-V RVV 1.0 Vectorized BitLinear TL1 GEMV Kernel
inline void gemv_bitlinear_tl1_rvv(
    const float* __restrict__ act_f32,
    const uint8_t* __restrict__ w_tl1,
    float* __restrict__ out_f32,
    size_t M,
    size_t K,
    size_t N,
    float scale
) {
    // 2-bit ternary weights unpacking lookup table:
    // 00 -> -1, 01 -> 0, 10 -> +1, 11 -> 0
    const float lut[4] = {-1.0f, 0.0f, 1.0f, 0.0f};

    for (size_t n = 0; n < N; ++n) {
        const uint8_t* w_row = w_tl1 + n * (K / 4);
        float acc = 0.0f;
        
        size_t k = 0;
        // Vector processing with variable vector length (vl)
        while (k < K) {
            size_t vl = __riscv_vsetvl_e32m1(K - k);
            vfloat32m1_t v_acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            
            for (size_t i = 0; i < vl; ++i) {
                size_t global_k = k + i;
                uint8_t byte = w_row[global_k / 4];
                uint8_t code = (byte >> ((global_k % 4) * 2)) & 0x03;
                float w_val = lut[code];
                acc += act_f32[global_k] * w_val;
            }
            k += vl;
        }
        out_f32[n] = acc * scale;
    }
}

// Vectorized RMSNorm on RISC-V RVV 1.0
inline void rms_norm_rvv(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    size_t dim,
    float eps = 1e-5f
) {
    float sum_sq = 0.0f;
    size_t i = 0;
    while (i < dim) {
        size_t vl = __riscv_vsetvl_e32m1(dim - i);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + i, vl);
        vfloat32m1_t vx_sq = __riscv_vfmul_vv_f32m1(vx, vx, vl);
        // Reduce sum
        vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t v_red = __riscv_vfredusum_vs_f32m1_f32m1(vx_sq, v_zero, vl);
        sum_sq += __riscv_vfmv_f_s_f32m1_f32(v_red);
        i += vl;
    }
    
    float inv_rms = 1.0f / sqrtf((sum_sq / static_cast<float>(dim)) + eps);
    
    i = 0;
    while (i < dim) {
        size_t vl = __riscv_vsetvl_e32m1(dim - i);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + i, vl);
        vfloat32m1_t vw = __riscv_vle32_v_f32m1(w + i, vl);
        vfloat32m1_t v_norm = __riscv_vfmul_vf_f32m1(vx, inv_rms, vl);
        vfloat32m1_t v_out = __riscv_vfmul_vv_f32m1(v_norm, vw, vl);
        __riscv_vse32_v_f32m1(out + i, v_out, vl);
        i += vl;
    }
}

} // namespace rvv
} // namespace tenzo

#else

// Fallback scalar / portable emulation for non-RVV platforms
namespace tenzo {
namespace rvv {

inline void gemv_bitlinear_tl1_rvv(
    const float* __restrict__ act_f32,
    const uint8_t* __restrict__ w_tl1,
    float* __restrict__ out_f32,
    size_t M,
    size_t K,
    size_t N,
    float scale
) {
    const float lut[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
    for (size_t n = 0; n < N; ++n) {
        const uint8_t* w_row = w_tl1 + n * (K / 4);
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            uint8_t byte = w_row[k / 4];
            uint8_t code = (byte >> ((k % 4) * 2)) & 0x03;
            acc += act_f32[k] * lut[code];
        }
        out_f32[n] = acc * scale;
    }
}

inline void rms_norm_rvv(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    size_t dim,
    float eps = 1e-5f
) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum_sq += x[i] * x[i];
    }
    float inv_rms = 1.0f / sqrtf((sum_sq / static_cast<float>(dim)) + eps);
    for (size_t i = 0; i < dim; ++i) {
        out[i] = x[i] * inv_rms * w[i];
    }
}

} // namespace rvv
} // namespace tenzo

#endif // defined(__riscv) && defined(__riscv_vector)

#endif // TENZO_ARCH_RISCV_RVV_H
