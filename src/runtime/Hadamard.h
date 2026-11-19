#ifndef TENZO_HADAMARD_H
#define TENZO_HADAMARD_H

// Fast Walsh-Hadamard transform for the D = 128 KV rotation of the ternary_rot KV mode.
//   rotate(x)   = signs .* FWHT(x) / sqrt(128)      == x @ R            (R = H diag(signs) / sqrt(128), see kv_ptq.py)
//   unrotate(y) = FWHT(signs .* y) / sqrt(128)      == y @ R^T          (R orthogonal, so this inverts rotate)
// AVX2 (and FMA-free): strides 64..8 are butterflies across registers, strides 4, 2, 1 use in-register shuffles.

#include "HadamardSigns.h"
#include <immintrin.h>
#include <cmath>

namespace tenzo {
namespace hadamard {

constexpr int D = 128;

// In-place unnormalized FWHT of 128 floats (any order of stages gives the same transform).
inline void fwht128(float* x) {
    __m256 v[16];
    for (int i = 0; i < 16; ++i) v[i] = _mm256_loadu_ps(x + 8 * i);
    for (int stride = 8; stride >= 1; stride >>= 1) {  // register strides 64, 32, 16, 8 (in units of 8 floats)
        for (int base = 0; base < 16; base += 2 * stride) {
            for (int j = base; j < base + stride; ++j) {
                __m256 a = v[j], b = v[j + stride];
                v[j] = _mm256_add_ps(a, b);
                v[j + stride] = _mm256_sub_ps(a, b);
            }
        }
    }
    for (int i = 0; i < 16; ++i) {
        __m256 t = v[i];
        // stride 4: (lo, hi) -> (lo + hi, lo - hi)
        __m256 lo = _mm256_permute2f128_ps(t, t, 0x00), hi = _mm256_permute2f128_ps(t, t, 0x11);
        t = _mm256_blend_ps(_mm256_add_ps(lo, hi), _mm256_sub_ps(lo, hi), 0xF0);
        // stride 2: pairs (0,2) (1,3) within each 128-bit lane
        __m256 a = _mm256_shuffle_ps(t, t, _MM_SHUFFLE(1, 0, 1, 0)), b = _mm256_shuffle_ps(t, t, _MM_SHUFFLE(3, 2, 3, 2));
        t = _mm256_blend_ps(_mm256_add_ps(a, b), _mm256_sub_ps(a, b), 0xCC);
        // stride 1: pairs (0,1) (2,3)
        a = _mm256_shuffle_ps(t, t, _MM_SHUFFLE(2, 2, 0, 0));
        b = _mm256_shuffle_ps(t, t, _MM_SHUFFLE(3, 3, 1, 1));
        t = _mm256_blend_ps(_mm256_add_ps(a, b), _mm256_sub_ps(a, b), 0xAA);
        _mm256_storeu_ps(x + 8 * i, t);
    }
}

// y = rotate(x); x and y may alias.
inline void rotate(const float* x, float* y) {
    alignas(32) float t[D];
    for (int i = 0; i < D; i += 8) _mm256_store_ps(t + i, _mm256_loadu_ps(x + i));
    fwht128(t);
    const __m256 inv = _mm256_set1_ps(1.0f / std::sqrt(128.0f));
    for (int i = 0; i < D; i += 8)
        _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_load_ps(t + i), _mm256_load_ps(kHadamardSigns128 + i)), inv));
}

// x = unrotate(y); x and y may alias.
inline void unrotate(const float* y, float* x) {
    alignas(32) float t[D];
    for (int i = 0; i < D; i += 8) _mm256_store_ps(t + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), _mm256_load_ps(kHadamardSigns128 + i)));
    fwht128(t);
    const __m256 inv = _mm256_set1_ps(1.0f / std::sqrt(128.0f));
    for (int i = 0; i < D; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_load_ps(t + i), inv));
}

} // namespace hadamard
} // namespace tenzo

#endif // TENZO_HADAMARD_H
