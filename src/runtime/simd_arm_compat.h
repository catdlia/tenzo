/**
 * @file simd_arm_compat.h
 * @brief High-Performance ARM NEON / AVX2 Compatibility Layer for Tenzo Engine
 * Maps x86_64 AVX2 / SSE intrinsics directly to ARM NEON SIMD on AArch64.
 */

#ifndef TENZO_SIMD_ARM_COMPAT_H
#define TENZO_SIMD_ARM_COMPAT_H

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <immintrin.h>

#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

// --- Types ---
typedef float32x4_t __m128;
typedef int8x16_t   __m128i;

struct __m256 {
    float32x4_t lo;
    float32x4_t hi;
};

struct __m256i {
    int8x16_t lo;
    int8x16_t hi;
};

// --- Popcount ---
#define _mm_popcnt_u64(x) (static_cast<int64_t>(__builtin_popcountll(static_cast<uint64_t>(x))))

// --- 128-bit Float Intrinsics ---
inline __m128 _mm_setzero_ps() {
    return vdupq_n_f32(0.0f);
}

inline __m128 _mm_set1_ps(float f) {
    return vdupq_n_f32(f);
}

inline __m128 _mm_loadu_ps(const float* p) {
    return vld1q_f32(p);
}

inline __m128 _mm_load_ps(const float* p) {
    return vld1q_f32(p);
}

inline void _mm_storeu_ps(float* p, __m128 a) {
    vst1q_f32(p, a);
}

inline void _mm_store_ps(float* p, __m128 a) {
    vst1q_f32(p, a);
}

inline float _mm_cvtss_f32(__m128 a) {
    return vgetq_lane_f32(a, 0);
}

inline __m128 _mm_add_ps(__m128 a, __m128 b) {
    return vaddq_f32(a, b);
}

inline __m128 _mm_sub_ps(__m128 a, __m128 b) {
    return vsubq_f32(a, b);
}

inline __m128 _mm_mul_ps(__m128 a, __m128 b) {
    return vmulq_f32(a, b);
}

inline __m128 _mm_fmadd_ps(__m128 a, __m128 b, __m128 c) {
    return vfmaq_f32(c, a, b);
}

// --- 128-bit Integer Intrinsics ---
inline __m128i _mm_setzero_si128() {
    return vdupq_n_s8(0);
}

inline __m128i _mm_set1_epi8(int8_t c) {
    return vdupq_n_s8(c);
}

inline __m128i _mm_set1_epi16(int16_t w) {
    return vreinterpretq_s8_s16(vdupq_n_s16(w));
}

inline __m128i _mm_set1_epi32(int32_t i) {
    return vreinterpretq_s8_s32(vdupq_n_s32(i));
}

inline __m128i _mm_loadu_si128(const __m128i* p) {
    return vld1q_s8(reinterpret_cast<const int8_t*>(p));
}

inline __m128i _mm_load_si128(const __m128i* p) {
    return vld1q_s8(reinterpret_cast<const int8_t*>(p));
}

inline void _mm_storeu_si128(__m128i* p, __m128i a) {
    vst1q_s8(reinterpret_cast<int8_t*>(p), a);
}

inline void _mm_store_si128(__m128i* p, __m128i a) {
    vst1q_s8(reinterpret_cast<int8_t*>(p), a);
}

inline __m128i _mm_loadl_epi64(const __m128i* p) {
    int64x1_t val = vld1_s64(reinterpret_cast<const int64_t*>(p));
    return vcombine_s8(vreinterpret_s8_s64(val), vdup_n_s8(0));
}

inline __m128i _mm_cvtsi32_si128(int a) {
    int32x4_t v = vdupq_n_s32(0);
    v = vsetq_lane_s32(a, v, 0);
    return vreinterpretq_s8_s32(v);
}

inline int _mm_cvtsi128_si32(__m128i a) {
    return vgetq_lane_s32(vreinterpretq_s32_s8(a), 0);
}

inline __m128i _mm_and_si128(__m128i a, __m128i b) {
    return vandq_s8(a, b);
}

inline __m128i _mm_or_si128(__m128i a, __m128i b) {
    return vorrq_s8(a, b);
}

inline __m128i _mm_xor_si128(__m128i a, __m128i b) {
    return veorq_s8(a, b);
}

inline __m128i _mm_add_epi8(__m128i a, __m128i b) {
    return vaddq_s8(a, b);
}

inline __m128i _mm_sub_epi8(__m128i a, __m128i b) {
    return vsubq_s8(a, b);
}

inline __m128i _mm_add_epi16(__m128i a, __m128i b) {
    return vreinterpretq_s8_s16(vaddq_s16(vreinterpretq_s16_s8(a), vreinterpretq_s16_s8(b)));
}

inline __m128i _mm_add_epi32(__m128i a, __m128i b) {
    return vreinterpretq_s8_s32(vaddq_s32(vreinterpretq_s32_s8(a), vreinterpretq_s32_s8(b)));
}

#define _mm_srli_epi16(a, imm) \
    (vreinterpretq_s8_u16(vshrq_n_u16(vreinterpretq_u16_s8(a), (imm))))

#define _mm_srli_si128(a, imm) \
    (vextq_s8((a), vdupq_n_s8(0), (imm)))

inline __m128i _mm_shuffle_epi8(__m128i a, __m128i b) {
    uint8x16_t mask = vcgtq_s8(b, vdupq_n_s8(-1));
    uint8x16_t idx = vreinterpretq_u8_s8(b);
    uint8x16_t lut = vreinterpretq_u8_s8(a);
    return vreinterpretq_s8_u8(vandq_u8(vqtbl1q_u8(lut, idx), mask));
}

inline __m128i _mm_unpacklo_epi8(__m128i a, __m128i b) {
    int8x8x2_t res = vzip_s8(vget_low_s8(a), vget_low_s8(b));
    return vcombine_s8(res.val[0], res.val[1]);
}

inline __m128i _mm_unpackhi_epi8(__m128i a, __m128i b) {
    int8x8x2_t res = vzip_s8(vget_high_s8(a), vget_high_s8(b));
    return vcombine_s8(res.val[0], res.val[1]);
}

inline __m128i _mm_unpacklo_epi16(__m128i a, __m128i b) {
    int16x4x2_t res = vzip_s16(vget_low_s16(vreinterpretq_s16_s8(a)), vget_low_s16(vreinterpretq_s16_s8(b)));
    return vreinterpretq_s8_s16(vcombine_s16(res.val[0], res.val[1]));
}

inline __m128i _mm_unpackhi_epi16(__m128i a, __m128i b) {
    int16x4x2_t res = vzip_s16(vget_high_s16(vreinterpretq_s16_s8(a)), vget_high_s16(vreinterpretq_s16_s8(b)));
    return vreinterpretq_s8_s16(vcombine_s16(res.val[0], res.val[1]));
}

// FP16 to FP32 conversion for GGUF Q4_0 scales
inline __m128 _mm_cvtph_ps(__m128i a) {
    uint32_t val = static_cast<uint32_t>(vgetq_lane_u32(vreinterpretq_u32_s8(a), 0));
    uint16_t h0 = static_cast<uint16_t>(val & 0xFFFF);
    uint16_t h1 = static_cast<uint16_t>((val >> 16) & 0xFFFF);

    auto f16_to_f32 = [](uint16_t h) -> float {
        uint32_t w = static_cast<uint32_t>(h) << 16;
        uint32_t sign = w & 0x80000000;
        uint32_t two_w = w + w;
        uint32_t exp = (two_w >> 24) + ((127 - 15) << 7);
        if ((two_w >> 24) == 0) return 0.0f;
        uint32_t res = sign | (exp << 23) | ((two_w >> 1) & 0x007FFFFF);
        float f;
        std::memcpy(&f, &res, sizeof(float));
        return f;
    };

    float arr[4] = { f16_to_f32(h0), f16_to_f32(h1), 0.0f, 0.0f };
    return vld1q_f32(arr);
}

// --- 256-bit Float Intrinsics ---
inline __m256 _mm256_setzero_ps() {
    __m256 r;
    r.lo = vdupq_n_f32(0.0f);
    r.hi = vdupq_n_f32(0.0f);
    return r;
}

inline __m256 _mm256_set1_ps(float f) {
    __m256 r;
    r.lo = vdupq_n_f32(f);
    r.hi = vdupq_n_f32(f);
    return r;
}

inline __m256 _mm256_loadu_ps(const float* p) {
    __m256 r;
    r.lo = vld1q_f32(p);
    r.hi = vld1q_f32(p + 4);
    return r;
}

inline __m256 _mm256_load_ps(const float* p) {
    return _mm256_loadu_ps(p);
}

inline void _mm256_storeu_ps(float* p, __m256 a) {
    vst1q_f32(p, a.lo);
    vst1q_f32(p + 4, a.hi);
}

inline void _mm256_store_ps(float* p, __m256 a) {
    _mm256_storeu_ps(p, a);
}

inline __m256 _mm256_add_ps(__m256 a, __m256 b) {
    __m256 r;
    r.lo = vaddq_f32(a.lo, b.lo);
    r.hi = vaddq_f32(a.hi, b.hi);
    return r;
}

inline __m256 _mm256_sub_ps(__m256 a, __m256 b) {
    __m256 r;
    r.lo = vsubq_f32(a.lo, b.lo);
    r.hi = vsubq_f32(a.hi, b.hi);
    return r;
}

inline __m256 _mm256_mul_ps(__m256 a, __m256 b) {
    __m256 r;
    r.lo = vmulq_f32(a.lo, b.lo);
    r.hi = vmulq_f32(a.hi, b.hi);
    return r;
}

inline __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) {
    __m256 r;
    r.lo = vfmaq_f32(c.lo, a.lo, b.lo);
    r.hi = vfmaq_f32(c.hi, a.hi, b.hi);
    return r;
}

inline __m256 _mm256_fmsub_ps(__m256 a, __m256 b, __m256 c) {
    __m256 r;
    r.lo = vsubq_f32(vmulq_f32(a.lo, b.lo), c.lo);
    r.hi = vsubq_f32(vmulq_f32(a.hi, b.hi), c.hi);
    return r;
}

inline __m256 _mm256_max_ps(__m256 a, __m256 b) {
    __m256 r;
    r.lo = vmaxq_f32(a.lo, b.lo);
    r.hi = vmaxq_f32(a.hi, b.hi);
    return r;
}

// --- 256-bit Integer Intrinsics ---
inline __m256i _mm256_setzero_si256() {
    __m256i r;
    r.lo = vdupq_n_s8(0);
    r.hi = vdupq_n_s8(0);
    return r;
}

inline __m256i _mm256_set1_epi8(int8_t c) {
    __m256i r;
    r.lo = vdupq_n_s8(c);
    r.hi = vdupq_n_s8(c);
    return r;
}

inline __m256i _mm256_set1_epi16(int16_t w) {
    __m256i r;
    r.lo = vreinterpretq_s8_s16(vdupq_n_s16(w));
    r.hi = vreinterpretq_s8_s16(vdupq_n_s16(w));
    return r;
}

inline __m256i _mm256_set1_epi32(int32_t i) {
    __m256i r;
    r.lo = vreinterpretq_s8_s32(vdupq_n_s32(i));
    r.hi = vreinterpretq_s8_s32(vdupq_n_s32(i));
    return r;
}

inline __m256i _mm256_loadu_si256(const __m256i* p) {
    const int8_t* ptr = reinterpret_cast<const int8_t*>(p);
    __m256i r;
    r.lo = vld1q_s8(ptr);
    r.hi = vld1q_s8(ptr + 16);
    return r;
}

inline __m256i _mm256_load_si256(const __m256i* p) {
    return _mm256_loadu_si256(p);
}

inline void _mm256_storeu_si256(__m256i* p, __m256i a) {
    int8_t* ptr = reinterpret_cast<int8_t*>(p);
    vst1q_s8(ptr, a.lo);
    vst1q_s8(ptr + 16, a.hi);
}

inline void _mm256_store_si256(__m256i* p, __m256i a) {
    _mm256_storeu_si256(p, a);
}

inline __m256i _mm256_and_si256(__m256i a, __m256i b) {
    __m256i r;
    r.lo = vandq_s8(a.lo, b.lo);
    r.hi = vandq_s8(a.hi, b.hi);
    return r;
}

inline __m256i _mm256_or_si256(__m256i a, __m256i b) {
    __m256i r;
    r.lo = vorrq_s8(a.lo, b.lo);
    r.hi = vorrq_s8(a.hi, b.hi);
    return r;
}

inline __m256i _mm256_xor_si256(__m256i a, __m256i b) {
    __m256i r;
    r.lo = veorq_s8(a.lo, b.lo);
    r.hi = veorq_s8(a.hi, b.hi);
    return r;
}

inline __m256i _mm256_add_epi8(__m256i a, __m256i b) {
    __m256i r;
    r.lo = vaddq_s8(a.lo, b.lo);
    r.hi = vaddq_s8(a.hi, b.hi);
    return r;
}

inline __m256i _mm256_add_epi16(__m256i a, __m256i b) {
    __m256i r;
    r.lo = vreinterpretq_s8_s16(vaddq_s16(vreinterpretq_s16_s8(a.lo), vreinterpretq_s16_s8(b.lo)));
    r.hi = vreinterpretq_s8_s16(vaddq_s16(vreinterpretq_s16_s8(a.hi), vreinterpretq_s16_s8(b.hi)));
    return r;
}

inline __m256i _mm256_add_epi32(__m256i a, __m256i b) {
    __m256i r;
    r.lo = vreinterpretq_s8_s32(vaddq_s32(vreinterpretq_s32_s8(a.lo), vreinterpretq_s32_s8(b.lo)));
    r.hi = vreinterpretq_s8_s32(vaddq_s32(vreinterpretq_s32_s8(a.hi), vreinterpretq_s32_s8(b.hi)));
    return r;
}

#define _mm256_srli_epi16(a, imm) \
    ([](__m256i v) -> __m256i { \
        __m256i res; \
        res.lo = vreinterpretq_s8_u16(vshrq_n_u16(vreinterpretq_u16_s8(v.lo), (imm))); \
        res.hi = vreinterpretq_s8_u16(vshrq_n_u16(vreinterpretq_u16_s8(v.hi), (imm))); \
        return res; \
    }(a))

inline __m256i _mm256_shuffle_epi8(__m256i a, __m256i b) {
    __m256i r;
    uint8x16_t mask_lo = vcgtq_s8(b.lo, vdupq_n_s8(-1));
    uint8x16_t idx_lo = vreinterpretq_u8_s8(b.lo);
    uint8x16_t lut_lo = vreinterpretq_u8_s8(a.lo);
    r.lo = vreinterpretq_s8_u8(vandq_u8(vqtbl1q_u8(lut_lo, idx_lo), mask_lo));

    uint8x16_t mask_hi = vcgtq_s8(b.hi, vdupq_n_s8(-1));
    uint8x16_t idx_hi = vreinterpretq_u8_s8(b.hi);
    uint8x16_t lut_hi = vreinterpretq_u8_s8(a.hi);
    r.hi = vreinterpretq_s8_u8(vandq_u8(vqtbl1q_u8(lut_hi, idx_hi), mask_hi));
    return r;
}

#define _mm256_extracti128_si256(a, imm) ((imm) == 0 ? (a).lo : (a).hi)
#define _mm256_castsi256_si128(a) ((a).lo)

inline __m256i _mm256_cvtepi8_epi16(__m128i a) {
    __m256i r;
    r.lo = vreinterpretq_s8_s16(vmovl_s8(vget_low_s8(a)));
    r.hi = vreinterpretq_s8_s16(vmovl_high_s8(a));
    return r;
}

inline __m256i _mm256_cvtepi16_epi32(__m128i a) {
    int16x8_t s16 = vreinterpretq_s16_s8(a);
    __m256i r;
    r.lo = vreinterpretq_s8_s32(vmovl_s16(vget_low_s16(s16)));
    r.hi = vreinterpretq_s8_s32(vmovl_high_s16(s16));
    return r;
}

inline __m256i _mm256_cvtepi8_epi32(__m128i a) {
    int8x8_t lo8 = vget_low_s8(a);
    int16x8_t s16 = vmovl_s8(lo8);
    __m256i r;
    r.lo = vreinterpretq_s8_s32(vmovl_s16(vget_low_s16(s16)));
    r.hi = vreinterpretq_s8_s32(vmovl_high_s16(s16));
    return r;
}

inline __m256 _mm256_cvtepi32_ps(__m256i a) {
    __m256 r;
    r.lo = vcvtq_f32_s32(vreinterpretq_s32_s8(a.lo));
    r.hi = vcvtq_f32_s32(vreinterpretq_s32_s8(a.hi));
    return r;
}

inline __m256i _mm256_madd_epi16(__m256i a, __m256i b) {
    int16x8_t a_lo = vreinterpretq_s16_s8(a.lo);
    int16x8_t b_lo = vreinterpretq_s16_s8(b.lo);
    int32x4_t lo_0 = vmull_s16(vget_low_s16(a_lo), vget_low_s16(b_lo));
    int32x4_t lo_1 = vmull_high_s16(a_lo, b_lo);
    int32x4_t res_lo = vpaddq_s32(lo_0, lo_1);

    int16x8_t a_hi = vreinterpretq_s16_s8(a.hi);
    int16x8_t b_hi = vreinterpretq_s16_s8(b.hi);
    int32x4_t hi_0 = vmull_s16(vget_low_s16(a_hi), vget_low_s16(b_hi));
    int32x4_t hi_1 = vmull_high_s16(a_hi, b_hi);
    int32x4_t res_hi = vpaddq_s32(hi_0, hi_1);

    __m256i r;
    r.lo = vreinterpretq_s8_s32(res_lo);
    r.hi = vreinterpretq_s8_s32(res_hi);
    return r;
}

#endif // __ARM_NEON

#endif // TENZO_SIMD_ARM_COMPAT_H
