// GEMMEndToEnd.cpp - Native C++ GEMM implementation with packing
// This is a STANDALONE benchmark that doesn't need MLIR JIT
// Used to validate the packing algorithm correctness and measure performance
//
// PADDING SUPPORT (Day 1 of Etap 3):
// - Handles ARBITRARY matrix sizes (not just multiples of MR=6, NR=16)
// - M not divisible by 6? → Last A panel is zero-padded
// - N not divisible by 16? → Last B panel is zero-padded
// - Edge micro-kernels use temp buffer to avoid writing outside bounds
//
// FUSION SUPPORT (Day 2 of Etap 3):
// - Bias + ReLU fused into micro-kernel
// - Y = ReLU(X × W + B) computed in registers before store
// - Zero extra memory traffic!
//
// MULTITHREADING (Optimized for Intel Alder Lake hybrid architecture):
// - 2 P-cores (4 threads) + 4 E-cores (4 threads) = 6 cores, 8 threads
// - Use thread binding for better cache locality
// - Parallelize over M dimension (ic loop) for best load balance
//
// Expected Performance:
// - Single-threaded: ~25-30 GFLOPS (one P-core)
// - Multi-threaded: ~80-120 GFLOPS (all cores)

#include "PipelineTests.h"
#include "context/HardwareInfo.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <immintrin.h>  // AVX2 intrinsics
#include <cmath>
#include <omp.h>        // OpenMP for parallelization
#include <thread>
#include <vector>
#include <atomic>
#include <pthread.h>    // Thread pinning (pthread_setaffinity_np)

#ifdef TENZO_HAS_OPENBLAS
#include <cblas.h>
#endif

namespace {

// Micro-kernel dimensions
constexpr int MR = 6;
constexpr int NR = 16;

// Cache blocking sizes (tuned for L1/L2/L3)
// L2 = 1280KB per core, L3 = 10MB shared
constexpr int KC = 256;   // K-panel depth (~24KB for B panel fits in L1)
constexpr int MC = 96;    // M-block size (96*256*4 = 98KB fits in L2)
constexpr int NC = 256;   // N-block size

// OpenMP configuration
inline void configureOpenMP() {
    // For hybrid architecture (P-cores + E-cores):
    // - Use close binding to keep threads on same cores
    // - Limit to physical cores for best performance
    omp_set_dynamic(0);  // Disable dynamic adjustment

    // Use number of physical cores (6 on Alder Lake i5-1235U)
    int numCores = omp_get_num_procs();
    // For hybrid, we might want to use fewer threads
    // But let's try all cores first
    omp_set_num_threads(numCores > 6 ? 6 : numCores);
}

// Aligned memory allocation
inline float* aligned_alloc_floats(size_t count) {
    return static_cast<float*>(aligned_alloc(32, count * sizeof(float)));
}

//===----------------------------------------------------------------------===//
// Pack Matrix B: B[K][N] → B_packed
// Layout: For each panel j, store B[0..K-1, j*NR..j*NR+NR-1] sequentially
// PADDING: Handles arbitrary N by zero-padding last panel if N % NR != 0
//===----------------------------------------------------------------------===//
void packB(const float* B, float* B_packed, int K, int N, int ldb) {
    // Number of panels = ceil(N / NR)
    int numFullPanels = N / NR;
    int remainder = N % NR;
    int numPanels = numFullPanels + (remainder > 0 ? 1 : 0);

    // Process full panels with vectorized code
    for (int panelIdx = 0; panelIdx < numFullPanels; panelIdx++) {
        int jStart = panelIdx * NR;
        float* dst = B_packed + panelIdx * K * NR;

        for (int k = 0; k < K; k++) {
            const float* src = B + k * ldb + jStart;

            // Copy 16 elements using AVX2
            __m256 v0 = _mm256_loadu_ps(src);
            __m256 v1 = _mm256_loadu_ps(src + 8);
            _mm256_storeu_ps(dst + k * NR, v0);
            _mm256_storeu_ps(dst + k * NR + 8, v1);
        }
    }

    // Handle last partial panel (if any) with zero padding
    if (remainder > 0) {
        int jStart = numFullPanels * NR;
        float* dst = B_packed + numFullPanels * K * NR;

        for (int k = 0; k < K; k++) {
            const float* src = B + k * ldb + jStart;

            // Copy valid elements and zero-pad the rest
            for (int j = 0; j < NR; j++) {
                if (jStart + j < N) {
                    dst[k * NR + j] = src[j];
                } else {
                    dst[k * NR + j] = 0.0f;  // Zero padding
                }
            }
        }
    }
}

//===----------------------------------------------------------------------===//
// Pack Matrix A: A[M][K] → A_packed
// Layout: Column-major within MR=6 panels
// A[i*MR+0..5, k] stored sequentially for each k
// PADDING: Handles arbitrary M by zero-padding last panel if M % MR != 0
//===----------------------------------------------------------------------===//
void packA(const float* A, float* A_packed, int M, int K, int lda) {
    // Number of panels = ceil(M / MR)
    int numFullPanels = M / MR;
    int remainder = M % MR;
    int numPanels = numFullPanels + (remainder > 0 ? 1 : 0);

    // Process full panels
    for (int panelIdx = 0; panelIdx < numFullPanels; panelIdx++) {
        int iStart = panelIdx * MR;
        float* dst = A_packed + panelIdx * K * MR;

        for (int k = 0; k < K; k++) {
            // Copy MR=6 elements from column k
            for (int i = 0; i < MR; i++) {
                dst[k * MR + i] = A[(iStart + i) * lda + k];
            }
        }
    }

    // Handle last partial panel (if any) with zero padding
    if (remainder > 0) {
        int iStart = numFullPanels * MR;
        float* dst = A_packed + numFullPanels * K * MR;

        for (int k = 0; k < K; k++) {
            // Copy valid rows and zero-pad the rest
            for (int i = 0; i < MR; i++) {
                if (iStart + i < M) {
                    dst[k * MR + i] = A[(iStart + i) * lda + k];
                } else {
                    dst[k * MR + i] = 0.0f;  // Zero padding
                }
            }
        }
    }
}

//===----------------------------------------------------------------------===//
// Micro-kernel 6x16 with explicit AVX2 FMA - FULLY OPTIMIZED
// - K-loop unrolled by 4 for better instruction-level parallelism
// - Software prefetching for next B panel
// - Explicit register naming to help compiler
//===----------------------------------------------------------------------===//
void microKernel6x16(const float* __restrict__ A_packed,
                     const float* __restrict__ B_packed,
                     float* __restrict__ C, int ldc, int K) {
    // 12 accumulators in explicit registers
    __m256 c00 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 c01 = _mm256_loadu_ps(C + 0 * ldc + 8);
    __m256 c10 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c11 = _mm256_loadu_ps(C + 1 * ldc + 8);
    __m256 c20 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
    __m256 c30 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 c31 = _mm256_loadu_ps(C + 3 * ldc + 8);
    __m256 c40 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 c41 = _mm256_loadu_ps(C + 4 * ldc + 8);
    __m256 c50 = _mm256_loadu_ps(C + 5 * ldc);
    __m256 c51 = _mm256_loadu_ps(C + 5 * ldc + 8);

    // Main K-loop, unrolled by 4
    int k = 0;
    for (; k + 3 < K; k += 4) {
        // Prefetch next B panel (2 cache lines ahead)
        _mm_prefetch((const char*)(B_packed + (k + 8) * NR), _MM_HINT_T0);

        // Iteration 0
        {
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

        // Iteration 1
        {
            __m256 b0 = _mm256_loadu_ps(B_packed + (k+1) * NR);
            __m256 b1 = _mm256_loadu_ps(B_packed + (k+1) * NR + 8);

            __m256 a0 = _mm256_broadcast_ss(A_packed + (k+1) * MR + 0);
            __m256 a1 = _mm256_broadcast_ss(A_packed + (k+1) * MR + 1);
            __m256 a2 = _mm256_broadcast_ss(A_packed + (k+1) * MR + 2);
            __m256 a3 = _mm256_broadcast_ss(A_packed + (k+1) * MR + 3);
            __m256 a4 = _mm256_broadcast_ss(A_packed + (k+1) * MR + 4);
            __m256 a5 = _mm256_broadcast_ss(A_packed + (k+1) * MR + 5);

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

        // Iteration 2
        {
            __m256 b0 = _mm256_loadu_ps(B_packed + (k+2) * NR);
            __m256 b1 = _mm256_loadu_ps(B_packed + (k+2) * NR + 8);

            __m256 a0 = _mm256_broadcast_ss(A_packed + (k+2) * MR + 0);
            __m256 a1 = _mm256_broadcast_ss(A_packed + (k+2) * MR + 1);
            __m256 a2 = _mm256_broadcast_ss(A_packed + (k+2) * MR + 2);
            __m256 a3 = _mm256_broadcast_ss(A_packed + (k+2) * MR + 3);
            __m256 a4 = _mm256_broadcast_ss(A_packed + (k+2) * MR + 4);
            __m256 a5 = _mm256_broadcast_ss(A_packed + (k+2) * MR + 5);

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

        // Iteration 3
        {
            __m256 b0 = _mm256_loadu_ps(B_packed + (k+3) * NR);
            __m256 b1 = _mm256_loadu_ps(B_packed + (k+3) * NR + 8);

            __m256 a0 = _mm256_broadcast_ss(A_packed + (k+3) * MR + 0);
            __m256 a1 = _mm256_broadcast_ss(A_packed + (k+3) * MR + 1);
            __m256 a2 = _mm256_broadcast_ss(A_packed + (k+3) * MR + 2);
            __m256 a3 = _mm256_broadcast_ss(A_packed + (k+3) * MR + 3);
            __m256 a4 = _mm256_broadcast_ss(A_packed + (k+3) * MR + 4);
            __m256 a5 = _mm256_broadcast_ss(A_packed + (k+3) * MR + 5);

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
    }

    // Cleanup loop for remaining K
    for (; k < K; k++) {
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

    // Store results
    _mm256_storeu_ps(C + 0 * ldc, c00);
    _mm256_storeu_ps(C + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C + 1 * ldc, c10);
    _mm256_storeu_ps(C + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc, c20);
    _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc, c30);
    _mm256_storeu_ps(C + 3 * ldc + 8, c31);
    _mm256_storeu_ps(C + 4 * ldc, c40);
    _mm256_storeu_ps(C + 4 * ldc + 8, c41);
    _mm256_storeu_ps(C + 5 * ldc, c50);
    _mm256_storeu_ps(C + 5 * ldc + 8, c51);
}

//===----------------------------------------------------------------------===//
// Micro-kernel 6x16 with FUSED Bias + ReLU - FULLY OPTIMIZED
// K-loop unrolled by 4, then applies Bias and ReLU in registers
//===----------------------------------------------------------------------===//
void microKernel6x16_BiasReLU(const float* __restrict__ A_packed,
                              const float* __restrict__ B_packed,
                              float* __restrict__ C, int ldc, int K,
                              const float* bias, bool use_relu) {
    // 12 accumulators in explicit registers
    __m256 c00 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 c01 = _mm256_loadu_ps(C + 0 * ldc + 8);
    __m256 c10 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c11 = _mm256_loadu_ps(C + 1 * ldc + 8);
    __m256 c20 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
    __m256 c30 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 c31 = _mm256_loadu_ps(C + 3 * ldc + 8);
    __m256 c40 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 c41 = _mm256_loadu_ps(C + 4 * ldc + 8);
    __m256 c50 = _mm256_loadu_ps(C + 5 * ldc);
    __m256 c51 = _mm256_loadu_ps(C + 5 * ldc + 8);

    // Main K-loop, unrolled by 4
    int k = 0;
    for (; k + 3 < K; k += 4) {
        _mm_prefetch((const char*)(B_packed + (k + 8) * NR), _MM_HINT_T0);

        #define KERNEL_ITER(kk) { \
            __m256 b0 = _mm256_loadu_ps(B_packed + (kk) * NR); \
            __m256 b1 = _mm256_loadu_ps(B_packed + (kk) * NR + 8); \
            __m256 a0 = _mm256_broadcast_ss(A_packed + (kk) * MR + 0); \
            __m256 a1 = _mm256_broadcast_ss(A_packed + (kk) * MR + 1); \
            __m256 a2 = _mm256_broadcast_ss(A_packed + (kk) * MR + 2); \
            __m256 a3 = _mm256_broadcast_ss(A_packed + (kk) * MR + 3); \
            __m256 a4 = _mm256_broadcast_ss(A_packed + (kk) * MR + 4); \
            __m256 a5 = _mm256_broadcast_ss(A_packed + (kk) * MR + 5); \
            c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01); \
            c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11); \
            c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21); \
            c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31); \
            c40 = _mm256_fmadd_ps(a4, b0, c40); c41 = _mm256_fmadd_ps(a4, b1, c41); \
            c50 = _mm256_fmadd_ps(a5, b0, c50); c51 = _mm256_fmadd_ps(a5, b1, c51); \
        }

        KERNEL_ITER(k);
        KERNEL_ITER(k+1);
        KERNEL_ITER(k+2);
        KERNEL_ITER(k+3);

        #undef KERNEL_ITER
    }

    // Cleanup
    for (; k < K; k++) {
        __m256 b0 = _mm256_loadu_ps(B_packed + k * NR);
        __m256 b1 = _mm256_loadu_ps(B_packed + k * NR + 8);
        __m256 a0 = _mm256_broadcast_ss(A_packed + k * MR + 0);
        __m256 a1 = _mm256_broadcast_ss(A_packed + k * MR + 1);
        __m256 a2 = _mm256_broadcast_ss(A_packed + k * MR + 2);
        __m256 a3 = _mm256_broadcast_ss(A_packed + k * MR + 3);
        __m256 a4 = _mm256_broadcast_ss(A_packed + k * MR + 4);
        __m256 a5 = _mm256_broadcast_ss(A_packed + k * MR + 5);
        c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31);
        c40 = _mm256_fmadd_ps(a4, b0, c40); c41 = _mm256_fmadd_ps(a4, b1, c41);
        c50 = _mm256_fmadd_ps(a5, b0, c50); c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    // Bias + ReLU fusion (in registers!)
    if (bias != nullptr) {
        __m256 bias0 = _mm256_loadu_ps(bias);
        __m256 bias1 = _mm256_loadu_ps(bias + 8);
        c00 = _mm256_add_ps(c00, bias0); c01 = _mm256_add_ps(c01, bias1);
        c10 = _mm256_add_ps(c10, bias0); c11 = _mm256_add_ps(c11, bias1);
        c20 = _mm256_add_ps(c20, bias0); c21 = _mm256_add_ps(c21, bias1);
        c30 = _mm256_add_ps(c30, bias0); c31 = _mm256_add_ps(c31, bias1);
        c40 = _mm256_add_ps(c40, bias0); c41 = _mm256_add_ps(c41, bias1);
        c50 = _mm256_add_ps(c50, bias0); c51 = _mm256_add_ps(c51, bias1);
    }

    if (use_relu) {
        __m256 zero = _mm256_setzero_ps();
        c00 = _mm256_max_ps(c00, zero); c01 = _mm256_max_ps(c01, zero);
        c10 = _mm256_max_ps(c10, zero); c11 = _mm256_max_ps(c11, zero);
        c20 = _mm256_max_ps(c20, zero); c21 = _mm256_max_ps(c21, zero);
        c30 = _mm256_max_ps(c30, zero); c31 = _mm256_max_ps(c31, zero);
        c40 = _mm256_max_ps(c40, zero); c41 = _mm256_max_ps(c41, zero);
        c50 = _mm256_max_ps(c50, zero); c51 = _mm256_max_ps(c51, zero);
    }

    // Store
    _mm256_storeu_ps(C + 0 * ldc, c00); _mm256_storeu_ps(C + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C + 1 * ldc, c10); _mm256_storeu_ps(C + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc, c20); _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc, c30); _mm256_storeu_ps(C + 3 * ldc + 8, c31);
    _mm256_storeu_ps(C + 4 * ldc, c40); _mm256_storeu_ps(C + 4 * ldc + 8, c41);
    _mm256_storeu_ps(C + 5 * ldc, c50); _mm256_storeu_ps(C + 5 * ldc + 8, c51);
}

//===----------------------------------------------------------------------===//
// Helper: Round up to next multiple
//===----------------------------------------------------------------------===//
inline int roundUp(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

//===----------------------------------------------------------------------===//
// Macro-kernel with 5-loop nest (GotoBLAS-style)
// Supports ARBITRARY matrix sizes with proper padding
//===----------------------------------------------------------------------===//
void gemmWithPacking(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {

    float* blockA = aligned_alloc_floats(MC * KC);
    float* blockB = aligned_alloc_floats(KC * NC);

    for (int jc = 0; jc < N; jc += NC) {
        int ncActual = std::min(NC, N - jc);
        int ncPadded = roundUp(ncActual, NR);

        for (int pc = 0; pc < K; pc += KC) {
            int kcActual = std::min(KC, K - pc);

            packB(B + pc * ldb + jc, blockB, kcActual, ncActual, ldb);

            for (int ic = 0; ic < M; ic += MC) {
                int mcActual = std::min(MC, M - ic);
                int mcPadded = roundUp(mcActual, MR);

                packA(A + ic * lda + pc, blockA, mcActual, kcActual, lda);

                int numMicroJ = ncPadded / NR;
                int numMicroI = mcPadded / MR;

                for (int jr = 0; jr < numMicroJ; jr++) {
                    for (int ir = 0; ir < numMicroI; ir++) {
                        int cRow = ic + ir * MR;
                        int cCol = jc + jr * NR;

                        if (cRow >= M || cCol >= N) continue;

                        const float* aPtr = blockA + ir * kcActual * MR;
                        const float* bPtr = blockB + jr * kcActual * NR;

                        int rowsToWrite = std::min(MR, M - cRow);
                        int colsToWrite = std::min(NR, N - cCol);

                        if (rowsToWrite == MR && colsToWrite == NR) {
                            float* cPtr = C + cRow * ldc + cCol;
                            microKernel6x16(aPtr, bPtr, cPtr, ldc, kcActual);
                        } else {
                            alignas(32) float tempC[MR * NR] = {0};

                            for (int i = 0; i < rowsToWrite; i++) {
                                for (int j = 0; j < colsToWrite; j++) {
                                    tempC[i * NR + j] = C[(cRow + i) * ldc + (cCol + j)];
                                }
                            }

                            microKernel6x16(aPtr, bPtr, tempC, NR, kcActual);

                            for (int i = 0; i < rowsToWrite; i++) {
                                for (int j = 0; j < colsToWrite; j++) {
                                    C[(cRow + i) * ldc + (cCol + j)] = tempC[i * NR + j];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    free(blockA);
    free(blockB);
}

//===----------------------------------------------------------------------===//
// ADAPTIVE Macro-kernel - Uses topology-computed KC/MC/NC from cache sizes
// Instead of hardcoded KC=256/MC=96/NC=256, computes from actual L1/L2/L3
//===----------------------------------------------------------------------===//
void gemmAdaptive(const float* A, const float* B, float* C,
                  int M, int N, int K, int lda, int ldb, int ldc,
                  int aKC, int aMC, int aNC) {

    float* blockA = aligned_alloc_floats(aMC * aKC);
    float* blockB = aligned_alloc_floats(aKC * aNC);

    for (int jc = 0; jc < N; jc += aNC) {
        int ncActual = std::min(aNC, N - jc);
        int ncPadded = roundUp(ncActual, NR);

        for (int pc = 0; pc < K; pc += aKC) {
            int kcActual = std::min(aKC, K - pc);

            packB(B + pc * ldb + jc, blockB, kcActual, ncActual, ldb);

            for (int ic = 0; ic < M; ic += aMC) {
                int mcActual = std::min(aMC, M - ic);
                int mcPadded = roundUp(mcActual, MR);

                packA(A + ic * lda + pc, blockA, mcActual, kcActual, lda);

                int numMicroJ = ncPadded / NR;
                int numMicroI = mcPadded / MR;

                for (int jr = 0; jr < numMicroJ; jr++) {
                    for (int ir = 0; ir < numMicroI; ir++) {
                        int cRow = ic + ir * MR;
                        int cCol = jc + jr * NR;
                        if (cRow >= M || cCol >= N) continue;

                        const float* aPtr = blockA + ir * kcActual * MR;
                        const float* bPtr = blockB + jr * kcActual * NR;

                        int rowsToWrite = std::min(MR, M - cRow);
                        int colsToWrite = std::min(NR, N - cCol);

                        if (rowsToWrite == MR && colsToWrite == NR) {
                            microKernel6x16(aPtr, bPtr, C + cRow * ldc + cCol, ldc, kcActual);
                        } else {
                            alignas(32) float tempC[MR * NR] = {0};
                            for (int i = 0; i < rowsToWrite; i++)
                                for (int j = 0; j < colsToWrite; j++)
                                    tempC[i * NR + j] = C[(cRow + i) * ldc + (cCol + j)];

                            microKernel6x16(aPtr, bPtr, tempC, NR, kcActual);

                            for (int i = 0; i < rowsToWrite; i++)
                                for (int j = 0; j < colsToWrite; j++)
                                    C[(cRow + i) * ldc + (cCol + j)] = tempC[i * NR + j];
                        }
                    }
                }
            }
        }
    }

    free(blockA);
    free(blockB);
}

//===----------------------------------------------------------------------===//
// PER-CORE INDEPENDENT GEMM — Zero-sync parallelism
//
// Philosophy: Instead of splitting one GEMM across threads (which creates
// synchronization, false sharing, and redundant packing), we split the M
// dimension into INDEPENDENT sub-problems. Each physical core runs its own
// COMPLETE GotoBLAS GEMM pipeline on its rows — no barriers, no shared data.
//
// Each core:
//   1. Gets pinned to its physical CPU (pthread_setaffinity_np)
//   2. Allocates its OWN packing buffers (no false sharing)
//   3. Runs a complete 5-loop GEMM on its assigned rows
//   4. Writes to its own non-overlapping C region
//
// This mirrors how real production BLAS libraries work (e.g. BLIS).
//===----------------------------------------------------------------------===//
void gemmPerCoreIndependent(const float* A, const float* B, float* C,
                            int M, int N, int K, int lda, int ldb, int ldc,
                            const tenzo::TopologyInfo& topo,
                            int aKC, int aMC, int aNC) {

    // Build work assignments using topology
    auto split = tenzo::HeterogeneousWorkSplit::compute(M, topo);
    int numWorkers = (int)split.assignments.size();

    if (numWorkers <= 1) {
        gemmAdaptive(A, B, C, M, N, K, lda, ldb, ldc, aKC, aMC, aNC);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(numWorkers);

    for (auto& tw : split.assignments) {
        workers.emplace_back([&, tw, aKC, aMC, aNC]() {
            // Pin to specific CPU core
            #ifdef __linux__
            if (tw.cpuId >= 0) {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(tw.cpuId, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            }
            #endif

            int mStart = tw.rowStart;
            int mEnd = tw.rowEnd;
            if (mStart >= mEnd) return;
            int localM = mEnd - mStart;

            // Each core has its OWN packing buffers — no false sharing
            float* myBlockA = aligned_alloc_floats(aMC * aKC);
            float* myBlockB = aligned_alloc_floats(aKC * aNC);

            // Run a COMPLETE, INDEPENDENT 5-loop GEMM on my rows
            for (int jc = 0; jc < N; jc += aNC) {
                int ncActual = std::min(aNC, N - jc);
                int ncPadded = roundUp(ncActual, NR);

                for (int pc = 0; pc < K; pc += aKC) {
                    int kcActual = std::min(aKC, K - pc);

                    // Each core packs its own B — this IS redundant across cores,
                    // but avoids synchronization entirely. For large K and N,
                    // the B panel lives in L3 anyway.
                    packB(B + pc * ldb + jc, myBlockB, kcActual, ncActual, ldb);

                    for (int icRel = 0; icRel < localM; icRel += aMC) {
                        int ic = mStart + icRel;
                        int mcActual = std::min(aMC, mEnd - ic);
                        int mcPadded = roundUp(mcActual, MR);

                        packA(A + ic * lda + pc, myBlockA, mcActual, kcActual, lda);

                        int numMicroJ = ncPadded / NR;
                        int numMicroI = mcPadded / MR;

                        for (int jr = 0; jr < numMicroJ; jr++) {
                            for (int ir = 0; ir < numMicroI; ir++) {
                                int cRow = ic + ir * MR;
                                int cCol = jc + jr * NR;
                                if (cRow >= mEnd || cCol >= N) continue;

                                const float* aPtr = myBlockA + ir * kcActual * MR;
                                const float* bPtr = myBlockB + jr * kcActual * NR;

                                int rowsToWrite = std::min(MR, mEnd - cRow);
                                int colsToWrite = std::min(NR, N - cCol);

                                if (rowsToWrite == MR && colsToWrite == NR) {
                                    microKernel6x16(aPtr, bPtr,
                                        C + cRow * ldc + cCol, ldc, kcActual);
                                } else {
                                    alignas(32) float tempC[MR * NR] = {0};
                                    for (int i = 0; i < rowsToWrite; i++)
                                        for (int j = 0; j < colsToWrite; j++)
                                            tempC[i * NR + j] = C[(cRow + i) * ldc + (cCol + j)];

                                    microKernel6x16(aPtr, bPtr, tempC, NR, kcActual);

                                    for (int i = 0; i < rowsToWrite; i++)
                                        for (int j = 0; j < colsToWrite; j++)
                                            C[(cRow + i) * ldc + (cCol + j)] = tempC[i * NR + j];
                                }
                            }
                        }
                    }
                }
            }

            free(myBlockA);
            free(myBlockB);
        });
    }

    for (auto& w : workers) w.join();
}

// Per-core-type blocking: each core type gets its own KC/MC/NC based on its cache
void gemmPerCoreIndependentPerType(const float* A, const float* B, float* C,
                                    int M, int N, int K, int lda, int ldb, int ldc,
                                    const tenzo::HardwareInfo& hwInfo) {

    auto split = tenzo::HeterogeneousWorkSplit::compute(M, hwInfo.topology);
    int numWorkers = (int)split.assignments.size();

    if (numWorkers <= 1) {
        auto p = hwInfo.getOptimalMicroKernelParams();
        gemmAdaptive(A, B, C, M, N, K, lda, ldb, ldc, p.KC, p.MC, p.NC);
        return;
    }

    // Pre-compute blocking params for each core type
    auto pParams = hwInfo.getOptimalMicroKernelParamsForCoreType(tenzo::CoreType::P_CORE);
    auto eParams = hwInfo.getOptimalMicroKernelParamsForCoreType(tenzo::CoreType::E_CORE);

    std::vector<std::thread> workers;
    workers.reserve(numWorkers);

    for (auto& tw : split.assignments) {
        workers.emplace_back([&, tw, pParams, eParams]() {
            // Pin to specific CPU core
            #ifdef __linux__
            if (tw.cpuId >= 0) {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(tw.cpuId, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            }
            #endif

            // Select blocking params for THIS core's type
            int myKC, myMC, myNC;
            if (tw.coreType == tenzo::CoreType::P_CORE) {
                myKC = pParams.KC; myMC = pParams.MC; myNC = pParams.NC;
            } else if (tw.coreType == tenzo::CoreType::E_CORE) {
                myKC = eParams.KC; myMC = eParams.MC; myNC = eParams.NC;
            } else {
                myKC = pParams.KC; myMC = pParams.MC; myNC = pParams.NC;
            }

            int mStart = tw.rowStart;
            int mEnd = tw.rowEnd;
            if (mStart >= mEnd) return;
            int localM = mEnd - mStart;

            float* myBlockA = aligned_alloc_floats(myMC * myKC);
            float* myBlockB = aligned_alloc_floats(myKC * myNC);

            for (int jc = 0; jc < N; jc += myNC) {
                int ncActual = std::min(myNC, N - jc);
                int ncPadded = roundUp(ncActual, NR);

                for (int pc = 0; pc < K; pc += myKC) {
                    int kcActual = std::min(myKC, K - pc);

                    packB(B + pc * ldb + jc, myBlockB, kcActual, ncActual, ldb);

                    for (int icRel = 0; icRel < localM; icRel += myMC) {
                        int ic = mStart + icRel;
                        int mcActual = std::min(myMC, mEnd - ic);
                        int mcPadded = roundUp(mcActual, MR);

                        packA(A + ic * lda + pc, myBlockA, mcActual, kcActual, lda);

                        int numMicroJ = ncPadded / NR;
                        int numMicroI = mcPadded / MR;

                        for (int jr = 0; jr < numMicroJ; jr++) {
                            for (int ir = 0; ir < numMicroI; ir++) {
                                int cRow = ic + ir * MR;
                                int cCol = jc + jr * NR;
                                if (cRow >= mEnd || cCol >= N) continue;

                                const float* aPtr = myBlockA + ir * kcActual * MR;
                                const float* bPtr = myBlockB + jr * kcActual * NR;

                                int rowsToWrite = std::min(MR, mEnd - cRow);
                                int colsToWrite = std::min(NR, N - cCol);

                                if (rowsToWrite == MR && colsToWrite == NR) {
                                    microKernel6x16(aPtr, bPtr,
                                        C + cRow * ldc + cCol, ldc, kcActual);
                                } else {
                                    alignas(32) float tempC[MR * NR] = {0};
                                    for (int i = 0; i < rowsToWrite; i++)
                                        for (int j = 0; j < colsToWrite; j++)
                                            tempC[i * NR + j] = C[(cRow + i) * ldc + (cCol + j)];

                                    microKernel6x16(aPtr, bPtr, tempC, NR, kcActual);

                                    for (int i = 0; i < rowsToWrite; i++)
                                        for (int j = 0; j < colsToWrite; j++)
                                            C[(cRow + i) * ldc + (cCol + j)] = tempC[i * NR + j];
                                }
                            }
                        }
                    }
                }
            }

            free(myBlockA);
            free(myBlockB);
        });
    }

    for (auto& w : workers) w.join();
}

//===----------------------------------------------------------------------===//
// PARALLEL Macro-kernel - Adaptive parallelism for hybrid architectures
//
// Key optimizations for Intel Alder Lake (P-cores + E-cores):
// 1. Limit threads to avoid E-cores (they cause barrier slowdown)
// 2. Use adaptive threshold based on problem size
// 3. Each thread handles contiguous M rows for better cache locality
// 4. Respects OMP_NUM_THREADS environment variable
//===----------------------------------------------------------------------===//
void gemmWithPacking_Parallel(const float* A, const float* B, float* C,
                              int M, int N, int K, int lda, int ldb, int ldc) {

    // Get max threads from environment or default to 4 (P-cores)
    int maxThreads = 4;  // Default: P-cores only
    const char* envThreads = std::getenv("OMP_NUM_THREADS");
    if (envThreads) {
        maxThreads = std::atoi(envThreads);
        if (maxThreads < 1) maxThreads = 1;
        if (maxThreads > 8) maxThreads = 8;
    }

    // Calculate total work (in FLOPs)
    long long totalWork = (long long)M * N * K * 2;

    // Adaptive thread count based on problem size
    int numThreads;

    if (totalWork < 256LL * 256 * 256 * 2) {
        // Small problem: single-threaded is faster (no thread overhead)
        numThreads = 1;
    } else if (totalWork < 512LL * 512 * 512 * 2) {
        // Medium problem: 2 threads
        numThreads = std::min(2, maxThreads);
    } else {
        // Large problem: use configured max threads
        numThreads = maxThreads;
    }

    // Fall back to single-threaded for very small matrices
    if (M < 192 || N < 192) {
        gemmWithPacking(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }

    // If only 1 thread needed, use optimized single-threaded version
    if (numThreads == 1) {
        gemmWithPacking(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }

    // Simple approach: divide M among threads
    // Each thread computes its own rows completely
    auto threadWork = [&](int tid, int mStart, int mEnd) {
        if (mStart >= mEnd) return;

        int localM = mEnd - mStart;

        float* blockA = aligned_alloc_floats(MC * KC);
        float* blockB = aligned_alloc_floats(KC * NC);

        for (int jc = 0; jc < N; jc += NC) {
            int ncActual = std::min(NC, N - jc);
            int ncPadded = roundUp(ncActual, NR);

            for (int pc = 0; pc < K; pc += KC) {
                int kcActual = std::min(KC, K - pc);

                // Each thread packs its own B (some redundancy but simpler)
                packB(B + pc * ldb + jc, blockB, kcActual, ncActual, ldb);

                for (int icRel = 0; icRel < localM; icRel += MC) {
                    int ic = mStart + icRel;
                    int mcActual = std::min(MC, mEnd - ic);
                    int mcPadded = roundUp(mcActual, MR);

                    packA(A + ic * lda + pc, blockA, mcActual, kcActual, lda);

                    int numMicroJ = ncPadded / NR;
                    int numMicroI = mcPadded / MR;

                    for (int jr = 0; jr < numMicroJ; jr++) {
                        for (int ir = 0; ir < numMicroI; ir++) {
                            int cRow = ic + ir * MR;
                            int cCol = jc + jr * NR;

                            if (cRow >= mEnd || cCol >= N) continue;

                            const float* aPtr = blockA + ir * kcActual * MR;
                            const float* bPtr = blockB + jr * kcActual * NR;

                            int rowsToWrite = std::min(MR, mEnd - cRow);
                            int colsToWrite = std::min(NR, N - cCol);

                            if (rowsToWrite == MR && colsToWrite == NR) {
                                float* cPtr = C + cRow * ldc + cCol;
                                microKernel6x16(aPtr, bPtr, cPtr, ldc, kcActual);
                            } else {
                                alignas(32) float tempC[MR * NR] = {0};

                                for (int i = 0; i < rowsToWrite; i++) {
                                    for (int j = 0; j < colsToWrite; j++) {
                                        tempC[i * NR + j] = C[(cRow + i) * ldc + (cCol + j)];
                                    }
                                }

                                microKernel6x16(aPtr, bPtr, tempC, NR, kcActual);

                                for (int i = 0; i < rowsToWrite; i++) {
                                    for (int j = 0; j < colsToWrite; j++) {
                                        C[(cRow + i) * ldc + (cCol + j)] = tempC[i * NR + j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        free(blockA);
        free(blockB);
    };

    // Divide M among threads
    int rowsPerThread = (M + numThreads - 1) / numThreads;
    // Round to MC for better blocking
    rowsPerThread = ((rowsPerThread + MC - 1) / MC) * MC;

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; t++) {
        int mStart = t * rowsPerThread;
        int mEnd = std::min(mStart + rowsPerThread, M);

        if (mStart < M) {
            threads.emplace_back(threadWork, t, mStart, mEnd);
        }
    }

    for (auto& t : threads) {
        t.join();
    }
}

//===----------------------------------------------------------------------===//
// GEMM with FUSED Bias + ReLU (Day 2 of Etap 3)
// Y = ReLU(A × B + bias) - complete neural network layer in one pass!
//===----------------------------------------------------------------------===//
void gemmWithPacking_BiasReLU(const float* A, const float* B, float* C,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              const float* bias, bool use_relu) {
    float* blockA = aligned_alloc_floats(MC * KC);
    float* blockB = aligned_alloc_floats(KC * NC);

    for (int jc = 0; jc < N; jc += NC) {
        int ncActual = std::min(NC, N - jc);
        int ncPadded = roundUp(ncActual, NR);

        for (int pc = 0; pc < K; pc += KC) {
            int kcActual = std::min(KC, K - pc);
            bool isLastKPanel = (pc + KC >= K);

            packB(B + pc * ldb + jc, blockB, kcActual, ncActual, ldb);

            for (int ic = 0; ic < M; ic += MC) {
                int mcActual = std::min(MC, M - ic);
                int mcPadded = roundUp(mcActual, MR);

                packA(A + ic * lda + pc, blockA, mcActual, kcActual, lda);

                int numMicroJ = ncPadded / NR;
                int numMicroI = mcPadded / MR;

                for (int jr = 0; jr < numMicroJ; jr++) {
                    for (int ir = 0; ir < numMicroI; ir++) {
                        int cRow = ic + ir * MR;
                        int cCol = jc + jr * NR;

                        if (cRow >= M || cCol >= N) continue;

                        const float* aPtr = blockA + ir * kcActual * MR;
                        const float* bPtr = blockB + jr * kcActual * NR;

                        int rowsToWrite = std::min(MR, M - cRow);
                        int colsToWrite = std::min(NR, N - cCol);

                        const float* biasPtr = nullptr;
                        if (bias != nullptr && isLastKPanel) {
                            biasPtr = bias + cCol;
                        }
                        bool applyRelu = use_relu && isLastKPanel;

                        if (rowsToWrite == MR && colsToWrite == NR) {
                            float* cPtr = C + cRow * ldc + cCol;
                            microKernel6x16_BiasReLU(aPtr, bPtr, cPtr, ldc, kcActual,
                                                    biasPtr, applyRelu);
                        } else {
                            alignas(32) float tempC[MR * NR] = {0};
                            alignas(32) float tempBias[NR] = {0};

                            for (int i = 0; i < rowsToWrite; i++) {
                                for (int j = 0; j < colsToWrite; j++) {
                                    tempC[i * NR + j] = C[(cRow + i) * ldc + (cCol + j)];
                                }
                            }

                            if (biasPtr != nullptr) {
                                for (int j = 0; j < colsToWrite; j++) {
                                    tempBias[j] = biasPtr[j];
                                }
                            }

                            microKernel6x16_BiasReLU(aPtr, bPtr, tempC, NR, kcActual,
                                                    biasPtr ? tempBias : nullptr, applyRelu);

                            for (int i = 0; i < rowsToWrite; i++) {
                                for (int j = 0; j < colsToWrite; j++) {
                                    C[(cRow + i) * ldc + (cCol + j)] = tempC[i * NR + j];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    free(blockA);
    free(blockB);
}

//===----------------------------------------------------------------------===//
// Naive GEMM (for comparison)
//===----------------------------------------------------------------------===//
void gemmNaive(const float* A, const float* B, float* C,
               int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = C[i * ldc + j];
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

//===----------------------------------------------------------------------===//
// Benchmark runner
//===----------------------------------------------------------------------===//
double runGemmBenchmark(void (*gemm)(const float*, const float*, float*, int, int, int, int, int, int),
                    int M, int N, int K, int warmup, int iterations) {
    // Allocate matrices
    float* A = aligned_alloc_floats(M * K);
    float* B = aligned_alloc_floats(K * N);
    float* C = aligned_alloc_floats(M * N);

    // Initialize
    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(rand() % 100) / 100.0f;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        memset(C, 0, M * N * sizeof(float));
        gemm(A, B, C, M, N, K, K, N, N);
    }

    // Timed runs - DON'T include memset in timing!
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        // Reset C before each iteration (outside timing for naive, but our kernel accumulates)
        // Actually for fair comparison, we should NOT reset - just measure pure compute
        gemm(A, B, C, M, N, K, K, N, N);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    double avgTime = elapsed / iterations;

    // Calculate GFLOPS: 2 * M * N * K operations per GEMM
    double flops = 2.0 * M * N * K;
    double gflops = (flops / avgTime) / 1e9;

    free(A);
    free(B);
    free(C);

    return gflops;
}

// Verify correctness of BiasReLU fusion
bool verifyBiasReLUCorrectness(int M, int N, int K) {
    float* A = aligned_alloc_floats(M * K);
    float* B = aligned_alloc_floats(K * N);
    float* bias = aligned_alloc_floats(N);
    float* C_separate = aligned_alloc_floats(M * N);  // MatMul + Bias + ReLU done separately
    float* C_fused = aligned_alloc_floats(M * N);     // All fused in micro-kernel

    // Initialize with reproducible values
    srand(123);
    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 100) / 100.0f - 0.5f;  // [-0.5, 0.5]
    for (int i = 0; i < K * N; i++) B[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    for (int i = 0; i < N; i++) bias[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    memset(C_separate, 0, M * N * sizeof(float));
    memset(C_fused, 0, M * N * sizeof(float));

    // Method 1: Separate operations (baseline)
    // Step 1: MatMul
    gemmWithPacking(A, B, C_separate, M, N, K, K, N, N);
    // Step 2: Add Bias (each row gets same bias)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C_separate[i * N + j] += bias[j];
        }
    }
    // Step 3: ReLU
    for (int i = 0; i < M * N; i++) {
        C_separate[i] = std::max(0.0f, C_separate[i]);
    }

    // Method 2: Fused in micro-kernel
    gemmWithPacking_BiasReLU(A, B, C_fused, M, N, K, K, N, N, bias, true);

    // Compare results
    float maxError = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = std::abs(C_separate[i] - C_fused[i]);
        maxError = std::max(maxError, error);
    }

    free(A);
    free(B);
    free(bias);
    free(C_separate);
    free(C_fused);

    return maxError < 1e-3f;
}

// Verify correctness
bool verifyCorrectness(int M, int N, int K) {
    float* A = aligned_alloc_floats(M * K);
    float* B = aligned_alloc_floats(K * N);
    float* C_naive = aligned_alloc_floats(M * N);
    float* C_packed = aligned_alloc_floats(M * N);

    // Initialize with reproducible values
    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(rand() % 100) / 100.0f;
    memset(C_naive, 0, M * N * sizeof(float));
    memset(C_packed, 0, M * N * sizeof(float));

    // Run both versions
    gemmNaive(A, B, C_naive, M, N, K, K, N, N);
    gemmWithPacking(A, B, C_packed, M, N, K, K, N, N);

    // Compare results
    float maxError = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = std::abs(C_naive[i] - C_packed[i]);
        maxError = std::max(maxError, error);
    }

    free(A);
    free(B);
    free(C_naive);
    free(C_packed);

    // Allow small numerical error due to FMA vs separate mul+add
    return maxError < 1e-3f;
}

} // anonymous namespace

namespace tenzo {

void runGEMMEndToEndBenchmark(mlir::MLIRContext &context) {
    llvm::outs() << "\n";
    llvm::outs() << "╔════════════════════════════════════════════════════════╗\n";
    llvm::outs() << "║  🚀 GEMM END-TO-END BENCHMARK (Native AVX2)           ║\n";
    llvm::outs() << "║  GotoBLAS-style Packing + 6x16 Micro-kernel           ║\n";
    llvm::outs() << "║  ✨ PADDING + BIAS/ReLU FUSION SUPPORT                ║\n";
    llvm::outs() << "╚════════════════════════════════════════════════════════╝\n\n";

    auto hwInfo = HardwareInfo::detect();
    hwInfo.print();
    llvm::outs() << "\n";

    // Configuration
    llvm::outs() << "📊 Configuration:\n";
    llvm::outs() << "   MR=" << MR << ", NR=" << NR << " (micro-kernel)\n";
    llvm::outs() << "   KC=" << KC << ", MC=" << MC << ", NC=" << NC << " (blocking)\n";
    llvm::outs() << "   Warmup: 5, Iterations: 20\n\n";

    // ═══════════════════════════════════════════════════════════════════════
    // PART 1: Verify Padding Correctness
    // ═══════════════════════════════════════════════════════════════════════
    llvm::outs() << "🔍 [Day 1] Verifying PADDING correctness...\n";

    // Test 1: Perfect divisible (baseline)
    llvm::outs() << "   Test 96x96x256 (perfect divisible)... ";
    if (verifyCorrectness(96, 96, 256)) {
        llvm::outs() << "✅ PASS\n";
    } else {
        llvm::outs() << "❌ FAIL\n";
        return;
    }

    // Test 2: M not divisible by MR=6
    llvm::outs() << "   Test 100x96x256 (M=100, not divisible by 6)... ";
    if (verifyCorrectness(100, 96, 256)) {
        llvm::outs() << "✅ PASS\n";
    } else {
        llvm::outs() << "❌ FAIL\n";
        return;
    }

    // Test 3: N not divisible by NR=16
    llvm::outs() << "   Test 96x100x256 (N=100, not divisible by 16)... ";
    if (verifyCorrectness(96, 100, 256)) {
        llvm::outs() << "✅ PASS\n";
    } else {
        llvm::outs() << "❌ FAIL\n";
        return;
    }

    // Test 4: Both M and N not divisible
    llvm::outs() << "   Test 253x257x128 (neither divisible)... ";
    if (verifyCorrectness(253, 257, 128)) {
        llvm::outs() << "✅ PASS\n";
    } else {
        llvm::outs() << "❌ FAIL\n";
        return;
    }

    // Test 5: MNIST-like size (784 input features)
    llvm::outs() << "   Test 784x128x64 (MNIST-like layer)... ";
    if (verifyCorrectness(784, 128, 64)) {
        llvm::outs() << "✅ PASS\n";
    } else {
        llvm::outs() << "❌ FAIL\n";
        return;
    }

    llvm::outs() << "   ✅ All padding tests passed!\n\n";

    // ═══════════════════════════════════════════════════════════════════════
    // PART 2: Verify BiasReLU Fusion Correctness
    // ═══════════════════════════════════════════════════════════════════════
    llvm::outs() << "🔍 [Day 2] Verifying BIAS+ReLU FUSION correctness...\n";

    llvm::outs() << "   Test 96x96x256 (divisible)... ";
    if (verifyBiasReLUCorrectness(96, 96, 256)) {
        llvm::outs() << "✅ PASS\n";
    } else {
        llvm::outs() << "❌ FAIL\n";
        return;
    }

    llvm::outs() << "   Test 100x100x128 (non-divisible)... ";
    if (verifyBiasReLUCorrectness(100, 100, 128)) {
        llvm::outs() << "✅ PASS\n";
    } else {
        llvm::outs() << "❌ FAIL\n";
        return;
    }

    llvm::outs() << "   Test 784x128x64 (MNIST layer)... ";
    if (verifyBiasReLUCorrectness(784, 128, 64)) {
        llvm::outs() << "✅ PASS\n";
    } else {
        llvm::outs() << "❌ FAIL\n";
        return;
    }

    llvm::outs() << "   ✅ All BiasReLU fusion tests passed!\n\n";

    // ═══════════════════════════════════════════════════════════════════════
    // PART 3: Performance Benchmarks
    // ═══════════════════════════════════════════════════════════════════════

    // Test sizes
    struct TestCase { int M, N, K; };
    TestCase tests[] = {
        {256, 256, 256},
        {512, 512, 512},
        {768, 768, 768},
        {1024, 1024, 1024},
    };

    llvm::outs() << "📈 Performance Results (MatMul only):\n";
    llvm::outs() << "╔══════════════╦═══════════════════╦═══════════════════╦════════════╗\n";
    llvm::outs() << "║   Size       ║  Naive (GFLOPS)   ║  Packed (GFLOPS)  ║   Speedup  ║\n";
    llvm::outs() << "╠══════════════╬═══════════════════╬═══════════════════╬════════════╣\n";

    for (const auto& test : tests) {
        // Run naive benchmark (fewer iterations due to slowness)
        double gflopsNaive = runGemmBenchmark(gemmNaive, test.M, test.N, test.K, 2, 5);

        // Run packed benchmark - more warmup and iterations for stable results
        double gflopsPacked = runGemmBenchmark(gemmWithPacking, test.M, test.N, test.K, 10, 50);

        double speedup = gflopsPacked / gflopsNaive;

        char buf[256];
        snprintf(buf, sizeof(buf),
                "║ %4dx%4dx%4d ║      %6.1f       ║      %6.1f       ║    %5.1fx   ║\n",
                test.M, test.N, test.K, gflopsNaive, gflopsPacked, speedup);
        llvm::outs() << buf;
    }

    llvm::outs() << "╚══════════════╩═══════════════════╩═══════════════════╩════════════╝\n\n";

    // ═══════════════════════════════════════════════════════════════════════
    // PART 4: BiasReLU Fusion Performance (Day 2 highlight!)
    // ═══════════════════════════════════════════════════════════════════════
    llvm::outs() << "⚡ [Day 2] BiasReLU Fusion Performance:\n";
    llvm::outs() << "╔══════════════╦═══════════════════╦═══════════════════╦════════════╗\n";
    llvm::outs() << "║   Size       ║  MatMul (GFLOPS)  ║  +Bias+ReLU       ║  Overhead  ║\n";
    llvm::outs() << "╠══════════════╬═══════════════════╬═══════════════════╬════════════╣\n";

    for (const auto& test : tests) {
        // Allocate bias for this test
        float* bias = aligned_alloc_floats(test.N);
        for (int i = 0; i < test.N; i++) bias[i] = 0.1f;

        // Benchmark pure MatMul
        double gflopsPure = runGemmBenchmark(gemmWithPacking, test.M, test.N, test.K, 10, 50);

        // Benchmark MatMul + Bias + ReLU (fused)
        float* A = aligned_alloc_floats(test.M * test.K);
        float* B = aligned_alloc_floats(test.K * test.N);
        float* C = aligned_alloc_floats(test.M * test.N);

        for (int i = 0; i < test.M * test.K; i++) A[i] = 0.5f;
        for (int i = 0; i < test.K * test.N; i++) B[i] = 0.5f;
        memset(C, 0, test.M * test.N * sizeof(float));

        // Warmup
        for (int i = 0; i < 10; i++) {
            gemmWithPacking_BiasReLU(A, B, C, test.M, test.N, test.K, test.K, test.N, test.N, bias, true);
        }

        // Timed runs - DON'T include memset
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 50; i++) {
            gemmWithPacking_BiasReLU(A, B, C, test.M, test.N, test.K, test.K, test.N, test.N, bias, true);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        double avgTime = elapsed / 50;
        double flops = 2.0 * test.M * test.N * test.K;
        double gflopsFused = (flops / avgTime) / 1e9;

        double overhead = ((gflopsPure / gflopsFused) - 1.0) * 100.0;

        char buf[256];
        snprintf(buf, sizeof(buf),
                "║ %4dx%4dx%4d ║      %6.1f       ║      %6.1f       ║   %+5.1f%%   ║\n",
                test.M, test.N, test.K, gflopsPure, gflopsFused, overhead);
        llvm::outs() << buf;

        free(A);
        free(B);
        free(C);
        free(bias);
    }

    llvm::outs() << "╚══════════════╩═══════════════════╩═══════════════════╩════════════╝\n\n";

    // ═══════════════════════════════════════════════════════════════════════
    // PART 5: Parallel vs Single-threaded Performance
    // ═══════════════════════════════════════════════════════════════════════
    llvm::outs() << "🚀 [Multi-threading] Single vs Parallel (Adaptive) Performance:\n";
    llvm::outs() << "   CPU: Intel Alder Lake (2 P-cores + 4 E-cores)\n";

    // Show actual thread config
    const char* envThreads = std::getenv("OMP_NUM_THREADS");
    int maxT = envThreads ? std::atoi(envThreads) : 4;
    llvm::outs() << "   Max threads: " << maxT << " (set via OMP_NUM_THREADS, default=4 for P-cores)\n";
    llvm::outs() << "   Strategy: Small=1T, Medium=2T, Large=" << maxT << "T\n\n";

    llvm::outs() << "╔══════════════╦═══════════════════╦═══════════════════╦════════════╗\n";
    llvm::outs() << "║   Size       ║  Single (GFLOPS)  ║  Parallel (GFLOPS)║   Speedup  ║\n";
    llvm::outs() << "╠══════════════╬═══════════════════╬═══════════════════╬════════════╣\n";

    for (const auto& test : tests) {
        // Single-threaded
        double gflopsSingle = runGemmBenchmark(gemmWithPacking, test.M, test.N, test.K, 5, 30);

        // Parallel
        double gflopsParallel = runGemmBenchmark(gemmWithPacking_Parallel, test.M, test.N, test.K, 5, 30);

        double speedup = gflopsParallel / gflopsSingle;

        char buf[256];
        snprintf(buf, sizeof(buf),
                "║ %4dx%4dx%4d ║      %6.1f       ║      %6.1f       ║    %5.2fx   ║\n",
                test.M, test.N, test.K, gflopsSingle, gflopsParallel, speedup);
        llvm::outs() << buf;
    }

    llvm::outs() << "╚══════════════╩═══════════════════╩═══════════════════╩════════════╝\n\n";

    // ═══════════════════════════════════════════════════════════════════════
    // PART 6: True 2-Core Parallel - Run 2 independent GEMMs simultaneously
    // This avoids the shared-data synchronization overhead
    // ═══════════════════════════════════════════════════════════════════════
    llvm::outs() << "⚡ [2-Core Parallel] Two Independent GEMMs (no sync overhead):\n";
    llvm::outs() << "   Each thread runs its own complete GEMM - perfect parallelism test\n\n";

    // Test multiple sizes to find the sweet spot
    int parallelSizes[] = {256, 512, 768, 1024};

    llvm::outs() << "╔══════════════╦═══════════════════╦═══════════════════╦════════════╗\n";
    llvm::outs() << "║   Size       ║  1-Core (GFLOPS)  ║  2-Core (GFLOPS)  ║   Speedup  ║\n";
    llvm::outs() << "╠══════════════╬═══════════════════╬═══════════════════╬════════════╣\n";

    for (int size : parallelSizes) {
        int M = size, N = size, K = size;

        // Allocate 2 sets of matrices
        float* A1 = aligned_alloc_floats(M * K);
        float* B1 = aligned_alloc_floats(K * N);
        float* C1 = aligned_alloc_floats(M * N);
        float* A2 = aligned_alloc_floats(M * K);
        float* B2 = aligned_alloc_floats(K * N);
        float* C2 = aligned_alloc_floats(M * N);

        // Initialize
        for (int i = 0; i < M * K; i++) { A1[i] = 0.5f; A2[i] = 0.5f; }
        for (int i = 0; i < K * N; i++) { B1[i] = 0.5f; B2[i] = 0.5f; }
        memset(C1, 0, M * N * sizeof(float));
        memset(C2, 0, M * N * sizeof(float));

        int iters = (size <= 512) ? 50 : 30;

        // Warmup single-threaded
        for (int i = 0; i < 10; i++) {
            gemmWithPacking(A1, B1, C1, M, N, K, K, N, N);
        }

        // Benchmark single GEMM
        auto start1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            gemmWithPacking(A1, B1, C1, M, N, K, K, N, N);
        }
        auto end1 = std::chrono::high_resolution_clock::now();
        double time1 = std::chrono::duration<double>(end1 - start1).count() / iters;
        double gflops1 = (2.0 * M * N * K / time1) / 1e9;

        // Warmup parallel
        for (int i = 0; i < 5; i++) {
            std::thread t1([&]() { gemmWithPacking(A1, B1, C1, M, N, K, K, N, N); });
            std::thread t2([&]() { gemmWithPacking(A2, B2, C2, M, N, K, K, N, N); });
            t1.join(); t2.join();
        }

        // Benchmark 2 independent GEMMs in parallel
        auto start2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            std::thread thread1([&]() { gemmWithPacking(A1, B1, C1, M, N, K, K, N, N); });
            std::thread thread2([&]() { gemmWithPacking(A2, B2, C2, M, N, K, K, N, N); });
            thread1.join();
            thread2.join();
        }
        auto end2 = std::chrono::high_resolution_clock::now();
        double time2 = std::chrono::duration<double>(end2 - start2).count() / iters;
        // Total GFLOPS = 2 GEMMs worth of work
        double gflops2 = (2.0 * 2.0 * M * N * K / time2) / 1e9;
        double speedup = gflops2 / gflops1;

        char buf[256];
        snprintf(buf, sizeof(buf),
                "║ %4dx%4dx%4d ║      %6.1f       ║      %6.1f       ║    %5.2fx   ║\n",
                M, N, K, gflops1, gflops2, speedup);
        llvm::outs() << buf;
        llvm::outs().flush();

        free(A1); free(B1); free(C1);
        free(A2); free(B2); free(C2);
    }

    llvm::outs() << "╚══════════════╩═══════════════════╩═══════════════════╩════════════╝\n";
    llvm::outs().flush();

    // ═══════════════════════════════════════════════════════════════════════
    // PART 7: Per-Core Independent GEMM (Zero-Sync Parallelism)
    // ═══════════════════════════════════════════════════════════════════════
    auto mkp = hwInfo.getOptimalMicroKernelParams();

    llvm::outs() << "\n🏗️  [Per-Core Independent] Zero-Sync Parallelism:\n";
    llvm::outs() << "   Philosophy: Each core runs its OWN complete GEMM — no barriers, no shared data\n";

    if (hwInfo.topology.isHybrid()) {
        auto pParams = hwInfo.getOptimalMicroKernelParamsForCoreType(CoreType::P_CORE);
        auto eParams = hwInfo.getOptimalMicroKernelParamsForCoreType(CoreType::E_CORE);
        auto split = HeterogeneousWorkSplit::compute(1024, hwInfo.topology);
        llvm::outs() << "   Detected: " << hwInfo.topology.numPCores << " P-cores ("
                     << hwInfo.topology.threadsPerPCore << " threads/core, "
                     << hwInfo.topology.pCoreMaxFreqKHz / 1000 << " MHz) + "
                     << hwInfo.topology.numECores << " E-cores ("
                     << hwInfo.topology.threadsPerECore << " thread/core, "
                     << hwInfo.topology.eCoreMaxFreqKHz / 1000 << " MHz)\n";
        llvm::outs() << "   Per-type blocking:\n";
        llvm::outs() << "     P-core: L1d=" << hwInfo.topology.pCoreCache.l1dKB
                     << "KB → KC=" << pParams.KC << " MC=" << pParams.MC << " NC=" << pParams.NC << "\n";
        llvm::outs() << "     E-core: L1d=" << hwInfo.topology.eCoreCache.l1dKB
                     << "KB → KC=" << eParams.KC << " MC=" << eParams.MC << " NC=" << eParams.NC << "\n";

        llvm::outs() << "   Work split (weighted by freq):\n";
        for (auto& tw : split.assignments) {
            const char* typeStr = tw.coreType == CoreType::P_CORE ? "P" : "E";
            llvm::outs() << "     Core " << tw.threadId
                         << " [" << typeStr << ", cpu" << tw.cpuId << "]: "
                         << (tw.rowEnd - tw.rowStart) << " rows\n";
        }
    } else {
        llvm::outs() << "   Homogeneous architecture\n";
    }

    llvm::outs() << "\n╔══════════════╦═══════════════╦═══════════════╦═══════════════╦═══════════════╗\n";
    llvm::outs() << "║   Size       ║  1T Hardcoded ║  N-Core HC    ║  N-Core Adapt ║  N-Core PerTy ║\n";
    llvm::outs() << "╠══════════════╬═══════════════╬═══════════════╬═══════════════╬═══════════════╣\n";

    for (const auto& test : tests) {
        double g1 = runGemmBenchmark(gemmWithPacking, test.M, test.N, test.K, 5, 30);

        float* A = aligned_alloc_floats(test.M * test.K);
        float* B = aligned_alloc_floats(test.K * test.N);
        float* C = aligned_alloc_floats(test.M * test.N);
        for (int i = 0; i < test.M * test.K; i++) A[i] = 0.5f;
        for (int i = 0; i < test.K * test.N; i++) B[i] = 0.5f;

        // N-Core Hardcoded
        for (int w = 0; w < 3; w++) {
            memset(C, 0, test.M * test.N * sizeof(float));
            gemmPerCoreIndependent(A, B, C, test.M, test.N, test.K,
                                   test.K, test.N, test.N, hwInfo.topology,
                                   KC, MC, NC);
        }
        auto s = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 30; i++)
            gemmPerCoreIndependent(A, B, C, test.M, test.N, test.K,
                                   test.K, test.N, test.N, hwInfo.topology,
                                   KC, MC, NC);
        auto e = std::chrono::high_resolution_clock::now();
        double g2 = (2.0 * test.M * test.N * test.K /
                    (std::chrono::duration<double>(e - s).count() / 30)) / 1e9;

        // N-Core Adaptive (uniform, from cpu0 cache)
        for (int w = 0; w < 3; w++) {
            memset(C, 0, test.M * test.N * sizeof(float));
            gemmPerCoreIndependent(A, B, C, test.M, test.N, test.K,
                                   test.K, test.N, test.N, hwInfo.topology,
                                   mkp.KC, mkp.MC, mkp.NC);
        }
        s = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 30; i++)
            gemmPerCoreIndependent(A, B, C, test.M, test.N, test.K,
                                   test.K, test.N, test.N, hwInfo.topology,
                                   mkp.KC, mkp.MC, mkp.NC);
        e = std::chrono::high_resolution_clock::now();
        double g3 = (2.0 * test.M * test.N * test.K /
                    (std::chrono::duration<double>(e - s).count() / 30)) / 1e9;

        // N-Core Per-Type (each core uses its own cache-tuned params)
        for (int w = 0; w < 3; w++) {
            memset(C, 0, test.M * test.N * sizeof(float));
            gemmPerCoreIndependentPerType(A, B, C, test.M, test.N, test.K,
                                          test.K, test.N, test.N, hwInfo);
        }
        s = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 30; i++)
            gemmPerCoreIndependentPerType(A, B, C, test.M, test.N, test.K,
                                          test.K, test.N, test.N, hwInfo);
        e = std::chrono::high_resolution_clock::now();
        double g4 = (2.0 * test.M * test.N * test.K /
                    (std::chrono::duration<double>(e - s).count() / 30)) / 1e9;

        char buf[256];
        snprintf(buf, sizeof(buf),
                "║ %4dx%4dx%4d ║    %6.1f     ║    %6.1f     ║    %6.1f     ║    %6.1f     ║\n",
                test.M, test.N, test.K, g1, g2, g3, g4);
        llvm::outs() << buf;
        llvm::outs().flush();

        free(A); free(B); free(C);
    }

    llvm::outs() << "╚══════════════╩═══════════════╩═══════════════╩═══════════════╩═══════════════╝\n\n";

    // ═══════════════════════════════════════════════════════════════════════
    // PART 8: Fair OpenBLAS Comparison (single-thread AND multi-thread)
    // ═══════════════════════════════════════════════════════════════════════
#ifdef TENZO_HAS_OPENBLAS
    llvm::outs() << "🔬 [OpenBLAS] Fair Comparison (1T + MT):\n\n";

    llvm::outs() << "╔══════════════╦═══════════════════╦═══════════════════╦═══════════════════╦════════════╗\n";
    llvm::outs() << "║   Size       ║  BLAS 1T (GFLOPS) ║  BLAS MT (GFLOPS) ║  Tenzo Best       ║   vs 1T    ║\n";
    llvm::outs() << "╠══════════════╬═══════════════════╬═══════════════════╬═══════════════════╬════════════╣\n";

    for (const auto& test : tests) {
        float* A = aligned_alloc_floats(test.M * test.K);
        float* B = aligned_alloc_floats(test.K * test.N);
        float* C = aligned_alloc_floats(test.M * test.N);
        for (int i = 0; i < test.M * test.K; i++) A[i] = 0.5f;
        for (int i = 0; i < test.K * test.N; i++) B[i] = 0.5f;

        // OpenBLAS single-thread
        openblas_set_num_threads(1);
        for (int w = 0; w < 5; w++) {
            memset(C, 0, test.M * test.N * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        test.M, test.N, test.K, 1.0f,
                        A, test.K, B, test.N, 0.0f, C, test.N);
        }
        auto s1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 30; i++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        test.M, test.N, test.K, 1.0f,
                        A, test.K, B, test.N, 0.0f, C, test.N);
        auto e1 = std::chrono::high_resolution_clock::now();
        double gflops1T = (2.0 * test.M * test.N * test.K /
                          (std::chrono::duration<double>(e1 - s1).count() / 30)) / 1e9;

        // OpenBLAS multi-thread
        openblas_set_num_threads(hwInfo.topology.totalLogicalCpus());
        for (int w = 0; w < 5; w++) {
            memset(C, 0, test.M * test.N * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        test.M, test.N, test.K, 1.0f,
                        A, test.K, B, test.N, 0.0f, C, test.N);
        }
        auto sM = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 30; i++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        test.M, test.N, test.K, 1.0f,
                        A, test.K, B, test.N, 0.0f, C, test.N);
        auto eM = std::chrono::high_resolution_clock::now();
        double gflopsMT = (2.0 * test.M * test.N * test.K /
                          (std::chrono::duration<double>(eM - sM).count() / 30)) / 1e9;

        // Tenzo per-core independent with adaptive blocking
        for (int w = 0; w < 3; w++) {
            memset(C, 0, test.M * test.N * sizeof(float));
            gemmPerCoreIndependent(A, B, C, test.M, test.N, test.K,
                                   test.K, test.N, test.N, hwInfo.topology,
                                   mkp.KC, mkp.MC, mkp.NC);
        }
        auto sT = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 30; i++)
            gemmPerCoreIndependent(A, B, C, test.M, test.N, test.K,
                                   test.K, test.N, test.N, hwInfo.topology,
                                   mkp.KC, mkp.MC, mkp.NC);
        auto eT = std::chrono::high_resolution_clock::now();
        double gflopsBest = (2.0 * test.M * test.N * test.K /
                            (std::chrono::duration<double>(eT - sT).count() / 30)) / 1e9;

        double vs1T = gflopsBest / gflops1T;

        char buf[256];
        snprintf(buf, sizeof(buf),
                "║ %4dx%4dx%4d ║      %6.1f       ║      %6.1f       ║      %6.1f       ║    %5.2fx   ║\n",
                test.M, test.N, test.K, gflops1T, gflopsMT, gflopsBest, vs1T);
        llvm::outs() << buf;
        llvm::outs().flush();

        free(A); free(B); free(C);
    }

    llvm::outs() << "╚══════════════╩═══════════════════╩═══════════════════╩═══════════════════╩════════════╝\n\n";
#else
    llvm::outs() << "ℹ️  OpenBLAS not available (build with -DTENZO_HAS_OPENBLAS and link -lopenblas)\n\n";
#endif

    llvm::outs() << "📊 Analysis:\n";
    llvm::outs() << "   - Per-core independent: each core runs full GEMM pipeline, zero sync\n";
    llvm::outs() << "   - Adaptive blocking: KC=" << mkp.KC << " MC=" << mkp.MC
                 << " NC=" << mkp.NC << " (from actual cache sizes)\n";
    llvm::outs() << "   - Hardcoded blocking: KC=" << KC << " MC=" << MC << " NC=" << NC << "\n";

    float peakGflops = hwInfo.getTheoreticalPeakGFLOPS();
    llvm::outs() << "   - Theoretical Peak: " << peakGflops << " GFLOPS\n\n";

    // ═══════════════════════════════════════════════════════════════════════
    // PART 9: Runtime Parameter Sweep (find best KC/MC/NC experimentally)
    // ═══════════════════════════════════════════════════════════════════════
    llvm::outs() << "🔬 [Parameter Sweep] Finding optimal KC/MC/NC for 512×512:\n\n";

    struct BlockingConfig {
        int kc, mc, nc;
        const char* label;
    };

    BlockingConfig configs[] = {
        {256,  96,  256,  "HC (256/96/256)   "},
        {256, 144,  256,  "MC+  (256/144/256) "},
        {256, 192,  512,  "NC+  (256/192/512) "},
        {128,  72,  128,  "Small(128/72/128)  "},
        {512, 192,  512,  "Large(512/192/512) "},
        {mkp.KC, mkp.MC, mkp.NC, "Adapt (computed)   "},
        {256, 192,  256,  "Wide (256/192/256) "},
        {384, 192,  384,  "Med  (384/192/384) "},
    };

    int sweepM = 512, sweepN = 512, sweepK = 512;
    double bestGflops = 0;
    int bestIdx = 0;

    llvm::outs() << "╔══════════════════════════╦═══════════════╦═══════════════╗\n";
    llvm::outs() << "║  Config                  ║  1T (GFLOPS)  ║  N-Core Total ║\n";
    llvm::outs() << "╠══════════════════════════╬═══════════════╬═══════════════╣\n";

    for (int ci = 0; ci < 8; ci++) {
        auto& cfg = configs[ci];
        float* A = aligned_alloc_floats(sweepM * sweepK);
        float* B = aligned_alloc_floats(sweepK * sweepN);
        float* C = aligned_alloc_floats(sweepM * sweepN);
        for (int i = 0; i < sweepM * sweepK; i++) A[i] = 0.5f;
        for (int i = 0; i < sweepK * sweepN; i++) B[i] = 0.5f;

        // 1T benchmark
        for (int w = 0; w < 3; w++) {
            memset(C, 0, sweepM * sweepN * sizeof(float));
            gemmAdaptive(A, B, C, sweepM, sweepN, sweepK,
                        sweepK, sweepN, sweepN, cfg.kc, cfg.mc, cfg.nc);
        }
        auto s = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; i++)
            gemmAdaptive(A, B, C, sweepM, sweepN, sweepK,
                        sweepK, sweepN, sweepN, cfg.kc, cfg.mc, cfg.nc);
        auto e = std::chrono::high_resolution_clock::now();
        double g1t = (2.0 * sweepM * sweepN * sweepK /
                     (std::chrono::duration<double>(e - s).count() / 20)) / 1e9;

        // N-Core benchmark
        for (int w = 0; w < 2; w++) {
            memset(C, 0, sweepM * sweepN * sizeof(float));
            gemmPerCoreIndependent(A, B, C, sweepM, sweepN, sweepK,
                                   sweepK, sweepN, sweepN, hwInfo.topology,
                                   cfg.kc, cfg.mc, cfg.nc);
        }
        s = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; i++)
            gemmPerCoreIndependent(A, B, C, sweepM, sweepN, sweepK,
                                   sweepK, sweepN, sweepN, hwInfo.topology,
                                   cfg.kc, cfg.mc, cfg.nc);
        e = std::chrono::high_resolution_clock::now();
        double gNc = (2.0 * sweepM * sweepN * sweepK /
                     (std::chrono::duration<double>(e - s).count() / 20)) / 1e9;

        if (gNc > bestGflops) {
            bestGflops = gNc;
            bestIdx = ci;
        }

        char buf[256];
        snprintf(buf, sizeof(buf),
                "║  %s ║    %6.1f     ║    %6.1f     ║\n",
                cfg.label, g1t, gNc);
        llvm::outs() << buf;
        llvm::outs().flush();

        free(A); free(B); free(C);
    }

    llvm::outs() << "╚══════════════════════════╩═══════════════╩═══════════════╝\n";
    llvm::outs() << "   🏆 Best: " << configs[bestIdx].label
                 << " → " << bestGflops << " GFLOPS\n\n";

    llvm::outs() << "✅ BENCHMARK COMPLETE!\n";
}

} // namespace tenzo

