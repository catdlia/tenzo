// LinearLayer.cpp - Neural Network Linear Layer Implementation
//
// Day 3 of Etap 3: Real-world inference test
//
// Uses optimized GEMM with BiasReLU fusion for inference

#include "LinearLayer.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace tenzo {

// Forward declaration of GEMM functions from GEMMEndToEnd.cpp
// These are in anonymous namespace, so we need to redefine or expose them
// For simplicity, we'll include a minimal implementation here

namespace {

constexpr int MR = 6;
constexpr int NR = 16;
constexpr int KC = 256;
constexpr int MC = 96;
constexpr int NC = 256;

inline float* aligned_alloc_floats(size_t count) {
    return static_cast<float*>(aligned_alloc(32, count * sizeof(float)));
}

inline int roundUp(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

void packB(const float* B, float* B_packed, int K, int N, int ldb) {
    int numFullPanels = N / NR;
    int remainder = N % NR;

    for (int panelIdx = 0; panelIdx < numFullPanels; panelIdx++) {
        int jStart = panelIdx * NR;
        float* dst = B_packed + panelIdx * K * NR;

        for (int k = 0; k < K; k++) {
            const float* src = B + k * ldb + jStart;
            __m256 v0 = _mm256_loadu_ps(src);
            __m256 v1 = _mm256_loadu_ps(src + 8);
            _mm256_storeu_ps(dst + k * NR, v0);
            _mm256_storeu_ps(dst + k * NR + 8, v1);
        }
    }

    if (remainder > 0) {
        int jStart = numFullPanels * NR;
        float* dst = B_packed + numFullPanels * K * NR;

        for (int k = 0; k < K; k++) {
            const float* src = B + k * ldb + jStart;
            for (int j = 0; j < NR; j++) {
                dst[k * NR + j] = (jStart + j < N) ? src[j] : 0.0f;
            }
        }
    }
}

void packA(const float* A, float* A_packed, int M, int K, int lda) {
    int numFullPanels = M / MR;
    int remainder = M % MR;

    for (int panelIdx = 0; panelIdx < numFullPanels; panelIdx++) {
        int iStart = panelIdx * MR;
        float* dst = A_packed + panelIdx * K * MR;

        for (int k = 0; k < K; k++) {
            for (int i = 0; i < MR; i++) {
                dst[k * MR + i] = A[(iStart + i) * lda + k];
            }
        }
    }

    if (remainder > 0) {
        int iStart = numFullPanels * MR;
        float* dst = A_packed + numFullPanels * K * MR;

        for (int k = 0; k < K; k++) {
            for (int i = 0; i < MR; i++) {
                dst[k * MR + i] = (iStart + i < M) ? A[(iStart + i) * lda + k] : 0.0f;
            }
        }
    }
}

void microKernel6x16_BiasReLU(const float* A_packed, const float* B_packed,
                              float* C, int ldc, int K,
                              const float* bias, bool use_relu) {
    __m256 acc[6][2];

    for (int i = 0; i < 6; i++) {
        acc[i][0] = _mm256_loadu_ps(C + i * ldc);
        acc[i][1] = _mm256_loadu_ps(C + i * ldc + 8);
    }

    for (int k = 0; k < K; k++) {
        __m256 b0 = _mm256_loadu_ps(B_packed + k * NR);
        __m256 b1 = _mm256_loadu_ps(B_packed + k * NR + 8);

        for (int i = 0; i < 6; i++) {
            __m256 a = _mm256_broadcast_ss(A_packed + k * MR + i);
            acc[i][0] = _mm256_fmadd_ps(a, b0, acc[i][0]);
            acc[i][1] = _mm256_fmadd_ps(a, b1, acc[i][1]);
        }
    }

    if (bias != nullptr) {
        __m256 bias0 = _mm256_loadu_ps(bias);
        __m256 bias1 = _mm256_loadu_ps(bias + 8);

        for (int i = 0; i < 6; i++) {
            acc[i][0] = _mm256_add_ps(acc[i][0], bias0);
            acc[i][1] = _mm256_add_ps(acc[i][1], bias1);
        }
    }

    if (use_relu) {
        __m256 zero = _mm256_setzero_ps();
        for (int i = 0; i < 6; i++) {
            acc[i][0] = _mm256_max_ps(acc[i][0], zero);
            acc[i][1] = _mm256_max_ps(acc[i][1], zero);
        }
    }

    for (int i = 0; i < 6; i++) {
        _mm256_storeu_ps(C + i * ldc, acc[i][0]);
        _mm256_storeu_ps(C + i * ldc + 8, acc[i][1]);
    }
}

void gemmBiasReLU(const float* A, const float* B, float* C,
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

} // anonymous namespace

//===----------------------------------------------------------------------===//
// LinearLayer Implementation
//===----------------------------------------------------------------------===//

void LinearLayer::printInfo() const {
    llvm::outs() << "  LinearLayer: " << inputSize << " → " << outputSize;
    if (useRelu) llvm::outs() << " + ReLU";
    llvm::outs() << " (" << (memoryUsage() / 1024) << " KB)\n";
}

void LinearLayer::forward(const float* input, float* output, int batchSize) {
    // Y = X × W + B  where X is [batchSize × inputSize], W is [inputSize × outputSize]
    // Result Y is [batchSize × outputSize]

    // Initialize output to zero
    memset(output, 0, batchSize * outputSize * sizeof(float));

    // Use optimized GEMM with BiasReLU fusion
    gemmBiasReLU(input, weights, output,
                 batchSize, outputSize, inputSize,
                 inputSize, outputSize, outputSize,
                 bias, useRelu);
}

//===----------------------------------------------------------------------===//
// SimpleMLP Implementation
//===----------------------------------------------------------------------===//

void SimpleMLP::forward(const float* input, float* output, int batchSize) {
    if (layers.empty()) return;

    // Allocate intermediate buffers
    int maxSize = 0;
    for (const auto* layer : layers) {
        maxSize = std::max(maxSize, std::max(layer->inputSize, layer->outputSize));
    }

    float* buffer1 = aligned_alloc_floats(batchSize * maxSize);
    float* buffer2 = aligned_alloc_floats(batchSize * maxSize);

    // First layer: input → buffer1
    layers[0]->forward(input, buffer1, batchSize);

    // Middle layers: alternate buffers
    float* currentInput = buffer1;
    float* currentOutput = buffer2;

    for (size_t i = 1; i < layers.size(); i++) {
        layers[i]->forward(currentInput, currentOutput, batchSize);
        std::swap(currentInput, currentOutput);
    }

    // Copy final result to output
    int finalSize = layers.back()->outputSize;
    memcpy(output, currentInput, batchSize * finalSize * sizeof(float));

    free(buffer1);
    free(buffer2);
}

int SimpleMLP::predict(const float* input) {
    int outputSize = layers.back()->outputSize;
    float* output = aligned_alloc_floats(outputSize);

    forward(input, output, 1);

    // Argmax
    int maxIdx = 0;
    float maxVal = output[0];
    for (int i = 1; i < outputSize; i++) {
        if (output[i] > maxVal) {
            maxVal = output[i];
            maxIdx = i;
        }
    }

    free(output);
    return maxIdx;
}

void SimpleMLP::printArchitecture() const {
    llvm::outs() << "SimpleMLP Architecture:\n";
    for (size_t i = 0; i < layers.size(); i++) {
        llvm::outs() << "  [" << i << "] ";
        layers[i]->printInfo();
    }
    llvm::outs() << "  Total parameters: " << totalParameters() << "\n";
}

//===----------------------------------------------------------------------===//
// Benchmark: MNIST-like inference
//===----------------------------------------------------------------------===//

void runLinearLayerBenchmark() {
    llvm::outs() << "\n";
    llvm::outs() << "╔════════════════════════════════════════════════════════╗\n";
    llvm::outs() << "║  🧠 LINEAR LAYER INFERENCE BENCHMARK                   ║\n";
    llvm::outs() << "║  Day 3: Real-world Neural Network Test                 ║\n";
    llvm::outs() << "╚════════════════════════════════════════════════════════╝\n\n";

    // ═══════════════════════════════════════════════════════════════════════
    // Test 1: Single Layer (MNIST first layer: 784 → 128)
    // ═══════════════════════════════════════════════════════════════════════
    llvm::outs() << "📐 Test 1: Single Linear Layer (784 → 128 + ReLU)\n";
    llvm::outs() << "   This is typical for MNIST first hidden layer\n\n";

    LinearLayer layer1(784, 128, true);
    layer1.initXavier(42);

    layer1.printInfo();
    llvm::outs() << "\n";

    // Create fake MNIST input (batch of 64 images, 28x28 = 784 pixels)
    int batchSize = 64;
    float* input = aligned_alloc_floats(batchSize * 784);
    float* output = aligned_alloc_floats(batchSize * 128);

    // Initialize with random "pixel" values [0, 1]
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < batchSize * 784; i++) {
        input[i] = dist(gen);
    }

    // Warmup
    for (int i = 0; i < 10; i++) {
        layer1.forward(input, output, batchSize);
    }

    // Benchmark
    int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        layer1.forward(input, output, batchSize);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    double avgTime = elapsed / iterations;
    double samplesPerSec = batchSize / avgTime;

    // FLOPs: 2 * batchSize * inputSize * outputSize (matmul)
    //        + batchSize * outputSize (bias)
    //        + batchSize * outputSize (relu - negligible)
    double flops = 2.0 * batchSize * 784 * 128;
    double gflops = (flops / avgTime) / 1e9;

    llvm::outs() << "   Results:\n";
    llvm::outs() << "   - Batch size: " << batchSize << "\n";
    llvm::outs() << "   - Avg time per batch: " << (avgTime * 1000) << " ms\n";
    llvm::outs() << "   - Throughput: " << samplesPerSec << " samples/sec\n";
    llvm::outs() << "   - Performance: " << gflops << " GFLOPS\n\n";

    free(input);
    free(output);

    // ═══════════════════════════════════════════════════════════════════════
    // Test 2: Full MNIST MLP (784 → 256 → 128 → 10)
    // ═══════════════════════════════════════════════════════════════════════
    llvm::outs() << "📐 Test 2: Full MNIST MLP (784 → 256 → 128 → 10)\n";
    llvm::outs() << "   3-layer network for digit classification\n\n";

    SimpleMLP mlp;
    mlp.addLayer(784, 256, true);   // Hidden 1
    mlp.addLayer(256, 128, true);   // Hidden 2
    mlp.addLayer(128, 10, false);   // Output (no ReLU, use softmax later)
    mlp.initXavier(42);

    mlp.printArchitecture();
    llvm::outs() << "\n";

    // Create batch input
    batchSize = 64;
    input = aligned_alloc_floats(batchSize * 784);
    output = aligned_alloc_floats(batchSize * 10);

    for (int i = 0; i < batchSize * 784; i++) {
        input[i] = dist(gen);
    }

    // Warmup
    for (int i = 0; i < 10; i++) {
        mlp.forward(input, output, batchSize);
    }

    // Benchmark
    iterations = 100;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        mlp.forward(input, output, batchSize);
    }
    end = std::chrono::high_resolution_clock::now();

    elapsed = std::chrono::duration<double>(end - start).count();
    avgTime = elapsed / iterations;
    samplesPerSec = batchSize / avgTime;

    // Total FLOPs for 3 layers
    flops = 2.0 * batchSize * (784 * 256 + 256 * 128 + 128 * 10);
    gflops = (flops / avgTime) / 1e9;

    llvm::outs() << "   Results:\n";
    llvm::outs() << "   - Batch size: " << batchSize << "\n";
    llvm::outs() << "   - Avg time per batch: " << (avgTime * 1000) << " ms\n";
    llvm::outs() << "   - Throughput: " << samplesPerSec << " samples/sec\n";
    llvm::outs() << "   - Performance: " << gflops << " GFLOPS\n\n";

    // Test single prediction
    llvm::outs() << "   Single prediction test:\n";
    int prediction = mlp.predict(input);
    llvm::outs() << "   - Predicted class: " << prediction << "\n\n";

    free(input);
    free(output);

    // ═══════════════════════════════════════════════════════════════════════
    // Test 3: Larger batch sizes
    // ═══════════════════════════════════════════════════════════════════════
    llvm::outs() << "📐 Test 3: Scaling with batch size\n";
    llvm::outs() << "╔══════════════╦═══════════════════╦═══════════════════╗\n";
    llvm::outs() << "║  Batch Size  ║  Samples/sec      ║  GFLOPS           ║\n";
    llvm::outs() << "╠══════════════╬═══════════════════╬═══════════════════╣\n";

    int batchSizes[] = {1, 8, 32, 64, 128, 256};

    for (int bs : batchSizes) {
        input = aligned_alloc_floats(bs * 784);
        output = aligned_alloc_floats(bs * 10);

        for (int i = 0; i < bs * 784; i++) {
            input[i] = dist(gen);
        }

        // Warmup
        for (int i = 0; i < 5; i++) {
            mlp.forward(input, output, bs);
        }

        // Benchmark
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 50; i++) {
            mlp.forward(input, output, bs);
        }
        end = std::chrono::high_resolution_clock::now();

        elapsed = std::chrono::duration<double>(end - start).count();
        avgTime = elapsed / 50;
        samplesPerSec = bs / avgTime;
        flops = 2.0 * bs * (784 * 256 + 256 * 128 + 128 * 10);
        gflops = (flops / avgTime) / 1e9;

        char buf[128];
        snprintf(buf, sizeof(buf), "║     %4d     ║      %8.0f      ║      %6.2f       ║\n",
                 bs, samplesPerSec, gflops);
        llvm::outs() << buf;

        free(input);
        free(output);
    }

    llvm::outs() << "╚══════════════╩═══════════════════╩═══════════════════╝\n\n";

    llvm::outs() << "✅ LINEAR LAYER BENCHMARK COMPLETE!\n";
    llvm::outs() << "   Your Tenzo compiler can now run real neural network inference!\n\n";
}

} // namespace tenzo

