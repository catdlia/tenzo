// LinearLayer.h - Neural Network Linear Layer using Tenzo GEMM
//
// Day 3 of Etap 3: Real-world inference test
//
// This implements a fully-connected (linear) layer:
//   Y = ReLU(X × W + B)
//
// Uses the optimized gemmWithPacking_BiasReLU for computation.

#pragma once

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <random>

namespace tenzo {

//===----------------------------------------------------------------------===//
// LinearLayer: Fully-connected neural network layer
//===----------------------------------------------------------------------===//
class LinearLayer {
public:
    // Layer dimensions
    int inputSize;   // Number of input features
    int outputSize;  // Number of output neurons

    // Weights and biases (owned by this layer)
    float* weights;  // [inputSize x outputSize] - row-major
    float* bias;     // [outputSize]

    // Activation
    bool useRelu;

    // Constructor
    LinearLayer(int inSize, int outSize, bool relu = true)
        : inputSize(inSize), outputSize(outSize), useRelu(relu) {
        // Allocate aligned memory
        weights = static_cast<float*>(aligned_alloc(32, inputSize * outputSize * sizeof(float)));
        bias = static_cast<float*>(aligned_alloc(32, outputSize * sizeof(float)));

        // Initialize to zero
        memset(weights, 0, inputSize * outputSize * sizeof(float));
        memset(bias, 0, outputSize * sizeof(float));
    }

    // Destructor
    ~LinearLayer() {
        if (weights) free(weights);
        if (bias) free(bias);
    }

    // Initialize with random weights (Xavier initialization)
    void initXavier(unsigned seed = 42) {
        std::mt19937 gen(seed);
        float scale = std::sqrt(2.0f / (inputSize + outputSize));
        std::normal_distribution<float> dist(0.0f, scale);

        for (int i = 0; i < inputSize * outputSize; i++) {
            weights[i] = dist(gen);
        }
        for (int i = 0; i < outputSize; i++) {
            bias[i] = 0.0f;  // Bias typically starts at 0
        }
    }

    // Initialize with specific values (for testing)
    void initConstant(float weightVal, float biasVal) {
        for (int i = 0; i < inputSize * outputSize; i++) {
            weights[i] = weightVal;
        }
        for (int i = 0; i < outputSize; i++) {
            bias[i] = biasVal;
        }
    }

    // Load weights from binary file
    bool loadWeights(const std::string& weightFile, const std::string& biasFile) {
        std::ifstream wf(weightFile, std::ios::binary);
        if (!wf) return false;
        wf.read(reinterpret_cast<char*>(weights), inputSize * outputSize * sizeof(float));
        wf.close();

        std::ifstream bf(biasFile, std::ios::binary);
        if (!bf) return false;
        bf.read(reinterpret_cast<char*>(bias), outputSize * sizeof(float));
        bf.close();

        return true;
    }

    // Save weights to binary file
    bool saveWeights(const std::string& weightFile, const std::string& biasFile) {
        std::ofstream wf(weightFile, std::ios::binary);
        if (!wf) return false;
        wf.write(reinterpret_cast<char*>(weights), inputSize * outputSize * sizeof(float));
        wf.close();

        std::ofstream bf(biasFile, std::ios::binary);
        if (!bf) return false;
        bf.write(reinterpret_cast<char*>(bias), outputSize * sizeof(float));
        bf.close();

        return true;
    }

    // Get memory usage in bytes
    size_t memoryUsage() const {
        return (inputSize * outputSize + outputSize) * sizeof(float);
    }

    // Print layer info
    void printInfo() const;

    // Forward pass (declaration - implementation in cpp file)
    void forward(const float* input, float* output, int batchSize);

    // No copy
    LinearLayer(const LinearLayer&) = delete;
    LinearLayer& operator=(const LinearLayer&) = delete;
};

//===----------------------------------------------------------------------===//
// Simple MLP (Multi-Layer Perceptron) for MNIST
//===----------------------------------------------------------------------===//
class SimpleMLP {
public:
    std::vector<LinearLayer*> layers;

    SimpleMLP() = default;

    ~SimpleMLP() {
        for (auto* layer : layers) {
            delete layer;
        }
    }

    // Add a layer
    void addLayer(int inputSize, int outputSize, bool relu = true) {
        layers.push_back(new LinearLayer(inputSize, outputSize, relu));
    }

    // Initialize all layers with Xavier
    void initXavier(unsigned seed = 42) {
        for (auto* layer : layers) {
            layer->initXavier(seed++);
        }
    }

    // Forward pass through all layers
    void forward(const float* input, float* output, int batchSize);

    // Predict class (argmax of output)
    int predict(const float* input);

    // Print network architecture
    void printArchitecture() const;

    // Get total parameter count
    size_t totalParameters() const {
        size_t total = 0;
        for (const auto* layer : layers) {
            total += layer->inputSize * layer->outputSize + layer->outputSize;
        }
        return total;
    }

    // No copy
    SimpleMLP(const SimpleMLP&) = delete;
    SimpleMLP& operator=(const SimpleMLP&) = delete;
};

// Run MNIST-like inference benchmark
void runLinearLayerBenchmark();

} // namespace tenzo

