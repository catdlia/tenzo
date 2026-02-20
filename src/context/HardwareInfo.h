#pragma once

#include <string>

namespace tenzo {

/// Hardware capabilities detected at runtime
struct HardwareInfo {
    std::string cpuName;
    bool hasAVX2 = false;
    bool hasAVX512 = false;
    bool hasVNNI = false;
    bool hasFMA = false;
    bool hasAMX = false;      // Intel Advanced Matrix Extensions (Sapphire Rapids+)
    bool hasNEON = false;     // ARM NEON

    int physicalCores = 1;
    int logicalCores = 1;
    float baseFreqGHz = 1.0f;
    float turboFreqGHz = 1.0f;

    // Cache sizes in KB
    int l1CacheKB = 32;
    int l2CacheKB = 256;
    int l3CacheMB = 0;

    /// Returns recommended tile sizes based on hardware
    struct TileSizes {
        int M, N, K;
    };

    TileSizes getOptimalTileSizes() const {
        if (hasAVX512) {
            // AVX-512: 16 floats per register, 32 ZMM registers
            return {16, 32, 8};
        } else if (hasAVX2) {
            // AVX2: 8 floats per register, 16 YMM registers
            return {8, 16, 4};
        } else if (hasNEON) {
            // ARM NEON: 4 floats per register
            return {4, 8, 4};
        } else {
            // SSE or scalar fallback
            return {4, 4, 4};
        }
    }

    /// Strategy for micro-kernel selection
    struct Strategy {
        int tileM, tileN, tileK;
        bool useAVX512;
        bool useAVX2;
        bool useAMX;
        bool useNEON;
        bool useParallel;       // Enable multithreading
        int numThreads;         // Number of threads to use
        int unrollFactor;       // Loop unrolling factor
    };

    /// Get comprehensive micro-kernel strategy based on hardware
    Strategy getMicroKernelStrategy() const {
        Strategy s;

        // Tile sizes based on vector width
        if (hasAVX512) {
            s.tileM = 16; s.tileN = 32; s.tileK = 8;
            s.unrollFactor = 8;
        } else if (hasAVX2) {
            s.tileM = 8; s.tileN = 16; s.tileK = 4;
            s.unrollFactor = 4;
        } else if (hasNEON) {
            s.tileM = 4; s.tileN = 8; s.tileK = 4;
            s.unrollFactor = 4;
        } else {
            s.tileM = 4; s.tileN = 4; s.tileK = 4;
            s.unrollFactor = 2;
        }

        // Instruction set flags
        s.useAVX512 = hasAVX512;
        s.useAVX2 = hasAVX2 && !hasAVX512;  // Prefer AVX-512 if available
        s.useAMX = hasAMX;
        s.useNEON = hasNEON;

        // Parallelism: use physical cores to avoid hyperthreading overhead
        s.useParallel = (physicalCores > 1);
        s.numThreads = physicalCores;

        return s;
    }

    /// Calculate theoretical peak GFLOPS
    float getTheoreticalPeakGFLOPS() const {
        int opsPerCycle = 2;  // Scalar FMA

        if (hasAVX512 && hasFMA) {
            opsPerCycle = 32;  // 16 floats * 2 (FMA)
        } else if (hasAVX2 && hasFMA) {
            opsPerCycle = 16;  // 8 floats * 2 (FMA)
        } else if (hasNEON) {
            opsPerCycle = 8;   // 4 floats * 2 (FMA)
        }

        float freq = turboFreqGHz > 1.0f ? turboFreqGHz : baseFreqGHz;
        return physicalCores * freq * opsPerCycle;
    }

    /// Check if GPU dispatch is recommended for given matrix size
    static bool shouldUseGPU(int matrixSize) {
        // Threshold: GPU is better for large matrices due to PCIe overhead
        // For Intel UHD, this threshold is quite high
        return matrixSize >= 2048;
    }

    /// Detect hardware capabilities using LLVM
    static HardwareInfo detect();

    /// Print hardware info
    void print() const;
};

} // namespace tenzo

