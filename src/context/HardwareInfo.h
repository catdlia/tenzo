#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

namespace tenzo {

/// Core type classification for hybrid architectures
enum class CoreType {
    P_CORE,     // Performance core (high frequency, HT capable)
    E_CORE,     // Efficiency core (lower frequency, no HT)
    UNKNOWN     // Fallback for homogeneous or undetectable architectures
};

/// Per-logical-CPU information parsed from sysfs
struct CoreInfo {
    int cpuId = -1;             // Linux CPU id (0..N-1)
    int coreId = -1;            // Physical core id from topology/core_id
    int clusterId = -1;         // Cluster id from topology/cluster_id
    int packageId = 0;          // Physical package id
    int maxFreqKHz = 0;         // From cpufreq/cpuinfo_max_freq
    bool isHyperthread = false; // true if this is the 2nd thread on a core
    CoreType type = CoreType::UNKNOWN;
};

/// Cache hierarchy information
struct CacheInfo {
    int l1dKB = 32;     // L1 data cache per core
    int l1iKB = 32;     // L1 instruction cache per core
    int l2KB = 256;     // L2 cache per core (or per cluster)
    int l3KB = 0;       // L3 cache shared (in KB)
    int lineSize = 64;  // Cache line size in bytes
};

/// Complete CPU topology — the foundation for all optimization decisions
struct TopologyInfo {
    std::vector<CoreInfo> allCpus;   // All logical CPUs
    CacheInfo cache;                 // Default cache (from cpu0)
    CacheInfo pCoreCache;            // P-core specific cache sizes
    CacheInfo eCoreCache;            // E-core specific cache sizes

    int numPCores = 0;               // Physical P-cores (not threads)
    int numECores = 0;               // Physical E-cores
    int numPThreads = 0;             // Logical threads on P-cores (incl. HT)
    int numEThreads = 0;             // Logical threads on E-cores
    int threadsPerPCore = 1;         // HT threads per P-core (usually 2)
    int threadsPerECore = 1;         // Threads per E-core (usually 1)
    int pCoreMaxFreqKHz = 0;
    int eCoreMaxFreqKHz = 0;
    float pToERatio = 1.0f;          // Frequency ratio: pFreq / eFreq

    bool isHybrid() const { return numPCores > 0 && numECores > 0; }
    int totalPhysicalCores() const { return numPCores + numECores; }
    int totalLogicalCpus() const { return (int)allCpus.size(); }

    /// Get the set of CPU IDs that are primary threads (no hyperthreads)
    std::vector<int> getPrimaryPCoreCpuIds() const;
    std::vector<int> getECoreCpuIds() const;
    std::vector<int> getAllPrimaryCpuIds() const; // P-primaries + all E-cores
};

/// Hardware capabilities detected at runtime
struct HardwareInfo {
    std::string cpuName;
    bool hasAVX2 = false;
    bool hasAVX512 = false;
    bool hasVNNI = false;
    bool hasFMA = false;
    bool hasAMX = false;
    bool hasNEON = false;

    int physicalCores = 1;
    int logicalCores = 1;
    float baseFreqGHz = 1.0f;
    float turboFreqGHz = 1.0f;

    // Full topology (populated on Linux with sysfs)
    TopologyInfo topology;

    /// Returns recommended tile sizes based on hardware
    struct TileSizes {
        int M, N, K;
    };

    TileSizes getOptimalTileSizes() const {
        if (hasAVX512) {
            return {16, 32, 8};
        } else if (hasAVX2) {
            return {8, 16, 4};
        } else if (hasNEON) {
            return {4, 8, 4};
        } else {
            return {4, 4, 4};
        }
    }

    /// Micro-kernel parameters computed from actual hardware
    struct MicroKernelParams {
        int MR, NR;          // Micro-kernel tile (register-level)
        int unrollK;         // K-loop unroll factor
        int KC, MC, NC;      // Cache blocking sizes
    };

    /// Compute optimal micro-kernel params from detected topology
    MicroKernelParams getOptimalMicroKernelParams() const;

    /// Compute optimal params for a specific core type (uses per-type cache)
    MicroKernelParams getOptimalMicroKernelParamsForCoreType(CoreType type) const;

    /// Strategy for micro-kernel selection
    struct Strategy {
        int tileM, tileN, tileK;
        bool useAVX512;
        bool useAVX2;
        bool useAMX;
        bool useNEON;
        bool useParallel;
        int numThreads;
        int unrollFactor;
    };

    Strategy getMicroKernelStrategy() const {
        Strategy s;
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
        s.useAVX512 = hasAVX512;
        s.useAVX2 = hasAVX2 && !hasAVX512;
        s.useAMX = hasAMX;
        s.useNEON = hasNEON;
        s.useParallel = (physicalCores > 1);
        s.numThreads = physicalCores;
        return s;
    }

    float getTheoreticalPeakGFLOPS() const {
        int opsPerCycle = 2;
        if (hasAVX512 && hasFMA) {
            opsPerCycle = 32;
        } else if (hasAVX2 && hasFMA) {
            opsPerCycle = 16;
        } else if (hasNEON) {
            opsPerCycle = 8;
        }
        float freq = turboFreqGHz > 1.0f ? turboFreqGHz : baseFreqGHz;
        return physicalCores * freq * opsPerCycle;
    }

    static bool shouldUseGPU(int matrixSize) {
        return matrixSize >= 2048;
    }

    /// Detect hardware capabilities using LLVM + sysfs
    static HardwareInfo detect();

    /// Print hardware info (with topology details)
    void print() const;
};

/// Work distribution for heterogeneous cores
struct HeterogeneousWorkSplit {
    struct ThreadWork {
        int threadId;
        int cpuId;           // Linux CPU to pin to (-1 = no pinning)
        CoreType coreType;
        int rowStart, rowEnd; // Row range [rowStart, rowEnd)
    };
    std::vector<ThreadWork> assignments;

    /// Compute weighted work distribution for M rows across detected topology
    static HeterogeneousWorkSplit compute(int totalRows,
                                          const TopologyInfo& topo,
                                          bool usePCoresOnly = false);
};

} // namespace tenzo
