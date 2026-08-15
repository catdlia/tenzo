#pragma once
#include <string>
#include <vector>
#include <memory>

namespace tenzo {

enum class CoreType {
    P_CORE,
    E_CORE,
    UNKNOWN
};

struct CoreInfo {
    int cpuId = -1;
    int coreId = -1;
    int clusterId = -1;
    int packageId = 0;
    int maxFreqKHz = 0;
    bool isHyperthread = false;
    CoreType type = CoreType::UNKNOWN;
};

struct CacheInfo {
    int l1dKB = 32;
    int l1iKB = 32;
    int l2KB = 256;
    int l3KB = 0;
    int lineSize = 64;
};

struct TopologyInfo {
    std::vector<CoreInfo> allCpus;
    CacheInfo cache;
    CacheInfo pCoreCache;
    CacheInfo eCoreCache;

    int numPCores = 0;
    int numECores = 0;
    int numPThreads = 0;
    int numEThreads = 0;
    int threadsPerPCore = 1;
    int threadsPerECore = 1;
    int pCoreMaxFreqKHz = 0;
    int eCoreMaxFreqKHz = 0;
    float pToERatio = 1.0f;

    bool isHybrid() const { return numPCores > 0 && numECores > 0; }
    int totalPhysicalCores() const { return numPCores + numECores; }
    int totalLogicalCpus() const { return (int)allCpus.size(); }

    std::vector<int> getPrimaryPCoreCpuIds() const;
    std::vector<int> getECoreCpuIds() const;
    std::vector<int> getAllPrimaryCpuIds() const;
};

struct HeterogeneousWorkSplit {
    struct ThreadWork {
        int threadId;
        int cpuId;
        CoreType coreType;
        int rowStart, rowEnd;
    };
    std::vector<ThreadWork> assignments;

    static HeterogeneousWorkSplit compute(int totalRows, const TopologyInfo& topo, bool usePCoresOnly = false);
};

struct MicroKernelParams {
    int MR, NR;
    int unrollK;
    int KC, MC, NC;
    int VEC_SIZE = 8;
};

struct TileSizes { int M, N, K; };
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

class HardwareProfile {
public:
    virtual ~HardwareProfile() = default;

    virtual std::string getCpuName() const = 0;
    virtual bool hasAVX2() const = 0;
    virtual bool hasAVX512() const = 0;
    virtual bool hasVNNI() const = 0;
    virtual bool hasFMA() const = 0;
    virtual bool hasAMX() const = 0;
    virtual bool hasNEON() const = 0;
    virtual int getPhysicalCores() const = 0;
    virtual int getLogicalCores() const = 0;
    virtual float getBaseFreqGHz() const = 0;
    virtual float getTurboFreqGHz() const = 0;
    virtual const TopologyInfo& getTopology() const = 0;

    virtual TileSizes getOptimalTileSizes() const = 0;
    virtual MicroKernelParams getOptimalMicroKernelParams() const = 0;
    virtual MicroKernelParams getOptimalMicroKernelParamsForCoreType(CoreType type) const = 0;
    virtual Strategy getMicroKernelStrategy() const = 0;
    virtual float getTheoreticalPeakGFLOPS() const = 0;
    virtual bool shouldUseGPU(int matrixSize) const = 0;
    virtual void print() const = 0;

    static std::shared_ptr<HardwareProfile> detect();
};

} // namespace tenzo
