#pragma once
#include "HardwareProfile.h"
namespace tenzo {

class X86HardwareProfile : public HardwareProfile {
private:
    std::string cpuName;
    bool hasAVX2_flag = false;
    bool hasAVX512_flag = false;
    bool hasVNNI_flag = false;
    bool hasFMA_flag = false;
    bool hasAMX_flag = false;
    bool hasNEON_flag = false;

    int physicalCores = 1;
    int logicalCores = 1;
    float baseFreqGHz = 1.0f;
    float turboFreqGHz = 1.0f;
    TopologyInfo topology;

public:
    X86HardwareProfile();
    
    std::string getCpuName() const override { return cpuName; }
    bool hasAVX2() const override { return hasAVX2_flag; }
    bool hasAVX512() const override { return hasAVX512_flag; }
    bool hasVNNI() const override { return hasVNNI_flag; }
    bool hasFMA() const override { return hasFMA_flag; }
    bool hasAMX() const override { return hasAMX_flag; }
    bool hasNEON() const override { return hasNEON_flag; }
    int getPhysicalCores() const override { return physicalCores; }
    int getLogicalCores() const override { return logicalCores; }
    float getBaseFreqGHz() const override { return baseFreqGHz; }
    float getTurboFreqGHz() const override { return turboFreqGHz; }
    const TopologyInfo& getTopology() const override { return topology; }
    
    TileSizes getOptimalTileSizes() const override;
    MicroKernelParams getOptimalMicroKernelParams() const override;
    MicroKernelParams getOptimalMicroKernelParamsForCoreType(CoreType type) const override;
    Strategy getMicroKernelStrategy() const override;
    float getTheoreticalPeakGFLOPS() const override;
    bool shouldUseGPU(int matrixSize) const override;
    void print() const override;
};

} // namespace tenzo
