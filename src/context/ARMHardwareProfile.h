#pragma once
#include "HardwareProfile.h"

namespace tenzo {

/// Stub ARM hardware profile for future cross-compilation support.
/// Detects NEON/SVE features and big.LITTLE topology on aarch64 Linux.
/// On non-ARM hosts, returns sensible NEON defaults.
class ARMHardwareProfile : public HardwareProfile {
private:
    std::string cpuName;
    bool hasNEON_flag = true;  // Assume NEON on any ARMv8+
    bool hasSVE_flag = false;
    bool hasSVE2_flag = false;

    int physicalCores = 1;
    int logicalCores = 1;
    float baseFreqGHz = 1.5f;
    float turboFreqGHz = 2.0f;
    TopologyInfo topology;

public:
    ARMHardwareProfile();

    std::string getCpuName() const override { return cpuName; }
    bool hasAVX2() const override { return false; }
    bool hasAVX512() const override { return false; }
    bool hasVNNI() const override { return false; }
    bool hasFMA() const override { return false; }
    bool hasAMX() const override { return false; }
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

    // ARM-specific
    bool hasSVE() const { return hasSVE_flag; }
    bool hasSVE2() const { return hasSVE2_flag; }
};

} // namespace tenzo
