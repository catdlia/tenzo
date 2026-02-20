#include "HardwareInfo.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include <fstream>
#include <sstream>
#include <thread>

namespace tenzo {

HardwareInfo HardwareInfo::detect() {
    HardwareInfo info;

    // Get CPU name from LLVM
    info.cpuName = llvm::sys::getHostCPUName().str();

    // Get CPU features from LLVM
    llvm::StringMap<bool> features;
    llvm::sys::getHostCPUFeatures(features);

    // Check for AVX2
    auto avx2It = features.find("avx2");
    if (avx2It != features.end()) {
        info.hasAVX2 = avx2It->second;
    }

    // Check for AVX-512F (foundation)
    auto avx512fIt = features.find("avx512f");
    if (avx512fIt != features.end()) {
        info.hasAVX512 = avx512fIt->second;
    }

    // Check for FMA (Fused Multiply-Add)
    auto fmaIt = features.find("fma");
    if (fmaIt != features.end()) {
        info.hasFMA = fmaIt->second;
    }

    // Check for AVX-VNNI (vector neural network instructions)
    auto vnniIt = features.find("avxvnni");
    if (vnniIt != features.end()) {
        info.hasVNNI = vnniIt->second;
    }
    auto avx512vnniIt = features.find("avx512vnni");
    if (avx512vnniIt != features.end() && avx512vnniIt->second) {
        info.hasVNNI = true;
    }

    // Check for AMX (Advanced Matrix Extensions - Sapphire Rapids+)
    auto amxIt = features.find("amx-tile");
    if (amxIt != features.end()) {
        info.hasAMX = amxIt->second;
    }

    // Check for ARM NEON
    auto neonIt = features.find("neon");
    if (neonIt != features.end()) {
        info.hasNEON = neonIt->second;
    }

    // Get core count from std::thread
    info.logicalCores = std::thread::hardware_concurrency();

    // Try to get more details from /proc/cpuinfo (Linux)
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        int coreCount = 0;

        while (std::getline(cpuinfo, line)) {
            // Count physical cores
            if (line.find("core id") != std::string::npos) {
                coreCount++;
            }

            // Get CPU MHz (current frequency)
            if (line.find("cpu MHz") != std::string::npos) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    float mhz = std::stof(line.substr(pos + 1));
                    info.baseFreqGHz = mhz / 1000.0f;
                }
            }
        }

        // Deduplicate core count (hyperthreading counts twice)
        if (coreCount > 0) {
            info.physicalCores = coreCount / 2;  // Rough estimate
        }
    }
#endif

    // Fallback for physical cores
    if (info.physicalCores <= 0) {
        // Estimate: logical / 2 for hyperthreaded CPUs
        info.physicalCores = std::max(1, info.logicalCores / 2);
    }

    // Try to detect Intel hybrid architecture (P-cores vs E-cores)
    // 12th gen Alder Lake / 13th gen Raptor Lake
    if (info.cpuName.find("alderlake") != std::string::npos ||
        info.cpuName.find("raptorlake") != std::string::npos) {
        // Intel hybrid architecture detection
        // i3-1215U: 2 P-cores (2 threads each) + 4 E-cores (1 thread each) = 6 cores, 8 threads
        if (info.logicalCores == 8) {
            // Likely i3-1215U or similar: 2P + 4E = 6 cores, 8 threads
            info.physicalCores = 6;
        } else if (info.logicalCores == 12) {
            // Likely i5 U-series: 2P + 8E = 10 cores, 12 threads
            info.physicalCores = 10;
        } else if (info.logicalCores == 16) {
            // Likely i5 desktop: 6P + 8E = 14 cores, or i7: 8P + 8E = 16 cores
            info.physicalCores = 12;  // Conservative estimate
        } else if (info.logicalCores == 20) {
            // i7 P-series: 6P + 8E = 14 cores, 20 threads
            info.physicalCores = 14;
        } else if (info.logicalCores == 24) {
            // i9: 8P + 16E = 24 cores, 32 threads? Or 8P + 8E = 16, 24 threads
            info.physicalCores = 16;
        }
    }

    return info;
}

void HardwareInfo::print() const {
    llvm::outs() << "\n🖥️  Hardware Info:\n";
    llvm::outs() << "   CPU: " << cpuName << "\n";
    llvm::outs() << "   Cores: " << physicalCores << " physical, "
                 << logicalCores << " logical\n";

    llvm::outs() << "\n⚡ Instruction Sets:\n";
    llvm::outs() << "   AVX2:    " << (hasAVX2 ? "✅ Yes" : "❌ No") << "\n";
    llvm::outs() << "   AVX-512: " << (hasAVX512 ? "✅ Yes" : "❌ No") << "\n";
    llvm::outs() << "   FMA:     " << (hasFMA ? "✅ Yes" : "❌ No") << "\n";
    llvm::outs() << "   VNNI:    " << (hasVNNI ? "✅ Yes" : "❌ No") << "\n";
    llvm::outs() << "   AMX:     " << (hasAMX ? "✅ Yes" : "❌ No") << "\n";

    auto tiles = getOptimalTileSizes();
    llvm::outs() << "\n📐 Optimal Configuration:\n";
    llvm::outs() << "   Tile Size: " << tiles.M << "x" << tiles.N << "x" << tiles.K << "\n";

    float rpeak = getTheoreticalPeakGFLOPS();
    llvm::outs() << "   Peak GFLOPS (theoretical): " << rpeak << "\n";

    auto strategy = getMicroKernelStrategy();
    llvm::outs() << "\n🎯 Strategy:\n";
    llvm::outs() << "   Use Parallel: " << (strategy.useParallel ? "Yes" : "No") << "\n";
    llvm::outs() << "   Threads: " << strategy.numThreads << "\n";
    llvm::outs() << "   Unroll Factor: " << strategy.unrollFactor << "\n";
}

} // namespace tenzo
