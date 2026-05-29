#include "ARMHardwareProfile.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <climits>
#include <fstream>
#include <map>
#include <set>
#include <thread>

namespace tenzo {

// ============================================================
// Helper: read a single value from a sysfs file (shared with x86)
// ============================================================
static std::string readSysfsFile(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open())
    return "";
  std::string val;
  std::getline(f, val);
  while (!val.empty() &&
         (val.back() == '\n' || val.back() == '\r' || val.back() == ' '))
    val.pop_back();
  return val;
}

static int readSysfsInt(const std::string &path, int defaultVal = -1) {
  auto s = readSysfsFile(path);
  if (s.empty())
    return defaultVal;
  try {
    return std::stoi(s);
  } catch (...) {
    return defaultVal;
  }
}

// ============================================================
// ARM Feature Detection
// ============================================================
#ifdef __aarch64__
static void detectARMFeatures(bool &neon, bool &sve, bool &sve2,
                               std::string &cpuName) {
  // On aarch64 Linux, NEON is always present (mandatory in ARMv8)
  neon = true;

  // Check /proc/cpuinfo for features
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (cpuinfo.is_open()) {
    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (line.find("Features") != std::string::npos) {
        sve = (line.find("sve") != std::string::npos);
        sve2 = (line.find("sve2") != std::string::npos);
      }
      if (line.find("CPU implementer") != std::string::npos && cpuName.empty()) {
        // Try to identify the CPU
        if (line.find("0x41") != std::string::npos)
          cpuName = "ARM Cortex";
        else if (line.find("0x51") != std::string::npos)
          cpuName = "Qualcomm Kryo";
        else if (line.find("0x61") != std::string::npos)
          cpuName = "Apple Silicon";
      }
    }
  }

  if (cpuName.empty())
    cpuName = "generic-aarch64";
}
#endif

// ============================================================
// Topology detection for ARM (big.LITTLE / DynamIQ)
// ============================================================
static TopologyInfo detectARMTopology() {
  TopologyInfo topo;

  int numCpus = (int)std::thread::hardware_concurrency();
  if (numCpus <= 0)
    numCpus = 1;

#ifdef __linux__
  // Same sysfs approach as x86 — works on ARM Linux too
  std::map<int, int> coreIdToMaxFreq;

  for (int cpu = 0; cpu < numCpus; cpu++) {
    std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);

    CoreInfo ci;
    ci.cpuId = cpu;
    ci.coreId = readSysfsInt(base + "/topology/core_id", cpu);
    ci.clusterId = readSysfsInt(base + "/topology/cluster_id", -1);
    ci.packageId = readSysfsInt(base + "/topology/physical_package_id", 0);
    ci.maxFreqKHz = readSysfsInt(base + "/cpufreq/cpuinfo_max_freq", 0);
    ci.isHyperthread = false; // ARM doesn't have SMT (usually)
    ci.type = CoreType::UNKNOWN;

    coreIdToMaxFreq[ci.coreId] =
        std::max(coreIdToMaxFreq[ci.coreId], ci.maxFreqKHz);
    topo.allCpus.push_back(ci);
  }

  // Detect big.LITTLE: same strategy as x86 P/E classification
  int globalMaxFreq = 0, globalMinFreq = INT_MAX;
  for (auto &[coreId, freq] : coreIdToMaxFreq) {
    if (freq > 0) {
      globalMaxFreq = std::max(globalMaxFreq, freq);
      globalMinFreq = std::min(globalMinFreq, freq);
    }
  }

  bool isBigLittle = (globalMaxFreq > 0 && globalMinFreq > 0 &&
                       globalMaxFreq > globalMinFreq * 1.1);

  if (isBigLittle) {
    int pThreshold = (int)(globalMaxFreq * 0.9);
    std::set<int> bigCoreIds;
    for (auto &[coreId, freq] : coreIdToMaxFreq) {
      if (freq >= pThreshold)
        bigCoreIds.insert(coreId);
    }

    for (auto &ci : topo.allCpus) {
      ci.type = bigCoreIds.count(ci.coreId) ? CoreType::P_CORE
                                              : CoreType::E_CORE;
    }
  }

  // Count cores
  std::set<int> uniqueP, uniqueE;
  for (auto &ci : topo.allCpus) {
    if (ci.type == CoreType::P_CORE) {
      uniqueP.insert(ci.coreId);
      topo.numPThreads++;
    } else if (ci.type == CoreType::E_CORE) {
      uniqueE.insert(ci.coreId);
      topo.numEThreads++;
    }
  }
  topo.numPCores = (int)uniqueP.size();
  topo.numECores = (int)uniqueE.size();

  if (isBigLittle) {
    topo.pCoreMaxFreqKHz = globalMaxFreq;
    topo.eCoreMaxFreqKHz = globalMinFreq;
    topo.pToERatio =
        (float)topo.pCoreMaxFreqKHz / std::max(1, topo.eCoreMaxFreqKHz);
  }

  // Cache detection (same sysfs as x86)
  auto readCacheForCpu = [](int cpuNum) -> CacheInfo {
    CacheInfo c;
    for (int idx = 0; idx < 4; idx++) {
      std::string cbase = "/sys/devices/system/cpu/cpu" +
                          std::to_string(cpuNum) + "/cache/index" +
                          std::to_string(idx);
      int level = readSysfsInt(cbase + "/level", -1);
      if (level < 0)
        continue;
      std::string type = readSysfsFile(cbase + "/type");
      int sizeKB = readSysfsInt(cbase + "/size", 0);
      int lineSize = readSysfsInt(cbase + "/coherency_line_size", 64);

      if (level == 1 && type == "Data")
        c.l1dKB = sizeKB;
      else if (level == 1 && type == "Instruction")
        c.l1iKB = sizeKB;
      else if (level == 2)
        c.l2KB = sizeKB;
      else if (level == 3)
        c.l3KB = sizeKB;
      c.lineSize = lineSize;
    }
    return c;
  };

  topo.cache = readCacheForCpu(0);
  topo.pCoreCache = topo.cache;
  topo.eCoreCache = topo.cache;

  if (topo.numPCores > 0)
    topo.threadsPerPCore = 1; // ARM typically no SMT
  if (topo.numECores > 0)
    topo.threadsPerECore = 1;
#else
  // Non-Linux fallback: single homogeneous core
  CoreInfo ci;
  ci.cpuId = 0;
  ci.coreId = 0;
  topo.allCpus.push_back(ci);
  // Default cache for generic ARM
  topo.cache = {32, 32, 256, 0, 64};
  topo.pCoreCache = topo.cache;
  topo.eCoreCache = topo.cache;
#endif

  return topo;
}

// ============================================================
// Constructor
// ============================================================
ARMHardwareProfile::ARMHardwareProfile() {
#ifdef __aarch64__
  detectARMFeatures(hasNEON_flag, hasSVE_flag, hasSVE2_flag, cpuName);
  topology = detectARMTopology();

  logicalCores = (int)std::thread::hardware_concurrency();
  std::set<int> uniqueCores;
  for (auto &ci : topology.allCpus) {
    if (!ci.isHyperthread)
      uniqueCores.insert(ci.coreId);
  }
  physicalCores =
      uniqueCores.empty() ? logicalCores : (int)uniqueCores.size();

  // Frequency from topology
  int maxFreq = 0;
  for (auto &ci : topology.allCpus)
    maxFreq = std::max(maxFreq, ci.maxFreqKHz);
  if (maxFreq > 0)
    turboFreqGHz = maxFreq / 1000000.0f;
#else
  // Cross-compilation stub: provide ARM-like defaults
  cpuName = "arm-cross-compile-stub";
  hasNEON_flag = true;
  hasSVE_flag = false;
  hasSVE2_flag = false;
  physicalCores = 4;
  logicalCores = 4;
  baseFreqGHz = 1.5f;
  turboFreqGHz = 2.0f;

  // Reasonable defaults for a Cortex-A76-like core
  topology.cache = {64, 64, 256, 0, 64};
  topology.pCoreCache = topology.cache;
  topology.eCoreCache = topology.cache;
#endif
}

// ============================================================
// Micro-kernel params for ARM NEON
// ============================================================
MicroKernelParams ARMHardwareProfile::getOptimalMicroKernelParams() const {
  return getOptimalMicroKernelParamsForCoreType(CoreType::UNKNOWN);
}

MicroKernelParams
ARMHardwareProfile::getOptimalMicroKernelParamsForCoreType(CoreType type) const {
  MicroKernelParams p;

  // NEON: 128-bit SIMD → 4 floats per vector
  // SVE would be variable-length, but we start with NEON fixed-width
  if (hasSVE_flag) {
    // SVE: assume 256-bit for now (Neoverse V1)
    p.MR = 8;
    p.NR = 16;
    p.unrollK = 4;
    p.VEC_SIZE = 8; // 256-bit / 32-bit
  } else {
    // NEON: 128-bit
    p.MR = 8;
    p.NR = 12;
    p.unrollK = 4;
    p.VEC_SIZE = 4; // 128-bit / 32-bit
  }

  const CacheInfo &c = (type == CoreType::P_CORE)   ? topology.pCoreCache
                       : (type == CoreType::E_CORE) ? topology.eCoreCache
                                                     : topology.cache;

  int l1dBytes = c.l1dKB * 1024;
  int l2Bytes = c.l2KB * 1024;

  // KC: B panel [KC × NR] in ~60% of L1d
  if (l1dBytes > 0 && p.NR > 0) {
    p.KC = (int)(0.6 * l1dBytes) / (p.NR * (int)sizeof(float));
    p.KC = (p.KC / p.unrollK) * p.unrollK;
    p.KC = std::max(p.KC, p.unrollK);
  } else {
    p.KC = 256;
  }

  // MC: A block [MC × KC] in ~80% of L2
  if (l2Bytes > 0) {
    p.MC = (int)(0.8 * l2Bytes) / (p.KC * (int)sizeof(float));
    p.MC = (p.MC / p.MR) * p.MR;
    p.MC = std::max(p.MC, p.MR);
  } else {
    p.MC = 96;
  }

  // NC: reasonable default (ARM usually no shared L3)
  p.NC = 256;

  return p;
}

TileSizes ARMHardwareProfile::getOptimalTileSizes() const {
  if (hasSVE_flag)
    return {8, 16, 4};
  return {4, 8, 4}; // NEON
}

Strategy ARMHardwareProfile::getMicroKernelStrategy() const {
  Strategy s;
  if (hasSVE_flag) {
    s.tileM = 8;
    s.tileN = 16;
    s.tileK = 4;
    s.unrollFactor = 4;
  } else {
    s.tileM = 4;
    s.tileN = 8;
    s.tileK = 4;
    s.unrollFactor = 4;
  }
  s.useAVX512 = false;
  s.useAVX2 = false;
  s.useAMX = false;
  s.useNEON = hasNEON_flag;
  s.useParallel = (physicalCores > 1);
  s.numThreads = physicalCores;
  return s;
}

float ARMHardwareProfile::getTheoreticalPeakGFLOPS() const {
  // NEON: 4 floats × 2 FMA ports (on Cortex-A76+) = 8 FLOPS/cycle
  // SVE 256-bit: 8 floats × 2 = 16 FLOPS/cycle
  int opsPerCycle = hasSVE_flag ? 16 : 8;
  float freq = turboFreqGHz > 1.0f ? turboFreqGHz : baseFreqGHz;
  return physicalCores * freq * opsPerCycle;
}

bool ARMHardwareProfile::shouldUseGPU(int matrixSize) const {
  // ARM GPUs (Mali, Adreno) are weaker than desktop GPUs
  return matrixSize >= 4096;
}

void ARMHardwareProfile::print() const {
  llvm::outs() << "\n🖥️  Hardware Info (ARM):\n";
  llvm::outs() << "   CPU: " << cpuName << "\n";
  llvm::outs() << "   Cores: " << physicalCores << " physical, " << logicalCores
               << " logical\n";

  if (topology.isHybrid()) {
    llvm::outs() << "\n🏗️  big.LITTLE Architecture:\n";
    llvm::outs() << "   big cores: " << topology.numPCores << " ("
                 << topology.pCoreMaxFreqKHz / 1000 << " MHz)\n";
    llvm::outs() << "   LITTLE cores: " << topology.numECores << " ("
                 << topology.eCoreMaxFreqKHz / 1000 << " MHz)\n";
  }

  llvm::outs() << "\n💾 Cache: L1d=" << topology.cache.l1dKB
               << "KB  L2=" << topology.cache.l2KB << "KB\n";

  llvm::outs() << "\n⚡ Instruction Sets:\n";
  llvm::outs() << "   NEON:  " << (hasNEON_flag ? "✅ Yes" : "❌ No") << "\n";
  llvm::outs() << "   SVE:   " << (hasSVE_flag ? "✅ Yes" : "❌ No") << "\n";
  llvm::outs() << "   SVE2:  " << (hasSVE2_flag ? "✅ Yes" : "❌ No") << "\n";

  auto mkp = getOptimalMicroKernelParams();
  llvm::outs() << "\n📐 Optimal Micro-Kernel:\n";
  llvm::outs() << "   MR×NR: " << mkp.MR << "×" << mkp.NR << "\n";
  llvm::outs() << "   VEC_SIZE: " << mkp.VEC_SIZE << "\n";
  llvm::outs() << "   KC=" << mkp.KC << ", MC=" << mkp.MC << ", NC=" << mkp.NC
               << "\n";

  llvm::outs() << "   Peak GFLOPS (theoretical): " << getTheoreticalPeakGFLOPS()
               << "\n";
}

} // namespace tenzo
