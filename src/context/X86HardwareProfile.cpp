#include "X86HardwareProfile.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <thread>

namespace tenzo {

// ============================================================
// Helper: read a single value from a sysfs file
// ============================================================
static std::string readSysfsFile(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open())
    return "";
  std::string val;
  std::getline(f, val);
  // Trim whitespace
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

/// Parse "48K" or "1280K" or "10240K" into KB
static int parseCacheSizeKB(const std::string &s) {
  if (s.empty())
    return 0;
  std::string num;
  for (char c : s) {
    if (std::isdigit(c))
      num += c;
  }
  if (num.empty())
    return 0;
  int val = std::stoi(num);
  // If ends with 'M', multiply
  if (s.back() == 'M' || s.find('M') != std::string::npos)
    val *= 1024;
  return val;
}

/// Parse thread_siblings_list like "0-1" or "4" to get list of cpu ids
static std::vector<int> parseCpuList(const std::string &s) {
  std::vector<int> result;
  if (s.empty())
    return result;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    auto dash = token.find('-');
    if (dash != std::string::npos) {
      int start = std::stoi(token.substr(0, dash));
      int end = std::stoi(token.substr(dash + 1));
      for (int i = start; i <= end; i++)
        result.push_back(i);
    } else {
      result.push_back(std::stoi(token));
    }
  }
  return result;
}

// ============================================================
// Topology Detection via Linux sysfs
// ============================================================
static TopologyInfo detectTopology() {
  TopologyInfo topo;

  int numCpus = (int)std::thread::hardware_concurrency();
  if (numCpus <= 0)
    numCpus = 1;

  // ---- Phase 1: Gather per-CPU info ----
  std::set<int> seenCoreIds;
  std::map<int, int> coreIdToMaxFreq; // core_id -> max freq
  std::map<int, std::vector<int>>
      coreIdToCpus; // core_id -> list of cpus sharing it

  for (int cpu = 0; cpu < numCpus; cpu++) {
    std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);

    CoreInfo ci;
    ci.cpuId = cpu;
    ci.coreId = readSysfsInt(base + "/topology/core_id", cpu);
    ci.clusterId = readSysfsInt(base + "/topology/cluster_id", -1);
    ci.packageId = readSysfsInt(base + "/topology/physical_package_id", 0);
    ci.maxFreqKHz = readSysfsInt(base + "/cpufreq/cpuinfo_max_freq", 0);

    // Parse thread siblings to detect hyperthreading
    auto siblings =
        parseCpuList(readSysfsFile(base + "/topology/thread_siblings_list"));
    ci.isHyperthread = (siblings.size() > 1 && cpu != siblings[0]);

    coreIdToCpus[ci.coreId].push_back(cpu);
    if (ci.maxFreqKHz > 0) {
      coreIdToMaxFreq[ci.coreId] =
          std::max(coreIdToMaxFreq[ci.coreId], ci.maxFreqKHz);
    }

    topo.allCpus.push_back(ci);
  }

  // ---- Phase 2: Classify P-core vs E-core ----
  // Strategy: group by max frequency. The highest frequency group = P-cores.
  // If all cores same frequency → all UNKNOWN (homogeneous).
  int globalMaxFreq = 0;
  int globalMinFreq = INT_MAX;
  for (auto &[coreId, freq] : coreIdToMaxFreq) {
    globalMaxFreq = std::max(globalMaxFreq, freq);
    if (freq > 0)
      globalMinFreq = std::min(globalMinFreq, freq);
  }

  bool isHybrid = (globalMaxFreq > 0 && globalMinFreq > 0 &&
                   globalMaxFreq > globalMinFreq * 1.1); // >10% difference

  std::set<int> pCoreCoreIds, eCoreCoreIds;

  if (isHybrid) {
    // Threshold: any core within 10% of max is P-core
    int pThreshold = (int)(globalMaxFreq * 0.9);
    for (auto &[coreId, freq] : coreIdToMaxFreq) {
      if (freq >= pThreshold) {
        pCoreCoreIds.insert(coreId);
      } else {
        eCoreCoreIds.insert(coreId);
      }
    }
  }

  // Apply classification back to all CPUs
  for (auto &ci : topo.allCpus) {
    if (isHybrid) {
      ci.type =
          pCoreCoreIds.count(ci.coreId) ? CoreType::P_CORE : CoreType::E_CORE;
    } else {
      ci.type = CoreType::UNKNOWN; // homogeneous
    }
  }

  // ---- Phase 3: Count unique physical cores ----
  std::set<int> uniquePCores, uniqueECores;
  for (auto &ci : topo.allCpus) {
    if (ci.isHyperthread)
      continue; // Don't double-count HT
    if (ci.type == CoreType::P_CORE)
      uniquePCores.insert(ci.coreId);
    else if (ci.type == CoreType::E_CORE)
      uniqueECores.insert(ci.coreId);
  }

  topo.numPCores = (int)uniquePCores.size();
  topo.numECores = (int)uniqueECores.size();

  // Count threads
  for (auto &ci : topo.allCpus) {
    if (ci.type == CoreType::P_CORE)
      topo.numPThreads++;
    else if (ci.type == CoreType::E_CORE)
      topo.numEThreads++;
  }

  topo.pCoreMaxFreqKHz = isHybrid ? globalMaxFreq : 0;
  topo.eCoreMaxFreqKHz = isHybrid ? globalMinFreq : 0;
  topo.pToERatio = (topo.eCoreMaxFreqKHz > 0)
                       ? (float)topo.pCoreMaxFreqKHz / topo.eCoreMaxFreqKHz
                       : 1.0f;

  // ---- Phase 4: Detect cache hierarchy per core type ----
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
      std::string sizeStr = readSysfsFile(cbase + "/size");
      int sizeKB = parseCacheSizeKB(sizeStr);
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

  // Read default cache from cpu0
  topo.cache = readCacheForCpu(0);

  if (isHybrid) {
    // Find a representative P-core and E-core CPU
    int pCoreCpu = -1, eCoreCpu = -1;
    for (auto &ci : topo.allCpus) {
      if (ci.type == CoreType::P_CORE && !ci.isHyperthread && pCoreCpu < 0)
        pCoreCpu = ci.cpuId;
      if (ci.type == CoreType::E_CORE && eCoreCpu < 0)
        eCoreCpu = ci.cpuId;
    }
    if (pCoreCpu >= 0)
      topo.pCoreCache = readCacheForCpu(pCoreCpu);
    else
      topo.pCoreCache = topo.cache;
    if (eCoreCpu >= 0)
      topo.eCoreCache = readCacheForCpu(eCoreCpu);
    else
      topo.eCoreCache = topo.cache;
  } else {
    topo.pCoreCache = topo.cache;
    topo.eCoreCache = topo.cache;
  }

  // ---- Phase 5: Compute threads per core ----
  if (topo.numPCores > 0)
    topo.threadsPerPCore = topo.numPThreads / topo.numPCores;
  if (topo.numECores > 0)
    topo.threadsPerECore = topo.numEThreads / topo.numECores;

  return topo;
}

// ============================================================
// TopologyInfo helpers
// ============================================================
std::vector<int> TopologyInfo::getPrimaryPCoreCpuIds() const {
  std::vector<int> result;
  for (auto &ci : allCpus) {
    if (ci.type == CoreType::P_CORE && !ci.isHyperthread)
      result.push_back(ci.cpuId);
  }
  return result;
}

std::vector<int> TopologyInfo::getECoreCpuIds() const {
  std::vector<int> result;
  for (auto &ci : allCpus) {
    if (ci.type == CoreType::E_CORE && !ci.isHyperthread)
      result.push_back(ci.cpuId);
  }
  return result;
}

std::vector<int> TopologyInfo::getAllPrimaryCpuIds() const {
  auto pCores = getPrimaryPCoreCpuIds();
  auto eCores = getECoreCpuIds();
  pCores.insert(pCores.end(), eCores.begin(), eCores.end());

  // For homogeneous architectures (all UNKNOWN type), include all
  // non-hyperthread cores so we don't fall back to single-thread
  if (pCores.empty()) {
    for (auto &ci : allCpus) {
      if (!ci.isHyperthread)
        pCores.push_back(ci.cpuId);
    }
  }

  return pCores;
}

// ============================================================
// MicroKernel Parameter Calculator
// ============================================================
MicroKernelParams
X86HardwareProfile::getOptimalMicroKernelParams() const {
  return getOptimalMicroKernelParamsForCoreType(CoreType::UNKNOWN);
}

MicroKernelParams
X86HardwareProfile::getOptimalMicroKernelParamsForCoreType(
    CoreType type) const {
  MicroKernelParams p;

  // ---- MR, NR from ISA ----
  if (hasAVX512_flag && hasFMA_flag) {
    p.MR = 14;
    p.NR = 32;
    p.unrollK = 8;
  } else if (hasAVX2_flag && hasFMA_flag) {
    // MR=4, NR=8 to free up YMM registers for bitwise operations and avoid register spilling
    p.MR = 4;
    p.NR = 8;
    p.unrollK = 4;
  } else if (hasNEON_flag) {
    p.MR = 8;
    p.NR = 12;
    p.unrollK = 4;
  } else {
    p.MR = 4;
    p.NR = 4;
    p.unrollK = 2;
  }

  // ---- Select cache for this core type ----
  const CacheInfo &c = (type == CoreType::P_CORE)   ? topology.pCoreCache
                       : (type == CoreType::E_CORE) ? topology.eCoreCache
                                                    : topology.cache;

  int l1dBytes = c.l1dKB * 1024;
  int l2Bytes = c.l2KB * 1024;
  int l3Bytes = c.l3KB > 0 ? c.l3KB * 1024 : topology.cache.l3KB * 1024;

  // KC: B panel [KC × NR] should fit in ~60% of L1d
  if (l1dBytes > 0 && p.NR > 0) {
    p.KC = (int)(0.6 * l1dBytes) / (p.NR * (int)sizeof(float));
    p.KC = (p.KC / p.unrollK) * p.unrollK;
    p.KC = std::max(p.KC, p.unrollK);
  } else {
    p.KC = 256;
  }

  // MC: A block [MC × KC] should fit in ~80% of L2
  if (l2Bytes > 0) {
    p.MC = (int)(0.8 * l2Bytes) / (p.KC * (int)sizeof(float));
    p.MC = (p.MC / p.MR) * p.MR;
    p.MC = std::max(p.MC, p.MR);
  } else {
    p.MC = 96;
  }

  // NC: B panel [KC × NC] should fit in L3 / numActiveCores
  int numCores = topology.totalPhysicalCores();
  if (numCores <= 0)
    numCores = physicalCores > 0 ? physicalCores : 1;
  if (l3Bytes > 0 && numCores > 0) {
    int l3PerCore = l3Bytes / numCores;
    p.NC = (int)(0.5 * l3PerCore) / (p.KC * (int)sizeof(float));
    p.NC = (p.NC / p.NR) * p.NR;
    p.NC = std::max(p.NC, p.NR);
    p.NC = std::min(p.NC, 4096);
  } else {
    p.NC = 256;
  }

  return p;
}

// ============================================================
// Heterogeneous Work Distribution
// ============================================================
HeterogeneousWorkSplit HeterogeneousWorkSplit::compute(int totalRows,
                                                       const TopologyInfo &topo,
                                                       bool usePCoresOnly) {

  HeterogeneousWorkSplit split;

  // We need MR for alignment to avoid partial tiles in the middle of the work
  // split Default to 6 if we can't detect it easily here, but ideally this
  // should be a parameter
  int MR = 6;

  if (!topo.isHybrid() || usePCoresOnly) {
    auto cpuIds = topo.isHybrid() ? topo.getPrimaryPCoreCpuIds()
                                  : topo.getAllPrimaryCpuIds();
    if (cpuIds.empty()) {
      split.assignments.push_back({0, -1, CoreType::UNKNOWN, 0, totalRows});
      return split;
    }

    int numThreads = (int)cpuIds.size();
    int rowsPerThread = (totalRows / (numThreads * MR)) * MR;
    int offset = 0;

    for (int i = 0; i < numThreads; i++) {
      int rows;
      if (i == numThreads - 1) {
        rows = totalRows - offset;
      } else {
        rows = rowsPerThread;
      }
      if (rows <= 0 && i > 0)
        break;
      split.assignments.push_back(
          {i, cpuIds[i], CoreType::UNKNOWN, offset, offset + rows});
      offset += rows;
    }
    return split;
  }

  // ---- Hybrid architecture: weighted distribution ----
  auto pCpus = topo.getPrimaryPCoreCpuIds();
  auto eCpus = topo.getECoreCpuIds();

  // Weighted capacity: P-cores are often 2x-3x faster than E-cores
  // because of frequency + wider OOO + bigger caches.
  // We'll use (pFreq * 2) for P-cores vs (eFreq * 1) for E-cores as a
  // heuristic.
  float pWeight = (float)topo.pCoreMaxFreqKHz * 2.5f;
  float eWeight = (float)topo.eCoreMaxFreqKHz * 1.0f;

  double totalCapacity =
      (double)pCpus.size() * pWeight + (double)eCpus.size() * eWeight;

  if (totalCapacity <= 0) {
    split.assignments.push_back({0, -1, CoreType::UNKNOWN, 0, totalRows});
    return split;
  }

  int offset = 0;
  int threadId = 0;

  // Assign rows to P-cores
  for (int i = 0; i < (int)pCpus.size(); i++) {
    double share = (double)pWeight / totalCapacity;
    int rows = (int)(totalRows * share);
    // Align to MR
    rows = (rows / MR) * MR;
    if (rows < MR)
      rows = MR;

    if (offset + rows > totalRows)
      rows = totalRows - offset;
    if (rows <= 0)
      break;

    split.assignments.push_back(
        {threadId++, pCpus[i], CoreType::P_CORE, offset, offset + rows});
    offset += rows;
  }

  // Assign rows to E-cores
  for (int i = 0; i < (int)eCpus.size(); i++) {
    int rows;
    if (i == (int)eCpus.size() - 1) {
      rows = totalRows - offset;
    } else {
      double share = (double)eWeight / totalCapacity;
      rows = (int)(totalRows * share);
      rows = (rows / MR) * MR;
    }

    if (rows < 0)
      rows = 0;
    if (offset + rows > totalRows)
      rows = totalRows - offset;

    split.assignments.push_back(
        {threadId++, eCpus[i], CoreType::E_CORE, offset, offset + rows});
    offset += rows;
  }

  return split;
}

// ============================================================
// Main detection entry point
// ============================================================
X86HardwareProfile::X86HardwareProfile() {

  // Get CPU name from LLVM
  cpuName = llvm::sys::getHostCPUName().str();

  // Get CPU features from LLVM
  auto features = llvm::sys::getHostCPUFeatures();
  hasAVX2_flag = features.lookup("avx2");
  hasAVX512_flag = features.lookup("avx512f");
  hasFMA_flag = features.lookup("fma");
  hasVNNI_flag = features.lookup("avxvnni") || features.lookup("avx512vnni");
  hasAMX_flag = features.lookup("amx-tile");
  hasNEON_flag = features.lookup("neon");

  logicalCores = std::thread::hardware_concurrency();

  // Detect full topology via sysfs
#ifdef __linux__
  topology = detectTopology();

  // Populate legacy fields from topology
  if (topology.isHybrid()) {
    physicalCores = topology.totalPhysicalCores();
  } else {
    // Homogeneous: count unique core_ids
    std::set<int> uniqueCores;
    for (auto &ci : topology.allCpus) {
      if (!ci.isHyperthread)
        uniqueCores.insert(ci.coreId);
    }
    physicalCores = uniqueCores.empty() ? std::max(1, logicalCores / 2)
                                        : (int)uniqueCores.size();
  }

  // Use max freq from topology
  int maxFreq = 0;
  for (auto &ci : topology.allCpus) {
    maxFreq = std::max(maxFreq, ci.maxFreqKHz);
  }
  if (maxFreq > 0) {
    turboFreqGHz = maxFreq / 1000000.0f;
  }

  // Get base freq from /proc/cpuinfo (current frequency)
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (cpuinfo.is_open()) {
    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (line.find("cpu MHz") != std::string::npos) {
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
          float mhz = std::stof(line.substr(pos + 1));
          baseFreqGHz = mhz / 1000.0f;
          break;
        }
      }
    }
  }
#else
  // Non-Linux fallback
  physicalCores = std::max(1, logicalCores / 2);
  baseFreqGHz = 1.0f;
#endif

  if (physicalCores <= 0) {
    physicalCores = std::max(1, logicalCores / 2);
  }
}

// ============================================================
// Print with topology details
// ============================================================
void X86HardwareProfile::print() const {
  llvm::outs() << "\n🖥️  Hardware Info:\n";
  llvm::outs() << "   CPU: " << cpuName << "\n";
  llvm::outs() << "   Cores: " << physicalCores << " physical, " << logicalCores
               << " logical\n";

  // Topology details
  if (topology.isHybrid()) {
    llvm::outs() << "\n🏗️  Hybrid Architecture:\n";
    llvm::outs() << "   P-cores: " << topology.numPCores << " ("
                 << topology.numPThreads << " threads, "
                 << topology.threadsPerPCore << " threads/core, "
                 << topology.pCoreMaxFreqKHz / 1000 << " MHz max)\n";
    llvm::outs() << "   E-cores: " << topology.numECores << " ("
                 << topology.numEThreads << " threads, "
                 << topology.threadsPerECore << " thread/core, "
                 << topology.eCoreMaxFreqKHz / 1000 << " MHz max)\n";
    llvm::outs() << "   P/E ratio: " << topology.pToERatio << "x\n";
  }

  // Cache info — per core type if hybrid
  llvm::outs() << "\n💾 Cache Hierarchy:\n";
  if (topology.isHybrid()) {
    llvm::outs() << "   P-core: L1d=" << topology.pCoreCache.l1dKB
                 << "KB  L2=" << topology.pCoreCache.l2KB << "KB\n";
    llvm::outs() << "   E-core: L1d=" << topology.eCoreCache.l1dKB
                 << "KB  L2=" << topology.eCoreCache.l2KB << "KB\n";
  } else {
    llvm::outs() << "   L1d: " << topology.cache.l1dKB << " KB\n";
    llvm::outs() << "   L2:  " << topology.cache.l2KB << " KB\n";
  }
  if (topology.cache.l3KB > 0) {
    llvm::outs() << "   L3:  " << topology.cache.l3KB / 1024
                 << " MB (shared)\n";
  }
  llvm::outs() << "   Line: " << topology.cache.lineSize << " B\n";

  llvm::outs() << "\n⚡ Instruction Sets:\n";
  llvm::outs() << "   AVX2:    " << (hasAVX2_flag ? "✅ Yes" : "❌ No") << "\n";
  llvm::outs() << "   AVX-512: " << (hasAVX512_flag ? "✅ Yes" : "❌ No")
               << "\n";
  llvm::outs() << "   FMA:     " << (hasFMA_flag ? "✅ Yes" : "❌ No") << "\n";
  llvm::outs() << "   VNNI:    " << (hasVNNI_flag ? "✅ Yes" : "❌ No") << "\n";
  llvm::outs() << "   AMX:     " << (hasAMX_flag ? "✅ Yes" : "❌ No") << "\n";

  // Optimal micro-kernel params — per core type
  if (topology.isHybrid()) {
    auto pParams = getOptimalMicroKernelParamsForCoreType(CoreType::P_CORE);
    auto eParams = getOptimalMicroKernelParamsForCoreType(CoreType::E_CORE);
    llvm::outs() << "\n📐 Per-Core-Type Blocking (computed):\n";
    llvm::outs() << "   P-core: KC=" << pParams.KC << " MC=" << pParams.MC
                 << " NC=" << pParams.NC << "\n";
    llvm::outs() << "   E-core: KC=" << eParams.KC << " MC=" << eParams.MC
                 << " NC=" << eParams.NC << "\n";
  } else {
    auto mkp = getOptimalMicroKernelParams();
    llvm::outs() << "\n📐 Optimal Micro-Kernel (computed):\n";
    llvm::outs() << "   MR×NR: " << mkp.MR << "×" << mkp.NR << "\n";
    llvm::outs() << "   Unroll K: " << mkp.unrollK << "\n";
    llvm::outs() << "   KC=" << mkp.KC << ", MC=" << mkp.MC << ", NC=" << mkp.NC
                 << "\n";
  }

  auto tiles = getOptimalTileSizes();
  llvm::outs() << "\n📐 MLIR Tile Sizes:\n";
  llvm::outs() << "   Tile Size: " << tiles.M << "x" << tiles.N << "x"
               << tiles.K << "\n";

  float rpeak = getTheoreticalPeakGFLOPS();
  llvm::outs() << "   Peak GFLOPS (theoretical): " << rpeak << "\n";

  auto strategy = getMicroKernelStrategy();
  llvm::outs() << "\n🎯 Strategy:\n";
  llvm::outs() << "   Use Parallel: " << (strategy.useParallel ? "Yes" : "No")
               << "\n";
  llvm::outs() << "   Threads: " << strategy.numThreads << "\n";
  llvm::outs() << "   Unroll Factor: " << strategy.unrollFactor << "\n";

  // Show heterogeneous work split for 1024 rows
  if (topology.isHybrid()) {
    auto split = HeterogeneousWorkSplit::compute(1024, topology);
    llvm::outs() << "\n🔀 Work Distribution (example: 1024 rows):\n";
    for (auto &tw : split.assignments) {
      const char *typeStr = tw.coreType == CoreType::P_CORE   ? "P"
                            : tw.coreType == CoreType::E_CORE ? "E"
                                                              : "?";
      llvm::outs() << "   Thread " << tw.threadId << " [" << typeStr
                   << "-core, cpu" << tw.cpuId << "]: rows " << tw.rowStart
                   << ".." << tw.rowEnd << " (" << (tw.rowEnd - tw.rowStart)
                   << " rows)\n";
    }
  }
}

TileSizes X86HardwareProfile::getOptimalTileSizes() const {
  if (hasAVX512_flag)
    return {16, 32, 8};
  if (hasAVX2_flag)
    return {8, 16, 4};
  if (hasNEON_flag)
    return {4, 8, 4};
  return {4, 4, 4};
}

Strategy X86HardwareProfile::getMicroKernelStrategy() const {
  Strategy s;
  if (hasAVX512_flag) {
    s.tileM = 16;
    s.tileN = 32;
    s.tileK = 8;
    s.unrollFactor = 8;
  } else if (hasAVX2_flag) {
    s.tileM = 8;
    s.tileN = 16;
    s.tileK = 4;
    s.unrollFactor = 4;
  } else if (hasNEON_flag) {
    s.tileM = 4;
    s.tileN = 8;
    s.tileK = 4;
    s.unrollFactor = 4;
  } else {
    s.tileM = 4;
    s.tileN = 4;
    s.tileK = 4;
    s.unrollFactor = 2;
  }
  s.useAVX512 = hasAVX512_flag;
  s.useAVX2 = hasAVX2_flag && !hasAVX512_flag;
  s.useAMX = hasAMX_flag;
  s.useNEON = hasNEON_flag;
  s.useParallel = (physicalCores > 1);
  s.numThreads = physicalCores;
  return s;
}

float X86HardwareProfile::getTheoreticalPeakGFLOPS() const {
  int opsPerCycle = 2;
  if (hasAVX512_flag && hasFMA_flag)
    opsPerCycle = 32;
  else if (hasAVX2_flag && hasFMA_flag)
    opsPerCycle = 16;
  else if (hasNEON_flag)
    opsPerCycle = 8;
  float freq = turboFreqGHz > 1.0f ? turboFreqGHz : baseFreqGHz;
  return physicalCores * freq * opsPerCycle;
}

bool X86HardwareProfile::shouldUseGPU(int matrixSize) const {
  return matrixSize >= 2048;
}

} // namespace tenzo
