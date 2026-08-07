#include "AutoTuner.h"
#include "context/TenzoContext.h"
#include "passes/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <fstream>
#include <sstream>

namespace tenzo {

// Forward declaration
MicroKernelParams
paramsFromHw(const std::shared_ptr<tenzo::HardwareProfile> hwInfo);

static std::string sanitize(std::string name) {
  std::replace(name.begin(), name.end(), ' ', '_');
  std::replace(name.begin(), name.end(), '/', '_');
  return name;
}

std::string AutoTuner::getConfigPath(const std::string &cpuName) {
  std::string home = getenv("HOME") ? getenv("HOME") : ".";
  std::string dir = home + "/.tenzo/cache";
  std::filesystem::create_directories(dir);
  return dir + "/tuner_" + sanitize(cpuName) + ".conf";
}

void AutoTuner::saveConfig(const std::string &path,
                           const MicroKernelParams &params) {
  std::ofstream f(path);
  if (!f)
    return;
  f << params.MR << " " << params.NR << " " << params.KC << " " << params.MC
    << " " << params.NC << " " << params.VEC_SIZE << "\n";
  llvm::outs() << "[AutoTuner] Configuration saved to " << path << "\n";
}

bool AutoTuner::loadConfig(const std::string &path, MicroKernelParams &params) {
  std::ifstream f(path);
  if (!f)
    return false;
  f >> params.MR >> params.NR >> params.KC >> params.MC >> params.NC >>
      params.VEC_SIZE;
  return f.good();
}

MicroKernelParams
AutoTuner::tune(const std::shared_ptr<tenzo::HardwareProfile> &hwInfo) {
  std::string path = getConfigPath(hwInfo->getCpuName());
  MicroKernelParams cached;
  if (loadConfig(path, cached)) {
    llvm::outs() << "[AutoTuner] Using cached config for " << hwInfo->getCpuName()
                 << ": " << cached.MR << "x" << cached.NR
                 << " (KC=" << cached.KC << ")\n";
    return cached;
  }

  auto candidates = generateCandidates(hwInfo);
  llvm::outs() << "[AutoTuner] No cache found. Starting tuning with "
               << candidates.size() << " candidates...\n";

  MicroKernelParams bestParams;
  double bestGflops = -1.0;

  for (const auto &p : candidates) {
    auto result = benchmark(p, hwInfo);
    llvm::outs() << "  Candidate " << p.MR << "x" << p.NR << ": "
                 << result.gflops << " GFLOPS\n";

    if (result.gflops > bestGflops) {
      bestGflops = result.gflops;
      bestParams = p;
    }
  }

  if (bestGflops > 0) {
    saveConfig(path, bestParams);
  } else {
    // Fallback to defaults from std::shared_ptr<tenzo::HardwareProfile> if
    // tuning failed
    bestParams = paramsFromHw(hwInfo);
  }

  return bestParams;
}

// Helper for defaults
MicroKernelParams
paramsFromHw(const std::shared_ptr<tenzo::HardwareProfile> hwInfo) {
  auto mkp = hwInfo->getOptimalMicroKernelParams();
  MicroKernelParams p;
  p.MR = mkp.MR;
  p.NR = mkp.NR;
  p.KC = mkp.KC;
  p.MC = mkp.MC;
  p.NC = mkp.NC;
  p.VEC_SIZE = hwInfo->hasAVX512() ? 16 : 8;
  return p;
}

std::vector<MicroKernelParams> AutoTuner::generateCandidates(
    const std::shared_ptr<tenzo::HardwareProfile> &hwInfo) {
  std::vector<MicroKernelParams> candidates;
  if (hwInfo->hasAVX2() && !hwInfo->hasAVX512()) {
    candidates.push_back({6, 16, 256, 96, 256, 8});  // Standard FP32 6x16
    candidates.push_back({4, 16, 256, 96, 256, 8});  // 1.58-bit candidate (4x16, 8 YMM freed)
    candidates.push_back({4, 8, 256, 96, 256, 8});   // 1.58-bit candidate (4x8, 10 YMM freed)
    candidates.push_back({4, 24, 256, 96, 256, 8});
  } else if (hwInfo->hasAVX512()) {
    candidates.push_back({14, 32, 512, 128, 512, 16});
    candidates.push_back({8, 32, 512, 128, 512, 16});  // 1.58-bit candidate for AVX512
    candidates.push_back({8, 48, 512, 128, 512, 16});
  }
  return candidates;
}

AutoTuner::TuningResult
AutoTuner::benchmark(const MicroKernelParams &params,
                     const std::shared_ptr<tenzo::HardwareProfile> &hwInfo) {
  // This would invoke the MLIR compiler pipeline and run the kernel.
  // For this demonstration, we'll simulate a benchmark result.
  TuningResult res;
  res.params = params;
  res.gflops = 1.0; // Simulated
  return res;
}

} // namespace tenzo
