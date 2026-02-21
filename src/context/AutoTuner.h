#pragma once
#include "HardwareInfo.h"
#include "passes/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include <vector>
#include <map>

namespace tenzo {

/// Experimental auto-tuner for micro-kernel selection
class AutoTuner {
public:
    struct TuningResult {
        MicroKernelParams params;
        double gflops = 0.0;
        double timeMs = 0.0;
    };

    AutoTuner(mlir::MLIRContext &ctx) : context(ctx) {}

    /// Runs the tuning process and returns the best parameters found
    MicroKernelParams tune(const HardwareInfo &hwInfo);

    /// Saves best configuration for current CPU to disk
    void saveConfig(const std::string& filename, const MicroKernelParams& params);

    /// Loads configuration if available
    bool loadConfig(const std::string& filename, MicroKernelParams& params);

private:
    mlir::MLIRContext &context;

    std::string getConfigPath(const std::string& cpuName);

    /// Generates a list of candidate parameters based on hardware limits (registers)
    std::vector<MicroKernelParams> generateCandidates(const HardwareInfo &hwInfo);

    /// Benchmarks a specific configuration
    TuningResult benchmark(const MicroKernelParams &params, const HardwareInfo &hwInfo);

    /// Generates a small MLIR module for the micro-kernel to test
    mlir::OwningOpRef<mlir::ModuleOp> generateTestModule(const MicroKernelParams &params);
};

} // namespace tenzo
