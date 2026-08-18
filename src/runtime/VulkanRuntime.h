#pragma once

#include <vector>
#include <array>
#include <utility>
#include <cstdint>
#include <string>
#include <memory>

namespace tenzo {
namespace runtime {

/// High-Performance Vulkan compute runtime for cross-platform GPU inference
class VulkanRuntime {
public:
    /// Initialize Vulkan runtime
    /// Returns false if Vulkan is not available
    static bool initialize();

    /// Cleanup Vulkan resources
    static void cleanup();

    /// Check if Vulkan is available and initialized
    static bool isAvailable();

    /// Get device name (e.g., "Qualcomm Adreno 750", "Mali-G720", "Intel UHD Graphics")
    static std::string getDeviceName();

    /// Generic SPIR-V dispatch with push constants
    static bool executeShader(
        const std::vector<uint32_t>& spirvBinary,
        const std::vector<std::pair<void*, size_t>>& inputBuffers,
        std::pair<void*, size_t> outputBuffer,
        const void* pushConstantsData,
        size_t pushConstantsSize,
        std::array<uint32_t, 3> numWorkgroups
    );

    /// GPU-Accelerated BitLinear 1.58-bit (TL1) MatVec: y = scale * (W_packed * x)
    static bool executeBitLinearTL1(
        const float* x,
        const uint8_t* W_packed,
        float* y,
        uint32_t N,
        uint32_t K,
        float scale
    );

    /// GPU-Accelerated Dense FP32 GEMV: y = alpha * (W * x)
    static bool executeGemmF32(
        const float* x,
        const float* W,
        float* y,
        uint32_t N,
        uint32_t K,
        float alpha = 1.0f
    );

    /// GPU-Accelerated RMSNorm
    static bool executeRMSNorm(
        const float* x,
        const float* w,
        float* y,
        uint32_t dim,
        float eps = 1e-5f
    );

private:
    static bool s_initialized;
    static std::string s_deviceName;
};

} // namespace runtime
} // namespace tenzo
