#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>

namespace tenzo {
namespace runtime {

/// Simple Vulkan compute runtime for executing SPIR-V shaders
class VulkanRuntime {
public:
    /// Initialize Vulkan runtime
    /// Returns false if Vulkan is not available
    static bool initialize();

    /// Cleanup Vulkan resources
    static void cleanup();

    /// Check if Vulkan is available and initialized
    static bool isAvailable();

    /// Get device name (e.g., "Intel UHD Graphics 770")
    static std::string getDeviceName();

    /// Execute a SPIR-V shader with the given buffers
    /// @param spirvBinary The compiled SPIR-V shader code
    /// @param inputBuffers Vector of input buffer pointers and sizes
    /// @param outputBuffer Output buffer pointer and size
    /// @param workgroupSize Local workgroup size (x, y, z)
    /// @param numWorkgroups Number of workgroups to dispatch (x, y, z)
    /// @return true on success
    static bool execute(
        const std::vector<uint32_t>& spirvBinary,
        const std::vector<std::pair<void*, size_t>>& inputBuffers,
        std::pair<void*, size_t> outputBuffer,
        std::array<uint32_t, 3> workgroupSize,
        std::array<uint32_t, 3> numWorkgroups
    );

    /// Simplified MatMul execution
    /// @param spirvBinary The compiled SPIR-V shader
    /// @param A Input matrix A (row-major)
    /// @param B Input matrix B (row-major)
    /// @param C Output matrix C (row-major)
    /// @param M, N, K Matrix dimensions: C[M,N] = A[M,K] * B[K,N]
    static bool executeMatMul(
        const std::vector<uint32_t>& spirvBinary,
        const float* A, const float* B, float* C,
        uint32_t M, uint32_t N, uint32_t K
    );

private:
    static bool s_initialized;
    static std::string s_deviceName;
};

} // namespace runtime
} // namespace tenzo

