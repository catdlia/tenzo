#ifndef TENZO_ROCM_RUNTIME_H
#define TENZO_ROCM_RUNTIME_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace tenzo {

class ROCmRuntime {
public:
    static ROCmRuntime& getInstance();

    bool initialize();
    bool isAvailable() const { return initialized_; }
    std::string getDeviceName() const { return device_name_; }

    // Execute BitLinear TL1 GEMV via ROCm / HIP or Vulkan translation
    bool executeBitLinearTL1(
        const float* d_act,
        const uint8_t* d_weight,
        float* d_out,
        size_t M,
        size_t K,
        size_t N,
        float scale
    );

    // Execute RMSNorm
    bool executeRMSNorm(
        const float* d_in,
        const float* d_weight,
        float* d_out,
        size_t dim,
        float eps = 1e-5f
    );

    // Execute FP32 GEMM
    bool executeGemmF32(
        const float* d_A,
        const float* d_B,
        float* d_C,
        size_t M,
        size_t K,
        size_t N
    );

private:
    ROCmRuntime();
    ~ROCmRuntime();

    bool initialized_ = false;
    bool using_vulkan_translation_ = false;
    std::string device_name_ = "ROCm Not Initialized";
};

} // namespace tenzo

#endif // TENZO_ROCM_RUNTIME_H
