#include "CUDARuntime.h"
#include "VulkanRuntime.h"
#include <iostream>
#include <dlfcn.h>

namespace tenzo {

CUDARuntime& CUDARuntime::getInstance() {
    static CUDARuntime instance;
    return instance;
}

CUDARuntime::CUDARuntime() {}

CUDARuntime::~CUDARuntime() {}

bool CUDARuntime::initialize() {
    if (initialized_) return true;

    // 1. Try loading real CUDA Driver API
    void* libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!libcuda) {
        libcuda = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    }
    
    if (libcuda) {
        device_name_ = "NVIDIA CUDA Hardware Device (Native Driver)";
        initialized_ = true;
        using_vulkan_translation_ = false;
        std::cout << "🚀 [CUDA Engine] Native NVIDIA CUDA Driver detected.\n";
        return true;
    }

    // 2. Fallback to Vulkan Translation Layer (as requested)
    std::cout << "⚡ [CUDA Engine] Native CUDA Driver not found. Initializing Vulkan Translation Layer...\n";
    if (runtime::VulkanRuntime::initialize()) {
        initialized_ = true;
        using_vulkan_translation_ = true;
        device_name_ = "CUDA (Vulkan Translation Layer: " + runtime::VulkanRuntime::getDeviceName() + ")";
        std::cout << "🚀 [CUDA Engine] CUDA-on-Vulkan translation active: " << runtime::VulkanRuntime::getDeviceName() << "\n";
        return true;
    }

    device_name_ = "CUDA Backend Unavailable";
    return false;
}

bool CUDARuntime::executeBitLinearTL1(
    const float* d_act,
    const uint8_t* d_weight,
    float* d_out,
    size_t M,
    size_t K,
    size_t N,
    float scale
) {
    if (!initialized_) {
        if (!initialize()) return false;
    }

    if (using_vulkan_translation_) {
        return runtime::VulkanRuntime::executeBitLinearTL1(d_act, d_weight, d_out, static_cast<uint32_t>(N), static_cast<uint32_t>(K), scale);
    }

    return false;
}

bool CUDARuntime::executeRMSNorm(
    const float* d_in,
    const float* d_weight,
    float* d_out,
    size_t dim,
    float eps
) {
    if (!initialized_) {
        if (!initialize()) return false;
    }

    if (using_vulkan_translation_) {
        return runtime::VulkanRuntime::executeRMSNorm(d_in, d_weight, d_out, static_cast<uint32_t>(dim), eps);
    }

    return false;
}

bool CUDARuntime::executeGemmF32(
    const float* d_A,
    const float* d_B,
    float* d_C,
    size_t M,
    size_t K,
    size_t N
) {
    if (!initialized_) {
        if (!initialize()) return false;
    }

    if (using_vulkan_translation_) {
        return runtime::VulkanRuntime::executeGemmF32(d_A, d_B, d_C, static_cast<uint32_t>(N), static_cast<uint32_t>(K), 1.0f);
    }

    return false;
}

} // namespace tenzo
