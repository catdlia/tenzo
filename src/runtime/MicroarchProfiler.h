#ifndef TENZO_MICROARCH_PROFILER_H
#define TENZO_MICROARCH_PROFILER_H

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace tenzo {

struct MicroarchProfile {
    std::string cpu_arch;           // "x86_64", "aarch64", "riscv64", etc.
    std::string cpu_vendor;         // "Intel", "AMD", "Qualcomm", "ARM", "Generic"
    int num_cores = 1;
    int num_threads = 1;
    
    // Cache Hierarchy (in Bytes)
    size_t l1d_cache_size = 32 * 1024;    // Default 32 KB
    size_t l2_cache_size = 512 * 1024;    // Default 512 KB
    size_t l3_cache_size = 8 * 1024 * 1024; // Default 8 MB
    size_t cache_line_size = 64;          // Default 64 Bytes
    
    // Vector SIMD Capabilities
    int vector_bits = 128;                // 128 (NEON/SSE), 256 (AVX2), 512 (AVX-512), or RVV VLEN
    bool has_fp16 = false;
    bool has_dotprod = false;
    bool has_i8mm = false;
    bool has_sve = false;
    bool has_sve2 = false;
    bool has_rvv = false;
    
    // GPU Capabilities
    bool has_gpu = false;
    std::string gpu_device_name = "None";
    std::string gpu_api = "None";        // "Vulkan", "CUDA", "ROCm", "Mock"
    int gpu_subgroup_size = 32;          // 32 (NV/Mali/Intel), 64 (AMD/Adreno)
    size_t gpu_shared_memory_bytes = 32 * 1024; // 32-64 KB
    
    // Computed Optimal Tile Dimensions
    size_t tile_m = 1;                   // Batch size / token tile
    size_t tile_n = 64;                  // Output channel tile
    size_t tile_k = 64;                  // Inner reduction tile
    int optimal_threads = 4;
};

class MicroarchProfiler {
public:
    static MicroarchProfile& getProfile();
    static void printReport(const MicroarchProfile& prof);
    static void calibrateTiling(MicroarchProfile& prof, size_t hidden_size, size_t intermediate_size);
    
private:
    static void detectCPU(MicroarchProfile& prof);
    static void detectCaches(MicroarchProfile& prof);
    static void detectSIMD(MicroarchProfile& prof);
    static void detectGPU(MicroarchProfile& prof);
};

} // namespace tenzo

#endif // TENZO_MICROARCH_PROFILER_H
