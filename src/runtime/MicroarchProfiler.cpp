#include "MicroarchProfiler.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <thread>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

#if defined(__linux__) || defined(__ANDROID__)
#include <sys/auxv.h>
#endif

namespace tenzo {

MicroarchProfile& MicroarchProfiler::getProfile() {
    static MicroarchProfile profile;
    static bool initialized = false;
    if (!initialized) {
        detectCPU(profile);
        detectCaches(profile);
        detectSIMD(profile);
        detectGPU(profile);
        calibrateTiling(profile, 2560, 6912);
        initialized = true;
    }
    return profile;
}

void MicroarchProfiler::detectCPU(MicroarchProfile& prof) {
    prof.num_threads = std::thread::hardware_concurrency();
    if (prof.num_threads <= 0) prof.num_threads = 4;
    prof.num_cores = prof.num_threads;

#if defined(__x86_64__) || defined(_M_X64)
    prof.cpu_arch = "x86_64";
    prof.cpu_vendor = "Intel/AMD";
#elif defined(__aarch64__) || defined(_M_ARM64)
    prof.cpu_arch = "aarch64";
    prof.cpu_vendor = "ARM";
#elif defined(__riscv) || defined(__riscv_xlen)
    prof.cpu_arch = "riscv64";
    prof.cpu_vendor = "RISC-V";
#else
    prof.cpu_arch = "generic";
    prof.cpu_vendor = "Generic";
#endif

#if defined(__linux__)
    // Parse /proc/cpuinfo if available
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("Hardware") != std::string::npos || line.find("model name") != std::string::npos) {
                auto pos = line.find(':');
                if (pos != std::string::npos) {
                    prof.cpu_vendor = line.substr(pos + 2);
                }
                break;
            }
        }
    }
#endif
}

void MicroarchProfiler::detectCaches(MicroarchProfile& prof) {
    // Default sensible values
    prof.l1d_cache_size = 32 * 1024;
    prof.l2_cache_size = 512 * 1024;
    prof.l3_cache_size = 8 * 1024 * 1024;
    prof.cache_line_size = 64;

#if defined(_SC_LEVEL1_DCACHE_SIZE)
    long l1 = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    if (l1 > 0) prof.l1d_cache_size = static_cast<size_t>(l1);
#endif
#if defined(_SC_LEVEL2_CACHE_SIZE)
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l2 > 0) prof.l2_cache_size = static_cast<size_t>(l2);
#endif
#if defined(_SC_LEVEL3_CACHE_SIZE)
    long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (l3 > 0) prof.l3_cache_size = static_cast<size_t>(l3);
#endif
#if defined(_SC_LEVEL1_DCACHE_LINESIZE)
    long cls = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (cls > 0) prof.cache_line_size = static_cast<size_t>(cls);
#endif
}

void MicroarchProfiler::detectSIMD(MicroarchProfile& prof) {
#if defined(__x86_64__) || defined(_M_X64)
    prof.vector_bits = 256; // AVX2 default
    #if defined(__AVX512F__)
    prof.vector_bits = 512;
    #endif
    prof.has_fp16 = true;
#elif defined(__aarch64__) || defined(_M_ARM64)
    prof.vector_bits = 128; // NEON default
    prof.has_dotprod = true;
    prof.has_fp16 = true;
    #if defined(__ARM_FEATURE_SVE2)
    prof.has_sve2 = true;
    prof.vector_bits = 256;
    #elif defined(__ARM_FEATURE_SVE)
    prof.has_sve = true;
    prof.vector_bits = 256;
    #endif
    #if defined(__ARM_FEATURE_MATMUL_INT8)
    prof.has_i8mm = true;
    #endif
#elif defined(__riscv)
    prof.vector_bits = 128;
    #if defined(__riscv_vector)
    prof.has_rvv = true;
    #endif
#endif
}

void MicroarchProfiler::detectGPU(MicroarchProfile& prof) {
#if defined(TENZO_HAS_VULKAN)
    prof.has_gpu = true;
    prof.gpu_api = "Vulkan";
    prof.gpu_device_name = "Vulkan Heterogeneous Acceleration Device";
    prof.gpu_subgroup_size = 32;
    prof.gpu_shared_memory_bytes = 32 * 1024;
#else
    prof.has_gpu = false;
    prof.gpu_api = "None";
#endif
}

void MicroarchProfiler::calibrateTiling(MicroarchProfile& prof, size_t hidden_size, size_t intermediate_size) {
    // Determine optimal inner reduction tile K and output tile N based on L1d/L2 cache capacity
    // Working set for GEMV: (K * 2/8 bytes for TL1 weights) + (K * 4 bytes for input activation)
    // Target fitting 4 active tile buffers into L1D (typically 32-64KB)
    size_t target_l1_bytes = prof.l1d_cache_size / 2;
    size_t bytes_per_k = 4 + 1; // 4 bytes input + 0.25..1 byte weight
    
    size_t ideal_k = target_l1_bytes / bytes_per_k;
    // Align to vector register width
    size_t vec_elems = prof.vector_bits / 32;
    if (vec_elems < 4) vec_elems = 4;
    
    prof.tile_k = 64;
    while (prof.tile_k * 2 <= ideal_k && prof.tile_k < 512) {
        prof.tile_k *= 2;
    }
    
    // Output tile N: fit active accumulator rows into L2 cache
    prof.tile_n = 64;
    if (prof.l2_cache_size >= 1024 * 1024) {
        prof.tile_n = 128;
    }
    
    // Optimal thread pool allocation (leave 1 core for OS background tasks if >= 4 cores)
    if (prof.num_threads >= 8) {
        prof.optimal_threads = prof.num_threads - 1;
    } else if (prof.num_threads >= 4) {
        prof.optimal_threads = prof.num_threads;
    } else {
        prof.optimal_threads = prof.num_threads;
    }
}

void MicroarchProfiler::printReport(const MicroarchProfile& prof) {
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║      Tenzo Hardware Microarchitectural Profile v1.0    ║\n";
    std::cout << "╠════════════════════════════════════════════════════════╣\n";
    std::cout << "║ CPU Architecture:       " << prof.cpu_arch << " (" << prof.cpu_vendor << ")\n";
    std::cout << "║ Hardware Threads:       " << prof.num_threads << " threads (Optimal: " << prof.optimal_threads << ")\n";
    std::cout << "║ Cache Hierarchy:        L1d: " << (prof.l1d_cache_size / 1024) << "KB | L2: " << (prof.l2_cache_size / 1024) << "KB | L3: " << (prof.l3_cache_size / (1024*1024)) << "MB\n";
    std::cout << "║ Vector Width (VLEN):    " << prof.vector_bits << "-bit SIMD\n";
    std::cout << "║ SIMD Extensions:        ";
    if (prof.has_dotprod) std::cout << "DotProd ";
    if (prof.has_i8mm) std::cout << "i8mm ";
    if (prof.has_sve2) std::cout << "SVE2 ";
    if (prof.has_rvv) std::cout << "RVV-1.0 ";
    std::cout << "\n";
    std::cout << "║ GPU Acceleration:       " << (prof.has_gpu ? prof.gpu_api : "Disabled / CPU-only") << "\n";
    std::cout << "║ Calibrated Auto-Tiles:  M=" << prof.tile_m << ", N=" << prof.tile_n << ", K=" << prof.tile_k << "\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";
}

} // namespace tenzo
