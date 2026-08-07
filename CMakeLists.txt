cmake_minimum_required(VERSION 3.20)
project(tenzo-compiler)

set(CMAKE_CXX_STANDARD 17)

# ============================================================
# 🚀 TURBO BOOST: Build Acceleration
# ============================================================

# CCache - MUST be set BEFORE project() takes effect on compilers
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "🚀 Using CCache: ${CCACHE_PROGRAM}")
endif()

# LLD Linker - 10-20x faster than default ld
find_program(LLD_PATH lld)
if(LLD_PATH)
    add_link_options("-fuse-ld=lld")
    message(STATUS "🚀 Using LLD linker: ${LLD_PATH}")
endif()

# ============================================================

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# Optional Vulkan support - try to find it automatically
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    option(TENZO_HAS_VULKAN "Enable Vulkan GPU runtime" ON)
else()
    option(TENZO_HAS_VULKAN "Enable Vulkan GPU runtime" OFF)
endif()

if(TENZO_HAS_VULKAN AND Vulkan_FOUND)
    add_definitions(-DTENZO_HAS_VULKAN)
    message(STATUS "Vulkan support enabled: ${Vulkan_LIBRARY}")
else()
    message(STATUS "Vulkan support disabled (SDK not found)")
endif()

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

include(TableGen)
include(AddMLIR)

# ============================================================
# 🚀 MICRO-ARCHITECTURE OPTIMIZATION FLAGS (Phase 3)
# ============================================================
# These flags enable aggressive optimizations for the host CPU

# Use native CPU instructions (AVX2, AVX-512, etc.)
option(TENZO_USE_NATIVE_ARCH "Use -march=native for host CPU optimizations" ON)

# Enable fast-math for aggressive floating-point optimizations
option(TENZO_FAST_MATH "Enable -ffast-math (sacrifices precision for speed)" ON)

if(TENZO_USE_NATIVE_ARCH)
    add_compile_options(-march=native)
    message(STATUS "🚀 Using native CPU architecture optimizations")
endif()

if(TENZO_FAST_MATH)
    add_compile_options(-ffast-math -fno-math-errno)
    message(STATUS "🚀 Fast-math enabled (reduced precision)")
endif()

# Always enable these safe optimizations
add_compile_options(-funroll-loops)

# For Release builds, enable LTO (Link-Time Optimization)
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
    # LTO disabled to fix linker error (undefined symbol: main)
    # if(IPO_SUPPORTED)
    #     set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    #     message(STATUS "🚀 Link-Time Optimization (LTO) enabled")
    # endif()
endif()

# OpenMP for parallel loops
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "🚀 OpenMP found: ${OpenMP_CXX_VERSION}")
endif()

# ============================================================

# --- TABLEGEN ---
set(LLVM_TARGET_DEFINITIONS src/dialect/TenzoOps.td)
mlir_tablegen(TenzoOps.h.inc -gen-op-decls)
mlir_tablegen(TenzoOps.cpp.inc -gen-op-defs)
mlir_tablegen(TenzoDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TenzoDialect.cpp.inc -gen-dialect-defs)

add_custom_target(TenzoOpsIncGen DEPENDS
        TenzoOps.h.inc TenzoOps.cpp.inc
        TenzoDialect.h.inc TenzoDialect.cpp.inc)

# --- БІНАРНИК ---
add_executable(tenzo-cli
        src/main.cpp
        src/dialect/TenzoDialect.cpp
        src/passes/Lowering.cpp
        src/context/TenzoContext.cpp
        src/context/HardwareProfile.cpp
        src/context/X86HardwareProfile.cpp
        src/context/ARMHardwareProfile.cpp
        src/tests/PipelineTests.cpp
        src/tests/MicroKernelBench.cpp
        src/tests/BLISBenchmark.cpp
        src/tests/GenerateMicroKernel.cpp
        src/passes/FusionPass.cpp
        src/passes/LinalgLowering.cpp
        src/passes/Bufferization.cpp
        src/passes/LLVMLowering.cpp
        src/passes/TransformStrategy.cpp
        src/passes/OptimalVectorization.cpp
        src/passes/PackingPass.cpp
        src/passes/ExplicitMicroKernelPass.cpp
        src/passes/FusedTernaryKernelPass.cpp
        src/passes/PackingKernels.cpp
        src/passes/MacroKernelPass.cpp
        # Tests
        src/tests/PackingBenchmark.cpp
        src/tests/FullGEMMBenchmark.cpp
        src/tests/GEMMEndToEnd.cpp
        src/tests/LinearLayer.cpp
        # GPU/SPIR-V
        src/passes/gpu/GPULowering.cpp
        src/runtime/VulkanRuntime.cpp
        src/runtime/ThreadPool.cpp
        src/runtime/ExecutionContext.cpp
        src/context/AutoTuner.cpp
        src/tests/GPUTests.cpp
        src/tests/DynamicInferenceTest.cpp
        src/tests/ZeroCopyBridgeTest.cpp
        src/tests/EndToEndMathTest.cpp
)

add_dependencies(tenzo-cli TenzoOpsIncGen)

target_link_libraries(tenzo-cli PRIVATE
        # --- MLIR Ліби ---
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRInferTypeOpInterface
        MLIRBytecodeOpInterface
        MLIRFuncDialect
        MLIRArithDialect
        MLIRPass
        MLIRTransforms
        MLIRLinalgDialect
        MLIRTensorDialect
        MLIRAffineDialect
        MLIRAffineTransforms
        MLIRAffineUtils
        MLIRAffineToStandard
        MLIRVectorDialect
        MLIRLinalgTransforms
        MLIRBufferizationDialect
        MLIRBufferizationTransforms
        MLIRBufferizationToMemRef
        MLIRMemRefDialect
        MLIRMemRefTransforms
        MLIRLLVMDialect
        MLIRMemRefToLLVM
        MLIRFuncToLLVM
        MLIRArithToLLVM
        MLIRControlFlowToLLVM
        MLIRReconcileUnrealizedCasts
        MLIRLinalgToStandard
        MLIRSCFToControlFlow
        MLIRTargetLLVMIRExport
        MLIRExecutionEngine
        MLIRBuiltinToLLVMIRTranslation
        MLIRLLVMToLLVMIRTranslation

        # Index dialect and conversion
        MLIRIndexDialect
        MLIRIndexToLLVM

        # SCF dialect
        MLIRSCFDialect
        MLIRSCFTransforms

        # Transform Dialect
        MLIRTransformDialect
        MLIRTransformDialectTransforms
        MLIRLinalgTransformOps
        MLIRSCFTransformOps
        MLIRTensorTransformOps

        # Vector Pass Dependencies
        MLIRVectorToLLVMPass
        MLIRVectorToLLVM
        MLIRVectorToSCF
        MLIRVectorTransforms

        # --- ASYNC & PARALLELISM LIBS ---
        MLIRAsyncDialect
        MLIRAsyncTransforms
        MLIRAsyncToLLVM

        # --- OpenMP for Parallelism ---
        MLIROpenMPDialect
        MLIRSCFToOpenMP
        MLIROpenMPToLLVM
        MLIROpenMPToLLVMIRTranslation

        # GPU Dialect and Transforms
        MLIRGPUDialect
        MLIRGPUTransforms
        MLIRGPUToSPIRV
        MLIRSCFToGPU

        # SPIR-V Dialect and Transforms
        MLIRSPIRVDialect
        MLIRSPIRVTransforms
        MLIRSPIRVConversion
        MLIRSPIRVSerialization
        MLIRArithToSPIRV
        MLIRFuncToSPIRV
        MLIRMemRefToSPIRV
        MLIRSCFToSPIRV
        MLIRControlFlowToSPIRV

        # --- LLVM Ліби ---
        ${LLVM_LIBRARIES}
)

# Link Vulkan if available
if(TENZO_HAS_VULKAN AND Vulkan_FOUND)
    target_link_libraries(tenzo-cli PRIVATE Vulkan::Vulkan)
endif()

# Link OpenMP for parallel loop execution
if(OpenMP_CXX_FOUND)
    target_link_libraries(tenzo-cli PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link OpenBLAS for GEMM comparison benchmarks (optional)
find_library(OPENBLAS_LIB openblas)
if(OPENBLAS_LIB)
    target_compile_definitions(tenzo-cli PRIVATE TENZO_HAS_OPENBLAS)
    target_link_libraries(tenzo-cli PRIVATE ${OPENBLAS_LIB})
    message(STATUS "OpenBLAS found: ${OPENBLAS_LIB}")
else()
    message(STATUS "OpenBLAS not found — comparison benchmarks disabled")
endif()

# ============================================================
# 🔬 MICRO-KERNEL ISOLATED BENCHMARK
# ============================================================
# This is a standalone benchmark for measuring pure micro-kernel performance
# WITHOUT any MLIR/JIT overhead. Goal: 45-50 GFLOPS on 6x16x256.

add_executable(micro_bench
    src/tests/IsolatedMicroKernel.cpp
)

# Disable LTO for this target (conflicts with standalone benchmark)
set_property(TARGET micro_bench PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)

# Enable AVX2 + FMA for intrinsics baseline
target_compile_options(micro_bench PRIVATE
    -mavx2
    -mfma
    -O3
    -funroll-loops
    -ffast-math
    -fno-lto
)

# Link math library
target_link_libraries(micro_bench PRIVATE m)

# Optional: Link MLIR-generated kernel if available
# To enable: cmake -DUSE_MLIR_KERNEL=ON ..
option(USE_MLIR_KERNEL "Link MLIR-generated micro-kernel" OFF)
if(USE_MLIR_KERNEL)
    # Check for pre-compiled MLIR object file first
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/micro_kernel_mlir.o")
        message(STATUS "🔗 Linking pre-compiled MLIR micro-kernel: micro_kernel_mlir.o")
        target_compile_definitions(micro_bench PRIVATE HAS_MLIR_KERNEL)
        target_sources(micro_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/micro_kernel_mlir.o)
        set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/micro_kernel_mlir.o
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED FALSE)
    else()
        # Try to compile from MLIR source
        find_program(MLIR_OPT mlir-opt HINTS /usr/lib/llvm-21/bin)
        find_program(MLIR_TRANSLATE mlir-translate HINTS /usr/lib/llvm-21/bin)
        find_program(LLC_PROGRAM llc-21 llc HINTS /usr/lib/llvm-21/bin)

        if(MLIR_OPT AND MLIR_TRANSLATE AND LLC_PROGRAM)
            message(STATUS "🔧 Found MLIR tools, will compile from source")
            message(STATUS "   mlir-opt: ${MLIR_OPT}")
            message(STATUS "   mlir-translate: ${MLIR_TRANSLATE}")
            message(STATUS "   llc: ${LLC_PROGRAM}")

            set(MLIR_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/micro_kernel.mlir)
            set(MLIR_LLVM ${CMAKE_CURRENT_BINARY_DIR}/micro_kernel_llvm.mlir)
            set(LLVM_IR ${CMAKE_CURRENT_BINARY_DIR}/micro_kernel.ll)
            set(OBJECT_FILE ${CMAKE_CURRENT_BINARY_DIR}/micro_kernel_mlir.o)

            # Step 1: MLIR → LLVM Dialect
            add_custom_command(
                OUTPUT ${MLIR_LLVM}
                COMMAND ${MLIR_OPT} ${MLIR_SOURCE}
                        --convert-to-llvm
                        --reconcile-unrealized-casts
                        -o ${MLIR_LLVM}
                DEPENDS ${MLIR_SOURCE}
                COMMENT "🔄 Converting MLIR to LLVM Dialect"
            )

            # Step 2: LLVM Dialect → LLVM IR
            add_custom_command(
                OUTPUT ${LLVM_IR}
                COMMAND ${MLIR_TRANSLATE} --mlir-to-llvmir ${MLIR_LLVM} -o ${LLVM_IR}
                DEPENDS ${MLIR_LLVM}
                COMMENT "🔄 Translating LLVM Dialect to LLVM IR"
            )

            # Step 3: LLVM IR → Object File
            add_custom_command(
                OUTPUT ${OBJECT_FILE}
                COMMAND ${LLC_PROGRAM} ${LLVM_IR}
                        -O3 -mcpu=native -mattr=+avx2,+fma
                        -filetype=obj -o ${OBJECT_FILE}
                DEPENDS ${LLVM_IR}
                COMMENT "🔄 Compiling LLVM IR to object file"
            )

            # Add to micro_bench
            target_compile_definitions(micro_bench PRIVATE HAS_MLIR_KERNEL)
            target_sources(micro_bench PRIVATE ${OBJECT_FILE})
            set_source_files_properties(${OBJECT_FILE}
                PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        else()
            message(WARNING "⚠️ MLIR tools not found, cannot compile micro-kernel")
            message(WARNING "   Please compile manually and provide micro_kernel_mlir.o")
        endif()
    endif()
endif()

