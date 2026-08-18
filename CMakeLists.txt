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
# Architecture & Optimization Flags
# ============================================================
option(TENZO_STANDALONE_RUNTIME "Build standalone C++ inference engine without MLIR compiler" OFF)
option(TENZO_USE_NATIVE_ARCH "Use -march=native for host CPU optimizations" ON)
option(TENZO_FAST_MATH "Enable -ffast-math (sacrifices precision for speed)" ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64" OR ANDROID)
    set(TENZO_RUNTIME_FLAGS -O3 -funroll-loops -fopenmp)
    message(STATUS "🚀 Target Architecture: ARM64 / AArch64 (NEON / DotProd)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(TENZO_RUNTIME_FLAGS -mavx2 -mavxvnni -mfma -mf16c -O3 -funroll-loops -fopenmp)
    message(STATUS "🚀 Target Architecture: x86_64 (AVX2 / FMA / VNNI / F16C)")
else()
    set(TENZO_RUNTIME_FLAGS -O3 -funroll-loops -fopenmp)
    message(STATUS "🚀 Target Architecture: Generic CPU")
endif()

if(TENZO_FAST_MATH)
    add_compile_options(-ffast-math -fno-math-errno)
endif()
add_compile_options(-funroll-loops)

# ============================================================
# LLVM / MLIR Detection (Full Compiler vs Standalone Runtime)
# ============================================================
if(NOT TENZO_STANDALONE_RUNTIME)
    find_package(LLVM QUIET CONFIG)
    find_package(MLIR QUIET CONFIG)
endif()

if(LLVM_FOUND AND MLIR_FOUND AND NOT TENZO_STANDALONE_RUNTIME)
    set(TENZO_HAS_MLIR_COMPILER ON)
    message(STATUS "🚀 Full MLIR Compiler Pipeline Enabled")

    # Optional Vulkan support
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        option(TENZO_HAS_VULKAN "Enable Vulkan GPU runtime" ON)
        add_definitions(-DTENZO_HAS_VULKAN)
        message(STATUS "Vulkan support enabled: ${Vulkan_LIBRARY}")
    else()
        option(TENZO_HAS_VULKAN "Enable Vulkan GPU runtime" OFF)
    endif()

    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

    include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
    include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

    link_directories(${LLVM_BUILD_LIBRARY_DIR})
    add_definitions(${LLVM_DEFINITIONS})

    include(TableGen)
    include(AddMLIR)
else()
    set(TENZO_HAS_MLIR_COMPILER OFF)
    message(STATUS "⚡ Building Tenzo Standalone C++ Runtime & CLI (Zero LLVM/MLIR dependency)")
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# OpenMP for parallel loops
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "🚀 OpenMP found: ${OpenMP_CXX_VERSION}")
endif()

# ============================================================
# Full Compiler Targets (Only when MLIR is present)
# ============================================================
# ============================================================
# Full Compiler Targets (Only when MLIR is present)
# ============================================================
if(TENZO_HAS_MLIR_COMPILER)
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
            src/runtime/KVCacheManager.cpp
            src/runtime/Sampler.cpp
            src/runtime/Tokenizer.cpp
            src/context/AutoTuner.cpp
            src/tests/GPUTests.cpp
            src/tests/DynamicInferenceTest.cpp
            src/tests/ZeroCopyBridgeTest.cpp
            src/tests/EndToEndMathTest.cpp src/tests/TestTL1.cpp
            src/tests/GenerationTest.cpp
    )

    add_dependencies(tenzo-cli TenzoOpsIncGen)
    target_link_libraries(tenzo-cli PRIVATE
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
            MLIRIndexDialect
            MLIRIndexToLLVM
            MLIRSCFDialect
            MLIRSCFTransforms
            MLIRTransformDialect
            MLIRTransformDialectTransforms
            MLIRLinalgTransformOps
            MLIRSCFTransformOps
            MLIRTensorTransformOps
            MLIRVectorToLLVMPass
            MLIRVectorToLLVM
            MLIRVectorToSCF
            MLIRVectorTransforms
            MLIRAsyncDialect
            MLIRAsyncTransforms
            MLIRAsyncToLLVM
            MLIROpenMPDialect
            MLIRSCFToOpenMP
            MLIROpenMPToLLVM
            MLIROpenMPToLLVMIRTranslation
            MLIRGPUDialect
            MLIRGPUTransforms
            MLIRGPUToSPIRV
            MLIRSCFToGPU
            MLIRSPIRVDialect
            MLIRSPIRVTransforms
            MLIRSPIRVConversion
            MLIRSPIRVSerialization
            MLIRArithToSPIRV
            MLIRFuncToSPIRV
            MLIRMemRefToSPIRV
            MLIRSCFToSPIRV
            MLIRControlFlowToSPIRV
            ${LLVM_LIBRARIES}
    )

    if(TENZO_HAS_VULKAN AND Vulkan_FOUND)
        target_link_libraries(tenzo-cli PRIVATE Vulkan::Vulkan)
    endif()

    if(OpenMP_CXX_FOUND)
        target_link_libraries(tenzo-cli PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_executable(micro_bench
        src/tests/IsolatedMicroKernel.cpp
    )
    set_property(TARGET micro_bench PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    target_compile_options(micro_bench PRIVATE
        ${TENZO_RUNTIME_FLAGS}
        -fno-lto
    )
    target_link_libraries(micro_bench PRIVATE m)
endif()

# ============================================================
# 📦 Tenzo C/C++ Standalone Runtime Libraries & Executables
# (Zero-dependency on LLVM/MLIR, builds natively on ARM, Termux, Android, x86)
# ============================================================

# Static runtime library
add_library(tenzo_runtime_static STATIC
    src/runtime/TenzoEngine.cpp
)
target_include_directories(tenzo_runtime_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(tenzo_runtime_static PRIVATE
    ${TENZO_RUNTIME_FLAGS}
)
target_link_libraries(tenzo_runtime_static PUBLIC
    OpenMP::OpenMP_CXX
    m
)
set_target_properties(tenzo_runtime_static PROPERTIES OUTPUT_NAME "tenzo_runtime")

# Shared runtime library
add_library(tenzo_runtime_shared SHARED
    src/runtime/TenzoEngine.cpp
)
target_include_directories(tenzo_runtime_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(tenzo_runtime_shared PRIVATE
    ${TENZO_RUNTIME_FLAGS} -DTENZO_BUILD_SHARED
)
target_link_libraries(tenzo_runtime_shared PUBLIC
    OpenMP::OpenMP_CXX
    m
)
set_target_properties(tenzo_runtime_shared PROPERTIES OUTPUT_NAME "tenzo_runtime")

# Standalone C++ Inference Executable (Example)
add_executable(tenzo_basic_inference
    examples/basic_inference.cpp
)
target_include_directories(tenzo_basic_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(tenzo_basic_inference PRIVATE
    ${TENZO_RUNTIME_FLAGS}
)
target_link_libraries(tenzo_basic_inference PRIVATE
    tenzo_runtime_static
    OpenMP::OpenMP_CXX
    m
)

# Production Tenzo CLI Inference Tool
add_executable(tenzo-inference
    src/main_inference.cpp
)
target_include_directories(tenzo-inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(tenzo-inference PRIVATE
    ${TENZO_RUNTIME_FLAGS}
)
target_link_libraries(tenzo-inference PRIVATE
    tenzo_runtime_static
    OpenMP::OpenMP_CXX
    m
)

# ============================================================
# 🐍 Python Bridge (pybind11 - optional)
# ============================================================
find_package(Python COMPONENTS Interpreter Development QUIET)
if(Python_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(tenzo_runtime src/bindings/PybindModule.cpp)
    set_target_properties(tenzo_runtime PROPERTIES INCLUDE_DIRECTORIES "")
    target_include_directories(tenzo_runtime PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_options(tenzo_runtime PRIVATE ${TENZO_RUNTIME_FLAGS})
    target_link_libraries(tenzo_runtime PRIVATE
        OpenMP::OpenMP_CXX
    )
endif()

