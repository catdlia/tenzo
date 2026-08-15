#include "TenzoContext.h"
#include "dialect/TenzoDialect.h"
#include "passes/Passes.h"

// --- Dialect Headers ---
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

// Translation to LLVM IR
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

// GPU/SPIR-V Dialects
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

// Async Dialect for Multithreading
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"

// OpenMP Dialect
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

// --- Pass Registration Headers ---
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

// Interfaces
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"

// Transform dialect extensions
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"

void tenzo::registerAllDialects(mlir::MLIRContext &context) {
    // 1. Prepare registry
    mlir::DialectRegistry registry;
    
    // Register standard dialects
    registry.insert<tenzo::TenzoDialect, 
                    mlir::func::FuncDialect,
                    mlir::arith::ArithDialect,
                    mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::memref::MemRefDialect,
                    mlir::LLVM::LLVMDialect,
                    mlir::vector::VectorDialect,
                    mlir::scf::SCFDialect,
                    mlir::affine::AffineDialect,
                    mlir::index::IndexDialect,
                    mlir::cf::ControlFlowDialect,
                    mlir::transform::TransformDialect,
                    mlir::gpu::GPUDialect,
                    mlir::spirv::SPIRVDialect,
                    mlir::ub::UBDialect,
                    mlir::async::AsyncDialect,
                    mlir::omp::OpenMPDialect>();

    // 2. Register translations and external models
    mlir::registerAllTranslations();
    mlir::registerAllToLLVMIRTranslations(registry);
    mlir::arith::registerConvertArithToLLVMInterface(registry);
    mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
    mlir::registerConvertFuncToLLVMInterface(registry);
    mlir::index::registerConvertIndexToLLVMInterface(registry);
    mlir::registerConvertMemRefToLLVMInterface(registry);
    mlir::vector::registerConvertVectorToLLVMInterface(registry);
    mlir::registerConvertOpenMPToLLVMInterface(registry);

    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    mlir::tensor::registerTilingInterfaceExternalModels(registry);

    // Transform dialect extensions
    mlir::linalg::registerTransformDialectExtension(registry);
    mlir::scf::registerTransformDialectExtension(registry);
    mlir::tensor::registerTransformDialectExtension(registry);

    // 3. Append to context and load
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    // 4. Register passes
    mlir::registerTransformsPasses();
    mlir::registerLinalgPasses();
    mlir::registerSCFPasses();
    mlir::affine::registerAffinePasses();
    mlir::vector::registerVectorPasses();
    mlir::bufferization::registerBufferizationPasses();
    mlir::memref::registerMemRefPasses();
    mlir::func::registerFuncPasses();
    mlir::arith::registerArithPasses();
    mlir::registerGPUPasses();
    mlir::spirv::registerSPIRVPasses();
}

void tenzo::addHardwareAwareGEMMPipeline(mlir::OpPassManager &pm, 
                                          const std::shared_ptr<tenzo::HardwareProfile> &hwInfo) {
    auto mkParams = hwInfo->getOptimalMicroKernelParams();
    
    // Convert tenzo::MicroKernelParams to tenzo::MicroKernelParams
    tenzo::MicroKernelParams params;
    params.MR = mkParams.MR;
    params.NR = mkParams.NR;
    params.KC = mkParams.KC;
    params.MC = mkParams.MC;
    params.NC = mkParams.NC;
    params.VEC_SIZE = hwInfo->hasAVX512() ? 16 : 8;

    llvm::outs() << "[TenzoContext] Initializing hardware-aware pipeline:\n";
    llvm::outs() << "  Target ISA: " << (hwInfo->hasAVX512() ? "AVX-512" : "AVX2") << "\n";
    llvm::outs() << "  Micro-kernel: " << params.MR << "x" << params.NR << "\n";
    llvm::outs() << "  Cache blocking: KC=" << params.KC << ", MC=" << params.MC << ", NC=" << params.NC << "\n";

    // 1. Packing
    addPackingPass(pm, params);

    // 2. Macro-Kernel Generation (5-loop nest)
    pm.addPass(createGenerateMacroKernelPass(params));

    // 3. Explicit Micro-Kernel Generation (vectorized FMAs)
    addExplicitMicroKernelPass(pm, params);
}
