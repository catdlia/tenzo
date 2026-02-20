#include "TenzoContext.h"
#include "dialect/TenzoDialect.h"

// --- Dialect Headers ---
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"

// Transform dialect extensions
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"

void tenzo::registerAllDialects(mlir::MLIRContext &context) {
    // 1. Реєструємо діалекти
    context.getOrLoadDialect<tenzo::TenzoDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    context.getOrLoadDialect<mlir::tensor::TensorDialect>();
    context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::vector::VectorDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::affine::AffineDialect>();
    context.getOrLoadDialect<mlir::index::IndexDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context.getOrLoadDialect<mlir::transform::TransformDialect>();

    // GPU/SPIR-V Dialects
    context.getOrLoadDialect<mlir::gpu::GPUDialect>();
    context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

    // Async Dialect for Multithreading
    context.getOrLoadDialect<mlir::async::AsyncDialect>();

    // OpenMP Dialect
    context.getOrLoadDialect<mlir::omp::OpenMPDialect>();

    // 2. Реєструємо інтерфейси
    mlir::DialectRegistry registry;
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    mlir::tensor::registerTilingInterfaceExternalModels(registry);

    // Transform dialect extensions для Linalg, SCF та Tensor
    mlir::linalg::registerTransformDialectExtension(registry);
    mlir::scf::registerTransformDialectExtension(registry);
    mlir::tensor::registerTransformDialectExtension(registry);

    context.appendDialectRegistry(registry);

    // 3. Реєструємо паси

    // Core transforms
    mlir::registerTransformsPasses();

    // Global Registration
    mlir::registerLinalgPasses();
    mlir::registerSCFPasses();

    // Nested Registration
    mlir::affine::registerAffinePasses();
    mlir::vector::registerVectorPasses();
    mlir::bufferization::registerBufferizationPasses();
    mlir::memref::registerMemRefPasses();
    mlir::func::registerFuncPasses();
    mlir::arith::registerArithPasses();

    // GPU/SPIR-V passes
    mlir::registerGPUPasses();
    mlir::spirv::registerSPIRVPasses();

    // ВИДАЛЕНО: mlir::registerConvertVectorToLLVMPass();
    // Ми будемо додавати цей пас напряму через create...()
}