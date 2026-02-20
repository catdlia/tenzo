#include "GPULowering.h"

// GPU Dialect
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"

// SPIR-V Dialect
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Target/SPIRV/Serialization.h"

// Conversions
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRVPass.h"

// SPIRV conversion utilities
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

// Additional for unified conversion
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"

// Linalg
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

// Common
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace {

//===----------------------------------------------------------------------===//
// Pass: Set SPIR-V target environment on GPU module
//===----------------------------------------------------------------------===//
struct SetSPIRVTargetEnvPass
    : public mlir::PassWrapper<SetSPIRVTargetEnvPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetSPIRVTargetEnvPass)

    void runOnOperation() override {
        auto module = getOperation();
        auto *ctx = &getContext();

        llvm::outs() << "[SetSPIRVTargetEnv] Setting target environment\n";

        // Create Vulkan 1.0 target environment
        auto triple = mlir::spirv::VerCapExtAttr::get(
            mlir::spirv::Version::V_1_0,
            {mlir::spirv::Capability::Shader},
            {},
            ctx);
        auto targetAttr = mlir::spirv::TargetEnvAttr::get(
            triple,
            mlir::spirv::getDefaultResourceLimits(ctx),
            mlir::spirv::ClientAPI::Vulkan,
            mlir::spirv::Vendor::Unknown,
            mlir::spirv::DeviceType::Unknown,
            /*deviceId=*/0);

        // Set target env on each gpu.module
        module.walk([&](mlir::gpu::GPUModuleOp gpuModule) {
            gpuModule->setAttr(mlir::spirv::getTargetEnvAttrName(), targetAttr);
            llvm::outs() << "[SetSPIRVTargetEnv] Set on: " << gpuModule.getName() << "\n";
        });
    }
};

//===----------------------------------------------------------------------===//
// Pass: Set SPIR-V entry point ABI attributes on GPU kernels
//===----------------------------------------------------------------------===//
struct SetSPIRVABIPass
    : public mlir::PassWrapper<SetSPIRVABIPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetSPIRVABIPass)

    void runOnOperation() override {
        auto module = getOperation();
        auto *ctx = &getContext();

        llvm::errs() << "[SetSPIRVABI] Processing module\n";

        // Walk all gpu.func operations and add spirv.entry_point_abi
        module.walk([&](mlir::gpu::GPUFuncOp funcOp) {
            llvm::errs() << "[SetSPIRVABI] Found gpu.func: " << funcOp.getName() << "\n";

            if (funcOp.isKernel()) {
                // Default workgroup size [1, 1, 1]
                auto workgroupSize = mlir::DenseI32ArrayAttr::get(ctx, {1, 1, 1});
                auto abiAttr = mlir::spirv::EntryPointABIAttr::get(ctx, workgroupSize, std::nullopt);
                funcOp->setAttr("spirv.entry_point_abi", abiAttr);
                llvm::errs() << "[SetSPIRVABI] Set spirv.entry_point_abi attribute\n";
            }
        });
    }
};

//===----------------------------------------------------------------------===//
// Pass: Create spirv.module wrapper for spirv ops inside gpu.module
//===----------------------------------------------------------------------===//
struct CreateSPIRVModulePass
    : public mlir::PassWrapper<CreateSPIRVModulePass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CreateSPIRVModulePass)

    void runOnOperation() override {
        auto gpuModule = getOperation();
        auto *ctx = &getContext();

        llvm::outs() << "[CreateSPIRVModule] Processing: " << gpuModule.getName() << "\n";
        llvm::outs().flush();

        // Check if spirv.module already exists
        bool hasSpirvModule = false;
        gpuModule.walk([&](mlir::spirv::ModuleOp op) {
            hasSpirvModule = true;
        });

        if (hasSpirvModule) {
            llvm::outs() << "[CreateSPIRVModule] spirv.module already exists\n";
            return;
        }

        llvm::outs() << "[CreateSPIRVModule] Collecting spirv ops...\n";
        llvm::outs().flush();

        // First, collect all spirv ops we need to move
        llvm::SmallVector<mlir::Operation*> opsToMove;
        for (auto &op : *gpuModule.getBody()) {
            llvm::outs() << "[CreateSPIRVModule] Checking op: " << op.getName() << "\n";
            llvm::outs().flush();
            if (op.getDialect() &&
                op.getDialect()->getNamespace() == "spirv") {
                opsToMove.push_back(&op);
            }
        }

        if (opsToMove.empty()) {
            llvm::outs() << "[CreateSPIRVModule] No spirv ops found\n";
            return;
        }

        llvm::outs() << "[CreateSPIRVModule] Found " << opsToMove.size() << " spirv op(s)\n";
        llvm::outs().flush();

        // Create target environment with required extensions
        llvm::outs() << "[CreateSPIRVModule] Creating target env...\n";
        llvm::outs().flush();

        auto triple = mlir::spirv::VerCapExtAttr::get(
            mlir::spirv::Version::V_1_0,
            {mlir::spirv::Capability::Shader},
            {mlir::spirv::Extension::SPV_KHR_storage_buffer_storage_class},
            ctx);

        llvm::outs() << "[CreateSPIRVModule] Triple created\n";
        llvm::outs().flush();

        auto targetEnvAttr = mlir::spirv::TargetEnvAttr::get(
            triple,
            mlir::spirv::getDefaultResourceLimits(ctx),
            mlir::spirv::ClientAPI::Vulkan,
            mlir::spirv::Vendor::Unknown,
            mlir::spirv::DeviceType::Unknown,
            0);

        llvm::outs() << "[CreateSPIRVModule] Creating spirv.module...\n";
        llvm::outs().flush();

        // Create spirv.module at the END of gpu.module (after collecting ops)
        mlir::OpBuilder builder(ctx);
        // Insert before terminator (gpu.module_end)
        builder.setInsertionPoint(gpuModule.getBody()->getTerminator());

        auto spirvModule = builder.create<mlir::spirv::ModuleOp>(
            gpuModule.getLoc(),
            mlir::spirv::AddressingModel::Logical,
            mlir::spirv::MemoryModel::GLSL450);

        llvm::outs() << "[CreateSPIRVModule] spirv.module created, setting attrs...\n";
        llvm::outs().flush();

        // Set required attributes for serialization
        spirvModule->setAttr(mlir::spirv::getTargetEnvAttrName(), targetEnvAttr);
        spirvModule->setAttr("vce_triple", triple);  // Required for serialization

        llvm::outs() << "[CreateSPIRVModule] Moving ops...\n";
        llvm::outs().flush();

        // Now clone collected ops into spirv.module (moving can cause issues)
        mlir::Block *spirvBody = spirvModule.getBody();

        if (!spirvBody) {
            llvm::outs() << "[CreateSPIRVModule] ERROR: spirvBody is null!\n";
            return;
        }

        mlir::OpBuilder innerBuilder(ctx);
        innerBuilder.setInsertionPointToStart(spirvBody);

        for (auto *op : opsToMove) {
            llvm::outs() << "[CreateSPIRVModule] Cloning: " << op->getName() << "\n";
            llvm::outs().flush();

            // Clone the operation
            mlir::Operation *cloned = innerBuilder.clone(*op);
            (void)cloned;
        }

        // Now erase original ops
        llvm::outs() << "[CreateSPIRVModule] Erasing original ops...\n";
        llvm::outs().flush();

        for (auto *op : opsToMove) {
            op->erase();
        }

        llvm::outs() << "[CreateSPIRVModule] Done! Created spirv.module with "
                     << opsToMove.size() << " ops\n";
        llvm::outs().flush();
    }
};

//===----------------------------------------------------------------------===//
// Pass: Convert index types to i32 for SPIR-V compatibility
//===----------------------------------------------------------------------===//
struct ConvertIndexToI32Pass
    : public mlir::PassWrapper<ConvertIndexToI32Pass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertIndexToI32Pass)

    void runOnOperation() override {
        auto gpuModule = getOperation();
        auto *ctx = &getContext();
        mlir::OpBuilder builder(ctx);

        llvm::errs() << "[ConvertIndexToI32] Processing GPU module\n";

        // Convert gpu.block_id results from index to i32
        gpuModule.walk([&](mlir::gpu::BlockIdOp op) {
            builder.setInsertionPointAfter(op);
            auto i32Type = builder.getI32Type();
            auto indexCast = builder.create<mlir::arith::IndexCastOp>(
                op.getLoc(), i32Type, op.getResult());

            // Replace all uses of the index result with i32 result
            // except the cast itself
            op.getResult().replaceAllUsesExcept(indexCast.getResult(), indexCast);
        });

        llvm::errs() << "[ConvertIndexToI32] Done\n";
    }
};

//===----------------------------------------------------------------------===//
// Pass: Comprehensive GPU to SPIRV conversion
// ...existing code...
// This pass converts gpu.module contents to SPIRV using all conversion patterns
//===----------------------------------------------------------------------===//
struct GPUToSPIRVConversionPass
    : public mlir::PassWrapper<GPUToSPIRVConversionPass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUToSPIRVConversionPass)

    void runOnOperation() override {
        auto gpuModule = getOperation();
        auto *ctx = &getContext();
        mlir::OpBuilder builder(ctx);

        llvm::errs() << "[GPUToSPIRV] Converting GPU module: " << gpuModule.getName() << "\n";

        // Create SPIR-V target environment (Vulkan 1.0 + basic capabilities)
        auto triple = mlir::spirv::VerCapExtAttr::get(
            mlir::spirv::Version::V_1_0,
            {mlir::spirv::Capability::Shader, mlir::spirv::Capability::Addresses},
            {},
            ctx);
        auto targetAttr = mlir::spirv::TargetEnvAttr::get(
            triple,
            mlir::spirv::getDefaultResourceLimits(ctx),
            mlir::spirv::ClientAPI::Vulkan,
            mlir::spirv::Vendor::Unknown,
            mlir::spirv::DeviceType::Unknown,
            /*deviceId=*/0);

        // Create type converter with target environment
        mlir::SPIRVTypeConverter typeConverter(targetAttr);

        // Setup conversion target - use ConversionTarget with SPIRV legal dialect
        mlir::ConversionTarget target(*ctx);
        target.addLegalDialect<mlir::spirv::SPIRVDialect>();
        target.addLegalOp<mlir::gpu::GPUModuleOp, mlir::gpu::ModuleEndOp>();

        // Collect all conversion patterns
        mlir::RewritePatternSet patterns(ctx);

        // GPU patterns (most important - converts gpu.func to spirv.func!)
        mlir::populateGPUToSPIRVPatterns(typeConverter, patterns);

        // Arith patterns (handles index -> i32 conversion!)
        mlir::arith::populateArithToSPIRVPatterns(typeConverter, patterns);

        // MemRef patterns (handles memref.load/store)
        mlir::populateMemRefToSPIRVPatterns(typeConverter, patterns);

        // Func patterns
        mlir::populateFuncToSPIRVPatterns(typeConverter, patterns);

        // SCF patterns (for any remaining loops) - needs ScfToSPIRVContext
        mlir::ScfToSPIRVContext scfContext;
        mlir::populateSCFToSPIRVPatterns(typeConverter, scfContext, patterns);

        // ControlFlow patterns
        mlir::cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);

        // Builtin patterns for standard operations
        mlir::populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);

        if (mlir::failed(mlir::applyPartialConversion(gpuModule, target, std::move(patterns)))) {
            llvm::errs() << "[GPUToSPIRV] Conversion failed!\n";
            signalPassFailure();
            return;
        }

        llvm::errs() << "[GPUToSPIRV] Conversion succeeded!\n";
    }
};

} // namespace

namespace tenzo {
namespace gpu {

void addLinalgToGPUPasses(mlir::OpPassManager &pm) {
    // PHASE 1: Linalg -> Parallel Loops (SCF)
    pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());

    // PHASE 2: Canonicalize before GPU mapping
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // PHASE 3: Annotate parallel loops with GPU mapping
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());

    // PHASE 4: Convert annotated scf.parallel -> gpu.launch
    pm.addPass(mlir::createParallelLoopToGpuPass());

    // PHASE 5: Outline GPU kernels into gpu.module
    pm.addPass(mlir::createGpuLauchSinkIndexComputationsPass());
    pm.addPass(mlir::createGpuKernelOutliningPass());

    // PHASE 6: Cleanup
    pm.addPass(mlir::createCanonicalizerPass());
}

void addGPUToSPIRVPasses(mlir::OpPassManager &pm) {
    // PHASE 1: Set SPIR-V target environment on GPU modules
    pm.addPass(std::make_unique<SetSPIRVTargetEnvPass>());

    // PHASE 2: Set SPIR-V ABI attributes on GPU kernels
    pm.addPass(std::make_unique<SetSPIRVABIPass>());

    // PHASE 3: Prepare for SPIR-V conversion
    pm.addPass(mlir::createLowerAffinePass());

    // PHASE 4: Map MemRef storage classes BEFORE conversion
    pm.addPass(mlir::createMapMemRefStorageClassPass());

    // PHASE 5: Use our comprehensive conversion pass that handles index types
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(std::make_unique<GPUToSPIRVConversionPass>());

    // PHASE 6: Create spirv.module wrapper for serialization
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(std::make_unique<CreateSPIRVModulePass>());

    // PHASE 7: SPIR-V post-processing
    auto &gpuModulePM = pm.nest<mlir::gpu::GPUModuleOp>();
    auto &spirvModulePM = gpuModulePM.nest<mlir::spirv::ModuleOp>();

    // Lower ABI attributes
    spirvModulePM.addPass(mlir::spirv::createSPIRVLowerABIAttributesPass());

    // Update VCE
    spirvModulePM.addPass(mlir::spirv::createSPIRVUpdateVCEPass());

    // PHASE 8: Cleanup
    pm.addPass(mlir::createCanonicalizerPass());
}

std::vector<uint32_t> serializeSPIRVModule(mlir::ModuleOp module) {
    std::vector<uint32_t> binary;

    llvm::outs() << "[Serialize] Looking for spirv.module...\n";

    // Find SPIR-V module inside the MLIR module (may be inside gpu.module)
    mlir::spirv::ModuleOp spirvModule = nullptr;

    // Walk through all ops to find spirv.module
    module.walk([&](mlir::spirv::ModuleOp op) {
        llvm::outs() << "[Serialize] Found spirv.module!\n";
        spirvModule = op;
    });

    // If not found, check if we have spirv ops inside gpu.module
    if (!spirvModule) {
        bool hasSpirvOps = false;
        module.walk([&](mlir::spirv::FuncOp op) {
            hasSpirvOps = true;
        });

        if (hasSpirvOps) {
            llvm::outs() << "Found spirv.func but no spirv.module wrapper\n";
            return binary;
        }

        llvm::outs() << "No SPIR-V module or ops found!\n";
        return binary;
    }

    llvm::outs() << "[Serialize] Serializing spirv.module...\n";

    // Serialize to binary
    llvm::SmallVector<uint32_t> spirvBinary;
    mlir::spirv::SerializationOptions options;
    options.emitSymbolName = true;
    options.emitDebugInfo = false;

    if (mlir::failed(mlir::spirv::serialize(spirvModule, spirvBinary, options))) {
        llvm::outs() << "Failed to serialize SPIR-V module!\n";
        return binary;
    }

    llvm::outs() << "[Serialize] Success! Size: " << spirvBinary.size() * 4 << " bytes\n";

    binary.assign(spirvBinary.begin(), spirvBinary.end());
    return binary;
}

} // namespace gpu
} // namespace tenzo

