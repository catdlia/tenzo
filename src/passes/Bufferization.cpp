#include "passes/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Bufferizable interface implementations
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"

using namespace mlir;

namespace {
struct TensorExtractToMemRefLoad : public OpRewritePattern<tensor::ExtractOp> {
    using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tensor::ExtractOp op, PatternRewriter &rewriter) const override {
        Value tensor = op.getTensor();
        if (auto cast = tensor.getDefiningOp<UnrealizedConversionCastOp>()) {
            if (cast.getInputs().size() == 1 && mlir::isa<MemRefType>(cast.getInputs()[0].getType())) {
                rewriter.replaceOpWithNewOp<memref::LoadOp>(op, cast.getInputs()[0], op.getIndices());
                return success();
            }
        }
        if (auto toTensor = tensor.getDefiningOp<bufferization::ToTensorOp>()) {
            rewriter.replaceOpWithNewOp<memref::LoadOp>(op, toTensor.getOperand(), op.getIndices());
            return success();
        }
        return failure();
    }
};

struct LowerTensorExtractPass : public PassWrapper<LowerTensorExtractPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTensorExtractPass)
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<TensorExtractToMemRefLoad>(&getContext());
        (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }
};
} // namespace

void tenzo::addTenzoBufferizationPasses(mlir::OpPassManager &pm) {
    // 1. Empty -> alloc_tensor
    pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
    
    // 2. OneShot Bufferization
    mlir::bufferization::OneShotBufferizePassOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.copyBeforeWrite = false;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));

    // 3. Convert function return memrefs to out parameters AFTER bufferization
    pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass());

    // 4. Convert bufferization ops -> memref ops & lower remaining tensor.extract
    pm.addPass(mlir::createConvertBufferizationToMemRefPass());
    pm.addNestedPass<func::FuncOp>(std::make_unique<LowerTensorExtractPass>());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    
    // 5. Cleanup
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
}
