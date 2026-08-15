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

struct InlineAllocaScopePattern : public OpRewritePattern<memref::AllocaScopeOp> {
    using OpRewritePattern<memref::AllocaScopeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(memref::AllocaScopeOp scopeOp, PatternRewriter &rewriter) const override {
        if (scopeOp.getRegion().empty()) return failure();
        Block *body = &scopeOp.getRegion().front();
        Operation *term = body->getTerminator();

        ValueRange results = term->getOperands();
        for (Operation &op : llvm::make_early_inc_range(body->without_terminator())) {
            op.moveBefore(scopeOp);
        }

        if (scopeOp.getNumResults() == 0) {
            rewriter.eraseOp(scopeOp);
        } else {
            rewriter.replaceOp(scopeOp, results);
        }
        return success();
    }
};

struct InlineAllocaScopePass : public PassWrapper<InlineAllocaScopePass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InlineAllocaScopePass)

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<InlineAllocaScopePattern>(&getContext());
        (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }
};

struct SetRestrictOnToTensorPattern : public OpRewritePattern<bufferization::ToTensorOp> {
    using OpRewritePattern<bufferization::ToTensorOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(bufferization::ToTensorOp op, PatternRewriter &rewriter) const override {
        if (op.getRestrict()) return failure();
        rewriter.modifyOpInPlace(op, [&]() {
            op.setRestrict(true);
        });
        return success();
    }
};

struct FixToTensorPass : public PassWrapper<FixToTensorPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FixToTensorPass)
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<SetRestrictOnToTensorPattern>(&getContext());
        (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }
};

struct EmitCInterfacePass : public PassWrapper<EmitCInterfacePass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitCInterfacePass)
    void runOnOperation() override {
        ModuleOp module = getOperation();
        if (auto mainFunc = module.lookupSymbol<func::FuncOp>("main")) {
            mainFunc->setAttr("llvm.emit_c_interface", UnitAttr::get(&getContext()));
        }
    }
};

} // namespace

void tenzo::addTenzoBufferizationPasses(mlir::OpPassManager &pm) {
    // 0. Set restrict attribute on bufferization.to_tensor ops
    pm.addNestedPass<func::FuncOp>(std::make_unique<FixToTensorPass>());
    pm.addPass(std::make_unique<EmitCInterfacePass>());

    // 1. Empty -> alloc_tensor
    pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
    
    // 2. OneShot Bufferization
    mlir::bufferization::OneShotBufferizePassOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.copyBeforeWrite = false;
    options.allowUnknownOps = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));

    // 3. Convert function return memrefs to out parameters AFTER bufferization
    pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass());

    // 4. Convert bufferization ops -> memref ops & lower remaining tensor.extract
    pm.addPass(mlir::createConvertBufferizationToMemRefPass());
    pm.addNestedPass<func::FuncOp>(std::make_unique<LowerTensorExtractPass>());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    
    // 5. Cleanup & Inline alloca_scope ops
    pm.addNestedPass<func::FuncOp>(std::make_unique<InlineAllocaScopePass>());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
}
