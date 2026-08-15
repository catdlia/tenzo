// PackingKernels.cpp - Memory layout transformation for optimal micro-kernel performance
//
// GotoBLAS-style matrix packing with PADDING support for arbitrary sizes:
// - Pack B: B[K][N] (row-major) → B_packed[K*NR] per panel (sequential for micro-kernel)
// - Pack A: A[M][K] (row-major) → A_packed[K*MR] per panel (sequential for broadcast)
//
// PADDING STRATEGY (for arbitrary matrix sizes):
// - M not divisible by MR? → Pad A with zeros to next multiple of MR
// - N not divisible by NR? → Pad B with zeros to next multiple of NR
// - K can be any size (no padding needed for K dimension)
//
// Memory Layout for Packed B (CRITICAL for performance):
//   Original B[K][N]: row-major, stride = N
//   Packed B: For panel j (columns j*NR to j*NR+NR):
//     Layout: B[0,j*NR+0], B[0,j*NR+1], ..., B[0,j*NR+15],  <- row 0
//             B[1,j*NR+0], B[1,j*NR+1], ..., B[1,j*NR+15],  <- row 1
//             ...
//             B[K-1,j*NR+0], ..., B[K-1,j*NR+15]            <- row K-1
//
// This allows micro-kernel to:
// 1. Load B[k, 0:16] as TWO sequential AVX2 vectors (16 floats)
// 2. Zero strided access = 100% cache efficiency!
//
// Performance target:
// - Packing bandwidth: 30-50 GB/s on DDR4 (memory bound)
// - NOT compute bound (vectorized copying with AVX2)

#include "passes/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

constexpr int64_t MR = 6;   // Micro-kernel height
constexpr int64_t NR = 16;  // Micro-kernel width
constexpr int64_t VEC_SIZE = 8;  // AVX2 vector size

//===----------------------------------------------------------------------===//
// Generate Vectorized Packing for Matrix B with PADDING support
//===----------------------------------------------------------------------===//
// Transform B[K][N] (row-major) → B_packed (panel format for NR=16)
// Uses vector::TransferReadOp with padding_value for automatic OOB handling
//
func::FuncOp generatePackBFunction(OpBuilder &builder, Location loc, MLIRContext *ctx, AffineMap routingMap = {}) {
    auto f32Type = builder.getF32Type();
    auto indexType = builder.getIndexType();

    auto srcType = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);
    auto dstType = MemRefType::get({ShapedType::kDynamic}, f32Type);

    auto funcType = builder.getFunctionType(
        {srcType, dstType, indexType, indexType}, {}
    );

    auto func = builder.create<func::FuncOp>(loc, "pack_matrix_B", funcType);
    func.setPublic();

    auto &entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    Value src = entryBlock.getArgument(0);
    Value dst = entryBlock.getArgument(1);
    Value K = entryBlock.getArgument(2);
    Value N = entryBlock.getArgument(3);

    // Constants
    auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto cNR = builder.create<arith::ConstantIndexOp>(loc, NR);
    auto cNRminus1 = builder.create<arith::ConstantIndexOp>(loc, NR - 1);
    auto cVecSize = builder.create<arith::ConstantIndexOp>(loc, VEC_SIZE);
    auto zeroF32 = builder.create<arith::ConstantOp>(loc, f32Type, builder.getF32FloatAttr(0.0f));

    auto vecType = VectorType::get({VEC_SIZE}, f32Type);

    // Calculate number of panels with ceiling division: (N + NR - 1) / NR
    auto nPlusNRm1 = builder.create<arith::AddIOp>(loc, N, cNRminus1);
    auto numPanels = builder.create<arith::DivUIOp>(loc, nPlusNRm1, cNR);

    // Outer loop: panels
    builder.create<scf::ForOp>(
        loc, c0, numPanels, c1, ValueRange{},
        [&](OpBuilder &b, Location loc, Value panelIdx, ValueRange) {
            Value jStart = b.create<arith::MulIOp>(loc, panelIdx, cNR);
            Value KtimesNR = b.create<arith::MulIOp>(loc, K, cNR);
            Value dstBase = b.create<arith::MulIOp>(loc, panelIdx, KtimesNR);

            // Inner loop: K dimension
            b.create<scf::ForOp>(
                loc, c0, K, c1, ValueRange{},
                [&](OpBuilder &b, Location loc, Value k, ValueRange) {
                    Value kTimesNR = b.create<arith::MulIOp>(loc, k, cNR);
                    Value dstOffset = b.create<arith::AddIOp>(loc, dstBase, kTimesNR);

                    // Use transfer_read with padding and optional routing map for Implicit Shuffle
                    auto map = routingMap ? routingMap : b.getMultiDimIdentityMap(2);
                    Value srcVec0 = b.create<vector::TransferReadOp>(
                        loc, vecType, src, ValueRange{k, jStart}, zeroF32, map);

                    Value jStartPlus8 = b.create<arith::AddIOp>(loc, jStart, cVecSize);
                    Value srcVec1 = b.create<vector::TransferReadOp>(
                        loc, vecType, src, ValueRange{k, jStartPlus8}, zeroF32, map);

                    // Store to packed buffer
                    b.create<vector::TransferWriteOp>(loc, srcVec0, dst, ValueRange{dstOffset});
                    Value dstOffsetPlus8 = b.create<arith::AddIOp>(loc, dstOffset, cVecSize);
                    b.create<vector::TransferWriteOp>(loc, srcVec1, dst, ValueRange{dstOffsetPlus8});

                    b.create<scf::YieldOp>(loc);
                });

            b.create<scf::YieldOp>(loc);
        });

    builder.create<func::ReturnOp>(loc);
    return func;
}

//===----------------------------------------------------------------------===//
// Generate Vectorized Packing for Matrix A with PADDING support
//===----------------------------------------------------------------------===//
// Transform A[M][K] (row-major) → A_packed (panel format for MR=6)
// Uses arith::SelectOp for conditional padding (no complex scf::IfOp)
//
func::FuncOp generatePackAFunction(OpBuilder &builder, Location loc, MLIRContext *ctx, AffineMap routingMap = {}) {
    auto f32Type = builder.getF32Type();
    auto indexType = builder.getIndexType();

    auto srcType = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);
    auto dstType = MemRefType::get({ShapedType::kDynamic}, f32Type);

    auto funcType = builder.getFunctionType(
        {srcType, dstType, indexType, indexType}, {}
    );

    auto func = builder.create<func::FuncOp>(loc, "pack_matrix_A", funcType);
    func.setPublic();

    auto &entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    Value src = entryBlock.getArgument(0);
    Value dst = entryBlock.getArgument(1);
    Value M = entryBlock.getArgument(2);
    Value K = entryBlock.getArgument(3);

    // Constants
    auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto cMR = builder.create<arith::ConstantIndexOp>(loc, MR);
    auto cMRminus1 = builder.create<arith::ConstantIndexOp>(loc, MR - 1);
    auto zeroF32 = builder.create<arith::ConstantOp>(loc, f32Type, builder.getF32FloatAttr(0.0f));

    // Calculate number of panels with ceiling division: (M + MR - 1) / MR
    auto mPlusMRm1 = builder.create<arith::AddIOp>(loc, M, cMRminus1);
    auto numPanels = builder.create<arith::DivUIOp>(loc, mPlusMRm1, cMR);

    // Outer loop: panels
    builder.create<scf::ForOp>(
        loc, c0, numPanels, c1, ValueRange{},
        [&](OpBuilder &b, Location loc, Value panelIdx, ValueRange) {
            Value iStart = b.create<arith::MulIOp>(loc, panelIdx, cMR);
            Value KtimesMR = b.create<arith::MulIOp>(loc, K, cMR);
            Value dstBase = b.create<arith::MulIOp>(loc, panelIdx, KtimesMR);

            // Middle loop: K dimension
            b.create<scf::ForOp>(
                loc, c0, K, c1, ValueRange{},
                [&](OpBuilder &b, Location loc, Value k, ValueRange) {
                    Value kTimesMR = b.create<arith::MulIOp>(loc, k, cMR);
                    Value dstOffset = b.create<arith::AddIOp>(loc, dstBase, kTimesMR);

                    // Copy MR=6 elements with bounds checking using SelectOp
                    for (int64_t i = 0; i < MR; i++) {
                        auto ci = b.create<arith::ConstantIndexOp>(loc, i);
                        Value srcRow = b.create<arith::AddIOp>(loc, iStart, ci);
                        Value dstIdx = b.create<arith::AddIOp>(loc, dstOffset, ci);

                        // Check if row is valid (srcRow < M)
                        Value isValidRow = b.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::ult, srcRow, M);

                        // Clamp srcRow to valid range for safe load: min(srcRow, M-1)
                        Value maxValidRow = b.create<arith::SubIOp>(loc, M, c1);
                        Value clampedRow = b.create<arith::MinUIOp>(loc, srcRow, maxValidRow);

                        // Load from clamped index (always safe)
                        Value loadedVal = b.create<memref::LoadOp>(
                            loc, src, ValueRange{clampedRow, k});

                        // Select between loaded value and zero based on validity
                        Value resultVal = b.create<arith::SelectOp>(
                            loc, isValidRow, loadedVal, zeroF32.getResult());

                        // Store to dst
                        b.create<memref::StoreOp>(loc, resultVal, dst, ValueRange{dstIdx});
                    }

                    b.create<scf::YieldOp>(loc);
                });

            b.create<scf::YieldOp>(loc);
        });

    builder.create<func::ReturnOp>(loc);
    return func;
}

//===----------------------------------------------------------------------===//
// Pass: Generate Packing Functions
//===----------------------------------------------------------------------===//
struct GeneratePackingKernelsPass
    : public PassWrapper<GeneratePackingKernelsPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GeneratePackingKernelsPass)

    void runOnOperation() override {
        auto module = getOperation();
        auto *ctx = &getContext();

        llvm::outs() << "[PackingKernels] ====== Generating Packing Functions ======\n";
        llvm::outs() << "[PackingKernels] Target: Vectorized memory layout transformation\n";
        llvm::outs() << "[PackingKernels] MR = " << MR << ", NR = " << NR << "\n\n";

        OpBuilder builder(ctx);
        builder.setInsertionPointToStart(module.getBody());

        // Generate pack_matrix_B function
        llvm::outs() << "[PackingKernels] Generating pack_matrix_B...\n";
        auto packBFunc = generatePackBFunction(builder, module.getLoc(), ctx);
        module.push_back(packBFunc);
        llvm::outs() << "[PackingKernels] ✅ pack_matrix_B created\n";

        // Generate pack_matrix_A function
        llvm::outs() << "[PackingKernels] Generating pack_matrix_A...\n";
        auto packAFunc = generatePackAFunction(builder, module.getLoc(), ctx);
        module.push_back(packAFunc);
        llvm::outs() << "[PackingKernels] ✅ pack_matrix_A created\n";

        llvm::outs() << "[PackingKernels] ==========================================\n";
        llvm::outs() << "[PackingKernels] SUCCESS: Generated 2 packing functions with PADDING support\n";
        llvm::outs() << "[PackingKernels] - pack_matrix_B: B[K][N] → B_packed (NR=16 panels, zero-padded)\n";
        llvm::outs() << "[PackingKernels] - pack_matrix_A: A[M][K] → A_packed (MR=6 panels, zero-padded)\n";
        llvm::outs() << "[PackingKernels] Now supports ARBITRARY matrix sizes (not just multiples of MR/NR)!\n";
        llvm::outs() << "[PackingKernels] ==========================================\n";
    }
};

} // namespace

namespace tenzo {

std::unique_ptr<mlir::Pass> createGeneratePackingKernelsPass() {
    return std::make_unique<GeneratePackingKernelsPass>();
}

} // namespace tenzo

