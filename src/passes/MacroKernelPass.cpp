// MacroKernelPass.cpp - GotoBLAS-style 5-loop GEMM implementation
//
// The 5-Loop Architecture:
//   Loop 1 (jc): Partition N for L3 cache (Nc = 128-256)
//   Loop 2 (pc): Partition K into panels (Kc = 256-384)
//     → PACK B: B[pc:pc+Kc, jc:jc+Nc] → B_packed[Kc*Nc]
//   Loop 3 (ic): Partition M for L2 cache (Mc = 64-128)
//     → PACK A: A[ic:ic+Mc, pc:pc+Kc] → A_packed[Mc*Kc]
//   Loop 4 (jr): Iterate over 16-wide micro-kernels
//   Loop 5 (ir): Iterate over 6-high micro-kernels
//     → MICRO-KERNEL: 123.7 GFLOPS per call!
//
// Packed Buffer Layouts:
//   B_packed: For each panel j (j=0,1,2,...):
//     B[0, j*16..j*16+15], B[1, j*16..j*16+15], ..., B[Kc-1, j*16..j*16+15]
//     Layout is sequential: Kc * 16 floats per panel
//
//   A_packed: For each panel i (i=0,1,2,...):
//     Column-major within panel: A[i*6+0..5, 0], A[i*6+0..5, 1], ..., A[i*6+0..5, Kc-1]
//     Layout: Kc * 6 floats per panel
//
// Goal: 60-80 GFLOPS on 512×512, scaling to 90+ on 1024×1024

#include "passes/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// Forward declaration
void generateInlineMicroKernel(OpBuilder &b, Location loc,
                               Value packedA, Value packedB, Value C,
                               int64_t kSize, const tenzo::MicroKernelParams &params);

//===----------------------------------------------------------------------===//
// Generate Macro-Kernel Function
//===----------------------------------------------------------------------===//
func::FuncOp generateMacroKernelFunction(OpBuilder &builder, Location loc, 
                                         MLIRContext *ctx, 
                                         const tenzo::MicroKernelParams &params) {
    auto f32Type = builder.getF32Type();
    auto indexType = builder.getIndexType();

    // Function signature: gemm_macro(A, B, C, M, N, K)
    auto aType = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);
    auto bType = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);
    auto cType = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);

    auto funcType = builder.getFunctionType(
        {aType, bType, cType, indexType, indexType, indexType}, {}
    );

    auto func = builder.create<func::FuncOp>(loc, "gemm_macro_kernel", funcType);
    func.setPublic();

    auto &entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    Value A = entryBlock.getArgument(0);
    Value B = entryBlock.getArgument(1);
    Value C = entryBlock.getArgument(2);
    Value M = entryBlock.getArgument(3);
    Value N = entryBlock.getArgument(4);
    Value K = entryBlock.getArgument(5);

    // Constants from params
    auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto cMR = builder.create<arith::ConstantIndexOp>(loc, params.MR);
    auto cNR = builder.create<arith::ConstantIndexOp>(loc, params.NR);
    auto cKC = builder.create<arith::ConstantIndexOp>(loc, params.KC);
    auto cMC = builder.create<arith::ConstantIndexOp>(loc, params.MC);
    auto cNC = builder.create<arith::ConstantIndexOp>(loc, params.NC);

    // Allocate packed buffers
    auto blockAType = MemRefType::get({params.MC * params.KC}, f32Type);
    Value blockA = builder.create<memref::AllocOp>(loc, blockAType);

    auto blockBType = MemRefType::get({params.KC * params.NC}, f32Type);
    Value blockB = builder.create<memref::AllocOp>(loc, blockBType);

    llvm::outs() << "[MacroKernel] Generating macro-kernel with sizes: "
                 << "MC=" << params.MC << ", KC=" << params.KC << ", NC=" << params.NC 
                 << ", MR=" << params.MR << ", NR=" << params.NR << "\n";

    // ========== LOOP 1: jc (partition N for L3) ==========
    builder.create<scf::ForOp>(
        loc, c0, N, cNC, ValueRange{},
        [&](OpBuilder &b1, Location loc, Value jc, ValueRange) {
            auto ncRemaining = b1.create<arith::SubIOp>(loc, N, jc);
            auto ncActual = b1.create<arith::MinSIOp>(loc, ncRemaining, cNC);

            // ========== LOOP 2: pc (partition K into panels) ==========
            b1.create<scf::ForOp>(
                loc, c0, K, cKC, ValueRange{},
                [&](OpBuilder &b2, Location loc, Value pc, ValueRange) {
                    auto kcRemaining = b2.create<arith::SubIOp>(loc, K, pc);
                    auto kcActual = b2.create<arith::MinSIOp>(loc, kcRemaining, cKC);

                    // PACK B
                    auto numBPanels = b2.create<arith::DivSIOp>(loc, ncActual, cNR);
                    b2.create<scf::ForOp>(
                        loc, c0, numBPanels, c1, ValueRange{},
                        [&](OpBuilder &bPack, Location loc, Value bPanelIdx, ValueRange) {
                            auto jSrc = bPack.create<arith::MulIOp>(loc, bPanelIdx, cNR);
                            auto jStart = bPack.create<arith::AddIOp>(loc, jc, jSrc);
                            auto kcTimesNR = bPack.create<arith::MulIOp>(loc, cKC, cNR);
                            auto bDstBase = bPack.create<arith::MulIOp>(loc, bPanelIdx, kcTimesNR);

                            bPack.create<scf::ForOp>(
                                loc, c0, kcActual, c1, ValueRange{},
                                [&](OpBuilder &bPackInner, Location loc, Value kLocal, ValueRange) {
                                    auto kSrc = bPackInner.create<arith::AddIOp>(loc, pc, kLocal);
                                    auto kTimesNR = bPackInner.create<arith::MulIOp>(loc, kLocal, cNR);
                                    auto bDstOffset = bPackInner.create<arith::AddIOp>(loc, bDstBase, kTimesNR);

                                    for (int64_t j = 0; j < params.NR; j++) {
                                        auto cj = bPackInner.create<arith::ConstantIndexOp>(loc, j);
                                        auto jSrcCol = bPackInner.create<arith::AddIOp>(loc, jStart, cj);
                                        auto val = bPackInner.create<memref::LoadOp>(loc, B, ValueRange{kSrc, jSrcCol});
                                        auto bDstIdx = bPackInner.create<arith::AddIOp>(loc, bDstOffset, cj);
                                        bPackInner.create<memref::StoreOp>(loc, val, blockB, ValueRange{bDstIdx});
                                    }
                                    bPackInner.create<scf::YieldOp>(loc);
                                });
                            bPack.create<scf::YieldOp>(loc);
                        });

                    // ========== LOOP 3: ic (partition M for L2) ==========
                    b2.create<scf::ForOp>(
                        loc, c0, M, cMC, ValueRange{},
                        [&](OpBuilder &b3, Location loc, Value ic, ValueRange) {
                            auto mcRemaining = b3.create<arith::SubIOp>(loc, M, ic);
                            auto mcActual = b3.create<arith::MinSIOp>(loc, mcRemaining, cMC);

                            // PACK A
                            auto numAPanels = b3.create<arith::DivSIOp>(loc, mcActual, cMR);
                            b3.create<scf::ForOp>(
                                loc, c0, numAPanels, c1, ValueRange{},
                                [&](OpBuilder &aPack, Location loc, Value aPanelIdx, ValueRange) {
                                    auto iSrc = aPack.create<arith::MulIOp>(loc, aPanelIdx, cMR);
                                    auto iStart = aPack.create<arith::AddIOp>(loc, ic, iSrc);
                                    auto kcTimesMR = aPack.create<arith::MulIOp>(loc, cKC, cMR);
                                    auto aDstBase = aPack.create<arith::MulIOp>(loc, aPanelIdx, kcTimesMR);

                                    aPack.create<scf::ForOp>(
                                        loc, c0, kcActual, c1, ValueRange{},
                                        [&](OpBuilder &aPackK, Location loc, Value kLocal, ValueRange) {
                                            auto kSrc = aPackK.create<arith::AddIOp>(loc, pc, kLocal);
                                            auto kTimesMR = aPackK.create<arith::MulIOp>(loc, kLocal, cMR);
                                            auto aDstOffset = aPackK.create<arith::AddIOp>(loc, aDstBase, kTimesMR);

                                            for (int64_t i = 0; i < params.MR; i++) {
                                                auto ci = aPackK.create<arith::ConstantIndexOp>(loc, i);
                                                auto iSrcRow = aPackK.create<arith::AddIOp>(loc, iStart, ci);
                                                auto val = aPackK.create<memref::LoadOp>(loc, A, ValueRange{iSrcRow, kSrc});
                                                auto aDstIdx = aPackK.create<arith::AddIOp>(loc, aDstOffset, ci);
                                                aPackK.create<memref::StoreOp>(loc, val, blockA, ValueRange{aDstIdx});
                                            }
                                            aPackK.create<scf::YieldOp>(loc);
                                        });
                                    aPack.create<scf::YieldOp>(loc);
                                });

                            // LOOP 4 & 5: Micro-kernel
                            auto numMicroJ = b3.create<arith::DivSIOp>(loc, ncActual, cNR);
                            auto numMicroI = b3.create<arith::DivSIOp>(loc, mcActual, cMR);

                            b3.create<scf::ForOp>(
                                loc, c0, numMicroJ, c1, ValueRange{},
                                [&](OpBuilder &b4, Location loc, Value jr, ValueRange) {
                                    b4.create<scf::ForOp>(
                                        loc, c0, numMicroI, c1, ValueRange{},
                                        [&](OpBuilder &b5, Location loc, Value ir, ValueRange) {
                                            auto aOffset = b5.create<arith::MulIOp>(loc, ir, 
                                                b5.create<arith::MulIOp>(loc, cKC, cMR));
                                            auto bOffset = b5.create<arith::MulIOp>(loc, jr, 
                                                b5.create<arith::MulIOp>(loc, cKC, cNR));

                                            auto irTimesMR = b5.create<arith::MulIOp>(loc, ir, cMR);
                                            auto cRowStart = b5.create<arith::AddIOp>(loc, ic, irTimesMR);
                                            auto jrTimesNR = b5.create<arith::MulIOp>(loc, jr, cNR);
                                            auto cColStart = b5.create<arith::AddIOp>(loc, jc, jrTimesNR);

                                            // Subviews
                                            SmallVector<OpFoldResult> aOff = {OpFoldResult(aOffset)};
                                            SmallVector<OpFoldResult> aSz = {b5.getIndexAttr(params.KC * params.MR)};
                                            SmallVector<OpFoldResult> aStride = {b5.getIndexAttr(1)};
                                            auto aSubview = b5.create<memref::SubViewOp>(loc, blockA, aOff, aSz, aStride);

                                            SmallVector<OpFoldResult> bOff = {OpFoldResult(bOffset)};
                                            SmallVector<OpFoldResult> bSz = {b5.getIndexAttr(params.KC * params.NR)};
                                            SmallVector<OpFoldResult> bStride = {b5.getIndexAttr(1)};
                                            auto bSubview = b5.create<memref::SubViewOp>(loc, blockB, bOff, bSz, bStride);

                                            SmallVector<OpFoldResult> cOff = {OpFoldResult(cRowStart), OpFoldResult(cColStart)};
                                            SmallVector<OpFoldResult> cSz = {b5.getIndexAttr(params.MR), b5.getIndexAttr(params.NR)};
                                            SmallVector<OpFoldResult> cStride = {b5.getIndexAttr(1), b5.getIndexAttr(1)};
                                            auto cSubview = b5.create<memref::SubViewOp>(loc, C, cOff, cSz, cStride);

                                            auto aCast = b5.create<memref::CastOp>(loc, MemRefType::get({params.KC * params.MR}, f32Type), aSubview);
                                            auto bCast = b5.create<memref::CastOp>(loc, MemRefType::get({params.KC * params.NR}, f32Type), bSubview);
                                            auto cCast = b5.create<memref::CastOp>(loc, MemRefType::get({params.MR, params.NR}, f32Type), cSubview);

                                            generateInlineMicroKernel(b5, loc, aCast, bCast, cCast, params.KC, params);
                                            b5.create<scf::YieldOp>(loc);
                                        });
                                    b4.create<scf::YieldOp>(loc);
                                });
                            b3.create<scf::YieldOp>(loc);
                        });
                    b2.create<scf::YieldOp>(loc);
                });
            b1.create<scf::YieldOp>(loc);
        });

    builder.create<memref::DeallocOp>(loc, blockA);
    builder.create<memref::DeallocOp>(loc, blockB);
    builder.create<func::ReturnOp>(loc);
    return func;
}

void generateInlineMicroKernel(OpBuilder &b, Location loc,
                               Value packedA, Value packedB, Value C,
                               int64_t kSize, const tenzo::MicroKernelParams &params) {
    auto f32Type = b.getF32Type();
    auto vecType = VectorType::get({params.VEC_SIZE}, f32Type);
    auto zeroF32 = b.create<arith::ConstantOp>(loc, f32Type, b.getF32FloatAttr(0.0f));

    auto c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    auto cK = b.create<arith::ConstantIndexOp>(loc, kSize);
    auto cMR = b.create<arith::ConstantIndexOp>(loc, params.MR);
    auto cNR = b.create<arith::ConstantIndexOp>(loc, params.NR);

    int64_t numVecsPerNR = params.NR / params.VEC_SIZE;

    // Load initial C
    SmallVector<Value> accums;
    for (int64_t i = 0; i < params.MR; i++) {
        auto iIdx = b.create<arith::ConstantIndexOp>(loc, i);
        for (int64_t j = 0; j < numVecsPerNR; j++) {
            auto jOffset = b.create<arith::ConstantIndexOp>(loc, j * params.VEC_SIZE);
            auto cVec = b.create<vector::TransferReadOp>(loc, vecType, C, ValueRange{iIdx, jOffset}, zeroF32);
            accums.push_back(cVec);
        }
    }

    auto forOp = b.create<scf::ForOp>(
        loc, c0, cK, b.create<arith::ConstantIndexOp>(loc, 1), accums,
        [&](OpBuilder &kb, Location loc, Value k, ValueRange iterArgs) {
            SmallVector<Value> newAccums(iterArgs.begin(), iterArgs.end());
            auto kTimesMR = kb.create<arith::MulIOp>(loc, k, cMR);
            auto kTimesNR = kb.create<arith::MulIOp>(loc, k, cNR);

            SmallVector<Value> bVecs;
            for (int64_t j = 0; j < numVecsPerNR; j++) {
                auto jOffset = kb.create<arith::ConstantIndexOp>(loc, j * params.VEC_SIZE);
                auto bIdx = kb.create<arith::AddIOp>(loc, kTimesNR, jOffset);
                auto bv = kb.create<vector::TransferReadOp>(loc, vecType, packedB, ValueRange{bIdx}, zeroF32);
                bVecs.push_back(bv);
            }

            for (int64_t i = 0; i < params.MR; i++) {
                auto ci = kb.create<arith::ConstantIndexOp>(loc, i);
                auto aIdx = kb.create<arith::AddIOp>(loc, kTimesMR, ci);
                auto aScalar = kb.create<memref::LoadOp>(loc, packedA, ValueRange{aIdx});
                auto aVec = kb.create<vector::BroadcastOp>(loc, vecType, aScalar);

                for (int64_t j = 0; j < numVecsPerNR; j++) {
                    newAccums[i * numVecsPerNR + j] = kb.create<vector::FMAOp>(
                        loc, aVec, bVecs[j], newAccums[i * numVecsPerNR + j]);
                }
            }
            kb.create<scf::YieldOp>(loc, newAccums);
        });

    auto results = forOp.getResults();
    for (int64_t i = 0; i < params.MR; i++) {
        auto iIdx = b.create<arith::ConstantIndexOp>(loc, i);
        for (int64_t j = 0; j < numVecsPerNR; j++) {
            auto jOffset = b.create<arith::ConstantIndexOp>(loc, j * params.VEC_SIZE);
            b.create<vector::TransferWriteOp>(loc, results[i * numVecsPerNR + j], C, ValueRange{iIdx, jOffset});
        }
    }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
struct GenerateMacroKernelPass
    : public PassWrapper<GenerateMacroKernelPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateMacroKernelPass)

    tenzo::MicroKernelParams params;
    GenerateMacroKernelPass(const tenzo::MicroKernelParams &p) : params(p) {}

    void runOnOperation() override {
        auto module = getOperation();
        OpBuilder builder(&getContext());
        builder.setInsertionPointToStart(module.getBody());
        auto macroFunc = generateMacroKernelFunction(builder, module.getLoc(), &getContext(), params);
        module.push_back(macroFunc);
    }
};

} // namespace

namespace tenzo {

std::unique_ptr<mlir::Pass> createGenerateMacroKernelPass(const MicroKernelParams &params) {
    return std::make_unique<GenerateMacroKernelPass>(params);
}

} // namespace tenzo

