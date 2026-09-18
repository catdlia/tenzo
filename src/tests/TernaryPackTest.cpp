// TernaryPackTest.cpp - Deep verification of tenzo.ternary_pack and AVX2 fused micro-kernel
#include "tests/TernaryPackTest.h"
#include "dialect/TenzoDialect.h"
#include "passes/Passes.h"
#include "context/HardwareProfile.h"
#include "runtime/MemRefUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"

#include <immintrin.h>
#include <cassert>
#include <vector>
#include <iostream>
#include <cstdint>

namespace tenzo {

// ============================================================================
// AVX2 reference function for fused 1.58-bit ternary packing (4 rows x 16 cols)
// Follows the exact 7-step sequence specified in R3:
// 1. Multiply accumulators by scale
// 2. Convert to 32-bit ints (_mm256_cvtps_epi32)
// 3. Shift values by +1 (domain {-1, 0, 1} -> {0, 1, 2})
// 4. Narrowing to 16-bit and 8-bit (_mm256_packs_epi32, _mm256_packs_epi16)
// 5. Cross-lane permute (_mm256_permute4x64_epi64)
// 6. Final packing of 4 bytes into 1 (_mm_maddubs_epi16, _mm_madd_epi16, _mm_shuffle_epi8)
// 7. Write final 32-bit blocks (4 bytes per row) to uint8_t output memref
// ============================================================================
static void avx2_fused_ternary_pack_4x16(const float* input_4x16, float scale, uint8_t* output_4x4) {
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256i v_one = _mm256_set1_epi32(1);

    // Multipliers for maddubs: [1, 4, 1, 4, ...]
    __m128i mult1 = _mm_setr_epi8(1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4);
    // Multipliers for madd: [1, 16, 1, 16, ...]
    __m128i mult2 = _mm_setr_epi16(1, 16, 1, 16, 1, 16, 1, 16);
    // Shuffle mask to collect dwords 0, 2, 1, 3 (groups 0, 1, 2, 3) into lowest 32 bits
    __m128i shuf_mask = _mm_setr_epi8(0, 8, 4, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    for (int row = 0; row < 4; row++) {
        // Load 16 floats (2 YMM accumulators)
        __m256 acc_lo = _mm256_loadu_ps(input_4x16 + row * 16 + 0);
        __m256 acc_hi = _mm256_loadu_ps(input_4x16 + row * 16 + 8);

        // Step 1: Multiply accumulators by scale
        __m256 scaled_lo = _mm256_mul_ps(acc_lo, v_scale);
        __m256 scaled_hi = _mm256_mul_ps(acc_hi, v_scale);

        // Step 2: Convert to 32-bit ints (_mm256_cvtps_epi32)
        __m256i i32_lo = _mm256_cvtps_epi32(scaled_lo);
        __m256i i32_hi = _mm256_cvtps_epi32(scaled_hi);

        // Step 3: Shift values by +1 (domain {-1, 0, 1} -> {0, 1, 2})
        __m256i sh_lo = _mm256_add_epi32(i32_lo, v_one);
        __m256i sh_hi = _mm256_add_epi32(i32_hi, v_one);

        // Step 4: Narrowing to 16-bit and 8-bit (_mm256_packs_epi32, _mm256_packs_epi16)
        __m256i pk16 = _mm256_packs_epi32(sh_lo, sh_hi); // 16x i16 in-lane
        __m256i pk8 = _mm256_packs_epi16(pk16, pk16);    // 32x i8 in-lane

        // Step 5: Cross-lane permute (_mm256_permute4x64_epi64)
        __m256i perm = _mm256_permute4x64_epi64(pk8, 0xD8);

        // Step 6: Final packing of 4 bytes into 1 (_mm_maddubs_epi16, _mm_madd_epi16, _mm_shuffle_epi8)
        __m128i r128 = _mm256_castsi256_si128(perm);
        __m128i pair = _mm_maddubs_epi16(r128, mult1);
        __m128i dwords = _mm_madd_epi16(pair, mult2);
        __m128i gathered = _mm_shuffle_epi8(dwords, shuf_mask);

        // Step 7: Write final 32-bit block (4 bytes per row) to uint8_t output memref
        uint32_t packed32 = _mm_cvtsi128_si32(gathered);
        *(reinterpret_cast<uint32_t*>(output_4x4 + row * 4)) = packed32;
    }
}

void runTernaryPackTest(mlir::MLIRContext &context) {
    llvm::outs() << "\n╔════════════════════════════════════════════════════════╗\n";
    llvm::outs() << "║  🧪 TERNARY PACK & AVX2 FUSED MICRO-KERNEL TESTS       ║\n";
    llvm::outs() << "╚════════════════════════════════════════════════════════╝\n\n";

    // ------------------------------------------------------------------------
    // TEST 1: TableGen Dialect Definition Verification (R1) & Textual Parsing
    // ------------------------------------------------------------------------
    llvm::outs() << "[Test 1/9] Verifying tenzo.ternary_pack Dialect Definition & Parser...\n";
    {
        mlir::OpBuilder builder(&context);
        auto loc = builder.getUnknownLoc();
        auto f32Type = builder.getF32Type();
        auto i8Type = builder.getI8Type();

        auto inTensorType = mlir::RankedTensorType::get({4, 16}, f32Type);
        auto outTensorType = mlir::RankedTensorType::get({4, 4}, i8Type);

        auto module = mlir::ModuleOp::create(loc);
        builder.setInsertionPointToStart(module.getBody());

        // 1a. C++ API with scale
        auto funcType = builder.getFunctionType({inTensorType, f32Type}, {outTensorType});
        auto func = builder.create<mlir::func::FuncOp>(loc, "test_pack_with_scale", funcType);
        auto &entryBlock = *func.addEntryBlock();
        builder.setInsertionPointToStart(&entryBlock);

        mlir::Value inVal = entryBlock.getArgument(0);
        mlir::Value scaleVal = entryBlock.getArgument(1);

        auto packOp = builder.create<tenzo::TernaryPackOp>(
            loc, outTensorType, inVal, scaleVal, builder.getI32IntegerAttr(4));

        assert(packOp && "Failed to create TernaryPackOp with scale!");
        assert(packOp.getValuesPerByte() == 4 && "values_per_byte default should be 4");
        assert(packOp.getInput() == inVal);
        assert(packOp.getScale() == scaleVal);
        assert(packOp.getResult().getType() == outTensorType);
        builder.create<mlir::func::ReturnOp>(loc, packOp.getResult());

        // 1b. C++ API without scale (optional scale omitted)
        builder.setInsertionPointToStart(module.getBody());
        auto funcTypeNoScale = builder.getFunctionType({inTensorType}, {outTensorType});
        auto funcNoScale = builder.create<mlir::func::FuncOp>(loc, "test_pack_no_scale", funcTypeNoScale);
        auto &entryBlockNoScale = *funcNoScale.addEntryBlock();
        builder.setInsertionPointToStart(&entryBlockNoScale);

        auto packOpNoScale = builder.create<tenzo::TernaryPackOp>(
            loc, outTensorType, entryBlockNoScale.getArgument(0), mlir::Value(), builder.getI32IntegerAttr(4));
        assert(packOpNoScale && "Failed to create TernaryPackOp without scale!");
        assert(!packOpNoScale.getScale() && "Scale should be null when omitted!");
        builder.create<mlir::func::ReturnOp>(loc, packOpNoScale.getResult());

        // Verification of MLIR module
        assert(mlir::succeeded(module.verify()) && "Module verification failed for TernaryPackOp!");

        // 1c. Textual round-trip parsing test
        const char *mlirText = R"mlir(
            module {
                func.func @parsed_pack_scale(%arg0: tensor<4x16xf32>, %arg1: f32) -> tensor<4x4xi8> {
                    %0 = tenzo.ternary_pack %arg0, %arg1 : (tensor<4x16xf32>, f32) -> tensor<4x4xi8>
                    return %0 : tensor<4x4xi8>
                }
                func.func @parsed_pack_noscale(%arg0: tensor<4x16xf32>) -> tensor<4x4xi8> {
                    %0 = tenzo.ternary_pack %arg0 : (tensor<4x16xf32>) -> tensor<4x4xi8>
                    return %0 : tensor<4x4xi8>
                }
            }
        )mlir";
        auto parsedModule = mlir::parseSourceString<mlir::ModuleOp>(mlirText, &context);
        assert(parsedModule && "Failed to parse textual MLIR containing tenzo.ternary_pack!");
        assert(mlir::succeeded(parsedModule->verify()) && "Parsed module verification failed!");

        llvm::outs() << "  ✅ TernaryPackOp verified with scale, without scale, and via textual MLIR parser!\n";
    }

    // ------------------------------------------------------------------------
    // TEST 2: Fused Micro-Kernel Generation Pass Verification (R2 & R3)
    // ------------------------------------------------------------------------
    llvm::outs() << "[Test 2/9] Verifying ExplicitMicroKernelPass on tenzo.ternary_pack (MR=4, NR=16)...\n";
    {
        mlir::OpBuilder builder(&context);
        auto loc = builder.getUnknownLoc();
        auto f32Type = builder.getF32Type();
        auto i8Type = builder.getI8Type();

        // Shapes: A is [4 x 64], B is [64 x 16], C is [4 x 4] uint8 memref
        auto aMemRefType = mlir::MemRefType::get({4, 64}, f32Type);
        auto bMemRefType = mlir::MemRefType::get({64, 16}, f32Type);
        auto cMemRefType = mlir::MemRefType::get({4, 4}, i8Type);

        auto module = mlir::ModuleOp::create(loc);
        builder.setInsertionPointToStart(module.getBody());

        auto funcType = builder.getFunctionType(
            {aMemRefType, bMemRefType, cMemRefType}, {});
        auto func = builder.create<mlir::func::FuncOp>(loc, "fused_ternary_pack_microkernel", funcType);
        auto &entryBlock = *func.addEntryBlock();
        builder.setInsertionPointToStart(&entryBlock);

        mlir::Value A = entryBlock.getArgument(0);
        mlir::Value B = entryBlock.getArgument(1);
        mlir::Value C = entryBlock.getArgument(2);

        // Create linalg.matmul with tenzo.ternary_pack attribute and scale factor
        auto matmulOp = builder.create<mlir::linalg::MatmulOp>(
            loc, mlir::ValueRange{A, B}, mlir::ValueRange{C});
        matmulOp->setAttr("tenzo.ternary_pack", builder.getUnitAttr());
        matmulOp->setAttr("bit_width", builder.getI32IntegerAttr(2));
        matmulOp->setAttr("scale", builder.getF32FloatAttr(1.0f));

        builder.create<mlir::func::ReturnOp>(loc);

        // Run ExplicitMicroKernelPass
        mlir::PassManager pm(&context);
        tenzo::MicroKernelParams params;
        params.MR = 6;  // Global default is 6
        params.NR = 16;
        params.VEC_SIZE = 8;
        tenzo::addExplicitMicroKernelPass(pm, params);

        mlir::LogicalResult result = pm.run(module);
        assert(mlir::succeeded(result) && "ExplicitMicroKernelPass failed on tenzo.ternary_pack!");
        assert(mlir::succeeded(module.verify()) && "ExplicitMicroKernelPass produced invalid IR!");

        // Verify that linalg.matmul was replaced and scf.for / vector transfer ops were generated
        bool foundMatmul = false;
        bool foundSCFFor = false;
        bool foundTransferWrite = false;
        bool foundVectorShuffle = false;

        func.walk([&](mlir::Operation *op) {
            if (llvm::isa<mlir::linalg::MatmulOp>(op)) foundMatmul = true;
            if (llvm::isa<mlir::scf::ForOp>(op)) foundSCFFor = true;
            if (llvm::isa<mlir::vector::TransferWriteOp>(op)) foundTransferWrite = true;
            if (llvm::isa<mlir::vector::ShuffleOp>(op)) foundVectorShuffle = true;
        });

        assert(!foundMatmul && "linalg.matmul should have been lowered/erased!");
        assert(foundSCFFor && "K-loop (scf.for) should have been generated!");
        assert(foundTransferWrite && "Final packed transfer writes should have been generated!");
        assert(foundVectorShuffle && "Vector shuffle for packing should have been generated!");

        llvm::outs() << "  ✅ ExplicitMicroKernelPass successfully generated valid MR=4, NR=16 micro-kernel AST!\n";
    }

    // ------------------------------------------------------------------------
    // TEST 3: Standalone TernaryPackOp and Consumer Op Replacement
    // ------------------------------------------------------------------------
    llvm::outs() << "[Test 3/9] Verifying Standalone TernaryPackOp & Consumer Replacement...\n";
    {
        mlir::OpBuilder builder(&context);
        auto loc = builder.getUnknownLoc();
        auto f32Type = builder.getF32Type();
        auto i8Type = builder.getI8Type();

        auto module = mlir::ModuleOp::create(loc);
        builder.setInsertionPointToStart(module.getBody());

        // 3a. Standalone tenzo.ternary_pack lowering
        auto inTensorType = mlir::RankedTensorType::get({4, 16}, f32Type);
        auto outTensorType = mlir::RankedTensorType::get({4, 4}, i8Type);
        auto funcType = builder.getFunctionType({inTensorType}, {outTensorType});
        auto func = builder.create<mlir::func::FuncOp>(loc, "standalone_pack", funcType);
        auto &entryBlock = *func.addEntryBlock();
        builder.setInsertionPointToStart(&entryBlock);

        auto packOp = builder.create<tenzo::TernaryPackOp>(
            loc, outTensorType, entryBlock.getArgument(0), mlir::Value(), builder.getI32IntegerAttr(4));
        builder.create<mlir::func::ReturnOp>(loc, packOp.getResult());

        mlir::PassManager pm(&context);
        tenzo::MicroKernelParams params;
        params.MR = 6;
        params.NR = 16;
        params.VEC_SIZE = 8;
        tenzo::addExplicitMicroKernelPass(pm, params);

        mlir::LogicalResult res = pm.run(module);
        assert(mlir::succeeded(res) && "Standalone TernaryPackOp lowering failed!");
        assert(mlir::succeeded(module.verify()) && "Standalone TernaryPackOp lowering produced invalid IR!");

        bool foundTernaryOp = false;
        func.walk([&](tenzo::TernaryPackOp) { foundTernaryOp = true; });
        assert(!foundTernaryOp && "Standalone TernaryPackOp should have been lowered!");

        llvm::outs() << "  ✅ Standalone TernaryPackOp successfully lowered and verified!\n";
    }

    // ------------------------------------------------------------------------
    // TEST 4: FP32 Isolation Verification (Preserve MR=6, NR=16 for standard GEMM)
    // ------------------------------------------------------------------------
    llvm::outs() << "[Test 4/9] Verifying FP32 Pipeline Isolation (MR=6, NR=16 unmodified)...\n";
    {
        mlir::OpBuilder builder(&context);
        auto loc = builder.getUnknownLoc();
        auto f32Type = builder.getF32Type();

        // FP32 microkernel: 6x16
        auto aMemRefType = mlir::MemRefType::get({6, 64}, f32Type);
        auto bMemRefType = mlir::MemRefType::get({64, 16}, f32Type);
        auto cMemRefType = mlir::MemRefType::get({6, 16}, f32Type);

        auto module = mlir::ModuleOp::create(loc);
        builder.setInsertionPointToStart(module.getBody());

        auto funcType = builder.getFunctionType({aMemRefType, bMemRefType, cMemRefType}, {});
        auto func = builder.create<mlir::func::FuncOp>(loc, "fp32_microkernel_6x16", funcType);
        auto &entryBlock = *func.addEntryBlock();
        builder.setInsertionPointToStart(&entryBlock);

        mlir::Value A = entryBlock.getArgument(0);
        mlir::Value B = entryBlock.getArgument(1);
        mlir::Value C = entryBlock.getArgument(2);

        // Standard FP32 matmul (no ternary pack attr)
        builder.create<mlir::linalg::MatmulOp>(loc, mlir::ValueRange{A, B}, mlir::ValueRange{C});
        builder.create<mlir::func::ReturnOp>(loc);

        mlir::PassManager pm(&context);
        tenzo::MicroKernelParams params;
        params.MR = 6;
        params.NR = 16;
        params.VEC_SIZE = 8;
        tenzo::addExplicitMicroKernelPass(pm, params);

        mlir::LogicalResult result = pm.run(module);
        assert(mlir::succeeded(result) && "ExplicitMicroKernelPass failed on standard FP32 GEMM!");
        assert(mlir::succeeded(module.verify()) && "FP32 GEMM lowering produced invalid IR!");

        bool foundMatmul = false;
        int transferWriteCount = 0;
        func.walk([&](mlir::Operation *op) {
            if (llvm::isa<mlir::linalg::MatmulOp>(op)) foundMatmul = true;
            if (llvm::isa<mlir::vector::TransferWriteOp>(op)) transferWriteCount++;
        });

        assert(!foundMatmul && "Standard FP32 linalg.matmul should have been lowered!");
        // For 6x16 with VEC_SIZE=8: 6 rows * 2 vectors = 12 transfer writes to C
        assert(transferWriteCount == 12 && "Standard FP32 must produce exactly 12 transfer writes (6x2)!");

        llvm::outs() << "  ✅ Standard FP32 GEMM generation intact: exactly 12 accumulators stored (6x16)!\n";
    }

    // ------------------------------------------------------------------------
    // TEST 5: AVX2 Fused Ternary Pack Bit-Exact Math Verification
    // ------------------------------------------------------------------------
    llvm::outs() << "[Test 5/9] Verifying AVX2 Fused Ternary Bit-Exact Arithmetic (Scales 1.0 and 0.5)...\n";
    {
        alignas(32) float input[4 * 16];
        alignas(32) uint8_t output[4 * 4];

        // Scale 1.0f tests with heterogeneous row 0:
        // group 0 (0..3): +1.0f -> 0xAA
        // group 1 (4..7): 0.0f -> 0x55
        // group 2 (8..11): -1.0f -> 0x00
        // group 3 (12..15): [-1, 0, 1, -1] -> 0x24
        for (int j = 0; j < 4; j++) input[0 * 16 + j] = 1.0f;
        for (int j = 4; j < 8; j++) input[0 * 16 + j] = 0.0f;
        for (int j = 8; j < 12; j++) input[0 * 16 + j] = -1.0f;
        input[0 * 16 + 12] = -1.0f; input[0 * 16 + 13] = 0.0f; input[0 * 16 + 14] = 1.0f; input[0 * 16 + 15] = -1.0f;

        // Row 1: homogeneous 0.0f
        for (int j = 0; j < 16; j++) input[1 * 16 + j] = 0.0f;
        // Row 2: homogeneous -1.0f
        for (int j = 0; j < 16; j++) input[2 * 16 + j] = -1.0f;
        // Row 3: homogeneous +1.0f
        for (int j = 0; j < 16; j++) input[3 * 16 + j] = 1.0f;

        avx2_fused_ternary_pack_4x16(input, 1.0f, output);

        assert(output[0 * 4 + 0] == 0xAA && "Row 0 Group 0 failed!");
        assert(output[0 * 4 + 1] == 0x55 && "Row 0 Group 1 failed!");
        assert(output[0 * 4 + 2] == 0x00 && "Row 0 Group 2 failed!");
        assert(output[0 * 4 + 3] == 0x24 && "Row 0 Group 3 failed!");
        for (int b = 0; b < 4; b++) assert(output[1 * 4 + b] == 0x55 && "Row 1 (all 0) failed packing check!");
        for (int b = 0; b < 4; b++) assert(output[2 * 4 + b] == 0x00 && "Row 2 (all -1) failed packing check!");
        for (int b = 0; b < 4; b++) assert(output[3 * 4 + b] == 0xAA && "Row 3 (all +1) failed packing check!");

        // Scale 0.5f test on unscaled values with heterogeneous row 0
        for (int j = 0; j < 4; j++) input[0 * 16 + j] = 2.0f;
        for (int j = 4; j < 8; j++) input[0 * 16 + j] = 0.0f;
        for (int j = 8; j < 12; j++) input[0 * 16 + j] = -2.0f;
        input[0 * 16 + 12] = -2.0f; input[0 * 16 + 13] = 0.0f; input[0 * 16 + 14] = 2.0f; input[0 * 16 + 15] = -2.0f;

        for (int j = 0; j < 16; j++) input[1 * 16 + j] = 0.0f;
        for (int j = 0; j < 16; j++) input[2 * 16 + j] = -2.0f;
        for (int j = 0; j < 16; j++) input[3 * 16 + j] = 2.0f;

        avx2_fused_ternary_pack_4x16(input, 0.5f, output);
        assert(output[0 * 4 + 0] == 0xAA && "Row 0 (scale=0.5) Group 0 failed!");
        assert(output[0 * 4 + 1] == 0x55 && "Row 0 (scale=0.5) Group 1 failed!");
        assert(output[0 * 4 + 2] == 0x00 && "Row 0 (scale=0.5) Group 2 failed!");
        assert(output[0 * 4 + 3] == 0x24 && "Row 0 (scale=0.5) Group 3 failed!");
        for (int b = 0; b < 4; b++) assert(output[1 * 4 + b] == 0x55 && "Row 1 (scale=0.5) failed!");
        for (int b = 0; b < 4; b++) assert(output[2 * 4 + b] == 0x00 && "Row 2 (scale=0.5) failed!");
        for (int b = 0; b < 4; b++) assert(output[3 * 4 + b] == 0xAA && "Row 3 (scale=0.5) failed!");

        llvm::outs() << "  ✅ Bit-exact validation passed across ternary domains and scaling factors!\n";
    }

    // ------------------------------------------------------------------------
    // TEST 6: True End-to-End JIT Execution of Fused Microkernel via ExecutionEngine
    // ------------------------------------------------------------------------
    llvm::outs() << "[Test 6/9] Verifying End-to-End JIT Execution of Fused Micro-Kernel...\n";
    {
        mlir::OpBuilder builder(&context);
        auto loc = builder.getUnknownLoc();
        auto f32Type = builder.getF32Type();
        auto i8Type = builder.getI8Type();

        // K=16, M=4, N=16, Output packed is 4x4 uint8
        auto aMemRefType = mlir::MemRefType::get({4, 16}, f32Type);
        auto bMemRefType = mlir::MemRefType::get({16, 16}, f32Type);
        auto cMemRefType = mlir::MemRefType::get({4, 4}, i8Type);

        auto module = mlir::ModuleOp::create(loc);
        builder.setInsertionPointToStart(module.getBody());

        auto funcType = builder.getFunctionType({aMemRefType, bMemRefType, cMemRefType}, {});
        auto func = builder.create<mlir::func::FuncOp>(loc, "jit_fused_kernel", funcType);
        func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
        auto &entryBlock = *func.addEntryBlock();
        builder.setInsertionPointToStart(&entryBlock);

        mlir::Value A = entryBlock.getArgument(0);
        mlir::Value B = entryBlock.getArgument(1);
        mlir::Value C = entryBlock.getArgument(2);

        auto matmulOp = builder.create<mlir::linalg::MatmulOp>(
            loc, mlir::ValueRange{A, B}, mlir::ValueRange{C});
        matmulOp->setAttr("tenzo.ternary_pack", builder.getUnitAttr());
        matmulOp->setAttr("scale", builder.getF32FloatAttr(1.0f));

        builder.create<mlir::func::ReturnOp>(loc);

        // Step 1: ExplicitMicroKernelPass
        mlir::PassManager pm(&context);
        tenzo::MicroKernelParams params;
        params.MR = 6;
        params.NR = 16;
        params.VEC_SIZE = 8;
        tenzo::addExplicitMicroKernelPass(pm, params);
        assert(mlir::succeeded(pm.run(module)) && "ExplicitMicroKernelPass failed in JIT test!");
        assert(mlir::succeeded(module.verify()) && "Module verification failed after microkernel pass!");

        // Step 2: Lower to LLVM
        mlir::PassManager llvmPM(&context);
        tenzo::TileSizes tiles;
        tiles.M = 4;
        tiles.N = 16;
        tiles.K = 16;
        tenzo::addTenzoToLLVMPasses(llvmPM, /*enableVectorization=*/false, tiles, /*enableParallel=*/false, /*useExplicitKernel=*/false);
        assert(mlir::succeeded(llvmPM.run(module)) && "LLVM lowering failed in JIT test!");

        // Step 3: JIT Compilation
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        tenzo::registerAllTenzoDialectTranslations(context);
        mlir::registerLLVMDialectTranslation(context);
        mlir::registerBuiltinDialectTranslation(context);

        mlir::ExecutionEngineOptions engineOptions;
        engineOptions.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);
        engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

        auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
        assert(maybeEngine && "Failed to create ExecutionEngine for fused microkernel!");
        auto engine = std::move(maybeEngine.get());

        // Step 4: Populate test matrices:
        // Let B be identity matrix 16x16, so C = A * I = A.
        alignas(32) float A_data[4 * 16];
        alignas(32) float B_data[16 * 16] = {0};
        alignas(32) uint8_t C_data[4 * 4] = {0};

        // B = Identity matrix
        for (int k = 0; k < 16; k++) {
            B_data[k * 16 + k] = 1.0f;
        }

        // Row 0: all +1.0f
        for (int j = 0; j < 16; j++) A_data[0 * 16 + j] = 1.0f;
        // Row 1: all 0.0f
        for (int j = 0; j < 16; j++) A_data[1 * 16 + j] = 0.0f;
        // Row 2: all -1.0f
        for (int j = 0; j < 16; j++) A_data[2 * 16 + j] = -1.0f;
        // Row 3: heterogeneous pattern (all 4 distinct groups in one row)
        // Group 0: all +1.0f -> 0xAA
        // Group 1: all 0.0f -> 0x55
        // Group 2: all -1.0f -> 0x00
        // Group 3: mixed [-1, 0, 1, -1] -> 0x24
        for (int j = 0; j < 4; j++) A_data[3 * 16 + j] = 1.0f;
        for (int j = 4; j < 8; j++) A_data[3 * 16 + j] = 0.0f;
        for (int j = 8; j < 12; j++) A_data[3 * 16 + j] = -1.0f;
        A_data[3 * 16 + 12] = -1.0f; A_data[3 * 16 + 13] = 0.0f; A_data[3 * 16 + 14] = 1.0f; A_data[3 * 16 + 15] = -1.0f;

        auto descA = tenzo::runtime::MemRefDescriptor<float, 2>::create(A_data, {4, 16});
        auto descB = tenzo::runtime::MemRefDescriptor<float, 2>::create(B_data, {16, 16});
        auto descC = tenzo::runtime::MemRefDescriptor<uint8_t, 2>::create(C_data, {4, 4});

        auto expectedFPtr = engine->lookup("_mlir_ciface_jit_fused_kernel");
        assert(expectedFPtr && "Failed to lookup _mlir_ciface_jit_fused_kernel!");
        auto fn = reinterpret_cast<void (*)(void*, void*, void*)>(*expectedFPtr);
        fn(&descA, &descB, &descC);

        // Step 5: Verify bit-exact results from executed MLIR microkernel!
        for (int b = 0; b < 4; b++) {
            assert(C_data[0 * 4 + b] == 0xAA && "JIT Execution: Row 0 mismatch!");
            assert(C_data[1 * 4 + b] == 0x55 && "JIT Execution: Row 1 mismatch!");
            assert(C_data[2 * 4 + b] == 0x00 && "JIT Execution: Row 2 mismatch!");
        }
        assert(C_data[3 * 4 + 0] == 0xAA && "JIT Execution: Row 3 Group 0 mismatch!");
        assert(C_data[3 * 4 + 1] == 0x55 && "JIT Execution: Row 3 Group 1 mismatch!");
        assert(C_data[3 * 4 + 2] == 0x00 && "JIT Execution: Row 3 Group 2 mismatch!");
        assert(C_data[3 * 4 + 3] == 0x24 && "JIT Execution: Row 3 Group 3 mismatch!");

        llvm::outs() << "  ✅ JIT Execution of MLIR Fused Micro-Kernel PASSED with bit-exact outputs:\n";
        llvm::outs() << "     Row 0 (+1s): 0x" << llvm::utohexstr(C_data[0]) << " (expected 0xaa)\n";
        llvm::outs() << "     Row 1 ( 0s): 0x" << llvm::utohexstr(C_data[4]) << " (expected 0x55)\n";
        llvm::outs() << "     Row 2 (-1s): 0x" << llvm::utohexstr(C_data[8]) << " (expected 0x00)\n";
        llvm::outs() << "     Row 3 (hetero): [0x" << llvm::utohexstr(C_data[12]) << ", 0x" << llvm::utohexstr(C_data[13])
                     << ", 0x" << llvm::utohexstr(C_data[14]) << ", 0x" << llvm::utohexstr(C_data[15]) << "] (expected [0xaa, 0x55, 0x00, 0x24])\n";
    }

    // ------------------------------------------------------------------------
    // TEST 7: Pipeline Fusion of linalg.matmul -> bufferization.to_tensor -> tenzo.ternary_pack
    // ------------------------------------------------------------------------
    llvm::outs() << "[Test 7/9] Verifying Fused Matmul + ToTensor + TernaryPackOp Pipeline...\n";
    {
        mlir::OpBuilder builder(&context);
        auto loc = builder.getUnknownLoc();
        auto f32Type = builder.getF32Type();
        auto i8Type = builder.getI8Type();

        auto aMemRefType = mlir::MemRefType::get({4, 16}, f32Type);
        auto bMemRefType = mlir::MemRefType::get({16, 16}, f32Type);
        auto cMemRefType = mlir::MemRefType::get({4, 16}, f32Type); // standard FP32 C buffer
        auto cTensorType = mlir::RankedTensorType::get({4, 16}, f32Type);
        auto outTensorType = mlir::RankedTensorType::get({4, 4}, i8Type);

        auto module = mlir::ModuleOp::create(loc);
        builder.setInsertionPointToStart(module.getBody());

        auto funcType = builder.getFunctionType({aMemRefType, bMemRefType, cMemRefType}, {outTensorType});
        auto func = builder.create<mlir::func::FuncOp>(loc, "pipeline_fused_pack", funcType);
        auto &entryBlock = *func.addEntryBlock();
        builder.setInsertionPointToStart(&entryBlock);

        mlir::Value A = entryBlock.getArgument(0);
        mlir::Value B = entryBlock.getArgument(1);
        mlir::Value C = entryBlock.getArgument(2);

        // linalg.matmul producing C: memref<4x16xf32>
        auto matmulOp = builder.create<mlir::linalg::MatmulOp>(
            loc, mlir::ValueRange{A, B}, mlir::ValueRange{C});

        // to_tensor bridging to tenzo.ternary_pack
        auto toTensorOp = builder.create<mlir::bufferization::ToTensorOp>(
            loc, cTensorType, C);

        // tenzo.ternary_pack consuming to_tensor
        auto packOp = builder.create<tenzo::TernaryPackOp>(
            loc, outTensorType, toTensorOp.getResult(), mlir::Value(), builder.getI32IntegerAttr(4));

        builder.create<mlir::func::ReturnOp>(loc, packOp.getResult());

        assert(mlir::succeeded(module.verify()) && "Initial pipeline IR verification failed!");

        // Run ExplicitMicroKernelPass
        mlir::PassManager pm(&context);
        tenzo::MicroKernelParams params;
        params.MR = 6;
        params.NR = 16;
        params.VEC_SIZE = 8;
        tenzo::addExplicitMicroKernelPass(pm, params);

        mlir::LogicalResult res = pm.run(module);
        assert(mlir::succeeded(res) && "ExplicitMicroKernelPass failed on Matmul+ToTensor+TernaryPack pipeline!");
        assert(mlir::succeeded(module.verify()) && "Microkernel pass produced invalid IR!");

        bool foundMatmul = false;
        bool foundPackOp = false;
        func.walk([&](mlir::linalg::MatmulOp) { foundMatmul = true; });
        func.walk([&](tenzo::TernaryPackOp) { foundPackOp = true; });

        assert(!foundMatmul && "linalg.matmul should have been fused into micro-kernel!");
        assert(!foundPackOp && "tenzo.ternary_pack should have been fused and eliminated!");

        llvm::outs() << "  ✅ Pipeline Matmul + ToTensor + TernaryPackOp successfully fused!\n";
    }

    // ------------------------------------------------------------------------
    // TEST 8: Standalone TernaryPackOp End-to-End JIT Execution (Scale 0.5f & Negative)
    // ------------------------------------------------------------------------
    llvm::outs() << "[Test 8/9] Verifying Standalone TernaryPackOp JIT Execution (Scales 0.5 & -1.0)...\n";
    {
        mlir::OpBuilder builder(&context);
        auto loc = builder.getUnknownLoc();
        auto f32Type = builder.getF32Type();
        auto i8Type = builder.getI8Type();

        auto inMemRefType = mlir::MemRefType::get({4, 16}, f32Type);
        auto inTensorType = mlir::RankedTensorType::get({4, 16}, f32Type);
        auto outMemRefType = mlir::MemRefType::get({4, 4}, i8Type);

        auto module = mlir::ModuleOp::create(loc);
        builder.setInsertionPointToStart(module.getBody());

        // Function with C interface taking inMemRef, scale, outMemRef
        auto funcType = builder.getFunctionType({inMemRefType, f32Type, outMemRefType}, {});
        auto func = builder.create<mlir::func::FuncOp>(loc, "jit_standalone_pack", funcType);
        func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
        auto &entryBlock = *func.addEntryBlock();
        builder.setInsertionPointToStart(&entryBlock);

        mlir::Value inMem = entryBlock.getArgument(0);
        mlir::Value scaleVal = entryBlock.getArgument(1);
        mlir::Value outMem = entryBlock.getArgument(2);

        auto toTensorOp = builder.create<mlir::bufferization::ToTensorOp>(loc, inTensorType, inMem);
        auto packOp = builder.create<tenzo::TernaryPackOp>(
            loc, mlir::RankedTensorType::get({4, 4}, i8Type), toTensorOp.getResult(), scaleVal, builder.getI32IntegerAttr(4));

        // Store packed result to outMem
        auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto zeroI8 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 8);
        for (int64_t i = 0; i < 4; i++) {
            auto iIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
            auto readVec = builder.create<mlir::vector::TransferReadOp>(
                loc, mlir::VectorType::get({4}, i8Type), packOp.getResult(), mlir::ValueRange{iIdx, c0}, zeroI8);
            builder.create<mlir::vector::TransferWriteOp>(
                loc, readVec, outMem, mlir::ValueRange{iIdx, c0});
        }
        builder.create<mlir::func::ReturnOp>(loc);

        // Lower with ExplicitMicroKernelPass
        mlir::PassManager pm(&context);
        tenzo::MicroKernelParams params;
        params.MR = 6;
        params.NR = 16;
        params.VEC_SIZE = 8;
        tenzo::addExplicitMicroKernelPass(pm, params);
        assert(mlir::succeeded(pm.run(module)) && "Standalone pack microkernel lowering failed!");
        assert(mlir::succeeded(module.verify()) && "Module verification failed after standalone pack!");

        // Lower to LLVM
        mlir::PassManager llvmPM(&context);
        llvmPM.addPass(mlir::createCanonicalizerPass());
        llvmPM.addPass(mlir::createConvertBufferizationToMemRefPass());
        llvmPM.addPass(mlir::createReconcileUnrealizedCastsPass());
        tenzo::TileSizes tiles{4, 16, 16};
        tenzo::addTenzoToLLVMPasses(llvmPM, false, tiles, false, false);
        assert(mlir::succeeded(llvmPM.run(module)) && "LLVM lowering failed for standalone pack!");

        // JIT Compile
        mlir::ExecutionEngineOptions engineOptions;
        engineOptions.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);
        engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

        auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
        assert(maybeEngine && "Failed to create ExecutionEngine for standalone pack!");
        auto engine = std::move(maybeEngine.get());

        alignas(32) float in_data[4 * 16];
        alignas(32) uint8_t out_data[4 * 4] = {0};

        // Populate unscaled values for scale 0.5f:
        // Row 0: +2.0f -> scaled to +1.0f -> packed 0xAA
        for (int j = 0; j < 16; j++) in_data[0 * 16 + j] = 2.0f;
        // Row 1: 0.0f -> scaled to 0.0f -> packed 0x55
        for (int j = 0; j < 16; j++) in_data[1 * 16 + j] = 0.0f;
        // Row 2: -2.0f -> scaled to -1.0f -> packed 0x00
        for (int j = 0; j < 16; j++) in_data[2 * 16 + j] = -2.0f;
        // Row 3: alternating [-2.0f, 0.0f, +2.0f, -2.0f] -> scaled to [-1, 0, 1, -1] -> packed 0x24
        for (int j = 0; j < 16; j++) {
            int pat = j % 4;
            if (pat == 0) in_data[3 * 16 + j] = -2.0f;
            else if (pat == 1) in_data[3 * 16 + j] = 0.0f;
            else if (pat == 2) in_data[3 * 16 + j] = 2.0f;
            else in_data[3 * 16 + j] = -2.0f;
        }

        auto descIn = tenzo::runtime::MemRefDescriptor<float, 2>::create(in_data, {4, 16});
        auto descOut = tenzo::runtime::MemRefDescriptor<uint8_t, 2>::create(out_data, {4, 4});

        auto fptr = engine->lookup("_mlir_ciface_jit_standalone_pack");
        assert(fptr && "Failed to lookup _mlir_ciface_jit_standalone_pack!");
        float scale = 0.5f;
        auto fn = reinterpret_cast<void (*)(void*, float, void*)>(*fptr);
        fn(&descIn, scale, &descOut);

        for (int b = 0; b < 4; b++) {
            assert(out_data[0 * 4 + b] == 0xAA && "Standalone JIT: Row 0 mismatch!");
            assert(out_data[1 * 4 + b] == 0x55 && "Standalone JIT: Row 1 mismatch!");
            assert(out_data[2 * 4 + b] == 0x00 && "Standalone JIT: Row 2 mismatch!");
            assert(out_data[3 * 4 + b] == 0x24 && "Standalone JIT: Row 3 mismatch!");
        }

        // Test with negative scale -0.5f
        float negScale = -0.5f;
        fn(&descIn, negScale, &descOut);
        for (int b = 0; b < 4; b++) {
            assert(out_data[0 * 4 + b] == 0x00 && "Standalone JIT NegScale: Row 0 mismatch!");
            assert(out_data[1 * 4 + b] == 0x55 && "Standalone JIT NegScale: Row 1 mismatch!");
            assert(out_data[2 * 4 + b] == 0xAA && "Standalone JIT NegScale: Row 2 mismatch!");
            assert(out_data[3 * 4 + b] == 0x86 && "Standalone JIT NegScale: Row 3 mismatch!");
        }

        llvm::outs() << "  ✅ Standalone JIT Execution of tenzo.ternary_pack PASSED with positive and negative scaling!\n";
    }

    // ------------------------------------------------------------------------
    // TEST 9: Edge Case & Negative Path Verification
    // ------------------------------------------------------------------------
    llvm::outs() << "[Test 9/9] Verifying Negative Paths & Edge Cases (K divisibility, values_per_byte, 0D scale)...\n";
    {
        mlir::OpBuilder builder(&context);
        auto loc = builder.getUnknownLoc();
        auto f32Type = builder.getF32Type();
        auto i8Type = builder.getI8Type();

        // 9a. Matmul with K=6 (not divisible by 4) must be rejected by microkernel pass
        {
            auto aType = mlir::MemRefType::get({4, 6}, f32Type);
            auto bType = mlir::MemRefType::get({6, 16}, f32Type);
            auto cType = mlir::MemRefType::get({4, 4}, i8Type);
            auto module = mlir::ModuleOp::create(loc);
            builder.setInsertionPointToStart(module.getBody());

            auto funcType = builder.getFunctionType({aType, bType, cType}, {});
            auto func = builder.create<mlir::func::FuncOp>(loc, "test_invalid_k", funcType);
            auto &block = *func.addEntryBlock();
            builder.setInsertionPointToStart(&block);
            auto matmul = builder.create<mlir::linalg::MatmulOp>(
                loc, mlir::ValueRange{block.getArgument(0), block.getArgument(1)},
                mlir::ValueRange{block.getArgument(2)});
            matmul->setAttr("tenzo.ternary_pack", builder.getUnitAttr());
            builder.create<mlir::func::ReturnOp>(loc);

            mlir::PassManager pm(&context);
            tenzo::MicroKernelParams params;
            params.MR = 6; params.NR = 16; params.VEC_SIZE = 8;
            tenzo::addExplicitMicroKernelPass(pm, params);
            (void)pm.run(module);

            // Matmul must NOT have been lowered because K=6 is not divisible by 4
            bool foundMatmul = false;
            func.walk([&](mlir::linalg::MatmulOp) { foundMatmul = true; });
            assert(foundMatmul && "ExplicitMicroKernelPass should reject K not divisible by 4!");
        }

        // 9b. TernaryPackOp with values_per_byte != 4 must be rejected
        {
            auto inTensorType = mlir::RankedTensorType::get({4, 16}, f32Type);
            auto outTensorType = mlir::RankedTensorType::get({4, 4}, i8Type);
            auto module = mlir::ModuleOp::create(loc);
            builder.setInsertionPointToStart(module.getBody());

            auto funcType = builder.getFunctionType({inTensorType}, {outTensorType});
            auto func = builder.create<mlir::func::FuncOp>(loc, "test_invalid_vpb", funcType);
            auto &block = *func.addEntryBlock();
            builder.setInsertionPointToStart(&block);

            // Create TernaryPackOp with values_per_byte = 2
            auto packOp = builder.create<tenzo::TernaryPackOp>(
                loc, outTensorType, block.getArgument(0), mlir::Value(), builder.getI32IntegerAttr(2));
            builder.create<mlir::func::ReturnOp>(loc, packOp.getResult());

            mlir::PassManager pm(&context);
            tenzo::MicroKernelParams params;
            params.MR = 6; params.NR = 16; params.VEC_SIZE = 8;
            tenzo::addExplicitMicroKernelPass(pm, params);
            (void)pm.run(module);

            bool foundPack = false;
            func.walk([&](tenzo::TernaryPackOp) { foundPack = true; });
            assert(foundPack && "ExplicitMicroKernelPass should reject values_per_byte != 4!");
        }

        // 9c. Standalone TernaryPackOp with 0D tensor scale
        {
            auto inMemRefType = mlir::MemRefType::get({4, 16}, f32Type);
            auto inTensorType = mlir::RankedTensorType::get({4, 16}, f32Type);
            auto scaleTensorType = mlir::RankedTensorType::get({}, f32Type);
            auto outMemRefType = mlir::MemRefType::get({4, 4}, i8Type);

            auto module = mlir::ModuleOp::create(loc);
            builder.setInsertionPointToStart(module.getBody());

            auto funcType = builder.getFunctionType({inMemRefType, scaleTensorType, outMemRefType}, {});
            auto func = builder.create<mlir::func::FuncOp>(loc, "test_0d_tensor_scale", funcType);
            auto &block = *func.addEntryBlock();
            builder.setInsertionPointToStart(&block);

            auto toTensor = builder.create<mlir::bufferization::ToTensorOp>(loc, inTensorType, block.getArgument(0));
            auto packOp = builder.create<tenzo::TernaryPackOp>(
                loc, mlir::RankedTensorType::get({4, 4}, i8Type), toTensor.getResult(), block.getArgument(1), builder.getI32IntegerAttr(4));

            auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            auto zeroI8 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 8);
            for (int64_t i = 0; i < 4; i++) {
                auto iIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
                auto readVec = builder.create<mlir::vector::TransferReadOp>(
                    loc, mlir::VectorType::get({4}, i8Type), packOp.getResult(), mlir::ValueRange{iIdx, c0}, zeroI8);
                builder.create<mlir::vector::TransferWriteOp>(
                    loc, readVec, block.getArgument(2), mlir::ValueRange{iIdx, c0});
            }
            builder.create<mlir::func::ReturnOp>(loc);

            mlir::PassManager pm(&context);
            tenzo::MicroKernelParams params;
            params.MR = 6; params.NR = 16; params.VEC_SIZE = 8;
            tenzo::addExplicitMicroKernelPass(pm, params);
            assert(mlir::succeeded(pm.run(module)) && "0D tensor scale lowering failed!");
            assert(mlir::succeeded(module.verify()) && "0D tensor scale module verification failed!");
        }

        llvm::outs() << "  ✅ All negative paths and edge cases passed verification!\n";
    }

    llvm::outs() << "\n🎉 ALL TERNARY PACK & AVX2 FUSED MICRO-KERNEL TESTS PASSED! 🎉\n\n";
}

} // namespace tenzo
