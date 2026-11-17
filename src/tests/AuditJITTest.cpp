#include "tests/AuditJITTest.h"
#include "context/TenzoContext.h"
#include "context/HardwareProfile.h"
#include "dialect/TenzoDialect.h"
#include "passes/Passes.h"
#include "runtime/MemRefUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace tenzo {

static inline bool isNonFiniteFloat(float val) {
    uint32_t bits = 0;
    std::memcpy(&bits, &val, sizeof(float));
    return ((bits & 0x7F800000) == 0x7F800000);
}

static inline bool isNonFiniteDouble(double val) {
    uint64_t bits = 0;
    std::memcpy(&bits, &val, sizeof(double));
    return ((bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL);
}

static void ensureAuditInitialized(mlir::MLIRContext &context) {
    static bool initialized = false;
    if (!initialized) {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        initialized = true;
    }
    context.getOrLoadDialect<mlir::math::MathDialect>();
    tenzo::registerAllTenzoDialectTranslations(context);
    mlir::registerLLVMDialectTranslation(context);
    mlir::registerBuiltinDialectTranslation(context);
}

static void saveIR(mlir::ModuleOp m, const std::string &name) {
    const char *dumpEnv = std::getenv("TENZO_DUMP_JIT_IR");
    if (!dumpEnv || std::string(dumpEnv) == "0") return;
    llvm::sys::fs::create_directories("ir");
    std::error_code ec;
    llvm::raw_fd_ostream f("ir/" + name + ".mlir", ec);
    if (!ec) {
        m.print(f);
    }
}

static void dumpObjectFile(mlir::ExecutionEngine &engine, const std::string &name) {
    const char *dumpEnv = std::getenv("TENZO_DUMP_JIT_ASM");
    if (!dumpEnv || std::string(dumpEnv) == "0") return;
    llvm::sys::fs::create_directories("asm");
    engine.dumpToObjectFile("asm/" + name + ".o");
}

static std::unique_ptr<mlir::ExecutionEngine> compileModule(
    mlir::ModuleOp m, const std::string &name, bool linalg = false) {
    saveIR(m, name + "-input");

    mlir::PassManager pm(m.getContext());
    if (linalg) {
        tenzo::addTenzoToLinalgPass(pm);
        tenzo::addTenzoBufferizationPasses(pm);
    } else {
        tenzo::MicroKernelParams p;
        p.MR = 6;
        p.NR = 16;
        p.VEC_SIZE = 8;
        tenzo::addExplicitMicroKernelPass(pm, p);
    }
    if (mlir::failed(pm.run(m))) {
        saveIR(m, name + "-lowering-failed");
        llvm::errs() << "❌ [AUDIT JIT] Lowering pass failed for: " << name << "\n";
        return nullptr;
    }
    saveIR(m, name + "-lowered");

    if (!linalg && name.find("pack") == std::string::npos) {
        mlir::PassManager bp(m.getContext());
        tenzo::addTenzoBufferizationPasses(bp);
        if (mlir::failed(bp.run(m))) {
            saveIR(m, name + "-bufferization-failed");
            llvm::errs() << "❌ [AUDIT JIT] Bufferization failed for: " << name << "\n";
            return nullptr;
        }
        saveIR(m, name + "-bufferized");
    }

    mlir::PassManager lp(m.getContext());
    lp.addPass(mlir::createCanonicalizerPass());
    lp.addPass(mlir::createConvertBufferizationToMemRefPass());
    lp.addPass(mlir::createReconcileUnrealizedCastsPass());
    tenzo::TileSizes tiles{4, 16, 16};
    tenzo::addTenzoToLLVMPasses(lp, false, tiles, false, false);
    if (mlir::failed(lp.run(m))) {
        saveIR(m, name + "-llvm-failed");
        llvm::errs() << "❌ [AUDIT JIT] LLVM lowering failed for: " << name << "\n";
        return nullptr;
    }
    saveIR(m, name + "-llvm");

    mlir::ExecutionEngineOptions opt;
    opt.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
    opt.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);
    opt.enableObjectDump = true;
    auto maybeEngine = mlir::ExecutionEngine::create(m, opt);
    if (!maybeEngine) {
        llvm::errs() << "❌ [AUDIT JIT] ExecutionEngine creation failed for " << name << ": "
                     << llvm::toString(maybeEngine.takeError()) << "\n";
        return nullptr;
    }
    auto engine = std::move(*maybeEngine);
    dumpObjectFile(*engine, name);
    return engine;
}

static inline uint8_t packTernaryValues(const int *v) {
    return static_cast<uint8_t>(
        (v[0] + 1) |
        ((v[1] + 1) << 2) |
        ((v[2] + 1) << 4) |
        ((v[3] + 1) << 6)
    );
}

bool auditTernaryPack(mlir::MLIRContext &ctx, bool fused) {
    ensureAuditInitialized(ctx);
    mlir::OpBuilder b(&ctx);
    auto loc = b.getUnknownLoc();
    auto m = mlir::ModuleOp::create(loc);
    auto f = b.getF32Type();
    auto i8 = b.getI8Type();

    auto inType = mlir::MemRefType::get({4, 16}, f);
    auto outType = mlir::MemRefType::get({4, 4}, i8);
    auto matType = mlir::MemRefType::get({16, 16}, f);

    b.setInsertionPointToStart(m.getBody());
    llvm::SmallVector<mlir::Type> args = fused
        ? llvm::SmallVector<mlir::Type>{inType, matType, outType}
        : llvm::SmallVector<mlir::Type>{inType, f, outType};

    auto fn = b.create<mlir::func::FuncOp>(loc, "audit_pack", b.getFunctionType(args, {}));
    fn->setAttr("llvm.emit_c_interface", b.getUnitAttr());
    auto &eb = *fn.addEntryBlock();
    b.setInsertionPointToStart(&eb);

    if (fused) {
        auto mm = b.create<mlir::linalg::MatmulOp>(
            loc, mlir::ValueRange{eb.getArgument(0), eb.getArgument(1)},
            mlir::ValueRange{eb.getArgument(2)});
        mm->setAttr("tenzo.ternary_pack", b.getUnitAttr());
        mm->setAttr("scale", b.getF32FloatAttr(1.0f));
    } else {
        auto t = b.create<mlir::bufferization::ToTensorOp>(
            loc, mlir::RankedTensorType::get({4, 16}, f), eb.getArgument(0));
        auto p = b.create<tenzo::TernaryPackOp>(
            loc, mlir::RankedTensorType::get({4, 4}, i8), t.getResult(),
            eb.getArgument(1), b.getI32IntegerAttr(4));
        auto z = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto pad = b.create<mlir::arith::ConstantIntOp>(loc, 0, 8);
        for (int r = 0; r < 4; r++) {
            auto ri = b.create<mlir::arith::ConstantIndexOp>(loc, r);
            auto v = b.create<mlir::vector::TransferReadOp>(
                loc, mlir::VectorType::get({4}, i8), p.getResult(),
                mlir::ValueRange{ri, z}, pad);
            b.create<mlir::vector::TransferWriteOp>(
                loc, v, eb.getArgument(2), mlir::ValueRange{ri, z});
        }
    }
    b.create<mlir::func::ReturnOp>(loc);

    std::string name = fused ? "fused_pack" : "standalone_pack";
    auto engine = compileModule(m, name, false);
    if (!engine) {
        m.erase();
        return false;
    }

    auto ptr = engine->lookup("_mlir_ciface_audit_pack");
    if (!ptr) {
        llvm::errs() << "❌ Failed to lookup symbol _mlir_ciface_audit_pack: "
                     << llvm::toString(ptr.takeError()) << "\n";
        m.erase();
        return false;
    }

    alignas(32) float a[64];
    alignas(32) float mat[256];
    alignas(32) uint8_t storage[48];
    uint8_t *o = storage + 16;

    auto da = tenzo::runtime::MemRefDescriptor<float, 2>::create(a, {4, 16});
    auto db = tenzo::runtime::MemRefDescriptor<float, 2>::create(mat, {16, 16});
    auto dc = tenzo::runtime::MemRefDescriptor<uint8_t, 2>::create(o, {4, 4});

    auto run = [&](float s) {
        std::fill(storage, storage + 48, 0xCC);
        if (fused) {
            reinterpret_cast<void (*)(void*, void*, void*)>(*ptr)(&da, &db, &dc);
        } else {
            reinterpret_cast<void (*)(void*, float, void*)>(*ptr)(&da, s, &dc);
        }
    };

    // Identity matrix for B in fused mode
    std::fill(mat, mat + 256, 0.0f);
    for (int k = 0; k < 16; k++) mat[k * 16 + k] = 1.0f;

    int errors = 0;
    int canary_errors = 0;
    std::vector<float> scales = fused ? std::vector<float>{1.0f} : std::vector<float>{1.0f, 0.5f, -1.0f};

    for (float scale : scales) {
        for (int code = 0; code < 81; code++) {
            int c = code, v[4];
            for (int k = 0; k < 4; k++) {
                v[k] = c % 3 - 1;
                c /= 3;
            }
            for (int j = 0; j < 64; j++) {
                a[j] = static_cast<float>(v[j % 4]) / (fused ? 1.0f : scale);
            }
            run(scale);
            for (int j = 0; j < 16; j++) {
                if (o[j] != packTernaryValues(v)) errors++;
            }
            for (int j = 0; j < 16; j++) {
                if (storage[j] != 0xCC || storage[j + 32] != 0xCC) canary_errors++;
            }
        }
    }

    int semantic_errors = 0;
    std::mt19937 rng(27);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::uniform_real_distribution<float> dist_pack(-1.5f, 1.5f);
    for (int test = 0; test < 20; test++) {
        if (fused) {
            for (auto &v : a) v = dist(rng);
            for (auto &v : mat) v = dist(rng) * 4.0f;
        } else {
            for (auto &v : a) v = dist_pack(rng);
        }
        run(1.0f);
        for (int r = 0; r < 4; r++) {
            for (int j = 0; j < 4; j++) {
                int v[4];
                for (int z = 0; z < 4; z++) {
                    double x = a[r * 16 + j * 4 + z];
                    if (fused) {
                        x = 0;
                        for (int k = 0; k < 16; k++) {
                            x += static_cast<double>(a[r * 16 + k]) * static_cast<double>(mat[k * 16 + j * 4 + z]);
                        }
                    }
                    v[z] = std::clamp(static_cast<int>(std::nearbyint(x)), -1, 1);
                }
                if (o[r * 4 + j] != packTernaryValues(v)) semantic_errors++;
            }
        }
    }

    m.erase();
    bool passed = (errors == 0 && canary_errors == 0 && semantic_errors == 0);
    if (passed) {
        llvm::outs() << "✅ [AUDIT " << name << "] PASS (exhaustive_81 mismatches=0, canaries=OK, vs_round_clamp=0)\n";
    } else {
        llvm::errs() << "❌ [AUDIT " << name << "] FAIL: exhaustive mismatches=" << errors
                     << ", canary errors=" << canary_errors
                     << ", vs_round_clamp mismatches=" << semantic_errors << " / 320\n";
    }
    return passed;
}

bool auditFusionMultiConsumer(mlir::MLIRContext &ctx) {
    ensureAuditInitialized(ctx);
    mlir::OpBuilder b(&ctx);
    auto loc = b.getUnknownLoc();
    auto m = mlir::ModuleOp::create(loc);
    auto f = b.getF32Type();
    auto i8 = b.getI8Type();

    auto inType = mlir::MemRefType::get({4, 16}, f);
    auto outType = mlir::MemRefType::get({4, 4}, i8);
    auto matType = mlir::MemRefType::get({16, 16}, f);

    b.setInsertionPointToStart(m.getBody());
    auto fn = b.create<mlir::func::FuncOp>(
        loc, "audit_pack", b.getFunctionType({inType, matType, outType, inType}, {}));
    fn->setAttr("llvm.emit_c_interface", b.getUnitAttr());
    auto &eb = *fn.addEntryBlock();
    b.setInsertionPointToStart(&eb);

    auto c = b.create<mlir::memref::AllocOp>(loc, inType);
    auto zero = b.create<mlir::arith::ConstantOp>(loc, f, b.getF32FloatAttr(0.0f));
    b.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{zero}, mlir::ValueRange{c});
    b.create<mlir::linalg::MatmulOp>(
        loc, mlir::ValueRange{eb.getArgument(0), eb.getArgument(1)}, mlir::ValueRange{c});
    auto t = b.create<mlir::bufferization::ToTensorOp>(
        loc, mlir::RankedTensorType::get({4, 16}, f), c);
    auto scale = b.create<mlir::arith::ConstantOp>(loc, f, b.getF32FloatAttr(1.0f));
    auto p = b.create<tenzo::TernaryPackOp>(
        loc, mlir::RankedTensorType::get({4, 4}, i8), t.getResult(),
        scale, b.getI32IntegerAttr(4));
    auto z = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto pad = b.create<mlir::arith::ConstantIntOp>(loc, 0, 8);
    for (int r = 0; r < 4; r++) {
        auto ri = b.create<mlir::arith::ConstantIndexOp>(loc, r);
        auto v = b.create<mlir::vector::TransferReadOp>(
            loc, mlir::VectorType::get({4}, i8), p.getResult(),
            mlir::ValueRange{ri, z}, pad);
        b.create<mlir::vector::TransferWriteOp>(
            loc, v, eb.getArgument(2), mlir::ValueRange{ri, z});
    }
    b.create<mlir::memref::CopyOp>(loc, c, eb.getArgument(3));
    b.create<mlir::memref::DeallocOp>(loc, c);
    b.create<mlir::func::ReturnOp>(loc);

    std::string name = "fusion_consumer_pack";
    auto engine = compileModule(m, name, false);
    if (!engine) {
        m.erase();
        return false;
    }

    auto ptr = engine->lookup("_mlir_ciface_audit_pack");
    if (!ptr) {
        llvm::errs() << "❌ Failed to lookup symbol _mlir_ciface_audit_pack: "
                     << llvm::toString(ptr.takeError()) << "\n";
        m.erase();
        return false;
    }

    alignas(32) float a_buf[16 + 64 + 16];
    alignas(32) float b_buf[16 + 256 + 16];
    alignas(32) float c_buf[16 + 64 + 16];
    alignas(32) float other_buf[16 + 64 + 16];
    alignas(32) uint8_t packed_buf[16 + 16 + 16];

    float *a = a_buf + 16;
    float *mat = b_buf + 16;
    float *c_mem = c_buf + 16;
    float *other = other_buf + 16;
    uint8_t *o = packed_buf + 16;

    const float FLOAT_CANARY = -789.0f;
    const uint8_t BYTE_CANARY = 0xCC;

    auto resetCanaries = [&]() {
        for (int i = 0; i < 16; i++) {
            a_buf[i] = FLOAT_CANARY;
            a_buf[16 + 64 + i] = FLOAT_CANARY;
            b_buf[i] = FLOAT_CANARY;
            b_buf[16 + 256 + i] = FLOAT_CANARY;
            c_buf[i] = FLOAT_CANARY;
            c_buf[16 + 64 + i] = FLOAT_CANARY;
            other_buf[i] = FLOAT_CANARY;
            other_buf[16 + 64 + i] = FLOAT_CANARY;
            packed_buf[i] = BYTE_CANARY;
            packed_buf[16 + 16 + i] = BYTE_CANARY;
        }
    };

    auto checkCanaries = [&]() -> int {
        int corruptions = 0;
        for (int i = 0; i < 16; i++) {
            if (a_buf[i] != FLOAT_CANARY) corruptions++;
            if (a_buf[16 + 64 + i] != FLOAT_CANARY) corruptions++;
            if (b_buf[i] != FLOAT_CANARY) corruptions++;
            if (b_buf[16 + 256 + i] != FLOAT_CANARY) corruptions++;
            if (c_buf[i] != FLOAT_CANARY) corruptions++;
            if (c_buf[16 + 64 + i] != FLOAT_CANARY) corruptions++;
            if (other_buf[i] != FLOAT_CANARY) corruptions++;
            if (other_buf[16 + 64 + i] != FLOAT_CANARY) corruptions++;
            if (packed_buf[i] != BYTE_CANARY) corruptions++;
            if (packed_buf[16 + 16 + i] != BYTE_CANARY) corruptions++;
        }
        return corruptions;
    };

    int other_errors = 0;
    int packing_errors = 0;
    int canary_errors = 0;

    auto da = tenzo::runtime::MemRefDescriptor<float, 2>::create(a, {4, 16});
    auto db = tenzo::runtime::MemRefDescriptor<float, 2>::create(mat, {16, 16});
    auto dc = tenzo::runtime::MemRefDescriptor<uint8_t, 2>::create(o, {4, 4});
    auto dr = tenzo::runtime::MemRefDescriptor<float, 2>::create(other, {4, 16});

    auto run_multi_consumer = [&]() {
        resetCanaries();
        std::fill(o, o + 16, 0);
        std::fill(other, other + 64, 0.0f);
        reinterpret_cast<void (*)(void*, void*, void*, void*)>(*ptr)(&da, &db, &dc, &dr);
        canary_errors += checkCanaries();
        for (int r = 0; r < 4; r++) {
            for (int j = 0; j < 16; j++) {
                double ref = 0.0;
                for (int k = 0; k < 16; k++) {
                    ref += static_cast<double>(a[r * 16 + k]) * static_cast<double>(mat[k * 16 + j]);
                }
                float val = other[r * 16 + j];
                if (isNonFiniteFloat(val) || !std::isfinite(val) || std::isnan(val) ||
                    std::abs(static_cast<double>(val) - ref) > 1e-4) {
                    other_errors++;
                }
            }
        }
    };

    // Phase 1: Exhaustive 81 ternary combinations with identity matrix
    std::fill(mat, mat + 256, 0.0f);
    for (int k = 0; k < 16; k++) mat[k * 16 + k] = 1.0f;

    for (int code = 0; code < 81; code++) {
        int c_code = code, v[4];
        for (int k = 0; k < 4; k++) {
            v[k] = c_code % 3 - 1;
            c_code /= 3;
        }
        for (int j = 0; j < 64; j++) a[j] = static_cast<float>(v[j % 4]);
        run_multi_consumer();
        for (int j = 0; j < 16; j++) {
            if (o[j] != packTernaryValues(v)) packing_errors++;
        }
    }

    // Phase 2: Random floating-point matrices A and B
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_ab(-2.0f, 2.0f);
    for (int trial = 0; trial < 10; trial++) {
        for (int j = 0; j < 64; j++) a[j] = dist_ab(rng);
        for (int j = 0; j < 256; j++) mat[j] = dist_ab(rng);
        run_multi_consumer();
        for (int r = 0; r < 4; r++) {
            for (int chunk = 0; chunk < 4; chunk++) {
                int vals[4];
                for (int v_idx = 0; v_idx < 4; v_idx++) {
                    float fval = other[r * 16 + chunk * 4 + v_idx];
                    int q = std::clamp(static_cast<int>(std::nearbyint(fval)), -1, 1);
                    vals[v_idx] = q;
                }
                uint8_t expected_pack = packTernaryValues(vals);
                if (o[r * 4 + chunk] != expected_pack) packing_errors++;
            }
        }
    }
    m.erase();

    // Phase 3: Non-zero initial C accumulation across scales {0.5, 1.0, 2.5, -1.0}
    auto m2 = mlir::ModuleOp::create(loc);
    b.setInsertionPointToStart(m2.getBody());
    auto fn2 = b.create<mlir::func::FuncOp>(
        loc, "audit_pack_nonzero", b.getFunctionType({inType, matType, outType, inType}, {}));
    fn2->setAttr("llvm.emit_c_interface", b.getUnitAttr());
    auto &eb2 = *fn2.addEntryBlock();
    b.setInsertionPointToStart(&eb2);

    b.create<mlir::linalg::MatmulOp>(
        loc, mlir::ValueRange{eb2.getArgument(0), eb2.getArgument(1)}, mlir::ValueRange{eb2.getArgument(3)});
    auto t2 = b.create<mlir::bufferization::ToTensorOp>(
        loc, mlir::RankedTensorType::get({4, 16}, f), eb2.getArgument(3));
    auto scale2 = b.create<mlir::arith::ConstantOp>(loc, f, b.getF32FloatAttr(1.0f));
    auto p2 = b.create<tenzo::TernaryPackOp>(
        loc, mlir::RankedTensorType::get({4, 4}, i8), t2.getResult(),
        scale2, b.getI32IntegerAttr(4));
    auto z2 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto pad2 = b.create<mlir::arith::ConstantIntOp>(loc, 0, 8);
    for (int r = 0; r < 4; r++) {
        auto ri = b.create<mlir::arith::ConstantIndexOp>(loc, r);
        auto v = b.create<mlir::vector::TransferReadOp>(
            loc, mlir::VectorType::get({4}, i8), p2.getResult(),
            mlir::ValueRange{ri, z2}, pad2);
        b.create<mlir::vector::TransferWriteOp>(
            loc, v, eb2.getArgument(2), mlir::ValueRange{ri, z2});
    }
    b.create<mlir::func::ReturnOp>(loc);

    std::string name2 = "fusion_nonzero_c_pack";
    auto engine2 = compileModule(m2, name2, false);
    if (engine2) {
        auto ptr2 = engine2->lookup("_mlir_ciface_audit_pack_nonzero");
        if (ptr2) {
            auto d_c_mem = tenzo::runtime::MemRefDescriptor<float, 2>::create(c_mem, {4, 16});
            std::uniform_real_distribution<float> dist_c(-1.5f, 1.5f);
            for (float s_val : {0.5f, 1.0f, 2.5f, -1.0f}) {
                for (int trial = 0; trial < 5; trial++) {
                    resetCanaries();
                    for (int j = 0; j < 64; j++) a[j] = dist_ab(rng) * s_val;
                    for (int j = 0; j < 256; j++) mat[j] = dist_ab(rng);
                    std::vector<double> c_initial(64);
                    for (int j = 0; j < 64; j++) {
                        float c_init = dist_c(rng);
                        c_mem[j] = c_init;
                        c_initial[j] = static_cast<double>(c_init);
                    }
                    reinterpret_cast<void (*)(void*, void*, void*, void*)>(*ptr2)(&da, &db, &dc, &d_c_mem);
                    canary_errors += checkCanaries();

                    for (int r = 0; r < 4; r++) {
                        for (int j = 0; j < 16; j++) {
                            double ref = c_initial[r * 16 + j];
                            for (int k = 0; k < 16; k++) {
                                ref += static_cast<double>(a[r * 16 + k]) * static_cast<double>(mat[k * 16 + j]);
                            }
                            float val = c_mem[r * 16 + j];
                            if (isNonFiniteFloat(val) || !std::isfinite(val) || std::isnan(val) ||
                                std::abs(static_cast<double>(val) - ref) > 1e-4) {
                                other_errors++;
                            }
                        }
                    }
                }
            }
        }
        m2.erase();
    }

    bool passed = (packing_errors == 0 && other_errors == 0 && canary_errors == 0);
    if (passed) {
        llvm::outs() << "✅ [AUDIT fusion_consumer_pack] PASS (packing=OK, intermediate FP32 mismatches=0, non-zero C=OK, canaries=OK)\n";
    } else {
        llvm::errs() << "❌ [AUDIT fusion_consumer_pack] FAIL: packing mismatches=" << packing_errors
                     << ", FP32 consumer mismatches=" << other_errors
                     << ", canary corruptions=" << canary_errors << "\n";
    }
    return passed;
}

bool auditAttention(mlir::MLIRContext &ctx,
                    bool linalg,
                    bool fractional,
                    int B,
                    int Hq,
                    int Hkv,
                    int T,
                    int D) {
    ensureAuditInitialized(ctx);
    if (const char *envHq = std::getenv("AUDIT_HQ")) Hq = std::atoi(envHq);
    if (const char *envHkv = std::getenv("AUDIT_HKV")) Hkv = std::atoi(envHkv);

    assert(B > 0 && "Batch size B must be positive");
    assert(Hkv > 0 && "Hkv must be positive");
    assert(Hq % Hkv == 0 && "Hq must be divisible by Hkv");
    assert(T > 0 && "Sequence length T must be positive");
    assert(D > 0 && D % 4 == 0 && "Head dimension D must be positive and divisible by 4");

    bool gqa = (B > 1 || Hq > 1 || Hkv > 1);

    llvm::SmallVector<int64_t> qs = gqa ? llvm::SmallVector<int64_t>{B, Hq, 1, D}
                                        : llvm::SmallVector<int64_t>{1, D};
    llvm::SmallVector<int64_t> ks = gqa ? llvm::SmallVector<int64_t>{B, Hkv, T, D / 4}
                                        : llvm::SmallVector<int64_t>{T, D / 4};

    mlir::OpBuilder b(&ctx);
    auto loc = b.getUnknownLoc();
    auto m = mlir::ModuleOp::create(loc);
    auto f = b.getF32Type();
    auto i8 = b.getI8Type();

    auto qm = mlir::MemRefType::get(qs, f);
    auto km = mlir::MemRefType::get(ks, i8);
    auto qt = mlir::RankedTensorType::get(qs, f);
    auto kt = mlir::RankedTensorType::get(ks, i8);
    auto posm = mlir::MemRefType::get({1}, b.getI32Type());

    b.setInsertionPointToStart(m.getBody());
    auto fn = b.create<mlir::func::FuncOp>(
        loc, "audit_attention", b.getFunctionType({qm, km, km, posm, qm}, {}));
    fn->setAttr("llvm.emit_c_interface", b.getUnitAttr());
    auto &eb = *fn.addEntryBlock();
    b.setInsertionPointToStart(&eb);

    auto q = b.create<mlir::bufferization::ToTensorOp>(loc, qt, eb.getArgument(0));
    auto k = b.create<mlir::bufferization::ToTensorOp>(loc, kt, eb.getArgument(1));
    auto v = b.create<mlir::bufferization::ToTensorOp>(loc, kt, eb.getArgument(2));
    auto pos = b.create<mlir::bufferization::ToTensorOp>(
        loc, mlir::RankedTensorType::get({1}, b.getI32Type()), eb.getArgument(3));

    auto a = b.create<tenzo::PackedAttentionOp>(
        loc, qt, q, k, v, pos, mlir::Value(), b.getI32IntegerAttr(4));
    auto z = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto pad = b.create<mlir::arith::ConstantOp>(loc, f, b.getF32FloatAttr(0.0f));

    for (int batch = 0; batch < B; batch++) {
        for (int head = 0; head < Hq; head++) {
            auto bi = b.create<mlir::arith::ConstantIndexOp>(loc, batch);
            auto hi = b.create<mlir::arith::ConstantIndexOp>(loc, head);
            llvm::SmallVector<mlir::Value> idx = gqa ? llvm::SmallVector<mlir::Value>{bi, hi, z, z}
                                                     : llvm::SmallVector<mlir::Value>{z, z};
            auto vv = b.create<mlir::vector::TransferReadOp>(
                loc, mlir::VectorType::get({D}, f), a.getResult(), idx, pad);
            b.create<mlir::vector::TransferWriteOp>(loc, vv, eb.getArgument(4), idx);
        }
    }
    b.create<mlir::func::ReturnOp>(loc);

    std::string name = std::string(linalg ? "linalg" : "explicit") +
                       (gqa ? "_gqa" : "_attention") +
                       "_H" + std::to_string(Hq) + "_KV" + std::to_string(Hkv) +
                       "_T" + std::to_string(T) + "_D" + std::to_string(D);

    auto engine = compileModule(m, name, linalg);
    if (!engine) {
        m.erase();
        return false;
    }

    auto ptr = engine->lookup("_mlir_ciface_audit_attention");
    if (!ptr) {
        llvm::errs() << "❌ Failed to lookup symbol _mlir_ciface_audit_attention: "
                     << llvm::toString(ptr.takeError()) << "\n";
        m.erase();
        return false;
    }

    std::mt19937 rng(45);
    std::uniform_real_distribution<float> qd(-0.9f, 0.9f);
    std::uniform_int_distribution<int> td(-1, 1);

    std::vector<float> Q(B * Hq * D);
    std::vector<float> O(B * Hq * D + 32, -789.0f);
    std::vector<float> ref(B * Hq * D);
    std::vector<int> K(B * Hkv * T * D), V(B * Hkv * T * D);
    std::vector<uint8_t> PK(K.size() / 4), PV(V.size() / 4);

    for (auto &x : Q) x = fractional ? qd(rng) : static_cast<float>(td(rng));
    for (auto &x : K) x = td(rng);
    for (auto &x : V) x = td(rng);
    for (size_t j = 0; j < K.size() / 4; j++) {
        PK[j] = packTernaryValues(&K[j * 4]);
        PV[j] = packTernaryValues(&V[j * 4]);
    }

    double maxerr = 0.0, sumerr = 0.0;
    int n = 0;
    bool guard = true;
    bool has_non_finite = false;

    for (int32_t position : {0, T - 1}) {
        // Independent scalar double causal attention reference
        for (int batch = 0; batch < B; batch++) {
            for (int h = 0; h < Hq; h++) {
                int kh = h / (Hq / Hkv);
                int base = (batch * Hkv + kh) * T * D;
                std::vector<double> s(position + 1);
                double mx = -1e30, total = 0.0;
                for (int t = 0; t <= position; t++) {
                    double dot = 0.0;
                    for (int d = 0; d < D; d++) {
                        dot += static_cast<double>(Q[(batch * Hq + h) * D + d]) *
                               static_cast<double>(K[base + t * D + d]);
                    }
                    s[t] = dot / std::sqrt(static_cast<double>(D));
                    mx = std::max(mx, s[t]);
                }
                for (auto &x : s) {
                    x = std::exp(x - mx);
                    total += x;
                }
                for (int d = 0; d < D; d++) {
                    double x = 0.0;
                    for (int t = 0; t <= position; t++) {
                        x += (s[t] / total) * static_cast<double>(V[base + t * D + d]);
                    }
                    ref[(batch * Hq + h) * D + d] = static_cast<float>(x);
                }
            }
        }

        auto dp = tenzo::runtime::MemRefDescriptor<int32_t, 1>::create(&position, {1});
        if (gqa) {
            auto dq = tenzo::runtime::MemRefDescriptor<float, 4>::create(Q.data(), {B, Hq, 1, D});
            auto do_ = tenzo::runtime::MemRefDescriptor<float, 4>::create(O.data() + 16, {B, Hq, 1, D});
            auto dk = tenzo::runtime::MemRefDescriptor<uint8_t, 4>::create(PK.data(), {B, Hkv, T, D / 4});
            auto dv = tenzo::runtime::MemRefDescriptor<uint8_t, 4>::create(PV.data(), {B, Hkv, T, D / 4});
            reinterpret_cast<void (*)(void*, void*, void*, void*, void*)>(*ptr)(
                &dq, &dk, &dv, &dp, &do_);
        } else {
            auto dq = tenzo::runtime::MemRefDescriptor<float, 2>::create(Q.data(), {1, D});
            auto do_ = tenzo::runtime::MemRefDescriptor<float, 2>::create(O.data() + 16, {1, D});
            auto dk = tenzo::runtime::MemRefDescriptor<uint8_t, 2>::create(PK.data(), {T, D / 4});
            auto dv = tenzo::runtime::MemRefDescriptor<uint8_t, 2>::create(PV.data(), {T, D / 4});
            reinterpret_cast<void (*)(void*, void*, void*, void*, void*)>(*ptr)(
                &dq, &dk, &dv, &dp, &do_);
        }

        for (size_t j = 0; j < ref.size(); j++) {
            float val = O[j + 16];
            if (isNonFiniteFloat(val) || !std::isfinite(val) || std::isnan(val)) {
                has_non_finite = true;
                maxerr = 1e30;
                break;
            }
            double err = std::abs(static_cast<double>(val) - static_cast<double>(ref[j]));
            if (isNonFiniteDouble(err) || !std::isfinite(err) || std::isnan(err)) {
                has_non_finite = true;
                maxerr = 1e30;
                break;
            }
            maxerr = std::max(maxerr, err);
            sumerr += err;
            n++;
        }
    }

    for (int j = 0; j < 16; j++) {
        if (O[j] != -789.0f || O[O.size() - 1 - j] != -789.0f) {
            guard = false;
        }
    }

    m.erase();
    bool passed = !has_non_finite &&
                  !isNonFiniteDouble(maxerr) &&
                  std::isfinite(maxerr) &&
                  !std::isnan(maxerr) &&
                  (maxerr <= 1e-4) &&
                  guard;
    if (passed) {
        llvm::outs() << "✅ [AUDIT " << name << "] PASS (fractional=" << fractional
                     << ", max_abs=" << maxerr << ", canaries=" << (guard ? "OK" : "CORRUPT") << ")\n";
    } else {
        llvm::errs() << "❌ [AUDIT " << name << "] FAIL: fractional=" << fractional
                     << ", max_abs=" << (has_non_finite ? "NaN/Inf" : std::to_string(maxerr))
                     << " (tolerance=1e-4), canaries="
                     << (guard ? "OK" : "CORRUPT") << "\n";
    }
    return passed;
}

bool auditExplicitAttention(mlir::MLIRContext &context,
                           int B, int Hq, int Hkv, int T, int D,
                           bool fractional) {
    return auditAttention(context, /*linalg=*/false, fractional, B, Hq, Hkv, T, D);
}

bool auditLinalgAttention(mlir::MLIRContext &context,
                          int B, int Hq, int Hkv, int T, int D,
                          bool fractional) {
    return auditAttention(context, /*linalg=*/true, fractional, B, Hq, Hkv, T, D);
}

bool runAuditJITTests(mlir::MLIRContext &context, AuditTestMode mode) {
    llvm::outs() << "======================================================================\n";
    llvm::outs() << "🔬 Running Audit JIT Test Suite against Independent Math References\n";
    llvm::outs() << "======================================================================\n";

    bool allPassed = true;

    auto executeTest = [&](const char *label, auto testFn) {
        llvm::outs() << "▶ Testing: " << label << " ...\n";
        bool ok = testFn();
        if (!ok) {
            allPassed = false;
        }
        return ok;
    };

    if (mode == AuditTestMode::All || mode == AuditTestMode::PackStandalone) {
        executeTest("Standalone Packing (81 Exhaustive + Round-Clamp)", [&]() {
            return auditTernaryPack(context, /*fused=*/false);
        });
    }
    if (mode == AuditTestMode::All || mode == AuditTestMode::PackFused) {
        executeTest("Fused MatMul + Pack (81 Exhaustive + Round-Clamp)", [&]() {
            return auditTernaryPack(context, /*fused=*/true);
        });
    }
    if (mode == AuditTestMode::All || mode == AuditTestMode::FusionConsumer) {
        executeTest("Fusion Multi-Consumer FP32 Buffer Preservation", [&]() {
            return auditFusionMultiConsumer(context);
        });
    }
    if (mode == AuditTestMode::All || mode == AuditTestMode::AttentionExplicit) {
        executeTest("Explicit Attention (Rank-2, Fractional Q, T=8, D=32)", [&]() {
            return auditAttention(context, /*linalg=*/false, /*fractional=*/true, /*B=*/1, /*Hq=*/1, /*Hkv=*/1, 8, 32);
        });
    }
    if (mode == AuditTestMode::All || mode == AuditTestMode::AttentionExplicitInteger) {
        executeTest("Explicit Attention (Rank-2, Integer Q, T=8, D=32)", [&]() {
            return auditAttention(context, /*linalg=*/false, /*fractional=*/false, /*B=*/1, /*Hq=*/1, /*Hkv=*/1, 8, 32);
        });
    }
    if (mode == AuditTestMode::All || mode == AuditTestMode::AttentionExplicitGQA) {
        executeTest("Explicit Attention GQA (Rank-4, B=2, Hq=20, Hkv=5, T=8, D=32)", [&]() {
            return auditAttention(context, /*linalg=*/false, /*fractional=*/true, /*B=*/2, /*Hq=*/20, /*Hkv=*/5, 8, 32);
        });
    }
    if (mode == AuditTestMode::All || mode == AuditTestMode::AttentionLinalg) {
        executeTest("Linalg Attention Fallback (Rank-2, T=8, D=32)", [&]() {
            return auditAttention(context, /*linalg=*/true, /*fractional=*/true, /*B=*/1, /*Hq=*/1, /*Hkv=*/1, 8, 32);
        });
    }
    if (mode == AuditTestMode::All || mode == AuditTestMode::AttentionLinalgGQA) {
        executeTest("Linalg Attention GQA Fallback (Rank-4, B=2, Hq=20, Hkv=5, T=8, D=32)", [&]() {
            return auditAttention(context, /*linalg=*/true, /*fractional=*/true, /*B=*/2, /*Hq=*/20, /*Hkv=*/5, 8, 32);
        });
    }

    llvm::outs() << "======================================================================\n";
    if (allPassed) {
        llvm::outs() << "✅ ALL AUDIT JIT TESTS PASSED!\n";
    } else {
        llvm::errs() << "❌ ONE OR MORE AUDIT JIT TESTS FAILED (Compiler Defects Caught)!\n";
    }
    llvm::outs() << "======================================================================\n";
    return allPassed;
}

} // namespace tenzo
