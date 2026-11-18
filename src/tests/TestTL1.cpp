#include "context/TenzoContext.h"
#include "runtime/ExecutionContext.h"
#include "runtime/MemRefUtils.h"
#include "passes/Passes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <string>

using namespace mlir;

namespace tenzo {

static inline bool isNonFiniteFloat(float val) {
    uint32_t bits = 0;
    std::memcpy(&bits, &val, sizeof(float));
    return ((bits & 0x7F800000) == 0x7F800000);
}

bool runTestTL1(MLIRContext& context, const std::string& custom_weights_path = "") {
    llvm::outs() << "--- Running Test TL1 ---\n";

    // 1. Locate and parse MLIR module
    const char* default_mlir = R"(
module {
  func.func @main(%arg0: tensor<1x1x128xf32>, %arg1: memref<2x64x32xi8>) -> tensor<1x1x128xf32> attributes {llvm.emit_c_interface} {
    %scale = arith.constant 1.5 : f32
    %w_tens = bufferization.to_tensor %arg1 restrict : memref<2x64x32xi8> to tensor<2x64x32xi8>
    %out = "tenzo.bitlinear_tl1"(%arg0, %w_tens, %scale) : (tensor<1x1x128xf32>, tensor<2x64x32xi8>, f32) -> tensor<1x1x128xf32>
    return %out : tensor<1x1x128xf32>
  }
}
)";

    std::vector<std::string> mlir_candidates = {"/app/test_tl1.mlir", "test_tl1.mlir"};
    std::string mlir_path;
    for (const auto& p : mlir_candidates) {
        if (std::ifstream(p).is_open()) { mlir_path = p; break; }
    }

    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (!mlir_path.empty()) {
        module = mlir::parseSourceFile<mlir::ModuleOp>(mlir_path, &context);
    } else {
        llvm::SourceMgr sourceMgr;
        sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(default_mlir), llvm::SMLoc());
        module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    }

    if (!module) {
        llvm::errs() << "❌ [TL1-FAIL] Failed to load or parse test_tl1.mlir\n";
        return false;
    }

    // 2. Locate weights fixture file
    std::vector<std::string> weights_candidates;
    if (!custom_weights_path.empty()) {
        weights_candidates.push_back(custom_weights_path);
    } else {
        weights_candidates = {"/app/weights_tl1_test.bin", "weights_tl1_test.bin"};
    }

    std::string weights_path;
    for (const auto& p : weights_candidates) {
        if (std::ifstream(p).is_open()) { weights_path = p; break; }
    }
    if (weights_path.empty()) {
        llvm::errs() << "❌ [TL1-FAIL] [TL1-FIXTURE-INVALID] Failed to locate TL1 weights fixture ("
                     << (custom_weights_path.empty() ? "default" : custom_weights_path) << ")\n";
        return false;
    }

    std::ifstream file(weights_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        llvm::errs() << "❌ [TL1-FAIL] [TL1-FIXTURE-INVALID] Cannot open weights fixture: " << weights_path << "\n";
        return false;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 3. Strict Fixture Size Validation: 2 blocks x 64 k_half x 32 bytes = 4096 bytes
    constexpr size_t EXPECTED_TL1_SIZE = 4096;
    if (size != static_cast<std::streamsize>(EXPECTED_TL1_SIZE)) {
        llvm::errs() << "❌ [TL1-FAIL] [TL1-FIXTURE-INVALID] Corrupted TL1 fixture: size is " << size
                     << " bytes, expected exactly " << EXPECTED_TL1_SIZE << " bytes!\n";
        return false;
    }

    std::vector<uint8_t> weights(EXPECTED_TL1_SIZE);
    file.read(reinterpret_cast<char*>(weights.data()), EXPECTED_TL1_SIZE);
    if (!file) {
        llvm::errs() << "❌ [TL1-FAIL] [TL1-FIXTURE-INVALID] Failed to read full fixture content: " << weights_path << "\n";
        return false;
    }

    // 4. Validate TL1 packed nibble bounds: valid indices in [0, 8]
    for (size_t i = 0; i < EXPECTED_TL1_SIZE; ++i) {
        uint8_t b = weights[i];
        uint8_t low_nibble = b & 0x0F;
        uint8_t high_nibble = (b >> 4) & 0x0F;
        if (low_nibble > 8 || high_nibble > 8) {
            llvm::errs() << "❌ [TL1-FAIL] [TL1-FIXTURE-INVALID] Corrupted TL1 fixture: invalid nibble value at byte " << i
                         << " (low=" << int(low_nibble) << ", high=" << int(high_nibble) << " > 8)\n";
            return false;
        }
    }

    PassManager pm(&context);
    addTenzoToLinalgPass(pm);
    addTenzoBufferizationPasses(pm);
    addTenzoToLLVMPasses(pm, false, {8, 16, 4}, true);

    if (failed(pm.run(module.get()))) {
        llvm::errs() << "❌ [TL1-FAIL] MLIR compilation failed\n";
        return false;
    }

    // 5. Execute Forward Pass
    runtime::ExecutionContext engine(context, module.get());

    std::vector<int64_t> in_shape = {1, 1, 128};
    runtime::Tensor input(in_shape);
    for (int i = 0; i < 128; ++i) {
        input.data[i] = 2.0f; // Fill with 2.0
    }

    // Contract of tenzo.bitlinear_tl1 (BitLinearTL1LoweringToLinalg, runtime pack_ternary_weights_raw, export_bitnet.py):
    // weights are [n_blocks = N/64, K/2, 32] bytes. Here N = K = 128 -> [2, 64, 32] = 4096 bytes.
    std::vector<int64_t> w_shape = {2, 64, 32};
    runtime::Tensor w_tensor(w_shape); // Use default float Tensor but we'll cast its memory
    std::memcpy(reinterpret_cast<int8_t*>(w_tensor.data), weights.data(), weights.size());

    std::vector<int64_t> out_shape = {1, 1, 128};
    runtime::Tensor output(out_shape);

    std::vector<runtime::Tensor*> inputs = {&input, &w_tensor};
    std::vector<runtime::Tensor*> outputs = {&output};

    engine.forward(inputs, outputs);

    llvm::outs() << "Output: ";
    for (int i = 0; i < 10; ++i) {
        llvm::outs() << output.data[i] << " ";
    }
    llvm::outs() << "\n";

    // 6. Assert forward outputs are finite and non-zero
    bool nonZero = false;
    for (int i = 0; i < 128; ++i) {
        float val = output.data[i];
        if (isNonFiniteFloat(val) || std::isnan(val) || !std::isfinite(val)) {
            llvm::errs() << "❌ [TL1-FAIL] Non-finite output in TL1 forward pass at index " << i << ": " << val << "\n";
            return false;
        }
        if (std::abs(val) > 1e-6f) {
            nonZero = true;
        }
    }

    if (!nonZero) {
        llvm::errs() << "❌ [TL1-FAIL] TL1 forward pass produced all-zero output!\n";
        return false;
    }

    llvm::outs() << "✅ [PASS] TL1 Fixture execution verified correct.\n";
    return true;
}

} // namespace tenzo
