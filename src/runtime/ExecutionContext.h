#pragma once
#include <vector>
#include <string>
#include <memory>
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

namespace tenzo {
namespace runtime {

/**
 * High-level Tensor abstraction for Runtime.
 */
struct Tensor {
    std::vector<int64_t> shape;
    float* data;
    bool is_owned = false;

    Tensor(const std::vector<int64_t>& s, float* d = nullptr) : shape(s), data(d) {
        if (!data) {
            size_t size = 1;
            for (auto dim : shape) size *= dim;
            data = static_cast<float*>(std::malloc(size * sizeof(float)));
            is_owned = true;
        }
    }

    ~Tensor() {
        if (is_owned && data) {
            std::free(data);
        }
    }

    // No copy, only move for safety
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept : shape(std::move(other.shape)), data(other.data), is_owned(other.is_owned) {
        other.data = nullptr;
        other.is_owned = false;
    }
};

/**
 * ExecutionContext: Manages JIT compilation and inference.
 */
class ExecutionContext {
public:
    ExecutionContext(mlir::MLIRContext& context, mlir::ModuleOp module);
    ~ExecutionContext() = default;

    // Run inference with dynamic inputs
    void forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);

private:
    std::unique_ptr<mlir::ExecutionEngine> engine;
};

} // namespace runtime
} // namespace tenzo
