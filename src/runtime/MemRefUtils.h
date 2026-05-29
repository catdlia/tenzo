#pragma once
#include <cstdint>
#include <vector>

namespace tenzo {
namespace runtime {

/**
 * MLIR StridedMemRefType descriptor for C++ interop.
 * This matches the ABI used by MLIR's ExecutionEngine.
 */
template <typename T, std::size_t Rank>
struct MemRefDescriptor {
    T* allocated;
    T* aligned;
    intptr_t offset;
    intptr_t sizes[Rank];
    intptr_t strides[Rank];

    // Helper to create a descriptor from a flat pointer and shape
    static MemRefDescriptor create(T* ptr, const std::vector<int64_t>& shape) {
        MemRefDescriptor desc;
        desc.allocated = ptr;
        desc.aligned = ptr;
        desc.offset = 0;
        
        for (std::size_t i = 0; i < Rank; ++i) {
            desc.sizes[i] = shape[i];
        }

        // Compute row-major strides
        intptr_t stride = 1;
        for (int i = Rank - 1; i >= 0; --i) {
            desc.strides[i] = stride;
            stride *= shape[i];
        }
        return desc;
    }
};

using MemRef2D = MemRefDescriptor<float, 2>;

} // namespace runtime
} // namespace tenzo
