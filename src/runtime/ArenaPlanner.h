#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include "runtime/MemRefUtils.h"

namespace tenzo {

class ArenaPlanner {
public:
    explicit ArenaPlanner(size_t total_size_bytes = 64 * 1024 * 1024) 
        : arena_size(total_size_bytes), buffer_size(total_size_bytes / 2) {
        
        // Allocate contiguous arena memory
        arena_ptr = static_cast<uint8_t*>(std::malloc(arena_size));
        if (!arena_ptr) {
            throw std::runtime_error("ArenaPlanner: Failed to allocate arena memory!");
        }

        buffer_A = arena_ptr;
        buffer_B = arena_ptr + buffer_size;
        current_read_buffer = buffer_A;
        current_write_buffer = buffer_B;
    }

    ~ArenaPlanner() {
        if (arena_ptr) {
            std::free(arena_ptr);
            arena_ptr = nullptr;
        }
    }

    // Ping-Pong buffer swap
    void swap_buffers() {
        std::swap(current_read_buffer, current_write_buffer);
    }

    uint8_t* get_read_buffer() const { return current_read_buffer; }
    uint8_t* get_write_buffer() const { return current_write_buffer; }

    // Helper to construct a MemRefDescriptor pointing to the current write buffer
    runtime::MemRefDescriptor<float, 2> create_write_view(const std::vector<int64_t>& shape) {
        std::vector<intptr_t> strides(shape.size());
        intptr_t current_stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = current_stride;
            current_stride *= shape[i];
        }

        return runtime::MemRefDescriptor<float, 2>::create_view(
            reinterpret_cast<float*>(current_write_buffer),
            shape,
            strides,
            0
        );
    }

    void reset() {
        current_read_buffer = buffer_A;
        current_write_buffer = buffer_B;
    }

private:
    size_t arena_size;
    size_t buffer_size;
    uint8_t* arena_ptr = nullptr;
    uint8_t* buffer_A = nullptr;
    uint8_t* buffer_B = nullptr;
    uint8_t* current_read_buffer = nullptr;
    uint8_t* current_write_buffer = nullptr;
};

} // namespace tenzo
