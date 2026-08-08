#pragma once
#include <vector>
#include <cstdint>
#include "MemRefUtils.h"
#include "ExecutionContext.h" // For Tensor

namespace tenzo {
namespace runtime {

/**
 * Manages the KV-Cache buffers for autoregressive generation.
 */
class KVCacheManager {
public:
    KVCacheManager(int max_seq_len, int num_layers, int embed_dim, int num_heads = 1);
    ~KVCacheManager();

    // Reset sequence length to 0 (start new generation)
    void reset();

    // Get the current sequence length
    int get_current_seq_len() const { return current_seq_len; }

    // Increment the sequence length after processing tokens
    void increment_seq_len(int n = 1);

    // Get the entire K and V cache tensors
    Tensor* get_k_cache();
    Tensor* get_v_cache();

private:
    int max_seq_len;
    int num_layers;
    int embed_dim;
    int current_seq_len;

    std::vector<float> k_buffer;
    std::vector<float> v_buffer;
    
    Tensor* k_tensor;
    Tensor* v_tensor;
};

} // namespace runtime
} // namespace tenzo
