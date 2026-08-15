#pragma once
#include <vector>
#include <cstdint>
#include "MemRefUtils.h"
#include "ExecutionContext.h" // For Tensor

namespace tenzo {
namespace runtime {

/**
 * Manages the per-layer KV-Cache buffers for autoregressive generation.
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

    // Get K and V cache tensors for a specific layer
    Tensor* get_k_cache(int layer_idx = 0);
    Tensor* get_v_cache(int layer_idx = 0);

    int get_num_layers() const { return num_layers; }

private:
    int max_seq_len;
    int num_layers;
    int embed_dim;
    int current_seq_len;

    std::vector<std::vector<float>> k_buffers;
    std::vector<std::vector<float>> v_buffers;
    
    std::vector<Tensor*> k_tensors;
    std::vector<Tensor*> v_tensors;
};

} // namespace runtime
} // namespace tenzo
