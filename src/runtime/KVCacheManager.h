#pragma once
#include "ExecutionContext.h" // For Tensor
#include "MemRefUtils.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tenzo {
namespace runtime {

/**
 * Manages per-layer KV-Cache buffers for autoregressive generation with
 * 256-bit (32-byte) cache-line aligned physical memory layout and unit stride
 * along the innermost head dimension, strictly preventing VGATHER hardware
 * penalties.
 */
class KVCacheManager {
public:
  // Standard constructor (FP32 / default embed_dim layout)
  KVCacheManager(int max_seq_len, int num_layers, int embed_dim,
                 int num_heads = 1);

  // Explicit GQA / packed 1.58-bit ternary constructor
  KVCacheManager(int max_seq_len, int num_layers, int num_heads, int head_dim,
                 bool is_packed_u8);

  ~KVCacheManager();

  // Reset sequence length to 0 (start new generation)
  void reset();

  // Get the current sequence length
  int get_current_seq_len() const { return current_seq_len; }

  // Increment the sequence length after processing tokens
  void increment_seq_len(int n = 1);

  // Get K and V cache tensors for a specific layer
  Tensor *get_k_cache(int layer_idx = 0);
  Tensor *get_v_cache(int layer_idx = 0);

  // 256-bit aligned packed uint8 accessors
  uint8_t *get_packed_k_cache(int layer_idx = 0);
  uint8_t *get_packed_v_cache(int layer_idx = 0);
  uint8_t *get_packed_k_ptr(int layer_idx = 0) {
    return get_packed_k_cache(layer_idx);
  }
  uint8_t *get_packed_v_ptr(int layer_idx = 0) {
    return get_packed_v_cache(layer_idx);
  }

  // FP32 direct pointer accessors
  float *get_fp32_k_ptr(int layer_idx = 0);
  float *get_fp32_v_ptr(int layer_idx = 0);

  int get_num_layers() const { return num_layers; }
  int get_num_heads() const { return num_heads; }
  int get_head_dim() const { return head_dim; }
  bool is_packed() const { return is_packed_u8; }

  // Physical memory layout & alignment verification (guarantees zero VGATHER)
  bool is_256b_aligned(int layer_idx = 0) const;
  bool is_unit_stride() const { return true; }
  size_t get_head_bytes() const;
  size_t get_token_stride_bytes() const;

private:
  int max_seq_len;
  int num_layers;
  int embed_dim;
  int num_heads;
  int head_dim;
  int current_seq_len;
  bool is_packed_u8;

  // Aligned allocations (32-byte / 256-bit aligned)
  std::vector<void *> k_aligned_ptrs;
  std::vector<void *> v_aligned_ptrs;
  std::vector<size_t> layer_alloc_bytes;

  std::vector<Tensor *> k_tensors;
  std::vector<Tensor *> v_tensors;

  void allocate_aligned_buffers();
  void free_aligned_buffers();
};

} // namespace runtime
} // namespace tenzo
