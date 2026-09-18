#include "KVCacheManager.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace tenzo {
namespace runtime {

KVCacheManager::KVCacheManager(int max_seq_len, int num_layers, int embed_dim,
                               int num_heads)
    : max_seq_len(max_seq_len), num_layers(num_layers), embed_dim(embed_dim),
      num_heads(num_heads > 0 ? num_heads : 1),
      head_dim((num_heads > 0) ? (embed_dim / num_heads) : embed_dim),
      current_seq_len(0), is_packed_u8(false) {
  allocate_aligned_buffers();
}

KVCacheManager::KVCacheManager(int max_seq_len, int num_layers, int num_heads,
                               int head_dim, bool is_packed_u8)
    : max_seq_len(max_seq_len), num_layers(num_layers),
      embed_dim(num_heads * head_dim), num_heads(num_heads > 0 ? num_heads : 1),
      head_dim(head_dim), current_seq_len(0), is_packed_u8(is_packed_u8) {
  allocate_aligned_buffers();
}

void KVCacheManager::allocate_aligned_buffers() {
  layer_alloc_bytes.resize(num_layers);
  k_aligned_ptrs.resize(num_layers, nullptr);
  v_aligned_ptrs.resize(num_layers, nullptr);

  size_t head_byte_size = is_packed_u8
                              ? static_cast<size_t>(head_dim / 4)
                              : static_cast<size_t>(head_dim * sizeof(float));
  // Pad head_byte_size to 32 bytes to ensure every head and token starts on a
  // 256-bit boundary
  size_t aligned_head_bytes = (head_byte_size + 31) & ~static_cast<size_t>(31);
  size_t total_layer_bytes =
      static_cast<size_t>(num_heads) * max_seq_len * aligned_head_bytes;

  std::vector<int64_t> shape;
  if (num_heads > 1) {
    int d = is_packed_u8 ? (head_dim / 4) : head_dim;
    shape = {1, num_heads, max_seq_len, d};
  } else {
    int d = is_packed_u8 ? (embed_dim / 4) : embed_dim;
    shape = {1, max_seq_len, d};
  }

  for (int l = 0; l < num_layers; ++l) {
    layer_alloc_bytes[l] = total_layer_bytes;
    void *k_ptr = nullptr;
    void *v_ptr = nullptr;
    if (posix_memalign(&k_ptr, 32, total_layer_bytes) != 0 ||
        posix_memalign(&v_ptr, 32, total_layer_bytes) != 0) {
      throw std::bad_alloc();
    }
    std::memset(k_ptr, 0, total_layer_bytes);
    std::memset(v_ptr, 0, total_layer_bytes);

    k_aligned_ptrs[l] = k_ptr;
    v_aligned_ptrs[l] = v_ptr;

    k_tensors.push_back(new Tensor(shape, static_cast<float *>(k_ptr)));
    v_tensors.push_back(new Tensor(shape, static_cast<float *>(v_ptr)));
  }
}

void KVCacheManager::free_aligned_buffers() {
  for (void *p : k_aligned_ptrs) {
    if (p)
      std::free(p);
  }
  k_aligned_ptrs.clear();

  for (void *p : v_aligned_ptrs) {
    if (p)
      std::free(p);
  }
  v_aligned_ptrs.clear();
}

KVCacheManager::~KVCacheManager() {
  for (auto *t : k_tensors)
    delete t;
  for (auto *t : v_tensors)
    delete t;
  k_tensors.clear();
  v_tensors.clear();
  free_aligned_buffers();
}

void KVCacheManager::reset() {
  current_seq_len = 0;
  for (int l = 0; l < num_layers; ++l) {
    if (k_aligned_ptrs[l])
      std::memset(k_aligned_ptrs[l], 0, layer_alloc_bytes[l]);
    if (v_aligned_ptrs[l])
      std::memset(v_aligned_ptrs[l], 0, layer_alloc_bytes[l]);
  }
}

void KVCacheManager::increment_seq_len(int n) {
  current_seq_len += n;
  if (current_seq_len > max_seq_len) {
    throw std::runtime_error(
        "KV Cache overflow: current sequence length exceeded max_seq_len");
  }
}

Tensor *KVCacheManager::get_k_cache(int layer_idx) {
  if (layer_idx < 0 || layer_idx >= num_layers) {
    throw std::out_of_range("Invalid layer index in get_k_cache");
  }
  return k_tensors[layer_idx];
}

Tensor *KVCacheManager::get_v_cache(int layer_idx) {
  if (layer_idx < 0 || layer_idx >= num_layers) {
    throw std::out_of_range("Invalid layer index in get_v_cache");
  }
  return v_tensors[layer_idx];
}

uint8_t *KVCacheManager::get_packed_k_cache(int layer_idx) {
  if (layer_idx < 0 || layer_idx >= num_layers) {
    throw std::out_of_range("Invalid layer index in get_packed_k_cache");
  }
  return static_cast<uint8_t *>(k_aligned_ptrs[layer_idx]);
}

uint8_t *KVCacheManager::get_packed_v_cache(int layer_idx) {
  if (layer_idx < 0 || layer_idx >= num_layers) {
    throw std::out_of_range("Invalid layer index in get_packed_v_cache");
  }
  return static_cast<uint8_t *>(v_aligned_ptrs[layer_idx]);
}

float *KVCacheManager::get_fp32_k_ptr(int layer_idx) {
  if (layer_idx < 0 || layer_idx >= num_layers) {
    throw std::out_of_range("Invalid layer index in get_fp32_k_ptr");
  }
  return static_cast<float *>(k_aligned_ptrs[layer_idx]);
}

float *KVCacheManager::get_fp32_v_ptr(int layer_idx) {
  if (layer_idx < 0 || layer_idx >= num_layers) {
    throw std::out_of_range("Invalid layer index in get_fp32_v_ptr");
  }
  return static_cast<float *>(v_aligned_ptrs[layer_idx]);
}

bool KVCacheManager::is_256b_aligned(int layer_idx) const {
  if (layer_idx < 0 || layer_idx >= num_layers)
    return false;
  uintptr_t k_addr = reinterpret_cast<uintptr_t>(k_aligned_ptrs[layer_idx]);
  uintptr_t v_addr = reinterpret_cast<uintptr_t>(v_aligned_ptrs[layer_idx]);
  if ((k_addr % 32) != 0 || (v_addr % 32) != 0)
    return false;

  // Check token stride alignment
  size_t head_bytes = get_head_bytes();
  if ((head_bytes % 32) != 0)
    return false;

  return true;
}

size_t KVCacheManager::get_head_bytes() const {
  return is_packed_u8 ? static_cast<size_t>(head_dim / 4)
                      : static_cast<size_t>(head_dim * sizeof(float));
}

size_t KVCacheManager::get_token_stride_bytes() const {
  return get_head_bytes();
}

} // namespace runtime
} // namespace tenzo
