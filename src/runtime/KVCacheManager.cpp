#include "KVCacheManager.h"
#include <stdexcept>

namespace tenzo {
namespace runtime {

KVCacheManager::KVCacheManager(int max_seq_len, int num_layers, int embed_dim, int num_heads)
    : max_seq_len(max_seq_len), num_layers(num_layers), embed_dim(embed_dim), current_seq_len(0) {
    
    size_t total_elements = static_cast<size_t>(num_layers) * max_seq_len * embed_dim;
    k_buffer.resize(total_elements, 0.0f);
    v_buffer.resize(total_elements, 0.0f);

    std::vector<int64_t> shape;
    if (num_heads > 1) {
        int head_dim = embed_dim / num_heads;
        shape = {num_layers, num_heads, max_seq_len, head_dim};
    } else {
        shape = {num_layers, max_seq_len, embed_dim};
    }
    k_tensor = new Tensor(shape, k_buffer.data());
    v_tensor = new Tensor(shape, v_buffer.data());
}

KVCacheManager::~KVCacheManager() {
    // k_tensor and v_tensor don't own the data, so they just get deleted
    delete k_tensor;
    delete v_tensor;
}

void KVCacheManager::reset() {
    current_seq_len = 0;
    // We don't strictly need to zero out the buffers, but doing so could prevent NaN propagation if bugs exist
    std::fill(k_buffer.begin(), k_buffer.end(), 0.0f);
    std::fill(v_buffer.begin(), v_buffer.end(), 0.0f);
}

void KVCacheManager::increment_seq_len(int n) {
    current_seq_len += n;
    if (current_seq_len > max_seq_len) {
        throw std::runtime_error("KV Cache overflow: current sequence length exceeded max_seq_len");
    }
}

Tensor* KVCacheManager::get_k_cache() {
    return k_tensor;
}

Tensor* KVCacheManager::get_v_cache() {
    return v_tensor;
}

} // namespace runtime
} // namespace tenzo
