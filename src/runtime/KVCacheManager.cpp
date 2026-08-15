#include "KVCacheManager.h"
#include <stdexcept>

namespace tenzo {
namespace runtime {

KVCacheManager::KVCacheManager(int max_seq_len, int num_layers, int embed_dim, int num_heads)
    : max_seq_len(max_seq_len), num_layers(num_layers), embed_dim(embed_dim), current_seq_len(0) {
    
    size_t layer_elements = static_cast<size_t>(max_seq_len) * embed_dim;
    k_buffers.resize(num_layers, std::vector<float>(layer_elements, 0.0f));
    v_buffers.resize(num_layers, std::vector<float>(layer_elements, 0.0f));

    std::vector<int64_t> shape;
    if (num_heads > 1) {
        int head_dim = embed_dim / num_heads;
        shape = {1, num_heads, max_seq_len, head_dim};
    } else {
        shape = {1, max_seq_len, embed_dim};
    }

    for (int l = 0; l < num_layers; ++l) {
        k_tensors.push_back(new Tensor(shape, k_buffers[l].data()));
        v_tensors.push_back(new Tensor(shape, v_buffers[l].data()));
    }
}

KVCacheManager::~KVCacheManager() {
    for (auto* t : k_tensors) delete t;
    for (auto* t : v_tensors) delete t;
}

void KVCacheManager::reset() {
    current_seq_len = 0;
    for (int l = 0; l < num_layers; ++l) {
        std::fill(k_buffers[l].begin(), k_buffers[l].end(), 0.0f);
        std::fill(v_buffers[l].begin(), v_buffers[l].end(), 0.0f);
    }
}

void KVCacheManager::increment_seq_len(int n) {
    current_seq_len += n;
    if (current_seq_len > max_seq_len) {
        throw std::runtime_error("KV Cache overflow: current sequence length exceeded max_seq_len");
    }
}

Tensor* KVCacheManager::get_k_cache(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= num_layers) {
        throw std::out_of_range("Invalid layer index in get_k_cache");
    }
    return k_tensors[layer_idx];
}

Tensor* KVCacheManager::get_v_cache(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= num_layers) {
        throw std::out_of_range("Invalid layer index in get_v_cache");
    }
    return v_tensors[layer_idx];
}

} // namespace runtime
} // namespace tenzo
