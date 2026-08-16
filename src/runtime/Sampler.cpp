#include "runtime/Sampler.h"
#include <numeric>
#include <cassert>

namespace tenzo {
namespace runtime {

Sampler::Sampler(const SamplerConfig& config)
    : config_(config), rng_(config.seed) {}

int32_t Sampler::sample(const float* logits, size_t vocab_size, const std::vector<int32_t>& past_tokens) {
    if (config_.temperature <= 0.0f) {
        return sample_greedy(logits, vocab_size);
    }

    struct TokenLogit {
        int32_t id;
        float val;
    };

    std::vector<TokenLogit> items(vocab_size);
    
    // Copy logits and apply repetition penalty
    float inv_temp = 1.0f / config_.temperature;
    for (size_t i = 0; i < vocab_size; ++i) {
        items[i] = {static_cast<int32_t>(i), logits[i]};
    }

    if (config_.repetition_penalty != 1.0f && !past_tokens.empty()) {
        for (int32_t tok_id : past_tokens) {
            if (tok_id >= 0 && static_cast<size_t>(tok_id) < vocab_size) {
                if (items[tok_id].val > 0.0f) {
                    items[tok_id].val /= config_.repetition_penalty;
                } else {
                    items[tok_id].val *= config_.repetition_penalty;
                }
            }
        }
    }

    // Top-K selection using std::partial_sort (O(K log K) instead of O(V log V))
    size_t k = (config_.top_k > 0 && static_cast<size_t>(config_.top_k) < vocab_size) 
               ? static_cast<size_t>(config_.top_k) : std::min<size_t>(vocab_size, 50);

    std::partial_sort(items.begin(), items.begin() + k, items.end(), 
                      [](const TokenLogit& a, const TokenLogit& b) {
                          return a.val > b.val;
                      });

    // Scale by temperature with max subtraction for numerical stability
    float max_val = items[0].val;
    float sum_exp = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        items[i].val = std::exp((items[i].val - max_val) * inv_temp);
        sum_exp += items[i].val;
    }

    float inv_sum = 1.0f / (sum_exp > 1e-9f ? sum_exp : 1.0f);
    for (size_t i = 0; i < k; ++i) {
        items[i].val *= inv_sum;
    }

    // Top-P (Nucleus) truncation
    size_t cutoff = k;
    if (config_.top_p > 0.0f && config_.top_p < 1.0f) {
        float cum_p = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            cum_p += items[i].val;
            if (cum_p >= config_.top_p) {
                cutoff = i + 1;
                break;
            }
        }
    }

    // Re-normalize probabilities over cutoff items
    float active_sum = 0.0f;
    for (size_t i = 0; i < cutoff; ++i) {
        active_sum += items[i].val;
    }

    std::uniform_real_distribution<float> dist(0.0f, active_sum);
    float r = dist(rng_);
    float cum = 0.0f;

    for (size_t i = 0; i < cutoff; ++i) {
        cum += items[i].val;
        if (r <= cum) {
            return items[i].id;
        }
    }

    return items[0].id;
}

int32_t Sampler::sample_greedy(const float* logits, size_t vocab_size) {
    int32_t max_id = 0;
    float max_val = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_id = static_cast<int32_t>(i);
        }
    }
    return max_id;
}

int32_t Sampler::sample_multinomial(const float* logits, size_t vocab_size) {
    return sample(logits, vocab_size);
}

} // namespace runtime
} // namespace tenzo
