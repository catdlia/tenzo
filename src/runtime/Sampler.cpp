#include "runtime/Sampler.h"
#include <numeric>
#include <cassert>

namespace tenzo {
namespace runtime {

Sampler::Sampler(const SamplerConfig& config)
    : config_(config), rng_(config.seed) {}

int32_t Sampler::sample(const float* logits, size_t vocab_size) {
    if (config_.temperature <= 0.0f) {
        return sample_greedy(logits, vocab_size);
    }
    return sample_multinomial(logits, vocab_size);
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
    struct TokenLogit {
        int32_t id;
        float val;
    };

    std::vector<TokenLogit> items(vocab_size);
    float inv_temp = 1.0f / config_.temperature;
    
    float max_val = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    for (size_t i = 0; i < vocab_size; ++i) {
        items[i] = {static_cast<int32_t>(i), (logits[i] - max_val) * inv_temp};
    }

    // Sort descending by scaled logit
    std::sort(items.begin(), items.end(), [](const TokenLogit& a, const TokenLogit& b) {
        return a.val > b.val;
    });

    // Compute softmax probabilities
    float sum_exp = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        items[i].val = std::exp(items[i].val);
        sum_exp += items[i].val;
    }

    for (size_t i = 0; i < vocab_size; ++i) {
        items[i].val /= sum_exp;
    }

    // Top-K truncation
    size_t cutoff = vocab_size;
    if (config_.top_k > 0 && static_cast<size_t>(config_.top_k) < cutoff) {
        cutoff = static_cast<size_t>(config_.top_k);
    }

    // Top-P (Nucleus) truncation
    if (config_.top_p > 0.0f && config_.top_p < 1.0f) {
        float cum_p = 0.0f;
        for (size_t i = 0; i < cutoff; ++i) {
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

} // namespace runtime
} // namespace tenzo
