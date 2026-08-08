#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>

namespace tenzo {
namespace runtime {

struct SamplerConfig {
    float temperature = 0.0f; // 0.0f = Greedy search (ArgMax)
    float top_p = 0.9f;
    int top_k = 40;
    uint64_t seed = 42;
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& config = SamplerConfig());

    // Given unnormalized logits array of size vocab_size, sample next token ID
    int32_t sample(const float* logits, size_t vocab_size);

    void set_temperature(float temp) { config_.temperature = temp; }
    void set_top_p(float p) { config_.top_p = p; }

private:
    SamplerConfig config_;
    std::mt19937 rng_;

    int32_t sample_greedy(const float* logits, size_t vocab_size);
    int32_t sample_multinomial(const float* logits, size_t vocab_size);
};

} // namespace runtime
} // namespace tenzo
