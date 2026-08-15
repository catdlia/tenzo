#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

namespace tenzo {
namespace runtime {

class Tokenizer {
public:
    Tokenizer() = default;

    // Load vocabulary file exported by Python frontend (tokenizer.vocab text file)
    bool load(const std::string& vocab_path);

    // Encode text string to token IDs using Byte-Pair Encoding (or character byte fallback)
    std::vector<int32_t> encode(const std::string& text) const;

    // Decode token IDs back to text string
    std::string decode(const std::vector<int32_t>& tokens) const;

    size_t vocab_size() const { return id_to_token_.size(); }

private:
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> bpe_ranks_;
};

} // namespace runtime
} // namespace tenzo
