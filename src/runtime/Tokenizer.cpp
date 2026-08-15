#include "runtime/Tokenizer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace tenzo {
namespace runtime {

bool Tokenizer::load(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        return false;
    }

    token_to_id_.clear();
    id_to_token_.clear();
    bpe_ranks_.clear();

    std::string line;
    int rank = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream ss(line);
        int32_t id;
        std::string token;
        if (ss >> id >> token) {
            if (id >= static_cast<int32_t>(id_to_token_.size())) {
                id_to_token_.resize(id + 1);
            }
            // Replace escaped spaces if present
            std::string clean_token = token;
            size_t pos = 0;
            while ((pos = clean_token.find("Ġ", pos)) != std::string::npos) {
                clean_token.replace(pos, 2, " ");
            }
            id_to_token_[id] = clean_token;
            token_to_id_[clean_token] = id;
            bpe_ranks_[clean_token] = rank++;
        }
    }
    return !id_to_token_.empty();
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> tokens;
    if (text.empty() || id_to_token_.empty()) return tokens;

    // Add BOS token if present (128000 for Llama3/BitNet)
    auto bos_it = token_to_id_.find("<|begin_of_text|>");
    if (bos_it != token_to_id_.end()) {
        tokens.push_back(bos_it->second);
    } else if (id_to_token_.size() > 128000) {
        tokens.push_back(128000);
    }

    // Greedy longest-first matching with leading space support
    size_t i = 0;
    while (i < text.length()) {
        size_t longest_len = 0;
        int32_t best_id = -1;

        for (size_t len = text.length() - i; len >= 1; --len) {
            std::string sub = text.substr(i, len);
            
            // Direct match
            auto it = token_to_id_.find(sub);
            if (it != token_to_id_.end()) {
                longest_len = len;
                best_id = it->second;
                break;
            }

            // Match with leading space if word boundary
            if (i == 0 || text[i-1] == ' ') {
                auto it_sp = token_to_id_.find(" " + sub);
                if (it_sp != token_to_id_.end()) {
                    longest_len = len;
                    best_id = it_sp->second;
                    break;
                }
            }
        }

        if (best_id != -1) {
            tokens.push_back(best_id);
            i += longest_len;
        } else {
            // Byte fallback or unknown character
            std::string single_char = text.substr(i, 1);
            auto it = token_to_id_.find(single_char);
            if (it != token_to_id_.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(0); // Unknown token
            }
            i += 1;
        }
    }

    return tokens;
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string text;
    for (int32_t id : tokens) {
        if (id >= 0 && id < static_cast<int32_t>(id_to_token_.size())) {
            text += id_to_token_[id];
        }
    }
    return text;
}

} // namespace runtime
} // namespace tenzo
