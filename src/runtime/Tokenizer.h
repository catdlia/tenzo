#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tenzo {
namespace runtime {

class Tokenizer {
public:
    Tokenizer() = default;

    // Load HF tokenizer.json (preferred) or legacy tokenizer.vocab.
    // When given a tokenizer.vocab or directory path, searches:
    //   1. <path> directly if .json
    //   2. <dir>/tokenizer.json
    //   3. TENZO_TOKENIZER_JSON environment variable (when vocab >= 128k or explicit)
    //   4. tools/tokenizer/bitnet_b158_tokenizer.json (when vocab >= 128k)
    // Falls back to legacy tokenizer.vocab for small mock/unit-test vocabularies.
    bool load(const std::string& path);

    // Encode UTF-8 text to token IDs using HF Llama-3 BPE (or legacy fallback).
    // When add_bos is true (default), prepends BOS token ID in BPE mode (matching HF).
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true) const;

    // Decode token IDs back to exact UTF-8 byte sequence.
    std::string decode(const std::vector<int32_t>& tokens) const;

    // Decode a single token ID back to its raw bytes/UTF-8 fragment.
    std::string decode_token(int32_t token_id) const;

    bool is_stop_token(int32_t token_id) const;
    bool uses_bpe() const { return bpe_loaded_; }
    size_t vocab_size() const { return id_to_decoded_.size(); }
    int32_t bos_token_id() const { return bos_token_id_; }
    int32_t eos_token_id() const { return eos_token_id_; }
    int32_t eot_token_id() const { return eot_token_id_; }

private:
    struct MergeEntry {
        uint32_t rank = 0;
        int32_t merged_id = -1;
    };

    struct AddedToken {
        std::string content;
        int32_t id = -1;
        bool special = false;
    };

    void clear();
    bool load_json_file(const std::string& json_path);
    bool load_legacy_vocab(const std::string& vocab_path);
    void encode_bpe_piece(const uint8_t* bytes, size_t len, std::vector<int32_t>& out) const;
    void encode_segment(const std::string& seg, std::vector<int32_t>& out) const;
    void set_decoded_entry(int32_t id, std::string decoded, bool is_special);

    bool bpe_loaded_ = false;
    bool ignore_merges_ = true;
    int32_t bos_token_id_ = 128000;
    int32_t eos_token_id_ = 128001;
    int32_t eot_token_id_ = 128009;
    size_t max_raw_token_bytes_ = 0;

    std::unordered_map<std::string, int32_t> raw_vocab_to_id_;
    std::unordered_map<uint64_t, MergeEntry> merges_;
    std::vector<int32_t> byte_to_sym_id_;
    std::vector<AddedToken> added_tokens_;
    bool added_first_byte_[256] = {};

    std::unordered_map<std::string, int32_t> legacy_token_to_id_;

    std::vector<std::string> id_to_decoded_;
    std::vector<uint8_t> id_is_valid_;
    std::vector<uint8_t> id_is_special_;
};

} // namespace runtime
} // namespace tenzo
