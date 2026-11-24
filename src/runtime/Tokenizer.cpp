#include "runtime/Tokenizer.h"
#include "runtime/UnicodeTables.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

namespace tenzo {
namespace runtime {
namespace {

struct GPT2ByteTables {
    std::array<int16_t, 0x144> u2b{};
    GPT2ByteTables() {
        u2b.fill(-1);
        std::array<bool, 256> direct{};
        for (int b = '!'; b <= '~'; ++b) direct[b] = true;
        for (int b = 0xA1; b <= 0xAC; ++b) direct[b] = true;
        for (int b = 0xAE; b <= 0xFF; ++b) direct[b] = true;
        for (int b = 0; b < 256; ++b) {
            if (direct[b]) u2b[b] = static_cast<int16_t>(b);
        }
        uint32_t next_cp = 256;
        for (int b = 0; b < 256; ++b) {
            if (!direct[b]) {
                u2b[next_cp++] = static_cast<int16_t>(b);
            }
        }
    }
};

const GPT2ByteTables& get_gpt2_tables() {
    static const GPT2ByteTables tables;
    return tables;
}

inline void append_utf8(uint32_t cp, std::string& out) {
    if (cp <= 0x7Fu) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

inline bool next_utf8_cp(const std::string& s, size_t& idx, uint32_t& cp) {
    if (idx >= s.size()) return false;
    const uint8_t b0 = static_cast<uint8_t>(s[idx]);
    if (b0 < 0x80u) {
        cp = b0;
        idx += 1;
        return true;
    }
    auto is_cont = [&](size_t pos) -> bool {
        return pos < s.size() && (static_cast<uint8_t>(s[pos]) & 0xC0u) == 0x80u;
    };
    if ((b0 & 0xE0u) == 0xC0u && b0 >= 0xC2u && is_cont(idx + 1)) {
        cp = ((b0 & 0x1Fu) << 6) | (static_cast<uint8_t>(s[idx + 1]) & 0x3Fu);
        idx += 2;
        return true;
    }
    if ((b0 & 0xF0u) == 0xE0u && is_cont(idx + 1) && is_cont(idx + 2)) {
        const uint8_t b1 = static_cast<uint8_t>(s[idx + 1]);
        const uint32_t cand = ((b0 & 0x0Fu) << 12) |
                              ((b1 & 0x3Fu) << 6) |
                              (static_cast<uint8_t>(s[idx + 2]) & 0x3Fu);
        if (cand >= 0x800u && !(cand >= 0xD800u && cand <= 0xDFFFu)) {
            cp = cand;
            idx += 3;
            return true;
        }
    }
    if ((b0 & 0xF8u) == 0xF0u && b0 <= 0xF4u && is_cont(idx + 1) && is_cont(idx + 2) && is_cont(idx + 3)) {
        const uint32_t cand = ((b0 & 0x07u) << 18) |
                              ((static_cast<uint8_t>(s[idx + 1]) & 0x3Fu) << 12) |
                              ((static_cast<uint8_t>(s[idx + 2]) & 0x3Fu) << 6) |
                              (static_cast<uint8_t>(s[idx + 3]) & 0x3Fu);
        if (cand >= 0x10000u && cand <= 0x10FFFFu) {
            cp = cand;
            idx += 4;
            return true;
        }
    }
    cp = 0xFFFDu;
    idx += 1;
    return true;
}

// Convert a GPT-2 ByteLevel token string back to its raw underlying bytes.
std::string gpt2_token_to_raw_bytes(const std::string& tok) {
    const auto& tbl = get_gpt2_tables();
    std::string out;
    out.reserve(tok.size());
    size_t i = 0;
    uint32_t cp = 0;
    while (next_utf8_cp(tok, i, cp)) {
        if (cp < tbl.u2b.size() && tbl.u2b[cp] >= 0) {
            out.push_back(static_cast<char>(static_cast<uint8_t>(tbl.u2b[cp])));
        } else {
            append_utf8(cp, out);
        }
    }
    return out;
}

// Minimal, strict, dependency-free JSON reader for HF tokenizer.json.
class JsonCursor {
public:
    explicit JsonCursor(const std::string& s) : src_(s) {}

    void skip_ws() {
        while (pos_ < src_.size()) {
            const char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    char peek() {
        skip_ws();
        return pos_ < src_.size() ? src_[pos_] : '\0';
    }

    bool consume(char expected) {
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == expected) {
            ++pos_;
            return true;
        }
        return false;
    }

    bool parse_string(std::string& out) {
        skip_ws();
        if (pos_ >= src_.size() || src_[pos_] != '"') return false;
        ++pos_;
        out.clear();
        while (pos_ < src_.size()) {
            const char c = src_[pos_++];
            if (c == '"') return true;
            if (c == '\\') {
                if (pos_ >= src_.size()) return false;
                const char esc = src_[pos_++];
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        uint32_t u1 = 0;
                        if (!parse_hex4(u1)) return false;
                        if (u1 >= 0xD800u && u1 <= 0xDBFFu &&
                            pos_ + 6 <= src_.size() && src_[pos_] == '\\' && src_[pos_ + 1] == 'u') {
                            size_t saved = pos_;
                            pos_ += 2;
                            uint32_t u2 = 0;
                            if (parse_hex4(u2) && u2 >= 0xDC00u && u2 <= 0xDFFFu) {
                                uint32_t full = 0x10000u + (((u1 - 0xD800u) << 10) | (u2 - 0xDC00u));
                                append_utf8(full, out);
                            } else {
                                pos_ = saved;
                                append_utf8(0xFFFDu, out);
                            }
                        } else {
                            append_utf8(u1, out);
                        }
                        break;
                    }
                    default:
                        return false;
                }
            } else {
                out.push_back(c);
            }
        }
        return false;
    }

    bool parse_int(int64_t& out) {
        skip_ws();
        if (pos_ >= src_.size()) return false;
        bool neg = false;
        if (src_[pos_] == '-') {
            neg = true;
            ++pos_;
        }
        if (pos_ >= src_.size() || !std::isdigit(static_cast<unsigned char>(src_[pos_]))) return false;
        uint64_t val = 0;
        while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
            val = val * 10 + static_cast<uint64_t>(src_[pos_++] - '0');
        }
        out = neg ? -static_cast<int64_t>(val) : static_cast<int64_t>(val);
        return true;
    }

    bool parse_bool(bool& out) {
        skip_ws();
        if (src_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            out = true;
            return true;
        }
        if (src_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            out = false;
            return true;
        }
        return false;
    }

    bool skip_value() {
        const char ch = peek();
        if (ch == '\0') return false;
        if (ch == '"') {
            std::string dummy;
            return parse_string(dummy);
        }
        if (ch == '{') {
            ++pos_;
            if (consume('}')) return true;
            std::string key;
            while (true) {
                if (!parse_string(key) || !consume(':') || !skip_value()) return false;
                if (consume('}')) return true;
                if (!consume(',')) return false;
            }
        }
        if (ch == '[') {
            ++pos_;
            if (consume(']')) return true;
            while (true) {
                if (!skip_value()) return false;
                if (consume(']')) return true;
                if (!consume(',')) return false;
            }
        }
        // null, true, false, or number
        while (pos_ < src_.size()) {
            const char c = src_[pos_];
            if (c == ',' || c == '}' || c == ']' || c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                return true;
            }
            ++pos_;
        }
        return true;
    }

private:
    bool parse_hex4(uint32_t& val) {
        if (pos_ + 4 > src_.size()) return false;
        val = 0;
        for (int i = 0; i < 4; ++i) {
            const unsigned char c = static_cast<unsigned char>(src_[pos_++]);
            val <<= 4;
            if (c >= '0' && c <= '9') val |= (c - '0');
            else if (c >= 'a' && c <= 'f') val |= (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') val |= (c - 'A' + 10);
            else return false;
        }
        return true;
    }

    const std::string& src_;
    size_t pos_ = 0;
};

struct CPInfo {
    uint32_t cp;
    size_t byte_off;
};

inline bool is_cr_lf(uint32_t cp) {
    return cp == '\r' || cp == '\n';
}

inline uint32_t ascii_lower(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') ? (cp + 32u) : cp;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool file_readable(const std::string& p) {
    std::ifstream f(p);
    return f.good();
}

std::string parent_dir(const std::string& p) {
    size_t slash = p.find_last_of("/\\");
    return slash == std::string::npos ? "." : p.substr(0, slash);
}

} // namespace

void Tokenizer::clear() {
    bpe_loaded_ = false;
    ignore_merges_ = true;
    bos_token_id_ = 128000;
    eos_token_id_ = 128001;
    eot_token_id_ = 128009;
    max_raw_token_bytes_ = 0;
    raw_vocab_to_id_.clear();
    merges_.clear();
    byte_to_sym_id_.clear();
    added_tokens_.clear();
    std::fill(std::begin(added_first_byte_), std::end(added_first_byte_), false);
    legacy_token_to_id_.clear();
    id_to_decoded_.clear();
    id_is_valid_.clear();
    id_is_special_.clear();
}

void Tokenizer::set_decoded_entry(int32_t id, std::string decoded, bool is_special) {
    if (id < 0) return;
    const size_t uid = static_cast<size_t>(id);
    if (uid >= id_to_decoded_.size()) {
        id_to_decoded_.resize(uid + 1);
        id_is_valid_.resize(uid + 1, 0);
        id_is_special_.resize(uid + 1, 0);
    }
    id_to_decoded_[uid] = std::move(decoded);
    id_is_valid_[uid] = 1;
    if (is_special) id_is_special_[uid] = 1;
}

bool Tokenizer::load_json_file(const std::string& json_path) {
    std::ifstream file(json_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;
    const std::streamsize sz = file.tellg();
    if (sz <= 0) return false;
    std::string content(static_cast<size_t>(sz), '\0');
    file.seekg(0, std::ios::beg);
    if (!file.read(&content[0], sz)) return false;

    clear();
    JsonCursor cur(content);
    if (!cur.consume('{')) return false;

    std::vector<std::pair<std::string, std::string>> raw_merges;
    std::unordered_map<std::string, int32_t> sym_to_id;
    int32_t next_synth_id = 0;

    if (!cur.consume('}')) {
        std::string top_key;
        while (true) {
            if (!cur.parse_string(top_key) || !cur.consume(':')) return false;

            if (top_key == "added_tokens") {
                if (!cur.consume('[')) return false;
                if (!cur.consume(']')) {
                    while (true) {
                        if (!cur.consume('{')) return false;
                        AddedToken tok;
                        if (!cur.consume('}')) {
                            std::string ak;
                            while (true) {
                                if (!cur.parse_string(ak) || !cur.consume(':')) return false;
                                if (ak == "id") {
                                    int64_t v = -1;
                                    if (!cur.parse_int(v)) return false;
                                    tok.id = static_cast<int32_t>(v);
                                } else if (ak == "content") {
                                    if (!cur.parse_string(tok.content)) return false;
                                } else if (ak == "special") {
                                    if (!cur.parse_bool(tok.special)) return false;
                                } else {
                                    if (!cur.skip_value()) return false;
                                }
                                if (cur.consume('}')) break;
                                if (!cur.consume(',')) return false;
                            }
                        }
                        if (tok.id >= 0 && !tok.content.empty()) {
                            added_tokens_.push_back(tok);
                            added_first_byte_[static_cast<uint8_t>(tok.content[0])] = true;
                            set_decoded_entry(tok.id, tok.content, tok.special);
                            if (tok.id >= next_synth_id) next_synth_id = tok.id + 1;
                            if (tok.content == "<|begin_of_text|>" || tok.content == "<s>") bos_token_id_ = tok.id;
                            if (tok.content == "<|end_of_text|>" || tok.content == "<|endoftext|>" || tok.content == "</s>") {
                                eos_token_id_ = tok.id;
                            }
                            if (tok.content == "<|eot_id|>") eot_token_id_ = tok.id;
                        }
                        if (cur.consume(']')) break;
                        if (!cur.consume(',')) return false;
                    }
                }
            } else if (top_key == "model") {
                if (!cur.consume('{')) return false;
                if (!cur.consume('}')) {
                    std::string mk;
                    while (true) {
                        if (!cur.parse_string(mk) || !cur.consume(':')) return false;
                        if (mk == "ignore_merges") {
                            if (!cur.parse_bool(ignore_merges_)) return false;
                        } else if (mk == "vocab") {
                            if (!cur.consume('{')) return false;
                            if (!cur.consume('}')) {
                                raw_vocab_to_id_.reserve(131072);
                                sym_to_id.reserve(131072);
                                std::string vk;
                                while (true) {
                                    int64_t vid = -1;
                                    if (!cur.parse_string(vk) || !cur.consume(':') || !cur.parse_int(vid)) return false;
                                    const int32_t id32 = static_cast<int32_t>(vid);
                                    std::string raw = gpt2_token_to_raw_bytes(vk);
                                    if (raw.size() > max_raw_token_bytes_) max_raw_token_bytes_ = raw.size();
                                    raw_vocab_to_id_[raw] = id32;
                                    sym_to_id[raw] = id32;
                                    set_decoded_entry(id32, std::move(raw), false);
                                    if (id32 >= next_synth_id) next_synth_id = id32 + 1;
                                    if (cur.consume('}')) break;
                                    if (!cur.consume(',')) return false;
                                }
                            }
                        } else if (mk == "merges") {
                            if (!cur.consume('[')) return false;
                            if (!cur.consume(']')) {
                                raw_merges.reserve(300000);
                                std::string mline, ma, mb;
                                while (true) {
                                    if (cur.peek() == '[') {
                                        if (!cur.consume('[') || !cur.parse_string(ma) || !cur.consume(',') ||
                                            !cur.parse_string(mb) || !cur.consume(']')) {
                                            return false;
                                        }
                                        raw_merges.emplace_back(gpt2_token_to_raw_bytes(ma),
                                                                gpt2_token_to_raw_bytes(mb));
                                    } else {
                                        if (!cur.parse_string(mline)) return false;
                                        size_t sp = mline.find(' ');
                                        if (sp == std::string::npos) return false;
                                        raw_merges.emplace_back(
                                            gpt2_token_to_raw_bytes(mline.substr(0, sp)),
                                            gpt2_token_to_raw_bytes(mline.substr(sp + 1)));
                                    }
                                    if (cur.consume(']')) break;
                                    if (!cur.consume(',')) return false;
                                }
                            }
                        } else {
                            if (!cur.skip_value()) return false;
                        }
                        if (cur.consume('}')) break;
                        if (!cur.consume(',')) return false;
                    }
                }
            } else {
                if (!cur.skip_value()) return false;
            }

            if (cur.consume('}')) break;
            if (!cur.consume(',')) return false;
        }
    }

    if (raw_vocab_to_id_.empty()) return false;

    // Sort added tokens longest-first so leftmost-longest split matches HF AddedVocabulary.
    std::stable_sort(added_tokens_.begin(), added_tokens_.end(),
                     [](const AddedToken& a, const AddedToken& b) {
                         return a.content.size() > b.content.size();
                     });

    // Populate single-byte lookup table.
    byte_to_sym_id_.assign(256, -1);
    for (int b = 0; b < 256; ++b) {
        std::string s(1, static_cast<char>(static_cast<uint8_t>(b)));
        auto it = sym_to_id.find(s);
        if (it != sym_to_id.end()) {
            byte_to_sym_id_[b] = it->second;
        }
    }

    // Build integer-pair merge table (zero string allocation inside BPE loop).
    auto get_or_add_sym = [&](const std::string& raw) -> int32_t {
        auto it = sym_to_id.find(raw);
        if (it != sym_to_id.end()) return it->second;
        int32_t id = next_synth_id++;
        sym_to_id[raw] = id;
        return id;
    };

    merges_.reserve(raw_merges.size() * 2 + 16);
    for (uint32_t rank = 0; rank < static_cast<uint32_t>(raw_merges.size()); ++rank) {
        const std::string& a = raw_merges[rank].first;
        const std::string& b = raw_merges[rank].second;
        const int32_t id_a = get_or_add_sym(a);
        const int32_t id_b = get_or_add_sym(b);
        const int32_t id_ab = get_or_add_sym(a + b);
        const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(id_a)) << 32) |
                             static_cast<uint64_t>(static_cast<uint32_t>(id_b));
        merges_.emplace(key, MergeEntry{rank, id_ab});
    }

    bpe_loaded_ = true;
    return true;
}

bool Tokenizer::load_legacy_vocab(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) return false;

    clear();
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        int32_t id;
        std::string token;
        if (ss >> id >> token) {
            std::string clean_token = token;
            size_t pos = 0;
            while ((pos = clean_token.find("\xc4\xa0", pos)) != std::string::npos) {
                clean_token.replace(pos, 2, " ");
                pos += 1;
            }
            pos = 0;
            while ((pos = clean_token.find("\xc4\x8a", pos)) != std::string::npos) {
                clean_token.replace(pos, 2, "\n");
                pos += 1;
            }
            set_decoded_entry(id, clean_token, id >= 128000);
            legacy_token_to_id_[clean_token] = id;
        }
    }
    if (legacy_token_to_id_.count("<|begin_of_text|>")) bos_token_id_ = legacy_token_to_id_["<|begin_of_text|>"];
    if (legacy_token_to_id_.count("<|endoftext|>")) eos_token_id_ = legacy_token_to_id_["<|endoftext|>"];
    if (legacy_token_to_id_.count("<|end_of_text|>")) eos_token_id_ = legacy_token_to_id_["<|end_of_text|>"];
    if (legacy_token_to_id_.count("<|eot_id|>")) eot_token_id_ = legacy_token_to_id_["<|eot_id|>"];
    return !id_to_decoded_.empty();
}

bool Tokenizer::load(const std::string& path) {
    // 1. Direct .json path
    if (ends_with(path, ".json")) {
        return load_json_file(path);
    }

    // 2. Side-by-side <dir>/tokenizer.json
    const std::string dir = ends_with(path, ".vocab") ? parent_dir(path) : path;
    const std::string sibling_json = dir + "/tokenizer.json";
    if (file_readable(sibling_json) && load_json_file(sibling_json)) {
        return true;
    }

    // 3. Explicit TENZO_TOKENIZER_JSON override when provided
    if (const char* env_json = std::getenv("TENZO_TOKENIZER_JSON")) {
        if (*env_json && file_readable(env_json) && load_json_file(env_json)) {
            return true;
        }
    }

    // 4. Legacy tokenizer.vocab (needed to support tiny mock test fixtures)
    const std::string vocab_file = ends_with(path, ".vocab") ? path : (dir + "/tokenizer.vocab");
    bool legacy_ok = load_legacy_vocab(vocab_file);

    // 5. Upgrade valid >=128k BitNet/Llama-3 vocabs to full BPE tokenizer.json when available
    if (legacy_ok && id_to_decoded_.size() >= 128000) {
        for (const char* cand : {
                 "models/bitnet-1.58b-v2/tokenizer.json",
                 "/app/models/bitnet-1.58b-v2/tokenizer.json",
                 "tools/tokenizer/bitnet_b158_tokenizer.json",
                 "/app/tools/tokenizer/bitnet_b158_tokenizer.json",
                 "../tools/tokenizer/bitnet_b158_tokenizer.json",
             }) {
            if (file_readable(cand) && load_json_file(cand)) {
                return true;
            }
        }
    }

    return legacy_ok;
}

void Tokenizer::encode_bpe_piece(const uint8_t* bytes, size_t len, std::vector<int32_t>& out) const {
    if (len == 0) return;

    if (ignore_merges_ && len <= max_raw_token_bytes_) {
        std::string key(reinterpret_cast<const char*>(bytes), len);
        auto it = raw_vocab_to_id_.find(key);
        if (it != raw_vocab_to_id_.end()) {
            out.push_back(it->second);
            return;
        }
    }

    std::vector<int32_t> syms(len);
    for (size_t i = 0; i < len; ++i) {
        syms[i] = byte_to_sym_id_[bytes[i]];
    }

    while (syms.size() >= 2) {
        uint32_t best_rank = std::numeric_limits<uint32_t>::max();
        int32_t best_a = -1;
        int32_t best_b = -1;
        int32_t best_merged = -1;

        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(syms[i])) << 32) |
                                 static_cast<uint64_t>(static_cast<uint32_t>(syms[i + 1]));
            auto it = merges_.find(key);
            if (it != merges_.end() && it->second.rank < best_rank) {
                best_rank = it->second.rank;
                best_a = syms[i];
                best_b = syms[i + 1];
                best_merged = it->second.merged_id;
            }
        }

        if (best_rank == std::numeric_limits<uint32_t>::max()) {
            break;
        }

        size_t w = 0;
        size_t r = 0;
        while (r < syms.size()) {
            if (r + 1 < syms.size() && syms[r] == best_a && syms[r + 1] == best_b) {
                syms[w++] = best_merged;
                r += 2;
            } else {
                syms[w++] = syms[r++];
            }
        }
        syms.resize(w);
    }

    for (int32_t s : syms) {
        if (s >= 0 && static_cast<size_t>(s) < id_is_valid_.size() && id_is_valid_[s]) {
            out.push_back(s);
        }
    }
}

void Tokenizer::encode_segment(const std::string& seg, std::vector<int32_t>& out) const {
    if (seg.empty()) return;

    // Decode UTF-8 codepoints once with exact byte offsets for pre-tokenization.
    std::vector<CPInfo> cps;
    cps.reserve(seg.size() + 1);
    size_t byte_pos = 0;
    while (byte_pos < seg.size()) {
        const size_t start_off = byte_pos;
        uint32_t cp = 0;
        next_utf8_cp(seg, byte_pos, cp);
        cps.push_back(CPInfo{cp, start_off});
    }
    const size_t n = cps.size();
    cps.push_back(CPInfo{0u, seg.size()});

    const uint8_t* raw_ptr = reinterpret_cast<const uint8_t*>(seg.data());
    size_t i = 0;
    while (i < n) {
        const uint32_t cp0 = cps[i].cp;
        size_t next_i = i;

        // Branch 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cp0 == '\'' && i + 1 < n) {
            const uint32_t c1 = ascii_lower(cps[i + 1].cp);
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                next_i = i + 2;
            } else if (i + 2 < n) {
                const uint32_t c2 = ascii_lower(cps[i + 2].cp);
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    next_i = i + 3;
                }
            }
        }

        // Branch 2: [^\r\n\p{L}\p{N}]?\p{L}+
        if (next_i == i) {
            if (detail::is_unicode_letter(cp0)) {
                size_t j = i + 1;
                while (j < n && detail::is_unicode_letter(cps[j].cp)) ++j;
                next_i = j;
            } else if (!is_cr_lf(cp0) && !detail::is_unicode_number(cp0) &&
                       i + 1 < n && detail::is_unicode_letter(cps[i + 1].cp)) {
                size_t j = i + 2;
                while (j < n && detail::is_unicode_letter(cps[j].cp)) ++j;
                next_i = j;
            }
        }

        // Branch 3: \p{N}{1,3}
        if (next_i == i && detail::is_unicode_number(cp0)) {
            size_t j = i + 1;
            if (j < n && detail::is_unicode_number(cps[j].cp)) ++j;
            if (j < n && detail::is_unicode_number(cps[j].cp)) ++j;
            next_i = j;
        }

        // Branch 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
        if (next_i == i) {
            const size_t k = (cp0 == ' ') ? (i + 1) : i;
            if (k < n && !detail::is_unicode_space(cps[k].cp) &&
                !detail::is_unicode_letter(cps[k].cp) &&
                !detail::is_unicode_number(cps[k].cp)) {
                size_t j = k + 1;
                while (j < n && !detail::is_unicode_space(cps[j].cp) &&
                       !detail::is_unicode_letter(cps[j].cp) &&
                       !detail::is_unicode_number(cps[j].cp)) {
                    ++j;
                }
                while (j < n && is_cr_lf(cps[j].cp)) ++j;
                next_i = j;
            }
        }

        // Branches 5, 6, 7: whitespace runs (\s*[\r\n]+ | \s+(?!\S) | \s+)
        if (next_i == i && detail::is_unicode_space(cp0)) {
            size_t j = i;
            size_t last_nl_plus1 = 0;
            while (j < n && detail::is_unicode_space(cps[j].cp)) {
                if (is_cr_lf(cps[j].cp)) last_nl_plus1 = j + 1;
                ++j;
            }
            if (last_nl_plus1 > i) {
                // Branch 5: \s*[\r\n]+
                next_i = last_nl_plus1;
            } else if (j == n) {
                // Branch 6: \s+(?!\S) at end of segment
                next_i = j;
            } else if (j - i >= 2) {
                // Branch 6: \s+(?!\S) backtracks 1 whitespace codepoint before non-whitespace
                next_i = j - 1;
            } else {
                // Branch 7: single \s+ before non-whitespace
                next_i = j;
            }
        }

        if (next_i == i) {
            next_i = i + 1;
        }

        const size_t b_start = cps[i].byte_off;
        const size_t b_end = cps[next_i].byte_off;
        encode_bpe_piece(raw_ptr + b_start, b_end - b_start, out);
        i = next_i;
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> tokens;
    if (id_to_decoded_.empty()) return tokens;

    if (!bpe_loaded_) {
        if (text.empty()) return tokens;
        if (add_bos) {
            auto bos_it = legacy_token_to_id_.find("<|begin_of_text|>");
            if (bos_it != legacy_token_to_id_.end()) {
                tokens.push_back(bos_it->second);
            } else if (id_to_decoded_.size() > 128000) {
                tokens.push_back(128000);
            }
        }
        size_t i = 0;
        while (i < text.length()) {
            size_t longest_len = 0;
            int32_t best_id = -1;
            for (size_t len = std::min<size_t>(32, text.length() - i); len >= 1; --len) {
                std::string sub = text.substr(i, len);
                auto it = legacy_token_to_id_.find(sub);
                if (it != legacy_token_to_id_.end()) {
                    longest_len = len;
                    best_id = it->second;
                    break;
                }
                if (i == 0 || text[i - 1] == ' ') {
                    auto it_sp = legacy_token_to_id_.find(" " + sub);
                    if (it_sp != legacy_token_to_id_.end()) {
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
                std::string single_char = text.substr(i, 1);
                auto it = legacy_token_to_id_.find(single_char);
                tokens.push_back(it != legacy_token_to_id_.end() ? it->second : 0);
                i += 1;
            }
        }
        return tokens;
    }

    if (add_bos && bos_token_id_ >= 0) {
        tokens.push_back(bos_token_id_);
    }
    if (text.empty()) return tokens;

    // Split on added_tokens (leftmost-longest, before regex pre-tokenization).
    size_t seg_start = 0;
    size_t i = 0;
    while (i < text.size()) {
        int32_t matched_id = -1;
        size_t matched_len = 0;
        if (added_first_byte_[static_cast<uint8_t>(text[i])]) {
            const size_t avail = text.size() - i;
            for (const auto& at : added_tokens_) {
                if (at.content.size() <= avail &&
                    text.compare(i, at.content.size(), at.content) == 0) {
                    matched_id = at.id;
                    matched_len = at.content.size();
                    break; // added_tokens_ is sorted descending by length
                }
            }
        }
        if (matched_len > 0) {
            if (i > seg_start) {
                encode_segment(text.substr(seg_start, i - seg_start), tokens);
            }
            tokens.push_back(matched_id);
            i += matched_len;
            seg_start = i;
        } else {
            ++i;
        }
    }
    if (seg_start < text.size()) {
        encode_segment(text.substr(seg_start), tokens);
    }

    return tokens;
}

std::string Tokenizer::decode_token(int32_t token_id) const {
    if (token_id >= 0 && static_cast<size_t>(token_id) < id_to_decoded_.size() && id_is_valid_[token_id]) {
        return id_to_decoded_[token_id];
    }
    return "";
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string text;
    for (int32_t id : tokens) {
        if (id >= 0 && static_cast<size_t>(id) < id_to_decoded_.size() && id_is_valid_[id]) {
            text += id_to_decoded_[id];
        }
    }
    return text;
}

bool Tokenizer::is_stop_token(int32_t tid) const {
    if (tid == eos_token_id_ || tid == eot_token_id_) return true;
    if (tid >= 128000 && tid <= 128256) return true;
    if (tid >= 0 && static_cast<size_t>(tid) < id_is_special_.size() && id_is_special_[tid]) return true;
    if (tid >= 0 && static_cast<size_t>(tid) < id_to_decoded_.size()) {
        const std::string& s = id_to_decoded_[tid];
        if (!s.empty()) {
            if (s == "<|end_of_text|>" || s == "<|endoftext|>" || s == "<|eot_id|>" || s == "</s>" || s == "<eos>") {
                return true;
            }
            if (s.find("endoftext") != std::string::npos || s.find("eot_id") != std::string::npos ||
                s.find("end_of_eot") != std::string::npos || s.find("end_of_text") != std::string::npos ||
                s.find("end_header_id") != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

} // namespace runtime
} // namespace tenzo
