#ifndef TENZO_TENSOR_DUMP_H
#define TENZO_TENSOR_DUMP_H

// Opt-in intermediate tensor dumps for layer-by-layer comparison against tools/reference/golden_bitnet.py.
//   TENZO_DUMP_DIR     output directory (must exist); unset => dumping disabled, zero work besides one bool test
//   TENZO_DUMP_POS     comma-separated token positions (default: 0)
//   TENZO_DUMP_LAYERS  comma-separated layer indices (default: 0)
// Each tensor is written as <dir>/[L<k>.]<name>_p<pos>.f32 (raw little-endian float32) and listed in
// <dir>/manifest.json ({name, layer, pos, shape, file}); names follow the golden reference.

#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace tenzo {

class TensorDumper {
public:
    TensorDumper() {
        const char* dir = std::getenv("TENZO_DUMP_DIR");
        if (!dir || !*dir) return;
        dir_ = dir;
        enabled_ = true;
        parse(std::getenv("TENZO_DUMP_POS"), pos_, 0);
        parse(std::getenv("TENZO_DUMP_LAYERS"), layers_, 0);
    }

    bool enabled() const { return enabled_; }
    bool want(int pos, int layer) const {
        return enabled_ && pos_.count(pos) && (layer < 0 || layers_.count(layer));
    }

    // layer < 0: model-level tensor (embed, final_norm, logits).
    void dump(const char* name, int layer, int pos, const float* data, size_t n) {
        if (!want(pos, layer)) return;
        std::string file = (layer >= 0 ? "L" + std::to_string(layer) + "." : std::string()) + name + "_p" +
                           std::to_string(pos) + ".f32";
        std::string path = dir_ + "/" + file;
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(data, sizeof(float), n, f);
            std::fclose(f);
        }
        std::ostringstream e;
        e << "{\"name\": \"" << name << "\", \"layer\": " << (layer >= 0 ? std::to_string(layer) : "null")
          << ", \"pos\": " << pos << ", \"shape\": [" << n << "], \"file\": \"" << file << "\"}";
        entries_.push_back(e.str());
        write_manifest();
    }

private:
    static void parse(const char* s, std::set<int>& out, int dflt) {
        if (!s || !*s) { out.insert(dflt); return; }
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, ','))
            if (!tok.empty()) out.insert(std::atoi(tok.c_str()));
    }
    void write_manifest() const {
        std::string path = dir_ + "/manifest.json";
        if (FILE* f = std::fopen(path.c_str(), "w")) {
            std::fputs("{\"source\": \"tenzo\", \"tensors\": [\n", f);
            for (size_t i = 0; i < entries_.size(); ++i) {
                std::fputs(entries_[i].c_str(), f);
                std::fputs(i + 1 < entries_.size() ? ",\n" : "\n", f);
            }
            std::fputs("]}\n", f);
            std::fclose(f);
        }
    }

    bool enabled_ = false;
    std::string dir_;
    std::set<int> pos_, layers_;
    std::vector<std::string> entries_;
};

} // namespace tenzo

#endif // TENZO_TENSOR_DUMP_H
