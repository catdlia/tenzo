/**
 * @file tenzo.hpp
 * @brief Modern C++ RAII wrapper for the Tenzo Inference Engine SDK (v0.3.0)
 */

#ifndef TENZO_HPP
#define TENZO_HPP

#include "tenzo.h"
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace tenzo {

class Engine {
public:
    explicit Engine(const tenzo_config_t& config = tenzo_default_config()) {
        engine_ = tenzo_create_engine(&config);
        if (!engine_) {
            throw std::runtime_error("Failed to initialize Tenzo execution engine");
        }
    }

    ~Engine() {
        if (engine_) {
            tenzo_destroy_engine(engine_);
            engine_ = nullptr;
        }
    }

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    Engine(Engine&& other) noexcept : engine_(other.engine_) {
        other.engine_ = nullptr;
    }

    Engine& operator=(Engine&& other) noexcept {
        if (this != &other) {
            if (engine_) tenzo_destroy_engine(engine_);
            engine_ = other.engine_;
            other.engine_ = nullptr;
        }
        return *this;
    }

    void load_model(const std::string& weights_path, const std::string& mlir_path) {
        tenzo_status_t status = tenzo_load_model(engine_, weights_path.c_str(), mlir_path.c_str());
        if (status != TENZO_SUCCESS) {
            throw std::runtime_error("Failed to load Tenzo model from " + weights_path);
        }
    }

    void reset() {
        tenzo_reset(engine_);
    }

    int get_seq_len() const {
        return tenzo_get_seq_len(engine_);
    }

    void prefill_token(int token_id) {
        tenzo_status_t status = tenzo_prefill_token(engine_, token_id);
        if (status != TENZO_SUCCESS) {
            throw std::runtime_error("Failed to prefill token");
        }
    }

    int generate_step(
        int cur_token,
        const tenzo_sampling_params_t& params,
        const std::vector<int>& past_tokens
    ) {
        int next_tok = tenzo_generate_step(
            engine_,
            cur_token,
            &params,
            past_tokens.data(),
            static_cast<int>(past_tokens.size())
        );
        if (next_tok < 0) {
            throw std::runtime_error("Failed during tenzo_generate_step");
        }
        return next_tok;
    }

    tenzo_engine_t get_raw_handle() const { return engine_; }

private:
    tenzo_engine_t engine_ = nullptr;
};

} // namespace tenzo

#endif /* TENZO_HPP */
