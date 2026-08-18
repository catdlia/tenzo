/**
 * @file TenzoEngine.h
 * @brief Tenzo Native C++ Inference Engine Interface (Phase E)
 */

#ifndef TENZO_ENGINE_H
#define TENZO_ENGINE_H

#include "tenzo.h"
#include "simd_arm_compat.h"
#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace tenzo {

// 256-bit SIMD vector wrapper with 32-byte alignment
struct alignas(32) Vec256i {
    __m256i v;
    Vec256i() : v(_mm256_setzero_si256()) {}
    Vec256i(__m256i val) : v(val) {}
    operator __m256i() const { return v; }
    Vec256i& operator=(__m256i val) { v = val; return *this; }
};

struct LayerWeightsInternal {
    std::vector<int8_t> q_w; float q_scale = 1.0f;
    std::vector<int8_t> k_w; float k_scale = 1.0f;
    std::vector<int8_t> v_w; float v_scale = 1.0f;
    std::vector<int8_t> out_w; float out_scale = 1.0f;
    std::vector<int8_t> gate_w; float gate_scale = 1.0f;
    std::vector<int8_t> up_w; float up_scale = 1.0f;
    std::vector<int8_t> down_w; float down_scale = 1.0f;

    std::vector<float> in_norm;
    std::vector<float> attn_sub_norm;
    std::vector<float> post_norm;
    std::vector<float> ffn_sub_norm;

    // Fast direct pointers
    const int8_t* q_w_ptr = nullptr;
    const int8_t* k_w_ptr = nullptr;
    const int8_t* v_w_ptr = nullptr;
    const int8_t* out_w_ptr = nullptr;
    const int8_t* gate_w_ptr = nullptr;
    const int8_t* up_w_ptr = nullptr;
    const int8_t* down_w_ptr = nullptr;

    const float* in_norm_ptr = nullptr;
    const float* attn_sub_norm_ptr = nullptr;
    const float* post_norm_ptr = nullptr;
    const float* ffn_sub_norm_ptr = nullptr;
};

class FusedKVCache {
public:
    int num_layers;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
    std::string kv_mode; // "int8_fused", "tl1_fused", "fp32"
    int cur_seq_len = 0;

    // Buffers
    std::vector<float> fp32_k;
    std::vector<float> fp32_v;

    // INT8 Fused Buffers (4x compression)
    std::vector<int8_t> int8_k;
    std::vector<int8_t> int8_v;
    std::vector<float> int8_k_scales;
    std::vector<float> int8_v_scales;

    // Ternary TL1 Fused Buffers (14.2x compression vs FP32)
    std::vector<uint8_t> tl1_k;
    std::vector<uint8_t> tl1_v;
    std::vector<float> tl1_k_scales;
    std::vector<float> tl1_v_scales;

    // SIMD Popcount Bitplanes Buffers (Zero-FMA Attention: 32 bytes/head)
    struct alignas(32) PopcountBitplanes {
        uint64_t kp0, kp1, kn0, kn1;
    };
    std::vector<PopcountBitplanes> popcount_k;
    std::vector<float> popcount_k_scales;
    std::vector<uint8_t> popcount_v;
    std::vector<float> popcount_v_scales;

    FusedKVCache(
        int num_layers = 30,
        int num_q_heads = 20,
        int num_kv_heads = 5,
        int head_dim = 128,
        int max_seq_len = 8192,
        const std::string& kv_mode = "int8_fused"
    );

    void reset();
    int get_seq_len() const { return cur_seq_len; }
    void increment_seq_len(int count = 1) { cur_seq_len += count; }

    void forward_attention_raw(
        int layer_idx,
        float* __restrict__ q_ptr,
        float* __restrict__ k_ptr,
        float* __restrict__ v_ptr,
        float* __restrict__ out_ptr
    );
};

// Paged Attention Block Manager (Dynamic Virtual Memory Paging)
class PagedBlockManager {
public:
    int total_blocks;
    int block_size;
    std::vector<int32_t> free_blocks;

    PagedBlockManager(int total_blocks = 512, int block_size = 16)
        : total_blocks(total_blocks), block_size(block_size) {
        reset();
    }

    void reset() {
        free_blocks.clear();
        free_blocks.reserve(total_blocks);
        for (int i = total_blocks - 1; i >= 0; --i) {
            free_blocks.push_back(i);
        }
    }

    int32_t allocate_block() {
        if (free_blocks.empty()) return -1;
        int32_t blk = free_blocks.back();
        free_blocks.pop_back();
        return blk;
    }

    void free_block(int32_t blk) {
        if (blk >= 0 && blk < total_blocks) {
            free_blocks.push_back(blk);
        }
    }

    size_t num_free_blocks() const { return free_blocks.size(); }
    size_t num_allocated_blocks() const { return total_blocks - free_blocks.size(); }
};

// Paged KV-Cache with Dynamic Block-Table Translation
class PagedKVCache {
public:
    int num_layers;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    static constexpr int BLOCK_SIZE = 16;
    int total_blocks;

    PagedBlockManager block_mgr;
    std::vector<int32_t> block_table; // Sequence logical-to-physical block mapping
    int cur_seq_len = 0;

    // Physical Page Pools: [total_blocks, num_layers, num_kv_heads, BLOCK_SIZE, head_dim]
    std::vector<int8_t> paged_k_pool;
    std::vector<int8_t> paged_v_pool;
    std::vector<float> paged_k_scales;
    std::vector<float> paged_v_scales;

    PagedKVCache(
        int num_layers = 30,
        int num_q_heads = 20,
        int num_kv_heads = 5,
        int head_dim = 128,
        int total_blocks = 512
    );

    void reset();
    int get_seq_len() const { return cur_seq_len; }
    void increment_seq_len(int count = 1) { cur_seq_len += count; }

    void forward_attention_raw(
        int layer_idx,
        float* __restrict__ q_ptr,
        float* __restrict__ k_ptr,
        float* __restrict__ v_ptr,
        float* __restrict__ out_ptr
    );
};

class TenzoEngineImpl {
public:
    tenzo_config_t config;
    int q_dim;
    int kv_dim;
    int ffn_dim;
    std::vector<LayerWeightsInternal> layers;
    std::vector<float> final_norm_w;
    const float* final_norm_ptr = nullptr;

    // Embeddings / LM Head
    bool is_lm_i8 = false;
    std::vector<float> embed_w_f32;
    std::vector<int8_t> embed_w_i8;
    std::vector<float> embed_scales;

    FusedKVCache kv_cache;
    PagedKVCache paged_kv_cache;
    bool use_paged_kv = false;

    // Preallocated Zero-Allocation Scratch Buffers
    std::vector<float> buf_x;
    std::vector<float> buf_norm_x;
    std::vector<float> buf_q;
    std::vector<float> buf_k;
    std::vector<float> buf_v;
    std::vector<float> buf_attn_out;
    std::vector<float> buf_attn_sub;
    std::vector<float> buf_out;
    std::vector<float> buf_h1;
    std::vector<float> buf_post_norm;
    std::vector<float> buf_gate;
    std::vector<float> buf_up;
    std::vector<float> buf_act;
    std::vector<float> buf_ffn_norm;
    std::vector<float> buf_down;
    std::vector<Vec256i> lut_buf;
    std::vector<float> logits_scratch;
    std::vector<int8_t> buf_x_i8;
    std::vector<Vec256i> buf_x_expanded;

    __m256i* lut_ptr() { return reinterpret_cast<__m256i*>(lut_buf.data()); }
    const __m256i* lut_ptr() const { return reinterpret_cast<const __m256i*>(lut_buf.data()); }

    explicit TenzoEngineImpl(const tenzo_config_t& cfg);

    void reset();
    int get_seq_len() const { return use_paged_kv ? paged_kv_cache.get_seq_len() : kv_cache.get_seq_len(); }

    void set_layer_weights(int layer_idx, const tenzo_layer_weights_t* w);
    void set_final_norm(const float* norm_w);
    void set_lm_head_f32(const float* embed_w, int vocab_size, int hidden_size);
    void set_lm_head_i8(const int8_t* embed_w, const float* scales, int vocab_size, int hidden_size);

    tenzo_status_t load_model_from_files(const char* weights_path, const char* mlir_path);

    void embedding_lookup(int token_id);
    void forward_layer_raw(int layer_idx);
    void forward_layer_gpu(int layer_idx);
    void compute_logits();
    int sample_top_k_top_p(const tenzo_sampling_params_t* params, const int* past_tokens, int past_tokens_len);

    void prefill_token(int token_id);
    int generate_step(int cur_token, const tenzo_sampling_params_t* params, const int* past_tokens, int past_tokens_len);
};

} // namespace tenzo

struct tenzo_engine_opaque {
    std::unique_ptr<tenzo::TenzoEngineImpl> impl;
};

#endif /* TENZO_ENGINE_H */
