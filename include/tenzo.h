/**
 * @file tenzo.h
 * @brief Tenzo High-Performance AI Compiler & Inference Engine C-API (v0.3.0)
 * 
 * Provides an ABI-stable, pure C interface for embedding the Tenzo native runtime
 * in C, C++, Rust, Swift, Go, and mobile/edge applications with zero Python dependencies.
 */

#ifndef TENZO_H
#define TENZO_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef TENZO_BUILD_SHARED
    #define TENZO_API __declspec(dllexport)
  #else
    #define TENZO_API __declspec(dllimport)
  #endif
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define TENZO_API __attribute__((visibility("default")))
  #else
    #define TENZO_API
  #endif
#endif

#define TENZO_VERSION_MAJOR 0
#define TENZO_VERSION_MINOR 3
#define TENZO_VERSION_PATCH 0
#define TENZO_VERSION_STRING "0.3.0"

/* Opaque handle to the Tenzo Native Execution Engine */
typedef struct tenzo_engine_opaque* tenzo_engine_t;

/* Status codes */
typedef enum {
    TENZO_SUCCESS = 0,
    TENZO_ERROR_INVALID_ARGUMENT = -1,
    TENZO_ERROR_OUT_OF_MEMORY = -2,
    TENZO_ERROR_IO = -3,
    TENZO_ERROR_MODEL_PARSE = -4,
    TENZO_ERROR_UNSUPPORTED = -5
} tenzo_status_t;

/* Engine configuration */
typedef struct {
    int hidden_size;     /* Default: 2560 for BitNet 2B */
    int num_q_heads;     /* Default: 20 */
    int num_kv_heads;    /* Default: 5 */
    int head_dim;        /* Default: 128 */
    int num_layers;      /* Default: 30 */
    int ffn_dim;         /* Default: 6912 */
    int vocab_size;      /* Default: 128256 */
    int max_seq_len;     /* Default: 8192 */
    const char* kv_mode; /* "int8_fused", "tl1_fused", or "fp32" */
} tenzo_config_t;

/* Sampling parameters */
typedef struct {
    float temperature;        /* Default: 0.7f (<=0 for greedy argmax) */
    float top_p;              /* Default: 0.9f */
    int top_k;                /* Default: 40 */
    float repetition_penalty; /* Default: 1.15f */
} tenzo_sampling_params_t;

/* Layer weights (packed 2-bit TL1 / INT8 / FP32) */
typedef struct {
    const int8_t* q_w;    float q_scale;
    const int8_t* k_w;    float k_scale;
    const int8_t* v_w;    float v_scale;
    const int8_t* out_w;  float out_scale;
    const int8_t* gate_w; float gate_scale;
    const int8_t* up_w;   float up_scale;
    const int8_t* down_w; float down_scale;
    const float* in_norm;
    const float* attn_sub_norm;
    const float* post_norm;
    const float* ffn_sub_norm;
} tenzo_layer_weights_t;

/**
 * @brief Returns the version string of the Tenzo Runtime.
 */
TENZO_API const char* tenzo_get_version(void);

/**
 * @brief Creates default configuration for BitNet 2B 1.58b architecture.
 */
TENZO_API tenzo_config_t tenzo_default_config(void);

/**
 * @brief Creates default sampling parameters.
 */
TENZO_API tenzo_sampling_params_t tenzo_default_sampling_params(void);

/**
 * @brief Initializes a Tenzo execution engine with preallocated zero-allocation scratch buffers.
 * @param config Pointer to engine configuration.
 * @return Handle to engine, or NULL on failure.
 */
TENZO_API tenzo_engine_t tenzo_create_engine(const tenzo_config_t* config);

/**
 * @brief Destroys engine instance and frees associated buffers.
 */
TENZO_API void tenzo_destroy_engine(tenzo_engine_t engine);

/**
 * @brief Resets KV-Cache sequence length and state for a new conversation/session.
 */
TENZO_API void tenzo_reset(tenzo_engine_t engine);

/**
 * @brief Gets current active KV-cache sequence length.
 */
TENZO_API int tenzo_get_seq_len(tenzo_engine_t engine);

/**
 * @brief Configures layer weights directly in memory.
 */
TENZO_API tenzo_status_t tenzo_set_layer_weights(
    tenzo_engine_t engine,
    int layer_idx,
    const tenzo_layer_weights_t* weights
);

/**
 * @brief Sets final layer RMSNorm weights.
 */
TENZO_API tenzo_status_t tenzo_set_final_norm(tenzo_engine_t engine, const float* norm_w);

/**
 * @brief Sets LM Head / Embedding table (FP32 mode).
 */
TENZO_API tenzo_status_t tenzo_set_lm_head_f32(
    tenzo_engine_t engine,
    const float* embed_w,
    int vocab_size,
    int hidden_size
);

/**
 * @brief Sets LM Head / Embedding table (Quantized INT8 Symmetric mode).
 */
TENZO_API tenzo_status_t tenzo_set_lm_head_i8(
    tenzo_engine_t engine,
    const int8_t* embed_w,
    const float* embed_scales,
    int vocab_size,
    int hidden_size
);

/**
 * @brief Automatically parses MLIR graph and loads binary weights file.
 * @param engine Target engine handle.
 * @param weights_path Path to weights.bin
 * @param mlir_path Path to model.mlir
 */
TENZO_API tenzo_status_t tenzo_load_model(
    tenzo_engine_t engine,
    const char* weights_path,
    const char* mlir_path
);

/**
 * @brief Prefills a single prompt token into the KV-Cache.
 */
TENZO_API tenzo_status_t tenzo_prefill_token(tenzo_engine_t engine, int token_id);

/**
 * @brief Executes a single autoregressive step: Transformer forward + LM Head + Sampler.
 * @param engine Target engine handle.
 * @param cur_token Current input token ID.
 * @param params Sampling parameters.
 * @param past_tokens Array of past token IDs for repetition penalty.
 * @param past_tokens_len Number of elements in past_tokens array.
 * @return Generated next token ID, or -1 on error.
 */
TENZO_API int tenzo_generate_step(
    tenzo_engine_t engine,
    int cur_token,
    const tenzo_sampling_params_t* params,
    const int* past_tokens,
    int past_tokens_len
);

#ifdef __cplusplus
}
#endif

#endif /* TENZO_H */
