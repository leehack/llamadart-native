#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "llama.h"

#if defined(_WIN32)
#  if defined(llamadart_lib_EXPORTS)
#    define LLAMADART_API __declspec(dllexport)
#  else
#    define LLAMADART_API __declspec(dllimport)
#  endif
#else
#  define LLAMADART_API __attribute__((visibility("default")))
#endif

// Opaque MTP speculative decoding state owned by libllamadart.
struct llama_dart_mtp;

// Sets the log level for llama.cpp
LLAMADART_API void llama_dart_set_log_level(int level);

// Creates a llama.cpp draft-mtp speculative decoding state against the target
// model. The draft context is owned by the returned handle and is freed with
// llama_dart_mtp_free.
LLAMADART_API struct llama_dart_mtp * llama_dart_mtp_init(
    struct llama_model * model,
    struct llama_context * ctx_tgt,
    struct llama_context_params ctx_params,
    int32_t draft_token_max,
    int32_t draft_token_min,
    float min_probability,
    bool backend_sampling);

// Creates a llama.cpp draft-mtp speculative decoding state against a separately
// loaded draft model, equivalent to llama.cpp's --model-draft path.
LLAMADART_API struct llama_dart_mtp * llama_dart_mtp_init_with_draft_model(
    struct llama_model * draft_model,
    struct llama_context * ctx_tgt,
    struct llama_context_params ctx_params,
    int32_t draft_token_max,
    int32_t draft_token_min,
    float min_probability,
    bool backend_sampling);

LLAMADART_API void llama_dart_mtp_free(struct llama_dart_mtp * mtp);

LLAMADART_API struct llama_context * llama_dart_mtp_get_draft_context(
    struct llama_dart_mtp * mtp);

LLAMADART_API bool llama_dart_mtp_begin(
    struct llama_dart_mtp * mtp,
    llama_seq_id seq_id,
    const llama_token * prompt,
    int32_t prompt_count);

LLAMADART_API bool llama_dart_mtp_process_batch(
    struct llama_dart_mtp * mtp,
    struct llama_batch batch);

LLAMADART_API int32_t llama_dart_mtp_draft(
    struct llama_dart_mtp * mtp,
    llama_seq_id seq_id,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt,
    int32_t prompt_count,
    int32_t draft_token_max,
    llama_token * out_tokens,
    int32_t out_capacity);

LLAMADART_API void llama_dart_mtp_accept(
    struct llama_dart_mtp * mtp,
    llama_seq_id seq_id,
    uint16_t accepted_count);

LLAMADART_API int32_t llama_dart_sampler_sample_and_accept_n(
    struct llama_sampler * sampler,
    struct llama_context * ctx,
    const int32_t * idxs,
    int32_t idx_count,
    const llama_token * draft_tokens,
    int32_t draft_count,
    llama_token * out_tokens,
    int32_t out_capacity);

#ifdef __cplusplus
}
#endif
