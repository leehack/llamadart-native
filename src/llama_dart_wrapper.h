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

// Opaque n-gram speculative decoding state owned by libllamadart.
struct llama_dart_ngram;

// Opaque upstream speculative decoding state owned by libllamadart.
struct llama_dart_speculative;

// Primitive C mirror of the upstream common_params_speculative knobs used by
// libllamadart. Most positive integer controls override upstream defaults.
// Fields that accept zero as a meaningful override use negative values as
// their "use upstream default" sentinel.
struct llama_dart_speculative_params {
    // Optional comma-separated upstream speculative type names, e.g.
    // "ngram-mod,draft-mtp". When set, this is preferred over type_mask so
    // callers do not depend on upstream enum ordinals.
    const char * type_names;

    // Fallback bit mask of upstream common_speculative_type values. For
    // example, 1 << 3 enables draft-mtp. A zero mask disables speculation.
    uint32_t type_mask;

    // Pass negative values for nullable numeric controls to preserve upstream
    // llama.cpp defaults.
    int32_t draft_token_max;
    int32_t draft_token_min;
    float draft_min_probability;
    float draft_split_probability;
    bool backend_sampling;

    int32_t ngram_size_n;
    int32_t ngram_size_m;
    int32_t ngram_min_hits;

    int32_t ngram_match;
    int32_t ngram_token_min;
    int32_t ngram_token_max;

    const char * ngram_cache_static_path;
    const char * ngram_cache_dynamic_path;
};

// Sets the log level for llama.cpp
LLAMADART_API void llama_dart_set_log_level(int level);

// Creates a sampler that applies llama.cpp's reasoning-token budget before an
// optional grammar sampler. The returned sampler owns grammar_sampler.
//
// When pause_grammar_while_reasoning is true, grammar constraints are paused
// while generation is inside the reasoning block. prompt_tokens determine
// whether a template already leaves generation inside a reasoning block.
LLAMADART_API struct llama_sampler * llama_dart_sampler_init_reasoning_budget(
    const struct llama_vocab * vocab,
    const char * start_tag,
    const char * end_tag,
    const char * forced_message,
    int32_t budget_tokens,
    bool pause_grammar_while_reasoning,
    struct llama_sampler * grammar_sampler,
    const llama_token * prompt_tokens,
    int32_t prompt_token_count);

LLAMADART_API struct llama_dart_speculative * llama_dart_speculative_init(
    struct llama_model * target_model,
    struct llama_model * draft_model,
    struct llama_context * target_context,
    struct llama_context_params context_params,
    const struct llama_dart_speculative_params * params);

LLAMADART_API void llama_dart_speculative_free(
    struct llama_dart_speculative * speculative);

LLAMADART_API struct llama_context * llama_dart_speculative_get_draft_context(
    struct llama_dart_speculative * speculative);

LLAMADART_API bool llama_dart_speculative_need_embd(
    struct llama_dart_speculative * speculative);

LLAMADART_API bool llama_dart_speculative_need_embd_nextn(
    struct llama_dart_speculative * speculative);

LLAMADART_API bool llama_dart_speculative_begin(
    struct llama_dart_speculative * speculative,
    llama_seq_id seq_id,
    const llama_token * prompt,
    int32_t prompt_count);

LLAMADART_API bool llama_dart_speculative_process_batch(
    struct llama_dart_speculative * speculative,
    struct llama_batch batch);

LLAMADART_API int32_t llama_dart_speculative_draft(
    struct llama_dart_speculative * speculative,
    llama_seq_id seq_id,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt,
    int32_t prompt_count,
    int32_t draft_token_max,
    llama_token * out_tokens,
    int32_t out_capacity);

LLAMADART_API void llama_dart_speculative_accept(
    struct llama_dart_speculative * speculative,
    llama_seq_id seq_id,
    uint16_t accepted_count);

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

// Creates a llama.cpp ngram-simple speculative decoding state. The returned
// handle uses token history only; it does not allocate a draft model/context.
LLAMADART_API struct llama_dart_ngram * llama_dart_ngram_simple_init(
    int32_t ngram_size,
    int32_t draft_token_max);

LLAMADART_API void llama_dart_ngram_free(struct llama_dart_ngram * ngram);

LLAMADART_API bool llama_dart_ngram_begin(
    struct llama_dart_ngram * ngram,
    llama_seq_id seq_id,
    const llama_token * prompt,
    int32_t prompt_count);

LLAMADART_API bool llama_dart_ngram_process_batch(
    struct llama_dart_ngram * ngram,
    struct llama_batch batch);

LLAMADART_API int32_t llama_dart_ngram_draft(
    struct llama_dart_ngram * ngram,
    llama_seq_id seq_id,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt,
    int32_t prompt_count,
    int32_t draft_token_max,
    llama_token * out_tokens,
    int32_t out_capacity);

LLAMADART_API void llama_dart_ngram_accept(
    struct llama_dart_ngram * ngram,
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
