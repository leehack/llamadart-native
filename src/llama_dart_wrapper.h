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

// Opaque experimental text-to-speech state owned by libllamadart.
struct llama_dart_tts;

// Opaque mtmd context supplied by the caller.
struct mtmd_context;

#define LLAMA_DART_TTS_API_VERSION 1

enum llama_dart_tts_status {
    LLAMA_DART_TTS_STATUS_OK = 0,
    LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT = -1,
    LLAMA_DART_TTS_STATUS_UNSUPPORTED = -2,
    LLAMA_DART_TTS_STATUS_INVALID_STATE = -3,
    LLAMA_DART_TTS_STATUS_SPEAKER_DECODE_FAILED = -4,
    LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR = -5,
    LLAMA_DART_TTS_STATUS_CANCELLED = -6,
};

enum llama_dart_tts_model_type {
    LLAMA_DART_TTS_MODEL_TYPE_NONE = 0,
    LLAMA_DART_TTS_MODEL_TYPE_QWEN3 = 1,
    LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN = 255,
};

enum llama_dart_tts_capability {
    LLAMA_DART_TTS_CAPABILITY_LANGUAGE = 1u << 0,
    LLAMA_DART_TTS_CAPABILITY_SPEAKER_REFERENCE = 1u << 1,
};

enum llama_dart_tts_state {
    LLAMA_DART_TTS_STATE_IDLE = 0,
    LLAMA_DART_TTS_STATE_PROCESSING_PROMPT = 1,
    LLAMA_DART_TTS_STATE_GENERATING = 2,
    LLAMA_DART_TTS_STATE_COMPLETED = 3,
    LLAMA_DART_TTS_STATE_CANCELLED = 4,
    LLAMA_DART_TTS_STATE_FAILED = 5,
};

struct llama_dart_tts_info {
    // Set to sizeof(struct llama_dart_tts_info) before calling get_info.
    uint32_t struct_size;
    uint32_t api_version;
    int32_t model_type;
    uint32_t capabilities;
    int32_t sample_rate;
    int32_t channels;
};

struct llama_dart_tts_request {
    // Set by llama_dart_tts_request_default. Callers should start from it.
    uint32_t struct_size;
    const char * text;
    size_t text_length;
    const unsigned char * speaker_audio;
    size_t speaker_audio_length;
    const char * language;
    llama_seq_id sequence_id;
    int32_t prompt_batch_size;
    int32_t max_frames;
    int32_t top_k;
    float top_p;
    float min_p;
    float temperature;
    uint32_t seed;
};

struct llama_dart_tts_progress {
    // Set to sizeof(struct llama_dart_tts_progress) before each step call.
    uint32_t struct_size;
    int32_t state;
    int32_t prompt_tokens_remaining;
    int32_t frames_generated;
    bool truncated;
};

struct llama_dart_tts_output_info {
    // Set to sizeof(struct llama_dart_tts_output_info) before calling.
    uint32_t struct_size;
    int32_t sample_rate;
    int32_t channels;
    int64_t sample_count;
};

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

// Returns the version of libllamadart's stable symbol contract around
// experimental upstream audio-generation internals.
LLAMADART_API uint32_t llama_dart_tts_api_version(void);

// Returns model-independent defaults for a TTS request.
LLAMADART_API struct llama_dart_tts_request llama_dart_tts_request_default(void);

// Reads audio-generation capability from an initialized mtmd context.
// Model type UNKNOWN is intentionally rejected by llama_dart_tts_init until a
// stable capability contract is defined for that upstream generator.
LLAMADART_API enum llama_dart_tts_status llama_dart_tts_get_info(
    const struct mtmd_context * mtmd,
    struct llama_dart_tts_info * out_info);

// Creates a TTS task wrapper. The caller retains the llama and mtmd contexts.
LLAMADART_API struct llama_dart_tts * llama_dart_tts_init(
    struct llama_context * llama,
    struct mtmd_context * mtmd,
    enum llama_dart_tts_status * out_status);

LLAMADART_API void llama_dart_tts_free(struct llama_dart_tts * tts);

// Starts one synthesis task. The llama context must have embeddings enabled.
// Other calls on the llama context must remain idle until the task completes,
// is cancelled, or is reset.
LLAMADART_API enum llama_dart_tts_status llama_dart_tts_start(
    struct llama_dart_tts * tts,
    const struct llama_dart_tts_request * request);

// Performs one prompt batch or one generation-frame step. Upstream currently
// exposes complete PCM only after generation ends; this is not chunked audio
// streaming.
LLAMADART_API enum llama_dart_tts_status llama_dart_tts_step(
    struct llama_dart_tts * tts,
    struct llama_dart_tts_progress * out_progress);

// May be called from another thread. Cancellation is observed between native
// prompt and generation steps.
LLAMADART_API void llama_dart_tts_cancel(struct llama_dart_tts * tts);

// Clears task state and the task sequence from the caller-owned llama context.
LLAMADART_API enum llama_dart_tts_status llama_dart_tts_reset(
    struct llama_dart_tts * tts);

LLAMADART_API enum llama_dart_tts_status llama_dart_tts_get_output_info(
    const struct llama_dart_tts * tts,
    struct llama_dart_tts_output_info * out_info);

// Copies float32 mono PCM from the completed task. out_samples may be NULL to
// query the number of samples remaining from sample_offset.
LLAMADART_API enum llama_dart_tts_status llama_dart_tts_read_pcm(
    const struct llama_dart_tts * tts,
    int64_t sample_offset,
    float * out_samples,
    size_t out_capacity,
    size_t * out_count);

// Returns a task-owned diagnostic string, valid until the next task call.
LLAMADART_API const char * llama_dart_tts_last_error(
    const struct llama_dart_tts * tts);

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

// Legacy ABI query. Current upstream speculative implementations configure
// their required target outputs during initialization.
LLAMADART_API bool llama_dart_speculative_need_embd(
    struct llama_dart_speculative * speculative);

// Preserves the historical true result for MTP sessions. Current upstream
// configures next-token embeddings during speculative initialization.
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
