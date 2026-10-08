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

// Upstream mtmd types, declared in mtmd.h.
struct mtmd_context_params;
struct mtmd_bitmap;
struct mtmd_input_chunk;
struct mtmd_input_chunks;
struct mtmd_input_text;

// Order in which exit teardown frees tracked objects: every object of a lower
// stage before any object of a higher one, so an object goes before the
// objects it uses.
enum llama_dart_exit_stage {
    // Per-request state over a context: speculative, TTS and sampler state.
    LLAMA_DART_EXIT_STAGE_SESSION = 0,
    // ggml schedulers.
    LLAMA_DART_EXIT_STAGE_SCHEDULER = 1,
    // llama.cpp contexts.
    LLAMA_DART_EXIT_STAGE_CONTEXT = 2,
    // Other objects that use a model, such as mtmd contexts and ggml buffers.
    LLAMA_DART_EXIT_STAGE_MODEL_USER = 3,
    // ggml backends.
    LLAMA_DART_EXIT_STAGE_BACKEND = 4,
    // Models.
    LLAMA_DART_EXIT_STAGE_MODEL = 5,
};

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

enum llama_dart_vulkan_status {
    LLAMA_DART_VULKAN_STATUS_OK = 0,
    LLAMA_DART_VULKAN_STATUS_INVALID_ARGUMENT = -1,
    // This libllamadart is built for a platform without ggml-vulkan, such as
    // the Apple platforms, and never looks for a Vulkan loader.
    LLAMA_DART_VULKAN_STATUS_UNSUPPORTED = -2,
    // The system has no Vulkan loader, or the loader lacks an entry point
    // that every Vulkan version has.
    LLAMA_DART_VULKAN_STATUS_NO_LOADER = -3,
    // The loader could not create an instance or list its devices, as when
    // no Vulkan driver is installed.
    LLAMA_DART_VULKAN_STATUS_LOADER_ERROR = -4,
    // There is no device with this index.
    LLAMA_DART_VULKAN_STATUS_NO_DEVICE = -5,
};

struct llama_dart_vulkan_device_info {
    // Set to sizeof(struct llama_dart_vulkan_device_info) before calling.
    uint32_t struct_size;
    // vkEnumerateInstanceVersion, or Vulkan 1.0 where the loader predates
    // that function. Versions are encoded as VK_MAKE_API_VERSION does: major
    // in bits 22-28, minor in bits 12-21.
    uint32_t instance_api_version;
    // Index of the device in vkEnumeratePhysicalDevices.
    uint32_t physical_device_index;
    // VkPhysicalDeviceProperties: apiVersion, driverVersion, vendorID,
    // deviceID and deviceType (a VkPhysicalDeviceType).
    uint32_t api_version;
    uint32_t driver_version;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t device_type;
    // VkPhysicalDeviceSubgroupProperties.subgroupSize, which is what
    // ggml-vulkan uses as the device's subgroup size. 0 when it cannot be
    // read: the loader or the device is older than Vulkan 1.1.
    uint32_t subgroup_size;
    // VkPhysicalDeviceProperties.deviceName, which is also the description
    // of the ggml device.
    char device_name[256];
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

// Exception barrier
//
// llama.cpp reports some failures by throwing a C++ exception, which must not
// cross the C ABI: in a Dart FFI caller it ends the process. Every function in
// this header that calls llama.cpp or allocates catches the exception instead,
// records its message for the calling thread and returns a failure value:
//
// - a function that returns a pointer returns NULL;
// - a function that returns bool returns false;
// - a function that returns an int32_t status or count returns
//   LLAMA_DART_STATUS_EXCEPTION, which none of them but llama_dart_tokenize
//   returns otherwise;
// - llama_dart_sampler_sample returns LLAMA_TOKEN_NULL, which it does not
//   return otherwise;
// - a function that returns size_t returns 0;
// - llama_dart_ggml_backend_sched_graph_compute returns GGML_STATUS_FAILED;
// - a function that returns llama_dart_tts_status returns
//   LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR and fails the task;
// - a function that returns nothing only records the message.
//
// Where the failure value is also what llama.cpp returns for a failure of its
// own, llama_dart_last_error tells the two apart: each of these functions
// clears the calling thread's last error when it is called, so after it
// returns, llama_dart_last_error is non-NULL only if it caught an exception.
// Read it from the same thread before anything else runs there.
//
// The functions that free an object (llama_dart_exit_free and the
// llama_dart_tts, speculative, mtp and ngram free functions) are the
// exception: they set the last error when they catch an exception and do not
// clear it otherwise, because a Dart finalizer may run one on a thread
// between a call that failed there and the read of its error. Call
// llama_dart_clear_last_error first to learn whether a free threw.
//
// The functions without a barrier are llama_dart_last_error,
// llama_dart_clear_last_error, llama_dart_set_log_level, the
// llama_dart_exit_ functions other than llama_dart_exit_track and
// llama_dart_exit_free, the draft context and need_embd getters,
// llama_dart_tts_eval_callback, the llama_dart_vulkan_ functions, which
// report through their own status, and the llama_dart_tts_ functions that only
// read or set fields of the task (api_version, request_default, cancel,
// get_output_info, read_pcm, last_error). They leave the last error
// unchanged.
//
// After a caught exception, the objects that were passed to the call may be
// partly updated. Free them, or reset a sampler, instead of continuing the
// generation they were used for. A function that creates an object frees
// what it had created before the exception, so there is nothing to free for
// a NULL result. A call in flight has ended when the function returns.
//
// The barrier does not cover a failed GGML_ASSERT or GGML_ABORT in llama.cpp,
// which abort the process, or a signal such as SIGSEGV.

// Returned instead of an int32_t status or count after a caught exception.
enum llama_dart_status {
    LLAMA_DART_STATUS_EXCEPTION = INT32_MIN,
};

// Message of the exception that the calling thread's last function with a
// barrier caught, or NULL when it caught none. The string is truncated to 511
// bytes and stays valid until the same thread calls another function with a
// barrier. Each thread has its own.
LLAMADART_API const char * llama_dart_last_error(void);

// Clears the calling thread's last error.
LLAMADART_API void llama_dart_clear_last_error(void);

// Vulkan device facts
//
// The devices that ggml-vulkan registers on this system, read from the Vulkan
// loader directly: the same devices in the same order, so index N here is
// ggml's device "VulkanN". The functions create a Vulkan instance and destroy
// it again. They do not load ggml-vulkan and create no logical device, so they
// are safe to call before deciding whether to use the Vulkan backend at all,
// and they work in a libllamadart whose bundle has no ggml-vulkan. The result
// is read once per process. GGML_VK_VISIBLE_DEVICES is honored as ggml does.
//
// ggml-vulkan requires Vulkan 1.2. It registers no device when
// instance_api_version is below 1.2; the devices are still listed here. It
// does register a device whose own api_version is below 1.2, and then calls
// Vulkan 1.2 functions that such a driver does not have, so check both.
//
// The order mirrors llama.cpp v0.6.0. Two cases can differ: a device below
// Vulkan 1.2 next to other GPUs, where ggml reads the 16-bit storage feature
// from a structure that such a driver does not fill in, and more than 16
// devices. device_name and the ids identify the device either way.

// Number of devices, or a negative llama_dart_vulkan_status when Vulkan
// cannot be queried. 0 when the loader works and ggml would use no device,
// as with only a CPU implementation such as lavapipe.
LLAMADART_API int32_t llama_dart_vulkan_get_device_count(void);

// Describes device index. Returns LLAMA_DART_VULKAN_STATUS_OK, the negative
// status that llama_dart_vulkan_get_device_count returns, or NO_DEVICE or
// INVALID_ARGUMENT. out_info is written only for OK.
LLAMADART_API int32_t llama_dart_vulkan_get_device_info(
    int32_t index,
    struct llama_dart_vulkan_device_info * out_info);

// Exit teardown
//
// libllamadart keeps a registry of live native objects and frees what is left
// of it when the process exits without freeing them itself, as a Flutter quit
// or hot restart does. ggml-metal aborts in its static destructor while any
// Metal buffer is still allocated, so teardown has to run before that
// destructor. On Apple platforms it runs during exit, before the first static
// of libllamadart is destroyed. Elsewhere it runs only when
// llama_dart_exit_teardown is called.
//
// Objects are tracked, before the creating call returns, by
// llama_dart_model_load_from_file, llama_dart_init_from_model,
// llama_dart_mtmd_init_from_file and the llama_dart_tts,
// llama_dart_speculative, llama_dart_mtp and llama_dart_ngram init functions,
// and by llama_dart_exit_track.
//
// Teardown waits a bounded time for the calls in flight and then frees the
// tracked objects in llama_dart_exit_stage order, latest tracked first within
// a stage. If a call is still in flight when the wait ends, it frees nothing.
// It leaves TTS, speculative and MTP state alone while the llama or mtmd
// context that state was created over is not tracked: a caller that creates
// contexts with the upstream functions may be using them in calls teardown
// cannot see. With nothing to free and no creating call or free in flight, it
// does not wait.
//
// A call in flight is the time a thread spends inside one of:
// - the three llama_dart creating functions named above and
//   llama_dart_exit_free;
// - llama_dart_decode, llama_dart_encode and the other llama_dart functions
//   below that wrap an upstream function of the same name, and the three
//   llama_dart_mtmd_bitmap_init functions;
// - the libllamadart functions that create, free, or run a task, draft or
//   batch on, TTS, speculative, MTP or n-gram state;
// - llama_dart_sampler_sample_and_accept_n;
// or between llama_dart_exit_call_begin and llama_dart_exit_call_end.
//
// Once teardown has begun, the objects a thread holds may be freed as soon as
// it has no call in flight. From then on, the end of a thread's outermost call
// in flight never returns to its caller. Neither does a function documented
// as "blocks after teardown" when it is called outside a call in flight;
// inside one it works as before, since teardown is waiting for that call. The
// thread that runs teardown is exempt: there, such a function returns without
// tracking, untracking or freeing anything. A thread blocked this way stays
// blocked until the process is gone, so a static destructor or atexit handler
// of another image that joins such a thread hangs the exit.
//
// A native call that is not a call in flight is not waited for. Teardown only
// allows a thread 250 ms after its last call in flight to reach the next one,
// which covers the short calls between two decodes. A longer call on a tracked
// object that is not a call in flight is a use after free at exit, also where
// exiting with the object alive was harmless. From Dart, make every such call
// through a llama_dart_ function that is a call in flight.

// Tracks object so that exit teardown frees it with free_fn. Tracking an
// address again replaces its entry. Returns false for a null argument or an
// unknown stage. Blocks after teardown.
LLAMADART_API bool llama_dart_exit_track(
    void * object,
    void (*free_fn)(void *),
    int32_t stage);

// Stops tracking object without freeing it. Returns whether it was tracked.
// Blocks after teardown.
LLAMADART_API bool llama_dart_exit_untrack(void * object);

// Stops tracking object and frees it with the function it was tracked with.
// Does nothing when object is not tracked, so each tracked object is freed
// once, whether by this call or by teardown. Usable as a Dart NativeFinalizer
// callback. Blocks after teardown.
LLAMADART_API void llama_dart_exit_free(void * object);

// Number of tracked objects.
LLAMADART_API int32_t llama_dart_exit_tracked_count(void);

// Mark the start and end of a call in flight on the calling thread, for C and
// C++ callers of native functions that libllamadart does not wrap. They nest.
// Each begin needs one end on the same thread: a thread that never reaches end
// keeps teardown from freeing anything. Do not call them from Dart, where an
// isolate that is killed during the native call in between never reaches end.
// begin blocks after teardown.
LLAMADART_API void llama_dart_exit_call_begin(void);
LLAMADART_API void llama_dart_exit_call_end(void);

// Sets how long teardown waits for calls in flight. Negative values are
// treated as zero. The default is 2000 ms.
LLAMADART_API void llama_dart_exit_set_wait_ms(int32_t wait_ms);

// Runs exit teardown now; later runs do nothing. Afterwards tracked objects
// are unusable and other threads that reach libllamadart stay blocked, so
// call it only as the last step before the process exits and follow it
// directly with exit or _exit on the same thread. It is meant for native
// hosts. Do not bind it from Dart: a Dart program that returns from main
// after it waits forever for its blocked isolates.
LLAMADART_API void llama_dart_exit_teardown(void);

// llama_model_load_from_file that tracks the model in the MODEL stage. The
// load is cancelled when teardown begins. llama.cpp gets a progress callback
// even when params has none, so it does not log its own progress dots. Free
// the model with llama_dart_exit_free. Blocks after teardown.
LLAMADART_API struct llama_model * llama_dart_model_load_from_file(
    const char * path_model,
    struct llama_model_params params);

// llama_init_from_model that tracks the context in the CONTEXT stage. Free it
// with llama_dart_exit_free. Blocks after teardown.
LLAMADART_API struct llama_context * llama_dart_init_from_model(
    struct llama_model * model,
    struct llama_context_params params);

// mtmd_init_from_file that tracks the context in the MODEL_USER stage. Free it
// with llama_dart_exit_free. Blocks after teardown.
LLAMADART_API struct mtmd_context * llama_dart_mtmd_init_from_file(
    const char * mmproj_fname,
    const struct llama_model * text_model,
    const struct mtmd_context_params * ctx_params);

// llama_decode and llama_encode as calls in flight that also wait for the
// backend to finish (llama_synchronize) before they return. Both block after
// teardown.
LLAMADART_API int32_t llama_dart_decode(
    struct llama_context * ctx,
    struct llama_batch batch);
LLAMADART_API int32_t llama_dart_encode(
    struct llama_context * ctx,
    struct llama_batch batch);

// The upstream functions of the same name, without the llama_dart_ prefix, as
// calls in flight. Each takes and returns what the upstream function does and
// blocks after teardown. The mtmd helpers that decode also wait for the llama
// context's backend before they return.
LLAMADART_API void llama_dart_synchronize(struct llama_context * ctx);

LLAMADART_API llama_token llama_dart_sampler_sample(
    struct llama_sampler * smpl,
    struct llama_context * ctx,
    int32_t idx);

// llama_sampler_accept as a call in flight. Returns false after a caught
// exception, such as the one a grammar sampler throws for a token that its
// grammar rejects, and true otherwise. Blocks after teardown.
LLAMADART_API bool llama_dart_sampler_accept(
    struct llama_sampler * smpl,
    llama_token token);

// The grammar constructor that compiles caller-supplied trigger patterns,
// which throws for a pattern that is not a valid regular expression. NULL
// after a caught exception, and for a grammar that does not parse.
// llama_sampler_init_grammar needs no wrapper: it reports a grammar that does
// not parse by returning NULL.
LLAMADART_API struct llama_sampler * llama_dart_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

// Returns what llama_tokenize returns: the number of tokens, or the negated
// number needed when n_tokens_max is too small. After a caught exception it
// returns LLAMA_DART_STATUS_EXCEPTION, which is also llama_tokenize's own
// value for a result of more than INT32_MAX tokens, so only
// llama_dart_last_error tells the two apart.
LLAMADART_API int32_t llama_dart_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special);

// Returns what llama_token_to_piece returns: the number of bytes, or the
// negated number needed when length is too small. LLAMA_DART_STATUS_EXCEPTION
// after a caught exception and in no other case. llama.cpp throws for a token
// that is not in the vocabulary, which includes LLAMA_TOKEN_NULL.
LLAMADART_API int32_t llama_dart_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token token,
    char * buf,
    int32_t length,
    int32_t lstrip,
    bool special);

// llama_memory_clear, which clears the backend's buffers when data is true.
// Returns false after a caught exception and true otherwise.
LLAMADART_API bool llama_dart_memory_clear(llama_memory_t mem, bool data);

LLAMADART_API bool llama_dart_state_save_file(
    struct llama_context * ctx,
    const char * path_session,
    const llama_token * tokens,
    size_t n_token_count);

LLAMADART_API bool llama_dart_state_load_file(
    struct llama_context * ctx,
    const char * path_session,
    llama_token * tokens_out,
    size_t n_token_capacity,
    size_t * n_token_count_out);

LLAMADART_API size_t llama_dart_state_seq_get_size_ext(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    uint32_t flags);

LLAMADART_API size_t llama_dart_state_seq_get_data_ext(
    struct llama_context * ctx,
    uint8_t * dst,
    size_t size,
    llama_seq_id seq_id,
    uint32_t flags);

LLAMADART_API size_t llama_dart_state_seq_set_data_ext(
    struct llama_context * ctx,
    const uint8_t * src,
    size_t size,
    llama_seq_id dest_seq_id,
    uint32_t flags);

// The adapter is not tracked: llama.cpp frees it with its model.
LLAMADART_API struct llama_adapter_lora * llama_dart_adapter_lora_init(
    struct llama_model * model,
    const char * path_lora);

// The mtmd constructors of an audio or image input, which allocate what they
// decode. Each returns the bitmap, or NULL when the input cannot be decoded
// or after a caught exception. Free it with mtmd_bitmap_free.
// llama_dart_mtmd_bitmap_init_from_buf and _from_file are
// mtmd_helper_bitmap_init_from_buf and _from_file with upstream's default
// options and without a placeholder; they return the bitmap of the result.
LLAMADART_API struct mtmd_bitmap * llama_dart_mtmd_bitmap_init_from_audio(
    size_t n_samples,
    const float * data);

LLAMADART_API struct mtmd_bitmap * llama_dart_mtmd_bitmap_init_from_buf(
    struct mtmd_context * ctx,
    const unsigned char * buf,
    size_t len);

LLAMADART_API struct mtmd_bitmap * llama_dart_mtmd_bitmap_init_from_file(
    struct mtmd_context * ctx,
    const char * fname);

LLAMADART_API int32_t llama_dart_mtmd_tokenize(
    const struct mtmd_context * ctx,
    struct mtmd_input_chunks * output,
    const struct mtmd_input_text * text,
    const struct mtmd_bitmap * const * bitmaps,
    size_t n_bitmaps);

LLAMADART_API int32_t llama_dart_mtmd_encode_chunk(
    struct mtmd_context * ctx,
    const struct mtmd_input_chunk * chunk);

LLAMADART_API int32_t llama_dart_mtmd_helper_eval_chunks(
    struct mtmd_context * ctx,
    struct llama_context * lctx,
    const struct mtmd_input_chunks * chunks,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * new_n_past);

LLAMADART_API int32_t llama_dart_mtmd_helper_eval_chunk_single(
    struct mtmd_context * ctx,
    struct llama_context * lctx,
    const struct mtmd_input_chunk * chunk,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * new_n_past);

LLAMADART_API int32_t llama_dart_mtmd_helper_decode_image_chunk(
    struct mtmd_context * ctx,
    struct llama_context * lctx,
    const struct mtmd_input_chunk * chunk,
    float * encoded_embd,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    llama_pos * new_n_past,
    int32_t (*callback)(struct llama_batch batch, void * user_data),
    void * user_data);

LLAMADART_API enum ggml_status llama_dart_ggml_backend_sched_graph_compute(
    ggml_backend_sched_t sched,
    struct ggml_cgraph * graph);

// The ggml functions that reach a backend, which a GPU backend may answer
// with an exception: ggml-vulkan throws for a device it does not support and
// for a Vulkan error. A function that returns a pointer returns NULL after a
// caught exception. The others return false after one; where upstream
// returns nothing they return true otherwise.
LLAMADART_API ggml_backend_t llama_dart_ggml_backend_dev_init(
    ggml_backend_dev_t device,
    const char * params);

LLAMADART_API ggml_backend_buffer_t llama_dart_ggml_backend_alloc_ctx_tensors(
    struct ggml_context * ctx,
    ggml_backend_t backend);

LLAMADART_API bool llama_dart_ggml_backend_tensor_set(
    struct ggml_tensor * tensor,
    const void * data,
    size_t offset,
    size_t size);

LLAMADART_API bool llama_dart_ggml_backend_tensor_get(
    const struct ggml_tensor * tensor,
    void * data,
    size_t offset,
    size_t size);

LLAMADART_API bool llama_dart_ggml_backend_sched_alloc_graph(
    ggml_backend_sched_t sched,
    struct ggml_cgraph * graph);

LLAMADART_API bool llama_dart_ggml_backend_sched_synchronize(
    ggml_backend_sched_t sched);

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

// May be called from another thread. A step returns CANCELLED when the cancel
// arrives before the step starts or before its frame generation or audio
// output returns. With llama_dart_tts_eval_callback installed, the audio
// decode stops at its next chunk boundary instead of running to the end.
LLAMADART_API void llama_dart_tts_cancel(struct llama_dart_tts * tts);

// Attaches a caller-owned cancel byte to the active task. Any thread may set
// *flag to nonzero. The task reads *flag only inside llama_dart_tts_step. Once
// a read sees nonzero, the task behaves as after llama_dart_tts_cancel, even if
// *flag returns to zero. The task drops the flag when it ends, is reset, or is
// freed; *flag must stay readable until then. Returns INVALID_STATE when no
// task is active.
LLAMADART_API enum llama_dart_tts_status llama_dart_tts_set_cancel_flag(
    struct llama_dart_tts * tts,
    const int8_t * flag);

// ggml_backend_sched_eval_callback for mtmd_context_params.cb_eval;
// cb_eval_user_data is unused. While llama_dart_tts_step runs on the calling
// thread, it splits mtmd graph computes into chunks and, once the task is
// cancelled, ends each split's compute at its next chunk boundary. Outside a
// step it requests no tensors.
LLAMADART_API bool llama_dart_tts_eval_callback(
    struct ggml_tensor * tensor,
    bool ask,
    void * user_data);

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
