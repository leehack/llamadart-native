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

// Opaque extended batch (also declared by newer upstream headers).
struct llama_batch_ext;

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
    // Per-request state over a context: speculative, TTS, live ASR and sampler
    // state.
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
    // Reported only to callers that accept it, see llama_dart_tts_get_info_ext.
    LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO = 2,
    // Reported only to callers that accept it, see llama_dart_tts_get_info_ext.
    LLAMA_DART_TTS_MODEL_TYPE_SOPRANO = 3,
    // Reported only to callers that accept it, see llama_dart_tts_get_info_ext.
    LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX = 4,
    // Reported only to callers that accept it, see llama_dart_tts_get_info_ext.
    LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS = 5,
    LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN = 255,
};

enum llama_dart_tts_capability {
    LLAMA_DART_TTS_CAPABILITY_LANGUAGE = 1u << 0,
    LLAMA_DART_TTS_CAPABILITY_SPEAKER_REFERENCE = 1u << 1,
    LLAMA_DART_TTS_CAPABILITY_VOICE = 1u << 2,
    // llama_dart_tts_start_streaming is accepted, see its contract. Set for
    // Qwen3-TTS, LFM3-Audio and Pocket TTS.
    LLAMA_DART_TTS_CAPABILITY_STREAMING_PCM = 1u << 3,
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
// - a function that returns llama_dart_asr_stream_status returns
//   LLAMA_DART_ASR_STREAM_STATUS_UPSTREAM_ERROR and fails the session;
// - a function that returns nothing only records the message.
//
// Where the failure value is also what llama.cpp returns for a failure of its
// own, llama_dart_last_error tells the two apart: each of these functions
// clears the calling thread's last error when it is called, so after it
// returns, llama_dart_last_error is non-NULL only if it caught an exception.
// Read it from the same thread before anything else runs there.
//
// The functions that free an object (llama_dart_exit_free and the
// llama_dart_tts, llama_dart_asr_stream and speculative free functions) are the
// exception: they set the last error when they catch an exception and do not
// clear it otherwise, because a Dart finalizer may run one on a thread
// between a call that failed there and the read of its error. Call
// llama_dart_clear_last_error first to learn whether a free threw.
//
// The functions without a barrier are llama_dart_build_identity,
// llama_dart_last_error, llama_dart_clear_last_error,
// llama_dart_set_log_level, the
// llama_dart_exit_ functions other than llama_dart_exit_track and
// llama_dart_exit_free, the draft context getter and
// llama_dart_speculative_need_embd_nextn,
// llama_dart_tts_eval_callback, llama_dart_mtmd_eval_callback,
// llama_dart_cancel_flag_abort_callback, the llama_dart_vulkan_ functions,
// which report through their own status, and the llama_dart_tts_ functions
// that only read or set fields of the task (api_version, request_default,
// supported_model_types, cancel, get_output_info, read_pcm, last_error), and
// the llama_dart_asr_stream_ functions api_version, supported_modes,
// params_default, cancel and last_error.
// They leave the last error unchanged.
//
// After a caught exception, the objects that were passed to the call may be
// partly updated. What a caller may still do with each of them:
// - A sampler or sampler chain (llama_dart_sampler_sample, _accept and
//   _sample_and_accept_n): free it, or call llama_sampler_reset on it and
//   start the generation over. The context it sampled from is unchanged.
// - A llama context (llama_dart_decode, llama_dart_encode, llama_dart_process,
//   llama_dart_synchronize, llama_dart_memory_clear, the state functions and
//   the mtmd evaluation helpers): free it. Its memory and its outputs are in
//   no defined state.
// - An mtmd context (llama_dart_mtmd_tokenize, llama_dart_mtmd_encode_chunk,
//   llama_dart_mtmd_encode_chunk_cancellable and the evaluation helpers): free
//   it.
// - A model whose adapter failed to load (llama_dart_adapter_lora_init): free
//   it.
// - Speculative, MTP and n-gram state: free it.
// - A TTS task: it is FAILED. Free it and the llama context it ran on.
// - A live ASR session: it is FAILED. Free it and the llama context it ran on.
// - A ggml backend, scheduler or buffer (the llama_dart_ggml_backend_
//   functions other than the two device queries): free it.
// - Nothing changes for the model or vocabulary of llama_dart_tokenize,
//   llama_dart_token_to_piece and the grammar constructor, for the mtmd
//   context of the bitmap constructors, or for the device of the two device
//   queries, or for the context/model of llama_dart_batch_ext_init: these
//   calls only read them, and they stay usable.
// A function that creates an object frees what it had created before the
// exception, so there is nothing to free for a NULL result. A call in flight
// has ended when the function returns.
//
// On Windows the rule is stricter. llama.cpp's own libraries are compiled
// with /EHsc, so while an exception unwinds through one of their extern "C"
// functions the destructors of that function's locals may not run, and memory
// or a lock may stay held. After a caught exception there, free every object
// that was passed to the call, a sampler, a model and an mtmd context
// included, and neither reset nor reuse it.
//
// The barrier does not cover a failed GGML_ASSERT or GGML_ABORT in llama.cpp,
// which abort the process, or a signal such as SIGSEGV. Do not cancel a
// thread that is inside libllamadart: glibc cancels by a forced unwind, which
// the barrier would catch without rethrowing, and that ends the process.

// Returned instead of an int32_t status or count after a caught exception.
enum llama_dart_status {
    LLAMA_DART_STATUS_EXCEPTION = INT32_MIN,
};

// Message of the exception that the calling thread's last function with a
// barrier caught, or NULL when it caught none. The string is valid UTF-8 of
// at most 511 bytes: a byte of the exception's message that is not part of a
// well-formed sequence is replaced by U+FFFD, and a longer message is cut
// between characters. It stays valid until the same thread calls another
// function with a barrier. Each thread has its own.
LLAMADART_API const char * llama_dart_last_error(void);

// Clears the calling thread's last error.
LLAMADART_API void llama_dart_clear_last_error(void);

// Build identity
//
// Names the inputs this library was built from: the llamadart-native and
// llama.cpp commits with a digest of any uncommitted change, the ggml tree,
// the carried patches that applied, the build configuration, compiler,
// target, ggml backends and ggml build switches. Two libraries that return
// the same non-empty string were built from the same inputs; a different
// string means a state file that one saved may not load correctly in the
// other. The string holds no path, host name or time. It is empty when the
// build could not name its inputs (no git checkout), and an empty identity
// matches nothing, itself included. The string is a constant in the library,
// valid while the library is loaded; the call touches no llama.cpp object or
// state, so it needs no exit teardown guard. See docs/build_identity.md.
LLAMADART_API const char * llama_dart_build_identity(void);

// Vulkan device facts
//
// The devices that ggml-vulkan registers on this system, read from the Vulkan
// loader directly: the same devices in the same order, so index N here is
// ggml's device "VulkanN". The functions create a Vulkan instance and destroy
// it again. They do not load ggml-vulkan and create no logical device, and
// they work in a libllamadart whose bundle has no ggml-vulkan. The result is
// read once per process. GGML_VK_VISIBLE_DEVICES is honored as ggml does.
//
// Creating the instance makes the Vulkan loader load the system's GPU drivers
// and layers into the process, as registering ggml-vulkan does. A driver that
// crashes there takes the process down, and nothing here can catch that. Call
// these functions only when the Vulkan backend is being considered, not as a
// general probe, and not on a device where Vulkan is ruled out for other
// reasons.
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
// of libllamadart is destroyed.
//
// On Linux, Android excepted, owned bundle statics are retained until the
// process is gone. C exit() runs no wrapper handler: normal host callbacks,
// dependency destructors and libc stream flushing proceed without a forced
// _exit, an automatic free, a wait or late-call refusal.
//
// Before a native host calls C exit(), it must stop and join its workers and
// shut down its Dart isolates / Flutter engine. Await isolate exit and native
// finalizer completion, not just a kill request. A direct C exit() with live
// Dart isolates can abort the VM even without llama.cpp (llamadart#977).
// Retaining owned statics does not protect a GPU driver's or BLAS library's
// destructors. Do not call explicit teardown before joining workers: teardown
// parks later guarded calls and can deadlock a host join.
//
// libllamadart remains loaded across dlclose so held native objects and their
// finalizer callbacks stay valid. quick_exit and _exit run no cleanup.
// Elsewhere nothing changes. llama_dart_exit_teardown explicitly frees on
// every platform; automatic Apple exit teardown remains as described above.
//
// Objects are tracked, before the creating call returns, by
// llama_dart_model_load_from_file, llama_dart_init_from_model,
// llama_dart_mtmd_init_from_file and the llama_dart_tts,
// llama_dart_asr_stream and llama_dart_speculative init functions,
// and by llama_dart_exit_track.
//
// Teardown waits a bounded time for the calls in flight and then frees the
// tracked objects in llama_dart_exit_stage order, latest tracked first within
// a stage. If a call is still in flight when the wait ends, it frees nothing.
// It leaves TTS, live ASR, speculative and MTP state alone while the llama or
// mtmd context that state was created over is not tracked: a caller that creates
// contexts with the upstream functions may be using them in calls teardown
// cannot see. With nothing to free and no creating call or free in flight, it
// does not wait.
//
// A call in flight is the time a thread spends inside one of:
// - the three llama_dart creating functions named above and
//   llama_dart_exit_free;
// - llama_dart_decode, llama_dart_encode, llama_dart_process and the other
//   llama_dart functions below that wrap an upstream function of the same
//   name, llama_dart_mtmd_encode_chunk_cancellable, and the three
//   llama_dart_mtmd_bitmap_init functions;
// - the libllamadart functions that create, free, or run a task, draft or
//   batch on, TTS, live ASR, speculative, MTP or n-gram state;
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
// Creation-time owner API v1. The parent creates a pending token before
// spawning a killable worker. Attach a stable worker finalizer using release,
// then activate before creating payloads. Abandon closes pending tokens only;
// an active token is released after the worker stops or cooperatively disposes.
// Tokens are opaque never-reused identities; never dereference them. Duplicate
// release is harmless. Release waits for admitted creators/ordinary frees and
// drains registered objects in exit-stage order outside the registry mutex.
// It must not race ordinary inference on the owner; stop the worker first.
LLAMADART_API uint32_t llama_dart_owner_api_version(void);
LLAMADART_API void * llama_dart_owner_create(void);
LLAMADART_API bool llama_dart_owner_activate(void * owner);
LLAMADART_API void llama_dart_owner_abandon(void * owner);
LLAMADART_API void llama_dart_owner_release(void * owner);
LLAMADART_API int32_t llama_dart_owner_count(void);
// Adopts an existing entry without replacing its cleanup identity, or tracks a
// raw object. Returns 2 for existing tracking, 1 for new tracking, 0 on failure.
// Adoption does not close an allocation-to-adoption gap; use owned creators.
LLAMADART_API int32_t llama_dart_owner_track(
    void * owner, void * object, void (*free_fn)(void *), int32_t stage);
// For adopted raw objects only, immediately before their ordinary raw free.
LLAMADART_API bool llama_dart_owner_untrack(void * owner, void * object);
LLAMADART_API struct llama_model * llama_dart_owner_model_load_from_file(
    void * owner, const char * path, struct llama_model_params params);
LLAMADART_API struct llama_context * llama_dart_owner_init_from_model(
    void * owner, struct llama_model * model, struct llama_context_params params);
LLAMADART_API struct mtmd_context * llama_dart_owner_mtmd_init_from_file(
    void * owner, const char * path, const struct llama_model * model,
    const struct mtmd_context_params * params);

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

// Capability probe for an explicit mtmd_context_params.image_max_tokens.
// Reads projector metadata only; does not load weights or initialize a GPU.
// Returns 1 for a Gemma 4 vision projector (gemma4v/gemma4uv) and a limit
// in [70, 1120], 0 for an unsupported projector/limit, -1 for a read error
// or invalid arguments (llama_dart_last_error describes the error).
// It is an upstream resize budget: minimum-size/patch alignment can exceed
// it, especially for extreme aspect ratios. Callers enforcing a ceiling must
// inspect image-chunk token counts before encoding. It does not limit prompt
// or marker tokens. Defaults
// are unchanged unless the caller explicitly sets image_max_tokens.
LLAMADART_API int32_t llama_dart_mtmd_supports_image_token_limit(
    const char * mmproj_fname, int32_t max_tokens);

// GPU memory fit, API v1. Wraps llama.cpp's common_fit_params, which is not
// part of libllama's C API, and measures what the fitted parameters need per
// device. Bindings must check llama_dart_fit_api_version() before use.
#define LLAMA_DART_FIT_MAX_DEVICES 16

enum llama_dart_fit_status {
    // The parameters are projected to fit, and may be unchanged. A fit that
    // leaves nothing on the GPU is a success with n_gpu_layers == 0.
    LLAMA_DART_FIT_STATUS_SUCCESS = 0,
    // No fitting allocation exists, or the parameters need a change that is
    // not allowed: n_gpu_layers was set, or the parameters ask for a tensor
    // or row split. Parameters that already fit are never refused. On a host
    // without a GPU device a margin above total memory is also a failure.
    // mparams and cparams are left as they were; the tensor_split and
    // tensor_buft_overrides buffers are unspecified.
    LLAMA_DART_FIT_STATUS_FAILURE = 1,
    // A hard error, such as a model that cannot be read.
    LLAMA_DART_FIT_STATUS_ERROR = 2,
};

// Memory of one device, or of the host, at the fitted parameters. The host
// entry is named "Host" and has the total and free memory of the CPU device.
// On Windows, free_bytes of a Vulkan GPU is the smaller of what the driver
// reports and what the display adapter has free for every process. The driver
// reports a per-process budget, which does not deduct memory that other
// processes hold.
struct llama_dart_fit_device {
    char name[64];
    int64_t total_bytes;
    int64_t free_bytes;
    uint64_t model_bytes;
    uint64_t context_bytes;
    uint64_t compute_bytes;
};

struct llama_dart_fit_report {
    // Entries of devices in use: every non-CPU device of the model, also one
    // that holds nothing at the fitted parameters; at most
    // LLAMA_DART_FIT_MAX_DEVICES.
    uint32_t n_devices;
    // Repeating layers, as llama_model_n_layer: the output layer and the
    // NextN/MTP layers are not counted.
    uint32_t n_layers;
    uint32_t n_ctx_train;
    // NextN/MTP layers that the loader counts in n_gpu_layers but n_layers
    // leaves out; 0 before v1.0.0 and for a model without them. A fit that
    // changes n_gpu_layers sets it on the loader's scale, so all of the model
    // is offloaded exactly when
    // n_gpu_layers >= n_layers + n_layers_skipped + 1.
    uint32_t n_layers_skipped;
    struct llama_dart_fit_device devices[LLAMA_DART_FIT_MAX_DEVICES];
    struct llama_dart_fit_device host;
};

// Returns 1.
LLAMADART_API uint32_t llama_dart_fit_api_version(void);

// Fits mparams and cparams to the free memory of the GPU devices, measuring
// with no_alloc loads of path_model, and returns a llama_dart_fit_status, or
// LLAMA_DART_STATUS_EXCEPTION after a caught exception or for a null argument
// (llama_dart_last_error describes it). It is a call in flight that tracks
// what it creates: it blocks after teardown, and teardown waits for it.
//
// Only parameters that have their llama_model_default_params value change
// (n_gpu_layers must be -1), and n_ctx changes only when it is 0, down to
// n_ctx_min; pass UINT32_MAX to keep the context. On success mparams has
// n_gpu_layers, tensor_split and tensor_buft_overrides set when the load
// needed a change, and cparams n_ctx. A GPU load that offloads nothing
// returns with n_gpu_layers == 0. A load that fits as it is leaves
// n_gpu_layers at -1 and mparams pointing where it did. On a host without a
// GPU device a context of 0 comes back as the trained context size.
//
// tensor_split needs llama_max_devices() elements, tensor_buft_overrides
// llama_max_tensor_buft_overrides(), and margins llama_max_devices() elements
// of bytes to leave free per device. When the fit changed the load, mparams
// points into the caller's tensor_split and tensor_buft_overrides, which must
// outlive the model load that uses mparams. The override patterns they point
// at are static storage inside llama-common that stays valid while the library
// is loaded and must not be freed. report is filled on success and may be NULL.
//
// Fits are serialized with each other and with llama_dart_set_log_level, which
// a log callback must therefore not call: it would wait for the fit that is
// running it. llama.cpp swaps the global log callback while it measures;
// the carried llama.cpp patch 0007 keeps that state in static storage, so a
// thread that logs meanwhile reaches the original callback, and lowers only
// the fit's own log lines. llama_log_set itself is not atomic: a thread that
// logs in the instant the callback is installed or restored, or that calls
// llama_log_set, still races with it.
//
// On Windows the free memory of a Vulkan or CUDA GPU is the smaller of the
// driver's number and the display adapter's free dedicated memory for all
// processes (D3DKMTQueryStatistics, found by the device's LUID), applied as
// extra margin. A Vulkan or CUDA GPU of the model whose adapter memory cannot
// be read fails the fit with LLAMA_DART_STATUS_EXCEPTION (including TCC,
// GGML_CUDA_DEVICES virtual devices, and unreadable nvcuda.dll or D3DKMT): the
// fit never runs on the driver's number alone there. It reads the model's device
// list with a vocabulary-only load first. Elsewhere, and for other devices,
// nothing changes. The version stays 1.
//
// Teardown cannot cancel a fit, and waits for one that is running.
LLAMADART_API int32_t llama_dart_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    const size_t * margins,
    uint32_t n_ctx_min,
    struct llama_dart_fit_report * report);

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

// llama_batch_ext_init reads ctx and its model, so it is a call in flight.
// Returns NULL after a caught exception, or without an error when built against
// upstream headers without the extended-batch API. The batch is caller-owned;
// it is untracked. Free it with llama_batch_ext_free, not llama_dart_exit_free.
// Blocks after teardown. At llama.cpp v0.7.0 upstream batch add/set/clear/free
// only use batch-owned storage (embedding data is copied); they need no exit
// guard. Keep the batch exclusive to this thread until process returns, and
// process it only with the live context it was initialized from.
LLAMADART_API struct llama_batch_ext * llama_dart_batch_ext_init(
    struct llama_context * ctx);

// Matches upstream LLAMA_PROCESS_TYPE_DECODE. int32_t in the signature allows
// the same header and exports with older upstream headers that lack this enum.
enum llama_dart_process_type {
    LLAMA_DART_PROCESS_TYPE_DECODE = 1,
};

enum llama_dart_process_status {
    // Also an upstream invalid-batch status; inspect the requested type.
    LLAMA_DART_PROCESS_STATUS_UNSUPPORTED_TYPE = -1,
    // Built against upstream headers without llama_process / extended batches.
    LLAMA_DART_PROCESS_STATUS_UNSUPPORTED = -4,
};

// llama_process as a call in flight, with the same exception barrier and
// backend wait as llama_dart_decode, including when processing throws.
// Only LLAMA_DART_PROCESS_TYPE_DECODE (upstream DECODE) is accepted. All other
// types, including upstream ENCODE, return
// LLAMA_DART_PROCESS_STATUS_UNSUPPORTED_TYPE without accessing ctx or batch;
// this is not an exception and clears the last error. Supported calls pass
// upstream statuses through; exceptions return LLAMA_DART_STATUS_EXCEPTION.
// When the upstream API is absent, DECODE returns the typed UNSUPPORTED status.
// ctx and batch must be valid for supported DECODE. The caller owns the
// batch. Blocks after teardown, including for unsupported types.
LLAMADART_API int32_t llama_dart_process(
    struct llama_context * ctx,
    int32_t type,
    struct llama_batch_ext * batch);

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

// Cancel byte
//
// A caller cancels a request by setting a byte it owns to nonzero, from any
// thread. These functions let the byte stop the llama.cpp decode or clip
// encode in flight. Each read of the byte is a relaxed atomic load.

// Returned by llama_dart_mtmd_encode_chunk_cancellable for an encode that the
// cancel byte stopped or kept from starting.
enum llama_dart_mtmd_encode_status {
    LLAMA_DART_MTMD_ENCODE_CANCELLED = 2,
};

// ggml_abort_callback for llama_set_abort_callback, with the cancel byte as
// abort_callback_data: true when the byte is nonzero, false for NULL data.
// Once the byte is nonzero, a decode or encode on the context stops after the
// node being computed and returns 2. On a transformer model llama_decode
// removes the positions of the ubatch it stopped and keeps those of earlier
// ubatches. On a recurrent or hybrid model (llama_model_is_recurrent,
// llama_model_is_hybrid) the sequence's memory is undefined after a 2,
// because upstream cannot roll recurrent state back: clear the sequence and
// prefill it again. Upstream installs the callback only on the CPU backend,
// so a graph on a GPU backend runs to its end, though a multi-ubatch decode
// still stops at the next ubatch's CPU split. On a context that
// llama_dart_tts_* drives, a stop during the prompt step fails the task with
// LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR rather than cancelling it; reset the
// task to reuse it. The byte must stay readable until
// llama_set_abort_callback(ctx, NULL, NULL) returns or the context is freed.
// The callback touches no llama.cpp object.
LLAMADART_API bool llama_dart_cancel_flag_abort_callback(void * data);

// ggml_backend_sched_eval_callback for mtmd_context_params.cb_eval, in place
// of llama_dart_tts_eval_callback; cb_eval_user_data is unused. While
// llama_dart_tts_step runs on the calling thread it answers as
// llama_dart_tts_eval_callback does. While
// llama_dart_mtmd_encode_chunk_cancellable runs on the calling thread, it
// splits the encode's graph computes into chunks as the TTS callback does and,
// once the call's cancel byte is nonzero, ends each scheduler split's compute
// after its next computed node. Otherwise it requests no tensors.
LLAMADART_API bool llama_dart_mtmd_eval_callback(
    struct ggml_tensor * tensor,
    bool ask,
    void * user_data);

// mtmd_encode_chunk that *cancel_flag can stop. Returns
// LLAMA_DART_MTMD_ENCODE_CANCELLED without encoding when the byte is nonzero
// on entry, and after an encode that llama_dart_mtmd_eval_callback stopped;
// mtmd_get_output_embd then holds incomplete embeddings, and the mtmd context
// stays usable. The encode stops early only when ctx was created with
// llama_dart_mtmd_eval_callback as cb_eval; with another callback the byte is
// read on entry only. A NULL cancel_flag encodes as
// llama_dart_mtmd_encode_chunk does. The byte must stay readable until the
// call returns.
LLAMADART_API int32_t llama_dart_mtmd_encode_chunk_cancellable(
    struct mtmd_context * ctx,
    const struct mtmd_input_chunk * chunk,
    const int8_t * cancel_flag);

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

// Both queries ask the backend again: ggml-vulkan lists the physical devices
// each time. After a caught exception they return false and the outputs are
// zero. On Windows the free memory of a Vulkan or CUDA GPU is the smaller of
// the driver's number and the adapter's free dedicated memory for all
// processes; when that cannot be read, the driver's number is returned.
LLAMADART_API bool llama_dart_ggml_backend_dev_memory(
    ggml_backend_dev_t device,
    size_t * free,
    size_t * total);

LLAMADART_API bool llama_dart_ggml_backend_dev_get_props(
    ggml_backend_dev_t device,
    struct ggml_backend_dev_props * props);

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
// is cancelled, or is reset. request->seed seeds every codebook: the talker
// sampler and the projector's code predictor. LLAMA_DEFAULT_SEED draws one
// random seed per task. The same request and seed give the same PCM on every
// task, including after earlier or cancelled tasks on the same mtmd context.
LLAMADART_API enum llama_dart_tts_status llama_dart_tts_start(
    struct llama_dart_tts * tts,
    const struct llama_dart_tts_request * request);

// Performs one prompt batch or one generation-frame step. For a task started
// with llama_dart_tts_start or llama_dart_tts_start_with_voice, upstream
// exposes complete PCM only after generation ends. For a task started with
// llama_dart_tts_start_streaming, a step may append decoded PCM, see there.
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

// For a task started with llama_dart_tts_start_streaming this is also valid
// while the task is in PROCESSING_PROMPT or GENERATING, with sample_count the
// samples produced so far. Otherwise it is valid only in COMPLETED. It is
// INVALID_STATE after CANCELLED, FAILED and llama_dart_tts_reset.
LLAMADART_API enum llama_dart_tts_status llama_dart_tts_get_output_info(
    const struct llama_dart_tts * tts,
    struct llama_dart_tts_output_info * out_info);

// Copies float32 mono PCM from the completed task. out_samples may be NULL to
// query the number of samples remaining from sample_offset. For a task started
// with llama_dart_tts_start_streaming the PCM is also readable while the task
// generates: it only grows, and samples already readable never change, so a
// caller reads from the offset it last reached. Not readable in CANCELLED,
// FAILED or after llama_dart_tts_reset (INVALID_STATE). Both reads run behind
// the exit-teardown guard like every other llama_dart_tts_ call. As for every
// task call, do not overlap them with llama_dart_tts_step on the same task.
LLAMADART_API enum llama_dart_tts_status llama_dart_tts_read_pcm(
    const struct llama_dart_tts * tts,
    int64_t sample_offset,
    float * out_samples,
    size_t out_capacity,
    size_t * out_count);

// Returns a task-owned diagnostic string, valid until the next task call.
LLAMADART_API const char * llama_dart_tts_last_error(
    const struct llama_dart_tts * tts);

// Model-type opt-in (llamadart-native#130). llama_dart_tts_get_info and
// llama_dart_tts_init accept QWEN3 only and report every other generator as
// UNKNOWN. The _ext variants take a mask of (1u << model type) naming every
// type the caller handles; a type outside the mask, or one this build cannot
// drive, is reported and refused exactly as those exports do.
//
// LFM3_AUDIO (LFM3-Audio and LFM3-Omni) speaks request->text through the
// model's "tts" chat template, rendered by llama.cpp with request->language
// (a code such as "en" or a language name; NULL or "" for the template's
// default, English) and the voice of llama_dart_tts_start_with_voice. It
// rejects speaker references with UNSUPPORTED, samples audio codes with the
// model's own GGUF settings (request top_k, top_p, min_p and temperature are
// validated but unused). It has no text/audio interleaving or sequence
// continuation, and streams PCM only through llama_dart_tts_start_streaming.
//
// SOPRANO (llamadart-native#123, a Qwen3 backbone with a Vocos decoder) has
// no capability bits: llama_dart_tts_start returns UNSUPPORTED for a speaker
// reference or a non-empty language. Frame steps decode backbone tokens only;
// the step that ends generation (end of speech, max_frames or the end of the
// context) decodes the whole utterance's audio, and with
// llama_dart_tts_eval_callback installed a cancel stops that decode at its
// next chunk boundary. Fewer than two frames fail that step with
// UPSTREAM_ERROR. request->seed seeds the backbone sampler, the only sampler.

// CHATTERBOX (Chatterbox Turbo and Multilingual V3) and POCKET_TTS speak
// with a predefined voice of the projector or clone a speaker reference;
// see llama_dart_tts_get_voice_count. A voice and a speaker reference together
// are INVALID_ARGUMENT, and a speaker reference needs
// LLAMA_DART_TTS_CAPABILITY_SPEAKER_REFERENCE, else UNSUPPORTED. With neither,
// the first predefined voice speaks, as in upstream llama-tts; a Chatterbox
// without predefined voices then returns INVALID_ARGUMENT, and Pocket TTS
// speaks unconditioned. Chatterbox returns complete PCM at the end of the task;
// POCKET_TTS can stream, see llama_dart_tts_start_streaming.
//
// CHATTERBOX samples backbone tokens with the request's sampling settings;
// request->seed seeds that sampler and the flow decoder. It decodes the whole
// utterance's audio on the step that ends generation, which a cancel
// interrupts as for SOPRANO. Turbo has no capability bit for a language and
// rejects one. Multilingual reports LANGUAGE: request->language is a code
// such as "fr", not validated, that upstream lower-cases and prefixes to the
// text. Multilingual decodes an unconditional twin of the task sequence
// for classifier-free guidance at sequence n_seq_max / 2 + sequence_id, which
// the task owns and clears like its own: llama_dart_tts_init_ext returns
// UNSUPPORTED for a context with n_seq_max < 2, and a request whose
// sequence_id is not below n_seq_max / 2 is INVALID_ARGUMENT. Without
// kv_unified each of the two sequences holds n_ctx / n_seq_max positions.
//
// Chatterbox relies on the talker sampler's GGUF suppress tokens to exclude
// its text tokens.
//
// POCKET_TTS generates from the hidden state with the sampling settings of
// its weights: request top_k, top_p, min_p and temperature are validated but
// unused, and request->seed seeds the flow decoder. It has no language
// capability: the language is that of the weights.

// Returns the mask of model types llama_dart_tts_init_ext can create in this
// build. QWEN3 is always set; LFM3_AUDIO, SOPRANO, CHATTERBOX and POCKET_TTS
// only when the bundled mtmd has their generators.
LLAMADART_API uint32_t llama_dart_tts_supported_model_types(void);

LLAMADART_API enum llama_dart_tts_status llama_dart_tts_get_info_ext(
    const struct mtmd_context * mtmd,
    uint32_t accepted_model_types,
    struct llama_dart_tts_info * out_info);

// LFM3_AUDIO also needs the model's "tts" chat template, else UNSUPPORTED.
LLAMADART_API struct llama_dart_tts * llama_dart_tts_init_ext(
    struct llama_context * llama,
    struct mtmd_context * mtmd,
    uint32_t accepted_model_types,
    enum llama_dart_tts_status * out_status);

// llama_dart_tts_start with a voice; NULL is the model's default voice and
// otherwise identical to llama_dart_tts_start. A voice needs
// LLAMA_DART_TTS_CAPABILITY_VOICE, else UNSUPPORTED. It is one of the names
// llama_dart_tts_get_voice_name lists, matched in any case; any other name is
// INVALID_ARGUMENT, never a fallback to the default. LFM3_AUDIO voices are
// "female" (default) and "male".
LLAMADART_API enum llama_dart_tts_status llama_dart_tts_start_with_voice(
    struct llama_dart_tts * tts,
    const struct llama_dart_tts_request * request,
    const char * voice);

// llama_dart_tts_start_with_voice (voice may be NULL) for a task whose PCM is
// readable while it generates. Requires LLAMA_DART_TTS_CAPABILITY_STREAMING_PCM,
// else UNSUPPORTED with the task and the llama context untouched. Soprano and
// Chatterbox decode their whole utterance at the end and never report it.
// Each llama_dart_tts_step appends the PCM decoded since the previous one, which
// is nothing until the model's decode window fills (Qwen3-TTS 72 frames, Pocket
// TTS 8 frames and each text chunk, LFM3-Audio every frame). The step that ends
// generation (end of speech or max_frames) flushes the decoder and appends its
// tail, so the COMPLETED PCM equals that of llama_dart_tts_start with the same
// request and seed. A step that observes a cancel appends nothing. Does not
// change any struct, status value or llama_dart_tts_api_version.
LLAMADART_API enum llama_dart_tts_status llama_dart_tts_start_streaming(
    struct llama_dart_tts * tts,
    const struct llama_dart_tts_request * request,
    const char * voice);

// Writes the number of voices llama_dart_tts_start_with_voice accepts for the
// projector, 0 for a model without LLAMA_DART_TTS_CAPABILITY_VOICE. A model
// type outside accepted_model_types, or one this build cannot drive, is
// UNSUPPORTED as in llama_dart_tts_get_info_ext, with *out_count 0.
LLAMADART_API enum llama_dart_tts_status llama_dart_tts_get_voice_count(
    const struct mtmd_context * mtmd,
    uint32_t accepted_model_types,
    int32_t * out_count);

// Copies the index-th voice name, NUL-terminated, into out_name and writes
// its length without the NUL to *out_length. out_name may be NULL with
// out_capacity 0 to query the length. An index outside
// [0, voice count), or an out_capacity not above the length, is
// INVALID_ARGUMENT; the latter still writes *out_length. The order is
// upstream's; index 0 is the voice used when a request names none.
LLAMADART_API enum llama_dart_tts_status llama_dart_tts_get_voice_name(
    const struct mtmd_context * mtmd,
    uint32_t accepted_model_types,
    int32_t index,
    char * out_name,
    size_t out_capacity,
    size_t * out_length);

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

// MTP requires empty target memory in every sequence. Reset the target before
// initialization, then ingest the prompt. A populated target is rejected before
// mutation; MTP rollback-capability probes decode and clear temporary tokens.
// Both target and draft must support partial removal or recurrent snapshots.
// Other speculative strategies do not have this empty-memory precondition.
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

LLAMADART_API int32_t llama_dart_sampler_sample_and_accept_n(
    struct llama_sampler * sampler,
    struct llama_context * ctx,
    const int32_t * idxs,
    int32_t idx_count,
    const llama_token * draft_tokens,
    int32_t draft_count,
    llama_token * out_tokens,
    int32_t out_capacity);

// Live speech-to-text (llamadart-native#148). A session turns 16 kHz mono PCM
// into text while the audio arrives, over the experimental mtmd_audio_stream
// of llama.cpp. libllamadart drives the stream, the decode of its rows and the
// greedy sampling, one bounded unit of work per llama_dart_asr_stream_step.
// The caller keeps the llama and mtmd contexts; the mtmd context must have been
// created with the text model. Other calls on the llama context must stay
// idle until the session finishes, is cancelled or is reset.
//
// Every function but api_version, supported_modes, params_default, cancel and
// last_error runs behind the exit-teardown guard and the exception barrier of
// llama_dart_exit_*. A function that returns llama_dart_asr_stream_status
// returns UPSTREAM_ERROR when it catches an exception and fails the session
// like any other upstream error: it is FAILED, and a caller frees it and the
// llama context it ran on. The session is tracked in
// LLAMA_DART_EXIT_STAGE_SESSION before init returns and is freed by
// llama_dart_asr_stream_free or llama_dart_exit_free. Like a TTS task, exit
// teardown leaves it alone while its llama or mtmd context is untracked.
#define LLAMA_DART_ASR_STREAM_API_VERSION 1

// Opaque live ASR session owned by libllamadart.
struct llama_dart_asr_stream;

enum llama_dart_asr_stream_status {
    LLAMA_DART_ASR_STREAM_STATUS_OK = 0,
    LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT = -1,
    // The model, projector, mode or option has no live transcription here.
    LLAMA_DART_ASR_STREAM_STATUS_UNSUPPORTED = -2,
    LLAMA_DART_ASR_STREAM_STATUS_INVALID_STATE = -3,
    LLAMA_DART_ASR_STREAM_STATUS_UPSTREAM_ERROR = -5,
    LLAMA_DART_ASR_STREAM_STATUS_CANCELLED = -6,
    // The context cannot hold more audio. The state is FINISHED and the text
    // so far stays readable.
    LLAMA_DART_ASR_STREAM_STATUS_CONTEXT_FULL = -7,
};

// How a model transcribes, not which model it is. A caller names the modes it
// handles in accepted_modes (a mask of 1u << mode), so a later mode never
// reaches one that does not know it.
enum llama_dart_asr_stream_mode {
    LLAMA_DART_ASR_STREAM_MODE_NONE = 0,
    // One text token per 80 ms of audio, a few tokens after it (Voxtral
    // Realtime). The context grows with the audio and never shifts.
    LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY = 1,
    // The text of a 0.96 s chunk of audio at a time, under the prompt of the
    // model's "asr" chat template (LFM3-Audio). The sequence restarts at a
    // quiet chunk after 15 s and always after 30 s.
    LLAMA_DART_ASR_STREAM_MODE_PROMPTED_CHUNKS = 2,
};

enum llama_dart_asr_stream_state {
    // The queue holds no audio and the input has not been finished.
    LLAMA_DART_ASR_STREAM_STATE_NEEDS_AUDIO = 0,
    LLAMA_DART_ASR_STREAM_STATE_WORKING = 1,
    // The input was finished and its text is complete, or the model ended it,
    // or the context is full.
    LLAMA_DART_ASR_STREAM_STATE_FINISHED = 2,
    LLAMA_DART_ASR_STREAM_STATE_CANCELLED = 3,
    LLAMA_DART_ASR_STREAM_STATE_FAILED = 4,
};

struct llama_dart_asr_stream_info {
    // Set to sizeof(struct llama_dart_asr_stream_info) before calling get_info.
    uint32_t struct_size;
    uint32_t api_version;
    int32_t mode;
    // Mono float PCM at this rate: 16000.
    int32_t sample_rate;
    int32_t channels;
    // Audio per step of text: 80 for TOKEN_DELAY, 960 for PROMPTED_CHUNKS.
    int32_t text_interval_ms;
    // Audio the model needs before it writes the text of the first of it.
    int32_t delay_ms;
    // Audio one session can take, or 0 when it is not bounded by the context
    // (PROMPTED_CHUNKS restarts its sequence). For TOKEN_DELAY it is the rows
    // of the context after the prefix, 80 ms each.
    int64_t max_session_ms;
};

struct llama_dart_asr_stream_params {
    // Set by llama_dart_asr_stream_params_default. Callers should start from it.
    uint32_t struct_size;
    // Audio push holds before it takes no more (default 30000).
    int32_t max_buffered_ms;
    // The context sequence the session decodes in (default 0). A restart of
    // the transcript removes this sequence from the context's memory.
    llama_seq_id sequence_id;
};

struct llama_dart_asr_stream_progress {
    // Set to sizeof(struct llama_dart_asr_stream_progress) before each step.
    uint32_t struct_size;
    int32_t state;
    // Sequence restarts so far. PROMPTED_CHUNKS drops its sequence and starts
    // over, and a word at the cut can be lost. read_text after every step to
    // tell which restart a piece of text follows.
    int32_t segment_breaks;
    int32_t tokens_sampled;
    // Audio the queue still holds, and audio handed to the model so far.
    int64_t samples_buffered;
    int64_t samples_consumed;
    // Bytes read_text would return now.
    int64_t text_available;
};

// Returns the version of the session contract.
LLAMADART_API uint32_t llama_dart_asr_stream_api_version(void);

// Returns the mask of modes this build can drive (1u << mode for each).
LLAMADART_API uint32_t llama_dart_asr_stream_supported_modes(void);

LLAMADART_API struct llama_dart_asr_stream_params
llama_dart_asr_stream_params_default(void);

// Reads what the loaded model can do, without creating a session. UNSUPPORTED
// when the projector has no text stream (Qwen3-ASR and every model that
// llama.cpp cannot stream), when the model only prefills a chat turn (Gemma 4,
// LFM3-Omni, LFM3-Audio without its "asr" template), when its mode is not in
// accepted_modes, when n_batch cannot hold the first batch (39 rows for
// TOKEN_DELAY) or when the audio is not 16 kHz. llama_dart_asr_stream_init is
// authoritative: it returns UNSUPPORTED for anything it cannot open. The
// fields of info are set only for OK.
LLAMADART_API enum llama_dart_asr_stream_status llama_dart_asr_stream_get_info(
    const struct llama_context * llama,
    const struct mtmd_context * mtmd,
    uint32_t accepted_modes,
    struct llama_dart_asr_stream_info * out_info);

// Creates a session. The caller retains the llama and mtmd contexts. params
// may be NULL for the defaults. Returns NULL with *out_status set when it
// cannot.
LLAMADART_API struct llama_dart_asr_stream * llama_dart_asr_stream_init(
    struct llama_context * llama,
    struct mtmd_context * mtmd,
    uint32_t accepted_modes,
    const struct llama_dart_asr_stream_params * params,
    enum llama_dart_asr_stream_status * out_status);

LLAMADART_API void llama_dart_asr_stream_free(struct llama_dart_asr_stream * stream);

// Copies up to sample_count samples into the session's queue and reports in
// *out_accepted how many it took: fewer than offered when the queue reaches
// max_buffered_ms. It never drops audio; offer the rest after steps have
// drained the queue. INVALID_STATE after finish, in a final state or after a
// failure.
LLAMADART_API enum llama_dart_asr_stream_status llama_dart_asr_stream_push(
    struct llama_dart_asr_stream * stream,
    const float * samples,
    size_t sample_count,
    size_t * out_accepted);

// Ends the input: the model pads and transcribes what is queued, and the
// session reaches FINISHED once its text is out. The pause flush of
// mtmd_audio_stream_flush(end = false) is not exposed, because it changes the
// transcript. Calling it twice is INVALID_STATE.
LLAMADART_API enum llama_dart_asr_stream_status llama_dart_asr_stream_finish(
    struct llama_dart_asr_stream * stream);

// One bounded unit of work: at most one step of audio from the queue to the
// model, one upstream stream step, and when it returns rows one decode and one
// sample. It is a single call in flight for exit teardown, so teardown and a
// cancel wait for at most one step. OK with the state in progress (also when
// the model needs more audio); CANCELLED after a cancel; CONTEXT_FULL when the
// context has no room for the rows (state FINISHED). Once the state is
// FINISHED, CANCELLED or FAILED a step returns INVALID_STATE (CANCELLED for a
// cancelled session) until the session is reset.
LLAMADART_API enum llama_dart_asr_stream_status llama_dart_asr_stream_step(
    struct llama_dart_asr_stream * stream,
    struct llama_dart_asr_stream_progress * out_progress);

// Moves the UTF-8 text produced since the last read into out, complete code
// points only, without a terminating zero, and sets *out_len to its size. out
// may be NULL to query the size; otherwise capacity must be at least 4. Text
// stays readable after FINISHED and CONTEXT_FULL, and is dropped by reset.
LLAMADART_API enum llama_dart_asr_stream_status llama_dart_asr_stream_read_text(
    struct llama_dart_asr_stream * stream,
    char * out,
    size_t capacity,
    size_t * out_len);

// May be called from any thread. The next step returns CANCELLED, releases the
// stream and the sequence and produces no more text.
LLAMADART_API void llama_dart_asr_stream_cancel(struct llama_dart_asr_stream * stream);

// Attaches a caller-owned cancel byte. Any thread may set *flag to nonzero; a
// step reads it on entry and then behaves as after cancel, even if *flag
// returns to zero. The session drops the flag when it ends, is reset or is
// freed; *flag must stay readable until then.
LLAMADART_API enum llama_dart_asr_stream_status llama_dart_asr_stream_set_cancel_flag(
    struct llama_dart_asr_stream * stream,
    const int8_t * flag);

// Drops the upstream stream, the sequence, the queue and the text, so the
// session can take a new input. Works in every state.
LLAMADART_API enum llama_dart_asr_stream_status llama_dart_asr_stream_reset(
    struct llama_dart_asr_stream * stream);

// Returns a session-owned diagnostic string, valid until the next call on the
// session.
LLAMADART_API const char * llama_dart_asr_stream_last_error(
    const struct llama_dart_asr_stream * stream);

#ifdef __cplusplus
}
#endif
