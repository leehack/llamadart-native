#pragma once

#if defined(LLAMADART_TESTING)

#include "llama_dart_wrapper.h"

#include <cstdint>

#if defined(_WIN32) && !defined(LLAMADART_API)
#define LLAMADART_TTS_TEST_HOOK_API __declspec(dllexport)
#else
#define LLAMADART_TTS_TEST_HOOK_API
#endif

#if !defined(LLAMADART_API)
#define LLAMADART_API LLAMADART_TTS_TEST_HOOK_API
#endif

extern "C" {

// Exercises the real llama_dart_tts_finish_streamed_output with an injected
// flush status and optional cancellation during flush, then probes both read
// exports.
LLAMADART_API void llama_dart_test_tts_finish_streamed_output(
    int32_t flush_status, bool cancel_during_flush, int32_t *out_status,
    int32_t *out_state, int32_t *out_info_status, int32_t *out_read_status);

// Creates a tracked streamed TTS handle in GENERATING state with dummy PCM
// (no generator/mmproj needed) for read-guard teardown tests.
LLAMADART_API struct llama_dart_tts *
llama_dart_test_tts_create_streamed_dummy(struct llama_context *llama);

// Arms or releases the read-guard latch:
// kind: 0 = disabled, 1 = get_output_info, 2 = read_pcm.
LLAMADART_API void llama_dart_test_tts_arm_read_latch(int32_t kind);
LLAMADART_API bool llama_dart_test_tts_read_latch_entered(void);
LLAMADART_API void llama_dart_test_tts_release_read_latch(void);
}

#endif
