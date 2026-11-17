#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_wrapper.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

int main(void) {
    assert(llama_dart_asr_stream_api_version() == LLAMA_DART_ASR_STREAM_API_VERSION);

    const uint32_t token_delay = 1u << LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY;
    const uint32_t prompted = 1u << LLAMA_DART_ASR_STREAM_MODE_PROMPTED_CHUNKS;
    assert(llama_dart_asr_stream_supported_modes() == (token_delay | prompted));
    assert((llama_dart_asr_stream_supported_modes() & (1u << LLAMA_DART_ASR_STREAM_MODE_NONE)) == 0);

    struct llama_dart_asr_stream_params params = llama_dart_asr_stream_params_default();
    assert(params.struct_size == sizeof(params));
    assert(params.max_buffered_ms == 30000);
    assert(params.sequence_id == 0);

    struct llama_dart_asr_stream_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    assert(llama_dart_asr_stream_get_info(NULL, NULL, token_delay, &info) ==
           LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT);
    assert(llama_dart_asr_stream_get_info(NULL, NULL, token_delay, NULL) ==
           LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT);
    info.struct_size = sizeof(info) - 1;
    assert(llama_dart_asr_stream_get_info(NULL, NULL, token_delay, &info) ==
           LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT);

    enum llama_dart_asr_stream_status status = LLAMA_DART_ASR_STREAM_STATUS_OK;
    assert(llama_dart_asr_stream_init(NULL, NULL, token_delay | prompted, &params, &status) == NULL);
    assert(status == LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT);
    status = LLAMA_DART_ASR_STREAM_STATUS_OK;
    assert(llama_dart_asr_stream_init(NULL, NULL, token_delay | prompted, NULL, &status) == NULL);
    assert(status == LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT);
    assert(llama_dart_asr_stream_init(NULL, NULL, token_delay | prompted, NULL, NULL) == NULL);
    params.struct_size = sizeof(params) - 1;
    status = LLAMA_DART_ASR_STREAM_STATUS_OK;
    assert(llama_dart_asr_stream_init(NULL, NULL, token_delay | prompted, &params, &status) == NULL);
    assert(status == LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT);

    const float samples[4] = {0};
    size_t accepted = 7;
    assert(llama_dart_asr_stream_push(NULL, samples, 4, &accepted) ==
           LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT);
    assert(llama_dart_asr_stream_finish(NULL) == LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT);

    struct llama_dart_asr_stream_progress progress;
    memset(&progress, 0, sizeof(progress));
    progress.struct_size = sizeof(progress);
    assert(llama_dart_asr_stream_step(NULL, &progress) == LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT);
    progress.struct_size = sizeof(progress) - 1;
    assert(llama_dart_asr_stream_step(NULL, &progress) == LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT);

    char text[8];
    size_t length = 7;
    assert(llama_dart_asr_stream_read_text(NULL, text, sizeof(text), &length) ==
           LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT);
    assert(llama_dart_asr_stream_read_text(NULL, NULL, 0, &length) ==
           LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT);
    assert(llama_dart_asr_stream_reset(NULL) == LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT);

    const int8_t flag = 1;
    assert(llama_dart_asr_stream_set_cancel_flag(NULL, &flag) ==
           LLAMA_DART_ASR_STREAM_STATUS_INVALID_ARGUMENT);

    llama_dart_asr_stream_cancel(NULL);
    llama_dart_asr_stream_free(NULL);
    assert(strcmp(llama_dart_asr_stream_last_error(NULL), "invalid ASR stream handle") == 0);

    // A failure that is not a caught exception leaves no last error.
    assert(llama_dart_last_error() == NULL);
    return 0;
}
